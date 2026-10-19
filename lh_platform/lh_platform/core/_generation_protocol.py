"""Bounded JSON protocol shared by generation authority processes."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.connection import Connection
from typing import Any

from lh_platform.core.types import (
    Action,
    Observation,
    Role,
    Step,
    Task,
    TaskDomain,
    Trajectory,
)
from lh_platform.models.base import Message, ModelOutput

GENERATION_AUTHORITY_SCHEMA = "lh_generation_authority/v1"
_MAX_WIRE_BYTES = 64 * 1024 * 1024


class GenerationAuthorityError(RuntimeError):
    """The isolated model authority could not complete a protocol operation."""


_STRICT_WORKER_ENV_NAMES = frozenset(
    {
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_BASE",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    }
)
_STRICT_WORKER_ENV_MARKERS = (
    "API_KEY",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "CLIENT_SECRET",
    "PASSWORD",
    "BEARER_TOKEN",
)
_STRICT_METHOD_RESERVED_KEYS = frozenset({"model_id", "model_registry", "model_kwargs"})


def _scrub_strict_worker_environment() -> None:
    """Remove inherited model endpoints and credentials from an adapter worker.

    The Python worker is not a general OS/network sandbox: reviewed adapters
    are still responsible for obeying ``strict_proxy_compatible``.  This
    scrub prevents the normal accidental bypass where an upstream adapter
    reads the controller's OpenAI/vLLM configuration from its constructor or
    inherited environment.
    """

    for name in tuple(os.environ):
        normalized = name.upper()
        if normalized in _STRICT_WORKER_ENV_NAMES or any(
            marker in normalized for marker in _STRICT_WORKER_ENV_MARKERS
        ):
            os.environ.pop(name, None)


def _strict_method_constructor_kwargs(
    vetted_constructor_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine fixed proxy inputs with an inventory-vetted constructor plan."""

    extra = dict(vetted_constructor_kwargs or {})
    reserved = sorted(_STRICT_METHOD_RESERVED_KEYS.intersection(extra))
    if reserved:
        raise ValueError(f"strict method constructor kwargs contain reserved keys: {reserved!r}")
    return {
        "model_id": "generation_authority_proxy",
        "model_registry": None,
        "model_kwargs": {},
        **copy.deepcopy(extra),
    }


def _wire_send(connection: Connection, payload: Mapping[str, Any]) -> None:
    """Send one JSON object without unpickling method-controlled objects."""

    try:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GenerationAuthorityError(
            f"authority wire payload is not JSON-serializable: {type(error).__name__}"
        ) from error
    if len(encoded) > _MAX_WIRE_BYTES:
        raise GenerationAuthorityError("authority wire payload exceeds size limit")
    connection.send_bytes(encoded)


def _wire_recv(connection: Connection) -> dict[str, Any]:
    """Receive exactly one bounded JSON object from an isolated process."""

    try:
        raw = connection.recv_bytes(_MAX_WIRE_BYTES)
    except OSError as error:
        raise GenerationAuthorityError("authority wire payload exceeds size limit") from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationAuthorityError(
            f"authority wire payload is not valid JSON: {type(error).__name__}"
        ) from error
    if not isinstance(value, Mapping):
        raise GenerationAuthorityError("authority wire payload must be a JSON object")
    return dict(value)


def _safe_mapping(value: object) -> dict[str, Any]:
    """Return a defensive mapping with only pickle-safe primitive structure."""

    if not isinstance(value, Mapping):
        return {}
    return copy.deepcopy(dict(value))


def _serialize_output(output: ModelOutput) -> dict[str, Any]:
    """Serialize only the portable portion of a model response.

    ``raw`` frequently contains an SDK object and is intentionally omitted:
    model adapters must use normalized ``text``/token/logprob fields instead.
    """

    return {
        "text": str(output.text or ""),
        "prompt_tokens": int(output.prompt_tokens or 0),
        "completion_tokens": int(output.completion_tokens or 0),
        "cost_usd": float(output.cost_usd or 0.0),
        "logprobs": copy.deepcopy(output.logprobs),
        "info": _safe_mapping(output.info),
        "generation_id": output.generation_id,
        "generation_event_sha256": output.generation_event_sha256,
    }


def _deserialize_output(payload: Mapping[str, Any]) -> ModelOutput:
    """Rebuild the public ``ModelOutput`` returned to a method adapter."""

    return ModelOutput(
        text=str(payload.get("text") or ""),
        prompt_tokens=int(payload.get("prompt_tokens") or 0),
        completion_tokens=int(payload.get("completion_tokens") or 0),
        cost_usd=float(payload.get("cost_usd") or 0.0),
        raw=None,
        logprobs=copy.deepcopy(payload.get("logprobs")),
        info=_safe_mapping(payload.get("info")),
        generation_id=(
            str(payload["generation_id"]) if isinstance(payload.get("generation_id"), str) else None
        ),
        generation_event_sha256=(
            str(payload["generation_event_sha256"])
            if isinstance(payload.get("generation_event_sha256"), str)
            else None
        ),
    )


def _require_text(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise GenerationAuthorityError(f"{context} must be text")
    return value


def _require_mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenerationAuthorityError(f"{context} must be an object")
    return dict(value)


def _require_number(value: object, *, context: str, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenerationAuthorityError(f"{context} must be numeric")
    return float(value)


def _action_from_wire(payload: object) -> Action:
    """Validate a method worker's JSON action before Runner sees it."""

    data = _require_mapping(payload, context="method action")
    tool_name = data.get("tool_name")
    if tool_name is not None and not isinstance(tool_name, str):
        raise GenerationAuthorityError("method action.tool_name must be text or null")
    tool_args = data.get("tool_args")
    if tool_args is not None and not isinstance(tool_args, Mapping):
        raise GenerationAuthorityError("method action.tool_args must be an object or null")
    return Action(
        action_id=_require_text(data.get("action_id"), context="method action.action_id"),
        raw=_require_text(data.get("raw"), context="method action.raw"),
        action_type=_require_text(data.get("action_type"), context="method action.action_type"),
        tool_name=tool_name,
        tool_args=copy.deepcopy(dict(tool_args)) if isinstance(tool_args, Mapping) else None,
        info=_safe_mapping(data.get("info")),
        t=_require_number(data.get("t"), context="method action.t"),
    )


def _observation_from_wire(payload: object) -> Observation:
    data = _require_mapping(payload, context="observation")
    is_error = data.get("is_error", False)
    if not isinstance(is_error, bool):
        raise GenerationAuthorityError("observation.is_error must be boolean")
    return Observation(
        obs_id=_require_text(data.get("obs_id"), context="observation.obs_id"),
        content=_require_text(data.get("content"), context="observation.content"),
        is_error=is_error,
        info=_safe_mapping(data.get("info")),
        t=_require_number(data.get("t"), context="observation.t"),
    )


def _task_from_wire(payload: object) -> Task:
    data = _require_mapping(payload, context="task")
    expected_outputs = data.get("expected_outputs", [])
    if not isinstance(expected_outputs, list) or not all(
        isinstance(item, str) for item in expected_outputs
    ):
        raise GenerationAuthorityError("task.expected_outputs must be a text list")
    max_steps = data.get("max_steps", 30)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise GenerationAuthorityError("task.max_steps must be a positive integer")
    try:
        domain = TaskDomain(_require_text(data.get("domain"), context="task.domain"))
    except ValueError as error:
        raise GenerationAuthorityError("task.domain is unknown") from error
    return Task(
        task_id=_require_text(data.get("task_id"), context="task.task_id"),
        domain=domain,
        instruction=_require_text(data.get("instruction"), context="task.instruction"),
        metadata=_safe_mapping(data.get("metadata")),
        max_steps=max_steps,
        timeout_s=_require_number(data.get("timeout_s"), context="task.timeout_s", default=600.0),
        expected_outputs=list(expected_outputs),
    )


def _step_from_wire(payload: object) -> Step:
    data = _require_mapping(payload, context="trajectory step")
    step_idx = data.get("step_idx")
    if isinstance(step_idx, bool) or not isinstance(step_idx, int) or step_idx < 0:
        raise GenerationAuthorityError("trajectory step.step_idx must be nonnegative integer")
    try:
        role = Role(_require_text(data.get("role"), context="trajectory step.role"))
    except ValueError as error:
        raise GenerationAuthorityError("trajectory step.role is unknown") from error
    action = data.get("action")
    observation = data.get("observation")
    return Step(
        step_idx=step_idx,
        role=role,
        content=_require_text(data.get("content"), context="trajectory step.content"),
        action=_action_from_wire(action) if action is not None else None,
        observation=_observation_from_wire(observation) if observation is not None else None,
        prompt_tokens=int(
            _require_number(data.get("prompt_tokens"), context="trajectory step.prompt_tokens")
        ),
        completion_tokens=int(
            _require_number(
                data.get("completion_tokens"),
                context="trajectory step.completion_tokens",
            )
        ),
        cost_usd=_require_number(data.get("cost_usd"), context="trajectory step.cost_usd"),
        latency_s=_require_number(data.get("latency_s"), context="trajectory step.latency_s"),
        info=_safe_mapping(data.get("info")),
        t=_require_number(data.get("t"), context="trajectory step.t"),
    )


def _trajectory_from_wire(payload: object) -> Trajectory:
    data = _require_mapping(payload, context="trajectory")
    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        raise GenerationAuthorityError("trajectory.steps must be a list")
    trajectory = Trajectory(
        task_id=_require_text(data.get("task_id"), context="trajectory.task_id"),
        method=_require_text(data.get("method"), context="trajectory.method"),
        model=_require_text(data.get("model"), context="trajectory.model"),
        benchmark=_require_text(data.get("benchmark"), context="trajectory.benchmark"),
        started_at=_require_number(data.get("started_at"), context="trajectory.started_at"),
        finished_at=(
            _require_number(data.get("finished_at"), context="trajectory.finished_at")
            if data.get("finished_at") is not None
            else None
        ),
        success=bool(data.get("success", False)),
        final_answer=(
            _require_text(data.get("final_answer"), context="trajectory.final_answer")
            if data.get("final_answer") is not None
            else None
        ),
        info=_safe_mapping(data.get("info")),
    )
    for raw_step in raw_steps:
        trajectory.append(_step_from_wire(raw_step))
    return trajectory


def _messages_from_wire(value: object) -> list[Message]:
    if not isinstance(value, list):
        raise GenerationAuthorityError("generation request messages must be a list")
    messages: list[Message] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise GenerationAuthorityError("generation request message is malformed")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise GenerationAuthorityError("generation request message is malformed")
        messages.append(Message(role=role, content=content))
    return messages


def _runtime_snapshot(model: object) -> dict[str, Any]:
    producer = getattr(model, "runtime_evidence", None)
    if not callable(producer):
        return {"model_instantiated": True}
    try:
        evidence = producer()
    except Exception as error:  # pragma: no cover - adapter-specific failure
        return {
            "model_instantiated": True,
            "runtime_evidence_error": type(error).__name__,
        }
    return _safe_mapping(evidence)


def _generation_events(model: object, cursor: int) -> tuple[dict[str, Any], ...]:
    producer = getattr(model, "generation_events_since", None)
    if not callable(producer):
        return ()
    try:
        events = producer(cursor)
    except Exception as error:  # pragma: no cover - adapter-specific failure
        raise GenerationAuthorityError(
            f"generation event authority failed: {type(error).__name__}"
        ) from error
    if not isinstance(events, (list, tuple)):
        raise GenerationAuthorityError("generation authority returned malformed events")
    return tuple(copy.deepcopy(dict(event)) for event in events if isinstance(event, Mapping))


def _generation_cursor(model: object) -> int:
    producer = getattr(model, "generation_cursor", None)
    if not callable(producer):
        raise GenerationAuthorityError("model adapter has no generation cursor")
    cursor = producer()
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise GenerationAuthorityError("model adapter returned malformed generation cursor")
    return cursor


def _send_error(connection: Connection, error: BaseException) -> None:
    _wire_send(
        connection,
        {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


class AuthorityModelProxy:
    """The only model object exposed to a method in an authoritative run."""

    name = "generation_authority_proxy"

    def __init__(
        self,
        channel: Connection,
        *,
        model_name: str,
        timeout_s: float = 300.0,
        on_transport_failure: Callable[[], None] | None = None,
    ) -> None:
        self._channel = channel
        # Compatibility metadata only.  It is not used for provenance; the
        # authority records the actual requested model in each event.
        self.model_name = model_name
        self._timeout_s = float(timeout_s)
        self._on_transport_failure = on_transport_failure
        self._last_runtime: dict[str, Any] = {"model_instantiated": False}

    def _mark_transport_failure(self) -> None:
        if self._on_transport_failure is not None:
            self._on_transport_failure()

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        _wire_send(self._channel, dict(payload))
        if not self._channel.poll(self._timeout_s):
            self._mark_transport_failure()
            raise GenerationAuthorityError("generation authority timed out")
        try:
            response = _wire_recv(self._channel)
        except (EOFError, OSError, GenerationAuthorityError):
            self._mark_transport_failure()
            raise
        if not isinstance(response, Mapping):
            raise GenerationAuthorityError("generation authority returned malformed response")
        if response.get("ok") is not True:
            error_type = str(response.get("error_type") or "GenerationAuthorityError")
            detail = str(response.get("error") or "unknown authority error")
            raise GenerationAuthorityError(f"{error_type}: {detail}")
        return copy.deepcopy(dict(response))

    def generate(self, messages: Sequence[Message], **kwargs: object) -> ModelOutput:
        response = self._request(
            {
                "op": "generate",
                "messages": [
                    {"role": str(message.role), "content": str(message.content)}
                    for message in messages
                ],
                "kwargs": dict(kwargs),
            }
        )
        output = response.get("output")
        if not isinstance(output, Mapping):
            raise GenerationAuthorityError("generation authority omitted model output")
        runtime = response.get("runtime")
        if isinstance(runtime, Mapping):
            self._last_runtime = _safe_mapping(runtime)
        return _deserialize_output(output)

    def chat(self, user: str, system: str | None = None, **kwargs: object) -> ModelOutput:
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=user))
        return self.generate(messages, **kwargs)

    def runtime_evidence(self) -> dict[str, Any]:
        """Compatibility snapshot only; Runner never trusts this channel."""

        return copy.deepcopy(self._last_runtime)
