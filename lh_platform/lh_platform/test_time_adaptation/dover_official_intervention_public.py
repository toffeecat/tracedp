"""Call Microsoft DoVer intervention code under a public-input Terminal contract."""

# Portions reproduce the DoVer intervention system prompt from microsoft/ACV (MIT), https://github.com/microsoft/ACV. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from lh_platform.benchmarks.terminal_forward_recovery_contract import (
    TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT,
)
from lh_platform.test_time_adaptation.terminal_log_attribution import (
    build_terminal_log_trace_case,
)

DOVER_OFFICIAL_INTERVENTION_ADAPTED_PUBLIC_METHOD_ID = (
    "dover_official_intervention_adapted_public_v1"
)
DOVER_OFFICIAL_INTERVENTION_SYSTEM_PROMPT = (
    "You write precise, minimal, and structured interventions for Magentic-One "
    "multi-agent runs."
)
_UPSTREAM_SOURCE = "microsoft/ACV:misc/DoVer/scripts/intervention_recommender_trial.py"
_UPSTREAM_CALLS = (
    "_trial_context_window",
    "_build_intervention_prompt",
    "parse_llm_json_relaxed",
)
_REFERENCE_ANSWER_LABEL = "Reference ground truth (for your guidance only): "
_ALLOWED_CATEGORIES = frozenset(
    {"orchestrator_ledger", "orchestrator_instruction", "subagent_instruction"}
)


class DoVerInterventionCompatibilityError(ValueError):
    """Report an upstream, provider, or response contract mismatch."""


class DoVerInterventionResumeAmbiguousError(RuntimeError):
    """Report an interrupted provider call that cannot be repeated safely."""


class DoVerInterventionActorResponse(Protocol):
    """Expose the same provider evidence required by other model-backed baselines."""

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    reasoning_tokens_reported: bool
    stop_reason: str
    latency_seconds: float
    provider_identity_evidence: str
    served_provider_reported: bool


class DoVerInterventionActor(Protocol):
    """Generate one intervention from an upstream-built public prompt."""

    transport: str
    session_id: str | None
    client_id: str
    user_agent: str

    def generate(self, prompt: str) -> DoVerInterventionActorResponse: ...

    def set_raw_response_recorder(
        self, recorder: Callable[[dict[str, Any], float], None] | None
    ) -> None: ...

    def response_from_raw(
        self, response: Mapping[str, Any], *, latency_seconds: float
    ) -> DoVerInterventionActorResponse: ...


@dataclass(frozen=True)
class DoVerPublicTrialV1:
    """Represent one public failed trial without an evaluator-owned answer."""

    task_instruction: str
    steps: tuple[dict[str, Any], ...]
    failed_step_index: int
    failed_agent: str
    diagnosed_reason: str
    diagnosis_source: str
    trial_index: int
    trial_plan: tuple[Any, ...] = ()
    fully_fulfilled_plan_steps: tuple[Any, ...] = ()
    partially_fulfilled_plan_steps: tuple[Any, ...] = ()
    unfulfilled_plan_steps: tuple[Any, ...] = ()
    trial_context: str = ""
    trial_reflection: str = ""
    prior_trials_summary: str = ""
    trial_execution: str = ""
    trial_overall_summary: str = ""

    def __post_init__(self) -> None:
        """Reject an incomplete public failure description before an upstream call."""

        if not self.task_instruction.strip() or not self.steps:
            raise ValueError("DoVer public trial requires a task and executed steps")
        if not 0 <= self.failed_step_index < len(self.steps):
            raise ValueError("DoVer failed step is outside the public trial")
        if (
            self.trial_index < 0
            or not self.failed_agent.strip()
            or not self.diagnosis_source.strip()
        ):
            raise ValueError("DoVer public trial identity is incomplete")

    def upstream_arguments(self) -> dict[str, Any]:
        """Translate public fields to the official intervention prompt signature."""

        return {
            "problem": self.task_instruction.strip(),
            "ground_truth": "",
            "step_idx": self.failed_step_index,
            "agent": self.failed_agent.strip(),
            "reason": self.diagnosed_reason.strip() or None,
            "trial_index": self.trial_index,
            "trial_plan": list(self.trial_plan),
            "trial_progress": {
                "fully_fulfilled_plan_steps": list(self.fully_fulfilled_plan_steps),
                "partially_fulfilled_plan_steps": list(
                    self.partially_fulfilled_plan_steps
                ),
                "unfulfilled_plan_steps": list(self.unfulfilled_plan_steps),
            },
            "trial_context": self.trial_context,
            "trial_reflection": self.trial_reflection,
            "prior_trials_summary": self.prior_trials_summary,
            "trial_execution": self.trial_execution,
            "trial_overall_summary": self.trial_overall_summary,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize only method-visible trial evidence."""

        return {
            "task_instruction": self.task_instruction,
            "steps": [dict(step) for step in self.steps],
            "failed_step_index": self.failed_step_index,
            "failed_agent": self.failed_agent,
            "diagnosed_reason": self.diagnosed_reason,
            "diagnosis_source": self.diagnosis_source,
            "trial_index": self.trial_index,
            "trial_plan": list(self.trial_plan),
            "fully_fulfilled_plan_steps": list(self.fully_fulfilled_plan_steps),
            "partially_fulfilled_plan_steps": list(
                self.partially_fulfilled_plan_steps
            ),
            "unfulfilled_plan_steps": list(self.unfulfilled_plan_steps),
            "trial_context": self.trial_context,
            "trial_reflection": self.trial_reflection,
            "prior_trials_summary": self.prior_trials_summary,
            "trial_execution": self.trial_execution,
            "trial_overall_summary": self.trial_overall_summary,
        }


def build_dover_public_trial_from_terminal_logs(
    *,
    case_id: str,
    task_id: str,
    task_instruction: str,
    failure_summary: str,
    terminal_reason: str,
    official_reward: float | None,
    session_records: Sequence[Mapping[str, Any]],
    observer_records: Sequence[Mapping[str, Any]],
    failed_action_id: str,
    diagnosed_reason: str,
    diagnosis_source: str,
) -> DoVerPublicTrialV1:
    """Project the shared canonical Terminal trace into DoVer's public step contract."""

    compilation = build_terminal_log_trace_case(
        case_id=case_id,
        task_id=task_id,
        task_specification=task_instruction,
        failure_summary=failure_summary,
        terminal_reason=terminal_reason,
        official_reward=official_reward,
        session_records=session_records,
        observer_records=observer_records,
    )
    failed_event_id = next(
        (
            event.event_id
            for event in compilation.trace_case.events
            if dict(event.metadata).get("tool_call_id") == failed_action_id
        ),
        None,
    )
    if failed_event_id is None:
        raise ValueError("selected failed action is absent from the canonical public trace")
    return build_dover_public_trial_from_case(
        compilation.trace_case,
        task_instruction=task_instruction,
        failure_summary=failure_summary,
        failed_event_id=failed_event_id,
        diagnosed_reason=diagnosed_reason,
        diagnosis_source=diagnosis_source,
    )


def build_dover_public_trial_from_case(
    case: Any,
    *,
    task_instruction: str,
    failure_summary: str,
    failed_event_id: str,
    diagnosed_reason: str,
    diagnosis_source: str,
) -> DoVerPublicTrialV1:
    """Project a public trace case into DoVer's step contract at one hypothesized event.

    Every event becomes one DoVer step of the single ``TerminalActor``; the failed step is
    the event a locator blamed (``step-k`` or a final ``turn-*``), by its event id.
    """

    steps: list[dict[str, Any]] = []
    failed_step_index: int | None = None
    for event in case.events:
        steps.append(
            {
                "name": "TerminalActor",
                "content": event.content,
                "timestamp": event.sequence,
            }
        )
        if event.event_id == failed_event_id:
            failed_step_index = len(steps) - 1
    if failed_step_index is None:
        raise ValueError("hypothesized event is absent from the public trace")
    execution = "\n\n".join(
        f"[Step {index}] {step['name']}: {step['content']}"
        for index, step in enumerate(steps)
    )
    return DoVerPublicTrialV1(
        task_instruction=task_instruction,
        steps=tuple(steps),
        failed_step_index=failed_step_index,
        failed_agent="TerminalActor",
        diagnosed_reason=diagnosed_reason,
        diagnosis_source=diagnosis_source,
        trial_index=0,
        trial_context=(
            "One continuous Terminal runtime attempt projected from the shared "
            "canonical public trace."
        ),
        trial_execution=execution,
        trial_overall_summary=failure_summary,
    )


@dataclass(frozen=True)
class DoVerPublicInterventionReceiptV1:
    """Persist the exact upstream prompt, model response, and Terminal adaptation."""

    request: dict[str, Any]
    upstream_script: str
    prompt: str
    raw_response: str
    category: str
    replacement_text: str
    provider: str
    model: str
    transport: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    reasoning_tokens_reported: bool
    stop_reason: str
    latency_seconds: float
    provider_identity_evidence: str
    served_provider_reported: bool
    transport_session_id: str | None
    transport_client_id: str
    transport_user_agent: str

    def to_dict(self) -> dict[str, Any]:
        """Publish method provenance and the exact executable continuation fields."""

        return {
            "schema_version": "dover_native_public_intervention_receipt_v1",
            "method_id": DOVER_OFFICIAL_INTERVENTION_ADAPTED_PUBLIC_METHOD_ID,
            "implementation_status": "implementation_unverified",
            "upstream_source": _UPSTREAM_SOURCE,
            "upstream_script": self.upstream_script,
            "upstream_functions_called": list(_UPSTREAM_CALLS),
            "semantic_adaptations": [
                "reference_answer_omitted",
                "empty_reference_answer_label_removed_after_upstream_prompt_builder",
                "terminal_public_tool_steps_projected_as_dover_steps",
                "generated_replacement_delivered_as_terminal_continuation_guidance",
            ],
            "request": self.request,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "parsed_intervention": {
                "category": self.category,
                "replacement_text": self.replacement_text,
            },
            "provider_receipt": {
                "provider": self.provider,
                "model": self.model,
                "transport": self.transport,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "reasoning_tokens_reported": self.reasoning_tokens_reported,
                "provider_internal_reasoning_reported": self.reasoning_tokens_reported,
                "stop_reason": self.stop_reason,
                "latency_seconds": self.latency_seconds,
                "provider_identity_evidence": self.provider_identity_evidence,
                "served_provider_reported": self.served_provider_reported,
                "transport_session_id": self.transport_session_id,
                "transport_client_id": self.transport_client_id,
                "transport_user_agent": self.transport_user_agent,
            },
            "terminal_execution": {
                "continuation_prompt_mode": TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT,
                "targeted_guidance": self.replacement_text,
                "repair_state": None,
            },
        }


def _load_upstream_module(script: Path) -> ModuleType:
    """Load the configured official script without copying its prompt implementation."""

    if not script.is_file():
        raise DoVerInterventionCompatibilityError("DoVer upstream script is unavailable")
    spec = importlib.util.spec_from_file_location("longh_dover_official_intervention", script)
    if spec is None or spec.loader is None:
        raise DoVerInterventionCompatibilityError("DoVer upstream script cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in _UPSTREAM_CALLS:
        if not callable(getattr(module, name, None)):
            raise DoVerInterventionCompatibilityError(
                f"DoVer upstream function is unavailable: {name}"
            )
    return module


class DoVerOfficialInterventionPublicAdapter:
    """Execute the official intervention prompt and parser with the answer removed."""

    def __init__(self, upstream_script: Path) -> None:
        self.upstream_script = upstream_script.resolve()
        self._upstream = _load_upstream_module(self.upstream_script)

    def build_prompt(self, trial: DoVerPublicTrialV1) -> str:
        """Call the official context-window and prompt builders on public evidence."""

        _, public_prompt = self.build_prompt_pair(trial)
        return public_prompt

    def build_prompt_pair(self, trial: DoVerPublicTrialV1) -> tuple[str, str]:
        """Return the exact upstream prompt beside its public-safe transformation."""

        arguments = trial.upstream_arguments()
        context = self._upstream._trial_context_window(  # noqa: SLF001
            list(trial.steps), trial.failed_step_index, window=2
        )
        prompt = str(
            self._upstream._build_intervention_prompt(  # noqa: SLF001
                **arguments,
                context_window=context,
            )
        )
        if _REFERENCE_ANSWER_LABEL not in prompt:
            raise DoVerInterventionCompatibilityError(
                "DoVer upstream reference-answer prompt contract changed"
            )
        # Public benchmarks cannot expose or imply an evaluator-owned answer field.
        return prompt, prompt.replace(_REFERENCE_ANSWER_LABEL, "", 1)

    def parse_response(self, raw_response: str) -> tuple[str, str]:
        """Call the official relaxed parser, then enforce its documented output schema."""

        parsed = self.parse_json_object_relaxed(raw_response)
        category = parsed.get("category")
        replacement_text = parsed.get("replacement_text")
        if category not in _ALLOWED_CATEGORIES:
            raise DoVerInterventionCompatibilityError("DoVer intervention category is invalid")
        if not isinstance(replacement_text, str) or not replacement_text.strip():
            raise DoVerInterventionCompatibilityError("DoVer replacement text is empty")
        if replacement_text != replacement_text.strip():
            raise DoVerInterventionCompatibilityError("DoVer replacement text is not normalized")
        return str(category), replacement_text

    def parse_json_object_relaxed(self, raw_response: str) -> dict[str, Any]:
        """Expose the author's fenced-JSON parser without changing its behavior."""

        parsed = self._upstream.parse_llm_json_relaxed(raw_response)
        if not isinstance(parsed, dict):
            raise DoVerInterventionCompatibilityError("DoVer response is not a JSON object")
        return dict(parsed)


class DurableDoVerPublicInterventionGenerator:
    """Persist one model-backed intervention and fail closed after an ambiguous call."""

    def __init__(
        self,
        *,
        adapter: DoVerOfficialInterventionPublicAdapter,
        actor: DoVerInterventionActor,
        provider: str,
        model: str,
        output: Path,
    ) -> None:
        self._adapter = adapter
        self._actor = actor
        self._provider = provider
        self._model = model
        self._output = output
        self._actor.set_raw_response_recorder(self._record_raw_response)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        """Publish one complete receipt without exposing partial JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _record_raw_response(
        self, payload: dict[str, Any], latency_seconds: float
    ) -> None:
        """Save the provider payload before response parsing or process interruption."""

        self._write_json(
            self._output / "provider_raw_response.json",
            {"response": payload, "latency_seconds": latency_seconds},
        )

    def _response_from_durable_raw(self) -> DoVerInterventionActorResponse:
        """Rebuild the validated actor response without issuing another provider call."""

        raw_path = self._output / "provider_raw_response.json"
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("response"), dict)
            or not isinstance(payload.get("latency_seconds"), (int, float))
        ):
            raise DoVerInterventionCompatibilityError(
                "DoVer durable raw provider response is invalid"
            )
        return self._actor.response_from_raw(
            payload["response"],
            latency_seconds=float(payload["latency_seconds"]),
        )

    def generate(self, trial: DoVerPublicTrialV1) -> DoVerPublicInterventionReceiptV1:
        """Call the actor once, preserving request identity across resume."""

        request = trial.to_dict()
        prompt = self._adapter.build_prompt(trial)
        inflight = self._output / "intervention.inflight.json"
        receipt_path = self._output / "intervention.json"
        raw_response_path = self._output / "provider_raw_response.json"
        if receipt_path.is_file():
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            return self._load_receipt(payload, request=request, prompt=prompt)
        if inflight.exists():
            inflight_payload = json.loads(inflight.read_text(encoding="utf-8"))
            expected_inflight = {
                "schema_version": "dover_native_public_intervention_inflight_v1",
                "method_id": DOVER_OFFICIAL_INTERVENTION_ADAPTED_PUBLIC_METHOD_ID,
                "request": request,
                "prompt": prompt,
                "upstream_script": str(self._adapter.upstream_script),
            }
            if inflight_payload != expected_inflight:
                raise DoVerInterventionCompatibilityError(
                    "DoVer inflight request identity drifted"
                )
            if not raw_response_path.is_file():
                raise DoVerInterventionResumeAmbiguousError(
                    "DoVer intervention call may have completed without a durable response"
                )
            response = self._response_from_durable_raw()
        else:
            self._write_json(
                inflight,
                {
                    "schema_version": "dover_native_public_intervention_inflight_v1",
                    "method_id": DOVER_OFFICIAL_INTERVENTION_ADAPTED_PUBLIC_METHOD_ID,
                    "request": request,
                    "prompt": prompt,
                    "upstream_script": str(self._adapter.upstream_script),
                },
            )
            response = self._actor.generate(prompt)
        if response.provider != self._provider or response.model != self._model:
            raise DoVerInterventionCompatibilityError(
                "DoVer intervention provider or model identity drifted"
            )
        category, replacement_text = self._adapter.parse_response(response.text)
        receipt = DoVerPublicInterventionReceiptV1(
            request=request,
            upstream_script=str(self._adapter.upstream_script),
            prompt=prompt,
            raw_response=response.text,
            category=category,
            replacement_text=replacement_text,
            provider=response.provider,
            model=response.model,
            transport=self._actor.transport,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning_tokens=response.reasoning_tokens,
            reasoning_tokens_reported=response.reasoning_tokens_reported,
            stop_reason=response.stop_reason,
            latency_seconds=response.latency_seconds,
            provider_identity_evidence=response.provider_identity_evidence,
            served_provider_reported=response.served_provider_reported,
            transport_session_id=self._actor.session_id,
            transport_client_id=self._actor.client_id,
            transport_user_agent=self._actor.user_agent,
        )
        self._write_json(receipt_path, receipt.to_dict())
        inflight.unlink()
        return receipt

    def _load_receipt(
        self, payload: object, *, request: dict[str, Any], prompt: str
    ) -> DoVerPublicInterventionReceiptV1:
        """Reconstruct a completed response without repeating a model call."""

        if not isinstance(payload, dict):
            raise DoVerInterventionCompatibilityError("DoVer receipt is not an object")
        if (
            payload.get("schema_version") != "dover_native_public_intervention_receipt_v1"
            or payload.get("method_id")
            != DOVER_OFFICIAL_INTERVENTION_ADAPTED_PUBLIC_METHOD_ID
            or payload.get("upstream_source") != _UPSTREAM_SOURCE
            or payload.get("upstream_functions_called") != list(_UPSTREAM_CALLS)
            or payload.get("request") != request
            or payload.get("prompt") != prompt
            or payload.get("upstream_script") != str(self._adapter.upstream_script)
        ):
            raise DoVerInterventionCompatibilityError("DoVer durable request identity drifted")
        parsed = payload.get("parsed_intervention")
        provider = payload.get("provider_receipt")
        if not isinstance(parsed, dict) or not isinstance(provider, dict):
            raise DoVerInterventionCompatibilityError("DoVer durable response is incomplete")
        expected_provider_contract = {
            "provider": self._provider,
            "model": self._model,
            "transport": self._actor.transport,
            "transport_session_id": self._actor.session_id,
            "transport_client_id": self._actor.client_id,
            "transport_user_agent": self._actor.user_agent,
        }
        if any(provider.get(key) != value for key, value in expected_provider_contract.items()):
            raise DoVerInterventionCompatibilityError(
                "DoVer durable provider contract drifted"
            )
        reasoning_reported = provider.get("reasoning_tokens_reported")
        reasoning_tokens = provider.get("reasoning_tokens")
        if (
            not isinstance(reasoning_reported, bool)
            or type(reasoning_tokens) is not int
            or reasoning_tokens < 0
            or provider.get("provider_internal_reasoning_reported")
            is not reasoning_reported
            or (not reasoning_reported and reasoning_tokens != 0)
        ):
            raise DoVerInterventionCompatibilityError(
                "DoVer durable reasoning-usage disclosure is invalid"
            )
        category, replacement_text = self._adapter.parse_response(
            str(payload.get("raw_response", ""))
        )
        if parsed != {"category": category, "replacement_text": replacement_text}:
            raise DoVerInterventionCompatibilityError("DoVer parsed response drifted")
        return DoVerPublicInterventionReceiptV1(
            request=request,
            upstream_script=str(self._adapter.upstream_script),
            prompt=prompt,
            raw_response=str(payload["raw_response"]),
            category=category,
            replacement_text=replacement_text,
            provider=str(provider.get("provider", "")),
            model=str(provider.get("model", "")),
            transport=str(provider.get("transport", "")),
            input_tokens=int(provider.get("input_tokens", 0)),
            output_tokens=int(provider.get("output_tokens", 0)),
            reasoning_tokens=int(provider.get("reasoning_tokens", 0)),
            reasoning_tokens_reported=bool(provider.get("reasoning_tokens_reported")),
            stop_reason=str(provider.get("stop_reason", "")),
            latency_seconds=float(provider.get("latency_seconds", 0.0)),
            provider_identity_evidence=str(provider.get("provider_identity_evidence", "")),
            served_provider_reported=bool(provider.get("served_provider_reported")),
            transport_session_id=(
                str(provider["transport_session_id"])
                if provider.get("transport_session_id") is not None
                else None
            ),
            transport_client_id=str(provider.get("transport_client_id", "")),
            transport_user_agent=str(provider.get("transport_user_agent", "")),
        )


__all__ = [
    "DOVER_OFFICIAL_INTERVENTION_ADAPTED_PUBLIC_METHOD_ID",
    "DOVER_OFFICIAL_INTERVENTION_SYSTEM_PROMPT",
    "DoVerInterventionCompatibilityError",
    "DoVerInterventionResumeAmbiguousError",
    "DoVerOfficialInterventionPublicAdapter",
    "DoVerPublicInterventionReceiptV1",
    "DoVerPublicTrialV1",
    "DurableDoVerPublicInterventionGenerator",
    "build_dover_public_trial_from_terminal_logs",
]
