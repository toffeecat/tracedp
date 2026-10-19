"""Runner-owned model runtime and per-action generation evidence.

This module binds actions to model events observed during one predict call.  It
also owns the conservative fallbacks used when adapters or authority channels
cannot provide trustworthy runtime evidence.  It never decides run admission
or benchmark outcomes.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from typing import Any

from lh_platform.core.artifact_redaction import redact_secrets
from lh_platform.core.generation_authority import (
    GENERATION_AUTHORITY_SCHEMA,
    GenerationAuthority,
)
from lh_platform.core.types import Action


def authority_runtime_evidence(
    authority: GenerationAuthority,
    *,
    cause: BaseException | None = None,
) -> dict[str, Any]:
    """Read the control-owned runtime snapshot without extending a broken run.

    A model or worker timeout makes the authority non-reusable.  Calling its
    control channel again with the normal generation timeout would turn one
    failed task into a long-lived zombie process, so emit a conservative
    observed record instead.
    """

    if authority.broken:
        return {
            "model_instantiated": False,
            "generation_authority_error": (
                type(cause).__name__ if cause is not None else "transport_failure"
            ),
            "generation_authority_broken": True,
        }
    try:
        return authority.runtime_evidence()
    except Exception as error:
        return {
            "model_instantiated": False,
            "generation_authority_error": type(error).__name__,
            "generation_authority_broken": True,
        }


def method_model_runtime_evidence(method: Any) -> dict[str, Any]:
    """Read one conservative runtime snapshot from a method-owned model.

    This helper is used around every ``predict`` call, rather than only once
    after a task.  A model can otherwise make one valid call early in a run
    and let unrelated later actions inherit that aggregate credibility.
    """

    producer = getattr(method, "model_runtime_evidence", None)
    if not callable(producer):
        return {"model_instantiated": False}
    try:
        observed = producer()
    except Exception as error:
        return {
            "model_instantiated": False,
            "runtime_evidence_error": type(error).__name__,
        }
    return (
        redact_secrets(dict(observed or {}))
        if isinstance(observed, dict)
        else {
            "model_instantiated": False,
            "runtime_evidence_error": "non_mapping",
        }
    )


def method_model_generation_cursor(method: Any) -> int:
    """Read the model-owned generation cursor before one predict call."""

    producer = getattr(method, "model_generation_cursor", None)
    if not callable(producer):
        return 0
    try:
        cursor = producer()
    except Exception:
        return 0
    return (
        int(cursor)
        if isinstance(cursor, int) and not isinstance(cursor, bool) and cursor >= 0
        else 0
    )


def method_model_generation_events_since(method: Any, cursor: int) -> tuple[dict[str, Any], ...]:
    """Return runner-visible copies of new model events, never adapter hints."""

    producer = getattr(method, "model_generation_events_since", None)
    if not callable(producer):
        return ()
    try:
        events = producer(cursor)
    except Exception:
        return ()
    if not isinstance(events, (list, tuple)):
        return ()
    return tuple(copy.deepcopy(dict(event)) for event in events if isinstance(event, dict))


def freeze_action(action: Action) -> Action:
    """Return an independent action snapshot suitable for persistence/use."""

    return copy.deepcopy(action)


def merge_benchmark_execution_annotations(
    persisted: Action,
    executed: Action,
) -> None:
    """Copy only benchmark-owned parse annotations onto a frozen action.

    Web/tool benchmarks historically annotate ``Action`` in place with parsed
    action-language data.  Passing an execution copy prevents arbitrary
    mutation of model evidence, while this narrow merge retains the useful
    parser result for downstream trajectory collection.  A benchmark that
    changes raw model text or the model-response binding is a contract error,
    not a reason to silently rewrite provenance.
    """

    if executed.raw != persisted.raw:
        raise RuntimeError("benchmark.step mutated action.raw after runner authentication")
    if executed.info.get("model_generation") != persisted.info.get("model_generation"):
        raise RuntimeError("benchmark.step mutated runner-owned model_generation evidence")
    persisted.action_type = executed.action_type
    persisted.tool_name = executed.tool_name
    persisted.tool_args = copy.deepcopy(executed.tool_args)
    for key in ("original_action_type", "action_language", "parsed_action"):
        if key in executed.info:
            persisted.info[key] = copy.deepcopy(executed.info[key])


def _runtime_count(snapshot: dict[str, Any], key: str) -> int:
    value = snapshot.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def in_process_action_authority_evidence(
    *,
    cursor_start: int,
    cursor_end: int,
) -> dict[str, Any]:
    """Mark legacy same-process evidence as diagnostic-only.

    The response hashes remain useful while developing adapters, but a method
    shares the process with that model object and can therefore not supply the
    isolation required by the official SWE-bench exporter.
    """

    return {
        "schema": GENERATION_AUTHORITY_SCHEMA,
        "mode": "in_process_untrusted",
        "generation_cursor_start": cursor_start,
        "generation_cursor_end": cursor_end,
    }


def action_model_generation_evidence(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    generation_events: Sequence[dict[str, Any]],
    authority: dict[str, Any],
    action: Action,
) -> dict[str, Any]:
    """Bind a persisted action to model activity observed during ``predict``.

    This is deliberately runner-owned evidence: adapters do not get to claim
    a remote generation by merely annotating an ``Action``.  The action hash
    additionally prevents changing its raw text after the runtime delta was
    recorded.  If an adapter stores a separate ``patch_text``, bind that text
    too so the official SWE-bench exporter can authenticate the exact patch.
    """

    raw = str(action.raw or "")
    raw_sha256 = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
    evidence: dict[str, Any] = {
        "schema": "lh_action_model_generation/v3",
        "action_sha256": raw_sha256,
        "authority": copy.deepcopy(authority),
        "model_instantiated_after": bool(after.get("model_instantiated")),
        "requested_model": after.get("requested_model"),
        "remote_call_delta": max(
            0,
            _runtime_count(after, "remote_call_count")
            - _runtime_count(before, "remote_call_count"),
        ),
        "remote_error_delta": max(
            0,
            _runtime_count(after, "remote_error_count")
            - _runtime_count(before, "remote_error_count"),
        ),
        "dummy_fallback_delta": max(
            0,
            _runtime_count(after, "dummy_fallback_count")
            - _runtime_count(before, "dummy_fallback_count"),
        ),
        "dummy_model_used": bool(after.get("dummy_model_used")),
        "dummy_fallback_used": bool(after.get("dummy_fallback_used")),
        "generation_event_count": len(generation_events),
        "raw_response_binding": _bind_content_to_generation_event(
            raw,
            generation_events,
        ),
    }
    patch_text = action.info.get("patch_text")
    if isinstance(patch_text, str):
        evidence["patch_text_sha256"] = (
            "sha256:" + hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
        )
        evidence["patch_text_response_binding"] = _bind_content_to_generation_event(
            patch_text,
            generation_events,
        )
    return evidence


def _bind_content_to_generation_event(
    content: str,
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Bind exact output text to a trusted model event from this predict call."""

    content_sha256 = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
    matching = [event for event in events if event.get("response_sha256") == content_sha256]
    if not matching:
        return {
            "schema": "lh_model_response_binding/v1",
            "status": "unbound",
            "content_sha256": content_sha256,
            "reason": "no_model_response_in_this_predict_matches_action_content",
        }
    # Prefer the most recent response when a model returned identical text
    # more than once in one predict call.  The selection remains explicit and
    # revalidated by the official exporter.
    selected = matching[-1]
    selected_event = {
        key: selected.get(key)
        for key in (
            "schema",
            "generation_id",
            "sequence",
            "requested_model",
            "backend",
            "outcome",
            "request_sha256",
            "response_sha256",
            "response_chars",
            "event_sha256",
        )
    }
    return {
        "schema": "lh_model_response_binding/v1",
        "status": "bound"
        if selected_event.get("outcome") == "remote_success"
        else "non_remote_response",
        "content_sha256": content_sha256,
        "generation_event": selected_event,
    }


def summarize_model_runtime(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-task model snapshots into one run-level observation."""

    if not samples:
        return {
            "model_instantiated": False,
            "dummy_model_used": False,
            "dummy_fallback_used": False,
            "task_observation_count": 0,
        }
    backends = sorted({str(sample.get("backend")) for sample in samples if sample.get("backend")})
    numeric_keys = (
        "generation_count",
        "remote_call_count",
        "remote_error_count",
        "dummy_fallback_count",
    )
    out: dict[str, Any] = {
        "task_observation_count": len(samples),
        "model_instantiated": any(bool(sample.get("model_instantiated")) for sample in samples),
        "dummy_model_used": any(bool(sample.get("dummy_model_used")) for sample in samples),
        "dummy_fallback_used": any(bool(sample.get("dummy_fallback_used")) for sample in samples),
        "backends": backends,
    }
    for key in numeric_keys:
        values = [sample.get(key) for sample in samples]
        numeric = [
            value
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric:
            out[key] = sum(numeric)
    requested_models = sorted(
        {str(sample.get("requested_model")) for sample in samples if sample.get("requested_model")}
    )
    if requested_models:
        out["requested_models"] = requested_models
    return out


__all__ = [
    "authority_runtime_evidence",
    "method_model_runtime_evidence",
    "method_model_generation_cursor",
    "method_model_generation_events_since",
    "freeze_action",
    "merge_benchmark_execution_annotations",
    "in_process_action_authority_evidence",
    "action_model_generation_evidence",
    "summarize_model_runtime",
]
