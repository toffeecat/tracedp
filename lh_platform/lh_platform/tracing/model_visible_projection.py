"""Build one credential-safe trace projection for every model-based tracer."""

from __future__ import annotations

import json

from lh_platform.core.artifact_redaction import redact_artifact
from lh_platform.tracing.types import TraceInputV1, trace_input_from_dict

TERMINAL_CANONICAL_PUBLIC_EVENT_METADATA_KEY = "terminal_canonical_public_event_json_v1"


def _redact_embedded_terminal_events(payload: dict[str, object]) -> dict[str, object]:
    """Redact structured sidecars before their JSON encoding reaches text regexes."""

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("model-visible trace events are malformed")
    sanitized_sidecars: dict[int, str] = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("model-visible trace event is malformed")
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("model-visible trace metadata is malformed")
        raw = metadata.pop(TERMINAL_CANONICAL_PUBLIC_EVENT_METADATA_KEY, None)
        if raw is None:
            continue
        if not isinstance(raw, str):
            raise ValueError("Terminal public sidecar must be encoded as JSON text")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("Terminal public sidecar is not valid JSON") from error
        sanitized = redact_artifact(decoded)
        if not isinstance(sanitized, dict):
            raise ValueError("Terminal public sidecar must encode one object")
        sanitized_sidecars[index] = json.dumps(
            sanitized,
            ensure_ascii=False,
            sort_keys=True,
        )

    sanitized_payload = redact_artifact(payload)
    if not isinstance(sanitized_payload, dict):
        raise TypeError("redacted trace must remain a mapping")
    sanitized_events = sanitized_payload.get("events")
    if not isinstance(sanitized_events, list):
        raise ValueError("redacted trace events are malformed")
    for index, sidecar in sanitized_sidecars.items():
        event = sanitized_events[index]
        if not isinstance(event, dict) or not isinstance(event.get("metadata"), dict):
            raise ValueError("redacted trace metadata is malformed")
        event["metadata"][TERMINAL_CANONICAL_PUBLIC_EVENT_METADATA_KEY] = sidecar
    return sanitized_payload


def canonical_model_visible_trace(case: TraceInputV1) -> tuple[TraceInputV1, bool]:
    """Redact model-visible text while preserving frozen task and event identity."""

    payload = case.to_dict()
    sanitized = _redact_embedded_terminal_events(payload)
    projected = trace_input_from_dict(sanitized)
    if (
        projected.case_id != case.case_id
        or projected.task_id != case.task_id
        or tuple((event.event_id, event.sequence, event.kind) for event in projected.events)
        != tuple((event.event_id, event.sequence, event.kind) for event in case.events)
    ):
        raise ValueError("model-visible redaction changed frozen trace identity")
    return projected, projected.to_dict() != case.to_dict()


__all__ = [
    "TERMINAL_CANONICAL_PUBLIC_EVENT_METADATA_KEY",
    "canonical_model_visible_trace",
]
