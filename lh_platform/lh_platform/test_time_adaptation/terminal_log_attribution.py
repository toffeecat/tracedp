"""Compile an actor-visible Terminal Pi session into a root-blind trace case."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lh_platform.tracing.model_visible_projection import (
    TERMINAL_CANONICAL_PUBLIC_EVENT_METADATA_KEY,
)
from lh_platform.tracing.types import (
    TraceCaseV1,
    TraceContextV1,
    TraceEventV1,
    TraceOutcomeV1,
)

TERMINAL_CANONICAL_PUBLIC_TRACE_SCHEMA = "terminal_canonical_public_trace_v1"
TERMINAL_PUBLIC_STATE_PROMPT_PROJECTION = (
    "path_metadata_and_base64_size_without_content_v1"
)
TERMINAL_ATTRIBUTION_FULL_SELECTION_RULE = (
    "all_protocol_complete_natural_failures"
)
TERMINAL_ATTRIBUTION_DEVELOPMENT_SELECTION_RULE = (
    "first_n_protocol_complete_natural_failures_by_task_id_at_snapshot"
)
TERMINAL_ATTRIBUTION_SELECTION_RULES = frozenset(
    {
        TERMINAL_ATTRIBUTION_FULL_SELECTION_RULE,
        TERMINAL_ATTRIBUTION_DEVELOPMENT_SELECTION_RULE,
    }
)
_CANONICAL_PUBLIC_FIELDS = (
    "actor_input",
    "observation_before",
    "raw_model_response",
    "parsed_action",
    "tool_call",
    "tool_execution_status",
    "tool_result",
    "public_state_delta",
)
_PUBLIC_OBSERVATION_FIELDS = {
    "path",
    "exists",
    "file_type",
    "mode",
    "content_base64",
}


def _compact(text: str, *, limit: int = 6000) -> str:
    """Bound each visible field without applying outcome-dependent truncation."""

    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated equally for every method]"


def _content_text(content: object) -> str:
    """Extract only actor-visible text from a Pi message content field."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for item in content:
        if isinstance(item, Mapping) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _public_observation(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Accept only the public file fields emitted by the Terminal observer."""

    if set(row) != _PUBLIC_OBSERVATION_FIELDS:
        return None
    path = row.get("path")
    exists = row.get("exists")
    file_type = row.get("file_type")
    mode = row.get("mode")
    content = row.get("content_base64")
    if (
        not isinstance(path, str)
        or not path.startswith("/app/")
        or ".." in path
        or not isinstance(exists, bool)
        or (file_type is not None and not isinstance(file_type, str))
        or (mode is not None and not isinstance(mode, str))
        or (content is not None and not isinstance(content, str))
    ):
        return None
    if not exists and any(value is not None for value in (file_type, mode, content)):
        return None
    return {
        "path": path,
        "exists": exists,
        "file_type": file_type,
        "mode": mode,
        "content_base64": content,
    }


def _public_state_delta(observer: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return an exact public-path delta only when both snapshots are complete."""

    if observer is None or observer.get("observation_complete") is not True:
        return None
    before = observer.get("before")
    after = observer.get("after")
    if not isinstance(before, list) or not isinstance(after, list):
        return None
    if any(not isinstance(row, Mapping) for row in (*before, *after)):
        return None
    normalized_before = [_public_observation(row) for row in before]
    normalized_after = [_public_observation(row) for row in after]
    if any(row is None for row in (*normalized_before, *normalized_after)):
        return None
    before_by_path = {str(row["path"]): row for row in normalized_before if row is not None}
    after_by_path = {str(row["path"]): row for row in normalized_after if row is not None}
    if (
        len(before_by_path) != len(before)
        or len(after_by_path) != len(after)
        or set(before_by_path) != set(after_by_path)
    ):
        return None
    changed_paths = sorted(
        path
        for path in set(before_by_path) | set(after_by_path)
        if before_by_path.get(path) != after_by_path.get(path)
    )
    return {
        "observation_complete": True,
        "changed_paths": changed_paths,
        "changes": [
            {
                "path": path,
                "before": before_by_path.get(path),
                "after": after_by_path.get(path),
            }
            for path in changed_paths
        ],
    }


def _validate_parsed_action(value: object) -> dict[str, Any]:
    """Validate the two actor action variants exposed by the canonical projection."""

    if not isinstance(value, Mapping):
        raise ValueError("canonical public parsed action must be an object")
    action_type = value.get("type")
    if action_type == "assistant_message":
        if set(value) != {"type", "assistant_text"} or not isinstance(
            value.get("assistant_text"), str
        ):
            raise ValueError("canonical public assistant action fields changed")
    elif action_type == "tool_call":
        if set(value) != {"type", "tool_name", "arguments", "assistant_text"}:
            raise ValueError("canonical public tool action fields changed")
        if (
            not isinstance(value.get("tool_name"), str)
            or not value["tool_name"]
            or not isinstance(value.get("arguments"), Mapping)
            or not isinstance(value.get("assistant_text"), str)
        ):
            raise ValueError("canonical public tool action values are invalid")
    else:
        raise ValueError("canonical public parsed action type is invalid")
    return dict(value)


def _validate_tool_call(value: object) -> dict[str, Any] | None:
    """Accept only the public Pi dispatch identity and actor-selected arguments."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "tool_call_id",
        "tool_name",
        "arguments",
    }:
        raise ValueError("canonical public tool call fields changed")
    if (
        not isinstance(value.get("tool_call_id"), str)
        or not value["tool_call_id"]
        or not isinstance(value.get("tool_name"), str)
        or not value["tool_name"]
        or not isinstance(value.get("arguments"), Mapping)
    ):
        raise ValueError("canonical public tool call values are invalid")
    return dict(value)


def _validate_tool_result(value: object) -> dict[str, Any] | None:
    """Validate the normalized durable Pi tool result without accepting extra fields."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"status", "is_error", "text"}:
        raise ValueError("canonical public tool result fields changed")
    status = value.get("status")
    is_error = value.get("is_error")
    if (
        status not in {"returned", "error"}
        or not isinstance(is_error, bool)
        or not isinstance(value.get("text"), str)
        or (status == "error") != is_error
    ):
        raise ValueError("canonical public tool result values are invalid")
    return dict(value)


def _validate_public_state_delta(value: object) -> dict[str, Any] | None:
    """Reject unknown state fields before a sidecar becomes method-visible."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "observation_complete",
        "changed_paths",
        "changes",
    }:
        raise ValueError("canonical public state delta fields changed")
    paths = value.get("changed_paths")
    changes = value.get("changes")
    if (
        value.get("observation_complete") is not True
        or not isinstance(paths, list)
        or paths != sorted(set(paths))
        or not all(isinstance(path, str) for path in paths)
        or not isinstance(changes, list)
        or len(changes) != len(paths)
    ):
        raise ValueError("canonical public state delta values are invalid")
    normalized_changes: list[dict[str, Any]] = []
    for path, change in zip(paths, changes, strict=True):
        if not isinstance(change, Mapping) or set(change) != {"path", "before", "after"}:
            raise ValueError("canonical public state change fields changed")
        before = change.get("before")
        after = change.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError("canonical public state change snapshots are invalid")
        normalized_before = _public_observation(before)
        normalized_after = _public_observation(after)
        if (
            normalized_before is None
            or normalized_after is None
            or change.get("path") != path
            or normalized_before["path"] != path
            or normalized_after["path"] != path
            or normalized_before == normalized_after
        ):
            raise ValueError("canonical public state change values are invalid")
        normalized_changes.append(
            {"path": path, "before": normalized_before, "after": normalized_after}
        )
    return {
        "observation_complete": True,
        "changed_paths": list(paths),
        "changes": normalized_changes,
    }


def _validated_observer(
    observer: Mapping[str, Any] | None,
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    result: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Use observer evidence only when it matches the exact sequential Pi call."""

    if observer is None:
        return None, None
    overlap_status = observer.get("overlapping_tool_calls")
    if overlap_status is True:
        return None, "overlapping_tool_calls"
    if overlap_status is not False:
        return None, "missing_or_invalid_overlap_status"
    if observer.get("tool_name") != tool_name:
        return None, "tool_name_mismatch"
    if observer.get("input") != dict(arguments):
        return None, "tool_input_mismatch"
    if not isinstance(observer.get("environment_step_executed"), bool):
        return None, "missing_execution_status"
    if result is None or not isinstance(result.get("isError"), bool):
        return None, "missing_durable_tool_result"
    if observer.get("is_error") != result.get("isError"):
        return None, "tool_result_mismatch"
    return observer, None


def _method_visible_event_content(event: CanonicalTerminalPublicEventV1) -> str:
    """Render every canonical field while retaining legacy labels used by LongH."""

    action = event.parsed_action
    actor_text = str(action.get("assistant_text", ""))
    parts = [
        "Actor input:\n" + (event.actor_input or "[not persisted]"),
        "Observation before:\n" + (event.observation_before or "[not persisted]"),
        "Raw model response:\n" + (event.raw_model_response or "[not persisted]"),
        "Actor message:\n" + (actor_text or "[no text]"),
    ]
    if event.tool_call is not None:
        result = event.tool_result
        parts.extend(
            [
                f"Tool: {event.tool_call['tool_name']}",
                "Arguments: "
                + json.dumps(event.tool_call["arguments"], ensure_ascii=False, sort_keys=True),
                "Environment step executed: "
                + (
                    str(event.tool_execution_status).lower()
                    if event.tool_execution_status is not None
                    else "unknown"
                ),
                "Tool result status: " + (str(result["status"]) if result else "missing"),
                "Tool error: "
                + (str(result["is_error"]).lower() if result else "unknown"),
                "Result:\n" + (str(result["text"]) if result else "[missing]"),
                "Public state delta: "
                + (
                    json.dumps(
                        _public_state_delta_prompt_view(event.public_state_delta),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if event.public_state_delta is not None
                    else "[not persisted]"
                ),
            ]
        )
    return "\n".join(parts)


def _public_state_delta_prompt_view(delta: Mapping[str, Any]) -> dict[str, Any]:
    """Keep write identity visible without sending binary file snapshots to tracers."""

    def summarize(observation: Mapping[str, Any]) -> dict[str, Any]:
        content = observation.get("content_base64")
        return {
            "path": observation.get("path"),
            "exists": observation.get("exists"),
            "file_type": observation.get("file_type"),
            "mode": observation.get("mode"),
            "content_base64": (
                None
                if content is None
                else {
                    "omitted_from_prompt": True,
                    "encoded_character_count": len(content),
                }
            ),
        }

    return {
        "projection": TERMINAL_PUBLIC_STATE_PROMPT_PROJECTION,
        "observation_complete": delta["observation_complete"],
        "changed_paths": list(delta["changed_paths"]),
        "changes": [
            {
                "path": change["path"],
                "before": summarize(change["before"]),
                "after": summarize(change["after"]),
            }
            for change in delta["changes"]
        ],
    }


def _structured_public_event_prompt_view(
    event: CanonicalTerminalPublicEventV1,
) -> dict[str, Any]:
    """Keep the native schema while enforcing the shared public-state projection."""

    payload = event.to_dict()
    if event.public_state_delta is not None:
        payload["public_state_delta"] = _public_state_delta_prompt_view(
            event.public_state_delta
        )
    return payload


@dataclass(frozen=True)
class CanonicalTerminalPublicEventV1:
    """Expose one event through explicit public fields without filling missing evidence."""

    event_id: str
    sequence: int
    kind: str
    parent_event_ids: tuple[str, ...]
    actor_input: str | None
    observation_before: str | None
    raw_model_response: str | None
    parsed_action: dict[str, Any] | None
    tool_call: dict[str, Any] | None
    tool_execution_status: bool | None
    tool_result: dict[str, Any] | None
    public_state_delta: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "parent_event_ids": list(self.parent_event_ids),
            **{field: getattr(self, field) for field in _CANONICAL_PUBLIC_FIELDS},
        }

    def applicable_fields(self) -> frozenset[str]:
        common = {
            "actor_input",
            "observation_before",
            "raw_model_response",
            "parsed_action",
        }
        if self.tool_call is not None:
            common.update(
                {
                    "tool_call",
                    "tool_execution_status",
                    "tool_result",
                    "public_state_delta",
                }
            )
        return frozenset(common)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalTerminalPublicEventV1:
        """Validate one persisted event before it becomes method-visible input."""

        expected = {
            "event_id",
            "sequence",
            "kind",
            "parent_event_ids",
            *_CANONICAL_PUBLIC_FIELDS,
        }
        if set(payload) != expected:
            raise ValueError("canonical public event fields changed")
        event_id = payload.get("event_id")
        sequence = payload.get("sequence")
        kind = payload.get("kind")
        parents = payload.get("parent_event_ids")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("canonical public event identity is invalid")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("canonical public event sequence is invalid")
        if not isinstance(kind, str) or not kind:
            raise ValueError("canonical public event kind is invalid")
        if not isinstance(parents, list) or not all(
            isinstance(parent, str) and parent for parent in parents
        ):
            raise ValueError("canonical public event parents are invalid")
        for field in ("actor_input", "observation_before", "raw_model_response"):
            value = payload.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"canonical public event {field} must be text or null")
        for field in ("parsed_action", "tool_call", "tool_result", "public_state_delta"):
            value = payload.get(field)
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(f"canonical public event {field} must be an object or null")
        execution_status = payload.get("tool_execution_status")
        if execution_status is not None and not isinstance(execution_status, bool):
            raise ValueError("canonical public tool execution status must be boolean or null")
        parsed_action = _validate_parsed_action(payload.get("parsed_action"))
        tool_call = _validate_tool_call(payload.get("tool_call"))
        tool_result = _validate_tool_result(payload.get("tool_result"))
        state_delta = _validate_public_state_delta(payload.get("public_state_delta"))
        if (tool_call is None) != (parsed_action.get("type") != "tool_call"):
            raise ValueError("canonical public action and tool call disagree")
        if tool_call is not None and (
            parsed_action.get("tool_name") != tool_call["tool_name"]
            or parsed_action.get("arguments") != tool_call["arguments"]
        ):
            raise ValueError("canonical public parsed action and tool call changed")
        if tool_call is None and any(
            value is not None for value in (execution_status, tool_result, state_delta)
        ):
            raise ValueError("non-tool event contains tool-only public evidence")
        return cls(
            event_id=event_id,
            sequence=sequence,
            kind=kind,
            parent_event_ids=tuple(parents),
            actor_input=payload.get("actor_input"),
            observation_before=payload.get("observation_before"),
            raw_model_response=payload.get("raw_model_response"),
            parsed_action=parsed_action,
            tool_call=tool_call,
            tool_execution_status=execution_status,
            tool_result=tool_result,
            public_state_delta=state_delta,
        )


@dataclass(frozen=True)
class CanonicalTerminalPublicTraceV1:
    """Freeze the same structured, root-blind trace projection for every static method."""

    case_id: str
    task_id: str
    events: tuple[CanonicalTerminalPublicEventV1, ...]

    @staticmethod
    def source_semantics() -> dict[str, str]:
        """Describe which durable source, if any, can populate every public field."""

        return {
            "actor_input": "present_only_if_exact_actor_input_is_persisted",
            "observation_before": "latest_public_user_or_tool_result_before_actor_output",
            "raw_model_response": "present_only_if_the_durable_pi_message_contains_exact_raw_response_text",
            "parsed_action": "normalized_durable_pi_assistant_action",
            "tool_call": "durable_pi_tool_dispatch",
            "tool_execution_status": "observer_execution_status_when_available",
            "tool_result": "durable_pi_tool_result",
            "public_state_delta": "outcome_blind_public_paths_only",
        }

    def field_coverage(self) -> dict[str, dict[str, int | float]]:
        coverage: dict[str, dict[str, int | float]] = {}
        for field in _CANONICAL_PUBLIC_FIELDS:
            applicable = sum(field in event.applicable_fields() for event in self.events)
            present = sum(
                field in event.applicable_fields() and getattr(event, field) is not None
                for event in self.events
            )
            coverage[field] = {
                "present": present,
                "applicable": applicable,
                "coverage": 0.0 if applicable == 0 else present / applicable,
            }
        return coverage

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TERMINAL_CANONICAL_PUBLIC_TRACE_SCHEMA,
            "case_id": self.case_id,
            "task_id": self.task_id,
            "source_semantics": self.source_semantics(),
            "events": [event.to_dict() for event in self.events],
            "field_coverage": self.field_coverage(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalTerminalPublicTraceV1:
        """Load a sidecar only when its schema, semantics, and coverage are exact."""

        expected = {
            "schema_version",
            "case_id",
            "task_id",
            "source_semantics",
            "events",
            "field_coverage",
        }
        if set(payload) != expected:
            raise ValueError("canonical public trace fields changed")
        if payload.get("schema_version") != TERMINAL_CANONICAL_PUBLIC_TRACE_SCHEMA:
            raise ValueError("canonical public trace schema changed")
        case_id = payload.get("case_id")
        task_id = payload.get("task_id")
        if not isinstance(case_id, str) or not case_id or not isinstance(task_id, str) or not task_id:
            raise ValueError("canonical public trace identity is invalid")
        if payload.get("source_semantics") != cls.source_semantics():
            raise ValueError("canonical public trace source semantics changed")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("canonical public trace events are invalid")
        events = tuple(
            CanonicalTerminalPublicEventV1.from_dict(event)
            for event in raw_events
            if isinstance(event, Mapping)
        )
        if len(events) != len(raw_events):
            raise ValueError("canonical public trace event is not an object")
        if [event.sequence for event in events] != list(range(len(events))):
            raise ValueError("canonical public trace event order changed")
        if len({event.event_id for event in events}) != len(events):
            raise ValueError("canonical public trace event identities changed")
        seen: set[str] = set()
        for event in events:
            if set(event.parent_event_ids) - seen:
                raise ValueError("canonical public trace event parents changed")
            seen.add(event.event_id)
        trace = cls(case_id=case_id, task_id=task_id, events=events)
        if payload.get("field_coverage") != trace.field_coverage():
            raise ValueError("canonical public trace field coverage changed")
        return trace

    def bind_trace_case(self, trace: TraceCaseV1) -> TraceCaseV1:
        """Make the validated sidecar the sole event representation sent to methods."""

        if trace.case_id != self.case_id or trace.task_id != self.task_id:
            raise ValueError("canonical public trace does not match the blind trace identity")
        legacy_envelope = [
            (event.event_id, event.sequence, event.kind, event.parent_event_ids)
            for event in trace.events
        ]
        public_envelope = [
            (event.event_id, event.sequence, event.kind, event.parent_event_ids)
            for event in self.events
        ]
        if legacy_envelope != public_envelope:
            raise ValueError("canonical public trace event envelope changed")
        projected_events = tuple(
            TraceEventV1(
                event_id=public.event_id,
                sequence=public.sequence,
                kind=public.kind,
                content=_method_visible_event_content(public),
                parent_event_ids=public.parent_event_ids,
                metadata=tuple(
                    sorted(
                        {
                            "public_projection_schema": TERMINAL_CANONICAL_PUBLIC_TRACE_SCHEMA,
                            TERMINAL_CANONICAL_PUBLIC_EVENT_METADATA_KEY: json.dumps(
                                _structured_public_event_prompt_view(public),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            **(
                                {
                                    "tool_call_id": str(public.tool_call["tool_call_id"]),
                                    "tool_name": str(public.tool_call["tool_name"]),
                                    "tool_result_status": (
                                        str(public.tool_result["status"])
                                        if public.tool_result is not None
                                        else "missing"
                                    ),
                                }
                                if public.tool_call is not None
                                else {}
                            ),
                        }.items()
                    )
                ),
            )
            for public in self.events
        )
        return TraceCaseV1(
            case_id=trace.case_id,
            task_id=trace.task_id,
            events=projected_events,
            interruption_event_id=trace.interruption_event_id,
            observed_outcome=trace.observed_outcome,
            context=trace.context,
        )


def aggregate_terminal_canonical_field_coverage(
    rows: Sequence[Mapping[str, Mapping[str, int | float]]],
) -> dict[str, dict[str, int | float]]:
    """Aggregate validated per-case coverage without counting inapplicable fields."""

    aggregate: dict[str, dict[str, int | float]] = {}
    for field in _CANONICAL_PUBLIC_FIELDS:
        present = sum(int(row[field]["present"]) for row in rows)
        applicable = sum(int(row[field]["applicable"]) for row in rows)
        aggregate[field] = {
            "present": present,
            "applicable": applicable,
            "coverage": 0.0 if applicable == 0 else present / applicable,
        }
    return aggregate


@dataclass(frozen=True)
class TerminalLogTraceCompilation:
    """One blind trace and the durable session coverage used to create it."""

    trace_case: TraceCaseV1
    assistant_turn_count: int
    tool_call_count: int
    tool_result_count: int
    tool_error_count: int
    missing_tool_result_ids: tuple[str, ...]
    observer_validation_errors: tuple[str, ...]
    canonical_public_projection: CanonicalTerminalPublicTraceV1


def build_terminal_log_trace_case(
    *,
    case_id: str,
    task_id: str,
    task_specification: str,
    failure_summary: str,
    terminal_reason: str,
    official_reward: float | None,
    session_records: Sequence[Mapping[str, Any]],
    observer_records: Sequence[Mapping[str, Any]] = (),
) -> TerminalLogTraceCompilation:
    """Project a complete or timeout-ended Pi log without requiring state replay.

    The function never receives a solution, verifier implementation, reference
    answer, hidden root, or tracer output. Tool calls without a returned result
    remain visible as terminal-boundary evidence instead of invalidating the
    otherwise useful attribution trajectory.
    """

    if not case_id or not task_id:
        raise ValueError("Terminal log attribution requires case and task identities")
    if not task_specification.strip() or not failure_summary.strip():
        raise ValueError("Terminal log attribution requires public task and failure text")
    if not terminal_reason.strip():
        raise ValueError("Terminal log attribution requires a terminal reason")
    if official_reward is not None and (
        isinstance(official_reward, bool)
        or not isinstance(official_reward, (int, float))
        or float(official_reward) != 0.0
    ):
        raise ValueError("Terminal failure trace requires reward zero or unavailable")

    user_texts: list[str] = []
    result_by_call: dict[str, Mapping[str, Any]] = {}
    call_ids: list[str] = []
    assistant_turns: list[tuple[Mapping[str, Any], str]] = []
    latest_observation = ""
    for row in session_records:
        message = row.get("message")
        if row.get("type") != "message" or not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "user":
            text = _content_text(message.get("content"))
            user_texts.append(text)
            latest_observation = text
        elif role == "assistant":
            assistant_turns.append((message, latest_observation))
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, Mapping) and block.get("type") == "toolCall":
                        call_id = block.get("id")
                        if not isinstance(call_id, str) or not call_id:
                            raise ValueError("Terminal Pi tool call has no stable identity")
                        call_ids.append(call_id)
        elif role == "toolResult":
            call_id = message.get("toolCallId")
            if (
                not isinstance(call_id, str)
                or not call_id
                or call_id in result_by_call
            ):
                raise ValueError("Terminal Pi tool result identity is invalid")
            result_by_call[call_id] = message
            latest_observation = _content_text(message.get("content"))

    if not user_texts or not any(
        _normalized(task_specification) in _normalized(text) for text in user_texts
    ):
        raise ValueError("Terminal Pi session is not bound to the public task instruction")
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("Terminal Pi tool-call identities are duplicated")
    if not set(result_by_call) <= set(call_ids):
        raise ValueError("Terminal Pi session contains an orphan tool result")
    observer_by_call: dict[str, Mapping[str, Any]] = {}
    for observer in observer_records:
        call_id = observer.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id or call_id in observer_by_call:
            raise ValueError("Terminal public-state observer identity is invalid")
        observer_by_call[call_id] = observer
    if not set(observer_by_call) <= set(call_ids):
        raise ValueError("Terminal public-state observer contains an orphan tool call")

    events: list[TraceEventV1] = []
    canonical_events: list[CanonicalTerminalPublicEventV1] = []
    observer_validation_errors: list[str] = []
    tool_errors = 0
    tool_ordinal = 0
    for turn_index, (message, observation_before) in enumerate(
        assistant_turns, start=1
    ):
        content = message.get("content")
        full_actor_text = _content_text(content)
        actor_text = _compact(full_actor_text)
        raw_response = message.get("rawResponse")
        if not isinstance(raw_response, str):
            raw_response = None
        calls = [
            block
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "toolCall"
        ] if isinstance(content, list) else []
        if not calls and actor_text.strip():
            event_id = f"turn-{turn_index:04d}"
            parent_event_ids = () if not events else (events[-1].event_id,)
            events.append(
                TraceEventV1(
                    event_id=event_id,
                    sequence=len(events),
                    kind="terminal_actor_decision",
                    content=f"Actor message:\n{actor_text}",
                    parent_event_ids=parent_event_ids,
                    metadata=(("assistant_turn", str(turn_index)),),
                )
            )
            canonical_events.append(
                CanonicalTerminalPublicEventV1(
                    event_id=event_id,
                    sequence=len(canonical_events),
                    kind="terminal_actor_decision",
                    parent_event_ids=parent_event_ids,
                    actor_input=None,
                    observation_before=observation_before or None,
                    raw_model_response=raw_response,
                    parsed_action={
                        "type": "assistant_message",
                        "assistant_text": full_actor_text,
                    },
                    tool_call=None,
                    tool_execution_status=None,
                    tool_result=None,
                    public_state_delta=None,
                )
            )
            continue
        for call_in_turn, call in enumerate(calls, start=1):
            tool_ordinal += 1
            call_id = str(call["id"])
            tool_name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("Terminal Pi tool call has no tool name")
            if not isinstance(arguments, Mapping):
                raise ValueError("Terminal Pi tool arguments must be an object")
            result = result_by_call.get(call_id)
            observer, observer_error = _validated_observer(
                observer_by_call.get(call_id),
                tool_name=tool_name,
                arguments=arguments,
                result=result,
            )
            if observer_error is not None:
                observer_validation_errors.append(f"{call_id}:{observer_error}")
            if result is None:
                result_text = "[missing at terminal boundary]"
                canonical_result_text = ""
                result_status = "missing"
                is_error = None
            else:
                raw_is_error = result.get("isError")
                if not isinstance(raw_is_error, bool):
                    raise ValueError("Terminal Pi tool result has no error status")
                is_error = raw_is_error
                tool_errors += int(is_error)
                result_status = "error" if is_error else "returned"
                canonical_result_text = _content_text(result.get("content"))
                result_text = _compact(canonical_result_text)
            event_id = f"step-{tool_ordinal:04d}"
            parent_event_ids = () if not events else (events[-1].event_id,)
            events.append(
                TraceEventV1(
                    event_id=event_id,
                    sequence=len(events),
                    kind="terminal_actor_tool_step",
                    content=(
                        f"Actor message:\n{actor_text or '[no text]'}\n"
                        f"Tool: {tool_name}\n"
                        f"Arguments: {json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True)}\n"
                        f"Tool result status: {result_status}\n"
                        f"Tool error: {str(is_error).lower() if is_error is not None else 'unknown'}\n"
                        f"Result:\n{result_text}"
                    ),
                    parent_event_ids=parent_event_ids,
                    metadata=tuple(
                        sorted(
                            {
                                "assistant_turn": str(turn_index),
                                "call_in_turn": str(call_in_turn),
                                "tool_call_id": call_id,
                                "tool_name": tool_name,
                                "tool_result_status": result_status,
                            }.items()
                        )
                    ),
                )
            )
            parsed_action = {
                "type": "tool_call",
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "assistant_text": full_actor_text,
            }
            canonical_events.append(
                CanonicalTerminalPublicEventV1(
                    event_id=event_id,
                    sequence=len(canonical_events),
                    kind="terminal_actor_tool_step",
                    parent_event_ids=parent_event_ids,
                    actor_input=None,
                    observation_before=observation_before or None,
                    raw_model_response=raw_response,
                    parsed_action=parsed_action,
                    tool_call={
                        "tool_call_id": call_id,
                        "tool_name": tool_name,
                        "arguments": dict(arguments),
                    },
                    tool_execution_status=(
                        observer.get("environment_step_executed")
                        if observer is not None
                        and isinstance(observer.get("environment_step_executed"), bool)
                        else None
                    ),
                    tool_result=(
                        None
                        if result is None
                        else {
                            "status": result_status,
                            "is_error": is_error,
                            "text": canonical_result_text,
                        }
                    ),
                    public_state_delta=_public_state_delta(observer),
                )
            )

    if not events:
        raise ValueError("Terminal Pi session contains no localizable actor event")
    missing = tuple(call_id for call_id in call_ids if call_id not in result_by_call)
    trace = TraceCaseV1(
        case_id=case_id,
        task_id=task_id,
        events=tuple(events),
        interruption_event_id=events[-1].event_id,
        observed_outcome=TraceOutcomeV1(
            success=False,
            terminal_reason=terminal_reason,
            score=None if official_reward is None else float(official_reward),
        ),
        context=TraceContextV1(
            task_specification=task_specification,
            initial_state=(
                "The official Terminal-Bench task environment was initialized before "
                "the first actor message."
            ),
            evaluator_failure=failure_summary,
        ),
    )
    return TerminalLogTraceCompilation(
        trace_case=trace,
        assistant_turn_count=len(assistant_turns),
        tool_call_count=len(call_ids),
        tool_result_count=len(result_by_call),
        tool_error_count=tool_errors,
        missing_tool_result_ids=missing,
        observer_validation_errors=tuple(observer_validation_errors),
        canonical_public_projection=CanonicalTerminalPublicTraceV1(
            case_id=case_id,
            task_id=task_id,
            events=tuple(canonical_events),
        ),
    )


__all__ = [
    "TERMINAL_CANONICAL_PUBLIC_EVENT_METADATA_KEY",
    "TERMINAL_ATTRIBUTION_DEVELOPMENT_SELECTION_RULE",
    "TERMINAL_ATTRIBUTION_FULL_SELECTION_RULE",
    "TERMINAL_ATTRIBUTION_SELECTION_RULES",
    "TERMINAL_CANONICAL_PUBLIC_TRACE_SCHEMA",
    "TERMINAL_PUBLIC_STATE_PROMPT_PROJECTION",
    "CanonicalTerminalPublicEventV1",
    "CanonicalTerminalPublicTraceV1",
    "TerminalLogTraceCompilation",
    "aggregate_terminal_canonical_field_coverage",
    "build_terminal_log_trace_case",
]
