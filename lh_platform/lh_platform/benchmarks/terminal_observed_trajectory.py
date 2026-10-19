"""Bind Pi native tool calls to exact actor-visible filesystem transitions."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.benchmarks.terminal_public_state import (
    PublicFileObservation,
    validate_actor_visible_path,
)


class TerminalObservationError(ValueError):
    """The observer log cannot support a sequential environment transition."""


@dataclass(frozen=True)
class TerminalFileWrite:
    path: str
    before: PublicFileObservation
    after: PublicFileObservation


@dataclass(frozen=True)
class TerminalObservedTransition:
    sequence: int
    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any]
    environment_step_executed: bool
    is_error: bool
    before: tuple[PublicFileObservation, ...]
    after: tuple[PublicFileObservation, ...]
    writes: tuple[TerminalFileWrite, ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise TerminalObservationError(f"invalid JSON at {path.name}:{line_number}") from error
        if not isinstance(row, dict):
            raise TerminalObservationError(
                f"observer rows must be objects: {path.name}:{line_number}"
            )
        rows.append(row)
    return rows


def _session_tool_chain(
    rows: list[dict[str, Any]],
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, bool], set[str]]:
    calls = []
    results: dict[str, bool] = {}
    unavailable_tool_ids: set[str] = set()
    for row in rows:
        message = row.get("message")
        if row.get("type") != "message" or not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                arguments = block.get("arguments")
                if not isinstance(arguments, dict):
                    raise TerminalObservationError("native tool call has invalid arguments")
                calls.append((str(block.get("id", "")), str(block.get("name", "")), arguments))
        elif role == "toolResult":
            tool_call_id = str(message.get("toolCallId", ""))
            if tool_call_id in results:
                raise TerminalObservationError("duplicate native tool result")
            is_error = message.get("isError")
            if type(is_error) is not bool:
                raise TerminalObservationError("native tool result has no error status")
            results[tool_call_id] = is_error
            content = message.get("content") or []
            texts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            tool_name = str(message.get("toolName", ""))
            if is_error and texts == [f"Tool {tool_name} not found"]:
                unavailable_tool_ids.add(tool_call_id)
    call_ids = [tool_call_id for tool_call_id, _, _ in calls]
    if any(not tool_call_id for tool_call_id in call_ids) or len(call_ids) != len(set(call_ids)):
        raise TerminalObservationError("native tool-call IDs must be non-empty and unique")
    if set(call_ids) != set(results):
        raise TerminalObservationError("native session tool-call/result coverage mismatch")
    return calls, results, unavailable_tool_ids


def _decode_observation(row: dict[str, Any]) -> PublicFileObservation:
    encoded = row.get("content_base64")
    try:
        content = base64.b64decode(encoded, validate=True) if isinstance(encoded, str) else None
    except (binascii.Error, ValueError) as error:
        raise TerminalObservationError("public file observation has invalid bytes") from error
    exists = row.get("exists")
    if type(exists) is not bool:
        raise TerminalObservationError("public file observation has invalid existence state")
    file_type = str(row["file_type"]) if row.get("file_type") is not None else None
    if exists and file_type == "regular file" and content is None:
        raise TerminalObservationError("existing public file must include exact bytes")
    if exists and file_type == "directory" and content is not None:
        raise TerminalObservationError("public directory cannot include file bytes")
    if exists and file_type not in {"regular file", "directory"}:
        raise TerminalObservationError("existing public path has unsupported type")
    if not exists and any(
        row.get(key) is not None for key in ("file_type", "mode", "content_base64")
    ):
        raise TerminalObservationError("missing public file cannot include file metadata")
    return PublicFileObservation(
        path=str(row.get("path", "")),
        exists=exists,
        file_type=file_type,
        mode=str(row["mode"]) if row.get("mode") is not None else None,
        content=content,
    )


def _decode_state(
    rows: object,
    *,
    public_paths: tuple[str, ...],
) -> tuple[PublicFileObservation, ...]:
    if not isinstance(rows, list):
        raise TerminalObservationError("public filesystem snapshot must be a list")
    observations = tuple(_decode_observation(row) for row in rows if isinstance(row, dict))
    if len(observations) != len(rows):
        raise TerminalObservationError("public filesystem snapshot contains a non-object")
    paths = tuple(observation.path for observation in observations)
    if len(paths) != len(set(paths)) or set(paths) != set(public_paths):
        raise TerminalObservationError("public filesystem snapshot path coverage mismatch")
    by_path = {observation.path: observation for observation in observations}
    return tuple(by_path[path] for path in public_paths)


def load_observed_terminal_transitions(
    *,
    session_path: Path,
    observer_path: Path,
    public_paths: tuple[str, ...],
) -> tuple[TerminalObservedTransition, ...]:
    """Require a complete sequential Pi tool-call → filesystem transition chain."""
    public_paths = tuple(validate_actor_visible_path(path) for path in public_paths)
    if len(public_paths) != len(set(public_paths)):
        raise TerminalObservationError("public paths must be unique")
    calls, results, unavailable_tool_ids = _session_tool_chain(_read_jsonl(session_path))
    observer_rows = _read_jsonl(observer_path)
    observer_ids = [str(row.get("tool_call_id", "")) for row in observer_rows]
    call_ids = [tool_call_id for tool_call_id, _, _ in calls]
    if len(observer_ids) != len(set(observer_ids)) or (
        set(observer_ids) | unavailable_tool_ids
    ) != set(call_ids):
        raise TerminalObservationError("observer tool-call coverage or order mismatch")
    observer_by_id = {
        str(row["tool_call_id"]): (index, row) for index, row in enumerate(observer_rows)
    }
    executed_observer_ids = [
        tool_call_id
        for tool_call_id in observer_ids
        if observer_by_id[tool_call_id][1].get("environment_step_executed") is True
    ]
    executed_call_ids = [
        tool_call_id
        for tool_call_id in call_ids
        if tool_call_id in observer_by_id
        and observer_by_id[tool_call_id][1].get("environment_step_executed") is True
    ]
    if executed_observer_ids != executed_call_ids:
        raise TerminalObservationError("executed observer tool-call order mismatch")

    transitions = []
    for sequence, (tool_call_id, tool_name, tool_input) in enumerate(calls):
        if tool_call_id in unavailable_tool_ids:
            transitions.append(
                TerminalObservedTransition(
                    sequence=sequence,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    environment_step_executed=False,
                    is_error=True,
                    before=(),
                    after=(),
                    writes=(),
                )
            )
            continue
        row = observer_by_id[tool_call_id][1]
        rejected_parallel = (
            row.get("overlapping_tool_calls") is True
            and row.get("environment_step_executed") is False
            and row.get("error_stage") == "parallel_rejection"
        )
        if row.get("overlapping_tool_calls") is True and not rejected_parallel:
            raise TerminalObservationError("parallel tool execution cannot form a causal ledger")
        if row.get("tool_name") != tool_name:
            raise TerminalObservationError("observer tool name mismatch")
        if row.get("input") != tool_input:
            raise TerminalObservationError("observer tool input mismatch")
        if row.get("observation_complete") is not True and not rejected_parallel:
            raise TerminalObservationError("observer filesystem transition is incomplete")
        environment_step_executed = row.get("environment_step_executed")
        if type(environment_step_executed) is not bool:
            raise TerminalObservationError("observer has no environment-step status")
        is_error = row.get("is_error")
        if type(is_error) is not bool or is_error != results[tool_call_id]:
            raise TerminalObservationError("observer tool result mismatch")
        before = (
            () if rejected_parallel else _decode_state(row.get("before"), public_paths=public_paths)
        )
        after = (
            () if rejected_parallel else _decode_state(row.get("after"), public_paths=public_paths)
        )
        writes = (
            ()
            if rejected_parallel
            else tuple(
                TerminalFileWrite(path=path, before=before_state, after=after_state)
                for path, before_state, after_state in zip(public_paths, before, after, strict=True)
                if before_state != after_state
            )
        )
        transitions.append(
            TerminalObservedTransition(
                sequence=sequence,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_input=tool_input,
                environment_step_executed=environment_step_executed,
                is_error=is_error,
                before=before,
                after=after,
                writes=writes,
            )
        )
    return tuple(transitions)
