"""Validate and stage one historical Pi session for exact-context continuation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from lh_platform.benchmarks.terminal_pi_stream import (
    PiPersistentSessionToolReceipt,
    load_pi_persistent_session_tool_receipt,
)


@dataclass(frozen=True)
class FrozenPiContinuationSource:
    """Actor-visible state restored by Pi before a continuation prompt."""

    session_id: str
    session_version: int
    session_cwd: str
    message_count: int
    leaf_id: str
    task_instruction: str
    terminal_stop_reason: str
    tool_receipt: PiPersistentSessionToolReceipt


def normalize_terminal_instruction_boundary(instruction: str) -> str:
    """Remove at most one file-ending newline before instruction comparison."""

    if instruction.endswith("\r\n"):
        return instruction[:-2]
    if instruction.endswith(("\r", "\n")):
        return instruction[:-1]
    return instruction


def terminal_instructions_are_equivalent(source: str, received: str) -> bool:
    """Accept only the final newline difference introduced by Harbor's file loader."""

    return normalize_terminal_instruction_boundary(
        source
    ) == normalize_terminal_instruction_boundary(received)


def _active_branch(rows: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    """Reproduce Pi 0.84.1's last-entry leaf and parent traversal."""

    entries = [row for row in rows if row.get("type") != "session"]
    by_id: dict[str, dict[str, object]] = {}
    for row in entries:
        entry_id = row.get("id")
        parent_id = row.get("parentId")
        if (
            not isinstance(entry_id, str)
            or not entry_id
            or entry_id in by_id
            or (parent_id is not None and parent_id not in by_id)
        ):
            raise ValueError("historical Pi session tree is invalid")
        by_id[entry_id] = row
    if not entries:
        raise ValueError("historical Pi session has no active branch")
    branch: list[dict[str, object]] = []
    current: dict[str, object] | None = entries[-1]
    while current is not None:
        branch.append(current)
        parent_id = current.get("parentId")
        current = by_id.get(parent_id) if isinstance(parent_id, str) else None
    branch.reverse()
    return tuple(branch)


def _message_text(message: dict[str, object]) -> str:
    """Extract the text Pi will rebuild for one user message."""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("historical Pi user message content is invalid")
    texts = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if len(texts) != len(content):
        raise ValueError("historical Pi task prompt contains non-text content")
    return "\n".join(texts)


def load_frozen_pi_continuation_source(
    path: Path,
    *,
    expected_instruction: str,
    expected_provider: str,
    expected_model: str,
    expected_thinking: str,
    expected_terminal_stop_reason: str,
    expected_cwd: Path,
) -> FrozenPiContinuationSource:
    """Load the exact active context that upstream SessionManager will restore."""

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"historical Pi session JSON is invalid at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError("historical Pi session row is not an object")
        rows.append(row)
    if not rows or rows[0].get("type") != "session":
        raise ValueError("historical Pi session header is missing")
    header = rows[0]
    session_id = header.get("id")
    session_version = header.get("version")
    session_cwd = header.get("cwd")
    if (
        not isinstance(session_id, str)
        or not session_id
        or session_version != 3
        or not isinstance(session_cwd, str)
        or Path(session_cwd).resolve() != expected_cwd.resolve()
        or not expected_cwd.is_dir()
        or len([row for row in rows if row.get("type") == "session"]) != 1
    ):
        raise ValueError("historical Pi session identity differs")

    branch = _active_branch(rows)
    user_messages: list[str] = []
    assistant_messages: list[dict[str, object]] = []
    model_changes: list[tuple[object, object]] = []
    thinking_changes: list[object] = []
    for row in branch:
        if row.get("type") == "model_change":
            model_changes.append((row.get("provider"), row.get("modelId")))
        elif row.get("type") == "thinking_level_change":
            thinking_changes.append(row.get("thinkingLevel"))
        elif row.get("type") == "message":
            message = row.get("message")
            if not isinstance(message, dict):
                raise ValueError("historical Pi message entry is invalid")
            role = message.get("role")
            if role == "user":
                user_messages.append(_message_text(message))
            elif role == "assistant":
                if (
                    message.get("provider") != expected_provider
                    or message.get("model") != expected_model
                ):
                    raise ValueError("historical Pi assistant model identity differs")
                assistant_messages.append(message)
    if model_changes[-1:] != [(expected_provider, expected_model)]:
        raise ValueError("historical Pi active model differs")
    if thinking_changes[-1:] != [expected_thinking]:
        raise ValueError("historical Pi active thinking level differs")
    if len(user_messages) != 1 or not terminal_instructions_are_equivalent(
        user_messages[0], expected_instruction
    ):
        raise ValueError("historical Pi task instruction differs")
    if not assistant_messages:
        raise ValueError("historical Pi session has no assistant response")
    terminal_stop_reason = assistant_messages[-1].get("stopReason")
    if terminal_stop_reason != expected_terminal_stop_reason:
        raise ValueError("historical Pi terminal stop reason differs")

    tool_receipt = load_pi_persistent_session_tool_receipt(
        path,
        expected_session_id=session_id,
        expected_provider=expected_provider,
        expected_model=expected_model,
    )
    if tool_receipt.uncompleted_tool_call_ids:
        raise ValueError("historical Pi session has incomplete tool calls")
    messages = [row for row in branch if row.get("type") == "message"]
    leaf_id = branch[-1].get("id")
    if not isinstance(leaf_id, str) or not leaf_id:
        raise ValueError("historical Pi active leaf is invalid")
    return FrozenPiContinuationSource(
        session_id=session_id,
        session_version=session_version,
        session_cwd=session_cwd,
        message_count=len(messages),
        leaf_id=leaf_id,
        task_instruction=user_messages[0],
        terminal_stop_reason=expected_terminal_stop_reason,
        tool_receipt=tool_receipt,
    )


def active_session_message_count(path: Path) -> int:
    """Count message rows on a (possibly continued) Pi session's active branch.

    The frozen-source loader validates identity and terminal state, which a continued
    session deliberately violates; a resumed actor only needs the current count of
    `message` rows the active branch holds before the next prompt, so each attempt can
    bind to the file's present content instead of the frozen source count.
    """

    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return len([row for row in _active_branch(rows) if row.get("type") == "message"])


def stage_pi_continuation_session(source: Path, session_dir: Path) -> Path:
    """Copy an immutable source session before Pi appends the rescue branch."""

    source_bytes = source.read_bytes()
    session_dir.mkdir(parents=True, exist_ok=True)
    destination = session_dir / source.name
    if destination.exists():
        raise FileExistsError("historical Pi continuation session already exists")
    shutil.copy2(source, destination)
    if destination.read_bytes() != source_bytes or source.read_bytes() != source_bytes:
        raise RuntimeError("historical Pi continuation staging changed source bytes")
    return destination


__all__ = [
    "FrozenPiContinuationSource",
    "active_session_message_count",
    "load_frozen_pi_continuation_source",
    "normalize_terminal_instruction_boundary",
    "stage_pi_continuation_session",
    "terminal_instructions_are_equivalent",
]
