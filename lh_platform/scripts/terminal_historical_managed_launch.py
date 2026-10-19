"""Shared launch helpers for the historical managed circuit-fibsqrt arms.

Both the zero-model probe and the paid arm must derive the replay prefix, the
source session, and the on-disk path spelling the exact same way, otherwise the
two scripts could silently disagree about which failure state they manage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lh_platform.benchmarks.terminal_pi_invocation import is_neutral_actor_cwd_of  # noqa: E402
from lh_platform.benchmarks.terminal_pi_session_continuation import (
    load_frozen_pi_continuation_source,
)
from lh_platform.benchmarks.terminal_replay_population import (
    load_frozen_terminal_replay_calls,
)

DEFAULT_PROBE_COMMANDS = (
    "ls -la /app/",
    "cd /app && ./sim 208",
    "cd /app && wc -l gates.txt",
)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject any other top-level shape."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def frozen_source_cwd(session: Path) -> Path:
    """Return the Actor-visible working directory exactly as Pi persisted it.

    The session header is the only safe source of truth for the spelling, and the
    session file must still live at `<cwd>/pi/sessions/<name>.jsonl`.
    """

    rows = [
        json.loads(line)
        for line in session.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    headers = [row for row in rows if row.get("type") == "session"]
    if len(headers) != 1 or not isinstance(headers[0].get("cwd"), str):
        raise ValueError("frozen Pi session header is invalid")
    header_cwd = Path(str(headers[0]["cwd"]))
    grandparent = session.parent.parent.parent
    if not (header_cwd.is_dir() and grandparent.is_dir()):
        raise ValueError("frozen Pi session working directory is missing")
    # A source collected with neutral_actor_cwd records a hash-named directory that
    # is recomputed from the trial log directory instead of the trial directory itself.
    if not (
        os.path.samefile(header_cwd, grandparent)
        or is_neutral_actor_cwd_of(header_cwd, grandparent)
    ):
        raise ValueError("frozen Pi session does not live under its persisted cwd")
    return header_cwd


def _source_tool_result_texts(session_path: Path) -> dict[str, str]:
    """Map each source tool call id to its recorded result text (verbatim)."""

    texts: dict[str, str] = {}
    for line in session_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        message = row.get("message")
        if row.get("type") != "message" or not isinstance(message, dict):
            continue
        if message.get("role") != "toolResult":
            continue
        call_id = str(message.get("toolCallId", ""))
        if not call_id or call_id in texts:
            continue
        content = message.get("content")
        pieces: list[str] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    pieces.append(block["text"])
        texts[call_id] = "".join(pieces)
    return texts


def _resolve_jit_memory(arm: dict[str, object]) -> Path | None:
    """Resolve a frozen JIT memory file declared by an arm and require it to exist."""

    declared = arm.get("jit_memory_path")
    if declared is None:
        return None
    path = Path(str(declared)).expanduser().resolve()
    if path.suffix != ".json":
        raise ValueError("declared JIT memory must be a JSON file")
    if not path.is_file():
        raise FileNotFoundError(f"declared JIT memory is missing: {path}")
    return path


def prefix_from_frozen_source(
    *,
    source_session: Path,
    source_session_cwd: Path,
    task_instruction: str,
    provider: str,
    model: str,
    thinking: str,
    expected_terminal_stop_reason: str = "length",
) -> tuple[list[dict[str, object]], int, str]:
    """Derive the replay prefix from the frozen Pi source instead of a copy.

    The per-call `environment_step_executed` flag is not a formatting choice.  In
    the historical session the Actor submitted several terminal calls as one
    parallel batch, and the tool interface rejected the duplicates with
    `parallel terminal tool calls are not supported` before they reached the
    environment.  Marking those calls as executed would replay them for real and
    silently change both the failure state and the error parity.  The public
    observation log is therefore the only authoritative source for that flag.
    """

    declared = frozen_source_cwd(source_session)
    if not os.path.samefile(declared, source_session_cwd):
        raise ValueError("requested source cwd is not the persisted Pi cwd")
    source = load_frozen_pi_continuation_source(
        source_session,
        expected_instruction=task_instruction,
        expected_provider=provider,
        expected_model=model,
        expected_thinking=thinking,
        expected_terminal_stop_reason=expected_terminal_stop_reason,
        expected_cwd=declared,
    )
    observer = source_session.parent.parent.parent / "terminal-public-state.jsonl"
    if not observer.is_file():
        raise ValueError("frozen Pi source lacks its public observation log")
    replay_calls = load_frozen_terminal_replay_calls(
        session_path=source_session,
        observer_path=observer,
    )
    if len(replay_calls) != len(source.tool_receipt.tool_calls):
        raise ValueError("frozen source replay calls do not cover the Pi tool calls")
    source_error_texts = _source_tool_result_texts(source_session)
    prefix: list[dict[str, object]] = []
    for call in replay_calls:
        prefix.append(
            {
                "source_tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "arguments": dict(call.arguments),
                "environment_step_executed": call.environment_step_executed,
                "source_is_error": call.source_is_error,
                # Raw source result text: the replay parses it only to classify
                # unreproducible source-side timing conditions (timeouts), never to
                # change the replayed commands.
                "source_error_text": source_error_texts.get(call.tool_call_id),
            }
        )
    return prefix, source.message_count, source.session_id
