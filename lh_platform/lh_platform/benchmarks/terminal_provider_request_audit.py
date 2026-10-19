"""Validate the provider-visible Pi prompt and tool declarations."""

from __future__ import annotations

import json
from pathlib import Path


def _tool_name(tool: object) -> str:
    """Read one OpenAI-compatible function/custom or Anthropic tool name."""

    if not isinstance(tool, dict):
        raise ValueError("provider request tool is not an object")
    if "type" not in tool and isinstance(tool.get("name"), str) and "input_schema" in tool:
        # Anthropic Messages tool declaration: {"name", "description", "input_schema"}.
        return tool["name"]
    body = tool.get("function") if tool.get("type") == "function" else tool.get("custom")
    if not isinstance(body, dict) or not isinstance(body.get("name"), str):
        raise ValueError("provider request tool name is missing")
    return body["name"]


def audit_terminal_provider_requests(
    path: Path,
    *,
    registration_enabled: bool,
    registration_phase_required: bool,
) -> dict[str, object]:
    """Recompute stable request visibility from the append-only provider audit."""

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("provider request audit is empty")
    expected_by_attempt: dict[int, int] = {}
    first_tool_names: tuple[str, ...] | None = None
    full_request_count = 0
    for row in rows:
        if row.get("schema_version") != "terminal_provider_request_audit_v1":
            raise ValueError("provider request audit schema differs")
        attempt = row.get("actor_attempt_index")
        sequence = row.get("sequence")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ValueError("provider request Actor attempt is invalid")
        if sequence != expected_by_attempt.get(attempt, 0):
            raise ValueError("provider request sequence is not contiguous")
        expected_by_attempt[attempt] = sequence + 1
        if sequence == 0:
            tools = row.get("tools")
            systems = row.get("system_messages")
            if not isinstance(tools, list) or not isinstance(systems, list) or not systems:
                raise ValueError("first provider request lacks full prompt or tools")
            names = tuple(sorted(_tool_name(tool) for tool in tools))
            if len(set(names)) != len(names):
                raise ValueError("provider request tool names are not unique")
            if first_tool_names is None:
                first_tool_names = names
            elif names != first_tool_names:
                raise ValueError("provider request tools differ across Actor attempts")
            full_request_count += 1
        elif row.get("tools_match_first") is not True or row.get(
            "system_messages_match_first"
        ) is not True:
            raise ValueError("provider request prompt or tools changed within an Actor attempt")
    assert first_tool_names is not None
    required = {"bash", "read", "write"}
    if registration_enabled:
        required.add("register_runtime_check")
    if registration_phase_required:
        required.add("complete_runtime_check_registration")
    missing = sorted(required.difference(first_tool_names))
    if missing:
        raise ValueError(f"provider request lacks required tools: {missing}")
    return {
        "schema_version": "terminal_provider_request_audit_result_v1",
        "request_count": len(rows),
        "actor_attempt_count": len(expected_by_attempt),
        "full_request_count": full_request_count,
        "provider_tool_names": list(first_tool_names),
        "required_tool_names": sorted(required),
        "tools_stable_within_actor_attempt": True,
        "system_messages_stable_within_actor_attempt": True,
    }


__all__ = ["audit_terminal_provider_requests"]
