"""Validate the observable terminal state of one host Pi process."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _read_token_count(value: object) -> int:
    """Accept only provider token counts that are exact non-negative integers."""

    if type(value) is not int or value < 0:
        raise ValueError("host Pi token usage must be a non-negative integer")
    return value


@dataclass(frozen=True)
class PiUsageEvent:
    """One provider usage receipt, retained for tier-aware cost accounting."""

    input_tokens: int
    output_tokens: int
    cache_tokens: int


@dataclass(frozen=True)
class PiToolCall:
    """One assistant-authored tool request from the durable Pi stream."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    event_sequence: int


@dataclass(frozen=True)
class PiToolExecution:
    """One Pi tool completion, keyed to the assistant request identity."""

    tool_call_id: str
    tool_name: str
    is_error: bool
    result: object
    event_sequence: int


@dataclass(frozen=True)
class PiPersistentSessionToolReceipt:
    """Tool evidence from Pi's active append-only session branch."""

    session_id: str
    tool_calls: tuple[PiToolCall, ...]
    tool_executions: tuple[PiToolExecution, ...]
    uncompleted_tool_call_ids: frozenset[str]


@dataclass(frozen=True)
class PiRpcResponse:
    """One correlated control response emitted by Pi's documented RPC mode."""

    request_id: str
    command: str
    success: bool
    data: object
    event_sequence: int


@dataclass
class PiStreamReceipt:
    """Accumulate model identity, usage, and fatal provider events from Pi JSONL."""

    expected_provider: str
    expected_model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    reasoning_tokens: int = 0
    actionable_assistant_messages: int = 0
    # Truncation is reported separately from failure: a `length` stop is a normal
    # terminal stop that proves the provider hit the output budget.
    last_assistant_stop_reason: str | None = None
    truncated_assistant_messages: int = 0
    failure: str | None = None
    provider_failure_pending: bool = False
    provider_failure_message: str | None = None
    provider_retry_count: int = 0
    provider_recovered_retry_count: int = 0
    provider_exhausted_retry_count: int = 0
    provider_error_messages: list[str] = field(default_factory=list)
    provider_error_kinds: list[str] = field(default_factory=list)
    provider_http_statuses: list[int] = field(default_factory=list)
    usage_events: list[PiUsageEvent] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    tool_calls: list[PiToolCall] = field(default_factory=list)
    tool_executions: list[PiToolExecution] = field(default_factory=list)
    session_event_sequences: list[int] = field(default_factory=list)
    assistant_event_sequences: list[int] = field(default_factory=list)
    actionable_assistant_event_sequences: list[int] = field(default_factory=list)
    rpc_responses: list[PiRpcResponse] = field(default_factory=list)
    agent_settled_sequences: list[int] = field(default_factory=list)
    aborted_assistant_sequences: list[int] = field(default_factory=list)
    observed_event_count: int = 0

    def is_terminal_uncompleted_tool_call(self, tool_call_id: str) -> bool:
        """Allow a missing completion only in the final assistant tool-call batch."""

        completed = {item.tool_call_id for item in self.tool_executions}
        if tool_call_id in completed or not self.assistant_event_sequences:
            return False
        final_assistant_sequence = self.assistant_event_sequences[-1]
        return any(
            item.tool_call_id == tool_call_id
            and item.event_sequence == final_assistant_sequence
            for item in self.tool_calls
        )

    def _record_provider_error(self, message: str) -> None:
        """Retain provider diagnostics even when Pi later recovers the request."""
        if self.provider_error_messages and self.provider_error_messages[-1] == message:
            return
        self.provider_error_messages.append(message)
        lowered = message.lower()
        if "repetition_truncation" in lowered:
            kind = "model_truncation"
        elif "rate limit" in lowered or re.search(r"\b429\b", lowered):
            kind = "rate_limit"
        elif "timed out" in lowered or "timeout" in lowered:
            kind = "timeout"
        elif "connection error" in lowered:
            kind = "connection"
        elif "service unavailable" in lowered or "endpoint unavailable" in lowered:
            kind = "service_unavailable"
        else:
            kind = "other"
        self.provider_error_kinds.append(kind)
        for match in re.findall(r"(?<!\d)([45]\d\d)(?!\d)", message):
            status = int(match)
            if status not in self.provider_http_statuses:
                self.provider_http_statuses.append(status)

    def observe(self, event: dict[str, Any]) -> None:
        """Consume one Pi event while preserving the first fatal runtime failure."""
        event_sequence = self.observed_event_count
        self.observed_event_count += 1
        if event.get("type") == "response":
            request_id = event.get("id")
            command = event.get("command")
            success = event.get("success")
            if (
                not isinstance(request_id, str)
                or not request_id
                or not isinstance(command, str)
                or not command
                or not isinstance(success, bool)
            ):
                self.failure = self.failure or "host Pi RPC response is invalid"
            else:
                self.rpc_responses.append(
                    PiRpcResponse(
                        request_id=request_id,
                        command=command,
                        success=success,
                        data=event.get("data"),
                        event_sequence=event_sequence,
                    )
                )
            return
        if event.get("type") == "agent_settled":
            self.agent_settled_sequences.append(event_sequence)
            return
        if event.get("type") == "session":
            session_id = event.get("id")
            if not isinstance(session_id, str) or not session_id:
                self.failure = self.failure or "host Pi session identity is missing"
            else:
                self.session_ids.append(session_id)
                self.session_event_sequences.append(event_sequence)
            return
        if event.get("type") == "tool_execution_end":
            tool_call_id = event.get("toolCallId")
            tool_name = event.get("toolName")
            is_error = event.get("isError")
            if (
                not isinstance(tool_call_id, str)
                or not tool_call_id
                or not isinstance(tool_name, str)
                or not tool_name
                or not isinstance(is_error, bool)
            ):
                self.failure = self.failure or "host Pi tool execution receipt is invalid"
            else:
                self.tool_executions.append(
                    PiToolExecution(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        is_error=is_error,
                        result=event.get("result"),
                        event_sequence=event_sequence,
                    )
                )
            return
        if event.get("type") == "auto_retry_start":
            self.provider_retry_count += 1
            detail = event.get("errorMessage")
            if detail:
                self._record_provider_error(str(detail))
            return
        if event.get("type") == "auto_retry_end":
            success = event.get("success")
            if success is False:
                detail = str(
                    event.get("finalError") or self.provider_failure_message or "unknown error"
                )
                self._record_provider_error(detail)
                self.failure = self.failure or f"host Pi exhausted provider retries: {detail}"
                self.provider_exhausted_retry_count += 1
                self.provider_failure_pending = False
                self.provider_failure_message = None
            elif success is True:
                self.provider_recovered_retry_count += 1
                self.provider_failure_pending = False
                self.provider_failure_message = None
            return

        message = event.get("message")
        if event.get("type") != "message_end" or not isinstance(message, dict):
            return
        if message.get("role") != "assistant":
            return
        self.assistant_event_sequences.append(event_sequence)
        stop_reason = message.get("stopReason")
        if isinstance(stop_reason, str) and stop_reason:
            self.last_assistant_stop_reason = stop_reason
            if stop_reason == "length":
                # A length stop means the provider cut the response at the frozen
                # output budget.  It is a normal terminal stop, not a failure, but
                # it must be counted so a run report can separate "the actor could
                # not finish" from "the method chose wrong".
                self.truncated_assistant_messages += 1
        if stop_reason == "aborted":
            self.aborted_assistant_sequences.append(event_sequence)
        if message.get("provider") != self.expected_provider:
            self.failure = self.failure or "host Pi model provider changed"
        if message.get("model") != self.expected_model:
            self.failure = self.failure or "host Pi model identity changed"
        if message.get("stopReason") == "error" or message.get("errorMessage"):
            # Pi may persist a failed attempt before its own retry succeeds.
            self.provider_failure_pending = True
            self.provider_failure_message = str(
                message.get("errorMessage") or message.get("rawStopReason") or "unknown error"
            )
            self._record_provider_error(self.provider_failure_message)

        usage = message.get("usage") or {}
        if not isinstance(usage, dict):
            self.failure = self.failure or "host Pi token usage is invalid"
            return
        try:
            reasoning = _read_token_count(usage.get("reasoning", 0))
            input_tokens = _read_token_count(usage.get("input", 0))
            output_tokens = _read_token_count(usage.get("output", 0))
            cache_tokens = _read_token_count(usage.get("cacheRead", 0))
        except ValueError as error:
            self.failure = self.failure or str(error)
            return
        if reasoning != 0:
            self.failure = self.failure or "host Pi emitted reasoning tokens with thinking off"
        self.reasoning_tokens += reasoning
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_tokens += cache_tokens
        self.usage_events.append(
            PiUsageEvent(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_tokens=cache_tokens,
            )
        )
        # Pi retries error responses and never executes their partial tool calls.
        if message.get("stopReason") in {"error", "aborted"}:
            return
        content = message.get("content")
        if isinstance(content, list) and content:
            self.actionable_assistant_messages += 1
            self.actionable_assistant_event_sequences.append(event_sequence)
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "toolCall":
                    continue
                tool_call_id = item.get("id")
                tool_name = item.get("name")
                arguments = item.get("arguments")
                if (
                    not isinstance(tool_call_id, str)
                    or not tool_call_id
                    or not isinstance(tool_name, str)
                    or not tool_name
                    or not isinstance(arguments, dict)
                ):
                    self.failure = self.failure or "host Pi tool call is invalid"
                    continue
                self.tool_calls.append(
                    PiToolCall(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=dict(arguments),
                        event_sequence=event_sequence,
                    )
                )

    def controlled_abort_is_settled(
        self, *, prompt_request_id: str, abort_request_id: str
    ) -> bool:
        """Return true only after Pi accepted both controls and became idle."""

        by_id = {item.request_id: item for item in self.rpc_responses}
        prompt = by_id.get(prompt_request_id)
        abort = by_id.get(abort_request_id)
        return bool(
            prompt is not None
            and prompt.command == "prompt"
            and prompt.success
            and abort is not None
            and abort.command == "abort"
            and abort.success
            and self.aborted_assistant_sequences
            and self.agent_settled_sequences
            and self.agent_settled_sequences[-1]
            > self.aborted_assistant_sequences[-1]
        )

    def require_controlled_abort(
        self, *, prompt_request_id: str, abort_request_id: str
    ) -> None:
        """Reject a submission without complete Pi abort and settled receipts."""

        if not self.controlled_abort_is_settled(
            prompt_request_id=prompt_request_id,
            abort_request_id=abort_request_id,
        ):
            raise RuntimeError("host Pi controlled abort receipt is incomplete")

    def require_rpc_session_identity(
        self,
        *,
        request_id: str,
        expected_session_id: str,
        expected_message_count: int | None = None,
        expected_session_file: Path | None = None,
    ) -> None:
        """Bind the live RPC process to Pi's documented pre-prompt session state."""

        response = next(
            (item for item in self.rpc_responses if item.request_id == request_id),
            None,
        )
        data = response.data if response is not None else None
        observed_session_file = data.get("sessionFile") if isinstance(data, dict) else None
        session_file_matches = expected_session_file is None or (
            isinstance(observed_session_file, str)
            and Path(observed_session_file).resolve() == expected_session_file.resolve()
        )
        if (
            response is None
            or response.command != "get_state"
            or not response.success
            or not isinstance(data, dict)
            or data.get("sessionId") != expected_session_id
            or (
                expected_message_count is not None
                and data.get("messageCount") != expected_message_count
            )
            or not session_file_matches
        ):
            raise RuntimeError("host Pi RPC session identity differs")

    def require_valid(
        self, *, return_code: int, controlled_abort_verified: bool = False
    ) -> None:
        """Reject a process that did not produce one usable assistant action or answer."""
        if return_code != 0:
            raise RuntimeError(f"host Pi exited with code {return_code}")
        if self.failure:
            raise RuntimeError(self.failure)
        if self.provider_failure_pending and not (
            controlled_abort_verified and self.aborted_assistant_sequences
        ):
            raise RuntimeError(f"host Pi provider response failed: {self.provider_failure_message}")
        if self.actionable_assistant_messages == 0:
            raise RuntimeError("host Pi produced no actionable assistant message")


def load_pi_stream_receipt(
    path: Path,
    *,
    expected_provider: str,
    expected_model: str,
) -> PiStreamReceipt:
    """Replay a persisted Pi JSONL stream through the live runtime validator."""
    receipt = PiStreamReceipt(
        expected_provider=expected_provider,
        expected_model=expected_model,
    )
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            receipt.observe(event)
    return receipt


def load_pi_persistent_session_tool_receipt(
    path: Path,
    *,
    expected_session_id: str,
    expected_provider: str,
    expected_model: str,
) -> PiPersistentSessionToolReceipt:
    """Replay the active branch using Pi 0.84.1 SessionManager tree semantics."""

    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"host Pi persistent session JSON is invalid at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError("host Pi persistent session row is not an object")
        rows.append(row)
    headers = [row for row in rows if row.get("type") == "session"]
    if (
        not rows
        or rows[0].get("type") != "session"
        or len(headers) != 1
        or headers[0].get("id") != expected_session_id
        or headers[0].get("version") != 3
    ):
        raise ValueError("host Pi persistent session identity differs")

    entries = [row for row in rows if row.get("type") != "session"]
    by_id: dict[str, dict[str, Any]] = {}
    for row in entries:
        entry_id = row.get("id")
        parent_id = row.get("parentId")
        if (
            not isinstance(entry_id, str)
            or not entry_id
            or entry_id in by_id
            or (parent_id is not None and parent_id not in by_id)
        ):
            raise ValueError("host Pi persistent session tree is invalid")
        by_id[entry_id] = row

    # SessionManager sets the last appended entry as the active leaf after reload.
    branch: list[dict[str, Any]] = []
    current = entries[-1] if entries else None
    while current is not None:
        branch.append(current)
        parent_id = current.get("parentId")
        current = by_id.get(parent_id) if isinstance(parent_id, str) else None
    branch.reverse()

    calls: list[PiToolCall] = []
    executions: list[PiToolExecution] = []
    call_ids: set[str] = set()
    call_names: dict[str, str] = {}
    execution_ids: set[str] = set()
    final_assistant_call_ids: set[str] = set()
    for event_sequence, row in enumerate(branch):
        message = row.get("message")
        if row.get("type") != "message" or not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            # Any later assistant message supersedes the preceding tool-call batch.
            final_assistant_call_ids = set()
            if (
                message.get("provider") != expected_provider
                or message.get("model") != expected_model
            ):
                raise ValueError("host Pi persistent session model identity differs")
            if message.get("stopReason") in {"error", "aborted"}:
                continue
            assistant_call_ids: set[str] = set()
            content = message.get("content")
            if not isinstance(content, list):
                raise ValueError("host Pi persistent assistant content is invalid")
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "toolCall":
                    continue
                tool_call_id = item.get("id")
                tool_name = item.get("name")
                arguments = item.get("arguments")
                if (
                    not isinstance(tool_call_id, str)
                    or not tool_call_id
                    or tool_call_id in call_ids
                    or not isinstance(tool_name, str)
                    or not tool_name
                    or not isinstance(arguments, dict)
                ):
                    raise ValueError("host Pi persistent tool call is invalid")
                call_ids.add(tool_call_id)
                call_names[tool_call_id] = tool_name
                assistant_call_ids.add(tool_call_id)
                calls.append(
                    PiToolCall(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=dict(arguments),
                        event_sequence=event_sequence,
                    )
                )
            final_assistant_call_ids = assistant_call_ids
        elif role == "toolResult":
            tool_call_id = message.get("toolCallId")
            tool_name = message.get("toolName")
            is_error = message.get("isError")
            content = message.get("content")
            details = message.get("details", {})
            if (
                not isinstance(tool_call_id, str)
                or tool_call_id not in call_ids
                or tool_call_id in execution_ids
                or not isinstance(tool_name, str)
                or not tool_name
                or tool_name != call_names.get(tool_call_id)
                or not isinstance(is_error, bool)
                or not isinstance(content, list)
                or not isinstance(details, dict)
            ):
                raise ValueError("host Pi persistent tool result is invalid")
            execution_ids.add(tool_call_id)
            executions.append(
                PiToolExecution(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    is_error=is_error,
                    result={"content": content, "details": details},
                    event_sequence=event_sequence,
                )
            )

    uncompleted = call_ids - execution_ids
    if uncompleted and not uncompleted.issubset(final_assistant_call_ids):
        raise ValueError("host Pi persistent session has a non-terminal incomplete tool call")
    return PiPersistentSessionToolReceipt(
        session_id=expected_session_id,
        tool_calls=tuple(calls),
        tool_executions=tuple(executions),
        uncompleted_tool_call_ids=frozenset(uncompleted),
    )


__all__ = [
    "PiPersistentSessionToolReceipt",
    "PiRpcResponse",
    "PiStreamReceipt",
    "PiToolCall",
    "PiToolExecution",
    "PiUsageEvent",
    "load_pi_persistent_session_tool_receipt",
    "load_pi_stream_receipt",
]
