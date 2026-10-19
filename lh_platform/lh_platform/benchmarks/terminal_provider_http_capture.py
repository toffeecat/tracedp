"""Load observed HTTP bytes without reconstructing missing requests or responses."""

from __future__ import annotations

import base64
import json
from pathlib import Path

SCHEMA = "terminal_provider_http_capture_v1"


def load_terminal_provider_http_capture(path: Path, *, start_offset: int = 0) -> dict[str, object]:
    """Validate event order and expose complete or interrupted transport observations.

    A transport error can be fully recorded even when its response is partial.
    A missing capture close or missing terminal request event remains incomplete.
    Raw response bytes are still content-encoded; consumers must use the saved encoding.
    """
    captures: dict[str, dict] = {}
    requests: list[dict] = []
    messages: list[dict] = []
    with path.open("rb") as handle:
        handle.seek(start_offset)
        content = handle.read().decode("utf-8")
    for line in content.splitlines():
        row = json.loads(line)
        if row["schema_version"] != SCHEMA:
            raise ValueError("unsupported provider HTTP capture schema")
        capture_id = row["capture_id"]
        event = row["event"]
        if event == "capture_started":
            if capture_id in captures or row["sequence"] != 0:
                raise ValueError("duplicate or unordered HTTP capture start")
            captures[capture_id] = {
                "sequence": -1,
                "closed": False,
                "requests": {},
                "valid": True,
                "actor_attempt_index": row["actor_attempt_index"],
            }
        capture = captures[capture_id]
        if capture["closed"] or row["sequence"] != capture["sequence"] + 1:
            raise ValueError("HTTP capture events are out of order")
        if row["actor_attempt_index"] != capture["actor_attempt_index"]:
            raise ValueError("HTTP capture actor attempt changed")
        capture["sequence"] = row["sequence"]
        if event == "capture_started":
            continue
        if event == "parsed_message":
            messages.append(
                {
                    "capture_id": capture_id,
                    "session_id": row["session_id"],
                    "message": row["message"],
                }
            )
            continue
        if event == "capture_finished":
            pending = [
                key for key, value in capture["requests"].items() if value["terminal_event"] is None
            ]
            if row["pending_request_ids"] != pending:
                raise ValueError("HTTP capture close differs from observed pending requests")
            capture["closed"] = True
            capture["valid"] = (
                capture["valid"] and row["dispatcher_still_installed"] is True and not pending
            )
            continue
        request_id = row["request_id"]
        if event == "request_dispatched":
            if request_id != len(capture["requests"]):
                raise ValueError("HTTP request IDs are out of order")
            request = {
                "capture_id": capture_id,
                "request_id": request_id,
                "actor_attempt_index": row["actor_attempt_index"],
                "session_id": row["session_id"],
                "method": row["method"],
                "endpoint": row["endpoint"],
                "starts": 0,
                "request_sent": False,
                "status": None,
                "content_encoding": None,
                "request_body": bytearray(),
                "response_body": bytearray(),
                "terminal_event": None,
            }
            capture["requests"][request_id] = request
            requests.append(request)
            continue
        request = capture["requests"][request_id]
        if request["terminal_event"] is not None:
            raise ValueError("HTTP request has events after its terminal event")
        if event == "request_started":
            request["starts"] += 1
            if request["starts"] > 1:
                capture["valid"] = False  # Undici-internal replay is outside this version's scope.
        elif event == "request_body_chunk":
            if request["starts"] == 0 or request["request_sent"]:
                raise ValueError("HTTP request body is outside its send interval")
            request["request_body"].extend(base64.b64decode(row["bytes_base64"], validate=True))
        elif event == "request_sent":
            if request["request_sent"] or request["starts"] == 0:
                raise ValueError("HTTP request sent marker is duplicated or lacks a start")
            request["request_sent"] = True
        elif event == "response_started":
            if row["status"] >= 200:
                if request["status"] is not None:
                    raise ValueError("HTTP request has multiple final responses")
                request["status"] = row["status"]
                request["content_encoding"] = row["content_encoding"]
        elif event == "response_body_chunk":
            if request["status"] is None:
                raise ValueError("HTTP response body lacks its status")
            request["response_body"].extend(base64.b64decode(row["bytes_base64"], validate=True))
        elif event in {"response_completed", "response_error", "dispatch_error"}:
            if event == "response_completed" and (
                request["status"] is None or not request["request_sent"]
            ):
                raise ValueError("HTTP response completion lacks request or response evidence")
            request["terminal_event"] = event
        else:
            raise ValueError(f"unknown HTTP capture event: {event}")
    for request in requests:
        request["request_body"] = bytes(request["request_body"])
        request["response_body"] = bytes(request["response_body"])
    return {
        "capture_complete": bool(captures)
        and all(c["closed"] and c["valid"] for c in captures.values()),
        "capture_count": len(captures),
        "requests": requests,
        "parsed_messages": messages,
    }


def require_terminal_provider_http_capture(
    path: Path, *, start_offset: int = 0
) -> dict[str, object]:
    """Reject incomplete opt-in recording rather than accepting partial logs as full input."""
    capture = load_terminal_provider_http_capture(path, start_offset=start_offset)
    if not capture["capture_complete"]:
        raise ValueError("provider HTTP capture is incomplete or its dispatcher was replaced")
    return capture
