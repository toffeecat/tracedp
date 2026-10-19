"""Byte-transparent provider proxy that charges every model call to a shared budget pool.

Point an actor's provider ``baseUrl`` at this proxy.  Each POST is reserved in the
shared cross-role ledger before it is forwarded; a refused reservation returns HTTP 429
without contacting the provider.  Responses are streamed back unchanged.  Usage is
settled from the response: OpenAI JSON ``usage`` or Anthropic SSE ``message_start`` /
``message_delta`` usage.  When the response carries no usage, the call is charged at its
reservation and the ledger marks metering incomplete.

The proxy does not rewrite requests, so the caller must cap output tokens itself at or
below the pool's per-call cap; a response that reports more output is a contract
violation recorded by the ledger.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from lh_platform.tracefix_bench.scorer_contracts import ResourceRole
from lh_platform.tracefix_bench.shared_budget import BudgetRefusedError, FileSharedBudgetLedger

_HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}


def _usage_from_json(body: bytes) -> tuple[int | None, int | None]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    return (
        prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None,
        completion if isinstance(completion, int) and not isinstance(completion, bool) else None,
    )


def _usage_from_sse(body: bytes) -> tuple[int | None, int | None]:
    """Anthropic streams input usage in message_start and output usage in message_delta."""

    input_tokens = output_tokens = None
    for line in body.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        usage = event.get("usage")
        if event.get("type") == "message_start" and isinstance(event.get("message"), dict):
            usage = event["message"].get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("input_tokens"), int):
                input_tokens = usage["input_tokens"]
            if isinstance(usage.get("output_tokens"), int):
                output_tokens = usage["output_tokens"]
            # OpenAI-compatible streams put usage on the final chunk.
            if isinstance(usage.get("prompt_tokens"), int):
                input_tokens = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                output_tokens = usage["completion_tokens"]
    return input_tokens, output_tokens


def _never_sent(error: urllib.error.URLError) -> bool:
    """A failure before any byte reached the provider: tunnel refused, connection refused, DNS.

    Timeouts and resets after connecting stay at the conservative reservation charge,
    because the provider may already have generated tokens.
    """

    reason = error.reason
    return (
        isinstance(reason, (ConnectionRefusedError, socket.gaierror))
        or (isinstance(reason, OSError) and str(reason).startswith("Tunnel connection failed"))
    )


class SharedBudgetProviderProxy:
    """Forward provider calls for one role while drawing from one shared pool."""

    def __init__(
        self,
        *,
        upstream_base_url: str,
        ledger_path: Path,
        role: ResourceRole,
        timeout_seconds: int = 960,
    ) -> None:
        parsed = urlsplit(upstream_base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("budget proxy upstream base URL is invalid")
        self._upstream = upstream_base_url.rstrip("/")
        self._base_path = parsed.path.rstrip("/")
        self._ledger = FileSharedBudgetLedger(Path(ledger_path))
        self._role = role
        self._timeout = timeout_seconds
        self.refusals = 0
        # Upstream error responses (529 overloaded, 429 rate limit) that carried no usage.
        self.upstream_errors = 0
        self.undelivered = 0
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def __enter__(self) -> SharedBudgetProviderProxy:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                owner._forward(self)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}{self._base_path}"
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _upstream_url(self, path: str) -> str:
        suffix = path[len(self._base_path) :] if path.startswith(self._base_path) else path
        return self._upstream + suffix

    def _send(self, handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length) if length else b""
        try:
            reservation = self._ledger.reserve(self._role, input_tokens=len(body) // 4)
        except BudgetRefusedError as error:
            with self._lock:
                self.refusals += 1
            payload = json.dumps(
                {
                    "error": {
                        "type": "shared_budget_exhausted",
                        # Pi retries any 429 unless the text says "out of budget".
                        "message": f"out of budget: {error}",
                        "not_dispatched": True,
                    }
                }
            ).encode()
            self._send(handler, 429, payload, "application/json")
            return
        started = time.monotonic()
        headers = {k: v for k, v in handler.headers.items() if k.casefold() not in _HOP_HEADERS}
        request = urllib.request.Request(
            self._upstream_url(handler.path), data=body, headers=headers, method="POST"
        )
        response_body = b""
        content_type = "application/json"
        status = 502
        undelivered = False
        try:
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                    status = response.status
                    content_type = response.headers.get("Content-Type", content_type)
                    response_body = response.read()
            except urllib.error.HTTPError as error:
                status = error.code
                content_type = error.headers.get("Content-Type", content_type)
                response_body = error.read()
            except urllib.error.URLError as error:
                if not _never_sent(error):
                    raise
                undelivered = True
                response_body = json.dumps(
                    {"error": {"type": "upstream_unreachable", "message": f"request not delivered: {error.reason}"}}
                ).encode()
        finally:
            if "event-stream" in content_type:
                used_input, used_output = _usage_from_sse(response_body)
            else:
                used_input, used_output = _usage_from_json(response_body)
            if undelivered:
                # The local HTTP proxy refused CONNECT (503), so the request never left the host.
                used_input = used_output = 0
                with self._lock:
                    self.undelivered += 1
            elif status >= 400 and used_input is None and used_output is None and response_body:
                # The provider answered with an error and generated nothing. A transport failure
                # without a response stays at the conservative reservation charge.
                used_input = used_output = 0
                with self._lock:
                    self.upstream_errors += 1
            self._ledger.settle(
                reservation,
                input_tokens=used_input,
                output_tokens=used_output,
                wall_time_seconds=time.monotonic() - started,
            )
        self._send(handler, status, response_body, content_type)


__all__ = ["SharedBudgetProviderProxy"]
