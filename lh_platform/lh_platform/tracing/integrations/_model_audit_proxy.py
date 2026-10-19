"""Per-response model provenance and decoding control for isolated tracers."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from lh_platform.tracefix_bench.shared_budget import BudgetRefusedError


@dataclass(frozen=True)
class ModelAuditSummaryV1:
    model_calls: int
    response_model_ids: tuple[str, ...]
    reasoning_tokens: int
    reasoning_tokens_reported: bool


class AuditedOpenAIEndpoint:
    """Forward OpenAI-compatible calls while enforcing one model and decoding spec."""

    def __init__(
        self,
        *,
        upstream_base_url: str,
        expected_model_id: str,
        max_tokens: int,
        max_model_calls: int | None = None,
        temperature: float,
        generation_seed: int,
        timeout_seconds: int,
        request_contract: Mapping[str, object] | None = None,
        budget_ledger_path: str | None = None,
        budget_role: str | None = None,
        request_overrides: Mapping[str, object] | None = None,
    ) -> None:
        parsed = urlsplit(upstream_base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("model audit upstream base URL is invalid")
        if not expected_model_id:
            raise ValueError("model audit expected model id is empty")
        if max_tokens < 1 or timeout_seconds < 1:
            raise ValueError("model audit limits must be positive")
        if max_model_calls is not None and max_model_calls < 1:
            raise ValueError("model audit call limit must be positive or null")
        self._upstream_base_url = upstream_base_url.rstrip("/")
        self._base_path = parsed.path.rstrip("/")
        self._expected_model_id = expected_model_id
        self._max_tokens = max_tokens
        self._max_model_calls = max_model_calls
        self._temperature = temperature
        self._generation_seed = generation_seed
        self._timeout_seconds = timeout_seconds
        self._request_contract = (
            None
            if request_contract is None
            else json.loads(json.dumps(dict(request_contract), sort_keys=True))
        )
        # A shared cross-role budget pool (see tracefix_bench.shared_budget): every
        # completion is reserved before dispatch and settled from reported usage.
        if (budget_ledger_path is None) != (budget_role is None):
            raise ValueError("shared budget requires both a ledger path and a role")
        self._budget_ledger = None
        self._budget_role = None
        if budget_ledger_path is not None:
            from lh_platform.tracefix_bench.scorer_contracts import ResourceRole
            from lh_platform.tracefix_bench.shared_budget import FileSharedBudgetLedger

            self._budget_ledger = FileSharedBudgetLedger(Path(budget_ledger_path))
            self._budget_role = ResourceRole(budget_role)
            if max_tokens > self._budget_ledger.budget().per_call_output_tokens:
                raise ValueError("audit max_tokens exceeds the shared per-call output cap")
        self._request_overrides = (
            json.loads(json.dumps(dict(request_overrides))) if request_overrides else {}
        )
        self._request_state = threading.local()
        self._lock = threading.Lock()
        self._model_calls = 0
        self._attempted_model_calls = 0
        self._response_model_ids: list[str] = []
        self._reasoning_tokens: list[int] = []
        self._errors: list[str] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def __enter__(self) -> AuditedOpenAIEndpoint:
        audit = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                audit._forward(self)

            def do_POST(self) -> None:  # noqa: N802
                audit._forward(self)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{port}{self._base_path}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _completion_request(self, path: str) -> bool:
        return urlsplit(path).path.endswith("/chat/completions")

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._errors.append(message)

    def _rewrite_request(self, path: str, body: bytes) -> bytes:
        if not self._completion_request(path):
            return body
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("model audit completion request is not JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("model audit completion request must be an object")
        if payload.get("model") != self._expected_model_id:
            raise ValueError(
                "model audit request model mismatch: "
                f"expected={self._expected_model_id!r} observed={payload.get('model')!r}"
            )
        thinking = payload.get("chat_template_kwargs")
        if thinking is None:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif thinking != {"enable_thinking": False}:
            raise ValueError("model audit thinking contract must be explicitly disabled")
        if self._request_contract is not None:
            contract = self._request_contract
            messages = payload.get("messages")
            if (
                contract.get("schema") != "lh_transition_grounded_preference_request_contract/v1"
                or not isinstance(messages, list)
                or len(messages) != 2
                or messages[0]
                != {
                    "role": "system",
                    "content": contract.get("system_message"),
                }
                or not isinstance(messages[1], dict)
                or messages[1].get("role") != "user"
                or not isinstance(messages[1].get("content"), str)
                or not messages[1]["content"]
                or payload.get("response_format") != contract.get("response_format")
            ):
                raise ValueError("model audit request contract changed")
            extra_body = contract.get("extra_body")
            if not isinstance(extra_body, dict) or any(
                payload.get(key) != value for key, value in extra_body.items()
            ):
                raise ValueError("model audit request extra body changed")
        with self._lock:
            if (
                self._max_model_calls is not None
                and self._attempted_model_calls >= self._max_model_calls
            ):
                raise BudgetRefusedError(
                    f"model-call budget exceeded: max={self._max_model_calls}"
                )
            self._attempted_model_calls += 1
        if self._budget_ledger is not None:
            # Rough pre-dispatch input estimate; the pool is settled from reported usage.
            estimate = len(json.dumps(payload.get("messages", ""), ensure_ascii=False)) // 4
            self._request_state.reservation = self._budget_ledger.reserve(
                self._budget_role, input_tokens=estimate
            )
            self._request_state.started = time.monotonic()
        payload.update(self._request_overrides)
        payload["temperature"] = self._temperature
        payload["seed"] = self._generation_seed
        payload["max_tokens"] = self._max_tokens
        if self._request_contract is not None:
            allowed_fields = {
                "model",
                "messages",
                "response_format",
                "chat_template_kwargs",
                "temperature",
                "seed",
                "max_tokens",
            }
            if set(payload) != allowed_fields:
                raise ValueError("model audit request fields changed")
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

    def _record_response(self, path: str, body: bytes) -> None:
        if not self._completion_request(path):
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("model audit completion response is not JSON") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
            raise ValueError("model audit completion response omitted model id")
        response_model = payload["model"]
        if response_model != self._expected_model_id:
            raise ValueError(
                "model audit response model mismatch: "
                f"expected={self._expected_model_id!r} observed={response_model!r}"
            )
        usage = payload.get("usage")
        self._settle_budget(usage if isinstance(usage, Mapping) else None)
        completion_details = (
            usage.get("completion_tokens_details")
            if isinstance(usage, Mapping)
            else None
        )
        reasoning_tokens = (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, Mapping)
            else None
        )
        thinking_disabled = self._request_overrides.get("thinking") == {"type": "disabled"}
        if reasoning_tokens is None and thinking_disabled:
            # MiniMax omits completion_tokens_details once thinking is disabled; the
            # <think> content guard below still rejects any reasoning that leaks through.
            reasoning_tokens = 0
        if isinstance(reasoning_tokens, bool) or not isinstance(reasoning_tokens, int):
            raise ValueError("model audit response did not report reasoning tokens")
        # Some providers reason inside the content while reporting zero reasoning tokens.
        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                message = choice.get("message") if isinstance(choice, Mapping) else None
                content = message.get("content") if isinstance(message, Mapping) else None
                if isinstance(content, str) and "<think>" in content:
                    raise ValueError("model audit response contains hidden <think> reasoning")
        if reasoning_tokens != 0:
            raise ValueError(
                f"model audit response used reasoning tokens: {reasoning_tokens}"
            )
        with self._lock:
            self._model_calls += 1
            self._response_model_ids.append(response_model)
            self._reasoning_tokens.append(reasoning_tokens)

    def _settle_budget(self, usage: Mapping[str, object] | None) -> None:
        reservation = getattr(self._request_state, "reservation", None)
        if self._budget_ledger is None or reservation is None:
            return
        self._request_state.reservation = None

        def _count(key: str) -> int | None:
            value = usage.get(key) if usage is not None else None
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        self._budget_ledger.settle(
            reservation,
            input_tokens=_count("prompt_tokens"),
            output_tokens=_count("completion_tokens"),
            wall_time_seconds=time.monotonic() - self._request_state.started,
        )

    def _record_upstream_http_error(self, path: str, status: int, response_body: bytes) -> None:
        if not self._completion_request(path):
            return
        # A dispatched request that failed still consumed its reservation.
        self._settle_budget(None)
        detail = " ".join(response_body.decode("utf-8", errors="replace").split())
        if not detail:
            detail = "<empty body>"
        self._record_error(f"upstream HTTP {status}: {detail[:2000]}")

    def _upstream_url(self, path: str) -> str:
        split = urlsplit(path)
        incoming_path = split.path
        if self._base_path and incoming_path.startswith(self._base_path):
            suffix = incoming_path[len(self._base_path) :]
        else:
            suffix = incoming_path
        url = self._upstream_base_url + (suffix if suffix.startswith("/") else f"/{suffix}")
        if split.query:
            url += f"?{split.query}"
        return url

    def _local_budget_refusal(self, handler: BaseHTTPRequestHandler, error: BudgetRefusedError) -> None:
        """Reject a local pool limit with a non-retryable status and no upstream call."""

        self._record_error(f"{type(error).__name__}: {error}")
        payload = json.dumps(
            {
                "error": {
                    "type": "local_budget_exceeded",
                    # Pi's retry classifier treats "out of budget" as non-retryable.
                    "message": f"out of budget: {error}",
                    "not_dispatched": True,
                }
            }
        ).encode()
        handler.send_response(429)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            body = handler.rfile.read(content_length) if content_length else b""
            body = self._rewrite_request(handler.path, body)
            headers = {
                key: value
                for key, value in handler.headers.items()
                if key.casefold() not in {"host", "content-length", "connection", "accept-encoding"}
            }
            request = urllib.request.Request(
                self._upstream_url(handler.path),
                data=body if handler.command == "POST" else None,
                headers=headers,
                method=handler.command,
            )
            try:
                with urllib.request.urlopen(  # noqa: S310
                    request,
                    timeout=self._timeout_seconds,
                ) as response:
                    status = response.status
                    response_body = response.read()
                    content_type = response.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as error:
                status = error.code
                response_body = error.read()
                content_type = error.headers.get("Content-Type", "application/json")
            if 200 <= status < 300:
                self._record_response(handler.path, response_body)
            else:
                self._record_upstream_http_error(handler.path, status, response_body)
            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(response_body)))
            handler.end_headers()
            handler.wfile.write(response_body)
        except BudgetRefusedError as error:
            # Raised inside request rewrite, before urlopen. No provider call was made,
            # and reserve() does not return an id when it refuses.
            self._local_budget_refusal(handler, error)
        except Exception as error:  # the client must fail closed on audit errors
            message = f"{type(error).__name__}: {error}"
            self._record_error(message)
            # Charge an open reservation conservatively; dispatch may have happened.
            self._settle_budget(None)
            payload = json.dumps(
                {"error": {"message": message, "type": "model_audit_error"}}
            ).encode()
            handler.send_response(502)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

    def snapshot(self, *, allow_no_calls: bool = False) -> ModelAuditSummaryV1:
        """Return audited totals; an explicit no-call path must have no dispatch."""
        with self._lock:
            errors = tuple(self._errors)
            calls = self._model_calls
            attempts = self._attempted_model_calls
            model_ids = tuple(sorted(set(self._response_model_ids)))
            reasoning_tokens = tuple(self._reasoning_tokens)
        if errors:
            raise RuntimeError("; ".join(errors))
        if allow_no_calls and calls == 0 and attempts == 0:
            # No response exists, so neither model identity nor provider usage is attested.
            return ModelAuditSummaryV1(0, (), 0, False)
        if calls < 1:
            raise RuntimeError("model audit observed no successful completion responses")
        if model_ids != (self._expected_model_id,):
            raise RuntimeError("model audit did not verify the expected response model")
        if len(reasoning_tokens) != calls:
            raise RuntimeError("model audit reasoning-token coverage is incomplete")
        return ModelAuditSummaryV1(
            model_calls=calls,
            response_model_ids=model_ids,
            reasoning_tokens=sum(reasoning_tokens),
            reasoning_tokens_reported=True,
        )


__all__ = ["AuditedOpenAIEndpoint", "ModelAuditSummaryV1"]
