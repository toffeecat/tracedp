"""Strict remote boundary for ToolBench's upstream ``rapidapi_wrapper``.

This module implements the narrow HTTP protocol in
``toolbench/inference/Downstream_tasks/rapidapi.py``.  It intentionally does
not import or execute ToolBench's local ``server.py`` because that path uses
``exec``/``eval`` over tool definitions and is neither safe nor an auditable
replacement for the original remote environment.

Calling this protocol proves only that the agent interacted with the
RapidAPI-style environment.  It is not ToolEval and it never emits a semantic
ToolBench success verdict.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from http.client import HTTPException, HTTPSConnection
from typing import Any
from urllib.parse import urlsplit

from lh_platform.core.artifact_redaction import redact_text
from lh_platform.core.evidence_contract import canonical_sha256

# The upstream service URL is supplied by the operator; there is no default endpoint.
UPSTREAM_RAPIDAPI_SERVICE_URL = ""
UPSTREAM_REQUEST_TIMEOUT_S = 15.0
UPSTREAM_RATE_LIMIT_DELAY_S = 2.0
UPSTREAM_RATE_LIMIT_RECOVERY_DELAY_S = 10.0
UPSTREAM_MAX_OBSERVATION_LENGTH = 1024
# This is a security invariant, not an experiment tuning knob.  A config may
# reduce the limit for a constrained run, but must never cause an untrusted
# proxy response to allocate more than this amount in the controller process.
UPSTREAM_MAX_RESPONSE_HARD_CAP_BYTES = 1_048_576
UPSTREAM_MAX_RESPONSE_BYTES = UPSTREAM_MAX_RESPONSE_HARD_CAP_BYTES
REMOTE_RECEIPT_SCHEMA = "lh_toolbench_rapidapi_execution_receipt/v1"
_VALID_OBSERVATION_COMPRESS_METHODS = frozenset({"truncate", "filter", "random"})


class ToolBenchRemoteProtocolError(RuntimeError):
    """Raised when a requested remote ToolBench protocol run is not admissible."""


class _PinnedTLSFailure(RuntimeError):
    """A TLS identity check failed before an authenticated HTTP request."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class ToolBenchRapidAPIInvocation:
    """The source fields used by upstream ``rapidapi_wrapper._step``."""

    function_name: str
    category: str
    tool_name: str
    api_name: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ToolBenchRapidAPIInvocation:
        required = ("function_name", "category", "tool_name", "api_name")
        values = {name: str(raw.get(name) or "").strip() for name in required}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ToolBenchRemoteProtocolError(
                "ToolBench invocation contract is incomplete: " + ", ".join(missing)
            )
        return cls(**values)

    def public_dict(self) -> dict[str, str]:
        return {
            "function_name": self.function_name,
            "category": self.category,
            "tool_name": self.tool_name,
            "api_name": self.api_name,
        }


@dataclass(frozen=True)
class ToolBenchRemoteExecution:
    """One protocol response plus a secret-free receipt for trajectory evidence."""

    observation: str
    status_code: int
    is_error: bool
    receipt: dict[str, Any]


class ToolBenchRemoteClient:
    """Execute the upstream remote RapidAPI request contract without local eval.

    The authentication key is deliberately process-memory-only.  Every public
    receipt uses a payload with ``toolbench_key`` removed and a one-way service
    URL digest so artifacts cannot leak credentials through headers or URLs.
    The client requires an exact DER leaf-certificate pin.  Benchmark code
    must additionally admit the endpoint through a repository-reviewed proxy
    identity lock before constructing it for a real run.  Requiring both
    prevents this public low-level API from becoming an unpinned credential
    sender outside the benchmark adapter.
    """

    def __init__(
        self,
        *,
        toolbench_key: str,
        expected_tls_certificate_sha256: str,
        service_url: str = UPSTREAM_RAPIDAPI_SERVICE_URL,
        request_timeout_s: float = UPSTREAM_REQUEST_TIMEOUT_S,
        rate_limit_delay_s: float = UPSTREAM_RATE_LIMIT_DELAY_S,
        rate_limit_recovery_delay_s: float = UPSTREAM_RATE_LIMIT_RECOVERY_DELAY_S,
        max_observation_length: int = UPSTREAM_MAX_OBSERVATION_LENGTH,
        max_response_bytes: int = UPSTREAM_MAX_RESPONSE_BYTES,
        observ_compress_method: str = "truncate",
    ) -> None:
        key = str(toolbench_key or "").strip()
        if not key:
            raise ToolBenchRemoteProtocolError(
                "TOOLBENCH_KEY is required for upstream_remote execution"
            )
        endpoint = str(service_url or "").strip()
        parsed_endpoint = urlsplit(endpoint)
        if (
            parsed_endpoint.scheme != "https"
            or not parsed_endpoint.hostname
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
            or bool(parsed_endpoint.fragment)
        ):
            raise ToolBenchRemoteProtocolError(
                "ToolBench upstream_remote requires an explicit audited HTTPS service_url; "
                "the historical plaintext HTTP endpoint is intentionally blocked"
            )
        try:
            _ = parsed_endpoint.port
        except ValueError as error:
            raise ToolBenchRemoteProtocolError(
                "ToolBench upstream_remote service_url has an invalid port"
            ) from error
        try:
            timeout = float(request_timeout_s)
            initial_delay = float(rate_limit_delay_s)
            recovery_delay = float(rate_limit_recovery_delay_s)
            observation_limit = int(max_observation_length)
            response_limit = int(max_response_bytes)
        except (TypeError, ValueError) as error:
            raise ToolBenchRemoteProtocolError(
                "ToolBench remote timing/observation options are invalid"
            ) from error
        if (
            timeout <= 0
            or initial_delay < 0
            or recovery_delay < 0
            or observation_limit <= 0
            or response_limit <= 0
        ):
            raise ToolBenchRemoteProtocolError(
                "ToolBench remote timing/observation options are out of range"
            )
        if response_limit > UPSTREAM_MAX_RESPONSE_HARD_CAP_BYTES:
            raise ToolBenchRemoteProtocolError(
                "max_response_bytes exceeds the immutable ToolBench remote hard cap "
                f"({UPSTREAM_MAX_RESPONSE_HARD_CAP_BYTES} bytes)"
            )
        compress_method = str(observ_compress_method or "").strip()
        if compress_method not in _VALID_OBSERVATION_COMPRESS_METHODS:
            choices = ", ".join(sorted(_VALID_OBSERVATION_COMPRESS_METHODS))
            raise ToolBenchRemoteProtocolError(
                f"observ_compress_method must be one of {choices}; got {compress_method!r}"
            )
        certificate_sha256 = str(expected_tls_certificate_sha256 or "").strip().lower()
        if not _is_sha256(certificate_sha256):
            raise ToolBenchRemoteProtocolError(
                "expected_tls_certificate_sha256 must be a 64-character SHA-256 TLS pin"
            )
        self._toolbench_key = key
        self._service_url = endpoint
        self._request_timeout_s = timeout
        self._rate_limit_delay_s = initial_delay
        self._rate_limit_recovery_delay_s = recovery_delay
        self._max_observation_length = observation_limit
        self._max_response_bytes = response_limit
        self._observ_compress_method = compress_method
        self._expected_tls_certificate_sha256 = certificate_sha256
        self._service_url_sha256 = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()

    @property
    def environment_fingerprint(self) -> str:
        return (
            "toolbench_rapidapi_wrapper_protocol/v1;"
            f"service_url_sha256={self._service_url_sha256};"
            f"tls_certificate_sha256={self._expected_tls_certificate_sha256};"
            f"observ_compress_method={self._observ_compress_method};"
            f"max_observation_length={self._max_observation_length};"
            f"max_response_bytes={self._max_response_bytes}"
        )

    def execute(
        self,
        invocation: ToolBenchRapidAPIInvocation,
        tool_input: str,
    ) -> ToolBenchRemoteExecution:
        """Call the original remote request protocol and return its status code.

        The payload and status mapping intentionally follow the current
        upstream ``rapidapi_wrapper._step`` implementation.  Transport or
        malformed-response errors become status 12 so the unified runner can
        record an observable failed interaction rather than silently falling
        back to a synthetic acknowledgement.
        """

        input_text = str(tool_input)
        private_payload = {
            "category": invocation.category,
            "tool_name": invocation.tool_name,
            "api_name": invocation.api_name,
            "tool_input": input_text,
            "strip": self._observ_compress_method,
            "toolbench_key": self._toolbench_key,
        }
        public_payload = {
            key: value for key, value in private_payload.items() if key != "toolbench_key"
        }
        if self._rate_limit_delay_s:
            time.sleep(self._rate_limit_delay_s)
        try:
            # ``HTTPSConnection`` gives us one connection on which we can
            # complete the TLS handshake and verify the DER pin *before*
            # serializing an authenticated HTTP request.  It also does not
            # follow redirects, so credentials cannot be forwarded to a
            # second, unpinned host.
            http_status, raw_body = self._post_with_pinned_tls(private_payload)
        except _PinnedTLSFailure as error:
            return self._transport_failure(
                public_payload,
                http_status=None,
                error_type=error.reason,
            )
        except (HTTPException, OSError, TimeoutError, ValueError) as error:
            return self._transport_failure(
                public_payload,
                http_status=None,
                error_type=type(error).__name__,
            )
        if http_status != 200:
            return self._transport_failure(
                public_payload,
                http_status=http_status,
                error_type="unexpected_http_status",
            )
        try:
            response_payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._malformed_response(public_payload, raw_body, "invalid_json")
        # The upstream wrapper only requires ``error`` before it serializes
        # the whole mapping.  In particular, it does not require a
        # ``response`` member, so this bridge must not reject that valid
        # upstream response shape.
        if not isinstance(response_payload, Mapping) or "error" not in response_payload:
            return self._malformed_response(public_payload, raw_body, "missing_error")
        # ``rapidapi_wrapper`` only inspects these two keys and then serializes
        # the whole mapping. Retain arbitrary extra server fields in the model
        # observation, but never persist their body verbatim in the receipt.
        payload = self._redact_known_key(dict(response_payload))
        upstream_error = str(payload.get("error") or "")
        status_code = _upstream_status_code(upstream_error)
        if status_code == 10 and self._rate_limit_recovery_delay_s:
            time.sleep(self._rate_limit_recovery_delay_s)
        observation = _truncate_observation(
            redact_text(json.dumps(payload)),
            self._max_observation_length,
        )
        return ToolBenchRemoteExecution(
            observation=observation,
            status_code=status_code,
            is_error=status_code != 0,
            receipt={
                "schema": REMOTE_RECEIPT_SCHEMA,
                "execution_mode": "upstream_remote",
                "protocol_source": "OpenBMB/ToolBench rapidapi_wrapper._step",
                "request": self._public_request_receipt(public_payload),
                "response": {
                    "http_status": http_status,
                    "response_sha256": hashlib.sha256(raw_body).hexdigest(),
                    "error": redact_text(upstream_error),
                    "toolbench_status_code": status_code,
                    "observation_sha256": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
                },
            },
        )

    def _post_with_pinned_tls(
        self,
        private_payload: Mapping[str, Any],
    ) -> tuple[int, bytes]:
        """POST only after the exact peer certificate has been authenticated.

        ``urllib`` exposes a certificate only after it has already sent a
        request and can automatically follow redirects.  This lower-level
        transport deliberately does neither.  The same socket used for the
        pin check carries the authenticated POST, preventing a check/request
        time-of-check-to-time-of-use gap.
        """

        endpoint = urlsplit(self._service_url)
        hostname = endpoint.hostname
        if not hostname:
            raise _PinnedTLSFailure("peer_certificate_unavailable")
        try:
            port = endpoint.port
        except ValueError as error:
            raise _PinnedTLSFailure("invalid_service_url_port") from error
        path = endpoint.path or "/"
        if endpoint.query:
            path = f"{path}?{endpoint.query}"
        body = json.dumps(private_payload).encode("utf-8")
        connection = HTTPSConnection(
            hostname,
            port=port,
            timeout=self._request_timeout_s,
        )
        try:
            # ``connect`` completes TLS before any request bytes are emitted.
            connection.connect()
            socket = connection.sock
            certificate_getter = getattr(socket, "getpeercert", None)
            if not callable(certificate_getter):
                raise _PinnedTLSFailure("peer_certificate_unavailable")
            certificate = certificate_getter(binary_form=True)
            if not isinstance(certificate, bytes) or not certificate:
                raise _PinnedTLSFailure("peer_certificate_unavailable")
            observed = hashlib.sha256(certificate).hexdigest()
            if not hmac.compare_digest(observed, self._expected_tls_certificate_sha256):
                raise _PinnedTLSFailure("peer_certificate_mismatch")
            # HTTPSConnection never follows redirects. A 3xx response is
            # returned to the caller as an unexpected status rather than
            # replaying the key-bearing request against a new endpoint.
            connection.request(
                "POST",
                path,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "toolbench_key": self._toolbench_key,
                },
            )
            response = connection.getresponse()
            raw_status = getattr(response, "status", None)
            if raw_status is None:
                getcode = getattr(response, "getcode", None)
                if not callable(getcode):
                    raise ValueError("ToolBench remote response lacks an HTTP status")
                raw_status = getcode()
            return int(raw_status), _read_bounded_response(
                response,
                max_response_bytes=self._max_response_bytes,
            )
        finally:
            connection.close()

    def invalid_function(self, action_name: str) -> ToolBenchRemoteExecution:
        """Return upstream status 1 without sending an unauditable HTTP call."""

        payload = {"error": f"No such function name: {action_name}", "response": ""}
        observation = _truncate_observation(json.dumps(payload), self._max_observation_length)
        return ToolBenchRemoteExecution(
            observation=observation,
            status_code=1,
            is_error=True,
            receipt={
                "schema": REMOTE_RECEIPT_SCHEMA,
                "execution_mode": "upstream_remote",
                "protocol_source": "OpenBMB/ToolBench rapidapi_wrapper._step",
                "request": {
                    "service_url_sha256": self._service_url_sha256,
                    "not_sent": True,
                    "reason": "unknown_function_name",
                },
                "response": {
                    "http_status": None,
                    "toolbench_status_code": 1,
                    "error": payload["error"],
                    "observation_sha256": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
                },
            },
        )

    def _transport_failure(
        self,
        public_payload: Mapping[str, Any],
        *,
        http_status: int | None,
        error_type: str,
    ) -> ToolBenchRemoteExecution:
        error = (
            f"request invalid, data error. status_code={http_status}"
            if http_status is not None
            else "request invalid, data error"
        )
        payload = {"error": error, "response": ""}
        observation = _truncate_observation(json.dumps(payload), self._max_observation_length)
        return ToolBenchRemoteExecution(
            observation=observation,
            status_code=12,
            is_error=True,
            receipt={
                "schema": REMOTE_RECEIPT_SCHEMA,
                "execution_mode": "upstream_remote",
                "protocol_source": "OpenBMB/ToolBench rapidapi_wrapper._step",
                "request": self._public_request_receipt(public_payload),
                "response": {
                    "http_status": http_status,
                    "transport_error_type": error_type,
                    "toolbench_status_code": 12,
                    "error": error,
                    "observation_sha256": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
                },
            },
        )

    def _malformed_response(
        self,
        public_payload: Mapping[str, Any],
        raw_body: bytes,
        reason: str,
    ) -> ToolBenchRemoteExecution:
        error = "request invalid, data error"
        payload = {"error": error, "response": ""}
        observation = _truncate_observation(json.dumps(payload), self._max_observation_length)
        return ToolBenchRemoteExecution(
            observation=observation,
            status_code=12,
            is_error=True,
            receipt={
                "schema": REMOTE_RECEIPT_SCHEMA,
                "execution_mode": "upstream_remote",
                "protocol_source": "OpenBMB/ToolBench rapidapi_wrapper._step",
                "request": self._public_request_receipt(public_payload),
                "response": {
                    "http_status": 200,
                    "malformed_response": reason,
                    "response_sha256": hashlib.sha256(raw_body).hexdigest(),
                    "toolbench_status_code": 12,
                    "error": error,
                    "observation_sha256": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
                },
            },
        )

    def _public_request_receipt(self, public_payload: Mapping[str, Any]) -> dict[str, Any]:
        payload = self._redact_known_key(dict(public_payload))
        return {
            "service_url_sha256": self._service_url_sha256,
            "method": "POST",
            "payload": payload,
            "payload_sha256": canonical_sha256(payload),
        }

    def _redact_known_key(self, value: Any) -> Any:
        """Redact the exact in-memory key even when a server echoes it under
        an arbitrary field name.

        Generic key-name redaction cannot protect an untrusted remote service
        that returns ``{"debug": "<TOOLBENCH_KEY>"}``.  This explicit value
        pass runs before artifacts/observations are materialized.
        """

        if isinstance(value, str):
            return value.replace(self._toolbench_key, "<redacted>")
        if isinstance(value, Mapping):
            return {
                str(self._redact_known_key(str(key))): self._redact_known_key(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact_known_key(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_known_key(item) for item in value)
        return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _read_bounded_response(response: Any, *, max_response_bytes: int) -> bytes:
    """Read a remote body under a hard byte cap before parsing or logging it.

    Observation truncation protects only the stored text after allocation.
    The cap here protects the process from a compromised or malformed proxy.
    A declared oversized body and an undeclared oversized stream both fail
    closed without producing a semantic ToolBench observation.
    """

    getheader = getattr(response, "getheader", None)
    if callable(getheader):
        content_length = getheader("Content-Length")
        if content_length not in (None, ""):
            try:
                declared_size = int(str(content_length).strip())
            except (TypeError, ValueError) as error:
                raise _PinnedTLSFailure("invalid_content_length") from error
            if declared_size < 0:
                raise _PinnedTLSFailure("invalid_content_length")
            if declared_size > max_response_bytes:
                raise _PinnedTLSFailure("response_too_large")
    body = response.read(max_response_bytes + 1)
    if not isinstance(body, bytes):
        raise _PinnedTLSFailure("invalid_response_body")
    if len(body) > max_response_bytes:
        raise _PinnedTLSFailure("response_too_large")
    return body


def _upstream_status_code(error: str) -> int:
    """Match the status branch in upstream ``rapidapi_wrapper._step``."""

    mapping = {
        "API not working error...": 6,
        "Unauthorized error...": 7,
        "Unsubscribed error...": 8,
        "Too many requests error...": 9,
        "Rate limit per minute error...": 10,
        "Message error...": 11,
    }
    return mapping.get(error, 0)


def _truncate_observation(value: str, limit: int) -> str:
    """Keep the upstream wrapper's ``step`` truncation behavior exactly."""

    return value if len(value) <= limit else value[:limit] + "..."
