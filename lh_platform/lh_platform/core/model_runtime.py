"""Model runtime policy, live-server preflight, and observed evidence."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from lh_platform.core.artifact_redaction import redact_secrets

logger = logging.getLogger(__name__)


RUNTIME_CONTRACT_SCHEMA = "lh_model_runtime_contract/v1"
MAX_MODELS_RESPONSE_BYTES = 1024 * 1024
_PROCESS_STOP_GRACE_S = 0.1
_RUNTIME_KEYS = frozenset(
    {
        "require_live_server",
        "allow_dummy_fallback",
        "expected_model",
        "healthcheck_timeout_s",
        "model_revision",
        "tokenizer",
        "chat_template",
        "server_version",
    }
)


class ModelRuntimePreflightError(RuntimeError):
    """Raised when a requested strict model runtime cannot be proven live."""


def _coerce_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field} must be a boolean")


@dataclass(frozen=True)
class ModelRuntimePolicy:
    """Requested runtime facts for a model-backed experiment."""

    require_live_server: bool = False
    allow_dummy_fallback: bool = True
    expected_model: str | None = None
    healthcheck_timeout_s: float = 5.0
    model_revision: str | None = None
    tokenizer: str | None = None
    chat_template: str | None = None
    server_version: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        default_expected_model: str | None = None,
    ) -> ModelRuntimePolicy:
        data = dict(raw or {})
        require_live_server = _coerce_bool(
            data.get("require_live_server", False),
            field="runtime.require_live_server",
        )
        allow_dummy_fallback = _coerce_bool(
            data.get("allow_dummy_fallback", True),
            field="runtime.allow_dummy_fallback",
        )
        expected = str(data.get("expected_model") or default_expected_model or "").strip() or None
        try:
            timeout = float(data.get("healthcheck_timeout_s", 5.0))
        except (TypeError, ValueError) as error:
            raise ValueError("runtime.healthcheck_timeout_s must be numeric") from error
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("runtime.healthcheck_timeout_s must be finite and > 0")
        if require_live_server and allow_dummy_fallback:
            raise ValueError(
                "strict local runtime requires allow_dummy_fallback=false; "
                "otherwise an unavailable server could be misreported as a model run"
            )
        if require_live_server and not expected:
            raise ValueError(
                "strict local runtime requires runtime.expected_model or model.kwargs.model"
            )

        def optional_string(key: str) -> str | None:
            value = str(data.get(key) or "").strip()
            return value or None

        return cls(
            require_live_server=require_live_server,
            allow_dummy_fallback=allow_dummy_fallback,
            expected_model=expected,
            healthcheck_timeout_s=timeout,
            model_revision=optional_string("model_revision"),
            tokenizer=optional_string("tokenizer"),
            chat_template=optional_string("chat_template"),
            server_version=optional_string("server_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelRuntimeEvidence:
    """The observed result of a model-runtime preflight."""

    schema: str
    provider: str
    requested_model: str | None
    expected_model: str | None
    base_url: str | None
    require_live_server: bool
    dummy_fallback_allowed: bool
    passed: bool
    reachable: bool | None
    served_models: tuple[str, ...]
    reason: str
    checked_at: float

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(asdict(self))


def split_model_kwargs_and_runtime(
    raw: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a flat model profile into invocation kwargs and runtime facts."""

    kwargs = dict(raw or {})
    runtime_raw = kwargs.pop("runtime", {})
    if runtime_raw is None:
        runtime_raw = {}
    if not isinstance(runtime_raw, Mapping):
        raise ValueError("model runtime must be a mapping")
    runtime = dict(runtime_raw)
    for key in _RUNTIME_KEYS:
        if key in kwargs:
            runtime.setdefault(key, kwargs.pop(key))
    kwargs.pop("model_id", None)
    for key in tuple(kwargs):
        if key.startswith("_"):
            kwargs.pop(key)
    return kwargs, runtime


def _models_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _request_openai_models(endpoint: str, api_key: str, timeout_s: float) -> list[str]:
    """Read model ids under a hard deadline in a terminable child process."""

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_request_openai_models_worker,
        args=(sender, endpoint, api_key, timeout_s),
        daemon=True,
    )
    deadline = time.monotonic() + timeout_s
    started = False
    sender_open = True
    try:
        process.start()
        started = True
        sender.close()
        sender_open = False
        remaining = max(0.0, deadline - time.monotonic())
        if not receiver.poll(remaining):
            raise TimeoutError("/v1/models request exceeded total deadline")
        try:
            outcome, payload = receiver.recv()
        except EOFError as error:
            raise OSError("/v1/models worker exited without a result") from error
        process.join(max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            raise TimeoutError("/v1/models worker did not exit before deadline")
    finally:
        if sender_open:
            sender.close()
        receiver.close()
        if started:
            stopped = _stop_process_bounded(process)
            if stopped:
                process.close()
        else:
            process.close()
    if outcome == "ok":
        return list(payload)
    if outcome == "network_error":
        raise URLError(str(payload))
    raise ValueError(str(payload))


def _stop_process_bounded(process: Any) -> bool:
    """Stop a worker without introducing an unbounded cleanup wait."""

    if not process.is_alive():
        return True
    try:
        process.terminate()
    except Exception:
        logger.exception("model_runtime: process.terminate failed")
    process.join(_PROCESS_STOP_GRACE_S)
    if process.is_alive():
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                logger.exception("model_runtime: process.kill failed")
        process.join(_PROCESS_STOP_GRACE_S)
    return not process.is_alive()


def _request_openai_models_worker(
    sender: Any,
    endpoint: str,
    api_key: str,
    timeout_s: float,
) -> None:
    try:
        sender.send(("ok", _request_openai_models_once(endpoint, api_key, timeout_s)))
    except (OSError, URLError) as error:
        sender.send(("network_error", type(error).__name__))
    except Exception as error:
        sender.send(("invalid_response", type(error).__name__))
    finally:
        sender.close()


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request_openai_models_once(
    endpoint: str,
    api_key: str,
    timeout_s: float,
) -> list[str]:
    """Perform one bounded request without forwarding credentials on redirect."""

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, headers=headers, method="GET")
    opener = build_opener(_RejectRedirects())
    try:
        with opener.open(request, timeout=timeout_s) as response:  # nosec B310
            body = response.read(MAX_MODELS_RESPONSE_BYTES + 1)
    except HTTPError as error:
        raise URLError(f"redirect_or_http_error:{error.code}") from error
    if len(body) > MAX_MODELS_RESPONSE_BYTES:
        raise ValueError(f"/v1/models response exceeds {MAX_MODELS_RESPONSE_BYTES} byte limit")
    payload = json.loads(body.decode("utf-8"))
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, list):
        raise ValueError("/v1/models response does not contain a data list")
    model_ids = sorted(
        {
            str(item.get("id")).strip()
            for item in data
            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
        }
    )
    if not model_ids:
        raise ValueError("/v1/models response contains no model ids")
    return model_ids


def _construct_openai_client(*, api_key: str, base_url: str) -> None:
    """Prove that the same OpenAI client used by the model can initialize."""

    from openai import OpenAI

    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def preflight_model_runtime(
    *,
    provider: str,
    model_kwargs: Mapping[str, Any],
    policy: ModelRuntimePolicy,
) -> ModelRuntimeEvidence:
    """Verify an explicitly strict local OpenAI-compatible model runtime."""

    requested_model = str(model_kwargs.get("model") or "").strip() or None
    base_url = str(model_kwargs.get("base_url") or "").strip() or None
    resolved_api_key = str(
        model_kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY") or ""
    ).strip()
    common = {
        "schema": RUNTIME_CONTRACT_SCHEMA,
        "provider": provider,
        "requested_model": requested_model,
        "expected_model": policy.expected_model,
        "base_url": base_url,
        "require_live_server": policy.require_live_server,
        "dummy_fallback_allowed": policy.allow_dummy_fallback,
        "checked_at": time.time(),
    }
    if not policy.require_live_server:
        return ModelRuntimeEvidence(
            **common,
            passed=True,
            reachable=None,
            served_models=(),
            reason="live_server_not_required",
        )
    if provider != "openai_chat":
        return ModelRuntimeEvidence(
            **common,
            passed=False,
            reachable=None,
            served_models=(),
            reason="strict_live_server_requires_openai_compatible_provider",
        )
    if not base_url:
        return ModelRuntimeEvidence(
            **common,
            passed=False,
            reachable=False,
            served_models=(),
            reason="strict_live_server_requires_base_url",
        )
    if not resolved_api_key:
        return ModelRuntimeEvidence(
            **common,
            passed=False,
            reachable=None,
            served_models=(),
            reason="strict_live_server_requires_nonempty_api_key",
        )
    expected = policy.expected_model
    if requested_model != expected:
        return ModelRuntimeEvidence(
            **common,
            passed=False,
            reachable=None,
            served_models=(),
            reason=f"requested_model_mismatch:{requested_model!r}!={expected!r}",
        )
    try:
        served = tuple(
            _request_openai_models(
                _models_endpoint(base_url),
                resolved_api_key,
                policy.healthcheck_timeout_s,
            )
        )
    except (OSError, URLError, ValueError, json.JSONDecodeError) as error:
        return ModelRuntimeEvidence(
            **common,
            passed=False,
            reachable=False,
            served_models=(),
            reason=f"live_server_unreachable:{type(error).__name__}",
        )
    if expected not in served:
        return ModelRuntimeEvidence(
            **common,
            passed=False,
            reachable=True,
            served_models=served,
            reason=f"expected_model_not_served:{expected}",
        )
    try:
        _construct_openai_client(api_key=resolved_api_key, base_url=base_url)
    except Exception as error:
        return ModelRuntimeEvidence(
            **common,
            passed=False,
            reachable=True,
            served_models=served,
            reason=f"openai_client_initialization_failed:{type(error).__name__}",
        )
    return ModelRuntimeEvidence(
        **common,
        passed=True,
        reachable=True,
        served_models=served,
        reason="live_server_verified",
    )


def failed_preflight_evidence(
    *,
    provider: str,
    model_kwargs: Mapping[str, Any],
    policy: ModelRuntimePolicy,
    error: Exception,
) -> ModelRuntimeEvidence:
    """Turn a preflight exception into an artifact-safe failed record."""

    return ModelRuntimeEvidence(
        schema=RUNTIME_CONTRACT_SCHEMA,
        provider=provider,
        requested_model=str(model_kwargs.get("model") or "").strip() or None,
        expected_model=policy.expected_model,
        base_url=str(model_kwargs.get("base_url") or "").strip() or None,
        require_live_server=policy.require_live_server,
        dummy_fallback_allowed=policy.allow_dummy_fallback,
        passed=False,
        reachable=False if policy.require_live_server else None,
        served_models=(),
        reason=f"preflight_exception:{type(error).__name__}",
        checked_at=time.time(),
    )


def skipped_preflight_evidence(
    *,
    provider: str,
    model_kwargs: Mapping[str, Any],
    policy: ModelRuntimePolicy,
    reason: str,
) -> ModelRuntimeEvidence:
    """Record that no server request occurred due to an earlier rejection."""

    return ModelRuntimeEvidence(
        schema=RUNTIME_CONTRACT_SCHEMA,
        provider=provider,
        requested_model=str(model_kwargs.get("model") or "").strip() or None,
        expected_model=policy.expected_model,
        base_url=str(model_kwargs.get("base_url") or "").strip() or None,
        require_live_server=policy.require_live_server,
        dummy_fallback_allowed=policy.allow_dummy_fallback,
        passed=False,
        reachable=None,
        served_models=(),
        reason=f"preflight_skipped:{reason}",
        checked_at=time.time(),
    )


__all__ = [
    "ModelRuntimeEvidence",
    "ModelRuntimePolicy",
    "ModelRuntimePreflightError",
    "RUNTIME_CONTRACT_SCHEMA",
    "failed_preflight_evidence",
    "preflight_model_runtime",
    "skipped_preflight_evidence",
    "split_model_kwargs_and_runtime",
]
