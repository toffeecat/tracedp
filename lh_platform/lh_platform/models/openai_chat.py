"""OpenAI-compatible chat model.

Works with OpenAI directly, or any service that exposes the
`/v1/chat/completions` endpoint (DeepSeek, Moonshot, Ollama, vLLM,
Anthropic-via-OpenAI-proxy, etc.) via `OPENAI_BASE_URL`.

If `OPENAI_API_KEY` is missing, compatibility/plumbing runs may fall back to
the DummyModel behavior.  Strict local-model profiles set
``allow_dummy_fallback=false``; then a missing client is a hard error rather
than a fabricated Qwen/vLLM result.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

from lh_platform.core.artifact_redaction import redact_secrets
from lh_platform.core.model_runtime import ModelRuntimePreflightError
from lh_platform.core.registry import MODELS
from lh_platform.models.base import Message, ModelBase, ModelOutput

PRICING_PER_1K = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
    "gpt-4.1": {"prompt": 0.002, "completion": 0.008},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "deepseek-chat": {"prompt": 0.00014, "completion": 0.00028},
    "moonshot-v1-8k": {"prompt": 0.0006, "completion": 0.0006},
}


@MODELS.register(
    "openai_chat",
    aliases=[
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-3.5-turbo",
        "deepseek-chat",
        "moonshot-v1-8k",
    ],
)
class OpenAIChatModel(ModelBase):
    name = "openai_chat"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        allow_dummy_fallback: bool = True,
        **kwargs: object,
    ) -> None:
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.allow_dummy_fallback = bool(allow_dummy_fallback)
        raw_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._api_key = "" if raw_key is None else raw_key.strip()
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self._client = None
        self._client_init_error: str | None = None
        self._dummy_fallback_count = 0
        self._remote_call_count = 0
        self._remote_error_count = 0
        self._served_model_ids: set[str] = set()
        if not self._api_key:
            unavailable_message = (
                "[openai_chat] OPENAI_API_KEY not set; strict runtime will fail before "
                "any dummy output is produced."
                if not self.allow_dummy_fallback
                else "[openai_chat] OPENAI_API_KEY not set; calls will fall back to a dummy echo."
            )
            warnings.warn(
                unavailable_message,
                stacklevel=2,
            )
        else:
            if self._api_key == "dummy" and not self._base_url:
                unavailable_message = (
                    "[openai_chat] OPENAI_API_KEY='dummy' without OPENAI_BASE_URL; "
                    "strict runtime will fail before any dummy output is produced."
                    if not self.allow_dummy_fallback
                    else "[openai_chat] OPENAI_API_KEY='dummy' without OPENAI_BASE_URL; "
                    "calls will fall back to a dummy echo."
                )
                warnings.warn(
                    unavailable_message,
                    stacklevel=2,
                )
            else:
                try:
                    from openai import OpenAI

                    self._client = OpenAI(api_key=self._api_key or "dummy", base_url=self._base_url)
                except Exception as e:
                    self._client_init_error = type(e).__name__
                    warnings.warn(
                        f"[openai_chat] failed to construct OpenAI client: {e}", stacklevel=2
                    )
                    self._client = None

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        p = PRICING_PER_1K.get(self.model_name)
        if not p:
            return 0.0
        return (prompt_tokens / 1000.0) * p["prompt"] + (completion_tokens / 1000.0) * p[
            "completion"
        ]

    @staticmethod
    def _normalize_logprobs(raw_logprobs: Any) -> dict[str, Any] | None:
        """Best-effort normalization of OpenAI / vLLM logprob payloads.

        Returns None if the payload is absent or unparseable.
        """
        if raw_logprobs is None:
            return None
        # OpenAI v1 ChatCompletions returns object with .content (list of
        # ChoiceLogprob items) when logprobs=True was passed. Some proxies
        # (vLLM, llama.cpp) may return a flat dict.
        if hasattr(raw_logprobs, "content"):
            content = raw_logprobs.content
            if not content:
                return None
            tokens = []
            token_logprobs = []
            top_logprobs = []
            for item in content:
                tokens.append(getattr(item, "token", "") or "")
                token_logprobs.append(float(getattr(item, "logprob", 0.0) or 0.0))
                # top_logprobs per token is a list of {token, logprob, bytes}; we
                # flatten to {str(tok): float(lp)} for downstream methods.
                tops = {}
                for t in getattr(item, "top_logprobs", None) or []:
                    t_tok = getattr(t, "token", "")
                    if t_tok:
                        tops[t_tok] = float(getattr(t, "logprob", 0.0) or 0.0)
                top_logprobs.append(tops)
            return {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }
        if isinstance(raw_logprobs, dict):
            return raw_logprobs
        return None

    def generate(self, messages: list[Message], **kwargs: object) -> ModelOutput:
        if self._client is None:
            if not self.allow_dummy_fallback:
                raise ModelRuntimePreflightError(
                    "openai-compatible client is unavailable and dummy fallback is disabled"
                )
            self._dummy_fallback_count += 1
            text = "Action: answer(42)"
            warnings.warn(
                "[openai_chat] no API client available; returning dummy response.",
                stacklevel=2,
            )
            output = ModelOutput(
                text=text,
                prompt_tokens=10,
                completion_tokens=4,
                cost_usd=0.0,
                info={
                    "backend": "dummy_fallback",
                    "dummy_fallback_used": True,
                    "requested_model": self.model_name,
                },
            )
            return self._record_generation_event(
                messages,
                output,
                outcome="dummy_fallback",
                backend="dummy_fallback",
                requested_model=self.model_name,
            )
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        # Compose request kwargs. `seed`, `logprobs`/`top_logprobs` come
        # either from kwargs (call-site override) or instance defaults.
        req_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": msg_dicts,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if "seed" in kwargs and kwargs["seed"] is not None:
            req_kwargs["seed"] = kwargs["seed"]
        elif self.seed is not None:
            req_kwargs["seed"] = self.seed
        if kwargs.get("logprobs"):
            req_kwargs["logprobs"] = True
            req_kwargs["top_logprobs"] = int(kwargs.get("top_logprobs", 5))
        try:
            resp = self._client.chat.completions.create(**req_kwargs)
        except Exception as e:
            self._remote_error_count += 1
            warnings.warn(f"[openai_chat] call failed: {e}", stacklevel=2)
            output = ModelOutput(
                text=f"Action: error({e})",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                info={
                    "backend": "openai_compatible_error",
                    "dummy_fallback_used": False,
                    "requested_model": self.model_name,
                },
            )
            return self._record_generation_event(
                messages,
                output,
                outcome="remote_error",
                backend="openai_compatible_error",
                requested_model=self.model_name,
            )
        self._remote_call_count += 1
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) if usage else 0
        ct = getattr(usage, "completion_tokens", 0) if usage else 0
        served_model = str(getattr(resp, "model", "") or "").strip()
        if served_model:
            self._served_model_ids.add(served_model)
        raw_lp = getattr(choice, "logprobs", None)
        normalized_lp = self._normalize_logprobs(raw_lp)
        output = ModelOutput(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=self._estimate_cost(pt, ct),
            raw=resp,
            logprobs=normalized_lp,
            info={
                "seed": req_kwargs.get("seed"),
                "backend": "openai_compatible",
                "dummy_fallback_used": False,
                "requested_model": self.model_name,
                "served_model": served_model or None,
            },
        )
        return self._record_generation_event(
            messages,
            output,
            outcome="remote_success",
            backend="openai_compatible",
            requested_model=self.model_name,
        )

    def runtime_evidence(self) -> dict[str, Any]:
        """Report actual backend usage without exposing credentials."""

        generation_count = (
            self._dummy_fallback_count + self._remote_call_count + self._remote_error_count
        )
        return {
            "backend": "openai_compatible" if self._client is not None else "unavailable",
            "model_instantiated": True,
            "requested_model": self.model_name,
            "base_url": redact_secrets({"base_url": self._base_url}).get("base_url"),
            "allow_dummy_fallback": self.allow_dummy_fallback,
            "client_constructed": self._client is not None,
            "client_init_error": self._client_init_error,
            "generation_count": generation_count,
            "generation_error_count": self._remote_error_count,
            "model_identity_ids": sorted(self._served_model_ids),
            "remote_call_count": self._remote_call_count,
            "remote_error_count": self._remote_error_count,
            "served_model_ids": sorted(self._served_model_ids),
            "dummy_fallback_count": self._dummy_fallback_count,
            "dummy_model_used": False,
            "dummy_fallback_used": self._dummy_fallback_count > 0,
        }
