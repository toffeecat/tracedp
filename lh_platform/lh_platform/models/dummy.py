"""Dummy model: returns a fixed deterministic action.

Always available. Used for smoke tests and CI.
"""

from __future__ import annotations

import random
from typing import Any

from lh_platform.core.registry import MODELS
from lh_platform.models.base import Message, ModelBase, ModelOutput


@MODELS.register("dummy")
class DummyModel(ModelBase):
    name = "dummy"

    def __init__(self, seed: int = 0, **kwargs: object) -> None:
        self._rng = random.Random(seed)
        self._generation_count = 0

    def generate(self, messages: list[Message], **kwargs: object) -> ModelOutput:
        self._generation_count += 1
        text = "Action: answer(42)"
        logprobs = None
        if kwargs.get("logprobs"):
            # Trivial constant-logprob payload so logprobs-aware methods
            # (e.g. jitrl) don't crash in CI. Token length matches text.
            tokens = text.split() or ["Action:"]
            const = -1.0
            logprobs = {
                "tokens": tokens,
                "token_logprobs": [const] * len(tokens),
                "top_logprobs": [{t: const} for t in tokens],
            }
        output = ModelOutput(
            text=text,
            prompt_tokens=10,
            completion_tokens=4,
            cost_usd=0.0,
            logprobs=logprobs,
            info={"dummy": True},
        )
        return self._record_generation_event(
            messages,
            output,
            outcome="dummy_model",
            backend="dummy",
            requested_model=self.name,
        )

    def runtime_evidence(self) -> dict[str, Any]:
        return {
            "backend": "dummy",
            "model_instantiated": True,
            "generation_count": self._generation_count,
            "dummy_model_used": True,
            "dummy_fallback_used": False,
        }
