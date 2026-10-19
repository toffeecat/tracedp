"""Frozen model identity for all LongH-controlled inference roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class ControlledRuntimeIdentityError(ValueError):
    """Reject a new controlled run whose inference identity has drifted."""


@dataclass(frozen=True)
class ControlledRuntimeIdentity:
    provider: str
    model: str
    thinking: str

    @property
    def model_name(self) -> str:
        return f"{self.provider}/{self.model}"

    def require(self, *, provider: str, model: str, thinking: str) -> None:
        """Validate identity before any controlled model request is issued."""

        observed = (provider, model, thinking)
        expected = (self.provider, self.model, self.thinking)
        if observed != expected:
            raise ControlledRuntimeIdentityError(
                "LongH controlled runtime must use "
                f"{self.provider}/{self.model} with thinking={self.thinking}; "
                f"received {provider}/{model} with thinking={thinking}"
            )


CONTROLLED_RUNTIME = ControlledRuntimeIdentity(
    provider="opencode-go",
    model="mimo-v2.5",
    thinking="off",
)

RuntimeModelPurpose = Literal[
    "longh_controlled",
    "official_grader",
    "legacy_reproduction",
]

_EXEMPT_RUNTIME_PURPOSES = frozenset({"official_grader", "legacy_reproduction"})


def require_runtime_model_identity(
    *,
    provider: str,
    model: str,
    thinking: str,
    purpose: RuntimeModelPurpose,
    expected_identity: ControlledRuntimeIdentity | None = None,
) -> None:
    """Enforce the default or an explicitly frozen per-cell model identity."""

    if purpose == "longh_controlled":
        (expected_identity or CONTROLLED_RUNTIME).require(
            provider=provider,
            model=model,
            thinking=thinking,
        )
        return
    if expected_identity is not None:
        raise ControlledRuntimeIdentityError(
            "an explicit cell identity is only valid for longh_controlled runs"
        )
    if purpose not in _EXEMPT_RUNTIME_PURPOSES:
        raise ControlledRuntimeIdentityError(
            f"unsupported runtime model purpose: {purpose!r}"
        )
