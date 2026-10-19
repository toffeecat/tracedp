"""ComponentRef identity helper for built-in agentic-RL adapters."""

from __future__ import annotations

from collections.abc import Mapping

from lh_platform.training.agentic_rl_runtime import ComponentRef, canonical_sha256


def builtin_component(
    *,
    role: str,
    name: str,
    revision: str = "v1",
    config: Mapping[str, object] | None = None,
) -> ComponentRef:
    identity = {
        "owner": "lh_platform",
        "role": role,
        "name": name,
        "revision": revision,
        "config": dict(config or {}),
    }
    return ComponentRef(
        role=role,
        name=name,
        revision=revision,
        sha256=canonical_sha256(identity),
    )
