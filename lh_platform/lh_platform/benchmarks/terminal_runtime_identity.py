"""Frozen identity for the host-Pi Terminal runtime cell."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from lh_platform.models.controlled_runtime import CONTROLLED_RUNTIME


@dataclass(frozen=True)
class TerminalRuntimeIdentity:
    provider: str
    model: str
    transport: str
    thinking: str
    credential_env: str

    @property
    def model_name(self) -> str:
        return f"{self.provider}/{self.model}"


HOST_PI_RUNTIME = TerminalRuntimeIdentity(
    provider=CONTROLLED_RUNTIME.provider,
    model=CONTROLLED_RUNTIME.model,
    transport="host_pi_unix_rpc",
    thinking=CONTROLLED_RUNTIME.thinking,
    credential_env="OPENCODE_API_KEY",
)


# Every actor cell freezes one thinking level; a model not listed here is thinking=off.
THINKING_BY_MODEL = {("minimax-cn", "MiniMax-M3.1-Flash-Preview"): "low"}

# Models the pinned Pi catalog does not know, registered in each session's models.json.
# Values follow Pi's own MiniMax-M3 entry (minimax-cn.json); cost is metered by tokens.
CUSTOM_HOST_PI_MODELS = {
    ("minimax-cn", "MiniMax-M3.1-Flash-Preview"): {
        "id": "MiniMax-M3.1-Flash-Preview",
        "name": "MiniMax-M3.1-Flash-Preview",
        "api": "anthropic-messages",
        "baseUrl": "https://api.minimaxi.com/anthropic",
        "reasoning": True,
        "input": ["text"],
        "contextWindow": 512000,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    },
}


def resolve_host_pi_runtime(
    *, provider: str = "opencode-go", model: str, thinking: str
) -> TerminalRuntimeIdentity:
    """Create a frozen Terminal actor identity for one explicitly named cell.

    MiMo remains the canonical controlled cell. Alternative actors are allowed
    only in separately labelled compatibility and cost cells, preventing mixed
    source populations from being analysed as one experiment.
    """

    credentials = {
        ("opencode-go", "mimo-v2.5"): "OPENCODE_API_KEY",
        ("opencode-go", "gpt-5.6-luna"): "OPENCODE_API_KEY",
        # MiniMax documents that M2.x always emits thinking, while M3 can
        # honor an explicit disabled-thinking contract.
        ("minimax-cn", "MiniMax-M3"): "MINIMAX_CN_API_KEY",
        # Pi 0.84.1 has no catalog entry, so the
        # model is registered per session (CUSTOM_HOST_PI_MODELS); thinking is low.
        ("minimax-cn", "MiniMax-M3.1-Flash-Preview"): "MINIMAX_CN_PLAN_API_KEY",
    }
    if provider == "minimax-cn" and model.startswith("MiniMax-M2."):
        raise ValueError(
            "MiniMax M2.x cannot run a thinking=off Terminal cell; use MiniMax-M3"
        )
    credential_env = credentials.get((provider, model))
    if credential_env is None:
        raise ValueError(f"unsupported host Pi Terminal runtime: {provider}/{model}")
    allowed_thinking = THINKING_BY_MODEL.get((provider, model), "off")
    if thinking != allowed_thinking:
        raise ValueError(
            f"Terminal source cells for {provider}/{model} require thinking={allowed_thinking}"
        )
    return TerminalRuntimeIdentity(
        provider=provider,
        model=model,
        transport=HOST_PI_RUNTIME.transport,
        thinking=thinking,
        credential_env=credential_env,
    )


def frozen_terminal_identity_from(
    payload: Mapping[str, object],
    *,
    allow_legacy_mimo_transport: bool = False,
) -> TerminalRuntimeIdentity:
    """Validate and reconstruct the actor identity frozen by a Terminal cell."""

    provider = payload.get("provider")
    model = payload.get("model")
    transport = payload.get("transport")
    thinking = payload.get("thinking")
    if (
        transport is None
        and allow_legacy_mimo_transport
        and provider == CONTROLLED_RUNTIME.provider
        and model == CONTROLLED_RUNTIME.model
        and thinking == CONTROLLED_RUNTIME.thinking
    ):
        transport = HOST_PI_RUNTIME.transport
    if not all(isinstance(value, str) and value for value in (provider, model, transport, thinking)):
        raise ValueError("Terminal actor identity must be explicit and complete")
    identity = resolve_host_pi_runtime(
        provider=provider,
        model=model,
        thinking=thinking,
    )
    if transport != identity.transport:
        raise ValueError("Terminal actor transport differs from the frozen runtime")
    return identity


def terminal_identity_payload(identity: TerminalRuntimeIdentity) -> dict[str, str]:
    """Serialize only the scientific identity, never its credential binding."""

    return {
        "provider": identity.provider,
        "model": identity.model,
        "transport": identity.transport,
        "thinking": identity.thinking,
    }


def replay_population_identity(config: Mapping[str, object]) -> TerminalRuntimeIdentity:
    """Read a replay-population identity, including the bounded legacy MiMo form."""

    schema = config.get("schema_version")
    if schema not in {None, "terminal_zero_model_replay_population_run_v2"}:
        raise ValueError("Terminal replay population config version is unsupported")
    return frozen_terminal_identity_from(
        config,
        allow_legacy_mimo_transport=schema is None,
    )


def replay_run_identity(config: Mapping[str, object]) -> TerminalRuntimeIdentity:
    """Read the source identity propagated by a zero-model replay run."""

    schema = config.get("schema_version")
    if schema not in {
        "terminal_zero_model_replay_run_v1",
        "terminal_zero_model_replay_run_v2",
    }:
        raise ValueError("Terminal replay run config version is unsupported")
    return frozen_terminal_identity_from(
        config,
        allow_legacy_mimo_transport=schema == "terminal_zero_model_replay_run_v1",
    )


def public_write_population_identity(
    config: Mapping[str, object],
) -> TerminalRuntimeIdentity:
    """Read the actor identity bound to a public-write population."""

    schema = config.get("schema_version")
    if schema == "terminal_public_write_probe_population_run_v1":
        return HOST_PI_RUNTIME
    if schema not in {
        "terminal_public_write_probe_population_run_v2",
        "terminal_public_write_probe_population_run_v3",
    }:
        raise ValueError("Terminal public-write population config version is unsupported")
    payload = config.get("source_actor_identity")
    return frozen_terminal_identity_from(payload if isinstance(payload, Mapping) else {})


def three_arm_probe_identity(config: Mapping[str, object]) -> TerminalRuntimeIdentity:
    """Read the actor identity preserved by a three-arm probe run."""

    schema = config.get("schema_version")
    if schema == "terminal_public_write_three_arm_probe_run_v1":
        return frozen_terminal_identity_from(
            config,
            allow_legacy_mimo_transport=True,
        )
    if schema not in {
        "terminal_public_write_three_arm_probe_run_v2",
        "terminal_public_write_three_arm_probe_run_v3",
    }:
        raise ValueError("Terminal three-arm probe config version is unsupported")
    payload = config.get("source_actor_identity")
    return frozen_terminal_identity_from(payload if isinstance(payload, Mapping) else {})
