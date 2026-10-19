"""Checkpoint-specific model I/O capabilities and template parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from lh_platform.training.model_io.fingerprints import canonical_sha256
from lh_platform.training.model_io.messages import ModelIOContractError

SystemRolePolicy = Literal["native", "fold_into_first_user", "forbid"]
ReasoningCapability = Literal[
    "fixed_off",
    "switchable",
    "fixed_on",
    "prompt_instructed",
]
ReasoningMode = Literal["off", "on"]
ReasoningDelimiters = Literal["none", "paired", "closing_only"]
AssistantTurnEndMode = Literal[
    "infer_from_empty_assistant",
    "infer_from_distinct_assistant_suffix",
    "golden_fixture",
]


@dataclass(frozen=True)
class ModelIOProfile:
    profile_id: str
    renderer_id: str
    supported_roles: tuple[str, ...]
    system_role_policy: SystemRolePolicy
    reasoning_capability: ReasoningCapability
    selected_reasoning_mode: ReasoningMode
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    prompt_injects_reasoning_start: bool = False
    generated_reasoning_delimiters: ReasoningDelimiters = "none"
    assistant_turn_end_mode: AssistantTurnEndMode = "infer_from_empty_assistant"

    def __post_init__(self) -> None:
        if not self.profile_id or self.renderer_id != "hf_native_chat_template_v1":
            raise ModelIOContractError("model I/O profile identity/renderer is invalid")
        if (
            not self.supported_roles
            or len(set(self.supported_roles)) != len(self.supported_roles)
            or not set(self.supported_roles) <= {"system", "user", "assistant", "tool"}
        ):
            raise ModelIOContractError("model I/O supported roles are invalid")
        if self.system_role_policy not in {"native", "fold_into_first_user", "forbid"}:
            raise ModelIOContractError("system role policy is invalid")
        if self.reasoning_capability not in {
            "fixed_off",
            "switchable",
            "fixed_on",
            "prompt_instructed",
        }:
            raise ModelIOContractError("reasoning capability is invalid")
        if self.selected_reasoning_mode not in {"off", "on"}:
            raise ModelIOContractError("selected reasoning mode is invalid")
        if self.generated_reasoning_delimiters not in {
            "none",
            "paired",
            "closing_only",
        }:
            raise ModelIOContractError("generated reasoning delimiters are invalid")
        if self.reasoning_capability == "fixed_off" and self.selected_reasoning_mode != "off":
            raise ModelIOContractError("fixed_off profile cannot select reasoning on")
        if self.reasoning_capability == "fixed_on" and self.selected_reasoning_mode != "on":
            raise ModelIOContractError("fixed_on profile cannot select reasoning off")
        thinking = self.template_kwargs.get("enable_thinking")
        if self.reasoning_capability == "switchable":
            expected = self.selected_reasoning_mode == "on"
            if thinking is not expected:
                raise ModelIOContractError(
                    "switchable reasoning profile requires explicit enable_thinking"
                )
        elif "enable_thinking" in self.template_kwargs:
            raise ModelIOContractError("fixed reasoning profile must not pass enable_thinking")
        if self.selected_reasoning_mode == "off" and (
            self.prompt_injects_reasoning_start or self.generated_reasoning_delimiters != "none"
        ):
            raise ModelIOContractError("non-thinking profile declares reasoning delimiters")
        if self.generated_reasoning_delimiters == "closing_only" and not (
            self.selected_reasoning_mode == "on" and self.prompt_injects_reasoning_start
        ):
            raise ModelIOContractError(
                "closing-only reasoning requires a prompt-injected opening delimiter"
            )
        if self.assistant_turn_end_mode not in {
            "infer_from_empty_assistant",
            "infer_from_distinct_assistant_suffix",
        }:
            raise ModelIOContractError("golden fixture turn-end mode is not implemented")
        canonical_sha256(dict(self.template_kwargs))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelIOProfile:
        expected = {
            "profile_id",
            "renderer_id",
            "supported_roles",
            "system_role_policy",
            "reasoning_capability",
            "selected_reasoning_mode",
            "template_kwargs",
            "prompt_injects_reasoning_start",
            "generated_reasoning_delimiters",
            "assistant_turn_end_mode",
        }
        if set(raw) != expected:
            raise ModelIOContractError("model I/O profile fields are invalid")
        roles = raw.get("supported_roles")
        template_kwargs = raw.get("template_kwargs")
        if not isinstance(roles, (list, tuple)) or not all(isinstance(role, str) for role in roles):
            raise ModelIOContractError("model I/O supported roles are invalid")
        if not isinstance(template_kwargs, Mapping):
            raise ModelIOContractError("model I/O template kwargs are invalid")
        prompt_injects_start = raw.get("prompt_injects_reasoning_start")
        if type(prompt_injects_start) is not bool:
            raise ModelIOContractError("reasoning-start injection flag is invalid")
        return cls(
            profile_id=str(raw.get("profile_id", "")),
            renderer_id=str(raw.get("renderer_id", "")),
            supported_roles=tuple(roles),
            system_role_policy=str(raw.get("system_role_policy", "")),
            reasoning_capability=str(raw.get("reasoning_capability", "")),
            selected_reasoning_mode=str(raw.get("selected_reasoning_mode", "")),
            template_kwargs=dict(template_kwargs),
            prompt_injects_reasoning_start=prompt_injects_start,
            generated_reasoning_delimiters=str(raw.get("generated_reasoning_delimiters", "")),
            assistant_turn_end_mode=str(raw.get("assistant_turn_end_mode", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "renderer_id": self.renderer_id,
            "supported_roles": list(self.supported_roles),
            "system_role_policy": self.system_role_policy,
            "reasoning_capability": self.reasoning_capability,
            "selected_reasoning_mode": self.selected_reasoning_mode,
            "template_kwargs": dict(self.template_kwargs),
            "prompt_injects_reasoning_start": self.prompt_injects_reasoning_start,
            "generated_reasoning_delimiters": self.generated_reasoning_delimiters,
            "assistant_turn_end_mode": self.assistant_turn_end_mode,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def qwen3_instruct_2507_profile() -> ModelIOProfile:
    """Native fixed non-thinking contract for Qwen3-Instruct-2507."""

    return ModelIOProfile(
        profile_id="qwen3_instruct_2507_native_chat_v1",
        renderer_id="hf_native_chat_template_v1",
        supported_roles=("system", "user", "assistant", "tool"),
        system_role_policy="native",
        reasoning_capability="fixed_off",
        selected_reasoning_mode="off",
        template_kwargs={},
        prompt_injects_reasoning_start=False,
        generated_reasoning_delimiters="none",
        assistant_turn_end_mode="infer_from_empty_assistant",
    )


def qwen2_5_prompt_reasoning_profile() -> ModelIOProfile:
    """Qwen2.5 profile whose actor prompt requests paired reasoning tags.

    Qwen2.5 has no chat-template reasoning switch.  Treating it as either
    template-switchable Qwen3 or a checkpoint-fixed thinker misstates where
    the delimiters originate.
    """

    return ModelIOProfile(
        profile_id="qwen2_5_prompt_instructed_paired_reasoning_v1",
        renderer_id="hf_native_chat_template_v1",
        supported_roles=("system", "user", "assistant", "tool"),
        system_role_policy="native",
        reasoning_capability="prompt_instructed",
        selected_reasoning_mode="on",
        template_kwargs={},
        prompt_injects_reasoning_start=False,
        generated_reasoning_delimiters="paired",
        assistant_turn_end_mode="infer_from_empty_assistant",
    )


def qwen2_5_instruct_profile() -> ModelIOProfile:
    """Native Qwen2.5-Instruct chat without prompt-injected reasoning tags."""

    return ModelIOProfile(
        profile_id="qwen2_5_instruct_native_chat_v1",
        renderer_id="hf_native_chat_template_v1",
        supported_roles=("system", "user", "assistant", "tool"),
        system_role_policy="native",
        reasoning_capability="fixed_off",
        selected_reasoning_mode="off",
        template_kwargs={},
        prompt_injects_reasoning_start=False,
        generated_reasoning_delimiters="none",
        assistant_turn_end_mode="infer_from_empty_assistant",
    )


def qwen3_2504_profile(*, reasoning_mode: ReasoningMode) -> ModelIOProfile:
    """Explicit experiment-visible use profile for switchable Qwen3-2504."""

    if reasoning_mode not in {"off", "on"}:
        raise ModelIOContractError("Qwen3-2504 reasoning mode must be explicit")
    enabled = reasoning_mode == "on"
    return ModelIOProfile(
        profile_id=(
            "qwen3_2504_native_chat_thinking_v1"
            if enabled
            else "qwen3_2504_native_chat_nonthinking_v1"
        ),
        renderer_id="hf_native_chat_template_v1",
        supported_roles=("system", "user", "assistant", "tool"),
        system_role_policy="native",
        reasoning_capability="switchable",
        selected_reasoning_mode=reasoning_mode,
        template_kwargs={"enable_thinking": enabled},
        prompt_injects_reasoning_start=False,
        generated_reasoning_delimiters="paired" if enabled else "none",
        assistant_turn_end_mode=(
            "infer_from_distinct_assistant_suffix" if enabled else "infer_from_empty_assistant"
        ),
    )


def qwen3_thinking_2507_profile() -> ModelIOProfile:
    """Fixed-thinking Qwen3-2507 whose prompt owns the opening delimiter."""

    return ModelIOProfile(
        profile_id="qwen3_thinking_2507_native_chat_v1",
        renderer_id="hf_native_chat_template_v1",
        supported_roles=("system", "user", "assistant", "tool"),
        system_role_policy="native",
        reasoning_capability="fixed_on",
        selected_reasoning_mode="on",
        template_kwargs={},
        prompt_injects_reasoning_start=True,
        generated_reasoning_delimiters="closing_only",
        assistant_turn_end_mode="infer_from_distinct_assistant_suffix",
    )


def gemma3_it_profile() -> ModelIOProfile:
    """Gemma instruction-tuned role contract; system text folds into user."""

    return ModelIOProfile(
        profile_id="gemma3_it_native_chat_v1",
        renderer_id="hf_native_chat_template_v1",
        supported_roles=("user", "assistant"),
        system_role_policy="fold_into_first_user",
        reasoning_capability="fixed_off",
        selected_reasoning_mode="off",
        template_kwargs={},
        prompt_injects_reasoning_start=False,
        generated_reasoning_delimiters="none",
        assistant_turn_end_mode="infer_from_empty_assistant",
    )


def model_io_profile_by_id(profile_id: str) -> ModelIOProfile:
    factories = {
        "qwen2_5_instruct_native_chat_v1": qwen2_5_instruct_profile,
        "qwen2_5_prompt_instructed_paired_reasoning_v1": (qwen2_5_prompt_reasoning_profile),
        "qwen3_instruct_2507_native_chat_v1": qwen3_instruct_2507_profile,
        "qwen3_2504_native_chat_nonthinking_v1": lambda: qwen3_2504_profile(reasoning_mode="off"),
        "qwen3_2504_native_chat_thinking_v1": lambda: qwen3_2504_profile(reasoning_mode="on"),
        "qwen3_thinking_2507_native_chat_v1": qwen3_thinking_2507_profile,
        "gemma3_it_native_chat_v1": gemma3_it_profile,
    }
    try:
        return factories[str(profile_id)]()
    except KeyError as error:
        raise ModelIOContractError(f"unregistered model I/O profile: {profile_id}") from error


__all__ = [
    "ModelIOProfile",
    "gemma3_it_profile",
    "model_io_profile_by_id",
    "qwen2_5_instruct_profile",
    "qwen2_5_prompt_reasoning_profile",
    "qwen3_2504_profile",
    "qwen3_instruct_2507_profile",
    "qwen3_thinking_2507_profile",
]
