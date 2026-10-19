"""Local Transformers + PEFT implementation of the trainable-policy boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import re
import shutil
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from lh_platform.training.agentic_rl_components import builtin_component
from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    CheckpointArtifact,
    ComponentRef,
    OnPolicyActionSample,
    PolicyActionRequest,
    PolicyUpdateResult,
    token_ids_sha256,
    validate_action_sample_for_request,
)
from lh_platform.training.experience_bridge import (
    ExperienceRecord,
    validate_experience_records,
)
from lh_platform.training.model_io import (
    ConversationIR,
    ModelIOContractError,
    ModelIOProfile,
    NativeChatRenderer,
)
from lh_platform.training.supervised_experience import (
    SupervisedExperienceRecord,
    validate_supervised_experience_records,
)

logger = logging.getLogger(__name__)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PARAMETER_CHECKPOINT_IDENTITY_SCHEMA = "lh_parameter_checkpoint_identity/v1"
RUNTIME_POLICY_IDENTITY_SCHEMA = "lh_runtime_policy_identity/v1"


def _canonical_identity_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"schema": schema, **dict(payload)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _RenderedPolicyRequest:
    input_ids: tuple[int, ...]
    model_io_profile_sha256: str | None
    normalized_messages_sha256: str | None
    chat_template_sha256: str | None


@dataclass(frozen=True)
class _TeacherForcedTokenRecord:
    """Minimal exact-token view shared by RL and supervised batches."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    allowed_response_token_ids: tuple[tuple[int, ...], ...] = ()
    sampling_temperature: float = 1.0


@dataclass(frozen=True)
class TransformersPeftPolicyConfig:
    model_path: str
    base_checkpoint_sha256: str
    model_revision: str = "local"
    device: str = "cuda"
    dtype: str = "bfloat16"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_epsilon: float = 1e-8
    adamw_amsgrad: bool = False
    adamw_maximize: bool = False
    adamw_foreach: bool | None = None
    adamw_fused: bool | None = None
    adamw_capturable: bool = False
    adamw_differentiable: bool = False
    clip_ratio: float = 0.2
    clip_ratio_high: float | None = None
    reference_kl_coef: float = 0.0
    reference_kl_type: str = "low_var_kl"
    max_grad_norm: float = 1.0
    max_prompt_tokens: int = 2048
    max_new_tokens: int = 32
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float | None = None
    apply_chat_template: bool = False
    generation_micro_batch_size: int = 1
    update_micro_batch_size: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    lora_exclude_modules: tuple[str, ...] = ()
    model_loader: str = "auto_causal_lm"
    quantization: str = "none"
    quantization_double: bool = False
    allocator_warmup: bool = True
    supervised_gradient_checkpointing: bool = False
    attn_implementation: str = "sdpa"
    use_liger_kernel: bool = False
    liger_kernel_profile: str = "full"
    entropy_coef: float = 0.0
    initialization_seed: int = 0
    adapter_path: str | None = None
    adapter_checkpoint_sha256: str | None = None
    adapter_artifact_sha256: str | None = None
    model_io_profile: Mapping[str, Any] | None = None
    model_io_profile_sha256: str | None = None
    response_contract_registry_revision: str = "lh_response_contract_registry/v1"
    inference_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_path:
            raise AgenticRLContractError("local policy model_path is required")
        if not isinstance(self.inference_only, bool):
            raise AgenticRLContractError("inference_only must be boolean")
        if not _SHA256.fullmatch(self.base_checkpoint_sha256):
            raise AgenticRLContractError("base_checkpoint_sha256 must bind the local model source")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise AgenticRLContractError("unsupported local policy dtype")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise AgenticRLContractError("optimizer learning rate/weight decay is invalid")
        if (
            len(self.adamw_betas) != 2
            or not 0 <= self.adamw_betas[0] < 1
            or not 0 <= self.adamw_betas[1] < 1
            or not math.isfinite(self.adamw_epsilon)
            or self.adamw_epsilon <= 0
        ):
            raise AgenticRLContractError("AdamW beta/epsilon configuration is invalid")
        for name, value in (
            ("adamw_amsgrad", self.adamw_amsgrad),
            ("adamw_maximize", self.adamw_maximize),
            ("adamw_capturable", self.adamw_capturable),
            ("adamw_differentiable", self.adamw_differentiable),
        ):
            if not isinstance(value, bool):
                raise AgenticRLContractError(f"{name} must be boolean")
        if self.adamw_foreach is not None and not isinstance(self.adamw_foreach, bool):
            raise AgenticRLContractError("adamw_foreach must be boolean or None")
        if self.adamw_fused is not None and not isinstance(self.adamw_fused, bool):
            raise AgenticRLContractError("adamw_fused must be boolean or None")
        if self.adamw_foreach is True and self.adamw_fused is True:
            raise AgenticRLContractError("AdamW foreach and fused cannot both be true")
        if self.adamw_fused is True and self.adamw_differentiable:
            raise AgenticRLContractError("fused AdamW cannot be differentiable")
        if not 0 < self.clip_ratio < 1 or self.max_grad_norm <= 0:
            raise AgenticRLContractError("PPO clip or gradient norm is invalid")
        if self.clip_ratio_high is not None and not (self.clip_ratio < self.clip_ratio_high < 1):
            raise AgenticRLContractError(
                "clip_ratio_high must satisfy clip_ratio < clip_ratio_high < 1"
            )
        if self.reference_kl_coef < 0:
            raise AgenticRLContractError("reference KL coefficient must be non-negative")
        if self.reference_kl_type != "low_var_kl":
            raise AgenticRLContractError("only low_var_kl reference loss is supported")
        if (
            self.max_prompt_tokens < 1
            or self.max_new_tokens < 1
            or self.temperature <= 0
            or self.top_p != 1.0
        ):
            raise AgenticRLContractError("sampling configuration is invalid")
        if (self.top_k is None) != (self.repetition_penalty is None):
            raise AgenticRLContractError(
                "sampling warpers must be either both legacy or both explicit"
            )
        if self.top_k is not None and (self.top_k != 0 or self.repetition_penalty != 1.0):
            raise AgenticRLContractError(
                "on-policy teacher forcing requires full-support neutral sampling warpers"
            )
        if (
            isinstance(self.generation_micro_batch_size, bool)
            or self.generation_micro_batch_size < 1
        ):
            raise AgenticRLContractError("generation micro batch size is invalid")
        if isinstance(self.update_micro_batch_size, bool) or self.update_micro_batch_size < 1:
            raise AgenticRLContractError("update micro batch size is invalid")
        if self.lora_r < 1 or self.lora_alpha < 1 or not self.lora_target_modules:
            raise AgenticRLContractError("LoRA configuration is invalid")
        if not math.isfinite(self.lora_dropout) or self.lora_dropout != 0.0:
            raise AgenticRLContractError("exact-token policy updates require lora_dropout=0")
        if self.model_loader not in {
            "auto_causal_lm",
            "gemma3_conditional_generation",
        }:
            raise AgenticRLContractError("unsupported Transformers model loader")
        if self.quantization not in {"none", "bitsandbytes_nf4"}:
            raise AgenticRLContractError("unsupported base-model quantization")
        if not isinstance(self.allocator_warmup, bool):
            raise AgenticRLContractError("allocator_warmup must be boolean")
        if not self.allocator_warmup and self.quantization != "bitsandbytes_nf4":
            raise AgenticRLContractError(
                "allocator warmup may only be disabled for bitsandbytes NF4 loading"
            )
        if not isinstance(self.supervised_gradient_checkpointing, bool):
            raise AgenticRLContractError("supervised_gradient_checkpointing must be boolean")
        if self.inference_only and self.supervised_gradient_checkpointing:
            raise AgenticRLContractError("inference-only policy forbids gradient checkpointing")
        if self.attn_implementation not in {
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_4",
        }:
            raise AgenticRLContractError(
                "attn_implementation must be eager, sdpa, flash_attention_2, or flash_attention_4"
            )
        if self.liger_kernel_profile not in {"full", "qwen3_swiglu_only"}:
            raise AgenticRLContractError("unsupported Liger kernel profile")
        if not self.use_liger_kernel and self.liger_kernel_profile != "full":
            raise AgenticRLContractError(
                "a selective Liger kernel profile requires use_liger_kernel=True"
            )
        if self.entropy_coef < 0:
            raise AgenticRLContractError("entropy_coef must be non-negative")
        if self.quantization == "bitsandbytes_nf4":
            if not self.device.startswith("cuda"):
                raise AgenticRLContractError("NF4 QLoRA requires a CUDA device")
            if self.dtype != "bfloat16":
                raise AgenticRLContractError("NF4 QLoRA compute dtype must be bfloat16")
        if self.model_loader == "gemma3_conditional_generation" and not (
            {"vision_tower", "multi_modal_projector"} <= set(self.lora_exclude_modules)
        ):
            raise AgenticRLContractError(
                "Gemma3 text-only LoRA must exclude vision_tower and multi_modal_projector"
            )
        adapter_fields = (
            self.adapter_path,
            self.adapter_checkpoint_sha256,
            self.adapter_artifact_sha256,
        )
        if any(value is None for value in adapter_fields) and any(
            value is not None for value in adapter_fields
        ):
            raise AgenticRLContractError(
                "adapter_path, adapter_checkpoint_sha256, and adapter_artifact_sha256 "
                "must be set together"
            )
        if self.adapter_checkpoint_sha256 is not None and not _SHA256.fullmatch(
            self.adapter_checkpoint_sha256
        ):
            raise AgenticRLContractError("adapter checkpoint identity must be SHA256")
        if self.adapter_artifact_sha256 is not None and not _SHA256.fullmatch(
            self.adapter_artifact_sha256
        ):
            raise AgenticRLContractError("adapter artifact identity must be SHA256")
        if (self.model_io_profile is None) != (self.model_io_profile_sha256 is None):
            raise AgenticRLContractError(
                "model I/O profile and profile hash must be configured together"
            )
        if self.model_io_profile is not None:
            try:
                profile = ModelIOProfile.from_dict(self.model_io_profile)
            except ModelIOContractError as error:
                raise AgenticRLContractError(
                    f"policy model I/O profile is invalid: {error}"
                ) from error
            if profile.sha256 != self.model_io_profile_sha256:
                raise AgenticRLContractError("policy model I/O profile hash drifted")
        if self.response_contract_registry_revision != "lh_response_contract_registry/v1":
            raise AgenticRLContractError("response contract registry revision is unsupported")


def model_load_plan(config: TransformersPeftPolicyConfig) -> dict[str, Any]:
    """Expose the exact loader/quantization contract before allocating a GPU."""

    plan = {
        "model_loader": config.model_loader,
        "model_revision": config.model_revision,
        "quantization": config.quantization,
        "compute_dtype": config.dtype,
        "double_quantization": config.quantization_double,
        "prepare_model_for_kbit_training": (
            config.quantization == "bitsandbytes_nf4" and not config.inference_only
        ),
        "text_only_trainable": config.model_loader == "gemma3_conditional_generation",
        "attn_implementation": config.attn_implementation,
        "use_liger_kernel": config.use_liger_kernel,
        "lora_target_modules": list(config.lora_target_modules),
        "lora_exclude_modules": list(config.lora_exclude_modules),
    }
    if not config.allocator_warmup:
        plan["allocator_warmup"] = False
    if config.inference_only:
        plan["inference_only"] = True
        plan["optimizer_present"] = False
    if config.use_liger_kernel:
        plan["liger_kernel_profile"] = config.liger_kernel_profile
    if config.supervised_gradient_checkpointing:
        plan["supervised_gradient_checkpointing"] = True
    if config.adapter_checkpoint_sha256 is not None:
        plan["initial_adapter_checkpoint_sha256"] = config.adapter_checkpoint_sha256
        plan["initial_adapter_artifact_sha256"] = config.adapter_artifact_sha256
    return plan


def policy_dependency_preflight(
    config: TransformersPeftPolicyConfig,
    *,
    module_available: Any = None,
) -> dict[str, Any]:
    """Validate local architecture and dependencies before any model allocation."""

    def default_module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    available = module_available or default_module_available
    required = {"torch", "transformers", "peft"}
    if config.quantization == "bitsandbytes_nf4":
        required.update({"accelerate", "bitsandbytes"})
    if config.use_liger_kernel:
        required.add("liger_kernel")
    if config.attn_implementation == "flash_attention_2":
        required.add("flash_attn.flash_attn_interface")
    if config.attn_implementation == "flash_attention_4":
        required.add("flash_attn.cute")
    missing = sorted(name for name in required if not available(name))
    if missing:
        raise AgenticRLContractError(
            "policy backend dependencies are missing: " + ", ".join(missing)
        )
    if config.adapter_path is not None:
        manifest_path = Path(config.adapter_path) / "checkpoint_manifest.json"
        try:
            adapter_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AgenticRLContractError(
                f"cannot read initial adapter manifest: {error}"
            ) from error
        if adapter_manifest.get("base_checkpoint_sha256") != config.base_checkpoint_sha256:
            raise AgenticRLContractError("initial adapter binds a different base checkpoint")
        if adapter_manifest.get("checkpoint_sha256") != config.adapter_checkpoint_sha256:
            raise AgenticRLContractError("initial adapter checkpoint identity mismatch")
        actual_artifact_sha256 = adapter_artifact_tree_sha256(config.adapter_path)
        if adapter_manifest.get("adapter_artifact_sha256") != actual_artifact_sha256:
            raise AgenticRLContractError("initial adapter manifest does not bind its files")
        if actual_artifact_sha256 != config.adapter_artifact_sha256:
            raise AgenticRLContractError("initial adapter artifact identity mismatch")
        try:
            adapter_config = json.loads(
                (Path(config.adapter_path) / "adapter_config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise AgenticRLContractError(
                f"cannot read initial PEFT adapter config: {error}"
            ) from error
        _validate_adapter_config_payload(adapter_config, config)
    config_path = Path(config.model_path) / "config.json"
    try:
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AgenticRLContractError(
            f"cannot read local model architecture metadata: {error}"
        ) from error
    architectures = metadata.get("architectures")
    if not isinstance(architectures, list) or not all(
        isinstance(value, str) for value in architectures
    ):
        raise AgenticRLContractError("local model config lacks architectures")
    if (
        config.model_loader == "gemma3_conditional_generation"
        and "Gemma3ForConditionalGeneration" not in architectures
    ):
        raise AgenticRLContractError(
            "Gemma3 loader requires Gemma3ForConditionalGeneration architecture"
        )
    if config.liger_kernel_profile == "qwen3_swiglu_only" and not any(
        "qwen3" in architecture.lower() for architecture in architectures
    ):
        raise AgenticRLContractError(
            "qwen3_swiglu_only Liger profile requires a Qwen3 architecture"
        )
    receipt = {
        "schema": "lh_policy_dependency_preflight/v1",
        "required_modules": sorted(required),
        "architectures": architectures,
        "model_load_plan": model_load_plan(config),
    }
    return {
        **receipt,
        "sha256": hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def transformers_peft_component(config: TransformersPeftPolicyConfig) -> ComponentRef:
    """Return backend identity without embedding host-specific model paths."""

    component_config: dict[str, Any] = {
        "revision": "v1",
        "model_revision": config.model_revision,
        "device_type": config.device.split(":", 1)[0],
        "dtype": config.dtype,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "clip_ratio": config.clip_ratio,
        "reference_kl_coef": config.reference_kl_coef,
        "reference_kl_type": config.reference_kl_type,
        "reference_policy": "peft_disabled_frozen_base",
        "reference_kl_aggregation": "token_mean",
        "max_grad_norm": config.max_grad_norm,
        "max_new_tokens": config.max_new_tokens,
        "max_prompt_tokens": config.max_prompt_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "apply_chat_template": config.apply_chat_template,
        "generation_micro_batch_size": config.generation_micro_batch_size,
        "update_micro_batch_size": config.update_micro_batch_size,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target_modules": config.lora_target_modules,
        "lora_exclude_modules": config.lora_exclude_modules,
        "model_load_plan": model_load_plan(config),
        "initialization_seed": config.initialization_seed,
    }
    if config.adapter_checkpoint_sha256 is not None:
        component_config["initial_adapter_checkpoint_sha256"] = config.adapter_checkpoint_sha256
        component_config["initial_adapter_artifact_sha256"] = config.adapter_artifact_sha256
    if config.model_io_profile_sha256 is not None:
        component_config.update(
            {
                "model_io_profile_sha256": config.model_io_profile_sha256,
                "response_contract_registry_revision": (config.response_contract_registry_revision),
            }
        )
    if config.top_k is not None:
        component_config.update(
            {
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
            }
        )
    if (
        config.adamw_betas != (0.9, 0.999)
        or config.adamw_epsilon != 1e-8
        or config.adamw_amsgrad
        or config.adamw_maximize
        or config.adamw_foreach is not None
        or config.adamw_fused is not None
        or config.adamw_capturable
        or config.adamw_differentiable
    ):
        component_config["adamw_execution"] = {
            "betas": config.adamw_betas,
            "epsilon": config.adamw_epsilon,
            "amsgrad": config.adamw_amsgrad,
            "maximize": config.adamw_maximize,
            "foreach": config.adamw_foreach,
            "fused": config.adamw_fused,
            "capturable": config.adamw_capturable,
            "differentiable": config.adamw_differentiable,
        }
    if config.inference_only:
        component_config["inference_only"] = True
        component_config["optimizer_present"] = False
    return builtin_component(
        role="policy_backend",
        name=(
            "transformers_frozen_causal_lm"
            if config.inference_only
            else "transformers_peft_causal_lm"
        ),
        config=component_config,
    )


def checkpoint_tree_sha256(path: str | Path) -> str:
    """Stream a local checkpoint tree into one reviewable source digest."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = tuple(item for item in sorted(root.rglob("*")) if item.is_file())
    if not files:
        raise AgenticRLContractError("local checkpoint tree is empty")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def adapter_artifact_tree_sha256(path: str | Path) -> str:
    """Bind every saved adapter/tokenizer byte except the self-describing manifest."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    required = {"adapter_config.json"}
    names = {item.name for item in root.iterdir() if item.is_file()}
    if not required.issubset(names) or not (
        {"adapter_model.safetensors", "adapter_model.bin"} & names
    ):
        raise AgenticRLContractError("adapter artifact lacks PEFT config or weights")
    digest = hashlib.sha256()
    files = tuple(
        item
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.relative_to(root).as_posix() != "checkpoint_manifest.json"
    )
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _apply_liger_kernel(model_path: str, *, profile: str = "full") -> str:
    """Patch the exact Transformers architecture/profile or fail closed.

    The policy computes its own action-token objective from logits, so Liger's
    fused linear cross entropy is deliberately disabled. RMSNorm, RoPE, and
    SwiGLU remain eligible. A requested optimization must never silently fall
    back, because that would make the recorded execution factor false.
    """

    config_path = Path(model_path) / "config.json"
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AgenticRLContractError(
            f"use_liger_kernel=True but model architecture metadata is unavailable: {error}"
        ) from error
    arch = (model_config.get("architectures") or [""])[0].lower()
    if profile not in {"full", "qwen3_swiglu_only"}:
        raise AgenticRLContractError(f"unsupported Liger kernel profile {profile!r}")
    if profile == "qwen3_swiglu_only" and "qwen3" not in arch:
        raise AgenticRLContractError(
            "qwen3_swiglu_only Liger profile requires a Qwen3 architecture"
        )
    try:
        if "qwen3" in arch:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3

            patch = apply_liger_kernel_to_qwen3
            patch_name = "qwen3"
        elif "qwen2" in arch:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2

            patch = apply_liger_kernel_to_qwen2
            patch_name = "qwen2"
        elif "llama" in arch:
            from liger_kernel.transformers import apply_liger_kernel_to_llama

            patch = apply_liger_kernel_to_llama
            patch_name = "llama"
        elif "mistral" in arch:
            from liger_kernel.transformers import apply_liger_kernel_to_mistral

            patch = apply_liger_kernel_to_mistral
            patch_name = "mistral"
        elif "gemma2" in arch:
            from liger_kernel.transformers import apply_liger_kernel_to_gemma2

            patch = apply_liger_kernel_to_gemma2
            patch_name = "gemma2"
        else:
            raise AgenticRLContractError(
                f"use_liger_kernel=True but architecture {arch!r} has no admitted patch"
            )
    except ImportError as error:
        raise AgenticRLContractError(
            "use_liger_kernel=True but the required Liger architecture patch is unavailable"
        ) from error
    if profile == "qwen3_swiglu_only":
        patch(
            rope=False,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
            rms_norm=False,
            swiglu=True,
        )
        return "qwen3_swiglu_only"
    patch(fused_linear_cross_entropy=False)
    return patch_name


def _validate_adapter_config_payload(
    payload: dict[str, Any],
    config: TransformersPeftPolicyConfig,
) -> None:
    expected_targets = set(config.lora_target_modules)
    actual_targets = set(payload.get("target_modules") or ())
    if (
        int(payload.get("r", -1)) != config.lora_r
        or int(payload.get("lora_alpha", -1)) != config.lora_alpha
        or float(payload.get("lora_dropout", -1.0)) != config.lora_dropout
        or actual_targets != expected_targets
    ):
        raise AgenticRLContractError(
            "initial adapter PEFT structure differs from configured LoRA contract"
        )


def verify_zero_initialized_lora_delta(model: Any) -> bool:
    """Prove that the untrained adapter cannot change base-model behavior."""

    projections = [
        parameter
        for name, parameter in model.named_parameters()
        if ".lora_B." in name or ".lora_embedding_B" in name
    ]
    if not projections:
        raise AgenticRLContractError(
            "PEFT policy exposes no LoRA output projection for initialization audit"
        )
    if any(int(parameter.detach().count_nonzero().item()) != 0 for parameter in projections):
        raise AgenticRLContractError(
            "PEFT policy initialized a non-zero delta; base-model evaluation is invalid"
        )
    return True


def verify_text_only_trainable_parameters(model: Any) -> bool:
    """Fail if a Gemma3 adapter makes any vision/projector parameter trainable."""

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise AgenticRLContractError("text-only adapter exposes no trainable parameters")
    forbidden = [
        name for name in trainable if "vision_tower" in name or "multi_modal_projector" in name
    ]
    if forbidden:
        raise AgenticRLContractError(
            "Gemma3 text-only adapter exposed vision trainables: " + ", ".join(forbidden)
        )
    return True


def encode_policy_prompt(
    tokenizer: Any,
    prompt_text: str,
    *,
    apply_chat_template: bool,
    return_tensors: str,
) -> Any:
    """Encode the actor prompt with the same chat boundary as pinned G2PO."""

    rendered = render_policy_prompt(
        tokenizer,
        prompt_text,
        apply_chat_template=apply_chat_template,
    )
    return tokenizer(
        rendered,
        return_tensors=return_tensors,
        add_special_tokens=not apply_chat_template,
    )


def render_policy_prompt(
    tokenizer: Any,
    prompt_text: str,
    *,
    apply_chat_template: bool,
) -> str:
    """Render one prompt before either scalar or batched tokenization."""

    rendered = str(prompt_text)
    if apply_chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": rendered}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return str(rendered)


def decode_semantic_action_text(
    tokenizer: Any,
    token_ids: tuple[int, ...],
) -> str:
    """Keep special-token-only generations as an explicit invalid action."""

    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return decoded or "<EMPTY_MODEL_RESPONSE>"


def _truncate_response_at_stop_texts(
    tokenizer: Any,
    token_ids: tuple[int, ...],
    stop_texts: tuple[str, ...],
) -> tuple[tuple[int, ...], str | None]:
    """Keep the shortest sampled token prefix containing an action boundary.

    The returned token IDs remain an exact prefix of the behavior-policy
    sample, so their generation-time log probabilities can be sliced without
    decoding and re-tokenizing the action.
    """

    if not token_ids or not stop_texts:
        return token_ids, None

    def decode_prefix(length: int) -> str:
        return tokenizer.decode(
            token_ids[:length],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    full_text = decode_prefix(len(token_ids))
    matches = tuple(
        (full_text.find(stop_text), stop_text) for stop_text in stop_texts if stop_text in full_text
    )
    if not matches:
        return token_ids, None
    _, first_stop = min(matches, key=lambda item: (item[0], stop_texts.index(item[1])))
    lower = 1
    upper = len(token_ids)
    while lower < upper:
        middle = (lower + upper) // 2
        if first_stop in decode_prefix(middle):
            upper = middle
        else:
            lower = middle + 1
    return token_ids[:lower], first_stop


def low_variance_kl(current_logprobs: Any, reference_logprobs: Any) -> Any:
    """G2PO/veRL ``low_var_kl`` (k3) estimator on sampled tokens."""

    log_ratio = reference_logprobs - current_logprobs
    return (log_ratio.exp() - log_ratio - 1.0).clamp(min=-10.0, max=10.0)


class TransformersPeftPolicy:
    """Action-only clipped policy updates over exact rollout token spans."""

    def __init__(self, config: TransformersPeftPolicyConfig) -> None:
        self.config = config
        self.dependency_preflight = policy_dependency_preflight(config)
        self.component = transformers_peft_component(config)
        self._samples: list[OnPolicyActionSample] = []
        self._optimizer_step = 0
        self._closed = False

        try:
            import torch
            import transformers as transformers_package
            from peft import (
                LoraConfig,
                PeftModel,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                Gemma3ForConditionalGeneration,
            )
        except ImportError as error:  # pragma: no cover - remote dependency gate.
            raise AgenticRLContractError(
                "TransformersPeftPolicy requires torch, transformers, and peft"
            ) from error

        self._torch = torch
        self._device = torch.device(config.device)
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.dtype]
        torch.manual_seed(config.initialization_seed)
        if self._device.type == "cuda":
            if not torch.cuda.is_available():
                raise AgenticRLContractError("configured CUDA policy but CUDA is unavailable")
            torch.cuda.manual_seed_all(config.initialization_seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            revision=None if config.model_revision == "local" else config.model_revision,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self._model_io_renderer = (
            NativeChatRenderer(
                self.tokenizer,
                ModelIOProfile.from_dict(config.model_io_profile),
            )
            if config.model_io_profile is not None
            else None
        )
        self.model_io_profile_sha256 = (
            self._model_io_renderer.profile.sha256 if self._model_io_renderer is not None else None
        )
        self.chat_template_sha256 = (
            self._model_io_renderer.chat_template_sha256
            if self._model_io_renderer is not None
            else None
        )
        self.liger_patch = (
            _apply_liger_kernel(
                config.model_path,
                profile=config.liger_kernel_profile,
            )
            if config.use_liger_kernel
            else None
        )
        model_class = (
            Gemma3ForConditionalGeneration
            if config.model_loader == "gemma3_conditional_generation"
            else AutoModelForCausalLM
        )
        dtype_keyword = (
            "dtype"
            if int(transformers_package.__version__.split(".", 1)[0]) >= 5
            else "torch_dtype"
        )
        load_kwargs: dict[str, Any] = {
            "revision": None if config.model_revision == "local" else config.model_revision,
            "local_files_only": True,
            dtype_keyword: dtype,
            "attn_implementation": config.attn_implementation,
        }
        if config.quantization == "bitsandbytes_nf4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=config.quantization_double,
            )
            load_kwargs["device_map"] = {
                "": self._device.index if self._device.index is not None else 0
            }
        original_allocator_warmup = None
        if not config.allocator_warmup:
            original_allocator_warmup = transformers_package.modeling_utils.caching_allocator_warmup
            transformers_package.modeling_utils.caching_allocator_warmup = (
                lambda _model, _device_map, _quantizer: None
            )
        try:
            base = model_class.from_pretrained(config.model_path, **load_kwargs)
        finally:
            if original_allocator_warmup is not None:
                transformers_package.modeling_utils.caching_allocator_warmup = (
                    original_allocator_warmup
                )
        if config.quantization == "bitsandbytes_nf4" and not config.inference_only:
            base = prepare_model_for_kbit_training(
                base,
                use_gradient_checkpointing=config.supervised_gradient_checkpointing,
            )
        elif config.quantization != "bitsandbytes_nf4":
            base = base.to(self._device)
        if config.inference_only:
            if config.adapter_path is None:
                self.model = base
            else:
                self.model = PeftModel.from_pretrained(
                    base,
                    config.adapter_path,
                    is_trainable=False,
                )
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.adapter_initial_delta_zero = False
            self.optimizer = None
        else:
            if config.adapter_path is None:
                lora = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=list(config.lora_target_modules),
                    exclude_modules=list(config.lora_exclude_modules) or None,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(base, lora)
            else:
                self.model = PeftModel.from_pretrained(
                    base,
                    config.adapter_path,
                    is_trainable=True,
                )
                loaded_config = self.model.peft_config.get("default")
                _validate_adapter_config_payload(
                    {
                        "r": getattr(loaded_config, "r", None),
                        "lora_alpha": getattr(loaded_config, "lora_alpha", None),
                        "lora_dropout": getattr(loaded_config, "lora_dropout", None),
                        "target_modules": list(getattr(loaded_config, "target_modules", ()) or ()),
                    },
                    config,
                )
            if config.supervised_gradient_checkpointing:
                enable_checkpointing = getattr(
                    self.model,
                    "gradient_checkpointing_enable",
                    None,
                )
                if not callable(enable_checkpointing):
                    raise AgenticRLContractError(
                        "model does not support supervised gradient checkpointing"
                    )
                try:
                    enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    enable_checkpointing()
            if config.model_loader == "gemma3_conditional_generation":
                verify_text_only_trainable_parameters(self.model)
            self.adapter_initial_delta_zero = (
                verify_zero_initialized_lora_delta(self.model)
                if config.adapter_path is None
                else False
            )
            trainable = [
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ]
            if not trainable:
                raise AgenticRLContractError("PEFT policy exposes no trainable parameters")
            self.optimizer = torch.optim.AdamW(
                trainable,
                lr=config.learning_rate,
                betas=config.adamw_betas,
                eps=config.adamw_epsilon,
                weight_decay=config.weight_decay,
                amsgrad=config.adamw_amsgrad,
                maximize=config.adamw_maximize,
                foreach=config.adamw_foreach,
                fused=config.adamw_fused,
                capturable=config.adamw_capturable,
                differentiable=config.adamw_differentiable,
            )
        self.checkpoint_sha256 = self._checkpoint_sha256()

    def close(self) -> None:
        """Release model and optimizer storage at an explicit experiment boundary."""

        if getattr(self, "_closed", False):
            return
        # Mark closed first so cleanup is idempotent even when initialization
        # failed halfway through or one cleanup hook raises.
        self._closed = True
        cleanup_error: Exception | None = None
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception as error:
                cleanup_error = error
        samples = getattr(self, "_samples", None)
        if samples is not None:
            samples.clear()
        for field in ("optimizer", "model", "_model_io_renderer", "tokenizer"):
            if hasattr(self, field):
                delattr(self, field)
        torch = getattr(self, "_torch", None)
        device = getattr(self, "_device", None)
        if torch is not None and getattr(device, "type", None) == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as error:
                cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            raise cleanup_error

    def __del__(self) -> None:
        # Import machinery and torch internals may already be torn down during
        # interpreter finalization. Explicit experiment boundaries call close();
        # the destructor is only a best-effort fallback before shutdown begins.
        if sys.is_finalizing():
            return
        try:
            self.close()
        except Exception:
            logger.exception("transformers_peft_policy: close() failed during __del__")

    def sampling_cursor(self) -> int:
        return len(self._samples)

    def sampling_events_since(self, cursor: int) -> tuple[OnPolicyActionSample, ...]:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            return ()
        return tuple(self._samples[cursor:])

    def _forward_logits_without_cache(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None = None,
    ) -> Any:
        """Run a full-prefix scoring forward without retaining generation KV cache."""

        model_config = getattr(self.model, "config", None)
        has_cache_setting = hasattr(model_config, "use_cache")
        previous_use_cache = model_config.use_cache if has_cache_setting else None
        if has_cache_setting:
            model_config.use_cache = False
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        try:
            return self.model(**kwargs).logits
        finally:
            if has_cache_setting:
                model_config.use_cache = previous_use_cache

    def _render_action_request(
        self,
        request: PolicyActionRequest,
    ) -> _RenderedPolicyRequest:
        renderer = getattr(self, "_model_io_renderer", None)
        conversation = request.conversation
        if renderer is not None:
            if conversation is None:
                conversation = ConversationIR.from_legacy_user_prompt(str(request.prompt_text))
            if (
                request.model_io_profile_id is not None
                and request.model_io_profile_id != renderer.profile.profile_id
            ):
                raise AgenticRLContractError(
                    "policy request model I/O profile differs from loaded profile"
                )
            rendered = renderer.render_prompt(conversation)
            return _RenderedPolicyRequest(
                input_ids=rendered.input_ids,
                model_io_profile_sha256=rendered.model_io_profile_sha256,
                normalized_messages_sha256=rendered.normalized_messages_sha256,
                chat_template_sha256=rendered.chat_template_sha256,
            )

        if conversation is not None:
            try:
                prompt_text = conversation.to_legacy_user_prompt()
            except ModelIOContractError as error:
                raise AgenticRLContractError(
                    f"legacy policy cannot flatten conversation: {error}"
                ) from error
        else:
            prompt_text = str(request.prompt_text)
        encoded = encode_policy_prompt(
            self.tokenizer,
            prompt_text,
            apply_chat_template=self.config.apply_chat_template,
            return_tensors="pt",
        )
        return _RenderedPolicyRequest(
            input_ids=tuple(int(token) for token in encoded["input_ids"][0].tolist()),
            model_io_profile_sha256=None,
            normalized_messages_sha256=None,
            chat_template_sha256=None,
        )

    @staticmethod
    def _finish_reason(
        *,
        matched_stop_text: str | None,
        response_ids: tuple[int, ...],
        eos_token_id: int | None,
    ) -> str:
        if matched_stop_text is not None:
            return "response_contract_stop"
        if eos_token_id is not None and response_ids[-1:] == (int(eos_token_id),):
            return "native_eos"
        return "max_tokens_or_backend_stop"

    def _termination_ids_for_request(
        self,
        request: PolicyActionRequest,
    ) -> tuple[int, ...]:
        if request.response_stop_texts:
            if len(request.response_stop_texts) != 1:
                raise AgenticRLContractError(
                    "constrained action decoding requires exactly one explicit "
                    "response-contract stop sequence"
                )
            encoded = self.tokenizer(
                request.response_stop_texts[0],
                add_special_tokens=False,
            )
            token_ids = encoded["input_ids"]
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            while (
                isinstance(token_ids, (list, tuple))
                and len(token_ids) == 1
                and isinstance(token_ids[0], (list, tuple))
            ):
                token_ids = token_ids[0]
            termination = tuple(int(token_id) for token_id in token_ids)
            if not termination:
                raise AgenticRLContractError(
                    "response-contract stop sequence tokenized to an empty suffix"
                )
            return termination
        renderer = getattr(self, "_model_io_renderer", None)
        if renderer is not None:
            conversation = request.conversation
            if conversation is None:
                conversation = ConversationIR.from_legacy_user_prompt(str(request.prompt_text))
            return renderer.native_turn_end_ids(conversation)
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise AgenticRLContractError("legacy constrained action requires an EOS token")
        return (int(eos),)

    def sample_actions(
        self,
        requests: tuple[PolicyActionRequest, ...],
        *,
        seed: int,
    ) -> tuple[OnPolicyActionSample, ...]:
        torch = self._torch
        self.model.eval()
        samples: list[OnPolicyActionSample] = []
        temperatures = tuple(self._request_temperature(request) for request in requests)
        with torch.no_grad():
            if (
                self.config.generation_micro_batch_size > 1
                and requests
                and all(not request.admissible_actions for request in requests)
                and all(request.sampling_seed is None for request in requests)
                and len(set(temperatures)) == 1
            ):
                return self._sample_free_action_batches(
                    requests,
                    temperature=temperatures[0],
                    seed=seed,
                )
            for offset, request in enumerate(requests):
                local_seed = (
                    int(request.sampling_seed)
                    if request.sampling_seed is not None
                    else int(seed) + offset
                )
                torch.manual_seed(local_seed)
                if self._device.type == "cuda":
                    torch.cuda.manual_seed_all(local_seed)
                temperature = temperatures[offset]
                matched_stop_text = None
                rendered_request = self._render_action_request(request)
                input_ids = torch.tensor(
                    [rendered_request.input_ids],
                    dtype=torch.long,
                    device=self._device,
                )
                if input_ids.shape[1] > self.config.max_prompt_tokens:
                    raise AgenticRLContractError(
                        "prompt exceeds the locked source max_prompt_tokens"
                    )
                attention_mask = torch.ones_like(input_ids)
                prompt_ids = rendered_request.input_ids
                if request.admissible_actions:
                    (
                        action_text,
                        action_ids,
                        old_log_probs,
                        allowed_action_token_ids,
                    ) = self._sample_constrained_action(
                        input_ids,
                        request.admissible_actions,
                        termination_ids=self._termination_ids_for_request(request),
                        temperature=temperature,
                    )
                else:
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **self._generation_sampling_kwargs(temperature),
                        max_new_tokens=self.config.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    action_tensor = generated.sequences[0, input_ids.shape[1] :]
                    action_ids = tuple(int(token) for token in action_tensor.tolist())
                    if not action_ids:
                        raise AgenticRLContractError("local policy generated no action tokens")
                    action_ids, matched_stop_text = _truncate_response_at_stop_texts(
                        self.tokenizer,
                        action_ids,
                        request.response_stop_texts,
                    )
                    if len(generated.scores) < len(action_ids):
                        raise AgenticRLContractError(
                            "generation scores do not align with emitted action tokens"
                        )
                    old_log_probs = tuple(
                        float(
                            torch.log_softmax(score[0].float(), dim=-1)[action_ids[index]]
                            .detach()
                            .cpu()
                        )
                        for index, score in enumerate(generated.scores[: len(action_ids)])
                    )
                    allowed_action_token_ids = ()
                    action_text = decode_semantic_action_text(self.tokenizer, action_ids)
                if request.admissible_actions:
                    matched_stop_text = None
                sample = OnPolicyActionSample(
                    traj_uid=request.traj_uid,
                    step_idx=request.step_idx,
                    action_text=action_text,
                    prompt_token_ids=prompt_ids,
                    action_token_ids=action_ids,
                    old_action_token_log_probs=old_log_probs,
                    policy_checkpoint_sha256=self.checkpoint_sha256,
                    rollout_seed=local_seed,
                    response_mask=(1,) * len(action_ids),
                    model_io_profile_sha256=(rendered_request.model_io_profile_sha256),
                    response_contract_sha256=request.response_contract_sha256,
                    normalized_messages_sha256=(rendered_request.normalized_messages_sha256),
                    chat_template_sha256=rendered_request.chat_template_sha256,
                    matched_stop_text=matched_stop_text,
                    finish_reason=(
                        "constrained_action"
                        if request.admissible_actions
                        else self._finish_reason(
                            matched_stop_text=matched_stop_text,
                            response_ids=action_ids,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    ),
                    sampling_temperature=temperature,
                    allowed_action_token_ids=allowed_action_token_ids,
                )
                validate_action_sample_for_request(
                    request,
                    sample,
                    expected_model_io_profile_sha256=getattr(self, "model_io_profile_sha256", None),
                )
                samples.append(sample)
                self._samples.append(sample)
        return tuple(samples)

    def _request_temperature(self, request: PolicyActionRequest) -> float:
        temperature = (
            self.config.temperature
            if request.sampling_temperature is None
            else float(request.sampling_temperature)
        )
        if not math.isfinite(temperature) or temperature <= 0:
            raise AgenticRLContractError("request sampling temperature must be finite and positive")
        return temperature

    def _generation_sampling_kwargs(self, temperature: float) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": self.config.top_p,
        }
        if self.config.top_k is not None:
            kwargs.update(
                {
                    "top_k": self.config.top_k,
                    "repetition_penalty": self.config.repetition_penalty,
                }
            )
        return kwargs

    def _sample_free_action_batches(
        self,
        requests: tuple[PolicyActionRequest, ...],
        *,
        temperature: float,
        seed: int,
    ) -> tuple[OnPolicyActionSample, ...]:
        """Batch unrestricted source-style responses without changing projection."""

        torch = self._torch
        samples: list[OnPolicyActionSample] = []
        batch_size = self.config.generation_micro_batch_size
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            chunk_seed = int(seed) + start
            torch.manual_seed(chunk_seed)
            if self._device.type == "cuda":
                torch.cuda.manual_seed_all(chunk_seed)
            if getattr(self, "_model_io_renderer", None) is None:
                rendered_texts = [
                    render_policy_prompt(
                        self.tokenizer,
                        (
                            request.conversation.to_legacy_user_prompt()
                            if request.conversation is not None
                            else str(request.prompt_text)
                        ),
                        apply_chat_template=self.config.apply_chat_template,
                    )
                    for request in chunk
                ]
                encoded = self.tokenizer(
                    rendered_texts,
                    return_tensors="pt",
                    add_special_tokens=not self.config.apply_chat_template,
                    padding=True,
                )
                input_ids = encoded["input_ids"].to(self._device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = attention_mask.to(self._device)
                rendered_requests = tuple(
                    _RenderedPolicyRequest(
                        input_ids=tuple(
                            int(token)
                            for token in input_ids[row][attention_mask[row].bool()].tolist()
                        ),
                        model_io_profile_sha256=None,
                        normalized_messages_sha256=None,
                        chat_template_sha256=None,
                    )
                    for row in range(len(chunk))
                )
            else:
                rendered_requests = tuple(self._render_action_request(request) for request in chunk)
                prompt_width = max(len(row.input_ids) for row in rendered_requests)
                input_ids = torch.full(
                    (len(chunk), prompt_width),
                    int(self.tokenizer.pad_token_id),
                    dtype=torch.long,
                    device=self._device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row_index, rendered_request in enumerate(rendered_requests):
                    length = len(rendered_request.input_ids)
                    input_ids[row_index, prompt_width - length :] = torch.tensor(
                        rendered_request.input_ids,
                        dtype=torch.long,
                        device=self._device,
                    )
                    attention_mask[row_index, prompt_width - length :] = 1
            prompt_lengths = [len(row.input_ids) for row in rendered_requests]
            if any(int(length) > self.config.max_prompt_tokens for length in prompt_lengths):
                raise AgenticRLContractError("prompt exceeds the locked source max_prompt_tokens")
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self._generation_sampling_kwargs(temperature),
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            prompt_width = int(input_ids.shape[1])
            for row, request in enumerate(chunk):
                rendered_request = rendered_requests[row]
                prompt_ids = rendered_request.input_ids
                response_ids = tuple(
                    int(token) for token in generated.sequences[row, prompt_width:].tolist()
                )
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is not None and eos_token_id in response_ids:
                    response_ids = response_ids[: response_ids.index(eos_token_id) + 1]
                response_ids, matched_stop_text = _truncate_response_at_stop_texts(
                    self.tokenizer,
                    response_ids,
                    request.response_stop_texts,
                )
                if not response_ids:
                    raise AgenticRLContractError("local policy generated no action tokens")
                if len(generated.scores) < len(response_ids):
                    raise AgenticRLContractError(
                        "generation scores do not align with emitted action tokens"
                    )
                old_log_probs = tuple(
                    float(
                        torch.log_softmax(generated.scores[index][row].float(), dim=-1)[token]
                        .detach()
                        .cpu()
                    )
                    for index, token in enumerate(response_ids)
                )
                sample = OnPolicyActionSample(
                    traj_uid=request.traj_uid,
                    step_idx=request.step_idx,
                    action_text=decode_semantic_action_text(self.tokenizer, response_ids),
                    prompt_token_ids=prompt_ids,
                    action_token_ids=response_ids,
                    old_action_token_log_probs=old_log_probs,
                    policy_checkpoint_sha256=self.checkpoint_sha256,
                    rollout_seed=chunk_seed,
                    response_mask=(1,) * len(response_ids),
                    model_io_profile_sha256=(rendered_request.model_io_profile_sha256),
                    response_contract_sha256=request.response_contract_sha256,
                    normalized_messages_sha256=(rendered_request.normalized_messages_sha256),
                    chat_template_sha256=rendered_request.chat_template_sha256,
                    matched_stop_text=matched_stop_text,
                    finish_reason=self._finish_reason(
                        matched_stop_text=matched_stop_text,
                        response_ids=response_ids,
                        eos_token_id=self.tokenizer.eos_token_id,
                    ),
                    sampling_temperature=temperature,
                    allowed_action_token_ids=(),
                )
                validate_action_sample_for_request(
                    request,
                    sample,
                    expected_model_io_profile_sha256=getattr(self, "model_io_profile_sha256", None),
                )
                samples.append(sample)
                self._samples.append(sample)
        return tuple(samples)

    def audit_old_action_log_probs(
        self,
        samples: tuple[OnPolicyActionSample, ...],
        *,
        max_absolute_error: float,
        max_mean_absolute_error: float,
        required_token_coverage: float = 1.0,
    ) -> dict[str, Any]:
        """Recompute recorded behavior-policy probabilities without an update.

        Generation-time scores and update-time teacher forcing must describe the
        same action-token distribution. This gate checks that invariant at the
        exact sampled checkpoint, including constrained-action support and the
        recorded sampling temperature.
        """

        if not samples:
            raise AgenticRLContractError("old-logprob audit requires policy samples")
        for label, value in {
            "max_absolute_error": max_absolute_error,
            "max_mean_absolute_error": max_mean_absolute_error,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise AgenticRLContractError(f"{label} must be finite and non-negative")
        if (
            not math.isfinite(required_token_coverage)
            or required_token_coverage < 0
            or required_token_coverage > 1
        ):
            raise AgenticRLContractError("required_token_coverage must be finite and within [0, 1]")
        keys = [(sample.traj_uid, sample.step_idx) for sample in samples]
        if len(set(keys)) != len(keys):
            raise AgenticRLContractError("old-logprob audit received duplicate sample keys")

        checkpoint_before = self._checkpoint_sha256()
        if checkpoint_before != self.checkpoint_sha256:
            raise AgenticRLContractError(
                "policy parameters differ from the recorded checkpoint before audit"
            )
        self.model.eval()
        errors: list[float] = []
        sample_summaries: list[dict[str, Any]] = []
        with self._torch.no_grad():
            for sample in samples:
                if sample.policy_checkpoint_sha256 != checkpoint_before:
                    raise AgenticRLContractError(
                        "old-logprob audit sample belongs to another policy checkpoint"
                    )
                if (
                    not sample.prompt_token_ids
                    or not sample.action_token_ids
                    or len(sample.old_action_token_log_probs) != len(sample.action_token_ids)
                ):
                    raise AgenticRLContractError(
                        "old-logprob audit sample lacks aligned token evidence"
                    )
                recomputed = self._teacher_forced_action_log_probs(sample)
                old = self._torch.tensor(
                    sample.old_action_token_log_probs,
                    dtype=self._torch.float32,
                    device=self._device,
                )
                token_errors = (recomputed.float() - old).abs().detach().cpu()
                values = [float(value) for value in token_errors.tolist()]
                errors.extend(values)
                sample_summaries.append(
                    {
                        "traj_uid": sample.traj_uid,
                        "step_idx": sample.step_idx,
                        "action_token_count": len(values),
                        "max_absolute_error": max(values),
                        "mean_absolute_error": sum(values) / len(values),
                    }
                )
        checkpoint_after = self._checkpoint_sha256()
        if checkpoint_after != checkpoint_before:
            raise AgenticRLContractError("old-logprob audit mutated policy parameters")

        maximum = max(errors)
        mean = sum(errors) / len(errors)
        expected_token_count = sum(len(sample.action_token_ids) for sample in samples)
        token_coverage = len(errors) / expected_token_count
        return {
            "schema": "lh_teacher_forced_old_logprob_audit/v1",
            "sample_count": len(samples),
            "action_token_count": len(errors),
            "checkpoint_before_sha256": checkpoint_before,
            "checkpoint_after_sha256": checkpoint_after,
            "max_absolute_error_threshold": max_absolute_error,
            "max_mean_absolute_error_threshold": max_mean_absolute_error,
            "required_token_coverage": required_token_coverage,
            "token_coverage": token_coverage,
            "max_absolute_error": maximum,
            "mean_absolute_error": mean,
            "passed": (
                maximum <= max_absolute_error
                and mean <= max_mean_absolute_error
                and token_coverage >= required_token_coverage
            ),
            "samples": sample_summaries,
            "claim_scope": "old_policy_probability_consistency_not_policy_improvement",
        }

    def score_admissible_actions(
        self,
        prompt_text: str,
        admissible_actions: tuple[str, ...],
        *,
        temperature: float,
    ) -> dict[str, Any]:
        """Score every admissible action under one frozen token-trie policy.

        This is a diagnostic candidate-choice distribution. It deliberately
        does not claim to marginalize over the source actor's free-form
        ``<think>`` text.
        """

        if not math.isfinite(temperature) or temperature <= 0:
            raise AgenticRLContractError("action scoring temperature must be finite and positive")
        checkpoint_before = self._checkpoint_sha256()
        if checkpoint_before != self.checkpoint_sha256:
            raise AgenticRLContractError(
                "policy parameters differ from the registered scoring checkpoint"
            )
        request = PolicyActionRequest(
            traj_uid="frozen-action-scoring",
            step_idx=0,
            prompt_text=str(prompt_text),
        )
        termination_ids = self._termination_ids_for_request(request)
        sequences = self._admissible_action_token_sequences(admissible_actions)
        rendered_request = self._render_action_request(request)
        prompt_ids = rendered_request.input_ids
        if len(prompt_ids) > self.config.max_prompt_tokens:
            raise AgenticRLContractError("prompt exceeds the locked source max_prompt_tokens")

        paths = {tokens: (*tokens, *termination_ids) for tokens in sequences}
        self._validate_admissible_action_path_lengths(tuple(paths.values()))
        outgoing: dict[tuple[int, ...], set[int]] = {}
        for path in paths.values():
            for index, token in enumerate(path):
                outgoing.setdefault(path[:index], set()).add(token)
        prefixes = tuple(sorted(outgoing, key=lambda item: (len(item), item)))
        edge_log_probs: dict[tuple[tuple[int, ...], int], float] = {}
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            batch_size = self.config.generation_micro_batch_size
            prefixes_by_depth: dict[int, list[tuple[int, ...]]] = {}
            for prefix in prefixes:
                prefixes_by_depth.setdefault(len(prefix), []).append(prefix)
            for depth in sorted(prefixes_by_depth):
                same_depth = prefixes_by_depth[depth]
                for start in range(0, len(same_depth), batch_size):
                    chunk = same_depth[start : start + batch_size]
                    input_ids = torch.tensor(
                        [prompt_ids + prefix for prefix in chunk],
                        dtype=torch.long,
                        device=self._device,
                    )
                    logits = self._forward_logits_without_cache(input_ids=input_ids)[:, -1].float()
                    for row, prefix in enumerate(chunk):
                        allowed = tuple(sorted(outgoing[prefix]))
                        allowed_ids = torch.tensor(
                            allowed,
                            dtype=torch.long,
                            device=self._device,
                        )
                        local = torch.log_softmax(
                            logits[row, allowed_ids] / temperature,
                            dim=-1,
                        )
                        for position, token in enumerate(allowed):
                            edge_log_probs[(prefix, token)] = float(local[position].detach().cpu())

        rows: list[dict[str, Any]] = []
        for tokens, action in sorted(sequences.items(), key=lambda item: item[1]):
            path = paths[tokens]
            log_probability = sum(
                edge_log_probs[(path[:index], token)] for index, token in enumerate(path)
            )
            rows.append(
                {
                    "action": action,
                    "action_token_ids": list(tokens),
                    "action_token_count": len(tokens),
                    "termination_token_id": (
                        termination_ids[0] if len(termination_ids) == 1 else None
                    ),
                    "termination_token_ids": list(termination_ids),
                    "scored_token_ids": list(path),
                    "scored_token_count": len(path),
                    "log_probability": log_probability,
                    "probability": math.exp(log_probability),
                }
            )
        probability_sum = sum(row["probability"] for row in rows)
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise AgenticRLContractError("admissible action trie probabilities do not normalize")
        checkpoint_after = self._checkpoint_sha256()
        if checkpoint_after != checkpoint_before:
            raise AgenticRLContractError("action scoring mutated policy parameters")
        return {
            "schema": "lh_frozen_admissible_action_distribution/v1",
            "policy_checkpoint_sha256": checkpoint_before,
            "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
            "prompt_token_count": len(prompt_ids),
            "temperature": temperature,
            "normalization": "per_prefix_admissible_token_trie",
            "probability_sum": probability_sum,
            "actions": rows,
            "claim_scope": (
                "frozen_token_trie_candidate_distribution_not_free_text_action_marginal"
            ),
        }

    def encode_frozen_actor_texts(
        self,
        texts: tuple[str, ...],
        *,
        representation: str,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode actor-visible text without changing the frozen checkpoint.

        ``last_hidden`` follows the same prompt serialization contract used by
        generation. ``input_embedding_mean`` is reserved for canonical action
        probe strings and deliberately does not apply a chat template.
        """

        if representation not in {"last_hidden", "input_embedding_mean"}:
            raise AgenticRLContractError("unsupported frozen actor representation")
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise AgenticRLContractError("frozen actor encoding requires non-empty text")
        checkpoint_before = self._checkpoint_sha256()
        if checkpoint_before != self.checkpoint_sha256:
            raise AgenticRLContractError(
                "policy parameters differ from the registered encoding checkpoint"
            )
        torch = self._torch
        output: list[tuple[float, ...]] = []
        batch_size = self.config.generation_micro_batch_size
        self.model.eval()
        with torch.no_grad():
            if representation == "last_hidden":
                sequences = []
                for text in texts:
                    encoded = encode_policy_prompt(
                        self.tokenizer,
                        text,
                        apply_chat_template=self.config.apply_chat_template,
                        return_tensors="pt",
                    )
                    token_ids = tuple(int(value) for value in encoded["input_ids"][0])
                    sequences.append(token_ids[-self.config.max_prompt_tokens :])
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    raise AgenticRLContractError("actor encoding requires a pad token")
                for start in range(0, len(sequences), batch_size):
                    chunk = sequences[start : start + batch_size]
                    width = max(len(row) for row in chunk)
                    input_ids = torch.tensor(
                        [(int(pad_token_id),) * (width - len(row)) + row for row in chunk],
                        dtype=torch.long,
                        device=self._device,
                    )
                    attention_mask = torch.tensor(
                        [(0,) * (width - len(row)) + (1,) * len(row) for row in chunk],
                        dtype=torch.long,
                        device=self._device,
                    )
                    result = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    hidden_states = getattr(result, "hidden_states", None)
                    if not hidden_states:
                        raise AgenticRLContractError("policy model did not expose hidden states")
                    for vector in hidden_states[-1][:, -1].float().cpu():
                        output.append(tuple(float(value) for value in vector))
            else:
                embedding = self.model.get_input_embeddings()
                for start in range(0, len(texts), batch_size):
                    chunk = texts[start : start + batch_size]
                    tokenized = self.tokenizer(
                        list(chunk),
                        add_special_tokens=False,
                        padding=True,
                        return_tensors="pt",
                    )
                    input_ids = tokenized["input_ids"].to(self._device)
                    attention_mask = tokenized["attention_mask"].to(self._device)
                    if input_ids.numel() == 0 or (attention_mask.sum(dim=1) == 0).any():
                        raise AgenticRLContractError(
                            "action probe text tokenized to an empty sequence"
                        )
                    vectors = (embedding(input_ids).float() * attention_mask.unsqueeze(-1)).sum(
                        dim=1
                    ) / attention_mask.sum(dim=1, keepdim=True)
                    output.extend(
                        tuple(float(value) for value in vector) for vector in vectors.cpu()
                    )
        if self._checkpoint_sha256() != checkpoint_before:
            raise AgenticRLContractError("actor encoding mutated policy parameters")
        return tuple(output)

    def _teacher_forced_action_log_probs(
        self,
        sample: OnPolicyActionSample,
    ) -> Any:
        """Return differentiable action-token logprobs for one recorded sample."""

        return self._teacher_forced_token_log_probs(
            prompt_ids=sample.prompt_token_ids,
            response_ids=sample.action_token_ids,
            allowed_response_token_ids=sample.allowed_action_token_ids,
            sampling_temperature=sample.sampling_temperature,
        )

    def _teacher_forced_token_log_probs(
        self,
        *,
        prompt_ids: tuple[int, ...],
        response_ids: tuple[int, ...],
        allowed_response_token_ids: tuple[tuple[int, ...], ...],
        sampling_temperature: float,
    ) -> Any:
        """Score exact token IDs without decode/retokenize."""

        torch = self._torch
        input_ids = torch.tensor(
            [prompt_ids + response_ids],
            dtype=torch.long,
            device=self._device,
        )
        logits = self._forward_logits_without_cache(input_ids=input_ids)[0]
        start = len(prompt_ids) - 1
        positions = torch.arange(
            start,
            start + len(response_ids),
            device=self._device,
        )
        selected_logits = logits[positions].float()
        log_probs, _entropy = self._response_log_probs_from_logits(
            selected_logits=selected_logits,
            response_ids=response_ids,
            allowed_response_token_ids=allowed_response_token_ids,
            sampling_temperature=sampling_temperature,
        )
        return log_probs

    def _response_log_probs_from_logits(
        self,
        *,
        selected_logits: Any,
        response_ids: tuple[int, ...],
        allowed_response_token_ids: tuple[tuple[int, ...], ...],
        sampling_temperature: float,
    ) -> tuple[Any, Any]:
        """Project response-position logits onto the recorded policy support.

        Returns (log_probs, entropy) per response token.
        """

        torch = self._torch
        action_ids = torch.tensor(
            response_ids,
            dtype=torch.long,
            device=self._device,
        )
        if allowed_response_token_ids:
            constrained_log_probs = []
            constrained_entropy = []
            for token_index, allowed in enumerate(allowed_response_token_ids):
                if response_ids[token_index] not in allowed:
                    raise AgenticRLContractError(
                        "recorded action token is outside constrained policy support"
                    )
                allowed_ids = torch.tensor(
                    allowed,
                    dtype=torch.long,
                    device=self._device,
                )
                allowed_logits = selected_logits[token_index, allowed_ids] / sampling_temperature
                log_probs = torch.log_softmax(allowed_logits, dim=-1)
                selected_position = allowed.index(response_ids[token_index])
                constrained_log_probs.append(log_probs[selected_position])
                probs = log_probs.exp()
                constrained_entropy.append(-(probs * log_probs).sum())
            return torch.stack(constrained_log_probs), torch.stack(constrained_entropy)
        scaled_logits = selected_logits / sampling_temperature
        log_probs = torch.log_softmax(scaled_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        token_log_probs = log_probs.gather(1, action_ids.unsqueeze(1)).squeeze(1)
        return token_log_probs, entropy

    def _teacher_forced_batch_log_probs(
        self,
        records: tuple[ExperienceRecord | _TeacherForcedTokenRecord, ...],
    ) -> tuple[Any, ...]:
        """Score a micro-batch without changing any per-step token identity."""

        if not records:
            return ()
        torch = self._torch
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise AgenticRLContractError("batched update requires a pad token")
        sequences = tuple(row.prompt_ids + row.response_ids for row in records)
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(records), max_length),
            int(pad_token_id),
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.zeros(
            (len(records), max_length),
            dtype=torch.long,
            device=self._device,
        )
        for batch_index, sequence in enumerate(sequences):
            width = len(sequence)
            input_ids[batch_index, :width] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=self._device,
            )
            attention_mask[batch_index, :width] = 1
        logits = self._forward_logits_without_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        outputs = []
        for batch_index, row in enumerate(records):
            start = len(row.prompt_ids) - 1
            stop = start + len(row.response_ids)
            outputs.append(
                self._response_log_probs_from_logits(
                    selected_logits=logits[batch_index, start:stop].float(),
                    response_ids=row.response_ids,
                    allowed_response_token_ids=row.allowed_response_token_ids,
                    sampling_temperature=row.sampling_temperature,
                )
            )
        return tuple(outputs)

    def _reference_token_log_probs(
        self,
        *,
        prompt_ids: tuple[int, ...],
        response_ids: tuple[int, ...],
        allowed_response_token_ids: tuple[tuple[int, ...], ...],
        sampling_temperature: float,
    ) -> Any:
        """Score the frozen base policy without allocating a second model."""

        disable_adapter = getattr(self.model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise AgenticRLContractError(
                "reference KL requires a PEFT model with disable_adapter()"
            )
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            with self._torch.no_grad(), disable_adapter():
                return self._teacher_forced_token_log_probs(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    allowed_response_token_ids=allowed_response_token_ids,
                    sampling_temperature=sampling_temperature,
                ).detach()
        finally:
            if was_training:
                self.model.train()

    def _reference_batch_log_probs(
        self,
        records: tuple[ExperienceRecord, ...],
    ) -> tuple[Any, ...]:
        """Score one micro-batch under the frozen PEFT base policy."""

        disable_adapter = getattr(self.model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise AgenticRLContractError(
                "reference KL requires a PEFT model with disable_adapter()"
            )
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            with self._torch.no_grad(), disable_adapter():
                return tuple(
                    log_probs.detach()
                    for log_probs, _entropy in self._teacher_forced_batch_log_probs(records)
                )
        finally:
            if was_training:
                self.model.train()

    def apply_experience_records(
        self,
        records: tuple[ExperienceRecord, ...],
    ) -> PolicyUpdateResult:
        """Apply one externally prepared exact-token clipped-PG batch."""

        if self.config.inference_only:
            raise AgenticRLContractError("inference-only policy forbids experience updates")
        records = validate_experience_records(records)
        torch = self._torch

        before_sha256 = self.checkpoint_sha256
        if any(row.policy_checkpoint_sha256 != before_sha256 for row in records):
            raise AgenticRLContractError(
                "experience batch was not sampled by the current policy checkpoint"
            )
        before_parameters = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        # Gradients remain enabled in eval mode.  Disabling dropout is required
        # so the PPO ratio and gradient do not depend on micro-batch partitioning.
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        policy_objective_value = 0.0
        kl_loss_value = 0.0
        entropy_value = 0.0
        action_tokens = sum(row.active_token_count for row in records)
        for start in range(0, len(records), self.config.update_micro_batch_size):
            micro_batch = records[start : start + self.config.update_micro_batch_size]
            current_batch = self._teacher_forced_batch_log_probs(micro_batch)
            reference_batch = (
                self._reference_batch_log_probs(micro_batch)
                if self.config.reference_kl_coef > 0
                else (None,) * len(micro_batch)
            )
            micro_batch_loss = torch.zeros((), device=self._device)
            for row, current_and_entropy, reference in zip(
                micro_batch,
                current_batch,
                reference_batch,
                strict=True,
            ):
                current, token_entropy = current_and_entropy
                old = torch.tensor(
                    row.response_old_logprobs,
                    dtype=torch.float32,
                    device=self._device,
                )
                mask = torch.tensor(
                    row.response_mask,
                    dtype=torch.bool,
                    device=self._device,
                )
                current = current[mask]
                old = old[mask]
                ratio = torch.exp(current - old)
                advantage = torch.tensor(
                    row.step_advantage,
                    dtype=torch.float32,
                    device=self._device,
                )
                unclipped = ratio * advantage
                clip_low = 1.0 - self.config.clip_ratio
                clip_high = 1.0 + (
                    self.config.clip_ratio_high
                    if self.config.clip_ratio_high is not None
                    else self.config.clip_ratio
                )
                clipped = torch.clamp(ratio, clip_low, clip_high) * advantage
                objective = row.actor_token_weight * torch.minimum(unclipped, clipped).sum()
                micro_batch_loss = micro_batch_loss - objective
                if self.config.entropy_coef > 0:
                    row_entropy = token_entropy[mask].sum() / action_tokens
                    micro_batch_loss = micro_batch_loss - self.config.entropy_coef * row_entropy
                    entropy_value += float(row_entropy.detach().cpu())
                if reference is not None:
                    row_kl = low_variance_kl(current, reference[mask]).sum() / action_tokens
                    micro_batch_loss = micro_batch_loss + self.config.reference_kl_coef * row_kl
                    kl_loss_value += float(row_kl.detach().cpu())
                policy_objective_value += float(objective.detach().cpu())
            micro_batch_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self._optimizer_step += 1
        delta_sq = 0.0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            delta = parameter.detach().float().cpu() - before_parameters[name]
            delta_sq += float(torch.sum(delta * delta))
        delta_l2 = math.sqrt(delta_sq)
        self.checkpoint_sha256 = self._checkpoint_sha256()
        return PolicyUpdateResult(
            optimizer_step=self._optimizer_step,
            checkpoint_before_sha256=before_sha256,
            checkpoint_after_sha256=self.checkpoint_sha256,
            trainable_parameter_delta_l2=delta_l2,
            metrics={
                "loss": (
                    -policy_objective_value
                    + self.config.reference_kl_coef * kl_loss_value
                    - self.config.entropy_coef * entropy_value
                ),
                "objective": policy_objective_value,
                "policy_objective": policy_objective_value,
                "reference_kl_loss": kl_loss_value,
                "reference_kl_coef": self.config.reference_kl_coef,
                "entropy": entropy_value,
                "entropy_coef": self.config.entropy_coef,
                "grad_norm": float(grad_norm.detach().float().cpu()),
                "action_tokens": float(action_tokens),
            },
        )

    def apply_supervised_records(
        self,
        records: tuple[SupervisedExperienceRecord, ...],
    ) -> PolicyUpdateResult:
        """Apply one response-token-only supervised competence update.

        No reward, advantage, state identity, or episode outcome is accepted by
        this seam.  Normalizing once by the full response-token count makes the
        result independent of the configured micro-batch partition.
        """

        if self.config.inference_only:
            raise AgenticRLContractError("inference-only policy forbids supervised updates")
        records = validate_supervised_experience_records(records)
        torch = self._torch
        before_sha256 = self.checkpoint_sha256
        before_parameters = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        response_tokens = sum(row.active_token_count for row in records)
        if response_tokens < 1:
            raise AgenticRLContractError("supervised batch has no response tokens")
        if self.config.supervised_gradient_checkpointing:
            self.model.train()
        else:
            self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for start in range(0, len(records), self.config.update_micro_batch_size):
            micro_batch = records[start : start + self.config.update_micro_batch_size]
            scoring_rows = tuple(
                _TeacherForcedTokenRecord(
                    prompt_ids=row.prompt_token_ids,
                    response_ids=row.response_token_ids,
                )
                for row in micro_batch
            )
            scored = self._teacher_forced_batch_log_probs(scoring_rows)
            micro_batch_loss = torch.zeros((), device=self._device)
            for token_log_probs, _token_entropy in scored:
                row_loss = -token_log_probs.sum() / response_tokens
                micro_batch_loss = micro_batch_loss + row_loss
            micro_batch_loss.backward()
            loss_value += float(micro_batch_loss.detach().cpu())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self._optimizer_step += 1
        delta_sq = 0.0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            delta = parameter.detach().float().cpu() - before_parameters[name]
            delta_sq += float(torch.sum(delta * delta))
        delta_l2 = math.sqrt(delta_sq)
        self.checkpoint_sha256 = self._checkpoint_sha256()
        return PolicyUpdateResult(
            optimizer_step=self._optimizer_step,
            checkpoint_before_sha256=before_sha256,
            checkpoint_after_sha256=self.checkpoint_sha256,
            trainable_parameter_delta_l2=delta_l2,
            metrics={
                "loss": loss_value,
                "supervised_response_ce": loss_value,
                "grad_norm": float(grad_norm.detach().float().cpu()),
                "response_tokens": float(response_tokens),
                "examples": float(len(records)),
            },
        )

    def materialize_checkpoint(self, output_dir: Path) -> CheckpointArtifact:
        if self.config.inference_only:
            raise AgenticRLContractError(
                "inference-only policy cannot materialize a trainable adapter"
            )
        target = Path(output_dir) / "checkpoint"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise AgenticRLContractError(f"checkpoint destination already exists: {target}")
        staging = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        staging.mkdir()
        try:
            self.model.save_pretrained(staging, safe_serialization=True)
            self.tokenizer.save_pretrained(staging)
            adapter_artifact_sha256 = adapter_artifact_tree_sha256(staging)
            manifest = staging / "checkpoint_manifest.json"
            from lh_platform.core.atomic_io import (
                atomic_publish_directory,
                atomic_write_json,
            )

            atomic_write_json(
                manifest,
                {
                    "schema": "lh_transformers_peft_checkpoint/v1",
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "base_checkpoint_sha256": self.config.base_checkpoint_sha256,
                    "initial_adapter_checkpoint_sha256": (self.config.adapter_checkpoint_sha256),
                    "adapter_artifact_sha256": adapter_artifact_sha256,
                    "optimizer_steps": self._optimizer_step,
                    "policy_backend": asdict(self.component),
                },
            )
            atomic_publish_directory(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return CheckpointArtifact(
            path=str(target / "checkpoint_manifest.json"),
            sha256=self.checkpoint_sha256,
        )

    def _sample_constrained_action(
        self,
        prompt_input_ids: Any,
        admissible_actions: tuple[str, ...],
        *,
        termination_ids: tuple[int, ...],
        temperature: float,
    ) -> tuple[str, tuple[int, ...], tuple[float, ...], tuple[tuple[int, ...], ...]]:
        """Sample from a token trie over environment-admissible action strings.

        Uses KV cache incremental decoding: one full-prefix forward for the
        prompt, then single-token forwards for each trie step.
        """

        torch = self._torch
        if not termination_ids:
            raise AgenticRLContractError(
                "constrained action sampling requires termination token IDs"
            )
        sequences = self._admissible_action_token_sequences(admissible_actions)
        paths = {(*tokens, *termination_ids): action for tokens, action in sequences.items()}
        self._validate_admissible_action_path_lengths(tuple(paths))

        model_config = getattr(self.model, "config", None)
        has_cache_setting = hasattr(model_config, "use_cache")
        previous_use_cache = model_config.use_cache if has_cache_setting else None
        if has_cache_setting:
            model_config.use_cache = True

        try:
            prompt_ids = prompt_input_ids[0].tolist()
            current_ids = torch.tensor(
                [tuple(int(t) for t in prompt_ids)],
                dtype=torch.long,
                device=self._device,
            )
            outputs = self.model(input_ids=current_ids)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1].float()

            prefix: tuple[int, ...] = ()
            selected_tokens: list[int] = []
            selected_log_probs: list[float] = []
            supports: list[tuple[int, ...]] = []
            for _ in range(self.config.max_new_tokens):
                allowed: set[int] = set()
                for path in paths:
                    if path[: len(prefix)] != prefix or len(path) == len(prefix):
                        continue
                    allowed.add(path[len(prefix)])
                if not allowed:
                    raise AgenticRLContractError("admissible-action token trie became empty")
                allowed_ids = tuple(sorted(allowed))
                allowed_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=self._device)
                allowed_logits = logits[allowed_tensor] / temperature
                log_probabilities = torch.log_softmax(allowed_logits, dim=-1)
                probabilities = torch.exp(log_probabilities)
                chosen_index = int(torch.multinomial(probabilities, num_samples=1).item())
                token = allowed_ids[chosen_index]
                selected_tokens.append(token)
                selected_log_probs.append(float(log_probabilities[chosen_index].cpu()))
                supports.append(allowed_ids)
                prefix = (*prefix, token)
                if prefix in paths:
                    return (
                        paths[prefix],
                        tuple(selected_tokens),
                        tuple(selected_log_probs),
                        tuple(supports),
                    )
                next_ids = torch.tensor([[token]], dtype=torch.long, device=self._device)
                outputs = self.model(input_ids=next_ids, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1].float()
        finally:
            if has_cache_setting:
                model_config.use_cache = previous_use_cache
        raise AgenticRLContractError("no admissible action terminated within max_new_tokens")

    def _admissible_action_token_sequences(
        self,
        admissible_actions: tuple[str, ...],
    ) -> dict[tuple[int, ...], str]:
        sequences: dict[tuple[int, ...], str] = {}
        for action in admissible_actions:
            text = str(action).strip()
            if not text:
                continue
            tokens = tuple(
                int(token)
                for token in self.tokenizer(
                    text,
                    add_special_tokens=False,
                )["input_ids"]
            )
            if tokens in sequences and sequences[tokens] != text:
                raise AgenticRLContractError(
                    "distinct admissible actions collapse to the same token sequence"
                )
            if tokens:
                sequences[tokens] = text
        if not sequences:
            raise AgenticRLContractError("environment exposed no tokenizable admissible actions")
        return sequences

    def _validate_admissible_action_path_lengths(
        self,
        paths: tuple[tuple[int, ...], ...],
    ) -> None:
        if any(len(path) > self.config.max_new_tokens for path in paths):
            raise AgenticRLContractError(
                "admissible action plus native termination exceeds max_new_tokens; "
                "increase the locked action budget"
            )

    def _trainable_parameter_state_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"lh_trainable_parameter_state/v1\x00")
        for name, parameter in sorted(self.model.named_parameters()):
            if not parameter.requires_grad:
                continue
            encoded_name = name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            tensor = parameter.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(self._torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _checkpoint_sha256(self) -> str:
        """Identify parameter state without conflating it with the inference backend."""

        if self.config.inference_only and self.config.adapter_path is None:
            return self.config.base_checkpoint_sha256
        adapter_state_sha256 = self._trainable_parameter_state_sha256()
        if self.config.inference_only:
            adapter_identity = (
                self.config.adapter_artifact_sha256
                or self.config.adapter_checkpoint_sha256
                or adapter_state_sha256
            )
        else:
            adapter_identity = adapter_state_sha256
        return _canonical_identity_sha256(
            PARAMETER_CHECKPOINT_IDENTITY_SCHEMA,
            {
                "base_checkpoint_tree_sha256": self.config.base_checkpoint_sha256,
                "adapter_parameter_state_sha256": adapter_identity,
            },
        )

    def runtime_policy_identity_sha256(self) -> str:
        """Bind parameter state to behavior-relevant runtime interpretation."""

        return _canonical_identity_sha256(
            RUNTIME_POLICY_IDENTITY_SCHEMA,
            {
                "parameter_checkpoint_sha256": self._checkpoint_sha256(),
                "backend_component_sha256": self.component.sha256,
                "model_io_profile_sha256": self.model_io_profile_sha256,
                "chat_template_sha256": self.chat_template_sha256,
            },
        )

    def policy_identity_breakdown(self) -> dict[str, Any]:
        """Expose parameter, implementation, and runtime identities separately."""

        parameter_checkpoint = self._checkpoint_sha256()
        runtime_identity = self.runtime_policy_identity_sha256()
        return {
            "base_checkpoint_tree_sha256": self.config.base_checkpoint_sha256,
            "backend_revision": self.component.revision,
            "backend_component_sha256": self.component.sha256,
            "adapter_parameter_state_sha256": self._trainable_parameter_state_sha256(),
            "parameter_checkpoint_sha256": parameter_checkpoint,
            "runtime_policy_identity_sha256": runtime_identity,
            "composite_policy_identity_sha256": runtime_identity,
        }
