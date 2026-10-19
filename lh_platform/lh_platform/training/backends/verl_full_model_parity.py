"""One frozen full-model update through local and isolated veRL backends."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.training.agentic_rl_runtime import (
    PolicyActionRequest,
    canonical_sha256,
)
from lh_platform.training.backends.local_transformers import LocalTransformersBackend
from lh_platform.training.backends.verl import VerLBackend
from lh_platform.training.experience_bridge import (
    ExperienceRecord,
    experience_batch_sha256,
)
from lh_platform.training.model_io import (
    AssistantTarget,
    ConversationIR,
    build_model_io_backend_parity_receipt,
    require_model_io_backend_parity,
)
from lh_platform.training.runtime_preflight import await_exclusive_gpu
from lh_platform.training.transformers_peft_policy import (
    TransformersPeftPolicy,
    TransformersPeftPolicyConfig,
    encode_policy_prompt,
)

VERL_FULL_MODEL_PARITY_SCHEMA = "lh_verl_full_model_frozen_update_parity/v2"


@dataclass(frozen=True)
class FullModelParityReceipt:
    schema: str
    passed: bool
    scope: str
    preflight_path: str
    preflight_sha256: str
    policy_config_sha256: str
    experience_batch_sha256: str
    local_backend: Mapping[str, Any]
    verl_backend: Mapping[str, Any]
    local_update: Mapping[str, Any]
    verl_update: Mapping[str, Any]
    local_checkpoint: Mapping[str, Any]
    verl_checkpoint: Mapping[str, Any]
    model_io_backend_parity: Mapping[str, Any]
    loss_absolute_error: float
    grad_norm_absolute_error: float
    parameter_delta_absolute_error: float
    checkpoint_identity_exact: bool
    tolerance: float


def policy_config_from_preflight(
    preflight: Mapping[str, Any],
) -> TransformersPeftPolicyConfig:
    """Restore the persisted policy without changing its typed tuple fields."""

    canonical = preflight.get("policy_config")
    if isinstance(canonical, Mapping):
        payload = dict(canonical)
        for field_name in ("adamw_betas", "lora_target_modules", "lora_exclude_modules"):
            if field_name in payload:
                payload[field_name] = tuple(payload[field_name])
        config = TransformersPeftPolicyConfig(**payload)
        expected_sha256 = preflight.get("policy_config_sha256")
        if (
            not isinstance(expected_sha256, str)
            or canonical_sha256(asdict(config)) != expected_sha256
        ):
            raise ValueError("preflight canonical policy config hash mismatch")
        top_level_checkpoint = preflight.get("base_checkpoint_sha256")
        if top_level_checkpoint is not None and config.base_checkpoint_sha256 != str(
            top_level_checkpoint
        ):
            raise ValueError("preflight policy config base checkpoint mismatch")
        return config

    resolved = preflight.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("preflight has no resolved_config mapping")
    return TransformersPeftPolicyConfig(
        model_path=str(resolved["model_path"]),
        base_checkpoint_sha256=str(preflight["base_checkpoint_sha256"]),
        model_revision=str(resolved.get("configured_model_revision", "local")),
        device=str(resolved.get("device", "cuda")),
        dtype=str(resolved.get("dtype", "bfloat16")),
        learning_rate=float(resolved.get("learning_rate", 1e-5)),
        weight_decay=float(resolved.get("weight_decay", 0.0)),
        clip_ratio=float(resolved.get("clip_ratio", 0.2)),
        reference_kl_coef=float(resolved.get("reference_kl_coef", 0.0)),
        reference_kl_type=str(resolved.get("reference_kl_type", "low_var_kl")),
        max_grad_norm=float(resolved.get("max_grad_norm", 1.0)),
        max_prompt_tokens=int(resolved.get("max_prompt_tokens", 2048)),
        max_new_tokens=int(resolved.get("max_action_tokens", 32)),
        temperature=float(resolved.get("train_temperature", 1.0)),
        top_p=float(resolved.get("top_p", 1.0)),
        top_k=(int(resolved["top_k"]) if resolved.get("top_k") is not None else None),
        repetition_penalty=(
            float(resolved["repetition_penalty"])
            if resolved.get("repetition_penalty") is not None
            else None
        ),
        apply_chat_template=bool(resolved.get("apply_chat_template", False)),
        generation_micro_batch_size=int(resolved.get("generation_micro_batch_size", 1)),
        update_micro_batch_size=int(resolved.get("update_micro_batch_size", 1)),
        lora_r=int(resolved.get("lora_r", 8)),
        lora_alpha=int(resolved.get("lora_alpha", 16)),
        lora_dropout=float(resolved.get("lora_dropout", 0.0)),
        lora_target_modules=tuple(resolved.get("lora_target_modules", ("q_proj", "v_proj"))),
        lora_exclude_modules=tuple(resolved.get("lora_exclude_modules", ())),
        model_loader=str(resolved.get("model_loader", "auto_causal_lm")),
        quantization=str(resolved.get("quantization", "none")),
        quantization_double=bool(resolved.get("quantization_double", False)),
        attn_implementation=str(resolved.get("attn_implementation", "sdpa")),
        use_liger_kernel=bool(resolved.get("use_liger_kernel", False)),
        liger_kernel_profile=str(resolved.get("liger_kernel_profile", "full")),
        initialization_seed=int(resolved.get("initialization_seed", 0)),
        adapter_path=(
            str(resolved["adapter_path"]) if resolved.get("adapter_path") is not None else None
        ),
        adapter_checkpoint_sha256=(
            str(resolved["adapter_checkpoint_sha256"])
            if resolved.get("adapter_checkpoint_sha256") is not None
            else None
        ),
        adapter_artifact_sha256=(
            str(resolved["adapter_artifact_sha256"])
            if resolved.get("adapter_artifact_sha256") is not None
            else None
        ),
    )


def run_full_model_parity(
    *,
    preflight_path: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
    tolerance: float = 1e-5,
    allow_shared_gpu: bool = False,
    wait_for_gpu_seconds: float = 0.0,
) -> FullModelParityReceipt:
    preflight_path = Path(preflight_path).resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    config = policy_config_from_preflight(preflight)
    admit_full_model_parity_runtime(
        config,
        allow_shared_gpu=allow_shared_gpu,
        wait_for_gpu_seconds=wait_for_gpu_seconds,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    local_policy = TransformersPeftPolicy(config)
    local_backend = LocalTransformersBackend(local_policy)
    try:
        model_io_request = build_model_io_parity_request(config)
        local_model_io_samples = local_policy.sample_actions(
            (model_io_request,),
            seed=0,
        )
        record = _build_frozen_record(local_policy)
        records = (record,)
        local_backend.submit(records)
        local_update = local_backend.update()
        local_checkpoint = local_backend.checkpoint(output_dir / "local")
    finally:
        local_policy.close()
    del local_policy
    import torch

    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()

    verl_backend = VerLBackend(
        source_root=source_root,
        policy_payload={"kind": "transformers_peft", "config": asdict(config)},
    )
    try:
        if verl_backend.checkpoint_sha256 != record.policy_checkpoint_sha256:
            raise RuntimeError("isolated veRL policy did not reproduce initial checkpoint")
        verl_model_io_samples = verl_backend.sample_actions(
            (model_io_request,),
            seed=0,
        )
        model_io_backend_parity = build_model_io_backend_parity_receipt(
            local_model_io_samples,
            verl_model_io_samples,
            compared_backend="verl",
        )
        require_model_io_backend_parity(model_io_backend_parity)
        verl_backend.submit(records)
        verl_update = verl_backend.update()
        verl_checkpoint = verl_backend.checkpoint(output_dir / "verl")
        verl_identity = asdict(verl_backend.identity)
    finally:
        verl_backend.close()

    loss_error = abs(float(local_update.metrics["loss"]) - float(verl_update.metrics["loss"]))
    grad_error = abs(
        float(local_update.metrics["grad_norm"]) - float(verl_update.metrics["grad_norm"])
    )
    delta_error = abs(
        local_update.trainable_parameter_delta_l2 - verl_update.trainable_parameter_delta_l2
    )
    checkpoint_exact = (
        local_update.checkpoint_after_sha256
        == verl_update.checkpoint_after_sha256
        == local_checkpoint.sha256
        == verl_checkpoint.sha256
    )
    passed = (
        loss_error <= tolerance
        and grad_error <= tolerance
        and delta_error <= tolerance
        and checkpoint_exact
        and model_io_backend_parity["passed"] is True
    )
    receipt = FullModelParityReceipt(
        schema=VERL_FULL_MODEL_PARITY_SCHEMA,
        passed=passed,
        scope=(
            "single frozen exact-token full-model update; not live environment rollout "
            "or policy improvement"
        ),
        preflight_path=str(preflight_path),
        preflight_sha256=_sha256_file(preflight_path),
        policy_config_sha256=canonical_sha256(asdict(config)),
        experience_batch_sha256=experience_batch_sha256(records),
        local_backend=asdict(local_backend.identity),
        verl_backend=verl_identity,
        local_update=asdict(local_update),
        verl_update=asdict(verl_update),
        local_checkpoint=asdict(local_checkpoint),
        verl_checkpoint=asdict(verl_checkpoint),
        model_io_backend_parity=model_io_backend_parity,
        loss_absolute_error=loss_error,
        grad_norm_absolute_error=grad_error,
        parameter_delta_absolute_error=delta_error,
        checkpoint_identity_exact=checkpoint_exact,
        tolerance=tolerance,
    )
    atomic_write_json(output_dir / "receipt.json", asdict(receipt))
    return receipt


def admit_full_model_parity_runtime(
    config: TransformersPeftPolicyConfig,
    *,
    allow_shared_gpu: bool,
    wait_for_gpu_seconds: float,
    admission_fn=await_exclusive_gpu,
) -> None:
    """Apply the shared GPU admission contract before model or output creation."""

    admission_fn(
        device=config.device,
        allow_shared_gpu=allow_shared_gpu,
        timeout_seconds=wait_for_gpu_seconds,
    )


def build_model_io_parity_request(
    config: TransformersPeftPolicyConfig,
) -> PolicyActionRequest:
    """Build one deterministic pre-update sample shared by local and veRL.

    A one-action trie removes policy-choice variance while preserving exact
    prompt rendering, response tokens, termination, masks, log-prob width, and
    model-I/O fingerprints. Optimizer parity remains a separate frozen batch.
    """

    contract_id = "full_model_parity_native_turn/v1"
    contract_sha256 = canonical_sha256(
        {
            "contract_id": contract_id,
            "termination": "checkpoint_native_assistant_turn_end",
            "admissible_actions": ["look"],
        }
    )
    kwargs: dict[str, Any]
    if config.model_io_profile is not None:
        kwargs = {
            "conversation": ConversationIR.from_legacy_user_prompt(
                "Verify exact model-I/O backend transport."
            ),
            "model_io_profile_id": str(config.model_io_profile["profile_id"]),
        }
    else:
        kwargs = {"prompt_text": "Verify exact model-I/O backend transport."}
    return PolicyActionRequest(
        traj_uid="full-model-parity-model-io",
        step_idx=0,
        response_contract_id=contract_id,
        response_contract_sha256=contract_sha256,
        admissible_actions=("look",),
        response_stop_texts=(),
        **kwargs,
    )


def _build_frozen_record(policy: TransformersPeftPolicy) -> ExperienceRecord:
    prompt = (
        "Task: verify one exact-token backend update.\n"
        "Observation: You are in a test room.\n"
        "Admissible actions: look\n"
    )
    renderer = getattr(policy, "_model_io_renderer", None)
    if renderer is not None:
        turn = renderer.render_assistant_turn(
            ConversationIR.from_legacy_user_prompt(prompt),
            AssistantTarget("look"),
        )
        prompt_ids = turn.prompt_ids
        action_ids = turn.response_ids
        response_mask = turn.response_mask
    else:
        action = "<think>Inspect the room.</think><action>look</action>"
        encoded_prompt = encode_policy_prompt(
            policy.tokenizer,
            prompt,
            apply_chat_template=policy.config.apply_chat_template,
            return_tensors="pt",
        )
        prompt_ids = tuple(int(token) for token in encoded_prompt["input_ids"][0].tolist())
        action_ids = tuple(
            int(token)
            for token in policy.tokenizer(
                action,
                add_special_tokens=False,
            )["input_ids"]
        )
        response_mask = (1,) * len(action_ids)
    if not action_ids:
        raise RuntimeError("full-model parity action tokenization is empty")
    provisional = ExperienceRecord(
        traj_uid="full-model-parity-traj",
        group_uid="full-model-parity-group",
        step_idx=0,
        prompt_ids=prompt_ids,
        response_ids=action_ids,
        response_mask=response_mask,
        response_old_logprobs=(0.0,) * len(action_ids),
        step_advantage=0.75,
        actor_token_weight=1.0 / len(action_ids),
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        rollout_seed=0,
        sampling_temperature=policy.config.temperature,
        is_last_episode=True,
    )
    with policy._torch.no_grad():
        (old_logprobs, _entropy) = policy._teacher_forced_batch_log_probs((provisional,))[0]
    return replace(
        provisional,
        response_old_logprobs=tuple(
            float(value) for value in old_logprobs.detach().float().cpu().tolist()
        ),
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "VERL_FULL_MODEL_PARITY_SCHEMA",
    "FullModelParityReceipt",
    "admit_full_model_parity_runtime",
    "build_model_io_parity_request",
    "policy_config_from_preflight",
    "run_full_model_parity",
]
