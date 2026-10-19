"""Shared agentic-RL harness: GPU admission, policy config, rollout contract.

Callers that only need process/GPU admission or a Transformers policy config
must import from this module — not from ``agentic_rl_p0``, which owns the
paired G2PO-TN vs Verifier-Refined experiment only.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.training.agentic_rl_config import AgenticRLConfigError
from lh_platform.training.g2po.actor_protocol import (
    CONTEXT_EXTENSION_PLACEMENT_REVISION,
)
from lh_platform.training.transformers_peft_policy import TransformersPeftPolicyConfig


def agentic_rl_rollout_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the hashed rollout contract without rewriting legacy identities."""

    rollout_config: dict[str, Any] = {
        "group_size": args.group_size,
        "max_episode_steps": args.max_episode_steps,
        "task_groups_per_update": args.train_task_groups,
        "train_task_selection": args.train_task_selection,
        "action_space": "free_text_then_source_alfworld_projection",
        "history_length": args.history_length,
        "actor_protocol_id": args.actor_protocol_id,
        "show_admissible_actions": bool(getattr(args, "show_admissible_actions", True)),
        "actor_awareness_mode": args.actor_awareness_mode,
        "actor_context_extension_revision": (
            CONTEXT_EXTENSION_PLACEMENT_REVISION if args.actor_awareness_mode != "none" else "none"
        ),
        "train_temperature": args.train_temperature,
        "eval_temperature": args.eval_temperature,
        "max_response_tokens": args.max_action_tokens,
        "generation_micro_batch_size": int(getattr(args, "generation_micro_batch_size", 1)),
    }
    if getattr(args, "top_k", None) is not None:
        rollout_config.update(
            {
                "actor_profile_id": args.actor_profile_id,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            }
        )
    return rollout_config


def validate_agentic_rl_execution_admission(args: argparse.Namespace) -> None:
    """Reject diagnostic-only sampling profiles before any policy update."""

    if getattr(
        args, "execution_mode", "paired_train_eval"
    ) == "controlled_diagnostic_only" and not bool(getattr(args, "preflight_only", False)):
        raise AgenticRLConfigError(
            "controlled diagnostic actor profiles require the gated factorial runner"
        )


def transformers_policy_config_from_args(
    args: argparse.Namespace,
    *,
    base_checkpoint_sha256: str,
) -> TransformersPeftPolicyConfig:
    """Build the one policy config shared by natural and controlled runners."""

    return TransformersPeftPolicyConfig(
        model_path=str(Path(args.model_path).resolve()),
        base_checkpoint_sha256=base_checkpoint_sha256,
        model_revision=str(getattr(args, "configured_model_revision", "local")),
        device=args.device,
        dtype=args.dtype,
        learning_rate=args.learning_rate,
        clip_ratio=args.clip_ratio,
        clip_ratio_high=getattr(args, "clip_ratio_high", None),
        reference_kl_coef=args.reference_kl_coef,
        reference_kl_type=args.reference_kl_type,
        max_grad_norm=args.max_grad_norm,
        update_micro_batch_size=args.update_micro_batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_action_tokens,
        temperature=args.train_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        apply_chat_template=args.apply_chat_template,
        generation_micro_batch_size=int(getattr(args, "generation_micro_batch_size", 1)),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=tuple(args.lora_target_modules),
        lora_exclude_modules=tuple(args.lora_exclude_modules),
        model_loader=args.model_loader,
        quantization=args.quantization,
        quantization_double=args.quantization_double,
        allocator_warmup=bool(getattr(args, "allocator_warmup", True)),
        attn_implementation=str(getattr(args, "attn_implementation", "sdpa")),
        use_liger_kernel=bool(getattr(args, "use_liger_kernel", False)),
        liger_kernel_profile=str(getattr(args, "liger_kernel_profile", "full")),
        entropy_coef=float(getattr(args, "entropy_coef", 0.0)),
        initialization_seed=args.initialization_seed,
        adapter_path=getattr(args, "adapter_path", None),
        adapter_checkpoint_sha256=getattr(
            args,
            "adapter_checkpoint_sha256",
            None,
        ),
        adapter_artifact_sha256=getattr(
            args,
            "adapter_artifact_sha256",
            None,
        ),
        model_io_profile=(
            dict(args.model_io_profile)
            if not bool(getattr(args, "legacy_model_io_inferred", True))
            else None
        ),
        model_io_profile_sha256=(
            str(args.model_io_profile_sha256)
            if not bool(getattr(args, "legacy_model_io_inferred", True))
            else None
        ),
    )


def parse_compute_apps(output: str) -> tuple[dict[str, Any], ...]:
    apps: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        memory = parts[1].removesuffix("MiB").strip()
        if not memory.isdigit():
            continue
        apps.append(
            {
                "pid": int(parts[0]),
                "used_memory_mib": int(memory),
                "process_name": parts[2],
            }
        )
    return tuple(apps)


def query_compute_apps() -> tuple[dict[str, Any], ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot verify exclusive GPU availability: " + result.stderr.strip())
    return parse_compute_apps(result.stdout)


def assert_exclusive_gpu(
    *,
    device: str,
    allow_shared_gpu: bool,
    compute_apps: tuple[Mapping[str, Any], ...],
) -> None:
    if not device.startswith("cuda") or allow_shared_gpu or not compute_apps:
        return
    owners = ", ".join(
        f"pid={app['pid']} memory={app['used_memory_mib']}MiB process={app['process_name']}"
        for app in compute_apps
    )
    raise RuntimeError(
        "GPU is already in use; refusing to interfere with another experiment: " + owners
    )


__all__ = [
    "agentic_rl_rollout_config_from_args",
    "assert_exclusive_gpu",
    "parse_compute_apps",
    "query_compute_apps",
    "transformers_policy_config_from_args",
    "validate_agentic_rl_execution_admission",
]
