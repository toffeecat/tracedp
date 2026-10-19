"""Normal-actor collection runner for residual-state credit calibration Pilot A."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.training.agentic_rl_components import AlfWorldFactDecomposition
from lh_platform.training.agentic_rl_config import (
    AgenticRLRuntimeBindings,
    bind_agentic_rl_runtime,
    load_agentic_rl_experiment,
)
from lh_platform.training.agentic_rl_harness import (
    assert_exclusive_gpu,
    query_compute_apps,
    transformers_policy_config_from_args,
)
from lh_platform.training.agentic_rl_p0 import run_p0
from lh_platform.training.agentic_rl_runtime import (
    AgenticRolloutBatch,
    RunEventLogger,
    canonical_sha256,
    write_agentic_rollout_artifact,
)
from lh_platform.training.alfworld_agentic_suite import OfficialAlfWorldBranchFactory
from lh_platform.training.alfworld_controlled_actor import (
    collect_controlled_actor_rollout,
)
from lh_platform.training.alfworld_controlled_aliasing import (
    load_alfworld_controlled_aliasing_recipe,
)
from lh_platform.training.config_provenance import repository_config_tree_bindings
from lh_platform.training.controlled_actor_factorial_run import (
    resolve_controlled_task_binding,
)
from lh_platform.training.dataset import sha256_file
from lh_platform.training.residual_credit.calibration import (
    analyze_residual_credit_calibration_artifacts,
    load_residual_credit_calibration_config,
    residual_calibration_recipe_contracts,
)
from lh_platform.training.transformers_peft_policy import (
    TransformersPeftPolicy,
    checkpoint_tree_sha256,
)

RESIDUAL_CREDIT_CALIBRATION_RUN_SCHEMA = "lh_residual_credit_calibration_run/v1"


class ResidualCreditCalibrationRunError(RuntimeError):
    """Raised when Pilot A cannot preserve its preregistered collection design."""


def run_residual_credit_calibration_pilot(
    config_path: str | Path,
    *,
    model_path: str | Path,
    source_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str,
    preflight_only: bool,
    allow_shared_gpu: bool,
) -> Path:
    """Collect three development pairs with one frozen normal actor and audit LOTO."""

    resolved = load_residual_credit_calibration_config(config_path)
    root = resolved.workspace_root
    payload = resolved.payload
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    event_logger = RunEventLogger(target / "run_events.jsonl", run_id=uuid.uuid4().hex)
    event_logger.emit(
        "residual_credit_calibration_started",
        config_path=str(resolved.config_path),
        config_sha256=resolved.sha256,
        requested_device=device,
        preflight_only=preflight_only,
        policy_update_allowed=False,
    )

    bindings = repository_config_tree_bindings(resolved)
    experiment = load_agentic_rl_experiment(root / payload["experiment_ref"])
    base_args = bind_agentic_rl_runtime(
        experiment,
        AgenticRLRuntimeBindings(
            model_path=Path(model_path),
            source_root=Path(source_root),
            data_root=Path(data_root),
            output_dir=target / "base_preflight",
            device=device,
            preflight_only=True,
            allow_shared_gpu=allow_shared_gpu,
        ),
    )
    runtime = payload["runtime"]
    base_args.generation_micro_batch_size = runtime["generation_micro_batch_size"]
    base_preflight_path = run_p0(base_args)
    base_preflight = json.loads(base_preflight_path.read_text(encoding="utf-8"))
    preflight = {
        "schema": "lh_residual_credit_calibration_preflight/v1",
        "config_path": str(resolved.config_path),
        "config_sha256": resolved.sha256,
        "config_file_sha256": sha256_file(resolved.config_path),
        "config_bindings": [dict(binding) for binding in bindings],
        "base_preflight_path": str(base_preflight_path),
        "base_preflight_sha256": sha256_file(base_preflight_path),
        "base_checkpoint_sha256": base_preflight["base_checkpoint_sha256"],
        "source_revision": base_preflight["source_revision"],
        "configured_pair_count": len(payload["controlled_recipe_refs"]),
        "formal_minimum_pair_count": payload["admission"]["formal_minimum_pair_count"],
        "formal_launch_eligible": all(bool(binding["matches_head"]) for binding in bindings),
        "gpu_loaded": False,
        "policy_update_allowed": False,
        "claim_scope": "development_heldout_calibration_preflight_only",
    }
    preflight_path = target / "residual_credit_calibration_preflight.json"
    atomic_write_json(preflight_path, preflight)
    event_logger.emit(
        "residual_credit_calibration_preflight_written",
        path=str(preflight_path),
        sha256=sha256_file(preflight_path),
        formal_launch_eligible=preflight["formal_launch_eligible"],
    )
    if preflight_only:
        return preflight_path
    if not preflight["formal_launch_eligible"]:
        raise ResidualCreditCalibrationRunError(
            "Pilot A config tree must be committed unchanged before GPU collection"
        )

    compute_apps = query_compute_apps() if device.startswith("cuda") else ()
    assert_exclusive_gpu(
        device=device,
        allow_shared_gpu=allow_shared_gpu,
        compute_apps=compute_apps,
    )
    base_checkpoint_sha256 = checkpoint_tree_sha256(model_path)
    if base_checkpoint_sha256 != preflight["base_checkpoint_sha256"]:
        raise ResidualCreditCalibrationRunError("model checkpoint changed after Pilot A preflight")
    policy_args = argparse.Namespace(**vars(base_args))
    policy_args.preflight_only = False
    policy_config = transformers_policy_config_from_args(
        policy_args,
        base_checkpoint_sha256=base_checkpoint_sha256,
    )
    if float(base_args.train_temperature) != float(runtime["sampling_temperature"]):
        raise ResidualCreditCalibrationRunError(
            "Pilot A sampling temperature differs from the bound actor config"
        )
    event_logger.emit(
        "residual_credit_calibration_policy_load_started",
        base_checkpoint_sha256=base_checkpoint_sha256,
    )
    policy: TransformersPeftPolicy | None = None
    initial_checkpoint_sha256 = ""
    run_failed = False
    try:
        policy = TransformersPeftPolicy(policy_config)
        initial_checkpoint_sha256 = policy.checkpoint_sha256
        event_logger.emit(
            "residual_credit_calibration_policy_loaded",
            policy_checkpoint_sha256=initial_checkpoint_sha256,
            adapter_initial_delta_zero=policy.adapter_initial_delta_zero,
        )
        return _collect_and_analyze_loaded_policy(
            resolved=resolved,
            policy=policy,
            initial_checkpoint_sha256=initial_checkpoint_sha256,
            source_root=Path(source_root),
            data_root=Path(data_root),
            target=target,
            bindings=bindings,
            sampling_temperature=float(runtime["sampling_temperature"]),
            preflight_path=preflight_path,
            event_logger=event_logger,
        )
    except Exception as error:
        run_failed = True
        event_logger.emit(
            "residual_credit_calibration_failed",
            error_type=type(error).__name__,
            error_message=str(error),
            policy_update_executed=False,
        )
        raise
    finally:
        if policy is not None:
            try:
                policy.close()
            except Exception as cleanup_error:
                event_logger.emit(
                    "residual_credit_calibration_policy_release_failed",
                    error_type=type(cleanup_error).__name__,
                    error_message=str(cleanup_error),
                )
                if not run_failed:
                    raise
            else:
                event_logger.emit(
                    "residual_credit_calibration_policy_released",
                    policy_checkpoint_sha256=initial_checkpoint_sha256,
                )


def _collect_and_analyze_loaded_policy(
    *,
    resolved: Any,
    policy: TransformersPeftPolicy,
    initial_checkpoint_sha256: str,
    source_root: Path,
    data_root: Path,
    target: Path,
    bindings: tuple[Any, ...],
    sampling_temperature: float,
    preflight_path: Path,
    event_logger: RunEventLogger,
) -> Path:
    payload = resolved.payload
    runtime = payload["runtime"]
    data_root_path = data_root.resolve()
    branch_factory = OfficialAlfWorldBranchFactory(
        source_root=source_root.resolve(),
        data_root=data_root_path,
    )
    annotated_paths: list[Path] = []
    recipe_receipts: list[dict[str, Any]] = []
    contracts_by_recipe = {
        contract["recipe_sha256"]: contract
        for contract in residual_calibration_recipe_contracts(resolved)
    }
    for recipe_index, recipe_ref in enumerate(payload["controlled_recipe_refs"]):
        recipe_path = resolved.workspace_root / recipe_ref
        recipe = load_alfworld_controlled_aliasing_recipe(recipe_path)
        task, metadata, instruction, task_manifest_sha256 = resolve_controlled_task_binding(
            recipe,
            data_root_path,
        )
        seed = runtime["sampling_seed"] + recipe_index * runtime["sampling_seed_stride"]
        rollout_config_sha256 = canonical_sha256(
            {
                "schema": "lh_residual_credit_calibration_rollout_config/v1",
                "pilot_config_sha256": resolved.sha256,
                "recipe_sha256": recipe.sha256,
                "actor_state_exposure": "normal",
                "samples_per_state": runtime["samples_per_state"],
                "max_episode_steps": runtime["max_episode_steps"],
                "sampling_seed": seed,
                "sampling_temperature": sampling_temperature,
                "action_space": "free_text_then_source_alfworld_projection",
            }
        )
        event_logger.emit(
            "residual_credit_pair_collection_started",
            recipe_id=recipe.recipe_id,
            recipe_sha256=recipe.sha256,
            sampling_seed=seed,
            samples_per_state=runtime["samples_per_state"],
        )
        batch = collect_controlled_actor_rollout(
            recipe,
            task=task,
            task_metadata=metadata,
            instruction=instruction,
            policy=policy,
            branch_factory=branch_factory,
            artifacts_dir=target / "environment" / recipe.recipe_id,
            actor_awareness_mode="none",
            samples_per_state=runtime["samples_per_state"],
            max_episode_steps=runtime["max_episode_steps"],
            sampling_seed=seed,
            task_manifest_sha256=task_manifest_sha256,
            rollout_config_sha256=rollout_config_sha256,
            sampling_temperature=sampling_temperature,
        )
        contract = contracts_by_recipe[recipe.sha256]
        batch = AgenticRolloutBatch(
            records=tuple(
                replace(
                    row,
                    extra={
                        **row.extra,
                        "pilot_base_checkpoint_sha256": contract["expected_base_checkpoint_sha256"],
                        "pilot_model_revision": contract["expected_model_revision"],
                        "pilot_model_reference": contract["expected_model_reference"],
                    },
                )
                for row in batch.records
            ),
            policy_samples=batch.policy_samples,
            task_manifest_sha256=batch.task_manifest_sha256,
            policy_checkpoint_sha256=batch.policy_checkpoint_sha256,
        ).validate()
        if batch.policy_checkpoint_sha256 != initial_checkpoint_sha256:
            raise ResidualCreditCalibrationRunError(
                "Pilot A recipe collections used different policy checkpoints"
            )
        recipe_dir = target / "pairs" / recipe.recipe_id
        raw_path = recipe_dir / "frozen_model_policy_rollout.jsonl"
        raw_sha256 = write_agentic_rollout_artifact(raw_path, batch)
        annotated_records = AlfWorldFactDecomposition().annotate(batch.records)
        annotated = AgenticRolloutBatch(
            records=annotated_records,
            policy_samples=batch.policy_samples,
            task_manifest_sha256=batch.task_manifest_sha256,
            policy_checkpoint_sha256=batch.policy_checkpoint_sha256,
        ).validate()
        annotated_path = recipe_dir / "frozen_annotated_rollout.jsonl"
        annotated_sha256 = write_agentic_rollout_artifact(annotated_path, annotated)
        annotated_paths.append(annotated_path)
        terminal = [row for row in annotated.records if row.terminal]
        receipt = {
            "recipe_id": recipe.recipe_id,
            "recipe_path": str(recipe_path),
            "recipe_file_sha256": sha256_file(recipe_path),
            "recipe_semantic_sha256": recipe.sha256,
            "raw_rollout_path": str(raw_path),
            "raw_rollout_sha256": raw_sha256,
            "annotated_rollout_path": str(annotated_path),
            "annotated_rollout_sha256": annotated_sha256,
            "trajectory_count": len(terminal),
            "transition_count": len(annotated.records),
            "success_count": sum(bool(row.success) for row in terminal),
            "sampling_seed": seed,
        }
        recipe_receipts.append(receipt)
        event_logger.emit("residual_credit_pair_collection_completed", **receipt)

    estimator = payload["estimator"]
    admission = payload["admission"]
    calibration_path = target / "residual_credit_calibration.json"
    calibration = analyze_residual_credit_calibration_artifacts(
        annotated_paths,
        gamma=float(estimator["gamma"]),
        success_reward=float(estimator["success_reward"]),
        random_split_seeds=tuple(estimator["matched_random_split_seeds"]),
        minimum_pair_count=int(admission["formal_minimum_pair_count"]),
        minimum_loto_coverage=float(admission["minimum_loto_coverage"]),
        recipe_contracts=residual_calibration_recipe_contracts(resolved),
        preregistration={
            "config_path": str(resolved.config_path),
            "config_file_sha256": sha256_file(resolved.config_path),
            "config_sha256": resolved.sha256,
            "bindings": [dict(binding) for binding in bindings],
        },
        output_path=calibration_path,
    )
    result = {
        "schema": RESIDUAL_CREDIT_CALIBRATION_RUN_SCHEMA,
        "config_path": str(resolved.config_path),
        "config_sha256": resolved.sha256,
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "initial_policy_checkpoint_sha256": initial_checkpoint_sha256,
        "pair_receipts": recipe_receipts,
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "formal_calibration_gate_passed": calibration["calibration_gate_passed"],
        "formal_calibration_blocked_by": calibration["blocked_by"],
        "policy_update_executed": False,
        "claim_scope": (
            "development_residual_credit_calibration_not_policy_improvement_"
            "recursive_dp_or_parent_child_credit_decoupling"
        ),
    }
    result_path = target / "residual_credit_calibration_run.json"
    atomic_write_json(result_path, result)
    event_logger.emit(
        "residual_credit_calibration_completed",
        result_path=str(result_path),
        result_sha256=sha256_file(result_path),
        formal_calibration_gate_passed=result["formal_calibration_gate_passed"],
        formal_calibration_blocked_by=result["formal_calibration_blocked_by"],
        policy_update_executed=False,
    )
    return result_path


__all__ = [
    "ResidualCreditCalibrationRunError",
    "run_residual_credit_calibration_pilot",
]
