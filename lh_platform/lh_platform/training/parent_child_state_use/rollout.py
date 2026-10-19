"""Frozen full-rollout factorial for residual-state actor exposure.

This module measures behavior before credit estimation.  One frozen policy is
loaded once, then evaluated on the same ALFWorld tasks under four preregistered
prompt conditions.  It never computes advantages or updates model weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.training.adapter_identity import (
    AdapterIdentityError,
    load_adapter_identity,
)
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
    RolloutRequest,
    canonical_sha256,
    write_agentic_rollout_artifact,
)
from lh_platform.training.alfworld_agentic_suite import (
    AlfWorldNativeTaskSuite,
    AlfWorldSuiteConfig,
    OfficialAlfWorldBranchFactory,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.dataset import sha256_file
from lh_platform.training.evidence_artifacts import (
    reserve_fresh_evidence_directory,
    write_json_exclusive,
)
from lh_platform.training.parent_child_state_use.rollout_analysis import (
    PREREGISTERED_MODES,
    STATE_EXPOSURE_ROLLOUT_CLAIM_SCOPE,
    STATE_EXPOSURE_SNAPSHOT_ONLY_CLAIM_SCOPE,
    StateExposureRolloutError,
    compare_state_exposure_evidence_mode_records,
    summarize_state_exposure_rollouts,
)
from lh_platform.training.transformers_peft_policy import (
    TransformersPeftPolicy,
    checkpoint_tree_sha256,
)

STATE_EXPOSURE_ROLLOUT_SCHEMA = "lh_alfworld_state_exposure_rollout/v1"
STATE_EXPOSURE_ROLLOUT_RESULT_SCHEMA = "lh_alfworld_state_exposure_rollout_result/v1"


def load_state_exposure_rollout_config(
    path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise StateExposureRolloutError(
            f"cannot load state-exposure rollout config: {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema") != (STATE_EXPOSURE_ROLLOUT_SCHEMA):
        raise StateExposureRolloutError("state-exposure rollout schema differs")
    required = {
        "id",
        "experiment_ref",
        "state_exposure_modes",
        "seed",
        "task_count",
        "max_episode_steps",
        "generation_micro_batch_size",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise StateExposureRolloutError(
            "state-exposure rollout config is missing: " + ", ".join(missing)
        )
    modes = payload["state_exposure_modes"]
    if not isinstance(modes, list) or tuple(str(item) for item in modes) != (PREREGISTERED_MODES):
        raise StateExposureRolloutError(
            "state-exposure rollout requires exactly the preregistered four modes"
        )
    for name in (
        "seed",
        "task_count",
        "max_episode_steps",
        "generation_micro_batch_size",
    ):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise StateExposureRolloutError(f"{name} must be a positive integer")
    transition_evidence_mode = str(payload.get("transition_evidence_mode", "restore_replay"))
    if transition_evidence_mode not in {"restore_replay", "snapshot_only"}:
        raise StateExposureRolloutError(
            "transition_evidence_mode must be restore_replay or snapshot_only"
        )
    normalized = dict(payload)
    normalized["transition_evidence_mode"] = transition_evidence_mode
    return config_path, normalized


def run_alfworld_state_exposure_rollout(
    config_path: str | Path,
    *,
    model_path: str | Path,
    source_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str,
    preflight_only: bool = False,
    allow_shared_gpu: bool = False,
    adapter_path: str | Path | None = None,
) -> Path:
    """Run the four prompt conditions with one frozen policy and shared tasks."""

    config_file, payload = load_state_exposure_rollout_config(config_path)
    workspace = _workspace_root(config_file)
    target = reserve_fresh_evidence_directory(
        output_dir,
        protected_paths=(
            Path(model_path),
            Path(source_root),
            Path(data_root),
            config_file,
            workspace / "configs",
            *((Path(adapter_path),) if adapter_path is not None else ()),
        ),
    )
    experiment = load_agentic_rl_experiment(workspace / str(payload["experiment_ref"]))
    if experiment.payload["runtime"]["actor_awareness_mode"] != "none":
        raise StateExposureRolloutError(
            "state-exposure rollout requires the normal legacy actor setting"
        )
    args = bind_agentic_rl_runtime(
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
    args.generation_micro_batch_size = int(payload["generation_micro_batch_size"])
    base_preflight_path = run_p0(args)
    try:
        adapter_binding = load_adapter_identity(adapter_path) if adapter_path is not None else None
    except AdapterIdentityError as error:
        raise StateExposureRolloutError(str(error)) from error
    expected_base = str(args.configured_model_checkpoint_tree_sha256)
    if adapter_binding is not None and (adapter_binding.base_checkpoint_sha256 != expected_base):
        raise StateExposureRolloutError("adapter binds a different configured base model")
    config_sha256 = sha256_file(config_file)
    preflight = {
        "schema": "lh_alfworld_state_exposure_rollout_preflight/v1",
        "config_path": str(config_file),
        "config_sha256": config_sha256,
        "experiment_sha256": experiment.sha256,
        "base_preflight_path": str(base_preflight_path),
        "base_preflight_sha256": sha256_file(base_preflight_path),
        "base_checkpoint_sha256": expected_base,
        "adapter_binding": (adapter_binding.to_dict() if adapter_binding is not None else None),
        "state_exposure_modes": list(PREREGISTERED_MODES),
        "transition_evidence_mode": payload["transition_evidence_mode"],
        "shared_seed": int(payload["seed"]),
        "gpu_loaded": False,
        "policy_update_executed": False,
        "claim_scope": "state_exposure_rollout_preflight_only",
    }
    preflight_path = target / "state_exposure_rollout_preflight.json"
    write_json_exclusive(preflight_path, preflight)
    if preflight_only:
        return preflight_path

    actual_base = checkpoint_tree_sha256(model_path)
    if sha256_file(config_file) != config_sha256:
        raise StateExposureRolloutError("state-exposure config changed after structured preflight")
    if actual_base != expected_base:
        raise StateExposureRolloutError("base checkpoint changed after state-exposure preflight")
    if adapter_binding is not None:
        try:
            current_adapter = load_adapter_identity(adapter_binding.path)
        except AdapterIdentityError as error:
            raise StateExposureRolloutError(str(error)) from error
        if current_adapter != adapter_binding:
            raise StateExposureRolloutError("adapter changed after state-exposure preflight")
    assert_exclusive_gpu(
        device=device,
        allow_shared_gpu=allow_shared_gpu,
        compute_apps=(query_compute_apps() if device.startswith("cuda") else ()),
    )
    policy_args = argparse.Namespace(**vars(args))
    policy_args.preflight_only = False
    if adapter_binding is not None:
        policy_args.adapter_path = adapter_binding.path
        policy_args.adapter_checkpoint_sha256 = adapter_binding.checkpoint_sha256
        policy_args.adapter_artifact_sha256 = adapter_binding.adapter_artifact_sha256
    policy = TransformersPeftPolicy(
        transformers_policy_config_from_args(
            policy_args,
            base_checkpoint_sha256=actual_base,
        )
    )
    policy_checkpoint_before = policy.checkpoint_sha256
    seed = int(payload["seed"])
    task_count = int(payload["task_count"])
    max_steps = int(payload["max_episode_steps"])
    decomposition = AlfWorldFactDecomposition()
    records_by_mode: dict[str, tuple[TrainingTrajectoryRecord, ...]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for mode in PREREGISTERED_MODES:
        suite = AlfWorldNativeTaskSuite(
            AlfWorldSuiteConfig(
                role="eval_task_suite",
                manifest_path=str(args.eval_manifest),
                data_root=str(Path(data_root).resolve()),
                artifacts_dir=str(target / mode / "environment"),
                source_revision=str(args.source_revision),
                train_task_groups_per_update=task_count,
                train_task_selection=str(args.train_task_selection),
                eval_tasks_per_horizon=int(args.eval_tasks_per_horizon),
                history_length=int(args.history_length),
                train_temperature=float(args.train_temperature),
                eval_temperature=float(args.eval_temperature),
                actor_protocol_id=str(args.actor_protocol_id),
                model_io_profile_id=getattr(args, "model_io_profile_id", None),
                actor_awareness_mode="none",
                actor_state_exposure_mode=mode,
                evaluation_selection="all_manifest_tasks_exactly_once",
                transition_evidence_mode=str(payload["transition_evidence_mode"]),
            ),
            branch_factory=OfficialAlfWorldBranchFactory(
                source_root=Path(source_root).resolve(),
                data_root=Path(data_root).resolve(),
            ),
        )
        rollout = suite.collect_frozen(
            policy,
            RolloutRequest(
                update_index=0,
                seed=seed,
                group_size=1,
                max_episode_steps=max_steps,
                rollout_config_sha256=canonical_sha256(
                    {
                        "config": payload,
                        "mode": mode,
                        "shared_seed": seed,
                    }
                ),
            ),
        ).validate()
        annotated_records = tuple(decomposition.annotate(rollout.records))
        annotated = AgenticRolloutBatch(
            records=annotated_records,
            policy_samples=rollout.policy_samples,
            task_manifest_sha256=rollout.task_manifest_sha256,
            policy_checkpoint_sha256=rollout.policy_checkpoint_sha256,
        ).validate()
        if annotated.policy_checkpoint_sha256 != policy_checkpoint_before:
            raise StateExposureRolloutError("frozen policy changed during state-exposure rollout")
        artifact_path = target / mode / "annotated_rollout.jsonl"
        artifact_sha256 = write_agentic_rollout_artifact(
            artifact_path,
            annotated,
        )
        records_by_mode[mode] = annotated.records
        artifacts[mode] = {
            "path": str(artifact_path),
            "sha256": artifact_sha256,
        }
    policy_checkpoint_after = policy.checkpoint_sha256
    if policy_checkpoint_after != policy_checkpoint_before:
        raise StateExposureRolloutError("state-exposure factorial mutated the frozen policy")
    summary = summarize_state_exposure_rollouts(records_by_mode)
    if summary["transition_evidence_mode"] != payload["transition_evidence_mode"]:
        raise StateExposureRolloutError("collected transition evidence mode differs from config")
    result = {
        "schema": STATE_EXPOSURE_ROLLOUT_RESULT_SCHEMA,
        "config": payload,
        "config_path": str(config_file),
        "config_sha256": config_sha256,
        "base_checkpoint_sha256": actual_base,
        "adapter_binding": (adapter_binding.to_dict() if adapter_binding is not None else None),
        "policy_checkpoint_before_sha256": policy_checkpoint_before,
        "policy_checkpoint_after_sha256": policy_checkpoint_after,
        "policy_update_executed": False,
        "artifacts": artifacts,
        "summary": summary,
        "transition_evidence_mode": payload["transition_evidence_mode"],
        "claim_scope": summary["claim_scope"],
    }
    result_path = target / "state_exposure_rollout_result.json"
    atomic_write_json(result_path, result, overwrite=False)
    return result_path


def _workspace_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "lh_platform").is_dir() and (parent / "configs").is_dir():
            return parent
    raise StateExposureRolloutError(f"cannot locate LongHorizon workspace from {path}")


__all__ = [
    "PREREGISTERED_MODES",
    "STATE_EXPOSURE_ROLLOUT_CLAIM_SCOPE",
    "STATE_EXPOSURE_SNAPSHOT_ONLY_CLAIM_SCOPE",
    "STATE_EXPOSURE_ROLLOUT_RESULT_SCHEMA",
    "STATE_EXPOSURE_ROLLOUT_SCHEMA",
    "StateExposureRolloutError",
    "compare_state_exposure_evidence_mode_records",
    "load_state_exposure_rollout_config",
    "run_alfworld_state_exposure_rollout",
    "summarize_state_exposure_rollouts",
]
