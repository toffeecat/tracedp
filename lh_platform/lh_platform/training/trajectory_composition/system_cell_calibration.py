"""Fail-closed selection of a support-scarce, nondegenerate N1 system cell.

This module consumes already-collected calibration artifacts.  It never imports
a model runtime, launches an environment, accesses target outcomes, or admits
training data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.alfworld_state import AlfWorldPhysicalCallCounters
from lh_platform.training.contracts import (
    TRAINING_DATASET_SCHEMA,
    TrainingTrajectoryRecord,
)
from lh_platform.training.subproblems.alfworld_semantic_verifier import (
    ALFWORLD_SEMANTIC_VERIFIER_ID,
)
from lh_platform.training.trajectory_composition.n1_plan import (
    ArtifactReferenceV1,
    ValidatedN1Plan,
)
from lh_platform.training.trajectory_composition.natural_support import (
    NATURAL_FRAGMENT_CANDIDATE_SCHEMA,
    NATURAL_SUPPORT_TASK_RECEIPT_SCHEMA,
    NaturalFragmentCandidateV1,
    NaturalSupportError,
)

SYSTEM_CELL_CALIBRATION_RESULT_SCHEMA = "lh_system_cell_calibration_result/v1"
SYSTEM_CELL_SELECTION_RECEIPT_SCHEMA = "lh_system_cell_selection_receipt/v1"
SYSTEM_CELL_SELECTION_PASS = (
    "SUPPORT_SCARCE_NONDEGENERATE_SYSTEM_CELL_SELECTED_NOT_N1_AUTHORIZATION"
)
SYSTEM_CELL_SELECTION_SATURATED = "DIRECT_SUPPORT_SATURATED_NO_FIRST_SUPPORT_RECOVERY_TEST"
SYSTEM_CELL_SELECTION_NO_ELIGIBLE = "NO_ELIGIBLE_SYSTEM_CELL_N1_BLOCKED"


class SystemCellCalibrationError(ValueError):
    """Raised when calibration evidence is incomplete, leaked, or inconsistent."""


@dataclass(frozen=True)
class _TrajectoryMetrics:
    task_id: str
    traj_uid: str
    sample_index: int
    success: bool
    logical_actions: int
    valid_actions: int
    admissible_actions: int
    verified_child_effect_ids: tuple[str, ...]
    physical_calls: AlfWorldPhysicalCallCounters


@dataclass(frozen=True)
class SystemCellSelection:
    receipt: Mapping[str, Any]
    selected_system_cell_id: str | None
    selected_target_samples_per_task: int | None


def select_system_cell_from_calibration(
    validated_plan: ValidatedN1Plan,
    calibration_result_path: str | Path,
    *,
    workspace_root: str | Path,
) -> SystemCellSelection:
    """Validate a complete calibration result and apply its preregistered rule."""

    workspace = Path(workspace_root).resolve()
    result_path = Path(calibration_result_path).resolve()
    if not result_path.is_file() or not result_path.is_relative_to(workspace):
        raise SystemCellCalibrationError("calibration result must be a file inside workspace_root")
    result = _read_mapping(result_path)
    _expect_exact_keys(
        result,
        {
            "schema",
            "plan_file_sha256",
            "plan_content_sha256",
            "calibration_task_manifest_sha256",
            "target_outcomes_accessed",
            "candidate_results",
            "execution_counts",
        },
        "calibration result",
    )
    if result["schema"] != SYSTEM_CELL_CALIBRATION_RESULT_SCHEMA:
        raise SystemCellCalibrationError("unsupported system-cell calibration schema")
    if (
        result["plan_file_sha256"] != validated_plan.plan_file_sha256
        or result["plan_content_sha256"] != validated_plan.plan_content_sha256
    ):
        raise SystemCellCalibrationError("calibration result is bound to another N1 plan")
    calibration_reference = _mapping(
        validated_plan.payload["calibration_task_manifest"],
        "calibration_task_manifest",
    )
    if result["calibration_task_manifest_sha256"] != calibration_reference["sha256"]:
        raise SystemCellCalibrationError("calibration task-manifest identity changed")
    if result["target_outcomes_accessed"] is not False:
        raise SystemCellCalibrationError("system-cell selection accessed target outcomes")

    candidate_rows = _sequence(result["candidate_results"], "candidate_results")
    cells = validated_plan.candidate_cells
    if len(candidate_rows) != len(cells):
        raise SystemCellCalibrationError("calibration candidate denominator is incomplete")
    expected_cell_ids = tuple(cell.system_cell_id for cell in cells)
    if (
        tuple(
            str(_mapping(row, "candidate result").get("system_cell_id")) for row in candidate_rows
        )
        != expected_cell_ids
    ):
        raise SystemCellCalibrationError(
            "calibration candidates must retain declared system-cell order"
        )

    calibration_tasks = validated_plan.calibration_manifest.tasks
    task_by_id = {task.task_id: task for task in calibration_tasks}
    task_ordinals = {task.task_id: index for index, task in enumerate(calibration_tasks)}
    samples_per_task = _positive_integer(
        validated_plan.payload["calibration_samples_per_task"],
        "calibration_samples_per_task",
    )
    seed_plan = _mapping(validated_plan.payload["seed_plan"], "seed_plan")
    environment_seeds = _mapping(
        seed_plan["calibration_environment_seeds"],
        "calibration_environment_seeds",
    )
    sampling_seed_base = _nonnegative_integer(
        seed_plan["calibration_sampling_seed_base"],
        "calibration_sampling_seed_base",
    )
    max_episode_steps = _positive_integer(
        _mapping(validated_plan.payload["rollout"], "rollout")["max_episode_steps"],
        "max_episode_steps",
    )
    if seed_plan["sampling_seed_schedule"] != (
        "base_plus_trajectory_ordinal_times_horizon_plus_step"
    ):
        raise SystemCellCalibrationError("calibration sampling-seed schedule changed")

    artifact_paths: set[Path] = set()
    trajectory_uids: set[str] = set()
    metrics_by_cell: dict[str, dict[str, tuple[_TrajectoryMetrics, ...]]] = {}
    combined_physical_calls = AlfWorldPhysicalCallCounters()
    model_samples = 0
    for cell, raw_candidate in zip(cells, candidate_rows, strict=True):
        candidate = _mapping(raw_candidate, f"candidate_results[{cell.system_cell_id}]")
        _expect_exact_keys(
            candidate,
            {"system_cell_id", "system_cell_content_sha256", "task_results"},
            f"candidate_results[{cell.system_cell_id}]",
        )
        if (
            candidate["system_cell_id"] != cell.system_cell_id
            or candidate["system_cell_content_sha256"] != cell.sha256
        ):
            raise SystemCellCalibrationError("calibration system-cell identity changed")
        task_rows = _sequence(candidate["task_results"], "task_results")
        if len(task_rows) != len(calibration_tasks):
            raise SystemCellCalibrationError(
                f"calibration task denominator incomplete for {cell.system_cell_id}"
            )
        if tuple(str(_mapping(row, "task result").get("task_id")) for row in task_rows) != tuple(
            task.task_id for task in calibration_tasks
        ):
            raise SystemCellCalibrationError("calibration task results must retain manifest order")
        cell_metrics: dict[str, tuple[_TrajectoryMetrics, ...]] = {}
        for raw_task_result in task_rows:
            task_result = _mapping(raw_task_result, "task result")
            _expect_exact_keys(
                task_result,
                {"task_id", "group_uid", "environment_seed", "rollouts"},
                "calibration task result",
            )
            task_id = str(task_result["task_id"])
            task = task_by_id.get(task_id)
            if task is None or task_result["group_uid"] != task.group_uid:
                raise SystemCellCalibrationError("calibration task/group identity changed")
            environment_seed = _nonnegative_integer(
                task_result["environment_seed"],
                "environment_seed",
            )
            if environment_seed != environment_seeds[task_id]:
                raise SystemCellCalibrationError("calibration environment seed changed")
            rollout_rows = _sequence(task_result["rollouts"], "rollouts")
            if len(rollout_rows) != samples_per_task:
                raise SystemCellCalibrationError("calibration fixed rollout group is incomplete")
            task_metrics: list[_TrajectoryMetrics] = []
            for expected_index, raw_rollout in enumerate(rollout_rows):
                rollout = _mapping(raw_rollout, "calibration rollout")
                _expect_exact_keys(
                    rollout,
                    {
                        "sample_index",
                        "sampling_seed_base",
                        "traj_uid",
                        "trajectory_artifact",
                        "natural_support_receipt",
                        "physical_call_totals",
                    },
                    "calibration rollout",
                )
                sample_index = _nonnegative_integer(
                    rollout["sample_index"],
                    "sample_index",
                )
                if sample_index != expected_index:
                    raise SystemCellCalibrationError(
                        "calibration sample indices must be contiguous and ordered"
                    )
                expected_sampling_seed_base = (
                    sampling_seed_base
                    + (task_ordinals[task_id] * samples_per_task + sample_index) * max_episode_steps
                )
                if rollout["sampling_seed_base"] != expected_sampling_seed_base:
                    raise SystemCellCalibrationError("calibration sampling seed changed")
                traj_uid = _identifier(rollout["traj_uid"], "traj_uid")
                if traj_uid in trajectory_uids:
                    raise SystemCellCalibrationError("calibration trajectory UID was reused")
                trajectory_uids.add(traj_uid)
                trajectory_reference = _artifact_reference(
                    rollout["trajectory_artifact"],
                    label="trajectory_artifact",
                )
                trajectory_path = _resolve_unique_reference(
                    trajectory_reference,
                    workspace=workspace,
                    label="trajectory_artifact",
                    seen=artifact_paths,
                )
                support_reference = _artifact_reference(
                    rollout["natural_support_receipt"],
                    label="natural_support_receipt",
                )
                support_path = _resolve_unique_reference(
                    support_reference,
                    workspace=workspace,
                    label="natural_support_receipt",
                    seen=artifact_paths,
                )
                physical = _physical_calls(rollout["physical_call_totals"])
                trajectory_metrics = _validate_trajectory_artifact(
                    trajectory_path,
                    task_id=task_id,
                    group_uid=task.group_uid,
                    traj_uid=traj_uid,
                    environment_seed=environment_seed,
                    sampling_seed_base=expected_sampling_seed_base,
                    system_cell_id=cell.system_cell_id,
                    system_cell_content_sha256=cell.sha256,
                    policy_id=cell.policy_identity.policy_id,
                    policy_checkpoint_sha256=(cell.policy_identity.policy_checkpoint_sha256),
                    model_io_profile_sha256=(cell.policy_identity.model_io_profile_sha256),
                    response_contract_sha256=str(
                        cell.actor_protocol.config["response_contract_sha256"]
                    ),
                    behavior_logprob_mode=cell.sampling_policy.behavior_logprob_mode,
                    sampling_temperature=cell.sampling_policy.temperature,
                    max_episode_steps=max_episode_steps,
                    physical_calls=physical,
                )
                child_effect_ids = _validate_natural_support_receipt(
                    support_path,
                    task_id=task_id,
                    group_uid=task.group_uid,
                    task_type=task.task_type,
                    horizon_bucket=task.horizon_bucket,
                    traj_uid=traj_uid,
                    trajectory_steps=trajectory_metrics["logical_actions"],
                    source_success=trajectory_metrics["success"],
                    policy_checkpoint_sha256=(cell.policy_identity.policy_checkpoint_sha256),
                )
                metrics = _TrajectoryMetrics(
                    task_id=task_id,
                    traj_uid=traj_uid,
                    sample_index=sample_index,
                    success=trajectory_metrics["success"],
                    logical_actions=trajectory_metrics["logical_actions"],
                    valid_actions=trajectory_metrics["valid_actions"],
                    admissible_actions=trajectory_metrics["admissible_actions"],
                    verified_child_effect_ids=child_effect_ids,
                    physical_calls=physical,
                )
                task_metrics.append(metrics)
                model_samples += metrics.logical_actions
                combined_physical_calls = combined_physical_calls.plus(physical)
            cell_metrics[task_id] = tuple(task_metrics)
        metrics_by_cell[cell.system_cell_id] = cell_metrics

    expected_execution_counts = {
        "model_samples": model_samples,
        "logical_semantic_actions": combined_physical_calls.logical_semantic_action_calls,
        "environment_resets": combined_physical_calls.environment_reset_calls,
        "environment_steps": combined_physical_calls.environment_step_calls,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_examples_admitted": 0,
    }
    if result["execution_counts"] != expected_execution_counts:
        raise SystemCellCalibrationError(
            "calibration execution counts do not reconcile with trajectories"
        )
    _validate_plan_budgets(validated_plan, expected_execution_counts)

    gate = _mapping(validated_plan.payload["selection_gate"], "selection_gate")
    group_sizes = tuple(
        _positive_integer(value, "candidate_target_samples_per_task")
        for value in _sequence(
            validated_plan.payload["candidate_target_samples_per_task"],
            "candidate_target_samples_per_task",
        )
    )
    pair_receipts: list[dict[str, Any]] = []
    by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    for group_size in group_sizes:
        for cell_id in gate["tie_break_cell_order"]:
            pair = _pair_metrics(
                metrics_by_cell[cell_id],
                system_cell_id=cell_id,
                group_size=group_size,
                gate=gate,
            )
            pair_receipts.append(pair)
            by_pair[(group_size, cell_id)] = pair

    selected_pair: tuple[int, str] | None = None
    for group_size in group_sizes:
        for cell_id in gate["tie_break_cell_order"]:
            if by_pair[(group_size, cell_id)]["eligible"]:
                selected_pair = (group_size, cell_id)
                break
        if selected_pair is not None:
            break
    if selected_pair is not None:
        selected_group_size, selected_cell_id = selected_pair
        selected_cell = next(cell for cell in cells if cell.system_cell_id == selected_cell_id)
        decision = SYSTEM_CELL_SELECTION_PASS
        selected = {
            "system_cell_id": selected_cell_id,
            "system_cell_content_sha256": selected_cell.sha256,
            "policy_checkpoint_sha256": (selected_cell.policy_identity.policy_checkpoint_sha256),
            "target_samples_per_task": selected_group_size,
            "selection_metrics": by_pair[selected_pair],
        }
    else:
        selected_group_size = None
        selected_cell_id = None
        all_saturated = bool(pair_receipts) and all(
            pair["direct_support_task_rate"] > gate["max_direct_support_task_rate"]
            for pair in pair_receipts
        )
        decision = (
            SYSTEM_CELL_SELECTION_SATURATED if all_saturated else SYSTEM_CELL_SELECTION_NO_ELIGIBLE
        )
        selected = None

    receipt: dict[str, Any] = {
        "schema": SYSTEM_CELL_SELECTION_RECEIPT_SCHEMA,
        "decision": decision,
        "scientific_result": False,
        "claim_scope": ("disjoint_calibration_system_cell_and_group_size_selection_only"),
        "calibration_result_path": str(result_path),
        "calibration_result_file_sha256": _sha256_file(result_path),
        "plan_file_sha256": validated_plan.plan_file_sha256,
        "plan_content_sha256": validated_plan.plan_content_sha256,
        "calibration_task_count": len(calibration_tasks),
        "candidate_system_cell_count": len(cells),
        "calibration_samples_per_task": samples_per_task,
        "candidate_target_samples_per_task": list(group_sizes),
        "selection_rule": gate["selection_rule"],
        "pair_metrics": pair_receipts,
        "selected": selected,
        "target_outcomes_accessed": False,
        "target_run_ready": False,
        "n1_execution_authorized": False,
        "training_authorized": False,
        "calibration_execution_counts": expected_execution_counts,
        "selection_execution_counts": {
            "model_samples": 0,
            "environment_resets": 0,
            "environment_steps": 0,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "training_examples_admitted": 0,
        },
        "combined_calibration_physical_call_totals": (combined_physical_calls.to_dict()),
        "combined_calibration_physical_call_totals_sha256": (combined_physical_calls.sha256),
        "raw_fragments_training_eligible": False,
        "blocked_by": [
            "selected_actor_availability_and_parity_receipt",
            "integrated_n1_runner_receipt",
            "n1_execution_authorization",
        ],
    }
    receipt["selection_receipt_sha256"] = _canonical_sha256(receipt)
    return SystemCellSelection(
        receipt=receipt,
        selected_system_cell_id=selected_cell_id,
        selected_target_samples_per_task=selected_group_size,
    )


def write_system_cell_selection_receipt(
    selection: SystemCellSelection,
    output_path: str | Path,
) -> Path:
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.tmp")
    staging.write_text(
        json.dumps(selection.receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(target)
    return target


def _validate_trajectory_artifact(
    path: Path,
    *,
    task_id: str,
    group_uid: str,
    traj_uid: str,
    environment_seed: int,
    sampling_seed_base: int,
    system_cell_id: str,
    system_cell_content_sha256: str,
    policy_id: str,
    policy_checkpoint_sha256: str,
    model_io_profile_sha256: str,
    response_contract_sha256: str,
    behavior_logprob_mode: str,
    sampling_temperature: float,
    max_episode_steps: int,
    physical_calls: AlfWorldPhysicalCallCounters,
) -> dict[str, Any]:
    records: list[TrainingTrajectoryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SystemCellCalibrationError(f"blank trajectory row at {path}:{line_number}")
            try:
                raw = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise SystemCellCalibrationError(
                    f"invalid trajectory JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(raw, dict):
                raise SystemCellCalibrationError("trajectory row must be a mapping")
            expected_fields = set(TrainingTrajectoryRecord.__dataclass_fields__)
            if set(raw) != expected_fields:
                raise SystemCellCalibrationError(
                    "trajectory record fields differ from the frozen v2 contract"
                )
            try:
                record = TrainingTrajectoryRecord.from_dict(raw)
            except (TypeError, ValueError) as error:
                raise SystemCellCalibrationError(
                    f"invalid training trajectory row: {error}"
                ) from error
            records.append(record)
    if not records or len(records) > max_episode_steps:
        raise SystemCellCalibrationError("trajectory action denominator is invalid")
    if [record.step_idx for record in records] != list(range(len(records))):
        raise SystemCellCalibrationError("trajectory steps must be contiguous from zero")
    if any(record.schema_version != TRAINING_DATASET_SCHEMA for record in records):
        raise SystemCellCalibrationError("trajectory schema changed")
    if any(
        record.task_id != task_id
        or record.group_uid != group_uid
        or record.traj_uid != traj_uid
        or record.seed != environment_seed
        or record.model_id != policy_id
        for record in records
    ):
        raise SystemCellCalibrationError("trajectory task/seed identity changed")
    if [record.terminal for record in records] != [False] * (len(records) - 1) + [True]:
        raise SystemCellCalibrationError("trajectory must have one final terminal row")
    success_values = {record.success for record in records}
    if len(success_values) != 1:
        raise SystemCellCalibrationError("trajectory success label changed across rows")
    if any(
        record.trajectory_length != len(records)
        or not record.prompt_ids
        or not record.response_ids
        or len(record.response_ids) != len(record.response_mask)
        or len(record.response_ids) != len(record.response_old_logprobs)
        or record.action_token_count != len(record.response_ids)
        or any(value != 1 for value in record.response_mask)
        or any(not math.isfinite(value) for value in record.response_old_logprobs)
        or not math.isfinite(record.reward)
        or not math.isfinite(record.timestamp)
        or not _is_sha256(record.prompt_hash)
        or not _is_sha256(record.rollout_config_hash)
        or not _is_sha256(record.goal_hash)
        or not _is_sha256(record.action_affordance_hash)
        for record in records
    ):
        raise SystemCellCalibrationError(
            "trajectory generation-time token/action denominator is incomplete"
        )
    for left, right in zip(records, records[1:], strict=False):
        if left.extra.get("post_state_snapshot_sha256") != right.state_snapshot_sha256:
            raise SystemCellCalibrationError("trajectory pre/post snapshot adjacency changed")
    valid_actions = 0
    admissible_actions = 0
    for record in records:
        extra = record.extra
        required = {
            "system_cell_id": system_cell_id,
            "system_cell_content_sha256": system_cell_content_sha256,
            "policy_checkpoint_sha256": policy_checkpoint_sha256,
            "model_io_profile_sha256": model_io_profile_sha256,
            "response_contract_sha256": response_contract_sha256,
            "behavior_logprob_mode": behavior_logprob_mode,
            "sampling_temperature": sampling_temperature,
            "transition_evidence_mode": "snapshot_only",
            "restore_verified": False,
            "training_eligible": False,
        }
        if any(extra.get(key) != value for key, value in required.items()):
            raise SystemCellCalibrationError(
                "trajectory system-cell/provenance/training guard changed"
            )
        if (
            extra.get("sampling_seed_base") != sampling_seed_base
            or extra.get("sampling_seed") != sampling_seed_base + record.step_idx
            or extra.get("sampling_seed_schedule")
            != "base_plus_trajectory_ordinal_times_horizon_plus_step"
        ):
            raise SystemCellCalibrationError("trajectory per-step sampling-seed schedule changed")
        if not _is_sha256(record.state_snapshot_sha256) or not _is_sha256(
            str(extra.get("post_state_snapshot_sha256", ""))
        ):
            raise SystemCellCalibrationError("trajectory pre/post snapshot identity is missing")
        if not isinstance(extra.get("is_action_valid"), bool):
            raise SystemCellCalibrationError("trajectory valid-action evidence is missing")
        valid_actions += int(extra["is_action_valid"])
        action = str(extra.get("environment_action", ""))
        admissible = extra.get("alfworld_admissible_actions")
        if not action or not isinstance(admissible, list):
            raise SystemCellCalibrationError("trajectory admissible-action evidence is missing")
        admissible_actions += int(action in admissible)
        row_calls = _physical_calls(extra.get("alfworld_trajectory_physical_call_totals"))
        if row_calls != physical_calls:
            raise SystemCellCalibrationError(
                "trajectory physical-call total differs across records"
            )
    if (
        physical_calls.source_worker_reset_calls != 1
        or physical_calls.state_restore_calls != 0
        or physical_calls.batch_environment_close_calls != 0
        or physical_calls.batch_environment_load_calls != 0
        or physical_calls.batch_environment_reset_calls != 0
        or physical_calls.snapshot_only_worker_step_calls != len(records)
        or physical_calls.unverified_worker_step_calls != 0
        or physical_calls.candidate_probe_first_worker_step_calls != 0
        or physical_calls.candidate_probe_replay_worker_step_calls != 0
        or physical_calls.rollout_first_worker_step_calls != 0
        or physical_calls.rollout_verification_worker_step_calls != 0
        or physical_calls.logical_semantic_action_calls != len(records)
        or physical_calls.environment_reset_calls != 1
        or physical_calls.environment_step_calls != len(records)
    ):
        raise SystemCellCalibrationError(
            "calibration source must use snapshot-only calls without replay probes"
        )
    return {
        "success": next(iter(success_values)),
        "logical_actions": len(records),
        "valid_actions": valid_actions,
        "admissible_actions": admissible_actions,
    }


def _validate_natural_support_receipt(
    path: Path,
    *,
    task_id: str,
    group_uid: str,
    task_type: str,
    horizon_bucket: str,
    traj_uid: str,
    trajectory_steps: int,
    source_success: bool,
    policy_checkpoint_sha256: str,
) -> tuple[str, ...]:
    receipt = _read_mapping(path)
    _expect_exact_keys(
        receipt,
        {
            "schema",
            "task_id",
            "group_uid",
            "task_type",
            "horizon_bucket",
            "traj_uid",
            "trajectory_steps",
            "source_parent_success",
            "parent_verifier_agrees",
            "verified_fragment_candidates",
            "final_verified_goal_ids",
            "final_parent_verifier_receipt_sha256",
        },
        "natural-support receipt",
    )
    required = {
        "schema": NATURAL_SUPPORT_TASK_RECEIPT_SCHEMA,
        "task_id": task_id,
        "group_uid": group_uid,
        "task_type": task_type,
        "horizon_bucket": horizon_bucket,
        "traj_uid": traj_uid,
        "trajectory_steps": trajectory_steps,
        "source_parent_success": source_success,
        "parent_verifier_agrees": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise SystemCellCalibrationError("natural-support receipt identity changed")
    if not _is_sha256(str(receipt["final_parent_verifier_receipt_sha256"])):
        raise SystemCellCalibrationError("natural-support parent verifier hash is invalid")
    final_goal_ids = tuple(
        _identifier(value, "final_verified_goal_id")
        for value in _sequence(
            receipt["final_verified_goal_ids"],
            "final_verified_goal_ids",
        )
    )
    if len(final_goal_ids) != len(set(final_goal_ids)):
        raise SystemCellCalibrationError("natural-support final goals are duplicated")
    candidates = _sequence(
        receipt["verified_fragment_candidates"],
        "verified_fragment_candidates",
    )
    effect_ids: list[str] = []
    candidate_fields = {
        "schema",
        "task_id",
        "traj_uid",
        "source_policy_checkpoint_sha256",
        "boundary_step_idx",
        "child_goal_id",
        "child_node_ids",
        "prefix_action_count",
        "prefix_action_trace_sha256",
        "boundary_action",
        "persistent_at_trajectory_end",
        "source_parent_success",
        "independent_verifier_id",
        "independent_verifier_receipt_sha256",
        "replay_ready",
        "training_eligible",
    }
    for candidate_raw in candidates:
        candidate = _mapping(candidate_raw, "verified fragment candidate")
        _expect_exact_keys(candidate, candidate_fields, "verified fragment candidate")
        try:
            parsed = NaturalFragmentCandidateV1(
                schema=str(candidate["schema"]),
                task_id=str(candidate["task_id"]),
                traj_uid=str(candidate["traj_uid"]),
                source_policy_checkpoint_sha256=str(candidate["source_policy_checkpoint_sha256"]),
                boundary_step_idx=_nonnegative_integer(
                    candidate["boundary_step_idx"], "boundary_step_idx"
                ),
                child_goal_id=str(candidate["child_goal_id"]),
                child_node_ids=tuple(
                    _identifier(value, "child_node_id")
                    for value in _sequence(candidate["child_node_ids"], "child_node_ids")
                ),
                prefix_action_count=_positive_integer(
                    candidate["prefix_action_count"], "prefix_action_count"
                ),
                prefix_action_trace_sha256=str(candidate["prefix_action_trace_sha256"]),
                boundary_action=str(candidate["boundary_action"]),
                persistent_at_trajectory_end=_boolean(
                    candidate["persistent_at_trajectory_end"],
                    "persistent_at_trajectory_end",
                ),
                source_parent_success=_boolean(
                    candidate["source_parent_success"],
                    "source_parent_success",
                ),
                independent_verifier_id=str(candidate["independent_verifier_id"]),
                independent_verifier_receipt_sha256=str(
                    candidate["independent_verifier_receipt_sha256"]
                ),
                replay_ready=_boolean(candidate["replay_ready"], "replay_ready"),
                training_eligible=_boolean(candidate["training_eligible"], "training_eligible"),
            )
        except NaturalSupportError as error:
            raise SystemCellCalibrationError(
                f"invalid natural child-effect candidate: {error}"
            ) from error
        if (
            parsed.schema != NATURAL_FRAGMENT_CANDIDATE_SCHEMA
            or parsed.task_id != task_id
            or parsed.traj_uid != traj_uid
            or parsed.source_policy_checkpoint_sha256 != policy_checkpoint_sha256
            or parsed.source_parent_success is not source_success
            or parsed.independent_verifier_id != ALFWORLD_SEMANTIC_VERIFIER_ID
            or parsed.boundary_step_idx >= trajectory_steps
        ):
            raise SystemCellCalibrationError("natural child-effect candidate provenance changed")
        effect_ids.append(parsed.child_goal_id)
    if len(effect_ids) != len(set(effect_ids)):
        raise SystemCellCalibrationError("natural-support receipt repeats a child effect")
    return tuple(sorted(effect_ids)) if not source_success else ()


def _pair_metrics(
    by_task: Mapping[str, tuple[_TrajectoryMetrics, ...]],
    *,
    system_cell_id: str,
    group_size: int,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    task_count = len(by_task)
    direct_support_tasks = 0
    logical_actions = 0
    valid_actions = 0
    admissible_actions = 0
    failed_trajectories = 0
    failed_with_child_effect = 0
    failed_tasks_with_child_effect = 0
    for trajectories in by_task.values():
        retained = trajectories[:group_size]
        direct_support_tasks += int(any(item.success for item in retained))
        task_has_failed_effect = False
        for item in retained:
            logical_actions += item.logical_actions
            valid_actions += item.valid_actions
            admissible_actions += item.admissible_actions
            if not item.success:
                failed_trajectories += 1
                if item.verified_child_effect_ids:
                    failed_with_child_effect += 1
                    task_has_failed_effect = True
        failed_tasks_with_child_effect += int(task_has_failed_effect)
    direct_rate = direct_support_tasks / task_count
    valid_rate = valid_actions / logical_actions
    admissible_rate = admissible_actions / logical_actions
    eligibility = {
        "direct_support_above_minimum": (direct_rate >= gate["min_direct_support_task_rate"]),
        "direct_support_not_saturated": (direct_rate <= gate["max_direct_support_task_rate"]),
        "valid_action_competence": (valid_rate >= gate["min_valid_semantic_action_rate"]),
        "admissible_action_competence": (
            admissible_rate >= gate["min_admissible_semantic_action_rate"]
        ),
        "failed_trajectory_child_effect_support": (
            failed_with_child_effect >= gate["min_failed_trajectories_with_child_effect"]
        ),
        "failed_task_child_effect_support": (
            failed_tasks_with_child_effect >= gate["min_failed_tasks_with_child_effect"]
        ),
    }
    return {
        "system_cell_id": system_cell_id,
        "target_samples_per_task": group_size,
        "task_count": task_count,
        "direct_support_task_count": direct_support_tasks,
        "direct_support_task_rate": direct_rate,
        "logical_semantic_action_count": logical_actions,
        "valid_semantic_action_count": valid_actions,
        "valid_semantic_action_rate": valid_rate,
        "admissible_semantic_action_count": admissible_actions,
        "admissible_semantic_action_rate": admissible_rate,
        "failed_trajectory_count": failed_trajectories,
        "failed_trajectories_with_child_effect": failed_with_child_effect,
        "failed_tasks_with_child_effect": failed_tasks_with_child_effect,
        "eligibility_checks": eligibility,
        "eligible": all(eligibility.values()),
    }


def _validate_plan_budgets(
    validated_plan: ValidatedN1Plan,
    counts: Mapping[str, int],
) -> None:
    budgets = _mapping(validated_plan.payload["budgets"], "budgets")
    mapping = {
        "model_samples": "max_model_samples",
        "logical_semantic_actions": "max_logical_semantic_actions",
        "environment_resets": "max_environment_reset_calls",
        "environment_steps": "max_environment_step_calls",
        "optimizer_steps": "max_optimizer_steps",
        "parameter_updates": "max_parameter_updates",
        "training_examples_admitted": "max_training_examples_admitted",
    }
    for count_key, budget_key in mapping.items():
        if counts[count_key] > budgets[budget_key]:
            raise SystemCellCalibrationError(f"calibration {count_key} exceeds prospective budget")


def _artifact_reference(value: Any, *, label: str) -> ArtifactReferenceV1:
    try:
        return ArtifactReferenceV1.from_dict(
            _mapping(value, label),
            label=label,
        )
    except ValueError as error:
        raise SystemCellCalibrationError(str(error)) from error


def _resolve_unique_reference(
    reference: ArtifactReferenceV1,
    *,
    workspace: Path,
    label: str,
    seen: set[Path],
) -> Path:
    try:
        path = reference.resolve(workspace, label=label)
    except Exception as error:
        raise SystemCellCalibrationError(str(error)) from error
    if path in seen:
        raise SystemCellCalibrationError(f"calibration artifact reused: {path}")
    seen.add(path)
    return path


def _physical_calls(value: Any) -> AlfWorldPhysicalCallCounters:
    if not isinstance(value, Mapping):
        raise SystemCellCalibrationError("physical-call receipt must be a mapping")
    try:
        return AlfWorldPhysicalCallCounters.from_dict(value)
    except (TypeError, ValueError) as error:
        raise SystemCellCalibrationError(f"invalid physical-call receipt: {error}") from error


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemCellCalibrationError(
            f"cannot read calibration artifact {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise SystemCellCalibrationError(f"calibration artifact must be a mapping: {path}")
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _expect_exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        raise SystemCellCalibrationError(
            f"{label} fields differ: missing={missing}, unknown={unknown}"
        )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemCellCalibrationError(f"{label} must be a mapping")
    return dict(value)


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SystemCellCalibrationError(f"{label} must be a sequence")
    return tuple(value)


def _identifier(value: Any, label: str) -> str:
    text = str(value)
    if not text or text != text.strip() or any(character in text for character in "\r\n"):
        raise SystemCellCalibrationError(f"{label} must be canonical")
    return text


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SystemCellCalibrationError(f"{label} must be boolean")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemCellCalibrationError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    result = _nonnegative_integer(value, label)
    if result < 1:
        raise SystemCellCalibrationError(f"{label} must be positive")
    return result


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SystemCellCalibrationError(
            f"selection receipt is not canonical JSON: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SYSTEM_CELL_CALIBRATION_RESULT_SCHEMA",
    "SYSTEM_CELL_SELECTION_NO_ELIGIBLE",
    "SYSTEM_CELL_SELECTION_PASS",
    "SYSTEM_CELL_SELECTION_RECEIPT_SCHEMA",
    "SYSTEM_CELL_SELECTION_SATURATED",
    "SystemCellCalibrationError",
    "SystemCellSelection",
    "select_system_cell_from_calibration",
    "write_system_cell_selection_receipt",
]
