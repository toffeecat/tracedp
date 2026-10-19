"""Pure analysis for frozen residual-state exposure rollouts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.parent_child_state_use.exposure import (
    ActorStateExposureMode,
)

STATE_EXPOSURE_ROLLOUT_CLAIM_SCOPE = (
    "frozen_full_rollout_state_exposure_diagnostic_not_policy_improvement_or_endogenous_awareness"
)
STATE_EXPOSURE_SNAPSHOT_ONLY_CLAIM_SCOPE = (
    "frozen_full_rollout_state_exposure_snapshot_only_behavior_screen_"
    "not_restore_replay_credit_policy_improvement_or_endogenous_awareness"
)
PREREGISTERED_MODES = (
    ActorStateExposureMode.NORMAL.value,
    ActorStateExposureMode.FLAT_ORACLE.value,
    ActorStateExposureMode.GRAPH_ORACLE.value,
    ActorStateExposureMode.CORRUPTED_GRAPH.value,
)


class StateExposureRolloutError(RuntimeError):
    """Raised when the frozen four-condition comparison is not identifiable."""


def summarize_state_exposure_rollouts(
    records_by_mode: Mapping[str, Sequence[TrainingTrajectoryRecord]],
) -> dict[str, Any]:
    """Compare full trajectories while preserving each exposure as one arm."""

    if tuple(records_by_mode) != PREREGISTERED_MODES:
        raise StateExposureRolloutError("rollout summary requires the preregistered mode order")
    transition_evidence_modes = {
        str(record.extra.get("transition_evidence_mode", "restore_replay"))
        for records in records_by_mode.values()
        for record in records
    }
    if len(transition_evidence_modes) != 1:
        raise StateExposureRolloutError("state-exposure arms mix transition evidence modes")
    transition_evidence_mode = next(iter(transition_evidence_modes))
    if transition_evidence_mode not in {"restore_replay", "snapshot_only"}:
        raise StateExposureRolloutError("state-exposure transition evidence mode is unsupported")
    if transition_evidence_mode == "restore_replay" and any(
        record.extra.get("restore_verified") is False
        for records in records_by_mode.values()
        for record in records
    ):
        raise StateExposureRolloutError(
            "restore/replay state-exposure record is not restore verified"
        )
    claim_scope = (
        STATE_EXPOSURE_ROLLOUT_CLAIM_SCOPE
        if transition_evidence_mode == "restore_replay"
        else STATE_EXPOSURE_SNAPSHOT_ONLY_CLAIM_SCOPE
    )
    arms: dict[str, dict[str, Any]] = {}
    task_realizations: list[tuple[str, ...]] = []
    for mode in PREREGISTERED_MODES:
        records = tuple(records_by_mode[mode])
        if not records:
            raise StateExposureRolloutError(f"state-exposure arm is empty: {mode}")
        arms[mode] = _summarize_arm(records, mode=mode)
        task_realizations.append(tuple(arms[mode]["trajectory_task_ids"]))
    if any(tasks != task_realizations[0] for tasks in task_realizations[1:]):
        raise StateExposureRolloutError("state-exposure arms must realize the same task set")
    contrasts = {
        "flat_minus_normal": _metric_delta(
            arms[ActorStateExposureMode.FLAT_ORACLE.value],
            arms[ActorStateExposureMode.NORMAL.value],
        ),
        "graph_minus_flat": _metric_delta(
            arms[ActorStateExposureMode.GRAPH_ORACLE.value],
            arms[ActorStateExposureMode.FLAT_ORACLE.value],
        ),
        "correct_graph_minus_corrupted_graph": _metric_delta(
            arms[ActorStateExposureMode.GRAPH_ORACLE.value],
            arms[ActorStateExposureMode.CORRUPTED_GRAPH.value],
        ),
    }
    return {
        "schema": "lh_alfworld_state_exposure_rollout_summary/v1",
        "state_exposure_modes": list(PREREGISTERED_MODES),
        "shared_task_set": True,
        "shared_task_ids": sorted(set(task_realizations[0])),
        "shared_task_occurrences": dict(Counter(task_realizations[0])),
        "arms": arms,
        "contrasts": contrasts,
        "gates": {
            "correct_vs_corrupted_graph_initial_prompt_token_match": (
                arms[ActorStateExposureMode.GRAPH_ORACLE.value]["initial_prompt_token_counts"]
                == arms[ActorStateExposureMode.CORRUPTED_GRAPH.value]["initial_prompt_token_counts"]
            ),
            "all_arms_have_executable_actions": all(
                arm["executable_transition_count"] > 0 for arm in arms.values()
            ),
            "normal_has_outcome_variation": _has_outcome_variation(
                arms[ActorStateExposureMode.NORMAL.value]
            ),
            "flat_has_outcome_variation": _has_outcome_variation(
                arms[ActorStateExposureMode.FLAT_ORACLE.value]
            ),
            "flat_progress_exceeds_normal": (
                contrasts["flat_minus_normal"]["unique_goal_achievement_count"] > 0
                and contrasts["flat_minus_normal"]["net_progress_transition_count"] > 0
            ),
            "correct_graph_progress_exceeds_corrupted": (
                contrasts["correct_graph_minus_corrupted_graph"]["unique_goal_achievement_count"]
                > 0
                and contrasts["correct_graph_minus_corrupted_graph"][
                    "net_progress_transition_count"
                ]
                > 0
            ),
        },
        "policy_update_executed": False,
        "transition_evidence_mode": transition_evidence_mode,
        "restore_replay_evidence": transition_evidence_mode == "restore_replay",
        "claim_scope": claim_scope,
    }


def compare_state_exposure_evidence_mode_records(
    restore_records_by_mode: Mapping[str, Sequence[TrainingTrajectoryRecord]],
    snapshot_records_by_mode: Mapping[str, Sequence[TrainingTrajectoryRecord]],
) -> dict[str, Any]:
    """Prove that the faster evidence mode did not change actor behavior.

    Only receipt/path/timestamp fields that necessarily differ between runs are
    removed. Exact prompt/response IDs, old logprobs, actions, observations,
    outcomes, snapshots, budgets, and goal annotations remain in the digest.
    """

    if (
        tuple(restore_records_by_mode) != PREREGISTERED_MODES
        or tuple(snapshot_records_by_mode) != PREREGISTERED_MODES
    ):
        raise StateExposureRolloutError(
            "evidence-mode parity requires the preregistered mode order"
        )
    per_mode: dict[str, dict[str, Any]] = {}
    total = 0
    different = 0
    for mode in PREREGISTERED_MODES:
        restore_records = tuple(restore_records_by_mode[mode])
        snapshot_records = tuple(snapshot_records_by_mode[mode])
        if len(restore_records) != len(snapshot_records):
            raise StateExposureRolloutError(f"evidence-mode parity row count differs for {mode}")
        restore_projection = tuple(
            _behavior_record_projection(record) for record in restore_records
        )
        snapshot_projection = tuple(
            _behavior_record_projection(record) for record in snapshot_records
        )
        difference_indices = [
            index
            for index, (left, right) in enumerate(
                zip(restore_projection, snapshot_projection, strict=True)
            )
            if left != right
        ]
        total += len(restore_projection)
        different += len(difference_indices)
        per_mode[mode] = {
            "transition_count": len(restore_projection),
            "behavior_difference_count": len(difference_indices),
            "first_behavior_difference_index": (
                difference_indices[0] if difference_indices else None
            ),
            "restore_projection_sha256": _canonical_sha256(restore_projection),
            "snapshot_projection_sha256": _canonical_sha256(snapshot_projection),
        }
    return {
        "schema": "lh_state_exposure_evidence_mode_parity/v1",
        "transition_count": total,
        "behavior_difference_count": different,
        "behavior_parity": different == 0,
        "compared_fields": (
            "all_behavior_and_environment_semantics_with_facts_as_an_unordered_set_"
            "excluding_timestamp_rollout_config_hash_and_evidence_mode_receipts"
        ),
        "per_mode": per_mode,
        "claim_scope": (
            "snapshot_only_behavior_parity_with_restore_replay_"
            "not_restore_replay_credit_or_policy_improvement"
        ),
    }


def _behavior_record_projection(record: TrainingTrajectoryRecord) -> dict[str, Any]:
    payload = record.to_dict()
    payload.pop("timestamp", None)
    # The compared runs intentionally use different evidence-mode configs.
    # Their config hashes are provenance, not actor/environment behavior.
    payload.pop("rollout_config_hash", None)
    extra = dict(payload.get("extra") or {})
    for field in (
        "transition_evidence_mode",
        "restore_verified",
        "restore_probe_receipt_sha256",
        "restore_probe_receipt_path",
        "state_snapshot_artifact_path",
    ):
        extra.pop(field, None)
    payload["extra"] = _canonicalize_fact_collections(extra)
    return payload


def _canonicalize_fact_collections(value: object) -> object:
    """Canonicalize only fields whose schema defines an unordered fact set."""

    if isinstance(value, dict):
        normalized = {key: _canonicalize_fact_collections(item) for key, item in value.items()}
        for key, item in tuple(normalized.items()):
            if (key == "facts" or key.endswith("_facts")) and isinstance(item, list):
                normalized[key] = sorted(item, key=_canonical_json)
        return normalized
    if isinstance(value, list):
        return [_canonicalize_fact_collections(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    import json

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _canonical_sha256(value: object) -> str:
    import hashlib

    encoded = _canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summarize_arm(
    records: Sequence[TrainingTrajectoryRecord],
    *,
    mode: str,
) -> dict[str, Any]:
    by_trajectory: dict[str, list[TrainingTrajectoryRecord]] = defaultdict(list)
    for record in records:
        if record.benchmark != "alfworld_native":
            raise StateExposureRolloutError("state-exposure arm is not ALFWorld")
        if record.extra.get("actor_state_exposure_mode") != mode:
            raise StateExposureRolloutError(f"state-exposure receipt differs from arm: {mode}")
        if record.extra.get("goal_annotation_verified") is not True:
            raise StateExposureRolloutError("state-exposure rollout lacks verified goal annotation")
        by_trajectory[record.traj_uid].append(record)
    successful = 0
    repeat_count = 0
    revisit_count = 0
    progress_trajectories = 0
    unique_goal_achievement_count = 0
    final_residual_counts = []
    for trajectory in by_trajectory.values():
        ordered = sorted(trajectory, key=lambda item: item.step_idx)
        successful += int(any(item.success for item in ordered))
        previous_action = None
        seen_observations: set[str] = set()
        trajectory_progress = False
        trajectory_unique_goals: set[str] = set()
        for record in ordered:
            action = str(
                record.extra.get("environment_action") or record.parsed_action.get("command", "")
            )
            observation = str(record.extra.get("alfworld_raw_observation", record.observation))
            if previous_action is not None and action == previous_action:
                repeat_count += 1
            if observation in seen_observations:
                revisit_count += 1
            previous_action = action
            seen_observations.add(observation)
            trajectory_progress = trajectory_progress or bool(
                record.extra.get("newly_satisfied_goal_ids")
            )
            trajectory_unique_goals.update(
                str(goal_id) for goal_id in (record.extra.get("newly_satisfied_goal_ids") or ())
            )
        progress_trajectories += int(trajectory_progress)
        unique_goal_achievement_count += len(trajectory_unique_goals)
        final_residual_counts.append(len(ordered[-1].extra.get("residual_goal_ids_after") or ()))
    trajectory_task_ids = sorted(
        ordered[0].task_id
        for trajectory in by_trajectory.values()
        for ordered in (sorted(trajectory, key=lambda item: item.step_idx),)
    )
    initial_prompt_token_counts = sorted(
        (ordered[0].task_id, len(ordered[0].prompt_ids))
        for trajectory in by_trajectory.values()
        for ordered in (sorted(trajectory, key=lambda item: item.step_idx),)
    )
    progress_transition_count = sum(
        int(bool(record.extra.get("newly_satisfied_goal_ids"))) for record in records
    )
    regression_transition_count = sum(
        int(bool(record.extra.get("regressed_goal_ids"))) for record in records
    )
    return {
        "task_ids": sorted({record.task_id for record in records}),
        "trajectory_task_ids": trajectory_task_ids,
        "initial_prompt_token_counts": [
            {"task_id": task_id, "token_count": token_count}
            for task_id, token_count in initial_prompt_token_counts
        ],
        "trajectory_count": len(by_trajectory),
        "transition_count": len(records),
        "success_trajectory_count": successful,
        "failed_trajectory_count": len(by_trajectory) - successful,
        "executable_transition_count": sum(
            int(record.extra.get("is_action_valid") is True) for record in records
        ),
        # Gross boundary crossings are retained for backward-compatible
        # diagnostics, but must not be interpreted as durable progress.
        "progress_transition_count": progress_transition_count,
        "unique_goal_achievement_count": unique_goal_achievement_count,
        "net_progress_transition_count": (progress_transition_count - regression_transition_count),
        "progress_trajectory_count": progress_trajectories,
        "regression_transition_count": regression_transition_count,
        "phase_switch_transition_count": sum(
            int(
                tuple(record.extra.get("active_frontier_ids") or ())
                != tuple(record.extra.get("active_frontier_ids_after") or ())
            )
            for record in records
        ),
        "consecutive_action_repeat_count": repeat_count,
        "observation_revisit_count": revisit_count,
        "mean_final_residual_goal_count": (sum(final_residual_counts) / len(final_residual_counts)),
        "mean_prompt_token_count": (
            sum(len(record.prompt_ids) for record in records) / len(records)
        ),
    }


def _metric_delta(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, float | int]:
    fields = (
        "success_trajectory_count",
        "executable_transition_count",
        "progress_transition_count",
        "unique_goal_achievement_count",
        "net_progress_transition_count",
        "progress_trajectory_count",
        "regression_transition_count",
        "phase_switch_transition_count",
        "consecutive_action_repeat_count",
        "observation_revisit_count",
        "mean_final_residual_goal_count",
        "mean_prompt_token_count",
    )
    return {field: treatment[field] - control[field] for field in fields}


def _has_outcome_variation(arm: Mapping[str, Any]) -> bool:
    return int(arm["success_trajectory_count"]) > 0 and int(arm["failed_trajectory_count"]) > 0


__all__ = [
    "PREREGISTERED_MODES",
    "STATE_EXPOSURE_ROLLOUT_CLAIM_SCOPE",
    "STATE_EXPOSURE_SNAPSHOT_ONLY_CLAIM_SCOPE",
    "StateExposureRolloutError",
    "compare_state_exposure_evidence_mode_records",
    "summarize_state_exposure_rollouts",
]
