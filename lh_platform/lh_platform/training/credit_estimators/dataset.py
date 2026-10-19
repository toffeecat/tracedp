"""Fail-closed projection from canonical environment records to credit visits."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from lh_platform.training.contracts import TrainingTrajectoryRecord

from .contracts import CreditEvidenceError, CreditVisit, FrozenBehaviorEvidence
from .credit_keys import verifier_refined_key


def credit_visits_from_trajectory_records(
    records: Iterable[TrainingTrajectoryRecord],
    *,
    gamma: float,
    success_reward: float = 1.0,
    evaluation: str = "all",
    transition_evidence_requirement: Literal[
        "restore_replay", "snapshot_or_restore"
    ] = "restore_replay",
) -> tuple[CreditVisit, ...]:
    """Create estimator inputs without re-deriving benchmark-owned semantics.

    The adapter consumes only verified residual/frontier labels already stored
    by ALFWorld or AppWorld collectors.  A3 feature evidence is copied if
    present and is validated again by ``A3Estimator`` for the current
    within-observation diagnostic.  Full A3 remains separately gated.
    Snapshot-only evidence is opt-in for frozen calibration; online callers
    retain the restore/replay default.
    """

    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise CreditEvidenceError("gamma must be between zero and one")
    if not math.isfinite(success_reward) or success_reward <= 0:
        raise CreditEvidenceError("success_reward must be positive")
    if evaluation not in {"all", "trajectory_start"}:
        raise CreditEvidenceError("evaluation must be all or trajectory_start")
    if transition_evidence_requirement not in {
        "restore_replay",
        "snapshot_or_restore",
    }:
        raise CreditEvidenceError("transition evidence requirement is unsupported")
    rows = tuple(records)
    if not rows:
        raise CreditEvidenceError("trajectory records are empty")
    by_trajectory: dict[str, list[TrainingTrajectoryRecord]] = defaultdict(list)
    for row in rows:
        if not row.active:
            raise CreditEvidenceError("inactive/PAD records cannot enter credit estimation")
        by_trajectory[row.traj_uid].append(row)

    visits = []
    for trajectory_uid, trajectory in sorted(by_trajectory.items()):
        ordered = sorted(trajectory, key=lambda row: row.step_idx)
        if [row.step_idx for row in ordered] != list(range(len(ordered))):
            raise CreditEvidenceError(f"trajectory {trajectory_uid!r} has non-contiguous steps")
        if not ordered[-1].terminal or sum(row.terminal for row in ordered) != 1:
            raise CreditEvidenceError(
                f"trajectory {trajectory_uid!r} must end in one terminal transition"
            )
        if len({row.success for row in ordered}) != 1:
            raise CreditEvidenceError(f"trajectory {trajectory_uid!r} success label drifted")
        terminal_success = bool(ordered[-1].success)
        for row in ordered:
            _validate_outcome_evidence(row, terminal_success=terminal_success)
        terminal_return = success_reward if terminal_success else 0.0
        for position, row in enumerate(ordered):
            extra = row.extra
            if extra.get("goal_annotation_verified") is not True:
                raise CreditEvidenceError(
                    "credit record lacks verified residual/frontier annotation"
                )
            if extra.get("actor_awareness_mode") != "none":
                raise CreditEvidenceError(
                    "credit ablation requires the normal actor without oracle exposure"
                )
            if (
                extra.get("actor_state_exposure_mode", "normal") != "normal"
                or extra.get("actor_state_exposure") is not None
            ):
                raise CreditEvidenceError(
                    "credit ablation requires normal actor input without state exposure"
                )
            _validate_transition_evidence(
                row,
                requirement=transition_evidence_requirement,
            )
            residual = _required_text(extra, "residual_goal_signature")
            frontier = _required_text(extra, "frontier_signature")
            if (
                row.remaining_budget is None
                or row.remaining_budget < 0
                or extra.get("remaining_budget_verified") is not True
            ):
                raise CreditEvidenceError("credit record lacks remaining horizon")
            anchor = str(extra.get("anchor_observation") or row.observation)
            coarse_key = (row.benchmark, row.group_uid, anchor)
            phase_key = verifier_refined_key(
                coarse_key=coarse_key,
                residual_signature=residual,
                frontier_signature=frontier,
                remaining_budget=int(row.remaining_budget),
            )
            visits.append(
                CreditVisit(
                    visit_uid=f"{trajectory_uid}:{row.step_idx}",
                    trajectory_uid=trajectory_uid,
                    dataset_id=row.dataset_id,
                    benchmark=row.benchmark,
                    task_id=row.task_id,
                    step_idx=row.step_idx,
                    coarse_key=coarse_key,
                    phase_key=phase_key,
                    history_key=(
                        row.benchmark,
                        row.group_uid,
                        row.prompt_hash,
                    )
                    if row.prompt_hash
                    else None,
                    exact_key=(
                        row.benchmark,
                        row.group_uid,
                        row.state_snapshot_sha256,
                    )
                    if row.state_snapshot_sha256
                    else None,
                    return_value=(gamma ** (len(ordered) - position - 1)) * terminal_return,
                    success=terminal_success,
                    remaining_horizon=int(row.remaining_budget),
                    evaluation_visit=evaluation == "all" or position == 0,
                    behavior=_behavior_evidence(extra),
                )
            )
    return tuple(visit.validate() for visit in visits)


def _required_text(extra: Mapping[str, Any], key: str) -> str:
    value = extra.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CreditEvidenceError(f"credit record lacks {key}")
    return value.strip()


def _validate_transition_evidence(
    row: TrainingTrajectoryRecord,
    *,
    requirement: Literal["restore_replay", "snapshot_or_restore"],
) -> None:
    extra = row.extra
    mode = extra.get("transition_evidence_mode", "restore_replay")
    if mode == "restore_replay":
        if extra.get("restore_verified") is not True:
            raise CreditEvidenceError(
                "credit ablation requires restore/replay-verified transitions"
            )
        return
    if requirement == "restore_replay" or mode != "snapshot_only":
        raise CreditEvidenceError("credit ablation requires restore/replay-verified transitions")
    if extra.get("restore_verified") is not False:
        raise CreditEvidenceError(
            "snapshot-only calibration must explicitly deny restore verification"
        )
    artifact_path = extra.get("state_snapshot_artifact_path")
    if not row.state_snapshot_sha256 or not isinstance(artifact_path, str) or not artifact_path:
        raise CreditEvidenceError(
            "snapshot-only calibration requires content-addressed snapshot evidence"
        )


def _validate_outcome_evidence(row: TrainingTrajectoryRecord, *, terminal_success: bool) -> None:
    evidence = row.outcome_evidence
    benchmark = row.benchmark.lower()
    if benchmark.startswith("alfworld"):
        won = evidence.get("won")
        if not isinstance(won, bool):
            raise CreditEvidenceError("ALFWorld outcome evidence must retain official won")
        if row.terminal and won is not terminal_success:
            raise CreditEvidenceError("ALFWorld terminal won differs from trajectory success")
        return
    if benchmark.startswith("appworld"):
        evaluator = evidence.get("official_evaluator")
        if not isinstance(evaluator, Mapping):
            raise CreditEvidenceError(
                "AppWorld outcome evidence must retain the official evaluator"
            )
        success = evaluator.get("success")
        if not isinstance(success, bool):
            raise CreditEvidenceError("AppWorld official evaluator must retain boolean success")
        if row.terminal and success is not terminal_success:
            raise CreditEvidenceError(
                "AppWorld terminal evaluator success differs from trajectory success"
            )
        return
    raise CreditEvidenceError(
        f"credit estimator has no benchmark-owned outcome validator for {row.benchmark!r}"
    )


def _behavior_evidence(
    extra: Mapping[str, Any],
) -> FrozenBehaviorEvidence | None:
    raw = extra.get("a3_behavior_features")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise CreditEvidenceError("a3_behavior_features must be a mapping")
    output = []
    for name, vector in sorted(raw.items()):
        if not isinstance(name, str) or not isinstance(vector, (list, tuple)):
            raise CreditEvidenceError("A3 behavior feature payload is malformed")
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as error:
            raise CreditEvidenceError("A3 behavior feature vector is not numeric") from error
        output.append((name, values))
    raw_action_ids = extra.get("a3_behavior_action_ids")
    if not isinstance(raw_action_ids, (list, tuple)):
        raise CreditEvidenceError("A3 behavior sidecar lacks canonical action IDs")
    raw_action_weights = extra.get("a3_behavior_action_weights")
    if not isinstance(raw_action_weights, (list, tuple)):
        raise CreditEvidenceError("A3 behavior sidecar lacks action weights")
    return FrozenBehaviorEvidence(
        features=tuple(output),
        action_ids=tuple(str(value) for value in raw_action_ids),
        action_weights=tuple(float(value) for value in raw_action_weights),
        embedding_source=str(extra.get("a3_behavior_embedding_source", "")),
        actor_visible=extra.get("a3_behavior_actor_visible") is True,
        frozen=extra.get("a3_behavior_frozen") is True,
        source_batch_id=str(extra.get("a3_behavior_source_batch_id", "")),
        target_batch_id=str(extra.get("a3_behavior_target_batch_id", "")),
        source_artifact_sha256=str(extra.get("a3_behavior_source_artifact_sha256", "")),
        target_artifact_sha256=str(extra.get("a3_behavior_target_artifact_sha256", "")),
        policy_checkpoint_sha256=str(extra.get("a3_behavior_policy_checkpoint_sha256", "")),
        materialization_sha256=str(extra.get("a3_behavior_materialization_sha256", "")),
    ).validate()


__all__ = ["credit_visits_from_trajectory_records"]
