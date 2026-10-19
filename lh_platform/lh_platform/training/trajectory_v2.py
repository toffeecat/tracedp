"""Fail-closed accounting for active, observable environment transitions.

The v2 contract is intentionally independent from a trainer implementation.
It establishes which rows may enter a graph, receive credit, or contribute to
the actor loss.  In particular, a padded row or a row following ``terminal``
is not a transition and is rejected before any aggregation happens.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from lh_platform.training.contracts import TrainingTrajectoryRecord


class TrajectoryContractError(ValueError):
    """Raised when a purported environment trajectory cannot be trusted."""


# A rollback is a planner *decision kind*, not a third source of transitions.
# This keeps every non-executor decision at the same observable boundary.
_DECISION_ROLES = frozenset({"executor", "planner"})
_PLANNER_DECISIONS = frozenset({"subgoal", "route", "rollback"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TrajectoryWeightSummary:
    trajectories: int
    transitions: int
    token_count: int
    total_gradient_mass_by_trajectory: dict[str, float]
    rollout_length_gradient_mass_correlation: float | None


def validate_active_trajectory_records(
    records: Iterable[TrainingTrajectoryRecord], *, require_v2: bool = True
) -> tuple[TrainingTrajectoryRecord, ...]:
    """Validate only rows that are legal training transitions.

    ``require_v2`` is deliberately true for all new credit modes.  Legacy
    records remain readable by the dataset layer but cannot silently become a
    G2PO/verified-state training batch.
    """

    rows = tuple(records)
    if not rows:
        raise TrajectoryContractError("at least one active environment transition is required")
    by_trajectory: dict[str, list[TrainingTrajectoryRecord]] = defaultdict(list)
    for row in rows:
        if row.extra.get("training_eligible") is False:
            raise TrajectoryContractError(
                "analysis-only records cannot enter graph, credit, or loss"
            )
        if not row.active:
            raise TrajectoryContractError("inactive/PAD rows cannot enter graph, credit, or loss")
        if row.extra.get("padding") or row.extra.get("post_terminal"):
            raise TrajectoryContractError(
                "padding or post-terminal rows cannot enter graph, credit, or loss"
            )
        if not row.action_raw.strip():
            raise TrajectoryContractError(
                "an active transition requires an observable semantic action"
            )
        if row.action_token_count <= 0:
            raise TrajectoryContractError(
                "action_token_count must be positive for active transitions"
            )
        if row.decision_role not in _DECISION_ROLES:
            raise TrajectoryContractError(f"unsupported decision_role: {row.decision_role!r}")
        if row.decision_role == "planner":
            if row.extra.get("planner_decision_kind") not in _PLANNER_DECISIONS:
                raise TrajectoryContractError(
                    "planner rows require an observable subgoal, route, or rollback decision"
                )
            if any(
                key in row.extra
                for key in ("internal_thought", "chain_of_thought", "planner_thought")
            ):
                raise TrajectoryContractError("planner rows must not store internal thought text")
        if require_v2:
            _require_v2_metadata(row)
        by_trajectory[row.traj_uid].append(row)

    for traj_uid, trajectory in by_trajectory.items():
        ordered = sorted(trajectory, key=lambda row: row.step_idx)
        actual_steps = len(ordered)
        if [row.step_idx for row in ordered] != list(range(actual_steps)):
            raise TrajectoryContractError(
                f"trajectory {traj_uid!r} must have unique contiguous step_idx values from zero"
            )
        terminal_indices = [index for index, row in enumerate(ordered) if row.terminal]
        if terminal_indices != [actual_steps - 1]:
            raise TrajectoryContractError(
                f"trajectory {traj_uid!r} must have exactly one terminal final transition"
            )
        if require_v2 and any(row.trajectory_length != actual_steps for row in ordered):
            raise TrajectoryContractError(
                f"trajectory {traj_uid!r} trajectory_length must equal its active transition count"
            )
    return rows


def trajectory_step_token_weights(
    records: Iterable[TrainingTrajectoryRecord], *, require_v2: bool = True
) -> dict[tuple[str, int], float]:
    """Return the fixed trajectory-step-token weight ``1 / (T_i * L_i,t)``."""

    rows = validate_active_trajectory_records(records, require_v2=require_v2)
    by_trajectory: dict[str, list[TrainingTrajectoryRecord]] = defaultdict(list)
    for row in rows:
        by_trajectory[row.traj_uid].append(row)
    return {
        (row.traj_uid, row.step_idx): 1.0 / (len(trajectory) * row.action_token_count)
        for trajectory in by_trajectory.values()
        for row in trajectory
    }


def summarize_trajectory_weights(
    records: Iterable[TrainingTrajectoryRecord], *, require_v2: bool = True
) -> TrajectoryWeightSummary:
    """Expose the length-to-gradient-mass diagnostic required by the study."""

    rows = validate_active_trajectory_records(records, require_v2=require_v2)
    weights = trajectory_step_token_weights(rows, require_v2=require_v2)
    lengths: dict[str, int] = defaultdict(int)
    masses: dict[str, float] = defaultdict(float)
    tokens = 0
    for row in rows:
        lengths[row.traj_uid] += 1
        tokens += row.action_token_count
        masses[row.traj_uid] += weights[(row.traj_uid, row.step_idx)] * row.action_token_count
    return TrajectoryWeightSummary(
        trajectories=len(lengths),
        transitions=len(rows),
        token_count=tokens,
        total_gradient_mass_by_trajectory=dict(sorted(masses.items())),
        rollout_length_gradient_mass_correlation=_pearson(
            list(lengths.values()), list(masses.values())
        ),
    )


def _require_v2_metadata(row: TrainingTrajectoryRecord) -> None:
    if row.trajectory_length is None or row.trajectory_length < 1:
        raise TrajectoryContractError("v2 active transition requires trajectory_length")
    if not row.terminal_reason:
        raise TrajectoryContractError("v2 active transition requires terminal_reason")
    for field, value in (
        ("state_snapshot_sha256", row.state_snapshot_sha256),
        ("goal_hash", row.goal_hash),
        ("action_affordance_hash", row.action_affordance_hash),
        ("rollout_config_hash", row.rollout_config_hash),
    ):
        if not _SHA256.fullmatch(value or ""):
            raise TrajectoryContractError(
                f"v2 active transition requires lowercase SHA-256 {field}"
            )
    if row.remaining_budget is None or row.remaining_budget < 0:
        raise TrajectoryContractError("v2 active transition requires non-negative remaining_budget")
    if row.extra.get("remaining_budget_verified") is not True:
        raise TrajectoryContractError(
            "v2 active transition requires environment-verified remaining_budget"
        )
    if row.extra.get("action_affordance_verified") is not True:
        raise TrajectoryContractError(
            "v2 active transition requires environment-verified action affordances"
        )
    if row.extra.get("action_token_count_verified") is not True:
        raise TrajectoryContractError(
            "v2 active transition requires tokenizer-verified action_token_count"
        )


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return None if denominator == 0 else numerator / denominator
