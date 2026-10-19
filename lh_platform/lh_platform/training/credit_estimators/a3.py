"""Within-observation behavioral-kernel diagnostic.

The public ``A3-G2PO`` name is reserved for the future estimator that passes a
held-out behavior-distance gate and reuses evidence across observations.  This
module deliberately keeps the historical class names as import compatibility,
but it cannot publish an A3 method claim while candidates remain restricted to
one coarse observation group.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from .contracts import (
    CreditEstimate,
    CreditEvidenceError,
    CreditPrediction,
    CreditVisit,
    Exclusion,
    summarize_predictions,
    validate_visits,
)


@dataclass(frozen=True)
class A3Config:
    components: tuple[str, ...] = ("progress", "transition")
    bandwidth: float = 1.0
    horizon_scaling: bool = True
    pooling: Literal["fixed", "adaptive"] = "adaptive"
    fixed_pooling_weight: float = 0.5
    exclusion: Exclusion = "trajectory"
    claim_mode: Literal["full", "frozen_diagnostic", "diagnostic"] = "diagnostic"

    def __post_init__(self) -> None:
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("A3 components must be non-empty and unique")
        if not math.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("A3 bandwidth must be positive")
        if not 0.0 <= self.fixed_pooling_weight <= 1.0:
            raise ValueError("fixed_pooling_weight must be between zero and one")
        if self.pooling not in {"fixed", "adaptive"}:
            raise ValueError("A3 pooling must be fixed or adaptive")
        if self.claim_mode not in {"full", "frozen_diagnostic", "diagnostic"}:
            raise ValueError("A3 claim_mode must be full, frozen_diagnostic, or diagnostic")


class A3Estimator:
    """Compatibility wrapper for a within-observation kernel diagnostic."""

    estimator_id = "within_observation_behavior_kernel"

    def __init__(self, config: A3Config | None = None) -> None:
        self.config = config or A3Config()

    def fit_predict(self, visits: Sequence[CreditVisit]) -> CreditEstimate:
        rows = validate_visits(visits)
        evaluation = tuple(row for row in rows if row.evaluation_visit)
        # Every row can enter a fit neighborhood even when it is not scored.
        # Validate the entire candidate pool so a hidden non-evaluation row
        # cannot silently change the action coordinate system or lineage.
        self._validate_evidence(rows)
        predictions = []
        for target in evaluation:
            candidates = [
                row
                for row in rows
                if row.coarse_key == target.coarse_key
                and (
                    self.config.exclusion != "trajectory"
                    or row.trajectory_uid != target.trajectory_uid
                )
            ]
            if not candidates:
                continue
            contributions = self._trajectory_contributions(target, candidates)
            coarse_return = mean(item.return_value for item in contributions)
            coarse_success = mean(item.success_value for item in contributions)
            positive = [item for item in contributions if item.weight > 1e-12]
            total_weight = sum(item.weight for item in positive)
            if total_weight <= 0:
                fine_return, fine_success, fine_ess = coarse_return, coarse_success, 0.0
            else:
                fine_return = (
                    sum(item.weight * item.return_value for item in positive) / total_weight
                )
                fine_success = (
                    sum(item.weight * item.success_value for item in positive) / total_weight
                )
                fine_ess = total_weight**2 / sum(item.weight**2 for item in positive)
            if self.config.pooling == "fixed":
                return_weight = success_weight = self.config.fixed_pooling_weight
            else:
                return_weight = self._adaptive_weight(
                    positive,
                    contributions,
                    outcome="return",
                    ess=fine_ess,
                )
                success_weight = self._adaptive_weight(
                    positive,
                    contributions,
                    outcome="success",
                    ess=fine_ess,
                )
            predictions.append(
                CreditPrediction(
                    visit_uid=target.visit_uid,
                    trajectory_uid=target.trajectory_uid,
                    predicted_return=(1.0 - return_weight) * coarse_return
                    + return_weight * fine_return,
                    predicted_success=(1.0 - success_weight) * coarse_success
                    + success_weight * fine_success,
                    coarse_return=coarse_return,
                    fine_return=fine_return,
                    pooling_weight=return_weight,
                    coarse_ess=float(len(contributions)),
                    fine_ess=fine_ess,
                    excluded_trajectory=self.config.exclusion == "trajectory",
                    success_pooling_weight=success_weight,
                )
            )
        result = tuple(predictions)
        return CreditEstimate(
            estimator_id=(
                f"a3_{'-'.join(self.config.components)}_{self.config.pooling}_"
                f"bw{self.config.bandwidth:g}_{self.config.exclusion}"
            ),
            estimator_role="within_observation_kernel_diagnostic_not_a3",
            predictions=result,
            metrics=summarize_predictions(rows, result),
            metadata=(
                ("components", self.config.components),
                ("bandwidth", self.config.bandwidth),
                ("horizon_scaling", self.config.horizon_scaling),
                (
                    "horizon_kernel_semantics",
                    "query_conditioned_target_remaining_horizon"
                    if self.config.horizon_scaling
                    else "unscaled",
                ),
                ("pooling", self.config.pooling),
                ("exclusion", self.config.exclusion),
                ("claim_mode", self.config.claim_mode),
                ("candidate_scope", "within_coarse_observation"),
            ),
        )

    def _validate_evidence(self, rows: Sequence[CreditVisit]) -> None:
        for row in rows:
            row.feature_vector(self.config.components)
        by_coarse: dict[object, set[tuple[tuple[str, ...], tuple[float, ...]]]] = {}
        for row in rows:
            if row.behavior is None:
                raise CreditEvidenceError("credit visit lacks frozen behavior evidence")
            by_coarse.setdefault(row.coarse_key, set()).add(
                (row.behavior.action_ids, row.behavior.action_weights)
            )
        if any(len(action_axes) != 1 for action_axes in by_coarse.values()):
            raise CreditEvidenceError(
                "A3 behavior action axes differ within one coarse observation group"
            )
        if self.config.claim_mode == "diagnostic":
            return
        if any(
            row.behavior is None
            or row.behavior.embedding_source != "frozen_actor_behavior_head"
            or not row.behavior.actor_visible
            or not row.behavior.frozen
            or not row.behavior.source_batch_id
            or not row.behavior.target_batch_id
            or row.behavior.source_batch_id == row.behavior.target_batch_id
            or not _sha256(row.behavior.source_artifact_sha256)
            or not _sha256(row.behavior.target_artifact_sha256)
            or row.behavior.source_artifact_sha256 == row.behavior.target_artifact_sha256
            or not _sha256(row.behavior.policy_checkpoint_sha256)
            or not _sha256(row.behavior.materialization_sha256)
            for row in rows
        ):
            raise CreditEvidenceError("full A3 requires frozen actor-visible behavior features")
        if self.config.claim_mode == "full":
            raise CreditEvidenceError(
                "full A3-G2PO is not implemented: bounded cross-observation "
                "retrieval and its held-out behavior-distance gate are required"
            )

    def _kernel(self, left: CreditVisit, right: CreditVisit) -> float:
        left_vector = left.feature_vector(self.config.components)
        right_vector = right.feature_vector(self.config.components)
        if len(left_vector) != len(right_vector):
            raise CreditEvidenceError("A3 behavior feature dimensions drifted")
        distance = math.sqrt(
            sum(
                (lvalue - rvalue) ** 2
                for lvalue, rvalue in zip(left_vector, right_vector, strict=False)
            )
        )
        if self.config.horizon_scaling:
            distance *= max(1, left.remaining_horizon)
        return math.exp(-0.5 * (distance / self.config.bandwidth) ** 2)

    def _trajectory_contributions(
        self, target: CreditVisit, candidates: Sequence[CreditVisit]
    ) -> tuple[_KernelContribution, ...]:
        by_trajectory: dict[str, list[CreditVisit]] = {}
        for row in candidates:
            by_trajectory.setdefault(row.trajectory_uid, []).append(row)
        return tuple(
            _KernelContribution(
                trajectory_uid=trajectory_uid,
                members=tuple(members),
                weight=mean(self._kernel(target, row) for row in members),
                return_value=mean(row.return_value for row in members),
                success_value=mean(float(row.success) for row in members),
            )
            for trajectory_uid, members in sorted(by_trajectory.items())
        )

    def _adaptive_weight(
        self,
        weighted: Sequence[_KernelContribution],
        all_contributions: Sequence[_KernelContribution],
        *,
        outcome: Literal["return", "success"],
        ess: float,
    ) -> float:
        if not weighted or ess <= 1.0:
            return 0.0
        total = sum(item.weight for item in weighted)
        values = [
            (
                item.weight,
                item.return_value if outcome == "return" else item.success_value,
            )
            for item in weighted
        ]
        weighted_mean = sum(weight * value for weight, value in values) / total
        noise = sum(weight * (value - weighted_mean) ** 2 for weight, value in values) / total
        local_means = []
        local_noises = []
        for center in all_contributions:
            neighbors = []
            for other in all_contributions:
                if other.trajectory_uid == center.trajectory_uid:
                    continue
                weight = self._contribution_kernel(center, other)
                if weight <= 1e-12:
                    continue
                value = other.return_value if outcome == "return" else other.success_value
                neighbors.append((weight, value))
            neighbor_weight = sum(weight for weight, _ in neighbors)
            if neighbor_weight <= 0:
                continue
            local_mean = sum(weight * value for weight, value in neighbors) / neighbor_weight
            neighbor_ess = neighbor_weight**2 / sum(weight**2 for weight, _ in neighbors)
            local_noise = (
                sum(weight * (value - local_mean) ** 2 for weight, value in neighbors)
                / neighbor_weight
            )
            local_means.append(local_mean)
            local_noises.append(local_noise / neighbor_ess)
        if len(local_means) < 2:
            return 0.0
        center = mean(local_means)
        between = mean((value - center) ** 2 for value in local_means)
        between = max(0.0, between - mean(local_noises))
        denominator = between + noise / ess
        return between / denominator if denominator > 0 else 0.0

    def _contribution_kernel(self, left: _KernelContribution, right: _KernelContribution) -> float:
        """Compare trajectories with the same visit-averaged kernel semantics.

        Using an arbitrary first visit makes the empirical-Bayes nuisance
        estimate depend on record ordering.  The average pairwise kernel is
        permutation invariant and matches the trajectory-equal weighting used
        for the target neighborhood.
        """

        return mean(
            self._kernel(left_visit, right_visit)
            for left_visit in left.members
            for right_visit in right.members
        )


@dataclass(frozen=True)
class _KernelContribution:
    trajectory_uid: str
    members: tuple[CreditVisit, ...]
    weight: float
    return_value: float
    success_value: float


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
