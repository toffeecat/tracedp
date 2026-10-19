"""PhasePool-G2PO: phase-conditioned LOTO estimates with partial pooling."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from .contracts import (
    CreditEstimate,
    CreditPrediction,
    CreditVisit,
    Exclusion,
    summarize_predictions,
    validate_visits,
)


@dataclass(frozen=True)
class PhasePoolConfig:
    pooling: Literal[
        "hard",
        "fixed",
        "empirical_bayes",
        "beta_binomial",
    ] = "empirical_bayes"
    fixed_pooling_weight: float = 0.5
    exclusion: Exclusion = "trajectory"

    def __post_init__(self) -> None:
        if self.pooling not in {
            "hard",
            "fixed",
            "empirical_bayes",
            "beta_binomial",
        }:
            raise ValueError("unsupported PhasePool pooling mode")
        if not 0.0 <= self.fixed_pooling_weight <= 1.0:
            raise ValueError("fixed_pooling_weight must be between zero and one")


class PhasePoolEstimator:
    """Keep phase identity hard while shrinking only its empirical value."""

    estimator_id = "phasepool_g2po"

    def __init__(self, config: PhasePoolConfig | None = None) -> None:
        self.config = config or PhasePoolConfig()

    def fit_predict(self, visits: Sequence[CreditVisit]) -> CreditEstimate:
        rows = validate_visits(visits)
        evaluation = tuple(row for row in rows if row.evaluation_visit)
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
            coarse = _trajectory_values(candidates)
            fine = _trajectory_values(
                [row for row in candidates if row.phase_key == target.phase_key]
            )
            if not coarse:
                continue
            coarse_return = mean(value[0] for value in coarse)
            coarse_success = mean(value[1] for value in coarse)
            if not fine:
                fine_return, fine_success, weight = coarse_return, coarse_success, 0.0
            else:
                fine_return = mean(value[0] for value in fine)
                fine_success = mean(value[1] for value in fine)
                weight = self._weight(candidates, fine, outcome=0)
            success_weight = self._weight(candidates, fine, outcome=1) if fine else 0.0
            predictions.append(
                CreditPrediction(
                    visit_uid=target.visit_uid,
                    trajectory_uid=target.trajectory_uid,
                    predicted_return=(1.0 - weight) * coarse_return + weight * fine_return,
                    predicted_success=(1.0 - success_weight) * coarse_success
                    + success_weight * fine_success,
                    coarse_return=coarse_return,
                    fine_return=fine_return,
                    pooling_weight=weight,
                    coarse_ess=float(len(coarse)),
                    fine_ess=float(len(fine)),
                    excluded_trajectory=self.config.exclusion == "trajectory",
                    success_pooling_weight=success_weight,
                )
            )
        result = tuple(predictions)
        return CreditEstimate(
            estimator_id=(f"phasepool_{self.config.pooling}_{self.config.exclusion}"),
            estimator_role="phasepool_g2po",
            predictions=result,
            metrics=summarize_predictions(rows, result),
            metadata=(
                ("pooling", self.config.pooling),
                ("exclusion", self.config.exclusion),
                ("fixed_pooling_weight", self.config.fixed_pooling_weight),
            ),
        )

    def _weight(
        self,
        candidates: Sequence[CreditVisit],
        fine: Sequence[tuple[float, float]],
        *,
        outcome: int,
    ) -> float:
        if self.config.pooling == "hard":
            return 1.0
        if self.config.pooling == "fixed":
            return self.config.fixed_pooling_weight
        by_phase_trajectory: dict[object, dict[str, list[CreditVisit]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in candidates:
            by_phase_trajectory[row.phase_key][row.trajectory_uid].append(row)
        by_phase = {
            phase_key: [
                (
                    mean(row.return_value for row in trajectory),
                    mean(float(row.success) for row in trajectory),
                )
                for trajectory in trajectories.values()
            ]
            for phase_key, trajectories in by_phase_trajectory.items()
        }
        # Beta-Binomial is a model for binary success only.  Discounted return
        # remains a continuous target and therefore keeps the Gaussian
        # empirical-Bayes shrinkage used by the primary PhasePool estimator.
        if self.config.pooling == "beta_binomial" and outcome == 1:
            return _beta_binomial_weight(by_phase, fine_count=len(fine))
        phase_means = [mean(value[outcome] for value in values) for values in by_phase.values()]
        if len(phase_means) < 2:
            return 0.0
        between = _variance(phase_means)
        noise_terms = [
            _sample_variance([value[outcome] for value in values]) / len(values)
            for values in by_phase.values()
            if len(values) > 1
        ]
        between = max(0.0, between - (mean(noise_terms) if noise_terms else 0.0))
        fine_values = [value[outcome] for value in fine]
        noise = (
            _sample_variance(fine_values) / len(fine_values) if len(fine_values) > 1 else math.inf
        )
        denominator = between + noise
        return between / denominator if denominator > 0.0 and math.isfinite(denominator) else 0.0


def _trajectory_values(rows: Sequence[CreditVisit]) -> list[tuple[float, float]]:
    by_trajectory: dict[str, list[CreditVisit]] = defaultdict(list)
    for row in rows:
        by_trajectory[row.trajectory_uid].append(row)
    return [
        (
            mean(row.return_value for row in trajectory),
            mean(float(row.success) for row in trajectory),
        )
        for trajectory in by_trajectory.values()
    ]


def self_influence_diagnostic(visits: Sequence[CreditVisit]) -> dict[str, float | int]:
    """Compute self-influence I_i = V_in(s_i) - V_{-i}(s_i) per evaluation visit.

    Measures how much including trajectory i's own return changes the coarse
    group value estimate.  High self-influence indicates LOTO is critical.
    """

    rows = validate_visits(visits)
    evaluation = [row for row in rows if row.evaluation_visit]
    influences: list[float] = []
    for target in evaluation:
        same_coarse = [row for row in rows if row.coarse_key == target.coarse_key]
        all_values = _trajectory_values(same_coarse)
        excluded_values = _trajectory_values(
            [row for row in same_coarse if row.trajectory_uid != target.trajectory_uid]
        )
        if not all_values or not excluded_values:
            continue
        v_in = mean(value[0] for value in all_values)
        v_loto = mean(value[0] for value in excluded_values)
        influences.append(abs(v_in - v_loto))
    if not influences:
        return {"self_influence_mean": 0.0, "self_influence_max": 0.0, "evaluated_count": 0}
    return {
        "self_influence_mean": mean(influences),
        "self_influence_max": max(influences),
        "evaluated_count": len(influences),
    }


def _variance(values: Sequence[float]) -> float:
    center = mean(values)
    return mean((value - center) ** 2 for value in values)


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return math.inf
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def _beta_binomial_weight(
    by_phase: dict[object, list[tuple[float, float]]],
    *,
    fine_count: int,
) -> float:
    """Moment-estimated Beta prior weight for binary continuation success.

    The target trajectory has already been excluded.  With no identifiable
    between-phase heterogeneity the weight is zero; deterministic separation
    yields concentration zero and therefore the hard-split endpoint.
    """

    if len(by_phase) < 2 or fine_count < 1:
        return 0.0
    phase_rates = [mean(value[1] for value in values) for values in by_phase.values()]
    total_count = sum(len(values) for values in by_phase.values())
    if total_count <= 0:
        return 0.0
    global_rate = (
        sum(sum(value[1] for value in values) for values in by_phase.values()) / total_count
    )
    observed_between = _variance(phase_rates)
    sampling_noise = mean(
        global_rate * (1.0 - global_rate) / len(values) for values in by_phase.values() if values
    )
    latent_between = max(0.0, observed_between - sampling_noise)
    if latent_between <= 0.0:
        return 0.0
    concentration = max(
        0.0,
        global_rate * (1.0 - global_rate) / latent_between - 1.0,
    )
    return fine_count / (fine_count + concentration)
