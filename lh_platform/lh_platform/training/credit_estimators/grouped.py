"""G2PO-style grouped empirical means used as frozen-data baselines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
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
class GroupedMeanConfig:
    key: Literal["coarse", "phase", "history", "exact"] = "coarse"
    weighting: Literal["visit", "trajectory"] = "trajectory"
    exclusion: Exclusion = "trajectory"


class GroupedMeanEstimator:
    """Full-pooling G2PO endpoint with explicit occupancy and leakage choices."""

    estimator_id = "grouped_mean"

    def __init__(self, config: GroupedMeanConfig | None = None) -> None:
        self.config = config or GroupedMeanConfig()

    def fit_predict(self, visits: Sequence[CreditVisit]) -> CreditEstimate:
        rows = validate_visits(visits)
        key_of = {
            "coarse": lambda row: row.coarse_key,
            "phase": lambda row: row.phase_key,
            "history": lambda row: row.history_key,
            "exact": lambda row: row.exact_key,
        }[self.config.key]
        if self.config.key in {"history", "exact"} and any(key_of(row) is None for row in rows):
            raise ValueError(f"{self.config.key} grouped estimator lacks required identity")
        by_key: dict[Hashable, list[CreditVisit]] = defaultdict(list)
        for row in rows:
            by_key[key_of(row)].append(row)
        predictions = []
        for target in rows:
            if not target.evaluation_visit:
                continue
            members = [
                row
                for row in by_key[key_of(target)]
                if self.config.exclusion != "trajectory"
                or row.trajectory_uid != target.trajectory_uid
            ]
            values = _values(members, weighting=self.config.weighting)
            if not values:
                continue
            return_value = mean(value[0] for value in values)
            success_value = mean(value[1] for value in values)
            predictions.append(
                CreditPrediction(
                    visit_uid=target.visit_uid,
                    trajectory_uid=target.trajectory_uid,
                    predicted_return=return_value,
                    predicted_success=success_value,
                    coarse_return=return_value,
                    fine_return=return_value,
                    pooling_weight=0.0,
                    coarse_ess=float(len(values)),
                    fine_ess=float(len(values)),
                    excluded_trajectory=self.config.exclusion == "trajectory",
                )
            )
        result = tuple(predictions)
        return CreditEstimate(
            estimator_id=(
                f"g2po_{self.config.weighting}_{self.config.key}_{self.config.exclusion}"
            ),
            estimator_role="g2po_frozen_credit_baseline",
            predictions=result,
            metrics=summarize_predictions(rows, result),
            metadata=(
                ("key", self.config.key),
                ("weighting", self.config.weighting),
                ("exclusion", self.config.exclusion),
            ),
        )


def _values(
    members: Sequence[CreditVisit], *, weighting: Literal["visit", "trajectory"]
) -> list[tuple[float, float]]:
    if weighting == "visit":
        return [(row.return_value, float(row.success)) for row in members]
    by_trajectory: dict[str, list[CreditVisit]] = defaultdict(list)
    for row in members:
        by_trajectory[row.trajectory_uid].append(row)
    return [
        (
            mean(row.return_value for row in trajectory),
            mean(float(row.success) for row in trajectory),
        )
        for trajectory in by_trajectory.values()
    ]
