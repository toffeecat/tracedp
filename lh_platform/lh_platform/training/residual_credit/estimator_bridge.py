"""Adapter from Pilot A calibration visits onto credit_estimators.

This is the seam that collapses the residual_credit PhasePool / TN-LOTO /
matched-random shadow: collection stays in residual_credit; fit_predict lives
only in credit_estimators.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import replace
from statistics import mean
from typing import Any, Literal, Protocol

from lh_platform.training.credit_estimators import (
    CreditEstimate,
    CreditVisit,
    GroupedMeanConfig,
    GroupedMeanEstimator,
    MatchedRandomConfig,
    MatchedRandomEstimator,
    PhasePoolConfig,
    PhasePoolEstimator,
)

GroupedKey = Literal["coarse", "phase", "exact"]


class _CalibrationRow(Protocol):
    visit_uid: str
    trajectory_uid: str
    pair_uid: str
    step_idx: int
    coarse_key: Hashable
    refined_key: Hashable
    exact_key: Hashable
    discounted_return: float
    success: bool
    remaining_budget: int | None
    evaluation_visit: bool
    progress_sign: int


def credit_visits_from_calibration(
    rows: Sequence[_CalibrationRow],
) -> tuple[CreditVisit, ...]:
    """Map Pilot A visits onto the shared CreditVisit interface.

    ``refined_key`` becomes ``phase_key`` — the fine identity PhasePool and
    MatchedRandom already shrink or shuffle against ``coarse_key``.
    """

    visits: list[CreditVisit] = []
    for row in rows:
        if row.remaining_budget is None:
            raise ValueError("budget-refined calibration requires remaining_budget on every visit")
        visits.append(
            CreditVisit(
                visit_uid=row.visit_uid,
                trajectory_uid=row.trajectory_uid,
                dataset_id=row.pair_uid,
                benchmark="residual_credit_calibration",
                task_id=row.pair_uid,
                step_idx=row.step_idx,
                coarse_key=row.coarse_key,
                phase_key=row.refined_key,
                history_key=None,
                exact_key=row.exact_key,
                return_value=row.discounted_return,
                success=row.success,
                remaining_horizon=row.remaining_budget,
                evaluation_visit=row.evaluation_visit,
            ).validate()
        )
    return tuple(visits)


def fit_grouped_loto(
    credit_visits: Sequence[CreditVisit],
    *,
    key: GroupedKey,
    exclusion: Literal["none", "trajectory"] = "trajectory",
    weighting: Literal["visit", "trajectory"] = "trajectory",
) -> CreditEstimate:
    return GroupedMeanEstimator(
        GroupedMeanConfig(key=key, weighting=weighting, exclusion=exclusion)
    ).fit_predict(credit_visits)


def fit_phasepool_eb(
    credit_visits: Sequence[CreditVisit],
) -> CreditEstimate:
    return PhasePoolEstimator(
        PhasePoolConfig(pooling="empirical_bayes", exclusion="trajectory")
    ).fit_predict(credit_visits)


def fit_matched_random(
    credit_visits: Sequence[CreditVisit],
    *,
    seed: int,
) -> CreditEstimate:
    return MatchedRandomEstimator(
        MatchedRandomConfig(seed=seed, exclusion="trajectory")
    ).fit_predict(credit_visits)


def prediction_map(
    estimate: CreditEstimate,
) -> dict[str, tuple[float, float]]:
    return {
        row.visit_uid: (row.predicted_return, row.predicted_success) for row in estimate.predictions
    }


def all_visit_predictions(
    credit_visits: Sequence[CreditVisit],
    *,
    key: GroupedKey,
    exclusion: Literal["none", "trajectory"] = "trajectory",
    weighting: Literal["visit", "trajectory"] = "trajectory",
) -> dict[str, tuple[float, float]]:
    """Predict every visit so Pilot A edge diagnostics can score transitions."""

    scored = tuple(replace(row, evaluation_visit=True) for row in credit_visits)
    return prediction_map(
        fit_grouped_loto(scored, key=key, exclusion=exclusion, weighting=weighting)
    )


def summarize_hard_split_estimate(
    rows: Sequence[_CalibrationRow],
    estimate: CreditEstimate,
    *,
    key_of: Mapping[str, Hashable],
    success_reward: float,
    all_predictions: Mapping[str, tuple[float, float]],
    by_uid: Mapping[str, _CalibrationRow],
    value_ranking: Callable[..., float | None],
) -> dict[str, Any]:
    """Attach Pilot-A-only diagnostics on top of a shared CreditEstimate."""

    evaluation = tuple(row for row in rows if row.evaluation_visit)
    evaluation_keys = {key_of[row.visit_uid] for row in evaluation}
    group_sizes = [
        len({row.trajectory_uid for row in rows if key_of[row.visit_uid] == key})
        for key in evaluation_keys
    ]
    singleton_groups = sum(size < 2 for size in group_sizes)
    evaluation_predictions = {
        row.visit_uid: all_predictions[row.visit_uid]
        for row in evaluation
        if row.visit_uid in all_predictions
    }
    edge_rows: list[tuple[int, float]] = []
    progress_edges: list[float] = []
    start_progress_edges: list[float] = []
    for row in rows:
        if row.visit_uid not in all_predictions:
            continue
        following = _next_predicted_value(
            row,
            by_uid=by_uid,
            predictions=all_predictions,
            success_reward=success_reward,
        )
        if following is None or row.progress_sign == 0:
            continue
        edge = following - all_predictions[row.visit_uid][0]
        edge_rows.append((row.progress_sign, edge))
        if row.progress_sign > 0:
            progress_edges.append(edge)
            if row.evaluation_visit:
                start_progress_edges.append(edge)
    edge_correct = [(edge > 0 if sign > 0 else edge < 0) for sign, edge in edge_rows]
    metrics = estimate.metrics
    return {
        "group_count": len(group_sizes),
        "trajectory_ess_min": min(group_sizes) if group_sizes else 0,
        "trajectory_ess_mean": mean(group_sizes) if group_sizes else 0.0,
        "singleton_group_rate": (singleton_groups / len(group_sizes) if group_sizes else 1.0),
        "loto_prediction_count": metrics.prediction_count,
        "loto_coverage": metrics.coverage,
        "return_mse": metrics.return_mse,
        "success_brier": metrics.success_brier,
        "value_ranking_accuracy": value_ranking(
            evaluation,
            predictions=evaluation_predictions,
        ),
        "verifier_edge_sign_count": len(edge_rows),
        "verifier_edge_sign_accuracy": mean(edge_correct) if edge_correct else None,
        "good_action_false_punishment_rate": (
            mean(edge <= 0 for edge in progress_edges) if progress_edges else None
        ),
        "early_action_false_punishment_rate": (
            mean(edge <= 0 for edge in start_progress_edges) if start_progress_edges else None
        ),
    }


def summarize_phasepool_estimate(
    rows: Sequence[_CalibrationRow],
    estimate: CreditEstimate,
    *,
    value_ranking: Callable[..., float | None],
) -> dict[str, Any]:
    evaluation = tuple(row for row in rows if row.evaluation_visit)
    predictions = prediction_map(estimate)
    evaluation_predictions = {
        row.visit_uid: predictions[row.visit_uid]
        for row in evaluation
        if row.visit_uid in predictions
    }
    lambdas = [row.pooling_weight for row in estimate.predictions]
    fine_ess = [int(row.fine_ess) for row in estimate.predictions]
    metrics = estimate.metrics
    return {
        "estimator_role": "oracle_state_partial_pooling_diagnostic_not_a3",
        "loto_prediction_count": metrics.prediction_count,
        "loto_coverage": metrics.coverage,
        "return_mse": metrics.return_mse,
        "success_brier": metrics.success_brier,
        "value_ranking_accuracy": value_ranking(
            evaluation,
            predictions=evaluation_predictions,
        ),
        "pooling_weight_mean": metrics.pooling_weight_mean,
        "pooling_weight_min": min(lambdas) if lambdas else None,
        "pooling_weight_max": max(lambdas) if lambdas else None,
        "fine_trajectory_ess_mean": mean(fine_ess) if fine_ess else 0.0,
        "fallback_to_coarse_rate": (mean(weight == 0.0 for weight in lambdas) if lambdas else None),
    }


def summarize_matched_random_estimate(estimate: CreditEstimate) -> dict[str, Any]:
    metrics = estimate.metrics
    metadata = dict(estimate.metadata)
    original_sizes = metadata.get("original_group_sizes", ())
    return {
        "loto_prediction_count": metrics.prediction_count,
        "loto_coverage": metrics.coverage,
        "return_mse": metrics.return_mse,
        "success_brier": metrics.success_brier,
        # Size-matched sham: report the preserved fine-group occupancy, not
        # LOTO fine_ess (which excludes the evaluated trajectory).
        "trajectory_ess_mean": (
            mean(float(size) for size in original_sizes) if original_sizes else 0.0
        ),
    }


def _next_predicted_value(
    row: _CalibrationRow,
    *,
    by_uid: Mapping[str, _CalibrationRow],
    predictions: Mapping[str, tuple[float, float]],
    success_reward: float,
) -> float | None:
    terminal = getattr(row, "terminal", False)
    if terminal:
        return success_reward if row.success else 0.0
    next_uid = getattr(row, "next_visit_uid", None)
    if next_uid is None or next_uid not in by_uid:
        return None
    prediction = predictions.get(str(next_uid))
    return prediction[0] if prediction is not None else None


__all__ = [
    "all_visit_predictions",
    "credit_visits_from_calibration",
    "fit_grouped_loto",
    "fit_matched_random",
    "fit_phasepool_eb",
    "prediction_map",
    "summarize_hard_split_estimate",
    "summarize_matched_random_estimate",
    "summarize_phasepool_estimate",
]
