"""Public contracts for frozen-batch credit-estimator experiments.

These records sit after environment/verifier annotation and before PPO.  They
must not infer rewards, phases, or behavior representations themselves; those
remain owned by the benchmark collector and the frozen representation job.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


class CreditEvidenceError(ValueError):
    """Raised when an estimator would exceed the evidence carried by a batch."""


@dataclass(frozen=True)
class FrozenBehaviorEvidence:
    """Auditable A3 feature sidecar with a canonical action coordinate system."""

    features: tuple[tuple[str, tuple[float, ...]], ...]
    action_ids: tuple[str, ...]
    action_weights: tuple[float, ...]
    embedding_source: str
    actor_visible: bool
    frozen: bool
    source_batch_id: str
    target_batch_id: str
    source_artifact_sha256: str = ""
    target_artifact_sha256: str = ""
    policy_checkpoint_sha256: str = ""
    materialization_sha256: str = ""

    def validate(self) -> FrozenBehaviorEvidence:
        if not self.action_ids or len(set(self.action_ids)) != len(self.action_ids):
            raise CreditEvidenceError("behavior action IDs must be non-empty and unique")
        if any(not action_id for action_id in self.action_ids):
            raise CreditEvidenceError("behavior action ID is empty")
        if (
            len(self.action_weights) != len(self.action_ids)
            or any(not math.isfinite(value) or value <= 0 for value in self.action_weights)
            or not math.isclose(sum(self.action_weights), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise CreditEvidenceError("behavior action weights must be positive and sum to one")
        names = [name for name, _ in self.features]
        if len(names) != len(set(names)):
            raise CreditEvidenceError("behavior feature component names are duplicated")
        for name, vector in self.features:
            if not name or not vector or not all(math.isfinite(value) for value in vector):
                raise CreditEvidenceError("behavior features must be named finite vectors")
            if len(vector) % len(self.action_ids) != 0:
                raise CreditEvidenceError(
                    "behavior feature dimensions do not align with canonical action IDs"
                )
        return self

    def feature_vector(self, components: Sequence[str]) -> tuple[float, ...]:
        values = dict(self.features)
        missing = [component for component in components if component not in values]
        if missing:
            raise CreditEvidenceError("missing behavior feature components: " + ", ".join(missing))
        return tuple(value for component in components for value in values[component])


@dataclass(frozen=True)
class CreditVisit:
    """One immutable estimator observation.

    ``behavior_features`` is deliberately a named tuple of frozen vectors.  A
    full A3 claim requires actor-visible features produced before this
    estimator runs; the estimator never derives them from outcome labels.
    """

    visit_uid: str
    trajectory_uid: str
    dataset_id: str
    benchmark: str
    task_id: str
    step_idx: int
    coarse_key: Hashable
    phase_key: Hashable
    history_key: Hashable | None
    exact_key: Hashable | None
    return_value: float
    success: bool
    remaining_horizon: int
    evaluation_visit: bool = True
    behavior: FrozenBehaviorEvidence | None = None

    def validate(self) -> CreditVisit:
        if not all((self.visit_uid, self.trajectory_uid, self.dataset_id, self.benchmark)):
            raise CreditEvidenceError("credit visit identity is incomplete")
        if self.step_idx < 0 or self.remaining_horizon < 0:
            raise CreditEvidenceError("step index and remaining horizon must be non-negative")
        if not math.isfinite(self.return_value):
            raise CreditEvidenceError("credit return must be finite")
        if self.behavior is not None:
            self.behavior.validate()
        return self

    def feature_vector(self, components: Sequence[str]) -> tuple[float, ...]:
        if self.behavior is None:
            raise CreditEvidenceError("credit visit lacks frozen behavior evidence")
        return self.behavior.feature_vector(components)


@dataclass(frozen=True)
class CreditPrediction:
    visit_uid: str
    trajectory_uid: str
    predicted_return: float
    predicted_success: float
    coarse_return: float
    fine_return: float
    pooling_weight: float
    coarse_ess: float
    fine_ess: float
    excluded_trajectory: bool
    success_pooling_weight: float | None = None


@dataclass(frozen=True)
class CreditMetrics:
    prediction_count: int
    evaluation_count: int
    coverage: float
    return_mse: float | None
    success_brier: float | None
    fine_ess_mean: float
    fine_ess_min: float
    pooling_weight_mean: float | None
    success_pooling_weight_mean: float | None = None


@dataclass(frozen=True)
class CreditEstimate:
    estimator_id: str
    estimator_role: str
    predictions: tuple[CreditPrediction, ...]
    metrics: CreditMetrics
    metadata: tuple[tuple[str, object], ...] = ()


class CreditEstimator(Protocol):
    """One frozen-data estimator arm."""

    estimator_id: str

    def fit_predict(self, visits: Sequence[CreditVisit]) -> CreditEstimate: ...


Exclusion = Literal["none", "trajectory"]


def validate_visits(visits: Sequence[CreditVisit]) -> tuple[CreditVisit, ...]:
    rows = tuple(row.validate() for row in visits)
    if not rows:
        raise CreditEvidenceError("credit estimator batch is empty")
    visit_uids = [row.visit_uid for row in rows]
    if len(visit_uids) != len(set(visit_uids)):
        raise CreditEvidenceError("credit visit_uid is duplicated")
    return rows


def phase_aliasing_group_counts(visits: Sequence[object]) -> tuple[int, int]:
    """Count coarse groups and multi-trajectory groups spanning fine identities."""

    phases_by_coarse: dict[object, set[object]] = defaultdict(set)
    trajectories_by_coarse: dict[object, set[str]] = defaultdict(set)
    for row in visits:
        coarse_key = row.coarse_key
        phases_by_coarse[coarse_key].add(row.phase_key)
        trajectories_by_coarse[coarse_key].add(str(row.trajectory_uid))
    cross_phase = sum(
        len(phases) > 1 and len(trajectories_by_coarse[coarse_key]) > 1
        for coarse_key, phases in phases_by_coarse.items()
    )
    return len(phases_by_coarse), cross_phase


def summarize_predictions(
    rows: Sequence[CreditVisit], predictions: Sequence[CreditPrediction]
) -> CreditMetrics:
    targets = {row.visit_uid: row for row in rows if row.evaluation_visit}
    predicted = [prediction for prediction in predictions if prediction.visit_uid in targets]
    return CreditMetrics(
        prediction_count=len(predicted),
        evaluation_count=len(targets),
        coverage=len(predicted) / len(targets) if targets else 0.0,
        return_mse=(
            sum(
                (prediction.predicted_return - targets[prediction.visit_uid].return_value) ** 2
                for prediction in predicted
            )
            / len(predicted)
            if predicted
            else None
        ),
        success_brier=(
            sum(
                (prediction.predicted_success - float(targets[prediction.visit_uid].success)) ** 2
                for prediction in predicted
            )
            / len(predicted)
            if predicted
            else None
        ),
        fine_ess_mean=(
            sum(prediction.fine_ess for prediction in predicted) / len(predicted)
            if predicted
            else 0.0
        ),
        fine_ess_min=min((prediction.fine_ess for prediction in predicted), default=0.0),
        pooling_weight_mean=(
            sum(prediction.pooling_weight for prediction in predicted) / len(predicted)
            if predicted
            else None
        ),
        success_pooling_weight_mean=(
            sum(
                prediction.success_pooling_weight
                if prediction.success_pooling_weight is not None
                else prediction.pooling_weight
                for prediction in predicted
            )
            / len(predicted)
            if predicted
            else None
        ),
    )
