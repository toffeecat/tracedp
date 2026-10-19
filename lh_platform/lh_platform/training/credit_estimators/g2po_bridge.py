"""Bridge cross-fitted estimators into the unchanged G2PO advantage seam.

Also owns the **online** Verifier-Refined routing into G2PO-TN with refined
keys.  Validation of env / goal-graph semantics stays in ``training.dppo``;
this module only builds Credit State Identity keys and crosses the G2PO seam.

Two Verifier-Refined paths, one locality:

* Online full-pool P0 → :func:`assign_verifier_refined_g2po_credit`
* Cross-fitted GroupedMean(phase) / PhasePool → :func:`assign_estimator_g2po_credit`
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from lh_platform.training.g2po.credit import (
    G2POCreditAssignment,
    G2POCreditConfig,
    G2POCreditMode,
    G2POTransition,
    assign_g2po_credit_with_state_values,
    assign_g2po_tn_with_state_keys,
)

from .contracts import CreditEstimate, CreditEstimator, CreditEvidenceError, CreditVisit
from .credit_keys import verifier_refined_online_key


@dataclass(frozen=True)
class EstimatorG2POCreditResult:
    estimate: CreditEstimate
    assignments: tuple[G2POCreditAssignment, ...]


def assign_estimator_g2po_credit(
    transitions: Sequence[G2POTransition],
    visits: Sequence[CreditVisit],
    *,
    estimator: CreditEstimator,
    node_identity: Literal["coarse", "phase"],
    mode: G2POCreditMode | str = G2POCreditMode.SOURCE,
    config: G2POCreditConfig | None = None,
) -> EstimatorG2POCreditResult:
    """Align by trajectory/step and preserve G2PO's discounted-return scale."""

    if len(transitions) != len(visits):
        raise CreditEvidenceError("G2PO transitions and credit visits differ in length")
    estimate = estimator.fit_predict(visits)
    predictions = {prediction.visit_uid: prediction for prediction in estimate.predictions}
    visit_by_identity = {(visit.trajectory_uid, visit.step_idx): visit for visit in visits}
    ordered_visits = []
    for transition in transitions:
        visit = visit_by_identity.get((transition.trajectory_id, transition.step_index))
        if visit is None or visit.visit_uid not in predictions:
            raise CreditEvidenceError(
                "cross-fitted estimator lacks full transition coverage; online update blocked"
            )
        ordered_visits.append(visit)
    keys = tuple(
        visit.coarse_key if node_identity == "coarse" else visit.phase_key
        for visit in ordered_visits
    )
    values = tuple(predictions[visit.visit_uid].predicted_return for visit in ordered_visits)
    assignments = assign_g2po_credit_with_state_values(
        transitions,
        state_keys=keys,
        state_values=values,
        mode=mode,
        config=config,
    )
    return EstimatorG2POCreditResult(estimate=estimate, assignments=assignments)


def assign_verifier_refined_g2po_credit(
    transitions: Sequence[G2POTransition],
    *,
    mode: str,
    config: G2POCreditConfig | None = None,
) -> tuple[G2POCreditAssignment, ...]:
    """Online Verifier-Refined: refined keys through the G2PO-TN seam.

    Callers must already have fail-closed validated transitions (``dppo`` owns
    that).  Prefer :class:`~lh_platform.training.agentic_rl_components.credit_method_adapters.EstimatorG2POMethod`
    with ``GroupedMeanEstimator(key=\"phase\")`` for cross-fitted / LOTO
    Verifier-Refined — that path uses :func:`assign_estimator_g2po_credit` and
    must not be silently swapped for this full-pool online route.
    """

    rows = tuple(transitions)
    if not rows:
        return ()
    keys = tuple(_verifier_refined_online_state_key(row, mode=mode) for row in rows)
    return assign_g2po_tn_with_state_keys(rows, state_keys=keys, config=config)


def _verifier_refined_online_state_key(row: G2POTransition, *, mode: str) -> Hashable:
    selected = str(mode)
    if selected == "verified_state_dp":
        identity = getattr(row, "state_identity", None)
        if identity is None:
            raise CreditEvidenceError(
                "verified_state_dp requires a verified state identity on every transition"
            )
        return identity.key
    if selected == "oracle_phase_refined_g2po_tn":
        residual = getattr(row, "residual_goal_signature", None)
        frontier = getattr(row, "frontier_signature", None)
        if not isinstance(residual, str) or not isinstance(frontier, str):
            raise CreditEvidenceError(
                "oracle Verifier-Refined requires residual and frontier signatures"
            )
        return verifier_refined_online_key(
            anchor_observation=_hashable(row.anchor_observation),
            residual_signature=residual,
            frontier_signature=frontier,
        )
    raise CreditEvidenceError(f"unsupported Verifier-Refined online mode {selected!r}")


def _hashable(value: Any) -> Hashable:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_hashable(item) for item in value))
    return value


__all__ = [
    "EstimatorG2POCreditResult",
    "assign_estimator_g2po_credit",
    "assign_verifier_refined_g2po_credit",
]
