"""TraceDP v2: recovery-value lookahead built on the adaptive information-gain base.

The base policy is ``select_information_gain_repair_reserve``, unchanged. TraceDP adds one
thing, a bounded lookahead that values a check by the recovery it opens rather than by how
surely it confirms the task: a refuted check reaches the Actor as feedback (or unlocks a
repair), and that raises the probability that the Actor resolves the failure.

Two guarantees make the increment attributable (REFLECTION_AND_REDESIGN_2026-10-16 §4.2):

* at horizon 1 the choice *is* the base choice, at every boundary;
* at deeper horizons the lookahead overrides the base only when its recovery value is
  strictly higher; any tie keeps the base choice.

The two resolution rates are frozen from the development and calibration groups' neutral
continuation runs (2026-10-16, Beta(1, 1) smoothing): the no-localization arm resolved 15
of 31 (no new information), and managed runs in which a refuted check reached the Actor
resolved 19 of 31 (db-wal-recovery, which no continuation can resolve, is included).
"""

from __future__ import annotations

from collections.abc import Mapping

from lh_platform.test_time_adaptation.runtime_information_gain_selector import (
    RuntimeInformationGainLiveResolver,
    _ENTROPY_FLOOR,
    _entropy,
    _probability,
    repair_reserve_units,
    select_information_gain_repair_reserve,
)
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionCandidate,
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import RuntimeJointResolvedChoice
from lh_platform.test_time_adaptation.runtime_owned_option_dp import (
    RuntimeAtomicActionEstimate,
    RuntimeOwnedOptionProjection,
)

RECOVERY_LOOKAHEAD_SELECTOR_ID = "tracedp_recovery_lookahead_on_information_gain_v1"
_MODEL_SCHEMA = "runtime_recovery_lookahead_selection_v1"
RESOLVE_WITHOUT_INFORMATION = (15 + 1) / (31 + 2)
RESOLVE_AFTER_REFUTED_FEEDBACK = (19 + 1) / (31 + 2)
RECOVERY_RATE_PROVENANCE = (
    "dev+calibration neutral-continuation runs 2026-10-16: no-localization 15/31, "
    "refuted-check feedback 19/31, Beta(1,1)"
)
_TIE = 1e-9


def _check_value(
    check: RuntimeJointActionCandidate,
    *,
    budget: int,
    remaining: tuple[RuntimeJointActionCandidate, ...],
    by_id: Mapping[str, RuntimeAtomicActionEstimate],
    repairs_after: Mapping[str, tuple[float, int]],
    reserve: int,
    depth: int,
) -> float:
    """Recovery value of running ``check`` now and acting well afterwards."""

    estimate = by_id[check.instance_action_id]
    refuted = _probability(estimate, "refuted")
    after_refuted = RESOLVE_AFTER_REFUTED_FEEDBACK
    repair = repairs_after.get(check.instance_action_id)
    if repair is not None and repair[1] <= budget - check.cost_units:
        supported, _cost = repair
        after_refuted = max(after_refuted, supported + (1.0 - supported) * RESOLVE_AFTER_REFUTED_FEEDBACK)
    rest = tuple(item for item in remaining if item.instance_action_id != check.instance_action_id)
    after_other = _state_value(
        budget=budget - check.cost_units, checks=rest, by_id=by_id,
        repairs_after=repairs_after, reserve=reserve, depth=depth - 1,
    )
    return refuted * after_refuted + (1.0 - refuted) * after_other


def _eligible(check: RuntimeJointActionCandidate, budget: int, reserve: int, by_id) -> bool:
    # Exactly the base policy's eligibility: affordable, keeps the repair reserve, uncertain.
    return (
        check.cost_units <= budget
        and budget - check.cost_units >= reserve
        and _entropy(by_id[check.instance_action_id]) > _ENTROPY_FLOOR
    )


def _state_value(*, budget, checks, by_id, repairs_after, reserve, depth) -> float:
    """Best recovery value from a state where no refutation has been seen yet."""

    best = RESOLVE_WITHOUT_INFORMATION  # continuing the Actor is always available
    if depth <= 0:
        return best
    for check in checks:
        if _eligible(check, budget, reserve, by_id):
            best = max(best, _check_value(
                check, budget=budget, remaining=checks, by_id=by_id,
                repairs_after=repairs_after, reserve=reserve, depth=depth,
            ))
    return best


def select_recovery_lookahead(
    slate: RuntimeJointActionSlate,
    estimates: tuple[RuntimeAtomicActionEstimate, ...],
    *,
    horizon: int,
) -> tuple[RuntimeJointActionCandidate, str, dict[str, object]]:
    """The base choice at horizon 1; a strictly better recovery value may override it."""

    base, base_rule, base_scores = select_information_gain_repair_reserve(slate, estimates)
    record: dict[str, object] = {"base_choice": base.instance_action_id, "base_rule": base_rule,
                                 "base_scores": base_scores, "horizon": horizon}
    if horizon <= 1 or base_rule == "time_reserve_submission":
        return base, f"base:{base_rule}", record
    by_id = {item.instance_action_id: item for item in estimates}
    budget = slate.budget_before
    reserve = repair_reserve_units(slate)
    checks = tuple(c for c in slate.candidates if c.action_type is RuntimeJointActionType.RUN_REGISTERED_CHECK)
    repairs_after: dict[str, tuple[float, int]] = {}
    for option in slate.conditional_future_options:
        if option.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR:
            estimate = by_id.get(option.instance_action_id)
            if estimate is not None:
                value = (_probability(estimate, "supported"), option.cost_units)
                previous = repairs_after.get(option.prerequisite_action_id)
                if previous is None or value[0] > previous[0]:
                    repairs_after[option.prerequisite_action_id] = value
    values: dict[str, float] = {}
    for candidate in slate.candidates:
        kind = candidate.action_type
        if kind is RuntimeJointActionType.CONTINUE_ACTOR:
            values[candidate.instance_action_id] = RESOLVE_WITHOUT_INFORMATION
        elif kind is RuntimeJointActionType.RUN_TARGETED_REPAIR and candidate.cost_units <= budget:
            supported = _probability(by_id[candidate.instance_action_id], "supported")
            values[candidate.instance_action_id] = supported + (1.0 - supported) * RESOLVE_WITHOUT_INFORMATION
        elif kind is RuntimeJointActionType.RUN_REGISTERED_CHECK and _eligible(candidate, budget, reserve, by_id):
            values[candidate.instance_action_id] = _check_value(
                candidate, budget=budget, remaining=checks, by_id=by_id,
                repairs_after=repairs_after, reserve=reserve, depth=horizon,
            )
    record["recovery_values"] = values
    base_value = values.get(base.instance_action_id, RESOLVE_WITHOUT_INFORMATION)
    best_id = min(values, key=lambda key: (-values[key], key)) if values else base.instance_action_id
    if values and values[best_id] > base_value + _TIE:
        chosen = next(c for c in slate.candidates if c.instance_action_id == best_id)
        return chosen, "lookahead_override", record
    return base, f"base:{base_rule}", record


class RuntimeRecoveryLookaheadLiveResolver(RuntimeInformationGainLiveResolver):
    """Same live binding and estimates as the base; adds only the recovery lookahead."""

    def _select(
        self,
        slate: RuntimeJointActionSlate,
        projection: RuntimeOwnedOptionProjection,
        estimates: tuple[RuntimeAtomicActionEstimate, ...],
        horizon: int,
    ) -> RuntimeJointResolvedChoice:
        selected, rule, record = select_recovery_lookahead(slate, estimates, horizon=horizon)
        return RuntimeJointResolvedChoice(
            selected,
            RECOVERY_LOOKAHEAD_SELECTOR_ID,
            dict(record.get("recovery_values") or {}),
            None,
            {
                "schema_version": _MODEL_SCHEMA,
                "selector_identity": RECOVERY_LOOKAHEAD_SELECTOR_ID,
                "fired_rule": rule,
                "horizon": horizon,
                "resolve_without_information": RESOLVE_WITHOUT_INFORMATION,
                "resolve_after_refuted_feedback": RESOLVE_AFTER_REFUTED_FEEDBACK,
                "recovery_rate_provenance": RECOVERY_RATE_PROVENANCE,
                "base_choice": record["base_choice"],
                "projection": projection.to_dict(),
                "atomic_estimates": [item.to_dict() for item in estimates],
            },
        )

    def _select_recorded(
        self,
        slate: RuntimeJointActionSlate,
        model: Mapping[str, object],
        estimates: tuple[RuntimeAtomicActionEstimate, ...],
    ) -> RuntimeJointResolvedChoice:
        horizon = model.get("horizon")
        if (
            model.get("schema_version") != _MODEL_SCHEMA
            or model.get("selector_identity") != RECOVERY_LOOKAHEAD_SELECTOR_ID
            or not isinstance(horizon, int)
            or model.get("resolve_without_information") != RESOLVE_WITHOUT_INFORMATION
            or model.get("resolve_after_refuted_feedback") != RESOLVE_AFTER_REFUTED_FEEDBACK
        ):
            raise ValueError("recorded recovery-lookahead selection identity differs")
        selected, rule, record = select_recovery_lookahead(slate, estimates, horizon=horizon)
        if model.get("fired_rule") != rule or model.get("base_choice") != record["base_choice"]:
            raise ValueError("recorded recovery-lookahead selection differs")
        return RuntimeJointResolvedChoice(
            selected, RECOVERY_LOOKAHEAD_SELECTOR_ID,
            dict(record.get("recovery_values") or {}), None, dict(model),
        )


__all__ = [
    "RECOVERY_LOOKAHEAD_SELECTOR_ID",
    "RESOLVE_AFTER_REFUTED_FEEDBACK",
    "RESOLVE_WITHOUT_INFORMATION",
    "RuntimeRecoveryLookaheadLiveResolver",
    "select_recovery_lookahead",
]
