"""Adaptive information gain with a repair reserve, on the same slate and estimates as TraceDP.

The strong no-lookahead comparator required by the evaluation protocol and by the binary
baseline goal: it sees exactly the Runtime-owned slate, the frozen atomic estimates and
the budget that the option DP sees, and differs only in how it chooses.

Rule, applied at every boundary (ties break on the public instance id):

1. a time-reserve submission, when the Runtime offers one, is taken (same as the fixed
   time-reserve selectors);
2. an affordable targeted repair, once evidence has unlocked it, is taken, preferring the
   higher estimated ``supported`` probability per cost unit;
3. otherwise the registered check with the largest outcome entropy per cost unit is run,
   but only if the budget left after it still pays the cheapest conditional repair the
   slate declares (the repair reserve), and only if its outcome is uncertain;
4. otherwise the Actor continues.

The outcome entropy of a check is the information its result carries about the public
predicate; this is the adaptive greedy of Golovin and Krause (2011), Algorithm 1, with the
check outcome as the observation, and the reserve of
``AdaptiveInformationGainRepairReservePolicy`` in ``canonical_verification_policies``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionCandidate,
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    RuntimeJointResolvedChoice,
)
from lh_platform.test_time_adaptation.runtime_owned_option_dp import (
    RuntimeAtomicActionEstimate,
    RuntimeOwnedOptionProjection,
)
from lh_platform.test_time_adaptation.runtime_owned_option_live import (
    RuntimeOwnedOptionLiveResolver,
)

INFORMATION_GAIN_SELECTOR_ID = "runtime_adaptive_information_gain_repair_reserve_v1"
_MODEL_SCHEMA = "runtime_information_gain_selection_v1"
_ENTROPY_FLOOR = 1e-12


def _entropy(estimate: RuntimeAtomicActionEstimate) -> float:
    return -math.fsum(
        item.probability * math.log(item.probability)
        for item in estimate.outcomes
        if item.probability > 0.0
    )


def _probability(estimate: RuntimeAtomicActionEstimate, outcome_id: str) -> float:
    return next(
        (item.probability for item in estimate.outcomes if item.outcome_id == outcome_id),
        0.0,
    )


def repair_reserve_units(slate: RuntimeJointActionSlate) -> int:
    """Cheapest conditional repair the slate declares; zero when no repair can follow."""

    costs = [
        option.cost_units
        for option in slate.conditional_future_options
        if option.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR
    ]
    return min(costs) if costs else 0


def select_information_gain_repair_reserve(
    slate: RuntimeJointActionSlate,
    estimates: tuple[RuntimeAtomicActionEstimate, ...],
) -> tuple[RuntimeJointActionCandidate, str, dict[str, dict[str, object]]]:
    """Return the selected candidate, the rule that fired, and every candidate's score."""

    by_id = {item.instance_action_id: item for item in estimates}
    budget = slate.budget_before
    reserve = repair_reserve_units(slate)
    scores: dict[str, dict[str, object]] = {}
    submissions, repairs, checks = [], [], []
    continued = None
    for candidate in slate.candidates:
        estimate = by_id.get(candidate.instance_action_id)
        if estimate is None:
            raise ValueError("information-gain selector lacks an estimate for a candidate")
        kind = candidate.action_type
        affordable = candidate.cost_units <= budget
        if kind is RuntimeJointActionType.CONTINUE_ACTOR:
            continued = candidate
            scores[candidate.instance_action_id] = {"rule": "continue"}
        elif kind is RuntimeJointActionType.SUBMIT_CURRENT_STATE:
            submissions.append(candidate)
            scores[candidate.instance_action_id] = {"rule": "submit"}
        elif kind is RuntimeJointActionType.RUN_TARGETED_REPAIR:
            value = _probability(estimate, "supported") / max(candidate.cost_units, 1)
            scores[candidate.instance_action_id] = {
                "rule": "repair",
                "supported_per_cost": value,
                "affordable": affordable,
            }
            if affordable:
                repairs.append((value, candidate))
        elif kind is RuntimeJointActionType.RUN_REGISTERED_CHECK:
            entropy = _entropy(estimate)
            reserved = budget - candidate.cost_units >= reserve
            value = entropy / max(candidate.cost_units, 1)
            scores[candidate.instance_action_id] = {
                "rule": "check",
                "entropy": entropy,
                "entropy_per_cost": value,
                "affordable": affordable,
                "keeps_repair_reserve": reserved,
            }
            if affordable and reserved and entropy > _ENTROPY_FLOOR:
                checks.append((value, candidate))
        else:
            raise ValueError("information-gain selector received an unmodeled action")
    if continued is None:
        raise ValueError("information-gain selector requires the continuation action")

    def best(pairs):
        # Largest score first; the public id breaks exact ties deterministically.
        return min(pairs, key=lambda pair: (-pair[0], pair[1].instance_action_id))[1]

    if submissions:
        return min(submissions, key=lambda c: c.instance_action_id), "time_reserve_submission", scores
    if repairs:
        return best(repairs), "unlocked_repair", scores
    if checks:
        return best(checks), "max_entropy_check_with_repair_reserve", scores
    return continued, "continue_actor", scores


class RuntimeInformationGainLiveResolver(RuntimeOwnedOptionLiveResolver):
    """Bind live state and frozen estimates exactly as TraceDP does, then choose greedily."""

    def _select(
        self,
        slate: RuntimeJointActionSlate,
        projection: RuntimeOwnedOptionProjection,
        estimates: tuple[RuntimeAtomicActionEstimate, ...],
        horizon: int,
    ) -> RuntimeJointResolvedChoice:
        del horizon
        selected, rule, scores = select_information_gain_repair_reserve(slate, estimates)
        return RuntimeJointResolvedChoice(
            selected,
            INFORMATION_GAIN_SELECTOR_ID,
            scores,
            None,
            {
                "schema_version": _MODEL_SCHEMA,
                "selector_identity": INFORMATION_GAIN_SELECTOR_ID,
                "fired_rule": rule,
                "repair_reserve_units": repair_reserve_units(slate),
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
        selected, rule, scores = select_information_gain_repair_reserve(slate, estimates)
        if (
            model.get("schema_version") != _MODEL_SCHEMA
            or model.get("selector_identity") != INFORMATION_GAIN_SELECTOR_ID
            or model.get("fired_rule") != rule
            or model.get("repair_reserve_units") != repair_reserve_units(slate)
        ):
            raise ValueError("recorded information-gain selection differs")
        return RuntimeJointResolvedChoice(
            selected, INFORMATION_GAIN_SELECTOR_ID, scores, None, dict(model)
        )


__all__ = [
    "INFORMATION_GAIN_SELECTOR_ID",
    "RuntimeInformationGainLiveResolver",
    "repair_reserve_units",
    "select_information_gain_repair_reserve",
]
