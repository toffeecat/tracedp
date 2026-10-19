"""TraceDP v3o: residual obligation state and state-dependent recovery value.

The v2 runtime takes any unlocked repair before it compares values, and plans with depth
1 at the replan that follows a refuted check, so feedback and repair were never weighed.
This selector replaces only the choice. Each registered public check is an obligation
with a state compiled from the public projection (supported, refuted for the first time,
refuted again after a remedy, unknown), and handing back to the Actor is worth the
product of the obligations' recovery probabilities (the obligations are an AND):

    V_A(z) = prod_k q(z_k),  q(S) = 1, q(R0) = rho_fb, q(R1) = rho_again, q(P) = rho_rep,
                             q(U) = p_k + (1 - p_k) * rho_blind

A memoized Bellman backup of depth 3 compares checks, repairs and the hand-back at every
boundary, replans included. The rates are the frozen obligation-level counts of revision
14 (``scripts/obligation_rate_census.py``); a repair is priced at its measured cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache

from lh_platform.test_time_adaptation.runtime_information_gain_selector import (
    RuntimeInformationGainLiveResolver,
    _probability,
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
from lh_platform.test_time_adaptation.runtime_recovery_lookahead_selector import (
    RESOLVE_WITHOUT_INFORMATION,
)
from lh_platform.tracefix_bench.supervisor_dp import LAMBDA_PER_INPUT_TOKEN, LAMBDA_PER_SECOND

OBLIGATION_VALUE_SELECTOR_ID = "tracedp_obligation_value_v1"
_MODEL_SCHEMA = "runtime_obligation_value_selection_v1"
HORIZON = 3
# Reversible obligations of the 195 valid managed runs, Beta(1, 1).
RHO_FEEDBACK = (151 + 1) / (155 + 2)       # refuted, fed back, never repaired
RHO_REPAIR = (51 + 1) / (53 + 2)           # refuted, then a targeted repair ran
RHO_FEEDBACK_AGAIN = (7 + 1) / (9 + 2)     # refuted again at a newer version, fed back
RHO_BLIND = RESOLVE_WITHOUT_INFORMATION    # no new information
REPAIR_INPUT_TOKENS = 7943                 # mean of 63 measured repairs, uncached input
REPAIR_SECONDS = 226                       # mean of 73 measured repairs
REPAIR_PRICE = LAMBDA_PER_INPUT_TOKEN * REPAIR_INPUT_TOKENS + LAMBDA_PER_SECOND * REPAIR_SECONDS
RATE_PROVENANCE = (
    "reversible obligations, "
    "feedback 151/155, repair 51/53, refuted again then feedback 7/9, Beta(1,1)"
)
_TIE = 1e-12
_UNKNOWN = ("U", "Ui")  # unknown or stale; ``i``: the predicate already has an incident
_REFUTED = ("R0", "R1")


@dataclass(frozen=True)
class Obligation:
    """One public obligation as the selector sees it at this boundary."""

    task_id: str
    proposition: str
    status: str                 # S, R0, R1, U, Ui
    supported: float            # p_k: frozen estimate, unknown outcome removed
    refuted: float              # raw outcome probabilities of the check, if offered
    unknown: float
    check_id: str | None        # candidate instance id of its check at this boundary
    check_cost: int
    repair_id: str | None       # candidate id, or conditional option id after the check
    repair_cost: int
    repair_after_check: bool    # the repair needs this boundary's check to refute first

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _key(candidate) -> tuple[str, str] | None:
    task = candidate.target_binding.task_id
    proposition = candidate.execution_contract.target_predicate
    return (task, proposition) if task and proposition else None


def compile_obligations(
    slate: RuntimeJointActionSlate,
    projection: RuntimeOwnedOptionProjection,
    estimates: tuple[RuntimeAtomicActionEstimate, ...],
    incidents: tuple[Mapping[str, object], ...],
) -> tuple[Obligation, ...]:
    """Obligations whose value an action at this boundary can change, in projection order."""

    by_id = {item.instance_action_id: item for item in estimates}
    checks = {
        _key(c): c for c in slate.candidates if c.action_type is RuntimeJointActionType.RUN_REGISTERED_CHECK
    }
    repairs = {
        _key(c): c for c in slate.candidates if c.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR
    }
    check_keys = {c.instance_action_id: key for key, c in checks.items()}
    options: dict[tuple[str, str], object] = {}
    for option in slate.conditional_future_options:
        key = check_keys.get(option.prerequisite_action_id)
        if (
            option.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR
            and key is not None
            and "refuted" in option.prerequisite_outcomes
            and (key not in options or option.cost_units < options[key].cost_units)
        ):
            options[key] = option
    again = {
        (str(item.get("task_id")), str(item.get("proposition")))
        for item in incidents
        if item.get("opening_evidence_id") != item.get("latest_evidence_id")
    }
    obligations = []
    for predicate in projection.acceptance_predicates:
        key = (predicate.task_id, predicate.proposition)
        check, repair = checks.get(key), repairs.get(key)
        option = options.get(key)
        has_incident = predicate.incident_id is not None
        if predicate.evidence_state == "supported":
            status = "S"
        elif predicate.evidence_state == "refuted":
            status = "R1" if key in again else "R0"
        else:
            status = "Ui" if has_incident else "U"
        in_play = check is not None or (repair is not None and status in _REFUTED)
        if not in_play:
            continue  # nothing at this boundary changes it: a common factor of every action
        s = _probability(by_id[check.instance_action_id], "supported") if check else 0.0
        r = _probability(by_id[check.instance_action_id], "refuted") if check else 0.0
        u = _probability(by_id[check.instance_action_id], "unknown") if check else 0.0
        if repair is not None:
            repair_id, repair_cost, after = repair.instance_action_id, repair.cost_units, False
        elif option is not None:
            repair_id, repair_cost, after = option.option_id, option.cost_units, True
        else:
            repair_id, repair_cost, after = None, 0, False
        obligations.append(Obligation(
            task_id=predicate.task_id, proposition=predicate.proposition, status=status,
            supported=(s / (s + r)) if s + r > 0 else 0.5, refuted=r, unknown=u,
            check_id=check.instance_action_id if check else None,
            check_cost=check.cost_units if check else 0,
            repair_id=repair_id, repair_cost=repair_cost, repair_after_check=after,
        ))
    return tuple(obligations)


def _q(status: str, supported: float) -> float:
    if status == "S":
        return 1.0
    if status == "R0":
        return RHO_FEEDBACK
    if status == "R1":
        return RHO_FEEDBACK_AGAIN
    if status == "P":
        return RHO_REPAIR
    return supported + (1.0 - supported) * RHO_BLIND


def plan_values(
    obligations: tuple[Obligation, ...], budget: int, horizon: int = HORIZON
) -> dict[str, float]:
    """Value of continuing, and of each check and repair this boundary offers."""

    n = len(obligations)

    def hand_back(states: tuple[str, ...]) -> float:
        value = 1.0
        for item, status in zip(obligations, states):
            value *= _q(status.rstrip("c"), item.supported)
        return value

    @lru_cache(maxsize=None)
    def value(states: tuple[str, ...], repairable: frozenset[int], left: int, depth: int) -> float:
        best = hand_back(states)
        if depth <= 0:
            return best
        for k in range(n):
            q = check_value(states, repairable, left, depth, k)
            if q is not None:
                best = max(best, q)
            q = repair_value(states, repairable, left, depth, k)
            if q is not None:
                best = max(best, q)
        return best

    def check_value(states, repairable, left, depth, k):
        item = obligations[k]
        if item.check_id is None or states[k] not in _UNKNOWN or item.check_cost > left:
            return None
        refuted_status = "R1" if states[k] == "Ui" else "R0"
        opened = repairable | {k} if item.repair_id is not None and item.repair_after_check else repairable
        after = left - item.check_cost
        return (
            (1.0 - item.refuted - item.unknown) * value(_set(states, k, "S"), repairable, after, depth - 1)
            + item.refuted * value(_set(states, k, refuted_status), opened, after, depth - 1)
            + item.unknown * value(_set(states, k, states[k] + "c"), repairable, after, depth - 1)
        )

    def repair_value(states, repairable, left, depth, k):
        item = obligations[k]
        if k not in repairable or states[k] not in _REFUTED or item.repair_cost > left:
            return None
        return -REPAIR_PRICE + value(_set(states, k, "P"), repairable - {k}, left - item.repair_cost, depth - 1)

    start = tuple(item.status for item in obligations)
    offered = frozenset(k for k, item in enumerate(obligations) if item.repair_id is not None and not item.repair_after_check)
    values = {"continue_actor": hand_back(start)}
    for k, item in enumerate(obligations):
        q = check_value(start, offered, budget, horizon, k)
        if q is not None:
            values[item.check_id] = q
        q = repair_value(start, offered, budget, horizon, k)
        if q is not None:
            values[item.repair_id] = q
    return values


def _set(states: tuple[str, ...], k: int, status: str) -> tuple[str, ...]:
    return states[:k] + (status,) + states[k + 1:]


def select_obligation_value(
    slate: RuntimeJointActionSlate,
    estimates: tuple[RuntimeAtomicActionEstimate, ...],
    obligations: tuple[Obligation, ...],
) -> tuple[RuntimeJointActionCandidate, str, dict[str, float], str]:
    """Return the choice, the fired rule, every modeled value and the base's choice."""

    base, base_rule, _scores = select_information_gain_repair_reserve(slate, estimates)
    if base_rule == "time_reserve_submission":
        return base, base_rule, {}, base.instance_action_id
    values = plan_values(obligations, slate.budget_before)
    # One check per boundary, and the Actor may settle before a later one: among plans of
    # equal depth-H value (the product commutes), the larger immediate value goes first.
    immediate = plan_values(obligations, slate.budget_before, horizon=1)
    kind_rank = {
        RuntimeJointActionType.CONTINUE_ACTOR: 0,
        RuntimeJointActionType.RUN_REGISTERED_CHECK: 1,
        RuntimeJointActionType.RUN_TARGETED_REPAIR: 2,
    }
    modeled = [c for c in slate.candidates if c.instance_action_id in values]
    top = max(values[c.instance_action_id] for c in modeled)
    tied = [c for c in modeled if values[c.instance_action_id] >= top - _TIE]
    first = max(immediate[c.instance_action_id] for c in tied)
    chosen = min(
        (c for c in tied if immediate[c.instance_action_id] >= first - _TIE),
        key=lambda c: (kind_rank[c.action_type], c.instance_action_id),
    )
    return chosen, "obligation_value", values, base.instance_action_id


class RuntimeObligationValueLiveResolver(RuntimeInformationGainLiveResolver):
    """Same live binding and estimates as v2; the choice is the obligation-value backup."""

    def _select(
        self,
        slate: RuntimeJointActionSlate,
        projection: RuntimeOwnedOptionProjection,
        estimates: tuple[RuntimeAtomicActionEstimate, ...],
        horizon: int,
    ) -> RuntimeJointResolvedChoice:
        if self.public_projection is None:
            raise RuntimeError("obligation-value resolver lacks Kernel state")
        raw = self.public_projection().get("predicate_incidents") or []
        incidents = tuple(item for item in raw if isinstance(item, Mapping))
        obligations = compile_obligations(slate, projection, estimates, incidents)
        selected, rule, values, base_choice = select_obligation_value(slate, estimates, obligations)
        return RuntimeJointResolvedChoice(
            selected,
            OBLIGATION_VALUE_SELECTOR_ID,
            dict(values),
            None,
            {
                "schema_version": _MODEL_SCHEMA,
                "selector_identity": OBLIGATION_VALUE_SELECTOR_ID,
                "fired_rule": rule,
                "selected": selected.instance_action_id,
                "horizon": HORIZON,
                "native_horizon": horizon,
                "rates": _rates(),
                "rate_provenance": RATE_PROVENANCE,
                "base_choice": base_choice,
                "obligations": [item.to_dict() for item in obligations],
                # Checks or repairs that no obligation claimed: audit for a broken binding.
                "unmodeled_candidates": [
                    c.instance_action_id for c in slate.candidates
                    if c.instance_action_id not in values
                    and c.action_type is not RuntimeJointActionType.SUBMIT_CURRENT_STATE
                ],
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
        recorded = model.get("obligations")
        if (
            model.get("schema_version") != _MODEL_SCHEMA
            or model.get("selector_identity") != OBLIGATION_VALUE_SELECTOR_ID
            or model.get("horizon") != HORIZON
            or model.get("rates") != _rates()
            or not isinstance(recorded, list)
        ):
            raise ValueError("recorded obligation-value selection identity differs")
        obligations = tuple(Obligation(**dict(item)) for item in recorded)
        selected, rule, values, base_choice = select_obligation_value(slate, estimates, obligations)
        if (
            model.get("fired_rule") != rule
            or model.get("selected") != selected.instance_action_id
            or model.get("base_choice") != base_choice
        ):
            raise ValueError("recorded obligation-value selection differs")
        return RuntimeJointResolvedChoice(
            selected, OBLIGATION_VALUE_SELECTOR_ID, dict(values), None, dict(model)
        )


def _rates() -> dict[str, float]:
    return {
        "rho_feedback": RHO_FEEDBACK,
        "rho_repair": RHO_REPAIR,
        "rho_feedback_again": RHO_FEEDBACK_AGAIN,
        "rho_blind": RHO_BLIND,
        "repair_price": REPAIR_PRICE,
    }


__all__ = [
    "HORIZON",
    "OBLIGATION_VALUE_SELECTOR_ID",
    "Obligation",
    "RHO_FEEDBACK",
    "RHO_FEEDBACK_AGAIN",
    "RHO_REPAIR",
    "RuntimeObligationValueLiveResolver",
    "compile_obligations",
    "plan_values",
    "select_obligation_value",
]
