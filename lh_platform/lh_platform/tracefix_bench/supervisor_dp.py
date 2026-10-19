"""TraceDP v3: supervisor DP over recovery meta-actions.

Before the Actor resumes, the supervisor holds a belief over the failure type
(irreversible, env_dependency, reversible), gathers zero-model-call evidence, and may
roll the environment back or write a DoVer intervention. Every choice is the argmax of a
finite-horizon memoized Bellman backup

    V(z) = max( P(resolve | z),  max_u [ -lambda * c(u) + E_y V(z') ] ),

so a probe is worth running only through how its outcome changes the later choice
(value of information), and an expensive component is used only where its recovery gain
exceeds its priced cost. All probabilities are the frozen calibration counts and its
supplement (development and calibration groups, Beta/Dirichlet(1) smoothing).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil, log2

SUPERVISOR_DP_ID = "tracedp_supervisor_dp_v1"
TYPES = ("irreversible", "env_dependency", "reversible")

# Dirichlet(1,1,1) over the type counts of the 6 development and calibration cells (1, 1, 4).
PRIOR = {"irreversible": 2 / 9, "env_dependency": 2 / 9, "reversible": 5 / 9}
# P(resolve | type, hand-off configuration), Beta(1,1) over the calibration counts.
RESOLVE = {
    ("irreversible", "no_rollback"): 1 / 17,
    ("irreversible", "rollback"): 3 / 5,
    ("irreversible", "rollback_intervention"): 4 / 5,
    ("env_dependency", "undelivered"): 1 / 20,
    ("env_dependency", "delivered"): 20 / 33,
    ("reversible", "v2_runtime"): 7 / 8,
}
REPLAY_ACCEPTED = 6 / 7       # rollback snapshots accepted today: 5 of 5
INTERVENTION_VALID = 10 / 11  # DoVer interventions with a valid replacement: 9 of 9
# Price of resources in resolve-probability units. Tokens: 0.01 per 100k input tokens
# Wall time: 0.01 per hour, the same weight as 100k tokens (an
# implementation constant: replay time is priced but gives no rate).
LAMBDA_PER_INPUT_TOKEN = 0.01 / 100_000
LAMBDA_PER_SECOND = 0.01 / 3600
COST = {
    "probe_inventory": {"seconds": 20.0, "input_tokens": 0},
    "probe_clean_context": {"seconds": 30.0, "input_tokens": 0},
    "intervene": {"seconds": 60.0, "input_tokens": 25_000},
}
REPLAY_SECONDS = 150.0


@dataclass(frozen=True)
class PlanState:
    inventory: str = "unknown"      # unknown | debt | clear
    clean: str = "unknown"          # unknown | fail | pass | na
    rollback: str = "none"          # none | done | failed
    intervention: str = "none"      # none | done | invalid


@dataclass(frozen=True)
class PlanContext:
    prefix_calls: int               # tool calls in the failed trace (bisection range)
    script_required: bool           # the instruction asks for an executable .py artifact


def belief(state: PlanState, context: PlanContext) -> dict[str, float]:
    """Posterior over failure types given the evidence gathered so far."""

    weights = dict(PRIOR)
    if not context.script_required or state.clean == "na":
        weights["env_dependency"] = 0.0
    if state.inventory == "debt":
        return {"irreversible": 1.0, "env_dependency": 0.0, "reversible": 0.0}
    if state.inventory == "clear":
        weights["irreversible"] = 0.0
    if state.clean == "fail":
        return {"irreversible": 0.0, "env_dependency": 1.0, "reversible": 0.0}
    if state.clean == "pass":
        weights["env_dependency"] = 0.0
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def resume_value(state: PlanState, context: PlanContext) -> float:
    """Expected recovery if the Actor resumes now with this hand-off."""

    b = belief(state, context)
    if state.rollback == "done":
        irreversible = RESOLVE[("irreversible", "rollback_intervention" if state.intervention == "done" else "rollback")]
    else:
        irreversible = RESOLVE[("irreversible", "no_rollback")]
    env = RESOLVE[("env_dependency", "delivered" if state.clean == "fail" else "undelivered")]
    return (
        b["irreversible"] * irreversible
        + b["env_dependency"] * env
        + b["reversible"] * RESOLVE[("reversible", "v2_runtime")]
    )


def action_cost(action: str, context: PlanContext) -> float:
    if action == "bisect_rollback":
        replays = max(1, ceil(log2(context.prefix_calls + 1)))
        return LAMBDA_PER_SECOND * REPLAY_SECONDS * replays
    cost = COST[action]
    return LAMBDA_PER_SECOND * cost["seconds"] + LAMBDA_PER_INPUT_TOKEN * cost["input_tokens"]


def outcomes(action: str, state: PlanState, context: PlanContext) -> list[tuple[float, PlanState]]:
    """The outcome distribution of a meta-action under the current belief."""

    b = belief(state, context)
    if action == "probe_inventory":
        return [(b["irreversible"], replace(state, inventory="debt")),
                (1 - b["irreversible"], replace(state, inventory="clear"))]
    if action == "probe_clean_context":
        return [(b["env_dependency"], replace(state, clean="fail")),
                (1 - b["env_dependency"], replace(state, clean="pass"))]
    if action == "bisect_rollback":
        return [(REPLAY_ACCEPTED, replace(state, rollback="done")),
                (1 - REPLAY_ACCEPTED, replace(state, rollback="failed"))]
    if action == "intervene":
        return [(INTERVENTION_VALID, replace(state, intervention="done")),
                (1 - INTERVENTION_VALID, replace(state, intervention="invalid"))]
    raise ValueError(f"unknown meta-action {action}")


def available(state: PlanState, context: PlanContext) -> list[str]:
    actions = []
    if state.inventory == "unknown":
        actions.append("probe_inventory")
    if state.clean == "unknown" and context.script_required:
        actions.append("probe_clean_context")
    if state.inventory == "debt" and state.rollback == "none":
        actions.append("bisect_rollback")
    if state.rollback == "done" and state.intervention == "none":
        actions.append("intervene")
    return actions


def plan_values(state: PlanState, context: PlanContext) -> dict[str, float]:
    """Q value of every available meta-action, and of resuming now."""

    @lru_cache(maxsize=None)
    def value(s: PlanState) -> float:
        best = resume_value(s, context)
        for action in available(s, context):
            best = max(best, q(s, action))
        return best

    def q(s: PlanState, action: str) -> float:
        return -action_cost(action, context) + sum(p * value(nxt) for p, nxt in outcomes(action, s, context) if p > 0)

    values = {"resume": resume_value(state, context)}
    for action in available(state, context):
        values[action] = q(state, action)
    return values


def best_action(state: PlanState, context: PlanContext) -> tuple[str, dict[str, float]]:
    """The argmax meta-action; ties keep the cheaper choice (resume first, then listed order)."""

    values = plan_values(state, context)
    order = ["resume", *available(state, context)]
    choice = max(order, key=lambda name: (values[name], -order.index(name)))
    return choice, values


# --- Closed loop -------------------------------------------
# The frozen counts behind RESOLVE, as Beta(s+1, f+1). One in-case hand-off of a
# configuration that ends with evidence of the same failure adds one failure: the
# posterior mean becomes (s+1)/(s+f+3).
COUNTS = {
    ("irreversible", "no_rollback"): (0, 15),
    ("irreversible", "rollback"): (2, 1),
    ("irreversible", "rollback_intervention"): (3, 0),
    ("env_dependency", "undelivered"): (0, 18),
    ("env_dependency", "delivered"): (19, 12),
    ("reversible", "v2_runtime"): (6, 0),
}
RESTORE_SECONDS = 30.0
REENTRY_PROBE_SECONDS = 60.0
MAX_REENTRIES = 2


def posterior_resolve(key: tuple[str, str], in_case_failures: int) -> float:
    successes, failures = COUNTS[key]
    return (successes + 1) / (successes + failures + 2 + in_case_failures)


@dataclass(frozen=True)
class ReentryEvidence:
    missing_inputs: tuple[str, ...]      # inputs lost since the start, at this settlement
    clean_context_failing: bool          # a required script still fails in a clean context
    rolled_back: bool                    # the plan stage rolled back
    intervention_delivered: bool         # the plan stage delivered a DoVer intervention
    env_evidence_delivered: int          # times env evidence has been delivered so far
    restores: int                        # restores already made by earlier re-entries


def reentry_values(evidence: ReentryEvidence) -> dict[str, float]:
    """Values of the closed-loop actions at one natural settlement.

    The evidence at a settlement is direct (a missing input, a failing clean run), so
    the failure type is observed rather than believed; the uncertainty left is whether
    the next hand-off resolves it, which the in-case posterior prices.
    """

    values = {"defer": 0.0}
    if evidence.missing_inputs:
        config = "rollback_intervention" if evidence.intervention_delivered else "rollback"
        handed_off = int(evidence.rolled_back) + evidence.restores
        values["defer"] = posterior_resolve(("irreversible", "no_rollback"), 0)
        values["restore_inputs"] = (
            posterior_resolve(("irreversible", config), handed_off)
            - LAMBDA_PER_SECOND * RESTORE_SECONDS
        )
    elif evidence.clean_context_failing:
        values["defer"] = posterior_resolve(("env_dependency", "undelivered"), 0)
        values["redeliver_env_evidence"] = posterior_resolve(
            ("env_dependency", "delivered"), evidence.env_evidence_delivered
        )
    else:
        # Nothing the supervisor can see is wrong: the v2 runtime decides as usual.
        values["defer"] = posterior_resolve(("reversible", "v2_runtime"), 0)
    return values


def reentry_decision(evidence: ReentryEvidence) -> tuple[str, dict[str, float]]:
    values = reentry_values(evidence)
    choice = max(values, key=lambda name: (values[name], name == "defer"))
    return choice, values


__all__ = [
    "MAX_REENTRIES", "PlanContext", "PlanState", "ReentryEvidence", "SUPERVISOR_DP_ID", "belief",
    "best_action", "plan_values", "posterior_resolve", "reentry_decision", "resume_value",
]
