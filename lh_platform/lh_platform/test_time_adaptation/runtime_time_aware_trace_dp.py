"""Plan bounded runtime actions with explicit wall-clock and Actor-step costs.

This locally designed planner keeps the legacy runtime planner unchanged. It
uses the same finite-horizon Bellman structure while making the v4 execution
contract and cost vector explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_trace_dp import (
    RuntimeJointTransitionOutcome,
)


@dataclass(frozen=True)
class RuntimeTimedPlanningState:
    """Represent one forecast boundary with auditable residual and resource state."""

    state_id: str
    public_goal_complete: bool
    unblocked_fraction: float
    remaining_intervention_budget: int
    remaining_actor_steps: int
    remaining_time_ms: int
    residual_predicate_states: tuple[tuple[str, str], ...]
    outcome_phase: str

    def __post_init__(self) -> None:
        predicate_ids = [predicate_id for predicate_id, _ in self.residual_predicate_states]
        if (
            not self.state_id
            or not 0.0 <= self.unblocked_fraction <= 1.0
            or self.remaining_intervention_budget < 0
            or self.remaining_actor_steps < 0
            or self.remaining_time_ms < 0
            or not self.outcome_phase
            or len(predicate_ids) != len(set(predicate_ids))
            or any(not predicate_id or not state for predicate_id, state in self.residual_predicate_states)
        ):
            raise ValueError("timed runtime planning state is invalid")


@dataclass(frozen=True)
class RuntimeTimedActionTransition:
    """Bind one runtime action to outcomes, duration, and intervention costs."""

    state_id: str
    instance_action_id: str
    action_type: RuntimeJointActionType
    intervention_cost_units: int
    expected_false_interventions: float
    duration_ms: int
    prediction_evidence_id: str
    outcomes: tuple[RuntimeJointTransitionOutcome, ...]

    def __post_init__(self) -> None:
        if (
            not self.state_id
            or not self.instance_action_id
            or not isinstance(self.action_type, RuntimeJointActionType)
            or self.intervention_cost_units < 0
            or not 0.0 <= self.expected_false_interventions <= 1.0
            or self.duration_ms < 1
            or not self.prediction_evidence_id
            or not self.outcomes
            or not math.isclose(
                sum(item.probability for item in self.outcomes),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("timed runtime action transition is invalid")
        if self.action_type is RuntimeJointActionType.CONTINUE_ACTOR:
            if self.intervention_cost_units != 0:
                raise ValueError("continue_actor cannot consume intervention budget")
        elif self.intervention_cost_units < 1:
            raise ValueError("runtime interventions require a positive cost")


@dataclass(frozen=True)
class RuntimeTimeAwarePolicyValue:
    """Keep task utility and each expected cost separately auditable."""

    # Objective keys are expectations of probabilities, fractions, and measured
    # millisecond costs.  Comparing them at full IEEE-754 precision makes the
    # ordering depend on representation noise: two hand-written probability
    # triples that each denote 1.0 can sum to 1.0 and to 0.9999999999999999, and
    # that one-ULP gap then decides the primary objective over a real difference
    # in the next one.  Keys are therefore compared on a fixed decimal grid that
    # is far below any meaningful difference in these quantities.
    OBJECTIVE_PRECISION_DECIMALS = 9

    public_success_probability: float
    expected_unblocked_fraction: float
    expected_duration_ms: float
    expected_false_interventions: float
    expected_intervention_cost: float

    @property
    def priority(self) -> tuple[float, float, float, float, float]:
        """Prefer task completion, then progress, then lower measured costs.

        Every key is rounded onto the shared grid before comparison.  Ties fall
        through to the next key and, when all keys tie, to the caller's stable
        candidate order, so replay stays deterministic.
        """

        digits = self.OBJECTIVE_PRECISION_DECIMALS
        return (
            round(self.public_success_probability, digits),
            round(self.expected_unblocked_fraction, digits),
            -round(self.expected_duration_ms, digits),
            -round(self.expected_false_interventions, digits),
            -round(self.expected_intervention_cost, digits),
        )


@dataclass(frozen=True)
class RuntimeTimeAwarePlanDecision:
    instance_action_id: str | None
    value: RuntimeTimeAwarePolicyValue
    q_values: tuple[tuple[str, RuntimeTimeAwarePolicyValue], ...]


class RuntimeTimeAwareTraceDPPlanner:
    """Run exact finite-horizon recursion over explicit timed transitions."""

    def __init__(
        self,
        *,
        states: tuple[RuntimeTimedPlanningState, ...],
        transitions: tuple[RuntimeTimedActionTransition, ...],
        horizon: int,
    ) -> None:
        if horizon < 1:
            raise ValueError("time-aware TraceDP horizon must be positive")
        self.states = {item.state_id: item for item in states}
        if len(self.states) != len(states):
            raise ValueError("time-aware TraceDP states must be unique")
        self.transitions: dict[str, list[RuntimeTimedActionTransition]] = {}
        seen: set[tuple[str, str]] = set()
        for transition in transitions:
            key = (transition.state_id, transition.instance_action_id)
            source = self.states.get(transition.state_id)
            if key in seen or source is None:
                raise ValueError("time-aware TraceDP transition identity is invalid")
            targets = [self.states.get(item.next_state_id) for item in transition.outcomes]
            if any(target is None for target in targets):
                raise ValueError("time-aware TraceDP transition targets unknown state")
            actor_step_cost = int(
                transition.action_type is not RuntimeJointActionType.SUBMIT_CURRENT_STATE
            )
            feasible_at_source = (
                transition.intervention_cost_units
                <= source.remaining_intervention_budget
                and transition.duration_ms <= source.remaining_time_ms
                and source.remaining_actor_steps >= actor_step_cost
            )
            if feasible_at_source:
                for target in targets:
                    assert target is not None
                    if (
                        target.remaining_intervention_budget
                        > source.remaining_intervention_budget
                        - transition.intervention_cost_units
                        or target.remaining_actor_steps
                        > source.remaining_actor_steps - actor_step_cost
                        or target.remaining_time_ms
                        > source.remaining_time_ms - transition.duration_ms
                    ):
                        raise ValueError("time-aware TraceDP transition creates resources")
            seen.add(key)
            self.transitions.setdefault(transition.state_id, []).append(transition)
        self.horizon = horizon
        self._cache: dict[tuple[str, int], RuntimeTimeAwarePolicyValue] = {}

    @staticmethod
    def _stop_value(state: RuntimeTimedPlanningState) -> RuntimeTimeAwarePolicyValue:
        """Value settlement without inventing future execution or cost."""

        return RuntimeTimeAwarePolicyValue(
            public_success_probability=1.0 if state.public_goal_complete else 0.0,
            expected_unblocked_fraction=state.unblocked_fraction,
            expected_duration_ms=0.0,
            expected_false_interventions=0.0,
            expected_intervention_cost=0.0,
        )

    def _legal(
        self, state: RuntimeTimedPlanningState
    ) -> tuple[RuntimeTimedActionTransition, ...]:
        """Require budget, clock, and Actor continuation for every live action."""

        return tuple(
            item
            for item in self.transitions.get(state.state_id, ())
            if item.intervention_cost_units <= state.remaining_intervention_budget
            and item.duration_ms <= state.remaining_time_ms
            and (
                item.action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE
                or state.remaining_actor_steps > 0
            )
        )

    def _action_value(
        self, transition: RuntimeTimedActionTransition, depth: int
    ) -> RuntimeTimeAwarePolicyValue:
        """Accumulate current cost and branch-weighted future values."""

        branches = [
            (outcome.probability, self._best(outcome.next_state_id, depth - 1))
            for outcome in transition.outcomes
        ]
        return RuntimeTimeAwarePolicyValue(
            public_success_probability=sum(
                probability * value.public_success_probability
                for probability, value in branches
            ),
            expected_unblocked_fraction=sum(
                probability * value.expected_unblocked_fraction
                for probability, value in branches
            ),
            expected_duration_ms=(
                transition.duration_ms
                + sum(
                    probability * value.expected_duration_ms
                    for probability, value in branches
                )
            ),
            expected_false_interventions=(
                transition.expected_false_interventions
                + sum(
                    probability * value.expected_false_interventions
                    for probability, value in branches
                )
            ),
            expected_intervention_cost=(
                transition.intervention_cost_units
                + sum(
                    probability * value.expected_intervention_cost
                    for probability, value in branches
                )
            ),
        )

    def _best(self, state_id: str, depth: int) -> RuntimeTimeAwarePolicyValue:
        """Cache the best continuation value for one state and remaining horizon."""

        key = (state_id, depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        state = self.states[state_id]
        best = self._stop_value(state)
        if depth > 0 and not state.public_goal_complete:
            legal = self._legal(state)
            if legal:
                # A live runtime has no implicit pause action. At an internal
                # decision epoch it must choose among the executable slate.
                best = self._action_value(legal[0], depth)
                for transition in legal[1:]:
                    value = self._action_value(transition, depth)
                    if value.priority > best.priority:
                        best = value
        self._cache[key] = best
        return best

    def plan(self, state_id: str) -> RuntimeTimeAwarePlanDecision:
        """Return every legal first-step value for independent replay."""

        if state_id not in self.states:
            raise ValueError("time-aware TraceDP start state is unknown")
        state = self.states[state_id]
        q_values = tuple(
            (item.instance_action_id, self._action_value(item, self.horizon))
            for item in self._legal(state)
        )
        best_id: str | None = None
        best_value = self._stop_value(state)
        if q_values:
            best_id, best_value = q_values[0]
            for action_id, value in q_values[1:]:
                if value.priority > best_value.priority:
                    best_id = action_id
                    best_value = value
        return RuntimeTimeAwarePlanDecision(best_id, best_value, q_values)


__all__ = [
    "RuntimeTimedActionTransition",
    "RuntimeTimedPlanningState",
    "RuntimeTimeAwarePlanDecision",
    "RuntimeTimeAwarePolicyValue",
    "RuntimeTimeAwareTraceDPPlanner",
]
