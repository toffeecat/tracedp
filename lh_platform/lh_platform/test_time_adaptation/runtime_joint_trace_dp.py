"""Finite-horizon planning over explicit heterogeneous runtime transitions.

The Bellman recursion is generic. Transition probabilities and next states are
inputs with provenance; this module does not infer them from action names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionType,
)


@dataclass(frozen=True)
class RuntimeJointPlanningState:
    """Represent the residual task facts needed to distinguish runtime outcomes."""

    state_id: str
    public_goal_complete: bool
    unblocked_fraction: float
    remaining_intervention_budget: int
    remaining_actor_steps: int
    output_phase: str
    missing_resource_types: tuple[str, ...]
    repeat_disappearance_count: int
    last_intervention_effect: str

    def __post_init__(self) -> None:
        if (
            not self.state_id
            or not 0.0 <= self.unblocked_fraction <= 1.0
            or self.remaining_intervention_budget < 0
            or self.remaining_actor_steps < 0
            or not self.output_phase
            or self.repeat_disappearance_count < 0
            or not self.last_intervention_effect
        ):
            raise ValueError("runtime joint planning state is invalid")


@dataclass(frozen=True)
class RuntimeJointTransitionOutcome:
    probability: float
    next_state_id: str
    outcome_id: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
            or not self.next_state_id
            or not self.outcome_id
        ):
            raise ValueError("runtime joint transition outcome is invalid")


@dataclass(frozen=True)
class RuntimeJointActionTransition:
    """Bind one action instance to explicit outcomes and accounting costs."""

    state_id: str
    instance_action_id: str
    action_type: RuntimeJointActionType
    intervention_cost_units: int
    expected_false_interventions: float
    prediction_evidence_id: str
    outcomes: tuple[RuntimeJointTransitionOutcome, ...]

    def __post_init__(self) -> None:
        if (
            not self.state_id
            or not self.instance_action_id
            or not isinstance(self.action_type, RuntimeJointActionType)
            or self.intervention_cost_units < 0
            or not 0.0 <= self.expected_false_interventions <= 1.0
            or not self.prediction_evidence_id
            or not self.outcomes
            or not math.isclose(
                sum(item.probability for item in self.outcomes),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("runtime joint action transition is invalid")
        if self.action_type is RuntimeJointActionType.CONTINUE_ACTOR:
            if self.intervention_cost_units != 0:
                raise ValueError("continue_actor cannot consume intervention budget")
        elif self.intervention_cost_units < 1:
            raise ValueError("runtime interventions require a positive cost")


@dataclass(frozen=True)
class RuntimeJointPolicyValue:
    public_success_probability: float
    expected_unblocked_fraction: float
    expected_false_interventions: float
    expected_intervention_cost: float

    @property
    def priority(self) -> tuple[float, float, float, float]:
        return (
            self.public_success_probability,
            self.expected_unblocked_fraction,
            -self.expected_false_interventions,
            -self.expected_intervention_cost,
        )


@dataclass(frozen=True)
class RuntimeJointPlanDecision:
    instance_action_id: str | None
    value: RuntimeJointPolicyValue
    q_values: tuple[tuple[str, RuntimeJointPolicyValue], ...]


class RuntimeJointTraceDPPlanner:
    """Run exact Bellman recursion over a finite, independently auditable tree."""

    def __init__(
        self,
        *,
        states: tuple[RuntimeJointPlanningState, ...],
        transitions: tuple[RuntimeJointActionTransition, ...],
        horizon: int,
    ) -> None:
        if horizon < 1:
            raise ValueError("runtime joint TraceDP horizon must be positive")
        self.states = {item.state_id: item for item in states}
        if len(self.states) != len(states):
            raise ValueError("runtime joint TraceDP states must be unique")
        self.transitions: dict[str, list[RuntimeJointActionTransition]] = {}
        seen: set[tuple[str, str]] = set()
        for transition in transitions:
            key = (transition.state_id, transition.instance_action_id)
            if key in seen or transition.state_id not in self.states:
                raise ValueError("runtime joint TraceDP transition identity is invalid")
            if any(item.next_state_id not in self.states for item in transition.outcomes):
                raise ValueError("runtime joint TraceDP transition targets unknown state")
            seen.add(key)
            self.transitions.setdefault(transition.state_id, []).append(transition)
        self.horizon = horizon
        self._cache: dict[tuple[str, int], RuntimeJointPolicyValue] = {}

    @staticmethod
    def _stop_value(state: RuntimeJointPlanningState) -> RuntimeJointPolicyValue:
        return RuntimeJointPolicyValue(
            public_success_probability=1.0 if state.public_goal_complete else 0.0,
            expected_unblocked_fraction=state.unblocked_fraction,
            expected_false_interventions=0.0,
            expected_intervention_cost=0.0,
        )

    def _legal(
        self, state: RuntimeJointPlanningState
    ) -> tuple[RuntimeJointActionTransition, ...]:
        return tuple(
            item
            for item in self.transitions.get(state.state_id, ())
            if item.intervention_cost_units <= state.remaining_intervention_budget
            and (
                item.action_type is not RuntimeJointActionType.CONTINUE_ACTOR
                or state.remaining_actor_steps > 0
            )
        )

    def _action_value(
        self, transition: RuntimeJointActionTransition, depth: int
    ) -> RuntimeJointPolicyValue:
        branches = [
            (outcome.probability, self._best(outcome.next_state_id, depth - 1))
            for outcome in transition.outcomes
        ]
        return RuntimeJointPolicyValue(
            public_success_probability=sum(
                probability * value.public_success_probability
                for probability, value in branches
            ),
            expected_unblocked_fraction=sum(
                probability * value.expected_unblocked_fraction
                for probability, value in branches
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

    def _best(self, state_id: str, depth: int) -> RuntimeJointPolicyValue:
        key = (state_id, depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        state = self.states[state_id]
        best = self._stop_value(state)
        if depth > 0 and not state.public_goal_complete:
            for transition in self._legal(state):
                value = self._action_value(transition, depth)
                if value.priority > best.priority:
                    best = value
        self._cache[key] = best
        return best

    def plan(self, state_id: str) -> RuntimeJointPlanDecision:
        """Return every first-step value so a finalizer can recompute selection."""

        if state_id not in self.states:
            raise ValueError("runtime joint TraceDP start state is unknown")
        state = self.states[state_id]
        q_values = tuple(
            sorted(
                (
                    (item.instance_action_id, self._action_value(item, self.horizon))
                    for item in self._legal(state)
                ),
                key=lambda item: item[0],
            )
        )
        stop = self._stop_value(state)
        best_id: str | None = None
        best_value = stop
        for action_id, value in q_values:
            if value.priority > best_value.priority:
                best_id = action_id
                best_value = value
        return RuntimeJointPlanDecision(best_id, best_value, q_values)


__all__ = [
    "RuntimeJointActionTransition",
    "RuntimeJointPlanDecision",
    "RuntimeJointPlanningState",
    "RuntimeJointPolicyValue",
    "RuntimeJointTraceDPPlanner",
    "RuntimeJointTransitionOutcome",
]
