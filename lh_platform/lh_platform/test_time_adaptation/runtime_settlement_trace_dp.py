"""Plan the natural-settlement review choice and optionally apply frozen JIT."""

from __future__ import annotations

from dataclasses import dataclass

from lh_platform.test_time_adaptation.runtime_jit_action_advantage import (
    FrozenRuntimeJITMemory,
    RuntimeLegalActionSlate,
    settlement_management_state,
)
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    TRACEDP_JIT_SETTLEMENT_METHOD,
    TRACEDP_SETTLEMENT_METHOD,
    RuntimeJointResolvedChoice,
)
from lh_platform.test_time_adaptation.runtime_joint_trace_dp import (
    RuntimeJointActionTransition,
    RuntimeJointPlanningState,
    RuntimeJointPolicyValue,
    RuntimeJointTraceDPPlanner,
    RuntimeJointTransitionOutcome,
)

_NEUTRAL_SUCCESS_PROBABILITY = 0.5


def _state(
    state_id: str,
    *,
    complete: bool,
    budget: int,
    actor_steps: int,
    output_phase: str,
    last_effect: str,
) -> RuntimeJointPlanningState:
    """Build one explicit settlement state without inferring hidden task facts."""

    return RuntimeJointPlanningState(
        state_id=state_id,
        public_goal_complete=complete,
        unblocked_fraction=1.0,
        remaining_intervention_budget=budget,
        remaining_actor_steps=actor_steps,
        output_phase=output_phase,
        missing_resource_types=(),
        repeat_disappearance_count=0,
        last_intervention_effect=last_effect,
    )


def _value_payload(value: RuntimeJointPolicyValue) -> dict[str, float]:
    return {
        "public_success_probability": value.public_success_probability,
        "expected_unblocked_fraction": value.expected_unblocked_fraction,
        "expected_false_interventions": value.expected_false_interventions,
        "expected_intervention_cost": value.expected_intervention_cost,
    }


def _compile_settlement_plan(slate: RuntimeJointActionSlate):
    """Compile a two-step tree that keeps review delivery and effect distinct."""

    state_key = settlement_management_state(slate)
    candidates = {item.action_type: item for item in slate.candidates}
    if set(candidates) != {
        RuntimeJointActionType.CONTINUE_ACTOR,
        RuntimeJointActionType.REQUEST_SEMANTIC_CHECK,
    }:
        raise ValueError("settlement TraceDP requires the frozen two-action slate")
    continue_action = candidates[RuntimeJointActionType.CONTINUE_ACTOR]
    review_action = candidates[RuntimeJointActionType.REQUEST_SEMANTIC_CHECK]
    remaining_after_review = slate.budget_before - review_action.cost_units
    states = (
        _state(
            "settlement",
            complete=False,
            budget=slate.budget_before,
            actor_steps=1,
            output_phase=state_key.phase,
            last_effect="none",
        ),
        _state(
            "review_changed",
            complete=False,
            budget=remaining_after_review,
            actor_steps=1,
            output_phase="output_present_unverified",
            last_effect="output_changed",
        ),
        _state(
            "review_unchanged",
            complete=False,
            budget=remaining_after_review,
            actor_steps=1,
            output_phase=state_key.phase,
            last_effect="no_observed_effect",
        ),
        _state(
            "direct_official_success",
            complete=True,
            budget=slate.budget_before,
            actor_steps=0,
            output_phase="officially_succeeded",
            last_effect="official_succeeded",
        ),
        _state(
            "direct_official_failure",
            complete=False,
            budget=slate.budget_before,
            actor_steps=0,
            output_phase="officially_failed",
            last_effect="official_failed",
        ),
        _state(
            "reviewed_official_success",
            complete=True,
            budget=remaining_after_review,
            actor_steps=0,
            output_phase="officially_succeeded",
            last_effect="official_succeeded_after_delivery",
        ),
        _state(
            "reviewed_official_failure",
            complete=False,
            budget=remaining_after_review,
            actor_steps=0,
            output_phase="officially_failed",
            last_effect="official_failed_after_delivery",
        ),
    )
    direct_evaluation_outcomes = (
        RuntimeJointTransitionOutcome(
            _NEUTRAL_SUCCESS_PROBABILITY,
            "direct_official_success",
            "official_success",
        ),
        RuntimeJointTransitionOutcome(
            1.0 - _NEUTRAL_SUCCESS_PROBABILITY,
            "direct_official_failure",
            "official_failure",
        ),
    )
    reviewed_evaluation_outcomes = (
        RuntimeJointTransitionOutcome(
            _NEUTRAL_SUCCESS_PROBABILITY,
            "reviewed_official_success",
            "official_success_after_delivery",
        ),
        RuntimeJointTransitionOutcome(
            1.0 - _NEUTRAL_SUCCESS_PROBABILITY,
            "reviewed_official_failure",
            "official_failure_after_delivery",
        ),
    )
    transitions = (
        RuntimeJointActionTransition(
            state_id="settlement",
            instance_action_id=continue_action.instance_action_id,
            action_type=continue_action.action_type,
            intervention_cost_units=continue_action.cost_units,
            expected_false_interventions=0.0,
            prediction_evidence_id="locally-designed-neutral-settlement-prior-v1",
            outcomes=direct_evaluation_outcomes,
        ),
        RuntimeJointActionTransition(
            state_id="settlement",
            instance_action_id=review_action.instance_action_id,
            action_type=review_action.action_type,
            intervention_cost_units=review_action.cost_units,
            expected_false_interventions=0.5,
            prediction_evidence_id="locally-designed-neutral-review-effect-prior-v1",
            outcomes=(
                RuntimeJointTransitionOutcome(0.5, "review_changed", "output_changed"),
                RuntimeJointTransitionOutcome(
                    0.5, "review_unchanged", "no_observed_effect"
                ),
            ),
        ),
        *(
            RuntimeJointActionTransition(
                state_id=state_id,
                instance_action_id=f"continue_after_{state_id}",
                action_type=RuntimeJointActionType.CONTINUE_ACTOR,
                intervention_cost_units=0,
                expected_false_interventions=0.0,
                prediction_evidence_id="locally-designed-neutral-settlement-prior-v1",
                outcomes=reviewed_evaluation_outcomes,
            )
            for state_id in ("review_changed", "review_unchanged")
        ),
    )
    decision = RuntimeJointTraceDPPlanner(
        states=states,
        transitions=transitions,
        horizon=2,
    ).plan("settlement")
    if decision.instance_action_id is None:
        raise RuntimeError("settlement TraceDP selected an implicit stop")
    return state_key, decision


def _rank_base_scores(decision) -> tuple[tuple[str, float], ...]:
    """Project the full Bellman priority into a frozen order-preserving score."""

    ordered = sorted(
        decision.q_values,
        key=lambda item: (item[1].priority, item[0]),
        reverse=True,
    )
    if len(ordered) == 1:
        return ((ordered[0][0], 1.0),)
    denominator = len(ordered) - 1
    return tuple(
        (action_id, 1.0 - rank / denominator)
        for rank, (action_id, _) in enumerate(ordered)
    )


@dataclass(frozen=True)
class RuntimeSettlementTraceDPResolver:
    """Resolve one compiler-owned settlement slate with TraceDP and frozen JIT."""

    method_id: str
    jit_memory: FrozenRuntimeJITMemory | None = None
    test_task_id: str | None = None

    def __post_init__(self) -> None:
        if self.method_id not in {
            TRACEDP_SETTLEMENT_METHOD,
            TRACEDP_JIT_SETTLEMENT_METHOD,
        }:
            raise ValueError("settlement TraceDP method identity differs")
        jit_enabled = self.method_id == TRACEDP_JIT_SETTLEMENT_METHOD
        if jit_enabled != (self.jit_memory is not None and self.test_task_id is not None):
            raise ValueError("settlement TraceDP JIT configuration differs")

    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        """Select from the exact live slate and preserve every score for replay."""

        continue_action = next(
            item
            for item in slate.candidates
            if item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        )
        if (
            len(slate.candidates) == 1
            and slate.candidates[0] is continue_action
        ):
            return RuntimeJointResolvedChoice(
                continue_action,
                "tracedp_observation_boundary_continue_v1",
                {},
                None,
            )
        if tuple(item.action_type for item in slate.candidates) != (
            RuntimeJointActionType.CONTINUE_ACTOR,
            RuntimeJointActionType.REQUEST_SEMANTIC_CHECK,
        ):
            # This policy has no calibrated value model for choosing between
            # multiple task-local review instances. Preserve the live session
            # and expose the missing capability instead of aborting the trial.
            jit_fallback = (
                {
                    "chosen_instance_action_id": continue_action.instance_action_id,
                    "chosen_action_type": continue_action.action_type.value,
                    "admitted": False,
                    "fallback_reason": "unsupported_settlement_slate_shape",
                    "capability_signature": slate.capability_signature,
                }
                if self.jit_memory is not None
                else None
            )
            return RuntimeJointResolvedChoice(
                continue_action,
                "tracedp_unsupported_settlement_slate_fallback_v1",
                {},
                jit_fallback,
            )
        state, plan = _compile_settlement_plan(slate)
        q_values = {
            action_id: _value_payload(value) for action_id, value in plan.q_values
        }
        selected_id = plan.instance_action_id
        jit_payload = None
        rule_id = "tracedp_neutral_settlement_bellman_v1"
        if self.jit_memory is not None and self.test_task_id is not None:
            jit = self.jit_memory.score_legal_actions(
                state=state,
                test_task_id=self.test_task_id,
                legal_slate=RuntimeLegalActionSlate(
                    joint_slate=slate,
                    instance_base_scores=_rank_base_scores(plan),
                ),
            )
            selected_id = jit.chosen_instance_action_id
            jit_payload = jit.to_dict()
            rule_id = (
                "tracedp_plus_frozen_jit_advantage_v1"
                if jit.admitted
                else "tracedp_jit_support_fallback_v1"
            )
        selected = next(
            item for item in slate.candidates if item.instance_action_id == selected_id
        )
        return RuntimeJointResolvedChoice(selected, rule_id, q_values, jit_payload)


__all__ = ["RuntimeSettlementTraceDPResolver"]
