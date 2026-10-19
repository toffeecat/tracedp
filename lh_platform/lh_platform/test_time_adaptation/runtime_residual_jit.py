"""Apply frozen JIT advantages to the existing residual TraceDP action priority."""

from __future__ import annotations

from dataclasses import dataclass

from lh_platform.test_time_adaptation.runtime_jit_action_advantage import (
    FrozenRuntimeJITMemory,
    RuntimeLegalActionSlate,
    residual_management_state,
)
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionCandidate,
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD,
    RuntimeJointResolvedChoice,
)

_RESIDUAL_MANAGEMENT_ACTIONS = frozenset(
    {
        RuntimeJointActionType.REQUEST_PREDICATE_DIAGNOSIS,
        RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS,
        RuntimeJointActionType.REQUEST_REGISTERED_CHECK,
        RuntimeJointActionType.RUN_REGISTERED_CHECK,
        RuntimeJointActionType.REQUEST_OUTPUT_COMPLETION,
        RuntimeJointActionType.REQUEST_SEMANTIC_CHECK,
    }
)


def _base_choice(slate: RuntimeJointActionSlate) -> RuntimeJointActionCandidate:
    """Reproduce residual v3 priority before applying learned advantages."""

    submission = next(
        (
            item
            for item in slate.candidates
            if item.action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE
        ),
        None,
    )
    if submission is not None:
        return submission
    return next(
        (
            item
            for item in slate.candidates
            if item.action_type in _RESIDUAL_MANAGEMENT_ACTIONS
        ),
        slate.candidates[0],
    )


def _fresh_first_failure(
    slate: RuntimeJointActionSlate,
) -> RuntimeJointActionCandidate | None:
    """Identify the first decision made at the newly observed failure boundary."""

    for candidate in slate.candidates:
        if candidate.action_type is not RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS:
            continue
        if candidate.instance_action_id.split(":", 3)[:3] == [
            "diagnose-failure",
            slate.trigger_action_id,
            "1",
        ]:
            return candidate
    return None


@dataclass(frozen=True)
class RuntimeResidualJITResolver:
    """Correct one exact live residual slate while preserving unsupported fallback."""

    method_id: str
    jit_memory: FrozenRuntimeJITMemory
    test_task_id: str

    def __post_init__(self) -> None:
        if (
            self.method_id != TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD
            or not self.test_task_id
        ):
            raise ValueError("residual TraceDP JIT configuration differs")

    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        """Score compiler-owned instances and retain the complete retrieval receipt."""

        base = _base_choice(slate)
        if base.action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE:
            return RuntimeJointResolvedChoice(
                base,
                "shared_time_reserve_submission_v1",
                {},
                None,
            )
        if len(slate.candidates) == 1:
            return RuntimeJointResolvedChoice(
                base,
                "tracedp_residual_observation_boundary_continue_v1",
                {},
                None,
            )
        fresh_failure = _fresh_first_failure(slate)
        if fresh_failure is None:
            # One JIT episode represents one choice at a newly observed first
            # failure. A later boundary must return to TraceDP instead of
            # repeatedly reusing the same zero-cost continuation advantage.
            return RuntimeJointResolvedChoice(
                base,
                "tracedp_residual_after_jit_boundary_v1",
                {},
                None,
            )
        legal_slate = RuntimeLegalActionSlate(
            joint_slate=slate,
            instance_base_scores=tuple(
                (
                    candidate.instance_action_id,
                    1.0 if candidate is base else 0.0,
                )
                for candidate in slate.candidates
            ),
        )
        jit = self.jit_memory.score_legal_actions(
            state=residual_management_state(slate),
            test_task_id=self.test_task_id,
            legal_slate=legal_slate,
        )
        selected = next(
            item
            for item in slate.candidates
            if item.instance_action_id == jit.chosen_instance_action_id
        )
        return RuntimeJointResolvedChoice(
            selected,
            (
                "tracedp_residual_plus_frozen_jit_advantage_v1"
                if jit.admitted
                else "tracedp_residual_jit_support_fallback_v1"
            ),
            {},
            jit.to_dict(),
        )


__all__ = ["RuntimeResidualJITResolver"]
