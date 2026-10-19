"""Persist selector decisions and execution costs for a compiler-owned action slate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionCandidate,
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)

FIXED_JOINT_SLATE_METHOD = "fixed_joint_slate_runtime_v1"
FIXED_JOINT_SLATE_V2_METHOD = "fixed_joint_slate_runtime_v2"
FIXED_JOINT_SLATE_V3_METHOD = "fixed_joint_slate_runtime_v3"
FIXED_JOINT_SLATE_V4_METHOD = "fixed_joint_slate_runtime_v4"
FIXED_JOINT_SLATE_V5_METHOD = "fixed_joint_slate_runtime_v5"
ROTATING_JOINT_SOURCE_METHOD = "rotating_joint_slate_source_v1"
ROTATING_FAILURE_DIAGNOSIS_SOURCE_METHOD = "rotating_failure_diagnosis_source_v1"
ROTATING_FAILURE_DIAGNOSIS_SOURCE_V2_METHOD = "rotating_failure_diagnosis_source_v2"
ROTATING_FAILURE_DIAGNOSIS_SOURCE_V3_METHOD = "rotating_failure_diagnosis_source_v3"
ROTATING_FAILURE_DIAGNOSIS_SOURCE_V4_METHOD = "rotating_failure_diagnosis_source_v4"
ROTATING_FAILURE_DIAGNOSIS_SOURCE_V5_METHOD = "rotating_failure_diagnosis_source_v5"
ROTATING_SETTLEMENT_SOURCE_METHOD = "rotating_settlement_source_v1"
ROTATING_RESIDUAL_SOURCE_METHOD = "rotating_residual_source_v1"
TRACEDP_SETTLEMENT_METHOD = "tracedp_settlement_runtime_v1"
TRACEDP_JIT_SETTLEMENT_METHOD = "tracedp_jit_settlement_runtime_v1"
TRACEDP_FAILURE_DIAGNOSIS_METHOD = "tracedp_failure_diagnosis_runtime_v1"
TRACEDP_RESIDUAL_RUNTIME_METHOD = "tracedp_residual_runtime_v1"
TRACEDP_RESIDUAL_RUNTIME_V2_METHOD = "tracedp_residual_runtime_v2"
TRACEDP_RESIDUAL_RUNTIME_V3_METHOD = "tracedp_residual_runtime_v3"
TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD = "tracedp_jit_residual_runtime_v1"
BOUNDED_TRACEDP_DEV_METHOD = "tracedp_bounded_runtime_dev_v1"
BOUNDED_TRACEDP_DEV_V2_METHOD = "tracedp_bounded_runtime_dev_v2"
BOUNDED_TRACEDP_DEV_V3_METHOD = "tracedp_bounded_runtime_dev_v3"
BOUNDED_TRACEDP_DEV_V4_METHOD = "tracedp_bounded_runtime_dev_v4"
BOUNDED_TRACEDP_DEV_V6_METHOD = "tracedp_bounded_runtime_dev_v6"
BOUNDED_TRACEDP_DEV_V7_METHOD = "tracedp_bounded_runtime_dev_v7"
TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD = "tracedp_event_runtime_dev_v8"
FIXED_RUNTIME_OPTION_CHAIN_DEV_METHOD = "fixed_runtime_option_chain_dev_v1"
FIXED_DELAY_OPTION_CHAIN_DEV_METHOD = "fixed_delay_option_chain_dev_v1"
FORCED_FIRST_FIXED_SOURCE_DEV_METHOD = "forced_first_fixed_source_dev_v1"
FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD = "forced_first_option_dp_source_dev_v1"
TRACEDP_RUNTIME_OWNED_OPTION_DP_DEV_METHOD = (
    "tracedp_runtime_owned_option_dp_dev_v1"
)
TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD = "tracedp_guidance_phase_option_dp_dev_v1"
# Strong no-lookahead comparator: same slate, estimates and budget, greedy choice.
ADAPTIVE_INFORMATION_GAIN_RUNTIME_DEV_METHOD = (
    "adaptive_information_gain_repair_reserve_runtime_dev_v1"
)
# Information-gain base plus recovery-value lookahead only.
TRACEDP_RECOVERY_LOOKAHEAD_DEV_METHOD = "tracedp_recovery_lookahead_dev_v2"
# Residual obligation state with state-dependent value.
TRACEDP_OBLIGATION_VALUE_DEV_METHOD = "tracedp_obligation_value_dev_v1"
TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_DEV_METHOD = (
    "tracedp_runtime_owned_option_dp_jit_dev_v1"
)
# Verified-transition memory adapting the DP's estimate model.
TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_V2_METHOD = (
    "tracedp_runtime_owned_option_dp_jit_v2"
)
# Every selector that consumes the frozen atomic estimates through the live resolver,
# including the information-gain comparator, which shares that binding but not Bellman.
RUNTIME_OWNED_OPTION_DP_DEV_METHODS = frozenset(
    {
        TRACEDP_RUNTIME_OWNED_OPTION_DP_DEV_METHOD,
        ADAPTIVE_INFORMATION_GAIN_RUNTIME_DEV_METHOD,
        TRACEDP_RECOVERY_LOOKAHEAD_DEV_METHOD,
        TRACEDP_OBLIGATION_VALUE_DEV_METHOD,
        TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD,
        TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_DEV_METHOD,
        TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_V2_METHOD,
        FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD,
    }
)
RUNTIME_OPTION_CHAIN_METHODS = frozenset(
    {
        FIXED_RUNTIME_OPTION_CHAIN_DEV_METHOD,
        FIXED_DELAY_OPTION_CHAIN_DEV_METHOD,
        *RUNTIME_OWNED_OPTION_DP_DEV_METHODS,
    }
)
RUNTIME_OPTION_CHAIN_MODELED_ACTION_TYPES = frozenset(
    {
        RuntimeJointActionType.CONTINUE_ACTOR,
        RuntimeJointActionType.RUN_REGISTERED_CHECK,
        RuntimeJointActionType.RUN_TARGETED_REPAIR,
        RuntimeJointActionType.SUBMIT_CURRENT_STATE,
    }
)
RUNTIME_OPTION_ACTION_DOMAIN_CONTRACT = "runtime_option_action_domain_v1"
# Same-boundary follow-up decision after a check that did not open a repair replan.
RUNTIME_OPTION_FOLLOWUP_TRIGGER = "runtime_option_followup_trigger"
BOUNDED_TRACEDP_DEV_METHODS = frozenset(
    {
        BOUNDED_TRACEDP_DEV_METHOD,
        BOUNDED_TRACEDP_DEV_V2_METHOD,
        BOUNDED_TRACEDP_DEV_V3_METHOD,
        BOUNDED_TRACEDP_DEV_V4_METHOD,
        BOUNDED_TRACEDP_DEV_V6_METHOD,
        BOUNDED_TRACEDP_DEV_V7_METHOD,
        TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
    }
)
TRACEDP_RESIDUAL_METHODS = frozenset(
    {
        TRACEDP_FAILURE_DIAGNOSIS_METHOD,
        TRACEDP_RESIDUAL_RUNTIME_METHOD,
        TRACEDP_RESIDUAL_RUNTIME_V2_METHOD,
        TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
        TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD,
    }
)

FAILURE_DIAGNOSIS_SOURCE_METHODS = frozenset(
    {
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_METHOD,
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_V2_METHOD,
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_V3_METHOD,
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_V4_METHOD,
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_V5_METHOD,
    }
)
FAILURE_DIAGNOSIS_JOINT_METHODS = frozenset(
    {
        *FAILURE_DIAGNOSIS_SOURCE_METHODS,
        ROTATING_RESIDUAL_SOURCE_METHOD,
        *TRACEDP_RESIDUAL_METHODS,
    }
)
EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS = frozenset(
    {
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_V5_METHOD,
        ROTATING_RESIDUAL_SOURCE_METHOD,
        *TRACEDP_RESIDUAL_METHODS,
    }
)
# v1 residual behavior is frozen after the real run exposed that it inherited
# diagnosis-only slate suppression. v2 and later compose diagnosis with checks.
FAILURE_DIAGNOSIS_ONLY_JOINT_METHODS = frozenset(
    {
        *FAILURE_DIAGNOSIS_SOURCE_METHODS,
        TRACEDP_FAILURE_DIAGNOSIS_METHOD,
        TRACEDP_RESIDUAL_RUNTIME_METHOD,
        TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD,
    }
)
TIME_RESERVE_JOINT_METHODS = frozenset(
    {
        FIXED_JOINT_SLATE_V4_METHOD,
        FIXED_JOINT_SLATE_V5_METHOD,
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_V2_METHOD,
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_V3_METHOD,
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_V4_METHOD,
        ROTATING_FAILURE_DIAGNOSIS_SOURCE_V5_METHOD,
        *TRACEDP_RESIDUAL_METHODS,
        ROTATING_RESIDUAL_SOURCE_METHOD,
    }
)
ACTOR_RESERVE_JOINT_METHODS = frozenset({FIXED_JOINT_SLATE_V3_METHOD, *TIME_RESERVE_JOINT_METHODS})
SEMANTIC_STABILITY_JOINT_METHODS = frozenset(
    {
        FIXED_JOINT_SLATE_V2_METHOD,
        FIXED_JOINT_SLATE_V3_METHOD,
        FIXED_JOINT_SLATE_V4_METHOD,
        TRACEDP_RESIDUAL_RUNTIME_V2_METHOD,
        TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    }
)
SETTLEMENT_REVIEW_JOINT_METHODS = frozenset(
    {
        FIXED_JOINT_SLATE_V5_METHOD,
        ROTATING_SETTLEMENT_SOURCE_METHOD,
        ROTATING_RESIDUAL_SOURCE_METHOD,
        TRACEDP_SETTLEMENT_METHOD,
        TRACEDP_JIT_SETTLEMENT_METHOD,
        *TRACEDP_RESIDUAL_METHODS,
    }
)
PLANNED_JOINT_METHODS = frozenset(
    {
        TRACEDP_SETTLEMENT_METHOD,
        TRACEDP_JIT_SETTLEMENT_METHOD,
        TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD,
        *RUNTIME_OWNED_OPTION_DP_DEV_METHODS,
    }
)
JIT_JOINT_METHODS = frozenset(
    {
        TRACEDP_JIT_SETTLEMENT_METHOD,
        TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD,
        TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_DEV_METHOD,
        TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_V2_METHOD,
    }
)
SETTLEMENT_ONLY_JOINT_METHODS = frozenset(
    {
        ROTATING_SETTLEMENT_SOURCE_METHOD,
        TRACEDP_SETTLEMENT_METHOD,
        TRACEDP_JIT_SETTLEMENT_METHOD,
    }
)
ROTATING_SOURCE_JOINT_METHODS = frozenset(
    {
        *FAILURE_DIAGNOSIS_SOURCE_METHODS,
        ROTATING_JOINT_SOURCE_METHOD,
        ROTATING_SETTLEMENT_SOURCE_METHOD,
        ROTATING_RESIDUAL_SOURCE_METHOD,
    }
)
RPC_JOINT_METHODS = frozenset({*TIME_RESERVE_JOINT_METHODS, *SETTLEMENT_REVIEW_JOINT_METHODS})
JOINT_POLICY_METHODS = frozenset(
    {
        FIXED_JOINT_SLATE_METHOD,
        FIXED_JOINT_SLATE_V2_METHOD,
        FIXED_JOINT_SLATE_V3_METHOD,
        *RPC_JOINT_METHODS,
        *ROTATING_SOURCE_JOINT_METHODS,
        *RUNTIME_OPTION_CHAIN_METHODS,
    }
)

# The bounded selector changes only action selection. It inherits the complete
# residual-v3 trigger, capability, and executor profile so method comparisons do
# not silently change the slate presented to the selector.
RUNTIME_BEHAVIOR_PROFILE_BASE_METHOD = {    BOUNDED_TRACEDP_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    BOUNDED_TRACEDP_DEV_V2_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    BOUNDED_TRACEDP_DEV_V3_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    BOUNDED_TRACEDP_DEV_V4_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    BOUNDED_TRACEDP_DEV_V6_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    BOUNDED_TRACEDP_DEV_V7_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    FIXED_RUNTIME_OPTION_CHAIN_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    FIXED_DELAY_OPTION_CHAIN_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    TRACEDP_RUNTIME_OWNED_OPTION_DP_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    ADAPTIVE_INFORMATION_GAIN_RUNTIME_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    TRACEDP_RECOVERY_LOOKAHEAD_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    TRACEDP_OBLIGATION_VALUE_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_V2_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD: TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
}


def _inherit_runtime_behavior(methods: frozenset[str]) -> frozenset[str]:
    """Add selectors whose declared base method has the same runtime behavior."""

    return frozenset(
        {
            *methods,
            *(
                method_id
                for method_id, base_method in RUNTIME_BEHAVIOR_PROFILE_BASE_METHOD.items()
                if base_method in methods
            ),
        }
    )


TRACEDP_RESIDUAL_METHODS = _inherit_runtime_behavior(TRACEDP_RESIDUAL_METHODS)
FAILURE_DIAGNOSIS_JOINT_METHODS = _inherit_runtime_behavior(
    FAILURE_DIAGNOSIS_JOINT_METHODS
)
EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS = _inherit_runtime_behavior(
    EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS
)
FAILURE_DIAGNOSIS_ONLY_JOINT_METHODS = _inherit_runtime_behavior(
    FAILURE_DIAGNOSIS_ONLY_JOINT_METHODS
)
TIME_RESERVE_JOINT_METHODS = _inherit_runtime_behavior(TIME_RESERVE_JOINT_METHODS)
ACTOR_RESERVE_JOINT_METHODS = _inherit_runtime_behavior(ACTOR_RESERVE_JOINT_METHODS)
SEMANTIC_STABILITY_JOINT_METHODS = _inherit_runtime_behavior(
    SEMANTIC_STABILITY_JOINT_METHODS
)
# Runtime-owned option-chain methods are excluded from settlement review even
# though they inherit the residual-v3 behavior profile.  Their declared selector
# domain (`RUNTIME_OPTION_CHAIN_MODELED_ACTION_TYPES`) excludes exactly the
# semantic-review actions a natural settlement asks for, and `select()` enforces
# that domain by rejecting any slate without its projection.  Letting the option
# chain claim settlement review therefore makes the two contracts contradict each
# other, which aborted a real run at the first natural settlement with
# "runtime option action domain receipt is invalid" and left no official score.
SETTLEMENT_REVIEW_JOINT_METHODS = _inherit_runtime_behavior(
    SETTLEMENT_REVIEW_JOINT_METHODS
) - RUNTIME_OPTION_CHAIN_METHODS
RPC_JOINT_METHODS = _inherit_runtime_behavior(RPC_JOINT_METHODS)
JOINT_POLICY_METHODS = _inherit_runtime_behavior(JOINT_POLICY_METHODS)


@dataclass(frozen=True)
class RuntimeJointResolvedChoice:
    """Return an auditable learned or planned choice to the lifecycle writer."""

    selected: RuntimeJointActionCandidate
    rule_id: str
    bellman_values: Mapping[str, object]
    jit_advantages: Mapping[str, object] | None
    planning_model: Mapping[str, object] | None = None


class RuntimeJointDecisionResolver(Protocol):
    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice: ...


def project_runtime_option_action_domain(
    slate: RuntimeJointActionSlate,
) -> tuple[RuntimeJointActionSlate, dict[str, object]]:
    """Apply one shared, explicit selector domain to fixed and Bellman policies."""

    projected = replace(
        slate,
        candidates=tuple(
            item
            for item in slate.candidates
            if item.action_type in RUNTIME_OPTION_CHAIN_MODELED_ACTION_TYPES
        ),
    )
    if not projected.candidates or not any(
        item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        for item in projected.candidates
    ):
        raise ValueError("runtime option action domain lacks continuation")
    receipt = {
        "schema_version": RUNTIME_OPTION_ACTION_DOMAIN_CONTRACT,
        "modeled_action_types": sorted(
            item.value for item in RUNTIME_OPTION_CHAIN_MODELED_ACTION_TYPES
        ),
        "source_slate": slate.to_dict(),
        "projected_slate": projected.to_dict(),
        "excluded_candidates": [
            item.to_dict()
            for item in slate.candidates
            if item.action_type not in RUNTIME_OPTION_CHAIN_MODELED_ACTION_TYPES
        ],
    }
    return projected, receipt


def _validate_runtime_option_action_domain(
    slate: RuntimeJointActionSlate,
    receipt: object,
) -> None:
    """Rebuild a recorded domain projection from its complete source slate."""

    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version",
        "modeled_action_types",
        "source_slate",
        "projected_slate",
        "excluded_candidates",
    }:
        raise ValueError("runtime option action domain receipt is invalid")
    source_payload = receipt.get("source_slate")
    if not isinstance(source_payload, Mapping):
        raise ValueError("runtime option action domain source slate is missing")
    source = RuntimeJointActionSlate.from_dict(source_payload)
    projected, expected = project_runtime_option_action_domain(source)
    if projected != slate or expected != dict(receipt):
        raise ValueError("runtime option action domain projection differs")


@dataclass(frozen=True)
class RuntimeJointSelection:
    """Bind one selected instance to the immutable slate that justified it."""

    decision_id: str
    selected: RuntimeJointActionCandidate
    rule_id: str
    slate: RuntimeJointActionSlate


@dataclass(frozen=True)
class _OpenJointDecision:
    decision_id: str
    instance_action_id: str
    action_type: RuntimeJointActionType
    cost_units: int
    executed_action_id: str | None = None


@dataclass(frozen=True)
class RuntimeOptionChain:
    """Persist the check-to-replan obligation across Host process restarts."""

    chain_id: str
    parent_trigger_action_id: str
    prerequisite_decision_id: str
    prerequisite_instance_action_id: str
    expected_check_action_id: str
    conditional_future_option_ids: tuple[str, ...]


class RuntimeJointSlatePolicy:
    """Record fixed, rotating, or resolved choices with their actual decision evidence."""

    schema_version = "runtime_joint_policy_event_v1"
    # The bounded selector is a development Harbor identity and remains outside formal results.
    supported_method_ids = (
        JOINT_POLICY_METHODS
        | BOUNDED_TRACEDP_DEV_METHODS
        | {FORCED_FIRST_FIXED_SOURCE_DEV_METHOD}
    )

    def __init__(
        self,
        *,
        method_id: str,
        budget_units: int,
        event_log_path: Path,
        source_arm_index: int | None = None,
        decision_resolver: RuntimeJointDecisionResolver | None = None,
    ) -> None:
        if method_id not in self.supported_method_ids or budget_units < 1:
            raise ValueError("runtime joint policy identity or budget is invalid")
        if method_id in ROTATING_SOURCE_JOINT_METHODS or method_id in {
            FORCED_FIRST_FIXED_SOURCE_DEV_METHOD,
            FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD,
        }:
            if (
                isinstance(source_arm_index, bool)
                or not isinstance(source_arm_index, int)
                or source_arm_index < 0
            ):
                raise ValueError("rotating joint source policy requires an arm index")
        elif source_arm_index is not None:
            raise ValueError("fixed joint policy cannot receive a source arm index")
        planned_method = (
            method_id in PLANNED_JOINT_METHODS
            or method_id in BOUNDED_TRACEDP_DEV_METHODS
        )
        if planned_method != (decision_resolver is not None):
            raise ValueError("planned joint policy resolver identity differs")
        self.method_id = method_id
        self.total_budget_units = budget_units
        self.remaining_budget_units = budget_units
        self.event_log_path = event_log_path
        self.source_arm_index = source_arm_index
        self.decision_resolver = decision_resolver
        self._sequence = 0
        self._open_decision: _OpenJointDecision | None = None
        self._open_option_chain: RuntimeOptionChain | None = None
        self._first_source_choice_used = False
        self._fixed_delay_used = False

    def _append(self, event_type: str, payload: dict[str, object]) -> None:
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_path.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "schema_version": self.schema_version,
                        "sequence": self._sequence,
                        "event_type": event_type,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._sequence += 1

    def _source_protocol(self, slate: RuntimeJointActionSlate) -> dict[str, object] | None:
        """Bind the one exploratory choice to its unchanged continuation rule."""
        if self.method_id not in {
            FORCED_FIRST_FIXED_SOURCE_DEV_METHOD,
            FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD,
        }:
            return None
        eligible = len(slate.candidates) > 1 and not slate.trigger_action_id.startswith("actor-bridge-")
        phase = (
            "continuation" if self._first_source_choice_used
            else "forced_first_action" if eligible else "before_first_action"
        )
        return {
            "schema_version": "runtime_source_continuation_v1",
            "phase": phase,
            "continuation_policy_id": (
                TRACEDP_RUNTIME_OWNED_OPTION_DP_DEV_METHOD
                if self.method_id == FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD
                else FIXED_JOINT_SLATE_METHOD
            ),
            "first_action_index": self.source_arm_index,
        }

    def _choose(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        """Apply the frozen rule without mutating policy or filesystem state."""

        source = self._source_protocol(slate)
        if source is not None and source["phase"] == "forced_first_action":
            assert self.source_arm_index is not None
            if self.source_arm_index >= len(slate.candidates):
                raise ValueError("forced first source action is absent from the live slate")
            if self.method_id == FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD:
                assert self.decision_resolver is not None
                base = self.decision_resolver.resolve(slate)
                return RuntimeJointResolvedChoice(
                    slate.candidates[self.source_arm_index],
                    "forced_first_option_dp_source_arm_v1",
                    base.bellman_values,
                    None,
                    base.planning_model,
                )
            return RuntimeJointResolvedChoice(
                slate.candidates[self.source_arm_index],
                "forced_first_joint_source_arm_v1", {}, None,
            )

        if (
            self.method_id == FIXED_DELAY_OPTION_CHAIN_DEV_METHOD
            and not self._fixed_delay_used
        ):
            continued = next(
                item
                for item in slate.candidates
                if item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
            )
            return RuntimeJointResolvedChoice(
                continued, "fixed_delay_one_actor_action_v1", {}, None
            )

        if self.method_id in {
            FIXED_JOINT_SLATE_METHOD,
            FIXED_JOINT_SLATE_V2_METHOD,
            FIXED_JOINT_SLATE_V3_METHOD,
            FIXED_JOINT_SLATE_V4_METHOD,
            FIXED_JOINT_SLATE_V5_METHOD,
            FIXED_RUNTIME_OPTION_CHAIN_DEV_METHOD,
            FIXED_DELAY_OPTION_CHAIN_DEV_METHOD,
            FORCED_FIRST_FIXED_SOURCE_DEV_METHOD,
        }:
            available = tuple(
                item
                for item in slate.candidates
                if item.action_type is not RuntimeJointActionType.CONTINUE_ACTOR
            )
            if self.method_id in {
                FIXED_JOINT_SLATE_V4_METHOD,
                FIXED_JOINT_SLATE_V5_METHOD,
            }:
                submission = next(
                    (
                        item
                        for item in available
                        if item.action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE
                    ),
                    None,
                )
                if submission is not None:
                    return RuntimeJointResolvedChoice(
                        submission, "fixed_time_reserve_submission_v1", {}, None
                    )
            return RuntimeJointResolvedChoice(
                available[0] if available else slate.candidates[0],
                "fixed_first_nonempty_joint_action_v1",
                {},
                None,
            )
        if (
            self.method_id
            in (BOUNDED_TRACEDP_DEV_METHODS | RUNTIME_OWNED_OPTION_DP_DEV_METHODS)
            and self.decision_resolver is not None
        ):
            # The bounded planner owns every legal slate, including time-reserve submission.
            return self.decision_resolver.resolve(slate)
        if self.method_id in FAILURE_DIAGNOSIS_JOINT_METHODS:
            submission = next(
                (
                    item
                    for item in slate.candidates
                    if item.action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE
                ),
                None,
            )
            if submission is not None:
                return RuntimeJointResolvedChoice(
                    submission, "shared_time_reserve_submission_v1", {}, None
                )
            if (
                self.method_id in TRACEDP_RESIDUAL_METHODS
                and self.decision_resolver is None
            ):
                managed_action = next(
                    (
                        item
                        for item in slate.candidates
                        if item.action_type
                        in {
                            RuntimeJointActionType.REQUEST_PREDICATE_DIAGNOSIS,
                            RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS,
                            RuntimeJointActionType.REQUEST_REGISTERED_CHECK,
                            RuntimeJointActionType.RUN_REGISTERED_CHECK,
                            RuntimeJointActionType.REQUEST_OUTPUT_COMPLETION,
                            RuntimeJointActionType.REQUEST_SEMANTIC_CHECK,
                        }
                    ),
                    None,
                )
                return RuntimeJointResolvedChoice(
                    managed_action or slate.candidates[0],
                    "tracedp_residual_state_priority_v1",
                    {},
                    None,
                )
        if self.decision_resolver is not None:
            return self.decision_resolver.resolve(slate)
        assert self.source_arm_index is not None
        return RuntimeJointResolvedChoice(
            slate.candidates[self.source_arm_index % len(slate.candidates)],
            "frozen_rotating_joint_source_arm_v1",
            {},
            None,
        )

    def select(
        self,
        slate: RuntimeJointActionSlate,
        *,
        action_domain_projection: Mapping[str, object] | None = None,
    ) -> RuntimeJointSelection:
        """Choose once from the frozen slate and durably record the full boundary."""

        if self._open_decision is not None:
            raise RuntimeError("runtime joint policy already has an open decision")
        if self.method_id in RUNTIME_OPTION_CHAIN_METHODS:
            _validate_runtime_option_action_domain(slate, action_domain_projection)
        elif action_domain_projection is not None:
            raise ValueError("non-option policy received an action domain projection")
        if slate.budget_before != self.remaining_budget_units:
            raise ValueError("runtime joint slate and policy budgets differ")
        choice = self._choose(slate)
        source_protocol = self._source_protocol(slate)
        selected = choice.selected
        rule_id = choice.rule_id
        decision_id = f"runtime-joint-decision-{self._sequence:06d}"
        self._append(
            "decision_selected",
            {
                "decision_id": decision_id,
                "method_id": self.method_id,
                "rule_id": rule_id,
                "source_arm_index": self.source_arm_index,
                "selected_instance_action_id": selected.instance_action_id,
                "selected_action_type": selected.action_type.value,
                "slate": slate.to_dict(),
                "bellman_values": choice.bellman_values or None,
                "jit_advantages": choice.jit_advantages,
                **({"source_protocol": source_protocol} if source_protocol is not None else {}),
                **({"planning_model": choice.planning_model} if choice.planning_model is not None else {}),
                **(
                    {"action_domain_projection": dict(action_domain_projection)}
                    if action_domain_projection is not None
                    else {}
                ),
            },
        )
        if source_protocol is not None and source_protocol["phase"] == "forced_first_action":
            # A persisted continue also consumes exploration; no execution event follows it.
            self._first_source_choice_used = True
        if rule_id == "fixed_delay_one_actor_action_v1":
            self._fixed_delay_used = True
        if selected.action_type is not RuntimeJointActionType.CONTINUE_ACTOR:
            self._open_decision = _OpenJointDecision(
                decision_id=decision_id,
                instance_action_id=selected.instance_action_id,
                action_type=selected.action_type,
                cost_units=selected.cost_units,
            )
        return RuntimeJointSelection(decision_id, selected, rule_id, slate)

    def record_execution_started(self, *, action_id: str) -> None:
        """Bind the selected instance before the environment or Actor is affected."""

        current = self._open_decision
        if current is None or current.executed_action_id is not None or not action_id:
            raise RuntimeError("runtime joint decision cannot start execution")
        self._append(
            "execution_started",
            {
                "decision_id": current.decision_id,
                "instance_action_id": current.instance_action_id,
                "action_type": current.action_type.value,
                "executed_action_id": action_id,
            },
        )
        self._open_decision = _OpenJointDecision(
            current.decision_id,
            current.instance_action_id,
            current.action_type,
            current.cost_units,
            action_id,
        )

    def open_option_chain(
        self,
        *,
        selection: RuntimeJointSelection,
        expected_check_action_id: str,
        conditional_future_option_ids: tuple[str, ...],
    ) -> RuntimeOptionChain:
        """Durably require an evidence-driven replan before Actor execution resumes."""

        current = self._open_decision
        if (
            self.method_id not in RUNTIME_OPTION_CHAIN_METHODS
            or self._open_option_chain is not None
            or current is None
            or current.decision_id != selection.decision_id
            or current.instance_action_id != selection.selected.instance_action_id
            or current.action_type is not RuntimeJointActionType.RUN_REGISTERED_CHECK
            or not expected_check_action_id
            or not conditional_future_option_ids
            or len(set(conditional_future_option_ids))
            != len(conditional_future_option_ids)
        ):
            raise RuntimeError("runtime option chain cannot be opened")
        chain = RuntimeOptionChain(
            chain_id=f"runtime-option-chain:{selection.decision_id}",
            parent_trigger_action_id=selection.slate.trigger_action_id,
            prerequisite_decision_id=selection.decision_id,
            prerequisite_instance_action_id=selection.selected.instance_action_id,
            expected_check_action_id=expected_check_action_id,
            conditional_future_option_ids=conditional_future_option_ids,
        )
        self._append(
            "option_chain_opened",
            {
                "chain_id": chain.chain_id,
                "parent_trigger_action_id": chain.parent_trigger_action_id,
                "prerequisite_decision_id": chain.prerequisite_decision_id,
                "prerequisite_instance_action_id": (
                    chain.prerequisite_instance_action_id
                ),
                "expected_check_action_id": chain.expected_check_action_id,
                "conditional_future_option_ids": list(
                    chain.conditional_future_option_ids
                ),
            },
        )
        self._open_option_chain = chain
        return chain

    def close_option_chain(
        self,
        *,
        evidence_state: str | None,
        reason: str,
        replan_trigger_action_id: str | None,
    ) -> None:
        """Close the persisted chain only after its terminal branch is durable."""

        chain = self._open_option_chain
        if chain is None or not reason:
            raise RuntimeError("runtime option chain cannot be closed")
        if evidence_state not in {None, "supported", "refuted", "unknown"}:
            raise ValueError("runtime option chain evidence state is invalid")
        self._append(
            "option_chain_closed",
            {
                "chain_id": chain.chain_id,
                "evidence_state": evidence_state,
                "reason": reason,
                "replan_trigger_action_id": replan_trigger_action_id,
            },
        )
        self._open_option_chain = None

    def record_outcome(self, *, status: str, reason: str | None) -> None:
        """Charge every started intervention and close its durable decision."""

        current = self._open_decision
        if current is None or current.executed_action_id is None:
            raise RuntimeError("runtime joint decision has no started execution")
        if status not in {"completed", "failed", "cancelled_with_unknown_effect"}:
            raise ValueError("runtime joint decision outcome status is unsupported")
        self.remaining_budget_units -= current.cost_units
        self._append(
            "execution_outcome",
            {
                "decision_id": current.decision_id,
                "instance_action_id": current.instance_action_id,
                "action_type": current.action_type.value,
                "executed_action_id": current.executed_action_id,
                "status": status,
                "reason": reason,
                "cost_units": current.cost_units,
                "remaining_budget": self.remaining_budget_units,
            },
        )
        self._open_decision = None

    def abandon_before_environment_effect(self, *, reason: str) -> None:
        """Close an action whose Kernel history proves the environment was untouched."""

        current = self._open_decision
        if current is None or not reason:
            raise RuntimeError("runtime joint decision cannot be abandoned")
        self._append(
            "decision_abandoned",
            {
                "decision_id": current.decision_id,
                "instance_action_id": current.instance_action_id,
                "action_type": current.action_type.value,
                "executed_action_id": current.executed_action_id,
                "reason": reason,
                "remaining_budget": self.remaining_budget_units,
            },
        )
        self._open_decision = None

    @property
    def has_open_decision(self) -> bool:
        return self._open_decision is not None

    @property
    def open_executed_action_id(self) -> str | None:
        """Expose whether a replayed open decision crossed the execution boundary."""

        return self._open_decision.executed_action_id if self._open_decision is not None else None

    @property
    def open_action_type(self) -> RuntimeJointActionType | None:
        """Expose the selected stable action type for resume-time reconciliation."""

        return self._open_decision.action_type if self._open_decision is not None else None

    @property
    def open_option_chain_state(self) -> RuntimeOptionChain | None:
        """Expose one replayed check-to-replan obligation to the live Gateway."""

        return self._open_option_chain

    @classmethod
    def load(
        cls,
        *,
        method_id: str,
        budget_units: int,
        event_log_path: Path,
        source_arm_index: int | None = None,
        decision_resolver: RuntimeJointDecisionResolver | None = None,
    ) -> RuntimeJointSlatePolicy:
        """Replay selection, execution, and budget state after a host interruption."""

        policy = cls(
            method_id=method_id,
            budget_units=budget_units,
            event_log_path=event_log_path,
            source_arm_index=source_arm_index,
            decision_resolver=decision_resolver,
        )
        for expected_sequence, line in enumerate(event_log_path.read_text().splitlines()):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != cls.schema_version
                or row.get("sequence") != expected_sequence
                or not isinstance(row.get("payload"), dict)
            ):
                raise ValueError("runtime joint policy event sequence differs")
            event_type = row.get("event_type")
            payload = row["payload"]
            if event_type == "decision_selected":
                if policy._open_decision is not None:
                    raise ValueError("runtime joint policy decisions overlap")
                slate_payload = payload.get("slate")
                if not isinstance(slate_payload, dict):
                    raise ValueError("runtime joint policy slate is missing")
                slate = RuntimeJointActionSlate.from_dict(slate_payload)
                action_domain_projection = payload.get("action_domain_projection")
                if method_id in RUNTIME_OPTION_CHAIN_METHODS:
                    _validate_runtime_option_action_domain(
                        slate, action_domain_projection
                    )
                elif action_domain_projection is not None:
                    raise ValueError(
                        "non-option policy recorded an action domain projection"
                    )
                if method_id in (
                    BOUNDED_TRACEDP_DEV_METHODS
                    | RUNTIME_OWNED_OPTION_DP_DEV_METHODS
                ):
                    choice = decision_resolver.resolve_recorded(slate, payload.get("planning_model"))
                else:
                    choice = policy._choose(slate)
                selected = choice.selected
                rule_id = choice.rule_id
                source_protocol = policy._source_protocol(slate)
                if (
                    slate.budget_before != policy.remaining_budget_units
                    or payload.get("method_id") != method_id
                    or payload.get("rule_id") != rule_id
                    or payload.get("source_arm_index") != source_arm_index
                    or payload.get("selected_instance_action_id") != selected.instance_action_id
                    or payload.get("selected_action_type") != selected.action_type.value
                    or payload.get("bellman_values") != (choice.bellman_values or None)
                    or payload.get("jit_advantages") != choice.jit_advantages
                    or payload.get("source_protocol") != source_protocol
                ):
                    raise ValueError("runtime joint policy selection differs")
                if source_protocol is not None and source_protocol["phase"] == "forced_first_action":
                    policy._first_source_choice_used = True
                if rule_id == "fixed_delay_one_actor_action_v1":
                    if policy._fixed_delay_used:
                        raise ValueError("runtime fixed delay was selected more than once")
                    policy._fixed_delay_used = True
                if selected.action_type is not RuntimeJointActionType.CONTINUE_ACTOR:
                    policy._open_decision = _OpenJointDecision(
                        str(payload["decision_id"]),
                        selected.instance_action_id,
                        selected.action_type,
                        selected.cost_units,
                    )
            elif event_type == "execution_started":
                current = policy._open_decision
                action_id = payload.get("executed_action_id")
                if (
                    current is None
                    or current.executed_action_id is not None
                    or payload.get("decision_id") != current.decision_id
                    or payload.get("instance_action_id") != current.instance_action_id
                    or payload.get("action_type") != current.action_type.value
                    or not isinstance(action_id, str)
                    or not action_id
                ):
                    raise ValueError("runtime joint policy execution start differs")
                policy._open_decision = _OpenJointDecision(
                    current.decision_id,
                    current.instance_action_id,
                    current.action_type,
                    current.cost_units,
                    action_id,
                )
            elif event_type == "execution_outcome":
                current = policy._open_decision
                expected_remaining = policy.remaining_budget_units - (
                    current.cost_units if current is not None else 0
                )
                if (
                    current is None
                    or current.executed_action_id is None
                    or payload.get("decision_id") != current.decision_id
                    or payload.get("instance_action_id") != current.instance_action_id
                    or payload.get("action_type") != current.action_type.value
                    or payload.get("executed_action_id") != current.executed_action_id
                    or payload.get("cost_units") != current.cost_units
                    or payload.get("remaining_budget") != expected_remaining
                    or payload.get("status")
                    not in {"completed", "failed", "cancelled_with_unknown_effect"}
                ):
                    raise ValueError("runtime joint policy execution outcome differs")
                policy.remaining_budget_units = expected_remaining
                policy._open_decision = None
            elif event_type == "decision_abandoned":
                current = policy._open_decision
                if (
                    current is None
                    or payload.get("decision_id") != current.decision_id
                    or payload.get("instance_action_id") != current.instance_action_id
                    or payload.get("action_type") != current.action_type.value
                    or payload.get("executed_action_id") != current.executed_action_id
                    or not isinstance(payload.get("reason"), str)
                    or not payload["reason"]
                    or payload.get("remaining_budget") != policy.remaining_budget_units
                ):
                    raise ValueError("runtime joint policy abandonment differs")
                policy._open_decision = None
            elif event_type == "option_chain_opened":
                option_ids = payload.get("conditional_future_option_ids")
                chain = RuntimeOptionChain(
                    chain_id=str(payload.get("chain_id", "")),
                    parent_trigger_action_id=str(
                        payload.get("parent_trigger_action_id", "")
                    ),
                    prerequisite_decision_id=str(
                        payload.get("prerequisite_decision_id", "")
                    ),
                    prerequisite_instance_action_id=str(
                        payload.get("prerequisite_instance_action_id", "")
                    ),
                    expected_check_action_id=str(
                        payload.get("expected_check_action_id", "")
                    ),
                    conditional_future_option_ids=(
                        tuple(str(item) for item in option_ids)
                        if isinstance(option_ids, list)
                        else ()
                    ),
                )
                current = policy._open_decision
                if (
                    method_id not in RUNTIME_OPTION_CHAIN_METHODS
                    or policy._open_option_chain is not None
                    or current is None
                    or current.action_type
                    is not RuntimeJointActionType.RUN_REGISTERED_CHECK
                    or chain.chain_id
                    != f"runtime-option-chain:{current.decision_id}"
                    or chain.prerequisite_decision_id != current.decision_id
                    or chain.prerequisite_instance_action_id
                    != current.instance_action_id
                    or not chain.parent_trigger_action_id
                    or not chain.expected_check_action_id
                    or not chain.conditional_future_option_ids
                    or len(set(chain.conditional_future_option_ids))
                    != len(chain.conditional_future_option_ids)
                ):
                    raise ValueError("runtime option chain opening differs")
                policy._open_option_chain = chain
            elif event_type == "option_chain_closed":
                chain = policy._open_option_chain
                evidence_state = payload.get("evidence_state")
                replan_id = payload.get("replan_trigger_action_id")
                if (
                    chain is None
                    or payload.get("chain_id") != chain.chain_id
                    or evidence_state
                    not in {None, "supported", "refuted", "unknown"}
                    or not isinstance(payload.get("reason"), str)
                    or not payload["reason"]
                    or replan_id is not None
                    and (not isinstance(replan_id, str) or not replan_id)
                ):
                    raise ValueError("runtime option chain closure differs")
                policy._open_option_chain = None
            else:
                raise ValueError("runtime joint policy event type is unsupported")
            policy._sequence += 1
        return policy


__all__ = [
    "ACTOR_RESERVE_JOINT_METHODS",
    "BOUNDED_TRACEDP_DEV_METHOD",
    "BOUNDED_TRACEDP_DEV_METHODS",
    "BOUNDED_TRACEDP_DEV_V2_METHOD",
    "BOUNDED_TRACEDP_DEV_V3_METHOD",
    "BOUNDED_TRACEDP_DEV_V4_METHOD",
    "BOUNDED_TRACEDP_DEV_V6_METHOD",
    "BOUNDED_TRACEDP_DEV_V7_METHOD",
    "TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD",
    "FAILURE_DIAGNOSIS_JOINT_METHODS",
    "EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS",
    "FAILURE_DIAGNOSIS_ONLY_JOINT_METHODS",
    "FAILURE_DIAGNOSIS_SOURCE_METHODS",
    "FIXED_DELAY_OPTION_CHAIN_DEV_METHOD",
    "FIXED_JOINT_SLATE_METHOD",
    "FIXED_JOINT_SLATE_V2_METHOD",
    "FIXED_JOINT_SLATE_V3_METHOD",
    "FIXED_JOINT_SLATE_V4_METHOD",
    "FIXED_JOINT_SLATE_V5_METHOD",
    "FIXED_RUNTIME_OPTION_CHAIN_DEV_METHOD",
    "FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD",
    "JOINT_POLICY_METHODS",
    "JIT_JOINT_METHODS",
    "PLANNED_JOINT_METHODS",
    "ROTATING_FAILURE_DIAGNOSIS_SOURCE_METHOD",
    "ROTATING_FAILURE_DIAGNOSIS_SOURCE_V2_METHOD",
    "ROTATING_FAILURE_DIAGNOSIS_SOURCE_V3_METHOD",
    "ROTATING_JOINT_SOURCE_METHOD",
    "ROTATING_SETTLEMENT_SOURCE_METHOD",
    "ROTATING_RESIDUAL_SOURCE_METHOD",
    "ROTATING_SOURCE_JOINT_METHODS",
    "RPC_JOINT_METHODS",
    "RUNTIME_BEHAVIOR_PROFILE_BASE_METHOD",
    "RUNTIME_OPTION_CHAIN_METHODS",
    "RUNTIME_OPTION_CHAIN_MODELED_ACTION_TYPES",
    "RUNTIME_OPTION_ACTION_DOMAIN_CONTRACT",
    "RUNTIME_OPTION_FOLLOWUP_TRIGGER",
    "RUNTIME_OWNED_OPTION_DP_DEV_METHODS",
    "SETTLEMENT_ONLY_JOINT_METHODS",
    "SETTLEMENT_REVIEW_JOINT_METHODS",
    "SEMANTIC_STABILITY_JOINT_METHODS",
    "TIME_RESERVE_JOINT_METHODS",
    "TRACEDP_JIT_SETTLEMENT_METHOD",
    "TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD",
    "TRACEDP_RUNTIME_OWNED_OPTION_DP_DEV_METHOD",
    "TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD",
    "ADAPTIVE_INFORMATION_GAIN_RUNTIME_DEV_METHOD",
    "TRACEDP_RECOVERY_LOOKAHEAD_DEV_METHOD",
    "TRACEDP_OBLIGATION_VALUE_DEV_METHOD",
    "TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_DEV_METHOD",
    "TRACEDP_FAILURE_DIAGNOSIS_METHOD",
    "TRACEDP_RESIDUAL_METHODS",
    "TRACEDP_RESIDUAL_RUNTIME_METHOD",
    "TRACEDP_RESIDUAL_RUNTIME_V2_METHOD",
    "TRACEDP_RESIDUAL_RUNTIME_V3_METHOD",
    "TRACEDP_SETTLEMENT_METHOD",
    "RuntimeJointDecisionResolver",
    "RuntimeJointResolvedChoice",
    "RuntimeJointSelection",
    "RuntimeJointSlatePolicy",
    "RuntimeOptionChain",
    "project_runtime_option_action_domain",
]
