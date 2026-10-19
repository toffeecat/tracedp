"""Typed historical O/R/S evidence and candidate-level raw pattern scoring.

This module records original (O), replacement (R), and matched-sham (S) replay
observations without turning a candidate-local record into a causal verdict.  It
performs no I/O and remains unbound to the registry, immutable release, complete
repetition roster, or formal budget evaluator.  Formal causal labels therefore
belong to a future bound wrapper; every record here remains
``implementation_unverified`` and ``unbound_candidate_only``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracefix_bench.contracts import ExecutionSemantics, TraceFixTrack
from lh_platform.tracefix_bench.dataset_contracts import TRACEFIX_BENCH_SUITE_ID
from lh_platform.tracefix_bench.run_contracts import ResourceCostV1
from lh_platform.tracefix_bench.scorer_contracts import (
    BudgetScope,
    EvidenceSource,
    ProgressDirection,
    ProgressMetricContractV1,
    SCORER_REGISTRY_SUITE_VERSION_V0_1,
    VerifierContractV1,
)

HISTORICAL_IMPLEMENTATION_STATUS = "implementation_unverified"
HISTORICAL_INPUT_BINDING_STATUS = "unbound_candidate_only"
HISTORICAL_EVIDENCE_REF_SCHEMA = (
    "longhorizon_tracefix_bench/historical_evidence_ref/v1"
)
HISTORICAL_INTERVENTION_PLAN_SCHEMA = (
    "longhorizon_tracefix_bench/historical_intervention_plan/v1"
)
HISTORICAL_INTERVENTION_EXECUTION_SCHEMA = (
    "longhorizon_tracefix_bench/historical_intervention_execution/v1"
)
SHAM_CONTRACT_SCHEMA = "longhorizon_tracefix_bench/historical_sham_contract/v1"
HISTORICAL_REPETITION_SPEC_SCHEMA = (
    "longhorizon_tracefix_bench/historical_repetition_spec/v1"
)
HISTORICAL_REPETITION_CONTRACT_SCHEMA = (
    "longhorizon_tracefix_bench/historical_repetition_contract/v1"
)
HISTORICAL_OFFICIAL_OUTCOME_SCHEMA = (
    "longhorizon_tracefix_bench/historical_official_outcome/v1"
)
HISTORICAL_PROGRESS_OBSERVATION_SCHEMA = (
    "longhorizon_tracefix_bench/historical_progress_observation/v1"
)
HISTORICAL_SHAM_FIELD_MATCH_SCHEMA = (
    "longhorizon_tracefix_bench/historical_sham_field_match/v1"
)
HISTORICAL_SHAM_MATCHER_DECISION_SCHEMA = (
    "longhorizon_tracefix_bench/historical_sham_matcher_decision/v1"
)
HISTORICAL_PARITY_DECISION_SCHEMA = (
    "longhorizon_tracefix_bench/historical_parity_decision/v1"
)
HISTORICAL_CANDIDATE_COMPARISON_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/historical_candidate_comparison_receipt/v1"
)
HISTORICAL_ARM_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/historical_arm_receipt/v1"
)
HISTORICAL_THREE_ARM_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/historical_three_arm_receipt/v1"
)
HISTORICAL_PROGRESS_EFFECTS_SCHEMA = (
    "longhorizon_tracefix_bench/historical_progress_effects/v1"
)
HISTORICAL_CANDIDATE_SCORE_SCHEMA = (
    "longhorizon_tracefix_bench/historical_candidate_score/v1"
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


def _identifier(value: str, label: str) -> None:
    """Require a stable portable identifier without rewriting it."""

    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable non-empty identifier")


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _optional_text(value: str | None, label: str) -> None:
    if value is not None:
        _text(value, label)


def _text_tuple(
    values: tuple[str, ...],
    label: str,
    *,
    unique: bool = False,
    nonempty: bool = False,
) -> None:
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple")
    for value in values:
        _text(value, f"{label} item")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{label} must not repeat values")
    if nonempty and not values:
        raise ValueError(f"{label} must not be empty")


def _finite(value: int | float | None, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")


def _record_tuple(values: tuple[object, ...], kind: type, label: str) -> None:
    if type(values) is not tuple or any(type(value) is not kind for value in values):
        raise ValueError(f"{label} must be a tuple of {kind.__name__}")


def _cost_tuple(
    values: tuple[ResourceCostV1, ...],
    label: str,
    *,
    require_nonempty: bool,
) -> None:
    """Keep candidate accounting separate from whole-run overhead."""

    _record_tuple(values, ResourceCostV1, label)
    if require_nonempty and not values:
        raise ValueError(f"{label} must preserve at least one resource state")
    if any(cost.scope is BudgetScope.WHOLE_RUN for cost in values):
        raise ValueError(f"{label} cannot contain whole-run costs")
    keys = tuple((cost.role, cost.scope, cost.dimension) for cost in values)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} cannot repeat a role, scope, and dimension")


def _unverified(value: str) -> None:
    if value != HISTORICAL_IMPLEMENTATION_STATUS:
        raise ValueError("historical records remain implementation_unverified")


class HistoricalArmRole(StrEnum):
    """Name the original, replacement, and matched-sham replay arms."""

    ORIGINAL = "original"
    REPLACEMENT = "replacement"
    MATCHED_SHAM = "matched_sham"


_ALL_ARM_ROLES = frozenset(HistoricalArmRole)


class HistoricalArmExecutionStatus(StrEnum):
    """Separate completed arms from method, infrastructure, and resume states."""

    COMPLETED = "completed"
    METHOD_FAILED = "method_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    IN_FLIGHT = "in_flight"


class HistoricalParityStatus(StrEnum):
    """Represent observed equality, inequality, or unavailable comparison."""

    VALID = "valid"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


class HistoricalParityDimension(StrEnum):
    """Freeze replay dimensions that a later formal wrapper must bind."""

    PRE_ACTION_STATE = "pre_action_state"
    RESTORE = "restore"
    ACTOR = "actor"
    RUNTIME_ENVIRONMENT = "runtime_environment"
    CONTINUATION_SEED = "continuation_seed"
    ACTION_PROTOCOL = "action_protocol"
    VERIFIER_PROTOCOL = "verifier_protocol"
    BUDGET_COMPARABILITY = "budget_comparability"
    ARM_ISOLATION = "arm_isolation"
    SHAM_MATCHING = "sham_matching"


_ALL_PARITY_DIMENSIONS = frozenset(HistoricalParityDimension)


class HistoricalValueAvailability(StrEnum):
    """State whether an official observation is usable."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class HistoricalCandidateStatus(StrEnum):
    """Describe the lifecycle state of one preregistered candidate."""

    EXECUTED = "executed"
    CANDIDATE_NOT_EXECUTED = "candidate_not_executed"
    METHOD_FAILED = "method_failed"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    AMBIGUOUS_RESUME = "ambiguous_resume"


class HistoricalCandidateDecisionStatus(StrEnum):
    """Describe candidate scoring without implying a formal causal verdict."""

    PATTERN_AVAILABLE = "pattern_available"
    CANDIDATE_NOT_EXECUTED = "candidate_not_executed"
    METHOD_FAILED = "method_failed"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    AMBIGUOUS_RESUME = "ambiguous_resume"


class HistoricalOutcomePattern(StrEnum):
    """Enumerate raw O/R/S outcome and progress patterns."""

    ORIGINAL_SUCCESS = "original_success"
    REPLACEMENT_ONLY_SUCCESS = "replacement_only_success"
    REPLACEMENT_AND_SHAM_SUCCESS = "replacement_and_sham_success"
    SHAM_ONLY_SUCCESS = "sham_only_success"
    ALL_FAILED_WITH_REPLACEMENT_PROGRESS = "all_failed_with_replacement_progress"
    ALL_FAILED_WITH_SHARED_PROGRESS = "all_failed_with_shared_progress"
    ALL_FAILED_WITH_NO_MEANINGFUL_PROGRESS = "all_failed_with_no_meaningful_progress"
    ALL_FAILED_WITH_PROGRESS_UNAVAILABLE = "all_failed_with_progress_unavailable"
    ALL_FAILED_WITH_PROGRESS_OUT_OF_DOMAIN = "all_failed_with_progress_out_of_domain"


class HistoricalCausalClass(StrEnum):
    """Reserve formal labels for a future registry-bound repetition reducer."""

    PARITY_INVALID = "parity_invalid"
    ORIGINAL_NOT_FAILED = "original_not_failed"
    REPAIR_SPECIFIC_RESOLUTION = "repair_specific_resolution"
    SHARED_GAIN = "shared_gain"
    NO_EFFECT = "no_effect"
    REPAIR_SPECIFIC_PROGRESS = "repair_specific_progress"


class HistoricalCandidateNotExecutedReason(StrEnum):
    """Record why a planned candidate received no replay arms."""

    LOCATOR_ABSTAINED = "locator_abstained"
    FIXER_ABSTAINED = "fixer_abstained"
    NO_CATALOG_ACTION = "no_catalog_action"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STOPPED_AFTER_STRICT_ROOT = "stopped_after_strict_root"


class HistoricalInvalidReason(StrEnum):
    """Keep non-pattern terminal reasons closed and machine countable."""

    CONTRACT_IDENTITY_MISMATCH = "contract_identity_mismatch"
    BENCHMARK_DOMAIN_MISMATCH = "benchmark_domain_mismatch"
    REPETITION_PLAN_MISMATCH = "repetition_plan_mismatch"
    ARM_IDENTITY_MISMATCH = "arm_identity_mismatch"
    ARM_MISSING = "arm_missing"
    ARM_METHOD_FAILURE = "arm_method_failure"
    ARM_INFRASTRUCTURE_FAILURE = "arm_infrastructure_failure"
    ARM_IN_FLIGHT = "arm_in_flight"
    OFFICIAL_OUTCOME_UNAVAILABLE = "official_outcome_unavailable"
    SCORER_INPUT_INVALID = "scorer_input_invalid"


class HistoricalEvidenceOwnerScope(StrEnum):
    """Distinguish arm-exclusive artifacts from candidate-shared evidence."""

    ARM_EXCLUSIVE = "arm_exclusive"
    CANDIDATE_SHARED = "candidate_shared"


class HistoricalEvidenceKind(StrEnum):
    """Name the runtime artifact role of one historical evidence reference."""

    INTERVENTION_EXECUTION = "intervention_execution"
    RESTORE = "restore"
    TERMINAL_STATE = "terminal_state"
    OFFICIAL_RAW_OUTPUT = "official_raw_output"
    OFFICIAL_VERIFIER = "official_verifier"
    OFFICIAL_AVAILABILITY = "official_availability"
    PROGRESS_EVALUATOR = "progress_evaluator"
    PROGRESS_AVAILABILITY = "progress_availability"
    ACTION_LEDGER = "action_ledger"
    PARITY = "parity"
    SHAM_MATCHER = "sham_matcher"


class HistoricalInterventionKind(StrEnum):
    """Close the three intervention roles used by historical replay."""

    ORIGINAL_CONTINUATION = "original_continuation"
    REPLACEMENT_REPAIR = "replacement_repair"
    MATCHED_SHAM_CONTROL = "matched_sham_control"


class ShamSemantics(StrEnum):
    """Freeze the v0.1 matched-sham meaning."""

    MATCHED_GENERIC_DEBT_RELEASE = "matched_generic_debt_release"


class HistoricalShamMatchField(StrEnum):
    """Close the fields that the typed R/S resolver may compare."""

    CANDIDATE_EVENT_GROUP = "candidate_event_group"
    TARGET_EVENT_GROUP = "target_event_group"
    ACTION_PROTOCOL = "action_protocol"
    CONTROLLED_ACTIONS = "controlled_actions"
    POST_INTERVENTION_STATE = "post_intervention_state"


class HistoricalShamFieldRelation(StrEnum):
    """Preserve the resolver's raw field relation."""

    MATCHED = "matched"
    DIFFERENT = "different"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class HistoricalEvidenceRefV1:
    """Bind an artifact to its exact ledger and historical candidate owner."""

    artifact_ref: str
    kind: HistoricalEvidenceKind
    owner_scope: HistoricalEvidenceOwnerScope
    run_id: str
    work_item_id: str
    action_id: str
    case_id: str
    candidate_id: str
    repetition_id: str
    arm: HistoricalArmRole | None
    ledger_binding_status: str = HISTORICAL_INPUT_BINDING_STATUS
    schema: str = HISTORICAL_EVIDENCE_REF_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_EVIDENCE_REF_SCHEMA:
            raise ValueError("unsupported historical evidence reference schema")
        _text(self.artifact_ref, "artifact_ref")
        for label in (
            "run_id",
            "work_item_id",
            "action_id",
            "case_id",
            "candidate_id",
            "repetition_id",
        ):
            _identifier(getattr(self, label), label)
        if not isinstance(self.kind, HistoricalEvidenceKind) or not isinstance(
            self.owner_scope, HistoricalEvidenceOwnerScope
        ):
            raise ValueError("evidence kind and owner scope must use closed enums")
        if self.owner_scope is HistoricalEvidenceOwnerScope.ARM_EXCLUSIVE:
            if not isinstance(self.arm, HistoricalArmRole):
                raise ValueError("arm-exclusive evidence requires one closed arm owner")
        elif self.arm is not None:
            raise ValueError("candidate-shared evidence cannot claim an arm owner")
        if self.ledger_binding_status != HISTORICAL_INPUT_BINDING_STATUS:
            raise ValueError("historical evidence owner remains unbound to the run ledger")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "artifact_ref": self.artifact_ref,
            "kind": self.kind.value,
            "owner_scope": self.owner_scope.value,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "action_id": self.action_id,
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "repetition_id": self.repetition_id,
            "arm": self.arm.value if self.arm is not None else None,
            "ledger_binding_status": self.ledger_binding_status,
        }


def _require_arm_evidence(
    reference: HistoricalEvidenceRefV1,
    *,
    kind: HistoricalEvidenceKind,
    run_id: str,
    work_item_id: str,
    action_id: str,
    case_id: str,
    candidate_id: str,
    repetition_id: str,
    arm: HistoricalArmRole,
    allow_candidate_shared: bool = False,
) -> None:
    """Validate artifact role and owner without inferring equality from a path."""

    if type(reference) is not HistoricalEvidenceRefV1 or reference.kind is not kind:
        raise ValueError(f"{kind.value} evidence must use its typed reference")
    if (
        reference.run_id,
        reference.work_item_id,
        reference.action_id,
        reference.case_id,
        reference.candidate_id,
        reference.repetition_id,
    ) != (
        run_id,
        work_item_id,
        action_id,
        case_id,
        candidate_id,
        repetition_id,
    ):
        raise ValueError(f"{kind.value} evidence owner tuple drifted")
    arm_owned = (
        reference.owner_scope is HistoricalEvidenceOwnerScope.ARM_EXCLUSIVE
        and reference.arm is arm
    )
    candidate_owned = (
        allow_candidate_shared
        and reference.owner_scope is HistoricalEvidenceOwnerScope.CANDIDATE_SHARED
        and reference.arm is None
    )
    if not (arm_owned or candidate_owned):
        raise ValueError(f"{kind.value} evidence has the wrong arm ownership")


def _require_candidate_evidence(
    reference: HistoricalEvidenceRefV1,
    *,
    kind: HistoricalEvidenceKind,
    run_id: str,
    work_item_id: str,
    action_id: str,
    case_id: str,
    candidate_id: str,
    repetition_id: str,
) -> None:
    if type(reference) is not HistoricalEvidenceRefV1 or reference.kind is not kind:
        raise ValueError(f"{kind.value} evidence must use its typed reference")
    if (
        reference.owner_scope is not HistoricalEvidenceOwnerScope.CANDIDATE_SHARED
        or reference.arm is not None
        or (
            reference.run_id,
            reference.work_item_id,
            reference.action_id,
            reference.case_id,
            reference.candidate_id,
            reference.repetition_id,
        )
        != (
            run_id,
            work_item_id,
            action_id,
            case_id,
            candidate_id,
            repetition_id,
        )
    ):
        raise ValueError(f"{kind.value} evidence must be candidate-shared")


@dataclass(frozen=True)
class HistoricalInterventionPlanV1:
    """Bind an intervention role to its exact candidate target and actions."""

    run_id: str
    work_item_id: str
    action_id: str
    case_id: str
    candidate_id: str
    repetition_id: str
    arm: HistoricalArmRole
    intervention_id: str
    kind: HistoricalInterventionKind
    candidate_event_group: tuple[str, ...]
    target_event_group: tuple[str, ...]
    controlled_actions: tuple[str, ...]
    action_protocol_id: str
    sham_semantics: ShamSemantics | None
    schema: str = HISTORICAL_INTERVENTION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_INTERVENTION_PLAN_SCHEMA:
            raise ValueError("unsupported historical intervention plan schema")
        for label in (
            "run_id",
            "work_item_id",
            "action_id",
            "case_id",
            "candidate_id",
            "repetition_id",
            "intervention_id",
            "action_protocol_id",
        ):
            _identifier(getattr(self, label), label)
        if not isinstance(self.arm, HistoricalArmRole) or not isinstance(
            self.kind, HistoricalInterventionKind
        ):
            raise ValueError("intervention arm and kind must use closed enums")
        expected_kind = {
            HistoricalArmRole.ORIGINAL: HistoricalInterventionKind.ORIGINAL_CONTINUATION,
            HistoricalArmRole.REPLACEMENT: HistoricalInterventionKind.REPLACEMENT_REPAIR,
            HistoricalArmRole.MATCHED_SHAM: HistoricalInterventionKind.MATCHED_SHAM_CONTROL,
        }[self.arm]
        if self.kind is not expected_kind:
            raise ValueError("intervention kind contradicts its replay arm")
        _text_tuple(
            self.candidate_event_group,
            "candidate_event_group",
            unique=True,
            nonempty=True,
        )
        _text_tuple(
            self.target_event_group,
            "target_event_group",
            unique=True,
            nonempty=True,
        )
        _text_tuple(
            self.controlled_actions,
            "controlled_actions",
            nonempty=True,
        )
        if self.arm is HistoricalArmRole.MATCHED_SHAM:
            if not isinstance(self.sham_semantics, ShamSemantics):
                raise ValueError("matched-sham plan requires closed sham semantics")
        elif self.sham_semantics is not None:
            raise ValueError("only the matched-sham arm can claim sham semantics")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "action_id": self.action_id,
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "repetition_id": self.repetition_id,
            "arm": self.arm.value,
            "intervention_id": self.intervention_id,
            "kind": self.kind.value,
            "candidate_event_group": list(self.candidate_event_group),
            "target_event_group": list(self.target_event_group),
            "controlled_actions": list(self.controlled_actions),
            "action_protocol_id": self.action_protocol_id,
            "sham_semantics": (
                self.sham_semantics.value if self.sham_semantics is not None else None
            ),
        }


@dataclass(frozen=True)
class HistoricalInterventionExecutionV1:
    """Bind one typed plan to its observed post-intervention state and artifact."""

    plan: HistoricalInterventionPlanV1
    post_intervention_state_id: str | None
    executed_actions: tuple[str, ...]
    controlled_action_count: int
    execution_evidence_ref: HistoricalEvidenceRefV1
    schema: str = HISTORICAL_INTERVENTION_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_INTERVENTION_EXECUTION_SCHEMA:
            raise ValueError("unsupported historical intervention execution schema")
        if type(self.plan) is not HistoricalInterventionPlanV1:
            raise ValueError("intervention execution requires a typed plan")
        _optional_text(self.post_intervention_state_id, "post_intervention_state_id")
        _text_tuple(self.executed_actions, "executed_actions")
        if (
            type(self.controlled_action_count) is not int
            or self.controlled_action_count < 0
            or self.controlled_action_count > len(self.plan.controlled_actions)
            or self.controlled_action_count > len(self.executed_actions)
        ):
            raise ValueError("controlled_action_count must fit plan and execution")
        controlled_prefix = self.executed_actions[: self.controlled_action_count]
        if controlled_prefix != self.plan.controlled_actions[: self.controlled_action_count]:
            raise ValueError("executed controlled actions must preserve the plan prefix")
        controlled_complete = self.controlled_action_count == len(
            self.plan.controlled_actions
        )
        if not controlled_complete and len(self.executed_actions) != self.controlled_action_count:
            raise ValueError("actor continuation cannot begin before controlled actions finish")
        if controlled_complete:
            _text(self.post_intervention_state_id, "post_intervention_state_id")
        elif self.post_intervention_state_id is not None:
            raise ValueError("partial controlled execution has no post-intervention state")
        _require_arm_evidence(
            self.execution_evidence_ref,
            kind=HistoricalEvidenceKind.INTERVENTION_EXECUTION,
            run_id=self.plan.run_id,
            work_item_id=self.plan.work_item_id,
            action_id=self.plan.action_id,
            case_id=self.plan.case_id,
            candidate_id=self.plan.candidate_id,
            repetition_id=self.plan.repetition_id,
            arm=self.plan.arm,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "plan": self.plan.to_dict(),
            "post_intervention_state_id": self.post_intervention_state_id,
            "executed_actions": list(self.executed_actions),
            "controlled_action_count": self.controlled_action_count,
            "execution_evidence_ref": self.execution_evidence_ref.to_dict(),
        }

    @property
    def actor_actions(self) -> tuple[str, ...]:
        """Return only the closed-loop Actor continuation after controlled actions."""

        return self.executed_actions[self.controlled_action_count :]


@dataclass(frozen=True)
class ShamContractV1:
    """Declare which closed R/S fields must match or may differ."""

    sham_contract_id: str
    sham_contract_version: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    semantics: ShamSemantics
    matched_fields: tuple[HistoricalShamMatchField, ...]
    permitted_differences: tuple[HistoricalShamMatchField, ...]
    source_ref: str
    implementation_status: str = HISTORICAL_IMPLEMENTATION_STATUS
    schema: str = SHAM_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHAM_CONTRACT_SCHEMA:
            raise ValueError("unsupported sham contract schema")
        for label in (
            "sham_contract_id",
            "sham_contract_version",
            "benchmark_id",
            "benchmark_version",
            "adapter_version",
        ):
            _identifier(getattr(self, label), label)
        if not isinstance(self.semantics, ShamSemantics):
            raise ValueError("sham semantics must use the closed enum")
        for values, label in (
            (self.matched_fields, "matched_fields"),
            (self.permitted_differences, "permitted_differences"),
        ):
            if type(values) is not tuple or any(
                not isinstance(value, HistoricalShamMatchField) for value in values
            ):
                raise ValueError(f"{label} must use closed sham fields")
            if len(set(values)) != len(values):
                raise ValueError(f"{label} cannot repeat fields")
        if not self.matched_fields:
            raise ValueError("matched sham requires at least one matched field")
        if set(self.matched_fields) & set(self.permitted_differences):
            raise ValueError("a sham field cannot be both matched and permitted to differ")
        if set(self.matched_fields) | set(self.permitted_differences) != set(
            HistoricalShamMatchField
        ):
            raise ValueError("sham contract must classify every closed resolver field")
        _text(self.source_ref, "source_ref")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sham_contract_id": self.sham_contract_id,
            "sham_contract_version": self.sham_contract_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "semantics": self.semantics.value,
            "matched_fields": [field.value for field in self.matched_fields],
            "permitted_differences": [
                field.value for field in self.permitted_differences
            ],
            "source_ref": self.source_ref,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class HistoricalRepetitionSpecV1:
    """Bind one repetition to an independent seed and preregistered arm order."""

    repetition_id: str
    continuation_seed: int
    arm_order: tuple[HistoricalArmRole, ...]
    schema: str = HISTORICAL_REPETITION_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_REPETITION_SPEC_SCHEMA:
            raise ValueError("unsupported historical repetition spec schema")
        _identifier(self.repetition_id, "repetition_id")
        if type(self.continuation_seed) is not int or self.continuation_seed < 0:
            raise ValueError("continuation_seed must be a nonnegative integer")
        if (
            type(self.arm_order) is not tuple
            or len(self.arm_order) != 3
            or any(not isinstance(role, HistoricalArmRole) for role in self.arm_order)
            or set(self.arm_order) != _ALL_ARM_ROLES
        ):
            raise ValueError("arm_order must be one permutation of O, R, and S")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "repetition_id": self.repetition_id,
            "continuation_seed": self.continuation_seed,
            "arm_order": [role.value for role in self.arm_order],
        }


@dataclass(frozen=True)
class HistoricalRepetitionContractV1:
    """Freeze repetitions and the raw pattern required by a future reducer."""

    repetition_contract_id: str
    repetition_contract_version: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    repetitions: tuple[HistoricalRepetitionSpecV1, ...]
    continuation_seed_rule_id: str
    continuation_seed_rule_source_ref: str
    root_identity_matching_rule_id: str
    root_identity_matching_rule_source_ref: str
    strict_reducer_id: str
    strict_reducer_version: str
    strict_reducer_source_ref: str
    all_repetitions_required: bool
    required_pattern: HistoricalOutcomePattern
    implementation_status: str = HISTORICAL_IMPLEMENTATION_STATUS
    schema: str = HISTORICAL_REPETITION_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_REPETITION_CONTRACT_SCHEMA:
            raise ValueError("unsupported historical repetition contract schema")
        for label in (
            "repetition_contract_id",
            "repetition_contract_version",
            "benchmark_id",
            "benchmark_version",
            "adapter_version",
            "continuation_seed_rule_id",
            "root_identity_matching_rule_id",
            "strict_reducer_id",
            "strict_reducer_version",
        ):
            _identifier(getattr(self, label), label)
        for label in (
            "continuation_seed_rule_source_ref",
            "root_identity_matching_rule_source_ref",
            "strict_reducer_source_ref",
        ):
            _text(getattr(self, label), label)
        _record_tuple(self.repetitions, HistoricalRepetitionSpecV1, "repetitions")
        if len(self.repetitions) < 3:
            raise ValueError("historical verification requires at least three repetitions")
        ids = tuple(item.repetition_id for item in self.repetitions)
        seeds = tuple(item.continuation_seed for item in self.repetitions)
        if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
            raise ValueError("repetition IDs and continuation seeds must be unique")
        if any(
            {item.arm_order[position] for item in self.repetitions}
            != _ALL_ARM_ROLES
            for position in range(3)
        ):
            raise ValueError("arm order must rotate every role through every position")
        if type(self.all_repetitions_required) is not bool or not self.all_repetitions_required:
            raise ValueError("v0.1 strict reducer requires every planned repetition")
        if self.required_pattern is not HistoricalOutcomePattern.REPLACEMENT_ONLY_SUCCESS:
            raise ValueError("the strict reducer requires replacement-only raw success")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "repetition_contract_id": self.repetition_contract_id,
            "repetition_contract_version": self.repetition_contract_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "repetitions": [item.to_dict() for item in self.repetitions],
            "continuation_seed_rule_id": self.continuation_seed_rule_id,
            "continuation_seed_rule_source_ref": self.continuation_seed_rule_source_ref,
            "root_identity_matching_rule_id": self.root_identity_matching_rule_id,
            "root_identity_matching_rule_source_ref": self.root_identity_matching_rule_source_ref,
            "strict_reducer_id": self.strict_reducer_id,
            "strict_reducer_version": self.strict_reducer_version,
            "strict_reducer_source_ref": self.strict_reducer_source_ref,
            "all_repetitions_required": self.all_repetitions_required,
            "required_pattern": self.required_pattern.value,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class HistoricalOfficialOutcomeV1:
    """Preserve one arm's benchmark-owned final result and availability evidence."""

    run_id: str
    work_item_id: str
    action_id: str
    case_id: str
    candidate_id: str
    repetition_id: str
    arm: HistoricalArmRole
    verifier_contract_id: str
    availability: HistoricalValueAvailability
    success: bool | None
    raw_output_ref: HistoricalEvidenceRefV1 | None
    terminal_state_id: str | None
    verifier_receipt_ref: HistoricalEvidenceRefV1 | None
    availability_evidence_ref: HistoricalEvidenceRefV1
    schema: str = HISTORICAL_OFFICIAL_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_OFFICIAL_OUTCOME_SCHEMA:
            raise ValueError("unsupported historical official outcome schema")
        for label in (
            "run_id",
            "work_item_id",
            "action_id",
            "case_id",
            "candidate_id",
            "repetition_id",
            "verifier_contract_id",
        ):
            _identifier(getattr(self, label), label)
        if not isinstance(self.arm, HistoricalArmRole) or not isinstance(
            self.availability, HistoricalValueAvailability
        ):
            raise ValueError("official outcome arm and availability must use closed enums")
        _require_arm_evidence(
            self.availability_evidence_ref,
            kind=HistoricalEvidenceKind.OFFICIAL_AVAILABILITY,
            run_id=self.run_id,
            work_item_id=self.work_item_id,
            action_id=self.action_id,
            case_id=self.case_id,
            candidate_id=self.candidate_id,
            repetition_id=self.repetition_id,
            arm=self.arm,
            allow_candidate_shared=(
                self.availability is HistoricalValueAvailability.UNAVAILABLE
            ),
        )
        if self.availability is HistoricalValueAvailability.AVAILABLE:
            if type(self.success) is not bool or self.raw_output_ref is None or self.verifier_receipt_ref is None:
                raise ValueError("available official outcome requires boolean and raw artifacts")
            _text(self.terminal_state_id, "terminal_state_id")
            _require_arm_evidence(
                self.raw_output_ref,
                kind=HistoricalEvidenceKind.OFFICIAL_RAW_OUTPUT,
                run_id=self.run_id,
                work_item_id=self.work_item_id,
                action_id=self.action_id,
                case_id=self.case_id,
                candidate_id=self.candidate_id,
                repetition_id=self.repetition_id,
                arm=self.arm,
            )
            _require_arm_evidence(
                self.verifier_receipt_ref,
                kind=HistoricalEvidenceKind.OFFICIAL_VERIFIER,
                run_id=self.run_id,
                work_item_id=self.work_item_id,
                action_id=self.action_id,
                case_id=self.case_id,
                candidate_id=self.candidate_id,
                repetition_id=self.repetition_id,
                arm=self.arm,
            )
        elif any(
            value is not None
            for value in (
                self.success,
                self.raw_output_ref,
                self.terminal_state_id,
                self.verifier_receipt_ref,
            )
        ):
            raise ValueError("unavailable official outcome cannot carry result values")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "action_id": self.action_id,
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "repetition_id": self.repetition_id,
            "arm": self.arm.value,
            "verifier_contract_id": self.verifier_contract_id,
            "availability": self.availability.value,
            "success": self.success,
            "raw_output_ref": self.raw_output_ref.to_dict() if self.raw_output_ref else None,
            "terminal_state_id": self.terminal_state_id,
            "verifier_receipt_ref": (
                self.verifier_receipt_ref.to_dict() if self.verifier_receipt_ref else None
            ),
            "availability_evidence_ref": self.availability_evidence_ref.to_dict(),
        }


@dataclass(frozen=True)
class HistoricalProgressObservationV1:
    """Preserve the official progress value at the frozen terminal timepoint."""

    run_id: str
    work_item_id: str
    action_id: str
    case_id: str
    candidate_id: str
    repetition_id: str
    arm: HistoricalArmRole
    terminal_state_id: str | None
    evaluator_contract_id: str
    progress_contract_id: str
    timepoint: str
    availability: HistoricalValueAvailability
    value: float | None
    evaluator_receipt_ref: HistoricalEvidenceRefV1 | None
    availability_evidence_ref: HistoricalEvidenceRefV1
    schema: str = HISTORICAL_PROGRESS_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_PROGRESS_OBSERVATION_SCHEMA:
            raise ValueError("unsupported historical progress observation schema")
        for label in (
            "run_id",
            "work_item_id",
            "action_id",
            "case_id",
            "candidate_id",
            "repetition_id",
            "evaluator_contract_id",
            "progress_contract_id",
            "timepoint",
        ):
            _identifier(getattr(self, label), label)
        if not isinstance(self.arm, HistoricalArmRole) or not isinstance(
            self.availability, HistoricalValueAvailability
        ):
            raise ValueError("progress arm and availability must use closed enums")
        _require_arm_evidence(
            self.availability_evidence_ref,
            kind=HistoricalEvidenceKind.PROGRESS_AVAILABILITY,
            run_id=self.run_id,
            work_item_id=self.work_item_id,
            action_id=self.action_id,
            case_id=self.case_id,
            candidate_id=self.candidate_id,
            repetition_id=self.repetition_id,
            arm=self.arm,
            allow_candidate_shared=(
                self.availability is HistoricalValueAvailability.UNAVAILABLE
            ),
        )
        if self.availability is HistoricalValueAvailability.AVAILABLE:
            _finite(self.value, "progress value")
            _text(self.terminal_state_id, "terminal_state_id")
            if self.evaluator_receipt_ref is None:
                raise ValueError("available progress requires evaluator evidence")
            _require_arm_evidence(
                self.evaluator_receipt_ref,
                kind=HistoricalEvidenceKind.PROGRESS_EVALUATOR,
                run_id=self.run_id,
                work_item_id=self.work_item_id,
                action_id=self.action_id,
                case_id=self.case_id,
                candidate_id=self.candidate_id,
                repetition_id=self.repetition_id,
                arm=self.arm,
            )
        elif any(
            value is not None
            for value in (self.value, self.terminal_state_id, self.evaluator_receipt_ref)
        ):
            raise ValueError("unavailable progress cannot carry result values")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "action_id": self.action_id,
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "repetition_id": self.repetition_id,
            "arm": self.arm.value,
            "terminal_state_id": self.terminal_state_id,
            "evaluator_contract_id": self.evaluator_contract_id,
            "progress_contract_id": self.progress_contract_id,
            "timepoint": self.timepoint,
            "availability": self.availability.value,
            "value": self.value,
            "evaluator_receipt_ref": (
                self.evaluator_receipt_ref.to_dict() if self.evaluator_receipt_ref else None
            ),
            "availability_evidence_ref": self.availability_evidence_ref.to_dict(),
        }


@dataclass(frozen=True)
class HistoricalShamFieldMatchV1:
    """Record one closed sham field relation resolved from typed arm data."""

    field: HistoricalShamMatchField
    relation: HistoricalShamFieldRelation
    evidence_ref: HistoricalEvidenceRefV1
    schema: str = HISTORICAL_SHAM_FIELD_MATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_SHAM_FIELD_MATCH_SCHEMA:
            raise ValueError("unsupported historical sham field match schema")
        if not isinstance(self.field, HistoricalShamMatchField) or not isinstance(
            self.relation, HistoricalShamFieldRelation
        ):
            raise ValueError("sham field and relation must use closed enums")
        if type(self.evidence_ref) is not HistoricalEvidenceRefV1:
            raise ValueError("sham field relation requires typed evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "field": self.field.value,
            "relation": self.relation.value,
            "evidence_ref": self.evidence_ref.to_dict(),
        }


@dataclass(frozen=True)
class HistoricalShamMatcherDecisionV1:
    """Bind a typed matcher decision to exact replacement and sham plans."""

    run_id: str
    work_item_id: str
    action_id: str
    case_id: str
    candidate_id: str
    repetition_id: str
    replacement_intervention_id: str
    sham_intervention_id: str
    matcher_contract_id: str | None
    status: HistoricalParityStatus
    field_matches: tuple[HistoricalShamFieldMatchV1, ...]
    evidence_ref: HistoricalEvidenceRefV1
    schema: str = HISTORICAL_SHAM_MATCHER_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_SHAM_MATCHER_DECISION_SCHEMA:
            raise ValueError("unsupported historical sham matcher decision schema")
        for label in (
            "run_id",
            "work_item_id",
            "action_id",
            "case_id",
            "candidate_id",
            "repetition_id",
            "replacement_intervention_id",
            "sham_intervention_id",
        ):
            _identifier(getattr(self, label), label)
        if not isinstance(self.status, HistoricalParityStatus):
            raise ValueError("sham matcher status must use the closed parity enum")
        _record_tuple(self.field_matches, HistoricalShamFieldMatchV1, "field_matches")
        _require_candidate_evidence(
            self.evidence_ref,
            kind=HistoricalEvidenceKind.SHAM_MATCHER,
            run_id=self.run_id,
            work_item_id=self.work_item_id,
            action_id=self.action_id,
            case_id=self.case_id,
            candidate_id=self.candidate_id,
            repetition_id=self.repetition_id,
        )
        if any(match.evidence_ref != self.evidence_ref for match in self.field_matches):
            raise ValueError("sham field relations must share the matcher artifact")
        fields = tuple(match.field for match in self.field_matches)
        if len(set(fields)) != len(fields):
            raise ValueError("sham matcher cannot repeat a closed field")
        if self.matcher_contract_id is None:
            if self.status is not HistoricalParityStatus.UNAVAILABLE or self.field_matches:
                raise ValueError("missing typed matcher must remain unavailable")
        else:
            _identifier(self.matcher_contract_id, "matcher_contract_id")
            if self.status is HistoricalParityStatus.UNAVAILABLE or not self.field_matches:
                raise ValueError("typed matcher decision requires field relations and a verdict")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "action_id": self.action_id,
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "repetition_id": self.repetition_id,
            "replacement_intervention_id": self.replacement_intervention_id,
            "sham_intervention_id": self.sham_intervention_id,
            "matcher_contract_id": self.matcher_contract_id,
            "status": self.status.value,
            "field_matches": [match.to_dict() for match in self.field_matches],
            "evidence_ref": self.evidence_ref.to_dict(),
        }


@dataclass(frozen=True)
class HistoricalParityDecisionV1:
    """Bind one parity decision to raw identities and candidate-owned evidence."""

    dimension: HistoricalParityDimension
    status: HistoricalParityStatus
    original_identity: str
    replacement_identity: str
    sham_identity: str
    evidence_ref: HistoricalEvidenceRefV1
    schema: str = HISTORICAL_PARITY_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_PARITY_DECISION_SCHEMA:
            raise ValueError("unsupported historical parity decision schema")
        if not isinstance(self.dimension, HistoricalParityDimension) or not isinstance(
            self.status, HistoricalParityStatus
        ):
            raise ValueError("parity dimension and status must use closed enums")
        for label in ("original_identity", "replacement_identity", "sham_identity"):
            _text(getattr(self, label), label)
        if type(self.evidence_ref) is not HistoricalEvidenceRefV1:
            raise ValueError("parity decision requires typed evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dimension": self.dimension.value,
            "status": self.status.value,
            "original_identity": self.original_identity,
            "replacement_identity": self.replacement_identity,
            "sham_identity": self.sham_identity,
            "evidence_ref": self.evidence_ref.to_dict(),
        }


@dataclass(frozen=True)
class HistoricalCandidateComparisonReceiptV1:
    """Own candidate-shared parity and typed sham matcher evidence."""

    run_id: str
    work_item_id: str
    action_id: str
    case_id: str
    candidate_id: str
    repetition_id: str
    arm_roles: tuple[HistoricalArmRole, ...]
    parity_evidence_ref: HistoricalEvidenceRefV1
    sham_matcher_decision: HistoricalShamMatcherDecisionV1
    schema: str = HISTORICAL_CANDIDATE_COMPARISON_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_CANDIDATE_COMPARISON_RECEIPT_SCHEMA:
            raise ValueError("unsupported historical comparison receipt schema")
        for label in (
            "run_id",
            "work_item_id",
            "action_id",
            "case_id",
            "candidate_id",
            "repetition_id",
        ):
            _identifier(getattr(self, label), label)
        if (
            type(self.arm_roles) is not tuple
            or any(type(role) is not HistoricalArmRole for role in self.arm_roles)
            or self.arm_roles != tuple(HistoricalArmRole)
        ):
            raise ValueError("comparison arm_roles must be the exact ordered O/R/S tuple")
        _require_candidate_evidence(
            self.parity_evidence_ref,
            kind=HistoricalEvidenceKind.PARITY,
            run_id=self.run_id,
            work_item_id=self.work_item_id,
            action_id=self.action_id,
            case_id=self.case_id,
            candidate_id=self.candidate_id,
            repetition_id=self.repetition_id,
        )
        if type(self.sham_matcher_decision) is not HistoricalShamMatcherDecisionV1:
            raise ValueError("comparison receipt requires a typed sham matcher decision")
        matcher = self.sham_matcher_decision
        if (
            matcher.run_id,
            matcher.work_item_id,
            matcher.action_id,
            matcher.case_id,
            matcher.candidate_id,
            matcher.repetition_id,
        ) != (
            self.run_id,
            self.work_item_id,
            self.action_id,
            self.case_id,
            self.candidate_id,
            self.repetition_id,
        ):
            raise ValueError("sham matcher owner tuple drifted from comparison receipt")

    @property
    def sham_matching_evidence_ref(self) -> HistoricalEvidenceRefV1:
        return self.sham_matcher_decision.evidence_ref

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "action_id": self.action_id,
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "repetition_id": self.repetition_id,
            "arm_roles": [role.value for role in self.arm_roles],
            "parity_evidence_ref": self.parity_evidence_ref.to_dict(),
            "sham_matcher_decision": self.sham_matcher_decision.to_dict(),
        }


@dataclass(frozen=True)
class HistoricalArmReceiptV1:
    """Record one replay arm from restore through official consumption."""

    run_id: str
    work_item_id: str
    arm: HistoricalArmRole
    execution_status: HistoricalArmExecutionStatus
    case_id: str
    candidate_id: str
    repetition_id: str
    intervention: HistoricalInterventionExecutionV1
    pre_action_state_id: str
    restore_protocol_id: str
    restore_receipt_ref: HistoricalEvidenceRefV1
    restored_state_id: str
    actor_id: str
    runtime_id: str
    environment_id: str
    adapter_version: str
    system_cell_id: str
    verifier_protocol_id: str
    budget_id: str
    isolation_id: str
    sham_contract_id: str
    sham_contract_version: str
    continuation_seed: int
    terminal_state_id: str | None
    terminal_receipt_ref: HistoricalEvidenceRefV1 | None
    official_outcome: HistoricalOfficialOutcomeV1
    progress: HistoricalProgressObservationV1
    costs: tuple[ResourceCostV1, ...]
    action_ledger_refs: tuple[HistoricalEvidenceRefV1, ...]
    schema: str = HISTORICAL_ARM_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_ARM_RECEIPT_SCHEMA:
            raise ValueError("unsupported historical arm receipt schema")
        if not isinstance(self.arm, HistoricalArmRole) or not isinstance(
            self.execution_status, HistoricalArmExecutionStatus
        ):
            raise ValueError("arm role and execution status must use closed enums")
        for label in (
            "run_id",
            "work_item_id",
            "case_id",
            "candidate_id",
            "repetition_id",
            "restore_protocol_id",
            "adapter_version",
            "system_cell_id",
            "verifier_protocol_id",
            "budget_id",
            "isolation_id",
            "sham_contract_id",
            "sham_contract_version",
        ):
            _identifier(getattr(self, label), label)
        for label in (
            "pre_action_state_id",
            "restored_state_id",
            "actor_id",
            "runtime_id",
            "environment_id",
        ):
            _text(getattr(self, label), label)
        if type(self.continuation_seed) is not int or self.continuation_seed < 0:
            raise ValueError("continuation_seed must be a nonnegative integer")
        if type(self.intervention) is not HistoricalInterventionExecutionV1:
            raise ValueError("arm receipt requires a typed intervention execution")
        plan = self.intervention.plan
        if (
            plan.run_id,
            plan.work_item_id,
            plan.case_id,
            plan.candidate_id,
            plan.repetition_id,
            plan.arm,
        ) != (
            self.run_id,
            self.work_item_id,
            self.case_id,
            self.candidate_id,
            self.repetition_id,
            self.arm,
        ):
            raise ValueError("intervention execution owner drifted from its arm")
        _require_arm_evidence(
            self.restore_receipt_ref,
            kind=HistoricalEvidenceKind.RESTORE,
            run_id=self.run_id,
            work_item_id=self.work_item_id,
            action_id=plan.action_id,
            case_id=self.case_id,
            candidate_id=self.candidate_id,
            repetition_id=self.repetition_id,
            arm=self.arm,
        )
        if type(self.official_outcome) is not HistoricalOfficialOutcomeV1 or type(
            self.progress
        ) is not HistoricalProgressObservationV1:
            raise ValueError("arm receipt requires typed official and progress observations")
        if (
            self.official_outcome.run_id,
            self.official_outcome.work_item_id,
            self.official_outcome.case_id,
            self.official_outcome.candidate_id,
            self.official_outcome.repetition_id,
            self.official_outcome.arm,
        ) != (
            self.run_id,
            self.work_item_id,
            self.case_id,
            self.candidate_id,
            self.repetition_id,
            self.arm,
        ) or (
            self.progress.run_id,
            self.progress.work_item_id,
            self.progress.case_id,
            self.progress.candidate_id,
            self.progress.repetition_id,
            self.progress.arm,
        ) != (
            self.run_id,
            self.work_item_id,
            self.case_id,
            self.candidate_id,
            self.repetition_id,
            self.arm,
        ):
            raise ValueError("official or progress observation owner drifted from its arm")
        _cost_tuple(self.costs, "arm costs", require_nonempty=True)
        _record_tuple(
            self.action_ledger_refs,
            HistoricalEvidenceRefV1,
            "action_ledger_refs",
        )
        if not self.action_ledger_refs:
            raise ValueError("arm receipt requires action ledger evidence")
        for reference in self.action_ledger_refs:
            _require_arm_evidence(
                reference,
                kind=HistoricalEvidenceKind.ACTION_LEDGER,
                run_id=self.run_id,
                work_item_id=self.work_item_id,
                action_id=plan.action_id,
                case_id=self.case_id,
                candidate_id=self.candidate_id,
                repetition_id=self.repetition_id,
                arm=self.arm,
            )
        if len({ref.artifact_ref for ref in self.action_ledger_refs}) != len(
            self.action_ledger_refs
        ):
            raise ValueError("action ledger evidence cannot repeat artifacts")
        if self.execution_status is HistoricalArmExecutionStatus.COMPLETED:
            if self.intervention.controlled_action_count != len(plan.controlled_actions):
                raise ValueError("completed arm requires the full controlled action plan")
            _text(self.terminal_state_id, "terminal_state_id")
            _text(self.intervention.post_intervention_state_id, "post_intervention_state_id")
            if self.terminal_receipt_ref is None:
                raise ValueError("completed arm requires terminal-state evidence")
            _require_arm_evidence(
                self.terminal_receipt_ref,
                kind=HistoricalEvidenceKind.TERMINAL_STATE,
                run_id=self.run_id,
                work_item_id=self.work_item_id,
                action_id=plan.action_id,
                case_id=self.case_id,
                candidate_id=self.candidate_id,
                repetition_id=self.repetition_id,
                arm=self.arm,
            )
            for observation_state in (
                self.official_outcome.terminal_state_id,
                self.progress.terminal_state_id,
            ):
                if observation_state is not None and observation_state != self.terminal_state_id:
                    raise ValueError("official observation belongs to another terminal state")
        else:
            _optional_text(self.terminal_state_id, "terminal_state_id")
            if self.terminal_receipt_ref is not None:
                _require_arm_evidence(
                    self.terminal_receipt_ref,
                    kind=HistoricalEvidenceKind.TERMINAL_STATE,
                    run_id=self.run_id,
                    work_item_id=self.work_item_id,
                    action_id=plan.action_id,
                    case_id=self.case_id,
                    candidate_id=self.candidate_id,
                    repetition_id=self.repetition_id,
                    arm=self.arm,
                )

    @property
    def intervention_id(self) -> str:
        return self.intervention.plan.intervention_id

    @property
    def intervention_receipt_ref(self) -> HistoricalEvidenceRefV1:
        return self.intervention.execution_evidence_ref

    @property
    def action_protocol_id(self) -> str:
        return self.intervention.plan.action_protocol_id

    @property
    def controlled_actions(self) -> tuple[str, ...]:
        return self.intervention.plan.controlled_actions

    @property
    def actor_actions(self) -> tuple[str, ...]:
        return self.intervention.actor_actions

    @property
    def executed_actions(self) -> tuple[str, ...]:
        return self.intervention.executed_actions

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "arm": self.arm.value,
            "execution_status": self.execution_status.value,
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "repetition_id": self.repetition_id,
            "intervention": self.intervention.to_dict(),
            "pre_action_state_id": self.pre_action_state_id,
            "restore_protocol_id": self.restore_protocol_id,
            "restore_receipt_ref": self.restore_receipt_ref.to_dict(),
            "restored_state_id": self.restored_state_id,
            "actor_id": self.actor_id,
            "runtime_id": self.runtime_id,
            "environment_id": self.environment_id,
            "adapter_version": self.adapter_version,
            "system_cell_id": self.system_cell_id,
            "verifier_protocol_id": self.verifier_protocol_id,
            "budget_id": self.budget_id,
            "isolation_id": self.isolation_id,
            "sham_contract_id": self.sham_contract_id,
            "sham_contract_version": self.sham_contract_version,
            "continuation_seed": self.continuation_seed,
            "terminal_state_id": self.terminal_state_id,
            "terminal_receipt_ref": (
                self.terminal_receipt_ref.to_dict() if self.terminal_receipt_ref else None
            ),
            "official_outcome": self.official_outcome.to_dict(),
            "progress": self.progress.to_dict(),
            "costs": [cost.to_dict() for cost in self.costs],
            "action_ledger_refs": [ref.to_dict() for ref in self.action_ledger_refs],
        }


def _append_evidence(
    ordered: list[HistoricalEvidenceRefV1],
    by_artifact: dict[str, HistoricalEvidenceRefV1],
    reference: HistoricalEvidenceRefV1 | None,
) -> None:
    """Build a unique closure while rejecting one artifact with conflicting owners."""

    if reference is None:
        return
    previous = by_artifact.get(reference.artifact_ref)
    if previous is not None:
        if previous != reference:
            raise ValueError(
                "one evidence artifact cannot cross run, work-item, action, or typed owners"
            )
        return
    by_artifact[reference.artifact_ref] = reference
    ordered.append(reference)


def _evidence_closure(
    arms: tuple[HistoricalArmReceiptV1, ...],
    parity: tuple[HistoricalParityDecisionV1, ...],
    comparison: HistoricalCandidateComparisonReceiptV1 | None,
) -> tuple[HistoricalEvidenceRefV1, ...]:
    """Return the exact ordered closure of every nested historical artifact."""

    ordered: list[HistoricalEvidenceRefV1] = []
    by_artifact: dict[str, HistoricalEvidenceRefV1] = {}
    for arm in arms:
        official = arm.official_outcome
        progress = arm.progress
        for reference in (
            arm.intervention.execution_evidence_ref,
            arm.restore_receipt_ref,
            arm.terminal_receipt_ref,
            official.raw_output_ref,
            official.verifier_receipt_ref,
            official.availability_evidence_ref,
            progress.evaluator_receipt_ref,
            progress.availability_evidence_ref,
            *arm.action_ledger_refs,
        ):
            _append_evidence(ordered, by_artifact, reference)
    for decision in parity:
        _append_evidence(ordered, by_artifact, decision.evidence_ref)
    if comparison is not None:
        matcher = comparison.sham_matcher_decision
        _append_evidence(ordered, by_artifact, comparison.parity_evidence_ref)
        _append_evidence(ordered, by_artifact, matcher.evidence_ref)
        for match in matcher.field_matches:
            _append_evidence(ordered, by_artifact, match.evidence_ref)
    return tuple(ordered)


@dataclass(frozen=True)
class HistoricalThreeArmReceiptV1:
    """Keep candidate-local O/R/S evidence neutral and structurally closed."""

    suite_id: str
    suite_version: str
    population_id: str
    registry_id: str
    scorer_id: str
    scorer_version: str
    run_id: str
    work_item_id: str
    source_task_id: str
    case_id: str
    track: TraceFixTrack
    execution_semantics: ExecutionSemantics
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    system_cell_id: str
    checkpoint_id: str
    actor_id: str
    runtime_id: str
    environment_id: str
    locator_treatment_id: str
    fixer_treatment_id: str
    budget_id: str
    repetition_contract_id: str
    repetition_contract_version: str
    repetition_id: str
    candidate_id: str
    candidate_rank: int
    candidate_event_group: tuple[str, ...]
    original_intervention_id: str
    replacement_intervention_id: str
    sham_intervention_id: str
    sham_contract_id: str
    sham_contract_version: str
    candidate_status: HistoricalCandidateStatus
    not_executed_reason: HistoricalCandidateNotExecutedReason | None
    actual_arm_order: tuple[HistoricalArmRole, ...]
    arms: tuple[HistoricalArmReceiptV1, ...]
    parity: tuple[HistoricalParityDecisionV1, ...]
    comparison_receipt: HistoricalCandidateComparisonReceiptV1 | None
    incremental_costs: tuple[ResourceCostV1, ...]
    cumulative_costs: tuple[ResourceCostV1, ...]
    raw_evidence_refs: tuple[HistoricalEvidenceRefV1, ...]
    implementation_status: str = HISTORICAL_IMPLEMENTATION_STATUS
    input_binding_status: str = HISTORICAL_INPUT_BINDING_STATUS
    schema: str = HISTORICAL_THREE_ARM_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_THREE_ARM_RECEIPT_SCHEMA:
            raise ValueError("unsupported historical three-arm receipt schema")
        if (
            self.suite_id != TRACEFIX_BENCH_SUITE_ID
            or self.suite_version != SCORER_REGISTRY_SUITE_VERSION_V0_1
        ):
            raise ValueError("historical receipt must use the frozen v0.1 suite")
        for label in (
            "population_id",
            "registry_id",
            "scorer_id",
            "scorer_version",
            "run_id",
            "work_item_id",
            "case_id",
            "benchmark_id",
            "benchmark_version",
            "adapter_version",
            "system_cell_id",
            "locator_treatment_id",
            "fixer_treatment_id",
            "budget_id",
            "repetition_contract_id",
            "repetition_contract_version",
            "repetition_id",
            "candidate_id",
            "original_intervention_id",
            "replacement_intervention_id",
            "sham_intervention_id",
            "sham_contract_id",
            "sham_contract_version",
        ):
            _identifier(getattr(self, label), label)
        for label in (
            "source_task_id",
            "checkpoint_id",
            "actor_id",
            "runtime_id",
            "environment_id",
        ):
            _text(getattr(self, label), label)
        if self.track is not TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION:
            raise ValueError("historical receipt must use the historical track")
        if self.execution_semantics is not ExecutionSemantics.EXACT_CHECKPOINT_REPLAY:
            raise ValueError("historical receipt requires exact checkpoint replay")
        if type(self.candidate_rank) is not int or self.candidate_rank < 1:
            raise ValueError("candidate_rank must be a positive integer")
        _text_tuple(
            self.candidate_event_group,
            "candidate_event_group",
            unique=True,
            nonempty=True,
        )
        if not isinstance(self.candidate_status, HistoricalCandidateStatus):
            raise ValueError("candidate_status must use the closed enum")
        if self.not_executed_reason is not None and not isinstance(
            self.not_executed_reason, HistoricalCandidateNotExecutedReason
        ):
            raise ValueError("not_executed_reason must use the closed enum")
        _record_tuple(self.arms, HistoricalArmReceiptV1, "arms")
        _record_tuple(self.parity, HistoricalParityDecisionV1, "parity")
        if self.comparison_receipt is not None and type(
            self.comparison_receipt
        ) is not HistoricalCandidateComparisonReceiptV1:
            raise ValueError("comparison_receipt must use the typed record")
        if type(self.actual_arm_order) is not tuple or any(
            not isinstance(role, HistoricalArmRole) for role in self.actual_arm_order
        ):
            raise ValueError("actual_arm_order must use closed arm roles")
        started = self.candidate_status is not HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED
        _cost_tuple(self.incremental_costs, "incremental_costs", require_nonempty=started)
        _cost_tuple(self.cumulative_costs, "cumulative_costs", require_nonempty=started)
        _record_tuple(
            self.raw_evidence_refs,
            HistoricalEvidenceRefV1,
            "raw_evidence_refs",
        )
        if any(
            (reference.run_id, reference.work_item_id)
            != (self.run_id, self.work_item_id)
            for reference in self.raw_evidence_refs
        ):
            raise ValueError("historical evidence drifted across run or work item")
        if self.candidate_status is HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED:
            if self.not_executed_reason is None:
                raise ValueError("unexecuted candidate requires a closed reason")
            if any(
                (
                    self.arms,
                    self.parity,
                    self.actual_arm_order,
                    self.incremental_costs,
                    self.cumulative_costs,
                    self.raw_evidence_refs,
                )
            ) or self.comparison_receipt is not None:
                raise ValueError("unexecuted candidate cannot claim replay evidence")
        elif self.not_executed_reason is not None:
            raise ValueError("started candidate cannot carry a normal stop reason")
        roles = tuple(arm.arm for arm in self.arms)
        if len(set(roles)) != len(roles):
            raise ValueError("candidate cannot repeat an arm role")
        if roles != self.actual_arm_order:
            raise ValueError("arm tuple must preserve actual execution order")
        if self.candidate_status is HistoricalCandidateStatus.EXECUTED:
            dimensions = tuple(decision.dimension for decision in self.parity)
            if (
                len(roles) != 3
                or set(roles) != _ALL_ARM_ROLES
                or any(
                    arm.execution_status is not HistoricalArmExecutionStatus.COMPLETED
                    for arm in self.arms
                )
            ):
                raise ValueError("executed candidate requires three completed arms")
            if (
                len(dimensions) != len(_ALL_PARITY_DIMENSIONS)
                or set(dimensions) != _ALL_PARITY_DIMENSIONS
                or len(set(dimensions)) != len(dimensions)
            ):
                raise ValueError("executed candidate requires every parity dimension once")
            if self.comparison_receipt is None:
                raise ValueError("executed candidate requires typed comparison evidence")
        arm_by_role = {arm.arm: arm for arm in self.arms}
        expected_interventions = {
            HistoricalArmRole.ORIGINAL: self.original_intervention_id,
            HistoricalArmRole.REPLACEMENT: self.replacement_intervention_id,
            HistoricalArmRole.MATCHED_SHAM: self.sham_intervention_id,
        }
        if len(set(expected_interventions.values())) != 3:
            raise ValueError("O, R, and S intervention identities must be distinct")
        for role, arm in arm_by_role.items():
            plan = arm.intervention.plan
            if (
                arm.run_id,
                arm.work_item_id,
                arm.case_id,
                arm.candidate_id,
                arm.repetition_id,
                arm.intervention_id,
                plan.candidate_event_group,
                plan.target_event_group,
            ) != (
                self.run_id,
                self.work_item_id,
                self.case_id,
                self.candidate_id,
                self.repetition_id,
                expected_interventions[role],
                self.candidate_event_group,
                self.candidate_event_group,
            ):
                raise ValueError("typed intervention target drifted from its candidate")
            if (
                arm.budget_id != self.budget_id
                or arm.sham_contract_id != self.sham_contract_id
                or arm.sham_contract_version != self.sham_contract_version
            ):
                raise ValueError("arm budget or sham identity drifted from candidate")
        if self.comparison_receipt is not None:
            comparison = self.comparison_receipt
            if (
                comparison.run_id,
                comparison.work_item_id,
                comparison.case_id,
                comparison.candidate_id,
                comparison.repetition_id,
                comparison.arm_roles,
            ) != (
                self.run_id,
                self.work_item_id,
                self.case_id,
                self.candidate_id,
                self.repetition_id,
                tuple(HistoricalArmRole),
            ) or set(arm_by_role) != _ALL_ARM_ROLES:
                raise ValueError("comparison evidence owner drifted from candidate")
            matcher = comparison.sham_matcher_decision
            if (
                matcher.replacement_intervention_id != self.replacement_intervention_id
                or matcher.sham_intervention_id != self.sham_intervention_id
            ):
                raise ValueError("sham matcher intervention identity drifted")
        for decision in self.parity:
            expected_ref = (
                self.comparison_receipt.sham_matching_evidence_ref
                if self.comparison_receipt is not None
                and decision.dimension is HistoricalParityDimension.SHAM_MATCHING
                else self.comparison_receipt.parity_evidence_ref
                if self.comparison_receipt is not None
                else None
            )
            if expected_ref is None or decision.evidence_ref != expected_ref:
                raise ValueError("parity decision evidence drifted from comparison receipt")
        expected_closure = _evidence_closure(
            self.arms,
            self.parity,
            self.comparison_receipt,
        )
        if self.raw_evidence_refs != expected_closure:
            raise ValueError("raw_evidence_refs must equal the exact nested evidence closure")
        _unverified(self.implementation_status)
        if self.input_binding_status != HISTORICAL_INPUT_BINDING_STATUS:
            raise ValueError("candidate receipt remains unbound to a formal run")

    @property
    def sham_field_matches(self) -> tuple[HistoricalShamFieldMatchV1, ...]:
        if self.comparison_receipt is None:
            return ()
        return self.comparison_receipt.sham_matcher_decision.field_matches

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "implementation_status": self.implementation_status,
            "input_binding_status": self.input_binding_status,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "registry_id": self.registry_id,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "source_task_id": self.source_task_id,
            "case_id": self.case_id,
            "track": self.track.value,
            "execution_semantics": self.execution_semantics.value,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "system_cell_id": self.system_cell_id,
            "checkpoint_id": self.checkpoint_id,
            "actor_id": self.actor_id,
            "runtime_id": self.runtime_id,
            "environment_id": self.environment_id,
            "locator_treatment_id": self.locator_treatment_id,
            "fixer_treatment_id": self.fixer_treatment_id,
            "budget_id": self.budget_id,
            "repetition_contract_id": self.repetition_contract_id,
            "repetition_contract_version": self.repetition_contract_version,
            "repetition_id": self.repetition_id,
            "candidate_id": self.candidate_id,
            "candidate_rank": self.candidate_rank,
            "candidate_event_group": list(self.candidate_event_group),
            "original_intervention_id": self.original_intervention_id,
            "replacement_intervention_id": self.replacement_intervention_id,
            "sham_intervention_id": self.sham_intervention_id,
            "sham_contract_id": self.sham_contract_id,
            "sham_contract_version": self.sham_contract_version,
            "candidate_status": self.candidate_status.value,
            "not_executed_reason": (
                self.not_executed_reason.value if self.not_executed_reason else None
            ),
            "actual_arm_order": [role.value for role in self.actual_arm_order],
            "arms": [arm.to_dict() for arm in self.arms],
            "parity": [decision.to_dict() for decision in self.parity],
            "comparison_receipt": (
                self.comparison_receipt.to_dict() if self.comparison_receipt else None
            ),
            "incremental_costs": [cost.to_dict() for cost in self.incremental_costs],
            "cumulative_costs": [cost.to_dict() for cost in self.cumulative_costs],
            "raw_evidence_refs": [ref.to_dict() for ref in self.raw_evidence_refs],
        }


@dataclass(frozen=True)
class HistoricalProgressEffectsV1:
    """Preserve raw, direction-normalized, and thresholded O/R/S contrasts."""

    original: float
    replacement: float
    matched_sham: float
    direction: ProgressDirection
    tolerance: float
    minimum_meaningful_difference: float
    raw_replacement_minus_original: float
    raw_sham_minus_original: float
    raw_replacement_minus_sham: float
    directed_replacement_minus_original: float
    directed_sham_minus_original: float
    directed_replacement_minus_sham: float
    meaningful_replacement_minus_original: bool
    meaningful_sham_minus_original: bool
    meaningful_replacement_minus_sham: bool
    schema: str = HISTORICAL_PROGRESS_EFFECTS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_PROGRESS_EFFECTS_SCHEMA:
            raise ValueError("unsupported historical progress effects schema")
        if not isinstance(self.direction, ProgressDirection):
            raise ValueError("progress direction must use the closed enum")
        for label in (
            "original",
            "replacement",
            "matched_sham",
            "tolerance",
            "minimum_meaningful_difference",
            "raw_replacement_minus_original",
            "raw_sham_minus_original",
            "raw_replacement_minus_sham",
            "directed_replacement_minus_original",
            "directed_sham_minus_original",
            "directed_replacement_minus_sham",
        ):
            _finite(getattr(self, label), label)
        if self.tolerance < 0 or self.minimum_meaningful_difference <= self.tolerance:
            raise ValueError("meaningful threshold must exceed nonnegative tolerance")
        expected_raw = (
            self.replacement - self.original,
            self.matched_sham - self.original,
            self.replacement - self.matched_sham,
        )
        actual_raw = (
            self.raw_replacement_minus_original,
            self.raw_sham_minus_original,
            self.raw_replacement_minus_sham,
        )
        if actual_raw != expected_raw:
            raise ValueError("raw effects must be recomputed from O, R, and S")
        sign = 1.0 if self.direction is ProgressDirection.HIGHER_IS_BETTER else -1.0
        expected_directed = tuple(sign * value for value in expected_raw)
        actual_directed = (
            self.directed_replacement_minus_original,
            self.directed_sham_minus_original,
            self.directed_replacement_minus_sham,
        )
        if actual_directed != expected_directed:
            raise ValueError("directed effects must follow the registered direction")
        expected_meaningful = tuple(
            value >= self.minimum_meaningful_difference for value in expected_directed
        )
        actual_meaningful = (
            self.meaningful_replacement_minus_original,
            self.meaningful_sham_minus_original,
            self.meaningful_replacement_minus_sham,
        )
        if any(type(value) is not bool for value in actual_meaningful):
            raise ValueError("meaningful progress flags must be exact booleans")
        if actual_meaningful != expected_meaningful:
            raise ValueError("meaningful flags must use the inclusive frozen threshold")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "original": self.original,
            "replacement": self.replacement,
            "matched_sham": self.matched_sham,
            "direction": self.direction.value,
            "tolerance": self.tolerance,
            "minimum_meaningful_difference": self.minimum_meaningful_difference,
            "raw_replacement_minus_original": self.raw_replacement_minus_original,
            "raw_sham_minus_original": self.raw_sham_minus_original,
            "raw_replacement_minus_sham": self.raw_replacement_minus_sham,
            "directed_replacement_minus_original": self.directed_replacement_minus_original,
            "directed_sham_minus_original": self.directed_sham_minus_original,
            "directed_replacement_minus_sham": self.directed_replacement_minus_sham,
            "meaningful_replacement_minus_original": self.meaningful_replacement_minus_original,
            "meaningful_sham_minus_original": self.meaningful_sham_minus_original,
            "meaningful_replacement_minus_sham": self.meaningful_replacement_minus_sham,
        }


def _pattern_from_values(
    outcomes: tuple[bool | None, bool | None, bool | None],
    progress_values: tuple[float | None, float | None, float | None],
    effects: HistoricalProgressEffectsV1 | None,
) -> HistoricalOutcomePattern | None:
    """Derive a raw pattern without consulting parity or budget comparability."""

    if any(type(value) is not bool for value in outcomes):
        return None
    original, replacement, sham = outcomes
    if original:
        return HistoricalOutcomePattern.ORIGINAL_SUCCESS
    if replacement and not sham:
        return HistoricalOutcomePattern.REPLACEMENT_ONLY_SUCCESS
    if replacement and sham:
        return HistoricalOutcomePattern.REPLACEMENT_AND_SHAM_SUCCESS
    if sham:
        return HistoricalOutcomePattern.SHAM_ONLY_SUCCESS
    if any(value is None for value in progress_values):
        return HistoricalOutcomePattern.ALL_FAILED_WITH_PROGRESS_UNAVAILABLE
    if effects is None:
        return HistoricalOutcomePattern.ALL_FAILED_WITH_PROGRESS_OUT_OF_DOMAIN
    if (
        effects.meaningful_replacement_minus_original
        and effects.meaningful_replacement_minus_sham
    ):
        return HistoricalOutcomePattern.ALL_FAILED_WITH_REPLACEMENT_PROGRESS
    if (
        effects.meaningful_replacement_minus_original
        and effects.meaningful_sham_minus_original
        and not effects.meaningful_replacement_minus_sham
    ):
        return HistoricalOutcomePattern.ALL_FAILED_WITH_SHARED_PROGRESS
    return HistoricalOutcomePattern.ALL_FAILED_WITH_NO_MEANINGFUL_PROGRESS


@dataclass(frozen=True)
class HistoricalCandidateScoreV1:
    """Return one candidate's raw pattern while retaining evidence boundaries."""

    suite_id: str
    suite_version: str
    population_id: str
    registry_id: str
    scorer_id: str
    scorer_version: str
    run_id: str
    work_item_id: str
    source_task_id: str
    case_id: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    system_cell_id: str
    checkpoint_id: str
    actor_id: str
    runtime_id: str
    environment_id: str
    locator_treatment_id: str
    fixer_treatment_id: str
    budget_id: str
    repetition_contract_id: str
    repetition_contract_version: str
    repetition_id: str
    sham_contract_id: str
    sham_contract_version: str
    progress_contract_id: str
    verifier_contract_id: str
    candidate_id: str
    candidate_rank: int
    candidate_event_group: tuple[str, ...]
    original_intervention_id: str
    replacement_intervention_id: str
    sham_intervention_id: str
    receipt_status: HistoricalCandidateStatus
    derived_status: HistoricalCandidateStatus
    decision_status: HistoricalCandidateDecisionStatus
    not_executed_reason: HistoricalCandidateNotExecutedReason | None
    invalid_reason: HistoricalInvalidReason | None
    outcome_pattern: HistoricalOutcomePattern | None
    causal_class: None
    original_official_success: bool | None
    replacement_official_success: bool | None
    sham_official_success: bool | None
    original_progress_value: float | None
    replacement_progress_value: float | None
    sham_progress_value: float | None
    progress_effects: HistoricalProgressEffectsV1 | None
    parity: tuple[HistoricalParityDecisionV1, ...]
    sham_only_success: bool | None
    strict_resolution_pattern_observed: bool
    incremental_costs: tuple[ResourceCostV1, ...]
    cumulative_costs: tuple[ResourceCostV1, ...]
    raw_evidence_refs: tuple[HistoricalEvidenceRefV1, ...]
    comparison_action_id: str | None = None
    implementation_status: str = HISTORICAL_IMPLEMENTATION_STATUS
    input_binding_status: str = HISTORICAL_INPUT_BINDING_STATUS
    schema: str = HISTORICAL_CANDIDATE_SCORE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_CANDIDATE_SCORE_SCHEMA:
            raise ValueError("unsupported historical candidate score schema")
        if (
            self.suite_id != TRACEFIX_BENCH_SUITE_ID
            or self.suite_version != SCORER_REGISTRY_SUITE_VERSION_V0_1
        ):
            raise ValueError("historical score must use the frozen v0.1 suite")
        for label in (
            "population_id",
            "registry_id",
            "scorer_id",
            "scorer_version",
            "run_id",
            "work_item_id",
            "case_id",
            "benchmark_id",
            "benchmark_version",
            "adapter_version",
            "system_cell_id",
            "locator_treatment_id",
            "fixer_treatment_id",
            "budget_id",
            "repetition_contract_id",
            "repetition_contract_version",
            "repetition_id",
            "sham_contract_id",
            "sham_contract_version",
            "progress_contract_id",
            "verifier_contract_id",
            "candidate_id",
            "original_intervention_id",
            "replacement_intervention_id",
            "sham_intervention_id",
        ):
            _identifier(getattr(self, label), label)
        for label in (
            "source_task_id",
            "checkpoint_id",
            "actor_id",
            "runtime_id",
            "environment_id",
        ):
            _text(getattr(self, label), label)
        if type(self.candidate_rank) is not int or self.candidate_rank < 1:
            raise ValueError("candidate_rank must be a positive integer")
        _text_tuple(
            self.candidate_event_group,
            "candidate_event_group",
            unique=True,
            nonempty=True,
        )
        if (
            not isinstance(self.receipt_status, HistoricalCandidateStatus)
            or not isinstance(self.derived_status, HistoricalCandidateStatus)
            or not isinstance(self.decision_status, HistoricalCandidateDecisionStatus)
        ):
            raise ValueError("score statuses must use closed enums")
        if self.causal_class is not None:
            raise ValueError("candidate-local scorer cannot emit a causal class")
        if self.outcome_pattern is not None and not isinstance(
            self.outcome_pattern, HistoricalOutcomePattern
        ):
            raise ValueError("outcome_pattern must use the closed raw enum")
        if self.progress_effects is not None and type(
            self.progress_effects
        ) is not HistoricalProgressEffectsV1:
            raise ValueError("progress_effects must use the typed effect record")
        _record_tuple(self.parity, HistoricalParityDecisionV1, "parity")
        _record_tuple(
            self.raw_evidence_refs,
            HistoricalEvidenceRefV1,
            "raw_evidence_refs",
        )
        if len({ref.artifact_ref for ref in self.raw_evidence_refs}) != len(
            self.raw_evidence_refs
        ):
            raise ValueError("score evidence closure cannot repeat artifacts")
        if any(
            (
                ref.run_id,
                ref.work_item_id,
                ref.case_id,
                ref.candidate_id,
                ref.repetition_id,
            )
            != (
                self.run_id,
                self.work_item_id,
                self.case_id,
                self.candidate_id,
                self.repetition_id,
            )
            for ref in self.raw_evidence_refs
        ):
            raise ValueError("score evidence owner tuple drifted from candidate")
        if self.comparison_action_id is not None:
            _identifier(self.comparison_action_id, "comparison_action_id")
        # Validate nested evidence here as scores can be constructed without receipts.
        for decision in self.parity:
            if self.comparison_action_id is None:
                raise ValueError("score parity requires its exact comparison action owner")
            _require_candidate_evidence(
                decision.evidence_ref,
                kind=(
                    HistoricalEvidenceKind.SHAM_MATCHER
                    if decision.dimension is HistoricalParityDimension.SHAM_MATCHING
                    else HistoricalEvidenceKind.PARITY
                ),
                run_id=self.run_id,
                work_item_id=self.work_item_id,
                action_id=self.comparison_action_id,
                case_id=self.case_id,
                candidate_id=self.candidate_id,
                repetition_id=self.repetition_id,
            )
            if decision.evidence_ref not in self.raw_evidence_refs:
                raise ValueError("score parity evidence is absent from its raw evidence closure")
        stable_executed = (
            self.receipt_status is HistoricalCandidateStatus.EXECUTED
            and self.derived_status is HistoricalCandidateStatus.EXECUTED
        )
        if stable_executed:
            dimensions = tuple(decision.dimension for decision in self.parity)
            if (
                len(dimensions) != len(_ALL_PARITY_DIMENSIONS)
                or set(dimensions) != _ALL_PARITY_DIMENSIONS
                or len(set(dimensions)) != len(dimensions)
            ):
                raise ValueError("executed score requires every parity dimension once")
        # Invalid records retain raw parity claims; only admitted patterns require
        # unavailable budget parity until a formal registry binder exists.
        if self.decision_status is HistoricalCandidateDecisionStatus.PATTERN_AVAILABLE:
            if any(
                decision.dimension is HistoricalParityDimension.BUDGET_COMPARABILITY
                and decision.status is not HistoricalParityStatus.UNAVAILABLE
                for decision in self.parity
            ):
                raise ValueError("candidate-only pattern cannot verify registry budget parity")
        for label in (
            "original_progress_value",
            "replacement_progress_value",
            "sham_progress_value",
        ):
            value = getattr(self, label)
            if value is not None:
                _finite(value, label)
        progress_values = (
            self.original_progress_value,
            self.replacement_progress_value,
            self.sham_progress_value,
        )
        if self.progress_effects is not None and progress_values != (
            self.progress_effects.original,
            self.progress_effects.replacement,
            self.progress_effects.matched_sham,
        ):
            raise ValueError("progress effects must retain all raw values")
        outcomes = (
            self.original_official_success,
            self.replacement_official_success,
            self.sham_official_success,
        )
        expected_pattern = _pattern_from_values(
            outcomes,
            progress_values,
            self.progress_effects,
        )
        expected_strict = (
            self.outcome_pattern is HistoricalOutcomePattern.REPLACEMENT_ONLY_SUCCESS
        )
        if (
            type(self.strict_resolution_pattern_observed) is not bool
            or self.strict_resolution_pattern_observed != expected_strict
        ):
            raise ValueError("strict pattern flag must follow raw O/R/S outcomes")
        expected_sham_only = (
            expected_pattern is HistoricalOutcomePattern.SHAM_ONLY_SUCCESS
        )
        if self.outcome_pattern is not None:
            if self.outcome_pattern is not expected_pattern:
                raise ValueError("outcome pattern contradicts raw outcomes or progress")
            if self.sham_only_success is not expected_sham_only:
                raise ValueError("sham-only diagnostic contradicts the raw pattern")
        elif self.sham_only_success is not None:
            raise ValueError("score without a raw pattern cannot claim sham-only success")
        status_mismatch = self.receipt_status is not self.derived_status
        if status_mismatch:
            if (
                self.decision_status
                is not HistoricalCandidateDecisionStatus.INFRASTRUCTURE_INVALID
                or self.invalid_reason is not HistoricalInvalidReason.SCORER_INPUT_INVALID
                or self.outcome_pattern is not None
                or self.not_executed_reason is not None
            ):
                raise ValueError("declared and derived status mismatch must fail closed")
        elif self.derived_status is HistoricalCandidateStatus.EXECUTED:
            allowed = {
                HistoricalCandidateDecisionStatus.PATTERN_AVAILABLE,
                HistoricalCandidateDecisionStatus.INFRASTRUCTURE_INVALID,
            }
            if self.decision_status not in allowed:
                raise ValueError("executed candidate has an impossible decision status")
        elif self.derived_status is HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED:
            if self.decision_status not in {
                HistoricalCandidateDecisionStatus.CANDIDATE_NOT_EXECUTED,
                HistoricalCandidateDecisionStatus.INFRASTRUCTURE_INVALID,
            }:
                raise ValueError("unexecuted candidate has an impossible decision status")
        else:
            expected_decision = {
                HistoricalCandidateStatus.METHOD_FAILED:
                    HistoricalCandidateDecisionStatus.METHOD_FAILED,
                HistoricalCandidateStatus.INFRASTRUCTURE_INVALID:
                    HistoricalCandidateDecisionStatus.INFRASTRUCTURE_INVALID,
                HistoricalCandidateStatus.AMBIGUOUS_RESUME:
                    HistoricalCandidateDecisionStatus.AMBIGUOUS_RESUME,
            }[self.derived_status]
            if self.decision_status is not expected_decision:
                raise ValueError("derived lifecycle status contradicts score decision")
        if self.decision_status is HistoricalCandidateDecisionStatus.PATTERN_AVAILABLE:
            if (
                self.outcome_pattern is None
                or any(type(value) is not bool for value in outcomes)
                or self.invalid_reason is not None
                or self.not_executed_reason is not None
            ):
                raise ValueError("available raw pattern requires complete official booleans")
        elif self.decision_status is HistoricalCandidateDecisionStatus.CANDIDATE_NOT_EXECUTED:
            if (
                self.not_executed_reason is None
                or self.invalid_reason is not None
                or self.outcome_pattern is not None
                or any(value is not None for value in outcomes + progress_values)
                or self.progress_effects is not None
                or self.parity
                or self.raw_evidence_refs
                or self.comparison_action_id is not None
            ):
                raise ValueError("unexecuted score cannot contain replay observations")
        else:
            if self.not_executed_reason is not None or not isinstance(
                self.invalid_reason, HistoricalInvalidReason
            ):
                raise ValueError("failed or incomplete score requires one invalid reason")
            if self.outcome_pattern is not None:
                raise ValueError("failed or incomplete score cannot carry a raw pattern")
            required_reason = {
                HistoricalCandidateDecisionStatus.METHOD_FAILED:
                    HistoricalInvalidReason.ARM_METHOD_FAILURE,
                HistoricalCandidateDecisionStatus.AMBIGUOUS_RESUME:
                    HistoricalInvalidReason.ARM_IN_FLIGHT,
            }.get(self.decision_status)
            if required_reason is not None and self.invalid_reason is not required_reason:
                raise ValueError("decision status contradicts its required reason")
            if (
                self.invalid_reason is HistoricalInvalidReason.OFFICIAL_OUTCOME_UNAVAILABLE
                and all(type(value) is bool for value in outcomes)
            ):
                raise ValueError("unavailable official outcome cannot carry three booleans")
            if self.decision_status is HistoricalCandidateDecisionStatus.INFRASTRUCTURE_INVALID:
                allowed_infrastructure_reasons = {
                    HistoricalCandidateStatus.EXECUTED: {
                        HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH,
                        HistoricalInvalidReason.BENCHMARK_DOMAIN_MISMATCH,
                        HistoricalInvalidReason.REPETITION_PLAN_MISMATCH,
                        HistoricalInvalidReason.ARM_IDENTITY_MISMATCH,
                        HistoricalInvalidReason.OFFICIAL_OUTCOME_UNAVAILABLE,
                    },
                    HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED: {
                        HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH,
                        HistoricalInvalidReason.BENCHMARK_DOMAIN_MISMATCH,
                        HistoricalInvalidReason.REPETITION_PLAN_MISMATCH,
                    },
                    HistoricalCandidateStatus.INFRASTRUCTURE_INVALID: {
                        HistoricalInvalidReason.ARM_MISSING,
                        HistoricalInvalidReason.ARM_INFRASTRUCTURE_FAILURE,
                    },
                }
                if (
                    not status_mismatch
                    and self.invalid_reason
                    not in allowed_infrastructure_reasons[self.derived_status]
                ):
                    raise ValueError("infrastructure reason contradicts lifecycle status")
        require_costs = self.derived_status is not HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED
        _cost_tuple(self.incremental_costs, "incremental_costs", require_nonempty=require_costs)
        _cost_tuple(self.cumulative_costs, "cumulative_costs", require_nonempty=require_costs)
        _unverified(self.implementation_status)
        if self.input_binding_status != HISTORICAL_INPUT_BINDING_STATUS:
            raise ValueError("candidate score remains unbound to a formal run")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "implementation_status": self.implementation_status,
            "input_binding_status": self.input_binding_status,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "registry_id": self.registry_id,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "source_task_id": self.source_task_id,
            "case_id": self.case_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "system_cell_id": self.system_cell_id,
            "checkpoint_id": self.checkpoint_id,
            "actor_id": self.actor_id,
            "runtime_id": self.runtime_id,
            "environment_id": self.environment_id,
            "locator_treatment_id": self.locator_treatment_id,
            "fixer_treatment_id": self.fixer_treatment_id,
            "budget_id": self.budget_id,
            "repetition_contract_id": self.repetition_contract_id,
            "repetition_contract_version": self.repetition_contract_version,
            "repetition_id": self.repetition_id,
            "sham_contract_id": self.sham_contract_id,
            "sham_contract_version": self.sham_contract_version,
            "progress_contract_id": self.progress_contract_id,
            "verifier_contract_id": self.verifier_contract_id,
            "candidate_id": self.candidate_id,
            "candidate_rank": self.candidate_rank,
            "candidate_event_group": list(self.candidate_event_group),
            "original_intervention_id": self.original_intervention_id,
            "replacement_intervention_id": self.replacement_intervention_id,
            "sham_intervention_id": self.sham_intervention_id,
            "receipt_status": self.receipt_status.value,
            "derived_status": self.derived_status.value,
            "decision_status": self.decision_status.value,
            "not_executed_reason": (
                self.not_executed_reason.value if self.not_executed_reason else None
            ),
            "invalid_reason": self.invalid_reason.value if self.invalid_reason else None,
            "outcome_pattern": self.outcome_pattern.value if self.outcome_pattern else None,
            "causal_class": None,
            "original_official_success": self.original_official_success,
            "replacement_official_success": self.replacement_official_success,
            "sham_official_success": self.sham_official_success,
            "original_progress_value": self.original_progress_value,
            "replacement_progress_value": self.replacement_progress_value,
            "sham_progress_value": self.sham_progress_value,
            "progress_effects": self.progress_effects.to_dict() if self.progress_effects else None,
            "parity": [decision.to_dict() for decision in self.parity],
            "sham_only_success": self.sham_only_success,
            "strict_resolution_pattern_observed": self.strict_resolution_pattern_observed,
            "incremental_costs": [cost.to_dict() for cost in self.incremental_costs],
            "cumulative_costs": [cost.to_dict() for cost in self.cumulative_costs],
            "raw_evidence_refs": [ref.to_dict() for ref in self.raw_evidence_refs],
            "comparison_action_id": self.comparison_action_id,
        }


def _domain(contract: object) -> tuple[str, str, str]:
    return (
        getattr(contract, "benchmark_id"),
        getattr(contract, "benchmark_version"),
        getattr(contract, "adapter_version"),
    )


def _effect_record(
    values: tuple[float, float, float],
    contract: ProgressMetricContractV1,
) -> HistoricalProgressEffectsV1:
    original, replacement, sham = values
    raw = (replacement - original, sham - original, replacement - sham)
    sign = 1.0 if contract.direction is ProgressDirection.HIGHER_IS_BETTER else -1.0
    directed = tuple(sign * value for value in raw)
    threshold = contract.minimum_meaningful_difference
    return HistoricalProgressEffectsV1(
        original=original,
        replacement=replacement,
        matched_sham=sham,
        direction=contract.direction,
        tolerance=contract.tolerance,
        minimum_meaningful_difference=threshold,
        raw_replacement_minus_original=raw[0],
        raw_sham_minus_original=raw[1],
        raw_replacement_minus_sham=raw[2],
        directed_replacement_minus_original=directed[0],
        directed_sham_minus_original=directed[1],
        directed_replacement_minus_sham=directed[2],
        meaningful_replacement_minus_original=directed[0] >= threshold,
        meaningful_sham_minus_original=directed[1] >= threshold,
        meaningful_replacement_minus_sham=directed[2] >= threshold,
    )


def _derived_candidate_status(
    receipt: HistoricalThreeArmReceiptV1,
) -> HistoricalCandidateStatus:
    """Derive lifecycle priority from arm states before trusting top-level status."""

    if receipt.candidate_status is HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED:
        return HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED
    statuses = {arm.execution_status for arm in receipt.arms}
    if HistoricalArmExecutionStatus.IN_FLIGHT in statuses:
        return HistoricalCandidateStatus.AMBIGUOUS_RESUME
    if HistoricalArmExecutionStatus.INFRASTRUCTURE_FAILED in statuses:
        return HistoricalCandidateStatus.INFRASTRUCTURE_INVALID
    if HistoricalArmExecutionStatus.METHOD_FAILED in statuses:
        return HistoricalCandidateStatus.METHOD_FAILED
    if len(receipt.arms) != 3 or {arm.arm for arm in receipt.arms} != _ALL_ARM_ROLES:
        return HistoricalCandidateStatus.INFRASTRUCTURE_INVALID
    return HistoricalCandidateStatus.EXECUTED


def _parity_identity(
    arm: HistoricalArmReceiptV1,
    dimension: HistoricalParityDimension,
    comparison: HistoricalCandidateComparisonReceiptV1,
) -> str:
    """Resolve parity identities from typed arm fields, never caller labels."""

    if dimension is HistoricalParityDimension.PRE_ACTION_STATE:
        return arm.pre_action_state_id
    if dimension is HistoricalParityDimension.RESTORE:
        return f"{arm.restore_protocol_id}:{arm.restored_state_id}"
    if dimension is HistoricalParityDimension.ACTOR:
        return arm.actor_id
    if dimension is HistoricalParityDimension.RUNTIME_ENVIRONMENT:
        return ":".join(
            (arm.runtime_id, arm.environment_id, arm.adapter_version, arm.system_cell_id)
        )
    if dimension is HistoricalParityDimension.CONTINUATION_SEED:
        return str(arm.continuation_seed)
    if dimension is HistoricalParityDimension.ACTION_PROTOCOL:
        return arm.action_protocol_id
    if dimension is HistoricalParityDimension.VERIFIER_PROTOCOL:
        return arm.verifier_protocol_id
    if dimension is HistoricalParityDimension.BUDGET_COMPARABILITY:
        return arm.budget_id
    if dimension is HistoricalParityDimension.ARM_ISOLATION:
        return arm.isolation_id
    if dimension is HistoricalParityDimension.SHAM_MATCHING:
        matcher = comparison.sham_matcher_decision
        return matcher.matcher_contract_id or "typed_matcher_unavailable"
    raise ValueError("unsupported historical parity dimension")


def _resolve_sham_field(
    replacement: HistoricalArmReceiptV1,
    sham: HistoricalArmReceiptV1,
    field: HistoricalShamMatchField,
) -> HistoricalShamFieldRelation:
    """Resolve closed sham fields only from typed R/S plan and execution data."""

    if field is HistoricalShamMatchField.CANDIDATE_EVENT_GROUP:
        values = (
            replacement.intervention.plan.candidate_event_group,
            sham.intervention.plan.candidate_event_group,
        )
    elif field is HistoricalShamMatchField.TARGET_EVENT_GROUP:
        values = (
            replacement.intervention.plan.target_event_group,
            sham.intervention.plan.target_event_group,
        )
    elif field is HistoricalShamMatchField.ACTION_PROTOCOL:
        values = (replacement.action_protocol_id, sham.action_protocol_id)
    elif field is HistoricalShamMatchField.CONTROLLED_ACTIONS:
        values = (replacement.controlled_actions, sham.controlled_actions)
    elif field is HistoricalShamMatchField.POST_INTERVENTION_STATE:
        values = (
            replacement.intervention.post_intervention_state_id,
            sham.intervention.post_intervention_state_id,
        )
        if any(value is None for value in values):
            return HistoricalShamFieldRelation.UNAVAILABLE
    else:
        raise ValueError("unsupported closed sham field")
    return (
        HistoricalShamFieldRelation.MATCHED
        if values[0] == values[1]
        else HistoricalShamFieldRelation.DIFFERENT
    )


def _identity_reason(
    receipt: HistoricalThreeArmReceiptV1,
    progress_contract: ProgressMetricContractV1,
    verifier_contract: VerifierContractV1,
    sham_contract: ShamContractV1,
    repetition_contract: HistoricalRepetitionContractV1,
) -> HistoricalInvalidReason | None:
    """Reject local identity claims without rewriting their raw parity evidence."""

    if not (
        _domain(receipt)
        == _domain(progress_contract)
        == _domain(verifier_contract)
        == _domain(sham_contract)
        == _domain(repetition_contract)
    ):
        return HistoricalInvalidReason.BENCHMARK_DOMAIN_MISMATCH
    if (
        progress_contract.evidence_source is not EvidenceSource.OFFICIAL_BENCHMARK
        or len(progress_contract.timepoints) != 1
        or receipt.sham_contract_id != sham_contract.sham_contract_id
        or receipt.sham_contract_version != sham_contract.sham_contract_version
        or receipt.repetition_contract_id
        != repetition_contract.repetition_contract_id
        or receipt.repetition_contract_version
        != repetition_contract.repetition_contract_version
    ):
        return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
    repetition = next(
        (
            item
            for item in repetition_contract.repetitions
            if item.repetition_id == receipt.repetition_id
        ),
        None,
    )
    if repetition is None or (
        receipt.candidate_status is not HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED
        and receipt.actual_arm_order
        != repetition.arm_order[: len(receipt.actual_arm_order)]
    ):
        return HistoricalInvalidReason.REPETITION_PLAN_MISMATCH
    if receipt.candidate_status is HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED:
        return None
    arm_by_role = {arm.arm: arm for arm in receipt.arms}
    if set(arm_by_role) != _ALL_ARM_ROLES:
        return HistoricalInvalidReason.ARM_MISSING
    for arm in arm_by_role.values():
        if (
            arm.official_outcome.verifier_contract_id
            != verifier_contract.verifier_contract_id
            or arm.progress.progress_contract_id
            != progress_contract.progress_contract_id
            or arm.progress.evaluator_contract_id
            != progress_contract.evaluator_contract_id
            or arm.progress.timepoint != progress_contract.timepoints[0]
            or arm.verifier_protocol_id != verifier_contract.verifier_contract_id
        ):
            return HistoricalInvalidReason.ARM_IDENTITY_MISMATCH
    if receipt.comparison_receipt is None:
        return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
    comparison = receipt.comparison_receipt
    matcher = comparison.sham_matcher_decision
    replacement = arm_by_role[HistoricalArmRole.REPLACEMENT]
    sham = arm_by_role[HistoricalArmRole.MATCHED_SHAM]
    if sham.intervention.plan.sham_semantics is not sham_contract.semantics:
        return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
    expected_fields = set(sham_contract.matched_fields) | set(
        sham_contract.permitted_differences
    )
    observed_matches = {match.field: match for match in matcher.field_matches}
    if matcher.matcher_contract_id is None:
        if matcher.status is not HistoricalParityStatus.UNAVAILABLE or observed_matches:
            return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
    else:
        if set(observed_matches) != expected_fields:
            return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
        for field, match in observed_matches.items():
            relation = _resolve_sham_field(replacement, sham, field)
            if match.relation is not relation:
                return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
        matched_fields_valid = all(
            observed_matches[field].relation is HistoricalShamFieldRelation.MATCHED
            for field in sham_contract.matched_fields
        )
        any_unavailable = any(
            match.relation is HistoricalShamFieldRelation.UNAVAILABLE
            for match in observed_matches.values()
        )
        expected_status = (
            HistoricalParityStatus.UNAVAILABLE
            if any_unavailable
            else HistoricalParityStatus.VALID
            if matched_fields_valid
            else HistoricalParityStatus.INVALID
        )
        if matcher.status is not expected_status:
            return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
    decisions = {decision.dimension: decision for decision in receipt.parity}
    if set(decisions) != _ALL_PARITY_DIMENSIONS or len(decisions) != len(receipt.parity):
        return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
    ordered_arms = tuple(arm_by_role[role] for role in HistoricalArmRole)
    for dimension, decision in decisions.items():
        expected_ref = (
            comparison.sham_matching_evidence_ref
            if dimension is HistoricalParityDimension.SHAM_MATCHING
            else comparison.parity_evidence_ref
        )
        if decision.evidence_ref != expected_ref:
            return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
        raw = tuple(
            _parity_identity(arm, dimension, comparison) for arm in ordered_arms
        )
        observed = (
            decision.original_identity,
            decision.replacement_identity,
            decision.sham_identity,
        )
        if observed != raw:
            return HistoricalInvalidReason.ARM_IDENTITY_MISMATCH
        if dimension is HistoricalParityDimension.BUDGET_COMPARABILITY:
            # A raw verdict is admissible evidence, but not candidate-only eligibility.
            if decision.status is not HistoricalParityStatus.UNAVAILABLE:
                return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
            continue
        if dimension is HistoricalParityDimension.SHAM_MATCHING:
            if decision.status is not matcher.status:
                return HistoricalInvalidReason.CONTRACT_IDENTITY_MISMATCH
            continue
        raw_valid = len(set(raw)) == (
            3 if dimension is HistoricalParityDimension.ARM_ISOLATION else 1
        )
        expected_status = (
            HistoricalParityStatus.VALID
            if raw_valid
            else HistoricalParityStatus.INVALID
        )
        if (
            decision.status is not HistoricalParityStatus.UNAVAILABLE
            and decision.status is not expected_status
        ):
            return HistoricalInvalidReason.ARM_IDENTITY_MISMATCH
    if any(arm.pre_action_state_id != receipt.checkpoint_id for arm in ordered_arms):
        return HistoricalInvalidReason.ARM_IDENTITY_MISMATCH
    if any(arm.actor_id != receipt.actor_id for arm in ordered_arms):
        return HistoricalInvalidReason.ARM_IDENTITY_MISMATCH
    if any(arm.restored_state_id != receipt.checkpoint_id for arm in ordered_arms):
        return HistoricalInvalidReason.ARM_IDENTITY_MISMATCH
    if any(
        (
            arm.runtime_id,
            arm.environment_id,
            arm.adapter_version,
            arm.system_cell_id,
            arm.continuation_seed,
        )
        != (
            receipt.runtime_id,
            receipt.environment_id,
            receipt.adapter_version,
            receipt.system_cell_id,
            repetition.continuation_seed,
        )
        for arm in ordered_arms
    ):
        return HistoricalInvalidReason.ARM_IDENTITY_MISMATCH
    return None


def classify_historical_candidate_v1(
    receipt: HistoricalThreeArmReceiptV1,
    progress_contract: ProgressMetricContractV1,
    verifier_contract: VerifierContractV1,
    sham_contract: ShamContractV1,
    repetition_contract: HistoricalRepetitionContractV1,
) -> HistoricalCandidateScoreV1:
    """Score one candidate-local receipt as a raw outcome/progress pattern."""

    if type(receipt) is not HistoricalThreeArmReceiptV1:
        raise ValueError("historical scorer requires a typed three-arm receipt")
    if type(progress_contract) is not ProgressMetricContractV1:
        raise ValueError("historical scorer requires a typed progress contract")
    if type(verifier_contract) is not VerifierContractV1:
        raise ValueError("historical scorer requires a typed verifier contract")
    if type(sham_contract) is not ShamContractV1:
        raise ValueError("historical scorer requires a typed sham contract")
    if type(repetition_contract) is not HistoricalRepetitionContractV1:
        raise ValueError("historical scorer requires a typed repetition contract")

    derived_status = _derived_candidate_status(receipt)
    arm_by_role = {arm.arm: arm for arm in receipt.arms}
    official = {
        role: (
            arm_by_role[role].official_outcome.success
            if role in arm_by_role
            and arm_by_role[role].official_outcome.availability
            is HistoricalValueAvailability.AVAILABLE
            else None
        )
        for role in HistoricalArmRole
    }
    progress = {
        role: (
            arm_by_role[role].progress.value
            if role in arm_by_role
            and arm_by_role[role].progress.availability
            is HistoricalValueAvailability.AVAILABLE
            else None
        )
        for role in HistoricalArmRole
    }
    outcomes = tuple(official[role] for role in HistoricalArmRole)
    progress_values = tuple(progress[role] for role in HistoricalArmRole)
    decision = {
        HistoricalCandidateStatus.EXECUTED:
            HistoricalCandidateDecisionStatus.INFRASTRUCTURE_INVALID,
        HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED:
            HistoricalCandidateDecisionStatus.CANDIDATE_NOT_EXECUTED,
        HistoricalCandidateStatus.METHOD_FAILED:
            HistoricalCandidateDecisionStatus.METHOD_FAILED,
        HistoricalCandidateStatus.INFRASTRUCTURE_INVALID:
            HistoricalCandidateDecisionStatus.INFRASTRUCTURE_INVALID,
        HistoricalCandidateStatus.AMBIGUOUS_RESUME:
            HistoricalCandidateDecisionStatus.AMBIGUOUS_RESUME,
    }[derived_status]
    invalid_reason: HistoricalInvalidReason | None = None
    effects: HistoricalProgressEffectsV1 | None = None
    pattern: HistoricalOutcomePattern | None = None

    if receipt.candidate_status is not derived_status:
        invalid_reason = HistoricalInvalidReason.SCORER_INPUT_INVALID
        decision = HistoricalCandidateDecisionStatus.INFRASTRUCTURE_INVALID
    elif derived_status is HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED:
        invalid_reason = _identity_reason(
            receipt,
            progress_contract,
            verifier_contract,
            sham_contract,
            repetition_contract,
        )
        if invalid_reason is not None:
            decision = HistoricalCandidateDecisionStatus.INFRASTRUCTURE_INVALID
    elif derived_status is HistoricalCandidateStatus.AMBIGUOUS_RESUME:
        invalid_reason = HistoricalInvalidReason.ARM_IN_FLIGHT
    elif derived_status is HistoricalCandidateStatus.INFRASTRUCTURE_INVALID:
        invalid_reason = (
            HistoricalInvalidReason.ARM_MISSING
            if set(arm_by_role) != _ALL_ARM_ROLES
            else HistoricalInvalidReason.ARM_INFRASTRUCTURE_FAILURE
        )
    elif derived_status is HistoricalCandidateStatus.METHOD_FAILED:
        invalid_reason = HistoricalInvalidReason.ARM_METHOD_FAILURE
    else:
        invalid_reason = _identity_reason(
            receipt,
            progress_contract,
            verifier_contract,
            sham_contract,
            repetition_contract,
        )
        if invalid_reason is None and any(value is None for value in outcomes):
            invalid_reason = HistoricalInvalidReason.OFFICIAL_OUTCOME_UNAVAILABLE
        elif invalid_reason is None:
            if all(value is not None for value in progress_values) and all(
                progress_contract.domain_min <= value <= progress_contract.domain_max
                for value in progress_values
            ):
                effects = _effect_record(progress_values, progress_contract)
            pattern = _pattern_from_values(outcomes, progress_values, effects)
            decision = HistoricalCandidateDecisionStatus.PATTERN_AVAILABLE

    if decision is not HistoricalCandidateDecisionStatus.PATTERN_AVAILABLE:
        pattern = None
        effects = None
        if derived_status is HistoricalCandidateStatus.CANDIDATE_NOT_EXECUTED:
            official = {role: None for role in HistoricalArmRole}
            progress = {role: None for role in HistoricalArmRole}

    return HistoricalCandidateScoreV1(
        suite_id=receipt.suite_id,
        suite_version=receipt.suite_version,
        population_id=receipt.population_id,
        registry_id=receipt.registry_id,
        scorer_id=receipt.scorer_id,
        scorer_version=receipt.scorer_version,
        run_id=receipt.run_id,
        work_item_id=receipt.work_item_id,
        source_task_id=receipt.source_task_id,
        case_id=receipt.case_id,
        benchmark_id=receipt.benchmark_id,
        benchmark_version=receipt.benchmark_version,
        adapter_version=receipt.adapter_version,
        system_cell_id=receipt.system_cell_id,
        checkpoint_id=receipt.checkpoint_id,
        actor_id=receipt.actor_id,
        runtime_id=receipt.runtime_id,
        environment_id=receipt.environment_id,
        locator_treatment_id=receipt.locator_treatment_id,
        fixer_treatment_id=receipt.fixer_treatment_id,
        budget_id=receipt.budget_id,
        repetition_contract_id=receipt.repetition_contract_id,
        repetition_contract_version=receipt.repetition_contract_version,
        repetition_id=receipt.repetition_id,
        sham_contract_id=receipt.sham_contract_id,
        sham_contract_version=receipt.sham_contract_version,
        progress_contract_id=progress_contract.progress_contract_id,
        verifier_contract_id=verifier_contract.verifier_contract_id,
        candidate_id=receipt.candidate_id,
        candidate_rank=receipt.candidate_rank,
        candidate_event_group=receipt.candidate_event_group,
        original_intervention_id=receipt.original_intervention_id,
        replacement_intervention_id=receipt.replacement_intervention_id,
        sham_intervention_id=receipt.sham_intervention_id,
        receipt_status=receipt.candidate_status,
        derived_status=derived_status,
        decision_status=decision,
        not_executed_reason=(
            receipt.not_executed_reason
            if decision is HistoricalCandidateDecisionStatus.CANDIDATE_NOT_EXECUTED
            else None
        ),
        invalid_reason=invalid_reason,
        outcome_pattern=pattern,
        causal_class=None,
        original_official_success=official[HistoricalArmRole.ORIGINAL],
        replacement_official_success=official[HistoricalArmRole.REPLACEMENT],
        sham_official_success=official[HistoricalArmRole.MATCHED_SHAM],
        original_progress_value=progress[HistoricalArmRole.ORIGINAL],
        replacement_progress_value=progress[HistoricalArmRole.REPLACEMENT],
        sham_progress_value=progress[HistoricalArmRole.MATCHED_SHAM],
        progress_effects=effects,
        parity=receipt.parity,
        sham_only_success=(
            pattern is HistoricalOutcomePattern.SHAM_ONLY_SUCCESS
            if pattern is not None
            else None
        ),
        strict_resolution_pattern_observed=(
            pattern is HistoricalOutcomePattern.REPLACEMENT_ONLY_SUCCESS
        ),
        incremental_costs=receipt.incremental_costs,
        cumulative_costs=receipt.cumulative_costs,
        raw_evidence_refs=receipt.raw_evidence_refs,
        comparison_action_id=(
            receipt.comparison_receipt.action_id
            if receipt.comparison_receipt is not None
            else None
        ),
    )


__all__ = [
    "HISTORICAL_ARM_RECEIPT_SCHEMA",
    "HISTORICAL_CANDIDATE_COMPARISON_RECEIPT_SCHEMA",
    "HISTORICAL_CANDIDATE_SCORE_SCHEMA",
    "HISTORICAL_EVIDENCE_REF_SCHEMA",
    "HISTORICAL_IMPLEMENTATION_STATUS",
    "HISTORICAL_INPUT_BINDING_STATUS",
    "HISTORICAL_INTERVENTION_EXECUTION_SCHEMA",
    "HISTORICAL_INTERVENTION_PLAN_SCHEMA",
    "HISTORICAL_OFFICIAL_OUTCOME_SCHEMA",
    "HISTORICAL_PARITY_DECISION_SCHEMA",
    "HISTORICAL_PROGRESS_EFFECTS_SCHEMA",
    "HISTORICAL_PROGRESS_OBSERVATION_SCHEMA",
    "HISTORICAL_REPETITION_CONTRACT_SCHEMA",
    "HISTORICAL_REPETITION_SPEC_SCHEMA",
    "HISTORICAL_SHAM_FIELD_MATCH_SCHEMA",
    "HISTORICAL_SHAM_MATCHER_DECISION_SCHEMA",
    "HISTORICAL_THREE_ARM_RECEIPT_SCHEMA",
    "SHAM_CONTRACT_SCHEMA",
    "HistoricalArmExecutionStatus",
    "HistoricalArmReceiptV1",
    "HistoricalArmRole",
    "HistoricalCandidateComparisonReceiptV1",
    "HistoricalCandidateDecisionStatus",
    "HistoricalCandidateNotExecutedReason",
    "HistoricalCandidateScoreV1",
    "HistoricalCandidateStatus",
    "HistoricalCausalClass",
    "HistoricalEvidenceKind",
    "HistoricalEvidenceOwnerScope",
    "HistoricalEvidenceRefV1",
    "HistoricalInterventionExecutionV1",
    "HistoricalInterventionKind",
    "HistoricalInterventionPlanV1",
    "HistoricalInvalidReason",
    "HistoricalOfficialOutcomeV1",
    "HistoricalOutcomePattern",
    "HistoricalParityDecisionV1",
    "HistoricalParityDimension",
    "HistoricalParityStatus",
    "HistoricalProgressEffectsV1",
    "HistoricalProgressObservationV1",
    "HistoricalRepetitionContractV1",
    "HistoricalRepetitionSpecV1",
    "HistoricalShamFieldMatchV1",
    "HistoricalShamFieldRelation",
    "HistoricalShamMatchField",
    "HistoricalShamMatcherDecisionV1",
    "HistoricalThreeArmReceiptV1",
    "HistoricalValueAvailability",
    "ShamContractV1",
    "ShamSemantics",
    "classify_historical_candidate_v1",
]
