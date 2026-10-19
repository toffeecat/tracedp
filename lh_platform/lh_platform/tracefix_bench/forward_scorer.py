"""Pure, locally designed forward-recovery case scoring for benchmark v0.1.

The records below keep method execution, official final evaluation, progress,
optional safety oracles, attribution gold, and current-execution root
verification as separate evidence layers.  The scorer performs no I/O and
does not call an Actor, environment, verifier, or oracle.

This module is deliberately provisional.  It does not bind a run plan,
registry snapshot, immutable release, action ledger, result index, or saved
decoder.  Every output therefore remains ``implementation_unverified`` and
``provisional_unbound`` until an independent oracle and a real persisted entry
path cover the complete lifecycle.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracefix_bench.contracts import (
    AttributionGoldV1,
    BenchmarkSplit,
    ExecutionSemantics,
    FixActionKind,
    GoldDisposition,
    LocatorAdviceV1,
    LocatorAdviceMode,
    TraceFixTrack,
    UsageValueStatus,
)
from lh_platform.tracefix_bench.dataset_contracts import TRACEFIX_BENCH_SUITE_ID
from lh_platform.tracefix_bench.resource_budget import (
    BudgetOverallDecision,
    BudgetSubjectKind,
    ResourceBudgetDecisionReceiptV1,
    evaluate_resource_budget_v1,
)
from lh_platform.tracefix_bench.run_contracts import ResourceCostV1
from lh_platform.tracefix_bench.scorer_contracts import (
    EvidenceSource,
    OracleContractV1,
    OracleKind,
    ProgressDirection,
    ProgressMetricContractV1,
    ResourceBudgetContractV1,
    SCORER_REGISTRY_SUITE_VERSION_V0_1,
    VerifierContractV1,
)

FORWARD_IMPLEMENTATION_STATUS = "implementation_unverified"
FORWARD_DESIGN_ORIGIN = "locally_designed"
FORWARD_INPUT_BINDING_STATUS = "provisional_unbound"
FORWARD_TRANSITION_SCHEMA = "longhorizon_tracefix_bench/forward_transition/v1"
FORWARD_EXECUTION_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/forward_execution_receipt/v1"
)
FORWARD_OFFICIAL_OUTCOME_SCHEMA = (
    "longhorizon_tracefix_bench/forward_official_outcome/v1"
)
FORWARD_PROGRESS_OBSERVATION_SCHEMA = (
    "longhorizon_tracefix_bench/forward_progress_observation/v1"
)
FORWARD_PROGRESS_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/forward_progress_receipt/v1"
)
FORWARD_FALSE_REPAIR_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/forward_false_repair_oracle_receipt/v1"
)
FORWARD_FALSE_REPAIR_POLICY_SCHEMA = (
    "longhorizon_tracefix_bench/forward_false_repair_oracle_policy/v1"
)
FORWARD_REGRESSION_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/forward_regression_oracle_receipt/v1"
)
FORWARD_REGRESSION_POLICY_SCHEMA = (
    "longhorizon_tracefix_bench/forward_regression_oracle_policy/v1"
)
FORWARD_PROTECTED_ITEM_HISTORY_SCHEMA = (
    "longhorizon_tracefix_bench/forward_protected_item_history/v1"
)
FORWARD_PROTECTED_ITEM_OBSERVATION_SCHEMA = (
    "longhorizon_tracefix_bench/forward_protected_item_observation/v1"
)
FORWARD_ROOT_GOLD_BINDING_SCHEMA = (
    "longhorizon_tracefix_bench/forward_root_gold_binding/v1"
)
FORWARD_EVIDENCE_REF_CLOSURE_SCHEMA = (
    "longhorizon_tracefix_bench/forward_evidence_ref_closure/v1"
)
FORWARD_CASE_SCORE_SCHEMA = "longhorizon_tracefix_bench/forward_case_score/v1"
FORWARD_PROGRESS_AGGREGATION_RULE_ID = "end_minus_start_directed_v1"
FORWARD_PROGRESS_AGGREGATION_RULE_SOURCE_REF = (
    "docs/experiments/traceback/benchmarks/longhorizon_tracefix_bench/"
    "SCORER_DESIGN_V0_1.md#failure-resolved-and-progress"
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


def _identifier(value: str, label: str) -> None:
    """Require a portable identity without silently normalizing it."""

    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable non-empty identifier")


def _text(value: str | None, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _optional_text(value: str | None, label: str) -> None:
    if value is not None:
        _text(value, label)


def _text_tuple(
    values: tuple[str, ...],
    label: str,
    *,
    unique: bool,
    nonempty: bool = False,
) -> None:
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple")
    if nonempty and not values:
        raise ValueError(f"{label} must be non-empty")
    for value in values:
        _text(value, f"{label} item")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{label} must contain exact unique values")


def _finite(value: int | float, label: str) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")


def _event_group(value: tuple[str, ...] | None, label: str) -> None:
    if value is None:
        return
    _text_tuple(value, label, unique=True, nonempty=True)


def _same_group(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Compare conjunctive event groups as exact sets."""

    return frozenset(left) == frozenset(right)


def _domain(value: object) -> tuple[str, str, str]:
    return (
        getattr(value, "benchmark_id"),
        getattr(value, "benchmark_version"),
        getattr(value, "adapter_version"),
    )


def _execution_identity_matches(
    receipt: object,
    execution: ForwardRecoveryExecutionReceiptV1,
) -> bool:
    """Compare the repeated cross-file identity needed before dereferencing evidence."""

    return (
        getattr(receipt, "run_id") == execution.run_id
        and getattr(receipt, "work_item_id") == execution.work_item_id
        and getattr(receipt, "source_task_id") == execution.source_task_id
        and getattr(receipt, "case_id") == execution.case_id
        and getattr(receipt, "execution_receipt_id") == execution.receipt_id
        and _domain(receipt) == _domain(execution)
    )


def _local_status(design_origin: str, implementation_status: str) -> None:
    if design_origin != FORWARD_DESIGN_ORIGIN:
        raise ValueError("forward scorer records must remain locally_designed")
    if implementation_status != FORWARD_IMPLEMENTATION_STATUS:
        raise ValueError("forward scorer records cannot promote implementation status")


def _costs(values: tuple[ResourceCostV1, ...]) -> None:
    if type(values) is not tuple or any(type(value) is not ResourceCostV1 for value in values):
        raise ValueError("costs must be a tuple of ResourceCostV1 records")
    keys = tuple((value.role, value.scope, value.dimension) for value in values)
    if len(set(keys)) != len(keys):
        raise ValueError("costs cannot repeat a role, scope, and dimension")


def _dedupe_refs(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Preserve first-seen evidence order while removing exact duplicates."""

    output: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for reference in group:
            if reference not in seen:
                output.append(reference)
                seen.add(reference)
    return tuple(output)


def _present_refs(*references: str | None) -> tuple[str, ...]:
    """Keep only explicit references without manufacturing absence evidence."""

    return tuple(reference for reference in references if reference is not None)


def _official_refs(outcome: OfficialRecoveryOutcomeV1 | None) -> tuple[str, ...]:
    if type(outcome) is not OfficialRecoveryOutcomeV1:
        return ()
    return _present_refs(
        outcome.receipt_id,
        outcome.terminal_receipt_ref,
        outcome.evaluator_receipt_ref,
        outcome.unavailable_evidence_ref,
    )


def _progress_refs(receipt: ForwardProgressReceiptV1 | None) -> tuple[str, ...]:
    if receipt is None:
        return ()
    return _dedupe_refs(
        (receipt.receipt_id,),
        tuple(
            reference
            for observation in receipt.observations
            for reference in _present_refs(
                observation.evaluator_receipt_ref,
                observation.unavailable_evidence_ref,
            )
        ),
    )


def _false_repair_refs(receipt: FalseRepairOracleReceiptV1 | None) -> tuple[str, ...]:
    if receipt is None:
        return ()
    return _present_refs(
        receipt.receipt_id,
        receipt.evaluator_receipt_ref,
        receipt.unavailable_evidence_ref,
    )


def _regression_refs(receipt: RegressionOracleReceiptV1 | None) -> tuple[str, ...]:
    if receipt is None:
        return ()
    return _dedupe_refs(
        _present_refs(
            receipt.receipt_id,
            receipt.protected_item_history_receipt_ref,
            receipt.unavailable_evidence_ref,
        ),
        tuple(
            observation.evaluator_receipt_ref
            for history in receipt.protected_item_histories
            for observation in history.observations
        ),
    )


class ForwardExecutionStatus(StrEnum):
    """Separate complete work from method, infrastructure, and resume states."""

    COMPLETED = "completed"
    METHOD_FAILED = "method_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    IN_FLIGHT = "in_flight"


class ForwardStopReason(StrEnum):
    """Freeze why the recovery path stopped without interpreting free text."""

    FIXER_ABSTAINED = "fixer_abstained"
    ACTOR_TERMINATED = "actor_terminated"
    ENVIRONMENT_TERMINATED = "environment_terminated"
    BUDGET_EXHAUSTED = "budget_exhausted"
    METHOD_FAILED = "method_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    INTERRUPTED_UNKNOWN = "interrupted_unknown"


_STOP_REASONS_BY_EXECUTION_STATUS = {
    ForwardExecutionStatus.COMPLETED: frozenset(
        {
            ForwardStopReason.FIXER_ABSTAINED,
            ForwardStopReason.ACTOR_TERMINATED,
            ForwardStopReason.ENVIRONMENT_TERMINATED,
            ForwardStopReason.BUDGET_EXHAUSTED,
        }
    ),
    ForwardExecutionStatus.METHOD_FAILED: frozenset(
        {ForwardStopReason.METHOD_FAILED}
    ),
    ForwardExecutionStatus.INFRASTRUCTURE_FAILED: frozenset(
        {ForwardStopReason.INFRASTRUCTURE_FAILED}
    ),
    ForwardExecutionStatus.IN_FLIGHT: frozenset(
        {ForwardStopReason.INTERRUPTED_UNKNOWN}
    ),
}


class ForwardOfficialOutcomeStatus(StrEnum):
    """Distinguish an official decision from missing and ambiguous terminal state."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    TERMINAL_UNKNOWN = "terminal_unknown"


class ForwardEvidenceStatus(StrEnum):
    """Keep absent registration, case-level unavailability, and invalidity apart."""

    REGISTERED_AVAILABLE = "registered_available"
    REGISTERED_UNAVAILABLE = "registered_unavailable"
    UNREGISTERED = "unregistered"
    METHOD_NOT_EVALUABLE = "method_not_evaluable"
    INVALID = "invalid"


class ForwardOracleReceiptStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ForwardProgressStateRole(StrEnum):
    """Bind an official progress timepoint to an executed state role."""

    FAILED_START = "failed_start"
    TRANSITION_AFTER = "transition_after"
    FINAL = "final"


class ForwardDecisionStatus(StrEnum):
    SCORED = "scored"
    METHOD_FAILED = "method_failed"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    AMBIGUOUS_RESUME = "ambiguous_resume"


class ForwardInvalidReason(StrEnum):
    """Record the first fail-closed reason selected by the pure scorer."""

    EXECUTION_METHOD_FAILURE = "execution_method_failure"
    EXECUTION_INFRASTRUCTURE_FAILURE = "execution_infrastructure_failure"
    EXECUTION_IN_FLIGHT = "execution_in_flight"
    FORWARD_EXECUTION_SEMANTICS_INVALID = "forward_execution_semantics_invalid"
    SAME_ACTOR_IDENTITY_MISMATCH = "same_actor_identity_mismatch"
    OFFICIAL_VERIFIER_CONTRACT_MISSING = "official_verifier_contract_missing"
    OFFICIAL_OUTCOME_MISSING = "official_outcome_missing"
    OFFICIAL_OUTCOME_UNAVAILABLE = "official_outcome_unavailable"
    OFFICIAL_TERMINAL_UNKNOWN = "official_terminal_unknown"
    OFFICIAL_OUTCOME_IDENTITY_MISMATCH = "official_outcome_identity_mismatch"
    PROGRESS_CONTRACT_MISMATCH = "progress_contract_mismatch"
    PROGRESS_RECEIPT_MISSING = "progress_receipt_missing"
    PROGRESS_RECEIPT_IDENTITY_MISMATCH = "progress_receipt_identity_mismatch"
    PROGRESS_VALUE_OUT_OF_DOMAIN = "progress_value_out_of_domain"
    ORACLE_CONTRACT_MISMATCH = "oracle_contract_mismatch"
    REGISTERED_ORACLE_RECEIPT_MISSING = "registered_oracle_receipt_missing"
    ORACLE_RECEIPT_IDENTITY_MISMATCH = "oracle_receipt_identity_mismatch"
    ROOT_GOLD_CASE_MISMATCH = "root_gold_case_mismatch"
    ROOT_GOLD_EVENT_OUTSIDE_ROSTER = "root_gold_event_outside_roster"
    BUDGET_RECEIPT_IDENTITY_MISMATCH = "budget_receipt_identity_mismatch"


@dataclass(frozen=True)
class ForwardRecoveryTransitionV1:
    """Preserve one actually executed transition and its raw receipt identity."""

    transition_id: str
    ordinal: int
    state_id_before: str
    action_id: str
    action_kind: FixActionKind
    state_id_after: str
    action_receipt_ref: str
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_TRANSITION_SCHEMA:
            raise ValueError("unsupported forward transition schema")
        _identifier(self.transition_id, "transition_id")
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise ValueError("transition ordinal must be a positive integer")
        for label, value in (
            ("state_id_before", self.state_id_before),
            ("action_id", self.action_id),
            ("state_id_after", self.state_id_after),
            ("action_receipt_ref", self.action_receipt_ref),
        ):
            _text(value, label)
        if not isinstance(self.action_kind, FixActionKind):
            raise ValueError("action_kind must use the closed FixActionKind enum")
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "transition_id": self.transition_id,
            "ordinal": self.ordinal,
            "state_id_before": self.state_id_before,
            "action_id": self.action_id,
            "action_kind": self.action_kind.value,
            "state_id_after": self.state_id_after,
            "action_receipt_ref": self.action_receipt_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ForwardRecoveryExecutionReceiptV1:
    """Store one root-optional same-Actor recovery path before scoring."""

    receipt_id: str
    run_id: str
    population_id: str
    registry_id: str
    scorer_id: str
    scorer_version: str
    case_id: str
    source_task_id: str
    work_item_id: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    system_cell_id: str
    treatment_id: str
    locator_treatment_id: str
    fixer_treatment_id: str
    repetition_id: str
    budget_id: str
    split: BenchmarkSplit
    execution_semantics: ExecutionSemantics
    source_actor_identity: str
    recovery_actor_identity: str
    same_actor_identity_receipt_ref: str | None
    failed_start_state_id: str
    public_event_ids: tuple[str, ...]
    locator_advice: LocatorAdviceV1
    locator_advice_receipt_ref: str
    identified_root_event_group: tuple[str, ...] | None
    transitions: tuple[ForwardRecoveryTransitionV1, ...]
    final_state_id: str | None
    status: ForwardExecutionStatus
    stop_reason: ForwardStopReason
    terminal_receipt_ref: str | None
    costs: tuple[ResourceCostV1, ...]
    raw_receipt_refs: tuple[str, ...]
    status_detail: str | None = None
    track: TraceFixTrack = TraceFixTrack.FORWARD_RECOVERY
    suite_id: str = TRACEFIX_BENCH_SUITE_ID
    suite_version: str = SCORER_REGISTRY_SUITE_VERSION_V0_1
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_EXECUTION_RECEIPT_SCHEMA:
            raise ValueError("unsupported forward execution receipt schema")
        for label, value in (
            ("receipt_id", self.receipt_id),
            ("run_id", self.run_id),
            ("population_id", self.population_id),
            ("registry_id", self.registry_id),
            ("scorer_id", self.scorer_id),
            ("scorer_version", self.scorer_version),
            ("case_id", self.case_id),
            ("work_item_id", self.work_item_id),
            ("benchmark_id", self.benchmark_id),
            ("benchmark_version", self.benchmark_version),
            ("adapter_version", self.adapter_version),
            ("system_cell_id", self.system_cell_id),
            ("treatment_id", self.treatment_id),
            ("locator_treatment_id", self.locator_treatment_id),
            ("fixer_treatment_id", self.fixer_treatment_id),
            ("repetition_id", self.repetition_id),
            ("budget_id", self.budget_id),
        ):
            _identifier(value, label)
        _text(self.source_task_id, "source_task_id")
        for label, value in (
            ("source_actor_identity", self.source_actor_identity),
            ("recovery_actor_identity", self.recovery_actor_identity),
            ("failed_start_state_id", self.failed_start_state_id),
        ):
            _text(value, label)
        _optional_text(
            self.same_actor_identity_receipt_ref,
            "same_actor_identity_receipt_ref",
        )
        _text(self.locator_advice_receipt_ref, "locator_advice_receipt_ref")
        _optional_text(self.final_state_id, "final_state_id")
        _optional_text(self.terminal_receipt_ref, "terminal_receipt_ref")
        _optional_text(self.status_detail, "status_detail")
        if self.track is not TraceFixTrack.FORWARD_RECOVERY:
            raise ValueError("forward execution receipt must use the forward track")
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID:
            raise ValueError("forward execution receipt suite_id differs")
        if self.suite_version != SCORER_REGISTRY_SUITE_VERSION_V0_1:
            raise ValueError("forward execution receipt suite_version differs")
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError("execution_semantics must use its closed enum")
        if not isinstance(self.split, BenchmarkSplit):
            raise ValueError("split must use the closed BenchmarkSplit enum")
        if type(self.locator_advice) is not LocatorAdviceV1:
            raise ValueError("locator_advice must be a frozen LocatorAdviceV1")
        if not isinstance(self.status, ForwardExecutionStatus):
            raise ValueError("status must use the forward execution enum")
        if not isinstance(self.stop_reason, ForwardStopReason):
            raise ValueError("stop_reason must use the forward stop enum")
        if self.stop_reason not in _STOP_REASONS_BY_EXECUTION_STATUS[self.status]:
            raise ValueError("stop_reason contradicts forward execution status")
        _text_tuple(
            self.public_event_ids,
            "public_event_ids",
            unique=True,
            nonempty=True,
        )
        _event_group(self.identified_root_event_group, "identified_root_event_group")
        if any(
            event_id not in self.public_event_ids
            for group in self.locator_advice.ranked_event_groups
            for event_id in group
        ):
            raise ValueError("locator advice references an event outside the public roster")
        if self.identified_root_event_group is not None and not set(
            self.identified_root_event_group
        ) <= set(self.public_event_ids):
            raise ValueError("identified root references an event outside the public roster")
        if self.locator_advice.mode in {
            LocatorAdviceMode.NONE,
            LocatorAdviceMode.ABSTAIN,
        }:
            if self.identified_root_event_group is not None:
                raise ValueError("empty locator advice cannot identify a root")
        elif self.identified_root_event_group is not None and not any(
            _same_group(self.identified_root_event_group, group)
            for group in self.locator_advice.ranked_event_groups
        ):
            raise ValueError("identified root must be one frozen locator-advice group")
        if type(self.transitions) is not tuple or any(
            type(value) is not ForwardRecoveryTransitionV1
            for value in self.transitions
        ):
            raise ValueError("transitions must be a tuple of forward transition records")
        transition_ids = tuple(value.transition_id for value in self.transitions)
        if len(set(transition_ids)) != len(transition_ids):
            raise ValueError("transition ids must be unique")
        if tuple(value.ordinal for value in self.transitions) != tuple(
            range(1, len(self.transitions) + 1)
        ):
            raise ValueError("transition ordinals must be contiguous and ordered")
        expected_state = self.failed_start_state_id
        for transition in self.transitions:
            if transition.state_id_before != expected_state:
                raise ValueError("forward transitions must form one continuous state chain")
            expected_state = transition.state_id_after
        if self.final_state_id is not None and self.final_state_id != expected_state:
            raise ValueError("final_state_id differs from the executed transition chain")
        if self.status is ForwardExecutionStatus.COMPLETED:
            if self.final_state_id is None or self.terminal_receipt_ref is None:
                raise ValueError("completed execution requires final state and terminal receipt")
        if self.status is ForwardExecutionStatus.IN_FLIGHT and self.terminal_receipt_ref is not None:
            raise ValueError("in-flight execution cannot claim a terminal receipt")
        _costs(self.costs)
        _text_tuple(
            self.raw_receipt_refs,
            "raw_receipt_refs",
            unique=True,
            nonempty=True,
        )
        owned_refs = {
            reference
            for reference in (
                self.same_actor_identity_receipt_ref,
                self.locator_advice_receipt_ref,
                self.terminal_receipt_ref,
                *(transition.action_receipt_ref for transition in self.transitions),
                *(
                    reference
                    for cost in self.costs
                    for reference in (cost.receipt_ref, cost.unavailable_evidence_ref)
                ),
            )
            if reference is not None
        }
        if not owned_refs <= set(self.raw_receipt_refs):
            raise ValueError("raw_receipt_refs must close over execution-owned evidence")
        _local_status(self.design_origin, self.implementation_status)

    @property
    def transition_ids(self) -> tuple[str, ...]:
        return tuple(value.transition_id for value in self.transitions)

    @property
    def observed_state_ids(self) -> tuple[str, ...]:
        return (self.failed_start_state_id,) + tuple(
            value.state_id_after for value in self.transitions
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "population_id": self.population_id,
            "registry_id": self.registry_id,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "case_id": self.case_id,
            "source_task_id": self.source_task_id,
            "work_item_id": self.work_item_id,
            "track": self.track.value,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "system_cell_id": self.system_cell_id,
            "treatment_id": self.treatment_id,
            "locator_treatment_id": self.locator_treatment_id,
            "fixer_treatment_id": self.fixer_treatment_id,
            "repetition_id": self.repetition_id,
            "budget_id": self.budget_id,
            "split": self.split.value,
            "execution_semantics": self.execution_semantics.value,
            "source_actor_identity": self.source_actor_identity,
            "recovery_actor_identity": self.recovery_actor_identity,
            "same_actor_identity_receipt_ref": self.same_actor_identity_receipt_ref,
            "failed_start_state_id": self.failed_start_state_id,
            "public_event_ids": list(self.public_event_ids),
            "locator_advice": self.locator_advice.to_dict(),
            "locator_advice_receipt_ref": self.locator_advice_receipt_ref,
            "identified_root_event_group": (
                list(self.identified_root_event_group)
                if self.identified_root_event_group is not None
                else None
            ),
            "transitions": [value.to_dict() for value in self.transitions],
            "final_state_id": self.final_state_id,
            "status": self.status.value,
            "stop_reason": self.stop_reason.value,
            "terminal_receipt_ref": self.terminal_receipt_ref,
            "costs": [value.to_dict() for value in self.costs],
            "raw_receipt_refs": list(self.raw_receipt_refs),
            "status_detail": self.status_detail,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class OfficialRecoveryOutcomeV1:
    """Store an independently evaluated official final outcome."""

    receipt_id: str
    run_id: str
    work_item_id: str
    source_task_id: str
    case_id: str
    execution_receipt_id: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    verifier_contract_id: str
    official_verifier_id: str
    official_verifier_version: str
    final_success_field: str
    final_state_id: str | None
    status: ForwardOfficialOutcomeStatus
    failure_resolved: bool | None
    terminal_receipt_ref: str | None
    evaluator_receipt_ref: str | None
    unavailable_reason: str | None = None
    unavailable_evidence_ref: str | None = None
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_OFFICIAL_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_OFFICIAL_OUTCOME_SCHEMA:
            raise ValueError("unsupported forward official outcome schema")
        for label, value in (
            ("receipt_id", self.receipt_id),
            ("run_id", self.run_id),
            ("work_item_id", self.work_item_id),
            ("case_id", self.case_id),
            ("execution_receipt_id", self.execution_receipt_id),
            ("benchmark_id", self.benchmark_id),
            ("benchmark_version", self.benchmark_version),
            ("adapter_version", self.adapter_version),
            ("verifier_contract_id", self.verifier_contract_id),
            ("official_verifier_id", self.official_verifier_id),
            ("official_verifier_version", self.official_verifier_version),
            ("final_success_field", self.final_success_field),
        ):
            _identifier(value, label)
        _text(self.source_task_id, "source_task_id")
        if not isinstance(self.status, ForwardOfficialOutcomeStatus):
            raise ValueError("official outcome status must use its closed enum")
        _optional_text(self.final_state_id, "official final_state_id")
        _optional_text(self.terminal_receipt_ref, "official terminal_receipt_ref")
        _optional_text(self.evaluator_receipt_ref, "official evaluator_receipt_ref")
        _optional_text(self.unavailable_reason, "official unavailable_reason")
        _optional_text(
            self.unavailable_evidence_ref,
            "official unavailable_evidence_ref",
        )
        if self.status is ForwardOfficialOutcomeStatus.AVAILABLE:
            if type(self.failure_resolved) is not bool:
                raise ValueError("available official outcome requires a boolean decision")
            if (
                self.final_state_id is None
                or self.terminal_receipt_ref is None
                or self.evaluator_receipt_ref is None
            ):
                raise ValueError(
                    "available official outcome requires terminal and evaluator receipts"
                )
            if self.unavailable_reason is not None or self.unavailable_evidence_ref is not None:
                raise ValueError("available official outcome cannot carry unavailability data")
        else:
            if self.failure_resolved is not None:
                raise ValueError("unavailable official outcome must keep decision null")
            if self.unavailable_reason is None or self.unavailable_evidence_ref is None:
                raise ValueError(
                    "non-available official outcome requires evidenced unavailability"
                )
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "source_task_id": self.source_task_id,
            "case_id": self.case_id,
            "execution_receipt_id": self.execution_receipt_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "verifier_contract_id": self.verifier_contract_id,
            "official_verifier_id": self.official_verifier_id,
            "official_verifier_version": self.official_verifier_version,
            "final_success_field": self.final_success_field,
            "final_state_id": self.final_state_id,
            "status": self.status.value,
            "failure_resolved": self.failure_resolved,
            "terminal_receipt_ref": self.terminal_receipt_ref,
            "evaluator_receipt_ref": self.evaluator_receipt_ref,
            "unavailable_reason": self.unavailable_reason,
            "unavailable_evidence_ref": self.unavailable_evidence_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ForwardProgressObservationV1:
    """Preserve one official progress value or evidenced case-level absence."""

    timepoint: str
    state_role: ForwardProgressStateRole
    state_id: str
    transition_id: str | None
    value_status: UsageValueStatus
    value: float | None
    evaluator_receipt_ref: str | None
    unavailable_reason: str | None = None
    unavailable_evidence_ref: str | None = None
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_PROGRESS_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_PROGRESS_OBSERVATION_SCHEMA:
            raise ValueError("unsupported forward progress observation schema")
        _identifier(self.timepoint, "progress timepoint")
        _text(self.state_id, "progress state_id")
        _optional_text(self.transition_id, "progress transition_id")
        if not isinstance(self.state_role, ForwardProgressStateRole):
            raise ValueError("progress state_role must use its closed enum")
        if (self.state_role is ForwardProgressStateRole.TRANSITION_AFTER) != (
            self.transition_id is not None
        ):
            raise ValueError("only transition progress may name a transition_id")
        if type(self.value_status) is not UsageValueStatus or self.value_status not in {
            UsageValueStatus.OBSERVED,
            UsageValueStatus.UNAVAILABLE,
        }:
            raise ValueError("official progress must be observed or unavailable")
        _optional_text(self.evaluator_receipt_ref, "progress evaluator_receipt_ref")
        _optional_text(self.unavailable_reason, "progress unavailable_reason")
        _optional_text(
            self.unavailable_evidence_ref,
            "progress unavailable_evidence_ref",
        )
        if self.value_status is UsageValueStatus.UNAVAILABLE:
            if self.value is not None:
                raise ValueError("unavailable progress must keep its value null")
            if self.evaluator_receipt_ref is not None:
                raise ValueError("unavailable progress must use its absence evidence ref")
            if self.unavailable_reason is None or self.unavailable_evidence_ref is None:
                raise ValueError("unavailable progress requires independent evidence")
        else:
            _finite(self.value, "progress value")
            if self.evaluator_receipt_ref is None:
                raise ValueError("available progress requires an evaluator receipt")
            if self.unavailable_reason is not None or self.unavailable_evidence_ref is not None:
                raise ValueError("available progress cannot carry unavailability data")
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "timepoint": self.timepoint,
            "state_role": self.state_role.value,
            "state_id": self.state_id,
            "transition_id": self.transition_id,
            "value_status": self.value_status.value,
            "value": self.value,
            "evaluator_receipt_ref": self.evaluator_receipt_ref,
            "unavailable_reason": self.unavailable_reason,
            "unavailable_evidence_ref": self.unavailable_evidence_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


def _consistent_progress_states(
    observations: tuple[ForwardProgressObservationV1, ...],
) -> None:
    """Require one status and value per state within one frozen metric/evaluator.

    Roles and receipt paths do not create new states.  A repeated authoritative
    receipt or another receipt for the same state must preserve its observation.
    """

    by_state: dict[str, tuple[UsageValueStatus, float | None]] = {}
    for observation in observations:
        semantic_value = (observation.value_status, observation.value)
        previous = by_state.setdefault(observation.state_id, semantic_value)
        if previous != semantic_value:
            raise ValueError("same-state progress observations must agree on status and value")


def _validate_progress_observation_structure(
    observations: tuple[ForwardProgressObservationV1, ...],
) -> tuple[ForwardProgressObservationV1, ForwardProgressObservationV1]:
    """Require unique ordered state roles and return the named progress endpoints."""

    if type(observations) is not tuple or len(observations) < 2 or any(
        type(value) is not ForwardProgressObservationV1 for value in observations
    ):
        raise ValueError("progress requires at least two typed observations")
    if (
        observations[0].state_role is not ForwardProgressStateRole.FAILED_START
        or observations[-1].state_role is not ForwardProgressStateRole.FINAL
        or any(
            value.state_role is not ForwardProgressStateRole.TRANSITION_AFTER
            for value in observations[1:-1]
        )
    ):
        raise ValueError("progress observations require ordered start and final roles")
    timepoints = tuple(value.timepoint for value in observations)
    if len(set(timepoints)) != len(timepoints):
        raise ValueError("progress observations cannot repeat a timepoint")
    state_bindings = tuple(
        (value.state_role, value.transition_id) for value in observations
    )
    if len(set(state_bindings)) != len(state_bindings):
        raise ValueError("progress observations cannot reuse a frozen state role")
    _consistent_progress_states(observations)
    # Select endpoint states by their validated roles, not by caller-chosen slots.
    failed_start = next(
        value for value in observations
        if value.state_role is ForwardProgressStateRole.FAILED_START
    )
    final = next(
        value for value in observations
        if value.state_role is ForwardProgressStateRole.FINAL
    )
    return failed_start, final


@dataclass(frozen=True)
class ForwardProgressReceiptV1:
    """Bind ordered progress observations to one execution and evaluator."""

    receipt_id: str
    run_id: str
    work_item_id: str
    source_task_id: str
    case_id: str
    execution_receipt_id: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    progress_contract_id: str
    metric_id: str
    metric_version: str
    evaluator_contract_id: str
    evaluator_id: str
    evaluator_version: str
    observations: tuple[ForwardProgressObservationV1, ...]
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_PROGRESS_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_PROGRESS_RECEIPT_SCHEMA:
            raise ValueError("unsupported forward progress receipt schema")
        for label, value in (
            ("receipt_id", self.receipt_id),
            ("run_id", self.run_id),
            ("work_item_id", self.work_item_id),
            ("case_id", self.case_id),
            ("execution_receipt_id", self.execution_receipt_id),
            ("benchmark_id", self.benchmark_id),
            ("benchmark_version", self.benchmark_version),
            ("adapter_version", self.adapter_version),
            ("progress_contract_id", self.progress_contract_id),
            ("metric_id", self.metric_id),
            ("metric_version", self.metric_version),
            ("evaluator_contract_id", self.evaluator_contract_id),
            ("evaluator_id", self.evaluator_id),
            ("evaluator_version", self.evaluator_version),
        ):
            _identifier(value, label)
        _text(self.source_task_id, "source_task_id")
        _validate_progress_observation_structure(self.observations)
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "source_task_id": self.source_task_id,
            "case_id": self.case_id,
            "execution_receipt_id": self.execution_receipt_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "progress_contract_id": self.progress_contract_id,
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "evaluator_contract_id": self.evaluator_contract_id,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "observations": [value.to_dict() for value in self.observations],
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ForwardFalseRepairOraclePolicyV1:
    """Freeze the independent criterion and evaluator used for false repair."""

    oracle_contract_id: str
    oracle_version: str
    outcome_field: str
    oracle_source_ref: str
    criterion_id: str
    criterion_version: str
    evaluator_id: str
    evaluator_version: str
    evaluator_source_ref: str
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_FALSE_REPAIR_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_FALSE_REPAIR_POLICY_SCHEMA:
            raise ValueError("unsupported false-repair oracle policy schema")
        for label in (
            "oracle_contract_id",
            "oracle_version",
            "outcome_field",
            "criterion_id",
            "criterion_version",
            "evaluator_id",
            "evaluator_version",
        ):
            _identifier(getattr(self, label), label)
        _text(self.oracle_source_ref, "false-repair oracle_source_ref")
        _text(self.evaluator_source_ref, "false-repair evaluator_source_ref")
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the frozen criterion and evaluator identity."""

        return {
            "schema": self.schema,
            "oracle_contract_id": self.oracle_contract_id,
            "oracle_version": self.oracle_version,
            "outcome_field": self.outcome_field,
            "oracle_source_ref": self.oracle_source_ref,
            "criterion_id": self.criterion_id,
            "criterion_version": self.criterion_version,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "evaluator_source_ref": self.evaluator_source_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ForwardRegressionOraclePolicyV1:
    """Freeze one regression evaluator and its complete protected-item roster."""

    oracle_contract_id: str
    oracle_version: str
    outcome_field: str
    oracle_source_ref: str
    kind: OracleKind
    protected_item_ids: tuple[str, ...]
    protected_roster_ref: str
    evaluator_id: str
    evaluator_version: str
    evaluator_source_ref: str
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_REGRESSION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_REGRESSION_POLICY_SCHEMA:
            raise ValueError("unsupported regression oracle policy schema")
        for label in (
            "oracle_contract_id",
            "oracle_version",
            "outcome_field",
            "evaluator_id",
            "evaluator_version",
        ):
            _identifier(getattr(self, label), label)
        if type(self.kind) is not OracleKind or self.kind not in {
            OracleKind.TRANSIENT_REGRESSION,
            OracleKind.FINAL_REGRESSION,
        }:
            raise ValueError("regression policy kind must be transient or final")
        _text_tuple(
            self.protected_item_ids,
            "protected_item_ids",
            unique=True,
            nonempty=True,
        )
        _text(self.oracle_source_ref, "regression oracle_source_ref")
        _text(self.protected_roster_ref, "protected_roster_ref")
        _text(self.evaluator_source_ref, "regression evaluator_source_ref")
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the endpoint contract and the exact protected roster."""

        return {
            "schema": self.schema,
            "oracle_contract_id": self.oracle_contract_id,
            "oracle_version": self.oracle_version,
            "outcome_field": self.outcome_field,
            "oracle_source_ref": self.oracle_source_ref,
            "kind": self.kind.value,
            "protected_item_ids": list(self.protected_item_ids),
            "protected_roster_ref": self.protected_roster_ref,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "evaluator_source_ref": self.evaluator_source_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


def _validate_binary_oracle_fields(
    *,
    status: ForwardOracleReceiptStatus,
    occurred: bool | None,
    evaluator_receipt_ref: str | None,
    unavailable_reason: str | None,
    unavailable_evidence_ref: str | None,
    label: str,
) -> None:
    """Apply one availability rule to independently evaluated booleans."""

    if not isinstance(status, ForwardOracleReceiptStatus):
        raise ValueError(f"{label} status must use its closed enum")
    _optional_text(evaluator_receipt_ref, f"{label} evaluator_receipt_ref")
    _optional_text(unavailable_reason, f"{label} unavailable_reason")
    _optional_text(unavailable_evidence_ref, f"{label} unavailable_evidence_ref")
    if status is ForwardOracleReceiptStatus.AVAILABLE:
        if type(occurred) is not bool or evaluator_receipt_ref is None:
            raise ValueError(f"available {label} requires a boolean and evaluator receipt")
        if unavailable_reason is not None or unavailable_evidence_ref is not None:
            raise ValueError(f"available {label} cannot carry unavailability data")
    else:
        if occurred is not None:
            raise ValueError(f"unavailable {label} must keep the outcome null")
        if unavailable_reason is None or unavailable_evidence_ref is None:
            raise ValueError(f"unavailable {label} requires independent evidence")


@dataclass(frozen=True)
class FalseRepairOracleReceiptV1:
    """Store one independent false-repair decision without trusting the method."""

    receipt_id: str
    run_id: str
    work_item_id: str
    source_task_id: str
    case_id: str
    execution_receipt_id: str
    final_state_id: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    oracle_contract_id: str
    oracle_version: str
    outcome_field: str
    criterion_id: str
    criterion_version: str
    evaluator_id: str
    evaluator_version: str
    status: ForwardOracleReceiptStatus
    false_repair_occurred: bool | None
    evaluator_receipt_ref: str | None
    unavailable_reason: str | None = None
    unavailable_evidence_ref: str | None = None
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_FALSE_REPAIR_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_FALSE_REPAIR_RECEIPT_SCHEMA:
            raise ValueError("unsupported false-repair oracle receipt schema")
        for label, value in (
            ("receipt_id", self.receipt_id),
            ("run_id", self.run_id),
            ("work_item_id", self.work_item_id),
            ("case_id", self.case_id),
            ("execution_receipt_id", self.execution_receipt_id),
            ("benchmark_id", self.benchmark_id),
            ("benchmark_version", self.benchmark_version),
            ("adapter_version", self.adapter_version),
            ("oracle_contract_id", self.oracle_contract_id),
            ("oracle_version", self.oracle_version),
            ("outcome_field", self.outcome_field),
            ("criterion_id", self.criterion_id),
            ("criterion_version", self.criterion_version),
            ("evaluator_id", self.evaluator_id),
            ("evaluator_version", self.evaluator_version),
        ):
            _identifier(value, label)
        _text(self.source_task_id, "source_task_id")
        _text(self.final_state_id, "false-repair final_state_id")
        _validate_binary_oracle_fields(
            status=self.status,
            occurred=self.false_repair_occurred,
            evaluator_receipt_ref=self.evaluator_receipt_ref,
            unavailable_reason=self.unavailable_reason,
            unavailable_evidence_ref=self.unavailable_evidence_ref,
            label="false repair",
        )
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "source_task_id": self.source_task_id,
            "case_id": self.case_id,
            "execution_receipt_id": self.execution_receipt_id,
            "final_state_id": self.final_state_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "oracle_contract_id": self.oracle_contract_id,
            "oracle_version": self.oracle_version,
            "outcome_field": self.outcome_field,
            "criterion_id": self.criterion_id,
            "criterion_version": self.criterion_version,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "status": self.status.value,
            "false_repair_occurred": self.false_repair_occurred,
            "evaluator_receipt_ref": self.evaluator_receipt_ref,
            "unavailable_reason": self.unavailable_reason,
            "unavailable_evidence_ref": self.unavailable_evidence_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ForwardProtectedItemObservationV1:
    """Bind one protected-item judgment to an exact executed state."""

    protected_item_id: str
    state_role: ForwardProgressStateRole
    state_id: str
    transition_id: str | None
    evaluator_id: str
    evaluator_version: str
    satisfied: bool
    evaluator_receipt_ref: str
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_PROTECTED_ITEM_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_PROTECTED_ITEM_OBSERVATION_SCHEMA:
            raise ValueError("unsupported protected-item observation schema")
        for label, value in (
            ("protected_item_id", self.protected_item_id),
            ("evaluator_id", self.evaluator_id),
            ("evaluator_version", self.evaluator_version),
        ):
            _identifier(value, label)
        _text(self.state_id, "protected-item state_id")
        _optional_text(self.transition_id, "protected-item transition_id")
        _text(self.evaluator_receipt_ref, "protected-item evaluator_receipt_ref")
        if not isinstance(self.state_role, ForwardProgressStateRole):
            raise ValueError("protected-item state_role must use its closed enum")
        if (self.state_role is ForwardProgressStateRole.TRANSITION_AFTER) != (
            self.transition_id is not None
        ):
            raise ValueError("only transition observations may name a transition_id")
        if type(self.satisfied) is not bool:
            raise ValueError("protected-item satisfaction must be boolean")
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protected_item_id": self.protected_item_id,
            "state_role": self.state_role.value,
            "state_id": self.state_id,
            "transition_id": self.transition_id,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "satisfied": self.satisfied,
            "evaluator_receipt_ref": self.evaluator_receipt_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ForwardProtectedItemHistoryV1:
    """Keep one ordered, typed state history for a protected item."""

    protected_item_id: str
    observations: tuple[ForwardProtectedItemObservationV1, ...]
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_PROTECTED_ITEM_HISTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_PROTECTED_ITEM_HISTORY_SCHEMA:
            raise ValueError("unsupported protected-item history schema")
        _identifier(self.protected_item_id, "protected_item_id")
        if type(self.observations) is not tuple or len(self.observations) < 2 or any(
            type(value) is not ForwardProtectedItemObservationV1
            for value in self.observations
        ):
            raise ValueError("protected-item history requires typed endpoint observations")
        if any(
            value.protected_item_id != self.protected_item_id
            for value in self.observations
        ):
            raise ValueError("protected-item observations must preserve the item identity")
        if (
            self.observations[0].state_role is not ForwardProgressStateRole.FAILED_START
            or self.observations[-1].state_role is not ForwardProgressStateRole.FINAL
            or any(
                value.state_role is not ForwardProgressStateRole.TRANSITION_AFTER
                for value in self.observations[1:-1]
            )
        ):
            raise ValueError("protected-item history must be failed-start, transitions, final")
        state_bindings = tuple(
            (value.state_role, value.state_id, value.transition_id)
            for value in self.observations
        )
        if len(set(state_bindings)) != len(state_bindings):
            raise ValueError("protected-item history cannot repeat a state binding")
        # Roles may alias one state; the same evaluator must preserve its judgment.
        state_judgments: dict[tuple[str, str, str, str], bool] = {}
        for observation in self.observations:
            key = (
                observation.state_id,
                observation.protected_item_id,
                observation.evaluator_id,
                observation.evaluator_version,
            )
            if key in state_judgments and state_judgments[key] is not observation.satisfied:
                raise ValueError("one protected state cannot have conflicting satisfaction")
            state_judgments[key] = observation.satisfied
        if self.observations[0].satisfied is not True:
            raise ValueError("protected items must be satisfied at failed start")
        _local_status(self.design_origin, self.implementation_status)

    @property
    def transient_regression(self) -> bool:
        return (
            any(value.satisfied is False for value in self.observations[1:-1])
            and self.observations[-1].satisfied
        )

    @property
    def final_regression(self) -> bool:
        return not self.observations[-1].satisfied

    @property
    def state_bindings(
        self,
    ) -> tuple[tuple[ForwardProgressStateRole, str, str | None], ...]:
        return tuple(
            (value.state_role, value.state_id, value.transition_id)
            for value in self.observations
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protected_item_id": self.protected_item_id,
            "observations": [value.to_dict() for value in self.observations],
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class RegressionOracleReceiptV1:
    """Store one shared history from which both regression endpoints derive."""

    receipt_id: str
    run_id: str
    work_item_id: str
    source_task_id: str
    case_id: str
    execution_receipt_id: str
    final_state_id: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    evaluator_id: str
    evaluator_version: str
    protected_item_histories: tuple[ForwardProtectedItemHistoryV1, ...]
    protected_item_history_receipt_ref: str | None
    status: ForwardOracleReceiptStatus
    unavailable_reason: str | None = None
    unavailable_evidence_ref: str | None = None
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_REGRESSION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_REGRESSION_RECEIPT_SCHEMA:
            raise ValueError("unsupported regression oracle receipt schema")
        for label, value in (
            ("receipt_id", self.receipt_id),
            ("run_id", self.run_id),
            ("work_item_id", self.work_item_id),
            ("case_id", self.case_id),
            ("execution_receipt_id", self.execution_receipt_id),
            ("benchmark_id", self.benchmark_id),
            ("benchmark_version", self.benchmark_version),
            ("adapter_version", self.adapter_version),
            ("evaluator_id", self.evaluator_id),
            ("evaluator_version", self.evaluator_version),
        ):
            _identifier(value, label)
        _text(self.source_task_id, "source_task_id")
        _text(self.final_state_id, "regression final_state_id")
        if not isinstance(self.status, ForwardOracleReceiptStatus):
            raise ValueError("regression receipt status must use its closed enum")
        if type(self.protected_item_histories) is not tuple or any(
            type(value) is not ForwardProtectedItemHistoryV1
            for value in self.protected_item_histories
        ):
            raise ValueError("protected_item_histories must contain typed histories")
        protected_item_ids = tuple(
            value.protected_item_id for value in self.protected_item_histories
        )
        if len(set(protected_item_ids)) != len(
            protected_item_ids
        ):
            raise ValueError("regression receipt cannot repeat a protected item")
        _optional_text(
            self.protected_item_history_receipt_ref,
            "protected_item_history_receipt_ref",
        )
        _optional_text(self.unavailable_reason, "regression unavailable_reason")
        _optional_text(
            self.unavailable_evidence_ref,
            "regression unavailable_evidence_ref",
        )
        if self.status is ForwardOracleReceiptStatus.AVAILABLE:
            if (
                not self.protected_item_histories
                or self.protected_item_history_receipt_ref is None
            ):
                raise ValueError(
                    "available regression requires raw protected-item histories"
                )
            if self.unavailable_reason is not None or self.unavailable_evidence_ref is not None:
                raise ValueError("available regression cannot carry unavailability data")
            expected_bindings = self.protected_item_histories[0].state_bindings
            if any(
                history.state_bindings != expected_bindings
                for history in self.protected_item_histories[1:]
            ):
                raise ValueError("all protected items must share one exact state history")
            if any(
                observation.evaluator_id != self.evaluator_id
                or observation.evaluator_version != self.evaluator_version
                for history in self.protected_item_histories
                for observation in history.observations
            ):
                raise ValueError("all state observations must use the receipt evaluator")
        else:
            if self.protected_item_histories or self.protected_item_history_receipt_ref is not None:
                raise ValueError(
                    "unavailable regression cannot claim complete protected-item histories"
                )
            if self.unavailable_reason is None or self.unavailable_evidence_ref is None:
                raise ValueError("unavailable regression requires independent evidence")
        _local_status(self.design_origin, self.implementation_status)

    @property
    def transient_regression_occurred(self) -> bool | None:
        if self.status is ForwardOracleReceiptStatus.UNAVAILABLE:
            return None
        return any(value.transient_regression for value in self.protected_item_histories)

    @property
    def final_regression_occurred(self) -> bool | None:
        if self.status is ForwardOracleReceiptStatus.UNAVAILABLE:
            return None
        return any(value.final_regression for value in self.protected_item_histories)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "source_task_id": self.source_task_id,
            "case_id": self.case_id,
            "execution_receipt_id": self.execution_receipt_id,
            "final_state_id": self.final_state_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "protected_item_histories": [
                value.to_dict() for value in self.protected_item_histories
            ],
            "protected_item_history_receipt_ref": (
                self.protected_item_history_receipt_ref
            ),
            "status": self.status.value,
            "transient_regression_occurred": self.transient_regression_occurred,
            "final_regression_occurred": self.final_regression_occurred,
            "unavailable_reason": self.unavailable_reason,
            "unavailable_evidence_ref": self.unavailable_evidence_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ForwardRootGoldBindingV1:
    """Bind scorer-only attribution gold to one frozen population and case."""

    receipt_id: str
    population_id: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    case_id: str
    source_task_id: str
    split: BenchmarkSplit
    gold_authority_receipt_refs: tuple[str, ...]
    population_binding_source_ref: str
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_ROOT_GOLD_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_ROOT_GOLD_BINDING_SCHEMA:
            raise ValueError("unsupported forward root-gold binding schema")
        for label, value in (
            ("receipt_id", self.receipt_id),
            ("population_id", self.population_id),
            ("benchmark_id", self.benchmark_id),
            ("benchmark_version", self.benchmark_version),
            ("adapter_version", self.adapter_version),
            ("case_id", self.case_id),
        ):
            _identifier(value, label)
        _text(self.source_task_id, "source_task_id")
        if not isinstance(self.split, BenchmarkSplit):
            raise ValueError("root-gold split must use the closed enum")
        _text_tuple(
            self.gold_authority_receipt_refs,
            "gold_authority_receipt_refs",
            unique=True,
            nonempty=True,
        )
        _text(self.population_binding_source_ref, "population_binding_source_ref")
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "population_id": self.population_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "case_id": self.case_id,
            "source_task_id": self.source_task_id,
            "split": self.split.value,
            "gold_authority_receipt_refs": list(self.gold_authority_receipt_refs),
            "population_binding_source_ref": self.population_binding_source_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ForwardEvidenceRefClosureV1:
    """Preserve the exact evidence closure used by one scorer decision."""

    execution_refs: tuple[str, ...]
    official_refs: tuple[str, ...]
    budget_refs: tuple[str, ...]
    progress_refs: tuple[str, ...]
    oracle_refs: tuple[str, ...]
    root_gold_refs: tuple[str, ...]
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_EVIDENCE_REF_CLOSURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_EVIDENCE_REF_CLOSURE_SCHEMA:
            raise ValueError("unsupported forward evidence-ref closure schema")
        for label, values in (
            ("execution_refs", self.execution_refs),
            ("official_refs", self.official_refs),
            ("budget_refs", self.budget_refs),
            ("progress_refs", self.progress_refs),
            ("oracle_refs", self.oracle_refs),
            ("root_gold_refs", self.root_gold_refs),
        ):
            _text_tuple(
                values,
                label,
                unique=True,
                nonempty=label == "execution_refs",
            )
        _local_status(self.design_origin, self.implementation_status)

    @property
    def raw_evidence_refs(self) -> tuple[str, ...]:
        return _dedupe_refs(
            self.execution_refs,
            self.official_refs,
            self.budget_refs,
            self.progress_refs,
            self.oracle_refs,
            self.root_gold_refs,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "execution_refs": list(self.execution_refs),
            "official_refs": list(self.official_refs),
            "budget_refs": list(self.budget_refs),
            "progress_refs": list(self.progress_refs),
            "oracle_refs": list(self.oracle_refs),
            "root_gold_refs": list(self.root_gold_refs),
            "raw_evidence_refs": list(self.raw_evidence_refs),
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ForwardRecoveryCaseScoreV1:
    """Store one provisional pure decision with every endpoint kept separate."""

    case_id: str
    source_task_id: str
    run_id: str
    population_id: str
    registry_id: str
    scorer_id: str
    scorer_version: str
    work_item_id: str
    execution_receipt_id: str
    official_outcome_receipt_id: str | None
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    system_cell_id: str
    treatment_id: str
    locator_treatment_id: str
    fixer_treatment_id: str
    repetition_id: str
    budget_id: str
    split: BenchmarkSplit
    execution_semantics: ExecutionSemantics
    decision_status: ForwardDecisionStatus
    invalid_reason: ForwardInvalidReason | None
    stop_reason: ForwardStopReason
    official_failure_resolved: bool | None
    fixed_denominator_failure_resolved: bool | None
    budget_evidence_status: ForwardEvidenceStatus
    budget_overall_decision: BudgetOverallDecision | None
    failure_resolved_at_budget: bool | None
    progress_evidence_status: ForwardEvidenceStatus
    progress_contract_id: str | None
    progress_observations: tuple[ForwardProgressObservationV1, ...]
    directed_progress_change: float | None
    root_gold_evidence_status: ForwardEvidenceStatus
    root_identified: bool | None
    root_verification_evidence_status: ForwardEvidenceStatus
    strict_root_verification_observed: bool | None
    verified_root: bool | None
    false_repair_evidence_status: ForwardEvidenceStatus
    false_repair_occurred: bool | None
    transient_regression_evidence_status: ForwardEvidenceStatus
    transient_regression_occurred: bool | None
    final_regression_evidence_status: ForwardEvidenceStatus
    final_regression_occurred: bool | None
    costs: tuple[ResourceCostV1, ...]
    evidence_ref_closure: ForwardEvidenceRefClosureV1
    raw_evidence_refs: tuple[str, ...]
    method_failure: bool
    infrastructure_failure: bool
    incomplete: bool
    track: TraceFixTrack = TraceFixTrack.FORWARD_RECOVERY
    suite_id: str = TRACEFIX_BENCH_SUITE_ID
    suite_version: str = SCORER_REGISTRY_SUITE_VERSION_V0_1
    input_binding_status: str = FORWARD_INPUT_BINDING_STATUS
    design_origin: str = FORWARD_DESIGN_ORIGIN
    implementation_status: str = FORWARD_IMPLEMENTATION_STATUS
    schema: str = FORWARD_CASE_SCORE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORWARD_CASE_SCORE_SCHEMA:
            raise ValueError("unsupported forward case-score schema")
        for label, value in (
            ("case_id", self.case_id),
            ("run_id", self.run_id),
            ("population_id", self.population_id),
            ("registry_id", self.registry_id),
            ("scorer_id", self.scorer_id),
            ("scorer_version", self.scorer_version),
            ("work_item_id", self.work_item_id),
            ("execution_receipt_id", self.execution_receipt_id),
            ("benchmark_id", self.benchmark_id),
            ("benchmark_version", self.benchmark_version),
            ("adapter_version", self.adapter_version),
            ("system_cell_id", self.system_cell_id),
            ("treatment_id", self.treatment_id),
            ("locator_treatment_id", self.locator_treatment_id),
            ("fixer_treatment_id", self.fixer_treatment_id),
            ("repetition_id", self.repetition_id),
            ("budget_id", self.budget_id),
        ):
            _identifier(value, label)
        _text(self.source_task_id, "source_task_id")
        _optional_text(self.official_outcome_receipt_id, "official_outcome_receipt_id")
        if self.track is not TraceFixTrack.FORWARD_RECOVERY:
            raise ValueError("forward score must use the forward track")
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID:
            raise ValueError("forward score suite_id differs")
        if self.suite_version != SCORER_REGISTRY_SUITE_VERSION_V0_1:
            raise ValueError("forward score suite_version differs")
        if self.input_binding_status != FORWARD_INPUT_BINDING_STATUS:
            raise ValueError("pure forward score must remain provisional_unbound")
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError("execution_semantics must use its closed enum")
        if not isinstance(self.split, BenchmarkSplit):
            raise ValueError("split must use the closed BenchmarkSplit enum")
        if not isinstance(self.decision_status, ForwardDecisionStatus):
            raise ValueError("decision_status must use its closed enum")
        if self.invalid_reason is not None and not isinstance(
            self.invalid_reason, ForwardInvalidReason
        ):
            raise ValueError("invalid_reason must use its closed enum or null")
        if not isinstance(self.stop_reason, ForwardStopReason):
            raise ValueError("stop_reason must use its closed enum")
        if self.budget_overall_decision is not None and not isinstance(
            self.budget_overall_decision, BudgetOverallDecision
        ):
            raise ValueError("budget_overall_decision must use its closed enum or null")
        if not isinstance(self.budget_evidence_status, ForwardEvidenceStatus):
            raise ValueError("budget_evidence_status must use its closed enum")
        for label, value in (
            ("official_failure_resolved", self.official_failure_resolved),
            (
                "fixed_denominator_failure_resolved",
                self.fixed_denominator_failure_resolved,
            ),
            ("failure_resolved_at_budget", self.failure_resolved_at_budget),
            ("root_identified", self.root_identified),
            ("strict_root_verification_observed", self.strict_root_verification_observed),
            ("verified_root", self.verified_root),
            ("false_repair_occurred", self.false_repair_occurred),
            ("transient_regression_occurred", self.transient_regression_occurred),
            ("final_regression_occurred", self.final_regression_occurred),
        ):
            if value is not None and type(value) is not bool:
                raise ValueError(f"{label} must be boolean or null")
        # No current receipt can establish root verification, even for direct callers.
        if self.strict_root_verification_observed is not None or self.verified_root is not None:
            raise ValueError("forward root verification must remain unavailable")
        evidence_pairs = (
            (self.root_gold_evidence_status, self.root_identified, "root gold"),
            (
                self.root_verification_evidence_status,
                self.strict_root_verification_observed,
                "root verification",
            ),
            (
                self.false_repair_evidence_status,
                self.false_repair_occurred,
                "false repair",
            ),
            (
                self.transient_regression_evidence_status,
                self.transient_regression_occurred,
                "transient regression",
            ),
            (
                self.final_regression_evidence_status,
                self.final_regression_occurred,
                "final regression",
            ),
        )
        for status, value, label in evidence_pairs:
            if not isinstance(status, ForwardEvidenceStatus):
                raise ValueError(f"{label} evidence status must use its closed enum")
            if (status is ForwardEvidenceStatus.REGISTERED_AVAILABLE) != (
                value is not None
            ):
                raise ValueError(f"{label} availability contradicts its value")
        if not isinstance(self.progress_evidence_status, ForwardEvidenceStatus):
            raise ValueError("progress evidence status must use its closed enum")
        _optional_text(self.progress_contract_id, "progress_contract_id")
        if type(self.progress_observations) is not tuple:
            raise ValueError("progress observations must be a tuple")
        progress_endpoints = (
            _validate_progress_observation_structure(self.progress_observations)
            if self.progress_observations
            else None
        )
        if self.progress_evidence_status is ForwardEvidenceStatus.REGISTERED_AVAILABLE:
            if self.progress_contract_id is None or not self.progress_observations:
                raise ValueError("available progress requires contract and observations")
            if any(
                value.value_status is UsageValueStatus.UNAVAILABLE
                for value in self.progress_observations
            ):
                raise ValueError("available progress cannot include an unavailable value")
            if self.directed_progress_change is None:
                raise ValueError("available progress requires the frozen directed change")
        elif self.progress_evidence_status is ForwardEvidenceStatus.REGISTERED_UNAVAILABLE:
            if self.progress_contract_id is None or not self.progress_observations:
                raise ValueError("registered unavailable progress requires its raw observations")
            if not any(
                value.value_status is UsageValueStatus.UNAVAILABLE
                for value in self.progress_observations
            ):
                raise ValueError("registered unavailable progress requires an unavailable value")
            if self.directed_progress_change is not None:
                raise ValueError("unavailable progress cannot carry directed change")
        else:
            if (
                self.progress_contract_id is not None
                or self.progress_observations
                or self.directed_progress_change is not None
            ):
                raise ValueError(
                    "unregistered, failed, or invalid progress cannot carry partial payloads"
                )
        if self.directed_progress_change is not None:
            _finite(self.directed_progress_change, "directed_progress_change")
            if progress_endpoints is None:
                raise ValueError("directed progress requires both endpoint observations")
            failed_start, final = progress_endpoints
            if (
                failed_start.state_id == final.state_id
                and self.directed_progress_change != 0
            ):
                raise ValueError("identical progress endpoint states require zero change")
        _costs(self.costs)
        if type(self.evidence_ref_closure) is not ForwardEvidenceRefClosureV1:
            raise ValueError("evidence_ref_closure must use its typed schema")
        _text_tuple(
            self.raw_evidence_refs,
            "raw_evidence_refs",
            unique=True,
            nonempty=True,
        )
        if self.raw_evidence_refs != self.evidence_ref_closure.raw_evidence_refs:
            raise ValueError("raw evidence refs must equal the exact nested evidence closure")
        if self.execution_receipt_id not in self.evidence_ref_closure.execution_refs:
            raise ValueError("execution evidence refs must include the execution receipt")
        if (
            self.official_outcome_receipt_id is not None
            and self.official_outcome_receipt_id
            not in self.evidence_ref_closure.official_refs
        ):
            raise ValueError("official evidence refs must include the outcome receipt")
        cost_refs = {
            reference
            for cost in self.costs
            for reference in (cost.receipt_ref, cost.unavailable_evidence_ref)
            if reference is not None
        }
        if not cost_refs <= set(self.evidence_ref_closure.execution_refs):
            raise ValueError("execution evidence refs must include every cost receipt")
        progress_observation_refs = {
            reference
            for observation in self.progress_observations
            for reference in (
                observation.evaluator_receipt_ref,
                observation.unavailable_evidence_ref,
            )
            if reference is not None
        }
        if not progress_observation_refs <= set(self.evidence_ref_closure.progress_refs):
            raise ValueError("progress evidence refs must include every observation receipt")
        for label, value in (
            ("method_failure", self.method_failure),
            ("infrastructure_failure", self.infrastructure_failure),
            ("incomplete", self.incomplete),
        ):
            if type(value) is not bool:
                raise ValueError(f"{label} must be boolean")
        scored = self.decision_status is ForwardDecisionStatus.SCORED
        method_failed = self.decision_status is ForwardDecisionStatus.METHOD_FAILED
        if scored:
            if self.root_verification_evidence_status not in {
                ForwardEvidenceStatus.UNREGISTERED,
                ForwardEvidenceStatus.REGISTERED_UNAVAILABLE,
            }:
                raise ValueError("scored root verification must be unregistered or unavailable")
            if (
                self.invalid_reason is not None
                or self.official_failure_resolved is None
                or self.official_outcome_receipt_id is None
            ):
                raise ValueError("scored forward case requires one official decision")
            if (
                self.fixed_denominator_failure_resolved
                is not self.official_failure_resolved
            ):
                raise ValueError("completed work must preserve its official fixed-denominator value")
            if self.method_failure or self.infrastructure_failure or self.incomplete:
                raise ValueError("scored forward case cannot carry failure flags")
            if self.failure_resolved_at_budget is not None or self.verified_root is not None:
                raise ValueError("unbound forward scores cannot publish budget-qualified endpoints")
            if self.budget_evidence_status is ForwardEvidenceStatus.REGISTERED_AVAILABLE:
                if self.budget_overall_decision is None or not self.evidence_ref_closure.budget_refs:
                    raise ValueError("available budget evidence requires a decision and refs")
            elif self.budget_evidence_status in {
                ForwardEvidenceStatus.UNREGISTERED,
                ForwardEvidenceStatus.REGISTERED_UNAVAILABLE,
            }:
                if self.budget_overall_decision is not None:
                    raise ValueError("unavailable budget evidence cannot carry a decision")
                if (
                    self.budget_evidence_status is ForwardEvidenceStatus.UNREGISTERED
                    and self.evidence_ref_closure.budget_refs
                ):
                    raise ValueError("unregistered budget evidence cannot carry budget refs")
            else:
                raise ValueError("scored cases require a non-failure budget evidence status")
            if (
                self.progress_evidence_status is ForwardEvidenceStatus.UNREGISTERED
            ) != (not self.evidence_ref_closure.progress_refs):
                raise ValueError("progress registration must match its evidence-ref group")
            if (
                self.root_gold_evidence_status is ForwardEvidenceStatus.UNREGISTERED
            ) != (not self.evidence_ref_closure.root_gold_refs):
                raise ValueError("root-gold registration must match its evidence-ref group")
        elif method_failed:
            if (
                self.invalid_reason is not ForwardInvalidReason.EXECUTION_METHOD_FAILURE
                or self.official_failure_resolved is not None
                or self.fixed_denominator_failure_resolved is not False
                or self.failure_resolved_at_budget is not None
                or self.verified_root is not None
                or self.budget_overall_decision is not None
                or self.budget_evidence_status
                is not ForwardEvidenceStatus.METHOD_NOT_EVALUABLE
            ):
                raise ValueError("method failure must be unresolved only in the fixed denominator")
            if any(
                status is not ForwardEvidenceStatus.METHOD_NOT_EVALUABLE
                for status in (
                    self.progress_evidence_status,
                    self.root_gold_evidence_status,
                    self.root_verification_evidence_status,
                    self.false_repair_evidence_status,
                    self.transient_regression_evidence_status,
                    self.final_regression_evidence_status,
                )
            ):
                raise ValueError("method failure endpoints must remain not evaluable")
        else:
            if self.invalid_reason is None:
                raise ValueError("non-scored forward decision requires a closed reason")
            endpoints = (
                self.official_failure_resolved,
                self.fixed_denominator_failure_resolved,
                self.failure_resolved_at_budget,
                self.directed_progress_change,
                self.root_identified,
                self.strict_root_verification_observed,
                self.verified_root,
                self.false_repair_occurred,
                self.transient_regression_occurred,
                self.final_regression_occurred,
            )
            if any(value is not None for value in endpoints):
                raise ValueError("non-scored decisions must keep every endpoint null")
            if self.budget_overall_decision is not None:
                raise ValueError("non-scored decisions cannot claim budget qualification")
            if self.budget_evidence_status is not ForwardEvidenceStatus.INVALID:
                raise ValueError("non-scored decisions must mark budget evidence invalid")
            if any(
                status is not ForwardEvidenceStatus.INVALID
                for status in (
                    self.progress_evidence_status,
                    self.root_gold_evidence_status,
                    self.root_verification_evidence_status,
                    self.false_repair_evidence_status,
                    self.transient_regression_evidence_status,
                    self.final_regression_evidence_status,
                )
            ):
                raise ValueError("non-scored decisions must mark endpoints invalid")
        if self.method_failure != (
            self.decision_status is ForwardDecisionStatus.METHOD_FAILED
        ):
            raise ValueError("method_failure flag must follow final decision")
        infra = self.decision_status is ForwardDecisionStatus.INFRASTRUCTURE_INVALID
        ambiguous = self.decision_status is ForwardDecisionStatus.AMBIGUOUS_RESUME
        if self.infrastructure_failure != infra or self.incomplete != (infra or ambiguous):
            raise ValueError("infrastructure and incomplete flags must follow decision")
        _local_status(self.design_origin, self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "track": self.track.value,
            "input_binding_status": self.input_binding_status,
            "case_id": self.case_id,
            "source_task_id": self.source_task_id,
            "run_id": self.run_id,
            "population_id": self.population_id,
            "registry_id": self.registry_id,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "work_item_id": self.work_item_id,
            "execution_receipt_id": self.execution_receipt_id,
            "official_outcome_receipt_id": self.official_outcome_receipt_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "system_cell_id": self.system_cell_id,
            "treatment_id": self.treatment_id,
            "locator_treatment_id": self.locator_treatment_id,
            "fixer_treatment_id": self.fixer_treatment_id,
            "repetition_id": self.repetition_id,
            "budget_id": self.budget_id,
            "split": self.split.value,
            "execution_semantics": self.execution_semantics.value,
            "decision_status": self.decision_status.value,
            "invalid_reason": self.invalid_reason.value if self.invalid_reason else None,
            "stop_reason": self.stop_reason.value,
            "official_failure_resolved": self.official_failure_resolved,
            "fixed_denominator_failure_resolved": (
                self.fixed_denominator_failure_resolved
            ),
            "budget_evidence_status": self.budget_evidence_status.value,
            "budget_overall_decision": (
                self.budget_overall_decision.value
                if self.budget_overall_decision is not None
                else None
            ),
            "failure_resolved_at_budget": self.failure_resolved_at_budget,
            "progress_evidence_status": self.progress_evidence_status.value,
            "progress_contract_id": self.progress_contract_id,
            "progress_observations": [
                value.to_dict() for value in self.progress_observations
            ],
            "directed_progress_change": self.directed_progress_change,
            "root_gold_evidence_status": self.root_gold_evidence_status.value,
            "root_identified": self.root_identified,
            "root_verification_evidence_status": (
                self.root_verification_evidence_status.value
            ),
            "strict_root_verification_observed": (
                self.strict_root_verification_observed
            ),
            "verified_root": self.verified_root,
            "false_repair_evidence_status": self.false_repair_evidence_status.value,
            "false_repair_occurred": self.false_repair_occurred,
            "transient_regression_evidence_status": (
                self.transient_regression_evidence_status.value
            ),
            "transient_regression_occurred": self.transient_regression_occurred,
            "final_regression_evidence_status": (
                self.final_regression_evidence_status.value
            ),
            "final_regression_occurred": self.final_regression_occurred,
            "costs": [value.to_dict() for value in self.costs],
            "evidence_ref_closure": self.evidence_ref_closure.to_dict(),
            "raw_evidence_refs": list(self.raw_evidence_refs),
            "method_failure": self.method_failure,
            "infrastructure_failure": self.infrastructure_failure,
            "incomplete": self.incomplete,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


def _forward_evidence_ref_closure(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    verifier_contract: VerifierContractV1 | None,
    official_outcome: OfficialRecoveryOutcomeV1 | None,
    budget_contract_ref: str | None,
    budget_decision: ResourceBudgetDecisionReceiptV1 | None,
    progress_contract: ProgressMetricContractV1 | None,
    progress_receipt: ForwardProgressReceiptV1 | None,
    false_repair_contract: OracleContractV1 | None,
    false_repair_policy: ForwardFalseRepairOraclePolicyV1 | None,
    false_repair_receipt: FalseRepairOracleReceiptV1 | None,
    transient_regression_contract: OracleContractV1 | None,
    transient_regression_policy: ForwardRegressionOraclePolicyV1 | None,
    final_regression_contract: OracleContractV1 | None,
    final_regression_policy: ForwardRegressionOraclePolicyV1 | None,
    regression_receipt: RegressionOracleReceiptV1 | None,
    optional_root_gold: AttributionGoldV1 | None,
    root_gold_binding: ForwardRootGoldBindingV1 | None,
    root_verification_contract: OracleContractV1 | None,
) -> ForwardEvidenceRefClosureV1:
    """Collect every supplied evidence reference before any fail-closed return."""

    return ForwardEvidenceRefClosureV1(
        execution_refs=_dedupe_refs(
            (execution.receipt_id,),
            execution.raw_receipt_refs,
        ),
        official_refs=_dedupe_refs(
            _present_refs(
                verifier_contract.source_ref if verifier_contract is not None else None,
            ),
            _official_refs(official_outcome),
        ),
        budget_refs=_dedupe_refs(
            _present_refs(
                budget_contract_ref,
                budget_decision.budget_contract_ref
                if budget_decision is not None
                else None,
            ),
            budget_decision.usage_receipt_refs
            if budget_decision is not None
            else (),
        ),
        progress_refs=_dedupe_refs(
            _present_refs(
                progress_contract.source_ref if progress_contract is not None else None,
                progress_contract.aggregation_rule_source_ref
                if progress_contract is not None
                else None,
            ),
            _progress_refs(progress_receipt),
        ),
        oracle_refs=_dedupe_refs(
            _present_refs(
                false_repair_contract.source_ref
                if false_repair_contract is not None
                else None,
                false_repair_policy.oracle_source_ref
                if false_repair_policy is not None
                else None,
                false_repair_policy.evaluator_source_ref
                if false_repair_policy is not None
                else None,
                transient_regression_contract.source_ref
                if transient_regression_contract is not None
                else None,
                transient_regression_policy.oracle_source_ref
                if transient_regression_policy is not None
                else None,
                transient_regression_policy.protected_roster_ref
                if transient_regression_policy is not None
                else None,
                transient_regression_policy.evaluator_source_ref
                if transient_regression_policy is not None
                else None,
                final_regression_contract.source_ref
                if final_regression_contract is not None
                else None,
                final_regression_policy.oracle_source_ref
                if final_regression_policy is not None
                else None,
                final_regression_policy.protected_roster_ref
                if final_regression_policy is not None
                else None,
                final_regression_policy.evaluator_source_ref
                if final_regression_policy is not None
                else None,
                root_verification_contract.source_ref
                if root_verification_contract is not None
                else None,
            ),
            _false_repair_refs(false_repair_receipt),
            _regression_refs(regression_receipt),
        ),
        root_gold_refs=_dedupe_refs(
            optional_root_gold.authority_receipt_refs
            if optional_root_gold is not None
            else (),
            _present_refs(
                optional_root_gold.case_ref.source_artifact_ref
                if optional_root_gold is not None
                else None,
                root_gold_binding.receipt_id if root_gold_binding is not None else None,
                root_gold_binding.population_binding_source_ref
                if root_gold_binding is not None
                else None,
            ),
            root_gold_binding.gold_authority_receipt_refs
            if root_gold_binding is not None
            else (),
        ),
    )


def _invalid_score(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    decision_status: ForwardDecisionStatus,
    reason: ForwardInvalidReason,
    official_outcome: OfficialRecoveryOutcomeV1 | None,
    evidence_ref_closure: ForwardEvidenceRefClosureV1,
) -> ForwardRecoveryCaseScoreV1:
    """Create one fail-closed provisional score without partial endpoints."""

    method_failed = decision_status is ForwardDecisionStatus.METHOD_FAILED
    evidence_status = (
        ForwardEvidenceStatus.METHOD_NOT_EVALUABLE
        if method_failed
        else ForwardEvidenceStatus.INVALID
    )
    return ForwardRecoveryCaseScoreV1(
        case_id=execution.case_id,
        source_task_id=execution.source_task_id,
        run_id=execution.run_id,
        population_id=execution.population_id,
        registry_id=execution.registry_id,
        scorer_id=execution.scorer_id,
        scorer_version=execution.scorer_version,
        work_item_id=execution.work_item_id,
        execution_receipt_id=execution.receipt_id,
        official_outcome_receipt_id=(
            official_outcome.receipt_id
            if type(official_outcome) is OfficialRecoveryOutcomeV1
            else None
        ),
        benchmark_id=execution.benchmark_id,
        benchmark_version=execution.benchmark_version,
        adapter_version=execution.adapter_version,
        system_cell_id=execution.system_cell_id,
        treatment_id=execution.treatment_id,
        locator_treatment_id=execution.locator_treatment_id,
        fixer_treatment_id=execution.fixer_treatment_id,
        repetition_id=execution.repetition_id,
        budget_id=execution.budget_id,
        split=execution.split,
        execution_semantics=execution.execution_semantics,
        decision_status=decision_status,
        invalid_reason=reason,
        stop_reason=execution.stop_reason,
        official_failure_resolved=None,
        fixed_denominator_failure_resolved=False if method_failed else None,
        budget_evidence_status=evidence_status,
        budget_overall_decision=None,
        failure_resolved_at_budget=None,
        progress_evidence_status=evidence_status,
        progress_contract_id=None,
        progress_observations=(),
        directed_progress_change=None,
        root_gold_evidence_status=evidence_status,
        root_identified=None,
        root_verification_evidence_status=evidence_status,
        strict_root_verification_observed=None,
        verified_root=None,
        false_repair_evidence_status=evidence_status,
        false_repair_occurred=None,
        transient_regression_evidence_status=evidence_status,
        transient_regression_occurred=None,
        final_regression_evidence_status=evidence_status,
        final_regression_occurred=None,
        costs=execution.costs,
        evidence_ref_closure=evidence_ref_closure,
        raw_evidence_refs=evidence_ref_closure.raw_evidence_refs,
        method_failure=method_failed,
        infrastructure_failure=(
            decision_status is ForwardDecisionStatus.INFRASTRUCTURE_INVALID
        ),
        incomplete=decision_status
        in {
            ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            ForwardDecisionStatus.AMBIGUOUS_RESUME,
        },
    )


def _official_contract_identity_error(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    verifier_contract: VerifierContractV1,
    outcome: OfficialRecoveryOutcomeV1,
) -> ForwardInvalidReason | None:
    """Bind every official status, including unavailable ones, to this work item."""

    if (
        _domain(verifier_contract) != _domain(execution)
        or not _execution_identity_matches(outcome, execution)
        or outcome.verifier_contract_id != verifier_contract.verifier_contract_id
        or outcome.official_verifier_id != verifier_contract.official_verifier_id
        or outcome.official_verifier_version
        != verifier_contract.official_verifier_version
        or outcome.final_success_field != verifier_contract.final_success_field
    ):
        return ForwardInvalidReason.OFFICIAL_OUTCOME_IDENTITY_MISMATCH
    return None


def _official_identity_error(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    verifier_contract: VerifierContractV1,
    outcome: OfficialRecoveryOutcomeV1,
) -> ForwardInvalidReason | None:
    """Bind an available official success to the exact terminal state."""

    error = _official_contract_identity_error(
        execution=execution,
        verifier_contract=verifier_contract,
        outcome=outcome,
    )
    if error is not None:
        return error
    if (
        outcome.final_state_id != execution.final_state_id
        or outcome.terminal_receipt_ref != execution.terminal_receipt_ref
    ):
        return ForwardInvalidReason.OFFICIAL_OUTCOME_IDENTITY_MISMATCH
    return None


def _budget_decision(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    contract: ResourceBudgetContractV1 | None,
    contract_ref: str | None,
    receipt: ResourceBudgetDecisionReceiptV1 | None,
) -> tuple[
    ForwardInvalidReason | None,
    ForwardEvidenceStatus,
    BudgetOverallDecision | None,
]:
    """Recompute one provisional budget decision from the frozen contract."""

    if contract is None:
        if contract_ref is not None or receipt is not None:
            return (
                ForwardInvalidReason.BUDGET_RECEIPT_IDENTITY_MISMATCH,
                ForwardEvidenceStatus.INVALID,
                None,
            )
        return None, ForwardEvidenceStatus.UNREGISTERED, None
    # Contract identity is mandatory even when references or usage are unavailable.
    if contract.budget_id != execution.budget_id:
        return (
            ForwardInvalidReason.BUDGET_RECEIPT_IDENTITY_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    if contract_ref is None:
        if receipt is None:
            return None, ForwardEvidenceStatus.REGISTERED_UNAVAILABLE, None
        return (
            ForwardInvalidReason.BUDGET_RECEIPT_IDENTITY_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    _text(contract_ref, "budget_contract_ref")
    if receipt is None:
        return None, ForwardEvidenceStatus.REGISTERED_UNAVAILABLE, None
    if (
        receipt.subject_kind is not BudgetSubjectKind.WORK_ITEM
        or receipt.subject_id != execution.work_item_id
        or receipt.budget_id != execution.budget_id
        or receipt.budget_contract_ref != contract_ref
        or receipt.usage_aggregation_receipt_ref not in execution.raw_receipt_refs
    ):
        return (
            ForwardInvalidReason.BUDGET_RECEIPT_IDENTITY_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    recomputed = evaluate_resource_budget_v1(
        budget=contract,
        subject_kind=BudgetSubjectKind.WORK_ITEM,
        subject_id=execution.work_item_id,
        costs=execution.costs,
        budget_contract_ref=contract_ref,
        usage_aggregation_receipt_ref=receipt.usage_aggregation_receipt_ref,
    )
    if recomputed != receipt:
        return (
            ForwardInvalidReason.BUDGET_RECEIPT_IDENTITY_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    return (
        None,
        ForwardEvidenceStatus.REGISTERED_AVAILABLE,
        receipt.overall_decision,
    )


def _progress_decision(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    verifier_contract: VerifierContractV1,
    contract: ProgressMetricContractV1 | None,
    receipt: ForwardProgressReceiptV1 | None,
) -> tuple[
    ForwardInvalidReason | None,
    ForwardEvidenceStatus,
    str | None,
    tuple[ForwardProgressObservationV1, ...],
    float | None,
]:
    """Validate at most one official progress contract for the current domain."""

    if contract is None:
        if receipt is not None:
            return (
                ForwardInvalidReason.PROGRESS_CONTRACT_MISMATCH,
                ForwardEvidenceStatus.INVALID,
                None,
                (),
                None,
            )
        return None, ForwardEvidenceStatus.UNREGISTERED, None, (), None
    if (
        contract.evidence_source is not EvidenceSource.OFFICIAL_BENCHMARK
        or _domain(contract) != _domain(execution)
        or contract.evaluator_contract_id != verifier_contract.verifier_contract_id
        or contract.aggregation_rule_id != FORWARD_PROGRESS_AGGREGATION_RULE_ID
        or contract.aggregation_rule_source_ref
        != FORWARD_PROGRESS_AGGREGATION_RULE_SOURCE_REF
        or len(contract.timepoints) < 2
    ):
        return (
            ForwardInvalidReason.PROGRESS_CONTRACT_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
            (),
            None,
        )
    if receipt is None:
        return (
            ForwardInvalidReason.PROGRESS_RECEIPT_MISSING,
            ForwardEvidenceStatus.INVALID,
            None,
            (),
            None,
        )
    if (
        not _execution_identity_matches(receipt, execution)
        or receipt.progress_contract_id != contract.progress_contract_id
        or receipt.metric_id != contract.metric_id
        or receipt.metric_version != contract.metric_version
        or receipt.evaluator_contract_id != contract.evaluator_contract_id
        or receipt.evaluator_id != verifier_contract.official_verifier_id
        or receipt.evaluator_version != verifier_contract.official_verifier_version
        or tuple(value.timepoint for value in receipt.observations)
        != contract.timepoints
    ):
        return (
            ForwardInvalidReason.PROGRESS_RECEIPT_IDENTITY_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
            (),
            None,
        )
    transition_positions = {
        transition.transition_id: (transition.ordinal, transition.state_id_after)
        for transition in execution.transitions
    }
    observation_positions: list[int] = []
    for observation in receipt.observations:
        if observation.state_role is ForwardProgressStateRole.FAILED_START:
            valid_state = observation.state_id == execution.failed_start_state_id
            position = 0
        elif observation.state_role is ForwardProgressStateRole.FINAL:
            valid_state = observation.state_id == execution.final_state_id
            position = len(execution.transitions) + 1
        else:
            transition = transition_positions.get(observation.transition_id)
            valid_state = transition is not None and observation.state_id == transition[1]
            position = transition[0] if transition is not None else -1
        if not valid_state:
            return (
                ForwardInvalidReason.PROGRESS_RECEIPT_IDENTITY_MISMATCH,
                ForwardEvidenceStatus.INVALID,
                None,
                (),
                None,
            )
        observation_positions.append(position)
    if observation_positions != sorted(observation_positions) or (
        len(receipt.observations) > 1
        and (
            receipt.observations[0].state_role
            is not ForwardProgressStateRole.FAILED_START
            or receipt.observations[-1].state_role
            is not ForwardProgressStateRole.FINAL
        )
    ):
        return (
            ForwardInvalidReason.PROGRESS_RECEIPT_IDENTITY_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
            (),
            None,
        )
    for observation in receipt.observations:
        if observation.value is not None and not (
            contract.domain_min <= observation.value <= contract.domain_max
        ):
            return (
                ForwardInvalidReason.PROGRESS_VALUE_OUT_OF_DOMAIN,
                ForwardEvidenceStatus.INVALID,
                None,
                (),
                None,
            )
    if any(
        value.value_status is UsageValueStatus.UNAVAILABLE
        for value in receipt.observations
    ):
        return (
            None,
            ForwardEvidenceStatus.REGISTERED_UNAVAILABLE,
            contract.progress_contract_id,
            receipt.observations,
            None,
        )
    change = None
    if len(receipt.observations) > 1:
        first = receipt.observations[0].value
        last = receipt.observations[-1].value
        if first is None or last is None:
            raise ValueError("available progress observations cannot have null values")
        change = (
            last - first
            if contract.direction is ProgressDirection.HIGHER_IS_BETTER
            else first - last
        )
    return (
        None,
        ForwardEvidenceStatus.REGISTERED_AVAILABLE,
        contract.progress_contract_id,
        receipt.observations,
        change,
    )


def _oracle_contract_error(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    contract: OracleContractV1,
    expected_kind: OracleKind,
) -> ForwardInvalidReason | None:
    if contract.kind is not expected_kind or _domain(contract) != _domain(execution):
        return ForwardInvalidReason.ORACLE_CONTRACT_MISMATCH
    return None


def _false_repair_decision(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    contract: OracleContractV1 | None,
    policy: ForwardFalseRepairOraclePolicyV1 | None,
    receipt: FalseRepairOracleReceiptV1 | None,
) -> tuple[ForwardInvalidReason | None, ForwardEvidenceStatus, bool | None]:
    if contract is None:
        if policy is not None or receipt is not None:
            return (
                ForwardInvalidReason.ORACLE_CONTRACT_MISMATCH,
                ForwardEvidenceStatus.INVALID,
                None,
            )
        return None, ForwardEvidenceStatus.UNREGISTERED, None
    contract_error = _oracle_contract_error(
        execution=execution,
        contract=contract,
        expected_kind=OracleKind.FALSE_REPAIR,
    )
    if contract_error is not None:
        return contract_error, ForwardEvidenceStatus.INVALID, None
    if receipt is None:
        return (
            ForwardInvalidReason.REGISTERED_ORACLE_RECEIPT_MISSING,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    if (
        not _execution_identity_matches(receipt, execution)
        or receipt.final_state_id != execution.final_state_id
        or receipt.oracle_contract_id != contract.oracle_contract_id
        or receipt.oracle_version != contract.oracle_version
        or receipt.outcome_field != contract.outcome_field
    ):
        return (
            ForwardInvalidReason.ORACLE_RECEIPT_IDENTITY_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    if policy is None:
        return None, ForwardEvidenceStatus.REGISTERED_UNAVAILABLE, None
    if (
        policy.oracle_contract_id != contract.oracle_contract_id
        or policy.oracle_version != contract.oracle_version
        or policy.outcome_field != contract.outcome_field
        or policy.oracle_source_ref != contract.source_ref
        or receipt.criterion_id != policy.criterion_id
        or receipt.criterion_version != policy.criterion_version
        or receipt.evaluator_id != policy.evaluator_id
        or receipt.evaluator_version != policy.evaluator_version
    ):
        return (
            ForwardInvalidReason.ORACLE_CONTRACT_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    if receipt.status is ForwardOracleReceiptStatus.UNAVAILABLE:
        return None, ForwardEvidenceStatus.REGISTERED_UNAVAILABLE, None
    return (
        None,
        ForwardEvidenceStatus.REGISTERED_AVAILABLE,
        receipt.false_repair_occurred,
    )


def _regression_history_error(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    receipt: RegressionOracleReceiptV1 | None,
    any_registered_contract: bool,
    transient_policy: ForwardRegressionOraclePolicyV1 | None,
    final_policy: ForwardRegressionOraclePolicyV1 | None,
) -> ForwardInvalidReason | None:
    """Bind the one shared history to the executed state chain and both policies."""

    if receipt is None:
        return None
    if not any_registered_contract:
        return ForwardInvalidReason.ORACLE_CONTRACT_MISMATCH
    expected_bindings = (
        (
            ForwardProgressStateRole.FAILED_START,
            execution.failed_start_state_id,
            None,
        ),
        *tuple(
            (
                ForwardProgressStateRole.TRANSITION_AFTER,
                transition.state_id_after,
                transition.transition_id,
            )
            for transition in execution.transitions
        ),
        (ForwardProgressStateRole.FINAL, execution.final_state_id, None),
    )
    if (
        not _execution_identity_matches(receipt, execution)
        or receipt.final_state_id != execution.final_state_id
        or (
            receipt.status is ForwardOracleReceiptStatus.AVAILABLE
            and any(
                history.state_bindings != expected_bindings
                for history in receipt.protected_item_histories
            )
        )
    ):
        return ForwardInvalidReason.ORACLE_RECEIPT_IDENTITY_MISMATCH
    if transient_policy is not None and final_policy is not None:
        shared_policy_fields = (
            "protected_item_ids",
            "protected_roster_ref",
            "evaluator_id",
            "evaluator_version",
            "evaluator_source_ref",
        )
        if any(
            getattr(transient_policy, field) != getattr(final_policy, field)
            for field in shared_policy_fields
        ):
            return ForwardInvalidReason.ORACLE_CONTRACT_MISMATCH
    return None


def _regression_decision(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    contract: OracleContractV1 | None,
    policy: ForwardRegressionOraclePolicyV1 | None,
    receipt: RegressionOracleReceiptV1 | None,
    kind: OracleKind,
) -> tuple[ForwardInvalidReason | None, ForwardEvidenceStatus, bool | None]:
    if contract is None:
        if policy is not None:
            return (
                ForwardInvalidReason.ORACLE_CONTRACT_MISMATCH,
                ForwardEvidenceStatus.INVALID,
                None,
            )
        return None, ForwardEvidenceStatus.UNREGISTERED, None
    contract_error = _oracle_contract_error(
        execution=execution,
        contract=contract,
        expected_kind=kind,
    )
    if contract_error is not None:
        return contract_error, ForwardEvidenceStatus.INVALID, None
    if receipt is None:
        return (
            ForwardInvalidReason.REGISTERED_ORACLE_RECEIPT_MISSING,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    if policy is not None and policy.kind is not kind:
        return (
            ForwardInvalidReason.ORACLE_CONTRACT_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    if policy is None:
        return None, ForwardEvidenceStatus.REGISTERED_UNAVAILABLE, None
    protected_ids = tuple(
        history.protected_item_id for history in receipt.protected_item_histories
    )
    if (
        policy.oracle_contract_id != contract.oracle_contract_id
        or policy.oracle_version != contract.oracle_version
        or policy.outcome_field != contract.outcome_field
        or policy.oracle_source_ref != contract.source_ref
        or policy.kind is not kind
        or receipt.evaluator_id != policy.evaluator_id
        or receipt.evaluator_version != policy.evaluator_version
        or (
            receipt.status is ForwardOracleReceiptStatus.AVAILABLE
            and protected_ids != policy.protected_item_ids
        )
    ):
        return (
            ForwardInvalidReason.ORACLE_CONTRACT_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    if receipt.status is ForwardOracleReceiptStatus.UNAVAILABLE:
        return None, ForwardEvidenceStatus.REGISTERED_UNAVAILABLE, None
    return (
        None,
        ForwardEvidenceStatus.REGISTERED_AVAILABLE,
        (
            receipt.transient_regression_occurred
            if kind is OracleKind.TRANSIENT_REGRESSION
            else receipt.final_regression_occurred
        ),
    )


def _root_gold_decision(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    gold: AttributionGoldV1 | None,
    binding: ForwardRootGoldBindingV1 | None,
) -> tuple[ForwardInvalidReason | None, ForwardEvidenceStatus, bool | None]:
    """Use optional static gold only for exact-group RootIdentified."""

    if gold is None:
        if binding is not None:
            return (
                ForwardInvalidReason.ROOT_GOLD_CASE_MISMATCH,
                ForwardEvidenceStatus.INVALID,
                None,
            )
        return None, ForwardEvidenceStatus.UNREGISTERED, None
    if binding is None:
        return (
            ForwardInvalidReason.ROOT_GOLD_CASE_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    case_ref = gold.case_ref
    if (
        case_ref.case_id != execution.case_id
        or case_ref.source_task_id != execution.source_task_id
        or case_ref.split is not execution.split
        or _domain(case_ref) != _domain(execution)
        or binding.population_id != execution.population_id
        or binding.case_id != execution.case_id
        or binding.source_task_id != execution.source_task_id
        or binding.split is not execution.split
        or _domain(binding) != _domain(execution)
        or binding.gold_authority_receipt_refs != gold.authority_receipt_refs
    ):
        return (
            ForwardInvalidReason.ROOT_GOLD_CASE_MISMATCH,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    if any(
        event_id not in execution.public_event_ids
        for group in gold.accepted_event_groups
        for event_id in group
    ):
        return (
            ForwardInvalidReason.ROOT_GOLD_EVENT_OUTSIDE_ROSTER,
            ForwardEvidenceStatus.INVALID,
            None,
        )
    if gold.disposition is not GoldDisposition.SCORABLE:
        return None, ForwardEvidenceStatus.REGISTERED_UNAVAILABLE, None
    candidate = execution.identified_root_event_group
    identified = candidate is not None and any(
        _same_group(candidate, accepted)
        for accepted in gold.accepted_event_groups
    )
    return None, ForwardEvidenceStatus.REGISTERED_AVAILABLE, identified


def _root_verification_decision(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    oracle_contract: OracleContractV1 | None,
) -> tuple[ForwardInvalidReason | None, ForwardEvidenceStatus, bool | None]:
    """Keep formal root verification unavailable until a neutral binder exists."""

    if oracle_contract is None:
        return None, ForwardEvidenceStatus.UNREGISTERED, None
    contract_error = _oracle_contract_error(
        execution=execution,
        contract=oracle_contract,
        expected_kind=OracleKind.FORWARD_ROOT_VERIFICATION,
    )
    if contract_error is not None:
        return contract_error, ForwardEvidenceStatus.INVALID, None
    # Historical three-arm work items have different track-scoped identities.
    # A future parent-linked neutral receipt must be specified before scoring.
    return None, ForwardEvidenceStatus.REGISTERED_UNAVAILABLE, None


def score_forward_recovery_case_v1(
    *,
    execution: ForwardRecoveryExecutionReceiptV1,
    verifier_contract: VerifierContractV1 | None,
    official_outcome: OfficialRecoveryOutcomeV1 | None,
    budget_contract: ResourceBudgetContractV1 | None = None,
    budget_contract_ref: str | None = None,
    budget_decision: ResourceBudgetDecisionReceiptV1 | None = None,
    progress_contract: ProgressMetricContractV1 | None = None,
    progress_receipt: ForwardProgressReceiptV1 | None = None,
    false_repair_contract: OracleContractV1 | None = None,
    false_repair_policy: ForwardFalseRepairOraclePolicyV1 | None = None,
    false_repair_receipt: FalseRepairOracleReceiptV1 | None = None,
    transient_regression_contract: OracleContractV1 | None = None,
    transient_regression_policy: ForwardRegressionOraclePolicyV1 | None = None,
    final_regression_contract: OracleContractV1 | None = None,
    final_regression_policy: ForwardRegressionOraclePolicyV1 | None = None,
    regression_receipt: RegressionOracleReceiptV1 | None = None,
    optional_root_gold: AttributionGoldV1 | None = None,
    root_gold_binding: ForwardRootGoldBindingV1 | None = None,
    root_verification_contract: OracleContractV1 | None = None,
) -> ForwardRecoveryCaseScoreV1:
    """Score one completed root-optional recovery from already frozen receipts.

    The fail-closed order is execution state, same-Actor semantics, official
    outcome, budget identity, progress, independent safety oracles, optional
    attribution gold, and the root-verification registration boundary.  Any
    identity failure invalidates the whole provisional cell instead of leaving
    a selectively usable endpoint.
    """

    if type(execution) is not ForwardRecoveryExecutionReceiptV1:
        raise TypeError("execution must be a ForwardRecoveryExecutionReceiptV1")
    optional_types = (
        (verifier_contract, VerifierContractV1, "verifier_contract"),
        (official_outcome, OfficialRecoveryOutcomeV1, "official_outcome"),
        (budget_contract, ResourceBudgetContractV1, "budget_contract"),
        (budget_decision, ResourceBudgetDecisionReceiptV1, "budget_decision"),
        (progress_contract, ProgressMetricContractV1, "progress_contract"),
        (progress_receipt, ForwardProgressReceiptV1, "progress_receipt"),
        (false_repair_contract, OracleContractV1, "false_repair_contract"),
        (
            false_repair_policy,
            ForwardFalseRepairOraclePolicyV1,
            "false_repair_policy",
        ),
        (false_repair_receipt, FalseRepairOracleReceiptV1, "false_repair_receipt"),
        (
            transient_regression_contract,
            OracleContractV1,
            "transient_regression_contract",
        ),
        (
            transient_regression_policy,
            ForwardRegressionOraclePolicyV1,
            "transient_regression_policy",
        ),
        (final_regression_contract, OracleContractV1, "final_regression_contract"),
        (
            final_regression_policy,
            ForwardRegressionOraclePolicyV1,
            "final_regression_policy",
        ),
        (regression_receipt, RegressionOracleReceiptV1, "regression_receipt"),
        (optional_root_gold, AttributionGoldV1, "optional_root_gold"),
        (root_gold_binding, ForwardRootGoldBindingV1, "root_gold_binding"),
        (
            root_verification_contract,
            OracleContractV1,
            "root_verification_contract",
        ),
    )
    for value, expected_type, label in optional_types:
        if value is not None and type(value) is not expected_type:
            raise TypeError(f"{label} must be {expected_type.__name__} or null")
    evidence_ref_closure = _forward_evidence_ref_closure(
        execution=execution,
        verifier_contract=verifier_contract,
        official_outcome=official_outcome,
        budget_contract_ref=budget_contract_ref,
        budget_decision=budget_decision,
        progress_contract=progress_contract,
        progress_receipt=progress_receipt,
        false_repair_contract=false_repair_contract,
        false_repair_policy=false_repair_policy,
        false_repair_receipt=false_repair_receipt,
        transient_regression_contract=transient_regression_contract,
        transient_regression_policy=transient_regression_policy,
        final_regression_contract=final_regression_contract,
        final_regression_policy=final_regression_policy,
        regression_receipt=regression_receipt,
        optional_root_gold=optional_root_gold,
        root_gold_binding=root_gold_binding,
        root_verification_contract=root_verification_contract,
    )
    if execution.status is ForwardExecutionStatus.IN_FLIGHT:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.AMBIGUOUS_RESUME,
            reason=ForwardInvalidReason.EXECUTION_IN_FLIGHT,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    if execution.status is ForwardExecutionStatus.INFRASTRUCTURE_FAILED:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=ForwardInvalidReason.EXECUTION_INFRASTRUCTURE_FAILURE,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    if execution.status is ForwardExecutionStatus.METHOD_FAILED:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.METHOD_FAILED,
            reason=ForwardInvalidReason.EXECUTION_METHOD_FAILURE,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    if execution.execution_semantics is ExecutionSemantics.FROZEN_TRACE:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=ForwardInvalidReason.FORWARD_EXECUTION_SEMANTICS_INVALID,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    if (
        execution.source_actor_identity != execution.recovery_actor_identity
        or execution.same_actor_identity_receipt_ref is None
    ):
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=ForwardInvalidReason.SAME_ACTOR_IDENTITY_MISMATCH,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    if verifier_contract is None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=ForwardInvalidReason.OFFICIAL_VERIFIER_CONTRACT_MISSING,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    if type(verifier_contract) is not VerifierContractV1:
        raise TypeError("verifier_contract must be a VerifierContractV1 or null")
    if official_outcome is None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=ForwardInvalidReason.OFFICIAL_OUTCOME_MISSING,
            official_outcome=None,
            evidence_ref_closure=evidence_ref_closure,
        )
    if type(official_outcome) is not OfficialRecoveryOutcomeV1:
        raise TypeError("official_outcome must be an OfficialRecoveryOutcomeV1 or null")
    official_contract_error = _official_contract_identity_error(
        execution=execution,
        verifier_contract=verifier_contract,
        outcome=official_outcome,
    )
    if official_contract_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=official_contract_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    if official_outcome.status is ForwardOfficialOutcomeStatus.TERMINAL_UNKNOWN:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.AMBIGUOUS_RESUME,
            reason=ForwardInvalidReason.OFFICIAL_TERMINAL_UNKNOWN,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    if official_outcome.status is ForwardOfficialOutcomeStatus.UNAVAILABLE:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=ForwardInvalidReason.OFFICIAL_OUTCOME_UNAVAILABLE,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    official_error = _official_identity_error(
        execution=execution,
        verifier_contract=verifier_contract,
        outcome=official_outcome,
    )
    if official_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=official_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )

    budget_error, budget_evidence_status, budget_status = _budget_decision(
        execution=execution,
        contract=budget_contract,
        contract_ref=budget_contract_ref,
        receipt=budget_decision,
    )
    if budget_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=budget_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    (
        progress_error,
        progress_status,
        progress_contract_id,
        progress_observations,
        directed_progress_change,
    ) = _progress_decision(
        execution=execution,
        verifier_contract=verifier_contract,
        contract=progress_contract,
        receipt=progress_receipt,
    )
    if progress_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=progress_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    false_error, false_status, false_repair = _false_repair_decision(
        execution=execution,
        contract=false_repair_contract,
        policy=false_repair_policy,
        receipt=false_repair_receipt,
    )
    if false_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=false_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    regression_history_error = _regression_history_error(
        execution=execution,
        receipt=regression_receipt,
        any_registered_contract=(
            transient_regression_contract is not None
            or final_regression_contract is not None
        ),
        transient_policy=transient_regression_policy,
        final_policy=final_regression_policy,
    )
    if regression_history_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=regression_history_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    transient_error, transient_status, transient_regression = _regression_decision(
        execution=execution,
        contract=transient_regression_contract,
        policy=transient_regression_policy,
        receipt=regression_receipt,
        kind=OracleKind.TRANSIENT_REGRESSION,
    )
    if transient_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=transient_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    final_error, final_status, final_regression = _regression_decision(
        execution=execution,
        contract=final_regression_contract,
        policy=final_regression_policy,
        receipt=regression_receipt,
        kind=OracleKind.FINAL_REGRESSION,
    )
    if final_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=final_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    root_gold_error, root_gold_status, root_identified = _root_gold_decision(
        execution=execution,
        gold=optional_root_gold,
        binding=root_gold_binding,
    )
    if root_gold_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=root_gold_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )
    root_error, root_status, strict_root = _root_verification_decision(
        execution=execution,
        oracle_contract=root_verification_contract,
    )
    if root_error is not None:
        return _invalid_score(
            execution=execution,
            decision_status=ForwardDecisionStatus.INFRASTRUCTURE_INVALID,
            reason=root_error,
            official_outcome=official_outcome,
            evidence_ref_closure=evidence_ref_closure,
        )

    if official_outcome.failure_resolved is None:
        raise ValueError("available official outcome cannot have a null decision")

    return ForwardRecoveryCaseScoreV1(
        case_id=execution.case_id,
        source_task_id=execution.source_task_id,
        run_id=execution.run_id,
        population_id=execution.population_id,
        registry_id=execution.registry_id,
        scorer_id=execution.scorer_id,
        scorer_version=execution.scorer_version,
        work_item_id=execution.work_item_id,
        execution_receipt_id=execution.receipt_id,
        official_outcome_receipt_id=official_outcome.receipt_id,
        benchmark_id=execution.benchmark_id,
        benchmark_version=execution.benchmark_version,
        adapter_version=execution.adapter_version,
        system_cell_id=execution.system_cell_id,
        treatment_id=execution.treatment_id,
        locator_treatment_id=execution.locator_treatment_id,
        fixer_treatment_id=execution.fixer_treatment_id,
        repetition_id=execution.repetition_id,
        budget_id=execution.budget_id,
        split=execution.split,
        execution_semantics=execution.execution_semantics,
        decision_status=ForwardDecisionStatus.SCORED,
        invalid_reason=None,
        stop_reason=execution.stop_reason,
        official_failure_resolved=official_outcome.failure_resolved,
        fixed_denominator_failure_resolved=official_outcome.failure_resolved,
        budget_evidence_status=budget_evidence_status,
        budget_overall_decision=budget_status,
        failure_resolved_at_budget=None,
        progress_evidence_status=progress_status,
        progress_contract_id=progress_contract_id,
        progress_observations=progress_observations,
        directed_progress_change=directed_progress_change,
        root_gold_evidence_status=root_gold_status,
        root_identified=root_identified,
        root_verification_evidence_status=root_status,
        strict_root_verification_observed=strict_root,
        verified_root=None,
        false_repair_evidence_status=false_status,
        false_repair_occurred=false_repair,
        transient_regression_evidence_status=transient_status,
        transient_regression_occurred=transient_regression,
        final_regression_evidence_status=final_status,
        final_regression_occurred=final_regression,
        costs=execution.costs,
        evidence_ref_closure=evidence_ref_closure,
        raw_evidence_refs=evidence_ref_closure.raw_evidence_refs,
        method_failure=False,
        infrastructure_failure=False,
        incomplete=False,
    )


__all__ = [
    "FORWARD_CASE_SCORE_SCHEMA",
    "FORWARD_DESIGN_ORIGIN",
    "FORWARD_EVIDENCE_REF_CLOSURE_SCHEMA",
    "FORWARD_EXECUTION_RECEIPT_SCHEMA",
    "FORWARD_FALSE_REPAIR_POLICY_SCHEMA",
    "FORWARD_FALSE_REPAIR_RECEIPT_SCHEMA",
    "FORWARD_IMPLEMENTATION_STATUS",
    "FORWARD_INPUT_BINDING_STATUS",
    "FORWARD_OFFICIAL_OUTCOME_SCHEMA",
    "FORWARD_PROGRESS_OBSERVATION_SCHEMA",
    "FORWARD_PROGRESS_AGGREGATION_RULE_ID",
    "FORWARD_PROGRESS_AGGREGATION_RULE_SOURCE_REF",
    "FORWARD_PROGRESS_RECEIPT_SCHEMA",
    "FORWARD_PROTECTED_ITEM_HISTORY_SCHEMA",
    "FORWARD_PROTECTED_ITEM_OBSERVATION_SCHEMA",
    "FORWARD_REGRESSION_RECEIPT_SCHEMA",
    "FORWARD_REGRESSION_POLICY_SCHEMA",
    "FORWARD_ROOT_GOLD_BINDING_SCHEMA",
    "FORWARD_TRANSITION_SCHEMA",
    "FalseRepairOracleReceiptV1",
    "ForwardDecisionStatus",
    "ForwardEvidenceRefClosureV1",
    "ForwardEvidenceStatus",
    "ForwardExecutionStatus",
    "ForwardInvalidReason",
    "ForwardOfficialOutcomeStatus",
    "ForwardOracleReceiptStatus",
    "ForwardFalseRepairOraclePolicyV1",
    "ForwardProgressObservationV1",
    "ForwardProgressReceiptV1",
    "ForwardProgressStateRole",
    "ForwardProtectedItemHistoryV1",
    "ForwardProtectedItemObservationV1",
    "ForwardRecoveryCaseScoreV1",
    "ForwardRecoveryExecutionReceiptV1",
    "ForwardRecoveryTransitionV1",
    "ForwardRegressionOraclePolicyV1",
    "ForwardRootGoldBindingV1",
    "ForwardStopReason",
    "OfficialRecoveryOutcomeV1",
    "RegressionOracleReceiptV1",
    "score_forward_recovery_case_v1",
]
