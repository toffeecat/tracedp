"""Fail-closed structural inspection for TraceFix-Bench work-item ledgers.

The inspector consumes typed contracts supplied by stricter filesystem loaders
and binders. It never writes an event, dispatches an action, restores mutable
state, or claims that a caller-constructed binding receipt is durable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracefix_bench.contracts import (
    ExecutionSemantics,
    TraceFailureDomain,
)
from lh_platform.tracefix_bench.run_contracts import (
    RUN_CONTRACT_IMPLEMENTATION_STATUS,
    _validate_action_failure,
    ActionPlanBindingStatus,
    ExpectedWorkItemV1,
    LedgerEventPhase,
    RunActionFailureKind,
    RunLedgerEventV1,
    WorkItemActionKind,
    WorkItemActionPlanV1,
    WorkItemActionSpecV1,
)
from lh_platform.tracefix_bench.scorer_contracts import ResourceRole


PERSISTED_LEDGER_ARTIFACT_SCHEMA = (
    "longhorizon_tracefix_bench/persisted_ledger_artifact/v1"
)
LEDGER_ARTIFACT_INVENTORY_SCHEMA = (
    "longhorizon_tracefix_bench/ledger_artifact_inventory/v1"
)
EXECUTION_RECOVERY_BINDING_SCHEMA = (
    "longhorizon_tracefix_bench/execution_recovery_binding/v1"
)
WORK_ITEM_LEDGER_INSPECTION_SCHEMA = (
    "longhorizon_tracefix_bench/work_item_ledger_inspection/v1"
)
LEDGER_INSPECTION_PAYLOAD_ORIGIN = "inspector_derived"

_IDENTIFIER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _identifier(value: str, label: str) -> None:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable non-empty identifier")


def _unique_texts(values: tuple[str, ...], label: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or not value.strip() for value in values
    ):
        raise ValueError(f"{label} must be a tuple of non-empty strings")
    if len({value.casefold() for value in values}) != len(values):
        raise ValueError(f"{label} must be unique after case folding")


def _unverified(value: str) -> None:
    if type(value) is not str or value != RUN_CONTRACT_IMPLEMENTATION_STATUS:
        raise ValueError("ledger contracts cannot promote implementation status")


class LedgerArtifactKind(StrEnum):
    """Identify action artifacts; dispatch evidence never proves acceptance.

    Local lease/fence files are outside this inventory. External acceptance
    needs a future dedicated receipt contract and cannot be inferred here.
    """

    INPUT = "input"
    DISPATCH_EVIDENCE = "dispatch_evidence"
    RAW_RECEIPT = "raw_receipt"
    SCORER_DECISION = "scorer_decision"


class LedgerInventoryBindingStatus(StrEnum):
    """Distinguish a claimed inventory from a complete canonical inspection."""

    UNBOUND = "unbound"
    CANONICAL_COMPLETE_BOUND = "canonical_complete_bound"


@dataclass(frozen=True)
class PersistedLedgerArtifactV1:
    """Describe one schema-decoded artifact and its exact work-item owner."""

    artifact_ref: str
    artifact_kind: LedgerArtifactKind
    artifact_schema: str
    run_id: str
    work_item_id: str
    action_id: str
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = PERSISTED_LEDGER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PERSISTED_LEDGER_ARTIFACT_SCHEMA:
            raise ValueError("unsupported persisted ledger artifact schema")
        _text(self.artifact_ref, "artifact_ref")
        _text(self.artifact_schema, "artifact_schema")
        for label in ("run_id", "work_item_id", "action_id"):
            _identifier(getattr(self, label), label)
        if not isinstance(self.artifact_kind, LedgerArtifactKind):
            raise ValueError("artifact_kind must use the closed ledger vocabulary")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the decoded schema identity and exact owner."""

        return {
            "schema": self.schema,
            "artifact_ref": self.artifact_ref,
            "artifact_kind": self.artifact_kind.value,
            "artifact_schema": self.artifact_schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "action_id": self.action_id,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class LedgerArtifactInventoryV1:
    """Freeze one canonical-scope inventory and whether absence claims are bound."""

    run_id: str
    work_item_id: str
    canonical_scope_ref: str
    artifacts: tuple[PersistedLedgerArtifactV1, ...]
    binding_status: LedgerInventoryBindingStatus
    binding_evidence_ref: str | None
    complete_inventory: bool
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = LEDGER_ARTIFACT_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LEDGER_ARTIFACT_INVENTORY_SCHEMA:
            raise ValueError("unsupported ledger artifact inventory schema")
        for label in ("run_id", "work_item_id"):
            _identifier(getattr(self, label), label)
        _text(self.canonical_scope_ref, "canonical_scope_ref")
        if type(self.artifacts) is not tuple or any(
            type(item) is not PersistedLedgerArtifactV1 for item in self.artifacts
        ):
            raise ValueError("artifacts must be PersistedLedgerArtifactV1 records")
        refs = tuple(item.artifact_ref for item in self.artifacts)
        _unique_texts(refs, "artifact refs")
        if any(
            item.run_id != self.run_id or item.work_item_id != self.work_item_id
            for item in self.artifacts
        ):
            raise ValueError("inventory artifacts must use its exact run/work owner")
        artifact_keys = tuple(
            (item.artifact_kind, item.action_id.casefold()) for item in self.artifacts
        )
        if len(set(artifact_keys)) != len(artifact_keys):
            raise ValueError("inventory cannot repeat one artifact kind per action")
        if not isinstance(self.binding_status, LedgerInventoryBindingStatus):
            raise ValueError("inventory binding_status must use the closed vocabulary")
        if self.binding_status is LedgerInventoryBindingStatus.UNBOUND:
            if self.binding_evidence_ref is not None or self.complete_inventory is not False:
                raise ValueError("unbound inventory cannot claim binding or completeness")
        else:
            _text(self.binding_evidence_ref, "binding_evidence_ref")
            if self.complete_inventory is not True:
                raise ValueError("bound canonical inventory must be complete")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize inventory scope, contents, and the binding boundary."""

        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "canonical_scope_ref": self.canonical_scope_ref,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "binding_status": self.binding_status.value,
            "binding_evidence_ref": self.binding_evidence_ref,
            "complete_inventory": self.complete_inventory,
            "implementation_status": self.implementation_status,
        }


class ExecutionRecoveryMode(StrEnum):
    """Freeze the recovery prerequisite implied by execution semantics."""

    IMMUTABLE_INPUT_RELOAD = "immutable_input_reload"
    FRESH_BRANCH_RECREATION = "fresh_branch_recreation"
    APPEND_ONLY_LOG_REPLAY = "append_only_log_replay"
    EXACT_CHECKPOINT_RESTORE = "exact_checkpoint_restore"


_RECOVERY_MODE_BY_EXECUTION = {
    ExecutionSemantics.FROZEN_TRACE: ExecutionRecoveryMode.IMMUTABLE_INPUT_RELOAD,
    ExecutionSemantics.FRESH_BRANCH: ExecutionRecoveryMode.FRESH_BRANCH_RECREATION,
    ExecutionSemantics.PERSISTENT_SESSION: ExecutionRecoveryMode.APPEND_ONLY_LOG_REPLAY,
    ExecutionSemantics.EXACT_CHECKPOINT_REPLAY: (
        ExecutionRecoveryMode.EXACT_CHECKPOINT_RESTORE
    ),
}


class ExecutionRecoveryBindingStatus(StrEnum):
    """State whether a recovery prerequisite was checked by its source adapter."""

    UNBOUND = "unbound"
    SOURCE_BOUND = "source_bound"


@dataclass(frozen=True)
class ExecutionRecoveryBindingV1:
    """Bind recovery evidence to one exact post-action continuation point."""

    run_id: str
    work_item_id: str
    after_action_id: str
    execution_semantics: ExecutionSemantics
    recovery_mode: ExecutionRecoveryMode
    binding_status: ExecutionRecoveryBindingStatus
    evidence_refs: tuple[str, ...]
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = EXECUTION_RECOVERY_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_RECOVERY_BINDING_SCHEMA:
            raise ValueError("unsupported execution recovery binding schema")
        for label in ("run_id", "work_item_id", "after_action_id"):
            _identifier(getattr(self, label), label)
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError("execution_semantics must use the closed vocabulary")
        if not isinstance(self.recovery_mode, ExecutionRecoveryMode):
            raise ValueError("recovery_mode must use the closed vocabulary")
        if self.recovery_mode is not _RECOVERY_MODE_BY_EXECUTION[
            self.execution_semantics
        ]:
            raise ValueError("recovery mode does not match execution semantics")
        if not isinstance(self.binding_status, ExecutionRecoveryBindingStatus):
            raise ValueError("recovery binding_status must use the closed vocabulary")
        _unique_texts(self.evidence_refs, "recovery evidence refs")
        if self.binding_status is ExecutionRecoveryBindingStatus.UNBOUND:
            if self.evidence_refs:
                raise ValueError("unbound recovery cannot carry evidence refs")
        elif not self.evidence_refs:
            raise ValueError("source-bound recovery requires evidence refs")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the continuation point and source binding state."""

        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "after_action_id": self.after_action_id,
            "execution_semantics": self.execution_semantics.value,
            "recovery_mode": self.recovery_mode.value,
            "binding_status": self.binding_status.value,
            "evidence_refs": list(self.evidence_refs),
            "implementation_status": self.implementation_status,
        }


class WorkItemLedgerStatus(StrEnum):
    """Close structural, binding, crash-window, and terminal classifications."""

    ACTION_PLAN_UNBOUND = "action_plan_unbound"
    INVENTORY_UNBOUND = "inventory_unbound"
    CLEAN_NOT_STARTED = "clean_not_started"
    INPUT_PERSISTED_PREPARED_EVENT_MISSING = "input_persisted_prepared_event_missing"
    DISPATCH_STATE_UNKNOWN = "dispatch_state_unknown"
    DISPATCH_EVIDENCE_TERMINAL_UNKNOWN = "dispatch_evidence_terminal_unknown"
    RAW_PERSISTED_TERMINAL_EVENT_MISSING = "raw_persisted_terminal_event_missing"
    RAW_PERSISTED_IN_FLIGHT_EVENT_MISSING = "raw_persisted_in_flight_event_missing"
    AMBIGUOUS_IN_FLIGHT = "ambiguous_in_flight"
    RECOVERY_PREREQUISITE_UNBOUND = "recovery_prerequisite_unbound"
    ACTION_COMPLETED_NEXT_ACTION_PENDING = "action_completed_next_action_pending"
    SCORER_DECISION_UNKNOWN = "scorer_decision_unknown"
    SCORER_DECISION_PERSISTED_EVENT_MISSING = (
        "scorer_decision_persisted_event_missing"
    )
    SCORED_TERMINAL = "scored_terminal"
    METHOD_FAILED_TERMINAL = "method_failed_terminal"
    INFRASTRUCTURE_FAILED_TERMINAL = "infrastructure_failed_terminal"
    CONFLICTED = "conflicted"


class LedgerContinuation(StrEnum):
    """Name the sole continuation permitted by each inspection state."""

    HOLD_FOR_ACTION_PLAN_BINDING = "hold_for_action_plan_binding"
    HOLD_FOR_INVENTORY_BINDING = "hold_for_inventory_binding"
    HOLD_FOR_EXECUTION_CAPABILITY = "hold_for_execution_capability"
    HOLD_FOR_LEDGER_WRITER = "hold_for_ledger_writer"
    HOLD_FOR_MANUAL_DISPATCH_REVIEW = "hold_for_manual_dispatch_review"
    HOLD_FOR_RAW_RECONCILER = "hold_for_raw_reconciler"
    HOLD_FOR_MANUAL_IN_FLIGHT_REVIEW = "hold_for_manual_in_flight_review"
    HOLD_FOR_RECOVERY_BINDING = "hold_for_recovery_binding"
    HOLD_FOR_SCORER_DECISION_REVIEW = "hold_for_scorer_decision_review"
    NONE_TERMINAL = "none_terminal"
    MANUAL_RECONCILIATION = "manual_reconciliation"


_STATUS_POLICY = {
    WorkItemLedgerStatus.ACTION_PLAN_UNBOUND: (
        LedgerContinuation.HOLD_FOR_ACTION_PLAN_BINDING, False
    ),
    WorkItemLedgerStatus.INVENTORY_UNBOUND: (
        LedgerContinuation.HOLD_FOR_INVENTORY_BINDING, False
    ),
    WorkItemLedgerStatus.CLEAN_NOT_STARTED: (
        LedgerContinuation.HOLD_FOR_EXECUTION_CAPABILITY, False
    ),
    WorkItemLedgerStatus.INPUT_PERSISTED_PREPARED_EVENT_MISSING: (
        LedgerContinuation.HOLD_FOR_LEDGER_WRITER, False
    ),
    WorkItemLedgerStatus.DISPATCH_STATE_UNKNOWN: (
        LedgerContinuation.HOLD_FOR_MANUAL_DISPATCH_REVIEW, False
    ),
    WorkItemLedgerStatus.DISPATCH_EVIDENCE_TERMINAL_UNKNOWN: (
        LedgerContinuation.HOLD_FOR_MANUAL_IN_FLIGHT_REVIEW, False
    ),
    WorkItemLedgerStatus.RAW_PERSISTED_TERMINAL_EVENT_MISSING: (
        LedgerContinuation.HOLD_FOR_RAW_RECONCILER, False
    ),
    WorkItemLedgerStatus.RAW_PERSISTED_IN_FLIGHT_EVENT_MISSING: (
        LedgerContinuation.MANUAL_RECONCILIATION, False
    ),
    WorkItemLedgerStatus.AMBIGUOUS_IN_FLIGHT: (
        LedgerContinuation.HOLD_FOR_MANUAL_IN_FLIGHT_REVIEW, False
    ),
    WorkItemLedgerStatus.RECOVERY_PREREQUISITE_UNBOUND: (
        LedgerContinuation.HOLD_FOR_RECOVERY_BINDING, False
    ),
    WorkItemLedgerStatus.ACTION_COMPLETED_NEXT_ACTION_PENDING: (
        LedgerContinuation.HOLD_FOR_EXECUTION_CAPABILITY, False
    ),
    WorkItemLedgerStatus.SCORER_DECISION_UNKNOWN: (
        LedgerContinuation.HOLD_FOR_SCORER_DECISION_REVIEW, False
    ),
    WorkItemLedgerStatus.SCORER_DECISION_PERSISTED_EVENT_MISSING: (
        LedgerContinuation.HOLD_FOR_LEDGER_WRITER, False
    ),
    WorkItemLedgerStatus.SCORED_TERMINAL: (
        LedgerContinuation.NONE_TERMINAL, False
    ),
    WorkItemLedgerStatus.METHOD_FAILED_TERMINAL: (
        LedgerContinuation.NONE_TERMINAL, False
    ),
    WorkItemLedgerStatus.INFRASTRUCTURE_FAILED_TERMINAL: (
        LedgerContinuation.NONE_TERMINAL, False
    ),
    WorkItemLedgerStatus.CONFLICTED: (
        LedgerContinuation.MANUAL_RECONCILIATION, False
    ),
}


class LedgerConflictCode(StrEnum):
    """Expose structural conflicts without parsing diagnostic prose."""

    PLAN_OWNER_MISMATCH = "plan_owner_mismatch"
    INVENTORY_OWNER_MISMATCH = "inventory_owner_mismatch"
    EVENT_OWNER_MISMATCH = "event_owner_mismatch"
    EVENT_SEQUENCE_MISMATCH = "event_sequence_mismatch"
    EVENT_ID_DUPLICATE = "event_id_duplicate"
    PREDECESSOR_MISMATCH = "predecessor_mismatch"
    ACTION_NOT_PLANNED = "action_not_planned"
    ACTION_ORDER_MISMATCH = "action_order_mismatch"
    ACTION_REOPENED = "action_reopened"
    ACTION_KIND_ROLE_MISMATCH = "action_kind_role_mismatch"
    ACTION_PHASE_SEQUENCE_INVALID = "action_phase_sequence_invalid"
    ACTION_STARTED_BEFORE_PRIOR_TERMINAL = "action_started_before_prior_terminal"
    ACTION_AFTER_WORK_ITEM_TERMINAL = "action_after_work_item_terminal"
    MULTIPLE_SCORER_DECISIONS = "multiple_scorer_decisions"
    INVENTORY_ACTION_NOT_PLANNED = "inventory_action_not_planned"
    DISPATCH_EVIDENCE_WITHOUT_PREPARED_EVENT = (
        "dispatch_evidence_without_prepared_event"
    )
    REFERENCED_ARTIFACT_MISSING = "referenced_artifact_missing"
    ARTIFACT_KIND_MISMATCH = "artifact_kind_mismatch"
    ARTIFACT_ACTION_OWNER_MISMATCH = "artifact_action_owner_mismatch"
    ARTIFACT_REF_CROSS_ACTION = "artifact_ref_cross_action"
    ARTIFACT_RESIDUE_UNCLASSIFIABLE = "artifact_residue_unclassifiable"
    RECOVERY_BINDING_MISMATCH = "recovery_binding_mismatch"
    PLAN_EXHAUSTED_WITHOUT_SCORER_DECISION = (
        "plan_exhausted_without_scorer_decision"
    )


_FAILURE_STATUSES = frozenset(
    {
        WorkItemLedgerStatus.METHOD_FAILED_TERMINAL,
        WorkItemLedgerStatus.INFRASTRUCTURE_FAILED_TERMINAL,
    }
)


class RecoveryInspectionStatus(StrEnum):
    """Keep recovery prerequisites independent from artifact residue state."""

    NOT_APPLICABLE = "not_applicable"
    IMMUTABLE_INPUT = "immutable_input"
    UNBOUND = "unbound"
    SOURCE_BOUND = "source_bound"
    MISMATCHED = "mismatched"


@dataclass(frozen=True)
class WorkItemLedgerInspectionV1:
    """Carry one internally closed, inspector-derived lifecycle classification."""

    run_id: str
    work_item_id: str
    execution_semantics: ExecutionSemantics
    plan_binding_status: ActionPlanBindingStatus
    inventory_binding_status: LedgerInventoryBindingStatus
    status: WorkItemLedgerStatus
    continuation: LedgerContinuation
    can_continue_without_action_retry: bool
    automatic_action_retry_allowed: bool
    event_count: int
    planned_action_ids: tuple[str, ...]
    planned_action_kinds: tuple[WorkItemActionKind, ...]
    observed_action_ids: tuple[str, ...]
    observed_action_kinds: tuple[WorkItemActionKind, ...]
    last_action_id: str | None
    last_phase: LedgerEventPhase | None
    next_action_id: str | None
    unreferenced_input_refs: tuple[str, ...]
    unreferenced_raw_receipt_refs: tuple[str, ...]
    unreferenced_scorer_decision_refs: tuple[str, ...]
    dispatch_evidence_refs: tuple[str, ...]
    last_action_dispatch_evidence_ref: str | None
    recovery_status: RecoveryInspectionStatus
    failure_domain: TraceFailureDomain | None
    failure_kind: RunActionFailureKind | None
    conflict_codes: tuple[LedgerConflictCode, ...]
    payload_origin: str = LEDGER_INSPECTION_PAYLOAD_ORIGIN
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = WORK_ITEM_LEDGER_INSPECTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WORK_ITEM_LEDGER_INSPECTION_SCHEMA:
            raise ValueError("unsupported work-item ledger inspection schema")
        for label in ("run_id", "work_item_id"):
            _identifier(getattr(self, label), label)
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError("inspection execution semantics must use the closed enum")
        if not isinstance(self.plan_binding_status, ActionPlanBindingStatus):
            raise ValueError("inspection plan binding must use the closed enum")
        if not isinstance(self.inventory_binding_status, LedgerInventoryBindingStatus):
            raise ValueError("inspection inventory binding must use the closed enum")
        if not isinstance(self.status, WorkItemLedgerStatus):
            raise ValueError("status must use the closed ledger vocabulary")
        expected_continuation, expected_can_continue = _STATUS_POLICY[self.status]
        if self.continuation is not expected_continuation:
            raise ValueError("continuation does not match ledger status")
        if self.can_continue_without_action_retry is not expected_can_continue:
            raise ValueError("continuation permission does not match ledger status")
        if self.automatic_action_retry_allowed is not False:
            raise ValueError("ledger inspection cannot authorize an action retry")
        if type(self.event_count) is not int or self.event_count < 0:
            raise ValueError("event_count must be a non-negative integer")
        _unique_texts(self.planned_action_ids, "planned_action_ids")
        for action_id in self.planned_action_ids:
            _identifier(action_id, "planned_action_id")
        if type(self.observed_action_ids) is not tuple or any(
            type(value) is not str or not value.strip()
            for value in self.observed_action_ids
        ):
            raise ValueError(
                "observed_action_ids must be a tuple of non-empty strings"
            )
        for action_id in self.observed_action_ids:
            _identifier(action_id, "observed_action_id")
        if (
            self.status is not WorkItemLedgerStatus.CONFLICTED
            and len({value.casefold() for value in self.observed_action_ids})
            != len(self.observed_action_ids)
        ):
            raise ValueError("non-conflicted observed action IDs must be unique")
        if not self.planned_action_ids:
            raise ValueError("inspection requires a nonempty planned action roster")
        for ids, kinds, label in (
            (self.planned_action_ids, self.planned_action_kinds, "planned"),
            (self.observed_action_ids, self.observed_action_kinds, "observed"),
        ):
            if (
                type(kinds) is not tuple
                or len(kinds) != len(ids)
                or any(not isinstance(kind, WorkItemActionKind) for kind in kinds)
            ):
                raise ValueError(f"{label} action kinds must align with their typed roster")
        for value, label in (
            (self.last_action_id, "last_action_id"),
            (self.next_action_id, "next_action_id"),
        ):
            if value is not None:
                _identifier(value, label)
        if self.event_count == 0:
            if (
                self.last_action_id is not None
                or self.last_phase is not None
                or self.observed_action_ids
            ):
                raise ValueError("empty ledger cannot have a last action or phase")
        else:
            if (
                self.last_action_id is None
                or self.last_phase is None
                or not self.observed_action_ids
                or not isinstance(self.last_phase, LedgerEventPhase)
            ):
                raise ValueError("nonempty ledger requires a typed last action and phase")
            if (
                self.status is not WorkItemLedgerStatus.CONFLICTED
                and self.last_action_id != self.observed_action_ids[-1]
            ):
                raise ValueError("last_action_id must equal the final observed action")
        if self.event_count < len(self.observed_action_ids):
            raise ValueError("event_count cannot be smaller than observed action count")
        for label in (
            "unreferenced_input_refs",
            "unreferenced_raw_receipt_refs",
            "unreferenced_scorer_decision_refs",
            "dispatch_evidence_refs",
        ):
            _unique_texts(getattr(self, label), label)
        if self.last_action_dispatch_evidence_ref is not None:
            _text(
                self.last_action_dispatch_evidence_ref,
                "last_action_dispatch_evidence_ref",
            )
            if self.last_action_dispatch_evidence_ref not in self.dispatch_evidence_refs:
                raise ValueError(
                    "last-action dispatch evidence must occur in dispatch_evidence_refs"
                )
        visible_artifact_refs = (
            self.unreferenced_input_refs
            + self.unreferenced_raw_receipt_refs
            + self.unreferenced_scorer_decision_refs
            + self.dispatch_evidence_refs
        )
        _unique_texts(visible_artifact_refs, "inspection artifact refs")
        if type(self.conflict_codes) is not tuple or any(
            not isinstance(code, LedgerConflictCode) for code in self.conflict_codes
        ):
            raise ValueError("conflict_codes must use the closed conflict vocabulary")
        if len(set(self.conflict_codes)) != len(self.conflict_codes):
            raise ValueError("conflict_codes cannot repeat")
        if self.status is WorkItemLedgerStatus.CONFLICTED:
            if not self.conflict_codes:
                raise ValueError("conflicted inspection requires conflict codes")
        elif self.conflict_codes:
            raise ValueError("non-conflicted inspection cannot contain conflicts")
        if self.status is not WorkItemLedgerStatus.CONFLICTED and (
            self.observed_action_ids
            != self.planned_action_ids[: len(self.observed_action_ids)]
        ):
            raise ValueError("observed actions must be the exact planned prefix")
        next_action_statuses = {
            WorkItemLedgerStatus.CLEAN_NOT_STARTED,
            WorkItemLedgerStatus.INPUT_PERSISTED_PREPARED_EVENT_MISSING,
            WorkItemLedgerStatus.RECOVERY_PREREQUISITE_UNBOUND,
            WorkItemLedgerStatus.ACTION_COMPLETED_NEXT_ACTION_PENDING,
        }
        if self.status in next_action_statuses:
            if self.next_action_id is None:
                raise ValueError("continuation status requires next_action_id")
            if (
                len(self.observed_action_ids) >= len(self.planned_action_ids)
                or self.next_action_id
                != self.planned_action_ids[len(self.observed_action_ids)]
            ):
                raise ValueError("next_action_id must equal the next planned action")
        elif self.next_action_id is not None:
            raise ValueError("non-continuation status cannot carry next_action_id")
        if self.status is WorkItemLedgerStatus.ACTION_PLAN_UNBOUND:
            if self.plan_binding_status is not ActionPlanBindingStatus.UNBOUND:
                raise ValueError("action-plan-unbound status requires an unbound plan")
        elif self.status is WorkItemLedgerStatus.INVENTORY_UNBOUND:
            if (
                self.plan_binding_status is not ActionPlanBindingStatus.SOURCE_BOUND
                or self.inventory_binding_status is not LedgerInventoryBindingStatus.UNBOUND
            ):
                raise ValueError("inventory-unbound status has inconsistent bindings")
        elif self.status is not WorkItemLedgerStatus.CONFLICTED and (
            self.plan_binding_status is not ActionPlanBindingStatus.SOURCE_BOUND
            or self.inventory_binding_status
            is not LedgerInventoryBindingStatus.CANONICAL_COMPLETE_BOUND
        ):
            raise ValueError(
                "classified non-conflict status requires bound plan and inventory"
            )
        if self.status in _FAILURE_STATUSES:
            if self.last_phase is not LedgerEventPhase.FAILED_RAW:
                raise ValueError("failure status requires a failed_raw final phase")
            # The nonempty observed roster above identifies the failed action.
            _validate_action_failure(
                self.observed_action_kinds[-1], self.failure_domain, self.failure_kind
            )
            method_status = self.status is WorkItemLedgerStatus.METHOD_FAILED_TERMINAL
            if method_status != (self.failure_domain is TraceFailureDomain.METHOD):
                raise ValueError("failure status and failure domain disagree")
        elif self.failure_domain is not None or self.failure_kind is not None:
            raise ValueError("non-failure status cannot carry failure classification")
        if not isinstance(self.recovery_status, RecoveryInspectionStatus):
            raise ValueError("recovery_status must use the closed vocabulary")
        if self.recovery_status is RecoveryInspectionStatus.MISMATCHED:
            if LedgerConflictCode.RECOVERY_BINDING_MISMATCH not in self.conflict_codes:
                raise ValueError("mismatched recovery requires its conflict code")
        elif LedgerConflictCode.RECOVERY_BINDING_MISMATCH in self.conflict_codes:
            raise ValueError("recovery conflict requires mismatched recovery status")
        if self.recovery_status in {
            RecoveryInspectionStatus.IMMUTABLE_INPUT,
            RecoveryInspectionStatus.SOURCE_BOUND,
            RecoveryInspectionStatus.UNBOUND,
        }:
            if self.last_phase is not LedgerEventPhase.COMPLETED_RAW:
                raise ValueError("recovery prerequisite requires a completed action")
            immutable = self.execution_semantics is ExecutionSemantics.FROZEN_TRACE
            if immutable != (self.recovery_status is RecoveryInspectionStatus.IMMUTABLE_INPUT):
                raise ValueError("recovery prerequisite disagrees with execution semantics")
        if (
            self.status is WorkItemLedgerStatus.INPUT_PERSISTED_PREPARED_EVENT_MISSING
            and self.last_phase is LedgerEventPhase.COMPLETED_RAW
            and self.recovery_status is RecoveryInspectionStatus.NOT_APPLICABLE
        ):
            raise ValueError("next-action input residue must preserve recovery status")
        if self.status is WorkItemLedgerStatus.RECOVERY_PREREQUISITE_UNBOUND:
            if self.recovery_status is not RecoveryInspectionStatus.UNBOUND:
                raise ValueError("recovery-unbound status requires unbound recovery")
        if self.status is WorkItemLedgerStatus.ACTION_COMPLETED_NEXT_ACTION_PENDING:
            if self.recovery_status not in {
                RecoveryInspectionStatus.IMMUTABLE_INPUT,
                RecoveryInspectionStatus.SOURCE_BOUND,
            }:
                raise ValueError("next-action-pending requires its recovery prerequisite")
        if self.status is not WorkItemLedgerStatus.CONFLICTED:
            self._validate_action_lifecycle_summary()
        self._validate_status_payload()
        if self.payload_origin != LEDGER_INSPECTION_PAYLOAD_ORIGIN:
            raise ValueError("inspection payload must be marked inspector_derived")
        _unverified(self.implementation_status)

    def _validate_action_lifecycle_summary(self) -> None:
        """Derive the exact event count from the frozen roster and final phase.

        Every preceding action must have completed its three-event non-scorer
        lifecycle. Conflicted diagnostics bypass this normal-chain invariant.
        """

        scorer_positions = tuple(
            index for index, kind in enumerate(self.planned_action_kinds)
            if kind is WorkItemActionKind.SCORE
        )
        if scorer_positions and scorer_positions != (len(self.planned_action_ids) - 1,):
            raise ValueError("the only planned scorer must be the final action")
        if self.observed_action_kinds != self.planned_action_kinds[
            : len(self.observed_action_ids)
        ]:
            raise ValueError("observed action kinds must be the exact planned prefix")
        if not self.observed_action_ids:
            return
        last_is_scorer = self.observed_action_kinds[-1] is WorkItemActionKind.SCORE
        if self.last_phase is LedgerEventPhase.SCORER_DECISION and not last_is_scorer:
            raise ValueError("scorer decision phase requires a scorer action")
        scorer_statuses = {
            WorkItemLedgerStatus.SCORER_DECISION_UNKNOWN,
            WorkItemLedgerStatus.SCORER_DECISION_PERSISTED_EVENT_MISSING,
            WorkItemLedgerStatus.SCORED_TERMINAL,
        }
        if self.status in scorer_statuses and not last_is_scorer:
            raise ValueError("scorer lifecycle status requires a scorer action")
        if last_is_scorer and self.recovery_status in {
            RecoveryInspectionStatus.IMMUTABLE_INPUT,
            RecoveryInspectionStatus.SOURCE_BOUND,
            RecoveryInspectionStatus.UNBOUND,
        }:
            raise ValueError("the final scorer cannot carry a continuation prerequisite")
        final_action_event_count = {
            LedgerEventPhase.PREPARED: 1,
            LedgerEventPhase.IN_FLIGHT: 2,
            LedgerEventPhase.COMPLETED_RAW: 3,
            LedgerEventPhase.FAILED_RAW: 3,
            LedgerEventPhase.SCORER_DECISION: 4,
        }[self.last_phase]
        expected_event_count = (
            3 * (len(self.observed_action_ids) - 1) + final_action_event_count
        )
        if self.event_count != expected_event_count:
            raise ValueError("event_count must equal the complete lifecycle prefix length")

    def _validate_status_payload(self) -> None:
        """Close phase, next-action, and residue truth tables per status."""

        residues = (
            len(self.unreferenced_input_refs),
            len(self.unreferenced_raw_receipt_refs),
            len(self.unreferenced_scorer_decision_refs),
        )
        if self.status is WorkItemLedgerStatus.CLEAN_NOT_STARTED:
            if (
                self.event_count != 0
                or residues != (0, 0, 0)
                or self.next_action_id is None
                or self.dispatch_evidence_refs
            ):
                raise ValueError("clean-not-started payload is contradictory")
        elif self.status is WorkItemLedgerStatus.INPUT_PERSISTED_PREPARED_EVENT_MISSING:
            if residues != (1, 0, 0) or self.next_action_id is None:
                raise ValueError("prepared-event-missing payload is contradictory")
            if self.last_phase not in (None, LedgerEventPhase.COMPLETED_RAW):
                raise ValueError("prepared-event-missing has an invalid prior phase")
        elif self.status is WorkItemLedgerStatus.DISPATCH_STATE_UNKNOWN:
            if self.last_phase is not LedgerEventPhase.PREPARED or residues != (0, 0, 0):
                raise ValueError("dispatch-unknown payload is contradictory")
            if self.last_action_dispatch_evidence_ref is not None:
                raise ValueError("dispatch-unknown cannot carry matching dispatch evidence")
        elif self.status is WorkItemLedgerStatus.DISPATCH_EVIDENCE_TERMINAL_UNKNOWN:
            if (
                self.last_phase is not LedgerEventPhase.PREPARED
                or residues != (0, 0, 0)
                or self.last_action_dispatch_evidence_ref is None
            ):
                raise ValueError("dispatch-evidence payload is contradictory")
        elif self.status is WorkItemLedgerStatus.RAW_PERSISTED_TERMINAL_EVENT_MISSING:
            if (
                self.last_phase is not LedgerEventPhase.IN_FLIGHT
                or residues != (0, 1, 0)
            ):
                raise ValueError("raw-terminal-missing payload is contradictory")
        elif self.status is WorkItemLedgerStatus.RAW_PERSISTED_IN_FLIGHT_EVENT_MISSING:
            if self.last_phase is not LedgerEventPhase.PREPARED or residues != (0, 1, 0):
                raise ValueError("raw-in-flight-missing payload is contradictory")
        elif self.status is WorkItemLedgerStatus.AMBIGUOUS_IN_FLIGHT:
            if self.last_phase is not LedgerEventPhase.IN_FLIGHT or residues != (0, 0, 0):
                raise ValueError("ambiguous-in-flight payload is contradictory")
        elif self.status in {
            WorkItemLedgerStatus.RECOVERY_PREREQUISITE_UNBOUND,
            WorkItemLedgerStatus.ACTION_COMPLETED_NEXT_ACTION_PENDING,
        }:
            if (
                self.last_phase is not LedgerEventPhase.COMPLETED_RAW
                or residues != (0, 0, 0)
                or self.next_action_id is None
            ):
                raise ValueError("post-action continuation payload is contradictory")
        elif self.status is WorkItemLedgerStatus.SCORER_DECISION_UNKNOWN:
            if self.last_phase is not LedgerEventPhase.COMPLETED_RAW or residues != (0, 0, 0):
                raise ValueError("scorer-decision-unknown payload is contradictory")
        elif self.status is WorkItemLedgerStatus.SCORER_DECISION_PERSISTED_EVENT_MISSING:
            if self.last_phase is not LedgerEventPhase.COMPLETED_RAW or residues != (0, 0, 1):
                raise ValueError("scorer-decision-event-missing payload is contradictory")
        elif self.status is WorkItemLedgerStatus.SCORED_TERMINAL:
            if self.last_phase is not LedgerEventPhase.SCORER_DECISION or residues != (0, 0, 0):
                raise ValueError("scored payload is contradictory")
            if self.observed_action_ids != self.planned_action_ids:
                raise ValueError("scored terminal must exhaust the frozen action roster")
        elif self.status in _FAILURE_STATUSES:
            if residues != (0, 0, 0):
                raise ValueError(
                    "failed terminal payload cannot retain crash-window residue"
                )
        elif self.status in {
            WorkItemLedgerStatus.ACTION_PLAN_UNBOUND,
            WorkItemLedgerStatus.INVENTORY_UNBOUND,
            WorkItemLedgerStatus.CONFLICTED,
        }:
            # These statuses preserve diagnostics without asserting a lifecycle point.
            return
        else:
            raise ValueError("ledger status is missing a payload truth-table branch")

    def to_dict(self) -> dict[str, object]:
        """Serialize the closed inspection truth table without free-form conflicts."""

        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "execution_semantics": self.execution_semantics.value,
            "plan_binding_status": self.plan_binding_status.value,
            "inventory_binding_status": self.inventory_binding_status.value,
            "status": self.status.value,
            "continuation": self.continuation.value,
            "can_continue_without_action_retry": self.can_continue_without_action_retry,
            "automatic_action_retry_allowed": self.automatic_action_retry_allowed,
            "event_count": self.event_count,
            "planned_action_ids": list(self.planned_action_ids),
            "planned_action_kinds": [kind.value for kind in self.planned_action_kinds],
            "observed_action_ids": list(self.observed_action_ids),
            "observed_action_kinds": [kind.value for kind in self.observed_action_kinds],
            "last_action_id": self.last_action_id,
            "last_phase": None if self.last_phase is None else self.last_phase.value,
            "next_action_id": self.next_action_id,
            "unreferenced_input_refs": list(self.unreferenced_input_refs),
            "unreferenced_raw_receipt_refs": list(self.unreferenced_raw_receipt_refs),
            "unreferenced_scorer_decision_refs": list(
                self.unreferenced_scorer_decision_refs
            ),
            "dispatch_evidence_refs": list(self.dispatch_evidence_refs),
            "last_action_dispatch_evidence_ref": self.last_action_dispatch_evidence_ref,
            "recovery_status": self.recovery_status.value,
            "failure_domain": (
                None if self.failure_domain is None else self.failure_domain.value
            ),
            "failure_kind": None if self.failure_kind is None else self.failure_kind.value,
            "conflict_codes": [code.value for code in self.conflict_codes],
            "payload_origin": self.payload_origin,
            "implementation_status": self.implementation_status,
        }


def _phase_prefix_is_valid(
    phases: tuple[LedgerEventPhase, ...], *, scorer_action: bool
) -> bool:
    """Accept incomplete prefixes while rejecting skipped or repeated phases."""

    completed = (
        LedgerEventPhase.PREPARED,
        LedgerEventPhase.IN_FLIGHT,
        LedgerEventPhase.COMPLETED_RAW,
    )
    failed = (
        LedgerEventPhase.PREPARED,
        LedgerEventPhase.IN_FLIGHT,
        LedgerEventPhase.FAILED_RAW,
    )
    lifecycles = [failed]
    if scorer_action:
        lifecycles.append(completed + (LedgerEventPhase.SCORER_DECISION,))
    else:
        lifecycles.append(completed)
    return any(phases == lifecycle[: len(phases)] for lifecycle in lifecycles)


def _inspection(
    work_item: ExpectedWorkItemV1,
    action_plan: WorkItemActionPlanV1,
    inventory: LedgerArtifactInventoryV1,
    *,
    status: WorkItemLedgerStatus,
    events: tuple[RunLedgerEventV1, ...],
    observed_action_ids: tuple[str, ...],
    next_action_id: str | None,
    unreferenced: dict[LedgerArtifactKind, tuple[PersistedLedgerArtifactV1, ...]],
    recovery_status: RecoveryInspectionStatus = RecoveryInspectionStatus.NOT_APPLICABLE,
    failure_domain: TraceFailureDomain | None = None,
    failure_kind: RunActionFailureKind | None = None,
    conflict_codes: tuple[LedgerConflictCode, ...] = (),
) -> WorkItemLedgerInspectionV1:
    continuation, can_continue = _STATUS_POLICY[status]
    last_event = None if not events else events[-1]
    # Preserve the first observed kind even when later events contradict it.
    observed_kinds: dict[str, WorkItemActionKind] = {}
    for event in events:
        observed_kinds.setdefault(event.action_id, event.action_kind)
    return WorkItemLedgerInspectionV1(
        run_id=work_item.run_id,
        work_item_id=work_item.work_item_id,
        execution_semantics=work_item.execution_semantics,
        plan_binding_status=action_plan.binding_status,
        inventory_binding_status=inventory.binding_status,
        status=status,
        continuation=continuation,
        can_continue_without_action_retry=can_continue,
        automatic_action_retry_allowed=False,
        event_count=len(events),
        planned_action_ids=tuple(action.action_id for action in action_plan.actions),
        planned_action_kinds=tuple(action.action_kind for action in action_plan.actions),
        observed_action_ids=observed_action_ids,
        observed_action_kinds=tuple(
            observed_kinds[action_id] for action_id in observed_action_ids
        ),
        last_action_id=None if last_event is None else last_event.action_id,
        last_phase=None if last_event is None else last_event.phase,
        next_action_id=next_action_id,
        unreferenced_input_refs=tuple(
            item.artifact_ref for item in unreferenced[LedgerArtifactKind.INPUT]
        ),
        unreferenced_raw_receipt_refs=tuple(
            item.artifact_ref for item in unreferenced[LedgerArtifactKind.RAW_RECEIPT]
        ),
        unreferenced_scorer_decision_refs=tuple(
            item.artifact_ref
            for item in unreferenced[LedgerArtifactKind.SCORER_DECISION]
        ),
        dispatch_evidence_refs=tuple(
            item.artifact_ref
            for item in inventory.artifacts
            if item.artifact_kind is LedgerArtifactKind.DISPATCH_EVIDENCE
        ),
        last_action_dispatch_evidence_ref=(
            None
            if last_event is None
            else next(
                (
                    item.artifact_ref
                    for item in inventory.artifacts
                    if item.artifact_kind is LedgerArtifactKind.DISPATCH_EVIDENCE
                    and item.action_id == last_event.action_id
                ),
                None,
            )
        ),
        recovery_status=recovery_status,
        failure_domain=failure_domain,
        failure_kind=failure_kind,
        conflict_codes=tuple(dict.fromkeys(conflict_codes)),
    )


def _recovery_is_bound(
    work_item: ExpectedWorkItemV1,
    last_action_id: str,
    recovery: ExecutionRecoveryBindingV1 | None,
) -> tuple[bool, bool]:
    """Return (bound, mismatched) for the exact post-action recovery point."""

    if recovery is not None:
        matches = (
            recovery.run_id == work_item.run_id
            and recovery.work_item_id == work_item.work_item_id
            and recovery.after_action_id == last_action_id
            and recovery.execution_semantics is work_item.execution_semantics
            and recovery.recovery_mode
            is _RECOVERY_MODE_BY_EXECUTION[work_item.execution_semantics]
        )
        if not matches:
            return (False, True)
    if work_item.execution_semantics is ExecutionSemantics.FROZEN_TRACE:
        return (True, False)
    if recovery is None:
        return (False, False)
    return (
        recovery.binding_status is ExecutionRecoveryBindingStatus.SOURCE_BOUND,
        False,
    )


def inspect_work_item_ledger_v1(
    work_item: ExpectedWorkItemV1,
    action_plan: WorkItemActionPlanV1,
    events: tuple[RunLedgerEventV1, ...],
    inventory: LedgerArtifactInventoryV1,
    recovery: ExecutionRecoveryBindingV1 | None = None,
) -> WorkItemLedgerInspectionV1:
    """Validate one full planned prefix and classify its sole structural next step.

    Positive continuations require a source-bound action plan and a complete,
    canonical, schema-decoded inventory. Mutable-state continuations also
    require the recovery mode fixed by the work item's execution semantics.
    """

    if type(work_item) is not ExpectedWorkItemV1:
        raise ValueError("work_item must be exactly ExpectedWorkItemV1")
    if type(action_plan) is not WorkItemActionPlanV1:
        raise ValueError("action_plan must be exactly WorkItemActionPlanV1")
    if type(events) is not tuple or any(
        type(event) is not RunLedgerEventV1 for event in events
    ):
        raise ValueError("events must be an ordered tuple of RunLedgerEventV1")
    if type(inventory) is not LedgerArtifactInventoryV1:
        raise ValueError("inventory must be exactly LedgerArtifactInventoryV1")
    if recovery is not None and type(recovery) is not ExecutionRecoveryBindingV1:
        raise ValueError("recovery must be ExecutionRecoveryBindingV1 or null")

    conflicts: list[LedgerConflictCode] = []
    if (
        action_plan.run_id != work_item.run_id
        or action_plan.work_item_id != work_item.work_item_id
        or action_plan.execution_semantics is not work_item.execution_semantics
    ):
        conflicts.append(LedgerConflictCode.PLAN_OWNER_MISMATCH)
    if (
        inventory.run_id != work_item.run_id
        or inventory.work_item_id != work_item.work_item_id
    ):
        conflicts.append(LedgerConflictCode.INVENTORY_OWNER_MISMATCH)

    planned_actions = action_plan.actions
    planned_by_id = {action.action_id: action for action in planned_actions}
    observed_action_ids: list[str] = []
    action_events: dict[str, list[RunLedgerEventV1]] = {}
    event_ids: set[str] = set()
    previous_event: RunLedgerEventV1 | None = None
    previous_action_id: str | None = None

    for expected_sequence, event in enumerate(events):
        if event.run_id != work_item.run_id or event.work_item_id != work_item.work_item_id:
            conflicts.append(LedgerConflictCode.EVENT_OWNER_MISMATCH)
        if event.event_sequence != expected_sequence:
            conflicts.append(LedgerConflictCode.EVENT_SEQUENCE_MISMATCH)
        event_key = event.event_id.casefold()
        if event_key in event_ids:
            conflicts.append(LedgerConflictCode.EVENT_ID_DUPLICATE)
        event_ids.add(event_key)
        if previous_event is None:
            if event.predecessor_event_id is not None:
                conflicts.append(LedgerConflictCode.PREDECESSOR_MISMATCH)
        elif event.predecessor_event_id != previous_event.event_id:
            conflicts.append(LedgerConflictCode.PREDECESSOR_MISMATCH)

        spec = planned_by_id.get(event.action_id)
        if spec is None:
            conflicts.append(LedgerConflictCode.ACTION_NOT_PLANNED)
        elif (
            event.action_kind is not spec.action_kind
            or event.resource_role is not spec.resource_role
        ):
            conflicts.append(LedgerConflictCode.ACTION_KIND_ROLE_MISMATCH)

        if event.action_id != previous_action_id:
            if event.action_id in action_events:
                conflicts.append(LedgerConflictCode.ACTION_REOPENED)
            else:
                expected_action_index = len(observed_action_ids)
                if (
                    expected_action_index >= len(planned_actions)
                    or event.action_id != planned_actions[expected_action_index].action_id
                ):
                    conflicts.append(LedgerConflictCode.ACTION_ORDER_MISMATCH)
                observed_action_ids.append(event.action_id)
                action_events[event.action_id] = []
            if previous_event is not None and previous_event.phase not in {
                LedgerEventPhase.COMPLETED_RAW,
                LedgerEventPhase.FAILED_RAW,
                LedgerEventPhase.SCORER_DECISION,
            }:
                conflicts.append(
                    LedgerConflictCode.ACTION_STARTED_BEFORE_PRIOR_TERMINAL
                )
            if previous_event is not None and previous_event.phase in {
                LedgerEventPhase.FAILED_RAW,
                LedgerEventPhase.SCORER_DECISION,
            }:
                conflicts.append(LedgerConflictCode.ACTION_AFTER_WORK_ITEM_TERMINAL)
        action_events.setdefault(event.action_id, []).append(event)
        previous_event = event
        previous_action_id = event.action_id

    for action_id in observed_action_ids:
        records = action_events[action_id]
        spec = planned_by_id.get(action_id)
        if spec is None:
            continue
        phases = tuple(record.phase for record in records)
        if not _phase_prefix_is_valid(
            phases, scorer_action=spec.resource_role is ResourceRole.SCORER
        ):
            conflicts.append(LedgerConflictCode.ACTION_PHASE_SEQUENCE_INVALID)
    if sum(event.phase is LedgerEventPhase.SCORER_DECISION for event in events) > 1:
        conflicts.append(LedgerConflictCode.MULTIPLE_SCORER_DECISIONS)

    inventory_by_ref: dict[str, PersistedLedgerArtifactV1] = {}
    for item in inventory.artifacts:
        if item.action_id not in planned_by_id:
            conflicts.append(LedgerConflictCode.INVENTORY_ACTION_NOT_PLANNED)
        if (
            item.artifact_kind is LedgerArtifactKind.DISPATCH_EVIDENCE
            and item.action_id not in action_events
        ):
            # Unclassified dispatch evidence cannot authorize a missing action start.
            conflicts.append(
                LedgerConflictCode.DISPATCH_EVIDENCE_WITHOUT_PREPARED_EVENT
            )
        inventory_by_ref[item.artifact_ref] = item

    referenced_refs: set[str] = set()
    ref_owners: dict[str, str] = {}
    for event in events:
        event_refs = (
            (LedgerArtifactKind.INPUT, event.input_ref),
            (LedgerArtifactKind.RAW_RECEIPT, event.raw_receipt_ref),
            (LedgerArtifactKind.SCORER_DECISION, event.scorer_decision_ref),
        )
        for expected_kind, artifact_ref in event_refs:
            if artifact_ref is None:
                continue
            prior_owner = ref_owners.setdefault(artifact_ref, event.action_id)
            if prior_owner != event.action_id:
                conflicts.append(LedgerConflictCode.ARTIFACT_REF_CROSS_ACTION)
            inventory_item = inventory_by_ref.get(artifact_ref)
            if inventory_item is None:
                conflicts.append(LedgerConflictCode.REFERENCED_ARTIFACT_MISSING)
                continue
            referenced_refs.add(artifact_ref)
            if inventory_item.artifact_kind is not expected_kind:
                conflicts.append(LedgerConflictCode.ARTIFACT_KIND_MISMATCH)
            if inventory_item.action_id != event.action_id:
                conflicts.append(LedgerConflictCode.ARTIFACT_ACTION_OWNER_MISMATCH)

    unreferenced = {
        kind: tuple(
            item
            for item in inventory.artifacts
            if item.artifact_kind is kind and item.artifact_ref not in referenced_refs
        )
        for kind in LedgerArtifactKind
    }
    observed_tuple = tuple(observed_action_ids)
    next_spec: WorkItemActionSpecV1 | None = None
    if len(observed_action_ids) < len(planned_actions):
        next_spec = planned_actions[len(observed_action_ids)]

    recovery_status = RecoveryInspectionStatus.NOT_APPLICABLE
    if recovery is not None and (
        recovery.run_id != work_item.run_id
        or recovery.work_item_id != work_item.work_item_id
        or recovery.execution_semantics is not work_item.execution_semantics
        or not events
        or recovery.after_action_id != events[-1].action_id
        or events[-1].phase is not LedgerEventPhase.COMPLETED_RAW
        or events[-1].resource_role is ResourceRole.SCORER
        or next_spec is None
    ):
        conflicts.append(LedgerConflictCode.RECOVERY_BINDING_MISMATCH)
        recovery_status = RecoveryInspectionStatus.MISMATCHED

    # Inspect recovery before residue branches so an input file cannot hide it.
    if (
        recovery_status is not RecoveryInspectionStatus.MISMATCHED
        and events
        and events[-1].phase is LedgerEventPhase.COMPLETED_RAW
        and events[-1].resource_role is not ResourceRole.SCORER
        and next_spec is not None
    ):
        recovery_bound, recovery_mismatched = _recovery_is_bound(
            work_item, events[-1].action_id, recovery
        )
        if recovery_mismatched:
            conflicts.append(LedgerConflictCode.RECOVERY_BINDING_MISMATCH)
            recovery_status = RecoveryInspectionStatus.MISMATCHED
        elif work_item.execution_semantics is ExecutionSemantics.FROZEN_TRACE:
            recovery_status = RecoveryInspectionStatus.IMMUTABLE_INPUT
        else:
            recovery_status = (
                RecoveryInspectionStatus.SOURCE_BOUND
                if recovery_bound else RecoveryInspectionStatus.UNBOUND
            )

    if conflicts:
        return _inspection(
            work_item, action_plan, inventory,
            status=WorkItemLedgerStatus.CONFLICTED,
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=None,
            unreferenced=unreferenced,
            conflict_codes=tuple(conflicts),
            recovery_status=recovery_status,
        )
    if action_plan.binding_status is ActionPlanBindingStatus.UNBOUND:
        return _inspection(
            work_item, action_plan, inventory,
            status=WorkItemLedgerStatus.ACTION_PLAN_UNBOUND,
            recovery_status=recovery_status,
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=None,
            unreferenced=unreferenced,
        )
    if inventory.binding_status is LedgerInventoryBindingStatus.UNBOUND:
        return _inspection(
            work_item, action_plan, inventory,
            status=WorkItemLedgerStatus.INVENTORY_UNBOUND,
            recovery_status=recovery_status,
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=None,
            unreferenced=unreferenced,
        )

    inputs = unreferenced[LedgerArtifactKind.INPUT]
    raws = unreferenced[LedgerArtifactKind.RAW_RECEIPT]
    decisions = unreferenced[LedgerArtifactKind.SCORER_DECISION]
    residue_count = len(inputs) + len(raws) + len(decisions)

    if not events:
        if not inventory.artifacts:
            status = WorkItemLedgerStatus.CLEAN_NOT_STARTED
            next_action_id = planned_actions[0].action_id
        elif (
            len(inputs) == 1
            and residue_count == 1
            and inputs[0].action_id == planned_actions[0].action_id
        ):
            status = WorkItemLedgerStatus.INPUT_PERSISTED_PREPARED_EVENT_MISSING
            next_action_id = planned_actions[0].action_id
        else:
            return _inspection(
                work_item, action_plan, inventory,
                status=WorkItemLedgerStatus.CONFLICTED,
                events=events,
                observed_action_ids=observed_tuple,
                next_action_id=None,
                unreferenced=unreferenced,
                conflict_codes=(LedgerConflictCode.ARTIFACT_RESIDUE_UNCLASSIFIABLE,),
            )
        return _inspection(
            work_item, action_plan, inventory,
            status=status,
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=next_action_id,
            unreferenced=unreferenced,
        )

    last_event = events[-1]
    current_spec = planned_by_id[last_event.action_id]
    if (
        len(inputs) == 1
        and residue_count == 1
        and last_event.phase is LedgerEventPhase.COMPLETED_RAW
        and current_spec.resource_role is not ResourceRole.SCORER
        and next_spec is not None
        and inputs[0].action_id == next_spec.action_id
    ):
        return _inspection(
            work_item, action_plan, inventory,
            status=WorkItemLedgerStatus.INPUT_PERSISTED_PREPARED_EVENT_MISSING,
            recovery_status=recovery_status,
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=next_spec.action_id,
            unreferenced=unreferenced,
        )

    if (
        len(raws) == 1
        and residue_count == 1
        and last_event.phase in {LedgerEventPhase.PREPARED, LedgerEventPhase.IN_FLIGHT}
        and raws[0].action_id == last_event.action_id
    ):
        return _inspection(
            work_item, action_plan, inventory,
            status=(
                WorkItemLedgerStatus.RAW_PERSISTED_IN_FLIGHT_EVENT_MISSING
                if last_event.phase is LedgerEventPhase.PREPARED
                else WorkItemLedgerStatus.RAW_PERSISTED_TERMINAL_EVENT_MISSING
            ),
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=None,
            unreferenced=unreferenced,
        )

    if last_event.phase is LedgerEventPhase.PREPARED and residue_count == 0:
        dispatch_evidence_present = any(
            item.action_id == last_event.action_id
            for item in unreferenced[LedgerArtifactKind.DISPATCH_EVIDENCE]
        )
        return _inspection(
            work_item, action_plan, inventory,
            status=(
                WorkItemLedgerStatus.DISPATCH_EVIDENCE_TERMINAL_UNKNOWN
                if dispatch_evidence_present
                else WorkItemLedgerStatus.DISPATCH_STATE_UNKNOWN
            ),
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=None,
            unreferenced=unreferenced,
        )
    if last_event.phase is LedgerEventPhase.IN_FLIGHT and residue_count == 0:
        return _inspection(
            work_item, action_plan, inventory,
            status=WorkItemLedgerStatus.AMBIGUOUS_IN_FLIGHT,
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=None,
            unreferenced=unreferenced,
        )

    if last_event.phase is LedgerEventPhase.COMPLETED_RAW:
        if current_spec.resource_role is ResourceRole.SCORER:
            if (
                len(decisions) == 1
                and residue_count == 1
                and decisions[0].action_id == last_event.action_id
            ):
                status = WorkItemLedgerStatus.SCORER_DECISION_PERSISTED_EVENT_MISSING
            elif residue_count == 0:
                status = WorkItemLedgerStatus.SCORER_DECISION_UNKNOWN
            else:
                status = WorkItemLedgerStatus.CONFLICTED
            return _inspection(
                work_item, action_plan, inventory,
                status=status,
                events=events,
                observed_action_ids=observed_tuple,
                next_action_id=None,
                unreferenced=unreferenced,
                conflict_codes=(
                    () if status is not WorkItemLedgerStatus.CONFLICTED
                    else (LedgerConflictCode.ARTIFACT_RESIDUE_UNCLASSIFIABLE,)
                ),
            )
        if residue_count == 0 and next_spec is not None:
            status = (
                WorkItemLedgerStatus.ACTION_COMPLETED_NEXT_ACTION_PENDING
                if recovery_status in {
                    RecoveryInspectionStatus.IMMUTABLE_INPUT,
                    RecoveryInspectionStatus.SOURCE_BOUND,
                }
                else WorkItemLedgerStatus.RECOVERY_PREREQUISITE_UNBOUND
            )
            return _inspection(
                work_item, action_plan, inventory,
                status=status,
                recovery_status=recovery_status,
                events=events,
                observed_action_ids=observed_tuple,
                next_action_id=next_spec.action_id,
                unreferenced=unreferenced,
            )
        if residue_count == 0 and next_spec is None:
            return _inspection(
                work_item, action_plan, inventory,
                status=WorkItemLedgerStatus.CONFLICTED,
                events=events,
                observed_action_ids=observed_tuple,
                next_action_id=None,
                unreferenced=unreferenced,
                conflict_codes=(
                    LedgerConflictCode.PLAN_EXHAUSTED_WITHOUT_SCORER_DECISION,
                ),
            )

    if last_event.phase is LedgerEventPhase.SCORER_DECISION and residue_count == 0:
        return _inspection(
            work_item, action_plan, inventory,
            status=WorkItemLedgerStatus.SCORED_TERMINAL,
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=None,
            unreferenced=unreferenced,
        )

    if last_event.phase is LedgerEventPhase.FAILED_RAW and residue_count == 0:
        if last_event.failure_domain is TraceFailureDomain.METHOD:
            status = WorkItemLedgerStatus.METHOD_FAILED_TERMINAL
        else:
            status = WorkItemLedgerStatus.INFRASTRUCTURE_FAILED_TERMINAL
        return _inspection(
            work_item, action_plan, inventory,
            status=status,
            events=events,
            observed_action_ids=observed_tuple,
            next_action_id=None,
            unreferenced=unreferenced,
            failure_domain=last_event.failure_domain,
            failure_kind=last_event.failure_kind,
        )

    return _inspection(
        work_item, action_plan, inventory,
        status=WorkItemLedgerStatus.CONFLICTED,
        recovery_status=recovery_status,
        events=events,
        observed_action_ids=observed_tuple,
        next_action_id=None,
        unreferenced=unreferenced,
        conflict_codes=(LedgerConflictCode.ARTIFACT_RESIDUE_UNCLASSIFIABLE,),
    )


__all__ = [
    "EXECUTION_RECOVERY_BINDING_SCHEMA",
    "LEDGER_ARTIFACT_INVENTORY_SCHEMA",
    "LEDGER_INSPECTION_PAYLOAD_ORIGIN",
    "PERSISTED_LEDGER_ARTIFACT_SCHEMA",
    "WORK_ITEM_LEDGER_INSPECTION_SCHEMA",
    "ExecutionRecoveryBindingStatus",
    "ExecutionRecoveryBindingV1",
    "ExecutionRecoveryMode",
    "LedgerArtifactInventoryV1",
    "LedgerArtifactKind",
    "LedgerConflictCode",
    "LedgerContinuation",
    "LedgerInventoryBindingStatus",
    "PersistedLedgerArtifactV1",
    "RecoveryInspectionStatus",
    "WorkItemLedgerInspectionV1",
    "WorkItemLedgerStatus",
    "inspect_work_item_ledger_v1",
]
