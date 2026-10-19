"""Pure run, action-ledger, and result-roster contracts for TraceFix-Bench v0.1.

These locally designed records freeze planned work before method execution and
retain every planned cell after execution. They do not read files, execute
actions, score outcomes, or establish that a published release is valid.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracefix_bench.contracts import (
    BenchmarkSplit,
    ExecutionSemantics,
    TraceFailureDomain,
    TraceFixTrack,
    UsageValueStatus,
)
from lh_platform.tracefix_bench.dataset_contracts import TRACEFIX_BENCH_SUITE_ID
from lh_platform.tracefix_bench.run_identity_contracts import (
    ARTIFACT_TOPOLOGY_ID_V0_1,
    METERING_CONTRACT_ID_V0_1,
    RUN_METERING_BOUNDARY_ID_V0_1,
    WORK_ITEM_METERING_BOUNDARY_ID_V0_1,
)
from lh_platform.tracefix_bench.scorer_contracts import (
    BudgetScope,
    ResourceDimension,
    ResourceLimitV1,
    ResourceRole,
    SCORER_REGISTRY_SUITE_VERSION_V0_1,
    STATIC_NO_FIXER_TREATMENT_ID,
)

RUN_BUDGET_SCHEMA = "longhorizon_tracefix_bench/run_budget/v1"
TREATMENT_ARM_SCHEMA = "longhorizon_tracefix_bench/treatment_arm/v1"
TRACK_RUN_CELL_SCHEMA = "longhorizon_tracefix_bench/track_run_cell/v1"
TRACK_RUN_MATRIX_SCHEMA = "longhorizon_tracefix_bench/track_run_matrix/v1"
EXPECTED_WORK_ITEM_SCHEMA = "longhorizon_tracefix_bench/expected_work_item/v1"
TRACEFIX_RUN_PLAN_SCHEMA = "longhorizon_tracefix_bench/run_plan/v2"
RUN_ONLY_OPERATION_SPEC_SCHEMA = "longhorizon_tracefix_bench/run_only_operation_spec/v1"
RUN_ONLY_INPUT_ROOT_SCHEMA = "longhorizon_tracefix_bench/run_only_input_root/v1"
RUN_ONLY_OPERATION_PLAN_SCHEMA = "longhorizon_tracefix_bench/run_only_operation_plan/v1"
ALLOWED_CLOSE_PLAN_SCHEMA = "longhorizon_tracefix_bench/allowed_close_plan/v1"
RUN_ONLY_USAGE_RECEIPT_SCHEMA = "longhorizon_tracefix_bench/run_only_usage_receipt/v1"
WORK_ITEM_BOUNDARY_SCHEMA_V2 = "longhorizon_tracefix_bench/work_item_boundary/v2"
RUN_BOUNDARY_SCHEMA_V2 = "longhorizon_tracefix_bench/run_boundary/v2"
RESOURCE_BALANCE_SCHEMA = "longhorizon_tracefix_bench/resource_balance/v1"
RUN_LEDGER_EVENT_SCHEMA = "longhorizon_tracefix_bench/run_ledger_event/v1"
WORK_ITEM_ACTION_SPEC_SCHEMA = (
    "longhorizon_tracefix_bench/work_item_action_spec/v1"
)
WORK_ITEM_ACTION_PLAN_SCHEMA = (
    "longhorizon_tracefix_bench/work_item_action_plan/v1"
)
RESOURCE_COST_SCHEMA = "longhorizon_tracefix_bench/resource_cost/v1"
WORK_ITEM_RESULT_INDEX_ENTRY_SCHEMA = (
    "longhorizon_tracefix_bench/work_item_result_index_entry/v1"
)
TRACEFIX_RESULT_INDEX_SCHEMA = "longhorizon_tracefix_bench/result_index/v1"
RUN_CONTRACT_IMPLEMENTATION_STATUS = "implementation_unverified"

_IDENTIFIER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_COUNT_DIMENSIONS = frozenset(
    {
        ResourceDimension.MODEL_CALLS,
        ResourceDimension.INPUT_TOKENS,
        ResourceDimension.OUTPUT_TOKENS,
        ResourceDimension.CACHE_READ_TOKENS,
        ResourceDimension.CACHE_WRITE_TOKENS,
        ResourceDimension.TOOL_CALLS,
        ResourceDimension.ENVIRONMENT_ACTIONS,
        ResourceDimension.PROBE_ACTIONS,
        ResourceDimension.CHECK_ACTIONS,
        ResourceDimension.REPAIR_ACTIONS,
        ResourceDimension.REPLAY_ATTEMPTS,
    }
)


def _identifier(value: str, label: str) -> None:
    """Require a portable, single-component identity without normalization."""

    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable non-empty identifier")


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _unique_texts(values: tuple[str, ...], label: str, *, portable: bool) -> None:
    if type(values) is not tuple:
        raise ValueError(f"{label} must be a tuple")
    for value in values:
        if portable:
            _identifier(value, f"{label} item")
        else:
            _text(value, f"{label} item")
    if len({value.casefold() for value in values}) != len(values):
        raise ValueError(f"{label} must be unique after case folding")


def _records(values: tuple[object, ...], kind: type, label: str) -> None:
    if type(values) is not tuple or any(type(value) is not kind for value in values):
        raise ValueError(f"{label} must be a tuple of {kind.__name__}")


def _count(value: int, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _finite(value: int | float, label: str, *, dimension: ResourceDimension) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    if dimension in _COUNT_DIMENSIONS and type(value) is not int:
        raise ValueError(f"{label} must be an integer for count and token dimensions")


def _unverified(value: str) -> None:
    if type(value) is not str or value != RUN_CONTRACT_IMPLEMENTATION_STATUS:
        raise ValueError("run contracts cannot promote implementation status")


@dataclass(frozen=True)
class RunBudgetV1:
    """Keep whole-run limits separate from each work item's registered budget."""

    budget_id: str
    limits: tuple[ResourceLimitV1, ...]
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = RUN_BUDGET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUN_BUDGET_SCHEMA:
            raise ValueError("unsupported run budget schema")
        _identifier(self.budget_id, "run budget_id")
        _records(self.limits, ResourceLimitV1, "run limits")
        if not self.limits or any(limit.scope is not BudgetScope.WHOLE_RUN for limit in self.limits):
            raise ValueError("run budget requires only whole-run resource limits")
        keys = tuple((limit.role, limit.scope, limit.dimension) for limit in self.limits)
        if len(set(keys)) != len(keys):
            raise ValueError("run budget cannot repeat a role, scope, and dimension")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize every whole-run limit and the unverified state."""

        return {
            "schema": self.schema,
            "budget_id": self.budget_id,
            "limits": [limit.to_dict() for limit in self.limits],
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class TreatmentArmV1:
    """Bind one public treatment identity to its method pair and configurations."""

    treatment_id: str
    locator_treatment_id: str
    fixer_treatment_id: str
    method_config_refs: tuple[str, ...]
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = TREATMENT_ARM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TREATMENT_ARM_SCHEMA:
            raise ValueError("unsupported treatment arm schema")
        for label in (
            "treatment_id",
            "locator_treatment_id",
            "fixer_treatment_id",
        ):
            _identifier(getattr(self, label), label)
        _unique_texts(self.method_config_refs, "method_config_refs", portable=False)
        if not self.method_config_refs:
            raise ValueError("treatment arm requires at least one method config ref")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the treatment-to-method mapping without runtime outcomes."""

        return {
            "schema": self.schema,
            "treatment_id": self.treatment_id,
            "locator_treatment_id": self.locator_treatment_id,
            "fixer_treatment_id": self.fixer_treatment_id,
            "method_config_refs": list(self.method_config_refs),
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class TrackRunCellV1:
    """Declare one registry-compatible system, treatment, and budget join row."""

    system_cell_id: str
    treatment_id: str
    budget_id: str
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = TRACK_RUN_CELL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACK_RUN_CELL_SCHEMA:
            raise ValueError("unsupported track run cell schema")
        for label in ("system_cell_id", "treatment_id", "budget_id"):
            _identifier(getattr(self, label), label)
        _unverified(self.implementation_status)

    def identity(self) -> tuple[str, str, str]:
        """Return the ordered registry join key used by the structural binder."""

        return (self.system_cell_id, self.treatment_id, self.budget_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize the explicit legal join row without inferred cross products."""

        return {
            "schema": self.schema,
            "system_cell_id": self.system_cell_id,
            "treatment_id": self.treatment_id,
            "budget_id": self.budget_id,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class TrackRunMatrixV1:
    """Freeze one cohort and its complete registry-compatible join roster.

    Historical arm-order schedules and sham semantics require later typed
    contracts; repetition IDs here do not claim that those policies are bound.
    """

    matrix_id: str
    track: TraceFixTrack
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    split: BenchmarkSplit
    execution_semantics: ExecutionSemantics
    registered_cells: tuple[TrackRunCellV1, ...]
    repetition_ids: tuple[str, ...]
    complete_registry_join: bool = True
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = TRACK_RUN_MATRIX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACK_RUN_MATRIX_SCHEMA:
            raise ValueError("unsupported track run matrix schema")
        for label in (
            "matrix_id",
            "benchmark_id",
            "benchmark_version",
            "adapter_version",
        ):
            _identifier(getattr(self, label), label)
        if not isinstance(self.track, TraceFixTrack):
            raise ValueError("matrix track must be a TraceFixTrack")
        if not isinstance(self.split, BenchmarkSplit):
            raise ValueError("matrix split must be a BenchmarkSplit")
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError(
                "matrix execution_semantics must be an ExecutionSemantics"
            )
        _records(self.registered_cells, TrackRunCellV1, "registered_cells")
        if not self.registered_cells:
            raise ValueError("matrix registered_cells cannot be empty")
        cell_keys = tuple(
            tuple(part.casefold() for part in cell.identity())
            for cell in self.registered_cells
        )
        if len(set(cell_keys)) != len(cell_keys):
            raise ValueError("matrix registered_cells must be unique after case folding")
        system_bindings: dict[str, tuple[str, str]] = {}
        treatment_spellings: dict[str, str] = {}
        budget_spellings: dict[str, str] = {}
        for cell in self.registered_cells:
            system_key = cell.system_cell_id.casefold()
            system_binding = (cell.system_cell_id, cell.budget_id)
            prior_system = system_bindings.setdefault(system_key, system_binding)
            if prior_system != system_binding:
                raise ValueError(
                    "one matrix system cell must retain exact spelling and one budget"
                )
            for value, spellings, label in (
                (cell.treatment_id, treatment_spellings, "treatment"),
                (cell.budget_id, budget_spellings, "budget"),
            ):
                prior = spellings.setdefault(value.casefold(), value)
                if prior != value:
                    raise ValueError(
                        f"matrix {label} IDs must retain exact spelling"
                    )
        _unique_texts(self.repetition_ids, "repetition_ids", portable=True)
        if not self.repetition_ids:
            raise ValueError("matrix repetition_ids cannot be empty")
        if self.complete_registry_join is not True:
            raise ValueError("v0.1 matrices require the complete registry join")
        _unverified(self.implementation_status)

    @property
    def system_cell_ids(self) -> tuple[str, ...]:
        """Expose system IDs in first registered-cell order for diagnostics."""

        return tuple(dict.fromkeys(cell.system_cell_id for cell in self.registered_cells))

    @property
    def treatment_ids(self) -> tuple[str, ...]:
        """Expose treatment IDs in first registered-cell order for diagnostics."""

        return tuple(dict.fromkeys(cell.treatment_id for cell in self.registered_cells))

    @property
    def budget_ids(self) -> tuple[str, ...]:
        """Expose budgets in first registered-cell order for diagnostics."""

        return tuple(dict.fromkeys(cell.budget_id for cell in self.registered_cells))

    def cohort_identity(self) -> tuple[object, ...]:
        """Return the complete dataset-derived cohort key."""

        return (
            self.track,
            self.benchmark_id,
            self.benchmark_version,
            self.adapter_version,
            self.split,
            self.execution_semantics,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize cohort identity, legal join rows, and repetition order."""

        return {
            "schema": self.schema,
            "matrix_id": self.matrix_id,
            "track": self.track.value,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "split": self.split.value,
            "execution_semantics": self.execution_semantics.value,
            "registered_cells": [cell.to_dict() for cell in self.registered_cells],
            "repetition_ids": list(self.repetition_ids),
            "complete_registry_join": self.complete_registry_join,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ExpectedWorkItemV1:
    """Freeze one complete method cell independently of execution order or result."""

    work_item_id: str
    run_id: str
    population_id: str
    case_id: str
    source_task_id: str
    track: TraceFixTrack
    split: BenchmarkSplit
    execution_semantics: ExecutionSemantics
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    system_cell_id: str
    treatment_id: str
    locator_treatment_id: str
    fixer_treatment_id: str
    budget_id: str
    repetition_id: str
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = EXPECTED_WORK_ITEM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXPECTED_WORK_ITEM_SCHEMA:
            raise ValueError("unsupported expected work item schema")
        for label in (
            "work_item_id", "run_id", "population_id", "case_id", "benchmark_id",
            "benchmark_version", "adapter_version", "system_cell_id",
            "treatment_id", "locator_treatment_id", "fixer_treatment_id", "budget_id",
            "repetition_id",
        ):
            _identifier(getattr(self, label), label)
        _text(self.source_task_id, "source_task_id")
        if not isinstance(self.track, TraceFixTrack):
            raise ValueError("track must be a TraceFixTrack")
        if not isinstance(self.split, BenchmarkSplit):
            raise ValueError("split must be a BenchmarkSplit")
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError("execution_semantics must be an ExecutionSemantics")
        if self.track is TraceFixTrack.STATIC_ATTRIBUTION:
            if self.fixer_treatment_id != STATIC_NO_FIXER_TREATMENT_ID:
                raise ValueError("static attribution requires the fixed no-fixer treatment")
        elif self.fixer_treatment_id.casefold() == STATIC_NO_FIXER_TREATMENT_ID:
            raise ValueError("non-static work items require a registered fixer treatment")
        _unverified(self.implementation_status)

    def identity(self) -> tuple[str, ...]:
        """Expose the complete stable cell key for plan-level duplicate rejection."""

        return (
            self.run_id, self.population_id, self.case_id, self.source_task_id,
            self.track.value, self.split.value, self.execution_semantics.value,
            self.benchmark_id, self.benchmark_version, self.adapter_version,
            self.system_cell_id, self.treatment_id, self.locator_treatment_id,
            self.fixer_treatment_id,
            self.budget_id, self.repetition_id,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the full cell identity without losing scorer-side fields."""

        return {
            "schema": self.schema,
            "work_item_id": self.work_item_id,
            "run_id": self.run_id,
            "population_id": self.population_id,
            "case_id": self.case_id,
            "source_task_id": self.source_task_id,
            "track": self.track.value,
            "split": self.split.value,
            "execution_semantics": self.execution_semantics.value,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "system_cell_id": self.system_cell_id,
            "treatment_id": self.treatment_id,
            "locator_treatment_id": self.locator_treatment_id,
            "fixer_treatment_id": self.fixer_treatment_id,
            "budget_id": self.budget_id,
            "repetition_id": self.repetition_id,
            "implementation_status": self.implementation_status,
        }


class WorkItemActionKind(StrEnum):
    """Name an action's runtime responsibility independently of its display ID."""

    LOCATE = "locate"
    FIX = "fix"
    ACT = "act"
    ENVIRONMENT_TRANSITION = "environment_transition"
    RESTORE = "restore"
    PROBE = "probe"
    CHECK = "check"
    REPAIR = "repair"
    REPLAY = "replay"
    OFFICIAL_VERIFY = "official_verify"
    SCORE = "score"
    RECOVER = "recover"
    PERSIST = "persist"
    ORCHESTRATE = "orchestrate"


_RESOURCE_ROLE_BY_ACTION_KIND = {
    WorkItemActionKind.LOCATE: ResourceRole.LOCATOR,
    WorkItemActionKind.FIX: ResourceRole.FIXER,
    WorkItemActionKind.ACT: ResourceRole.ACTOR,
    WorkItemActionKind.ENVIRONMENT_TRANSITION: ResourceRole.ENVIRONMENT,
    WorkItemActionKind.RESTORE: ResourceRole.RESTORE,
    WorkItemActionKind.PROBE: ResourceRole.PROBE,
    WorkItemActionKind.CHECK: ResourceRole.CHECK,
    WorkItemActionKind.REPAIR: ResourceRole.REPAIR,
    WorkItemActionKind.REPLAY: ResourceRole.REPLAY,
    WorkItemActionKind.OFFICIAL_VERIFY: ResourceRole.OFFICIAL_VERIFIER,
    WorkItemActionKind.SCORE: ResourceRole.SCORER,
    WorkItemActionKind.RECOVER: ResourceRole.RECOVERY,
    WorkItemActionKind.PERSIST: ResourceRole.PERSISTENCE,
    WorkItemActionKind.ORCHESTRATE: ResourceRole.ORCHESTRATION,
}


def _resource_role_for_action_kind(action_kind: WorkItemActionKind) -> ResourceRole:
    """Resolve the sole resource role for a typed action without changing state."""

    if type(action_kind) is not WorkItemActionKind:
        raise ValueError("action_kind must use the closed action vocabulary")
    return _RESOURCE_ROLE_BY_ACTION_KIND[action_kind]


class ActionPlanBindingStatus(StrEnum):
    """Separate an explicit roster from a roster checked against runtime sources."""

    UNBOUND = "unbound"
    SOURCE_BOUND = "source_bound"


@dataclass(frozen=True)
class WorkItemActionSpecV1:
    """Freeze one portable action identity, order position, kind, and resource role."""

    action_id: str
    action_sequence: int
    action_kind: WorkItemActionKind
    resource_role: ResourceRole
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = WORK_ITEM_ACTION_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WORK_ITEM_ACTION_SPEC_SCHEMA:
            raise ValueError("unsupported work-item action spec schema")
        _identifier(self.action_id, "action_id")
        _count(self.action_sequence, "action_sequence")
        if not isinstance(self.action_kind, WorkItemActionKind):
            raise ValueError("action_kind must use the closed action vocabulary")
        if not isinstance(self.resource_role, ResourceRole):
            raise ValueError("resource_role must use the closed resource vocabulary")
        if self.resource_role is not _resource_role_for_action_kind(self.action_kind):
            raise ValueError("action kind and resource role do not match")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the exact action identity and its closed responsibility."""

        return {
            "schema": self.schema,
            "action_id": self.action_id,
            "action_sequence": self.action_sequence,
            "action_kind": self.action_kind.value,
            "resource_role": self.resource_role.value,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class WorkItemActionPlanV1:
    """Freeze one complete ordered action roster without inventing track defaults."""

    action_plan_id: str
    run_id: str
    work_item_id: str
    execution_semantics: ExecutionSemantics
    actions: tuple[WorkItemActionSpecV1, ...]
    binding_status: ActionPlanBindingStatus
    binding_evidence_ref: str | None
    complete_action_roster: bool = True
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = WORK_ITEM_ACTION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WORK_ITEM_ACTION_PLAN_SCHEMA:
            raise ValueError("unsupported work-item action plan schema")
        for label in ("action_plan_id", "run_id", "work_item_id"):
            _identifier(getattr(self, label), label)
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError("action plan requires closed execution semantics")
        _records(self.actions, WorkItemActionSpecV1, "actions")
        if not self.actions:
            raise ValueError("action plan requires a nonempty explicit roster")
        if tuple(action.action_sequence for action in self.actions) != tuple(
            range(len(self.actions))
        ):
            raise ValueError("action plan sequences must be the ordered range 0..n-1")
        _unique_texts(
            tuple(action.action_id for action in self.actions),
            "action IDs",
            portable=True,
        )
        scorer_positions = tuple(
            action.action_sequence
            for action in self.actions
            if action.resource_role is ResourceRole.SCORER
        )
        if len(scorer_positions) > 1:
            raise ValueError("action plan can contain at most one scorer action")
        if scorer_positions and scorer_positions != (len(self.actions) - 1,):
            raise ValueError("a scorer action must be the final planned action")
        if not isinstance(self.binding_status, ActionPlanBindingStatus):
            raise ValueError("binding_status must use the closed plan vocabulary")
        if self.binding_status is ActionPlanBindingStatus.UNBOUND:
            if self.binding_evidence_ref is not None:
                raise ValueError("unbound action plan cannot carry binding evidence")
        else:
            _text(self.binding_evidence_ref, "binding_evidence_ref")
        if self.complete_action_roster is not True:
            raise ValueError("v0.1 action plans must claim one complete ordered roster")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the roster and whether an external source binder checked it."""

        return {
            "schema": self.schema,
            "action_plan_id": self.action_plan_id,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "execution_semantics": self.execution_semantics.value,
            "actions": [action.to_dict() for action in self.actions],
            "binding_status": self.binding_status.value,
            "binding_evidence_ref": self.binding_evidence_ref,
            "complete_action_roster": self.complete_action_roster,
            "implementation_status": self.implementation_status,
        }


class RunOnlyOperationStage(StrEnum):
    """Separate pre-cutoff run costs from post-cutoff diagnostic closing."""

    RUN_INITIALIZE = "run_initialize"
    WORK_ITEM_CLOSE = "work_item_close"
    RUN_PRECLOSE = "run_preclose"
    POST_RUN_CLOSE = "post_run_close"


class RunOnlyOperationKind(StrEnum):
    """Name every required run-owned or permitted close-only operation."""

    INITIALIZE_METER = "initialize_meter"
    PUBLISH_RUN_PLAN = "publish_run_plan"
    PUBLISH_ACTION_PLAN = "publish_action_plan"
    PUBLISH_WORK_ITEM_BOUNDARY = "publish_work_item_boundary"
    MATERIALIZE_WORK_ITEM_METER_SOURCES = "materialize_work_item_meter_sources"
    PUBLISH_WORK_ITEM_USAGE = "publish_work_item_usage"
    PUBLISH_WORK_ITEM_BUDGET_DECISION = "publish_work_item_budget_decision"
    PUBLISH_RESULT_ENTRY = "publish_result_entry"
    RECONCILE_RUN_METER = "reconcile_run_meter"
    PUBLISH_RUN_BOUNDARY = "publish_run_boundary"
    MATERIALIZE_FROZEN_METER_SOURCES = "materialize_frozen_meter_sources"
    STRICT_RELOAD_AND_BIND = "strict_reload_and_bind"
    PUBLISH_RUN_USAGE = "publish_run_usage"
    PUBLISH_RUN_BUDGET_DECISION = "publish_run_budget_decision"
    PUBLISH_RESULT_INDEX = "publish_result_index"


class RunOnlyInputClosureKind(StrEnum):
    """Describe one closed semantic input set without selectors or globs."""

    EXACT_ARTIFACT = "exact_artifact"
    PRE_RUN_COLLECTOR_SNAPSHOT = "pre_run_collector_snapshot"
    WORK_ITEM_CUTOFF_SNAPSHOT = "work_item_cutoff_snapshot"
    WORK_ITEM_BOUNDARY_CLOSURE = "work_item_boundary_closure"
    WORK_ITEM_METER_SOURCE_CLOSURE = "work_item_meter_source_closure"
    STRICT_PREDECESSOR_PRE_CUTOFF_OPERATION_CLOSURE = (
        "strict_predecessor_pre_cutoff_run_operation_closure"
    )
    COMPLETE_PRE_CUTOFF_OPERATION_CLOSURE = (
        "complete_pre_cutoff_run_operation_closure"
    )
    RUN_PRECLOSE_COLLECTOR_CLOSURE = "run_preclose_collector_closure"
    RUN_FROZEN_SNAPSHOT = "run_frozen_snapshot"
    RUN_METER_SOURCE_CLOSURE = "run_meter_source_closure"


_RUN_ONLY_KIND_POLICY: dict[RunOnlyOperationKind, tuple[RunOnlyOperationStage, ResourceRole]] = {
    RunOnlyOperationKind.INITIALIZE_METER: (RunOnlyOperationStage.RUN_INITIALIZE, ResourceRole.ORCHESTRATION),
    RunOnlyOperationKind.PUBLISH_RUN_PLAN: (RunOnlyOperationStage.RUN_INITIALIZE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.PUBLISH_ACTION_PLAN: (RunOnlyOperationStage.RUN_INITIALIZE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.PUBLISH_WORK_ITEM_BOUNDARY: (RunOnlyOperationStage.WORK_ITEM_CLOSE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.MATERIALIZE_WORK_ITEM_METER_SOURCES: (RunOnlyOperationStage.WORK_ITEM_CLOSE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.PUBLISH_WORK_ITEM_USAGE: (RunOnlyOperationStage.WORK_ITEM_CLOSE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.PUBLISH_WORK_ITEM_BUDGET_DECISION: (RunOnlyOperationStage.WORK_ITEM_CLOSE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.PUBLISH_RESULT_ENTRY: (RunOnlyOperationStage.WORK_ITEM_CLOSE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.RECONCILE_RUN_METER: (RunOnlyOperationStage.RUN_PRECLOSE, ResourceRole.ORCHESTRATION),
    RunOnlyOperationKind.PUBLISH_RUN_BOUNDARY: (RunOnlyOperationStage.POST_RUN_CLOSE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.MATERIALIZE_FROZEN_METER_SOURCES: (RunOnlyOperationStage.POST_RUN_CLOSE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.STRICT_RELOAD_AND_BIND: (RunOnlyOperationStage.POST_RUN_CLOSE, ResourceRole.ORCHESTRATION),
    RunOnlyOperationKind.PUBLISH_RUN_USAGE: (RunOnlyOperationStage.POST_RUN_CLOSE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.PUBLISH_RUN_BUDGET_DECISION: (RunOnlyOperationStage.POST_RUN_CLOSE, ResourceRole.PERSISTENCE),
    RunOnlyOperationKind.PUBLISH_RESULT_INDEX: (RunOnlyOperationStage.POST_RUN_CLOSE, ResourceRole.PERSISTENCE),
}
_RUN_ONLY_ITEM_CLOSE_KINDS = (
    RunOnlyOperationKind.PUBLISH_WORK_ITEM_BOUNDARY,
    RunOnlyOperationKind.MATERIALIZE_WORK_ITEM_METER_SOURCES,
    RunOnlyOperationKind.PUBLISH_WORK_ITEM_USAGE,
    RunOnlyOperationKind.PUBLISH_WORK_ITEM_BUDGET_DECISION,
    RunOnlyOperationKind.PUBLISH_RESULT_ENTRY,
)
_RUN_ONLY_POST_RUN_KINDS = (
    RunOnlyOperationKind.PUBLISH_RUN_BOUNDARY,
    RunOnlyOperationKind.MATERIALIZE_FROZEN_METER_SOURCES,
    RunOnlyOperationKind.STRICT_RELOAD_AND_BIND,
    RunOnlyOperationKind.PUBLISH_RUN_USAGE,
    RunOnlyOperationKind.PUBLISH_RUN_BUDGET_DECISION,
    RunOnlyOperationKind.PUBLISH_RESULT_INDEX,
)

_RUN_ONLY_GLOBAL_INPUT_ROOT_REFS = {
    RunOnlyInputClosureKind.PRE_RUN_COLLECTOR_SNAPSHOT:
        "collector/pre_run_bootstrap_snapshot",
    RunOnlyInputClosureKind.STRICT_PREDECESSOR_PRE_CUTOFF_OPERATION_CLOSURE:
        "run_plan.json",
    RunOnlyInputClosureKind.COMPLETE_PRE_CUTOFF_OPERATION_CLOSURE:
        "run_plan.json",
    RunOnlyInputClosureKind.RUN_PRECLOSE_COLLECTOR_CLOSURE:
        "collector/run_preclose_snapshot",
    RunOnlyInputClosureKind.RUN_FROZEN_SNAPSHOT: "collector/run_frozen_snapshot",
    RunOnlyInputClosureKind.RUN_METER_SOURCE_CLOSURE: "run_meter_sources.json",
}
_RUN_ONLY_EXACT_GLOBAL_REFS = frozenset(
    {
        "run_plan.json",
        "run_boundary.json",
        "run_usage.json",
        "run_budget_decision.json",
        "run_close/publish_run_boundary.json",
        "run_close/materialize_frozen_meter_sources.json",
        "run_close/strict_reload_and_bind.json",
    }
)
_RUN_ONLY_EXACT_ITEM_FILENAMES = frozenset(
    {
        "work_item_usage.json",
        "work_item_budget_decision.json",
        "result_entry.json",
    }
)


def _exact_artifact_item_id(anchor_ref: str) -> str | None:
    """Return the item encoded by one admitted exact artifact reference."""

    if anchor_ref in _RUN_ONLY_EXACT_GLOBAL_REFS:
        return None
    parts = anchor_ref.split("/")
    if len(parts) == 2 and parts[0] == "action_plans" and parts[1].endswith(".json"):
        item_id = parts[1][:-5]
        _identifier(item_id, "exact action-plan item ID")
        return item_id
    if (
        len(parts) == 3
        and parts[0] == "work_items"
        and parts[2] in _RUN_ONLY_EXACT_ITEM_FILENAMES
    ):
        _identifier(parts[1], "exact work-item artifact item ID")
        return parts[1]
    raise ValueError("exact artifact anchor_ref is not an admitted canonical path")


@dataclass(frozen=True)
class RunOnlyInputRootV1:
    """Name one typed input closure; later binders must expand it exactly."""

    closure_kind: RunOnlyInputClosureKind
    anchor_ref: str
    related_work_item_id: str | None
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = RUN_ONLY_INPUT_ROOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUN_ONLY_INPUT_ROOT_SCHEMA:
            raise ValueError("unsupported run-only input-root schema")
        if type(self.closure_kind) is not RunOnlyInputClosureKind:
            raise ValueError("run-only input root requires a closed closure kind")
        _text(self.anchor_ref, "run-only input anchor_ref")
        if any(token in self.anchor_ref for token in ("*", "?", "[", "#")):
            raise ValueError("run-only input roots cannot use globs or selectors")
        item_kinds = {
            RunOnlyInputClosureKind.WORK_ITEM_CUTOFF_SNAPSHOT,
            RunOnlyInputClosureKind.WORK_ITEM_BOUNDARY_CLOSURE,
            RunOnlyInputClosureKind.WORK_ITEM_METER_SOURCE_CLOSURE,
        }
        if self.closure_kind is RunOnlyInputClosureKind.EXACT_ARTIFACT:
            expected_item = _exact_artifact_item_id(self.anchor_ref)
            if self.related_work_item_id != expected_item:
                raise ValueError("exact artifact item identity disagrees with its path")
        elif self.closure_kind in item_kinds:
            _identifier(self.related_work_item_id, "input-root related_work_item_id")
            expected_ref = {
                RunOnlyInputClosureKind.WORK_ITEM_CUTOFF_SNAPSHOT:
                    f"collector/work_items/{self.related_work_item_id}/cutoff_snapshot",
                RunOnlyInputClosureKind.WORK_ITEM_BOUNDARY_CLOSURE:
                    f"work_items/{self.related_work_item_id}/work_item_boundary.json",
                RunOnlyInputClosureKind.WORK_ITEM_METER_SOURCE_CLOSURE:
                    f"work_items/{self.related_work_item_id}/work_item_meter_sources.json",
            }[self.closure_kind]
            if self.anchor_ref != expected_ref:
                raise ValueError("work-item input root must use its canonical anchor")
        else:
            if self.related_work_item_id is not None:
                raise ValueError("global input root cannot name a work item")
            if self.anchor_ref != _RUN_ONLY_GLOBAL_INPUT_ROOT_REFS[self.closure_kind]:
                raise ValueError("global input root must use its canonical anchor")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize one typed closure anchor without a free-form selector."""

        return {
            "schema": self.schema,
            "closure_kind": self.closure_kind.value,
            "anchor_ref": self.anchor_ref,
            "related_work_item_id": self.related_work_item_id,
            "implementation_status": self.implementation_status,
        }


def _run_only_operation_id(kind: RunOnlyOperationKind, item_id: str | None) -> str:
    """Derive a stable one-component ID from a closed kind and optional item."""

    return kind.value if item_id is None else f"{kind.value}-{item_id}"


def _run_only_paths(
    kind: RunOnlyOperationKind, item_id: str | None, operation_id: str
) -> tuple[str, str | None, str | None, tuple[str, ...]]:
    """Derive all allowed paths; callers cannot choose an alternate artifact."""

    if kind in _RUN_ONLY_POST_RUN_KINDS:
        terminal = f"run_close/{kind.value}.json"
        usage = meter = None
    else:
        base = f"run_operations/{operation_id}"
        terminal, usage, meter = (f"{base}/terminal.json", f"{base}/usage.json", f"{base}/meter.json")
    item_outputs = {
        RunOnlyOperationKind.PUBLISH_WORK_ITEM_BOUNDARY: "work_item_boundary.json",
        RunOnlyOperationKind.MATERIALIZE_WORK_ITEM_METER_SOURCES: "work_item_meter_sources.json",
        RunOnlyOperationKind.PUBLISH_WORK_ITEM_USAGE: "work_item_usage.json",
        RunOnlyOperationKind.PUBLISH_WORK_ITEM_BUDGET_DECISION: "work_item_budget_decision.json",
        RunOnlyOperationKind.PUBLISH_RESULT_ENTRY: "result_entry.json",
    }
    run_outputs = {
        RunOnlyOperationKind.PUBLISH_RUN_PLAN: "run_plan.json",
        RunOnlyOperationKind.PUBLISH_RUN_BOUNDARY: "run_boundary.json",
        RunOnlyOperationKind.MATERIALIZE_FROZEN_METER_SOURCES: "run_meter_sources.json",
        RunOnlyOperationKind.PUBLISH_RUN_USAGE: "run_usage.json",
        RunOnlyOperationKind.PUBLISH_RUN_BUDGET_DECISION: "run_budget_decision.json",
        RunOnlyOperationKind.PUBLISH_RESULT_INDEX: "result_index.json",
    }
    if kind is RunOnlyOperationKind.PUBLISH_ACTION_PLAN:
        outputs = (f"action_plans/{item_id}.json",)
    elif kind in item_outputs:
        outputs = (f"work_items/{item_id}/{item_outputs[kind]}",)
    elif kind in run_outputs:
        outputs = (run_outputs[kind],)
    else:
        outputs = ()
    return terminal, usage, meter, outputs


def _input_root(
    closure_kind: RunOnlyInputClosureKind,
    anchor_ref: str,
    item_id: str | None = None,
) -> RunOnlyInputRootV1:
    """Build one canonical typed input root for the formal roster."""

    return RunOnlyInputRootV1(
        closure_kind=closure_kind,
        anchor_ref=anchor_ref,
        related_work_item_id=item_id,
    )


def _exact_input(anchor_ref: str, item_id: str | None = None) -> RunOnlyInputRootV1:
    return _input_root(RunOnlyInputClosureKind.EXACT_ARTIFACT, anchor_ref, item_id)


def _item_closure_inputs(item_id: str) -> tuple[RunOnlyInputRootV1, ...]:
    """Return the five stable per-item originals consumed after item close."""

    return (
        _input_root(
            RunOnlyInputClosureKind.WORK_ITEM_BOUNDARY_CLOSURE,
            f"work_items/{item_id}/work_item_boundary.json",
            item_id,
        ),
        _input_root(
            RunOnlyInputClosureKind.WORK_ITEM_METER_SOURCE_CLOSURE,
            f"work_items/{item_id}/work_item_meter_sources.json",
            item_id,
        ),
        _exact_input(f"work_items/{item_id}/work_item_usage.json", item_id),
        _exact_input(
            f"work_items/{item_id}/work_item_budget_decision.json", item_id
        ),
        _exact_input(f"work_items/{item_id}/result_entry.json", item_id),
    )


def _run_only_input_roots(
    kind: RunOnlyOperationKind,
    item_id: str | None,
    item_ids: tuple[str, ...],
) -> tuple[RunOnlyInputRootV1, ...]:
    """Derive semantic reads from kind and the complete ordered denominator."""

    pre_run = _input_root(
        RunOnlyInputClosureKind.PRE_RUN_COLLECTOR_SNAPSHOT,
        "collector/pre_run_bootstrap_snapshot",
    )
    run_plan = _exact_input("run_plan.json")
    if kind in (
        RunOnlyOperationKind.INITIALIZE_METER,
        RunOnlyOperationKind.PUBLISH_RUN_PLAN,
    ):
        return (pre_run,)
    if item_id is not None:
        action_plan = _exact_input(f"action_plans/{item_id}.json", item_id)
        boundary = _input_root(
            RunOnlyInputClosureKind.WORK_ITEM_BOUNDARY_CLOSURE,
            f"work_items/{item_id}/work_item_boundary.json",
            item_id,
        )
        cutoff = _input_root(
            RunOnlyInputClosureKind.WORK_ITEM_CUTOFF_SNAPSHOT,
            f"collector/work_items/{item_id}/cutoff_snapshot",
            item_id,
        )
        meter_sources = _input_root(
            RunOnlyInputClosureKind.WORK_ITEM_METER_SOURCE_CLOSURE,
            f"work_items/{item_id}/work_item_meter_sources.json",
            item_id,
        )
        usage = _exact_input(f"work_items/{item_id}/work_item_usage.json", item_id)
        budget = _exact_input(
            f"work_items/{item_id}/work_item_budget_decision.json", item_id
        )
        if kind is RunOnlyOperationKind.PUBLISH_ACTION_PLAN:
            return (run_plan,)
        if kind is RunOnlyOperationKind.PUBLISH_WORK_ITEM_BOUNDARY:
            return (run_plan, action_plan, cutoff)
        if kind is RunOnlyOperationKind.MATERIALIZE_WORK_ITEM_METER_SOURCES:
            return (boundary, cutoff)
        if kind is RunOnlyOperationKind.PUBLISH_WORK_ITEM_USAGE:
            return (action_plan, boundary, meter_sources)
        if kind is RunOnlyOperationKind.PUBLISH_WORK_ITEM_BUDGET_DECISION:
            return (run_plan, boundary, usage)
        if kind is RunOnlyOperationKind.PUBLISH_RESULT_ENTRY:
            return (run_plan, action_plan, boundary, usage, budget)
    action_plans = tuple(
        _exact_input(f"action_plans/{expected_item_id}.json", expected_item_id)
        for expected_item_id in item_ids
    )
    item_closures = tuple(
        root for expected_item_id in item_ids
        for root in _item_closure_inputs(expected_item_id)
    )
    if kind is RunOnlyOperationKind.RECONCILE_RUN_METER:
        strict_predecessor_operations = _input_root(
            RunOnlyInputClosureKind.STRICT_PREDECESSOR_PRE_CUTOFF_OPERATION_CLOSURE,
            "run_plan.json",
        )
        return (
            (run_plan,)
            + action_plans
            + item_closures
            + (
                strict_predecessor_operations,
                _input_root(
                    RunOnlyInputClosureKind.RUN_PRECLOSE_COLLECTOR_CLOSURE,
                    "collector/run_preclose_snapshot",
                ),
            )
        )
    run_boundary = _exact_input("run_boundary.json")
    frozen_snapshot = _input_root(
        RunOnlyInputClosureKind.RUN_FROZEN_SNAPSHOT,
        "collector/run_frozen_snapshot",
    )
    run_sources = _input_root(
        RunOnlyInputClosureKind.RUN_METER_SOURCE_CLOSURE,
        "run_meter_sources.json",
    )
    strict_terminal = _exact_input("run_close/strict_reload_and_bind.json")
    if kind is RunOnlyOperationKind.PUBLISH_RUN_BOUNDARY:
        return (frozen_snapshot,)
    if kind is RunOnlyOperationKind.MATERIALIZE_FROZEN_METER_SOURCES:
        return (run_boundary, frozen_snapshot)
    if kind is RunOnlyOperationKind.STRICT_RELOAD_AND_BIND:
        complete_pre_cutoff_operations = _input_root(
            RunOnlyInputClosureKind.COMPLETE_PRE_CUTOFF_OPERATION_CLOSURE,
            "run_plan.json",
        )
        return (
            (run_plan,)
            + action_plans
            + item_closures
            + (
                complete_pre_cutoff_operations,
                run_boundary,
                run_sources,
                _exact_input("run_close/publish_run_boundary.json"),
                _exact_input("run_close/materialize_frozen_meter_sources.json"),
            )
        )
    if kind is RunOnlyOperationKind.PUBLISH_RUN_USAGE:
        return (run_plan, run_boundary, run_sources, strict_terminal)
    if kind is RunOnlyOperationKind.PUBLISH_RUN_BUDGET_DECISION:
        return (run_plan, _exact_input("run_usage.json"), strict_terminal)
    if kind is RunOnlyOperationKind.PUBLISH_RESULT_INDEX:
        result_entries = tuple(
            _exact_input(
                f"work_items/{expected_item_id}/result_entry.json",
                expected_item_id,
            )
            for expected_item_id in item_ids
        )
        return (
            (run_plan,)
            + result_entries
            + (
                run_boundary,
                run_sources,
                strict_terminal,
                _exact_input("run_usage.json"),
                _exact_input("run_budget_decision.json"),
            )
        )
    raise ValueError("run-only kind has no canonical input-root policy")


@dataclass(frozen=True)
class RunOnlyOperationSpecV1:
    """Freeze order and semantic reads; actual dispatch needs separate evidence."""

    operation_id: str
    stage: RunOnlyOperationStage
    kind: RunOnlyOperationKind
    resource_role: ResourceRole
    related_work_item_id: str | None
    depends_on_operation_ids: tuple[str, ...]
    input_roots: tuple[RunOnlyInputRootV1, ...]
    attempt_limit: int
    requirement: str
    terminal_ref: str
    usage_ref: str | None
    meter_ref: str | None
    output_refs: tuple[str, ...]
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = RUN_ONLY_OPERATION_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUN_ONLY_OPERATION_SPEC_SCHEMA:
            raise ValueError("unsupported run-only operation spec schema")
        if type(self.kind) is not RunOnlyOperationKind or type(self.stage) is not RunOnlyOperationStage:
            raise ValueError("run-only kind and stage require closed enums")
        if type(self.resource_role) is not ResourceRole or (self.stage, self.resource_role) != _RUN_ONLY_KIND_POLICY[self.kind]:
            raise ValueError("run-only kind, stage, and role disagree")
        if self.kind in _RUN_ONLY_ITEM_CLOSE_KINDS or self.kind is RunOnlyOperationKind.PUBLISH_ACTION_PLAN:
            _identifier(self.related_work_item_id, "related_work_item_id")
        elif self.related_work_item_id is not None:
            raise ValueError("global run-only operation cannot name a work item")
        _identifier(self.operation_id, "operation_id")
        if self.operation_id != _run_only_operation_id(self.kind, self.related_work_item_id):
            raise ValueError("run-only operation_id must be canonical")
        _unique_texts(self.depends_on_operation_ids, "operation dependencies", portable=True)
        _records(self.input_roots, RunOnlyInputRootV1, "run-only input roots")
        if not self.input_roots:
            raise ValueError("run-only operation requires typed input roots")
        operation_closures = tuple(
            root.closure_kind
            for root in self.input_roots
            if root.closure_kind in {
                RunOnlyInputClosureKind.STRICT_PREDECESSOR_PRE_CUTOFF_OPERATION_CLOSURE,
                RunOnlyInputClosureKind.COMPLETE_PRE_CUTOFF_OPERATION_CLOSURE,
            }
        )
        expected_operation_closure = {
            RunOnlyOperationKind.RECONCILE_RUN_METER:
                RunOnlyInputClosureKind.STRICT_PREDECESSOR_PRE_CUTOFF_OPERATION_CLOSURE,
            RunOnlyOperationKind.STRICT_RELOAD_AND_BIND:
                RunOnlyInputClosureKind.COMPLETE_PRE_CUTOFF_OPERATION_CLOSURE,
        }.get(self.kind)
        if expected_operation_closure is None:
            if operation_closures:
                raise ValueError(
                    "only reconcile or strict reload may consume operation closures"
                )
        elif operation_closures != (expected_operation_closure,):
            raise ValueError(
                "run-only operation requires its exact pre-cutoff operation closure"
            )
        if type(self.attempt_limit) is not int or self.attempt_limit != 1:
            raise ValueError("run-only attempt_limit must equal one")
        if type(self.requirement) is not str or self.requirement != "required":
            raise ValueError("run-only operation requirement must be required")
        paths = _run_only_paths(self.kind, self.related_work_item_id, self.operation_id)
        if (self.terminal_ref, self.usage_ref, self.meter_ref, self.output_refs) != paths:
            raise ValueError("run-only operation refs must equal canonical paths")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the exact planned owner, dependency, and path declaration."""

        return {
            "schema": self.schema, "operation_id": self.operation_id,
            "stage": self.stage.value, "kind": self.kind.value,
            "resource_role": self.resource_role.value,
            "related_work_item_id": self.related_work_item_id,
            "depends_on_operation_ids": list(self.depends_on_operation_ids),
            "input_roots": [root.to_dict() for root in self.input_roots],
            "attempt_limit": self.attempt_limit, "requirement": self.requirement,
            "terminal_ref": self.terminal_ref, "usage_ref": self.usage_ref,
            "meter_ref": self.meter_ref, "output_refs": list(self.output_refs),
            "implementation_status": self.implementation_status,
        }


def canonical_run_only_operations_v1(item_ids: tuple[str, ...]) -> tuple[RunOnlyOperationSpecV1, ...]:
    """Construct the formal roster and dependencies from the ordered denominator."""

    _unique_texts(item_ids, "run-only expected work item IDs", portable=True)
    if not item_ids:
        raise ValueError("run-only roster requires expected work items")
    operations: list[RunOnlyOperationSpecV1] = []

    def add(kind: RunOnlyOperationKind, item_id: str | None, dependencies: tuple[str, ...]) -> str:
        operation_id = _run_only_operation_id(kind, item_id)
        stage, role = _RUN_ONLY_KIND_POLICY[kind]
        terminal, usage, meter, outputs = _run_only_paths(kind, item_id, operation_id)
        operations.append(RunOnlyOperationSpecV1(
            operation_id=operation_id, stage=stage, kind=kind, resource_role=role,
            related_work_item_id=item_id, depends_on_operation_ids=dependencies,
            input_roots=_run_only_input_roots(kind, item_id, item_ids),
            attempt_limit=1, requirement="required", terminal_ref=terminal,
            usage_ref=usage, meter_ref=meter, output_refs=outputs,
        ))
        return operation_id

    first = add(RunOnlyOperationKind.INITIALIZE_METER, None, ())
    plan = add(RunOnlyOperationKind.PUBLISH_RUN_PLAN, None, (first,))
    action_ids = {
        item_id: add(RunOnlyOperationKind.PUBLISH_ACTION_PLAN, item_id, (plan,))
        for item_id in item_ids
    }
    result_ids = []
    for item_id in item_ids:
        previous = action_ids[item_id]
        for kind in _RUN_ONLY_ITEM_CLOSE_KINDS:
            previous = add(kind, item_id, (previous,))
        result_ids.append(previous)
    previous = add(RunOnlyOperationKind.RECONCILE_RUN_METER, None, tuple(result_ids))
    for kind in _RUN_ONLY_POST_RUN_KINDS:
        previous = add(kind, None, (previous,))
    return tuple(operations)


@dataclass(frozen=True)
class AllowedClosePlanV1:
    """Record the closed allowed suffix without claiming any close occurred."""

    post_work_item_operation_ids: tuple[str, ...]
    post_run_operation_ids: tuple[str, ...]
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = ALLOWED_CLOSE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ALLOWED_CLOSE_PLAN_SCHEMA:
            raise ValueError("unsupported allowed close plan schema")
        _unique_texts(self.post_work_item_operation_ids, "post-work-item operation IDs", portable=True)
        _unique_texts(self.post_run_operation_ids, "post-run operation IDs", portable=True)
        if not self.post_work_item_operation_ids or not self.post_run_operation_ids:
            raise ValueError("allowed close plan requires both close rosters")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize only the allowed roster, never observed close evidence."""

        return {
            "schema": self.schema,
            "post_work_item_operation_ids": list(self.post_work_item_operation_ids),
            "post_run_operation_ids": list(self.post_run_operation_ids),
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class RunOnlyOperationPlanV1:
    """Bind the complete required operation roster to one run and contract set."""

    plan_id: str
    run_id: str
    run_plan_schema_id: str
    artifact_topology_id: str
    metering_contract_id: str
    work_item_metering_boundary_id: str
    run_metering_boundary_id: str
    run_only_usage_receipt_schema_id: str
    work_item_boundary_schema_id: str
    run_boundary_schema_id: str
    expected_work_item_ids: tuple[str, ...]
    required_boundary_refs: tuple[str, ...]
    operations: tuple[RunOnlyOperationSpecV1, ...]
    allowed_close_plan: AllowedClosePlanV1
    complete_operation_roster: bool
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = RUN_ONLY_OPERATION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUN_ONLY_OPERATION_PLAN_SCHEMA:
            raise ValueError("unsupported run-only operation plan schema")
        _identifier(self.plan_id, "run-only plan_id")
        _identifier(self.run_id, "run-only run_id")
        for label, expected in (
            ("run_plan_schema_id", TRACEFIX_RUN_PLAN_SCHEMA),
            ("artifact_topology_id", ARTIFACT_TOPOLOGY_ID_V0_1),
            ("metering_contract_id", METERING_CONTRACT_ID_V0_1),
            ("work_item_metering_boundary_id", WORK_ITEM_METERING_BOUNDARY_ID_V0_1),
            ("run_metering_boundary_id", RUN_METERING_BOUNDARY_ID_V0_1),
            ("run_only_usage_receipt_schema_id", RUN_ONLY_USAGE_RECEIPT_SCHEMA),
            ("work_item_boundary_schema_id", WORK_ITEM_BOUNDARY_SCHEMA_V2),
            ("run_boundary_schema_id", RUN_BOUNDARY_SCHEMA_V2),
        ):
            if type(getattr(self, label)) is not str or getattr(self, label) != expected:
                raise ValueError(f"run-only {label} must equal the frozen contract")
        _unique_texts(self.expected_work_item_ids, "run-only expected work item IDs", portable=True)
        if not self.expected_work_item_ids:
            raise ValueError("run-only plan requires expected work items")
        expected_refs = tuple(
            f"work_items/{item_id}/work_item_boundary.json"
            for item_id in self.expected_work_item_ids
        ) + ("run_boundary.json",)
        if self.required_boundary_refs != expected_refs:
            raise ValueError("run-only boundary refs must follow expected work-item order")
        _records(self.operations, RunOnlyOperationSpecV1, "run-only operations")
        operation_ids = tuple(op.operation_id for op in self.operations)
        _unique_texts(operation_ids, "run-only operation IDs", portable=True)
        seen: set[str] = set()
        for operation in self.operations:
            if any(dependency not in seen for dependency in operation.depends_on_operation_ids):
                raise ValueError("run-only dependencies must name earlier operations")
            seen.add(operation.operation_id)
        if self.operations != canonical_run_only_operations_v1(self.expected_work_item_ids):
            raise ValueError("run-only operation roster must equal the canonical formal roster")
        if type(self.allowed_close_plan) is not AllowedClosePlanV1:
            raise ValueError("allowed_close_plan must be typed")
        item_close = tuple(
            op.operation_id for op in self.operations
            if op.stage is RunOnlyOperationStage.WORK_ITEM_CLOSE
        )
        run_close = tuple(
            op.operation_id for op in self.operations
            if op.stage is RunOnlyOperationStage.POST_RUN_CLOSE
        )
        if (self.allowed_close_plan.post_work_item_operation_ids,
            self.allowed_close_plan.post_run_operation_ids) != (item_close, run_close):
            raise ValueError("allowed close roster must match the canonical operation suffix")
        if type(self.complete_operation_roster) is not bool or not self.complete_operation_roster:
            raise ValueError("run-only operation roster must declare completeness")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize all identities and the complete structural roster."""

        return {
            "schema": self.schema, "plan_id": self.plan_id, "run_id": self.run_id,
            "run_plan_schema_id": self.run_plan_schema_id,
            "artifact_topology_id": self.artifact_topology_id,
            "metering_contract_id": self.metering_contract_id,
            "work_item_metering_boundary_id": self.work_item_metering_boundary_id,
            "run_metering_boundary_id": self.run_metering_boundary_id,
            "run_only_usage_receipt_schema_id": self.run_only_usage_receipt_schema_id,
            "work_item_boundary_schema_id": self.work_item_boundary_schema_id,
            "run_boundary_schema_id": self.run_boundary_schema_id,
            "expected_work_item_ids": list(self.expected_work_item_ids),
            "required_boundary_refs": list(self.required_boundary_refs),
            "operations": [op.to_dict() for op in self.operations],
            "allowed_close_plan": self.allowed_close_plan.to_dict(),
            "complete_operation_roster": self.complete_operation_roster,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class TraceFixRunPlanV1:
    """Declare one frozen topology and metering policy with the complete roster."""

    plan_id: str
    run_id: str
    suite_id: str
    suite_version: str
    population_id: str
    release_record_ref: str
    scorer_registry_id: str
    artifact_topology_id: str
    metering_contract_id: str
    work_item_metering_boundary_id: str
    run_metering_boundary_id: str
    method_config_refs: tuple[str, ...]
    treatment_arms: tuple[TreatmentArmV1, ...]
    track_run_matrices: tuple[TrackRunMatrixV1, ...]
    source_task_roster: tuple[str, ...]
    expected_work_item_count: int
    expected_work_items: tuple[ExpectedWorkItemV1, ...]
    work_item_action_plans: tuple[WorkItemActionPlanV1, ...]
    run_only_operation_plan: RunOnlyOperationPlanV1
    run_budget: RunBudgetV1 | None = None
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = TRACEFIX_RUN_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACEFIX_RUN_PLAN_SCHEMA:
            raise ValueError("unsupported run plan schema")
        for label in ("plan_id", "run_id", "population_id", "scorer_registry_id"):
            _identifier(getattr(self, label), label)
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID:
            raise ValueError("run plan suite_id must be LongHorizon-TraceFix-Bench")
        if self.suite_version != SCORER_REGISTRY_SUITE_VERSION_V0_1:
            raise ValueError("run plan suite_version must equal the v0.1 freeze")
        # Slash-bearing contract IDs are exact version declarations, not path components.
        for label, expected in (
            ("artifact_topology_id", ARTIFACT_TOPOLOGY_ID_V0_1),
            ("metering_contract_id", METERING_CONTRACT_ID_V0_1),
            ("work_item_metering_boundary_id", WORK_ITEM_METERING_BOUNDARY_ID_V0_1),
            ("run_metering_boundary_id", RUN_METERING_BOUNDARY_ID_V0_1),
        ):
            value = getattr(self, label)
            if type(value) is not str or value != expected:
                raise ValueError(f"run plan {label} must equal the v0.1 freeze")
        _text(self.release_record_ref, "release_record_ref")
        _unique_texts(self.method_config_refs, "method_config_refs", portable=False)
        _unique_texts(self.source_task_roster, "source_task_roster", portable=False)
        if not self.method_config_refs or not self.source_task_roster:
            raise ValueError("run plan requires method config refs and a source-task roster")
        _records(self.treatment_arms, TreatmentArmV1, "treatment_arms")
        _records(self.track_run_matrices, TrackRunMatrixV1, "track_run_matrices")
        if not self.treatment_arms or not self.track_run_matrices:
            raise ValueError("run plan requires treatment arms and track matrices")
        arm_ids = tuple(arm.treatment_id for arm in self.treatment_arms)
        matrix_ids = tuple(matrix.matrix_id for matrix in self.track_run_matrices)
        _unique_texts(arm_ids, "treatment arm IDs", portable=True)
        _unique_texts(matrix_ids, "track matrix IDs", portable=True)
        cohort_keys = tuple(
            (
                matrix.track.value,
                matrix.benchmark_id.casefold(),
                matrix.benchmark_version.casefold(),
                matrix.adapter_version.casefold(),
                matrix.split.value,
                matrix.execution_semantics.value,
            )
            for matrix in self.track_run_matrices
        )
        if len(set(cohort_keys)) != len(cohort_keys):
            raise ValueError("run plan cannot repeat a track cohort matrix")
        declared_configs = set(self.method_config_refs)
        referenced_configs = {
            value
            for arm in self.treatment_arms
            for value in arm.method_config_refs
        }
        if declared_configs != referenced_configs:
            raise ValueError("method config refs must close exactly over treatment arms")
        arms_by_id = {arm.treatment_id.casefold(): arm for arm in self.treatment_arms}
        matrix_treatment_ids = {
            cell.treatment_id.casefold()
            for matrix in self.track_run_matrices
            for cell in matrix.registered_cells
        }
        if matrix_treatment_ids != set(arms_by_id):
            raise ValueError("matrix treatment IDs must close exactly over treatment arms")
        if any(
            cell.treatment_id
            != arms_by_id[cell.treatment_id.casefold()].treatment_id
            for matrix in self.track_run_matrices
            for cell in matrix.registered_cells
        ):
            raise ValueError("matrix treatment spelling must match its registered arm")
        _count(self.expected_work_item_count, "expected_work_item_count")
        _records(self.expected_work_items, ExpectedWorkItemV1, "expected_work_items")
        if not self.expected_work_items or self.expected_work_item_count != len(self.expected_work_items):
            raise ValueError("run plan count must equal its nonempty work-item roster")
        _records(
            self.work_item_action_plans,
            WorkItemActionPlanV1,
            "work_item_action_plans",
        )
        if len(self.work_item_action_plans) != self.expected_work_item_count:
            raise ValueError("run plan requires one action plan per expected work item")
        if self.run_budget is not None and type(self.run_budget) is not RunBudgetV1:
            raise ValueError("run_budget must be a RunBudgetV1 or null")
        item_ids = tuple(item.work_item_id for item in self.expected_work_items)
        _unique_texts(item_ids, "work_item_ids", portable=True)
        action_plan_item_ids = tuple(
            action_plan.work_item_id for action_plan in self.work_item_action_plans
        )
        _unique_texts(
            tuple(
                action_plan.action_plan_id
                for action_plan in self.work_item_action_plans
            ),
            "action plan IDs",
            portable=True,
        )
        if action_plan_item_ids != item_ids:
            raise ValueError(
                "action plans must follow the exact expected work-item order"
            )
        if type(self.run_only_operation_plan) is not RunOnlyOperationPlanV1 or (
            self.run_only_operation_plan.plan_id != self.plan_id
            or self.run_only_operation_plan.run_id != self.run_id
            or self.run_only_operation_plan.run_plan_schema_id != self.schema
            or self.run_only_operation_plan.artifact_topology_id != self.artifact_topology_id
            or self.run_only_operation_plan.metering_contract_id != self.metering_contract_id
            or self.run_only_operation_plan.work_item_metering_boundary_id != self.work_item_metering_boundary_id
            or self.run_only_operation_plan.run_metering_boundary_id != self.run_metering_boundary_id
            or self.run_only_operation_plan.expected_work_item_ids != item_ids
        ):
            raise ValueError("run-only plan must bind the run, contract identities, and ordered work items")
        if any(
            action_plan.run_id != self.run_id
            or action_plan.execution_semantics is not item.execution_semantics
            for item, action_plan in zip(
                self.expected_work_items,
                self.work_item_action_plans,
                strict=True,
            )
        ):
            raise ValueError(
                "action plans must bind each work item's run and execution semantics"
            )
        composite = tuple(tuple(part.casefold() for part in item.identity()) for item in self.expected_work_items)
        if len(set(composite)) != len(composite):
            raise ValueError("run plan cannot repeat a complete work-item identity")
        source_tasks = {task.casefold(): task for task in self.source_task_roster}
        if {item.source_task_id.casefold() for item in self.expected_work_items} != set(source_tasks):
            raise ValueError("work items must cover exactly the declared source-task roster")
        case_identity: dict[str, tuple[str, ...]] = {}
        for item in self.expected_work_items:
            if item.run_id != self.run_id or item.population_id != self.population_id:
                raise ValueError("work item does not bind this run and population")
            if item.source_task_id != source_tasks[item.source_task_id.casefold()]:
                raise ValueError("work item source-task spelling must match the roster")
            arm = arms_by_id.get(item.treatment_id.casefold())
            if arm is None or (
                item.treatment_id != arm.treatment_id
                or item.locator_treatment_id != arm.locator_treatment_id
                or item.fixer_treatment_id != arm.fixer_treatment_id
            ):
                raise ValueError("work item treatment does not match its registered arm")
            if item.track is TraceFixTrack.STATIC_ATTRIBUTION:
                if arm.fixer_treatment_id != STATIC_NO_FIXER_TREATMENT_ID:
                    raise ValueError("static treatment arm must use the no-fixer identity")
            elif arm.fixer_treatment_id.casefold() == STATIC_NO_FIXER_TREATMENT_ID:
                raise ValueError("non-static treatment arm requires a fixer identity")
            # A case may repeat across treatments, but its source and environment cannot drift.
            identity = (
                item.case_id, item.source_task_id, item.track.value, item.split.value,
                item.execution_semantics.value, item.benchmark_id,
                item.benchmark_version, item.adapter_version,
            )
            previous = case_identity.setdefault(item.case_id.casefold(), identity)
            if previous != identity:
                raise ValueError("one case cannot change source or benchmark identity across cells")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize release, policy, source roster, and every planned cell."""

        return {
            "schema": self.schema,
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "release_record_ref": self.release_record_ref,
            "scorer_registry_id": self.scorer_registry_id,
            "artifact_topology_id": self.artifact_topology_id,
            "metering_contract_id": self.metering_contract_id,
            "work_item_metering_boundary_id": self.work_item_metering_boundary_id,
            "run_metering_boundary_id": self.run_metering_boundary_id,
            "method_config_refs": list(self.method_config_refs),
            "treatment_arms": [arm.to_dict() for arm in self.treatment_arms],
            "track_run_matrices": [
                matrix.to_dict() for matrix in self.track_run_matrices
            ],
            "source_task_roster": list(self.source_task_roster),
            "expected_work_item_count": self.expected_work_item_count,
            "expected_work_items": [item.to_dict() for item in self.expected_work_items],
            "work_item_action_plans": [
                action_plan.to_dict()
                for action_plan in self.work_item_action_plans
            ],
            "run_only_operation_plan": self.run_only_operation_plan.to_dict(),
            "run_budget": None if self.run_budget is None else self.run_budget.to_dict(),
            "implementation_status": self.implementation_status,
        }


class LedgerEventPhase(StrEnum):
    """Close the append-only action lifecycle without performing recovery."""

    PREPARED = "prepared"
    IN_FLIGHT = "in_flight"
    COMPLETED_RAW = "completed_raw"
    FAILED_RAW = "failed_raw"
    SCORER_DECISION = "scorer_decision"


class RunActionFailureKind(StrEnum):
    """Classify method and evaluation-apparatus failures with a closed vocabulary."""

    METHOD_BUDGET_EXHAUSTED = "method_budget_exhausted"
    METHOD_OUTPUT_INVALID = "method_output_invalid"
    RESPONSE_TRUNCATED = "response_truncated"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_FAILURE = "provider_failure"
    WORKER_EXECUTION_FAILED = "worker_execution_failed"
    ADAPTER_INPUT_INVALID = "adapter_input_invalid"
    ENVIRONMENT_ACTION_FAILED = "environment_action_failed"
    RESTORE_FAILED = "restore_failed"
    OFFICIAL_VERIFIER_FAILED = "official_verifier_failed"
    SCORER_FAILED = "scorer_failed"
    LEDGER_PERSISTENCE_FAILED = "ledger_persistence_failed"
    RESUME_ARTIFACT_MISMATCH = "resume_artifact_mismatch"


_FAILURE_KINDS_BY_DOMAIN = {
    TraceFailureDomain.METHOD: frozenset(
        {
            RunActionFailureKind.METHOD_BUDGET_EXHAUSTED,
            RunActionFailureKind.METHOD_OUTPUT_INVALID,
            RunActionFailureKind.RESPONSE_TRUNCATED,
        }
    ),
    TraceFailureDomain.INFRASTRUCTURE: frozenset(
        {
            RunActionFailureKind.PROVIDER_RATE_LIMIT,
            RunActionFailureKind.PROVIDER_TIMEOUT,
            RunActionFailureKind.PROVIDER_FAILURE,
            RunActionFailureKind.WORKER_EXECUTION_FAILED,
            RunActionFailureKind.ADAPTER_INPUT_INVALID,
            RunActionFailureKind.ENVIRONMENT_ACTION_FAILED,
            RunActionFailureKind.RESTORE_FAILED,
            RunActionFailureKind.OFFICIAL_VERIFIER_FAILED,
            RunActionFailureKind.SCORER_FAILED,
            RunActionFailureKind.LEDGER_PERSISTENCE_FAILED,
            RunActionFailureKind.RESUME_ARTIFACT_MISMATCH,
        }
    ),
}

_METHOD_EXECUTION_ROLES = frozenset(
    {
        ResourceRole.LOCATOR,
        ResourceRole.FIXER,
        ResourceRole.ACTOR,
        ResourceRole.PROBE,
        ResourceRole.CHECK,
        ResourceRole.REPAIR,
        ResourceRole.REPLAY,
    }
)
_FAILURE_ROLES_BY_KIND = {
    RunActionFailureKind.METHOD_BUDGET_EXHAUSTED: _METHOD_EXECUTION_ROLES,
    RunActionFailureKind.METHOD_OUTPUT_INVALID: _METHOD_EXECUTION_ROLES,
    RunActionFailureKind.RESPONSE_TRUNCATED: _METHOD_EXECUTION_ROLES,
    RunActionFailureKind.PROVIDER_RATE_LIMIT: _METHOD_EXECUTION_ROLES,
    RunActionFailureKind.PROVIDER_TIMEOUT: _METHOD_EXECUTION_ROLES,
    RunActionFailureKind.PROVIDER_FAILURE: _METHOD_EXECUTION_ROLES,
    RunActionFailureKind.WORKER_EXECUTION_FAILED: frozenset(
        {
            ResourceRole.LOCATOR,
            ResourceRole.FIXER,
            ResourceRole.ACTOR,
            ResourceRole.ENVIRONMENT,
            ResourceRole.RESTORE,
            ResourceRole.PROBE,
            ResourceRole.CHECK,
            ResourceRole.REPAIR,
            ResourceRole.REPLAY,
            ResourceRole.OFFICIAL_VERIFIER,
            ResourceRole.SCORER,
        }
    ),
    RunActionFailureKind.ADAPTER_INPUT_INVALID: frozenset(
        {
            ResourceRole.LOCATOR,
            ResourceRole.FIXER,
            ResourceRole.ACTOR,
            ResourceRole.ENVIRONMENT,
            ResourceRole.RESTORE,
            ResourceRole.PROBE,
            ResourceRole.CHECK,
            ResourceRole.REPAIR,
            ResourceRole.REPLAY,
            ResourceRole.OFFICIAL_VERIFIER,
            ResourceRole.SCORER,
        }
    ),
    RunActionFailureKind.ENVIRONMENT_ACTION_FAILED: frozenset(
        {ResourceRole.ENVIRONMENT}
    ),
    RunActionFailureKind.RESTORE_FAILED: frozenset(
        {ResourceRole.RESTORE, ResourceRole.RECOVERY}
    ),
    RunActionFailureKind.OFFICIAL_VERIFIER_FAILED: frozenset(
        {ResourceRole.OFFICIAL_VERIFIER}
    ),
    RunActionFailureKind.SCORER_FAILED: frozenset({ResourceRole.SCORER}),
    RunActionFailureKind.LEDGER_PERSISTENCE_FAILED: frozenset(
        {ResourceRole.PERSISTENCE}
    ),
    RunActionFailureKind.RESUME_ARTIFACT_MISMATCH: frozenset(
        {ResourceRole.RECOVERY, ResourceRole.ORCHESTRATION}
    ),
}


def _validate_action_failure(
    action_kind: WorkItemActionKind,
    failure_domain: TraceFailureDomain | None,
    failure_kind: RunActionFailureKind | None,
) -> None:
    """Apply the same failure domain and role contract to events and summaries."""

    resource_role = _resource_role_for_action_kind(action_kind)
    if type(failure_domain) is not TraceFailureDomain:
        raise ValueError("failed raw action requires a closed failure domain")
    if (
        type(failure_kind) is not RunActionFailureKind
        or failure_kind not in _FAILURE_KINDS_BY_DOMAIN[failure_domain]
    ):
        raise ValueError("failed raw action has an invalid failure kind")
    if resource_role not in _FAILURE_ROLES_BY_KIND[failure_kind]:
        raise ValueError("failed raw action kind is invalid for its resource role")


@dataclass(frozen=True)
class ResourceBalanceV1:
    """Record a pre-action remaining budget value or explicit unavailability."""

    role: ResourceRole
    scope: BudgetScope
    dimension: ResourceDimension
    status: UsageValueStatus
    remaining_value: int | float | None
    schema: str = RESOURCE_BALANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESOURCE_BALANCE_SCHEMA:
            raise ValueError("unsupported resource balance schema")
        if not isinstance(self.role, ResourceRole) or not isinstance(self.scope, BudgetScope):
            raise ValueError("resource balance role and scope must use closed enums")
        if not isinstance(self.dimension, ResourceDimension) or not isinstance(self.status, UsageValueStatus):
            raise ValueError("resource balance dimension and status must use closed enums")
        if self.status is UsageValueStatus.UNAVAILABLE:
            if self.remaining_value is not None:
                raise ValueError("unavailable resource balance must have null value")
        else:
            _finite(self.remaining_value, "remaining_value", dimension=self.dimension)

    def to_dict(self) -> dict[str, object]:
        """Preserve budget scope, role, value, and availability together."""

        return {
            "schema": self.schema,
            "role": self.role.value,
            "scope": self.scope.value,
            "dimension": self.dimension.value,
            "status": self.status.value,
            "remaining_value": self.remaining_value,
        }


@dataclass(frozen=True)
class RunLedgerEventV1:
    """Describe one phase; runner preflight must close the full event chain."""

    event_id: str
    run_id: str
    work_item_id: str
    action_id: str
    action_kind: WorkItemActionKind
    event_sequence: int
    predecessor_event_id: str | None
    phase: LedgerEventPhase
    resource_role: ResourceRole
    input_ref: str | None
    budget_before: tuple[ResourceBalanceV1, ...]
    raw_receipt_ref: str | None
    failure_domain: TraceFailureDomain | None
    failure_kind: RunActionFailureKind | None
    scorer_decision_ref: str | None
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = RUN_LEDGER_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUN_LEDGER_EVENT_SCHEMA:
            raise ValueError("unsupported run ledger event schema")
        for label in ("event_id", "run_id", "work_item_id", "action_id"):
            _identifier(getattr(self, label), label)
        _count(self.event_sequence, "event_sequence")
        if self.predecessor_event_id is not None:
            _identifier(self.predecessor_event_id, "predecessor_event_id")
            if self.predecessor_event_id.casefold() == self.event_id.casefold():
                raise ValueError("ledger event cannot precede itself")
        if (
            not isinstance(self.action_kind, WorkItemActionKind)
            or not isinstance(self.phase, LedgerEventPhase)
            or not isinstance(self.resource_role, ResourceRole)
        ):
            raise ValueError(
                "ledger action kind, phase, and resource role must use closed enums"
            )
        if self.resource_role is not _resource_role_for_action_kind(self.action_kind):
            raise ValueError("ledger action kind and resource role do not match")
        _records(self.budget_before, ResourceBalanceV1, "budget_before")
        balance_keys = tuple((value.role, value.scope, value.dimension) for value in self.budget_before)
        if len(set(balance_keys)) != len(balance_keys):
            raise ValueError("budget_before cannot repeat a role, scope, and dimension")
        if self.phase is LedgerEventPhase.PREPARED:
            _text(self.input_ref, "prepared input_ref")
            if not self.budget_before:
                raise ValueError("prepared action requires explicit budget-before values")
            if any(balance.role is not self.resource_role for balance in self.budget_before):
                raise ValueError("prepared budget role must match action resource role")
        else:
            if self.predecessor_event_id is None:
                raise ValueError("non-prepared phase requires a predecessor event")
            if self.input_ref is not None or self.budget_before:
                raise ValueError("only prepared phase may carry input and budget-before values")
        if self.phase in (LedgerEventPhase.COMPLETED_RAW, LedgerEventPhase.FAILED_RAW):
            _text(self.raw_receipt_ref, "raw_receipt_ref")
        elif self.raw_receipt_ref is not None:
            raise ValueError("only raw terminal phases may carry a raw receipt")
        if self.phase is LedgerEventPhase.FAILED_RAW:
            _validate_action_failure(
                self.action_kind, self.failure_domain, self.failure_kind
            )
        elif self.failure_domain is not None or self.failure_kind is not None:
            raise ValueError("only failed raw action may carry failure classification")
        if self.phase is LedgerEventPhase.SCORER_DECISION:
            if self.resource_role is not ResourceRole.SCORER:
                raise ValueError("scorer decision must use the scorer resource role")
            _text(self.scorer_decision_ref, "scorer_decision_ref")
        elif self.scorer_decision_ref is not None:
            raise ValueError("only scorer decision may carry a decision reference")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize every lifecycle field, including explicit nulls."""

        return {
            "schema": self.schema,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "work_item_id": self.work_item_id,
            "action_id": self.action_id,
            "action_kind": self.action_kind.value,
            "event_sequence": self.event_sequence,
            "predecessor_event_id": self.predecessor_event_id,
            "phase": self.phase.value,
            "resource_role": self.resource_role.value,
            "input_ref": self.input_ref,
            "budget_before": [balance.to_dict() for balance in self.budget_before],
            "raw_receipt_ref": self.raw_receipt_ref,
            "failure_domain": None if self.failure_domain is None else self.failure_domain.value,
            "failure_kind": (
                None if self.failure_kind is None else self.failure_kind.value
            ),
            "scorer_decision_ref": self.scorer_decision_ref,
            "implementation_status": self.implementation_status,
        }


class WorkItemResultStatus(StrEnum):
    """Retain missing and invalid planned cells in the result denominator."""

    COMPLETED = "completed"
    METHOD_FAILED = "method_failed"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    AMBIGUOUS_RESUME = "ambiguous_resume"
    NOT_STARTED = "not_started"


class WorkItemResultReason(StrEnum):
    """Bind each result status to one auditable lifecycle reason."""

    COMPLETED_WITH_DECISION = "completed_with_decision"
    METHOD_TERMINAL_FAILURE = "method_terminal_failure"
    INFRASTRUCTURE_TERMINAL_FAILURE = "infrastructure_terminal_failure"
    SCORER_INPUT_INVALID = "scorer_input_invalid"
    ORPHAN_IN_FLIGHT = "orphan_in_flight"
    PLANNED_NOT_STARTED = "planned_not_started"


_RESULT_REASONS_BY_STATUS = {
    WorkItemResultStatus.COMPLETED: frozenset(
        {WorkItemResultReason.COMPLETED_WITH_DECISION}
    ),
    WorkItemResultStatus.METHOD_FAILED: frozenset(
        {WorkItemResultReason.METHOD_TERMINAL_FAILURE}
    ),
    WorkItemResultStatus.INFRASTRUCTURE_INVALID: frozenset(
        {
            WorkItemResultReason.INFRASTRUCTURE_TERMINAL_FAILURE,
            WorkItemResultReason.SCORER_INPUT_INVALID,
        }
    ),
    WorkItemResultStatus.AMBIGUOUS_RESUME: frozenset(
        {WorkItemResultReason.ORPHAN_IN_FLIGHT}
    ),
    WorkItemResultStatus.NOT_STARTED: frozenset(
        {WorkItemResultReason.PLANNED_NOT_STARTED}
    ),
}


@dataclass(frozen=True)
class ResourceCostV1:
    """Keep one scoped usage value and its source or unavailability evidence."""

    role: ResourceRole
    scope: BudgetScope
    dimension: ResourceDimension
    status: UsageValueStatus
    value: int | float | None
    receipt_ref: str | None
    unavailable_reason: str | None = None
    unavailable_evidence_ref: str | None = None
    schema: str = RESOURCE_COST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESOURCE_COST_SCHEMA:
            raise ValueError("unsupported resource cost schema")
        if not isinstance(self.role, ResourceRole) or not isinstance(self.scope, BudgetScope):
            raise ValueError("resource cost role and scope must use closed enums")
        if not isinstance(self.dimension, ResourceDimension):
            raise ValueError("resource cost dimension must use the closed enum")
        if (
            self.scope is BudgetScope.PER_MODEL_CALL
            and self.dimension is not ResourceDimension.OUTPUT_TOKENS
        ):
            raise ValueError("per-model-call cost is reserved for maximum output tokens")
        if not isinstance(self.status, UsageValueStatus):
            raise ValueError("resource cost status must be a UsageValueStatus")
        if self.status is UsageValueStatus.UNAVAILABLE:
            if self.value is not None:
                raise ValueError("unavailable cost requires a null value")
            _text(self.unavailable_reason, "resource cost unavailable_reason")
            if self.receipt_ref is not None:
                _text(self.receipt_ref, "resource cost usage receipt_ref")
            if self.unavailable_evidence_ref is not None:
                _text(
                    self.unavailable_evidence_ref,
                    "resource cost unavailable_evidence_ref",
                )
        else:
            _finite(self.value, "resource cost value", dimension=self.dimension)
            _text(self.receipt_ref, "resource cost usage receipt_ref")
            if self.unavailable_reason is not None or self.unavailable_evidence_ref is not None:
                raise ValueError(
                    "available cost cannot carry unavailability reason or evidence"
                )

    def to_dict(self) -> dict[str, object]:
        """Serialize cost availability without silently converting missing to zero."""

        return {
            "schema": self.schema,
            "role": self.role.value,
            "scope": self.scope.value,
            "dimension": self.dimension.value,
            "status": self.status.value,
            "value": self.value,
            "receipt_ref": self.receipt_ref,
            "unavailable_reason": self.unavailable_reason,
            "unavailable_evidence_ref": self.unavailable_evidence_ref,
        }


@dataclass(frozen=True)
class WorkItemResultIndexEntryV1:
    """Keep one expected cell and its raw, decision, and cost references."""

    expected_work_item: ExpectedWorkItemV1
    status: WorkItemResultStatus
    status_reason: WorkItemResultReason
    ledger_event_refs: tuple[str, ...]
    raw_receipt_refs: tuple[str, ...]
    scorer_decision_ref: str | None
    costs: tuple[ResourceCostV1, ...]
    budget_validation_ref: str | None = None
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = WORK_ITEM_RESULT_INDEX_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WORK_ITEM_RESULT_INDEX_ENTRY_SCHEMA:
            raise ValueError("unsupported work-item result index entry schema")
        if type(self.expected_work_item) is not ExpectedWorkItemV1:
            raise ValueError("result entry requires one complete expected work item")
        if not isinstance(self.status, WorkItemResultStatus):
            raise ValueError("result status must use the closed enum")
        if self.status_reason not in _RESULT_REASONS_BY_STATUS[self.status]:
            raise ValueError("result reason does not match its result status")
        _unique_texts(self.ledger_event_refs, "ledger_event_refs", portable=False)
        _unique_texts(self.raw_receipt_refs, "raw_receipt_refs", portable=False)
        _records(self.costs, ResourceCostV1, "costs")
        cost_keys = tuple((cost.role, cost.scope, cost.dimension) for cost in self.costs)
        if len(set(cost_keys)) != len(cost_keys):
            raise ValueError("costs cannot repeat a role, scope, and dimension")
        if any(cost.scope is BudgetScope.WHOLE_RUN for cost in self.costs):
            raise ValueError("work-item results cannot embed whole-run costs")
        if self.status is WorkItemResultStatus.NOT_STARTED:
            if (
                self.ledger_event_refs
                or self.raw_receipt_refs
                or self.scorer_decision_ref is not None
                or self.costs
                or self.budget_validation_ref is not None
            ):
                raise ValueError(
                    "not-started work item cannot have results, costs, or budget validation"
                )
        else:
            if not self.ledger_event_refs or not self.costs:
                raise ValueError(
                    "every started work item requires ledger evidence and cost states"
                )
            _text(self.budget_validation_ref, "budget_validation_ref")
        if self.status is WorkItemResultStatus.COMPLETED:
            if not self.raw_receipt_refs:
                raise ValueError("completed work item requires raw evidence")
            _text(self.scorer_decision_ref, "scorer_decision_ref")
        elif self.status is WorkItemResultStatus.METHOD_FAILED:
            # A failed_raw action closes the chain before the scorer runs.
            if not self.raw_receipt_refs or self.scorer_decision_ref is not None:
                raise ValueError("method terminal failure requires raw evidence and no decision")
        elif self.status is WorkItemResultStatus.INFRASTRUCTURE_INVALID:
            if not self.raw_receipt_refs:
                raise ValueError("infrastructure-invalid work item requires raw evidence")
            scorer_rejected_input = (
                self.status_reason is WorkItemResultReason.SCORER_INPUT_INVALID
            )
            if scorer_rejected_input:
                # The scorer completed and returned an invalid-input decision;
                # this differs from an infrastructure failed_raw scorer action.
                _text(self.scorer_decision_ref, "scorer_decision_ref")
            elif self.scorer_decision_ref is not None:
                raise ValueError(
                    "terminal infrastructure failure cannot carry a scorer decision"
                )
        elif self.status is WorkItemResultStatus.AMBIGUOUS_RESUME:
            # Earlier completed actions retain raw receipts. The finalizer must
            # bind their owners and prove the last in-flight action has no raw.
            if self.scorer_decision_ref is not None:
                raise ValueError(
                    "ambiguous in-flight work item cannot claim a scorer decision"
                )
        elif self.scorer_decision_ref is not None:
            raise ValueError("invalid or ambiguous work item cannot carry a scorer decision")
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the entire expected cell and its explicit result state."""

        return {
            "schema": self.schema,
            "expected_work_item": self.expected_work_item.to_dict(),
            "status": self.status.value,
            "status_reason": self.status_reason.value,
            "ledger_event_refs": list(self.ledger_event_refs),
            "raw_receipt_refs": list(self.raw_receipt_refs),
            "scorer_decision_ref": self.scorer_decision_ref,
            "costs": [cost.to_dict() for cost in self.costs],
            "budget_validation_ref": self.budget_validation_ref,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class TraceFixResultIndexV1:
    """Bind every planned cell to exactly one indexed result without shrinking it."""

    result_index_id: str
    plan_id: str
    run_id: str
    suite_id: str
    suite_version: str
    population_id: str
    release_record_ref: str
    scorer_registry_id: str
    source_task_roster: tuple[str, ...]
    expected_work_item_count: int
    expected_work_items: tuple[ExpectedWorkItemV1, ...]
    results: tuple[WorkItemResultIndexEntryV1, ...]
    run_budget_validation_ref: str | None = None
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = TRACEFIX_RESULT_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACEFIX_RESULT_INDEX_SCHEMA:
            raise ValueError("unsupported result index schema")
        for label in ("result_index_id", "plan_id", "run_id", "population_id", "scorer_registry_id"):
            _identifier(getattr(self, label), label)
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID or self.suite_version != SCORER_REGISTRY_SUITE_VERSION_V0_1:
            raise ValueError("result index suite identity must equal the v0.1 freeze")
        _text(self.release_record_ref, "release_record_ref")
        _unique_texts(self.source_task_roster, "source_task_roster", portable=False)
        if not self.source_task_roster:
            raise ValueError("result index requires a source-task roster")
        _count(self.expected_work_item_count, "expected_work_item_count")
        _records(self.expected_work_items, ExpectedWorkItemV1, "expected_work_items")
        _records(self.results, WorkItemResultIndexEntryV1, "results")
        if (
            self.expected_work_item_count == 0
            or len(self.expected_work_items) != self.expected_work_item_count
            or len(self.results) != self.expected_work_item_count
        ):
            raise ValueError("result index must retain every expected work item")
        ids = tuple(item.work_item_id for item in self.expected_work_items)
        _unique_texts(ids, "expected work item IDs", portable=True)
        if tuple(entry.expected_work_item for entry in self.results) != self.expected_work_items:
            raise ValueError("result entries must follow the exact expected work-item roster")
        if any(item.run_id != self.run_id or item.population_id != self.population_id for item in self.expected_work_items):
            raise ValueError("result work item does not bind this run and population")
        source_tasks = {task.casefold(): task for task in self.source_task_roster}
        if {item.source_task_id.casefold() for item in self.expected_work_items} != set(source_tasks):
            raise ValueError("result work items must cover the declared source-task roster")
        if any(item.source_task_id != source_tasks[item.source_task_id.casefold()] for item in self.expected_work_items):
            raise ValueError("result source-task spelling must match the roster")
        composite = tuple(tuple(part.casefold() for part in item.identity()) for item in self.expected_work_items)
        if len(set(composite)) != len(composite):
            raise ValueError("result index cannot repeat a complete work-item identity")
        if self.run_budget_validation_ref is not None:
            _text(self.run_budget_validation_ref, "run_budget_validation_ref")
        # Runner preflight must close treatment, system-cell, budget, and registry refs.
        # It must also compare this roster with the persisted plan and release.
        _unverified(self.implementation_status)

    def to_dict(self) -> dict[str, object]:
        """Serialize the full fixed denominator and all per-cell statuses."""

        return {
            "schema": self.schema,
            "result_index_id": self.result_index_id,
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "release_record_ref": self.release_record_ref,
            "scorer_registry_id": self.scorer_registry_id,
            "source_task_roster": list(self.source_task_roster),
            "expected_work_item_count": self.expected_work_item_count,
            "expected_work_items": [item.to_dict() for item in self.expected_work_items],
            "results": [entry.to_dict() for entry in self.results],
            "run_budget_validation_ref": self.run_budget_validation_ref,
            "implementation_status": self.implementation_status,
        }


__all__ = [
    "ALLOWED_CLOSE_PLAN_SCHEMA", "RUN_ONLY_INPUT_ROOT_SCHEMA",
    "RUN_ONLY_OPERATION_PLAN_SCHEMA",
    "RUN_ONLY_OPERATION_SPEC_SCHEMA", "RUN_ONLY_USAGE_RECEIPT_SCHEMA",
    "WORK_ITEM_BOUNDARY_SCHEMA_V2", "RUN_BOUNDARY_SCHEMA_V2",
    "EXPECTED_WORK_ITEM_SCHEMA", "RESOURCE_BALANCE_SCHEMA", "RESOURCE_COST_SCHEMA",
    "RUN_BUDGET_SCHEMA", "RUN_CONTRACT_IMPLEMENTATION_STATUS", "RUN_LEDGER_EVENT_SCHEMA",
    "STATIC_NO_FIXER_TREATMENT_ID", "TRACK_RUN_CELL_SCHEMA", "TRACK_RUN_MATRIX_SCHEMA",
    "TREATMENT_ARM_SCHEMA",
    "TRACEFIX_RESULT_INDEX_SCHEMA", "TRACEFIX_RUN_PLAN_SCHEMA",
    "WORK_ITEM_ACTION_PLAN_SCHEMA", "WORK_ITEM_ACTION_SPEC_SCHEMA",
    "WORK_ITEM_RESULT_INDEX_ENTRY_SCHEMA",
    "ActionPlanBindingStatus",
    "AllowedClosePlanV1", "RunOnlyInputClosureKind", "RunOnlyInputRootV1",
    "RunOnlyOperationKind", "RunOnlyOperationPlanV1", "RunOnlyOperationSpecV1",
    "RunOnlyOperationStage", "canonical_run_only_operations_v1",
    "ExpectedWorkItemV1", "LedgerEventPhase", "ResourceBalanceV1", "ResourceCostV1",
    "RunActionFailureKind", "RunBudgetV1", "RunLedgerEventV1",
    "TraceFixResultIndexV1", "TraceFixRunPlanV1", "TrackRunCellV1", "TrackRunMatrixV1",
    "TreatmentArmV1", "WorkItemActionKind", "WorkItemActionPlanV1",
    "WorkItemActionSpecV1", "WorkItemResultIndexEntryV1",
    "WorkItemResultReason", "WorkItemResultStatus",
]
