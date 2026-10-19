"""Structural binding for TraceFix-Bench run plans and result indexes.

This locally designed boundary rereads every case through the validated dataset
loaders, expands preregistered complete matrices, and rejects denominator drift.
It also binds plan treatment arms and system-cell names to typed registry
registrations. It does not execute methods, inspect ledger chains, or validate
scientific outcomes, so every successful receipt remains implementation_unverified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracefix_bench.contracts import (
    BenchmarkCaseRefV1,
    ExecutionSemantics,
    TraceFixTrack,
)
from lh_platform.tracefix_bench.datasets import (
    ScorerCaseBundleV1,
    ScorerTraceFixDatasetV1,
    load_public_case,
    load_scorer_case,
)
from lh_platform.tracefix_bench.run_contracts import (
    RUN_CONTRACT_IMPLEMENTATION_STATUS,
    RUN_ONLY_OPERATION_PLAN_SCHEMA,
    STATIC_NO_FIXER_TREATMENT_ID,
    TRACEFIX_RUN_PLAN_SCHEMA,
    ExpectedWorkItemV1,
    RunOnlyOperationPlanV1,
    TraceFixResultIndexV1,
    TraceFixRunPlanV1,
    TrackRunMatrixV1,
    TreatmentArmV1,
)
from lh_platform.tracefix_bench.run_identity_contracts import (
    ARTIFACT_TOPOLOGY_ID_V0_1,
    METERING_CONTRACT_ID_V0_1,
    RUN_METERING_BOUNDARY_ID_V0_1,
    WORK_ITEM_METERING_BOUNDARY_ID_V0_1,
)

RUN_PLAN_BINDING_RECEIPT_SCHEMA = "longhorizon_tracefix_bench/run_plan_binding_receipt/v3"
RESULT_BINDING_RECEIPT_SCHEMA = "longhorizon_tracefix_bench/result_binding_receipt/v2"
STRUCTURAL_BOUND_STATUS = "structural_bound"


class RunBindingErrorCode(StrEnum):
    """Keep every structural rejection machine-readable and fail closed."""

    DATASET_NOT_LOADER_VALIDATED = "dataset_not_loader_validated"
    PLAN_CONTRACT_INVALID = "plan_contract_invalid"
    RESULT_INDEX_CONTRACT_INVALID = "result_index_contract_invalid"
    RELEASE_IDENTITY_MISMATCH = "release_identity_mismatch"
    REGISTRY_IDENTITY_MISMATCH = "registry_identity_mismatch"
    POPULATION_IDENTITY_MISMATCH = "population_identity_mismatch"
    SUITE_IDENTITY_MISMATCH = "suite_identity_mismatch"
    CASE_REVALIDATION_FAILED = "case_revalidation_failed"
    PUBLIC_TRACK_NOT_ADMITTED = "public_track_not_admitted"
    CASE_REFERENCE_MISMATCH = "case_reference_mismatch"
    SOURCE_TASK_ROSTER_MISMATCH = "source_task_roster_mismatch"
    COHORT_MATRIX_MISMATCH = "cohort_matrix_mismatch"
    REGISTRY_REFERENCE_MISMATCH = "registry_reference_mismatch"
    WORK_ITEM_ROSTER_MISMATCH = "work_item_roster_mismatch"
    COMPARISON_FAMILY_ROSTER_MISMATCH = "comparison_family_roster_mismatch"
    RESULT_PLAN_MISMATCH = "result_plan_mismatch"


class TraceFixRunBindingError(ValueError):
    """Report one closed structural failure without promoting evidence state."""

    def __init__(self, code: RunBindingErrorCode, detail: str) -> None:
        if not isinstance(code, RunBindingErrorCode):
            raise ValueError("run binding error requires a closed code")
        if type(detail) is not str or not detail.strip():
            raise ValueError("run binding error requires non-empty detail")
        self.code = code
        self.detail = detail
        super().__init__(f"TraceFix run binding failed: {code.value}: {detail}")


def _count(value: int, label: str) -> None:
    """Reject booleans and negative receipt counts."""

    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


@dataclass(frozen=True, init=False)
class RunPlanBindingReceiptV1:
    """Record structural closure created only by this module's checked binder."""

    run_plan: TraceFixRunPlanV1
    plan_id: str
    run_id: str
    population_id: str
    scorer_registry_id: str
    release_record_ref: str
    artifact_topology_id: str
    metering_contract_id: str
    work_item_metering_boundary_id: str
    run_metering_boundary_id: str
    run_plan_schema_id: str
    run_only_operation_plan_schema_id: str
    run_only_operation_plan_id: str
    run_only_operation_plan: RunOnlyOperationPlanV1
    run_only_operation_count: int
    ordered_run_only_operation_ids: tuple[str, ...]
    allowed_post_work_item_operation_ids: tuple[str, ...]
    allowed_post_run_operation_ids: tuple[str, ...]
    case_count: int
    cohort_count: int
    work_item_count: int
    case_ids: tuple[str, ...]
    matrix_ids: tuple[str, ...]
    source_denominator_entry_ids: tuple[str, ...]
    population_preflight_status: str
    binding_status: str = STRUCTURAL_BOUND_STATUS
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = RUN_PLAN_BINDING_RECEIPT_SCHEMA

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject construction outside the checked run-plan binder."""

        raise TypeError("RunPlanBindingReceiptV1 is created only by bind_run_plan_v1")

    def __post_init__(self) -> None:
        if self.schema != RUN_PLAN_BINDING_RECEIPT_SCHEMA:
            raise ValueError("unsupported run-plan binding receipt schema")
        if type(self.run_plan) is not TraceFixRunPlanV1:
            raise ValueError("run-plan binding receipt requires the exact checked plan")
        for label in (
            "plan_id",
            "run_id",
            "population_id",
            "scorer_registry_id",
            "release_record_ref",
        ):
            _text(getattr(self, label), label)
        # These fields attest only what the typed plan declared at binding time.
        for label, expected in (
            ("artifact_topology_id", ARTIFACT_TOPOLOGY_ID_V0_1),
            ("metering_contract_id", METERING_CONTRACT_ID_V0_1),
            ("work_item_metering_boundary_id", WORK_ITEM_METERING_BOUNDARY_ID_V0_1),
            ("run_metering_boundary_id", RUN_METERING_BOUNDARY_ID_V0_1),
            ("run_plan_schema_id", TRACEFIX_RUN_PLAN_SCHEMA),
            ("run_only_operation_plan_schema_id", RUN_ONLY_OPERATION_PLAN_SCHEMA),
        ):
            value = getattr(self, label)
            if type(value) is not str or value != expected:
                raise ValueError(f"binding receipt {label} must equal the v0.1 freeze")
        _text(self.run_only_operation_plan_id, "run_only_operation_plan_id")
        if type(self.run_only_operation_plan) is not RunOnlyOperationPlanV1:
            raise ValueError("binding receipt requires the complete typed run-only plan")
        for label in ("case_count", "cohort_count", "work_item_count", "run_only_operation_count"):
            _count(getattr(self, label), label)
        for label in (
            "case_ids",
            "matrix_ids",
            "source_denominator_entry_ids",
        ):
            values = getattr(self, label)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(f"{label} must be a tuple of non-empty text")
            if len({value.casefold() for value in values}) != len(values):
                raise ValueError(f"{label} must be unique after case folding")
        if self.case_count != len(self.case_ids):
            raise ValueError("binding receipt case count does not match its roster")
        if self.cohort_count != len(self.matrix_ids):
            raise ValueError("binding receipt cohort count does not match matrix roster")
        for label in (
            "ordered_run_only_operation_ids",
            "allowed_post_work_item_operation_ids",
            "allowed_post_run_operation_ids",
        ):
            values = getattr(self, label)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(f"{label} must contain non-empty IDs")
            if len({value.casefold() for value in values}) != len(values):
                raise ValueError(f"{label} cannot repeat an operation ID")
        if self.run_only_operation_count != len(self.ordered_run_only_operation_ids):
            raise ValueError("binding receipt operation count does not match its roster")
        embedded_plan = self.run_only_operation_plan
        if (
            self.run_plan.plan_id != self.plan_id
            or self.run_plan.run_id != self.run_id
            or self.run_plan.population_id != self.population_id
            or self.run_plan.scorer_registry_id != self.scorer_registry_id
            or self.run_plan.release_record_ref != self.release_record_ref
            or self.run_plan.artifact_topology_id != self.artifact_topology_id
            or self.run_plan.metering_contract_id != self.metering_contract_id
            or self.run_plan.work_item_metering_boundary_id != self.work_item_metering_boundary_id
            or self.run_plan.run_metering_boundary_id != self.run_metering_boundary_id
            or self.run_plan.schema != self.run_plan_schema_id
            or self.run_plan.run_only_operation_plan != embedded_plan
            or self.run_plan.expected_work_item_count != self.work_item_count
        ):
            raise ValueError("binding receipt summaries differ from the exact checked run plan")
        embedded_operation_ids = tuple(
            operation.operation_id for operation in embedded_plan.operations
        )
        if (
            embedded_plan.plan_id != self.plan_id
            or embedded_plan.run_id != self.run_id
            or embedded_plan.schema != self.run_only_operation_plan_schema_id
            or embedded_plan.plan_id != self.run_only_operation_plan_id
            or len(embedded_plan.operations) != self.run_only_operation_count
            or embedded_operation_ids != self.ordered_run_only_operation_ids
            or embedded_plan.allowed_close_plan.post_work_item_operation_ids
            != self.allowed_post_work_item_operation_ids
            or embedded_plan.allowed_close_plan.post_run_operation_ids
            != self.allowed_post_run_operation_ids
        ):
            raise ValueError(
                "binding receipt summaries must equal the complete embedded run-only plan"
            )
        if (
            embedded_plan.run_plan_schema_id != self.run_plan_schema_id
            or embedded_plan.artifact_topology_id != self.artifact_topology_id
            or embedded_plan.metering_contract_id != self.metering_contract_id
            or embedded_plan.work_item_metering_boundary_id != self.work_item_metering_boundary_id
            or embedded_plan.run_metering_boundary_id != self.run_metering_boundary_id
        ):
            raise ValueError(
                "binding receipt contract identities disagree with the embedded run-only plan"
            )
        if self.work_item_count != len(embedded_plan.expected_work_item_ids):
            raise ValueError("binding receipt work-item count disagrees with run-only plan")
        _text(self.population_preflight_status, "population_preflight_status")
        if self.binding_status != STRUCTURAL_BOUND_STATUS:
            raise ValueError("run-plan binding receipt can only be structural_bound")
        if self.implementation_status != RUN_CONTRACT_IMPLEMENTATION_STATUS:
            raise ValueError("run-plan binding receipt cannot promote implementation")

    def to_dict(self) -> dict[str, object]:
        """Serialize the exact structural scope of the binding decision."""

        return {
            "schema": self.schema,
            "run_plan": self.run_plan.to_dict(),
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "population_id": self.population_id,
            "scorer_registry_id": self.scorer_registry_id,
            "release_record_ref": self.release_record_ref,
            "artifact_topology_id": self.artifact_topology_id,
            "metering_contract_id": self.metering_contract_id,
            "work_item_metering_boundary_id": self.work_item_metering_boundary_id,
            "run_metering_boundary_id": self.run_metering_boundary_id,
            "run_plan_schema_id": self.run_plan_schema_id,
            "run_only_operation_plan_schema_id": self.run_only_operation_plan_schema_id,
            "run_only_operation_plan_id": self.run_only_operation_plan_id,
            "run_only_operation_plan": self.run_only_operation_plan.to_dict(),
            "run_only_operation_count": self.run_only_operation_count,
            "ordered_run_only_operation_ids": list(self.ordered_run_only_operation_ids),
            "allowed_post_work_item_operation_ids": list(self.allowed_post_work_item_operation_ids),
            "allowed_post_run_operation_ids": list(self.allowed_post_run_operation_ids),
            "case_count": self.case_count,
            "cohort_count": self.cohort_count,
            "work_item_count": self.work_item_count,
            "case_ids": list(self.case_ids),
            "matrix_ids": list(self.matrix_ids),
            "source_denominator_entry_ids": list(self.source_denominator_entry_ids),
            "population_preflight_status": self.population_preflight_status,
            "binding_status": self.binding_status,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True, init=False)
class ResultBindingReceiptV1:
    """Record that a result index retains the structurally bound plan grid."""

    result_index: TraceFixResultIndexV1
    result_index_id: str
    plan_id: str
    run_id: str
    population_id: str
    scorer_registry_id: str
    expected_work_item_count: int
    retained_result_count: int
    plan_binding_receipt: RunPlanBindingReceiptV1
    binding_status: str = STRUCTURAL_BOUND_STATUS
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = RESULT_BINDING_RECEIPT_SCHEMA

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject construction outside the checked result-index binder."""

        raise TypeError("ResultBindingReceiptV1 is created only by bind_result_index_v1")

    def __post_init__(self) -> None:
        if self.schema != RESULT_BINDING_RECEIPT_SCHEMA:
            raise ValueError("unsupported result binding receipt schema")
        if type(self.result_index) is not TraceFixResultIndexV1:
            raise ValueError("result binding receipt requires the exact checked index")
        for label in (
            "result_index_id",
            "plan_id",
            "run_id",
            "population_id",
            "scorer_registry_id",
        ):
            _text(getattr(self, label), label)
        _count(self.expected_work_item_count, "expected_work_item_count")
        _count(self.retained_result_count, "retained_result_count")
        if self.expected_work_item_count != self.retained_result_count:
            raise ValueError("result binding receipt must retain the complete grid")
        if type(self.plan_binding_receipt) is not RunPlanBindingReceiptV1:
            raise ValueError("result receipt requires its run-plan binding receipt")
        if (
            self.result_index.result_index_id != self.result_index_id
            or self.result_index.plan_id != self.plan_id
            or self.result_index.run_id != self.run_id
            or self.result_index.population_id != self.population_id
            or self.result_index.scorer_registry_id != self.scorer_registry_id
            or self.result_index.expected_work_item_count != self.expected_work_item_count
            or len(self.result_index.results) != self.retained_result_count
            or self.plan_binding_receipt.plan_id != self.plan_id
            or self.plan_binding_receipt.run_id != self.run_id
            or self.plan_binding_receipt.population_id != self.population_id
            or self.plan_binding_receipt.scorer_registry_id != self.scorer_registry_id
            or self.plan_binding_receipt.work_item_count != self.expected_work_item_count
        ):
            raise ValueError("result receipt does not match its plan binding receipt")
        if self.binding_status != STRUCTURAL_BOUND_STATUS:
            raise ValueError("result binding receipt can only be structural_bound")
        if self.implementation_status != RUN_CONTRACT_IMPLEMENTATION_STATUS:
            raise ValueError("result binding receipt cannot promote implementation")

    def to_dict(self) -> dict[str, object]:
        """Serialize plan identity and the retained fixed denominator."""

        return {
            "schema": self.schema,
            "result_index": self.result_index.to_dict(),
            "result_index_id": self.result_index_id,
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "population_id": self.population_id,
            "scorer_registry_id": self.scorer_registry_id,
            "expected_work_item_count": self.expected_work_item_count,
            "retained_result_count": self.retained_result_count,
            "plan_binding_receipt": self.plan_binding_receipt.to_dict(),
            "binding_status": self.binding_status,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class _BoundCaseV1:
    """Retain only the case fields needed to expand one run matrix."""

    case_ref: BenchmarkCaseRefV1
    track: TraceFixTrack
    execution_semantics: ExecutionSemantics

    def cohort_identity(self) -> tuple[object, ...]:
        return (
            self.track,
            self.case_ref.benchmark_id,
            self.case_ref.benchmark_version,
            self.case_ref.adapter_version,
            self.case_ref.split,
            self.execution_semantics,
        )


def _make_run_plan_binding_receipt_v1(
    *,
    plan: TraceFixRunPlanV1,
    bound_cases: tuple[_BoundCaseV1, ...],
    matrices: dict[tuple[object, ...], TrackRunMatrixV1],
    expected_work_item_count: int,
    dataset: ScorerTraceFixDatasetV1,
) -> RunPlanBindingReceiptV1:
    """Construct a receipt only after bind_run_plan_v1 completed every check."""

    receipt = object.__new__(RunPlanBindingReceiptV1)
    values: dict[str, object] = {
        "run_plan": plan,
        "plan_id": plan.plan_id,
        "run_id": plan.run_id,
        "population_id": plan.population_id,
        "scorer_registry_id": plan.scorer_registry_id,
        "release_record_ref": plan.release_record_ref,
        "artifact_topology_id": plan.artifact_topology_id,
        "metering_contract_id": plan.metering_contract_id,
        "work_item_metering_boundary_id": plan.work_item_metering_boundary_id,
        "run_metering_boundary_id": plan.run_metering_boundary_id,
        "run_plan_schema_id": plan.schema,
        "run_only_operation_plan_schema_id": plan.run_only_operation_plan.schema,
        "run_only_operation_plan_id": plan.run_only_operation_plan.plan_id,
        "run_only_operation_plan": plan.run_only_operation_plan,
        "run_only_operation_count": len(plan.run_only_operation_plan.operations),
        "ordered_run_only_operation_ids": tuple(
            operation.operation_id for operation in plan.run_only_operation_plan.operations
        ),
        "allowed_post_work_item_operation_ids": (
            plan.run_only_operation_plan.allowed_close_plan.post_work_item_operation_ids
        ),
        "allowed_post_run_operation_ids": (
            plan.run_only_operation_plan.allowed_close_plan.post_run_operation_ids
        ),
        "case_count": len(bound_cases),
        "cohort_count": len(matrices),
        "work_item_count": expected_work_item_count,
        "case_ids": tuple(case.case_ref.case_id for case in bound_cases),
        "matrix_ids": tuple(matrix.matrix_id for matrix in plan.track_run_matrices),
        "source_denominator_entry_ids": (dataset.preflight_receipt.source_denominator_entry_ids),
        "population_preflight_status": dataset.preflight_receipt.preflight_status,
        "binding_status": STRUCTURAL_BOUND_STATUS,
        "implementation_status": RUN_CONTRACT_IMPLEMENTATION_STATUS,
        "schema": RUN_PLAN_BINDING_RECEIPT_SCHEMA,
    }
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    receipt.__post_init__()
    return receipt


def _make_result_binding_receipt_v1(
    *,
    index: TraceFixResultIndexV1,
    plan_binding_receipt: RunPlanBindingReceiptV1,
) -> ResultBindingReceiptV1:
    """Construct a result receipt only after the public binder closes the index."""

    receipt = object.__new__(ResultBindingReceiptV1)
    values: dict[str, object] = {
        "result_index": index,
        "result_index_id": index.result_index_id,
        "plan_id": index.plan_id,
        "run_id": index.run_id,
        "population_id": index.population_id,
        "scorer_registry_id": index.scorer_registry_id,
        "expected_work_item_count": index.expected_work_item_count,
        "retained_result_count": len(index.results),
        "plan_binding_receipt": plan_binding_receipt,
        "binding_status": STRUCTURAL_BOUND_STATUS,
        "implementation_status": RUN_CONTRACT_IMPLEMENTATION_STATUS,
        "schema": RESULT_BINDING_RECEIPT_SCHEMA,
    }
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    receipt.__post_init__()
    return receipt


def _fail(code: RunBindingErrorCode, detail: str) -> None:
    raise TraceFixRunBindingError(code, detail)


def _validate_dataset_identity(
    plan: TraceFixRunPlanV1,
    dataset: ScorerTraceFixDatasetV1,
) -> None:
    """Close plan identity against all validated release layers."""

    public_index = dataset.public_dataset.index
    scorer_index = dataset.index
    registry = dataset.registry
    release = dataset.release_record
    suite_values = {
        plan.suite_id,
        public_index.suite_id,
        scorer_index.suite_id,
        registry.suite_id,
        release.suite_id,
    }
    version_values = {
        plan.suite_version,
        public_index.suite_version,
        scorer_index.suite_version,
        registry.suite_version,
        release.suite_version,
    }
    if len(suite_values) != 1 or len(version_values) != 1:
        _fail(
            RunBindingErrorCode.SUITE_IDENTITY_MISMATCH,
            "plan, indexes, registry, and release must use one exact suite identity",
        )
    population_values = {
        plan.population_id,
        public_index.population_id,
        scorer_index.population_id,
        registry.population_id,
        release.population_id,
    }
    if len(population_values) != 1:
        _fail(
            RunBindingErrorCode.POPULATION_IDENTITY_MISMATCH,
            "plan, indexes, registry, and release must use one population",
        )
    if plan.scorer_registry_id != registry.registry_id:
        _fail(
            RunBindingErrorCode.REGISTRY_IDENTITY_MISMATCH,
            "plan scorer_registry_id does not equal the loaded registry",
        )
    release_ref = dataset.canonical_release_record_path.as_posix()
    if plan.release_record_ref != release_ref:
        _fail(
            RunBindingErrorCode.RELEASE_IDENTITY_MISMATCH,
            "plan release_record_ref is not the loaded canonical release record",
        )


def _load_bound_cases(
    plan: TraceFixRunPlanV1,
    dataset: ScorerTraceFixDatasetV1,
) -> tuple[_BoundCaseV1, ...]:
    """Reread every public/private pair through the existing strict loaders."""

    bound: list[_BoundCaseV1] = []
    source_roster: list[str] = []
    seen_sources: set[str] = set()
    for public_entry in dataset.public_dataset.index.cases:
        try:
            public_case = load_public_case(dataset.public_dataset, public_entry.case_id)
            scorer_bundle: ScorerCaseBundleV1 = load_scorer_case(
                dataset,
                public_case=public_case,
            )
        except Exception as error:
            raise TraceFixRunBindingError(
                RunBindingErrorCode.CASE_REVALIDATION_FAILED,
                f"case {public_entry.case_id} failed strict loader revalidation",
            ) from error
        if public_case.track not in scorer_bundle.admission.eligible_tracks:
            _fail(
                RunBindingErrorCode.PUBLIC_TRACK_NOT_ADMITTED,
                f"case {public_entry.case_id} public track is not admitted",
            )
        case_ref = scorer_bundle.case_ref
        if (
            case_ref.case_id != public_entry.case_id
            or case_ref.case_ordinal != public_entry.case_ordinal
            or public_case.track is not public_entry.track
            or public_case.execution_semantics is not public_entry.execution_semantics
        ):
            _fail(
                RunBindingErrorCode.CASE_REFERENCE_MISMATCH,
                f"case {public_entry.case_id} changed indexed identity",
            )
        folded_source = case_ref.source_task_id.casefold()
        if folded_source not in seen_sources:
            seen_sources.add(folded_source)
            source_roster.append(case_ref.source_task_id)
        bound.append(
            _BoundCaseV1(
                case_ref=case_ref,
                track=public_case.track,
                execution_semantics=public_case.execution_semantics,
            )
        )
    if tuple(source_roster) != plan.source_task_roster:
        _fail(
            RunBindingErrorCode.SOURCE_TASK_ROSTER_MISMATCH,
            "plan source-task roster must equal first occurrence order in the release",
        )
    return tuple(bound)


def _matrix_by_cohort(
    plan: TraceFixRunPlanV1,
    bound_cases: tuple[_BoundCaseV1, ...],
) -> dict[tuple[object, ...], TrackRunMatrixV1]:
    """Require one matrix for every populated cohort and no empty cohort matrix."""

    matrices = {matrix.cohort_identity(): matrix for matrix in plan.track_run_matrices}
    case_cohorts = {case.cohort_identity() for case in bound_cases}
    if set(matrices) != case_cohorts:
        _fail(
            RunBindingErrorCode.COHORT_MATRIX_MISMATCH,
            "track matrices must cover each populated dataset cohort exactly once",
        )
    return matrices


def _contract_domain(contract: object) -> tuple[str, str, str]:
    """Return the exact benchmark identity frozen by one registry contract."""

    return (
        contract.benchmark_id,
        contract.benchmark_version,
        contract.adapter_version,
    )


def _matrix_domain(matrix: TrackRunMatrixV1) -> tuple[str, str, str]:
    """Return the exact benchmark identity consumed by one populated matrix."""

    return (
        matrix.benchmark_id,
        matrix.benchmark_version,
        matrix.adapter_version,
    )


def _validate_scorer_domain_coverage(
    plan: TraceFixRunPlanV1,
    dataset: ScorerTraceFixDatasetV1,
) -> None:
    """Require every populated causal/recovery domain to have its frozen authority."""

    registry = dataset.registry
    metrics = {
        contract.progress_contract_id.casefold(): contract for contract in registry.progress_metrics
    }
    verifiers = {
        contract.verifier_contract_id.casefold(): contract for contract in registry.verifiers
    }
    oracles = {contract.oracle_contract_id.casefold(): contract for contract in registry.oracles}
    scorer_by_track = {scorer.track: scorer for scorer in registry.scorers}
    for track in TraceFixTrack:
        populated_domains = {
            _matrix_domain(matrix) for matrix in plan.track_run_matrices if matrix.track is track
        }
        if not populated_domains:
            continue
        scorer = scorer_by_track[track]
        progress_domains = {
            _contract_domain(metrics[identity.casefold()])
            for identity in scorer.progress_contract_ids
        }
        verifier_domains = {
            _contract_domain(verifiers[identity.casefold()])
            for identity in scorer.verifier_contract_ids
        }
        oracle_domains = {
            _contract_domain(oracles[identity.casefold()])
            for identity in scorer.oracle_contract_ids
        }
        if track is TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION:
            if progress_domains != populated_domains or verifier_domains != populated_domains:
                _fail(
                    RunBindingErrorCode.REGISTRY_REFERENCE_MISMATCH,
                    "historical populated domains must close exactly over progress and verifier contracts",
                )
        elif track is TraceFixTrack.FORWARD_RECOVERY:
            if (
                verifier_domains != populated_domains
                or not progress_domains <= populated_domains
                or not oracle_domains <= populated_domains
            ):
                _fail(
                    RunBindingErrorCode.REGISTRY_REFERENCE_MISMATCH,
                    "forward populated domains must close over verifiers and contain optional metric/oracle domains",
                )


def _validate_registry_references(
    plan: TraceFixRunPlanV1,
    dataset: ScorerTraceFixDatasetV1,
) -> None:
    """Bind every plan method and explicit system join to registry-owned meaning."""

    registry = dataset.registry
    scorer_by_track = {scorer.track: scorer for scorer in registry.scorers}
    arms_by_id: dict[str, TreatmentArmV1] = {
        arm.treatment_id.casefold(): arm for arm in plan.treatment_arms
    }
    registered_treatments = {
        registration.treatment_id.casefold(): registration for registration in registry.treatments
    }
    registered_systems = {
        registration.system_cell_id.casefold(): registration
        for registration in registry.system_cells
    }
    for arm in plan.treatment_arms:
        registration = registered_treatments.get(arm.treatment_id.casefold())
        if registration is None or (
            arm.treatment_id != registration.treatment_id
            or arm.locator_treatment_id != registration.locator_treatment_id
            or arm.fixer_treatment_id != registration.fixer_treatment_id
            or arm.method_config_refs != registration.method_config_refs
        ):
            _fail(
                RunBindingErrorCode.REGISTRY_REFERENCE_MISMATCH,
                f"treatment arm {arm.treatment_id} differs from its registry definition",
            )
    for matrix in plan.track_run_matrices:
        scorer = scorer_by_track[matrix.track]
        expected_cells: list[tuple[str, str, str]] = []
        for system_cell_id in scorer.system_cell_ids:
            system = registered_systems[system_cell_id.casefold()]
            if (
                system.population_id != plan.population_id
                or system.benchmark_id != matrix.benchmark_id
                or system.benchmark_version != matrix.benchmark_version
                or system.adapter_version != matrix.adapter_version
                or matrix.track not in system.applicable_tracks
                or system.execution_semantics is not matrix.execution_semantics
            ):
                continue
            for treatment_id in scorer.treatment_ids:
                treatment = registered_treatments[treatment_id.casefold()]
                if treatment.fixer_treatment_id == system.fixer_treatment_id:
                    expected_cells.append(
                        (
                            system.system_cell_id,
                            treatment.treatment_id,
                            system.budget_id,
                        )
                    )
        actual_cells = tuple(cell.identity() for cell in matrix.registered_cells)
        if actual_cells != tuple(expected_cells):
            _fail(
                RunBindingErrorCode.REGISTRY_REFERENCE_MISMATCH,
                f"matrix {matrix.matrix_id} must equal the ordered complete registry-compatible join",
            )
        for cell in matrix.registered_cells:
            arm = arms_by_id.get(cell.treatment_id.casefold())
            if arm is None or arm.treatment_id != cell.treatment_id:
                _fail(
                    RunBindingErrorCode.REGISTRY_REFERENCE_MISMATCH,
                    f"matrix {matrix.matrix_id} treatment has no exact plan arm",
                )
            if matrix.track is TraceFixTrack.STATIC_ATTRIBUTION:
                if arm.fixer_treatment_id != STATIC_NO_FIXER_TREATMENT_ID:
                    _fail(
                        RunBindingErrorCode.REGISTRY_REFERENCE_MISMATCH,
                        "static matrix treatment arm does not use static_no_fixer",
                    )
            elif arm.fixer_treatment_id == STATIC_NO_FIXER_TREATMENT_ID:
                _fail(
                    RunBindingErrorCode.REGISTRY_REFERENCE_MISMATCH,
                    "non-static matrix treatment arm uses static_no_fixer",
                )
    declared_configs = set(plan.method_config_refs)
    referenced_configs = {value for arm in plan.treatment_arms for value in arm.method_config_refs}
    if declared_configs != referenced_configs:
        _fail(
            RunBindingErrorCode.REGISTRY_REFERENCE_MISMATCH,
            "plan config refs do not close exactly over treatment arms",
        )


def _work_item_signature(item: ExpectedWorkItemV1) -> tuple[object, ...]:
    """Compare a planned cell without deriving or trusting its display ID."""

    return (
        item.run_id,
        item.population_id,
        item.case_id,
        item.source_task_id,
        item.track,
        item.split,
        item.execution_semantics,
        item.benchmark_id,
        item.benchmark_version,
        item.adapter_version,
        item.system_cell_id,
        item.treatment_id,
        item.locator_treatment_id,
        item.fixer_treatment_id,
        item.budget_id,
        item.repetition_id,
    )


def _expected_matrix_signatures(
    plan: TraceFixRunPlanV1,
    bound_cases: tuple[_BoundCaseV1, ...],
) -> tuple[tuple[object, ...], ...]:
    """Expand case, legal registry-cell, and repetition order canonically."""

    arms = {arm.treatment_id: arm for arm in plan.treatment_arms}
    expanded: list[tuple[object, ...]] = []
    for matrix in plan.track_run_matrices:
        cohort_cases = tuple(
            case for case in bound_cases if case.cohort_identity() == matrix.cohort_identity()
        )
        for case in cohort_cases:
            for cell in matrix.registered_cells:
                arm = arms[cell.treatment_id]
                for repetition_id in matrix.repetition_ids:
                    ref = case.case_ref
                    expanded.append(
                        (
                            plan.run_id,
                            plan.population_id,
                            ref.case_id,
                            ref.source_task_id,
                            matrix.track,
                            matrix.split,
                            matrix.execution_semantics,
                            matrix.benchmark_id,
                            matrix.benchmark_version,
                            matrix.adapter_version,
                            cell.system_cell_id,
                            cell.treatment_id,
                            arm.locator_treatment_id,
                            arm.fixer_treatment_id,
                            cell.budget_id,
                            repetition_id,
                        )
                    )
    return tuple(expanded)


def _validate_comparison_families(
    plan: TraceFixRunPlanV1,
    dataset: ScorerTraceFixDatasetV1,
) -> None:
    """Require every paired treatment to retain the same full registered roster."""

    items = plan.expected_work_items
    registered_systems = {system.system_cell_id: system for system in dataset.registry.system_cells}
    for family in dataset.registry.comparison_families:
        family_domain = (
            family.benchmark_id,
            family.benchmark_version,
            family.adapter_version,
        )
        matrices = tuple(
            matrix
            for matrix in plan.track_run_matrices
            if matrix.track is family.track
            and _matrix_domain(matrix) == family_domain
            and matrix.execution_semantics is family.execution_semantics
        )
        if not matrices:
            _fail(
                RunBindingErrorCode.COMPARISON_FAMILY_ROSTER_MISMATCH,
                f"family {family.family_id} has no populated matching cohort",
            )
        family_cells = {
            (
                system_cell_id,
                treatment_id,
                registered_systems[system_cell_id].budget_id,
            )
            for system_cell_id in family.system_cell_ids
            for treatment_id in family.treatment_ids
        }
        for matrix in matrices:
            matrix_cells = {cell.identity() for cell in matrix.registered_cells}
            if not family_cells <= matrix_cells:
                _fail(
                    RunBindingErrorCode.COMPARISON_FAMILY_ROSTER_MISMATCH,
                    f"family {family.family_id} legal cells are not represented in every track cohort",
                )
        rosters: list[tuple[tuple[object, ...], ...]] = []
        for treatment_id in family.treatment_ids:
            roster = tuple(
                (
                    item.case_id,
                    item.source_task_id,
                    item.benchmark_id,
                    item.benchmark_version,
                    item.adapter_version,
                    item.split,
                    item.execution_semantics,
                    item.system_cell_id,
                    item.budget_id,
                    item.repetition_id,
                )
                for item in items
                if item.track is family.track
                and (
                    item.benchmark_id,
                    item.benchmark_version,
                    item.adapter_version,
                )
                == family_domain
                and item.execution_semantics is family.execution_semantics
                and item.treatment_id == treatment_id
                and item.system_cell_id in family.system_cell_ids
                and item.budget_id in family.budget_ids
            )
            rosters.append(roster)
        if any(roster != rosters[0] for roster in rosters[1:]):
            _fail(
                RunBindingErrorCode.COMPARISON_FAMILY_ROSTER_MISMATCH,
                f"family {family.family_id} treatments do not share one paired roster",
            )


def bind_run_plan_v1(
    plan: TraceFixRunPlanV1,
    dataset: ScorerTraceFixDatasetV1,
) -> RunPlanBindingReceiptV1:
    """Bind the grid and retain plan IDs without claiming artifact or meter proof."""

    if not isinstance(plan, TraceFixRunPlanV1):
        _fail(
            RunBindingErrorCode.PLAN_CONTRACT_INVALID,
            "plan must be a TraceFixRunPlanV1",
        )
    if not isinstance(dataset, ScorerTraceFixDatasetV1):
        _fail(
            RunBindingErrorCode.DATASET_NOT_LOADER_VALIDATED,
            "dataset must be returned by load_scorer_dataset",
        )
    _validate_dataset_identity(plan, dataset)
    bound_cases = _load_bound_cases(plan, dataset)
    matrices = _matrix_by_cohort(plan, bound_cases)
    _validate_scorer_domain_coverage(plan, dataset)
    _validate_registry_references(plan, dataset)
    expected = _expected_matrix_signatures(plan, bound_cases)
    actual = tuple(_work_item_signature(item) for item in plan.expected_work_items)
    if actual != expected:
        _fail(
            RunBindingErrorCode.WORK_ITEM_ROSTER_MISMATCH,
            "work-item roster must equal the canonical complete registry-join expansion",
        )
    _validate_comparison_families(plan, dataset)
    return _make_run_plan_binding_receipt_v1(
        plan=plan,
        bound_cases=bound_cases,
        matrices=matrices,
        expected_work_item_count=len(expected),
        dataset=dataset,
    )


def bind_result_index_v1(
    index: TraceFixResultIndexV1,
    plan: TraceFixRunPlanV1,
    dataset: ScorerTraceFixDatasetV1,
) -> ResultBindingReceiptV1:
    """Rebind the plan, then require the result index to retain its entire grid."""

    if not isinstance(index, TraceFixResultIndexV1):
        _fail(
            RunBindingErrorCode.RESULT_INDEX_CONTRACT_INVALID,
            "index must be a TraceFixResultIndexV1",
        )
    plan_binding_receipt = bind_run_plan_v1(plan, dataset)
    if (
        index.plan_id != plan.plan_id
        or index.run_id != plan.run_id
        or index.suite_id != plan.suite_id
        or index.suite_version != plan.suite_version
        or index.population_id != plan.population_id
        or index.release_record_ref != plan.release_record_ref
        or index.scorer_registry_id != plan.scorer_registry_id
        or index.source_task_roster != plan.source_task_roster
        or index.expected_work_item_count != plan.expected_work_item_count
        or index.expected_work_items != plan.expected_work_items
        or len(index.results) != plan.expected_work_item_count
    ):
        _fail(
            RunBindingErrorCode.RESULT_PLAN_MISMATCH,
            "result index identity or fixed denominator differs from the bound plan",
        )
    if (plan.run_budget is None) != (index.run_budget_validation_ref is None):
        _fail(
            RunBindingErrorCode.RESULT_PLAN_MISMATCH,
            "run budget and run-budget validation reference must be present together",
        )
    return _make_result_binding_receipt_v1(
        index=index,
        plan_binding_receipt=plan_binding_receipt,
    )


__all__ = [
    "RESULT_BINDING_RECEIPT_SCHEMA",
    "RUN_PLAN_BINDING_RECEIPT_SCHEMA",
    "STRUCTURAL_BOUND_STATUS",
    "ResultBindingReceiptV1",
    "RunBindingErrorCode",
    "RunPlanBindingReceiptV1",
    "TraceFixRunBindingError",
    "bind_result_index_v1",
    "bind_run_plan_v1",
]
