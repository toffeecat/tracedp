"""Pure exchange contracts for LongHorizon-TraceFix-Bench v0.1.

These types freeze the public/scorer boundary and the Trace-to-Fix seam.  They
intentionally contain no loader, runner, executor, scorer, or release claim;
the implementation therefore remains ``implementation_unverified``.
Method payloads are constructed from explicit public types.  Semantic redaction
and source projection remain responsibilities of a future independently audited
adapter policy and loader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracing.unified import TraceFailureDomain, TraceFindingKind

TRACEFIX_BENCH_IMPLEMENTATION_STATUS = "implementation_unverified"
BENCHMARK_CASE_REF_SCHEMA = "longhorizon_tracefix_bench/case_ref/v1"
SOURCE_DENOMINATOR_ENTRY_SCHEMA = "longhorizon_tracefix_bench/source_denominator_entry/v1"
PUBLIC_TRACE_PACKET_SCHEMA = "longhorizon_tracefix_bench/public_trace_packet/v1"
PUBLIC_TRACE_PROJECTION_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/public_trace_projection_receipt/v1"
)
LOCATOR_RECEIPT_SCHEMA = "longhorizon_tracefix_bench/locator_receipt/v1"
FIXER_REQUEST_SCHEMA = "longhorizon_tracefix_bench/fixer_request/v1"
ATTRIBUTION_GOLD_SCHEMA = "longhorizon_tracefix_bench/attribution_gold/v1"
CASE_ADMISSION_SCHEMA = "longhorizon_tracefix_bench/case_admission/v1"
LOCATOR_MODEL_CALL_USAGE_SCHEMA = "longhorizon_tracefix_bench/locator_model_call_usage/v1"
LOCATOR_MODEL_CALL_USAGE_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/locator_model_call_usage_receipt/v1"
)
PROVIDER_TOKEN_BUCKET_SCHEMA = "longhorizon_tracefix_bench/provider_token_bucket/v1"
PROVIDER_TOKEN_PARTITION_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/provider_token_partition_receipt/v1"
)

_FORBIDDEN_PUBLIC_EVENT_FIELD_NAMES = frozenset(
    {
        "actor_identity",
        "context",
        "gold",
        "metadata",
        "method_id",
        "observed_outcome",
        "projection_receipt_id",
        "raw_content",
        "source_artifact_ref",
        "source_task_id",
        "system_cell_id",
    }
)


class TraceFixTrack(StrEnum):
    """Keep the three benchmark estimands separate at every exchange boundary."""

    STATIC_ATTRIBUTION = "static_attribution"
    HISTORICAL_CAUSAL_VERIFICATION = "historical_causal_verification"
    FORWARD_RECOVERY = "forward_recovery"


class ExecutionSemantics(StrEnum):
    """Declare the mutable-state semantics under which one case is consumed."""

    FROZEN_TRACE = "frozen_trace"
    FRESH_BRANCH = "fresh_branch"
    PERSISTENT_SESSION = "persistent_session"
    EXACT_CHECKPOINT_REPLAY = "exact_checkpoint_replay"


class BenchmarkSplit(StrEnum):
    """Represent the four task-clustered partitions defined by the v0.1 contract."""

    DEVELOPMENT = "development"
    QUALIFICATION = "qualification"
    CONFIRMATORY = "confirmatory"
    MEMORY_TRAIN = "memory_train"


class PopulationRole(StrEnum):
    """Separate engineering evidence from a future scientific denominator."""

    DEVELOPMENT = "development"
    INFRASTRUCTURE_QUALIFICATION = "infrastructure_qualification"
    SCIENTIFIC_EVALUATION = "scientific_evaluation"


class ExposureStatus(StrEnum):
    """Record whether method outputs or private labels have exposed a source task."""

    UNEXPOSED = "unexposed"
    EXPOSED = "exposed"
    UNKNOWN = "unknown"


class CaseOrigin(StrEnum):
    """Distinguish observed actor failures from explicitly controlled failures."""

    NATURAL_FAILURE = "natural_failure"
    CONTROLLED_FAILURE = "controlled_failure"


class SourceOutcomeStatus(StrEnum):
    """Preserve every source-population outcome before any case materialization."""

    SUCCESS = "success"
    NATURAL_FAILURE = "natural_failure"
    ACTOR_PARSER_INVALID = "actor_parser_invalid"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"


class MaterializationStatus(StrEnum):
    """Record the separate decision to materialize a source row as benchmark cases."""

    NOT_ATTEMPTED = "not_attempted"
    ADMITTED = "admitted"
    REJECTED = "rejected"


class FieldCoverageStatus(StrEnum):
    """Describe why a public runtime field is present or absent."""

    PRESENT = "present"
    OMITTED_BY_SOURCE = "omitted_by_source"
    REDACTED = "redacted"
    NOT_APPLICABLE = "not_applicable"


class PublicEventKind(StrEnum):
    """Define the closed event vocabulary accepted by the public projection."""

    TASK = "task"
    OBSERVATION = "observation"
    MODEL_INPUT = "model_input"
    MODEL_OUTPUT = "model_output"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ENVIRONMENT_ACTION = "environment_action"
    ENVIRONMENT_OBSERVATION = "environment_observation"
    CHECKPOINT = "checkpoint"
    FEEDBACK = "feedback"
    FAILURE = "failure"


class PublicFieldValueKind(StrEnum):
    """Make every event field declare one primitive public value type."""

    TEXT = "text"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    TEXT_LIST = "text_list"


class LocatorStatus(StrEnum):
    """Keep method abstention distinct from method and infrastructure failures."""

    COMPLETED = "completed"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"
    INVALID = "invalid"


class LocatorFailureKind(StrEnum):
    """Classify locator failures without relying on free-form status text."""

    METHOD_BUDGET_EXHAUSTED = "method_budget_exhausted"
    METHOD_OUTPUT_INVALID = "method_output_invalid"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_FAILURE = "provider_failure"
    WORKER_EXECUTION_FAILED = "worker_execution_failed"
    RESPONSE_TRUNCATED = "response_truncated"
    RESUME_ARTIFACT_MISMATCH = "resume_artifact_mismatch"
    ADAPTER_INPUT_INVALID = "adapter_input_invalid"


# Keep method failures separate from failures of providers, workers, or adapters.
_LOCATOR_FAILURE_KINDS_BY_DOMAIN: dict[TraceFailureDomain, frozenset[LocatorFailureKind]] = {
    TraceFailureDomain.METHOD: frozenset(
        {
            LocatorFailureKind.METHOD_BUDGET_EXHAUSTED,
            LocatorFailureKind.METHOD_OUTPUT_INVALID,
            LocatorFailureKind.RESPONSE_TRUNCATED,
        }
    ),
    TraceFailureDomain.INFRASTRUCTURE: frozenset(
        {
            LocatorFailureKind.PROVIDER_RATE_LIMIT,
            LocatorFailureKind.PROVIDER_TIMEOUT,
            LocatorFailureKind.PROVIDER_FAILURE,
            LocatorFailureKind.WORKER_EXECUTION_FAILED,
            LocatorFailureKind.RESUME_ARTIFACT_MISMATCH,
            LocatorFailureKind.ADAPTER_INPUT_INVALID,
        }
    ),
}


class UsageValueStatus(StrEnum):
    """State whether each usage value was observed, estimated, or unavailable."""

    OBSERVED = "observed"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class ModelCallRosterStatus(StrEnum):
    """State whether the roster contains every model call for one locator cell."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class CanonicalInputTokenBucket(StrEnum):
    """Name the three mutually exclusive canonical input-token buckets."""

    INPUT = "input_tokens"
    CACHE_READ = "cache_read_tokens"
    CACHE_WRITE = "cache_write_tokens"


class TokenPartitionStatus(StrEnum):
    """Separate demonstrated provider-field disjointness from unresolved mapping."""

    PROVEN_DISJOINT = "proven_disjoint"
    NOT_PROVEN = "not_proven"


class LocatorAdviceMode(StrEnum):
    """Control how much locator information may cross into a fixer cell."""

    NONE = "none"
    RANK_ONLY = "rank_only"
    RANK_WITH_PUBLIC_EXPLANATIONS = "rank_with_public_explanations"
    ABSTAIN = "abstain"


class FixActionKind(StrEnum):
    """Cover upstream probe kinds and current Terminal continuation treatments."""

    INSPECT = "inspect"
    CONTINUE = "continue"
    TARGETED_INTERVENTION = "targeted_intervention"
    REPLAY = "replay"
    REPAIR = "repair"
    CRITIQUE_THEN_REPLAN = "critique_then_replan"
    ADAPTED_TARGETED_GUIDANCE = "adapted_targeted_guidance"
    PROBE = "probe"
    CHECK = "check"


class GoldDisposition(StrEnum):
    """State whether static attribution has an independently usable oracle."""

    SCORABLE = "scorable"
    UNIDENTIFIABLE = "unidentifiable"
    NOT_APPLICABLE = "not_applicable"


class GoldLabelKind(StrEnum):
    """Distinguish one conjunctive root from alternative acceptable roots."""

    UNIQUE = "unique"
    SET_VALUED = "set_valued"


class GoldAuthority(StrEnum):
    """Identify the independent authority that supports an attribution label."""

    INDEPENDENT_HUMAN_REVIEW = "independent_human_review"
    # Blind adjudication by a model stronger than the actors under study.  It is a
    # development label, never scientific gold.
    MODEL_ADJUDICATED = "model_adjudicated"
    AUTHOR_PROVIDED = "author_provided"
    INTERVENTION_VERIFIED = "intervention_verified"
    OFFICIAL_BENCHMARK = "official_benchmark"


def _require_text(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _require_unique_texts(values: tuple[str, ...], *, label: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must contain non-empty text")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _validate_optional_limit(value: int | float | None, *, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric or null")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")


def _validate_optional_finite(value: int | float | None, *, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric or null")


def _validate_named_costs(values: tuple[tuple[str, float], ...], *, label: str) -> None:
    names = [name for name, _ in values]
    _require_unique_texts(tuple(names), label=f"{label} names")
    for _, value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} values must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} values must be finite and non-negative")


@dataclass(frozen=True)
class BenchmarkCaseRefV1:
    """Hold scorer-side case and source-task identity outside public packets."""

    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    case_id: str
    source_denominator_entry_id: str
    source_task_id: str
    task_group_id: str
    split: BenchmarkSplit
    case_ordinal: int
    source_artifact_ref: str
    population_role: PopulationRole
    exposure_status: ExposureStatus
    case_origin: CaseOrigin
    schema: str = BENCHMARK_CASE_REF_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BENCHMARK_CASE_REF_SCHEMA:
            raise ValueError(f"unsupported benchmark case reference schema: {self.schema}")
        for label, value in (
            ("benchmark_id", self.benchmark_id),
            ("benchmark_version", self.benchmark_version),
            ("adapter_version", self.adapter_version),
            ("case_id", self.case_id),
            ("source_denominator_entry_id", self.source_denominator_entry_id),
            ("source_task_id", self.source_task_id),
            ("task_group_id", self.task_group_id),
            ("source_artifact_ref", self.source_artifact_ref),
        ):
            _require_text(value, label=label)
        for label, value, enum_type in (
            ("split", self.split, BenchmarkSplit),
            ("population_role", self.population_role, PopulationRole),
            ("exposure_status", self.exposure_status, ExposureStatus),
            ("case_origin", self.case_origin, CaseOrigin),
        ):
            if type(value) is not enum_type:
                raise ValueError(f"{label} must use {enum_type.__name__}")
        if (
            isinstance(self.case_ordinal, bool)
            or not isinstance(self.case_ordinal, int)
            or self.case_ordinal < 0
        ):
            raise ValueError("case_ordinal must be a non-negative integer")
        if (
            self.population_role is PopulationRole.SCIENTIFIC_EVALUATION
            and self.split is not BenchmarkSplit.CONFIRMATORY
        ):
            raise ValueError("scientific evaluation cases must use the confirmatory split")
        if (
            self.population_role is PopulationRole.SCIENTIFIC_EVALUATION
            and self.exposure_status is not ExposureStatus.UNEXPOSED
        ):
            raise ValueError("scientific evaluation cases must be unexposed")

    def to_dict(self) -> dict[str, object]:
        """Serialize scorer identity without projecting it into method input."""

        return {
            "schema": self.schema,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "case_id": self.case_id,
            "source_denominator_entry_id": self.source_denominator_entry_id,
            "source_task_id": self.source_task_id,
            "task_group_id": self.task_group_id,
            "split": self.split.value,
            "case_ordinal": self.case_ordinal,
            "source_artifact_ref": self.source_artifact_ref,
            "population_role": self.population_role.value,
            "exposure_status": self.exposure_status.value,
            "case_origin": self.case_origin.value,
        }


@dataclass(frozen=True)
class SourceDenominatorEntryV1:
    """Record one source outcome whether or not it materializes a failed case."""

    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    population_id: str
    entry_id: str
    source_task_id: str
    task_group_id: str
    source_artifact_ref: str
    split: BenchmarkSplit
    population_role: PopulationRole
    outcome_status: SourceOutcomeStatus
    exposure_status: ExposureStatus
    protocol_complete: bool
    source_receipt_refs: tuple[str, ...]
    materialization_status: MaterializationStatus
    materialized_case_ids: tuple[str, ...] = ()
    status_detail: str | None = None
    schema: str = SOURCE_DENOMINATOR_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_DENOMINATOR_ENTRY_SCHEMA:
            raise ValueError(f"unsupported source denominator entry schema: {self.schema}")
        for label, value in (
            ("benchmark_id", self.benchmark_id),
            ("benchmark_version", self.benchmark_version),
            ("adapter_version", self.adapter_version),
            ("population_id", self.population_id),
            ("entry_id", self.entry_id),
            ("source_task_id", self.source_task_id),
            ("task_group_id", self.task_group_id),
            ("source_artifact_ref", self.source_artifact_ref),
        ):
            _require_text(value, label=label)
        if not isinstance(self.protocol_complete, bool):
            raise ValueError("protocol_complete must be boolean")
        if not self.source_receipt_refs:
            raise ValueError("source denominator entry requires source receipt references")
        _require_unique_texts(
            self.source_receipt_refs,
            label="source_receipt_refs",
        )
        _require_unique_texts(
            self.materialized_case_ids,
            label="materialized_case_ids",
        )
        if self.materialization_status is MaterializationStatus.ADMITTED:
            if not self.protocol_complete or not self.materialized_case_ids:
                raise ValueError("admitted materialization requires a complete protocol and cases")
        elif self.materialized_case_ids:
            raise ValueError("non-admitted source rows cannot reference materialized cases")
        if self.population_role is PopulationRole.SCIENTIFIC_EVALUATION:
            if self.split is not BenchmarkSplit.CONFIRMATORY:
                raise ValueError("scientific source rows must use the confirmatory split")
            if self.exposure_status is not ExposureStatus.UNEXPOSED:
                raise ValueError("scientific source rows must be unexposed")
        if self.status_detail is not None:
            _require_text(self.status_detail, label="source outcome status_detail")

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete scorer-side source denominator row."""

        return {
            "schema": self.schema,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "population_id": self.population_id,
            "entry_id": self.entry_id,
            "source_task_id": self.source_task_id,
            "task_group_id": self.task_group_id,
            "source_artifact_ref": self.source_artifact_ref,
            "split": self.split.value,
            "population_role": self.population_role.value,
            "outcome_status": self.outcome_status.value,
            "exposure_status": self.exposure_status.value,
            "protocol_complete": self.protocol_complete,
            "source_receipt_refs": list(self.source_receipt_refs),
            "materialization_status": self.materialization_status.value,
            "materialized_case_ids": list(self.materialized_case_ids),
            "status_detail": self.status_detail,
        }


@dataclass(frozen=True)
class FieldCoverageV1:
    """Expose only a public field name and its projection status to methods."""

    field_name: str
    status: FieldCoverageStatus

    def __post_init__(self) -> None:
        _require_text(self.field_name, label="field_name")

    def to_dict(self) -> dict[str, str]:
        """Return the complete method-visible coverage decision."""

        return {
            "field_name": self.field_name,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ProjectionFieldDecisionV1:
    """Keep projection evidence and rationale in the scorer-only receipt."""

    field_name: str
    status: FieldCoverageStatus
    evidence_refs: tuple[str, ...] = ()
    detail: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.field_name, label="projection field_name")
        _require_unique_texts(self.evidence_refs, label="projection evidence_refs")
        if self.detail is not None:
            _require_text(self.detail, label="projection field detail")

    def to_dict(self) -> dict[str, object]:
        """Serialize the scorer-only decision and its supporting evidence."""

        return {
            "field_name": self.field_name,
            "status": self.status.value,
            "evidence_refs": list(self.evidence_refs),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PublicTraceFieldV1:
    """Represent one named event field with an explicit primitive value type."""

    field_name: str
    value_kind: PublicFieldValueKind
    value: str | int | float | bool | tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.field_name, label="public event field_name")
        if self.field_name.strip().casefold() in _FORBIDDEN_PUBLIC_EVENT_FIELD_NAMES:
            raise ValueError(
                f"public event field is reserved for private or raw state: {self.field_name}"
            )
        value = self.value
        if self.value_kind is PublicFieldValueKind.TEXT:
            _require_text(value, label=f"public text field {self.field_name}")
        elif self.value_kind is PublicFieldValueKind.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"public integer field {self.field_name} must be an integer")
        elif self.value_kind is PublicFieldValueKind.NUMBER:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"public number field {self.field_name} must be finite")
        elif self.value_kind is PublicFieldValueKind.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"public boolean field {self.field_name} must be boolean")
        elif self.value_kind is PublicFieldValueKind.TEXT_LIST:
            if not isinstance(value, tuple):
                raise ValueError(f"public text-list field {self.field_name} must be a tuple")
            _require_unique_texts(value, label=f"public text-list field {self.field_name}")
        else:
            raise ValueError(f"unsupported public field value kind: {self.value_kind}")

    def to_dict(self) -> dict[str, object]:
        """Serialize a typed field without a free-form metadata sidecar."""

        value: object = list(self.value) if isinstance(self.value, tuple) else self.value
        return {
            "field_name": self.field_name,
            "value_kind": self.value_kind.value,
            "value": value,
        }


@dataclass(frozen=True)
class PublicTraceEventV1:
    """Expose one opaque event through a closed kind and typed public fields."""

    opaque_event_id: str
    sequence: int
    kind: PublicEventKind
    opaque_parent_event_ids: tuple[str, ...]
    typed_public_fields: tuple[PublicTraceFieldV1, ...]

    def __post_init__(self) -> None:
        _require_text(self.opaque_event_id, label="opaque_event_id")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("public event sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("public event sequence must be non-negative")
        _require_unique_texts(
            self.opaque_parent_event_ids,
            label=f"public event {self.opaque_event_id} parent ids",
        )
        field_names = tuple(field.field_name for field in self.typed_public_fields)
        _require_unique_texts(
            field_names,
            label=f"public event {self.opaque_event_id} field names",
        )

    def to_dict(self) -> dict[str, object]:
        """Construct the event payload from its fixed public schema."""

        return {
            "opaque_event_id": self.opaque_event_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "opaque_parent_event_ids": list(self.opaque_parent_event_ids),
            "typed_public_fields": [field.to_dict() for field in self.typed_public_fields],
        }


@dataclass(frozen=True)
class PublicFailureSignalV1:
    """State an observed failure without exposing a private numeric score."""

    observed_failure: bool = True
    public_feedback: str | None = None

    def __post_init__(self) -> None:
        if self.observed_failure is not True:
            raise ValueError("public failure signal must state observed_failure=true")
        if self.public_feedback is not None:
            _require_text(self.public_feedback, label="public failure feedback")

    def to_dict(self) -> dict[str, object]:
        """Serialize the bounded public failure signal."""

        return {
            "observed_failure": self.observed_failure,
            "public_feedback": self.public_feedback,
        }


@dataclass(frozen=True)
class PublicTraceProjectionReceiptV1:
    """Persist scorer-only evidence for one public trace projection."""

    projection_receipt_id: str
    case_id: str
    benchmark_id: str
    adapter_version: str
    policy_id: str
    field_decisions: tuple[ProjectionFieldDecisionV1, ...]
    source_visibility_receipt_refs: tuple[str, ...]
    rejected_private_fields: tuple[str, ...]
    opaque_roster_receipt_ref: str
    public_cutoff_event_ref: str
    schema: str = PUBLIC_TRACE_PROJECTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_TRACE_PROJECTION_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported public trace projection receipt schema: {self.schema}")
        for label, value in (
            ("projection_receipt_id", self.projection_receipt_id),
            ("case_id", self.case_id),
            ("benchmark_id", self.benchmark_id),
            ("adapter_version", self.adapter_version),
            ("policy_id", self.policy_id),
            ("opaque_roster_receipt_ref", self.opaque_roster_receipt_ref),
            ("public_cutoff_event_ref", self.public_cutoff_event_ref),
        ):
            _require_text(value, label=label)
        field_names = tuple(decision.field_name for decision in self.field_decisions)
        if not field_names:
            raise ValueError("projection receipt requires field decisions")
        _require_unique_texts(field_names, label="projection receipt field names")
        if not self.source_visibility_receipt_refs:
            raise ValueError("projection receipt requires source visibility evidence")
        _require_unique_texts(
            self.source_visibility_receipt_refs,
            label="source visibility receipt refs",
        )
        _require_unique_texts(
            self.rejected_private_fields,
            label="rejected private fields",
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize projection evidence for scorer and audit storage only."""

        return {
            "schema": self.schema,
            "projection_receipt_id": self.projection_receipt_id,
            "case_id": self.case_id,
            "benchmark_id": self.benchmark_id,
            "adapter_version": self.adapter_version,
            "policy_id": self.policy_id,
            "field_decisions": [decision.to_dict() for decision in self.field_decisions],
            "source_visibility_receipt_refs": list(self.source_visibility_receipt_refs),
            "rejected_private_fields": list(self.rejected_private_fields),
            "opaque_roster_receipt_ref": self.opaque_roster_receipt_ref,
            "public_cutoff_event_ref": self.public_cutoff_event_ref,
        }


@dataclass(frozen=True)
class PublicTracePacketV1:
    """Expose only allowlisted public trace fields under opaque identifiers."""

    case_id: str
    track: TraceFixTrack
    public_task: str
    public_initial_observation: str | None
    events: tuple[PublicTraceEventV1, ...]
    public_failure_signal: PublicFailureSignalV1
    execution_semantics: ExecutionSemantics
    field_coverage: tuple[FieldCoverageV1, ...]
    public_capabilities: tuple[str, ...] = ()
    schema: str = PUBLIC_TRACE_PACKET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_TRACE_PACKET_SCHEMA:
            raise ValueError(f"unsupported public trace packet schema: {self.schema}")
        _require_text(self.case_id, label="public case_id")
        _require_text(self.public_task, label="public_task")
        if self.public_initial_observation is not None:
            _require_text(
                self.public_initial_observation,
                label="public_initial_observation",
            )
        if not self.events:
            raise ValueError("public packet requires at least one public event")
        event_ids = tuple(event.opaque_event_id for event in self.events)
        _require_unique_texts(event_ids, label="public event ids")
        sequences = tuple(event.sequence for event in self.events)
        if tuple(sorted(sequences)) != sequences or len(sequences) != len(set(sequences)):
            raise ValueError("public event sequences must be unique and increasing")
        event_sequence_by_id = {event.opaque_event_id: event.sequence for event in self.events}
        for event in self.events:
            unknown_parent_ids = set(event.opaque_parent_event_ids) - set(event_ids)
            if unknown_parent_ids:
                raise ValueError(
                    f"public event references unknown parent ids: {sorted(unknown_parent_ids)}"
                )
            later_parent_ids = {
                parent_id
                for parent_id in event.opaque_parent_event_ids
                if event_sequence_by_id[parent_id] >= event.sequence
            }
            if later_parent_ids:
                raise ValueError(
                    f"public event parents must precede their child: {sorted(later_parent_ids)}"
                )
        coverage_names = tuple(item.field_name for item in self.field_coverage)
        if not coverage_names:
            raise ValueError("public packet requires explicit field coverage decisions")
        _require_unique_texts(coverage_names, label="field coverage names")
        _require_unique_texts(self.public_capabilities, label="public capabilities")

    @property
    def event_ids(self) -> frozenset[str]:
        """Return the opaque public event roster used for reference validation."""

        return frozenset(event.opaque_event_id for event in self.events)

    def to_dict(self) -> dict[str, object]:
        """Serialize exactly the allowlisted method-visible public packet."""

        return self.to_method_payload()

    def to_method_payload(self) -> dict[str, object]:
        """Build the method payload exclusively from explicit public fields."""

        return {
            "schema": self.schema,
            "case_id": self.case_id,
            "track": self.track.value,
            "public_task": self.public_task,
            "public_initial_observation": self.public_initial_observation,
            "events": [event.to_dict() for event in self.events],
            "public_failure_signal": self.public_failure_signal.to_dict(),
            "execution_semantics": self.execution_semantics.value,
            "field_coverage": [item.to_dict() for item in self.field_coverage],
            "public_capabilities": list(self.public_capabilities),
        }


@dataclass(frozen=True)
class LocatorBudgetV1:
    """Freeze per-work-item locator limits without hiding resource dimensions."""

    max_model_calls_total: int | None = None
    max_input_tokens_total: int | None = None
    max_output_tokens_total: int | None = None
    max_output_tokens_per_model_call: int | None = None
    max_cache_read_tokens_total: int | None = None
    max_cache_write_tokens_total: int | None = None
    max_tool_calls_total: int | None = None
    max_environment_actions_total: int | None = None
    max_probe_actions_total: int | None = None
    max_check_actions_total: int | None = None
    max_repair_actions_total: int | None = None
    max_replay_attempts_total: int | None = None
    max_wall_time_seconds_total: float | None = None
    max_monetary_cost_total: float | None = None

    def __post_init__(self) -> None:
        values = (
            ("max_model_calls_total", self.max_model_calls_total),
            ("max_input_tokens_total", self.max_input_tokens_total),
            ("max_output_tokens_total", self.max_output_tokens_total),
            (
                "max_output_tokens_per_model_call",
                self.max_output_tokens_per_model_call,
            ),
            ("max_cache_read_tokens_total", self.max_cache_read_tokens_total),
            ("max_cache_write_tokens_total", self.max_cache_write_tokens_total),
            ("max_tool_calls_total", self.max_tool_calls_total),
            ("max_environment_actions_total", self.max_environment_actions_total),
            ("max_probe_actions_total", self.max_probe_actions_total),
            ("max_check_actions_total", self.max_check_actions_total),
            ("max_repair_actions_total", self.max_repair_actions_total),
            ("max_replay_attempts_total", self.max_replay_attempts_total),
            ("max_wall_time_seconds_total", self.max_wall_time_seconds_total),
            ("max_monetary_cost_total", self.max_monetary_cost_total),
        )
        if all(value is None for _, value in values):
            raise ValueError("locator budget must declare at least one limit")
        for label, value in values:
            _validate_optional_limit(value, label=label)
        for label, value in values[:-2]:
            if value is not None and not isinstance(value, int):
                raise ValueError(f"{label} must be an integer or null")

    def to_dict(self) -> dict[str, int | float | None]:
        """Preserve each resource dimension for later paired reporting."""

        return {
            "max_model_calls_total": self.max_model_calls_total,
            "max_input_tokens_total": self.max_input_tokens_total,
            "max_output_tokens_total": self.max_output_tokens_total,
            "max_output_tokens_per_model_call": (self.max_output_tokens_per_model_call),
            "max_cache_read_tokens_total": self.max_cache_read_tokens_total,
            "max_cache_write_tokens_total": self.max_cache_write_tokens_total,
            "max_tool_calls_total": self.max_tool_calls_total,
            "max_environment_actions_total": self.max_environment_actions_total,
            "max_probe_actions_total": self.max_probe_actions_total,
            "max_check_actions_total": self.max_check_actions_total,
            "max_repair_actions_total": self.max_repair_actions_total,
            "max_replay_attempts_total": self.max_replay_attempts_total,
            "max_wall_time_seconds_total": self.max_wall_time_seconds_total,
            "max_monetary_cost_total": self.max_monetary_cost_total,
        }


@dataclass(frozen=True)
class RankedLocationV1:
    """Represent one ordered event group produced by a locator."""

    rank: int
    event_ids: tuple[str, ...]
    kind: TraceFindingKind
    label: str
    score: float | None = None
    public_explanation: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("location rank must be a positive integer")
        if not self.event_ids:
            raise ValueError("ranked location must contain at least one event")
        _require_unique_texts(self.event_ids, label="ranked location event_ids")
        _require_text(self.label, label="ranked location label")
        _validate_optional_finite(self.score, label="ranked location score")
        if self.public_explanation is not None:
            _require_text(self.public_explanation, label="public explanation")

    def to_dict(self) -> dict[str, object]:
        """Serialize normalized output while retaining optional method diagnostics."""

        return {
            "rank": self.rank,
            "event_ids": list(self.event_ids),
            "kind": self.kind.value,
            "label": self.label,
            "score": self.score,
            "public_explanation": self.public_explanation,
        }


@dataclass(frozen=True)
class LocatorModelCallUsageV1:
    """Record one call's output usage without inventing a value when absent."""

    call_id: str
    output_tokens: int | None
    output_status: UsageValueStatus
    usage_receipt_ref: str | None
    unavailable_reason: str | None = None
    unavailable_evidence_ref: str | None = None
    schema: str = LOCATOR_MODEL_CALL_USAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LOCATOR_MODEL_CALL_USAGE_SCHEMA:
            raise ValueError("unsupported locator model-call usage schema")
        _require_text(self.call_id, label="model call_id")
        if not isinstance(self.output_status, UsageValueStatus):
            raise ValueError("model-call output_status must use the closed enum")
        _validate_optional_limit(self.output_tokens, label="model-call output_tokens")
        if self.output_tokens is not None and type(self.output_tokens) is not int:
            raise ValueError("model-call output_tokens must be an integer or null")
        if self.output_status is UsageValueStatus.UNAVAILABLE:
            if self.output_tokens is not None:
                raise ValueError("unavailable model-call output must have a null value")
            _require_text(
                self.unavailable_reason,
                label="model-call output unavailable_reason",
            )
            if self.usage_receipt_ref is not None:
                _require_text(
                    self.usage_receipt_ref,
                    label="model-call usage receipt",
                )
            if self.unavailable_evidence_ref is not None:
                _require_text(
                    self.unavailable_evidence_ref,
                    label="model-call unavailable evidence",
                )
        else:
            if self.output_tokens is None:
                raise ValueError("available model-call output requires a value")
            _require_text(self.usage_receipt_ref, label="model-call usage receipt")
            if self.unavailable_reason is not None or self.unavailable_evidence_ref is not None:
                raise ValueError("available model-call output cannot carry unavailability evidence")

    def to_dict(self) -> dict[str, object]:
        """Serialize one call and the evidence quality of its output count."""

        return {
            "schema": self.schema,
            "call_id": self.call_id,
            "output_tokens": self.output_tokens,
            "output_status": self.output_status.value,
            "usage_receipt_ref": self.usage_receipt_ref,
            "unavailable_reason": self.unavailable_reason,
            "unavailable_evidence_ref": self.unavailable_evidence_ref,
        }


@dataclass(frozen=True)
class LocatorModelCallUsageReceiptV1:
    """Bind an ordered call roster to the receipt used for aggregate recomputation."""

    aggregation_receipt_ref: str
    roster_status: ModelCallRosterStatus
    calls: tuple[LocatorModelCallUsageV1, ...]
    missing_call_evidence_refs: tuple[str, ...] = ()
    schema: str = LOCATOR_MODEL_CALL_USAGE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LOCATOR_MODEL_CALL_USAGE_RECEIPT_SCHEMA:
            raise ValueError("unsupported locator model-call usage receipt schema")
        _require_text(
            self.aggregation_receipt_ref,
            label="model-call usage aggregation receipt",
        )
        if not isinstance(self.roster_status, ModelCallRosterStatus):
            raise ValueError("model-call roster_status must use the closed enum")
        if type(self.calls) is not tuple or any(
            type(call) is not LocatorModelCallUsageV1 for call in self.calls
        ):
            raise ValueError("model-call usage roster must contain typed call records")
        call_ids = tuple(call.call_id for call in self.calls)
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("model-call usage roster cannot repeat call_id")
        if type(self.missing_call_evidence_refs) is not tuple:
            raise ValueError("missing_call_evidence_refs must be a tuple")
        _require_unique_texts(
            self.missing_call_evidence_refs,
            label="missing model-call evidence refs",
        )
        if self.roster_status is ModelCallRosterStatus.COMPLETE:
            if self.missing_call_evidence_refs:
                raise ValueError("complete model-call roster cannot claim missing calls")
        elif not self.missing_call_evidence_refs:
            raise ValueError("incomplete model-call roster requires missing-call evidence")

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete or explicitly incomplete per-call roster."""

        return {
            "schema": self.schema,
            "aggregation_receipt_ref": self.aggregation_receipt_ref,
            "roster_status": self.roster_status.value,
            "calls": [call.to_dict() for call in self.calls],
            "missing_call_evidence_refs": list(self.missing_call_evidence_refs),
        }


@dataclass(frozen=True)
class ProviderTokenBucketV1:
    """Map provider fields to one canonical non-overlapping input-token bucket."""

    bucket: CanonicalInputTokenBucket
    value: int | None
    status: UsageValueStatus
    provider_field_names: tuple[str, ...]
    usage_receipt_ref: str | None
    unavailable_reason: str | None = None
    unavailable_evidence_ref: str | None = None
    schema: str = PROVIDER_TOKEN_BUCKET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROVIDER_TOKEN_BUCKET_SCHEMA:
            raise ValueError("unsupported provider token bucket schema")
        if not isinstance(self.bucket, CanonicalInputTokenBucket):
            raise ValueError("token bucket must use the closed enum")
        if type(self.provider_field_names) is not tuple:
            raise ValueError("provider_field_names must be a tuple")
        _require_unique_texts(
            self.provider_field_names,
            label=f"{self.bucket.value} provider fields",
        )
        if not isinstance(self.status, UsageValueStatus):
            raise ValueError("token bucket status must use the closed enum")
        _validate_optional_limit(self.value, label=f"{self.bucket.value} value")
        if self.value is not None and type(self.value) is not int:
            raise ValueError("token bucket value must be an integer or null")
        if self.status is UsageValueStatus.UNAVAILABLE:
            if self.value is not None:
                raise ValueError("unavailable token bucket must have a null value")
            _require_text(
                self.unavailable_reason,
                label=f"{self.bucket.value} unavailable_reason",
            )
            if self.usage_receipt_ref is not None:
                _require_text(
                    self.usage_receipt_ref,
                    label=f"{self.bucket.value} usage receipt",
                )
            if self.unavailable_evidence_ref is not None:
                _require_text(
                    self.unavailable_evidence_ref,
                    label=f"{self.bucket.value} unavailable evidence",
                )
        else:
            if self.value is None or not self.provider_field_names:
                raise ValueError("available token bucket requires a value and provider fields")
            _require_text(
                self.usage_receipt_ref,
                label=f"{self.bucket.value} usage receipt",
            )
            if self.unavailable_reason is not None or self.unavailable_evidence_ref is not None:
                raise ValueError("available token bucket cannot carry unavailability evidence")

    def to_dict(self) -> dict[str, object]:
        """Serialize one canonical bucket and its provider-field mapping."""

        return {
            "schema": self.schema,
            "bucket": self.bucket.value,
            "value": self.value,
            "status": self.status.value,
            "provider_field_names": list(self.provider_field_names),
            "usage_receipt_ref": self.usage_receipt_ref,
            "unavailable_reason": self.unavailable_reason,
            "unavailable_evidence_ref": self.unavailable_evidence_ref,
        }


@dataclass(frozen=True)
class ProviderTokenPartitionReceiptV1:
    """Make provider-to-canonical token partition evidence machine-checkable."""

    partition_receipt_ref: str
    provider_name: str
    provider_usage_receipt_ref: str
    partition_status: TokenPartitionStatus
    buckets: tuple[ProviderTokenBucketV1, ...]
    unresolved_buckets: tuple[CanonicalInputTokenBucket, ...] = ()
    schema: str = PROVIDER_TOKEN_PARTITION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROVIDER_TOKEN_PARTITION_RECEIPT_SCHEMA:
            raise ValueError("unsupported provider token partition receipt schema")
        _require_text(self.partition_receipt_ref, label="token partition receipt")
        _require_text(self.provider_name, label="token partition provider_name")
        _require_text(
            self.provider_usage_receipt_ref,
            label="provider usage receipt",
        )
        if not isinstance(self.partition_status, TokenPartitionStatus):
            raise ValueError("partition_status must use the closed enum")
        if type(self.buckets) is not tuple or any(
            type(bucket) is not ProviderTokenBucketV1 for bucket in self.buckets
        ):
            raise ValueError("token partition must contain typed bucket records")
        expected_buckets = set(CanonicalInputTokenBucket)
        actual_buckets = tuple(bucket.bucket for bucket in self.buckets)
        if set(actual_buckets) != expected_buckets or len(actual_buckets) != len(expected_buckets):
            raise ValueError("token partition requires exactly one record per canonical bucket")
        if type(self.unresolved_buckets) is not tuple or any(
            not isinstance(bucket, CanonicalInputTokenBucket) for bucket in self.unresolved_buckets
        ):
            raise ValueError("unresolved_buckets must use the closed bucket enum")
        if len(set(self.unresolved_buckets)) != len(self.unresolved_buckets):
            raise ValueError("unresolved_buckets must be unique")
        source_fields = [
            field_name for bucket in self.buckets for field_name in bucket.provider_field_names
        ]
        if len(source_fields) != len(set(source_fields)):
            raise ValueError("canonical token buckets cannot reuse a provider field")
        if self.partition_status is TokenPartitionStatus.PROVEN_DISJOINT:
            if self.unresolved_buckets:
                raise ValueError("proven token partition cannot contain unresolved buckets")
        else:
            if set(self.unresolved_buckets) != expected_buckets:
                raise ValueError(
                    "unproven token partition must mark every canonical bucket unresolved"
                )
            by_bucket = {bucket.bucket: bucket for bucket in self.buckets}
            if any(
                by_bucket[bucket].status is not UsageValueStatus.UNAVAILABLE
                for bucket in self.unresolved_buckets
            ):
                raise ValueError("unresolved token buckets must be unavailable")

    def to_dict(self) -> dict[str, object]:
        """Serialize the typed mapping and every unresolved canonical bucket."""

        return {
            "schema": self.schema,
            "partition_receipt_ref": self.partition_receipt_ref,
            "provider_name": self.provider_name,
            "provider_usage_receipt_ref": self.provider_usage_receipt_ref,
            "partition_status": self.partition_status.value,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
            "unresolved_buckets": [bucket.value for bucket in self.unresolved_buckets],
        }


@dataclass(frozen=True)
class LocatorUsageV1:
    """Record every locator cost dimension without treating missing data as zero."""

    model_calls: int | None
    input_tokens: int | None
    output_tokens: int | None
    max_observed_output_tokens_per_model_call: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    tool_calls: int | None
    environment_actions: int | None
    probe_actions: int | None
    check_actions: int | None
    repair_actions: int | None
    replay_attempts: int | None
    wall_time_seconds: float | None
    monetary_cost: float | None
    value_status: tuple[tuple[str, UsageValueStatus], ...]
    model_call_usage_receipt: LocatorModelCallUsageReceiptV1
    provider_token_partition_receipt: ProviderTokenPartitionReceiptV1

    def __post_init__(self) -> None:
        values = (
            ("model_calls", self.model_calls),
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            (
                "max_observed_output_tokens_per_model_call",
                self.max_observed_output_tokens_per_model_call,
            ),
            ("cache_read_tokens", self.cache_read_tokens),
            ("cache_write_tokens", self.cache_write_tokens),
            ("tool_calls", self.tool_calls),
            ("environment_actions", self.environment_actions),
            ("probe_actions", self.probe_actions),
            ("check_actions", self.check_actions),
            ("repair_actions", self.repair_actions),
            ("replay_attempts", self.replay_attempts),
            ("wall_time_seconds", self.wall_time_seconds),
            ("monetary_cost", self.monetary_cost),
        )
        if type(self.value_status) is not tuple:
            raise ValueError("locator usage value_status must be a tuple")
        if any(
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not UsageValueStatus
            for entry in self.value_status
        ):
            raise ValueError(
                "locator usage value_status entries must be exact "
                "tuple[str, UsageValueStatus] pairs"
            )
        status_names = tuple(name for name, _ in self.value_status)
        expected_names = tuple(name for name, _ in values)
        _require_unique_texts(status_names, label="locator usage status names")
        if status_names != expected_names:
            raise ValueError("locator usage statuses must follow canonical dimension order")
        statuses = dict(self.value_status)
        for index, (label, value) in enumerate(values):
            _validate_optional_limit(value, label=label)
            if index < 12 and value is not None and not isinstance(value, int):
                raise ValueError(f"{label} must be an integer or null")
            status = statuses[label]
            if not isinstance(status, UsageValueStatus):
                raise ValueError(f"{label} must use a closed usage value status")
            if value is None and status is not UsageValueStatus.UNAVAILABLE:
                raise ValueError(f"missing {label} must be marked unavailable")
            if value is not None and status is UsageValueStatus.UNAVAILABLE:
                raise ValueError(f"available {label} must be observed or estimated")
        per_call_status = statuses["max_observed_output_tokens_per_model_call"]
        if per_call_status is UsageValueStatus.ESTIMATED:
            raise ValueError(
                "maximum per-call output must be observed or unavailable, not estimated"
            )
        if type(self.model_call_usage_receipt) is not LocatorModelCallUsageReceiptV1:
            raise ValueError("locator usage requires a typed model-call usage receipt")
        if type(self.provider_token_partition_receipt) is not ProviderTokenPartitionReceiptV1:
            raise ValueError("locator usage requires a typed provider token partition receipt")
        self._validate_model_call_aggregation(statuses)
        self._validate_token_partition(statuses)
        if self.model_calls == 0 and any(
            value is not None and value > 0
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.cache_read_tokens,
                self.cache_write_tokens,
            )
        ):
            raise ValueError("zero model calls cannot carry positive token usage")

    def _validate_model_call_aggregation(
        self,
        statuses: dict[str, UsageValueStatus],
    ) -> None:
        """Recompute call count, total output, and maximum from the typed roster."""

        receipt = self.model_call_usage_receipt
        if receipt.roster_status is ModelCallRosterStatus.INCOMPLETE:
            for name, value in (
                ("model_calls", self.model_calls),
                ("output_tokens", self.output_tokens),
                (
                    "max_observed_output_tokens_per_model_call",
                    self.max_observed_output_tokens_per_model_call,
                ),
            ):
                if value is not None or statuses[name] is not UsageValueStatus.UNAVAILABLE:
                    raise ValueError(f"incomplete call roster requires unavailable {name}")
            return

        if (
            self.model_calls != len(receipt.calls)
            or statuses["model_calls"] is not UsageValueStatus.OBSERVED
        ):
            raise ValueError("model_calls must equal the complete typed call roster")
        call_statuses = tuple(call.output_status for call in receipt.calls)
        if any(status is UsageValueStatus.UNAVAILABLE for status in call_statuses):
            expected_total = None
            expected_total_status = UsageValueStatus.UNAVAILABLE
        else:
            expected_total = sum(
                call.output_tokens for call in receipt.calls if call.output_tokens is not None
            )
            expected_total_status = (
                UsageValueStatus.ESTIMATED
                if any(status is UsageValueStatus.ESTIMATED for status in call_statuses)
                else UsageValueStatus.OBSERVED
            )
        if (
            self.output_tokens != expected_total
            or statuses["output_tokens"] is not expected_total_status
        ):
            raise ValueError("output_tokens must be recomputed from the complete call roster")

        all_outputs_observed = bool(receipt.calls) and all(
            status is UsageValueStatus.OBSERVED for status in call_statuses
        )
        expected_max = (
            max(call.output_tokens for call in receipt.calls if call.output_tokens is not None)
            if all_outputs_observed
            else None
        )
        expected_max_status = (
            UsageValueStatus.OBSERVED if all_outputs_observed else UsageValueStatus.UNAVAILABLE
        )
        if (
            self.max_observed_output_tokens_per_model_call != expected_max
            or statuses["max_observed_output_tokens_per_model_call"] is not expected_max_status
        ):
            raise ValueError("maximum output must be derived from every observed call")
        if (
            expected_max is not None
            and expected_total is not None
            and expected_max > expected_total
        ):
            raise ValueError("maximum per-call output cannot exceed total output")

    def _validate_token_partition(
        self,
        statuses: dict[str, UsageValueStatus],
    ) -> None:
        """Match canonical input totals to the typed provider-field partition."""

        usage_by_bucket = {
            CanonicalInputTokenBucket.INPUT: (
                self.input_tokens,
                statuses["input_tokens"],
            ),
            CanonicalInputTokenBucket.CACHE_READ: (
                self.cache_read_tokens,
                statuses["cache_read_tokens"],
            ),
            CanonicalInputTokenBucket.CACHE_WRITE: (
                self.cache_write_tokens,
                statuses["cache_write_tokens"],
            ),
        }
        receipt_by_bucket = {
            bucket.bucket: bucket for bucket in self.provider_token_partition_receipt.buckets
        }
        for bucket, (value, status) in usage_by_bucket.items():
            receipt_bucket = receipt_by_bucket[bucket]
            if value != receipt_bucket.value or status is not receipt_bucket.status:
                raise ValueError(f"{bucket.value} must match the provider token partition")

    def to_dict(self) -> dict[str, object]:
        """Return raw values and their per-dimension evidence status."""

        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "max_observed_output_tokens_per_model_call": (
                self.max_observed_output_tokens_per_model_call
            ),
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "tool_calls": self.tool_calls,
            "environment_actions": self.environment_actions,
            "probe_actions": self.probe_actions,
            "check_actions": self.check_actions,
            "repair_actions": self.repair_actions,
            "replay_attempts": self.replay_attempts,
            "wall_time_seconds": self.wall_time_seconds,
            "monetary_cost": self.monetary_cost,
            "value_status": {name: status.value for name, status in self.value_status},
            "model_call_usage_receipt": self.model_call_usage_receipt.to_dict(),
            "provider_token_partition_receipt": (self.provider_token_partition_receipt.to_dict()),
        }


@dataclass(frozen=True)
class LocatorReceiptV1:
    """Bind normalized locator output and usage to the exact public request."""

    method_id: str
    status: LocatorStatus
    request: PublicTracePacketV1
    budget: LocatorBudgetV1
    predictions: tuple[RankedLocationV1, ...]
    usage: LocatorUsageV1
    failure_domain: TraceFailureDomain | None = None
    failure_kind: LocatorFailureKind | None = None
    status_detail: str | None = None
    schema: str = LOCATOR_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LOCATOR_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported locator receipt schema: {self.schema}")
        _require_text(self.method_id, label="locator method_id")
        expected_ranks = list(range(1, len(self.predictions) + 1))
        if [item.rank for item in self.predictions] != expected_ranks:
            raise ValueError("locator prediction ranks must be contiguous and ordered")
        canonical_groups = tuple(frozenset(prediction.event_ids) for prediction in self.predictions)
        if len(canonical_groups) != len(set(canonical_groups)):
            raise ValueError("locator predictions cannot repeat an event group")
        predicted_ids = {
            event_id for prediction in self.predictions for event_id in prediction.event_ids
        }
        unknown_ids = predicted_ids - self.request.event_ids
        if unknown_ids:
            raise ValueError(f"locator predictions reference unknown events: {sorted(unknown_ids)}")
        if self.status is LocatorStatus.COMPLETED and not self.predictions:
            raise ValueError("completed locator receipt must contain predictions")
        if self.status is not LocatorStatus.COMPLETED and self.predictions:
            raise ValueError("non-completed locator receipt cannot contain predictions")
        if self.failure_kind is not None and not isinstance(self.failure_kind, LocatorFailureKind):
            raise ValueError("locator failure_kind must use the closed failure vocabulary")
        if self.failure_domain is not None and not isinstance(
            self.failure_domain, TraceFailureDomain
        ):
            raise ValueError("locator failure_domain must use the closed failure vocabulary")
        failed = self.status in {LocatorStatus.BLOCKED, LocatorStatus.INVALID}
        if failed and (self.failure_domain is None or self.failure_kind is None):
            raise ValueError(
                "blocked or invalid locator receipt requires failure_domain and failure_kind"
            )
        if not failed and (self.failure_domain is not None or self.failure_kind is not None):
            raise ValueError("completed or abstained locator receipt cannot carry failure metadata")
        if failed:
            allowed_kinds = _LOCATOR_FAILURE_KINDS_BY_DOMAIN[self.failure_domain]
            if self.failure_kind not in allowed_kinds:
                raise ValueError(
                    f"locator failure_kind {self.failure_kind.value} is invalid for "
                    f"failure_domain {self.failure_domain.value}"
                )
        if self.status is not LocatorStatus.COMPLETED and self.status_detail is None:
            raise ValueError("non-completed locator receipt requires status_detail")
        if self.status_detail is not None:
            _require_text(self.status_detail, label="locator status_detail")

    def to_dict(self) -> dict[str, object]:
        """Persist enough public evidence to replay advice binding later."""

        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "status": self.status.value,
            "request": self.request.to_dict(),
            "budget": self.budget.to_dict(),
            "predictions": [item.to_dict() for item in self.predictions],
            "usage": self.usage.to_dict(),
            "failure_domain": (
                self.failure_domain.value if self.failure_domain is not None else None
            ),
            "failure_kind": (self.failure_kind.value if self.failure_kind is not None else None),
            "status_detail": self.status_detail,
        }


@dataclass(frozen=True)
class LocatorAdviceV1:
    """Carry only preregistered locator information into one fixer cell."""

    mode: LocatorAdviceMode
    source_method_id: str | None
    preregistered_top_k: int
    ranked_event_groups: tuple[tuple[str, ...], ...] = ()
    public_explanations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.preregistered_top_k, bool)
            or not isinstance(self.preregistered_top_k, int)
            or self.preregistered_top_k < 0
        ):
            raise ValueError("preregistered_top_k must be a non-negative integer")
        if self.mode is LocatorAdviceMode.NONE:
            if self.source_method_id is not None or self.ranked_event_groups:
                raise ValueError("no-locator advice cannot identify a method or carry event groups")
        else:
            if self.source_method_id is None:
                raise ValueError("locator-derived advice requires source_method_id")
            _require_text(self.source_method_id, label="source locator method_id")
        if self.mode is LocatorAdviceMode.NONE:
            if self.ranked_event_groups or self.public_explanations:
                raise ValueError("no-locator advice cannot carry locator content")
            if self.preregistered_top_k != 0:
                raise ValueError("no-locator advice must use preregistered_top_k=0")
        elif self.mode is LocatorAdviceMode.ABSTAIN:
            if self.ranked_event_groups or self.public_explanations:
                raise ValueError("abstained advice cannot carry locator content")
            if self.preregistered_top_k < 1:
                raise ValueError("abstained advice must preserve its positive top-k cutoff")
        else:
            if self.preregistered_top_k < 1:
                raise ValueError("ranked advice requires a positive preregistered_top_k")
            if not self.ranked_event_groups:
                raise ValueError("ranked advice must contain at least one event group")
            if len(self.ranked_event_groups) > self.preregistered_top_k:
                raise ValueError("ranked advice exceeds its preregistered top-k cutoff")
            for event_group in self.ranked_event_groups:
                if not event_group:
                    raise ValueError("ranked advice event groups must be non-empty")
                _require_unique_texts(event_group, label="ranked advice event group")
            canonical_groups = tuple(
                frozenset(event_group) for event_group in self.ranked_event_groups
            )
            if len(canonical_groups) != len(set(canonical_groups)):
                raise ValueError("ranked advice cannot repeat an event group")
        if self.mode is LocatorAdviceMode.RANK_WITH_PUBLIC_EXPLANATIONS:
            if len(self.public_explanations) != len(self.ranked_event_groups):
                raise ValueError("public explanations must align one-to-one with event groups")
            for explanation in self.public_explanations:
                _require_text(explanation, label="public explanation")
        elif self.public_explanations:
            raise ValueError("only explanation-enabled advice may carry explanations")

    def to_dict(self) -> dict[str, object]:
        """Serialize runner/scorer advice identity and its controlled content."""

        return {
            "mode": self.mode.value,
            "source_method_id": self.source_method_id,
            "preregistered_top_k": self.preregistered_top_k,
            "ranked_event_groups": [list(group) for group in self.ranked_event_groups],
            "public_explanations": list(self.public_explanations),
        }

    def to_method_payload(self) -> dict[str, object]:
        """Hide locator identity so a fixer cannot branch on the source method."""

        return {
            "mode": self.mode.value,
            "preregistered_top_k": self.preregistered_top_k,
            "ranked_event_groups": [list(group) for group in self.ranked_event_groups],
            "public_explanations": list(self.public_explanations),
        }


@dataclass(frozen=True)
class FixActionV1:
    """Describe a benchmark-owned action that a fixer may select by id."""

    action_id: str
    kind: FixActionKind
    public_description: str
    target_event_ids: tuple[str, ...] = ()
    estimated_cost: tuple[tuple[str, float], ...] = ()
    repeatable: bool = False

    def __post_init__(self) -> None:
        _require_text(self.action_id, label="fix action_id")
        _require_text(self.public_description, label="fix action public_description")
        _require_unique_texts(self.target_event_ids, label="fix action target_event_ids")
        _validate_named_costs(self.estimated_cost, label="fix action estimated_cost")
        if not isinstance(self.repeatable, bool):
            raise ValueError("fix action repeatable must be boolean")

    def to_dict(self) -> dict[str, object]:
        """Return the public catalog entry without an executable command or patch."""

        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "public_description": self.public_description,
            "target_event_ids": list(self.target_event_ids),
            "estimated_cost": {name: value for name, value in self.estimated_cost},
            "repeatable": self.repeatable,
        }


@dataclass(frozen=True)
class RecoveryStateV1:
    """Describe the public terminal state and frozen actor cell for recovery."""

    execution_semantics: ExecutionSemantics
    public_terminal_state_ref: str
    actor_identity: tuple[tuple[str, str], ...]
    field_coverage: tuple[FieldCoverageV1, ...]
    action_catalog: tuple[FixActionV1, ...]
    public_verifier_id: str
    runtime_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.execution_semantics is ExecutionSemantics.FROZEN_TRACE:
            raise ValueError("forward recovery requires mutable execution semantics")
        _require_text(self.public_terminal_state_ref, label="public_terminal_state_ref")
        _require_text(self.public_verifier_id, label="public_verifier_id")
        actor_keys = tuple(key for key, _ in self.actor_identity)
        _require_unique_texts(actor_keys, label="actor identity keys")
        for key, value in self.actor_identity:
            _require_text(value, label=f"actor identity value for {key}")
        required_actor_keys = {
            "provider",
            "model",
            "transport",
            "thinking",
            "output_budget",
            "actor_contract_id",
            "system_prompt_id",
            "tool_schema_id",
            "adapter_runtime_id",
            "decoding_config_id",
        }
        missing_actor_keys = required_actor_keys - set(actor_keys)
        if missing_actor_keys:
            raise ValueError(
                f"actor identity is missing frozen fields: {sorted(missing_actor_keys)}"
            )
        coverage_names = tuple(item.field_name for item in self.field_coverage)
        if not coverage_names:
            raise ValueError("recovery state requires explicit field coverage decisions")
        _require_unique_texts(coverage_names, label="recovery field coverage names")
        action_ids = tuple(item.action_id for item in self.action_catalog)
        if not action_ids:
            raise ValueError("recovery state must provide at least one benchmark action")
        _require_unique_texts(action_ids, label="recovery action ids")
        _require_unique_texts(self.runtime_capabilities, label="runtime capabilities")

    def to_dict(self) -> dict[str, object]:
        """Serialize runner/scorer state, including the frozen actor identity."""

        return {
            "execution_semantics": self.execution_semantics.value,
            "public_terminal_state_ref": self.public_terminal_state_ref,
            "actor_identity": {key: value for key, value in self.actor_identity},
            "field_coverage": [item.to_dict() for item in self.field_coverage],
            "action_catalog": [item.to_dict() for item in self.action_catalog],
            "public_verifier_id": self.public_verifier_id,
            "runtime_capabilities": list(self.runtime_capabilities),
        }

    def to_method_payload(self) -> dict[str, object]:
        """Expose recovery state while hiding the actor identity from fixers."""

        return {
            "execution_semantics": self.execution_semantics.value,
            "public_terminal_state_ref": self.public_terminal_state_ref,
            "field_coverage": [item.to_dict() for item in self.field_coverage],
            "action_catalog": [item.to_dict() for item in self.action_catalog],
            "public_verifier_id": self.public_verifier_id,
            "runtime_capabilities": list(self.runtime_capabilities),
        }


@dataclass(frozen=True)
class FixerRequestV1:
    """Bind one standardized locator advice object to one isolated recovery cell."""

    case_id: str
    system_cell_id: str
    public_trace: PublicTracePacketV1
    recovery_state: RecoveryStateV1
    locator_advice: LocatorAdviceV1
    budget: tuple[tuple[str, float], ...]
    schema: str = FIXER_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FIXER_REQUEST_SCHEMA:
            raise ValueError(f"unsupported fixer request schema: {self.schema}")
        _require_text(self.case_id, label="fixer request case_id")
        _require_text(self.system_cell_id, label="fixer request system_cell_id")
        if self.public_trace.case_id != self.case_id:
            raise ValueError("fixer request and public trace case_id must match")
        if self.public_trace.track is not TraceFixTrack.FORWARD_RECOVERY:
            raise ValueError("fixer request requires a forward_recovery public trace")
        if self.public_trace.execution_semantics is not self.recovery_state.execution_semantics:
            raise ValueError("public trace and recovery state execution semantics must match")
        if not self.budget:
            raise ValueError("fixer request must declare a resource budget")
        _validate_named_costs(self.budget, label="fixer request budget")
        known_event_ids = self.public_trace.event_ids
        advice_ids = {
            event_id
            for event_group in self.locator_advice.ranked_event_groups
            for event_id in event_group
        }
        action_target_ids = {
            event_id
            for action in self.recovery_state.action_catalog
            for event_id in action.target_event_ids
        }
        unknown_ids = (advice_ids | action_target_ids) - known_event_ids
        if unknown_ids:
            raise ValueError(
                f"fixer request references unknown trace events: {sorted(unknown_ids)}"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize runner/scorer bindings, including cell and locator identity."""

        return {
            "schema": self.schema,
            "case_id": self.case_id,
            "system_cell_id": self.system_cell_id,
            "public_trace": self.public_trace.to_dict(),
            "recovery_state": self.recovery_state.to_dict(),
            "locator_advice": self.locator_advice.to_dict(),
            "budget": {name: value for name, value in self.budget},
        }

    def to_method_payload(self) -> dict[str, object]:
        """Hide cell and locator identity from the bounded fixer request."""

        return {
            "schema": self.schema,
            "case_id": self.case_id,
            "public_trace": self.public_trace.to_method_payload(),
            "recovery_state": self.recovery_state.to_method_payload(),
            "locator_advice": self.locator_advice.to_method_payload(),
            "budget": {name: value for name, value in self.budget},
        }


@dataclass(frozen=True)
class AttributionGoldV1:
    """Store scorer-only event attribution gold with its independent authority."""

    case_ref: BenchmarkCaseRefV1
    disposition: GoldDisposition
    label_kind: GoldLabelKind | None
    accepted_event_groups: tuple[tuple[str, ...], ...]
    authority: GoldAuthority
    authority_receipt_refs: tuple[str, ...]
    claim_scope: str
    causal_distance: int | None = None
    distance_metric_id: str | None = None
    reviewers_saw_method_outputs: bool = False
    schema: str = ATTRIBUTION_GOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ATTRIBUTION_GOLD_SCHEMA:
            raise ValueError(f"unsupported attribution gold schema: {self.schema}")
        _require_text(self.claim_scope, label="attribution gold claim_scope")
        if not self.authority_receipt_refs:
            raise ValueError("attribution gold requires at least one authority receipt")
        _require_unique_texts(
            self.authority_receipt_refs,
            label="attribution gold authority_receipt_refs",
        )
        if not isinstance(self.reviewers_saw_method_outputs, bool):
            raise ValueError("reviewers_saw_method_outputs must be boolean")
        if self.reviewers_saw_method_outputs and not (
            self.case_ref.population_role is PopulationRole.DEVELOPMENT
            and self.case_ref.exposure_status is ExposureStatus.EXPOSED
        ):
            raise ValueError(
                "method-visible gold review is limited to exposed development evidence"
            )
        if self.causal_distance is not None:
            if (
                isinstance(self.causal_distance, bool)
                or not isinstance(self.causal_distance, int)
                or self.causal_distance < 0
            ):
                raise ValueError("causal_distance must be a non-negative integer or null")
        if (self.causal_distance is None) != (self.distance_metric_id is None):
            raise ValueError(
                "causal_distance and distance_metric_id must be present or absent together"
            )
        if self.distance_metric_id is not None:
            _require_text(self.distance_metric_id, label="distance_metric_id")
        if self.label_kind is not None and not isinstance(self.label_kind, GoldLabelKind):
            raise ValueError("label_kind must use the closed attribution label vocabulary")
        if self.disposition is GoldDisposition.SCORABLE:
            if self.label_kind is None or not self.accepted_event_groups:
                raise ValueError("scorable gold requires a label kind and accepted groups")
            for group in self.accepted_event_groups:
                if not group:
                    raise ValueError("accepted event groups must be non-empty")
                _require_unique_texts(group, label="attribution gold event group")
            canonical_groups = tuple(frozenset(group) for group in self.accepted_event_groups)
            if len(canonical_groups) != len(set(canonical_groups)):
                raise ValueError("accepted event groups must be distinct")
            if self.label_kind is GoldLabelKind.UNIQUE and len(self.accepted_event_groups) != 1:
                raise ValueError("unique attribution gold requires exactly one event group")
            if self.label_kind is GoldLabelKind.SET_VALUED and len(self.accepted_event_groups) < 2:
                raise ValueError(
                    "set-valued attribution gold requires at least two alternative groups"
                )
        elif (
            self.label_kind is not None
            or self.accepted_event_groups
            or self.causal_distance is not None
            or self.distance_metric_id is not None
        ):
            raise ValueError("non-scorable attribution gold cannot carry labels or causal distance")

    def to_dict(self) -> dict[str, object]:
        """Serialize the private oracle for a scorer-only loader."""

        return {
            "schema": self.schema,
            "case_ref": self.case_ref.to_dict(),
            "disposition": self.disposition.value,
            "label_kind": self.label_kind.value if self.label_kind is not None else None,
            "accepted_event_groups": [list(group) for group in self.accepted_event_groups],
            "authority": self.authority.value,
            "authority_receipt_refs": list(self.authority_receipt_refs),
            "claim_scope": self.claim_scope,
            "causal_distance": self.causal_distance,
            "distance_metric_id": self.distance_metric_id,
            "reviewers_saw_method_outputs": self.reviewers_saw_method_outputs,
        }


@dataclass(frozen=True)
class CaseAdmissionV1:
    """Store scorer-side track admission and the private source-task reference."""

    case_ref: BenchmarkCaseRefV1
    eligible_tracks: tuple[TraceFixTrack, ...]
    scorer_fields_withheld_from_methods: bool
    attribution_gold_disposition: GoldDisposition | None
    checkpoint_parity_qualified: bool
    matched_three_arm_qualified: bool
    state_coverage_qualified: bool
    same_actor_qualified: bool
    official_verifier_qualified: bool
    admission_receipt_refs: tuple[str, ...]
    exclusion_reasons: tuple[str, ...] = ()
    schema: str = CASE_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASE_ADMISSION_SCHEMA:
            raise ValueError(f"unsupported case admission schema: {self.schema}")
        if len(self.eligible_tracks) != len(set(self.eligible_tracks)):
            raise ValueError("eligible_tracks must be unique")
        _require_unique_texts(
            self.admission_receipt_refs,
            label="admission_receipt_refs",
        )
        _require_unique_texts(self.exclusion_reasons, label="exclusion_reasons")
        boolean_fields = (
            self.scorer_fields_withheld_from_methods,
            self.checkpoint_parity_qualified,
            self.matched_three_arm_qualified,
            self.state_coverage_qualified,
            self.same_actor_qualified,
            self.official_verifier_qualified,
        )
        if not all(isinstance(value, bool) for value in boolean_fields):
            raise ValueError("case admission qualification fields must be boolean")
        if self.eligible_tracks and not self.admission_receipt_refs:
            raise ValueError("eligible cases require admission receipts")
        if not self.eligible_tracks and not self.exclusion_reasons:
            raise ValueError("ineligible cases require an exclusion reason")
        if self.eligible_tracks and not self.scorer_fields_withheld_from_methods:
            raise ValueError("eligible cases must withhold scorer-only fields from methods")
        static_eligible = TraceFixTrack.STATIC_ATTRIBUTION in self.eligible_tracks
        forward_eligible = TraceFixTrack.FORWARD_RECOVERY in self.eligible_tracks
        if static_eligible and self.attribution_gold_disposition is None:
            raise ValueError("static attribution eligibility requires a gold disposition")
        if (
            not static_eligible
            and not forward_eligible
            and self.attribution_gold_disposition is not None
        ):
            raise ValueError("historical-only admission cannot assign attribution gold disposition")
        if (
            self.case_ref.split is BenchmarkSplit.CONFIRMATORY
            and self.eligible_tracks
            and self.case_ref.exposure_status is not ExposureStatus.UNEXPOSED
        ):
            raise ValueError("eligible confirmatory cases must be unexposed")
        if TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION in self.eligible_tracks:
            if not (
                self.checkpoint_parity_qualified
                and self.matched_three_arm_qualified
                and self.official_verifier_qualified
            ):
                raise ValueError(
                    "historical causal eligibility requires parity, matched arms, and a verifier"
                )
        if TraceFixTrack.FORWARD_RECOVERY in self.eligible_tracks:
            if not (
                self.state_coverage_qualified
                and self.same_actor_qualified
                and self.official_verifier_qualified
            ):
                raise ValueError(
                    "forward recovery eligibility requires state coverage, same actor, and verifier"
                )

    def to_dict(self) -> dict[str, object]:
        """Serialize admission evidence separately from public method payloads."""

        return {
            "schema": self.schema,
            "case_ref": self.case_ref.to_dict(),
            "eligible_tracks": [track.value for track in self.eligible_tracks],
            "scorer_fields_withheld_from_methods": self.scorer_fields_withheld_from_methods,
            "attribution_gold_disposition": (
                self.attribution_gold_disposition.value
                if self.attribution_gold_disposition is not None
                else None
            ),
            "checkpoint_parity_qualified": self.checkpoint_parity_qualified,
            "matched_three_arm_qualified": self.matched_three_arm_qualified,
            "state_coverage_qualified": self.state_coverage_qualified,
            "same_actor_qualified": self.same_actor_qualified,
            "official_verifier_qualified": self.official_verifier_qualified,
            "admission_receipt_refs": list(self.admission_receipt_refs),
            "exclusion_reasons": list(self.exclusion_reasons),
        }


__all__ = [
    "ATTRIBUTION_GOLD_SCHEMA",
    "BENCHMARK_CASE_REF_SCHEMA",
    "SOURCE_DENOMINATOR_ENTRY_SCHEMA",
    "FIXER_REQUEST_SCHEMA",
    "LOCATOR_RECEIPT_SCHEMA",
    "LOCATOR_MODEL_CALL_USAGE_RECEIPT_SCHEMA",
    "LOCATOR_MODEL_CALL_USAGE_SCHEMA",
    "CASE_ADMISSION_SCHEMA",
    "PROVIDER_TOKEN_BUCKET_SCHEMA",
    "PROVIDER_TOKEN_PARTITION_RECEIPT_SCHEMA",
    "PUBLIC_TRACE_PACKET_SCHEMA",
    "PUBLIC_TRACE_PROJECTION_RECEIPT_SCHEMA",
    "TRACEFIX_BENCH_IMPLEMENTATION_STATUS",
    "AttributionGoldV1",
    "BenchmarkCaseRefV1",
    "BenchmarkSplit",
    "CaseOrigin",
    "CanonicalInputTokenBucket",
    "ExecutionSemantics",
    "ExposureStatus",
    "FieldCoverageStatus",
    "FieldCoverageV1",
    "FixActionKind",
    "FixActionV1",
    "FixerRequestV1",
    "GoldAuthority",
    "GoldDisposition",
    "GoldLabelKind",
    "LocatorAdviceMode",
    "LocatorAdviceV1",
    "LocatorBudgetV1",
    "LocatorFailureKind",
    "LocatorModelCallUsageReceiptV1",
    "LocatorModelCallUsageV1",
    "LocatorReceiptV1",
    "LocatorStatus",
    "LocatorUsageV1",
    "MaterializationStatus",
    "ModelCallRosterStatus",
    "PopulationRole",
    "CaseAdmissionV1",
    "ProjectionFieldDecisionV1",
    "ProviderTokenBucketV1",
    "ProviderTokenPartitionReceiptV1",
    "PublicEventKind",
    "PublicFailureSignalV1",
    "PublicFieldValueKind",
    "PublicTraceEventV1",
    "PublicTracePacketV1",
    "PublicTraceFieldV1",
    "PublicTraceProjectionReceiptV1",
    "RankedLocationV1",
    "RecoveryStateV1",
    "SourceDenominatorEntryV1",
    "SourceOutcomeStatus",
    "TraceFailureDomain",
    "TraceFixTrack",
    "TokenPartitionStatus",
    "UsageValueStatus",
]
