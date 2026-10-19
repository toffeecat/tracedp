"""Pure scorer registry contracts for LongHorizon-TraceFix-Bench v0.1.

These locally designed records freeze scorer selection and scoring policy before
method output exists. They do not execute scorers or promote implementation status.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracefix_bench.contracts import ExecutionSemantics, TraceFixTrack
from lh_platform.tracefix_bench.dataset_contracts import TRACEFIX_BENCH_SUITE_ID
from lh_platform.tracing.unified import TraceFindingKind

SCORER_REGISTRY_SCHEMA = "longhorizon_tracefix_bench/scorer_registry/v1"
SCORER_REGISTRY_FILENAME = "scorer_registry.json"
SCORER_REGISTRATION_SCHEMA = "longhorizon_tracefix_bench/scorer_registration/v1"
PROGRESS_METRIC_CONTRACT_SCHEMA = "longhorizon_tracefix_bench/progress_metric_contract/v1"
VERIFIER_CONTRACT_SCHEMA = "longhorizon_tracefix_bench/verifier_contract/v1"
ORACLE_CONTRACT_SCHEMA = "longhorizon_tracefix_bench/oracle_contract/v1"
RESOURCE_BUDGET_CONTRACT_SCHEMA = "longhorizon_tracefix_bench/resource_budget_contract/v1"
AGGREGATE_PLAN_SCHEMA = "longhorizon_tracefix_bench/aggregate_plan/v1"
COMPARISON_FAMILY_SCHEMA = "longhorizon_tracefix_bench/comparison_family/v1"
TREATMENT_ARM_REGISTRATION_SCHEMA = (
    "longhorizon_tracefix_bench/treatment_arm_registration/v1"
)
SYSTEM_CELL_REGISTRATION_SCHEMA = (
    "longhorizon_tracefix_bench/system_cell_registration/v1"
)
SCORER_REGISTRY_IMPLEMENTATION_STATUS = "implementation_unverified"
SCORER_REGISTRY_SUITE_VERSION_V0_1 = "0.1.0-dev.1"
STATIC_HIT_CUTOFFS_V0_1 = (1, 3)
STATIC_PRIMARY_K_V0_1 = 3
STATIC_ALLOWED_FINDING_KINDS_V0_1 = (
    TraceFindingKind.STEP_ATTRIBUTION,
    TraceFindingKind.ROOT_CAUSE,
)
STATIC_NO_FIXER_TREATMENT_ID = "static_no_fixer"

_IDENTIFIER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


def _identifier(value: str, label: str) -> None:
    """Keep registry identities portable and unchanged by serialization."""

    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable non-empty identifier")


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _unique_identifiers(values: tuple[str, ...], label: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be a tuple")
    for value in values:
        _identifier(value, f"{label} item")
    if len({value.casefold() for value in values}) != len(values):
        raise ValueError(f"{label} must be unique after case folding")


def _unique_texts(values: tuple[str, ...], label: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be a tuple")
    for value in values:
        _text(value, f"{label} item")
    if len({value.casefold() for value in values}) != len(values):
        raise ValueError(f"{label} must be unique after case folding")


def _records(values: tuple[object, ...], kind: type, label: str) -> None:
    if not isinstance(values, tuple) or any(not isinstance(value, kind) for value in values):
        raise ValueError(f"{label} must be a tuple of {kind.__name__}")


def _unique_record_ids(values: tuple[object, ...], attribute: str, label: str) -> None:
    identities = tuple(getattr(value, attribute) for value in values)
    _unique_identifiers(identities, label)


def _unique_tracks(values: tuple[TraceFixTrack, ...], label: str) -> None:
    """Require a nonempty, duplicate-free tuple from the closed track enum."""

    if (
        type(values) is not tuple
        or not values
        or any(not isinstance(value, TraceFixTrack) for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{label} must be nonempty unique TraceFixTrack values")


def _finite(value: float, label: str, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")


def _benchmark_domain(contract: object) -> tuple[str, str, str]:
    """Compare a metric or oracle with the official verifier it depends on."""

    return (
        getattr(contract, "benchmark_id").casefold(),
        getattr(contract, "benchmark_version").casefold(),
        getattr(contract, "adapter_version").casefold(),
    )


def _exact_benchmark_domain(contract: object) -> tuple[str, str, str]:
    """Preserve registered spelling when closing runtime-facing domain IDs."""

    return (
        getattr(contract, "benchmark_id"),
        getattr(contract, "benchmark_version"),
        getattr(contract, "adapter_version"),
    )


class BuiltinScorerKey(StrEnum):
    """Select only built-in scorer implementations; JSON cannot name import paths."""

    STATIC_ATTRIBUTION_V1 = "static_attribution_v1"
    HISTORICAL_CAUSAL_VERIFICATION_V1 = "historical_causal_verification_v1"
    FORWARD_RECOVERY_V1 = "forward_recovery_v1"


_KEY_BY_TRACK = {
    TraceFixTrack.STATIC_ATTRIBUTION: BuiltinScorerKey.STATIC_ATTRIBUTION_V1,
    TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION: BuiltinScorerKey.HISTORICAL_CAUSAL_VERIFICATION_V1,
    TraceFixTrack.FORWARD_RECOVERY: BuiltinScorerKey.FORWARD_RECOVERY_V1,
}


class ProgressDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class ProgressUnavailablePolicy(StrEnum):
    """Keep missing progress outside numeric denominators instead of imputing zero."""

    NULL_EXCLUDED_FROM_PROGRESS_DENOMINATOR = (
        "null_excluded_from_progress_denominator"
    )


class EvidenceSource(StrEnum):
    OFFICIAL_BENCHMARK = "official_benchmark"
    INDEPENDENT_ORACLE = "independent_oracle"


class OracleKind(StrEnum):
    FALSE_REPAIR = "false_repair"
    TRANSIENT_REGRESSION = "transient_regression"
    FINAL_REGRESSION = "final_regression"
    FORWARD_ROOT_VERIFICATION = "forward_root_verification"


class BudgetScope(StrEnum):
    PER_MODEL_CALL = "per_model_call"
    PER_ACTION = "per_action"
    PER_WORK_ITEM = "per_work_item"
    WHOLE_RUN = "whole_run"


class ResourceRole(StrEnum):
    LOCATOR = "locator"
    FIXER = "fixer"
    ACTOR = "actor"
    ENVIRONMENT = "environment"
    RESTORE = "restore"
    PROBE = "probe"
    CHECK = "check"
    REPAIR = "repair"
    REPLAY = "replay"
    OFFICIAL_VERIFIER = "official_verifier"
    SCORER = "scorer"
    RECOVERY = "recovery"
    PERSISTENCE = "persistence"
    ORCHESTRATION = "orchestration"


class ResourceDimension(StrEnum):
    MODEL_CALLS = "model_calls"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    CACHE_READ_TOKENS = "cache_read_tokens"
    CACHE_WRITE_TOKENS = "cache_write_tokens"
    TOOL_CALLS = "tool_calls"
    ENVIRONMENT_ACTIONS = "environment_actions"
    PROBE_ACTIONS = "probe_actions"
    CHECK_ACTIONS = "check_actions"
    REPAIR_ACTIONS = "repair_actions"
    REPLAY_ATTEMPTS = "replay_attempts"
    WALL_TIME_SECONDS = "wall_time_seconds"
    MONETARY_COST = "monetary_cost"


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


class MultiplicityCorrection(StrEnum):
    HOLM = "holm"


class BuiltinEndpointKey(StrEnum):
    """Keep preregistered primary endpoints tied to the track that defines them."""

    STATIC_HIT_AT_1 = "static_hit_at_1"
    STATIC_HIT_AT_3 = "static_hit_at_3"
    STATIC_FIXED_DENOMINATOR_MRR = "static_fixed_denominator_mrr"
    HISTORICAL_VERIFIED_ROOT_AT_B = "historical_verified_root_at_b"
    FORWARD_FAILURE_RESOLVED_AT_B = "forward_failure_resolved_at_b"


_ENDPOINTS_BY_TRACK = {
    TraceFixTrack.STATIC_ATTRIBUTION: frozenset(
        {
            BuiltinEndpointKey.STATIC_HIT_AT_1,
            BuiltinEndpointKey.STATIC_HIT_AT_3,
            BuiltinEndpointKey.STATIC_FIXED_DENOMINATOR_MRR,
        }
    ),
    TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION: frozenset(
        {BuiltinEndpointKey.HISTORICAL_VERIFIED_ROOT_AT_B}
    ),
    TraceFixTrack.FORWARD_RECOVERY: frozenset(
        {BuiltinEndpointKey.FORWARD_FAILURE_RESOLVED_AT_B}
    ),
}


class BinaryReductionPolicy(StrEnum):
    UNAVAILABLE = "unavailable"
    ONE_ENDPOINT_PER_SOURCE_TASK = "one_endpoint_per_source_task"
    PREREGISTERED_SOURCE_TASK_REDUCER = "preregistered_source_task_reducer"


@dataclass(frozen=True)
class ProgressMetricContractV1:
    """Freeze the source, direction, range, and meaningful difference for progress."""

    progress_contract_id: str
    metric_id: str
    metric_version: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    evaluator_contract_id: str
    value_field: str
    unit: str
    direction: ProgressDirection
    domain_min: float
    domain_max: float
    timepoints: tuple[str, ...]
    tolerance: float
    minimum_meaningful_difference: float
    aggregation_rule_id: str
    aggregation_rule_source_ref: str
    unavailable_value_policy: ProgressUnavailablePolicy
    evidence_source: EvidenceSource
    source_ref: str
    schema: str = PROGRESS_METRIC_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROGRESS_METRIC_CONTRACT_SCHEMA:
            raise ValueError("unsupported progress metric contract schema")
        for name in (
            "progress_contract_id",
            "metric_id",
            "metric_version",
            "benchmark_id",
            "benchmark_version",
            "adapter_version",
            "evaluator_contract_id",
            "value_field",
            "aggregation_rule_id",
        ):
            _identifier(getattr(self, name), name)
        _text(self.unit, "unit")
        _text(self.source_ref, "source_ref")
        _text(self.aggregation_rule_source_ref, "aggregation_rule_source_ref")
        if (
            not isinstance(self.direction, ProgressDirection)
            or not isinstance(self.evidence_source, EvidenceSource)
            or not isinstance(
                self.unavailable_value_policy,
                ProgressUnavailablePolicy,
            )
        ):
            raise ValueError(
                "progress direction, evidence source, and unavailable policy "
                "must use closed enums"
            )
        if self.evidence_source is not EvidenceSource.OFFICIAL_BENCHMARK:
            raise ValueError(
                "v0.1 progress must come from the registered official benchmark evaluator"
            )
        _finite(self.domain_min, "domain_min")
        _finite(self.domain_max, "domain_max")
        if self.domain_min >= self.domain_max:
            raise ValueError("progress domain_min must be less than domain_max")
        _unique_identifiers(self.timepoints, "timepoints")
        if not self.timepoints:
            raise ValueError("progress requires at least one timepoint")
        _finite(self.tolerance, "tolerance", minimum=0)
        _finite(self.minimum_meaningful_difference, "minimum_meaningful_difference", minimum=0)
        if self.minimum_meaningful_difference <= self.tolerance:
            raise ValueError("meaningful difference must exceed measurement tolerance")

    def to_dict(self) -> dict[str, object]:
        """Serialize one closed progress definition without measured outcomes."""

        return {
            "schema": self.schema,
            "progress_contract_id": self.progress_contract_id,
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "evaluator_contract_id": self.evaluator_contract_id,
            "value_field": self.value_field,
            "unit": self.unit,
            "direction": self.direction.value,
            "domain_min": self.domain_min,
            "domain_max": self.domain_max,
            "timepoints": list(self.timepoints),
            "tolerance": self.tolerance,
            "minimum_meaningful_difference": self.minimum_meaningful_difference,
            "aggregation_rule_id": self.aggregation_rule_id,
            "aggregation_rule_source_ref": self.aggregation_rule_source_ref,
            "unavailable_value_policy": self.unavailable_value_policy.value,
            "evidence_source": self.evidence_source.value,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class VerifierContractV1:
    """Bind official final success to one benchmark and adapter version."""

    verifier_contract_id: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    official_verifier_id: str
    official_verifier_version: str
    final_success_field: str
    source_ref: str
    schema: str = VERIFIER_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VERIFIER_CONTRACT_SCHEMA:
            raise ValueError("unsupported verifier contract schema")
        for name in (
            "verifier_contract_id", "benchmark_id", "benchmark_version", "adapter_version",
            "official_verifier_id", "official_verifier_version", "final_success_field",
        ):
            _identifier(getattr(self, name), name)
        _text(self.source_ref, "source_ref")

    def to_dict(self) -> dict[str, str]:
        """Serialize an identity contract, not an executable verifier."""

        return {
            "schema": self.schema,
            "verifier_contract_id": self.verifier_contract_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "official_verifier_id": self.official_verifier_id,
            "official_verifier_version": self.official_verifier_version,
            "final_success_field": self.final_success_field,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class OracleContractV1:
    """Require an independent, versioned source for optional forward outcomes."""

    oracle_contract_id: str
    kind: OracleKind
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    oracle_version: str
    outcome_field: str
    source_ref: str
    independent_of_method: bool = True
    schema: str = ORACLE_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ORACLE_CONTRACT_SCHEMA:
            raise ValueError("unsupported oracle contract schema")
        if not isinstance(self.kind, OracleKind):
            raise ValueError("oracle kind must use the closed enum")
        for name in (
            "oracle_contract_id", "benchmark_id", "benchmark_version", "adapter_version",
            "oracle_version", "outcome_field",
        ):
            _identifier(getattr(self, name), name)
        _text(self.source_ref, "source_ref")
        if self.independent_of_method is not True:
            raise ValueError("forward oracle must be independent of method output")

    def to_dict(self) -> dict[str, object]:
        """Serialize oracle availability prerequisites without claiming an outcome."""

        return {
            "schema": self.schema,
            "oracle_contract_id": self.oracle_contract_id,
            "kind": self.kind.value,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "oracle_version": self.oracle_version,
            "outcome_field": self.outcome_field,
            "source_ref": self.source_ref,
            "independent_of_method": self.independent_of_method,
        }


@dataclass(frozen=True)
class ResourceLimitV1:
    """Freeze one dimension of a budget vector with role and counting scope."""

    role: ResourceRole
    scope: BudgetScope
    dimension: ResourceDimension
    max_value: int | float

    def __post_init__(self) -> None:
        if not isinstance(self.role, ResourceRole) or not isinstance(self.scope, BudgetScope):
            raise ValueError("resource limit role and scope must use closed enums")
        if not isinstance(self.dimension, ResourceDimension):
            raise ValueError("resource limit dimension must use the closed enum")
        _finite(self.max_value, "max_value", minimum=0)
        if self.dimension in _COUNT_DIMENSIONS and type(self.max_value) is not int:
            raise ValueError("count and token limits must be integers")
        if (
            self.scope is BudgetScope.PER_MODEL_CALL
            and self.dimension is not ResourceDimension.OUTPUT_TOKENS
        ):
            raise ValueError("per-model-call scope is reserved for model output tokens")

    def to_dict(self) -> dict[str, object]:
        """Serialize one resource dimension without implying actual usage."""

        return {
            "role": self.role.value,
            "scope": self.scope.value,
            "dimension": self.dimension.value,
            "max_value": self.max_value,
        }


@dataclass(frozen=True)
class ResourceBudgetContractV1:
    """Give one stable budget identity to a complete, nonempty resource vector."""

    budget_id: str
    limits: tuple[ResourceLimitV1, ...]
    schema: str = RESOURCE_BUDGET_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESOURCE_BUDGET_CONTRACT_SCHEMA:
            raise ValueError("unsupported resource budget contract schema")
        _identifier(self.budget_id, "budget_id")
        _records(self.limits, ResourceLimitV1, "limits")
        if not self.limits:
            raise ValueError("resource budget requires at least one limit")
        if any(limit.scope is BudgetScope.WHOLE_RUN for limit in self.limits):
            raise ValueError("whole-run limits require a separate run budget contract")
        keys = tuple(
            (limit.role, limit.scope, limit.dimension) for limit in self.limits
        )
        if len(set(keys)) != len(keys):
            raise ValueError("budget vector cannot repeat role, scope, and dimension")

    def to_dict(self) -> dict[str, object]:
        """Serialize the budget ID and every registered resource limit."""

        return {
            "schema": self.schema,
            "budget_id": self.budget_id,
            "limits": [limit.to_dict() for limit in self.limits],
        }


@dataclass(frozen=True)
class TreatmentArmRegistrationV1:
    """Freeze one arm and the separately stable locator and fixer components."""

    treatment_id: str
    applicable_tracks: tuple[TraceFixTrack, ...]
    locator_treatment_id: str
    fixer_treatment_id: str
    locator_config_refs: tuple[str, ...]
    fixer_config_refs: tuple[str, ...]
    method_config_refs: tuple[str, ...]
    schema: str = TREATMENT_ARM_REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TREATMENT_ARM_REGISTRATION_SCHEMA:
            raise ValueError("unsupported treatment arm registration schema")
        for name in (
            "treatment_id",
            "locator_treatment_id",
            "fixer_treatment_id",
        ):
            _identifier(getattr(self, name), name)
        _unique_tracks(self.applicable_tracks, "treatment applicable_tracks")
        _unique_texts(self.locator_config_refs, "locator_config_refs")
        _unique_texts(self.fixer_config_refs, "fixer_config_refs")
        _unique_texts(self.method_config_refs, "method_config_refs")
        if not self.locator_config_refs or not self.method_config_refs:
            raise ValueError(
                "treatment registration requires locator and combined method config refs"
            )
        component_refs = {
            reference
            for references in (self.locator_config_refs, self.fixer_config_refs)
            for reference in references
        }
        if component_refs != set(self.method_config_refs):
            raise ValueError(
                "method_config_refs must close exactly over locator and fixer config refs"
            )
        if (
            any(track is not TraceFixTrack.STATIC_ATTRIBUTION for track in self.applicable_tracks)
            and not self.fixer_config_refs
        ):
            raise ValueError("non-static treatment registration requires fixer config refs")
        if TraceFixTrack.STATIC_ATTRIBUTION in self.applicable_tracks:
            if (
                self.applicable_tracks != (TraceFixTrack.STATIC_ATTRIBUTION,)
                or self.fixer_treatment_id != STATIC_NO_FIXER_TREATMENT_ID
                or self.fixer_config_refs
            ):
                raise ValueError(
                    "static treatment must exclusively use the fixed no-fixer identity"
                )
        elif self.fixer_treatment_id.casefold() == STATIC_NO_FIXER_TREATMENT_ID:
            raise ValueError("non-static treatment cannot use the no-fixer identity")

    def to_dict(self) -> dict[str, object]:
        """Serialize the registry-authoritative treatment meaning."""

        return {
            "schema": self.schema,
            "treatment_id": self.treatment_id,
            "applicable_tracks": [track.value for track in self.applicable_tracks],
            "locator_treatment_id": self.locator_treatment_id,
            "fixer_treatment_id": self.fixer_treatment_id,
            "locator_config_refs": list(self.locator_config_refs),
            "fixer_config_refs": list(self.fixer_config_refs),
            "method_config_refs": list(self.method_config_refs),
        }


@dataclass(frozen=True)
class SystemCellRegistrationV1:
    """Freeze comparison axes and one actor identity shared across runtime roles."""

    system_cell_id: str
    population_id: str
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    applicable_tracks: tuple[TraceFixTrack, ...]
    execution_semantics: ExecutionSemantics
    fixer_treatment_id: str
    budget_id: str
    actor_identity_contract_ref: str
    source_actor_config_refs: tuple[str, ...]
    recovery_actor_config_refs: tuple[str, ...]
    system_config_refs: tuple[str, ...]
    schema: str = SYSTEM_CELL_REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SYSTEM_CELL_REGISTRATION_SCHEMA:
            raise ValueError("unsupported system cell registration schema")
        for name in (
            "system_cell_id",
            "population_id",
            "benchmark_id",
            "benchmark_version",
            "adapter_version",
            "fixer_treatment_id",
            "budget_id",
        ):
            _identifier(getattr(self, name), name)
        _unique_tracks(self.applicable_tracks, "system cell applicable_tracks")
        if len(self.applicable_tracks) != 1:
            raise ValueError("one system cell must freeze exactly one track")
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError("system cell execution_semantics must use the closed enum")
        _text(self.actor_identity_contract_ref, "actor_identity_contract_ref")
        _unique_texts(self.source_actor_config_refs, "source_actor_config_refs")
        _unique_texts(self.recovery_actor_config_refs, "recovery_actor_config_refs")
        _unique_texts(self.system_config_refs, "system_config_refs")
        if not self.source_actor_config_refs or not self.system_config_refs:
            raise ValueError(
                "system cell registration requires source-actor and combined config refs"
            )
        if self.actor_identity_contract_ref not in self.source_actor_config_refs:
            raise ValueError(
                "source actor config refs must bind the canonical actor identity contract"
            )
        if (
            self.recovery_actor_config_refs
            and self.actor_identity_contract_ref
            not in self.recovery_actor_config_refs
        ):
            raise ValueError(
                "recovery actor config refs must bind the canonical actor identity contract"
            )
        if self.applicable_tracks[0] is TraceFixTrack.STATIC_ATTRIBUTION:
            if self.recovery_actor_config_refs:
                raise ValueError("static system cells cannot declare a recovery actor")
        elif not self.recovery_actor_config_refs:
            raise ValueError(
                "non-static source and recovery actors must bind one canonical identity contract"
            )
        elif self.source_actor_config_refs != self.recovery_actor_config_refs:
            raise ValueError(
                "non-static source and recovery actors must use the same config refs"
            )
        actor_refs = {
            reference
            for references in (
                self.source_actor_config_refs,
                self.recovery_actor_config_refs,
            )
            for reference in references
        }
        if actor_refs != set(self.system_config_refs):
            raise ValueError(
                "system_config_refs must close exactly over source and recovery actor refs"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize the registry-authoritative system-cell meaning."""

        return {
            "schema": self.schema,
            "system_cell_id": self.system_cell_id,
            "population_id": self.population_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "applicable_tracks": [track.value for track in self.applicable_tracks],
            "execution_semantics": self.execution_semantics.value,
            "fixer_treatment_id": self.fixer_treatment_id,
            "budget_id": self.budget_id,
            "actor_identity_contract_ref": self.actor_identity_contract_ref,
            "source_actor_config_refs": list(self.source_actor_config_refs),
            "recovery_actor_config_refs": list(self.recovery_actor_config_refs),
            "system_config_refs": list(self.system_config_refs),
        }


@dataclass(frozen=True)
class ScorerRegistrationV1:
    """Assign exactly one built-in scorer key and contract references to a track."""

    scorer_id: str
    scorer_version: str
    track: TraceFixTrack
    implementation_key: BuiltinScorerKey
    progress_contract_ids: tuple[str, ...]
    verifier_contract_ids: tuple[str, ...]
    oracle_contract_ids: tuple[str, ...]
    budget_ids: tuple[str, ...]
    treatment_ids: tuple[str, ...]
    system_cell_ids: tuple[str, ...]
    schema: str = SCORER_REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCORER_REGISTRATION_SCHEMA:
            raise ValueError("unsupported scorer registration schema")
        _identifier(self.scorer_id, "scorer_id")
        _identifier(self.scorer_version, "scorer_version")
        if not isinstance(self.track, TraceFixTrack) or self.implementation_key is not _KEY_BY_TRACK.get(self.track):
            raise ValueError("track must select its fixed built-in scorer key")
        for name in (
            "progress_contract_ids",
            "verifier_contract_ids",
            "oracle_contract_ids",
            "budget_ids",
            "treatment_ids",
            "system_cell_ids",
        ):
            _unique_identifiers(getattr(self, name), name)
        if self.track is TraceFixTrack.STATIC_ATTRIBUTION and (
            self.progress_contract_ids
            or self.verifier_contract_ids
            or self.oracle_contract_ids
        ):
            raise ValueError("static attribution cannot consume progress, verifier, or forward oracle contracts")
        if self.track is TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION and (
            not self.progress_contract_ids
            or not self.verifier_contract_ids
            or self.oracle_contract_ids
        ):
            raise ValueError("historical scorer requires progress and official verifier, without forward oracles")
        if self.track is TraceFixTrack.FORWARD_RECOVERY and not self.verifier_contract_ids:
            raise ValueError("forward scorer requires an official final verifier contract")

    def to_dict(self) -> dict[str, object]:
        """Serialize scorer selection using only the closed allowlist key."""

        return {
            "schema": self.schema,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "track": self.track.value,
            "implementation_key": self.implementation_key.value,
            "progress_contract_ids": list(self.progress_contract_ids),
            "verifier_contract_ids": list(self.verifier_contract_ids),
            "oracle_contract_ids": list(self.oracle_contract_ids),
            "budget_ids": list(self.budget_ids),
            "treatment_ids": list(self.treatment_ids),
            "system_cell_ids": list(self.system_cell_ids),
        }


@dataclass(frozen=True)
class AggregatePlanV1:
    """Freeze equal-weight task clustering and unavailable-data behavior."""

    aggregate_plan_id: str
    repetition_weighting: str = "equal_within_case"
    case_weighting: str = "equal_within_source_task"
    source_task_weighting: str = "equal_across_source_tasks"
    cluster_unit: str = "source_task_id"
    bootstrap_unit: str = "source_task_id"
    infrastructure_invalid_policy: str = "confirmatory_null_incomplete"
    cross_track_overall_score: bool = False
    cross_benchmark_progress_average: bool = False
    schema: str = AGGREGATE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != AGGREGATE_PLAN_SCHEMA:
            raise ValueError("unsupported aggregate plan schema")
        _identifier(self.aggregate_plan_id, "aggregate_plan_id")
        for name in (
            "repetition_weighting",
            "case_weighting",
            "source_task_weighting",
            "cluster_unit",
            "bootstrap_unit",
            "infrastructure_invalid_policy",
        ):
            if type(getattr(self, name)) is not str:
                raise ValueError(f"aggregate plan {name} must be text")
        if (
            type(self.cross_track_overall_score) is not bool
            or type(self.cross_benchmark_progress_average) is not bool
        ):
            raise ValueError("aggregate plan boolean fields must be booleans")
        expected = {
            "repetition_weighting": "equal_within_case",
            "case_weighting": "equal_within_source_task",
            "source_task_weighting": "equal_across_source_tasks",
            "cluster_unit": "source_task_id",
            "bootstrap_unit": "source_task_id",
            "infrastructure_invalid_policy": "confirmatory_null_incomplete",
            "cross_track_overall_score": False,
            "cross_benchmark_progress_average": False,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("aggregate plan must retain the frozen v0.1 semantics")

    def to_dict(self) -> dict[str, object]:
        """Serialize weighting, cluster, and missing-cell rules explicitly."""

        return {
            "schema": self.schema,
            "aggregate_plan_id": self.aggregate_plan_id,
            "repetition_weighting": self.repetition_weighting,
            "case_weighting": self.case_weighting,
            "source_task_weighting": self.source_task_weighting,
            "cluster_unit": self.cluster_unit,
            "bootstrap_unit": self.bootstrap_unit,
            "infrastructure_invalid_policy": self.infrastructure_invalid_policy,
            "cross_track_overall_score": self.cross_track_overall_score,
            "cross_benchmark_progress_average": self.cross_benchmark_progress_average,
        }


@dataclass(frozen=True)
class ComparisonFamilyV1:
    """Preregister one domain, semantics, and paired compatible system family."""

    family_id: str
    track: TraceFixTrack
    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    execution_semantics: ExecutionSemantics
    endpoint_ids: tuple[BuiltinEndpointKey, ...]
    treatment_ids: tuple[str, ...]
    budget_ids: tuple[str, ...]
    system_cell_ids: tuple[str, ...]
    multiplicity_correction: MultiplicityCorrection
    binary_reduction_policy: BinaryReductionPolicy
    binary_reducer_ref: str | None = None
    paired_roster_required: bool = True
    schema: str = COMPARISON_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COMPARISON_FAMILY_SCHEMA:
            raise ValueError("unsupported comparison family schema")
        _identifier(self.family_id, "family_id")
        for name in ("benchmark_id", "benchmark_version", "adapter_version"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.track, TraceFixTrack):
            raise ValueError("comparison family track must use the closed enum")
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError(
                "comparison family execution_semantics must use the closed enum"
            )
        if not isinstance(self.endpoint_ids, tuple) or not self.endpoint_ids:
            raise ValueError("comparison family endpoint_ids cannot be empty")
        if any(
            not isinstance(endpoint, BuiltinEndpointKey)
            or endpoint not in _ENDPOINTS_BY_TRACK[self.track]
            for endpoint in self.endpoint_ids
        ):
            raise ValueError("comparison endpoint must belong to its registered track")
        if len(set(self.endpoint_ids)) != len(self.endpoint_ids):
            raise ValueError("comparison endpoints must be unique")
        for name in ("treatment_ids", "budget_ids", "system_cell_ids"):
            values = getattr(self, name)
            _unique_identifiers(values, name)
            if not values:
                raise ValueError(f"comparison family {name} cannot be empty")
        if len(self.treatment_ids) < 2:
            raise ValueError("paired comparison requires at least two treatments")
        if self.multiplicity_correction is not MultiplicityCorrection.HOLM:
            raise ValueError("v0.1 comparison families require Holm correction")
        if not isinstance(self.binary_reduction_policy, BinaryReductionPolicy):
            raise ValueError("binary reduction policy must use the closed enum")
        if self.binary_reduction_policy is BinaryReductionPolicy.PREREGISTERED_SOURCE_TASK_REDUCER:
            _text(self.binary_reducer_ref, "binary_reducer_ref")
        elif self.binary_reducer_ref is not None:
            raise ValueError("binary_reducer_ref requires a preregistered reducer policy")
        if self.paired_roster_required is not True:
            raise ValueError("comparison family requires an identical paired roster")

    def to_dict(self) -> dict[str, object]:
        """Serialize the exact family before outcome visibility."""

        return {
            "schema": self.schema,
            "family_id": self.family_id,
            "track": self.track.value,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "adapter_version": self.adapter_version,
            "execution_semantics": self.execution_semantics.value,
            "endpoint_ids": [item.value for item in self.endpoint_ids],
            "treatment_ids": list(self.treatment_ids),
            "budget_ids": list(self.budget_ids),
            "system_cell_ids": list(self.system_cell_ids),
            "multiplicity_correction": self.multiplicity_correction.value,
            "binary_reduction_policy": self.binary_reduction_policy.value,
            "binary_reducer_ref": self.binary_reducer_ref,
            "paired_roster_required": self.paired_roster_required,
        }


@dataclass(frozen=True)
class ScorerRegistryV1:
    """Close all three scorer registrations and their versioned policy references."""

    registry_id: str
    suite_id: str
    suite_version: str
    population_id: str
    scorers: tuple[ScorerRegistrationV1, ...]
    progress_metrics: tuple[ProgressMetricContractV1, ...]
    verifiers: tuple[VerifierContractV1, ...]
    oracles: tuple[OracleContractV1, ...]
    budgets: tuple[ResourceBudgetContractV1, ...]
    treatments: tuple[TreatmentArmRegistrationV1, ...]
    system_cells: tuple[SystemCellRegistrationV1, ...]
    aggregate_plan: AggregatePlanV1
    comparison_families: tuple[ComparisonFamilyV1, ...]
    source_refs: tuple[str, ...]
    registration_receipt_refs: tuple[str, ...]
    hit_cutoffs: tuple[int, ...] = STATIC_HIT_CUTOFFS_V0_1
    primary_k: int = STATIC_PRIMARY_K_V0_1
    allowed_finding_kinds: tuple[TraceFindingKind, ...] = STATIC_ALLOWED_FINDING_KINDS_V0_1
    implementation_status: str = SCORER_REGISTRY_IMPLEMENTATION_STATUS
    schema: str = SCORER_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCORER_REGISTRY_SCHEMA:
            raise ValueError("unsupported scorer registry schema")
        _identifier(self.registry_id, "registry_id")
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID:
            raise ValueError("scorer registry suite_id must be LongHorizon-TraceFix-Bench")
        if self.suite_version != SCORER_REGISTRY_SUITE_VERSION_V0_1:
            raise ValueError("scorer registry suite_version must equal the v0.1 freeze")
        _identifier(self.population_id, "population_id")
        if (
            type(self.implementation_status) is not str
            or self.implementation_status != SCORER_REGISTRY_IMPLEMENTATION_STATUS
        ):
            raise ValueError("scorer registry cannot promote implementation status")
        if (
            type(self.hit_cutoffs) is not tuple
            or any(type(cutoff) is not int or cutoff < 1 for cutoff in self.hit_cutoffs)
            or tuple(sorted(set(self.hit_cutoffs))) != self.hit_cutoffs
        ):
            raise ValueError("static hit cutoffs must be ordered unique positive integers")
        if type(self.primary_k) is not int or self.primary_k < 1:
            raise ValueError("static primary K must be a positive integer")
        if self.hit_cutoffs != STATIC_HIT_CUTOFFS_V0_1 or self.primary_k != STATIC_PRIMARY_K_V0_1:
            raise ValueError("static hit cutoffs and primary K must equal the v0.1 freeze")
        if (
            type(self.allowed_finding_kinds) is not tuple
            or any(
                not isinstance(kind, TraceFindingKind)
                for kind in self.allowed_finding_kinds
            )
        ):
            raise ValueError("static finding kinds must use the closed enum")
        if self.allowed_finding_kinds != STATIC_ALLOWED_FINDING_KINDS_V0_1:
            raise ValueError("static finding kinds must equal the v0.1 freeze")
        for name, kind in (
            ("scorers", ScorerRegistrationV1),
            ("progress_metrics", ProgressMetricContractV1),
            ("verifiers", VerifierContractV1),
            ("oracles", OracleContractV1),
            ("budgets", ResourceBudgetContractV1),
            ("treatments", TreatmentArmRegistrationV1),
            ("system_cells", SystemCellRegistrationV1),
            ("comparison_families", ComparisonFamilyV1),
        ):
            _records(getattr(self, name), kind, name)
        if not isinstance(self.aggregate_plan, AggregatePlanV1):
            raise ValueError("aggregate_plan must be an AggregatePlanV1")
        for values, attribute, label in (
            (self.scorers, "scorer_id", "scorer IDs"),
            (
                self.progress_metrics,
                "progress_contract_id",
                "progress contract IDs",
            ),
            (self.verifiers, "verifier_contract_id", "verifier contract IDs"),
            (self.oracles, "oracle_contract_id", "oracle contract IDs"),
            (self.budgets, "budget_id", "budget IDs"),
            (self.treatments, "treatment_id", "treatment IDs"),
            (self.system_cells, "system_cell_id", "system cell IDs"),
            (self.comparison_families, "family_id", "comparison family IDs"),
        ):
            _unique_record_ids(values, attribute, label)
        if len(self.scorers) != len(TraceFixTrack) or {entry.track for entry in self.scorers} != set(TraceFixTrack):
            raise ValueError("scorer registry requires exactly one registration per track")
        _unique_texts(self.source_refs, "source_refs")
        _unique_texts(self.registration_receipt_refs, "registration_receipt_refs")
        if not self.source_refs or not self.registration_receipt_refs:
            raise ValueError("scorer registry requires source and registration receipt references")
        source_refs = set(self.source_refs)
        for contract in (*self.progress_metrics, *self.verifiers, *self.oracles):
            if contract.source_ref not in source_refs:
                raise ValueError("benchmark contract source_ref is absent from registry sources")
        for registration in self.treatments:
            config_refs = (
                *registration.locator_config_refs,
                *registration.fixer_config_refs,
                *registration.method_config_refs,
            )
            if any(reference not in self.source_refs for reference in config_refs):
                raise ValueError(
                    "treatment method config ref is absent from registry sources"
                )
        for registration in self.system_cells:
            if registration.population_id != self.population_id:
                raise ValueError(
                    "system cell population_id must equal the registry population"
                )
            config_refs = (
                registration.actor_identity_contract_ref,
                *registration.source_actor_config_refs,
                *registration.recovery_actor_config_refs,
                *registration.system_config_refs,
            )
            if any(reference not in self.source_refs for reference in config_refs):
                raise ValueError(
                    "system cell config ref is absent from registry sources"
                )
        for contract in self.progress_metrics:
            if contract.aggregation_rule_source_ref not in source_refs:
                raise ValueError(
                    "progress aggregation source_ref is absent from registry sources"
                )
        for family in self.comparison_families:
            if (
                family.binary_reducer_ref is not None
                and family.binary_reducer_ref not in source_refs
            ):
                raise ValueError("binary reducer source_ref is absent from registry sources")

        # Every named contract must be reachable from a scorer or comparison family.
        domains = (
            (
                "progress_contract_ids",
                self.progress_metrics,
                "progress_contract_id",
            ),
            ("verifier_contract_ids", self.verifiers, "verifier_contract_id"),
            ("oracle_contract_ids", self.oracles, "oracle_contract_id"),
            ("budget_ids", self.budgets, "budget_id"),
        )
        for reference_name, contracts, identity_name in domains:
            declared = {getattr(contract, identity_name).casefold() for contract in contracts}
            referenced = {
                identity.casefold()
                for scorer in self.scorers
                for identity in getattr(scorer, reference_name)
            }
            if reference_name == "budget_ids":
                referenced.update(
                    identity.casefold()
                    for family in self.comparison_families
                    for identity in family.budget_ids
                )
            if declared != referenced:
                raise ValueError(f"{reference_name} must close exactly over declared contracts")
        metrics_by_id = {
            item.progress_contract_id.casefold(): item
            for item in self.progress_metrics
        }
        verifiers_by_id = {
            item.verifier_contract_id.casefold(): item for item in self.verifiers
        }
        oracles_by_id = {item.oracle_contract_id.casefold(): item for item in self.oracles}
        budgets_by_id = {item.budget_id.casefold(): item for item in self.budgets}
        treatments_by_id = {
            item.treatment_id.casefold(): item for item in self.treatments
        }
        systems_by_id = {
            item.system_cell_id.casefold(): item for item in self.system_cells
        }
        contract_reference_groups = (
            (
                "progress_contract_ids",
                metrics_by_id,
                "progress_contract_id",
            ),
            (
                "verifier_contract_ids",
                verifiers_by_id,
                "verifier_contract_id",
            ),
            ("oracle_contract_ids", oracles_by_id, "oracle_contract_id"),
            ("budget_ids", budgets_by_id, "budget_id"),
        )
        for scorer in self.scorers:
            for reference_name, contracts_by_id, identity_name in contract_reference_groups:
                for reference in getattr(scorer, reference_name):
                    contract = contracts_by_id.get(reference.casefold())
                    if contract is None or reference != getattr(contract, identity_name):
                        raise ValueError(
                            f"scorer {reference_name} must use exact registered spelling"
                        )
        locator_components: dict[str, tuple[str, tuple[str, ...]]] = {}
        fixer_components: dict[str, tuple[str, tuple[str, ...]]] = {}
        for registration in self.treatments:
            for component_id, config_refs, components, label in (
                (
                    registration.locator_treatment_id,
                    registration.locator_config_refs,
                    locator_components,
                    "locator",
                ),
                (
                    registration.fixer_treatment_id,
                    registration.fixer_config_refs,
                    fixer_components,
                    "fixer",
                ),
            ):
                key = component_id.casefold()
                prior = components.get(key)
                current = (component_id, config_refs)
                if prior is not None and prior != current:
                    raise ValueError(
                        f"{label} treatment ID must retain exact spelling and config refs"
                    )
                components[key] = current
        for metric in self.progress_metrics:
            evaluator = verifiers_by_id.get(metric.evaluator_contract_id.casefold())
            if (
                evaluator is None
                or metric.evaluator_contract_id != evaluator.verifier_contract_id
                or _benchmark_domain(evaluator) != _benchmark_domain(metric)
            ):
                raise ValueError(
                    "progress evaluator contract must resolve exactly in the same benchmark domain"
                )
        scorer_by_track = {scorer.track: scorer for scorer in self.scorers}
        referenced_treatments: set[str] = set()
        referenced_systems: set[str] = set()
        treatment_tracks: dict[str, set[TraceFixTrack]] = {
            identity: set() for identity in treatments_by_id
        }
        system_tracks: dict[str, set[TraceFixTrack]] = {
            identity: set() for identity in systems_by_id
        }
        for scorer in self.scorers:
            for identity in scorer.treatment_ids:
                registration = treatments_by_id.get(identity.casefold())
                if (
                    registration is None
                    or identity != registration.treatment_id
                    or scorer.track not in registration.applicable_tracks
                ):
                    raise ValueError(
                        "scorer treatment must resolve with exact spelling and apply to its track"
                )
                referenced_treatments.add(identity.casefold())
                treatment_tracks[identity.casefold()].add(scorer.track)
            for identity in scorer.system_cell_ids:
                registration = systems_by_id.get(identity.casefold())
                if (
                    registration is None
                    or identity != registration.system_cell_id
                    or scorer.track not in registration.applicable_tracks
                ):
                    raise ValueError(
                        "scorer system cell must resolve with exact spelling and apply to its track"
                )
                referenced_systems.add(identity.casefold())
                system_tracks[identity.casefold()].add(scorer.track)
        for registration in self.system_cells:
            track = registration.applicable_tracks[0]
            scorer = scorer_by_track[track]
            budget = budgets_by_id.get(registration.budget_id.casefold())
            if (
                budget is None
                or registration.budget_id != budget.budget_id
                or registration.budget_id not in scorer.budget_ids
            ):
                raise ValueError(
                    "system cell budget must resolve exactly in its track scorer"
                )
            track_treatments = tuple(
                treatments_by_id[identity.casefold()]
                for identity in scorer.treatment_ids
            )
            if not any(
                treatment.fixer_treatment_id == registration.fixer_treatment_id
                for treatment in track_treatments
            ):
                raise ValueError(
                    "system cell fixer must resolve through a treatment in its track scorer"
                )
            system_domain = _exact_benchmark_domain(registration)
            progress_domains = {
                _exact_benchmark_domain(metrics_by_id[identity.casefold()])
                for identity in scorer.progress_contract_ids
            }
            verifier_domains = {
                _exact_benchmark_domain(verifiers_by_id[identity.casefold()])
                for identity in scorer.verifier_contract_ids
            }
            if (
                track is TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION
                and (
                    system_domain not in progress_domains
                    or system_domain not in verifier_domains
                )
            ):
                raise ValueError(
                    "historical system cell domain requires its progress and verifier contracts"
                )
            if (
                track is TraceFixTrack.FORWARD_RECOVERY
                and system_domain not in verifier_domains
            ):
                raise ValueError(
                    "forward system cell domain requires its official verifier contract"
                )
        for family in self.comparison_families:
            track_scorer = scorer_by_track[family.track]
            for budget_id in family.budget_ids:
                budget = budgets_by_id.get(budget_id.casefold())
                if (
                    budget is None
                    or budget_id != budget.budget_id
                    or budget_id not in track_scorer.budget_ids
                ):
                    raise ValueError(
                        "comparison family budget must resolve exactly in its track scorer"
                    )
            scorer_treatments = set(track_scorer.treatment_ids)
            scorer_systems = set(track_scorer.system_cell_ids)
            for identity in family.treatment_ids:
                registration = treatments_by_id.get(identity.casefold())
                if (
                    registration is None
                    or identity != registration.treatment_id
                    or family.track not in registration.applicable_tracks
                    or identity not in scorer_treatments
                ):
                    raise ValueError(
                        "comparison treatment must resolve exactly and belong to its track scorer"
                    )
                referenced_treatments.add(identity.casefold())
            for identity in family.system_cell_ids:
                registration = systems_by_id.get(identity.casefold())
                if (
                    registration is None
                    or identity != registration.system_cell_id
                    or family.track not in registration.applicable_tracks
                    or identity not in scorer_systems
                ):
                    raise ValueError(
                        "comparison system cell must resolve exactly and belong to its track scorer"
                )
                referenced_systems.add(identity.casefold())
            family_treatments = tuple(
                treatments_by_id[identity.casefold()]
                for identity in family.treatment_ids
            )
            family_systems = tuple(
                systems_by_id[identity.casefold()]
                for identity in family.system_cell_ids
            )
            family_fixers = {
                treatment.fixer_treatment_id for treatment in family_treatments
            }
            if len(family_fixers) != 1:
                raise ValueError(
                    "one comparison family must freeze one fixer treatment"
                )
            family_fixer = next(iter(family_fixers))
            family_domain = (
                family.benchmark_id,
                family.benchmark_version,
                family.adapter_version,
            )
            if any(
                (
                    system.benchmark_id,
                    system.benchmark_version,
                    system.adapter_version,
                )
                != family_domain
                or system.applicable_tracks != (family.track,)
                or system.execution_semantics is not family.execution_semantics
                or system.budget_id not in family.budget_ids
                or system.fixer_treatment_id != family_fixer
                for system in family_systems
            ):
                raise ValueError(
                    "comparison family system cells must match its domain, track, semantics, budget, and fixer"
                )
            system_budget_ids = tuple(
                dict.fromkeys(system.budget_id for system in family_systems)
            )
            if system_budget_ids != family.budget_ids:
                raise ValueError(
                    "comparison family budgets must equal system-cell budget order"
                )
        if referenced_treatments != set(treatments_by_id):
            raise ValueError(
                "treatment registrations must close exactly over scorer or comparison use"
            )
        if referenced_systems != set(systems_by_id):
            raise ValueError(
                "system cell registrations must close exactly over scorer or comparison use"
            )
        if any(
            treatment_tracks[identity] != set(registration.applicable_tracks)
            for identity, registration in treatments_by_id.items()
        ):
            raise ValueError(
                "treatment applicable_tracks must equal explicit scorer-track use"
            )
        if any(
            system_tracks[identity] != set(registration.applicable_tracks)
            for identity, registration in systems_by_id.items()
        ):
            raise ValueError(
                "system cell applicable_tracks must equal explicit scorer-track use"
            )
        for scorer in self.scorers:
            metric_domains = {
                _benchmark_domain(metrics_by_id[identity.casefold()])
                for identity in scorer.progress_contract_ids
            }
            verifier_domains = {
                _benchmark_domain(verifiers_by_id[identity.casefold()])
                for identity in scorer.verifier_contract_ids
            }
            oracle_domains = {
                _benchmark_domain(oracles_by_id[identity.casefold()])
                for identity in scorer.oracle_contract_ids
            }
            system_domains = {
                _benchmark_domain(systems_by_id[identity.casefold()])
                for identity in scorer.system_cell_ids
            }
            registered_verifier_ids = {
                identity.casefold() for identity in scorer.verifier_contract_ids
            }
            metric_evaluator_ids = {
                metrics_by_id[identity.casefold()].evaluator_contract_id.casefold()
                for identity in scorer.progress_contract_ids
            }
            if not metric_evaluator_ids <= registered_verifier_ids:
                raise ValueError(
                    "progress evaluators must be registered by the consuming scorer"
                )
            if scorer.track is TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION:
                if metric_domains != verifier_domains or metric_domains != system_domains:
                    raise ValueError(
                        "historical progress, verifier, and system-cell domains must match"
                    )
                historical_metrics = tuple(
                    metrics_by_id[identity.casefold()]
                    for identity in scorer.progress_contract_ids
                )
                domain_counts: dict[tuple[str, str, str], int] = {}
                for metric in historical_metrics:
                    domain = _benchmark_domain(metric)
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    if len(metric.timepoints) != 1:
                        raise ValueError(
                            "historical classification progress requires one frozen timepoint"
                        )
                # v0.1 has no preregistered rule for choosing or composing metrics.
                if any(count != 1 for count in domain_counts.values()):
                    raise ValueError(
                        "historical scorer requires exactly one classification progress contract per benchmark domain"
                    )
                historical_verifier_counts: dict[tuple[str, str, str], int] = {}
                for identity in scorer.verifier_contract_ids:
                    domain = _benchmark_domain(
                        verifiers_by_id[identity.casefold()]
                    )
                    historical_verifier_counts[domain] = (
                        historical_verifier_counts.get(domain, 0) + 1
                    )
                if any(
                    count != 1 for count in historical_verifier_counts.values()
                ):
                    raise ValueError(
                        "historical scorer requires exactly one verifier per benchmark domain"
                    )
            elif scorer.track is TraceFixTrack.FORWARD_RECOVERY:
                if (
                    verifier_domains != system_domains
                    or not metric_domains <= verifier_domains
                    or not oracle_domains <= verifier_domains
                ):
                    raise ValueError(
                        "forward system-cell, metric, and oracle domains require the exact official verifier roster"
                    )
                forward_verifier_counts: dict[tuple[str, str, str], int] = {}
                for identity in scorer.verifier_contract_ids:
                    domain = _benchmark_domain(
                        verifiers_by_id[identity.casefold()]
                    )
                    forward_verifier_counts[domain] = (
                        forward_verifier_counts.get(domain, 0) + 1
                    )
                if any(count != 1 for count in forward_verifier_counts.values()):
                    raise ValueError(
                        "forward scorer requires exactly one verifier per benchmark domain"
                    )
                forward_metric_counts: dict[tuple[str, str, str], int] = {}
                for identity in scorer.progress_contract_ids:
                    domain = _benchmark_domain(metrics_by_id[identity.casefold()])
                    forward_metric_counts[domain] = (
                        forward_metric_counts.get(domain, 0) + 1
                    )
                if any(count > 1 for count in forward_metric_counts.values()):
                    raise ValueError(
                        "forward scorer allows at most one progress contract per benchmark domain"
                    )
                forward_oracle_keys: set[tuple[tuple[str, str, str], OracleKind]] = set()
                for identity in scorer.oracle_contract_ids:
                    oracle = oracles_by_id[identity.casefold()]
                    key = (_benchmark_domain(oracle), oracle.kind)
                    if key in forward_oracle_keys:
                        raise ValueError(
                            "forward scorer allows at most one oracle per domain and kind"
                        )
                    forward_oracle_keys.add(key)
        if len(self.comparison_families) == 0:
            raise ValueError("scorer registry requires at least one preregistered comparison family")

    def to_dict(self) -> dict[str, object]:
        """Serialize policy and references without exposing dynamic execution hooks."""

        return {
            "schema": self.schema,
            "registry_id": self.registry_id,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "implementation_status": self.implementation_status,
            "scorers": [item.to_dict() for item in self.scorers],
            "hit_cutoffs": list(self.hit_cutoffs),
            "primary_k": self.primary_k,
            "allowed_finding_kinds": [kind.value for kind in self.allowed_finding_kinds],
            "progress_metrics": [item.to_dict() for item in self.progress_metrics],
            "verifiers": [item.to_dict() for item in self.verifiers],
            "oracles": [item.to_dict() for item in self.oracles],
            "budgets": [item.to_dict() for item in self.budgets],
            "treatments": [item.to_dict() for item in self.treatments],
            "system_cells": [item.to_dict() for item in self.system_cells],
            "aggregate_plan": self.aggregate_plan.to_dict(),
            "comparison_families": [item.to_dict() for item in self.comparison_families],
            "source_refs": list(self.source_refs),
            "registration_receipt_refs": list(self.registration_receipt_refs),
        }


__all__ = [
    "AGGREGATE_PLAN_SCHEMA", "COMPARISON_FAMILY_SCHEMA", "ORACLE_CONTRACT_SCHEMA",
    "PROGRESS_METRIC_CONTRACT_SCHEMA", "RESOURCE_BUDGET_CONTRACT_SCHEMA",
    "SCORER_REGISTRATION_SCHEMA", "SCORER_REGISTRY_FILENAME", "SCORER_REGISTRY_SCHEMA",
    "SCORER_REGISTRY_SUITE_VERSION_V0_1",
    "SCORER_REGISTRY_IMPLEMENTATION_STATUS", "STATIC_ALLOWED_FINDING_KINDS_V0_1",
    "STATIC_HIT_CUTOFFS_V0_1", "STATIC_NO_FIXER_TREATMENT_ID",
    "STATIC_PRIMARY_K_V0_1",
    "SYSTEM_CELL_REGISTRATION_SCHEMA", "TREATMENT_ARM_REGISTRATION_SCHEMA",
    "VERIFIER_CONTRACT_SCHEMA",
    "AggregatePlanV1", "BinaryReductionPolicy", "BudgetScope", "BuiltinEndpointKey",
    "BuiltinScorerKey",
    "ComparisonFamilyV1", "EvidenceSource", "MultiplicityCorrection", "OracleContractV1",
    "OracleKind", "ProgressDirection", "ProgressMetricContractV1",
    "ProgressUnavailablePolicy", "ResourceBudgetContractV1",
    "ResourceDimension", "ResourceLimitV1", "ResourceRole", "ScorerRegistrationV1",
    "ScorerRegistryV1", "SystemCellRegistrationV1", "TreatmentArmRegistrationV1",
    "VerifierContractV1",
]
