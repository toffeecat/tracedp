"""Bind a completed managed runtime trace to Harbor's independent evaluator."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from lh_platform.benchmarks.runtime_task_contract_compiler import (
    SUPPORTED_TASK_CONTRACT_COMPILERS,
    compile_initial_resource_task_graph,
    compile_public_instruction_dependency_evidence,
)
from lh_platform.benchmarks.terminal_delegated_tools import (
    ACTOR_CONVERGENCE_RESERVE_SECONDS,
    ACTOR_FAILURE_DIAGNOSIS_THRESHOLD,
    ACTOR_TOOL_RECEIPT_SCHEMA,
    OUTPUT_PROGRESS_ACTION_THRESHOLD,
    OUTPUT_SEMANTIC_STABILITY_ACTION_THRESHOLD,
    TerminalEnvironmentToolGateway,
)
from lh_platform.benchmarks.terminal_managed_runtime_identity import (
    DYNAMIC_RUNTIME_METHODS,
    DYNAMIC_TRACEDP_V1,
    DYNAMIC_TRACEDP_V2,
    DYNAMIC_TRACEDP_V3,
    DYNAMIC_TRACEDP_V4,
    EFFECT_LEDGER_RUNTIME_METHODS,
    FIXED_TWO_STAGE_RUNTIME_METHOD,
    INTERVENTION_BUDGET_METHODS,
    JOINT_RUNTIME_METHODS,
    MANAGED_OBSERVATION_METHOD,
    OUTPUT_PROGRESS_RUNTIME_METHODS,
    OUTPUT_SEMANTIC_RUNTIME_METHODS,
    POLICY_RUNTIME_METHODS,
    SHARED_INPUT_PROTECTION_METHOD,
    SUPPORTED_RUNTIME_METHODS,
)
from lh_platform.benchmarks.terminal_pi_stream import (
    PiToolCall,
    PiToolExecution,
    load_pi_persistent_session_tool_receipt,
    load_pi_stream_receipt,
)
from lh_platform.benchmarks.terminal_provider_request_audit import (
    audit_terminal_provider_requests,
)
from lh_platform.benchmarks.terminal_resource_driver import (
    TERMINAL_RESOURCE_OBSERVATION_V1,
    TERMINAL_RESOURCE_OBSERVATION_V2,
    ResourceSpec,
    expand_sqlite_resource_family_specs,
)
from lh_platform.benchmarks.terminal_runtime_check import (
    RuntimeCheckContract,
    RuntimePredicateRepairContract,
    runtime_check_configuration_payload,
    runtime_repair_configuration_payload,
)
from lh_platform.benchmarks.terminal_runtime_check_audit import audit_runtime_check_evidence
from lh_platform.benchmarks.terminal_runtime_guidance_audit import (
    audit_runtime_predicate_guidance_followup,
)
from lh_platform.benchmarks.terminal_runtime_repair_audit import (
    audit_runtime_targeted_repairs,
)
from lh_platform.benchmarks.terminal_verifier_contract import (
    load_trial_public_instruction,
    load_trial_pytest_nodes,
)
from lh_platform.test_time_adaptation.adapters.terminal_runtime import (
    TERMINAL_RUNTIME_ADAPTER_V1,
)
from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    AcceptancePredicateAggregation,
    AcceptancePredicateState,
    ActionPhase,
    DependencyRole,
    EvidenceState,
    PortableRuntimeKernel,
    ResourceKind,
    ResourceObservation,
)
from lh_platform.test_time_adaptation.runtime_bounded_trace_dp import (
    RuntimeBoundedTraceDPResolver,
    RuntimeBoundedTraceDPV4Resolver,
    RuntimeBoundedTraceDPV6Resolver,
    RuntimeBoundedTraceDPV7Resolver,
    RuntimeEventTraceDPV8Resolver,
)
from lh_platform.test_time_adaptation.runtime_compact_prediction import (
    V7_PROVIDER_TIMEOUT_OVERRUN_MARGIN_MS,
    prediction_prompt_v7,
    public_prediction_context_v7,
)
from lh_platform.test_time_adaptation.runtime_event_admission import (
    V8_CUMULATIVE_PLANNER_BUDGET_MS,
    V8_SEMANTIC_REVIEW_THRESHOLD,
    bind_event_admission_to_resource_gate_v8,
    durable_event_consumptions_v8,
    planner_cumulative_budget_gate_v8,
    planner_event_admission_v8,
    planner_time_reserve_gate_v8,
)
from lh_platform.test_time_adaptation.runtime_guidance_phase_dp import (
    GuidedCompletionConfiguration,
    RuntimeGuidancePhaseLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_information_gain_selector import (
    RuntimeInformationGainLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_recovery_lookahead_selector import (
    RuntimeRecoveryLookaheadLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_obligation_value_selector import (
    RuntimeObligationValueLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_jit_action_advantage import (
    FrozenRuntimeJITMemory,
)
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionCandidate,
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    ACTOR_RESERVE_JOINT_METHODS,
    ADAPTIVE_INFORMATION_GAIN_RUNTIME_DEV_METHOD,
    TRACEDP_RECOVERY_LOOKAHEAD_DEV_METHOD,
    TRACEDP_OBLIGATION_VALUE_DEV_METHOD,
    BOUNDED_TRACEDP_DEV_METHODS,
    BOUNDED_TRACEDP_DEV_V4_METHOD,
    BOUNDED_TRACEDP_DEV_V6_METHOD,
    BOUNDED_TRACEDP_DEV_V7_METHOD,
    EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS,
    FAILURE_DIAGNOSIS_JOINT_METHODS,
    FIXED_JOINT_SLATE_V3_METHOD,
    JIT_JOINT_METHODS,
    PLANNED_JOINT_METHODS,
    ROTATING_SOURCE_JOINT_METHODS,
    RPC_JOINT_METHODS,
    RUNTIME_OPTION_CHAIN_METHODS,
    RUNTIME_OPTION_FOLLOWUP_TRIGGER,
    RUNTIME_OWNED_OPTION_DP_DEV_METHODS,
    SEMANTIC_STABILITY_JOINT_METHODS,
    SETTLEMENT_REVIEW_JOINT_METHODS,
    TIME_RESERVE_JOINT_METHODS,
    TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
    TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD,
    TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD,
    TRACEDP_JIT_SETTLEMENT_METHOD,
    TRACEDP_RESIDUAL_METHODS,
    RuntimeJointDecisionResolver,
    RuntimeJointSlatePolicy,
)
from lh_platform.test_time_adaptation.runtime_output_obligation import (
    RuntimeInterventionDeliveryStatus,
    RuntimeInterventionEffectLedger,
    RuntimeInterventionEffectRecord,
    RuntimeInterventionEffectStatus,
)
from lh_platform.test_time_adaptation.runtime_owned_option_live import (
    RuntimeFrozenAtomicEstimateConfiguration,
    RuntimeOwnedOptionLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_parent_coordinate_prediction import (
    prediction_prompt_v6,
    public_prediction_context_v6,
)
from lh_platform.test_time_adaptation.runtime_residual_jit import (
    RuntimeResidualJITResolver,
)
from lh_platform.test_time_adaptation.runtime_settlement_trace_dp import (
    RuntimeSettlementTraceDPResolver,
)
from lh_platform.test_time_adaptation.runtime_trace_dp import (
    DynamicTraceDPRuntimePolicy,
)
from lh_platform.training.pi_alfworld_runtime import (
    PiActorCompatibilityError,
    PiAuthenticatedOpenAiActor,
)

_RUNTIME_CONFIG_SCHEMA_V1 = "terminal_managed_runtime_config_v1"
_RUNTIME_CONFIG_SCHEMA_V2 = "terminal_managed_runtime_config_v2"
_DYNAMIC_METHOD_V1 = DYNAMIC_TRACEDP_V1
_DYNAMIC_METHOD_V2 = DYNAMIC_TRACEDP_V2
_DYNAMIC_METHOD_V3 = DYNAMIC_TRACEDP_V3
_DYNAMIC_METHOD_V4 = DYNAMIC_TRACEDP_V4
_DYNAMIC_METHODS = DYNAMIC_RUNTIME_METHODS
_FIXED_METHOD = FIXED_TWO_STAGE_RUNTIME_METHOD
_POLICY_METHODS = POLICY_RUNTIME_METHODS
_JOINT_METHODS = JOINT_RUNTIME_METHODS
_RPC_METHODS = RPC_JOINT_METHODS
_TIME_RESERVE_METHODS = TIME_RESERVE_JOINT_METHODS
_ACTOR_RESERVE_METHODS = ACTOR_RESERVE_JOINT_METHODS
_SETTLEMENT_REVIEW_METHODS = SETTLEMENT_REVIEW_JOINT_METHODS
_PLANNED_METHODS = PLANNED_JOINT_METHODS | BOUNDED_TRACEDP_DEV_METHODS
_TIME_AWARE_SELECTOR_METHODS = (
    BOUNDED_TRACEDP_DEV_METHODS | RUNTIME_OPTION_CHAIN_METHODS
)
_ROTATING_SOURCE_METHODS = ROTATING_SOURCE_JOINT_METHODS
_BUDGET_METHODS = INTERVENTION_BUDGET_METHODS
_DYNAMIC_DECISION_SCHEMA = "dynamic_tracedp_runtime_decision_v1"
_CTRF_SOURCE = "verifier/ctrf.json"
_SUPPORTED_METHODS = SUPPORTED_RUNTIME_METHODS
_PUBLIC_PROJECTION_V1 = "portable_runtime_public_projection_v1"
_PUBLIC_PROJECTION_V2 = "portable_runtime_public_projection_v2"
_PUBLIC_PROJECTION_V3 = "portable_runtime_public_projection_v3"
_PUBLIC_PROJECTION_V4 = "portable_runtime_public_projection_v4"
_PUBLIC_PROJECTION_V5 = "portable_runtime_public_projection_v5"
_PUBLIC_PROJECTION_V6 = "portable_runtime_public_projection_v6"
_ACTOR_EXECUTION_HEAD_CHARACTERS = 256
_ACTOR_EXECUTION_TAIL_CHARACTERS = 1024
_ACTOR_EXECUTION_WINDOW = 4
_ACTOR_EXECUTION_SUMMARY_KEYS = {
    "content_head",
    "content_tail",
    "content_length_chars",
    "content_truncated",
    "is_error",
    "environment_step_executed",
    "return_code",
    "requested_timeout_seconds",
    "effective_timeout_seconds",
    "process_group_containment",
}
_MISSING_PREDICTION_FALLBACK_RULE_IDS = frozenset(
    {
        "bounded_dp_missing_prediction_continue_v1",
        "bounded_dp_v4_missing_prediction_continue_v1",
        "bounded_dp_v6_missing_prediction_continue_v1",
        "bounded_dp_v7_missing_prediction_continue_v1",
    }
)
_PLANNER_NO_CALL_RESERVE_RULE_IDS = frozenset(
    {
        "bounded_dp_v7_precall_reserve_gate_continue_v1",
        "bounded_dp_v8_precall_reserve_continue_v1",
    }
)
_PLANNER_SINGLE_CANDIDATE_RULE_IDS = frozenset(
    {"bounded_dp_v8_single_candidate_deterministic_continue_v1"}
)
_PLANNER_EVENT_SUPPRESSED_RULE_IDS = frozenset(
    {"bounded_dp_v8_event_gate_suppressed_continue_v1"}
)
_PLANNER_CONTRACT_INVALID_RULE_IDS = frozenset(
    {"bounded_dp_v8_contract_invalid_prediction_fallback_continue_v1"}
)
_PLANNER_CUMULATIVE_BUDGET_RULE_IDS = frozenset(
    {"bounded_dp_v8_cumulative_planner_budget_exhausted_continue_v1"}
)
_MANAGED_AGENT_IMPORT = "lh_platform.benchmarks.harbor_opencode_go_pi:HostManagedOpenCodeGoPi"
# Harbor may launch the managed host directly or its historical-continuation subclass,
# which replays a frozen failure prefix first; both record the base identity at runtime.
_MANAGED_AGENT_LAUNCH_IMPORTS = frozenset(
    {
        _MANAGED_AGENT_IMPORT,
        "lh_platform.benchmarks.harbor_opencode_go_pi:HostTerminalHistoricalManagedOpenCodeGoPi",
    }
)
_RESOURCE_OBSERVATION_KEYS = {
    "resource_id",
    "kind",
    "observed",
    "exists",
    "size_bytes",
    "modified_time",
    "modified_time_detail",
    "change_time",
    "change_time_detail",
    "inode",
    "file_type",
    "mode",
    "group_id",
}
_DIRECTORY_SUMMARY_KEYS = {
    "descendant_entry_count",
    "descendant_regular_file_count",
    "descendant_total_file_bytes",
    "descendant_latest_modified_time",
    "descendant_latest_change_time",
    "descendant_observation_complete",
}
_RESOURCE_OBSERVATION_KEY_SETS = {
    frozenset(_RESOURCE_OBSERVATION_KEYS),
    frozenset(_RESOURCE_OBSERVATION_KEYS | _DIRECTORY_SUMMARY_KEYS),
}
_BASH_CONTAINMENT_KEYS = {
    "mode",
    "supported",
    "bash_version",
    "setsid_version",
    "kill_after_seconds",
    "outer_timeout_margin_seconds",
    "process_table_source",
    "scope",
}
_BASH_CONTAINMENT_V4_KEYS = _BASH_CONTAINMENT_KEYS | {"external_cancellation"}
_BASH_CONTAINMENT_V5_KEYS = _BASH_CONTAINMENT_V4_KEYS | {"pipeline_exit_status"}
_BASH_CONTAINMENT_V5_MODE = "timeout_process_group_procfs_pipefail_v5"
_BASH_PIPEFAIL_V1_MODE = "bash_pipefail_v1"
_BASH_PIPEFAIL_ERRTRAP_V2_MODE = "bash_pipefail_errtrap_v2"


def _validate_bash_containment_identity(
    *,
    runtime_config: Mapping[str, object],
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    """Bind the image preflight to final metadata and every Bash tool receipt."""

    capability = runtime_config.get("bash_process_group_containment")
    if (
        not isinstance(capability, dict)
        or (
            capability.get("mode") == "explicit_process_group_procfs_v3"
            and set(capability) != _BASH_CONTAINMENT_KEYS
        )
        or (
            capability.get("mode") == "timeout_process_group_procfs_v4"
            and set(capability) != _BASH_CONTAINMENT_V4_KEYS
        )
        or (
            capability.get("mode") == _BASH_CONTAINMENT_V5_MODE
            and set(capability) != _BASH_CONTAINMENT_V5_KEYS
        )
        or capability.get("mode")
        not in {
            "explicit_process_group_procfs_v3",
            "timeout_process_group_procfs_v4",
            _BASH_CONTAINMENT_V5_MODE,
        }
        or capability.get("supported") is not True
        or not isinstance(capability.get("bash_version"), str)
        or not str(capability["bash_version"]).startswith("GNU bash, version ")
        or not isinstance(capability.get("setsid_version"), str)
        or not str(capability["setsid_version"]).startswith("setsid from util-linux ")
        or capability.get("kill_after_seconds") != 5
        or capability.get("outer_timeout_margin_seconds") != 30
        or capability.get("process_table_source") != "linux_proc_stat"
        or capability.get("scope") != "over_budget_same_process_group_descendants"
        or (
            capability.get("mode")
            in {"timeout_process_group_procfs_v4", _BASH_CONTAINMENT_V5_MODE}
            and capability.get("external_cancellation")
            != "unsupported_session_becomes_non_resumable"
        )
        or (
            capability.get("mode") == _BASH_CONTAINMENT_V5_MODE
            and capability.get("pipeline_exit_status")
            != "gnu_bash_rightmost_nonzero_v1"
        )
        or metadata.get("bash_process_group_containment") != capability
    ):
        raise ValueError("managed Bash containment capability identity differs")
    return capability


def _runtime_check_execution_wrapper(runtime_config: Mapping[str, object]) -> str:
    """Derive the registered-check wrapper from the frozen Bash capability."""

    capability = runtime_config.get("bash_process_group_containment")
    if not isinstance(capability, Mapping):
        raise ValueError("managed Bash containment capability is missing")
    mode = capability.get("mode")
    if mode == _BASH_CONTAINMENT_V5_MODE:
        return _BASH_PIPEFAIL_ERRTRAP_V2_MODE
    if mode in {"explicit_process_group_procfs_v3", "timeout_process_group_procfs_v4"}:
        return _BASH_PIPEFAIL_V1_MODE
    raise ValueError("managed Bash containment capability mode is unsupported")


@dataclass(frozen=True)
class HarborCTRFEvaluationValidity:
    """Carry explicit evidence that official tests ran on usable infrastructure."""

    source_artifact: str
    trial_name: str
    official_tests_started: bool
    evaluation_infrastructure_succeeded: bool
    test_count: int
    passed_count: int
    failed_count: int
    skipped_count: int
    pending_count: int
    other_count: int
    invalid_reason: str | None

    @property
    def is_valid(self) -> bool:
        """Require both execution and infrastructure evidence before scoring."""

        return (
            self.official_tests_started
            and self.evaluation_infrastructure_succeeded
            and self.invalid_reason is None
        )

    def as_receipt(self) -> dict[str, object]:
        """Expose the exact validity evidence in the persisted evaluation receipt."""

        return asdict(self)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path.name}")
    return value


def _write_object(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _invalid_ctrf(*, trial_name: str, reason: str) -> HarborCTRFEvaluationValidity:
    return HarborCTRFEvaluationValidity(
        source_artifact=_CTRF_SOURCE,
        trial_name=trial_name,
        official_tests_started=False,
        evaluation_infrastructure_succeeded=False,
        test_count=0,
        passed_count=0,
        failed_count=0,
        skipped_count=0,
        pending_count=0,
        other_count=0,
        invalid_reason=reason,
    )


def _parse_phase_timestamp(value: object) -> float | None:
    """Parse Harbor's timezone-aware ISO phase timestamps for CTRF containment."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def _observed_pytest_nodes(tests: list[object]) -> tuple[str, ...] | None:
    """Normalize CTRF paths while preserving record multiplicity and parameters."""

    observed: list[str] = []
    for record in tests:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            return None
        parts = record["name"].split("::")
        if len(parts) < 2:
            return None
        parts[0] = Path(parts[0]).name
        observed.append("::".join(parts))
    return tuple(observed)


def _verifier_infrastructure_failure(tests: list[object]) -> str | None:
    """Detect failures that occur before the verifier can inspect the candidate."""

    for record in tests:
        if not isinstance(record, dict) or record.get("status") != "failed":
            continue
        trace = record.get("trace")
        if (
            isinstance(trace, str)
            and "selenium.common.exceptions.SessionNotCreatedException" in trace
            and "webdriver.Chrome(" in trace
        ):
            return "verifier_browser_session_not_created"
    return None


def load_harbor_ctrf_evaluation_validity(
    trial: Path,
) -> HarborCTRFEvaluationValidity:
    """Translate Harbor's CTRF artifact into explicit scientific-validity evidence."""

    try:
        result = _read_object(trial / "result.json")
        trial_config = _read_object(trial / "config.json")
    except (json.JSONDecodeError, OSError, ValueError):
        return _invalid_ctrf(trial_name=trial.name, reason="trial_identity_unreadable")
    embedded_config = result.get("config")
    if (
        result.get("trial_name") != trial.name
        or trial_config.get("trial_name") != trial.name
        or not isinstance(embedded_config, dict)
        or embedded_config.get("trial_name") != trial.name
    ):
        return _invalid_ctrf(trial_name=trial.name, reason="trial_identity_mismatch")
    verifier_phase = result.get("verifier")
    phase_start = (
        _parse_phase_timestamp(verifier_phase.get("started_at"))
        if isinstance(verifier_phase, dict)
        else None
    )
    phase_stop = (
        _parse_phase_timestamp(verifier_phase.get("finished_at"))
        if isinstance(verifier_phase, dict)
        else None
    )
    if phase_start is None or phase_stop is None or phase_stop < phase_start:
        return _invalid_ctrf(trial_name=trial.name, reason="verifier_phase_timing_invalid")

    path = trial / _CTRF_SOURCE
    if not path.is_file():
        return _invalid_ctrf(trial_name=trial.name, reason="ctrf_missing")
    try:
        ctrf = _read_object(path)
    except (json.JSONDecodeError, OSError, ValueError):
        return _invalid_ctrf(trial_name=trial.name, reason="ctrf_unreadable")
    results = ctrf.get("results")
    summary = results.get("summary") if isinstance(results, dict) else None
    tests = results.get("tests") if isinstance(results, dict) else None
    if not isinstance(summary, dict) or not isinstance(tests, list):
        return _invalid_ctrf(trial_name=trial.name, reason="ctrf_results_missing")

    count_names = ("tests", "passed", "failed", "skipped", "pending", "other")
    counts: dict[str, int] = {}
    for name in count_names:
        value = summary.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return _invalid_ctrf(trial_name=trial.name, reason="ctrf_summary_invalid")
        counts[name] = value
    started = summary.get("start")
    stopped = summary.get("stop")
    if (
        not isinstance(started, (int, float))
        or isinstance(started, bool)
        or not isinstance(stopped, (int, float))
        or isinstance(stopped, bool)
        or not math.isfinite(float(started))
        or not math.isfinite(float(stopped))
        or stopped < started
    ):
        return _invalid_ctrf(trial_name=trial.name, reason="ctrf_timing_invalid")
    # Harbor records phase timestamps on the host while pytest records CTRF
    # timestamps inside the Docker VM.  Their absolute wall clocks can differ;
    # elapsed intervals remain comparable when that offset is constant.
    if stopped - started > phase_stop - phase_start:
        return _invalid_ctrf(
            trial_name=trial.name,
            reason="ctrf_duration_exceeds_verifier_phase",
        )
    if counts["tests"] < 1:
        return _invalid_ctrf(trial_name=trial.name, reason="official_tests_not_started")
    if len(tests) != counts["tests"]:
        return _invalid_ctrf(trial_name=trial.name, reason="ctrf_test_records_incomplete")
    if sum(counts[name] for name in count_names[1:]) != counts["tests"]:
        return _invalid_ctrf(trial_name=trial.name, reason="ctrf_summary_inconsistent")
    if counts["passed"] + counts["failed"] < 1:
        return _invalid_ctrf(trial_name=trial.name, reason="official_tests_not_completed")
    if any(counts[name] for name in ("skipped", "pending", "other")):
        return _invalid_ctrf(trial_name=trial.name, reason="official_tests_incomplete")

    observed_counts = {name: 0 for name in count_names[1:]}
    for record in tests:
        if not isinstance(record, dict) or not isinstance(record.get("status"), str):
            return _invalid_ctrf(trial_name=trial.name, reason="ctrf_test_record_invalid")
        status = record["status"]
        observed_counts[status if status in observed_counts else "other"] += 1
    if any(observed_counts[name] != counts[name] for name in observed_counts):
        return _invalid_ctrf(trial_name=trial.name, reason="ctrf_test_records_inconsistent")
    embedded_task = embedded_config.get("task") if isinstance(embedded_config, dict) else None
    task_path = embedded_task.get("path") if isinstance(embedded_task, dict) else None
    expected_nodes = (
        load_trial_pytest_nodes(trial, Path(task_path)) if isinstance(task_path, str) else None
    )
    observed_nodes = _observed_pytest_nodes(tests)
    if expected_nodes is None:
        return _invalid_ctrf(
            trial_name=trial.name,
            reason="official_test_contract_unverified",
        )
    if observed_nodes is None:
        return _invalid_ctrf(
            trial_name=trial.name,
            reason="ctrf_test_record_invalid",
        )
    if len(observed_nodes) != len(set(observed_nodes)):
        return _invalid_ctrf(
            trial_name=trial.name,
            reason="official_test_contract_duplicated",
        )
    if set(observed_nodes) != expected_nodes:
        return _invalid_ctrf(
            trial_name=trial.name,
            reason="official_test_contract_incomplete",
        )
    infrastructure_failure = _verifier_infrastructure_failure(tests)
    if infrastructure_failure is not None:
        return HarborCTRFEvaluationValidity(
            source_artifact=_CTRF_SOURCE,
            trial_name=trial.name,
            official_tests_started=True,
            evaluation_infrastructure_succeeded=False,
            test_count=counts["tests"],
            passed_count=counts["passed"],
            failed_count=counts["failed"],
            skipped_count=counts["skipped"],
            pending_count=counts["pending"],
            other_count=counts["other"],
            invalid_reason=infrastructure_failure,
        )
    return HarborCTRFEvaluationValidity(
        source_artifact=_CTRF_SOURCE,
        trial_name=trial.name,
        official_tests_started=True,
        evaluation_infrastructure_succeeded=True,
        test_count=counts["tests"],
        passed_count=counts["passed"],
        failed_count=counts["failed"],
        skipped_count=counts["skipped"],
        pending_count=counts["pending"],
        other_count=counts["other"],
        invalid_reason=None,
    )


def _execution_summary_from_actor_receipt(row: Mapping[str, object]) -> dict[str, object]:
    """Rebuild bounded evidence from the exact content returned before supervision."""

    result = row.get("tool_result")
    if not isinstance(result, Mapping):
        raise ValueError("Actor execution summary lacks a delegated tool result")
    content = result.get("content")
    details = result.get("details")
    boundary = row.get("pre_supervision_content_chars")
    if (
        not isinstance(content, str)
        or not isinstance(details, Mapping)
        or type(result.get("is_error")) is not bool
        or type(result.get("environment_step_executed")) is not bool
        or type(boundary) is not int
        or not 0 <= boundary <= len(content)
    ):
        raise ValueError("Actor execution summary source receipt is invalid")
    source_content = content[:boundary]
    content_head = source_content[:_ACTOR_EXECUTION_HEAD_CHARACTERS]
    remaining = source_content[len(content_head):]
    content_tail = (
        remaining
        if len(remaining) <= _ACTOR_EXECUTION_TAIL_CHARACTERS
        else source_content[-_ACTOR_EXECUTION_TAIL_CHARACTERS:]
    )

    def integer_detail(name: str) -> int | None:
        """Match the Gateway's strict integer admission without coercion."""

        value = details.get(name)
        return value if type(value) is int else None

    containment = details.get("process_group_containment")
    return {
        "content_head": content_head,
        "content_tail": content_tail,
        "content_length_chars": len(source_content),
        "content_truncated": (
            len(content_head) + len(content_tail) < len(source_content)
        ),
        "is_error": result["is_error"],
        "environment_step_executed": result["environment_step_executed"],
        "return_code": integer_detail("return_code"),
        "requested_timeout_seconds": integer_detail(
            "requested_command_timeout_seconds"
        ),
        "effective_timeout_seconds": integer_detail(
            "effective_command_timeout_seconds"
        ),
        "process_group_containment": (
            containment if isinstance(containment, str) and containment else None
        ),
    }


def _kernel_execution_summary_payload(value: object) -> dict[str, object] | None:
    """Project the replayed dataclass without using the Kernel serializer."""

    if value is None:
        return None
    return {key: getattr(value, key) for key in _ACTOR_EXECUTION_SUMMARY_KEYS}


def _validate_actor_execution_summaries(
    *,
    trial: Path,
    kernel: PortableRuntimeKernel,
    recorded: Mapping[str, object],
) -> None:
    """Join each V6 summary to its terminal pre-supervision result independently."""

    finish_payloads: dict[str, Mapping[str, object]] = {}
    for row in _read_jsonl(trial / "agent/runtime-events.jsonl"):
        payload = row.get("payload")
        if row.get("event_type") != "action_finished" or not isinstance(
            payload, Mapping
        ):
            continue
        action_id = payload.get("action_id")
        if not isinstance(action_id, str) or action_id in finish_payloads:
            raise ValueError("Actor execution summary event identity is invalid")
        finish_payloads[action_id] = payload

    observer_path = trial / "agent/terminal-public-state.jsonl"
    observer_rows = _read_jsonl(observer_path) if observer_path.is_file() else []
    receipts: dict[str, Mapping[str, object]] = {}
    for row in observer_rows:
        action_id = row.get("tool_call_id")
        if not isinstance(action_id, str) or action_id in receipts:
            raise ValueError("Actor execution summary receipt identity is invalid")
        receipts[action_id] = row

    expected_window = []
    for action in kernel.actions.values():
        if action.actor != "actor":
            continue
        finish_payload = finish_payloads.get(action.action_id)
        if finish_payload is None:
            if action.execution_summary is not None:
                raise ValueError("unfinished Actor action contains an execution summary")
            continue
        receipt = receipts.get(action.action_id)
        tool_result = receipt.get("tool_result") if receipt is not None else None
        event_summary = finish_payload.get("execution_summary")
        kernel_summary = _kernel_execution_summary_payload(action.execution_summary)
        if tool_result is None:
            if event_summary is not None or kernel_summary is not None:
                raise ValueError("Actor execution summary lacks a source tool result")
            continue
        if receipt is None:
            raise ValueError("Actor execution summary lacks a terminal receipt")
        rebuilt = _execution_summary_from_actor_receipt(receipt)
        if (
            not isinstance(event_summary, Mapping)
            or set(event_summary) != _ACTOR_EXECUTION_SUMMARY_KEYS
            or dict(event_summary) != rebuilt
            or kernel_summary != rebuilt
        ):
            raise ValueError("Actor execution summary differs from its source receipt")
        expected_window.append(
            {
                "action_id": action.action_id,
                "tool_name": action.tool_name,
                "phase": action.phase.value,
                "execution_summary": rebuilt,
            }
        )
    projected = recorded.get("recent_actor_execution_summaries")
    if projected != expected_window[-_ACTOR_EXECUTION_WINDOW:]:
        raise ValueError("actor-final execution summary window differs")


def _validate_actor_final_state(
    recorded: dict[str, object],
    kernel: PortableRuntimeKernel,
    *,
    trial: Path,
    allow_legacy_projection: bool,
) -> None:
    """Validate the actor-close snapshot independently of later evaluation events."""

    if kernel.official_evaluation is None:
        expected = kernel.public_projection()
    else:
        # Recompute the pre-evaluation snapshot with the Kernel's own state rules.
        official_evaluation = kernel.official_evaluation
        evaluated_statuses = dict(kernel.task_statuses)
        kernel.official_evaluation = None
        try:
            kernel._refresh_task_statuses()
            expected = kernel.public_projection()
        finally:
            kernel.official_evaluation = official_evaluation
            kernel.task_statuses = evaluated_statuses

    recorded_schema = recorded.get("schema_version")
    if recorded_schema == _PUBLIC_PROJECTION_V5:
        historical_expected = dict(expected)
        historical_expected["schema_version"] = _PUBLIC_PROJECTION_V5
        historical_expected.pop("recent_actor_execution_summaries")
        if recorded != historical_expected:
            raise ValueError("actor-final runtime state differs from its event log")
    elif recorded_schema in {
        _PUBLIC_PROJECTION_V1,
        _PUBLIC_PROJECTION_V2,
        _PUBLIC_PROJECTION_V3,
        _PUBLIC_PROJECTION_V4,
    }:
        if not allow_legacy_projection:
            raise ValueError("legacy actor-final projection requires an explicit flag")
        legacy_expected = dict(expected)
        legacy_expected["schema_version"] = recorded_schema
        legacy_expected.pop("task_revisions")
        legacy_evidence = [dict(item) for item in legacy_expected["evidence"]]
        for item in legacy_evidence:
            item.pop("dependency_claims")
            item.pop("dependency_claims_version_bound")
        legacy_expected["evidence"] = legacy_evidence
        if recorded_schema in {_PUBLIC_PROJECTION_V3, _PUBLIC_PROJECTION_V4}:
            legacy_task_nodes = [dict(item) for item in legacy_expected["task_nodes"]]
            for node in legacy_task_nodes:
                node.pop("required_output_resource_ids", None)
                node.pop("output_requirement", None)
            legacy_expected["task_nodes"] = legacy_task_nodes
        if recorded_schema in {_PUBLIC_PROJECTION_V1, _PUBLIC_PROJECTION_V2}:
            legacy_expected.pop("task_nodes")
            legacy_expected.pop("runtime_supervision_disabled_reason")
            legacy_expected.pop("failed_runtime_intervention_versions")
        if recorded_schema == _PUBLIC_PROJECTION_V1:
            legacy_expected.pop("actor_session_closed")
            legacy_expected.pop("official_evaluation")
        if recorded_schema in {_PUBLIC_PROJECTION_V1, _PUBLIC_PROJECTION_V2}:
            # Historical v1/v2 observations predate v3 observation fields.
            legacy_resources = [dict(item) for item in legacy_expected["resources"]]
            for resource in legacy_resources:
                resource.pop("inode", None)
                resource.pop("modified_time_detail", None)
            legacy_expected["resources"] = legacy_resources
        legacy_expected.pop("recent_actor_execution_summaries")
        if recorded != legacy_expected:
            raise ValueError("actor-final runtime state differs from its event log")
    elif recorded_schema == _PUBLIC_PROJECTION_V6:
        _validate_actor_execution_summaries(
            trial=trial,
            kernel=kernel,
            recorded=recorded,
        )
        if recorded != expected:
            raise ValueError("actor-final runtime state differs from its event log")
    else:
        raise ValueError("actor-final runtime state differs from its event log")


def _nested_mapping(value: Mapping[str, object], key: str, *, label: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"{label} is missing")
    return nested


def _validate_resource_and_retention_identity(
    *,
    trial: Path,
    runtime_config: Mapping[str, object],
    agent_kwargs: Mapping[str, object],
    standalone_kwargs: Mapping[str, object],
    metadata: Mapping[str, object],
    kernel: PortableRuntimeKernel,
) -> None:
    """Bind declared resources and retained bytes across every runtime artifact."""

    public_paths = runtime_config.get("public_paths")
    budget = runtime_config.get("retention_budget_bytes")
    runtime_resources = runtime_config.get("resources")
    if (
        not isinstance(public_paths, list)
        or not all(isinstance(path, str) and path for path in public_paths)
        or len(public_paths) != len(set(public_paths))
        or not isinstance(budget, int)
        or isinstance(budget, bool)
        or budget < 1
        or not isinstance(runtime_resources, list)
        or not runtime_resources
    ):
        raise ValueError("managed runtime resource configuration is invalid")
    discovery_schema = runtime_config.get("database_sidecar_discovery")
    if discovery_schema not in {
        None,
        "sqlite_conventional_sidecars_v1",
        "sqlite_conventional_sidecars_v2",
    }:
        raise ValueError("managed runtime database discovery identity is invalid")
    discovery_enabled = discovery_schema is not None
    for kwargs in (agent_kwargs, standalone_kwargs):
        if (
            kwargs.get("retention_budget_bytes") != budget
            or not isinstance(kwargs.get("resource_specs"), list)
            or kwargs.get("discover_database_sidecars", False) is not discovery_enabled
        ):
            raise ValueError("managed runtime resource configuration differs across artifacts")

    keys = ("resource_id", "kind", "provenance", "group_id")
    normalized_runtime: list[dict[str, object]] = []
    for row in runtime_resources:
        if not isinstance(row, dict):
            raise ValueError("managed runtime resource record is invalid")
        normalized_runtime.append({key: row.get(key) for key in keys})
    for kwargs in (agent_kwargs, standalone_kwargs):
        raw_specs = kwargs["resource_specs"]
        parsed_specs = []
        for row in raw_specs:
            if not isinstance(row, dict) or set(row) != set(keys):
                raise ValueError("managed runtime resource declaration is invalid")
            parsed_specs.append(
                ResourceSpec(
                    resource_id=str(row["resource_id"]),
                    kind=ResourceKind(row["kind"]),
                    provenance=str(row["provenance"]),
                    group_id=(str(row["group_id"]) if row["group_id"] is not None else None),
                )
            )
        effective_specs = (
            expand_sqlite_resource_family_specs(parsed_specs)
            if discovery_enabled
            else tuple(parsed_specs)
        )
        normalized_specs = [
            {
                "resource_id": item.resource_id,
                "kind": item.kind.value,
                "provenance": item.provenance,
                "group_id": item.group_id,
            }
            for item in effective_specs
        ]
        if normalized_specs != normalized_runtime:
            raise ValueError("managed runtime resource declarations differ")
        expected_public_paths = list(kwargs.get("public_paths", []))
        if discovery_enabled:
            expected_public_paths = list(
                dict.fromkeys(
                    (*expected_public_paths, *(item.resource_id for item in effective_specs))
                )
            )
        if expected_public_paths != public_paths:
            raise ValueError("managed runtime resource configuration differs across artifacts")

    kernel_resources = {
        resource_id: observation for resource_id, observation in kernel.observations.items()
    }
    declared_resource_ids = [row["resource_id"] for row in normalized_runtime]
    if (
        not all(
            isinstance(resource_id, str) and resource_id for resource_id in declared_resource_ids
        )
        or len(declared_resource_ids) != len(set(declared_resource_ids))
        or set(declared_resource_ids) != set(kernel_resources)
    ):
        raise ValueError("managed runtime resources differ from observed runtime state")
    for row in normalized_runtime:
        resource_id = row["resource_id"]
        observation = kernel_resources.get(resource_id)
        if (
            not isinstance(resource_id, str)
            or observation is None
            or row["kind"] != observation.kind.value
            or row["group_id"] != observation.group_id
            or not isinstance(row["provenance"], str)
            or not row["provenance"]
        ):
            raise ValueError("managed runtime resource differs from Kernel observation")

    receipt_path = trial / "agent/runtime-retention.jsonl"
    receipts = _read_jsonl(receipt_path)
    if len(receipts) != len(runtime_resources):
        raise ValueError("managed runtime retention does not cover every resource")
    visibility = runtime_config.get("managed_state_visibility")
    private_root_value = runtime_config.get("managed_private_artifact_root")
    if visibility is None and private_root_value is None:
        retention_storage_root = trial / "agent"
        require_exported_copy = False
    elif (
        visibility == "host_private_until_actor_exit_v1"
        and isinstance(private_root_value, str)
        and Path(private_root_value).is_absolute()
        and not Path(private_root_value).resolve().is_relative_to(trial.resolve())
    ):
        retention_storage_root = Path(private_root_value).resolve()
        require_exported_copy = True
    else:
        raise ValueError("managed runtime private-state boundary differs")
    runtime_events = _read_jsonl(trial / "agent/runtime-events.jsonl")
    initialized_rows = [row for row in runtime_events if row.get("event_type") == "initialized"]
    if len(initialized_rows) != 1:
        raise ValueError("managed runtime initial observation boundary is missing")
    initial_payload = initialized_rows[0].get("payload")
    initial_rows = (
        initial_payload.get("observations") if isinstance(initial_payload, dict) else None
    )
    if not isinstance(initial_rows, list):
        raise ValueError("managed runtime initial observations are missing")
    initial_by_resource = {
        row.get("resource_id"): row for row in initial_rows if isinstance(row, dict)
    }
    if len(initial_by_resource) != len(initial_rows):
        raise ValueError("managed runtime initial observation identities are invalid")

    retained_total = 0
    for index, (resource, receipt) in enumerate(zip(runtime_resources, receipts, strict=True)):
        status = receipt.get("status")
        retained = receipt.get("bytes_retained")
        if (
            receipt.get("schema_version") != "terminal_resource_retention_v4"
            or receipt.get("resource_id") != resource.get("resource_id")
            or status != resource.get("retention_status")
            or status not in {"full_bytes", "metadata_only", "missing", "failed"}
            or not isinstance(retained, int)
            or isinstance(retained, bool)
            or retained < 0
        ):
            raise ValueError("managed runtime retention receipt differs from runtime-config")
        source_before = receipt.get("source_before")
        source_after_download = receipt.get("source_after_download")
        kernel_initial_observation = receipt.get("kernel_initial_observation")
        if (
            not isinstance(source_before, dict)
            or frozenset(source_before) not in _RESOURCE_OBSERVATION_KEY_SETS
            or source_before.get("resource_id") != resource.get("resource_id")
            or source_before.get("kind") != resource.get("kind")
            or type(source_before.get("observed")) is not bool
            or (
                source_after_download is not None
                and (
                    not isinstance(source_after_download, dict)
                    or frozenset(source_after_download)
                    not in _RESOURCE_OBSERVATION_KEY_SETS
                    or source_after_download.get("resource_id") != resource.get("resource_id")
                    or source_after_download.get("kind") != resource.get("kind")
                )
            )
            or (
                not isinstance(kernel_initial_observation, dict)
                or frozenset(kernel_initial_observation)
                not in _RESOURCE_OBSERVATION_KEY_SETS
                or kernel_initial_observation.get("resource_id") != resource.get("resource_id")
                or kernel_initial_observation.get("kind") != resource.get("kind")
            )
        ):
            raise ValueError("managed runtime retention source identity is incomplete")
        if status == "full_bytes" and (
            source_after_download != source_before
            or kernel_initial_observation != source_before
            or source_before.get("exists") is not True
            or source_before.get("file_type") != "regular file"
            or source_before.get("size_bytes") != retained
            or type(source_before.get("modified_time")) is not int
            or not isinstance(source_before.get("modified_time_detail"), str)
            or type(source_before.get("change_time")) is not int
            or not isinstance(source_before.get("change_time_detail"), str)
            or type(source_before.get("inode")) is not int
            or not isinstance(source_before.get("mode"), str)
        ):
            raise ValueError("managed runtime retained bytes lack a stable source boundary")
        if kernel_initial_observation != initial_by_resource.get(resource.get("resource_id")):
            raise ValueError("managed runtime retention differs from Kernel initialization")
        retained_total += retained
        storage_path = receipt.get("storage_path")
        if status == "full_bytes":
            expected_path = (
                retention_storage_root / "runtime-retention" / f"resource-{index:04d}.bin"
            )
            stat = expected_path.stat() if expected_path.is_file() else None
            if (
                storage_path != str(expected_path)
                or stat is None
                or stat.st_size != retained
                or stat.st_mtime_ns != receipt.get("storage_modified_time_ns")
                or stat.st_ctime_ns != receipt.get("storage_change_time_ns")
                or stat.st_ino != receipt.get("storage_inode")
            ):
                raise ValueError("managed runtime retained bytes are unavailable")
            exported_path = trial / "agent/runtime-retention" / f"resource-{index:04d}.bin"
            if require_exported_copy and (
                not exported_path.is_file() or exported_path.stat().st_size != retained
            ):
                raise ValueError("managed runtime retained bytes were not published")
        elif storage_path is not None or retained != 0:
            raise ValueError("managed runtime metadata-only retention names bytes")
    if (
        retained_total > budget
        or metadata.get("runtime_retained_bytes") != retained_total
        or metadata.get("runtime_resource_count") != len(runtime_resources)
    ):
        raise ValueError("managed runtime retention totals differ across artifacts")


def _validate_actor_session_events(
    *,
    trial: Path,
    runtime_config: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """Derive Pi resume claims from the durable attempt lifecycle log."""

    if (
        runtime_config.get("actor_session_event_schema")
        != "terminal_managed_actor_session_event_v1"
    ):
        raise ValueError("managed Actor session event schema differs")
    actor_session_id = runtime_config["actor_session_id"]
    environment_session_id = runtime_config["environment_session_id"]
    rows = _read_jsonl(trial / "agent/actor-session-events.jsonl")
    open_attempt: int | None = None
    attempt_count = 0
    completed = False
    for sequence, row in enumerate(rows):
        payload = row.get("payload")
        event_type = row.get("event_type")
        if (
            row.get("schema_version") != "terminal_managed_actor_session_event_v1"
            or row.get("sequence") != sequence
            or not isinstance(payload, dict)
            or payload.get("actor_session_id") != actor_session_id
            or payload.get("environment_session_id") != environment_session_id
        ):
            raise ValueError("managed Actor session event identity differs")
        attempt_index = payload.get("attempt_index")
        if event_type == "actor_attempt_started":
            if completed or open_attempt is not None or attempt_index != attempt_count:
                raise ValueError("managed Actor attempt start sequence differs")
            open_attempt = attempt_count
            attempt_count += 1
        elif event_type in {
            "actor_attempt_interrupted",
            "actor_attempt_completed",
            "actor_session_budget_exhausted",
        }:
            if open_attempt is None or attempt_index != open_attempt:
                raise ValueError("managed Actor attempt terminal event differs")
            open_attempt = None
            if event_type in {
                "actor_attempt_completed",
                "actor_session_budget_exhausted",
            }:
                completed = True
        else:
            raise ValueError("managed Actor session event type is unsupported")
    if open_attempt is not None or not completed or attempt_count < 1:
        raise ValueError("managed Actor session lifecycle is incomplete")
    resume_count = attempt_count - 1
    # An isolated guidance trial closed at its own bound (timeout, cancellation) leaves an
    # incomplete usage receipt without an Actor session event; the Actor records that as
    # incomplete accounting, so the durable receipts must count here too.
    guidance_path = trial / "agent/isolated-guidance-usage.jsonl"
    guidance_usage = _read_jsonl(guidance_path) if guidance_path.is_file() else []
    if list(metadata.get("isolated_guidance_usage") or []) != guidance_usage:
        raise ValueError("isolated guidance usage metadata differs from durable receipts")
    actor_accounting_complete = not any(
        row.get("event_type") in {"actor_attempt_interrupted", "actor_session_budget_exhausted"}
        for row in rows
    ) and all(row.get("token_accounting_complete") is True for row in guidance_usage)
    method = runtime_config.get("runtime_method")
    planner_accounting = metadata.get("runtime_planner_accounting")
    if method in BOUNDED_TRACEDP_DEV_METHODS:
        if not isinstance(planner_accounting, dict) or not isinstance(
            planner_accounting.get("token_accounting_complete"), bool
        ):
            raise ValueError("managed runtime planner accounting metadata is missing")
        token_accounting_complete = (
            actor_accounting_complete and planner_accounting["token_accounting_complete"]
        )
        token_accounting_semantics = (
            "complete_actor_and_runtime_planner_usage"
            if token_accounting_complete
            else "lower_bound_with_unknown_actor_or_runtime_planner_usage"
        )
    else:
        token_accounting_complete = actor_accounting_complete
        token_accounting_semantics = (
            "complete_message_end_usage"
            if token_accounting_complete
            else "lower_bound_from_received_message_end_usage"
        )
    terminal_reason = (
        "budget_exhausted"
        if rows[-1].get("event_type") == "actor_session_budget_exhausted"
        else "natural_completion"
    )
    recorded_terminal_reason = metadata.get("actor_session_terminal_reason")
    capabilities = runtime_config.get("runtime_capabilities")
    if (
        capabilities
        != {
            "pi_subprocess_resume_in_live_environment": True,
            "harbor_host_process_resume": False,
        }
        or metadata.get("pi_subprocess_resume_in_live_environment") is not True
        or metadata.get("harbor_host_process_resume") is not False
        or metadata.get("actor_session_resume_count") != resume_count
        or metadata.get("actor_session_resumed") is not (resume_count > 0)
        or recorded_terminal_reason not in {None, terminal_reason}
        or (terminal_reason == "budget_exhausted" and recorded_terminal_reason != terminal_reason)
        or metadata.get("token_accounting_complete") is not token_accounting_complete
        or metadata.get("token_accounting_semantics") != token_accounting_semantics
    ):
        raise ValueError("managed Actor resume metadata differs from durable events")


def _validate_runtime_planner_accounting(
    *,
    trial: Path,
    runtime_config: Mapping[str, object],
    agent_kwargs: Mapping[str, object],
    standalone_kwargs: Mapping[str, object],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Independently rebuild bounded-planner identity, calls, and provider usage."""

    method = runtime_config.get("runtime_method")
    actor_runtime = runtime_config.get("actor_runtime")
    if not isinstance(actor_runtime, Mapping):
        raise ValueError("managed Actor runtime identity is missing")
    configured = actor_runtime.get("runtime_planner")
    recorded = metadata.get("runtime_planner_accounting")
    if method not in BOUNDED_TRACEDP_DEV_METHODS:
        if configured is not None or recorded is not None:
            raise ValueError("non-bounded runtime contains planner accounting")
        return {
            "model_call_count": 0,
            "admitted_prediction_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "elapsed_ms": 0,
            "token_accounting_complete": True,
            "ambiguous_started_call": False,
        }
    planner_identity_keys = {
        "provider",
        "model",
        "transport",
        "base_url",
        "maximum_output_tokens",
        "timeout_seconds",
        "thinking",
        "response_format",
    }
    if method in {
        BOUNDED_TRACEDP_DEV_V7_METHOD,
        TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
    }:
        planner_identity_keys.update(
            {
                "precall_actor_convergence_reserve_ms",
                "precall_provider_timeout_overrun_margin_ms",
                "timeout_is_hard_wall_clock",
            }
        )
    if method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD:
        planner_identity_keys.update(
            {
                "event_admission_schema",
                "cumulative_planner_budget_ms",
                "semantic_review_threshold",
            }
        )
    if not isinstance(configured, Mapping) or set(configured) != planner_identity_keys:
        raise ValueError("managed runtime planner identity differs")
    expected_kwargs = {
        "runtime_planner_base_url": configured.get("base_url"),
        "runtime_planner_max_output_tokens": configured.get("maximum_output_tokens"),
        "runtime_planner_timeout_seconds": configured.get("timeout_seconds"),
    }
    if (
        configured.get("provider") != actor_runtime.get("provider")
        or configured.get("model") != actor_runtime.get("model")
        or configured.get("transport") != "pi_authenticated_openai_chat_completion"
        or configured.get("thinking") != "off"
        or configured.get("response_format") != "json_object"
        or not isinstance(configured.get("base_url"), str)
        or not configured["base_url"]
        or type(configured.get("maximum_output_tokens")) is not int
        or configured["maximum_output_tokens"] < 1
        or type(configured.get("timeout_seconds")) is not int
        or configured["timeout_seconds"] < 1
        or (
            method in {
                BOUNDED_TRACEDP_DEV_V7_METHOD,
                TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
            }
            and (
                configured.get("precall_actor_convergence_reserve_ms")
                != ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000
                or configured.get("precall_provider_timeout_overrun_margin_ms")
                != V7_PROVIDER_TIMEOUT_OVERRUN_MARGIN_MS
                or configured.get("timeout_is_hard_wall_clock") is not False
            )
        )
        or (
            method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
            and (
                configured.get("event_admission_schema")
                != "runtime_planner_event_admission_v1"
                or configured.get("cumulative_planner_budget_ms")
                != V8_CUMULATIVE_PLANNER_BUDGET_MS
                or configured.get("semantic_review_threshold")
                != V8_SEMANTIC_REVIEW_THRESHOLD
            )
        )
        or any(
            kwargs.get(key) != value
            for kwargs in (agent_kwargs, standalone_kwargs)
            for key, value in expected_kwargs.items()
        )
    ):
        raise ValueError("managed runtime planner configuration differs")

    path = trial / "agent/runtime-public-predictions.jsonl"
    rows = _read_jsonl(path) if path.is_file() else []
    pending: Mapping[str, object] | None = None
    terminal_rows: list[Mapping[str, object]] = []
    expected_receipt_schema = {
        BOUNDED_TRACEDP_DEV_V4_METHOD: "runtime_public_prediction_receipt_v2",
        BOUNDED_TRACEDP_DEV_V6_METHOD: "runtime_public_prediction_receipt_v4",
        BOUNDED_TRACEDP_DEV_V7_METHOD: "runtime_public_prediction_receipt_v6",
        TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD: "runtime_public_prediction_receipt_v7",
    }.get(method, "runtime_public_prediction_receipt_v1")
    for row in rows:
        if row.get("schema_version") != expected_receipt_schema:
            raise ValueError("runtime planner receipt schema differs")
        status = row.get("status")
        if status == "started":
            if pending is not None:
                raise ValueError("runtime planner calls overlap")
            pending = row
            continue
        if pending is None:
            raise ValueError("runtime planner terminal receipt lacks a start")
        if status not in {
            "prediction_admitted_uncalibrated",
            "prediction_failed",
            "model_abstained",
        }:
            raise ValueError("runtime planner terminal status differs")
        for key in (
            "context",
            "prompt",
            "requested_provider",
            "requested_model",
            "configured_timeout_seconds",
            "effective_timeout_seconds",
        ):
            if row.get(key) != pending.get(key):
                raise ValueError("runtime planner call identity changed")
        if method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD and row.get(
            "event_admission"
        ) != pending.get("event_admission"):
            raise ValueError("runtime planner v8 event admission changed during call")
        context = row.get("context")
        # V7 intentionally exposes only the decision boundary and compact
        # candidate catalog. The complete executable slate stays in the
        # runtime-owned decision receipt and must never be required in the
        # provider prompt merely to validate its live clock.
        clock_source = (
            context.get("boundary")
            if method in {
                BOUNDED_TRACEDP_DEV_V7_METHOD,
                TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
            }
            and isinstance(context, Mapping)
            else context.get("slate")
            if isinstance(context, Mapping)
            else None
        )
        remaining_ms = (
            clock_source.get("actor_time_remaining_ms")
            if isinstance(clock_source, Mapping)
            else None
        )
        configured_timeout = row.get("configured_timeout_seconds")
        effective_timeout = row.get("effective_timeout_seconds")
        timeout_fields_present = configured_timeout is not None or effective_timeout is not None
        if (
            row.get("requested_provider") != configured.get("provider")
            or row.get("requested_model") != configured.get("model")
            or type(row.get("elapsed_ms")) is not int
            or row["elapsed_ms"] < 1
        ):
            raise ValueError("runtime planner receipt identity or duration differs")
        if timeout_fields_present and (
            configured_timeout != configured.get("timeout_seconds")
            or type(effective_timeout) is not int
            or type(remaining_ms) is not int
            or not 1 <= effective_timeout <= configured_timeout
            or effective_timeout * 1_000 > remaining_ms - 1_000
        ):
            raise ValueError("runtime planner timeout exceeds the live clock")
        terminal_rows.append(row)
        pending = None

    complete = pending is None
    decision_path = trial / "agent/runtime-decisions.jsonl"
    decision_rows = _read_jsonl(decision_path) if decision_path.is_file() else []
    selected_decisions = [
        row.get("payload")
        for row in decision_rows
        if row.get("event_type") == "decision_selected"
    ]
    if any(not isinstance(row, Mapping) for row in selected_decisions):
        raise ValueError("runtime planner decision record is invalid")

    def matching_decisions(context: object) -> list[Mapping[str, object]]:
        """Pair a provider call with one runtime-owned executable decision."""

        if not isinstance(context, Mapping):
            return []
        if method not in {
            BOUNDED_TRACEDP_DEV_V7_METHOD,
            TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
        }:
            slate_payload = context.get("slate")
            return [row for row in selected_decisions if row.get("slate") == slate_payload]
        boundary = context.get("boundary")
        if not isinstance(boundary, Mapping):
            return []
        matches = []
        for row in selected_decisions:
            slate_payload = row.get("slate")
            if not isinstance(slate_payload, Mapping):
                continue
            try:
                slate_object = RuntimeJointActionSlate.from_dict(slate_payload)
            except (KeyError, TypeError, ValueError):
                continue
            expected_boundary = {
                "session_id": slate_object.session_id,
                "trigger_action_id": slate_object.trigger_action_id,
                "resource_versions": [
                    list(item) for item in slate_object.resource_versions
                ],
                "intervention_budget": slate_object.budget_before,
                "actor_time_remaining_ms": slate_object.actor_time_remaining_ms,
                "no_progress_actor_actions": slate_object.no_progress_actor_actions,
            }
            if dict(boundary) == expected_boundary:
                matches.append(row)
        return matches

    if method in {
        BOUNDED_TRACEDP_DEV_V6_METHOD,
        BOUNDED_TRACEDP_DEV_V7_METHOD,
        TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
    }:
        compact = method in {
            BOUNDED_TRACEDP_DEV_V7_METHOD,
            TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
        }
        version_label = (
            "v8"
            if method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
            else "v7"
            if compact
            else "v6"
        )
        check_configuration = runtime_config.get("runtime_check_configuration")
        configured_check_contracts = (
            tuple(check_configuration.get("contracts", ()))
            if isinstance(check_configuration, Mapping)
            else ()
        )
        if not all(isinstance(item, Mapping) for item in configured_check_contracts):
            raise ValueError(f"runtime planner {version_label} check contracts differ")
        event_log_path = trial / "agent/runtime-events.jsonl"
        for receipt in terminal_rows:
            context = receipt.get("context")
            if compact:
                matches = matching_decisions(context)
                if len(matches) != 1:
                    raise ValueError("runtime planner receipt lacks one matching decision")
                slate_payload = matches[0].get("slate")
            else:
                slate_payload = (
                    context.get("slate") if isinstance(context, Mapping) else None
                )
            if not isinstance(slate_payload, Mapping):
                raise ValueError(f"runtime planner {version_label} context lacks a slate")
            slate_object = RuntimeJointActionSlate.from_dict(slate_payload)
            boundary_kernel = _kernel_at_joint_decision_boundary(
                event_log_path,
                trigger_action_id=slate_object.trigger_action_id,
            )
            expected_context = (
                public_prediction_context_v7(
                    slate_object,
                    boundary_kernel.public_projection(),
                    configured_check_contracts=configured_check_contracts,
                )
                if compact
                else public_prediction_context_v6(
                    slate_object,
                    boundary_kernel.public_projection(),
                    configured_check_contracts=configured_check_contracts,
                )
            )
            if context != expected_context:
                raise ValueError(
                    f"runtime planner {version_label} context differs from Kernel boundary"
                )
            expected_prompt = (
                prediction_prompt_v7(expected_context)
                if compact
                else prediction_prompt_v6(expected_context)
            )
            if receipt.get("prompt") != expected_prompt:
                raise ValueError(
                    f"runtime planner {version_label} prompt differs from Kernel context"
                )

        raw_path = trial / "agent/runtime-public-prediction-provider.jsonl"
        raw_rows = _read_jsonl(raw_path) if raw_path.is_file() else []
        response_receipts = [
            row for row in terminal_rows if isinstance(row.get("response"), Mapping)
        ]
        if len(raw_rows) != len(response_receipts):
            raise ValueError(
                f"runtime planner {version_label} raw provider call count differs"
            )
        response_parser = PiAuthenticatedOpenAiActor(
            executable="/dev/null",
            provider=str(configured["provider"]),
            model=str(configured["model"]),
            base_url=str(configured["base_url"]),
            max_tokens=int(configured["maximum_output_tokens"]),
            system_prompt="",
            timeout_seconds=int(configured["timeout_seconds"]),
            json_object_output=True,
        )
        for raw_row, receipt in zip(raw_rows, response_receipts, strict=True):
            raw_response = raw_row.get("response")
            latency_seconds = raw_row.get("latency_seconds")
            response = receipt["response"]
            if (
                set(raw_row) != {"response", "latency_seconds"}
                or not isinstance(raw_response, Mapping)
                or not isinstance(latency_seconds, (int, float))
                or not isinstance(response, Mapping)
            ):
                raise ValueError(
                    f"runtime planner {version_label} raw provider response differs"
                )
            try:
                rebuilt_response = asdict(
                    response_parser.response_from_raw(
                        raw_response,
                        latency_seconds=float(latency_seconds),
                    )
                )
            except (PiActorCompatibilityError, TypeError, ValueError) as error:
                raise ValueError(
                    f"runtime planner {version_label} raw provider response differs"
                ) from error
            if dict(response) != rebuilt_response:
                raise ValueError(
                    f"runtime planner {version_label} raw provider response differs"
                )
            expected_usage = {
                "input_tokens": rebuilt_response["input_tokens"],
                "output_tokens": rebuilt_response["output_tokens"],
                "reasoning_tokens": rebuilt_response["reasoning_tokens"],
                "reasoning_tokens_reported": rebuilt_response[
                    "reasoning_tokens_reported"
                ],
            }
            if receipt.get("model_usage") != expected_usage:
                raise ValueError(
                    f"runtime planner {version_label} usage differs from provider response"
                )
    matched_decision_ids: set[str] = set()
    receipt_by_decision_id: dict[str, Mapping[str, object]] = {}
    admitted_models = []
    for receipt in terminal_rows:
        context = receipt.get("context")
        matches = matching_decisions(context)
        if len(matches) != 1:
            raise ValueError("runtime planner receipt lacks one matching decision")
        decision = matches[0]
        slate = decision.get("slate")
        decision_id = decision.get("decision_id")
        if not isinstance(decision_id, str) or decision_id in matched_decision_ids:
            raise ValueError("runtime planner receipt reuses a decision")
        matched_decision_ids.add(decision_id)
        receipt_by_decision_id[decision_id] = receipt
        planning_model = receipt.get("planning_model")
        if receipt.get("status") == "prediction_admitted_uncalibrated":
            if not isinstance(planning_model, Mapping) or decision.get(
                "planning_model"
            ) != planning_model:
                raise ValueError("runtime planner prediction differs from its decision")
            if (
                method == BOUNDED_TRACEDP_DEV_V4_METHOD
                and (
                    planning_model.get("schema_version")
                    != "runtime_bounded_prediction_v2"
                    or planning_model.get("residual_acceptance_state")
                    != context.get("residual_acceptance_state")
                )
            ):
                raise ValueError("runtime planner v4 residual state differs")
            if method in {
                BOUNDED_TRACEDP_DEV_V6_METHOD,
                BOUNDED_TRACEDP_DEV_V7_METHOD,
                TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
            }:
                compact = method in {
                    BOUNDED_TRACEDP_DEV_V7_METHOD,
                    TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
                }
                version_label = (
                    "v8"
                    if method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
                    else "v7"
                    if compact
                    else "v6"
                )
                live_candidates = (
                    slate.get("candidates") if isinstance(slate, Mapping) else None
                )
                expected_catalog = (
                    [
                        {"candidate_index": index, "candidate": candidate}
                        for index, candidate in enumerate(live_candidates)
                    ]
                    if isinstance(live_candidates, list)
                    else None
                )
                if (
                    planning_model.get("schema_version")
                    != (
                        "runtime_bounded_prediction_v7"
                        if method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
                        else "runtime_bounded_prediction_v6"
                        if method == BOUNDED_TRACEDP_DEV_V7_METHOD
                        else "runtime_bounded_prediction_v4"
                    )
                    or planning_model.get("prediction_output_schema")
                    != (
                        "compact_root_verdict_terminal_aggregate_v7"
                        if compact
                        else "flat_parent_coordinates_verdict_actor_time_v6"
                    )
                    or planning_model.get("residual_acceptance_state")
                    != context.get("residual_acceptance_state")
                    or planning_model.get("candidate_catalog") != expected_catalog
                ):
                    raise ValueError(
                        f"runtime planner {version_label} normalized state differs"
                    )
                if (
                    method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
                    and planning_model.get("event_admission")
                    != receipt.get("event_admission")
                ):
                    raise ValueError("runtime planner v8 admission differs from decision")
                response = receipt.get("response")
                response_text = (
                    response.get("text") if isinstance(response, Mapping) else None
                )
                try:
                    response_forecast = (
                        json.loads(response_text)
                        if isinstance(response_text, str)
                        else None
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"runtime planner {version_label} response is not valid JSON"
                    ) from error
                if response_forecast != planning_model.get("normalized_forecast"):
                    raise ValueError(
                        f"runtime planner {version_label} normalized forecast differs from provider response"
                    )
                resolver = (
                    RuntimeEventTraceDPV8Resolver(lambda _: None)
                    if method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
                    else RuntimeBoundedTraceDPV7Resolver(lambda _: None)
                    if method == BOUNDED_TRACEDP_DEV_V7_METHOD
                    else RuntimeBoundedTraceDPV6Resolver(lambda _: None)
                )
                resolver.resolve_recorded(
                    RuntimeJointActionSlate.from_dict(slate), planning_model
                )
            admitted_models.append(planning_model)
        elif method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD:
            failure_kind = receipt.get("failure_kind")
            if (
                failure_kind
                not in {
                    "contract_invalid",
                    "model_abstained",
                    "provider_error",
                    "response_parse_error",
                }
                or (
                    receipt.get("status") == "model_abstained"
                    and failure_kind != "model_abstained"
                )
                or (
                    receipt.get("status") == "prediction_failed"
                    and failure_kind == "model_abstained"
                )
            ):
                raise ValueError("runtime planner v8 failure kind differs")
            expected_failure = {
                "schema_version": "runtime_planner_prediction_failure_v1",
                "slate": slate,
                "event_admission": receipt.get("event_admission"),
                "provider_called": True,
                "failure_kind": failure_kind,
                **(
                    {
                        "error_type": receipt.get("error_type"),
                        "error_message": receipt.get("error_message"),
                    }
                    if receipt.get("status") == "prediction_failed"
                    else {}
                ),
            }
            if (
                planning_model is not None
                or receipt.get("planning_failure") != expected_failure
                or decision.get("planning_model") != expected_failure
            ):
                raise ValueError("runtime planner v8 failure evidence differs")
            RuntimeEventTraceDPV8Resolver(lambda _: None).resolve_recorded(
                RuntimeJointActionSlate.from_dict(slate), expected_failure
            )
        elif planning_model is not None or decision.get("planning_model") is not None:
            raise ValueError("runtime planner fallback decision contains a prediction")

    v8_no_provider_ids: set[str] = set()
    if method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD:
        consumed_event_rows = durable_event_consumptions_v8(
            selected_decisions, terminal_rows
        )
        consumed_event_decision_ids = {
            str(row["decision_id"]) for row in consumed_event_rows
        }
        admitted_event_keys: set[str] = set()
        cumulative_elapsed_ms = 0
        event_log_path = trial / "agent/runtime-events.jsonl"
        for decision in selected_decisions:
            decision_id = decision.get("decision_id")
            slate_payload = decision.get("slate")
            model = decision.get("planning_model")
            if (
                not isinstance(decision_id, str)
                or not isinstance(slate_payload, Mapping)
                or not isinstance(model, Mapping)
            ):
                raise ValueError("runtime planner v8 decision evidence is incomplete")
            slate_object = RuntimeJointActionSlate.from_dict(slate_payload)
            boundary_kernel = _kernel_at_joint_decision_boundary(
                event_log_path,
                trigger_action_id=slate_object.trigger_action_id,
            )
            expected_event = planner_event_admission_v8(
                slate_object,
                boundary_kernel.public_projection(),
                previously_admitted_event_keys=admitted_event_keys,
                semantic_review_threshold=V8_SEMANTIC_REVIEW_THRESHOLD,
                configured_check_contracts=configured_check_contracts,
            )
            schema = model.get("schema_version")
            if schema == "runtime_planner_event_admission_v1":
                if model != expected_event or expected_event["decision"] != "do_not_call_provider":
                    raise ValueError("runtime planner v8 event no-call history differs")
                v8_no_provider_ids.add(decision_id)
            elif schema in {
                "runtime_planner_cumulative_budget_no_call_v1",
                "runtime_planner_cumulative_budget_no_call_v2",
            }:
                if expected_event["decision"] != "call_provider":
                    raise ValueError("runtime planner v8 cumulative gate lacks an event")
                expected_budget = planner_cumulative_budget_gate_v8(
                    slate_object,
                    cumulative_planner_elapsed_ms=cumulative_elapsed_ms,
                    planner_ceiling_ms=int(configured["timeout_seconds"]) * 1_000,
                    provider_timeout_overrun_margin_ms=int(
                        configured["precall_provider_timeout_overrun_margin_ms"]
                    ),
                    cumulative_planner_budget_ms=V8_CUMULATIVE_PLANNER_BUDGET_MS,
                )
                if (
                    schema == "runtime_planner_cumulative_budget_no_call_v2"
                    and expected_budget is not None
                ):
                    expected_budget = bind_event_admission_to_resource_gate_v8(
                        slate_object, expected_budget, expected_event
                    )
                if expected_budget is None or model != expected_budget:
                    raise ValueError("runtime planner v8 cumulative budget history differs")
                v8_no_provider_ids.add(decision_id)
            elif schema in {
                "runtime_planner_time_reserve_no_call_v1",
                "runtime_planner_time_reserve_no_call_v2",
            }:
                if expected_event["decision"] != "call_provider":
                    raise ValueError("runtime planner v8 reserve gate lacks an event")
                expected_time = planner_time_reserve_gate_v8(
                    slate_object,
                    planner_ceiling_ms=int(configured["timeout_seconds"]) * 1_000,
                    actor_convergence_reserve_ms=int(
                        configured["precall_actor_convergence_reserve_ms"]
                    ),
                    provider_timeout_overrun_margin_ms=int(
                        configured["precall_provider_timeout_overrun_margin_ms"]
                    ),
                )
                if (
                    schema == "runtime_planner_time_reserve_no_call_v2"
                    and expected_time is not None
                ):
                    expected_time = bind_event_admission_to_resource_gate_v8(
                        slate_object, expected_time, expected_event
                    )
                if expected_time is None or model != expected_time:
                    raise ValueError("runtime planner v8 reserve history differs")
                v8_no_provider_ids.add(decision_id)
            elif schema in {
                "runtime_planner_prediction_failure_v1",
                "runtime_bounded_prediction_v7",
            }:
                admission = model.get("event_admission")
                if admission != expected_event or expected_event["decision"] != "call_provider":
                    raise ValueError("runtime planner v8 admitted event history differs")
                receipt = receipt_by_decision_id.get(decision_id)
                if (
                    receipt is None
                    or decision_id not in consumed_event_decision_ids
                ):
                    raise ValueError("runtime planner v8 admitted event lacks a provider call")
                admitted_event_keys.update(expected_event["new_event_keys"])
                cumulative_elapsed_ms += int(receipt["elapsed_ms"])
            else:
                raise ValueError("runtime planner v8 decision model identity differs")
            choice = RuntimeEventTraceDPV8Resolver(lambda _: None).resolve_recorded(
                slate_object, model
            )
            if decision.get("rule_id") != choice.rule_id:
                raise ValueError("runtime planner v8 replayed rule differs")

    # V7 can deliberately skip provider I/O when the live clock cannot honor
    # the planner ceiling and Actor convergence reserve. Such a model has no
    # provider receipt, but its exact time-gate evidence must still replay.
    precall_decision_ids: set[str] = set()
    for decision in selected_decisions:
        decision_id = decision.get("decision_id")
        if not isinstance(decision_id, str):
            raise ValueError("runtime planner decision identity differs")
        if decision_id in matched_decision_ids:
            continue
        if decision_id in v8_no_provider_ids:
            precall_decision_ids.add(decision_id)
            continue
        planning_model = decision.get("planning_model")
        if planning_model is None:
            continue
        if (
            method == BOUNDED_TRACEDP_DEV_V7_METHOD
            and isinstance(planning_model, Mapping)
            and planning_model.get("schema_version") == "runtime_planner_no_call_v1"
        ):
            if decision_id in precall_decision_ids:
                raise ValueError("runtime planner v7 no-call decision identity differs")
            slate_payload = decision.get("slate")
            if not isinstance(slate_payload, Mapping):
                raise ValueError("runtime planner v7 no-call slate is missing")
            if (
                planning_model.get("planner_ceiling_ms")
                != int(configured["timeout_seconds"]) * 1_000
                or planning_model.get("actor_convergence_reserve_ms")
                != configured["precall_actor_convergence_reserve_ms"]
                or planning_model.get("provider_timeout_overrun_margin_ms")
                != configured["precall_provider_timeout_overrun_margin_ms"]
            ):
                raise ValueError("runtime planner v7 no-call configuration differs")
            choice = RuntimeBoundedTraceDPV7Resolver(lambda _: None).resolve_recorded(
                RuntimeJointActionSlate.from_dict(slate_payload), planning_model
            )
            if decision.get("rule_id") != choice.rule_id:
                raise ValueError("runtime planner v7 no-call rule differs")
            precall_decision_ids.add(decision_id)
            continue
        raise ValueError("runtime planner fallback decision contains a prediction")
    decision_models = [
        row.get("planning_model")
        for row in selected_decisions
        if isinstance(row.get("planning_model"), Mapping)
        and str(row["planning_model"].get("schema_version", "")).startswith(
            "runtime_bounded_prediction_"
        )
        and row.get("decision_id") not in precall_decision_ids
    ]
    if decision_models != admitted_models:
        raise ValueError("runtime planner admitted predictions differ from decisions")
    totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    for row in terminal_rows:
        usage = row.get("model_usage")
        if not isinstance(usage, Mapping):
            complete = False
            continue
        if set(usage) != {
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "reasoning_tokens_reported",
        } or not isinstance(usage.get("reasoning_tokens_reported"), bool):
            raise ValueError("runtime planner usage schema differs")
        for field in totals:
            value = usage.get(field)
            if type(value) is not int or value < 0:
                raise ValueError("runtime planner token usage is invalid")
            totals[field] += value
    rebuilt = {
        "model_call_count": len(terminal_rows),
        "admitted_prediction_count": sum(
            row.get("status") == "prediction_admitted_uncalibrated"
            for row in terminal_rows
        ),
        **totals,
        "elapsed_ms": sum(int(row["elapsed_ms"]) for row in terminal_rows),
        "token_accounting_complete": complete,
        "ambiguous_started_call": pending is not None,
    }
    if recorded != rebuilt:
        raise ValueError("runtime planner accounting differs from durable receipts")
    return rebuilt


def _validate_actor_runtime_identity(
    *,
    trial: Path,
    runtime_config: Mapping[str, object],
    agent_config: Mapping[str, object],
    standalone_agent: Mapping[str, object],
    agent_kwargs: Mapping[str, object],
    standalone_kwargs: Mapping[str, object],
    result: Mapping[str, object],
    agent_result: Mapping[str, object],
    metadata: Mapping[str, object],
    kernel: PortableRuntimeKernel,
) -> None:
    """Bind the configured Actor to Pi streams and Harbor token accounting."""

    actor_runtime = runtime_config.get("actor_runtime")
    if not isinstance(actor_runtime, dict):
        raise ValueError("managed Actor runtime identity is missing")
    expected_keys = {
        "agent_import",
        "model_name",
        "provider",
        "model",
        "thinking",
        "transport",
        "pi_version",
        "host_pi_timeout_seconds",
        "host_pi_max_output_tokens",
        "observe_public_state",
        "public_path_count",
    }
    method = runtime_config.get("runtime_method")
    planner_accounting = _validate_runtime_planner_accounting(
        trial=trial,
        runtime_config=runtime_config,
        agent_kwargs=agent_kwargs,
        standalone_kwargs=standalone_kwargs,
        metadata=metadata,
    )
    if method in _RPC_METHODS:
        expected_keys.add("actor_control_mode")
    if method in BOUNDED_TRACEDP_DEV_METHODS:
        expected_keys.add("runtime_planner")
    model_name = agent_config.get("model_name")
    if (
        set(actor_runtime) != expected_keys
        or actor_runtime.get("agent_import") != _MANAGED_AGENT_IMPORT
        or agent_config.get("name") not in _MANAGED_AGENT_LAUNCH_IMPORTS
        or standalone_agent.get("name") != agent_config.get("name")
        or not isinstance(model_name, str)
        or "/" not in model_name
        or actor_runtime.get("model_name") != model_name
    ):
        raise ValueError("managed Actor runtime identity differs")
    provider, model = model_name.split("/", maxsplit=1)
    if actor_runtime.get("provider") != provider or actor_runtime.get("model") != model:
        raise ValueError("managed Actor provider or model differs")

    config_bindings = {
        "pi_version": "pi_version",
        "thinking": "thinking",
        "host_pi_timeout_seconds": "host_pi_timeout_seconds",
        "host_pi_max_output_tokens": "host_pi_max_output_tokens",
        "observe_public_state": "observe_public_state",
    }
    # A kwarg the launch config omits ran with the host constructor's default.
    constructor_defaults = {"observe_public_state": True}
    missing = object()
    for runtime_key, kwargs_key in config_bindings.items():
        expected = actor_runtime.get(runtime_key)
        default = constructor_defaults.get(kwargs_key, missing)
        if (
            agent_kwargs.get(kwargs_key, default) != expected
            or standalone_kwargs.get(kwargs_key, default) != expected
        ):
            raise ValueError("managed Actor runtime kwargs differ across artifacts")
    public_paths = runtime_config.get("public_paths")
    if not isinstance(public_paths, list) or actor_runtime.get("public_path_count") != len(
        public_paths
    ):
        raise ValueError("managed Actor public observation scope differs")
    metadata_bindings = {
        "provider": "provider",
        "model": "model",
        "pi_version": "pi_version",
        "thinking": "thinking",
        "transport": "transport",
        "host_pi_max_output_tokens": "max_output_tokens",
        "observe_public_state": "public_state_observation_enabled",
        "public_path_count": "public_path_count",
    }
    if (
        any(
            metadata.get(metadata_key) != actor_runtime.get(runtime_key)
            for runtime_key, metadata_key in metadata_bindings.items()
        )
        or metadata.get("credential_location") != "host_only"
    ):
        raise ValueError("managed Actor runtime metadata differs")
    if method in _RPC_METHODS and (
        actor_runtime.get("actor_control_mode") != "pi_jsonl_rpc"
        or metadata.get("actor_control_mode") != "pi_jsonl_rpc"
    ):
        raise ValueError("managed Actor control mode differs")
    agent_info = result.get("agent_info")
    model_info = agent_info.get("model_info") if isinstance(agent_info, dict) else None
    if (
        not isinstance(agent_info, dict)
        or agent_info.get("name") != "pi"
        or agent_info.get("version") != actor_runtime.get("pi_version")
        or not isinstance(model_info, dict)
        or model_info.get("provider") != provider
        or model_info.get("name") != model
    ):
        raise ValueError("managed Actor Harbor agent_info differs")

    events = _read_jsonl(trial / "agent/actor-session-events.jsonl")
    attempt_count = sum(row.get("event_type") == "actor_attempt_started" for row in events)
    total_input = total_output = total_cache = total_reasoning = 0
    controlled_submission_attempts: set[int] = set()
    settlement_review_attempts: set[int] = set()
    for attempt_index in range(attempt_count):
        stream_path = trial / "agent" / f"pi-attempt-{attempt_index:03d}.txt"
        stderr_path = trial / "agent" / f"pi-attempt-{attempt_index:03d}.stderr.txt"
        if not stream_path.is_file() or not stderr_path.is_file():
            raise ValueError("managed Actor Pi attempt stream is missing")
        receipt = load_pi_stream_receipt(
            stream_path,
            expected_provider=provider,
            expected_model=model,
        )
        identity_failures = {
            "host Pi model provider changed",
            "host Pi model identity changed",
            "host Pi emitted reasoning tokens with thinking off",
        }
        if receipt.failure in identity_failures:
            raise ValueError(f"managed Actor Pi identity differs: {receipt.failure}")
        if receipt.failure and not receipt.failure.startswith(
            "host Pi exhausted provider retries:"
        ):
            raise ValueError(f"managed Actor Pi evidence is invalid: {receipt.failure}")
        total_input += receipt.input_tokens + receipt.cache_tokens
        total_output += receipt.output_tokens
        total_cache += receipt.cache_tokens
        total_reasoning += receipt.reasoning_tokens
        terminal_event = next(
            (
                row.get("event_type")
                for row in events
                if isinstance(row.get("payload"), dict)
                and row["payload"].get("attempt_index") == attempt_index
                and row.get("event_type") != "actor_attempt_started"
            ),
            None,
        )
        actor_session_id = runtime_config.get("actor_session_id")
        if receipt.session_ids and receipt.session_ids != [actor_session_id]:
            raise ValueError("managed Actor Pi stream session differs")
        if terminal_event == "actor_attempt_completed":
            if method in _RPC_METHODS:
                try:
                    receipt.require_rpc_session_identity(
                        request_id=f"actor-state-{attempt_index:03d}",
                        expected_session_id=str(actor_session_id),
                    )
                except RuntimeError as error:
                    raise ValueError(
                        "managed Actor completed Pi stream lacks RPC session identity"
                    ) from error
            elif receipt.session_ids != [actor_session_id]:
                raise ValueError("managed Actor completed Pi stream lacks session identity")
            controlled_abort = method in _TIME_RESERVE_METHODS and any(
                item.request_id == f"runtime-submit-{attempt_index:03d}"
                and item.command == "abort"
                and item.success
                for item in receipt.rpc_responses
            )
            try:
                receipt.require_valid(
                    return_code=0,
                    controlled_abort_verified=controlled_abort,
                )
            except RuntimeError as error:
                raise ValueError("managed Actor completed Pi stream is invalid") from error
            if method in _RPC_METHODS:
                prompt_request_id = f"actor-prompt-{attempt_index:03d}"
                abort_request_id = f"runtime-submit-{attempt_index:03d}"
                prompt_accepted = any(
                    item.request_id == prompt_request_id
                    and item.command == "prompt"
                    and item.success
                    for item in receipt.rpc_responses
                )
                abort_accepted = any(
                    item.request_id == abort_request_id and item.command == "abort" and item.success
                    for item in receipt.rpc_responses
                )
                if not prompt_accepted or not receipt.agent_settled_sequences:
                    raise ValueError("managed Actor RPC completion receipt is incomplete")
                if abort_accepted:
                    try:
                        receipt.require_controlled_abort(
                            prompt_request_id=prompt_request_id,
                            abort_request_id=abort_request_id,
                        )
                    except RuntimeError as error:
                        raise ValueError(
                            "managed Actor RPC submission receipt is incomplete"
                        ) from error
                    controlled_submission_attempts.add(attempt_index)
                if method in _SETTLEMENT_REVIEW_METHODS:
                    review_accepted = any(
                        item.request_id == f"runtime-settlement-review-{attempt_index:03d}"
                        and item.command == "prompt"
                        and item.success
                        for item in receipt.rpc_responses
                    )
                    if review_accepted:
                        if len(receipt.agent_settled_sequences) < 2:
                            raise ValueError(
                                "managed Actor settlement review lacks final settlement"
                            )
                        review_actions = [
                            action
                            for action_id, action in kernel.actions.items()
                            if action_id == f"runtime-joint-settlement-review-{attempt_index:03d}"
                            and action.tool_name == "runtime_semantic_guidance"
                        ]
                        user_texts = []
                        for rpc_row in _read_jsonl(stream_path):
                            message = rpc_row.get("message")
                            if (
                                rpc_row.get("type") != "message_end"
                                or not isinstance(message, dict)
                                or message.get("role") != "user"
                            ):
                                continue
                            content = message.get("content")
                            if isinstance(content, list):
                                user_texts.extend(
                                    item.get("text")
                                    for item in content
                                    if isinstance(item, dict)
                                    and item.get("type") == "text"
                                    and isinstance(item.get("text"), str)
                                )
                        if (
                            len(review_actions) != 1
                            or (
                                _expected_semantic_notice(
                                    list(
                                        review_actions[0].arguments.get("output_resource_ids", ())
                                    ),
                                    policy_id=method,
                                ).removeprefix("\n\n")
                            )
                            not in user_texts
                        ):
                            raise ValueError("managed Actor settlement review content differs")
                        settlement_review_attempts.add(attempt_index)
    if method in _TIME_RESERVE_METHODS:
        kernel_submission_attempts = {
            int(action_id.rsplit("-", maxsplit=1)[1])
            for action_id, action in kernel.actions.items()
            if action.actor == "runtime" and action.tool_name == "submit_current_state"
        }
        if controlled_submission_attempts != kernel_submission_attempts:
            raise ValueError("managed Actor RPC submissions differ from Kernel")
    if method in _SETTLEMENT_REVIEW_METHODS:
        kernel_review_attempts = {
            int(action_id.rsplit("-", maxsplit=1)[1])
            for action_id, action in kernel.actions.items()
            if action.actor == "runtime"
            and action.tool_name == "runtime_semantic_guidance"
            and action_id.startswith("runtime-joint-settlement-review-")
        }
        if settlement_review_attempts != kernel_review_attempts:
            raise ValueError("managed Actor settlement reviews differ from Kernel")
    # Isolated repair trials run the same Actor in a clone; the host adds their usage to
    # the Harbor totals and appends one receipt per trial. Re-derive each from its own
    # Pi stream instead of trusting the receipt.
    usage_path = trial / "agent" / "isolated-guidance-usage.jsonl"
    for usage in _read_jsonl(usage_path) if usage_path.is_file() else []:
        streams = sorted(
            path
            for path in Path(str(usage.get("logs_dir", ""))).glob("pi-attempt-*.txt")
            if not path.name.endswith(".stderr.txt")
        )
        if not streams:
            raise ValueError("isolated guidance trial Pi stream is missing")
        trial_input = trial_output = trial_cache = trial_reasoning = 0
        for stream_path in streams:
            receipt = load_pi_stream_receipt(
                stream_path, expected_provider=provider, expected_model=model
            )
            if receipt.failure and not receipt.failure.startswith(
                "host Pi exhausted provider retries:"
            ):
                raise ValueError(
                    f"isolated guidance trial Pi evidence is invalid: {receipt.failure}"
                )
            trial_input += receipt.input_tokens + receipt.cache_tokens
            trial_output += receipt.output_tokens
            trial_cache += receipt.cache_tokens
            trial_reasoning += receipt.reasoning_tokens
        if (
            usage.get("input_tokens_including_cache") != trial_input
            or usage.get("output_tokens") != trial_output
            or usage.get("cache_read_tokens") != trial_cache
        ):
            raise ValueError("isolated guidance usage differs from its Pi stream")
        total_input += trial_input
        total_output += trial_output
        total_cache += trial_cache
        total_reasoning += trial_reasoning
    if (
        agent_result.get("n_input_tokens")
        != total_input + int(planner_accounting["input_tokens"])
        or agent_result.get("n_output_tokens")
        != total_output + int(planner_accounting["output_tokens"])
        or agent_result.get("n_cache_tokens") != total_cache
        or metadata.get("reasoning_tokens")
        != total_reasoning + int(planner_accounting["reasoning_tokens"])
    ):
        raise ValueError("managed Actor token accounting differs from Pi streams")


def _reconcile_interrupted_pi_session_tools(
    *,
    trial: Path,
    actor_session_id: str,
    expected_provider: str,
    expected_model: str,
    stream_calls: list[PiToolCall],
    stream_executions: list[PiToolExecution],
) -> tuple[list[PiToolCall], list[PiToolExecution], frozenset[str]]:
    """Recover only an exact persistent-session suffix after Host interruption."""

    session_dir = trial / "agent/pi/sessions"
    paths = sorted(
        path for path in session_dir.glob("*.jsonl") if not path.name.startswith("._")
    )
    if len(paths) != 1:
        raise ValueError("managed Actor persistent Pi session file differs")
    persistent = load_pi_persistent_session_tool_receipt(
        paths[0],
        expected_session_id=actor_session_id,
        expected_provider=expected_provider,
        expected_model=expected_model,
    )

    def call_identity(item: PiToolCall) -> tuple[object, ...]:
        """Compare request semantics without source-specific sequence numbers."""

        return item.tool_call_id, item.tool_name, item.arguments

    def execution_identity(item: PiToolExecution) -> tuple[object, ...]:
        """Compare completion semantics across stream and session formats."""

        return item.tool_call_id, item.tool_name, item.is_error, item.result

    persistent_calls = list(persistent.tool_calls)
    persistent_executions = list(persistent.tool_executions)
    if [call_identity(item) for item in stream_calls] != [
        call_identity(item) for item in persistent_calls[: len(stream_calls)]
    ]:
        raise ValueError("managed Actor Pi stream is not a persistent-session call prefix")
    if [execution_identity(item) for item in stream_executions] != [
        execution_identity(item)
        for item in persistent_executions[: len(stream_executions)]
    ]:
        raise ValueError("managed Actor Pi stream is not a persistent-session result prefix")
    return persistent_calls, persistent_executions, persistent.uncompleted_tool_call_ids


def _validate_actor_tool_receipts(
    *,
    trial: Path,
    kernel: PortableRuntimeKernel,
    runtime_config: Mapping[str, object],
) -> None:
    """Join Pi requests, Pi results, gateway receipts, and Kernel actions exactly."""

    actions = [action for action in kernel.actions.values() if action.actor == "actor"]
    path = trial / "agent/terminal-public-state.jsonl"
    if path.is_file():
        rows = _read_jsonl(path)
    elif actions:
        raise ValueError("managed Actor tool receipt log is missing")
    else:
        # A naturally settled text-only Actor never crosses the tool gateway.
        rows = []
    containment = runtime_config.get("bash_process_group_containment")
    if not isinstance(containment, Mapping):
        raise ValueError("managed Bash containment capability is missing")
    if len(rows) != len(actions):
        raise ValueError("managed Actor tool receipts do not cover every action")
    actor_runtime = runtime_config.get("actor_runtime")
    if not isinstance(actor_runtime, Mapping):
        raise ValueError("managed Actor runtime identity is missing")
    provider = actor_runtime.get("provider")
    model = actor_runtime.get("model")
    if not isinstance(provider, str) or not isinstance(model, str):
        raise ValueError("managed Actor model identity is missing")
    events = _read_jsonl(trial / "agent/actor-session-events.jsonl")
    attempt_count = sum(row.get("event_type") == "actor_attempt_started" for row in events)
    interrupted_attempts = {
        payload["attempt_index"]
        for row in events
        if row.get("event_type") in {"actor_attempt_interrupted", "actor_session_budget_exhausted"}
        and isinstance((payload := row.get("payload")), dict)
        and type(payload.get("attempt_index")) is int
    }
    pi_calls = []
    pi_executions = []
    call_attempt_by_id: dict[str, int] = {}
    terminal_uncompleted_ids: set[str] = set()
    actor_session_id = runtime_config.get("actor_session_id")
    runtime_method = runtime_config.get("runtime_method")
    for attempt_index in range(attempt_count):
        receipt = load_pi_stream_receipt(
            trial / "agent" / f"pi-attempt-{attempt_index:03d}.txt",
            expected_provider=provider,
            expected_model=model,
        )
        if receipt.session_ids and receipt.session_ids != [actor_session_id]:
            raise ValueError("managed Actor Pi stream session differs")
        rpc_session_bound = False
        if runtime_method in _RPC_METHODS:
            try:
                receipt.require_rpc_session_identity(
                    request_id=f"actor-state-{attempt_index:03d}",
                    expected_session_id=str(actor_session_id),
                )
                rpc_session_bound = True
            except RuntimeError:
                rpc_session_bound = False
        if (
            (
                receipt.tool_calls
                or receipt.tool_executions
                or receipt.actionable_assistant_messages
                or receipt.usage_events
            )
            and not receipt.session_ids
            and not rpc_session_bound
        ):
            raise ValueError("managed Actor Pi tool activity lacks session identity")
        evidence_sequences = [
            *receipt.assistant_event_sequences,
            *(item.event_sequence for item in receipt.tool_executions),
        ]
        rpc_state_sequences = [
            item.event_sequence
            for item in receipt.rpc_responses
            if item.request_id == f"actor-state-{attempt_index:03d}"
            and item.command == "get_state"
            and item.success
        ]
        identity_sequences = (
            rpc_state_sequences if rpc_session_bound else receipt.session_event_sequences
        )
        if evidence_sequences and (
            len(identity_sequences) != 1 or identity_sequences[0] >= min(evidence_sequences)
        ):
            raise ValueError("managed Actor Pi session does not precede attempt evidence")
        attempt_calls = {item.tool_call_id: item for item in receipt.tool_calls}
        for call in receipt.tool_calls:
            if call.tool_call_id in call_attempt_by_id:
                raise ValueError("managed Actor Pi tool call identities are duplicated")
            call_attempt_by_id[call.tool_call_id] = attempt_index
            if receipt.is_terminal_uncompleted_tool_call(call.tool_call_id):
                terminal_uncompleted_ids.add(call.tool_call_id)
        for execution in receipt.tool_executions:
            call = attempt_calls.get(execution.tool_call_id)
            if call is None or execution.event_sequence <= call.event_sequence:
                raise ValueError("managed Actor Pi tool execution precedes its request")
        pi_calls.extend(receipt.tool_calls)
        pi_executions.extend(receipt.tool_executions)
    session_paths = tuple(
        path
        for path in (trial / "agent/pi/sessions").glob("*.jsonl")
        if not path.name.startswith("._")
    )
    # A single-attempt session has an unambiguous durable suffix. Resumed sessions
    # need per-attempt leaf anchors before persistent evidence can fill a gap.
    if attempt_count == 1 and attempt_count - 1 in interrupted_attempts and (
        len(pi_calls) != len(actions) or session_paths
    ):
        pi_calls, pi_executions, recovered_uncompleted_ids = (
            _reconcile_interrupted_pi_session_tools(
                trial=trial,
                actor_session_id=str(actor_session_id),
                expected_provider=provider,
                expected_model=model,
                stream_calls=pi_calls,
                stream_executions=pi_executions,
            )
        )
        for call in pi_calls:
            call_attempt_by_id.setdefault(call.tool_call_id, attempt_count - 1)
        terminal_uncompleted_ids.update(recovered_uncompleted_ids)
    pi_executions_by_id = {item.tool_call_id: item for item in pi_executions}
    if len(pi_executions_by_id) != len(pi_executions):
        raise ValueError("managed Actor Pi tool execution identities are duplicated")
    locally_rejected_ids = set()
    for call in pi_calls:
        execution = pi_executions_by_id.get(call.tool_call_id)
        if (
            execution is None
            or not execution.is_error
            or not isinstance(execution.result, Mapping)
            or not isinstance(execution.result.get("content"), list)
        ):
            continue
        texts = [
            item.get("text")
            for item in execution.result["content"]
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        ]
        # Pi rejects unknown tools before the gateway, just like argument validation.
        truncated_call_notice = (
            f'Tool call "{call.tool_name}" was not executed: the response hit the output '
            "token limit, so its arguments may be truncated. Re-issue the tool call with "
            "complete arguments."
        )
        if (
            texts == [f"Tool {call.tool_name} not found"]
            or texts == [truncated_call_notice]
            or any(
                text.startswith(f'Validation failed for tool "{call.tool_name}":') for text in texts
            )
        ):
            locally_rejected_ids.add(call.tool_call_id)
    if locally_rejected_ids.intersection(action.action_id for action in actions):
        raise ValueError("Pi-local validation rejection reached the Kernel")
    pi_calls = [item for item in pi_calls if item.tool_call_id not in locally_rejected_ids]
    pi_executions = [
        item for item in pi_executions if item.tool_call_id not in locally_rejected_ids
    ]
    if len(pi_calls) != len(actions):
        raise ValueError("managed Actor Pi calls do not cover every Kernel action")
    if [item.tool_call_id for item in pi_calls] != [action.action_id for action in actions]:
        raise ValueError("managed Actor Pi tool call order differs from Kernel")
    actions_by_id = {action.action_id: action for action in actions}
    calls_by_id = {item.tool_call_id: item for item in pi_calls}
    row_ids = [row.get("tool_call_id") for row in rows]
    if (
        len(rows) != len(actions)
        or any(not isinstance(action_id, str) for action_id in row_ids)
        or len(set(row_ids)) != len(row_ids)
        or set(row_ids) != set(actions_by_id)
    ):
        raise ValueError("managed Actor gateway receipts do not cover every Kernel action")
    executions = {item.tool_call_id: item for item in pi_executions}
    expected_execution_ids = [
        str(row["tool_call_id"])
        for row in rows
        if row.get("tool_result") is not None or isinstance(row.get("error_message"), str)
    ]
    actual_execution_ids = [item.tool_call_id for item in pi_executions]
    missing_execution_ids = {
        action_id for action_id in expected_execution_ids if action_id not in executions
    }
    if actual_execution_ids != [
        action_id for action_id in expected_execution_ids if action_id not in missing_execution_ids
    ] or any(
        call_attempt_by_id.get(action_id) not in interrupted_attempts
        or action_id not in terminal_uncompleted_ids
        for action_id in missing_execution_ids
    ):
        raise ValueError("managed Actor Pi tool executions differ from gateway receipts")

    for sequence, row in enumerate(rows):
        action_id = row["tool_call_id"]
        action = actions_by_id[action_id]
        pi_call = calls_by_id[action_id]
        result = row.get("tool_result")
        error_stage = row.get("error_stage")
        error_type = row.get("error_type")
        error_message = row.get("error_message")
        recovery_reason = row.get("recovery_reason")
        if (
            row.get("schema_version") != ACTOR_TOOL_RECEIPT_SCHEMA
            or row.get("sequence") != sequence
            or row.get("tool_call_id") != action.action_id
            or row.get("tool_name") != action.tool_name
            or row.get("input") != dict(action.arguments)
            or not isinstance(row.get("is_error"), bool)
            or row.get("environment_step_executed") not in {True, False, None}
            or not isinstance(row.get("observation_complete"), bool)
            or not isinstance(row.get("before"), list)
            or not isinstance(row.get("after"), list)
            or pi_call.tool_name != action.tool_name
            or pi_call.arguments != dict(action.arguments)
        ):
            raise ValueError("managed Actor tool receipt differs from Kernel action")
        if result is not None:
            if (
                not isinstance(result, dict)
                or set(result) != {"content", "is_error", "environment_step_executed", "details"}
                or not isinstance(result.get("content"), str)
                or not isinstance(result.get("is_error"), bool)
                or not isinstance(result.get("environment_step_executed"), bool)
                or not isinstance(result.get("details"), dict)
                or result.get("is_error") != row.get("is_error")
                or result.get("environment_step_executed") != row.get("environment_step_executed")
            ):
                raise ValueError("managed Actor raw tool result is invalid")
            details = result["details"]
            if "pre_supervision_content_chars" in row:
                boundary = row["pre_supervision_content_chars"]
                if type(boundary) is not int or not 0 <= boundary <= len(result["content"]):
                    raise ValueError("managed tool content boundary is invalid")
                # Match the recorded suffix; never discover boundaries by markers.
                notice = TerminalEnvironmentToolGateway._runtime_notice(
                    details.get("runtime_interventions", [])
                )
                if result["content"][boundary:] != notice:
                    raise ValueError("managed tool content boundary includes supervision")
            if action.tool_name == "bash" and result.get("environment_step_executed") is True:
                requested_timeout = details.get("requested_command_timeout_seconds")
                effective_timeout = details.get("effective_command_timeout_seconds")
                if (
                    details.get("process_group_containment") != containment.get("mode")
                    or not isinstance(requested_timeout, int)
                    or isinstance(requested_timeout, bool)
                    or not 1 <= requested_timeout <= 600
                    or not isinstance(effective_timeout, int)
                    or isinstance(effective_timeout, bool)
                    or not 1 <= effective_timeout <= requested_timeout
                    or not isinstance(details.get("return_code"), int)
                    or isinstance(details.get("return_code"), bool)
                ):
                    raise ValueError("managed Bash execution containment differs")
            elif action.tool_name != "bash" and "process_group_containment" in details:
                raise ValueError("non-Bash tool claims process-group containment")
            if error_message is not None:
                raise ValueError("managed Actor result and propagated exception fields coexist")
            if recovery_reason is not None:
                raise ValueError("managed Actor result has stray recovery evidence")
            if error_stage == "parallel_rejection":
                if (
                    error_type is not None
                    or row.get("overlapping_tool_calls") is not True
                    or row.get("environment_step_executed") is not False
                    or row.get("observation_complete") is not False
                    or row.get("before") != []
                    or row.get("after") != []
                    or action.phase is not ActionPhase.BLOCKED
                ):
                    raise ValueError("managed Actor parallel rejection receipt is invalid")
            elif error_stage is not None:
                if (
                    error_stage
                    not in {
                        "after_snapshot",
                        "resource_after_snapshot",
                        "after_snapshot_and_resource_after_snapshot",
                    }
                    or not isinstance(error_type, str)
                    or not error_type
                    or row.get("observation_complete") is not False
                ):
                    raise ValueError("managed Actor post-observation receipt is invalid")
            elif error_type is not None:
                raise ValueError("managed Actor result has stray exception fields")
        if (
            action.phase
            in {
                ActionPhase.COMPLETED,
                ActionPhase.FAILED_WITHOUT_EFFECT,
                ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
                ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
            }
            and result is None
            and not isinstance(row.get("error_message"), str)
        ):
            raise ValueError("managed Actor terminal action lacks its tool result")
        if result is not None or isinstance(error_message, str):
            if recovery_reason is not None:
                raise ValueError("managed Actor execution has stray recovery evidence")
            pi_execution = executions.get(action.action_id)
            if result is None:
                if (
                    not isinstance(error_type, str)
                    or not error_type
                    or row.get("is_error") is not True
                    or row.get("observation_complete") is not False
                    or row.get("after") != []
                ):
                    raise ValueError("managed Actor execution exception receipt is invalid")
                if error_stage in {"resource_before_snapshot", "before_snapshot"}:
                    if (
                        row.get("before") != []
                        or row.get("environment_step_executed") is not False
                        or action.phase is not ActionPhase.BLOCKED
                    ):
                        raise ValueError("managed Actor pre-execution exception receipt is invalid")
                elif error_stage == "tool_execution":
                    if row.get("environment_step_executed") is not None or action.phase not in {
                        ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
                        ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
                    }:
                        raise ValueError("managed Actor tool exception receipt is invalid")
                else:
                    raise ValueError("managed Actor exception stage is invalid")
                expected_pi_result = {
                    "content": [
                        {
                            "type": "text",
                            "text": f"{error_type}: {error_message}",
                        }
                    ],
                    "details": {},
                }
                expected_pi_error = True
            else:
                expected_pi_result = {
                    "content": [{"type": "text", "text": result["content"]}],
                    # The locked Pi extension converts gateway errors into thrown
                    # Error objects, which preserve text but intentionally drop details.
                    "details": {} if result["is_error"] else result["details"],
                }
                expected_pi_error = result["is_error"]
            if (
                pi_execution is not None
                and (
                    pi_execution.tool_name != action.tool_name
                    or pi_execution.is_error != expected_pi_error
                    or pi_execution.result != expected_pi_result
                )
            ) or (
                pi_execution is None
                and (
                    call_attempt_by_id.get(action.action_id) not in interrupted_attempts
                    or action.action_id not in terminal_uncompleted_ids
                )
            ):
                raise ValueError("managed Actor Pi tool result differs from gateway")
        else:
            common_missing_result_invalid = (
                row.get("is_error") is not True
                or row.get("observation_complete") is not False
                or row.get("after") != []
                or executions.get(action.action_id) is not None
            )
            cancelled_by_tool = (
                error_stage == "tool_execution"
                and error_type == "CancelledError"
                and recovery_reason is None
                and row.get("environment_step_executed") is None
                and action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
            )
            interrupted_by_host = (
                error_stage == "host_runtime_interruption"
                and error_type == "HostRuntimeRestart"
                and set(row)
                == {
                    "schema_version",
                    "sequence",
                    "tool_call_id",
                    "tool_name",
                    "input",
                    "overlapping_tool_calls",
                    "environment_step_executed",
                    "observation_complete",
                    "before",
                    "after",
                    "is_error",
                    "latency_ms",
                    "tool_result",
                    "error_stage",
                    "error_type",
                    "recovery_reason",
                }
                and isinstance(recovery_reason, str)
                and bool(recovery_reason)
                and recovery_reason == action.reason
                and row.get("overlapping_tool_calls") is False
                and row.get("before") == []
                and row.get("latency_ms") == {}
                and call_attempt_by_id.get(action.action_id) in interrupted_attempts
                and (
                    (
                        action.phase is ActionPhase.BLOCKED
                        and row.get("environment_step_executed") is False
                    )
                    or (
                        action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
                        and row.get("environment_step_executed") is None
                    )
                )
            )
            if common_missing_result_invalid or not (cancelled_by_tool or interrupted_by_host):
                raise ValueError("managed Actor missing exception evidence is invalid")
        if action.phase is ActionPhase.BLOCKED and result is not None:
            if (
                result.get("is_error") is not True
                or result.get("environment_step_executed") is not False
            ):
                raise ValueError("managed Actor blocked receipt claims execution")


def _validate_managed_check_configuration(
    trial: Path, runtime_config: Mapping[str, object],
    agent_kwargs: Mapping[str, object], standalone_kwargs: Mapping[str, object],
) -> None:
    """Bind optional checks to both Harbor configs and the gateway's frozen file."""
    fields = {
        "runtime_check_contracts": [],
        "predicate_diagnosis_enabled": False,
        "allow_check_registration": False,
        "require_check_registration_phase": False,
    }
    configured = {key: agent_kwargs.get(key, default) for key, default in fields.items()}
    standalone = {key: standalone_kwargs.get(key, default) for key, default in fields.items()}
    configured["runtime_check_contracts"] = configured["runtime_check_contracts"] or []
    standalone["runtime_check_contracts"] = standalone["runtime_check_contracts"] or []
    if configured != standalone:
        raise ValueError("managed runtime check settings differ across Harbor configs")
    contracts = tuple(
        RuntimeCheckContract(**{**row, "resource_ids": tuple(row.get("resource_ids", ()))})
        for row in configured["runtime_check_contracts"]
    )
    enabled = configured["predicate_diagnosis_enabled"]
    registration = configured["allow_check_registration"]
    phase_required = configured["require_check_registration_phase"]
    if (
        type(enabled) is not bool
        or type(registration) is not bool
        or type(phase_required) is not bool
    ):
        raise ValueError("managed runtime check flags are invalid")
    payload = runtime_check_configuration_payload(
        contracts, diagnosis_enabled=enabled, registration_enabled=registration,
        registration_phase_required=phase_required,
    )
    active = bool(contracts or enabled or registration or phase_required)
    if runtime_config.get("runtime_check_configuration") != (payload if active else None):
        raise ValueError("managed runtime checks differ from frozen runtime-config")
    path = trial / "agent/runtime-check-contracts.json"
    if active and not path.is_file():
        raise ValueError("managed runtime checks lack gateway configuration")
    if path.is_file() and _read_object(path) != payload:
        raise ValueError("managed runtime gateway check configuration differs")


def _validate_managed_repair_configuration(
    runtime_config: Mapping[str, object],
    agent_kwargs: Mapping[str, object],
    standalone_kwargs: Mapping[str, object],
) -> None:
    """Bind the frozen repair menu to both Harbor configuration copies."""

    configured = agent_kwargs.get("runtime_predicate_repair_contracts") or []
    standalone = standalone_kwargs.get("runtime_predicate_repair_contracts") or []
    if configured != standalone or not isinstance(configured, list):
        raise ValueError("managed runtime repair settings differ across Harbor configs")
    try:
        contracts = tuple(
            RuntimePredicateRepairContract(
                **{**row, "resource_ids": tuple(row.get("resource_ids", ()))},
            )
            for row in configured
            if isinstance(row, Mapping)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("managed runtime repair settings are invalid") from error
    if len(contracts) != len(configured):
        raise ValueError("managed runtime repair settings are invalid")
    budget = runtime_config.get("runtime_intervention_budget")
    if contracts and (
        runtime_config.get("runtime_method") not in _JOINT_METHODS
        or type(budget) is not int
        or budget < 1
    ):
        raise ValueError("managed runtime repairs require a budgeted joint policy")
    expected = runtime_repair_configuration_payload(contracts) if contracts else None
    if runtime_config.get("runtime_repair_configuration") != expected:
        raise ValueError("managed runtime repairs differ from frozen runtime-config")


def _validate_runtime_atomic_estimate_configuration(
    runtime_config: Mapping[str, object],
    agent_kwargs: Mapping[str, object],
    standalone_kwargs: Mapping[str, object],
) -> RuntimeFrozenAtomicEstimateConfiguration | None:
    """Bind zero-model estimates to both Harbor configs and the runtime commit."""

    method = runtime_config.get("runtime_method")
    configured = agent_kwargs.get("runtime_atomic_estimate_configuration")
    standalone = standalone_kwargs.get("runtime_atomic_estimate_configuration")
    frozen = runtime_config.get("runtime_atomic_estimate_configuration")
    if configured != standalone or configured != frozen:
        raise ValueError("runtime atomic estimate configuration differs across artifacts")
    if method not in RUNTIME_OWNED_OPTION_DP_DEV_METHODS:
        if frozen is not None:
            raise ValueError("non-option method contains atomic estimate configuration")
        return None
    if not isinstance(frozen, Mapping):
        raise ValueError("Runtime-owned option DP lacks frozen atomic estimates")
    try:
        return RuntimeFrozenAtomicEstimateConfiguration.from_dict(frozen)
    except (TypeError, ValueError) as error:
        raise ValueError("runtime atomic estimate configuration is invalid") from error


_ACCEPTANCE_MANIFEST_KEYS = {
    "predicate_id",
    "task_id",
    "proposition",
    "source_start",
    "source_end",
    "source_excerpt",
    "resource_ids",
    "aggregation",
}


def _acceptance_definition(value: object) -> dict[str, object]:
    """Project immutable predicate fields without trusting its mutable maturity."""

    return {
        "predicate_id": value.predicate_id,
        "task_id": value.task_id,
        "proposition": value.proposition,
        "provenance": value.provenance.value,
        "source_start": value.source_start,
        "source_end": value.source_end,
        "source_excerpt": value.source_excerpt,
        "resource_ids": list(value.resource_ids),
        "aggregation": value.aggregation.value,
    }


def _validate_acceptance_predicate_configuration(
    *,
    trial: Path,
    runtime_config: Mapping[str, object],
    agent_kwargs: Mapping[str, object],
    standalone_kwargs: Mapping[str, object],
    kernel: PortableRuntimeKernel,
    check_evidence_audit: Mapping[str, object] | None,
) -> None:
    """Rebuild predicate definitions and admit only independently audited checks."""

    manifest = runtime_config.get("runtime_acceptance_predicate_manifest")
    agent_manifest = agent_kwargs.get("runtime_acceptance_predicate_manifest")
    standalone_manifest = standalone_kwargs.get("runtime_acceptance_predicate_manifest")
    semantic_review = runtime_config.get(
        "runtime_acceptance_predicate_semantic_review"
    )
    run_class = runtime_config.get("run_class")
    if agent_manifest != manifest or standalone_manifest != manifest:
        raise ValueError("managed acceptance manifest differs across Harbor configs")
    if manifest is None:
        if (
            kernel.acceptance_predicate_enabled
            or kernel.acceptance_predicates
            or run_class is not None
            or agent_kwargs.get("run_class") is not None
            or standalone_kwargs.get("run_class") is not None
            or semantic_review is not None
            or agent_kwargs.get("runtime_acceptance_predicate_semantic_review")
            is not None
            or standalone_kwargs.get("runtime_acceptance_predicate_semantic_review")
            is not None
        ):
            raise ValueError("legacy runtime unexpectedly contains acceptance predicates")
        return
    if (
        run_class != "exposed_development_canary"
        or agent_kwargs.get("run_class") != run_class
        or standalone_kwargs.get("run_class") != run_class
    ):
        raise ValueError("unverified acceptance predicates require a development canary")
    if not isinstance(manifest, list):
        raise ValueError("managed acceptance manifest is invalid")

    instruction = runtime_config.get("public_instruction")
    resources = runtime_config.get("resources")
    if not isinstance(instruction, str) or not isinstance(resources, list):
        raise ValueError("managed acceptance manifest lacks frozen public inputs")
    predicate_ids = [
        row.get("predicate_id") for row in manifest if isinstance(row, dict)
    ]
    raw_reviewed_checks = agent_kwargs.get("runtime_check_contracts") or []
    if not isinstance(raw_reviewed_checks, list):
        raise ValueError("managed acceptance semantic review checks are invalid")
    try:
        reviewed_check_contracts = runtime_check_configuration_payload(
            tuple(
                RuntimeCheckContract(
                    **{
                        **row,
                        "resource_ids": tuple(row.get("resource_ids", ())),
                    }
                )
                for row in raw_reviewed_checks
                if isinstance(row, dict)
            ),
            diagnosis_enabled=False,
        )["contracts"]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "managed acceptance semantic review checks are invalid"
        ) from error
    if len(reviewed_check_contracts) != len(raw_reviewed_checks):
        raise ValueError("managed acceptance semantic review checks are invalid")
    if (
        semantic_review
        != agent_kwargs.get("runtime_acceptance_predicate_semantic_review")
        or semantic_review
        != standalone_kwargs.get("runtime_acceptance_predicate_semantic_review")
        or not isinstance(semantic_review, dict)
        or set(semantic_review)
        != {
            "schema_version",
            "review_scope",
            "reviewer",
            "public_instruction",
            "predicate_ids",
            "reviewed_manifest",
            "reviewed_check_contracts",
            "semantic_entailment_reviewed",
            "hidden_evaluator_or_reference_answer_used",
            "baseline_access_policy",
        }
        or semantic_review.get("schema_version")
        != "runtime_acceptance_predicate_public_semantic_review_v2"
        or semantic_review.get("review_scope")
        != "public_instruction_and_explicit_manifest_only"
        or not isinstance(semantic_review.get("reviewer"), str)
        or not str(semantic_review["reviewer"]).strip()
        # The canonical instruction is stripped (load_trial_public_instruction); a review
        # that kept the file's trailing newline names the same public text.
        or not isinstance(semantic_review.get("public_instruction"), str)
        or str(semantic_review["public_instruction"]).strip() != instruction
        or semantic_review.get("predicate_ids") != predicate_ids
        or semantic_review.get("reviewed_manifest") != manifest
        or semantic_review.get("reviewed_check_contracts")
        != reviewed_check_contracts
        or semantic_review.get("semantic_entailment_reviewed") is not True
        or semantic_review.get("hidden_evaluator_or_reference_answer_used") is not False
        or semantic_review.get("baseline_access_policy")
        != "same_manifest_and_checks_for_all_methods"
    ):
        raise ValueError("managed acceptance semantic review receipt is invalid")
    public_task_ids = {
        node.task_id for node in kernel.task_nodes.values() if node.public_requirement
    }
    resource_ids = {
        row.get("resource_id") for row in resources if isinstance(row, dict)
    }
    expected_definitions: dict[str, dict[str, object]] = {}
    for row in manifest:
        if not isinstance(row, dict) or set(row) != _ACCEPTANCE_MANIFEST_KEYS:
            raise ValueError("managed public acceptance manifest schema is invalid")
        source_start = row.get("source_start")
        source_end = row.get("source_end")
        source_excerpt = row.get("source_excerpt")
        raw_resource_ids = row.get("resource_ids")
        predicate_id = row.get("predicate_id")
        task_id = row.get("task_id")
        proposition = row.get("proposition")
        aggregation = row.get("aggregation")
        if (
            not isinstance(predicate_id, str)
            or not predicate_id
            or predicate_id in expected_definitions
            or task_id not in public_task_ids
            or not isinstance(proposition, str)
            or not proposition.strip()
            or type(source_start) is not int
            or type(source_end) is not int
            or not isinstance(source_excerpt, str)
            or source_end <= source_start
            or instruction[source_start:source_end] != source_excerpt
            or not isinstance(raw_resource_ids, list)
            or not raw_resource_ids
            or any(
                not isinstance(resource_id, str) or resource_id not in resource_ids
                for resource_id in raw_resource_ids
            )
            or len(raw_resource_ids) != len(set(raw_resource_ids))
            or aggregation not in {"all", "any"}
        ):
            raise ValueError("managed public acceptance predicate is invalid")
        expected_definitions[predicate_id] = {
            **row,
            "provenance": "public_contract",
        }
    definition_signatures = [
        (
            row["task_id"],
            row["proposition"],
            row["source_start"],
            row["source_end"],
            row["source_excerpt"],
            tuple(row["resource_ids"]),
        )
        for row in expected_definitions.values()
    ]
    if len(definition_signatures) != len(set(definition_signatures)):
        raise ValueError("managed public acceptance predicate definitions are duplicated")

    events = _read_jsonl(trial / "agent/runtime-events.jsonl")
    completed_registrations = [
        action
        for action in kernel.actions.values()
        if action.actor == "actor"
        and action.tool_name == "register_runtime_check"
        and action.phase is ActionPhase.COMPLETED
    ]
    expected_bindings: dict[str, str] = {}
    expected_check_contracts: dict[str, dict[str, object]] = {}
    configured_contracts = agent_kwargs.get("runtime_check_contracts") or []
    if not isinstance(configured_contracts, list):
        raise ValueError("managed acceptance checks are invalid")
    for row in configured_contracts:
        if not isinstance(row, dict):
            raise ValueError("managed acceptance check is invalid")
        check_id = row.get("check_id")
        predicate_id = row.get("predicate_id")
        if (check_id is None) != (predicate_id is None):
            raise ValueError("managed acceptance check binding is incomplete")
        if check_id is not None:
            if (
                not isinstance(check_id, str)
                or not isinstance(predicate_id, str)
                or predicate_id not in expected_definitions
                or check_id in expected_bindings
                or row.get("task_id")
                != expected_definitions[predicate_id]["task_id"]
                or row.get("proposition")
                != expected_definitions[predicate_id]["proposition"]
                or row.get("resource_ids")
                != expected_definitions[predicate_id]["resource_ids"]
            ):
                raise ValueError("managed acceptance check binding is invalid")
            expected_bindings[check_id] = predicate_id
            expected_check_contracts[check_id] = row

    for action in completed_registrations:
        arguments = action.arguments
        task_id = arguments.get("task_id")
        proposition = arguments.get("proposition")
        source_excerpt = arguments.get("source_excerpt")
        raw_resource_ids = arguments.get("resource_ids", [])
        node = kernel.task_nodes.get(str(task_id))
        if (
            node is None
            or not node.public_requirement
            or not isinstance(proposition, str)
            or not isinstance(source_excerpt, str)
            or not source_excerpt
            or not isinstance(raw_resource_ids, list)
        ):
            raise ValueError("completed Actor predicate registration is invalid")
        source_start = instruction.find(source_excerpt)
        if (
            source_start < 0
            or instruction.find(source_excerpt, source_start + 1) >= 0
        ):
            raise ValueError("Actor predicate source excerpt is absent or ambiguous")
        definition_core = {
            "task_id": task_id,
            "proposition": proposition,
            "source_start": source_start,
            "source_end": source_start + len(source_excerpt),
            "source_excerpt": source_excerpt,
            "resource_ids": raw_resource_ids,
        }
        public_matches = [
            predicate_id
            for predicate_id, definition in expected_definitions.items()
            if all(definition.get(key) == value for key, value in definition_core.items())
        ]
        if len(public_matches) > 1:
            raise ValueError("Actor check matches multiple public predicates")
        if public_matches:
            predicate_id = public_matches[0]
        else:
            predicate_id = f"actor-predicate:{action.action_id}"
            expected_definitions[predicate_id] = {
                "predicate_id": predicate_id,
                **definition_core,
                "provenance": "actor_inferred",
                "aggregation": "advisory",
            }
        check_id = f"actor-check:{action.action_id}"
        if check_id in expected_bindings:
            raise ValueError("dynamic acceptance check identity is duplicated")
        expected_bindings[check_id] = predicate_id
        expected_check_contracts[check_id] = {
            "task_id": task_id,
            "proposition": proposition,
            "arguments": {
                "command": arguments.get("command"),
                "timeout": arguments.get("timeout", 120),
            },
            "resource_ids": raw_resource_ids,
            "check_id": check_id,
            "predicate_id": predicate_id,
        }

    actual_definitions = {
        predicate_id: _acceptance_definition(predicate)
        for predicate_id, predicate in kernel.acceptance_predicates.items()
    }
    if (
        not kernel.acceptance_predicate_enabled
        or actual_definitions != expected_definitions
        or kernel.acceptance_predicate_check_bindings != expected_bindings
    ):
        raise ValueError("managed acceptance predicate definitions or bindings differ")

    audited_rows = (
        check_evidence_audit.get("checks", [])
        if isinstance(check_evidence_audit, Mapping)
        else []
    )
    if not isinstance(audited_rows, list):
        raise ValueError("managed acceptance check audit is invalid")
    audited_by_evidence = {
        row.get("evidence_id"): row
        for row in audited_rows
        if isinstance(row, dict) and isinstance(row.get("evidence_id"), str)
    }
    for evidence in kernel.evidence.values():
        if evidence.predicate_id is None:
            if evidence.check_id is not None:
                raise ValueError("managed unscoped evidence cites an acceptance check")
            continue
        contract = expected_check_contracts.get(str(evidence.check_id))
        source_action = kernel.actions.get(str(evidence.source_action_id))
        audited = audited_by_evidence.get(evidence.evidence_id)
        if (
            not evidence.check_id
            or not evidence.source_action_id
            or expected_bindings.get(evidence.check_id) != evidence.predicate_id
            or contract is None
            or source_action is None
            or source_action.phase
            not in {
                ActionPhase.COMPLETED,
                ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
                ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
            }
            or audited is None
            or audited.get("check_id") != evidence.check_id
            or audited.get("predicate_id") != evidence.predicate_id
            or audited.get("source_action_id") != evidence.source_action_id
        ):
            raise ValueError("managed predicate evidence lacks an audited bound check")
        actor_source_matches = (
            source_action.actor == "actor"
            and source_action.tool_name == "bash"
            and source_action.arguments == contract.get("arguments")
        )
        runtime_source_matches = (
            source_action.actor == "runtime"
            and source_action.tool_name == "runtime_registered_check_execution"
            and source_action.arguments.get("check_id") == evidence.check_id
            and source_action.arguments.get("predicate_id") == evidence.predicate_id
            and source_action.arguments.get("task_id") == contract.get("task_id")
            and source_action.arguments.get("proposition")
            == contract.get("proposition")
            and source_action.arguments.get("resource_ids")
            == contract.get("resource_ids")
            and source_action.arguments.get("check_arguments")
            == contract.get("arguments")
        )
        if not actor_source_matches and not runtime_source_matches:
            raise ValueError("managed predicate evidence source differs from its check")

    latest_incident_by_predicate: dict[str, str] = {}
    for event in events:
        event_type = event.get("event_type")
        if event_type not in {"evidence_recorded", "predicate_incident_updated"}:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("managed acceptance incident event is invalid")
        incident = (
            payload.get("predicate_incident_transition")
            if event_type == "evidence_recorded"
            else payload
        )
        if incident is None:
            continue
        if not isinstance(incident, dict):
            raise ValueError("managed acceptance incident event is invalid")
        evidence = kernel.evidence.get(str(incident.get("evidence_id", "")))
        if evidence is not None and evidence.predicate_id is not None:
            latest_incident_by_predicate[evidence.predicate_id] = str(
                incident.get("incident_id", "")
            )
    state_names = {
        EvidenceState.SUPPORTED: "current_supported",
        EvidenceState.REFUTED: "current_refuted",
        EvidenceState.UNKNOWN: "current_unknown",
        EvidenceState.STALE: "stale",
    }
    for predicate_id, predicate in kernel.acceptance_predicates.items():
        evidence_rows = [
            evidence
            for evidence in kernel.evidence.values()
            if evidence.predicate_id == predicate_id
        ]
        latest = evidence_rows[-1] if evidence_rows else None
        if (
            predicate.state.value
            != (state_names[latest.state] if latest is not None else "never_checked")
            or predicate.latest_evidence_id
            != (latest.evidence_id if latest is not None else None)
            or predicate.latest_incident_id
            != latest_incident_by_predicate.get(predicate_id)
        ):
            raise ValueError("managed acceptance predicate maturity differs from evidence")

    projection = kernel.public_projection()
    expected_summary: dict[str, str] = {}
    for task_id in kernel.task_nodes:
        required = [
            predicate
            for predicate in kernel.acceptance_predicates.values()
            if predicate.task_id == task_id
            and predicate.aggregation is not AcceptancePredicateAggregation.ADVISORY
        ]
        if not required:
            continue
        all_rows = [
            predicate
            for predicate in required
            if predicate.aggregation is AcceptancePredicateAggregation.ALL
        ]
        any_rows = [
            predicate
            for predicate in required
            if predicate.aggregation is AcceptancePredicateAggregation.ANY
        ]
        if any(
            predicate.state is AcceptancePredicateState.CURRENT_REFUTED
            for predicate in all_rows
        ) or any_rows and all(
            predicate.state is AcceptancePredicateState.CURRENT_REFUTED
            for predicate in any_rows
        ):
            expected_summary[task_id] = "blocked"
        elif all(
            predicate.state is AcceptancePredicateState.CURRENT_SUPPORTED
            for predicate in all_rows
        ) and (
            not any_rows
            or any(
                predicate.state is AcceptancePredicateState.CURRENT_SUPPORTED
                for predicate in any_rows
            )
        ):
            expected_summary[task_id] = "awaiting_official_evaluation"
        else:
            expected_summary[task_id] = "unresolved"
    if projection.get("task_acceptance_summary") != expected_summary:
        raise ValueError("managed task acceptance summary differs")


def _validate_trial_identity(
    *,
    trial: Path,
    trial_config: Mapping[str, object],
    runtime_config: Mapping[str, object],
    result: Mapping[str, object],
    kernel: PortableRuntimeKernel,
    check_evidence_audit: Mapping[str, object] | None,
    allow_legacy_identity: bool,
) -> tuple[str, str]:
    """Join both Harbor configs, result metadata, and runtime history exactly."""

    config_schema = runtime_config.get("schema_version")
    if config_schema not in {_RUNTIME_CONFIG_SCHEMA_V1, _RUNTIME_CONFIG_SCHEMA_V2}:
        raise ValueError("managed runtime-config schema is unsupported")
    if config_schema == _RUNTIME_CONFIG_SCHEMA_V1:
        if (
            "runtime_dataset_adapter" in runtime_config
            or "runtime_dataset_adapter_migration" in runtime_config
        ):
            raise ValueError("legacy runtime-config contains adapter fields")
    else:
        if runtime_config.get("runtime_dataset_adapter") != {
            "adapter_id": TERMINAL_RUNTIME_ADAPTER_V1,
            "dataset_family": "terminal",
            "instantiation_scope": "initial_kernel_inputs_v1",
        }:
            raise ValueError("managed runtime-config adapter identity differs")
        if "runtime_dataset_adapter_migration" not in runtime_config:
            raise ValueError("managed runtime-config adapter migration differs")
        migration = runtime_config.get("runtime_dataset_adapter_migration")
        if migration not in (
            None,
            {
                "from_schema": _RUNTIME_CONFIG_SCHEMA_V1,
                "to_schema": _RUNTIME_CONFIG_SCHEMA_V2,
                "reason": "bind_existing_terminal_kernel_to_explicit_adapter",
            },
        ):
            raise ValueError("managed runtime-config adapter migration differs")
    method = runtime_config.get("runtime_method")
    if method not in _SUPPORTED_METHODS:
        raise ValueError("managed runtime method identity is unsupported")
    if runtime_config.get("environment_session_id") != kernel.session_id:
        raise ValueError("managed runtime-config session differs from the event log")
    harbor_config = _nested_mapping(result, "config", label="Harbor result config")
    task_config = _nested_mapping(harbor_config, "task", label="Harbor task config")
    standalone_task = _nested_mapping(trial_config, "task", label="standalone Harbor task config")
    task_path = task_config.get("path")
    if not isinstance(task_path, str) or standalone_task.get("path") != task_path:
        raise ValueError("managed runtime task identity differs from Harbor config")
    if not allow_legacy_identity:
        public_roots = [
            node
            for node in kernel.task_nodes.values()
            if node.public_requirement and node.parent_id is None
        ]
        public_instruction = runtime_config.get("public_instruction")
        compiler_id = runtime_config.get("task_contract_compiler")
        frozen_instruction = load_trial_public_instruction(trial, Path(task_path))
        if (
            len(public_roots) != 1
            or not isinstance(public_instruction, str)
            or not public_instruction
            or public_roots[0].title != public_instruction
            or frozen_instruction != public_instruction
            or compiler_id not in SUPPORTED_TASK_CONTRACT_COMPILERS
        ):
            raise ValueError("managed runtime public task contract differs")
        recorded_public_evidence = runtime_config.get("public_dependency_evidence_ids")
        resource_rows = runtime_config.get("resources")
        event_rows = _read_jsonl(trial / "agent/runtime-events.jsonl")
        initialized = next(
            (row for row in event_rows if row.get("event_type") == "initialized"),
            None,
        )
        initial_payload = initialized.get("payload") if isinstance(initialized, dict) else None
        observation_rows = (
            initial_payload.get("observations") if isinstance(initial_payload, dict) else None
        )
        if not isinstance(resource_rows, list) or not isinstance(observation_rows, list):
            raise ValueError("managed runtime public dependency inputs are missing")
        observation_schema = runtime_config.get(
            "resource_observation_schema",
            TERMINAL_RESOURCE_OBSERVATION_V1,
        )
        if observation_schema not in {
            TERMINAL_RESOURCE_OBSERVATION_V1,
            TERMINAL_RESOURCE_OBSERVATION_V2,
        }:
            raise ValueError("managed runtime resource observation schema is unsupported")
        if observation_schema == TERMINAL_RESOURCE_OBSERVATION_V1 and any(
            isinstance(row, dict) and _DIRECTORY_SUMMARY_KEYS.intersection(row)
            for row in observation_rows
        ):
            raise ValueError("legacy resource observation contains directory summaries")
        if observation_schema == TERMINAL_RESOURCE_OBSERVATION_V2 and any(
            isinstance(row, dict)
            and row.get("exists") is True
            and row.get("file_type") == "directory"
            and not _DIRECTORY_SUMMARY_KEYS <= set(row)
            for row in observation_rows
        ):
            raise ValueError("directory resource observation summary is missing")
        resources = tuple(
            ResourceSpec(
                resource_id=str(row["resource_id"]),
                kind=ResourceKind(row["kind"]),
                provenance=str(row["provenance"]),
                group_id=(str(row["group_id"]) if row.get("group_id") is not None else None),
            )
            for row in resource_rows
            if isinstance(row, dict)
        )
        observations = tuple(
            ResourceObservation(
                resource_id=str(row["resource_id"]),
                kind=ResourceKind(row["kind"]),
                observed=row.get("observed") is True,
                exists=row.get("exists") if type(row.get("exists")) is bool else None,
                size_bytes=row.get("size_bytes")
                if isinstance(row.get("size_bytes"), int)
                else None,
                modified_time=row.get("modified_time")
                if isinstance(row.get("modified_time"), int)
                else None,
                modified_time_detail=(
                    str(row["modified_time_detail"])
                    if row.get("modified_time_detail") is not None
                    else None
                ),
                change_time=row.get("change_time")
                if isinstance(row.get("change_time"), int)
                else None,
                change_time_detail=(
                    str(row["change_time_detail"])
                    if row.get("change_time_detail") is not None
                    else None
                ),
                inode=row.get("inode") if isinstance(row.get("inode"), int) else None,
                file_type=str(row["file_type"]) if row.get("file_type") is not None else None,
                mode=str(row["mode"]) if row.get("mode") is not None else None,
                group_id=str(row["group_id"]) if row.get("group_id") is not None else None,
                descendant_entry_count=(
                    int(row["descendant_entry_count"])
                    if isinstance(row.get("descendant_entry_count"), int)
                    else None
                ),
                descendant_regular_file_count=(
                    int(row["descendant_regular_file_count"])
                    if isinstance(row.get("descendant_regular_file_count"), int)
                    else None
                ),
                descendant_total_file_bytes=(
                    int(row["descendant_total_file_bytes"])
                    if isinstance(row.get("descendant_total_file_bytes"), int)
                    else None
                ),
                descendant_latest_modified_time=(
                    str(row["descendant_latest_modified_time"])
                    if row.get("descendant_latest_modified_time") is not None
                    else None
                ),
                descendant_latest_change_time=(
                    str(row["descendant_latest_change_time"])
                    if row.get("descendant_latest_change_time") is not None
                    else None
                ),
                descendant_observation_complete=(
                    row.get("descendant_observation_complete")
                    if type(row.get("descendant_observation_complete")) is bool
                    else None
                ),
            )
            for row in observation_rows
            if isinstance(row, dict)
        )
        expected_specs = compile_public_instruction_dependency_evidence(
            instruction=public_instruction,
            resources=resources,
            initial_observations=observations,
            compiler_id=str(compiler_id),
        )
        expected_graph = compile_initial_resource_task_graph(
            instruction=public_instruction,
            resources=resources,
            initial_observations=observations,
            compiler_id=str(compiler_id),
        )
        expected_public_nodes = {
            node.task_id: node for node in expected_graph if node.public_requirement
        }
        actual_public_nodes = {
            node.task_id: node for node in kernel.task_nodes.values() if node.public_requirement
        }
        if actual_public_nodes != expected_public_nodes:
            raise ValueError("managed runtime public output obligations differ")
        expected_public_evidence = [item.evidence_id for item in expected_specs]
        actual_public_ids = [
            evidence_id
            for evidence_id in kernel.evidence
            if evidence_id.startswith(f"{compiler_id}:")
        ]
        if (
            recorded_public_evidence != expected_public_evidence
            or actual_public_ids != expected_public_evidence
            or [
                evidence_id
                for evidence_id, evidence in kernel.evidence.items()
                if not evidence.dependency_claims_version_bound
            ]
            != expected_public_evidence
        ):
            raise ValueError("managed runtime public dependency evidence differs")
        for item in expected_specs:
            evidence = kernel.evidence.get(item.evidence_id)
            if (
                evidence is None
                or evidence.task_id != item.task_id
                or evidence.proposition != item.proposition
                or evidence.state is not EvidenceState.SUPPORTED
                or evidence.resource_versions
                or evidence.dependency_claims != ((item.resource_id, DependencyRole.CONFIRMED),)
                or evidence.dependency_claims_version_bound
            ):
                raise ValueError("managed runtime public dependency evidence differs")

    task_name = result.get("task_name")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("managed runtime task identity is missing")
    if (
        result.get("trial_name") != trial.name
        or trial_config.get("trial_name") != trial.name
        or harbor_config.get("trial_name") != trial.name
    ):
        raise ValueError("managed runtime trial identity differs from Harbor config")
    expected_task_id = task_name.rsplit("/", 1)[-1]
    if (
        not isinstance(task_path, str)
        or standalone_task.get("path") != task_path
        or Path(task_path).name != expected_task_id
    ):
        raise ValueError("managed runtime task identity differs from Harbor config")

    agent_result = _nested_mapping(result, "agent_result", label="Harbor agent result")
    metadata = _nested_mapping(agent_result, "metadata", label="Harbor agent metadata")
    configured_phase_required = bool(
        runtime_config.get("runtime_check_configuration", {}).get(
            "actor_registration_phase_required", False
        )
        if isinstance(runtime_config.get("runtime_check_configuration"), dict)
        else False
    )
    if metadata.get("actor_check_registration_phase_required", False) is not (
        configured_phase_required
    ):
        raise ValueError("managed Actor check registration phase metadata differs")
    if metadata.get("continuous_environment_session") != kernel.session_id:
        raise ValueError("managed runtime result session differs from the event log")
    if metadata.get("runtime_method") != method:
        raise ValueError("managed runtime result method differs from runtime-config")
    if metadata.get("continuous_actor_session") is not True:
        raise ValueError("managed runtime actor session was not completed continuously")
    if metadata.get("actor_session_resumable") is True:
        raise ValueError("managed runtime actor session remains resumable")
    if not allow_legacy_identity:
        actor_session_id = runtime_config.get("actor_session_id")
        actor_resume_limit = runtime_config.get("actor_resume_limit")
        actor_resume_count = metadata.get("actor_session_resume_count")
        if (
            not isinstance(actor_session_id, str)
            or not actor_session_id
            or metadata.get("actor_session_id") != actor_session_id
            or not isinstance(actor_resume_limit, int)
            or isinstance(actor_resume_limit, bool)
            or actor_resume_limit < 0
            or not isinstance(actor_resume_count, int)
            or isinstance(actor_resume_count, bool)
            or not 0 <= actor_resume_count <= actor_resume_limit
            or metadata.get("actor_session_resume_limit") != actor_resume_limit
            or not isinstance(metadata.get("actor_session_resumed"), bool)
        ):
            raise ValueError("managed runtime actor resume identity differs")
    agent_config = _nested_mapping(harbor_config, "agent", label="Harbor agent config")
    standalone_agent = _nested_mapping(
        trial_config, "agent", label="standalone Harbor agent config"
    )
    for key in ("name", "model_name"):
        if (
            not isinstance(agent_config.get(key), str)
            or not agent_config[key]
            or standalone_agent.get(key) != agent_config[key]
        ):
            raise ValueError("managed runtime agent identity differs across Harbor config")
    agent_kwargs = _nested_mapping(agent_config, "kwargs", label="Harbor agent kwargs")
    standalone_kwargs = _nested_mapping(
        standalone_agent, "kwargs", label="standalone Harbor agent kwargs"
    )
    _validate_managed_check_configuration(trial, runtime_config, agent_kwargs, standalone_kwargs)
    _validate_managed_repair_configuration(runtime_config, agent_kwargs, standalone_kwargs)
    _validate_runtime_atomic_estimate_configuration(
        runtime_config, agent_kwargs, standalone_kwargs
    )
    _validate_acceptance_predicate_configuration(
        trial=trial,
        runtime_config=runtime_config,
        agent_kwargs=agent_kwargs,
        standalone_kwargs=standalone_kwargs,
        kernel=kernel,
        check_evidence_audit=check_evidence_audit,
    )
    if (
        agent_kwargs.get("runtime_method") != method
        or standalone_kwargs.get("runtime_method") != method
    ):
        raise ValueError("managed runtime configured method differs from runtime-config")

    restore = runtime_config.get("restore_missing_prerequisites")
    budget = runtime_config.get("runtime_intervention_budget")
    if (
        not isinstance(restore, bool)
        or not isinstance(budget, int)
        or isinstance(budget, bool)
        or budget < 0
        or agent_kwargs.get("restore_missing_prerequisites") is not restore
        or standalone_kwargs.get("restore_missing_prerequisites") is not restore
        # Harbor omits kwargs that retain the agent constructor's zero default.
        or agent_kwargs.get("runtime_intervention_budget", 0) != budget
        or standalone_kwargs.get("runtime_intervention_budget", 0) != budget
        or metadata.get("runtime_intervention_budget") != budget
    ):
        raise ValueError("managed runtime method configuration differs across artifacts")
    if not allow_legacy_identity and (
        agent_kwargs.get("actor_resume_limit", 1) != runtime_config.get("actor_resume_limit")
        or standalone_kwargs.get("actor_resume_limit", 1)
        != runtime_config.get("actor_resume_limit")
    ):
        raise ValueError("managed runtime actor resume configuration differs")
    expected_joint_arm = runtime_config.get("joint_source_arm_index")
    expected_joint_schema = (
        RuntimeJointSlatePolicy.schema_version if method in _JOINT_METHODS else None
    )
    if not allow_legacy_identity and (
        runtime_config.get("joint_policy_schema") != expected_joint_schema
        or agent_kwargs.get("joint_source_arm_index") != expected_joint_arm
        or standalone_kwargs.get("joint_source_arm_index") != expected_joint_arm
        or (method not in _JOINT_METHODS and expected_joint_arm is not None)
        or (
            method in _ROTATING_SOURCE_METHODS
            and (
                isinstance(expected_joint_arm, bool)
                or not isinstance(expected_joint_arm, int)
                or expected_joint_arm < 0
            )
        )
        or (method not in _ROTATING_SOURCE_METHODS and expected_joint_arm is not None)
    ):
        raise ValueError("managed runtime joint-policy configuration differs")
    jit_task_id = runtime_config.get("jit_test_task_id")
    jit_source_path = runtime_config.get("jit_memory_source_path")
    jit_artifact = runtime_config.get("jit_memory_artifact")
    jit_payload = runtime_config.get("jit_memory_payload")
    if method in JIT_JOINT_METHODS:
        copied_memory_path = trial / "agent/runtime-jit-memory.json"
        if (
            not isinstance(jit_task_id, str)
            or not jit_task_id
            or Path(task_path).name != jit_task_id
            or not isinstance(jit_source_path, str)
            or not Path(jit_source_path).is_absolute()
            or agent_kwargs.get("jit_test_task_id") != jit_task_id
            or standalone_kwargs.get("jit_test_task_id") != jit_task_id
            or agent_kwargs.get("jit_memory_path") != jit_source_path
            or standalone_kwargs.get("jit_memory_path") != jit_source_path
            or jit_artifact != "runtime-jit-memory.json"
            or not isinstance(jit_payload, dict)
            or not copied_memory_path.is_file()
            or json.loads(copied_memory_path.read_text()) != jit_payload
        ):
            raise ValueError("managed runtime JIT memory identity differs")
        memory = FrozenRuntimeJITMemory.load(copied_memory_path)
        memory.population.identity(
            jit_task_id,
            split=memory.population.evaluation_split,
        )
    elif not allow_legacy_identity and any(
        value is not None for value in (jit_task_id, jit_source_path, jit_artifact, jit_payload)
    ):
        raise ValueError("non-JIT runtime contains JIT memory configuration")
    expected_progress_threshold = (
        OUTPUT_PROGRESS_ACTION_THRESHOLD if method in OUTPUT_PROGRESS_RUNTIME_METHODS else None
    )
    if not allow_legacy_identity and (
        runtime_config.get("output_progress_action_threshold") != expected_progress_threshold
        or metadata.get("output_progress_action_threshold") != expected_progress_threshold
    ):
        raise ValueError("managed runtime output-progress configuration differs")
    expected_failure_diagnosis_threshold = (
        1
        if method in EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS
        else ACTOR_FAILURE_DIAGNOSIS_THRESHOLD
        if method in FAILURE_DIAGNOSIS_JOINT_METHODS
        else None
    )
    if not allow_legacy_identity and (
        runtime_config.get("actor_failure_diagnosis_threshold")
        != expected_failure_diagnosis_threshold
        or metadata.get("actor_failure_diagnosis_threshold") != expected_failure_diagnosis_threshold
    ):
        raise ValueError("managed runtime failure-diagnosis configuration differs")
    expected_semantic_stability_threshold = (
        OUTPUT_SEMANTIC_STABILITY_ACTION_THRESHOLD
        if method in SEMANTIC_STABILITY_JOINT_METHODS
        else None
    )
    if not allow_legacy_identity and (
        runtime_config.get("output_semantic_stability_action_threshold")
        != expected_semantic_stability_threshold
        or metadata.get("output_semantic_stability_action_threshold")
        != expected_semantic_stability_threshold
    ):
        raise ValueError("managed runtime semantic-stability configuration differs")
    expected_convergence_reserve = (
        ACTOR_CONVERGENCE_RESERVE_SECONDS if method in _ACTOR_RESERVE_METHODS else None
    )
    if not allow_legacy_identity and (
        runtime_config.get("actor_convergence_reserve_seconds") != expected_convergence_reserve
        or metadata.get("actor_convergence_reserve_seconds") != expected_convergence_reserve
    ):
        raise ValueError("managed runtime convergence configuration differs")
    expected_effect_schema = (
        "runtime_intervention_effect_event_v1" if method in EFFECT_LEDGER_RUNTIME_METHODS else None
    )
    if (
        not allow_legacy_identity
        and runtime_config.get("intervention_effect_schema") != expected_effect_schema
    ):
        raise ValueError("managed runtime intervention-effect configuration differs")
    if not allow_legacy_identity:
        _validate_bash_containment_identity(
            runtime_config=runtime_config,
            metadata=metadata,
        )
        _validate_resource_and_retention_identity(
            trial=trial,
            runtime_config=runtime_config,
            agent_kwargs=agent_kwargs,
            standalone_kwargs=standalone_kwargs,
            metadata=metadata,
            kernel=kernel,
        )
        _validate_actor_session_events(
            trial=trial,
            runtime_config=runtime_config,
            metadata=metadata,
        )
        _validate_actor_runtime_identity(
            trial=trial,
            runtime_config=runtime_config,
            agent_config=agent_config,
            standalone_agent=standalone_agent,
            agent_kwargs=agent_kwargs,
            standalone_kwargs=standalone_kwargs,
            result=result,
            agent_result=agent_result,
            metadata=metadata,
            kernel=kernel,
        )
        _validate_actor_tool_receipts(
            trial=trial,
            kernel=kernel,
            runtime_config=runtime_config,
        )
    runtime_action_count = sum(action.actor == "runtime" for action in kernel.actions.values())
    completed_count = sum(
        action.actor == "runtime" and action.phase is ActionPhase.COMPLETED
        for action in kernel.actions.values()
    )
    cancelled_count = sum(
        action.actor == "runtime" and action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
        for action in kernel.actions.values()
    )
    failed_count = runtime_action_count - completed_count - cancelled_count
    if not allow_legacy_identity and (
        metadata.get("runtime_intervention_count") != completed_count
        or metadata.get("runtime_intervention_attempt_count") != runtime_action_count
        or metadata.get("runtime_intervention_completed_count") != completed_count
        or metadata.get("runtime_intervention_failure_count") != failed_count
        or metadata.get("runtime_intervention_cancelled_count") != cancelled_count
    ):
        raise ValueError("managed runtime intervention metadata differs from history")
    if method == MANAGED_OBSERVATION_METHOD:
        if restore or budget != 0 or runtime_action_count != 0:
            raise ValueError("observation-only runtime contains intervention behavior")
    elif method == SHARED_INPUT_PROTECTION_METHOD:
        if not restore or budget != 0:
            raise ValueError("shared input protection configuration is invalid")
    elif restore or budget < 1:
        raise ValueError("dynamic TraceDP runtime configuration is invalid")
    return task_name, method


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"runtime decision log line {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"runtime decision log line {line_number} is not an object")
        rows.append(row)
    return rows


def _kernel_action_version_snapshots(
    event_log_path: Path,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Reconstruct resource versions at each durable Kernel action terminus."""

    observations: dict[str, object] = {}
    versions: dict[str, int] = {}
    snapshots: dict[str, dict[str, int]] = {}
    terminal_order: dict[str, int] = {}

    def apply_observations(raw: object) -> None:
        if not isinstance(raw, list):
            raise ValueError("runtime event observations are invalid")
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("resource_id"), str):
                raise ValueError("runtime event observation is invalid")
            resource_id = item["resource_id"]
            if observations.get(resource_id) != item:
                observations[resource_id] = item
                versions[resource_id] = versions.get(resource_id, 0) + 1

    for index, row in enumerate(_read_jsonl(event_log_path)):
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime event payload is invalid")
        event_type = row.get("event_type")
        if index == 0:
            if event_type != "initialized":
                raise ValueError("runtime event log lacks initialization")
            apply_observations(payload.get("observations"))
        elif event_type in {"observations_refreshed", "action_finished"}:
            apply_observations(payload.get("observations"))
        if event_type in {"action_finished", "action_blocked", "action_blocked_proposal"}:
            action_id = payload.get("action_id")
            if not isinstance(action_id, str) or not action_id:
                raise ValueError("runtime terminal action identity is invalid")
            snapshots[action_id] = dict(versions)
            terminal_order[action_id] = index
    return snapshots, terminal_order


def _kernel_at_joint_decision_boundary(
    event_log_path: Path, *, trigger_action_id: str
) -> PortableRuntimeKernel:
    """Replay the exact Kernel prefix consumed before one joint policy selection."""

    rows = _read_jsonl(event_log_path)
    terminal_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("event_type")
            in {"action_finished", "action_blocked", "action_blocked_proposal"}
            and isinstance(row.get("payload"), Mapping)
            and row["payload"].get("action_id") == trigger_action_id
        ),
        None,
    )
    if terminal_index is None:
        raise ValueError("runtime planner v6 trigger boundary is missing")

    # Checks and incident transitions are admitted after the triggering action
    # finishes but before policy selection. Model inference emits no Kernel event.
    boundary_end = terminal_index + 1
    while boundary_end < len(rows) and rows[boundary_end].get("event_type") in {
        "actor_failure_boundary_recorded",
        "acceptance_predicate_registered",
        "acceptance_predicate_check_bound",
        "evidence_recorded",
        "predicate_incident_updated",
    }:
        boundary_end += 1
    with tempfile.TemporaryDirectory() as directory:
        prefix_path = Path(directory) / "runtime-joint-boundary.jsonl"
        prefix_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows[:boundary_end]
            ),
            encoding="utf-8",
        )
        return PortableRuntimeKernel.load(prefix_path)


def _kernel_action_changed_resources(event_log_path: Path) -> dict[str, frozenset[str]]:
    """Recompute which resource observations changed at each action terminus."""

    observations: dict[str, object] = {}
    changed_by_action: dict[str, frozenset[str]] = {}
    for index, row in enumerate(_read_jsonl(event_log_path)):
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime event payload is invalid")
        event_type = row.get("event_type")
        raw_observations = (
            payload.get("observations")
            if index == 0 or event_type in {"observations_refreshed", "action_finished"}
            else None
        )
        changed: set[str] = set()
        if raw_observations is not None:
            if not isinstance(raw_observations, list):
                raise ValueError("runtime event observations are invalid")
            for item in raw_observations:
                if not isinstance(item, dict) or not isinstance(item.get("resource_id"), str):
                    raise ValueError("runtime event observation is invalid")
                resource_id = str(item["resource_id"])
                if observations.get(resource_id) != item:
                    observations[resource_id] = item
                    changed.add(resource_id)
        if event_type in {"action_finished", "action_blocked", "action_blocked_proposal"}:
            action_id = payload.get("action_id")
            if not isinstance(action_id, str) or not action_id:
                raise ValueError("runtime terminal action identity is invalid")
            changed_by_action[action_id] = frozenset(changed)
    return changed_by_action


def _intervention_delivery_receipts(
    observer_path: Path, *, intervention_kind: str
) -> dict[str, tuple[str, dict[str, object], str]]:
    """Extract one notice kind from gateway results already bound to Pi events."""

    delivered: dict[str, tuple[str, dict[str, object], str]] = {}
    # The actor-receipt validator separately rejects a missing log when tools ran.
    rows = _read_jsonl(observer_path) if observer_path.is_file() else []
    for row in rows:
        actor_action_id = row.get("tool_call_id")
        result = row.get("tool_result")
        if not isinstance(actor_action_id, str) or not isinstance(result, dict):
            continue
        details = result.get("details")
        interventions = details.get("runtime_interventions") if isinstance(details, dict) else None
        if interventions is None:
            continue
        if not isinstance(interventions, list):
            raise ValueError("runtime intervention gateway receipt is invalid")
        for intervention in interventions:
            if not isinstance(intervention, dict):
                raise ValueError("runtime intervention gateway receipt is invalid")
            if intervention.get("intervention_kind") != intervention_kind:
                continue
            action_id = intervention.get("action_id")
            content = result.get("content")
            if (
                not isinstance(action_id, str)
                or action_id in delivered
                or not isinstance(content, str)
            ):
                raise ValueError("runtime intervention delivery receipt is invalid")
            delivered[action_id] = (actor_action_id, intervention, content)
    return delivered


def _pi_interruption_delivery_context(
    trial: Path, runtime_config: Mapping[str, object]
) -> tuple[frozenset[int], dict[str, int], frozenset[str], frozenset[str]]:
    """Bind each Pi call and completion to durable interrupted-attempt evidence."""

    actor_runtime = runtime_config.get("actor_runtime")
    if not isinstance(actor_runtime, Mapping):
        raise ValueError("managed Actor runtime identity is missing")
    provider = actor_runtime.get("provider")
    model = actor_runtime.get("model")
    if not isinstance(provider, str) or not isinstance(model, str):
        raise ValueError("managed Actor model identity is missing")
    actor_events = _read_jsonl(trial / "agent/actor-session-events.jsonl")
    interrupted = frozenset(
        int(payload["attempt_index"])
        for row in actor_events
        if row.get("event_type") in {"actor_attempt_interrupted", "actor_session_budget_exhausted"}
        and isinstance((payload := row.get("payload")), dict)
        and type(payload.get("attempt_index")) is int
    )
    attempt_count = sum(row.get("event_type") == "actor_attempt_started" for row in actor_events)
    call_attempt: dict[str, int] = {}
    executions: set[str] = set()
    terminal_uncompleted: set[str] = set()
    for attempt_index in range(attempt_count):
        receipt = load_pi_stream_receipt(
            trial / "agent" / f"pi-attempt-{attempt_index:03d}.txt",
            expected_provider=provider,
            expected_model=model,
        )
        for call in receipt.tool_calls:
            if call.tool_call_id in call_attempt:
                raise ValueError("managed Actor Pi tool call identities are duplicated")
            call_attempt[call.tool_call_id] = attempt_index
            if receipt.is_terminal_uncompleted_tool_call(call.tool_call_id):
                terminal_uncompleted.add(call.tool_call_id)
        executions.update(item.tool_call_id for item in receipt.tool_executions)
    return (
        interrupted,
        call_attempt,
        frozenset(executions),
        frozenset(terminal_uncompleted),
    )


def _expected_progress_notice(
    *, output_resource_ids: list[object], output_requirement: object, action_count: object
) -> str:
    """Rebuild the fixed progress message independently of gateway state."""

    if (
        not output_resource_ids
        or not all(isinstance(item, str) for item in output_resource_ids)
        or output_requirement not in {"all", "any"}
        or not isinstance(action_count, int)
        or isinstance(action_count, bool)
    ):
        raise ValueError("runtime progress notice arguments are invalid")
    paths = ", ".join(str(item) for item in output_resource_ids)
    requirement = (
        "at least one of these outputs" if output_requirement == "any" else "all of these outputs"
    )
    return (
        "\n\n[Runtime supervisor]\n"
        f"Runtime supervisor observed {action_count} Actor actions without a change "
        f"to the declared public outputs. Produce {requirement}: {paths}. Then "
        "validate the result using publicly available checks."
    )


def _ends_with_runtime_notice_batch_member(delivered_content: str, expected_notice: str) -> bool:
    """Validate one notice when the gateway emits one header for the whole batch."""

    header = "\n\n[Runtime supervisor]\n"
    if not expected_notice.startswith(header):
        raise ValueError("expected runtime notice lacks its canonical header")
    notice_body = expected_notice[len(header) :]
    return delivered_content.endswith(expected_notice) or delivered_content.endswith(
        "\n" + notice_body
    )


def _validate_cancelled_delivery_boundary(
    *,
    record: RuntimeInterventionEffectRecord,
    replayed: Mapping[str, object] | None,
    delivery_receipt: tuple[str, dict[str, object], str] | None,
    interrupted_attempts: frozenset[int],
    call_attempt_by_id: Mapping[str, int],
    pi_execution_ids: frozenset[str],
    terminal_uncompleted_ids: frozenset[str],
) -> None:
    """Bind an unknown delivery to the last unresolved Pi batch and its gateway row."""

    trigger_action_id = replayed.get("trigger_action_id") if isinstance(replayed, Mapping) else None
    attempt_index = (
        call_attempt_by_id.get(trigger_action_id) if isinstance(trigger_action_id, str) else None
    )
    correction_trigger = record.delivery_correction_trigger_action_id
    corrected_delivery_invalid = correction_trigger is not None and (
        delivery_receipt is None or correction_trigger != trigger_action_id
    )
    recovered_delivery_invalid = correction_trigger is None and delivery_receipt is not None
    if (
        attempt_index not in interrupted_attempts
        or trigger_action_id in pi_execution_ids
        or trigger_action_id not in terminal_uncompleted_ids
        or corrected_delivery_invalid
        or recovered_delivery_invalid
    ):
        raise ValueError("cancelled runtime delivery lacks an interrupted Pi boundary")


def _planned_action_by_runtime_execution(
    decision_path: Path,
) -> dict[str, str]:
    """Bind each executed Kernel action to its policy-selected action identity."""

    pending: str | None = None
    bindings: dict[str, str] = {}
    for row in _read_jsonl(decision_path):
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime decision payload is invalid")
        if row.get("event_type") == "decision_planned":
            action_id = payload.get("action_id")
            pending = action_id if isinstance(action_id, str) else None
        elif row.get("event_type") == "decision_selected":
            action_id = payload.get("selected_instance_action_id")
            pending = (
                action_id
                if payload.get("selected_action_type") != "continue_actor"
                and isinstance(action_id, str)
                else None
            )
        elif row.get("event_type") in {
            "decision_execution_started",
            "execution_started",
        }:
            executed = payload.get("executed_action_id")
            if pending is None or not isinstance(executed, str) or executed in bindings:
                raise ValueError("runtime decision execution binding is invalid")
            bindings[executed] = pending
        elif row.get("event_type") in {
            "decision_outcome",
            "decision_abandoned",
            "execution_outcome",
        }:
            pending = None
    return bindings


def _joint_decision_by_runtime_execution(decision_path: Path) -> dict[str, str]:
    """Bind each joint runtime execution to its durable one-shot decision."""

    pending: str | None = None
    bindings: dict[str, str] = {}
    for row in _read_jsonl(decision_path):
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime joint decision payload is invalid")
        if row.get("event_type") == "decision_selected":
            decision_id = payload.get("decision_id")
            pending = (
                decision_id
                if payload.get("selected_action_type") != "continue_actor"
                and isinstance(decision_id, str)
                else None
            )
        elif row.get("event_type") == "execution_started":
            executed = payload.get("executed_action_id")
            if pending is None or not isinstance(executed, str) or executed in bindings:
                raise ValueError("runtime joint execution binding is invalid")
            bindings[executed] = pending
        elif row.get("event_type") in {"execution_outcome", "decision_abandoned"}:
            pending = None
    return bindings


def _joint_trigger_by_runtime_execution(decision_path: Path) -> dict[str, str]:
    """Bind delivery to the action whose slate selected the runtime execution."""

    pending: str | None = None
    bindings: dict[str, str] = {}
    for row in _read_jsonl(decision_path):
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime joint decision payload is invalid")
        if row.get("event_type") == "decision_selected":
            slate = payload.get("slate")
            trigger = slate.get("trigger_action_id") if isinstance(slate, dict) else None
            pending = (
                trigger
                if payload.get("selected_action_type") != "continue_actor"
                and isinstance(trigger, str)
                else None
            )
        elif row.get("event_type") == "execution_started":
            executed = payload.get("executed_action_id")
            if pending is None or not isinstance(executed, str) or executed in bindings:
                raise ValueError("runtime joint trigger binding is invalid")
            bindings[executed] = pending
        elif row.get("event_type") in {"execution_outcome", "decision_abandoned"}:
            pending = None
    return bindings


def _summarize_joint_decision_evidence(
    decision_payloads: list[Mapping[str, object]],
) -> dict[str, object]:
    """Rebuild actual selector use from durable decisions, independent of configuration."""

    rule_ids = [payload.get("rule_id") for payload in decision_payloads]
    if any(not isinstance(rule_id, str) or not rule_id for rule_id in rule_ids):
        raise ValueError("runtime joint decision rule identity is missing")

    bellman_decision_count = sum(
        isinstance(values := payload.get("bellman_values"), Mapping) and bool(values)
        for payload in decision_payloads
    )
    jit_payloads = [
        jit
        for payload in decision_payloads
        if isinstance(jit := payload.get("jit_advantages"), Mapping)
    ]
    admitted_jit_decision_count = sum(
        jit.get("admitted") is True for jit in jit_payloads
    )
    jit_changed_action_decision_count = 0
    for jit in jit_payloads:
        scores = jit.get("scores")
        if jit.get("admitted") is not True or not isinstance(scores, list) or not scores:
            continue
        chosen_id = jit.get("chosen_instance_action_id")
        chosen_scores = [
            float(item["base_score"])
            for item in scores
            if item["instance_action_id"] == chosen_id
        ]
        if len(chosen_scores) != 1:
            raise ValueError("runtime JIT chosen action score is missing or ambiguous")
        # A tied base optimum is still a base-policy choice, regardless of tie order.
        jit_changed_action_decision_count += chosen_scores[0] < max(
            float(item["base_score"]) for item in scores
        )
    missing_prediction_fallback_count = sum(
        rule_id in _MISSING_PREDICTION_FALLBACK_RULE_IDS for rule_id in rule_ids
    )
    planner_no_call_reserve_count = sum(
        rule_id in _PLANNER_NO_CALL_RESERVE_RULE_IDS for rule_id in rule_ids
    )
    planner_single_candidate_no_call_count = sum(
        rule_id in _PLANNER_SINGLE_CANDIDATE_RULE_IDS for rule_id in rule_ids
    )
    planner_event_suppressed_no_call_count = sum(
        rule_id in _PLANNER_EVENT_SUPPRESSED_RULE_IDS for rule_id in rule_ids
    )
    planner_contract_invalid_fallback_count = sum(
        rule_id in _PLANNER_CONTRACT_INVALID_RULE_IDS for rule_id in rule_ids
    )
    planner_cumulative_budget_no_call_count = sum(
        rule_id in _PLANNER_CUMULATIVE_BUDGET_RULE_IDS for rule_id in rule_ids
    )
    summary = {
        "decision_count": len(decision_payloads),
        "observed_rule_ids": sorted(set(rule_ids)),
        "bellman_decision_count": bellman_decision_count,
        "missing_prediction_fallback_count": missing_prediction_fallback_count,
        "planner_no_call_reserve_count": planner_no_call_reserve_count,
        "jit_payload_decision_count": len(jit_payloads),
        "admitted_jit_decision_count": admitted_jit_decision_count,
        "jit_changed_action_decision_count": jit_changed_action_decision_count,
        "learned_scores_or_bellman_values_used": bool(
            bellman_decision_count or admitted_jit_decision_count
        ),
    }
    # Preserve historical method summaries byte-for-byte. V8 adds counters only
    # when a V8 rule actually appears in the durable decision stream.
    if any(str(rule_id).startswith("bounded_dp_v8_") for rule_id in rule_ids):
        summary.update(
            planner_single_candidate_no_call_count=(
                planner_single_candidate_no_call_count
            ),
            planner_event_suppressed_no_call_count=(
                planner_event_suppressed_no_call_count
            ),
            planner_contract_invalid_fallback_count=(
                planner_contract_invalid_fallback_count
            ),
            planner_cumulative_budget_no_call_count=(
                planner_cumulative_budget_no_call_count
            ),
        )
    return summary


def _expected_semantic_notice(output_resource_ids: list[object], *, policy_id: str) -> str:
    """Rebuild the versioned actor-visible message independently of gateway state."""

    if not output_resource_ids or not all(isinstance(item, str) for item in output_resource_ids):
        raise ValueError("runtime semantic output paths are invalid")
    paths = ", ".join(str(item) for item in output_resource_ids)
    if policy_id == "tracedp_residual_runtime_v3":
        return (
            "\n\n[Runtime supervisor]\n"
            "Runtime supervisor observed a declared output before official evaluation: "
            f"{paths}. Re-read the public task and verify the output through its required "
            "external consumption path. Existence alone does not prove completion. For "
            "a package or directory, check the required install, import, build, "
            "invocation, and behavior from outside its source tree as applicable. For a "
            "file or data artifact, check its public schema, boundaries, serialization, "
            "and behavior as applicable. Use a behaviorally distinct check, complete any "
            "missing delivery step, and correct the output when public evidence supports "
            "a change."
        )
    notice = (
        "\n\n[Runtime supervisor]\n"
        "Runtime supervisor observed a complete declared output before official "
        "evaluation. Perform a behaviorally distinct public semantic check of: "
        f"{paths}. Inspect selection, aggregation units, boundaries between fields or "
        "records, library defaults, and serialization. Repeating the same calculation "
        "is not an independent check. "
    )
    if policy_id in {_DYNAMIC_METHOD_V4, _FIXED_METHOD, *_JOINT_METHODS}:
        return notice + (
            "When plausible interpretations give different results, enumerate them and "
            "cite public task, schema, or documentation evidence for the chosen "
            "interpretation. Do not keep the current value merely because it was written "
            "first. Preserve independently declared field and record boundaries unless "
            "a public source explicitly defines a joined serialization. Correct the "
            "output when the evidence supports a different result."
        )
    if policy_id == _DYNAMIC_METHOD_V3:
        return notice + "Correct the output if the independent result differs."
    raise ValueError("runtime semantic notice policy is unsupported")


def _expected_convergence_notice(output_resource_ids: list[object]) -> str:
    """Rebuild the time-aware finalization message independently of the gateway."""

    if not output_resource_ids or not all(isinstance(item, str) for item in output_resource_ids):
        raise ValueError("runtime convergence output paths are invalid")
    paths = ", ".join(str(item) for item in output_resource_ids)
    return (
        "\n\n[Runtime supervisor]\n"
        "Runtime supervisor observed that the declared output exists and the Actor "
        "session entered its finalization reserve. Perform only a quick public check "
        "if it can finish immediately, then stop calling tools and return the final "
        f"response. Output: {paths}."
    )


def _expected_failure_diagnosis_notice(failure_count: object, *, minimum_failure_count: int) -> str:
    """Rebuild the generic failure message without using gateway formatting code."""

    if (
        not isinstance(failure_count, int)
        or isinstance(failure_count, bool)
        or failure_count < minimum_failure_count
    ):
        raise ValueError("runtime failure diagnosis count is invalid")
    return (
        "\n\n[Runtime supervisor]\n"
        f"Runtime supervisor observed {failure_count} failed Actor tool executions "
        "since the last managed resource change. Reassess the current approach before "
        "repeating it. Use only public task instructions, current tool results, and "
        "available environment capabilities to identify which assumption or validation "
        "path failed; choose a behaviorally different next check or fix. Do not claim "
        "the task requirement is false merely because a tool is unavailable."
    )


def _expected_tracedp_failure_context(
    kernel: PortableRuntimeKernel, *, failure_action_id: object, candidate_task_ids: object
) -> dict[str, object]:
    """Rebuild graph-bound diagnosis input from the durable Kernel state."""

    if not isinstance(failure_action_id, str) or not isinstance(candidate_task_ids, list):
        raise ValueError("TraceDP failure diagnosis identity is invalid")
    failed_action = kernel.actions.get(failure_action_id)
    if failed_action is None:
        raise ValueError("TraceDP failed transition is missing")
    task_ids = tuple(str(item) for item in candidate_task_ids)
    task_rows: list[dict[str, object]] = []
    resource_ids: set[str] = set()
    for task_id in task_ids:
        node = kernel.task_nodes.get(task_id)
        if node is None:
            raise ValueError("TraceDP candidate task is missing")
        resource_ids.update(node.prerequisite_resource_ids)
        resource_ids.update(node.hypothesized_prerequisite_resource_ids)
        resource_ids.update(node.required_output_resource_ids)
        task_rows.append(
            {
                "task_id": task_id,
                "parent_id": node.parent_id,
                "status": kernel.task_statuses[task_id].value,
                "source": node.source.value,
                "confirmed_prerequisite_resource_ids": list(node.prerequisite_resource_ids),
                "hypothesized_prerequisite_resource_ids": list(
                    node.hypothesized_prerequisite_resource_ids
                ),
                "required_output_resource_ids": list(node.required_output_resource_ids),
            }
        )
    return {
        "failed_transition": {
            "action_id": failed_action.action_id,
            "tool_name": failed_action.tool_name,
            "phase": failed_action.phase.value,
            "reason": failed_action.reason,
        },
        "candidate_tasks": task_rows,
        "resource_facts": [
            {
                "resource_id": resource_id,
                "version": kernel.resource_versions[resource_id],
                "observed": kernel.observations[resource_id].observed,
                "exists": kernel.observations[resource_id].exists,
                "kind": kernel.observations[resource_id].kind.value,
            }
            for resource_id in sorted(resource_ids)
        ],
        "evidence": [
            {
                "evidence_id": evidence.evidence_id,
                "task_id": evidence.task_id,
                "state": evidence.state.value,
                "dependency_claims": [
                    [resource_id, role.value] for resource_id, role in evidence.dependency_claims
                ],
            }
            for evidence in kernel.evidence.values()
            if evidence.task_id in task_ids
        ],
    }


def _kernel_at_failure_boundary(
    event_log_path: Path, *, failure_action_id: object
) -> PortableRuntimeKernel:
    """Replay only the durable prefix visible when TraceDP issued its diagnosis."""

    if not isinstance(failure_action_id, str):
        raise ValueError("TraceDP failure diagnosis identity is invalid")
    rows = _read_jsonl(event_log_path)
    boundary_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("event_type") == "actor_failure_boundary_recorded"
            and isinstance(row.get("payload"), dict)
            and row["payload"].get("action_id") == failure_action_id
        ),
        None,
    )
    if boundary_index is None:
        raise ValueError("TraceDP failure boundary is missing")
    with tempfile.TemporaryDirectory() as directory:
        prefix_path = Path(directory) / "runtime-prefix.jsonl"
        prefix_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in rows[: boundary_index + 1]
            )
        )
        return PortableRuntimeKernel.load(prefix_path)


def _expected_tracedp_failure_notice(context: dict[str, object]) -> str:
    """Render an independent exact oracle for the Actor-visible graph projection."""

    failed = context["failed_transition"]
    tasks = context["candidate_tasks"]
    resources = context["resource_facts"]
    assert isinstance(failed, dict) and isinstance(tasks, list) and isinstance(resources, list)
    task_lines = [
        "- {task_id} (parent={parent_id}, status={status}, source={source}); "
        "confirmed inputs={confirmed}; hypothesized inputs={hypothesized}; "
        "required outputs={outputs}".format(
            task_id=row["task_id"],
            parent_id=row["parent_id"],
            status=row["status"],
            source=row["source"],
            confirmed=row["confirmed_prerequisite_resource_ids"],
            hypothesized=row["hypothesized_prerequisite_resource_ids"],
            outputs=row["required_output_resource_ids"],
        )
        for row in tasks
    ]
    resource_lines = [
        "- {resource_id}: version={version}, observed={observed}, exists={exists}, "
        "kind={kind}".format(**row)
        for row in resources
    ]
    return (
        "\n\n[Runtime supervisor]\n"
        "TraceDP residual-state diagnosis follows the observed failed transition "
        f"{failed['action_id']} (tool={failed['tool_name']}, phase={failed['phase']}, "
        f"reason={failed['reason']}). No managed resource version changed at this "
        "boundary, so the failure refutes that execution attempt, not the public task.\n"
        "Open task graph:\n"
        + "\n".join(task_lines)
        + "\nCurrent resource facts:\n"
        + "\n".join(resource_lines)
        + "\nUse this versioned graph to choose a different next check or fix. Treat "
        "hypothesized inputs and absent observations as uncertain; do not claim a root "
        "cause without new evidence."
    )


def _replay_runtime_failure_diagnoses(
    event_log_path: Path, kernel: PortableRuntimeKernel
) -> dict[str, dict[str, object]]:
    """Derive repeated-failure triggers from action outcomes and resource observations."""

    observations: dict[str, object] = {}
    versions: dict[str, int] = {}
    actors: dict[str, str] = {}
    failure_count = 0
    latest_boundary: dict[str, object] | None = None
    diagnoses: dict[str, dict[str, object]] = {}
    completed_incident_keys: set[
        tuple[
            tuple[tuple[str, int], ...],
            tuple[str, str, tuple[tuple[str, int], ...], str],
        ]
    ] = set()

    def apply(raw: object) -> bool:
        if not isinstance(raw, list):
            raise ValueError("runtime failure observations are invalid")
        changed = False
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("resource_id"), str):
                raise ValueError("runtime failure observation is invalid")
            resource_id = str(item["resource_id"])
            if observations.get(resource_id) != item:
                observations[resource_id] = item
                versions[resource_id] = versions.get(resource_id, 0) + 1
                changed = True
        return changed

    for index, row in enumerate(_read_jsonl(event_log_path)):
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime failure event payload is invalid")
        event_type = row.get("event_type")
        if index == 0:
            if event_type != "initialized":
                raise ValueError("runtime failure replay lacks initialization")
            apply(payload.get("observations"))
            continue
        if event_type == "observations_refreshed":
            if apply(payload.get("observations")):
                failure_count = 0
                latest_boundary = None
            continue
        if event_type == "action_proposed":
            action_id = payload.get("action_id")
            actor = payload.get("actor")
            if not isinstance(action_id, str) or not isinstance(actor, str):
                raise ValueError("runtime failure action proposal is invalid")
            actors[action_id] = actor
            action = kernel.actions.get(action_id)
            if (
                actor == "runtime"
                and action is not None
                and action.tool_name == "runtime_failure_diagnosis_guidance"
            ):
                if latest_boundary is None or action_id in diagnoses:
                    raise ValueError("runtime failure diagnosis lacks a current boundary")
                expected_versions = [list(item) for item in sorted(versions.items())]
                if (
                    action.arguments.get("failure_action_id") != latest_boundary["action_id"]
                    or action.arguments.get("failure_count_since_resource_change")
                    != latest_boundary["failure_count_since_resource_change"]
                    or action.arguments.get("failure_resource_versions") != expected_versions
                    or action.arguments.get("failure_candidate_task_ids")
                    != latest_boundary["candidate_task_ids"]
                    or action.arguments.get("failure_association_evidence_id")
                    != "session_open_task_scope_v1"
                ):
                    raise ValueError("runtime failure diagnosis boundary differs")
                raw_obligation_key = action.arguments.get("obligation_version_key")
                if (
                    not isinstance(raw_obligation_key, list)
                    or len(raw_obligation_key) != 4
                    or not isinstance(raw_obligation_key[2], list)
                ):
                    raise ValueError("runtime failure diagnosis obligation differs")
                obligation_versions: list[tuple[str, int]] = []
                for item in raw_obligation_key[2]:
                    if (
                        not isinstance(item, list)
                        or len(item) != 2
                        or not isinstance(item[0], str)
                        or isinstance(item[1], bool)
                        or not isinstance(item[1], int)
                    ):
                        raise ValueError("runtime failure diagnosis obligation differs")
                    obligation_versions.append((item[0], item[1]))
                incident_key = (
                    tuple(sorted(versions.items())),
                    (
                        str(raw_obligation_key[0]),
                        str(raw_obligation_key[1]),
                        tuple(obligation_versions),
                        str(raw_obligation_key[3]),
                    ),
                )
                if action.phase is ActionPhase.COMPLETED:
                    if incident_key in completed_incident_keys:
                        raise ValueError("runtime failure diagnosis incident was repeated")
                    completed_incident_keys.add(incident_key)
                diagnoses[action_id] = {
                    "trigger_action_id": latest_boundary["action_id"],
                    **latest_boundary,
                }
            continue
        if event_type == "action_finished":
            action_id = payload.get("action_id")
            if not isinstance(action_id, str) or action_id not in actors:
                raise ValueError("runtime failure action outcome is invalid")
            changed = apply(payload.get("observations"))
            if changed:
                failure_count = 0
                latest_boundary = None
            if (
                actors[action_id] == "actor"
                and payload.get("phase") in {
                    ActionPhase.FAILED_WITHOUT_EFFECT.value,
                    ActionPhase.FAILED_AFTER_PARTIAL_EFFECT.value,
                    ActionPhase.FAILED_WITH_UNKNOWN_EFFECT.value,
                }
            ):
                failure_count += 1
            continue
        if event_type == "actor_failure_boundary_recorded":
            task_ids = payload.get("candidate_task_ids")
            expected_versions = [list(item) for item in sorted(versions.items())]
            if (
                payload.get("failure_count_since_resource_change") != failure_count
                or payload.get("failure_count_since_resource_change", 0) < 1
                or payload.get("resource_versions") != expected_versions
                or not isinstance(task_ids, list)
                or task_ids != sorted(task_ids)
                or any(task_id not in kernel.task_nodes for task_id in task_ids)
                or payload.get("association_evidence_id") != "session_open_task_scope_v1"
            ):
                raise ValueError("runtime Actor failure boundary replay differs")
            latest_boundary = {
                "action_id": payload.get("action_id"),
                "failure_count_since_resource_change": failure_count,
                "resource_versions": expected_versions,
                "candidate_task_ids": task_ids,
                "association_evidence_id": "session_open_task_scope_v1",
            }
    return diagnoses


def _replay_runtime_guidance_progress(
    event_log_path: Path, kernel: PortableRuntimeKernel
) -> dict[str, dict[str, object]]:
    """Recompute guidance triggers from Kernel events without trusting policy logs."""

    output_ids = frozenset(
        resource_id
        for node in kernel.task_nodes.values()
        for resource_id in node.required_output_resource_ids
    )
    observations: dict[str, object] = {}
    versions: dict[str, int] = {}
    actors: dict[str, str] = {}
    no_progress_actions = 0
    last_actor_action_id: str | None = None
    guidance: dict[str, dict[str, object]] = {}

    def apply_observations(raw: object) -> bool:
        if not isinstance(raw, list):
            raise ValueError("runtime event observations are invalid")
        output_changed = False
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("resource_id"), str):
                raise ValueError("runtime event observation is invalid")
            resource_id = str(item["resource_id"])
            if observations.get(resource_id) != item:
                observations[resource_id] = item
                versions[resource_id] = versions.get(resource_id, 0) + 1
                output_changed = output_changed or resource_id in output_ids
        return output_changed

    for index, row in enumerate(_read_jsonl(event_log_path)):
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime event payload is invalid")
        event_type = row.get("event_type")
        if index == 0:
            if event_type != "initialized":
                raise ValueError("runtime event log lacks initialization")
            apply_observations(payload.get("observations"))
            continue
        if event_type == "action_proposed":
            action_id = payload.get("action_id")
            actor = payload.get("actor")
            if not isinstance(action_id, str) or actor not in {
                "actor",
                "runtime",
                "actor_bridge",
            }:
                raise ValueError("runtime proposed action identity is invalid")
            actors[action_id] = str(actor)
            continue
        if event_type == "observations_refreshed":
            apply_observations(payload.get("observations"))
            continue
        if event_type != "action_finished":
            continue
        action_id = payload.get("action_id")
        if not isinstance(action_id, str) or action_id not in actors:
            raise ValueError("runtime finished action identity is invalid")
        output_changed = apply_observations(payload.get("observations"))
        actor = actors[action_id]
        if actor == "actor":
            no_progress_actions = 0 if output_changed else no_progress_actions + 1
            last_actor_action_id = action_id
            continue
        action = kernel.actions.get(action_id)
        if (
            actor == "actor_bridge"
            and action is not None
            and action.tool_name == "actor_natural_settlement_trigger"
        ):
            last_actor_action_id = action_id
            continue
        if action is not None and action.tool_name == "runtime_guidance":
            guidance[action_id] = {
                "trigger_action_id": last_actor_action_id,
                "no_progress_actor_actions": no_progress_actions,
                "resource_versions": dict(versions),
            }
            no_progress_actions = 0
    return guidance


def _replay_runtime_semantic_obligations(
    event_log_path: Path, kernel: PortableRuntimeKernel
) -> dict[str, dict[str, object]]:
    """Rebuild present-output triggers from observations outside the policy log."""

    observations: dict[str, dict[str, object]] = {}
    versions: dict[str, int] = {}
    actors: dict[str, str] = {}
    last_trigger_action_id: str | None = None
    semantic: dict[str, dict[str, object]] = {}

    def apply(raw: object) -> None:
        if not isinstance(raw, list):
            raise ValueError("runtime event observations are invalid")
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("resource_id"), str):
                raise ValueError("runtime event observation is invalid")
            resource_id = str(item["resource_id"])
            if observations.get(resource_id) != item:
                observations[resource_id] = item
                versions[resource_id] = versions.get(resource_id, 0) + 1

    for index, row in enumerate(_read_jsonl(event_log_path)):
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime event payload is invalid")
        event_type = row.get("event_type")
        if index == 0:
            if event_type != "initialized":
                raise ValueError("runtime event log lacks initialization")
            apply(payload.get("observations"))
            continue
        if event_type == "action_proposed":
            action_id = payload.get("action_id")
            actor = payload.get("actor")
            if not isinstance(action_id, str) or actor not in {
                "actor",
                "runtime",
                "actor_bridge",
            }:
                raise ValueError("runtime proposed action identity is invalid")
            actors[action_id] = str(actor)
            action = kernel.actions.get(action_id)
            if (
                actor == "runtime"
                and action is not None
                and action.tool_name
                in {"runtime_semantic_guidance", "runtime_convergence_guidance"}
            ):
                task_id = action.arguments.get("task_id")
                task = kernel.task_nodes.get(task_id) if isinstance(task_id, str) else None
                if (
                    task is None
                    or not task.public_requirement
                    or not task.required_output_resource_ids
                ):
                    raise ValueError("runtime output guidance task is invalid")
                resources = tuple(sorted(task.required_output_resource_ids))
                existence = tuple(
                    observations.get(resource_id, {}).get("exists") for resource_id in resources
                )
                present = (
                    any(value is True for value in existence)
                    if task.output_requirement.value == "any"
                    else all(value is True for value in existence)
                )
                if not present:
                    raise ValueError("runtime output guidance preceded complete output")
                semantic[action_id] = {
                    "trigger_action_id": last_trigger_action_id,
                    "task_id": task.task_id,
                    "obligation_id": f"public-output-obligation:{task.task_id}",
                    "output_resource_ids": list(resources),
                    "output_requirement": task.output_requirement.value,
                    "resource_versions": [[item, versions[item]] for item in resources],
                    "phase": "output_present_unverified",
                }
            continue
        if event_type == "observations_refreshed":
            apply(payload.get("observations"))
            continue
        if event_type != "action_finished":
            continue
        action_id = payload.get("action_id")
        if not isinstance(action_id, str) or action_id not in actors:
            raise ValueError("runtime finished action identity is invalid")
        apply(payload.get("observations"))
        if actors[action_id] in {"actor", "actor_bridge"}:
            last_trigger_action_id = action_id
            continue
    return semantic


def _expected_output_supervision_triggers(
    event_log_path: Path,
    kernel: PortableRuntimeKernel,
    *,
    budget_units: int,
    policy_id: str,
) -> tuple[tuple[str, str, str], ...]:
    """Derive mandatory fixed-controller triggers without trusting runtime actions."""

    output_tasks = tuple(
        node
        for node in kernel.task_nodes.values()
        if node.public_requirement and node.required_output_resource_ids
    )
    observations: dict[str, dict[str, object]] = {}
    versions: dict[str, int] = {}
    actors: dict[str, str] = {}
    no_progress_actions = 0
    remaining_budget = budget_units
    pending_actor_action_id: str | None = None
    supervision_enabled = True
    guided_versions: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    checked_versions: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    expected: list[tuple[str, str, str]] = []

    def apply(raw: object) -> bool:
        if not isinstance(raw, list):
            raise ValueError("runtime event observations are invalid")
        changed = False
        output_ids = {
            resource_id
            for task in output_tasks
            for resource_id in task.required_output_resource_ids
        }
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("resource_id"), str):
                raise ValueError("runtime event observation is invalid")
            resource_id = str(item["resource_id"])
            if observations.get(resource_id) != item:
                observations[resource_id] = item
                versions[resource_id] = versions.get(resource_id, 0) + 1
                changed = changed or resource_id in output_ids
        return changed

    def select_output_supervision() -> None:
        """Choose after all higher-priority repairs for one Actor trigger finish."""

        nonlocal no_progress_actions, pending_actor_action_id
        action_id = pending_actor_action_id
        if action_id is None:
            return
        pending_actor_action_id = None
        if not supervision_enabled or remaining_budget < 1:
            return
        if policy_id in OUTPUT_SEMANTIC_RUNTIME_METHODS:
            for task in output_tasks:
                resource_ids = tuple(sorted(task.required_output_resource_ids))
                existence = tuple(
                    observations.get(resource_id, {}).get("exists") for resource_id in resource_ids
                )
                complete = (
                    any(value is True for value in existence)
                    if task.output_requirement.value == "any"
                    else all(value is True for value in existence)
                )
                version_key = tuple((item, versions[item]) for item in resource_ids)
                check_key = (task.task_id, version_key)
                if complete and check_key not in checked_versions:
                    expected.append((action_id, "runtime_semantic_guidance", task.task_id))
                    checked_versions.add(check_key)
                    return
        if no_progress_actions < OUTPUT_PROGRESS_ACTION_THRESHOLD:
            return
        for task in output_tasks:
            resource_ids = tuple(sorted(task.required_output_resource_ids))
            existence = tuple(
                observations.get(resource_id, {}).get("exists") for resource_id in resource_ids
            )
            complete = (
                any(value is True for value in existence)
                if task.output_requirement.value == "any"
                else all(value is True for value in existence)
            )
            version_key = tuple((item, versions[item]) for item in resource_ids)
            guidance_key = (task.task_id, version_key)
            if not complete and guidance_key not in guided_versions:
                expected.append((action_id, "runtime_guidance", task.task_id))
                guided_versions.add(guidance_key)
                no_progress_actions = 0
                return

    for index, row in enumerate(_read_jsonl(event_log_path)):
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime event payload is invalid")
        event_type = row.get("event_type")
        if index == 0:
            if event_type != "initialized":
                raise ValueError("runtime event log lacks initialization")
            apply(payload.get("observations"))
            continue
        if event_type == "action_proposed":
            action_id = payload.get("action_id")
            actor = payload.get("actor")
            if not isinstance(action_id, str) or actor not in {
                "actor",
                "runtime",
                "actor_bridge",
            }:
                raise ValueError("runtime proposed action identity is invalid")
            actors[action_id] = str(actor)
            action = kernel.actions.get(action_id)
            if actor == "actor" or (
                actor == "runtime"
                and action is not None
                and action.tool_name in {"runtime_guidance", "runtime_semantic_guidance"}
            ):
                select_output_supervision()
            continue
        if event_type == "runtime_supervision_disabled":
            supervision_enabled = False
            pending_actor_action_id = None
            continue
        if event_type != "action_finished":
            continue
        action_id = payload.get("action_id")
        if not isinstance(action_id, str) or action_id not in actors:
            raise ValueError("runtime finished action identity is invalid")
        output_changed = apply(payload.get("observations"))
        if actors[action_id] == "runtime":
            remaining_budget = max(0, remaining_budget - 1)
            action = kernel.actions.get(action_id)
            if action is not None and action.tool_name == "runtime_guidance":
                no_progress_actions = 0
            continue
        if actors[action_id] == "actor_bridge":
            continue
        no_progress_actions = 0 if output_changed else no_progress_actions + 1
        pending_actor_action_id = action_id
    select_output_supervision()
    return tuple(expected)


def _validate_joint_decisions(
    path: Path,
    kernel: PortableRuntimeKernel,
    *,
    event_log_path: Path,
    budget_units: int,
    method_id: str,
    source_arm_index: int | None,
    decision_resolver: RuntimeJointDecisionResolver | None = None,
    actor_time_budget_ms: int | None = None,
    predicate_guidance_evidence: Mapping[str, Mapping[str, object]] | None = None,
    registered_check_guidance_evidence: Mapping[str, Mapping[str, object]] | None = None,
    runtime_check_execution_evidence: Mapping[str, Mapping[str, object]] | None = None,
    targeted_repair_execution_evidence: Mapping[str, Mapping[str, object]] | None = None,
) -> int:
    """Independently bind every joint selection to one real runtime action."""

    if not path.is_file() or not path.read_text().strip():
        supervised_boundary_exists = any(
            (
                action.actor == "actor"
                and action.phase
                in {
                    ActionPhase.COMPLETED,
                    ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
                    ActionPhase.FAILED_WITHOUT_EFFECT,
                    ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
                }
            )
            or (
                action.actor == "actor_bridge"
                and action.tool_name
                in {"actor_time_reserve_trigger", "actor_natural_settlement_trigger"}
            )
            for action in kernel.actions.values()
        )
        if supervised_boundary_exists or any(
            action.actor == "runtime" for action in kernel.actions.values()
        ):
            raise ValueError("runtime joint policy decision log is missing or empty")
        return budget_units
    replayed = RuntimeJointSlatePolicy.load(
        method_id=method_id,
        budget_units=budget_units,
        event_log_path=path,
        source_arm_index=source_arm_index,
        decision_resolver=decision_resolver,
    )
    if replayed.has_open_decision:
        raise ValueError("runtime joint policy has an unresolved decision")
    action_versions, action_order = _kernel_action_version_snapshots(event_log_path)
    guidance_progress = _replay_runtime_guidance_progress(event_log_path, kernel)
    semantic_obligations = _replay_runtime_semantic_obligations(event_log_path, kernel)
    failure_diagnoses = _replay_runtime_failure_diagnoses(event_log_path, kernel)
    pending: dict[str, object] | None = None
    selected_trigger_ids: set[str] = set()
    executed_action_ids: set[str] = set()
    previous_time_remaining_ms: int | None = None
    completed_chain_check: dict[str, object] | None = None
    active_option_chain: dict[str, object] | None = None
    for row in _read_jsonl(path):
        event_type = row.get("event_type")
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("runtime joint policy payload is missing")
        if event_type == "decision_selected":
            if pending is not None:
                raise ValueError("runtime joint policy decisions overlap")
            slate_payload = payload.get("slate")
            if not isinstance(slate_payload, dict):
                raise ValueError("runtime joint policy slate is missing")
            slate = RuntimeJointActionSlate.from_dict(slate_payload)
            trigger = kernel.actions.get(slate.trigger_action_id)
            if method_id == FIXED_JOINT_SLATE_V3_METHOD:
                if (
                    slate.actor_time_budget_ms != actor_time_budget_ms
                    or slate.actor_time_remaining_ms is None
                    or (
                        previous_time_remaining_ms is not None
                        and slate.actor_time_remaining_ms > previous_time_remaining_ms
                    )
                ):
                    raise ValueError("runtime joint Actor time state differs")
                previous_time_remaining_ms = slate.actor_time_remaining_ms
            elif method_id in _RPC_METHODS:
                time_aware_selector = method_id in _TIME_AWARE_SELECTOR_METHODS
                if time_aware_selector:
                    if (
                        slate.actor_time_budget_ms != actor_time_budget_ms
                        or slate.actor_time_remaining_ms is None
                        or (
                            previous_time_remaining_ms is not None
                            and slate.actor_time_remaining_ms > previous_time_remaining_ms
                        )
                    ):
                        raise ValueError("time-aware selector Actor state differs")
                    previous_time_remaining_ms = slate.actor_time_remaining_ms
                bridge_trigger = trigger is not None and trigger.actor == "actor_bridge"
                if bridge_trigger:
                    if trigger.tool_name == "actor_time_reserve_trigger":
                        if (
                            method_id not in _TIME_RESERVE_METHODS
                            or slate.actor_time_budget_ms != actor_time_budget_ms
                            or slate.actor_time_remaining_ms is None
                            or slate.actor_time_remaining_ms
                            > ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000
                        ):
                            raise ValueError("runtime joint Actor Bridge time state differs")
                    elif trigger.tool_name == RUNTIME_OPTION_FOLLOWUP_TRIGGER:
                        check_id = trigger.arguments.get("after_check_action_id")
                        check = kernel.actions.get(check_id) if isinstance(check_id, str) else None
                        if (
                            method_id not in RUNTIME_OPTION_CHAIN_METHODS
                            or not time_aware_selector
                            or set(trigger.arguments) != {"after_check_action_id"}
                            or trigger.phase is not ActionPhase.COMPLETED
                            or check is None
                            or check.actor != "runtime"
                            or check.tool_name != "runtime_registered_check_execution"
                            or check_id not in executed_action_ids
                            or action_order[trigger.action_id] <= action_order[check_id]
                            or any(
                                action.actor == "actor"
                                and action_order[check_id]
                                < action_order[action_id]
                                < action_order[trigger.action_id]
                                for action_id, action in kernel.actions.items()
                            )
                            or trigger.action_id
                            != "actor-bridge-runtime-option-followup:" + check_id
                        ):
                            raise ValueError("runtime option follow-up trigger differs")
                    elif (
                        trigger.tool_name == "actor_natural_settlement_trigger"
                        and method_id in RUNTIME_OPTION_CHAIN_METHODS
                    ):
                        # Option-chain methods decide again where the Actor settles
                        # on its own; the trigger is a bridge fact with no Actor
                        # action after the previous boundary changing its meaning.
                        if (
                            not time_aware_selector
                            or set(trigger.arguments) != {"review_index"}
                            or not isinstance(trigger.arguments.get("review_index"), int)
                            or trigger.phase is not ActionPhase.COMPLETED
                            or not trigger.action_id.startswith(
                                "actor-bridge-natural-settlement-"
                            )
                        ):
                            raise ValueError("runtime natural settlement trigger differs")
                    elif trigger.tool_name == "runtime_option_replan_trigger":
                        option_ids = trigger.arguments.get(
                            "conditional_future_option_ids"
                        )
                        repair_candidates = [
                            item
                            for item in slate.candidates
                            if item.action_type
                            is RuntimeJointActionType.RUN_TARGETED_REPAIR
                        ]
                        if (
                            method_id not in RUNTIME_OPTION_CHAIN_METHODS
                            or not time_aware_selector
                            or completed_chain_check is None
                            or not isinstance(option_ids, list)
                            or not option_ids
                            or option_ids
                            != completed_chain_check["conditional_future_option_ids"]
                            or trigger.arguments
                            != {
                                "option_chain_id": completed_chain_check[
                                    "option_chain_id"
                                ],
                                "parent_trigger_action_id": completed_chain_check[
                                    "parent_trigger_action_id"
                                ],
                                "prerequisite_decision_id": completed_chain_check[
                                    "decision_id"
                                ],
                                "prerequisite_instance_action_id": (
                                    completed_chain_check["instance_action_id"]
                                ),
                                "prerequisite_evidence_state": (
                                    completed_chain_check["evidence_state"]
                                ),
                                "conditional_future_option_ids": option_ids,
                            }
                            or trigger.phase is not ActionPhase.COMPLETED
                            or action_order[trigger.action_id]
                            <= action_order[str(completed_chain_check["action_id"])]
                            or any(
                                action.actor == "actor"
                                and action_order[str(completed_chain_check["action_id"])]
                                < action_order[action_id]
                                < action_order[trigger.action_id]
                                for action_id, action in kernel.actions.items()
                            )
                            or any(
                                item.conditional_future_option_id not in option_ids
                                for item in repair_candidates
                            )
                            or any(
                                item.action_type
                                not in {
                                    RuntimeJointActionType.CONTINUE_ACTOR,
                                    RuntimeJointActionType.RUN_TARGETED_REPAIR,
                                }
                                for item in slate.candidates
                            )
                            or slate.conditional_future_options
                        ):
                            raise ValueError(
                                "runtime option replan trigger differs"
                            )
                        completed_chain_check = None
                    elif (
                        method_id in _SETTLEMENT_REVIEW_METHODS
                        and trigger.tool_name == "actor_natural_settlement_trigger"
                    ):
                        if (
                            not time_aware_selector
                            and (
                                slate.actor_time_budget_ms is not None
                                or slate.actor_time_remaining_ms is not None
                            )
                        ):
                            raise ValueError("settlement review recorded Actor time state")
                    else:
                        raise ValueError("runtime joint Actor Bridge trigger differs")
                elif not time_aware_selector and (
                    slate.actor_time_budget_ms is not None
                    or slate.actor_time_remaining_ms is not None
                ):
                    raise ValueError("tool-triggered v4 slate recorded Actor time state")
            elif (
                slate.actor_time_budget_ms is not None or slate.actor_time_remaining_ms is not None
            ):
                raise ValueError("non-time-aware joint method recorded Actor time state")
            if (
                trigger is None
                or trigger.actor
                not in ({"actor", "actor_bridge"} if method_id in _RPC_METHODS else {"actor"})
                or slate.trigger_action_id not in action_versions
                or slate.trigger_action_id in selected_trigger_ids
                or dict(slate.resource_versions) != action_versions[slate.trigger_action_id]
            ):
                raise ValueError("runtime joint policy trigger state differs")
            selected_trigger_ids.add(slate.trigger_action_id)
            selected = next(
                (
                    item
                    for item in slate.candidates
                    if item.instance_action_id == payload.get("selected_instance_action_id")
                ),
                None,
            )
            if (
                selected is None
                or payload.get("selected_action_type") != selected.action_type.value
            ):
                raise ValueError("runtime joint selected action is absent from its slate")
            if selected.action_type is RuntimeJointActionType.CONTINUE_ACTOR:
                continue
            pending = {
                "decision_id": payload.get("decision_id"),
                "trigger_action_id": slate.trigger_action_id,
                "candidate": selected,
                "executed_action_id": None,
                "no_progress_actor_actions": slate.no_progress_actor_actions,
                "progress_trigger_threshold": slate.progress_trigger_threshold,
                "actor_time_budget_ms": slate.actor_time_budget_ms,
                "actor_time_remaining_ms": slate.actor_time_remaining_ms,
            }
        elif event_type == "execution_started":
            action_id = payload.get("executed_action_id")
            if (
                pending is None
                or pending["executed_action_id"] is not None
                or payload.get("decision_id") != pending["decision_id"]
                or not isinstance(action_id, str)
                or not action_id
            ):
                raise ValueError("runtime joint execution start differs")
            pending["executed_action_id"] = action_id
        elif event_type == "execution_outcome":
            if pending is None:
                raise ValueError("runtime joint outcome has no selected action")
            action_id = pending["executed_action_id"]
            candidate = pending["candidate"]
            if not isinstance(action_id, str) or not isinstance(
                candidate, RuntimeJointActionCandidate
            ):
                raise ValueError("runtime joint outcome lacks execution identity")
            expected_tool = {
                RuntimeJointActionType.RESTORE_RESOURCE: "restore_resource",
                RuntimeJointActionType.REQUEST_OUTPUT_COMPLETION: "runtime_guidance",
                RuntimeJointActionType.REQUEST_ACTOR_CONVERGENCE: ("runtime_convergence_guidance"),
                RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS: (
                    "runtime_failure_diagnosis_guidance"
                ),
                RuntimeJointActionType.SUBMIT_CURRENT_STATE: "submit_current_state",
                RuntimeJointActionType.REQUEST_SEMANTIC_CHECK: ("runtime_semantic_guidance"),
                RuntimeJointActionType.REQUEST_REGISTERED_CHECK: (
                    "runtime_registered_check_guidance"
                ),
                RuntimeJointActionType.RUN_REGISTERED_CHECK: (
                    "runtime_registered_check_execution"
                ),
                RuntimeJointActionType.RUN_TARGETED_REPAIR: (
                    "runtime_targeted_repair_execution"
                ),
                RuntimeJointActionType.REQUEST_PREDICATE_DIAGNOSIS: "runtime_predicate_guidance",
            }.get(candidate.action_type)
            action = kernel.actions.get(action_id)
            trigger_id = str(pending["trigger_action_id"])
            binding = candidate.target_binding
            if (
                action is None
                or action.actor != "runtime"
                or action.tool_name != expected_tool
                or action.arguments.get("resource_id") != binding.resource_id
                or action_id not in action_order
                or action_order[action_id] <= action_order[trigger_id]
                or binding.resource_version
                != action_versions[trigger_id].get(str(binding.resource_id))
                or action_id in executed_action_ids
            ):
                raise ValueError("runtime joint outcome has no matching Kernel action")
            expected_status = (
                "completed"
                if action.phase is ActionPhase.COMPLETED
                else "cancelled_with_unknown_effect"
                if action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
                else "failed"
            )
            if payload.get("status") != expected_status:
                raise ValueError("runtime joint outcome phase differs from Kernel")
            if expected_tool == "runtime_registered_check_guidance":
                evidence = (registered_check_guidance_evidence or {}).get(action_id)
                if (
                    evidence is None
                    or evidence.get("runtime_action_id") != action_id
                    or evidence.get("trigger_action_id") != trigger_id
                    or evidence.get("instance_action_id") != candidate.instance_action_id
                    or evidence.get("arguments") != dict(action.arguments)
                    or action.arguments.get("task_id") != binding.task_id
                ):
                    raise ValueError(
                        "runtime joint registered check guidance lacks audited binding"
                    )
            elif expected_tool == "runtime_registered_check_execution":
                evidence = (runtime_check_execution_evidence or {}).get(action_id)
                if (
                    evidence is None
                    or evidence.get("runtime_action_id") != action_id
                    or evidence.get("trigger_action_id") != trigger_id
                    or evidence.get("instance_action_id") != candidate.instance_action_id
                    or evidence.get("arguments") != dict(action.arguments)
                    or action.arguments.get("task_id") != binding.task_id
                ):
                    raise ValueError(
                        "runtime joint registered check execution lacks audited binding"
                    )
                receipt = evidence.get("receipt")
                evidence_state = (
                    receipt.get("evidence_state")
                    if isinstance(receipt, Mapping)
                    else None
                )
                matching_options = [
                    item.option_id
                    for item in slate.conditional_future_options
                    if item.prerequisite_action_id == candidate.instance_action_id
                ]
                completed_chain_check = (
                    {
                        "action_id": action_id,
                        "decision_id": pending["decision_id"],
                        "instance_action_id": candidate.instance_action_id,
                        "parent_trigger_action_id": trigger_id,
                        "evidence_state": evidence_state,
                        "conditional_future_option_ids": matching_options,
                        "option_chain_id": (
                            active_option_chain["chain_id"]
                            if active_option_chain is not None
                            else None
                        ),
                    }
                    if method_id in RUNTIME_OPTION_CHAIN_METHODS
                    and active_option_chain is not None
                    and active_option_chain.get("expected_check_action_id")
                    == action_id
                    and evidence_state in {"refuted", "unknown"}
                    and matching_options
                    else None
                )
            elif expected_tool == "runtime_targeted_repair_execution":
                evidence = (targeted_repair_execution_evidence or {}).get(action_id)
                if (
                    evidence is None
                    or evidence.get("runtime_action_id") != action_id
                    or evidence.get("trigger_action_id") != trigger_id
                    or evidence.get("instance_action_id") != candidate.instance_action_id
                    or evidence.get("arguments") != dict(action.arguments)
                    or action.arguments.get("task_id") != binding.task_id
                ):
                    raise ValueError(
                        "runtime joint targeted repair execution lacks audited binding"
                    )
            elif expected_tool == "runtime_predicate_guidance":
                evidence = (predicate_guidance_evidence or {}).get(action_id)
                if (
                    evidence is None
                    or evidence.get("runtime_action_id") != action_id
                    or evidence.get("trigger_action_id") != trigger_id
                    or evidence.get("instance_action_id") != candidate.instance_action_id
                    or evidence.get("arguments") != dict(action.arguments)
                    or action.arguments.get("task_id") != binding.task_id
                ):
                    raise ValueError("runtime joint predicate guidance lacks audited execution binding")
            elif expected_tool == "runtime_guidance":
                evidence = guidance_progress.get(action_id)
                if (
                    evidence is None
                    or evidence.get("trigger_action_id") != trigger_id
                    or action.arguments.get("task_id") != binding.task_id
                    or action.arguments.get("obligation_id") != binding.obligation_id
                    or action.arguments.get("obligation_version_key")
                    != binding.to_dict()["obligation_version_key"]
                    or action.arguments.get("no_progress_actor_actions")
                    != pending["no_progress_actor_actions"]
                    or action.arguments.get("trigger_threshold")
                    != pending["progress_trigger_threshold"]
                ):
                    raise ValueError("runtime joint progress evidence differs")
            elif expected_tool == "runtime_semantic_guidance":
                evidence = semantic_obligations.get(action_id)
                if (
                    evidence is None
                    or evidence.get("trigger_action_id") != trigger_id
                    or action.arguments.get("task_id") != binding.task_id
                    or action.arguments.get("obligation_id") != binding.obligation_id
                    or action.arguments.get("obligation_version_key")
                    != binding.to_dict()["obligation_version_key"]
                ):
                    raise ValueError("runtime joint semantic evidence differs")
            elif expected_tool == "runtime_failure_diagnosis_guidance":
                evidence = failure_diagnoses.get(action_id)
                failure_action_id = action.arguments.get("failure_action_id")
                expected_instance_id = (
                    f"diagnose-failure:{failure_action_id}:"
                    f"{action.arguments.get('failure_count_since_resource_change')}:"
                    f"{binding.obligation_id}"
                )
                if (
                    evidence is None
                    or not isinstance(failure_action_id, str)
                    or evidence.get("trigger_action_id") != failure_action_id
                    # A retained failure boundary may be selected at a later action only
                    # while the compiler-bound resource version remains unchanged.
                    or failure_action_id not in action_order
                    or action_order[failure_action_id] > action_order[trigger_id]
                    or candidate.instance_action_id != expected_instance_id
                    or action.arguments.get("task_id") != binding.task_id
                    or action.arguments.get("obligation_id") != binding.obligation_id
                    or action.arguments.get("obligation_version_key")
                    != binding.to_dict()["obligation_version_key"]
                ):
                    raise ValueError("runtime joint failure diagnosis differs")
            elif expected_tool == "runtime_convergence_guidance":
                action_remaining = action.arguments.get("actor_time_remaining_ms")
                if (
                    action.arguments.get("task_id") != binding.task_id
                    or action.arguments.get("obligation_id") != binding.obligation_id
                    or action.arguments.get("obligation_version_key")
                    != binding.to_dict()["obligation_version_key"]
                    or action.arguments.get("actor_time_budget_ms")
                    != pending["actor_time_budget_ms"]
                    or not isinstance(action_remaining, int)
                    or isinstance(action_remaining, bool)
                    or action_remaining < 0
                    or action_remaining > pending["actor_time_remaining_ms"]
                    or action.arguments.get("convergence_reserve_ms")
                    != ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000
                    or pending["actor_time_remaining_ms"] > ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000
                ):
                    raise ValueError("runtime joint convergence evidence differs")
            elif expected_tool == "submit_current_state":
                if (
                    trigger.actor != "actor_bridge"
                    or action.arguments.get("task_id") != binding.task_id
                    or action.arguments.get("obligation_id") != binding.obligation_id
                    or action.arguments.get("obligation_version_key")
                    != binding.to_dict()["obligation_version_key"]
                    or action.arguments.get("joint_decision_id") != pending["decision_id"]
                    or action.arguments.get("actor_time_budget_ms")
                    != pending["actor_time_budget_ms"]
                    or action.arguments.get("actor_time_remaining_ms")
                    != pending["actor_time_remaining_ms"]
                    or action.arguments.get("convergence_reserve_ms")
                    != ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000
                    or action.phase is not ActionPhase.COMPLETED
                ):
                    raise ValueError("runtime joint state submission evidence differs")
            executed_action_ids.add(action_id)
            pending = None
        elif event_type == "option_chain_opened":
            option_ids = payload.get("conditional_future_option_ids")
            if (
                method_id not in RUNTIME_OPTION_CHAIN_METHODS
                or active_option_chain is not None
                or pending is None
                or payload.get("prerequisite_decision_id") != pending["decision_id"]
                or payload.get("prerequisite_instance_action_id")
                != pending["candidate"].instance_action_id
                or not isinstance(option_ids, list)
                or not option_ids
            ):
                raise ValueError("runtime option chain opening differs")
            active_option_chain = payload
        elif event_type == "option_chain_closed":
            replan_id = payload.get("replan_trigger_action_id")
            if (
                active_option_chain is None
                or payload.get("chain_id") != active_option_chain.get("chain_id")
                or replan_id is not None
                and (
                    not isinstance(replan_id, str)
                    or replan_id not in kernel.actions
                    or kernel.actions[replan_id].tool_name
                    != "runtime_option_replan_trigger"
                )
            ):
                raise ValueError("runtime option chain closure differs")
            active_option_chain = None
        elif event_type == "decision_abandoned":
            if (
                pending is None
                or payload.get("decision_id") != pending["decision_id"]
                or (
                    isinstance(pending["executed_action_id"], str)
                    and pending["executed_action_id"] in kernel.actions
                )
            ):
                raise ValueError("runtime joint abandonment crossed the Kernel boundary")
            pending = None
        else:
            raise ValueError("runtime joint policy event type is unsupported")
    if pending is not None:
        raise ValueError("runtime joint policy has an unresolved action")
    if active_option_chain is not None or completed_chain_check is not None:
        raise ValueError("runtime option chain is not closed")
    runtime_action_ids = {
        action_id for action_id, action in kernel.actions.items() if action.actor == "runtime"
    }
    if executed_action_ids != runtime_action_ids:
        raise ValueError("runtime joint decisions do not cover Kernel actions")
    eligible_actor_ids = {
        action_id
        for action_id, action in kernel.actions.items()
        if action.actor == "actor"
        and action.phase
        in {
            ActionPhase.COMPLETED,
            ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
            ActionPhase.FAILED_WITHOUT_EFFECT,
            ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
        }
    }
    kernel_rows = _read_jsonl(event_log_path)
    disable_orders = [
        index
        for index, row in enumerate(kernel_rows)
        if row.get("event_type") == "runtime_supervision_disabled"
    ]
    if len(disable_orders) > 1:
        raise ValueError("runtime joint supervision was disabled repeatedly")
    expected_trigger_ids = eligible_actor_ids
    allowed_trigger_ids = set(eligible_actor_ids)
    if method_id in _RPC_METHODS:
        allowed_bridge_tools = set()
        if method_id in _TIME_RESERVE_METHODS:
            allowed_bridge_tools.add("actor_time_reserve_trigger")
        if method_id in _SETTLEMENT_REVIEW_METHODS:
            allowed_bridge_tools.add("actor_natural_settlement_trigger")
        if method_id in RUNTIME_OPTION_CHAIN_METHODS:
            allowed_bridge_tools.add("runtime_option_replan_trigger")
            allowed_bridge_tools.add(RUNTIME_OPTION_FOLLOWUP_TRIGGER)
            allowed_bridge_tools.add("actor_natural_settlement_trigger")
        bridge_trigger_ids = {
            action_id
            for action_id, action in kernel.actions.items()
            if action.actor == "actor_bridge" and action.tool_name in allowed_bridge_tools
        }
        allowed_trigger_ids.update(bridge_trigger_ids)
        expected_trigger_ids.update(
            action_id
            for action_id in bridge_trigger_ids
            if kernel.actions[action_id].tool_name
            in {
                "runtime_option_replan_trigger",
                RUNTIME_OPTION_FOLLOWUP_TRIGGER,
                "actor_natural_settlement_trigger",
            }
            and not (
                method_id in _SETTLEMENT_REVIEW_METHODS
                and kernel.actions[action_id].tool_name
                == "actor_natural_settlement_trigger"
            )
        )
    if disable_orders:
        before_disable = {
            action_id
            for action_id in eligible_actor_ids
            if action_order[action_id] < disable_orders[0]
        }
        if before_disable:
            before_disable.remove(max(before_disable, key=lambda item: action_order[item]))
        expected_trigger_ids = before_disable
    if not expected_trigger_ids <= selected_trigger_ids or any(
        action_id not in allowed_trigger_ids
        or (disable_orders and action_order[action_id] > disable_orders[0])
        for action_id in selected_trigger_ids
    ):
        raise ValueError("runtime joint decisions do not cover every Actor trigger")
    return replayed.remaining_budget_units


def _validate_dynamic_decisions(
    path: Path,
    kernel: PortableRuntimeKernel,
    *,
    event_log_path: Path,
    budget_units: int,
    policy_id: str,
) -> int:
    """Replay policy bookkeeping, then bind each decision to exact Kernel history."""

    if not path.is_file():
        raise ValueError("dynamic TraceDP runtime decision log is missing")
    if not path.read_text().strip():
        raise ValueError("dynamic TraceDP runtime decision log is empty")
    # Reuse the runtime's own replay contract for sequence, schema, and budget history.
    replayed = DynamicTraceDPRuntimePolicy.load(
        budget_units=budget_units,
        decision_log_path=path,
        policy_id=policy_id,
    )
    if replayed.has_open_decision:
        raise ValueError("dynamic TraceDP runtime has an unresolved decision")

    action_versions, action_order = _kernel_action_version_snapshots(event_log_path)
    guidance_progress = _replay_runtime_guidance_progress(event_log_path, kernel)
    semantic_obligations = _replay_runtime_semantic_obligations(event_log_path, kernel)
    kernel_rows = _read_jsonl(event_log_path)
    disable_orders = [
        index
        for index, row in enumerate(kernel_rows)
        if row.get("event_type") == "runtime_supervision_disabled"
    ]
    if len(disable_orders) > 1:
        raise ValueError("dynamic TraceDP runtime has repeated disable events")
    pending: dict[str, object] | None = None
    executed_action_ids: set[str] = set()
    planned_trigger_ids: set[str] = set()
    trigger_terminal_state: dict[str, str] = {}
    remaining_budget = budget_units
    active_trigger: str | None = None
    current_versions: dict[str, int] | None = None
    current_action_order = -1
    for row in _read_jsonl(path):
        event_type = row.get("event_type")
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("dynamic TraceDP runtime decision payload is missing")
        if event_type == "decision_planned":
            if pending is not None:
                raise ValueError("dynamic TraceDP runtime has an unresolved decision")
            decision_id = payload.get("decision_id")
            planned_action_id = payload.get("action_id")
            resource_id = payload.get("resource_id")
            trigger_action_id = payload.get("trigger_action_id")
            if payload.get("policy_id") != policy_id or not isinstance(decision_id, str):
                raise ValueError("dynamic TraceDP planned decision identity is invalid")
            trigger = kernel.actions.get(trigger_action_id)
            if (
                not isinstance(trigger_action_id, str)
                or trigger is None
                or trigger.actor != "actor"
                or trigger_action_id not in action_versions
            ):
                raise ValueError("dynamic TraceDP trigger action is invalid")
            planned_trigger_ids.add(trigger_action_id)
            trigger_terminal_state[trigger_action_id] = (
                "abstain" if planned_action_id is None else "action"
            )
            if trigger_action_id != active_trigger:
                if action_order[trigger_action_id] <= current_action_order:
                    raise ValueError("dynamic TraceDP trigger action order is invalid")
                active_trigger = trigger_action_id
                current_versions = action_versions[trigger_action_id]
                current_action_order = action_order[trigger_action_id]
            if payload.get("resource_versions") != current_versions:
                raise ValueError("dynamic TraceDP resource versions differ from Kernel history")
            diagnostics = payload.get("diagnostics")
            if policy_id == _FIXED_METHOD:
                candidate_rows = (
                    diagnostics.get("candidates") if isinstance(diagnostics, dict) else None
                )
                affordable = sorted(
                    candidate_id
                    for candidate_id, row_value in (
                        candidate_rows.items() if isinstance(candidate_rows, dict) else ()
                    )
                    if isinstance(row_value, dict)
                    and isinstance(row_value.get("cost_units"), int)
                    and not isinstance(row_value.get("cost_units"), bool)
                    and int(row_value["cost_units"]) <= remaining_budget
                )
                expected_fixed_action = affordable[0] if affordable else None
                if (
                    not isinstance(diagnostics, dict)
                    or diagnostics.get("decision_rule") != "fixed_first_legal_candidate_v1"
                    or diagnostics.get("q_values") != {}
                    or diagnostics.get("priority_advantages") != {}
                    or planned_action_id != expected_fixed_action
                ):
                    raise ValueError("fixed runtime decision rule differs")
            # A null action is a durable abstention and therefore needs no outcome row.
            if planned_action_id is None:
                if (
                    resource_id is not None
                    or not isinstance(diagnostics, dict)
                    or diagnostics.get("chosen_action_id") != "abstain"
                ):
                    raise ValueError("dynamic TraceDP abstention diagnostics are invalid")
                continue
            if not isinstance(planned_action_id, str) or not isinstance(resource_id, str):
                raise ValueError("dynamic TraceDP planned action identity is invalid")
            candidates = diagnostics.get("candidates") if isinstance(diagnostics, dict) else None
            candidate = candidates.get(planned_action_id) if isinstance(candidates, dict) else None
            cost_units = candidate.get("cost_units") if isinstance(candidate, dict) else None
            action_kind = (
                candidate.get("action_kind", "restore_resource")
                if isinstance(candidate, dict)
                else None
            )
            if (
                not isinstance(diagnostics, dict)
                or diagnostics.get("chosen_action_id") != planned_action_id
                or not isinstance(candidate, dict)
                or candidate.get("resource_id") != resource_id
                or not isinstance(cost_units, int)
                or isinstance(cost_units, bool)
                or cost_units < 1
                or action_kind
                not in {
                    "restore_resource",
                    "runtime_guidance",
                    "runtime_semantic_guidance",
                }
                or (
                    action_kind == "runtime_guidance"
                    and policy_id not in OUTPUT_PROGRESS_RUNTIME_METHODS
                )
                or (
                    action_kind == "runtime_semantic_guidance"
                    and policy_id not in OUTPUT_SEMANTIC_RUNTIME_METHODS
                )
            ):
                raise ValueError("dynamic TraceDP planned diagnostics binding is invalid")
            pending = {
                "decision_id": decision_id,
                "planned_action_id": planned_action_id,
                "resource_id": resource_id,
                "cost_units": cost_units,
                "action_kind": action_kind,
                "trigger_action_id": trigger_action_id,
                "trigger_evidence": diagnostics.get("trigger_evidence"),
                "executed_action_id": None,
            }
        elif event_type == "decision_execution_started":
            if pending is None:
                raise ValueError("dynamic TraceDP execution start has no planned decision")
            executed_action_id = payload.get("executed_action_id")
            if (
                payload.get("decision_id") != pending["decision_id"]
                or payload.get("planned_action_id") != pending["planned_action_id"]
                or payload.get("resource_id") != pending["resource_id"]
                or not isinstance(executed_action_id, str)
                or not executed_action_id
                or pending["executed_action_id"] is not None
            ):
                raise ValueError("dynamic TraceDP execution start differs from its plan")
            pending["executed_action_id"] = executed_action_id
        elif event_type == "decision_outcome":
            if pending is None:
                raise ValueError("dynamic TraceDP outcome has no planned decision")
            decision_id = pending["decision_id"]
            planned_action_id = pending["planned_action_id"]
            resource_id = pending["resource_id"]
            bound_action_id = pending["executed_action_id"]
            if (
                payload.get("decision_id") != decision_id
                or payload.get("planned_action_id") != planned_action_id
                or payload.get("resource_id") != resource_id
                or payload.get("executed_action_id") != bound_action_id
                or payload.get("cost_units") != pending["cost_units"]
            ):
                raise ValueError("dynamic TraceDP outcome differs from its plan")
            executed_action_id = payload.get("executed_action_id")
            if not isinstance(executed_action_id, str) or not executed_action_id:
                raise ValueError("dynamic TraceDP executed action identity is missing")
            action = kernel.actions.get(executed_action_id)
            expected_tool = pending["action_kind"]
            if (
                action is None
                or action.actor != "runtime"
                or action.tool_name != expected_tool
                or action.arguments.get("resource_id") != resource_id
                or executed_action_id not in action_versions
                or action_order[executed_action_id] <= current_action_order
            ):
                raise ValueError("dynamic TraceDP outcome has no runtime action")
            if expected_tool == "runtime_guidance":
                task_id = action.arguments.get("task_id")
                task = kernel.task_nodes.get(task_id) if isinstance(task_id, str) else None
                trigger_evidence = pending.get("trigger_evidence")
                replayed_progress = guidance_progress.get(executed_action_id)
                if (
                    task is None
                    or not task.required_output_resource_ids
                    or not isinstance(trigger_evidence, dict)
                    or replayed_progress is None
                    or replayed_progress.get("trigger_action_id")
                    != pending.get("trigger_action_id")
                    or replayed_progress.get("resource_versions") != current_versions
                    or replayed_progress.get("no_progress_actor_actions")
                    != trigger_evidence.get("no_progress_actor_actions")
                    or replayed_progress.get("no_progress_actor_actions")
                    != action.arguments.get("no_progress_actor_actions")
                    or int(replayed_progress["no_progress_actor_actions"])
                    < OUTPUT_PROGRESS_ACTION_THRESHOLD
                    or action.arguments.get("resource_version")
                    != current_versions.get(str(resource_id))
                    or action.arguments.get("output_resource_ids")
                    != list(task.required_output_resource_ids)
                    or action.arguments.get("output_requirement") != task.output_requirement.value
                    or action.arguments.get("no_progress_actor_actions")
                    != trigger_evidence.get("no_progress_actor_actions")
                    or action.arguments.get("trigger_threshold") != OUTPUT_PROGRESS_ACTION_THRESHOLD
                    or trigger_evidence.get("trigger_threshold") != OUTPUT_PROGRESS_ACTION_THRESHOLD
                    or trigger_evidence.get("progress_definition")
                    != "declared_public_output_observation_changed"
                ):
                    raise ValueError("dynamic TraceDP guidance evidence differs")
            elif expected_tool == "runtime_semantic_guidance":
                replayed_semantic = semantic_obligations.get(executed_action_id)
                trigger_evidence = pending.get("trigger_evidence")
                action_key = action.arguments.get("obligation_version_key")
                expected_key = (
                    [
                        replayed_semantic["task_id"],
                        replayed_semantic["output_requirement"],
                        replayed_semantic["resource_versions"],
                        replayed_semantic["phase"],
                    ]
                    if replayed_semantic is not None
                    else None
                )
                obligation_rows = (
                    trigger_evidence.get("obligations")
                    if isinstance(trigger_evidence, dict)
                    else None
                )
                obligation = (
                    obligation_rows.get(replayed_semantic["obligation_id"])
                    if isinstance(obligation_rows, dict) and replayed_semantic is not None
                    else None
                )
                if (
                    replayed_semantic is None
                    or replayed_semantic.get("trigger_action_id")
                    != pending.get("trigger_action_id")
                    or action.arguments.get("task_id") != replayed_semantic["task_id"]
                    or action.arguments.get("obligation_id") != replayed_semantic["obligation_id"]
                    or action.arguments.get("output_resource_ids")
                    != replayed_semantic["output_resource_ids"]
                    or action.arguments.get("output_requirement")
                    != replayed_semantic["output_requirement"]
                    or action.arguments.get("phase") != replayed_semantic["phase"]
                    or action_key != expected_key
                    or not isinstance(obligation, dict)
                    or obligation.get("resource_versions") != replayed_semantic["resource_versions"]
                    or obligation.get("phase") != replayed_semantic["phase"]
                ):
                    raise ValueError("dynamic TraceDP semantic evidence differs")
            expected_status = (
                "completed"
                if action.phase is ActionPhase.COMPLETED
                else "cancelled_with_unknown_effect"
                if action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
                else "failed"
            )
            if payload.get("status") != expected_status:
                raise ValueError("dynamic TraceDP outcome differs from runtime action")
            expected_remaining = remaining_budget - int(pending["cost_units"])
            if payload.get("remaining_budget") != expected_remaining:
                raise ValueError("dynamic TraceDP outcome remaining budget differs")
            version_after = action_versions[executed_action_id].get(str(resource_id))
            if payload.get("resource_version_after") != version_after:
                raise ValueError("dynamic TraceDP outcome resource version differs")
            if executed_action_id in executed_action_ids:
                raise ValueError("dynamic TraceDP runtime action has duplicate outcomes")
            executed_action_ids.add(executed_action_id)
            trigger_terminal_state[str(pending["trigger_action_id"])] = "action_terminal"
            remaining_budget = expected_remaining
            current_versions = action_versions[executed_action_id]
            current_action_order = action_order[executed_action_id]
            pending = None
        elif event_type == "decision_abandoned":
            if pending is None:
                raise ValueError("dynamic TraceDP abandoned decision has no planned decision")
            decision_id = pending["decision_id"]
            planned_action_id = pending["planned_action_id"]
            resource_id = pending["resource_id"]
            reserved_action_id = pending["executed_action_id"]
            if (
                payload.get("decision_id") != decision_id
                or payload.get("planned_action_id") != planned_action_id
                or payload.get("resource_id") != resource_id
                or payload.get("reserved_action_id") != reserved_action_id
            ):
                raise ValueError("dynamic TraceDP abandoned decision differs from its plan")
            cost_units = payload.get("cost_units")
            if (
                "executed_action_id" in payload
                or "executed_action" in payload
                or not isinstance(payload.get("reason"), str)
                or not payload["reason"]
                or not isinstance(cost_units, int)
                or isinstance(cost_units, bool)
                or cost_units != 0
                or payload.get("remaining_budget") != remaining_budget
                or payload.get("environment_execution_started") is not False
                or (isinstance(reserved_action_id, str) and reserved_action_id in kernel.actions)
            ):
                raise ValueError("dynamic TraceDP abandoned decision must remain pre-execution")
            # Abandonment closes the plan without creating or covering a Kernel action.
            if active_trigger is None:
                raise ValueError("dynamic TraceDP abandonment lacks an Actor trigger")
            trigger_terminal_state[active_trigger] = "abandoned"
            pending = None
        else:
            raise ValueError("dynamic TraceDP runtime decision event type is unsupported")
    if pending is not None:
        raise ValueError("dynamic TraceDP runtime has an unresolved decision")
    runtime_action_ids = {
        action_id for action_id, action in kernel.actions.items() if action.actor == "runtime"
    }
    if executed_action_ids != runtime_action_ids:
        raise ValueError("dynamic TraceDP decision log does not cover runtime actions")
    eligible_actor_ids = {
        action_id
        for action_id, action in kernel.actions.items()
        if action.actor == "actor"
        and action.phase in {ActionPhase.COMPLETED, ActionPhase.FAILED_AFTER_PARTIAL_EFFECT}
    }
    if disable_orders:
        disable_order = disable_orders[0]
        before_disable = {
            action_id for action_id in eligible_actor_ids if action_order[action_id] < disable_order
        }
        # The last completed Actor action may be the trigger whose supervisor
        # failed before it could durably write a decision.
        if before_disable:
            before_disable.remove(
                max(before_disable, key=lambda action_id: action_order[action_id])
            )
        expected_trigger_ids = before_disable
    else:
        expected_trigger_ids = eligible_actor_ids
    if not expected_trigger_ids <= planned_trigger_ids:
        raise ValueError("dynamic TraceDP decision log omits an Actor trigger")
    if any(
        action_id not in eligible_actor_ids
        or (disable_orders and action_order[action_id] > disable_orders[0])
        for action_id in planned_trigger_ids
    ):
        raise ValueError("dynamic TraceDP decision trigger is outside supervision")
    actor_interrupted = any(
        row.get("event_type") in {"actor_attempt_interrupted", "actor_session_budget_exhausted"}
        for row in _read_jsonl(path.parent / "actor-session-events.jsonl")
    )
    exempt_trigger_id: str | None = None
    if disable_orders:
        prior = [
            action_id
            for action_id in eligible_actor_ids
            if action_order[action_id] < disable_orders[0]
        ]
        if prior:
            exempt_trigger_id = max(prior, key=lambda item: action_order[item])
    for trigger_id in planned_trigger_ids:
        if trigger_id == exempt_trigger_id:
            continue
        terminal_state = trigger_terminal_state.get(trigger_id)
        if terminal_state in {"abstain", "action_terminal"}:
            continue
        if terminal_state == "abandoned" and actor_interrupted:
            continue
        raise ValueError("dynamic TraceDP trigger lacks a terminal stop decision")
    if replayed.remaining_budget_units != remaining_budget:
        raise ValueError("dynamic TraceDP replayed remaining budget differs")
    if policy_id in OUTPUT_PROGRESS_RUNTIME_METHODS:
        expected_output_triggers = _expected_output_supervision_triggers(
            event_log_path,
            kernel,
            budget_units=budget_units,
            policy_id=policy_id,
        )
        actual_output_triggers = tuple(
            (
                str(payload["trigger_action_id"]),
                kernel.actions[action_id].tool_name,
                str(kernel.actions[action_id].arguments.get("task_id")),
            )
            for action_id, payload in sorted(
                {**guidance_progress, **semantic_obligations}.items(),
                key=lambda item: action_order[item[0]],
            )
        )
        if actual_output_triggers != expected_output_triggers:
            raise ValueError("dynamic TraceDP output-supervision triggers differ")
    return remaining_budget


def _matches_legacy_evaluation_receipt(
    recorded: Mapping[str, object], current: Mapping[str, object]
) -> bool:
    """Validate the bounded v1 receipt shape before an explicit legacy upgrade."""

    expected = {
        "schema_version": "terminal_managed_runtime_evaluation_v1",
        "task_name": current["task_name"],
        "runtime_method": current["runtime_method"],
        "environment_session_id": current["environment_session_id"],
        "actor_action_count": current["actor_action_count"],
        "runtime_intervention_count": current["runtime_intervention_attempt_count"],
        "official_reward": current["official_reward"],
        "official_success": current["official_success"],
        "trial_exception": current["trial_exception"],
        "public_task_status": current["public_task_status"],
    }
    return recorded == expected


def _option_selector_for_replay(
    method: str,
    *,
    atomic_configuration: Mapping[str, object],
    recorded_checks: list[Mapping[str, object]],
    runtime_config: Mapping[str, object],
    agent_kwargs: Mapping[str, object],
) -> RuntimeOwnedOptionLiveResolver:
    """Rebuild the exact selector a Runtime-owned option method ran with.

    The guidance-phase planner replays only with its own resolver, its guided-completion
    forecasts and the horizon/residual settings it recorded; the generic option resolver
    would replay its models through the wrong Bellman tree.
    """

    shared = {
        "configuration": RuntimeFrozenAtomicEstimateConfiguration.from_dict(
            atomic_configuration
        ),
        "configured_checks": tuple(
            RuntimeCheckContract(
                **{**dict(item), "resource_ids": tuple(item.get("resource_ids", ()))}
            )
            for item in recorded_checks
        ),
    }
    if method == ADAPTIVE_INFORMATION_GAIN_RUNTIME_DEV_METHOD:
        return RuntimeInformationGainLiveResolver(**shared)
    if method == TRACEDP_RECOVERY_LOOKAHEAD_DEV_METHOD:
        return RuntimeRecoveryLookaheadLiveResolver(**shared)
    if method == TRACEDP_OBLIGATION_VALUE_DEV_METHOD:
        return RuntimeObligationValueLiveResolver(**shared)
    if method == TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD:
        guided = runtime_config.get("runtime_guided_completion_configuration")
        if not isinstance(guided, Mapping):
            raise ValueError("guidance phase result lacks its guided completion forecasts")
        return RuntimeGuidancePhaseLiveResolver(
            guided_configuration=GuidedCompletionConfiguration.from_dict(dict(guided)),
            planning_horizon_cap=agent_kwargs.get("runtime_planning_horizon_cap"),
            residual_structure=str(
                agent_kwargs.get("runtime_residual_structure", "task_linked")
            ),
            **shared,
        )
    return RuntimeOwnedOptionLiveResolver(**shared)


def finalize_terminal_managed_runtime_result(
    trial: Path,
    *,
    evaluation_validity: HarborCTRFEvaluationValidity,
    allow_legacy_projection: bool = False,
) -> dict[str, object]:
    """Append an official result only after independent validity and identity checks."""

    agent = trial / "agent"
    event_log_path = agent / "runtime-events.jsonl"
    kernel = PortableRuntimeKernel.load(event_log_path)
    runtime_config = _read_object(agent / "runtime-config.json")
    targeted_repair_audit = audit_runtime_targeted_repairs(
        repair_configuration=runtime_config.get("runtime_repair_configuration"),
        check_configuration=runtime_config.get("runtime_check_configuration"),
        event_log_path=event_log_path,
        joint_policy_path=agent / "runtime-decisions.jsonl",
        receipt_path=agent / "runtime-targeted-repair-executions.jsonl",
    )
    check_evidence_audit = audit_runtime_check_evidence(
        configuration_path=agent / "runtime-check-contracts.json",
        event_log_path=event_log_path,
        observer_path=agent / "terminal-public-state.jsonl",
        runtime_execution_path=agent / "runtime-check-executions.jsonl",
        execution_wrapper=_runtime_check_execution_wrapper(runtime_config),
        targeted_repair_bindings={
            row["runtime_action_id"]: row
            for row in (targeted_repair_audit or {}).get("check_bindings", [])
        },
    )
    if not kernel.actor_session_closed:
        raise ValueError("managed runtime actor session is not closed")
    recorded_final = _read_object(agent / "runtime-final-state.json")
    provider_request_audit = None
    provider_audit_schema = runtime_config.get("provider_request_audit_schema")
    if provider_audit_schema is not None:
        if provider_audit_schema != "terminal_provider_request_audit_v1":
            raise ValueError("provider request audit schema differs")
        check_configuration = runtime_config.get("runtime_check_configuration")
        if check_configuration is not None and not isinstance(check_configuration, dict):
            raise ValueError("runtime check configuration is invalid")
        provider_request_audit = audit_terminal_provider_requests(
            agent / "provider-request-audit.jsonl",
            registration_enabled=bool(
                (check_configuration or {}).get("actor_registration_enabled", False)
            ),
            registration_phase_required=bool(
                (check_configuration or {}).get(
                    "actor_registration_phase_required", False
                )
            ),
        )
    result = _read_object(trial / "result.json")
    trial_config = _read_object(trial / "config.json")
    task_name, method = _validate_trial_identity(
        trial=trial,
        trial_config=trial_config,
        runtime_config=runtime_config,
        result=result,
        kernel=kernel,
        check_evidence_audit=check_evidence_audit,
        allow_legacy_identity=(
            allow_legacy_projection
            and recorded_final.get("schema_version") != _PUBLIC_PROJECTION_V6
        ),
    )
    # Validate receipt-derived execution summaries only after the lower-level
    # Actor tool chain has established which terminal results are authentic.
    _validate_actor_final_state(
        recorded_final,
        kernel,
        trial=trial,
        allow_legacy_projection=allow_legacy_projection,
    )
    guidance_followup_audit = None
    if check_evidence_audit is not None and (
        check_evidence_audit.get("guidance_bindings")
        or check_evidence_audit.get("check_guidance_bindings")
    ):
        guidance_followup_audit = audit_runtime_predicate_guidance_followup(
            agent_dir=agent,
            kernel=kernel,
            actor_runtime=_nested_mapping(
                runtime_config, "actor_runtime", label="Actor runtime",
            ),
            check_evidence_audit=check_evidence_audit,
        )
    agent_result = _nested_mapping(result, "agent_result", label="Harbor agent result")
    metadata = _nested_mapping(agent_result, "metadata", label="Harbor agent metadata")
    audited_phase = (
        check_evidence_audit.get("registration_phase")
        if check_evidence_audit is not None else None
    )
    audited_phase_status = (
        audited_phase.get("status") if isinstance(audited_phase, dict) else None
    )
    if metadata.get("actor_check_registration_phase_status") != audited_phase_status:
        raise ValueError("managed Actor check registration phase metadata differs")
    if result.get("exception_info"):
        raise ValueError("managed runtime Harbor result contains an exception")
    if not evaluation_validity.is_valid:
        raise ValueError(
            "official evaluation is not scientifically valid: "
            f"{evaluation_validity.invalid_reason or 'validity evidence failed'}"
        )
    observed_validity = load_harbor_ctrf_evaluation_validity(trial)
    if evaluation_validity != observed_validity:
        raise ValueError("evaluation validity input differs from this trial's CTRF")
    if method in _POLICY_METHODS:
        budget_units = runtime_config.get("runtime_intervention_budget")
        if not isinstance(budget_units, int) or isinstance(budget_units, bool) or budget_units < 1:
            raise ValueError("dynamic TraceDP runtime budget is invalid")
        remaining_budget = _validate_dynamic_decisions(
            agent / "runtime-decisions.jsonl",
            kernel,
            event_log_path=event_log_path,
            budget_units=budget_units,
            policy_id=method,
        )
        if metadata.get("runtime_intervention_budget_remaining") != remaining_budget:
            raise ValueError("dynamic TraceDP result remaining budget differs")
    elif method in _JOINT_METHODS:
        budget_units = runtime_config.get("runtime_intervention_budget")
        source_arm_index = runtime_config.get("joint_source_arm_index")
        if not isinstance(budget_units, int) or isinstance(budget_units, bool) or budget_units < 1:
            raise ValueError("runtime joint policy budget is invalid")
        atomic_configuration = runtime_config.get(
            "runtime_atomic_estimate_configuration"
        )
        check_configuration = runtime_config.get("runtime_check_configuration")
        recorded_checks = (
            check_configuration.get("contracts", [])
            if isinstance(check_configuration, Mapping)
            else []
        )
        if not isinstance(recorded_checks, list) or any(
            not isinstance(item, Mapping) for item in recorded_checks
        ):
            raise ValueError("runtime check configuration is invalid")
        decision_resolver = (
            _option_selector_for_replay(
                method,
                atomic_configuration=atomic_configuration,
                recorded_checks=recorded_checks,
                runtime_config=runtime_config,
                agent_kwargs=trial_config.get("agent", {}).get("kwargs", {}),
            )
            if method in RUNTIME_OWNED_OPTION_DP_DEV_METHODS
            and isinstance(atomic_configuration, Mapping)
            else RuntimeEventTraceDPV8Resolver(lambda _: None)
            if method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
            else RuntimeBoundedTraceDPV7Resolver(lambda _: None)
            if method == BOUNDED_TRACEDP_DEV_V7_METHOD
            else RuntimeBoundedTraceDPV6Resolver(lambda _: None)
            if method == BOUNDED_TRACEDP_DEV_V6_METHOD
            else RuntimeBoundedTraceDPV4Resolver(lambda _: None)
            if method == BOUNDED_TRACEDP_DEV_V4_METHOD
            else RuntimeBoundedTraceDPResolver(lambda _: None)
            if method in BOUNDED_TRACEDP_DEV_METHODS
            else RuntimeResidualJITResolver(
                method_id=method,
                jit_memory=FrozenRuntimeJITMemory.load(
                    agent / "runtime-jit-memory.json"
                ),
                test_task_id=str(runtime_config["jit_test_task_id"]),
            )
            if method == TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD
            else RuntimeSettlementTraceDPResolver(
                method_id=method,
                jit_memory=(
                    FrozenRuntimeJITMemory.load(agent / "runtime-jit-memory.json")
                    if method == TRACEDP_JIT_SETTLEMENT_METHOD
                    else None
                ),
                test_task_id=(
                    str(runtime_config["jit_test_task_id"])
                    if method == TRACEDP_JIT_SETTLEMENT_METHOD
                    else None
                ),
            )
            if method in _PLANNED_METHODS
            else None
        )
        remaining_budget = _validate_joint_decisions(
            agent / "runtime-decisions.jsonl",
            kernel,
            event_log_path=event_log_path,
            budget_units=budget_units,
            method_id=method,
            source_arm_index=(source_arm_index if isinstance(source_arm_index, int) else None),
            decision_resolver=decision_resolver,
            predicate_guidance_evidence={
                row["runtime_action_id"]: row
                for row in (check_evidence_audit or {}).get("guidance_bindings", [])
            },
            registered_check_guidance_evidence={
                row["runtime_action_id"]: row
                for row in (check_evidence_audit or {}).get(
                    "check_guidance_bindings", []
                )
            },
            runtime_check_execution_evidence={
                row["runtime_action_id"]: row
                for row in (check_evidence_audit or {}).get(
                    "runtime_execution_bindings", []
                )
            },
            targeted_repair_execution_evidence={
                row["runtime_action_id"]: row
                for row in (targeted_repair_audit or {}).get("check_bindings", [])
            },
            actor_time_budget_ms=(
                int(
                    _nested_mapping(runtime_config, "actor_runtime", label="Actor runtime")[
                        "host_pi_timeout_seconds"
                    ]
                )
                * 1000
                if method in _ACTOR_RESERVE_METHODS
                else None
            ),
        )
        if metadata.get("runtime_intervention_budget_remaining") != remaining_budget:
            raise ValueError("runtime joint result remaining budget differs")

    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or not math.isfinite(float(reward))
    ):
        raise ValueError("managed runtime trial lacks an official numeric reward")
    score = float(reward)
    reward_path = trial / "verifier/reward.txt"
    try:
        reward_text = reward_path.read_text().strip()
        reward_file_score = float(reward_text)
    except (OSError, ValueError):
        raise ValueError("managed runtime verifier reward artifact is invalid") from None
    if not math.isfinite(reward_file_score) or reward_file_score != score:
        raise ValueError("managed runtime verifier reward differs from Harbor result")
    success = score == 1.0
    public_task_ids = [
        task_id
        for task_id, node in kernel.task_nodes.items()
        if node.public_requirement and node.parent_id is None
    ]
    if len(public_task_ids) != 1:
        raise ValueError("managed runtime requires exactly one public task root")

    runtime_actions = [action for action in kernel.actions.values() if action.actor == "runtime"]
    failed_phases = {
        ActionPhase.BLOCKED,
        ActionPhase.FAILED_WITHOUT_EFFECT,
        ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
        ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
    }
    terminal_phases = failed_phases | {
        ActionPhase.COMPLETED,
        ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT,
    }
    if any(action.phase not in terminal_phases for action in runtime_actions):
        raise ValueError("managed runtime has an unfinished intervention")

    selected_actions = (
        _planned_action_by_runtime_execution(agent / "runtime-decisions.jsonl")
        if method in _BUDGET_METHODS
        else {}
    )
    joint_decision_ids = (
        _joint_decision_by_runtime_execution(agent / "runtime-decisions.jsonl")
        if method in _JOINT_METHODS
        else {}
    )
    joint_trigger_ids = (
        _joint_trigger_by_runtime_execution(agent / "runtime-decisions.jsonl")
        if method in _JOINT_METHODS
        else {}
    )
    interrupted_attempts: frozenset[int] = frozenset()
    call_attempt_by_id: dict[str, int] = {}
    pi_execution_ids: frozenset[str] = frozenset()
    terminal_uncompleted_ids: frozenset[str] = frozenset()
    if method in OUTPUT_PROGRESS_RUNTIME_METHODS or method in FAILURE_DIAGNOSIS_JOINT_METHODS:
        (
            interrupted_attempts,
            call_attempt_by_id,
            pi_execution_ids,
            terminal_uncompleted_ids,
        ) = _pi_interruption_delivery_context(trial, runtime_config)
    if method in {_DYNAMIC_METHOD_V2, _DYNAMIC_METHOD_V3}:
        progress_actions = {
            action_id: action
            for action_id, action in kernel.actions.items()
            if action.actor == "runtime" and action.tool_name == "runtime_guidance"
        }
        progress_receipts = _intervention_delivery_receipts(
            agent / "terminal-public-state.jsonl",
            intervention_kind="runtime_guidance",
        )
        if set(progress_receipts) != set(progress_actions):
            raise ValueError("runtime progress Pi deliveries differ from Kernel actions")
        progress_replay = _replay_runtime_guidance_progress(event_log_path, kernel)
        for action_id, action in progress_actions.items():
            replayed_progress = progress_replay.get(action_id)
            delivery_receipt = progress_receipts[action_id]
            actor_action_id, intervention, delivered_content = delivery_receipt
            expected_intervention = {
                "action_id": action_id,
                "decision_action_id": selected_actions.get(action_id),
                "resource_id": action.arguments.get("resource_id"),
                "policy": method,
                "intervention_kind": "runtime_guidance",
                "status": "completed",
                "task_id": action.arguments.get("task_id"),
                "output_resource_ids": action.arguments.get("output_resource_ids"),
                "output_requirement": action.arguments.get("output_requirement"),
                "no_progress_actor_actions": action.arguments.get("no_progress_actor_actions"),
            }
            if (
                action.phase is not ActionPhase.COMPLETED
                or replayed_progress is None
                or actor_action_id != replayed_progress.get("trigger_action_id")
                or intervention != expected_intervention
                or not delivered_content.endswith(
                    _expected_progress_notice(
                        output_resource_ids=list(action.arguments.get("output_resource_ids", ())),
                        output_requirement=action.arguments.get("output_requirement"),
                        action_count=action.arguments.get("no_progress_actor_actions"),
                    )
                )
            ):
                raise ValueError("runtime progress Pi delivery content differs")

    effect_ledger: RuntimeInterventionEffectLedger | None = None
    pending_effect_action_ids: list[str] = []
    projected_effects: list[dict[str, object]] = []
    if method in EFFECT_LEDGER_RUNTIME_METHODS:
        effect_path = agent / "runtime-intervention-effects.jsonl"
        effect_ledger = RuntimeInterventionEffectLedger(effect_path, load_existing=True)
        tracked_actions = {
            action_id: action
            for action_id, action in kernel.actions.items()
            if action.actor == "runtime"
            and (
                action.tool_name
                in {
                    "runtime_semantic_guidance",
                    "runtime_convergence_guidance",
                    "runtime_failure_diagnosis_guidance",
                }
                or (
                    method in {_DYNAMIC_METHOD_V4, _FIXED_METHOD, *_JOINT_METHODS}
                    and action.tool_name == "runtime_guidance"
                )
            )
        }
        records = {record.action_id: record for record in effect_ledger.records}
        if set(records) != set(tracked_actions):
            raise ValueError("runtime effects do not cover tracked interventions")
        semantic_replay = _replay_runtime_semantic_obligations(event_log_path, kernel)
        semantic_delivery_receipts = _intervention_delivery_receipts(
            agent / "terminal-public-state.jsonl",
            intervention_kind="runtime_semantic_guidance",
        )
        progress_delivery_receipts = (
            _intervention_delivery_receipts(
                agent / "terminal-public-state.jsonl",
                intervention_kind="runtime_guidance",
            )
            if method in {_DYNAMIC_METHOD_V4, _FIXED_METHOD, *_JOINT_METHODS}
            else {}
        )
        convergence_delivery_receipts = (
            _intervention_delivery_receipts(
                agent / "terminal-public-state.jsonl",
                intervention_kind="runtime_convergence_guidance",
            )
            if method == FIXED_JOINT_SLATE_V3_METHOD
            else {}
        )
        failure_diagnosis_delivery_receipts = (
            _intervention_delivery_receipts(
                agent / "terminal-public-state.jsonl",
                intervention_kind="runtime_failure_diagnosis_guidance",
            )
            if method in FAILURE_DIAGNOSIS_JOINT_METHODS
            else {}
        )
        delivery_receipts = {
            **semantic_delivery_receipts,
            **progress_delivery_receipts,
            **convergence_delivery_receipts,
            **failure_diagnosis_delivery_receipts,
        }
        expected_gateway_receipt_ids = {
            action_id
            for action_id, record in records.items()
            if record.delivery_status is RuntimeInterventionDeliveryStatus.DELIVERED
            or record.delivery_correction_trigger_action_id is not None
        }
        settlement_delivery_ids = (
            {
                action_id
                for action_id in records
                if action_id.startswith("runtime-joint-settlement-review-")
            }
            if method in _SETTLEMENT_REVIEW_METHODS
            else set()
        )
        if not expected_gateway_receipt_ids <= set(delivery_receipts).union(
            settlement_delivery_ids
        ) or not set(delivery_receipts) <= set(records):
            raise ValueError("runtime Pi deliveries differ from effect ledger")
        progress_replay = _replay_runtime_guidance_progress(event_log_path, kernel)
        failure_diagnosis_replay = _replay_runtime_failure_diagnoses(event_log_path, kernel)
        action_versions, action_order = _kernel_action_version_snapshots(event_log_path)
        changed_by_action = _kernel_action_changed_resources(event_log_path)
        terminal_effect = (
            RuntimeInterventionEffectStatus.OFFICIAL_SUCCEEDED_AFTER_DELIVERY
            if success
            else RuntimeInterventionEffectStatus.OFFICIAL_FAILED_AFTER_DELIVERY
        )
        for action_id in sorted(records):
            record = records[action_id]
            action = tracked_actions[action_id]
            key = action.arguments.get("obligation_version_key")
            recorded_key = [
                record.obligation_version_key[0],
                record.obligation_version_key[1],
                [list(item) for item in record.obligation_version_key[2]],
                record.obligation_version_key[3],
            ]
            if action.arguments.get("obligation_id") != record.obligation_id or key != recorded_key:
                raise ValueError("runtime intervention effect identity differs from Kernel")
            is_semantic = action.tool_name == "runtime_semantic_guidance"
            is_convergence = action.tool_name == "runtime_convergence_guidance"
            is_failure_diagnosis = action.tool_name == "runtime_failure_diagnosis_guidance"
            replayed = (
                failure_diagnosis_replay.get(action_id)
                if is_failure_diagnosis
                else semantic_replay.get(action_id)
                if is_semantic or is_convergence
                else progress_replay.get(action_id)
            )
            delivery_receipt = delivery_receipts.get(action_id)
            if action.phase is not ActionPhase.COMPLETED or replayed is None:
                raise ValueError("runtime delivery lacks completed gateway evidence")
            expected_intervention = {
                "action_id": action_id,
                "decision_action_id": selected_actions.get(action_id),
                "resource_id": action.arguments.get("resource_id"),
                "policy": method,
                "intervention_kind": action.tool_name,
                "status": "completed",
                "task_id": action.arguments.get("task_id"),
                "obligation_id": record.obligation_id,
                "obligation_version_key": key,
                "output_resource_ids": action.arguments.get("output_resource_ids"),
                "output_requirement": action.arguments.get("output_requirement"),
                "affected_task_ids": action.arguments.get("affected_task_ids"),
            }
            if method in _JOINT_METHODS:
                expected_intervention["joint_decision_id"] = joint_decision_ids.get(action_id)
                expected_intervention["phase"] = action.arguments.get("phase")
            if is_semantic:
                expected_notice = _expected_semantic_notice(
                    list(action.arguments.get("output_resource_ids", ())),
                    policy_id=method,
                )
            elif is_failure_diagnosis:
                expected_intervention.update(
                    {
                        "failure_action_id": action.arguments.get("failure_action_id"),
                        "failure_count_since_resource_change": (
                            action.arguments.get("failure_count_since_resource_change")
                        ),
                        "failure_resource_versions": action.arguments.get(
                            "failure_resource_versions"
                        ),
                        "failure_candidate_task_ids": action.arguments.get(
                            "failure_candidate_task_ids"
                        ),
                        "failure_association_evidence_id": action.arguments.get(
                            "failure_association_evidence_id"
                        ),
                    }
                )
                if method in TRACEDP_RESIDUAL_METHODS:
                    failure_kernel = _kernel_at_failure_boundary(
                        event_log_path,
                        failure_action_id=action.arguments.get("failure_action_id"),
                    )
                    expected_context = _expected_tracedp_failure_context(
                        failure_kernel,
                        failure_action_id=action.arguments.get("failure_action_id"),
                        candidate_task_ids=action.arguments.get("failure_candidate_task_ids"),
                    )
                    if action.arguments.get("failure_diagnosis_context") != expected_context:
                        raise ValueError("TraceDP failure diagnosis context differs")
                    expected_intervention["failure_diagnosis_context"] = expected_context
                    expected_notice = _expected_tracedp_failure_notice(expected_context)
                else:
                    expected_notice = _expected_failure_diagnosis_notice(
                        action.arguments.get("failure_count_since_resource_change"),
                        minimum_failure_count=(
                            1
                            if method in EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS
                            else ACTOR_FAILURE_DIAGNOSIS_THRESHOLD
                        ),
                    )
            elif is_convergence:
                expected_intervention.update(
                    {
                        "actor_time_budget_ms": action.arguments.get("actor_time_budget_ms"),
                        "actor_time_remaining_ms": action.arguments.get("actor_time_remaining_ms"),
                        "convergence_reserve_ms": action.arguments.get("convergence_reserve_ms"),
                    }
                )
                expected_notice = _expected_convergence_notice(
                    list(action.arguments.get("output_resource_ids", ()))
                )
            else:
                expected_intervention.update(
                    {
                        "phase": action.arguments.get("phase"),
                        "no_progress_actor_actions": action.arguments.get(
                            "no_progress_actor_actions"
                        ),
                    }
                )
                expected_notice = _expected_progress_notice(
                    output_resource_ids=list(action.arguments.get("output_resource_ids", ())),
                    output_requirement=action.arguments.get("output_requirement"),
                    action_count=action.arguments.get("no_progress_actor_actions"),
                )
            if delivery_receipt is not None:
                actor_action_id, intervention, delivered_content = delivery_receipt
                expected_delivery_trigger = (
                    joint_trigger_ids.get(action_id)
                    if method in _JOINT_METHODS
                    else replayed.get("trigger_action_id")
                )
                if (
                    actor_action_id != expected_delivery_trigger
                    or intervention != expected_intervention
                    or not _ends_with_runtime_notice_batch_member(
                        delivered_content, expected_notice
                    )
                ):
                    raise ValueError("runtime Pi delivery content differs")
            if record.delivery_status is RuntimeInterventionDeliveryStatus.DELIVERED:
                trigger_action_id = replayed.get("trigger_action_id")
                settlement_delivery = action_id in settlement_delivery_ids
                if settlement_delivery:
                    trigger = kernel.actions.get(str(trigger_action_id))
                    valid_delivery = (
                        trigger is not None
                        and trigger.actor == "actor_bridge"
                        and trigger.tool_name == "actor_natural_settlement_trigger"
                    )
                else:
                    valid_delivery = (
                        delivery_receipt is not None
                        and isinstance(trigger_action_id, str)
                        and trigger_action_id in pi_execution_ids
                    )
                if not valid_delivery:
                    raise ValueError("runtime delivery lacks completed Pi evidence")
            elif record.delivery_status is RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN:
                _validate_cancelled_delivery_boundary(
                    record=record,
                    replayed=replayed,
                    delivery_receipt=delivery_receipt,
                    interrupted_attempts=interrupted_attempts,
                    call_attempt_by_id=call_attempt_by_id,
                    pi_execution_ids=pi_execution_ids,
                    terminal_uncompleted_ids=terminal_uncompleted_ids,
                )
            else:
                raise ValueError("runtime delivery status is unsupported")
            projected_status = record.effect_status
            projected_evidence = record.effect_evidence_id
            if record.effect_status is RuntimeInterventionEffectStatus.PENDING:
                if record.delivery_status is not RuntimeInterventionDeliveryStatus.DELIVERED:
                    raise ValueError("pending runtime effect lacks delivery")
                pending_effect_action_ids.append(action_id)
                projected_status = terminal_effect
                projected_evidence = "official-evaluation:" + task_name
            elif record.effect_status is RuntimeInterventionEffectStatus.OUTPUT_CHANGED:
                evidence_action_id = record.effect_evidence_id
                evidence_action = (
                    kernel.actions.get(evidence_action_id)
                    if isinstance(evidence_action_id, str)
                    else None
                )
                output_ids = set(action.arguments.get("output_resource_ids", ()))
                if (
                    evidence_action is None
                    or evidence_action.actor != "actor"
                    or evidence_action_id not in action_order
                    or action_id not in action_order
                    or action_order[evidence_action_id] <= action_order[action_id]
                    or not changed_by_action.get(evidence_action_id, frozenset()).intersection(
                        output_ids
                    )
                    or all(
                        action_versions[evidence_action_id].get(resource_id)
                        == dict(record.obligation_version_key[2]).get(resource_id)
                        for resource_id in output_ids
                    )
                ):
                    raise ValueError("runtime output-change evidence is invalid")
            elif record.effect_status in {
                RuntimeInterventionEffectStatus.OFFICIAL_SUCCEEDED_AFTER_DELIVERY,
                RuntimeInterventionEffectStatus.OFFICIAL_FAILED_AFTER_DELIVERY,
            }:
                if (
                    record.effect_status is not terminal_effect
                    or record.effect_evidence_id != "official-evaluation:" + task_name
                ):
                    raise ValueError("runtime official effect differs")
            elif not (
                record.effect_status is RuntimeInterventionEffectStatus.UNKNOWN
                and record.delivery_status is RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN
                and record.effect_evidence_id is None
            ):
                raise ValueError("runtime effect status is unsupported")
            projected_effects.append(
                {
                    "action_id": action_id,
                    "obligation_id": record.obligation_id,
                    "delivery_status": record.delivery_status.value,
                    "effect_status": projected_status.value,
                    "effect_evidence_id": projected_evidence,
                }
            )

    # Finish all validation before the first event-log or receipt mutation.
    if kernel.official_evaluation is not None and kernel.official_evaluation != {
        "score": score,
        "succeeded": success,
    }:
        raise ValueError("official evaluation differs from the runtime event log")
    official_task_status = "completed" if success else "failed"
    receipt: dict[str, object] = {
        "schema_version": "terminal_managed_runtime_evaluation_v2",
        "task_name": task_name,
        "runtime_method": method,
        "environment_session_id": kernel.session_id,
        "actor_projection_schema": recorded_final["schema_version"],
        "legacy_actor_projection_accepted": recorded_final["schema_version"]
        in {
            _PUBLIC_PROJECTION_V1,
            _PUBLIC_PROJECTION_V2,
            _PUBLIC_PROJECTION_V3,
            _PUBLIC_PROJECTION_V4,
            _PUBLIC_PROJECTION_V5,
        },
        "evaluation_validity": evaluation_validity.as_receipt(),
        "actor_action_count": sum(action.actor == "actor" for action in kernel.actions.values()),
        "runtime_intervention_attempt_count": len(runtime_actions),
        "runtime_intervention_completed_count": sum(
            action.phase is ActionPhase.COMPLETED for action in runtime_actions
        ),
        "runtime_intervention_failed_count": sum(
            action.phase in failed_phases for action in runtime_actions
        ),
        "runtime_intervention_cancelled_count": sum(
            action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT for action in runtime_actions
        ),
        "official_reward": score,
        "official_success": success,
        "trial_exception": result.get("exception_info"),
        "public_task_status": official_task_status,
        "token_accounting_complete": metadata.get("token_accounting_complete"),
        "token_accounting_semantics": metadata.get("token_accounting_semantics"),
        "scientific_result": False,
        "formal_experiment_authorized": False,
        "admissible_claim_scope": "runtime_lifecycle_development",
    }
    if method in _DYNAMIC_METHODS:
        # Q-values and probability estimates are policy-authored evidence, not an oracle.
        receipt["dynamic_diagnostics_evidence"] = {
            "chosen_action_binding": "verified_against_planned_decision",
            "candidate_resource_binding": "verified_against_planned_decision",
            "probabilities_and_q_values": "self_reported_unverified",
        }
    elif method == _FIXED_METHOD:
        receipt["fixed_policy_evidence"] = {
            "decision_rule": "fixed_first_legal_candidate_v1",
            "candidate_order": "lexicographic_action_id",
            "learned_scores_or_bellman_values_used": False,
        }
    elif method in _JOINT_METHODS:
        decision_payloads = [
            row["payload"]
            for row in _read_jsonl(agent / "runtime-decisions.jsonl")
            if row.get("event_type") == "decision_selected"
            and isinstance(row.get("payload"), dict)
        ]
        joint_decision_evidence = _summarize_joint_decision_evidence(
            decision_payloads
        )
        joint_policy_evidence = {
            "schema_version": "joint_policy_execution_evidence_v2",
            "configured_policy_identity": {"runtime_method": method},
            "observed_decision_evidence": joint_decision_evidence,
            "one_selection_per_actor_trigger": True,
            "one_or_zero_runtime_actions_per_selection": True,
            "implementation_status": "implementation_unverified",
        }
        receipt["joint_policy_evidence"] = joint_policy_evidence
        if method in JIT_JOINT_METHODS:
            jit_decision_count = int(
                joint_decision_evidence["jit_payload_decision_count"]
            )
            admitted_jit_count = int(
                joint_decision_evidence["admitted_jit_decision_count"]
            )
            receipt["jit_policy_evidence"] = {
                "decision_count": jit_decision_count,
                "admitted_decision_count": admitted_jit_count,
                "fallback_count": jit_decision_count - admitted_jit_count,
                "changed_action_count": joint_decision_evidence[
                    "jit_changed_action_decision_count"
                ],
                "memory_artifact": "runtime-jit-memory.json",
                "memory_frozen_during_actor": True,
            }
    if method in EFFECT_LEDGER_RUNTIME_METHODS:
        receipt["runtime_intervention_effects"] = projected_effects
    if check_evidence_audit is not None:
        receipt["runtime_check_evidence"] = check_evidence_audit
    if targeted_repair_audit is not None:
        receipt["runtime_targeted_repair_evidence"] = {
            key: value
            for key, value in targeted_repair_audit.items()
            if key != "check_bindings"
        }
    if kernel.acceptance_predicate_enabled:
        # Preserve the actor-close residual state beside the authoritative Harbor result.
        receipt["acceptance_predicate_evidence"] = {
            "schema_version": "runtime_acceptance_predicate_evaluation_v1",
            "implementation_status": "implementation_unverified",
            "run_class": runtime_config.get("run_class"),
            "public_semantic_review": runtime_config.get(
                "runtime_acceptance_predicate_semantic_review"
            ),
            "acceptance_predicates_at_actor_close": recorded_final.get(
                "acceptance_predicates"
            ),
            "predicate_incidents_at_actor_close": recorded_final.get(
                "predicate_incidents", []
            ),
            "task_acceptance_summary_at_actor_close": recorded_final.get(
                "task_acceptance_summary"
            ),
            "official_evaluation_at_actor_close": recorded_final.get(
                "official_evaluation"
            ),
            "official_result": {"reward": score, "succeeded": success},
            "official_result_does_not_backfill_predicates": True,
        }
    if provider_request_audit is not None:
        receipt["provider_request_audit"] = provider_request_audit
    if guidance_followup_audit is not None:
        receipt["runtime_predicate_guidance_followup"] = guidance_followup_audit
    receipt_path = agent / "runtime-evaluation-receipt.json"
    upgrade_legacy_receipt = False
    if receipt_path.exists():
        recorded_receipt = _read_object(receipt_path)
        if recorded_receipt != receipt:
            upgrade_legacy_receipt = allow_legacy_projection and _matches_legacy_evaluation_receipt(
                recorded_receipt, receipt
            )
            if not upgrade_legacy_receipt:
                raise ValueError("managed runtime evaluation receipt differs")
    if kernel.official_evaluation is None:
        kernel.record_official_evaluation(score=score, succeeded=success)
    if effect_ledger is not None:
        terminal_effect = (
            RuntimeInterventionEffectStatus.OFFICIAL_SUCCEEDED_AFTER_DELIVERY
            if success
            else RuntimeInterventionEffectStatus.OFFICIAL_FAILED_AFTER_DELIVERY
        )
        for action_id in pending_effect_action_ids:
            effect_ledger.record_effect(
                action_id=action_id,
                status=terminal_effect,
                evidence_id="official-evaluation:" + task_name,
            )
    evaluated_projection = kernel.public_projection()
    evaluated_statuses = dict(evaluated_projection["task_statuses"])
    evaluated_statuses[public_task_ids[0]] = official_task_status
    evaluated_projection["task_statuses"] = evaluated_statuses
    evaluated_nodes = [dict(node) for node in evaluated_projection["task_nodes"]]
    for node in evaluated_nodes:
        if node.get("task_id") == public_task_ids[0]:
            node["status"] = official_task_status
    evaluated_projection["task_nodes"] = evaluated_nodes
    _write_object(agent / "runtime-evaluated-state.json", evaluated_projection)
    if not receipt_path.exists() or upgrade_legacy_receipt:
        _write_object(receipt_path, receipt)
    return receipt


__all__ = [
    "HarborCTRFEvaluationValidity",
    "finalize_terminal_managed_runtime_result",
    "load_harbor_ctrf_evaluation_validity",
]
