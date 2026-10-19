"""Controlled P0 experiment and badcase aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.code_provenance import git_provenance
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.models.base import ModelBase
from lh_platform.tracing.adapters import (
    AllAtOnceTracer,
    CounterfactualReplayTracer,
    TraceInvestigationError,
)
from lh_platform.tracing.controlled import (
    CONTROLLED_TRACEBACK_DATASET_ID,
    ControlledTracebackCell,
    controlled_traceback_cells,
)
from lh_platform.tracing.controlled_replay import ControlledReplayController
from lh_platform.tracing.runner import TraceInvestigationRunner
from lh_platform.tracing.types import (
    InvestigationReceipt,
)

TRACEBACK_P0_RESULT_SCHEMA = "lh_controlled_traceback_p0/v3"
TRACEBACK_P0_CLAIM_SCOPE = "controlled_executable_replay_mechanism_only_not_self_generated"
TRACEBACK_P0_TRACERS = ("all_at_once", "counterfactual_replay")


def _sha256(value: object) -> str:
    return "sha256:" + canonical_sha256(value)


TRACEBACK_P0_POLICY = {
    "schema": "lh_controlled_traceback_p0_policy/v3",
    "dataset_id": CONTROLLED_TRACEBACK_DATASET_ID,
    "fault_types": [
        "history_memory_deletion",
        "observation_staleness",
        "orchestrator_routing_error",
        "premature_termination",
        "tool_argument_corruption",
        "tool_response_corruption",
    ],
    "tracers": list(TRACEBACK_P0_TRACERS),
    "replay_contract": {
        "controller_id": ControlledReplayController.controller_id,
        "arms": ["fault_baseline", "predicted_repair", "matched_sham"],
        "exact_restore_required": True,
        "runner_owned_replay": True,
        "sham_must_differ_from_baseline": True,
        "replay_calls_per_case": 3,
    },
    "effect_gate": {
        "required_case_count": 6,
        "maximum_invalid_output_count": 0,
        "minimum_exact_localization_rate": 0.5,
        "minimum_intervention_verified_root_cause_rate": 0.5,
        "require_single_served_model_identity": True,
    },
    "claim_scope": TRACEBACK_P0_CLAIM_SCOPE,
}
TRACEBACK_P0_POLICY_SHA256 = _sha256(TRACEBACK_P0_POLICY)


@dataclass(frozen=True)
class TracebackCaseEvaluationV1:
    case_id: str
    fault_type: str
    tracer_id: str
    exact_localization: bool
    component_correct: bool
    cause_type_correct: bool
    intervention_verified_root_cause: bool
    badcase_labels: tuple[str, ...]
    model_calls: int
    receipt: InvestigationReceipt | None
    diagnostic_output: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fault_type": self.fault_type,
            "tracer_id": self.tracer_id,
            "exact_localization": self.exact_localization,
            "component_correct": self.component_correct,
            "cause_type_correct": self.cause_type_correct,
            "intervention_verified_root_cause": self.intervention_verified_root_cause,
            "badcase_labels": list(self.badcase_labels),
            "model_calls": self.model_calls,
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "diagnostic_output": self.diagnostic_output,
            "error": self.error,
        }


@dataclass(frozen=True)
class TracebackMethodMetricsV1:
    tracer_id: str
    case_count: int
    exact_localization_rate: float
    component_accuracy: float
    cause_type_accuracy: float
    intervention_verified_root_cause_rate: float
    invalid_output_count: int
    model_calls: int
    replay_calls: int
    badcase_counts: tuple[tuple[str, int], ...]
    cases: tuple[TracebackCaseEvaluationV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracer_id": self.tracer_id,
            "case_count": self.case_count,
            "exact_localization_rate": self.exact_localization_rate,
            "component_accuracy": self.component_accuracy,
            "cause_type_accuracy": self.cause_type_accuracy,
            "intervention_verified_root_cause_rate": (self.intervention_verified_root_cause_rate),
            "invalid_output_count": self.invalid_output_count,
            "model_calls": self.model_calls,
            "replay_calls": self.replay_calls,
            "badcase_counts": dict(self.badcase_counts),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class TracebackP0Result:
    model_id: str
    model_runtime_evidence: dict[str, Any]
    code_provenance: dict[str, Any]
    methods: tuple[TracebackMethodMetricsV1, ...]
    effect_gate_passed: bool
    blocked_by: tuple[str, ...]
    artifact_sha256: str
    dataset_id: str = CONTROLLED_TRACEBACK_DATASET_ID
    case_count: int = 6
    claim_scope: str = TRACEBACK_P0_CLAIM_SCOPE
    policy_sha256: str = TRACEBACK_P0_POLICY_SHA256
    schema: str = TRACEBACK_P0_RESULT_SCHEMA

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "model_id": self.model_id,
            "model_runtime_evidence": self.model_runtime_evidence,
            "code_provenance": self.code_provenance,
            "dataset_id": self.dataset_id,
            "case_count": self.case_count,
            "methods": [method.to_dict() for method in self.methods],
            "effect_gate_passed": self.effect_gate_passed,
            "blocked_by": list(self.blocked_by),
            "claim_scope": self.claim_scope,
            "policy_sha256": self.policy_sha256,
        }
        if include_sha256:
            payload["artifact_sha256"] = self.artifact_sha256
        return payload


def _badcase_labels(
    cell: ControlledTracebackCell,
    tracer_id: str,
    receipt: InvestigationReceipt | None,
    *,
    exact_localization: bool,
    cause_type_correct: bool,
) -> tuple[str, ...]:
    if receipt is None:
        return ("TRACER_OUTPUT_INVALID",)
    labels: list[str] = []
    if not exact_localization:
        labels.append("LOCALIZATION_WRONG")
        if cell.case.interruption_event_id in receipt.claim.causal_event_ids:
            labels.append("LAST_STEP_BIAS")
    elif not cause_type_correct:
        labels.append("CAUSE_TYPE_WRONG")
    elif tracer_id == "all_at_once":
        labels.append("LOCALIZATION_RIGHT_CAUSAL_TEST_NOT_RUN")
    elif not receipt.root_cause_verified:
        labels.append("CAUSE_RIGHT_REPAIR_FAIL")
    if (
        cell.label.component in {"tool", "orchestrator", "environment_observation"}
        and receipt.claim.component == "actor"
    ):
        labels.append("TOOL_FAILURE_MISATTRIBUTED_TO_MODEL")
    return tuple(sorted(set(labels)))


def _evaluate_method(
    *,
    model: ModelBase,
    tracer_id: str,
    cells: tuple[ControlledTracebackCell, ...],
) -> TracebackMethodMetricsV1:
    evaluations: list[TracebackCaseEvaluationV1] = []
    for cell in cells:
        if tracer_id == "all_at_once":
            tracer = AllAtOnceTracer(model=model)
        elif tracer_id == "counterfactual_replay":
            tracer = CounterfactualReplayTracer(model=model)
        else:
            raise ValueError(f"unsupported P0 tracer: {tracer_id}")

        receipt: InvestigationReceipt | None = None
        diagnostic_output: str | None = None
        error: str | None = None
        generation_cursor = model.generation_cursor()
        try:
            replay_controller = (
                ControlledReplayController(cell.case.case_id)
                if tracer_id == "counterfactual_replay"
                else None
            )
            receipt = TraceInvestigationRunner(
                tracer,
                replay_controller=replay_controller,
            ).run(cell.case)
        except (TraceInvestigationError, ValueError) as caught:
            error = str(caught)
            diagnostic_output = getattr(caught, "diagnostic_output", None)
        physical_model_calls = len(model.generation_events_since(generation_cursor))

        exact_localization = bool(
            receipt and set(receipt.claim.causal_event_ids) == set(cell.label.causal_event_ids)
        )
        component_correct = bool(receipt and receipt.claim.component == cell.label.component)
        cause_type_correct = bool(receipt and receipt.claim.cause_type == cell.label.cause_type)
        intervention_verified = bool(receipt and receipt.root_cause_verified)
        evaluations.append(
            TracebackCaseEvaluationV1(
                case_id=cell.case.case_id,
                fault_type=cell.label.fault_type,
                tracer_id=tracer_id,
                exact_localization=exact_localization,
                component_correct=component_correct,
                cause_type_correct=cause_type_correct,
                intervention_verified_root_cause=intervention_verified,
                badcase_labels=_badcase_labels(
                    cell,
                    tracer_id,
                    receipt,
                    exact_localization=exact_localization,
                    cause_type_correct=cause_type_correct,
                ),
                model_calls=physical_model_calls,
                receipt=receipt,
                diagnostic_output=diagnostic_output,
                error=error,
            )
        )

    denominator = len(evaluations)
    badcase_counts: dict[str, int] = {}
    for evaluation in evaluations:
        for label in evaluation.badcase_labels:
            badcase_counts[label] = badcase_counts.get(label, 0) + 1
    return TracebackMethodMetricsV1(
        tracer_id=tracer_id,
        case_count=denominator,
        exact_localization_rate=sum(row.exact_localization for row in evaluations) / denominator,
        component_accuracy=sum(row.component_correct for row in evaluations) / denominator,
        cause_type_accuracy=sum(row.cause_type_correct for row in evaluations) / denominator,
        intervention_verified_root_cause_rate=sum(
            row.intervention_verified_root_cause for row in evaluations
        )
        / denominator,
        invalid_output_count=sum(row.receipt is None for row in evaluations),
        model_calls=sum(row.model_calls for row in evaluations),
        replay_calls=sum(row.receipt.usage.replay_calls for row in evaluations if row.receipt),
        badcase_counts=tuple(sorted(badcase_counts.items())),
        cases=tuple(evaluations),
    )


def run_controlled_traceback_p0(
    *,
    model: ModelBase,
    tracer_ids: tuple[str, ...] = TRACEBACK_P0_TRACERS,
) -> TracebackP0Result:
    """Run paired localization and causal-replay arms on six hidden-label faults."""

    if set(tracer_ids) != set(TRACEBACK_P0_TRACERS) or len(tracer_ids) != len(TRACEBACK_P0_TRACERS):
        raise ValueError("controlled P0 requires exactly the paired tracer arms")
    cells = controlled_traceback_cells()
    methods = tuple(
        _evaluate_method(model=model, tracer_id=tracer_id, cells=cells) for tracer_id in tracer_ids
    )
    by_tracer = {method.tracer_id: method for method in methods}
    replay = by_tracer["counterfactual_replay"]
    runtime_evidence = model.runtime_evidence()
    blocked_by: list[str] = []
    for method in methods:
        if method.case_count != len(cells):
            blocked_by.append(f"{method.tracer_id}_case_coverage_incomplete")
        if method.invalid_output_count:
            blocked_by.append(f"{method.tracer_id}_output_invalid")
    if replay.exact_localization_rate < 0.5:
        blocked_by.append("exact_localization_below_0_5")
    if replay.intervention_verified_root_cause_rate < 0.5:
        blocked_by.append("verified_root_cause_rate_below_0_5")
    expected_replay_calls = len(cells) * 3
    if replay.replay_calls != expected_replay_calls:
        blocked_by.append("counterfactual_replay_call_count_mismatch")
    expected_model_calls = len(cells) * len(methods)
    if sum(method.model_calls for method in methods) != expected_model_calls:
        blocked_by.append("physical_model_call_count_mismatch")
    if runtime_evidence.get("dummy_model_used") is True:
        blocked_by.append("dummy_model_used")
    if runtime_evidence.get("dummy_fallback_used") is True:
        blocked_by.append("dummy_fallback_used")
    if int(runtime_evidence.get("remote_error_count") or 0) > 0:
        blocked_by.append("remote_model_errors_observed")
    remote_calls = runtime_evidence.get("remote_call_count")
    if remote_calls is not None and int(remote_calls) != expected_model_calls:
        blocked_by.append("remote_model_call_count_mismatch")
    if runtime_evidence.get("backend") == "openai_compatible":
        served_model_ids = runtime_evidence.get("served_model_ids") or []
        if len(served_model_ids) != 1:
            blocked_by.append("served_model_identity_not_unique")

    model_id = str(getattr(model, "model_name", None) or model.name)
    unhashed = TracebackP0Result(
        model_id=model_id,
        model_runtime_evidence=runtime_evidence,
        code_provenance=git_provenance(Path(__file__)),
        methods=methods,
        effect_gate_passed=not blocked_by,
        blocked_by=tuple(blocked_by),
        artifact_sha256="",
        case_count=len(cells),
    )
    return TracebackP0Result(
        model_id=unhashed.model_id,
        model_runtime_evidence=unhashed.model_runtime_evidence,
        code_provenance=unhashed.code_provenance,
        methods=unhashed.methods,
        effect_gate_passed=unhashed.effect_gate_passed,
        blocked_by=unhashed.blocked_by,
        artifact_sha256=_sha256(unhashed.to_dict(include_sha256=False)),
        case_count=unhashed.case_count,
    )


def write_controlled_traceback_p0(
    *,
    model: ModelBase,
    output_path: str | Path,
    tracer_ids: tuple[str, ...] = TRACEBACK_P0_TRACERS,
) -> TracebackP0Result:
    """Run the controlled P0 and publish one immutable JSON artifact."""

    result = run_controlled_traceback_p0(model=model, tracer_ids=tracer_ids)
    atomic_write_json(output_path, result.to_dict(), overwrite=False)
    return result
