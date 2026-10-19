"""Held-out controlled experiment for same-checkpoint postmortems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.code_provenance import git_provenance
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.models.base import ModelBase
from lh_platform.tracing.adapters import TraceInvestigationError
from lh_platform.tracing.self_trace import (
    SELF_TRACE_CLAIM_SCOPE,
    SelfTraceExperimentRunner,
    SelfTraceReceiptV1,
    SelfTraceRuntimeBindingV1,
)
from lh_platform.tracing.self_trace_admission import admit_self_trace_run
from lh_platform.tracing.self_trace_controlled import (
    CONTROLLED_SELF_TRACE_DATASET_ID,
    ControlledSelfTraceCell,
)
from lh_platform.tracing.self_trace_datasets import self_trace_dataset
from lh_platform.tracing.self_trace_methods import SelfTraceMethodV1, self_trace_method

SELF_TRACE_P1_RESULT_SCHEMA = "lh_controlled_self_trace_p1/v3"


def _sha256(value: object) -> str:
    return "sha256:" + canonical_sha256(value)


def _self_trace_p1_policy(
    dataset_id: str,
    trace_method_id: str,
    sampling_seed: int = 0,
    run_admission_sha256: str | None = None,
) -> dict[str, Any]:
    dataset = self_trace_dataset(dataset_id)
    method = self_trace_method(trace_method_id)
    return {
        "schema": "lh_controlled_self_trace_p1_policy/v3",
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset.sha256,
        "trace_method_id": method.trace_method_id,
        "trace_method_capabilities": list(method.capabilities),
        "trace_method_evidence_ceiling": method.evidence_ceiling.value,
        "sampling_seed": sampling_seed,
        "run_admission_sha256": run_admission_sha256,
        "required_case_count": 6,
        "required_fault_type_count": 6,
        "maximum_invalid_output_count": 0,
        "minimum_exact_localization_rate": 0.5,
        "minimum_intervention_verified_root_cause_rate": 0.5,
        "required_self_trace_binding_rate": 1.0,
        "one_actor_and_one_fresh_diagnostician_instance_per_case": True,
        "claim_scope": SELF_TRACE_CLAIM_SCOPE,
    }


SELF_TRACE_P1_POLICY = _self_trace_p1_policy(
    CONTROLLED_SELF_TRACE_DATASET_ID,
    "counterfactual_replay",
)
SELF_TRACE_P1_POLICY_SHA256 = _sha256(SELF_TRACE_P1_POLICY)


@dataclass(frozen=True)
class SelfTraceCaseEvaluationV1:
    case_id: str
    fault_type: str
    exact_localization: bool
    component_correct: bool
    cause_type_correct: bool
    intervention_verified_root_cause: bool
    self_trace_binding_verified: bool
    diagnosis_valid: bool
    badcase_labels: tuple[str, ...]
    receipt: SelfTraceReceiptV1 | None
    diagnostic_output: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fault_type": self.fault_type,
            "exact_localization": self.exact_localization,
            "component_correct": self.component_correct,
            "cause_type_correct": self.cause_type_correct,
            "intervention_verified_root_cause": (self.intervention_verified_root_cause),
            "self_trace_binding_verified": self.self_trace_binding_verified,
            "diagnosis_valid": self.diagnosis_valid,
            "badcase_labels": list(self.badcase_labels),
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "diagnostic_output": self.diagnostic_output,
            "error": self.error,
        }


@dataclass(frozen=True)
class SelfTraceP1Result:
    model_id: str
    system_cell_id: str
    system_cell_sha256: str
    checkpoint_tree_sha256: str
    dataset_id: str
    dataset_sha256: str
    trace_method_id: str
    trace_method_capabilities: tuple[str, ...]
    sampling_seed: int
    run_admission_sha256: str | None
    cases: tuple[SelfTraceCaseEvaluationV1, ...]
    exact_localization_rate: float
    component_accuracy: float
    cause_type_accuracy: float
    intervention_verified_root_cause_rate: float
    self_trace_binding_rate: float
    invalid_output_count: int
    badcase_counts: tuple[tuple[str, int], ...]
    effect_gate_passed: bool
    blocked_by: tuple[str, ...]
    code_provenance: dict[str, Any]
    artifact_sha256: str
    policy_sha256: str
    case_count: int = 6
    claim_scope: str = SELF_TRACE_CLAIM_SCOPE
    schema: str = SELF_TRACE_P1_RESULT_SCHEMA

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "model_id": self.model_id,
            "system_cell_id": self.system_cell_id,
            "system_cell_sha256": self.system_cell_sha256,
            "checkpoint_tree_sha256": self.checkpoint_tree_sha256,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "trace_method_id": self.trace_method_id,
            "trace_method_capabilities": list(self.trace_method_capabilities),
            "sampling_seed": self.sampling_seed,
            "run_admission_sha256": self.run_admission_sha256,
            "case_count": self.case_count,
            "cases": [case.to_dict() for case in self.cases],
            "exact_localization_rate": self.exact_localization_rate,
            "component_accuracy": self.component_accuracy,
            "cause_type_accuracy": self.cause_type_accuracy,
            "intervention_verified_root_cause_rate": (self.intervention_verified_root_cause_rate),
            "self_trace_binding_rate": self.self_trace_binding_rate,
            "invalid_output_count": self.invalid_output_count,
            "badcase_counts": dict(self.badcase_counts),
            "effect_gate_passed": self.effect_gate_passed,
            "blocked_by": list(self.blocked_by),
            "claim_scope": self.claim_scope,
            "policy_sha256": self.policy_sha256,
            "code_provenance": self.code_provenance,
        }
        if include_sha256:
            payload["artifact_sha256"] = self.artifact_sha256
        return payload


def _badcase_labels(
    cell: ControlledSelfTraceCell,
    receipt: SelfTraceReceiptV1 | None,
    method: SelfTraceMethodV1,
) -> tuple[str, ...]:
    if receipt is None:
        return ("EXECUTION_RECEIPT_INVALID", "SELF_TRACE_BINDING_INVALID")
    labels: list[str] = []
    if not receipt.self_trace_binding_verified:
        labels.append("SELF_TRACE_BINDING_INVALID")
    if not receipt.diagnosis_valid or receipt.investigation is None:
        labels.append("TRACER_OUTPUT_INVALID")
        return tuple(sorted(set(labels)))
    claim = receipt.investigation.claim
    if set(claim.causal_event_ids) != set(cell.label.causal_event_ids):
        labels.append("LOCALIZATION_WRONG")
        if receipt.persisted_trace.interruption_event_id in claim.causal_event_ids:
            labels.append("LAST_STEP_BIAS")
    if claim.component != cell.label.component:
        labels.append("COMPONENT_WRONG")
    if claim.cause_type != cell.label.fault_type:
        labels.append("CAUSE_TYPE_WRONG")
    if not method.supports_counterfactual_replay:
        labels.append("L4_EVIDENCE_UNAVAILABLE")
    elif not receipt.investigation.root_cause_verified:
        labels.append("CAUSE_RIGHT_REPAIR_FAIL")
    return tuple(sorted(set(labels)))


def run_controlled_self_trace_p1(
    *,
    actor_model_factory: Callable[[], ModelBase],
    diagnostician_model_factory: Callable[[], ModelBase],
    runtime_binding: SelfTraceRuntimeBindingV1,
    dataset_id: str = CONTROLLED_SELF_TRACE_DATASET_ID,
    trace_method_id: str = "counterfactual_replay",
) -> SelfTraceP1Result:
    """Run six held-out controlled interruptions with fresh same-checkpoint instances."""

    admission = admit_self_trace_run(
        system_cell=runtime_binding.system_cell,
        model_id=runtime_binding.expected_served_model_id,
        dataset_id=dataset_id,
        trace_method_id=trace_method_id,
        sampling_seed=runtime_binding.sampling_seed,
        expected_run_admission_sha256=runtime_binding.run_admission_sha256,
    )
    dataset = admission.dataset
    method = admission.method
    cells = dataset.cells
    policy = _self_trace_p1_policy(
        dataset.dataset_id,
        method.trace_method_id,
        runtime_binding.sampling_seed,
        admission.run_admission_sha256,
    )
    evaluations: list[SelfTraceCaseEvaluationV1] = []
    for cell in cells:
        receipt: SelfTraceReceiptV1 | None = None
        error: str | None = None
        try:
            receipt = SelfTraceExperimentRunner(
                actor_model=actor_model_factory(),
                diagnostician_model=diagnostician_model_factory(),
                runtime_binding=runtime_binding,
                replay_controller_id=dataset.replay_controller_id,
                trace_method_id=method.trace_method_id,
            ).run(cell)
        except (TraceInvestigationError, ValueError) as caught:
            error = str(caught)
        investigation = receipt.investigation if receipt else None
        exact = bool(
            investigation
            and set(investigation.claim.causal_event_ids) == set(cell.label.causal_event_ids)
        )
        component_correct = bool(
            investigation and investigation.claim.component == cell.label.component
        )
        cause_type_correct = bool(
            investigation and investigation.claim.cause_type == cell.label.fault_type
        )
        evaluations.append(
            SelfTraceCaseEvaluationV1(
                case_id=cell.case_id,
                fault_type=cell.label.fault_type.value,
                exact_localization=exact,
                component_correct=component_correct,
                cause_type_correct=cause_type_correct,
                intervention_verified_root_cause=bool(
                    investigation and investigation.root_cause_verified
                ),
                self_trace_binding_verified=bool(receipt and receipt.self_trace_binding_verified),
                diagnosis_valid=bool(receipt and receipt.diagnosis_valid),
                badcase_labels=_badcase_labels(cell, receipt, method),
                receipt=receipt,
                diagnostic_output=receipt.diagnostic_output if receipt else None,
                error=receipt.diagnosis_error if receipt else error,
            )
        )

    denominator = len(evaluations)
    exact_rate = sum(row.exact_localization for row in evaluations) / denominator
    component_accuracy = sum(row.component_correct for row in evaluations) / denominator
    cause_type_accuracy = sum(row.cause_type_correct for row in evaluations) / denominator
    verified_rate = sum(row.intervention_verified_root_cause for row in evaluations) / denominator
    binding_rate = sum(row.self_trace_binding_verified for row in evaluations) / denominator
    invalid_count = sum(not row.diagnosis_valid for row in evaluations)
    badcase_counts: dict[str, int] = {}
    for evaluation in evaluations:
        for label in evaluation.badcase_labels:
            badcase_counts[label] = badcase_counts.get(label, 0) + 1

    blocked_by: list[str] = []
    if denominator != int(policy["required_case_count"]):
        blocked_by.append("case_coverage_incomplete")
    if len({cell.label.fault_type for cell in cells}) != int(
        policy["required_fault_type_count"]
    ):
        blocked_by.append("fault_type_coverage_incomplete")
    if invalid_count > int(policy["maximum_invalid_output_count"]):
        blocked_by.append("tracer_output_invalid")
    if exact_rate < float(policy["minimum_exact_localization_rate"]):
        blocked_by.append("exact_localization_below_0_5")
    if verified_rate < float(policy["minimum_intervention_verified_root_cause_rate"]):
        blocked_by.append("verified_root_cause_rate_below_0_5")
    if binding_rate != float(policy["required_self_trace_binding_rate"]):
        blocked_by.append("self_trace_binding_incomplete")
    if not method.supports_counterfactual_replay:
        blocked_by.append("trace_method_lacks_counterfactual_replay")

    unhashed = SelfTraceP1Result(
        model_id=runtime_binding.expected_served_model_id,
        system_cell_id=runtime_binding.system_cell.system_cell_id,
        system_cell_sha256=runtime_binding.system_cell.sha256,
        checkpoint_tree_sha256=runtime_binding.checkpoint_tree_sha256,
        dataset_id=dataset.dataset_id,
        dataset_sha256=dataset.sha256,
        trace_method_id=method.trace_method_id,
        trace_method_capabilities=method.capabilities,
        sampling_seed=runtime_binding.sampling_seed,
        run_admission_sha256=admission.run_admission_sha256,
        cases=tuple(evaluations),
        exact_localization_rate=exact_rate,
        component_accuracy=component_accuracy,
        cause_type_accuracy=cause_type_accuracy,
        intervention_verified_root_cause_rate=verified_rate,
        self_trace_binding_rate=binding_rate,
        invalid_output_count=invalid_count,
        badcase_counts=tuple(sorted(badcase_counts.items())),
        effect_gate_passed=not blocked_by,
        blocked_by=tuple(blocked_by),
        code_provenance=git_provenance(Path(__file__)),
        artifact_sha256="",
        policy_sha256=_sha256(policy),
        case_count=denominator,
    )
    return SelfTraceP1Result(
        model_id=unhashed.model_id,
        system_cell_id=unhashed.system_cell_id,
        system_cell_sha256=unhashed.system_cell_sha256,
        checkpoint_tree_sha256=unhashed.checkpoint_tree_sha256,
        dataset_id=unhashed.dataset_id,
        dataset_sha256=unhashed.dataset_sha256,
        trace_method_id=unhashed.trace_method_id,
        trace_method_capabilities=unhashed.trace_method_capabilities,
        sampling_seed=unhashed.sampling_seed,
        run_admission_sha256=unhashed.run_admission_sha256,
        cases=unhashed.cases,
        exact_localization_rate=unhashed.exact_localization_rate,
        component_accuracy=unhashed.component_accuracy,
        cause_type_accuracy=unhashed.cause_type_accuracy,
        intervention_verified_root_cause_rate=(unhashed.intervention_verified_root_cause_rate),
        self_trace_binding_rate=unhashed.self_trace_binding_rate,
        invalid_output_count=unhashed.invalid_output_count,
        badcase_counts=unhashed.badcase_counts,
        effect_gate_passed=unhashed.effect_gate_passed,
        blocked_by=unhashed.blocked_by,
        code_provenance=unhashed.code_provenance,
        artifact_sha256=_sha256(unhashed.to_dict(include_sha256=False)),
        policy_sha256=unhashed.policy_sha256,
        case_count=unhashed.case_count,
    )


def write_controlled_self_trace_p1(
    *,
    actor_model_factory: Callable[[], ModelBase],
    diagnostician_model_factory: Callable[[], ModelBase],
    runtime_binding: SelfTraceRuntimeBindingV1,
    output_path: str | Path,
    dataset_id: str = CONTROLLED_SELF_TRACE_DATASET_ID,
    trace_method_id: str = "counterfactual_replay",
) -> SelfTraceP1Result:
    """Run P1 and atomically publish one immutable result artifact."""

    result = run_controlled_self_trace_p1(
        actor_model_factory=actor_model_factory,
        diagnostician_model_factory=diagnostician_model_factory,
        runtime_binding=runtime_binding,
        dataset_id=dataset_id,
        trace_method_id=trace_method_id,
    )
    atomic_write_json(output_path, result.to_dict(), overwrite=False)
    return result
