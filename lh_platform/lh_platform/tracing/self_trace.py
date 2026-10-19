"""Same-checkpoint controlled postmortem seam."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.core.system_cell import SystemCellManifestV1
from lh_platform.models.base import (
    Message,
    ModelBase,
    generation_event_sha256,
    sha256_text,
)
from lh_platform.tracing.adapters import TraceInvestigationError
from lh_platform.tracing.registry import TRACERS
from lh_platform.tracing.runner import TraceInvestigationRunner
from lh_platform.tracing.self_trace_controlled import (
    ControlledSelfTraceCell,
    SelfTraceReplayController,
)
from lh_platform.tracing.types import InvestigationReceipt, TraceCaseV1

SELF_TRACE_RECEIPT_SCHEMA = "lh_controlled_self_trace_receipt/v2"
SELF_TRACE_BINDING_RECEIPT_SCHEMA = "lh_controlled_self_trace_binding/v1"
SELF_TRACE_CLAIM_SCOPE = "controlled_same_checkpoint_postmortem_not_natural_model_error"
_RAW_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PREFIXED_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha256(value: object) -> str:
    return "sha256:" + canonical_sha256(value)


@dataclass(frozen=True)
class SelfTraceRuntimeBindingV1:
    """Bind both model instances to one frozen policy payload and served identity."""

    system_cell: SystemCellManifestV1
    checkpoint_tree_sha256: str
    expected_served_model_id: str
    sampling_seed: int = 0
    run_admission_sha256: str | None = None

    def __post_init__(self) -> None:
        if not _RAW_SHA256_RE.fullmatch(self.checkpoint_tree_sha256):
            raise ValueError("checkpoint tree SHA-256 must be a raw lowercase digest")
        if not self.expected_served_model_id.strip():
            raise ValueError("expected served model id must be non-empty")
        if (
            isinstance(self.sampling_seed, bool)
            or not isinstance(self.sampling_seed, int)
            or self.sampling_seed < 0
        ):
            raise ValueError("self-trace sampling seed must be a non-negative integer")
        if self.run_admission_sha256 is not None and not _PREFIXED_SHA256_RE.fullmatch(
            self.run_admission_sha256
        ):
            raise ValueError("self-trace run-admission SHA-256 is invalid")
        if not self.system_cell.frozen:
            raise ValueError("self-trace requires a frozen system cell")
        if self.system_cell.adaptive_components_allowed or self.system_cell.optimizer_present:
            raise ValueError("self-trace forbids adaptive components and optimizers")
        if self.checkpoint_tree_sha256 != self.system_cell.policy_identity.policy_checkpoint_sha256:
            raise ValueError("checkpoint tree does not match system-cell policy identity")


@dataclass(frozen=True)
class SelfTraceReceiptV1:
    case_id: str
    system_cell_id: str
    system_cell_sha256: str
    checkpoint_tree_sha256: str
    expected_served_model_id: str
    actor_generation_id: str
    actor_generation_event_sha256: str
    actor_instance_id: str
    diagnostician_generation_id: str
    diagnostician_generation_event_sha256: str
    diagnostician_instance_id: str
    actor_runtime_evidence: dict[str, Any]
    diagnostician_runtime_evidence: dict[str, Any]
    persisted_trace: TraceCaseV1
    persisted_trace_sha256: str
    investigation: InvestigationReceipt | None
    self_trace_binding_verified: bool
    diagnosis_valid: bool
    diagnosis_error: str | None
    diagnostic_output: str | None
    binding_receipt_sha256: str
    receipt_sha256: str
    claim_scope: str = SELF_TRACE_CLAIM_SCOPE
    schema: str = SELF_TRACE_RECEIPT_SCHEMA

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "case_id": self.case_id,
            "claim_scope": self.claim_scope,
            "system_cell_id": self.system_cell_id,
            "system_cell_sha256": self.system_cell_sha256,
            "checkpoint_tree_sha256": self.checkpoint_tree_sha256,
            "expected_served_model_id": self.expected_served_model_id,
            "actor_generation_id": self.actor_generation_id,
            "actor_generation_event_sha256": self.actor_generation_event_sha256,
            "actor_instance_id": self.actor_instance_id,
            "diagnostician_generation_id": self.diagnostician_generation_id,
            "diagnostician_generation_event_sha256": (self.diagnostician_generation_event_sha256),
            "diagnostician_instance_id": self.diagnostician_instance_id,
            "actor_runtime_evidence": self.actor_runtime_evidence,
            "diagnostician_runtime_evidence": self.diagnostician_runtime_evidence,
            "persisted_trace": self.persisted_trace.to_dict(),
            "persisted_trace_sha256": self.persisted_trace_sha256,
            "investigation": self.investigation.to_dict() if self.investigation else None,
            "self_trace_binding_verified": self.self_trace_binding_verified,
            "diagnosis_valid": self.diagnosis_valid,
            "diagnosis_error": self.diagnosis_error,
            "diagnostic_output": self.diagnostic_output,
            "binding_receipt_sha256": self.binding_receipt_sha256,
        }
        if include_sha256:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload

    def binding_payload(self) -> dict[str, Any]:
        """Return the execution/identity layer independent of diagnosis semantics."""

        return {
            "schema": SELF_TRACE_BINDING_RECEIPT_SCHEMA,
            "case_id": self.case_id,
            "system_cell_id": self.system_cell_id,
            "system_cell_sha256": self.system_cell_sha256,
            "checkpoint_tree_sha256": self.checkpoint_tree_sha256,
            "expected_served_model_id": self.expected_served_model_id,
            "actor_generation_id": self.actor_generation_id,
            "actor_generation_event_sha256": self.actor_generation_event_sha256,
            "actor_instance_id": self.actor_instance_id,
            "diagnostician_generation_id": self.diagnostician_generation_id,
            "diagnostician_generation_event_sha256": (
                self.diagnostician_generation_event_sha256
            ),
            "diagnostician_instance_id": self.diagnostician_instance_id,
            "actor_runtime_evidence": self.actor_runtime_evidence,
            "diagnostician_runtime_evidence": self.diagnostician_runtime_evidence,
            "persisted_trace_sha256": self.persisted_trace_sha256,
            "self_trace_binding_verified": self.self_trace_binding_verified,
        }


def _one_generation_event(model: ModelBase, cursor: int, output_text: str) -> dict[str, Any]:
    events = model.generation_events_since(cursor)
    if len(events) != 1:
        raise TraceInvestigationError("self-trace phase must emit exactly one model generation")
    event = events[0]
    if event.get("response_sha256") != sha256_text(output_text):
        raise TraceInvestigationError("generation response hash does not bind persisted output")
    if event.get("event_sha256") != generation_event_sha256(event):
        raise TraceInvestigationError("generation event integrity check failed")
    generation_id = event.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise TraceInvestigationError("generation id is missing")
    return event


def _instance_id(generation_id: str) -> str:
    parts = generation_id.rsplit(":", 2)
    if len(parts) != 3 or not parts[1]:
        raise TraceInvestigationError("generation id does not contain a model instance id")
    return parts[1]


def _validate_runtime(
    evidence: dict[str, Any],
    *,
    runtime_binding: SelfTraceRuntimeBindingV1,
) -> None:
    if evidence.get("dummy_model_used") is True or evidence.get("dummy_fallback_used") is True:
        raise TraceInvestigationError("self-trace forbids dummy model execution")
    if evidence.get("model_instantiated") is not True:
        raise TraceInvestigationError("self-trace requires an instantiated model runtime")
    if int(evidence.get("generation_error_count") or 0) != 0:
        raise TraceInvestigationError("self-trace observed a model generation error")
    if int(evidence.get("generation_count") or 0) != 1:
        raise TraceInvestigationError("each self-trace model session must make one generation")
    if evidence.get("model_identity_ids") != [runtime_binding.expected_served_model_id]:
        raise TraceInvestigationError("model identity does not match runtime binding")
    if evidence.get("policy_checkpoint_sha256") != runtime_binding.checkpoint_tree_sha256:
        raise TraceInvestigationError("checkpoint identity attestation does not match binding")
    if evidence.get("sampling_seed") != runtime_binding.sampling_seed:
        raise TraceInvestigationError("runtime sampling seed does not match binding")

    identity = runtime_binding.system_cell.policy_identity
    runtime = runtime_binding.system_cell.runtime.config
    expected_attestations = {
        "model_io_profile_sha256": identity.model_io_profile_sha256,
        "chat_template_sha256": runtime.get("chat_template_sha256"),
        "backend_component_sha256": runtime.get("backend_component_sha256"),
        "runtime_policy_identity_sha256": runtime.get("runtime_policy_identity_sha256"),
    }
    for key, expected in expected_attestations.items():
        if not isinstance(expected, str) or not expected:
            raise TraceInvestigationError(f"System Cell lacks required {key} attestation")
        if evidence.get(key) != expected:
            raise TraceInvestigationError(f"runtime {key} attestation does not match binding")
    if evidence.get("inference_only") is not True:
        raise TraceInvestigationError("self-trace runtime is not attested inference-only")
    if evidence.get("optimizer_present") is not False:
        raise TraceInvestigationError("self-trace runtime optimizer attestation is not false")


def _persist_round_trip(case: TraceCaseV1) -> TraceCaseV1:
    encoded = json.dumps(
        case.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    persisted = TraceCaseV1.from_dict(json.loads(encoded))
    if persisted.sha256 != case.sha256:
        raise TraceInvestigationError("persisted trace round-trip changed its identity")
    return persisted


class SelfTraceExperimentRunner:
    """Generate, persist, restart diagnosis, and causally verify one controlled failure."""

    def __init__(
        self,
        *,
        actor_model: ModelBase,
        diagnostician_model: ModelBase,
        runtime_binding: SelfTraceRuntimeBindingV1,
        replay_controller_id: str = "controlled_self_trace_executable_replay/v2",
        trace_method_id: str = "counterfactual_replay",
    ) -> None:
        if actor_model is diagnostician_model:
            raise ValueError("actor and diagnostician must be separate model instances")
        if runtime_binding.system_cell.action_objective.config.get("trace_method_id") != (
            trace_method_id
        ):
            raise ValueError("System Cell does not bind the requested self-trace trace method")
        self._actor_model = actor_model
        self._diagnostician_model = diagnostician_model
        self._binding = runtime_binding
        self._replay_controller_id = replay_controller_id
        self._trace_method_id = trace_method_id

    def run(self, cell: ControlledSelfTraceCell) -> SelfTraceReceiptV1:
        system_prompt, user_prompt = cell.template.actor_messages()
        actor_cursor = self._actor_model.generation_cursor()
        actor_output = self._actor_model.generate(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            temperature=self._binding.system_cell.sampling_policy.temperature,
            max_tokens=self._binding.system_cell.sampling_policy.max_response_tokens,
        )
        actor_event = _one_generation_event(self._actor_model, actor_cursor, actor_output.text)
        actor_generation_id = str(actor_event["generation_id"])
        actor_instance_id = _instance_id(actor_generation_id)
        materialized = cell.template.materialize(
            actor_output=actor_output.text,
            actor_metadata={
                "generation_id": actor_generation_id,
                "generation_event_sha256": str(actor_event["event_sha256"]),
                "policy_checkpoint_sha256": self._binding.checkpoint_tree_sha256,
                "system_cell_sha256": self._binding.system_cell.sha256,
            },
        )
        persisted = _persist_round_trip(materialized)

        diagnostician_cursor = self._diagnostician_model.generation_cursor()
        investigation: InvestigationReceipt | None = None
        diagnosis_error: str | None = None
        diagnostic_output: str | None = None
        try:
            investigation = TraceInvestigationRunner(
                TRACERS.create(self._trace_method_id, model=self._diagnostician_model),
                replay_controller=SelfTraceReplayController(
                    cell,
                    persisted,
                    controller_id=self._replay_controller_id,
                ),
            ).run(persisted)
        except TraceInvestigationError as caught:
            diagnosis_error = str(caught)
            diagnostic_output = caught.diagnostic_output
        diagnostic_events = self._diagnostician_model.generation_events_since(diagnostician_cursor)
        if len(diagnostic_events) != 1:
            raise TraceInvestigationError(
                "self-trace diagnosis must emit exactly one model generation"
            )
        diagnostic_event = diagnostic_events[0]
        if diagnostic_event.get("event_sha256") != generation_event_sha256(diagnostic_event):
            raise TraceInvestigationError("diagnostician generation event integrity check failed")
        if diagnostic_output is not None and diagnostic_event.get("response_sha256") != sha256_text(
            diagnostic_output
        ):
            raise TraceInvestigationError("diagnostician output does not match generation event")
        diagnostician_generation_id = str(diagnostic_event.get("generation_id") or "")
        diagnostician_instance_id = _instance_id(diagnostician_generation_id)
        if actor_instance_id == diagnostician_instance_id:
            raise TraceInvestigationError("diagnostician did not restart in a fresh model instance")

        actor_runtime = self._actor_model.runtime_evidence()
        diagnostician_runtime = self._diagnostician_model.runtime_evidence()
        _validate_runtime(
            actor_runtime,
            runtime_binding=self._binding,
        )
        _validate_runtime(
            diagnostician_runtime,
            runtime_binding=self._binding,
        )
        binding_verified = (
            self._binding.checkpoint_tree_sha256
            == self._binding.system_cell.policy_identity.policy_checkpoint_sha256
            and actor_instance_id != diagnostician_instance_id
        )
        unhashed = SelfTraceReceiptV1(
            case_id=cell.case_id,
            system_cell_id=self._binding.system_cell.system_cell_id,
            system_cell_sha256=self._binding.system_cell.sha256,
            checkpoint_tree_sha256=self._binding.checkpoint_tree_sha256,
            expected_served_model_id=self._binding.expected_served_model_id,
            actor_generation_id=actor_generation_id,
            actor_generation_event_sha256=str(actor_event["event_sha256"]),
            actor_instance_id=actor_instance_id,
            diagnostician_generation_id=diagnostician_generation_id,
            diagnostician_generation_event_sha256=str(diagnostic_event["event_sha256"]),
            diagnostician_instance_id=diagnostician_instance_id,
            actor_runtime_evidence=actor_runtime,
            diagnostician_runtime_evidence=diagnostician_runtime,
            persisted_trace=persisted,
            persisted_trace_sha256=persisted.sha256,
            investigation=investigation,
            self_trace_binding_verified=binding_verified,
            diagnosis_valid=investigation is not None,
            diagnosis_error=diagnosis_error,
            diagnostic_output=diagnostic_output,
            binding_receipt_sha256="",
            receipt_sha256="",
        )
        binding_receipt_sha256 = _sha256(unhashed.binding_payload())
        bound = SelfTraceReceiptV1(
            case_id=unhashed.case_id,
            system_cell_id=unhashed.system_cell_id,
            system_cell_sha256=unhashed.system_cell_sha256,
            checkpoint_tree_sha256=unhashed.checkpoint_tree_sha256,
            expected_served_model_id=unhashed.expected_served_model_id,
            actor_generation_id=unhashed.actor_generation_id,
            actor_generation_event_sha256=unhashed.actor_generation_event_sha256,
            actor_instance_id=unhashed.actor_instance_id,
            diagnostician_generation_id=unhashed.diagnostician_generation_id,
            diagnostician_generation_event_sha256=(unhashed.diagnostician_generation_event_sha256),
            diagnostician_instance_id=unhashed.diagnostician_instance_id,
            actor_runtime_evidence=unhashed.actor_runtime_evidence,
            diagnostician_runtime_evidence=unhashed.diagnostician_runtime_evidence,
            persisted_trace=unhashed.persisted_trace,
            persisted_trace_sha256=unhashed.persisted_trace_sha256,
            investigation=unhashed.investigation,
            self_trace_binding_verified=unhashed.self_trace_binding_verified,
            diagnosis_valid=unhashed.diagnosis_valid,
            diagnosis_error=unhashed.diagnosis_error,
            diagnostic_output=unhashed.diagnostic_output,
            binding_receipt_sha256=binding_receipt_sha256,
            receipt_sha256="",
        )
        return SelfTraceReceiptV1(
            case_id=bound.case_id,
            system_cell_id=bound.system_cell_id,
            system_cell_sha256=bound.system_cell_sha256,
            checkpoint_tree_sha256=bound.checkpoint_tree_sha256,
            expected_served_model_id=bound.expected_served_model_id,
            actor_generation_id=bound.actor_generation_id,
            actor_generation_event_sha256=bound.actor_generation_event_sha256,
            actor_instance_id=bound.actor_instance_id,
            diagnostician_generation_id=bound.diagnostician_generation_id,
            diagnostician_generation_event_sha256=bound.diagnostician_generation_event_sha256,
            diagnostician_instance_id=bound.diagnostician_instance_id,
            actor_runtime_evidence=bound.actor_runtime_evidence,
            diagnostician_runtime_evidence=bound.diagnostician_runtime_evidence,
            persisted_trace=bound.persisted_trace,
            persisted_trace_sha256=bound.persisted_trace_sha256,
            investigation=bound.investigation,
            self_trace_binding_verified=bound.self_trace_binding_verified,
            diagnosis_valid=bound.diagnosis_valid,
            diagnosis_error=bound.diagnosis_error,
            diagnostic_output=bound.diagnostic_output,
            binding_receipt_sha256=bound.binding_receipt_sha256,
            receipt_sha256=_sha256(bound.to_dict(include_sha256=False)),
        )
