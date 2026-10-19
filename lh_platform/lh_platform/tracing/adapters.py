"""Built-in tracing adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.models.base import Message, ModelBase
from lh_platform.tracing.registry import TRACERS
from lh_platform.tracing.types import (
    EvidenceLevel,
    RootCauseClaimV1,
    TraceCaseV1,
    TraceCauseType,
    TraceComponent,
    TraceInterventionV1,
    TraceUsageV1,
)


class TraceInvestigationError(RuntimeError):
    """Raised when a tracer cannot produce a valid, auditable claim."""

    def __init__(self, message: str, *, diagnostic_output: str | None = None) -> None:
        super().__init__(message)
        self.diagnostic_output = diagnostic_output


@dataclass(frozen=True)
class InvestigationDraft:
    tracer_id: str
    tracer_version: str
    claim: RootCauseClaimV1
    evidence_level: EvidenceLevel
    usage: TraceUsageV1
    diagnostic_output_sha256: str
    replay_required: bool = False


def _model_visible_case_payload(case: TraceCaseV1) -> dict[str, Any]:
    """Project a persisted trace into the diagnostician-visible contract.

    Case/task identifiers and per-event metadata can encode hidden execution
    identity, so they are neutralized.  Explicit non-localizable context is
    retained: task specification, initial state, and the benchmark failure item
    are required for natural traceback without becoming candidate event IDs.
    """

    payload = case.to_dict()
    payload["case_id"] = "model-visible-case"
    payload["task_id"] = "model-visible-task"
    for event in payload["events"]:
        event["metadata"] = {}
    return payload


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].lstrip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise TraceInvestigationError("diagnostician did not return valid JSON") from error
    if not isinstance(payload, dict):
        raise TraceInvestigationError("diagnostician JSON must be an object")
    return payload


def _claim_from_payload(payload: dict[str, Any], case: TraceCaseV1) -> RootCauseClaimV1:
    raw_ids = payload.get("causal_event_ids")
    if not isinstance(raw_ids, list) or not raw_ids or not all(isinstance(v, str) for v in raw_ids):
        raise TraceInvestigationError("causal_event_ids must be a non-empty string list")
    causal_event_ids = tuple(raw_ids)
    unknown = set(causal_event_ids) - case.event_ids
    if unknown:
        raise TraceInvestigationError(f"claim references unknown trace events: {sorted(unknown)}")

    try:
        component = TraceComponent(payload.get("component"))
    except (TypeError, ValueError) as error:
        allowed = ", ".join(value.value for value in TraceComponent)
        raise TraceInvestigationError(f"component must be one of: {allowed}") from error
    try:
        cause_type = TraceCauseType(payload.get("cause_type"))
    except (TypeError, ValueError) as error:
        allowed = ", ".join(value.value for value in TraceCauseType)
        raise TraceInvestigationError(f"cause_type must be one of: {allowed}") from error
    explanation = payload.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise TraceInvestigationError("explanation must be non-empty")

    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TraceInvestigationError("confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise TraceInvestigationError("confidence must be between 0 and 1")

    intervention = None
    raw_intervention = payload.get("intervention")
    if raw_intervention is not None:
        if not isinstance(raw_intervention, dict):
            raise TraceInvestigationError("intervention must be an object or null")
        target = raw_intervention.get("target_event_id")
        replacement = raw_intervention.get("replacement_content")
        if not isinstance(target, str) or target not in case.event_ids:
            raise TraceInvestigationError("intervention target_event_id must reference the trace")
        if not isinstance(replacement, str) or not replacement:
            raise TraceInvestigationError("intervention replacement_content must be non-empty")
        intervention = TraceInterventionV1(
            target_event_id=target,
            replacement_content=replacement,
        )

    return RootCauseClaimV1(
        causal_event_ids=causal_event_ids,
        component=component,
        cause_type=cause_type,
        explanation=explanation,
        confidence=float(confidence),
        intervention=intervention,
    )


@TRACERS.register("all_at_once")
class AllAtOnceTracer:
    """Ask one diagnostician to inspect the complete persisted trace."""

    tracer_id = "all_at_once"
    tracer_version = "1"
    tracer_capabilities = ("offline_diagnosis", "typed_localization")
    evidence_ceiling = EvidenceLevel.L3_LOCALIZED
    literature_status = "platform_builtin_control"
    execution_path = "post_interruption_offline"

    def __init__(self, *, model: ModelBase, require_intervention: bool = False) -> None:
        self._model = model
        self._require_intervention = require_intervention

    def investigate(self, case: TraceCaseV1) -> InvestigationDraft:
        system = (
            "You diagnose failed agent trajectories. Return one JSON object with exactly "
            "causal_event_ids, component, cause_type, explanation, confidence, and intervention. "
            "causal_event_ids must be a non-empty JSON string array using only ids in the trace. "
            "component must be one of actor, environment_observation, tool, memory, orchestrator. "
            "cause_type must be one of tool_argument_corruption, observation_staleness, "
            "tool_response_corruption, history_memory_deletion, orchestrator_routing_error, "
            "premature_termination. confidence must be a JSON number between 0 and 1. "
            "intervention is an object with target_event_id and replacement_content. "
            + (
                "For this replay tracer, intervention must be a non-null object. "
                if self._require_intervention
                else "For this static tracer, intervention may be null. "
            )
            + "Do not claim the repair was verified."
        )
        user = json.dumps(_model_visible_case_payload(case), ensure_ascii=False, sort_keys=True)
        output = self._model.generate(
            [Message(role="system", content=system), Message(role="user", content=user)],
        )
        try:
            payload = _json_object(output.text)
            claim = _claim_from_payload(payload, case)
        except TraceInvestigationError as error:
            raise TraceInvestigationError(
                str(error),
                diagnostic_output=output.text,
            ) from error
        return InvestigationDraft(
            tracer_id=self.tracer_id,
            tracer_version=self.tracer_version,
            claim=claim,
            evidence_level=EvidenceLevel.L3_LOCALIZED,
            usage=TraceUsageV1(
                model_calls=1,
                prompt_tokens=int(output.prompt_tokens or 0),
                completion_tokens=int(output.completion_tokens or 0),
            ),
            diagnostic_output_sha256="sha256:" + canonical_sha256(output.text),
        )


@TRACERS.register("counterfactual_replay")
class CounterfactualReplayTracer:
    """Diagnose once, then test the proposed repair against a matched sham."""

    tracer_id = "counterfactual_replay"
    tracer_version = "2"
    tracer_capabilities = (
        "counterfactual_replay",
        "offline_diagnosis",
        "snapshot_required",
        "typed_localization",
    )
    evidence_ceiling = EvidenceLevel.L4_COUNTERFACTUAL
    literature_status = "platform_builtin_control"
    execution_path = "post_interruption_offline_with_benchmark_replay"

    def __init__(self, *, model: ModelBase) -> None:
        self._diagnostician = AllAtOnceTracer(
            model=model,
            require_intervention=True,
        )

    def investigate(self, case: TraceCaseV1) -> InvestigationDraft:
        diagnosis = self._diagnostician.investigate(case)
        intervention = diagnosis.claim.intervention
        if intervention is None:
            raise TraceInvestigationError("counterfactual replay requires a proposed intervention")
        return InvestigationDraft(
            tracer_id=self.tracer_id,
            tracer_version=self.tracer_version,
            claim=diagnosis.claim,
            evidence_level=diagnosis.evidence_level,
            usage=TraceUsageV1(
                model_calls=diagnosis.usage.model_calls,
                prompt_tokens=diagnosis.usage.prompt_tokens,
                completion_tokens=diagnosis.usage.completion_tokens,
            ),
            diagnostic_output_sha256=diagnosis.diagnostic_output_sha256,
            replay_required=True,
        )
