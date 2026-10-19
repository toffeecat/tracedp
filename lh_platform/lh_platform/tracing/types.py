"""Canonical schemas for post-interruption traceback investigations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, TypeAlias

from lh_platform.core.evidence_contract import canonical_sha256

TRACE_CASE_SCHEMA = "lh_trace_case/v1"
TRACE_EPISODE_SCHEMA = "lh_trace_episode/v1"
INVESTIGATION_RECEIPT_SCHEMA = "lh_trace_investigation/v1"


def _sha256(value: object) -> str:
    return "sha256:" + canonical_sha256(value)


class EvidenceLevel(StrEnum):
    """Strongest evidence actually present in one investigation receipt."""

    L1_EXPLANATION = "L1_explanation"
    L2_TYPED = "L2_typed"
    L3_LOCALIZED = "L3_localized"
    L4_COUNTERFACTUAL = "L4_counterfactual"
    L5_RECOVERED = "L5_recovered"


class TraceComponent(StrEnum):
    ACTOR = "actor"
    ENVIRONMENT_OBSERVATION = "environment_observation"
    TOOL = "tool"
    MEMORY = "memory"
    ORCHESTRATOR = "orchestrator"


class TraceCauseType(StrEnum):
    TOOL_ARGUMENT_CORRUPTION = "tool_argument_corruption"
    OBSERVATION_STALENESS = "observation_staleness"
    TOOL_RESPONSE_CORRUPTION = "tool_response_corruption"
    HISTORY_MEMORY_DELETION = "history_memory_deletion"
    ORCHESTRATOR_ROUTING_ERROR = "orchestrator_routing_error"
    PREMATURE_TERMINATION = "premature_termination"


@dataclass(frozen=True)
class TraceOutcomeV1:
    success: bool
    terminal_reason: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEventV1:
    event_id: str
    sequence: int
    kind: str
    content: str
    parent_event_ids: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("trace event_id must be non-empty")
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("trace event sequence must be a non-negative integer")
        if not self.kind:
            raise ValueError("trace event kind must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "content": self.content,
            "parent_event_ids": list(self.parent_event_ids),
            "metadata": {key: value for key, value in self.metadata},
        }


@dataclass(frozen=True)
class TraceContextV1:
    """Non-localizable context shared by every tracer over one case."""

    task_specification: str
    initial_state: str
    evaluator_failure: str

    def __post_init__(self) -> None:
        if not self.task_specification.strip():
            raise ValueError("trace task specification must be non-empty")
        if not self.initial_state.strip():
            raise ValueError("trace initial state must be non-empty")
        if not self.evaluator_failure.strip():
            raise ValueError("trace evaluator failure must be non-empty")

    @property
    def prompt_text(self) -> str:
        return (
            f"Task specification:\n{self.task_specification}\n\n"
            f"Initial state:\n{self.initial_state}\n\n"
            f"Evaluator failure:\n{self.evaluator_failure}"
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TraceCaseV1:
    case_id: str
    task_id: str
    events: tuple[TraceEventV1, ...]
    interruption_event_id: str
    observed_outcome: TraceOutcomeV1
    context: TraceContextV1 | None = None
    schema: str = TRACE_CASE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACE_CASE_SCHEMA:
            raise ValueError(f"unsupported trace case schema: {self.schema}")
        if not self.case_id or not self.task_id:
            raise ValueError("trace case_id and task_id must be non-empty")
        if not self.events:
            raise ValueError("trace case must contain at least one event")
        if self.observed_outcome.success:
            raise ValueError("traceback investigation requires an observed failure")

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("trace event ids must be unique")
        sequences = [event.sequence for event in self.events]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("trace event sequence must be strictly increasing")
        if self.interruption_event_id not in set(event_ids):
            raise ValueError("interruption_event_id must reference a trace event")

        seen: set[str] = set()
        for event in self.events:
            unknown = set(event.parent_event_ids) - seen
            if unknown:
                raise ValueError(f"trace event parents must precede their child: {sorted(unknown)}")
            seen.add(event.event_id)

    @property
    def event_ids(self) -> frozenset[str]:
        return frozenset(event.event_id for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "case_id": self.case_id,
            "task_id": self.task_id,
            "events": [event.to_dict() for event in self.events],
            "interruption_event_id": self.interruption_event_id,
            "observed_outcome": self.observed_outcome.to_dict(),
        }
        if self.context is not None:
            payload["context"] = self.context.to_dict()
        return payload

    @property
    def context_text(self) -> str:
        """Global prompt context that is not a candidate causal event."""

        return self.context.prompt_text if self.context is not None else self.events[0].content

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceCaseV1:
        """Rebuild one persisted trace through the canonical validation boundary."""

        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("persisted trace events must be a list")
        events: list[TraceEventV1] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                raise ValueError("persisted trace event must be an object")
            raw_parents = raw_event.get("parent_event_ids", [])
            raw_metadata = raw_event.get("metadata", {})
            if not isinstance(raw_parents, list) or not all(
                isinstance(value, str) for value in raw_parents
            ):
                raise ValueError("persisted trace parents must be a string list")
            if not isinstance(raw_metadata, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in raw_metadata.items()
            ):
                raise ValueError("persisted trace metadata must be a string mapping")
            sequence = raw_event.get("sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                raise ValueError("persisted trace sequence must be an integer")
            events.append(
                TraceEventV1(
                    event_id=str(raw_event.get("event_id", "")),
                    sequence=sequence,
                    kind=str(raw_event.get("kind", "")),
                    content=str(raw_event.get("content", "")),
                    parent_event_ids=tuple(raw_parents),
                    metadata=tuple(sorted(raw_metadata.items())),
                )
            )

        raw_outcome = payload.get("observed_outcome")
        if not isinstance(raw_outcome, Mapping):
            raise ValueError("persisted trace observed_outcome must be an object")
        success = raw_outcome.get("success")
        if not isinstance(success, bool):
            raise ValueError("persisted trace outcome success must be boolean")
        raw_score = raw_outcome.get("score")
        if raw_score is not None and (
            isinstance(raw_score, bool) or not isinstance(raw_score, (int, float))
        ):
            raise ValueError("persisted trace outcome score must be numeric or null")
        raw_context = payload.get("context")
        context = None
        if raw_context is not None:
            if not isinstance(raw_context, Mapping):
                raise ValueError("persisted trace context must be an object")
            context = TraceContextV1(
                task_specification=str(raw_context.get("task_specification", "")),
                initial_state=str(raw_context.get("initial_state", "")),
                evaluator_failure=str(raw_context.get("evaluator_failure", "")),
            )
        return cls(
            schema=str(payload.get("schema", "")),
            case_id=str(payload.get("case_id", "")),
            task_id=str(payload.get("task_id", "")),
            events=tuple(events),
            interruption_event_id=str(payload.get("interruption_event_id", "")),
            observed_outcome=TraceOutcomeV1(
                success=success,
                terminal_reason=str(raw_outcome.get("terminal_reason", "")),
                score=None if raw_score is None else float(raw_score),
            ),
            context=context,
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class TraceEpisodeContextV1:
    """Non-localizable context for either a successful or failed episode."""

    task_specification: str
    initial_state: str
    evaluator_outcome: str

    def __post_init__(self) -> None:
        if not self.task_specification.strip():
            raise ValueError("trace episode task specification must be non-empty")
        if not self.initial_state.strip():
            raise ValueError("trace episode initial state must be non-empty")
        if not self.evaluator_outcome.strip():
            raise ValueError("trace episode evaluator outcome must be non-empty")

    @property
    def prompt_text(self) -> str:
        return (
            f"Task specification:\n{self.task_specification}\n\n"
            f"Initial state:\n{self.initial_state}\n\n"
            f"Evaluator outcome:\n{self.evaluator_outcome}"
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEpisodeV1:
    """One terminal episode shared by outcome detection and failure tracing."""

    case_id: str
    task_id: str
    events: tuple[TraceEventV1, ...]
    terminal_event_id: str
    observed_outcome: TraceOutcomeV1
    context: TraceEpisodeContextV1 | None = None
    schema: str = TRACE_EPISODE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACE_EPISODE_SCHEMA:
            raise ValueError(f"unsupported trace episode schema: {self.schema}")
        if not self.case_id or not self.task_id:
            raise ValueError("trace episode case_id and task_id must be non-empty")
        if not self.events:
            raise ValueError("trace episode must contain at least one event")

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("trace episode event ids must be unique")
        sequences = [event.sequence for event in self.events]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("trace episode event sequence must be strictly increasing")
        if self.terminal_event_id not in set(event_ids):
            raise ValueError("terminal_event_id must reference a trace episode event")

        seen: set[str] = set()
        for event in self.events:
            unknown = set(event.parent_event_ids) - seen
            if unknown:
                raise ValueError(
                    f"trace episode event parents must precede their child: {sorted(unknown)}"
                )
            seen.add(event.event_id)

    @property
    def event_ids(self) -> frozenset[str]:
        return frozenset(event.event_id for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "case_id": self.case_id,
            "task_id": self.task_id,
            "events": [event.to_dict() for event in self.events],
            "terminal_event_id": self.terminal_event_id,
            "observed_outcome": self.observed_outcome.to_dict(),
        }
        if self.context is not None:
            payload["context"] = self.context.to_dict()
        return payload

    @property
    def context_text(self) -> str:
        return self.context.prompt_text if self.context is not None else self.events[0].content

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceEpisodeV1:
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("persisted trace episode events must be a list")
        events: list[TraceEventV1] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                raise ValueError("persisted trace episode event must be an object")
            raw_parents = raw_event.get("parent_event_ids", [])
            raw_metadata = raw_event.get("metadata", {})
            if not isinstance(raw_parents, list) or not all(
                isinstance(value, str) for value in raw_parents
            ):
                raise ValueError("persisted trace episode parents must be a string list")
            if not isinstance(raw_metadata, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in raw_metadata.items()
            ):
                raise ValueError("persisted trace episode metadata must be a string mapping")
            sequence = raw_event.get("sequence")
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                raise ValueError("persisted trace episode sequence must be an integer")
            events.append(
                TraceEventV1(
                    event_id=str(raw_event.get("event_id", "")),
                    sequence=sequence,
                    kind=str(raw_event.get("kind", "")),
                    content=str(raw_event.get("content", "")),
                    parent_event_ids=tuple(raw_parents),
                    metadata=tuple(sorted(raw_metadata.items())),
                )
            )

        raw_outcome = payload.get("observed_outcome")
        if not isinstance(raw_outcome, Mapping):
            raise ValueError("persisted trace episode observed_outcome must be an object")
        success = raw_outcome.get("success")
        if not isinstance(success, bool):
            raise ValueError("persisted trace episode outcome success must be boolean")
        raw_score = raw_outcome.get("score")
        if raw_score is not None and (
            isinstance(raw_score, bool) or not isinstance(raw_score, (int, float))
        ):
            raise ValueError("persisted trace episode outcome score must be numeric or null")
        raw_context = payload.get("context")
        context = None
        if raw_context is not None:
            if not isinstance(raw_context, Mapping):
                raise ValueError("persisted trace episode context must be an object")
            context = TraceEpisodeContextV1(
                task_specification=str(raw_context.get("task_specification", "")),
                initial_state=str(raw_context.get("initial_state", "")),
                evaluator_outcome=str(raw_context.get("evaluator_outcome", "")),
            )
        return cls(
            schema=str(payload.get("schema", "")),
            case_id=str(payload.get("case_id", "")),
            task_id=str(payload.get("task_id", "")),
            events=tuple(events),
            terminal_event_id=str(payload.get("terminal_event_id", "")),
            observed_outcome=TraceOutcomeV1(
                success=success,
                terminal_reason=str(raw_outcome.get("terminal_reason", "")),
                score=None if raw_score is None else float(raw_score),
            ),
            context=context,
        )

    def as_failure_case(self) -> TraceCaseV1:
        if self.observed_outcome.success:
            raise ValueError("successful episode cannot become a failure case")
        context = None
        if self.context is not None:
            context = TraceContextV1(
                task_specification=self.context.task_specification,
                initial_state=self.context.initial_state,
                evaluator_failure=self.context.evaluator_outcome,
            )
        return TraceCaseV1(
            case_id=self.case_id,
            task_id=self.task_id,
            events=self.events,
            interruption_event_id=self.terminal_event_id,
            observed_outcome=self.observed_outcome,
            context=context,
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


TraceInputV1: TypeAlias = TraceCaseV1 | TraceEpisodeV1


def trace_input_from_dict(payload: Mapping[str, Any]) -> TraceInputV1:
    """Parse one canonical failure case or terminal episode, failing closed."""

    schema = payload.get("schema")
    if schema == TRACE_CASE_SCHEMA:
        return TraceCaseV1.from_dict(payload)
    if schema == TRACE_EPISODE_SCHEMA:
        return TraceEpisodeV1.from_dict(payload)
    raise ValueError(f"unsupported trace input schema: {schema}")


def trace_terminal_event_id(trace_input: TraceInputV1) -> str:
    """Return the terminal reference without conflating success with interruption."""

    if isinstance(trace_input, TraceEpisodeV1):
        return trace_input.terminal_event_id
    return trace_input.interruption_event_id


@dataclass(frozen=True)
class TraceInterventionV1:
    target_event_id: str
    replacement_content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RootCauseClaimV1:
    causal_event_ids: tuple[str, ...]
    component: TraceComponent
    cause_type: TraceCauseType
    explanation: str
    confidence: float
    intervention: TraceInterventionV1 | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "causal_event_ids": list(self.causal_event_ids),
            "component": self.component,
            "cause_type": self.cause_type,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "intervention": self.intervention.to_dict() if self.intervention else None,
        }


@dataclass(frozen=True)
class TraceUsageV1:
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    replay_calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayOutcomeV1:
    success: bool
    terminal_reason: str
    target_event_id: str
    input_content_sha256: str
    restored_state_sha256: str
    terminal_state_sha256: str
    outcome_receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayArmIdentityV1:
    """Protocol identity that distinguishes replay arms beyond one target action."""

    arm: str
    target_event_id: str
    target_action_sha256: str
    continuation_policy_sha256: str
    seed_schedule_sha256: str
    restore_protocol_receipt_sha256: str
    outcome_evidence_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in {"fault_baseline", "predicted_repair", "matched_sham"}:
            raise ValueError("unsupported replay arm identity")
        if not self.target_event_id:
            raise ValueError("replay arm target event is empty")
        digests = (
            self.target_action_sha256,
            self.continuation_policy_sha256,
            self.seed_schedule_sha256,
            self.restore_protocol_receipt_sha256,
            self.outcome_evidence_sha256,
        )
        if any(
            len(value) != 71
            or not value.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value[7:])
            for value in digests
        ):
            raise ValueError("replay arm identity requires canonical SHA-256 digests")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class CounterfactualReplayEvidenceV1:
    """Raw evidence returned by a benchmark-owned replay controller."""

    controller_id: str
    source_trace_sha256: str
    source_snapshot_sha256: str
    restore_verified: bool
    restore_receipt_sha256: str
    baseline: ReplayOutcomeV1
    repair: ReplayOutcomeV1
    sham: ReplayOutcomeV1
    replay_calls: int
    arm_identities: tuple[ReplayArmIdentityV1, ...] = ()

    def __post_init__(self) -> None:
        if self.arm_identities:
            arms = tuple(identity.arm for identity in self.arm_identities)
            if len(arms) != 3 or len(set(arms)) != 3 or set(arms) != {
                "fault_baseline",
                "predicted_repair",
                "matched_sham",
            }:
                raise ValueError("counterfactual evidence requires exactly three arm identities")


@dataclass(frozen=True)
class InterventionReceiptV1:
    """Hash-bound repair and matched-sham replay evidence."""

    controller_id: str
    source_trace_sha256: str
    target_event_id: str
    replacement_content_sha256: str
    source_snapshot_sha256: str
    restore_verified: bool
    restore_receipt_sha256: str
    baseline: ReplayOutcomeV1
    repair: ReplayOutcomeV1
    sham: ReplayOutcomeV1
    receipt_sha256: str

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "controller_id": self.controller_id,
            "source_trace_sha256": self.source_trace_sha256,
            "target_event_id": self.target_event_id,
            "replacement_content_sha256": self.replacement_content_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "restore_verified": self.restore_verified,
            "restore_receipt_sha256": self.restore_receipt_sha256,
            "baseline": self.baseline.to_dict(),
            "repair": self.repair.to_dict(),
            "sham": self.sham.to_dict(),
        }
        if include_sha256:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class InvestigationReceipt:
    case_id: str
    tracer_id: str
    tracer_version: str
    source_trace_sha256: str
    claim: RootCauseClaimV1
    evidence_level: EvidenceLevel
    root_cause_verified: bool
    intervention_receipt: InterventionReceiptV1 | None
    usage: TraceUsageV1
    diagnostic_output_sha256: str
    causal_validation_blockers: tuple[str, ...]
    receipt_sha256: str
    schema: str = INVESTIGATION_RECEIPT_SCHEMA

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "case_id": self.case_id,
            "tracer_id": self.tracer_id,
            "tracer_version": self.tracer_version,
            "source_trace_sha256": self.source_trace_sha256,
            "claim": self.claim.to_dict(),
            "evidence_level": self.evidence_level.value,
            "root_cause_verified": self.root_cause_verified,
            "intervention_receipt": (
                self.intervention_receipt.to_dict() if self.intervention_receipt else None
            ),
            "usage": self.usage.to_dict(),
            "diagnostic_output_sha256": self.diagnostic_output_sha256,
            "causal_validation_blockers": list(self.causal_validation_blockers),
        }
        if include_sha256:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload
