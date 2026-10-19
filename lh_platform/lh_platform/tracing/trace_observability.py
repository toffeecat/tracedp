"""Label-free evidence projections for controlled traceback comparisons."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.types import TraceCaseV1

if TYPE_CHECKING:
    from lh_platform.tracing.alfworld_controlled_fault_injection import (
        AlfWorldControlledFaultInjectionPlanV1,
    )
    from lh_platform.training.alfworld_agentic_suite import AlfWorldBranch

TRACE_SYMBOLIC_STATE_EVIDENCE_SCHEMA = "lh_trace_symbolic_state_evidence/v1"


class TraceObservabilityLevel(StrEnum):
    OBSERVATION_RICH_STATIC_V1 = "observation_rich_static_v1"
    PER_STEP_SYMBOLIC_STATE_V1 = "per_step_symbolic_state_v1"
    PER_STEP_SYMBOLIC_DELTA_V1 = "per_step_symbolic_delta_v1"


def _require_raw_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True)
class TraceSymbolicStateEventV1:
    event_id: str
    sequence: int
    pre_state_snapshot_sha256: str
    post_state_snapshot_sha256: str
    facts_before: tuple[str, ...]
    facts_after: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.event_id or self.sequence < 0:
            raise ValueError("symbolic state event identity is invalid")
        _require_raw_sha256(
            self.pre_state_snapshot_sha256, "symbolic pre-state snapshot"
        )
        _require_raw_sha256(
            self.post_state_snapshot_sha256, "symbolic post-state snapshot"
        )
        for label, facts in (
            ("facts before", self.facts_before),
            ("facts after", self.facts_after),
        ):
            if any(not isinstance(fact, str) or not fact.strip() for fact in facts):
                raise ValueError(f"symbolic {label} must contain nonempty strings")
            if tuple(sorted(set(facts))) != facts:
                raise ValueError(f"symbolic {label} must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["facts_before"] = list(self.facts_before)
        payload["facts_after"] = list(self.facts_after)
        return payload


@dataclass(frozen=True)
class TraceSymbolicStateEvidenceV1:
    source_trace_sha256: str
    events: tuple[TraceSymbolicStateEventV1, ...]
    schema: str = TRACE_SYMBOLIC_STATE_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACE_SYMBOLIC_STATE_EVIDENCE_SCHEMA:
            raise ValueError("unsupported symbolic state evidence schema")
        if not self.source_trace_sha256.startswith("sha256:") or not self.events:
            raise ValueError("symbolic state evidence binding is invalid")
        if tuple(event.sequence for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("symbolic state evidence must be contiguous from zero")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("symbolic state event ids must be unique")
        if not any(event.facts_before or event.facts_after for event in self.events):
            raise ValueError("symbolic state evidence contains no symbolic facts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_trace_sha256": self.source_trace_sha256,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> TraceSymbolicStateEvidenceV1:
        if set(payload) != {"schema", "source_trace_sha256", "events"}:
            raise ValueError("symbolic state evidence fields changed")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("symbolic state evidence events must be a list")
        events: list[TraceSymbolicStateEventV1] = []
        expected_event_fields = {
            "event_id",
            "sequence",
            "pre_state_snapshot_sha256",
            "post_state_snapshot_sha256",
            "facts_before",
            "facts_after",
        }
        for raw in raw_events:
            if not isinstance(raw, Mapping) or set(raw) != expected_event_fields:
                raise ValueError("symbolic state event fields changed")
            sequence = raw.get("sequence")
            facts_before = raw.get("facts_before")
            facts_after = raw.get("facts_after")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or not isinstance(facts_before, list)
                or not isinstance(facts_after, list)
                or not all(isinstance(fact, str) for fact in facts_before)
                or not all(isinstance(fact, str) for fact in facts_after)
            ):
                raise ValueError("symbolic state event values are invalid")
            events.append(
                TraceSymbolicStateEventV1(
                    event_id=str(raw.get("event_id", "")),
                    sequence=sequence,
                    pre_state_snapshot_sha256=str(
                        raw.get("pre_state_snapshot_sha256", "")
                    ),
                    post_state_snapshot_sha256=str(
                        raw.get("post_state_snapshot_sha256", "")
                    ),
                    facts_before=tuple(facts_before),
                    facts_after=tuple(facts_after),
                )
            )
        return cls(
            schema=str(payload.get("schema", "")),
            source_trace_sha256=str(payload.get("source_trace_sha256", "")),
            events=tuple(events),
        )

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


def collect_alfworld_trace_symbolic_state_evidence(
    plan: AlfWorldControlledFaultInjectionPlanV1,
    case: TraceCaseV1,
    *,
    branch_factory: Callable[[], AlfWorldBranch],
) -> TraceSymbolicStateEvidenceV1:
    """Replay only the frozen failure arm and capture environment-provided facts."""

    from lh_platform.tracing.alfworld_controlled_fault_injection import (
        _run_fault_injection_arm,
    )

    if case.task_id != plan.task_id:
        raise ValueError("symbolic state collection task binding changed")
    captured: list[TraceSymbolicStateEventV1] = []

    def observe(
        step_idx: int,
        before: Any,
        after: Any,
        pre_sha256: str,
        post_sha256: str,
    ) -> None:
        captured.append(
            TraceSymbolicStateEventV1(
                event_id=f"step-{step_idx:04d}",
                sequence=step_idx,
                pre_state_snapshot_sha256=pre_sha256,
                post_state_snapshot_sha256=post_sha256,
                facts_before=tuple(sorted(set(str(fact) for fact in before.facts))),
                facts_after=tuple(sorted(set(str(fact) for fact in after.facts))),
            )
        )

    arm = _run_fault_injection_arm(
        plan,
        branch_factory=branch_factory,
        arm="injected_failure",
        step_observer=observe,
    )
    if (
        arm.success
        or arm.terminal_reason != case.observed_outcome.terminal_reason
        or len(arm.steps) != len(case.events)
        or len(captured) != len(case.events)
    ):
        raise ValueError("symbolic state replay differs from the frozen failure")
    for event, step, state in zip(case.events, arm.steps, captured, strict=True):
        metadata = dict(event.metadata)
        if (
            event.event_id != state.event_id
            or event.sequence != state.sequence
            or metadata.get("actor_action") != step.action
            or metadata.get("observation_before") != step.observation_before
            or metadata.get("observation_after") != step.observation_after
            or metadata.get("state_snapshot_sha256")
            != step.pre_state_snapshot_sha256
            or state.post_state_snapshot_sha256
            != step.post_state_snapshot_sha256
        ):
            raise ValueError("symbolic state replay event binding changed")
    return TraceSymbolicStateEvidenceV1(
        source_trace_sha256=case.sha256,
        events=tuple(captured),
    )


def build_trace_observability_projection(
    case: TraceCaseV1,
    evidence_level: TraceObservabilityLevel,
    symbolic_state_evidence: TraceSymbolicStateEvidenceV1 | None = None,
) -> TraceCaseV1:
    """Project one frozen case onto an explicitly named evidence condition."""

    if evidence_level is TraceObservabilityLevel.OBSERVATION_RICH_STATIC_V1:
        if symbolic_state_evidence is not None:
            raise ValueError("static projection does not accept symbolic state evidence")
        return case
    if evidence_level in {
        TraceObservabilityLevel.PER_STEP_SYMBOLIC_STATE_V1,
        TraceObservabilityLevel.PER_STEP_SYMBOLIC_DELTA_V1,
    }:
        evidence = symbolic_state_evidence
        if evidence is None or evidence.source_trace_sha256 != case.sha256:
            raise ValueError("symbolic state evidence is not bound to the source trace")
        if len(evidence.events) != len(case.events):
            raise ValueError("symbolic state evidence does not cover every trace event")
        projected_events = []
        for event, state in zip(case.events, evidence.events, strict=True):
            metadata = dict(event.metadata)
            if (
                state.event_id != event.event_id
                or state.sequence != event.sequence
                or metadata.get("state_snapshot_sha256")
                != state.pre_state_snapshot_sha256
            ):
                raise ValueError("symbolic state event binding changed")
            metadata.update(
                {
                    "observability_level": evidence_level.value,
                    "post_state_snapshot_sha256": (
                        state.post_state_snapshot_sha256
                    ),
                    "symbolic_state_evidence_sha256": evidence.sha256,
                }
            )
            if evidence_level is TraceObservabilityLevel.PER_STEP_SYMBOLIC_STATE_V1:
                first_label = "Symbolic facts before action"
                second_label = "Symbolic facts after action"
                first_facts = state.facts_before
                second_facts = state.facts_after
            else:
                first_label = "Symbolic facts added by action"
                second_label = "Symbolic facts removed by action"
                first_facts = tuple(
                    sorted(set(state.facts_after) - set(state.facts_before))
                )
                second_facts = tuple(
                    sorted(set(state.facts_before) - set(state.facts_after))
                )
            first_payload = json.dumps(
                first_facts, ensure_ascii=False, separators=(",", ":")
            )
            second_payload = json.dumps(
                second_facts, ensure_ascii=False, separators=(",", ":")
            )
            projected_events.append(
                replace(
                    event,
                    content=(
                        f"{event.content}\n"
                        f"{first_label}: {first_payload}\n"
                        f"{second_label}: {second_payload}"
                    ),
                    metadata=tuple(sorted(metadata.items())),
                )
            )
        return replace(case, events=tuple(projected_events))
    raise ValueError(f"unsupported trace observability level: {evidence_level}")


__all__ = [
    "TRACE_SYMBOLIC_STATE_EVIDENCE_SCHEMA",
    "TraceObservabilityLevel",
    "TraceSymbolicStateEventV1",
    "TraceSymbolicStateEvidenceV1",
    "build_trace_observability_projection",
    "collect_alfworld_trace_symbolic_state_evidence",
]
