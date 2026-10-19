"""Gold-free historical candidates from terminal precondition evidence."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_precondition_failure_gold import (
    ALFWORLD_PRECONDITION_FAILURE_GOLD_SCHEMA,
)
from lh_platform.tracing.trace_observability import TraceSymbolicStateEvidenceV1
from lh_platform.tracing.types import TraceCaseV1

ALFWORLD_TERMINAL_PRECONDITION_DEPENDENCY_SCHEMA = (
    "lh_alfworld_terminal_precondition_dependency_slice/v1"
)


@dataclass(frozen=True)
class TerminalPreconditionCandidateReasonV1:
    event_id: str
    sequence: int
    violation_label: str
    reason_code: str
    matched_fact: str | None = None
    matched_action: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.event_id
            or self.sequence < 0
            or not self.violation_label
            or self.reason_code
            not in {"persistent_fact_transition", "missed_precondition_action"}
            or (self.matched_fact is None) is (self.matched_action is None)
        ):
            raise ValueError("terminal dependency candidate reason is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlfWorldTerminalPreconditionDependencyV1:
    source_trace_sha256: str
    symbolic_evidence_sha256: str
    source_precondition_gold_sha256: str
    rejected_action: str
    violation_labels: tuple[str, ...]
    candidate_event_ids: tuple[str, ...]
    candidate_reasons: tuple[TerminalPreconditionCandidateReasonV1, ...]
    unsupported_violation_labels: tuple[str, ...]
    candidate_generation_status: str
    terminal_anchor_included: bool = False
    privileged_causal_labels_used: bool = False
    schema: str = ALFWORLD_TERMINAL_PRECONDITION_DEPENDENCY_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != ALFWORLD_TERMINAL_PRECONDITION_DEPENDENCY_SCHEMA
            or not self.source_trace_sha256.startswith("sha256:")
            or not self.symbolic_evidence_sha256.startswith("sha256:")
            or not self.source_precondition_gold_sha256.startswith("sha256:")
            or not self.rejected_action
            or not self.violation_labels
            or self.terminal_anchor_included
            or self.privileged_causal_labels_used
            or self.candidate_generation_status
            not in {"candidates_proposed", "no_candidates", "unsupported_only"}
        ):
            raise ValueError("terminal dependency proposal identity is invalid")
        if len(set(self.candidate_event_ids)) != len(self.candidate_event_ids):
            raise ValueError("terminal dependency candidates must be unique")
        reason_event_ids = {reason.event_id for reason in self.candidate_reasons}
        if reason_event_ids != set(self.candidate_event_ids):
            raise ValueError("terminal dependency candidates lack reason evidence")
        if self.candidate_event_ids and self.candidate_generation_status != "candidates_proposed":
            raise ValueError("terminal dependency candidate status changed")
        if (
            not self.candidate_event_ids
            and self.candidate_generation_status == "candidates_proposed"
        ):
            raise ValueError("empty terminal dependency cannot claim candidates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_trace_sha256": self.source_trace_sha256,
            "symbolic_evidence_sha256": self.symbolic_evidence_sha256,
            "source_precondition_gold_sha256": self.source_precondition_gold_sha256,
            "rejected_action": self.rejected_action,
            "violation_labels": list(self.violation_labels),
            "candidate_event_ids": list(self.candidate_event_ids),
            "candidate_reasons": [reason.to_dict() for reason in self.candidate_reasons],
            "unsupported_violation_labels": list(self.unsupported_violation_labels),
            "candidate_generation_status": self.candidate_generation_status,
            "terminal_anchor_included": self.terminal_anchor_included,
            "privileged_causal_labels_used": self.privileged_causal_labels_used,
        }

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


Fact = tuple[str, tuple[str, ...]]


def _parse_fact(raw: str) -> tuple[dict[str, Any], Fact]:
    try:
        payload = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise ValueError("terminal dependency fact is not a literal object") from error
    if not isinstance(payload, dict):
        raise ValueError("terminal dependency fact must be an object")
    name = payload.get("name")
    args = payload.get("args")
    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(args, list)
        or any(not isinstance(value, str) or not value.strip() for value in args)
    ):
        raise ValueError("terminal dependency fact fields are invalid")
    return payload, (
        name.strip().lower(),
        tuple(value.strip().lower() for value in args),
    )


def _parse_facts(raw_facts: Sequence[str]) -> tuple[tuple[dict[str, Any], ...], frozenset[Fact]]:
    parsed = tuple(_parse_fact(raw) for raw in raw_facts)
    return tuple(raw for raw, _ in parsed), frozenset(fact for _, fact in parsed)


def _menu_from_event_content(content: str) -> tuple[str, ...]:
    prefix = "Admissible actions: "
    lines = [line for line in content.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("terminal dependency event menu is missing or ambiguous")
    try:
        raw_menu = ast.literal_eval(lines[0][len(prefix) :])
    except (SyntaxError, ValueError) as error:
        raise ValueError("terminal dependency event menu is invalid") from error
    if (
        not isinstance(raw_menu, list)
        or any(not isinstance(action, str) or not action for action in raw_menu)
        or len(raw_menu) != len(set(raw_menu))
    ):
        raise ValueError("terminal dependency event menu values are invalid")
    return tuple(raw_menu)


def _gold_content_sha256(gold: Mapping[str, Any]) -> str:
    canonical = dict(gold)
    claimed = canonical.pop("content_sha256", None)
    if claimed != "sha256:" + canonical_sha256(canonical):
        raise ValueError("terminal precondition gold content SHA-256 changed")
    return str(claimed)


def _values_for(facts: frozenset[Fact], name: str, first_arg: str) -> tuple[str, ...]:
    return tuple(
        sorted(args[1] for fact_name, args in facts if fact_name == name and args[0] == first_arg)
    )


def propose_alfworld_terminal_precondition_dependency(
    case: TraceCaseV1,
    symbolic_evidence: TraceSymbolicStateEvidenceV1,
    precondition_gold: Mapping[str, Any],
) -> AlfWorldTerminalPreconditionDependencyV1:
    """Map a terminal violation to historical state transitions or missed actions.

    The routine never accepts an injection id, source action, paired outcome, or
    terminal event id. Causal labels are reserved for the later evaluator.
    """

    direct_source_binding = symbolic_evidence.source_trace_sha256 == case.sha256
    projected_source_binding = all(
        dict(event.metadata).get("symbolic_state_evidence_sha256")
        == symbolic_evidence.sha256
        for event in case.events
    )
    if (
        not (direct_source_binding or projected_source_binding)
        or len(symbolic_evidence.events) != len(case.events)
        or precondition_gold.get("schema") != ALFWORLD_PRECONDITION_FAILURE_GOLD_SCHEMA
        or precondition_gold.get("task_id") != case.task_id
        or precondition_gold.get("pair_outcome_used") is not False
        or precondition_gold.get("tracer_output_used") is not False
        or precondition_gold.get("precondition_gold_assigned") is not True
    ):
        raise ValueError("terminal dependency source binding changed")
    gold_sha256 = _gold_content_sha256(precondition_gold)
    for event, state in zip(case.events, symbolic_evidence.events, strict=True):
        if (
            event.event_id != state.event_id
            or event.sequence != state.sequence
            or dict(event.metadata).get("state_snapshot_sha256")
            != state.pre_state_snapshot_sha256
        ):
            raise ValueError("terminal dependency event binding changed")
    for previous, current in pairwise(symbolic_evidence.events):
        if (
            previous.post_state_snapshot_sha256 != current.pre_state_snapshot_sha256
            or previous.facts_after != current.facts_before
        ):
            raise ValueError("terminal dependency symbolic chain is discontinuous")

    raw_terminal_facts, terminal_facts = _parse_facts(
        symbolic_evidence.events[-1].facts_after
    )
    terminal_facts_payload = sorted(raw_terminal_facts, key=canonical_sha256)
    if precondition_gold.get("source_pre_state_facts_sha256") != (
        "sha256:" + canonical_sha256(terminal_facts_payload)
    ):
        raise ValueError("terminal dependency facts differ from P0 terminal state")

    raw_labels = precondition_gold.get("violation_labels")
    raw_violations = precondition_gold.get("violations")
    grounded_arguments = precondition_gold.get("grounded_arguments")
    rejected_action = precondition_gold.get("proposal")
    if (
        not isinstance(raw_labels, list)
        or not raw_labels
        or any(not isinstance(label, str) or not label for label in raw_labels)
        or not isinstance(raw_violations, list)
        or [item.get("label") for item in raw_violations if isinstance(item, Mapping)]
        != raw_labels
        or not isinstance(grounded_arguments, Mapping)
        or not isinstance(rejected_action, str)
        or not rejected_action
    ):
        raise ValueError("terminal dependency violation gold is invalid")
    labels = tuple(raw_labels)
    args = {
        str(key): str(value).strip().lower()
        for key, value in grounded_arguments.items()
        if isinstance(key, str) and isinstance(value, str)
    }

    state_pairs = []
    for state in symbolic_evidence.events:
        _, before = _parse_facts(state.facts_before)
        _, after = _parse_facts(state.facts_after)
        state_pairs.append((before, after))

    reasons: list[TerminalPreconditionCandidateReasonV1] = []

    def add_fact_reason(sequence: int, label: str, fact: Fact) -> None:
        reasons.append(
            TerminalPreconditionCandidateReasonV1(
                event_id=case.events[sequence].event_id,
                sequence=sequence,
                violation_label=label,
                reason_code="persistent_fact_transition",
                matched_fact=str({"name": fact[0], "args": list(fact[1])}),
            )
        )

    def add_last_persistent_addition(label: str, fact: Fact) -> None:
        for sequence in range(len(state_pairs) - 1, -1, -1):
            before, after = state_pairs[sequence]
            if fact not in before and fact in after and all(
                fact in later_after for _, later_after in state_pairs[sequence:]
            ):
                add_fact_reason(sequence, label, fact)
                return

    def add_last_persistent_removal(label: str, fact: Fact) -> None:
        for sequence in range(len(state_pairs) - 1, -1, -1):
            before, after = state_pairs[sequence]
            if fact in before and fact not in after and all(
                fact not in later_after for _, later_after in state_pairs[sequence:]
            ):
                add_fact_reason(sequence, label, fact)
                return

    desired_actions: dict[str, set[str]] = {label: set() for label in labels}
    obj = args.get("object")
    tool = args.get("tool")
    receptacle = args.get("receptacle")

    for label in labels:
        if label == "agent_holds_other_object":
            add_last_persistent_addition(label, ("holdsany", ("agent1",)))
        elif label == "agent_does_not_hold_object" and obj:
            add_last_persistent_removal(label, ("holds", ("agent1", obj)))
            desired_actions[label].add(f"take {obj} from ")
        elif label == "agent_does_not_hold_tool" and tool:
            add_last_persistent_removal(label, ("holds", ("agent1", tool)))
            desired_actions[label].add(f"take {tool} from ")
        elif label == "agent_not_at_receptacle" and receptacle:
            terminal_locations = _values_for(
                terminal_facts, "atlocation", "agent1"
            )
            if len(terminal_locations) == 1:
                add_last_persistent_addition(
                    label, ("atlocation", ("agent1", terminal_locations[0]))
                )
            desired_actions[label].add(f"go to {receptacle}")
        elif label in {"receptacle_closed", "receptacle_already_closed"} and receptacle:
            add_last_persistent_removal(label, ("opened", (receptacle,)))
            desired_actions[label].add(f"open {receptacle}")
        elif label == "receptacle_already_open" and receptacle:
            add_last_persistent_addition(label, ("opened", (receptacle,)))
        elif label == "object_not_in_receptacle" and obj and receptacle:
            add_last_persistent_removal(
                label, ("inreceptacle", (obj, receptacle))
            )
            desired_actions[label].add(f"move {obj} to {receptacle}")
        elif label == "object_not_visible_or_held" and obj:
            add_last_persistent_removal(label, ("holds", ("agent1", obj)))
            desired_actions[label].add(f"take {obj} from ")
        elif label == "action_effect_already_satisfied":
            action_schema = precondition_gold.get("action_schema")
            if action_schema == "GotoLocation" and receptacle:
                receptacle_locations = _values_for(
                    terminal_facts, "receptacleatlocation", receptacle
                )
                if len(receptacle_locations) == 1:
                    add_last_persistent_addition(
                        label,
                        ("atlocation", ("agent1", receptacle_locations[0])),
                    )
            else:
                entity = obj or receptacle
                if entity:
                    add_last_persistent_addition(label, ("checked", (entity,)))

    for event in case.events:
        menu = _menu_from_event_content(event.content)
        observed_action = dict(event.metadata).get("actor_action")
        if not isinstance(observed_action, str) or not observed_action:
            raise ValueError("terminal dependency actor action is missing")
        for label, patterns in desired_actions.items():
            for pattern in sorted(patterns):
                matching = tuple(
                    action
                    for action in menu
                    if action == pattern or action.startswith(pattern)
                )
                for action in matching:
                    if action != observed_action:
                        reasons.append(
                            TerminalPreconditionCandidateReasonV1(
                                event_id=event.event_id,
                                sequence=event.sequence,
                                violation_label=label,
                                reason_code="missed_precondition_action",
                                matched_action=action,
                            )
                        )

    static_or_unreconstructed = {
        "unsupported_action_form",
        "argument_type_mismatch",
        "ambiguous_entity_reference",
        "noncanonical_entity_reference",
        "unknown_entity_reference",
        "object_not_pickupable",
        "object_not_cleanable",
        "object_not_heatable",
        "object_not_coolable",
        "object_not_toggleable",
        "object_not_sliceable",
        "receptacle_not_openable",
        "receptacle_wrong_appliance_type",
        "receptacle_cannot_contain_object",
        "tool_wrong_type",
        "compiled_dispatcher_omitted_grounded_action",
        "receptacle_has_no_location",
    }
    unsupported = tuple(sorted(set(labels).intersection(static_or_unreconstructed)))
    reason_priority = {
        "persistent_fact_transition": 0,
        "missed_precondition_action": 1,
    }
    reasons = sorted(
        set(reasons),
        key=lambda reason: (
            reason_priority[reason.reason_code],
            -reason.sequence,
            reason.event_id,
            reason.violation_label,
            reason.matched_fact or reason.matched_action or "",
        ),
    )
    candidate_event_ids: list[str] = []
    for reason in reasons:
        if reason.event_id not in candidate_event_ids:
            candidate_event_ids.append(reason.event_id)
    if candidate_event_ids:
        status = "candidates_proposed"
    elif len(unsupported) == len(labels):
        status = "unsupported_only"
    else:
        status = "no_candidates"
    return AlfWorldTerminalPreconditionDependencyV1(
        source_trace_sha256=case.sha256,
        symbolic_evidence_sha256=symbolic_evidence.sha256,
        source_precondition_gold_sha256=gold_sha256,
        rejected_action=rejected_action,
        violation_labels=labels,
        candidate_event_ids=tuple(candidate_event_ids),
        candidate_reasons=tuple(reasons),
        unsupported_violation_labels=unsupported,
        candidate_generation_status=status,
    )


__all__ = [
    "ALFWORLD_TERMINAL_PRECONDITION_DEPENDENCY_SCHEMA",
    "AlfWorldTerminalPreconditionDependencyV1",
    "TerminalPreconditionCandidateReasonV1",
    "propose_alfworld_terminal_precondition_dependency",
]
