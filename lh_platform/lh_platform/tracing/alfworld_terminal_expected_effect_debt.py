"""Task-conditioned missed-effect candidates for ALFWorld terminal failures."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_terminal_precondition_dependency import (
    propose_alfworld_terminal_precondition_dependency,
)
from lh_platform.tracing.trace_observability import TraceSymbolicStateEvidenceV1
from lh_platform.tracing.types import TraceCaseV1

ALFWORLD_TERMINAL_EXPECTED_EFFECT_DEBT_SCHEMA = (
    "lh_alfworld_terminal_expected_effect_debt_slice/v1"
)
ALFWORLD_TERMINAL_EXPECTED_EFFECT_DEBT_V2_SCHEMA = (
    "lh_alfworld_terminal_expected_effect_debt_slice/v2"
)

_TWO_OBJECT_GOAL = re.compile(
    r"^find two (?P<object>[a-z0-9]+) and put them in "
    r"(?P<receptacle>[a-z0-9]+)\.?$",
    flags=re.IGNORECASE,
)
_SHORT_TWO_OBJECT_GOAL = re.compile(
    r"^put two (?P<object>[a-z0-9]+) in "
    r"(?P<receptacle>[a-z0-9]+)\.?$",
    flags=re.IGNORECASE,
)
_MOVE_ACTION = re.compile(
    r"^move (?P<object>.+) to (?P<receptacle>.+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ExpectedEffectDebtCandidateReasonV1:
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
            not in {
                "missed_goal_effect_action",
                "missed_precondition_action",
                "persistent_fact_transition",
            }
            or (self.matched_fact is None) is (self.matched_action is None)
        ):
            raise ValueError("expected-effect debt candidate reason is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlfWorldTerminalExpectedEffectDebtV1:
    source_trace_sha256: str
    symbolic_evidence_sha256: str
    source_precondition_gold_sha256: str
    rejected_action: str
    task_specification: str
    goal_object_type: str | None
    goal_receptacle_type: str | None
    violation_labels: tuple[str, ...]
    candidate_event_ids: tuple[str, ...]
    expected_effect_debt_event_ids: tuple[str, ...]
    base_candidate_event_ids: tuple[str, ...]
    candidate_reasons: tuple[ExpectedEffectDebtCandidateReasonV1, ...]
    unsupported_violation_labels: tuple[str, ...]
    candidate_generation_status: str
    terminal_anchor_included: bool = False
    privileged_causal_labels_used: bool = False
    schema: str = ALFWORLD_TERMINAL_EXPECTED_EFFECT_DEBT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema
            not in {
                ALFWORLD_TERMINAL_EXPECTED_EFFECT_DEBT_SCHEMA,
                ALFWORLD_TERMINAL_EXPECTED_EFFECT_DEBT_V2_SCHEMA,
            }
            or not self.source_trace_sha256.startswith("sha256:")
            or not self.symbolic_evidence_sha256.startswith("sha256:")
            or not self.source_precondition_gold_sha256.startswith("sha256:")
            or not self.rejected_action
            or not self.task_specification
            or not self.violation_labels
            or self.terminal_anchor_included
            or self.privileged_causal_labels_used
            or self.candidate_generation_status
            not in {"candidates_proposed", "no_candidates", "unsupported_only"}
        ):
            raise ValueError("expected-effect debt proposal identity is invalid")
        if (
            len(set(self.candidate_event_ids)) != len(self.candidate_event_ids)
            or len(set(self.expected_effect_debt_event_ids))
            != len(self.expected_effect_debt_event_ids)
            or not set(self.expected_effect_debt_event_ids).issubset(
                self.candidate_event_ids
            )
            or not set(self.base_candidate_event_ids).issubset(self.candidate_event_ids)
        ):
            raise ValueError("expected-effect debt candidates are inconsistent")
        if {reason.event_id for reason in self.candidate_reasons} != set(
            self.candidate_event_ids
        ):
            raise ValueError("expected-effect debt candidates lack reason evidence")
        if self.candidate_event_ids and self.candidate_generation_status != (
            "candidates_proposed"
        ):
            raise ValueError("expected-effect debt candidate status changed")
        if (
            not self.candidate_event_ids
            and self.candidate_generation_status == "candidates_proposed"
        ):
            raise ValueError("empty expected-effect debt cannot claim candidates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_trace_sha256": self.source_trace_sha256,
            "symbolic_evidence_sha256": self.symbolic_evidence_sha256,
            "source_precondition_gold_sha256": self.source_precondition_gold_sha256,
            "rejected_action": self.rejected_action,
            "task_specification": self.task_specification,
            "goal_object_type": self.goal_object_type,
            "goal_receptacle_type": self.goal_receptacle_type,
            "violation_labels": list(self.violation_labels),
            "candidate_event_ids": list(self.candidate_event_ids),
            "expected_effect_debt_event_ids": list(
                self.expected_effect_debt_event_ids
            ),
            "base_candidate_event_ids": list(self.base_candidate_event_ids),
            "candidate_reasons": [reason.to_dict() for reason in self.candidate_reasons],
            "unsupported_violation_labels": list(self.unsupported_violation_labels),
            "candidate_generation_status": self.candidate_generation_status,
            "terminal_anchor_included": self.terminal_anchor_included,
            "privileged_causal_labels_used": self.privileged_causal_labels_used,
        }

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


def _menu_from_event_content(content: str) -> tuple[str, ...]:
    prefix = "Admissible actions: "
    lines = [line for line in content.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("expected-effect debt event menu is missing or ambiguous")
    try:
        menu = ast.literal_eval(lines[0][len(prefix) :])
    except (SyntaxError, ValueError) as error:
        raise ValueError("expected-effect debt event menu is invalid") from error
    if (
        not isinstance(menu, list)
        or any(not isinstance(action, str) or not action for action in menu)
        or len(menu) != len(set(menu))
    ):
        raise ValueError("expected-effect debt event menu values are invalid")
    return tuple(menu)


def _held_objects(raw_facts: tuple[str, ...]) -> tuple[str, ...]:
    held: set[str] = set()
    for raw in raw_facts:
        try:
            fact = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as error:
            raise ValueError("expected-effect debt fact is invalid") from error
        if not isinstance(fact, dict):
            raise ValueError("expected-effect debt fact must be an object")
        name = fact.get("name")
        args = fact.get("args")
        if (
            isinstance(name, str)
            and name.strip().casefold() == "holds"
            and isinstance(args, list)
            and len(args) == 2
            and all(isinstance(value, str) for value in args)
            and args[0].strip().casefold() == "agent1"
            and args[1].strip()
        ):
            held.add(args[1].strip().casefold())
    return tuple(sorted(held))


def _entity_type(entity: str) -> str:
    return re.sub(r"\s+\d+$", "", entity.strip().casefold())


def _parse_two_object_goal(
    task_specification: str,
    *,
    patterns: tuple[re.Pattern[str], ...],
) -> tuple[str | None, str | None]:
    matches = tuple(
        match
        for pattern in patterns
        if (match := pattern.fullmatch(task_specification)) is not None
    )
    if len(matches) > 1:
        raise ValueError("two-object goal grammar is ambiguous")
    if not matches:
        return None, None
    return (
        matches[0].group("object").casefold(),
        matches[0].group("receptacle").casefold(),
    )


def _propose_alfworld_terminal_expected_effect_debt(
    case: TraceCaseV1,
    symbolic_evidence: TraceSymbolicStateEvidenceV1,
    precondition_gold: dict[str, Any],
    *,
    goal_patterns: tuple[re.Pattern[str], ...],
    proposal_schema: str,
) -> AlfWorldTerminalExpectedEffectDebtV1:
    """Add task-goal missed deleters without reading controlled-fault labels."""

    base = propose_alfworld_terminal_precondition_dependency(
        case,
        symbolic_evidence,
        precondition_gold,
    )
    if case.context is None:
        raise ValueError("expected-effect debt requires the task specification")
    task_specification = case.context.task_specification.strip()
    goal_object_type, goal_receptacle_type = _parse_two_object_goal(
        task_specification,
        patterns=goal_patterns,
    )

    debt_reasons: list[ExpectedEffectDebtCandidateReasonV1] = []
    if (
        "agent_holds_other_object" in base.violation_labels
        and goal_object_type is not None
        and goal_receptacle_type is not None
    ):
        terminal_held = tuple(
            held
            for held in _held_objects(symbolic_evidence.events[-1].facts_after)
            if _entity_type(held) == goal_object_type
        )
        for event in reversed(case.events):
            observed_action = dict(event.metadata).get("actor_action")
            if not isinstance(observed_action, str) or not observed_action:
                raise ValueError("expected-effect debt actor action is missing")
            for action in _menu_from_event_content(event.content):
                match = _MOVE_ACTION.fullmatch(action)
                if (
                    match is not None
                    and match.group("object").strip().casefold() in terminal_held
                    and _entity_type(match.group("receptacle"))
                    == goal_receptacle_type
                    and action != observed_action
                ):
                    debt_reasons.append(
                        ExpectedEffectDebtCandidateReasonV1(
                            event_id=event.event_id,
                            sequence=event.sequence,
                            violation_label="agent_holds_other_object",
                            reason_code="missed_goal_effect_action",
                            matched_action=action,
                        )
                    )

    base_reasons = tuple(
        ExpectedEffectDebtCandidateReasonV1(
            event_id=reason.event_id,
            sequence=reason.sequence,
            violation_label=reason.violation_label,
            reason_code=reason.reason_code,
            matched_fact=reason.matched_fact,
            matched_action=reason.matched_action,
        )
        for reason in base.candidate_reasons
    )
    reasons = tuple(dict.fromkeys((*debt_reasons, *base_reasons)))
    expected_effect_debt_event_ids = tuple(
        dict.fromkeys(reason.event_id for reason in debt_reasons)
    )
    candidate_event_ids = tuple(dict.fromkeys(reason.event_id for reason in reasons))
    if candidate_event_ids:
        status = "candidates_proposed"
    elif len(base.unsupported_violation_labels) == len(base.violation_labels):
        status = "unsupported_only"
    else:
        status = "no_candidates"
    return AlfWorldTerminalExpectedEffectDebtV1(
        source_trace_sha256=base.source_trace_sha256,
        symbolic_evidence_sha256=base.symbolic_evidence_sha256,
        source_precondition_gold_sha256=base.source_precondition_gold_sha256,
        rejected_action=base.rejected_action,
        task_specification=task_specification,
        goal_object_type=goal_object_type,
        goal_receptacle_type=goal_receptacle_type,
        violation_labels=base.violation_labels,
        candidate_event_ids=candidate_event_ids,
        expected_effect_debt_event_ids=expected_effect_debt_event_ids,
        base_candidate_event_ids=base.candidate_event_ids,
        candidate_reasons=reasons,
        unsupported_violation_labels=base.unsupported_violation_labels,
        candidate_generation_status=status,
        schema=proposal_schema,
    )


def propose_alfworld_terminal_expected_effect_debt(
    case: TraceCaseV1,
    symbolic_evidence: TraceSymbolicStateEvidenceV1,
    precondition_gold: dict[str, Any],
) -> AlfWorldTerminalExpectedEffectDebtV1:
    """Frozen V1: recognize only the original long two-object goal template."""

    return _propose_alfworld_terminal_expected_effect_debt(
        case,
        symbolic_evidence,
        precondition_gold,
        goal_patterns=(_TWO_OBJECT_GOAL,),
        proposal_schema=ALFWORLD_TERMINAL_EXPECTED_EFFECT_DEBT_SCHEMA,
    )


def propose_alfworld_terminal_expected_effect_debt_v2(
    case: TraceCaseV1,
    symbolic_evidence: TraceSymbolicStateEvidenceV1,
    precondition_gold: dict[str, Any],
) -> AlfWorldTerminalExpectedEffectDebtV1:
    """Recognize both official ALFWorld two-object task phrasings."""

    return _propose_alfworld_terminal_expected_effect_debt(
        case,
        symbolic_evidence,
        precondition_gold,
        goal_patterns=(_TWO_OBJECT_GOAL, _SHORT_TWO_OBJECT_GOAL),
        proposal_schema=ALFWORLD_TERMINAL_EXPECTED_EFFECT_DEBT_V2_SCHEMA,
    )


__all__ = [
    "ALFWORLD_TERMINAL_EXPECTED_EFFECT_DEBT_SCHEMA",
    "ALFWORLD_TERMINAL_EXPECTED_EFFECT_DEBT_V2_SCHEMA",
    "AlfWorldTerminalExpectedEffectDebtV1",
    "ExpectedEffectDebtCandidateReasonV1",
    "propose_alfworld_terminal_expected_effect_debt",
    "propose_alfworld_terminal_expected_effect_debt_v2",
]
