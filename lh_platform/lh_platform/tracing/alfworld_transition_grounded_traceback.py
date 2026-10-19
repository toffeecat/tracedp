"""Transition-grounded ALFWorld traceback with replay-verifiable candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_terminal_expected_effect_debt import (
    propose_alfworld_terminal_expected_effect_debt_v2,
)
from lh_platform.tracing.trace_observability import TraceSymbolicStateEvidenceV1
from lh_platform.tracing.transition_dependency import (
    TransitionCandidateReplayBackend,
    TransitionDependencyProposalV1,
    validate_transition_candidates,
)
from lh_platform.tracing.types import (
    CounterfactualReplayEvidenceV1,
    EvidenceLevel,
    TraceCaseV1,
    TraceInterventionV1,
)

ALFWORLD_TRANSITION_GROUNDED_PROPOSAL_SCHEMA = "lh_alfworld_transition_grounded_proposal/v1"

_RELATION_BY_REASON = {
    "missed_goal_effect_action": "missed_required_transition",
    "missed_precondition_action": "missed_required_transition",
    "persistent_fact_transition": "persistent_conflicting_transition",
}


@dataclass(frozen=True)
class AlfWorldTransitionGroundedCandidateV1:
    event_id: str
    sequence: int
    observed_action: str
    violation_labels: tuple[str, ...]
    evidence_relations: tuple[str, ...]
    proposed_repair_actions: tuple[str, ...]
    fact_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.event_id or self.sequence < 0 or not self.observed_action:
            raise ValueError("transition-grounded candidate identity is invalid")
        if not self.violation_labels or not self.evidence_relations:
            raise ValueError("transition-grounded candidate lacks failure evidence")
        for values, label in (
            (self.violation_labels, "violation labels"),
            (self.evidence_relations, "evidence relations"),
            (self.proposed_repair_actions, "repair actions"),
            (self.fact_evidence, "fact evidence"),
        ):
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"transition-grounded {label} must be unique")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "violation_labels",
            "evidence_relations",
            "proposed_repair_actions",
            "fact_evidence",
        ):
            payload[field] = list(payload[field])
        return payload

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class AlfWorldTransitionGroundedProposalV1:
    source_trace_sha256: str
    symbolic_evidence_sha256: str
    failure_evidence_sha256: str
    rejected_action: str
    candidates: tuple[AlfWorldTransitionGroundedCandidateV1, ...]
    unsupported_violation_labels: tuple[str, ...]
    candidate_generation_status: str
    terminal_anchor_included: bool = False
    privileged_causal_labels_used: bool = False
    schema: str = ALFWORLD_TRANSITION_GROUNDED_PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != ALFWORLD_TRANSITION_GROUNDED_PROPOSAL_SCHEMA
            or not self.source_trace_sha256.startswith("sha256:")
            or not self.symbolic_evidence_sha256.startswith("sha256:")
            or not self.failure_evidence_sha256.startswith("sha256:")
            or not self.rejected_action
            or self.terminal_anchor_included
            or self.privileged_causal_labels_used
            or self.candidate_generation_status
            not in {"candidates_proposed", "no_candidates", "unsupported_only"}
        ):
            raise ValueError("transition-grounded proposal identity is invalid")
        event_ids = self.candidate_event_ids
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("transition-grounded candidate events must be unique")
        if bool(self.candidates) != (self.candidate_generation_status == "candidates_proposed"):
            raise ValueError("transition-grounded candidate status changed")

    @property
    def candidate_event_ids(self) -> tuple[str, ...]:
        return tuple(candidate.event_id for candidate in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_trace_sha256": self.source_trace_sha256,
            "symbolic_evidence_sha256": self.symbolic_evidence_sha256,
            "failure_evidence_sha256": self.failure_evidence_sha256,
            "rejected_action": self.rejected_action,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "unsupported_violation_labels": list(self.unsupported_violation_labels),
            "candidate_generation_status": self.candidate_generation_status,
            "terminal_anchor_included": self.terminal_anchor_included,
            "privileged_causal_labels_used": self.privileged_causal_labels_used,
        }

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class InterventionSupportedCauseV1:
    event_id: str
    replacement_action: str
    intervention_receipt_sha256: str
    arm_identity_evidence_sha256: str | None
    intervention_trial_count: int
    replay_call_count: int

    def __post_init__(self) -> None:
        if (
            not self.event_id
            or not self.replacement_action
            or not self.intervention_receipt_sha256.startswith("sha256:")
            or self.intervention_trial_count < 1
            or self.replay_call_count < 3
        ):
            raise ValueError("intervention-supported cause identity is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterventionSupportedRepairSiteV1:
    event_id: str
    replacement_action: str
    intervention_receipt_sha256: str
    arm_identity_evidence_sha256: str | None
    intervention_trial_count: int
    replay_call_count: int
    site_role: str = "compensating_repair_site"

    def __post_init__(self) -> None:
        if (
            not self.event_id
            or not self.replacement_action
            or not self.intervention_receipt_sha256.startswith("sha256:")
            or self.intervention_trial_count < 1
            or self.replay_call_count < 3
            or self.site_role != "compensating_repair_site"
        ):
            raise ValueError("intervention-supported repair-site identity is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlfWorldTransitionGroundedValidationV1:
    source_trace_sha256: str
    proposal_sha256: str
    controller_id: str
    verified_causes: tuple[InterventionSupportedCauseV1, ...]
    compensating_repair_sites: tuple[InterventionSupportedRepairSiteV1, ...]
    candidate_event_ids_tested: tuple[str, ...]
    intervention_trial_count: int
    replay_call_count: int
    max_intervention_trials: int
    candidate_set_exhausted: bool
    evidence_level: EvidenceLevel
    causal_validation_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.source_trace_sha256.startswith("sha256:")
            or not self.proposal_sha256.startswith("sha256:")
            or not self.controller_id
            or self.intervention_trial_count < 0
            or self.replay_call_count < 0
            or self.max_intervention_trials < 1
            or self.intervention_trial_count > self.max_intervention_trials
            or len(self.candidate_event_ids_tested) != len(set(self.candidate_event_ids_tested))
        ):
            raise ValueError("transition-grounded validation identity is invalid")
        if self.verified_causes or self.compensating_repair_sites:
            if self.evidence_level is not EvidenceLevel.L4_COUNTERFACTUAL:
                raise ValueError("verified transition sites require L4 evidence")
        elif (
            self.evidence_level is not EvidenceLevel.L3_LOCALIZED
            or not self.causal_validation_blockers
        ):
            raise ValueError("unverified transition validation must remain L3")
        if self.candidate_set_exhausted and "intervention_budget_exhausted" in (
            self.causal_validation_blockers
        ):
            raise ValueError("exhaustive transition validation cannot exhaust budget")

    @property
    def verified_event_ids(self) -> tuple[str, ...]:
        return tuple(cause.event_id for cause in self.verified_causes)

    @property
    def compensating_repair_event_ids(self) -> tuple[str, ...]:
        return tuple(site.event_id for site in self.compensating_repair_sites)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_trace_sha256": self.source_trace_sha256,
            "proposal_sha256": self.proposal_sha256,
            "controller_id": self.controller_id,
            "verified_causes": [cause.to_dict() for cause in self.verified_causes],
            "compensating_repair_sites": [
                site.to_dict() for site in self.compensating_repair_sites
            ],
            "candidate_event_ids_tested": list(self.candidate_event_ids_tested),
            "intervention_trial_count": self.intervention_trial_count,
            "replay_call_count": self.replay_call_count,
            "max_intervention_trials": self.max_intervention_trials,
            "candidate_set_exhausted": self.candidate_set_exhausted,
            "evidence_level": self.evidence_level.value,
            "causal_validation_blockers": list(self.causal_validation_blockers),
        }

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


class _FixedEventReplayBackend:
    def __init__(
        self,
        backend: TransitionCandidateReplayBackend,
        interventions: tuple[TraceInterventionV1, ...],
    ) -> None:
        self._backend = backend
        self._interventions = interventions
        self.controller_id = backend.controller_id

    def interventions_for_event(
        self,
        case: TraceCaseV1,
        event_id: str,
    ) -> tuple[TraceInterventionV1, ...]:
        del case
        if any(item.target_event_id != event_id for item in self._interventions):
            raise ValueError("fixed replay interventions changed target event")
        return self._interventions

    def evaluate(
        self,
        case: TraceCaseV1,
        intervention: TraceInterventionV1,
    ) -> CounterfactualReplayEvidenceV1:
        if intervention not in self._interventions:
            raise ValueError("replay intervention is outside the frozen event set")
        return self._backend.evaluate(case, intervention)


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def build_alfworld_transition_grounded_proposal(
    case: TraceCaseV1,
    symbolic_evidence: TraceSymbolicStateEvidenceV1,
    failure_evidence: Mapping[str, Any],
) -> AlfWorldTransitionGroundedProposalV1:
    """Build ordered candidates from task, terminal obligation, and state deltas.

    The input failure evidence is benchmark-owned one-step precondition evidence.
    It must not contain the controlled injection event or paired causal outcome.
    """

    source = propose_alfworld_terminal_expected_effect_debt_v2(
        case,
        symbolic_evidence,
        dict(failure_evidence),
    )
    events = {event.event_id: event for event in case.events}
    candidates: list[AlfWorldTransitionGroundedCandidateV1] = []
    for event_id in source.candidate_event_ids:
        event = events[event_id]
        reasons = tuple(
            reason for reason in source.candidate_reasons if reason.event_id == event_id
        )
        observed_action = dict(event.metadata).get("actor_action")
        if not isinstance(observed_action, str) or not observed_action:
            raise ValueError("transition-grounded candidate actor action is missing")
        candidates.append(
            AlfWorldTransitionGroundedCandidateV1(
                event_id=event_id,
                sequence=event.sequence,
                observed_action=observed_action,
                violation_labels=_unique([reason.violation_label for reason in reasons]),
                evidence_relations=_unique(
                    [_RELATION_BY_REASON[reason.reason_code] for reason in reasons]
                ),
                proposed_repair_actions=_unique(
                    [
                        reason.matched_action
                        for reason in reasons
                        if reason.matched_action is not None
                    ]
                ),
                fact_evidence=_unique(
                    [reason.matched_fact for reason in reasons if reason.matched_fact is not None]
                ),
            )
        )
    return AlfWorldTransitionGroundedProposalV1(
        source_trace_sha256=source.source_trace_sha256,
        symbolic_evidence_sha256=source.symbolic_evidence_sha256,
        failure_evidence_sha256=source.source_precondition_gold_sha256,
        rejected_action=source.rejected_action,
        candidates=tuple(candidates),
        unsupported_violation_labels=source.unsupported_violation_labels,
        candidate_generation_status=source.candidate_generation_status,
    )


def validate_alfworld_transition_grounded_proposal(
    case: TraceCaseV1,
    proposal: AlfWorldTransitionGroundedProposalV1,
    replay_backend: TransitionCandidateReplayBackend,
    *,
    max_intervention_trials: int,
) -> AlfWorldTransitionGroundedValidationV1:
    """Verify each proposed event against a bounded three-arm replay family.

    Candidate events are all visited unless the frozen intervention budget is
    insufficient. A successful action verifies that event within the tested
    intervention family; it does not claim uniqueness or global minimality.
    """

    if proposal.source_trace_sha256 != case.sha256:
        raise ValueError("transition-grounded proposal is not bound to the trace")
    if not replay_backend.controller_id:
        raise ValueError("transition-grounded replay controller identity is empty")
    if max_intervention_trials < 1:
        raise ValueError("transition-grounded replay budget must be positive")

    verified: list[InterventionSupportedCauseV1] = []
    compensating: list[InterventionSupportedRepairSiteV1] = []
    tested: list[str] = []
    trial_count = 0
    replay_call_count = 0
    blockers: list[str] = []
    for candidate in proposal.candidates:
        raw_interventions = replay_backend.interventions_for_event(
            case,
            candidate.event_id,
        )
        if len(raw_interventions) != len(set(raw_interventions)) or any(
            intervention.target_event_id != candidate.event_id for intervention in raw_interventions
        ):
            raise ValueError("transition-grounded interventions are not unique and bound")
        if candidate.proposed_repair_actions:
            allowed_actions = set(candidate.proposed_repair_actions)
            interventions = tuple(
                intervention
                for intervention in raw_interventions
                if intervention.replacement_content in allowed_actions
            )
        else:
            interventions = raw_interventions

        remaining = max_intervention_trials - trial_count
        if len(interventions) > remaining:
            blockers.append("intervention_budget_exhausted")
            break
        if not interventions:
            blockers.append("no_replayable_intervention")
            continue
        tested.append(candidate.event_id)

        single_event_proposal = TransitionDependencyProposalV1(
            source_trace_sha256=case.sha256,
            symbolic_evidence_sha256=proposal.symbolic_evidence_sha256,
            rejected_action=proposal.rejected_action,
            violated_precondition="|".join(candidate.violation_labels),
            failure_observed_after_event_id=case.interruption_event_id,
            dependency_event_ids=(candidate.event_id,),
            candidate_event_ids=(candidate.event_id,),
            explanation=(
                "Transition-grounded candidate selected from expected-versus-observed "
                "state evidence."
            ),
        )
        result = validate_transition_candidates(
            case,
            single_event_proposal,
            _FixedEventReplayBackend(replay_backend, interventions),
        )
        trial_count += result.intervention_trial_count
        replay_call_count += result.replay_call_count
        if result.verified_event_id is not None:
            intervention = result.verified_intervention
            receipt = result.intervention_receipt
            if intervention is None or receipt is None:
                raise ValueError("verified transition cause lacks replay evidence")
            values = {
                "event_id": result.verified_event_id,
                "replacement_action": intervention.replacement_content,
                "intervention_receipt_sha256": receipt.receipt_sha256,
                "arm_identity_evidence_sha256": (result.arm_identity_evidence_sha256),
                "intervention_trial_count": result.intervention_trial_count,
                "replay_call_count": result.replay_call_count,
            }
            if "missed_required_transition" in candidate.evidence_relations:
                verified.append(InterventionSupportedCauseV1(**values))
            else:
                compensating.append(
                    InterventionSupportedRepairSiteV1(
                        **values,
                    )
                )

    candidate_set_exhausted = (
        len(tested) == len(proposal.candidates) and "no_replayable_intervention" not in blockers
    )
    if not verified and "intervention_budget_exhausted" not in blockers:
        blockers.append(
            "no_transition_violation_supported_cause"
            if compensating
            else "no_intervention_supported_site"
        )
    return AlfWorldTransitionGroundedValidationV1(
        source_trace_sha256=case.sha256,
        proposal_sha256=proposal.sha256,
        controller_id=replay_backend.controller_id,
        verified_causes=tuple(verified),
        compensating_repair_sites=tuple(compensating),
        candidate_event_ids_tested=tuple(tested),
        intervention_trial_count=trial_count,
        replay_call_count=replay_call_count,
        max_intervention_trials=max_intervention_trials,
        candidate_set_exhausted=candidate_set_exhausted,
        evidence_level=(
            EvidenceLevel.L4_COUNTERFACTUAL
            if verified or compensating
            else EvidenceLevel.L3_LOCALIZED
        ),
        causal_validation_blockers=tuple(dict.fromkeys(blockers)),
    )


def build_transition_grounded_preference_candidate_payload(
    case: TraceCaseV1,
    proposal: AlfWorldTransitionGroundedProposalV1,
) -> dict[str, object]:
    """Project the exact gold-blind candidate evidence shown to the control."""

    if proposal.source_trace_sha256 != case.sha256:
        raise ValueError("preference candidate projection changed source trace")
    return {
        "schema": "lh_alfworld_terminal_expected_effect_debt_slice/v2",
        "source_trace_sha256": case.sha256,
        "candidate_event_ids": list(proposal.candidate_event_ids),
        "candidate_reasons": [
            {
                "event_id": candidate.event_id,
                "reason_code": "+".join(candidate.evidence_relations),
                "matched_action": (
                    candidate.proposed_repair_actions[0]
                    if candidate.proposed_repair_actions
                    else None
                ),
                "matched_fact": (candidate.fact_evidence[0] if candidate.fact_evidence else None),
            }
            for candidate in proposal.candidates
        ],
        "candidate_generation_status": (
            "candidates_proposed" if proposal.candidates else "no_candidates"
        ),
        "privileged_causal_labels_used": False,
        "transition_grounded_proposal_sha256": proposal.sha256,
    }


__all__ = [
    "ALFWORLD_TRANSITION_GROUNDED_PROPOSAL_SCHEMA",
    "AlfWorldTransitionGroundedCandidateV1",
    "AlfWorldTransitionGroundedProposalV1",
    "AlfWorldTransitionGroundedValidationV1",
    "InterventionSupportedCauseV1",
    "InterventionSupportedRepairSiteV1",
    "build_alfworld_transition_grounded_proposal",
    "build_transition_grounded_preference_candidate_payload",
    "validate_alfworld_transition_grounded_proposal",
]
