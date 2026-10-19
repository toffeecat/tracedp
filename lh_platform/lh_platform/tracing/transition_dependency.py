"""Label-free transition dependency proposals for failed trajectories."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.runner import build_intervention_receipt
from lh_platform.tracing.trace_observability import TraceSymbolicStateEvidenceV1
from lh_platform.tracing.types import (
    CounterfactualReplayEvidenceV1,
    EvidenceLevel,
    InterventionReceiptV1,
    TraceCaseV1,
    TraceInterventionV1,
)

TRANSITION_DEPENDENCY_PROPOSAL_SCHEMA = "lh_transition_dependency_proposal/v1"


@dataclass(frozen=True)
class TransitionDependencyProposalV1:
    source_trace_sha256: str
    symbolic_evidence_sha256: str
    rejected_action: str
    violated_precondition: str
    failure_observed_after_event_id: str
    dependency_event_ids: tuple[str, ...]
    candidate_event_ids: tuple[str, ...]
    explanation: str
    schema: str = TRANSITION_DEPENDENCY_PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRANSITION_DEPENDENCY_PROPOSAL_SCHEMA:
            raise ValueError("unsupported transition dependency proposal schema")
        if not self.source_trace_sha256.startswith("sha256:"):
            raise ValueError("transition proposal source trace binding is invalid")
        if not self.symbolic_evidence_sha256.startswith("sha256:"):
            raise ValueError("transition proposal evidence binding is invalid")
        if not all(
            value.strip()
            for value in (
                self.rejected_action,
                self.violated_precondition,
                self.failure_observed_after_event_id,
                self.explanation,
            )
        ):
            raise ValueError("transition proposal identity is incomplete")
        if not self.dependency_event_ids or not self.candidate_event_ids:
            raise ValueError("transition proposal requires dependency candidates")
        if len(set(self.dependency_event_ids)) != len(self.dependency_event_ids):
            raise ValueError("transition dependency event ids must be unique")
        if len(set(self.candidate_event_ids)) != len(self.candidate_event_ids):
            raise ValueError("transition candidate event ids must be unique")
        if not set(self.candidate_event_ids).issubset(self.dependency_event_ids):
            raise ValueError("transition candidates must belong to the dependency slice")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dependency_event_ids"] = list(self.dependency_event_ids)
        payload["candidate_event_ids"] = list(self.candidate_event_ids)
        return payload

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


class TransitionCandidateReplayBackend(Protocol):
    controller_id: str

    def interventions_for_event(
        self,
        case: TraceCaseV1,
        event_id: str,
    ) -> tuple[TraceInterventionV1, ...]: ...

    def evaluate(
        self,
        case: TraceCaseV1,
        intervention: TraceInterventionV1,
    ) -> CounterfactualReplayEvidenceV1: ...


@dataclass(frozen=True)
class TransitionCandidateValidationV1:
    source_trace_sha256: str
    proposal_sha256: str
    controller_id: str
    verified_event_id: str | None
    verified_intervention: TraceInterventionV1 | None
    intervention_trial_count: int
    replay_call_count: int
    evidence_level: EvidenceLevel
    intervention_receipt: InterventionReceiptV1 | None
    arm_identity_evidence_sha256: str | None = None
    causal_validation_blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.intervention_trial_count < 0 or self.replay_call_count < 0:
            raise ValueError("transition validation counts must be non-negative")
        if self.arm_identity_evidence_sha256 is not None and (
            len(self.arm_identity_evidence_sha256) != 71
            or not self.arm_identity_evidence_sha256.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in self.arm_identity_evidence_sha256[7:]
            )
        ):
            raise ValueError("transition arm identity evidence hash is invalid")
        if self.verified_event_id is None:
            if (
                self.verified_intervention is not None
                or self.intervention_receipt is not None
                or self.arm_identity_evidence_sha256 is not None
            ):
                raise ValueError("unverified transition validation cannot carry a receipt")
            if self.evidence_level is not EvidenceLevel.L3_LOCALIZED:
                raise ValueError("unverified transition validation must remain L3")
            if not self.causal_validation_blockers:
                raise ValueError("unverified transition validation requires blockers")
        else:
            if (
                self.verified_intervention is None
                or self.intervention_receipt is None
                or self.verified_event_id != self.verified_intervention.target_event_id
            ):
                raise ValueError("verified transition intervention target changed")
            if self.intervention_trial_count < 1 or self.replay_call_count < 3:
                raise ValueError("verified transition validation requires replay trials")
            if self.evidence_level is not EvidenceLevel.L4_COUNTERFACTUAL:
                raise ValueError("verified transition validation must carry L4 evidence")
            if self.causal_validation_blockers:
                raise ValueError("verified transition validation cannot carry blockers")


def validated_arm_identity_evidence_sha256(
    evidence: CounterfactualReplayEvidenceV1,
    intervention: TraceInterventionV1,
) -> str | None:
    if not evidence.arm_identities:
        return None
    identities = {identity.arm: identity for identity in evidence.arm_identities}
    baseline = identities["fault_baseline"]
    repair = identities["predicted_repair"]
    sham = identities["matched_sham"]
    if any(
        identity.target_event_id != intervention.target_event_id for identity in identities.values()
    ):
        raise ValueError("replay arm target event identity changed")
    replacement_sha256 = "sha256:" + canonical_sha256(intervention.replacement_content)
    if repair.target_action_sha256 != replacement_sha256:
        raise ValueError("repair arm action is not bound to the intervention")
    outcome_inputs = {
        "fault_baseline": evidence.baseline.input_content_sha256,
        "predicted_repair": evidence.repair.input_content_sha256,
        "matched_sham": evidence.sham.input_content_sha256,
    }
    if any(
        identity.target_action_sha256 != outcome_inputs[arm] for arm, identity in identities.items()
    ):
        raise ValueError("replay arm action identity differs from its outcome input")
    if sham.target_action_sha256 == baseline.target_action_sha256:
        raise ValueError("matched sham replayed the fault baseline action")
    if repair.target_action_sha256 == sham.target_action_sha256:
        raise ValueError("repair and matched sham target actions are identical")
    if (
        repair.continuation_policy_sha256 != sham.continuation_policy_sha256
        or repair.seed_schedule_sha256 != sham.seed_schedule_sha256
    ):
        raise ValueError("repair and matched sham continuation protocols differ")
    if len({identity.outcome_evidence_sha256 for identity in identities.values()}) != 3:
        raise ValueError("replay outcome evidence hashes are not arm-specific")
    return "sha256:" + canonical_sha256(
        [
            identities[arm].to_dict()
            for arm in ("fault_baseline", "predicted_repair", "matched_sham")
        ]
    )


def _failure_item(case: TraceCaseV1) -> Mapping[str, Any]:
    if case.context is None:
        raise ValueError("transition dependency tracing requires evaluator context")
    try:
        payload = json.loads(case.context.evaluator_failure)
    except json.JSONDecodeError as error:
        raise ValueError("transition dependency failure item is not JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("transition dependency failure item must be an object")
    return payload


def _predicate_names(facts: Sequence[str]) -> frozenset[str]:
    names: set[str] = set()
    for fact in facts:
        try:
            payload = ast.literal_eval(fact)
        except (SyntaxError, ValueError) as error:
            raise ValueError("symbolic transition fact is not a literal object") from error
        if not isinstance(payload, Mapping):
            raise ValueError("symbolic transition fact must be an object")
        name = payload.get("name")
        args = payload.get("args")
        if not isinstance(name, str) or not name or not isinstance(args, list):
            raise ValueError("symbolic transition fact fields are invalid")
        names.add(name)
    return frozenset(names)


def propose_transition_dependency_slice(
    case: TraceCaseV1,
    symbolic_evidence: TraceSymbolicStateEvidenceV1,
) -> TransitionDependencyProposalV1:
    """Propose the resource dependency slice behind one rejected suffix action."""

    if symbolic_evidence.source_trace_sha256 != case.sha256:
        raise ValueError("symbolic transition evidence is not bound to the trace")
    if len(symbolic_evidence.events) != len(case.events):
        raise ValueError("symbolic transition evidence does not cover the trace")
    for event, state in zip(case.events, symbolic_evidence.events, strict=True):
        if event.event_id != state.event_id or event.sequence != state.sequence:
            raise ValueError("symbolic transition event binding changed")

    failure = _failure_item(case)
    rejected_action = failure.get("rejected_suffix_action")
    observed_after = failure.get("rejection_observed_after_event_id")
    terminal_reason = failure.get("terminal_reason")
    if (
        not isinstance(rejected_action, str)
        or not rejected_action.startswith("take ")
        or not isinstance(observed_after, str)
        or observed_after != case.interruption_event_id
        or terminal_reason != "suffix_action_inadmissible"
        or terminal_reason != case.observed_outcome.terminal_reason
    ):
        raise ValueError("unsupported transition failure contract")

    predicate_pairs = [
        (_predicate_names(state.facts_before), _predicate_names(state.facts_after))
        for state in symbolic_evidence.events
    ]
    if "holdsany" not in predicate_pairs[-1][1]:
        raise ValueError("rejected take is not explained by an occupied inventory")
    producer_index = None
    for index in range(len(predicate_pairs) - 1, -1, -1):
        before, after = predicate_pairs[index]
        if "holdsany" not in before and "holdsany" in after:
            producer_index = index
            break
    if producer_index is None:
        raise ValueError("occupied inventory producer is absent from the trace")
    if any("holdsany" not in after for _, after in predicate_pairs[producer_index:]):
        raise ValueError("inventory dependency is not continuous to the rejection")

    dependency = tuple(event.event_id for event in case.events[producer_index:])
    candidates = (*dependency[1:], dependency[0]) if len(dependency) > 1 else dependency
    return TransitionDependencyProposalV1(
        source_trace_sha256=case.sha256,
        symbolic_evidence_sha256=symbolic_evidence.sha256,
        rejected_action=rejected_action,
        violated_precondition="inventory_empty_before_take",
        failure_observed_after_event_id=observed_after,
        dependency_event_ids=dependency,
        candidate_event_ids=candidates,
        explanation=(
            "The rejected take requires an empty inventory, but holdsany remained true "
            "from the resource-producing event through the failure observation."
        ),
    )


def validate_transition_candidates(
    case: TraceCaseV1,
    proposal: TransitionDependencyProposalV1,
    replay_backend: TransitionCandidateReplayBackend,
) -> TransitionCandidateValidationV1:
    """Test ranked candidates until one passes exact three-arm replay validation."""

    if proposal.source_trace_sha256 != case.sha256:
        raise ValueError("transition proposal is not bound to the source trace")
    if not set(proposal.candidate_event_ids).issubset(case.event_ids):
        raise ValueError("transition proposal references events outside the trace")
    if not replay_backend.controller_id:
        raise ValueError("transition replay controller identity is empty")

    trial_count = 0
    replay_call_count = 0
    for event_id in proposal.candidate_event_ids:
        interventions = replay_backend.interventions_for_event(case, event_id)
        for intervention in interventions:
            if intervention.target_event_id != event_id:
                raise ValueError("transition intervention target differs from its candidate")
            trial_count += 1
            evidence = replay_backend.evaluate(case, intervention)
            if evidence.controller_id != replay_backend.controller_id:
                raise ValueError("transition replay controller identity mismatch")
            replay_call_count += evidence.replay_calls
            receipt = build_intervention_receipt(
                case=case,
                intervention=intervention,
                evidence=evidence,
            )
            arm_identity_evidence_sha256 = validated_arm_identity_evidence_sha256(
                evidence, intervention
            )
            sham_is_distinct = (
                receipt.sham.input_content_sha256 != receipt.baseline.input_content_sha256
                if arm_identity_evidence_sha256 is None
                else True
            )
            verified = (
                receipt.restore_verified
                and not receipt.baseline.success
                and receipt.repair.success
                and not receipt.sham.success
                and sham_is_distinct
                and receipt.repair.target_event_id == event_id
                and receipt.sham.target_event_id == event_id
            )
            if verified:
                return TransitionCandidateValidationV1(
                    source_trace_sha256=case.sha256,
                    proposal_sha256=proposal.sha256,
                    controller_id=replay_backend.controller_id,
                    verified_event_id=event_id,
                    verified_intervention=intervention,
                    intervention_trial_count=trial_count,
                    replay_call_count=replay_call_count,
                    evidence_level=EvidenceLevel.L4_COUNTERFACTUAL,
                    intervention_receipt=receipt,
                    arm_identity_evidence_sha256=arm_identity_evidence_sha256,
                )
    return TransitionCandidateValidationV1(
        source_trace_sha256=case.sha256,
        proposal_sha256=proposal.sha256,
        controller_id=replay_backend.controller_id,
        verified_event_id=None,
        verified_intervention=None,
        intervention_trial_count=trial_count,
        replay_call_count=replay_call_count,
        evidence_level=EvidenceLevel.L3_LOCALIZED,
        intervention_receipt=None,
        causal_validation_blockers=("transition_candidates_exhausted",),
    )


__all__ = [
    "TRANSITION_DEPENDENCY_PROPOSAL_SCHEMA",
    "TransitionDependencyProposalV1",
    "TransitionCandidateReplayBackend",
    "TransitionCandidateValidationV1",
    "propose_transition_dependency_slice",
    "validate_transition_candidates",
    "validated_arm_identity_evidence_sha256",
]
