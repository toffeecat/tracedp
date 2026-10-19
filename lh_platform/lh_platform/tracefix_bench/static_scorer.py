"""Pure, locally designed static event-group scoring for the v0.1 contract.

This module implements SCORER_CONTRACT_V0_1 section 3, using the inherited
terminal attribution scorer's independent-gold and fixed-denominator boundary.
It performs no I/O and does not establish causal verification. Dataset release,
authority receipt contents, work-item identity, and registered budget binding
remain responsibilities of the loader and future runner receipt. The present
implementation remains implementation_unverified pending independent oracles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracefix_bench.contracts import (
    AttributionGoldV1,
    BenchmarkCaseRefV1,
    ExecutionSemantics,
    GoldAuthority,
    GoldDisposition,
    LocatorFailureKind,
    LocatorReceiptV1,
    LocatorStatus,
    LocatorUsageV1,
    PublicTracePacketV1,
    TraceFailureDomain,
    TraceFixTrack,
)
from lh_platform.tracefix_bench.scorer_contracts import (
    STATIC_ALLOWED_FINDING_KINDS_V0_1,
    BuiltinScorerKey,
    ScorerRegistryV1,
)
from lh_platform.tracing.unified import TraceFindingKind

STATIC_ATTRIBUTION_CASE_SCORE_SCHEMA = "longhorizon_tracefix_bench/static_attribution_case_score/v1"
STATIC_SCORER_IMPLEMENTATION_STATUS = "implementation_unverified"
STATIC_CASE_SCORE_BINDING_STATUS = "provisional_unbound"


class StaticAttributionDecisionStatus(StrEnum):
    """Separate denominator eligibility from locator and infrastructure outcomes."""

    MATCHED = "matched"
    UNMATCHED = "unmatched"
    ABSTAINED = "abstained"
    METHOD_FAILURE = "method_failure"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    CORRECT_REJECTION = "correct_rejection"
    UNSUPPORTED_LOCALIZATION = "unsupported_localization"
    NOT_APPLICABLE = "not_applicable"


class StaticAttributionInvalidReason(StrEnum):
    """Record closed invalidity reasons without interpreting free-form messages."""

    TRACK_MISMATCH = "track_mismatch"
    REQUEST_MISMATCH = "request_mismatch"
    GOLD_MISSING = "gold_missing"
    GOLD_CASE_MISMATCH = "gold_case_mismatch"
    GOLD_NOT_INDEPENDENT = "gold_not_independent"
    GOLD_EVENT_OUTSIDE_ROSTER = "gold_event_outside_roster"
    PREDICTION_EVENT_OUTSIDE_ROSTER = "prediction_event_outside_roster"
    LOCATOR_INFRASTRUCTURE_FAILURE = "locator_infrastructure_failure"
    LOCATOR_METHOD_FAILURE = "locator_method_failure"
    FINDING_KIND_NOT_REGISTERED = "finding_kind_not_registered"


_METHOD_REASONS = frozenset({
    StaticAttributionInvalidReason.LOCATOR_METHOD_FAILURE,
    StaticAttributionInvalidReason.FINDING_KIND_NOT_REGISTERED,
})

_INFRASTRUCTURE_REASONS = frozenset({
    StaticAttributionInvalidReason.TRACK_MISMATCH,
    StaticAttributionInvalidReason.REQUEST_MISMATCH,
    StaticAttributionInvalidReason.GOLD_MISSING,
    StaticAttributionInvalidReason.GOLD_CASE_MISMATCH,
    StaticAttributionInvalidReason.GOLD_NOT_INDEPENDENT,
    StaticAttributionInvalidReason.GOLD_EVENT_OUTSIDE_ROSTER,
    StaticAttributionInvalidReason.PREDICTION_EVENT_OUTSIDE_ROSTER,
    StaticAttributionInvalidReason.LOCATOR_INFRASTRUCTURE_FAILURE,
})


@dataclass(frozen=True)
class StaticAttributionCaseScoreV1:
    """Store one scorer-only decision and raw diagnostics, without aggregation."""

    case_id: str
    case_ref: BenchmarkCaseRefV1 | None
    suite_id: str
    suite_version: str
    population_id: str
    registry_id: str
    scorer_id: str
    scorer_version: str
    method_id: str
    locator_status: LocatorStatus
    locator_failure_domain: TraceFailureDomain | None
    locator_failure_kind: LocatorFailureKind | None
    gold_disposition: GoldDisposition | None
    decision_status: StaticAttributionDecisionStatus
    invalid_reason: StaticAttributionInvalidReason | None
    finding_kinds: tuple[TraceFindingKind, ...]
    usage: LocatorUsageV1
    localization_denominator_eligible: bool
    answered_denominator_eligible: bool
    first_match_rank: int | None
    matched_event_group: tuple[str, ...] | None
    hit_at_1: bool | None
    hit_at_3: bool | None
    fixed_denominator_reciprocal_rank: float | None
    answered_only_reciprocal_rank: float | None
    correct_rejection: bool | None
    unsupported_localization: bool | None
    method_failure: bool
    infrastructure_failure: bool
    incomplete: bool
    track: TraceFixTrack = TraceFixTrack.STATIC_ATTRIBUTION
    implementation_key: BuiltinScorerKey = BuiltinScorerKey.STATIC_ATTRIBUTION_V1
    input_binding_status: str = STATIC_CASE_SCORE_BINDING_STATUS
    implementation_status: str = STATIC_SCORER_IMPLEMENTATION_STATUS
    schema: str = STATIC_ATTRIBUTION_CASE_SCORE_SCHEMA

    def __post_init__(self) -> None:
        """Reject contradictory endpoints and Python bool-as-number coercion."""

        if self.schema != STATIC_ATTRIBUTION_CASE_SCORE_SCHEMA:
            raise ValueError("unsupported static case score schema")
        if self.implementation_status != STATIC_SCORER_IMPLEMENTATION_STATUS:
            raise ValueError("static score cannot promote implementation status")
        if self.input_binding_status != STATIC_CASE_SCORE_BINDING_STATUS:
            raise ValueError("pure static score must remain provisional and unbound")
        if self.track is not TraceFixTrack.STATIC_ATTRIBUTION or self.implementation_key is not BuiltinScorerKey.STATIC_ATTRIBUTION_V1:
            raise ValueError("static score must bind the static track and built-in scorer")
        for name in ("case_id", "suite_id", "suite_version", "population_id", "registry_id", "scorer_id", "scorer_version", "method_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if self.case_ref is not None and not isinstance(self.case_ref, BenchmarkCaseRefV1):
            raise ValueError("case_ref must be a typed case reference or null")
        for name, enum_type in (("locator_status", LocatorStatus), ("decision_status", StaticAttributionDecisionStatus)):
            if not isinstance(getattr(self, name), enum_type):
                raise ValueError(f"{name} must use its closed enum")
        for name, enum_type in (("locator_failure_domain", TraceFailureDomain), ("locator_failure_kind", LocatorFailureKind), ("gold_disposition", GoldDisposition), ("invalid_reason", StaticAttributionInvalidReason)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, enum_type):
                raise ValueError(f"{name} must use its closed enum or null")
        if (self.locator_failure_domain is None) != (self.locator_failure_kind is None):
            raise ValueError("locator failure domain and kind must be present together")
        failed = self.locator_status in {LocatorStatus.BLOCKED, LocatorStatus.INVALID}
        if failed != (self.locator_failure_domain is not None):
            raise ValueError("locator failure metadata must agree with its status")
        if type(self.finding_kinds) is not tuple or any(not isinstance(kind, TraceFindingKind) for kind in self.finding_kinds):
            raise ValueError("finding_kinds must be a tuple of closed finding kinds")
        if (self.locator_status is LocatorStatus.COMPLETED) != bool(self.finding_kinds):
            raise ValueError("only completed locators carry ranked finding kinds")
        if not isinstance(self.usage, LocatorUsageV1):
            raise ValueError("usage must preserve typed locator usage")
        for name in ("localization_denominator_eligible", "answered_denominator_eligible", "method_failure", "infrastructure_failure", "incomplete"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("hit_at_1", "hit_at_3", "correct_rejection", "unsupported_localization"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be boolean or null")
        for name in ("fixed_denominator_reciprocal_rank", "answered_only_reciprocal_rank"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite in [0, 1] or null")
        if self.first_match_rank is not None and (type(self.first_match_rank) is not int or not 1 <= self.first_match_rank <= len(self.finding_kinds)):
            raise ValueError("first_match_rank must refer to an existing positive rank")
        if (self.first_match_rank is None) != (self.matched_event_group is None):
            raise ValueError("first match rank and matched group must be present together")
        if self.matched_event_group is not None:
            group = self.matched_event_group
            if type(group) is not tuple or not group or any(type(event) is not str or not event.strip() for event in group) or len(set(group)) != len(group):
                raise ValueError("matched_event_group must contain distinct non-empty event ids")
        infra = self.decision_status is StaticAttributionDecisionStatus.INFRASTRUCTURE_INVALID
        if self.infrastructure_failure != infra or self.incomplete != infra:
            raise ValueError("infrastructure invalid scores must be incomplete")
        if (
            self.locator_failure_domain is TraceFailureDomain.INFRASTRUCTURE
            and not infra
            and self.decision_status is not StaticAttributionDecisionStatus.NOT_APPLICABLE
        ):
            raise ValueError("locator infrastructure failures cannot yield complete scores")
        expected_method_failure = (
            self.decision_status is StaticAttributionDecisionStatus.METHOD_FAILURE
        )
        if self.method_failure != expected_method_failure:
            raise ValueError("method failure flag must follow the final scorer decision")
        unsupported_kind = any(kind not in STATIC_ALLOWED_FINDING_KINDS_V0_1 for kind in self.finding_kinds)
        if self.invalid_reason is StaticAttributionInvalidReason.LOCATOR_METHOD_FAILURE:
            if self.locator_failure_domain is not TraceFailureDomain.METHOD:
                raise ValueError("locator method failure reason requires the method failure domain")
        if self.invalid_reason is StaticAttributionInvalidReason.FINDING_KIND_NOT_REGISTERED:
            if not unsupported_kind or self.locator_failure_domain is TraceFailureDomain.METHOD:
                raise ValueError("finding-kind failure requires a completed unsupported output")
        if unsupported_kind and not infra and self.decision_status is not StaticAttributionDecisionStatus.NOT_APPLICABLE and self.invalid_reason is not StaticAttributionInvalidReason.FINDING_KIND_NOT_REGISTERED:
            raise ValueError("unregistered finding kinds must produce method invalidity")
        if infra:
            if self.invalid_reason not in _INFRASTRUCTURE_REASONS:
                raise ValueError("infrastructure invalid requires an infrastructure reason")
        elif self.decision_status is StaticAttributionDecisionStatus.METHOD_FAILURE:
            if not self.method_failure or self.invalid_reason not in _METHOD_REASONS:
                raise ValueError("method failure decision requires its closed reason")
        elif self.invalid_reason is not None:
            raise ValueError("valid decisions cannot carry an invalid reason")
        if self.invalid_reason is StaticAttributionInvalidReason.GOLD_MISSING:
            if self.case_ref is not None or self.gold_disposition is not None:
                raise ValueError("missing gold cannot carry a case reference or disposition")
        if self.invalid_reason is StaticAttributionInvalidReason.GOLD_CASE_MISMATCH:
            if (
                self.case_ref is None
                or self.case_ref.case_id == self.case_id
                or self.gold_disposition is None
            ):
                raise ValueError("gold case mismatch requires a different recorded case id")
        if self.invalid_reason in {
            StaticAttributionInvalidReason.GOLD_NOT_INDEPENDENT,
            StaticAttributionInvalidReason.GOLD_EVENT_OUTSIDE_ROSTER,
            StaticAttributionInvalidReason.PREDICTION_EVENT_OUTSIDE_ROSTER,
        }:
            if (
                self.case_ref is None
                or self.case_ref.case_id != self.case_id
                or self.gold_disposition is None
            ):
                raise ValueError("gold-dependent invalidity requires matching recorded gold")
        if (
            self.invalid_reason
            is StaticAttributionInvalidReason.LOCATOR_INFRASTRUCTURE_FAILURE
            and (
                self.locator_failure_domain is not TraceFailureDomain.INFRASTRUCTURE
                or self.case_ref is None
                or self.case_ref.case_id != self.case_id
                or self.gold_disposition in {None, GoldDisposition.NOT_APPLICABLE}
            )
        ):
            raise ValueError(
                "locator infrastructure invalidity requires applicable matching gold"
            )
        if not infra and (self.case_ref is None or self.case_ref.case_id != self.case_id or self.gold_disposition is None):
            raise ValueError("valid decisions require matching case identity and gold")
        scorable = self.gold_disposition is GoldDisposition.SCORABLE
        if self.localization_denominator_eligible != scorable:
            raise ValueError("localization denominator membership must follow gold disposition")
        answered = scorable and self.decision_status in {StaticAttributionDecisionStatus.MATCHED, StaticAttributionDecisionStatus.UNMATCHED}
        if self.answered_denominator_eligible != answered:
            raise ValueError("answered denominator requires a valid completed scorable output")
        matched = self.decision_status is StaticAttributionDecisionStatus.MATCHED
        if matched != (self.first_match_rank is not None):
            raise ValueError("only matched decisions carry a first match")
        expected_rr = 1.0 / self.first_match_rank if matched else 0.0
        endpoints = (self.hit_at_1, self.hit_at_3, self.fixed_denominator_reciprocal_rank)
        if scorable and not infra:
            if endpoints != (matched and self.first_match_rank <= 1, matched and self.first_match_rank <= 3, expected_rr):
                raise ValueError("fixed-denominator endpoints must follow the first exact match")
        elif endpoints != (None, None, None):
            raise ValueError("excluded or incomplete endpoints must be null")
        if self.answered_only_reciprocal_rank != (expected_rr if answered else None):
            raise ValueError("answered-only reciprocal rank must follow answered eligibility")
        identifiable_decisions = {StaticAttributionDecisionStatus.MATCHED, StaticAttributionDecisionStatus.UNMATCHED, StaticAttributionDecisionStatus.ABSTAINED}
        if self.decision_status in identifiable_decisions and not scorable:
            raise ValueError("localization decisions require scorable gold")
        if self.decision_status is StaticAttributionDecisionStatus.ABSTAINED and self.locator_status is not LocatorStatus.ABSTAINED:
            raise ValueError("abstained decision requires locator abstention")
        if answered and self.locator_status is not LocatorStatus.COMPLETED:
            raise ValueError("answered score requires a completed locator")
        unidentifiable = self.gold_disposition is GoldDisposition.UNIDENTIFIABLE
        rejection = self.decision_status is StaticAttributionDecisionStatus.CORRECT_REJECTION
        unsupported = self.decision_status is StaticAttributionDecisionStatus.UNSUPPORTED_LOCALIZATION
        if (rejection or unsupported) and not unidentifiable:
            raise ValueError("rejection and unsupported localization require unidentifiable gold")
        if rejection and self.locator_status is not LocatorStatus.ABSTAINED:
            raise ValueError("correct rejection requires locator abstention")
        if unsupported and self.locator_status is not LocatorStatus.COMPLETED:
            raise ValueError("unsupported localization requires a completed locator")
        expected_rejection = rejection if unidentifiable and not infra else None
        expected_unsupported = unsupported if unidentifiable and not infra else None
        if self.correct_rejection is not expected_rejection or self.unsupported_localization is not expected_unsupported:
            raise ValueError("unidentifiable diagnostics must follow the decision")
        if not infra and (self.decision_status is StaticAttributionDecisionStatus.NOT_APPLICABLE) != (self.gold_disposition is GoldDisposition.NOT_APPLICABLE):
            raise ValueError("not-applicable cases must retain their excluded decision")

    def to_dict(self) -> dict[str, object]:
        """Serialize every score field while preserving null evidence and raw costs."""

        return {
            "schema": self.schema,
            "implementation_status": self.implementation_status,
            "input_binding_status": self.input_binding_status,
            "case_id": self.case_id,
            "case_ref": self.case_ref.to_dict() if self.case_ref is not None else None,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "registry_id": self.registry_id,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "implementation_key": self.implementation_key.value,
            "track": self.track.value,
            "method_id": self.method_id,
            "locator_status": self.locator_status.value,
            "locator_failure_domain": self.locator_failure_domain.value if self.locator_failure_domain is not None else None,
            "locator_failure_kind": self.locator_failure_kind.value if self.locator_failure_kind is not None else None,
            "gold_disposition": self.gold_disposition.value if self.gold_disposition is not None else None,
            "decision_status": self.decision_status.value,
            "invalid_reason": self.invalid_reason.value if self.invalid_reason is not None else None,
            "finding_kinds": [kind.value for kind in self.finding_kinds],
            "usage": self.usage.to_dict(),
            "localization_denominator_eligible": self.localization_denominator_eligible,
            "answered_denominator_eligible": self.answered_denominator_eligible,
            "first_match_rank": self.first_match_rank,
            "matched_event_group": list(self.matched_event_group) if self.matched_event_group is not None else None,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "fixed_denominator_reciprocal_rank": self.fixed_denominator_reciprocal_rank,
            "answered_only_reciprocal_rank": self.answered_only_reciprocal_rank,
            "correct_rejection": self.correct_rejection,
            "unsupported_localization": self.unsupported_localization,
            "method_failure": self.method_failure,
            "infrastructure_failure": self.infrastructure_failure,
            "incomplete": self.incomplete,
        }


def score_static_attribution_case_v1(
    public_case: PublicTracePacketV1,
    gold: AttributionGoldV1 | None,
    locator_receipt: LocatorReceiptV1,
    registry: ScorerRegistryV1,
) -> StaticAttributionCaseScoreV1:
    """Compare whole event sets; keep unknown infrastructure outcomes out of zeros.

    Malformed typed inputs raise ValueError for the caller's scorer-exception
    receipt. Known identity, oracle, and infrastructure failures return explicit
    incomplete records. A runner must still bind this call to a registered work
    item, dataset admission, and budget before consuming it in a formal result.
    """

    if not isinstance(public_case, PublicTracePacketV1) or not isinstance(locator_receipt, LocatorReceiptV1) or not isinstance(registry, ScorerRegistryV1):
        raise ValueError("static scorer requires typed public packet, locator receipt, and registry")
    if gold is not None and not isinstance(gold, AttributionGoldV1):
        raise ValueError("static scorer gold must be typed or null")
    if not isinstance(locator_receipt.status, LocatorStatus):
        raise ValueError("locator status must use its closed enum")
    kinds = tuple(prediction.kind for prediction in locator_receipt.predictions)
    if any(not isinstance(kind, TraceFindingKind) for kind in kinds):
        raise ValueError("locator finding kinds must use the closed enum")
    registration = next(entry for entry in registry.scorers if entry.track is TraceFixTrack.STATIC_ATTRIBUTION)
    reason = None
    if public_case.track is not TraceFixTrack.STATIC_ATTRIBUTION or public_case.execution_semantics is not ExecutionSemantics.FROZEN_TRACE:
        reason = StaticAttributionInvalidReason.TRACK_MISMATCH
    elif locator_receipt.request != public_case:
        reason = StaticAttributionInvalidReason.REQUEST_MISMATCH
    elif gold is None:
        reason = StaticAttributionInvalidReason.GOLD_MISSING
    elif gold.case_ref.case_id != public_case.case_id:
        reason = StaticAttributionInvalidReason.GOLD_CASE_MISMATCH
    elif not isinstance(gold.authority, GoldAuthority) or gold.reviewers_saw_method_outputs:
        reason = StaticAttributionInvalidReason.GOLD_NOT_INDEPENDENT
    elif any(not set(group) <= public_case.event_ids for group in gold.accepted_event_groups):
        reason = StaticAttributionInvalidReason.GOLD_EVENT_OUTSIDE_ROSTER
    elif any(not set(prediction.event_ids) <= public_case.event_ids for prediction in locator_receipt.predictions):
        reason = StaticAttributionInvalidReason.PREDICTION_EVENT_OUTSIDE_ROSTER
    disposition = gold.disposition if gold is not None else None
    rank = None
    matched_group = None
    if reason is not None:
        decision = StaticAttributionDecisionStatus.INFRASTRUCTURE_INVALID
    elif disposition is GoldDisposition.NOT_APPLICABLE:
        decision = StaticAttributionDecisionStatus.NOT_APPLICABLE
    elif locator_receipt.failure_domain is TraceFailureDomain.INFRASTRUCTURE:
        decision = StaticAttributionDecisionStatus.INFRASTRUCTURE_INVALID
        reason = StaticAttributionInvalidReason.LOCATOR_INFRASTRUCTURE_FAILURE
    elif locator_receipt.failure_domain is TraceFailureDomain.METHOD:
        decision = StaticAttributionDecisionStatus.METHOD_FAILURE
        reason = StaticAttributionInvalidReason.LOCATOR_METHOD_FAILURE
    elif any(kind not in registry.allowed_finding_kinds for kind in kinds):
        decision = StaticAttributionDecisionStatus.METHOD_FAILURE
        reason = StaticAttributionInvalidReason.FINDING_KIND_NOT_REGISTERED
    elif disposition is GoldDisposition.UNIDENTIFIABLE:
        decision = (StaticAttributionDecisionStatus.CORRECT_REJECTION if locator_receipt.status is LocatorStatus.ABSTAINED else StaticAttributionDecisionStatus.UNSUPPORTED_LOCALIZATION)
    elif locator_receipt.status is LocatorStatus.ABSTAINED:
        decision = StaticAttributionDecisionStatus.ABSTAINED
    else:
        accepted = {frozenset(group) for group in gold.accepted_event_groups}
        # Each rank is one conjunctive group; overlap and containment earn no credit.
        match = next((prediction for prediction in locator_receipt.predictions if frozenset(prediction.event_ids) in accepted), None)
        decision = StaticAttributionDecisionStatus.UNMATCHED
        if match is not None:
            decision = StaticAttributionDecisionStatus.MATCHED
            rank, matched_group = match.rank, match.event_ids

    infrastructure = decision is StaticAttributionDecisionStatus.INFRASTRUCTURE_INVALID
    scorable = disposition is GoldDisposition.SCORABLE
    answered = scorable and decision in {StaticAttributionDecisionStatus.MATCHED, StaticAttributionDecisionStatus.UNMATCHED}
    rr = 1.0 / rank if rank is not None else 0.0
    available = scorable and not infrastructure
    unidentifiable_available = disposition is GoldDisposition.UNIDENTIFIABLE and not infrastructure
    return StaticAttributionCaseScoreV1(
        case_id=public_case.case_id, case_ref=gold.case_ref if gold is not None else None,
        suite_id=registry.suite_id, suite_version=registry.suite_version,
        population_id=registry.population_id, registry_id=registry.registry_id,
        scorer_id=registration.scorer_id, scorer_version=registration.scorer_version,
        method_id=locator_receipt.method_id, locator_status=locator_receipt.status,
        locator_failure_domain=locator_receipt.failure_domain,
        locator_failure_kind=locator_receipt.failure_kind,
        gold_disposition=disposition, decision_status=decision, invalid_reason=reason,
        finding_kinds=kinds, usage=locator_receipt.usage,
        localization_denominator_eligible=scorable, answered_denominator_eligible=answered,
        first_match_rank=rank, matched_event_group=matched_group,
        hit_at_1=(rank is not None and rank <= 1) if available else None,
        hit_at_3=(rank is not None and rank <= 3) if available else None,
        fixed_denominator_reciprocal_rank=rr if available else None,
        answered_only_reciprocal_rank=rr if answered else None,
        correct_rejection=(decision is StaticAttributionDecisionStatus.CORRECT_REJECTION) if unidentifiable_available else None,
        unsupported_localization=(decision is StaticAttributionDecisionStatus.UNSUPPORTED_LOCALIZATION) if unidentifiable_available else None,
        method_failure=decision is StaticAttributionDecisionStatus.METHOD_FAILURE,
        infrastructure_failure=infrastructure, incomplete=infrastructure,
    )


__all__ = [
    "STATIC_ATTRIBUTION_CASE_SCORE_SCHEMA", "STATIC_CASE_SCORE_BINDING_STATUS",
    "STATIC_SCORER_IMPLEMENTATION_STATUS",
    "StaticAttributionCaseScoreV1", "StaticAttributionDecisionStatus",
    "StaticAttributionInvalidReason", "score_static_attribution_case_v1",
]
