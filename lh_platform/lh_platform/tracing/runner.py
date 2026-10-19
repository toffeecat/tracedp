"""Public traceback investigation seam."""

from __future__ import annotations

import re
from typing import Protocol

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.adapters import InvestigationDraft, TraceInvestigationError
from lh_platform.tracing.replay_receipts import (
    replay_outcome_sha256,
    restore_receipt_sha256,
)
from lh_platform.tracing.types import (
    CounterfactualReplayEvidenceV1,
    EvidenceLevel,
    InterventionReceiptV1,
    InvestigationReceipt,
    TraceCaseV1,
    TraceInterventionV1,
    TraceUsageV1,
)


class TraceAdapter(Protocol):
    def investigate(self, case: TraceCaseV1) -> InvestigationDraft: ...


class ReplayController(Protocol):
    controller_id: str

    def evaluate(
        self,
        case: TraceCaseV1,
        intervention: TraceInterventionV1,
    ) -> CounterfactualReplayEvidenceV1: ...


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def build_intervention_receipt(
    *,
    case: TraceCaseV1,
    intervention: TraceInterventionV1,
    evidence: CounterfactualReplayEvidenceV1,
) -> InterventionReceiptV1:
    """Validate raw three-arm evidence and bind it to one intervention receipt."""

    return TraceInvestigationRunner._intervention_receipt(
        case=case,
        intervention=intervention,
        evidence=evidence,
    )


class TraceInvestigationRunner:
    """Run one tracer and bind its claim to the exact source trace."""

    def __init__(
        self,
        tracer: TraceAdapter,
        *,
        replay_controller: ReplayController | None = None,
    ) -> None:
        self._tracer = tracer
        self._replay_controller = replay_controller

    def run(self, case: TraceCaseV1) -> InvestigationReceipt:
        draft = self._tracer.investigate(case)
        intervention_receipt: InterventionReceiptV1 | None = None
        usage = draft.usage
        if draft.replay_required:
            if self._replay_controller is None:
                raise TraceInvestigationError(
                    "counterfactual tracer requires a benchmark-owned replay controller"
                )
            intervention = draft.claim.intervention
            if intervention is None:
                raise TraceInvestigationError(
                    "counterfactual replay requires a proposed intervention"
                )
            evidence = self._replay_controller.evaluate(case, intervention)
            if evidence.controller_id != self._replay_controller.controller_id:
                raise TraceInvestigationError("replay controller identity mismatch")
            intervention_receipt = build_intervention_receipt(
                case=case,
                intervention=intervention,
                evidence=evidence,
            )
            usage = TraceUsageV1(
                model_calls=draft.usage.model_calls,
                prompt_tokens=draft.usage.prompt_tokens,
                completion_tokens=draft.usage.completion_tokens,
                replay_calls=evidence.replay_calls,
            )
        blockers = self._causal_validation_blockers(case, draft, intervention_receipt)
        root_cause_verified = not blockers
        evidence_level = (
            EvidenceLevel.L4_COUNTERFACTUAL if root_cause_verified else draft.evidence_level
        )
        unhashed = InvestigationReceipt(
            case_id=case.case_id,
            tracer_id=draft.tracer_id,
            tracer_version=draft.tracer_version,
            source_trace_sha256=case.sha256,
            claim=draft.claim,
            evidence_level=evidence_level,
            root_cause_verified=root_cause_verified,
            intervention_receipt=intervention_receipt,
            usage=usage,
            diagnostic_output_sha256=draft.diagnostic_output_sha256,
            causal_validation_blockers=blockers,
            receipt_sha256="",
        )
        digest = "sha256:" + canonical_sha256(unhashed.to_dict(include_sha256=False))
        return InvestigationReceipt(
            case_id=unhashed.case_id,
            tracer_id=unhashed.tracer_id,
            tracer_version=unhashed.tracer_version,
            source_trace_sha256=unhashed.source_trace_sha256,
            claim=unhashed.claim,
            evidence_level=unhashed.evidence_level,
            root_cause_verified=unhashed.root_cause_verified,
            intervention_receipt=unhashed.intervention_receipt,
            usage=unhashed.usage,
            diagnostic_output_sha256=unhashed.diagnostic_output_sha256,
            causal_validation_blockers=unhashed.causal_validation_blockers,
            receipt_sha256=digest,
        )

    @staticmethod
    def _intervention_receipt(
        *,
        case: TraceCaseV1,
        intervention: TraceInterventionV1,
        evidence: CounterfactualReplayEvidenceV1,
    ) -> InterventionReceiptV1:
        if not evidence.controller_id:
            raise TraceInvestigationError("replay controller_id must be non-empty")
        if evidence.replay_calls != 3:
            raise TraceInvestigationError(
                "counterfactual evidence requires exactly baseline, repair and sham calls"
            )
        if evidence.source_trace_sha256 != case.sha256:
            raise TraceInvestigationError("replay evidence source trace hash mismatch")
        digests = [
            evidence.source_snapshot_sha256,
            evidence.restore_receipt_sha256,
        ]
        for outcome in (evidence.baseline, evidence.repair, evidence.sham):
            digests.extend(
                [
                    outcome.input_content_sha256,
                    outcome.restored_state_sha256,
                    outcome.terminal_state_sha256,
                    outcome.outcome_receipt_sha256,
                ]
            )
            if outcome.target_event_id not in case.event_ids:
                raise TraceInvestigationError("replay outcome target must reference the trace")
            expected_reason = "verifier_accepted" if outcome.success else "verifier_rejected"
            if outcome.terminal_reason != expected_reason:
                raise TraceInvestigationError("replay outcome terminal reason is inconsistent")
        if any(not _SHA256_RE.fullmatch(value) for value in digests):
            raise TraceInvestigationError("counterfactual receipts require sha256-bound evidence")

        for arm, outcome in (
            ("fault_baseline", evidence.baseline),
            ("predicted_repair", evidence.repair),
            ("matched_sham", evidence.sham),
        ):
            expected = replay_outcome_sha256(
                controller_id=evidence.controller_id,
                case_id=case.case_id,
                arm=arm,
                source_snapshot_sha256=evidence.source_snapshot_sha256,
                outcome=outcome,
            )
            if outcome.outcome_receipt_sha256 != expected:
                raise TraceInvestigationError(f"{arm} outcome receipt hash mismatch")

        restore_verified = all(
            outcome.restored_state_sha256 == evidence.source_snapshot_sha256
            for outcome in (evidence.baseline, evidence.repair, evidence.sham)
        )
        if evidence.restore_verified != restore_verified:
            raise TraceInvestigationError("replay restore verification is inconsistent")
        if evidence.restore_receipt_sha256 != restore_receipt_sha256(
            case=case,
            evidence=evidence,
        ):
            raise TraceInvestigationError("replay restore receipt hash mismatch")

        replacement_sha256 = "sha256:" + canonical_sha256(intervention.replacement_content)
        unhashed = InterventionReceiptV1(
            controller_id=evidence.controller_id,
            source_trace_sha256=case.sha256,
            target_event_id=intervention.target_event_id,
            replacement_content_sha256=replacement_sha256,
            source_snapshot_sha256=evidence.source_snapshot_sha256,
            restore_verified=restore_verified,
            restore_receipt_sha256=evidence.restore_receipt_sha256,
            baseline=evidence.baseline,
            repair=evidence.repair,
            sham=evidence.sham,
            receipt_sha256="",
        )
        return InterventionReceiptV1(
            controller_id=unhashed.controller_id,
            source_trace_sha256=unhashed.source_trace_sha256,
            target_event_id=unhashed.target_event_id,
            replacement_content_sha256=unhashed.replacement_content_sha256,
            source_snapshot_sha256=unhashed.source_snapshot_sha256,
            restore_verified=unhashed.restore_verified,
            restore_receipt_sha256=unhashed.restore_receipt_sha256,
            baseline=unhashed.baseline,
            repair=unhashed.repair,
            sham=unhashed.sham,
            receipt_sha256="sha256:" + canonical_sha256(unhashed.to_dict(include_sha256=False)),
        )

    @staticmethod
    def _causal_validation_blockers(
        case: TraceCaseV1,
        draft: InvestigationDraft,
        receipt: InterventionReceiptV1 | None,
    ) -> tuple[str, ...]:
        intervention = draft.claim.intervention
        if receipt is None:
            return ("counterfactual_replay_not_run",)

        blockers: list[str] = []
        if intervention is None:
            blockers.append("intervention_missing")
        else:
            if intervention.target_event_id not in draft.claim.causal_event_ids:
                blockers.append("intervention_target_not_claimed_as_causal")
            if receipt.target_event_id != intervention.target_event_id:
                blockers.append("intervention_target_binding_mismatch")
            replacement_sha256 = "sha256:" + canonical_sha256(intervention.replacement_content)
            if receipt.replacement_content_sha256 != replacement_sha256:
                blockers.append("intervention_replacement_binding_mismatch")
            if receipt.repair.input_content_sha256 != replacement_sha256:
                blockers.append("repair_input_binding_mismatch")
            if receipt.repair.target_event_id != intervention.target_event_id:
                blockers.append("repair_target_binding_mismatch")
        if receipt.source_trace_sha256 != case.sha256:
            blockers.append("source_trace_binding_mismatch")
        expected_receipt_sha256 = "sha256:" + canonical_sha256(
            receipt.to_dict(include_sha256=False)
        )
        if receipt.receipt_sha256 != expected_receipt_sha256:
            blockers.append("intervention_receipt_hash_mismatch")
        if not receipt.restore_verified:
            blockers.append("restore_not_verified")
        if receipt.baseline.success:
            blockers.append("fault_baseline_did_not_reproduce_failure")
        if not receipt.repair.success:
            blockers.append("repair_replay_failed")
        if receipt.sham.success:
            blockers.append("sham_also_succeeded")
        if receipt.sham.input_content_sha256 == receipt.baseline.input_content_sha256:
            blockers.append("sham_not_distinct_from_fault_baseline")
        if receipt.sham.target_event_id != receipt.repair.target_event_id:
            blockers.append("sham_target_not_matched_to_repair")
        return tuple(sorted(set(blockers)))
