"""Pure hash-binding functions shared by replay authorities and the runner."""

from __future__ import annotations

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.types import (
    CounterfactualReplayEvidenceV1,
    ReplayOutcomeV1,
    TraceCaseV1,
)


def _sha256(value: object) -> str:
    return "sha256:" + canonical_sha256(value)


def replay_outcome_sha256(
    *,
    controller_id: str,
    case_id: str,
    arm: str,
    source_snapshot_sha256: str,
    outcome: ReplayOutcomeV1,
) -> str:
    return _sha256(
        {
            "controller_id": controller_id,
            "case_id": case_id,
            "arm": arm,
            "source_snapshot_sha256": source_snapshot_sha256,
            "restored_state_sha256": outcome.restored_state_sha256,
            "target_event_id": outcome.target_event_id,
            "input_content_sha256": outcome.input_content_sha256,
            "terminal_state_sha256": outcome.terminal_state_sha256,
            "success": outcome.success,
            "terminal_reason": outcome.terminal_reason,
        }
    )


def restore_receipt_sha256(
    *,
    case: TraceCaseV1,
    evidence: CounterfactualReplayEvidenceV1,
) -> str:
    return _sha256(
        {
            "controller_id": evidence.controller_id,
            "case_id": case.case_id,
            "source_trace_sha256": evidence.source_trace_sha256,
            "source_snapshot_sha256": evidence.source_snapshot_sha256,
            "baseline_restored_state_sha256": evidence.baseline.restored_state_sha256,
            "repair_restored_state_sha256": evidence.repair.restored_state_sha256,
            "sham_restored_state_sha256": evidence.sham.restored_state_sha256,
            "restore_verified": evidence.restore_verified,
        }
    )
