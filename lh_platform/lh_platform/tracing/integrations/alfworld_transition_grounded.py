"""Unified adapter for transition-grounded ALFWorld causal traceback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlparse

from lh_platform.tracing.alfworld_transition_grounded_traceback import (
    build_alfworld_transition_grounded_proposal,
    validate_alfworld_transition_grounded_proposal,
)
from lh_platform.tracing.trace_observability import TraceSymbolicStateEvidenceV1
from lh_platform.tracing.transition_dependency import TransitionCandidateReplayBackend
from lh_platform.tracing.types import EvidenceLevel, TraceCaseV1
from lh_platform.tracing.unified import (
    TraceDiagnosticEventV1,
    TraceDiagnosticStage,
    TraceDiagnosticStatus,
    TraceFindingKind,
    TraceFindingV1,
    TraceInputArtifactV1,
    TraceMethodDescriptorV1,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
)


def _load_json_artifact(artifact: TraceInputArtifactV1) -> Mapping[str, object]:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("transition-grounded evidence must use a local file URI")
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_file():
        raise ValueError(f"transition-grounded evidence is missing: {path}")
    raw = path.read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != artifact.content_sha256:
        raise ValueError("transition-grounded evidence artifact hash mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("transition-grounded evidence must be a JSON object")
    return payload


def _exact_artifact(
    request: TraceMethodRequestV1,
    modality: str,
) -> TraceInputArtifactV1:
    matches = tuple(artifact for artifact in request.artifacts if artifact.modality == modality)
    if len(matches) != 1:
        raise ValueError(f"transition-grounded tracing requires exactly one {modality} artifact")
    return matches[0]


class AlfWorldTransitionGroundedTraceAdapter:
    """Propose state-grounded events and verify them with bounded replay."""

    METHOD_ID = "alfworld.transition_grounded.counterfactual_v1"

    def __init__(
        self,
        *,
        replay_backend: TransitionCandidateReplayBackend,
        max_intervention_trials: int,
    ) -> None:
        if max_intervention_trials < 1:
            raise ValueError("transition-grounded replay budget must be positive")
        self._replay_backend = replay_backend
        self._max_intervention_trials = max_intervention_trials
        self.descriptor = TraceMethodDescriptorV1(
            method_id=self.METHOD_ID,
            display_name="ALFWorld Transition-Grounded Counterfactual",
            family="failure_attribution",
            capabilities=(
                "step_attribution",
                "expected_observed_transition_evidence",
                "bounded_counterfactual_replay",
                "intervention_supported_cause_and_repair_sites",
                "abstention",
            ),
            input_modalities=(
                "text_trajectory",
                "symbolic_state_evidence",
                "benchmark_precondition_failure_evidence",
            ),
            output_kinds=(
                TraceFindingKind.STEP_ATTRIBUTION,
                TraceFindingKind.ROOT_CAUSE,
                TraceFindingKind.RECOVERY_PROPOSAL,
            ),
            integration_level="platform_native",
            execution_path=("expected_effect_debt_v2_then_bounded_three_arm_replay"),
            evidence_ceiling=EvidenceLevel.L4_COUNTERFACTUAL,
            source_relpath=("lh_platform/tracing/alfworld_transition_grounded_traceback.py"),
            source_revision="v1",
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        if not isinstance(request.case, TraceCaseV1):
            raise ValueError("transition-grounded tracing requires TraceCaseV1")
        symbolic_artifact = _exact_artifact(request, "symbolic_state_evidence")
        failure_artifact = _exact_artifact(
            request,
            "benchmark_precondition_failure_evidence",
        )
        symbolic = TraceSymbolicStateEvidenceV1.from_dict(_load_json_artifact(symbolic_artifact))
        failure = _load_json_artifact(failure_artifact)
        proposal = build_alfworld_transition_grounded_proposal(
            request.case,
            symbolic,
            failure,
        )
        validation = validate_alfworld_transition_grounded_proposal(
            request.case,
            proposal,
            self._replay_backend,
            max_intervention_trials=self._max_intervention_trials,
        )
        common_attributes = (
            ("proposal_sha256", proposal.sha256),
            ("validation_sha256", validation.sha256),
            ("symbolic_artifact_sha256", symbolic_artifact.content_sha256),
            ("failure_artifact_sha256", failure_artifact.content_sha256),
            ("controller_id", validation.controller_id),
            ("candidate_set_exhausted", str(validation.candidate_set_exhausted).lower()),
            ("intervention_trial_count", str(validation.intervention_trial_count)),
            ("replay_call_count", str(validation.replay_call_count)),
            (
                "causal_validation_status",
                ",".join(validation.causal_validation_blockers) or "verified",
            ),
        )
        findings: list[TraceFindingV1] = [
            TraceFindingV1(
                kind=TraceFindingKind.STEP_ATTRIBUTION,
                event_ids=proposal.candidate_event_ids,
                label="transition_grounded_candidate_set",
                score=None,
                explanation=(
                    "Candidates are ordered from task-conditioned expected-versus-"
                    "observed state evidence before causal gold is opened."
                ),
                attributes=common_attributes,
            )
        ]
        for cause in validation.verified_causes:
            cause_attributes = (
                *common_attributes,
                (
                    "claim_scope",
                    "transition_violation_and_intervention_supported_within_frozen_candidate_family",
                ),
                ("site_role", "transition_violation_cause"),
                ("replacement_action", cause.replacement_action),
                (
                    "intervention_receipt_sha256",
                    cause.intervention_receipt_sha256,
                ),
            )
            if cause.arm_identity_evidence_sha256 is not None:
                cause_attributes = (
                    *cause_attributes,
                    (
                        "arm_identity_evidence_sha256",
                        cause.arm_identity_evidence_sha256,
                    ),
                )
            findings.extend(
                (
                    TraceFindingV1(
                        kind=TraceFindingKind.ROOT_CAUSE,
                        event_ids=(cause.event_id,),
                        label="intervention_supported_cause",
                        score=1.0,
                        explanation=(
                            "Exact-state replay reproduced the fault, accepted this "
                            "replacement, and rejected the matched sham."
                        ),
                        attributes=cause_attributes,
                    ),
                    TraceFindingV1(
                        kind=TraceFindingKind.RECOVERY_PROPOSAL,
                        event_ids=(cause.event_id,),
                        label="replay_verified_replacement_action",
                        score=1.0,
                        explanation=cause.replacement_action,
                        attributes=cause_attributes,
                    ),
                )
            )
        for site in validation.compensating_repair_sites:
            site_attributes = (
                *common_attributes,
                ("claim_scope", "repair_only_not_fault_localization"),
                ("site_role", site.site_role),
                ("replacement_action", site.replacement_action),
                (
                    "intervention_receipt_sha256",
                    site.intervention_receipt_sha256,
                ),
            )
            if site.arm_identity_evidence_sha256 is not None:
                site_attributes = (
                    *site_attributes,
                    (
                        "arm_identity_evidence_sha256",
                        site.arm_identity_evidence_sha256,
                    ),
                )
            findings.append(
                TraceFindingV1(
                    kind=TraceFindingKind.RECOVERY_PROPOSAL,
                    event_ids=(site.event_id,),
                    label="counterfactual_compensating_repair",
                    score=1.0,
                    explanation=site.replacement_action,
                    attributes=site_attributes,
                )
            )
        return TraceMethodOutputV1(
            findings=tuple(findings),
            evidence_level=validation.evidence_level,
            diagnostics=(
                TraceDiagnosticEventV1(
                    stage=TraceDiagnosticStage.ADAPTER_VALIDATION,
                    status=TraceDiagnosticStatus.COMPLETED,
                    attributes=(
                        ("proposal_sha256", proposal.sha256),
                        ("validation_sha256", validation.sha256),
                    ),
                ),
            ),
        )


__all__ = ["AlfWorldTransitionGroundedTraceAdapter"]
