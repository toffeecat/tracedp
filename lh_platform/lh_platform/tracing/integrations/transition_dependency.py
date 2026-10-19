"""Unified adapter for transition-dependency traceback and replay validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlparse

from lh_platform.tracing.trace_observability import TraceSymbolicStateEvidenceV1
from lh_platform.tracing.transition_dependency import (
    TransitionCandidateReplayBackend,
    propose_transition_dependency_slice,
    validate_transition_candidates,
)
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


def _load_symbolic_evidence(
    artifact: TraceInputArtifactV1,
) -> TraceSymbolicStateEvidenceV1:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("symbolic state evidence must use a local file URI")
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_file():
        raise ValueError(f"symbolic state evidence is missing: {path}")
    raw = path.read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != artifact.content_sha256:
        raise ValueError("symbolic state evidence artifact hash mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("symbolic state evidence must be a JSON object")
    return TraceSymbolicStateEvidenceV1.from_dict(payload)


class TransitionDependencyTraceAdapter:
    """Propose a state dependency slice, then validate ranked action candidates."""

    METHOD_ID = "transition_dependency.counterfactual"

    def __init__(self, *, replay_backend: TransitionCandidateReplayBackend) -> None:
        self._replay_backend = replay_backend
        self.descriptor = TraceMethodDescriptorV1(
            method_id=self.METHOD_ID,
            display_name="Transition Dependency Counterfactual",
            family="failure_attribution",
            capabilities=(
                "step_attribution",
                "transition_dependency",
                "counterfactual_replay",
            ),
            input_modalities=("text_trajectory", "symbolic_state_evidence"),
            output_kinds=(
                TraceFindingKind.STEP_ATTRIBUTION,
                TraceFindingKind.ROOT_CAUSE,
            ),
            integration_level="platform_native",
            execution_path="transition_dependency_then_three_arm_replay",
            evidence_ceiling=EvidenceLevel.L4_COUNTERFACTUAL,
            source_relpath="lh_platform/tracing/transition_dependency.py",
            source_revision="v1",
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        if not isinstance(request.case, TraceCaseV1):
            raise ValueError("transition dependency tracing requires TraceCaseV1")
        artifacts = tuple(
            artifact
            for artifact in request.artifacts
            if artifact.modality == "symbolic_state_evidence"
        )
        if len(artifacts) != 1:
            raise ValueError(
                "transition dependency tracing requires exactly one symbolic_state_evidence artifact"
            )
        evidence = _load_symbolic_evidence(artifacts[0])
        proposal = propose_transition_dependency_slice(request.case, evidence)
        validation = validate_transition_candidates(
            request.case,
            proposal,
            self._replay_backend,
        )
        common_attributes = (
            ("proposal_sha256", proposal.sha256),
            ("symbolic_artifact_sha256", artifacts[0].content_sha256),
            ("controller_id", validation.controller_id),
            ("intervention_trial_count", str(validation.intervention_trial_count)),
            ("replay_call_count", str(validation.replay_call_count)),
        )
        if validation.verified_event_id is None:
            findings = (
                TraceFindingV1(
                    kind=TraceFindingKind.STEP_ATTRIBUTION,
                    event_ids=proposal.candidate_event_ids,
                    label=proposal.violated_precondition,
                    score=None,
                    explanation=proposal.explanation,
                    attributes=(
                        *common_attributes,
                        (
                            "causal_validation_status",
                            ",".join(validation.causal_validation_blockers),
                        ),
                    ),
                ),
            )
        else:
            if validation.intervention_receipt is None:
                raise ValueError("verified transition is missing its replay receipt")
            verified_attributes = [
                *common_attributes,
                ("causal_validation_status", "verified"),
                (
                    "intervention_receipt_sha256",
                    validation.intervention_receipt.receipt_sha256,
                ),
            ]
            if validation.arm_identity_evidence_sha256 is not None:
                verified_attributes.append(
                    (
                        "arm_identity_evidence_sha256",
                        validation.arm_identity_evidence_sha256,
                    )
                )
            findings = (
                TraceFindingV1(
                    kind=TraceFindingKind.STEP_ATTRIBUTION,
                    event_ids=(validation.verified_event_id,),
                    label=proposal.violated_precondition,
                    score=1.0,
                    explanation=proposal.explanation,
                    attributes=tuple(verified_attributes),
                ),
                TraceFindingV1(
                    kind=TraceFindingKind.ROOT_CAUSE,
                    event_ids=(validation.verified_event_id,),
                    label=proposal.violated_precondition,
                    score=1.0,
                    explanation=(
                        "Three-arm replay verified that replacing this event repairs "
                        "the failure while a matched sham does not."
                    ),
                    attributes=tuple(verified_attributes),
                ),
            )
        return TraceMethodOutputV1(
            findings=findings,
            evidence_level=validation.evidence_level,
            diagnostics=(
                TraceDiagnosticEventV1(
                    stage=TraceDiagnosticStage.ADAPTER_VALIDATION,
                    status=TraceDiagnosticStatus.COMPLETED,
                    attributes=(
                        ("proposal_sha256", proposal.sha256),
                        ("controller_id", validation.controller_id),
                    ),
                ),
            ),
        )


__all__ = ["TransitionDependencyTraceAdapter"]
