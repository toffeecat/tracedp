"""Unified seam for heterogeneous trajectory tracing methods.

The public module deliberately exposes only catalog discovery and one run call.
Upstream source probing, process isolation, format conversion and evidence
normalization belong behind adapters, not in experiment callers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from lh_platform.core.artifact_redaction import redact_text
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.types import (
    EvidenceLevel,
    TraceCaseV1,
    TraceEpisodeV1,
    TraceInputV1,
    trace_input_from_dict,
)

TRACE_METHOD_RUN_SCHEMA = "lh_trace_method_run/v1"
TRACE_METHOD_RUN_V2_SCHEMA = "lh_trace_method_run/v2"
TRACE_METHOD_REQUEST_SCHEMA = "lh_trace_method_request/v1"
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROVIDER_ACCOUNT_HALT_FRAGMENTS = (
    "creditserror",
    "insufficient balance",
    "quota exceeded",
    "已达到 token plan 用量上限",
    "购买积分补充用量",
    '"code":2056',
    "(2056)",
    "token plan 速率限制",
    "已达到 token plan 速率限制",
    '"code":2062',
    "(2062)",
)


class TraceFindingKind(StrEnum):
    ROOT_CAUSE = "root_cause"
    STEP_ATTRIBUTION = "step_attribution"
    COMPONENT_ATTRIBUTION = "component_attribution"
    TRAJECTORY_RISK = "trajectory_risk"
    ERROR_CLASSIFICATION = "error_classification"
    RECOVERY_PROPOSAL = "recovery_proposal"
    PROCESS_SCORE = "process_score"


class TraceRunStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    INVALID = "invalid"


class TraceOutcomeScope(StrEnum):
    OBSERVED_FAILURE = "observed_failure"
    ANY_TERMINAL_OUTCOME = "any_terminal_outcome"


class TraceDiagnosticStage(StrEnum):
    ADMISSION = "admission"
    ADAPTER_EXECUTION = "adapter_execution"
    WORKER_LAUNCH = "worker_launch"
    MODEL_TRANSPORT = "model_transport"
    MODEL_RESPONSE = "model_response"
    UPSTREAM_PARSE = "upstream_parse"
    UPSTREAM_VALIDATION = "upstream_validation"
    ARTIFACT_PUBLICATION = "artifact_publication"
    ADAPTER_VALIDATION = "adapter_validation"
    PLATFORM_VALIDATION = "platform_validation"


class TraceDiagnosticStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class TraceFailureDomain(StrEnum):
    """Separate method outcomes from failures of the evaluation apparatus."""

    METHOD = "method"
    INFRASTRUCTURE = "infrastructure"


class TraceFailureKind(StrEnum):
    """Stable failure identities used by runners, receipts, and scorers."""

    METHOD_BUDGET_EXHAUSTED = "method_budget_exhausted"
    METHOD_OUTPUT_INVALID = "method_output_invalid"
    REDACTION_BLOCKED = "redaction_blocked"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    WORKER_EXECUTION_FAILED = "worker_execution_failed"


@dataclass(frozen=True)
class TraceInputArtifactV1:
    """Hash-bound, case-specific input consumed by non-text tracing methods."""

    artifact_id: str
    modality: str
    uri: str
    content_sha256: str
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.modality or not self.uri:
            raise ValueError("trace input artifact identity must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.content_sha256):
            raise ValueError("trace input artifact requires a canonical sha256 digest")
        keys = [key for key, _ in self.metadata]
        if len(keys) != len(set(keys)):
            raise ValueError("trace input artifact metadata keys must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "modality": self.modality,
            "uri": self.uri,
            "content_sha256": self.content_sha256,
            "metadata": {key: value for key, value in self.metadata},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TraceInputArtifactV1:
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_metadata.items()
        ):
            raise ValueError("trace input artifact metadata must be a string mapping")
        return cls(
            artifact_id=str(payload.get("artifact_id", "")),
            modality=str(payload.get("modality", "")),
            uri=str(payload.get("uri", "")),
            content_sha256=str(payload.get("content_sha256", "")),
            metadata=tuple(sorted(raw_metadata.items())),
        )


@dataclass(frozen=True)
class TraceMethodRequestV1:
    """One immutable request shared by text, visual and signal-based tracers."""

    case: TraceInputV1
    artifacts: tuple[TraceInputArtifactV1, ...] = ()
    schema: str = TRACE_METHOD_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACE_METHOD_REQUEST_SCHEMA:
            raise ValueError(f"unsupported trace method request schema: {self.schema}")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("trace input artifact ids must be unique")

    @property
    def available_modalities(self) -> frozenset[str]:
        return frozenset(("text_trajectory", *(artifact.modality for artifact in self.artifacts)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "case": self.case.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TraceMethodRequestV1:
        raw_case = payload.get("case")
        raw_artifacts = payload.get("artifacts", [])
        if not isinstance(raw_case, Mapping):
            raise ValueError("trace method request case must be an object")
        if not isinstance(raw_artifacts, list) or not all(
            isinstance(artifact, Mapping) for artifact in raw_artifacts
        ):
            raise ValueError("trace method request artifacts must be an object list")
        return cls(
            schema=str(payload.get("schema", "")),
            case=trace_input_from_dict(raw_case),
            artifacts=tuple(TraceInputArtifactV1.from_dict(artifact) for artifact in raw_artifacts),
        )

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TraceMethodDescriptorV1:
    method_id: str
    display_name: str
    family: str
    capabilities: tuple[str, ...]
    input_modalities: tuple[str, ...]
    output_kinds: tuple[TraceFindingKind, ...]
    integration_level: str
    execution_path: str
    evidence_ceiling: EvidenceLevel
    source_relpath: str | None
    source_revision: str | None
    unavailable_reason: str | None = None
    corpus_method_id: str | None = None
    outcome_scope: TraceOutcomeScope = TraceOutcomeScope.OBSERVED_FAILURE

    def __post_init__(self) -> None:
        if not self.method_id or not self.display_name or not self.family:
            raise ValueError("trace method identity must be non-empty")
        if not self.capabilities or not self.input_modalities or not self.output_kinds:
            raise ValueError(f"trace method descriptor is incomplete: {self.method_id}")
        if not self.integration_level or not self.execution_path:
            raise ValueError(f"trace method execution descriptor is incomplete: {self.method_id}")
        if self.source_revision is not None and self.source_relpath is None:
            raise ValueError("source revision requires a source path")
        if self.corpus_method_id is None:
            object.__setattr__(self, "corpus_method_id", self.method_id)
        elif not self.corpus_method_id:
            raise ValueError("corpus method id must be non-empty")
        if not isinstance(self.outcome_scope, TraceOutcomeScope):
            raise ValueError("trace method outcome scope is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "display_name": self.display_name,
            "family": self.family,
            "capabilities": list(self.capabilities),
            "input_modalities": list(self.input_modalities),
            "output_kinds": [kind.value for kind in self.output_kinds],
            "integration_level": self.integration_level,
            "execution_path": self.execution_path,
            "evidence_ceiling": self.evidence_ceiling.value,
            "source_relpath": self.source_relpath,
            "source_revision": self.source_revision,
            "unavailable_reason": self.unavailable_reason,
            "corpus_method_id": self.corpus_method_id,
            "outcome_scope": self.outcome_scope.value,
        }


@dataclass(frozen=True)
class TraceFindingV1:
    kind: TraceFindingKind
    event_ids: tuple[str, ...]
    label: str
    score: float | None
    explanation: str
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.label or not self.explanation:
            raise ValueError("trace finding label and explanation must be non-empty")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise ValueError("trace finding score must be finite numeric or null")
        if len(self.event_ids) != len(set(self.event_ids)):
            raise ValueError("trace finding event ids must be unique")
        keys = [key for key, _ in self.attributes]
        if len(keys) != len(set(keys)):
            raise ValueError("trace finding attribute keys must be unique")
        for key, value in self.attributes:
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("trace finding attributes must contain strings")
            if key.endswith("_sha256") and not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"trace finding attribute {key} is not canonical sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "event_ids": list(self.event_ids),
            "label": self.label,
            "score": self.score,
            "explanation": self.explanation,
            "attributes": {key: value for key, value in self.attributes},
        }


@dataclass(frozen=True)
class TraceDiagnosticArtifactV1:
    """One immutable output artifact referenced by a diagnostic event."""

    artifact_id: str
    uri: str
    media_type: str
    byte_count: int
    content_sha256: str
    visibility: str = "private"

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.uri or not self.media_type:
            raise ValueError("trace diagnostic artifact identity must be non-empty")
        if isinstance(self.byte_count, bool) or self.byte_count < 0:
            raise ValueError("trace diagnostic artifact byte count must be non-negative")
        if not _SHA256_PATTERN.fullmatch(self.content_sha256):
            raise ValueError("trace diagnostic artifact requires a canonical sha256 digest")
        if self.visibility not in {"private", "public"}:
            raise ValueError("trace diagnostic artifact visibility is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "uri": self.uri,
            "media_type": self.media_type,
            "byte_count": self.byte_count,
            "content_sha256": self.content_sha256,
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TraceDiagnosticArtifactV1:
        return cls(
            artifact_id=str(payload.get("artifact_id", "")),
            uri=str(payload.get("uri", "")),
            media_type=str(payload.get("media_type", "")),
            byte_count=int(payload.get("byte_count", -1)),
            content_sha256=str(payload.get("content_sha256", "")),
            visibility=str(payload.get("visibility", "")),
        )


@dataclass(frozen=True)
class TraceDiagnosticEventV1:
    """A typed execution-stage outcome retained on success and failure paths."""

    stage: TraceDiagnosticStage
    status: TraceDiagnosticStatus
    error_type: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()
    artifacts: tuple[TraceDiagnosticArtifactV1, ...] = ()

    def __post_init__(self) -> None:
        if self.status is TraceDiagnosticStatus.FAILED and not self.error_type:
            raise ValueError("failed trace diagnostic event requires an error type")
        if self.status is TraceDiagnosticStatus.COMPLETED and self.error_type is not None:
            raise ValueError("completed trace diagnostic event cannot contain an error type")
        keys = [key for key, _ in self.attributes]
        if len(keys) != len(set(keys)):
            raise ValueError("trace diagnostic attribute keys must be unique")
        for key, value in self.attributes:
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("trace diagnostic attributes must contain strings")
            if key.endswith("_sha256") and not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"trace diagnostic attribute {key} is not canonical sha256")
        attributes = dict(self.attributes)
        failure_domain = attributes.get("failure_domain")
        failure_kind = attributes.get("failure_kind")
        if (failure_domain is None) is not (failure_kind is None):
            raise ValueError("trace failure domain and kind must be reported together")
        if failure_domain is not None:
            domain = TraceFailureDomain(failure_domain)
            kind = TraceFailureKind(failure_kind)
            allowed_kinds = {
                TraceFailureDomain.METHOD: {
                    TraceFailureKind.METHOD_BUDGET_EXHAUSTED,
                    TraceFailureKind.METHOD_OUTPUT_INVALID,
                    TraceFailureKind.REDACTION_BLOCKED,
                },
                TraceFailureDomain.INFRASTRUCTURE: {
                    TraceFailureKind.PROVIDER_RATE_LIMIT,
                    TraceFailureKind.WORKER_EXECUTION_FAILED,
                },
            }
            if kind not in allowed_kinds[domain]:
                raise ValueError("trace failure domain and kind are incompatible")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("trace diagnostic artifact ids must be unique within one event")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "error_type": self.error_type,
            "attributes": {key: value for key, value in self.attributes},
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TraceDiagnosticEventV1:
        raw_attributes = payload.get("attributes", {})
        raw_artifacts = payload.get("artifacts", [])
        if not isinstance(raw_attributes, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_attributes.items()
        ):
            raise ValueError("trace diagnostic attributes must be a string mapping")
        if not isinstance(raw_artifacts, list) or not all(
            isinstance(artifact, Mapping) for artifact in raw_artifacts
        ):
            raise ValueError("trace diagnostic artifacts must be an object list")
        return cls(
            stage=TraceDiagnosticStage(str(payload.get("stage", ""))),
            status=TraceDiagnosticStatus(str(payload.get("status", ""))),
            error_type=(
                str(payload["error_type"]) if payload.get("error_type") is not None else None
            ),
            attributes=tuple(sorted(raw_attributes.items())),
            artifacts=tuple(
                TraceDiagnosticArtifactV1.from_dict(artifact) for artifact in raw_artifacts
            ),
        )


@dataclass(frozen=True)
class TraceMethodRunReceiptV1:
    method_id: str
    status: TraceRunStatus
    source_trace_sha256: str
    source_request_sha256: str
    findings: tuple[TraceFindingV1, ...]
    blockers: tuple[str, ...]
    evidence_level: EvidenceLevel
    descriptor: TraceMethodDescriptorV1
    receipt_sha256: str
    schema: str = TRACE_METHOD_RUN_SCHEMA

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "method_id": self.method_id,
            "status": self.status.value,
            "source_trace_sha256": self.source_trace_sha256,
            "source_request_sha256": self.source_request_sha256,
            "findings": [finding.to_dict() for finding in self.findings],
            "blockers": list(self.blockers),
            "evidence_level": self.evidence_level.value,
            "descriptor": self.descriptor.to_dict(),
        }
        if include_sha256:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class TraceMethodRunReceiptV2:
    method_id: str
    status: TraceRunStatus
    source_trace_sha256: str
    source_request_sha256: str
    findings: tuple[TraceFindingV1, ...]
    blockers: tuple[str, ...]
    diagnostics: tuple[TraceDiagnosticEventV1, ...]
    evidence_level: EvidenceLevel
    descriptor: TraceMethodDescriptorV1
    receipt_sha256: str
    schema: str = TRACE_METHOD_RUN_V2_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACE_METHOD_RUN_V2_SCHEMA:
            raise ValueError(f"unsupported trace method run schema: {self.schema}")
        failed = tuple(
            event for event in self.diagnostics if event.status is TraceDiagnosticStatus.FAILED
        )
        if self.status is TraceRunStatus.COMPLETED and failed:
            raise ValueError("completed trace receipt cannot contain failed diagnostics")
        if self.status is TraceRunStatus.INVALID and not failed:
            raise ValueError("invalid trace receipt requires a failed diagnostic event")
        if len(failed) > 1:
            raise ValueError("trace diagnostic failure must be unique")

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "method_id": self.method_id,
            "status": self.status.value,
            "source_trace_sha256": self.source_trace_sha256,
            "source_request_sha256": self.source_request_sha256,
            "findings": [finding.to_dict() for finding in self.findings],
            "blockers": list(self.blockers),
            "diagnostics": [event.to_dict() for event in self.diagnostics],
            "evidence_level": self.evidence_level.value,
            "descriptor": self.descriptor.to_dict(),
        }
        if include_sha256:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class TraceMethodOutputV1:
    findings: tuple[TraceFindingV1, ...]
    evidence_level: EvidenceLevel
    blockers: tuple[str, ...] = ()
    diagnostics: tuple[TraceDiagnosticEventV1, ...] = ()


class TraceMethodAdapter(Protocol):
    descriptor: TraceMethodDescriptorV1

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1: ...


@dataclass(frozen=True)
class _UnavailableAdapter:
    descriptor: TraceMethodDescriptorV1

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        del request
        reason = self.descriptor.unavailable_reason or "method adapter is not integrated"
        raise TraceMethodBlocked(reason)


class TraceMethodBlocked(RuntimeError):
    """Fail-closed admission outcome, distinct from method execution failure."""


class TraceInfrastructureHalt(RuntimeError):
    """Stop a matrix while retaining structured provider evidence."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: TraceFailureKind = TraceFailureKind.PROVIDER_RATE_LIMIT,
        attributes: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if failure_kind not in {
            TraceFailureKind.PROVIDER_RATE_LIMIT,
            TraceFailureKind.WORKER_EXECUTION_FAILED,
        }:
            raise ValueError("infrastructure halt requires an infrastructure failure kind")
        evidence = dict(attributes)
        evidence.setdefault("failure_domain", TraceFailureDomain.INFRASTRUCTURE.value)
        evidence.setdefault("failure_kind", failure_kind.value)
        if (
            evidence["failure_domain"] != TraceFailureDomain.INFRASTRUCTURE.value
            or evidence["failure_kind"] != failure_kind.value
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in evidence.items())
        ):
            raise ValueError("infrastructure halt attributes are inconsistent")
        super().__init__(message)
        self.failure_kind = failure_kind
        self.attributes = tuple(sorted(evidence.items()))


def is_provider_account_halt_message(message: object) -> bool:
    """Recognize account-wide failures without treating transient throttling as fatal."""

    lowered = str(message).casefold()
    return any(fragment in lowered for fragment in _PROVIDER_ACCOUNT_HALT_FRAGMENTS)


class TraceMethodExecutionError(RuntimeError):
    """Adapter failure carrying hash-bound evidence from stages already reached."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        diagnostics: tuple[TraceDiagnosticEventV1, ...],
    ) -> None:
        if not message or not error_type:
            raise ValueError("trace method execution error identity must be non-empty")
        if not diagnostics or not any(
            event.status is TraceDiagnosticStatus.FAILED for event in diagnostics
        ):
            raise ValueError("trace method execution error requires failed diagnostics")
        super().__init__(message)
        self.error_type = error_type
        self.diagnostics = diagnostics


class TraceMethodPlatform:
    """Deep module owning method dispatch, validation and audit receipts."""

    def __init__(self, adapters: tuple[TraceMethodAdapter, ...]) -> None:
        by_id: dict[str, TraceMethodAdapter] = {}
        for adapter in adapters:
            method_id = adapter.descriptor.method_id
            if method_id in by_id:
                raise ValueError(f"duplicate trace method adapter: {method_id}")
            by_id[method_id] = adapter
        self._adapters = by_id

    @classmethod
    def from_descriptors(
        cls,
        descriptors: tuple[TraceMethodDescriptorV1, ...],
    ) -> TraceMethodPlatform:
        return cls(tuple(_UnavailableAdapter(descriptor) for descriptor in descriptors))

    @classmethod
    def overlay(
        cls,
        *,
        catalog: tuple[TraceMethodDescriptorV1, ...],
        adapters: tuple[TraceMethodAdapter, ...],
        replace_method_ids: tuple[str, ...] = (),
    ) -> TraceMethodPlatform:
        adapter_by_id = {adapter.descriptor.method_id: adapter for adapter in adapters}
        if len(adapter_by_id) != len(adapters):
            raise ValueError("duplicate trace method adapter in overlay")
        catalog_ids = {descriptor.method_id for descriptor in catalog}
        unknown_replacements = set(replace_method_ids) - catalog_ids
        if unknown_replacements:
            raise ValueError(f"unknown catalog replacements: {sorted(unknown_replacements)}")
        combined: list[TraceMethodAdapter] = []
        for descriptor in catalog:
            if descriptor.method_id in replace_method_ids:
                continue
            combined.append(
                adapter_by_id.pop(descriptor.method_id, _UnavailableAdapter(descriptor))
            )
        combined.extend(adapter_by_id.values())
        return cls(tuple(combined))

    @staticmethod
    def default_catalog() -> tuple[TraceMethodDescriptorV1, ...]:
        return _default_catalog()

    def catalog(self) -> tuple[TraceMethodDescriptorV1, ...]:
        return tuple(self._adapters[method_id].descriptor for method_id in sorted(self._adapters))

    def run(
        self,
        method_id: str,
        request: TraceMethodRequestV1 | TraceCaseV1 | TraceEpisodeV1,
    ) -> TraceMethodRunReceiptV2:
        if isinstance(request, (TraceCaseV1, TraceEpisodeV1)):
            request = TraceMethodRequestV1(case=request)
        if not isinstance(request, TraceMethodRequestV1):
            raise TypeError(
                "trace method run requires TraceMethodRequestV1, TraceCaseV1 or TraceEpisodeV1"
            )
        case = request.case
        active_stage = TraceDiagnosticStage.ADMISSION
        diagnostics: tuple[TraceDiagnosticEventV1, ...] = ()
        try:
            adapter = self._adapters[method_id]
        except KeyError as error:
            raise ValueError(
                f"unknown trace method {method_id!r}; available={sorted(self._adapters)}"
            ) from error
        descriptor = adapter.descriptor
        try:
            if descriptor.unavailable_reason is not None:
                raise TraceMethodBlocked(descriptor.unavailable_reason)
            if (
                case.observed_outcome.success
                and descriptor.outcome_scope is TraceOutcomeScope.OBSERVED_FAILURE
            ):
                raise TraceMethodBlocked("method requires an observed failure")
            missing_modalities = sorted(
                set(descriptor.input_modalities) - request.available_modalities
            )
            if missing_modalities:
                raise TraceMethodBlocked(
                    "missing required input modalities: " + ", ".join(missing_modalities)
                )
            active_stage = TraceDiagnosticStage.ADAPTER_EXECUTION
            output = adapter.execute(request)
            findings = output.findings
            diagnostics = output.diagnostics
            if output.blockers:
                raise ValueError("completed trace method output cannot contain blockers")
            active_stage = TraceDiagnosticStage.PLATFORM_VALIDATION
            self._validate_findings(descriptor, case, findings)
            self._validate_evidence_ceiling(descriptor, output.evidence_level)
            diagnostics = (
                *diagnostics,
                TraceDiagnosticEventV1(
                    stage=TraceDiagnosticStage.PLATFORM_VALIDATION,
                    status=TraceDiagnosticStatus.COMPLETED,
                ),
            )
            status = TraceRunStatus.COMPLETED
            blockers = output.blockers
            evidence_level = output.evidence_level
        except TraceInfrastructureHalt as error:
            raise TraceInfrastructureHalt(
                redact_text(str(error)),
                failure_kind=error.failure_kind,
                attributes=tuple(
                    (key, redact_text(value)) for key, value in error.attributes
                ),
            ) from None
        except TraceMethodBlocked as error:
            findings = ()
            status = TraceRunStatus.BLOCKED
            blockers = (redact_text(str(error)),)
            diagnostics = (
                TraceDiagnosticEventV1(
                    stage=TraceDiagnosticStage.ADMISSION,
                    status=TraceDiagnosticStatus.FAILED,
                    error_type=type(error).__name__,
                ),
            )
            evidence_level = EvidenceLevel.L1_EXPLANATION
        except TraceMethodExecutionError as error:
            findings = ()
            status = TraceRunStatus.INVALID
            blockers = (redact_text(f"{error.error_type}: {error}"),)
            diagnostics = error.diagnostics
            evidence_level = EvidenceLevel.L1_EXPLANATION
        except Exception as error:
            findings = ()
            status = TraceRunStatus.INVALID
            blockers = (redact_text(f"{type(error).__name__}: {error}"),)
            diagnostics = (
                *diagnostics,
                TraceDiagnosticEventV1(
                    stage=active_stage,
                    status=TraceDiagnosticStatus.FAILED,
                    error_type=type(error).__name__,
                ),
            )
            evidence_level = EvidenceLevel.L1_EXPLANATION
        unhashed = TraceMethodRunReceiptV2(
            method_id=descriptor.method_id,
            status=status,
            source_trace_sha256=case.sha256,
            source_request_sha256=request.sha256,
            findings=findings,
            blockers=blockers,
            diagnostics=diagnostics,
            evidence_level=evidence_level,
            descriptor=descriptor,
            receipt_sha256="",
        )
        return TraceMethodRunReceiptV2(
            method_id=unhashed.method_id,
            status=unhashed.status,
            source_trace_sha256=unhashed.source_trace_sha256,
            source_request_sha256=unhashed.source_request_sha256,
            findings=unhashed.findings,
            blockers=unhashed.blockers,
            diagnostics=unhashed.diagnostics,
            evidence_level=unhashed.evidence_level,
            descriptor=unhashed.descriptor,
            receipt_sha256="sha256:" + canonical_sha256(unhashed.to_dict(include_sha256=False)),
        )

    @staticmethod
    def _validate_findings(
        descriptor: TraceMethodDescriptorV1,
        case: TraceInputV1,
        findings: tuple[TraceFindingV1, ...],
    ) -> None:
        if not findings:
            raise ValueError("completed trace method returned no findings")
        allowed_kinds = set(descriptor.output_kinds)
        for finding in findings:
            if finding.kind not in allowed_kinds:
                raise ValueError(
                    f"method {descriptor.method_id} returned undeclared finding kind "
                    f"{finding.kind.value}"
                )
            unknown = set(finding.event_ids) - case.event_ids
            if unknown:
                raise ValueError(f"trace finding references unknown events: {sorted(unknown)}")

    @staticmethod
    def _validate_evidence_ceiling(
        descriptor: TraceMethodDescriptorV1,
        evidence_level: EvidenceLevel,
    ) -> None:
        ordered = tuple(EvidenceLevel)
        if ordered.index(evidence_level) > ordered.index(descriptor.evidence_ceiling):
            raise ValueError(
                f"method {descriptor.method_id} exceeded its evidence ceiling: "
                f"{evidence_level.value} > {descriptor.evidence_ceiling.value}"
            )


def _catalog_descriptor(
    method_id: str,
    display_name: str,
    source_relpath: str | None,
    family: str,
    *,
    corpus_method_id: str | None = None,
    input_modalities: tuple[str, ...] | None = None,
) -> TraceMethodDescriptorV1:
    resolved_corpus_method_id = corpus_method_id or method_id
    shape = {
        "failure_attribution": (
            ("step_attribution", "component_attribution"),
            (TraceFindingKind.STEP_ATTRIBUTION, TraceFindingKind.COMPONENT_ATTRIBUTION),
            EvidenceLevel.L3_LOCALIZED,
        ),
        "risk_detection": (
            ("trajectory_risk",),
            (TraceFindingKind.TRAJECTORY_RISK,),
            EvidenceLevel.L2_TYPED,
        ),
        "recovery": (
            ("recovery_proposal",),
            (TraceFindingKind.RECOVERY_PROPOSAL,),
            EvidenceLevel.L2_TYPED,
        ),
        "reflection": (
            ("error_classification", "recovery_proposal"),
            (TraceFindingKind.ERROR_CLASSIFICATION, TraceFindingKind.RECOVERY_PROPOSAL),
            EvidenceLevel.L2_TYPED,
        ),
        "credit_assignment": (
            ("process_scoring",),
            (TraceFindingKind.PROCESS_SCORE,),
            EvidenceLevel.L2_TYPED,
        ),
        "benchmark": (
            ("error_classification",),
            (TraceFindingKind.ERROR_CLASSIFICATION,),
            EvidenceLevel.L2_TYPED,
        ),
    }[family]
    if method_id.startswith(("who_when.", "traceelephant.")):
        # LongH's adapted single-actor projection has no frozen component labels.
        shape = (
            ("step_attribution",),
            (TraceFindingKind.STEP_ATTRIBUTION,),
            EvidenceLevel.L3_LOCALIZED,
        )
    inferred_modalities = (
        ("text_trajectory", "image_sequence")
        if any(
            token in method_id
            for token in (
                "mobile",
                "gui",
                "safe",
                "phoenix",
                "vifailback",
                "cantest",
                "dense",
                "cosmo",
                "t2vlm",
                "hats",
            )
        )
        else ("text_trajectory",)
    )
    unavailable = (
        "no public source implementation was found in the supplied corpus"
        if source_relpath is None
        else "native adapter has not passed source and runtime admission"
    )
    return TraceMethodDescriptorV1(
        method_id=method_id,
        display_name=display_name,
        family=family,
        capabilities=shape[0],
        input_modalities=input_modalities or inferred_modalities,
        output_kinds=shape[1],
        integration_level="catalog_only",
        execution_path="paper_only" if source_relpath is None else "upstream_native_runner",
        evidence_ceiling=shape[2],
        source_relpath=source_relpath,
        source_revision=None,
        unavailable_reason=unavailable,
        corpus_method_id=corpus_method_id,
        outcome_scope=(
            TraceOutcomeScope.ANY_TERMINAL_OUTCOME
            if family == "risk_detection" or resolved_corpus_method_id in {"aegis", "agentdiagnose"}
            else TraceOutcomeScope.OBSERVED_FAILURE
        ),
    )


# The supplied corpus has 46 paper implementations in 45 top-level repositories
# (GUI-Critic-R1 is a MobileAgent subdirectory) plus 28 papers without public code.
_SOURCE_ROWS = (
    ("tracer", "TRACER", "code/agent-tracer", "risk_detection"),
    ("who_when", "Who&When", "code/Agents_Failure_Attribution", "failure_attribution"),
    ("lats", "Language Agent Tree Search", "code/LanguageAgentTreeSearch", "recovery"),
    ("thought_rollback", "Thought Rollback", "code/llmpebase", "recovery"),
    ("syncmind", "SyncMind", "code/SyncMind", "risk_detection"),
    ("agentracer", "AgenTracer", "code/AgenTracer", "failure_attribution"),
    ("agent_pro", "Agent-Pro", "code/Agent-Pro", "reflection"),
    ("critic", "CRITIC", "code/ProphetNet", "reflection"),
    ("agentrefine", "AgentRefine", "code/AgentRefine", "reflection"),
    ("aegis", "AEGIS", "code/AEGIS", "failure_attribution"),
    ("mobile_agent_v2", "Mobile-Agent-v2", "code/MobileAgent", "recovery"),
    ("se_agent", "SE-Agent", "code/SE-Agent", "reflection"),
    ("mobileuse", "MobileUse", "code/mobile-use", "reflection"),
    ("gui_critic_r1", "GUI-Critic-R1", "code/MobileAgent/GUI-Critic-R1", "failure_attribution"),
    (
        "embodied_agent_interface",
        "Embodied Agent Interface",
        "code/embodied-agent-interface",
        "benchmark",
    ),
    ("rise", "Recursive Introspection", "code/RISE", "reflection"),
    ("fiper", "Failure Prediction at Runtime", "code/fiper", "risk_detection"),
    ("safe_vla", "SAFE VLA", "code/SAFE", "risk_detection"),
    ("gui_reflection", "GUI-Reflection", "code/GUI_Reflection", "reflection"),
    ("agentissue_bench", "AgentIssue-Bench", "code/AgentIssue-Bench", "benchmark"),
    ("lessonl", "Lessons Learned", "code/LessonL", "reflection"),
    ("urst", "Uncertainty-Aware Self-Trained Evaluator", "code/URST", "risk_detection"),
    ("aed", "Adaptable Error Detection", "code/aed-code", "risk_detection"),
    ("eto", "Trial and Error", "code/ETO", "reflection"),
    ("steca", "STeCa", "code/STeCa", "credit_assignment"),
    ("traceelephant", "TraceElephant", "code/TraceElephant", "failure_attribution"),
    ("evotool", "EvoTool", "code/ACL_2026_EvoTool", "failure_attribution"),
    ("toolprmbench", "ToolPRMBench", "code/ToolPRMBench", "benchmark"),
    ("cantest", "Where Did It Go Wrong / CanTest", "code/CanTest", "failure_attribution"),
    ("plan_rewardbench", "Plan-RewardBench", "code/Plan-RewardBench", "benchmark"),
    ("ipr", "Watch Every Step / IPR", "code/IPR", "credit_assignment"),
    ("tools_fail", "Tools Fail", "code/tools-fail", "risk_detection"),
    ("re_rest", "Re-ReST", "code/Re-ReST", "reflection"),
    ("promst", "PROMST", "code/PROMST", "reflection"),
    ("backtrackagent", "BacktrackAgent", "code/backtrackagent", "failure_attribution"),
    (
        "ga_rollback",
        "Generator-Assistant Stepwise Rollback",
        "code/GA-Rollback",
        "failure_attribution",
    ),
    ("cola", "COLA", "code/COLA-demo", "recovery"),
    ("inferact", "Preemptive Detection / InferAct", "code/emnlp2025-inferact", "risk_detection"),
    ("agentdiagnose", "AgentDiagnose", "code/AgentDiagnose", "failure_attribution"),
    ("critictool", "CriticTool", "code/CriticTool", "reflection"),
    ("phoenix", "Phoenix", "code/Motion-based-Self-Reflection-Framework", "reflection"),
    ("vifailback", "ViFailback", "code/ViFailback", "failure_attribution"),
    ("hats", "HATS", "code/HATS", "credit_assignment"),
    ("t2vlm", "Temporally Consistent Rewards", "code/T2VLM", "credit_assignment"),
    ("dense_policy", "Dense Policy", "code/DensePolicy", "recovery"),
    ("cosmo", "COSMO", "code/VLN-COSMO", "recovery"),
)

_PAPER_ONLY_ROWS = (
    ("resilience_faulty_agents", "Resilience of Faulty Agents", "failure_attribution"),
    ("correct", "CORRECT", "failure_attribution"),
    ("imitation_expert_corrections", "On-policy Expert Corrections", "recovery"),
    (
        "interaction_to_prompt_rules",
        "Interaction Trajectories to Prompt Rules",
        "failure_attribution",
    ),
    ("tree_planner", "Tree-Planner", "recovery"),
    ("self_debug", "Teaching LLMs to Self-Debug", "reflection"),
    ("cannot_self_correct", "Large Language Models Cannot Self-Correct", "benchmark"),
    ("privileged_ai_feedback", "Privileged AI Feedback", "reflection"),
    ("dover", "DoVer", "failure_attribution"),
    ("longhorizonui", "LongHorizonUI", "recovery"),
    ("copper", "Reflective Multi-Agent Collaboration", "failure_attribution"),
    ("rlbf", "RL with Backtracking Feedback", "recovery"),
    ("errorprobe", "ErrorProbe", "failure_attribution"),
    ("verigui", "Don't Act Blindly / VeriGUI", "risk_detection"),
    ("reflectool", "ReflecTool", "benchmark"),
    ("masc", "MASC", "reflection"),
    ("architect", "ARCHITECT", "failure_attribution"),
    ("action_boundary_blindness", "Action Boundary Blindness", "risk_detection"),
    ("metareflection", "MetaReflection", "reflection"),
    ("reagent", "ReAgent", "recovery"),
    ("process_supervision", "Automated Process Supervision", "credit_assignment"),
    ("samule", "Multi-level Reflection / SAMuLE", "reflection"),
    ("code_as_monitor", "Code-as-Monitor", "risk_detection"),
    ("flare", "FLARE", "failure_attribution"),
    ("reach", "REACH", "recovery"),
    ("drama", "DRAMA", "recovery"),
    ("vca", "Video Curious Agent", "recovery"),
    ("mindexplore", "Long-Horizon VLA / MindExplore", "recovery"),
)


def _default_catalog() -> tuple[TraceMethodDescriptorV1, ...]:
    executable_shapes: dict[str, tuple[tuple[str, str, tuple[str, ...] | None], ...]] = {
        "tracer": (
            (
                "tracer.risk_aggregation",
                "TRACER Risk Aggregation",
                ("text_trajectory", "tracer_signal_bundle"),
            ),
        ),
        "who_when": tuple(
            (
                f"who_when.{variant}",
                f"Who&When {variant.replace('_', ' ').title()}",
                None,
            )
            for variant in ("all_at_once", "step_by_step", "binary_search")
        ),
        "traceelephant": tuple(
            (
                f"traceelephant.{variant}",
                f"TraceElephant {variant.replace('_', ' ').title()}",
                None,
            )
            for variant in ("all_at_once", "step_by_step", "binary_search")
        ),
        "aegis": (("aegis.error_attribution", "AEGIS Error Attribution", None),),
        "evotool": (("evotool.blame", "EvoTool Trajectory-Grounded Blame", None),),
        "agentdiagnose": (
            (
                "agentdiagnose.reasoning_quality",
                "AgentDiagnose Reasoning Quality",
                None,
            ),
        ),
    }
    rows: list[TraceMethodDescriptorV1] = []
    for method_id, name, source, family in _SOURCE_ROWS:
        shapes = executable_shapes.get(method_id, ((method_id, name, None),))
        rows.extend(
            _catalog_descriptor(
                executable_id,
                executable_name,
                source,
                family,
                corpus_method_id=method_id,
                input_modalities=modalities,
            )
            for executable_id, executable_name, modalities in shapes
        )
    rows.extend(
        _catalog_descriptor(method_id, name, None, family)
        for method_id, name, family in _PAPER_ONLY_ROWS
    )
    return tuple(sorted(rows, key=lambda item: item.method_id))


__all__ = [
    "TraceDiagnosticArtifactV1",
    "TraceDiagnosticEventV1",
    "TraceDiagnosticStage",
    "TraceDiagnosticStatus",
    "TraceFailureDomain",
    "TraceFailureKind",
    "TraceInputArtifactV1",
    "TraceFindingKind",
    "TraceFindingV1",
    "TraceMethodAdapter",
    "TraceMethodBlocked",
    "TraceInfrastructureHalt",
    "TraceMethodExecutionError",
    "TraceMethodDescriptorV1",
    "TraceMethodOutputV1",
    "TraceMethodPlatform",
    "TraceMethodRequestV1",
    "TraceMethodRunReceiptV1",
    "TraceMethodRunReceiptV2",
    "TraceOutcomeScope",
    "TraceRunStatus",
    "is_provider_account_halt_message",
]
