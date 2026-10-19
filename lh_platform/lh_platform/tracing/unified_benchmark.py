"""Frozen, gold-blind evaluation for heterogeneous unified tracer outputs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.integrations._provenance import require_clean_source_revision
from lh_platform.tracing.integrations._runtime import normalize_python_entrypoint
from lh_platform.tracing.types import TraceCaseV1, TraceEventV1, TraceOutcomeV1
from lh_platform.tracing.unified import (
    TraceFindingKind,
    TraceFindingV1,
    TraceMethodPlatform,
    TraceMethodRequestV1,
    TraceMethodRunReceiptV2,
    TraceRunStatus,
)

TRACE_BENCHMARK_MANIFEST_SCHEMA = "lh_trace_benchmark_manifest/v1"
TRACE_BENCHMARK_RESULT_SCHEMA = "lh_trace_benchmark_result/v3"
PREVIOUS_TRACE_BENCHMARK_RESULT_SCHEMA = "lh_trace_benchmark_result/v2"
LEGACY_TRACE_BENCHMARK_RESULT_SCHEMA = "lh_trace_benchmark_result/v1"
TRACE_BENCHMARK_CHECKPOINT_SCHEMA = "lh_trace_benchmark_checkpoint/v3"
HIGH_PROCESS_SCORE_ON_FAILURE_THRESHOLD = 0.8
WHO_WHEN_SELECTION_ALGORITHM = (
    "per_split_lowest_sha256(seed:split:filename)_after_length_filter"
)
DEFAULT_TEXT_TRACE_METHOD_IDS = (
    "who_when.all_at_once",
    "who_when.step_by_step",
    "who_when.binary_search",
    "traceelephant.all_at_once",
    "traceelephant.step_by_step",
    "traceelephant.binary_search",
    "aegis.error_attribution",
    "evotool.blame",
    "agentdiagnose.reasoning_quality",
)
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return "sha256:" + canonical_sha256(payload)


@lru_cache(maxsize=32)
def _python_environment_sha256(python_executable: str) -> str:
    script = """
import importlib.metadata
import json
import platform
import sys

distributions = sorted({
    (str(item.metadata.get("Name", "")).strip().casefold().replace("_", "-")
     + "==" + str(item.version))
    for item in importlib.metadata.distributions()
    if str(item.metadata.get("Name", "")).strip()
})
print(json.dumps({
    "implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "is_virtual_environment": sys.prefix != sys.base_prefix,
    "distributions": distributions,
}, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [python_executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"could not attest trace Python environment: {error[-1000:]}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("trace Python environment attestation is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError("trace Python environment attestation is not an object")
    return _canonical_digest(payload)


@dataclass(frozen=True)
class TraceBenchmarkGoldV1:
    event_id: str
    component: str
    failure_expected: bool

    def __post_init__(self) -> None:
        if not self.event_id or not self.component:
            raise ValueError("trace benchmark gold identity must be non-empty")
        if self.failure_expected is not True:
            raise ValueError("traceback benchmark requires an expected failure")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "component": self.component,
            "failure_expected": self.failure_expected,
        }


@dataclass(frozen=True)
class TraceBenchmarkCaseV1:
    request: TraceMethodRequestV1
    gold: TraceBenchmarkGoldV1
    source_path: str
    source_sha256: str


@dataclass(frozen=True)
class TraceBenchmarkDatasetV1:
    dataset_id: str
    source_revision: str
    manifest_sha256: str
    selection: tuple[tuple[str, object], ...]
    cases: tuple[TraceBenchmarkCaseV1, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.source_revision or not self.cases:
            raise ValueError("trace benchmark dataset identity is incomplete")
        if not _SHA256_PATTERN.fullmatch(self.manifest_sha256):
            raise ValueError("trace benchmark manifest hash is invalid")
        case_ids = [item.request.case.case_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("trace benchmark case ids must be unique")


def _string_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _load_who_when_case(
    *,
    payload: Mapping[str, Any],
    case_id: str,
) -> tuple[TraceMethodRequestV1, TraceBenchmarkGoldV1]:
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"trace benchmark case {case_id} has no history")
    question = payload.get("question")
    task_id = payload.get("question_ID")
    component = payload.get("mistake_agent")
    raw_step = payload.get("mistake_step")
    if not all(isinstance(value, str) and value for value in (question, task_id, component)):
        raise ValueError(f"trace benchmark case {case_id} identity is incomplete")
    try:
        gold_step = int(raw_step)
    except (TypeError, ValueError) as error:
        raise ValueError(f"trace benchmark case {case_id} has invalid gold step") from error
    if not 0 <= gold_step < len(history):
        raise ValueError(f"trace benchmark case {case_id} gold step is out of range")
    observed_correct = payload.get("is_correct", payload.get("is_corrected"))
    if observed_correct is not False:
        raise ValueError(f"trace benchmark case {case_id} is not an observed failure")

    events: list[TraceEventV1] = []
    for index, raw_event in enumerate(history):
        event = _string_mapping(raw_event, label="Who&When history event")
        content = event.get("content")
        role = event.get("role")
        name = event.get("name")
        if not isinstance(content, str) or not isinstance(role, str) or not role:
            raise ValueError(f"trace benchmark case {case_id} has invalid history event")
        if index == 0:
            content = f"Question: {question}\n\n{content}"
        kind = name if isinstance(name, str) and name else role
        metadata = (("role", role),)
        if isinstance(name, str) and name:
            metadata += (("name", name),)
        events.append(
            TraceEventV1(
                event_id=f"event-{index:04d}",
                sequence=index,
                kind=kind,
                content=content,
                parent_event_ids=() if index == 0 else (f"event-{index - 1:04d}",),
                metadata=metadata,
            )
        )
    trace_case = TraceCaseV1(
        case_id=case_id,
        task_id=task_id,
        events=tuple(events),
        interruption_event_id=events[-1].event_id,
        observed_outcome=TraceOutcomeV1(
            success=False,
            terminal_reason="upstream_labeled_failure",
        ),
    )
    return (
        TraceMethodRequestV1(case=trace_case),
        TraceBenchmarkGoldV1(
            event_id=f"event-{gold_step:04d}",
            component=component,
            failure_expected=True,
        ),
    )


def load_trace_benchmark_manifest(
    manifest_path: str | Path,
    *,
    corpus_root: str | Path,
    expected_manifest_sha256: str,
) -> TraceBenchmarkDatasetV1:
    """Load one frozen manifest while keeping its gold outside model-visible requests."""

    path = Path(manifest_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    manifest = _string_mapping(raw, label="trace benchmark manifest")
    if manifest.get("schema") != TRACE_BENCHMARK_MANIFEST_SCHEMA:
        raise ValueError("unsupported trace benchmark manifest schema")
    manifest_sha256 = _canonical_digest(manifest)
    if not _SHA256_PATTERN.fullmatch(expected_manifest_sha256):
        raise ValueError("expected trace benchmark manifest hash is invalid")
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("trace benchmark manifest hash mismatch")
    root = Path(corpus_root).resolve()
    source_relpath = manifest.get("source_relpath")
    if not isinstance(source_relpath, str) or not source_relpath:
        raise ValueError("trace benchmark source_relpath is invalid")
    source = (root / source_relpath).resolve()
    if not source.is_relative_to(root) or not source.is_dir():
        raise ValueError("trace benchmark source directory is missing or escaped corpus")
    revision = require_clean_source_revision(source, method_name="trace benchmark")
    if revision != manifest.get("source_revision"):
        raise ValueError("trace benchmark source revision mismatch")
    selection = _string_mapping(manifest.get("selection"), label="trace benchmark selection")
    max_events = selection.get("max_events")
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
        raise ValueError("trace benchmark max_events must be a positive integer")
    excluded = selection.get("excluded_source_paths", [])
    if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
        raise ValueError("trace benchmark exclusion list is invalid")

    expected_source_paths = _recompute_who_when_selection(
        source=source,
        selection=selection,
    )

    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("trace benchmark manifest must contain cases")
    manifest_source_paths = tuple(
        entry.get("source_path")
        for entry in raw_cases
        if isinstance(entry, Mapping)
    )
    if manifest_source_paths != expected_source_paths:
        raise ValueError(
            "trace benchmark cases do not match the executable frozen selection"
        )
    cases: list[TraceBenchmarkCaseV1] = []
    for raw_case in raw_cases:
        entry = _string_mapping(raw_case, label="trace benchmark case entry")
        case_id = entry.get("case_id")
        source_path = entry.get("source_path")
        expected_hash = entry.get("source_sha256")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("trace benchmark case id is invalid")
        if not isinstance(source_path, str) or not source_path or source_path in excluded:
            raise ValueError(f"trace benchmark case {case_id} source path is invalid")
        source_file = (source / source_path).resolve()
        if not source_file.is_relative_to(source) or not source_file.is_file():
            raise ValueError(f"trace benchmark case {case_id} source escaped or is missing")
        if _sha256_file(source_file) != expected_hash:
            raise ValueError(f"trace benchmark case {case_id} source hash mismatch")
        payload = _string_mapping(
            json.loads(source_file.read_text(encoding="utf-8")),
            label="Who&When source case",
        )
        request, gold = _load_who_when_case(payload=payload, case_id=case_id)
        if len(request.case.events) > max_events:
            raise ValueError(f"trace benchmark case {case_id} exceeds max_events")
        if entry.get("gold_event_id") != gold.event_id:
            raise ValueError(f"trace benchmark case {case_id} gold event mismatch")
        if entry.get("gold_component") != gold.component:
            raise ValueError(f"trace benchmark case {case_id} gold component mismatch")
        cases.append(
            TraceBenchmarkCaseV1(
                request=request,
                gold=gold,
                source_path=source_path,
                source_sha256=str(expected_hash),
            )
        )
    return TraceBenchmarkDatasetV1(
        dataset_id=str(manifest.get("dataset_id", "")),
        source_revision=revision,
        manifest_sha256=manifest_sha256,
        selection=tuple(sorted(selection.items())),
        cases=tuple(cases),
    )


def _recompute_who_when_selection(
    *,
    source: Path,
    selection: Mapping[str, Any],
) -> tuple[str, ...]:
    """Recompute the declared candidate ordering instead of trusting a case list."""

    if selection.get("algorithm") != WHO_WHEN_SELECTION_ALGORITHM:
        raise ValueError("unsupported trace benchmark selection algorithm")
    seed = selection.get("seed")
    splits = selection.get("splits")
    per_split_count = selection.get("per_split_count")
    max_events = selection.get("max_events")
    excluded = selection.get("excluded_source_paths", [])
    if not isinstance(seed, str) or not seed:
        raise ValueError("trace benchmark selection seed is invalid")
    if (
        not isinstance(splits, list)
        or not splits
        or not all(isinstance(split, str) and split for split in splits)
        or len(splits) != len(set(splits))
    ):
        raise ValueError("trace benchmark selection splits are invalid")
    if (
        isinstance(per_split_count, bool)
        or not isinstance(per_split_count, int)
        or per_split_count < 1
    ):
        raise ValueError("trace benchmark per_split_count must be positive")
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
        raise ValueError("trace benchmark max_events must be a positive integer")
    excluded_paths = set(excluded)
    selected: list[str] = []
    for split in splits:
        split_root = (source / "Who&When" / split).resolve()
        if not split_root.is_relative_to(source) or not split_root.is_dir():
            raise ValueError(f"trace benchmark selection split is missing: {split}")
        candidates: list[tuple[str, str]] = []
        for candidate in split_root.glob("*.json"):
            source_path = candidate.relative_to(source).as_posix()
            if source_path in excluded_paths:
                continue
            payload = _string_mapping(
                json.loads(candidate.read_text(encoding="utf-8")),
                label="Who&When selection candidate",
            )
            history = payload.get("history")
            if isinstance(history, list) and 0 < len(history) <= max_events:
                rank = hashlib.sha256(
                    f"{seed}:{split}:{candidate.name}".encode()
                ).hexdigest()
                candidates.append((rank, source_path))
        if len(candidates) < per_split_count:
            raise ValueError(
                f"trace benchmark split {split} has fewer than {per_split_count} candidates"
            )
        selected.extend(
            source_path
            for _, source_path in sorted(candidates)[:per_split_count]
        )
    return tuple(selected)


@dataclass(frozen=True)
class TraceBenchmarkScoreV1:
    step_exact: bool | None
    component_exact: bool | None
    failure_detected: bool | None
    process_score: float | None
    high_process_score_on_failure: bool | None

    def __post_init__(self) -> None:
        for value in (
            self.step_exact,
            self.component_exact,
            self.failure_detected,
            self.high_process_score_on_failure,
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError("trace benchmark score flags must be boolean or null")
        if self.process_score is not None and (
            isinstance(self.process_score, bool)
            or not isinstance(self.process_score, (int, float))
            or not math.isfinite(self.process_score)
        ):
            raise ValueError("trace benchmark process score must be finite numeric or null")

    def to_dict(self) -> dict[str, object]:
        return {
            "step_exact": self.step_exact,
            "component_exact": self.component_exact,
            "failure_detected": self.failure_detected,
            "process_score": self.process_score,
            "high_process_score_on_failure": self.high_process_score_on_failure,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceBenchmarkScoreV1:
        return cls(
            step_exact=payload.get("step_exact"),
            component_exact=payload.get("component_exact"),
            failure_detected=payload.get("failure_detected"),
            process_score=payload.get("process_score"),
            high_process_score_on_failure=payload.get(
                "high_process_score_on_failure"
            ),
        )


def _normalize_component(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def score_trace_method_receipt(
    receipt: TraceMethodRunReceiptV2,
    gold: TraceBenchmarkGoldV1,
) -> TraceBenchmarkScoreV1:
    """Score only semantics shared by a method and this dataset's gold."""

    if receipt.status is not TraceRunStatus.COMPLETED:
        return TraceBenchmarkScoreV1(None, None, None, None, None)
    step_findings = tuple(
        finding for finding in receipt.findings if finding.kind is TraceFindingKind.STEP_ATTRIBUTION
    )
    step_exact = (
        any(finding.event_ids == (gold.event_id,) for finding in step_findings)
        if step_findings
        else None
    )
    supports_component_attribution = (
        TraceFindingKind.COMPONENT_ATTRIBUTION in receipt.descriptor.output_kinds
    )
    component_findings = tuple(
        finding
        for finding in receipt.findings
        if finding.kind is TraceFindingKind.COMPONENT_ATTRIBUTION
    )
    predicted_components = {
        value
        for finding in component_findings
        for key, value in finding.attributes
        if key == "agent_name" and value
    }
    component_exact = (
        any(
            _normalize_component(value) == _normalize_component(gold.component)
            for value in predicted_components
        )
        if supports_component_attribution and predicted_components
        else None
    )
    supports_failure_detection = (
        TraceFindingKind.ERROR_CLASSIFICATION in receipt.descriptor.output_kinds
    )
    failure_detected: bool | None = None
    if supports_failure_detection:
        failure_detected = any(
            finding.kind is TraceFindingKind.COMPONENT_ATTRIBUTION
            or (
                finding.kind is TraceFindingKind.ERROR_CLASSIFICATION
                and finding.label != "no_fault_detected"
            )
            for finding in receipt.findings
        )
    process_findings = tuple(
        finding for finding in receipt.findings if finding.kind is TraceFindingKind.PROCESS_SCORE
    )
    process_score = process_findings[0].score if process_findings else None
    high_process = (
        process_score >= HIGH_PROCESS_SCORE_ON_FAILURE_THRESHOLD
        if process_score is not None and gold.failure_expected
        else None
    )
    return TraceBenchmarkScoreV1(
        step_exact=step_exact,
        component_exact=component_exact,
        failure_detected=failure_detected,
        process_score=process_score,
        high_process_score_on_failure=high_process,
    )


@dataclass(frozen=True)
class TraceBenchmarkFindingSummaryV1:
    """Structured native finding without free-text explanation or prompt material."""

    kind: str
    event_ids: tuple[str, ...]
    label: str
    score: float | None
    attributes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.kind not in {kind.value for kind in TraceFindingKind} or not self.label:
            raise ValueError("trace benchmark finding identity is invalid")
        if len(self.event_ids) != len(set(self.event_ids)):
            raise ValueError("trace benchmark finding event ids must be unique")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise ValueError("trace benchmark finding score must be finite numeric or null")
        keys = [key for key, _ in self.attributes]
        if len(keys) != len(set(keys)) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.attributes
        ):
            raise ValueError("trace benchmark finding attributes are invalid")

    @classmethod
    def from_finding(cls, finding: TraceFindingV1) -> TraceBenchmarkFindingSummaryV1:
        return cls(
            kind=finding.kind.value,
            event_ids=finding.event_ids,
            label=finding.label,
            score=finding.score,
            attributes=finding.attributes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "event_ids": list(self.event_ids),
            "label": self.label,
            "score": self.score,
            "attributes": {key: value for key, value in self.attributes},
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> TraceBenchmarkFindingSummaryV1:
        event_ids = payload.get("event_ids")
        attributes = payload.get("attributes")
        if not isinstance(event_ids, list) or not all(
            isinstance(item, str) for item in event_ids
        ):
            raise ValueError("trace benchmark finding event ids are invalid")
        if not isinstance(attributes, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in attributes.items()
        ):
            raise ValueError("trace benchmark finding attributes are invalid")
        return cls(
            kind=str(payload.get("kind", "")),
            event_ids=tuple(event_ids),
            label=str(payload.get("label", "")),
            score=payload.get("score"),
            attributes=tuple(sorted(attributes.items())),
        )


@dataclass(frozen=True)
class TraceBenchmarkCellV1:
    case_id: str
    method_id: str
    source_path: str
    source_sha256: str
    request_sha256: str
    status: str
    receipt_sha256: str
    score: TraceBenchmarkScoreV1
    findings: tuple[TraceBenchmarkFindingSummaryV1, ...]
    predicted_event_ids: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.method_id or not self.source_path:
            raise ValueError("trace benchmark cell identity is incomplete")
        for value in (self.source_sha256, self.request_sha256, self.receipt_sha256):
            if not _SHA256_PATTERN.fullmatch(value):
                raise ValueError("trace benchmark cell hash is invalid")
        if self.status not in {status.value for status in TraceRunStatus}:
            raise ValueError("trace benchmark cell status is invalid")
        if not all(
            isinstance(value, str)
            for values in (
                self.predicted_event_ids,
                self.predicted_labels,
                self.blockers,
            )
            for value in values
        ):
            raise ValueError("trace benchmark cell lists must contain strings")

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "method_id": self.method_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "request_sha256": self.request_sha256,
            "status": self.status,
            "receipt_sha256": self.receipt_sha256,
            "score": self.score.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
            "predicted_event_ids": list(self.predicted_event_ids),
            "predicted_labels": list(self.predicted_labels),
            "blockers": list(self.blockers),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceBenchmarkCellV1:
        score = _string_mapping(payload.get("score"), label="trace benchmark cell score")
        predicted_event_ids = payload.get("predicted_event_ids")
        predicted_labels = payload.get("predicted_labels")
        blockers = payload.get("blockers")
        findings = payload.get("findings")
        if not all(
            isinstance(value, list) and all(isinstance(item, str) for item in value)
            for value in (predicted_event_ids, predicted_labels, blockers)
        ):
            raise ValueError("trace benchmark cell string lists are invalid")
        if not isinstance(findings, list) or not all(
            isinstance(finding, Mapping) for finding in findings
        ):
            raise ValueError("trace benchmark cell findings are invalid")
        return cls(
            case_id=str(payload.get("case_id", "")),
            method_id=str(payload.get("method_id", "")),
            source_path=str(payload.get("source_path", "")),
            source_sha256=str(payload.get("source_sha256", "")),
            request_sha256=str(payload.get("request_sha256", "")),
            status=str(payload.get("status", "")),
            receipt_sha256=str(payload.get("receipt_sha256", "")),
            score=TraceBenchmarkScoreV1.from_dict(score),
            findings=tuple(
                TraceBenchmarkFindingSummaryV1.from_dict(finding)
                for finding in findings
            ),
            predicted_event_ids=tuple(predicted_event_ids),
            predicted_labels=tuple(predicted_labels),
            blockers=tuple(blockers),
        )


@dataclass(frozen=True)
class TraceBenchmarkRuntimeAttestationV1:
    requested_model_id: str
    advertised_model_ids: tuple[str, ...]
    endpoint_catalog_sha256: str
    attestation_level: str = "endpoint_catalog_only"

    def __post_init__(self) -> None:
        if not self.requested_model_id or self.requested_model_id not in self.advertised_model_ids:
            raise ValueError("requested model is absent from endpoint model catalog")
        if len(self.advertised_model_ids) != len(set(self.advertised_model_ids)):
            raise ValueError("endpoint model catalog contains duplicate ids")
        expected = _canonical_digest({"model_ids": list(self.advertised_model_ids)})
        if self.endpoint_catalog_sha256 != expected:
            raise ValueError("endpoint model catalog hash mismatch")
        if self.attestation_level != "endpoint_catalog_only":
            raise ValueError("unsupported trace benchmark runtime attestation")

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_model_id": self.requested_model_id,
            "advertised_model_ids": list(self.advertised_model_ids),
            "endpoint_catalog_sha256": self.endpoint_catalog_sha256,
            "attestation_level": self.attestation_level,
        }


@dataclass(frozen=True)
class TraceBenchmarkRunSpecV1:
    """Immutable execution identity used by results and resumable checkpoints."""

    dataset_manifest_sha256: str
    model_id: str
    method_ids: tuple[str, ...]
    method_descriptors_sha256: str
    platform_revision: str
    endpoint_base_url_sha256: str
    python_executable_sha256: str
    python_environment_sha256: str
    max_tokens: int
    timeout_seconds: int
    temperature: float
    generation_seed: int
    max_model_calls: int | None
    run_spec_sha256: str

    def __post_init__(self) -> None:
        hashes = (
            self.dataset_manifest_sha256,
            self.endpoint_base_url_sha256,
            self.python_executable_sha256,
            self.python_environment_sha256,
            self.method_descriptors_sha256,
            self.run_spec_sha256,
        )
        if not all(_SHA256_PATTERN.fullmatch(value) for value in hashes):
            raise ValueError("trace benchmark run spec hash is invalid")
        if not self.model_id or not self.method_ids or len(self.method_ids) != len(
            set(self.method_ids)
        ):
            raise ValueError("trace benchmark run spec identity is incomplete")
        if not re.fullmatch(r"[0-9a-f]{40}", self.platform_revision):
            raise ValueError("trace benchmark run spec platform revision is invalid")
        if self.max_tokens < 1 or self.timeout_seconds < 1:
            raise ValueError("trace benchmark run spec limits must be positive")
        if not math.isfinite(self.temperature) or not 0.0 <= self.temperature <= 2.0:
            raise ValueError("trace benchmark temperature must be in [0, 2]")
        if isinstance(self.generation_seed, bool) or self.generation_seed < 0:
            raise ValueError("trace benchmark generation seed must be non-negative")
        if self.max_model_calls is not None and self.max_model_calls < 1:
            raise ValueError("trace benchmark model-call limit must be positive or null")
        if self.run_spec_sha256 != _canonical_digest(
            self.to_dict(include_sha256=False)
        ):
            raise ValueError("trace benchmark run spec hash mismatch")

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "model_id": self.model_id,
            "method_ids": list(self.method_ids),
            "method_descriptors_sha256": self.method_descriptors_sha256,
            "platform_revision": self.platform_revision,
            "endpoint_base_url_sha256": self.endpoint_base_url_sha256,
            "python_executable_sha256": self.python_executable_sha256,
            "python_environment_sha256": self.python_environment_sha256,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "generation_seed": self.generation_seed,
        }
        if self.max_model_calls is not None:
            payload["max_model_calls"] = self.max_model_calls
        if include_sha256:
            payload["run_spec_sha256"] = self.run_spec_sha256
        return payload


def build_trace_benchmark_run_spec(
    *,
    dataset_manifest_sha256: str,
    model_id: str,
    method_ids: tuple[str, ...],
    method_descriptors: tuple[Mapping[str, object], ...],
    platform_revision: str,
    base_url: str,
    python_executable: str,
    max_tokens: int,
    timeout_seconds: int,
    temperature: float,
    generation_seed: int,
    max_model_calls: int | None = None,
) -> TraceBenchmarkRunSpecV1:
    """Hash every execution setting that can change a resumed benchmark cell."""

    python_entrypoint = normalize_python_entrypoint(python_executable)
    descriptor_ids = tuple(str(item.get("method_id", "")) for item in method_descriptors)
    if descriptor_ids != method_ids:
        raise ValueError("trace benchmark run spec descriptors do not match method ids")
    unhashed = {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "model_id": model_id,
        "method_ids": list(method_ids),
        "method_descriptors_sha256": _canonical_digest(
            {"method_descriptors": [dict(item) for item in method_descriptors]}
        ),
        "platform_revision": platform_revision,
        "endpoint_base_url_sha256": _canonical_digest(
            {"base_url": base_url.rstrip("/")}
        ),
        "python_executable_sha256": _sha256_file(python_entrypoint),
        "python_environment_sha256": _python_environment_sha256(
            str(python_entrypoint)
        ),
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "temperature": temperature,
        "generation_seed": generation_seed,
    }
    if max_model_calls is not None:
        unhashed["max_model_calls"] = max_model_calls
    return TraceBenchmarkRunSpecV1(
        dataset_manifest_sha256=dataset_manifest_sha256,
        model_id=model_id,
        method_ids=method_ids,
        method_descriptors_sha256=str(unhashed["method_descriptors_sha256"]),
        platform_revision=platform_revision,
        endpoint_base_url_sha256=str(unhashed["endpoint_base_url_sha256"]),
        python_executable_sha256=str(unhashed["python_executable_sha256"]),
        python_environment_sha256=str(unhashed["python_environment_sha256"]),
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        generation_seed=generation_seed,
        max_model_calls=max_model_calls,
        run_spec_sha256=_canonical_digest(unhashed),
    )


def attest_openai_model_endpoint(
    *,
    base_url: str,
    api_key: str,
    requested_model_id: str,
    timeout_seconds: int,
) -> TraceBenchmarkRuntimeAttestationV1:
    """Bind a run to the model catalog visible at its OpenAI-compatible endpoint."""

    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    catalog = _string_mapping(payload, label="OpenAI model catalog")
    raw_models = catalog.get("data")
    if not isinstance(raw_models, list):
        raise ValueError("OpenAI model catalog data is invalid")
    advertised: list[str] = []
    for item in raw_models:
        model = _string_mapping(item, label="OpenAI model catalog entry")
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            advertised.append(model_id)
    frozen_ids = tuple(sorted(advertised))
    return TraceBenchmarkRuntimeAttestationV1(
        requested_model_id=requested_model_id,
        advertised_model_ids=frozen_ids,
        endpoint_catalog_sha256=_canonical_digest({"model_ids": list(frozen_ids)}),
    )


@dataclass(frozen=True)
class TraceBenchmarkAggregateV1:
    method_id: str
    case_count: int
    completed_count: int
    blocked_count: int
    invalid_count: int
    step_evaluable_count: int
    step_exact_count: int
    component_evaluable_count: int
    component_exact_count: int
    failure_detection_evaluable_count: int
    failure_detected_count: int
    process_score_evaluable_count: int
    high_process_score_on_failure_count: int
    mean_process_score: float | None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class TraceBenchmarkResultV1:
    dataset_id: str
    dataset_manifest_sha256: str
    dataset_source_revision: str
    platform_revision: str
    model_id: str
    execution_id: str
    method_ids: tuple[str, ...]
    method_descriptors: tuple[Mapping[str, object], ...]
    runtime_attestation: TraceBenchmarkRuntimeAttestationV1
    run_spec: TraceBenchmarkRunSpecV1
    cells: tuple[TraceBenchmarkCellV1, ...]
    aggregates: tuple[TraceBenchmarkAggregateV1, ...]
    result_sha256: str
    schema: str = TRACE_BENCHMARK_RESULT_SCHEMA

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_source_revision": self.dataset_source_revision,
            "platform_revision": self.platform_revision,
            "model_id": self.model_id,
            "execution_id": self.execution_id,
            "method_ids": list(self.method_ids),
            "method_descriptors": [dict(item) for item in self.method_descriptors],
            "runtime_attestation": self.runtime_attestation.to_dict(),
            "run_spec": self.run_spec.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
            "aggregates": [aggregate.to_dict() for aggregate in self.aggregates],
            "high_process_score_on_failure_threshold": (HIGH_PROCESS_SCORE_ON_FAILURE_THRESHOLD),
        }
        if include_sha256:
            payload["result_sha256"] = self.result_sha256
        return payload


def _aggregate(
    method_id: str,
    cells: tuple[TraceBenchmarkCellV1, ...],
) -> TraceBenchmarkAggregateV1:
    selected = tuple(cell for cell in cells if cell.method_id == method_id)
    step = tuple(cell.score.step_exact for cell in selected if cell.score.step_exact is not None)
    component = tuple(
        cell.score.component_exact for cell in selected if cell.score.component_exact is not None
    )
    detected = tuple(
        cell.score.failure_detected for cell in selected if cell.score.failure_detected is not None
    )
    process = tuple(
        cell.score.process_score for cell in selected if cell.score.process_score is not None
    )
    high_process = tuple(
        cell.score.high_process_score_on_failure
        for cell in selected
        if cell.score.high_process_score_on_failure is not None
    )
    return TraceBenchmarkAggregateV1(
        method_id=method_id,
        case_count=len(selected),
        completed_count=sum(cell.status == TraceRunStatus.COMPLETED.value for cell in selected),
        blocked_count=sum(cell.status == TraceRunStatus.BLOCKED.value for cell in selected),
        invalid_count=sum(cell.status == TraceRunStatus.INVALID.value for cell in selected),
        step_evaluable_count=len(step),
        step_exact_count=sum(step),
        component_evaluable_count=len(component),
        component_exact_count=sum(component),
        failure_detection_evaluable_count=len(detected),
        failure_detected_count=sum(detected),
        process_score_evaluable_count=len(process),
        high_process_score_on_failure_count=sum(high_process),
        mean_process_score=sum(process) / len(process) if process else None,
    )


def run_trace_benchmark(
    *,
    platform: TraceMethodPlatform,
    dataset: TraceBenchmarkDatasetV1,
    model_id: str,
    execution_id: str,
    method_ids: tuple[str, ...],
    platform_revision: str,
    runtime_attestation: TraceBenchmarkRuntimeAttestationV1,
    run_spec: TraceBenchmarkRunSpecV1,
    resume_cells: tuple[TraceBenchmarkCellV1, ...] = (),
    progress_callback: Callable[[TraceBenchmarkCellV1, int, int], None] | None = None,
) -> TraceBenchmarkResultV1:
    if not model_id or not method_ids or len(method_ids) != len(set(method_ids)):
        raise ValueError("trace benchmark model and unique method ids are required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", execution_id):
        raise ValueError("trace benchmark execution id is invalid")
    available = {descriptor.method_id for descriptor in platform.catalog()}
    unknown = set(method_ids) - available
    if unknown:
        raise ValueError(f"trace benchmark contains unknown methods: {sorted(unknown)}")
    if not re.fullmatch(r"[0-9a-f]{40}", platform_revision):
        raise ValueError("trace benchmark platform revision must be a Git SHA-1")
    if runtime_attestation.requested_model_id != model_id:
        raise ValueError("trace benchmark runtime model does not match requested model")
    expected_run_identity = (
        dataset.manifest_sha256,
        model_id,
        method_ids,
        platform_revision,
    )
    observed_run_identity = (
        run_spec.dataset_manifest_sha256,
        run_spec.model_id,
        run_spec.method_ids,
        run_spec.platform_revision,
    )
    if observed_run_identity != expected_run_identity:
        raise ValueError("trace benchmark run spec identity mismatch")
    descriptors_by_id = {
        descriptor.method_id: descriptor for descriptor in platform.catalog()
    }
    selected_descriptors = tuple(
        descriptors_by_id[method_id].to_dict() for method_id in method_ids
    )
    expected_descriptor_sha256 = _canonical_digest(
        {"method_descriptors": [dict(item) for item in selected_descriptors]}
    )
    if run_spec.method_descriptors_sha256 != expected_descriptor_sha256:
        raise ValueError("trace benchmark run spec method provenance mismatch")
    cases_by_id = {
        item.request.case.case_id: item for item in dataset.cases
    }
    expected_pairs = tuple(
        (item.request.case.case_id, method_id)
        for item in dataset.cases
        for method_id in method_ids
    )
    cells_by_pair: dict[tuple[str, str], TraceBenchmarkCellV1] = {}
    for cell in resume_cells:
        pair = (cell.case_id, cell.method_id)
        item = cases_by_id.get(cell.case_id)
        if pair not in expected_pairs or pair in cells_by_pair or item is None:
            raise ValueError("trace benchmark checkpoint contains an unexpected cell")
        if (
            cell.source_path != item.source_path
            or cell.source_sha256 != item.source_sha256
            or cell.request_sha256 != item.request.sha256
        ):
            raise ValueError("trace benchmark checkpoint cell provenance mismatch")
        cells_by_pair[pair] = cell
    total_cells = len(dataset.cases) * len(method_ids)
    for item in dataset.cases:
        for method_id in method_ids:
            pair = (item.request.case.case_id, method_id)
            if pair in cells_by_pair:
                continue
            receipt = platform.run(method_id, item.request)
            cell = TraceBenchmarkCellV1(
                case_id=item.request.case.case_id,
                method_id=method_id,
                source_path=item.source_path,
                source_sha256=item.source_sha256,
                request_sha256=item.request.sha256,
                status=receipt.status.value,
                receipt_sha256=receipt.receipt_sha256,
                score=score_trace_method_receipt(receipt, item.gold),
                findings=tuple(
                    TraceBenchmarkFindingSummaryV1.from_finding(finding)
                    for finding in receipt.findings
                ),
                predicted_event_ids=tuple(
                    sorted(
                        {event_id for finding in receipt.findings for event_id in finding.event_ids}
                    )
                ),
                predicted_labels=tuple(finding.label for finding in receipt.findings),
                blockers=receipt.blockers,
            )
            cells_by_pair[pair] = cell
            if progress_callback is not None:
                progress_callback(cell, len(cells_by_pair), total_cells)
    frozen_cells = tuple(cells_by_pair[pair] for pair in expected_pairs)
    aggregates = tuple(_aggregate(method_id, frozen_cells) for method_id in method_ids)
    method_descriptors = selected_descriptors
    unhashed = TraceBenchmarkResultV1(
        dataset_id=dataset.dataset_id,
        dataset_manifest_sha256=dataset.manifest_sha256,
        dataset_source_revision=dataset.source_revision,
        platform_revision=platform_revision,
        model_id=model_id,
        execution_id=execution_id,
        method_ids=method_ids,
        method_descriptors=method_descriptors,
        runtime_attestation=runtime_attestation,
        run_spec=run_spec,
        cells=frozen_cells,
        aggregates=aggregates,
        result_sha256="",
    )
    return TraceBenchmarkResultV1(
        dataset_id=unhashed.dataset_id,
        dataset_manifest_sha256=unhashed.dataset_manifest_sha256,
        dataset_source_revision=unhashed.dataset_source_revision,
        platform_revision=unhashed.platform_revision,
        model_id=unhashed.model_id,
        execution_id=unhashed.execution_id,
        method_ids=unhashed.method_ids,
        method_descriptors=unhashed.method_descriptors,
        runtime_attestation=unhashed.runtime_attestation,
        run_spec=unhashed.run_spec,
        cells=unhashed.cells,
        aggregates=unhashed.aggregates,
        result_sha256="sha256:" + canonical_sha256(unhashed.to_dict(include_sha256=False)),
    )


def write_trace_benchmark_result(
    result: TraceBenchmarkResultV1,
    output_path: str | Path,
) -> None:
    atomic_write_json(output_path, result.to_dict(), overwrite=False)


@dataclass(frozen=True)
class TraceBenchmarkCheckpointV1:
    dataset_manifest_sha256: str
    model_id: str
    execution_id: str
    method_ids: tuple[str, ...]
    platform_revision: str
    runtime_attestation: TraceBenchmarkRuntimeAttestationV1
    run_spec: TraceBenchmarkRunSpecV1
    cells: tuple[TraceBenchmarkCellV1, ...]
    schema: str = TRACE_BENCHMARK_CHECKPOINT_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "model_id": self.model_id,
            "execution_id": self.execution_id,
            "method_ids": list(self.method_ids),
            "platform_revision": self.platform_revision,
            "runtime_attestation": self.runtime_attestation.to_dict(),
            "run_spec": self.run_spec.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
        }


def write_trace_benchmark_checkpoint(
    checkpoint: TraceBenchmarkCheckpointV1,
    checkpoint_path: str | Path,
    *,
    overwrite: bool,
) -> None:
    atomic_write_json(checkpoint_path, checkpoint.to_dict(), overwrite=overwrite)


def load_trace_benchmark_checkpoint(
    checkpoint_path: str | Path,
    *,
    dataset_manifest_sha256: str,
    model_id: str,
    execution_id: str,
    method_ids: tuple[str, ...],
    platform_revision: str,
    runtime_attestation: TraceBenchmarkRuntimeAttestationV1,
    run_spec: TraceBenchmarkRunSpecV1,
) -> TraceBenchmarkCheckpointV1:
    raw = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    payload = _string_mapping(raw, label="trace benchmark checkpoint")
    if payload.get("schema") != TRACE_BENCHMARK_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported trace benchmark checkpoint schema")
    expected_identity = {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "model_id": model_id,
        "execution_id": execution_id,
        "method_ids": list(method_ids),
        "platform_revision": platform_revision,
        "runtime_attestation": runtime_attestation.to_dict(),
        "run_spec": run_spec.to_dict(),
    }
    if any(payload.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("trace benchmark checkpoint identity mismatch")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or not all(
        isinstance(cell, Mapping) for cell in raw_cells
    ):
        raise ValueError("trace benchmark checkpoint cells are invalid")
    return TraceBenchmarkCheckpointV1(
        dataset_manifest_sha256=dataset_manifest_sha256,
        model_id=model_id,
        execution_id=execution_id,
        method_ids=method_ids,
        platform_revision=platform_revision,
        runtime_attestation=runtime_attestation,
        run_spec=run_spec,
        cells=tuple(TraceBenchmarkCellV1.from_dict(cell) for cell in raw_cells),
    )


__all__ = [
    "DEFAULT_TEXT_TRACE_METHOD_IDS",
    "HIGH_PROCESS_SCORE_ON_FAILURE_THRESHOLD",
    "TRACE_BENCHMARK_CHECKPOINT_SCHEMA",
    "LEGACY_TRACE_BENCHMARK_RESULT_SCHEMA",
    "PREVIOUS_TRACE_BENCHMARK_RESULT_SCHEMA",
    "TRACE_BENCHMARK_MANIFEST_SCHEMA",
    "TRACE_BENCHMARK_RESULT_SCHEMA",
    "WHO_WHEN_SELECTION_ALGORITHM",
    "TraceBenchmarkAggregateV1",
    "TraceBenchmarkCaseV1",
    "TraceBenchmarkCheckpointV1",
    "TraceBenchmarkDatasetV1",
    "TraceBenchmarkGoldV1",
    "TraceBenchmarkFindingSummaryV1",
    "TraceBenchmarkResultV1",
    "TraceBenchmarkRuntimeAttestationV1",
    "TraceBenchmarkRunSpecV1",
    "TraceBenchmarkScoreV1",
    "attest_openai_model_endpoint",
    "build_trace_benchmark_run_spec",
    "load_trace_benchmark_checkpoint",
    "load_trace_benchmark_manifest",
    "run_trace_benchmark",
    "score_trace_method_receipt",
    "write_trace_benchmark_checkpoint",
    "write_trace_benchmark_result",
]
