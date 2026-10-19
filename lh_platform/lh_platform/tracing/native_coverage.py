"""Prompt-free audit of catalog, source and native-adapter coverage."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.integrations._provenance import require_clean_source_revision
from lh_platform.tracing.unified import TraceMethodDescriptorV1, TraceMethodPlatform

TRACE_NATIVE_COVERAGE_SCHEMA = "lh_trace_native_coverage/v1"


@dataclass(frozen=True)
class NativeTraceCoverageRowV1:
    method_id: str
    corpus_method_id: str
    input_modalities: tuple[str, ...]
    output_kinds: tuple[str, ...]
    source_relpath: str | None
    observed_source_revision: str | None
    admitted_source_revision: str | None
    integration_level: str
    execution_path: str
    admission_status: str
    blocker: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "corpus_method_id": self.corpus_method_id,
            "input_modalities": list(self.input_modalities),
            "output_kinds": list(self.output_kinds),
            "source_relpath": self.source_relpath,
            "observed_source_revision": self.observed_source_revision,
            "admitted_source_revision": self.admitted_source_revision,
            "integration_level": self.integration_level,
            "execution_path": self.execution_path,
            "admission_status": self.admission_status,
            "blocker": self.blocker,
        }


@dataclass(frozen=True)
class NativeTraceCoverageV1:
    method_count: int
    corpus_method_count: int
    status_counts: tuple[tuple[str, int], ...]
    methods: tuple[NativeTraceCoverageRowV1, ...]
    coverage_sha256: str
    schema: str = TRACE_NATIVE_COVERAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.method_count != len(self.methods):
            raise ValueError("trace coverage method count mismatch")
        if self.corpus_method_count != len(
            {row.corpus_method_id for row in self.methods}
        ):
            raise ValueError("trace coverage corpus method count mismatch")
        expected_counts: dict[str, int] = {}
        for row in self.methods:
            expected_counts[row.admission_status] = (
                expected_counts.get(row.admission_status, 0) + 1
            )
        if self.status_counts != tuple(sorted(expected_counts.items())):
            raise ValueError("trace coverage status counts mismatch")
        expected_hash = "sha256:" + canonical_sha256(
            self.to_dict(include_sha256=False)
        )
        if self.coverage_sha256 != expected_hash:
            raise ValueError("trace coverage hash mismatch")

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "method_count": self.method_count,
            "corpus_method_count": self.corpus_method_count,
            "status_counts": dict(self.status_counts),
            "methods": [row.to_dict() for row in self.methods],
        }
        if include_sha256:
            payload["coverage_sha256"] = self.coverage_sha256
        return payload


def _source_observation(
    *,
    corpus_root: Path,
    descriptor: TraceMethodDescriptorV1,
) -> tuple[str, str | None, str | None]:
    if descriptor.source_relpath is None:
        return "paper_only", None, descriptor.unavailable_reason
    source = (corpus_root / descriptor.source_relpath).resolve()
    if not source.is_relative_to(corpus_root):
        return "source_path_invalid", None, "source path escaped corpus root"
    if not source.is_dir():
        return "source_missing", None, f"source directory is missing: {source}"
    try:
        revision = require_clean_source_revision(
            source,
            method_name=descriptor.corpus_method_id or descriptor.method_id,
        )
    except ValueError as error:
        message = str(error)
        status = "source_dirty" if " dirty:" in message else "source_unversioned"
        return status, None, message
    except (OSError, subprocess.SubprocessError) as error:
        return "source_unversioned", None, f"source inspection failed: {error}"
    return "source_clean", revision, None


def _coverage_row(
    *,
    corpus_root: Path,
    descriptor: TraceMethodDescriptorV1,
) -> NativeTraceCoverageRowV1:
    source_status, observed_revision, source_blocker = _source_observation(
        corpus_root=corpus_root,
        descriptor=descriptor,
    )
    if descriptor.source_revision is not None:
        if source_status != "source_clean":
            status = "native_provenance_invalid"
            blocker = source_blocker or "admitted source is no longer clean"
        elif descriptor.source_revision != observed_revision:
            status = "native_provenance_invalid"
            blocker = "admitted and observed source revisions differ"
        else:
            status = "native_admitted"
            blocker = None
    elif source_status != "source_clean":
        status = source_status
        blocker = source_blocker
    elif (descriptor.unavailable_reason or "").startswith("native admission failed:"):
        status = "native_admission_failed"
        blocker = descriptor.unavailable_reason
    else:
        status = "adapter_pending"
        blocker = descriptor.unavailable_reason
    return NativeTraceCoverageRowV1(
        method_id=descriptor.method_id,
        corpus_method_id=descriptor.corpus_method_id or descriptor.method_id,
        input_modalities=descriptor.input_modalities,
        output_kinds=tuple(kind.value for kind in descriptor.output_kinds),
        source_relpath=descriptor.source_relpath,
        observed_source_revision=observed_revision,
        admitted_source_revision=descriptor.source_revision,
        integration_level=descriptor.integration_level,
        execution_path=descriptor.execution_path,
        admission_status=status,
        blocker=blocker,
    )


def build_native_trace_coverage(
    *,
    corpus_root: str | Path,
    platform: TraceMethodPlatform,
) -> NativeTraceCoverageV1:
    root = Path(corpus_root).resolve()
    rows = tuple(
        _coverage_row(corpus_root=root, descriptor=descriptor)
        for descriptor in platform.catalog()
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.admission_status] = counts.get(row.admission_status, 0) + 1
    unhashed: dict[str, object] = {
        "schema": TRACE_NATIVE_COVERAGE_SCHEMA,
        "method_count": len(rows),
        "corpus_method_count": len({row.corpus_method_id for row in rows}),
        "status_counts": dict(sorted(counts.items())),
        "methods": [row.to_dict() for row in rows],
    }
    return NativeTraceCoverageV1(
        method_count=len(rows),
        corpus_method_count=len({row.corpus_method_id for row in rows}),
        status_counts=tuple(sorted(counts.items())),
        methods=rows,
        coverage_sha256="sha256:" + canonical_sha256(unhashed),
    )


def write_native_trace_coverage(
    report: NativeTraceCoverageV1,
    output_path: str | Path,
) -> None:
    atomic_write_json(output_path, report.to_dict(), overwrite=False)


__all__ = [
    "TRACE_NATIVE_COVERAGE_SCHEMA",
    "NativeTraceCoverageRowV1",
    "NativeTraceCoverageV1",
    "build_native_trace_coverage",
    "write_native_trace_coverage",
]
