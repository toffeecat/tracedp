"""Fail-closed public and scorer dataset loaders for TraceFix-Bench v0.1.

The public loader exposes only the public index and lazily decoded public
packets.  The scorer loader derives its private root from a bound release
record, performs a streaming whole-population preflight, and revalidates each
case when it is consumed.  This module keeps no run or consumption progress.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import lh_platform.tracefix_bench.serialization as _serialization
from lh_platform.tracefix_bench.contracts import (
    AttributionGoldV1,
    BenchmarkCaseRefV1,
    BenchmarkSplit,
    CaseAdmissionV1,
    ExposureStatus,
    MaterializationStatus,
    PopulationRole,
    PublicTracePacketV1,
    PublicTraceProjectionReceiptV1,
    SourceDenominatorEntryV1,
    TraceFixTrack,
)
from lh_platform.tracefix_bench.dataset_contracts import (
    PUBLIC_DATASET_INDEX_SCHEMA,
    SCORER_DATASET_INDEX_SCHEMA,
    TRACEFIX_BENCH_SUITE_ID,
    PublicCaseIndexEntryV1,
    PublicDatasetIndexV1,
    ScorerCaseIndexEntryV1,
    ScorerDatasetIndexV1,
    TraceFixReleaseRecordV1,
)
from lh_platform.tracefix_bench.scorer_contracts import (
    SCORER_REGISTRY_FILENAME,
    ScorerRegistryV1,
)

_PREFLIGHT_COMPLETE = "complete"
_RELEASE_RECORD_DIRECTORY = "release_records"
_RELEASE_RECORD_SUFFIX = ".release.json"
_RELEASE_NAME_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
)
_VALIDATED_DATASET_TOKEN = object()


class TraceFixDatasetValidationCode(StrEnum):
    """Classify exact-tree failures without exposing file contents."""

    ROOT_UNAVAILABLE_OR_INVALID = "root_unavailable_or_invalid"
    MISSING_REGISTERED_ENTRY = "missing_registered_entry"
    UNREGISTERED_ENTRY = "unregistered_entry"
    REGISTERED_ENTRY_INACCESSIBLE = "registered_entry_inaccessible"
    REGISTERED_ENTRY_WRONG_TYPE = "registered_entry_wrong_type"
    DIRECTORY_LISTING_FAILED = "directory_listing_failed"


class TraceFixDatasetValidationError(ValueError):
    """Preserve one offending path and a closed validation category."""

    def __init__(
        self,
        *,
        evidence_code: TraceFixDatasetValidationCode,
        offending_path: Path,
        cause_errno: int | None = None,
    ) -> None:
        if not isinstance(evidence_code, TraceFixDatasetValidationCode):
            raise ValueError("dataset validation error requires a closed evidence code")
        if not isinstance(offending_path, Path) or not offending_path.is_absolute():
            raise ValueError("dataset validation error requires an absolute path")
        if cause_errno is not None and type(cause_errno) is not int:
            raise ValueError("dataset validation cause_errno must be an integer")
        self.evidence_code = evidence_code
        self.offending_path = offending_path
        self.cause_errno = cause_errno
        super().__init__(f"TraceFix dataset validation failed: {evidence_code.value}")


@dataclass(frozen=True)
class PublicTraceFixDatasetV1:
    """Hold a validated public root and an O(1) opaque case lookup."""

    canonical_public_root: Path
    index: PublicDatasetIndexV1
    _validation_token: object = field(repr=False, compare=False)
    _case_entries_by_id: Mapping[str, PublicCaseIndexEntryV1] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_token is not _VALIDATED_DATASET_TOKEN:
            raise ValueError("public dataset must be created by load_public_dataset")
        if not isinstance(self.canonical_public_root, Path):
            raise ValueError("canonical_public_root must be a Path")
        if not isinstance(self.index, PublicDatasetIndexV1):
            raise ValueError("public dataset index has the wrong contract type")
        object.__setattr__(
            self,
            "_case_entries_by_id",
            MappingProxyType({entry.case_id: entry for entry in self.index.cases}),
        )

    def case_entry(self, case_id: str) -> PublicCaseIndexEntryV1:
        """Return one exact-case index entry without scanning the roster."""

        if type(case_id) is not str:
            raise ValueError("public case_id must be text")
        try:
            return self._case_entries_by_id[case_id]
        except KeyError:
            raise ValueError("public case_id is not in the frozen roster") from None


@dataclass(frozen=True)
class ScorerCaseBundleV1:
    """Group one revalidated scorer-only case bundle without public mutation."""

    case_id: str
    case_ref: BenchmarkCaseRefV1
    admission: CaseAdmissionV1
    projection_receipt: PublicTraceProjectionReceiptV1
    attribution_gold: AttributionGoldV1 | None

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("scorer bundle case_id must be non-empty text")
        if self.case_ref.case_id != self.case_id:
            raise ValueError("scorer bundle case reference does not match case_id")
        if self.admission.case_ref != self.case_ref:
            raise ValueError("scorer bundle admission does not match case reference")
        if self.projection_receipt.case_id != self.case_id:
            raise ValueError("scorer bundle projection does not match case_id")
        if (
            self.attribution_gold is not None
            and self.attribution_gold.case_ref != self.case_ref
        ):
            raise ValueError("scorer bundle gold does not match case reference")


@dataclass(frozen=True)
class TraceFixPopulationPreflightReceiptV1:
    """Record a complete population closure check without caching case payloads."""

    suite_id: str
    suite_version: str
    population_id: str
    public_index_schema: str
    scorer_index_schema: str
    scorer_registry_id: str
    case_count: int
    case_ids: tuple[str, ...]
    source_denominator_count: int
    source_denominator_entry_ids: tuple[str, ...]
    preflight_status: str = _PREFLIGHT_COMPLETE

    def __post_init__(self) -> None:
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID:
            raise ValueError("preflight suite_id is not TraceFix-Bench")
        if type(self.suite_version) is not str or not self.suite_version:
            raise ValueError("preflight suite_version must be non-empty text")
        if type(self.population_id) is not str or not self.population_id:
            raise ValueError("preflight population_id must be non-empty text")
        if type(self.scorer_registry_id) is not str or not self.scorer_registry_id:
            raise ValueError("preflight scorer_registry_id must be non-empty text")
        if self.public_index_schema != PUBLIC_DATASET_INDEX_SCHEMA:
            raise ValueError("preflight public index schema changed")
        if self.scorer_index_schema != SCORER_DATASET_INDEX_SCHEMA:
            raise ValueError("preflight scorer index schema changed")
        if (
            isinstance(self.case_count, bool)
            or not isinstance(self.case_count, int)
            or self.case_count < 0
            or self.case_count != len(self.case_ids)
        ):
            raise ValueError("preflight case count does not match its roster")
        if (
            isinstance(self.source_denominator_count, bool)
            or not isinstance(self.source_denominator_count, int)
            or self.source_denominator_count < 0
            or self.source_denominator_count
            != len(self.source_denominator_entry_ids)
        ):
            raise ValueError("preflight denominator count does not match its roster")
        if self.preflight_status != _PREFLIGHT_COMPLETE:
            raise ValueError("preflight receipt must describe complete validation")
        if len(self.case_ids) != len({value.casefold() for value in self.case_ids}):
            raise ValueError("preflight case roster must be unique after case folding")
        denominator_ids = self.source_denominator_entry_ids
        if len(denominator_ids) != len(
            {value.casefold() for value in denominator_ids}
        ):
            raise ValueError(
                "preflight denominator roster must be unique after case folding"
            )
        if denominator_ids != tuple(sorted(denominator_ids, key=str.casefold)):
            raise ValueError("preflight denominator roster must be canonically ordered")


@dataclass(frozen=True)
class ScorerTraceFixDatasetV1:
    """Hold one bound scorer dataset after mandatory population preflight.

    The retained denominator snapshot contains immutable typed denominator rows,
    not scorer case bundles.  It lets later case reads reject any population
    census drift while keeping memory proportional to denominator metadata.
    """

    canonical_scorer_root: Path
    canonical_release_record_path: Path
    public_dataset: PublicTraceFixDatasetV1
    index: ScorerDatasetIndexV1
    registry: ScorerRegistryV1
    release_record: TraceFixReleaseRecordV1
    preflight_receipt: TraceFixPopulationPreflightReceiptV1
    _case_source_bindings: tuple[tuple[str, str], ...] = field(repr=False)
    _denominator_snapshot: tuple[SourceDenominatorEntryV1, ...] = field(
        repr=False,
        compare=False,
    )
    _validation_token: object = field(repr=False, compare=False)
    _case_entries_by_id: Mapping[str, ScorerCaseIndexEntryV1] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_token is not _VALIDATED_DATASET_TOKEN:
            raise ValueError("scorer dataset must be created by load_scorer_dataset")
        if not isinstance(self.canonical_scorer_root, Path):
            raise ValueError("canonical_scorer_root must be a Path")
        if not isinstance(self.canonical_release_record_path, Path):
            raise ValueError("canonical_release_record_path must be a Path")
        if not isinstance(self.public_dataset, PublicTraceFixDatasetV1):
            raise ValueError("scorer dataset requires a public dataset")
        if not isinstance(self.index, ScorerDatasetIndexV1):
            raise ValueError("scorer dataset index has the wrong contract type")
        if not isinstance(self.registry, ScorerRegistryV1):
            raise ValueError("scorer dataset registry has the wrong contract type")
        _validate_registry_binding(self.registry, self.index)
        if not isinstance(self.release_record, TraceFixReleaseRecordV1):
            raise ValueError("scorer dataset release record has the wrong type")
        if not isinstance(
            self.preflight_receipt,
            TraceFixPopulationPreflightReceiptV1,
        ):
            raise ValueError("scorer dataset requires a preflight receipt")
        if self.index.case_ids != self.public_dataset.index.case_ids:
            raise ValueError("scorer and public dataset rosters must match")
        if (
            self.preflight_receipt.suite_id != self.index.suite_id
            or self.preflight_receipt.suite_version != self.index.suite_version
            or self.preflight_receipt.population_id != self.index.population_id
            or self.preflight_receipt.scorer_registry_id != self.index.scorer_registry_id
            or self.preflight_receipt.case_ids != self.index.case_ids
            or self.preflight_receipt.source_denominator_entry_ids
            != self.index.source_denominator_entry_ids
        ):
            raise ValueError("scorer index does not match its preflight receipt")
        if (
            self.release_record.suite_id != self.preflight_receipt.suite_id
            or self.release_record.suite_version
            != self.preflight_receipt.suite_version
            or self.release_record.population_id
            != self.preflight_receipt.population_id
            or self.release_record.scorer_registry_id
            != self.preflight_receipt.scorer_registry_id
            or self.release_record.case_ids != self.preflight_receipt.case_ids
            or self.release_record.source_denominator_entry_ids
            != self.preflight_receipt.source_denominator_entry_ids
        ):
            raise ValueError("release record does not match its preflight receipt")
        if (
            self.preflight_receipt.case_count == 0
            or self.preflight_receipt.source_denominator_count == 0
        ):
            raise ValueError("a bound scorer dataset must be non-empty")
        bindings = dict(self._case_source_bindings)
        if len(bindings) != len(self._case_source_bindings):
            raise ValueError("scorer case-to-source bindings must be unique")
        if set(bindings) != set(self.index.case_ids):
            raise ValueError("scorer case-to-source bindings must cover every case")
        if tuple(case_id for case_id, _ in self._case_source_bindings) != tuple(
            self.index.case_ids
        ):
            raise ValueError("scorer case-to-source bindings must use case order")
        if not set(bindings.values()).issubset(
            set(self.index.source_denominator_entry_ids)
        ):
            raise ValueError("scorer case-to-source bindings contain an unknown source")
        if not isinstance(self._denominator_snapshot, tuple):
            raise ValueError("scorer denominator snapshot must be a tuple")
        if len(self._denominator_snapshot) != self.index.source_denominator_count:
            raise ValueError("scorer denominator snapshot count changed")
        if not all(
            isinstance(entry, SourceDenominatorEntryV1)
            for entry in self._denominator_snapshot
        ):
            raise ValueError("scorer denominator snapshot has the wrong row type")
        if tuple(entry.entry_id for entry in self._denominator_snapshot) != tuple(
            self.index.source_denominator_entry_ids
        ):
            raise ValueError("scorer denominator snapshot roster changed")
        if any(
            entry.population_id != self.index.population_id
            for entry in self._denominator_snapshot
        ):
            raise ValueError("scorer denominator snapshot population changed")
        snapshot_bindings: dict[str, str] = {}
        for entry in self._denominator_snapshot:
            if entry.materialization_status is MaterializationStatus.ADMITTED:
                for case_id in entry.materialized_case_ids:
                    if case_id in snapshot_bindings:
                        raise ValueError(
                            "scorer denominator snapshot materializes a case twice"
                        )
                    snapshot_bindings[case_id] = entry.entry_id
        if tuple(
            (case_id, snapshot_bindings.get(case_id))
            for case_id in self.index.case_ids
        ) != self._case_source_bindings:
            raise ValueError(
                "scorer denominator snapshot does not match case bindings"
            )
        if set(snapshot_bindings) != set(self.index.case_ids):
            raise ValueError("scorer denominator snapshot does not cover every case")
        object.__setattr__(
            self,
            "_case_entries_by_id",
            MappingProxyType({entry.case_id: entry for entry in self.index.cases}),
        )

    def case_entry(self, case_id: str) -> ScorerCaseIndexEntryV1:
        """Return one exact scorer index entry without scanning the roster."""

        if type(case_id) is not str:
            raise ValueError("scorer case_id must be text")
        try:
            return self._case_entries_by_id[case_id]
        except KeyError:
            raise ValueError("scorer case_id is not in the bound roster") from None


@dataclass(frozen=True)
class _SourceClosureV1:
    """Keep only the denominator fields needed after one row is discarded."""

    benchmark_id: str
    benchmark_version: str
    adapter_version: str
    source_task_id: str
    task_group_id: str
    source_artifact_ref: str
    split: BenchmarkSplit
    population_role: PopulationRole
    exposure_status: ExposureStatus


@dataclass(frozen=True)
class _PreflightStateV1:
    """Return population metadata without retaining scorer case bundles."""

    receipt: TraceFixPopulationPreflightReceiptV1
    scorer_index: ScorerDatasetIndexV1
    scorer_registry: ScorerRegistryV1
    case_source_bindings: tuple[tuple[str, str], ...]
    denominator_snapshot: tuple[SourceDenominatorEntryV1, ...]


def _require_canonical_absolute_path(path: Path, *, label: str) -> Path:
    """Reject aliases before any no-follow filesystem traversal."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{label} must be an absolute Path")
    raw_path = os.fspath(path)
    if raw_path.startswith("//"):
        raise ValueError(f"{label} must start at the single POSIX root")
    normalized = Path(os.path.normpath(raw_path))
    if normalized != path or ".." in path.parts:
        raise ValueError(f"{label} must use a normalized absolute path")
    return path


def _directory_open_flags() -> int:
    """Build flags that reject symbolic links and non-directory targets."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise RuntimeError("this platform lacks required no-follow directory access")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_absolute_directory(path: Path, *, label: str) -> int:
    """Open every absolute path component without following symbolic links."""

    canonical = _require_canonical_absolute_path(path, label=label)
    flags = _directory_open_flags()
    try:
        current_fd = os.open(os.sep, flags)
    except OSError:
        raise ValueError(f"{label} root is not an accessible real directory") from None
    try:
        for component in canonical.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError:
                raise ValueError(
                    f"{label} is not an accessible real directory"
                ) from None
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise ValueError(f"{label} is not a real directory")
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _require_exact_names(
    actual: set[str],
    expected: set[str],
    *,
    parent: Path,
    label: str,
) -> None:
    """Fail closed on both missing and unregistered directory entries."""

    missing = expected - actual
    if missing:
        missing_name = sorted(missing, key=str.casefold)[0]
        raise TraceFixDatasetValidationError(
            evidence_code=TraceFixDatasetValidationCode.MISSING_REGISTERED_ENTRY,
            offending_path=parent / missing_name,
        )
    extra = actual - expected
    if extra:
        extra_name = sorted(extra, key=str.casefold)[0]
        raise TraceFixDatasetValidationError(
            evidence_code=TraceFixDatasetValidationCode.UNREGISTERED_ENTRY,
            offending_path=parent / extra_name,
        )


def _validate_exact_tree(
    root: Path,
    *,
    root_files: set[str],
    directory_files: Mapping[str, set[str]],
    label: str,
) -> None:
    """Validate a two-level immutable tree without metadata-file exceptions."""

    try:
        root_fd = _open_absolute_directory(root, label=label)
    except Exception as error:
        cause_errno = getattr(error, "errno", None)
        if type(cause_errno) is not int:
            cause_errno = None
        raise TraceFixDatasetValidationError(
            evidence_code=(
                TraceFixDatasetValidationCode.ROOT_UNAVAILABLE_OR_INVALID
            ),
            offending_path=root,
            cause_errno=cause_errno,
        ) from None
    try:
        try:
            root_entries = set(os.listdir(root_fd))
        except OSError as error:
            raise TraceFixDatasetValidationError(
                evidence_code=TraceFixDatasetValidationCode.DIRECTORY_LISTING_FAILED,
                offending_path=root,
                cause_errno=error.errno,
            ) from None
        _require_exact_names(
            root_entries,
            root_files | set(directory_files),
            parent=root,
            label=label,
        )
        for filename in root_files:
            file_path = root / filename
            try:
                mode = os.stat(filename, dir_fd=root_fd, follow_symlinks=False).st_mode
            except OSError as error:
                raise TraceFixDatasetValidationError(
                    evidence_code=(
                        TraceFixDatasetValidationCode.REGISTERED_ENTRY_INACCESSIBLE
                    ),
                    offending_path=file_path,
                    cause_errno=error.errno,
                ) from None
            if not stat.S_ISREG(mode):
                raise TraceFixDatasetValidationError(
                    evidence_code=TraceFixDatasetValidationCode.REGISTERED_ENTRY_WRONG_TYPE,
                    offending_path=file_path,
                )
        for directory_name, expected_files in directory_files.items():
            directory_path = root / directory_name
            try:
                mode = os.stat(
                    directory_name,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                ).st_mode
            except OSError as error:
                raise TraceFixDatasetValidationError(
                    evidence_code=(
                        TraceFixDatasetValidationCode.REGISTERED_ENTRY_INACCESSIBLE
                    ),
                    offending_path=directory_path,
                    cause_errno=error.errno,
                ) from None
            if not stat.S_ISDIR(mode):
                raise TraceFixDatasetValidationError(
                    evidence_code=TraceFixDatasetValidationCode.REGISTERED_ENTRY_WRONG_TYPE,
                    offending_path=directory_path,
                )
            try:
                directory_fd = os.open(
                    directory_name,
                    _directory_open_flags(),
                    dir_fd=root_fd,
                )
            except OSError as error:
                raise TraceFixDatasetValidationError(
                    evidence_code=(
                        TraceFixDatasetValidationCode.REGISTERED_ENTRY_INACCESSIBLE
                    ),
                    offending_path=directory_path,
                    cause_errno=error.errno,
                ) from None
            try:
                try:
                    directory_entries = set(os.listdir(directory_fd))
                except OSError as error:
                    raise TraceFixDatasetValidationError(
                        evidence_code=(
                            TraceFixDatasetValidationCode.DIRECTORY_LISTING_FAILED
                        ),
                        offending_path=directory_path,
                        cause_errno=error.errno,
                    ) from None
                _require_exact_names(
                    directory_entries,
                    expected_files,
                    parent=directory_path,
                    label=f"{label} {directory_name}",
                )
                for filename in expected_files:
                    file_path = directory_path / filename
                    try:
                        file_mode = os.stat(
                            filename,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        ).st_mode
                    except OSError as error:
                        raise TraceFixDatasetValidationError(
                            evidence_code=(
                                TraceFixDatasetValidationCode.REGISTERED_ENTRY_INACCESSIBLE
                            ),
                            offending_path=file_path,
                            cause_errno=error.errno,
                        ) from None
                    if not stat.S_ISREG(file_mode):
                        raise TraceFixDatasetValidationError(
                            evidence_code=(
                                TraceFixDatasetValidationCode.REGISTERED_ENTRY_WRONG_TYPE
                            ),
                            offending_path=file_path,
                        )
            finally:
                os.close(directory_fd)
    finally:
        os.close(root_fd)


def _public_tree_files(index: PublicDatasetIndexV1) -> dict[str, set[str]]:
    """Build the exact public tree roster from closed index paths."""

    return {"cases": {Path(entry.relative_path).name for entry in index.cases}}


def _scorer_tree_files(index: ScorerDatasetIndexV1) -> dict[str, set[str]]:
    """Build the exact scorer tree roster from closed index paths."""

    gold_files = {
        Path(entry.attribution_gold_path).name
        for entry in index.cases
        if entry.attribution_gold_path is not None
    }
    return {
        "case_refs": {Path(entry.case_ref_path).name for entry in index.cases},
        "admissions": {Path(entry.admission_path).name for entry in index.cases},
        "projection_receipts": {
            Path(entry.projection_receipt_path).name for entry in index.cases
        },
        "attribution_gold": gold_files,
        "source_denominator": {
            Path(entry.relative_path).name for entry in index.source_denominator
        },
    }


def _load_public_index(public_root: Path) -> PublicDatasetIndexV1:
    """Load the single root-level public index through the strict decoder."""

    payload = _serialization.load_json_object_strict(
        public_root / "public_index.json",
        label="TraceFix public index",
    )
    return _serialization.decode_public_index_v1(payload)


def _load_scorer_index(scorer_root: Path) -> ScorerDatasetIndexV1:
    """Load the single root-level scorer index through the strict decoder."""

    payload = _serialization.load_json_object_strict(
        scorer_root / "scorer_index.json",
        label="TraceFix scorer index",
    )
    return _serialization.decode_scorer_index_v1(payload)


def _validate_registry_binding(
    registry: ScorerRegistryV1,
    index: ScorerDatasetIndexV1,
) -> None:
    """Bind scoring policy to the exact immutable population identity."""

    if (
        registry.registry_id != index.scorer_registry_id
        or registry.suite_id != index.suite_id
        or registry.suite_version != index.suite_version
        or registry.population_id != index.population_id
    ):
        raise ValueError("scorer registry does not match its population index")


def _load_scorer_registry(
    scorer_root: Path,
    index: ScorerDatasetIndexV1,
) -> ScorerRegistryV1:
    """Decode only the fixed registry file and close its index binding."""

    registry = _serialization.decode_scorer_registry_v1(
        _serialization.load_json_object_strict(
            scorer_root / SCORER_REGISTRY_FILENAME,
            label="TraceFix scorer registry",
        )
    )
    _validate_registry_binding(registry, index)
    return registry


def _load_registered(
    root: Path,
    relative_path: str,
    *,
    directory: str,
    identifier: str,
    label: str,
) -> dict[str, object]:
    """Read one exact registered JSON file through the no-follow boundary."""

    return _serialization.load_registered_json_object_strict(
        root,
        relative_path,
        expected_directory=directory,
        expected_identifier=identifier,
        label=label,
    )


def load_public_dataset(public_root: Path) -> PublicTraceFixDatasetV1:
    """Load only the public index and validate its complete on-disk tree."""

    canonical_root = _require_canonical_absolute_path(
        public_root,
        label="TraceFix public root",
    )
    root_fd = _open_absolute_directory(canonical_root, label="TraceFix public root")
    os.close(root_fd)
    index = _load_public_index(canonical_root)
    _validate_exact_tree(
        canonical_root,
        root_files={"public_index.json"},
        directory_files=_public_tree_files(index),
        label="TraceFix public tree",
    )
    return PublicTraceFixDatasetV1(
        canonical_public_root=canonical_root,
        index=index,
        _validation_token=_VALIDATED_DATASET_TOKEN,
    )


def _require_loaded_public_dataset(dataset: object) -> PublicTraceFixDatasetV1:
    """Accept only a dataset carrying this module's loader provenance token."""

    if (
        not isinstance(dataset, PublicTraceFixDatasetV1)
        or dataset._validation_token is not _VALIDATED_DATASET_TOKEN
    ):
        raise ValueError("operation requires a loader-validated public dataset")
    return dataset


def _decode_public_case(
    dataset: PublicTraceFixDatasetV1,
    entry: PublicCaseIndexEntryV1,
) -> PublicTracePacketV1:
    """Decode one public packet after its surrounding tree has been validated."""

    payload = _load_registered(
        dataset.canonical_public_root,
        entry.relative_path,
        directory="cases",
        identifier=entry.case_id,
        label="TraceFix public case",
    )
    packet = _serialization.decode_public_trace_packet_v1(payload)
    if packet.case_id != entry.case_id:
        raise ValueError("public packet case_id does not match its index")
    if packet.track is not entry.track:
        raise ValueError("public packet track does not match its index")
    if packet.execution_semantics is not entry.execution_semantics:
        raise ValueError("public packet execution semantics do not match its index")
    return packet


def load_public_case(
    dataset: PublicTraceFixDatasetV1,
    case_id: str,
) -> PublicTracePacketV1:
    """Revalidate the public tree, then lazily decode one indexed packet."""

    loaded_dataset = _require_loaded_public_dataset(dataset)
    current_index = _load_public_index(loaded_dataset.canonical_public_root)
    if current_index != loaded_dataset.index:
        raise ValueError("public index changed after the dataset was loaded")
    _validate_exact_tree(
        loaded_dataset.canonical_public_root,
        root_files={"public_index.json"},
        directory_files=_public_tree_files(current_index),
        label="TraceFix public tree",
    )
    entry = loaded_dataset.case_entry(case_id)
    return _decode_public_case(loaded_dataset, entry)


def preflight_public_dataset(dataset: PublicTraceFixDatasetV1) -> None:
    """Strictly decode every public case after one current-tree validation."""

    loaded_dataset = _require_loaded_public_dataset(dataset)
    current_index = _load_public_index(loaded_dataset.canonical_public_root)
    if current_index != loaded_dataset.index:
        raise ValueError("public index changed after the dataset was loaded")
    _validate_exact_tree(
        loaded_dataset.canonical_public_root,
        root_files={"public_index.json"},
        directory_files=_public_tree_files(current_index),
        label="TraceFix public tree",
    )
    for entry in current_index.cases:
        _decode_public_case(loaded_dataset, entry)


def _paths_overlap(first: Path, second: Path) -> bool:
    """Detect equal or nested dataset roots without resolving symbolic links."""

    if first == second:
        return True
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _source_identity(entry: SourceDenominatorEntryV1) -> tuple[str, ...]:
    """Build the registered composite source identity for duplicate detection."""

    return (
        entry.benchmark_id,
        entry.benchmark_version,
        entry.adapter_version,
        entry.population_id,
        entry.source_task_id,
    )


def _source_closure(entry: SourceDenominatorEntryV1) -> _SourceClosureV1:
    """Retain only fields needed to bind future case references."""

    return _SourceClosureV1(
        benchmark_id=entry.benchmark_id,
        benchmark_version=entry.benchmark_version,
        adapter_version=entry.adapter_version,
        source_task_id=entry.source_task_id,
        task_group_id=entry.task_group_id,
        source_artifact_ref=entry.source_artifact_ref,
        split=entry.split,
        population_role=entry.population_role,
        exposure_status=entry.exposure_status,
    )


def _case_source_closure(case_ref: BenchmarkCaseRefV1) -> _SourceClosureV1:
    """Project a case reference onto the denominator identity fields."""

    return _SourceClosureV1(
        benchmark_id=case_ref.benchmark_id,
        benchmark_version=case_ref.benchmark_version,
        adapter_version=case_ref.adapter_version,
        source_task_id=case_ref.source_task_id,
        task_group_id=case_ref.task_group_id,
        source_artifact_ref=case_ref.source_artifact_ref,
        split=case_ref.split,
        population_role=case_ref.population_role,
        exposure_status=case_ref.exposure_status,
    )


def _load_denominator_closure(
    *,
    scorer_root: Path,
    scorer_index: ScorerDatasetIndexV1,
    expected_population_id: str,
    public_case_ids: tuple[str, ...],
) -> tuple[
    dict[str, _SourceClosureV1],
    dict[str, str],
    tuple[SourceDenominatorEntryV1, ...],
]:
    """Stream denominator rows into closure maps and an immutable typed snapshot."""

    public_case_id_set = set(public_case_ids)
    source_closure_by_id: dict[str, _SourceClosureV1] = {}
    case_to_source_entry_id: dict[str, str] = {}
    observed_source_identities: set[tuple[str, ...]] = set()
    denominator_snapshot: list[SourceDenominatorEntryV1] = []
    for index_entry in scorer_index.source_denominator:
        payload = _load_registered(
            scorer_root,
            index_entry.relative_path,
            directory="source_denominator",
            identifier=index_entry.entry_id,
            label="TraceFix source denominator row",
        )
        source_entry = _serialization.decode_source_denominator_entry_v1(payload)
        if source_entry.entry_id != index_entry.entry_id:
            raise ValueError("source denominator entry_id does not match its index")
        if source_entry.population_id != expected_population_id:
            raise ValueError("source denominator population_id changed")
        denominator_snapshot.append(source_entry)
        source_identity = _source_identity(source_entry)
        if source_identity in observed_source_identities:
            raise ValueError("source denominator contains a duplicate source identity")
        observed_source_identities.add(source_identity)
        source_closure_by_id[source_entry.entry_id] = _source_closure(source_entry)
        if source_entry.materialization_status is MaterializationStatus.ADMITTED:
            for case_id in source_entry.materialized_case_ids:
                if case_id not in public_case_id_set:
                    raise ValueError(
                        "source denominator materializes an unregistered case"
                    )
                if case_id in case_to_source_entry_id:
                    raise ValueError(
                        "a case is materialized by multiple source denominator rows"
                    )
                case_to_source_entry_id[case_id] = source_entry.entry_id
        elif source_entry.materialized_case_ids:
            raise ValueError(
                "a non-admitted source denominator row materializes a case"
            )
    if set(case_to_source_entry_id) != public_case_id_set:
        raise ValueError(
            "source denominator materialization does not cover the case roster"
        )
    return (
        source_closure_by_id,
        case_to_source_entry_id,
        tuple(denominator_snapshot),
    )


def _decode_scorer_bundle(
    *,
    scorer_root: Path,
    scorer_entry: ScorerCaseIndexEntryV1,
) -> ScorerCaseBundleV1:
    """Decode one private bundle without interpreting cross-record semantics."""

    case_ref = _serialization.decode_benchmark_case_ref_v1(
        _load_registered(
            scorer_root,
            scorer_entry.case_ref_path,
            directory="case_refs",
            identifier=scorer_entry.case_id,
            label="TraceFix scorer case reference",
        )
    )
    admission = _serialization.decode_case_admission_v1(
        _load_registered(
            scorer_root,
            scorer_entry.admission_path,
            directory="admissions",
            identifier=scorer_entry.case_id,
            label="TraceFix scorer admission",
        )
    )
    projection = _serialization.decode_projection_receipt_v1(
        _load_registered(
            scorer_root,
            scorer_entry.projection_receipt_path,
            directory="projection_receipts",
            identifier=scorer_entry.case_id,
            label="TraceFix scorer projection receipt",
        )
    )
    gold = None
    if scorer_entry.attribution_gold_path is not None:
        gold = _serialization.decode_attribution_gold_v1(
            _load_registered(
                scorer_root,
                scorer_entry.attribution_gold_path,
                directory="attribution_gold",
                identifier=scorer_entry.case_id,
                label="TraceFix scorer attribution gold",
            )
        )
    return ScorerCaseBundleV1(
        case_id=scorer_entry.case_id,
        case_ref=case_ref,
        admission=admission,
        projection_receipt=projection,
        attribution_gold=gold,
    )


def _validate_scorer_bundle(
    *,
    bundle: ScorerCaseBundleV1,
    scorer_entry: ScorerCaseIndexEntryV1,
    public_case: PublicTracePacketV1,
    public_entry: PublicCaseIndexEntryV1,
    source_closure_by_id: Mapping[str, _SourceClosureV1],
    case_to_source_entry_id: Mapping[str, str],
) -> None:
    """Close one public packet against all private case records."""

    case_ref = bundle.case_ref
    admission = bundle.admission
    projection = bundle.projection_receipt
    gold = bundle.attribution_gold
    if case_ref.case_id != public_case.case_id:
        raise ValueError("scorer case reference does not match the public case")
    if case_ref.case_ordinal != public_entry.case_ordinal:
        raise ValueError("scorer case ordinal does not match the public index")
    if public_case.track not in admission.eligible_tracks:
        raise ValueError("public case track is not eligible in its admission")
    expected_source_entry_id = case_to_source_entry_id.get(public_case.case_id)
    if expected_source_entry_id != case_ref.source_denominator_entry_id:
        raise ValueError(
            "case reference does not close to its materializing source row"
        )
    source_closure = source_closure_by_id.get(case_ref.source_denominator_entry_id)
    if source_closure is None:
        raise ValueError("case reference points to an unknown source denominator row")
    if _case_source_closure(case_ref) != source_closure:
        raise ValueError("case and source denominator identities do not match")
    if admission.case_ref != case_ref:
        raise ValueError("admission embeds a different case reference")
    if projection.case_id != public_case.case_id:
        raise ValueError("projection receipt case_id does not match the public case")
    if projection.projection_receipt_id != scorer_entry.projection_receipt_id:
        raise ValueError("scorer index points to a different projection receipt")
    if projection.benchmark_id != case_ref.benchmark_id:
        raise ValueError("projection benchmark identity does not match the case")
    if projection.adapter_version != case_ref.adapter_version:
        raise ValueError("projection adapter identity does not match the case")
    public_coverage = tuple(
        (item.field_name, item.status) for item in public_case.field_coverage
    )
    scorer_coverage = tuple(
        (item.field_name, item.status) for item in projection.field_decisions
    )
    if public_coverage != scorer_coverage:
        raise ValueError("public field coverage does not match projection decisions")
    if public_case.track is TraceFixTrack.STATIC_ATTRIBUTION and gold is None:
        raise ValueError("static attribution case requires explicit gold")
    if public_case.track is TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION:
        if gold is not None or admission.attribution_gold_disposition is not None:
            raise ValueError("historical causal case cannot carry attribution gold")
    elif gold is None:
        if admission.attribution_gold_disposition is not None:
            raise ValueError(
                "attribution gold disposition requires a registered gold record"
            )
    else:
        if admission.attribution_gold_disposition is None:
            raise ValueError("attribution gold record lacks an admission disposition")
        if gold.case_ref != case_ref:
            raise ValueError("attribution gold embeds a different case reference")
        if gold.disposition is not admission.attribution_gold_disposition:
            raise ValueError("attribution gold disposition differs from admission")
        for accepted_group in gold.accepted_event_groups:
            if not set(accepted_group).issubset(public_case.event_ids):
                raise ValueError("attribution gold references a non-public event")


def _preflight_tracefix_pair(
    *,
    public_dataset: PublicTraceFixDatasetV1,
    scorer_root: Path,
    expected_population_id: str,
    expected_suite_version: str,
) -> _PreflightStateV1:
    """Run the complete streaming preflight and retain only closure maps."""

    public_dataset = _require_loaded_public_dataset(public_dataset)
    if type(expected_population_id) is not str or not expected_population_id:
        raise ValueError("expected_population_id must be non-empty text")
    if type(expected_suite_version) is not str or not expected_suite_version:
        raise ValueError("expected_suite_version must be non-empty text")
    canonical_scorer_root = _require_canonical_absolute_path(
        scorer_root,
        label="TraceFix scorer root",
    )
    scorer_root_fd = _open_absolute_directory(
        canonical_scorer_root,
        label="TraceFix scorer root",
    )
    os.close(scorer_root_fd)
    if _paths_overlap(
        public_dataset.canonical_public_root,
        canonical_scorer_root,
    ):
        raise ValueError(
            "public and scorer roots must be separate non-nested directories"
        )

    current_public_index = _load_public_index(public_dataset.canonical_public_root)
    if current_public_index != public_dataset.index:
        raise ValueError("public index changed after the public dataset was loaded")
    scorer_index = _load_scorer_index(canonical_scorer_root)
    for label, value in (
        ("public population_id", current_public_index.population_id),
        ("scorer population_id", scorer_index.population_id),
    ):
        if value != expected_population_id:
            raise ValueError(f"{label} does not match the expected population")
    for label, value in (
        ("public suite_version", current_public_index.suite_version),
        ("scorer suite_version", scorer_index.suite_version),
    ):
        if value != expected_suite_version:
            raise ValueError(f"{label} does not match the expected version")
    if current_public_index.suite_id != scorer_index.suite_id:
        raise ValueError("public and scorer suite identities differ")
    if current_public_index.case_ids != scorer_index.case_ids:
        raise ValueError("public and scorer case rosters differ")

    _validate_exact_tree(
        public_dataset.canonical_public_root,
        root_files={"public_index.json"},
        directory_files=_public_tree_files(current_public_index),
        label="TraceFix public tree",
    )
    _validate_exact_tree(
        canonical_scorer_root,
        root_files={"scorer_index.json", SCORER_REGISTRY_FILENAME},
        directory_files=_scorer_tree_files(scorer_index),
        label="TraceFix scorer tree",
    )
    scorer_registry = _load_scorer_registry(canonical_scorer_root, scorer_index)

    (
        source_closure_by_id,
        case_to_source_entry_id,
        denominator_snapshot,
    ) = _load_denominator_closure(
        scorer_root=canonical_scorer_root,
        scorer_index=scorer_index,
        expected_population_id=expected_population_id,
        public_case_ids=current_public_index.case_ids,
    )
    scorer_entry_by_id = {entry.case_id: entry for entry in scorer_index.cases}
    for public_entry in current_public_index.cases:
        public_case = _decode_public_case(public_dataset, public_entry)
        scorer_entry = scorer_entry_by_id[public_entry.case_id]
        if (
            public_case.track is TraceFixTrack.STATIC_ATTRIBUTION
            and scorer_entry.attribution_gold_path is None
        ):
            raise ValueError("static attribution case has no registered gold path")
        if (
            public_case.track is TraceFixTrack.HISTORICAL_CAUSAL_VERIFICATION
            and scorer_entry.attribution_gold_path is not None
        ):
            raise ValueError("historical causal case registers attribution gold")
        bundle = _decode_scorer_bundle(
            scorer_root=canonical_scorer_root,
            scorer_entry=scorer_entry,
        )
        _validate_scorer_bundle(
            bundle=bundle,
            scorer_entry=scorer_entry,
            public_case=public_case,
            public_entry=public_entry,
            source_closure_by_id=source_closure_by_id,
            case_to_source_entry_id=case_to_source_entry_id,
        )

    receipt = TraceFixPopulationPreflightReceiptV1(
        suite_id=current_public_index.suite_id,
        suite_version=current_public_index.suite_version,
        population_id=current_public_index.population_id,
        public_index_schema=current_public_index.schema,
        scorer_index_schema=scorer_index.schema,
        scorer_registry_id=scorer_registry.registry_id,
        case_count=current_public_index.case_count,
        case_ids=current_public_index.case_ids,
        source_denominator_count=scorer_index.source_denominator_count,
        source_denominator_entry_ids=scorer_index.source_denominator_entry_ids,
    )
    return _PreflightStateV1(
        receipt=receipt,
        scorer_index=scorer_index,
        scorer_registry=scorer_registry,
        case_source_bindings=tuple(
            (case_id, case_to_source_entry_id[case_id])
            for case_id in current_public_index.case_ids
        ),
        denominator_snapshot=denominator_snapshot,
    )


def preflight_tracefix_pair(
    *,
    public_dataset: PublicTraceFixDatasetV1,
    scorer_root: Path,
    expected_population_id: str,
    expected_suite_version: str,
) -> TraceFixPopulationPreflightReceiptV1:
    """Validate a public/scorer pair without creating a consumable dataset."""

    return _preflight_tracefix_pair(
        public_dataset=public_dataset,
        scorer_root=scorer_root,
        expected_population_id=expected_population_id,
        expected_suite_version=expected_suite_version,
    ).receipt


def derive_tracefix_release_paths(
    release_record_path: Path,
) -> tuple[Path, Path, Path, str]:
    """Derive the only public/scorer sibling pair accepted by a release record."""

    record_path = _require_canonical_absolute_path(
        release_record_path,
        label="TraceFix release record path",
    )
    if record_path.parent.name != _RELEASE_RECORD_DIRECTORY:
        raise ValueError("release record must be inside the release_records directory")
    if not record_path.name.endswith(_RELEASE_RECORD_SUFFIX):
        raise ValueError("release record filename must end with .release.json")
    release_name = record_path.name[: -len(_RELEASE_RECORD_SUFFIX)]
    if (
        _RELEASE_NAME_PATTERN.fullmatch(release_name) is None
        or release_name.casefold() == "latest"
    ):
        raise ValueError("release record uses an invalid immutable release name")
    release_parent = record_path.parent.parent
    public_root = release_parent / f"{release_name}.public"
    scorer_root = release_parent / f"{release_name}.scorer"
    return record_path, public_root, scorer_root, release_name


def _validate_record_binding(
    *,
    record: TraceFixReleaseRecordV1,
    release_name: str,
    public_root: Path,
    scorer_root: Path,
    public_dataset: PublicTraceFixDatasetV1,
    receipt: TraceFixPopulationPreflightReceiptV1,
) -> None:
    """Close a bound release record against derived roots and preflight rosters."""

    if public_root != public_dataset.canonical_public_root:
        raise ValueError("release record derives a different public root")
    if record.public_directory_name != f"{release_name}.public":
        raise ValueError("release record public directory name changed")
    if record.scorer_directory_name != f"{release_name}.scorer":
        raise ValueError("release record scorer directory name changed")
    if scorer_root.name != record.scorer_directory_name:
        raise ValueError("release record derives a different scorer root")
    if (
        record.suite_id != receipt.suite_id
        or record.suite_version != receipt.suite_version
        or record.population_id != receipt.population_id
        or record.public_index_schema != receipt.public_index_schema
        or record.scorer_index_schema != receipt.scorer_index_schema
        or record.scorer_registry_id != receipt.scorer_registry_id
        or record.case_count != receipt.case_count
        or record.case_ids != receipt.case_ids
        or record.source_denominator_entry_ids
        != receipt.source_denominator_entry_ids
    ):
        raise ValueError("release record does not match population preflight")
    if receipt.case_count == 0 or receipt.source_denominator_count == 0:
        raise ValueError("a bound scorer dataset must be non-empty")


def load_scorer_dataset(
    *,
    public_dataset: PublicTraceFixDatasetV1,
    release_record_path: Path,
) -> ScorerTraceFixDatasetV1:
    """Load one bound scorer release after mandatory full-population preflight."""

    public_dataset = _require_loaded_public_dataset(public_dataset)
    record_path, public_root, scorer_root, release_name = (
        derive_tracefix_release_paths(release_record_path)
    )
    release_parent_fd = _open_absolute_directory(
        record_path.parent.parent,
        label="TraceFix release parent",
    )
    os.close(release_parent_fd)
    release_records_fd = _open_absolute_directory(
        record_path.parent,
        label="TraceFix release record directory",
    )
    os.close(release_records_fd)
    if public_root != public_dataset.canonical_public_root:
        raise ValueError("release record path does not bind the loaded public root")
    record = _serialization.decode_release_record_v1(
        _serialization.load_json_object_strict(
            record_path,
            label="TraceFix release record",
        )
    )
    state = _preflight_tracefix_pair(
        public_dataset=public_dataset,
        scorer_root=scorer_root,
        expected_population_id=record.population_id,
        expected_suite_version=record.suite_version,
    )
    _validate_record_binding(
        record=record,
        release_name=release_name,
        public_root=public_root,
        scorer_root=scorer_root,
        public_dataset=public_dataset,
        receipt=state.receipt,
    )
    return ScorerTraceFixDatasetV1(
        canonical_scorer_root=scorer_root,
        canonical_release_record_path=record_path,
        public_dataset=public_dataset,
        index=state.scorer_index,
        registry=state.scorer_registry,
        release_record=record,
        preflight_receipt=state.receipt,
        _case_source_bindings=state.case_source_bindings,
        _denominator_snapshot=state.denominator_snapshot,
        _validation_token=_VALIDATED_DATASET_TOKEN,
    )


def load_scorer_case(
    dataset: ScorerTraceFixDatasetV1,
    *,
    public_case: PublicTracePacketV1,
) -> ScorerCaseBundleV1:
    """Reread and revalidate one private bundle against the supplied public case."""

    if (
        not isinstance(dataset, ScorerTraceFixDatasetV1)
        or dataset._validation_token is not _VALIDATED_DATASET_TOKEN
    ):
        raise ValueError("scorer case loading requires a loader-validated dataset")
    if not isinstance(public_case, PublicTracePacketV1):
        raise ValueError("scorer case loading requires a public trace packet")
    scorer_entry = dataset.case_entry(public_case.case_id)
    public_entry = dataset.public_dataset.case_entry(public_case.case_id)

    current_record = _serialization.decode_release_record_v1(
        _serialization.load_json_object_strict(
            dataset.canonical_release_record_path,
            label="TraceFix release record",
        )
    )
    if current_record != dataset.release_record:
        raise ValueError("release record changed after scorer dataset preflight")
    current_public_index = _load_public_index(
        dataset.public_dataset.canonical_public_root
    )
    current_scorer_index = _load_scorer_index(dataset.canonical_scorer_root)
    if current_public_index != dataset.public_dataset.index:
        raise ValueError("public index changed after scorer dataset preflight")
    if current_scorer_index != dataset.index:
        raise ValueError("scorer index changed after scorer dataset preflight")
    _validate_exact_tree(
        dataset.public_dataset.canonical_public_root,
        root_files={"public_index.json"},
        directory_files=_public_tree_files(current_public_index),
        label="TraceFix public tree",
    )
    _validate_exact_tree(
        dataset.canonical_scorer_root,
        root_files={"scorer_index.json", SCORER_REGISTRY_FILENAME},
        directory_files=_scorer_tree_files(current_scorer_index),
        label="TraceFix scorer tree",
    )
    current_registry = _load_scorer_registry(
        dataset.canonical_scorer_root,
        current_scorer_index,
    )
    if current_registry != dataset.registry:
        raise ValueError("scorer registry changed after dataset preflight")

    (
        source_closure_by_id,
        case_to_source_entry_id,
        current_denominator_snapshot,
    ) = _load_denominator_closure(
        scorer_root=dataset.canonical_scorer_root,
        scorer_index=current_scorer_index,
        expected_population_id=current_scorer_index.population_id,
        public_case_ids=current_public_index.case_ids,
    )
    current_bindings = tuple(
        (case_id, case_to_source_entry_id[case_id])
        for case_id in current_public_index.case_ids
    )
    if current_bindings != dataset._case_source_bindings:
        raise ValueError("source denominator bindings changed after preflight")
    if current_denominator_snapshot != dataset._denominator_snapshot:
        raise ValueError("source denominator rows changed after preflight")

    reread_public_case = _decode_public_case(dataset.public_dataset, public_entry)
    if reread_public_case != public_case:
        raise ValueError("supplied public case differs from the bound on-disk packet")
    bundle = _decode_scorer_bundle(
        scorer_root=dataset.canonical_scorer_root,
        scorer_entry=scorer_entry,
    )
    _validate_scorer_bundle(
        bundle=bundle,
        scorer_entry=scorer_entry,
        public_case=reread_public_case,
        public_entry=public_entry,
        source_closure_by_id=source_closure_by_id,
        case_to_source_entry_id=case_to_source_entry_id,
    )
    return bundle


__all__ = [
    "PublicTraceFixDatasetV1",
    "ScorerCaseBundleV1",
    "ScorerTraceFixDatasetV1",
    "TraceFixDatasetValidationCode",
    "TraceFixDatasetValidationError",
    "TraceFixPopulationPreflightReceiptV1",
    "derive_tracefix_release_paths",
    "load_public_case",
    "load_public_dataset",
    "load_scorer_case",
    "load_scorer_dataset",
    "preflight_public_dataset",
    "preflight_tracefix_pair",
]
