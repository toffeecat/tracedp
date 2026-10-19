"""Closed dataset and release contracts for TraceFix-Bench v0.1.

The types in this module describe only index and release-record structure,
including portable case-insensitive file identities, canonical denominator
ordering, and canonical UTC binding evidence. Strict JSON decoding,
filesystem safety, cross-file closure, and publication remain separate
responsibilities of the loader and publication modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath

from lh_platform.tracefix_bench.contracts import ExecutionSemantics, TraceFixTrack

TRACEFIX_BENCH_SUITE_ID = "longhorizon_tracefix_bench"
PUBLIC_DATASET_INDEX_SCHEMA = "longhorizon_tracefix_bench/public_index/v1"
SCORER_DATASET_INDEX_SCHEMA = "longhorizon_tracefix_bench/scorer_index/v1"
TRACEFIX_RELEASE_RECORD_SCHEMA = "longhorizon_tracefix_bench/release_record/v1"
TRACEFIX_RECOVERY_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/recovery_receipt/v1"
)

_PORTABLE_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
)


class ReleasePublicationStatus(StrEnum):
    """Represent the only state in which a release record may exist."""

    BOUND = "bound"


class ReleaseBindingMode(StrEnum):
    """Distinguish normal publication from explicit recovery binding."""

    INITIAL_PUBLISH = "initial_publish"
    BIND_EXISTING_UNBOUND_PAIR = "bind_existing_unbound_pair"


class RecoveryPriorState(StrEnum):
    """Close recovery evidence to the only state recoverable in v0.1."""

    UNBOUND_DIRECTORY_PAIR = "unbound_directory_pair"


class ReleaseImplementationStatus(StrEnum):
    """Freeze implementation evidence state independently of mutable code constants."""

    IMPLEMENTATION_UNVERIFIED = "implementation_unverified"


def _require_text(value: str, *, label: str) -> None:
    """Require non-empty text without silently normalizing its identity."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _require_identifier(value: str, *, label: str) -> None:
    """Require a portable ASCII identifier that is safe as one path component."""

    _require_text(value, label=label)
    if _PORTABLE_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{label} must use portable ASCII letters, digits, '.', '_', or '-' "
            "and must start and end with a letter or digit"
        )


def _require_tuple(value: object, *, label: str) -> None:
    """Keep frozen collection fields immutable at the contract boundary."""

    if not isinstance(value, tuple):
        raise ValueError(f"{label} must be a tuple")


def _require_non_negative_integer(value: int, *, label: str) -> None:
    """Reject booleans and negative values where a count or ordinal is required."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _require_unique_texts(values: tuple[str, ...], *, label: str) -> None:
    """Require portable identities that remain unique after case folding."""

    _require_tuple(values, label=label)
    for value in values:
        _require_identifier(value, label=f"{label} item")
    folded_values = tuple(value.casefold() for value in values)
    if len(folded_values) != len(set(folded_values)):
        raise ValueError(f"{label} must be unique after case folding")


def _require_relative_file_path(
    value: str,
    *,
    label: str,
    expected_directory: str,
    expected_identifier: str,
) -> None:
    """Require the exact v0.1 POSIX path for one indexed JSON object."""

    _require_text(value, label=label)
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    expected_path = PurePosixPath(expected_directory) / f"{expected_identifier}.json"
    if path.is_absolute() or value != expected_path.as_posix():
        raise ValueError(f"{label} must equal {expected_path.as_posix()}")


def _require_directory_name(value: str, *, label: str) -> None:
    """Require an immutable release directory name rather than a path or alias."""

    _require_identifier(value, label=label)
    if value.casefold() == "latest":
        raise ValueError(f"{label} cannot use the mutable 'latest' alias")


def _require_utc_rfc3339_seconds(value: str, *, label: str) -> None:
    """Require canonical RFC 3339 UTC time with exactly second precision."""

    _require_text(value, label=label)
    expected_format = "YYYY-MM-DDTHH:MM:SSZ"
    timestamp_pattern = (
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
        r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
    )
    if re.fullmatch(timestamp_pattern, value) is None:
        raise ValueError(
            f"{label} must use UTC RFC3339 seconds format {expected_format}"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ValueError(
            f"{label} must be a valid UTC RFC3339 timestamp in {expected_format} format"
        ) from None


@dataclass(frozen=True)
class PublicCaseIndexEntryV1:
    """Index one public case using case_ordinal as its zero-based roster position."""

    case_id: str
    case_ordinal: int
    track: TraceFixTrack
    execution_semantics: ExecutionSemantics
    relative_path: str

    def __post_init__(self) -> None:
        _require_identifier(self.case_id, label="public case_id")
        _require_non_negative_integer(self.case_ordinal, label="case_ordinal")
        if not isinstance(self.track, TraceFixTrack):
            raise ValueError("public case track must be a TraceFixTrack")
        if not isinstance(self.execution_semantics, ExecutionSemantics):
            raise ValueError(
                "public case execution_semantics must be an ExecutionSemantics"
            )
        _require_relative_file_path(
            self.relative_path,
            label="public case relative_path",
            expected_directory="cases",
            expected_identifier=self.case_id,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one public index row with fixed field order."""

        return {
            "case_id": self.case_id,
            "case_ordinal": self.case_ordinal,
            "track": self.track.value,
            "execution_semantics": self.execution_semantics.value,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class PublicDatasetIndexV1:
    """Freeze the ordered public case roster for one population."""

    suite_id: str
    suite_version: str
    population_id: str
    case_count: int
    cases: tuple[PublicCaseIndexEntryV1, ...]
    schema: str = PUBLIC_DATASET_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_DATASET_INDEX_SCHEMA:
            raise ValueError(f"unsupported public dataset index schema: {self.schema}")
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID:
            raise ValueError(f"unsupported TraceFix-Bench suite_id: {self.suite_id}")
        _require_identifier(self.suite_version, label="suite_version")
        _require_identifier(self.population_id, label="population_id")
        _require_non_negative_integer(self.case_count, label="case_count")
        _require_tuple(self.cases, label="public cases")
        if any(not isinstance(item, PublicCaseIndexEntryV1) for item in self.cases):
            raise ValueError("public cases must contain PublicCaseIndexEntryV1 values")
        if self.case_count != len(self.cases):
            raise ValueError("public case_count must equal the number of case entries")
        case_ids = tuple(item.case_id for item in self.cases)
        if len(case_ids) != len(set(case_id.casefold() for case_id in case_ids)):
            raise ValueError("public case_id roster must be unique after case folding")
        ordinals = tuple(item.case_ordinal for item in self.cases)
        if ordinals != tuple(range(self.case_count)):
            raise ValueError(
                "public case_ordinal values must equal their zero-based roster positions"
            )
        paths = tuple(item.relative_path for item in self.cases)
        if len(paths) != len(set(path.casefold() for path in paths)):
            raise ValueError(
                "public case relative paths must be unique after case folding"
            )

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return opaque case IDs in the frozen ordinal order."""

        return tuple(item.case_id for item in self.cases)

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete public index with fixed field order."""

        return {
            "schema": self.schema,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "case_count": self.case_count,
            "cases": [item.to_dict() for item in self.cases],
        }


@dataclass(frozen=True)
class ScorerCaseIndexEntryV1:
    """Index scorer-only case records without embedding their contents."""

    case_id: str
    case_ref_path: str
    admission_path: str
    projection_receipt_id: str
    projection_receipt_path: str
    attribution_gold_path: str | None

    def __post_init__(self) -> None:
        _require_identifier(self.case_id, label="scorer case_id")
        _require_identifier(
            self.projection_receipt_id,
            label="projection_receipt_id",
        )
        for value, label, directory in (
            (self.case_ref_path, "case_ref_path", "case_refs"),
            (self.admission_path, "admission_path", "admissions"),
            (
                self.projection_receipt_path,
                "projection_receipt_path",
                "projection_receipts",
            ),
        ):
            _require_relative_file_path(
                value,
                label=label,
                expected_directory=directory,
                expected_identifier=self.case_id,
            )
        if self.attribution_gold_path is not None:
            _require_relative_file_path(
                self.attribution_gold_path,
                label="attribution_gold_path",
                expected_directory="attribution_gold",
                expected_identifier=self.case_id,
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize one scorer case index row with fixed field order."""

        return {
            "case_id": self.case_id,
            "case_ref_path": self.case_ref_path,
            "admission_path": self.admission_path,
            "projection_receipt_id": self.projection_receipt_id,
            "projection_receipt_path": self.projection_receipt_path,
            "attribution_gold_path": self.attribution_gold_path,
        }


@dataclass(frozen=True)
class SourceDenominatorIndexEntryV1:
    """Index one scorer-only source denominator row."""

    entry_id: str
    relative_path: str

    def __post_init__(self) -> None:
        _require_identifier(self.entry_id, label="source denominator entry_id")
        _require_relative_file_path(
            self.relative_path,
            label="source denominator relative_path",
            expected_directory="source_denominator",
            expected_identifier=self.entry_id,
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize one source denominator index row with fixed field order."""

        return {
            "entry_id": self.entry_id,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class ScorerDatasetIndexV1:
    """Freeze scorer paths and a canonically ordered source denominator roster."""

    suite_id: str
    suite_version: str
    population_id: str
    case_count: int
    source_denominator_count: int
    cases: tuple[ScorerCaseIndexEntryV1, ...]
    source_denominator: tuple[SourceDenominatorIndexEntryV1, ...]
    scorer_registry_id: str
    scorer_registry_path: str = "scorer_registry.json"
    schema: str = SCORER_DATASET_INDEX_SCHEMA

    def __post_init__(self) -> None:
        _require_identifier(self.scorer_registry_id, label="scorer_registry_id")
        if self.scorer_registry_path != "scorer_registry.json":
            raise ValueError("scorer registry path must equal the fixed root filename")
        if self.schema != SCORER_DATASET_INDEX_SCHEMA:
            raise ValueError(f"unsupported scorer dataset index schema: {self.schema}")
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID:
            raise ValueError(f"unsupported TraceFix-Bench suite_id: {self.suite_id}")
        _require_identifier(self.suite_version, label="suite_version")
        _require_identifier(self.population_id, label="population_id")
        _require_non_negative_integer(self.case_count, label="case_count")
        _require_non_negative_integer(
            self.source_denominator_count,
            label="source_denominator_count",
        )
        _require_tuple(self.cases, label="scorer cases")
        _require_tuple(self.source_denominator, label="source_denominator")
        if any(not isinstance(item, ScorerCaseIndexEntryV1) for item in self.cases):
            raise ValueError("scorer cases must contain ScorerCaseIndexEntryV1 values")
        if any(
            not isinstance(item, SourceDenominatorIndexEntryV1)
            for item in self.source_denominator
        ):
            raise ValueError(
                "source_denominator must contain SourceDenominatorIndexEntryV1 values"
            )
        if self.case_count != len(self.cases):
            raise ValueError("scorer case_count must equal the number of case entries")
        if self.source_denominator_count != len(self.source_denominator):
            raise ValueError(
                "source_denominator_count must equal the number of source entries"
            )
        if self.case_count > 0 and self.source_denominator_count == 0:
            raise ValueError("a non-empty case roster requires a source denominator")
        case_ids = tuple(item.case_id for item in self.cases)
        if len(case_ids) != len(set(case_id.casefold() for case_id in case_ids)):
            raise ValueError("scorer case_id roster must be unique after case folding")
        projection_receipt_ids = tuple(
            item.projection_receipt_id for item in self.cases
        )
        if len(projection_receipt_ids) != len(
            set(receipt_id.casefold() for receipt_id in projection_receipt_ids)
        ):
            raise ValueError(
                "projection receipt IDs must be unique after case folding"
            )
        case_paths = tuple(
            path
            for item in self.cases
            for path in (
                item.case_ref_path,
                item.admission_path,
                item.projection_receipt_path,
                item.attribution_gold_path,
            )
            if path is not None
        )
        if len(case_paths) != len(set(path.casefold() for path in case_paths)):
            raise ValueError(
                "scorer case record paths must be unique after case folding"
            )
        source_ids = tuple(item.entry_id for item in self.source_denominator)
        if len(source_ids) != len(set(entry_id.casefold() for entry_id in source_ids)):
            raise ValueError(
                "source denominator entry_id roster must be unique after case folding"
            )
        if source_ids != tuple(sorted(source_ids, key=str.casefold)):
            raise ValueError(
                "source denominator entries must be in canonical entry_id order"
            )
        source_paths = tuple(item.relative_path for item in self.source_denominator)
        if len(source_paths) != len(set(path.casefold() for path in source_paths)):
            raise ValueError(
                "source denominator relative paths must be unique after case folding"
            )

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return scorer case IDs in their indexed order."""

        return tuple(item.case_id for item in self.cases)

    @property
    def source_denominator_entry_ids(self) -> tuple[str, ...]:
        """Return source denominator IDs in their indexed order."""

        return tuple(item.entry_id for item in self.source_denominator)

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete scorer index with fixed field order."""

        return {
            "schema": self.schema,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "case_count": self.case_count,
            "source_denominator_count": self.source_denominator_count,
            "cases": [item.to_dict() for item in self.cases],
            "source_denominator": [item.to_dict() for item in self.source_denominator],
            "scorer_registry_id": self.scorer_registry_id,
            "scorer_registry_path": self.scorer_registry_path,
        }


@dataclass(frozen=True)
class TraceFixRecoveryReceiptV1:
    """Record operator-confirmed evidence for binding an unbound directory pair."""

    receipt_id: str
    suite_id: str
    suite_version: str
    population_id: str
    public_directory_name: str
    scorer_directory_name: str
    confirmed_at: str
    prior_state: RecoveryPriorState
    previous_publisher_terminated: bool
    schema: str = TRACEFIX_RECOVERY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACEFIX_RECOVERY_RECEIPT_SCHEMA:
            raise ValueError(
                f"unsupported TraceFix recovery receipt schema: {self.schema}"
            )
        _require_identifier(self.receipt_id, label="recovery receipt_id")
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID:
            raise ValueError("recovery receipt suite_id is not TraceFix-Bench")
        _require_identifier(self.suite_version, label="recovery suite_version")
        _require_identifier(self.population_id, label="recovery population_id")
        _require_directory_name(
            self.public_directory_name,
            label="recovery public_directory_name",
        )
        _require_directory_name(
            self.scorer_directory_name,
            label="recovery scorer_directory_name",
        )
        if (
            self.public_directory_name.casefold()
            == self.scorer_directory_name.casefold()
        ):
            raise ValueError("recovery directory names must differ")
        _require_utc_rfc3339_seconds(self.confirmed_at, label="confirmed_at")
        if self.prior_state is not RecoveryPriorState.UNBOUND_DIRECTORY_PAIR:
            raise ValueError(
                "recovery prior_state must be unbound_directory_pair"
            )
        if self.previous_publisher_terminated is not True:
            raise ValueError(
                "recovery requires confirmation that the previous publisher terminated"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize recovery evidence with a closed field set and fixed order."""

        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "public_directory_name": self.public_directory_name,
            "scorer_directory_name": self.scorer_directory_name,
            "confirmed_at": self.confirmed_at,
            "prior_state": self.prior_state.value,
            "previous_publisher_terminated": self.previous_publisher_terminated,
        }


@dataclass(frozen=True)
class TraceFixReleaseRecordV1:
    """Bind one non-empty directory pair and preserve how that binding occurred."""

    suite_id: str
    suite_version: str
    population_id: str
    public_directory_name: str
    scorer_directory_name: str
    public_index_schema: str
    scorer_index_schema: str
    scorer_registry_id: str
    case_count: int
    case_ids: tuple[str, ...]
    source_denominator_entry_ids: tuple[str, ...]
    publication_status: ReleasePublicationStatus
    bound_at: str
    binding_mode: ReleaseBindingMode
    recovery_receipt: TraceFixRecoveryReceiptV1 | None
    implementation_status: ReleaseImplementationStatus
    schema: str = TRACEFIX_RELEASE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        _require_identifier(self.scorer_registry_id, label="scorer_registry_id")
        if self.schema != TRACEFIX_RELEASE_RECORD_SCHEMA:
            raise ValueError(f"unsupported TraceFix release record schema: {self.schema}")
        if self.suite_id != TRACEFIX_BENCH_SUITE_ID:
            raise ValueError(f"unsupported TraceFix-Bench suite_id: {self.suite_id}")
        _require_identifier(self.suite_version, label="suite_version")
        _require_identifier(self.population_id, label="population_id")
        _require_directory_name(
            self.public_directory_name,
            label="public_directory_name",
        )
        _require_directory_name(
            self.scorer_directory_name,
            label="scorer_directory_name",
        )
        if (
            self.public_directory_name.casefold()
            == self.scorer_directory_name.casefold()
        ):
            raise ValueError(
                "public and scorer directory names must differ case-insensitively"
            )
        if self.public_index_schema != PUBLIC_DATASET_INDEX_SCHEMA:
            raise ValueError(
                f"unsupported public index schema: {self.public_index_schema}"
            )
        if self.scorer_index_schema != SCORER_DATASET_INDEX_SCHEMA:
            raise ValueError(
                f"unsupported scorer index schema: {self.scorer_index_schema}"
            )
        _require_non_negative_integer(self.case_count, label="case_count")
        _require_unique_texts(self.case_ids, label="release case_ids")
        _require_unique_texts(
            self.source_denominator_entry_ids,
            label="release source_denominator_entry_ids",
        )
        if self.source_denominator_entry_ids != tuple(
            sorted(self.source_denominator_entry_ids, key=str.casefold)
        ):
            raise ValueError(
                "release source_denominator_entry_ids must be in canonical order"
            )
        if self.case_count != len(self.case_ids):
            raise ValueError("release case_count must equal the case_id roster length")
        if self.publication_status is not ReleasePublicationStatus.BOUND:
            raise ValueError("a release record may only describe a bound directory pair")
        if self.case_count == 0 or not self.case_ids:
            raise ValueError("a bound release must contain at least one case")
        if not self.source_denominator_entry_ids:
            raise ValueError(
                "a bound release must contain at least one source denominator entry"
            )
        _require_utc_rfc3339_seconds(self.bound_at, label="bound_at")
        if not isinstance(self.binding_mode, ReleaseBindingMode):
            raise ValueError("binding_mode must be a ReleaseBindingMode")
        if self.binding_mode is ReleaseBindingMode.INITIAL_PUBLISH:
            if self.recovery_receipt is not None:
                raise ValueError(
                    "initial publication cannot include a recovery receipt"
                )
        elif self.binding_mode is ReleaseBindingMode.BIND_EXISTING_UNBOUND_PAIR:
            if not isinstance(self.recovery_receipt, TraceFixRecoveryReceiptV1):
                raise ValueError(
                    "binding an existing unbound pair requires a recovery receipt"
                )
            recovery = self.recovery_receipt
            if (
                recovery.suite_id != self.suite_id
                or recovery.suite_version != self.suite_version
                or recovery.population_id != self.population_id
                or recovery.public_directory_name != self.public_directory_name
                or recovery.scorer_directory_name != self.scorer_directory_name
            ):
                raise ValueError(
                    "recovery receipt does not bind this release identity"
                )
            confirmed_at = datetime.strptime(
                recovery.confirmed_at,
                "%Y-%m-%dT%H:%M:%SZ",
            )
            bound_at = datetime.strptime(self.bound_at, "%Y-%m-%dT%H:%M:%SZ")
            if confirmed_at > bound_at:
                raise ValueError("recovery confirmation cannot follow binding time")
        if (
            self.implementation_status
            is not ReleaseImplementationStatus.IMPLEMENTATION_UNVERIFIED
        ):
            raise ValueError(
                "release record cannot promote the current implementation status"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize the scorer-only release binding with fixed field order."""

        return {
            "schema": self.schema,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "population_id": self.population_id,
            "public_directory_name": self.public_directory_name,
            "scorer_directory_name": self.scorer_directory_name,
            "public_index_schema": self.public_index_schema,
            "scorer_index_schema": self.scorer_index_schema,
            "scorer_registry_id": self.scorer_registry_id,
            "case_count": self.case_count,
            "case_ids": list(self.case_ids),
            "source_denominator_entry_ids": list(
                self.source_denominator_entry_ids
            ),
            "publication_status": self.publication_status.value,
            "bound_at": self.bound_at,
            "binding_mode": self.binding_mode.value,
            "recovery_receipt": (
                None
                if self.recovery_receipt is None
                else self.recovery_receipt.to_dict()
            ),
            "implementation_status": self.implementation_status.value,
        }


__all__ = [
    "PUBLIC_DATASET_INDEX_SCHEMA",
    "SCORER_DATASET_INDEX_SCHEMA",
    "TRACEFIX_BENCH_SUITE_ID",
    "TRACEFIX_RECOVERY_RECEIPT_SCHEMA",
    "TRACEFIX_RELEASE_RECORD_SCHEMA",
    "PublicCaseIndexEntryV1",
    "PublicDatasetIndexV1",
    "RecoveryPriorState",
    "ReleaseBindingMode",
    "ReleaseImplementationStatus",
    "ReleasePublicationStatus",
    "ScorerCaseIndexEntryV1",
    "ScorerDatasetIndexV1",
    "SourceDenominatorIndexEntryV1",
    "TraceFixRecoveryReceiptV1",
    "TraceFixReleaseRecordV1",
]
