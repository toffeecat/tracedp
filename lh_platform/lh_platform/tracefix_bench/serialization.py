"""Strict JSON decoders and no-follow file access for TraceFix-Bench v0.1.

The decoders in this module preserve raw JSON types, reject unknown fields,
and construct only the closed contracts declared by ``contracts.py`` and
``dataset_contracts.py``.  Filesystem helpers keep registered reads beneath a
caller-selected root without following symbolic links.
"""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from lh_platform.tracefix_bench.contracts import (
    ATTRIBUTION_GOLD_SCHEMA,
    BENCHMARK_CASE_REF_SCHEMA,
    CASE_ADMISSION_SCHEMA,
    LOCATOR_MODEL_CALL_USAGE_RECEIPT_SCHEMA,
    LOCATOR_MODEL_CALL_USAGE_SCHEMA,
    PROVIDER_TOKEN_BUCKET_SCHEMA,
    PROVIDER_TOKEN_PARTITION_RECEIPT_SCHEMA,
    PUBLIC_TRACE_PACKET_SCHEMA,
    PUBLIC_TRACE_PROJECTION_RECEIPT_SCHEMA,
    SOURCE_DENOMINATOR_ENTRY_SCHEMA,
    AttributionGoldV1,
    BenchmarkCaseRefV1,
    BenchmarkSplit,
    CaseAdmissionV1,
    CaseOrigin,
    CanonicalInputTokenBucket,
    ExecutionSemantics,
    ExposureStatus,
    FieldCoverageStatus,
    FieldCoverageV1,
    GoldAuthority,
    GoldDisposition,
    GoldLabelKind,
    LocatorModelCallUsageReceiptV1,
    LocatorModelCallUsageV1,
    LocatorUsageV1,
    MaterializationStatus,
    ModelCallRosterStatus,
    PopulationRole,
    ProjectionFieldDecisionV1,
    ProviderTokenBucketV1,
    ProviderTokenPartitionReceiptV1,
    PublicEventKind,
    PublicFailureSignalV1,
    PublicFieldValueKind,
    PublicTraceEventV1,
    PublicTraceFieldV1,
    PublicTracePacketV1,
    PublicTraceProjectionReceiptV1,
    SourceDenominatorEntryV1,
    SourceOutcomeStatus,
    TraceFixTrack,
    TokenPartitionStatus,
    UsageValueStatus,
)
from lh_platform.tracefix_bench.dataset_contracts import (
    PUBLIC_DATASET_INDEX_SCHEMA,
    SCORER_DATASET_INDEX_SCHEMA,
    TRACEFIX_RECOVERY_RECEIPT_SCHEMA,
    TRACEFIX_RELEASE_RECORD_SCHEMA,
    PublicCaseIndexEntryV1,
    PublicDatasetIndexV1,
    RecoveryPriorState,
    ReleaseBindingMode,
    ReleaseImplementationStatus,
    ReleasePublicationStatus,
    ScorerCaseIndexEntryV1,
    ScorerDatasetIndexV1,
    SourceDenominatorIndexEntryV1,
    TraceFixRecoveryReceiptV1,
    TraceFixReleaseRecordV1,
)
from lh_platform.tracefix_bench import scorer_contracts as _sc
from lh_platform.tracing.unified import TraceFindingKind

_EnumT = TypeVar("_EnumT", bound=StrEnum)


class _StrictJsonError(ValueError):
    """Mark parser failures that must be reported without reproducing content."""


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate keys at every depth."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError("duplicate object key")
        result[key] = value
    return result


def _reject_non_finite_constant(_: str) -> object:
    """Reject the non-standard NaN and infinity constants accepted by json.loads."""

    raise _StrictJsonError("non-finite JSON constant")


def _parse_finite_float(token: str) -> float:
    """Reject finite-looking JSON numbers that overflow Python float."""

    value = float(token)
    if not math.isfinite(value):
        raise _StrictJsonError("non-finite JSON number")
    return value


def _decode_json_object(data: bytes, *, label: str, location: str) -> dict[str, object]:
    """Decode one UTF-8 JSON object without exposing its contents in errors."""

    try:
        text = data.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_non_finite_constant,
            parse_float=_parse_finite_float,
        )
    except (UnicodeDecodeError, ValueError, OverflowError):
        raise ValueError(f"{label} is not a strict UTF-8 JSON object: {location}") from None
    if type(payload) is not dict:
        raise ValueError(f"{label} must contain one top-level JSON object: {location}")
    return payload


def _read_fd(fd: int) -> bytes:
    """Read the opened inode fully; v0.1 has not frozen a byte-size limit."""

    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _open_flags(*, directory: bool) -> int:
    """Return fail-closed flags for a directory or regular-file open."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("this platform does not provide O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    if directory:
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        if not directory_flag:
            raise RuntimeError("this platform does not provide O_DIRECTORY")
        flags |= directory_flag
    else:
        # A nonblocking open prevents an attacker-controlled FIFO from hanging
        # before fstat can reject it as a non-regular file.
        flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _absolute_path_components(path: Path, *, label: str) -> tuple[str, ...]:
    """Return literal child names for an absolute path rooted at ``/``."""

    if not isinstance(path, Path):
        raise ValueError(f"{label} must be an absolute Path")
    raw_path = os.fspath(path)
    if not path.is_absolute() or not raw_path.startswith("/"):
        raise ValueError(f"{label} must be an absolute Path")
    if raw_path == "/":
        return ()
    if raw_path.startswith("//"):
        raise ValueError(f"{label} must start at the single POSIX root")
    components = tuple(raw_path[1:].split("/"))
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"{label} contains an invalid absolute path component")
    return components


def _open_absolute_path_fd(path: Path, *, label: str, directory: bool) -> int:
    """Open an absolute path from ``/`` while refusing symlinks at every step."""

    components = _absolute_path_components(path, label=label)
    if not components and not directory:
        raise ValueError(f"{label} must name a regular file below the POSIX root")

    opened_fds: list[int] = []
    try:
        try:
            opened_fds.append(os.open("/", _open_flags(directory=True)))
        except OSError:
            raise ValueError(f"{label} POSIX root is not accessible") from None

        for index, component in enumerate(components):
            is_final = index == len(components) - 1
            expect_directory = directory or not is_final
            try:
                child_fd = os.open(
                    component,
                    _open_flags(directory=expect_directory),
                    dir_fd=opened_fds[-1],
                )
            except OSError:
                raise ValueError(f"{label} is not accessible without symlinks") from None
            try:
                child_mode = os.fstat(child_fd).st_mode
                if expect_directory and not stat.S_ISDIR(child_mode):
                    raise ValueError(f"{label} path component is not a real directory")
                if not expect_directory and not stat.S_ISREG(child_mode):
                    raise ValueError(f"{label} target is not a regular file")
            except OSError:
                os.close(child_fd)
                raise ValueError(f"{label} is not accessible without symlinks") from None
            except BaseException:
                os.close(child_fd)
                raise
            opened_fds.append(child_fd)

        target_fd = opened_fds.pop()
        return target_fd
    finally:
        for opened_fd in reversed(opened_fds):
            os.close(opened_fd)


def _open_real_directory_fd(path: Path, *, label: str) -> int:
    """Open an absolute directory through a no-follow component walk."""

    return _open_absolute_path_fd(path, label=label, directory=True)


def require_real_directory(path: Path, *, label: str) -> Path:
    """Validate an absolute, non-symlink directory and preserve its path identity."""

    fd = _open_real_directory_fd(path, label=label)
    os.close(fd)
    return path


def _validate_registered_path(
    relative_path: str,
    *,
    expected_directory: str,
    expected_identifier: str,
    label: str,
) -> tuple[str, ...]:
    """Require ``<directory>/<identifier>.json`` without path normalization."""

    if type(relative_path) is not str or not relative_path:
        raise ValueError(f"{label} relative path must be non-empty text")
    for value, role in (
        (expected_directory, "expected directory"),
        (expected_identifier, "expected identifier"),
    ):
        if type(value) is not str or not value:
            raise ValueError(f"{label} {role} must be non-empty text")
        if any(separator in value for separator in ("/", "\\", "\x00")):
            raise ValueError(f"{label} {role} must be one path component")
        if value in {".", ".."}:
            raise ValueError(f"{label} {role} must be a real child name")
    if "\\" in relative_path or relative_path.startswith("/"):
        raise ValueError(f"{label} must use a relative POSIX path")
    parts = tuple(relative_path.split("/"))
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} contains an invalid POSIX path component")
    if parts != (expected_directory, f"{expected_identifier}.json"):
        raise ValueError(f"{label} does not match its registered identity")
    return parts


def load_registered_json_object_strict(
    root: Path,
    relative_path: str,
    *,
    expected_directory: str,
    expected_identifier: str,
    label: str,
) -> dict[str, object]:
    """Open one identity-bound registered child and decode its JSON object."""

    parts = _validate_registered_path(
        relative_path,
        expected_directory=expected_directory,
        expected_identifier=expected_identifier,
        label=label,
    )
    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        directory_fds.append(_open_real_directory_fd(root, label=f"{label} root"))
        for component in parts[:-1]:
            try:
                child_fd = os.open(
                    component,
                    _open_flags(directory=True),
                    dir_fd=directory_fds[-1],
                )
            except OSError:
                raise ValueError(
                    f"{label} has an inaccessible registered path: {relative_path}"
                ) from None
            try:
                if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                    raise ValueError(
                        f"{label} registered parent is not a directory: {relative_path}"
                    )
            except BaseException:
                os.close(child_fd)
                raise
            directory_fds.append(child_fd)
        try:
            file_fd = os.open(
                parts[-1],
                _open_flags(directory=False),
                dir_fd=directory_fds[-1],
            )
        except OSError:
            raise ValueError(
                f"{label} is not an accessible registered file: {relative_path}"
            ) from None
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ValueError(
                f"{label} registered target is not a regular file: {relative_path}"
            )
        return _decode_json_object(
            _read_fd(file_fd),
            label=label,
            location=relative_path,
        )
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def load_json_object_strict(path: Path, *, label: str) -> dict[str, object]:
    """Read an absolute regular file through a no-follow component walk."""

    fd = _open_absolute_path_fd(path, label=f"{label} path", directory=False)
    try:
        return _decode_json_object(_read_fd(fd), label=label, location=os.fspath(path))
    finally:
        os.close(fd)


def _require_object_fields(
    payload: object,
    expected_fields: frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    """Require one object with exactly the schema-declared field set."""

    if type(payload) is not dict:
        raise ValueError(f"{label} must be an object")
    if frozenset(payload) != expected_fields:
        raise ValueError(f"{label} fields do not match the closed schema")
    return payload


def _require_text(value: object, *, label: str) -> str:
    """Require the original JSON value to be a string without coercion."""

    if type(value) is not str:
        raise ValueError(f"{label} must be text")
    return value


def _require_optional_text(value: object, *, label: str) -> str | None:
    """Require an original JSON string or null without coercion."""

    if value is None:
        return None
    return _require_text(value, label=label)


def _require_integer(value: object, *, label: str) -> int:
    """Require an original JSON integer and reject booleans."""

    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _require_optional_integer(value: object, *, label: str) -> int | None:
    """Require an original JSON integer or null and reject booleans."""

    if value is None:
        return None
    return _require_integer(value, label=label)


def _require_number(value: object, *, label: str) -> int | float:
    """Require one finite JSON number without converting its numeric type."""

    if type(value) not in {int, float}:
        raise ValueError(f"{label} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be a finite number")
    return value


def _require_optional_number(
    value: object,
    *,
    label: str,
) -> int | float | None:
    """Require an original finite JSON number or null without coercion."""

    if value is None:
        return None
    return _require_number(value, label=label)


def _require_boolean(value: object, *, label: str) -> bool:
    """Require the original JSON value to be boolean."""

    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _require_list(value: object, *, label: str) -> list[object]:
    """Require the original JSON value to be a list."""

    if type(value) is not list:
        raise ValueError(f"{label} must be a list")
    return value


def _decode_text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    """Decode a JSON list of strings without normalizing any item."""

    return tuple(
        _require_text(item, label=f"{label} item")
        for item in _require_list(value, label=label)
    )


def _decode_enum(value: object, enum_type: type[_EnumT], *, label: str) -> _EnumT:
    """Decode a closed string enum while hiding an invalid source value."""

    raw_value = _require_text(value, label=label)
    try:
        return enum_type(raw_value)
    except ValueError:
        raise ValueError(f"{label} is not in the closed vocabulary") from None


def _require_schema(
    value: object,
    expected_schema: str,
    *,
    label: str,
) -> str:
    """Require the exact schema identifier without echoing a supplied value."""

    schema = _require_text(value, label=label)
    if schema != expected_schema:
        raise ValueError(f"{label} is not the supported schema")
    return schema


def _decode_public_case_index_entry(payload: object) -> PublicCaseIndexEntryV1:
    """Decode one closed public index row."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "case_id",
                "case_ordinal",
                "track",
                "execution_semantics",
                "relative_path",
            }
        ),
        label="public case index entry",
    )
    return PublicCaseIndexEntryV1(
        case_id=_require_text(data["case_id"], label="public case_id"),
        case_ordinal=_require_integer(data["case_ordinal"], label="case_ordinal"),
        track=_decode_enum(data["track"], TraceFixTrack, label="public case track"),
        execution_semantics=_decode_enum(
            data["execution_semantics"],
            ExecutionSemantics,
            label="public case execution_semantics",
        ),
        relative_path=_require_text(
            data["relative_path"],
            label="public case relative_path",
        ),
    )


def decode_public_index_v1(payload: object) -> PublicDatasetIndexV1:
    """Decode the exact public population index schema."""

    data = _require_object_fields(
        payload,
        frozenset(
            {"schema", "suite_id", "suite_version", "population_id", "case_count", "cases"}
        ),
        label="public dataset index",
    )
    schema = _require_schema(
        data["schema"],
        PUBLIC_DATASET_INDEX_SCHEMA,
        label="public index schema",
    )
    cases = tuple(
        _decode_public_case_index_entry(item)
        for item in _require_list(data["cases"], label="public index cases")
    )
    return PublicDatasetIndexV1(
        schema=schema,
        suite_id=_require_text(data["suite_id"], label="public index suite_id"),
        suite_version=_require_text(
            data["suite_version"],
            label="public index suite_version",
        ),
        population_id=_require_text(
            data["population_id"],
            label="public index population_id",
        ),
        case_count=_require_integer(
            data["case_count"],
            label="public index case_count",
        ),
        cases=cases,
    )


def _decode_public_trace_field(payload: object) -> PublicTraceFieldV1:
    """Decode one allowlisted typed event field."""

    data = _require_object_fields(
        payload,
        frozenset({"field_name", "value_kind", "value"}),
        label="public trace field",
    )
    value_kind = _decode_enum(
        data["value_kind"],
        PublicFieldValueKind,
        label="public trace field value_kind",
    )
    raw_value = data["value"]
    if value_kind is PublicFieldValueKind.TEXT:
        value: str | int | float | bool | tuple[str, ...] = _require_text(
            raw_value,
            label="public text field value",
        )
    elif value_kind is PublicFieldValueKind.INTEGER:
        value = _require_integer(raw_value, label="public integer field value")
    elif value_kind is PublicFieldValueKind.NUMBER:
        value = _require_number(raw_value, label="public number field value")
    elif value_kind is PublicFieldValueKind.BOOLEAN:
        value = _require_boolean(raw_value, label="public boolean field value")
    else:
        value = _decode_text_tuple(raw_value, label="public text-list field value")
    return PublicTraceFieldV1(
        field_name=_require_text(data["field_name"], label="public field_name"),
        value_kind=value_kind,
        value=value,
    )


def _decode_public_trace_event(payload: object) -> PublicTraceEventV1:
    """Decode one public event without accepting a metadata sidecar."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "opaque_event_id",
                "sequence",
                "kind",
                "opaque_parent_event_ids",
                "typed_public_fields",
            }
        ),
        label="public trace event",
    )
    return PublicTraceEventV1(
        opaque_event_id=_require_text(
            data["opaque_event_id"],
            label="public opaque_event_id",
        ),
        sequence=_require_integer(data["sequence"], label="public event sequence"),
        kind=_decode_enum(data["kind"], PublicEventKind, label="public event kind"),
        opaque_parent_event_ids=_decode_text_tuple(
            data["opaque_parent_event_ids"],
            label="public parent event ids",
        ),
        typed_public_fields=tuple(
            _decode_public_trace_field(item)
            for item in _require_list(
                data["typed_public_fields"],
                label="public typed fields",
            )
        ),
    )


def _decode_public_failure_signal(payload: object) -> PublicFailureSignalV1:
    """Decode the bounded public failure signal."""

    data = _require_object_fields(
        payload,
        frozenset({"observed_failure", "public_feedback"}),
        label="public failure signal",
    )
    return PublicFailureSignalV1(
        observed_failure=_require_boolean(
            data["observed_failure"],
            label="observed_failure",
        ),
        public_feedback=_require_optional_text(
            data["public_feedback"],
            label="public_feedback",
        ),
    )


def _decode_field_coverage(payload: object) -> FieldCoverageV1:
    """Decode method-visible coverage without scorer evidence or detail."""

    data = _require_object_fields(
        payload,
        frozenset({"field_name", "status"}),
        label="public field coverage",
    )
    return FieldCoverageV1(
        field_name=_require_text(data["field_name"], label="coverage field_name"),
        status=_decode_enum(
            data["status"],
            FieldCoverageStatus,
            label="coverage status",
        ),
    )


def decode_public_trace_packet_v1(payload: object) -> PublicTracePacketV1:
    """Decode exactly the v0.1 method-visible public allowlist."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema",
                "case_id",
                "track",
                "public_task",
                "public_initial_observation",
                "events",
                "public_failure_signal",
                "execution_semantics",
                "field_coverage",
                "public_capabilities",
            }
        ),
        label="public trace packet",
    )
    return PublicTracePacketV1(
        schema=_require_schema(
            data["schema"],
            PUBLIC_TRACE_PACKET_SCHEMA,
            label="public trace packet schema",
        ),
        case_id=_require_text(data["case_id"], label="public packet case_id"),
        track=_decode_enum(data["track"], TraceFixTrack, label="public packet track"),
        public_task=_require_text(data["public_task"], label="public_task"),
        public_initial_observation=_require_optional_text(
            data["public_initial_observation"],
            label="public_initial_observation",
        ),
        events=tuple(
            _decode_public_trace_event(item)
            for item in _require_list(data["events"], label="public events")
        ),
        public_failure_signal=_decode_public_failure_signal(
            data["public_failure_signal"]
        ),
        execution_semantics=_decode_enum(
            data["execution_semantics"],
            ExecutionSemantics,
            label="public packet execution_semantics",
        ),
        field_coverage=tuple(
            _decode_field_coverage(item)
            for item in _require_list(
                data["field_coverage"],
                label="public field_coverage",
            )
        ),
        public_capabilities=_decode_text_tuple(
            data["public_capabilities"],
            label="public_capabilities",
        ),
    )


def _decode_locator_model_call_usage_v1(
    payload: object,
) -> LocatorModelCallUsageV1:
    """Decode one call record without coercing missing output into zero."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema", "call_id", "output_tokens", "output_status",
                "usage_receipt_ref", "unavailable_reason", "unavailable_evidence_ref",
            }
        ),
        label="locator model-call usage",
    )
    return LocatorModelCallUsageV1(
        schema=_require_schema(
            data["schema"], LOCATOR_MODEL_CALL_USAGE_SCHEMA,
            label="locator model-call usage schema",
        ),
        call_id=_require_text(data["call_id"], label="locator model call_id"),
        output_tokens=_require_optional_integer(
            data["output_tokens"], label="locator model-call output_tokens"
        ),
        output_status=_decode_enum(
            data["output_status"], UsageValueStatus,
            label="locator model-call output_status",
        ),
        usage_receipt_ref=_require_optional_text(
            data["usage_receipt_ref"], label="locator model-call usage_receipt_ref"
        ),
        unavailable_reason=_require_optional_text(
            data["unavailable_reason"], label="locator model-call unavailable_reason"
        ),
        unavailable_evidence_ref=_require_optional_text(
            data["unavailable_evidence_ref"],
            label="locator model-call unavailable_evidence_ref",
        ),
    )


def _decode_locator_model_call_usage_receipt_v1(
    payload: object,
) -> LocatorModelCallUsageReceiptV1:
    """Decode the typed complete-or-incomplete per-call roster."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema", "aggregation_receipt_ref", "roster_status",
                "calls", "missing_call_evidence_refs",
            }
        ),
        label="locator model-call usage receipt",
    )
    return LocatorModelCallUsageReceiptV1(
        schema=_require_schema(
            data["schema"], LOCATOR_MODEL_CALL_USAGE_RECEIPT_SCHEMA,
            label="locator model-call usage receipt schema",
        ),
        aggregation_receipt_ref=_require_text(
            data["aggregation_receipt_ref"],
            label="locator model-call aggregation_receipt_ref",
        ),
        roster_status=_decode_enum(
            data["roster_status"], ModelCallRosterStatus,
            label="locator model-call roster_status",
        ),
        calls=tuple(
            _decode_locator_model_call_usage_v1(item)
            for item in _require_list(data["calls"], label="locator model-call roster")
        ),
        missing_call_evidence_refs=_decode_text_tuple(
            data["missing_call_evidence_refs"],
            label="missing locator model-call evidence refs",
        ),
    )


def _decode_provider_token_bucket_v1(payload: object) -> ProviderTokenBucketV1:
    """Decode one canonical input-token bucket and its provider field mapping."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema", "bucket", "value", "status", "provider_field_names",
                "usage_receipt_ref", "unavailable_reason", "unavailable_evidence_ref",
            }
        ),
        label="provider token bucket",
    )
    return ProviderTokenBucketV1(
        schema=_require_schema(
            data["schema"], PROVIDER_TOKEN_BUCKET_SCHEMA,
            label="provider token bucket schema",
        ),
        bucket=_decode_enum(
            data["bucket"], CanonicalInputTokenBucket,
            label="canonical input token bucket",
        ),
        value=_require_optional_integer(data["value"], label="provider token bucket value"),
        status=_decode_enum(
            data["status"], UsageValueStatus, label="provider token bucket status"
        ),
        provider_field_names=_decode_text_tuple(
            data["provider_field_names"], label="provider token field names"
        ),
        usage_receipt_ref=_require_optional_text(
            data["usage_receipt_ref"], label="provider token usage_receipt_ref"
        ),
        unavailable_reason=_require_optional_text(
            data["unavailable_reason"], label="provider token unavailable_reason"
        ),
        unavailable_evidence_ref=_require_optional_text(
            data["unavailable_evidence_ref"],
            label="provider token unavailable_evidence_ref",
        ),
    )


def _decode_provider_token_partition_receipt_v1(
    payload: object,
) -> ProviderTokenPartitionReceiptV1:
    """Decode typed provider-field partition evidence for all canonical buckets."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema", "partition_receipt_ref", "provider_name",
                "provider_usage_receipt_ref", "partition_status", "buckets",
                "unresolved_buckets",
            }
        ),
        label="provider token partition receipt",
    )
    return ProviderTokenPartitionReceiptV1(
        schema=_require_schema(
            data["schema"], PROVIDER_TOKEN_PARTITION_RECEIPT_SCHEMA,
            label="provider token partition receipt schema",
        ),
        partition_receipt_ref=_require_text(
            data["partition_receipt_ref"], label="provider token partition_receipt_ref"
        ),
        provider_name=_require_text(data["provider_name"], label="provider token provider_name"),
        provider_usage_receipt_ref=_require_text(
            data["provider_usage_receipt_ref"], label="provider usage receipt ref"
        ),
        partition_status=_decode_enum(
            data["partition_status"], TokenPartitionStatus,
            label="provider token partition_status",
        ),
        buckets=tuple(
            _decode_provider_token_bucket_v1(item)
            for item in _require_list(data["buckets"], label="provider token buckets")
        ),
        unresolved_buckets=tuple(
            _decode_enum(
                item, CanonicalInputTokenBucket,
                label="unresolved canonical token bucket",
            )
            for item in _require_list(
                data["unresolved_buckets"], label="unresolved canonical token buckets"
            )
        ),
    )


def decode_locator_usage_v1(payload: object) -> LocatorUsageV1:
    """Decode exact locator usage fields and reconstruct nested evidence records."""

    value_names = (
        "model_calls", "input_tokens", "output_tokens",
        "max_observed_output_tokens_per_model_call", "cache_read_tokens",
        "cache_write_tokens", "tool_calls", "environment_actions",
        "probe_actions", "check_actions", "repair_actions", "replay_attempts",
        "wall_time_seconds", "monetary_cost",
    )
    data = _require_object_fields(
        payload,
        frozenset(
            value_names
            + (
                "value_status", "model_call_usage_receipt",
                "provider_token_partition_receipt",
            )
        ),
        label="locator usage",
    )
    raw_statuses = _require_object_fields(
        data["value_status"], frozenset(value_names),
        label="locator usage value_status",
    )
    values: dict[str, int | float | None] = {
        name: _require_optional_integer(data[name], label=f"locator usage {name}")
        for name in value_names[:12]
    }
    values.update(
        {
            name: _require_optional_number(data[name], label=f"locator usage {name}")
            for name in value_names[12:]
        }
    )
    return LocatorUsageV1(
        **values,
        value_status=tuple(
            (
                name,
                _decode_enum(
                    raw_statuses[name], UsageValueStatus,
                    label=f"locator usage {name} status",
                ),
            )
            for name in value_names
        ),
        model_call_usage_receipt=_decode_locator_model_call_usage_receipt_v1(
            data["model_call_usage_receipt"]
        ),
        provider_token_partition_receipt=_decode_provider_token_partition_receipt_v1(
            data["provider_token_partition_receipt"]
        ),
    )


def _decode_scorer_case_index_entry(payload: object) -> ScorerCaseIndexEntryV1:
    """Decode one scorer-only case index row."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "case_id",
                "case_ref_path",
                "admission_path",
                "projection_receipt_id",
                "projection_receipt_path",
                "attribution_gold_path",
            }
        ),
        label="scorer case index entry",
    )
    return ScorerCaseIndexEntryV1(
        case_id=_require_text(data["case_id"], label="scorer case_id"),
        case_ref_path=_require_text(
            data["case_ref_path"],
            label="case_ref_path",
        ),
        admission_path=_require_text(
            data["admission_path"],
            label="admission_path",
        ),
        projection_receipt_id=_require_text(
            data["projection_receipt_id"],
            label="projection_receipt_id",
        ),
        projection_receipt_path=_require_text(
            data["projection_receipt_path"],
            label="projection_receipt_path",
        ),
        attribution_gold_path=_require_optional_text(
            data["attribution_gold_path"],
            label="attribution_gold_path",
        ),
    )


def _decode_source_denominator_index_entry(
    payload: object,
) -> SourceDenominatorIndexEntryV1:
    """Decode one scorer-only denominator path entry."""

    data = _require_object_fields(
        payload,
        frozenset({"entry_id", "relative_path"}),
        label="source denominator index entry",
    )
    return SourceDenominatorIndexEntryV1(
        entry_id=_require_text(data["entry_id"], label="denominator entry_id"),
        relative_path=_require_text(
            data["relative_path"],
            label="denominator relative_path",
        ),
    )


def decode_scorer_index_v1(payload: object) -> ScorerDatasetIndexV1:
    """Decode the exact scorer population index schema."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema",
                "suite_id",
                "suite_version",
                "population_id",
                "case_count",
                "source_denominator_count",
                "cases",
                "source_denominator",
                "scorer_registry_id",
                "scorer_registry_path",
            }
        ),
        label="scorer dataset index",
    )
    return ScorerDatasetIndexV1(
        scorer_registry_id=_require_text(data["scorer_registry_id"], label="scorer_registry_id"),
        scorer_registry_path=_require_text(data["scorer_registry_path"], label="scorer_registry_path"),
        schema=_require_schema(
            data["schema"],
            SCORER_DATASET_INDEX_SCHEMA,
            label="scorer index schema",
        ),
        suite_id=_require_text(data["suite_id"], label="scorer index suite_id"),
        suite_version=_require_text(
            data["suite_version"],
            label="scorer index suite_version",
        ),
        population_id=_require_text(
            data["population_id"],
            label="scorer index population_id",
        ),
        case_count=_require_integer(
            data["case_count"],
            label="scorer index case_count",
        ),
        source_denominator_count=_require_integer(
            data["source_denominator_count"],
            label="source_denominator_count",
        ),
        cases=tuple(
            _decode_scorer_case_index_entry(item)
            for item in _require_list(data["cases"], label="scorer index cases")
        ),
        source_denominator=tuple(
            _decode_source_denominator_index_entry(item)
            for item in _require_list(
                data["source_denominator"],
                label="scorer source_denominator",
            )
        ),
    )


def _decode_registry_fields(
    payload: object,
    fields: dict[str, Callable[[object], object]],
) -> dict[str, object]:
    """Apply a code-owned field allowlist without coercion or executable JSON."""

    data = _require_object_fields(
        payload, frozenset(fields), label="scorer registry contract"
    )
    return {name: decoder(data[name]) for name, decoder in fields.items()}


def _registry_text(value: object) -> str:
    return _require_text(value, label="scorer registry text")


def _registry_number(value: object) -> int | float:
    return _require_number(value, label="scorer registry number")


def _registry_integer(value: object) -> int:
    return _require_integer(value, label="scorer registry integer")


def _registry_boolean(value: object) -> bool:
    return _require_boolean(value, label="scorer registry boolean")


def _registry_texts(value: object) -> tuple[str, ...]:
    return _decode_text_tuple(value, label="scorer registry text list")


def _registry_enum(enum_type: type[_EnumT]) -> Callable[[object], _EnumT]:
    """Bind an enum chosen by code, never a class named in source JSON."""

    return lambda value: _decode_enum(value, enum_type, label="scorer registry enum")


def _registry_records(
    decoder: Callable[[object], object],
) -> Callable[[object], tuple[object, ...]]:
    """Keep array decoding explicit at every nested contract boundary."""

    return lambda value: tuple(
        decoder(item)
        for item in _require_list(value, label="scorer registry records")
    )


def _decode_progress_metric_contract_v1(payload: object) -> _sc.ProgressMetricContractV1:
    """Decode a complete versioned progress definition, without measurements."""

    return _sc.ProgressMetricContractV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "progress_contract_id": _registry_text,
        "metric_id": _registry_text,
        "metric_version": _registry_text,
        "benchmark_id": _registry_text,
        "benchmark_version": _registry_text,
        "adapter_version": _registry_text,
        "evaluator_contract_id": _registry_text,
        "value_field": _registry_text,
        "unit": _registry_text,
        "direction": _registry_enum(_sc.ProgressDirection),
        "domain_min": _registry_number,
        "domain_max": _registry_number,
        "timepoints": _registry_texts,
        "tolerance": _registry_number,
        "minimum_meaningful_difference": _registry_number,
        "aggregation_rule_id": _registry_text,
        "aggregation_rule_source_ref": _registry_text,
        "unavailable_value_policy": _registry_enum(_sc.ProgressUnavailablePolicy),
        "evidence_source": _registry_enum(_sc.EvidenceSource),
        "source_ref": _registry_text,
    }))


def _decode_verifier_contract_v1(payload: object) -> _sc.VerifierContractV1:
    """Decode a fixed official verifier identity without invoking it."""

    return _sc.VerifierContractV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "verifier_contract_id": _registry_text,
        "benchmark_id": _registry_text,
        "benchmark_version": _registry_text,
        "adapter_version": _registry_text,
        "official_verifier_id": _registry_text,
        "official_verifier_version": _registry_text,
        "final_success_field": _registry_text,
        "source_ref": _registry_text,
    }))


def _decode_oracle_contract_v1(payload: object) -> _sc.OracleContractV1:
    """Decode independent optional-outcome prerequisites as data only."""

    return _sc.OracleContractV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "oracle_contract_id": _registry_text,
        "kind": _registry_enum(_sc.OracleKind),
        "benchmark_id": _registry_text,
        "benchmark_version": _registry_text,
        "adapter_version": _registry_text,
        "oracle_version": _registry_text,
        "outcome_field": _registry_text,
        "source_ref": _registry_text,
        "independent_of_method": _registry_boolean,
    }))


def _decode_resource_limit_v1(payload: object) -> _sc.ResourceLimitV1:
    """Decode a raw numeric limit; the contract enforces integer dimensions."""

    return _sc.ResourceLimitV1(**_decode_registry_fields(payload, {
        "role": _registry_enum(_sc.ResourceRole),
        "scope": _registry_enum(_sc.BudgetScope),
        "dimension": _registry_enum(_sc.ResourceDimension),
        "max_value": _registry_number,
    }))


def _decode_resource_budget_contract_v1(payload: object) -> _sc.ResourceBudgetContractV1:
    """Decode every resource limit without supplying omitted defaults."""

    return _sc.ResourceBudgetContractV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "budget_id": _registry_text,
        "limits": _registry_records(_decode_resource_limit_v1),
    }))


def _decode_scorer_registration_v1(payload: object) -> _sc.ScorerRegistrationV1:
    """Decode one track selection through the built-in implementation enum."""

    return _sc.ScorerRegistrationV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "scorer_id": _registry_text,
        "scorer_version": _registry_text,
        "track": _registry_enum(TraceFixTrack),
        "implementation_key": _registry_enum(_sc.BuiltinScorerKey),
        "progress_contract_ids": _registry_texts,
        "verifier_contract_ids": _registry_texts,
        "oracle_contract_ids": _registry_texts,
        "budget_ids": _registry_texts,
        "treatment_ids": _registry_texts,
        "system_cell_ids": _registry_texts,
    }))


def _decode_treatment_arm_registration_v1(
    payload: object,
) -> _sc.TreatmentArmRegistrationV1:
    """Decode one registry-owned treatment definition without plan overrides."""

    return _sc.TreatmentArmRegistrationV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "treatment_id": _registry_text,
        "applicable_tracks": _registry_records(_registry_enum(TraceFixTrack)),
        "locator_treatment_id": _registry_text,
        "fixer_treatment_id": _registry_text,
        "locator_config_refs": _registry_texts,
        "fixer_config_refs": _registry_texts,
        "method_config_refs": _registry_texts,
    }))


def _decode_system_cell_registration_v1(
    payload: object,
) -> _sc.SystemCellRegistrationV1:
    """Decode one registry-owned system cell and its frozen config references."""

    return _sc.SystemCellRegistrationV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "system_cell_id": _registry_text,
        "population_id": _registry_text,
        "benchmark_id": _registry_text,
        "benchmark_version": _registry_text,
        "adapter_version": _registry_text,
        "applicable_tracks": _registry_records(_registry_enum(TraceFixTrack)),
        "execution_semantics": _registry_enum(ExecutionSemantics),
        "fixer_treatment_id": _registry_text,
        "budget_id": _registry_text,
        "actor_identity_contract_ref": _registry_text,
        "source_actor_config_refs": _registry_texts,
        "recovery_actor_config_refs": _registry_texts,
        "system_config_refs": _registry_texts,
    }))


def _decode_aggregate_plan_v1(payload: object) -> _sc.AggregatePlanV1:
    """Decode explicit weighting and unavailable-data policies."""

    return _sc.AggregatePlanV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "aggregate_plan_id": _registry_text,
        "repetition_weighting": _registry_text,
        "case_weighting": _registry_text,
        "source_task_weighting": _registry_text,
        "cluster_unit": _registry_text,
        "bootstrap_unit": _registry_text,
        "infrastructure_invalid_policy": _registry_text,
        "cross_track_overall_score": _registry_boolean,
        "cross_benchmark_progress_average": _registry_boolean,
    }))


def _decode_comparison_family_v1(payload: object) -> _sc.ComparisonFamilyV1:
    """Decode the preregistered paired family and closed endpoint vocabulary."""

    return _sc.ComparisonFamilyV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "family_id": _registry_text,
        "track": _registry_enum(TraceFixTrack),
        "benchmark_id": _registry_text,
        "benchmark_version": _registry_text,
        "adapter_version": _registry_text,
        "execution_semantics": _registry_enum(ExecutionSemantics),
        "endpoint_ids": _registry_records(_registry_enum(_sc.BuiltinEndpointKey)),
        "treatment_ids": _registry_texts,
        "budget_ids": _registry_texts,
        "system_cell_ids": _registry_texts,
        "multiplicity_correction": _registry_enum(_sc.MultiplicityCorrection),
        "binary_reduction_policy": _registry_enum(_sc.BinaryReductionPolicy),
        "binary_reducer_ref": lambda value: _require_optional_text(
            value, label="binary reducer reference"
        ),
        "paired_roster_required": _registry_boolean,
    }))


def decode_scorer_registry_v1(payload: object) -> _sc.ScorerRegistryV1:
    """Decode all frozen policy contracts and enforce their reference closure."""

    return _sc.ScorerRegistryV1(**_decode_registry_fields(payload, {
        "schema": _registry_text,
        "registry_id": _registry_text,
        "suite_id": _registry_text,
        "suite_version": _registry_text,
        "population_id": _registry_text,
        "implementation_status": _registry_text,
        "scorers": _registry_records(_decode_scorer_registration_v1),
        "hit_cutoffs": _registry_records(_registry_integer),
        "primary_k": _registry_integer,
        "allowed_finding_kinds": _registry_records(_registry_enum(TraceFindingKind)),
        "progress_metrics": _registry_records(_decode_progress_metric_contract_v1),
        "verifiers": _registry_records(_decode_verifier_contract_v1),
        "oracles": _registry_records(_decode_oracle_contract_v1),
        "budgets": _registry_records(_decode_resource_budget_contract_v1),
        "treatments": _registry_records(_decode_treatment_arm_registration_v1),
        "system_cells": _registry_records(_decode_system_cell_registration_v1),
        "aggregate_plan": _decode_aggregate_plan_v1,
        "comparison_families": _registry_records(_decode_comparison_family_v1),
        "source_refs": _registry_texts,
        "registration_receipt_refs": _registry_texts,
    }))


def decode_benchmark_case_ref_v1(payload: object) -> BenchmarkCaseRefV1:
    """Decode the complete scorer-side case identity record."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema",
                "benchmark_id",
                "benchmark_version",
                "adapter_version",
                "case_id",
                "source_denominator_entry_id",
                "source_task_id",
                "task_group_id",
                "split",
                "case_ordinal",
                "source_artifact_ref",
                "population_role",
                "exposure_status",
                "case_origin",
            }
        ),
        label="benchmark case reference",
    )
    return BenchmarkCaseRefV1(
        schema=_require_schema(
            data["schema"],
            BENCHMARK_CASE_REF_SCHEMA,
            label="benchmark case reference schema",
        ),
        benchmark_id=_require_text(data["benchmark_id"], label="benchmark_id"),
        benchmark_version=_require_text(
            data["benchmark_version"],
            label="benchmark_version",
        ),
        adapter_version=_require_text(
            data["adapter_version"],
            label="adapter_version",
        ),
        case_id=_require_text(data["case_id"], label="case_id"),
        source_denominator_entry_id=_require_text(
            data["source_denominator_entry_id"],
            label="source_denominator_entry_id",
        ),
        source_task_id=_require_text(data["source_task_id"], label="source_task_id"),
        task_group_id=_require_text(data["task_group_id"], label="task_group_id"),
        split=_decode_enum(data["split"], BenchmarkSplit, label="benchmark split"),
        case_ordinal=_require_integer(data["case_ordinal"], label="case_ordinal"),
        source_artifact_ref=_require_text(
            data["source_artifact_ref"],
            label="source_artifact_ref",
        ),
        population_role=_decode_enum(
            data["population_role"],
            PopulationRole,
            label="population_role",
        ),
        exposure_status=_decode_enum(
            data["exposure_status"],
            ExposureStatus,
            label="exposure_status",
        ),
        case_origin=_decode_enum(data["case_origin"], CaseOrigin, label="case_origin"),
    )


def decode_source_denominator_entry_v1(
    payload: object,
) -> SourceDenominatorEntryV1:
    """Decode one complete source-population outcome row."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema",
                "benchmark_id",
                "benchmark_version",
                "adapter_version",
                "population_id",
                "entry_id",
                "source_task_id",
                "task_group_id",
                "source_artifact_ref",
                "split",
                "population_role",
                "outcome_status",
                "exposure_status",
                "protocol_complete",
                "source_receipt_refs",
                "materialization_status",
                "materialized_case_ids",
                "status_detail",
            }
        ),
        label="source denominator entry",
    )
    return SourceDenominatorEntryV1(
        schema=_require_schema(
            data["schema"],
            SOURCE_DENOMINATOR_ENTRY_SCHEMA,
            label="source denominator schema",
        ),
        benchmark_id=_require_text(data["benchmark_id"], label="benchmark_id"),
        benchmark_version=_require_text(
            data["benchmark_version"],
            label="benchmark_version",
        ),
        adapter_version=_require_text(
            data["adapter_version"],
            label="adapter_version",
        ),
        population_id=_require_text(data["population_id"], label="population_id"),
        entry_id=_require_text(data["entry_id"], label="entry_id"),
        source_task_id=_require_text(data["source_task_id"], label="source_task_id"),
        task_group_id=_require_text(data["task_group_id"], label="task_group_id"),
        source_artifact_ref=_require_text(
            data["source_artifact_ref"],
            label="source_artifact_ref",
        ),
        split=_decode_enum(data["split"], BenchmarkSplit, label="benchmark split"),
        population_role=_decode_enum(
            data["population_role"],
            PopulationRole,
            label="population_role",
        ),
        outcome_status=_decode_enum(
            data["outcome_status"],
            SourceOutcomeStatus,
            label="outcome_status",
        ),
        exposure_status=_decode_enum(
            data["exposure_status"],
            ExposureStatus,
            label="exposure_status",
        ),
        protocol_complete=_require_boolean(
            data["protocol_complete"],
            label="protocol_complete",
        ),
        source_receipt_refs=_decode_text_tuple(
            data["source_receipt_refs"],
            label="source_receipt_refs",
        ),
        materialization_status=_decode_enum(
            data["materialization_status"],
            MaterializationStatus,
            label="materialization_status",
        ),
        materialized_case_ids=_decode_text_tuple(
            data["materialized_case_ids"],
            label="materialized_case_ids",
        ),
        status_detail=_require_optional_text(
            data["status_detail"],
            label="status_detail",
        ),
    )


def _decode_projection_field_decision(payload: object) -> ProjectionFieldDecisionV1:
    """Decode one scorer-only projection decision with its evidence."""

    data = _require_object_fields(
        payload,
        frozenset({"field_name", "status", "evidence_refs", "detail"}),
        label="projection field decision",
    )
    return ProjectionFieldDecisionV1(
        field_name=_require_text(data["field_name"], label="projection field_name"),
        status=_decode_enum(
            data["status"],
            FieldCoverageStatus,
            label="projection field status",
        ),
        evidence_refs=_decode_text_tuple(
            data["evidence_refs"],
            label="projection evidence_refs",
        ),
        detail=_require_optional_text(data["detail"], label="projection detail"),
    )


def decode_projection_receipt_v1(
    payload: object,
) -> PublicTraceProjectionReceiptV1:
    """Decode the scorer-only evidence behind one public projection."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema",
                "projection_receipt_id",
                "case_id",
                "benchmark_id",
                "adapter_version",
                "policy_id",
                "field_decisions",
                "source_visibility_receipt_refs",
                "rejected_private_fields",
                "opaque_roster_receipt_ref",
                "public_cutoff_event_ref",
            }
        ),
        label="public trace projection receipt",
    )
    return PublicTraceProjectionReceiptV1(
        schema=_require_schema(
            data["schema"],
            PUBLIC_TRACE_PROJECTION_RECEIPT_SCHEMA,
            label="projection receipt schema",
        ),
        projection_receipt_id=_require_text(
            data["projection_receipt_id"],
            label="projection_receipt_id",
        ),
        case_id=_require_text(data["case_id"], label="projection case_id"),
        benchmark_id=_require_text(
            data["benchmark_id"],
            label="projection benchmark_id",
        ),
        adapter_version=_require_text(
            data["adapter_version"],
            label="projection adapter_version",
        ),
        policy_id=_require_text(data["policy_id"], label="projection policy_id"),
        field_decisions=tuple(
            _decode_projection_field_decision(item)
            for item in _require_list(
                data["field_decisions"],
                label="projection field_decisions",
            )
        ),
        source_visibility_receipt_refs=_decode_text_tuple(
            data["source_visibility_receipt_refs"],
            label="source_visibility_receipt_refs",
        ),
        rejected_private_fields=_decode_text_tuple(
            data["rejected_private_fields"],
            label="rejected_private_fields",
        ),
        opaque_roster_receipt_ref=_require_text(
            data["opaque_roster_receipt_ref"],
            label="opaque_roster_receipt_ref",
        ),
        public_cutoff_event_ref=_require_text(
            data["public_cutoff_event_ref"],
            label="public_cutoff_event_ref",
        ),
    )


def decode_case_admission_v1(payload: object) -> CaseAdmissionV1:
    """Decode scorer-side track admission and qualification evidence."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema",
                "case_ref",
                "eligible_tracks",
                "scorer_fields_withheld_from_methods",
                "attribution_gold_disposition",
                "checkpoint_parity_qualified",
                "matched_three_arm_qualified",
                "state_coverage_qualified",
                "same_actor_qualified",
                "official_verifier_qualified",
                "admission_receipt_refs",
                "exclusion_reasons",
            }
        ),
        label="case admission",
    )
    schema = _require_schema(
        data["schema"],
        CASE_ADMISSION_SCHEMA,
        label="case admission schema",
    )
    raw_disposition = data["attribution_gold_disposition"]
    disposition = (
        None
        if raw_disposition is None
        else _decode_enum(
            raw_disposition,
            GoldDisposition,
            label="attribution_gold_disposition",
        )
    )
    return CaseAdmissionV1(
        schema=schema,
        case_ref=decode_benchmark_case_ref_v1(data["case_ref"]),
        eligible_tracks=tuple(
            _decode_enum(item, TraceFixTrack, label="eligible track")
            for item in _require_list(
                data["eligible_tracks"],
                label="eligible_tracks",
            )
        ),
        scorer_fields_withheld_from_methods=_require_boolean(
            data["scorer_fields_withheld_from_methods"],
            label="scorer_fields_withheld_from_methods",
        ),
        attribution_gold_disposition=disposition,
        checkpoint_parity_qualified=_require_boolean(
            data["checkpoint_parity_qualified"],
            label="checkpoint_parity_qualified",
        ),
        matched_three_arm_qualified=_require_boolean(
            data["matched_three_arm_qualified"],
            label="matched_three_arm_qualified",
        ),
        state_coverage_qualified=_require_boolean(
            data["state_coverage_qualified"],
            label="state_coverage_qualified",
        ),
        same_actor_qualified=_require_boolean(
            data["same_actor_qualified"],
            label="same_actor_qualified",
        ),
        official_verifier_qualified=_require_boolean(
            data["official_verifier_qualified"],
            label="official_verifier_qualified",
        ),
        admission_receipt_refs=_decode_text_tuple(
            data["admission_receipt_refs"],
            label="admission_receipt_refs",
        ),
        exclusion_reasons=_decode_text_tuple(
            data["exclusion_reasons"],
            label="exclusion_reasons",
        ),
    )


def _decode_event_groups(value: object) -> tuple[tuple[str, ...], ...]:
    """Decode alternative conjunctive event groups without flattening them."""

    return tuple(
        _decode_text_tuple(item, label="accepted event group")
        for item in _require_list(value, label="accepted_event_groups")
    )


def decode_attribution_gold_v1(payload: object) -> AttributionGoldV1:
    """Decode one scorer-only attribution or optional forward-root oracle."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema",
                "case_ref",
                "disposition",
                "label_kind",
                "accepted_event_groups",
                "authority",
                "authority_receipt_refs",
                "claim_scope",
                "causal_distance",
                "distance_metric_id",
                "reviewers_saw_method_outputs",
            }
        ),
        label="attribution gold",
    )
    schema = _require_schema(
        data["schema"],
        ATTRIBUTION_GOLD_SCHEMA,
        label="attribution gold schema",
    )
    raw_label_kind = data["label_kind"]
    label_kind = (
        None
        if raw_label_kind is None
        else _decode_enum(raw_label_kind, GoldLabelKind, label="gold label_kind")
    )
    return AttributionGoldV1(
        schema=schema,
        case_ref=decode_benchmark_case_ref_v1(data["case_ref"]),
        disposition=_decode_enum(
            data["disposition"],
            GoldDisposition,
            label="gold disposition",
        ),
        label_kind=label_kind,
        accepted_event_groups=_decode_event_groups(data["accepted_event_groups"]),
        authority=_decode_enum(
            data["authority"],
            GoldAuthority,
            label="gold authority",
        ),
        authority_receipt_refs=_decode_text_tuple(
            data["authority_receipt_refs"],
            label="authority_receipt_refs",
        ),
        claim_scope=_require_text(data["claim_scope"], label="claim_scope"),
        causal_distance=_require_optional_integer(
            data["causal_distance"],
            label="causal_distance",
        ),
        distance_metric_id=_require_optional_text(
            data["distance_metric_id"],
            label="distance_metric_id",
        ),
        reviewers_saw_method_outputs=_require_boolean(
            data["reviewers_saw_method_outputs"],
            label="reviewers_saw_method_outputs",
        ),
    )


def decode_recovery_receipt_v1(payload: object) -> TraceFixRecoveryReceiptV1:
    """Decode explicit operator evidence for binding an unbound directory pair."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema",
                "receipt_id",
                "suite_id",
                "suite_version",
                "population_id",
                "public_directory_name",
                "scorer_directory_name",
                "confirmed_at",
                "prior_state",
                "previous_publisher_terminated",
            }
        ),
        label="TraceFix recovery receipt",
    )
    return TraceFixRecoveryReceiptV1(
        schema=_require_schema(
            data["schema"],
            TRACEFIX_RECOVERY_RECEIPT_SCHEMA,
            label="recovery receipt schema",
        ),
        receipt_id=_require_text(data["receipt_id"], label="recovery receipt_id"),
        suite_id=_require_text(data["suite_id"], label="recovery suite_id"),
        suite_version=_require_text(
            data["suite_version"],
            label="recovery suite_version",
        ),
        population_id=_require_text(
            data["population_id"],
            label="recovery population_id",
        ),
        public_directory_name=_require_text(
            data["public_directory_name"],
            label="recovery public_directory_name",
        ),
        scorer_directory_name=_require_text(
            data["scorer_directory_name"],
            label="recovery scorer_directory_name",
        ),
        confirmed_at=_require_text(data["confirmed_at"], label="confirmed_at"),
        prior_state=_decode_enum(
            data["prior_state"],
            RecoveryPriorState,
            label="recovery prior_state",
        ),
        previous_publisher_terminated=_require_boolean(
            data["previous_publisher_terminated"],
            label="previous_publisher_terminated",
        ),
    )


def decode_release_record_v1(payload: object) -> TraceFixReleaseRecordV1:
    """Decode the scorer-only binding for one immutable directory pair."""

    data = _require_object_fields(
        payload,
        frozenset(
            {
                "schema",
                "suite_id",
                "suite_version",
                "population_id",
                "public_directory_name",
                "scorer_directory_name",
                "public_index_schema",
                "scorer_index_schema",
                "scorer_registry_id",
                "case_count",
                "case_ids",
                "source_denominator_entry_ids",
                "publication_status",
                "bound_at",
                "binding_mode",
                "recovery_receipt",
                "implementation_status",
            }
        ),
        label="TraceFix release record",
    )
    raw_recovery_receipt = data["recovery_receipt"]
    recovery_receipt = (
        None
        if raw_recovery_receipt is None
        else decode_recovery_receipt_v1(raw_recovery_receipt)
    )
    return TraceFixReleaseRecordV1(
        scorer_registry_id=_require_text(data["scorer_registry_id"], label="scorer_registry_id"),
        schema=_require_schema(
            data["schema"],
            TRACEFIX_RELEASE_RECORD_SCHEMA,
            label="release record schema",
        ),
        suite_id=_require_text(data["suite_id"], label="release suite_id"),
        suite_version=_require_text(
            data["suite_version"],
            label="release suite_version",
        ),
        population_id=_require_text(
            data["population_id"],
            label="release population_id",
        ),
        public_directory_name=_require_text(
            data["public_directory_name"],
            label="public_directory_name",
        ),
        scorer_directory_name=_require_text(
            data["scorer_directory_name"],
            label="scorer_directory_name",
        ),
        public_index_schema=_require_schema(
            data["public_index_schema"],
            PUBLIC_DATASET_INDEX_SCHEMA,
            label="public_index_schema",
        ),
        scorer_index_schema=_require_schema(
            data["scorer_index_schema"],
            SCORER_DATASET_INDEX_SCHEMA,
            label="scorer_index_schema",
        ),
        case_count=_require_integer(data["case_count"], label="release case_count"),
        case_ids=_decode_text_tuple(data["case_ids"], label="release case_ids"),
        source_denominator_entry_ids=_decode_text_tuple(
            data["source_denominator_entry_ids"],
            label="release source_denominator_entry_ids",
        ),
        publication_status=_decode_enum(
            data["publication_status"],
            ReleasePublicationStatus,
            label="publication_status",
        ),
        bound_at=_require_text(data["bound_at"], label="bound_at"),
        binding_mode=_decode_enum(
            data["binding_mode"],
            ReleaseBindingMode,
            label="binding_mode",
        ),
        recovery_receipt=recovery_receipt,
        implementation_status=_decode_enum(
            data["implementation_status"],
            ReleaseImplementationStatus,
            label="implementation_status",
        ),
    )


__all__ = [
    "decode_attribution_gold_v1",
    "decode_benchmark_case_ref_v1",
    "decode_case_admission_v1",
    "decode_locator_usage_v1",
    "decode_projection_receipt_v1",
    "decode_public_index_v1",
    "decode_public_trace_packet_v1",
    "decode_recovery_receipt_v1",
    "decode_release_record_v1",
    "decode_scorer_index_v1",
    "decode_scorer_registry_v1",
    "decode_source_denominator_entry_v1",
    "load_json_object_strict",
    "load_registered_json_object_strict",
    "require_real_directory",
]
