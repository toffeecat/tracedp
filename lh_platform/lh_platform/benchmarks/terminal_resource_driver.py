"""Observe, retain, and conditionally restore Terminal resources separately."""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol

from lh_platform.benchmarks.terminal_public_state import (
    validate_actor_visible_path,
    validate_configured_resource_path,
)
from lh_platform.benchmarks.terminal_replay_population import (
    TerminalPublicPathCompilation,
    is_terminal_database_path,
)
from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    ResourceKind,
    ResourceObservation,
)

_OBSERVATION_TIMEOUT_SECONDS = 60
_RETENTION_SCHEMA = "terminal_resource_retention_v4"
TERMINAL_RESOURCE_OBSERVATION_V1 = "terminal_resource_metadata_v1"
TERMINAL_RESOURCE_OBSERVATION_V2 = "terminal_resource_metadata_v2"
_SQLITE_CONVENTIONAL_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_OBSERVATION_KEYS = {
    "resource_id",
    "kind",
    "observed",
    "exists",
    "size_bytes",
    "modified_time",
    "modified_time_detail",
    "change_time",
    "change_time_detail",
    "inode",
    "file_type",
    "mode",
    "group_id",
}
_DIRECTORY_SUMMARY_KEYS = {
    "descendant_entry_count",
    "descendant_regular_file_count",
    "descendant_total_file_bytes",
    "descendant_latest_modified_time",
    "descendant_latest_change_time",
    "descendant_observation_complete",
}


def _observation_payload(value: ResourceObservation) -> dict[str, object]:
    """Persist every source identity field used by retention's stability check."""

    payload: dict[str, object] = {
        "resource_id": value.resource_id,
        "kind": value.kind.value,
        "observed": value.observed,
        "exists": value.exists,
        "size_bytes": value.size_bytes,
        "modified_time": value.modified_time,
        "modified_time_detail": value.modified_time_detail,
        "change_time": value.change_time,
        "change_time_detail": value.change_time_detail,
        "inode": value.inode,
        "file_type": value.file_type,
        "mode": value.mode,
        "group_id": value.group_id,
    }
    if value.descendant_entry_count is not None:
        payload.update(
            {
                "descendant_entry_count": value.descendant_entry_count,
                "descendant_regular_file_count": value.descendant_regular_file_count,
                "descendant_total_file_bytes": value.descendant_total_file_bytes,
                "descendant_latest_modified_time": value.descendant_latest_modified_time,
                "descendant_latest_change_time": value.descendant_latest_change_time,
                "descendant_observation_complete": value.descendant_observation_complete,
            }
        )
    return payload


def _decode_observation(value: object) -> ResourceObservation:
    """Reconstruct one persisted source observation without silent coercion."""

    if not isinstance(value, dict) or set(value) not in {
        frozenset(_OBSERVATION_KEYS),
        frozenset(_OBSERVATION_KEYS | _DIRECTORY_SUMMARY_KEYS),
    }:
        raise ValueError("retention source observation is missing")
    optional_ints = ("size_bytes", "modified_time", "change_time", "inode")
    optional_strings = (
        "modified_time_detail",
        "change_time_detail",
        "file_type",
        "mode",
        "group_id",
    )
    if (
        not isinstance(value.get("resource_id"), str)
        or not value["resource_id"]
        or type(value.get("observed")) is not bool
        or value.get("exists") is not None
        and type(value.get("exists")) is not bool
        or any(value.get(key) is not None and type(value.get(key)) is not int for key in optional_ints)
        or any(
            value.get(key) is not None and not isinstance(value.get(key), str)
            for key in optional_strings
        )
    ):
        raise ValueError("retention source observation is invalid")
    return ResourceObservation(
        resource_id=str(value.get("resource_id", "")),
        kind=ResourceKind(value.get("kind")),
        observed=value.get("observed") is True,
        exists=value.get("exists") if type(value.get("exists")) is bool else None,
        size_bytes=value.get("size_bytes") if type(value.get("size_bytes")) is int else None,
        modified_time=(
            value.get("modified_time") if type(value.get("modified_time")) is int else None
        ),
        modified_time_detail=(
            str(value["modified_time_detail"])
            if value.get("modified_time_detail") is not None
            else None
        ),
        change_time=(
            value.get("change_time") if type(value.get("change_time")) is int else None
        ),
        change_time_detail=(
            str(value["change_time_detail"])
            if value.get("change_time_detail") is not None
            else None
        ),
        inode=value.get("inode") if type(value.get("inode")) is int else None,
        file_type=(str(value["file_type"]) if value.get("file_type") is not None else None),
        mode=str(value["mode"]) if value.get("mode") is not None else None,
        group_id=str(value["group_id"]) if value.get("group_id") is not None else None,
        descendant_entry_count=(
            int(value["descendant_entry_count"])
            if isinstance(value.get("descendant_entry_count"), int)
            else None
        ),
        descendant_regular_file_count=(
            int(value["descendant_regular_file_count"])
            if isinstance(value.get("descendant_regular_file_count"), int)
            else None
        ),
        descendant_total_file_bytes=(
            int(value["descendant_total_file_bytes"])
            if isinstance(value.get("descendant_total_file_bytes"), int)
            else None
        ),
        descendant_latest_modified_time=(
            str(value["descendant_latest_modified_time"])
            if value.get("descendant_latest_modified_time") is not None
            else None
        ),
        descendant_latest_change_time=(
            str(value["descendant_latest_change_time"])
            if value.get("descendant_latest_change_time") is not None
            else None
        ),
        descendant_observation_complete=(
            value.get("descendant_observation_complete")
            if type(value.get("descendant_observation_complete")) is bool
            else None
        ),
    )


class TerminalResourceEnvironment(Protocol):
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> TerminalResourceExecResult: ...

    async def download_file(self, source_path: str, target_path: Path | str) -> None: ...

    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...


class TerminalResourceExecResult(Protocol):
    stdout: str | None
    return_code: int


@dataclass(frozen=True)
class ResourceSpec:
    """Bind one configured observation resource to provenance and a resource family."""

    resource_id: str
    kind: ResourceKind
    provenance: str
    group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_id",
            validate_configured_resource_path(self.resource_id),
        )
        if not self.provenance:
            raise ValueError("resource provenance is required")
        if "|" in self.resource_id or "\n" in self.resource_id:
            raise ValueError("resource path contains an observation delimiter")


class RetentionStatus(StrEnum):
    FULL_BYTES = "full_bytes"
    METADATA_ONLY = "metadata_only"
    MISSING = "missing"
    FAILED = "failed"


@dataclass(frozen=True)
class RetentionRecord:
    resource_id: str
    status: RetentionStatus
    bytes_retained: int
    storage_path: Path | None
    observed_size_bytes: int | None
    observed_mode: str | None
    reason: str | None
    source_before: ResourceObservation
    source_after_download: ResourceObservation | None
    kernel_initial_observation: ResourceObservation | None
    storage_modified_time_ns: int | None = None
    storage_change_time_ns: int | None = None
    storage_inode: int | None = None


@dataclass(frozen=True)
class RestorationCapability:
    resource_id: str
    supported: bool
    scope: str
    conditions: tuple[str, ...]
    cost_bytes: int
    reason: str | None


def expand_sqlite_resource_family_specs(
    resources: Sequence[ResourceSpec],
) -> tuple[ResourceSpec, ...]:
    """Add SQLite's documented sidecar names before the first Actor action."""

    expanded = list(resources)
    known_ids = {item.resource_id for item in resources}
    for database in resources:
        if database.kind is not ResourceKind.DATABASE:
            continue
        group_id = database.group_id or f"sqlite:{database.resource_id}"
        for suffix in _SQLITE_CONVENTIONAL_SIDECAR_SUFFIXES:
            resource_id = database.resource_id + suffix
            if resource_id in known_ids:
                continue
            expanded.append(
                ResourceSpec(
                    resource_id=resource_id,
                    kind=ResourceKind.DATABASE_SIDECAR,
                    provenance="sqlite-official-sidecar-convention-v1",
                    group_id=group_id,
                )
            )
            known_ids.add(resource_id)
    return tuple(expanded)


class TerminalResourceDriver:
    """Implement resource-family mechanics without task-specific diagnosis.

    Metadata observation never downloads file content. Retention is a separate,
    explicitly budgeted operation, and restore is available only for retained
    exact regular-file bytes with an exact readback check.
    """

    def __init__(
        self,
        *,
        environment: TerminalResourceEnvironment,
        resources: Sequence[ResourceSpec],
        retention_root: Path,
        receipt_path: Path,
        retention_budget_bytes: int,
        observation_schema: str = TERMINAL_RESOURCE_OBSERVATION_V2,
    ) -> None:
        if retention_budget_bytes < 0:
            raise ValueError("retention budget cannot be negative")
        if observation_schema not in {
            TERMINAL_RESOURCE_OBSERVATION_V1,
            TERMINAL_RESOURCE_OBSERVATION_V2,
        }:
            raise ValueError("terminal resource observation schema is unsupported")
        self.environment = environment
        self.resources = tuple(resources)
        resource_ids = [item.resource_id for item in self.resources]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("resource ids must be unique")
        self._specs = {item.resource_id: item for item in self.resources}
        self.retention_root = retention_root
        self.receipt_path = receipt_path
        self.retention_budget_bytes = retention_budget_bytes
        self.observation_schema = observation_schema
        self.retention_records: dict[str, RetentionRecord] = {}
        self._retained_total = 0

    async def _observe_specs(
        self, specs: Sequence[ResourceSpec]
    ) -> tuple[ResourceObservation, ...]:
        """Read one metadata boundary through a single cross-runtime call."""

        if not specs:
            return ()
        result = await self.environment.exec(
            "stat -c '%n|%F|%a|%s|%Y|%y|%Z|%z|%i' -- "
            + " ".join(shlex.quote(spec.resource_id) for spec in specs)
            + " 2>/dev/null",
            timeout_sec=_OBSERVATION_TIMEOUT_SECONDS,
        )
        if result.return_code not in {0, 1}:
            raise RuntimeError("terminal resource metadata command failed")
        by_path: dict[str, tuple[str, str, str, str, str, str, str, str]] = {}
        for line in str(result.stdout or "").splitlines():
            parts = line.split("|", maxsplit=8)
            if len(parts) != 9 or parts[0] in by_path:
                raise RuntimeError("terminal resource metadata output is invalid")
            by_path[parts[0]] = (
                parts[1],
                parts[2],
                parts[3],
                parts[4],
                parts[5],
                parts[6],
                parts[7],
                parts[8],
            )
        if not set(by_path) <= {spec.resource_id for spec in specs}:
            raise RuntimeError("terminal resource metadata escaped its declared scope")
        observations = []
        for spec in specs:
            values = by_path.get(spec.resource_id)
            if values is None:
                observations.append(
                    ResourceObservation(
                        resource_id=spec.resource_id,
                        kind=spec.kind,
                        observed=True,
                        exists=False,
                        group_id=spec.group_id,
                    )
                )
                continue
            (
                file_type,
                mode,
                size,
                modified_time,
                modified_time_detail,
                change_time,
                change_time_detail,
                inode,
            ) = values
            observation = ResourceObservation(
                    resource_id=spec.resource_id,
                    kind=spec.kind,
                    observed=True,
                    exists=True,
                    size_bytes=int(size),
                    modified_time=int(modified_time),
                    modified_time_detail=modified_time_detail,
                    change_time=int(change_time),
                    change_time_detail=change_time_detail,
                    inode=int(inode),
                    file_type=file_type,
                    mode=mode,
                    group_id=spec.group_id,
                )
            if (
                file_type == "directory"
                and self.observation_schema == TERMINAL_RESOURCE_OBSERVATION_V2
            ):
                observation = replace(
                    observation,
                    **await self._observe_directory_descendants(spec.resource_id),
                )
            observations.append(observation)
        return tuple(observations)

    async def _observe_directory_descendants(self, resource_id: str) -> dict[str, object]:
        """Summarize directory members without reading or hashing file contents."""

        find_command = (
            "find -- "
            + shlex.quote(resource_id)
            + " -xdev -mindepth 1 -printf '%y|%s|%T@|%C@\\n'"
        )
        awk_program = (
            "BEGIN { entries=0; files=0; bytes=0; max_m=0; max_c=0 } "
            "{ entries += 1; if ($1 == \"f\") { files += 1; bytes += $2 }; "
            "if ($3 > max_m) max_m=$3; if ($4 > max_c) max_c=$4 } "
            "END { printf \"%d|%d|%.0f|%.9f|%.9f\\n\", "
            "entries, files, bytes, max_m, max_c }"
        )
        pipeline = "set -o pipefail; " + find_command + " | awk -F '|' " + shlex.quote(
            awk_program
        )
        result = await self.environment.exec(
            "bash -c " + shlex.quote(pipeline),
            timeout_sec=_OBSERVATION_TIMEOUT_SECONDS,
        )
        parts = str(result.stdout or "").strip().split("|")
        if result.return_code != 0 or len(parts) != 5:
            raise RuntimeError("terminal directory descendant observation failed")
        try:
            entries, files, total_bytes = (int(parts[index]) for index in range(3))
            latest_modified, latest_changed = parts[3], parts[4]
            float(latest_modified)
            float(latest_changed)
        except ValueError as error:
            raise RuntimeError("terminal directory descendant summary is invalid") from error
        return {
            "descendant_entry_count": entries,
            "descendant_regular_file_count": files,
            "descendant_total_file_bytes": total_bytes,
            "descendant_latest_modified_time": latest_modified,
            "descendant_latest_change_time": latest_changed,
            "descendant_observation_complete": True,
        }

    async def _observe_one(self, spec: ResourceSpec) -> ResourceObservation:
        """Observe one resource through the same batched metadata protocol."""

        return (await self._observe_specs((spec,)))[0]

    async def observe(self) -> tuple[ResourceObservation, ...]:
        """Observe every declared resource regardless of retention eligibility."""

        return await self._observe_specs(self.resources)

    def unknown_observations(self) -> tuple[ResourceObservation, ...]:
        """Project an observation failure without preserving stale resource facts."""

        return tuple(
            ResourceObservation(
                resource_id=spec.resource_id,
                kind=spec.kind,
                observed=False,
                exists=None,
                group_id=spec.group_id,
            )
            for spec in self.resources
        )

    def _append_retention_receipt(self, record: RetentionRecord) -> None:
        """Persist scope and cost without treating metadata as saved bytes."""

        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        row = self._retention_receipt_payload(record)
        with self.receipt_path.open("a") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _retention_receipt_payload(record: RetentionRecord) -> dict[str, object]:
        """Serialize one retention record with both source observation boundaries."""

        return {
            "schema_version": _RETENTION_SCHEMA,
            "resource_id": record.resource_id,
            "status": record.status.value,
            "bytes_retained": record.bytes_retained,
            "storage_path": (
                str(record.storage_path) if record.storage_path is not None else None
            ),
            "observed_size_bytes": record.observed_size_bytes,
            "observed_mode": record.observed_mode,
            "reason": record.reason,
            "source_before": _observation_payload(record.source_before),
            "source_after_download": (
                _observation_payload(record.source_after_download)
                if record.source_after_download is not None
                else None
            ),
            "kernel_initial_observation": (
                _observation_payload(record.kernel_initial_observation)
                if record.kernel_initial_observation is not None
                else None
            ),
            "storage_modified_time_ns": record.storage_modified_time_ns,
            "storage_change_time_ns": record.storage_change_time_ns,
            "storage_inode": record.storage_inode,
        }

    def _rewrite_retention_receipts(self) -> None:
        """Replace the initial receipt after binding bytes to the Kernel boundary."""

        staging = self.receipt_path.with_suffix(".jsonl.writing")
        with staging.open("w") as output:
            for spec in self.resources:
                output.write(
                    json.dumps(
                        self._retention_receipt_payload(
                            self.retention_records[spec.resource_id]
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        staging.replace(self.receipt_path)

    async def retain_initial(self) -> tuple[RetentionRecord, ...]:
        """Retain initial exact bytes in resource order under one explicit budget."""

        if self.retention_records:
            raise RuntimeError("initial resource retention already ran")
        observations = await self.observe()
        records = []
        self.retention_root.mkdir(parents=True, exist_ok=True)
        for index, observation in enumerate(observations):
            storage_path: Path | None = None
            reason: str | None = None
            source_after: ResourceObservation | None = None
            if not observation.exists:
                status = RetentionStatus.MISSING
                retained = 0
            elif not observation.resource_id.startswith("/app/"):
                status = RetentionStatus.METADATA_ONLY
                retained = 0
                reason = "resource is outside exact-retention workspace scope"
            elif observation.file_type != "regular file":
                status = RetentionStatus.METADATA_ONLY
                retained = 0
                reason = "resource is not an exact regular file"
            elif observation.size_bytes is None:
                status = RetentionStatus.FAILED
                retained = 0
                reason = "resource size is unknown"
            elif self._retained_total + observation.size_bytes > self.retention_budget_bytes:
                status = RetentionStatus.METADATA_ONLY
                retained = 0
                reason = "retention byte budget is exhausted"
            else:
                storage_path = self.retention_root / f"resource-{index:04d}.bin"
                try:
                    await self.environment.download_file(
                        observation.resource_id,
                        storage_path,
                    )
                except Exception as error:
                    storage_path.unlink(missing_ok=True)
                    storage_path = None
                    status = RetentionStatus.FAILED
                    retained = 0
                    reason = f"exact-byte retention failed: {type(error).__name__}"
                else:
                    retained = storage_path.stat().st_size
                    try:
                        source_after = await self._observe_one(
                            self._specs[observation.resource_id]
                        )
                    except Exception as error:
                        source_after = None
                        reason = (
                            "post-retention source observation failed: "
                            f"{type(error).__name__}"
                        )
                    if source_after is None or source_after != observation:
                        storage_path.unlink(missing_ok=True)
                        storage_path = None
                        status = RetentionStatus.FAILED
                        retained = 0
                        reason = reason or (
                            "resource identity changed while exact bytes were retained"
                        )
                    elif retained != observation.size_bytes:
                        storage_path.unlink(missing_ok=True)
                        storage_path = None
                        status = RetentionStatus.FAILED
                        retained = 0
                        reason = "resource changed while exact bytes were retained"
                    else:
                        status = RetentionStatus.FULL_BYTES
                        self._retained_total += retained
            record = RetentionRecord(
                resource_id=observation.resource_id,
                status=status,
                bytes_retained=retained,
                storage_path=storage_path,
                observed_size_bytes=observation.size_bytes,
                observed_mode=observation.mode,
                reason=reason,
                source_before=observation,
                source_after_download=source_after,
                kernel_initial_observation=None,
                storage_modified_time_ns=(
                    storage_path.stat().st_mtime_ns
                    if storage_path is not None
                    else None
                ),
                storage_change_time_ns=(
                    storage_path.stat().st_ctime_ns
                    if storage_path is not None
                    else None
                ),
                storage_inode=(
                    storage_path.stat().st_ino if storage_path is not None else None
                ),
            )
            self.retention_records[record.resource_id] = record
            records.append(record)
            self._append_retention_receipt(record)
        return tuple(records)

    def bind_initial_observations(
        self, observations: Sequence[ResourceObservation]
    ) -> tuple[RetentionRecord, ...]:
        """Disable stale retained bytes before those observations initialize the Kernel."""

        if [item.resource_id for item in observations] != [
            item.resource_id for item in self.resources
        ]:
            raise ValueError("Kernel observations do not cover retained resources in order")
        for observation in observations:
            record = self.retention_records.get(observation.resource_id)
            if record is None:
                raise ValueError("initial retention is incomplete")
            bound_record = replace(record, kernel_initial_observation=observation)
            if record.status is not RetentionStatus.FULL_BYTES:
                self.retention_records[observation.resource_id] = bound_record
                continue
            if record.source_after_download == observation:
                self.retention_records[observation.resource_id] = bound_record
                continue
            if record.storage_path is not None:
                record.storage_path.unlink(missing_ok=True)
            self._retained_total -= record.bytes_retained
            self.retention_records[observation.resource_id] = replace(
                record,
                status=RetentionStatus.FAILED,
                bytes_retained=0,
                storage_path=None,
                reason="resource changed after retention before Kernel initialization",
                kernel_initial_observation=observation,
                storage_modified_time_ns=None,
                storage_change_time_ns=None,
                storage_inode=None,
            )
        self._rewrite_retention_receipts()
        return tuple(self.retention_records[item.resource_id] for item in self.resources)

    def load_initial_retention(self) -> tuple[RetentionRecord, ...]:
        """Restore retained-byte metadata from its ordered durable receipt."""

        if self.retention_records:
            raise RuntimeError("initial resource retention already loaded")
        if not self.receipt_path.is_file():
            raise FileNotFoundError(self.receipt_path)
        rows = [json.loads(line) for line in self.receipt_path.read_text().splitlines()]
        if len(rows) != len(self.resources):
            raise ValueError("retention receipt does not cover every declared resource")
        records = []
        for index, (spec, row) in enumerate(zip(self.resources, rows, strict=True)):
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != _RETENTION_SCHEMA
                or row.get("resource_id") != spec.resource_id
            ):
                raise ValueError("retention receipt resource order or schema differs")
            status = RetentionStatus(row["status"])
            raw_path = row.get("storage_path")
            storage_path = Path(raw_path) if isinstance(raw_path, str) else None
            expected_path = self.retention_root / f"resource-{index:04d}.bin"
            if status is RetentionStatus.FULL_BYTES:
                if storage_path != expected_path or not expected_path.is_file():
                    raise ValueError("retained resource bytes are unavailable")
                if expected_path.stat().st_size != row.get("bytes_retained"):
                    raise ValueError("retained resource byte count differs")
                if (
                    expected_path.stat().st_mtime_ns
                    != row.get("storage_modified_time_ns")
                    or expected_path.stat().st_ctime_ns
                    != row.get("storage_change_time_ns")
                    or expected_path.stat().st_ino != row.get("storage_inode")
                ):
                    raise ValueError("retained resource identity or version differs")
            elif storage_path is not None:
                raise ValueError("metadata-only retention cannot name stored bytes")
            record = RetentionRecord(
                resource_id=spec.resource_id,
                status=status,
                bytes_retained=int(row["bytes_retained"]),
                storage_path=storage_path,
                observed_size_bytes=(
                    int(row["observed_size_bytes"])
                    if row.get("observed_size_bytes") is not None
                    else None
                ),
                observed_mode=(
                    str(row["observed_mode"])
                    if row.get("observed_mode") is not None
                    else None
                ),
                reason=str(row["reason"]) if row.get("reason") is not None else None,
                source_before=_decode_observation(row.get("source_before")),
                source_after_download=(
                    _decode_observation(row["source_after_download"])
                    if row.get("source_after_download") is not None
                    else None
                ),
                kernel_initial_observation=(
                    _decode_observation(row["kernel_initial_observation"])
                    if row.get("kernel_initial_observation") is not None
                    else None
                ),
                storage_modified_time_ns=(
                    int(row["storage_modified_time_ns"])
                    if row.get("storage_modified_time_ns") is not None
                    else None
                ),
                storage_change_time_ns=(
                    int(row["storage_change_time_ns"])
                    if row.get("storage_change_time_ns") is not None
                    else None
                ),
                storage_inode=(
                    int(row["storage_inode"])
                    if row.get("storage_inode") is not None
                    else None
                ),
            )
            if (
                record.source_before.resource_id != spec.resource_id
                or record.source_before.kind is not spec.kind
                or record.source_before.group_id != spec.group_id
                or (
                    record.source_after_download is not None
                    and (
                        record.source_after_download.resource_id != spec.resource_id
                        or record.source_after_download.kind is not spec.kind
                        or record.source_after_download.group_id != spec.group_id
                    )
                )
                or (
                    record.kernel_initial_observation is not None
                    and (
                        record.kernel_initial_observation.resource_id != spec.resource_id
                        or record.kernel_initial_observation.kind is not spec.kind
                        or record.kernel_initial_observation.group_id != spec.group_id
                    )
                )
            ):
                raise ValueError("retention source observation differs from its resource")
            if status is RetentionStatus.FULL_BYTES and (
                record.source_after_download != record.source_before
                or (
                    record.kernel_initial_observation is not None
                    and record.kernel_initial_observation != record.source_before
                )
                or record.source_before.exists is not True
                or record.source_before.file_type != "regular file"
                or record.source_before.size_bytes != record.bytes_retained
                or record.source_before.modified_time is None
                or record.source_before.modified_time_detail is None
                or record.source_before.change_time is None
                or record.source_before.change_time_detail is None
                or record.source_before.inode is None
                or record.source_before.mode is None
            ):
                raise ValueError("retained bytes lack a complete stable source observation")
            self.retention_records[spec.resource_id] = record
            self._retained_total += record.bytes_retained
            records.append(record)
        return tuple(records)

    def restoration_capabilities(self) -> dict[str, RestorationCapability]:
        """Report per-resource restore support, scope, conditions, and byte cost."""

        result = {}
        for spec in self.resources:
            record = self.retention_records.get(spec.resource_id)
            supported = (
                record is not None
                and record.status is RetentionStatus.FULL_BYTES
                and record.storage_path is not None
                and record.kernel_initial_observation is not None
                and record.kernel_initial_observation == record.source_after_download
            )
            result[spec.resource_id] = RestorationCapability(
                resource_id=spec.resource_id,
                supported=supported,
                scope="one_exact_regular_file" if supported else "none",
                conditions=(
                    (
                        "retained bytes remain available",
                        "no concurrent writer changes the resource during restore and readback",
                        "other retained database-group members remain exact",
                    )
                    if supported
                    else ()
                ),
                cost_bytes=record.bytes_retained if record is not None else 0,
                reason=None if supported else "resource has no retained exact bytes",
            )
        return result

    async def _matches_initial_retention(self, spec: ResourceSpec) -> bool:
        """Compare one group member with its retained existence and exact bytes."""

        record = self.retention_records.get(spec.resource_id)
        if record is None:
            return False
        observation = await self._observe_one(spec)
        if record.status is RetentionStatus.MISSING:
            return observation.exists is False
        if (
            record.status is not RetentionStatus.FULL_BYTES
            or record.storage_path is None
            or observation.exists is not True
        ):
            return False
        return (
            await self._download_for_readback(spec.resource_id)
            == record.storage_path.read_bytes()
        )

    async def current_restoration_capabilities(
        self, resource_ids: Sequence[str]
    ) -> dict[str, RestorationCapability]:
        """Evaluate dynamic restore preconditions before exposing legal actions.

        Static retention says which bytes exist. This method additionally checks
        the live missing-resource and database-group conditions that the actual
        restore operation requires.
        """

        requested = tuple(validate_configured_resource_path(item) for item in resource_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("current restoration capability resources must be unique")
        if not set(requested) <= set(self._specs):
            raise ValueError("current restoration capability escaped declared scope")
        static = self.restoration_capabilities()
        result: dict[str, RestorationCapability] = {}
        for resource_id in requested:
            capability = static[resource_id]
            if not capability.supported:
                result[resource_id] = capability
                continue
            spec = self._specs[resource_id]
            target = await self._observe_one(spec)
            reason = None
            actual_cost_bytes = capability.cost_bytes
            if target.exists is not False:
                reason = "resource is not currently missing"
            elif spec.group_id is not None:
                group_members = tuple(
                    item
                    for item in self.resources
                    if item.group_id == spec.group_id
                )
                actual_cost_bytes = sum(
                    self.retention_records[member.resource_id].bytes_retained
                    for member in group_members
                    if member.resource_id in self.retention_records
                )
                for member in group_members:
                    if member.resource_id == resource_id:
                        continue
                    if not await self._matches_initial_retention(member):
                        reason = (
                            "database group changed after retention; automatic "
                            "restore is currently unsupported"
                        )
                        break
            result[resource_id] = RestorationCapability(
                resource_id=resource_id,
                supported=reason is None,
                scope=(capability.scope if reason is None else "none"),
                conditions=capability.conditions,
                cost_bytes=actual_cost_bytes,
                reason=reason,
            )
        return result

    async def _download_for_readback(self, resource_id: str) -> bytes:
        readback = self.retention_root / "restore-readback.bin"
        await self.environment.download_file(resource_id, readback)
        try:
            return readback.read_bytes()
        finally:
            readback.unlink(missing_ok=True)

    async def restore(
        self, resource_id: str
    ) -> tuple[ResourceObservation, ResourceObservation]:
        """Restore one retained file in the live environment and verify exact bytes."""

        resource_id = validate_actor_visible_path(resource_id)
        if resource_id not in self._specs:
            raise ValueError("resource is outside the declared observation scope")
        record = self.retention_records.get(resource_id)
        if (
            record is None
            or record.status is not RetentionStatus.FULL_BYTES
            or record.storage_path is None
        ):
            raise ValueError("resource has no retained exact bytes")
        before = await self._observe_one(self._specs[resource_id])
        await self.environment.upload_file(record.storage_path, resource_id)
        if record.observed_mode is None:
            raise RuntimeError("retained resource lacks its original mode")
        result = await self.environment.exec(
            f"chmod {record.observed_mode} -- {shlex.quote(resource_id)}",
            timeout_sec=_OBSERVATION_TIMEOUT_SECONDS,
        )
        if result.return_code != 0:
            raise RuntimeError(f"cannot restore resource mode: {resource_id}")
        after = await self._observe_one(self._specs[resource_id])
        retained_bytes = record.storage_path.read_bytes()
        if not after.exists or await self._download_for_readback(resource_id) != retained_bytes:
            raise RuntimeError("resource restore failed exact byte readback")
        return before, after

    async def restore_group_for_resource(
        self, resource_id: str
    ) -> tuple[tuple[ResourceObservation, ...], tuple[ResourceObservation, ...]]:
        """Restore every retained member of one database resource group.

        The gateway serializes actor actions around this operation. Exact
        readback proves the resulting file set, while the capability receipt
        still reports that database transaction consistency is a separate gate.
        """

        resource_id = validate_actor_visible_path(resource_id)
        spec = self._specs.get(resource_id)
        if spec is None:
            raise ValueError("resource is outside the declared observation scope")
        if spec.group_id is None:
            before, after = await self.restore(resource_id)
            return (before,), (after,)
        members = tuple(
            item for item in self.resources if item.group_id == spec.group_id
        )
        records = []
        for member in members:
            record = self.retention_records.get(member.resource_id)
            if (
                record is None
                or record.status is not RetentionStatus.FULL_BYTES
                or record.storage_path is None
                or record.observed_mode is None
            ):
                raise ValueError("database resource group lacks retained exact bytes")
            records.append(record)
        before = tuple([await self._observe_one(member) for member in members])
        for member, record in zip(members, records, strict=True):
            assert record.storage_path is not None
            await self.environment.upload_file(record.storage_path, member.resource_id)
            result = await self.environment.exec(
                f"chmod {record.observed_mode} -- {shlex.quote(member.resource_id)}",
                timeout_sec=_OBSERVATION_TIMEOUT_SECONDS,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"cannot restore database resource mode: {member.resource_id}"
                )
        after = tuple([await self._observe_one(member) for member in members])
        for member, record, observation in zip(
            members, records, after, strict=True
        ):
            assert record.storage_path is not None
            if (
                not observation.exists
                or await self._download_for_readback(member.resource_id)
                != record.storage_path.read_bytes()
            ):
                raise RuntimeError("database resource group failed exact byte readback")
        return before, after

    async def restore_missing_prerequisite(
        self, resource_id: str
    ) -> tuple[ResourceObservation, ResourceObservation]:
        """Restore one missing input only when its retained group did not change."""

        resource_id = validate_actor_visible_path(resource_id)
        spec = self._specs.get(resource_id)
        if spec is None:
            raise ValueError("resource is outside the declared observation scope")
        target = await self._observe_one(spec)
        if target.exists is not False:
            raise ValueError("shared input protection only restores a missing resource")
        if spec.group_id is not None:
            other_members = tuple(
                item
                for item in self.resources
                if item.group_id == spec.group_id and item.resource_id != resource_id
            )
            for member in other_members:
                if not await self._matches_initial_retention(member):
                    raise ValueError(
                        "database group changed after retention; automatic restore is unsafe"
                    )
        return await self.restore(resource_id)


def compile_terminal_resource_specs(
    compilation: TerminalPublicPathCompilation,
) -> tuple[ResourceSpec, ...]:
    """Project every public path candidate into a resource-family declaration."""

    specs = []
    for candidate in compilation.candidates:
        name = PurePosixPath(candidate.path).name.lower()
        sidecar_ending = next(
            (ending for ending in ("-journal", "-shm", "-wal") if name.endswith(ending)),
            None,
        )
        if sidecar_ending is not None:
            kind = ResourceKind.DATABASE_SIDECAR
            database_path = candidate.path[: -len(sidecar_ending)]
            group_id = f"sqlite:{database_path}"
        elif is_terminal_database_path(candidate.path):
            kind = ResourceKind.DATABASE
            group_id = f"sqlite:{candidate.path}"
        else:
            kind = ResourceKind.FILE
            group_id = None
        specs.append(
            ResourceSpec(
                resource_id=candidate.path,
                kind=kind,
                provenance="+".join(candidate.provenance),
                group_id=group_id,
            )
        )
    return tuple(specs)


__all__ = [
    "ResourceSpec",
    "RestorationCapability",
    "RetentionRecord",
    "RetentionStatus",
    "TerminalResourceDriver",
    "compile_terminal_resource_specs",
    "expand_sqlite_resource_family_specs",
]
