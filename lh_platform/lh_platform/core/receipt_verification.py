"""Verify and snapshot benchmark-native evidence receipts.

Benchmarks may create native evaluation artifacts, but only this module may
promote them into the run evidence contract.  Reads stay anchored to the
``RunStorage`` directory descriptor, every path component is opened without
following symlinks, and verified bytes are copied into a platform-owned,
hash-bound snapshot before their descriptors are returned.  This is not an OS
sandbox against arbitrary same-UID code.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.core.storage import RunStorage

MAX_BENCHMARK_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_BENCHMARK_RECEIPTS = 16
MAX_TOTAL_BENCHMARK_RECEIPT_BYTES = 16 * 1024 * 1024
_RECEIPT_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _receipt_path_parts(raw_path: Any) -> tuple[str, tuple[str, ...]]:
    """Parse a receipt path without normalizing away unsafe components."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("benchmark receipt path must be a non-empty string")
    relative = Path(raw_path)
    parts = relative.parts
    if relative.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("benchmark receipt path must be a safe run-relative path")
    return relative.as_posix(), tuple(parts)


def open_run_relative_artifact(
    storage: RunStorage,
    raw_path: Any,
) -> tuple[str, int]:
    """Open a receipt through no-follow directory FDs and return its FD.

    A check-then-reopen path routine has a TOCTOU hole: a plugin can replace a
    checked receipt with a symlink before the hasher opens it.  This routine
    walks every component from the retained run directory and returns the
    already-open regular-file descriptor that will be hashed.
    """

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("platform cannot securely verify benchmark receipts on this OS")
    relative_path, parts = _receipt_path_parts(raw_path)
    readonly = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    nofollow = os.O_NOFOLLOW
    directory_flags = readonly | os.O_DIRECTORY | nofollow
    root_fd = storage.duplicate_run_dir_fd()
    current_fd = root_fd
    file_fd: int | None = None
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(parts[-1], readonly | nofollow, dir_fd=current_fd)
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("benchmark receipt artifact must be a regular file")
        return relative_path, file_fd
    except Exception:
        if file_fd is not None:
            os.close(file_fd)
        raise
    finally:
        os.close(current_fd)


def read_open_receipt(file_fd: int) -> bytes:
    """Read bounded bytes from an already verified regular-file descriptor."""

    metadata = os.fstat(file_fd)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("benchmark receipt artifact must be a regular file")
    if metadata.st_size > MAX_BENCHMARK_RECEIPT_BYTES:
        raise ValueError(
            f"benchmark receipt artifact exceeds {MAX_BENCHMARK_RECEIPT_BYTES} byte limit"
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        block = os.read(file_fd, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        total += len(block)
        if total > MAX_BENCHMARK_RECEIPT_BYTES:
            raise ValueError(
                f"benchmark receipt artifact exceeds {MAX_BENCHMARK_RECEIPT_BYTES} byte limit"
            )
        chunks.append(block)


def _open_or_create_child_directory(parent_fd: int, name: str) -> int:
    """Open one platform-owned child directory without following a symlink."""

    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=parent_fd)


def _snapshot_verified_receipt(
    storage: RunStorage,
    *,
    index: int,
    name: str,
    payload: bytes,
) -> str:
    """Write source receipt bytes to a platform-owned hash-bound snapshot."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("platform cannot securely snapshot benchmark receipts on this OS")
    identity = hashlib.sha256(f"{index}\0{name}".encode()).hexdigest()[:16]
    filename = f"{index:02d}-{identity}.receipt"
    root_fd = storage.duplicate_run_dir_fd()
    artifacts_fd: int | None = None
    snapshots_fd: int | None = None
    output_fd: int | None = None
    try:
        artifacts_fd = _open_or_create_child_directory(root_fd, "artifacts")
        snapshots_fd = _open_or_create_child_directory(
            artifacts_fd,
            "verified_benchmark_receipts",
        )
        output_fd = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            0o600,
            dir_fd=snapshots_fd,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(output_fd, payload[offset:])
            if written <= 0:
                raise OSError("failed to write verified benchmark receipt snapshot")
            offset += written
        os.fsync(output_fd)
        return f"artifacts/verified_benchmark_receipts/{filename}"
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if snapshots_fd is not None:
            os.close(snapshots_fd)
        if artifacts_fd is not None:
            os.close(artifacts_fd)
        os.close(root_fd)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def validate_benchmark_receipts(
    payload: Any,
    *,
    storage: RunStorage,
    run_id: str,
    trajectory_manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Validate native receipt descriptors before they enter the contract.

    Each receipt must name this run and hash this exact trajectory manifest.
    The advertised artifact hash is recomputed from the retained run directory
    descriptor, then the verified bytes are snapshotted under platform control.
    """

    if payload is None:
        return []
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark final evidence must be a mapping")
    if not payload:
        return []
    if set(payload) != {"receipts"}:
        raise ValueError("benchmark final evidence may contain only receipts")
    raw_receipts = payload.get("receipts")
    if not isinstance(raw_receipts, list) or not raw_receipts:
        raise ValueError("benchmark final evidence receipts must be a non-empty list")
    if len(raw_receipts) > MAX_BENCHMARK_RECEIPTS:
        raise ValueError(f"benchmark final evidence exceeds {MAX_BENCHMARK_RECEIPTS} receipt limit")
    manifest_sha256 = canonical_sha256(dict(trajectory_manifest))
    prepared: list[tuple[str, str, str, str, bytes]] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    total_bytes = 0
    required = {
        "name",
        "schema",
        "path",
        "sha256",
        "run_id",
        "trajectory_manifest_sha256",
    }
    for index, raw in enumerate(raw_receipts):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(f"benchmark receipt {index} has an invalid descriptor schema")
        name = str(raw.get("name") or "")
        schema = str(raw.get("schema") or "")
        expected_sha256 = str(raw.get("sha256") or "")
        if (
            not name
            or not schema
            or "\n" in name
            or "\r" in name
            or "\n" in schema
            or "\r" in schema
        ):
            raise ValueError(f"benchmark receipt {index} has an unsafe name or schema")
        if not _RECEIPT_SHA256_RE.fullmatch(expected_sha256):
            raise ValueError(f"benchmark receipt {index} sha256 must be sha256:<64 lowercase hex>")
        if str(raw.get("run_id") or "") != run_id:
            raise ValueError(f"benchmark receipt {index} is not bound to this run_id")
        if str(raw.get("trajectory_manifest_sha256") or "") != manifest_sha256:
            raise ValueError(f"benchmark receipt {index} is not bound to this trajectory manifest")
        relative_path, artifact_fd = open_run_relative_artifact(
            storage,
            raw.get("path"),
        )
        try:
            receipt_bytes = read_open_receipt(artifact_fd)
        finally:
            os.close(artifact_fd)
        total_bytes += len(receipt_bytes)
        if total_bytes > MAX_TOTAL_BENCHMARK_RECEIPT_BYTES:
            raise ValueError(
                "benchmark final evidence exceeds "
                f"{MAX_TOTAL_BENCHMARK_RECEIPT_BYTES} total byte limit"
            )
        actual_sha256 = _sha256_bytes(receipt_bytes)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"benchmark receipt {index} sha256 does not match its artifact")
        if name in seen_names or relative_path in seen_paths:
            raise ValueError("benchmark final evidence has duplicate receipt names or paths")
        seen_names.add(name)
        seen_paths.add(relative_path)
        prepared.append((name, schema, relative_path, actual_sha256, receipt_bytes))

    receipts: list[dict[str, str]] = []
    for index, (name, schema, relative_path, actual_sha256, receipt_bytes) in enumerate(prepared):
        snapshot_path = _snapshot_verified_receipt(
            storage,
            index=index,
            name=name,
            payload=receipt_bytes,
        )
        receipts.append(
            {
                "name": name,
                "schema": schema,
                "path": snapshot_path,
                "source_path": relative_path,
                "sha256": actual_sha256,
                "run_id": run_id,
                "trajectory_manifest_sha256": manifest_sha256,
            }
        )
    return receipts


def collect_benchmark_receipts(
    finalizer: Any,
    *,
    storage: RunStorage,
    run_id: str,
    trajectory_manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Call an untrusted finalizer without surrendering the trusted manifest.

    A benchmark receives a disposable deep copy.  Validation always uses the
    independent copy captured before that call, so in-place mutation cannot
    redefine which trajectory bytes the receipt is supposed to bind.
    """

    if not callable(finalizer):
        raise TypeError("benchmark receipt finalizer must be callable")
    trusted_manifest = copy.deepcopy(dict(trajectory_manifest))
    raw = finalizer(
        run_id=run_id,
        trajectory_manifest=copy.deepcopy(trusted_manifest),
    )
    return validate_benchmark_receipts(
        raw,
        storage=storage,
        run_id=run_id,
        trajectory_manifest=trusted_manifest,
    )


__all__ = [
    "MAX_BENCHMARK_RECEIPT_BYTES",
    "MAX_BENCHMARK_RECEIPTS",
    "MAX_TOTAL_BENCHMARK_RECEIPT_BYTES",
    "collect_benchmark_receipts",
    "open_run_relative_artifact",
    "read_open_receipt",
    "validate_benchmark_receipts",
]
