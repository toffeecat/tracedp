"""Descriptor-safe primitives for run storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Runner-owned JSON artifacts can include long tool traces, but a corrupt or
# adversarial plugin must not turn manifest construction into unbounded memory
# use.  Native benchmark receipts are deliberately much smaller because they
# are proof envelopes, not raw logs.
MAX_RUN_ARTIFACT_BYTES = 64 * 1024 * 1024
_PORTABLE_FILE_NAME_MAX_BYTES = 255


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def make_task_run_id(prefix: str = "task") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def _validated_run_id(value: str) -> str:
    """Return one safe run-directory basename or reject the request.

    ``run_id`` participates directly in the result-root allocation and later
    in receipt identity.  Treating it as a filesystem path would let a CLI
    value such as ``../escape`` or ``/tmp/run`` bypass the configured result
    root even when all receipt paths themselves are relative.
    """

    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError("run_id must be a non-empty safe basename")
    if "\x00" in value or "/" in value or "\\" in value:
        raise ValueError("run_id must not contain a path separator")
    path = Path(value)
    if path.is_absolute() or path.name != value:
        raise ValueError("run_id must be a safe basename under results_root")
    return value


def _directory_open_flags() -> int:
    """Return fail-closed flags for opening a directory by trusted path."""

    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("platform requires O_DIRECTORY and O_NOFOLLOW for run storage")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _safe_run_file_name(value: str | Path) -> str:
    """Return one direct run-directory file name or reject a path."""

    raw = os.fspath(value)
    path = Path(raw)
    if (
        not raw
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != raw
        or raw in {".", ".."}
        or "\x00" in raw
        or "/" in raw
        or "\\" in raw
    ):
        raise ValueError("run artifact name must be one safe relative file name")
    return raw


def _task_trajectory_file_name(task_id: str) -> str:
    """Return a stable filename without treating an external task ID as a path.

    Established collectors and upstream bridges consume
    ``trajectory_<safe-task-id>.json``.  Preserve that public compatibility
    for a deliberately narrow portable identifier alphabet; only task IDs
    which could be interpreted as paths (or otherwise cannot be represented
    safely across filesystems) are replaced with an opaque digest.
    """

    value = str(task_id)
    legacy_name = f"trajectory_{value}.json"
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
        and len(legacy_name.encode("utf-8")) <= _PORTABLE_FILE_NAME_MAX_BYTES
    ):
        return legacy_name
    encoded = value.encode("utf-8")
    return "trajectory_task_sha256_" + hashlib.sha256(encoded).hexdigest() + ".json"


def _write_all(file_fd: int, payload: bytes) -> None:
    """Write all bytes to an already-open descriptor."""

    offset = 0
    while offset < len(payload):
        written = os.write(file_fd, payload[offset:])
        if written <= 0:
            raise OSError("failed to write run artifact")
        offset += written


def _read_all(file_fd: int, *, max_bytes: int = MAX_RUN_ARTIFACT_BYTES) -> bytes:
    """Read bounded bytes from an already-open regular artifact descriptor."""

    chunks: list[bytes] = []
    total = 0
    while True:
        block = os.read(file_fd, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        total += len(block)
        if total > max_bytes:
            raise ValueError(f"run artifact exceeds {max_bytes} byte limit")
        chunks.append(block)


def _create_unique_run_dir(root_fd: int, run_id: str) -> tuple[str, int]:
    """Atomically create and open a collision-free run directory.

    A separate ``exists()`` check is racy: two concurrent runners can both
    observe a missing name and then share the same ``exist_ok=True``
    directory.  The successful ``mkdir(exist_ok=False)`` itself is the only
    trustworthy allocation event, so every competing runner retries a suffix
    only after the filesystem reports ``FileExistsError``.
    """

    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index}"
        name = f"{run_id}{suffix}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=root_fd)
        except FileExistsError:
            index += 1
            continue
        try:
            return name, os.open(name, _directory_open_flags(), dir_fd=root_fd)
        except Exception:
            # Do not leave a directory that later runners could mistake for a
            # completed experiment if the secure descriptor cannot be opened.
            try:
                os.rmdir(name, dir_fd=root_fd)
            except OSError:
                pass
            raise


def _safe_artifact_relative_parts(value: str | Path) -> tuple[str, ...]:
    """Parse a nested artifact name without normalizing unsafe components."""

    raw = os.fspath(value)
    path = Path(raw)
    parts = path.parts
    if (
        not raw
        or path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or "\x00" in raw
        or "\\" in raw
    ):
        raise ValueError("artifact path must be a safe relative path")
    return tuple(parts)


class RunArtifactSink:
    """FD-backed write/read capability for benchmark-owned run artifacts.

    A bare ``Path`` cannot stay trustworthy once an adapter can mutate a
    results-root ancestor between a check and a write.  This capability never
    exposes its directory descriptor as a path: every operation walks from a
    duplicated, no-follow ``artifacts/`` FD and validates every component.
    It is a cooperative boundary for reviewed adapters, not an OS sandbox for
    arbitrary Python code.
    """

    def __init__(self, artifacts_fd: int) -> None:
        self._artifacts_fd: int | None = artifacts_fd

    def _duplicate_root_fd(self) -> int:
        if self._artifacts_fd is None:
            raise RuntimeError("run artifact sink is closed")
        return os.dup(self._artifacts_fd)

    def _open_parent(self, parts: tuple[str, ...], *, create: bool) -> tuple[int, str]:
        current_fd = self._duplicate_root_fd()
        try:
            for component in parts[:-1]:
                if create:
                    try:
                        os.mkdir(component, 0o700, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=current_fd,
                )
                os.close(current_fd)
                current_fd = next_fd
            return current_fd, parts[-1]
        except Exception:
            os.close(current_fd)
            raise

    def write_bytes(
        self,
        relative_path: str | Path,
        payload: bytes,
        *,
        overwrite: bool = True,
    ) -> None:
        """Atomically write bytes under the immutable artifact directory FD."""

        if not isinstance(payload, bytes):
            raise TypeError("artifact payload must be bytes")
        parts = _safe_artifact_relative_parts(relative_path)
        parent_fd, name = self._open_parent(parts, create=True)
        staging = f".{name}.{uuid.uuid4().hex}.tmp"
        staging_fd: int | None = None
        published = False
        try:
            if not overwrite:
                try:
                    existing_fd = os.open(
                        name,
                        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=parent_fd,
                    )
                except FileNotFoundError:
                    pass
                else:
                    os.close(existing_fd)
                    raise FileExistsError(f"artifact already exists: {relative_path}")
            staging_fd = os.open(
                staging,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            _write_all(staging_fd, payload)
            os.fsync(staging_fd)
            os.close(staging_fd)
            staging_fd = None
            os.replace(staging, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            published = True
        finally:
            if staging_fd is not None:
                os.close(staging_fd)
            if not published:
                try:
                    os.unlink(staging, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            os.close(parent_fd)

    def write_json(
        self,
        relative_path: str | Path,
        payload: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> None:
        """Write deterministic JSON under this sink without a lexical path."""

        encoded = (
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        self.write_bytes(relative_path, encoded, overwrite=overwrite)

    def read_bytes(
        self,
        relative_path: str | Path,
        *,
        max_bytes: int = MAX_RUN_ARTIFACT_BYTES,
    ) -> bytes:
        """Read one bounded regular artifact without following links/FIFOs."""

        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        parts = _safe_artifact_relative_parts(relative_path)
        parent_fd, name = self._open_parent(parts, create=False)
        file_fd: int | None = None
        try:
            file_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("artifact must be a regular file")
            if metadata.st_size > max_bytes:
                raise ValueError(f"artifact exceeds {max_bytes} byte limit")
            return _read_all(file_fd, max_bytes=max_bytes)
        finally:
            if file_fd is not None:
                os.close(file_fd)
            os.close(parent_fd)

    def run_relative_path(self, relative_path: str | Path) -> str:
        """Return the contract-safe run-relative spelling of a sink artifact."""

        return "artifacts/" + "/".join(_safe_artifact_relative_parts(relative_path))

    def close(self) -> None:
        if self._artifacts_fd is not None:
            try:
                os.close(self._artifacts_fd)
            except OSError:
                pass
            self._artifacts_fd = None
