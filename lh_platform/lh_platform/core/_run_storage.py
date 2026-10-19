"""Run-level artifact persistence built on descriptor-safe primitives."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lh_platform.core._storage_fd import (
    MAX_RUN_ARTIFACT_BYTES,
    RunArtifactSink,
    _create_unique_run_dir,
    _directory_open_flags,
    _read_all,
    _safe_run_file_name,
    _task_trajectory_file_name,
    _validated_run_id,
    _write_all,
    make_run_id,
    make_task_run_id,
)
from lh_platform.core.artifact_redaction import redact_artifact, redact_text
from lh_platform.core.evidence_contract import RUNNER_TRAJECTORY_MANIFEST_SCHEMA
from lh_platform.core.types import Trajectory

logger = logging.getLogger(__name__)


class RunStorage:
    """Single-run storage backend.

    Layout:
        results/<run_id>/
            run.json           - config snapshot
            trajectory.jsonl   - per-step records
            metrics.json       - evaluator outputs (one entry per task)
            artifacts/         - benchmark-native artifacts
            logs/agent.log     - text log
    """

    def __init__(self, root: Path, run_id: str | None = None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # The directory name is the only collision-free identity available to
        # downstream artifacts.  Retaining a requested id after
        # ``_unique_run_dir`` appends ``_2`` would let two receipts truthfully
        # claim the same run_id while living in different directories.
        self.requested_run_id = _validated_run_id(make_run_id() if run_id is None else run_id)
        root_fd = os.open(self.root, _directory_open_flags())
        self._run_dir_fd: int | None = None
        self._artifacts_dir_fd: int | None = None
        self._artifact_sinks: list[RunArtifactSink] = []
        self._trajectory_fh: Any | None = None
        allocated_name: str | None = None
        created_children: list[str] = []
        trajectory_fd: int | None = None
        try:
            self.run_id, self._run_dir_fd = _create_unique_run_dir(
                root_fd,
                self.requested_run_id,
            )
            allocated_name = self.run_id
            self.run_dir = self.root / self.run_id
            for child in ("artifacts", "logs"):
                os.mkdir(child, mode=0o700, dir_fd=self._run_dir_fd)
                created_children.append(child)
            self._artifacts_dir_fd = os.open(
                "artifacts",
                _directory_open_flags(),
                dir_fd=self._run_dir_fd,
            )
            trajectory_fd = os.open(
                "trajectory.jsonl",
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=self._run_dir_fd,
            )
            self._trajectory_fh = os.fdopen(trajectory_fd, "a", encoding="utf-8")
            trajectory_fd = None
        except Exception as error:
            cleanup_errors: list[str] = []
            if trajectory_fd is not None:
                try:
                    os.close(trajectory_fd)
                except OSError as cleanup_error:
                    cleanup_errors.append(f"trajectory fd close: {cleanup_error}")
            if self._trajectory_fh is not None:
                try:
                    self._trajectory_fh.close()
                except OSError as cleanup_error:
                    cleanup_errors.append(f"trajectory close: {cleanup_error}")
                self._trajectory_fh = None
            if self._artifacts_dir_fd is not None:
                try:
                    os.close(self._artifacts_dir_fd)
                except OSError as cleanup_error:
                    cleanup_errors.append(f"artifacts fd close: {cleanup_error}")
                self._artifacts_dir_fd = None
            if self._run_dir_fd is not None:
                try:
                    try:
                        os.unlink("trajectory.jsonl", dir_fd=self._run_dir_fd)
                    except FileNotFoundError:
                        pass
                    for child in reversed(created_children):
                        os.rmdir(child, dir_fd=self._run_dir_fd)
                except OSError as cleanup_error:
                    cleanup_errors.append(f"run contents rollback: {cleanup_error}")
                try:
                    os.close(self._run_dir_fd)
                except OSError as cleanup_error:
                    cleanup_errors.append(f"run fd close: {cleanup_error}")
                self._run_dir_fd = None
            if allocated_name is not None:
                try:
                    os.rmdir(allocated_name, dir_fd=root_fd)
                except OSError as cleanup_error:
                    cleanup_errors.append(f"run directory rollback: {cleanup_error}")
            if cleanup_errors:
                raise RuntimeError(
                    "run storage initialization failed and rollback was incomplete: "
                    + "; ".join(cleanup_errors)
                ) from error
            raise
        finally:
            os.close(root_fd)

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @staticmethod
    def _same_directory_identity(path: Path, directory_fd: int, *, label: str) -> None:
        """Ensure a public compatibility path still names the held directory.

        Upstream benchmark APIs historically accept ``Path`` rather than an
        FD-backed artifact writer.  Such a path is safe to expose only while
        its inode is still the directory this run allocated.  If a plugin has
        replaced a results-root ancestor or the artifacts child, fail closed
        instead of handing it a path into an attacker-controlled tree.
        """

        try:
            path_stat = os.stat(path, follow_symlinks=True)
        except OSError as error:
            raise RuntimeError(f"{label} path is no longer reachable from its run FD") from error
        fd_stat = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or path_stat.st_dev != fd_stat.st_dev
            or path_stat.st_ino != fd_stat.st_ino
        ):
            raise RuntimeError(f"{label} path no longer matches its allocated run directory")

    def external_artifacts_dir(self) -> Path:
        """Return the legacy Path sink only while it still resolves safely.

        New native adapters should eventually use a capability interface.  At
        this compatibility boundary, a stale/replaced path is an admission
        failure rather than an opportunity to write unverified artifacts.
        """

        if self._run_dir_fd is None or self._artifacts_dir_fd is None:
            raise RuntimeError("run storage is closed")
        self._same_directory_identity(self.run_dir, self._run_dir_fd, label="run")
        self._same_directory_identity(
            self.artifacts_dir,
            self._artifacts_dir_fd,
            label="artifacts",
        )
        return self.artifacts_dir

    def create_artifact_sink(self) -> RunArtifactSink:
        """Create a registered FD-backed capability for a benchmark adapter."""

        if self._artifacts_dir_fd is None:
            raise RuntimeError("run storage is closed")
        sink = RunArtifactSink(os.dup(self._artifacts_dir_fd))
        self._artifact_sinks.append(sink)
        return sink

    def duplicate_run_dir_fd(self) -> int:
        """Return a caller-owned FD for this run's original directory.

        A benchmark can rename or replace the configured results-root path
        after it receives an artifact path.  Receipt verification must stay
        anchored to the directory allocated by this storage instance, not a
        later path lookup through potentially replaced ancestors.
        """

        if self._run_dir_fd is None:
            raise RuntimeError("run storage is closed")
        return os.dup(self._run_dir_fd)

    def _atomic_write_bytes(self, name: str, payload: bytes) -> None:
        """Atomically publish Runner-owned bytes under the retained run FD."""

        target = _safe_run_file_name(name)
        staging = f".{target}.{uuid.uuid4().hex}.tmp"
        root_fd = self.duplicate_run_dir_fd()
        staging_fd: int | None = None
        published = False
        try:
            staging_fd = os.open(
                staging,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_fd,
            )
            _write_all(staging_fd, payload)
            os.fsync(staging_fd)
            os.close(staging_fd)
            staging_fd = None
            # ``replace`` moves the staged inode itself; it never follows an
            # existing target symlink. Both descriptors name the original run
            # directory allocated by this storage instance.
            os.replace(staging, target, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            try:
                os.fsync(root_fd)
            except OSError:
                # Some filesystems do not support directory fsync. The file
                # content is already fsynced; do not weaken path safety.
                pass
            published = True
        finally:
            if staging_fd is not None:
                os.close(staging_fd)
            if not published:
                try:
                    os.unlink(staging, dir_fd=root_fd)
                except FileNotFoundError:
                    pass
            os.close(root_fd)

    def _read_run_file_bytes(self, name: str) -> bytes:
        """Read one original run artifact without following any symlink."""

        target = _safe_run_file_name(name)
        root_fd = self.duplicate_run_dir_fd()
        file_fd: int | None = None
        try:
            file_fd = os.open(
                target,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"run artifact is not a regular file: {target}")
            if metadata.st_size > MAX_RUN_ARTIFACT_BYTES:
                raise ValueError(
                    f"run artifact exceeds {MAX_RUN_ARTIFACT_BYTES} byte limit: {target}"
                )
            return _read_all(file_fd)
        finally:
            if file_fd is not None:
                os.close(file_fd)
            os.close(root_fd)

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        encoded = (
            json.dumps(
                redact_artifact(payload),
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        self._atomic_write_bytes(name, encoded)

    def write_run(self, config: dict[str, Any]) -> None:
        self._write_json("run.json", config)

    def append_step(self, step: dict[str, Any]) -> None:
        self._trajectory_fh.write(json.dumps(redact_artifact(step), ensure_ascii=False) + "\n")
        self._trajectory_fh.flush()

    def append_step_task(self, step: dict[str, Any], task_id: str) -> None:
        out = dict(step)
        out.setdefault("info", {})
        if isinstance(out["info"], dict):
            out["info"]["task_id"] = task_id
        self._trajectory_fh.write(json.dumps(redact_artifact(out), ensure_ascii=False) + "\n")
        self._trajectory_fh.flush()

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        self._write_json("metrics.json", metrics)

    def write_evidence_contract(self, contract: dict[str, Any]) -> Path:
        """Atomically publish the run's provenance/evidence contract."""

        target = "evidence_contract.json"
        self._write_json(target, contract)
        return Path(target)

    def write_trajectory(self, traj: Trajectory) -> None:
        """Write full trajectory at the end (in addition to per-step appends)."""
        self._write_json("trajectory_full.json", traj.to_dict())

    def write_task_trajectory(self, traj: Trajectory) -> Path:
        task_id = traj.task_id or make_task_run_id()
        target = _task_trajectory_file_name(task_id)
        self._write_json(target, traj.to_dict())
        # The relative name is deliberate: callers must resolve it through
        # this storage object's held FD, never through a mutable root path.
        return Path(target)

    def build_task_trajectory_manifest(self, paths: Iterable[Path]) -> dict[str, Any]:
        """Return the exact persisted task trajectories for a completed run.

        The manifest is intentionally built from persisted files, not live
        ``Trajectory`` objects.  That makes its SHA-256 values the same bytes
        downstream collection and training admission will re-check.
        """

        entries: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        seen_task_ids: set[str] = set()
        for raw_path in paths:
            relative_path = _safe_run_file_name(raw_path)
            if relative_path == "trajectory_full.json":
                raise ValueError(f"invalid task trajectory artifact path: {relative_path}")
            try:
                payload = self._read_run_file_bytes(relative_path)
                raw = json.loads(payload.decode("utf-8"))
            except (OSError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid task trajectory artifact {relative_path}: {error}"
                ) from error
            task_id = str(raw.get("task_id") or "") if isinstance(raw, dict) else ""
            if not task_id:
                raise ValueError(f"task trajectory artifact lacks task_id: {relative_path}")
            if relative_path in seen_paths or task_id in seen_task_ids:
                raise ValueError(f"duplicate task trajectory artifact: {relative_path}")
            seen_paths.add(relative_path)
            seen_task_ids.add(task_id)
            digest = hashlib.sha256(payload).hexdigest()
            entries.append(
                {
                    "path": relative_path,
                    "task_id": task_id,
                    "sha256": digest,
                }
            )
        return {
            "schema": RUNNER_TRAJECTORY_MANIFEST_SCHEMA,
            "entries": sorted(entries, key=lambda item: item["path"]),
        }

    def log(self, message: str) -> None:
        root_fd = self.duplicate_run_dir_fd()
        logs_fd: int | None = None
        log_fd: int | None = None
        try:
            logs_fd = os.open("logs", _directory_open_flags(), dir_fd=root_fd)
            log_fd = os.open(
                "agent.log",
                os.O_WRONLY
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=logs_fd,
            )
            line = f"[{time.strftime('%H:%M:%S')}] {redact_text(message)}\n"
            _write_all(log_fd, line.encode("utf-8"))
        finally:
            if log_fd is not None:
                os.close(log_fd)
            if logs_fd is not None:
                os.close(logs_fd)
            os.close(root_fd)

    def close(self) -> None:
        try:
            if self._trajectory_fh is not None:
                self._trajectory_fh.close()
        except Exception:
            logger.exception("run_storage: trajectory file close failed")
        self._trajectory_fh = None
        for sink in self._artifact_sinks:
            sink.close()
        self._artifact_sinks.clear()
        if self._artifacts_dir_fd is not None:
            try:
                os.close(self._artifacts_dir_fd)
            except OSError:
                pass
            self._artifacts_dir_fd = None
        if self._run_dir_fd is not None:
            try:
                os.close(self._run_dir_fd)
            except OSError:
                pass
            self._run_dir_fd = None

    def __enter__(self) -> RunStorage:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
