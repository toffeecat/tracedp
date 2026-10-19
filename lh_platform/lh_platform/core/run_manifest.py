"""One run manifest per run directory, plus an append-only index under the run root.

Every entry point that creates a run directory calls :func:`write_run_manifest` once
before doing work.  The manifest binds the run to the exact code revision (with the
dirty flag), the config file it was launched from, and the interpreter.  The registry
(`docs/experiments/traceback/registry.yaml`) refers to a run by this manifest's path;
a scientific result without one is rejected by the registry renderer.

Re-entering an existing run directory (resume) keeps the original manifest and only
records the resume in the index, so the manifest always describes the first launch.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lh_platform.core.code_provenance import git_provenance

RUN_MANIFEST_SCHEMA = "longh_run_manifest_v1"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_INDEX_NAME = "INDEX.jsonl"
# Environment variables that locate storage; anything that looks like a secret is skipped.
_ENV_PREFIX = "LONGH_"
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _storage_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(_ENV_PREFIX)
        and not any(marker in key.upper() for marker in _SECRET_MARKERS)
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    staging = path.with_suffix(path.suffix + ".writing")
    with staging.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging.replace(path)


def _append_index(run_root: Path, row: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / RUN_INDEX_NAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_run_manifest(
    run_dir: Path,
    *,
    entry: str,
    config_path: Path | None,
    code_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe one launch without reading or copying private run content."""

    run_dir = run_dir.resolve()
    code = git_provenance(code_root or Path(__file__))
    resolved_config = config_path.resolve() if config_path is not None else None
    config_relpath = None
    if resolved_config is not None and code["repository_root"]:
        root = Path(code["repository_root"])
        if resolved_config.is_relative_to(root):
            config_relpath = str(resolved_config.relative_to(root))
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "entry": entry,
        "created_at": _now(),
        "config_path": str(resolved_config) if resolved_config is not None else None,
        "config_repo_relpath": config_relpath,
        "code": {
            "git_sha": code["git_sha"],
            "working_tree_dirty": code["working_tree_dirty"],
            "tracked_change_count": code["tracked_change_count"],
            "untracked_change_count": code["untracked_change_count"],
        },
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "storage_environment": _storage_environment(),
        "extra": dict(extra or {}),
    }


def write_run_manifest(
    run_dir: Path,
    *,
    entry: str,
    config_path: Path | None,
    run_root: Path | None = None,
    code_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the manifest once and record the launch (or resume) in the run index.

    ``run_root`` defaults to the parent of ``run_dir``.  Returns the manifest that is
    on disk after the call.
    """

    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    root = (run_root or run_dir.parent).resolve()
    path = run_dir / RUN_MANIFEST_NAME
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
            raise ValueError(f"unexpected run manifest schema in {path}")
        _append_index(
            root,
            {"event": "resumed", "run_id": manifest["run_id"], "at": _now(), "entry": entry},
        )
        return manifest
    manifest = build_run_manifest(
        run_dir, entry=entry, config_path=config_path, code_root=code_root, extra=extra
    )
    _atomic_write_json(path, manifest)
    _append_index(
        root,
        {
            "event": "created",
            "run_id": manifest["run_id"],
            "at": manifest["created_at"],
            "entry": entry,
            "manifest_path": str(path),
            "git_sha": manifest["code"]["git_sha"],
            "working_tree_dirty": manifest["code"]["working_tree_dirty"],
        },
    )
    return manifest


__all__ = [
    "RUN_INDEX_NAME",
    "RUN_MANIFEST_NAME",
    "RUN_MANIFEST_SCHEMA",
    "build_run_manifest",
    "write_run_manifest",
]
