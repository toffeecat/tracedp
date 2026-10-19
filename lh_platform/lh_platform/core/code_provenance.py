"""Collect auditable Git revision and worktree provenance for a run."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any


def git_provenance(start: Path | None = None) -> dict[str, Any]:
    """Collect revision facts without making a dirty worktree a hard failure."""

    root = _find_git_root(start or Path.cwd())
    if root is None:
        return {
            "repository_root": None,
            "git_sha": None,
            "working_tree_dirty": None,
            "tracked_change_count": None,
            "untracked_change_count": None,
            "status_sha256": None,
        }
    sha = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain=v1")
    if status is None:
        return {
            "repository_root": str(root),
            "git_sha": sha,
            "working_tree_dirty": None,
            "tracked_change_count": None,
            "untracked_change_count": None,
            "status_sha256": None,
        }
    lines = [line for line in status.splitlines() if line]
    return {
        "repository_root": str(root),
        "git_sha": sha or None,
        "working_tree_dirty": bool(lines),
        "tracked_change_count": sum(1 for line in lines if not line.startswith("??")),
        "untracked_change_count": sum(1 for line in lines if line.startswith("??")),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _find_git_root(start: Path) -> Path | None:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_output(root: Path, *args: str) -> str | None:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


__all__ = ["git_provenance"]
