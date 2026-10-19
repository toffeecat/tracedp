"""Shared provenance and isolation rules for supplied tracer source trees."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_FULL_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SourceCheckoutAttestationV1:
    """Describe a content-clean checkout without creating a content fingerprint."""

    version_label: str
    content_status: str
    file_mode_normalized: bool


def _meaningful_status_lines(status: str) -> tuple[str, ...]:
    """Drop only macOS metadata sidecars that cannot be imported as source."""

    meaningful: list[str] = []
    for line in status.splitlines():
        path = line[3:] if len(line) >= 4 else ""
        if line.startswith("?? ") and Path(path).name.startswith("._"):
            continue
        meaningful.append(line)
    return tuple(meaningful)


def require_content_clean_source_checkout(
    source: Path, *, method_name: str
) -> SourceCheckoutAttestationV1:
    """Accept FAT mode drift while rejecting content edits and untracked files."""

    branch = (
        subprocess.run(
            ["git", "-C", str(source), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        or "detached"
    )
    commit_date = subprocess.run(
        ["git", "-C", str(source), "show", "-s", "--format=%cI", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    raw_status = subprocess.run(
        [
            "git",
            "-c",
            "core.fileMode=false",
            "-C",
            str(source),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    dirty = _meaningful_status_lines(raw_status)
    if dirty:
        summary = "\n".join(dirty)
        raise ValueError(f"{method_name} source checkout has content changes: {summary[:500]}")
    return SourceCheckoutAttestationV1(
        version_label=f"{branch}@{commit_date}",
        content_status="clean",
        file_mode_normalized=True,
    )


def require_source_checkout_identity(
    source: Path,
    *,
    method_name: str,
    accepted_origin_urls: tuple[str, ...],
    required_paths: tuple[str, ...],
) -> str:
    """Verify that a clean directory is the intended upstream source project."""

    if not accepted_origin_urls or not required_paths:
        raise ValueError(f"{method_name} source identity contract is incomplete")
    origin = subprocess.run(
        ["git", "-C", str(source), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if origin not in accepted_origin_urls:
        raise ValueError(f"{method_name} source origin is not the admitted repository")
    missing = tuple(path for path in required_paths if not (source / path).is_file())
    if missing:
        raise ValueError(f"{method_name} source entrypoints are missing: {', '.join(missing)}")
    return origin


def require_clean_source_revision(source: Path, *, method_name: str) -> str:
    """Return the exact Git revision, rejecting code not represented by it."""

    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if not _FULL_GIT_REVISION.fullmatch(revision):
        raise ValueError(f"{method_name} source revision is not a full Git SHA")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"{method_name} source checkout is dirty: {dirty[:500]}")
    return revision


def isolated_worker_environment(secret_name: str, secret_value: str) -> dict[str, str]:
    """Keep worker imports explicit when its current directory is temporary."""

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment[secret_name] = secret_value
    project_root = str(Path(__file__).resolve().parents[3])
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        project_root if not inherited else project_root + os.pathsep + inherited
    )
    return environment


def strict_worker_code_loading_environment() -> dict[str, str]:
    """Return the complete non-secret environment controlling worker imports."""

    project_root = str(Path(__file__).resolve().parents[3])
    return {
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": project_root,
    }


def strict_worker_environment(secret_name: str, secret_value: str) -> dict[str, str]:
    """Build a minimal worker environment plus its one runtime secret."""

    environment = strict_worker_code_loading_environment()
    environment[secret_name] = secret_value
    return environment


__all__ = [
    "SourceCheckoutAttestationV1",
    "isolated_worker_environment",
    "require_clean_source_revision",
    "require_content_clean_source_checkout",
    "require_source_checkout_identity",
    "strict_worker_code_loading_environment",
    "strict_worker_environment",
]
