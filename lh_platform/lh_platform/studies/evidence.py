"""Resolve declared paper and source references into locally verified evidence."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def require_verified_source_root(compiled: Any) -> Path:
    """Resolve the first commit-verified source root or fail closed."""
    for evidence in getattr(compiled, "resolved_evidence", ()):
        if evidence.source_root and evidence.source_commit_verified:
            return Path(evidence.source_root)
    raise ValueError("compiled study has no verified upstream source root")


@dataclass(frozen=True)
class ResolvedEvidence:
    id: str
    paper_path: str
    paper_sha256: str
    quote_verified: bool
    source_root: str | None
    source_remote: str | None
    source_commit: str | None
    source_commit_verified: bool
    source_paths: tuple[str, ...]
    source_paths_verified: bool
    source_paths_clean: bool
    source_paths_at_commit: bool
    resolution_status: str


class EvidenceResolutionError(ValueError):
    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(message)


class EvidenceRegistry:
    """Verify immutable local paper and source artifacts without network access."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    def resolve(self, evidence: Any, index: int) -> ResolvedEvidence:
        prefix = f"evidence[{index}]"
        paper_path = self._path(evidence.paper_path)
        if not paper_path.is_file():
            raise EvidenceResolutionError(
                "paper_not_found",
                f"{prefix}.paper_path",
                f"paper file not found: {paper_path}",
            )
        actual_paper_hash = _sha256(str(paper_path))
        if actual_paper_hash != evidence.paper_sha256:
            raise EvidenceResolutionError(
                "paper_hash_mismatch",
                f"{prefix}.paper_sha256",
                f"expected {evidence.paper_sha256}, resolved {actual_paper_hash}",
            )
        if _normalize(evidence.quote) not in _pdf_text(str(paper_path)):
            raise EvidenceResolutionError(
                "quote_not_found",
                f"{prefix}.quote",
                "the declared quote was not found in the immutable paper PDF",
            )

        if evidence.artifact_status == "paper_only":
            return ResolvedEvidence(
                id=evidence.id,
                paper_path=str(paper_path),
                paper_sha256=actual_paper_hash,
                quote_verified=True,
                source_root=None,
                source_remote=None,
                source_commit=None,
                source_commit_verified=False,
                source_paths=(),
                source_paths_verified=False,
                source_paths_clean=False,
                source_paths_at_commit=False,
                resolution_status="paper_verified",
            )

        source_root = self._path(evidence.source_root)
        if not source_root.is_dir():
            raise EvidenceResolutionError(
                "source_root_not_found",
                f"{prefix}.source_root",
                f"source root not found: {source_root}",
            )
        source_remote = _git_remote(str(source_root))
        if _repository_identity(source_remote) != _repository_identity(evidence.source_url):
            raise EvidenceResolutionError(
                "source_url_mismatch",
                f"{prefix}.source_url",
                f"declared URL does not match git remote {source_remote}",
            )
        actual_commit = _git_head(str(source_root))
        if actual_commit != evidence.commit:
            raise EvidenceResolutionError(
                "source_commit_mismatch",
                f"{prefix}.commit",
                f"expected {evidence.commit}, resolved {actual_commit}",
            )
        resolved_paths = tuple(str(source_root / path) for path in evidence.source_paths)
        missing_paths = [path for path in resolved_paths if not Path(path).is_file()]
        if missing_paths:
            raise EvidenceResolutionError(
                "source_path_not_found",
                f"{prefix}.source_paths",
                f"source paths not found: {', '.join(missing_paths)}",
            )
        dirty_paths = _git_dirty_paths(str(source_root), evidence.source_paths)
        if dirty_paths:
            raise EvidenceResolutionError(
                "source_paths_dirty",
                f"{prefix}.source_paths",
                f"source evidence differs from commit {actual_commit}: {dirty_paths}",
            )
        missing_from_commit = _git_paths_missing_from_commit(
            str(source_root), actual_commit, evidence.source_paths
        )
        if missing_from_commit:
            raise EvidenceResolutionError(
                "source_path_not_in_commit",
                f"{prefix}.source_paths",
                f"source paths absent from commit {actual_commit}: {missing_from_commit}",
            )
        return ResolvedEvidence(
            id=evidence.id,
            paper_path=str(paper_path),
            paper_sha256=actual_paper_hash,
            quote_verified=True,
            source_root=str(source_root),
            source_remote=source_remote,
            source_commit=actual_commit,
            source_commit_verified=True,
            source_paths=resolved_paths,
            source_paths_verified=True,
            source_paths_clean=True,
            source_paths_at_commit=True,
            resolution_status="source_verified",
        )

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.workspace_root / path).resolve()


def find_workspace_root(source_path: Path | None) -> Path:
    starts = [source_path.parent] if source_path else []
    starts.append(Path.cwd())
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "lh_platform" / "pyproject.toml").is_file() and (
                candidate / "analysis"
            ).is_dir():
                return candidate
    return starts[0]


@lru_cache(maxsize=32)
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise EvidenceResolutionError(
            "pdf_reader_unavailable",
            "paper_path",
            "install the lh_platform package dependencies before verifying paper PDF quotes",
        ) from exc
    reader = PdfReader(path)
    return _normalize("\n".join(page.extract_text() or "" for page in reader.pages))


@lru_cache(maxsize=32)
def _git_head(source_root: str) -> str:
    completed = subprocess.run(
        ["git", "-C", source_root, "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise EvidenceResolutionError(
            "source_commit_unresolvable",
            "source_root",
            completed.stderr.strip() or "cannot resolve source commit",
        )
    return completed.stdout.strip()


@lru_cache(maxsize=32)
def _git_remote(source_root: str) -> str:
    completed = subprocess.run(
        ["git", "-C", source_root, "remote", "get-url", "origin"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise EvidenceResolutionError(
            "source_remote_unresolvable",
            "source_root",
            completed.stderr.strip() or "cannot resolve source remote",
        )
    return completed.stdout.strip()


def _git_dirty_paths(source_root: str, source_paths: tuple[str, ...]) -> str:
    completed = subprocess.run(
        ["git", "-C", source_root, "status", "--porcelain", "--", *source_paths],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise EvidenceResolutionError(
            "source_status_unresolvable",
            "source_paths",
            completed.stderr.strip() or "cannot inspect source worktree",
        )
    return completed.stdout.strip()


def _git_paths_missing_from_commit(
    source_root: str,
    commit: str,
    source_paths: tuple[str, ...],
) -> str:
    missing: list[str] = []
    for source_path in source_paths:
        completed = subprocess.run(
            ["git", "-C", source_root, "cat-file", "-e", f"{commit}:{source_path}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            missing.append(source_path)
    return ", ".join(missing)


def _repository_identity(url: str) -> str:
    match = re.search(r"github\.com[:/]([^/]+)/([^/#]+)", url.casefold())
    if not match:
        return url.casefold().rstrip("/")
    owner, repository = match.groups()
    return f"github.com/{owner}/{repository.removesuffix('.git')}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()
