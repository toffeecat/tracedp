"""Pinned, fail-closed Git source contract for the optional veRL backend."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from lh_platform.training.agentic_rl_runtime import canonical_sha256

GIT_SOURCE_CONTRACT_SCHEMA = "lh_git_source_contract/v1"
GIT_SOURCE_RECEIPT_SCHEMA = "lh_git_source_receipt/v1"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class SourceVerificationError(ValueError):
    """A backend source checkout differs from its reviewed contract."""


@dataclass(frozen=True)
class SourceFilePin:
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise SourceVerificationError("source pin path must be safe and relative")
        if _SHA64.fullmatch(self.sha256) is None:
            raise SourceVerificationError("source pin hash must be lowercase sha256")


@dataclass(frozen=True)
class GitSourceContract:
    schema: str
    backend_id: str
    canonical_origin: str
    commit_sha: str
    tree_sha: str
    files: tuple[SourceFilePin, ...]

    def __post_init__(self) -> None:
        if self.schema != GIT_SOURCE_CONTRACT_SCHEMA:
            raise SourceVerificationError("unsupported Git source contract schema")
        if not self.backend_id.strip():
            raise SourceVerificationError("source contract backend ID is required")
        if _normalize_git_origin(self.canonical_origin) != self.canonical_origin:
            raise SourceVerificationError("source contract origin is not canonical HTTPS")
        if _SHA40.fullmatch(self.commit_sha) is None:
            raise SourceVerificationError("source contract commit must be lowercase sha1")
        if _SHA40.fullmatch(self.tree_sha) is None:
            raise SourceVerificationError("source contract tree must be lowercase sha1")
        paths = tuple(item.relative_path for item in self.files)
        if not paths or paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise SourceVerificationError(
                "source contract files must be non-empty, unique, and sorted"
            )

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class VerifiedGitSource:
    schema: str
    backend_id: str
    source_root: str
    canonical_origin: str
    actual_origin: str
    commit_sha: str
    tree_sha: str
    clean_worktree: bool
    files: tuple[SourceFilePin, ...]
    contract_sha256: str


def verify_git_source(
    source_root: str | Path,
    contract: GitSourceContract,
) -> VerifiedGitSource:
    """Bind a runtime checkout without importing any code from it."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise SourceVerificationError("backend source root is not a directory")
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise SourceVerificationError("backend source root must be the Git top level")
    actual_origin = _normalize_git_origin(_git(root, "remote", "get-url", "origin"))
    if actual_origin != contract.canonical_origin:
        raise SourceVerificationError("backend source origin differs from contract")
    commit_sha = _git(root, "rev-parse", "HEAD")
    if commit_sha != contract.commit_sha:
        raise SourceVerificationError("backend source commit differs from contract")
    tree_sha = _git(root, "rev-parse", "HEAD^{tree}")
    if tree_sha != contract.tree_sha:
        raise SourceVerificationError("backend source tree differs from contract")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SourceVerificationError("backend source worktree is not clean")
    for pin in contract.files:
        path = (root / pin.relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SourceVerificationError(
                f"backend source API file is missing: {pin.relative_path}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pin.sha256:
            raise SourceVerificationError(
                f"backend source API file hash differs: {pin.relative_path}"
            )
    return VerifiedGitSource(
        schema=GIT_SOURCE_RECEIPT_SCHEMA,
        backend_id=contract.backend_id,
        source_root=str(root),
        canonical_origin=contract.canonical_origin,
        actual_origin=actual_origin,
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        clean_worktree=True,
        files=contract.files,
        contract_sha256=contract.sha256,
    )


def _git(root: Path, *args: str) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceVerificationError(
            "cannot verify backend Git source: " + " ".join(args)
        ) from error
    return completed.stdout.strip()


def _normalize_git_origin(value: str) -> str:
    raw = str(value).strip()
    if raw.startswith("git@") and ":" in raw:
        authority, path = raw.split(":", 1)
        host = authority.split("@", 1)[1]
    else:
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https", "ssh"} or not parsed.hostname:
            raise SourceVerificationError("unsupported backend Git origin")
        host = parsed.hostname
        path = parsed.path.lstrip("/")
    path = path.rstrip("/")
    if not path:
        raise SourceVerificationError("backend Git origin has no repository path")
    if not path.endswith(".git"):
        path += ".git"
    return f"https://{host.lower()}/{path}"


VERL_PINNED_SOURCE_CONTRACT = GitSourceContract(
    schema=GIT_SOURCE_CONTRACT_SCHEMA,
    backend_id="verl",
    canonical_origin="https://github.com/volcengine/verl.git",
    commit_sha="e003163181731412595257a72ec173071efb125f",
    tree_sha="134a18ecffec493d7c776d50f25e174d53949323",
    files=(
        SourceFilePin(
            relative_path="verl/experimental/agent_loop/agent_loop.py",
            sha256=("b4efb3c31b0954327bcf21c1f170b02ee6729d2434a6a4a75a167afd343d9f1e"),
        ),
        SourceFilePin(
            relative_path="verl/experimental/agent_loop/tool_agent_loop.py",
            sha256=("911864a9b3a41af23a204dfce90ec051c3ced646b3d41d4c50dd383de90abc1f"),
        ),
        SourceFilePin(
            relative_path="verl/trainer/ppo/core_algos.py",
            sha256=("9114f9e16c87e4c9ebf2fa016baf733c9bbc819766b53c8968aaa9e8abcd7916"),
        ),
        SourceFilePin(
            relative_path="verl/trainer/ppo/v1/trainer_base.py",
            sha256=("04ec07b702179a6b3ad154e0cf2d57060dea89d6bd854fc73a60b0249969f4bc"),
        ),
    ),
)


__all__ = [
    "GIT_SOURCE_CONTRACT_SCHEMA",
    "GIT_SOURCE_RECEIPT_SCHEMA",
    "VERL_PINNED_SOURCE_CONTRACT",
    "GitSourceContract",
    "SourceFilePin",
    "SourceVerificationError",
    "VerifiedGitSource",
    "verify_git_source",
]
