"""Fail-closed primary-source contracts for confirmatory studies."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SOURCE_CONTRACT_SCHEMA = "lh_source_contract/v1"


@dataclass(frozen=True)
class SourceEvidenceResult:
    id: str
    path: str
    sha256: str | None
    quote_verified: bool
    hash_verified: bool
    supports: tuple[str, ...]


@dataclass(frozen=True)
class SourceContractReport:
    schema: str
    verified: bool
    coverage: float
    upstream_commit: str | None
    actual_commit: str | None
    commit_verified: bool
    required_fields: tuple[str, ...]
    covered_fields: tuple[str, ...]
    blocked_by: tuple[str, ...]
    evidence: tuple[SourceEvidenceResult, ...]

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target


def verify_source_contract(
    contract_path: str | Path, *, source_root: str | Path
) -> SourceContractReport:
    """Verify exact quotes, required hashes, commit pin, and field coverage."""

    contract = _read_mapping(Path(contract_path))
    root = Path(source_root).resolve()
    blockers: list[str] = []
    if contract.get("schema") != SOURCE_CONTRACT_SCHEMA:
        blockers.append("unsupported_source_contract_schema")
    required = tuple(str(item) for item in contract.get("required_fields") or ())
    if not required:
        blockers.append("source_required_fields_empty")
    upstream_commit = str(contract.get("upstream_commit", "")).strip() or None
    actual_commit = _git_head(root) if upstream_commit else None
    commit_verified = upstream_commit is None or actual_commit == upstream_commit
    if upstream_commit and not commit_verified:
        blockers.append("source_commit_mismatch")
    covered: set[str] = set()
    results: list[SourceEvidenceResult] = []
    for index, item in enumerate(contract.get("evidence") or ()):
        if not isinstance(item, Mapping):
            blockers.append(f"source_evidence_invalid:{index}")
            continue
        evidence_id = str(item.get("id") or index)
        relative = Path(str(item.get("path", "")))
        path = (root / relative).resolve()
        path_ok = path.is_relative_to(root) and path.is_file()
        text = path.read_text(encoding="utf-8", errors="replace") if path_ok else ""
        quotes = tuple(str(quote) for quote in item.get("quotes") or ())
        quote_ok = bool(quotes) and all(quote in text for quote in quotes)
        actual_sha = _sha256(path) if path_ok else None
        declared_sha = str(item.get("sha256", "")).strip()
        hash_ok = bool(actual_sha) and bool(declared_sha) and actual_sha == declared_sha
        supports = tuple(str(field) for field in item.get("supports") or ())
        if quote_ok and hash_ok:
            covered.update(supports)
        if not path_ok:
            blockers.append(f"source_path_missing:{evidence_id}")
        if not quote_ok:
            blockers.append(f"source_quote_missing:{evidence_id}")
        if not hash_ok:
            blockers.append(f"source_hash_mismatch:{evidence_id}")
        results.append(
            SourceEvidenceResult(
                id=evidence_id,
                path=str(path),
                sha256=actual_sha,
                quote_verified=quote_ok,
                hash_verified=hash_ok,
                supports=supports,
            )
        )
    missing = sorted(set(required) - covered)
    if missing:
        blockers.append("source_fields_uncovered")
    blockers = list(dict.fromkeys(blockers))
    return SourceContractReport(
        schema="lh_source_contract_report/v1",
        verified=not blockers,
        coverage=(len(set(required) & covered) / len(set(required))) if required else 0.0,
        upstream_commit=upstream_commit,
        actual_commit=actual_commit,
        commit_verified=commit_verified,
        required_fields=required,
        covered_fields=tuple(sorted(covered)),
        blocked_by=tuple(blockers),
        evidence=tuple(results),
    )


def _read_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("source contract must be a mapping")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
