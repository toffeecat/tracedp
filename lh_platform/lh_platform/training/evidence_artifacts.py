"""Fail-closed filesystem primitives for formal evidence runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json


class EvidenceArtifactError(ValueError):
    """Raised when evidence output could collide or be overwritten."""


def reserve_fresh_evidence_directory(
    output_dir: str | Path,
    *,
    protected_paths: tuple[Path, ...],
) -> Path:
    """Atomically claim a new output directory outside every input tree."""

    target = Path(output_dir).resolve()
    for raw_protected in protected_paths:
        protected = Path(raw_protected).resolve()
        if target == protected or protected in target.parents or target in protected.parents:
            raise EvidenceArtifactError("evidence output collides with a protected input tree")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise EvidenceArtifactError("evidence output directory already exists") from error
    return target


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create one JSON artifact exactly once."""

    target = Path(path).resolve()
    try:
        atomic_write_json(target, payload)
    except FileExistsError as error:
        raise EvidenceArtifactError(f"evidence artifact already exists: {target}") from error
    return target


__all__ = [
    "EvidenceArtifactError",
    "reserve_fresh_evidence_directory",
    "write_json_exclusive",
]
