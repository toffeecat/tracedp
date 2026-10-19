"""Local-benchmark adapter helper."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def lh_root() -> Path:
    """Repository root: ``LH_ROOT`` when set, else computed from this file."""
    override = os.environ.get("LH_ROOT", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def eval_benchmark_path(slug: str) -> Path:
    return lh_root() / "eval-benchmark" / slug


def has_eval_benchmark(slug: str) -> bool:
    return (eval_benchmark_path(slug) / "repo").is_dir()


def read_jsonl(
    path: Path,
    limit: int | None = None,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Read up to ``limit`` parsed objects, optionally rejecting bad rows."""

    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                if strict:
                    raise ValueError(
                        f"invalid JSONL row {line_number} in {path}: {error.msg}"
                    ) from error
                continue
            if not isinstance(item, dict):
                if strict:
                    raise ValueError(f"JSONL row {line_number} in {path} is not an object")
                continue
            out.append(item)
    return out
