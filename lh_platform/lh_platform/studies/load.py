"""Load purpose-driven study protocols from YAML or JSON."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from lh_platform.studies.protocol import StudyProtocol


class StudyConfigError(ValueError):
    """Raised when a study file cannot be parsed into a protocol."""


def load_study(path: str | Path) -> StudyProtocol:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise StudyConfigError(f"study file not found: {source_path}")
    suffix = source_path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise StudyConfigError(f"unsupported study suffix: {suffix}")
    try:
        text = source_path.read_text(encoding="utf-8")
        raw: Any = yaml.safe_load(text) if suffix in {".yaml", ".yml"} else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise StudyConfigError(f"cannot parse study file {source_path}: {error}") from error
    if not isinstance(raw, dict):
        raise StudyConfigError("study file top-level must be a mapping")
    try:
        protocol = StudyProtocol.from_dict(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise StudyConfigError(f"invalid study structure: {error}") from error
    return replace(protocol, source_path=source_path)
