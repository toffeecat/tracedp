"""Load and validate an experiment config file.

A config file can be YAML or JSON. The loader supports two modes:

  1. Free-standing: the file is a complete experiment description
     (with experiment, model, matrix, run, gates sections).
  2. Split: the file references sub-configs in `model.path`, e.g.
       model:
         path: configs/models/qwen3_4b_vllm.yaml
     which is then merged in. This is what the example configs use.

The loader returns an `ExperimentConfig` (schema.py). Downstream code
(runner, gates, summary) only ever reads `ExperimentConfig.to_runner_kwargs()`.

Path resolution:
  - Relative paths in `model.path` are resolved against the **repo root**
    (the parent of `lh_platform/`). This way `configs/models/foo.yaml`
    works no matter where the experiment file lives.
  - `source_path` is still recorded on the loaded config for provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from lh_platform.config.schema import (
    SCHEMA_VERSION,
    ConfigError,
    ExperimentConfig,
    GatesConfig,
    MatrixConfig,
    ModelConfig,
    RunConfig,
)

SUPPORTED_SUFFIXES = {".yaml", ".yml", ".json"}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_yaml_or_json(text: str, suffix: str) -> dict[str, Any]:
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"top-level must be a mapping; got {type(loaded).__name__}")
    return loaded


def _resolve_repo_root(start: Path) -> Path:
    """Locate the workspace root for split experiment/model configuration.

    ``lh_platform/configs`` can itself contain experiment files, so merely
    checking for sibling ``lh_platform/`` and ``configs/`` directories chooses
    the package directory instead of the workspace.  The canonical workspace
    root is the ancestor that owns both the package's ``pyproject.toml`` and
    the root-level model profiles.
    """
    p = start
    for _ in range(8):
        if (p / "lh_platform" / "pyproject.toml").is_file() and (p / "configs" / "models").is_dir():
            return p
        p = p.parent
    return start


def _load_subconfig(repo_root: Path, ref_path: str) -> dict[str, Any]:
    """Load a referenced sub-config (model profile etc.)."""
    p = Path(ref_path)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    if not p.is_file():
        raise ConfigError(f"sub-config not found: {p}")
    text = _read_text(p)
    return _parse_yaml_or_json(text, p.suffix.lower())


def _resolve_model(repo_root: Path, model_section: dict[str, Any]) -> ModelConfig:
    """Resolve a `model:` section, optionally merging in a referenced profile."""
    if not isinstance(model_section, dict):
        raise ConfigError(f"model: expected mapping, got {type(model_section).__name__}")
    ref_path = model_section.get("path")
    if ref_path:
        sub = _load_subconfig(repo_root, ref_path)
        # Sub-config expected to have provider/kwargs. Merge with overrides.
        merged = dict(sub)
        merged.update({k: v for k, v in model_section.items() if k != "path"})
        # If both have kwargs, merge them (override wins).
        if "kwargs" in sub and "kwargs" in model_section:
            sub_kwargs = dict(sub.get("kwargs") or {})
            sub_kwargs.update(model_section["kwargs"])
            merged["kwargs"] = sub_kwargs
        if "runtime" in sub and "runtime" in model_section:
            sub_runtime = dict(sub.get("runtime") or {})
            sub_runtime.update(model_section["runtime"] or {})
            merged["runtime"] = sub_runtime
        return ModelConfig.from_dict(merged)
    return ModelConfig.from_dict(model_section)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a file."""
    p = Path(path).resolve()
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ConfigError(
            f"unsupported config suffix {suffix!r}; expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )
    raw = _parse_yaml_or_json(_read_text(p), suffix)
    return parse_config(raw, source_path=p)


def parse_config(raw: dict[str, Any], source_path: Path | None = None) -> ExperimentConfig:
    """Parse a raw dict into an ExperimentConfig."""
    if not isinstance(raw, dict):
        raise ConfigError(f"top-level must be a mapping; got {type(raw).__name__}")

    repo_root = _resolve_repo_root(source_path.parent) if source_path is not None else Path.cwd()

    experiment = raw.get("experiment") or {}
    if not isinstance(experiment, dict):
        raise ConfigError("experiment: expected mapping")
    name = str(experiment.get("name", "") or "").strip()
    if not name:
        if source_path:
            name = source_path.stem
        else:
            raise ConfigError("experiment.name is required")
    description = str(experiment.get("description", "") or "").strip()

    if "model" not in raw:
        raise ConfigError("model: section is required")
    model = _resolve_model(repo_root, raw["model"])

    if "matrix" not in raw:
        raise ConfigError("matrix: section is required")
    matrix = MatrixConfig.from_dict(raw["matrix"])

    run = RunConfig.from_dict(raw.get("run"))
    gates = GatesConfig.from_dict(raw.get("gates"))

    return ExperimentConfig(
        name=name,
        description=description,
        model=model,
        matrix=matrix,
        run=run,
        gates=gates,
        raw=raw,
        source_path=source_path,
        schema_version=SCHEMA_VERSION,
    )
