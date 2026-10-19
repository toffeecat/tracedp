"""Git provenance bindings for a structured YAML configuration tree."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from lh_platform.training.dataset import sha256_file


class ConfigProvenanceError(ValueError):
    """Raised when a referenced configuration tree cannot be bound safely."""


def repository_config_tree_bindings(
    resolved: Any,
) -> tuple[Mapping[str, Any], ...]:
    """Bind a resolved config and every recursively referenced YAML to HEAD."""

    root = Path(resolved.workspace_root).resolve()
    pending = [Path(resolved.config_path).resolve()]
    visited: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        if not path.is_relative_to(root) or not path.is_file():
            raise ConfigProvenanceError("config reference is missing or outside the workspace")
        visited.add(path)
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigProvenanceError(f"cannot inspect config reference: {path}") from error
        for reference in _yaml_references(document):
            pending.append((root / reference).resolve())
    return tuple(
        repository_file_binding(path) for path in sorted(visited, key=lambda item: item.as_posix())
    )


def repository_file_binding(path: Path) -> dict[str, Any]:
    """Bind one workspace file to the exact bytes committed at Git HEAD."""

    target = path.resolve()
    repository = Path(
        subprocess.check_output(
            ["git", "-C", str(target.parent), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    ).resolve()
    try:
        relative = target.relative_to(repository).as_posix()
    except ValueError as error:
        raise ConfigProvenanceError("config file is outside its resolved Git repository") from error
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    try:
        committed: bytes | None = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"HEAD:{relative}"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        committed = None
    return {
        "repository_root": str(repository),
        "relative_path": relative,
        "revision": revision,
        "file_sha256": sha256_file(target),
        "matches_head": committed is not None and committed == target.read_bytes(),
    }


def repository_subtree_binding(path: str | Path) -> dict[str, Any]:
    """Bind an imported source subtree to its clean content at Git HEAD."""

    target = Path(path).resolve()
    repository = Path(
        subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    ).resolve()
    try:
        relative = target.relative_to(repository).as_posix() or "."
    except ValueError as error:
        raise ConfigProvenanceError(
            "implementation subtree is outside its Git repository"
        ) from error
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    treeish = "HEAD^{tree}" if relative == "." else f"HEAD:{relative}"
    try:
        head_tree_object_id = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", treeish],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ConfigProvenanceError("implementation subtree is not present at Git HEAD") from error
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative,
        ],
        text=True,
    ).splitlines()
    return {
        "repository_root": str(repository),
        "relative_path": relative,
        "revision": revision,
        "head_tree_object_id": head_tree_object_id,
        "worktree_status": status,
        "matches_head": not status,
    }


def _yaml_references(value: Any, *, key: str = "") -> tuple[str, ...]:
    references: list[str] = []
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            label = str(child_key)
            if label.endswith("_ref") and isinstance(child, str):
                references.append(child)
            elif label.endswith("_refs") and isinstance(child, list):
                references.extend(str(item) for item in child if isinstance(item, str))
            else:
                references.extend(_yaml_references(child, key=label))
    elif isinstance(value, list):
        if key == "paired_experiments":
            references.extend(str(item) for item in value if isinstance(item, str))
        else:
            for child in value:
                references.extend(_yaml_references(child, key=key))
    return tuple(references)


__all__ = [
    "ConfigProvenanceError",
    "repository_config_tree_bindings",
    "repository_file_binding",
    "repository_subtree_binding",
]
