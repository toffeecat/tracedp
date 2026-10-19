"""Exact task-contract receipts for Terminal-Bench source and recovery runs."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

_SCHEMA_VERSION = "terminal_task_contract_receipt_v2"
_BENCHMARK_ID = "terminal-bench-2.1"
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "benchmark_id",
        "task_name",
        "receipt_visibility",
        "included_components",
        "files",
    }
)
_FILE_FIELDS = frozenset({"path", "byte_count", "mode"})
_CORE_FILES = ("instruction.md", "task.toml")
_RUNTIME_DIRECTORIES = ("environment", "tests")
_IGNORED_NAMES = frozenset({".DS_Store"})


def _runtime_files(task_dir: Path) -> tuple[Path, ...]:
    """Enumerate only files that define instructions, initial state, or scoring."""

    if not task_dir.is_dir():
        raise FileNotFoundError(f"Terminal task directory is missing: {task_dir}")
    paths: list[Path] = []
    for relative in _CORE_FILES:
        path = task_dir / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Terminal task contract file is missing or linked: {relative}")
        paths.append(path)
    for relative in _RUNTIME_DIRECTORIES:
        root = task_dir / relative
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Terminal task contract directory is missing or linked: {relative}")
        for path in root.rglob("*"):
            if path.name in _IGNORED_NAMES or path.name.startswith("._"):
                continue
            if path.is_symlink():
                raise ValueError(
                    f"Terminal task contract cannot freeze a linked path: "
                    f"{path.relative_to(task_dir)}"
                )
            if path.is_file():
                paths.append(path)
    return tuple(sorted(paths, key=lambda path: path.relative_to(task_dir).as_posix()))


def freeze_terminal_task_contract(
    task_dir: Path,
    receipt_dir: Path,
    *,
    benchmark_id: str = _BENCHMARK_ID,
) -> dict[str, object]:
    """Copy exact runtime-defining bytes before a source episode starts."""

    if receipt_dir.exists():
        raise FileExistsError(f"Terminal task contract receipt already exists: {receipt_dir}")
    if benchmark_id != _BENCHMARK_ID:
        raise ValueError("Terminal task contract benchmark differs")
    files = _runtime_files(task_dir)
    snapshot = receipt_dir / "snapshot"
    snapshot.mkdir(parents=True, mode=0o700)
    receipt_dir.chmod(0o700)
    entries: list[dict[str, object]] = []
    for source in files:
        relative = source.relative_to(task_dir)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append(
            {
                "path": relative.as_posix(),
                "byte_count": source.stat().st_size,
                "mode": f"{source.stat().st_mode & 0o777:03o}",
            }
        )
    manifest: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "task_name": task_dir.name,
        "receipt_visibility": "evaluator_only",
        "included_components": [
            "public_instruction",
            "task_configuration",
            "initial_environment",
            "private_verifier",
        ],
        "files": entries,
    }
    contract_path = receipt_dir / "contract.json"
    contract_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    contract_path.chmod(0o600)
    return manifest


def verify_terminal_task_contract(
    task_dir: Path,
    receipt_dir: Path,
    *,
    expected_task: str,
) -> dict[str, object]:
    """Require the continuation task to match every frozen runtime-defining byte."""

    contract_path = receipt_dir / "contract.json"
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
        raise ValueError("Terminal task contract receipt schema differs")
    if (
        value["schema_version"] != _SCHEMA_VERSION
        or value["benchmark_id"] != _BENCHMARK_ID
        or value["task_name"] != expected_task
        or task_dir.name != expected_task
        or value["receipt_visibility"] != "evaluator_only"
        or value["included_components"]
        != [
            "public_instruction",
            "task_configuration",
            "initial_environment",
            "private_verifier",
        ]
    ):
        raise ValueError("Terminal task contract belongs to a different task")
    rows = value["files"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Terminal task contract file inventory is empty")
    recorded: dict[str, tuple[int, str]] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != _FILE_FIELDS
            or not isinstance(row["path"], str)
            or not row["path"]
            or type(row["byte_count"]) is not int
            or row["byte_count"] < 0
            or not isinstance(row["mode"], str)
            or len(row["mode"]) != 3
            or any(character not in "01234567" for character in row["mode"])
            or row["path"] in recorded
        ):
            raise ValueError("Terminal task contract file inventory differs")
        recorded[row["path"]] = (row["byte_count"], row["mode"])
    current = {
        path.relative_to(task_dir).as_posix(): path for path in _runtime_files(task_dir)
    }
    if set(recorded) != set(current):
        raise ValueError("Terminal task contract file set differs")
    for relative, source in current.items():
        frozen = receipt_dir / "snapshot" / relative
        byte_count, mode = recorded[relative]
        if f"{source.stat().st_mode & 0o777:03o}" != mode:
            raise ValueError(f"Terminal task contract mode differs: {relative}")
        if (
            not frozen.is_file()
            or frozen.is_symlink()
            or f"{frozen.stat().st_mode & 0o777:03o}" != mode
            or source.stat().st_size != byte_count
            or frozen.stat().st_size != byte_count
            or source.read_bytes() != frozen.read_bytes()
        ):
            raise ValueError(f"Terminal task contract content differs: {relative}")
    return {
        **dict(value),
        "included_components": list(value["included_components"]),
        "files": [dict(row) for row in rows],
    }


def resolve_terminal_source_task_contract(
    source_map_receipt: Path,
    *,
    storage_root: Path,
    source_trial: Path,
    expected_task: str,
    expected_public_paths: tuple[str, ...],
) -> Path:
    """Resolve the exact task contract bound to one formal source trajectory."""

    root = storage_root.resolve()
    source_map = source_map_receipt.resolve()
    trial = source_trial.resolve()
    if root not in source_map.parents or root not in trial.parents:
        raise ValueError("Terminal source evidence must remain under LONGH_STORAGE_ROOT")
    value = json.loads(source_map.read_text(encoding="utf-8"))
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "benchmark_id", "sources"}
        or value["schema_version"] != "terminal_source_map_v2"
        or value["benchmark_id"] != _BENCHMARK_ID
        or not isinstance(value["sources"], list)
    ):
        raise ValueError("Terminal source map schema differs")
    matches = [
        row
        for row in value["sources"]
        if isinstance(row, Mapping) and row.get("task") == expected_task
    ]
    if len(matches) != 1:
        raise ValueError("Terminal source map must bind exactly one matching task")
    row = matches[0]
    public_paths = row.get("public_paths")
    if not isinstance(public_paths, list) or tuple(public_paths) != expected_public_paths:
        raise ValueError("Terminal source map public paths differ from the recovery case")
    if row.get("case_status") != "terminal":
        raise ValueError("Terminal forward recovery requires a terminal source case")

    run_relative = row.get("run_relative_path")
    contract_relative = row.get("task_contract_receipt_relative_path")
    if not isinstance(run_relative, str) or not isinstance(contract_relative, str):
        raise ValueError("Terminal source map lacks its run or task-contract binding")
    run_root = (root / run_relative).resolve()
    contract = (root / contract_relative).resolve()
    if root not in run_root.parents or root not in contract.parents:
        raise ValueError("Terminal source map escaped LONGH_STORAGE_ROOT")
    if trial.parent != run_root:
        raise ValueError("Terminal source trial differs from its frozen source map")
    if not (contract / "contract.json").is_file():
        raise ValueError("Terminal source task contract receipt is missing")
    return contract


__all__ = [
    "freeze_terminal_task_contract",
    "resolve_terminal_source_task_contract",
    "verify_terminal_task_contract",
]
