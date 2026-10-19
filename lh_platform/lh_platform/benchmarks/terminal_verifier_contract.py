"""Freeze and validate the public Terminal benchmark verifier source."""

from __future__ import annotations

import ast
import base64
import json
from pathlib import Path

SCHEMA_VERSION = "terminal_frozen_verifier_contract_v2"
FILENAME = "managed-runtime-verifier-contract.json"


def _has_parametrize(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Reject parameter expansion until an independent collector is available."""

    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr == "parametrize":
            return True
        if isinstance(target, ast.Name) and target.id == "parametrize":
            return True
    return False


def _decorator_name(decorator: ast.expr) -> str | None:
    """Return the final decorator attribute without trusting its import alias."""

    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return None


def _parameter_aliases(trees: tuple[ast.Module, ...]) -> set[str]:
    """Resolve parameterization aliases across every frozen verifier module."""

    parameter_aliases = {"parametrize"}
    changed = True
    while changed:
        changed = False
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in parameter_aliases:
                            local_name = alias.asname or alias.name
                            if local_name not in parameter_aliases:
                                parameter_aliases.add(local_name)
                                changed = True
                    continue
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                mentions_parameterization = any(
                    (
                        isinstance(child, ast.Name)
                        and child.id in parameter_aliases
                    )
                    or (
                        isinstance(child, ast.Attribute)
                        and child.attr == "parametrize"
                    )
                    or (
                        isinstance(child, ast.Constant)
                        and child.value == "parametrize"
                    )
                    for child in ast.walk(node.value)
                )
                if not mentions_parameterization:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in parameter_aliases:
                        parameter_aliases.add(target.id)
                        changed = True
    return parameter_aliases


def _validate_no_dynamic_pytest_expansion(
    tree: ast.Module, *, parameter_aliases: set[str]
) -> None:
    """Fail closed on pytest mechanisms whose collected node IDs require pytest."""

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
                raise ValueError("module-level pytestmark contract is unsupported")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "pytest_generate_tests":
                raise ValueError("dynamic official pytest contract is unsupported")
            for decorator in node.decorator_list:
                if _has_parametrize(node) or _decorator_name(decorator) in parameter_aliases:
                    raise ValueError("parameterized official pytest contract is unsupported")
                if (
                    isinstance(decorator, ast.Call)
                    and any(keyword.arg == "params" for keyword in decorator.keywords)
                ):
                    raise ValueError("parameterized official pytest fixture is unsupported")
            if node.name.startswith("test_") and node.decorator_list:
                raise ValueError("decorated official pytest test is unsupported")
        elif isinstance(node, ast.ClassDef):
            if any(
                _decorator_name(item) in parameter_aliases for item in node.decorator_list
            ):
                raise ValueError("parameterized official pytest class is unsupported")
            if node.name.startswith("Test") and node.decorator_list:
                raise ValueError("decorated official pytest class is unsupported")


def extract_pytest_nodes(test_sources: dict[str, str], test_script: str) -> tuple[str, ...]:
    """Extract exact non-parameterized pytest nodes from frozen source text."""

    if "pytest" not in test_script or not test_sources:
        raise ValueError("official pytest contract is unavailable")
    expected: set[str] = set()
    parsed = tuple(
        (filename, ast.parse(source, filename=filename))
        for filename, source in sorted(test_sources.items())
    )
    aliases = _parameter_aliases(tuple(tree for _, tree in parsed))
    for filename, tree in parsed:
        _validate_no_dynamic_pytest_expansion(tree, parameter_aliases=aliases)
        if not Path(filename).name.startswith("test"):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    # An assignment can replace a plain test function with a
                    # parametrized wrapper after the function definition.
                    if _has_parametrize(node) or node.name in aliases:
                        raise ValueError("parameterized official pytest contract is unsupported")
                    expected.add(f"{filename}::{node.name}")
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                        if _has_parametrize(child):
                            raise ValueError("parameterized official pytest contract is unsupported")
                        expected.add(f"{filename}::{node.name}::{child.name}")
    if not expected:
        raise ValueError("official pytest contract contains no tests")
    return tuple(sorted(expected))


def snapshot_task_verifier(task_path: Path) -> dict[str, object]:
    """Capture exact task bytes and independently derive the pytest node contract."""

    task_path = task_path.resolve()
    tests_root = task_path / "tests"
    script_path = tests_root / "test.sh"
    python_paths = sorted(
        path for path in tests_root.rglob("*.py") if not path.name.startswith("._")
    )
    if not script_path.is_file() or not python_paths:
        raise ValueError(f"official verifier source is missing: {task_path}")
    test_script = script_path.read_text()
    test_sources = {
        str(path.relative_to(tests_root)): path.read_text() for path in python_paths
    }
    source_files = []
    for path in sorted(item for item in task_path.rglob("*") if item.is_file()):
        relative = str(path.relative_to(task_path))
        source_files.append(
            {
                "path": relative,
                "mode": path.stat().st_mode & 0o7777,
                # Exact bytes avoid a digest-based identity claim and also cover binary fixtures.
                "bytes_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    return {
        "task_path": str(task_path),
        "test_script": test_script,
        "test_sources": test_sources,
        "expected_nodes": list(extract_pytest_nodes(test_sources, test_script)),
        "source_files": source_files,
    }


def write_job_verifier_contract(job_dir: Path, task_paths: tuple[Path, ...]) -> Path:
    """Persist one immutable-by-convention contract before Harbor starts."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tasks": [snapshot_task_verifier(path) for path in task_paths],
    }
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / FILENAME
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("frozen verifier contract differs from existing job contract")
        return path
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def _load_validated_task_snapshot(
    trial: Path, task_path: Path
) -> dict[str, object] | None:
    """Return one exact frozen task snapshot only while every source byte still matches."""

    path = trial.parent / FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        return None
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return None
    task_path = task_path.resolve()
    matches = [
        item
        for item in tasks
        if isinstance(item, dict) and item.get("task_path") == str(task_path)
    ]
    if len(matches) != 1:
        return None
    item = matches[0]
    try:
        current = snapshot_task_verifier(task_path)
    except (OSError, SyntaxError, ValueError):
        return None
    if current != item:
        return None
    return item


def load_trial_public_instruction(trial: Path, task_path: Path) -> str | None:
    """Read the public instruction from the already frozen task-byte contract."""

    item = _load_validated_task_snapshot(trial, task_path)
    if item is None:
        return None
    source_files = item.get("source_files")
    if not isinstance(source_files, list):
        return None
    matches = [
        row
        for row in source_files
        if isinstance(row, dict) and row.get("path") == "instruction.md"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("bytes_base64"), str):
        return None
    try:
        return base64.b64decode(matches[0]["bytes_base64"], validate=True).decode().strip()
    except (ValueError, UnicodeDecodeError):
        return None


def load_trial_pytest_nodes(trial: Path, task_path: Path) -> set[str] | None:
    """Validate current frozen text against the pre-run contract and return nodes."""

    item = _load_validated_task_snapshot(trial, task_path)
    if item is None:
        return None
    nodes = item.get("expected_nodes")
    if not isinstance(nodes, list) or not all(isinstance(node, str) for node in nodes):
        return None
    return set(nodes)


__all__ = [
    "FILENAME",
    "extract_pytest_nodes",
    "load_trial_public_instruction",
    "load_trial_pytest_nodes",
    "snapshot_task_verifier",
    "write_job_verifier_contract",
]
