"""One validated runtime contract shared by every model-backed native tracer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256


def normalize_python_entrypoint(python_executable: str) -> Path:
    """Return an absolute executable path without dereferencing venv symlinks."""

    discovered = Path(shutil.which(python_executable) or python_executable).expanduser()
    entrypoint = Path(os.path.abspath(discovered))
    if not entrypoint.is_file():
        raise ValueError(f"native trace Python executable is missing: {entrypoint}")
    return entrypoint


@dataclass(frozen=True)
class NativeTraceRuntimeV1:
    model_id: str
    base_url: str
    api_key: str = "EMPTY"
    python_executable: str = sys.executable
    max_tokens: int = 512
    max_model_calls: int | None = None
    timeout_seconds: int = 180
    temperature: float = 0.0
    generation_seed: int = 0
    diagnostic_artifact_root: str | None = None
    # MiniMax's OpenAI-compatible endpoint reasons inside the content (``<think>``)
    # while reporting zero reasoning tokens; this sends ``thinking: disabled``.
    disable_thinking: bool = False
    # Contract v0.2 R3: charge tracer calls to the arm's shared budget pool.
    budget_ledger_path: str | None = None
    budget_role: str | None = None

    def __post_init__(self) -> None:
        if (self.budget_ledger_path is None) != (self.budget_role is None):
            raise ValueError("native trace budget needs both a ledger path and a role")
        if not self.model_id or not self.base_url:
            raise ValueError("native trace runtime requires model id and base URL")
        if self.max_tokens < 1 or self.timeout_seconds < 1:
            raise ValueError("native trace runtime limits must be positive")
        if self.max_model_calls is not None and self.max_model_calls < 1:
            raise ValueError("native trace model-call limit must be positive or null")
        if not math.isfinite(self.temperature) or not 0.0 <= self.temperature <= 2.0:
            raise ValueError("native trace runtime temperature must be in [0, 2]")
        if isinstance(self.generation_seed, bool) or self.generation_seed < 0:
            raise ValueError("native trace runtime seed must be non-negative")
        executable = normalize_python_entrypoint(self.python_executable)
        object.__setattr__(self, "python_executable", str(executable))
        if self.diagnostic_artifact_root is not None:
            root = Path(self.diagnostic_artifact_root).expanduser()
            if not root.is_absolute():
                raise ValueError("native trace diagnostic artifact root must be absolute")
            object.__setattr__(self, "diagnostic_artifact_root", str(root.resolve()))

    def worker_arguments(self) -> tuple[str, ...]:
        arguments = (
            "--model",
            self.model_id,
            "--base-url",
            self.base_url,
            "--max-tokens",
            str(self.max_tokens),
        )
        if self.max_model_calls is not None:
            arguments += ("--max-model-calls", str(self.max_model_calls))
        arguments += (
            "--temperature",
            str(self.temperature),
            "--generation-seed",
            str(self.generation_seed),
            "--timeout-seconds",
            str(self.timeout_seconds),
        )
        # Only workers that accept these flags may receive them (Who&When does).
        if self.disable_thinking:
            arguments += ("--disable-thinking",)
        if self.budget_ledger_path is not None:
            arguments += (
                "--budget-ledger-path",
                self.budget_ledger_path,
                "--budget-role",
                str(self.budget_role),
            )
        return arguments


@lru_cache(maxsize=32)
def _probe_python_modules(
    python_executable: str,
    modules: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    script = """
import importlib
import json
import sys

errors = {}
for module_name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(module_name)
    except Exception as error:
        errors[module_name] = f"{type(error).__name__}: {error}"
print(json.dumps(errors, sort_keys=True))
"""
    completed = subprocess.run(
        [python_executable, "-c", script, json.dumps(modules)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        return (("<probe>", error[-500:]),)
    try:
        errors = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return (("<probe>", "invalid dependency probe response"),)
    if not isinstance(errors, dict):
        return (("<probe>", "dependency probe response is not an object"),)
    return tuple(sorted((str(key), str(value)) for key, value in errors.items()))


def require_python_modules(
    runtime: NativeTraceRuntimeV1,
    *,
    method_name: str,
    modules: tuple[str, ...],
) -> None:
    """Fail native admission before the benchmark if worker imports are unavailable."""

    if not method_name or not modules or len(modules) != len(set(modules)):
        raise ValueError("native trace dependency contract is invalid")
    errors = _probe_python_modules(runtime.python_executable, modules)
    if errors:
        missing = ", ".join(name for name, _ in errors)
        details = "; ".join(f"{name}={error}" for name, error in errors)
        raise ValueError(f"{method_name} runtime dependencies are missing: {missing}; {details}")


def require_python_worker_entrypoint(
    runtime: NativeTraceRuntimeV1,
    *,
    method_name: str,
    worker_path: str | Path,
    environment: Mapping[str, str],
    arguments: Sequence[str] = ("--help",),
) -> None:
    """Prove an isolated worker can import before a GPU-backed matrix starts.

    Direct package checks are insufficient when a worker imports LongH's own
    package graph at module load time. Callers may pass a worker-specific
    import-only command that also loads pinned upstream tracer code. It must not
    construct an API client or contact an endpoint.
    """

    worker = Path(worker_path).resolve()
    if (
        not method_name
        or not worker.is_file()
        or not arguments
        or any(not isinstance(argument, str) or not argument for argument in arguments)
    ):
        raise ValueError("native trace worker entrypoint contract is invalid")
    completed = subprocess.run(
        [runtime.python_executable, str(worker), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(environment),
        cwd=str(worker.parent),
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(
            f"{method_name} entrypoint import preflight failed: {error[-1000:]}"
        )


def attest_native_trace_python_environment(python_executable: str) -> dict[str, Any]:
    """Return a content-addressed inventory for an isolated native worker.

    A tracer matrix invokes the released implementations in a separate Python
    process.  Recording only the executable path leaves the installed package
    graph unconstrained, which makes a resumed or moved run non-reproducible.
    This attestation deliberately inventories all installed distributions rather
    than relying on the host process's environment.
    """

    entrypoint = normalize_python_entrypoint(python_executable)
    script = """
import importlib.metadata
import json
import platform
import sys

distributions = sorted({
    (str(item.metadata.get("Name", "")).strip().casefold().replace("_", "-")
     + "==" + str(item.version))
    for item in importlib.metadata.distributions()
    if str(item.metadata.get("Name", "")).strip()
})
print(json.dumps({
    "executable": sys.executable,
    "implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "is_virtual_environment": sys.prefix != sys.base_prefix,
    "distributions": distributions,
}, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [str(entrypoint), "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"could not attest native trace Python environment: {error[-1000:]}")
    try:
        environment = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("native trace Python environment attestation is invalid") from error
    if (
        not isinstance(environment, dict)
        or Path(str(environment.get("executable", ""))).resolve() != entrypoint.resolve()
        or not isinstance(environment.get("implementation"), str)
        or not isinstance(environment.get("python_version"), str)
        or not isinstance(environment.get("is_virtual_environment"), bool)
        or not isinstance(environment.get("distributions"), list)
        or not all(isinstance(item, str) and "==" in item for item in environment["distributions"])
    ):
        raise ValueError("native trace Python environment attestation changed")
    payload: dict[str, Any] = {
        "schema": "lh_native_trace_python_environment/v1",
        "python_resolved_path": str(entrypoint.resolve()),
        "python_file_sha256": "sha256:" + hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
        "environment": environment,
    }
    payload["content_sha256"] = "sha256:" + canonical_sha256(payload)
    return payload


__all__ = [
    "NativeTraceRuntimeV1",
    "attest_native_trace_python_environment",
    "normalize_python_entrypoint",
    "require_python_worker_entrypoint",
    "require_python_modules",
]
