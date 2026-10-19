"""Shared process and GPU admission checks for training workflows.

These checks belong to the training runtime, not to any credit estimator or
warm-start family. Callers may supply their domain error type and dependencies
so legacy exception contracts and focused tests remain stable.
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from lh_platform.training.agentic_rl_harness import (
    assert_exclusive_gpu,
    query_compute_apps,
)


class TrainingRuntimePreflightError(RuntimeError):
    """Raised when process initialization or GPU admission is unsafe."""


def preload_thread_unsafe_native_extensions(
    module_loader: Callable[[str], Any] = importlib.import_module,
    *,
    error_type: type[Exception] = TrainingRuntimePreflightError,
) -> dict[str, Any]:
    """Load native dependencies before AppWorld starts AnyIO portal threads."""

    modules = []
    for module_name in ("pandas", "datasets"):
        try:
            module = module_loader(module_name)
        except Exception as error:
            raise error_type(f"cannot preload native dependency {module_name}: {error}") from error
        modules.append(
            {
                "name": module_name,
                "version": str(getattr(module, "__version__", "unknown")),
                "module_file": str(getattr(module, "__file__", "")),
            }
        )
    return {
        "schema": "lh_native_import_preload/v1",
        "stage": "before_appworld_thread_and_cuda_initialization",
        "modules": modules,
        "reason": "avoid_threaded_lazy_native_import_sigsegv",
    }


def await_exclusive_gpu(
    *,
    device: str,
    allow_shared_gpu: bool,
    timeout_seconds: float,
    query_compute_apps_fn: Callable[[], Sequence[Any]] = query_compute_apps,
    assert_exclusive_gpu_fn: Callable[..., None] = assert_exclusive_gpu,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    error_type: type[Exception] = TrainingRuntimePreflightError,
) -> None:
    """Wait for exclusive GPU admission up to a finite, explicit deadline."""

    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise error_type("wait_for_gpu_seconds must be finite and non-negative")
    deadline = monotonic_fn() + timeout_seconds
    while True:
        compute_apps = query_compute_apps_fn() if device.startswith("cuda") else ()
        try:
            assert_exclusive_gpu_fn(
                device=device,
                allow_shared_gpu=allow_shared_gpu,
                compute_apps=compute_apps,
            )
            return
        except RuntimeError:
            now = monotonic_fn()
            if now >= deadline:
                raise
            sleep_fn(min(2.0, max(0.1, deadline - now)))
