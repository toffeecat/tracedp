"""Registries for benchmark / method / model / evaluator.

Adapter modules self-register through decorators, but strict model-provenance
runs must not import arbitrary method modules in the Runner or model-authority
process.  The module manifests below therefore support three explicit scopes:

* runtime: trusted models, benchmark adapters, evaluators;
* one method: exactly the selected adapter, inside its isolated worker;
* all methods: diagnostics/non-strict compatibility only.
"""

from __future__ import annotations

import importlib
import logging
import sys
import warnings
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}
        self._aliases: dict[str, str] = {}
        self._import_failures: list[dict[str, str]] = []

    def register(
        self, key: str, aliases: list[str] | None = None
    ) -> Callable[[type[Any]], type[Any]]:
        def deco(cls: type[Any]) -> type[Any]:
            if key in self._items:
                raise ValueError(f"[{self.name}] duplicate key: {key}")
            self._items[key] = cls
            cls.__registry_key__ = key  # type: ignore[attr-defined]
            for a in aliases or []:
                if a in self._items or a in self._aliases:
                    raise ValueError(f"[{self.name}] alias conflict: {a}")
                self._aliases[a] = key
            return cls

        return deco

    def add_alias(self, alias: str, key: str) -> None:
        """Register an alias `alias` -> `key` after the fact."""
        if alias in self._items or alias in self._aliases:
            raise ValueError(f"[{self.name}] alias already exists: {alias}")
        if key not in self._items:
            raise KeyError(f"[{self.name}] no such key: {key}")
        self._aliases[alias] = key

    def resolve(self, key: str) -> str:
        if key in self._items:
            return key
        if key in self._aliases:
            return self._aliases[key]
        raise KeyError(f"[{self.name}] no such key: {key}; available={sorted(self._items)}")

    def create(self, key: str, *args: Any, **kwargs: Any) -> Any:
        resolved = self.resolve(key)
        return self._items[resolved](*args, **kwargs)

    def registered(self, key: str) -> Callable[..., Any]:
        """Return a registered factory for metadata inspection without constructing it."""

        return self._items[self.resolve(key)]

    def has(self, key: str) -> bool:
        return key in self._items or key in self._aliases

    def available(self) -> list[str]:
        return sorted(self._items)

    def aliases(self) -> list[str]:
        return sorted(self._aliases)

    def import_failures(self) -> list[dict[str, str]]:
        return list(self._import_failures)

    def __contains__(self, key: str) -> bool:
        return key in self._items or key in self._aliases


BENCHMARKS = Registry("benchmark")
METHODS = Registry("method")
MODELS = Registry("model")
EVALUATORS = Registry("evaluator")

# Collect import failures across all registries for diagnostics.
_all_import_failures: list[dict[str, str]] = []


_MODEL_MODULES = (
    "lh_platform.models.dummy",
    "lh_platform.models.openai_chat",
)

_BENCHMARK_MODULES = (
    "lh_platform.benchmarks.webvoyager",
    "lh_platform.benchmarks.toolbench_eval",
    "lh_platform.benchmarks.openhands_swebench",
    "lh_platform.benchmarks.swebench_lite",
    "lh_platform.benchmarks.sotopia",
    "lh_platform.benchmarks.roadmapbench",
    "lh_platform.benchmarks.threat2traffic_bench",
    "lh_platform.benchmarks.scienceworld",
    "lh_platform.benchmarks.scienceworld_official",
    "lh_platform.benchmarks.terminal_bench",
)

_EVALUATOR_MODULES = (
    "lh_platform.evaluators.success",
    "lh_platform.evaluators.long_horizon",
    "lh_platform.evaluators.cost",
)


def _method_adapter_modules() -> dict[str, str]:
    """Load the inventory only in processes that inspect method adapters."""

    from lh_platform.methods.inventory import method_adapter_modules

    return method_adapter_modules()


def load_module_if_exists(dotted: str) -> None:
    """Try to import a module; ignore ImportError but propagate other errors."""
    try:
        importlib.import_module(dotted)
    except ImportError as e:
        _all_import_failures.append({"module": dotted, "error": str(e)})
    except Exception as e:
        warnings.warn(f"[registry] unexpected error importing {dotted}: {e}", stacklevel=2)
        _all_import_failures.append({"module": dotted, "error": str(e)})


def get_import_failures() -> list[dict[str, str]]:
    """Return list of adapter import failures from the last bootstrap call."""
    return list(_all_import_failures)


def _reset_runtime_import_diagnostics() -> None:
    """Reset bootstrap diagnostics without importing any adapter module."""

    _all_import_failures.clear()


def _clear_method_probe_cache() -> None:
    """Clear controller-side optional-repo probes outside the model authority."""

    try:
        from lh_platform.methods.lifecycle import clear_probe_cache

        clear_probe_cache()
    except Exception:
        logger.exception("registry: clear_probe_cache failed")


def bootstrap_model_registry() -> None:
    """Load only trusted model adapters.

    The generation-authority process uses this narrowest scope: importing a
    benchmark or method plugin there would let its module-level code alter the
    process that owns model provenance.
    """

    _reset_runtime_import_diagnostics()
    for dotted in _MODEL_MODULES:
        load_module_if_exists(dotted)


def bootstrap_runtime_registries() -> None:
    """Load controller runtime components without importing method adapters."""

    _reset_runtime_import_diagnostics()
    _clear_method_probe_cache()
    for dotted in (*_MODEL_MODULES, *_BENCHMARK_MODULES, *_EVALUATOR_MODULES):
        load_module_if_exists(dotted)


def bootstrap_method_registry(method_id: str | None = None) -> None:
    """Load one selected method adapter, or all adapters for diagnostics.

    ``method_id`` is intentionally resolved through a static module manifest,
    not decorator registration.  Strict runs invoke this only in the isolated
    method worker, which keeps adapter module-level code out of the Runner
    and model-authority processes.
    """

    module_by_id = _method_adapter_modules()
    if method_id is None:
        modules = tuple(module_by_id.values())
    else:
        dotted = module_by_id.get(str(method_id))
        modules = (dotted,) if dotted is not None else ()
    for dotted in modules:
        load_module_if_exists(dotted)


def is_builtin_method_id(method_id: str) -> bool:
    """Whether the static method manifest permits this id in strict mode."""

    return str(method_id) in _method_adapter_modules()


def loaded_method_adapter_modules() -> tuple[str, ...]:
    """Return declared adapter modules already imported in this process.

    Strict provenance starts from a clean controller process.  Importing an
    adapter before the isolated worker exists would let module-level code
    mutate controller globals, so this is an admission check rather than a
    diagnostic convenience.
    """

    return tuple(dotted for dotted in _method_adapter_modules().values() if dotted in sys.modules)


def bootstrap_default_registries() -> None:
    """Import all built-in adapters for diagnostics and legacy non-strict runs.

    Failures are non-fatal so that missing dependencies in one adapter do not
    break the rest of the platform. Strict provenance runs must use
    ``bootstrap_runtime_registries`` in their controller and
    ``bootstrap_method_registry(method_id)`` in their worker instead.
    """

    bootstrap_runtime_registries()
    bootstrap_method_registry()
