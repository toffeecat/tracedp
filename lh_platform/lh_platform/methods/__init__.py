"""Lazy public exports for method adapters and inventory access."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MethodBase": ("lh_platform.methods.base", "MethodBase"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
