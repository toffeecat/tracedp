"""Method Inventory access.

The Method Inventory is the source of truth for known method metadata. Python
method modules remain the implementation modules; this file reads the structured
inventory used by Run Credibility and consistency tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from lh_platform.core.method_capabilities import (
    canonical_integration_level,
    normalize_action_contracts,
)

_FORBIDDEN_STRICT_METHOD_KWARG_PARTS = (
    "api_key",
    "access_token",
    "auth_token",
    "base_url",
    "bearer_token",
    "client_secret",
    "model_id",
    "model_kwargs",
    "model_registry",
    "password",
    "secret",
)

_METHOD_MODULE_PREFIX = "lh_platform.methods"


@dataclass(frozen=True)
class MethodInventoryEntry:
    slug: str
    name: str
    integration_level: str
    supported_execution_paths: tuple[str, ...]
    # A deliberately small, source-audited allowlist for strict provenance
    # runs.  Such methods must obtain all LLM output through MethodBase's
    # bound generate-only proxy; native adapters that construct their own
    # client/subprocess stay false.
    strict_proxy_compatible: bool = False
    # Constructor parameters explicitly approved to cross into the strict
    # method worker. ``seed`` is handled separately by Run Admission and is
    # never passed to a constructor.
    strict_method_kwargs: tuple[str, ...] = ()
    requires_benchmark: str = ""
    # ``None`` denotes a generic prompt policy that follows task-provided
    # action syntax.  A populated tuple is a hard emission constraint.
    supported_action_contracts: tuple[str, ...] | None = None
    repo_url: str | None = None
    local_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_method_inventory() -> dict[str, MethodInventoryEntry]:
    path = Path(__file__).with_name("_inventory.yaml")
    raw = yaml.safe_load(path.read_text()) or {}
    entries: dict[str, MethodInventoryEntry] = {}
    for item in raw.get("methods", []):
        known = {
            "slug",
            "name",
            "integration_level",
            "supported_execution_paths",
            "strict_proxy_compatible",
            "strict_method_kwargs",
            "requires_benchmark",
            "supported_action_contracts",
            "repo_url",
            "local_path",
        }
        extra = {k: v for k, v in item.items() if k not in known}
        strict_proxy_compatible = item.get("strict_proxy_compatible", False)
        if not isinstance(strict_proxy_compatible, bool):
            raise ValueError(
                f"method inventory {item.get('slug')!r}.strict_proxy_compatible must be boolean"
            )
        raw_strict_method_kwargs = item.get("strict_method_kwargs", [])
        if not isinstance(raw_strict_method_kwargs, (list, tuple)) or not all(
            isinstance(key, str) and key.strip() for key in raw_strict_method_kwargs
        ):
            raise ValueError(
                f"method inventory {item.get('slug')!r}.strict_method_kwargs "
                "must be a list of non-empty strings"
            )
        strict_method_kwargs = tuple(key.strip() for key in raw_strict_method_kwargs)
        if len(set(strict_method_kwargs)) != len(strict_method_kwargs):
            raise ValueError(
                f"method inventory {item.get('slug')!r}.strict_method_kwargs "
                "must not contain duplicates"
            )
        if strict_method_kwargs and not strict_proxy_compatible:
            raise ValueError(
                f"method inventory {item.get('slug')!r}.strict_method_kwargs "
                "requires strict_proxy_compatible=true"
            )
        forbidden = [
            key
            for key in strict_method_kwargs
            if any(part in key.strip().lower() for part in _FORBIDDEN_STRICT_METHOD_KWARG_PARTS)
        ]
        if forbidden:
            raise ValueError(
                f"method inventory {item.get('slug')!r}.strict_method_kwargs "
                f"contains reserved model/credential fields: {forbidden!r}"
            )
        raw_contracts = item.get("supported_action_contracts")
        if raw_contracts is not None and (
            not isinstance(raw_contracts, (list, tuple))
            or not all(isinstance(contract, str) and contract.strip() for contract in raw_contracts)
        ):
            raise ValueError(
                "method inventory "
                f"{item.get('slug')!r}.supported_action_contracts must be a list of non-empty strings"
            )
        try:
            supported_action_contracts = normalize_action_contracts(raw_contracts)
        except ValueError as error:
            raise ValueError(
                "method inventory "
                f"{item.get('slug')!r}.supported_action_contracts must be a list of non-empty strings"
            ) from error
        integration_level = canonical_integration_level(item["integration_level"])
        slug = str(item["slug"])
        if not slug.isidentifier() or slug.startswith("_"):
            raise ValueError(f"method inventory slug must be a public Python identifier: {slug!r}")
        if slug in entries:
            raise ValueError(f"duplicate method inventory slug: {slug!r}")
        entry = MethodInventoryEntry(
            slug=slug,
            name=str(item.get("name") or item["slug"]),
            integration_level=integration_level,
            supported_execution_paths=tuple(
                str(p) for p in item.get("supported_execution_paths", [])
            ),
            strict_proxy_compatible=strict_proxy_compatible,
            strict_method_kwargs=strict_method_kwargs,
            requires_benchmark=str(item.get("requires_benchmark") or ""),
            supported_action_contracts=supported_action_contracts,
            repo_url=item.get("repo_url"),
            local_path=item.get("local_path"),
            extra=extra,
        )
        entries[entry.slug] = entry
    return entries


def method_adapter_modules() -> dict[str, str]:
    """Return the inventory-owned method id to built-in module mapping."""

    return {slug: f"{_METHOD_MODULE_PREFIX}.{slug}" for slug in load_method_inventory()}


def get_method_inventory_entry(slug: str) -> MethodInventoryEntry | None:
    return load_method_inventory().get(slug)


def method_integration_level(slug: str, fallback: str) -> str:
    entry = get_method_inventory_entry(slug)
    return entry.integration_level if entry is not None else fallback


def method_strict_proxy_compatible(slug: str) -> bool:
    """Whether an inventory-audited adapter may use strict model provenance."""

    entry = get_method_inventory_entry(slug)
    return bool(entry and entry.strict_proxy_compatible)


def method_supported_execution_paths(slug: str) -> tuple[str, ...]:
    entry = get_method_inventory_entry(slug)
    return entry.supported_execution_paths if entry is not None else ()
