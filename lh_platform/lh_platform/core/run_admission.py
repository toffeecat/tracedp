"""Fail-closed Run Admission at the Method Inventory seam."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lh_platform.core.method_capabilities import MethodCapabilities
from lh_platform.core.registry import (
    METHODS,
    bootstrap_method_registry,
    is_builtin_method_id,
)
from lh_platform.methods.inventory import (
    MethodInventoryEntry,
    get_method_inventory_entry,
)


@dataclass(frozen=True)
class StrictMethodLaunch:
    """The complete, inventory-vetted input allowed into a strict worker."""

    method_id: str
    constructor_kwargs: tuple[tuple[str, Any], ...] = ()
    seed: int | None = None

    def constructor_kwargs_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.constructor_kwargs))


@dataclass(frozen=True)
class MethodAdmission:
    """One decision returned before model or method dispatch."""

    method_id: str
    declared: MethodCapabilities
    strict_proxy_compatible: bool
    reason: str = ""
    strict_launch: StrictMethodLaunch | None = None

    @property
    def admitted(self) -> bool:
        return not self.reason

    @property
    def slug(self) -> str:
        return self.method_id

    @property
    def integration_level(self) -> str:
        return self.declared.integration_level

    @property
    def requires_benchmark(self) -> str:
        return self.declared.requires_benchmark

    @property
    def supported_action_contracts(self) -> tuple[str, ...] | None:
        return self.declared.supported_action_contracts


@dataclass(frozen=True)
class MethodCapabilityReceipt:
    """Audit comparison; declared capabilities remain authoritative."""

    declared: MethodCapabilities
    observed: MethodCapabilities
    mismatches: tuple[str, ...]

    @property
    def matches_declared(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared.to_dict(),
            "observed": self.observed.to_dict(),
            "mismatches": list(self.mismatches),
            "matches_declared": self.matches_declared,
            "claim_scope": "runtime_observation_does_not_override_method_inventory",
        }


def audit_runtime_capabilities(
    declared: MethodCapabilities,
    observed: MethodCapabilities,
) -> MethodCapabilityReceipt:
    mismatches = tuple(
        field
        for field in (
            "integration_level",
            "requires_benchmark",
            "supported_action_contracts",
        )
        if getattr(declared, field) != getattr(observed, field)
    )
    return MethodCapabilityReceipt(
        declared=declared,
        observed=observed,
        mismatches=mismatches,
    )


def resolve_method_admission(
    *,
    method_id: str,
    benchmark_id: str,
    benchmark_required_action_contract: str,
    strict: bool,
    preloaded_adapters: Sequence[str] = (),
    method_kwargs: Mapping[str, Any] | None = None,
) -> MethodAdmission:
    """Resolve all method-side admission facts through one interface.

    Policy: the method inventory entry is the single source of truth for
    ``integration_level``.  A method class attribute (e.g. ``paper_only``)
    is advisory only; admission gates compare against the inventory-declared
    level.  A method cannot self-promote by mutating its class attribute at
    runtime — the inventory wins.
    """

    entry = get_method_inventory_entry(method_id)
    method_type: Any | None = None
    reason = ""

    if strict:
        if entry is None:
            declared = MethodCapabilities("unknown")
        else:
            declared = _capabilities_from_inventory(entry)
        if preloaded_adapters:
            reason = (
                "strict generation provenance requires a fresh controller process; "
                "preloaded adapter modules=" + ", ".join(preloaded_adapters)
            )
        elif not is_builtin_method_id(method_id) or entry is None:
            reason = (
                "strict generation provenance requires a declared built-in method manifest entry"
            )
        elif not entry.strict_proxy_compatible:
            reason = (
                "strict generation provenance requires method inventory "
                "strict_proxy_compatible=true"
            )
    else:
        bootstrap_method_registry(method_id)
        if not METHODS.has(method_id):
            raise KeyError(f"method not registered after import: {method_id}")
        method_type = METHODS._items[METHODS.resolve(method_id)]
        declared = (
            _capabilities_from_inventory(entry)
            if entry is not None
            else MethodCapabilities.observe(method_type)
        )

    if not reason and declared.requires_benchmark and declared.requires_benchmark != benchmark_id:
        reason = f"method requires_benchmark={declared.requires_benchmark!r}, not {benchmark_id!r}"
    required_contract = str(benchmark_required_action_contract or "")
    supported_contracts = declared.supported_action_contracts
    if (
        not reason
        and required_contract
        and supported_contracts is not None
        and required_contract not in supported_contracts
    ):
        reason = (
            "method supported_action_contracts="
            f"{list(supported_contracts)!r} are incompatible with "
            f"benchmark required_action_contract={required_contract!r}"
        )

    launch: StrictMethodLaunch | None = None
    if strict and not reason and entry is not None:
        launch, reason = _strict_launch(entry, method_kwargs or {})

    return MethodAdmission(
        method_id=method_id,
        declared=declared,
        strict_proxy_compatible=bool(entry and entry.strict_proxy_compatible),
        reason=reason,
        strict_launch=launch,
    )


def _capabilities_from_inventory(entry: MethodInventoryEntry) -> MethodCapabilities:
    return MethodCapabilities(
        integration_level=entry.integration_level,
        requires_benchmark=entry.requires_benchmark,
        supported_action_contracts=entry.supported_action_contracts,
    )


def _strict_launch(
    entry: MethodInventoryEntry,
    method_kwargs: Mapping[str, Any],
) -> tuple[StrictMethodLaunch | None, str]:
    allowed = set(entry.strict_method_kwargs)
    supplied = {
        str(key): value for key, value in method_kwargs.items() if str(key) != "execution_path"
    }
    unsupported = sorted(set(supplied) - allowed - {"seed"})
    if unsupported:
        return None, (
            "strict method admission rejects method_kwargs not listed in Method Inventory "
            f"strict_method_kwargs: {unsupported!r}"
        )

    raw_seed = supplied.pop("seed", None)
    if raw_seed is not None and (isinstance(raw_seed, bool) or not isinstance(raw_seed, int)):
        return None, "strict method seed must be an integer or null"

    selected = {key: supplied[key] for key in sorted(allowed) if key in supplied}
    try:
        encoded = json.dumps(selected, allow_nan=False, sort_keys=True)
        safe_selected = json.loads(encoded)
    except (TypeError, ValueError) as error:
        return None, f"strict method kwargs must be JSON-safe: {error}"

    return (
        StrictMethodLaunch(
            method_id=entry.slug,
            constructor_kwargs=tuple(sorted(safe_selected.items())),
            seed=raw_seed,
        ),
        "",
    )
