"""Canonical method capability vocabulary and runtime observations.

The Method Inventory owns declared capabilities.  This module only defines
their shape and how an instantiated method may report an *observation* for
audit; an observation never upgrades or replaces the inventory contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

INTEGRATION_MOCK = "mock"
INTEGRATION_PAPER_ONLY = "paper_only"
INTEGRATION_PAPER_FAITHFUL = "paper-faithful"
INTEGRATION_REPO_IMPORT = "repo_import"
INTEGRATION_NATIVE_RUNNER = "native_runner"

METHOD_INTEGRATION_LEVELS = (
    INTEGRATION_MOCK,
    INTEGRATION_PAPER_ONLY,
    INTEGRATION_PAPER_FAITHFUL,
    INTEGRATION_REPO_IMPORT,
    INTEGRATION_NATIVE_RUNNER,
)
METHOD_INTEGRATION_RANK = {level: rank for rank, level in enumerate(METHOD_INTEGRATION_LEVELS)}


def canonical_integration_level(value: Any, *, allow_unknown: bool = False) -> str:
    """Return one canonical integration level or fail closed."""

    normalized = str(value or INTEGRATION_MOCK).strip().lower()
    if allow_unknown and normalized == "unknown":
        return normalized
    if normalized not in METHOD_INTEGRATION_RANK:
        raise ValueError(f"unknown method integration_level: {value!r}")
    return normalized


def normalize_action_contracts(value: Any) -> tuple[str, ...] | None:
    """Normalize an action-language declaration while preserving generic ``None``."""

    if value is None:
        return None
    if isinstance(value, str):
        contract = value.strip()
        return (contract,) if contract else ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(sorted({str(contract).strip() for contract in value if str(contract).strip()}))
    raise ValueError("supported_action_contracts must be null, a string, or a sequence")


@dataclass(frozen=True)
class MethodCapabilities:
    """Declared capability contract or explicitly untrusted runtime observation."""

    integration_level: str
    requires_benchmark: str = ""
    supported_action_contracts: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "integration_level",
            canonical_integration_level(self.integration_level, allow_unknown=True),
        )
        object.__setattr__(self, "requires_benchmark", str(self.requires_benchmark or ""))
        object.__setattr__(
            self,
            "supported_action_contracts",
            normalize_action_contracts(self.supported_action_contracts),
        )

    @classmethod
    def observe(cls, method: Any) -> MethodCapabilities:
        """Read a method instance/class for diagnostics, never for admission."""

        return cls(
            integration_level=getattr(method, "integration_level", INTEGRATION_MOCK),
            requires_benchmark=getattr(method, "requires_benchmark", "") or "",
            supported_action_contracts=getattr(
                method,
                "supported_action_contracts",
                None,
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MethodCapabilities:
        return cls(
            integration_level=value.get("integration_level", INTEGRATION_MOCK),
            requires_benchmark=value.get("requires_benchmark", "") or "",
            supported_action_contracts=value.get("supported_action_contracts"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
