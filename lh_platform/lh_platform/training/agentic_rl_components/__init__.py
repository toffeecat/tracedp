"""Built-in agentic-RL adapters split along runtime Protocol seams.

Owns DecompositionProvider and RLMethod adapters plus id registries.
Protocol interfaces live in ``agentic_rl_runtime``; this package supplies
adapters at those seams.

Layout:

- ``decomposition_adapters`` — annotate / GoalGraph receipts
- ``credit_method_adapters`` — prepare_update / clipped-PG credit arms
- ``registry`` — decomposition id → adapter
- ``identity`` — ``builtin_component`` ComponentRef helper
"""

from __future__ import annotations

from lh_platform.training.agentic_rl_components.credit_method_adapters import (
    EstimatorG2POMethod,
    UnifiedG2POMethod,
    build_g2po_family_method,
)
from lh_platform.training.agentic_rl_components.decomposition_adapters import (
    ALFWORLD_DECOMPOSITION_RESERVED_FIELDS,
    AlfWorldFactDecomposition,
    AppWorldEvaluatorDecomposition,
    NoDecomposition,
)
from lh_platform.training.agentic_rl_components.errors import AgenticRLComponentError
from lh_platform.training.agentic_rl_components.identity import builtin_component
from lh_platform.training.agentic_rl_components.registry import (
    build_decomposition_provider,
)

__all__ = [
    "ALFWORLD_DECOMPOSITION_RESERVED_FIELDS",
    "AgenticRLComponentError",
    "AlfWorldFactDecomposition",
    "AppWorldEvaluatorDecomposition",
    "EstimatorG2POMethod",
    "NoDecomposition",
    "UnifiedG2POMethod",
    "build_decomposition_provider",
    "build_g2po_family_method",
    "builtin_component",
]
