"""Id → DecompositionProvider adapter registry.

Credit-method factories stay in ``credit_method_adapters`` so credit assigner
registration does not pull ALFWorld/AppWorld annotate implementations.
"""

from __future__ import annotations

from lh_platform.training.agentic_rl_components.decomposition_adapters import (
    AlfWorldFactDecomposition,
    AppWorldEvaluatorDecomposition,
    NoDecomposition,
)
from lh_platform.training.agentic_rl_components.errors import AgenticRLComponentError


def build_decomposition_provider(decomposition_id: str):
    providers = {
        "none": NoDecomposition,
        "alfworld_fact_phases": AlfWorldFactDecomposition,
        "appworld_official_evaluator_goals": AppWorldEvaluatorDecomposition,
    }
    try:
        return providers[decomposition_id]()
    except KeyError as error:
        raise AgenticRLComponentError(
            f"unsupported decomposition provider {decomposition_id!r}"
        ) from error
