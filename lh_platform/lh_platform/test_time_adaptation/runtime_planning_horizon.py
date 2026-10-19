"""Planning-horizon cap for the runtime-owned option DP (no heavy dependencies).

`RuntimeOwnedOptionLiveResolver` computes a native horizon per decision boundary:
replan triggers plan one step, ordinary boundaries plan two. The
recovery-composition program needs a configurable cap so that the `Full - D`
ablation limits only how far each planning call expands -- never how many times
the closed loop may act afterwards.
"""

from __future__ import annotations

SUPPORTED_PLANNING_HORIZONS = (1, 2)


def effective_planning_horizon(native: int, cap: int | None) -> int:
    """Return the horizon a planning call may expand, capped when configured."""

    if native not in SUPPORTED_PLANNING_HORIZONS:
        raise ValueError("runtime-owned native horizon is unsupported")
    if cap is None:
        return native
    if cap not in SUPPORTED_PLANNING_HORIZONS:
        raise ValueError("runtime planning horizon cap is unsupported")
    return min(native, cap)
