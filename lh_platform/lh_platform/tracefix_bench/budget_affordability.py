"""Budget affordability gate for pool-consuming actions.

The shared input pool admits a dispatch when the settled plus pending uncached input plus
the request's full-context estimate fits (``shared_budget.reserve``). A targeted repair
spends pool input too, so it can use up the room the Actor needs for its next call; a
refused Actor call ends the run with score 0 (contract: exhaustion is never retried).

This module only decides whether a pool-consuming action may be chosen. It does not
change scoring, and it never submits anything on its own: under the kept contract the
gate is a constraint on selection, not a new way to end a run.

Units are input tokens throughout, matching the ledger.
"""

from __future__ import annotations

# Safety margin on the next-reservation projection. Re-estimated on 204 managed runs
# (1,277 steps, interleaved isolated sessions excluded): 95th percentile of underprediction
# 1,007 tokens, 99th 2,864, largest 4,013. The margin must cover the largest underprediction,
# so it is set above it; the earlier 2,000 came from one run and left the tail uncovered.
PROJECTION_MARGIN_TOKENS = 5_000

# Targeted repair uncached input: 90th percentile of measured isolated repair sessions.
REPAIR_INPUT_P90_TOKENS = 13_111


def next_reservation_estimate(
    last_reservation_tokens: int, visible_chars: int, last_output_tokens: int
) -> int:
    """Projected full-context estimate of the Actor's next dispatch.

    The next request is the last request plus what the last exchange appended: the
    Actor's output and the visible tool input and result (about four characters per
    token, the same rule as the proxy's ``len(body) // 4``).
    """

    for name, value in (
        ("last_reservation_tokens", last_reservation_tokens),
        ("visible_chars", visible_chars),
        ("last_output_tokens", last_output_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return last_reservation_tokens + visible_chars // 4 + last_output_tokens


def continuation_feasible(room: int, next_reservation_tokens: int) -> bool:
    """Whether the Actor's next dispatch fits the remaining input room (the ledger's test)."""

    return room >= next_reservation_tokens


def repair_admissible(
    room: int,
    next_reservation_tokens: int,
    *,
    repair_input_tokens: int = REPAIR_INPUT_P90_TOKENS,
    margin_tokens: int = PROJECTION_MARGIN_TOKENS,
) -> bool:
    """A targeted repair may be chosen only if the Actor's next call still fits afterwards.

    Checks take no pool input and are never gated here.
    """

    return room - repair_input_tokens >= next_reservation_tokens + margin_tokens


__all__ = [
    "PROJECTION_MARGIN_TOKENS",
    "REPAIR_INPUT_P90_TOKENS",
    "continuation_feasible",
    "next_reservation_estimate",
    "repair_admissible",
]
