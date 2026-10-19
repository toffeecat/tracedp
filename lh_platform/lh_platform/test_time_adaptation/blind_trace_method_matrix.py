"""Shared helpers for a same-case blind tracer method matrix."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from lh_platform.test_time_adaptation.blind_full_trace_localization import (
    BlindTraceRanking,
)

T = TypeVar("T")


def build_deterministic_baseline_ranking(
    event_ids: Sequence[str],
    *,
    method_id: str,
) -> BlindTraceRanking:
    """Rank up to three visible candidates without evaluator information."""

    if not event_ids:
        raise ValueError("blind trace baseline requires at least one event")
    ranking_count = min(3, len(event_ids))
    if method_id == "recent_step":
        ranked = tuple(reversed(event_ids[-ranking_count:]))
        explanation = "Most recent transitions before the terminal failure."
    elif method_id == "uniform_positions":
        if len(event_ids) <= 3:
            ranked = tuple(event_ids)
        else:
            ranked = tuple(
                event_ids[((index + 1) * len(event_ids) // 4) - 1]
                for index in range(3)
            )
        explanation = "Fixed positions spread across the visible candidate slate."
    else:
        raise ValueError(f"unknown deterministic baseline: {method_id}")
    return BlindTraceRanking(ranked_event_ids=ranked, explanation=explanation)


def partition_jobs(jobs: Sequence[T], *, worker_count: int) -> tuple[tuple[T, ...], ...]:
    """Distribute complete cases across workers without splitting a case."""

    if worker_count < 1:
        raise ValueError("worker count must be positive")
    partitions: list[list[T]] = [[] for _ in range(min(worker_count, len(jobs)))]
    for index, job in enumerate(jobs):
        partitions[index % len(partitions)].append(job)
    return tuple(tuple(partition) for partition in partitions)


__all__ = ["build_deterministic_baseline_ranking", "partition_jobs"]
