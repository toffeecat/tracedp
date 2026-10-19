"""Runner-owned metric namespaces and aggregation."""

from __future__ import annotations

import math
from statistics import stdev
from typing import Any

from lh_platform.core.types import Task, Trajectory

_RUNNER_OWNED_TASK_METRIC_KEYS = frozenset(
    {
        "task_id",
        "steps",
        "success",
        "diagnostic_outcome_success",
        "seed",
        "benchmark_eval_validity",
        "method_integration_level",
        "method_capability_receipt",
        "outcome_evidence",
        "run_credibility",
        "model_runtime",
    }
)

# Canonical task-level quantities whose arithmetic mean has a stable meaning.
# Ratios requiring pooled denominators are handled separately below.
_MEAN_TASK_METRIC_KEYS = frozenset(
    {
        "steps",
        "cost_to_success",
        "cost_per_task",
        "failure_step_index",
        "recovery_rate",
        "planning_drift_score",
        "context_decay_score",
        "error_compounding_score",
        "trajectory_efficiency",
    }
)


def benchmark_task_metrics(
    benchmark: Any, task: Task, trajectory: Trajectory
) -> tuple[dict[str, Any], list[str]]:
    """Read benchmark-owned metrics without letting adapters rewrite run truth.

    A native environment often exposes a primary score that the generic
    long-horizon evaluator cannot infer from text observations. Keep those
    values under explicit ``environment_*`` / ``benchmark_*`` namespaces and
    reject all canonical metric names.
    """

    producer = getattr(benchmark, "task_metrics", None)
    if producer is None:
        return {}, []
    try:
        raw = producer(task, trajectory)
    except Exception as error:
        return {}, [f"task_metrics failed: {type(error).__name__}: {error}"]
    if not isinstance(raw, dict):
        return {}, ["task_metrics must return a mapping"]
    accepted: dict[str, Any] = {}
    rejected: list[str] = []
    for key, value in raw.items():
        if not isinstance(key, str):
            rejected.append(repr(key))
            continue
        if key in _RUNNER_OWNED_TASK_METRIC_KEYS or not key.startswith(
            ("environment_", "benchmark_")
        ):
            rejected.append(key)
            continue
        accepted[key] = value
    return accepted, rejected


def _finite_numbers(per_task: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for metrics in per_task:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _add_summary(out: dict[str, Any], key: str, values: list[float]) -> None:
    if not values:
        return
    out[f"mean_{key}"] = sum(values) / len(values)
    out[f"n_{key}"] = len(values)
    out[f"std_{key}"] = stdev(values) if len(values) > 1 else 0.0


def aggregate_task_metrics(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only declared metrics and preserve every denominator.

    Undefined task metrics are represented by ``None`` and excluded with an
    explicit ``n_*``. ``tool_error_rate`` is pooled from raw counts rather than
    averaging per-task ratios. Metadata such as ``seed`` is never averaged.
    """

    if not per_task:
        return {}
    out: dict[str, Any] = {"task_count": len(per_task)}
    keys = set().union(*(metrics.keys() for metrics in per_task))
    aggregatable = set(_MEAN_TASK_METRIC_KEYS)
    aggregatable.update(key for key in keys if key.startswith(("environment_", "benchmark_")))
    for key in sorted(aggregatable):
        _add_summary(out, key, _finite_numbers(per_task, key))

    error_counts = _finite_numbers(per_task, "tool_error_count")
    step_counts = _finite_numbers(per_task, "tool_step_count")
    if len(error_counts) == len(per_task) == len(step_counts) and sum(step_counts) > 0:
        task_rates = _finite_numbers(per_task, "tool_error_rate")
        out["mean_tool_error_rate"] = sum(error_counts) / sum(step_counts)
        out["n_tool_error_rate"] = len(task_rates)
        out["std_tool_error_rate"] = stdev(task_rates) if len(task_rates) > 1 else 0.0
        out["total_tool_errors"] = sum(error_counts)
        out["total_tool_steps"] = sum(step_counts)

    successes = [1.0 if bool(metrics.get("success")) else 0.0 for metrics in per_task]
    success_rate = sum(successes) / len(successes)
    out["success_rate"] = success_rate
    out["mean_success"] = success_rate
    out["n_success"] = len(successes)
    out["std_success"] = stdev(successes) if len(successes) > 1 else 0.0
    return out
