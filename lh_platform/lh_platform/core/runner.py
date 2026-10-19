"""Stable public facade for synchronous experiment orchestration."""

from lh_platform.core._runner_metrics import aggregate_task_metrics
from lh_platform.core._runner_runtime import Runner

_aggregate = aggregate_task_metrics

__all__ = ["Runner"]
