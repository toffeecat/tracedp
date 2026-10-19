"""Success evaluator: just trusts trajectory.success.

This is the simplest evaluator and is the default for sanity checks.
"""

from __future__ import annotations

from typing import Any

from lh_platform.core.registry import EVALUATORS
from lh_platform.core.types import Task, Trajectory
from lh_platform.evaluators.base import EvaluatorBase


@EVALUATORS.register("success")
class SuccessEvaluator(EvaluatorBase):
    name = "success"

    def score(self, trajectory: Trajectory, task: Task) -> dict[str, Any]:
        return {
            "success": 1.0 if trajectory.success else 0.0,
            "steps": float(len(trajectory.steps)),
        }
