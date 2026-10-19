"""Cost evaluator: aggregates tokens and dollars."""

from __future__ import annotations

from typing import Any

from lh_platform.core.registry import EVALUATORS
from lh_platform.core.types import Task, Trajectory
from lh_platform.evaluators.base import EvaluatorBase


@EVALUATORS.register("cost")
class CostEvaluator(EvaluatorBase):
    name = "cost"

    def score(self, trajectory: Trajectory, task: Task) -> dict[str, Any]:
        total_prompt = sum(s.prompt_tokens for s in trajectory.steps)
        total_completion = sum(s.completion_tokens for s in trajectory.steps)
        total_cost = sum(s.cost_usd for s in trajectory.steps)
        return {
            "total_prompt_tokens": float(total_prompt),
            "total_completion_tokens": float(total_completion),
            "total_cost_usd": float(total_cost),
            "steps": float(len(trajectory.steps)),
        }
