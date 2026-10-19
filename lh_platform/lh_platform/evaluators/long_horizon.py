"""Long-horizon evaluator.

Computes the long-horizon specific metrics listed in the README.
All metrics are derived from a finished Trajectory (no extra LLM calls).

Metric definitions:
  success              : 1 if trajectory.success else 0
  steps                : number of steps in trajectory
  cost_to_success      : mean cost_usd per step on successful tasks; undefined otherwise
  cost_per_task        : total cost_usd across steps
  failure_step_index   : index of first ERROR or terminal is_error observation, -1 if success
  recovery_rate        : on tasks with errors, did we end with success=True; undefined otherwise
  planning_drift_score : 1 - mean cosine of action.raw token overlap with task.instruction tokens
                         (cheap proxy; lower = better retention)
  context_decay_score  : last/first positive prompt tokens among action-bearing steps
                         (proxy: how much context the method has accumulated)
  tool_error_rate      : count(obs.is_error=True) / steps
  error_compounding_score : given a window of K steps after the first error,
                            ratio of errors in that window vs the rest
  trajectory_efficiency : success / steps (0 if steps==0)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from lh_platform.core.registry import EVALUATORS
from lh_platform.core.types import Role, Step, Task, Trajectory
from lh_platform.evaluators.base import EvaluatorBase


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _cosine_proxy(a: str, b: str) -> float:
    """Cheap token-overlap proxy in [0, 1]."""
    ta = Counter(a.lower().split())
    tb = Counter(b.lower().split())
    if not ta or not tb:
        return 0.0
    dot = sum(ta[k] * tb.get(k, 0) for k in ta)
    na = (sum(v * v for v in ta.values())) ** 0.5
    nb = (sum(v * v for v in tb.values())) ** 0.5
    return _safe_div(dot, na * nb)


@EVALUATORS.register("long_horizon")
class LongHorizonEvaluator(EvaluatorBase):
    name = "long_horizon"
    ERROR_WINDOW = 5

    def score(self, trajectory: Trajectory, task: Task) -> dict[str, Any]:
        steps: list[Step] = list(trajectory.steps)
        n = len(steps)
        if n == 0:
            return {
                "success": 0.0,
                "steps": 0.0,
                "cost_to_success": None,
                "cost_per_task": 0.0,
                "failure_step_index": -1.0,
                "recovery_rate": None,
                "planning_drift_score": 0.0,
                "context_decay_score": None,
                "tool_error_count": 0.0,
                "tool_step_count": 0.0,
                "tool_error_rate": 0.0,
                "error_compounding_score": 0.0,
                "trajectory_efficiency": 0.0,
            }

        success = bool(trajectory.success)
        total_cost = sum(s.cost_usd for s in steps)

        # Failure step index: first observation that is_error=True, or first ERROR step
        first_err_idx = -1
        for s in steps:
            if s.role == Role.ERROR:
                first_err_idx = s.step_idx
                break
            if s.observation is not None and s.observation.is_error:
                first_err_idx = s.step_idx
                break
        if first_err_idx == -1 and not success:
            # No explicit error step but task didn't succeed: treat last step as failure
            first_err_idx = steps[-1].step_idx

        # Recovery is defined only when an explicit or terminal failure exists.
        recovery_rate = None if first_err_idx == -1 else (1.0 if success else 0.0)

        # Tool error rate
        err_count = sum(
            1
            for s in steps
            if (s.observation is not None and s.observation.is_error) or s.role == Role.ERROR
        )
        tool_error_rate = _safe_div(err_count, n)

        # Error compounding score
        if first_err_idx == -1:
            error_compounding = 0.0
        else:
            tail = [s for s in steps if s.step_idx > first_err_idx]
            head = [s for s in steps if s.step_idx <= first_err_idx]
            tail_errs = sum(
                1
                for s in tail
                if (s.observation is not None and s.observation.is_error) or s.role == Role.ERROR
            )
            head_errs = err_count - tail_errs
            tail_rate = _safe_div(tail_errs, len(tail))
            head_rate = _safe_div(head_errs, len(head))
            error_compounding = max(0.0, tail_rate - head_rate)

        # Planning drift score: 1 - mean cosine of action.raw with task.instruction
        sims: list[float] = []
        for s in steps:
            if s.action is not None and s.action.raw:
                sims.append(_cosine_proxy(s.action.raw, task.instruction))
        mean_sim = _safe_div(sum(sims), len(sims))
        planning_drift = 1.0 - mean_sim

        # Context accumulation is defined only over action-bearing model calls.
        action_prompt_tokens = [
            step.prompt_tokens
            for step in steps
            if step.action is not None and step.prompt_tokens > 0
        ]
        context_decay = (
            _safe_div(action_prompt_tokens[-1], action_prompt_tokens[0])
            if action_prompt_tokens
            else None
        )

        cost_to_success = _safe_div(total_cost, n) if success else None

        trajectory_efficiency = _safe_div(1.0 if success else 0.0, n)

        return {
            "success": 1.0 if success else 0.0,
            "steps": float(n),
            "cost_to_success": cost_to_success,
            "cost_per_task": total_cost,
            "failure_step_index": float(first_err_idx),
            "recovery_rate": recovery_rate,
            "planning_drift_score": planning_drift,
            "context_decay_score": context_decay,
            "tool_error_count": float(err_count),
            "tool_step_count": float(n),
            "tool_error_rate": tool_error_rate,
            "error_compounding_score": error_compounding,
            "trajectory_efficiency": trajectory_efficiency,
        }
