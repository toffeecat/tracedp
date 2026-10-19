"""Task-level execution records and failure semantics.

The Runner still owns orchestration and evidence composition.  This module
defines the narrow task-execution boundary: one task produces one trajectory,
one declared method capability, and one observed model-runtime receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lh_platform.core.storage import RunStorage
from lh_platform.core.types import Role, Step, Task, Trajectory


@dataclass(frozen=True)
class TaskExecutionRecord:
    """Result of one task loop before outcome evidence and scoring."""

    task: Task
    trajectory: Trajectory
    method_integration_level: str
    observed_model_runtime: dict[str, Any] = field(default_factory=dict)


def record_execution_failure(
    trajectory: Trajectory,
    *,
    stage: str,
    error: BaseException,
) -> None:
    """Attach one immutable-shaped failure marker to a task trajectory."""

    trajectory.info["execution_failure"] = {
        "stage": str(stage),
        "error_type": type(error).__name__,
        "message": str(error),
    }


def close_failed_task_trajectory(
    *,
    trajectory: Trajectory,
    storage: RunStorage,
    task: Task,
    error: BaseException,
) -> None:
    """Ensure every runner-level failure has one terminal persisted state."""

    if trajectory.finished_at is not None:
        return
    record_execution_failure(trajectory, stage="runner", error=error)
    step = Step(
        step_idx=len(trajectory.steps),
        role=Role.ERROR,
        content=f"runner task failed: {error}",
        info={"stage": "runner", "task_id": task.task_id},
    )
    trajectory.append(step)
    storage.append_step_task(step.to_dict(), task.task_id)
    trajectory.close(success=False)
