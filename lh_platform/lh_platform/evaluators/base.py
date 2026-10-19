"""Evaluator base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lh_platform.core.types import Task, Trajectory


class EvaluatorBase(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, trajectory: Trajectory, task: Task) -> dict[str, Any]: ...
