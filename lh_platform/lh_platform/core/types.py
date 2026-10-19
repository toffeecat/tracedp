"""Core type definitions for the unified trajectory format."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    OBSERVATION = "observation"
    ERROR = "error"


class TaskDomain(StrEnum):
    WEB = "web"
    TOOL = "tool"
    SWE = "swe"
    GUI = "gui"
    SCIENCE = "science"
    UNKNOWN = "unknown"


@dataclass
class Task:
    """A single benchmark task."""

    task_id: str
    domain: TaskDomain
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 30
    timeout_s: float = 600.0
    expected_outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["domain"] = self.domain.value
        return d


@dataclass
class Action:
    """Action emitted by a method."""

    action_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    raw: str = ""
    action_type: str = "text"
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    info: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    """Observation / tool result returned by the environment."""

    obs_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    is_error: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Step:
    """A single step in a trajectory."""

    step_idx: int
    role: Role
    content: str
    action: Action | None = None
    observation: Observation | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "step_idx": self.step_idx,
            "role": self.role.value,
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "info": self.info,
            "t": self.t,
        }
        if self.action is not None:
            d["action"] = self.action.to_dict()
        if self.observation is not None:
            d["observation"] = self.observation.to_dict()
        return d


@dataclass
class Trajectory:
    """The full trajectory of a single task execution."""

    task_id: str
    method: str
    model: str
    benchmark: str
    steps: list[Step] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    success: bool = False
    final_answer: str | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def append(self, step: Step) -> None:
        self.steps.append(step)

    def close(self, success: bool, final_answer: str | None = None) -> None:
        self.finished_at = time.time()
        self.success = success
        if final_answer is not None:
            self.final_answer = final_answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "method": self.method,
            "model": self.model,
            "benchmark": self.benchmark,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success": self.success,
            "final_answer": self.final_answer,
            "n_steps": len(self.steps),
            "info": self.info,
            "steps": [s.to_dict() for s in self.steps],
        }
