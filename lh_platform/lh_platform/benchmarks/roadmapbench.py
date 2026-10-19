"""RoadmapBench benchmark adapter.

RoadmapBench: roadmap-level multi-step reasoning benchmark.
Repo: https://github.com/UniPat-AI/RoadmapBench
Local: a checkout of the repository under the benchmark data root.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from lh_platform.benchmarks._local import eval_benchmark_path, has_eval_benchmark, read_jsonl
from lh_platform.benchmarks.base import BenchmarkBase
from lh_platform.core.registry import BENCHMARKS
from lh_platform.core.types import Action, Observation, Task, TaskDomain


@BENCHMARKS.register("roadmapbench")
class RoadmapBenchBenchmark(BenchmarkBase):
    name = "roadmapbench"
    # Validity: data not yet downloaded from upstream; step() mock.
    eval_validity = "smoke_only"
    has_real_data = False
    has_native_harness = False
    supports_native_eval = False
    is_mock = True

    def __init__(self, max_steps: int = 25) -> None:
        self.max_steps = max_steps
        candidates: list[Path] = [
            eval_benchmark_path("roadmapbench") / "repo" / "data" / "test.jsonl",
            eval_benchmark_path("roadmapbench") / "repo" / "data" / "roadmapbench.jsonl",
            eval_benchmark_path("roadmapbench") / "repo" / "data" / "tasks.jsonl",
        ]
        self._data_path: Path | None = next((p for p in candidates if p.is_file()), None)
        if self._data_path is None and has_eval_benchmark("roadmapbench"):
            warnings.warn(
                "[roadmapbench] repo cloned but no canonical data file found; using synthetic",
                stacklevel=2,
            )

    def load(self, n: int = 3) -> list[Task]:
        if self._data_path is not None:
            rows = read_jsonl(self._data_path, limit=n)
            if rows:
                return [
                    Task(
                        task_id=str(r.get("id") or r.get("task_id") or f"roadmap_{i}"),
                        domain=TaskDomain.TOOL,
                        instruction=self._format(r),
                        max_steps=self.max_steps,
                        metadata={"row": r, "benchmark": "roadmapbench"},
                    )
                    for i, r in enumerate(rows)
                ]
        return self._synthetic(n)

    def _format(self, r: dict[str, Any]) -> str:
        if "task" in r:
            return str(r["task"])
        if "question" in r:
            return str(r["question"])
        if "instruction" in r:
            return str(r["instruction"])
        return json.dumps(r)[:300]

    def _synthetic(self, n: int) -> list[Task]:
        return [
            Task(
                task_id=f"roadmap_synth_{i}",
                domain=TaskDomain.TOOL,
                instruction=(
                    f"Synthetic RoadmapBench task #{i}: produce a 5-step learning roadmap to "
                    f"master distributed systems, with each step having a single-line prerequisite."
                ),
                max_steps=self.max_steps,
                metadata={"synthetic": True, "benchmark": "roadmapbench"},
            )
            for i in range(n)
        ]

    def step(self, task: Task, action: Action) -> Observation:
        return Observation(
            content=f"[roadmapbench mock] task={task.task_id} action={action.raw[:120]}",
            info={"done": False, "continue": True},
        )
