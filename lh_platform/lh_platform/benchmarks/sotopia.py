"""Sotopia benchmark adapter.

Sotopia: social multi-agent benchmark (ICLR 2024 Spotlight).
Repo: https://github.com/sotopia-lab/sotopia
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


@BENCHMARKS.register("sotopia")
class SotopiaBenchmark(BenchmarkBase):
    name = "sotopia"
    # Validity: sotopia scenarios typically come from sotopia_database; step() mock.
    eval_validity = "smoke_only"
    has_real_data = False
    has_native_harness = False
    supports_native_eval = False
    is_mock = True

    def __init__(self, max_steps: int = 20) -> None:
        self.max_steps = max_steps
        candidates: list[Path] = [
            eval_benchmark_path("sotopia") / "repo" / "data" / "sotopia_data.jsonl",
            eval_benchmark_path("sotopia") / "repo" / "data" / "test.jsonl",
            eval_benchmark_path("sotopia") / "repo" / "data" / "scenarios.jsonl",
        ]
        self._data_path: Path | None = next((p for p in candidates if p.is_file()), None)
        if self._data_path is None and has_eval_benchmark("sotopia"):
            warnings.warn(
                "[sotopia] repo cloned but no canonical data file found; using synthetic",
                stacklevel=2,
            )

    def load(self, n: int = 3) -> list[Task]:
        if self._data_path is not None:
            rows = read_jsonl(self._data_path, limit=n)
            if rows:
                return [
                    Task(
                        task_id=str(r.get("id") or r.get("scenario_id") or f"sotopia_{i}"),
                        domain=TaskDomain.GUI,
                        instruction=self._format(r),
                        max_steps=self.max_steps,
                        metadata={"row": r, "benchmark": "sotopia"},
                    )
                    for i, r in enumerate(rows)
                ]
        return self._synthetic(n)

    def _format(self, r: dict[str, Any]) -> str:
        if "scenario" in r:
            return str(r["scenario"])
        if "scenario_text" in r:
            return str(r["scenario_text"])
        return json.dumps(r)[:300]

    def _synthetic(self, n: int) -> list[Task]:
        return [
            Task(
                task_id=f"sotopia_synth_{i}",
                domain=TaskDomain.GUI,
                instruction=(
                    f"Synthetic sotopia social task #{i}: two agents meet in a coffee shop; "
                    f"agent A wants a refund, agent B is the barista. Negotiate a resolution."
                ),
                max_steps=self.max_steps,
                metadata={"synthetic": True, "benchmark": "sotopia"},
            )
            for i in range(n)
        ]

    def step(self, task: Task, action: Action) -> Observation:
        return Observation(
            content=f"[sotopia mock] task={task.task_id} action={action.raw[:120]}",
            info={"done": False, "continue": True, "social_metrics": None},
        )
