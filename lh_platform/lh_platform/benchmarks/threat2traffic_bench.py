"""Threat2Traffic benchmark adapter.

Threat2Traffic: cybersecurity benchmark for multi-agent malware-traffic generation.
Repo: https://github.com/apos3637/Threat2Traffic
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


@BENCHMARKS.register("threat2traffic_bench")
class Threat2TrafficBenchmark(BenchmarkBase):
    name = "threat2traffic_bench"
    # Validity: data not yet downloaded from upstream; step() mock.
    eval_validity = "smoke_only"
    has_real_data = False
    has_native_harness = False
    supports_native_eval = False
    is_mock = True

    def __init__(self, max_steps: int = 15) -> None:
        self.max_steps = max_steps
        candidates: list[Path] = [
            eval_benchmark_path("threat2traffic") / "repo" / "data" / "test.jsonl",
            eval_benchmark_path("threat2traffic") / "repo" / "data" / "samples.jsonl",
            eval_benchmark_path("threat2traffic") / "repo" / "data" / "tasks.jsonl",
        ]
        self._data_path: Path | None = next((p for p in candidates if p.is_file()), None)
        if self._data_path is None and has_eval_benchmark("threat2traffic"):
            warnings.warn(
                "[threat2traffic_bench] repo cloned but no canonical data file found; using synthetic",
                stacklevel=2,
            )

    def load(self, n: int = 3) -> list[Task]:
        if self._data_path is not None:
            rows = read_jsonl(self._data_path, limit=n)
            if rows:
                return [
                    Task(
                        task_id=str(r.get("id") or r.get("sample_id") or f"threat_{i}"),
                        domain=TaskDomain.SWE,
                        instruction=self._format(r),
                        max_steps=self.max_steps,
                        metadata={"row": r, "benchmark": "threat2traffic"},
                    )
                    for i, r in enumerate(rows)
                ]
        return self._synthetic(n)

    def _format(self, r: dict[str, Any]) -> str:
        if "task" in r:
            return str(r["task"])
        if "description" in r:
            return str(r["description"])
        if "instruction" in r:
            return str(r["instruction"])
        return json.dumps(r)[:300]

    def _synthetic(self, n: int) -> list[Task]:
        return [
            Task(
                task_id=f"threat_synth_{i}",
                domain=TaskDomain.SWE,
                instruction=(
                    f"Synthetic Threat2Traffic task #{i}: given a threat-intel document, "
                    f"construct a sandboxed environment and capture malware traffic for analysis."
                ),
                max_steps=self.max_steps,
                metadata={"synthetic": True, "benchmark": "threat2traffic"},
            )
            for i in range(n)
        ]

    def step(self, task: Task, action: Action) -> Observation:
        return Observation(
            content=f"[threat2traffic_bench mock] task={task.task_id} action={action.raw[:120]}",
            info={"done": False, "continue": True},
        )
