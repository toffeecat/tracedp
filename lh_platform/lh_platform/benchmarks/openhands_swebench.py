"""OpenHands SWE-bench style benchmark adapter.

OpenHands is more of a platform than a benchmark. A local clone is a
large monorepo. For the MVP we
expose a thin adapter that pretends to dispatch SWE-bench-style tasks
to OpenHands via a CLI invocation; if OpenHands is not configured
(docker, model keys, etc.) the step is a mock.

Even in this smoke-only adapter, a SWE-bench JSONL can contain gold patches
and hidden tests.  ``Task.metadata`` is agent-visible, so raw dataset rows
must not be passed through it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from lh_platform.benchmarks._local import (
    eval_benchmark_path,
    has_eval_benchmark,
    read_jsonl,
)
from lh_platform.benchmarks.base import BenchmarkBase
from lh_platform.core.registry import BENCHMARKS
from lh_platform.core.types import Action, Observation, Task, TaskDomain


@BENCHMARKS.register("openhands_swebench")
class OpenHandsSWEBenchBenchmark(BenchmarkBase):
    name = "openhands_swebench"
    # Validity: real jsonl missing, step() always mock, no harness invoked.
    eval_validity = "smoke_only"
    has_real_data = False
    has_native_harness = False
    supports_native_eval = False
    is_mock = True

    def __init__(self, max_steps: int = 30) -> None:
        self.max_steps = max_steps
        # SWE-bench Lite style task list (synthesized if real one isn't present)
        self._candidates: list[Path] = [
            eval_benchmark_path("OpenHands") / "swe-bench-lite.jsonl",
        ]
        self._data_path: Path | None = next((p for p in self._candidates if p.is_file()), None)

    def load(self, n: int = 3) -> list[Task]:
        if self._data_path is not None:
            rows = read_jsonl(self._data_path, limit=n)
            if rows:
                return [
                    Task(
                        task_id=str(r.get("instance_id") or r.get("id") or f"swebench_{i}"),
                        domain=TaskDomain.SWE,
                        instruction=(
                            r.get("problem_statement")
                            or r.get("issue")
                            or r.get("instruction")
                            or f"SWE-bench style task {r.get('instance_id') or r.get('id') or i}"
                        ),
                        max_steps=self.max_steps,
                        metadata={
                            "instance_id": str(
                                r.get("instance_id") or r.get("id") or f"swebench_{i}"
                            ),
                            "data_source": "openhands_local_jsonl",
                        },
                    )
                    for i, r in enumerate(rows)
                ]
        return self._synthetic(n)

    def _synthetic(self, n: int) -> list[Task]:
        return [
            Task(
                task_id=f"swebench_synth_{i}",
                domain=TaskDomain.SWE,
                instruction=(
                    f"Synthetic SWE task #{i}: fix a bug in a small Python library; "
                    f"the failing test is in `tests/test_fix_{i}.py`."
                ),
                max_steps=self.max_steps,
            )
            for i in range(n)
        ]

    def step(self, task: Task, action: Action) -> Observation:
        if shutil.which("openhands") is not None or has_eval_benchmark("OpenHands"):
            # Real adapter would shell out: `openhands --task-id <id> ...`
            return Observation(
                content=f"[openhands mock] task={task.task_id} action={action.raw[:80]}",
                info={"done": False, "continue": True, "cli": "openhands"},
            )
        return Observation(
            content=f"[openhands mock] task={task.task_id} action={action.raw[:80]}",
            info={"done": False, "continue": True},
        )
