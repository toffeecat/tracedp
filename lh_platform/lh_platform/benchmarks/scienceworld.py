"""ScienceWorld smoke-only adapter.

This adapter wraps ScienceWorld (Shridhar et al., 2020), a simulator for
interactive scientific reasoning tasks. Each task presents an environment
where the agent must investigate hypotheses by performing experiments.

Action space:
  look, look <object>, open <object>, close <object>, put <object> in/on <receptacle>,
  pick up <object>, use <object>, use <object> with <object>, focus <object>,
  move <object> to <receptacle>, answer(<text>), done

This adapter is a smoke fixture: it has no bundled ScienceWorld task manifest
and its fallback transition system is not the official simulator. A reachable
HTTP endpoint alone is not sufficient evidence of native compatibility, so the
adapter never promotes itself to ``real_native``. Full integration must use the
pinned official Python API, its task manifest, and the official evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lh_platform.benchmarks.base import EVAL_SMOKE, BenchmarkBase
from lh_platform.core.registry import BENCHMARKS
from lh_platform.core.types import Action, Observation, Task, TaskDomain

# Fallback manifest for when ScienceWorld is not installed
_MANIFEST_PATH = Path(__file__).parent.parent.parent.parent / "data" / "scienceworld_manifest.json"


def _load_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or _MANIFEST_PATH
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


@BENCHMARKS.register("scienceworld")
class ScienceWorldBenchmark(BenchmarkBase):
    name = "scienceworld"
    eval_validity = EVAL_SMOKE
    has_real_data = False
    has_native_harness = False
    supports_native_eval = False
    is_mock = True

    VALID_ACTIONS = {
        "look",
        "open",
        "close",
        "put",
        "pick up",
        "use",
        "focus",
        "move",
        "answer",
        "done",
    }

    def __init__(self, max_steps: int = 30, server_url: str | None = None) -> None:
        self.max_steps = max_steps
        self._server_url = server_url
        self._manifest = _load_manifest()
        self._connected = False
        self._last_summary: dict[str, Any] = {}

        # Try connecting to ScienceWorld server
        if server_url:
            self._try_connect(server_url)

    def _try_connect(self, url: str) -> None:
        """Attempt to connect to a running ScienceWorld server."""
        try:
            import urllib.request

            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    self._connected = True
                    self._server_url = url
        except Exception:
            self._connected = False

    def load(self, n: int = 3) -> list[Task]:
        if self._connected:
            return self._load_from_server(n)
        return self._load_from_manifest(n)

    def _load_from_server(self, n: int) -> list[Task]:
        """Load tasks from a running ScienceWorld server."""
        try:
            import json as _json
            import urllib.request

            req = urllib.request.Request(f"{self._server_url}/tasks?limit={n}", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
            tasks = []
            for item in data.get("tasks", [])[:n]:
                tasks.append(
                    Task(
                        task_id=item.get("id", f"sw_{len(tasks)}"),
                        domain=TaskDomain.UNKNOWN,
                        instruction=item.get("instruction", ""),
                        max_steps=self.max_steps,
                        metadata={
                            "task_name": item.get("name", ""),
                            "variant": item.get("variant", 0),
                            "room": item.get("room", ""),
                        },
                    )
                )
            return tasks
        except Exception:
            return self._load_from_manifest(n)

    def _load_from_manifest(self, n: int) -> list[Task]:
        """Load tasks from local manifest file."""
        if not self._manifest:
            return self._fallback_tasks(n)

        tasks = []
        for item in self._manifest[:n]:
            tasks.append(
                Task(
                    task_id=item.get("task_id", f"sw_{len(tasks)}"),
                    domain=TaskDomain.UNKNOWN,
                    instruction=item.get("instruction", ""),
                    max_steps=self.max_steps,
                    metadata={
                        "task_name": item.get("task_name", ""),
                        "variant": item.get("variant", 0),
                        "room": item.get("room", ""),
                        "template": item.get("template", ""),
                    },
                )
            )
        return tasks

    def _fallback_tasks(self, n: int) -> list[Task]:
        """Hardcoded fallback tasks for smoke testing."""
        fallback = [
            {
                "task_id": "sw_heat_water",
                "instruction": "Heat water in the test tube until it boils. Report the temperature.",
                "task_name": "heat_test_tube",
                "room": "kitchen",
            },
            {
                "task_id": "sw_measure_volume",
                "instruction": "Measure 250 mL of water using the graduated cylinder.",
                "task_name": "measure_volume",
                "room": "kitchen",
            },
            {
                "task_id": "sw_mix_solutes",
                "instruction": "Dissolve 5g of salt in water and observe the result.",
                "task_name": "mix_substances",
                "room": "kitchen",
            },
        ]
        tasks = []
        for item in fallback[:n]:
            tasks.append(
                Task(
                    task_id=item["task_id"],
                    domain=TaskDomain.UNKNOWN,
                    instruction=item["instruction"],
                    max_steps=self.max_steps,
                    metadata={"task_name": item["task_name"], "room": item["room"]},
                )
            )
        return tasks

    def step(self, task: Task, action: Action) -> Observation:
        if self._connected and self._server_url:
            return self._step_server(task, action)
        return self._step_mock(task, action)

    def _step_server(self, task: Task, action: Action) -> Observation:
        """Send action to ScienceWorld server."""
        try:
            import json as _json
            import urllib.request

            payload = _json.dumps(
                {
                    "task_id": task.task_id,
                    "action": action.raw,
                }
            ).encode()
            req = urllib.request.Request(
                f"{self._server_url}/step",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = _json.loads(resp.read())

            done = result.get("done", False)
            score = result.get("score", 0.0)
            obs_text = result.get("observation", "")

            return Observation(
                content=obs_text[:2000],
                info={
                    "done": done,
                    "success": done and score >= 1.0,
                    "score": score,
                    "continue": not done,
                },
            )
        except Exception as e:
            return Observation(
                content=f"ScienceWorld server error: {e}",
                info={"done": False, "continue": True},
                is_error=True,
            )

    def _step_mock(self, task: Task, action: Action) -> Observation:
        """Simplified text simulation for smoke testing."""
        raw = (action.raw or "").strip()
        raw_lower = raw.lower()

        # Parse action
        if raw_lower.startswith("done") or raw_lower.startswith("stop"):
            return Observation(
                content="Task completed.",
                info={"done": True, "answer": raw, "success": False, "continue": False},
            )

        if raw_lower.startswith("answer("):
            answer = raw[7:-1] if raw.endswith(")") else raw[7:]
            return Observation(
                content=f"Answer submitted: {answer}",
                # The fallback manifest has no answer key or official evaluator.
                # Treating any terminal answer as a success fabricates a benchmark
                # result, so the smoke adapter must fail closed.
                info={
                    "done": True,
                    "answer": answer,
                    "success": False,
                    "continue": False,
                    "unscored_answer": True,
                    "evaluation_warning": "ScienceWorld smoke fallback has no answer key",
                },
            )

        # Simulate basic actions
        if raw_lower.startswith("look"):
            return Observation(
                content=(
                    f"Room: {task.metadata.get('room', 'kitchen')}. "
                    "You see: a test tube, a graduated cylinder, a beaker, "
                    "a Bunsen burner, salt, water, and a thermometer."
                ),
                info={"done": False, "continue": True},
            )

        if raw_lower.startswith("use"):
            return Observation(
                content="You perform the action. The environment responds.",
                info={"done": False, "continue": True},
            )

        if raw_lower.startswith("pick up") or raw_lower.startswith("take"):
            return Observation(
                content="You pick up the object.",
                info={"done": False, "continue": True},
            )

        if raw_lower.startswith("put") or raw_lower.startswith("place"):
            return Observation(
                content="You place the object.",
                info={"done": False, "continue": True},
            )

        if raw_lower.startswith("open"):
            return Observation(
                content="You open the container.",
                info={"done": False, "continue": True},
            )

        if raw_lower.startswith("focus"):
            return Observation(
                content="You focus your attention on the object.",
                info={"done": False, "continue": True},
            )

        # Unknown action
        return Observation(
            content=f"Action not recognized: {raw[:100]}",
            info={"done": False, "continue": True, "parse_error": True},
            is_error=True,
        )

    def close(self, artifacts_dir: Path | None = None) -> None:
        self._last_summary = {
            "connected": self._connected,
            "server_url": self._server_url,
            "is_mock": self.is_mock,
            "n_manifest_tasks": len(self._manifest),
        }

    @property
    def last_summary(self) -> dict[str, Any]:
        return dict(self._last_summary)
