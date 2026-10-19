"""Terminal-Bench adapter for aliasing research.

Terminal-Bench (ICLR 2026) is a Docker-based CLI benchmark where agents solve
realistic command-line tasks.  Unlike ALFWorld/TextWorld, there is NO admissible
action list — the agent issues arbitrary bash commands.  This makes it ideal for
studying temporal aliasing: the same terminal output can appear at different
steps while the underlying filesystem state has changed.

Validity levels:
  - Without a Terminal-Bench clone: fixture_only (synthetic manifest)
  - With a valid clone + Docker: real_native (upstream harness)
"""

from __future__ import annotations

import json
import os
import re as _re
import subprocess
import tomllib
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lh_platform.benchmarks.base import (
    EVAL_FIXTURE,
    EVAL_REAL_NATIVE,
    BenchmarkBase,
)
from lh_platform.core.registry import BENCHMARKS
from lh_platform.core.types import Action, Observation, Task, TaskDomain

_NORMALIZE_PATTERNS: list[tuple[_re.Pattern[str], str]] = [
    (_re.compile(r"/[\w./-]+"), "<PATH>"),
    (_re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\dZ]*\b"), "<TS>"),
    (_re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (_re.compile(r"\b[0-9a-f]{8,}\b"), "<HASH>"),
    # Timing/size values: 0.42s, 15ms, 18.3 MB, 5.2 MB/s
    (_re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|min|h|KB|MB|GB|TB)(?:/s)?\b"), "<N>"),
    # Line:col in compiler errors (:42:5: -> :<N>:<N>:)
    (_re.compile(r":\d+:\d+:"), ":<N>:<N>:"),
    # File:line after path normalization (<PATH>:45: -> <PATH>:<N>:)
    (_re.compile(r"(?<=<PATH>):\d+:"), ":<N>:"),
    # File:line at end or before non-digit (<PATH>:45 -> <PATH>:<N>)
    (_re.compile(r"(?<=<PATH>):\d+(?=\D|$)"), ":<N>"),
    # Line numbers in source display: "   42 |" or "line 42"
    (_re.compile(r"(?<=line )\d+"), "<N>"),
    (_re.compile(r"(?m)^\s+\d+(?=\s*\|)"), "   <N>"),
    # "N failed/passed" counts in test output
    (_re.compile(r"\b\d+(?= (?:failed|passed|error|warning))"), "<N>"),
    # "in N.Ns" timing at end of test output
    (_re.compile(r"(?<=in )\d+(?:\.\d+)?(?=s\b)"), "<N>"),
]


def _normalize_observation(text: str) -> str:
    """Replace paths, numbers, timestamps with placeholders for fuzzy matching."""
    for pattern, replacement in _NORMALIZE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# Aliasing observation: track repeated observations within a trajectory
@dataclass
class _TrajectoryAliasingState:
    """Per-trajectory aliasing observation state.

    Two-layer detection:
      - exact: identical raw text
      - normalized: text after replacing paths/numbers/timestamps
    """

    exact_observations: list[str] = field(default_factory=list)
    normalized_observations: list[str] = field(default_factory=list)
    temporal_aliasing_events: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0

    def record_observation(self, text: str, step_idx: int) -> dict[str, Any] | None:
        """Record observation and detect temporal aliasing at both layers."""
        normalized = _normalize_observation(text)
        self.exact_observations.append(text)
        self.normalized_observations.append(normalized)
        self.step_count += 1

        event: dict[str, Any] | None = None

        # Exact aliasing: identical raw text reappears after a gap
        exact_priors = [
            i for i, observation in enumerate(self.exact_observations[:-1]) if observation == text
        ]
        if exact_priors and exact_priors[-1] < step_idx - 1:
            event = {
                "step_idx": step_idx,
                "match_type": "exact",
                "first_seen": exact_priors[0],
                "last_seen": exact_priors[-1],
                "gap": step_idx - exact_priors[-1],
            }
            self.temporal_aliasing_events.append(event)
            return event

        # Fuzzy aliasing: normalized text matches but raw text differs
        norm_priors = [
            i
            for i, observation in enumerate(self.normalized_observations[:-1])
            if observation == normalized
        ]
        if norm_priors and norm_priors[-1] < step_idx - 1:
            event = {
                "step_idx": step_idx,
                "match_type": "normalized",
                "first_seen": norm_priors[0],
                "last_seen": norm_priors[-1],
                "gap": step_idx - norm_priors[-1],
            }
            self.temporal_aliasing_events.append(event)

        return event


@BENCHMARKS.register("terminal_bench")
class TerminalBenchAdapter(BenchmarkBase):
    name = "terminal_bench"
    required_action_contract = "bash_command"
    eval_validity = EVAL_FIXTURE
    has_real_data = False
    has_native_harness = True
    supports_native_eval = False
    is_mock = True

    def __init__(
        self,
        *,
        repo_path: Path | str | None = None,
        docker_image_prefix: str = "terminal-bench",
        max_steps: int = 50,
        timeout_per_command: float = 30.0,
        track_aliasing: bool = True,
        difficulty_filter: str | None = None,
        category_filter: str | None = None,
    ) -> None:
        if not repo_path:
            raise ValueError("TerminalBench requires an explicit repo_path")
        self._repo_path = Path(repo_path)
        self._docker_prefix = docker_image_prefix
        self._max_steps = max_steps
        self._timeout = timeout_per_command
        self._track_aliasing = track_aliasing
        self._difficulty_filter = difficulty_filter
        self._category_filter = category_filter

        self._tasks: list[dict[str, Any]] = []
        self._containers: dict[str, str] = {}  # task_id -> container_id
        self._aliasing_states: dict[str, _TrajectoryAliasingState] = {}
        self._step_counters: dict[str, int] = {}

        if self._repo_path.is_dir():
            self._discover_tasks()
            if self._difficulty_filter:
                self._tasks = [t for t in self._tasks if t["difficulty"] == self._difficulty_filter]
            if self._category_filter:
                self._tasks = [t for t in self._tasks if t["category"] == self._category_filter]
            if self._tasks:
                self.has_real_data = True
                self.eval_validity = EVAL_REAL_NATIVE
                self.supports_native_eval = True
                self.is_mock = False

    def _discover_tasks(self) -> None:
        """Discover tasks from Terminal-Bench repo structure."""
        candidates = [
            self._repo_path / "original-tasks",
            self._repo_path / "tasks",
            self._repo_path / "benchmarks" / "tasks",
        ]
        tasks_dir = next((d for d in candidates if d.is_dir()), None)
        if tasks_dir is None:
            return

        for task_dir in sorted(tasks_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_yaml = task_dir / "task.yaml"
            task_json = task_dir / "task.json"
            task_toml = task_dir / "task.toml"
            meta = self._parse_task_meta(
                task_dir,
                task_yaml,
                task_json,
                task_toml,
            )
            if meta:
                self._tasks.append(meta)

    def _parse_task_meta(
        self,
        task_dir: Path,
        yaml_path: Path,
        json_path: Path,
        toml_path: Path,
    ) -> dict[str, Any] | None:
        """Parse task metadata from YAML or JSON definition."""
        if toml_path.is_file():
            data = tomllib.loads(toml_path.read_text())
            return self._normalize_harbor_task(data, task_dir)
        if yaml_path.is_file():
            try:
                import yaml

                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                return self._normalize_task(data, task_dir)
            except Exception:
                return None
        if json_path.is_file():
            try:
                data = json.loads(json_path.read_text())
                return self._normalize_task(data, task_dir)
            except Exception:
                return None
        return None

    def _normalize_harbor_task(
        self,
        data: dict[str, Any],
        task_dir: Path,
    ) -> dict[str, Any]:
        """Normalize the Harbor task format used by Terminal-Bench 2.1."""

        task = data["task"]
        metadata = data.get("metadata", {})
        environment = data.get("environment", {})
        agent = data.get("agent", {})
        instruction = (task_dir / "instruction.md").read_text().strip()
        dockerfile = task_dir / "environment" / "Dockerfile"
        configured_image = environment.get("docker_image")
        return {
            "task_id": task["name"],
            "instruction": instruction,
            "docker_image": configured_image or f"{self._docker_prefix}/{task_dir.name}",
            "prefer_prebuilt_image": bool(configured_image),
            "dockerfile": str(dockerfile),
            "docker_context": str(dockerfile.parent),
            "tests_dir": str(task_dir / "tests"),
            "task_dir": str(task_dir),
            "max_steps": self._max_steps,
            "timeout_s": float(agent.get("timeout_sec", 900.0)),
            "test_script": "/tests/test.sh",
            "parser_name": "harbor_reward",
            "difficulty": metadata.get("difficulty", "unknown"),
            "category": metadata.get("category", "unknown"),
            "tags": metadata.get("tags", task.get("keywords", [])),
        }

    def _normalize_task(self, data: dict[str, Any], task_dir: Path) -> dict[str, Any]:
        """Normalize task definition to internal format.

        Real Terminal-Bench task.yaml schema:
          instruction, difficulty, category, tags, parser_name,
          max_agent_timeout_sec, max_test_timeout_sec, author_name
        Container is built from the task's Dockerfile.
        """
        task_id = data.get("id", data.get("name", task_dir.name))
        instruction = data.get("instruction", data.get("description", ""))
        timeout = data.get("max_agent_timeout_sec", 900.0)
        docker_image = data.get(
            "docker_image",
            data.get("image", f"{self._docker_prefix}/{task_dir.name}"),
        )
        return {
            "task_id": task_id,
            "instruction": instruction,
            "docker_image": docker_image,
            "task_dir": str(task_dir),
            "max_steps": self._max_steps,
            "timeout_s": timeout,
            "test_script": "run-tests.sh",
            "parser_name": data.get("parser_name", "pytest"),
            "difficulty": data.get("difficulty", "unknown"),
            "category": data.get("category", "unknown"),
            "tags": data.get("tags", []),
        }

    def load(self, n: int = 3) -> list[Task]:
        if not self._tasks:
            return self._fixture_tasks(n)
        return [
            Task(
                task_id=t["task_id"],
                domain=TaskDomain.TOOL,
                instruction=t["instruction"],
                max_steps=t["max_steps"],
                timeout_s=t["timeout_s"],
                metadata={
                    "docker_image": t["docker_image"],
                    "prefer_prebuilt_image": t.get("prefer_prebuilt_image", False),
                    "dockerfile": t.get("dockerfile"),
                    "docker_context": t.get("docker_context"),
                    "tests_dir": t.get("tests_dir"),
                    "task_dir": t["task_dir"],
                    "test_script": t["test_script"],
                    "parser_name": t.get("parser_name", "pytest"),
                    "difficulty": t["difficulty"],
                    "category": t["category"],
                    "tags": t.get("tags", []),
                    "benchmark": "terminal_bench",
                },
            )
            for t in self._tasks[:n]
        ]

    def _fixture_tasks(self, n: int) -> list[Task]:
        """Synthetic fixture tasks for smoke testing without the repo."""
        fixtures = [
            {
                "task_id": "tb_fixture_git_conflict",
                "instruction": "Resolve the merge conflict in /repo/main.py and commit the result.",
                "category": "git",
            },
            {
                "task_id": "tb_fixture_nginx_config",
                "instruction": "Configure nginx to serve /var/www/html on port 8080 and verify with curl.",
                "category": "sysadmin",
            },
            {
                "task_id": "tb_fixture_python_debug",
                "instruction": "Fix the failing pytest tests in /app/tests/ without modifying test files.",
                "category": "debugging",
            },
        ]
        return [
            Task(
                task_id=f["task_id"],
                domain=TaskDomain.TOOL,
                instruction=f["instruction"],
                max_steps=self._max_steps,
                metadata={
                    "category": f["category"],
                    "benchmark": "terminal_bench",
                    "fixture": True,
                },
            )
            for f in fixtures[:n]
        ]

    def step(self, task: Task, action: Action) -> Observation:
        """Execute a bash command in the task's Docker container."""
        task_id = task.task_id
        step_idx = self._step_counters.get(task_id, 0)
        self._step_counters[task_id] = step_idx + 1

        if self.is_mock:
            return self._mock_step(task, action, step_idx)

        container_id = self._ensure_container(task)
        if container_id is None:
            return Observation(
                content="[terminal_bench] ERROR: failed to start container",
                is_error=True,
                info={"done": True, "step_idx": step_idx},
            )

        output, exit_code = self._exec_in_container(container_id, action.raw)

        obs = Observation(
            content=output,
            is_error=exit_code != 0,
            info={
                "done": False,
                "exit_code": exit_code,
                "step_idx": step_idx,
                "command": action.raw,
            },
        )

        if self._track_aliasing:
            state = self._aliasing_states.setdefault(task_id, _TrajectoryAliasingState())
            aliasing_event = state.record_observation(output, step_idx)
            if aliasing_event:
                obs.info["temporal_aliasing_event"] = aliasing_event

        return obs

    def _mock_step(self, task: Task, action: Action, step_idx: int) -> Observation:
        """Mock step simulating category-specific temporal aliasing patterns.

        Debugging tasks: repeated error traces (high aliasing)
        Git tasks: repeated status output (medium aliasing)
        Sysadmin tasks: repeated permission/connection errors (high aliasing)
        Data tasks: unique outputs per step (low aliasing)
        """
        cmd = action.raw.strip()
        category = task.metadata.get("category", "unknown")
        # Python's string hash is randomized per process, which made the
        # temporal-aliasing metric irreproducible across runs; use a stable
        # checksum instead.
        task_hash = zlib.crc32(task.task_id.encode("utf-8")) % 1000

        # Category-specific "stuck" outputs that repeat when agent retries
        _STUCK_OUTPUTS = {
            "debugging": (
                f"Traceback (most recent call last):\n"
                f'  File "/app/main.py", line {12 + task_hash % 30}, in <module>\n'
                f"    result = process(data)\n"
                f"TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'"
            ),
            "software-engineering": (
                f"FAILED tests/test_solution.py::test_case_{task_hash % 5} - "
                f"AssertionError: assert {task_hash % 4 + 1} == 0\n"
                f"1 failed in 0.{task_hash % 9 + 1}s"
            ),
            "system-administration": ("bash: /app/setup.sh: Permission denied"),
            "security": (
                f"curl: (7) Failed to connect to localhost port {8000 + task_hash % 100} "
                f"after 2 ms: Couldn't connect to server"
            ),
            "data-science": (
                f"pandas.errors.ParserError: Error tokenizing data. "
                f"C error: Expected {task_hash % 10 + 3} fields in line {task_hash % 100 + 2}, saw {task_hash % 10 + 4}"
            ),
        }
        stuck_output = _STUCK_OUTPUTS.get(
            category,
            f"Error: operation failed (code={task_hash % 256})",
        )

        # Commands that naturally produce aliased (repeated) output
        is_retry_cmd = any(
            kw in cmd for kw in ("python", "pytest", "bash /app", "curl", "make", "gcc")
        )
        # Commands that produce unique output per step
        is_explore_cmd = any(
            kw in cmd for kw in ("cat", "ls", "find", "grep", "head", "tail", "wc")
        )

        if is_retry_cmd:
            output = stuck_output
        elif is_explore_cmd:
            output = f"[step {step_idx}] {cmd[:60]}\nfile_{step_idx}.txt  data_{step_idx}.csv  src_{step_idx}/"
        elif "git" in cmd:
            output = "On branch main\nnothing to commit, working tree clean"
        elif "pip" in cmd or "conda" in cmd:
            output = f"ERROR: Could not find a version that satisfies the requirement pkg{task_hash % 50}"
        else:
            output = f"$ {cmd}\n[exit 0]"

        obs = Observation(
            content=output,
            is_error=is_retry_cmd,
            info={"done": False, "step_idx": step_idx, "mock": True},
        )

        if self._track_aliasing:
            state = self._aliasing_states.setdefault(task.task_id, _TrajectoryAliasingState())
            aliasing_event = state.record_observation(output, step_idx)
            if aliasing_event:
                obs.info["temporal_aliasing_event"] = aliasing_event

        return obs

    def _ensure_container(self, task: Task) -> str | None:
        """Build image from task Dockerfile and start a container."""
        task_id = task.task_id
        if task_id in self._containers:
            return self._containers[task_id]

        image = task.metadata.get("docker_image", "")
        task_dir = task.metadata.get("task_dir", "")
        if not image:
            return None

        dockerfile = task.metadata.get("dockerfile")
        docker_context = task.metadata.get("docker_context")
        if not dockerfile and task_dir and Path(task_dir, "Dockerfile").is_file():
            dockerfile = str(Path(task_dir, "Dockerfile"))
            docker_context = task_dir
        if dockerfile and not task.metadata.get("prefer_prebuilt_image", False):
            try:
                build = subprocess.run(
                    [
                        "docker",
                        "build",
                        "-f",
                        dockerfile,
                        "-t",
                        image,
                        "-q",
                        docker_context,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if build.returncode != 0:
                    return None
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                return None

        try:
            run_command = [
                "docker",
                "run",
                "-d",
                "--rm",
                "-it",
                "--tmpfs",
                "/logs/verifier",
            ]
            tests_dir = task.metadata.get("tests_dir")
            if tests_dir:
                run_command.extend(["-v", f"{tests_dir}:/tests:ro"])
            container_proxy = os.environ.get("LONGH_CONTAINER_PROXY")
            if container_proxy:
                run_command.extend(
                    [
                        "-e",
                        f"http_proxy={container_proxy}",
                        "-e",
                        f"https_proxy={container_proxy}",
                    ]
                )
            run_command.extend([image, "/bin/bash"])
            result = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                container_id = result.stdout.strip()
                self._containers[task_id] = container_id
                return container_id
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    def _exec_in_container(self, container_id: str, command: str) -> tuple[str, int]:
        """Execute a command in the container and return (output, exit_code)."""
        try:
            result = subprocess.run(
                ["docker", "exec", container_id, "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr if output else result.stderr
            return output.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return f"[timeout after {self._timeout}s]", -1
        except (FileNotFoundError, OSError) as e:
            return f"[docker error: {e}]", -1

    def get_aliasing_report(self, task_id: str) -> dict[str, Any]:
        """Return aliasing observation metrics for a completed trajectory."""
        state = self._aliasing_states.get(task_id)
        if state is None:
            return {"task_id": task_id, "steps": 0, "temporal_aliasing_rate": 0.0}
        n_steps = state.step_count
        n_events = len(state.temporal_aliasing_events)
        unique_exact = len(set(state.exact_observations))
        unique_norm = len(set(state.normalized_observations))
        exact_events = sum(1 for e in state.temporal_aliasing_events if e["match_type"] == "exact")
        norm_events = n_events - exact_events
        return {
            "task_id": task_id,
            "steps": n_steps,
            "unique_observations_exact": unique_exact,
            "unique_observations_normalized": unique_norm,
            "exact_repetition_rate": 1.0 - (unique_exact / max(n_steps, 1)),
            "normalized_repetition_rate": 1.0 - (unique_norm / max(n_steps, 1)),
            "temporal_aliasing_events": n_events,
            "temporal_aliasing_rate": n_events / max(n_steps, 1),
            "exact_aliasing_events": exact_events,
            "normalized_aliasing_events": norm_events,
            "events": state.temporal_aliasing_events,
        }

    def get_all_aliasing_reports(self) -> list[dict[str, Any]]:
        """Return aliasing reports for all tracked trajectories."""
        return [self.get_aliasing_report(tid) for tid in self._aliasing_states]

    def close(self, artifacts_dir: Path | None = None) -> None:
        """Stop all running containers and optionally write aliasing report."""
        for _task_id, container_id in list(self._containers.items()):
            try:
                subprocess.run(
                    ["docker", "stop", container_id],
                    capture_output=True,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
        self._containers.clear()

        if artifacts_dir and self._track_aliasing:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "benchmark": "terminal_bench",
                "schema": "lh_terminal_bench_aliasing/v1",
                "reports": self.get_all_aliasing_reports(),
            }
            (artifacts_dir / "aliasing_report.json").write_text(json.dumps(report, indent=2))

    def outcome_evidence(self, task: Task):
        """Run the task's test script to determine success."""
        from lh_platform.core.outcome import (
            EVIDENCE_ABSENT,
            EVIDENCE_STRONG,
            OutcomeEvidence,
        )

        if self.is_mock:
            return OutcomeEvidence.fallback(False)

        container_id = self._containers.get(task.task_id)
        if not container_id:
            return OutcomeEvidence.fallback(False)

        test_script = task.metadata.get("test_script", "tests/run-tests.sh")
        output, exit_code = self._exec_in_container(container_id, f"bash {test_script}")
        if task.metadata.get("parser_name") == "harbor_reward":
            reward, reward_exit_code = self._exec_in_container(
                container_id,
                "cat /logs/verifier/reward.txt",
            )
            if reward_exit_code != 0 or reward.strip() not in {"0", "1"}:
                return OutcomeEvidence(
                    success=False,
                    reason="native Terminal-Bench verifier did not produce a reward",
                    evidence_quality=EVIDENCE_ABSENT,
                    source="terminal_bench_native_test",
                    test_results={"output": output[:2000]},
                ).validate()
            success = reward.strip() == "1"
        else:
            success = exit_code == 0
        return OutcomeEvidence(
            success=success,
            reason="native Terminal-Bench verifier completed",
            evidence_quality=EVIDENCE_STRONG,
            source="terminal_bench_native_test",
            test_results={"output": output[:2000]},
        ).validate()

    def task_metrics(self, task: Task, trajectory: object) -> dict[str, object]:
        """Expose aliasing stats as environment metrics for the Runner."""
        if not self._track_aliasing:
            return {}
        report = self.get_aliasing_report(task.task_id)
        return {
            "environment_temporal_aliasing_rate": report["temporal_aliasing_rate"],
            "environment_exact_repetition_rate": report["exact_repetition_rate"],
            "environment_normalized_repetition_rate": report["normalized_repetition_rate"],
            "environment_aliasing_events": report["temporal_aliasing_events"],
            "environment_steps": report["steps"],
        }
