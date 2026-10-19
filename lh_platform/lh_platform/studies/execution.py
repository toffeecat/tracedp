"""Durable, observable execution of prepared study commands."""

# Portions reference upstream verl source (Apache-2.0), https://github.com/volcengine/verl. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import selectors
import shutil
import subprocess
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RUN_STATE_SCHEMA = "lh_study_run/v1"
_METRIC_WORDS = (
    "loss",
    "reward",
    "success",
    "kl",
    "entropy",
    "grad",
    "step",
    "lr",
    "token",
    "throughput",
)
_PAIR_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_./-]*)\s*[=:]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.I,
)
_SMALL_CALIBRATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_SMALL_CALIBRATION_OVERRIDES = {
    "data.train_batch_size": "1",
    "data.val_batch_size": "1",
    "data.gen_batch_size": "1",
    "data.max_response_length": "256",
    "actor_rollout_ref.actor.ppo_mini_batch_size": "1",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "3072",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": "3072",
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": "3072",
    "actor_rollout_ref.model.enable_activation_offload": "True",
    "actor_rollout_ref.actor.fsdp_config.param_offload": "True",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "True",
    "actor_rollout_ref.rollout.gpu_memory_utilization": "0.20",
    "actor_rollout_ref.rollout.enforce_eager": "True",
    "actor_rollout_ref.rollout.free_cache_engine": "True",
    "env.rollout.n": "1",
    "ray_init.num_cpus": "4",
    "+ray_init.include_dashboard": "False",
    "trainer.logger": "['console']",
    "trainer.save_freq": "-1",
    "trainer.test_freq": "1",
}
_SMALL_CALIBRATION_ENV = {
    "LH_TORCH_ADAMW_FOREACH": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
_SMALL_CALIBRATION_TRAIN_ROWS = 2
_SMALL_CALIBRATION_VAL_ROWS = 4
_GRAPHGPO_1P5B_ALFWORLD_MIN_GPUS = 2
_GRAPHGPO_1P5B_ALFWORLD_GPU_POLICY_BASIS = {
    "source_path": "recipe/GraphGPO/run_qwen2.5_1.5b_alfworld_train.sh",
    "quote": "trainer.n_gpus_per_node=2",
}
_THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class TrialExecutionResult:
    trial_id: str
    status: str
    returncode: int | None
    elapsed_s: float
    command_sha256: str
    output_dir: str
    parsed_metric_events: int
    postrun_warning_count: int = 0


@dataclass(frozen=True)
class FormalResourceGate:
    decision: str
    requirements: Mapping[str, Any]
    inventory: Mapping[str, Any]
    blocked_by: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.decision in {"passed", "skipped"}

    def as_event(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "requirements": dict(self.requirements),
            "inventory": dict(self.inventory),
            "blocked_by": list(self.blocked_by),
        }


class StudyExecutor:
    def __init__(self, prepared_dir: str | Path) -> None:
        self.root = Path(prepared_dir).resolve()
        self.commands_path = self.root / "planned_commands.jsonl"
        if not self.commands_path.is_file():
            raise FileNotFoundError(self.commands_path)
        self.state_path = self.root / "study_run_state.json"
        self.lock_path = self.root / ".study_run.lock"

    def run(
        self,
        *,
        trial_ids: Iterable[str] = (),
        resume: bool = False,
        training_steps: int | None = None,
        calibration_small: bool = False,
    ) -> tuple[TrialExecutionResult, ...]:
        self._require_launch_allowed()
        trial_ids = tuple(trial_ids)
        lock_fd = self._acquire_lock()
        try:
            self._ensure_preprocessed(trial_ids)
            return self._run_locked(
                trial_ids=trial_ids,
                resume=resume,
                training_steps=training_steps,
                calibration_small=calibration_small,
            )
        finally:
            os.close(lock_fd)
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def _run_locked(
        self,
        *,
        trial_ids: Iterable[str],
        resume: bool,
        training_steps: int | None,
        calibration_small: bool,
    ) -> tuple[TrialExecutionResult, ...]:
        selected = set(trial_ids)
        commands = [
            row
            for row in _read_jsonl(self.commands_path)
            if not selected or row["trial_id"] in selected
        ]
        previous = self._read_state()
        completed = {
            item["trial_id"]
            for item in previous.get("trials", [])
            if item.get("status") == "completed"
        }
        results: list[TrialExecutionResult] = []
        for command in commands:
            if resume and command["trial_id"] in completed:
                continue
            running = {
                "trial_id": str(command["trial_id"]),
                "status": "running",
                "started_at": time.time(),
                "output_dir": str(command["output_dir"]),
            }
            self._write_state(
                [*previous.get("trials", []), *[asdict(item) for item in results], running]
            )
            result = self._run_trial(
                command,
                training_steps=training_steps,
                calibration_small=calibration_small,
            )
            results.append(result)
            self._write_state([*previous.get("trials", []), *[asdict(item) for item in results]])
            if result.status != "completed":
                break
        return tuple(results)

    def status(self) -> Mapping[str, Any]:
        return self._read_state()

    def _run_trial(
        self,
        command: Mapping[str, Any],
        *,
        training_steps: int | None,
        calibration_small: bool,
    ) -> TrialExecutionResult:
        trial_id = str(command["trial_id"])
        argv = [str(item) for item in command["argv"]]
        if training_steps is not None:
            argv = _replace_override(argv, "trainer.total_training_steps", str(training_steps))
            argv = _replace_override(argv, "trainer.total_epochs", str(training_steps))
        if calibration_small:
            calibration_overrides = {
                **_SMALL_CALIBRATION_OVERRIDES,
                "actor_rollout_ref.model.path": os.environ.get(
                    "LH_GRAPHGPO_CALIBRATION_MODEL",
                    _SMALL_CALIBRATION_MODEL,
                ),
            }
            for key, value in calibration_overrides.items():
                argv = _replace_override(argv, key, value)
        output = Path(str(command["output_dir"]))
        if training_steps is not None:
            output = output.parent / f"{output.name}-calibration-{training_steps}steps"
        if calibration_small:
            output = output.parent / f"{output.name}-small"
        logs = output / "logs"
        metrics_dir = output / "metrics"
        plots = output / "plots"
        for directory in (logs, metrics_dir, plots):
            directory.mkdir(parents=True, exist_ok=True)
        stdout_path = logs / "stdout.log"
        events_path = logs / "events.jsonl"
        metrics_path = metrics_dir / "metrics.jsonl"
        started = time.time()
        env = _execution_env(command.get("environment", {}))
        if calibration_small:
            env.setdefault("LH_ATTENTION_IMPLEMENTATION", "sdpa")
            for key, value in _SMALL_CALIBRATION_ENV.items():
                env.setdefault(key, value)
            argv, calibration_data = _apply_small_calibration_data(argv, output)
        else:
            calibration_data = {}
        command_sha = hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
        with (
            stdout_path.open("a", encoding="utf-8") as raw,
            events_path.open("a", encoding="utf-8") as events,
            metrics_path.open("a", encoding="utf-8") as metrics,
        ):
            startup_compat = _install_startup_compat(argv[0], env)
            if startup_compat:
                _jsonl(
                    events,
                    {
                        "event": "startup_compat_installed",
                        "ts": time.time(),
                        "trial_id": trial_id,
                        **startup_compat,
                    },
                )
            upstream_patch = _patch_upstream_transformers_import(env)
            if upstream_patch:
                _jsonl(
                    events,
                    {
                        "event": "upstream_runtime_compat_patched",
                        "ts": time.time(),
                        "trial_id": trial_id,
                        **upstream_patch,
                    },
                )
            if calibration_data:
                _jsonl(
                    events,
                    {
                        "event": "calibration_data_subset",
                        "ts": time.time(),
                        "trial_id": trial_id,
                        **calibration_data,
                    },
                )
            if calibration_small:
                _jsonl(
                    events,
                    {
                        "event": "calibration_runtime_profile",
                        "ts": time.time(),
                        "trial_id": trial_id,
                        "env": {
                            key: env[key] for key in sorted(_SMALL_CALIBRATION_ENV) if key in env
                        },
                    },
                )
            resource_gate = _formal_resource_gate(
                command,
                argv,
                calibration_small=calibration_small,
            )
            if resource_gate.decision != "skipped":
                _jsonl(
                    events,
                    {
                        "event": "formal_resource_gate",
                        "ts": time.time(),
                        "trial_id": trial_id,
                        **resource_gate.as_event(),
                    },
                )
            if not resource_gate.passed:
                raw.write(
                    "LongHorizon formal resource gate blocked launch: "
                    + ",".join(resource_gate.blocked_by)
                    + "\n"
                )
                raw.flush()
                status = "blocked"
                returncode = None
                parsed_count = 0
                diagnostics = {}
                _jsonl(
                    events,
                    {
                        "event": "trial_finished",
                        "ts": time.time(),
                        "trial_id": trial_id,
                        "status": status,
                        "returncode": returncode,
                        "postrun_warning_count": 0,
                        "blocked_by": list(resource_gate.blocked_by),
                    },
                )
                rows = _read_jsonl(metrics_path)
                _write_metrics_csv(rows, metrics_dir / "metrics.csv")
                _write_svg_curves(rows, plots / "training_curves.svg")
                return TrialExecutionResult(
                    trial_id=trial_id,
                    status=status,
                    returncode=returncode,
                    elapsed_s=time.time() - started,
                    command_sha256=command_sha,
                    output_dir=str(output),
                    parsed_metric_events=parsed_count,
                    postrun_warning_count=diagnostics.get("warning_count", 0),
                )
            _jsonl(
                events,
                {
                    "event": "trial_started",
                    "ts": started,
                    "trial_id": trial_id,
                    "argv": argv,
                    "command_sha256": command_sha,
                },
            )
            process = subprocess.Popen(
                argv,
                cwd=str(command["cwd"]),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            selector = selectors.DefaultSelector()
            assert process.stdout is not None
            selector.register(process.stdout, selectors.EVENT_READ)
            parsed_count = 0
            last_heartbeat = 0.0
            while process.poll() is None:
                ready = selector.select(timeout=5.0)
                for key, _ in ready:
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    raw.write(line)
                    raw.flush()
                    payload = parse_metric_line(line)
                    if payload:
                        parsed_count += 1
                        _jsonl(metrics, {"ts": time.time(), "trial_id": trial_id, **payload})
                now = time.time()
                if now - last_heartbeat >= 10:
                    sample = _gpu_sample()
                    _jsonl(
                        events,
                        {
                            "event": "heartbeat",
                            "ts": now,
                            "trial_id": trial_id,
                            "pid": process.pid,
                            **sample,
                        },
                    )
                    if sample:
                        _jsonl(metrics, {"ts": now, "trial_id": trial_id, **sample})
                    last_heartbeat = now
            for line in process.stdout:
                raw.write(line)
                payload = parse_metric_line(line)
                if payload:
                    parsed_count += 1
                    _jsonl(metrics, {"ts": time.time(), "trial_id": trial_id, **payload})
            returncode = process.wait()
            raw.flush()
            diagnostics = _scan_stdout_diagnostics(stdout_path)
            if diagnostics:
                _jsonl(
                    events,
                    {
                        "event": "postrun_log_diagnostics",
                        "ts": time.time(),
                        "trial_id": trial_id,
                        **diagnostics,
                    },
                )
            status = "completed" if returncode == 0 else "failed"
            _jsonl(
                events,
                {
                    "event": "trial_finished",
                    "ts": time.time(),
                    "trial_id": trial_id,
                    "status": status,
                    "returncode": returncode,
                    "postrun_warning_count": diagnostics.get("warning_count", 0),
                },
            )
        rows = _read_jsonl(metrics_path)
        _write_metrics_csv(rows, metrics_dir / "metrics.csv")
        _write_svg_curves(rows, plots / "training_curves.svg")
        return TrialExecutionResult(
            trial_id=trial_id,
            status=status,
            returncode=returncode,
            elapsed_s=time.time() - started,
            command_sha256=command_sha,
            output_dir=str(output),
            parsed_metric_events=parsed_count,
            postrun_warning_count=diagnostics.get("warning_count", 0),
        )

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"schema": RUN_STATE_SCHEMA, "trials": []}
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {"schema": RUN_STATE_SCHEMA, "trials": []}

    def _write_state(self, trials: list[Mapping[str, Any]]) -> None:
        payload = {"schema": RUN_STATE_SCHEMA, "updated_at": time.time(), "trials": trials}
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def _require_launch_allowed(self) -> None:
        summary_path = self.root / "prepare_summary.json"
        if not summary_path.is_file():
            return
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, Mapping) and summary.get("launch_allowed") is False:
            blocked = summary.get("blocked_by", [])
            raise RuntimeError(f"prepared study is not launchable: {blocked}")

    def _ensure_preprocessed(self, trial_ids: Iterable[str]) -> None:
        summary_path = self.root / "prepare_summary.json"
        if not summary_path.is_file():
            return
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, Mapping):
            return
        preprocess_argv = tuple(str(item) for item in summary.get("preprocess_argv", ()))
        if not preprocess_argv:
            return
        selected = set(trial_ids)
        commands = [
            row
            for row in _read_jsonl(self.commands_path)
            if not selected or row["trial_id"] in selected
        ]
        required_files = _required_training_files(commands)
        if not required_files or all(path.is_file() for path in required_files):
            return
        output = self.root / "preprocess"
        logs = output / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stdout_path = logs / "stdout.log"
        events_path = logs / "events.jsonl"
        started = time.time()
        env = _execution_env(dict(summary.get("preprocess_environment", {})))
        cwd = str(summary.get("preprocess_cwd") or self.root)
        command_sha = hashlib.sha256(
            json.dumps(preprocess_argv, separators=(",", ":")).encode()
        ).hexdigest()
        with (
            stdout_path.open("a", encoding="utf-8") as raw,
            events_path.open("a", encoding="utf-8") as events,
        ):
            _jsonl(
                events,
                {
                    "event": "preprocess_started",
                    "ts": started,
                    "argv": list(preprocess_argv),
                    "command_sha256": command_sha,
                    "required_files": [str(path) for path in required_files],
                },
            )
            process = subprocess.run(
                preprocess_argv,
                cwd=cwd,
                env=env,
                stdout=raw,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            missing = [path for path in required_files if not path.is_file()]
            status = "completed" if process.returncode == 0 and not missing else "failed"
            _jsonl(
                events,
                {
                    "event": "preprocess_finished",
                    "ts": time.time(),
                    "status": status,
                    "returncode": process.returncode,
                    "missing_files": [str(path) for path in missing],
                },
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"study preprocess failed with return code {process.returncode}: {stdout_path}"
            )
        if missing:
            raise RuntimeError(
                "study preprocess did not create required training files: "
                + ", ".join(str(path) for path in missing)
            )

    def _acquire_lock(self) -> int:
        try:
            return os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"study run already active: {self.lock_path}") from exc


def parse_metric_line(line: str) -> dict[str, float]:
    values: dict[str, float] = {}
    stripped = line.strip()
    if "{" in stripped and "}" in stripped:
        candidate = stripped[stripped.find("{") : stripped.rfind("}") + 1]
        try:
            mapping = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            mapping = None
        if isinstance(mapping, Mapping):
            _flatten_metrics(mapping, values)
    pairs = _PAIR_RE.findall(stripped)
    structured_upstream_metrics = (
        "step:" in stripped and " - " in stripped and any("/" in key for key, _ in pairs)
    )
    for key, value in pairs:
        if structured_upstream_metrics or any(word in key.lower() for word in _METRIC_WORDS):
            values[key] = float(value)
    return values


def _flatten_metrics(
    mapping: Mapping[str, Any], target: dict[str, float], prefix: str = ""
) -> None:
    for key, value in mapping.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _flatten_metrics(value, target, name)
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and any(word in name.lower() for word in _METRIC_WORDS)
        ):
            target[name] = float(value)


_STDOUT_DIAGNOSTIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("traceback", re.compile(r"Traceback \(most recent call last\):")),
    ("cuda_oom", re.compile(r"(CUDA out of memory|OutOfMemoryError)", re.I)),
    ("ray_oom", re.compile(r"(ray\.exceptions\.OutOfMemoryError|memory on the node)", re.I)),
    (
        "dataloader_worker_killed",
        re.compile(r"DataLoader worker .* killed by signal: Killed", re.I),
    ),
)


def _scan_stdout_diagnostics(path: Path, *, max_examples: int = 20) -> dict[str, Any]:
    """Surface suspicious stdout patterns even when upstream exits with code 0."""
    if not path.is_file():
        return {}
    warnings: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        for category, pattern in _STDOUT_DIAGNOSTIC_PATTERNS:
            if not pattern.search(line):
                continue
            counts[category] = counts.get(category, 0) + 1
            if len(warnings) < max_examples:
                warnings.append(
                    {
                        "category": category,
                        "line": line_number,
                        "text": line.strip()[:500],
                    }
                )
            break
    if not counts:
        return {}
    return {
        "warning_count": sum(counts.values()),
        "warning_counts": counts,
        "warnings": warnings,
    }


def _gpu_sample() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        values = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
        return {
            key: float(value)
            for key, value in zip(
                (
                    "gpu_utilization",
                    "gpu_memory_used_mib",
                    "gpu_memory_total_mib",
                    "gpu_temperature_c",
                    "gpu_power_w",
                ),
                values,
                strict=False,
            )
        }
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return {}


def _formal_resource_gate(
    command: Mapping[str, Any],
    argv: Iterable[str],
    *,
    calibration_small: bool,
) -> FormalResourceGate:
    if calibration_small:
        return FormalResourceGate(
            decision="skipped",
            requirements={"reason": "calibration_small"},
            inventory={},
        )
    if command.get("planned_execution_path") != "upstream_native_runner":
        return FormalResourceGate(
            decision="skipped",
            requirements={"reason": "not_upstream_native_runner"},
            inventory={},
        )

    argv = tuple(str(item) for item in argv)
    overrides = command.get("overrides", {})
    if not isinstance(overrides, Mapping):
        overrides = {}
    resource_budget = command.get("resource_budget", {})
    if not isinstance(resource_budget, Mapping):
        resource_budget = {}

    required_gpus = _first_int(
        _override_value(argv, "trainer.n_gpus_per_node"),
        overrides.get("trainer.n_gpus_per_node"),
        resource_budget.get("num_gpus"),
        resource_budget.get("gpus"),
        default=0,
    )
    model_path = str(
        _override_value(argv, "actor_rollout_ref.model.path")
        or overrides.get("actor_rollout_ref.model.path")
        or ""
    )
    environment = command.get("environment", {})
    if not isinstance(environment, Mapping):
        environment = {}
    study_id = str(command.get("study_id") or environment.get("LH_STUDY_ID") or "")
    variant_id = str(command.get("variant_id") or environment.get("LH_VARIANT_ID") or "")
    requirements = {
        "required_gpus": required_gpus,
        "model_path": model_path,
        "study_id": study_id,
        "variant_id": variant_id,
        "planned_execution_path": str(command.get("planned_execution_path")),
    }
    if _is_graphgpo_alfworld_formal_1p5b(
        study_id=study_id,
        variant_id=variant_id,
        model_path=model_path,
    ):
        requirements["policy_basis"] = _GRAPHGPO_1P5B_ALFWORLD_GPU_POLICY_BASIS
    inventory = _gpu_inventory()
    blockers: list[str] = []

    if required_gpus > 0 and inventory["gpu_count"] < required_gpus:
        blockers.append("gpu_count_insufficient")
    if required_gpus > 0 and not inventory["available"]:
        blockers.append("gpu_inventory_unavailable")
    if (
        _is_graphgpo_alfworld_formal_1p5b(
            study_id=study_id,
            variant_id=variant_id,
            model_path=model_path,
        )
        and required_gpus < _GRAPHGPO_1P5B_ALFWORLD_MIN_GPUS
    ):
        blockers.append("formal_1p5b_requires_at_least_2_gpus")

    return FormalResourceGate(
        decision="blocked" if blockers else "passed",
        requirements=requirements,
        inventory=inventory,
        blocked_by=tuple(dict.fromkeys(blockers)),
    )


def _gpu_inventory() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "gpu_count": 0,
            "gpus": [],
            "details": str(error),
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "gpu_count": 0,
            "gpus": [],
            "details": completed.stderr.strip() or completed.stdout.strip(),
        }
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        name, _, memory = line.rpartition(",")
        try:
            total_mib = int(float(memory.strip()))
        except ValueError:
            total_mib = 0
        gpus.append({"name": name.strip(), "memory_total_mib": total_mib})
    return {
        "available": bool(gpus),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


def _first_int(*values: Any, default: int = 0) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _is_graphgpo_alfworld_formal_1p5b(
    *,
    study_id: str,
    variant_id: str,
    model_path: str,
) -> bool:
    study = study_id.lower()
    variant = variant_id.lower()
    normalized = model_path.lower().replace("_", "-")
    return (
        "graphgpo" in study
        and "alfworld" in study
        and variant in {"graphgpo", "grpo"}
        and ("qwen2.5-1.5b" in normalized or "1.5b" in normalized)
    )


def _replace_override(argv: list[str], key: str, value: str) -> list[str]:
    prefix = key + "="
    replaced = [prefix + value if item.startswith(prefix) else item for item in argv]
    if not any(item.startswith(prefix) for item in argv):
        replaced.append(prefix + value)
    return replaced


def _override_value(argv: Iterable[str], key: str) -> str | None:
    prefix = key + "="
    for item in argv:
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def _apply_small_calibration_data(
    argv: list[str], output: Path
) -> tuple[list[str], dict[str, Any]]:
    """Use tiny parquet subsets for smoke calibration without changing formal commands."""
    mapping = {
        "data.train_files": _SMALL_CALIBRATION_TRAIN_ROWS,
        "data.val_files": _SMALL_CALIBRATION_VAL_ROWS,
    }
    data_dir = output / "calibration_data"
    summary: dict[str, Any] = {"rows": {}, "sources": {}, "files": {}}
    updated = list(argv)
    for key, limit in mapping.items():
        raw = _override_value(updated, key)
        if not raw:
            continue
        source = Path(raw)
        if not source.is_file() or source.suffix != ".parquet":
            continue
        data_dir.mkdir(parents=True, exist_ok=True)
        target = data_dir / f"{key.rsplit('.', 1)[-1]}-head-{limit}.parquet"
        row_count = _write_parquet_head(source, target, limit)
        updated = _replace_override(updated, key, str(target))
        summary["rows"][key] = row_count
        summary["sources"][key] = str(source)
        summary["files"][key] = str(target)
    return updated, summary if summary["files"] else {}


def _write_parquet_head(source: Path, target: Path, limit: int) -> int:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "calibration-small requires pandas/pyarrow to create parquet subsets"
        ) from exc
    frame = pd.read_parquet(source).head(limit)
    if len(frame) == 0:
        shutil.copyfile(source, target)
    else:
        frame.to_parquet(target, index=False)
    return int(len(frame))


def _required_training_files(commands: Iterable[Mapping[str, Any]]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for command in commands:
        overrides = command.get("overrides", {})
        if not isinstance(overrides, Mapping):
            continue
        for key in ("data.train_files", "data.val_files"):
            value = overrides.get(key)
            if value:
                paths.append(Path(str(value)))
    return tuple(dict.fromkeys(paths))


def _execution_env(extra: Mapping[str, Any]) -> dict[str, str]:
    env = {str(key): str(value) for key, value in os.environ.items()}
    for key, value in _THREAD_LIMIT_ENV.items():
        env.setdefault(key, value)
    env.update({str(key): str(value) for key, value in extra.items()})
    return env


def remove_managed_startup_compat_files(
    site_package_directories: Iterable[str | Path],
) -> dict[str, list[str]]:
    """Remove only stale global files owned by older LongHorizon launches.

    Prepared commands already bind their immutable ``python_compat`` directory
    through ``PYTHONPATH`` and Ray's runtime environment.  A global ``.pth``
    cannot represent more than one prepared run, so the former implementation
    made later experiments silently import an earlier run's ``sitecustomize``.
    """

    removed_pth_paths: list[str] = []
    removed_sitecustomize_paths: list[str] = []
    for raw_directory in site_package_directories:
        directory = Path(raw_directory)
        pth_path = directory / "longhorizon_graphgpo_compat.pth"
        if pth_path.exists():
            pth_path.unlink()
            removed_pth_paths.append(str(pth_path))
        sitecustomize_path = directory / "sitecustomize.py"
        if not sitecustomize_path.is_file():
            continue
        content = sitecustomize_path.read_text(encoding="utf-8", errors="ignore")
        if "LH_STARTUP_COMPAT_MANAGED" in content:
            sitecustomize_path.unlink()
            removed_sitecustomize_paths.append(str(sitecustomize_path))
    return {
        "pth_paths": removed_pth_paths,
        "sitecustomize_paths": removed_sitecustomize_paths,
    }


def _install_startup_compat(python_executable: str, env: Mapping[str, str]) -> dict[str, Any]:
    """Isolate one prepared compatibility path without mutating global startup."""

    compat_path = env.get("LH_STARTUP_COMPAT_PATH")
    if not compat_path:
        return {}
    compat = Path(compat_path).resolve()
    if not (compat / "sitecustomize.py").is_file():
        raise RuntimeError(f"prepared startup compat is missing: {compat}")
    script = r"""
import json
import site
from pathlib import Path

directories = []
try:
    directories.extend(site.getsitepackages())
except Exception:
    pass
try:
    directories.append(site.getusersitepackages())
except Exception:
    pass
removed_pth_paths = []
removed_sitecustomize_paths = []
for raw in dict.fromkeys(directories):
    directory = Path(raw)
    pth_path = directory / "longhorizon_graphgpo_compat.pth"
    if pth_path.exists():
        pth_path.unlink()
        removed_pth_paths.append(str(pth_path))
    sitecustomize_path = directory / "sitecustomize.py"
    if sitecustomize_path.is_file() and "LH_STARTUP_COMPAT_MANAGED" in sitecustomize_path.read_text(encoding="utf-8", errors="ignore"):
        sitecustomize_path.unlink()
        removed_sitecustomize_paths.append(str(sitecustomize_path))
print(json.dumps({"pth_paths": removed_pth_paths, "sitecustomize_paths": removed_sitecustomize_paths}, sort_keys=True))
"""
    completed = subprocess.run(
        [python_executable, "-c", script],
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to isolate prepared startup compat: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        removed = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "startup compat cleanup returned invalid evidence: " + completed.stdout.strip()
        ) from error
    return {
        "mode": "environment_only",
        "compat_path": str(compat),
        "removed_pth_paths": list(removed.get("pth_paths", [])),
        "removed_sitecustomize_paths": list(removed.get("sitecustomize_paths", [])),
    }


def _patch_upstream_transformers_import(env: Mapping[str, str]) -> dict[str, str]:
    source_root = env.get("LH_UPSTREAM_SOURCE_ROOT")
    if not source_root:
        return {}
    target = Path(source_root) / "verl/workers/fsdp_workers.py"
    if not target.is_file():
        return {}
    marker = "# LH_RUNTIME_COMPAT: transformers AutoModelForVision2Seq import fallback"
    attention_marker = "# LH_RUNTIME_COMPAT: configurable attention implementation"
    adamw_marker = "# LH_RUNTIME_COMPAT: configurable AdamW foreach"
    text = target.read_text(encoding="utf-8")
    changed = False
    old = (
        "        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq"
    )
    new = f"""        from transformers import AutoConfig, AutoModelForCausalLM
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            {marker}
            class AutoModelForVision2Seq:
                _model_mapping = {{}}
"""
    if marker not in text and old not in text:
        raise RuntimeError(f"upstream transformers import target not found: {target}")
    if marker not in text:
        text = text.replace(old, new, 1)
        changed = True

    attention_replacements = {
        'attn_implementation="flash_attention_2"': (
            f"{attention_marker}\n"
            '            attn_implementation=os.environ.get("LH_ATTENTION_IMPLEMENTATION", "flash_attention_2")'
        ),
        'attn_implementation="flash_attention_2",': (
            f"{attention_marker}\n"
            '            attn_implementation=os.environ.get("LH_ATTENTION_IMPLEMENTATION", "flash_attention_2"),'
        ),
    }
    if attention_marker not in text:
        for before, after in attention_replacements.items():
            if before in text:
                text = text.replace(before, after)
                changed = True
    adamw_before = """            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )"""
    adamw_after = f"""            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
                {adamw_marker}
                foreach=(os.environ.get("LH_TORCH_ADAMW_FOREACH", "true").lower() not in {{"0", "false", "no"}}),
            )"""
    if adamw_marker not in text:
        if adamw_before in text:
            text = text.replace(adamw_before, adamw_after, 1)
            changed = True
        elif "actor_optimizer = optim.AdamW(" in text:
            raise RuntimeError(f"upstream AdamW patch target changed: {target}")
    if changed:
        target.write_text(text, encoding="utf-8")
    return {"path": str(target), "status": "patched" if changed else "already_patched"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def backfill_trial_metrics(output_dir: str | Path) -> dict[str, Any]:
    """Reparse structured upstream metrics from durable stdout.

    This is intentionally an additive repair: the runner's live stream remains
    at ``metrics.jsonl`` and the reparsed rows are written to
    ``metrics_from_stdout.jsonl`` with line-level provenance.  CSV/SVG are
    regenerated from both sources so older runs stay visualizable after a
    parser improvement without pretending their original event stream changed.
    """
    output = Path(output_dir)
    logs = output / "logs"
    metrics_dir = output / "metrics"
    plots = output / "plots"
    stdout_path = logs / "stdout.log"
    primary_path = metrics_dir / "metrics.jsonl"
    reparsed_path = metrics_dir / "metrics_from_stdout.jsonl"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    reparsed: list[dict[str, Any]] = []
    if stdout_path.is_file():
        for line_number, line in enumerate(
            stdout_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            values = parse_metric_line(line)
            if values:
                reparsed.append(
                    {
                        **values,
                        "metric_source": "stdout_reparse",
                        "stdout_line": line_number,
                    }
                )
    reparsed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reparsed),
        encoding="utf-8",
    )
    merged = [*_read_jsonl(primary_path), *reparsed]
    _write_metrics_csv(merged, metrics_dir / "metrics.csv")
    _write_svg_curves(merged, plots / "training_curves.svg")
    return {
        "output_dir": str(output),
        "stdout": str(stdout_path),
        "primary_metrics": str(primary_path),
        "reparsed_metrics": str(reparsed_path),
        "parsed_metric_event_count": len(reparsed),
        "merged_metric_event_count": len(merged),
        "csv": str(metrics_dir / "metrics.csv"),
        "svg": str(plots / "training_curves.svg"),
    }


def backfill_study_metrics(
    prepared_dir: str | Path,
    *,
    trial_ids: Iterable[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Backfill every selected prepared-trial metric artifact without rerunning it."""
    root = Path(prepared_dir)
    selected = set(trial_ids)
    results: list[dict[str, Any]] = []
    for command in _read_jsonl(root / "planned_commands.jsonl"):
        trial_id = str(command.get("trial_id", ""))
        if selected and trial_id not in selected:
            continue
        output = command.get("output_dir")
        if isinstance(output, str) and output:
            result = backfill_trial_metrics(output)
            result["trial_id"] = trial_id
            results.append(result)
    return tuple(results)


def _jsonl(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    handle.flush()


def _write_metrics_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_svg_curves(rows: list[Mapping[str, Any]], path: Path) -> None:
    keys = [
        key
        for key in sorted({key for row in rows for key in row})
        if any(
            word in key.lower()
            for word in (
                "loss",
                "success",
                "reward",
                "kl",
                "gpu",
                "memory",
                "power",
                "temperature",
                "throughput",
                "valid",
                "length",
                "clip",
                "advantage",
                "returns",
            )
        )
    ]
    width, panel_h = 960, 220
    height = max(240, panel_h * max(1, len(keys)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
    ]
    for index, key in enumerate(keys):
        values = [
            (i, float(row[key]))
            for i, row in enumerate(rows)
            if isinstance(row.get(key), (int, float))
        ]
        if not values:
            continue
        top = index * panel_h + 28
        low, high = min(v for _, v in values), max(v for _, v in values)
        span = high - low or 1.0
        points = " ".join(
            f"{50 + i * 880 / max(1, len(rows) - 1):.1f},{top + 155 - (v - low) * 140 / span:.1f}"
            for i, v in values
        )
        parts.extend(
            [
                f'<text x="50" y="{top}" fill="#e5e7eb" font-family="monospace" font-size="14">{key}</text>',
                f'<polyline points="{points}" fill="none" stroke="#60a5fa" stroke-width="2"/>',
            ]
        )
    if not keys:
        parts.append(
            '<text x="40" y="80" fill="#e5e7eb" font-family="monospace">No parsed scalar metric curves yet.</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
