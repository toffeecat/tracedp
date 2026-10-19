"""Live, source-grounded progress snapshots for GraphGPO trials.

The upstream trainer has no structured progress stream.  This module therefore
only reports progress facts that can be tied to an upstream print, artifact, or
logged metric; it deliberately does not infer an in-flight rollout from a
post-update rollout dump.
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.studies.execution import _read_jsonl

GRAPHGPO_PROGRESS_SCHEMA = "lh_graphgpo_progress/v1"

__all__ = ["GRAPHGPO_PROGRESS_SCHEMA", "build_graphgpo_progress_snapshot"]
_DATALOADER_SIZE_RE = re.compile(
    r"Size of train dataloader:\s*(\d+),\s*Size of val dataloader:\s*(\d+)"
)
_VALIDATION_METRIC_RE = re.compile(r"(?m)\bstep:\s*\d+\s*-\s*val/")
_ROLLOUT_FILE_RE = re.compile(r"^(\d+)\.jsonl$")
_SEQUENTIAL_ROLLOUT_CHUNK_RE = re.compile(
    r"lh_rollout_chunk\s+phase=(?P<phase>train|validation)\s+"
    r"expected=(?P<expected>\d+)\s+completed=(?P<completed>\d+)\s+"
    r"current=(?P<current>\d+)\s+rollout_group=(?P<group_size>\d+)"
)
_SOURCE_QUOTES = (
    'print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")',
    "for test_data in self.val_dataloader:",
    "print('validation generation end')",
    "self._dump_generations(",
    '"training/global_step": self.global_steps,',
)
_SOURCE_PATH = Path("recipe/GraphGPO/graphgpo_ray_trainer.py")


def build_graphgpo_progress_snapshot(
    prepared_dir: str | Path,
    *,
    trial_id: str | None = None,
    output_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a live snapshot for one GraphGPO trial or every known trial.

    The selected public seam is the prepared-study artifact directory: the
    parser never alters the upstream checkout or a running training process.
    """
    root = Path(prepared_dir).resolve()
    planned = _planned_trials(root / "planned_commands.jsonl")
    state = _read_json(root / "study_run_state.json")
    latest = _latest_trial_states(state.get("trials", []))
    trial_ids = _selected_trial_ids(planned, latest, trial_id)
    trials = [
        _snapshot_trial(
            root,
            identifier,
            planned.get(identifier, {}),
            latest.get(identifier, {}),
        )
        for identifier in trial_ids
    ]
    source_contract = {
        "verified": bool(trials)
        and all(bool(item["source_contract"].get("verified")) for item in trials),
        "anchors": [{"path": str(_SOURCE_PATH), "quote": quote} for quote in _SOURCE_QUOTES],
    }
    if output_path is None:
        output_path = root / "graphgpo_progress.json"
    if markdown_path is None:
        markdown_path = root / "graphgpo_progress.md"
    json_target = Path(output_path)
    markdown_target = Path(markdown_path)
    payload = {
        "schema": GRAPHGPO_PROGRESS_SCHEMA,
        "generated_at": time.time(),
        "prepared_dir": str(root),
        "source_contract": source_contract,
        "trial_count": len(trials),
        "trials": trials,
        "artifact_paths": {"json": str(json_target), "markdown": str(markdown_target)},
    }
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_target.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _snapshot_trial(
    root: Path,
    trial_id: str,
    planned: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    output = _resolve_output_dir(root, state.get("output_dir") or planned.get("output_dir"))
    stdout_path = output / "logs/stdout.log"
    stdout = (
        stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    )
    metrics = _read_jsonl(output / "metrics/metrics.jsonl")
    overrides = planned.get("overrides", {})
    overrides = overrides if isinstance(overrides, Mapping) else {}
    expected_validation, expected_basis = _validation_expected_batches(stdout, overrides)
    completed_validation = stdout.count("validation generation end")
    completed_passes, current_pass = _validation_passes(completed_validation, expected_validation)
    validation_logged_passes = len(_VALIDATION_METRIC_RE.findall(stdout))
    rollout_steps = _rollout_steps(output, overrides)
    sequential_rollout = _sequential_rollout_progress(stdout)
    training_steps = _training_steps(metrics)
    completed_training = max(training_steps) if training_steps else None
    target_training = _positive_int(overrides.get("trainer.total_training_steps"))
    validation_active = bool(
        expected_validation and (current_pass > 0 or completed_passes > validation_logged_passes)
    )
    active_stage, diagnosis = _active_stage(
        status=str(state.get("status", "unknown")),
        validation_active=validation_active,
        rollout_steps=rollout_steps,
        completed_training=completed_training,
        target_training=target_training,
    )
    source_contract = _source_contract(planned)
    return {
        "trial_id": trial_id,
        "status": state.get("status", "not_started"),
        "output_dir": str(output),
        "source_contract": source_contract,
        "validation": {
            "expected_batches": expected_validation,
            "expected_basis": expected_basis,
            "completed_generation_batches": completed_validation,
            "completed_passes": completed_passes,
            "current_pass_completed_batches": current_pass,
            "logged_passes": validation_logged_passes,
        },
        "rollout": {
            "completed_steps": rollout_steps,
            "max_completed_step": max(rollout_steps) if rollout_steps else None,
        },
        "sequential_rollout": sequential_rollout,
        "training": {
            "completed_steps": training_steps,
            "completed_step": completed_training,
            "target_steps": target_training,
        },
        "active_stage": active_stage,
        "diagnosis": diagnosis,
        "limitations": [
            "rollouts/<step>.jsonl is not an in-flight rollout marker: upstream writes it after actor update.",
            "The upstream stdout has no boundary between rollout collection and actor-update execution.",
        ],
    }


def _selected_trial_ids(
    planned: Mapping[str, Mapping[str, Any]],
    latest: Mapping[str, Mapping[str, Any]],
    trial_id: str | None,
) -> list[str]:
    if trial_id is not None:
        if trial_id not in planned and trial_id not in latest:
            raise ValueError(f"unknown trial id: {trial_id}")
        return [trial_id]
    return sorted(set(planned) | set(latest))


def _planned_trials(path: Path) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["trial_id"]): item
        for item in _read_jsonl(path)
        if isinstance(item.get("trial_id"), str)
    }


def _latest_trial_states(raw_trials: Any) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for item in raw_trials if isinstance(raw_trials, list) else []:
        if isinstance(item, Mapping) and isinstance(item.get("trial_id"), str):
            latest[str(item["trial_id"])] = item
    return latest


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _resolve_output_dir(root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else root / path


def _validation_expected_batches(
    stdout: str,
    overrides: Mapping[str, Any],
) -> tuple[int | None, str]:
    matches = _DATALOADER_SIZE_RE.findall(stdout)
    if matches:
        return int(matches[-1][1]), "upstream_runtime_dataloader_size"
    rows = _parquet_row_count(overrides.get("data.val_files"))
    if rows is None:
        return None, "unavailable"
    batch_size = _positive_int(overrides.get("data.val_batch_size"))
    if batch_size is None:
        return 1, "prepared_val_rows_default_whole_dataset"
    return math.ceil(rows / batch_size), "prepared_val_rows_and_batch_size"


def _parquet_row_count(raw: Any) -> int | None:
    path = _single_path_value(raw)
    if path is None or not path.is_file() or path.suffix != ".parquet":
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _single_path_value(raw: Any) -> Path | None:
    if isinstance(raw, str):
        value = raw.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return None
            return _single_path_value(parsed)
        return Path(value) if value else None
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        return Path(raw[0])
    return None


def _validation_passes(completed: int, expected: int | None) -> tuple[int | None, int | None]:
    if expected is None or expected <= 0:
        return None, None
    return divmod(completed, expected)


def _rollout_steps(output: Path, overrides: Mapping[str, Any]) -> list[int]:
    raw_dir = overrides.get("trainer.rollout_data_dir")
    directory = Path(str(raw_dir)) if raw_dir else output / "rollouts"
    if not directory.is_dir():
        return []
    return sorted(
        int(match.group(1))
        for child in directory.iterdir()
        if child.is_file() and (match := _ROLLOUT_FILE_RE.match(child.name))
    )


def _sequential_rollout_progress(stdout: str) -> dict[str, dict[str, int] | None]:
    """Return the last durable chunk boundary emitted by the bounded runtime.

    The adapter writes the line immediately before a chunk begins.  Therefore
    ``completed_groups`` only counts chunks whose following boundary was reached;
    ``current_groups`` is a live in-flight amount, never a success claim.
    """

    latest: dict[str, dict[str, int] | None] = {"train": None, "validation": None}
    for match in _SEQUENTIAL_ROLLOUT_CHUNK_RE.finditer(stdout):
        phase = match.group("phase")
        latest[phase] = {
            "expected_groups": int(match.group("expected")),
            "completed_groups": int(match.group("completed")),
            "current_groups": int(match.group("current")),
            "rollout_group_size": int(match.group("group_size")),
        }
    return latest


def _training_steps(metrics: list[Mapping[str, Any]]) -> list[int]:
    steps = {
        int(value)
        for row in metrics
        for value in (row.get("training/global_step"),)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
    }
    return sorted(steps)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _active_stage(
    *,
    status: str,
    validation_active: bool,
    rollout_steps: list[int],
    completed_training: int | None,
    target_training: int | None,
) -> tuple[str, str]:
    if status in {"completed", "failed", "blocked"}:
        return "terminal", "trial_not_running"
    if validation_active:
        return "validation", "validation_in_progress"
    max_rollout = max(rollout_steps) if rollout_steps else None
    if max_rollout is not None and (completed_training is None or max_rollout > completed_training):
        return "post_update_pre_metric", "waiting_for_training_metric"
    if (
        completed_training is not None
        and target_training is not None
        and completed_training >= target_training
    ):
        return "metric_written_pending_exit", "waiting_for_trial_exit"
    return "between_steps", "no_inflight_boundary_from_upstream"


def _source_contract(planned: Mapping[str, Any]) -> dict[str, Any]:
    source_root = Path(str(planned.get("cwd", "")))
    source_path = source_root / _SOURCE_PATH
    if not source_path.is_file():
        return {
            "verified": False,
            "source_root": str(source_root),
            "blocked_by": ["upstream_source_missing"],
        }
    text = source_path.read_text(encoding="utf-8", errors="replace")
    missing = [quote for quote in _SOURCE_QUOTES if quote not in text]
    return {
        "verified": not missing,
        "source_root": str(source_root),
        "source_path": str(_SOURCE_PATH),
        "blocked_by": ["source_quote_missing"] if missing else [],
        "missing_quotes": missing,
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# GraphGPO live progress",
        "",
        f"- source contract verified: `{str(payload['source_contract']['verified']).lower()}`",
        "",
        "## Official upstream anchors",
        "",
    ]
    for anchor in payload["source_contract"]["anchors"]:
        lines.append(f"- `{anchor['path']}`: `{anchor['quote']}`")
    lines.append("")
    for trial in payload["trials"]:
        validation = trial["validation"]
        rollout = trial["rollout"]
        training = trial["training"]
        sequential = trial["sequential_rollout"]
        lines.extend(
            [
                f"## {trial['trial_id']}",
                "",
                f"- status: `{trial['status']}`",
                f"- active stage: `{trial['active_stage']}`",
                f"- diagnosis: `{trial['diagnosis']}`",
                f"- validation expected/completed: `{validation['expected_batches']}` / "
                f"`{validation['completed_generation_batches']}` batches; current pass "
                f"`{validation['current_pass_completed_batches']}`",
                f"- rollout completed steps: `{rollout['completed_steps']}`",
                f"- sequential validation groups completed/current/expected: "
                f"`{_render_sequential_groups(sequential['validation'])}`",
                f"- sequential train groups completed/current/expected: "
                f"`{_render_sequential_groups(sequential['train'])}`",
                f"- training completed/target: `{training['completed_step']}` / `{training['target_steps']}`",
                "- limitation: rollout artifact proves a completed update path, not an in-flight rollout boundary.",
                "",
            ]
        )
    return "\n".join(lines)


def _render_sequential_groups(progress: Mapping[str, int] | None) -> str:
    if progress is None:
        return "unavailable"
    return (
        f"{progress['completed_groups']}/{progress['current_groups']}/{progress['expected_groups']}"
    )
