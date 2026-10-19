"""Build fail-closed audit reports from prepared-study run artifacts."""

from __future__ import annotations

import ast
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.studies.execution import (
    _read_jsonl,
    _scan_stdout_diagnostics,
    _write_metrics_csv,
    _write_svg_curves,
)

RUN_AUDIT_SCHEMA = "lh_study_run_audit/v1"
CORE_ARTIFACTS = {
    "stdout": "logs/stdout.log",
    "events": "logs/events.jsonl",
    "metrics_jsonl": "metrics/metrics.jsonl",
    "metrics_csv": "metrics/metrics.csv",
    "training_curves_svg": "plots/training_curves.svg",
}
METRIC_SUMMARY_KEYS = (
    "training/global_step",
    "step",
    "val/success_rate",
    "episode/success_rate",
    "episode/reward/mean",
    "episode/valid_action_ratio",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "critic/returns/mean",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/grad_norm",
    "perf/time_per_step",
    "perf/throughput",
)
GRAPHGPO_STDOUT_PROGRESS_SOURCE = {
    "source_path": "verl/trainer/ppo/ray_trainer.py",
    "quotes": (
        'print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")',
        "print('validation generation end')",
    ),
}
EVIDENCE_INTEGRITY_BLOCKERS = frozenset(
    {
        "no_trials_found",
        "trial_not_completed",
        "core_artifacts_missing",
        "metrics_missing",
        "training_metrics_missing",
    }
)
RESOURCE_COMPATIBILITY_BLOCKERS = frozenset(
    {
        "gpu_count_insufficient",
        "gpu_inventory_unavailable",
        "formal_1p5b_requires_at_least_2_gpus",
    }
)
COMPARISON_BLOCKING_BLOCKERS = (
    EVIDENCE_INTEGRITY_BLOCKERS
    | RESOURCE_COMPATIBILITY_BLOCKERS
    | frozenset({"paper_reproduction_claim_not_supported_by_calibration_profile"})
)


def build_study_run_audit(
    prepared_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize run artifacts and state what evidence they can support."""
    root = Path(prepared_dir).resolve()
    state = _read_json(root / "study_run_state.json")
    prepare_summary = _read_json(root / "prepare_summary.json")
    runtime_report = _read_json(root / "runtime_validity_report.json")
    planned_trials = _planned_trials(root / "planned_commands.jsonl")
    trial_attempts = _latest_trial_attempts(state.get("trials", []))
    trials = [
        _audit_trial(
            root,
            item,
            attempt_count=count,
            planned=planned_trials.get(str(item.get("trial_id", "")), {}),
        )
        for item, count in trial_attempts
    ]
    decision, blockers = _audit_decision(
        trials,
        prepare_summary=prepare_summary,
        runtime_report=runtime_report,
    )
    if output_path is None:
        output_path = root / "study_run_audit.json"
    if markdown_path is None:
        markdown_path = root / "study_run_audit.md"
    json_target = Path(output_path)
    md_target = Path(markdown_path)
    payload: dict[str, Any] = {
        "schema": RUN_AUDIT_SCHEMA,
        "generated_at": time.time(),
        "prepared_dir": str(root),
        "decision": decision,
        "blocked_by": blockers,
        "artifact_paths": {
            "json": str(json_target),
            "markdown": str(md_target),
        },
        "prepare": {
            "launch_allowed": prepare_summary.get("launch_allowed"),
            "blocked_by": list(prepare_summary.get("blocked_by", []))
            if isinstance(prepare_summary.get("blocked_by"), list)
            else [],
        },
        "runtime": {
            "run_ready": runtime_report.get("run_ready"),
            "blocked_by": list(runtime_report.get("blocked_by", []))
            if isinstance(runtime_report.get("blocked_by"), list)
            else [],
        },
        "trial_count": len(trials),
        "trials": trials,
        "comparison_summary": _comparison_summary(trials),
        "formal_comparison_readiness": _formal_comparison_readiness(
            root,
            planned_trials=planned_trials,
            trials=trials,
        ),
    }
    json_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_target.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _audit_trial(
    root: Path,
    state_item: Mapping[str, Any],
    *,
    attempt_count: int,
    planned: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = _resolve_output_dir(root, str(state_item.get("output_dir", "")))
    _materialize_metric_artifacts(output_dir)
    artifacts = {
        name: {
            "path": str(output_dir / relative),
            "exists": (output_dir / relative).is_file(),
        }
        for name, relative in CORE_ARTIFACTS.items()
    }
    events = _read_jsonl(output_dir / "logs/events.jsonl")
    metrics = [
        *_read_jsonl(output_dir / "metrics/metrics.jsonl"),
        *_read_jsonl(output_dir / "metrics/metrics_from_stdout.jsonl"),
    ]
    metric_summary = _metric_summary(metrics)
    diagnostics = _postrun_diagnostics(events, output_dir / "logs/stdout.log")
    progress_summary = _stdout_progress_summary(output_dir / "logs/stdout.log")
    if progress_summary:
        progress_summary = {
            **progress_summary,
            **_validation_progress_denominator(planned, progress_summary),
        }
        progress_summary = {
            **progress_summary,
            **_validation_progress_timing(state_item, progress_summary),
        }
    resource_gate = _resource_gate_diagnostics(events)
    missing = [name for name, item in artifacts.items() if not item["exists"]]
    claim_scope = _claim_scope(planned, output_dir, events)
    blockers: list[str] = []
    if state_item.get("status") != "completed" or state_item.get("returncode") != 0:
        blockers.append("trial_not_completed")
    blockers.extend(resource_gate.get("blocked_by", []))
    if missing:
        blockers.append("core_artifacts_missing")
    if not metrics:
        blockers.append("metrics_missing")
    elif not metric_summary:
        blockers.append("training_metrics_missing")
    if diagnostics.get("warning_count", 0):
        blockers.append("stdout_postrun_warnings")
    if claim_scope == "calibration_only":
        blockers.append("paper_reproduction_claim_not_supported_by_calibration_profile")
    return {
        "trial_id": state_item.get("trial_id"),
        "variant_id": planned.get("variant_id")
        or _trial_variant_id(str(state_item.get("trial_id", ""))),
        "seed": planned.get("seed")
        if isinstance(planned.get("seed"), int)
        else _trial_seed(str(state_item.get("trial_id", ""))),
        "role": planned.get("role"),
        "status": state_item.get("status"),
        "returncode": state_item.get("returncode"),
        "elapsed_s": state_item.get("elapsed_s"),
        "attempt_count": attempt_count,
        "output_dir": str(output_dir),
        "claim_scope": claim_scope,
        "blockers": blockers,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "diagnostics": diagnostics,
        "progress_summary": progress_summary,
        "resource_gate": resource_gate,
        "metric_event_count": len(metrics),
        "metric_summary": metric_summary,
    }


def _latest_trial_attempts(raw_trials: Any) -> list[tuple[Mapping[str, Any], int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in raw_trials if isinstance(raw_trials, list) else []:
        if not isinstance(item, Mapping):
            continue
        trial_id = str(item.get("trial_id", ""))
        if not trial_id:
            continue
        grouped.setdefault(trial_id, []).append(item)
    return [(attempts[-1], len(attempts)) for attempts in grouped.values()]


def _materialize_metric_artifacts(output_dir: Path) -> None:
    metrics_jsonl = output_dir / "metrics/metrics.jsonl"
    reparsed_jsonl = output_dir / "metrics/metrics_from_stdout.jsonl"
    rows = [*_read_jsonl(metrics_jsonl), *_read_jsonl(reparsed_jsonl)]
    if not rows:
        return
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(rows, metrics_dir / "metrics.csv")
    _write_svg_curves(rows, plots_dir / "training_curves.svg")


def _stdout_progress_summary(stdout_path: Path) -> dict[str, Any]:
    if not stdout_path.is_file():
        return {}
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    counts = {
        "validation_meta_info_events": text.count("test_gen_batch meta info:"),
        "validation_generation_end_events": text.count("validation generation end"),
        "text_score_events": text.count("[text][score]"),
    }
    if not any(counts.values()):
        return {}
    return {
        "source_basis": GRAPHGPO_STDOUT_PROGRESS_SOURCE,
        **counts,
    }


def _validation_progress_denominator(
    planned: Mapping[str, Any],
    progress_summary: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _planned_val_row_count(planned)
    if expected is None or expected <= 0:
        return {}
    completed = progress_summary.get("validation_generation_end_events")
    if not isinstance(completed, int):
        return {"validation_expected_events": expected}
    completed_passes, current_pass_events = divmod(completed, expected)
    return {
        "validation_expected_events": expected,
        "validation_completed_passes": completed_passes,
        "validation_current_pass_events": current_pass_events,
        "validation_progress_fraction": current_pass_events / expected,
        "validation_cumulative_progress_fraction": completed / expected,
    }


def _validation_progress_timing(
    state_item: Mapping[str, Any],
    progress_summary: Mapping[str, Any],
) -> dict[str, Any]:
    completed = progress_summary.get("validation_generation_end_events")
    if not isinstance(completed, int) or completed <= 0:
        return {}
    started_at = state_item.get("started_at")
    if not isinstance(started_at, (int, float)) or isinstance(started_at, bool):
        return {}
    elapsed_s = max(0.0, time.time() - float(started_at))
    if elapsed_s <= 0:
        return {}
    events_per_hour = completed / elapsed_s * 3600.0
    timing: dict[str, Any] = {
        "validation_elapsed_s": elapsed_s,
        "validation_events_per_hour": events_per_hour,
    }
    expected = progress_summary.get("validation_expected_events")
    current_pass_events = progress_summary.get("validation_current_pass_events")
    if (
        isinstance(expected, int)
        and isinstance(current_pass_events, int)
        and expected > current_pass_events
        and events_per_hour > 0
    ):
        timing["validation_eta_remaining_s"] = (
            (expected - current_pass_events) / events_per_hour * 3600.0
        )
    return timing


def _planned_val_row_count(planned: Mapping[str, Any]) -> int | None:
    overrides = planned.get("overrides", {})
    if not isinstance(overrides, Mapping):
        return None
    raw = overrides.get("data.val_files")
    path = _single_path_value(raw)
    if path is None or path.suffix != ".parquet" or not path.is_file():
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
    if isinstance(raw, Path):
        return raw
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
                return Path(parsed[0])
        return Path(value)
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        return Path(raw[0])
    return None


def _planned_trials(path: Path) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["trial_id"]): item
        for item in _read_jsonl(path)
        if isinstance(item.get("trial_id"), str)
    }


def _audit_decision(
    trials: list[Mapping[str, Any]],
    *,
    prepare_summary: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if not trials:
        blockers.append("no_trials_found")
    if prepare_summary.get("launch_allowed") is False or runtime_report.get("run_ready") is False:
        blockers.append("prepared_study_not_launch_ready")
    for trial in trials:
        blockers.extend(str(item) for item in trial.get("blockers", []))
    if any(trial.get("claim_scope") == "resource_constrained_ablation" for trial in trials):
        blockers.append("paper_reproduction_claim_not_supported_by_resource_constrained_profile")
    non_calibration_variants = {
        _trial_variant_id(str(trial.get("trial_id", "")))
        for trial in trials
        if trial.get("claim_scope") != "calibration_only"
        and trial.get("status") == "completed"
        and trial.get("returncode") == 0
    }
    non_calibration_variants.discard("")
    if trials and non_calibration_variants and len(non_calibration_variants) < 2:
        blockers.append("comparison_trials_missing")
    blockers = list(dict.fromkeys(blockers))
    if any(item in EVIDENCE_INTEGRITY_BLOCKERS for item in blockers):
        return "requires_diagnosis", blockers
    if "paper_reproduction_claim_not_supported_by_calibration_profile" in blockers:
        return "calibration_evidence_only", blockers
    if "paper_reproduction_claim_not_supported_by_resource_constrained_profile" in blockers:
        return "resource_constrained_evidence_only", blockers
    if "stdout_postrun_warnings" in blockers:
        return "completed_with_warnings", blockers
    if "comparison_trials_missing" in blockers:
        return "single_trial_evidence_only", blockers
    return "ready_for_outcome_analysis", blockers


def _trial_variant_id(trial_id: str) -> str:
    parts = trial_id.split(":")
    if len(parts) >= 3:
        return parts[-2]
    return trial_id


def _trial_seed(trial_id: str) -> int | None:
    raw = trial_id.split(":")[-1] if ":" in trial_id else ""
    if raw.startswith("seed-"):
        try:
            return int(raw.removeprefix("seed-"))
        except ValueError:
            return None
    return None


def _comparison_summary(trials: list[Mapping[str, Any]]) -> dict[str, Any]:
    comparison_eligible = [
        trial
        for trial in trials
        if trial.get("status") == "completed"
        and trial.get("returncode") == 0
        and isinstance(trial.get("metric_summary"), Mapping)
        and trial.get("metric_summary")
        and trial.get("claim_scope") != "calibration_only"
        and not _comparison_blocking_blockers(trial)
    ]
    scopes = sorted(
        {
            str(trial.get("claim_scope", ""))
            for trial in comparison_eligible
            if trial.get("claim_scope")
        }
    )
    claim_scope = scopes[0] if len(scopes) == 1 else "mixed_or_absent"
    variants = sorted(
        {
            str(trial.get("variant_id", ""))
            for trial in comparison_eligible
            if trial.get("variant_id")
        }
    )
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for trial in comparison_eligible:
        seed = trial.get("seed")
        variant = trial.get("variant_id")
        if not isinstance(seed, int) or not isinstance(variant, str) or not variant:
            continue
        by_seed.setdefault(seed, {})[variant] = trial
    paired_seeds = [
        seed
        for seed, variants_for_seed in sorted(by_seed.items())
        if "graphgpo" in variants_for_seed and "grpo" in variants_for_seed
    ]
    pairwise_deltas: list[dict[str, Any]] = []
    for seed in paired_seeds:
        variants_for_seed = by_seed[seed]
        if "graphgpo" in variants_for_seed and "grpo" in variants_for_seed:
            pairwise_deltas.append(
                _metric_delta(
                    seed=seed,
                    treatment="graphgpo",
                    baseline="grpo",
                    treatment_trial=variants_for_seed["graphgpo"],
                    baseline_trial=variants_for_seed["grpo"],
                )
            )
    if pairwise_deltas and claim_scope == "resource_constrained_ablation":
        purpose = "resource_constrained_method_vs_baseline"
    else:
        purpose = "method_vs_baseline" if pairwise_deltas else "unpaired_run_inventory"
    return {
        "purpose": purpose,
        "claim_scope": claim_scope,
        "variants": variants,
        "paired_seeds": paired_seeds,
        "pairwise_deltas": pairwise_deltas,
    }


def _comparison_blocking_blockers(trial: Mapping[str, Any]) -> list[str]:
    blockers = [str(item) for item in trial.get("blockers", []) if isinstance(item, str)]
    return [blocker for blocker in blockers if blocker in COMPARISON_BLOCKING_BLOCKERS]


def _formal_comparison_readiness(
    root: Path,
    *,
    planned_trials: Mapping[str, Mapping[str, Any]],
    trials: list[Mapping[str, Any]],
) -> dict[str, Any]:
    clean_formal = {
        str(trial.get("trial_id")): trial
        for trial in trials
        if trial.get("status") == "completed"
        and trial.get("returncode") == 0
        and trial.get("claim_scope") == "paper_reproduction_formal"
        and not trial.get("blockers")
        and isinstance(trial.get("metric_summary"), Mapping)
        and trial.get("metric_summary")
    }
    planned_by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for planned in planned_trials.values():
        planned_scope = planned.get("claim_scope", "paper_reproduction_formal")
        if planned_scope != "paper_reproduction_formal":
            continue
        seed = planned.get("seed")
        variant = planned.get("variant_id")
        if not isinstance(seed, int) or variant not in {"graphgpo", "grpo"}:
            continue
        planned_by_seed.setdefault(seed, {})[str(variant)] = planned

    pair_statuses: list[dict[str, Any]] = []
    missing_trial_ids: list[str] = []
    clean_trial_ids: list[str] = []
    for seed, variants in sorted(planned_by_seed.items()):
        if not {"graphgpo", "grpo"}.issubset(variants):
            continue
        expected = {variant: str(variants[variant]["trial_id"]) for variant in ("graphgpo", "grpo")}
        clean = {
            variant: trial_id for variant, trial_id in expected.items() if trial_id in clean_formal
        }
        missing = [trial_id for trial_id in expected.values() if trial_id not in clean_formal]
        clean_trial_ids.extend(clean.values())
        missing_trial_ids.extend(missing)
        pair_statuses.append(
            {
                "seed": seed,
                "expected_trial_ids": expected,
                "clean_trial_ids": clean,
                "missing_trial_ids": missing,
                "ready": not missing,
            }
        )

    ready = bool(pair_statuses) and all(pair["ready"] for pair in pair_statuses)
    next_missing_pair = next((pair for pair in pair_statuses if not pair["ready"]), None)
    next_trial_ids = list(next_missing_pair["missing_trial_ids"]) if next_missing_pair else []
    run_commands = [
        f"PYTHONPATH=lh_platform python -m lh_platform.cli run-study {root} --trial-id {trial_id}"
        for trial_id in next_trial_ids
    ]
    gate_command = (
        f"PYTHONPATH=lh_platform python -m lh_platform.cli run-study {root} --resume --require-audit-ready"
        if pair_statuses
        else ""
    )
    blockers: list[str] = []
    if not pair_statuses:
        blockers.append("formal_graphgpo_grpo_pairs_not_planned")
    if missing_trial_ids:
        blockers.append("formal_pair_trials_missing")
    return {
        "ready": ready,
        "blocked_by": blockers,
        "target_variants": ["graphgpo", "grpo"],
        "planned_pair_count": len(pair_statuses),
        "pair_statuses": pair_statuses,
        "clean_trial_ids": clean_trial_ids,
        "missing_trial_ids": missing_trial_ids,
        "next_trial_ids": next_trial_ids,
        "next_run_commands": run_commands,
        "post_run_gate_command": gate_command,
    }


def _metric_delta(
    *,
    seed: int,
    treatment: str,
    baseline: str,
    treatment_trial: Mapping[str, Any],
    baseline_trial: Mapping[str, Any],
) -> dict[str, Any]:
    treatment_metrics = treatment_trial.get("metric_summary", {})
    baseline_metrics = baseline_trial.get("metric_summary", {})
    deltas: dict[str, float] = {}
    if isinstance(treatment_metrics, Mapping) and isinstance(baseline_metrics, Mapping):
        for key in METRIC_SUMMARY_KEYS:
            treatment_value = treatment_metrics.get(key)
            baseline_value = baseline_metrics.get(key)
            if isinstance(treatment_value, (int, float)) and isinstance(
                baseline_value, (int, float)
            ):
                deltas[key] = float(treatment_value) - float(baseline_value)
    return {
        "seed": seed,
        "treatment": treatment,
        "baseline": baseline,
        "metric_deltas": deltas,
        "treatment_trial_id": treatment_trial.get("trial_id"),
        "baseline_trial_id": baseline_trial.get("trial_id"),
        "caveats": sorted(
            {
                str(item)
                for trial in (treatment_trial, baseline_trial)
                for item in trial.get("blockers", [])
                if isinstance(item, str)
            }
        ),
    }


def _resolve_output_dir(root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = (path, root / path, root.parent / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (root / path).resolve()


def _postrun_diagnostics(events: list[Mapping[str, Any]], stdout_path: Path) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event") == "postrun_log_diagnostics":
            return {
                "warning_count": int(event.get("warning_count", 0)),
                "warning_counts": dict(event.get("warning_counts", {})),
                "warnings": list(event.get("warnings", [])),
            }
    return _scan_stdout_diagnostics(stdout_path)


def _resource_gate_diagnostics(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event") != "formal_resource_gate":
            continue
        return {
            "decision": str(event.get("decision", "")),
            "blocked_by": [
                str(item) for item in event.get("blocked_by", []) if isinstance(item, str)
            ],
            "requirements": dict(event.get("requirements", {}))
            if isinstance(event.get("requirements"), Mapping)
            else {},
            "inventory": dict(event.get("inventory", {}))
            if isinstance(event.get("inventory"), Mapping)
            else {},
        }
    return {}


def _claim_scope(
    planned: Mapping[str, Any],
    output_dir: Path,
    events: list[Mapping[str, Any]],
) -> str:
    if _is_calibration_run(output_dir, events):
        return "calibration_only"
    planned_scope = planned.get("claim_scope")
    if planned_scope == "resource_constrained_ablation":
        return "resource_constrained_ablation"
    return "paper_reproduction_formal"


def _is_calibration_run(output_dir: Path, events: list[Mapping[str, Any]]) -> bool:
    if "calibration" in output_dir.name:
        return True
    return any(event.get("event") == "calibration_runtime_profile" for event in events)


def _metric_summary(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in METRIC_SUMMARY_KEYS:
        for row in reversed(rows):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                summary[key] = float(value)
                break
    return summary


def _format_progress_fraction(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    return "(unknown)"


def _format_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "(unknown)"
    seconds = max(0, int(round(float(value))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, Mapping) else {}


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Study run audit",
        "",
        f"- decision: `{payload['decision']}`",
        f"- prepared dir: `{payload['prepared_dir']}`",
        f"- trial count: `{payload['trial_count']}`",
        f"- blocked by: `{', '.join(payload['blocked_by']) or '(none)'}`",
        "",
        "## Comparison summary",
        "",
    ]
    comparison = payload.get("comparison_summary", {})
    if isinstance(comparison, Mapping):
        lines.extend(
            [
                f"- purpose: `{comparison.get('purpose')}`",
                f"- variants: `{', '.join(comparison.get('variants', [])) or '(none)'}`",
                f"- paired seeds: `{', '.join(str(seed) for seed in comparison.get('paired_seeds', [])) or '(none)'}`",
                "",
            ]
        )
        for delta in comparison.get("pairwise_deltas", []):
            if not isinstance(delta, Mapping):
                continue
            caveats = [str(item) for item in delta.get("caveats", []) if isinstance(item, str)]
            lines.extend(
                [
                    f"### {delta.get('treatment')} vs {delta.get('baseline')} seed {delta.get('seed')}",
                    "",
                    f"- caveats: `{', '.join(caveats) or '(none)'}`",
                    "",
                    "| metric | delta |",
                    "| --- | ---: |",
                ]
            )
            metric_deltas = delta.get("metric_deltas", {})
            if isinstance(metric_deltas, Mapping) and metric_deltas:
                for key, value in metric_deltas.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"| `{key}` | {value:.6g} |")
            else:
                lines.append("| `(none)` |  |")
            lines.append("")
    readiness = payload.get("formal_comparison_readiness", {})
    lines.extend(["## Formal comparison readiness", ""])
    if isinstance(readiness, Mapping):
        lines.extend(
            [
                f"- ready: `{str(readiness.get('ready')).lower()}`",
                f"- blocked by: `{', '.join(readiness.get('blocked_by', [])) or '(none)'}`",
                f"- planned pair count: `{readiness.get('planned_pair_count')}`",
                f"- missing trial count: `{len(readiness.get('missing_trial_ids', []))}`",
                "",
            ]
        )
        commands = readiness.get("next_run_commands", [])
        if commands:
            lines.extend(["### Next run commands", ""])
            for command in commands:
                lines.append(f"```bash\n{command}\n```")
            gate = readiness.get("post_run_gate_command")
            if gate:
                lines.extend(["### Post-run gate", "", f"```bash\n{gate}\n```", ""])
    lines.extend(
        [
            "## Trials",
            "",
        ]
    )
    for trial in payload.get("trials", []):
        metrics = trial.get("metric_summary", {})
        lines.extend(
            [
                f"### {trial.get('trial_id')}",
                "",
                f"- status: `{trial.get('status')}`",
                f"- claim scope: `{trial.get('claim_scope')}`",
                f"- output: `{trial.get('output_dir')}`",
                f"- blockers: `{', '.join(trial.get('blockers', [])) or '(none)'}`",
                f"- warnings: `{trial.get('diagnostics', {}).get('warning_count', 0)}`",
                f"- metric events: `{trial.get('metric_event_count')}`",
                "",
            ]
        )
        progress = trial.get("progress_summary", {})
        if isinstance(progress, Mapping) and progress:
            lines.extend(
                [
                    "#### Progress summary",
                    "",
                    "| signal | count |",
                    "| --- | ---: |",
                    f"| `validation_meta_info_events` | {progress.get('validation_meta_info_events', 0)} |",
                    f"| `validation_generation_end_events` | {progress.get('validation_generation_end_events', 0)} |",
                    f"| `validation_expected_events` | {progress.get('validation_expected_events', '(unknown)')} |",
                    f"| `validation_completed_passes` | {progress.get('validation_completed_passes', '(unknown)')} |",
                    f"| `validation_current_pass_events` | {progress.get('validation_current_pass_events', '(unknown)')} |",
                    f"| `validation_progress_fraction` | {_format_progress_fraction(progress.get('validation_progress_fraction'))} |",
                    f"| `validation_cumulative_progress_fraction` | {_format_progress_fraction(progress.get('validation_cumulative_progress_fraction'))} |",
                    f"| `validation_elapsed` | {_format_seconds(progress.get('validation_elapsed_s'))} |",
                    f"| `validation_events_per_hour` | {_format_progress_fraction(progress.get('validation_events_per_hour'))} |",
                    f"| `validation_eta_remaining` | {_format_seconds(progress.get('validation_eta_remaining_s'))} |",
                    f"| `text_score_events` | {progress.get('text_score_events', 0)} |",
                    "",
                ]
            )
        lines.extend(
            [
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        if metrics:
            for key, value in metrics.items():
                lines.append(f"| `{key}` | {value:.6g} |")
        else:
            lines.append("| `(none)` |  |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
