"""Multi-checkpoint paired-prefix audit for horizon-dependent aliasing.

The analyzer consumes frozen per-checkpoint rows produced by
``natural_aliasing.materialize_natural_aliasing_rows``.  It never samples a
policy.  Its three-panel decomposition separates cumulative exposure from the
opportunity to revisit a coarse state and from aliasing conditional on that
opportunity.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from lh_platform.core.atomic_io import atomic_write_json, atomic_write_text
from lh_platform.training.dataset import sha256_file
from lh_platform.training.natural_aliasing import NATURAL_ROWS_SCHEMA

PAIRED_PREFIX_HORIZON_CONFIG_SCHEMA = "lh_paired_prefix_horizon_config/v1"
PAIRED_PREFIX_HORIZON_REPORT_SCHEMA = "lh_paired_prefix_horizon_report/v1"
COMPETENCE_BANDS = {
    "weak": (0.1, 0.25),
    "medium": (0.25, 0.4),
    "strong": (0.4, 0.6),
}


class PairedPrefixHorizonError(ValueError):
    """Raised when a purported formal G1 study is not actually paired."""


def analyze_paired_prefix_horizon_study(
    config_path: str | Path,
    *,
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Validate three frozen checkpoint strata and render the G1 mechanism chart."""

    config_file = Path(config_path).resolve()
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config = _validate_config(raw)
    benchmark = str(config["benchmark"])
    bindings = []
    checkpoint_reports = []
    seen_checkpoint_ids: set[str] = set()
    seen_model_hashes: set[str] = set()
    for binding in config["checkpoints"]:
        checkpoint_id = str(binding["checkpoint_id"])
        if checkpoint_id in seen_checkpoint_ids:
            raise PairedPrefixHorizonError("checkpoint ids must be unique")
        seen_checkpoint_ids.add(checkpoint_id)
        rows_path = _resolve(config_file, binding["rows_path"])
        expected_sha = str(binding["rows_sha256"])
        if not rows_path.is_file() or sha256_file(rows_path) != expected_sha:
            raise PairedPrefixHorizonError(f"paired-prefix rows hash mismatch: {checkpoint_id}")
        payload = _load_mapping(rows_path)
        if payload.get("schema") != NATURAL_ROWS_SCHEMA:
            raise PairedPrefixHorizonError("unsupported natural-aliasing rows schema")
        if tuple(payload.get("horizons", ())) != tuple(config["horizons"]):
            raise PairedPrefixHorizonError("checkpoint horizons differ from G1 config")
        collection = payload.get("collection_binding")
        if not isinstance(collection, Mapping):
            raise PairedPrefixHorizonError("natural rows lack collection lineage")
        if int(collection.get("source_rollout_horizon", -1)) != 32:
            raise PairedPrefixHorizonError(
                "formal G1 rows must derive from one frozen H=32 collection"
            )
        model_sha = str(collection.get("model_checkpoint_sha256", ""))
        if len(model_sha) != 64 or model_sha in seen_model_hashes:
            raise PairedPrefixHorizonError(
                "formal G1 requires three distinct model checkpoint hashes"
            )
        seen_model_hashes.add(model_sha)
        observed_competence = _competence_evidence(
            collection=collection,
            benchmark=benchmark,
            declared_band=str(binding["competence_band"]),
        )
        report = _checkpoint_report(
            checkpoint_id=checkpoint_id,
            competence_band=str(binding["competence_band"]),
            rows=[
                row
                for row in payload.get("rows", ())
                if isinstance(row, Mapping) and row.get("benchmark") == benchmark
            ],
            horizons=tuple(config["horizons"]),
            minimum_trajectories=int(config["minimum_trajectories_per_checkpoint"]),
            minimum_task_clusters=int(config["minimum_task_clusters_per_checkpoint"]),
            bootstrap_resamples=int(config["bootstrap_resamples"]),
            bootstrap_seed=int(config["bootstrap_seed"]),
        )
        checkpoint_reports.append(report)
        report["observed_competence"] = observed_competence
        bindings.append(
            {
                "checkpoint_id": checkpoint_id,
                "competence_band": str(binding["competence_band"]),
                "rows_path": str(rows_path),
                "rows_sha256": expected_sha,
                "model_checkpoint_sha256": model_sha,
                "observed_competence": observed_competence,
                "collection_binding": dict(collection),
            }
        )
    bands = {report["competence_band"] for report in checkpoint_reports}
    if bands != {"weak", "medium", "strong"}:
        raise PairedPrefixHorizonError(
            "formal G1 requires weak, medium, and strong competence strata"
        )
    supported = [
        report
        for report in checkpoint_reports
        if report["cumulative_exposure_delta"]["ci95_low"] > 0.0
        and report["conditional_aliasing_delta"]["ci95_low"] > 0.0
    ]
    required_support = int(config["minimum_supporting_checkpoints"])
    result = {
        "schema": PAIRED_PREFIX_HORIZON_REPORT_SCHEMA,
        "claim_scope": (
            "frozen_policy_paired_prefix_aliasing_horizon_association_"
            "not_phasepool_or_policy_improvement"
        ),
        "policy_update_executed": False,
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "horizons": list(config["horizons"]),
        "benchmark": benchmark,
        "competence_band_definition": {
            band: {"minimum_inclusive": bounds[0], "maximum": bounds[1]}
            for band, bounds in COMPETENCE_BANDS.items()
        },
        "checkpoint_count": len(checkpoint_reports),
        "checkpoint_bindings": bindings,
        "checkpoints": checkpoint_reports,
        "decision": {
            "correlation_supported": len(supported) >= required_support,
            "supporting_checkpoint_count": len(supported),
            "minimum_supporting_checkpoints": required_support,
            "interpretation": (
                "horizon association supported after opportunity decomposition"
                if len(supported) >= required_support
                else "horizon association not supported"
            ),
        },
    }
    output = Path(output_path).resolve()
    svg = output.with_suffix(".svg")
    atomic_write_json(output, result)
    atomic_write_text(svg, _render_svg(result))
    return output, svg


def _checkpoint_report(
    *,
    checkpoint_id: str,
    competence_band: str,
    rows: Any,
    horizons: tuple[int, ...],
    minimum_trajectories: int,
    minimum_task_clusters: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise PairedPrefixHorizonError("checkpoint has no paired-prefix rows")
    by_horizon: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    trajectory_horizons: dict[str, set[int]] = defaultdict(set)
    task_ids = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise PairedPrefixHorizonError("paired-prefix row is malformed")
        horizon = int(row.get("horizon", -1))
        if horizon not in horizons:
            raise PairedPrefixHorizonError("row contains an unregistered horizon")
        trajectory_id = str(row.get("trajectory_id", ""))
        task_id = str(row.get("task_id", ""))
        if not trajectory_id or not task_id:
            raise PairedPrefixHorizonError("paired-prefix row lacks identity")
        by_horizon[horizon].append(row)
        trajectory_horizons[trajectory_id].add(horizon)
        task_ids.add(task_id)
    if any(observed != set(horizons) for observed in trajectory_horizons.values()):
        raise PairedPrefixHorizonError(
            "every frozen trajectory must contribute all registered prefixes"
        )
    if len(trajectory_horizons) < minimum_trajectories:
        raise PairedPrefixHorizonError("checkpoint trajectory count is underpowered")
    if len(task_ids) < minimum_task_clusters:
        raise PairedPrefixHorizonError("checkpoint task-cluster count is underpowered")
    summaries = []
    exposure_curve = _kaplan_meier_exposure(rows, horizons)
    for horizon in horizons:
        members = by_horizon[horizon]
        strict_contexts = _context_union(members, "strict_context_ids")
        aliased_contexts = _context_union(members, "aliased_context_ids")
        repeat_contexts = _context_union(members, "repeat_context_ids")
        summaries.append(
            {
                "horizon": horizon,
                "trajectory_count": len(members),
                "at_risk_trajectory_count": sum(
                    bool(row.get("at_risk_at_horizon", True)) for row in members
                ),
                "cumulative_exposure_rate": exposure_curve[horizon]["cumulative_exposure_rate"],
                "repeat_opportunity_rate": (
                    len(repeat_contexts) / len(strict_contexts) if strict_contexts else 0.0
                ),
                "conditional_aliasing_burden": (
                    len(aliased_contexts) / len(repeat_contexts) if repeat_contexts else 0.0
                ),
                "risk_set_trajectory_count": exposure_curve[horizon]["risk_set_trajectory_count"],
                "strict_context_count": len(strict_contexts),
                "repeat_context_count": len(repeat_contexts),
                "aliased_context_count": len(aliased_contexts),
            }
        )
    task_metrics = _task_metrics(rows, horizons)
    exposure_delta = _bootstrap_delta(
        task_metrics,
        metric="cumulative_exposure_rate",
        low_horizon=horizons[0],
        high_horizon=horizons[-1],
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    conditional_delta = _bootstrap_delta(
        task_metrics,
        metric="conditional_aliasing_burden",
        low_horizon=horizons[0],
        high_horizon=horizons[-1],
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 1,
    )
    return {
        "checkpoint_id": checkpoint_id,
        "competence_band": competence_band,
        "trajectory_count": len(trajectory_horizons),
        "task_cluster_count": len(task_ids),
        "prefix_pairing_verified": True,
        "terminal_censoring_preserved": True,
        "summaries": summaries,
        "cumulative_exposure_delta": exposure_delta,
        "conditional_aliasing_delta": conditional_delta,
    }


def _task_metrics(
    rows: Sequence[Mapping[str, Any]], horizons: tuple[int, ...]
) -> dict[str, dict[int, dict[str, float]]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    output: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for task_id, task_rows in by_task.items():
        exposure_curve = _kaplan_meier_exposure(task_rows, horizons)
        by_horizon: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in task_rows:
            by_horizon[int(row["horizon"])].append(row)
        for horizon in horizons:
            members = by_horizon[horizon]
            strict_contexts = _context_union(members, "strict_context_ids")
            aliased_contexts = _context_union(members, "aliased_context_ids")
            repeat_contexts = _context_union(members, "repeat_context_ids")
            output[task_id][horizon] = {
                "cumulative_exposure_rate": exposure_curve[horizon]["cumulative_exposure_rate"],
                "repeat_opportunity_rate": (
                    len(repeat_contexts) / len(strict_contexts) if strict_contexts else 0.0
                ),
                "conditional_aliasing_burden": (
                    len(aliased_contexts) / len(repeat_contexts) if repeat_contexts else 0.0
                ),
            }
    if any(set(values) != set(horizons) for values in output.values()):
        raise PairedPrefixHorizonError("task clusters are not paired across horizons")
    return dict(output)


def _kaplan_meier_exposure(
    rows: Sequence[Mapping[str, Any]], horizons: tuple[int, ...]
) -> dict[int, dict[str, float | int]]:
    by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trajectory[str(row["trajectory_id"])].append(row)
    event_and_censor: list[tuple[int | None, int]] = []
    for members in by_trajectory.values():
        observed_counts = [int(row.get("observed_transition_count", -1)) for row in members]
        if not observed_counts or min(observed_counts) < 0:
            raise PairedPrefixHorizonError("paired-prefix row lacks observed transition count")
        censor_time = max(observed_counts)
        event_steps = [
            int(row["first_aliased_step"])
            for row in members
            if row.get("first_aliased_step") is not None
        ]
        event_time = min(event_steps) if event_steps else None
        if event_time is not None and not 1 <= event_time <= censor_time:
            raise PairedPrefixHorizonError("aliasing event falls outside the observed trajectory")
        event_and_censor.append((event_time, censor_time))
    survival = 1.0
    output: dict[int, dict[str, float | int]] = {}
    for step in range(1, horizons[-1] + 1):
        at_risk = sum(
            (event_time is None and censor_time >= step)
            or (event_time is not None and event_time >= step)
            for event_time, censor_time in event_and_censor
        )
        events = sum(event_time == step for event_time, _ in event_and_censor)
        if events:
            if at_risk <= 0:
                raise PairedPrefixHorizonError("aliasing event has an empty risk set")
            survival *= 1.0 - events / at_risk
        if step in horizons:
            output[step] = {
                "cumulative_exposure_rate": 1.0 - survival,
                "risk_set_trajectory_count": at_risk,
            }
    return output


def _context_union(rows: Sequence[Mapping[str, Any]], field: str) -> set[str]:
    output: set[str] = set()
    for row in rows:
        values = row.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise PairedPrefixHorizonError(f"paired-prefix row lacks valid {field}")
        output.update(values)
    return output


def _bootstrap_delta(
    task_metrics: Mapping[str, Mapping[int, Mapping[str, float]]],
    *,
    metric: str,
    low_horizon: int,
    high_horizon: int,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    deltas = [
        values[high_horizon][metric] - values[low_horizon][metric]
        for values in task_metrics.values()
    ]
    if not deltas:
        raise PairedPrefixHorizonError("bootstrap has no task-cluster deltas")
    rng = random.Random(seed)
    bootstrap = sorted(mean(rng.choice(deltas) for _ in deltas) for _ in range(resamples))
    return {
        "mean": mean(deltas),
        "ci95_low": _percentile(bootstrap, 0.025),
        "ci95_high": _percentile(bootstrap, 0.975),
        "bootstrap_resamples": resamples,
        "independent_task_cluster_count": len(deltas),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _validate_config(raw: Any) -> dict[str, Any]:
    expected = {
        "schema",
        "id",
        "benchmark",
        "horizons",
        "minimum_checkpoint_count",
        "minimum_trajectories_per_checkpoint",
        "minimum_task_clusters_per_checkpoint",
        "minimum_supporting_checkpoints",
        "bootstrap_resamples",
        "bootstrap_seed",
        "checkpoints",
        "claim_boundary",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise PairedPrefixHorizonError("paired-prefix config fields differ from schema")
    if raw["schema"] != PAIRED_PREFIX_HORIZON_CONFIG_SCHEMA:
        raise PairedPrefixHorizonError("unsupported paired-prefix config schema")
    if raw["horizons"] != [4, 8, 16, 32]:
        raise PairedPrefixHorizonError("formal G1 horizons must be 4/8/16/32")
    if raw["benchmark"] not in {"alfworld", "appworld"}:
        raise PairedPrefixHorizonError("formal G1 requires one registered benchmark")
    if raw["minimum_checkpoint_count"] != 3 or len(raw["checkpoints"]) != 3:
        raise PairedPrefixHorizonError("formal G1 requires exactly three checkpoints")
    if int(raw["minimum_trajectories_per_checkpoint"]) < 30:
        raise PairedPrefixHorizonError("formal G1 requires at least 30 trajectories")
    if int(raw["minimum_task_clusters_per_checkpoint"]) < 10:
        raise PairedPrefixHorizonError("formal G1 requires at least 10 task clusters")
    if int(raw["bootstrap_resamples"]) < 1000:
        raise PairedPrefixHorizonError("formal G1 requires at least 1000 bootstraps")
    if not 1 <= int(raw["minimum_supporting_checkpoints"]) <= 3:
        raise PairedPrefixHorizonError("supporting checkpoint threshold is invalid")
    if raw["claim_boundary"] != {
        "frozen_policy_only": True,
        "phasepool_effectiveness": False,
        "policy_improvement": False,
    }:
        raise PairedPrefixHorizonError("paired-prefix claim boundary widened")
    for binding in raw["checkpoints"]:
        if not isinstance(binding, Mapping) or set(binding) != {
            "checkpoint_id",
            "competence_band",
            "rows_path",
            "rows_sha256",
        }:
            raise PairedPrefixHorizonError("checkpoint binding fields differ from schema")
    return dict(raw)


def _competence_evidence(
    *,
    collection: Mapping[str, Any],
    benchmark: str,
    declared_band: str,
) -> dict[str, float | int | str]:
    evidence_by_benchmark = collection.get("actor_competence_by_benchmark")
    if not isinstance(evidence_by_benchmark, Mapping):
        raise PairedPrefixHorizonError("paired-prefix rows lack frozen actor competence evidence")
    evidence = evidence_by_benchmark.get(benchmark)
    if not isinstance(evidence, Mapping):
        raise PairedPrefixHorizonError("paired-prefix rows lack competence evidence for benchmark")
    successes = int(evidence.get("successful_trajectory_count", -1))
    trajectories = int(evidence.get("trajectory_count", -1))
    success_rate = float(evidence.get("success_rate", -1.0))
    if (
        successes < 0
        or trajectories <= 0
        or successes > trajectories
        or abs(success_rate - successes / trajectories) > 1e-12
    ):
        raise PairedPrefixHorizonError("frozen actor competence evidence is invalid")
    if declared_band not in COMPETENCE_BANDS:
        raise PairedPrefixHorizonError("unknown competence band")
    minimum, maximum = COMPETENCE_BANDS[declared_band]
    within_band = minimum <= success_rate < maximum
    if declared_band == "strong":
        within_band = minimum <= success_rate <= maximum
    if not within_band:
        raise PairedPrefixHorizonError(
            "declared competence band conflicts with frozen actor screen"
        )
    return {
        "benchmark": benchmark,
        "successful_trajectory_count": successes,
        "trajectory_count": trajectories,
        "success_rate": success_rate,
        "verified_band": declared_band,
    }


def _resolve(config_path: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PairedPrefixHorizonError(f"artifact is not a mapping: {path}")
    return value


def _render_svg(report: Mapping[str, Any]) -> str:
    width, height = 1120, 420
    panels = (
        ("cumulative_exposure_rate", "Cumulative exposure"),
        ("repeat_opportunity_rate", "Repeat opportunity"),
        ("conditional_aliasing_burden", "Aliasing given repeat"),
    )
    colors = {"weak": "#64748b", "medium": "#2563eb", "strong": "#dc2626"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="sans-serif" font-size="20" font-weight="700">Horizon → aliasing: exposure is not opportunity</text>',
    ]
    panel_width = 340
    for panel_index, (metric, title) in enumerate(panels):
        left = 30 + panel_index * 365
        top, bottom = 70, 340
        parts.extend(
            [
                f'<text x="{left}" y="55" font-family="sans-serif" font-size="15" font-weight="600">{title}</text>',
                f'<line x1="{left}" y1="{bottom}" x2="{left + panel_width}" y2="{bottom}" stroke="#94a3b8"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#94a3b8"/>',
            ]
        )
        for checkpoint in report["checkpoints"]:
            points = []
            for index, summary in enumerate(checkpoint["summaries"]):
                x = left + index * panel_width / 3
                y = bottom - float(summary[metric]) * (bottom - top)
                points.append(f"{x:.1f},{y:.1f}")
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[checkpoint["competence_band"]]}"/>'
                )
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[checkpoint["competence_band"]]}" stroke-width="2"/>'
            )
        for index, horizon in enumerate(report["horizons"]):
            x = left + index * panel_width / 3
            parts.append(
                f'<text x="{x:.1f}" y="362" text-anchor="middle" font-family="sans-serif" font-size="11">H={horizon}</text>'
            )
    legend_x = 820
    for index, band in enumerate(("weak", "medium", "strong")):
        y = 390
        x = legend_x + index * 95
        parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[band]}"/>')
        parts.append(
            f'<text x="{x + 9}" y="{y + 4}" font-family="sans-serif" font-size="11">{band}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


__all__ = [
    "PAIRED_PREFIX_HORIZON_CONFIG_SCHEMA",
    "PAIRED_PREFIX_HORIZON_REPORT_SCHEMA",
    "PairedPrefixHorizonError",
    "analyze_paired_prefix_horizon_study",
]
