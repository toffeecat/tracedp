"""Synthesize controlled and natural evidence about aliasing and horizon.

The report deliberately separates two questions:

1. Does more continuation aliasing make refined credit more useful?
2. Do longer natural trajectories encounter more aliasing?

The controlled sweep can answer the first question.  Only multi-horizon natural
rollouts can answer the second.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.training.dataset import sha256_file
from lh_platform.training.natural_aliasing import (
    LEGACY_NATURAL_ROWS_SCHEMA,
    NATURAL_ROWS_SCHEMA,
)

ALIASING_HORIZON_EVIDENCE_SCHEMA = "lh_aliasing_horizon_evidence/v1"
PILOT_REGISTRY_SCHEMA = "lh_aliasing_horizon_pilot_registry/v1"
CONTROLLED_SWEEP_SCHEMA = "lh_controlled_credit_aliasing_sweep/v1"


class AliasingHorizonEvidenceError(ValueError):
    """Raised when evidence cannot support an honest synthesis."""


def build_aliasing_horizon_evidence(
    sweep_path: str | Path,
    pilot_registry_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a claim-gated aliasing/horizon evidence report."""

    sweep_file = Path(sweep_path).resolve()
    registry_file = Path(pilot_registry_path).resolve()
    sweep = _load_mapping(sweep_file, json_document=True)
    registry = _load_mapping(registry_file, json_document=False)
    if sweep.get("schema") != CONTROLLED_SWEEP_SCHEMA:
        raise AliasingHorizonEvidenceError("unsupported controlled sweep schema")
    if registry.get("schema") != PILOT_REGISTRY_SCHEMA:
        raise AliasingHorizonEvidenceError("unsupported pilot registry schema")

    controlled_rows = _controlled_rows(sweep.get("cells"))
    dose_response = _dose_response(controlled_rows)
    aliasing_correlations = _correlate_aggregated_within(
        controlled_rows,
        condition_field="horizon",
        predictor_field="aliasing_probability",
        nuisance_field="group_size",
    )
    horizon_correlations = _correlate_aggregated_within(
        controlled_rows,
        condition_field="aliasing_probability",
        predictor_field="horizon",
        nuisance_field="group_size",
    )
    pilots = _load_pilot_evidence(registry.get("pilots"), registry_file.parent)
    natural = _natural_prevalence(pilots)
    natural_inference = _load_natural_inference(
        registry.get("natural_inference"),
        registry_file.parent,
    )
    blocked_by = _natural_correlation_blockers(
        natural,
        natural_inference=natural_inference,
    )
    controlled_observed = bool(aliasing_correlations) and all(
        row["spearman_rho"] is not None and row["spearman_rho"] > 0.0
        for row in aliasing_correlations
    )

    result = {
        "schema": ALIASING_HORIZON_EVIDENCE_SCHEMA,
        "claim_scope": (
            "controlled_aliasing_estimator_association_and_natural_prevalence_"
            "audit_not_causal_long_horizon_proof"
        ),
        "source_bindings": {
            "controlled_sweep": {
                "path": str(sweep_file),
                "sha256": sha256_file(sweep_file),
            },
            "pilot_registry": {
                "path": str(registry_file),
                "sha256": sha256_file(registry_file),
            },
            "natural_inference": None
            if natural_inference is None
            else natural_inference["artifact_binding"],
        },
        "controlled_dose_response": dose_response,
        "controlled_association": {
            "aliasing_to_refinement_gain": aliasing_correlations,
            "horizon_to_refinement_gain": horizon_correlations,
            "interpretation": (
                "The sweep identifies Method x Aliasing. It does not make "
                "aliasing prevalence a function of horizon."
            ),
        },
        "natural_prevalence": natural,
        "natural_inference": natural_inference,
        "pilot_evidence": pilots,
        "claim_gates": {
            "controlled_aliasing_effect_observed": controlled_observed,
            "natural_aliasing_horizon_correlation_established": not blocked_by,
            "blocked_by": blocked_by,
        },
    }
    if output_path is not None:
        atomic_write_json(output_path, result)
    return result


def _load_mapping(path: Path, *, json_document: bool) -> Mapping[str, Any]:
    if not path.is_file():
        raise AliasingHorizonEvidenceError(f"evidence source does not exist: {path}")
    if json_document:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AliasingHorizonEvidenceError(f"evidence source is not a mapping: {path}")
    return value


def _controlled_rows(cells: Any) -> list[dict[str, float]]:
    if not isinstance(cells, list) or not cells:
        raise AliasingHorizonEvidenceError("controlled sweep has no cells")
    rows = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise AliasingHorizonEvidenceError("controlled sweep cell is invalid")
        ladder = cell.get("estimator_ladder")
        if not isinstance(ladder, Mapping):
            raise AliasingHorizonEvidenceError("controlled sweep lacks estimator ladder")
        coarse = _metric_mean(ladder, "g2po_tn_loto", "success_brier")
        refined = _metric_mean(
            ladder,
            "budget_refined_tn_loto_hard_split",
            "success_brier",
        )
        if coarse is None or refined is None:
            continue
        rows.append(
            {
                "aliasing_probability": float(cell["aliasing_probability"]),
                "horizon": float(cell["horizon"]),
                "group_size": float(cell["group_size"]),
                "coarse_brier": coarse,
                "refined_brier": refined,
                "refinement_gain_brier": coarse - refined,
            }
        )
    if not rows:
        raise AliasingHorizonEvidenceError(
            "controlled sweep has no comparable coarse/refined Brier cells"
        )
    return rows


def _metric_mean(
    ladder: Mapping[str, Any],
    arm: str,
    metric: str,
) -> float | None:
    arm_value = ladder.get(arm)
    if not isinstance(arm_value, Mapping):
        return None
    metric_value = arm_value.get(metric)
    if not isinstance(metric_value, Mapping) or metric_value.get("mean") is None:
        return None
    return float(metric_value["mean"])


def _dose_response(rows: Iterable[Mapping[str, float]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[Mapping[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[(row["horizon"], row["aliasing_probability"])].append(row)
    return [
        {
            "horizon": int(horizon),
            "aliasing_probability": aliasing,
            "refinement_gain_brier_mean": mean(row["refinement_gain_brier"] for row in members),
            "coarse_brier_mean": mean(row["coarse_brier"] for row in members),
            "refined_brier_mean": mean(row["refined_brier"] for row in members),
            "cell_count": len(members),
        }
        for (horizon, aliasing), members in sorted(grouped.items())
    ]


def _correlate_aggregated_within(
    rows: Iterable[Mapping[str, float]],
    *,
    condition_field: str,
    predictor_field: str,
    nuisance_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[float, list[Mapping[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row[condition_field]].append(row)
    output = []
    for condition, members in sorted(grouped.items()):
        predictor_groups: dict[float, list[float]] = defaultdict(list)
        nuisance_levels = set()
        for row in members:
            predictor_groups[row[predictor_field]].append(row["refinement_gain_brier"])
            nuisance_levels.add(row[nuisance_field])
        predictor_values = sorted(predictor_groups)
        aggregated_gains = [mean(predictor_groups[value]) for value in predictor_values]
        output.append(
            {
                condition_field: int(condition) if condition_field == "horizon" else condition,
                "spearman_rho": _spearman(
                    predictor_values,
                    aggregated_gains,
                ),
                "sample_count": len(predictor_values),
                "nuisance_aggregation": f"mean_over_{nuisance_field}",
                "nuisance_level_count": len(nuisance_levels),
            }
        )
    return output


def _load_natural_inference(
    configured: Any,
    base_dir: Path,
) -> dict[str, Any] | None:
    if configured is None:
        return None
    allowed = {"source_path", "bootstrap_resamples", "seed"}
    if not isinstance(configured, Mapping) or not set(configured) <= allowed:
        raise AliasingHorizonEvidenceError(
            "natural inference may bind raw source_path, bootstrap_resamples, "
            "and seed only; statistics "
            "cannot be supplied inline in the registry"
        )
    if "source_path" not in configured:
        raise AliasingHorizonEvidenceError("natural inference lacks raw source_path")
    source_path = Path(str(configured["source_path"]))
    if not source_path.is_absolute():
        source_path = (base_dir / source_path).resolve()
    value = _load_mapping(source_path, json_document=True)
    if value.get("schema") not in {
        LEGACY_NATURAL_ROWS_SCHEMA,
        NATURAL_ROWS_SCHEMA,
    }:
        raise AliasingHorizonEvidenceError("unsupported natural rows schema")
    rows = _natural_rows(value.get("rows"))
    resamples = int(configured.get("bootstrap_resamples", 1000))
    seed = int(configured.get("seed", 0))
    ci_low, ci_high = _cluster_bootstrap_spearman(
        rows,
        resamples=resamples,
        seed=seed,
    )
    counts: dict[str, set[str]] = defaultdict(set)
    task_counts: dict[str, set[str]] = defaultdict(set)
    benchmark_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        counts[str(row["horizon"])].add(row["trajectory_id"])
        task_counts[str(row["horizon"])].add(row["task_id"])
        benchmark_rows[str(row.get("benchmark", "unspecified"))].append(row)
    burden_by_horizon: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        burden_by_horizon[str(row["horizon"])].append(float(row["aliasing_burden"]))
    return {
        "independent_trajectory_count_by_horizon": {
            horizon: len(trajectory_ids)
            for horizon, trajectory_ids in sorted(counts.items(), key=lambda item: int(item[0]))
        },
        "cluster_unit": "task_id",
        "independent_task_count_by_horizon": {
            horizon: len(task_ids)
            for horizon, task_ids in sorted(task_counts.items(), key=lambda item: int(item[0]))
        },
        "bootstrap_resamples": resamples,
        "positive_aliasing_row_count": sum(float(row["aliasing_burden"]) > 0.0 for row in rows),
        "mean_aliasing_burden_by_horizon": {
            horizon: mean(values)
            for horizon, values in sorted(burden_by_horizon.items(), key=lambda item: int(item[0]))
        },
        "spearman_rho": _spearman(
            [float(row["horizon"]) for row in rows],
            [float(row["aliasing_burden"]) for row in rows],
        ),
        "spearman_rho_ci_low": ci_low,
        "spearman_rho_ci_high": ci_high,
        "seed": seed,
        "by_benchmark": {
            benchmark: {
                "row_count": len(members),
                "spearman_rho": _spearman(
                    [float(row["horizon"]) for row in members],
                    [float(row["aliasing_burden"]) for row in members],
                ),
                "positive_aliasing_row_count": sum(
                    float(row["aliasing_burden"]) > 0.0 for row in members
                ),
            }
            for benchmark, members in sorted(benchmark_rows.items())
        },
        "artifact_binding": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
    }


def _natural_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise AliasingHorizonEvidenceError("natural rows artifact has no rows")
    output = []
    identities = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise AliasingHorizonEvidenceError("natural inference row is invalid")
        required = {"task_id", "trajectory_id", "horizon", "aliasing_burden"}
        if not required <= set(row):
            raise AliasingHorizonEvidenceError("natural inference row is incomplete")
        horizon = int(row["horizon"])
        burden = float(row["aliasing_burden"])
        if horizon <= 0 or not math.isfinite(burden) or burden < 0.0:
            raise AliasingHorizonEvidenceError("natural inference row values are invalid")
        identity = (str(row["task_id"]), str(row["trajectory_id"]), horizon)
        if identity in identities:
            raise AliasingHorizonEvidenceError("natural inference rows are duplicated")
        identities.add(identity)
        output.append(
            {
                "benchmark": str(row.get("benchmark", "unspecified")),
                "task_id": str(row["task_id"]),
                "trajectory_id": str(row["trajectory_id"]),
                "horizon": horizon,
                "aliasing_burden": burden,
            }
        )
    horizons_by_trajectory: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in output:
        horizons_by_trajectory[(row["task_id"], row["trajectory_id"])].add(row["horizon"])
    horizon_sets = {tuple(sorted(horizons)) for horizons in horizons_by_trajectory.values()}
    if len(horizon_sets) != 1 or len(next(iter(horizon_sets))) < 3:
        raise AliasingHorizonEvidenceError(
            "natural inference must use paired prefixes with the same three or "
            "more horizons for every trajectory"
        )
    return output


def _cluster_bootstrap_spearman(
    rows: list[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if resamples < 1:
        raise AliasingHorizonEvidenceError("bootstrap_resamples must be positive")
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    tasks = sorted(by_task)
    if len(tasks) < 2:
        return None, None
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [generator.choice(tasks) for _ in tasks]
        sampled_rows = [row for task in sample for row in by_task[task]]
        rho = _spearman(
            [float(row["horizon"]) for row in sampled_rows],
            [float(row["aliasing_burden"]) for row in sampled_rows],
        )
        if rho is not None:
            estimates.append(rho)
    if not estimates:
        return None, None
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_center = mean(left)
    right_center = mean(right)
    numerator = sum(
        (x - left_center) * (y - right_center) for x, y in zip(left, right, strict=False)
    )
    left_scale = math.sqrt(sum((x - left_center) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_center) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _load_pilot_evidence(pilots: Any, base_dir: Path) -> list[dict[str, Any]]:
    if not isinstance(pilots, list) or not pilots:
        raise AliasingHorizonEvidenceError("pilot registry has no pilots")
    output = []
    for pilot in pilots:
        if not isinstance(pilot, Mapping):
            raise AliasingHorizonEvidenceError("pilot registry entry is invalid")
        required = {
            "id",
            "source_path",
            "evidence_role",
            "metric_origin",
            "metrics",
            "claim_scope",
        }
        if not required <= set(pilot):
            raise AliasingHorizonEvidenceError("pilot registry entry is incomplete")
        source = Path(str(pilot["source_path"]))
        if not source.is_absolute():
            source = (base_dir / source).resolve()
        if not source.is_file():
            raise AliasingHorizonEvidenceError(f"pilot evidence source does not exist: {source}")
        metric_origin = str(pilot["metric_origin"])
        metrics = pilot["metrics"]
        if not isinstance(metrics, Mapping):
            raise AliasingHorizonEvidenceError("pilot metrics must be a mapping")
        if metric_origin == "extracted_from_source":
            extractor = str(pilot.get("metric_extractor", ""))
            metrics = _extract_metrics(source, extractor)
        elif metric_origin != "curated_descriptive_not_gate_input":
            raise AliasingHorizonEvidenceError(f"unsupported pilot metric origin: {metric_origin}")
        output.append(
            {
                "id": str(pilot["id"]),
                "evidence_role": str(pilot["evidence_role"]),
                "horizon": metrics.get("horizon", pilot.get("horizon")),
                "metrics": dict(metrics),
                "metric_origin": metric_origin,
                "metric_extractor": pilot.get("metric_extractor"),
                "claim_scope": str(pilot["claim_scope"]),
                "source_path": str(source),
                "source_sha256": sha256_file(source),
            }
        )
    return output


def _extract_metrics(source: Path, extractor: str) -> dict[str, Any]:
    if extractor != "phase_rich_official_anchor_report_v1":
        raise AliasingHorizonEvidenceError(
            f"unsupported pilot metric extractor: {extractor or '<missing>'}"
        )
    text = source.read_text(encoding="utf-8")
    horizon_match = re.search(r";\s*(\d+)\s+steps;", text, flags=re.IGNORECASE)
    groups_match = re.search(
        r"has\s+(\d+)\s+groups\s+and\s+(zero|\d+)\s+cross-frontier\s+collisions",
        text,
        flags=re.IGNORECASE,
    )
    trajectories_match = re.search(
        r"transitions\s*/\s*trajectories\s*\|\s*(\d+)\s*/\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if horizon_match is None or groups_match is None or trajectories_match is None:
        raise AliasingHorizonEvidenceError(
            "phase-rich report does not expose horizon, official-anchor groups, "
            "and trajectory counts in the registered format"
        )
    collision_text = groups_match.group(2).lower()
    return {
        "horizon": int(horizon_match.group(1)),
        "transition_count": int(trajectories_match.group(1)),
        "trajectory_count": int(trajectories_match.group(2)),
        "eligible_group_count": int(groups_match.group(1)),
        "collision_group_count": 0 if collision_text == "zero" else int(collision_text),
    }


def _natural_prevalence(pilots: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pilot in pilots:
        if pilot["evidence_role"] != "natural_prevalence":
            continue
        if pilot["metric_origin"] != "extracted_from_source":
            continue
        metrics = pilot["metrics"]
        horizon = pilot.get("horizon")
        eligible = metrics.get("eligible_group_count")
        collisions = metrics.get("collision_group_count")
        if horizon is None or eligible is None or collisions is None:
            raise AliasingHorizonEvidenceError(
                "natural prevalence pilot lacks horizon/group counts"
            )
        eligible_int = int(eligible)
        collision_int = int(collisions)
        if eligible_int <= 0 or not 0 <= collision_int <= eligible_int:
            raise AliasingHorizonEvidenceError("natural prevalence counts are invalid")
        rows.append(
            {
                "pilot_id": pilot["id"],
                "horizon": int(horizon),
                "eligible_group_count": eligible_int,
                "collision_group_count": collision_int,
                "collision_group_rate": collision_int / eligible_int,
                "independent_trajectory_count": int(metrics.get("trajectory_count", 0)),
                "metric_origin": pilot["metric_origin"],
                "source_sha256": pilot["source_sha256"],
            }
        )
    return sorted(rows, key=lambda row: (row["horizon"], row["pilot_id"]))


def _natural_correlation_blockers(
    natural_rows: list[Mapping[str, Any]],
    *,
    natural_inference: Any,
) -> list[str]:
    blocked = []
    inferred_horizons = set()
    if isinstance(natural_inference, Mapping):
        trajectory_counts = natural_inference.get("independent_trajectory_count_by_horizon")
        if isinstance(trajectory_counts, Mapping):
            inferred_horizons = {int(value) for value in trajectory_counts}
    horizons = inferred_horizons or {int(row["horizon"]) for row in natural_rows}
    if len(horizons) < 3:
        blocked.append("fewer_than_three_natural_horizon_bins")
    inferred_positive = (
        int(natural_inference.get("positive_aliasing_row_count", 0))
        if isinstance(natural_inference, Mapping)
        else 0
    )
    if (
        sum(int(row["collision_group_count"]) for row in natural_rows) == 0
        and inferred_positive == 0
    ):
        blocked.append("no_natural_collision_groups_observed")
    elif len(horizons) >= 3:
        rho = (
            natural_inference.get("spearman_rho")
            if isinstance(natural_inference, Mapping)
            else None
        )
        if rho is None:
            rho = _spearman(
                [float(row["horizon"]) for row in natural_rows],
                [float(row["collision_group_rate"]) for row in natural_rows],
            )
        if rho is None or rho <= 0.0:
            blocked.append("natural_horizon_collision_rate_not_positive")
    if not isinstance(natural_inference, Mapping):
        blocked.append("natural_cluster_bootstrap_inference_unavailable")
        return blocked

    trajectory_counts = natural_inference.get("independent_trajectory_count_by_horizon")
    if not isinstance(trajectory_counts, Mapping) or any(
        int(trajectory_counts.get(str(horizon), 0)) < 30 for horizon in horizons
    ):
        blocked.append("fewer_than_30_independent_trajectories_per_horizon")
    task_counts = natural_inference.get("independent_task_count_by_horizon")
    if not isinstance(task_counts, Mapping) or any(
        int(task_counts.get(str(horizon), 0)) < 10 for horizon in horizons
    ):
        blocked.append("fewer_than_10_independent_task_clusters_per_horizon")
    if natural_inference.get("cluster_unit") != "task_id":
        blocked.append("natural_cluster_unit_is_not_task_id")
    if int(natural_inference.get("bootstrap_resamples", 0)) < 1000:
        blocked.append("fewer_than_1000_cluster_bootstrap_resamples")
    ci_low = natural_inference.get("spearman_rho_ci_low")
    if ci_low is None or float(ci_low) <= 0.0:
        blocked.append("natural_spearman_ci_lower_bound_not_positive")
    return blocked
