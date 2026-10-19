"""CPU-only mechanism benchmark for continuation-credit aliasing.

The benchmark is synthetic on purpose: it varies aliasing prevalence, group
sample size, and horizon independently before any model or environment is
allowed to confound the estimator comparison.  ALFWorld remains the external
validity stage, not the mechanism-identification stage.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from lh_platform.core.atomic_io import atomic_write_json, atomic_write_text
from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.controlled_credit_aliasing_charts import (
    render_factorial_svg,
)
from lh_platform.training.dataset import sha256_file
from lh_platform.training.residual_credit import calibration as calibration_module
from lh_platform.training.residual_credit.calibration import (
    CalibrationVisit,
    ResidualCreditCalibrationError,
    build_residual_credit_calibration,
)

CONTROLLED_CREDIT_ALIASING_SWEEP_SCHEMA = "lh_controlled_credit_aliasing_sweep/v1"
CONTROLLED_CREDIT_ALIASING_SWEEP_FACTORIAL_SCHEMA = "lh_controlled_credit_aliasing_sweep/v2"
_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def run_controlled_credit_aliasing_sweep(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the preregistered factorial estimator benchmark without a GPU."""

    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _validate_config(raw)
    cells = []
    for aliasing in config["aliasing_probabilities"]:
        for effect_size in config["effect_size_deltas"]:
            for group_size in config["group_sizes"]:
                for phase_balance in config["phase_balances"]:
                    for horizon in config["horizons"]:
                        for revisit_rate in config["revisit_rates"]:
                            cells.append(
                                _run_cell(
                                    config,
                                    aliasing=float(aliasing),
                                    effect_size=float(effect_size),
                                    group_size=int(group_size),
                                    phase_balance=float(phase_balance),
                                    horizon=int(horizon),
                                    revisit_rate=(
                                        None if revisit_rate is None else float(revisit_rate)
                                    ),
                                )
                            )
    receipt = {
        "schema": config["schema"],
        "claim_scope": ("synthetic_estimator_mechanism_not_model_environment_or_policy_effect"),
        "gpu_required": False,
        "policy_update_executed": False,
        "config_path": str(path),
        "config": config,
        "config_file_sha256": sha256_file(path),
        "config_sha256": canonical_sha256(config),
        "implementation_files": [
            {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            {
                "path": str(Path(calibration_module.__file__).resolve()),
                "sha256": sha256_file(Path(calibration_module.__file__).resolve()),
            },
        ],
        "python_version": sys.version,
        "cell_count": len(cells),
        "cells": cells,
    }
    if output_path is not None:
        output = Path(output_path).resolve()
        if config["schema"] == CONTROLLED_CREDIT_ALIASING_SWEEP_FACTORIAL_SCHEMA:
            chart = output.with_suffix(".svg")
            atomic_write_text(chart, render_factorial_svg(receipt))
            receipt["chart"] = {
                "path": str(chart),
                "sha256": sha256_file(chart),
                "semantics": (
                    "phasepool_minus_coarse_brier_and_regret_to_coarse_hard_lower_envelope"
                ),
            }
        atomic_write_json(output, receipt)
    return receipt


def _run_cell(
    config: Mapping[str, Any],
    *,
    aliasing: float,
    effect_size: float,
    group_size: int,
    phase_balance: float,
    horizon: int,
    revisit_rate: float | None,
) -> dict[str, Any]:
    repetitions = []
    realized_revisit_rates = []
    for repetition in range(int(config["dataset_repetitions"])):
        rows = _generate_cell(
            aliasing_probability=aliasing,
            effect_size_delta=effect_size,
            group_size=group_size,
            phase_balance=phase_balance,
            horizon=horizon,
            revisit_probability=revisit_rate,
            pair_count=int(config["pairs_per_cell"]),
            gamma=float(config["gamma"]),
            success_reward=float(config["success_reward"]),
            seed=int(config["seed"]) + repetition,
        )
        evaluation_rows = [row for row in rows if row.evaluation_visit]
        revisit_rows = [
            row
            for row in rows
            if row.step_idx > 0
            and isinstance(row.coarse_key, tuple)
            and row.coarse_key[:1] == ("anchor",)
        ]
        realized_revisit_rates.append(
            len(revisit_rows) / len(evaluation_rows) if evaluation_rows else 0.0
        )
        repetitions.append(
            build_residual_credit_calibration(
                rows,
                success_reward=float(config["success_reward"]),
                random_split_seeds=tuple(config["random_split_seeds"]),
                minimum_pair_count=int(config["pairs_per_cell"]),
                minimum_loto_coverage=0.0,
            )
        )
    return {
        "aliasing_probability": aliasing,
        "effect_size_delta": effect_size,
        "group_size": group_size,
        "phase_balance": phase_balance,
        "horizon": horizon,
        "revisit_probability": revisit_rate,
        "realized_revisit_rate": _metric_summary(realized_revisit_rates),
        "dataset_repetition_count": len(repetitions),
        "trajectory_count_per_repetition": repetitions[0]["trajectory_count"],
        "ground_truth_aliased_pair_rate": (
            round(aliasing * int(config["pairs_per_cell"])) / int(config["pairs_per_cell"])
        ),
        "estimators": _aggregate_estimators(repetitions),
        "estimator_ladder": _aggregate_named_estimates(
            repetitions,
            field="estimator_ladder",
        ),
        "refinement_collision_prevalence": {
            metric: _metric_summary(
                [result["refinement_collision_prevalence"][metric] for result in repetitions]
            )
            for metric in repetitions[0]["refinement_collision_prevalence"]
        },
        "estimator_ladder_diagnostics": {
            metric: _metric_summary(
                [result["estimator_ladder_diagnostics"][metric] for result in repetitions]
            )
            for metric in repetitions[0]["estimator_ladder_diagnostics"]
        },
        "random_matched_size": _metric_summary(
            [result["random_matched_size"]["return_mse_mean"] for result in repetitions]
        ),
    }


def _validate_config(raw: Any) -> dict[str, Any]:
    expected_v1 = {
        "schema",
        "id",
        "aliasing_probabilities",
        "group_sizes",
        "horizons",
        "pairs_per_cell",
        "dataset_repetitions",
        "gamma",
        "success_reward",
        "seed",
        "random_split_seeds",
        "claim_boundary",
    }
    expected_v2 = expected_v1 | {
        "effect_size_deltas",
        "phase_balances",
        "revisit_rates",
    }
    if not isinstance(raw, Mapping) or frozenset(raw) not in {
        frozenset(expected_v1),
        frozenset(expected_v2),
    }:
        raise ResidualCreditCalibrationError(
            "controlled credit-aliasing sweep fields differ from schema"
        )
    if raw["schema"] not in {
        CONTROLLED_CREDIT_ALIASING_SWEEP_SCHEMA,
        CONTROLLED_CREDIT_ALIASING_SWEEP_FACTORIAL_SCHEMA,
    }:
        raise ResidualCreditCalibrationError("unsupported controlled credit-aliasing sweep schema")
    is_v2 = raw["schema"] == CONTROLLED_CREDIT_ALIASING_SWEEP_FACTORIAL_SCHEMA
    if is_v2 != (set(raw) == expected_v2):
        raise ResidualCreditCalibrationError(
            "controlled sweep schema and factorial fields disagree"
        )
    if not isinstance(raw["id"], str) or not raw["id"].strip():
        raise ResidualCreditCalibrationError("controlled sweep id is empty")
    aliasing = raw["aliasing_probabilities"]
    if (
        not isinstance(aliasing, list)
        or not aliasing
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
            for value in aliasing
        )
        or len(set(float(value) for value in aliasing)) != len(aliasing)
    ):
        raise ResidualCreditCalibrationError(
            "aliasing probabilities must be a non-empty list in [0, 1]"
        )
    group_sizes = raw["group_sizes"]
    if (
        not isinstance(group_sizes, list)
        or not group_sizes
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2 or value % 2
            for value in group_sizes
        )
        or len(set(group_sizes)) != len(group_sizes)
    ):
        raise ResidualCreditCalibrationError(
            "group sizes must be positive even integers of at least two"
        )
    effect_sizes = raw.get("effect_size_deltas", [0.6])
    if (
        not isinstance(effect_sizes, list)
        or not effect_sizes
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
            for value in effect_sizes
        )
        or len(set(float(value) for value in effect_sizes)) != len(effect_sizes)
    ):
        raise ResidualCreditCalibrationError("effect-size deltas must be unique in [0, 1]")
    phase_balances = raw.get("phase_balances", [0.5])
    if (
        not isinstance(phase_balances, list)
        or not phase_balances
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) < 1.0
            for value in phase_balances
        )
        or len(set(float(value) for value in phase_balances)) != len(phase_balances)
    ):
        raise ResidualCreditCalibrationError("phase balances must be unique in (0, 1)")
    if any(
        abs(group_size * float(balance) - round(group_size * float(balance))) > 1e-9
        for group_size in group_sizes
        for balance in phase_balances
    ):
        raise ResidualCreditCalibrationError(
            "every group-size and phase-balance product must be integral"
        )
    revisit_rates = raw.get("revisit_rates", [None])
    if (
        not isinstance(revisit_rates, list)
        or not revisit_rates
        or any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            )
            for value in revisit_rates
        )
    ):
        raise ResidualCreditCalibrationError("revisit rates must lie in [0, 1]")
    horizons = raw["horizons"]
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in horizons
        )
        or len(set(horizons)) != len(horizons)
    ):
        raise ResidualCreditCalibrationError("horizons must be positive integers")
    if (
        isinstance(raw["pairs_per_cell"], bool)
        or not isinstance(raw["pairs_per_cell"], int)
        or raw["pairs_per_cell"] < 20
    ):
        raise ResidualCreditCalibrationError("pairs_per_cell must remain at least 20")
    if (
        isinstance(raw["dataset_repetitions"], bool)
        or not isinstance(raw["dataset_repetitions"], int)
        or raw["dataset_repetitions"] < 2
    ):
        raise ResidualCreditCalibrationError("dataset_repetitions must be at least two")
    if not isinstance(raw["gamma"], (int, float)) or not 0.0 <= float(raw["gamma"]) <= 1.0:
        raise ResidualCreditCalibrationError("gamma must be in [0, 1]")
    if not isinstance(raw["success_reward"], (int, float)) or float(raw["success_reward"]) <= 0:
        raise ResidualCreditCalibrationError("success_reward must be positive")
    seeds = raw["random_split_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
    ):
        raise ResidualCreditCalibrationError("random split seeds must be unique")
    if isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int):
        raise ResidualCreditCalibrationError("controlled sweep seed must be an integer")
    if raw["claim_boundary"] != {
        "synthetic_mechanism_only": True,
        "model_awareness": False,
        "policy_improvement": False,
        "a3_behavioral_encoder": False,
    }:
        raise ResidualCreditCalibrationError("controlled sweep claim boundary widened")
    normalized = dict(raw)
    normalized["effect_size_deltas"] = list(effect_sizes)
    normalized["phase_balances"] = list(phase_balances)
    normalized["revisit_rates"] = list(revisit_rates)
    return normalized


def _generate_cell(
    *,
    aliasing_probability: float,
    group_size: int,
    horizon: int,
    pair_count: int,
    gamma: float,
    success_reward: float,
    seed: int,
    effect_size_delta: float = 0.6,
    phase_balance: float = 0.5,
    revisit_probability: float | None = None,
) -> tuple[CalibrationVisit, ...]:
    rng = random.Random(f"{seed}:{aliasing_probability:.8f}:{group_size}:{horizon}:{pair_count}")
    aliased_count = round(aliasing_probability * pair_count)
    pair_order = list(range(pair_count))
    rng.shuffle(pair_order)
    aliased_pairs = set(pair_order[:aliased_count])
    mode_one_count = round(group_size * phase_balance)
    mode_counts = (group_size - mode_one_count, mode_one_count)
    rows = []
    for pair_index in range(pair_count):
        aliased = pair_index in aliased_pairs
        probabilities = (
            (0.5 - effect_size_delta / 2.0, 0.5 + effect_size_delta / 2.0)
            if aliased
            else (0.5, 0.5)
        )
        for mode_index, probability in enumerate(probabilities):
            for replicate in range(mode_counts[mode_index]):
                success = rng.random() < probability
                revisits_anchor = (
                    replicate % 2 == 0
                    if revisit_probability is None
                    else rng.random() < revisit_probability
                )
                trajectory_uid = (
                    f"cell-a{aliasing_probability}-n{group_size}-h{horizon}-"
                    f"p{pair_index}-m{mode_index}-r{replicate}"
                )
                for step_idx in range(horizon):
                    terminal = step_idx == horizon - 1
                    visit_uid = f"{trajectory_uid}:{step_idx}"
                    next_uid = None if terminal else f"{trajectory_uid}:{step_idx + 1}"
                    remaining_budget = horizon - step_idx
                    revisits_coarse_anchor = step_idx == horizon // 2 and revisits_anchor
                    if step_idx == 0 or revisits_coarse_anchor:
                        coarse_key = ("anchor", pair_index)
                        refined_key = (
                            "anchor",
                            pair_index,
                            "mode",
                            mode_index,
                            "budget",
                            remaining_budget,
                        )
                    else:
                        coarse_key = (
                            "suffix",
                            pair_index,
                            mode_index,
                            step_idx,
                        )
                        refined_key = coarse_key
                    rows.append(
                        CalibrationVisit(
                            visit_uid=visit_uid,
                            trajectory_uid=trajectory_uid,
                            pair_uid=f"pair-{pair_index}",
                            step_idx=step_idx,
                            coarse_key=coarse_key,
                            refined_key=refined_key,
                            exact_key=refined_key,
                            next_visit_uid=next_uid,
                            discounted_return=(gamma ** (horizon - step_idx - 1))
                            * (success_reward if success else 0.0),
                            success=success,
                            terminal=terminal,
                            remaining_budget=remaining_budget,
                            evaluation_visit=step_idx == 0,
                        )
                    )
    return tuple(rows)


def _aggregate_estimators(
    repetitions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    output = {}
    for name in repetitions[0]["estimators"]:
        rows = [result["estimators"][name] for result in repetitions]
        output[name] = {
            metric: _metric_summary([row.get(metric) for row in rows])
            for metric in (
                "return_mse",
                "success_brier",
                "loto_coverage",
                "pooling_weight_mean",
                "fine_trajectory_ess_mean",
            )
            if any(row.get(metric) is not None for row in rows)
        }
        if rows[0].get("estimator_role") is not None:
            output[name]["estimator_role"] = rows[0]["estimator_role"]
    return output


def _aggregate_named_estimates(
    repetitions: list[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    output = {}
    for name in repetitions[0][field]:
        rows = [result[field][name] for result in repetitions]
        output[name] = {
            metric: _metric_summary([row.get(metric) for row in rows])
            for metric in (
                "return_mse",
                "success_brier",
                "loto_coverage",
                "prediction_mean",
                "pooling_weight_mean",
                "fine_trajectory_ess_mean",
            )
            if any(row.get(metric) is not None for row in rows)
        }
        output[name]["protocol"] = rows[0]["protocol"]
        if rows[0].get("estimator_role") is not None:
            output[name]["estimator_role"] = rows[0]["estimator_role"]
    return output


def _metric_summary(values: list[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"count": 0, "mean": None, "stddev": None, "ci95_half_width": None}
    if len(numeric) == 1:
        return {
            "count": 1,
            "mean": numeric[0],
            "stddev": None,
            "ci95_method": "insufficient_repetitions",
            "ci95_half_width": None,
        }
    deviation = stdev(numeric)
    degrees_of_freedom = len(numeric) - 1
    use_student_t = degrees_of_freedom in _T_CRITICAL_975
    critical = _T_CRITICAL_975[degrees_of_freedom] if use_student_t else 1.96
    return {
        "count": len(numeric),
        "mean": mean(numeric),
        "stddev": deviation,
        "ci95_method": (
            "student_t_two_sided_95" if use_student_t else "normal_approximation_two_sided_95"
        ),
        "ci95_half_width": critical * deviation / (len(numeric) ** 0.5),
    }


__all__ = [
    "CONTROLLED_CREDIT_ALIASING_SWEEP_SCHEMA",
    "CONTROLLED_CREDIT_ALIASING_SWEEP_FACTORIAL_SCHEMA",
    "run_controlled_credit_aliasing_sweep",
]
