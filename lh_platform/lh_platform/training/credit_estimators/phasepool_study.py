"""Scientific contract for cross-fitted PhasePool-G2PO experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PHASEPOOL_STUDY_SCHEMA = "lh_phasepool_credit_estimator_study/v1"
PHASEPOOL_CALIBRATION_ADMISSION_SCHEMA = "lh_phasepool_calibration_admission/v1"


class PhasePoolStudyError(ValueError):
    """Raised when a PhasePool study weakens its estimator/evidence gates."""


def load_phasepool_study_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PhasePoolStudyError(f"cannot read PhasePool study: {error}") from error
    if not isinstance(raw, dict):
        raise PhasePoolStudyError("PhasePool study must be a mapping")
    expected_fields = {
        "schema",
        "id",
        "claim_boundary",
        "fixed_harness",
        "actor_loss",
        "estimators",
        "gates",
    }
    if set(raw) != expected_fields or raw.get("schema") != PHASEPOOL_STUDY_SCHEMA:
        raise PhasePoolStudyError("PhasePool study fields or schema differ from v1")
    harness = _mapping(raw, "fixed_harness")
    expected_harness = {
        "actor_prompt_identical": True,
        "rollout_batch_identical_within_update": True,
        "actor_optimizer_identical": True,
        "decomposition_source": "environment_verified_training_only",
    }
    if dict(harness) != expected_harness:
        raise PhasePoolStudyError("PhasePool fixed harness is unsafe")
    actor_loss = _mapping(raw, "actor_loss")
    if dict(actor_loss) != {
        "unit": "trajectory_step_token",
        "per_token_weight": "1 / (trajectory_length * action_token_count)",
        "trajectory_total_gradient_mass": 1.0,
        "semantic_step_total_gradient_mass": "equal_within_trajectory",
    }:
        raise PhasePoolStudyError("PhasePool actor loss differs from G2PO-TN")
    estimators = raw.get("estimators")
    expected_estimators = [
        {
            "id": "g2po_tn_loto",
            "key": "coarse",
            "exclusion": "trajectory",
            "pooling": "full",
        },
        {
            "id": "phasepool_g2po",
            "key": "phase",
            "exclusion": "trajectory",
            "pooling": "empirical_bayes",
        },
    ]
    if estimators != expected_estimators:
        raise PhasePoolStudyError("PhasePool study requires the registered estimator pair")
    gates = _mapping(raw, "gates")
    required_gates = {
        "require_frozen_calibration_go_receipt": True,
        "require_single_shared_frozen_batch": True,
        "require_trajectory_loto": True,
        "require_success_and_failure": True,
        "require_per_task_group_outcome_variation": True,
        "require_full_prediction_coverage": True,
        "require_verified_phase_collision": True,
        "require_credit_target_difference": True,
        "on_failure": "stop_before_optimizer_update",
        "online_performance_claim_after_one_update": False,
    }
    if dict(gates) != required_gates:
        raise PhasePoolStudyError("PhasePool online admission gates are unsafe")
    claim = _mapping(raw, "claim_boundary")
    expected_claim = {
        "question": (
            "Does cross-fitted residual-state partial pooling improve credit "
            "calibration and one-update behavior relative to coarse G2PO-TN LOTO?"
        ),
        "estimator_calibration": True,
        "one_update_mechanism_screen": True,
        "policy_improvement": False,
        "recursive_dp": False,
        "autonomous_decomposition": False,
    }
    if dict(claim) != expected_claim:
        raise PhasePoolStudyError("PhasePool study overclaims current evidence")
    return raw


def assess_phasepool_calibration_report(
    report: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    """Require per-dataset estimator gains before any online update."""

    blocked: list[str] = []
    if report.get("schema") != "lh_credit_estimator_ablation/v1":
        blocked.append("calibration_report_schema_invalid")
    datasets = report.get("datasets")
    if not isinstance(datasets, Mapping) or len(datasets) < 2:
        return "no_go", ("calibration_requires_two_datasets",)
    benchmarks = {
        raw_dataset.get("benchmark")
        for raw_dataset in datasets.values()
        if isinstance(raw_dataset, Mapping)
    }
    if len(benchmarks) < 2:
        blocked.append("calibration_requires_two_benchmarks")
    for dataset_id, raw_dataset in datasets.items():
        if not isinstance(raw_dataset, Mapping):
            blocked.append(f"{dataset_id}_report_invalid")
            continue
        if raw_dataset.get("has_outcome_variation") is not True:
            blocked.append(f"{dataset_id}_outcome_variation_missing")
        if int(raw_dataset.get("cross_phase_coarse_group_count", 0)) < 1:
            blocked.append(f"{dataset_id}_phase_aliasing_missing")
        arms = raw_dataset.get("arms")
        if not isinstance(arms, Mapping):
            blocked.append(f"{dataset_id}_arms_missing")
            continue
        control = arms.get("g2po_tn_loto")
        treatment = arms.get("phasepool_eb_loto")
        random_arms = [
            value for name, value in arms.items() if str(name).startswith("matched_random_")
        ]
        if len(random_arms) != 3:
            blocked.append(f"{dataset_id}_matched_random_controls_incomplete")
        if not isinstance(control, Mapping) or not isinstance(treatment, Mapping):
            blocked.append(f"{dataset_id}_registered_arms_missing")
            continue
        control_metrics = control.get("metrics")
        treatment_metrics = treatment.get("metrics")
        if not isinstance(control_metrics, Mapping) or not isinstance(treatment_metrics, Mapping):
            blocked.append(f"{dataset_id}_metrics_missing")
            continue
        treatment_coverage = _finite_metric(treatment_metrics, "coverage")
        control_coverage = _finite_metric(control_metrics, "coverage")
        if treatment_coverage is None or treatment_coverage < 1.0:
            blocked.append(f"{dataset_id}_phasepool_coverage_incomplete")
        if control_coverage is None or control_coverage < 1.0:
            blocked.append(f"{dataset_id}_control_coverage_incomplete")
        fine_ess = _finite_metric(treatment_metrics, "fine_ess_mean")
        if fine_ess is None or fine_ess < 2.0:
            blocked.append(f"{dataset_id}_phasepool_fine_ess_below_two")
        for metric in ("return_mse", "success_brier"):
            treatment_value = _finite_metric(treatment_metrics, metric)
            control_value = _finite_metric(control_metrics, metric)
            if treatment_value is None or control_value is None or treatment_value >= control_value:
                blocked.append(f"{dataset_id}_{metric}_not_better_than_control")
            for index, random_arm in enumerate(random_arms):
                random_metrics = (
                    random_arm.get("metrics") if isinstance(random_arm, Mapping) else None
                )
                random_value = (
                    _finite_metric(random_metrics, metric)
                    if isinstance(random_metrics, Mapping)
                    else None
                )
                if (
                    treatment_value is None
                    or random_value is None
                    or treatment_value >= random_value
                ):
                    blocked.append(f"{dataset_id}_{metric}_not_better_than_random_{index}")
    return ("no_go" if blocked else "go"), tuple(dict.fromkeys(blocked))


def load_phasepool_calibration_admission(
    path: str | Path,
    *,
    expected_policy_checkpoint_sha256: str,
    expected_gamma: float,
) -> dict[str, Any]:
    """Verify a frozen calibration receipt and every artifact it binds."""

    receipt_path = Path(path).resolve()
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhasePoolStudyError(
            f"cannot read PhasePool calibration admission: {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema") != (
        PHASEPOOL_CALIBRATION_ADMISSION_SCHEMA
    ):
        raise PhasePoolStudyError("PhasePool calibration admission schema is invalid")
    if payload.get("decision") != "go" or payload.get("blocked_by") != []:
        raise PhasePoolStudyError("PhasePool calibration admission is not Go")
    if payload.get("policy_checkpoint_sha256") != expected_policy_checkpoint_sha256:
        raise PhasePoolStudyError("calibration policy checkpoint differs from online actor")
    contract = payload.get("online_method_contract")
    expected_contract = {
        "control_arm": "g2po_tn_loto",
        "treatment_arm": "phasepool_eb_loto",
        "gamma": float(expected_gamma),
        "success_reward": 10.0,
        "evaluation": "all",
    }
    if not isinstance(contract, Mapping) or dict(contract) != expected_contract:
        raise PhasePoolStudyError("calibration method contract differs from online credit")
    _verify_bound_file(payload, "report_path", "report_sha256")
    _verify_bound_file(payload, "config_path", "config_sha256")
    datasets = payload.get("dataset_artifacts")
    if not isinstance(datasets, list) or len(datasets) < 2:
        raise PhasePoolStudyError("calibration receipt lacks two dataset artifacts")
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            raise PhasePoolStudyError("calibration dataset artifact is invalid")
        _verify_bound_file(dataset, "records_path", "records_sha256")
    benchmark_set = {dataset.get("benchmark") for dataset in datasets}
    if benchmark_set != {"alfworld_native", "appworld_native"}:
        raise PhasePoolStudyError("online PhasePool admission requires ALFWorld and AppWorld")
    try:
        report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhasePoolStudyError(
            f"cannot recompute PhasePool calibration decision: {error}"
        ) from error
    if not isinstance(report, Mapping):
        raise PhasePoolStudyError("PhasePool calibration report is invalid")
    decision, blocked_by = assess_phasepool_calibration_report(report)
    if decision != "go" or blocked_by:
        raise PhasePoolStudyError("calibration report no longer satisfies Go gates")
    if (
        report.get("config_sha256") != payload.get("config_sha256")
        or report.get("policy_checkpoint_sha256") != payload.get("policy_checkpoint_sha256")
        or report.get("dataset_artifacts") != datasets
    ):
        raise PhasePoolStudyError("calibration receipt lineage differs from report")
    report_datasets = report.get("datasets")
    receipt_dataset_ids = {dataset.get("dataset_id") for dataset in datasets}
    if not isinstance(report_datasets, Mapping) or receipt_dataset_ids != set(report_datasets):
        raise PhasePoolStudyError("calibration dataset IDs differ from report")
    for dataset in datasets:
        dataset_report = report_datasets.get(dataset.get("dataset_id"))
        if not isinstance(dataset_report, Mapping) or dataset_report.get(
            "benchmark"
        ) != dataset.get("benchmark"):
            raise PhasePoolStudyError("calibration dataset benchmark differs from report")
    calibration_contract_sha = payload.get("credit_state_identity_contract_sha256")
    if calibration_contract_sha is not None:
        from .credit_keys import LEGACY_ONLINE_CONTRACT

        online_contract_sha = LEGACY_ONLINE_CONTRACT.contract_sha256()
        if calibration_contract_sha != online_contract_sha:
            raise PhasePoolStudyError(
                "credit state identity contract mismatch: "
                f"calibration={str(calibration_contract_sha)[:16]}... "
                f"online={online_contract_sha[:16]}..."
            )
    return payload


def _mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise PhasePoolStudyError(f"PhasePool study {key} must be a mapping")
    return value


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    import math

    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _verify_bound_file(payload: Mapping[str, Any], path_key: str, sha_key: str) -> None:
    from lh_platform.training.dataset import sha256_file

    raw_path = payload.get(path_key)
    expected = payload.get(sha_key)
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise PhasePoolStudyError(f"calibration receipt lacks {path_key}/{sha_key}")
    path = Path(raw_path).resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise PhasePoolStudyError(f"calibration artifact changed: {path_key}")


__all__ = [
    "PHASEPOOL_CALIBRATION_ADMISSION_SCHEMA",
    "PHASEPOOL_STUDY_SCHEMA",
    "PhasePoolStudyError",
    "assess_phasepool_calibration_report",
    "load_phasepool_calibration_admission",
    "load_phasepool_study_config",
]
