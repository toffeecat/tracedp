"""Held-out calibration audit for residual/frontier-conditioned credit keys.

This module deliberately does not compute PPO targets or update a policy.  It
asks the narrower prerequisite question: under the trajectory-normalized
sampling measure, does a verifier-derived key predict held-out continuation
better than the G2PO anchor and an equally fragmented random split?
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.training.agentic_rl_runtime import load_agentic_rollout_artifact
from lh_platform.training.credit_estimators.credit_keys import (
    anchor_observation_key,
    verifier_refined_key,
)
from lh_platform.training.dataset import sha256_file
from lh_platform.training.residual_credit.estimator_bridge import (
    all_visit_predictions,
    credit_visits_from_calibration,
    fit_grouped_loto,
    fit_matched_random,
    fit_phasepool_eb,
    prediction_map,
    summarize_hard_split_estimate,
    summarize_matched_random_estimate,
    summarize_phasepool_estimate,
)

RESIDUAL_CREDIT_CALIBRATION_SCHEMA = "lh_residual_credit_calibration/v1"
RESIDUAL_CREDIT_CALIBRATION_CLAIM_SCOPE = (
    "heldout_continuation_calibration_not_policy_improvement_or_recursive_dp"
)


class ResidualCreditCalibrationError(ValueError):
    """Raised when calibration inputs cannot support a held-out estimate."""


@dataclass(frozen=True)
class ResolvedResidualCreditCalibrationConfig:
    """Validated, repository-relative preregistration for Pilot A."""

    config_path: Path
    workspace_root: Path
    payload: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class CalibrationVisit:
    """One visit with a terminal Monte-Carlo return and verified key labels."""

    visit_uid: str
    trajectory_uid: str
    pair_uid: str
    step_idx: int
    coarse_key: Hashable
    refined_key: Hashable
    exact_key: Hashable
    next_visit_uid: str | None
    discounted_return: float
    success: bool
    terminal: bool
    remaining_budget: int | None = None
    progress_sign: int = 0
    evaluation_visit: bool = True

    def validate(self) -> CalibrationVisit:
        if not self.visit_uid or not self.trajectory_uid or not self.pair_uid:
            raise ResidualCreditCalibrationError("calibration visit identity is empty")
        if self.step_idx < 0:
            raise ResidualCreditCalibrationError("calibration step index is negative")
        if not math.isfinite(self.discounted_return):
            raise ResidualCreditCalibrationError("discounted return must be finite")
        if self.progress_sign not in {-1, 0, 1}:
            raise ResidualCreditCalibrationError("progress_sign must be -1, 0, or 1")
        if self.terminal != (self.next_visit_uid is None):
            raise ResidualCreditCalibrationError(
                "terminal visits must omit next_visit_uid and nonterminal visits must bind it"
            )
        if self.remaining_budget is not None and self.remaining_budget < 0:
            raise ResidualCreditCalibrationError(
                "remaining_budget must be non-negative when present"
            )
        return self


def load_residual_credit_calibration_config(
    config_path: str | Path,
) -> ResolvedResidualCreditCalibrationConfig:
    """Load the normal-actor, no-update residual calibration design."""

    from lh_platform.training.agentic_rl_config import load_agentic_rl_experiment
    from lh_platform.training.agentic_rl_runtime import canonical_sha256
    from lh_platform.training.alfworld_controlled_aliasing import (
        load_alfworld_controlled_aliasing_recipe,
    )

    path = Path(config_path).resolve()
    root = _find_workspace_root(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ResidualCreditCalibrationError(
            f"cannot load residual calibration config: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise ResidualCreditCalibrationError("residual calibration config must be a mapping")
    expected = {
        "schema",
        "id",
        "experiment_ref",
        "controlled_recipe_refs",
        "runtime",
        "estimator",
        "admission",
        "claim_boundary",
    }
    if set(payload) != expected or payload.get("schema") != (
        "lh_residual_credit_calibration_pilot/v1"
    ):
        raise ResidualCreditCalibrationError(
            "residual calibration config fields or schema are invalid"
        )
    experiment_ref = _workspace_reference(root, payload["experiment_ref"])
    load_agentic_rl_experiment(experiment_ref)
    recipe_refs = payload["controlled_recipe_refs"]
    if (
        not isinstance(recipe_refs, list)
        or len(recipe_refs) < 2
        or len(set(recipe_refs)) != len(recipe_refs)
    ):
        raise ResidualCreditCalibrationError(
            "controlled_recipe_refs must contain at least two unique recipes"
        )
    recipes = [
        load_alfworld_controlled_aliasing_recipe(_workspace_reference(root, ref))
        for ref in recipe_refs
    ]
    if len({recipe.sha256 for recipe in recipes}) != len(recipes):
        raise ResidualCreditCalibrationError("controlled recipes are semantically duplicated")

    runtime = _strict_mapping(
        payload,
        "runtime",
        {
            "actor_state_exposure",
            "pair_observability",
            "continuation_controller",
            "samples_per_state",
            "max_episode_steps",
            "sampling_seed",
            "sampling_seed_stride",
            "sampling_temperature",
            "generation_micro_batch_size",
            "policy_update_allowed",
        },
    )
    if (
        runtime["actor_state_exposure"] != "normal"
        or runtime["pair_observability"] != "privileged_only"
        or runtime["continuation_controller"] != "model_policy"
        or runtime["policy_update_allowed"] is not False
    ):
        raise ResidualCreditCalibrationError(
            "oracle hard-split calibration requires a normal actor, privileged-only "
            "pairs, model-policy continuation, and no update"
        )
    for field in (
        "samples_per_state",
        "max_episode_steps",
        "generation_micro_batch_size",
    ):
        if (
            isinstance(runtime[field], bool)
            or not isinstance(runtime[field], int)
            or runtime[field] < 1
        ):
            raise ResidualCreditCalibrationError(f"runtime.{field} must be positive")
    for field in ("sampling_seed", "sampling_seed_stride"):
        if isinstance(runtime[field], bool) or not isinstance(runtime[field], int):
            raise ResidualCreditCalibrationError(f"runtime.{field} must be an integer")
    temperature = runtime["sampling_temperature"]
    if not isinstance(temperature, (int, float)) or float(temperature) <= 0:
        raise ResidualCreditCalibrationError("runtime.sampling_temperature must be positive")

    estimator = _strict_mapping(
        payload,
        "estimator",
        {"gamma", "success_reward", "matched_random_split_seeds"},
    )
    seeds = estimator["matched_random_split_seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ResidualCreditCalibrationError(
            "estimator.matched_random_split_seeds must be non-empty and unique"
        )
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ResidualCreditCalibrationError("matched random split seeds must be integers")
    gamma = estimator["gamma"]
    reward = estimator["success_reward"]
    if not isinstance(gamma, (int, float)) or not 0 <= float(gamma) <= 1:
        raise ResidualCreditCalibrationError("estimator.gamma must be between zero and one")
    if not isinstance(reward, (int, float)) or float(reward) <= 0:
        raise ResidualCreditCalibrationError("estimator.success_reward must be positive")

    admission = _strict_mapping(
        payload,
        "admission",
        {
            "formal_minimum_pair_count",
            "minimum_loto_coverage",
            "require_refined_better_than_coarse",
            "require_refined_better_than_matched_random",
        },
    )
    if (
        not isinstance(admission["formal_minimum_pair_count"], int)
        or admission["formal_minimum_pair_count"] < 20
    ):
        raise ResidualCreditCalibrationError("formal_minimum_pair_count must remain at least 20")
    coverage = admission["minimum_loto_coverage"]
    if not isinstance(coverage, (int, float)) or not 0 <= float(coverage) <= 1:
        raise ResidualCreditCalibrationError(
            "admission.minimum_loto_coverage must be between zero and one"
        )
    if (
        admission["require_refined_better_than_coarse"] is not True
        or admission["require_refined_better_than_matched_random"] is not True
    ):
        raise ResidualCreditCalibrationError(
            "Pilot A must retain both calibration comparison gates"
        )
    claim = _strict_mapping(
        payload,
        "claim_boundary",
        {
            "heldout_continuation_calibration",
            "actor_can_infer_refined_state",
            "policy_improvement",
            "recursive_dp",
            "parent_child_credit_decoupling",
        },
    )
    if claim != {
        "heldout_continuation_calibration": True,
        "actor_can_infer_refined_state": False,
        "policy_improvement": False,
        "recursive_dp": False,
        "parent_child_credit_decoupling": False,
    }:
        raise ResidualCreditCalibrationError(
            "oracle hard-split calibration claim boundary was widened"
        )
    return ResolvedResidualCreditCalibrationConfig(
        config_path=path,
        workspace_root=root,
        payload=payload,
        sha256=canonical_sha256(payload),
    )


def residual_calibration_recipe_contracts(
    resolved: ResolvedResidualCreditCalibrationConfig,
) -> tuple[dict[str, Any], ...]:
    """Bind every preregistered recipe to its required collection settings."""

    from lh_platform.training.agentic_rl_config import load_agentic_rl_experiment
    from lh_platform.training.agentic_rl_runtime import canonical_sha256
    from lh_platform.training.alfworld_controlled_aliasing import (
        load_alfworld_controlled_aliasing_recipe,
    )

    runtime = resolved.payload["runtime"]
    experiment = load_agentic_rl_experiment(
        _workspace_reference(
            resolved.workspace_root,
            resolved.payload["experiment_ref"],
        )
    )
    model = experiment.payload["model"]
    contracts = []
    for index, reference in enumerate(resolved.payload["controlled_recipe_refs"]):
        path = _workspace_reference(resolved.workspace_root, reference)
        recipe = load_alfworld_controlled_aliasing_recipe(path)
        contracts.append(
            {
                "path": str(path),
                "file_sha256": sha256_file(path),
                "recipe_sha256": recipe.sha256,
                "sampling_seed": runtime["sampling_seed"] + index * runtime["sampling_seed_stride"],
                "expected_trajectory_count": 2 * runtime["samples_per_state"],
                "max_episode_steps": runtime["max_episode_steps"],
                "sampling_temperature": float(runtime["sampling_temperature"]),
                "expected_base_checkpoint_sha256": model["checkpoint_tree_sha256"],
                "expected_model_revision": model["revision"],
                "expected_model_reference": model["reference"],
                "expected_rollout_config_sha256": canonical_sha256(
                    {
                        "schema": "lh_residual_credit_calibration_rollout_config/v1",
                        "pilot_config_sha256": resolved.sha256,
                        "recipe_sha256": recipe.sha256,
                        "actor_state_exposure": "normal",
                        "samples_per_state": runtime["samples_per_state"],
                        "max_episode_steps": runtime["max_episode_steps"],
                        "sampling_seed": runtime["sampling_seed"]
                        + index * runtime["sampling_seed_stride"],
                        "sampling_temperature": float(runtime["sampling_temperature"]),
                        "action_space": ("free_text_then_source_alfworld_projection"),
                    }
                ),
            }
        )
    return tuple(contracts)


def build_residual_credit_calibration(
    visits: Iterable[CalibrationVisit],
    *,
    success_reward: float,
    random_split_seeds: tuple[int, ...],
    minimum_pair_count: int,
    minimum_loto_coverage: float,
) -> dict[str, Any]:
    """Compare coarse/refined/random/exact LOTO continuation estimators.

    Fit/predict is owned by ``credit_estimators`` (GroupedMean / PhasePool /
    MatchedRandom).  This module keeps collection validation, Pilot-A gate
    policy, and residual-specific edge diagnostics.
    """

    rows = tuple(row.validate() for row in visits)
    if any(row.remaining_budget is None for row in rows):
        raise ResidualCreditCalibrationError(
            "budget-refined calibration requires remaining_budget on every visit"
        )
    _validate_design(
        rows,
        success_reward=success_reward,
        random_split_seeds=random_split_seeds,
        minimum_pair_count=minimum_pair_count,
        minimum_loto_coverage=minimum_loto_coverage,
    )
    by_uid = {row.visit_uid: row for row in rows}
    coarse_keys = {row.visit_uid: row.coarse_key for row in rows}
    refined_keys = {row.visit_uid: row.refined_key for row in rows}
    exact_keys = {row.visit_uid: row.exact_key for row in rows}
    try:
        credit_visits = credit_visits_from_calibration(rows)
    except ValueError as error:
        raise ResidualCreditCalibrationError(str(error)) from error

    coarse_estimate = fit_grouped_loto(credit_visits, key="coarse")
    refined_estimate = fit_grouped_loto(credit_visits, key="phase")
    exact_estimate = fit_grouped_loto(credit_visits, key="exact")
    phasepool_estimate = fit_phasepool_eb(credit_visits)
    coarse_predictions = all_visit_predictions(credit_visits, key="coarse")
    refined_predictions = all_visit_predictions(credit_visits, key="phase")
    exact_predictions = all_visit_predictions(credit_visits, key="exact")

    estimators = {
        "coarse": summarize_hard_split_estimate(
            rows,
            coarse_estimate,
            key_of=coarse_keys,
            success_reward=success_reward,
            all_predictions=coarse_predictions,
            by_uid=by_uid,
            value_ranking=_value_ranking_accuracy,
        ),
        "refined": summarize_hard_split_estimate(
            rows,
            refined_estimate,
            key_of=refined_keys,
            success_reward=success_reward,
            all_predictions=refined_predictions,
            by_uid=by_uid,
            value_ranking=_value_ranking_accuracy,
        ),
        "exact_snapshot_upper_bound": summarize_hard_split_estimate(
            rows,
            exact_estimate,
            key_of=exact_keys,
            success_reward=success_reward,
            all_predictions=exact_predictions,
            by_uid=by_uid,
            value_ranking=_value_ranking_accuracy,
        ),
        "oracle_refined_partial_pooling_diagnostic": summarize_phasepool_estimate(
            rows,
            phasepool_estimate,
            value_ranking=_value_ranking_accuracy,
        ),
    }
    random_results = []
    for seed in random_split_seeds:
        estimate = fit_matched_random(credit_visits, seed=seed)
        random_results.append({"seed": seed, **summarize_matched_random_estimate(estimate)})

    random_mses = [
        float(item["return_mse"]) for item in random_results if item["return_mse"] is not None
    ]
    random_summary = {
        "seed_count": len(random_results),
        "return_mse_mean": mean(random_mses) if random_mses else None,
        "return_mse_median": median(random_mses) if random_mses else None,
        "return_mse_min": min(random_mses) if random_mses else None,
        "return_mse_max": max(random_mses) if random_mses else None,
        "success_brier_mean": _mean_available(
            random_results,
            field="success_brier",
        ),
        "loto_coverage_mean": _mean_available(
            random_results,
            field="loto_coverage",
        ),
        "trajectory_ess_mean": _mean_available(
            random_results,
            field="trajectory_ess_mean",
        ),
        "runs": random_results,
    }
    estimator_ladder = _build_estimator_ladder(
        credit_visits,
        rows=rows,
        tn_loto_coarse=estimators["coarse"],
        tn_loto_refined=estimators["refined"],
        phasepool=estimators["oracle_refined_partial_pooling_diagnostic"],
        matched_random=random_summary,
    )

    blocked_by: list[str] = []
    pair_count = len({row.pair_uid for row in rows})
    refined = estimators["refined"]
    coarse = estimators["coarse"]
    if pair_count < minimum_pair_count:
        blocked_by.append("pair_count_below_preregistered_minimum")
    if float(refined["loto_coverage"]) < minimum_loto_coverage:
        blocked_by.append("refined_loto_coverage_below_threshold")
    if refined["return_mse"] is None or coarse["return_mse"] is None:
        blocked_by.append("coarse_or_refined_loto_mse_unavailable")
    elif float(refined["return_mse"]) >= float(coarse["return_mse"]):
        blocked_by.append("refined_not_better_than_coarse")
    random_reference = random_summary["return_mse_mean"]
    if refined["return_mse"] is None or random_reference is None:
        blocked_by.append("matched_random_loto_mse_unavailable")
    elif float(refined["return_mse"]) >= float(random_reference):
        blocked_by.append("refined_not_better_than_matched_random")

    return {
        "schema": RESIDUAL_CREDIT_CALIBRATION_SCHEMA,
        "claim_scope": RESIDUAL_CREDIT_CALIBRATION_CLAIM_SCOPE,
        "trajectory_count": len({row.trajectory_uid for row in rows}),
        "visit_count": len(rows),
        "pair_count": pair_count,
        "success_reward": success_reward,
        "loto_unit": "whole_trajectory",
        "sampling_measure": "trajectory_mean_within_key_then_mean_across_trajectories",
        "estimators": estimators,
        "estimator_ladder": estimator_ladder,
        "estimator_ladder_diagnostics": _estimator_ladder_diagnostics(estimator_ladder),
        "refinement_collision_prevalence": _refinement_collision_prevalence(rows),
        "random_matched_size": random_summary,
        "calibration_gate_passed": not blocked_by,
        "blocked_by": blocked_by,
    }


def calibration_visits_from_records(
    records: Iterable[Any],
    *,
    gamma: float,
    success_reward: float,
) -> tuple[CalibrationVisit, ...]:
    """Project verified normal-actor rollout records into calibration visits."""

    if not 0.0 <= gamma <= 1.0:
        raise ResidualCreditCalibrationError("gamma must be between zero and one")
    if not math.isfinite(success_reward) or success_reward <= 0:
        raise ResidualCreditCalibrationError("success_reward must be positive")
    source = tuple(records)
    by_trajectory: dict[str, list[Any]] = defaultdict(list)
    for record in source:
        by_trajectory[str(record.traj_uid)].append(record)
    visits: list[CalibrationVisit] = []
    initial_keys_by_pair: dict[str, set[tuple[Hashable, Hashable]]] = defaultdict(set)
    for trajectory_uid, trajectory in sorted(by_trajectory.items()):
        ordered = sorted(trajectory, key=lambda row: int(row.step_idx))
        if not ordered:
            continue
        terminal = ordered[-1]
        if not terminal.terminal:
            raise ResidualCreditCalibrationError(f"trajectory {trajectory_uid!r} is incomplete")
        success_values = {bool(row.success) for row in ordered}
        if len(success_values) != 1:
            raise ResidualCreditCalibrationError(
                f"trajectory {trajectory_uid!r} success label drifted"
            )
        success = success_values.pop()
        pair_uid = _required_extra_text(ordered[0].extra, "controlled_recipe_sha256")
        for index, record in enumerate(ordered):
            extra = record.extra
            if extra.get("controlled_collision_verified") is not True:
                raise ResidualCreditCalibrationError(
                    "calibration record lacks controlled collision verification"
                )
            if extra.get("goal_annotation_verified") is not True:
                raise ResidualCreditCalibrationError(
                    "calibration record lacks verified residual/frontier annotation"
                )
            if extra.get("actor_awareness_mode") != "none":
                raise ResidualCreditCalibrationError(
                    "Pilot A requires the frozen normal actor without oracle exposure"
                )
            if (
                extra.get("actor_state_exposure_mode", "normal") != "normal"
                or extra.get("actor_state_exposure") is not None
            ):
                raise ResidualCreditCalibrationError(
                    "Pilot A requires normal actor input without state exposure"
                )
            if (
                extra.get("transition_evidence_mode", "restore_replay") != "restore_replay"
                or extra.get("restore_verified") is False
            ):
                raise ResidualCreditCalibrationError(
                    "Pilot A requires restore/replay-verified transitions"
                )
            if (
                extra.get("continuation_controller") != "model_policy"
                or extra.get("deterministic_continuation_program_used") is not False
            ):
                raise ResidualCreditCalibrationError("Pilot A requires model-policy continuations")
            if _required_extra_text(extra, "controlled_recipe_sha256") != pair_uid:
                raise ResidualCreditCalibrationError(
                    f"trajectory {trajectory_uid!r} changed controlled recipe"
                )
            residual = _required_extra_text(extra, "residual_goal_signature")
            frontier = _required_extra_text(extra, "frontier_signature")
            remaining_budget = record.remaining_budget
            if (
                isinstance(remaining_budget, bool)
                or not isinstance(remaining_budget, int)
                or remaining_budget < 0
                or extra.get("remaining_budget_verified") is not True
            ):
                raise ResidualCreditCalibrationError(
                    "calibration record lacks a verified remaining budget"
                )
            exact = str(record.state_snapshot_sha256).strip()
            if len(exact) != 64:
                raise ResidualCreditCalibrationError(
                    "calibration record lacks an exact pre-action snapshot hash"
                )
            coarse_key = anchor_observation_key(
                pair_uid=pair_uid,
                observation=str(record.observation),
            )
            refined_key = verifier_refined_key(
                coarse_key=coarse_key,
                residual_signature=residual,
                frontier_signature=frontier,
                remaining_budget=remaining_budget,
            )
            visit_uid = f"{trajectory_uid}:{record.step_idx}"
            next_uid = (
                f"{trajectory_uid}:{ordered[index + 1].step_idx}"
                if index + 1 < len(ordered)
                else None
            )
            newly_satisfied = tuple(extra.get("newly_satisfied_goal_ids", ()))
            regressed = tuple(extra.get("regressed_goal_ids", ()))
            progress_sign = -1 if regressed else (1 if newly_satisfied else 0)
            visits.append(
                CalibrationVisit(
                    visit_uid=visit_uid,
                    trajectory_uid=trajectory_uid,
                    pair_uid=pair_uid,
                    step_idx=int(record.step_idx),
                    coarse_key=coarse_key,
                    refined_key=refined_key,
                    exact_key=exact,
                    next_visit_uid=next_uid,
                    discounted_return=(gamma ** (len(ordered) - int(record.step_idx)))
                    * (success_reward if success else 0.0),
                    success=success,
                    terminal=bool(record.terminal),
                    remaining_budget=remaining_budget,
                    progress_sign=progress_sign,
                    evaluation_visit=index == 0,
                )
            )
            if index == 0:
                initial_keys_by_pair[pair_uid].add((coarse_key, refined_key))
    for pair_uid, keys in initial_keys_by_pair.items():
        coarse = {item[0] for item in keys}
        refined = {item[1] for item in keys}
        if len(coarse) != 1 or len(refined) < 2:
            raise ResidualCreditCalibrationError(
                f"pair {pair_uid!r} is not one coarse anchor with multiple verified states"
            )
    return tuple(visits)


def analyze_residual_credit_calibration_artifacts(
    rollout_paths: Iterable[str | Path],
    *,
    gamma: float,
    success_reward: float,
    random_split_seeds: tuple[int, ...],
    minimum_pair_count: int,
    minimum_loto_coverage: float,
    recipe_contracts: Iterable[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load immutable annotated rollouts and emit one calibration receipt."""

    paths = tuple(Path(path).resolve() for path in rollout_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ResidualCreditCalibrationError("rollout paths must be non-empty and unique")
    contracts = tuple(dict(contract) for contract in recipe_contracts)
    _validate_preregistration(preregistration, contracts)
    records = []
    policy_checkpoints: set[str] = set()
    policy_samples = []
    sources = []
    for path in paths:
        if not path.is_file():
            raise ResidualCreditCalibrationError(f"annotated rollout does not exist: {path}")
        batch = load_agentic_rollout_artifact(path)
        loaded = tuple(batch.records)
        if not loaded:
            raise ResidualCreditCalibrationError(f"annotated rollout is empty: {path}")
        records.extend(loaded)
        policy_samples.extend(batch.policy_samples)
        policy_checkpoints.add(batch.policy_checkpoint_sha256)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "record_count": len(loaded),
            }
        )
    _validate_artifacts_against_recipe_contracts(
        records,
        contracts,
        policy_checkpoints=policy_checkpoints,
        policy_samples=policy_samples,
    )
    visits = calibration_visits_from_records(
        records,
        gamma=gamma,
        success_reward=success_reward,
    )
    result = build_residual_credit_calibration(
        visits,
        success_reward=success_reward,
        random_split_seeds=random_split_seeds,
        minimum_pair_count=minimum_pair_count,
        minimum_loto_coverage=minimum_loto_coverage,
    )
    result["gamma"] = gamma
    result["sources"] = sources
    result["preregistration"] = dict(preregistration)
    result["recipe_contracts"] = [dict(contract) for contract in contracts]
    result["design"] = {
        "minimum_pair_count": minimum_pair_count,
        "minimum_loto_coverage": minimum_loto_coverage,
        "random_split_seeds": list(random_split_seeds),
    }
    if output_path is not None:
        atomic_write_json(output_path, result)
    return result


def _required_extra_text(extra: Mapping[str, Any], field: str) -> str:
    value = extra.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ResidualCreditCalibrationError(f"calibration record lacks {field}")
    return value.strip()


def _find_workspace_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "configs").is_dir() and (parent / "lh_platform").is_dir():
            return parent.resolve()
    raise ResidualCreditCalibrationError(
        "residual calibration config is outside the LongH workspace"
    )


def _workspace_reference(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ResidualCreditCalibrationError("workspace config reference is empty")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ResidualCreditCalibrationError(
            f"workspace config reference is missing or escapes the repository: {value}"
        )
    return path


def _strict_mapping(
    payload: Mapping[str, Any],
    field: str,
    expected_fields: set[str],
) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ResidualCreditCalibrationError(
            f"residual calibration {field} fields differ from schema"
        )
    return value


def _validate_preregistration(
    preregistration: Mapping[str, Any],
    contracts: tuple[dict[str, Any], ...],
) -> None:
    required = {"config_path", "config_file_sha256", "config_sha256", "bindings"}
    if set(preregistration) != required:
        raise ResidualCreditCalibrationError(
            "calibration preregistration receipt fields differ from schema"
        )
    if not all(
        isinstance(preregistration[field], str) and preregistration[field]
        for field in ("config_path", "config_file_sha256", "config_sha256")
    ):
        raise ResidualCreditCalibrationError("calibration preregistration identity is empty")
    bindings = preregistration["bindings"]
    if (
        not isinstance(bindings, list)
        or not bindings
        or not all(
            isinstance(binding, Mapping) and binding.get("matches_head") is True
            for binding in bindings
        )
    ):
        raise ResidualCreditCalibrationError("calibration config tree must match committed HEAD")
    if len(contracts) < 2 or len({contract.get("recipe_sha256") for contract in contracts}) != len(
        contracts
    ):
        raise ResidualCreditCalibrationError("calibration recipe contracts are empty or duplicated")


def _validate_artifacts_against_recipe_contracts(
    records: Iterable[Any],
    contracts: tuple[dict[str, Any], ...],
    *,
    policy_checkpoints: set[str] | None = None,
    policy_samples: Iterable[Any] = (),
) -> None:
    expected = {str(contract["recipe_sha256"]): contract for contract in contracts}
    by_recipe: dict[str, list[Any]] = defaultdict(list)
    all_trajectories: set[str] = set()
    recipe_by_sample_key: dict[tuple[str, int], str] = {}
    for record in records:
        recipe_sha256 = _required_extra_text(record.extra, "controlled_recipe_sha256")
        by_recipe[recipe_sha256].append(record)
        all_trajectories.add(str(record.traj_uid))
        recipe_by_sample_key[(str(record.traj_uid), int(record.step_idx))] = recipe_sha256
    if set(by_recipe) != set(expected):
        raise ResidualCreditCalibrationError(
            "rollout recipes differ from the preregistered recipe set"
        )
    trajectory_occurrences = sum(
        len({str(row.traj_uid) for row in members}) for members in by_recipe.values()
    )
    if trajectory_occurrences != len(all_trajectories):
        raise ResidualCreditCalibrationError("trajectory ids collide across recipes")
    if policy_checkpoints is None or len(policy_checkpoints) != 1:
        raise ResidualCreditCalibrationError(
            "Pilot A artifacts do not share one frozen policy checkpoint"
        )
    sample_count = 0
    for sample in policy_samples:
        sample_count += 1
        key = (str(sample.traj_uid), int(sample.step_idx))
        recipe_sha256 = recipe_by_sample_key.get(key)
        if recipe_sha256 is None:
            raise ResidualCreditCalibrationError(
                "policy sample does not map to a preregistered record"
            )
        if float(sample.sampling_temperature) != float(
            expected[recipe_sha256]["sampling_temperature"]
        ):
            raise ResidualCreditCalibrationError(
                "policy sample temperature differs from preregistration"
            )
    if sample_count != len(recipe_by_sample_key):
        raise ResidualCreditCalibrationError("policy sample count differs from rollout records")
    for recipe_sha256, members in by_recipe.items():
        contract = expected[recipe_sha256]
        by_trajectory: dict[str, list[Any]] = defaultdict(list)
        for record in members:
            by_trajectory[str(record.traj_uid)].append(record)
            if record.seed != contract["sampling_seed"]:
                raise ResidualCreditCalibrationError(
                    "rollout sampling seed differs from preregistration"
                )
            if (
                record.extra.get("pilot_base_checkpoint_sha256")
                != contract["expected_base_checkpoint_sha256"]
            ):
                raise ResidualCreditCalibrationError(
                    "rollout base checkpoint differs from preregistration"
                )
            if (
                record.extra.get("pilot_model_revision") != contract["expected_model_revision"]
                or record.extra.get("pilot_model_reference") != contract["expected_model_reference"]
            ):
                raise ResidualCreditCalibrationError(
                    "rollout model identity differs from preregistration"
                )
        if len(by_trajectory) != contract["expected_trajectory_count"]:
            raise ResidualCreditCalibrationError(
                "rollout trajectory count differs from preregistration"
            )
        rollout_hashes = {str(row.rollout_config_hash) for row in members}
        if rollout_hashes != {contract["expected_rollout_config_sha256"]}:
            raise ResidualCreditCalibrationError(
                "recipe rollout config hash differs from preregistration"
            )
        temperatures = {
            float(row.extra.get("sampling_temperature", float("nan"))) for row in members
        }
        if temperatures != {float(contract["sampling_temperature"])}:
            raise ResidualCreditCalibrationError(
                "rollout sampling temperature differs from preregistration"
            )
        for trajectory in by_trajectory.values():
            if len(trajectory) > contract["max_episode_steps"]:
                raise ResidualCreditCalibrationError(
                    "rollout exceeded the preregistered episode horizon"
                )


def _validate_design(
    rows: tuple[CalibrationVisit, ...],
    *,
    success_reward: float,
    random_split_seeds: tuple[int, ...],
    minimum_pair_count: int,
    minimum_loto_coverage: float,
) -> None:
    if not rows:
        raise ResidualCreditCalibrationError("calibration visits are empty")
    if not math.isfinite(success_reward) or success_reward <= 0:
        raise ResidualCreditCalibrationError("success_reward must be positive")
    if not random_split_seeds or len(set(random_split_seeds)) != len(random_split_seeds):
        raise ResidualCreditCalibrationError(
            "matched random split seeds must be non-empty and unique"
        )
    if minimum_pair_count < 1:
        raise ResidualCreditCalibrationError("minimum_pair_count must be positive")
    if not 0.0 <= minimum_loto_coverage <= 1.0:
        raise ResidualCreditCalibrationError("minimum_loto_coverage must be between zero and one")
    visit_uids = [row.visit_uid for row in rows]
    if len(set(visit_uids)) != len(visit_uids):
        raise ResidualCreditCalibrationError("calibration visit_uid is duplicated")
    by_trajectory: dict[str, list[CalibrationVisit]] = defaultdict(list)
    for row in rows:
        by_trajectory[row.trajectory_uid].append(row)
    for trajectory_uid, trajectory in by_trajectory.items():
        ordered = sorted(trajectory, key=lambda row: row.step_idx)
        if [row.step_idx for row in ordered] != list(range(len(ordered))):
            raise ResidualCreditCalibrationError(
                f"trajectory {trajectory_uid!r} has non-contiguous steps"
            )
        if sum(row.terminal for row in ordered) != 1 or not ordered[-1].terminal:
            raise ResidualCreditCalibrationError(
                f"trajectory {trajectory_uid!r} must end in one terminal visit"
            )
        if [row.step_idx for row in ordered if row.evaluation_visit] != [0]:
            raise ResidualCreditCalibrationError(
                f"trajectory {trajectory_uid!r} must evaluate only its registered start"
            )
        for current, following in zip(ordered, ordered[1:], strict=False):
            if current.next_visit_uid != following.visit_uid:
                raise ResidualCreditCalibrationError(
                    f"trajectory {trajectory_uid!r} has a broken next-visit link"
                )


def _population_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def _value_ranking_accuracy(
    rows: tuple[CalibrationVisit, ...],
    *,
    predictions: Mapping[str, tuple[float, float]],
) -> float | None:
    by_pair_state: dict[tuple[str, Hashable], list[CalibrationVisit]] = defaultdict(list)
    for row in rows:
        if row.visit_uid in predictions:
            by_pair_state[(row.pair_uid, row.refined_key)].append(row)
    scores: list[float] = []
    pair_ids = sorted({row.pair_uid for row in rows})
    for pair_uid in pair_ids:
        states = [
            (state, members)
            for (candidate_pair, state), members in by_pair_state.items()
            if candidate_pair == pair_uid
        ]
        for index, (_, left) in enumerate(states):
            for _, right in states[index + 1 :]:
                actual_delta = mean(row.discounted_return for row in left) - mean(
                    row.discounted_return for row in right
                )
                if actual_delta == 0:
                    continue
                predicted_delta = mean(predictions[row.visit_uid][0] for row in left) - mean(
                    predictions[row.visit_uid][0] for row in right
                )
                if predicted_delta == 0:
                    scores.append(0.5)
                else:
                    scores.append(float((predicted_delta > 0) == (actual_delta > 0)))
    return mean(scores) if scores else None


def _build_estimator_ladder(
    credit_visits,
    *,
    rows: tuple[CalibrationVisit, ...],
    tn_loto_coarse: Mapping[str, Any],
    tn_loto_refined: Mapping[str, Any],
    phasepool: Mapping[str, Any],
    matched_random: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose weighting, cross-fitting, refinement, and pooling separately."""

    return {
        "official_g2po_visit_in_sample": _summarize_start_predictions(
            rows,
            predictions=prediction_map(
                fit_grouped_loto(
                    credit_visits,
                    key="coarse",
                    weighting="visit",
                    exclusion="none",
                )
            ),
            protocol="visit_weighted_self_including",
        ),
        "g2po_tn_in_sample": _summarize_start_predictions(
            rows,
            predictions=prediction_map(
                fit_grouped_loto(
                    credit_visits,
                    key="coarse",
                    weighting="trajectory",
                    exclusion="none",
                )
            ),
            protocol="trajectory_normalized_self_including",
        ),
        "g2po_visit_loto": _summarize_start_predictions(
            rows,
            predictions=prediction_map(
                fit_grouped_loto(
                    credit_visits,
                    key="coarse",
                    weighting="visit",
                    exclusion="trajectory",
                )
            ),
            protocol="visit_weighted_whole_trajectory_loto",
        ),
        "g2po_tn_loto": {
            "protocol": "trajectory_normalized_whole_trajectory_loto",
            **dict(tn_loto_coarse),
        },
        "budget_refined_tn_loto_hard_split": {
            "protocol": "budget_refined_trajectory_normalized_loto_hard_split",
            **dict(tn_loto_refined),
        },
        "phasepool_tn_loto": {
            "protocol": "budget_refined_trajectory_normalized_loto_partial_pooling",
            **dict(phasepool),
        },
        "matched_size_random_tn_loto": {
            "protocol": "matched_size_random_split_trajectory_normalized_loto",
            "estimator_role": "sham_split_control",
            "return_mse": matched_random.get("return_mse_mean"),
            "success_brier": matched_random.get("success_brier_mean"),
            "loto_coverage": matched_random.get("loto_coverage_mean"),
            "fine_trajectory_ess_mean": matched_random.get("trajectory_ess_mean"),
        },
    }


def _mean_available(
    rows: Iterable[Mapping[str, Any]],
    *,
    field: str,
) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _summarize_start_predictions(
    rows: tuple[CalibrationVisit, ...],
    *,
    predictions: Mapping[str, tuple[float, float]],
    protocol: str,
) -> dict[str, Any]:
    evaluation = tuple(row for row in rows if row.evaluation_visit)
    predicted = [row for row in evaluation if row.visit_uid in predictions]
    summary = {
        "protocol": protocol,
        "prediction_count": len(predicted),
        "coverage": len(predicted) / len(evaluation),
        "return_mse": (
            mean((predictions[row.visit_uid][0] - row.discounted_return) ** 2 for row in predicted)
            if predicted
            else None
        ),
        "success_brier": (
            mean((predictions[row.visit_uid][1] - float(row.success)) ** 2 for row in predicted)
            if predicted
            else None
        ),
        "prediction_mean": (
            mean(predictions[row.visit_uid][0] for row in predicted) if predicted else None
        ),
        "value_ranking_accuracy": _value_ranking_accuracy(
            evaluation,
            predictions=predictions,
        ),
    }
    if "loto" in protocol:
        summary["loto_prediction_count"] = summary["prediction_count"]
        summary["loto_coverage"] = summary["coverage"]
    return summary


def _refinement_collision_prevalence(
    rows: tuple[CalibrationVisit, ...],
) -> dict[str, Any]:
    """Measure candidate key collisions, not causal continuation aliasing."""
    starts = tuple(row for row in rows if row.evaluation_visit)
    by_coarse: dict[Hashable, list[CalibrationVisit]] = defaultdict(list)
    for row in starts:
        by_coarse[row.coarse_key].append(row)
    split_groups = []
    heterogeneity = []
    for members in by_coarse.values():
        by_refined: dict[Hashable, list[float]] = defaultdict(list)
        for row in members:
            by_refined[row.refined_key].append(row.discounted_return)
        if len(by_refined) > 1:
            split_groups.append(members)
            heterogeneity.append(
                _population_variance([mean(values) for values in by_refined.values()])
            )
    split_trajectories = sum(len(group) for group in split_groups)
    return {
        "coarse_group_count": len(by_coarse),
        "coarse_groups_split_by_refinement": len(split_groups),
        "cross_refined_coarse_group_rate": (
            len(split_groups) / len(by_coarse) if by_coarse else 0.0
        ),
        "trajectory_fraction_in_split_groups": (
            split_trajectories / len(starts) if starts else 0.0
        ),
        "between_refined_return_variance_mean": (mean(heterogeneity) if heterogeneity else 0.0),
    }


def _estimator_ladder_diagnostics(
    ladder: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | None]:
    def mse(name: str) -> float | None:
        value = ladder[name].get("return_mse")
        return float(value) if value is not None else None

    def difference(left: str, right: str) -> float | None:
        left_value = mse(left)
        right_value = mse(right)
        if left_value is None or right_value is None:
            return None
        return left_value - right_value

    return {
        "visit_loto_minus_in_sample_mse": difference(
            "g2po_visit_loto",
            "official_g2po_visit_in_sample",
        ),
        "tn_loto_minus_in_sample_mse": difference(
            "g2po_tn_loto",
            "g2po_tn_in_sample",
        ),
        "hard_split_minus_coarse_tn_loto_mse": difference(
            "budget_refined_tn_loto_hard_split",
            "g2po_tn_loto",
        ),
        "phasepool_minus_coarse_tn_loto_mse": difference(
            "phasepool_tn_loto",
            "g2po_tn_loto",
        ),
        "phasepool_minus_hard_split_mse": difference(
            "phasepool_tn_loto",
            "budget_refined_tn_loto_hard_split",
        ),
        "hard_split_minus_matched_random_mse": difference(
            "budget_refined_tn_loto_hard_split",
            "matched_size_random_tn_loto",
        ),
    }


__all__ = [
    "CalibrationVisit",
    "RESIDUAL_CREDIT_CALIBRATION_CLAIM_SCOPE",
    "RESIDUAL_CREDIT_CALIBRATION_SCHEMA",
    "ResolvedResidualCreditCalibrationConfig",
    "ResidualCreditCalibrationError",
    "analyze_residual_credit_calibration_artifacts",
    "build_residual_credit_calibration",
    "calibration_visits_from_records",
    "load_residual_credit_calibration_config",
    "residual_calibration_recipe_contracts",
]
