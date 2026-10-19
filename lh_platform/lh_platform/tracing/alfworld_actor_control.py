"""Freeze outcome-blind ALFWorld actor-control pilots from causal-gold cases."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_action_counterfactual import (
    ClosedLoopContinuationPolicy,
)
from lh_platform.tracing.alfworld_controlled_fault_injection import (
    AlfWorldControlledFaultInjectionPlanV1,
)
from lh_platform.training.alfworld_agentic_suite import AlfWorldBranch

ACTOR_CONTROL_PILOT_MANIFEST_SCHEMA = "lh_alfworld_actor_control_pilot_manifest/v1"
ACTOR_CONTROL_SELECTION_RULE = "first_causal_admitted_case_per_task_in_frozen_manifest_order"
FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_FILE_SHA256 = (
    "6ebd346a0b14a614b6ddc90d2cf0a8980ecc5a0f45fb80c5f2ea353dce83744c"
)
FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_CONTENT_SHA256 = (
    "sha256:0723ce082582f9fbd95e1864c58de3727928fefbddec0fc9d58e2a1c0cb9da6b"
)
FROZEN_ACTOR_CONTROL_POPULATION_CASE_COUNT = 10
FROZEN_ACTOR_CONTROL_POPULATION_TASK_COUNT = 10


class AlfWorldActorControlError(ValueError):
    """Raised when a proposed actor-control pilot is not causally auditable."""


def _require_raw_sha256(label: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AlfWorldActorControlError(f"{label} must be a lowercase SHA-256")
    return value


def _require_prefixed_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise AlfWorldActorControlError(f"{label} must use the sha256: prefix")
    _require_raw_sha256(label, value.removeprefix("sha256:"))
    return value


def _require_mapping(label: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlfWorldActorControlError(f"{label} must be an object")
    return value


def _require_nonempty_string(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlfWorldActorControlError(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AlfWorldActorControlError(f"{label} must be a non-negative integer")
    return value


def resolve_alfworld_actor_control_execution_case(
    manifest: Mapping[str, Any],
    *,
    candidate_id: str | None,
    execution_role: str,
) -> Mapping[str, Any]:
    """Resolve one preregistered case without allowing outcome-based substitution."""

    if manifest.get("schema") != ACTOR_CONTROL_PILOT_MANIFEST_SCHEMA:
        raise AlfWorldActorControlError("actor-control manifest schema changed")
    admission_candidate_id = _require_nonempty_string(
        "admission candidate ID",
        manifest.get("admission_case_candidate_id"),
    )
    if execution_role not in {
        "admission",
        "population_case",
        "formal_population_case",
    }:
        raise AlfWorldActorControlError("actor-control execution role is unsupported")
    if execution_role == "admission":
        if candidate_id is not None and candidate_id != admission_candidate_id:
            raise AlfWorldActorControlError("admission candidate cannot be swapped")
        resolved_candidate_id = admission_candidate_id
    else:
        resolved_candidate_id = _require_nonempty_string("population candidate ID", candidate_id)
        if execution_role == "population_case" and resolved_candidate_id == admission_candidate_id:
            raise AlfWorldActorControlError("population execution cannot rerun the admission case")

    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise AlfWorldActorControlError("actor-control cases are missing")
    cases = tuple(_require_mapping("actor-control case", raw_case) for raw_case in raw_cases)
    candidate_ids = tuple(
        _require_nonempty_string("candidate ID", case.get("candidate_id")) for case in cases
    )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AlfWorldActorControlError("actor-control candidate IDs are duplicated")
    matches = tuple(
        case
        for case, frozen_candidate_id in zip(cases, candidate_ids, strict=True)
        if frozen_candidate_id == resolved_candidate_id
    )
    if len(matches) != 1:
        raise AlfWorldActorControlError(
            "requested actor-control candidate is not frozen in the pilot manifest"
        )
    return matches[0]


def validate_frozen_alfworld_actor_control_population_manifest(
    manifest: Mapping[str, Any],
    *,
    file_sha256: str,
) -> None:
    """Require the exact outcome-blind 10-task manifest frozen before execution."""

    if file_sha256 != FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_FILE_SHA256:
        raise AlfWorldActorControlError("actor-control population file is not frozen")
    canonical = dict(manifest)
    claimed_content_sha256 = canonical.pop("content_sha256", None)
    if (
        manifest.get("schema") != ACTOR_CONTROL_PILOT_MANIFEST_SCHEMA
        or claimed_content_sha256 != "sha256:" + canonical_sha256(canonical)
        or claimed_content_sha256 != FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_CONTENT_SHA256
        or manifest.get("selected_case_count") != FROZEN_ACTOR_CONTROL_POPULATION_CASE_COUNT
        or manifest.get("selected_task_count") != FROZEN_ACTOR_CONTROL_POPULATION_TASK_COUNT
        or manifest.get("selection_rule") != ACTOR_CONTROL_SELECTION_RULE
        or manifest.get("selection_uses_actor_control_outcomes") is not False
        or manifest.get("outcome_fields_present") is not False
    ):
        raise AlfWorldActorControlError(
            "actor-control population manifest is not the frozen preregistration"
        )
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 10:
        raise AlfWorldActorControlError("actor-control frozen population coverage changed")
    cases = tuple(
        _require_mapping("actor-control population case", raw_case) for raw_case in raw_cases
    )
    candidate_ids = tuple(
        _require_nonempty_string("candidate ID", case.get("candidate_id")) for case in cases
    )
    task_ids = tuple(_require_nonempty_string("task ID", case.get("task_id")) for case in cases)
    if len(set(candidate_ids)) != 10 or len(set(task_ids)) != 10:
        raise AlfWorldActorControlError("actor-control frozen population identity changed")
    admission_candidate_id = _require_nonempty_string(
        "admission candidate ID", manifest.get("admission_case_candidate_id")
    )
    if candidate_ids.count(admission_candidate_id) != 1:
        raise AlfWorldActorControlError("actor-control frozen admission binding changed")


def _require_bool(label: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise AlfWorldActorControlError(f"{label} must be a boolean")
    return value


def _task_bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values or isinstance(samples, bool) or samples <= 0:
        raise AlfWorldActorControlError("task bootstrap configuration is invalid")
    generator = random.Random(seed)
    size = len(values)
    estimates = sorted(
        sum(generator.choice(values) for _ in range(size)) / size for _ in range(samples)
    )
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def build_alfworld_actor_control_population_summary(
    *,
    pilot_manifest: Mapping[str, Any],
    case_receipts_by_candidate: Mapping[str, Mapping[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Aggregate paired seeds within tasks, then bootstrap independent tasks."""

    raw_cases = pilot_manifest.get("cases")
    raw_seeds = pilot_manifest.get("sampling_seeds")
    max_episode_steps = pilot_manifest.get("max_episode_steps")
    if (
        pilot_manifest.get("schema") != ACTOR_CONTROL_PILOT_MANIFEST_SCHEMA
        or not isinstance(raw_cases, list)
        or not raw_cases
        or not isinstance(raw_seeds, list)
        or not raw_seeds
        or isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
    ):
        raise AlfWorldActorControlError("actor-control population manifest is invalid")
    cases = tuple(
        _require_mapping("actor-control population case", raw_case) for raw_case in raw_cases
    )
    seeds = tuple(_require_nonnegative_int("paired sampling seed", seed) for seed in raw_seeds)
    if len(seeds) != len(set(seeds)):
        raise AlfWorldActorControlError("paired sampling seeds are duplicated")
    candidate_ids = tuple(
        _require_nonempty_string("candidate ID", case.get("candidate_id")) for case in cases
    )
    task_ids = tuple(_require_nonempty_string("task ID", case.get("task_id")) for case in cases)
    if (
        len(candidate_ids) != len(set(candidate_ids))
        or len(task_ids) != len(set(task_ids))
        or pilot_manifest.get("selected_case_count") != len(cases)
        or pilot_manifest.get("selected_task_count") != len(cases)
        or set(case_receipts_by_candidate) != set(candidate_ids)
    ):
        raise AlfWorldActorControlError("actor-control population coverage changed")

    metric_names = (
        "repair_masked",
        "actor_control_supported",
        "baseline_natural_recovery",
    )
    pair_counts = {name: 0 for name in metric_names}
    task_metric_values = {name: [] for name in metric_names}
    per_task: list[dict[str, Any]] = []
    for case in cases:
        candidate_id = _require_nonempty_string("candidate ID", case.get("candidate_id"))
        receipt = _require_mapping(
            "actor-control case receipt", case_receipts_by_candidate[candidate_id]
        )
        canonical_receipt = dict(receipt)
        claimed_content_sha256 = canonical_receipt.pop("content_sha256", None)
        if (
            receipt.get("schema") != "lh_alfworld_actor_control_case/v1"
            or claimed_content_sha256 != "sha256:" + canonical_sha256(canonical_receipt)
            or receipt.get("task_id") != case.get("task_id")
            or receipt.get("gold_step_idx") != case.get("gold_step_idx")
            or receipt.get("max_episode_steps") != max_episode_steps
            or receipt.get("paired_sampling_seeds") != list(seeds)
        ):
            raise AlfWorldActorControlError(
                f"actor-control case binding changed for {candidate_id}"
            )
        suffix = _require_mapping("T_suffix", receipt.get("T_suffix"))
        if suffix.get("success") is not True:
            raise AlfWorldActorControlError(
                f"T_suffix is invalid for actor-control case {candidate_id}"
            )
        raw_pairs = receipt.get("actor_pairs")
        if not isinstance(raw_pairs, list) or len(raw_pairs) != len(seeds):
            raise AlfWorldActorControlError(f"paired actor coverage changed for {candidate_id}")
        task_counts = {name: 0 for name in metric_names}
        for seed, raw_pair in zip(seeds, raw_pairs, strict=True):
            pair = _require_mapping("actor-control pair", raw_pair)
            baseline = _require_mapping("B_actor", pair.get("B_actor"))
            treatment = _require_mapping("T_actor", pair.get("T_actor"))
            baseline_success = _require_bool("B_actor success", baseline.get("success"))
            treatment_success = _require_bool("T_actor success", treatment.get("success"))
            expected = {
                "repair_masked": not treatment_success,
                "actor_control_supported": (treatment_success and not baseline_success),
                "baseline_natural_recovery": baseline_success,
            }
            if pair.get("sampling_seed_base") != seed or any(
                pair.get(name) is not value for name, value in expected.items()
            ):
                raise AlfWorldActorControlError(
                    f"paired actor metric binding changed for {candidate_id}"
                )
            for name, value in expected.items():
                pair_counts[name] += int(value)
                task_counts[name] += int(value)
        task_rates = {name: task_counts[name] / len(seeds) for name in metric_names}
        for name, rate in task_rates.items():
            task_metric_values[name].append(rate)
        per_task.append(
            {
                "candidate_id": candidate_id,
                "task_id": case.get("task_id"),
                "gold_step_idx": case.get("gold_step_idx"),
                "pair_counts": task_counts,
                "seed_averaged_rates": task_rates,
            }
        )

    display_names = {
        "repair_masked": "repair_mask_rate",
        "actor_control_supported": "actor_control_supported_rate",
        "baseline_natural_recovery": "baseline_natural_recovery_rate",
    }
    task_clustered_metrics: dict[str, Any] = {}
    for offset, name in enumerate(metric_names):
        values = task_metric_values[name]
        interval = _task_bootstrap_interval(
            values,
            samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
        )
        task_clustered_metrics[display_names[name]] = {
            "estimate": sum(values) / len(values),
            "confidence_interval_95": list(interval),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed + offset,
            "resampling_unit": "task",
        }
    summary: dict[str, Any] = {
        "schema": "lh_alfworld_actor_control_population_summary/v1",
        "case_count": len(cases),
        "task_count": len(cases),
        "paired_seed_count": len(cases) * len(seeds),
        "pair_counts": dict(sorted(pair_counts.items())),
        "task_clustered_metrics": task_clustered_metrics,
        "controlled_failure_claim_condition": (
            "B_actor=failure and T_actor=success for the same task and sampling seed"
        ),
        "per_task": per_task,
        "claim_scope": (
            "externally_injected_action_actor_control_under_frozen_actor_continuation; "
            "controlled-failure evidence is limited to paired B_actor=failure and "
            "T_actor=success outcomes; not natural actor errors or Self-Traceback"
        ),
    }
    summary["content_sha256"] = "sha256:" + canonical_sha256(summary)
    return summary


def build_alfworld_live_menu_constraint_population_summary(
    *,
    pilot_manifest: Mapping[str, Any],
    case_receipts_by_candidate: Mapping[str, Mapping[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Aggregate paired live-menu effects with task-level resampling."""

    raw_cases = pilot_manifest.get("cases")
    raw_seeds = pilot_manifest.get("sampling_seeds")
    max_episode_steps = pilot_manifest.get("max_episode_steps")
    if (
        pilot_manifest.get("schema") != ACTOR_CONTROL_PILOT_MANIFEST_SCHEMA
        or not isinstance(raw_cases, list)
        or not raw_cases
        or not isinstance(raw_seeds, list)
        or not raw_seeds
        or isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
    ):
        raise AlfWorldActorControlError("live-menu population manifest is invalid")
    cases = tuple(_require_mapping("live-menu population case", value) for value in raw_cases)
    seeds = tuple(_require_nonnegative_int("paired sampling seed", value) for value in raw_seeds)
    candidate_ids = tuple(
        _require_nonempty_string("candidate ID", case.get("candidate_id")) for case in cases
    )
    task_ids = tuple(_require_nonempty_string("task ID", case.get("task_id")) for case in cases)
    if (
        len(seeds) != len(set(seeds))
        or len(candidate_ids) != len(set(candidate_ids))
        or len(task_ids) != len(set(task_ids))
        or pilot_manifest.get("selected_case_count") != len(cases)
        or pilot_manifest.get("selected_task_count") != len(cases)
        or set(case_receipts_by_candidate) != set(candidate_ids)
    ):
        raise AlfWorldActorControlError("live-menu population coverage changed")

    outcome_names = ("menu_rescue", "menu_harm", "both_success", "both_failed")
    pair_counts = {name: 0 for name in outcome_names}
    metric_values = {
        "menu_rescue_rate": [],
        "menu_harm_rate": [],
        "raw_success_rate": [],
        "menu_success_rate": [],
        "menu_success_delta": [],
    }
    per_task: list[dict[str, Any]] = []
    for case in cases:
        candidate_id = _require_nonempty_string("candidate ID", case.get("candidate_id"))
        receipt = _require_mapping(
            "live-menu case receipt",
            case_receipts_by_candidate[candidate_id],
        )
        canonical_receipt = dict(receipt)
        claimed_content_sha256 = canonical_receipt.pop("content_sha256", None)
        if (
            receipt.get("schema") != "lh_alfworld_live_menu_constraint_case/v1"
            or claimed_content_sha256 != "sha256:" + canonical_sha256(canonical_receipt)
            or receipt.get("task_id") != case.get("task_id")
            or receipt.get("gold_step_idx") != case.get("gold_step_idx")
            or receipt.get("max_episode_steps") != max_episode_steps
            or receipt.get("paired_sampling_seeds") != list(seeds)
            or _require_mapping("T_suffix", receipt.get("T_suffix")).get("success") is not True
        ):
            raise AlfWorldActorControlError(f"live-menu case binding changed for {candidate_id}")
        raw_pairs = receipt.get("actor_pairs")
        if not isinstance(raw_pairs, list) or len(raw_pairs) != len(seeds):
            raise AlfWorldActorControlError(f"live-menu pair coverage changed for {candidate_id}")
        task_counts = {name: 0 for name in outcome_names}
        raw_success_count = 0
        menu_success_count = 0
        for seed, raw_pair in zip(seeds, raw_pairs, strict=True):
            pair = _require_mapping("live-menu actor pair", raw_pair)
            raw_arm = _require_mapping("T_raw", pair.get("T_raw"))
            menu_arm = _require_mapping("T_menu", pair.get("T_menu"))
            raw_success = _require_bool("T_raw success", raw_arm.get("success"))
            menu_success = _require_bool("T_menu success", menu_arm.get("success"))
            expected = {
                "menu_rescue": not raw_success and menu_success,
                "menu_harm": raw_success and not menu_success,
                "both_success": raw_success and menu_success,
                "both_failed": not raw_success and not menu_success,
            }
            if pair.get("sampling_seed_base") != seed or any(
                pair.get(name) is not value for name, value in expected.items()
            ):
                raise AlfWorldActorControlError(
                    f"live-menu paired metric binding changed for {candidate_id}"
                )
            if sum(int(value) for value in expected.values()) != 1:
                raise AlfWorldActorControlError("live-menu pair outcomes do not partition")
            raw_success_count += int(raw_success)
            menu_success_count += int(menu_success)
            for name, value in expected.items():
                pair_counts[name] += int(value)
                task_counts[name] += int(value)
        task_metrics = {
            "menu_rescue_rate": task_counts["menu_rescue"] / len(seeds),
            "menu_harm_rate": task_counts["menu_harm"] / len(seeds),
            "raw_success_rate": raw_success_count / len(seeds),
            "menu_success_rate": menu_success_count / len(seeds),
            "menu_success_delta": (menu_success_count - raw_success_count) / len(seeds),
        }
        for name, value in task_metrics.items():
            metric_values[name].append(value)
        per_task.append(
            {
                "candidate_id": candidate_id,
                "task_id": case.get("task_id"),
                "gold_step_idx": case.get("gold_step_idx"),
                "pair_counts": task_counts,
                "seed_averaged_metrics": task_metrics,
            }
        )

    task_clustered_metrics: dict[str, Any] = {}
    for offset, (name, values) in enumerate(metric_values.items()):
        task_clustered_metrics[name] = {
            "estimate": sum(values) / len(values),
            "confidence_interval_95": list(
                _task_bootstrap_interval(
                    values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + offset,
                )
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed + offset,
            "resampling_unit": "task",
        }
    summary: dict[str, Any] = {
        "schema": "lh_alfworld_live_menu_constraint_population_summary/v1",
        "case_count": len(cases),
        "task_count": len(cases),
        "paired_seed_count": len(cases) * len(seeds),
        "pair_counts": dict(sorted(pair_counts.items())),
        "task_clustered_metrics": task_clustered_metrics,
        "per_task": per_task,
        "primary_estimand": "mean_task_success_T_menu_minus_T_raw",
        "claim_scope": (
            "response_contract_aligned_live_menu_constraint_effect_after_a_"
            "known_correct_historical_action_under_the_frozen_actor"
        ),
        "out_of_menu_mediation_tested": False,
        "tracer_accuracy_tested": False,
    }
    summary["content_sha256"] = "sha256:" + canonical_sha256(summary)
    return summary


def build_alfworld_actor_control_pilot_manifest(
    *,
    causal_evaluation_manifest: Mapping[str, Any],
    controlled_fault_manifest: Mapping[str, Any],
    source_collection_result: Mapping[str, Any],
    source_file_sha256: str,
    causal_file_sha256: str,
    source_result_file_sha256: str,
    sampling_seeds: Sequence[int],
    max_episode_steps: int,
    task_limit: int,
) -> dict[str, Any]:
    """Join frozen causal gold to source actions without reading actor outcomes."""

    if causal_evaluation_manifest.get("schema") != (
        "lh_alfworld_controlled_fault_evaluation_dataset/v1"
    ):
        raise AlfWorldActorControlError("causal evaluation schema is unsupported")
    if controlled_fault_manifest.get("schema") not in {
        "lh_alfworld_controlled_fault_batch_manifest/v1",
        "lh_alfworld_controlled_fault_batch_manifest/v2",
    }:
        raise AlfWorldActorControlError("controlled-fault manifest schema is unsupported")
    if source_collection_result.get("schema") != ("lh_qwen3_n1_target_source_collection_result/v1"):
        raise AlfWorldActorControlError("source collection result schema is unsupported")

    input_hashes = {
        "controlled_fault_manifest_file_sha256": _require_raw_sha256(
            "controlled-fault manifest file SHA-256", source_file_sha256
        ),
        "causal_evaluation_manifest_file_sha256": _require_raw_sha256(
            "causal evaluation manifest file SHA-256", causal_file_sha256
        ),
        "source_collection_result_file_sha256": _require_raw_sha256(
            "source collection result file SHA-256", source_result_file_sha256
        ),
    }
    frozen_seeds = tuple(sampling_seeds)
    if (
        not frozen_seeds
        or len(set(frozen_seeds)) != len(frozen_seeds)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in frozen_seeds
        )
    ):
        raise AlfWorldActorControlError("sampling seeds must be unique non-negative integers")
    if isinstance(max_episode_steps, bool) or max_episode_steps <= 0:
        raise AlfWorldActorControlError("max episode steps must be positive")
    if isinstance(task_limit, bool) or task_limit <= 0:
        raise AlfWorldActorControlError("task limit must be positive")

    raw_candidates = controlled_fault_manifest.get("candidates")
    raw_cases = causal_evaluation_manifest.get("cases")
    if not isinstance(raw_candidates, list) or not isinstance(raw_cases, list):
        raise AlfWorldActorControlError("causal/source case lists are missing")
    candidates: dict[str, Mapping[str, Any]] = {}
    for raw_candidate in raw_candidates:
        candidate = _require_mapping("controlled-fault candidate", raw_candidate)
        candidate_id = _require_nonempty_string("candidate ID", candidate.get("candidate_id"))
        if candidate_id in candidates:
            raise AlfWorldActorControlError("controlled-fault candidate IDs are duplicated")
        candidates[candidate_id] = candidate

    selected: list[dict[str, Any]] = []
    selected_tasks: set[str] = set()
    for raw_case in raw_cases:
        case = _require_mapping("causal case", raw_case)
        task_id = _require_nonempty_string("causal case task ID", case.get("task_id"))
        if task_id in selected_tasks:
            continue
        candidate_id = _require_nonempty_string(
            "causal case candidate ID", case.get("candidate_id")
        )
        source = candidates.get(candidate_id)
        if source is None:
            raise AlfWorldActorControlError("causal case is missing its source candidate")
        source_task_id = _require_nonempty_string("source candidate task ID", source.get("task_id"))
        step_idx = _require_nonnegative_int("source step index", source.get("step_idx"))
        gold_step_idx = _require_nonnegative_int(
            "causal gold step index", case.get("gold_step_idx")
        )
        if source_task_id != task_id or step_idx != gold_step_idx:
            raise AlfWorldActorControlError("causal gold/source binding changed")
        source_action = _require_nonempty_string("source action", source.get("source_action"))
        injected_action = _require_nonempty_string("injected action", source.get("injected_action"))
        if source_action == injected_action:
            raise AlfWorldActorControlError("fault action is identical to the source action")
        remaining_budget = max_episode_steps - step_idx - 1
        suffix_action_count = _require_nonnegative_int(
            "source suffix action count", source.get("source_suffix_action_count")
        )
        if remaining_budget < 0 or suffix_action_count > remaining_budget:
            raise AlfWorldActorControlError("source suffix exceeds the frozen interaction budget")

        seed_list = list(frozen_seeds)
        selected.append(
            {
                "candidate_id": candidate_id,
                "task_id": task_id,
                "traj_uid": _require_nonempty_string("trajectory UID", source.get("traj_uid")),
                "gold_step_idx": gold_step_idx,
                "causal_gold_path": _require_nonempty_string(
                    "causal gold path", case.get("causal_gold_path")
                ),
                "causal_gold_file_sha256": _require_raw_sha256(
                    "causal gold file SHA-256", case.get("causal_gold_file_sha256")
                ),
                "source_path": _require_nonempty_string(
                    "source trajectory path", source.get("source_path")
                ),
                "source_file_sha256": _require_raw_sha256(
                    "source trajectory file SHA-256", source.get("source_file_sha256")
                ),
                "source_trajectory_sha256": _require_raw_sha256(
                    "source trajectory SHA-256", source.get("source_trajectory_sha256")
                ),
                "injection_plan_sha256": _require_prefixed_sha256(
                    "injection plan SHA-256", source.get("injection_plan_sha256")
                ),
                "trace_observability_class": _require_nonempty_string(
                    "trace observability class", case.get("trace_observability_class")
                ),
                "source_suffix_action_count": suffix_action_count,
                "remaining_interaction_budget": remaining_budget,
                "arms": {
                    "B_actor": {
                        "intervention_action": injected_action,
                        "continuation_authority": "frozen_actor_closed_loop",
                        "paired_sampling_seeds": seed_list,
                    },
                    "T_suffix": {
                        "intervention_action": source_action,
                        "continuation_authority": "recorded_successful_source_suffix",
                        "suffix_controller_sha256": _require_prefixed_sha256(
                            "suffix controller SHA-256",
                            source.get("suffix_controller_sha256"),
                        ),
                    },
                    "T_actor": {
                        "intervention_action": source_action,
                        "continuation_authority": "frozen_actor_closed_loop",
                        "paired_sampling_seeds": seed_list,
                    },
                },
            }
        )
        selected_tasks.add(task_id)
        if len(selected) >= task_limit:
            break
    if not selected:
        raise AlfWorldActorControlError("actor-control selection is empty")

    policy_identity = _require_mapping(
        "source policy identity", source_collection_result.get("policy_identity")
    )
    actor_identity = {
        "system_cell_id": _require_nonempty_string(
            "system cell ID", source_collection_result.get("system_cell_id")
        ),
        "system_cell_sha256": _require_raw_sha256(
            "system cell SHA-256", source_collection_result.get("system_cell_sha256")
        ),
        "parameter_checkpoint_sha256": _require_raw_sha256(
            "parameter checkpoint SHA-256",
            policy_identity.get("parameter_checkpoint_sha256"),
        ),
        "runtime_policy_identity_sha256": _require_raw_sha256(
            "runtime policy identity SHA-256",
            policy_identity.get("runtime_policy_identity_sha256"),
        ),
        "composite_policy_identity_sha256": _require_raw_sha256(
            "composite policy identity SHA-256",
            policy_identity.get("composite_policy_identity_sha256"),
        ),
    }
    admission_case = min(
        selected,
        key=lambda case: (
            int(case["remaining_interaction_budget"]),
            str(case["candidate_id"]),
        ),
    )
    manifest: dict[str, Any] = {
        "schema": ACTOR_CONTROL_PILOT_MANIFEST_SCHEMA,
        "claim": "known_correct_action_survival_under_frozen_actor_continuation",
        "selection_rule": ACTOR_CONTROL_SELECTION_RULE,
        "selection_uses_actor_control_outcomes": False,
        "outcome_fields_present": False,
        "input_file_sha256": input_hashes,
        "actor_identity": actor_identity,
        "sampling_seeds": list(frozen_seeds),
        "max_episode_steps": max_episode_steps,
        "selected_task_count": len(selected),
        "selected_case_count": len(selected),
        "admission_case_candidate_id": admission_case["candidate_id"],
        "admission_case_selection_rule": ("minimum_remaining_interaction_budget_then_candidate_id"),
        "necessity_minimality_uniqueness_tested": False,
        "cases": selected,
    }
    manifest["content_sha256"] = "sha256:" + canonical_sha256(manifest)
    return manifest


def _direct_step(branch: AlfWorldBranch, action: str) -> Any:
    direct = getattr(branch, "step_unverified", None)
    return direct(action) if callable(direct) else branch.step(action)


def _finish_arm(
    *,
    branch: AlfWorldBranch,
    arm: str,
    applied_action: str,
    sampling_seed_base: int | None,
    executed_actions: list[str],
    actor_steps: list[dict[str, Any]],
    source_suffix_outside_admissible_steps: list[int],
    last_step: Any,
    terminal_reason: str,
    pre_state_sha256: str,
    pre_state_admissible_actions: tuple[str, ...],
    intervention_restore_receipt_sha256: str,
) -> dict[str, Any]:
    final_checkpoint = branch.capture_checkpoint()
    return {
        "arm": arm,
        "applied_action": applied_action,
        "sampling_seed_base": sampling_seed_base,
        "success": bool(last_step is not None and last_step.won),
        "terminal_reason": terminal_reason,
        "pre_state_identity_verified": True,
        "pre_intervention_state_sha256": pre_state_sha256,
        "pre_intervention_admissible_actions": list(pre_state_admissible_actions),
        "intervention_restore_receipt_sha256": intervention_restore_receipt_sha256,
        "executed_actions": list(executed_actions),
        "actor_steps": actor_steps,
        "source_suffix_outside_admissible_steps": (source_suffix_outside_admissible_steps),
        "final_state_snapshot_sha256": (final_checkpoint.snapshot.state_snapshot_sha256),
        "physical_call_counts": dict(branch.physical_call_counters()),
    }


def _replay_actor_control_arm(
    *,
    plan: AlfWorldControlledFaultInjectionPlanV1,
    branch_factory: Callable[[], AlfWorldBranch],
    actor_policy: ClosedLoopContinuationPolicy,
    arm: str,
    intervention_action: str,
    sampling_seed_base: int | None,
    remaining_interaction_budget: int,
) -> dict[str, Any]:
    branch = branch_factory()
    executed_actions: list[str] = []
    actor_history: list[tuple[str, str]] = []
    actor_steps: list[dict[str, Any]] = []
    source_suffix_outside_admissible_steps: list[int] = []
    last_step: Any = None
    try:
        current = branch.reset()
        target_step_idx = plan.injection.step_idx
        for source_step in plan.steps[:target_step_idx]:
            checkpoint = branch.capture_checkpoint()
            if (
                checkpoint.snapshot.state_snapshot_sha256 != source_step.state_snapshot_sha256
                or tuple(current.admissible_actions) != source_step.admissible_actions
            ):
                raise AlfWorldActorControlError(
                    f"{arm} prefix state/menu identity changed at step {source_step.step_idx}"
                )
            actor_history.append((current.text, source_step.source_action))
            last_step = _direct_step(branch, source_step.source_action)
            executed_actions.append(source_step.source_action)
            current = last_step.observation
            if last_step.won or last_step.done:
                raise AlfWorldActorControlError(f"{arm} terminated before the intervention step")

        source_step = plan.steps[target_step_idx]
        checkpoint = branch.capture_checkpoint()
        pre_state_sha256 = checkpoint.snapshot.state_snapshot_sha256
        pre_state_admissible_actions = tuple(current.admissible_actions)
        if (
            pre_state_sha256 != plan.injection.source_pre_state_snapshot_sha256
            or pre_state_sha256 != source_step.state_snapshot_sha256
            or pre_state_admissible_actions != source_step.admissible_actions
            or intervention_action not in pre_state_admissible_actions
        ):
            raise AlfWorldActorControlError(f"{arm} intervention state/menu binding changed")

        intervention_observation = current.text
        last_step = branch.step(intervention_action)
        executed_actions.append(intervention_action)
        actor_history.append((intervention_observation, intervention_action))
        current = last_step.observation
        restore_receipt = str(last_step.restore_receipt_sha256)

        if arm == "T_suffix":
            suffix_diverged = False
            for suffix_step in plan.steps[target_step_idx + 1 :]:
                checkpoint = branch.capture_checkpoint()
                if (
                    checkpoint.snapshot.state_snapshot_sha256 != suffix_step.state_snapshot_sha256
                    or tuple(current.admissible_actions) != suffix_step.admissible_actions
                ):
                    suffix_diverged = True
                    break
                if suffix_step.source_action not in current.admissible_actions:
                    source_suffix_outside_admissible_steps.append(suffix_step.step_idx)
                last_step = _direct_step(branch, suffix_step.source_action)
                executed_actions.append(suffix_step.source_action)
                current = last_step.observation
                if last_step.won or last_step.done:
                    break
            terminal_reason = (
                "suffix_diverged"
                if suffix_diverged
                else "environment_won"
                if last_step.won
                else "environment_done"
                if last_step.done
                else "source_suffix_exhausted"
            )
        else:
            if sampling_seed_base is None:
                raise AlfWorldActorControlError(f"{arm} lacks a paired sampling seed")
            continuation_ordinal = 0
            while continuation_ordinal < remaining_interaction_budget and not (
                last_step.won or last_step.done
            ):
                sampling_seed = sampling_seed_base + continuation_ordinal
                action, raw_policy_evidence = actor_policy.choose_action(
                    arm_id=f"{plan.injection.injection_id}:{arm}:{sampling_seed_base}",
                    task_id=plan.task_id,
                    step_idx=target_step_idx + 1 + continuation_ordinal,
                    task_description=plan.task_description,
                    current_observation=current.text,
                    admissible_actions=tuple(current.admissible_actions),
                    history=tuple(actor_history),
                    remaining_budget=(remaining_interaction_budget - continuation_ordinal),
                    sampling_seed=sampling_seed,
                    sampling_temperature=plan.sampling_temperature,
                )
                policy_evidence = dict(raw_policy_evidence)
                visible_input_sha256 = policy_evidence.get("actor_visible_input_sha256")
                if (
                    policy_evidence.get("sampling_seed") != sampling_seed
                    or policy_evidence.get("remaining_budget")
                    != remaining_interaction_budget - continuation_ordinal
                    or not isinstance(visible_input_sha256, str)
                    or not visible_input_sha256.startswith("sha256:")
                ):
                    raise AlfWorldActorControlError(
                        f"{arm} actor-visible/seed/budget receipt is incomplete"
                    )
                policy_evidence["action_in_live_admissible_menu"] = (
                    action in current.admissible_actions
                )
                before_text = current.text
                last_step = _direct_step(branch, action)
                executed_actions.append(action)
                actor_history.append((before_text, action))
                actor_steps.append(policy_evidence)
                current = last_step.observation
                continuation_ordinal += 1
            terminal_reason = (
                "environment_won"
                if last_step.won
                else "environment_done"
                if last_step.done
                else "budget_exhausted"
            )

        return _finish_arm(
            branch=branch,
            arm=arm,
            applied_action=intervention_action,
            sampling_seed_base=sampling_seed_base,
            executed_actions=executed_actions,
            actor_steps=actor_steps,
            source_suffix_outside_admissible_steps=(source_suffix_outside_admissible_steps),
            last_step=last_step,
            terminal_reason=terminal_reason,
            pre_state_sha256=pre_state_sha256,
            pre_state_admissible_actions=pre_state_admissible_actions,
            intervention_restore_receipt_sha256=restore_receipt,
        )
    finally:
        branch.close()


def _validate_actor_control_execution_inputs(
    *,
    plan: AlfWorldControlledFaultInjectionPlanV1,
    sampling_seeds: Sequence[int],
    max_episode_steps: int,
) -> tuple[int, tuple[int, ...], int]:
    target_step_idx = plan.injection.step_idx
    if (
        tuple(step.step_idx for step in plan.steps) != tuple(range(len(plan.steps)))
        or target_step_idx < 0
        or target_step_idx >= len(plan.steps)
    ):
        raise AlfWorldActorControlError("controlled-fault plan steps are invalid")
    target_step = plan.steps[target_step_idx]
    if (
        plan.injection.source_action != target_step.source_action
        or plan.injection.source_pre_state_snapshot_sha256 != target_step.state_snapshot_sha256
        or plan.injection.injected_action == plan.injection.source_action
        or plan.injection.injected_action not in target_step.admissible_actions
    ):
        raise AlfWorldActorControlError("controlled-fault intervention binding changed")
    frozen_seeds = tuple(sampling_seeds)
    if (
        not frozen_seeds
        or len(frozen_seeds) != len(set(frozen_seeds))
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in frozen_seeds
        )
    ):
        raise AlfWorldActorControlError("paired sampling seeds are invalid")
    if (
        isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
        or max_episode_steps <= 0
    ):
        raise AlfWorldActorControlError("max episode steps must be a positive integer")
    remaining_budget = max_episode_steps - target_step_idx - 1
    if remaining_budget < plan.post_injection_source_action_count:
        raise AlfWorldActorControlError("interaction budget cannot replay the source suffix")
    return target_step_idx, frozen_seeds, remaining_budget


def run_alfworld_actor_control_case(
    *,
    plan: AlfWorldControlledFaultInjectionPlanV1,
    branch_factory: Callable[[], AlfWorldBranch],
    actor_policy: ClosedLoopContinuationPolicy,
    sampling_seeds: Sequence[int],
    max_episode_steps: int,
) -> dict[str, Any]:
    """Run one known-fault case with paired actor and successful-suffix arms."""

    target_step_idx, frozen_seeds, remaining_budget = _validate_actor_control_execution_inputs(
        plan=plan,
        sampling_seeds=sampling_seeds,
        max_episode_steps=max_episode_steps,
    )

    suffix = _replay_actor_control_arm(
        plan=plan,
        branch_factory=branch_factory,
        actor_policy=actor_policy,
        arm="T_suffix",
        intervention_action=plan.injection.source_action,
        sampling_seed_base=None,
        remaining_interaction_budget=remaining_budget,
    )
    actor_pairs = []
    for seed in frozen_seeds:
        baseline = _replay_actor_control_arm(
            plan=plan,
            branch_factory=branch_factory,
            actor_policy=actor_policy,
            arm="B_actor",
            intervention_action=plan.injection.injected_action,
            sampling_seed_base=seed,
            remaining_interaction_budget=remaining_budget,
        )
        treatment = _replay_actor_control_arm(
            plan=plan,
            branch_factory=branch_factory,
            actor_policy=actor_policy,
            arm="T_actor",
            intervention_action=plan.injection.source_action,
            sampling_seed_base=seed,
            remaining_interaction_budget=remaining_budget,
        )
        if (
            baseline["pre_intervention_state_sha256"] != treatment["pre_intervention_state_sha256"]
            or baseline["pre_intervention_admissible_actions"]
            != treatment["pre_intervention_admissible_actions"]
        ):
            raise AlfWorldActorControlError("paired actor arms restored different pre-states")
        actor_pairs.append(
            {
                "sampling_seed_base": seed,
                "B_actor": baseline,
                "T_actor": treatment,
                "repair_masked": bool(suffix["success"] and not treatment["success"]),
                "actor_control_supported": bool(
                    suffix["success"] and treatment["success"] and not baseline["success"]
                ),
                "baseline_natural_recovery": bool(baseline["success"]),
            }
        )
    receipt: dict[str, Any] = {
        "schema": "lh_alfworld_actor_control_case/v1",
        "task_id": plan.task_id,
        "traj_uid": plan.traj_uid,
        "injection_plan_sha256": plan.sha256,
        "gold_step_idx": target_step_idx,
        "max_episode_steps": max_episode_steps,
        "remaining_interaction_budget": remaining_budget,
        "paired_sampling_seeds": list(frozen_seeds),
        "T_suffix": suffix,
        "actor_pairs": actor_pairs,
        "necessity_minimality_uniqueness_tested": False,
    }
    receipt["content_sha256"] = "sha256:" + canonical_sha256(receipt)
    return receipt


def run_alfworld_live_menu_constraint_case(
    *,
    plan: AlfWorldControlledFaultInjectionPlanV1,
    branch_factory: Callable[[], AlfWorldBranch],
    raw_actor_policy: ClosedLoopContinuationPolicy,
    menu_actor_policy: ClosedLoopContinuationPolicy,
    sampling_seeds: Sequence[int],
    max_episode_steps: int,
    treatment_component_sha256: str,
) -> dict[str, Any]:
    """Compare two continuation languages after the same known-correct action.

    Both arms restore the same pre-intervention state, execute the admitted
    source action, and reuse the same seed schedule.  The only intended
    treatment is free response generation versus a response-contract-aligned
    token trie over the live environment menu.
    """

    _require_prefixed_sha256(
        "live-menu treatment component SHA-256",
        treatment_component_sha256,
    )
    target_step_idx, frozen_seeds, remaining_budget = _validate_actor_control_execution_inputs(
        plan=plan,
        sampling_seeds=sampling_seeds,
        max_episode_steps=max_episode_steps,
    )
    suffix = _replay_actor_control_arm(
        plan=plan,
        branch_factory=branch_factory,
        actor_policy=raw_actor_policy,
        arm="T_suffix",
        intervention_action=plan.injection.source_action,
        sampling_seed_base=None,
        remaining_interaction_budget=remaining_budget,
    )
    if suffix["success"] is not True:
        raise AlfWorldActorControlError(
            "live-menu experiment requires a successful frozen source suffix"
        )

    actor_pairs: list[dict[str, Any]] = []
    for seed in frozen_seeds:
        raw = _replay_actor_control_arm(
            plan=plan,
            branch_factory=branch_factory,
            actor_policy=raw_actor_policy,
            arm="T_raw",
            intervention_action=plan.injection.source_action,
            sampling_seed_base=seed,
            remaining_interaction_budget=remaining_budget,
        )
        menu = _replay_actor_control_arm(
            plan=plan,
            branch_factory=branch_factory,
            actor_policy=menu_actor_policy,
            arm="T_menu",
            intervention_action=plan.injection.source_action,
            sampling_seed_base=seed,
            remaining_interaction_budget=remaining_budget,
        )
        if (
            raw["pre_intervention_state_sha256"] != menu["pre_intervention_state_sha256"]
            or raw["pre_intervention_admissible_actions"]
            != menu["pre_intervention_admissible_actions"]
        ):
            raise AlfWorldActorControlError("paired live-menu arms restored different pre-states")
        raw_steps = raw["actor_steps"]
        menu_steps = menu["actor_steps"]
        if not raw_steps or not menu_steps:
            raise AlfWorldActorControlError(
                "live-menu comparison requires at least one continuation action"
            )
        if any(step.get("action_selection_mode") != "free_generation" for step in raw_steps) or any(
            step.get("action_selection_mode") != "serialized_live_menu_trie"
            or step.get("action_in_live_admissible_menu") is not True
            for step in menu_steps
        ):
            raise AlfWorldActorControlError(
                "live-menu action-selection receipt is not bound to its arm"
            )
        first_input_identical = raw_steps[0].get("actor_visible_input_sha256") == menu_steps[0].get(
            "actor_visible_input_sha256"
        )
        if not first_input_identical:
            raise AlfWorldActorControlError(
                "paired live-menu arms expose different first continuation inputs"
            )
        raw_first_action = raw["executed_actions"][target_step_idx + 1]
        menu_first_action = menu["executed_actions"][target_step_idx + 1]
        actor_pairs.append(
            {
                "sampling_seed_base": seed,
                "T_raw": raw,
                "T_menu": menu,
                "first_continuation_input_identical": True,
                "first_action_diverged": raw_first_action != menu_first_action,
                "menu_rescue": bool(not raw["success"] and menu["success"]),
                "menu_harm": bool(raw["success"] and not menu["success"]),
                "both_success": bool(raw["success"] and menu["success"]),
                "both_failed": bool(not raw["success"] and not menu["success"]),
            }
        )

    receipt: dict[str, Any] = {
        "schema": "lh_alfworld_live_menu_constraint_case/v1",
        "claim_scope": (
            "effect_of_response_contract_aligned_live_menu_constraint_after_"
            "known_correct_historical_action"
        ),
        "task_id": plan.task_id,
        "traj_uid": plan.traj_uid,
        "injection_plan_sha256": plan.sha256,
        "gold_step_idx": target_step_idx,
        "max_episode_steps": max_episode_steps,
        "remaining_interaction_budget": remaining_budget,
        "paired_sampling_seeds": list(frozen_seeds),
        "treatment_component_sha256": treatment_component_sha256,
        "T_suffix": suffix,
        "actor_pairs": actor_pairs,
        "out_of_menu_mediation_tested": False,
        "tracer_accuracy_tested": False,
    }
    receipt["content_sha256"] = "sha256:" + canonical_sha256(receipt)
    return receipt


__all__ = [
    "ACTOR_CONTROL_PILOT_MANIFEST_SCHEMA",
    "ACTOR_CONTROL_SELECTION_RULE",
    "FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_CONTENT_SHA256",
    "FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_FILE_SHA256",
    "AlfWorldActorControlError",
    "build_alfworld_actor_control_population_summary",
    "build_alfworld_live_menu_constraint_population_summary",
    "build_alfworld_actor_control_pilot_manifest",
    "resolve_alfworld_actor_control_execution_case",
    "run_alfworld_actor_control_case",
    "run_alfworld_live_menu_constraint_case",
    "validate_frozen_alfworld_actor_control_population_manifest",
]
