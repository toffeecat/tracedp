"""Audit residual/frontier aliasing in a completed frozen ALFWorld evaluation.

The evaluation runner deliberately records raw environment facts but does not
expose verifier state to the actor.  This module reconstructs the GoalGraph
projection after the run and measures how often a source G2PO anchor names
more than one residual/frontier state.  It is a prevalence audit only: it does
not estimate continuation value or authorize a policy update.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.studies.runtime import AlfworldTask, AlfworldTaskManifest
from lh_platform.training.dataset import sha256_file
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.base import goal_state_projection

FULL_EVALUATION_SCHEMA = "lh_agentic_rl_full_evaluation/v1"
EVALUATION_TRANSITION_SCHEMA = "lh_alfworld_evaluation_transition/v1"
ALIASING_AUDIT_SCHEMA = "lh_alfworld_full_evaluation_aliasing_audit/v1"


class AlfWorldEvaluationAliasingError(RuntimeError):
    """Raised when a completed evaluation cannot support a strict audit."""


def analyze_alfworld_evaluation_aliasing(
    result_path: str | Path,
    *,
    manifest_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Recompute verifier state and summarize natural anchor collisions."""

    result_file = Path(result_path).resolve()
    result = _load_mapping(result_file, "full evaluation result")
    if result.get("schema") != FULL_EVALUATION_SCHEMA:
        raise AlfWorldEvaluationAliasingError("unsupported full evaluation schema")
    run_id = _required_text(result, "run_id")

    trace_file = Path(_required_text(result, "evaluation_trace_path")).resolve()
    if not trace_file.is_file():
        raise AlfWorldEvaluationAliasingError("evaluation trace is missing")
    expected_trace_sha256 = _required_sha256(result, "evaluation_trace_sha256")
    if sha256_file(trace_file) != expected_trace_sha256:
        raise AlfWorldEvaluationAliasingError("evaluation trace SHA-256 mismatch")
    trace_rows = _load_trace(trace_file, run_id=run_id)
    expected_trace_rows = _required_nonnegative_int(result, "evaluation_trace_rows")
    if len(trace_rows) != expected_trace_rows:
        raise AlfWorldEvaluationAliasingError("evaluation trace row count mismatch")

    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise AlfWorldEvaluationAliasingError("evaluation manifest is missing")
    expected_manifest_sha256 = _required_sha256(result, "eval_manifest_sha256")
    if sha256_file(manifest_file) != expected_manifest_sha256:
        raise AlfWorldEvaluationAliasingError("evaluation manifest SHA-256 mismatch")
    try:
        manifest = AlfworldTaskManifest.read(manifest_file)
    except (OSError, TypeError, ValueError) as error:
        raise AlfWorldEvaluationAliasingError(
            f"cannot load evaluation manifest: {error}"
        ) from error
    if manifest.benchmark != "alfworld_native" or not manifest.tasks:
        raise AlfWorldEvaluationAliasingError("manifest is not a non-empty ALFWorld suite")

    root = Path(data_root).resolve()
    task_by_id = {task.task_id: task for task in manifest.tasks}
    if len(task_by_id) != len(manifest.tasks):
        raise AlfWorldEvaluationAliasingError("manifest task IDs are not unique")
    compiler = AlfWorldSubproblemCompiler()
    metadata_by_task: dict[str, dict[str, Any]] = {}
    graph_by_task = {}
    for task in manifest.tasks:
        metadata = _load_task_metadata(task, data_root=root)
        metadata_by_task[task.task_id] = metadata
        graph_by_task[task.task_id] = compiler.compile_task(metadata)

    by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        task_id = _required_text(row, "task_id")
        task = task_by_id.get(task_id)
        if task is None:
            raise AlfWorldEvaluationAliasingError(
                f"trace task is outside the bound manifest: {task_id}"
            )
        _assert_task_binding(row, task)
        by_trajectory[_required_text(row, "traj_uid")].append(row)

    evaluation = result.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise AlfWorldEvaluationAliasingError("full evaluation lacks summary")
    task_count = _required_nonnegative_int(evaluation, "task_count")
    if task_count != len(by_trajectory) or task_count != len(manifest.tasks):
        raise AlfWorldEvaluationAliasingError("evaluation, trace, and manifest task counts differ")

    projected_rows: list[dict[str, Any]] = []
    trajectory_success: dict[str, bool] = {}
    for trajectory_uid, rows in sorted(by_trajectory.items()):
        ordered = sorted(rows, key=lambda row: _required_nonnegative_int(row, "step_idx"))
        if [_required_nonnegative_int(row, "step_idx") for row in ordered] != list(
            range(len(ordered))
        ):
            raise AlfWorldEvaluationAliasingError(
                f"trajectory steps are not contiguous: {trajectory_uid}"
            )
        terminal_rows = [row for row in ordered if bool(row.get("terminal"))]
        if len(terminal_rows) != 1 or terminal_rows[0] is not ordered[-1]:
            raise AlfWorldEvaluationAliasingError(
                f"trajectory lacks one final terminal row: {trajectory_uid}"
            )
        task = task_by_id[_required_text(ordered[0], "task_id")]
        metadata = metadata_by_task[task.task_id]
        graph = graph_by_task[task.task_id]
        previous = None
        for row in ordered:
            before_facts = _required_sequence(row, "environment_facts")
            next_facts = _required_sequence(row, "next_environment_facts")
            before = compiler.project_state(metadata, before_facts, previous=previous)
            projection = goal_state_projection(graph, before)
            projected_rows.append(
                {
                    "traj_uid": trajectory_uid,
                    "task_id": task.task_id,
                    "group_uid": task.group_uid,
                    "task_type": task.task_type,
                    "step_idx": _required_nonnegative_int(row, "step_idx"),
                    "horizon_bucket": task.horizon_bucket,
                    "reference_steps": task.reference_steps,
                    "anchor_observation": _required_text(row, "anchor_observation"),
                    "residual_goal_signature": projection["residual_goal_signature"],
                    "frontier_signature": projection["frontier_signature"],
                    "is_action_valid": row.get("is_action_valid") is True,
                }
            )
            previous = compiler.project_state(metadata, next_facts, previous=before)
        trajectory_success[trajectory_uid] = bool(ordered[-1].get("success"))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in projected_rows:
        groups[(row["group_uid"], row["anchor_observation"])].append(row)
    candidate_collision_keys = {
        key
        for key, members in groups.items()
        if len({_fine_identity(member) for member in members}) > 1
    }
    cross_trajectory_collision_keys = {
        key
        for key in candidate_collision_keys
        if _has_cross_trajectory_fine_difference(groups[key])
    }

    trajectory_rows = _trajectory_summaries(
        by_trajectory=by_trajectory,
        projected_rows=projected_rows,
        task_by_id=task_by_id,
        trajectory_success=trajectory_success,
        candidate_collision_keys=candidate_collision_keys,
        cross_trajectory_collision_keys=cross_trajectory_collision_keys,
    )
    by_horizon = _horizon_summaries(trajectory_rows)
    by_task_type = _category_summaries(
        trajectory_rows,
        category_field="task_type",
    )
    transition_count = len(projected_rows)
    candidate_exposure_count = sum(
        (row["group_uid"], row["anchor_observation"]) in candidate_collision_keys
        for row in projected_rows
    )
    cross_exposure_count = sum(
        (row["group_uid"], row["anchor_observation"]) in cross_trajectory_collision_keys
        for row in projected_rows
    )
    success_count = sum(trajectory_success.values())
    declared_success_count = _required_nonnegative_int(evaluation, "success_count")
    if success_count != declared_success_count:
        raise AlfWorldEvaluationAliasingError("trace success count differs from evaluation result")

    payload = {
        "schema": ALIASING_AUDIT_SCHEMA,
        "claim_scope": (
            "frozen_actor_verifier_state_aliasing_prevalence_and_association_"
            "not_continuation_calibration_not_policy_improvement_not_causality"
        ),
        "source": {
            "full_evaluation_result_path": str(result_file),
            "full_evaluation_result_sha256": sha256_file(result_file),
            "evaluation_trace_path": str(trace_file),
            "evaluation_trace_sha256": expected_trace_sha256,
            "evaluation_manifest_path": str(manifest_file),
            "evaluation_manifest_sha256": expected_manifest_sha256,
            "policy_checkpoint_sha256": _required_sha256(result, "policy_checkpoint_sha256"),
            "run_id": run_id,
        },
        "annotation": {
            "compiler_id": compiler.compiler_id,
            "metadata_fields_used": [
                "task_type",
                "pddl_params.object_target",
                "pddl_params.parent_target",
                "pddl_params.toggle_target",
                "environment_facts",
            ],
            "expert_plan_used": False,
            "actor_prompt_modified": False,
            "policy_update_executed": False,
        },
        "actor": {
            "task_count": task_count,
            "success_count": success_count,
            "failure_count": task_count - success_count,
            "outcome_variation": 0 < success_count < task_count,
            "transition_count": transition_count,
            "valid_action_count": sum(int(row["is_action_valid"]) for row in projected_rows),
            "valid_action_rate": _rate(
                sum(int(row["is_action_valid"]) for row in projected_rows),
                transition_count,
            ),
        },
        "aliasing": {
            "coarse_group_count": len(groups),
            "candidate_collision_group_count": len(candidate_collision_keys),
            "candidate_collision_group_rate": _rate(len(candidate_collision_keys), len(groups)),
            "cross_trajectory_collision_group_count": len(cross_trajectory_collision_keys),
            "cross_trajectory_collision_group_rate": _rate(
                len(cross_trajectory_collision_keys), len(groups)
            ),
            "candidate_exposed_transition_count": candidate_exposure_count,
            "candidate_exposure_rate": _rate(candidate_exposure_count, transition_count),
            "cross_trajectory_exposed_transition_count": cross_exposure_count,
            "cross_trajectory_exposure_rate": _rate(cross_exposure_count, transition_count),
        },
        "association": {
            "reference_steps_vs_candidate_burden_pearson": _correlation(
                [float(row["reference_steps"]) for row in trajectory_rows],
                [float(row["candidate_aliasing_burden"]) for row in trajectory_rows],
            ),
            "reference_steps_vs_candidate_burden_spearman": _correlation(
                _ranks([float(row["reference_steps"]) for row in trajectory_rows]),
                _ranks([float(row["candidate_aliasing_burden"]) for row in trajectory_rows]),
            ),
            "task_type_fixed_effect_pearson": _fixed_effect_correlation(
                trajectory_rows,
                category_field="task_type",
            ),
            "group_uid_fixed_effect_pearson": _fixed_effect_correlation(
                trajectory_rows,
                category_field="group_uid",
            ),
            "interpretation": (
                "descriptive association under one frozen policy occupancy; "
                "no causal long-horizon claim"
            ),
        },
        "by_horizon": by_horizon,
        "by_task_type": by_task_type,
        "trajectory_rows": trajectory_rows,
        "collision_examples": _collision_examples(
            groups,
            candidate_collision_keys=candidate_collision_keys,
            cross_trajectory_collision_keys=cross_trajectory_collision_keys,
        ),
        "next_gate": {
            "multi_trajectory_value_calibration_ready": bool(cross_trajectory_collision_keys)
            and 0 < success_count < task_count,
            "reason": (
                "cross-trajectory residual/frontier collisions and outcome variation "
                "are both required before LOTO continuation calibration"
            ),
        },
    }
    atomic_write_json(output_path, payload)
    return payload


def _load_trace(path: Path, *, run_id: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AlfWorldEvaluationAliasingError(
                f"invalid evaluation trace JSON at line {line_number}"
            ) from error
        if not isinstance(row, Mapping):
            raise AlfWorldEvaluationAliasingError("evaluation trace row is not a mapping")
        if row.get("schema") != EVALUATION_TRANSITION_SCHEMA:
            raise AlfWorldEvaluationAliasingError("unsupported evaluation transition schema")
        if row.get("run_id") != run_id:
            raise AlfWorldEvaluationAliasingError("evaluation trace run ID mismatch")
        rows.append(row)
    if not rows:
        raise AlfWorldEvaluationAliasingError("evaluation trace is empty")
    sequences = [_required_nonnegative_int(row, "sequence") for row in rows]
    if sequences != list(range(len(rows))):
        raise AlfWorldEvaluationAliasingError("evaluation trace sequence is not contiguous")
    return rows


def _load_task_metadata(task: AlfworldTask, *, data_root: Path) -> dict[str, Any]:
    trajectory_path = (data_root / task.trajectory_path).resolve()
    try:
        trajectory_path.relative_to(data_root)
    except ValueError as error:
        raise AlfWorldEvaluationAliasingError(
            "ALFWorld trajectory path escapes data root"
        ) from error
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlfWorldEvaluationAliasingError(
            f"cannot load task metadata for {task.task_id}: {error}"
        ) from error
    if not isinstance(trajectory, Mapping) or trajectory.get("task_type") != task.task_type:
        raise AlfWorldEvaluationAliasingError("manifest and task metadata differ")
    params = trajectory.get("pddl_params")
    if not isinstance(params, Mapping):
        raise AlfWorldEvaluationAliasingError("task metadata lacks PDDL parameters")
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pddl_params": {
            "object_target": params.get("object_target", ""),
            "parent_target": params.get("parent_target", ""),
            "toggle_target": params.get("toggle_target", ""),
        },
    }


def _assert_task_binding(row: Mapping[str, Any], task: AlfworldTask) -> None:
    expected = {
        "split": task.split,
        "horizon_bucket": task.horizon_bucket,
        "reference_steps": task.reference_steps,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise AlfWorldEvaluationAliasingError(f"evaluation trace {field} differs from manifest")


def _fine_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["residual_goal_signature"]), str(row["frontier_signature"])


def _has_cross_trajectory_fine_difference(
    members: Iterable[Mapping[str, Any]],
) -> bool:
    rows = tuple(members)
    return any(
        left["traj_uid"] != right["traj_uid"] and _fine_identity(left) != _fine_identity(right)
        for index, left in enumerate(rows)
        for right in rows[index + 1 :]
    )


def _trajectory_summaries(
    *,
    by_trajectory: Mapping[str, list[Mapping[str, Any]]],
    projected_rows: list[dict[str, Any]],
    task_by_id: Mapping[str, AlfworldTask],
    trajectory_success: Mapping[str, bool],
    candidate_collision_keys: set[tuple[str, str]],
    cross_trajectory_collision_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    projected_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in projected_rows:
        projected_by_trajectory[row["traj_uid"]].append(row)
    summaries = []
    for trajectory_uid, rows in sorted(projected_by_trajectory.items()):
        task = task_by_id[rows[0]["task_id"]]
        count = len(rows)
        candidate_count = sum(
            (row["group_uid"], row["anchor_observation"]) in candidate_collision_keys
            for row in rows
        )
        cross_count = sum(
            (row["group_uid"], row["anchor_observation"]) in cross_trajectory_collision_keys
            for row in rows
        )
        summaries.append(
            {
                "traj_uid": trajectory_uid,
                "task_id": task.task_id,
                "group_uid": task.group_uid,
                "task_type": task.task_type,
                "horizon_bucket": task.horizon_bucket,
                "reference_steps": task.reference_steps,
                "realized_steps": count,
                "success": bool(trajectory_success[trajectory_uid]),
                "candidate_aliased_transition_count": candidate_count,
                "candidate_aliasing_burden": _rate(candidate_count, count),
                "cross_trajectory_aliased_transition_count": cross_count,
                "cross_trajectory_aliasing_burden": _rate(cross_count, count),
            }
        )
    if len(summaries) != len(by_trajectory):
        raise AlfWorldEvaluationAliasingError("trajectory projection lost rows")
    return summaries


def _horizon_summaries(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows_tuple = tuple(rows)
    summaries = _category_summaries(
        rows_tuple,
        category_field="horizon_bucket",
    )
    return {
        bucket: summaries.get(bucket, _summarize_category_members(()))
        for bucket in ("short", "medium", "long")
    }


def _category_summaries(
    rows: Iterable[Mapping[str, Any]],
    *,
    category_field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[category_field])].append(row)
    return {
        category: _summarize_category_members(members)
        for category, members in sorted(grouped.items())
    }


def _summarize_category_members(
    members: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = tuple(members)
    transition_count = sum(int(row["realized_steps"]) for row in materialized)
    candidate_count = sum(int(row["candidate_aliased_transition_count"]) for row in materialized)
    cross_count = sum(int(row["cross_trajectory_aliased_transition_count"]) for row in materialized)
    return {
        "trajectory_count": len(materialized),
        "success_count": sum(int(bool(row["success"])) for row in materialized),
        "transition_count": transition_count,
        "candidate_aliased_transition_count": candidate_count,
        "candidate_aliasing_exposure_rate": _rate(candidate_count, transition_count),
        "cross_trajectory_aliased_transition_count": cross_count,
        "cross_trajectory_aliasing_exposure_rate": _rate(cross_count, transition_count),
    }


def _collision_examples(
    groups: Mapping[tuple[str, str], list[dict[str, Any]]],
    *,
    candidate_collision_keys: set[tuple[str, str]],
    cross_trajectory_collision_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    examples = []
    for key in sorted(candidate_collision_keys):
        members = groups[key]
        examples.append(
            {
                "group_uid": key[0],
                "anchor_sha256": hashlib.sha256(key[1].encode("utf-8")).hexdigest(),
                "anchor_excerpt": " ".join(key[1].split())[:240],
                "visit_count": len(members),
                "trajectory_count": len({row["traj_uid"] for row in members}),
                "fine_state_count": len({_fine_identity(row) for row in members}),
                "cross_trajectory": key in cross_trajectory_collision_keys,
            }
        )
    return examples[:20]


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    scale_x = math.sqrt(sum(value * value for value in centered_x))
    scale_y = math.sqrt(sum(value * value for value in centered_y))
    if scale_x == 0 or scale_y == 0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / (scale_x * scale_y)


def _fixed_effect_correlation(
    rows: Iterable[Mapping[str, Any]],
    *,
    category_field: str,
) -> dict[str, Any]:
    """Correlate within-category deviations to remove composition effects."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[category_field])].append(row)
    residual_x: list[float] = []
    residual_y: list[float] = []
    informative_categories = 0
    for members in grouped.values():
        xs = [float(row["reference_steps"]) for row in members]
        ys = [float(row["candidate_aliasing_burden"]) for row in members]
        if len(members) < 2:
            continue
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        centered_x = [value - mean_x for value in xs]
        centered_y = [value - mean_y for value in ys]
        if any(value != 0 for value in centered_x) and any(value != 0 for value in centered_y):
            informative_categories += 1
        residual_x.extend(centered_x)
        residual_y.extend(centered_y)
    return {
        "category_field": category_field,
        "category_count": len(grouped),
        "informative_category_count": informative_categories,
        "trajectory_count": len(residual_x),
        "coefficient": _correlation(residual_x, residual_y),
    }


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index, _ in indexed[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlfWorldEvaluationAliasingError(f"cannot load {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise AlfWorldEvaluationAliasingError(f"{label} is not a mapping")
    return value


def _required_text(value: Mapping[str, Any], field: str) -> str:
    text = value.get(field)
    if not isinstance(text, str) or not text:
        raise AlfWorldEvaluationAliasingError(f"{field} must be non-empty text")
    return text


def _required_sha256(value: Mapping[str, Any], field: str) -> str:
    text = _required_text(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AlfWorldEvaluationAliasingError(f"{field} must be lowercase SHA-256")
    return text


def _required_nonnegative_int(value: Mapping[str, Any], field: str) -> int:
    raw = value.get(field)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise AlfWorldEvaluationAliasingError(f"{field} must be a non-negative integer")
    return raw


def _required_sequence(value: Mapping[str, Any], field: str) -> tuple[Any, ...]:
    raw = value.get(field)
    if not isinstance(raw, list):
        raise AlfWorldEvaluationAliasingError(f"{field} must be a list")
    return tuple(raw)


__all__ = [
    "ALIASING_AUDIT_SCHEMA",
    "AlfWorldEvaluationAliasingError",
    "analyze_alfworld_evaluation_aliasing",
]
