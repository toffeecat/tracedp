"""Turn frozen-estimator support gaps into an auditable collection plan."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Hashable, Sequence

from .contracts import CreditVisit, validate_visits

LOTO_COLLECTION_PLAN_SCHEMA = "lh_loto_collection_plan/v1"
LOTO_COLLECTION_PLAN_CLAIM_SCOPE = (
    "collection_priority_from_frozen_support_not_guaranteed_key_recurrence"
)


def build_loto_collection_plan(
    visits: Sequence[CreditVisit],
    *,
    required_fine_ess_after_exclusion: int = 2,
) -> dict[str, object]:
    """Identify tasks whose observed keys cannot support the registered LOTO study.

    A coarse key needs two independent trajectories for leave-one-trajectory-out
    coverage.  A fine key needs ``required_fine_ess_after_exclusion + 1``
    trajectories so every held-out member retains the requested fine ESS.
    Suggested counts are lower bounds on *matching key hits*, not promises that
    an additional rollout will revisit an emergent observation key.
    """

    rows = validate_visits(visits)
    if required_fine_ess_after_exclusion < 1:
        raise ValueError("required fine ESS after exclusion must be positive")
    evaluation = tuple(row for row in rows if row.evaluation_visit)
    by_coarse = _group_trajectories(rows, key="coarse")
    by_phase = _group_trajectories(rows, key="phase")

    coarse_shortfalls = []
    for coarse_key, trajectories in by_coarse.items():
        if len(trajectories) >= 2:
            continue
        targets = [row for row in evaluation if row.coarse_key == coarse_key]
        if not targets:
            continue
        coarse_shortfalls.append(
            _shortfall_payload(
                key=coarse_key,
                rows=targets,
                trajectory_count=len(trajectories),
                required_trajectory_count=2,
            )
        )

    required_fine_trajectories = required_fine_ess_after_exclusion + 1
    fine_shortfalls = []
    for phase_key, trajectories in by_phase.items():
        if len(trajectories) >= required_fine_trajectories:
            continue
        targets = [row for row in evaluation if row.phase_key == phase_key]
        if not targets:
            continue
        fine_shortfalls.append(
            _shortfall_payload(
                key=phase_key,
                rows=targets,
                trajectory_count=len(trajectories),
                required_trajectory_count=required_fine_trajectories,
            )
        )

    tasks: dict[str, dict[str, object]] = {}
    task_rows: dict[str, list[CreditVisit]] = defaultdict(list)
    for row in rows:
        task_rows[row.task_id].append(row)
    for task_id, owned_rows in sorted(task_rows.items()):
        owned_coarse = [item for item in coarse_shortfalls if task_id in item["task_ids"]]
        owned_fine = [item for item in fine_shortfalls if task_id in item["task_ids"]]
        outcomes = {row.success for row in owned_rows}
        outcome_deficit = 0 if len(outcomes) > 1 else 1
        key_deficits = [
            int(item["additional_matching_trajectories_required"])
            for item in (*owned_coarse, *owned_fine)
        ]
        lower_bound = max([outcome_deficit, *key_deficits], default=0)
        if owned_coarse:
            collection_strategy = "target_task_and_monitor_coarse_key_recurrence"
        elif owned_fine:
            collection_strategy = "target_task_and_monitor_phase_key_recurrence"
        elif outcome_deficit:
            collection_strategy = "target_task_for_outcome_variation"
        else:
            collection_strategy = "none"
        tasks[task_id] = {
            "trajectory_count": len({row.trajectory_uid for row in owned_rows}),
            "has_outcome_variation": len(outcomes) > 1,
            "coarse_shortfall_group_count": len(owned_coarse),
            "fine_shortfall_group_count": len(owned_fine),
            "minimum_additional_matching_trajectories": lower_bound,
            "collection_strategy": collection_strategy,
            "priority": (
                "high" if owned_coarse or outcome_deficit else "medium" if owned_fine else "none"
            ),
        }

    supported_evaluation = sum(
        len(by_coarse[row.coarse_key] - {row.trajectory_uid}) > 0 for row in evaluation
    )
    return {
        "schema": LOTO_COLLECTION_PLAN_SCHEMA,
        "claim_scope": LOTO_COLLECTION_PLAN_CLAIM_SCOPE,
        "required_fine_ess_after_exclusion": required_fine_ess_after_exclusion,
        "evaluation_visit_count": len(evaluation),
        "current_coarse_loto_coverage": (
            supported_evaluation / len(evaluation) if evaluation else 0.0
        ),
        "coarse_shortfalls": sorted(coarse_shortfalls, key=lambda item: str(item["key_sha256"])),
        "fine_shortfalls": sorted(fine_shortfalls, key=lambda item: str(item["key_sha256"])),
        "task_priorities": tasks,
        "strategy_task_counts": {
            strategy: sum(item["collection_strategy"] == strategy for item in tasks.values())
            for strategy in (
                "target_task_and_monitor_coarse_key_recurrence",
                "target_task_and_monitor_phase_key_recurrence",
                "target_task_for_outcome_variation",
                "none",
            )
        },
        "planning_caveat": (
            "counts are lower bounds on matching key hits; emergent observation "
            "keys may not recur in the next rollout; repeated singleton keys "
            "require an anchor/sampling review rather than blind scale-up"
        ),
    }


def _group_trajectories(rows: Sequence[CreditVisit], *, key: str) -> dict[Hashable, set[str]]:
    grouped: dict[Hashable, set[str]] = defaultdict(set)
    for row in rows:
        identity = row.coarse_key if key == "coarse" else row.phase_key
        grouped[identity].add(row.trajectory_uid)
    return grouped


def _shortfall_payload(
    *,
    key: Hashable,
    rows: Sequence[CreditVisit],
    trajectory_count: int,
    required_trajectory_count: int,
) -> dict[str, object]:
    return {
        "key_sha256": _key_sha256(key),
        "task_ids": sorted({row.task_id for row in rows}),
        "evaluation_visit_count": len(rows),
        "trajectory_count": trajectory_count,
        "required_trajectory_count": required_trajectory_count,
        "additional_matching_trajectories_required": max(
            0, required_trajectory_count - trajectory_count
        ),
    }


def _key_sha256(value: Hashable) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "LOTO_COLLECTION_PLAN_CLAIM_SCOPE",
    "LOTO_COLLECTION_PLAN_SCHEMA",
    "build_loto_collection_plan",
]
