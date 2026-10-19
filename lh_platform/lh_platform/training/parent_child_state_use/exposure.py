"""Explicit training-time state exposure controls for actor diagnostics."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from lh_platform.training.subproblems.base import (
    GoalGraph,
    GoalGraphError,
    GoalVector,
    project_actor_state_exposure,
)

__all__ = [
    "ActorStateExposureMode",
    "build_actor_state_exposure",
    "render_actor_state_exposure_extension",
]


class ActorStateExposureMode(StrEnum):
    NORMAL = "normal"
    LOCAL_ONLY = "local_only"
    CORRUPTED_HISTORY = "corrupted_history"
    FLAT_ORACLE = "flat_oracle"
    GRAPH_ORACLE = "graph_oracle"
    CORRUPTED_GRAPH = "corrupted_graph"


def build_actor_state_exposure(
    graph: GoalGraph,
    vector: GoalVector,
    *,
    remaining_budget: int,
    mode: ActorStateExposureMode | str,
) -> dict[str, Any]:
    """Build a privileged prompt payload without calling it model awareness."""

    selected = ActorStateExposureMode(mode)
    if selected in {
        ActorStateExposureMode.NORMAL,
        ActorStateExposureMode.LOCAL_ONLY,
        ActorStateExposureMode.CORRUPTED_HISTORY,
    }:
        raise GoalGraphError(f"{selected.value} does not expose oracle state")
    projection = project_actor_state_exposure(
        graph,
        vector,
        remaining_budget=remaining_budget,
    )
    subproblems = [dict(item) for item in projection.subproblems]
    active = list(projection.active_frontier_ids)
    all_ids = [str(item["id"]) for item in subproblems]
    inactive = [node_id for node_id in all_ids if node_id not in active]
    payload: dict[str, Any] = {
        "schema": projection.schema,
        "parent_task": projection.parent_task,
        "subproblems": subproblems,
        "all_subproblem_ids": all_ids,
        "residual_goal_ids": list(projection.residual_goal_ids),
        "active_frontier_ids": active,
        "inactive_candidate_ids": inactive,
        "remaining_budget": projection.remaining_budget,
    }
    if selected is ActorStateExposureMode.FLAT_ORACLE:
        payload["subproblems"] = [
            {key: value for key, value in item.items() if key != "requires"} for item in subproblems
        ]
        return payload
    if selected is ActorStateExposureMode.CORRUPTED_GRAPH:
        _corrupt_graph_payload(payload, graph.root_id)
    return payload


def render_actor_state_exposure_extension(
    graph: GoalGraph,
    vector: GoalVector,
    *,
    remaining_budget: int,
    mode: ActorStateExposureMode | str,
) -> tuple[str, dict[str, Any]]:
    payload = build_actor_state_exposure(
        graph,
        vector,
        remaining_budget=remaining_budget,
        mode=mode,
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"<actor_state_exposure>{encoded}</actor_state_exposure>",
        payload,
    )


def _corrupt_graph_payload(payload: dict[str, Any], root_id: str) -> None:
    active = list(payload["active_frontier_ids"])
    inactive = list(payload["inactive_candidate_ids"])
    candidates = [node_id for node_id in inactive if node_id != root_id]
    if len(active) != 1 or not candidates:
        raise GoalGraphError("corrupted graph control requires one frontier and one non-root decoy")
    correct = active[0]
    decoy = candidates[0]
    payload["active_frontier_ids"] = [decoy]
    payload["inactive_candidate_ids"] = [
        correct if node_id == decoy else node_id for node_id in inactive
    ]
    subproblems = payload["subproblems"]
    by_id = {str(item["id"]): item for item in subproblems}
    by_id[correct]["status"], by_id[decoy]["status"] = (
        by_id[decoy]["status"],
        by_id[correct]["status"],
    )
    requirements = [list(item["requires"]) for item in subproblems]
    if len(requirements) > 1:
        rotated = requirements[1:] + requirements[:1]
        for item, replacement in zip(subproblems, rotated, strict=True):
            item["requires"] = replacement
