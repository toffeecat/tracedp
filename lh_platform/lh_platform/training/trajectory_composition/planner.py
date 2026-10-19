"""Deterministic bounded search over verified trajectory fragments."""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Any

from lh_platform.training.trajectory_composition.contracts import (
    CompositionContractError,
    CompositionLevel,
    CompositionPlanV1,
    FragmentRecordV1,
    canonical_sha256,
)

COMPOSITION_SEARCH_SCHEMA = "lh_composition_search_receipt/v1"


@dataclass(frozen=True)
class CompositionPlannerConfig:
    composition_level: CompositionLevel
    max_action_cost: int
    max_fragments: int = 8
    min_source_trajectories: int = 2
    require_failed_parent_sources: bool = True
    max_explored_states: int = 20_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "composition_level", CompositionLevel(self.composition_level))
        if self.max_action_cost < 1:
            raise CompositionContractError("planner max_action_cost must be positive")
        if self.max_fragments < 1:
            raise CompositionContractError("planner max_fragments must be positive")
        if self.min_source_trajectories < 2:
            raise CompositionContractError(
                "support recovery requires at least two source trajectories"
            )
        if self.max_explored_states < 1:
            raise CompositionContractError("planner max_explored_states must be positive")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "composition_level": self.composition_level.value,
            "max_action_cost": self.max_action_cost,
            "max_fragments": self.max_fragments,
            "min_source_trajectories": self.min_source_trajectories,
            "require_failed_parent_sources": self.require_failed_parent_sources,
            "max_explored_states": self.max_explored_states,
        }


@dataclass(frozen=True)
class CompositionSearchReceipt:
    status: str
    task_instance_sha256: str
    parent_contract_sha256: str
    initial_state_snapshot_sha256: str
    initial_goal_ids: tuple[str, ...]
    target_goal_ids: tuple[str, ...]
    planner_config_sha256: str
    candidate_fragment_count: int
    eligible_fragment_count: int
    explored_state_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    plan: CompositionPlanV1 | None
    schema: str = COMPOSITION_SEARCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COMPOSITION_SEARCH_SCHEMA:
            raise CompositionContractError("unsupported composition-search schema")
        if self.status not in {
            "plan_found_replay_required",
            "no_path",
            "search_limit_reached",
            "initial_already_satisfies_parent",
        }:
            raise CompositionContractError("unsupported composition-search status")
        if (self.plan is not None) != (self.status == "plan_found_replay_required"):
            raise CompositionContractError("search status and plan presence disagree")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "task_instance_sha256": self.task_instance_sha256,
            "parent_contract_sha256": self.parent_contract_sha256,
            "initial_state_snapshot_sha256": self.initial_state_snapshot_sha256,
            "initial_goal_ids": list(self.initial_goal_ids),
            "target_goal_ids": list(self.target_goal_ids),
            "planner_config_sha256": self.planner_config_sha256,
            "candidate_fragment_count": self.candidate_fragment_count,
            "eligible_fragment_count": self.eligible_fragment_count,
            "explored_state_count": self.explored_state_count,
            "rejection_counts": dict(self.rejection_counts),
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }


@dataclass(frozen=True)
class _SearchNode:
    state_snapshot_sha256: str
    goal_ids: frozenset[str]
    fragment_uids: tuple[str, ...]
    source_traj_uids: tuple[str, ...]
    canonical_actions: tuple[str, ...]

    @property
    def action_cost(self) -> int:
        return len(self.canonical_actions)


def search_composition_plan(
    fragments: tuple[FragmentRecordV1, ...] | list[FragmentRecordV1],
    *,
    task_instance_sha256: str,
    parent_contract_sha256: str,
    initial_state_snapshot_sha256: str,
    initial_goal_ids: tuple[str, ...],
    target_goal_ids: tuple[str, ...],
    config: CompositionPlannerConfig,
) -> CompositionSearchReceipt:
    """Find the lowest-action path under exact or verifier-effect semantics.

    E1 search is a proposal generator, not a state-transition oracle.  Its
    symbolic effects only prioritize candidate paths; every returned plan is
    marked replay-required and remains unusable until complete parent replay.
    """

    candidates = tuple(sorted(fragments, key=lambda item: item.fragment_uid))
    initial_goals = frozenset(initial_goal_ids)
    target_goals = frozenset(target_goal_ids)
    if not target_goals:
        raise CompositionContractError("planner target_goal_ids cannot be empty")
    if target_goals <= initial_goals:
        return _receipt(
            status="initial_already_satisfies_parent",
            task_instance_sha256=task_instance_sha256,
            parent_contract_sha256=parent_contract_sha256,
            initial_state_snapshot_sha256=initial_state_snapshot_sha256,
            initial_goal_ids=initial_goals,
            target_goal_ids=target_goals,
            config=config,
            candidates=len(candidates),
            eligible=0,
            explored=0,
            rejections=Counter(),
            plan=None,
        )

    rejections: Counter[str] = Counter()
    eligible: list[FragmentRecordV1] = []
    seen_uids: set[str] = set()
    for fragment in candidates:
        reason = _fragment_rejection_reason(
            fragment,
            task_instance_sha256=task_instance_sha256,
            parent_contract_sha256=parent_contract_sha256,
            target_goal_ids=target_goals,
            config=config,
            seen_uids=seen_uids,
        )
        if reason is not None:
            rejections[reason] += 1
            continue
        seen_uids.add(fragment.fragment_uid)
        eligible.append(fragment)

    start = _SearchNode(
        state_snapshot_sha256=initial_state_snapshot_sha256,
        goal_ids=initial_goals,
        fragment_uids=(),
        source_traj_uids=(),
        canonical_actions=(),
    )
    serial = 0
    queue: list[tuple[int, int, tuple[str, ...], int, _SearchNode]] = [(0, 0, (), serial, start)]
    best_cost: dict[tuple[str, frozenset[str], frozenset[str], frozenset[str]], int] = {
        _node_identity(start, config): 0
    }
    explored = 0
    limit_reached = False

    while queue:
        _cost, _fragment_count, _path_key, _serial, node = heapq.heappop(queue)
        explored += 1
        if explored > config.max_explored_states:
            limit_reached = True
            break
        if _is_target(node, target_goals, config):
            unique_sources = set(node.source_traj_uids)
            if len(unique_sources) >= config.min_source_trajectories:
                selected = {item.fragment_uid: item for item in eligible}
                path = tuple(selected[fragment_uid] for fragment_uid in node.fragment_uids)
                plan = _build_plan(
                    path,
                    task_instance_sha256=task_instance_sha256,
                    parent_contract_sha256=parent_contract_sha256,
                    initial_state_snapshot_sha256=initial_state_snapshot_sha256,
                    initial_goal_ids=initial_goals,
                    target_goal_ids=target_goals,
                    config=config,
                )
                return _receipt(
                    status="plan_found_replay_required",
                    task_instance_sha256=task_instance_sha256,
                    parent_contract_sha256=parent_contract_sha256,
                    initial_state_snapshot_sha256=initial_state_snapshot_sha256,
                    initial_goal_ids=initial_goals,
                    target_goal_ids=target_goals,
                    config=config,
                    candidates=len(candidates),
                    eligible=len(eligible),
                    explored=explored,
                    rejections=rejections,
                    plan=plan,
                )
        if len(node.fragment_uids) >= config.max_fragments:
            continue
        used = set(node.fragment_uids)
        for fragment in eligible:
            if fragment.fragment_uid in used:
                continue
            next_node = _apply_fragment(node, fragment, config)
            if next_node is None:
                continue
            if next_node.action_cost > config.max_action_cost:
                continue
            identity = _node_identity(next_node, config)
            previous = best_cost.get(identity)
            if previous is not None and previous <= next_node.action_cost:
                continue
            best_cost[identity] = next_node.action_cost
            serial += 1
            heapq.heappush(
                queue,
                (
                    next_node.action_cost,
                    len(next_node.fragment_uids),
                    next_node.fragment_uids,
                    serial,
                    next_node,
                ),
            )

    return _receipt(
        status="search_limit_reached" if limit_reached else "no_path",
        task_instance_sha256=task_instance_sha256,
        parent_contract_sha256=parent_contract_sha256,
        initial_state_snapshot_sha256=initial_state_snapshot_sha256,
        initial_goal_ids=initial_goals,
        target_goal_ids=target_goals,
        config=config,
        candidates=len(candidates),
        eligible=len(eligible),
        explored=explored,
        rejections=rejections,
        plan=None,
    )


def _fragment_rejection_reason(
    fragment: FragmentRecordV1,
    *,
    task_instance_sha256: str,
    parent_contract_sha256: str,
    target_goal_ids: frozenset[str],
    config: CompositionPlannerConfig,
    seen_uids: set[str],
) -> str | None:
    if fragment.fragment_uid in seen_uids:
        return "duplicate_fragment_uid"
    if fragment.task_instance_sha256 != task_instance_sha256:
        return "task_instance_mismatch"
    if fragment.parent_contract_sha256 != parent_contract_sha256:
        return "parent_contract_mismatch"
    if not fragment.independent_effect_verified:
        return "effect_not_verified"
    if fragment.training_eligible:
        return "raw_fragment_mislabeled_training_eligible"
    if config.require_failed_parent_sources and fragment.source_parent_success:
        return "successful_parent_source_forbidden"
    if (
        config.composition_level is CompositionLevel.E0_EXACT_STATE
        and target_goal_ids <= set(fragment.post_goal_ids)
        and not fragment.source_parent_success
        and fragment.source_started_from_parent_initial
    ):
        return "e0_failed_parent_rollout_terminal_inconsistency"
    return None


def _apply_fragment(
    node: _SearchNode,
    fragment: FragmentRecordV1,
    config: CompositionPlannerConfig,
) -> _SearchNode | None:
    current_goals = set(node.goal_ids)
    if not set(fragment.required_goal_ids) <= current_goals:
        return None
    if config.composition_level is CompositionLevel.E0_EXACT_STATE:
        if fragment.pre_state_snapshot_sha256 != node.state_snapshot_sha256:
            return None
        if set(fragment.pre_goal_ids) != current_goals:
            return None
        next_goals = frozenset(fragment.post_goal_ids)
        next_snapshot = fragment.post_state_snapshot_sha256
    else:
        if set(fragment.deleted_goal_ids) & set(fragment.invariant_goal_ids):
            return None
        next_goals = frozenset(
            (current_goals - set(fragment.deleted_goal_ids)) | set(fragment.added_goal_ids)
        )
        next_snapshot = fragment.post_state_snapshot_sha256
    if next_goals == node.goal_ids:
        return None
    return _SearchNode(
        state_snapshot_sha256=next_snapshot,
        goal_ids=next_goals,
        fragment_uids=(*node.fragment_uids, fragment.fragment_uid),
        source_traj_uids=(*node.source_traj_uids, fragment.source_traj_uid),
        canonical_actions=(*node.canonical_actions, *fragment.canonical_actions),
    )


def _is_target(
    node: _SearchNode,
    target_goal_ids: frozenset[str],
    config: CompositionPlannerConfig,
) -> bool:
    if not target_goal_ids <= node.goal_ids:
        return False
    # For E0, a terminal fragment from a failed parent-initial rollout is
    # rejected earlier. Exact-state plans can still use independently collected
    # child-task fragments that started from certified intermediate states.
    return True


def _node_identity(
    node: _SearchNode,
    config: CompositionPlannerConfig,
) -> tuple[str, frozenset[str], frozenset[str], frozenset[str]]:
    snapshot = (
        node.state_snapshot_sha256
        if config.composition_level is CompositionLevel.E0_EXACT_STATE
        else "symbolic_replay_required"
    )
    return (
        snapshot,
        node.goal_ids,
        frozenset(node.fragment_uids),
        frozenset(node.source_traj_uids),
    )


def _build_plan(
    path: tuple[FragmentRecordV1, ...],
    *,
    task_instance_sha256: str,
    parent_contract_sha256: str,
    initial_state_snapshot_sha256: str,
    initial_goal_ids: frozenset[str],
    target_goal_ids: frozenset[str],
    config: CompositionPlannerConfig,
) -> CompositionPlanV1:
    actions = tuple(action for fragment in path for action in fragment.canonical_actions)
    fragment_uids = tuple(fragment.fragment_uid for fragment in path)
    source_uids = tuple(fragment.source_traj_uid for fragment in path)
    body = {
        "task_instance_sha256": task_instance_sha256,
        "parent_contract_sha256": parent_contract_sha256,
        "composition_level": config.composition_level.value,
        "initial_state_snapshot_sha256": initial_state_snapshot_sha256,
        "initial_goal_ids": sorted(initial_goal_ids),
        "target_goal_ids": sorted(target_goal_ids),
        "fragment_uids": list(fragment_uids),
        "source_traj_uids": list(source_uids),
        "canonical_actions": list(actions),
        "planner_config_sha256": config.sha256,
    }
    return CompositionPlanV1(
        plan_uid=f"composition-{canonical_sha256(body)[:24]}",
        task_instance_sha256=task_instance_sha256,
        parent_contract_sha256=parent_contract_sha256,
        composition_level=config.composition_level,
        initial_state_snapshot_sha256=initial_state_snapshot_sha256,
        initial_goal_ids=tuple(sorted(initial_goal_ids)),
        target_goal_ids=tuple(sorted(target_goal_ids)),
        fragment_uids=fragment_uids,
        source_traj_uids=source_uids,
        canonical_actions=actions,
        total_action_cost=len(actions),
        all_source_parents_failed=all(not fragment.source_parent_success for fragment in path),
        exact_state_junctions=config.composition_level is CompositionLevel.E0_EXACT_STATE,
        requires_full_replay=True,
        candidate_only=True,
        synthetic_behavior_policy=False,
        planner_config_sha256=config.sha256,
    )


def _receipt(
    *,
    status: str,
    task_instance_sha256: str,
    parent_contract_sha256: str,
    initial_state_snapshot_sha256: str,
    initial_goal_ids: frozenset[str],
    target_goal_ids: frozenset[str],
    config: CompositionPlannerConfig,
    candidates: int,
    eligible: int,
    explored: int,
    rejections: Counter[str],
    plan: CompositionPlanV1 | None,
) -> CompositionSearchReceipt:
    return CompositionSearchReceipt(
        status=status,
        task_instance_sha256=task_instance_sha256,
        parent_contract_sha256=parent_contract_sha256,
        initial_state_snapshot_sha256=initial_state_snapshot_sha256,
        initial_goal_ids=tuple(sorted(initial_goal_ids)),
        target_goal_ids=tuple(sorted(target_goal_ids)),
        planner_config_sha256=config.sha256,
        candidate_fragment_count=candidates,
        eligible_fragment_count=eligible,
        explored_state_count=explored,
        rejection_counts=tuple(sorted(rejections.items())),
        plan=plan,
    )
