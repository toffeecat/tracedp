"""Independent G2PO credit contracts for the unified ALFWorld platform.

This module is intentionally *not* GraphGPO under another name.  Its
``g2po_source_credit`` path follows the credit semantics in the pinned G2PO
``core_g2po.py`` source: observation grouping, discounted node scores,
next-node comparison, score-gain credit, and cross-step episode statistics.

``g2po_trajectory_normalized`` removes repeated-visit and rollout-length
weighting from graph statistics. Refined state identity is owned by DPPO and
enters only through the explicit G2PO-TN extension seam below.

Group values here pool every visit in the group, including the trajectory whose
action is being credited (author semantics).  That self-inclusion leaks the
trajectory's own terminal reward into its advantage (knowledge-base ch. 07), so
these paths are author-faithful baselines and diagnostics.  Formal credit
comparisons use the trajectory-LOTO variant ``g2po_tn_loto``, which
``agentic_rl_config`` enforces for cross-fitted estimators.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.trajectory_v2 import validate_active_trajectory_records

SUCCESS_REWARD = 10.0

__all__ = [
    "SUCCESS_REWARD",
    "G2POCreditMode",
    "G2POCreditConfig",
    "G2POTransition",
    "G2POCreditAssignment",
    "assign_g2po_credit",
    "assign_g2po_credit_with_state_values",
    "assign_g2po_tn_with_state_keys",
    "transitions_from_v2_records",
    "validate_g2po_transitions",
]


class G2POCreditMode(StrEnum):
    SOURCE = "g2po_source_credit"
    TRAJECTORY_NORMALIZED = "g2po_trajectory_normalized"
    TRAJECTORY_NORMALIZED_EDGE_PRESERVING = "g2po_tn_edge_preserving_v2"


@dataclass(frozen=True)
class G2POCreditConfig:
    gamma: float = 0.95
    step_advantage_weight: float = 1.0
    normalization: str = "mean_std_norm"
    invalid_action_penalty: float = 0.1
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.normalization not in {"mean_std_norm", "mean_norm"}:
            raise ValueError("normalization must be mean_std_norm or mean_norm")
        if self.invalid_action_penalty < 0:
            raise ValueError("invalid_action_penalty must be non-negative")


@dataclass(frozen=True)
class G2POTransition:
    group_uid: str
    trajectory_id: str
    step_index: int
    anchor_observation: Any
    action: str
    next_observation: Any
    episode_reward: float
    terminal: bool
    success: bool
    action_token_count: int
    valid_action: bool = True
    active: bool = True
    reward: float = 0.0
    decision_role: str = "executor"
    planner_decision_kind: str | None = None


@dataclass(frozen=True)
class G2POCreditAssignment:
    state_key: Hashable
    next_state_key: Hashable
    group_value: float
    next_group_value: float
    episode_advantage: float
    node_advantage: float
    edge_advantage: float
    step_advantage: float
    advantage: float
    actor_token_weight: float
    bellman_residual: float


def assign_g2po_credit(
    transitions: Iterable[G2POTransition],
    *,
    mode: G2POCreditMode | str = G2POCreditMode.SOURCE,
    config: G2POCreditConfig | None = None,
) -> tuple[G2POCreditAssignment, ...]:
    """Assign baseline G2PO credit in input order and reject non-transitions.

    Both baseline modes use the same actor token mass. This makes a comparison
    about credit attribution rather than letting a longer rollout add more
    loss simply because it contains more semantic actions or tokens.
    """

    selected_mode = G2POCreditMode(mode)
    rows = validate_g2po_transitions(transitions, mode=selected_mode)
    return _assign_grouped_credit(
        rows,
        mode=selected_mode,
        state_keys=tuple(_hashable(row.anchor_observation) for row in rows),
        config=config,
    )


def assign_g2po_tn_with_state_keys(
    transitions: Iterable[G2POTransition],
    *,
    state_keys: Iterable[Hashable],
    config: G2POCreditConfig | None = None,
) -> tuple[G2POCreditAssignment, ...]:
    """Reuse G2PO-TN statistics with externally verified decision-state keys.

    This is the deliberate extension seam used by DPPO. G2PO owns the
    trajectory-normalized estimator; DPPO owns which refined states may share
    continuation credit and must validate those keys before calling here.
    """

    rows = validate_g2po_transitions(
        transitions,
        mode=G2POCreditMode.TRAJECTORY_NORMALIZED,
    )
    keys = tuple(state_keys)
    if len(keys) != len(rows):
        raise ValueError("refined state key count must match G2PO transitions")
    return _assign_grouped_credit(
        rows,
        mode=G2POCreditMode.TRAJECTORY_NORMALIZED,
        state_keys=keys,
        config=config,
    )


def assign_g2po_credit_with_state_values(
    transitions: Iterable[G2POTransition],
    *,
    state_keys: Iterable[Hashable],
    state_values: Iterable[float],
    mode: G2POCreditMode | str = G2POCreditMode.SOURCE,
    config: G2POCreditConfig | None = None,
) -> tuple[G2POCreditAssignment, ...]:
    """Run the shared G2PO advantage/loss seam with cross-fitted state values.

    PhasePool and A3 replace only the group-value estimator. Episode credit,
    node/edge normalization, invalid-action treatment, and actor token mass stay
    owned by this baseline module, preventing a treatment-specific loss fork.
    """

    selected_mode = G2POCreditMode(mode)
    rows = validate_g2po_transitions(transitions, mode=selected_mode)
    keys = tuple(state_keys)
    values = tuple(float(value) for value in state_values)
    if len(keys) != len(rows) or len(values) != len(rows):
        raise ValueError("external state keys/values must align with G2PO transitions")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("external G2PO state values must be finite")
    if not rows:
        return ()
    cfg = config or G2POCreditConfig()
    by_group: dict[str, list[int]] = defaultdict(list)
    by_trajectory: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[row.group_uid].append(index)
        by_trajectory[(row.group_uid, row.trajectory_id)].append(index)
    for indices in by_trajectory.values():
        indices.sort(key=lambda index: rows[index].step_index)
    trajectory_lengths = {
        trajectory_id: len(indices) for trajectory_id, indices in by_trajectory.items()
    }
    next_keys: list[Hashable] = [""] * len(rows)
    next_values = [0.0] * len(rows)
    for indices in by_trajectory.values():
        for position, index in enumerate(indices):
            if position + 1 < len(indices):
                following = indices[position + 1]
                next_keys[index] = keys[following]
                next_values[index] = values[following]
            else:
                next_keys[index] = _terminal_key(rows[index], selected_mode)
                next_values[index] = SUCCESS_REWARD if next_keys[index] == "__success__" else 0.0

    episode_advantages = [0.0] * len(rows)
    node_advantages = [0.0] * len(rows)
    edge_advantages = [0.0] * len(rows)
    for indices in by_group.values():
        _assign_episode_advantages(indices, rows, selected_mode, cfg, episode_advantages)
        by_state: dict[Hashable, list[int]] = defaultdict(list)
        for index in indices:
            by_state[keys[index]].append(index)
        for members in by_state.values():
            raw_node = {
                index: next_values[index]
                - (0.0 if rows[index].valid_action else cfg.invalid_action_penalty)
                for index in members
            }
            if len(members) < 2:
                normalized_node = {index: 0.0 for index in members}
            else:
                normalized_node = _dispatch_normalize(
                    raw=raw_node, rows=rows, cfg=cfg, mode=selected_mode
                )
            for index, value in normalized_node.items():
                node_advantages[index] = value
        # Preserve G2PO's score-gain edge definition when swapping only the
        # state-value estimator.  Reward-discount TD residuals remain a
        # diagnostic below; they are not the baseline edge advantage.
        raw_edge = {index: next_values[index] - values[index] for index in indices}
        normalized_edge = _dispatch_normalize(
            raw=raw_edge, rows=rows, cfg=cfg, mode=selected_mode
        )
        for index, value in normalized_edge.items():
            edge_advantages[index] = value

    output = []
    for index, row in enumerate(rows):
        step_advantage = node_advantages[index] + edge_advantages[index]
        trajectory_length = trajectory_lengths[(row.group_uid, row.trajectory_id)]
        output.append(
            G2POCreditAssignment(
                state_key=keys[index],
                next_state_key=next_keys[index],
                group_value=values[index],
                next_group_value=next_values[index],
                episode_advantage=episode_advantages[index],
                node_advantage=node_advantages[index],
                edge_advantage=edge_advantages[index],
                step_advantage=step_advantage,
                advantage=episode_advantages[index] + cfg.step_advantage_weight * step_advantage,
                actor_token_weight=1.0 / (trajectory_length * row.action_token_count),
                bellman_residual=abs(values[index] - (row.reward + cfg.gamma * next_values[index])),
            )
        )
    return tuple(output)


def _assign_grouped_credit(
    rows: tuple[G2POTransition, ...],
    *,
    mode: G2POCreditMode,
    state_keys: tuple[Hashable, ...],
    config: G2POCreditConfig | None,
) -> tuple[G2POCreditAssignment, ...]:
    if not rows:
        return ()
    cfg = config or G2POCreditConfig()

    by_group: dict[str, list[int]] = defaultdict(list)
    by_trajectory: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[row.group_uid].append(index)
        by_trajectory[(row.group_uid, row.trajectory_id)].append(index)
    for indices in by_trajectory.values():
        indices.sort(key=lambda index: rows[index].step_index)
    trajectory_lengths = {
        trajectory_id: len(indices) for trajectory_id, indices in by_trajectory.items()
    }

    next_keys: list[Hashable] = ["" for _ in rows]
    for indices in by_trajectory.values():
        for position, index in enumerate(indices):
            row = rows[index]
            if position + 1 < len(indices):
                next_keys[index] = state_keys[indices[position + 1]]
            else:
                next_keys[index] = _terminal_key(row, mode)

    group_values: dict[tuple[str, Hashable], float] = {}
    for group_uid, indices in by_group.items():
        state_members: dict[Hashable, list[int]] = defaultdict(list)
        for index in indices:
            state_members[state_keys[index]].append(index)
        for key, members in state_members.items():
            group_values[(group_uid, key)] = _aggregate_group_value(
                members, rows, cfg.gamma, mode, trajectory_lengths
            )

    episode_advantages = [0.0] * len(rows)
    node_advantages = [0.0] * len(rows)
    edge_advantages = [0.0] * len(rows)
    for _group_uid, indices in by_group.items():
        _assign_episode_advantages(indices, rows, mode, cfg, episode_advantages)
        _assign_node_advantages(
            indices,
            rows,
            state_keys,
            next_keys,
            group_values,
            mode,
            cfg,
            node_advantages,
        )
        _assign_edge_advantages(
            indices,
            rows,
            state_keys,
            next_keys,
            group_values,
            mode,
            cfg,
            edge_advantages,
        )

    out: list[G2POCreditAssignment] = []
    for index, row in enumerate(rows):
        state_value = group_values[(row.group_uid, state_keys[index])]
        next_value = _value_for_next(row.group_uid, next_keys[index], group_values)
        step_advantage = node_advantages[index] + edge_advantages[index]
        trajectory_length = trajectory_lengths[(row.group_uid, row.trajectory_id)]
        out.append(
            G2POCreditAssignment(
                state_key=state_keys[index],
                next_state_key=next_keys[index],
                group_value=state_value,
                next_group_value=next_value,
                episode_advantage=episode_advantages[index],
                node_advantage=node_advantages[index],
                edge_advantage=edge_advantages[index],
                step_advantage=step_advantage,
                advantage=episode_advantages[index] + cfg.step_advantage_weight * step_advantage,
                actor_token_weight=1.0 / (trajectory_length * row.action_token_count),
                bellman_residual=abs(state_value - (row.reward + cfg.gamma * next_value)),
            )
        )
    return tuple(out)


def transitions_from_v2_records(
    records: Iterable[TrainingTrajectoryRecord],
    *,
    mode: G2POCreditMode | str,
) -> tuple[G2POTransition, ...]:
    """Create credit transitions only through the v2 environment trace gate.

    This is the baseline production ingress. DPPO has a separate projection
    that attaches verifier-owned residual-state evidence.
    """

    selected_mode = G2POCreditMode(mode)
    rows = validate_active_trajectory_records(records, require_v2=True)
    terminal_by_trajectory = {row.traj_uid: row for row in rows if row.terminal}
    converted: list[G2POTransition] = []
    for row in rows:
        terminal = terminal_by_trajectory[row.traj_uid]
        success = bool(terminal.success)
        converted.append(
            G2POTransition(
                group_uid=row.group_uid,
                trajectory_id=row.traj_uid,
                step_index=row.step_idx,
                anchor_observation=row.observation,
                action=row.action_raw,
                next_observation=row.next_observation,
                episode_reward=SUCCESS_REWARD if success else 0.0,
                terminal=row.terminal,
                success=success,
                action_token_count=row.action_token_count,
                valid_action=bool(row.extra.get("is_action_valid", True)),
                active=row.active,
                reward=row.reward,
                decision_role=row.decision_role,
                planner_decision_kind=(
                    str(row.extra["planner_decision_kind"])
                    if row.extra.get("planner_decision_kind") is not None
                    else None
                ),
            )
        )
    return validate_g2po_transitions(converted, mode=selected_mode)


def validate_g2po_transitions(
    transitions: Iterable[G2POTransition], *, mode: G2POCreditMode | str
) -> tuple[G2POTransition, ...]:
    """Reject anything that is not an active, complete environment trajectory.

    This is public because GRPO-U and all graph credit treatments must share
    exactly the same admission boundary before they are compared.
    """

    rows = tuple(transitions)
    G2POCreditMode(mode)
    by_trajectory: dict[tuple[str, str], list[G2POTransition]] = defaultdict(list)
    for row in rows:
        if not row.active:
            raise ValueError("PAD/done rows cannot enter G2PO graph, credit, or loss")
        if row.action_token_count <= 0:
            raise ValueError("every active G2PO transition needs action_token_count > 0")
        if not row.action:
            raise ValueError("every active G2PO transition needs an observable action")
        if row.decision_role not in {"executor", "planner"}:
            raise ValueError("G2PO transition decision_role must be executor or planner")
        if row.decision_role == "planner" and row.planner_decision_kind not in {
            "subgoal",
            "route",
            "rollback",
        }:
            raise ValueError(
                "planner transition requires an observable subgoal, route, or rollback decision"
            )
        by_trajectory[(row.group_uid, row.trajectory_id)].append(row)
    for trajectory_id, trajectory in by_trajectory.items():
        ordered = sorted(trajectory, key=lambda row: row.step_index)
        if [row.step_index for row in ordered] != list(range(len(ordered))):
            raise ValueError(f"trajectory {trajectory_id!r} has non-contiguous or duplicate steps")
        if [index for index, row in enumerate(ordered) if row.terminal] != [len(ordered) - 1]:
            raise ValueError(f"trajectory {trajectory_id!r} must end at one terminal transition")
        if len({row.episode_reward for row in ordered}) != 1:
            raise ValueError(f"trajectory {trajectory_id!r} must carry one episode outcome")
    return rows


def _terminal_key(row: G2POTransition, mode: G2POCreditMode) -> str:
    if mode is G2POCreditMode.SOURCE:
        # This is intentionally the source predicate, not an inferred success
        # flag: the official code tests terminal episode_rewards == 10.0.
        return "__success__" if row.episode_reward == SUCCESS_REWARD else "__failure__"
    return "__success__" if row.success else "__failure__"


def _aggregate_group_value(
    members: list[int],
    rows: tuple[G2POTransition, ...],
    gamma: float,
    mode: G2POCreditMode,
    trajectory_lengths: dict[tuple[str, str], int],
) -> float:
    trajectory_members: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in members:
        row = rows[index]
        trajectory_members[(row.group_uid, row.trajectory_id)].append(index)
    values_by_visit: list[float] = []
    values_by_trajectory: list[float] = []
    for trajectory_id, visit_indices in trajectory_members.items():
        length = trajectory_lengths[trajectory_id]
        values = [
            gamma ** (length - rows[index].step_index) * rows[index].episode_reward
            for index in visit_indices
        ]
        values_by_visit.extend(values)
        values_by_trajectory.append(sum(values) / len(values))
    values = values_by_visit if mode is G2POCreditMode.SOURCE else values_by_trajectory
    return sum(values) / len(values)


def _assign_episode_advantages(
    indices: list[int],
    rows: tuple[G2POTransition, ...],
    mode: G2POCreditMode,
    cfg: G2POCreditConfig,
    output: list[float],
) -> None:
    if mode is G2POCreditMode.SOURCE:
        raw = {index: rows[index].episode_reward for index in indices}
        normalized = _normalize_episode(raw, cfg)
    else:
        normalized = _dispatch_normalize(
            raw={index: rows[index].episode_reward for index in indices},
            rows=rows,
            cfg=cfg,
            mode=mode,
        )
    for index, value in normalized.items():
        output[index] = value


def _assign_node_advantages(
    indices: list[int],
    rows: tuple[G2POTransition, ...],
    state_keys: list[Hashable],
    next_keys: list[Hashable],
    group_values: dict[tuple[str, Hashable], float],
    mode: G2POCreditMode,
    cfg: G2POCreditConfig,
    output: list[float],
) -> None:
    by_state: dict[Hashable, list[int]] = defaultdict(list)
    for index in indices:
        by_state[state_keys[index]].append(index)
    for members in by_state.values():
        raw = {
            index: _value_for_next(rows[index].group_uid, next_keys[index], group_values)
            - (0.0 if rows[index].valid_action else cfg.invalid_action_penalty)
            for index in members
        }
        if len(members) < 2:
            normalized = {index: 0.0 for index in members}
        else:
            normalized = _dispatch_normalize(raw=raw, rows=rows, cfg=cfg, mode=mode)
        for index, value in normalized.items():
            output[index] = value


def _assign_edge_advantages(
    indices: list[int],
    rows: tuple[G2POTransition, ...],
    state_keys: list[Hashable],
    next_keys: list[Hashable],
    group_values: dict[tuple[str, Hashable], float],
    mode: G2POCreditMode,
    cfg: G2POCreditConfig,
    output: list[float],
) -> None:
    raw = {
        index: _value_for_next(rows[index].group_uid, next_keys[index], group_values)
        - group_values[(rows[index].group_uid, state_keys[index])]
        for index in indices
    }
    normalized = _dispatch_normalize(raw=raw, rows=rows, cfg=cfg, mode=mode)
    for index, value in normalized.items():
        output[index] = value


def _value_for_next(
    group_uid: str, key: Hashable, group_values: dict[tuple[str, Hashable], float]
) -> float:
    if key == "__success__":
        return SUCCESS_REWARD
    if key == "__failure__":
        return 0.0
    return group_values[(group_uid, key)]


def _normalize(raw: dict[int, float], cfg: G2POCreditConfig) -> dict[int, float]:
    values = list(raw.values())
    if len(values) <= 1:
        return {index: 0.0 for index in raw}
    mean = sum(values) / len(values)
    if cfg.normalization == "mean_norm":
        return {index: value - mean for index, value in raw.items()}
    scale = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) + cfg.epsilon
    )
    return {index: (value - mean) / scale for index, value in raw.items()}


def _normalize_episode(raw: dict[int, float], cfg: G2POCreditConfig) -> dict[int, float]:
    # Source episode_norm_reward leaves a singleton as an absolute outcome.
    if len(raw) <= 1:
        return dict(raw)
    return _normalize(raw, cfg)


def _trajectory_normalized(
    *, raw: dict[int, float], rows: tuple[G2POTransition, ...], cfg: G2POCreditConfig
) -> dict[int, float]:
    by_trajectory: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in raw:
        row = rows[index]
        by_trajectory[(row.group_uid, row.trajectory_id)].append(index)
    trajectory_means = {
        trajectory_id: sum(raw[index] for index in indices) / len(indices)
        for trajectory_id, indices in by_trajectory.items()
    }
    if len(trajectory_means) == 1:
        # Mirror the source episode singleton treatment while avoiding a fake
        # within-trajectory repeated-visit comparison.
        value = next(iter(trajectory_means.values()))
        return {index: value if len(raw) == 1 else 0.0 for index in raw}
    normalized_means = _normalize(
        {position: value for position, value in enumerate(trajectory_means.values())}, cfg
    )
    by_trajectory_value = {
        trajectory_id: normalized_means[position]
        for position, trajectory_id in enumerate(trajectory_means)
    }
    return {
        index: by_trajectory_value[(rows[index].group_uid, rows[index].trajectory_id)]
        for index in raw
    }


def _dispatch_normalize(
    *,
    raw: dict[int, float],
    rows: tuple[G2POTransition, ...],
    cfg: G2POCreditConfig,
    mode: G2POCreditMode,
) -> dict[int, float]:
    if mode is G2POCreditMode.SOURCE:
        return _normalize(raw, cfg)
    if mode is G2POCreditMode.TRAJECTORY_NORMALIZED_EDGE_PRESERVING:
        return _trajectory_normalized_edge_preserving(raw=raw, rows=rows, cfg=cfg)
    return _trajectory_normalized(raw=raw, rows=rows, cfg=cfg)


def _trajectory_normalized_edge_preserving(
    *, raw: dict[int, float], rows: tuple[G2POTransition, ...], cfg: G2POCreditConfig
) -> dict[int, float]:
    """TN-v2: equalize trajectory weight while preserving per-step edge values.

    Each step gets weight w_it = 1/T_i so every trajectory contributes equal
    total statistical mass.  Normalization uses the trajectory-balanced weighted
    mean and std, then each step keeps its own normalized value.
    """

    by_trajectory: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in raw:
        row = rows[index]
        by_trajectory[(row.group_uid, row.trajectory_id)].append(index)
    if len(by_trajectory) <= 1:
        value = next(iter(raw.values()), 0.0)
        return {index: value if len(raw) == 1 else 0.0 for index in raw}
    weights: dict[int, float] = {}
    for indices in by_trajectory.values():
        trajectory_weight = 1.0 / len(indices)
        for index in indices:
            weights[index] = trajectory_weight
    total_weight = sum(weights[index] for index in raw)
    weighted_mean = sum(weights[index] * raw[index] for index in raw) / total_weight
    weighted_var = (
        sum(weights[index] * (raw[index] - weighted_mean) ** 2 for index in raw) / total_weight
    )
    scale = math.sqrt(weighted_var) + cfg.epsilon
    return {index: (raw[index] - weighted_mean) / scale for index in raw}


def _hashable(value: Any) -> Hashable:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _hashable(tolist())
    raise TypeError(f"unsupported G2PO anchor observation type: {type(value)!r}")
