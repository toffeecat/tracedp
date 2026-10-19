"""One admission and actor-weight contract for the ALFWorld credit treatments.

The comparison is intentionally not an "official G2PO reproduction": G2PO-U
imports only the pinned *credit* semantics, while every treatment is trained
with the same trajectory-step-token actor mass.  This module makes the GRPO-U
control explicit so the control cannot accidentally retain a length-weighted
loss while the graph treatments do not.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.training.agentic_rl_runtime import AgenticRLContractError
from lh_platform.training.dppo.credit import DPPOCreditMode, assign_dppo_credit
from lh_platform.training.g2po.credit import (
    G2POCreditConfig,
    G2POCreditMode,
    G2POTransition,
    assign_g2po_credit,
    validate_g2po_transitions,
)


class UnifiedCreditTreatment(StrEnum):
    """The only treatments allowed in the registered ALFWorld comparison."""

    GRPO_U = "grpo_u"
    G2PO_U = "g2po_u"
    G2PO_TN = "g2po_tn"
    VERIFIED_STATE_DP = "verified_state_dp"
    ORACLE_PHASE_REFINED_G2PO_TN = "oracle_phase_refined_g2po_tn"


@dataclass(frozen=True)
class UnifiedCreditAssignment:
    treatment: UnifiedCreditTreatment
    advantage: float
    episode_advantage: float
    actor_token_weight: float
    state_key: Hashable | None = None
    next_state_key: Hashable | None = None
    group_value: float | None = None
    next_group_value: float | None = None
    node_advantage: float | None = None
    edge_advantage: float | None = None
    step_advantage: float | None = None
    bellman_residual: float | None = None


def assign_unified_credit(
    transitions: Iterable[G2POTransition],
    *,
    treatment: UnifiedCreditTreatment | str,
    g2po_config: G2POCreditConfig | None = None,
) -> tuple[UnifiedCreditAssignment, ...]:
    """Assign credit without changing which transition/token mass is trainable."""

    rows = tuple(transitions)
    if not rows:
        return ()
    selected = UnifiedCreditTreatment(treatment)
    if selected is UnifiedCreditTreatment.GRPO_U:
        # GRPO does not use state identities, but it must obey the same active
        # transition/terminal admission boundary as every graph treatment.
        validate_g2po_transitions(rows, mode=G2POCreditMode.SOURCE)
        return _assign_grpo_u(rows)

    if selected is UnifiedCreditTreatment.G2PO_U:
        assignments = assign_g2po_credit(rows, mode=G2POCreditMode.SOURCE, config=g2po_config)
    elif selected is UnifiedCreditTreatment.G2PO_TN:
        assignments = assign_g2po_credit(
            rows, mode=G2POCreditMode.TRAJECTORY_NORMALIZED, config=g2po_config
        )
    else:
        dppo_mode = {
            UnifiedCreditTreatment.VERIFIED_STATE_DP: DPPOCreditMode.VERIFIED_STATE,
            UnifiedCreditTreatment.ORACLE_PHASE_REFINED_G2PO_TN: (
                DPPOCreditMode.ORACLE_PHASE_REFINED_TN
            ),
        }[selected]
        assignments = assign_dppo_credit(rows, mode=dppo_mode, config=g2po_config)
    return tuple(
        UnifiedCreditAssignment(
            treatment=selected,
            advantage=item.advantage,
            episode_advantage=item.episode_advantage,
            actor_token_weight=item.actor_token_weight,
            state_key=item.state_key,
            next_state_key=item.next_state_key,
            group_value=item.group_value,
            next_group_value=item.next_group_value,
            node_advantage=item.node_advantage,
            edge_advantage=item.edge_advantage,
            step_advantage=item.step_advantage,
            bellman_residual=item.bellman_residual,
        )
        for item in assignments
    )


def _assign_grpo_u(rows: tuple[G2POTransition, ...]) -> tuple[UnifiedCreditAssignment, ...]:
    trajectory_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_group: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row.group_uid, row.trajectory_id)
        if key not in trajectory_rows:
            by_group[row.group_uid].append(key)
        trajectory_rows[key].append(index)

    trajectory_advantage: dict[tuple[str, str], float] = {}
    for _group_uid, trajectories in by_group.items():
        outcomes: list[float] = []
        for key in trajectories:
            rewards = {rows[index].episode_reward for index in trajectory_rows[key]}
            if len(rewards) != 1:  # defensive: admission already checks this.
                raise ValueError(f"GRPO-U trajectory {key!r} has inconsistent outcome rewards")
            outcomes.append(rewards.pop())
        for key, advantage in zip(trajectories, _group_relative_advantages(outcomes), strict=False):
            trajectory_advantage[key] = advantage

    return tuple(
        UnifiedCreditAssignment(
            treatment=UnifiedCreditTreatment.GRPO_U,
            advantage=trajectory_advantage[(row.group_uid, row.trajectory_id)],
            episode_advantage=trajectory_advantage[(row.group_uid, row.trajectory_id)],
            actor_token_weight=1.0
            / (len(trajectory_rows[(row.group_uid, row.trajectory_id)]) * row.action_token_count),
        )
        for row in rows
    )


def _group_relative_advantages(outcomes: list[float]) -> tuple[float, ...]:
    """Return GRPO advantages, or zeros for groups with no learning signal."""

    if len(outcomes) < 2:
        return tuple(0.0 for _ in outcomes)
    if any(not math.isfinite(value) for value in outcomes):
        raise AgenticRLContractError("GRPO-U outcomes must be finite")
    mean = sum(outcomes) / len(outcomes)
    scale = math.sqrt(sum((value - mean) ** 2 for value in outcomes) / (len(outcomes) - 1))
    if scale == 0.0:
        return tuple(0.0 for _ in outcomes)
    return tuple((value - mean) / scale for value in outcomes)
