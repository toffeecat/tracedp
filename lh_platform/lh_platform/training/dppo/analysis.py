"""Mechanism-first analysis for persisted DPPO/G2PO rollout artifacts."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, median
from typing import Any

from lh_platform.training.agentic_rl_runtime import (
    AgenticRolloutBatch,
    canonical_sha256,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.dppo.credit import (
    DPPOCreditMode,
)
from lh_platform.training.dppo.credit import (
    transitions_from_v2_records as dppo_transitions_from_v2_records,
)
from lh_platform.training.g2po.credit import (
    G2POCreditConfig,
    G2POCreditMode,
)
from lh_platform.training.g2po.credit import (
    transitions_from_v2_records as g2po_transitions_from_v2_records,
)
from lh_platform.training.subproblem_study import (
    assess_oracle_subproblem_trace,
    count_frontier_live_precondition_violations,
    has_frontier_audit_receipt,
)
from lh_platform.training.trajectory_v2 import validate_active_trajectory_records
from lh_platform.training.unified_credit import (
    UnifiedCreditAssignment,
    UnifiedCreditTreatment,
    assign_unified_credit,
)

ANALYSIS_SCHEMA = "lh_dppo_rollout_analysis/v1"
PAIRING_SCHEMA = "lh_dppo_rollout_pairing/v1"


class DPPORolloutAnalysisError(ValueError):
    """Raised when a persisted rollout cannot support the requested analysis."""


def build_frozen_credit_pairing(
    rollout: AgenticRolloutBatch,
    *,
    config: G2POCreditConfig | None = None,
    epsilon: float = 1e-12,
    require_physical_evidence: bool = True,
    expected_recipe: Any | None = None,
    expected_task_metadata: Mapping[str, Any] | None = None,
    expected_task_manifest_sha256: str | None = None,
    expected_gamefile: str | None = None,
    preregistration_bindings: tuple[Mapping[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Compute both credit arms from one validated model-policy rollout.

    This is deliberately a pre-update gate.  It proves that G2PO-TN and DPPO
    saw the same actions, outcomes, token spans, and old-policy log-probabilities.
    If state refinement does not change any target, another model load or PPO
    update cannot answer the credit question and is therefore blocked.
    """

    from lh_platform.training.agentic_rl_components import (
        AlfWorldFactDecomposition,
        UnifiedG2POMethod,
    )

    batch = rollout.validate()
    if require_physical_evidence and (
        expected_recipe is None
        or expected_task_metadata is None
        or expected_task_manifest_sha256 is None
        or expected_gamefile is None
        or not preregistration_bindings
    ):
        raise DPPORolloutAnalysisError(
            "formal frozen credit audit requires recipe and official task bindings"
        )
    if preregistration_bindings is not None and any(
        binding.get("matches_head") is not True for binding in preregistration_bindings
    ):
        raise DPPORolloutAnalysisError(
            "formal frozen credit audit requires an unchanged committed config tree"
        )
    if (
        expected_task_manifest_sha256 is not None
        and batch.task_manifest_sha256 != expected_task_manifest_sha256
    ):
        raise DPPORolloutAnalysisError("rollout task manifest differs from official task artifacts")
    rows = tuple(batch.records)
    if any(row.method_source != "on_policy_controlled_state" for row in rows):
        raise DPPORolloutAnalysisError(
            "frozen credit pairing requires model-policy controlled-state rollouts"
        )
    awareness_modes = {str(row.extra.get("actor_awareness_mode", "")) for row in rows}
    if len(awareness_modes) != 1 or next(iter(awareness_modes)) not in {
        "none",
        "oracle_goal_graph",
    }:
        raise DPPORolloutAnalysisError(
            "one frozen credit pairing must contain exactly one actor awareness mode"
        )
    if any(row.extra.get("model_response") != row.action_raw for row in rows):
        raise DPPORolloutAnalysisError(
            "controlled rollout action is not bound to the recorded model response"
        )

    annotated = AlfWorldFactDecomposition().annotate(rows)
    _validate_frozen_controlled_collision(
        rows,
        annotated,
        require_physical_evidence=require_physical_evidence,
        expected_recipe=expected_recipe,
        expected_task_metadata=expected_task_metadata,
        expected_gamefile=expected_gamefile,
    )
    cfg = config or G2POCreditConfig()
    control = UnifiedG2POMethod(
        UnifiedCreditTreatment.G2PO_TN,
        config=cfg,
    ).prepare_update(annotated)
    treatment = UnifiedG2POMethod(
        UnifiedCreditTreatment.ORACLE_PHASE_REFINED_G2PO_TN,
        config=cfg,
    ).prepare_update(annotated)
    control_targets = {(target.traj_uid, target.step_idx): target for target in control.targets}
    treatment_targets = {(target.traj_uid, target.step_idx): target for target in treatment.targets}
    if set(control_targets) != set(treatment_targets):
        raise DPPORolloutAnalysisError("frozen credit arms produced different policy target keys")
    if any(
        not math.isclose(
            control_targets[key].actor_token_weight,
            treatment_targets[key].actor_token_weight,
            rel_tol=0.0,
            abs_tol=epsilon,
        )
        for key in control_targets
    ):
        raise DPPORolloutAnalysisError("credit arms changed actor token mass")
    differences = [
        {
            "traj_uid": key[0],
            "step_idx": key[1],
            "control_advantage": control_targets[key].advantage,
            "treatment_advantage": treatment_targets[key].advantage,
            "absolute_difference": abs(
                control_targets[key].advantage - treatment_targets[key].advantage
            ),
        }
        for key in sorted(control_targets)
        if not math.isclose(
            control_targets[key].advantage,
            treatment_targets[key].advantage,
            rel_tol=0.0,
            abs_tol=epsilon,
        )
    ]
    frozen_rollout_sha256 = canonical_sha256(
        {
            "records": [row.to_dict() for row in rows],
            "policy_samples": [asdict(sample) for sample in batch.policy_samples],
            "task_manifest_sha256": batch.task_manifest_sha256,
            "policy_checkpoint_sha256": batch.policy_checkpoint_sha256,
        }
    )
    credit_blocked = [] if differences else ["no_credit_target_difference"]
    policy_update_blocked = [
        *credit_blocked,
        "teacher_forced_old_logprob_revalidation_pending",
    ]
    return {
        "schema": "lh_frozen_rollout_credit_pairing/v1",
        "frozen_rollout_sha256": frozen_rollout_sha256,
        "same_frozen_rollout": True,
        "actor_awareness_mode": next(iter(awareness_modes)),
        "policy_checkpoint_sha256": batch.policy_checkpoint_sha256,
        "task_manifest_sha256": batch.task_manifest_sha256,
        "official_task_binding_verified": bool(
            expected_recipe is not None
            and expected_task_metadata is not None
            and expected_task_manifest_sha256 is not None
            and expected_gamefile is not None
        ),
        "preregistration_bindings": [dict(binding) for binding in (preregistration_bindings or ())],
        "transition_count": len(rows),
        "trajectory_count": len({row.traj_uid for row in rows}),
        "controlled_state_ids": sorted(
            {
                str(row.extra["controlled_state_id"])
                for row in rows
                if row.extra.get("controlled_state_id") is not None
            }
        ),
        "control_credit_treatment": UnifiedCreditTreatment.G2PO_TN.value,
        "treatment_credit_treatment": (UnifiedCreditTreatment.ORACLE_PHASE_REFINED_G2PO_TN.value),
        "shared_objective": control.objective,
        "shared_actor_token_mass": True,
        "credit_target_difference_count": len(differences),
        "max_absolute_credit_target_difference": max(
            (row["absolute_difference"] for row in differences),
            default=0.0,
        ),
        "credit_target_differences": differences,
        "credit_difference_gate_passed": not credit_blocked,
        "credit_gate_blocked_by": credit_blocked,
        "paired_policy_update_eligible": False,
        "policy_update_blocked_by": policy_update_blocked,
        "claim_scope": ("frozen_rollout_credit_difference_not_policy_improvement"),
    }


def _validate_frozen_controlled_collision(
    rows: tuple[TrainingTrajectoryRecord, ...],
    annotated: tuple[TrainingTrajectoryRecord, ...],
    *,
    require_physical_evidence: bool,
    expected_recipe: Any | None,
    expected_task_metadata: Mapping[str, Any] | None,
    expected_gamefile: str | None,
) -> None:
    from lh_platform.training.controlled_aliasing import (
        validate_alfworld_replay_evidence,
    )

    if any(
        row.extra.get("controlled_collision_verified") is not True
        or row.extra.get("continuation_controller") != "model_policy"
        or row.extra.get("deterministic_continuation_program_used") is not False
        for row in rows
    ):
        raise DPPORolloutAnalysisError(
            "frozen credit pairing is not a model-policy controlled collision"
        )
    first = tuple(row for row in annotated if row.step_idx == 0)
    trajectories = {row.traj_uid for row in rows}
    if len(first) != len(trajectories):
        raise DPPORolloutAnalysisError(
            "controlled collision requires one step-zero record per trajectory"
        )
    state_ids = {str(row.extra.get("controlled_state_id", "")) for row in first}
    if len(state_ids) != 2 or "" in state_ids:
        raise DPPORolloutAnalysisError(
            "controlled collision requires exactly two registered states"
        )
    counts = Counter(str(row.extra["controlled_state_id"]) for row in first)
    if len(set(counts.values())) != 1:
        raise DPPORolloutAnalysisError("controlled collision states have unequal trajectory counts")
    fixed_fields = {
        "group_uid": {row.group_uid for row in first},
        "official_anchor": {row.observation for row in first},
        "raw_observation": {str(row.extra.get("alfworld_raw_observation", "")) for row in first},
        "action_affordance_hash": {row.action_affordance_hash for row in first},
        "remaining_budget": {row.remaining_budget for row in first},
        "goal_hash": {row.goal_hash for row in first},
        "actor_protocol_id": {str(row.extra.get("actor_protocol_id", "")) for row in first},
        "controlled_recipe_sha256": {
            str(row.extra.get("controlled_recipe_sha256", "")) for row in first
        },
    }
    drift = [name for name, values in fixed_fields.items() if len(values) != 1]
    if drift:
        raise DPPORolloutAnalysisError(
            "controlled collision fixed-factor drift: " + ", ".join(drift)
        )
    from lh_platform.training.g2po.actor_protocol import (
        g2po_anchor_tracker_source_sha256,
    )

    if any(
        row.extra.get("anchor_tracker_implementation_sha256") != g2po_anchor_tracker_source_sha256()
        for row in first
    ):
        raise DPPORolloutAnalysisError(
            "controlled collision is not bound to the local G2PO anchor projection"
        )
    if expected_recipe is not None and expected_task_metadata is not None:
        _validate_registered_recipe_and_task(
            rows,
            annotated,
            expected_recipe=expected_recipe,
            expected_task_metadata=expected_task_metadata,
        )
    frontier_by_state: dict[str, set[str]] = defaultdict(set)
    snapshot_by_state: dict[str, set[str]] = defaultdict(set)
    for row in first:
        state_id = str(row.extra["controlled_state_id"])
        frontier_by_state[state_id].add(str(row.extra["frontier_signature"]))
        snapshot_by_state[state_id].add(row.state_snapshot_sha256)
    if (
        any(len(values) != 1 for values in frontier_by_state.values())
        or len({next(iter(values)) for values in frontier_by_state.values()}) != 2
    ):
        raise DPPORolloutAnalysisError(
            "controlled collision does not expose two stable distinct frontiers"
        )
    if (
        any(len(values) != 1 for values in snapshot_by_state.values())
        or len({next(iter(values)) for values in snapshot_by_state.values()}) != 2
    ):
        raise DPPORolloutAnalysisError(
            "controlled collision does not expose two stable distinct snapshots"
        )
    if not require_physical_evidence:
        return
    for row in rows:
        replay_outcome = row.extra.get("verified_replay_outcome")
        if not isinstance(replay_outcome, Mapping):
            raise DPPORolloutAnalysisError(
                "controlled rollout lacks a persisted verified replay outcome"
            )
        errors = validate_alfworld_replay_evidence(
            snapshot_sha256=row.state_snapshot_sha256,
            snapshot_path=Path(str(row.extra.get("state_snapshot_artifact_path", ""))).resolve(),
            receipt_sha256=row.extra.get("restore_probe_receipt_sha256"),
            receipt_path=Path(str(row.extra.get("restore_probe_receipt_path", ""))).resolve(),
            action=row.extra.get("environment_action"),
            expected_replay_outcome=replay_outcome,
        )
        if errors:
            raise DPPORolloutAnalysisError(
                "controlled rollout physical evidence failed: " + ", ".join(errors)
            )
        _validate_record_against_replay_outcome(row, replay_outcome)
    _validate_trajectory_state_chain(
        rows,
        expected_recipe=expected_recipe,
        expected_gamefile=expected_gamefile,
    )


def _validate_registered_recipe_and_task(
    rows: tuple[TrainingTrajectoryRecord, ...],
    annotated: tuple[TrainingTrajectoryRecord, ...],
    *,
    expected_recipe: Any,
    expected_task_metadata: Mapping[str, Any],
) -> None:
    if any(
        row.task_id != expected_recipe.task_id
        or row.extra.get("controlled_recipe_sha256") != expected_recipe.sha256
        or row.extra.get("alfworld_task_metadata") != dict(expected_task_metadata)
        or row.goal_hash != canonical_sha256(dict(expected_task_metadata))
        for row in rows
    ):
        raise DPPORolloutAnalysisError(
            "controlled rollout differs from registered recipe/task metadata"
        )
    branches = {branch.branch_id: branch for branch in expected_recipe.branches}
    first = [row for row in annotated if row.step_idx == 0]
    for row in first:
        state_id = str(row.extra.get("controlled_state_id", ""))
        branch = branches.get(state_id)
        if branch is None:
            raise DPPORolloutAnalysisError(
                "controlled rollout state is absent from the registered recipe"
            )
        start = row.extra.get("controlled_start_state_evidence")
        prefix_steps = start.get("prefix_steps") if isinstance(start, Mapping) else None
        if not isinstance(prefix_steps, list):
            raise DPPORolloutAnalysisError(
                "controlled rollout registered prefix evidence is missing"
            )
        observed_actions = tuple(
            str(step.get("action", "")) for step in prefix_steps if isinstance(step, Mapping)
        )
        expected_actions = (*expected_recipe.setup_actions, *branch.actions)
        if observed_actions != expected_actions or len(prefix_steps) != len(expected_actions):
            raise DPPORolloutAnalysisError(
                "controlled rollout prefix differs from the registered recipe"
            )
        frontier = tuple(row.extra.get("active_frontier_ids", ()))
        if frontier != branch.expected_frontier_ids:
            raise DPPORolloutAnalysisError(
                "controlled rollout frontier differs from the registered recipe"
            )


def _validate_trajectory_state_chain(
    rows: tuple[TrainingTrajectoryRecord, ...],
    *,
    expected_recipe: Any,
    expected_gamefile: str,
) -> None:
    """Rebuild every before-state and graph anchor from verified outcomes."""

    from lh_platform.training.controlled_aliasing import (
        validate_alfworld_replay_evidence,
    )
    from lh_platform.training.g2po.actor_protocol import G2POAnchorTracker

    by_trajectory: dict[str, list[TrainingTrajectoryRecord]] = defaultdict(list)
    for row in rows:
        by_trajectory[row.traj_uid].append(row)
    for trajectory in by_trajectory.values():
        ordered = sorted(trajectory, key=lambda item: item.step_idx)
        if [row.step_idx for row in ordered] != list(range(len(ordered))):
            raise DPPORolloutAnalysisError("controlled rollout trajectory steps are not contiguous")
        start = ordered[0].extra.get("controlled_start_state_evidence")
        if not isinstance(start, Mapping):
            raise DPPORolloutAnalysisError(
                "controlled rollout step zero lacks prefix state evidence"
            )
        initial_prompt = start.get("initial_actor_prompt")
        prefix_steps = start.get("prefix_steps")
        if (
            not isinstance(initial_prompt, str)
            or not isinstance(prefix_steps, list)
            or not prefix_steps
        ):
            raise DPPORolloutAnalysisError("controlled rollout prefix state evidence is malformed")
        tracker = G2POAnchorTracker(initial_actor_prompt=initial_prompt)
        state_id = str(ordered[0].extra.get("controlled_state_id", ""))
        branch = next(
            (
                candidate
                for candidate in expected_recipe.branches
                if candidate.branch_id == state_id
            ),
            None,
        )
        if branch is None:
            raise DPPORolloutAnalysisError(
                "controlled rollout state is absent from registered recipe"
            )
        registered_prefix = (*expected_recipe.setup_actions, *branch.actions)
        action_history: list[str] = []
        current_outcome: Mapping[str, Any] | None = None
        current_anchor = ""
        saw_visible_prefix_outcome = False
        for prefix_index, prefix_step in enumerate(prefix_steps):
            if not isinstance(prefix_step, Mapping):
                raise DPPORolloutAnalysisError(
                    "controlled rollout prefix evidence row is malformed"
                )
            outcome = prefix_step.get("verified_replay_outcome")
            if not isinstance(outcome, Mapping):
                raise DPPORolloutAnalysisError("controlled rollout prefix lacks replay outcome")
            if prefix_step.get("verified_replay_outcome_sha256") != canonical_sha256(outcome):
                raise DPPORolloutAnalysisError("controlled rollout prefix replay digest differs")
            errors = validate_alfworld_replay_evidence(
                snapshot_sha256=prefix_step.get("state_snapshot_sha256"),
                snapshot_path=Path(
                    str(prefix_step.get("state_snapshot_artifact_path", ""))
                ).resolve(),
                receipt_sha256=prefix_step.get("restore_probe_receipt_sha256"),
                receipt_path=Path(str(prefix_step.get("restore_probe_receipt_path", ""))).resolve(),
                action=prefix_step.get("action"),
                expected_replay_outcome=outcome,
                expected_gamefile=expected_gamefile,
                expected_action_history=tuple(action_history),
            )
            if errors:
                raise DPPORolloutAnalysisError(
                    "controlled rollout prefix physical evidence failed: " + ", ".join(errors)
                )
            raw_observation = outcome.get("observation")
            actions = outcome.get("admissible_actions")
            if not isinstance(raw_observation, str) or not isinstance(actions, (list, tuple)):
                raise DPPORolloutAnalysisError(
                    "controlled rollout prefix outcome lacks actor fields"
                )
            saw_visible_prefix_outcome = saw_visible_prefix_outcome or not (
                raw_observation.startswith("Nothing happens")
            )
            current_anchor = tracker.project_next(
                raw_observation=raw_observation,
                admissible_actions=tuple(str(action) for action in actions),
            )
            current_outcome = outcome
            if prefix_step.get("action") != registered_prefix[prefix_index]:
                raise DPPORolloutAnalysisError(
                    "controlled rollout prefix differs from registered recipe"
                )
            action_history.append(str(prefix_step["action"]))
        if not saw_visible_prefix_outcome or current_outcome is None:
            raise DPPORolloutAnalysisError(
                "controlled rollout anchor still depends on unverified reset prompt"
            )

        for row in ordered:
            _validate_before_state_against_replay_outcome(row, current_outcome)
            if row.observation != current_anchor:
                raise DPPORolloutAnalysisError(
                    "controlled rollout current anchor differs from replay chain"
                )
            outcome = row.extra.get("verified_replay_outcome")
            if not isinstance(outcome, Mapping):
                raise DPPORolloutAnalysisError(
                    "controlled rollout state chain lacks replay outcome"
                )
            errors = validate_alfworld_replay_evidence(
                snapshot_sha256=row.state_snapshot_sha256,
                snapshot_path=Path(
                    str(row.extra.get("state_snapshot_artifact_path", ""))
                ).resolve(),
                receipt_sha256=row.extra.get("restore_probe_receipt_sha256"),
                receipt_path=Path(str(row.extra.get("restore_probe_receipt_path", ""))).resolve(),
                action=row.extra.get("environment_action"),
                expected_replay_outcome=outcome,
                expected_gamefile=expected_gamefile,
                expected_action_history=tuple(action_history),
            )
            if errors:
                raise DPPORolloutAnalysisError(
                    "controlled rollout action-history evidence failed: " + ", ".join(errors)
                )
            actions = outcome.get("admissible_actions")
            if not isinstance(outcome.get("observation"), str) or not isinstance(
                actions, (list, tuple)
            ):
                raise DPPORolloutAnalysisError(
                    "controlled rollout replay outcome lacks actor fields"
                )
            current_anchor = tracker.project_next(
                raw_observation=str(outcome["observation"]),
                admissible_actions=tuple(str(action) for action in actions),
            )
            if row.next_observation != current_anchor:
                raise DPPORolloutAnalysisError(
                    "controlled rollout next anchor differs from replay chain"
                )
            current_outcome = outcome
            action_history.append(str(row.extra.get("environment_action", "")))


def _validate_before_state_against_replay_outcome(
    row: TrainingTrajectoryRecord,
    replay_outcome: Mapping[str, Any],
) -> None:
    if str(row.extra.get("alfworld_raw_observation", "")) != str(
        replay_outcome.get("observation", "")
    ):
        raise DPPORolloutAnalysisError(
            "controlled rollout current observation differs from replay chain"
        )
    try:
        replay_facts = _canonical_fact_pairs(replay_outcome.get("facts"))
        row_facts = _canonical_fact_pairs(row.extra.get("alfworld_facts"))
    except (TypeError, ValueError) as error:
        raise DPPORolloutAnalysisError(
            "controlled rollout current fact evidence is malformed"
        ) from error
    if row_facts != replay_facts:
        raise DPPORolloutAnalysisError("controlled rollout current facts differ from replay chain")
    if tuple(row.extra.get("alfworld_admissible_actions", ())) != tuple(
        replay_outcome.get("admissible_actions", ())
    ):
        raise DPPORolloutAnalysisError(
            "controlled rollout current affordances differ from replay chain"
        )
    metadata = row.extra.get("alfworld_task_metadata")
    if not isinstance(metadata, Mapping) or row.goal_hash != canonical_sha256(metadata):
        raise DPPORolloutAnalysisError("controlled rollout task metadata differs from goal hash")


def _validate_record_against_replay_outcome(
    row: TrainingTrajectoryRecord,
    replay_outcome: Mapping[str, Any],
) -> None:
    """Bind all credit-bearing row fields to the restore-verified outcome."""

    from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
    from lh_platform.training.subproblems.base import goal_state_projection

    declared_sha256 = row.extra.get("verified_replay_outcome_sha256")
    if declared_sha256 != canonical_sha256(replay_outcome):
        raise DPPORolloutAnalysisError(
            "controlled rollout replay outcome digest differs from the record"
        )
    required = {"observation", "reward", "done", "won", "facts", "admissible_actions"}
    if not required.issubset(replay_outcome):
        raise DPPORolloutAnalysisError("controlled rollout replay outcome lacks verifier fields")
    replay_won = replay_outcome["won"]
    replay_done = replay_outcome["done"]
    if not isinstance(replay_won, bool) or not isinstance(replay_done, bool):
        raise DPPORolloutAnalysisError("controlled rollout replay terminal fields are not booleans")
    if row.extra.get("training_reward_projection") != "official_won_binary_0_10":
        raise DPPORolloutAnalysisError(
            "controlled rollout training reward projection is not registered"
        )
    expected_training_reward = 10.0 if replay_won else 0.0
    if not math.isclose(
        row.reward,
        expected_training_reward,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DPPORolloutAnalysisError(
            "controlled rollout reward differs from verified official won"
        )
    outcome = row.outcome_evidence
    if outcome.get("won") is not replay_won or outcome.get("environment_done") is not replay_done:
        raise DPPORolloutAnalysisError(
            "controlled rollout outcome flags differ from verified replay"
        )
    budget_terminal = outcome.get("budget_terminal")
    if not isinstance(budget_terminal, bool) or row.terminal is not bool(
        replay_done or budget_terminal
    ):
        raise DPPORolloutAnalysisError(
            "controlled rollout terminal flag differs from environment/budget evidence"
        )
    if row.terminal and row.success is not replay_won:
        raise DPPORolloutAnalysisError(
            "controlled rollout terminal success differs from verified replay"
        )
    if str(row.extra.get("alfworld_raw_next_observation", "")) != str(
        replay_outcome["observation"]
    ):
        raise DPPORolloutAnalysisError(
            "controlled rollout next observation differs from verified replay"
        )
    try:
        replay_facts = _canonical_fact_pairs(replay_outcome["facts"])
        row_facts = _canonical_fact_pairs(row.extra.get("alfworld_next_facts"))
    except (TypeError, ValueError) as error:
        raise DPPORolloutAnalysisError("controlled rollout fact evidence is malformed") from error
    if row_facts != replay_facts:
        raise DPPORolloutAnalysisError("controlled rollout next facts differ from verified replay")
    if tuple(row.extra.get("alfworld_next_admissible_actions", ())) != tuple(
        replay_outcome["admissible_actions"]
    ):
        raise DPPORolloutAnalysisError(
            "controlled rollout next affordances differ from verified replay"
        )

    metadata = row.extra.get("alfworld_task_metadata")
    if not isinstance(metadata, Mapping):
        raise DPPORolloutAnalysisError("controlled rollout task metadata is missing")
    compiler = AlfWorldSubproblemCompiler()
    graph = compiler.compile_task(metadata)
    vector = compiler.project_state(metadata, row.extra["alfworld_next_facts"])
    verifier_complete = not goal_state_projection(graph, vector)["residual_goal_ids"]
    if verifier_complete is not replay_won:
        raise DPPORolloutAnalysisError("official won disagrees with the independent goal compiler")


def _canonical_fact_pairs(value: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("facts must be a sequence")
    pairs: list[tuple[str, tuple[str, ...]]] = []
    for fact in value:
        if isinstance(fact, Mapping):
            name = fact.get("name")
            args = fact.get("args")
        elif isinstance(fact, (list, tuple)) and len(fact) == 2:
            name, args = fact
        else:
            raise TypeError("fact must be a mapping or pair")
        if not isinstance(name, str) or not isinstance(args, (list, tuple)):
            raise TypeError("fact name/args are malformed")
        pairs.append((name, tuple(str(argument) for argument in args)))
    return tuple(sorted(pairs))


def load_rollout_artifact(path: str | Path) -> tuple[TrainingTrajectoryRecord, ...]:
    source = Path(path).resolve()
    records: list[TrainingTrajectoryRecord] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DPPORolloutAnalysisError(f"cannot read rollout artifact: {error}") from error
    if not lines:
        raise DPPORolloutAnalysisError("rollout artifact is empty")
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DPPORolloutAnalysisError(
                f"rollout line {line_number} is not JSON: {error}"
            ) from error
        if not isinstance(row, Mapping) or row.get("schema") != (
            "lh_agentic_rl_rollout_transition/v1"
        ):
            raise DPPORolloutAnalysisError(f"rollout line {line_number} has an unsupported schema")
        record = row.get("record")
        if not isinstance(record, dict):
            raise DPPORolloutAnalysisError(f"rollout line {line_number} does not contain a record")
        parsed = TrainingTrajectoryRecord.from_dict(record)
        policy_sample = row.get("policy_sample")
        if isinstance(policy_sample, Mapping):
            parsed = replace(
                parsed,
                extra={
                    **parsed.extra,
                    "policy_sample_sha256": canonical_sha256(policy_sample),
                    "policy_checkpoint_sha256": policy_sample.get("policy_checkpoint_sha256"),
                },
            )
        records.append(parsed)
    return tuple(records)


def build_dppo_rollout_analysis(
    records: Iterable[TrainingTrajectoryRecord],
    *,
    config: G2POCreditConfig | None = None,
    epsilon: float = 1e-9,
) -> dict[str, Any]:
    """Measure outcome, collection quality, grouping efficiency, and credit change."""

    rows = validate_active_trajectory_records(tuple(records), require_v2=True)
    if not rows:
        raise DPPORolloutAnalysisError("rollout analysis requires active transitions")
    cfg = config or G2POCreditConfig()
    coarse_transitions = g2po_transitions_from_v2_records(
        rows,
        mode=G2POCreditMode.TRAJECTORY_NORMALIZED,
    )
    coarse_credit = assign_unified_credit(
        coarse_transitions,
        treatment=UnifiedCreditTreatment.G2PO_TN,
        g2po_config=cfg,
    )
    annotation_coverage = sum(_has_goal_annotation(row) for row in rows) / len(rows)
    payload: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "collection": _collection_metrics(rows),
        "annotation_coverage": annotation_coverage,
        "coarse_credit": _credit_metrics(rows, coarse_credit, epsilon=epsilon),
        "coarse_grouping": _group_metrics(rows, coarse_credit),
        "refined_credit": None,
        "refined_grouping": None,
        "mechanism": None,
        "counterfactual_credit": None,
        "observation_projection_audit": None,
        "claim_scope": "collection_and_credit_diagnostics_not_policy_effectiveness",
    }
    if annotation_coverage != 1.0:
        return payload

    refined_transitions = dppo_transitions_from_v2_records(
        rows,
        mode=DPPOCreditMode.ORACLE_PHASE_REFINED_TN,
    )
    refined_credit = assign_unified_credit(
        refined_transitions,
        treatment=UnifiedCreditTreatment.ORACLE_PHASE_REFINED_G2PO_TN,
        g2po_config=cfg,
    )
    mechanism = assess_oracle_subproblem_trace(rows)
    payload.update(
        {
            "refined_credit": _credit_metrics(rows, refined_credit, epsilon=epsilon),
            "refined_grouping": _group_metrics(rows, refined_credit),
            "mechanism": asdict(mechanism),
            "observation_projection_audit": _observation_projection_audit(rows),
            "counterfactual_credit": _compare_credit(
                rows,
                coarse_credit,
                refined_credit,
                epsilon=epsilon,
            ),
        }
    )
    return payload


def _observation_projection_audit(
    rows: tuple[TrainingTrajectoryRecord, ...],
) -> dict[str, Any]:
    """Compare collision coverage under official and diagnostic observations.

    Only the official anchor is used by G2PO/DPPO credit.  Raw environment text
    is a diagnostic negative control because the pinned ALFWorld manager
    intentionally replaces ``Nothing happens`` with the preceding visible
    anchor and appends admissible actions after the initial step.
    """

    def metrics(
        selected: tuple[tuple[TrainingTrajectoryRecord, str], ...],
        *,
        claim_scope: str,
    ) -> dict[str, Any]:
        identities: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        visits: Counter[tuple[str, str]] = Counter()
        tasks: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row, observation in selected:
            key = (row.group_uid, observation)
            identities[key].add(
                (
                    str(row.extra["residual_goal_signature"]),
                    str(row.extra["frontier_signature"]),
                )
            )
            visits[key] += 1
            tasks[key].add(row.task_id)
        collisions = {key for key, values in identities.items() if len(values) > 1}
        nothing_happens = {key for key in collisions if key[1].startswith("Nothing happens")}
        return {
            "coverage": len(selected) / len(rows),
            "observation_group_count": len(identities),
            "cross_frontier_observation_group_count": len(collisions),
            "cross_frontier_transition_count": sum(visits[key] for key in collisions),
            "cross_frontier_task_count": len(
                {task_id for key in collisions for task_id in tasks[key]}
            ),
            "nothing_happens_cross_frontier_group_count": len(nothing_happens),
            "claim_scope": claim_scope,
        }

    official = tuple((row, row.observation) for row in rows)
    raw = tuple(
        (row, str(row.extra["alfworld_raw_observation"]))
        for row in rows
        if isinstance(row.extra.get("alfworld_raw_observation"), str)
    )
    return {
        "official_anchor": metrics(official, claim_scope="credit_key_mechanism"),
        "raw_environment_observation": metrics(
            raw,
            claim_scope="diagnostic_only_not_credit_key",
        ),
    }


def compare_paired_rollouts(
    left: Iterable[TrainingTrajectoryRecord],
    right: Iterable[TrainingTrajectoryRecord],
) -> dict[str, Any]:
    """Check whether a declared credit-only pair actually used the same rollout."""

    left_rows = validate_active_trajectory_records(tuple(left), require_v2=True)
    right_rows = validate_active_trajectory_records(tuple(right), require_v2=True)
    left_by_key = {(row.traj_uid, row.step_idx): row for row in left_rows}
    right_by_key = {(row.traj_uid, row.step_idx): row for row in right_rows}
    shared = sorted(set(left_by_key) & set(right_by_key))
    field_matches: dict[str, int] = {
        "observation": 0,
        "action_raw": 0,
        "environment_action": 0,
        "prompt_hash": 0,
        "outcome": 0,
        "task_identity": 0,
        "actor_input": 0,
        "action": 0,
        "transition": 0,
        "policy_and_runtime": 0,
    }
    trace_fingerprint_matches = 0
    policy_sample_evidence = 0
    for key in shared:
        left_row = left_by_key[key]
        right_row = right_by_key[key]
        field_matches["observation"] += left_row.observation == right_row.observation
        field_matches["action_raw"] += left_row.action_raw == right_row.action_raw
        field_matches["environment_action"] += left_row.extra.get(
            "environment_action"
        ) == right_row.extra.get("environment_action")
        field_matches["prompt_hash"] += left_row.prompt_hash == right_row.prompt_hash
        left_identity = _trace_identity(left_row)
        right_identity = _trace_identity(right_row)
        for category in (
            "task_identity",
            "actor_input",
            "action",
            "transition",
            "outcome",
            "policy_and_runtime",
        ):
            field_matches[category] += left_identity[category] == right_identity[category]
        trace_fingerprint_matches += canonical_sha256(left_identity) == canonical_sha256(
            right_identity
        )
        policy_sample_evidence += bool(
            left_row.extra.get("policy_sample_sha256")
            and right_row.extra.get("policy_sample_sha256")
            and left_row.extra.get("policy_checkpoint_sha256")
            and right_row.extra.get("policy_checkpoint_sha256")
        )
    denominator = len(shared)
    rates = {
        name: count / denominator if denominator else 0.0 for name, count in field_matches.items()
    }
    policy_sample_evidence_coverage = policy_sample_evidence / denominator if denominator else 0.0
    trace_fingerprint_match_rate = trace_fingerprint_matches / denominator if denominator else 0.0
    return {
        "schema": PAIRING_SCHEMA,
        "left_transition_count": len(left_rows),
        "right_transition_count": len(right_rows),
        "shared_transition_count": denominator,
        "left_only_transition_count": len(set(left_by_key) - set(right_by_key)),
        "right_only_transition_count": len(set(right_by_key) - set(left_by_key)),
        "field_match_rate": rates,
        "trace_fingerprint_match_rate": trace_fingerprint_match_rate,
        "policy_sample_evidence_coverage": policy_sample_evidence_coverage,
        "exact_credit_only_rollout_match": (
            len(left_rows) == len(right_rows) == denominator
            and trace_fingerprint_match_rate == 1.0
            and policy_sample_evidence_coverage == 1.0
        ),
    }


def _trace_identity(row: TrainingTrajectoryRecord) -> dict[str, Any]:
    """Project all non-treatment evidence required for a credit-only pair."""

    extra = row.extra
    return {
        "task_identity": {
            "dataset_id": row.dataset_id,
            "benchmark": row.benchmark,
            "task_id": row.task_id,
            "split": row.split,
            "seed": row.seed,
            "traj_uid": row.traj_uid,
            "group_uid": row.group_uid,
            "step_idx": row.step_idx,
            "instruction": row.instruction,
            "goal_hash": row.goal_hash,
            "task_manifest_lock_sha256": extra.get("task_manifest_lock_sha256"),
            "alfworld_task_metadata": extra.get("alfworld_task_metadata"),
        },
        "actor_input": {
            "observation": row.observation,
            "prompt_hash": row.prompt_hash,
            "actor_prompt": extra.get("actor_prompt"),
            "actor_protocol_id": extra.get("actor_protocol_id"),
            "actor_awareness_mode": extra.get("actor_awareness_mode"),
            "actor_context_extension_revision": extra.get("actor_context_extension_revision"),
            "actor_awareness": extra.get("actor_awareness"),
            "remaining_budget": row.remaining_budget,
            "action_affordance_hash": row.action_affordance_hash,
        },
        "action": {
            "action_raw": row.action_raw,
            "action_type": row.action_type,
            "parsed_action": row.parsed_action,
            "environment_action": extra.get("environment_action"),
            "is_action_valid": extra.get("is_action_valid"),
            "invalid_action_reasons": extra.get("invalid_action_reasons"),
            "action_language": row.action_language,
            "action_token_count": row.action_token_count,
            "model_response": extra.get("model_response"),
        },
        "transition": {
            "next_observation": row.next_observation,
            "state_snapshot_sha256": row.state_snapshot_sha256,
            "alfworld_raw_observation": extra.get("alfworld_raw_observation"),
            "alfworld_raw_next_observation": extra.get("alfworld_raw_next_observation"),
            "alfworld_facts": extra.get("alfworld_facts"),
            "alfworld_next_facts": extra.get("alfworld_next_facts"),
            "restore_verified": extra.get("restore_verified"),
            "restore_probe_receipt_sha256": extra.get("restore_probe_receipt_sha256"),
        },
        "outcome": {
            "reward": row.reward,
            "terminal": row.terminal,
            "success": row.success,
            "outcome_evidence": row.outcome_evidence,
            "trajectory_length": row.trajectory_length,
            "terminal_reason": row.terminal_reason,
        },
        "policy_and_runtime": {
            "method_source": row.method_source,
            "model_id": row.model_id,
            "rollout_config_hash": row.rollout_config_hash,
            "policy_sample_sha256": extra.get("policy_sample_sha256"),
            "policy_checkpoint_sha256": extra.get("policy_checkpoint_sha256"),
            "sampling_temperature": extra.get("sampling_temperature"),
            "train_task_selection": extra.get("train_task_selection"),
            "remaining_budget_verified": extra.get("remaining_budget_verified"),
            "action_affordance_verified": extra.get("action_affordance_verified"),
            "action_token_count_verified": extra.get("action_token_count_verified"),
        },
    }


def analyze_rollout_artifact(
    annotated_path: str | Path,
    *,
    paired_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    annotated = load_rollout_artifact(annotated_path)
    analysis = build_dppo_rollout_analysis(annotated)
    result = {
        **analysis,
        "annotated_rollout_path": str(Path(annotated_path).resolve()),
        "paired_rollout": (
            compare_paired_rollouts(load_rollout_artifact(paired_path), annotated)
            if paired_path is not None
            else None
        ),
    }
    if output_path is not None:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _has_goal_annotation(row: TrainingTrajectoryRecord) -> bool:
    return bool(
        row.extra.get("goal_annotation_verified") is True
        and row.extra.get("goal_graph_sha256")
        and row.extra.get("residual_goal_signature")
        and row.extra.get("frontier_signature")
    )


def _collection_metrics(rows: tuple[TrainingTrajectoryRecord, ...]) -> dict[str, Any]:
    trajectories = {row.traj_uid for row in rows}
    terminal = [row for row in rows if row.terminal]
    token_counts = [row.action_token_count for row in rows]
    valid = [bool(row.extra.get("is_action_valid", True)) for row in rows]
    invalid_reasons = Counter(
        str(reason)
        for row in rows
        if not bool(row.extra.get("is_action_valid", True))
        for reason in row.extra.get("invalid_action_reasons", ())
    )
    task_types_by_id = {
        row.task_id: str(row.extra.get("alfworld_task_metadata", {}).get("task_type", "unknown"))
        for row in rows
    }
    task_type_counts = Counter(task_types_by_id.values())
    frontier_audit_rows = tuple(row for row in rows if has_frontier_audit_receipt(row))
    phase_changes_by_trajectory: dict[str, int] = defaultdict(int)
    legacy_frontier_disagreements = 0
    for row in sorted(rows, key=lambda item: (item.traj_uid, item.step_idx)):
        frontier_ids = row.extra.get("active_frontier_ids")
        if isinstance(frontier_ids, (list, tuple)):
            frontier = tuple(str(value) for value in frontier_ids)
            frontier_after_ids = row.extra.get("active_frontier_ids_after")
            if isinstance(frontier_after_ids, (list, tuple)):
                frontier_after = tuple(str(value) for value in frontier_after_ids)
                phase_changes_by_trajectory[row.traj_uid] += int(frontier != frontier_after)
            legacy = row.extra.get("active_subproblem_id")
            if legacy is not None and not (
                str(legacy) in frontier or (str(legacy) == "complete" and not frontier)
            ):
                legacy_frontier_disagreements += 1
    trajectories_with_phase_progression = sum(
        change_count > 0 for change_count in phase_changes_by_trajectory.values()
    )
    return {
        "transition_count": len(rows),
        "trajectory_count": len(trajectories),
        "task_count": len({row.task_id for row in rows}),
        "successful_trajectory_count": sum(row.success for row in terminal),
        "positive_reward_transition_count": sum(row.reward > 0 for row in rows),
        "valid_action_count": sum(valid),
        "valid_action_rate": sum(valid) / len(valid),
        "invalid_action_count": len(valid) - sum(valid),
        "invalid_action_reason_counts": dict(sorted(invalid_reasons.items())),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "phase_identity_source": "goal_graph_active_frontier_ids",
        "phase_change_transition_count": sum(phase_changes_by_trajectory.values()),
        "trajectories_with_phase_progression_count": (trajectories_with_phase_progression),
        "trajectories_with_phase_progression_rate": (
            trajectories_with_phase_progression / len(trajectories)
        ),
        "legacy_phase_frontier_disagreement_count": legacy_frontier_disagreements,
        "frontier_live_precondition_audit_coverage": (len(frontier_audit_rows) / len(rows)),
        "frontier_live_precondition_violation_count": (
            count_frontier_live_precondition_violations(frontier_audit_rows)
        ),
        "newly_satisfied_goal_transition_count": sum(
            bool(row.extra.get("newly_satisfied_goal_ids")) for row in rows
        ),
        "regressed_goal_transition_count": sum(
            bool(row.extra.get("regressed_goal_ids")) for row in rows
        ),
        "action_token_count": sum(token_counts),
        "action_tokens_min": min(token_counts),
        "action_tokens_median": median(token_counts),
        "action_tokens_max": max(token_counts),
    }


def _group_metrics(
    rows: tuple[TrainingTrajectoryRecord, ...],
    credit: tuple[UnifiedCreditAssignment, ...],
) -> dict[str, Any]:
    visits: dict[tuple[str, Any], int] = defaultdict(int)
    trajectories: dict[tuple[str, Any], set[str]] = defaultdict(set)
    for row, assignment in zip(rows, credit, strict=True):
        key = (row.group_uid, assignment.state_key)
        visits[key] += 1
        trajectories[key].add(row.traj_uid)
    visit_sizes = list(visits.values())
    effective_sizes = [len(value) for value in trajectories.values()]
    return {
        "state_group_count": len(visits),
        "mean_visit_count": mean(visit_sizes),
        "mean_effective_trajectory_count": mean(effective_sizes),
        "singleton_state_group_count": sum(size == 1 for size in effective_sizes),
        "singleton_state_group_rate": (
            sum(size == 1 for size in effective_sizes) / len(effective_sizes)
        ),
        "max_effective_trajectory_count": max(effective_sizes),
    }


def _credit_metrics(
    rows: tuple[TrainingTrajectoryRecord, ...],
    credit: tuple[UnifiedCreditAssignment, ...],
    *,
    epsilon: float,
) -> dict[str, Any]:
    advantages = [item.advantage for item in credit]
    episode = [item.episode_advantage for item in credit]
    node = [float(item.node_advantage or 0.0) for item in credit]
    edge = [float(item.edge_advantage or 0.0) for item in credit]
    invalid_indices = [
        index for index, row in enumerate(rows) if not bool(row.extra.get("is_action_valid", True))
    ]
    return {
        "nonzero_advantage_count": _nonzero_count(advantages, epsilon),
        "nonzero_episode_advantage_count": _nonzero_count(episode, epsilon),
        "nonzero_node_advantage_count": _nonzero_count(node, epsilon),
        "nonzero_edge_advantage_count": _nonzero_count(edge, epsilon),
        "mean_abs_advantage": mean(abs(value) for value in advantages),
        "max_abs_advantage": max(abs(value) for value in advantages),
        "invalid_action_nonzero_advantage_count": sum(
            abs(advantages[index]) > epsilon for index in invalid_indices
        ),
    }


def _compare_credit(
    rows: tuple[TrainingTrajectoryRecord, ...],
    coarse: tuple[UnifiedCreditAssignment, ...],
    refined: tuple[UnifiedCreditAssignment, ...],
    *,
    epsilon: float,
) -> dict[str, Any]:
    if len(coarse) != len(refined):
        raise DPPORolloutAnalysisError("credit comparison lengths differ")
    deltas = [
        refined_item.advantage - coarse_item.advantage
        for coarse_item, refined_item in zip(coarse, refined, strict=True)
    ]
    sign_disagreements = 0
    zero_to_nonzero = 0
    refined_identities_by_coarse: dict[tuple[str, Any], set[Any]] = defaultdict(set)
    transition_count_by_coarse: dict[tuple[str, Any], int] = defaultdict(int)
    for coarse_item, refined_item in zip(coarse, refined, strict=True):
        left = coarse_item.advantage
        right = refined_item.advantage
        if abs(left) <= epsilon < abs(right) or abs(right) <= epsilon < abs(left):
            zero_to_nonzero += 1
        if (
            abs(left) > epsilon
            and abs(right) > epsilon
            and math.copysign(1, left) != math.copysign(1, right)
        ):
            sign_disagreements += 1
    for row, coarse_item, refined_item in zip(rows, coarse, refined, strict=True):
        key = (row.group_uid, coarse_item.state_key)
        refined_identities_by_coarse[key].add(refined_item.state_key)
        transition_count_by_coarse[key] += 1
    split_groups = {
        key for key, identities in refined_identities_by_coarse.items() if len(identities) > 1
    }
    split_transitions = sum(transition_count_by_coarse[key] for key in split_groups)
    return {
        "advantage_changed_count": _nonzero_count(deltas, epsilon),
        "advantage_changed_rate": _nonzero_count(deltas, epsilon) / len(deltas),
        "mean_abs_advantage_delta": mean(abs(value) for value in deltas),
        "max_abs_advantage_delta": max(abs(value) for value in deltas),
        "credit_sign_disagreement_count": sign_disagreements,
        "zero_nonzero_disagreement_count": zero_to_nonzero,
        "coarse_state_groups_split_by_refinement": len(split_groups),
        "transitions_in_split_coarse_groups": split_transitions,
        "split_candidate_transition_rate": split_transitions / len(rows),
    }


def _nonzero_count(values: Iterable[float], epsilon: float) -> int:
    return sum(abs(value) > epsilon for value in values)
