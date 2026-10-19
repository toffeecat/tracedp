"""Snapshot-restored continuation and one-step action interventions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.studies.runtime import AlfworldTask
from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.alfworld_agentic_suite import (
    AlfWorldBranch,
    AlfWorldBranchCheckpoint,
    AlfWorldBranchObservation,
    BranchFactory,
)
from lh_platform.training.alfworld_controlled_aliasing import (
    AlfWorldControlledAliasingRecipe,
    ControlledAliasingBranchRecipe,
)
from lh_platform.training.parent_child_state_use.contracts import (
    FunctionalStateUseError,
    ReconstructedState,
)
from lh_platform.training.parent_child_state_use.state_pair import (
    require_reconstructed_state_match,
    require_verified_step_start,
)
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.base import GoalGraph, goal_state_projection


@dataclass
class VerifiedBranchCheckpoint:
    """One reconstructed state with a checkpoint verified against its evidence."""

    branch: AlfWorldBranch
    checkpoint: AlfWorldBranchCheckpoint
    expected_state: ReconstructedState

    def restore(self, *, context: str) -> AlfWorldBranchObservation:
        observation = self.branch.restore_checkpoint(self.checkpoint)
        require_reconstructed_state_match(
            observation,
            self.expected_state,
            context=context,
        )
        return observation

    def close(self) -> None:
        self.branch.close()


def prepare_verified_checkpoint(
    recipe: AlfWorldControlledAliasingRecipe,
    branch_recipe: ControlledAliasingBranchRecipe,
    *,
    task: AlfworldTask,
    branch_factory: BranchFactory,
    artifacts_dir: Path,
    expected_state: ReconstructedState,
    context: str,
) -> VerifiedBranchCheckpoint:
    branch = branch_factory(task, recipe.seed, artifacts_dir)
    try:
        observation = branch.reset()
        for action in (*recipe.setup_actions, *branch_recipe.actions):
            if action not in observation.admissible_actions:
                raise FunctionalStateUseError(
                    f"{context} reconstruction action is not admissible: {action}"
                )
            observation = branch.step(action).observation
        require_reconstructed_state_match(
            observation,
            expected_state,
            context=f"{context} prefix/{branch_recipe.branch_id}",
        )
        return VerifiedBranchCheckpoint(
            branch=branch,
            checkpoint=branch.capture_checkpoint(),
            expected_state=expected_state,
        )
    except Exception:
        branch.close()
        raise


def evaluate_continuation_programs(
    recipe: AlfWorldControlledAliasingRecipe,
    branch_recipe: ControlledAliasingBranchRecipe,
    *,
    task: AlfworldTask,
    task_metadata: Mapping[str, Any],
    branch_factory: BranchFactory,
    artifacts_dir: Path,
    compiler: AlfWorldSubproblemCompiler,
    graph: GoalGraph,
    expected_state: ReconstructedState,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prepared = prepare_verified_checkpoint(
        recipe,
        branch_recipe,
        task=task,
        branch_factory=branch_factory,
        artifacts_dir=artifacts_dir,
        expected_state=expected_state,
        context="continuation",
    )
    branch = prepared.branch
    try:
        for program in recipe.continuation_programs:
            observation = prepared.restore(
                context=(f"continuation restore/{branch_recipe.branch_id}/{program.program_id}"),
            )
            last = None
            first_action_snapshot_sha256 = ""
            for index, action in enumerate(program.actions):
                if action not in observation.admissible_actions:
                    raise FunctionalStateUseError(
                        f"continuation action is not admissible: {program.program_id}/{action}"
                    )
                last = branch.step(action)
                require_verified_step_start(
                    last,
                    expected_snapshot_sha256=(
                        expected_state.state_snapshot_sha256 if index == 0 else None
                    ),
                    context=(
                        f"continuation/{branch_recipe.branch_id}/{program.program_id}/{index}"
                    ),
                )
                if index == 0:
                    first_action_snapshot_sha256 = last.state_snapshot_sha256
                observation = last.observation
                if last.done and index != len(program.actions) - 1:
                    raise FunctionalStateUseError(
                        "continuation terminated before its final registered action"
                    )
            if last is None:
                raise FunctionalStateUseError("continuation program is empty")
            vector = compiler.project_state(task_metadata, observation.facts)
            verifier_complete = not goal_state_projection(graph, vector)["residual_goal_ids"]
            if bool(last.won) != verifier_complete:
                raise FunctionalStateUseError(
                    "official outcome disagrees with fact-goal completion"
                )
            rows.append(
                {
                    "program_id": program.program_id,
                    "actions": list(program.actions),
                    "action_count": len(program.actions),
                    "first_action": program.actions[0],
                    "official_won": bool(last.won),
                    "verifier_complete": verifier_complete,
                    "first_action_state_snapshot_sha256": first_action_snapshot_sha256,
                    "first_action_restore_verified": True,
                    "outcome_evidence_sha256": canonical_sha256(
                        {
                            "replay_outcome": dict(last.replay_outcome or {}),
                            "verifier_complete": verifier_complete,
                        }
                    ),
                }
            )
    finally:
        prepared.close()
    return rows


def shortest_successful_first_actions(
    rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    successful = [row for row in rows if row["official_won"]]
    if not successful:
        raise FunctionalStateUseError("state has no successful registered continuation")
    shortest = min(int(row["action_count"]) for row in successful)
    return tuple(
        sorted(
            {str(row["first_action"]) for row in successful if int(row["action_count"]) == shortest}
        )
    )


def label_one_step_actions(
    recipe: AlfWorldControlledAliasingRecipe,
    branch_recipe: ControlledAliasingBranchRecipe,
    *,
    task: AlfworldTask,
    task_metadata: Mapping[str, Any],
    branch_factory: BranchFactory,
    artifacts_dir: Path,
    compiler: AlfWorldSubproblemCompiler,
    graph: GoalGraph,
    actions: tuple[str, ...],
    target_actions: tuple[str, ...],
    expected_state: ReconstructedState,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    prepared = prepare_verified_checkpoint(
        recipe,
        branch_recipe,
        task=task,
        branch_factory=branch_factory,
        artifacts_dir=artifacts_dir,
        expected_state=expected_state,
        context="one-step label",
    )
    branch = prepared.branch
    try:
        for action in actions:
            observation = prepared.restore(
                context=f"one-step label restore/{branch_recipe.branch_id}/{action}",
            )
            before = compiler.project_state(task_metadata, observation.facts)
            if action not in observation.admissible_actions:
                raise FunctionalStateUseError("scored action is not environment admissible")
            result = branch.step(action)
            require_verified_step_start(
                result,
                expected_snapshot_sha256=expected_state.state_snapshot_sha256,
                context=f"one-step label/{branch_recipe.branch_id}/{action}",
            )
            after = compiler.project_state(
                task_metadata,
                result.observation.facts,
                previous=before,
            )
            label = compiler.label_transition(graph, before, action, after)
            before_residual = goal_state_projection(graph, before)["residual_goal_ids"]
            one_step_progress = bool(label.newly_satisfied_goal_ids) or (
                len(label.residual_goal_ids) < len(before_residual)
            )
            terminal_failure = bool(result.done and not result.won)
            rows[action] = {
                "target_by_shortest_successful_continuation": action in target_actions,
                "one_step_progress": one_step_progress,
                "newly_satisfied_goal_ids": list(label.newly_satisfied_goal_ids),
                "regresses_goal": bool(label.regressed_goal_ids),
                "regressed_goal_ids": list(label.regressed_goal_ids),
                "no_detected_goal_regression_or_immediate_failure": (
                    not label.regressed_goal_ids and not terminal_failure
                ),
                "environment_done": bool(result.done),
                "official_won": bool(result.won),
                "start_state_snapshot_sha256": result.state_snapshot_sha256,
                "restore_receipt_sha256": result.restore_receipt_sha256,
                "restore_verified": True,
            }
    finally:
        prepared.close()
    return rows


__all__ = [
    "evaluate_continuation_programs",
    "label_one_step_actions",
    "prepare_verified_checkpoint",
    "shortest_successful_first_actions",
    "VerifiedBranchCheckpoint",
]
