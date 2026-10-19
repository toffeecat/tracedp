"""Reconstruct and verify matched ALFWorld residual-state pairs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.studies.alfworld_subproblems import (
    derive_alfworld_active_subproblem,
)
from lh_platform.studies.runtime import AlfworldTask
from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.alfworld_agentic_suite import (
    AlfWorldBranchObservation,
    AlfWorldBranchStep,
    BranchFactory,
)
from lh_platform.training.alfworld_controlled_aliasing import (
    AlfWorldControlledAliasingRecipe,
    ControlledAliasingBranchRecipe,
)
from lh_platform.training.g2po.actor_protocol import (
    G2POActorHistoryStep,
    G2POAlfWorldActorProtocol,
    extract_alfworld_task_description,
)
from lh_platform.training.parent_child_state_use.contracts import (
    FunctionalStateUseError,
    ReconstructedState,
    VerifiedStateEvidence,
)
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.alfworld_semantic_verifier import (
    audit_alfworld_goal_graph_semantics,
    audit_alfworld_snapshot_composition,
)
from lh_platform.training.subproblems.base import (
    GoalGraph,
    goal_state_projection,
)


def reconstruct_state(
    recipe: AlfWorldControlledAliasingRecipe,
    branch_recipe: ControlledAliasingBranchRecipe,
    *,
    task: AlfworldTask,
    task_metadata: Mapping[str, Any],
    branch_factory: BranchFactory,
    artifacts_dir: Path,
    protocol: G2POAlfWorldActorProtocol,
    compiler: AlfWorldSubproblemCompiler,
    graph: GoalGraph,
) -> ReconstructedState:
    branch = branch_factory(task, recipe.seed, artifacts_dir)
    try:
        observation = branch.reset()
        task_description = extract_alfworld_task_description(observation.text)
        history: list[G2POActorHistoryStep] = []
        for action in (*recipe.setup_actions, *branch_recipe.actions):
            if action not in observation.admissible_actions:
                raise FunctionalStateUseError(
                    f"controlled state action is not admissible: {action}"
                )
            before = observation
            result = branch.step(action)
            if result.done:
                raise FunctionalStateUseError(
                    "controlled state terminated before frozen action scoring"
                )
            history.append(
                G2POActorHistoryStep(
                    observation=before.text,
                    environment_action=action,
                )
            )
            observation = result.observation
        vector = compiler.project_state(task_metadata, observation.facts)
        canonical_annotation = derive_alfworld_active_subproblem(
            task_metadata,
            observation.facts,
        )
        canonical_facts = canonical_annotation.verifier_input["facts"]
        if not isinstance(canonical_facts, list):
            raise FunctionalStateUseError("environment fact annotation is not canonical")
        graph_semantics = audit_alfworld_goal_graph_semantics(task_metadata, graph)
        snapshot_semantics = audit_alfworld_snapshot_composition(
            task_metadata,
            graph,
            vector,
            canonical_facts,
        )
        if not graph_semantics.verified or not snapshot_semantics.verified:
            raise FunctionalStateUseError("independent ALFWorld semantic verification failed")
        actual_frontier = tuple(goal_state_projection(graph, vector)["active_frontier_ids"])
        if actual_frontier != branch_recipe.expected_frontier_ids:
            raise FunctionalStateUseError("reconstructed frontier differs from preregistered state")
        normal_prompt = protocol.render_prompt(
            task_description=task_description,
            current_observation=observation.text,
            admissible_actions=observation.admissible_actions,
            history=history,
        )
        fingerprint = observation_fingerprint(observation)
        if recipe.probe_action not in observation.admissible_actions:
            raise FunctionalStateUseError(
                "state-binding probe action is not environment admissible"
            )
        proof = branch.step(recipe.probe_action)
        require_verified_step_start(
            proof,
            expected_snapshot_sha256=proof.state_snapshot_sha256,
            context=f"state binding probe/{branch_recipe.branch_id}",
        )
        if proof.done:
            raise FunctionalStateUseError("state-binding probe terminated the controlled state")
        if observation_fingerprint(proof.observation) != fingerprint:
            raise FunctionalStateUseError(
                "state-binding probe changed observation, facts, affordances, or gamefile"
            )
        return ReconstructedState(
            recipe=branch_recipe,
            observation=observation,
            task_description=task_description,
            history=tuple(history),
            graph=graph,
            vector=vector,
            normal_prompt=normal_prompt,
            evidence=VerifiedStateEvidence(
                state_snapshot_sha256=proof.state_snapshot_sha256,
                state_snapshot_artifact_path=proof.state_snapshot_artifact_path,
                state_restore_receipt_sha256=proof.restore_receipt_sha256,
                state_restore_receipt_path=proof.restore_receipt_path,
                observation_fingerprint_sha256=fingerprint,
                semantic_graph_verification=graph_semantics.to_dict(),
                semantic_snapshot_verification=snapshot_semantics.to_dict(),
            ),
        )
    finally:
        branch.close()


def validate_state_pair(
    states: tuple[ReconstructedState, ...],
) -> None:
    if len({state.observation.text for state in states}) != 1:
        raise FunctionalStateUseError("state pair changed the current raw observation")
    if len({state.observation.admissible_actions for state in states}) != 1:
        raise FunctionalStateUseError("state pair changed admissible actions")
    frontiers = {
        tuple(goal_state_projection(state.graph, state.vector)["active_frontier_ids"])
        for state in states
    }
    if len(frontiers) != len(states):
        raise FunctionalStateUseError("state pair does not expose distinct frontiers")


def observation_fingerprint(observation: AlfWorldBranchObservation) -> str:
    facts = sorted(
        (
            str(fact.get("name", "")),
            tuple(str(value) for value in (fact.get("args") or ())),
        )
        for fact in observation.facts
    )
    return canonical_sha256(
        {
            "text": observation.text,
            "facts": [[name, list(arguments)] for name, arguments in facts],
            "admissible_actions": list(observation.admissible_actions),
            "gamefile": observation.gamefile,
        }
    )


def require_reconstructed_state_match(
    observation: AlfWorldBranchObservation,
    expected: ReconstructedState,
    *,
    context: str,
) -> None:
    if observation_fingerprint(observation) != (expected.observation_fingerprint_sha256):
        raise FunctionalStateUseError(f"{context} did not reconstruct the scored environment state")


def require_verified_step_start(
    result: AlfWorldBranchStep,
    *,
    expected_snapshot_sha256: str | None,
    context: str,
) -> None:
    if (
        not result.state_snapshot_sha256
        or not result.state_snapshot_artifact_path
        or not result.restore_receipt_sha256
        or not result.restore_receipt_path
        or not isinstance(result.replay_outcome, Mapping)
    ):
        raise FunctionalStateUseError(f"{context} lacks snapshot/restore/replay evidence")
    if (
        expected_snapshot_sha256 is not None
        and result.state_snapshot_sha256 != expected_snapshot_sha256
    ):
        raise FunctionalStateUseError(f"{context} did not start from the scored state snapshot")


__all__ = [
    "observation_fingerprint",
    "reconstruct_state",
    "require_reconstructed_state_match",
    "require_verified_step_start",
    "validate_state_pair",
]
