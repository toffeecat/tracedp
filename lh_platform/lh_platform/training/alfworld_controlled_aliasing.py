"""Collect a real ALFWorld state pair that survives the pinned G2PO anchor."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lh_platform.studies.alfworld_subproblems import _normalize_facts
from lh_platform.studies.runtime import AlfworldTask, build_alfworld_task_manifest
from lh_platform.training.agentic_rl_components import AlfWorldFactDecomposition
from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.alfworld_agentic_suite import (
    AlfWorldBranch,
    OfficialAlfWorldBranchFactory,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.controlled_aliasing import (
    build_controlled_aliasing_manifest,
)
from lh_platform.training.controlled_continuations import (
    ALFWORLD_CONTROLLED_CONTINUATION_OUTCOME_ARTIFACT_SCHEMA,
    ALFWORLD_CONTROLLED_CONTINUATION_OUTCOME_SCHEMA,
    CONTROLLED_CONTINUATION_CONTROLLER_SOURCE,
    CONTROLLED_CONTINUATION_EVIDENCE_CLAIM_SCOPE,
    CONTROLLED_CONTINUATION_MATRIX_SCHEMA,
    ControlledContinuationCell,
    ControlledContinuationOutcomeEvidence,
    analyze_controlled_continuation_artifact,
)
from lh_platform.training.dataset import sha256_file
from lh_platform.training.g2po.actor_protocol import (
    G2POActorHistoryStep,
    G2POAlfWorldActorProtocol,
    G2POAnchorTracker,
    extract_alfworld_task_description,
    g2po_anchor_tracker_source_sha256,
)
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.base import goal_state_projection

ALFWORLD_CONTROLLED_ALIASING_RECIPE_SCHEMA = "lh_alfworld_controlled_aliasing_recipe/v2"
ALFWORLD_CONTROLLED_ALIASING_COLLECTION_SCHEMA = "lh_alfworld_controlled_aliasing_collection/v2"


class AlfWorldControlledAliasingError(RuntimeError):
    """Raised when a controlled fixture cannot prove its declared invariants."""


@dataclass(frozen=True)
class ControlledAliasingBranchRecipe:
    branch_id: str
    actions: tuple[str, ...]
    expected_frontier_ids: tuple[str, ...]


@dataclass(frozen=True)
class ControlledContinuationProgram:
    program_id: str
    actions: tuple[str, ...]


@dataclass(frozen=True)
class AlfWorldControlledAliasingRecipe:
    recipe_id: str
    task_id: str
    source_revision: str
    seed: int
    construction_source: str
    construction_selection: str
    setup_actions: tuple[str, ...]
    branches: tuple[ControlledAliasingBranchRecipe, ...]
    probe_action: str
    remaining_budget: int
    continuation_gamma: float
    continuation_programs: tuple[ControlledContinuationProgram, ...]
    schema: str = ALFWORLD_CONTROLLED_ALIASING_RECIPE_SCHEMA

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": self.schema,
                "recipe_id": self.recipe_id,
                "task_id": self.task_id,
                "source_revision": self.source_revision,
                "seed": self.seed,
                "construction_source": self.construction_source,
                "construction_selection": self.construction_selection,
                "setup_actions": list(self.setup_actions),
                "branches": [
                    {
                        "branch_id": branch.branch_id,
                        "actions": list(branch.actions),
                        "expected_frontier_ids": list(branch.expected_frontier_ids),
                    }
                    for branch in self.branches
                ],
                "probe_action": self.probe_action,
                "remaining_budget": self.remaining_budget,
                "continuation_gamma": self.continuation_gamma,
                "continuation_programs": [
                    {
                        "program_id": program.program_id,
                        "actions": list(program.actions),
                    }
                    for program in self.continuation_programs
                ],
            }
        )


@dataclass(frozen=True)
class _CollectedBranch:
    recipe: ControlledAliasingBranchRecipe
    record: TrainingTrajectoryRecord
    raw_observation: str
    admissible_actions: tuple[str, ...]
    official_anchor: str
    actor_prompt: str


def load_alfworld_controlled_aliasing_recipe(
    path: str | Path,
) -> AlfWorldControlledAliasingRecipe:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AlfWorldControlledAliasingError(
            f"cannot load controlled-aliasing recipe: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise AlfWorldControlledAliasingError("controlled-aliasing recipe must be a mapping")
    if raw.get("schema") != ALFWORLD_CONTROLLED_ALIASING_RECIPE_SCHEMA:
        raise AlfWorldControlledAliasingError("unsupported controlled-aliasing recipe schema")
    branch_rows = raw.get("branches")
    if not isinstance(branch_rows, list) or len(branch_rows) != 2:
        raise AlfWorldControlledAliasingError("controlled aliasing requires exactly two branches")
    branches: list[ControlledAliasingBranchRecipe] = []
    for item in branch_rows:
        if not isinstance(item, Mapping):
            raise AlfWorldControlledAliasingError("controlled branch recipe must be a mapping")
        expected_raw = item.get("expected_frontier_ids")
        if expected_raw == []:
            expected_frontier_ids: tuple[str, ...] = ()
        else:
            expected_frontier_ids = _actions(
                expected_raw,
                field="expected frontier ids",
            )
        branches.append(
            ControlledAliasingBranchRecipe(
                branch_id=str(item.get("id", "")).strip(),
                actions=_actions(item.get("actions"), field="branch actions"),
                expected_frontier_ids=expected_frontier_ids,
            )
        )
    if len({branch.branch_id for branch in branches}) != 2 or any(
        not branch.branch_id for branch in branches
    ):
        raise AlfWorldControlledAliasingError("controlled branches require two unique ids")
    if len({len(branch.actions) for branch in branches}) != 1:
        raise AlfWorldControlledAliasingError(
            "controlled branches must use equal action counts to hold step/budget fixed"
        )
    construction_source = str(raw.get("construction_source", ""))
    if construction_source != "environment_intervention":
        raise AlfWorldControlledAliasingError("unsupported fixture construction source")
    program_rows = raw.get("continuation_programs")
    if not isinstance(program_rows, list) or len(program_rows) < 2:
        raise AlfWorldControlledAliasingError(
            "controlled aliasing requires at least two continuation programs"
        )
    programs: list[ControlledContinuationProgram] = []
    for item in program_rows:
        if not isinstance(item, Mapping):
            raise AlfWorldControlledAliasingError(
                "controlled continuation program must be a mapping"
            )
        programs.append(
            ControlledContinuationProgram(
                program_id=str(item.get("id", "")).strip(),
                actions=_actions(
                    item.get("actions"),
                    field="continuation program actions",
                ),
            )
        )
    if len({program.program_id for program in programs}) != len(programs) or any(
        not program.program_id for program in programs
    ):
        raise AlfWorldControlledAliasingError("continuation programs require unique non-empty ids")
    continuation_gamma = float(raw.get("continuation_gamma", -1.0))
    if not 0.0 <= continuation_gamma <= 1.0:
        raise AlfWorldControlledAliasingError("continuation gamma must be between zero and one")
    recipe = AlfWorldControlledAliasingRecipe(
        recipe_id=str(raw.get("id", "")).strip(),
        task_id=str(raw.get("task_id", "")).strip(),
        source_revision=str(raw.get("source_revision", "")).strip(),
        seed=int(raw.get("seed", 0)),
        construction_source=construction_source,
        construction_selection=str(raw.get("construction_selection", "")).strip(),
        setup_actions=_actions(raw.get("setup_actions"), field="setup actions"),
        branches=tuple(branches),
        probe_action=str(raw.get("probe_action", "")).strip(),
        remaining_budget=int(raw.get("remaining_budget", -1)),
        continuation_gamma=continuation_gamma,
        continuation_programs=tuple(programs),
    )
    if not recipe.recipe_id or not recipe.task_id or not recipe.probe_action:
        raise AlfWorldControlledAliasingError("recipe id, task id, and probe action are required")
    if len(recipe.source_revision) != 40:
        raise AlfWorldControlledAliasingError(
            "recipe requires an exact 40-character source revision"
        )
    if recipe.remaining_budget < 0:
        raise AlfWorldControlledAliasingError("remaining budget must be non-negative")
    if recipe.construction_selection != "targeted_preregistered_after_cpu_pilot":
        raise AlfWorldControlledAliasingError(
            "controlled recipe must disclose its targeted post-pilot selection"
        )
    return recipe


def collect_alfworld_controlled_aliasing(
    recipe_path: str | Path,
    *,
    source_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
) -> Path:
    """Run both real branches and write an admitted fixture or fail closed."""

    recipe_path = Path(recipe_path).resolve()
    recipe = load_alfworld_controlled_aliasing_recipe(recipe_path)
    recipe_binding = _committed_file_binding(recipe_path)
    if recipe_binding["matches_head"] is not True:
        raise AlfWorldControlledAliasingError(
            "controlled recipe must be committed unchanged before formal collection"
        )
    platform_root = Path(str(recipe_binding["repository_root"]))
    _require_clean_checkout(platform_root)
    platform_revision = _git_revision(platform_root)
    source_root = Path(source_root).resolve()
    data_root = Path(data_root).resolve()
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    actual_revision = _git_revision(source_root)
    if actual_revision != recipe.source_revision:
        raise AlfWorldControlledAliasingError(
            "pinned G2PO source revision differs from controlled recipe"
        )
    _require_clean_checkout(source_root)
    branch_factory = OfficialAlfWorldBranchFactory(
        source_root=source_root,
        data_root=data_root,
    )
    manifest = build_alfworld_task_manifest(data_root)
    matches = [task for task in manifest.tasks if task.task_id == recipe.task_id]
    if len(matches) != 1:
        raise AlfWorldControlledAliasingError(
            f"recipe task id must resolve exactly once, found {len(matches)}"
        )
    task = matches[0]
    trajectory_path = data_root / task.trajectory_path
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    metadata = _task_metadata(task, trajectory)
    instruction = _task_instruction(trajectory)
    group_uid = f"controlled:{recipe.recipe_id}:{task.task_id}:s{recipe.seed}"
    collected: list[_CollectedBranch] = []
    for branch_recipe in recipe.branches:
        branch = branch_factory(
            task,
            recipe.seed,
            target / "environment" / branch_recipe.branch_id,
        )
        try:
            collected.append(
                _collect_branch(
                    branch,
                    task=task,
                    metadata=metadata,
                    instruction=instruction,
                    group_uid=group_uid,
                    recipe=recipe,
                    branch_recipe=branch_recipe,
                )
            )
        finally:
            branch.close()

    annotated = AlfWorldFactDecomposition().annotate(tuple(item.record for item in collected))
    by_id = {row.traj_uid.rsplit(":", 1)[-1]: row for row in annotated}
    collected = [
        _CollectedBranch(
            recipe=item.recipe,
            record=by_id[item.recipe.branch_id],
            raw_observation=item.raw_observation,
            admissible_actions=item.admissible_actions,
            official_anchor=item.official_anchor,
            actor_prompt=item.actor_prompt,
        )
        for item in collected
    ]
    _verify_pair_invariants(collected)
    rollout_path = target / "controlled_aliasing_rollout.jsonl"
    rollout_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema": "lh_agentic_rl_rollout_transition/v1",
                    "record": item.record.to_dict(),
                    "policy_sample": {},
                },
                sort_keys=True,
            )
            + "\n"
            for item in collected
        ),
        encoding="utf-8",
    )
    admission = build_controlled_aliasing_manifest(
        tuple(item.record for item in collected),
        source_rollout_path=rollout_path,
    )
    if not admission["admitted"]:
        raise AlfWorldControlledAliasingError(
            "controlled pair failed evidence admission: " + ", ".join(admission["blocked_by"])
        )
    admission_path = target / "controlled_aliasing_manifest.json"
    admission_path.write_text(
        json.dumps(admission, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    continuation_rows = _collect_continuation_matrix(
        branch_factory=branch_factory,
        task=task,
        task_metadata=metadata,
        recipe=recipe,
        collision_states=collected,
        output_dir=target / "continuations",
    )
    collision_group_uid = collected[0].record.group_uid
    collision_anchor = collected[0].official_anchor
    continuation_matrix = {
        "schema": CONTROLLED_CONTINUATION_MATRIX_SCHEMA,
        "recipe": {
            "path": str(recipe_path),
            "file_sha256": sha256_file(recipe_path),
            "semantic_sha256": recipe.sha256,
            "repository_binding": recipe_binding,
        },
        "collision": {
            "rollout_path": str(rollout_path),
            "rollout_sha256": sha256_file(rollout_path),
            "task_id": task.task_id,
            "group_uid": collision_group_uid,
            "official_anchor": collision_anchor,
            "official_anchor_sha256": canonical_sha256(collision_anchor),
        },
        "gamma": recipe.continuation_gamma,
        "rows": [row.to_dict() for row in continuation_rows],
        "claim_scope": ("controlled_continuation_program_outcomes_not_model_policy_rollouts"),
    }
    continuation_matrix_path = target / "controlled_continuation_matrix.json"
    continuation_matrix_path.write_text(
        json.dumps(continuation_matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    continuation_analysis_path = target / "controlled_continuation_analysis.json"
    continuation_analysis = analyze_controlled_continuation_artifact(
        continuation_matrix_path,
        output_path=continuation_analysis_path,
    )
    if not continuation_analysis["admitted"]:
        raise AlfWorldControlledAliasingError(
            "controlled continuations did not trigger the score mechanism: "
            + ", ".join(continuation_analysis["blocked_by"])
        )
    receipt = {
        "schema": ALFWORLD_CONTROLLED_ALIASING_COLLECTION_SCHEMA,
        "recipe": {
            "path": str(recipe_path),
            "sha256": sha256_file(recipe_path),
            "semantic_sha256": recipe.sha256,
            "construction_source": recipe.construction_source,
            "construction_selection": recipe.construction_selection,
            "reference_plan_consumed_by_collector": False,
            "repository_binding": recipe_binding,
        },
        "source": {
            "root": str(source_root),
            "revision": recipe.source_revision,
            "tracked_files_clean": True,
            "platform_repository_revision": platform_revision,
            "anchor_tracker_implementation_sha256": (g2po_anchor_tracker_source_sha256()),
        },
        "task": {
            "task_id": task.task_id,
            "group_uid": task.group_uid,
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "game_path": str(data_root / task.game_path),
            "game_sha256": sha256_file(data_root / task.game_path),
            "logic_domain_path": str(data_root / "logic" / "alfred.pddl"),
            "logic_domain_sha256": sha256_file(data_root / "logic" / "alfred.pddl"),
            "logic_grammar_path": str(data_root / "logic" / "alfred.twl2"),
            "logic_grammar_sha256": sha256_file(data_root / "logic" / "alfred.twl2"),
        },
        "pair_invariants": _pair_invariants(collected),
        "rollout_path": str(rollout_path),
        "rollout_sha256": sha256_file(rollout_path),
        "admission_manifest_path": str(admission_path),
        "continuation_matrix_path": str(continuation_matrix_path),
        "continuation_matrix_sha256": sha256_file(continuation_matrix_path),
        "continuation_analysis_path": str(continuation_analysis_path),
        "continuation_analysis_sha256": sha256_file(continuation_analysis_path),
        "continuation_mechanism": {
            "state_count": continuation_analysis["state_count"],
            "program_count": continuation_analysis["program_count"],
            "coarse_continuation_score": continuation_analysis["coarse_score"][
                "continuation_score"
            ],
            "refined_continuation_scores": [
                {
                    "state_id": row["state_id"],
                    "continuation_score": row["continuation_score"],
                }
                for row in continuation_analysis["refined_scores"]
            ],
            "continuation_score_range": continuation_analysis["continuation_score_range"],
            "empirical_mse_reduction": continuation_analysis["empirical_mse_reduction"],
            "mechanism_triggered": continuation_analysis["mechanism_triggered"],
        },
        "admitted": True,
        "claim_scope": CONTROLLED_CONTINUATION_EVIDENCE_CLAIM_SCOPE,
    }
    receipt_path = target / "collection_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path


def _collect_continuation_matrix(
    *,
    branch_factory: OfficialAlfWorldBranchFactory,
    task: AlfworldTask,
    task_metadata: Mapping[str, Any],
    recipe: AlfWorldControlledAliasingRecipe,
    collision_states: list[_CollectedBranch],
    output_dir: Path,
) -> list[ControlledContinuationCell]:
    candidates = {item.recipe.branch_id: item.record for item in collision_states}
    rows: list[ControlledContinuationCell] = []
    compiler = AlfWorldSubproblemCompiler()
    goal_graph = compiler.compile_task(task_metadata)
    for branch_recipe in recipe.branches:
        candidate = candidates[branch_recipe.branch_id]
        action_prefix = (*recipe.setup_actions, *branch_recipe.actions)
        for program in recipe.continuation_programs:
            branch = branch_factory(
                task,
                recipe.seed,
                output_dir / branch_recipe.branch_id / program.program_id,
            )
            try:
                observation = branch.reset()
                for action in action_prefix:
                    if action not in observation.admissible_actions:
                        raise AlfWorldControlledAliasingError(
                            "continuation prefix action is not admissible: "
                            f"{branch_recipe.branch_id}/{program.program_id}/{action}"
                        )
                    observation = branch.step(action).observation
                if observation.text != candidate.extra.get("alfworld_raw_observation"):
                    raise AlfWorldControlledAliasingError(
                        "continuation replay changed the candidate raw observation"
                    )
                if list(observation.admissible_actions) != candidate.extra.get(
                    "alfworld_admissible_actions"
                ):
                    raise AlfWorldControlledAliasingError(
                        "continuation replay changed candidate action affordances"
                    )
                if set(_normalize_facts(observation.facts)) != set(
                    _normalize_facts(candidate.extra.get("alfworld_facts", ()))
                ):
                    raise AlfWorldControlledAliasingError(
                        "continuation replay changed candidate environment facts"
                    )

                step_evidence: list[dict[str, Any]] = []
                executed_actions: list[str] = []
                last_step = None
                for action_index, action in enumerate(program.actions):
                    if action not in observation.admissible_actions:
                        raise AlfWorldControlledAliasingError(
                            "continuation action is not admissible: "
                            f"{branch_recipe.branch_id}/{program.program_id}/{action}"
                        )
                    result = branch.step(action)
                    if action_index == 0 and (
                        result.state_snapshot_sha256 != candidate.state_snapshot_sha256
                    ):
                        raise AlfWorldControlledAliasingError(
                            "continuation did not start from the admitted snapshot"
                        )
                    executed_actions.append(action)
                    step_evidence.append(
                        {
                            "action": action,
                            "state_snapshot_sha256": result.state_snapshot_sha256,
                            "state_snapshot_artifact_path": (result.state_snapshot_artifact_path),
                            "restore_receipt_sha256": (result.restore_receipt_sha256),
                            "restore_receipt_path": result.restore_receipt_path,
                            "restore_verified": True,
                        }
                    )
                    observation = result.observation
                    last_step = result
                    if result.done and action_index != len(program.actions) - 1:
                        raise AlfWorldControlledAliasingError(
                            "continuation terminated before its preregistered final action"
                        )
                if last_step is None:
                    raise AlfWorldControlledAliasingError(
                        "continuation program executed no actions"
                    )
                final_vector = compiler.project_state(
                    task_metadata,
                    observation.facts,
                )
                final_projection = goal_state_projection(goal_graph, final_vector)
                residual_goal_ids = tuple(final_projection["residual_goal_ids"])
                verifier_complete = not residual_goal_ids
                if last_step.won != verifier_complete:
                    raise AlfWorldControlledAliasingError(
                        "official ALFWorld outcome disagrees with final fact-goal verifier: "
                        f"{branch_recipe.branch_id}/{program.program_id}"
                    )
                benchmark_score = float(last_step.reward)
                expected_score = 10.0 if verifier_complete else 0.0
                if benchmark_score != expected_score:
                    raise AlfWorldControlledAliasingError(
                        "official ALFWorld score disagrees with final fact-goal verifier: "
                        f"{branch_recipe.branch_id}/{program.program_id}"
                    )
                if not isinstance(last_step.replay_outcome, Mapping):
                    raise AlfWorldControlledAliasingError(
                        "verified ALFWorld step did not expose its replay outcome: "
                        f"{branch_recipe.branch_id}/{program.program_id}"
                    )
                outcome_artifact = {
                    "schema": (ALFWORLD_CONTROLLED_CONTINUATION_OUTCOME_ARTIFACT_SCHEMA),
                    "state_id": branch_recipe.branch_id,
                    "program_id": program.program_id,
                    "task_metadata": dict(task_metadata),
                    "final_facts": list(observation.facts),
                    "replay_outcome": dict(last_step.replay_outcome),
                    "official_outcome": {
                        "benchmark_source": "official_alfworld_infos",
                        "benchmark_score": benchmark_score,
                        "source_reward": float(last_step.replay_outcome["reward"]),
                        "won": bool(last_step.won),
                        "done": bool(last_step.done),
                    },
                    "verifier_source": "environment_fact_goal_compiler",
                    "verifier_projection": final_projection,
                    "final_step_restore_receipt_sha256": step_evidence[-1][
                        "restore_receipt_sha256"
                    ],
                }
                outcome_artifact_path, outcome_artifact_sha256 = (
                    _write_continuation_outcome_artifact(
                        output_dir,
                        outcome_artifact,
                    )
                )
                outcome_evidence = ControlledContinuationOutcomeEvidence(
                    benchmark_score=benchmark_score,
                    official_won=bool(last_step.won),
                    environment_done=bool(last_step.done),
                    verifier_complete=verifier_complete,
                    residual_goal_ids=residual_goal_ids,
                    artifact_path=str(outcome_artifact_path),
                    artifact_sha256=outcome_artifact_sha256,
                    schema=ALFWORLD_CONTROLLED_CONTINUATION_OUTCOME_SCHEMA,
                )
                rows.append(
                    ControlledContinuationCell(
                        state_id=branch_recipe.branch_id,
                        frontier_ids=tuple(candidate.extra["active_frontier_ids"]),
                        state_snapshot_sha256=candidate.state_snapshot_sha256,
                        program_id=program.program_id,
                        program_actions=program.actions,
                        executed_actions=tuple(executed_actions),
                        first_action_state_snapshot_sha256=str(
                            step_evidence[0]["state_snapshot_sha256"]
                        ),
                        first_action_restore_verified=True,
                        outcome_evidence=outcome_evidence,
                        termination=("environment_done" if last_step.done else "program_exhausted"),
                        step_evidence=tuple(step_evidence),
                        controller_source=CONTROLLED_CONTINUATION_CONTROLLER_SOURCE,
                        model_policy_used=False,
                    )
                )
            finally:
                branch.close()
    return rows


def _write_continuation_outcome_artifact(
    output_dir: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    artifact_sha256 = hashlib.sha256(serialized).hexdigest()
    target = output_dir / "outcome_evidence" / f"{artifact_sha256}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(serialized)
    return target.resolve(), artifact_sha256


def _collect_branch(
    branch: AlfWorldBranch,
    *,
    task: AlfworldTask,
    metadata: Mapping[str, Any],
    instruction: str,
    group_uid: str,
    recipe: AlfWorldControlledAliasingRecipe,
    branch_recipe: ControlledAliasingBranchRecipe,
) -> _CollectedBranch:
    protocol = G2POAlfWorldActorProtocol()
    observation = branch.reset()
    task_description = extract_alfworld_task_description(observation.text)
    initial_prompt = protocol.render_prompt(
        task_description=task_description,
        current_observation=observation.text,
        admissible_actions=observation.admissible_actions,
        history=(),
    )
    tracker = G2POAnchorTracker(initial_actor_prompt=initial_prompt)
    official_anchor = observation.text
    history: list[G2POActorHistoryStep] = []
    action_prefix = (*recipe.setup_actions, *branch_recipe.actions)
    for action in action_prefix:
        if action not in observation.admissible_actions:
            raise AlfWorldControlledAliasingError(
                f"branch {branch_recipe.branch_id} action is not admissible: {action}"
            )
        before = observation
        result = branch.step(action)
        history.append(
            G2POActorHistoryStep(
                observation=before.text,
                environment_action=action,
            )
        )
        observation = result.observation
        official_anchor = tracker.project_next(
            raw_observation=observation.text,
            admissible_actions=observation.admissible_actions,
        )
    actor_prompt = protocol.render_prompt(
        task_description=task_description,
        current_observation=observation.text,
        admissible_actions=observation.admissible_actions,
        history=history,
    )
    if recipe.probe_action not in observation.admissible_actions:
        raise AlfWorldControlledAliasingError(
            f"branch {branch_recipe.branch_id} probe action is not admissible"
        )
    proof = branch.step(recipe.probe_action)
    probe_preserved_facts = set(_normalize_facts(observation.facts)) == set(
        _normalize_facts(proof.observation.facts)
    )
    probe_preserved_affordances = (
        observation.admissible_actions == proof.observation.admissible_actions
    )
    next_anchor = tracker.project_next(
        raw_observation=proof.observation.text,
        admissible_actions=proof.observation.admissible_actions,
    )
    record = TrainingTrajectoryRecord(
        dataset_id=f"controlled:{recipe.sha256}",
        benchmark="alfworld_native",
        task_id=task.task_id,
        split=task.split,
        seed=recipe.seed,
        traj_uid=f"{group_uid}:{branch_recipe.branch_id}",
        group_uid=group_uid,
        step_idx=0,
        instruction=instruction,
        observation=official_anchor,
        action_raw=(
            "<think>Run the preregistered recoverability probe.</think>"
            f"<action>{recipe.probe_action}</action>"
        ),
        action_type="text",
        parsed_action={"command": recipe.probe_action},
        next_observation=next_anchor,
        reward=proof.reward,
        terminal=True,
        success=proof.won,
        outcome_evidence={
            "won": proof.won,
            "environment_done": proof.done,
            "controlled_probe_only": True,
        },
        action_language="en",
        method_source="controlled_environment_intervention",
        model_id="none:deterministic_fixture_controller",
        trajectory_length=1,
        terminal_reason="controlled_probe_complete",
        state_snapshot_sha256=proof.state_snapshot_sha256,
        goal_hash=canonical_sha256(metadata),
        remaining_budget=recipe.remaining_budget,
        action_affordance_hash=canonical_sha256(list(observation.admissible_actions)),
        rollout_config_hash=recipe.sha256,
        action_token_count=1,
        prompt_hash=canonical_sha256(actor_prompt),
        extra={
            "remaining_budget_verified": False,
            "remaining_budget_source": "fixture_config_not_environment",
            "controlled_remaining_budget_equal_by_recipe": True,
            "action_affordance_verified": True,
            "action_token_count_verified": False,
            "training_eligible": False,
            "prompt_hash_source": "canonical_prompt_text_not_token_ids",
            "state_snapshot_artifact_sha256": proof.state_snapshot_sha256,
            "state_snapshot_artifact_path": proof.state_snapshot_artifact_path,
            "restore_verified": True,
            "restore_probe_receipt_sha256": proof.restore_receipt_sha256,
            "restore_probe_receipt_path": proof.restore_receipt_path,
            "alfworld_task_metadata": dict(metadata),
            "alfworld_facts": list(observation.facts),
            "alfworld_next_facts": list(proof.observation.facts),
            "alfworld_raw_observation": observation.text,
            "alfworld_raw_next_observation": proof.observation.text,
            "alfworld_admissible_actions": list(observation.admissible_actions),
            "actor_task_description": task_description,
            "actor_protocol_id": "official_g2po_alfworld_actor_b8ffaf56",
            "g2po_source_revision": recipe.source_revision,
            "official_anchor_projection_verified": True,
            "anchor_tracker_implementation_sha256": (g2po_anchor_tracker_source_sha256()),
            "actor_prompt": actor_prompt,
            "actor_history": [
                {
                    "observation": item.observation,
                    "environment_action": item.environment_action,
                }
                for item in history
            ],
            "environment_action": recipe.probe_action,
            "is_action_valid": True,
            "invalid_action_reasons": [],
            "fixture_recipe_sha256": recipe.sha256,
            "fixture_construction_source": recipe.construction_source,
            "fixture_construction_selection": recipe.construction_selection,
            "fixture_action_prefix": list(action_prefix),
            "environment_step_index": len(action_prefix),
            "reference_plan_consumed_by_collector": False,
            "probe_preserved_facts": probe_preserved_facts,
            "probe_preserved_admissible_actions": probe_preserved_affordances,
        },
    )
    return _CollectedBranch(
        recipe=branch_recipe,
        record=record,
        raw_observation=observation.text,
        admissible_actions=observation.admissible_actions,
        official_anchor=official_anchor,
        actor_prompt=actor_prompt,
    )


def _verify_pair_invariants(collected: list[_CollectedBranch]) -> None:
    invariants = _pair_invariants(collected)
    required_true = (
        "same_task",
        "same_group_uid",
        "same_seed",
        "same_step_count",
        "same_configured_continuation_budget",
        "same_probe_action",
        "same_actor_protocol",
        "probe_preserved_facts",
        "probe_preserved_admissible_actions",
        "same_raw_observation",
        "same_admissible_actions",
        "same_official_anchor",
        "different_frontier",
        "different_snapshot",
        "different_actor_prompt",
        "only_expected_transform_effects_differ",
    )
    failed = [name for name in required_true if invariants.get(name) is not True]
    expected = {
        item.recipe.branch_id: list(item.recipe.expected_frontier_ids) for item in collected
    }
    actual = {
        item.recipe.branch_id: item.record.extra.get("active_frontier_ids") for item in collected
    }
    if expected != actual:
        failed.append("expected_frontier_ids")
    if failed:
        raise AlfWorldControlledAliasingError(
            "controlled pair invariant failed: " + ", ".join(failed)
        )


def _pair_invariants(collected: list[_CollectedBranch]) -> dict[str, bool]:
    if len(collected) != 2:
        raise AlfWorldControlledAliasingError("controlled pair must contain two branches")
    left, right = collected
    left_facts = set(_normalize_facts(left.record.extra.get("alfworld_facts", ())))
    right_facts = set(_normalize_facts(right.record.extra.get("alfworld_facts", ())))
    fact_difference = left_facts.symmetric_difference(right_facts)
    metadata = left.record.extra.get("alfworld_task_metadata")
    task_type = str(metadata.get("task_type", "")) if isinstance(metadata, Mapping) else ""
    object_target = ""
    if isinstance(metadata, Mapping):
        params = metadata.get("pddl_params")
        if isinstance(params, Mapping):
            object_target = str(params.get("object_target", "")).strip().lower()
    target_instances = {
        args[0]
        for name, args in left_facts | right_facts
        if name == "objecttype" and len(args) >= 2 and args[-1] == f"{object_target}type"
    }
    exact_transform_difference = _only_expected_transform_effects_differ(
        fact_difference,
        task_type=task_type,
        target_instances=target_instances,
    )
    return {
        "same_task": left.record.task_id == right.record.task_id,
        "same_group_uid": left.record.group_uid == right.record.group_uid,
        "same_seed": left.record.seed == right.record.seed,
        "same_step_count": len(left.recipe.actions) == len(right.recipe.actions),
        "same_configured_continuation_budget": left.record.remaining_budget
        == right.record.remaining_budget,
        "same_probe_action": left.record.extra.get("environment_action")
        == right.record.extra.get("environment_action"),
        "same_actor_protocol": left.record.extra.get("actor_protocol_id")
        == right.record.extra.get("actor_protocol_id"),
        "probe_preserved_facts": bool(
            left.record.extra.get("probe_preserved_facts") is True
            and right.record.extra.get("probe_preserved_facts") is True
        ),
        "probe_preserved_admissible_actions": bool(
            left.record.extra.get("probe_preserved_admissible_actions") is True
            and right.record.extra.get("probe_preserved_admissible_actions") is True
        ),
        "same_raw_observation": left.raw_observation == right.raw_observation,
        "same_admissible_actions": left.admissible_actions == right.admissible_actions,
        "same_official_anchor": left.official_anchor == right.official_anchor,
        "different_frontier": left.record.extra.get("frontier_signature")
        != right.record.extra.get("frontier_signature"),
        "different_snapshot": left.record.state_snapshot_sha256
        != right.record.state_snapshot_sha256,
        "different_actor_prompt": left.actor_prompt != right.actor_prompt,
        "only_expected_transform_effects_differ": exact_transform_difference,
    }


def _only_expected_transform_effects_differ(
    fact_difference: set[tuple[str, tuple[str, ...]]],
    *,
    task_type: str,
    target_instances: set[str],
) -> bool:
    """Match the exact effect closure declared by ALFWorld's PDDL domain."""

    expected_predicates = {
        "pick_clean_then_place_in_recep": frozenset({"isclean"}),
        "pick_heat_then_place_in_recep": frozenset({"ishot", "not_iscool"}),
        "pick_cool_then_place_in_recep": frozenset({"iscool", "not_ishot"}),
    }.get(task_type)
    if expected_predicates is None or not target_instances:
        return False
    return bool(
        len(fact_difference) == len(expected_predicates)
        and {name for name, _args in fact_difference} == expected_predicates
        and all(len(args) == 1 and args[0] in target_instances for _name, args in fact_difference)
    )


def _task_metadata(task: AlfworldTask, trajectory: Mapping[str, Any]) -> dict[str, Any]:
    params = trajectory.get("pddl_params")
    if not isinstance(params, Mapping):
        raise AlfWorldControlledAliasingError("task trajectory lacks pddl_params")
    return {
        "task_id": task.task_id,
        "task_type": str(trajectory.get("task_type", "")),
        "pddl_params": {
            "object_target": params.get("object_target", ""),
            "parent_target": params.get("parent_target", ""),
            "toggle_target": params.get("toggle_target", ""),
        },
    }


def _task_instruction(trajectory: Mapping[str, Any]) -> str:
    annotations = trajectory.get("turk_annotations")
    anns = annotations.get("anns") if isinstance(annotations, Mapping) else None
    if not isinstance(anns, list) or not anns or not isinstance(anns[0], Mapping):
        raise AlfWorldControlledAliasingError("task trajectory lacks human instruction")
    instruction = str(anns[0].get("task_desc", "")).strip()
    if not instruction:
        raise AlfWorldControlledAliasingError("task instruction is empty")
    return instruction


def _actions(value: object, *, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise AlfWorldControlledAliasingError(f"{field} must be a non-empty string list")
    return tuple(str(item) for item in value)


def _git_revision(source_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AlfWorldControlledAliasingError(
            f"cannot verify pinned G2PO source revision: {error}"
        ) from error
    return result.stdout.strip()


def _require_clean_checkout(source_root: Path) -> None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AlfWorldControlledAliasingError(
            f"cannot verify pinned G2PO checkout cleanliness: {error}"
        ) from error
    if result.stdout.strip():
        raise AlfWorldControlledAliasingError("pinned G2PO checkout has tracked-file modifications")


def _committed_file_binding(path: Path) -> dict[str, Any]:
    try:
        root_result = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        root = Path(root_result.stdout.strip()).resolve()
        relative = path.relative_to(root).as_posix()
        revision = _git_revision(root)
        blob_result = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise AlfWorldControlledAliasingError(
            f"cannot bind controlled recipe to repository HEAD: {error}"
        ) from error
    head_payload = blob_result.stdout
    return {
        "repository_root": str(root),
        "repository_revision": revision,
        "repository_relative_path": relative,
        "head_blob_sha256": hashlib.sha256(head_payload).hexdigest(),
        "working_file_sha256": sha256_file(path),
        "matches_head": hashlib.sha256(head_payload).hexdigest() == sha256_file(path),
    }
