"""ALFWorld task-suite adapter for the platform-native agentic-RL runtime."""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from lh_platform.alfworld_state import (
    AlfWorldReplayStateController,
    AlfWorldStateSnapshot,
    project_scalar_textworld_outcome,
)
from lh_platform.studies.runtime import AlfworldTask, AlfworldTaskManifest
from lh_platform.training.agentic_rl_components import builtin_component
from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    AgenticRolloutBatch,
    EvaluationRequest,
    EvaluationResult,
    PolicyActionRequest,
    RolloutRequest,
    TrainablePolicy,
    canonical_sha256,
    token_ids_sha256,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.dataset import sha256_file
from lh_platform.training.g2po.actor_protocol import (
    CONTEXT_EXTENSION_PLACEMENT_REVISION,
    G2PO_ALFWORLD_ACTOR_PROTOCOL_ID,
    SUPPORTED_ALFWORLD_ACTOR_PROTOCOL_IDS,
    G2POActorHistoryStep,
    G2POAnchorTracker,
    build_alfworld_actor_protocol,
    extract_alfworld_task_description,
    g2po_anchor_tracker_source_sha256,
)
from lh_platform.training.model_io import ConversationIR
from lh_platform.training.parent_child_state_use.exposure import (
    ActorStateExposureMode,
    render_actor_state_exposure_extension,
)
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.base import (
    GoalGraph,
    GoalVector,
    project_actor_awareness,
)


class AlfWorldSuiteError(RuntimeError):
    """Raised when ALFWorld runtime evidence is incomplete or inconsistent."""


def alfworld_manifest_content_sha256(
    manifest_path: str | Path,
    data_root: str | Path,
) -> str:
    """Bind every selected ALFWorld game and trajectory byte to one digest."""

    manifest_path = Path(manifest_path).resolve()
    root = Path(data_root).resolve()
    manifest = AlfworldTaskManifest.read(manifest_path)
    issues = manifest.validate(root)
    if issues:
        raise AlfWorldSuiteError(
            "cannot hash invalid ALFWorld task manifest: "
            + "; ".join(f"{item.code}:{item.message}" for item in issues)
        )
    task_files: list[dict[str, str]] = []
    for task in sorted(
        manifest.tasks,
        key=lambda item: (item.split, item.task_id, item.group_uid),
    ):
        game_path = (root / task.game_path).resolve()
        trajectory_path = (root / task.trajectory_path).resolve()
        if not game_path.is_relative_to(root) or not trajectory_path.is_relative_to(root):
            raise AlfWorldSuiteError(
                f"ALFWorld task artifact resolves outside data root: {task.task_id}"
            )
        task_files.append(
            {
                "split": task.split,
                "task_id": task.task_id,
                "group_uid": task.group_uid,
                "game_path": task.game_path,
                "game_sha256": sha256_file(game_path),
                "trajectory_path": task.trajectory_path,
                "trajectory_sha256": sha256_file(trajectory_path),
            }
        )
    return canonical_sha256(
        {
            "schema": "lh_alfworld_manifest_content_lock/v1",
            "tasks": task_files,
        }
    )


@dataclass(frozen=True)
class AlfWorldSuiteConfig:
    role: str
    manifest_path: str
    data_root: str
    artifacts_dir: str
    source_revision: str
    train_task_groups_per_update: int = 2
    train_task_selection: str = "manifest_rotation"
    eval_tasks_per_horizon: int = 1
    history_length: int = 2
    train_temperature: float = 1.0
    eval_temperature: float = 0.4
    actor_protocol_id: str = G2PO_ALFWORLD_ACTOR_PROTOCOL_ID
    show_admissible_actions: bool = True
    model_io_profile_id: str | None = None
    actor_awareness_mode: str = "none"
    actor_state_exposure_mode: str = "normal"
    transition_evidence_mode: str = "restore_replay"
    evaluation_schedule_path: str | None = None
    evaluation_selection: str = "all_manifest_tasks_exactly_once"

    def __post_init__(self) -> None:
        if self.role not in {"train_task_suite", "eval_task_suite"}:
            raise AgenticRLContractError("ALFWorld suite role must be train or eval")
        if self.train_task_groups_per_update < 1 or self.eval_tasks_per_horizon < 1:
            raise AgenticRLContractError("ALFWorld task budgets must be positive")
        if self.history_length < 0:
            raise AgenticRLContractError("ALFWorld history/prompt configuration is invalid")
        if self.train_temperature <= 0 or self.eval_temperature <= 0:
            raise AgenticRLContractError("ALFWorld actor temperatures must be positive")
        if self.actor_protocol_id not in SUPPORTED_ALFWORLD_ACTOR_PROTOCOL_IDS:
            raise AgenticRLContractError(
                f"ALFWorld suite actor protocol is unsupported: {self.actor_protocol_id}"
            )
        if self.actor_awareness_mode not in {"none", "oracle_goal_graph"}:
            raise AgenticRLContractError(
                "ALFWorld actor awareness must be none or oracle_goal_graph"
            )
        if self.actor_state_exposure_mode not in {
            ActorStateExposureMode.NORMAL.value,
            ActorStateExposureMode.FLAT_ORACLE.value,
            ActorStateExposureMode.GRAPH_ORACLE.value,
            ActorStateExposureMode.CORRUPTED_GRAPH.value,
        }:
            raise AgenticRLContractError("ALFWorld rollout state exposure mode is unsupported")
        if (
            self.actor_awareness_mode != "none"
            and self.actor_state_exposure_mode != ActorStateExposureMode.NORMAL.value
        ):
            raise AgenticRLContractError(
                "legacy actor awareness and structured state exposure cannot be combined"
            )
        if self.transition_evidence_mode not in {
            "restore_replay",
            "snapshot_only",
        }:
            raise AgenticRLContractError("ALFWorld transition evidence mode is unsupported")
        if self.transition_evidence_mode == "snapshot_only" and self.role != "eval_task_suite":
            raise AgenticRLContractError(
                "snapshot_only transition evidence is restricted to evaluation suites"
            )
        if self.train_task_selection not in {
            "manifest_rotation",
            "task_type_round_robin",
        }:
            raise AgenticRLContractError("ALFWorld training task selection is unsupported")
        if self.evaluation_selection not in {
            "all_manifest_tasks_exactly_once",
            "calibration_manifest_exactly_once",
            "reserved_target_manifest_exactly_once",
            "source_seed_schedule",
        }:
            raise AgenticRLContractError("ALFWorld evaluation selection is unsupported")
        if (self.evaluation_schedule_path is not None) != (
            self.evaluation_selection == "source_seed_schedule"
        ):
            raise AgenticRLContractError("source-seed evaluation requires exactly one schedule")


@dataclass(frozen=True)
class AlfWorldBranchObservation:
    text: str
    facts: tuple[Mapping[str, Any], ...]
    admissible_actions: tuple[str, ...]
    gamefile: str


@dataclass(frozen=True)
class _EvaluationEpisode:
    episode_id: str
    environment_seed: int
    task: AlfworldTask


@dataclass(frozen=True)
class AlfWorldBranchStep:
    observation: AlfWorldBranchObservation
    reward: float
    done: bool
    won: bool
    state_snapshot_sha256: str
    state_snapshot_artifact_path: str
    restore_receipt_sha256: str
    restore_receipt_path: str
    replay_outcome: Mapping[str, Any] | None = None
    environment_outcome: Mapping[str, Any] | None = None
    post_state_snapshot_sha256: str = ""
    post_state_snapshot_artifact_path: str = ""
    physical_call_delta: Mapping[str, Any] | None = None
    physical_call_cumulative: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AlfWorldBranchCheckpoint:
    """Opaque in-process checkpoint used to fork verified environment actions."""

    observation: AlfWorldBranchObservation
    snapshot: AlfWorldStateSnapshot


class AlfWorldBranch(Protocol):
    def reset(self) -> AlfWorldBranchObservation: ...

    def step(self, action: str) -> AlfWorldBranchStep: ...

    def physical_call_counters(self) -> Mapping[str, Any]: ...

    def capture_checkpoint(self) -> AlfWorldBranchCheckpoint: ...

    def restore_checkpoint(
        self,
        checkpoint: AlfWorldBranchCheckpoint,
    ) -> AlfWorldBranchObservation: ...

    def close(self) -> None: ...


BranchFactory = Callable[[AlfworldTask, int, Path], AlfWorldBranch]


@dataclass
class _LiveTrajectory:
    task: AlfworldTask
    branch: AlfWorldBranch
    traj_uid: str
    group_uid: str
    observation: AlfWorldBranchObservation
    task_description: str
    history: list[G2POActorHistoryStep]
    anchor_tracker: G2POAnchorTracker
    anchor_observation: str
    records: list[TrainingTrajectoryRecord]
    task_metadata: Mapping[str, Any]
    goal_graph: GoalGraph
    goal_vector: GoalVector
    sampling_seed_base: int = 0
    environment_seed: int = 0
    done: bool = False
    won: bool = False
    terminal_reason: str = ""


@dataclass(frozen=True)
class _ActorStepContext:
    prompt: str
    awareness: Mapping[str, Any] | None
    state_exposure: Mapping[str, Any] | None


def _finalize_outcome_evidence(
    evidence: Mapping[str, Any],
    *,
    trajectory_won: bool,
) -> dict[str, Any]:
    """Promote terminal outcome to every transition without losing step evidence."""

    finalized = dict(evidence)
    finalized["step_won"] = bool(finalized.get("won", False))
    finalized["won"] = bool(trajectory_won)
    return finalized


class AlfWorldNativeTaskSuite:
    """Own task selection and environment traces; never own RL credit logic."""

    def __init__(
        self,
        config: AlfWorldSuiteConfig,
        *,
        branch_factory: BranchFactory,
    ) -> None:
        self.config = config
        self._branch_factory = branch_factory
        self._manifest_path = Path(config.manifest_path).resolve()
        self._data_root = Path(config.data_root).resolve()
        self._artifacts_dir = Path(config.artifacts_dir).resolve()
        self._actor_protocol = build_alfworld_actor_protocol(
            protocol_id=config.actor_protocol_id,
            history_length=config.history_length,
            show_admissible_actions=config.show_admissible_actions,
        )
        self._subproblem_compiler = AlfWorldSubproblemCompiler()
        self.manifest = AlfworldTaskManifest.read(self._manifest_path)
        issues = self.manifest.validate(self._data_root)
        if issues:
            raise AlfWorldSuiteError(
                "invalid ALFWorld task manifest: "
                + "; ".join(f"{item.code}:{item.message}" for item in issues)
            )
        self.manifest_sha256 = sha256_file(self._manifest_path)
        self.dataset_content_sha256 = alfworld_manifest_content_sha256(
            self._manifest_path,
            self._data_root,
        )
        splits = {task.split for task in self.manifest.tasks}
        if len(splits) != 1:
            raise AlfWorldSuiteError("ALFWorld suite manifest must contain exactly one split")
        self.split = next(iter(splits))
        self.evaluation_schedule_sha256: str | None = None
        self._evaluation_schedule: tuple[_EvaluationEpisode, ...] = ()
        if config.evaluation_schedule_path is not None:
            if config.role != "eval_task_suite":
                raise AlfWorldSuiteError("only an evaluation suite may bind an evaluation schedule")
            schedule_path = Path(config.evaluation_schedule_path).resolve()
            self._evaluation_schedule = self._load_evaluation_schedule(schedule_path)
            self.evaluation_schedule_sha256 = sha256_file(schedule_path)
        self.component = builtin_component(
            role=config.role,
            name="alfworld_native",
            config={
                "runtime": "fixed_game_textworld_restore_verified/v1",
                "source_revision": config.source_revision,
                "split": self.split,
                "dataset_content_sha256": self.dataset_content_sha256,
                "train_task_groups_per_update": config.train_task_groups_per_update,
                "train_task_selection": config.train_task_selection,
                "eval_tasks_per_horizon": config.eval_tasks_per_horizon,
                "history_length": config.history_length,
                "actor_protocol_id": config.actor_protocol_id,
                "model_io_profile_id": config.model_io_profile_id,
                "response_contract_id": self._actor_protocol.response_contract.id,
                "response_contract_sha256": (
                    self._actor_protocol.response_contract.descriptor.sha256
                ),
                "actor_awareness_mode": config.actor_awareness_mode,
                "actor_state_exposure_mode": config.actor_state_exposure_mode,
                "actor_context_extension_revision": (
                    CONTEXT_EXTENSION_PLACEMENT_REVISION
                    if (
                        config.actor_awareness_mode != "none"
                        or config.actor_state_exposure_mode != ActorStateExposureMode.NORMAL.value
                    )
                    else "none"
                ),
                "train_temperature": config.train_temperature,
                "eval_temperature": config.eval_temperature,
                "evaluation_schedule_sha256": self.evaluation_schedule_sha256,
                "evaluation_selection": config.evaluation_selection,
            },
        )

    def training_task_selection_receipt(self, *, seed: int) -> dict[str, Any]:
        tasks = self._rotated_tasks(
            seed=seed,
            count=self.config.train_task_groups_per_update,
        )
        return {
            "schema": "lh_alfworld_training_task_selection/v1",
            "strategy": self.config.train_task_selection,
            "seed": seed,
            "task_count": len(tasks),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "group_uid": task.group_uid,
                    "task_type": task.task_type,
                    "horizon_bucket": task.horizon_bucket,
                }
                for task in tasks
            ],
        }

    def collect(
        self,
        policy: TrainablePolicy,
        request: RolloutRequest,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> AgenticRolloutBatch:
        self._assert_dataset_content_lock()
        if self.config.role != "train_task_suite":
            raise AlfWorldSuiteError("evaluation suite cannot collect training rollouts")
        tasks = self._rotated_tasks(
            seed=request.seed,
            count=self.config.train_task_groups_per_update,
        )
        return self._collect_tasks(
            policy,
            request=request,
            tasks=tasks,
            group_size=request.group_size,
            progress_callback=progress_callback,
        )

    def collect_frozen(
        self,
        policy: TrainablePolicy,
        request: RolloutRequest,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> AgenticRolloutBatch:
        """Collect held-out transitions without entering the training API."""

        self._assert_dataset_content_lock()
        if self.config.role != "eval_task_suite":
            raise AlfWorldSuiteError("only evaluation suite can collect held-out data")
        tasks = self._rotated_tasks(
            seed=request.seed,
            count=self.config.train_task_groups_per_update,
        )
        return self._collect_tasks(
            policy,
            request=request,
            tasks=tasks,
            group_size=request.group_size,
            progress_callback=progress_callback,
        )

    def collect_frozen_all(
        self,
        policy: TrainablePolicy,
        request: RolloutRequest,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> AgenticRolloutBatch:
        """Collect one fixed-size group for every manifest task in manifest order."""

        self._assert_dataset_content_lock()
        if self.config.role != "eval_task_suite":
            raise AlfWorldSuiteError("only evaluation suite can collect held-out data")
        if self.config.evaluation_selection not in {
            "all_manifest_tasks_exactly_once",
            "calibration_manifest_exactly_once",
            "reserved_target_manifest_exactly_once",
        }:
            raise AlfWorldSuiteError(
                "exact-manifest frozen collection forbids an external evaluation schedule"
            )
        return self._collect_tasks(
            policy,
            request=request,
            tasks=self.manifest.tasks,
            group_size=request.group_size,
            progress_callback=progress_callback,
        )

    def evaluate(
        self,
        policy: TrainablePolicy,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        self._assert_dataset_content_lock()
        if self.config.role != "eval_task_suite":
            raise AlfWorldSuiteError("training suite cannot run held-out evaluation")
        if self._evaluation_schedule:
            return self._evaluate_tasks(
                policy,
                request,
                self._evaluation_schedule,
                selection_method="source_seed_schedule",
            )
        selected: list[AlfworldTask] = []
        for bucket in ("short", "medium", "long"):
            bucket_tasks = tuple(
                task for task in self.manifest.tasks if task.horizon_bucket == bucket
            )
            if len(bucket_tasks) < self.config.eval_tasks_per_horizon:
                raise AlfWorldSuiteError(
                    f"ALFWorld evaluation lacks {self.config.eval_tasks_per_horizon} {bucket} tasks"
                )
            offset = request.seed % len(bucket_tasks)
            selected.extend(
                bucket_tasks[(offset + index) % len(bucket_tasks)]
                for index in range(self.config.eval_tasks_per_horizon)
            )
        episodes = tuple(
            _EvaluationEpisode(
                episode_id=f"sampled-{index:04d}-{task.task_id}",
                environment_seed=request.seed + index,
                task=task,
            )
            for index, task in enumerate(selected)
        )
        return self._evaluate_tasks(
            policy, request, episodes, selection_method="per_horizon_seed_offset"
        )

    def evaluate_all(
        self,
        policy: TrainablePolicy,
        request: EvaluationRequest,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        transition_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> EvaluationResult:
        """Evaluate every task bound by the held-out manifest exactly once."""

        self._assert_dataset_content_lock()
        if self.config.role != "eval_task_suite":
            raise AlfWorldSuiteError("training suite cannot run held-out evaluation")
        if not self.manifest.tasks:
            raise AlfWorldSuiteError("ALFWorld evaluation manifest is empty")
        episodes = self._evaluation_schedule or tuple(
            _EvaluationEpisode(
                episode_id=f"manifest-{index:04d}-{task.task_id}",
                environment_seed=request.seed + index,
                task=task,
            )
            for index, task in enumerate(self.manifest.tasks)
        )
        return self._evaluate_tasks(
            policy,
            request,
            episodes,
            selection_method="full_manifest_exhaustive",
            progress_callback=progress_callback,
            transition_callback=transition_callback,
        )

    def _evaluate_tasks(
        self,
        policy: TrainablePolicy,
        request: EvaluationRequest,
        selected: tuple[_EvaluationEpisode, ...],
        *,
        selection_method: str = "",
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        transition_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> EvaluationResult:
        live: list[_LiveTrajectory] = []
        environment_seed_by_traj: dict[str, int] = {}
        run_root = self._artifacts_dir / f"eval-seed-{request.seed}-{uuid.uuid4().hex[:8]}"
        try:
            for episode in selected:
                task = episode.task
                traj_uid = f"eval:{episode.episode_id}"
                environment_seed = episode.environment_seed
                branch = self._branch_factory(
                    task,
                    environment_seed,
                    run_root / episode.episode_id / task.task_id,
                )
                environment_seed_by_traj[traj_uid] = environment_seed
                observation = branch.reset()
                task_metadata = self._task_metadata(task)
                goal_graph = self._subproblem_compiler.compile_task(task_metadata)
                goal_vector = self._subproblem_compiler.project_state(
                    task_metadata,
                    observation.facts,
                )
                initial_prompt = self._actor_protocol.render_prompt(
                    task_description=extract_alfworld_task_description(observation.text),
                    current_observation=observation.text,
                    admissible_actions=observation.admissible_actions,
                    history=(),
                )
                live.append(
                    _LiveTrajectory(
                        task=task,
                        branch=branch,
                        traj_uid=traj_uid,
                        group_uid=f"eval:{task.task_id}",
                        observation=observation,
                        task_description=extract_alfworld_task_description(observation.text),
                        history=[],
                        anchor_tracker=G2POAnchorTracker(
                            initial_actor_prompt=initial_prompt,
                            show_admissible_actions=self._actor_protocol.show_admissible_actions,
                        ),
                        anchor_observation=observation.text,
                        records=[],
                        task_metadata=task_metadata,
                        goal_graph=goal_graph,
                        goal_vector=goal_vector,
                    )
                )

            for step_index in range(request.max_episode_steps):
                active = [trajectory for trajectory in live if not trajectory.done]
                if not active:
                    break
                actor_contexts = tuple(
                    self._actor_context(
                        trajectory,
                        remaining_budget=request.max_episode_steps - step_index,
                    )
                    for trajectory in active
                )
                action_requests = tuple(
                    PolicyActionRequest(
                        traj_uid=trajectory.traj_uid,
                        step_idx=step_index,
                        conversation=ConversationIR.from_legacy_user_prompt(actor_context.prompt),
                        model_io_profile_id=self.config.model_io_profile_id,
                        response_contract_id=(self._actor_protocol.response_contract.id),
                        response_contract_sha256=(
                            self._actor_protocol.response_contract.descriptor.sha256
                        ),
                        admissible_actions=(),
                        sampling_temperature=self.config.eval_temperature,
                        response_stop_texts=(self._actor_protocol.response_contract.stop_texts),
                    )
                    for trajectory, actor_context in zip(active, actor_contexts, strict=True)
                )
                samples = policy.sample_actions(
                    action_requests,
                    seed=request.seed + step_index * 100_000,
                )
                if len(samples) != len(active):
                    raise AlfWorldSuiteError(
                        "policy did not return one sample per active evaluation branch"
                    )
                for trajectory, sample, _action_request, actor_context in zip(
                    active, samples, action_requests, actor_contexts, strict=True
                ):
                    if (sample.traj_uid, sample.step_idx) != (
                        trajectory.traj_uid,
                        step_index,
                    ):
                        raise AlfWorldSuiteError(
                            "policy sample key differs from evaluation request"
                        )
                    before = trajectory.observation
                    before_anchor = trajectory.anchor_observation
                    projection = self._actor_protocol.project_response(
                        sample.action_text,
                        admissible_actions=before.admissible_actions,
                    )
                    direct_step = getattr(
                        trajectory.branch,
                        "step_unverified",
                        trajectory.branch.step,
                    )
                    result = direct_step(projection.environment_action)
                    next_goal_vector = self._subproblem_compiler.project_state(
                        trajectory.task_metadata,
                        result.observation.facts,
                        previous=trajectory.goal_vector,
                    )
                    budget_terminal = step_index + 1 >= request.max_episode_steps
                    trajectory.done = bool(result.done or budget_terminal)
                    trajectory.won = bool(result.won)
                    trajectory.history.append(
                        G2POActorHistoryStep(
                            observation=before.text,
                            environment_action=projection.environment_action,
                        )
                    )
                    next_anchor = trajectory.anchor_tracker.project_next(
                        raw_observation=result.observation.text,
                        admissible_actions=result.observation.admissible_actions,
                    )
                    trajectory.observation = result.observation
                    trajectory.anchor_observation = next_anchor
                    if transition_callback is not None:
                        transition_callback(
                            {
                                "schema": "lh_alfworld_evaluation_transition/v1",
                                "task_id": trajectory.task.task_id,
                                "split": trajectory.task.split,
                                "horizon_bucket": trajectory.task.horizon_bucket,
                                "reference_steps": trajectory.task.reference_steps,
                                "traj_uid": trajectory.traj_uid,
                                "step_idx": step_index,
                                "environment_seed": environment_seed_by_traj[trajectory.traj_uid],
                                "prompt": actor_context.prompt,
                                "observation": before.text,
                                "anchor_observation": before_anchor,
                                "environment_facts": list(before.facts),
                                "admissible_actions": list(before.admissible_actions),
                                "model_output": sample.action_text,
                                "environment_action": projection.environment_action,
                                "is_action_valid": projection.is_action_valid,
                                "invalid_action_reasons": list(projection.invalid_reasons),
                                "action_token_ids": list(sample.action_token_ids),
                                "action_token_log_probs": list(sample.old_action_token_log_probs),
                                "sampling_seed": sample.rollout_seed,
                                "sampling_temperature": sample.sampling_temperature,
                                "policy_checkpoint_sha256": (sample.policy_checkpoint_sha256),
                                "next_observation": result.observation.text,
                                "next_anchor_observation": next_anchor,
                                "next_environment_facts": list(result.observation.facts),
                                "reward": result.reward,
                                "environment_done": result.done,
                                "budget_terminal": budget_terminal,
                                "terminal": trajectory.done,
                                "success": result.won,
                                "actor_awareness_mode": self.config.actor_awareness_mode,
                                "actor_awareness": actor_context.awareness,
                                "actor_state_exposure_mode": (
                                    self.config.actor_state_exposure_mode
                                ),
                                "actor_state_exposure": (actor_context.state_exposure),
                            }
                        )
                    trajectory.goal_vector = next_goal_vector
                if progress_callback is not None:
                    progress_callback(
                        {
                            "step_index": step_index,
                            "active_trajectories": sum(
                                int(not trajectory.done) for trajectory in live
                            ),
                            "completed_trajectories": sum(
                                int(trajectory.done) for trajectory in live
                            ),
                            "successful_trajectories": sum(
                                int(trajectory.won) for trajectory in live
                            ),
                            "total_trajectories": len(live),
                        }
                    )
        finally:
            for trajectory in live:
                trajectory.branch.close()

        by_horizon: dict[str, dict[str, int]] = {
            bucket: {"successes": 0, "tasks": 0} for bucket in ("short", "medium", "long")
        }
        for trajectory in live:
            bucket = trajectory.task.horizon_bucket
            by_horizon[bucket]["tasks"] += 1
            by_horizon[bucket]["successes"] += int(trajectory.won)
        action_steps = [len(trajectory.history) for trajectory in live]
        successes = sum(int(trajectory.won) for trajectory in live)
        return EvaluationResult(
            task_suite=self.component,
            task_manifest_sha256=self.manifest_sha256,
            policy_checkpoint_sha256=policy.checkpoint_sha256,
            task_count=len(live),
            success_count=successes,
            mean_action_steps=sum(action_steps) / len(action_steps),
            success_by_horizon=by_horizon,
            sampling_temperature=self.config.eval_temperature,
            episodes_per_task=1,
            stochastic_sampling=self.config.eval_temperature > 0.0,
            sampling_uncertainty_interval_available=False,
            evaluation_selection_method=selection_method,
        )

    def _assert_dataset_content_lock(self) -> None:
        actual = alfworld_manifest_content_sha256(
            self._manifest_path,
            self._data_root,
        )
        if actual != self.dataset_content_sha256:
            raise AlfWorldSuiteError(
                "ALFWorld dataset content lock changed after suite preflight: "
                f"expected {self.dataset_content_sha256}, got {actual}"
            )

    def _rotated_tasks(self, *, seed: int, count: int) -> tuple[AlfworldTask, ...]:
        tasks = self.manifest.tasks
        if not tasks:
            raise AlfWorldSuiteError("ALFWorld task manifest is empty")
        if self.config.train_task_selection == "task_type_round_robin":
            by_type: dict[str, list[AlfworldTask]] = {}
            for task in tasks:
                by_type.setdefault(task.task_type, []).append(task)
            task_types = sorted(by_type)
            type_offset = seed % len(task_types)
            base_round = seed // len(task_types)
            selected: list[AlfworldTask] = []
            for index in range(count):
                position = type_offset + index
                task_type = task_types[position % len(task_types)]
                candidates = by_type[task_type]
                candidate_index = (base_round + position // len(task_types)) % len(candidates)
                selected.append(candidates[candidate_index])
            return tuple(selected)
        offset = seed % len(tasks)
        return tuple(tasks[(offset + index) % len(tasks)] for index in range(count))

    def _collect_tasks(
        self,
        policy: TrainablePolicy,
        *,
        request: RolloutRequest,
        tasks: Sequence[AlfworldTask],
        group_size: int,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> AgenticRolloutBatch:
        checkpoint_sha256 = policy.checkpoint_sha256
        sampling_cursor = policy.sampling_cursor()
        live: list[_LiveTrajectory] = []
        run_root = self._artifacts_dir / (
            f"update-{request.update_index}-seed-{request.seed}-{uuid.uuid4().hex[:8]}"
        )
        try:
            for task_index, task in enumerate(tasks):
                group_uid = f"{task.task_id}:u{request.update_index}:s{request.seed}"
                branch_seed = request.seed + task_index
                for branch_index in range(group_size):
                    branch_root = run_root / task.task_id / f"branch-{branch_index}"
                    branch = self._branch_factory(task, branch_seed, branch_root)
                    observation = branch.reset()
                    expected_game = self._resolve_task_path(task.game_path)
                    if Path(observation.gamefile).resolve() != expected_game:
                        branch.close()
                        raise AlfWorldSuiteError(
                            "ALFWorld branch reset a game outside the selected manifest task"
                        )
                    task_description = extract_alfworld_task_description(observation.text)
                    task_metadata = self._task_metadata(task)
                    goal_graph = self._subproblem_compiler.compile_task(task_metadata)
                    goal_vector = self._subproblem_compiler.project_state(
                        task_metadata,
                        observation.facts,
                    )
                    initial_prompt = self._actor_protocol.render_prompt(
                        task_description=task_description,
                        current_observation=observation.text,
                        admissible_actions=observation.admissible_actions,
                        history=(),
                    )
                    live.append(
                        _LiveTrajectory(
                            task=task,
                            branch=branch,
                            traj_uid=f"{group_uid}:r{branch_index}",
                            group_uid=group_uid,
                            observation=observation,
                            task_description=task_description,
                            history=[],
                            anchor_tracker=G2POAnchorTracker(
                                initial_actor_prompt=initial_prompt,
                                show_admissible_actions=self._actor_protocol.show_admissible_actions,
                            ),
                            anchor_observation=observation.text,
                            records=[],
                            task_metadata=task_metadata,
                            goal_graph=goal_graph,
                            goal_vector=goal_vector,
                            sampling_seed_base=(
                                (
                                    request.sampling_seed_base
                                    if request.sampling_seed_base is not None
                                    else request.seed
                                )
                                + (task_index * group_size + branch_index)
                                * request.max_episode_steps
                            ),
                            environment_seed=branch_seed,
                        )
                    )

            for step_index in range(request.max_episode_steps):
                active = [trajectory for trajectory in live if not trajectory.done]
                if not active:
                    break
                actor_contexts = tuple(
                    self._actor_context(
                        trajectory,
                        remaining_budget=request.max_episode_steps - step_index,
                    )
                    for trajectory in active
                )
                requests = tuple(
                    PolicyActionRequest(
                        traj_uid=trajectory.traj_uid,
                        step_idx=step_index,
                        conversation=ConversationIR.from_legacy_user_prompt(actor_context.prompt),
                        model_io_profile_id=self.config.model_io_profile_id,
                        response_contract_id=(self._actor_protocol.response_contract.id),
                        response_contract_sha256=(
                            self._actor_protocol.response_contract.descriptor.sha256
                        ),
                        admissible_actions=(),
                        sampling_temperature=(
                            self.config.train_temperature
                            if self.config.role == "train_task_suite"
                            else self.config.eval_temperature
                        ),
                        sampling_seed=trajectory.sampling_seed_base + step_index,
                        response_stop_texts=(self._actor_protocol.response_contract.stop_texts),
                    )
                    for trajectory, actor_context in zip(active, actor_contexts, strict=True)
                )
                samples = policy.sample_actions(
                    requests,
                    seed=request.seed + step_index * 100_000,
                )
                if len(samples) != len(active):
                    raise AlfWorldSuiteError("policy did not return one sample per active branch")
                for trajectory, sample, _action_request, actor_context in zip(
                    active, samples, requests, actor_contexts, strict=True
                ):
                    if (sample.traj_uid, sample.step_idx) != (
                        trajectory.traj_uid,
                        step_index,
                    ):
                        raise AlfWorldSuiteError("policy sample key differs from request")
                    before = trajectory.observation
                    projection = self._actor_protocol.project_response(
                        sample.action_text,
                        admissible_actions=before.admissible_actions,
                    )
                    if self.config.transition_evidence_mode == "restore_replay":
                        result = trajectory.branch.step(projection.environment_action)
                    else:
                        snapshot_step = getattr(
                            trajectory.branch,
                            "step_snapshot_only",
                            None,
                        )
                        if not callable(snapshot_step):
                            raise AlfWorldSuiteError(
                                "snapshot_only evidence requires a compatible ALFWorld branch"
                            )
                        result = snapshot_step(projection.environment_action)
                    restore_verified = self.config.transition_evidence_mode == "restore_replay"
                    if restore_verified and (
                        not result.restore_receipt_sha256 or not result.restore_receipt_path
                    ):
                        raise AlfWorldSuiteError("restore_replay evidence is missing its receipt")
                    if not restore_verified and (
                        result.restore_receipt_sha256 or result.restore_receipt_path
                    ):
                        raise AlfWorldSuiteError(
                            "snapshot_only evidence must not claim a restore receipt"
                        )
                    next_goal_vector = self._subproblem_compiler.project_state(
                        trajectory.task_metadata,
                        result.observation.facts,
                        previous=trajectory.goal_vector,
                    )
                    next_anchor = trajectory.anchor_tracker.project_next(
                        raw_observation=result.observation.text,
                        admissible_actions=result.observation.admissible_actions,
                    )
                    budget_terminal = step_index + 1 >= request.max_episode_steps
                    terminal = bool(result.done or budget_terminal)
                    won = bool(result.won)
                    trajectory.done = terminal
                    trajectory.won = won
                    if won:
                        trajectory.terminal_reason = "success"
                    elif result.done:
                        trajectory.terminal_reason = "environment_done"
                    elif budget_terminal:
                        trajectory.terminal_reason = "max_episode_steps"
                    metadata = trajectory.task_metadata
                    record = TrainingTrajectoryRecord(
                        dataset_id=f"online:{self.manifest_sha256}",
                        benchmark="alfworld_native",
                        task_id=trajectory.task.task_id,
                        split=trajectory.task.split,
                        seed=trajectory.environment_seed,
                        traj_uid=trajectory.traj_uid,
                        group_uid=trajectory.group_uid,
                        step_idx=step_index,
                        instruction=self._task_instruction(trajectory.task),
                        observation=trajectory.anchor_observation,
                        action_raw=sample.action_text,
                        action_type="text",
                        parsed_action={"command": projection.environment_action},
                        next_observation=next_anchor,
                        reward=result.reward,
                        terminal=terminal,
                        success=won,
                        outcome_evidence={
                            "won": won,
                            "environment_done": result.done,
                            "terminal_reason": trajectory.terminal_reason if terminal else "",
                        },
                        action_language="en",
                        method_source="on_policy",
                        model_id=policy.component.name,
                        trajectory_length=None,
                        terminal_reason="pending",
                        decision_role="executor",
                        state_snapshot_sha256=result.state_snapshot_sha256,
                        goal_hash=canonical_sha256(metadata),
                        remaining_budget=request.max_episode_steps - step_index,
                        action_affordance_hash=canonical_sha256(list(before.admissible_actions)),
                        rollout_config_hash=request.rollout_config_sha256,
                        active=True,
                        action_token_count=len(sample.action_token_ids),
                        prompt_hash=token_ids_sha256(sample.prompt_token_ids),
                        prompt_ids=sample.prompt_token_ids,
                        response_ids=sample.action_token_ids,
                        response_mask=(1,) * len(sample.action_token_ids),
                        response_old_logprobs=sample.old_action_token_log_probs,
                        extra={
                            "remaining_budget_verified": True,
                            "action_affordance_verified": True,
                            "action_token_count_verified": True,
                            "task_manifest_lock_sha256": self.manifest_sha256,
                            "train_task_selection": self.config.train_task_selection,
                            "state_snapshot_artifact_sha256": result.state_snapshot_sha256,
                            "state_snapshot_artifact_path": result.state_snapshot_artifact_path,
                            "post_state_snapshot_sha256": (result.post_state_snapshot_sha256),
                            "post_state_snapshot_artifact_path": (
                                result.post_state_snapshot_artifact_path
                            ),
                            "alfworld_physical_call_delta": dict(result.physical_call_delta or {}),
                            "alfworld_physical_call_cumulative": dict(
                                result.physical_call_cumulative or {}
                            ),
                            "transition_evidence_mode": (self.config.transition_evidence_mode),
                            "restore_verified": restore_verified,
                            "restore_probe_receipt_sha256": result.restore_receipt_sha256,
                            "restore_probe_receipt_path": result.restore_receipt_path,
                            "alfworld_task_metadata": metadata,
                            "alfworld_facts": list(before.facts),
                            "alfworld_next_facts": list(result.observation.facts),
                            "alfworld_environment_outcome": dict(result.environment_outcome or {}),
                            "alfworld_raw_observation": before.text,
                            "alfworld_raw_next_observation": result.observation.text,
                            "alfworld_admissible_actions": list(before.admissible_actions),
                            "alfworld_next_admissible_actions": list(
                                result.observation.admissible_actions
                            ),
                            "actor_task_description": trajectory.task_description,
                            "actor_protocol_id": self.config.actor_protocol_id,
                            "response_contract_id": (self._actor_protocol.response_contract.id),
                            "requested_response_contract_sha256": (
                                self._actor_protocol.response_contract.descriptor.sha256
                            ),
                            "response_stop_texts": list(
                                self._actor_protocol.response_contract.stop_texts
                            ),
                            "g2po_source_revision": self.config.source_revision,
                            "official_anchor_projection_verified": True,
                            "anchor_tracker_implementation_sha256": (
                                g2po_anchor_tracker_source_sha256()
                            ),
                            "actor_awareness_mode": self.config.actor_awareness_mode,
                            "actor_state_exposure_mode": (self.config.actor_state_exposure_mode),
                            "actor_context_extension_revision": (
                                CONTEXT_EXTENSION_PLACEMENT_REVISION
                                if (
                                    self.config.actor_awareness_mode != "none"
                                    or self.config.actor_state_exposure_mode
                                    != ActorStateExposureMode.NORMAL.value
                                )
                                else "none"
                            ),
                            "actor_awareness": actor_context.awareness,
                            "actor_state_exposure": actor_context.state_exposure,
                            "actor_prompt": actor_context.prompt,
                            "model_response": sample.action_text,
                            "model_io_profile_sha256": (sample.model_io_profile_sha256),
                            "response_contract_sha256": (sample.response_contract_sha256),
                            "normalized_messages_sha256": (sample.normalized_messages_sha256),
                            "chat_template_sha256": sample.chat_template_sha256,
                            "matched_stop_text": sample.matched_stop_text,
                            "finish_reason": sample.finish_reason,
                            "environment_action": projection.environment_action,
                            "is_action_valid": projection.is_action_valid,
                            "invalid_action_reasons": list(projection.invalid_reasons),
                            "sampling_temperature": sample.sampling_temperature,
                            "sampling_seed": sample.rollout_seed,
                            "sampling_seed_base": trajectory.sampling_seed_base,
                            "sampling_seed_schedule": (
                                "base_plus_trajectory_ordinal_times_horizon_plus_step"
                            ),
                            "horizon_bucket": trajectory.task.horizon_bucket,
                            "reference_steps": trajectory.task.reference_steps,
                        },
                    )
                    trajectory.records.append(record)
                    trajectory.history.append(
                        G2POActorHistoryStep(
                            observation=before.text,
                            environment_action=projection.environment_action,
                        )
                    )
                    trajectory.observation = result.observation
                    trajectory.anchor_observation = next_anchor
                    trajectory.goal_vector = next_goal_vector
                if progress_callback is not None:
                    progress_callback(
                        {
                            "step_index": step_index,
                            "active_trajectories": sum(
                                int(not trajectory.done) for trajectory in live
                            ),
                            "completed_trajectories": sum(
                                int(trajectory.done) for trajectory in live
                            ),
                            "successful_trajectories": sum(
                                int(trajectory.won) for trajectory in live
                            ),
                            "total_trajectories": len(live),
                        }
                    )

            records: list[TrainingTrajectoryRecord] = []
            for trajectory in live:
                if not trajectory.records:
                    raise AlfWorldSuiteError("ALFWorld branch emitted no transitions")
                if not trajectory.done:
                    raise AlfWorldSuiteError("ALFWorld branch did not reach the configured horizon")
                length = len(trajectory.records)
                reason = trajectory.terminal_reason
                trajectory_physical_calls = trajectory.records[-1].extra.get(
                    "alfworld_physical_call_cumulative",
                    {},
                )
                if not isinstance(trajectory_physical_calls, Mapping):
                    raise AlfWorldSuiteError("ALFWorld physical-call total must be a mapping")
                for index, record in enumerate(trajectory.records):
                    records.append(
                        replace(
                            record,
                            trajectory_length=length,
                            terminal=index == length - 1,
                            success=trajectory.won,
                            terminal_reason=reason,
                            outcome_evidence=_finalize_outcome_evidence(
                                record.outcome_evidence,
                                trajectory_won=trajectory.won,
                            ),
                            extra={
                                **record.extra,
                                "alfworld_trajectory_physical_call_totals": dict(
                                    trajectory_physical_calls
                                ),
                            },
                        )
                    )

            sample_keys = {(record.traj_uid, record.step_idx) for record in records}
            policy_samples = tuple(
                sample
                for sample in policy.sampling_events_since(sampling_cursor)
                if (sample.traj_uid, sample.step_idx) in sample_keys
            )
            return AgenticRolloutBatch(
                records=tuple(records),
                policy_samples=policy_samples,
                task_manifest_sha256=self.manifest_sha256,
                policy_checkpoint_sha256=checkpoint_sha256,
            ).validate()
        finally:
            for trajectory in live:
                trajectory.branch.close()

    def _actor_context(
        self,
        trajectory: _LiveTrajectory,
        *,
        remaining_budget: int,
    ) -> _ActorStepContext:
        context_extension = None
        awareness = None
        state_exposure = None
        if self.config.actor_awareness_mode != "none":
            projection = project_actor_awareness(
                trajectory.goal_graph,
                trajectory.goal_vector,
                remaining_budget=remaining_budget,
            )
            awareness = asdict(projection)
            context_extension = f"<goal_awareness>{projection.to_json()}</goal_awareness>"
        elif self.config.actor_state_exposure_mode != ActorStateExposureMode.NORMAL.value:
            context_extension, state_exposure = render_actor_state_exposure_extension(
                trajectory.goal_graph,
                trajectory.goal_vector,
                remaining_budget=remaining_budget,
                mode=self.config.actor_state_exposure_mode,
            )
        return _ActorStepContext(
            prompt=self._actor_protocol.render_prompt(
                task_description=trajectory.task_description,
                current_observation=trajectory.observation.text,
                admissible_actions=trajectory.observation.admissible_actions,
                history=trajectory.history,
                context_extension=context_extension,
            ),
            awareness=awareness,
            state_exposure=state_exposure,
        )

    def _load_evaluation_schedule(
        self,
        path: Path,
    ) -> tuple[_EvaluationEpisode, ...]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AlfWorldSuiteError(f"cannot load evaluation schedule: {error}") from error
        if not isinstance(raw, Mapping):
            raise AlfWorldSuiteError("evaluation schedule must be a mapping")
        expected = {
            "schema",
            "source_revision",
            "split",
            "seed_start",
            "episode_count",
            "selection_contract",
            "episodes",
        }
        if set(raw) != expected:
            raise AlfWorldSuiteError("evaluation schedule fields differ from schema")
        if raw.get("schema") != "lh_g2po_evaluation_schedule/v1":
            raise AlfWorldSuiteError("evaluation schedule schema is unsupported")
        if raw.get("source_revision") != self.config.source_revision:
            raise AlfWorldSuiteError("evaluation schedule binds a different source revision")
        if raw.get("split") != self.split:
            raise AlfWorldSuiteError("evaluation schedule split differs from manifest")
        selection_contract = raw.get("selection_contract")
        if not isinstance(selection_contract, Mapping) or set(selection_contract) != {
            "eval_dataset",
            "worker_seed_formula",
            "worker_count",
            "source_files",
            "filesystem_walk_order_bound",
        }:
            raise AlfWorldSuiteError("evaluation schedule selection contract is invalid")
        if (
            selection_contract.get("eval_dataset") != "eval_in_distribution"
            or selection_contract.get("worker_seed_formula") != "1000 + episode_index"
            or selection_contract.get("worker_count") != raw.get("episode_count")
            or selection_contract.get("filesystem_walk_order_bound") is not True
        ):
            raise AlfWorldSuiteError(
                "evaluation schedule does not reproduce official worker selection"
            )
        source_files = selection_contract.get("source_files")
        if (
            not isinstance(source_files, Mapping)
            or not source_files
            or any(
                len(str(value)) != 64
                or any(character not in "0123456789abcdef" for character in str(value))
                for value in source_files.values()
            )
        ):
            raise AlfWorldSuiteError("evaluation schedule source file hashes are invalid")
        episodes = raw.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != raw.get("episode_count"):
            raise AlfWorldSuiteError("evaluation schedule episode_count is invalid")
        task_by_key = {(task.task_id, task.game_path): task for task in self.manifest.tasks}
        out: list[_EvaluationEpisode] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(episodes):
            if not isinstance(item, Mapping) or set(item) != {
                "episode_index",
                "episode_id",
                "environment_seed",
                "task_id",
                "game_path",
            }:
                raise AlfWorldSuiteError(
                    f"evaluation schedule episode {index} fields differ from schema"
                )
            if item.get("episode_index") != index:
                raise AlfWorldSuiteError("evaluation schedule indices are not contiguous")
            episode_id = str(item.get("episode_id", ""))
            if not episode_id or episode_id in seen_ids:
                raise AlfWorldSuiteError("evaluation schedule episode IDs are not unique")
            seed = item.get("environment_seed")
            seed_start = raw.get("seed_start")
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or isinstance(seed_start, bool)
                or not isinstance(seed_start, int)
                or seed != seed_start + index
            ):
                raise AlfWorldSuiteError(
                    "evaluation schedule seeds do not follow seed_start + episode_index"
                )
            task_key = (str(item.get("task_id", "")), str(item.get("game_path", "")))
            task = task_by_key.get(task_key)
            if task is None:
                raise AlfWorldSuiteError(
                    "evaluation schedule references a task outside its manifest"
                )
            seen_ids.add(episode_id)
            out.append(
                _EvaluationEpisode(
                    episode_id=episode_id,
                    environment_seed=seed,
                    task=task,
                )
            )
        if not out:
            raise AlfWorldSuiteError("evaluation schedule seed_start is invalid")
        return tuple(out)

    def _task_metadata(self, task: AlfworldTask) -> dict[str, Any]:
        trajectory = self._load_trajectory(task)
        params = trajectory.get("pddl_params")
        if not isinstance(params, Mapping):
            raise AlfWorldSuiteError("ALFWorld traj_data lacks pddl_params")
        return {
            "task_type": str(trajectory.get("task_type", "")),
            "pddl_params": {
                "object_target": params.get("object_target", ""),
                "parent_target": params.get("parent_target", ""),
                "toggle_target": params.get("toggle_target", ""),
            },
        }

    def _task_instruction(self, task: AlfworldTask) -> str:
        trajectory = self._load_trajectory(task)
        annotations = trajectory.get("turk_annotations")
        anns = annotations.get("anns") if isinstance(annotations, Mapping) else None
        if not isinstance(anns, list) or not anns or not isinstance(anns[0], Mapping):
            raise AlfWorldSuiteError("ALFWorld traj_data lacks task annotation")
        instruction = str(anns[0].get("task_desc", "")).strip()
        if not instruction:
            raise AlfWorldSuiteError("ALFWorld task annotation is empty")
        return instruction

    def _load_trajectory(self, task: AlfworldTask) -> Mapping[str, Any]:
        path = self._resolve_task_path(task.trajectory_path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AlfWorldSuiteError(f"cannot load ALFWorld traj_data: {error}") from error
        if not isinstance(value, Mapping):
            raise AlfWorldSuiteError("ALFWorld traj_data must be a mapping")
        if str(value.get("task_type", "")) != task.task_type:
            raise AlfWorldSuiteError("ALFWorld manifest task_type differs from traj_data")
        return value

    def _resolve_task_path(self, relative: str) -> Path:
        path = (self._data_root / relative).resolve()
        try:
            path.relative_to(self._data_root)
        except ValueError as error:
            raise AlfWorldSuiteError("ALFWorld manifest path escapes data_root") from error
        if not path.is_file():
            raise AlfWorldSuiteError(f"ALFWorld manifest artifact is missing: {path}")
        return path


def _patch_handcoded_expert_fact_names() -> None:
    """Guard the pinned ALFWorld handcoded expert against non-string fact names.

    TextWorld 1.7 can materialize a game-state Fact whose ``name`` attribute is
    not a plain ``str`` (observed as ``attribute name must be string, not 'type'``
    inside ``HandCodedTWAgent.get_facts``). The expert trace is only collected
    for provenance (``base.use_expert = True``); it is never projected into the
    actor or consumed as supervision. Coerce names defensively without changing
    any downstream semantics.
    """

    try:
        from alfworld.agents.expert import handcoded_expert
    except ImportError:  # pragma: no cover - env not installed in this runtime.
        return
    current = handcoded_expert.BasePolicy.get_facts

    def get_facts_safe(self, game_state: Any) -> list[str]:
        facts: list[str] = []
        for fact in game_state.get("facts") or []:
            name = getattr(fact, "name", None)
            names = getattr(fact, "names", None) or ()
            facts.append(
                f"{name} " + " ".join(str(n).strip() for n in names)
            )
        return facts

    if current is not get_facts_safe:
        handcoded_expert.BasePolicy.get_facts = get_facts_safe


class OfficialAlfWorldBranchFactory:
    """Create fixed-game scalar branches from the pinned official ALFWorld fork."""

    def __init__(self, *, source_root: str | Path, data_root: str | Path) -> None:
        self.source_root = Path(source_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.package_root = (
            self.source_root / "agent_system" / "environments" / "env_package" / "alfworld"
        )
        self.config_path = self.package_root.parent / "alfworld" / "configs" / "config_tw.yaml"
        if not self.package_root.is_dir() or not self.config_path.is_file():
            raise AlfWorldSuiteError("pinned ALFWorld source package/config is missing")

    def __call__(self, task: AlfworldTask, seed: int, artifacts_dir: Path) -> AlfWorldBranch:
        return _OfficialAlfWorldBranch(
            task=task,
            seed=seed,
            artifacts_dir=artifacts_dir,
            package_root=self.package_root,
            config_path=self.config_path,
            data_root=self.data_root,
        )


class _ScalarWorker:
    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self) -> Any:
        return self.env.reset()

    def step(self, action: str) -> Any:
        return self.env.step([action])


class _OfficialAlfWorldBranch:
    def __init__(
        self,
        *,
        task: AlfworldTask,
        seed: int,
        artifacts_dir: Path,
        package_root: Path,
        config_path: Path,
        data_root: Path,
    ) -> None:
        source_root = package_root.parents[3]
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        try:
            import alfworld
            import yaml
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        except ImportError as error:  # pragma: no cover - remote dependency gate.
            raise AlfWorldSuiteError(
                "official ALFWorld branch requires yaml, textworld, gym, and alfworld"
            ) from error
        _require_module_from(alfworld, package_root, "alfworld")
        _patch_handcoded_expert_fact_names()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        task_root = (data_root / task.game_path).resolve().parent
        config["dataset"]["data_path"] = str(task_root)
        config["dataset"]["eval_id_data_path"] = str(task_root)
        config["dataset"]["eval_ood_data_path"] = str(task_root)
        config["logic"]["domain"] = str(data_root / "logic" / "alfred.pddl")
        config["logic"]["grammar"] = str(data_root / "logic" / "alfred.twl2")
        config["env"]["domain_randomization"] = False
        config["dagger"]["training"]["max_nb_steps_per_episode"] = 10_000
        # Import the pinned scalar class directly.  The source helper's absolute
        # package import executes unrelated Ray/vision rollout registration,
        # which is not part of this TextWorld branch and must not widen the
        # calibration runtime dependency surface.
        base = AlfredTWEnv(config, train_eval="train")
        base.use_expert = True  # Requests facts; expert_plan is never projected or consumed.
        gamefile = (data_root / task.game_path).resolve()
        base.game_files = [str(gamefile)]
        base.num_games = 1
        env = base.init_env(batch_size=1)
        env.seed(seed)
        self._worker = _ScalarWorker(env)
        self._controller = AlfWorldReplayStateController(
            self._worker,
            output_dir=artifacts_dir,
        )
        self._current: AlfWorldBranchObservation | None = None
        self._training_supervision_allowed = task.split == "train"
        self._training_expert_plan: tuple[str, ...] | None = None

    def reset(self) -> AlfWorldBranchObservation:
        raw = self._controller.reset()
        # The ordinary runtime never depends on expert-plan availability.
        # Missing supervision is surfaced only if the quarantined train-only
        # accessor is called by the warm-start builder.
        try:
            self._training_expert_plan = _extract_training_expert_plan(raw)
        except AlfWorldSuiteError:
            self._training_expert_plan = None
        self._current = _branch_observation(raw)
        return self._current

    def training_reference_plan(self) -> tuple[str, ...]:
        """Return train-only TextWorld commands, quarantined from rollout data."""

        if not self._training_supervision_allowed:
            raise AlfWorldSuiteError("training reference plan is forbidden outside the train split")
        if self._training_expert_plan is None:
            raise AlfWorldSuiteError(
                "training reference plan is unavailable; reset first and require expert data"
            )
        return self._training_expert_plan

    def step(self, action: str) -> AlfWorldBranchStep:
        verified = self._controller.step(
            action,
            project_outcome=project_scalar_textworld_outcome,
        )
        try:
            self._training_expert_plan = _extract_training_expert_plan(verified.raw_outcome)
        except AlfWorldSuiteError:
            self._training_expert_plan = None
        observation = _branch_observation(verified.raw_outcome)
        self._current = observation
        won = bool(verified.outcome["won"])
        if verified.post_snapshot_artifact is None:
            raise AlfWorldSuiteError("verified step did not materialize its post-state snapshot")
        return AlfWorldBranchStep(
            observation=observation,
            reward=10.0 if won else 0.0,
            done=bool(verified.outcome["done"]),
            won=won,
            state_snapshot_sha256=verified.snapshot_artifact.state_snapshot_sha256,
            state_snapshot_artifact_path=verified.snapshot_artifact.payload_path,
            restore_receipt_sha256=verified.restore_receipt_sha256,
            restore_receipt_path=verified.restore_receipt_path,
            replay_outcome=dict(verified.outcome),
            environment_outcome=dict(verified.outcome),
            post_state_snapshot_sha256=(verified.post_snapshot_artifact.state_snapshot_sha256),
            post_state_snapshot_artifact_path=(verified.post_snapshot_artifact.payload_path),
            physical_call_delta=(
                verified.physical_call_delta.to_dict()
                if verified.physical_call_delta is not None
                else None
            ),
            physical_call_cumulative=(
                verified.physical_call_cumulative.to_dict()
                if verified.physical_call_cumulative is not None
                else None
            ),
        )

    def step_snapshot_only(self, action: str) -> AlfWorldBranchStep:
        """Advance once with a snapshot, without restore/replay verification."""

        observed = self._controller.step_snapshot_only(
            action,
            project_outcome=project_scalar_textworld_outcome,
        )
        try:
            self._training_expert_plan = _extract_training_expert_plan(observed.raw_outcome)
        except AlfWorldSuiteError:
            self._training_expert_plan = None
        observation = _branch_observation(observed.raw_outcome)
        self._current = observation
        won = bool(observed.outcome["won"])
        if observed.post_snapshot_artifact is None:
            raise AlfWorldSuiteError("snapshot-only step did not materialize its post-state")
        return AlfWorldBranchStep(
            observation=observation,
            reward=10.0 if won else 0.0,
            done=bool(observed.outcome["done"]),
            won=won,
            state_snapshot_sha256=(observed.snapshot_artifact.state_snapshot_sha256),
            state_snapshot_artifact_path=(observed.snapshot_artifact.payload_path),
            restore_receipt_sha256="",
            restore_receipt_path="",
            replay_outcome=None,
            environment_outcome=dict(observed.outcome),
            post_state_snapshot_sha256=(observed.post_snapshot_artifact.state_snapshot_sha256),
            post_state_snapshot_artifact_path=(observed.post_snapshot_artifact.payload_path),
            physical_call_delta=(
                observed.physical_call_delta.to_dict()
                if observed.physical_call_delta is not None
                else None
            ),
            physical_call_cumulative=(
                observed.physical_call_cumulative.to_dict()
                if observed.physical_call_cumulative is not None
                else None
            ),
        )

    def physical_call_counters(self) -> Mapping[str, Any]:
        return self._controller.physical_call_counters().to_dict()

    def capture_checkpoint(self) -> AlfWorldBranchCheckpoint:
        if self._current is None:
            raise AlfWorldSuiteError("ALFWorld branch requires reset before checkpoint capture")
        return AlfWorldBranchCheckpoint(
            observation=self._current,
            snapshot=self._controller.snapshot(),
        )

    def restore_checkpoint(
        self,
        checkpoint: AlfWorldBranchCheckpoint,
    ) -> AlfWorldBranchObservation:
        self._controller.restore(checkpoint.snapshot)
        self._current = checkpoint.observation
        return self._current

    def step_unverified(self, action: str) -> AlfWorldBranchStep:
        """Advance once for success-rate evaluation without restore/replay evidence."""

        raw, outcome, call_delta, call_cumulative = self._controller.step_unverified(
            action,
            project_outcome=project_scalar_textworld_outcome,
        )
        observation = _branch_observation(raw)
        self._current = observation
        won = bool(outcome["won"])
        return AlfWorldBranchStep(
            observation=observation,
            reward=10.0 if won else 0.0,
            done=bool(outcome["done"]),
            won=won,
            state_snapshot_sha256="",
            state_snapshot_artifact_path="",
            restore_receipt_sha256="",
            restore_receipt_path="",
            replay_outcome=dict(outcome),
            environment_outcome=dict(outcome),
            physical_call_delta=call_delta.to_dict(),
            physical_call_cumulative=call_cumulative.to_dict(),
        )

    def close(self) -> None:
        close = getattr(self._worker.env, "close", None)
        if callable(close):
            close()


def _branch_observation(raw: Any) -> AlfWorldBranchObservation:
    if not isinstance(raw, tuple):
        raise AlfWorldSuiteError("ALFWorld branch result must be a tuple")
    if len(raw) == 2:
        observations, infos = raw
    elif len(raw) == 4:
        observations, _rewards, _dones, infos = raw
    else:
        raise AlfWorldSuiteError("unsupported ALFWorld branch tuple shape")
    text = _scalar(observations, "observation")
    if not isinstance(infos, Mapping):
        raise AlfWorldSuiteError("ALFWorld info must be a mapping")
    gamefile = str(_scalar(infos.get("extra.gamefile"), "extra.gamefile"))
    admissible = _scalar(infos.get("admissible_commands"), "admissible_commands")
    facts = _scalar(infos.get("facts"), "facts")
    if not isinstance(admissible, (list, tuple)) or not admissible:
        raise AlfWorldSuiteError("ALFWorld branch exposed no admissible actions")
    if not isinstance(facts, (list, tuple)):
        raise AlfWorldSuiteError("ALFWorld branch exposed no current facts")
    normalized_facts = tuple(
        {
            "name": str(getattr(fact, "name", "")),
            "args": [str(value) for value in getattr(fact, "names", ())],
        }
        if not isinstance(fact, Mapping)
        else {"name": str(fact.get("name", "")), "args": list(fact.get("args") or ())}
        for fact in facts
    )
    return AlfWorldBranchObservation(
        text=str(text),
        facts=normalized_facts,
        admissible_actions=tuple(str(action) for action in admissible if str(action).strip()),
        gamefile=gamefile,
    )


def _extract_training_expert_plan(raw: Any) -> tuple[str, ...]:
    """Extract source-provided semantic commands without projecting them publicly."""

    if not isinstance(raw, tuple) or len(raw) not in {2, 4}:
        raise AlfWorldSuiteError("unsupported ALFWorld branch tuple shape")
    infos = raw[1] if len(raw) == 2 else raw[3]
    if not isinstance(infos, Mapping):
        raise AlfWorldSuiteError("ALFWorld info must be a mapping")
    plan = _scalar(infos.get("extra.expert_plan"), "extra.expert_plan")
    if not isinstance(plan, (list, tuple)) or not plan:
        raise AlfWorldSuiteError("ALFWorld train task exposed no expert plan")
    normalized = tuple(str(action).strip().lower() for action in plan if str(action).strip())
    if not normalized:
        raise AlfWorldSuiteError("ALFWorld expert plan contains no semantic actions")
    return normalized


def _scalar(value: Any, field: str) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    raise AlfWorldSuiteError(f"scalar ALFWorld branch requires one {field}")


def _require_module_from(module: Any, root: Path, name: str) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        locations = tuple(Path(value).resolve() for value in getattr(module, "__path__", ()))
        if not locations or any(not location.is_relative_to(root) for location in locations):
            raise AlfWorldSuiteError(f"imported {name} namespace is outside the pinned source root")
        return
    try:
        Path(module_file).resolve().relative_to(root)
    except ValueError as error:
        raise AlfWorldSuiteError(
            f"imported {name} module is outside the pinned source root"
        ) from error
