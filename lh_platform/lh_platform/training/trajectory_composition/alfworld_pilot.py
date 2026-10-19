"""Controlled ALFWorld mechanism pilot for replay-certified fragment composition.

This module deliberately uses train-split expert plans to construct a mechanism
check.  It does not estimate autonomous-agent performance and never emits an
on-policy training sample.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from lh_platform.alfworld_state import (
    AlfWorldReplayStateController,
    AlfWorldStateArtifact,
    project_scalar_textworld_outcome,
    write_alfworld_state_artifact,
)
from lh_platform.studies.runtime import AlfworldTask
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.alfworld_semantic_verifier import (
    ALFWORLD_SEMANTIC_VERIFIER_ID,
    audit_alfworld_snapshot_composition,
)
from lh_platform.training.subproblems.base import GoalGraph, GoalVector
from lh_platform.training.trajectory_composition.contracts import (
    CompositionLevel,
    FragmentRecordV1,
    FragmentSourceKind,
    ReplayEvidenceV1,
    canonical_sha256,
    certify_composition,
)
from lh_platform.training.trajectory_composition.planner import (
    CompositionPlannerConfig,
    search_composition_plan,
)

ALFWORLD_CVFC_PILOT_SCHEMA = "lh_alfworld_cvfc_controlled_task_pilot/v1"
ALFWORLD_INSTALLED_RUNTIME_SCHEMA = "lh_alfworld_installed_runtime/v1"
ALFWORLD_EXPERT_DERIVATION_ID = "look_at_split_acquire_vs_light_without_acquire/v1"
ALFWORLD_CVFC_REQUESTED_EXTRAS = ("gamefile", "expert_plan")


class AlfWorldCompositionPilotError(RuntimeError):
    """Raised when the controlled environment pilot cannot produce evidence."""


@dataclass(frozen=True)
class InstalledAlfWorldRuntimeIdentity:
    distribution: str
    version: str
    module_path: str
    module_sha256: str
    environment_module_path: str
    environment_module_sha256: str
    textworld_version: str
    adapter_config_sha256: str
    schema: str = ALFWORLD_INSTALLED_RUNTIME_SCHEMA

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class PilotSequenceResult:
    run_id: str
    requested_actions: tuple[str, ...]
    executed_actions: tuple[str, ...]
    initial_state_snapshot_sha256: str
    final_state_snapshot_sha256: str
    initial_goal_ids: tuple[str, ...]
    final_goal_ids: tuple[str, ...]
    parent_contract_verified: bool
    environment_won: bool
    environment_done: bool
    replay_completed: bool
    transition_restore_receipt_sha256s: tuple[str, ...]
    transition_rows: tuple[dict[str, Any], ...]
    final_parent_verifier_receipt: dict[str, Any]
    final_parent_verifier_receipt_sha256: str
    effect_evidence_sha256: str
    logical_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requested_actions": list(self.requested_actions),
            "executed_actions": list(self.executed_actions),
            "initial_state_snapshot_sha256": self.initial_state_snapshot_sha256,
            "final_state_snapshot_sha256": self.final_state_snapshot_sha256,
            "initial_goal_ids": list(self.initial_goal_ids),
            "final_goal_ids": list(self.final_goal_ids),
            "parent_contract_verified": self.parent_contract_verified,
            "environment_won": self.environment_won,
            "environment_done": self.environment_done,
            "replay_completed": self.replay_completed,
            "transition_restore_receipt_sha256s": list(self.transition_restore_receipt_sha256s),
            "transition_rows": list(self.transition_rows),
            "final_parent_verifier_receipt": self.final_parent_verifier_receipt,
            "final_parent_verifier_receipt_sha256": (self.final_parent_verifier_receipt_sha256),
            "effect_evidence_sha256": self.effect_evidence_sha256,
            "logical_action_count": self.logical_action_count,
        }


class _ScalarWorker:
    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self) -> Any:
        return self.env.reset()

    def step(self, action: str) -> Any:
        return self.env.step([action])


class _InstalledPilotBranch:
    def __init__(
        self,
        *,
        task: AlfworldTask,
        data_root: Path,
        seed: int,
        output_dir: Path,
        expected_alfworld_version: str,
    ) -> None:
        runtime = installed_alfworld_runtime_identity()
        if runtime.version != expected_alfworld_version:
            raise AlfWorldCompositionPilotError(
                "installed ALFWorld version differs from authorization: "
                f"{runtime.version!r} != {expected_alfworld_version!r}"
            )
        try:
            import textworld
            from alfworld.agents.environment.alfred_tw_env import (
                AlfredDemangler,
                AlfredExpert,
                AlfredInfos,
            )
        except ImportError as error:  # pragma: no cover - remote dependency gate.
            raise AlfWorldCompositionPilotError("ALFWorld runtime is unavailable") from error
        task_root = (data_root / task.game_path).resolve().parent
        gamefile = (data_root / task.game_path).resolve()
        if not gamefile.is_file() or not task_root.is_relative_to(data_root):
            raise AlfWorldCompositionPilotError("pilot task game is outside the data root")
        request_infos = textworld.EnvInfos(
            won=True,
            facts=True,
            admissible_commands=True,
            extras=list(ALFWORLD_CVFC_REQUESTED_EXTRAS),
        )
        wrappers = [
            AlfredDemangler(shuffle=False),
            AlfredInfos,
            AlfredExpert("handcoded"),
        ]
        env_id = textworld.gym.register_games(
            [str(gamefile)],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=10_000,
            wrappers=wrappers,
        )
        env = textworld.gym.make(env_id)
        env.seed(seed)
        self.worker = _ScalarWorker(env)
        self.controller = AlfWorldReplayStateController(
            self.worker,
            output_dir=output_dir,
        )
        self.output_dir = output_dir
        self.gamefile = gamefile
        self._last_raw: Any = None

    def reset(self) -> tuple[dict[str, Any], tuple[str, ...], AlfWorldStateArtifact]:
        raw = self.controller.reset()
        self._last_raw = raw
        observation = _project_reset_or_step(raw)
        if Path(observation["gamefile"]).resolve() != self.gamefile:
            raise AlfWorldCompositionPilotError("ALFWorld reset selected a different game")
        expert_plan = _extract_expert_plan(raw)
        initial = write_alfworld_state_artifact(
            self.controller.snapshot(),
            output_dir=self.output_dir,
        )
        return observation, expert_plan, initial

    def step(self, action: str) -> tuple[dict[str, Any], dict[str, Any]]:
        verified = self.controller.step(
            action,
            project_outcome=project_scalar_textworld_outcome,
        )
        self._last_raw = verified.raw_outcome
        observation = _project_reset_or_step(verified.raw_outcome)
        post = verified.post_snapshot_artifact
        if post is None:
            raise AlfWorldCompositionPilotError("verified action lacks post-state artifact")
        evidence = {
            "pre_state_snapshot_sha256": (verified.snapshot_artifact.state_snapshot_sha256),
            "pre_state_snapshot_artifact_path": (verified.snapshot_artifact.payload_path),
            "post_state_snapshot_sha256": post.state_snapshot_sha256,
            "post_state_snapshot_artifact_path": post.payload_path,
            "restore_receipt_sha256": verified.restore_receipt_sha256,
            "restore_receipt_path": verified.restore_receipt_path,
            "restore_verified": verified.restore_receipt.restore_verified,
            "physical_call_delta": (
                verified.physical_call_delta.to_dict()
                if verified.physical_call_delta is not None
                else {}
            ),
            "physical_call_cumulative": (
                verified.physical_call_cumulative.to_dict()
                if verified.physical_call_cumulative is not None
                else {}
            ),
            "reward": float(verified.outcome["reward"]),
            "done": bool(verified.outcome["done"]),
            "won": bool(verified.outcome["won"]),
        }
        return observation, evidence

    def step_snapshot_only(self, action: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute one source action with pre/post snapshots and no replay proof."""

        observed = self.controller.step_snapshot_only(
            action,
            project_outcome=project_scalar_textworld_outcome,
        )
        self._last_raw = observed.raw_outcome
        observation = _project_reset_or_step(observed.raw_outcome)
        post = observed.post_snapshot_artifact
        if post is None:
            raise AlfWorldCompositionPilotError("snapshot-only action lacks a post-state artifact")
        evidence = {
            "pre_state_snapshot_sha256": (observed.snapshot_artifact.state_snapshot_sha256),
            "pre_state_snapshot_artifact_path": (observed.snapshot_artifact.payload_path),
            "post_state_snapshot_sha256": post.state_snapshot_sha256,
            "post_state_snapshot_artifact_path": post.payload_path,
            "environment_outcome": dict(observed.outcome),
            "physical_call_delta": (
                observed.physical_call_delta.to_dict()
                if observed.physical_call_delta is not None
                else {}
            ),
            "physical_call_cumulative": (
                observed.physical_call_cumulative.to_dict()
                if observed.physical_call_cumulative is not None
                else {}
            ),
        }
        return observation, evidence

    def current_expert_plan(self) -> tuple[str, ...]:
        if self._last_raw is None:
            raise AlfWorldCompositionPilotError("expert plan requested before reset")
        return _extract_expert_plan(self._last_raw)

    def close(self) -> None:
        close = getattr(self.worker.env, "close", None)
        if callable(close):
            close()


def installed_alfworld_runtime_identity() -> InstalledAlfWorldRuntimeIdentity:
    """Attest the installed package used by this non-reproduction pilot."""

    try:
        import alfworld
    except ImportError as error:  # pragma: no cover - remote dependency gate.
        raise AlfWorldCompositionPilotError("ALFWorld package is unavailable") from error
    from alfworld.agents.environment import alfred_tw_env

    module_path = Path(alfworld.__file__ or "").resolve()
    environment_module_path = Path(alfred_tw_env.__file__ or "").resolve()
    if not module_path.is_file() or not environment_module_path.is_file():
        raise AlfWorldCompositionPilotError("ALFWorld module path is unavailable")
    return InstalledAlfWorldRuntimeIdentity(
        distribution="alfworld",
        version=metadata.version("alfworld"),
        module_path=str(module_path),
        module_sha256=_sha256_file(module_path),
        environment_module_path=str(environment_module_path),
        environment_module_sha256=_sha256_file(environment_module_path),
        textworld_version=metadata.version("textworld"),
        adapter_config_sha256=canonical_sha256(
            {
                "batch_size": 1,
                "asynchronous": False,
                "max_episode_steps": 10_000,
                "domain_randomization": False,
                "expert_type": "handcoded",
                "requested_infos": [
                    "won",
                    "facts",
                    "admissible_commands",
                    *(f"extra.{name}" for name in ALFWORLD_CVFC_REQUESTED_EXTRAS),
                ],
            }
        ),
    )


def run_controlled_alfworld_task_pilot(
    *,
    task: AlfworldTask,
    data_root: str | Path,
    output_dir: str | Path,
    seed: int,
    expected_alfworld_version: str,
    max_logical_actions: int,
) -> dict[str, Any]:
    """Run one fixed look-at task through source, composition, and sham arms."""

    if task.split != "train" or task.task_type != "look_at_obj_in_light":
        raise AlfWorldCompositionPilotError(
            "controlled expert pilot requires a train look_at_obj_in_light task"
        )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = Path(data_root).resolve()
    metadata_payload = _load_task_metadata(task, data)
    compiler = AlfWorldSubproblemCompiler()
    graph = compiler.compile_task(metadata_payload)
    task_instance_sha256 = canonical_sha256(
        {
            "task_id": task.task_id,
            "group_uid": task.group_uid,
            "game_path": task.game_path,
            "game_sha256": _sha256_file(data / task.game_path),
            "trajectory_path": task.trajectory_path,
            "trajectory_sha256": _sha256_file(data / task.trajectory_path),
        }
    )
    parent_contract_sha256 = canonical_sha256(
        {
            "graph_sha256": graph.sha256,
            "root_id": graph.root_id,
            "verifier_id": ALFWORLD_SEMANTIC_VERIFIER_ID,
            "task_metadata": metadata_payload,
        }
    )
    runtime = installed_alfworld_runtime_identity()
    environment_identity_sha256 = canonical_sha256(
        {
            "runtime": asdict(runtime),
            "task_instance_sha256": task_instance_sha256,
        }
    )

    expert = _run_sequence(
        run_id="expert_sanity",
        task=task,
        data_root=data,
        output_dir=root / "expert_trace",
        seed=seed,
        actions=None,
        max_dynamic_actions=max_logical_actions,
        expected_alfworld_version=expected_alfworld_version,
        metadata_payload=metadata_payload,
        graph=graph,
        compiler=compiler,
    )
    acquire_indices = tuple(
        index
        for index, row in enumerate(expert.transition_rows)
        if "acquired" in row["newly_satisfied_goal_ids"]
    )
    if len(acquire_indices) != 1:
        failure = _task_failure_receipt(
            task=task,
            seed=seed,
            runtime=runtime,
            expert=expert,
            reason="expert_trace_does_not_have_one_acquisition_boundary",
        )
        _write_json(root / "task_receipt.json", failure)
        return failure
    acquire_index = acquire_indices[0]
    source_a_actions = expert.executed_actions[: acquire_index + 1]
    source_b_discovery_actions = tuple(
        action for index, action in enumerate(expert.executed_actions) if index != acquire_index
    )

    source_a = _run_sequence(
        run_id="source_a_acquire",
        task=task,
        data_root=data,
        output_dir=root / "source_a",
        seed=seed,
        actions=source_a_actions,
        expected_alfworld_version=expected_alfworld_version,
        metadata_payload=metadata_payload,
        graph=graph,
        compiler=compiler,
    )
    source_b_discovery = _run_sequence(
        run_id="source_b_discovery_without_acquire",
        task=task,
        data_root=data,
        output_dir=root / "source_b_discovery",
        seed=seed,
        actions=source_b_discovery_actions,
        expected_alfworld_version=expected_alfworld_version,
        metadata_payload=metadata_payload,
        graph=graph,
        compiler=compiler,
    )
    light_indices = tuple(
        index
        for index, row in enumerate(source_b_discovery.transition_rows)
        if "light_on" in row["newly_satisfied_goal_ids"]
    )
    if len(light_indices) != 1:
        failure = _task_failure_receipt(
            task=task,
            seed=seed,
            runtime=runtime,
            expert=expert,
            source_a=source_a,
            source_b_discovery=source_b_discovery,
            reason="acquisition_removed_trace_does_not_have_one_light_boundary",
        )
        _write_json(root / "task_receipt.json", failure)
        return failure
    light_index = light_indices[0]
    source_b_actions = source_b_discovery.executed_actions[: light_index + 1]
    source_b = _run_sequence(
        run_id="source_b_light_without_acquire",
        task=task,
        data_root=data,
        output_dir=root / "source_b",
        seed=seed,
        actions=source_b_actions,
        expected_alfworld_version=expected_alfworld_version,
        metadata_payload=metadata_payload,
        graph=graph,
        compiler=compiler,
    )

    total_source_actions = len(source_a.executed_actions) + len(source_b.executed_actions)
    if total_source_actions > max_logical_actions:
        raise AlfWorldCompositionPilotError("derived source fragments exceed action budget")
    target_goal_ids = _parent_target_goal_ids(graph)
    derivation_sha256 = canonical_sha256(
        {
            "derivation_id": ALFWORLD_EXPERT_DERIVATION_ID,
            "expert_plan": list(expert.executed_actions),
            "acquire_index": acquire_index,
            "light_index_after_acquire_removal": light_index,
        }
    )
    policy_sha256 = canonical_sha256(
        {
            "runtime_identity_sha256": runtime.sha256,
            "derivation_sha256": derivation_sha256,
        }
    )
    fragments: list[FragmentRecordV1] = []
    fragment_errors: list[str] = []
    try:
        fragments.append(
            _fragment_from_sequence(
                result=source_a,
                task_instance_sha256=task_instance_sha256,
                parent_contract_sha256=parent_contract_sha256,
                source_traj_uid=f"{task.task_id}:source-a",
                required_goal_ids=_requirements_for_effects(
                    graph, set(source_a.final_goal_ids) - set(source_a.initial_goal_ids)
                ),
                source_policy_sha256=policy_sha256,
                source_action_provenance_sha256=derivation_sha256,
                graph=graph,
            )
        )
    except (ValueError, AlfWorldCompositionPilotError) as error:
        fragment_errors.append(f"source_a:{error}")
    try:
        fragments.append(
            _fragment_from_sequence(
                result=source_b,
                task_instance_sha256=task_instance_sha256,
                parent_contract_sha256=parent_contract_sha256,
                source_traj_uid=f"{task.task_id}:source-b",
                required_goal_ids=_requirements_for_effects(
                    graph, set(source_b.final_goal_ids) - set(source_b.initial_goal_ids)
                ),
                source_policy_sha256=policy_sha256,
                source_action_provenance_sha256=derivation_sha256,
                graph=graph,
            )
        )
    except (ValueError, AlfWorldCompositionPilotError) as error:
        fragment_errors.append(f"source_b:{error}")

    search = search_composition_plan(
        tuple(fragments),
        task_instance_sha256=task_instance_sha256,
        parent_contract_sha256=parent_contract_sha256,
        initial_state_snapshot_sha256=source_a.initial_state_snapshot_sha256,
        initial_goal_ids=source_a.initial_goal_ids,
        target_goal_ids=target_goal_ids,
        config=CompositionPlannerConfig(
            composition_level=CompositionLevel.E1_VERIFIED_EFFECT,
            max_action_cost=max_logical_actions,
            max_fragments=2,
            min_source_trajectories=2,
            require_failed_parent_sources=True,
        ),
    )
    composition: PilotSequenceResult | None = None
    composition_receipt = None
    if search.plan is not None:
        composition = _run_sequence(
            run_id="composition_full_parent_replay",
            task=task,
            data_root=data,
            output_dir=root / "composition",
            seed=seed,
            actions=search.plan.canonical_actions,
            expected_alfworld_version=expected_alfworld_version,
            metadata_payload=metadata_payload,
            graph=graph,
            compiler=compiler,
        )
        replay = ReplayEvidenceV1(
            plan_uid=search.plan.plan_uid,
            task_instance_sha256=task_instance_sha256,
            parent_contract_sha256=parent_contract_sha256,
            initial_state_snapshot_sha256=composition.initial_state_snapshot_sha256,
            final_state_snapshot_sha256=composition.final_state_snapshot_sha256,
            executed_canonical_actions=composition.executed_actions,
            final_goal_ids=composition.final_goal_ids,
            parent_contract_verified=composition.parent_contract_verified,
            replay_completed=composition.replay_completed,
            transition_restore_receipt_sha256s=(composition.transition_restore_receipt_sha256s),
            independent_parent_verifier_id=ALFWORLD_SEMANTIC_VERIFIER_ID,
            independent_parent_verifier_receipt_sha256=(
                composition.final_parent_verifier_receipt_sha256
            ),
            environment_identity_sha256=environment_identity_sha256,
            replay_seed=seed,
        )
        composition_receipt = certify_composition(search.plan, replay)

    sham_actions = (
        *source_a.executed_actions,
        *tuple(
            action for index, action in enumerate(source_b.executed_actions) if index != light_index
        ),
    )
    sham = _run_sequence(
        run_id="sham_drop_verified_light_action",
        task=task,
        data_root=data,
        output_dir=root / "sham",
        seed=seed,
        actions=tuple(sham_actions),
        expected_alfworld_version=expected_alfworld_version,
        metadata_payload=metadata_payload,
        graph=graph,
        compiler=compiler,
    )

    task_gate = {
        "expert_parent_success": expert.parent_contract_verified,
        "expert_environment_verifier_agree": (
            expert.environment_won == expert.parent_contract_verified
        ),
        "expert_trace_completed": expert.replay_completed,
        "source_a_parent_failed": not source_a.parent_contract_verified,
        "source_a_environment_verifier_agree": (
            source_a.environment_won == source_a.parent_contract_verified
        ),
        "source_b_parent_failed": not source_b.parent_contract_verified,
        "source_b_environment_verifier_agree": (
            source_b.environment_won == source_b.parent_contract_verified
        ),
        "source_a_acquired_effect": "acquired" in source_a.final_goal_ids,
        "source_b_light_effect": "light_on" in source_b.final_goal_ids,
        "two_fragments_materialized": len(fragments) == 2,
        "dp_plan_found": search.plan is not None,
        "full_replay_admitted": bool(
            composition_receipt is not None and composition_receipt.admitted
        ),
        "sham_parent_failed": not sham.parent_contract_verified,
        "no_fragment_training_eligible": all(
            not fragment.training_eligible for fragment in fragments
        ),
        "composition_not_on_policy": bool(
            composition_receipt is not None and not composition_receipt.on_policy_eligible
        ),
    }
    passed = all(task_gate.values()) and not fragment_errors
    payload = {
        "schema": ALFWORLD_CVFC_PILOT_SCHEMA,
        "scientific_result": False,
        "claim_scope": (
            "controlled_train_split_mechanism_evidence_not_natural_support_or_training_effect"
        ),
        "task": asdict(task),
        "seed": seed,
        "runtime": asdict(runtime),
        "environment_identity_sha256": environment_identity_sha256,
        "task_instance_sha256": task_instance_sha256,
        "parent_contract_sha256": parent_contract_sha256,
        "goal_graph_sha256": graph.sha256,
        "derivation": {
            "id": ALFWORLD_EXPERT_DERIVATION_ID,
            "sha256": derivation_sha256,
            "acquire_index": acquire_index,
            "light_index_after_acquire_removal": light_index,
        },
        "expert": expert.to_dict(),
        "source_a": source_a.to_dict(),
        "source_b_discovery": source_b_discovery.to_dict(),
        "source_b": source_b.to_dict(),
        "fragments": [fragment.to_dict() for fragment in fragments],
        "fragment_sha256s": [fragment.sha256 for fragment in fragments],
        "fragment_errors": fragment_errors,
        "search": search.to_dict(),
        "search_sha256": search.sha256,
        "composition": composition.to_dict() if composition is not None else None,
        "composition_receipt": (
            composition_receipt.to_dict() if composition_receipt is not None else None
        ),
        "composition_receipt_sha256": (
            composition_receipt.sha256 if composition_receipt is not None else None
        ),
        "sham": sham.to_dict(),
        "task_gate": task_gate,
        "passed": passed,
        "logical_action_count": sum(
            run.logical_action_count
            for run in (
                expert,
                source_a,
                source_b_discovery,
                source_b,
                sham,
            )
        )
        + (composition.logical_action_count if composition is not None else 0),
    }
    _write_json(root / "task_receipt.json", payload)
    return payload


def _run_sequence(
    *,
    run_id: str,
    task: AlfworldTask,
    data_root: Path,
    output_dir: Path,
    seed: int,
    actions: tuple[str, ...] | None,
    expected_alfworld_version: str,
    metadata_payload: dict[str, Any],
    graph: GoalGraph,
    compiler: AlfWorldSubproblemCompiler,
    max_dynamic_actions: int | None = None,
) -> PilotSequenceResult:
    branch = _InstalledPilotBranch(
        task=task,
        data_root=data_root,
        seed=seed,
        output_dir=output_dir,
        expected_alfworld_version=expected_alfworld_version,
    )
    rows: list[dict[str, Any]] = []
    requested = list(actions or ())
    executed: list[str] = []
    restore_receipts: list[str] = []
    dynamic_expert = actions is None
    if dynamic_expert and (max_dynamic_actions is None or max_dynamic_actions < 1):
        raise AlfWorldCompositionPilotError(
            "dynamic expert execution requires a positive action limit"
        )
    try:
        observation, _expert_plan, initial_artifact = branch.reset()
        vector = compiler.project_state(metadata_payload, observation["facts"])
        initial_goal_ids = _verified_goal_ids(graph, vector)
        environment_done = False
        environment_won = False
        index = 0
        while not environment_done:
            if dynamic_expert:
                if index >= int(max_dynamic_actions or 0):
                    break
                current_plan = branch.current_expert_plan()
                if not current_plan:
                    break
                action = current_plan[0]
                requested.append(action)
            else:
                if index >= len(requested):
                    break
                action = requested[index]
            before = vector
            next_observation, transition = branch.step(action)
            vector = compiler.project_state(
                metadata_payload,
                next_observation["facts"],
                previous=before,
            )
            label = compiler.label_transition(graph, before, action, vector)
            label_receipt = compiler.verify_annotation(label)
            semantic = audit_alfworld_snapshot_composition(
                metadata_payload,
                graph,
                vector,
                next_observation["facts"],
            )
            if not label_receipt.verified or not semantic.verified:
                raise AlfWorldCompositionPilotError(
                    "transition verifier rejected controlled ALFWorld evidence"
                )
            row = {
                "step_idx": index,
                "canonical_action": action,
                **transition,
                "goal_ids_before": list(_verified_goal_ids(graph, before)),
                "goal_ids_after": list(_verified_goal_ids(graph, vector)),
                "newly_satisfied_goal_ids": list(label.newly_satisfied_goal_ids),
                "regressed_goal_ids": list(label.regressed_goal_ids),
                "transition_label_sha256": label.annotation_sha256,
                "transition_label_verifier_receipt": asdict(label_receipt),
                "independent_semantic_receipt": semantic.to_dict(),
                "independent_semantic_receipt_sha256": canonical_sha256(semantic.to_dict()),
            }
            rows.append(row)
            executed.append(action)
            restore_receipts.append(str(transition["restore_receipt_sha256"]))
            environment_done = bool(transition["done"])
            environment_won = bool(transition["won"])
            index += 1
        final_artifact = write_alfworld_state_artifact(
            branch.controller.snapshot(),
            output_dir=output_dir,
        )
        final_semantic = audit_alfworld_snapshot_composition(
            metadata_payload,
            graph,
            vector,
            observation["facts"] if not rows else next_observation["facts"],
        )
        if not final_semantic.verified:
            raise AlfWorldCompositionPilotError("final independent semantic audit failed")
        final_receipt = final_semantic.to_dict()
        result = PilotSequenceResult(
            run_id=run_id,
            requested_actions=tuple(requested),
            executed_actions=tuple(executed),
            initial_state_snapshot_sha256=initial_artifact.state_snapshot_sha256,
            final_state_snapshot_sha256=final_artifact.state_snapshot_sha256,
            initial_goal_ids=initial_goal_ids,
            final_goal_ids=_verified_goal_ids(graph, vector),
            parent_contract_verified=final_semantic.parent_contract_passed,
            environment_won=environment_won,
            environment_done=environment_done,
            replay_completed=(
                environment_done if dynamic_expert else len(executed) == len(requested)
            ),
            transition_restore_receipt_sha256s=tuple(restore_receipts),
            transition_rows=tuple(rows),
            final_parent_verifier_receipt=final_receipt,
            final_parent_verifier_receipt_sha256=canonical_sha256(final_receipt),
            effect_evidence_sha256=canonical_sha256(
                {
                    "run_id": run_id,
                    "transition_labels": [row["transition_label_sha256"] for row in rows],
                    "semantic_receipts": [
                        row["independent_semantic_receipt_sha256"] for row in rows
                    ],
                    "final_parent_verifier_receipt_sha256": canonical_sha256(final_receipt),
                }
            ),
            logical_action_count=len(executed),
        )
        _write_json(output_dir / "sequence_receipt.json", result.to_dict())
        return result
    finally:
        branch.close()


def _fragment_from_sequence(
    *,
    result: PilotSequenceResult,
    task_instance_sha256: str,
    parent_contract_sha256: str,
    source_traj_uid: str,
    required_goal_ids: tuple[str, ...],
    source_policy_sha256: str,
    source_action_provenance_sha256: str,
    graph: GoalGraph,
) -> FragmentRecordV1:
    added = set(result.final_goal_ids) - set(result.initial_goal_ids)
    if not added:
        raise AlfWorldCompositionPilotError("source sequence has no verified effect")
    atom_by_id = {atom.goal_id: atom for atom in graph.atoms}
    invariants = tuple(
        sorted(goal_id for goal_id in required_goal_ids if atom_by_id[goal_id].invariant)
    )
    return FragmentRecordV1.build(
        task_instance_sha256=task_instance_sha256,
        parent_contract_sha256=parent_contract_sha256,
        split="train",
        source_traj_uid=source_traj_uid,
        source_kind=FragmentSourceKind.CONTROLLED_ORACLE,
        source_policy_id=ALFWORLD_EXPERT_DERIVATION_ID,
        source_policy_sha256=source_policy_sha256,
        source_parent_success=result.parent_contract_verified,
        source_started_from_parent_initial=True,
        pre_state_snapshot_sha256=result.initial_state_snapshot_sha256,
        post_state_snapshot_sha256=result.final_state_snapshot_sha256,
        pre_goal_ids=result.initial_goal_ids,
        post_goal_ids=result.final_goal_ids,
        required_goal_ids=required_goal_ids,
        invariant_goal_ids=invariants,
        canonical_actions=result.executed_actions,
        source_action_provenance_sha256=source_action_provenance_sha256,
        effect_verifier_id=ALFWORLD_SEMANTIC_VERIFIER_ID,
        effect_verifier_receipt_sha256=result.effect_evidence_sha256,
        restore_receipt_sha256s=result.transition_restore_receipt_sha256s,
    )


def _verified_goal_ids(graph: GoalGraph, vector: GoalVector) -> tuple[str, ...]:
    truth = vector.truth_map
    return tuple(
        sorted(
            atom.goal_id
            for atom in graph.atoms
            if (
                truth[atom.goal_id]
                if atom.invariant
                else atom.goal_id in vector.achievement_latches
            )
        )
    )


def _parent_target_goal_ids(graph: GoalGraph) -> tuple[str, ...]:
    root = next(node for node in graph.nodes if node.node_id == graph.root_id)
    return tuple(sorted(root.effect_goal_ids))


def _requirements_for_effects(
    graph: GoalGraph,
    effect_goal_ids: set[str],
) -> tuple[str, ...]:
    requirements = {
        goal_id
        for node in graph.nodes
        if set(node.effect_goal_ids) & effect_goal_ids
        for goal_id in node.precondition_goal_ids
    }
    return tuple(sorted(requirements))


def _load_task_metadata(task: AlfworldTask, data_root: Path) -> dict[str, Any]:
    trajectory_path = (data_root / task.trajectory_path).resolve()
    if not trajectory_path.is_relative_to(data_root):
        raise AlfWorldCompositionPilotError("trajectory path escapes data root")
    raw = json.loads(trajectory_path.read_text(encoding="utf-8"))
    params = raw.get("pddl_params")
    if not isinstance(params, dict):
        raise AlfWorldCompositionPilotError("ALFWorld trajectory lacks pddl_params")
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pddl_params": {
            "object_target": params.get("object_target", ""),
            "parent_target": params.get("parent_target", ""),
            "toggle_target": params.get("toggle_target", ""),
        },
    }


def _project_reset_or_step(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, tuple) or len(raw) not in {2, 4}:
        raise AlfWorldCompositionPilotError("unsupported ALFWorld result shape")
    observations = raw[0]
    infos = raw[1] if len(raw) == 2 else raw[3]
    if not isinstance(observations, (list, tuple)) or len(observations) != 1:
        raise AlfWorldCompositionPilotError("pilot requires one ALFWorld observation")
    if not isinstance(infos, Mapping):
        raise AlfWorldCompositionPilotError("ALFWorld infos must be a mapping")
    gamefile = _scalar_info(infos.get("extra.gamefile"), "extra.gamefile")
    facts = _scalar_info(infos.get("facts"), "facts")
    actions = _scalar_info(infos.get("admissible_commands"), "admissible_commands")
    if not isinstance(facts, (list, tuple)) or not isinstance(actions, (list, tuple)):
        raise AlfWorldCompositionPilotError("ALFWorld facts/actions are unavailable")
    normalized_facts = tuple(
        {
            "name": str(getattr(fact, "name", "")),
            "args": [str(value) for value in getattr(fact, "names", ())],
        }
        if not isinstance(fact, Mapping)
        else {
            "name": str(fact.get("name", "")),
            "args": [str(value) for value in fact.get("args") or ()],
        }
        for fact in facts
    )
    return {
        "text": str(observations[0]),
        "facts": normalized_facts,
        "admissible_actions": tuple(str(action) for action in actions),
        "gamefile": str(gamefile),
    }


def _extract_expert_plan(raw: Any) -> tuple[str, ...]:
    infos = raw[1] if isinstance(raw, tuple) and len(raw) == 2 else raw[3]
    if not isinstance(infos, Mapping):
        raise AlfWorldCompositionPilotError("ALFWorld infos must be a mapping")
    plan = _scalar_info(infos.get("extra.expert_plan"), "extra.expert_plan")
    if not isinstance(plan, (list, tuple)) or not plan:
        raise AlfWorldCompositionPilotError("train task has no expert plan")
    normalized = tuple(str(action).strip().lower() for action in plan if str(action).strip())
    if not normalized:
        raise AlfWorldCompositionPilotError("expert plan has no semantic actions")
    return normalized


def _scalar_info(value: Any, field: str) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    raise AlfWorldCompositionPilotError(f"expected one scalar ALFWorld {field}")


def _task_failure_receipt(
    *,
    task: AlfworldTask,
    seed: int,
    runtime: InstalledAlfWorldRuntimeIdentity,
    expert: PilotSequenceResult,
    reason: str,
    source_a: PilotSequenceResult | None = None,
    source_b_discovery: PilotSequenceResult | None = None,
) -> dict[str, Any]:
    return {
        "schema": ALFWORLD_CVFC_PILOT_SCHEMA,
        "scientific_result": False,
        "claim_scope": "controlled_train_split_mechanism_failure",
        "task": asdict(task),
        "seed": seed,
        "runtime": asdict(runtime),
        "expert": expert.to_dict(),
        "source_a": source_a.to_dict() if source_a is not None else None,
        "source_b_discovery": (
            source_b_discovery.to_dict() if source_b_discovery is not None else None
        ),
        "passed": False,
        "failure_reason": reason,
        "logical_action_count": sum(
            result.logical_action_count
            for result in (expert, source_a, source_b_discovery)
            if result is not None
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    try:
        staging.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
