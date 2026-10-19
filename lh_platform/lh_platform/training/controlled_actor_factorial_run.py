"""Bounded controlled actor-awareness collection and frozen-credit gate."""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lh_platform.studies.runtime import AlfworldTask, build_alfworld_task_manifest
from lh_platform.training.agentic_rl_components import AlfWorldFactDecomposition
from lh_platform.training.agentic_rl_config import (
    AgenticRLRuntimeBindings,
    bind_agentic_rl_runtime,
    load_agentic_rl_experiment,
    load_agentic_rl_factorial_manifest,
    load_controlled_actor_factorial_manifest,
)
from lh_platform.training.agentic_rl_harness import (
    assert_exclusive_gpu,
    query_compute_apps,
    transformers_policy_config_from_args,
)
from lh_platform.training.agentic_rl_p0 import run_p0
from lh_platform.training.agentic_rl_runtime import (
    AgenticRolloutBatch,
    RunEventLogger,
    canonical_sha256,
    write_agentic_rollout_artifact,
)
from lh_platform.training.alfworld_agentic_suite import OfficialAlfWorldBranchFactory
from lh_platform.training.alfworld_controlled_actor import (
    collect_controlled_actor_rollout,
)
from lh_platform.training.alfworld_controlled_aliasing import (
    AlfWorldControlledAliasingRecipe,
    _task_instruction,
    _task_metadata,
    load_alfworld_controlled_aliasing_recipe,
)
from lh_platform.training.config_provenance import repository_config_tree_bindings
from lh_platform.training.dataset import sha256_file
from lh_platform.training.dppo.analysis import build_frozen_credit_pairing
from lh_platform.training.g2po.credit import G2POCreditConfig
from lh_platform.training.transformers_peft_policy import (
    TransformersPeftPolicy,
    checkpoint_tree_sha256,
)

CONTROLLED_ACTOR_FACTORIAL_RESULT_SCHEMA = "lh_alfworld_controlled_actor_credit_factorial_result/v1"


class ControlledActorFactorialRunError(RuntimeError):
    """Raised when a controlled factorial cannot preserve its registered factors."""


def resolve_controlled_task_binding(
    recipe: AlfWorldControlledAliasingRecipe,
    data_root: str | Path,
) -> tuple[AlfworldTask, dict[str, Any], str, str]:
    """Rebuild the controlled task contract from official ALFWorld artifacts."""

    root = Path(data_root).resolve()
    manifest = build_alfworld_task_manifest(root)
    matches = [task for task in manifest.tasks if task.task_id == recipe.task_id]
    if len(matches) != 1:
        raise ControlledActorFactorialRunError(
            "controlled recipe task must resolve exactly once in ALFWorld data"
        )
    task = matches[0]
    trajectory_path = root / task.trajectory_path
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    metadata = _task_metadata(task, trajectory)
    instruction = _task_instruction(trajectory)
    task_manifest_sha256 = canonical_sha256(
        {
            "schema": "lh_controlled_actor_task_manifest/v1",
            "task": asdict(task),
            "game_sha256": sha256_file(root / task.game_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "logic_domain_sha256": sha256_file(root / "logic" / "alfred.pddl"),
            "logic_grammar_sha256": sha256_file(root / "logic" / "alfred.twl2"),
            "recipe_sha256": recipe.sha256,
        }
    )
    return task, metadata, instruction, task_manifest_sha256


def controlled_factorial_repository_bindings(
    resolved: Any,
) -> tuple[Mapping[str, Any], ...]:
    """Compatibility alias for the former controlled-only helper."""

    return repository_config_tree_bindings(resolved)


def run_controlled_actor_factorial(
    config_path: str | Path,
    *,
    model_path: str | Path,
    source_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str,
    preflight_only: bool,
    allow_shared_gpu: bool,
    generation_micro_batch_size: int,
) -> Path:
    """Run CPU preflight, then one frozen rollout per awareness level.

    This first tranche intentionally stops before PPO. It answers whether the
    base actor behaves differently and whether the registered credit key changes
    any action target on exactly the same rollout. V2 configurations also run
    teacher-forced old-logprob revalidation at the still-loaded initial policy;
    eligibility never executes an optimizer step in this command.
    """

    resolved = load_controlled_actor_factorial_manifest(config_path)
    root = resolved.workspace_root
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    event_logger = RunEventLogger(
        target / "run_events.jsonl",
        run_id=uuid.uuid4().hex,
    )
    event_logger.emit(
        "controlled_factorial_started",
        config_path=str(resolved.config_path),
        controlled_factorial_sha256=resolved.sha256,
        requested_device=device,
        preflight_only=preflight_only,
        generation_micro_batch_size=generation_micro_batch_size,
    )
    preregistration_bindings = controlled_factorial_repository_bindings(resolved)
    bindings_by_path = {
        str(binding["relative_path"]): binding for binding in preregistration_bindings
    }
    repository_binding = bindings_by_path[resolved.config_path.relative_to(root).as_posix()]
    recipe_path = root / resolved.payload["controlled_recipe"]["path"]
    recipe_binding = bindings_by_path[recipe_path.relative_to(root).as_posix()]

    factorial_path = root / resolved.payload["factorial"]["path"]
    factorial = load_agentic_rl_factorial_manifest(factorial_path)
    experiments = {
        awareness: load_agentic_rl_experiment(root / reference["path"])
        for awareness, reference in factorial.payload["paired_experiments"].items()
    }
    unaware = experiments["none"]
    base_args = bind_agentic_rl_runtime(
        unaware,
        AgenticRLRuntimeBindings(
            model_path=Path(model_path),
            source_root=Path(source_root),
            data_root=Path(data_root),
            output_dir=target / "base_preflight",
            device=device,
            preflight_only=True,
            allow_shared_gpu=allow_shared_gpu,
        ),
    )
    base_args.generation_micro_batch_size = generation_micro_batch_size
    base_preflight_path = run_p0(base_args)
    base_preflight = json.loads(base_preflight_path.read_text(encoding="utf-8"))
    preflight = {
        "schema": "lh_alfworld_controlled_actor_credit_preflight/v1",
        "controlled_factorial": {
            "path": str(resolved.config_path),
            "file_sha256": sha256_file(resolved.config_path),
            "semantic_sha256": resolved.sha256,
            "repository_binding": repository_binding,
        },
        "factorial": dict(resolved.payload["factorial"]),
        "controlled_recipe": {
            **dict(resolved.payload["controlled_recipe"]),
            "repository_binding": recipe_binding,
        },
        "runtime": dict(resolved.payload["runtime"]),
        "update_gate": dict(resolved.payload["update_gate"]),
        "claim_boundary": dict(resolved.payload["claim_boundary"]),
        "preregistration_bindings": [dict(binding) for binding in preregistration_bindings],
        "base_preflight_path": str(base_preflight_path),
        "base_preflight_sha256": sha256_file(base_preflight_path),
        "base_checkpoint_sha256": base_preflight["base_checkpoint_sha256"],
        "source_revision": base_preflight["source_revision"],
        "gpu_loaded": False,
        "formal_launch_eligible": bool(
            all(binding["matches_head"] for binding in preregistration_bindings)
        ),
        "claim_scope": "controlled_actor_factorial_preflight_only",
    }
    preflight_path = target / "controlled_factorial_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    event_logger.emit(
        "controlled_factorial_preflight_written",
        path=str(preflight_path),
        sha256=sha256_file(preflight_path),
        formal_launch_eligible=preflight["formal_launch_eligible"],
        gpu_loaded=False,
    )
    if preflight_only:
        return preflight_path
    if not preflight["formal_launch_eligible"]:
        raise ControlledActorFactorialRunError(
            "controlled factorial config tree must be committed unchanged"
        )

    compute_apps = query_compute_apps() if device.startswith("cuda") else ()
    event_logger.emit(
        "controlled_factorial_resource_preflight",
        requested_device=device,
        compute_apps_before_launch=list(compute_apps),
        allow_shared_gpu=allow_shared_gpu,
    )
    assert_exclusive_gpu(
        device=device,
        allow_shared_gpu=allow_shared_gpu,
        compute_apps=compute_apps,
    )
    base_checkpoint_sha256 = checkpoint_tree_sha256(model_path)
    if base_checkpoint_sha256 != base_preflight["base_checkpoint_sha256"]:
        raise ControlledActorFactorialRunError(
            "model checkpoint changed after controlled preflight"
        )
    policy_args = argparse.Namespace(**vars(base_args))
    policy_args.preflight_only = False
    policy_config = transformers_policy_config_from_args(
        policy_args,
        base_checkpoint_sha256=base_checkpoint_sha256,
    )
    event_logger.emit(
        "controlled_factorial_policy_load_started",
        base_checkpoint_sha256=base_checkpoint_sha256,
        requested_device=device,
    )
    policy = TransformersPeftPolicy(policy_config)
    event_logger.emit(
        "controlled_factorial_policy_loaded",
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        adapter_initial_delta_zero=policy.adapter_initial_delta_zero,
    )

    recipe = load_alfworld_controlled_aliasing_recipe(recipe_path)
    data_root = Path(data_root).resolve()
    source_root = Path(source_root).resolve()
    (
        task,
        metadata,
        instruction,
        controlled_task_manifest_sha256,
    ) = resolve_controlled_task_binding(
        recipe,
        data_root,
    )
    runtime = resolved.payload["runtime"]
    rollout_config_sha256 = canonical_sha256(
        {
            "schema": "lh_controlled_actor_rollout_config/v1",
            "controlled_factorial_sha256": resolved.sha256,
            "samples_per_state": runtime["samples_per_state"],
            "max_episode_steps": runtime["max_episode_steps"],
            "sampling_seed": runtime["sampling_seed"],
            "generation_micro_batch_size": generation_micro_batch_size,
            "action_space": "free_text_then_source_alfworld_projection",
        }
    )
    branch_factory = OfficialAlfWorldBranchFactory(
        source_root=source_root,
        data_root=data_root,
    )
    credit_config = G2POCreditConfig(
        gamma=base_args.gamma,
        step_advantage_weight=base_args.step_advantage_weight,
        normalization=base_args.credit_normalization,
        invalid_action_penalty=base_args.invalid_action_penalty,
        epsilon=base_args.credit_epsilon,
    )

    awareness_results: dict[str, Mapping[str, Any]] = {}
    initial_checkpoint_sha256 = policy.checkpoint_sha256
    update_gate = resolved.payload["update_gate"]
    require_old_logprob_audit = bool(
        update_gate.get("require_teacher_forced_old_logprob_revalidation", False)
    )
    for awareness in runtime["actor_awareness_modes"]:
        event_logger.emit(
            "controlled_awareness_rollout_started",
            actor_awareness_mode=awareness,
            samples_per_state=runtime["samples_per_state"],
            state_count=len(recipe.branches),
            max_episode_steps=runtime["max_episode_steps"],
            sampling_seed=runtime["sampling_seed"],
            policy_checkpoint_sha256=initial_checkpoint_sha256,
        )
        batch = collect_controlled_actor_rollout(
            recipe,
            task=task,
            task_metadata=metadata,
            instruction=instruction,
            policy=policy,
            branch_factory=branch_factory,
            artifacts_dir=target / "environment",
            actor_awareness_mode=str(awareness),
            samples_per_state=int(runtime["samples_per_state"]),
            max_episode_steps=int(runtime["max_episode_steps"]),
            sampling_seed=int(runtime["sampling_seed"]),
            task_manifest_sha256=controlled_task_manifest_sha256,
            rollout_config_sha256=rollout_config_sha256,
            sampling_temperature=base_args.train_temperature,
        )
        if batch.policy_checkpoint_sha256 != initial_checkpoint_sha256:
            raise ControlledActorFactorialRunError(
                "awareness rollouts did not use the same initial policy"
            )
        awareness_dir = target / str(awareness)
        raw_path = awareness_dir / "frozen_model_policy_rollout.jsonl"
        raw_sha256 = write_agentic_rollout_artifact(raw_path, batch)
        annotated_records = AlfWorldFactDecomposition().annotate(batch.records)
        annotated = AgenticRolloutBatch(
            records=annotated_records,
            policy_samples=batch.policy_samples,
            task_manifest_sha256=batch.task_manifest_sha256,
            policy_checkpoint_sha256=batch.policy_checkpoint_sha256,
        ).validate()
        annotated_path = awareness_dir / "frozen_annotated_rollout.jsonl"
        annotated_sha256 = write_agentic_rollout_artifact(
            annotated_path,
            annotated,
        )
        pairing = build_frozen_credit_pairing(
            batch,
            config=credit_config,
            expected_recipe=recipe,
            expected_task_metadata=metadata,
            expected_task_manifest_sha256=controlled_task_manifest_sha256,
            expected_gamefile=str((data_root / task.game_path).resolve()),
            preregistration_bindings=preregistration_bindings,
        )
        pairing_path = awareness_dir / "frozen_credit_pairing.json"
        pairing_path.write_text(
            json.dumps(pairing, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        old_logprob_audit: Mapping[str, Any] | None = None
        old_logprob_audit_path: Path | None = None
        if require_old_logprob_audit:
            old_logprob_audit = policy.audit_old_action_log_probs(
                batch.policy_samples,
                max_absolute_error=float(update_gate["teacher_forced_max_absolute_error"]),
                max_mean_absolute_error=float(
                    update_gate["teacher_forced_max_mean_absolute_error"]
                ),
                required_token_coverage=float(
                    update_gate["teacher_forced_required_token_coverage"]
                ),
            )
            old_logprob_audit_path = awareness_dir / "teacher_forced_old_logprob_audit.json"
            old_logprob_audit_path.write_text(
                json.dumps(old_logprob_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        awareness_results[str(awareness)] = {
            "raw_rollout_path": str(raw_path),
            "raw_rollout_sha256": raw_sha256,
            "annotated_rollout_path": str(annotated_path),
            "annotated_rollout_sha256": annotated_sha256,
            "credit_pairing_path": str(pairing_path),
            "credit_pairing_sha256": sha256_file(pairing_path),
            "teacher_forced_old_logprob_audit_path": (
                str(old_logprob_audit_path) if old_logprob_audit_path else None
            ),
            "teacher_forced_old_logprob_audit_sha256": (
                sha256_file(old_logprob_audit_path) if old_logprob_audit_path else None
            ),
            "teacher_forced_old_logprob_gate": (
                {
                    "passed": bool(old_logprob_audit["passed"]),
                    "sample_count": old_logprob_audit["sample_count"],
                    "action_token_count": old_logprob_audit["action_token_count"],
                    "required_token_coverage": old_logprob_audit["required_token_coverage"],
                    "token_coverage": old_logprob_audit["token_coverage"],
                    "max_absolute_error": old_logprob_audit["max_absolute_error"],
                    "mean_absolute_error": old_logprob_audit["mean_absolute_error"],
                }
                if old_logprob_audit
                else None
            ),
            "behavior": _behavior_summary(annotated),
            "credit_gate": {
                "difference_count": pairing["credit_target_difference_count"],
                "max_absolute_difference": pairing["max_absolute_credit_target_difference"],
                "credit_difference_gate_passed": pairing["credit_difference_gate_passed"],
                "blocked_by": pairing["credit_gate_blocked_by"],
                "paired_policy_update_eligible": pairing["paired_policy_update_eligible"],
                "policy_update_blocked_by": pairing["policy_update_blocked_by"],
            },
        }
        event_logger.emit(
            "controlled_awareness_rollout_completed",
            actor_awareness_mode=awareness,
            raw_rollout_path=str(raw_path),
            raw_rollout_sha256=raw_sha256,
            transition_count=len(batch.records),
            trajectory_count=len({row.traj_uid for row in batch.records}),
            success_count=awareness_results[str(awareness)]["behavior"]["success_count"],
            credit_target_difference_count=pairing["credit_target_difference_count"],
            credit_difference_gate_passed=pairing["credit_difference_gate_passed"],
            teacher_forced_old_logprob_gate_passed=(
                bool(old_logprob_audit["passed"]) if old_logprob_audit else False
            ),
            paired_policy_update_eligible=False,
        )

    all_credit_gates_passed = all(
        bool(result["credit_gate"]["credit_difference_gate_passed"])
        for result in awareness_results.values()
    )
    all_old_logprob_gates_passed = bool(
        require_old_logprob_audit
        and all(
            result["teacher_forced_old_logprob_gate"]["passed"]
            for result in awareness_results.values()
        )
    )
    paired_policy_update_eligible = bool(all_credit_gates_passed and all_old_logprob_gates_passed)
    policy_update_blocked_by: list[str] = []
    if not all_credit_gates_passed:
        policy_update_blocked_by.append("not_all_awareness_credit_gates_passed")
    if not require_old_logprob_audit:
        policy_update_blocked_by.append("teacher_forced_old_logprob_revalidation_pending")
    elif not all_old_logprob_gates_passed:
        policy_update_blocked_by.append("teacher_forced_old_logprob_mismatch")
    result = {
        "schema": CONTROLLED_ACTOR_FACTORIAL_RESULT_SCHEMA,
        "controlled_factorial_sha256": resolved.sha256,
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "initial_policy_checkpoint_sha256": initial_checkpoint_sha256,
        "controlled_task_manifest_sha256": controlled_task_manifest_sha256,
        "rollout_config_sha256": rollout_config_sha256,
        "awareness_results": awareness_results,
        "cross_awareness_sampling": "independent_descriptive_not_common_rng",
        "all_awareness_credit_difference_gates_passed": all_credit_gates_passed,
        "all_awareness_old_logprob_gates_passed": all_old_logprob_gates_passed,
        "policy_update_executed": False,
        "paired_policy_update_eligible": paired_policy_update_eligible,
        "policy_update_blocked_by": policy_update_blocked_by,
        "next_action": (
            "run_one_paired_update_from_frozen_rollout"
            if paired_policy_update_eligible
            else (
                "fix_generation_update_probability_mismatch"
                if require_old_logprob_audit and not all_old_logprob_gates_passed
                else (
                    "teacher_force_revalidate_old_logprobs_before_paired_update"
                    if all_credit_gates_passed
                    else "stop_before_policy_update_no_credit_difference"
                )
            )
        ),
        "claim_scope": (
            "controlled_actor_behavior_and_frozen_credit_diagnostics_not_policy_improvement"
        ),
    }
    result_path = target / "controlled_actor_factorial_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    event_logger.emit(
        "controlled_factorial_completed",
        result_path=str(result_path),
        result_sha256=sha256_file(result_path),
        all_awareness_credit_difference_gates_passed=all_credit_gates_passed,
        paired_policy_update_eligible=paired_policy_update_eligible,
        policy_update_executed=False,
        next_action=result["next_action"],
    )
    return result_path


def _behavior_summary(batch: AgenticRolloutBatch) -> dict[str, Any]:
    terminal_by_trajectory = {row.traj_uid: row for row in batch.records if row.terminal}
    state_by_trajectory = {
        row.traj_uid: str(row.extra["controlled_state_id"])
        for row in batch.records
        if row.step_idx == 0
    }
    states: dict[str, dict[str, Any]] = {}
    for traj_uid, terminal in terminal_by_trajectory.items():
        state = state_by_trajectory[traj_uid]
        bucket = states.setdefault(
            state,
            {
                "trajectories": 0,
                "successes": 0,
                "transitions": 0,
                "protocol_valid_actions": 0,
                "environment_admissible_actions": 0,
                "goal_progress_transitions": 0,
            },
        )
        bucket["trajectories"] += 1
        bucket["successes"] += int(terminal.success)
    invalid_reasons: dict[str, int] = {}
    for row in batch.records:
        state = str(row.extra["controlled_state_id"])
        bucket = states[state]
        bucket["transitions"] += 1
        bucket["protocol_valid_actions"] += int(row.extra.get("is_action_valid") is True)
        bucket["environment_admissible_actions"] += int(
            row.extra.get("environment_action") in row.extra.get("alfworld_admissible_actions", ())
        )
        bucket["goal_progress_transitions"] += int(bool(row.extra.get("newly_satisfied_goal_ids")))
        for reason in row.extra.get("invalid_action_reasons", ()):
            label = str(reason)
            invalid_reasons[label] = invalid_reasons.get(label, 0) + 1
    for bucket in states.values():
        transitions = int(bucket["transitions"])
        bucket["protocol_valid_action_rate"] = bucket["protocol_valid_actions"] / transitions
        bucket["environment_admissible_action_rate"] = (
            bucket["environment_admissible_actions"] / transitions
        )
    transition_count = len(batch.records)
    action_token_counts = [len(sample.action_token_ids) for sample in batch.policy_samples]
    prompt_token_counts = [len(sample.prompt_token_ids) for sample in batch.policy_samples]
    return {
        "trajectory_count": len(terminal_by_trajectory),
        "transition_count": transition_count,
        "success_count": sum(row.success for row in terminal_by_trajectory.values()),
        "protocol_valid_action_rate": sum(
            row.extra.get("is_action_valid") is True for row in batch.records
        )
        / transition_count,
        "environment_admissible_action_rate": sum(
            row.extra.get("environment_action") in row.extra.get("alfworld_admissible_actions", ())
            for row in batch.records
        )
        / transition_count,
        "invalid_action_reason_counts": dict(sorted(invalid_reasons.items())),
        "goal_progress_transition_count": sum(
            bool(row.extra.get("newly_satisfied_goal_ids")) for row in batch.records
        ),
        "mean_prompt_token_count": sum(prompt_token_counts) / transition_count,
        "mean_action_token_count": sum(action_token_counts) / transition_count,
        "max_action_token_count": max(action_token_counts),
        "by_controlled_state": dict(sorted(states.items())),
    }
