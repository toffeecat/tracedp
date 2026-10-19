"""Paired ALFWorld P0 for G2PO-TN versus Verifier-Refined G2PO credit."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.training.agentic_rl_components import (
    build_decomposition_provider,
    build_g2po_family_method,
)
from lh_platform.training.agentic_rl_config import (
    AgenticRLConfigError as AgenticRLP0ConfigError,
)
from lh_platform.training.agentic_rl_config import (
    namespace_from_experiment_config,
)
from lh_platform.training.agentic_rl_harness import (
    agentic_rl_rollout_config_from_args,
    assert_exclusive_gpu,
    query_compute_apps,
    transformers_policy_config_from_args,
    validate_agentic_rl_execution_admission,
)
from lh_platform.training.agentic_rl_runtime import (
    AgenticRLExperimentRunner,
    ArmRunPlan,
    EvaluationRequest,
    ExperimentArm,
    RolloutRequest,
    RunEventLogger,
    build_paired_runtime_lock,
    canonical_sha256,
    collect_on_policy_rollout,
)
from lh_platform.training.alfworld_agentic_suite import (
    AlfWorldNativeTaskSuite,
    AlfWorldSuiteConfig,
    OfficialAlfWorldBranchFactory,
)
from lh_platform.training.credit_estimators.phasepool_study import (
    load_phasepool_calibration_admission,
)
from lh_platform.training.dppo.analysis import analyze_rollout_artifact
from lh_platform.training.dynamic_sampling import DynamicSamplingConfig
from lh_platform.training.g2po.actor_parity import audit_alfworld_actor_profile
from lh_platform.training.g2po.actor_protocol import (
    CONTEXT_EXTENSION_PLACEMENT_REVISION,
)
from lh_platform.training.g2po.credit import G2POCreditConfig
from lh_platform.training.g2po.eval_population import (
    audit_g2po_valid_seen_schedule,
)
from lh_platform.training.g2po.source_parity import audit_g2po_source_parity
from lh_platform.training.transformers_peft_policy import (
    TransformersPeftPolicy,
    TransformersPeftPolicyConfig,
    checkpoint_tree_sha256,
    transformers_peft_component,
)

if TYPE_CHECKING:
    from lh_platform.training.toolbench_agentic_suite import ToolBenchNativeTaskSuite


P0_RESULT_SCHEMA = "lh_agentic_rl_decomposition_p0/v1"
P0_COMPONENT_IDS = {
    "policy_backend": "transformers_peft_causal_lm",
    "task_suite": "alfworld_native",
    "rl_method": "g2po_family_clipped_pg",
    "control_decomposition": "none",
    "treatment_decomposition": "alfworld_fact_phases",
    "control_credit_assigner": "g2po_tn",
    "treatment_credit_assigner": "oracle_phase_refined_g2po_tn",
}


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_p0(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_logger = RunEventLogger(
        output_dir / "run_events.jsonl",
        run_id=uuid.uuid4().hex,
    )
    event_logger.emit(
        "run_started",
        experiment_id=str(getattr(args, "experiment_id", "agentic_rl_p0")),
        experiment_config_path=str(getattr(args, "experiment_config_path", "")),
        experiment_config_sha256=str(getattr(args, "experiment_config_sha256", "")),
        resolved_config=_redacted_config(vars(args)),
        run_events_path=str(event_logger.path),
    )
    try:
        resolved_experiment_path = _persist_resolved_experiment(args, output_dir)
        if resolved_experiment_path is not None:
            event_logger.emit(
                "resolved_experiment_written",
                path=str(resolved_experiment_path),
                sha256=str(args.resolved_experiment_sha256),
                schema=str(args.resolved_experiment.get("schema", "")),
            )
        with ExitStack() as policy_cleanup:
            result = _run_p0(
                args,
                event_logger=event_logger,
                policy_cleanup=policy_cleanup,
            )
    except Exception as error:
        event_logger.emit(
            "run_failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
    event_logger.emit("run_completed", result_path=str(result))
    if result.name == "full_evaluation_result.json":
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["run_events_path"] = str(event_logger.path)
        payload["run_events_sha256"] = _sha256_file(event_logger.path)
        result.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _persist_resolved_experiment(
    args: argparse.Namespace,
    output_dir: Path,
) -> Path | None:
    payload = getattr(args, "resolved_experiment", None)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise AgenticRLP0ConfigError("resolved experiment receipt must be a mapping")
    actual_sha256 = canonical_sha256(payload)
    expected_sha256 = str(getattr(args, "resolved_experiment_sha256", ""))
    if actual_sha256 != expected_sha256:
        raise AgenticRLP0ConfigError("resolved experiment receipt changed after config validation")
    path = output_dir / "resolved_experiment.json"
    # Publish the resolved experiment atomically so a crash mid-write cannot
    # leave a truncated receipt that readers mistake for the real one.
    atomic_write_json(path, payload, overwrite=True)
    return path


def _redacted_config(value: Any, *, key: str = "") -> Any:
    sensitive = {"api_key", "password", "secret", "access_token", "auth_token"}
    if key.lower() in sensitive:
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _redacted_config(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redacted_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _run_p0(
    args: argparse.Namespace,
    *,
    event_logger: RunEventLogger,
    policy_cleanup: ExitStack,
) -> Path:
    validate_agentic_rl_execution_admission(args)
    output_dir = Path(args.output_dir).resolve()
    full_eval_only = getattr(args, "execution_mode", "paired_train_eval") == "full_eval_only"
    if bool(getattr(args, "full_eval_only", False)) and not full_eval_only:
        raise AgenticRLP0ConfigError(
            "runtime flags cannot override paired_train_eval structured intent"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path).resolve()
    source_root = Path(args.source_root).resolve()
    data_root = Path(args.data_root).resolve()
    train_manifest = Path(args.train_manifest).resolve()
    eval_manifest = Path(args.eval_manifest).resolve()
    debug_snapshot = _runtime_debug_snapshot()
    event_logger.emit("runtime_debug_snapshot", **debug_snapshot)
    _validate_component_selections(args)
    credit_config = G2POCreditConfig(
        gamma=float(getattr(args, "gamma", 0.95)),
        step_advantage_weight=float(getattr(args, "step_advantage_weight", 1.0)),
        normalization=str(getattr(args, "credit_normalization", "mean_std_norm")),
        invalid_action_penalty=float(getattr(args, "invalid_action_penalty", 0.1)),
        epsilon=float(getattr(args, "credit_epsilon", 1e-6)),
    )
    event_logger.emit(
        "configuration_preloaded",
        resolved_experiment_path=str(output_dir / "resolved_experiment.json"),
        resolved_experiment_sha256=str(getattr(args, "resolved_experiment_sha256", "")),
        components={
            name: str(getattr(args, f"{name}_id", expected))
            for name, expected in P0_COMPONENT_IDS.items()
        },
        control_state_identity_fields=list(
            getattr(args, "control_state_identity_fields", ["anchor_observation"])
        ),
        treatment_state_identity_fields=list(
            getattr(
                args,
                "treatment_state_identity_fields",
                [
                    "anchor_observation",
                    "residual_goal_signature",
                    "frontier_signature",
                ],
            )
        ),
        credit_config=asdict(credit_config),
    )
    actual_source_revision = _git_head(source_root)
    if actual_source_revision != args.source_revision:
        raise RuntimeError(
            "pinned source revision mismatch: "
            f"expected {args.source_revision}, got {actual_source_revision}"
        )
    event_logger.emit(
        "source_verified",
        source_root=str(source_root),
        source_revision=actual_source_revision,
    )
    source_parity = audit_g2po_source_parity(
        source_root=source_root,
        expected_revision=args.source_revision,
        expected_source_sha256=args.source_credit_file_sha256,
        fixture_path=args.source_parity_fixture,
        tolerance=args.source_parity_tolerance,
    )
    event_logger.emit(
        "source_credit_parity_verified",
        **asdict(source_parity),
    )
    actor_parity = audit_alfworld_actor_profile(
        source_root=source_root,
        actor_config_path=args.actor_config_path,
    )
    event_logger.emit(
        (
            "source_actor_parity_verified"
            if actor_parity.behavioral_contract_exact
            else "model_compatible_actor_profile_verified"
        ),
        actor_profile_id=args.actor_profile_id,
        evaluation_schedule_path=args.eval_schedule,
        evaluation_selection=args.eval_selection,
        **asdict(actor_parity),
    )
    evaluation_schedule_audit = None
    if args.eval_selection == "source_seed_schedule":
        if not args.eval_schedule:
            raise RuntimeError("source_seed_schedule evaluation requires a schedule path")
        event_logger.emit(
            "source_evaluation_schedule_audit_started",
            evaluation_schedule_path=args.eval_schedule,
            source_revision=args.source_revision,
        )
        evaluation_schedule_audit = audit_g2po_valid_seen_schedule(
            source_root=source_root,
            data_root=data_root,
            schedule_path=args.eval_schedule,
            expected_revision=args.source_revision,
        )
        event_logger.emit(
            "source_evaluation_schedule_audit_completed",
            **asdict(evaluation_schedule_audit),
        )

    base_checkpoint_sha256 = checkpoint_tree_sha256(model_path)
    expected_base_checkpoint_sha256 = str(
        getattr(args, "configured_model_checkpoint_tree_sha256", "")
    )
    if base_checkpoint_sha256 != expected_base_checkpoint_sha256:
        raise RuntimeError(
            "local base checkpoint differs from the structured model lock: "
            f"expected {expected_base_checkpoint_sha256}, got {base_checkpoint_sha256}"
        )
    event_logger.emit(
        "base_checkpoint_hashed",
        model_path=str(model_path),
        configured_model_reference=str(getattr(args, "configured_model_reference", "")),
        configured_model_revision=str(getattr(args, "configured_model_revision", "")),
        base_checkpoint_sha256=base_checkpoint_sha256,
    )
    policy_config = transformers_policy_config_from_args(
        args,
        base_checkpoint_sha256=base_checkpoint_sha256,
    )
    if full_eval_only:
        policy_config = replace(policy_config, inference_only=True)
    training_backend_preflight = _preflight_training_backend(
        args,
        policy_config=policy_config,
    )
    policy_backend = transformers_peft_component(policy_config)
    rollout_seeds = tuple(args.rollout_seed + index for index in range(args.updates))
    rollout_config = agentic_rl_rollout_config_from_args(args)
    rollout_config_sha256 = canonical_sha256(rollout_config)
    phasepool_run = (
        getattr(args, "comparison_primary_variable", "credit_state_identity")
        == "cross_fitted_credit_estimator"
    )
    if phasepool_run and args.updates != 1:
        raise AgenticRLP0ConfigError("PhasePool mechanism screen requires exactly one update")
    phasepool_calibration_admission = None
    if phasepool_run:
        receipt_path = getattr(args, "phasepool_calibration_receipt", None)
        if not receipt_path:
            raise AgenticRLP0ConfigError(
                "PhasePool online runtime lacks calibration admission receipt"
            )
    optimizer_config_sha256 = canonical_sha256(
        {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "clip_ratio": args.clip_ratio,
            "reference_kl_coef": args.reference_kl_coef,
            "reference_kl_type": args.reference_kl_type,
            "max_grad_norm": args.max_grad_norm,
            "update_micro_batch_size": args.update_micro_batch_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_target_modules": args.lora_target_modules,
        }
    )
    branch_factory = OfficialAlfWorldBranchFactory(
        source_root=source_root,
        data_root=data_root,
    )

    control_method = build_g2po_family_method(
        str(getattr(args, "control_credit_assigner_id", "g2po_tn")),
        estimator_config=getattr(args, "control_credit_estimator_config", None),
        config=credit_config,
        source_contract_sha256=source_parity.contract_sha256,
    )
    treatment_method = build_g2po_family_method(
        str(
            getattr(
                args,
                "treatment_credit_assigner_id",
                "oracle_phase_refined_g2po_tn",
            )
        ),
        estimator_config=getattr(args, "treatment_credit_estimator_config", None),
        config=credit_config,
        source_contract_sha256=source_parity.contract_sha256,
    )
    control_decomposition = build_decomposition_provider(
        str(getattr(args, "control_decomposition_id", "none"))
    )
    treatment_decomposition = build_decomposition_provider(
        str(getattr(args, "treatment_decomposition_id", "alfworld_fact_phases"))
    )
    control_arm_id = str(getattr(args, "control_arm_id", "g2po_tn_control"))
    treatment_arm_id = str(getattr(args, "treatment_arm_id", "dppo_treatment"))
    train_suite_control = _suite(
        role="train_task_suite",
        manifest=train_manifest,
        data_root=data_root,
        artifacts_dir=output_dir / "control" / "environment",
        source_revision=args.source_revision,
        train_task_groups=args.train_task_groups,
        train_task_selection=args.train_task_selection,
        eval_tasks_per_horizon=args.eval_tasks_per_horizon,
        history_length=args.history_length,
        train_temperature=args.train_temperature,
        eval_temperature=args.eval_temperature,
        actor_protocol_id=args.actor_protocol_id,
        show_admissible_actions=bool(getattr(args, "show_admissible_actions", True)),
        model_io_profile_id=_bound_model_io_profile_id(args),
        actor_awareness_mode=args.actor_awareness_mode,
        evaluation_schedule_path=None,
        evaluation_selection="all_manifest_tasks_exactly_once",
        branch_factory=branch_factory,
    )
    eval_suite_control = _suite(
        role="eval_task_suite",
        manifest=eval_manifest,
        data_root=data_root,
        artifacts_dir=output_dir / "control" / "evaluation_environment",
        source_revision=args.source_revision,
        train_task_groups=args.train_task_groups,
        train_task_selection=args.train_task_selection,
        eval_tasks_per_horizon=args.eval_tasks_per_horizon,
        history_length=args.history_length,
        train_temperature=args.train_temperature,
        eval_temperature=args.eval_temperature,
        actor_protocol_id=args.actor_protocol_id,
        show_admissible_actions=bool(getattr(args, "show_admissible_actions", True)),
        model_io_profile_id=_bound_model_io_profile_id(args),
        actor_awareness_mode=args.actor_awareness_mode,
        evaluation_schedule_path=args.eval_schedule,
        evaluation_selection=args.eval_selection,
        branch_factory=branch_factory,
    )
    _validate_runtime_component_instances(
        args,
        policy_backend=policy_backend,
        train_suite=train_suite_control,
        control_method=control_method,
        treatment_method=treatment_method,
        control_decomposition=control_decomposition,
        treatment_decomposition=treatment_decomposition,
    )
    train_task_selection_receipt = train_suite_control.training_task_selection_receipt(
        seed=args.rollout_seed
    )
    event_logger.emit(
        "task_suites_preloaded",
        train_manifest_path=str(train_manifest),
        train_manifest_sha256=train_suite_control.manifest_sha256,
        train_dataset_content_sha256=train_suite_control.dataset_content_sha256,
        train_task_count=len(train_suite_control.manifest.tasks),
        train_task_selection=train_task_selection_receipt,
        eval_manifest_path=str(eval_manifest),
        eval_manifest_sha256=eval_suite_control.manifest_sha256,
        eval_dataset_content_sha256=eval_suite_control.dataset_content_sha256,
        eval_task_count=len(eval_suite_control.manifest.tasks),
        eval_episode_count=args.eval_task_count,
        eval_selection=args.eval_selection,
        eval_schedule_path=args.eval_schedule,
        eval_schedule_sha256=eval_suite_control.evaluation_schedule_sha256,
    )
    compute_apps = query_compute_apps() if args.device.startswith("cuda") else ()
    event_logger.emit(
        "resource_preflight",
        requested_device=args.device,
        gpu_loaded=False,
        compute_apps_before_launch=list(compute_apps),
        allow_shared_gpu=bool(args.allow_shared_gpu),
    )
    preflight = {
        "schema": "lh_agentic_rl_decomposition_preflight/v1",
        "run_id": event_logger.run_id,
        "run_events_path": str(event_logger.path),
        "experiment_id": str(getattr(args, "experiment_id", "agentic_rl_p0")),
        "experiment_config_path": str(getattr(args, "experiment_config_path", "")),
        "experiment_config_sha256": str(getattr(args, "experiment_config_sha256", "")),
        "resolved_experiment_path": str(output_dir / "resolved_experiment.json"),
        "resolved_experiment_sha256": str(getattr(args, "resolved_experiment_sha256", "")),
        "resolved_config": _redacted_config(vars(args)),
        "runtime_debug_snapshot": debug_snapshot,
        "source_credit_parity": asdict(source_parity),
        "source_actor_parity": asdict(actor_parity),
        "source_evaluation_schedule_audit": (
            asdict(evaluation_schedule_audit) if evaluation_schedule_audit is not None else None
        ),
        "gpu_loaded": False,
        "source_revision": actual_source_revision,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "policy_config": asdict(policy_config),
        "policy_config_sha256": canonical_sha256(asdict(policy_config)),
        "policy_backend": asdict(policy_backend),
        "training_backend_preflight": training_backend_preflight,
        "train_task_suite": asdict(train_suite_control.component),
        "train_manifest_sha256": train_suite_control.manifest_sha256,
        "train_dataset_content_sha256": train_suite_control.dataset_content_sha256,
        "train_task_selection": train_task_selection_receipt,
        "eval_task_suite": asdict(eval_suite_control.component),
        "eval_manifest_sha256": eval_suite_control.manifest_sha256,
        "eval_dataset_content_sha256": eval_suite_control.dataset_content_sha256,
        "eval_unique_task_count": len(eval_suite_control.manifest.tasks),
        "eval_episode_count": args.eval_task_count,
        "eval_selection": args.eval_selection,
        "eval_schedule_path": args.eval_schedule,
        "eval_schedule_sha256": eval_suite_control.evaluation_schedule_sha256,
        "rollout_config": rollout_config,
        "optimizer_config_sha256": optimizer_config_sha256,
        "compute_apps_before_launch": list(compute_apps),
    }
    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preflight_sha256 = _sha256_file(preflight_path)
    event_logger.emit(
        "preflight_written",
        preflight_path=str(preflight_path),
        preflight_sha256=preflight_sha256,
        preflight_only=bool(args.preflight_only),
    )
    if args.preflight_only:
        return preflight_path
    assert_exclusive_gpu(
        device=args.device,
        allow_shared_gpu=args.allow_shared_gpu,
        compute_apps=compute_apps,
    )

    # Instantiate once to bind the exact initialized LoRA weights. The second
    # arm is constructed after freeing the first and must reproduce this hash.
    event_logger.emit(
        "policy_load_started",
        arm_id=control_arm_id,
        policy_backend=asdict(policy_backend),
        requested_device=args.device,
    )
    control_policy = _build_runtime_policy(args, policy_config)
    policy_cleanup.callback(_close_runtime_policy, control_policy)
    initial_checkpoint_sha256 = control_policy.checkpoint_sha256
    if phasepool_run:
        phasepool_calibration_admission = load_phasepool_calibration_admission(
            args.phasepool_calibration_receipt,
            expected_policy_checkpoint_sha256=initial_checkpoint_sha256,
            expected_gamma=float(args.gamma),
        )
        event_logger.emit(
            "phasepool_calibration_admission_verified",
            receipt_path=str(Path(args.phasepool_calibration_receipt).resolve()),
            report_sha256=phasepool_calibration_admission["report_sha256"],
            config_sha256=phasepool_calibration_admission["config_sha256"],
            policy_checkpoint_sha256=initial_checkpoint_sha256,
        )
    event_logger.emit(
        "policy_loaded",
        arm_id=control_arm_id,
        policy_checkpoint_sha256=initial_checkpoint_sha256,
        adapter_initial_delta_zero=control_policy.adapter_initial_delta_zero,
        requested_device=args.device,
    )
    if full_eval_only:
        optimizer_object_present = getattr(control_policy, "optimizer", None) is not None
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in control_policy.model.parameters()
            if parameter.requires_grad
        )
        if optimizer_object_present or trainable_parameter_count:
            raise AgenticRLP0ConfigError(
                "full evaluation policy must be optimizer-free and fully frozen"
            )
        evaluation_trace_path = output_dir / "evaluation_trace.jsonl"
        evaluation_trace_path.touch(exist_ok=False)
        trace_sequence = 0

        def write_evaluation_transition(payload: Mapping[str, Any]) -> None:
            nonlocal trace_sequence
            row = {
                "run_id": event_logger.run_id,
                "sequence": trace_sequence,
                **payload,
            }
            with evaluation_trace_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n"
                )
                handle.flush()
            trace_sequence += 1

        event_logger.emit(
            "full_evaluation_started",
            arm_id=control_arm_id,
            policy_checkpoint_sha256=initial_checkpoint_sha256,
            unique_task_count=len(eval_suite_control.manifest.tasks),
            episode_count=args.eval_task_count,
            max_episode_steps=args.max_episode_steps,
            eval_seed=args.eval_seed,
            transition_verification="none_success_metric_only",
            evaluation_trace_path=str(evaluation_trace_path),
        )
        evaluation = eval_suite_control.evaluate_all(
            control_policy,
            EvaluationRequest(
                seed=args.eval_seed,
                max_episode_steps=args.max_episode_steps,
            ),
            progress_callback=lambda payload: event_logger.emit(
                "full_evaluation_progress",
                arm_id=control_arm_id,
                **payload,
            ),
            transition_callback=write_evaluation_transition,
        )
        if evaluation.task_count != args.eval_task_count:
            raise RuntimeError(
                "evaluation episode count differs from the structured dataset contract"
            )
        evaluation_trace_sha256 = _sha256_file(evaluation_trace_path)
        event_logger.emit(
            "full_evaluation_completed",
            arm_id=control_arm_id,
            evaluation_trace_path=str(evaluation_trace_path),
            evaluation_trace_rows=trace_sequence,
            evaluation_trace_sha256=evaluation_trace_sha256,
            **asdict(evaluation),
        )
        result_path = output_dir / "full_evaluation_result.json"
        result_path.write_text(
            json.dumps(
                {
                    "schema": "lh_agentic_rl_full_evaluation/v1",
                    "claim_scope": "baseline_evaluation_only_no_training",
                    "run_id": event_logger.run_id,
                    "preflight_path": str(preflight_path),
                    "preflight_sha256": preflight_sha256,
                    "experiment_id": str(getattr(args, "experiment_id", "agentic_rl_full_eval")),
                    "base_checkpoint_sha256": base_checkpoint_sha256,
                    "policy_checkpoint_sha256": initial_checkpoint_sha256,
                    "adapter_initial_delta_zero": (control_policy.adapter_initial_delta_zero),
                    "optimizer_object_present": optimizer_object_present,
                    "trainable_parameter_count": trainable_parameter_count,
                    "eval_manifest_sha256": eval_suite_control.manifest_sha256,
                    "eval_dataset_content_sha256": (eval_suite_control.dataset_content_sha256),
                    "selection": args.eval_selection,
                    "generation_micro_batch_size": (policy_config.generation_micro_batch_size),
                    "policy_backend": asdict(policy_backend),
                    "training_backend_preflight": training_backend_preflight,
                    "rollout_config": rollout_config,
                    "rollout_config_sha256": rollout_config_sha256,
                    "evaluation_schedule_sha256": (eval_suite_control.evaluation_schedule_sha256),
                    "eval_seed": args.eval_seed,
                    "max_episode_steps": args.max_episode_steps,
                    "transition_verification": "none_success_metric_only",
                    "evaluation_trace_path": str(evaluation_trace_path),
                    "evaluation_trace_rows": trace_sequence,
                    "evaluation_trace_sha256": evaluation_trace_sha256,
                    "evaluation": asdict(evaluation),
                    "source_actor_parity": asdict(actor_parity),
                    "source_evaluation_schedule_audit": (
                        asdict(evaluation_schedule_audit)
                        if evaluation_schedule_audit is not None
                        else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _close_runtime_policy(control_policy)
        return result_path
    common = {
        "policy_backend": policy_backend,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "train_task_suite": train_suite_control.component,
        "train_manifest_sha256": train_suite_control.manifest_sha256,
        "eval_task_suite": eval_suite_control.component,
        "eval_manifest_sha256": eval_suite_control.manifest_sha256,
        "rl_method": control_method.component,
        "optimizer_config_sha256": optimizer_config_sha256,
        "rollout_config_sha256": rollout_config_sha256,
        "prompt_template_sha256": canonical_sha256(
            {
                "actor": args.resolved_experiment["actor"],
                "actor_awareness_mode": args.actor_awareness_mode,
                "actor_context_extension_revision": (
                    CONTEXT_EXTENSION_PLACEMENT_REVISION
                    if args.actor_awareness_mode != "none"
                    else "none"
                ),
            }
        ),
        "seed_schedule_sha256": canonical_sha256(list(rollout_seeds)),
    }
    control_arm = ExperimentArm(
        arm_id=control_arm_id,
        decomposition=control_decomposition.component,
        credit_assigner=control_method.credit_assigner,
        **common,
    )
    treatment_arm = ExperimentArm(
        arm_id=treatment_arm_id,
        decomposition=treatment_decomposition.component,
        credit_assigner=treatment_method.credit_assigner,
        **common,
    )
    paired_lock = build_paired_runtime_lock(control_arm, treatment_arm)
    (output_dir / "paired_runtime_lock.json").write_text(
        json.dumps(asdict(paired_lock), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runner = AgenticRLExperimentRunner()
    shared_rollout_overrides = None
    shared_phasepool_preflight = None
    if phasepool_run:
        shared_rollout = collect_on_policy_rollout(
            policy=control_policy,
            train_suite=train_suite_control,
            request=RolloutRequest(
                update_index=0,
                seed=rollout_seeds[0],
                group_size=args.group_size,
                max_episode_steps=args.max_episode_steps,
                rollout_config_sha256=rollout_config_sha256,
            ),
            expected_manifest_sha256=train_suite_control.manifest_sha256,
            expected_checkpoint_sha256=initial_checkpoint_sha256,
        )
        shared_phasepool_preflight = _preflight_shared_phasepool_batch(
            shared_rollout.records,
            control_decomposition=control_decomposition,
            treatment_decomposition=treatment_decomposition,
            control_method=control_method,
            treatment_method=treatment_method,
        )
        shared_rollout_overrides = {0: shared_rollout}
        event_logger.emit(
            "phasepool_shared_frozen_batch_admitted",
            **shared_phasepool_preflight,
        )
    control = runner.run_arm(
        ArmRunPlan(
            arm=control_arm,
            updates=args.updates,
            rollout_seeds=rollout_seeds,
            eval_seed=args.eval_seed,
            group_size=args.group_size,
            max_episode_steps=args.max_episode_steps,
        ),
        policy=control_policy,
        train_suite=train_suite_control,
        eval_suite=eval_suite_control,
        decomposition=control_decomposition,
        method=control_method,
        output_dir=output_dir / "control",
        event_logger=event_logger,
        training_backend=(
            control_policy
            if getattr(args, "training_backend", "local_transformers") == "verl"
            else None
        ),
        rollout_overrides=shared_rollout_overrides,
        dynamic_sampling_config=DynamicSamplingConfig(
            enabled=bool(getattr(args, "dynamic_sampling_enabled", False)),
            min_informative_groups=int(getattr(args, "dynamic_sampling_min_informative_groups", 1)),
            max_retries=int(getattr(args, "dynamic_sampling_max_retries", 3)),
        ),
    )
    _close_runtime_policy(control_policy)
    del control_policy
    _release_accelerator_cache()

    event_logger.emit(
        "policy_load_started",
        arm_id=treatment_arm_id,
        policy_backend=asdict(policy_backend),
        requested_device=args.device,
    )
    treatment_policy = _build_runtime_policy(args, policy_config)
    policy_cleanup.callback(_close_runtime_policy, treatment_policy)
    if treatment_policy.checkpoint_sha256 != initial_checkpoint_sha256:
        raise RuntimeError("treatment policy did not reproduce the locked initial checkpoint")
    event_logger.emit(
        "policy_loaded",
        arm_id=treatment_arm_id,
        policy_checkpoint_sha256=treatment_policy.checkpoint_sha256,
        requested_device=args.device,
    )
    train_suite_treatment = _suite(
        role="train_task_suite",
        manifest=train_manifest,
        data_root=data_root,
        artifacts_dir=output_dir / "treatment" / "environment",
        source_revision=args.source_revision,
        train_task_groups=args.train_task_groups,
        train_task_selection=args.train_task_selection,
        eval_tasks_per_horizon=args.eval_tasks_per_horizon,
        history_length=args.history_length,
        train_temperature=args.train_temperature,
        eval_temperature=args.eval_temperature,
        actor_protocol_id=args.actor_protocol_id,
        show_admissible_actions=bool(getattr(args, "show_admissible_actions", True)),
        model_io_profile_id=_bound_model_io_profile_id(args),
        actor_awareness_mode=args.actor_awareness_mode,
        evaluation_schedule_path=None,
        evaluation_selection="all_manifest_tasks_exactly_once",
        branch_factory=branch_factory,
    )
    eval_suite_treatment = _suite(
        role="eval_task_suite",
        manifest=eval_manifest,
        data_root=data_root,
        artifacts_dir=output_dir / "treatment" / "evaluation_environment",
        source_revision=args.source_revision,
        train_task_groups=args.train_task_groups,
        train_task_selection=args.train_task_selection,
        eval_tasks_per_horizon=args.eval_tasks_per_horizon,
        history_length=args.history_length,
        train_temperature=args.train_temperature,
        eval_temperature=args.eval_temperature,
        actor_protocol_id=args.actor_protocol_id,
        show_admissible_actions=bool(getattr(args, "show_admissible_actions", True)),
        model_io_profile_id=_bound_model_io_profile_id(args),
        actor_awareness_mode=args.actor_awareness_mode,
        evaluation_schedule_path=args.eval_schedule,
        evaluation_selection=args.eval_selection,
        branch_factory=branch_factory,
    )
    if (
        train_suite_treatment.dataset_content_sha256 != train_suite_control.dataset_content_sha256
        or eval_suite_treatment.dataset_content_sha256 != eval_suite_control.dataset_content_sha256
    ):
        raise RuntimeError("treatment task suites differ from preflight data content locks")
    event_logger.emit(
        "treatment_task_suites_verified",
        train_dataset_content_sha256=train_suite_treatment.dataset_content_sha256,
        eval_dataset_content_sha256=eval_suite_treatment.dataset_content_sha256,
    )
    treatment = runner.run_arm(
        ArmRunPlan(
            arm=treatment_arm,
            updates=args.updates,
            rollout_seeds=rollout_seeds,
            eval_seed=args.eval_seed,
            group_size=args.group_size,
            max_episode_steps=args.max_episode_steps,
        ),
        policy=treatment_policy,
        train_suite=train_suite_treatment,
        eval_suite=eval_suite_treatment,
        decomposition=treatment_decomposition,
        method=treatment_method,
        output_dir=output_dir / "treatment",
        event_logger=event_logger,
        training_backend=(
            treatment_policy
            if getattr(args, "training_backend", "local_transformers") == "verl"
            else None
        ),
        rollout_overrides=shared_rollout_overrides,
        dynamic_sampling_config=DynamicSamplingConfig(
            enabled=bool(getattr(args, "dynamic_sampling_enabled", False)),
            min_informative_groups=int(getattr(args, "dynamic_sampling_min_informative_groups", 1)),
            max_retries=int(getattr(args, "dynamic_sampling_max_retries", 3)),
        ),
    )
    rollout_analyses = tuple(
        analyze_rollout_artifact(
            output_dir / "treatment" / "rollouts" / f"update-{update_index:04d}-annotated.jsonl",
            paired_path=(
                output_dir / "control" / "rollouts" / f"update-{update_index:04d}-raw.jsonl"
            ),
            output_path=(
                output_dir / "analysis" / f"update-{update_index:04d}-rollout-analysis.json"
            ),
        )
        for update_index in range(args.updates)
    )
    initial_rollout_pair_exact = bool(
        rollout_analyses
        and rollout_analyses[0]["paired_rollout"]["exact_credit_only_rollout_match"]
    )
    event_logger.emit(
        "paired_rollout_analysis_completed",
        update_count=len(rollout_analyses),
        initial_rollout_pair_exact=initial_rollout_pair_exact,
        analysis_paths=[
            str(output_dir / "analysis" / f"update-{update_index:04d}-rollout-analysis.json")
            for update_index in range(args.updates)
        ],
    )
    decision = (
        assess_phasepool_one_update_decision(
            control,
            treatment,
            planned_updates=args.updates,
            initial_rollout_pair_exact=initial_rollout_pair_exact,
        )
        if phasepool_run
        else assess_p0_decision(
            control,
            treatment,
            planned_updates=args.updates,
            initial_rollout_pair_exact=initial_rollout_pair_exact,
        )
    )
    event_logger.emit(
        ("phasepool_one_update_decision_assessed" if phasepool_run else "p0_decision_assessed"),
        **decision,
    )
    result = {
        "schema": ("lh_agentic_rl_phasepool_one_update/v1" if phasepool_run else P0_RESULT_SCHEMA),
        "claim_scope": decision["claim_scope"],
        "paired_runtime_lock": asdict(paired_lock),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "train_dataset_content_sha256": train_suite_control.dataset_content_sha256,
        "eval_dataset_content_sha256": eval_suite_control.dataset_content_sha256,
        "initial_policy_checkpoint_sha256": initial_checkpoint_sha256,
        "training_backend_preflight": training_backend_preflight,
        "rollout_config": rollout_config,
        "control": _result_summary(control),
        "treatment": _result_summary(treatment),
        "rollout_analyses": list(rollout_analyses),
        "shared_phasepool_preflight": shared_phasepool_preflight,
        "phasepool_calibration_admission": phasepool_calibration_admission,
        "success_rate_delta": _success_rate(treatment) - _success_rate(control),
        "decision": decision,
        "next_gate": decision["next_action"],
    }
    result_path = output_dir / (
        "phasepool_one_update_result.json" if phasepool_run else "p0_result.json"
    )
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _close_runtime_policy(treatment_policy)
    return result_path


def _preflight_training_backend(
    args: argparse.Namespace,
    *,
    policy_config: TransformersPeftPolicyConfig,
) -> dict[str, Any]:
    backend = str(getattr(args, "training_backend", "local_transformers"))
    if backend == "local_transformers":
        return {
            "requested_backend": "local_transformers",
            "actual_backend": "local_transformers",
            "execution_mode": "in_process",
            "parity_receipt_required": False,
        }
    if backend != "verl":
        raise AgenticRLP0ConfigError("unsupported training backend")
    from lh_platform.training.backends.verl_full_model_parity import (
        VERL_FULL_MODEL_PARITY_SCHEMA,
    )
    from lh_platform.training.backends.verl_source import (
        VERL_PINNED_SOURCE_CONTRACT,
        verify_git_source,
    )

    source_root = Path(str(getattr(args, "verl_source_root", ""))).resolve()
    parity_path = Path(str(getattr(args, "verl_parity_receipt", ""))).resolve()
    verified = verify_git_source(source_root, VERL_PINNED_SOURCE_CONTRACT)
    try:
        receipt = json.loads(parity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AgenticRLP0ConfigError("cannot read veRL full-model parity receipt") from error
    expected_policy_sha256 = canonical_sha256(asdict(policy_config))
    verl_identity = receipt.get("verl_backend")
    if (
        receipt.get("schema") != VERL_FULL_MODEL_PARITY_SCHEMA
        or receipt.get("passed") is not True
        or receipt.get("checkpoint_identity_exact") is not True
        or receipt.get("policy_config_sha256") != expected_policy_sha256
        or not isinstance(verl_identity, Mapping)
        or verl_identity.get("source_contract_sha256") != verified.contract_sha256
    ):
        raise AgenticRLP0ConfigError(
            "veRL live rollout is blocked by missing or mismatched full-model parity"
        )
    return {
        "requested_backend": "verl",
        "actual_backend": "verl",
        "execution_mode": "isolated_process",
        "source_contract_sha256": verified.contract_sha256,
        "source_commit_sha": verified.commit_sha,
        "source_tree_sha": verified.tree_sha,
        "parity_receipt_path": str(parity_path),
        "parity_receipt_sha256": _sha256_file(parity_path),
        "policy_config_sha256": expected_policy_sha256,
        "full_model_parity_passed": True,
    }


def _build_runtime_policy(
    args: argparse.Namespace,
    policy_config: TransformersPeftPolicyConfig,
):
    backend = str(getattr(args, "training_backend", "local_transformers"))
    if backend == "local_transformers":
        return TransformersPeftPolicy(policy_config)
    if backend == "verl":
        from lh_platform.training.backends.verl import VerLBackend

        return VerLBackend(
            source_root=Path(args.verl_source_root),
            policy_payload={
                "kind": "transformers_peft",
                "config": asdict(policy_config),
            },
        )
    raise AgenticRLP0ConfigError("unsupported training backend")


def _close_runtime_policy(policy: Any) -> None:
    close = getattr(policy, "close", None)
    if callable(close):
        close()


def _bound_model_io_profile_id(args: argparse.Namespace) -> str | None:
    """Expose a profile ID only when the policy has an explicit v2 profile.

    Legacy model documents remain readable, but their inferred profile is not
    an executable contract and must not make task-suite requests look bound to
    a renderer that the policy deliberately did not load.
    """

    if bool(getattr(args, "legacy_model_io_inferred", True)):
        return None
    profile_id = getattr(args, "model_io_profile_id", None)
    return str(profile_id) if profile_id is not None else None


def assess_p0_decision(
    control: Any,
    treatment: Any,
    *,
    planned_updates: int,
    initial_rollout_pair_exact: bool = True,
) -> dict[str, Any]:
    """Turn one paired run into a fail-closed, predeclared next-step decision."""

    blocked: list[str] = []
    if planned_updates < 1:
        blocked.append("invalid_planned_update_count")
    if control.updates_completed != planned_updates:
        blocked.append("control_update_count_incomplete")
    if treatment.updates_completed != planned_updates:
        blocked.append("treatment_update_count_incomplete")
    if not initial_rollout_pair_exact:
        blocked.append("initial_credit_only_rollout_pair_mismatch")
    if any(item.trainable_parameter_delta_l2 <= 0 for item in control.update_results):
        blocked.append("control_zero_parameter_update")
    if any(item.trainable_parameter_delta_l2 <= 0 for item in treatment.update_results):
        blocked.append("treatment_zero_parameter_update")

    trace_gates = tuple(
        diagnostics.get("oracle_subproblem_trace_gate")
        for diagnostics in treatment.update_diagnostics
        if isinstance(diagnostics, Mapping)
    )
    valid_trace_gates = tuple(gate for gate in trace_gates if isinstance(gate, Mapping))
    if len(valid_trace_gates) != treatment.updates_completed:
        blocked.append("treatment_trace_gate_missing")
    elif any(float(gate.get("annotation_coverage", 0.0)) != 1.0 for gate in valid_trace_gates):
        blocked.append("treatment_annotation_coverage_incomplete")
    if not any(gate.get("passed") is True for gate in valid_trace_gates):
        blocked.append("decomposition_credit_mechanism_not_triggered")

    control_eval = control.evaluation
    treatment_eval = treatment.evaluation
    if control_eval.task_count <= 0 or treatment_eval.task_count != control_eval.task_count:
        blocked.append("paired_evaluation_task_count_mismatch")
    control_long = _horizon_rate(control_eval.success_by_horizon, "long")
    treatment_long = _horizon_rate(treatment_eval.success_by_horizon, "long")
    if control_long is None or treatment_long is None:
        blocked.append("long_horizon_evaluation_missing")
        long_delta = None
    else:
        long_delta = treatment_long - control_long
    overall_delta = (
        treatment_eval.success_count / treatment_eval.task_count
        - control_eval.success_count / control_eval.task_count
        if treatment_eval.task_count > 0 and control_eval.task_count > 0
        else None
    )

    if blocked:
        status = "mechanism_smoke_blocked"
        next_action = "inspect_blocked_by_before_any_more_training"
        claim_scope = "no_performance_claim"
    elif planned_updates < 5:
        status = "eligible_for_five_update_go_no_go"
        next_action = "run_five_update_same_seed_go_no_go"
        claim_scope = "mechanism_smoke_only_no_performance_claim"
    elif (
        long_delta is not None
        and overall_delta is not None
        and long_delta > 0
        and overall_delta >= 0
    ):
        status = "preliminary_go"
        next_action = "run_multi_seed_confirmatory_ablation"
        claim_scope = "preliminary_signal_not_effectiveness"
    elif (
        long_delta is not None
        and overall_delta is not None
        and (long_delta < 0 or overall_delta < 0)
    ):
        status = "preliminary_no_go"
        next_action = "stop_and_diagnose_credit_mechanism"
        claim_scope = "preliminary_negative_signal_not_effectiveness"
    else:
        status = "preliminary_inconclusive"
        next_action = "do_not_scale_without_more_discriminating_evaluation"
        claim_scope = "inconclusive_not_effectiveness"
    return {
        "schema": "lh_agentic_rl_p0_decision/v1",
        "status": status,
        "claim_scope": claim_scope,
        "performance_claim_allowed": False,
        "planned_updates": planned_updates,
        "mechanism_triggered_updates": sum(
            gate.get("passed") is True for gate in valid_trace_gates
        ),
        "control_success_rate": (
            control_eval.success_count / control_eval.task_count
            if control_eval.task_count > 0
            else None
        ),
        "treatment_success_rate": (
            treatment_eval.success_count / treatment_eval.task_count
            if treatment_eval.task_count > 0
            else None
        ),
        "overall_success_rate_delta": overall_delta,
        "control_long_horizon_success_rate": control_long,
        "treatment_long_horizon_success_rate": treatment_long,
        "long_horizon_success_rate_delta": long_delta,
        "blocked_by": tuple(dict.fromkeys(blocked)),
        "next_action": next_action,
    }


def assess_phasepool_one_update_decision(
    control: Any,
    treatment: Any,
    *,
    planned_updates: int,
    initial_rollout_pair_exact: bool = True,
) -> dict[str, Any]:
    """Admit only a completed estimator-mechanism screen, never an effect claim."""

    blocked: list[str] = []
    if planned_updates != 1:
        blocked.append("phasepool_screen_requires_exactly_one_update")
    if control.updates_completed != planned_updates:
        blocked.append("control_update_count_incomplete")
    if treatment.updates_completed != planned_updates:
        blocked.append("treatment_update_count_incomplete")
    if not initial_rollout_pair_exact:
        blocked.append("initial_credit_only_rollout_pair_mismatch")
    # Required prepare_update diagnostics contract (emitted by G2POFamilyMethod):
    #   diagnostics["outcome_variation"]: bool
    #   diagnostics["credit_estimator"]["metrics"]["coverage"]: float (== 1.0 to pass)
    # Absence is blocked as "*_unknown", not conflated with an incomplete value.
    for arm_name, arm in (("control", control), ("treatment", treatment)):
        for diagnostics in arm.update_diagnostics:
            if not isinstance(diagnostics, Mapping):
                blocked.append(f"{arm_name}_update_diagnostics_unknown")
                continue
            estimator = diagnostics.get("credit_estimator")
            if not isinstance(estimator, Mapping):
                blocked.append(f"{arm_name}_credit_estimator_diagnostics_unknown")
            else:
                metrics = estimator.get("metrics")
                if not isinstance(metrics, Mapping) or "coverage" not in metrics:
                    blocked.append(f"{arm_name}_cross_fitted_prediction_coverage_unknown")
                elif metrics.get("coverage") != 1.0:
                    blocked.append(f"{arm_name}_cross_fitted_prediction_coverage_incomplete")
            if "outcome_variation" not in diagnostics:
                blocked.append(f"{arm_name}_outcome_variation_unknown")
            elif diagnostics.get("outcome_variation") is not True:
                blocked.append(f"{arm_name}_outcome_variation_missing")
    status = "blocked_before_effect_interpretation" if blocked else "one_update_screen_completed"
    return {
        "schema": "lh_phasepool_one_update_decision/v1",
        "status": status,
        "claim_scope": "one_update_mechanism_screen_not_policy_improvement",
        "performance_claim_allowed": False,
        "planned_updates": planned_updates,
        "blocked_by": tuple(dict.fromkeys(blocked)),
        "next_action": (
            "inspect_blocked_by_before_any_more_training"
            if blocked
            else "run_preregistered_multi_seed_only_after_frozen_calibration_gate"
        ),
    }


def _preflight_shared_phasepool_batch(
    records: tuple[Any, ...],
    *,
    control_decomposition: Any,
    treatment_decomposition: Any,
    control_method: Any,
    treatment_method: Any,
) -> dict[str, Any]:
    """Prove the one-update screen changes only credit before either update."""

    control_records = tuple(control_decomposition.annotate(records))
    treatment_records = tuple(treatment_decomposition.annotate(records))
    if control_records != treatment_records:
        raise AgenticRLP0ConfigError("PhasePool arms produced different verifier annotations")
    control_update = control_method.prepare_update(control_records)
    treatment_update = treatment_method.prepare_update(treatment_records)
    trace_gate = treatment_update.diagnostics.get("oracle_subproblem_trace_gate")
    if not isinstance(trace_gate, Mapping) or trace_gate.get("passed") is not True:
        raise AgenticRLP0ConfigError(
            "PhasePool shared rollout did not exercise a verified phase collision"
        )
    control_targets = {
        (target.traj_uid, target.step_idx): target for target in control_update.targets
    }
    treatment_targets = {
        (target.traj_uid, target.step_idx): target for target in treatment_update.targets
    }
    if control_targets.keys() != treatment_targets.keys():
        raise AgenticRLP0ConfigError("PhasePool target identities do not align")
    changed = sum(
        abs(control_targets[identity].advantage - treatment_targets[identity].advantage) > 1.0e-12
        for identity in control_targets
    )
    if changed == 0:
        raise AgenticRLP0ConfigError(
            "PhasePool shared rollout produced no credit-target difference"
        )
    return {
        "schema": "lh_phasepool_shared_frozen_batch_preflight/v1",
        "transition_count": len(records),
        "changed_target_count": changed,
        "trace_gate": dict(trace_gate),
        "control_target_sha256": canonical_sha256(
            [asdict(target) for target in control_update.targets]
        ),
        "treatment_target_sha256": canonical_sha256(
            [asdict(target) for target in treatment_update.targets]
        ),
    }


def _horizon_rate(
    by_horizon: Mapping[str, Mapping[str, int]],
    horizon: str,
) -> float | None:
    row = by_horizon.get(horizon)
    if not isinstance(row, Mapping):
        return None
    tasks = int(row.get("tasks", 0))
    successes = int(row.get("successes", 0))
    if tasks <= 0 or successes < 0 or successes > tasks:
        return None
    return successes / tasks


def _validate_component_selections(args: argparse.Namespace) -> None:
    static_components = {
        "policy_backend": P0_COMPONENT_IDS["policy_backend"],
        "task_suite": P0_COMPONENT_IDS["task_suite"],
        "rl_method": P0_COMPONENT_IDS["rl_method"],
    }
    mismatches = {
        name: {
            "expected": expected,
            "configured": str(getattr(args, f"{name}_id", expected)),
        }
        for name, expected in static_components.items()
        if str(getattr(args, f"{name}_id", expected)) != expected
    }
    if mismatches:
        raise AgenticRLP0ConfigError(
            "runtime component implementation does not match configured IDs: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _validate_runtime_component_instances(
    args: argparse.Namespace,
    *,
    policy_backend: Any,
    train_suite: AlfWorldNativeTaskSuite,
    control_method: Any,
    treatment_method: Any,
    control_decomposition: Any,
    treatment_decomposition: Any,
) -> None:
    actual = {
        "policy_backend": policy_backend.name,
        "task_suite": train_suite.component.name,
        "rl_method": control_method.component.name,
        "control_decomposition": control_decomposition.component.name,
        "treatment_decomposition": treatment_decomposition.component.name,
        "control_credit_assigner": control_method.credit_assigner.name,
        "treatment_credit_assigner": treatment_method.credit_assigner.name,
    }
    configured = {name: str(getattr(args, f"{name}_id")) for name in P0_COMPONENT_IDS}
    if getattr(args, "execution_mode", "paired_train_eval") == "full_eval_only":
        configured["policy_backend"] = "transformers_frozen_causal_lm"
    if actual != configured:
        raise AgenticRLP0ConfigError(
            "resolved runtime components differ from configured IDs: "
            + json.dumps(
                {"configured": configured, "actual": actual},
                sort_keys=True,
            )
        )


def _runtime_debug_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in (
        "torch",
        "transformers",
        "peft",
        "textworld",
        "alfworld",
        "PyYAML",
    ):
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = "not_installed"
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": packages,
    }


def _suite(
    *,
    role: str,
    manifest: Path,
    data_root: Path,
    artifacts_dir: Path,
    source_revision: str,
    train_task_groups: int,
    train_task_selection: str,
    eval_tasks_per_horizon: int,
    history_length: int,
    train_temperature: float,
    eval_temperature: float,
    actor_protocol_id: str,
    show_admissible_actions: bool = True,
    model_io_profile_id: str | None,
    actor_awareness_mode: str,
    evaluation_schedule_path: str | None,
    evaluation_selection: str,
    branch_factory: OfficialAlfWorldBranchFactory,
) -> AlfWorldNativeTaskSuite:
    return AlfWorldNativeTaskSuite(
        AlfWorldSuiteConfig(
            role=role,
            manifest_path=str(manifest),
            data_root=str(data_root),
            artifacts_dir=str(artifacts_dir),
            source_revision=source_revision,
            train_task_groups_per_update=train_task_groups,
            train_task_selection=train_task_selection,
            eval_tasks_per_horizon=eval_tasks_per_horizon,
            history_length=history_length,
            train_temperature=train_temperature,
            eval_temperature=eval_temperature,
            actor_protocol_id=actor_protocol_id,
            show_admissible_actions=show_admissible_actions,
            model_io_profile_id=model_io_profile_id,
            actor_awareness_mode=actor_awareness_mode,
            evaluation_schedule_path=evaluation_schedule_path,
            evaluation_selection=evaluation_selection,
        ),
        branch_factory=branch_factory,
    )


def _toolbench_suite(
    *,
    role: str,
    tasks: list,
    executor: Any,
    artifacts_dir: Path,
    data_root: str,
    test_sets: tuple[str, ...],
    tasks_per_update: int,
    eval_task_count: int,
    max_episode_steps: int,
    history_length: int,
    train_temperature: float,
    eval_temperature: float,
    model_io_profile_id: str | None,
) -> ToolBenchNativeTaskSuite:
    from lh_platform.training.toolbench_agentic_suite import (
        ToolBenchNativeTaskSuite,
        ToolBenchSuiteConfig,
    )

    return ToolBenchNativeTaskSuite(
        ToolBenchSuiteConfig(
            role=role,
            data_root=data_root,
            artifacts_dir=str(artifacts_dir),
            test_sets=test_sets,
            tasks_per_update=tasks_per_update,
            eval_task_count=eval_task_count,
            max_episode_steps=max_episode_steps,
            history_length=history_length,
            train_temperature=train_temperature,
            eval_temperature=eval_temperature,
            model_io_profile_id=model_io_profile_id,
        ),
        tasks=tasks,
        executor=executor,
    )


def _result_summary(result: Any) -> dict[str, Any]:
    evaluation = result.evaluation
    return {
        "updates_completed": result.updates_completed,
        "checkpoint_sha256": result.checkpoint.sha256,
        "checkpoint_manifest": result.checkpoint.path,
        "evaluation": asdict(evaluation),
        "update_results": [asdict(item) for item in result.update_results],
        "update_diagnostics": list(result.update_diagnostics),
        "receipt_path": result.receipt_path,
    }


def _success_rate(result: Any) -> float:
    return result.evaluation.success_count / result.evaluation.task_count


def _release_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _git_head(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("source repository did not return a full lowercase Git revision")
    return revision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one structured G2PO-TN versus Verifier-Refined G2PO experiment."
    )
    parser.add_argument("config_path")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate source, manifests, components, and hashes without loading a model or GPU.",
    )
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Explicitly allow launch while another compute process owns GPU memory.",
    )
    parser.add_argument(
        "--full-eval-only",
        action="store_true",
        help=(
            "Compatibility assertion for configs whose runtime.mode is full_eval_only; "
            "the structured config, not this flag, determines execution."
        ),
    )
    parser.add_argument(
        "--generation-micro-batch-size",
        type=int,
        default=16,
        help=(
            "Batch unrestricted Transformers generation on one device; this is a "
            "logged platform execution factor, not vLLM bitwise parity."
        ),
    )
    parser.add_argument(
        "--phasepool-calibration-receipt",
        help=(
            "Required for PhasePool online updates; must be a Go receipt from "
            "the two-dataset frozen calibration runner."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    bindings = build_parser().parse_args(argv)
    args = namespace_from_experiment_config(
        bindings.config_path,
        model_path=bindings.model_path,
        source_root=bindings.source_root,
        data_root=bindings.data_root,
        output_dir=bindings.output_dir,
        device=bindings.device,
        preflight_only=bindings.preflight_only,
        allow_shared_gpu=bindings.allow_shared_gpu,
        phasepool_calibration_receipt=bindings.phasepool_calibration_receipt,
    )
    config_full_eval = args.execution_mode == "full_eval_only"
    if bindings.full_eval_only and not config_full_eval:
        raise AgenticRLP0ConfigError(
            "--full-eval-only cannot override a paired_train_eval experiment"
        )
    args.full_eval_only = config_full_eval
    args.generation_micro_batch_size = bindings.generation_micro_batch_size
    result = run_p0(args)
    print(result)


if __name__ == "__main__":
    main()
