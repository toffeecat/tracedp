"""Resolve portable agentic-RL experiment documents into one audited run contract.

The public seam is deliberately small: load a composition, then bind local
paths and device choices.  Dataset, model, source, and method documents remain
portable; machine-specific paths never become part of their scientific hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.credit_estimators.phasepool_study import (
    PHASEPOOL_STUDY_SCHEMA,
    load_phasepool_study_config,
)
from lh_platform.training.g2po.actor_parity import (
    G2PO_ACTOR_PARITY_SCOPE,
    MODEL_COMPATIBLE_ACTOR_PARITY_SCOPE,
    QWEN_PROMPT_INJECTED_THINKING_ACTOR_PARITY_SCOPE,
)
from lh_platform.training.g2po.actor_protocol import (
    G2PO_ALFWORLD_ACTOR_PROTOCOL_ID,
    MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_V2_ID,
    QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    AlfWorldResponseContract,
)
from lh_platform.training.g2po.source_parity import G2PO_SOURCE_PARITY_SCOPE
from lh_platform.training.model_io import (
    ModelIOContractError,
    ModelIOProfile,
    qwen3_instruct_2507_profile,
)
from lh_platform.training.subproblem_study import load_oracle_subproblem_study_config

DATASET_SCHEMA = "lh_agentic_rl_dataset/v1"
MODEL_SCHEMA = "lh_agentic_rl_model/v1"
MODEL_SCHEMA_V2 = "lh_agentic_rl_model/v2"
SOURCE_SCHEMA = "lh_agentic_rl_source/v1"
ACTOR_SCHEMA = "lh_agentic_rl_actor/v1"
ACTOR_SCHEMA_V2 = "lh_agentic_rl_actor/v2"
METHOD_SCHEMA = "lh_agentic_rl_method/v1"
EXPERIMENT_SCHEMA = "lh_agentic_rl_experiment/v1"
RESOLVED_SCHEMA = "lh_agentic_rl_resolved_experiment/v1"
FACTORIAL_SCHEMA = "lh_agentic_rl_factorial_manifest/v1"
CONTROLLED_ACTOR_FACTORIAL_SCHEMA = "lh_alfworld_controlled_actor_credit_factorial/v1"

_SUPPORTED = {
    "policy_backend": {"transformers_peft_causal_lm"},
    "task_suite": {"alfworld_native", "toolbench_native"},
    "rl_method": {"g2po_family_clipped_pg"},
    "decomposition": {"none", "alfworld_fact_phases", "toolbench_none"},
    "credit_assigner": {
        "g2po_tn",
        "oracle_phase_refined_g2po_tn",
        "g2po_tn_loto",
        "phasepool_g2po",
    },
}
_CONTROL_STATE_FIELDS = ["anchor_observation"]
_TREATMENT_STATE_FIELDS = [
    "anchor_observation",
    "residual_goal_signature",
    "frontier_signature",
]
_PHASEPOOL_STATE_FIELDS = [
    "anchor_observation",
    "residual_goal_signature",
    "frontier_signature",
    "remaining_budget",
]
_ALLOWED_ALGORITHMIC_DIFFERENCES = [
    "credit.mode",
    "credit.state_identity.fields",
    "implementation.credit_assigner",
    "implementation.decomposition",
]
_PHASEPOOL_ALLOWED_ALGORITHMIC_DIFFERENCES = [
    "credit.estimator.family",
    "credit.estimator.key",
    "credit.estimator.pooling",
    "credit.mode",
    "credit.state_identity.fields",
    "implementation.credit_assigner",
]


class AgenticRLConfigError(ValueError):
    """Raised when a composed experiment is invalid or scientifically confounded."""


@dataclass(frozen=True)
class AgenticRLRuntimeBindings:
    """Machine-local values intentionally excluded from portable config files."""

    model_path: Path
    source_root: Path
    data_root: Path
    output_dir: Path
    device: str | None = None
    preflight_only: bool = False
    allow_shared_gpu: bool = False
    training_backend: str = "local_transformers"
    verl_source_root: Path | None = None
    verl_parity_receipt: Path | None = None
    phasepool_calibration_receipt: Path | None = None


@dataclass(frozen=True)
class ResolvedAgenticRLExperiment:
    config_path: Path
    workspace_root: Path
    payload: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class ResolvedAgenticRLFactorial:
    """A validated set of paired experiments spanning one declared factorial."""

    config_path: Path
    workspace_root: Path
    payload: Mapping[str, Any]
    sha256: str


def load_controlled_actor_factorial_manifest(
    config_path: str | Path,
) -> ResolvedAgenticRLFactorial:
    """Resolve the controlled actor x credit experiment without hidden recollection.

    The ordinary factorial owns model, dataset, actor, loss, and credit identity.
    This outer preregistration changes only the rollout start state and requires
    both credit arms within an awareness level to consume one frozen rollout.
    """

    from lh_platform.training.alfworld_controlled_aliasing import (
        load_alfworld_controlled_aliasing_recipe,
    )
    from lh_platform.training.dataset import sha256_file

    path = Path(config_path).resolve()
    workspace_root = _workspace_root(path)
    manifest = _load_yaml(path, label="controlled actor factorial")
    _expect_schema(
        manifest,
        CONTROLLED_ACTOR_FACTORIAL_SCHEMA,
        "controlled actor factorial",
    )
    _expect_exact_keys(
        manifest,
        {
            "schema",
            "id",
            "factorial_ref",
            "controlled_recipe_ref",
            "runtime",
            "update_gate",
            "claim_boundary",
        },
        "controlled actor factorial",
    )
    factorial_path = _resolve_workspace_ref(
        workspace_root,
        manifest.get("factorial_ref"),
        "controlled actor factorial_ref",
    )
    factorial = load_agentic_rl_factorial_manifest(factorial_path)
    recipe_path = _resolve_workspace_ref(
        workspace_root,
        manifest.get("controlled_recipe_ref"),
        "controlled actor controlled_recipe_ref",
    )
    recipe = load_alfworld_controlled_aliasing_recipe(recipe_path)

    runtime = _mapping(manifest, "runtime", label="controlled actor factorial")
    _expect_exact_keys(
        runtime,
        {
            "actor_awareness_modes",
            "credit_treatments",
            "samples_per_state",
            "max_episode_steps",
            "sampling_seed",
            "common_random_numbers_across_awareness",
            "freeze_rollout_within_awareness",
            "model_policy_required",
            "deterministic_fixture_controller_allowed_after_collision",
        },
        "controlled actor factorial.runtime",
    )
    expected_awareness = ["none", "oracle_goal_graph"]
    expected_credit = ["g2po_tn", "oracle_phase_refined_g2po_tn"]
    if runtime.get("actor_awareness_modes") != expected_awareness:
        raise AgenticRLConfigError(
            "controlled actor factorial must span the registered awareness levels"
        )
    if runtime.get("credit_treatments") != expected_credit:
        raise AgenticRLConfigError(
            "controlled actor factorial must span the registered credit treatments"
        )
    samples_per_state = runtime.get("samples_per_state")
    max_episode_steps = runtime.get("max_episode_steps")
    sampling_seed = runtime.get("sampling_seed")
    if (
        isinstance(samples_per_state, bool)
        or not isinstance(samples_per_state, int)
        or samples_per_state < 2
    ):
        raise AgenticRLConfigError(
            "controlled actor factorial requires at least two samples per state"
        )
    if (
        isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
        or max_episode_steps < 1
    ):
        raise AgenticRLConfigError("controlled actor factorial max_episode_steps must be positive")
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
        raise AgenticRLConfigError("controlled actor factorial sampling_seed must be an integer")
    required_runtime_flags = {
        "common_random_numbers_across_awareness": False,
        "freeze_rollout_within_awareness": True,
        "model_policy_required": True,
        "deterministic_fixture_controller_allowed_after_collision": False,
    }
    for field, expected in required_runtime_flags.items():
        if runtime.get(field) is not expected:
            detail = (
                "frozen rollout is required between credit arms"
                if field == "freeze_rollout_within_awareness"
                else f"controlled actor runtime flag {field!r} is unsafe"
            )
            raise AgenticRLConfigError(detail)

    registered_rollouts = {
        awareness: load_agentic_rl_experiment(workspace_root / reference["path"]).payload[
            "runtime"
        ]["rollout"]
        for awareness, reference in factorial.payload["paired_experiments"].items()
    }
    registered_experiments = {
        awareness: load_agentic_rl_experiment(workspace_root / reference["path"])
        for awareness, reference in factorial.payload["paired_experiments"].items()
    }
    if any(
        experiment.payload["source"]["revision"] != recipe.source_revision
        for experiment in registered_experiments.values()
    ):
        raise AgenticRLConfigError(
            "controlled actor recipe source revision differs from the registered factorial"
        )
    if any(
        int(rollout["max_episode_steps"]) != max_episode_steps
        for rollout in registered_rollouts.values()
    ):
        raise AgenticRLConfigError("controlled actor horizon differs from the registered factorial")

    update_gate = _mapping(
        manifest,
        "update_gate",
        label="controlled actor factorial",
    )
    legacy_gate = {
        "require_credit_target_difference": True,
        "require_action_only_clipped_policy_gradient": True,
        "require_same_initial_checkpoint": True,
        "on_no_credit_difference": "stop_before_policy_update",
    }
    teacher_forced_gate = {
        **legacy_gate,
        "require_teacher_forced_old_logprob_revalidation": True,
        "teacher_forced_max_absolute_error": 0.02,
        "teacher_forced_max_mean_absolute_error": 0.005,
        "teacher_forced_required_token_coverage": 1.0,
    }
    if dict(update_gate) != legacy_gate and dict(update_gate) != teacher_forced_gate:
        raise AgenticRLConfigError("controlled actor update gate is unsafe")
    claim_boundary = _mapping(
        manifest,
        "claim_boundary",
        label="controlled actor factorial",
    )
    expected_claim = {
        "actor_behavior_and_credit_diagnostics_only": True,
        "causal_actor_awareness_effect": False,
        "policy_improvement": False,
        "autonomous_decomposition": False,
        "official_leaderboard_comparison": False,
    }
    if dict(claim_boundary) != expected_claim:
        raise AgenticRLConfigError("controlled actor claim boundary is unsafe")

    payload = {
        "schema": CONTROLLED_ACTOR_FACTORIAL_SCHEMA,
        "id": _required_string(manifest, "id", "controlled actor factorial"),
        "factorial": {
            "path": str(factorial_path.relative_to(workspace_root)),
            "sha256": factorial.sha256,
        },
        "controlled_recipe": {
            "path": str(recipe_path.relative_to(workspace_root)),
            "file_sha256": sha256_file(recipe_path),
            "semantic_sha256": recipe.sha256,
            "task_id": recipe.task_id,
            "state_count": len(recipe.branches),
            "state_ids": [branch.branch_id for branch in recipe.branches],
        },
        "runtime": dict(runtime),
        "update_gate": dict(update_gate),
        "claim_boundary": expected_claim,
    }
    return ResolvedAgenticRLFactorial(
        config_path=path,
        workspace_root=workspace_root,
        payload=payload,
        sha256=canonical_sha256(payload),
    )


def load_agentic_rl_factorial_manifest(
    config_path: str | Path,
) -> ResolvedAgenticRLFactorial:
    """Load the awareness x credit matrix and reject missing or confounded cells."""

    path = Path(config_path).resolve()
    workspace_root = _workspace_root(path)
    manifest = _load_yaml(path, label="factorial manifest")
    _expect_schema(manifest, FACTORIAL_SCHEMA, "factorial manifest")
    _expect_exact_keys(
        manifest,
        {
            "schema",
            "id",
            "factors",
            "paired_experiments",
            "cells",
            "claim_boundary",
        },
        "factorial manifest",
    )
    expected_factors = {
        "actor_awareness": ["none", "oracle_goal_graph"],
        "credit_state_identity": [
            "anchor_observation",
            "residual_goal_and_frontier",
        ],
    }
    if dict(_mapping(manifest, "factors", label="factorial manifest")) != expected_factors:
        raise AgenticRLConfigError("factorial manifest factors differ from the registered 2x2")

    expected_cells = {
        "unaware_g2po_tn": {
            "actor_awareness": "none",
            "credit_state_identity": "anchor_observation",
        },
        "unaware_dppo": {
            "actor_awareness": "none",
            "credit_state_identity": "residual_goal_and_frontier",
        },
        "aware_g2po_tn": {
            "actor_awareness": "oracle_goal_graph",
            "credit_state_identity": "anchor_observation",
        },
        "aware_dppo": {
            "actor_awareness": "oracle_goal_graph",
            "credit_state_identity": "residual_goal_and_frontier",
        },
    }
    cells = _mapping(manifest, "cells", label="factorial manifest")
    if dict(cells) != expected_cells:
        raise AgenticRLConfigError(
            "factorial manifest does not contain the exact four registered cells"
        )

    references = manifest.get("paired_experiments")
    if not isinstance(references, list) or len(references) != 2:
        raise AgenticRLConfigError("factorial manifest requires exactly two paired experiments")
    experiments = [
        load_agentic_rl_experiment(
            _resolve_workspace_ref(workspace_root, reference, "paired experiment")
        )
        for reference in references
    ]
    by_awareness: dict[str, ResolvedAgenticRLExperiment] = {}
    for experiment in experiments:
        awareness = _required_string(
            _mapping(experiment.payload, "runtime", label="resolved experiment"),
            "actor_awareness_mode",
            "resolved runtime",
        )
        if awareness in by_awareness:
            raise AgenticRLConfigError(
                f"factorial manifest repeats actor awareness cell {awareness!r}"
            )
        by_awareness[awareness] = experiment
    if set(by_awareness) != {"none", "oracle_goal_graph"}:
        raise AgenticRLConfigError("factorial manifest does not span both awareness levels")

    unaware = by_awareness["none"]
    aware = by_awareness["oracle_goal_graph"]
    if unaware.payload["documents"] != aware.payload["documents"]:
        raise AgenticRLConfigError(
            "factorial paired experiments change referenced scientific documents"
        )
    if unaware.payload["comparison_receipt"] != aware.payload["comparison_receipt"]:
        raise AgenticRLConfigError(
            "factorial paired experiments change the registered credit comparison"
        )
    if unaware.payload["runtime"]["rollout"] != aware.payload["runtime"]["rollout"]:
        raise AgenticRLConfigError("factorial paired experiments change rollout budgets")

    claim_boundary = _mapping(manifest, "claim_boundary", label="factorial manifest")
    _expect_exact_keys(
        claim_boundary,
        {
            "oracle_upper_bound",
            "autonomous_decomposition",
            "official_leaderboard_comparison",
        },
        "factorial manifest.claim_boundary",
    )
    if dict(claim_boundary) != {
        "oracle_upper_bound": True,
        "autonomous_decomposition": False,
        "official_leaderboard_comparison": False,
    }:
        raise AgenticRLConfigError("factorial manifest claim boundary is unsafe")

    payload = {
        "schema": FACTORIAL_SCHEMA,
        "id": _required_string(manifest, "id", "factorial manifest"),
        "factors": expected_factors,
        "cells": expected_cells,
        "paired_experiments": {
            awareness: {
                "path": str(experiment.config_path.relative_to(workspace_root)),
                "sha256": experiment.sha256,
            }
            for awareness, experiment in sorted(by_awareness.items())
        },
        "claim_boundary": dict(claim_boundary),
    }
    return ResolvedAgenticRLFactorial(
        config_path=path,
        workspace_root=workspace_root,
        payload=payload,
        sha256=canonical_sha256(payload),
    )


def load_agentic_rl_experiment(
    config_path: str | Path,
) -> ResolvedAgenticRLExperiment:
    """Load explicit config references and prove the requested paired comparison."""

    path = Path(config_path).resolve()
    workspace_root = _workspace_root(path)
    experiment = _load_yaml(path, label="experiment")
    _expect_schema(experiment, EXPERIMENT_SCHEMA, "experiment")
    _expect_exact_keys(
        experiment,
        {
            "schema",
            "id",
            "study_ref",
            "dataset_ref",
            "model_ref",
            "source_ref",
            "actor_ref",
            "arms",
            "runtime",
            "comparison",
        },
        "experiment",
    )

    refs = {
        "study": experiment["study_ref"],
        "dataset": experiment["dataset_ref"],
        "model": experiment["model_ref"],
        "source": experiment["source_ref"],
        "actor": experiment["actor_ref"],
    }
    arms = _mapping(experiment, "arms", label="experiment")
    _expect_exact_keys(arms, {"control", "treatment"}, "experiment.arms")
    for arm_name in ("control", "treatment"):
        arm = _mapping(arms, arm_name, label="experiment.arms")
        _expect_exact_keys(arm, {"id", "method_ref"}, f"experiment.arms.{arm_name}")
        refs[f"method_{arm_name}"] = arm["method_ref"]

    documents: dict[str, tuple[Path, dict[str, Any]]] = {}
    for role, ref in refs.items():
        document_path = _resolve_workspace_ref(workspace_root, ref, role)
        documents[role] = (document_path, _load_yaml(document_path, label=role))

    study_path, study = documents["study"]
    if study.get("schema") == PHASEPOOL_STUDY_SCHEMA:
        load_phasepool_study_config(study_path)
    else:
        load_oracle_subproblem_study_config(study_path)
    dataset_path, dataset = documents["dataset"]
    model_path, model = documents["model"]
    source_path, source = documents["source"]
    actor_path, actor = documents["actor"]
    control_path, control_method = documents["method_control"]
    treatment_path, treatment_method = documents["method_treatment"]
    _validate_source(source, workspace_root=workspace_root)
    _validate_dataset(
        dataset,
        workspace_root=workspace_root,
        source_revision=_required_string(source, "revision", "source"),
    )
    model = _validate_model(model)
    actor = _validate_actor(actor, source=source)
    _validate_model_actor_compatibility(model, actor)
    _validate_method(control_method, role="control")
    _validate_method(treatment_method, role="treatment")
    comparison_receipt = _validate_comparison(
        experiment,
        study=study,
        control_method=control_method,
        treatment_method=treatment_method,
    )
    runtime = _validate_runtime(_mapping(experiment, "runtime", label="experiment"))

    document_receipts = {
        role: {
            "path": str(path.relative_to(workspace_root)),
            "schema": str(document.get("schema", "")),
            "id": _document_id(document),
            "sha256": _sha256_file(path),
        }
        for role, (path, document) in sorted(documents.items())
    }
    payload: dict[str, Any] = {
        "schema": RESOLVED_SCHEMA,
        "id": _required_string(experiment, "id", "experiment"),
        "documents": document_receipts,
        "study": study,
        "dataset": dataset,
        "model": model,
        "source": source,
        "actor": actor,
        "arms": {
            "control": {
                "id": _required_string(
                    _mapping(arms, "control", label="arms"), "id", "control arm"
                ),
                "method": control_method,
            },
            "treatment": {
                "id": _required_string(
                    _mapping(arms, "treatment", label="arms"), "id", "treatment arm"
                ),
                "method": treatment_method,
            },
        },
        "runtime": runtime,
        "comparison_receipt": comparison_receipt,
    }
    return ResolvedAgenticRLExperiment(
        config_path=path,
        workspace_root=workspace_root,
        payload=payload,
        sha256=canonical_sha256(payload),
    )


def bind_agentic_rl_runtime(
    resolved: ResolvedAgenticRLExperiment,
    bindings: AgenticRLRuntimeBindings,
) -> argparse.Namespace:
    """Bind a validated portable composition to one machine and current adapter set."""

    payload = resolved.payload
    if canonical_sha256(payload) != resolved.sha256:
        raise AgenticRLConfigError("resolved experiment changed after composition validation")
    dataset = _mapping(payload, "dataset", label="resolved experiment")
    model = _mapping(payload, "model", label="resolved experiment")
    source = _mapping(payload, "source", label="resolved experiment")
    actor = _mapping(payload, "actor", label="resolved experiment")
    runtime = _mapping(payload, "runtime", label="resolved experiment")
    rollout = _mapping(runtime, "rollout", label="runtime")
    optimizer = _mapping(model, "optimizer", label="model")
    adaptation = _mapping(model, "adaptation", label="model")
    source_credit = _mapping(source, "credit_contract", label="source")
    actor_protocol = _mapping(actor, "protocol", label="actor")
    actor_generation = _mapping(actor, "generation", label="actor")
    actor_projection = _mapping(actor, "projection", label="actor")
    actor_anchor = _mapping(actor, "anchor", label="actor")
    actor_parity = _mapping(actor, "platform_parity", label="actor")
    model_io = _mapping(model, "model_io", label="model")
    evaluation = _mapping(dataset, "evaluation", label="dataset")
    manifests = _mapping(dataset, "manifests", label="dataset")
    arms = _mapping(payload, "arms", label="resolved experiment")
    control = _mapping(_mapping(arms, "control", label="arms"), "method", label="control arm")
    treatment = _mapping(_mapping(arms, "treatment", label="arms"), "method", label="treatment arm")
    control_implementation = _mapping(control, "implementation", label="control method")
    treatment_implementation = _mapping(treatment, "implementation", label="treatment method")
    credit = _mapping(control, "credit", label="control method")
    credit_statistics = _mapping(credit, "statistics", label="control credit")
    comparison = _mapping(payload, "comparison_receipt", label="resolved experiment")
    documents = _mapping(payload, "documents", label="resolved experiment")
    actor_document = _mapping(documents, "actor", label="resolved documents")

    values: dict[str, Any] = {
        "model_path": str(bindings.model_path),
        "source_root": str(bindings.source_root),
        "data_root": str(bindings.data_root),
        "output_dir": str(bindings.output_dir),
        "train_manifest": str(
            _resolve_workspace_ref(
                resolved.workspace_root, manifests["train"], "dataset train manifest"
            )
        ),
        "eval_manifest": str(
            _resolve_workspace_ref(
                resolved.workspace_root, manifests["eval"], "dataset eval manifest"
            )
        ),
        "eval_selection": _required_string(evaluation, "selection", "dataset evaluation"),
        "eval_schedule": (
            str(
                _resolve_workspace_ref(
                    resolved.workspace_root,
                    evaluation["schedule"],
                    "dataset evaluation schedule",
                )
            )
            if evaluation.get("schedule") is not None
            else None
        ),
        "eval_task_count": int(evaluation["task_count"]),
        "source_revision": _required_string(source, "revision", "source"),
        "source_credit_file": _required_string(
            source_credit, "source_file", "source credit contract"
        ),
        "source_credit_file_sha256": _required_string(
            source_credit, "source_file_sha256", "source credit contract"
        ),
        "source_parity_fixture": str(
            _resolve_workspace_ref(
                resolved.workspace_root,
                source_credit["parity_fixture"],
                "source parity fixture",
            )
        ),
        "source_parity_tolerance": float(source_credit["tolerance"]),
        "source_parity_scope": _required_string(source_credit, "scope", "source credit contract"),
        "actor_profile_id": _required_string(actor, "id", "actor"),
        "actor_protocol_id": _required_string(
            actor_protocol,
            "id",
            "actor protocol",
        ),
        "actor_config_path": str(
            _resolve_workspace_ref(
                resolved.workspace_root,
                actor_document["path"],
                "actor config document",
            )
        ),
        "actor_source_files": dict(_mapping(actor, "source_files", label="actor")),
        "history_length": int(actor_protocol["history_length"]),
        "show_admissible_actions": bool(actor_protocol.get("show_admissible_actions", True)),
        "apply_chat_template": model_io.get("renderer_id") == "hf_native_chat_template_v1",
        "configured_actor_schema": str(actor["schema"]),
        "legacy_response_contract_inferred": bool(actor["legacy_response_contract_inferred"]),
        "response_contract": dict(_mapping(actor, "response_contract", label="actor")),
        "response_contract_id": _required_string(
            _mapping(actor, "response_contract", label="actor"),
            "contract_id",
            "actor.response_contract",
        ),
        "response_contract_sha256": canonical_sha256(
            _mapping(actor, "response_contract", label="actor")
        ),
        "action_generation_mode": _required_string(actor_generation, "mode", "actor generation"),
        "do_sample": bool(actor_generation["do_sample"]),
        "train_temperature": float(actor_generation["train_temperature"]),
        "eval_temperature": float(actor_generation["eval_temperature"]),
        "actor_awareness_mode": _required_string(
            runtime,
            "actor_awareness_mode",
            "runtime",
        ),
        "temperature": float(actor_generation["train_temperature"]),
        "top_p": float(actor_generation["top_p"]),
        "top_k": (int(actor_generation["top_k"]) if "top_k" in actor_generation else None),
        "repetition_penalty": (
            float(actor_generation["repetition_penalty"])
            if "repetition_penalty" in actor_generation
            else None
        ),
        "max_prompt_tokens": int(actor_generation["max_prompt_tokens"]),
        "max_action_tokens": int(actor_generation["max_response_tokens"]),
        "action_projection_mode": _required_string(actor_projection, "mode", "actor projection"),
        "anchor_mode": _required_string(actor_anchor, "mode", "actor anchor"),
        "actor_engine_exact": bool(actor_parity["engine_exact"]),
        "actor_behavioral_contract_exact": bool(actor_parity["behavioral_contract_exact"]),
        "actor_parity_scope": _required_string(actor_parity, "scope", "actor parity"),
        "device": str(bindings.device or "cuda"),
        "preflight_only": bool(bindings.preflight_only),
        "allow_shared_gpu": bool(bindings.allow_shared_gpu),
        "training_backend": str(bindings.training_backend),
        "verl_source_root": (
            str(bindings.verl_source_root) if bindings.verl_source_root is not None else None
        ),
        "verl_parity_receipt": (
            str(bindings.verl_parity_receipt) if bindings.verl_parity_receipt is not None else None
        ),
        "phasepool_calibration_receipt": (
            str(bindings.phasepool_calibration_receipt)
            if bindings.phasepool_calibration_receipt is not None
            else None
        ),
        "dtype": _required_string(model, "dtype", "model"),
        "execution_mode": _required_string(runtime, "mode", "runtime"),
        "updates": int(rollout["updates"]),
        "rollout_seed": int(rollout["rollout_seed"]),
        "eval_seed": int(rollout["eval_seed"]),
        "initialization_seed": int(model["initialization_seed"]),
        "group_size": int(rollout["group_size"]),
        "train_task_groups": int(rollout["train_task_groups"]),
        "train_task_selection": _required_string(
            rollout,
            "train_task_selection",
            "runtime.rollout",
        ),
        "eval_tasks_per_horizon": int(rollout["eval_tasks_per_horizon"]),
        "max_episode_steps": int(rollout["max_episode_steps"]),
        "learning_rate": float(optimizer["learning_rate"]),
        "clip_ratio": float(_mapping(control, "objective", label="control method")["clip_ratio"]),
        "clip_ratio_high": (
            float(_mapping(control, "objective", label="control method")["clip_ratio_high"])
            if "clip_ratio_high" in _mapping(control, "objective", label="control method")
            else None
        ),
        "reference_kl_coef": float(
            _mapping(
                _mapping(control, "objective", label="control method"),
                "reference_kl",
                label="control objective",
            )["coefficient"]
        ),
        "reference_kl_type": _required_string(
            _mapping(
                _mapping(control, "objective", label="control method"),
                "reference_kl",
                label="control objective",
            ),
            "estimator",
            "control objective reference KL",
        ),
        "max_grad_norm": float(optimizer["max_grad_norm"]),
        "update_micro_batch_size": int(optimizer["update_micro_batch_size"]),
        "lora_r": int(adaptation["r"]),
        "lora_alpha": int(adaptation["alpha"]),
        "lora_target_modules": list(adaptation["target_modules"]),
        "lora_exclude_modules": list(adaptation.get("exclude_modules", [])),
        "model_loader": str(model.get("loader", "auto_causal_lm")),
        "quantization": str(
            _mapping(model, "quantization", label="model").get("mode", "none")
            if "quantization" in model
            else "none"
        ),
        "quantization_double": bool(
            _mapping(model, "quantization", label="model").get("double_quantization", False)
            if "quantization" in model
            else False
        ),
        "allocator_warmup": bool(
            _mapping(model, "quantization", label="model").get("allocator_warmup", True)
            if "quantization" in model
            else True
        ),
        "attn_implementation": str(model.get("attn_implementation", "sdpa")),
        "use_liger_kernel": bool(model.get("use_liger_kernel", False)),
        "liger_kernel_profile": str(model.get("liger_kernel_profile", "full")),
        "entropy_coef": float(
            _mapping(control, "objective", label="control method").get("entropy_coef", 0.0)
        ),
        "gamma": float(credit_statistics["gamma"]),
        "step_advantage_weight": float(credit_statistics["step_advantage_weight"]),
        "credit_normalization": _required_string(
            credit_statistics, "normalization", "credit statistics"
        ),
        "invalid_action_penalty": float(credit_statistics["invalid_action_penalty"]),
        "credit_epsilon": float(credit_statistics["epsilon"]),
        "experiment_id": str(payload["id"]),
        "configured_model_reference": _required_string(model, "reference", "model"),
        "configured_model_revision": _required_string(model, "revision", "model"),
        "configured_model_checkpoint_tree_sha256": _required_string(
            model,
            "checkpoint_tree_sha256",
            "model",
        ),
        "configured_model_schema": str(model["schema"]),
        "legacy_model_io_inferred": bool(model["legacy_model_io_inferred"]),
        "model_io_profile": dict(model_io),
        "model_io_profile_id": _required_string(model_io, "profile_id", "model.model_io"),
        "model_io_profile_sha256": canonical_sha256(model_io),
        "model_supports_think_tags": bool(
            _mapping(model, "actor_capabilities", label="model").get("think_tags", False)
            if model.get("legacy_model_io_inferred")
            else model_io.get("selected_reasoning_mode") == "on"
        ),
        "experiment_config_path": str(resolved.config_path),
        "experiment_config_sha256": _sha256_file(resolved.config_path),
        "resolved_experiment": dict(payload),
        "resolved_experiment_sha256": resolved.sha256,
        "policy_backend_id": _required_string(model, "backend", "model"),
        "task_suite_id": _required_string(dataset, "component", "dataset"),
        "rl_method_id": str(comparison["shared_contract"]["rl_method"]),
        "comparison_primary_variable": str(comparison["primary_variable"]),
        "control_decomposition_id": str(control_implementation["decomposition"]),
        "treatment_decomposition_id": str(treatment_implementation["decomposition"]),
        "control_credit_assigner_id": str(control_implementation["credit_assigner"]),
        "treatment_credit_assigner_id": str(treatment_implementation["credit_assigner"]),
        "control_state_identity_fields": list(comparison["control_state_identity"]),
        "treatment_state_identity_fields": list(comparison["treatment_state_identity"]),
        "control_credit_estimator_config": dict(control.get("credit", {}).get("estimator", {})),
        "treatment_credit_estimator_config": dict(treatment.get("credit", {}).get("estimator", {})),
        "control_arm_id": str(_mapping(arms, "control", label="arms")["id"]),
        "treatment_arm_id": str(_mapping(arms, "treatment", label="arms")["id"]),
        "dynamic_sampling_enabled": bool(
            _mapping(runtime, "dynamic_sampling", label="runtime").get("enabled", False)
            if "dynamic_sampling" in runtime
            else False
        ),
        "dynamic_sampling_min_informative_groups": int(
            _mapping(runtime, "dynamic_sampling", label="runtime").get("min_informative_groups", 1)
            if "dynamic_sampling" in runtime
            else 1
        ),
        "dynamic_sampling_max_retries": int(
            _mapping(runtime, "dynamic_sampling", label="runtime").get("max_retries", 3)
            if "dynamic_sampling" in runtime
            else 3
        ),
    }
    if (
        values["comparison_primary_variable"] == "cross_fitted_credit_estimator"
        and values["execution_mode"] != "full_eval_only"
        and values["phasepool_calibration_receipt"] is None
    ):
        raise AgenticRLConfigError(
            "PhasePool online runtime requires a frozen calibration admission receipt"
        )
    _validate_bound_values(values)
    _validate_training_backend_bindings(values)
    return argparse.Namespace(**values)


def namespace_from_experiment_config(
    config_path: str | Path,
    *,
    model_path: str | Path,
    source_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str | None,
    preflight_only: bool,
    allow_shared_gpu: bool,
    training_backend: str = "local_transformers",
    verl_source_root: str | Path | None = None,
    verl_parity_receipt: str | Path | None = None,
    phasepool_calibration_receipt: str | Path | None = None,
) -> argparse.Namespace:
    """Compatibility adapter for the existing native P0 runner."""

    return bind_agentic_rl_runtime(
        load_agentic_rl_experiment(config_path),
        AgenticRLRuntimeBindings(
            model_path=Path(model_path),
            source_root=Path(source_root),
            data_root=Path(data_root),
            output_dir=Path(output_dir),
            device=device,
            preflight_only=preflight_only,
            allow_shared_gpu=allow_shared_gpu,
            training_backend=training_backend,
            verl_source_root=(Path(verl_source_root) if verl_source_root is not None else None),
            verl_parity_receipt=(
                Path(verl_parity_receipt) if verl_parity_receipt is not None else None
            ),
            phasepool_calibration_receipt=(
                Path(phasepool_calibration_receipt)
                if phasepool_calibration_receipt is not None
                else None
            ),
        ),
    )


def _validate_training_backend_bindings(values: Mapping[str, Any]) -> None:
    backend = values.get("training_backend")
    source_root = values.get("verl_source_root")
    parity_receipt = values.get("verl_parity_receipt")
    if backend not in {"local_transformers", "verl"}:
        raise AgenticRLConfigError("training backend must be local_transformers or verl")
    if backend == "verl":
        if not source_root or not parity_receipt:
            raise AgenticRLConfigError(
                "veRL training requires runtime source root and full-model parity receipt"
            )
    elif source_root is not None or parity_receipt is not None:
        raise AgenticRLConfigError(
            "veRL runtime bindings are invalid for the local training backend"
        )


def _workspace_root(config_path: Path) -> Path:
    for parent in config_path.parents:
        if parent.name == "configs":
            return parent.parent.resolve()
    raise AgenticRLConfigError("experiment config must live below the workspace configs directory")


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AgenticRLConfigError(f"cannot load {label} config {path}: {error}") from error
    if not isinstance(raw, dict):
        raise AgenticRLConfigError(f"{label} config must be a YAML mapping")
    return raw


def _resolve_workspace_ref(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AgenticRLConfigError(f"{label} must be a non-empty workspace-relative path")
    reference = Path(value)
    if reference.is_absolute():
        raise AgenticRLConfigError(f"{label} must be workspace-relative")
    candidate = (root / reference).resolve()
    if candidate != root and root not in candidate.parents:
        raise AgenticRLConfigError(f"{label} escapes the workspace: {value}")
    return candidate


def _validate_dataset(
    raw: Mapping[str, Any],
    *,
    workspace_root: Path,
    source_revision: str,
) -> None:
    _expect_schema(raw, DATASET_SCHEMA, "dataset")
    _expect_exact_keys(
        raw,
        {"schema", "id", "component", "benchmark", "manifests", "evaluation"},
        "dataset",
    )
    _supported("task_suite", raw.get("component"), "dataset component")
    if raw.get("benchmark") not in {"alfworld_native", "toolbench_native"}:
        raise AgenticRLConfigError("dataset benchmark must be alfworld_native or toolbench_native")
    manifests = _mapping(raw, "manifests", label="dataset")
    _expect_exact_keys(manifests, {"train", "eval"}, "dataset.manifests")
    manifest_paths = {
        role: _resolve_workspace_ref(
            workspace_root,
            ref,
            f"dataset {role} manifest",
        )
        for role, ref in manifests.items()
    }
    evaluation = _mapping(raw, "evaluation", label="dataset")
    _expect_exact_keys(
        evaluation,
        {"split", "horizon_strata", "selection", "schedule", "task_count"},
        "dataset.evaluation",
    )
    if evaluation.get("horizon_strata") != [
        "short",
        "medium",
        "long",
    ]:
        raise AgenticRLConfigError("dataset requires short/medium/long evaluation")
    split = evaluation.get("split")
    selection = evaluation.get("selection")
    schedule = evaluation.get("schedule")
    task_count = evaluation.get("task_count")
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 1:
        raise AgenticRLConfigError("dataset evaluation task_count must be positive")
    if split == "valid_unseen":
        if selection != "all_manifest_tasks_exactly_once" or schedule is not None:
            raise AgenticRLConfigError("valid_unseen evaluation must use its full manifest")
    elif split == "valid_seen":
        if selection == "source_seed_schedule":
            if not isinstance(schedule, str):
                raise AgenticRLConfigError(
                    "official valid_seen evaluation requires a source seed schedule"
                )
            schedule_path = _resolve_workspace_ref(
                workspace_root,
                schedule,
                "dataset evaluation schedule",
            )
            if task_count != 128:
                raise AgenticRLConfigError(
                    "official G2PO valid_seen evaluation requires 128 episodes"
                )
            _validate_official_g2po_eval_population(
                manifest_path=manifest_paths["eval"],
                schedule_path=schedule_path,
                task_count=task_count,
                source_revision=source_revision,
            )
        elif selection == "calibration_manifest_exactly_once":
            if schedule is not None:
                raise AgenticRLConfigError(
                    "valid_seen calibration manifest cannot declare a schedule"
                )
            manifest = _load_json_mapping(
                manifest_paths["eval"],
                "valid_seen calibration manifest",
            )
            tasks = manifest.get("tasks")
            if not isinstance(tasks, list) or len(tasks) != task_count:
                raise AgenticRLConfigError(
                    "valid_seen calibration task count differs from its manifest"
                )
            if any(
                not isinstance(task, Mapping) or task.get("split") != "valid_seen" for task in tasks
            ):
                raise AgenticRLConfigError(
                    "valid_seen calibration manifest contains a different split"
                )
        else:
            raise AgenticRLConfigError(
                "valid_seen evaluation requires an official schedule or exact calibration manifest"
            )
    else:
        raise AgenticRLConfigError("dataset evaluation split is unsupported")


def _validate_official_g2po_eval_population(
    *,
    manifest_path: Path,
    schedule_path: Path,
    task_count: int,
    source_revision: str,
) -> None:
    manifest = _load_json_mapping(manifest_path, "official G2PO eval manifest")
    schedule = _load_json_mapping(schedule_path, "official G2PO eval schedule")
    if manifest.get("schema") != "lh_task_manifest/v1":
        raise AgenticRLConfigError("official G2PO eval manifest schema is invalid")
    if manifest.get("benchmark") != "alfworld_native":
        raise AgenticRLConfigError("official G2PO eval manifest benchmark is invalid")
    if manifest.get("dataset_root") != "ALFWORLD_DATA":
        raise AgenticRLConfigError(
            "official G2PO eval manifest must use portable ALFWORLD_DATA root"
        )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise AgenticRLConfigError("official G2PO eval manifest tasks are invalid")
    task_keys: set[tuple[str, str]] = set()
    for item in tasks:
        if not isinstance(item, Mapping) or item.get("split") != "valid_seen":
            raise AgenticRLConfigError("official G2PO eval manifest contains a non-valid_seen task")
        key = (str(item.get("task_id", "")), str(item.get("game_path", "")))
        if not all(key) or key in task_keys:
            raise AgenticRLConfigError("official G2PO eval manifest task identities are invalid")
        task_keys.add(key)

    expected_schedule_fields = {
        "schema",
        "source_revision",
        "split",
        "seed_start",
        "episode_count",
        "selection_contract",
        "episodes",
    }
    if set(schedule) != expected_schedule_fields:
        raise AgenticRLConfigError("official G2PO eval schedule fields differ from schema")
    if schedule.get("schema") != "lh_g2po_evaluation_schedule/v1":
        raise AgenticRLConfigError("official G2PO eval schedule schema is invalid")
    if schedule.get("source_revision") != source_revision:
        raise AgenticRLConfigError(
            "official G2PO eval schedule source revision differs from experiment source"
        )
    if schedule.get("split") != "valid_seen" or schedule.get("seed_start") != 1000:
        raise AgenticRLConfigError("official G2PO eval schedule split or seed_start is invalid")
    episodes = schedule.get("episodes")
    if (
        schedule.get("episode_count") != task_count
        or not isinstance(episodes, list)
        or len(episodes) != task_count
    ):
        raise AgenticRLConfigError(
            "official G2PO eval schedule episode_count differs from dataset contract"
        )
    selection = schedule.get("selection_contract")
    if not isinstance(selection, Mapping) or (
        selection.get("eval_dataset") != "eval_in_distribution"
        or selection.get("worker_seed_formula") != "1000 + episode_index"
        or selection.get("worker_count") != task_count
        or selection.get("filesystem_walk_order_bound") is not True
    ):
        raise AgenticRLConfigError("official G2PO eval schedule selection contract is invalid")
    source_files = selection.get("source_files")
    if (
        not isinstance(source_files, Mapping)
        or not source_files
        or any(
            not isinstance(path, str) or not isinstance(digest, str) or not _is_hex(digest, 64)
            for path, digest in source_files.items()
        )
    ):
        raise AgenticRLConfigError("official G2PO eval schedule source file hashes are invalid")

    episode_task_keys: set[tuple[str, str]] = set()
    episode_ids: set[str] = set()
    expected_episode_fields = {
        "episode_index",
        "episode_id",
        "environment_seed",
        "task_id",
        "game_path",
    }
    for index, item in enumerate(episodes):
        if not isinstance(item, Mapping) or set(item) != expected_episode_fields:
            raise AgenticRLConfigError(
                f"official G2PO eval schedule episode {index} fields are invalid"
            )
        episode_id = str(item.get("episode_id", ""))
        task_key = (str(item.get("task_id", "")), str(item.get("game_path", "")))
        if (
            item.get("episode_index") != index
            or item.get("environment_seed") != 1000 + index
            or not episode_id
            or episode_id in episode_ids
            or task_key not in task_keys
        ):
            raise AgenticRLConfigError(
                f"official G2PO eval schedule episode {index} is not source-reproducible"
            )
        episode_ids.add(episode_id)
        episode_task_keys.add(task_key)
    if episode_task_keys != task_keys:
        raise AgenticRLConfigError(
            "official G2PO eval manifest and schedule unique task populations differ"
        )


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AgenticRLConfigError(f"cannot load {label} {path}: {error}") from error
    if not isinstance(raw, dict):
        raise AgenticRLConfigError(f"{label} must be a JSON mapping")
    return raw


def _validate_model(raw: Mapping[str, Any]) -> dict[str, Any]:
    schema = raw.get("schema")
    if schema not in {MODEL_SCHEMA, MODEL_SCHEMA_V2}:
        raise AgenticRLConfigError("model schema is unsupported")
    common = {
        "schema",
        "id",
        "backend",
        "reference",
        "revision",
        "checkpoint_tree_sha256",
        "dtype",
        "initialization_seed",
        "adaptation",
        "optimizer",
    }
    schema_field = "actor_capabilities" if schema == MODEL_SCHEMA else "model_io"
    required = {*common, schema_field}
    allowed = {
        *required,
        "loader",
        "quantization",
        "attn_implementation",
        "use_liger_kernel",
        "liger_kernel_profile",
    }
    if not required <= set(raw) or not set(raw) <= allowed:
        raise AgenticRLConfigError("model fields differ from schema")
    _supported("policy_backend", raw.get("backend"), "policy backend")
    if not _is_hex(str(raw.get("revision", "")), 40):
        raise AgenticRLConfigError("model revision must be a full lowercase commit SHA")
    if not _is_hex(str(raw.get("checkpoint_tree_sha256", "")), 64):
        raise AgenticRLConfigError("model checkpoint tree hash must be a lowercase SHA-256")
    if raw.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise AgenticRLConfigError("model dtype is unsupported")
    if raw.get("attn_implementation", "sdpa") not in {
        "eager",
        "sdpa",
        "flash_attention_2",
        "flash_attention_4",
    }:
        raise AgenticRLConfigError("model attention implementation is unsupported")
    use_liger_kernel = raw.get("use_liger_kernel", False)
    if type(use_liger_kernel) is not bool:
        raise AgenticRLConfigError("model Liger kernel flag must be boolean")
    liger_kernel_profile = raw.get("liger_kernel_profile", "full")
    if liger_kernel_profile not in {"full", "qwen3_swiglu_only"}:
        raise AgenticRLConfigError("model Liger kernel profile is unsupported")
    if not use_liger_kernel and liger_kernel_profile != "full":
        raise AgenticRLConfigError("a selective model Liger profile requires use_liger_kernel=true")
    if schema == MODEL_SCHEMA:
        actor_capabilities = _mapping(raw, "actor_capabilities", label="model")
        _expect_exact_keys(
            actor_capabilities,
            {"think_tags"},
            "model.actor_capabilities",
        )
        if type(actor_capabilities.get("think_tags")) is not bool:
            raise AgenticRLConfigError("model think-tags capability must be boolean")
        profile = _infer_legacy_model_io_profile(raw)
    else:
        profile = _parse_model_io_profile(_mapping(raw, "model_io", label="model"))
    adaptation = _mapping(raw, "adaptation", label="model")
    required_adaptation = {"type", "r", "alpha", "target_modules"}
    if not required_adaptation <= set(adaptation) or not set(adaptation) <= {
        *required_adaptation,
        "exclude_modules",
    }:
        raise AgenticRLConfigError("model.adaptation fields differ from schema")
    if adaptation.get("type") != "lora":
        raise AgenticRLConfigError("current policy adapter supports only LoRA")
    modules = adaptation.get("target_modules")
    if (
        not isinstance(modules, list)
        or not modules
        or not all(isinstance(item, str) and item for item in modules)
    ):
        raise AgenticRLConfigError(
            "model adaptation target_modules must be a non-empty string list"
        )
    excludes = adaptation.get("exclude_modules", [])
    if not isinstance(excludes, list) or not all(
        isinstance(item, str) and item for item in excludes
    ):
        raise AgenticRLConfigError("model adaptation exclude_modules must be a string list")
    loader = raw.get("loader", "auto_causal_lm")
    if loader not in {"auto_causal_lm", "gemma3_conditional_generation"}:
        raise AgenticRLConfigError("model loader is unsupported")
    quantization = raw.get(
        "quantization",
        {"mode": "none", "double_quantization": False},
    )
    if not isinstance(quantization, Mapping) or not {
        "mode",
        "double_quantization",
    } <= set(quantization) <= {
        "mode",
        "double_quantization",
        "allocator_warmup",
    }:
        raise AgenticRLConfigError("model quantization fields differ from schema")
    if quantization.get("mode") not in {"none", "bitsandbytes_nf4"}:
        raise AgenticRLConfigError("model quantization mode is unsupported")
    if not isinstance(quantization.get("double_quantization"), bool):
        raise AgenticRLConfigError("model double quantization flag must be boolean")
    if not isinstance(quantization.get("allocator_warmup", True), bool):
        raise AgenticRLConfigError("model allocator warmup flag must be boolean")
    if (
        quantization.get("allocator_warmup", True) is False
        and quantization.get("mode") != "bitsandbytes_nf4"
    ):
        raise AgenticRLConfigError(
            "allocator warmup may only be disabled for bitsandbytes NF4 loading"
        )
    if loader == "gemma3_conditional_generation" and not {
        "vision_tower",
        "multi_modal_projector",
    } <= set(excludes):
        raise AgenticRLConfigError("Gemma3 text-only adapter must exclude vision modules")
    optimizer = _mapping(raw, "optimizer", label="model")
    _expect_exact_keys(
        optimizer,
        {"name", "learning_rate", "max_grad_norm", "update_micro_batch_size"},
        "model.optimizer",
    )
    if optimizer.get("name") != "AdamW":
        raise AgenticRLConfigError("current policy adapter supports only AdamW")
    if (
        isinstance(optimizer.get("update_micro_batch_size"), bool)
        or int(optimizer.get("update_micro_batch_size", 0)) < 1
    ):
        raise AgenticRLConfigError("model update micro batch size must be positive")
    normalized = dict(raw)
    normalized["model_io"] = profile.to_dict()
    normalized["legacy_model_io_inferred"] = schema == MODEL_SCHEMA
    return normalized


def _parse_model_io_profile(raw: Mapping[str, Any]) -> ModelIOProfile:
    _expect_exact_keys(
        raw,
        {
            "profile_id",
            "renderer",
            "supported_roles",
            "system_role_policy",
            "reasoning",
            "assistant_turn_end",
        },
        "model.model_io",
    )
    roles = raw.get("supported_roles")
    if not isinstance(roles, list) or not all(isinstance(value, str) for value in roles):
        raise AgenticRLConfigError("model I/O supported_roles must be a string list")
    reasoning = _mapping(raw, "reasoning", label="model.model_io")
    _expect_exact_keys(
        reasoning,
        {
            "capability",
            "selected_mode",
            "template_kwargs",
            "prompt_injects_start",
            "generated_delimiters",
        },
        "model.model_io.reasoning",
    )
    template_kwargs = reasoning.get("template_kwargs")
    if not isinstance(template_kwargs, Mapping):
        raise AgenticRLConfigError("model I/O template_kwargs must be a mapping")
    prompt_injects_start = reasoning.get("prompt_injects_start")
    if type(prompt_injects_start) is not bool:
        raise AgenticRLConfigError("model I/O prompt_injects_start must be boolean")
    turn_end = _mapping(raw, "assistant_turn_end", label="model.model_io")
    _expect_exact_keys(turn_end, {"mode"}, "model.model_io.assistant_turn_end")
    try:
        return ModelIOProfile(
            profile_id=_required_string(raw, "profile_id", "model.model_io"),
            renderer_id=_required_string(raw, "renderer", "model.model_io"),
            supported_roles=tuple(roles),
            system_role_policy=str(raw.get("system_role_policy", "")),
            reasoning_capability=str(reasoning.get("capability", "")),
            selected_reasoning_mode=str(reasoning.get("selected_mode", "")),
            template_kwargs=dict(template_kwargs),
            prompt_injects_reasoning_start=prompt_injects_start,
            generated_reasoning_delimiters=str(reasoning.get("generated_delimiters", "")),
            assistant_turn_end_mode=str(turn_end.get("mode", "")),
        )
    except ModelIOContractError as error:
        raise AgenticRLConfigError(f"invalid model I/O profile: {error}") from error


def _infer_legacy_model_io_profile(raw: Mapping[str, Any]) -> ModelIOProfile:
    reference = str(raw.get("reference", ""))
    if reference == "Qwen/Qwen3-4B-Instruct-2507":
        return qwen3_instruct_2507_profile()
    think_tags = bool(_mapping(raw, "actor_capabilities", label="model").get("think_tags"))
    if reference in {"Qwen/Qwen3-4B", "Qwen/Qwen3-8B"}:
        return ModelIOProfile(
            profile_id=f"legacy_inferred_{str(raw['id'])}",
            renderer_id="hf_native_chat_template_v1",
            supported_roles=("system", "user", "assistant", "tool"),
            system_role_policy="native",
            reasoning_capability="switchable",
            selected_reasoning_mode="on" if think_tags else "off",
            template_kwargs={"enable_thinking": think_tags},
            generated_reasoning_delimiters="paired" if think_tags else "none",
        )
    if reference == "Qwen/Qwen3-4B-Thinking-2507":
        return ModelIOProfile(
            profile_id="legacy_inferred_qwen3_thinking_2507_native_chat_v1",
            renderer_id="hf_native_chat_template_v1",
            supported_roles=("system", "user", "assistant", "tool"),
            system_role_policy="native",
            reasoning_capability="fixed_on",
            selected_reasoning_mode="on",
            template_kwargs={},
            prompt_injects_reasoning_start=True,
            generated_reasoning_delimiters="closing_only",
        )
    return ModelIOProfile(
        profile_id=f"legacy_inferred_{str(raw['id'])}",
        renderer_id="hf_native_chat_template_v1",
        supported_roles=("user", "assistant"),
        system_role_policy="fold_into_first_user",
        reasoning_capability="fixed_off",
        selected_reasoning_mode="off",
        template_kwargs={},
        generated_reasoning_delimiters="none",
    )


def _validate_source(raw: Mapping[str, Any], *, workspace_root: Path) -> None:
    _expect_schema(raw, SOURCE_SCHEMA, "source")
    _expect_exact_keys(raw, {"schema", "id", "repository", "revision", "credit_contract"}, "source")
    revision = _required_string(raw, "revision", "source")
    if not _is_hex(revision, 40):
        raise AgenticRLConfigError("source revision must be a full lowercase Git SHA")
    contract = _mapping(raw, "credit_contract", label="source")
    _expect_exact_keys(
        contract,
        {"source_file", "source_file_sha256", "parity_fixture", "tolerance", "scope"},
        "source.credit_contract",
    )
    if contract.get("source_file") != "g2po/core_g2po.py":
        raise AgenticRLConfigError("source credit file must be g2po/core_g2po.py")
    if not _is_hex(str(contract.get("source_file_sha256", "")), 64):
        raise AgenticRLConfigError("source credit file hash must be a lowercase SHA-256")
    _resolve_workspace_ref(workspace_root, contract.get("parity_fixture"), "source parity fixture")
    if float(contract.get("tolerance", 0.0)) <= 0:
        raise AgenticRLConfigError("source parity tolerance must be positive")
    if contract.get("scope") != G2PO_SOURCE_PARITY_SCOPE:
        raise AgenticRLConfigError(f"source parity scope must be {G2PO_SOURCE_PARITY_SCOPE}")


def _validate_actor(raw: Mapping[str, Any], *, source: Mapping[str, Any]) -> dict[str, Any]:
    schema = raw.get("schema")
    if schema not in {ACTOR_SCHEMA, ACTOR_SCHEMA_V2}:
        raise AgenticRLConfigError("actor schema is unsupported")
    actor_fields = {
        "schema",
        "id",
        "source_revision",
        "source_files",
        "protocol",
        "generation",
        "projection",
        "anchor",
        "official_evaluation",
        "official_runtime",
        "platform_parity",
    }
    if schema == ACTOR_SCHEMA_V2:
        actor_fields.add("response_contract")
    _expect_exact_keys(
        raw,
        actor_fields,
        "actor",
    )
    if raw.get("source_revision") != source.get("revision"):
        raise AgenticRLConfigError("actor and source documents bind different G2PO revisions")
    source_files = _mapping(raw, "source_files", label="actor")
    expected_files = {
        "agent_system/environments/prompts/alfworld.py",
        "agent_system/environments/env_package/alfworld/projection.py",
        "agent_system/environments/env_manager.py",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "examples/g2po_trainer/run_alfworld.sh",
        "verl/trainer/config/ppo_trainer.yaml",
    }
    _expect_exact_keys(source_files, expected_files, "actor.source_files")
    if any(not _is_hex(str(value), 64) for value in source_files.values()):
        raise AgenticRLConfigError("actor source file hashes must be lowercase SHA-256")

    protocol = _mapping(raw, "protocol", label="actor")
    protocol_fields = {
        "id",
        "renderer",
        "history_length",
        "task_description_source",
        "include_total_step_count",
        "current_step_indexing",
        "omit_help_from_action_listing",
    }
    if schema == ACTOR_SCHEMA:
        protocol_fields.add("apply_chat_template")
    # show_admissible_actions is optional (defaults True when absent).
    if type(protocol.get("show_admissible_actions", True)) is not bool:
        raise AgenticRLConfigError("actor show_admissible_actions must be boolean")
    protocol_fields_check = {k: v for k, v in protocol.items() if k != "show_admissible_actions"}
    _expect_exact_keys(protocol_fields_check, protocol_fields, "actor.protocol")
    history_length = protocol.get("history_length")
    if history_length not in {2, 8}:
        raise AgenticRLConfigError("actor history length must be 2 or 8")
    expected_protocol = {
        "history_length": history_length,
        "task_description_source": "reset_observation",
        "include_total_step_count": True,
        "current_step_indexing": "one_based",
        "omit_help_from_action_listing": True,
    }
    if schema == ACTOR_SCHEMA:
        expected_protocol["apply_chat_template"] = True
    protocol_id = protocol.get("id")
    renderer = protocol.get("renderer")
    protocol_common = {
        key: value for key, value in protocol.items() if key not in {"id", "renderer"}
    }
    registered_renderers = {
        G2PO_ALFWORLD_ACTOR_PROTOCOL_ID: "source_alfworld_prompt",
        MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID: (
            "source_alfworld_context_action_only_output"
        ),
        MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_V2_ID: ("source_alfworld_context_v1"),
        QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID: (
            "source_alfworld_context_qwen_prompt_injected_thinking_v1"
        ),
    }
    if (
        registered_renderers.get(str(protocol_id)) != renderer
        or protocol_common != expected_protocol
    ):
        raise AgenticRLConfigError("actor protocol is not a registered ALFWorld contract")

    generation = _mapping(raw, "generation", label="actor")
    legacy_generation = {
        "mode": "free_text",
        "do_sample": True,
        "train_temperature": 1.0,
        "eval_temperature": 0.4,
        "top_p": 1.0,
        "max_prompt_tokens": 2048,
        "max_response_tokens": 512,
    }
    full_support_generation = {
        **legacy_generation,
        "top_k": 0,
        "repetition_penalty": 1.0,
    }
    registered_generation_profiles = [
        legacy_generation,
        full_support_generation,
    ]
    if raw.get("id") == "qwen_prompt_injected_thinking_alfworld_actor_1024_v1":
        registered_generation_profiles.append(
            {
                **full_support_generation,
                "max_response_tokens": 1024,
            }
        )
    if dict(generation) not in registered_generation_profiles:
        raise AgenticRLConfigError("actor generation differs from pinned G2PO ALFWorld")

    if schema == ACTOR_SCHEMA:
        response_contract = (
            AlfWorldResponseContract.official_g2po()
            if protocol_id == G2PO_ALFWORLD_ACTOR_PROTOCOL_ID
            else AlfWorldResponseContract.action_only()
        ).descriptor.to_dict()
    else:
        response_contract = dict(_mapping(raw, "response_contract", label="actor"))
        registered_response_contracts = {
            G2PO_ALFWORLD_ACTOR_PROTOCOL_ID: (
                AlfWorldResponseContract.official_g2po().descriptor.to_dict()
            ),
            MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_V2_ID: (
                AlfWorldResponseContract.action_only().descriptor.to_dict()
            ),
            QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID: (
                AlfWorldResponseContract.prompt_injected_thinking().descriptor.to_dict()
            ),
        }
        expected_contract = registered_response_contracts.get(str(protocol_id))
        if response_contract != expected_contract:
            raise AgenticRLConfigError(
                "actor response contract is not a registered ALFWorld contract"
            )

    projection = _mapping(raw, "projection", label="actor")
    projection_fields = {
        "mode",
        "force_admissible_action",
        "invalid_action_penalty",
    }
    if schema == ACTOR_SCHEMA:
        projection_fields.update({"require_think_tags", "require_action_tags"})
    _expect_exact_keys(projection, projection_fields, "actor.projection")
    expected_projection = {
        "mode": "source_alfworld_projection",
        "force_admissible_action": False,
        "invalid_action_penalty": 0.1,
    }
    if schema == ACTOR_SCHEMA:
        expected_projection["require_action_tags"] = True
    projection_common = dict(projection)
    projection_common.pop("require_think_tags", None)
    expected_think_tags = protocol_id == G2PO_ALFWORLD_ACTOR_PROTOCOL_ID
    if projection_common != expected_projection or (
        schema == ACTOR_SCHEMA and projection.get("require_think_tags") is not expected_think_tags
    ):
        raise AgenticRLConfigError("actor projection differs from pinned G2PO ALFWorld")

    anchor = _mapping(raw, "anchor", label="actor")
    _expect_exact_keys(
        anchor,
        {
            "mode",
            "include_admissible_actions",
            "retain_on_nothing_happens",
            "combine_invisible_transform_observation",
        },
        "actor.anchor",
    )
    if dict(anchor) != {
        "mode": "source_alfworld_graph_anchor",
        "include_admissible_actions": True,
        "retain_on_nothing_happens": True,
        "combine_invisible_transform_observation": True,
    }:
        raise AgenticRLConfigError("actor graph anchor differs from pinned G2PO ALFWorld")

    evaluation = _mapping(raw, "official_evaluation", label="actor")
    _expect_exact_keys(
        evaluation,
        {"split", "task_count", "max_episode_steps"},
        "actor.official_evaluation",
    )
    if dict(evaluation) != {
        "split": "valid_seen",
        "task_count": 128,
        "max_episode_steps": 50,
    }:
        raise AgenticRLConfigError("actor official evaluation contract differs from G2PO")
    official_runtime = _mapping(raw, "official_runtime", label="actor")
    _expect_exact_keys(
        official_runtime,
        {"engine", "gpu_count", "group_size"},
        "actor.official_runtime",
    )
    if dict(official_runtime) != {"engine": "vllm", "gpu_count": 8, "group_size": 8}:
        raise AgenticRLConfigError("actor official runtime contract differs from G2PO")
    parity = _mapping(raw, "platform_parity", label="actor")
    _expect_exact_keys(
        parity,
        {"engine", "engine_exact", "behavioral_contract_exact", "scope"},
        "actor.platform_parity",
    )
    registered_parity = {
        G2PO_ALFWORLD_ACTOR_PROTOCOL_ID: (True, G2PO_ACTOR_PARITY_SCOPE),
        MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID: (
            False,
            MODEL_COMPATIBLE_ACTOR_PARITY_SCOPE,
        ),
        MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_V2_ID: (
            False,
            MODEL_COMPATIBLE_ACTOR_PARITY_SCOPE,
        ),
        QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID: (
            False,
            QWEN_PROMPT_INJECTED_THINKING_ACTOR_PARITY_SCOPE,
        ),
    }
    expected_parity = registered_parity.get(str(protocol_id))
    if expected_parity is None:
        raise AgenticRLConfigError("actor platform parity protocol is unregistered")
    if (
        parity.get("engine") != "transformers_peft"
        or parity.get("engine_exact") is not False
        or parity.get("behavioral_contract_exact") is not expected_parity[0]
        or parity.get("scope") != expected_parity[1]
    ):
        raise AgenticRLConfigError("actor platform parity boundary is invalid")
    normalized = dict(raw)
    normalized["response_contract"] = response_contract
    normalized["legacy_response_contract_inferred"] = schema == ACTOR_SCHEMA
    return normalized


def _validate_model_actor_compatibility(
    model: Mapping[str, Any],
    actor: Mapping[str, Any],
) -> None:
    response_contract = _mapping(actor, "response_contract", label="actor")
    requires_generated_pair = (
        response_contract.get("reasoning_requirement") == "paired_in_generated_response"
    )
    requires_generated_closing_only = (
        response_contract.get("reasoning_requirement") == "closing_only_in_generated_response"
    )
    if (
        model.get("legacy_model_io_inferred") is False
        and actor.get("legacy_response_contract_inferred") is True
    ):
        raise AgenticRLConfigError(
            "model I/O v2 requires an explicit actor response contract; "
            "legacy actor/v1 inference is read-only compatibility"
        )
    if model.get("legacy_model_io_inferred") is True:
        capabilities = _mapping(model, "actor_capabilities", label="model")
        if requires_generated_pair and capabilities.get("think_tags") is not True:
            raise AgenticRLConfigError(
                "actor requires <think> tags but the legacy model document "
                "declares that it does not support them"
            )
        return

    try:
        profile = ModelIOProfile.from_dict(_mapping(model, "model_io", label="model"))
    except ModelIOContractError as error:
        raise AgenticRLConfigError(f"invalid normalized model I/O profile: {error}") from error
    if requires_generated_pair and profile.generated_reasoning_delimiters != "paired":
        raise AgenticRLConfigError(
            "actor requires a generated <think>...</think> pair but the model "
            "I/O profile does not generate paired reasoning delimiters"
        )
    if requires_generated_closing_only and (
        profile.selected_reasoning_mode != "on"
        or not profile.prompt_injects_reasoning_start
        or profile.generated_reasoning_delimiters != "closing_only"
    ):
        raise AgenticRLConfigError(
            "actor requires a prompt-injected reasoning start and a generated "
            "closing delimiter, but the model I/O profile does not provide both"
        )
    if (
        not requires_generated_pair
        and not requires_generated_closing_only
        and profile.selected_reasoning_mode != "off"
    ):
        raise AgenticRLConfigError("action-only actor requires a non-thinking model I/O profile")


def _validate_method(raw: Mapping[str, Any], *, role: str) -> None:
    _expect_schema(raw, METHOD_SCHEMA, f"{role} method")
    _expect_exact_keys(
        raw,
        {
            "schema",
            "id",
            "label",
            "implementation",
            "objective",
            "actor_mass",
            "credit",
            "claim_boundary",
        },
        f"{role} method",
    )
    implementation = _mapping(raw, "implementation", label=f"{role} method")
    _expect_exact_keys(
        implementation,
        {"rl_method", "decomposition", "credit_assigner"},
        f"{role} method.implementation",
    )
    for component_role, key in (
        ("rl_method", "rl_method"),
        ("decomposition", "decomposition"),
        ("credit_assigner", "credit_assigner"),
    ):
        _supported(component_role, implementation.get(key), f"{role} {key}")
    objective = _mapping(raw, "objective", label=f"{role} method")
    _objective_required = {"name", "clip_ratio", "reference_kl"}
    _objective_optional = {"clip_ratio_high", "entropy_coef"}
    _actual_objective_keys = set(objective)
    if (
        not _objective_required <= _actual_objective_keys
        or not (_actual_objective_keys - _objective_required) <= _objective_optional
    ):
        raise AgenticRLConfigError(
            f"{role} method.objective fields differ from schema: "
            + json.dumps(
                {
                    "missing": sorted(_objective_required - _actual_objective_keys),
                    "unknown": sorted(
                        _actual_objective_keys - _objective_required - _objective_optional
                    ),
                }
            )
        )
    if (
        objective.get("name") != "clipped_policy_gradient"
        or not 0 < float(objective.get("clip_ratio", 0)) < 1
    ):
        raise AgenticRLConfigError(
            f"{role} method requires a valid clipped policy-gradient objective"
        )
    reference_kl = _mapping(
        objective,
        "reference_kl",
        label=f"{role} method objective",
    )
    expected_reference_kl = {
        "enabled": True,
        "coefficient": 0.01,
        "estimator": "low_var_kl",
        "reference": "peft_disabled_frozen_base",
        "aggregation": "token_mean",
    }
    if dict(reference_kl) != expected_reference_kl:
        raise AgenticRLConfigError(
            f"{role} method reference KL differs from the pinned G2PO recipe"
        )
    actor_mass = _mapping(raw, "actor_mass", label=f"{role} method")
    _expect_exact_keys(actor_mass, {"id", "unit", "per_token_weight"}, f"{role} method.actor_mass")
    credit = _mapping(raw, "credit", label=f"{role} method")
    expected_credit_fields = {"mode", "state_identity", "statistics"}
    credit_assigner = implementation.get("credit_assigner")
    if credit_assigner in {"g2po_tn_loto", "phasepool_g2po"}:
        expected_credit_fields.add("estimator")
    _expect_exact_keys(credit, expected_credit_fields, f"{role} method.credit")
    state_identity = _mapping(credit, "state_identity", label=f"{role} credit")
    _expect_exact_keys(state_identity, {"fields"}, f"{role} method.credit.state_identity")
    statistics = _mapping(credit, "statistics", label=f"{role} credit")
    _expect_exact_keys(
        statistics,
        {
            "aggregation",
            "gamma",
            "step_advantage_weight",
            "normalization",
            "invalid_action_penalty",
            "epsilon",
        },
        f"{role} method.credit.statistics",
    )
    if statistics.get("aggregation") != "trajectory_normalized":
        raise AgenticRLConfigError(f"{role} method must use trajectory-normalized statistics")
    if "estimator" in expected_credit_fields:
        estimator = _mapping(credit, "estimator", label=f"{role} credit")
        _expect_exact_keys(
            estimator,
            {
                "family",
                "key",
                "weighting",
                "exclusion",
                "pooling",
                "fixed_pooling_weight",
                "require_outcome_variation",
            },
            f"{role} method.credit.estimator",
        )
        common = {
            "weighting": "trajectory",
            "exclusion": "trajectory",
            "fixed_pooling_weight": 0.5,
            "require_outcome_variation": True,
        }
        if any(estimator.get(key) != value for key, value in common.items()):
            raise AgenticRLConfigError(
                f"{role} estimator must use trajectory LOTO and outcome variation"
            )
        expected_estimator = (
            {"family": "grouped_mean", "key": "coarse", "pooling": "full"}
            if credit_assigner == "g2po_tn_loto"
            else {
                "family": "phasepool",
                "key": "phase",
                "pooling": "empirical_bayes",
            }
        )
        if any(estimator.get(key) != value for key, value in expected_estimator.items()):
            raise AgenticRLConfigError(
                f"{role} method estimator differs from registered {credit_assigner}"
            )
    _mapping(raw, "claim_boundary", label=f"{role} method")


def _validate_comparison(
    experiment: Mapping[str, Any],
    *,
    study: Mapping[str, Any],
    control_method: Mapping[str, Any],
    treatment_method: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = _mapping(experiment, "comparison", label="experiment")
    _expect_exact_keys(
        comparison,
        {"primary_variable", "allowed_algorithmic_differences"},
        "experiment.comparison",
    )
    primary_variable = comparison.get("primary_variable")
    allowed_differences = {
        "credit_state_identity": _ALLOWED_ALGORITHMIC_DIFFERENCES,
        "cross_fitted_credit_estimator": (_PHASEPOOL_ALLOWED_ALGORITHMIC_DIFFERENCES),
    }
    if primary_variable not in allowed_differences:
        raise AgenticRLConfigError("comparison primary variable is unsupported")
    configured_differences = comparison.get("allowed_algorithmic_differences")
    registered_differences = allowed_differences[str(primary_variable)]
    if configured_differences != registered_differences:
        raise AgenticRLConfigError(
            "comparison allowed differences do not match the registered ablation"
        )

    control_algorithm = _method_algorithm(control_method)
    treatment_algorithm = _method_algorithm(treatment_method)
    study_actor_loss = _mapping(study, "actor_loss", label="study")
    expected_actor_mass = {
        "id": "trajectory_step_token_unit_mass",
        "unit": study_actor_loss.get("unit"),
        "per_token_weight": study_actor_loss.get("per_token_weight"),
    }
    if _mapping(control_method, "actor_mass", label="control method") != expected_actor_mass:
        raise AgenticRLConfigError(
            "method actor mass does not implement the scientific study contract"
        )
    if _mapping(control_method, "actor_mass", label="control method") != _mapping(
        treatment_method, "actor_mass", label="treatment method"
    ):
        raise AgenticRLConfigError("control and treatment actor mass must be identical")
    differences = sorted(_leaf_differences(control_algorithm, treatment_algorithm))
    if differences != registered_differences:
        raise AgenticRLConfigError(
            "method comparison contains hidden algorithmic drift: " + json.dumps(differences)
        )
    control_impl = _mapping(control_method, "implementation", label="control method")
    treatment_impl = _mapping(treatment_method, "implementation", label="treatment method")
    control_credit = _mapping(control_method, "credit", label="control method")
    treatment_credit = _mapping(treatment_method, "credit", label="treatment method")
    expected = {
        "control_decomposition": (control_impl.get("decomposition"), "none"),
        "treatment_decomposition": (
            treatment_impl.get("decomposition"),
            "alfworld_fact_phases",
        ),
        "control_credit_assigner": (control_impl.get("credit_assigner"), "g2po_tn"),
        "treatment_credit_assigner": (
            treatment_impl.get("credit_assigner"),
            "oracle_phase_refined_g2po_tn",
        ),
        "control_credit_mode": (
            control_credit.get("mode"),
            "g2po_trajectory_normalized",
        ),
        "treatment_credit_mode": (
            treatment_credit.get("mode"),
            "oracle_phase_refined_g2po_tn",
        ),
        "control_state_identity": (
            _mapping(control_credit, "state_identity", label="control credit").get("fields"),
            _CONTROL_STATE_FIELDS,
        ),
        "treatment_state_identity": (
            _mapping(treatment_credit, "state_identity", label="treatment credit").get("fields"),
            _TREATMENT_STATE_FIELDS,
        ),
    }
    if primary_variable == "cross_fitted_credit_estimator":
        expected = {
            "control_decomposition": (
                control_impl.get("decomposition"),
                "alfworld_fact_phases",
            ),
            "treatment_decomposition": (
                treatment_impl.get("decomposition"),
                "alfworld_fact_phases",
            ),
            "control_credit_assigner": (
                control_impl.get("credit_assigner"),
                "g2po_tn_loto",
            ),
            "treatment_credit_assigner": (
                treatment_impl.get("credit_assigner"),
                "phasepool_g2po",
            ),
            "control_credit_mode": (
                control_credit.get("mode"),
                "g2po_trajectory_normalized_loto",
            ),
            "treatment_credit_mode": (
                treatment_credit.get("mode"),
                "phasepool_g2po_loto",
            ),
            "control_state_identity": (
                _mapping(
                    control_credit,
                    "state_identity",
                    label="control credit",
                ).get("fields"),
                _CONTROL_STATE_FIELDS,
            ),
            "treatment_state_identity": (
                _mapping(
                    treatment_credit,
                    "state_identity",
                    label="treatment credit",
                ).get("fields"),
                _PHASEPOOL_STATE_FIELDS,
            ),
        }
    mismatches = {
        name: {"configured": actual, "expected": wanted}
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if mismatches:
        raise AgenticRLConfigError(
            "comparison does not implement the registered observation-vs-subproblem state test: "
            + json.dumps(mismatches, sort_keys=True)
        )
    shared_contract = {
        "rl_method": control_impl["rl_method"],
        "objective": control_method["objective"],
        "actor_mass": control_method["actor_mass"]["id"],
        "credit_statistics": control_credit["statistics"],
    }
    phasepool_comparison = primary_variable == "cross_fitted_credit_estimator"
    return {
        "primary_variable": primary_variable,
        "scientific_name": "phasepool_g2po" if phasepool_comparison else "dppo",
        "implementation_name": (
            "phasepool_g2po_loto" if phasepool_comparison else "oracle_phase_refined_g2po_tn"
        ),
        "non_claim": "policy_improvement" if phasepool_comparison else "verified_state_dp",
        "shared_contract": shared_contract,
        "shared_contract_sha256": canonical_sha256(shared_contract),
        "control_state_identity": _CONTROL_STATE_FIELDS,
        "treatment_state_identity": (
            _PHASEPOOL_STATE_FIELDS if phasepool_comparison else _TREATMENT_STATE_FIELDS
        ),
        "algorithmic_differences": differences,
    }


def _validate_runtime(raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"mode", "rollout", "actor_awareness_mode", "dynamic_sampling"}
    if not set(raw) <= allowed or not {"mode", "rollout"} <= set(raw):
        raise AgenticRLConfigError("experiment.runtime fields differ from schema")
    mode = raw.get("mode")
    if mode not in {
        "paired_train_eval",
        "full_eval_only",
        "controlled_diagnostic_only",
    }:
        raise AgenticRLConfigError("experiment runtime mode is unsupported")
    rollout = _mapping(raw, "rollout", label="runtime")
    expected = {
        "updates",
        "rollout_seed",
        "eval_seed",
        "group_size",
        "train_task_groups",
        "train_task_selection",
        "eval_tasks_per_horizon",
        "max_episode_steps",
    }
    _expect_exact_keys(rollout, expected, "experiment.runtime.rollout")
    awareness = raw.get("actor_awareness_mode", "none")
    if awareness not in {"none", "oracle_goal_graph"}:
        raise AgenticRLConfigError("runtime actor awareness mode is unsupported")
    if rollout.get("train_task_selection") not in {
        "manifest_rotation",
        "task_type_round_robin",
    }:
        raise AgenticRLConfigError("runtime training task selection is unsupported")
    dynamic_sampling = raw.get("dynamic_sampling")
    validated_dynamic_sampling: dict[str, Any] | None = None
    if dynamic_sampling is not None:
        if not isinstance(dynamic_sampling, Mapping):
            raise AgenticRLConfigError("experiment.runtime.dynamic_sampling must be a mapping")
        _expect_exact_keys(
            dynamic_sampling,
            {"enabled", "min_informative_groups", "max_retries"},
            "experiment.runtime.dynamic_sampling",
        )
        if not isinstance(dynamic_sampling["enabled"], bool):
            raise AgenticRLConfigError("runtime dynamic_sampling.enabled must be boolean")
        for key in ("min_informative_groups", "max_retries"):
            value = dynamic_sampling[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AgenticRLConfigError(
                    f"runtime dynamic_sampling.{key} must be a positive integer"
                )
        validated_dynamic_sampling = dict(dynamic_sampling)
    result = {
        "mode": mode,
        "actor_awareness_mode": awareness,
        "rollout": dict(rollout),
    }
    if validated_dynamic_sampling is not None:
        result["dynamic_sampling"] = validated_dynamic_sampling
    return result


def _validate_bound_values(values: Mapping[str, Any]) -> None:
    positive = (
        "updates",
        "group_size",
        "train_task_groups",
        "eval_tasks_per_horizon",
        "eval_task_count",
        "max_episode_steps",
        "max_action_tokens",
        "learning_rate",
        "reference_kl_coef",
        "max_grad_norm",
        "update_micro_batch_size",
        "lora_r",
        "lora_alpha",
        "temperature",
        "gamma",
        "step_advantage_weight",
        "credit_epsilon",
    )
    invalid = [name for name in positive if float(values[name]) <= 0]
    if invalid:
        raise AgenticRLConfigError("runtime values must be positive: " + ", ".join(invalid))
    if not 0 < float(values["clip_ratio"]) < 1:
        raise AgenticRLConfigError("clip_ratio must be between zero and one")
    if values["reference_kl_type"] != "low_var_kl":
        raise AgenticRLConfigError("reference KL estimator must be low_var_kl")
    if float(values["invalid_action_penalty"]) < 0:
        raise AgenticRLConfigError("invalid_action_penalty must be non-negative")


def _method_algorithm(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "implementation": raw["implementation"],
        "objective": raw["objective"],
        "actor_mass": raw["actor_mass"],
        "credit": raw["credit"],
    }


def _leaf_differences(left: Any, right: Any, *, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_leaf_differences(left[key], right[key], prefix=path))
        return differences
    return [] if left == right else [prefix]


def _mapping(container: Mapping[str, Any], key: str, *, label: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise AgenticRLConfigError(f"{label}.{key} must be a mapping")
    return value


def _required_string(container: Mapping[str, Any], key: str, label: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgenticRLConfigError(f"{label}.{key} must be a non-empty string")
    return value


def _expect_schema(raw: Mapping[str, Any], schema: str, label: str) -> None:
    if raw.get("schema") != schema:
        raise AgenticRLConfigError(f"{label}.schema must be {schema}")


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(raw)
    if actual != expected:
        raise AgenticRLConfigError(
            f"{label} fields differ from schema: "
            + json.dumps(
                {
                    "missing": sorted(expected - actual),
                    "unknown": sorted(actual - expected),
                }
            )
        )


def _supported(role: str, value: Any, label: str) -> None:
    if value not in _SUPPORTED[role]:
        raise AgenticRLConfigError(f"unsupported {label}: {value!r}")


def _document_id(raw: Mapping[str, Any]) -> str:
    if isinstance(raw.get("id"), str):
        return str(raw["id"])
    study = raw.get("study")
    if isinstance(study, Mapping) and isinstance(study.get("id"), str):
        return str(study["id"])
    raise AgenticRLConfigError("referenced document lacks an id")


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AgenticRLConfigError(f"cannot hash config document {path}: {error}") from error
    return digest.hexdigest()
