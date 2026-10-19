"""Audited runner for frozen ALFWorld residual-state action scoring."""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lh_platform.training.adapter_identity import (
    AdapterIdentityError,
    load_adapter_identity,
)
from lh_platform.training.agentic_rl_config import (
    AgenticRLRuntimeBindings,
    bind_agentic_rl_runtime,
    load_agentic_rl_experiment,
)
from lh_platform.training.agentic_rl_harness import (
    assert_exclusive_gpu,
    query_compute_apps,
    transformers_policy_config_from_args,
)
from lh_platform.training.agentic_rl_p0 import run_p0
from lh_platform.training.agentic_rl_runtime import RunEventLogger, canonical_sha256
from lh_platform.training.alfworld_agentic_suite import OfficialAlfWorldBranchFactory
from lh_platform.training.alfworld_controlled_aliasing import (
    load_alfworld_controlled_aliasing_recipe,
)
from lh_platform.training.config_provenance import (
    repository_config_tree_bindings,
    repository_subtree_binding,
)
from lh_platform.training.controlled_actor_factorial_run import resolve_controlled_task_binding
from lh_platform.training.dataset import sha256_file
from lh_platform.training.evidence_artifacts import (
    reserve_fresh_evidence_directory,
    write_json_exclusive,
)
from lh_platform.training.parent_child_state_use.benchmark import (
    FUNCTIONAL_STATE_USE_MANIFEST_SCHEMA,
    FunctionalStateUseError,
    collect_alfworld_functional_state_use_manifest,
    load_functional_state_use_benchmark,
    score_alfworld_functional_state_use_manifest,
)
from lh_platform.training.parent_child_state_use.exposure import ActorStateExposureMode
from lh_platform.training.transformers_peft_policy import (
    TransformersPeftPolicy,
    checkpoint_tree_sha256,
)


class FunctionalStateUseRunError(RuntimeError):
    """Raised when the formal frozen-policy benchmark cannot be admitted."""


_ENVIRONMENT_REUSE_PROVENANCE_KEYS = (
    "source_revision",
    "source_head_tree_object_id",
    "recipe_sha256",
    "task_manifest_sha256",
)


def validate_functional_state_use_manifest_for_scoring(
    manifest: Mapping[str, Any],
    *,
    recipe_sha256: str,
    task_id: str,
    pair_type: str,
    expected_pair_observability: str,
    state_exposure_modes: tuple[str, ...],
    current_collection_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Admit immutable environment evidence across model/scorer revisions."""

    resolved = dict(manifest)
    manifest_body = dict(resolved)
    declared_manifest_sha256 = str(manifest_body.pop("manifest_sha256", ""))
    if canonical_sha256(manifest_body) != declared_manifest_sha256:
        raise FunctionalStateUseRunError("functional environment manifest hash differs")
    if (
        manifest_body.get("schema") != FUNCTIONAL_STATE_USE_MANIFEST_SCHEMA
        or manifest_body.get("recipe_sha256") != recipe_sha256
        or manifest_body.get("task_id") != task_id
        or manifest_body.get("pair_type") != pair_type
        or manifest_body.get("expected_pair_observability") != expected_pair_observability
        or manifest_body.get("state_exposure_modes") != list(state_exposure_modes)
    ):
        raise FunctionalStateUseRunError("functional environment manifest contract differs")
    origin = manifest_body.get("collection_provenance")
    if not isinstance(origin, Mapping):
        raise FunctionalStateUseRunError("functional environment collection provenance is missing")
    mismatched_environment_inputs = [
        key
        for key in _ENVIRONMENT_REUSE_PROVENANCE_KEYS
        if str(origin.get(key, "")) != str(current_collection_provenance.get(key, ""))
    ]
    if mismatched_environment_inputs:
        raise FunctionalStateUseRunError(
            "functional environment manifest input provenance differs: "
            + ", ".join(mismatched_environment_inputs)
        )
    receipt = {
        "schema": "lh_functional_state_use_manifest_reuse_receipt/v1",
        "manifest_sha256": declared_manifest_sha256,
        "required_environment_provenance_keys": list(_ENVIRONMENT_REUSE_PROVENANCE_KEYS),
        "required_environment_inputs_match": True,
        "origin_benchmark_matches_current": (
            str(origin.get("benchmark_semantic_sha256", ""))
            == str(
                current_collection_provenance.get(
                    "benchmark_semantic_sha256",
                    "",
                )
            )
        ),
        "origin_platform_matches_current": (
            str(origin.get("platform_revision", ""))
            == str(current_collection_provenance.get("platform_revision", ""))
            and str(origin.get("implementation_head_tree_object_id", ""))
            == str(
                current_collection_provenance.get(
                    "implementation_head_tree_object_id",
                    "",
                )
            )
        ),
        "reusable_across_model_or_scoring_revision": True,
        "claim_scope": ("same_hash_bound_environment_evidence_not_same_model_or_scorer"),
    }
    return resolved, receipt


def functional_state_use_implementation_bindings(
    workspace_root: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Bind the complete importable platform package instead of an allowlist."""

    root = Path(workspace_root).resolve()
    return (repository_subtree_binding(root / "lh_platform" / "lh_platform"),)


def _run_functional_state_use_benchmark(
    config_path: str | Path,
    *,
    model_path: str | Path,
    source_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str,
    preflight_only: bool,
    allow_shared_gpu: bool,
    scoring_micro_batch_size: int,
    collect_only: bool = False,
    manifest_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
) -> Path:
    resolved = load_functional_state_use_benchmark(config_path)
    target = reserve_fresh_evidence_directory(
        output_dir,
        protected_paths=(
            Path(model_path),
            Path(source_root),
            Path(data_root),
            resolved.config_path,
            resolved.workspace_root / "configs",
            *((Path(manifest_path),) if manifest_path is not None else ()),
            *((Path(adapter_path),) if adapter_path is not None else ()),
        ),
    )
    event_logger = RunEventLogger(target / "run_events.jsonl", run_id=uuid.uuid4().hex)
    event_logger.emit(
        "functional_state_use_started",
        benchmark_sha256=resolved.sha256,
        requested_device=device,
        preflight_only=preflight_only,
        scoring_micro_batch_size=scoring_micro_batch_size,
        collect_only=collect_only,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        adapter_path=str(adapter_path) if adapter_path is not None else None,
    )
    bindings = repository_config_tree_bindings(resolved)
    implementation_bindings = functional_state_use_implementation_bindings(resolved.workspace_root)
    external_source_binding = repository_subtree_binding(source_root)
    experiment_path = resolved.workspace_root / resolved.payload["experiment"]["path"]
    experiment = load_agentic_rl_experiment(experiment_path)
    args = bind_agentic_rl_runtime(
        experiment,
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
    args.generation_micro_batch_size = scoring_micro_batch_size
    base_preflight_path = run_p0(args)
    base_preflight = json.loads(base_preflight_path.read_text(encoding="utf-8"))
    try:
        adapter_binding = load_adapter_identity(adapter_path) if adapter_path is not None else None
    except AdapterIdentityError as error:
        raise FunctionalStateUseRunError(str(error)) from error
    if adapter_binding is not None and (
        adapter_binding.base_checkpoint_sha256 != str(base_preflight["base_checkpoint_sha256"])
    ):
        raise FunctionalStateUseRunError("adapter binds a different configured base model")
    if str(external_source_binding["revision"]) != str(base_preflight["source_revision"]):
        raise FunctionalStateUseRunError(
            "external source binding differs from the admitted source revision"
        )
    formal_launch_eligible = all(
        bool(binding["matches_head"]) for binding in (*bindings, *implementation_bindings)
    ) and bool(external_source_binding["matches_head"])
    preflight = {
        "schema": "lh_alfworld_functional_state_use_preflight/v1",
        "benchmark": {
            "path": str(resolved.config_path),
            "file_sha256": sha256_file(resolved.config_path),
            "semantic_sha256": resolved.sha256,
        },
        "repository_bindings": [dict(binding) for binding in bindings],
        "implementation_bindings": [dict(binding) for binding in implementation_bindings],
        "platform_revision": str(implementation_bindings[0]["revision"]),
        "external_source_binding": dict(external_source_binding),
        "base_preflight_path": str(base_preflight_path),
        "base_preflight_sha256": sha256_file(base_preflight_path),
        "base_checkpoint_sha256": base_preflight["base_checkpoint_sha256"],
        "source_revision": base_preflight["source_revision"],
        "adapter_binding": (adapter_binding.to_dict() if adapter_binding is not None else None),
        "formal_launch_eligible": formal_launch_eligible,
        "gpu_loaded": False,
        "policy_update_executed": False,
        "claim_scope": "functional_state_use_preflight_only",
    }
    preflight_path = target / "functional_state_use_preflight.json"
    write_json_exclusive(preflight_path, preflight)
    event_logger.emit(
        "functional_state_use_preflight_written",
        path=str(preflight_path),
        formal_launch_eligible=formal_launch_eligible,
    )
    if preflight_only:
        return preflight_path
    if not formal_launch_eligible:
        raise FunctionalStateUseRunError(
            "functional benchmark config tree must be committed unchanged"
        )

    recipe_path = resolved.workspace_root / resolved.payload["controlled_recipe"]["path"]
    recipe = load_alfworld_controlled_aliasing_recipe(recipe_path)
    task, metadata, instruction, task_manifest_sha256 = resolve_controlled_task_binding(
        recipe,
        data_root,
    )
    collection_provenance = {
        "benchmark_semantic_sha256": resolved.sha256,
        "platform_revision": str(implementation_bindings[0]["revision"]),
        "implementation_head_tree_object_id": str(
            implementation_bindings[0]["head_tree_object_id"]
        ),
        "source_revision": str(external_source_binding["revision"]),
        "source_head_tree_object_id": str(external_source_binding["head_tree_object_id"]),
        "recipe_sha256": recipe.sha256,
        "task_manifest_sha256": task_manifest_sha256,
    }
    if collect_only and manifest_path is not None:
        raise FunctionalStateUseRunError(
            "collect-only cannot consume an existing environment manifest"
        )
    if manifest_path is None:
        manifest = collect_alfworld_functional_state_use_manifest(
            recipe,
            task=task,
            task_metadata=metadata,
            instruction=instruction,
            branch_factory=OfficialAlfWorldBranchFactory(
                source_root=Path(source_root).resolve(),
                data_root=Path(data_root).resolve(),
            ),
            artifacts_dir=target / "environment",
            state_exposure_modes=tuple(ActorStateExposureMode),
            expected_pair_observability=str(resolved.payload["expected_pair_observability"]),
            collection_provenance=collection_provenance,
        )
    else:
        try:
            loaded = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FunctionalStateUseRunError(
                "cannot load functional environment manifest"
            ) from error
        if not isinstance(loaded, Mapping):
            raise FunctionalStateUseRunError("functional environment manifest must be a mapping")
        manifest = dict(loaded)
    manifest, manifest_reuse_receipt = validate_functional_state_use_manifest_for_scoring(
        manifest,
        recipe_sha256=recipe.sha256,
        task_id=task.task_id,
        pair_type=str(resolved.payload["pair_type"]),
        expected_pair_observability=str(resolved.payload["expected_pair_observability"]),
        state_exposure_modes=tuple(mode.value for mode in ActorStateExposureMode),
        current_collection_provenance=collection_provenance,
    )
    manifest_reuse_receipt = {
        **manifest_reuse_receipt,
        "existing_manifest_supplied": manifest_path is not None,
    }
    collected_manifest_path = target / "functional_state_use_manifest.json"
    write_json_exclusive(collected_manifest_path, manifest)
    event_logger.emit(
        "functional_state_use_manifest_ready",
        manifest_path=str(collected_manifest_path),
        manifest_sha256=manifest["manifest_sha256"],
        environment_accessed=manifest_path is None,
        origin_benchmark_matches_current=manifest_reuse_receipt["origin_benchmark_matches_current"],
        origin_platform_matches_current=manifest_reuse_receipt["origin_platform_matches_current"],
    )
    if collect_only:
        return collected_manifest_path

    compute_apps = query_compute_apps() if device.startswith("cuda") else ()
    assert_exclusive_gpu(
        device=device,
        allow_shared_gpu=allow_shared_gpu,
        compute_apps=compute_apps,
    )
    checkpoint_sha256 = checkpoint_tree_sha256(model_path)
    if checkpoint_sha256 != base_preflight["base_checkpoint_sha256"]:
        raise FunctionalStateUseRunError(
            "model checkpoint changed after functional benchmark preflight"
        )
    policy_args = argparse.Namespace(**vars(args))
    policy_args.preflight_only = False
    if adapter_binding is not None:
        try:
            current_adapter_binding = load_adapter_identity(adapter_binding.path)
        except AdapterIdentityError as error:
            raise FunctionalStateUseRunError(str(error)) from error
        if current_adapter_binding != adapter_binding:
            raise FunctionalStateUseRunError(
                "adapter checkpoint changed after functional benchmark preflight"
            )
        policy_args.adapter_path = adapter_binding.path
        policy_args.adapter_checkpoint_sha256 = adapter_binding.checkpoint_sha256
        policy_args.adapter_artifact_sha256 = adapter_binding.adapter_artifact_sha256
    policy_config = transformers_policy_config_from_args(
        policy_args,
        base_checkpoint_sha256=checkpoint_sha256,
    )
    event_logger.emit(
        "functional_state_use_policy_load_started",
        base_checkpoint_sha256=checkpoint_sha256,
        adapter_checkpoint_sha256=(
            adapter_binding.checkpoint_sha256 if adapter_binding is not None else None
        ),
    )
    policy = TransformersPeftPolicy(policy_config)
    event_logger.emit(
        "functional_state_use_policy_loaded",
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        adapter_initial_delta_zero=policy.adapter_initial_delta_zero,
    )

    result = score_alfworld_functional_state_use_manifest(
        manifest,
        policy=policy,
        sampling_temperature=float(resolved.payload["action_scoring"]["temperature"]),
    )
    artifact: dict[str, Any] = {
        **result,
        "benchmark": {
            "path": str(resolved.config_path),
            "file_sha256": sha256_file(resolved.config_path),
            "semantic_sha256": resolved.sha256,
        },
        "source_revision": base_preflight["source_revision"],
        "base_checkpoint_sha256": checkpoint_sha256,
        "adapter_binding": (adapter_binding.to_dict() if adapter_binding is not None else None),
        "model": {
            "document": dict(experiment.payload["documents"]["model"]),
            "id": str(experiment.payload["model"]["id"]),
            "reference": str(experiment.payload["model"]["reference"]),
            "revision": str(experiment.payload["model"]["revision"]),
            "checkpoint_tree_sha256": str(experiment.payload["model"]["checkpoint_tree_sha256"]),
        },
        "policy_component": asdict(policy.component),
        "task_manifest_sha256": task_manifest_sha256,
        "environment_manifest_path": str(collected_manifest_path),
        "environment_manifest_sha256": manifest["manifest_sha256"],
        "environment_collection_provenance": dict(manifest["collection_provenance"]),
        "environment_manifest_reuse_receipt": manifest_reuse_receipt,
        "recipe": {
            "path": str(recipe_path),
            "file_sha256": sha256_file(recipe_path),
            "semantic_sha256": recipe.sha256,
            "construction_selection": recipe.construction_selection,
        },
        "repository_bindings": [dict(binding) for binding in bindings],
        "implementation_bindings": [dict(binding) for binding in implementation_bindings],
        "platform_revision": str(implementation_bindings[0]["revision"]),
        "external_source_binding": dict(external_source_binding),
        "run_events_path": str(event_logger.path),
    }
    result_path = target / "functional_state_use_result.json"
    write_json_exclusive(result_path, artifact)
    event_logger.emit(
        "functional_state_use_completed",
        result_path=str(result_path),
        result_sha256=sha256_file(result_path),
        policy_update_executed=False,
    )
    return result_path


def run_functional_state_use_benchmark(
    config_path: str | Path,
    *,
    model_path: str | Path,
    source_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str,
    preflight_only: bool,
    allow_shared_gpu: bool,
    scoring_micro_batch_size: int,
    collect_only: bool = False,
    manifest_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
) -> Path:
    """Run the formal benchmark with stable domain-level rejection errors."""

    try:
        return _run_functional_state_use_benchmark(
            config_path,
            model_path=model_path,
            source_root=source_root,
            data_root=data_root,
            output_dir=output_dir,
            device=device,
            preflight_only=preflight_only,
            allow_shared_gpu=allow_shared_gpu,
            scoring_micro_batch_size=scoring_micro_batch_size,
            collect_only=collect_only,
            manifest_path=manifest_path,
            adapter_path=adapter_path,
        )
    except FunctionalStateUseRunError:
        raise
    except RuntimeError as error:
        raise FunctionalStateUseRunError(
            f"functional state-use runtime rejected: {error}"
        ) from error


__all__ = [
    "FunctionalStateUseError",
    "FunctionalStateUseRunError",
    "functional_state_use_implementation_bindings",
    "run_functional_state_use_benchmark",
    "validate_functional_state_use_manifest_for_scoring",
]
