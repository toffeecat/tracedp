"""CPU-only prospective validation for natural same-instance CVFC N1 plans."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lh_platform.alfworld_state import (
    ALFWORLD_PHYSICAL_CALL_COUNTER_SCHEMA,
    ALFWORLD_STATE_CAPABILITY_SCHEMA,
    AlfWorldPhysicalCallCounters,
)
from lh_platform.core.system_cell import (
    HORIZON_FACTOR_MANIFEST_SCHEMA,
    SYSTEM_CELL_MANIFEST_SCHEMA,
    HorizonFactorManifestV1,
    SystemCellContractError,
    SystemCellManifestV1,
    canonical_sha256,
    load_horizon_factor_manifest,
    load_system_cell_manifest,
)
from lh_platform.studies.runtime import AlfworldTaskManifest
from lh_platform.training.trajectory_composition.contracts import (
    COMPOSITION_PLAN_SCHEMA,
    COMPOSITION_RECEIPT_SCHEMA,
    FRAGMENT_RECORD_SCHEMA,
    REPLAY_EVIDENCE_SCHEMA,
)
from lh_platform.training.trajectory_composition.junction import (
    E1_JUNCTION_REPLAY_RECEIPT_SCHEMA,
)
from lh_platform.training.trajectory_composition.selective_certification_smoke import (
    SELECTIVE_CERTIFICATION_SMOKE_DECISION,
)
from lh_platform.training.trajectory_composition.source_certification import (
    SNAPSHOT_TRANSITION_CERTIFICATION_SCHEMA,
)

N1_DESIGN_PLAN_SCHEMA = "lh_cvfc_n1_design_plan/v1"
N1_PLAN_VALIDATION_SCHEMA = "lh_cvfc_n1_plan_validation/v1"
N1_CLAIM_SCOPE = "frozen_policy_same_instance_support_and_replay_anatomy_no_training"
N1_CONTROL_IDS = (
    "C0_no_composition",
    "C1_equal_physical_call_extra_rollout",
    "C2_equal_physical_call_local_retry",
    "C3_critical_effect_drop_sham",
    "C4_privileged_expert_diagnostic",
)
N1_REQUIRED_STOP_CONDITIONS = frozenset(
    {
        "identity_mismatch",
        "task_or_group_leakage",
        "seed_reuse",
        "denominator_loss",
        "physical_call_mismatch",
        "direct_support_saturation",
        "authorization_missing",
    }
)


class N1PlanValidationError(ValueError):
    """Raised when a prospective N1 plan is incomplete, confounded, or executable."""


@dataclass(frozen=True)
class ArtifactReferenceV1:
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, label: str) -> ArtifactReferenceV1:
        _expect_exact_keys(payload, {"path", "sha256"}, label)
        reference = cls(path=str(payload["path"]), sha256=str(payload["sha256"]))
        if (
            not reference.path
            or Path(reference.path).is_absolute()
            or ".." in Path(reference.path).parts
        ):
            raise N1PlanValidationError(f"{label}.path must remain workspace-relative")
        _require_sha256(f"{label}.sha256", reference.sha256)
        return reference

    def resolve(self, workspace_root: Path, *, label: str) -> Path:
        path = (workspace_root / self.path).resolve()
        if not path.is_relative_to(workspace_root) or not path.is_file():
            raise N1PlanValidationError(f"{label} artifact is unavailable: {path}")
        actual = _sha256_file(path)
        if actual != self.sha256:
            raise N1PlanValidationError(f"{label} file SHA-256 changed: {actual} != {self.sha256}")
        return path

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ValidatedN1Plan:
    plan_path: Path
    plan_file_sha256: str
    plan_content_sha256: str
    payload: Mapping[str, Any]
    candidate_cells: tuple[SystemCellManifestV1, ...]
    horizon_factors: HorizonFactorManifestV1
    calibration_manifest: AlfworldTaskManifest
    target_manifest: AlfworldTaskManifest
    validation_receipt: Mapping[str, Any]


def validate_n1_design_plan(
    plan_path: str | Path,
    *,
    workspace_root: str | Path,
    data_root: str | Path,
) -> ValidatedN1Plan:
    """Expand and validate an N1 design without importing Torch or launching work."""

    path = Path(plan_path).resolve()
    workspace = Path(workspace_root).resolve()
    data = Path(data_root).resolve()
    if not path.is_file() or not path.is_relative_to(workspace):
        raise N1PlanValidationError("N1 plan must be a file inside workspace_root")
    payload = _read_mapping(path)
    expected_top = {
        "schema",
        "plan_id",
        "claim_scope",
        "execution_authorized",
        "training_authorized",
        "candidate_system_cells",
        "allowed_calibration_cell_differences",
        "infrastructure_evidence",
        "horizon_factor_manifest",
        "calibration_task_manifest",
        "target_task_manifest",
        "closed_task_manifests",
        "novelty_mode",
        "overlap_declaration",
        "calibration_samples_per_task",
        "candidate_target_samples_per_task",
        "rollout",
        "selection_gate",
        "seed_plan",
        "power",
        "controls",
        "budgets",
        "pipeline",
        "stop_conditions",
    }
    _expect_exact_keys(payload, expected_top, "N1 design plan")
    if payload.get("schema") != N1_DESIGN_PLAN_SCHEMA:
        raise N1PlanValidationError("unsupported N1 design-plan schema")
    plan_id = _identifier(payload.get("plan_id"), "plan_id")
    if payload.get("claim_scope") != N1_CLAIM_SCOPE:
        raise N1PlanValidationError("N1 claim_scope changed")
    if payload.get("execution_authorized") is not False:
        raise N1PlanValidationError("design plan must not authorize N1 execution")
    if payload.get("training_authorized") is not False:
        raise N1PlanValidationError("design plan must not authorize training")

    cell_refs_raw = _sequence(payload["candidate_system_cells"], "candidate_system_cells")
    if len(cell_refs_raw) < 2:
        raise N1PlanValidationError("system-cell calibration requires at least two candidates")
    cell_refs = tuple(
        ArtifactReferenceV1.from_dict(
            _mapping(value, f"candidate_system_cells[{index}]"),
            label=f"candidate_system_cells[{index}]",
        )
        for index, value in enumerate(cell_refs_raw)
    )
    cells = tuple(
        _load_cell(reference, workspace, index=index) for index, reference in enumerate(cell_refs)
    )
    cell_ids = tuple(cell.system_cell_id for cell in cells)
    if len(cell_ids) != len(set(cell_ids)) or len({cell.sha256 for cell in cells}) != len(cells):
        raise N1PlanValidationError("candidate system cells must be identity-unique")
    for cell in cells:
        _validate_n1_system_cell(cell)
    allowed_differences = _string_tuple(
        payload["allowed_calibration_cell_differences"],
        "allowed_calibration_cell_differences",
    )
    if not allowed_differences:
        raise N1PlanValidationError("candidate cells require declared allowed differences")
    if any(
        value
        not in {
            "policy_identity",
            "sampling_policy",
            "actor_protocol",
            "context",
            "memory",
            "runtime",
        }
        for value in allowed_differences
    ):
        raise N1PlanValidationError("N1 allowed cell difference is unsupported")
    observed_difference_paths = _validate_cell_differences(cells, allowed_differences)

    horizon_ref = ArtifactReferenceV1.from_dict(
        _mapping(payload["horizon_factor_manifest"], "horizon_factor_manifest"),
        label="horizon_factor_manifest",
    )
    horizon_path = horizon_ref.resolve(workspace, label="horizon_factor_manifest")
    try:
        horizon = load_horizon_factor_manifest(horizon_path)
    except SystemCellContractError as error:
        raise N1PlanValidationError(f"invalid horizon-factor manifest: {error}") from error
    if horizon.schema != HORIZON_FACTOR_MANIFEST_SCHEMA or horizon.intrinsic_horizon_claimed:
        raise N1PlanValidationError(
            "N1 must not treat ALFWorld expert trace terciles as intrinsic horizon"
        )

    calibration_ref = ArtifactReferenceV1.from_dict(
        _mapping(payload["calibration_task_manifest"], "calibration_task_manifest"),
        label="calibration_task_manifest",
    )
    target_ref = ArtifactReferenceV1.from_dict(
        _mapping(payload["target_task_manifest"], "target_task_manifest"),
        label="target_task_manifest",
    )
    calibration = _load_task_manifest(
        calibration_ref,
        workspace=workspace,
        data_root=data,
        label="calibration_task_manifest",
    )
    target = _load_task_manifest(
        target_ref,
        workspace=workspace,
        data_root=data,
        label="target_task_manifest",
    )
    overlap = _population_overlap(calibration, target)
    if overlap["task_ids"] or overlap["group_uids"]:
        raise N1PlanValidationError(f"calibration/target population leakage: {overlap}")

    closed_refs_raw = _sequence(payload["closed_task_manifests"], "closed_task_manifests")
    if not closed_refs_raw:
        raise N1PlanValidationError("N1 must bind closed task populations")
    closed_refs = tuple(
        ArtifactReferenceV1.from_dict(
            _mapping(value, f"closed_task_manifests[{index}]"),
            label=f"closed_task_manifests[{index}]",
        )
        for index, value in enumerate(closed_refs_raw)
    )
    closed_manifests = tuple(
        _load_task_manifest(
            reference,
            workspace=workspace,
            data_root=data,
            label=f"closed_task_manifests[{index}]",
        )
        for index, reference in enumerate(closed_refs)
    )
    novelty_mode = str(payload["novelty_mode"])
    overlap_declaration = str(payload["overlap_declaration"])
    closed_overlap = _closed_population_overlap((calibration, target), closed_manifests)
    if novelty_mode == "new_task_and_group_relative_to_closed_populations":
        if overlap_declaration:
            raise N1PlanValidationError("new-task mode forbids an overlap declaration")
        if any(value for value in closed_overlap.values()):
            raise N1PlanValidationError(f"N1 population overlaps closed tasks: {closed_overlap}")
    elif novelty_mode == "declared_replication":
        if not overlap_declaration.strip():
            raise N1PlanValidationError("replication mode requires an overlap declaration")
    else:
        raise N1PlanValidationError("unsupported N1 novelty_mode")

    infrastructure_evidence = _validate_infrastructure_evidence(
        _mapping(payload["infrastructure_evidence"], "infrastructure_evidence"),
        workspace=workspace,
    )
    calibration_samples = _positive_integer(
        payload["calibration_samples_per_task"], "calibration_samples_per_task", minimum=2
    )
    target_sample_candidates = tuple(
        _positive_integer(value, "candidate_target_samples_per_task", minimum=2)
        for value in _sequence(
            payload["candidate_target_samples_per_task"],
            "candidate_target_samples_per_task",
        )
    )
    if (
        not target_sample_candidates
        or target_sample_candidates != tuple(sorted(set(target_sample_candidates)))
        or target_sample_candidates[-1] > calibration_samples
    ):
        raise N1PlanValidationError(
            "candidate target group sizes must be unique, increasing, and covered by calibration"
        )
    rollout = _validate_rollout(_mapping(payload["rollout"], "rollout"))
    horizon_bindings = _validate_n1_horizon_bindings(
        horizon,
        cells=cells,
        target_manifest_file_sha256=target_ref.sha256,
        max_episode_steps=rollout["max_episode_steps"],
    )
    selection_gate = _validate_selection_gate(
        _mapping(payload["selection_gate"], "selection_gate"),
        candidate_cell_ids=cell_ids,
    )
    seed_receipt = _validate_seed_plan(
        _mapping(payload["seed_plan"], "seed_plan"),
        calibration=calibration,
        target=target,
    )
    power_receipt = _validate_power(
        _mapping(payload["power"], "power"),
        target_task_count=len(target.tasks),
    )
    controls = _validate_controls(_mapping(payload["controls"], "controls"))
    pipeline = _validate_pipeline(_mapping(payload["pipeline"], "pipeline"))
    budgets = _validate_budgets(
        _mapping(payload["budgets"], "budgets"),
        calibration_task_count=len(calibration.tasks),
        target_task_count=len(target.tasks),
        candidate_cell_count=len(cells),
        calibration_samples=calibration_samples,
        target_samples=max(target_sample_candidates),
        max_episode_steps=rollout["max_episode_steps"],
    )
    stop_conditions = frozenset(_string_tuple(payload["stop_conditions"], "stop_conditions"))
    missing_stops = sorted(N1_REQUIRED_STOP_CONDITIONS - stop_conditions)
    if missing_stops:
        raise N1PlanValidationError(f"N1 stop_conditions are incomplete: {missing_stops}")

    plan_file_sha256 = _sha256_file(path)
    plan_content_sha256 = canonical_sha256(payload)
    receipt = {
        "schema": N1_PLAN_VALIDATION_SCHEMA,
        "plan_id": plan_id,
        "plan_path": str(path),
        "plan_file_sha256": plan_file_sha256,
        "plan_content_sha256": plan_content_sha256,
        "claim_scope": N1_CLAIM_SCOPE,
        "cpu_only_validation": True,
        "model_framework_imported": False,
        "run_ready": False,
        "execution_authorized": False,
        "training_authorized": False,
        "candidate_system_cells": [
            {
                "reference": reference.to_dict(),
                "system_cell_id": cell.system_cell_id,
                "system_cell_content_sha256": cell.sha256,
                "policy_checkpoint_sha256": cell.policy_identity.policy_checkpoint_sha256,
            }
            for reference, cell in zip(cell_refs, cells, strict=True)
        ],
        "allowed_calibration_cell_differences": list(allowed_differences),
        "observed_calibration_cell_difference_paths": list(observed_difference_paths),
        "horizon_factor_manifest": {
            "reference": horizon_ref.to_dict(),
            "schema": horizon.schema,
            "content_sha256": horizon.sha256,
            "intrinsic_horizon_claimed": horizon.intrinsic_horizon_claimed,
            "n1_factor_bindings": horizon_bindings,
        },
        "populations": {
            "calibration": _population_receipt(calibration_ref, calibration),
            "target": _population_receipt(target_ref, target),
            "calibration_target_overlap": overlap,
            "closed_population_overlap": closed_overlap,
            "novelty_mode": novelty_mode,
            "overlap_declaration": overlap_declaration,
        },
        "infrastructure_evidence": infrastructure_evidence,
        "calibration_samples_per_task": calibration_samples,
        "candidate_target_samples_per_task": list(target_sample_candidates),
        "rollout": rollout,
        "selection_gate": selection_gate,
        "seed_plan": seed_receipt,
        "power": power_receipt,
        "controls": controls,
        "budgets": budgets,
        "pipeline": pipeline,
        "stop_conditions": sorted(stop_conditions),
        "synthetic_launch_receipt_constructed": True,
        "execution_counts": {
            "model_samples": 0,
            "environment_resets": 0,
            "environment_steps": 0,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "training_examples_admitted": 0,
        },
        "blocked_by": [
            "system_cell_calibration_receipt",
            "selected_actor_availability_and_parity_receipt",
            "integrated_n1_runner_receipt",
            "n1_execution_authorization",
        ],
    }
    receipt["validation_receipt_sha256"] = canonical_sha256(receipt)
    return ValidatedN1Plan(
        plan_path=path,
        plan_file_sha256=plan_file_sha256,
        plan_content_sha256=plan_content_sha256,
        payload=payload,
        candidate_cells=cells,
        horizon_factors=horizon,
        calibration_manifest=calibration,
        target_manifest=target,
        validation_receipt=receipt,
    )


def write_n1_plan_validation_receipt(
    validated: ValidatedN1Plan,
    output_path: str | Path,
) -> Path:
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.tmp")
    staging.write_text(
        json.dumps(validated.validation_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(target)
    return target


def _load_cell(
    reference: ArtifactReferenceV1,
    workspace: Path,
    *,
    index: int,
) -> SystemCellManifestV1:
    path = reference.resolve(workspace, label=f"candidate_system_cells[{index}]")
    try:
        cell = load_system_cell_manifest(path)
    except SystemCellContractError as error:
        raise N1PlanValidationError(f"invalid candidate system cell {index}: {error}") from error
    if cell.schema != SYSTEM_CELL_MANIFEST_SCHEMA:
        raise N1PlanValidationError("candidate system-cell schema changed")
    return cell


def _validate_infrastructure_evidence(
    payload: Mapping[str, Any],
    *,
    workspace: Path,
) -> dict[str, Any]:
    _expect_exact_keys(
        payload,
        {"selective_certification_smoke_closeout"},
        "infrastructure_evidence",
    )
    reference = ArtifactReferenceV1.from_dict(
        _mapping(
            payload["selective_certification_smoke_closeout"],
            "selective_certification_smoke_closeout",
        ),
        label="selective_certification_smoke_closeout",
    )
    path = reference.resolve(
        workspace,
        label="selective_certification_smoke_closeout",
    )
    closeout = _read_mapping(path)
    required = {
        "schema": "lh_alfworld_selective_certification_smoke_closeout/v1",
        "status": "completed",
        "decision": SELECTIVE_CERTIFICATION_SMOKE_DECISION,
        "scientific_result": False,
        "post_state_identity_verified": True,
        "outcome_verified": True,
        "restore_verified": True,
    }
    if any(closeout.get(key) != value for key, value in required.items()):
        raise N1PlanValidationError(
            "selective-certification smoke closeout is not the required closed PASS"
        )
    expected_counts = {
        "model_samples": 0,
        "logical_semantic_actions": 1,
        "environment_resets": 5,
        "environment_steps": 3,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_examples_admitted": 0,
    }
    if closeout.get("execution_counts") != expected_counts:
        raise N1PlanValidationError("selective-certification smoke execution denominator changed")
    physical_raw = _mapping(
        closeout.get("combined_physical_call_totals"),
        "selective-certification combined physical calls",
    )
    try:
        physical = AlfWorldPhysicalCallCounters.from_dict(physical_raw)
    except (TypeError, ValueError) as error:
        raise N1PlanValidationError(
            f"invalid selective-certification physical calls: {error}"
        ) from error
    if (
        physical.logical_semantic_action_calls != 1
        or physical.environment_reset_calls != 5
        or physical.environment_step_calls != 3
        or physical.source_worker_reset_calls != 2
        or physical.state_restore_calls != 3
        or physical.snapshot_only_worker_step_calls != 1
        or physical.candidate_probe_first_worker_step_calls != 1
        or physical.candidate_probe_replay_worker_step_calls != 1
        or closeout.get("combined_physical_call_totals_sha256") != physical.sha256
    ):
        raise N1PlanValidationError("selective-certification physical-call receipt changed")
    authorization_sha256 = str(closeout.get("authorization_sha256", ""))
    _require_sha256("selective-certification authorization_sha256", authorization_sha256)
    return {
        "selective_certification_smoke_closeout": reference.to_dict(),
        "decision": SELECTIVE_CERTIFICATION_SMOKE_DECISION,
        "authorization_sha256": authorization_sha256,
        "execution_counts": expected_counts,
        "combined_physical_call_totals_sha256": physical.sha256,
        "verified": True,
    }


def _validate_n1_system_cell(cell: SystemCellManifestV1) -> None:
    if not cell.frozen or not cell.policy_identity.frozen:
        raise N1PlanValidationError("N1 candidate policy/system cell must be frozen")
    if cell.adaptive_components_allowed or cell.optimizer_present:
        raise N1PlanValidationError("N1 candidate cell forbids adaptation and optimizer state")
    if cell.environment.config.get("benchmark") != "alfworld_native":
        raise N1PlanValidationError("N1 candidate environment must be ALFWorld native")
    if cell.action_objective.config.get("unit") != "canonical_semantic_action":
        raise N1PlanValidationError("N1 action objective must use canonical semantic actions")
    if cell.action_objective.config.get("reward_source") != "parent_outcome_only":
        raise N1PlanValidationError("N1 source reward must remain parent outcome only")
    if cell.action_objective.config.get("token_ownership") != "assistant_response_only":
        raise N1PlanValidationError("N1 token ownership must exclude environment/tool tokens")
    if not cell.sampling_policy.per_response_rng:
        raise N1PlanValidationError("N1 sampling requires per-response RNG identity")


def _validate_cell_differences(
    cells: tuple[SystemCellManifestV1, ...],
    allowed_prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    baseline = _flatten(cells[0].to_dict())
    differences: set[str] = set()
    for cell in cells[1:]:
        candidate = _flatten(cell.to_dict())
        for path in set(baseline) | set(candidate):
            if path in {"system_cell_id"}:
                continue
            if baseline.get(path) != candidate.get(path):
                differences.add(path)
    forbidden = sorted(
        path
        for path in differences
        if not any(path == prefix or path.startswith(f"{prefix}.") for prefix in allowed_prefixes)
    )
    if forbidden:
        raise N1PlanValidationError(
            f"candidate system cells differ outside declaration: {forbidden}"
        )
    if not differences:
        raise N1PlanValidationError("candidate system cells are scientifically identical")
    return tuple(sorted(differences))


def _load_task_manifest(
    reference: ArtifactReferenceV1,
    *,
    workspace: Path,
    data_root: Path,
    label: str,
) -> AlfworldTaskManifest:
    path = reference.resolve(workspace, label=label)
    try:
        manifest = AlfworldTaskManifest.read(path)
    except (OSError, TypeError, ValueError) as error:
        raise N1PlanValidationError(f"cannot load {label}: {error}") from error
    issues = manifest.validate(data_root)
    if issues:
        details = "; ".join(f"{issue.code}:{issue.path}:{issue.message}" for issue in issues[:10])
        raise N1PlanValidationError(f"{label} failed runtime-free validation: {details}")
    return manifest


def _population_overlap(
    left: AlfworldTaskManifest,
    right: AlfworldTaskManifest,
) -> dict[str, list[str]]:
    return {
        "task_ids": sorted(
            {task.task_id for task in left.tasks} & {task.task_id for task in right.tasks}
        ),
        "group_uids": sorted(
            {task.group_uid for task in left.tasks} & {task.group_uid for task in right.tasks}
        ),
    }


def _closed_population_overlap(
    active: tuple[AlfworldTaskManifest, ...],
    closed: tuple[AlfworldTaskManifest, ...],
) -> dict[str, list[str]]:
    active_tasks = {task.task_id for manifest in active for task in manifest.tasks}
    active_groups = {task.group_uid for manifest in active for task in manifest.tasks}
    closed_tasks = {task.task_id for manifest in closed for task in manifest.tasks}
    closed_groups = {task.group_uid for manifest in closed for task in manifest.tasks}
    return {
        "task_ids": sorted(active_tasks & closed_tasks),
        "group_uids": sorted(active_groups & closed_groups),
    }


def _validate_n1_horizon_bindings(
    manifest: HorizonFactorManifestV1,
    *,
    cells: tuple[SystemCellManifestV1, ...],
    target_manifest_file_sha256: str,
    max_episode_steps: int,
) -> dict[str, Any]:
    if manifest.intrinsic_horizon_claimed or any(
        getattr(manifest, field).intrinsic_horizon
        for field in (
            "goal_distance",
            "dependency_depth",
            "interaction_cap",
            "context_budget",
            "interruption",
            "actor_competence",
            "memory_runtime",
            "action_objective",
            "expert_trace_length_proxy",
        )
    ):
        raise N1PlanValidationError("N1 is not an intrinsic-horizon claim")
    expected_task_factors = {
        "goal_distance": "expert_reference_steps_proxy_v1",
        "dependency_depth": "goal_graph_dependency_depth_v1",
        "expert_trace_length_proxy": "expert_low_level_trace_length_tercile_v1",
    }
    for name, operationalization in expected_task_factors.items():
        factor = getattr(manifest, name)
        if (
            factor.mode != "observed_proxy"
            or factor.operationalization_id != operationalization
            or factor.source_contract_sha256 != target_manifest_file_sha256
        ):
            raise N1PlanValidationError(f"N1 horizon task factor is not target-bound: {name}")

    cell_ids = tuple(cell.system_cell_id for cell in cells)
    competence = manifest.actor_competence
    competence_source = canonical_sha256(
        {"candidate_system_cell_sha256s": [cell.sha256 for cell in cells]}
    )
    if (
        competence.mode != "manipulated"
        or competence.levels != cell_ids
        or competence.source_contract_sha256 != competence_source
    ):
        raise N1PlanValidationError("actor_competence factor is not bound to candidate cells")

    context_levels = tuple(
        sorted({str(cell.context.config["max_prompt_tokens"]) for cell in cells})
    )
    context_source = canonical_sha256(
        {"context_component_sha256s": sorted({cell.context.sha256 for cell in cells})}
    )
    context_mode = "held_constant" if len(context_levels) == 1 else "manipulated"
    context = manifest.context_budget
    if (
        context.mode != context_mode
        or context.levels != context_levels
        or context.source_contract_sha256 != context_source
    ):
        raise N1PlanValidationError("context_budget factor is not bound to candidate cells")

    memory_levels = tuple(
        sorted(
            {
                f"{cell.memory.component_id}@{cell.memory.revision}|"
                f"{cell.runtime.component_id}@{cell.runtime.revision}"
                for cell in cells
            }
        )
    )
    memory_source = canonical_sha256(
        {
            "memory_runtime_component_sha256s": sorted(
                {
                    canonical_sha256({"memory": cell.memory.sha256, "runtime": cell.runtime.sha256})
                    for cell in cells
                }
            )
        }
    )
    memory_mode = "held_constant" if len(memory_levels) == 1 else "manipulated"
    memory = manifest.memory_runtime
    if (
        memory.mode != memory_mode
        or memory.levels != memory_levels
        or memory.source_contract_sha256 != memory_source
    ):
        raise N1PlanValidationError("memory_runtime factor is not bound to candidate cells")

    objective_hashes = {cell.action_objective.sha256 for cell in cells}
    if len(objective_hashes) != 1:
        raise N1PlanValidationError("N1 candidate action objectives must remain identical")
    objective = manifest.action_objective
    if (
        objective.mode != "held_constant"
        or objective.levels != (cells[0].action_objective.component_id,)
        or objective.source_contract_sha256 != cells[0].action_objective.sha256
    ):
        raise N1PlanValidationError("action_objective factor is not system-cell bound")

    interaction = manifest.interaction_cap
    interaction_source = canonical_sha256({"max_episode_steps": max_episode_steps})
    if (
        interaction.mode != "held_constant"
        or interaction.levels != (str(max_episode_steps),)
        or interaction.source_contract_sha256 != interaction_source
    ):
        raise N1PlanValidationError("interaction_cap factor differs from rollout cap")
    interruption = manifest.interruption
    interruption_source = canonical_sha256(
        {
            "interruption": "none",
            "complete_fixed_group_required": True,
            "stop_after_candidate": False,
        }
    )
    if (
        interruption.mode != "held_constant"
        or interruption.levels != ("none",)
        or interruption.source_contract_sha256 != interruption_source
    ):
        raise N1PlanValidationError("interruption factor differs from complete-group contract")
    return {
        "actor_competence_source_sha256": competence_source,
        "context_budget_source_sha256": context_source,
        "memory_runtime_source_sha256": memory_source,
        "action_objective_source_sha256": cells[0].action_objective.sha256,
        "interaction_cap_source_sha256": interaction_source,
        "interruption_source_sha256": interruption_source,
        "target_task_manifest_file_sha256": target_manifest_file_sha256,
    }


def _validate_rollout(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "max_episode_steps",
        "source_transition_evidence_mode",
        "complete_fixed_group_required",
        "stop_after_candidate",
        "adaptive_prompt_change",
    }
    _expect_exact_keys(payload, expected, "rollout")
    steps = _positive_integer(payload["max_episode_steps"], "max_episode_steps")
    if payload["source_transition_evidence_mode"] != "snapshot_only":
        raise N1PlanValidationError("N1 efficient source collection must be snapshot_only")
    required_false = {
        "stop_after_candidate": False,
        "adaptive_prompt_change": False,
    }
    for key, expected_value in required_false.items():
        if payload[key] is not expected_value:
            raise N1PlanValidationError(f"unsafe N1 rollout flag: {key}")
    if payload["complete_fixed_group_required"] is not True:
        raise N1PlanValidationError("N1 must retain every fixed-group rollout")
    return {
        "max_episode_steps": steps,
        "source_transition_evidence_mode": "snapshot_only",
        "complete_fixed_group_required": True,
        "stop_after_candidate": False,
        "adaptive_prompt_change": False,
    }


def _validate_selection_gate(
    payload: Mapping[str, Any],
    *,
    candidate_cell_ids: tuple[str, ...],
) -> dict[str, Any]:
    expected = {
        "primary_unit",
        "min_direct_support_task_rate",
        "max_direct_support_task_rate",
        "min_valid_semantic_action_rate",
        "min_admissible_semantic_action_rate",
        "min_failed_trajectories_with_child_effect",
        "min_failed_tasks_with_child_effect",
        "selection_rule",
        "tie_break_cell_order",
        "uses_target_outcomes",
        "selected_cell_count",
    }
    _expect_exact_keys(payload, expected, "selection_gate")
    if payload["primary_unit"] != "task_instance":
        raise N1PlanValidationError("system-cell selection primary unit must be task_instance")
    minimum = _rate(payload["min_direct_support_task_rate"], "minimum direct support")
    maximum = _rate(payload["max_direct_support_task_rate"], "maximum direct support")
    if not minimum < maximum < 1.0:
        raise N1PlanValidationError("direct-support gate must be ordered and non-saturating")
    valid_rate = _rate(
        payload["min_valid_semantic_action_rate"],
        "minimum valid semantic action rate",
    )
    admissible_rate = _rate(
        payload["min_admissible_semantic_action_rate"],
        "minimum admissible semantic action rate",
    )
    if admissible_rate > valid_rate:
        raise N1PlanValidationError(
            "admissible-action threshold cannot exceed valid-action threshold"
        )
    child_count = _positive_integer(
        payload["min_failed_trajectories_with_child_effect"],
        "min_failed_trajectories_with_child_effect",
    )
    child_task_count = _positive_integer(
        payload["min_failed_tasks_with_child_effect"],
        "min_failed_tasks_with_child_effect",
    )
    if payload["selection_rule"] != "smallest_group_then_declared_cell_order":
        raise N1PlanValidationError("system-cell selection rule changed")
    tie_break = _string_tuple(payload["tie_break_cell_order"], "tie_break_cell_order")
    if tie_break != candidate_cell_ids:
        raise N1PlanValidationError(
            "tie-break order must list every candidate system cell exactly once"
        )
    if payload["uses_target_outcomes"] is not False:
        raise N1PlanValidationError("system-cell selection must not use target outcomes")
    if payload["selected_cell_count"] != 1:
        raise N1PlanValidationError("N1 must select exactly one target system cell")
    return {
        "primary_unit": "task_instance",
        "min_direct_support_task_rate": minimum,
        "max_direct_support_task_rate": maximum,
        "min_valid_semantic_action_rate": valid_rate,
        "min_admissible_semantic_action_rate": admissible_rate,
        "min_failed_trajectories_with_child_effect": child_count,
        "min_failed_tasks_with_child_effect": child_task_count,
        "selection_rule": "smallest_group_then_declared_cell_order",
        "tie_break_cell_order": list(tie_break),
        "uses_target_outcomes": False,
        "selected_cell_count": 1,
    }


def _validate_seed_plan(
    payload: Mapping[str, Any],
    *,
    calibration: AlfworldTaskManifest,
    target: AlfworldTaskManifest,
) -> dict[str, Any]:
    expected = {
        "calibration_environment_seeds",
        "target_environment_seeds",
        "calibration_sampling_seed_base",
        "target_sampling_seed_base",
        "forbidden_seeds",
        "sampling_seed_schedule",
        "common_random_numbers_across_candidate_cells",
    }
    _expect_exact_keys(payload, expected, "seed_plan")
    calibration_seeds = _seed_mapping(
        payload["calibration_environment_seeds"],
        expected_task_ids={task.task_id for task in calibration.tasks},
        label="calibration_environment_seeds",
    )
    target_seeds = _seed_mapping(
        payload["target_environment_seeds"],
        expected_task_ids={task.task_id for task in target.tasks},
        label="target_environment_seeds",
    )
    calibration_base = _nonnegative_integer(
        payload["calibration_sampling_seed_base"],
        "calibration_sampling_seed_base",
    )
    target_base = _nonnegative_integer(
        payload["target_sampling_seed_base"],
        "target_sampling_seed_base",
    )
    forbidden = frozenset(
        _nonnegative_integer(value, "forbidden_seed")
        for value in _sequence(payload["forbidden_seeds"], "forbidden_seeds")
    )
    all_planned = [*calibration_seeds.values(), *target_seeds.values()]
    if len(all_planned) != len(set(all_planned)):
        raise N1PlanValidationError("environment seeds must be unique across populations")
    if calibration_base == target_base:
        raise N1PlanValidationError("calibration and target sampling seed bases must differ")
    if any(value in forbidden for value in (*all_planned, calibration_base, target_base)):
        raise N1PlanValidationError("planned N1 seed reuses a forbidden seed")
    if payload["sampling_seed_schedule"] != (
        "base_plus_trajectory_ordinal_times_horizon_plus_step"
    ):
        raise N1PlanValidationError("N1 sampling-seed schedule changed")
    if payload["common_random_numbers_across_candidate_cells"] is not True:
        raise N1PlanValidationError(
            "calibration candidates must share the frozen response-seed schedule"
        )
    return {
        "calibration_environment_seeds": dict(sorted(calibration_seeds.items())),
        "target_environment_seeds": dict(sorted(target_seeds.items())),
        "calibration_sampling_seed_base": calibration_base,
        "target_sampling_seed_base": target_base,
        "forbidden_seeds": sorted(forbidden),
        "sampling_seed_schedule": ("base_plus_trajectory_ordinal_times_horizon_plus_step"),
        "common_random_numbers_across_candidate_cells": True,
        "seed_plan_sha256": canonical_sha256(payload),
    }


def _validate_power(payload: Mapping[str, Any], *, target_task_count: int) -> dict[str, Any]:
    expected = {
        "primary_unit",
        "interval_method",
        "planned_target_task_count",
        "worst_case_rate",
        "max_wilson_half_width",
        "reported_worst_case_half_width",
        "clustered_by_task",
        "transitions_treated_independent",
    }
    _expect_exact_keys(payload, expected, "power")
    if payload["primary_unit"] != "task_instance" or payload["interval_method"] != "wilson_95":
        raise N1PlanValidationError("N1 power must use task-level Wilson-95 precision")
    planned = _positive_integer(payload["planned_target_task_count"], "planned_target_task_count")
    if planned != target_task_count:
        raise N1PlanValidationError("power task count differs from target manifest")
    worst_rate = _rate(payload["worst_case_rate"], "worst_case_rate")
    if worst_rate != 0.5:
        raise N1PlanValidationError("N1 precision gate must use worst_case_rate=0.5")
    maximum = _positive_number(payload["max_wilson_half_width"], "max_wilson_half_width")
    reported = _positive_number(
        payload["reported_worst_case_half_width"],
        "reported_worst_case_half_width",
    )
    actual = _wilson_half_width(planned, worst_rate)
    if not math.isclose(reported, actual, rel_tol=0.0, abs_tol=1e-12):
        raise N1PlanValidationError(f"reported Wilson half-width changed: {reported} != {actual}")
    if actual > maximum:
        raise N1PlanValidationError(
            f"target population is underpowered for precision: {actual} > {maximum}"
        )
    if (
        payload["clustered_by_task"] is not True
        or payload["transitions_treated_independent"] is not False
    ):
        raise N1PlanValidationError("N1 power must retain task clustering")
    return {
        **dict(payload),
        "computed_worst_case_half_width": actual,
        "power_verified": True,
    }


def _validate_controls(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "control_ids",
        "physical_call_matching",
        "logical_only_matching",
        "frozen_before_target",
    }
    _expect_exact_keys(payload, expected, "controls")
    control_ids = _string_tuple(payload["control_ids"], "control_ids")
    if control_ids != N1_CONTROL_IDS:
        raise N1PlanValidationError("N1 matched-control ladder changed")
    if payload["physical_call_matching"] != "measured_reset_and_environment_step_calls":
        raise N1PlanValidationError("N1 controls must match measured physical calls")
    if payload["logical_only_matching"] is not False or payload["frozen_before_target"] is not True:
        raise N1PlanValidationError("N1 control matching/freeze flags are unsafe")
    return dict(payload)


def _validate_budgets(
    payload: Mapping[str, Any],
    *,
    calibration_task_count: int,
    target_task_count: int,
    candidate_cell_count: int,
    calibration_samples: int,
    target_samples: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    expected = {
        "max_model_samples",
        "max_logical_semantic_actions",
        "max_environment_reset_calls",
        "max_environment_step_calls",
        "max_optimizer_steps",
        "max_parameter_updates",
        "max_training_examples_admitted",
    }
    _expect_exact_keys(payload, expected, "budgets")
    values = {key: _nonnegative_integer(value, key) for key, value in payload.items()}
    minimum_source_actions = (
        calibration_task_count * calibration_samples * candidate_cell_count * max_episode_steps
        + target_task_count * target_samples * max_episode_steps
    )
    if values["max_model_samples"] < minimum_source_actions:
        raise N1PlanValidationError(
            "model-sample cap cannot cover the complete fixed source populations"
        )
    if values["max_logical_semantic_actions"] < values["max_model_samples"]:
        raise N1PlanValidationError("logical-action cap is below the model-sample cap")
    if (
        values["max_environment_reset_calls"] < 1
        or values["max_environment_step_calls"] < values["max_logical_semantic_actions"]
    ):
        raise N1PlanValidationError("physical environment-call caps are incomplete")
    for key in (
        "max_optimizer_steps",
        "max_parameter_updates",
        "max_training_examples_admitted",
    ):
        if values[key] != 0:
            raise N1PlanValidationError(f"N1 design requires {key}=0")
    return {**values, "minimum_complete_source_action_cap": minimum_source_actions}


def _validate_pipeline(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "state_capability_schema",
        "physical_call_counter_schema",
        "fragment_record_schema",
        "snapshot_transition_certification_schema",
        "composition_plan_schema",
        "e1_junction_replay_receipt_schema",
        "replay_evidence_schema",
        "composition_receipt_schema",
        "raw_fragments_training_eligible",
        "multi_source_on_policy_allowed",
    }
    _expect_exact_keys(payload, expected, "pipeline")
    locked = {
        "state_capability_schema": ALFWORLD_STATE_CAPABILITY_SCHEMA,
        "physical_call_counter_schema": ALFWORLD_PHYSICAL_CALL_COUNTER_SCHEMA,
        "fragment_record_schema": FRAGMENT_RECORD_SCHEMA,
        "snapshot_transition_certification_schema": (SNAPSHOT_TRANSITION_CERTIFICATION_SCHEMA),
        "composition_plan_schema": COMPOSITION_PLAN_SCHEMA,
        "e1_junction_replay_receipt_schema": E1_JUNCTION_REPLAY_RECEIPT_SCHEMA,
        "replay_evidence_schema": REPLAY_EVIDENCE_SCHEMA,
        "composition_receipt_schema": COMPOSITION_RECEIPT_SCHEMA,
        "raw_fragments_training_eligible": False,
        "multi_source_on_policy_allowed": False,
    }
    if dict(payload) != locked:
        raise N1PlanValidationError("N1 pipeline schemas or training guards changed")
    return locked


def _population_receipt(
    reference: ArtifactReferenceV1,
    manifest: AlfworldTaskManifest,
) -> dict[str, Any]:
    return {
        "reference": reference.to_dict(),
        "task_count": len(manifest.tasks),
        "task_ids": sorted(task.task_id for task in manifest.tasks),
        "group_uids": sorted(task.group_uid for task in manifest.tasks),
        "splits": sorted({task.split for task in manifest.tasks}),
        "task_types": sorted({task.task_type for task in manifest.tasks}),
        "horizon_buckets": sorted({task.horizon_bucket for task in manifest.tasks}),
    }


def _seed_mapping(
    value: Any,
    *,
    expected_task_ids: set[str],
    label: str,
) -> dict[str, int]:
    payload = _mapping(value, label)
    if set(payload) != expected_task_ids:
        raise N1PlanValidationError(
            f"{label} task keys differ: missing={sorted(expected_task_ids - set(payload))}, "
            f"unknown={sorted(set(payload) - expected_task_ids)}"
        )
    return {
        str(task_id): _nonnegative_integer(seed, f"{label}.{task_id}")
        for task_id, seed in payload.items()
    }


def _flatten(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(value[key], prefix=path))
        return output
    if isinstance(value, (list, tuple)):
        output = {}
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            output.update(_flatten(item, prefix=path))
        if not value:
            output[prefix] = []
        return output
    return {prefix: value}


def _wilson_half_width(total: int, rate: float) -> float:
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    radius = (
        z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    )
    return radius


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise N1PlanValidationError(f"cannot read N1 plan {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise N1PlanValidationError("N1 plan must be a mapping")
    return dict(value)


def _expect_exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise N1PlanValidationError(f"{label} fields differ: missing={missing}, unknown={unknown}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise N1PlanValidationError(f"{label} must be a mapping")
    return dict(value)


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise N1PlanValidationError(f"{label} must be a sequence")
    return tuple(value)


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    rows = tuple(str(item) for item in _sequence(value, label))
    if any(not item or item != item.strip() for item in rows) or len(rows) != len(set(rows)):
        raise N1PlanValidationError(f"{label} must contain unique canonical strings")
    return rows


def _identifier(value: Any, label: str) -> str:
    text = str(value)
    if not text or text != text.strip() or any(character in text for character in "\r\n"):
        raise N1PlanValidationError(f"{label} must be a canonical identifier")
    return text


def _positive_integer(value: Any, label: str, *, minimum: int = 1) -> int:
    result = _nonnegative_integer(value, label)
    if result < minimum:
        raise N1PlanValidationError(f"{label} must be >= {minimum}")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise N1PlanValidationError(f"{label} must be a non-negative integer")
    return value


def _rate(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0 <= result <= 1:
        raise N1PlanValidationError(f"{label} must be in [0, 1]")
    return result


def _positive_number(value: Any, label: str) -> float:
    result = _number(value, label)
    if result <= 0:
        raise N1PlanValidationError(f"{label} must be positive")
    return result


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise N1PlanValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise N1PlanValidationError(f"{label} must be finite")
    return result


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise N1PlanValidationError(f"{name} must be a lowercase SHA-256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "N1_CLAIM_SCOPE",
    "N1_CONTROL_IDS",
    "N1_DESIGN_PLAN_SCHEMA",
    "N1_PLAN_VALIDATION_SCHEMA",
    "ArtifactReferenceV1",
    "N1PlanValidationError",
    "ValidatedN1Plan",
    "validate_n1_design_plan",
    "write_n1_plan_validation_receipt",
]
