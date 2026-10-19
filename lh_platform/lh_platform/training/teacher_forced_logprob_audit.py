"""Preregistered teacher-forced audit of frozen behavior-policy evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lh_platform.training.agentic_rl_config import (
    AgenticRLConfigError,
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
    load_agentic_rollout_artifact,
)
from lh_platform.training.config_provenance import repository_file_binding
from lh_platform.training.controlled_actor_factorial_run import (
    CONTROLLED_ACTOR_FACTORIAL_RESULT_SCHEMA,
    controlled_factorial_repository_bindings,
)
from lh_platform.training.dataset import sha256_file
from lh_platform.training.transformers_peft_policy import (
    TransformersPeftPolicy,
    checkpoint_tree_sha256,
)

TEACHER_FORCED_AUDIT_SPEC_SCHEMA = "lh_teacher_forced_old_logprob_audit_spec/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TeacherForcedLogprobAuditSpec:
    config_path: Path
    workspace_root: Path
    audit_id: str
    controlled_factorial_path: Path
    controlled_factorial_sha256: str
    source_run_commit: str
    source_result_sha256: str
    rollout_sha256: Mapping[str, str]
    generation_micro_batch_size: int
    max_absolute_error: float
    max_mean_absolute_error: float
    required_token_coverage: float
    claim_boundary: Mapping[str, bool]
    file_sha256: str
    semantic_sha256: str


@dataclass(frozen=True)
class TeacherForcedAuditSource:
    result_path: Path
    result: Mapping[str, Any]
    preflight_path: Path
    preflight: Mapping[str, Any]
    rollout_paths: Mapping[str, Path]
    rollouts: Mapping[str, AgenticRolloutBatch]


class TeacherForcedLogprobAuditError(RuntimeError):
    """Raised before model load when frozen evidence is not the registered run."""


TEACHER_FORCED_AUDIT_RESULT_SCHEMA = "lh_teacher_forced_old_logprob_audit_result/v1"


def load_teacher_forced_logprob_audit_spec(
    config_path: str | Path,
) -> TeacherForcedLogprobAuditSpec:
    """Load an exact frozen-run binding and numeric old-logprob gate."""

    path = Path(config_path).resolve()
    workspace_root = _workspace_root(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AgenticRLConfigError(
            f"cannot load teacher-forced audit config {path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise AgenticRLConfigError("teacher-forced audit config must be a mapping")
    _expect_keys(
        raw,
        {
            "schema",
            "id",
            "controlled_factorial_ref",
            "source_run",
            "runtime",
            "thresholds",
            "claim_boundary",
        },
        "teacher-forced audit config",
    )
    if raw["schema"] != TEACHER_FORCED_AUDIT_SPEC_SCHEMA:
        raise AgenticRLConfigError("unsupported teacher-forced audit schema")
    audit_id = _nonempty_string(raw["id"], "teacher-forced audit id")
    controlled_reference = Path(
        _nonempty_string(
            raw["controlled_factorial_ref"],
            "controlled_factorial_ref",
        )
    )
    if controlled_reference.is_absolute():
        raise AgenticRLConfigError("controlled_factorial_ref must be workspace-relative")
    controlled_path = (workspace_root / controlled_reference).resolve()
    if workspace_root not in controlled_path.parents:
        raise AgenticRLConfigError("controlled_factorial_ref escapes the workspace")
    controlled = load_controlled_actor_factorial_manifest(controlled_path)

    source = _mapping(raw["source_run"], "source_run")
    _expect_keys(source, {"commit", "result_sha256", "rollouts"}, "source_run")
    commit = _nonempty_string(source["commit"], "source_run.commit")
    if not _COMMIT.fullmatch(commit):
        raise AgenticRLConfigError("source_run.commit must be a full Git commit")
    result_sha256 = _sha256(source["result_sha256"], "source_run.result_sha256")
    rollouts = _mapping(source["rollouts"], "source_run.rollouts")
    _expect_keys(rollouts, {"none", "oracle_goal_graph"}, "source_run.rollouts")
    rollout_sha256 = {
        awareness: _sha256(value, f"source_run.rollouts.{awareness}")
        for awareness, value in rollouts.items()
    }

    runtime = _mapping(raw["runtime"], "runtime")
    _expect_keys(runtime, {"generation_micro_batch_size"}, "runtime")
    generation_micro_batch_size = runtime["generation_micro_batch_size"]
    if (
        isinstance(generation_micro_batch_size, bool)
        or not isinstance(generation_micro_batch_size, int)
        or generation_micro_batch_size < 1
    ):
        raise AgenticRLConfigError("runtime.generation_micro_batch_size must be a positive integer")

    thresholds = _mapping(raw["thresholds"], "thresholds")
    _expect_keys(
        thresholds,
        {
            "max_absolute_error",
            "max_mean_absolute_error",
            "required_token_coverage",
        },
        "thresholds",
    )
    max_absolute_error = _nonnegative_float(
        thresholds["max_absolute_error"],
        "thresholds.max_absolute_error",
    )
    max_mean_absolute_error = _nonnegative_float(
        thresholds["max_mean_absolute_error"],
        "thresholds.max_mean_absolute_error",
    )
    required_token_coverage = _nonnegative_float(
        thresholds["required_token_coverage"],
        "thresholds.required_token_coverage",
    )
    if required_token_coverage != 1.0:
        raise AgenticRLConfigError("teacher-forced audit requires complete token coverage")

    claim_boundary = _mapping(raw["claim_boundary"], "claim_boundary")
    expected_claim = {
        "policy_update_executed": False,
        "policy_improvement": False,
        "old_policy_probability_consistency_only": True,
    }
    if dict(claim_boundary) != expected_claim:
        raise AgenticRLConfigError("teacher-forced audit claim boundary is unsafe")

    semantic_payload = {
        "schema": TEACHER_FORCED_AUDIT_SPEC_SCHEMA,
        "id": audit_id,
        "controlled_factorial_sha256": controlled.sha256,
        "source_run_commit": commit,
        "source_result_sha256": result_sha256,
        "rollout_sha256": rollout_sha256,
        "generation_micro_batch_size": generation_micro_batch_size,
        "max_absolute_error": max_absolute_error,
        "max_mean_absolute_error": max_mean_absolute_error,
        "required_token_coverage": required_token_coverage,
        "claim_boundary": expected_claim,
    }
    return TeacherForcedLogprobAuditSpec(
        config_path=path,
        workspace_root=workspace_root,
        audit_id=audit_id,
        controlled_factorial_path=controlled_path,
        controlled_factorial_sha256=controlled.sha256,
        source_run_commit=commit,
        source_result_sha256=result_sha256,
        rollout_sha256=rollout_sha256,
        generation_micro_batch_size=generation_micro_batch_size,
        max_absolute_error=max_absolute_error,
        max_mean_absolute_error=max_mean_absolute_error,
        required_token_coverage=required_token_coverage,
        claim_boundary=expected_claim,
        file_sha256=sha256_file(path),
        semantic_sha256=canonical_sha256(semantic_payload),
    )


def validate_teacher_forced_audit_source(
    spec: TeacherForcedLogprobAuditSpec,
    result_path: str | Path,
) -> TeacherForcedAuditSource:
    """Bind the exact result, preflight, and rollouts before allocating a model."""

    source_path = Path(result_path).resolve()
    if sha256_file(source_path) != spec.source_result_sha256:
        raise TeacherForcedLogprobAuditError(
            "controlled factorial result SHA256 differs from the audit registration"
        )
    result = _json_mapping(source_path, "controlled factorial result")
    if result.get("schema") != CONTROLLED_ACTOR_FACTORIAL_RESULT_SCHEMA:
        raise TeacherForcedLogprobAuditError("controlled factorial result schema is unsupported")
    if result.get("controlled_factorial_sha256") != (spec.controlled_factorial_sha256):
        raise TeacherForcedLogprobAuditError(
            "controlled factorial semantic identity differs from the audit spec"
        )
    if result.get("policy_update_executed") is not False:
        raise TeacherForcedLogprobAuditError(
            "old-logprob audit requires an immutable pre-update policy rollout"
        )
    if result.get("all_awareness_credit_difference_gates_passed") is not True:
        raise TeacherForcedLogprobAuditError(
            "registered frozen run did not pass both credit-difference gates"
        )

    preflight_path = _artifact_path(result.get("preflight_path"), source_path.parent)
    if sha256_file(preflight_path) != result.get("preflight_sha256"):
        raise TeacherForcedLogprobAuditError(
            "controlled factorial preflight SHA256 differs from its result binding"
        )
    preflight = _json_mapping(preflight_path, "controlled factorial preflight")
    binding = _mapping(
        _mapping(preflight.get("controlled_factorial"), "preflight controlled_factorial").get(
            "repository_binding"
        ),
        "preflight repository_binding",
    )
    if binding.get("revision") != spec.source_run_commit:
        raise TeacherForcedLogprobAuditError(
            "controlled factorial preflight commit differs from the audit registration"
        )
    if binding.get("file_sha256") != sha256_file(spec.controlled_factorial_path):
        raise TeacherForcedLogprobAuditError(
            "controlled factorial file differs from the source-run binding"
        )

    awareness_results = _mapping(result.get("awareness_results"), "awareness_results")
    if set(awareness_results) != set(spec.rollout_sha256):
        raise TeacherForcedLogprobAuditError(
            "controlled factorial result does not contain both registered awareness modes"
        )
    initial_checkpoint = result.get("initial_policy_checkpoint_sha256")
    rollout_paths: dict[str, Path] = {}
    rollouts: dict[str, AgenticRolloutBatch] = {}
    for awareness, expected_sha256 in spec.rollout_sha256.items():
        awareness_result = _mapping(
            awareness_results[awareness],
            f"awareness_results.{awareness}",
        )
        credit_gate = _mapping(
            awareness_result.get("credit_gate"),
            f"awareness_results.{awareness}.credit_gate",
        )
        if credit_gate.get("credit_difference_gate_passed") is not True:
            raise TeacherForcedLogprobAuditError(f"{awareness} credit difference gate did not pass")
        rollout_path = _artifact_path(
            awareness_result.get("raw_rollout_path"),
            source_path.parent,
        )
        actual_sha256 = sha256_file(rollout_path)
        if actual_sha256 != expected_sha256 or actual_sha256 != awareness_result.get(
            "raw_rollout_sha256"
        ):
            raise TeacherForcedLogprobAuditError(
                f"{awareness} rollout SHA256 differs from the audit registration"
            )
        rollout = load_agentic_rollout_artifact(rollout_path)
        if rollout.policy_checkpoint_sha256 != initial_checkpoint:
            raise TeacherForcedLogprobAuditError(
                f"{awareness} rollout checkpoint differs from the factorial result"
            )
        rollout_paths[awareness] = rollout_path
        rollouts[awareness] = rollout
    return TeacherForcedAuditSource(
        result_path=source_path,
        result=result,
        preflight_path=preflight_path,
        preflight=preflight,
        rollout_paths=rollout_paths,
        rollouts=rollouts,
    )


def run_teacher_forced_old_logprob_audit(
    config_path: str | Path,
    *,
    source_result: str | Path,
    model_path: str | Path,
    source_root: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    device: str,
    preflight_only: bool,
    allow_shared_gpu: bool,
) -> Path:
    """Audit frozen behavior probabilities at the exact initial checkpoint."""

    spec = load_teacher_forced_logprob_audit_spec(config_path)
    source = validate_teacher_forced_audit_source(spec, source_result)
    target = Path(output_path).resolve()
    reserve_teacher_forced_audit_output_directory(
        target,
        protected_paths=(
            spec.config_path,
            spec.controlled_factorial_path,
            source.result_path,
            source.preflight_path,
            *source.rollout_paths.values(),
            Path(model_path).resolve(),
            Path(source_root).resolve(),
            Path(data_root).resolve(),
        ),
    )
    event_logger = RunEventLogger(
        target.with_name(f"{target.stem}.events.jsonl"),
        run_id=canonical_sha256(
            {
                "audit_spec": spec.semantic_sha256,
                "source_result": spec.source_result_sha256,
                "output": str(target),
            }
        )[:32],
    )
    event_logger.emit(
        "teacher_forced_old_logprob_audit_started",
        audit_spec_sha256=spec.semantic_sha256,
        source_result_sha256=spec.source_result_sha256,
        requested_device=device,
        preflight_only=preflight_only,
    )

    controlled = load_controlled_actor_factorial_manifest(spec.controlled_factorial_path)
    factorial = load_agentic_rl_factorial_manifest(
        controlled.workspace_root / controlled.payload["factorial"]["path"]
    )
    unaware_experiment = load_agentic_rl_experiment(
        controlled.workspace_root / factorial.payload["paired_experiments"]["none"]["path"]
    )
    runtime_args = bind_agentic_rl_runtime(
        unaware_experiment,
        AgenticRLRuntimeBindings(
            model_path=Path(model_path),
            source_root=Path(source_root),
            data_root=Path(data_root),
            output_dir=target.parent / f"{target.stem}_base_preflight",
            device=device,
            preflight_only=True,
            allow_shared_gpu=allow_shared_gpu,
        ),
    )
    runtime_args.generation_micro_batch_size = spec.generation_micro_batch_size
    base_preflight_path = run_p0(runtime_args)
    base_preflight = _json_mapping(base_preflight_path, "base policy preflight")
    audit_config_binding = repository_file_binding(spec.config_path)
    controlled_bindings = controlled_factorial_repository_bindings(controlled)
    preflight = {
        "schema": "lh_teacher_forced_old_logprob_audit_preflight/v1",
        "audit_spec": {
            "path": str(spec.config_path),
            "file_sha256": spec.file_sha256,
            "semantic_sha256": spec.semantic_sha256,
            "repository_binding": audit_config_binding,
        },
        "source_result": {
            "path": str(source.result_path),
            "sha256": spec.source_result_sha256,
            "run_commit": spec.source_run_commit,
            "preflight_path": str(source.preflight_path),
            "preflight_sha256": sha256_file(source.preflight_path),
            "rollouts": {
                awareness: {
                    "path": str(source.rollout_paths[awareness]),
                    "sha256": spec.rollout_sha256[awareness],
                    "sample_count": len(source.rollouts[awareness].policy_samples),
                    "action_token_count": sum(
                        len(sample.action_token_ids)
                        for sample in source.rollouts[awareness].policy_samples
                    ),
                }
                for awareness in sorted(source.rollouts)
            },
        },
        "thresholds": {
            "max_absolute_error": spec.max_absolute_error,
            "max_mean_absolute_error": spec.max_mean_absolute_error,
            "required_token_coverage": spec.required_token_coverage,
        },
        "base_preflight_path": str(base_preflight_path),
        "base_preflight_sha256": sha256_file(base_preflight_path),
        "base_checkpoint_sha256": base_preflight["base_checkpoint_sha256"],
        "source_revision": base_preflight["source_revision"],
        "controlled_preregistration_bindings": [dict(binding) for binding in controlled_bindings],
        "formal_launch_eligible": bool(
            audit_config_binding["matches_head"]
            and all(binding["matches_head"] for binding in controlled_bindings)
        ),
        "gpu_loaded": False,
        "claim_scope": "old_policy_probability_consistency_preflight_only",
    }
    preflight_path = target.with_name(f"{target.stem}.preflight.json")
    _write_json_exclusive(preflight_path, preflight)
    event_logger.emit(
        "teacher_forced_old_logprob_audit_preflight_written",
        path=str(preflight_path),
        sha256=sha256_file(preflight_path),
        formal_launch_eligible=preflight["formal_launch_eligible"],
        gpu_loaded=False,
    )
    if preflight_only:
        return preflight_path
    if not preflight["formal_launch_eligible"]:
        raise TeacherForcedLogprobAuditError(
            "teacher-forced audit config tree must be committed unchanged"
        )

    compute_apps = query_compute_apps() if device.startswith("cuda") else ()
    assert_exclusive_gpu(
        device=device,
        allow_shared_gpu=allow_shared_gpu,
        compute_apps=compute_apps,
    )
    base_checkpoint_sha256 = checkpoint_tree_sha256(model_path)
    if base_checkpoint_sha256 != base_preflight["base_checkpoint_sha256"]:
        raise TeacherForcedLogprobAuditError(
            "model checkpoint changed after teacher-forced audit preflight"
        )
    policy_args = argparse.Namespace(**vars(runtime_args))
    policy_args.preflight_only = False
    policy_config = transformers_policy_config_from_args(
        policy_args,
        base_checkpoint_sha256=base_checkpoint_sha256,
    )
    event_logger.emit(
        "teacher_forced_old_logprob_policy_load_started",
        base_checkpoint_sha256=base_checkpoint_sha256,
        compute_apps_before_launch=list(compute_apps),
    )
    policy = TransformersPeftPolicy(policy_config)
    expected_policy_checkpoint = source.result["initial_policy_checkpoint_sha256"]
    if policy.checkpoint_sha256 != expected_policy_checkpoint:
        raise TeacherForcedLogprobAuditError(
            "reloaded initial policy identity differs from the frozen rollouts"
        )
    event_logger.emit(
        "teacher_forced_old_logprob_policy_loaded",
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        adapter_initial_delta_zero=policy.adapter_initial_delta_zero,
    )

    audits: dict[str, Mapping[str, Any]] = {}
    for awareness in sorted(source.rollouts):
        audit = policy.audit_old_action_log_probs(
            source.rollouts[awareness].policy_samples,
            max_absolute_error=spec.max_absolute_error,
            max_mean_absolute_error=spec.max_mean_absolute_error,
            required_token_coverage=spec.required_token_coverage,
        )
        audits[awareness] = audit
        event_logger.emit(
            "teacher_forced_old_logprob_awareness_completed",
            actor_awareness_mode=awareness,
            sample_count=audit["sample_count"],
            action_token_count=audit["action_token_count"],
            token_coverage=audit["token_coverage"],
            max_absolute_error=audit["max_absolute_error"],
            mean_absolute_error=audit["mean_absolute_error"],
            passed=audit["passed"],
        )
    passed = all(bool(audit["passed"]) for audit in audits.values())
    result = {
        "schema": TEACHER_FORCED_AUDIT_RESULT_SCHEMA,
        "audit_spec_path": str(spec.config_path),
        "audit_spec_file_sha256": spec.file_sha256,
        "audit_spec_semantic_sha256": spec.semantic_sha256,
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "source_result_path": str(source.result_path),
        "source_result_sha256": spec.source_result_sha256,
        "policy_checkpoint_sha256": policy.checkpoint_sha256,
        "awareness_audits": audits,
        "all_awareness_old_logprob_gates_passed": passed,
        "paired_policy_update_eligible": passed,
        "policy_update_executed": False,
        "blocked_by": [] if passed else ["teacher_forced_old_logprob_mismatch"],
        "next_action": (
            "run_one_paired_update_from_frozen_rollout"
            if passed
            else "fix_generation_update_probability_mismatch"
        ),
        "claim_boundary": dict(spec.claim_boundary),
        "claim_scope": "old_policy_probability_consistency_not_policy_improvement",
    }
    _write_json_exclusive(target, result)
    event_logger.emit(
        "teacher_forced_old_logprob_audit_completed",
        result_path=str(target),
        result_sha256=sha256_file(target),
        all_awareness_old_logprob_gates_passed=passed,
        policy_update_executed=False,
        next_action=result["next_action"],
    )
    return target


def _workspace_root(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "configs":
            return parent.parent.resolve()
    raise AgenticRLConfigError("audit config must live below workspace configs")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgenticRLConfigError(f"{label} must be a mapping")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise AgenticRLConfigError(f"{label} keys differ from the registered schema")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgenticRLConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if not _SHA256.fullmatch(text):
        raise AgenticRLConfigError(f"{label} must be a SHA256 digest")
    return text


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgenticRLConfigError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise AgenticRLConfigError(f"{label} must be finite and non-negative")
    return number


def _json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TeacherForcedLogprobAuditError(f"cannot read {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise TeacherForcedLogprobAuditError(f"{label} must be a JSON mapping")
    return value


def _artifact_path(value: Any, parent: Path) -> Path:
    text = _nonempty_string(value, "artifact path")
    candidate = Path(text)
    return candidate.resolve() if candidate.is_absolute() else (parent / candidate).resolve()


def reserve_teacher_forced_audit_output_directory(
    output_path: str | Path,
    *,
    protected_paths: tuple[Path, ...],
) -> Path:
    """Claim a fresh evidence directory before any log or model allocation."""

    target = Path(output_path).resolve()
    output_dir = target.parent
    for raw_protected in protected_paths:
        protected = Path(raw_protected).resolve()
        if target == protected or output_dir == protected:
            raise TeacherForcedLogprobAuditError(
                "teacher-forced audit output collides with input evidence"
            )
        if protected.is_dir() and protected in target.parents:
            raise TeacherForcedLogprobAuditError(
                "teacher-forced audit output cannot live inside an input evidence tree"
            )
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise TeacherForcedLogprobAuditError(
            "teacher-forced audit output directory already exists"
        ) from error
    return output_dir


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    from lh_platform.core.atomic_io import atomic_write_json

    try:
        atomic_write_json(path, payload)
    except FileExistsError as error:
        raise TeacherForcedLogprobAuditError(
            f"teacher-forced audit artifact already exists: {path}"
        ) from error
