"""Validation and preflight gates for the ALFWorld G2PO mechanism study."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.g2po.manifest import (
    G2POManifestError,
    load_and_validate_g2po_alfworld_manifest_pair,
)
from lh_platform.training.state_identity import (
    ObservationMergeAudit,
    StateIdentityError,
    audit_observation_merges,
    verified_state_identity,
)
from lh_platform.training.trajectory_v2 import (
    summarize_trajectory_weights,
    validate_active_trajectory_records,
)

G2PO_EXPERIMENT_SCHEMA = "lh_unified_agentic_rl_experiment/v1"
_REQUIRED_TREATMENTS = ("grpo_u", "g2po_u", "g2po_tn", "verified_state_dp")

__all__ = [
    "G2PO_EXPERIMENT_SCHEMA",
    "G2POStudyConfigError",
    "VerifiedStateMechanismGate",
    "G2POStudyPreflight",
    "load_g2po_experiment_config",
    "preflight_g2po_study",
]


class G2POStudyConfigError(ValueError):
    """Raised before a comparison configuration can launch a smoke run."""


@dataclass(frozen=True)
class VerifiedStateMechanismGate:
    passed: bool
    verified_audit_states: int
    shared_verified_state_groups: int
    blocked_by: tuple[str, ...]


@dataclass(frozen=True)
class G2POStudyPreflight:
    active_transition_count: int
    trajectory_count: int
    actor_gradient_mass_correlation: float | None
    state_merge_audit: ObservationMergeAudit
    verified_state_gate: VerifiedStateMechanismGate
    blocked_by: tuple[str, ...]


def load_g2po_experiment_config(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise G2POStudyConfigError("G2PO experiment configuration must be a mapping")
    _validate_config(raw)
    _validate_registered_manifest_pair(raw, config_path=source)
    return raw


def preflight_g2po_study(
    records: Iterable[TrainingTrajectoryRecord],
    *,
    require_verified_state: bool = False,
    min_restore_verified_audit_states: int = 100,
) -> G2POStudyPreflight:
    """Evaluate data/identity gates before GPU work begins.

    This does not inspect a GPU: GPU admission belongs after source, length,
    restore and state preflight all pass.  It deliberately treats a missing
    verified-state mechanism as *not exercised*, not as evidence against or
    for an algorithm.
    """

    rows = validate_active_trajectory_records(records, require_v2=True)
    weights = summarize_trajectory_weights(rows, require_v2=True)
    audit = audit_observation_merges(rows)
    blocked: list[str] = []
    if require_verified_state:
        for row in rows:
            try:
                verified_state_identity(row)
            except StateIdentityError:
                blocked.append("verified_state_identity_missing_or_unproven")
                break
    if audit.verified_transitions < min_restore_verified_audit_states:
        blocked.append("mechanism_not_triggered_insufficient_restore_verified_states")
    if audit.shared_verified_state_groups < 1:
        blocked.append("mechanism_not_triggered_no_shared_verified_state")
    gate = VerifiedStateMechanismGate(
        passed=not blocked,
        verified_audit_states=audit.verified_transitions,
        shared_verified_state_groups=audit.shared_verified_state_groups,
        blocked_by=tuple(dict.fromkeys(blocked)),
    )
    return G2POStudyPreflight(
        active_transition_count=len(rows),
        trajectory_count=weights.trajectories,
        actor_gradient_mass_correlation=weights.rollout_length_gradient_mass_correlation,
        state_merge_audit=audit,
        verified_state_gate=gate,
        blocked_by=gate.blocked_by if require_verified_state else (),
    )


def _validate_config(raw: Mapping[str, Any]) -> None:
    if raw.get("schema") != G2PO_EXPERIMENT_SCHEMA:
        raise G2POStudyConfigError(f"expected schema {G2PO_EXPERIMENT_SCHEMA}")
    treatments = raw.get("treatments")
    if not isinstance(treatments, list):
        raise G2POStudyConfigError("treatments must be a list")
    treatment_ids = tuple(
        str(item.get("id", "")) for item in treatments if isinstance(item, Mapping)
    )
    if treatment_ids != _REQUIRED_TREATMENTS:
        raise G2POStudyConfigError(
            "treatments must be exactly GRPO-U, G2PO-U, G2PO-TN, Verified-State-DP in order"
        )
    expected_credit = {
        "grpo_u": "trajectory_outcome_only",
        "g2po_u": "g2po_source_credit",
        "g2po_tn": "g2po_trajectory_normalized",
        "verified_state_dp": "verified_state_dp",
    }
    if any(
        not isinstance(item, Mapping)
        or item.get("credit_mode") != expected_credit.get(str(item.get("id")))
        for item in treatments
    ):
        raise G2POStudyConfigError("every treatment must use its registered credit mode")
    environment = raw.get("environment")
    if not isinstance(environment, Mapping):
        raise G2POStudyConfigError("ALFWorld environment contract is required")
    if (
        environment.get("benchmark") != "alfworld_native"
        or environment.get("split") != "valid_unseen"
    ):
        raise G2POStudyConfigError("the registered comparison is ALFWorld valid_unseen only")
    if not all(
        isinstance(environment.get(name), str) and environment.get(name)
        for name in ("train_manifest", "eval_manifest")
    ):
        raise G2POStudyConfigError("the comparison requires explicit train/eval task manifests")
    if environment.get("task_manifest_hash_required") is not True:
        raise G2POStudyConfigError("every treatment must be locked to the same task manifest hash")
    model = raw.get("model")
    if (
        not isinstance(model, Mapping)
        or model.get("initial_checkpoint") != "Qwen/Qwen2.5-1.5B-Instruct"
    ):
        raise G2POStudyConfigError("the registered comparison fixes Qwen2.5-1.5B-Instruct")
    rollout = raw.get("rollout")
    if not isinstance(rollout, Mapping) or rollout.get("group_size") != 8:
        raise G2POStudyConfigError("the registered ALFWorld comparison fixes group_size=8")
    if rollout.get("max_episode_steps") != 50 or rollout.get("max_action_response_tokens") != 512:
        raise G2POStudyConfigError(
            "the registered ALFWorld comparison fixes max episode/action limits"
        )
    loss = raw.get("actor_loss")
    expected_actor_loss = {
        "unit": "trajectory_step_token",
        "per_token_weight": "1 / (trajectory_length * action_token_count)",
        "trajectory_total_gradient_mass": 1.0,
        "semantic_step_total_gradient_mass": "equal_within_trajectory",
    }
    if not isinstance(loss, Mapping) or any(
        loss.get(field) != expected for field, expected in expected_actor_loss.items()
    ):
        raise G2POStudyConfigError(
            "actor loss must use fixed trajectory-step-token weighting and unit mass"
        )
    budget = raw.get("budget")
    if not isinstance(budget, Mapping):
        raise G2POStudyConfigError("budget is required")
    if budget.get("training_seeds") != [0, 1, 2]:
        raise G2POStudyConfigError("the registered comparison requires seeds [0, 1, 2]")
    if budget.get("maximum_environment_action_steps_per_treatment_seed") != 320000:
        raise G2POStudyConfigError(
            "the registered per-treatment/seed cap is 320000 environment action steps"
        )
    if (
        budget.get("updates") != 200
        or budget.get("task_groups_per_update") != 4
        or budget.get("rollouts_per_group") != 8
    ):
        raise G2POStudyConfigError("the registered comparison fixes the 200x4x8 update budget")
    if budget.get("smoke_updates_before_formal") != 10:
        raise G2POStudyConfigError("formal execution requires a 10-update smoke run")
    gates = raw.get("gates")
    mechanism = gates.get("mechanism") if isinstance(gates, Mapping) else None
    if (
        not isinstance(mechanism, Mapping)
        or mechanism.get("min_restore_verified_audit_states") != 100
    ):
        raise G2POStudyConfigError(
            "verified-state mechanism gate requires at least 100 restore-verified audit states"
        )
    if mechanism.get("require_shared_verified_state") is not True:
        raise G2POStudyConfigError(
            "verified-state mechanism requires measurable shared safe states"
        )
    source = gates.get("source") if isinstance(gates, Mapping) else None
    length = gates.get("length") if isinstance(gates, Mapping) else None
    restore = gates.get("restore") if isinstance(gates, Mapping) else None
    state = gates.get("state") if isinstance(gates, Mapping) else None
    gpu = gates.get("gpu") if isinstance(gates, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("requirement") != "g2po_source_fixture_matches_pinned_source"
    ):
        raise G2POStudyConfigError("source gate must require the pinned G2PO credit fixture")
    if not isinstance(length, Mapping) or length.get("fixture_lengths") != [2, 3, 5]:
        raise G2POStudyConfigError("length gate must cover a mixed 2/3/5-step batch")
    if not isinstance(restore, Mapping) or set(restore.get("required_comparisons") or ()) != {
        "observation",
        "reward",
        "done",
        "won",
        "wrapper_state",
    }:
        raise G2POStudyConfigError(
            "restore gate must compare observation, reward, done, won, wrapper state"
        )
    if not isinstance(state, Mapping) or state.get("on_missing") != "fail_closed":
        raise G2POStudyConfigError("verified-state identity must fail closed")
    if state.get("restore_probe_policy") != "duplicate_candidate_states_only":
        raise G2POStudyConfigError(
            "verified-state restore probes must be limited to duplicate candidate states"
        )
    if not isinstance(gpu, Mapping) or gpu.get("smoke_updates") != 10:
        raise G2POStudyConfigError("GPU gate must require the 10-update smoke evidence")
    remote = raw.get("remote_execution")
    required_return = {
        "source_lock",
        "config",
        "trace",
        "state_snapshot_manifest",
        "restore_probe_receipts",
        "checkpoint_manifest",
        "result",
    }
    if not isinstance(remote, Mapping) or remote.get("workspace_persistent") is not False:
        raise G2POStudyConfigError(
            "the remote execution contract must acknowledge its ephemeral workspace"
        )
    if set(remote.get("required_local_return_artifacts") or ()) != required_return:
        raise G2POStudyConfigError(
            "remote execution must return all source, trace, state, checkpoint, and result artifacts"
        )


def _validate_registered_manifest_pair(raw: Mapping[str, Any], *, config_path: Path) -> None:
    """Make the declared task population part of config admission, not a label."""

    environment = raw.get("environment")
    if not isinstance(environment, Mapping):  # guarded above; keeps this helper total.
        raise G2POStudyConfigError("ALFWorld environment contract is required")
    workspace_root = config_path.resolve().parents[2]
    try:
        train_path = _resolve_workspace_relative(
            workspace_root, str(environment["train_manifest"]), field="train_manifest"
        )
        eval_path = _resolve_workspace_relative(
            workspace_root, str(environment["eval_manifest"]), field="eval_manifest"
        )
        load_and_validate_g2po_alfworld_manifest_pair(train_path, eval_path)
    except (G2POManifestError, KeyError, ValueError) as error:
        raise G2POStudyConfigError(
            f"registered ALFWorld task manifests are invalid: {error}"
        ) from error


def _resolve_workspace_relative(workspace_root: Path, value: str, *, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise G2POStudyConfigError(f"{field} must be a workspace-relative path")
    resolved = (workspace_root / path).resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError as error:
        raise G2POStudyConfigError(f"{field} escapes the workspace") from error
    return resolved
