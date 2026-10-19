"""Fail-closed aggregation for the frozen formal ALFWorld actor-control cohort."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_actor_control import (
    FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_CONTENT_SHA256,
    FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_FILE_SHA256,
    AlfWorldActorControlError,
    build_alfworld_actor_control_population_summary,
    build_alfworld_live_menu_constraint_population_summary,
    validate_frozen_alfworld_actor_control_population_manifest,
)
from lh_platform.tracing.alfworld_controlled_fault_injection import (
    AlfWorldControlledFaultInjectionPlanV1,
    ControlledFaultInjectionError,
)
from lh_platform.training.g2po.actor_protocol import build_alfworld_actor_protocol
from lh_platform.training.serialized_response_trie_policy import (
    serialized_response_trie_component,
)

FROZEN_ACTOR_CONTROL_FORMAL_EXECUTION_REVISION = "ff510f0a7fbf1178980f82141ca75b616be1f693"
FROZEN_ACTOR_CONTROL_BOOTSTRAP_SAMPLES = 100_000
FROZEN_ACTOR_CONTROL_BOOTSTRAP_SEED = 26_072_999
FROZEN_ACTOR_CONTROL_TASK_MANIFEST_FILE_SHA256 = (
    "93db7840fd8a0f07e89981068c107c1191c1a1df379132080fbcc7f8a26948a5"
)
ACTOR_CONTROL_POPULATION_AGGREGATION_SCHEMA = "lh_alfworld_actor_control_population_aggregation/v1"
LIVE_MENU_CONSTRAINT_POPULATION_AGGREGATION_SCHEMA = (
    "lh_alfworld_live_menu_constraint_population_aggregation/v1"
)
FROZEN_LIVE_MENU_PREREGISTRATION_FILE_SHA256 = (
    "6d3cec14e9900ab17684d968a3bef67dc92c228da7d00ac918081d775f15e0c1"
)
FROZEN_LIVE_MENU_PREREGISTRATION_CONTENT_SHA256 = (
    "sha256:9a46e68a7dca33fd572d14e8fba67aad64727f5a91b28376380e944d9b7b9519"
)
FROZEN_LIVE_MENU_FORMAL_EXECUTION_REVISION = "17d8538ebb2ca1362ca3d7c871de2db9f45aedd5"


def _validate_frozen_live_menu_preregistration(
    payload: Mapping[str, Any],
    *,
    file_sha256: str,
) -> None:
    canonical = dict(payload)
    claimed_content_sha256 = canonical.pop("content_sha256", None)
    if (
        file_sha256 != FROZEN_LIVE_MENU_PREREGISTRATION_FILE_SHA256
        or claimed_content_sha256 != "sha256:" + canonical_sha256(canonical)
        or claimed_content_sha256 != FROZEN_LIVE_MENU_PREREGISTRATION_CONTENT_SHA256
        or payload.get("schema") != "lh_alfworld_live_menu_constraint_preregistration/v1"
        or payload.get("status") != "frozen_before_gpu_execution"
        or payload.get("formal_execution_revision") != FROZEN_LIVE_MENU_FORMAL_EXECUTION_REVISION
        or payload.get("treatment_component") != serialized_response_trie_component()
        or payload.get("scientific_results_present") is not False
    ):
        raise AlfWorldActorControlError(
            "live-menu preregistration is not the frozen pre-outcome artifact"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlfWorldActorControlError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise AlfWorldActorControlError(f"{label} must be a JSON object")
    return value


def _load_sha256sums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AlfWorldActorControlError(f"cannot read SHA256SUMS: {error}") from error
    entries: dict[str, str] = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name
            or name in entries
        ):
            raise AlfWorldActorControlError("formal actor-control SHA256SUMS is invalid")
        entries[name] = digest
    return entries


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlfWorldActorControlError(f"{label} is missing")
    return value


def _require_prefixed_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise AlfWorldActorControlError(f"{label} is not a prefixed SHA-256")
    return value


def _require_raw_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AlfWorldActorControlError(f"{label} is not a raw SHA-256")
    return value


def _validate_frozen_injection_plan(
    *,
    payload: Mapping[str, Any],
    case: Mapping[str, Any],
    max_episode_steps: int,
) -> AlfWorldControlledFaultInjectionPlanV1:
    try:
        plan = AlfWorldControlledFaultInjectionPlanV1.from_dict(payload)
    except ControlledFaultInjectionError as error:
        raise AlfWorldActorControlError(f"frozen injection plan is invalid: {error}") from error
    arms = _require_mapping(case.get("arms"), label="frozen actor-control arms")
    baseline = _require_mapping(arms.get("B_actor"), label="frozen B_actor arm")
    treatment = _require_mapping(arms.get("T_actor"), label="frozen T_actor arm")
    suffix = _require_mapping(arms.get("T_suffix"), label="frozen T_suffix arm")
    expected_plan_sha256 = _require_prefixed_sha256(
        case.get("injection_plan_sha256"),
        label="frozen injection plan SHA-256",
    )
    if (
        "sha256:" + canonical_sha256(payload) != expected_plan_sha256
        or plan.sha256 != expected_plan_sha256
        or plan.task_id != case.get("task_id")
        or plan.traj_uid != case.get("traj_uid")
        or plan.source_trajectory_sha256 != case.get("source_trajectory_sha256")
        or plan.task_manifest_lock_sha256 != FROZEN_ACTOR_CONTROL_TASK_MANIFEST_FILE_SHA256
        or plan.injection.step_idx != case.get("gold_step_idx")
        or plan.injection.source_action != treatment.get("intervention_action")
        or plan.injection.source_action != suffix.get("intervention_action")
        or plan.injection.injected_action != baseline.get("intervention_action")
        or plan.suffix_controller_sha256 != suffix.get("suffix_controller_sha256")
        or plan.post_injection_source_action_count != case.get("source_suffix_action_count")
        or max_episode_steps - plan.injection.step_idx - 1
        != case.get("remaining_interaction_budget")
    ):
        raise AlfWorldActorControlError("frozen injection plan semantic binding changed")
    return plan


def _validate_actor_control_arm(
    *,
    raw_arm: object,
    arm_name: str,
    applied_action: str,
    sampling_seed_base: int | None,
    plan: AlfWorldControlledFaultInjectionPlanV1,
    remaining_interaction_budget: int,
    policy_checkpoint_sha256: str,
) -> None:
    arm = _require_mapping(raw_arm, label=f"{arm_name} receipt")
    target_step_idx = plan.injection.step_idx
    target_step = plan.steps[target_step_idx]
    executed_actions = arm.get("executed_actions")
    actor_steps = arm.get("actor_steps")
    menu = arm.get("pre_intervention_admissible_actions")
    physical_call_counts = arm.get("physical_call_counts")
    if (
        arm.get("arm") != arm_name
        or arm.get("applied_action") != applied_action
        or arm.get("sampling_seed_base") != sampling_seed_base
        or arm.get("pre_state_identity_verified") is not True
        or arm.get("pre_intervention_state_sha256")
        != plan.injection.source_pre_state_snapshot_sha256
        or not isinstance(menu, list)
        or tuple(menu) != target_step.admissible_actions
        or not isinstance(executed_actions, list)
        or not all(isinstance(action, str) for action in executed_actions)
        or not isinstance(actor_steps, list)
        or not isinstance(arm.get("success"), bool)
        or not isinstance(physical_call_counts, Mapping)
    ):
        raise AlfWorldActorControlError(f"{arm_name} intervention receipt binding changed")
    _require_raw_sha256(
        arm.get("intervention_restore_receipt_sha256"),
        label=f"{arm_name} restore receipt",
    )
    _require_raw_sha256(
        arm.get("final_state_snapshot_sha256"),
        label=f"{arm_name} final state",
    )
    expected_prefix = tuple(step.source_action for step in plan.steps[:target_step_idx]) + (
        applied_action,
    )
    if tuple(executed_actions[: len(expected_prefix)]) != expected_prefix:
        raise AlfWorldActorControlError(f"{arm_name} prefix/intervention actions changed")
    executed_action_count = len(executed_actions)
    if (
        physical_call_counts.get("logical_semantic_action_calls") != executed_action_count
        or physical_call_counts.get("prefix_replay_environment_step_calls") != target_step_idx
        or physical_call_counts.get("state_restore_calls") != 1
        or physical_call_counts.get("worker_environment_step_calls") != executed_action_count + 1
        or physical_call_counts.get("unverified_worker_step_calls") != executed_action_count - 1
        or physical_call_counts.get("environment_step_calls")
        != executed_action_count + target_step_idx + 1
    ):
        raise AlfWorldActorControlError(f"{arm_name} physical call evidence changed")
    if arm_name == "T_suffix":
        if (
            actor_steps
            or arm.get("success") is not True
            or arm.get("terminal_reason") != "environment_won"
            or tuple(executed_actions) != tuple(step.source_action for step in plan.steps)
        ):
            raise AlfWorldActorControlError("T_suffix is not the recorded source controller")
        return
    if (
        sampling_seed_base is None
        or not actor_steps
        or len(actor_steps) > remaining_interaction_budget
    ):
        raise AlfWorldActorControlError(f"{arm_name} paired actor budget changed")
    if len(executed_actions) != len(expected_prefix) + len(actor_steps):
        raise AlfWorldActorControlError(f"{arm_name} actor action evidence is incomplete")
    terminal_reason = arm.get("terminal_reason")
    if terminal_reason == "environment_won":
        if arm.get("success") is not True:
            raise AlfWorldActorControlError(f"{arm_name} success/terminal evidence changed")
    elif terminal_reason == "budget_exhausted":
        if arm.get("success") is not False or len(actor_steps) != remaining_interaction_budget:
            raise AlfWorldActorControlError(f"{arm_name} budget exhaustion evidence changed")
    else:
        raise AlfWorldActorControlError(f"{arm_name} terminal evidence is unsupported")
    for ordinal, raw_step in enumerate(actor_steps):
        step = _require_mapping(raw_step, label=f"{arm_name} actor-visible step")
        action_index = len(expected_prefix) + ordinal
        token_ids = step.get("action_token_ids")
        token_log_probs = step.get("action_token_log_probs")
        if (
            step.get("arm_id") != f"{plan.injection.injection_id}:{arm_name}:{sampling_seed_base}"
            or step.get("step_idx") != target_step_idx + 1 + ordinal
            or step.get("sampling_seed") != sampling_seed_base + ordinal
            or step.get("remaining_budget") != remaining_interaction_budget - ordinal
            or step.get("sampling_temperature") != plan.sampling_temperature
            or step.get("policy_checkpoint_sha256") != policy_checkpoint_sha256
            or step.get("environment_action") != executed_actions[action_index]
            or step.get("format_valid") is not True
            or not isinstance(step.get("action_in_live_admissible_menu"), bool)
            or not isinstance(step.get("model_response"), str)
            or not step.get("model_response")
            or not isinstance(token_ids, list)
            or not token_ids
            or not isinstance(token_log_probs, list)
            or len(token_ids) != len(token_log_probs)
        ):
            raise AlfWorldActorControlError(f"{arm_name} actor-visible evidence changed")
        _require_prefixed_sha256(
            step.get("actor_visible_input_sha256"),
            label=f"{arm_name} actor-visible input",
        )


def _validate_actor_control_case_semantics(
    *,
    receipt: Mapping[str, Any],
    plan: AlfWorldControlledFaultInjectionPlanV1,
    sampling_seeds: tuple[int, ...],
    remaining_interaction_budget: int,
    policy_checkpoint_sha256: str,
) -> None:
    _validate_actor_control_arm(
        raw_arm=receipt.get("T_suffix"),
        arm_name="T_suffix",
        applied_action=plan.injection.source_action,
        sampling_seed_base=None,
        plan=plan,
        remaining_interaction_budget=remaining_interaction_budget,
        policy_checkpoint_sha256=policy_checkpoint_sha256,
    )
    raw_pairs = receipt.get("actor_pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(sampling_seeds):
        raise AlfWorldActorControlError("paired actor semantic coverage changed")
    for sampling_seed, raw_pair in zip(sampling_seeds, raw_pairs, strict=True):
        pair = _require_mapping(raw_pair, label="paired actor receipt")
        if pair.get("sampling_seed_base") != sampling_seed:
            raise AlfWorldActorControlError("paired actor seed binding changed")
        _validate_actor_control_arm(
            raw_arm=pair.get("B_actor"),
            arm_name="B_actor",
            applied_action=plan.injection.injected_action,
            sampling_seed_base=sampling_seed,
            plan=plan,
            remaining_interaction_budget=remaining_interaction_budget,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
        )
        _validate_actor_control_arm(
            raw_arm=pair.get("T_actor"),
            arm_name="T_actor",
            applied_action=plan.injection.source_action,
            sampling_seed_base=sampling_seed,
            plan=plan,
            remaining_interaction_budget=remaining_interaction_budget,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
        )


def _validate_live_menu_arm(
    *,
    raw_arm: object,
    arm_name: str,
    selection_mode: str,
    sampling_seed_base: int,
    plan: AlfWorldControlledFaultInjectionPlanV1,
    remaining_interaction_budget: int,
    policy_checkpoint_sha256: str,
) -> None:
    arm = _require_mapping(raw_arm, label=f"{arm_name} receipt")
    target_step_idx = plan.injection.step_idx
    target_step = plan.steps[target_step_idx]
    executed_actions = arm.get("executed_actions")
    actor_steps = arm.get("actor_steps")
    menu = arm.get("pre_intervention_admissible_actions")
    physical_call_counts = arm.get("physical_call_counts")
    protocol = build_alfworld_actor_protocol(
        protocol_id=plan.actor_protocol_id,
        history_length=1,
        show_admissible_actions=True,
    )
    if (
        arm.get("arm") != arm_name
        or arm.get("applied_action") != plan.injection.source_action
        or arm.get("sampling_seed_base") != sampling_seed_base
        or arm.get("pre_state_identity_verified") is not True
        or arm.get("pre_intervention_state_sha256")
        != plan.injection.source_pre_state_snapshot_sha256
        or not isinstance(menu, list)
        or tuple(menu) != target_step.admissible_actions
        or not isinstance(executed_actions, list)
        or not all(isinstance(action, str) for action in executed_actions)
        or not isinstance(actor_steps, list)
        or not actor_steps
        or len(actor_steps) > remaining_interaction_budget
        or not isinstance(arm.get("success"), bool)
        or not isinstance(physical_call_counts, Mapping)
    ):
        raise AlfWorldActorControlError(f"{arm_name} live-menu arm binding changed")
    _require_raw_sha256(
        arm.get("intervention_restore_receipt_sha256"),
        label=f"{arm_name} restore receipt",
    )
    _require_raw_sha256(
        arm.get("final_state_snapshot_sha256"),
        label=f"{arm_name} final state",
    )
    expected_prefix = tuple(step.source_action for step in plan.steps[:target_step_idx]) + (
        plan.injection.source_action,
    )
    if tuple(executed_actions[: len(expected_prefix)]) != expected_prefix:
        raise AlfWorldActorControlError(f"{arm_name} prefix/intervention actions changed")
    if len(executed_actions) != len(expected_prefix) + len(actor_steps):
        raise AlfWorldActorControlError(f"{arm_name} actor action evidence is incomplete")
    executed_action_count = len(executed_actions)
    if (
        physical_call_counts.get("logical_semantic_action_calls") != executed_action_count
        or physical_call_counts.get("prefix_replay_environment_step_calls") != target_step_idx
        or physical_call_counts.get("state_restore_calls") != 1
        or physical_call_counts.get("worker_environment_step_calls") != executed_action_count + 1
        or physical_call_counts.get("unverified_worker_step_calls") != executed_action_count - 1
        or physical_call_counts.get("environment_step_calls")
        != executed_action_count + target_step_idx + 1
    ):
        raise AlfWorldActorControlError(f"{arm_name} physical call evidence changed")
    terminal_reason = arm.get("terminal_reason")
    if terminal_reason == "environment_won":
        if arm.get("success") is not True:
            raise AlfWorldActorControlError(f"{arm_name} success evidence changed")
    elif terminal_reason == "budget_exhausted":
        if arm.get("success") is not False or len(actor_steps) != remaining_interaction_budget:
            raise AlfWorldActorControlError(f"{arm_name} budget exhaustion evidence changed")
    else:
        raise AlfWorldActorControlError(f"{arm_name} terminal evidence is unsupported")

    for ordinal, raw_step in enumerate(actor_steps):
        step = _require_mapping(raw_step, label=f"{arm_name} actor-visible step")
        action_index = len(expected_prefix) + ordinal
        token_ids = step.get("action_token_ids")
        token_log_probs = step.get("action_token_log_probs")
        allowed_token_ids = step.get("allowed_action_token_ids")
        candidate_count = step.get("constrained_response_candidate_count")
        live_actions = step.get("live_admissible_actions")
        response_candidates = step.get("serialized_response_candidates")
        format_valid = step.get("format_valid")
        if (
            step.get("arm_id") != f"{plan.injection.injection_id}:{arm_name}:{sampling_seed_base}"
            or step.get("step_idx") != target_step_idx + 1 + ordinal
            or step.get("sampling_seed") != sampling_seed_base + ordinal
            or step.get("remaining_budget") != remaining_interaction_budget - ordinal
            or step.get("sampling_temperature") != plan.sampling_temperature
            or step.get("policy_checkpoint_sha256") != policy_checkpoint_sha256
            or step.get("action_selection_mode") != selection_mode
            or step.get("environment_action") != executed_actions[action_index]
            or not isinstance(step.get("action_in_live_admissible_menu"), bool)
            or not isinstance(format_valid, bool)
            or not isinstance(step.get("model_response"), str)
            or not step.get("model_response")
            or not isinstance(token_ids, list)
            or not token_ids
            or not isinstance(token_log_probs, list)
            or len(token_ids) != len(token_log_probs)
            or not isinstance(allowed_token_ids, list)
            or not isinstance(live_actions, list)
            or not live_actions
            or not all(isinstance(action, str) and action for action in live_actions)
            or not isinstance(response_candidates, list)
        ):
            raise AlfWorldActorControlError(f"{arm_name} actor-visible evidence changed")
        _require_prefixed_sha256(
            step.get("actor_visible_input_sha256"),
            label=f"{arm_name} actor-visible input",
        )
        _require_prefixed_sha256(
            step.get("live_admissible_actions_sha256"),
            label=f"{arm_name} live admissible actions",
        )
        _require_prefixed_sha256(
            step.get("serialized_response_candidates_sha256"),
            label=f"{arm_name} serialized response candidates",
        )
        if (
            step.get("live_admissible_actions_sha256") != "sha256:" + canonical_sha256(live_actions)
            or step.get("serialized_response_candidates_sha256")
            != "sha256:" + canonical_sha256(response_candidates)
            or step.get("action_in_live_admissible_menu")
            is not (step.get("environment_action") in live_actions)
        ):
            raise AlfWorldActorControlError(
                f"{arm_name} live-menu candidate hashes are not reproducible"
            )
        if selection_mode == "free_generation":
            if (
                candidate_count != 0
                or response_candidates
                or allowed_token_ids
                or step.get("finish_reason") == "constrained_response"
            ):
                raise AlfWorldActorControlError("T_raw is not free generation")
        else:
            expected_candidates = [
                protocol.response_contract.serialize_supervision(action).content
                for action in live_actions
            ]
            if (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count != len(live_actions)
                or response_candidates != expected_candidates
                or len(allowed_token_ids) != len(token_ids)
                or any(
                    not isinstance(values, list) or not values or selected not in values
                    for selected, values in zip(token_ids, allowed_token_ids, strict=True)
                )
                or step.get("finish_reason") != "constrained_response"
                or step.get("action_in_live_admissible_menu") is not True
                or step.get("environment_action") not in live_actions
                or format_valid is not True
            ):
                raise AlfWorldActorControlError(
                    "T_menu is not a complete valid live-menu trie sample"
                )


def _validate_live_menu_case_semantics(
    *,
    receipt: Mapping[str, Any],
    plan: AlfWorldControlledFaultInjectionPlanV1,
    sampling_seeds: tuple[int, ...],
    remaining_interaction_budget: int,
    policy_checkpoint_sha256: str,
) -> None:
    _validate_actor_control_arm(
        raw_arm=receipt.get("T_suffix"),
        arm_name="T_suffix",
        applied_action=plan.injection.source_action,
        sampling_seed_base=None,
        plan=plan,
        remaining_interaction_budget=remaining_interaction_budget,
        policy_checkpoint_sha256=policy_checkpoint_sha256,
    )
    raw_pairs = receipt.get("actor_pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(sampling_seeds):
        raise AlfWorldActorControlError("paired live-menu semantic coverage changed")
    for seed, raw_pair in zip(sampling_seeds, raw_pairs, strict=True):
        pair = _require_mapping(raw_pair, label="paired live-menu receipt")
        if pair.get("sampling_seed_base") != seed:
            raise AlfWorldActorControlError("paired live-menu seed binding changed")
        _validate_live_menu_arm(
            raw_arm=pair.get("T_raw"),
            arm_name="T_raw",
            selection_mode="free_generation",
            sampling_seed_base=seed,
            plan=plan,
            remaining_interaction_budget=remaining_interaction_budget,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
        )
        _validate_live_menu_arm(
            raw_arm=pair.get("T_menu"),
            arm_name="T_menu",
            selection_mode="serialized_live_menu_trie",
            sampling_seed_base=seed,
            plan=plan,
            remaining_interaction_budget=remaining_interaction_budget,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
        )
        raw_arm = _require_mapping(pair.get("T_raw"), label="T_raw")
        menu_arm = _require_mapping(pair.get("T_menu"), label="T_menu")
        raw_steps = raw_arm.get("actor_steps")
        menu_steps = menu_arm.get("actor_steps")
        raw_success = raw_arm.get("success")
        menu_success = menu_arm.get("success")
        expected = {
            "menu_rescue": raw_success is False and menu_success is True,
            "menu_harm": raw_success is True and menu_success is False,
            "both_success": raw_success is True and menu_success is True,
            "both_failed": raw_success is False and menu_success is False,
        }
        if (
            not isinstance(raw_steps, list)
            or not isinstance(menu_steps, list)
            or pair.get("first_continuation_input_identical") is not True
            or raw_steps[0].get("actor_visible_input_sha256")
            != menu_steps[0].get("actor_visible_input_sha256")
            or pair.get("first_action_diverged")
            is not (
                raw_arm["executed_actions"][plan.injection.step_idx + 1]
                != menu_arm["executed_actions"][plan.injection.step_idx + 1]
            )
            or any(pair.get(name) is not value for name, value in expected.items())
        ):
            raise AlfWorldActorControlError("paired live-menu causal binding changed")


def _git_revision(repo_root: Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise AlfWorldActorControlError("actor-control aggregation worktree is dirty")
    return revision


def run_frozen_alfworld_actor_control_population_aggregation(
    *,
    pilot_manifest_path: Path,
    case_root: Path,
    output_path: Path,
    repo_root: Path,
    expected_aggregation_revision: str,
) -> dict[str, Any]:
    """Verify all ten formal case receipts and publish task-clustered metrics."""

    aggregation_revision = _git_revision(repo_root.resolve())
    if aggregation_revision != expected_aggregation_revision:
        raise AlfWorldActorControlError("actor-control aggregation revision changed")
    pilot_path = pilot_manifest_path.resolve()
    pilot_file_sha256 = _file_sha256(pilot_path)
    pilot = _load_object(pilot_path, label="actor-control pilot manifest")
    validate_frozen_alfworld_actor_control_population_manifest(
        pilot,
        file_sha256=pilot_file_sha256,
    )
    raw_cases = pilot.get("cases")
    if not isinstance(raw_cases, list):
        raise AlfWorldActorControlError("frozen actor-control cases are missing")
    cases = tuple(case for case in raw_cases if isinstance(case, Mapping))
    if len(cases) != len(raw_cases):
        raise AlfWorldActorControlError("frozen actor-control case is invalid")
    pilot_inputs = pilot.get("input_file_sha256")
    actor_identity = pilot.get("actor_identity")
    raw_sampling_seeds = pilot.get("sampling_seeds")
    max_episode_steps = pilot.get("max_episode_steps")
    if (
        not isinstance(pilot_inputs, Mapping)
        or not isinstance(actor_identity, Mapping)
        or not isinstance(raw_sampling_seeds, list)
        or len(raw_sampling_seeds) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_sampling_seeds)
        or isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
    ):
        raise AlfWorldActorControlError("frozen actor-control input identity is missing")
    sampling_seeds = tuple(raw_sampling_seeds)
    policy_checkpoint_sha256 = _require_raw_sha256(
        actor_identity.get("parameter_checkpoint_sha256"),
        label="frozen policy checkpoint",
    )

    root = case_root.resolve()
    expected_directories = {f"case-{index:04d}" for index in range(len(cases))}
    actual_directories = {
        path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("case-")
    }
    if actual_directories != expected_directories:
        raise AlfWorldActorControlError("formal actor-control case directories changed")

    case_receipts: dict[str, Mapping[str, Any]] = {}
    input_files: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        candidate_id = case.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise AlfWorldActorControlError("formal actor-control candidate ID is invalid")
        directory = root / f"case-{index:04d}"
        run_path = directory / "run_receipt.json"
        case_path = directory / "actor_control_case.json"
        injection_path = directory / "injection_plan.json"
        sums_path = directory / "SHA256SUMS"
        run_file_sha256 = _file_sha256(run_path)
        case_file_sha256 = _file_sha256(case_path)
        injection_file_sha256 = _file_sha256(injection_path)
        sums_file_sha256 = _file_sha256(sums_path)
        run_receipt = _load_object(run_path, label=f"run receipt {candidate_id}")
        case_receipt = _load_object(case_path, label=f"case receipt {candidate_id}")
        injection_plan_payload = _load_object(
            injection_path,
            label=f"injection plan {candidate_id}",
        )
        expected_sums = {
            "actor_control_case.json": case_file_sha256,
            "injection_plan.json": injection_file_sha256,
            "run_receipt.json": run_file_sha256,
        }
        if _load_sha256sums(sums_path) != expected_sums:
            raise AlfWorldActorControlError(
                f"formal actor-control artifact integrity changed for {candidate_id}"
            )
        injection_plan = _validate_frozen_injection_plan(
            payload=injection_plan_payload,
            case=case,
            max_episode_steps=max_episode_steps,
        )
        if (
            run_receipt.get("schema") != "lh_alfworld_actor_control_population_case_run/v1"
            or run_receipt.get("status") != "completed"
            or run_receipt.get("scientific_result") is not False
            or run_receipt.get("execution_role") != "formal_population_case"
            or run_receipt.get("git_head") != FROZEN_ACTOR_CONTROL_FORMAL_EXECUTION_REVISION
            or run_receipt.get("candidate_id") != candidate_id
            or run_receipt.get("pilot_manifest_file_sha256")
            != FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_FILE_SHA256
            or run_receipt.get("pilot_manifest_content_sha256")
            != FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_CONTENT_SHA256
            or run_receipt.get("source_trajectory_file_sha256") != case.get("source_file_sha256")
            or run_receipt.get("source_trajectory_path") != case.get("source_path")
            or run_receipt.get("causal_gold_file_sha256") != case.get("causal_gold_file_sha256")
            or run_receipt.get("causal_gold_path") != case.get("causal_gold_path")
            or run_receipt.get("task_manifest_file_sha256")
            != FROZEN_ACTOR_CONTROL_TASK_MANIFEST_FILE_SHA256
            or run_receipt.get("source_collection_result_file_sha256")
            != pilot_inputs.get("source_collection_result_file_sha256")
            or run_receipt.get("system_cell_sha256") != actor_identity.get("system_cell_sha256")
            or run_receipt.get("policy_checkpoint_sha256")
            != actor_identity.get("parameter_checkpoint_sha256")
            or run_receipt.get("case_receipt_file_sha256") != case_file_sha256
            or run_receipt.get("case_receipt_content_sha256") != case_receipt.get("content_sha256")
            or run_receipt.get("T_suffix_success") is not True
            or run_receipt.get("paired_seed_count") != 3
            or case_receipt.get("traj_uid") != case.get("traj_uid")
            or case_receipt.get("injection_plan_sha256") != case.get("injection_plan_sha256")
            or case_receipt.get("remaining_interaction_budget")
            != case.get("remaining_interaction_budget")
        ):
            raise AlfWorldActorControlError(
                f"formal actor-control run binding changed for {candidate_id}"
            )
        _validate_actor_control_case_semantics(
            receipt=case_receipt,
            plan=injection_plan,
            sampling_seeds=sampling_seeds,
            remaining_interaction_budget=int(case["remaining_interaction_budget"]),
            policy_checkpoint_sha256=policy_checkpoint_sha256,
        )
        case_receipts[candidate_id] = case_receipt
        input_files.extend(
            (
                {
                    "candidate_id": candidate_id,
                    "kind": "run_receipt",
                    "path": str(run_path),
                    "file_sha256": run_file_sha256,
                },
                {
                    "candidate_id": candidate_id,
                    "kind": "actor_control_case",
                    "path": str(case_path),
                    "file_sha256": case_file_sha256,
                },
                {
                    "candidate_id": candidate_id,
                    "kind": "injection_plan",
                    "path": str(injection_path),
                    "file_sha256": injection_file_sha256,
                },
                {
                    "candidate_id": candidate_id,
                    "kind": "SHA256SUMS",
                    "path": str(sums_path),
                    "file_sha256": sums_file_sha256,
                },
            )
        )

    summary = build_alfworld_actor_control_population_summary(
        pilot_manifest=pilot,
        case_receipts_by_candidate=case_receipts,
        bootstrap_samples=FROZEN_ACTOR_CONTROL_BOOTSTRAP_SAMPLES,
        bootstrap_seed=FROZEN_ACTOR_CONTROL_BOOTSTRAP_SEED,
    )
    receipt: dict[str, Any] = {
        "schema": ACTOR_CONTROL_POPULATION_AGGREGATION_SCHEMA,
        "status": "completed",
        "scientific_result": True,
        "aggregation_git_head": aggregation_revision,
        "formal_case_execution_revision": (FROZEN_ACTOR_CONTROL_FORMAL_EXECUTION_REVISION),
        "pilot_manifest_path": str(pilot_path),
        "pilot_manifest_file_sha256": pilot_file_sha256,
        "pilot_manifest_content_sha256": pilot.get("content_sha256"),
        "input_files": input_files,
        "summary": summary,
    }
    receipt["content_sha256"] = "sha256:" + canonical_sha256(receipt)
    atomic_write_json(output_path.resolve(), receipt, overwrite=False)
    return receipt


def run_frozen_alfworld_live_menu_constraint_population_aggregation(
    *,
    preregistration_path: Path,
    pilot_manifest_path: Path,
    case_root: Path,
    output_path: Path,
    repo_root: Path,
    expected_aggregation_revision: str,
) -> dict[str, Any]:
    """Verify and aggregate the frozen ten-task live-menu comparison."""

    aggregation_revision = _git_revision(repo_root.resolve())
    if aggregation_revision != expected_aggregation_revision:
        raise AlfWorldActorControlError("live-menu aggregation revision changed")
    prereg_path = preregistration_path.resolve()
    prereg_file_sha256 = _file_sha256(prereg_path)
    preregistration = _load_object(
        prereg_path,
        label="live-menu preregistration",
    )
    _validate_frozen_live_menu_preregistration(
        preregistration,
        file_sha256=prereg_file_sha256,
    )
    pilot_path = pilot_manifest_path.resolve()
    pilot_file_sha256 = _file_sha256(pilot_path)
    pilot = _load_object(pilot_path, label="live-menu pilot manifest")
    validate_frozen_alfworld_actor_control_population_manifest(
        pilot,
        file_sha256=pilot_file_sha256,
    )
    raw_cases = pilot.get("cases")
    raw_seeds = pilot.get("sampling_seeds")
    actor_identity = pilot.get("actor_identity")
    pilot_inputs = pilot.get("input_file_sha256")
    max_episode_steps = pilot.get("max_episode_steps")
    if (
        not isinstance(raw_cases, list)
        or not isinstance(raw_seeds, list)
        or not isinstance(actor_identity, Mapping)
        or not isinstance(pilot_inputs, Mapping)
        or isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
    ):
        raise AlfWorldActorControlError("live-menu frozen inputs are incomplete")
    cases = tuple(_require_mapping(value, label="live-menu case") for value in raw_cases)
    sampling_seeds = tuple(raw_seeds)
    policy_checkpoint_sha256 = _require_raw_sha256(
        actor_identity.get("parameter_checkpoint_sha256"),
        label="live-menu policy checkpoint",
    )
    treatment_component = serialized_response_trie_component()
    prereg_parent = _require_mapping(
        preregistration.get("parent_population"),
        label="live-menu preregistered parent population",
    )
    if (
        prereg_parent.get("manifest_file_sha256") != pilot_file_sha256
        or prereg_parent.get("manifest_content_sha256") != pilot.get("content_sha256")
        or prereg_parent.get("case_count") != len(cases)
        or prereg_parent.get("task_count") != len(cases)
        or prereg_parent.get("sampling_seeds") != list(sampling_seeds)
        or prereg_parent.get("max_episode_steps") != max_episode_steps
    ):
        raise AlfWorldActorControlError("live-menu preregistered population binding changed")
    root = case_root.resolve()
    expected_directories = {f"case-{index:04d}" for index in range(len(cases))}
    actual_directories = {
        path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("case-")
    }
    if actual_directories != expected_directories:
        raise AlfWorldActorControlError("formal live-menu case directories changed")

    case_receipts: dict[str, Mapping[str, Any]] = {}
    input_files: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        candidate_id = case.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise AlfWorldActorControlError("formal live-menu candidate ID is invalid")
        directory = root / f"case-{index:04d}"
        run_path = directory / "run_receipt.json"
        case_path = directory / "live_menu_constraint_case.json"
        injection_path = directory / "injection_plan.json"
        sums_path = directory / "SHA256SUMS"
        files = {
            "run_receipt.json": _file_sha256(run_path),
            "live_menu_constraint_case.json": _file_sha256(case_path),
            "injection_plan.json": _file_sha256(injection_path),
        }
        if _load_sha256sums(sums_path) != files:
            raise AlfWorldActorControlError(
                f"formal live-menu artifact integrity changed for {candidate_id}"
            )
        run_receipt = _load_object(run_path, label=f"live-menu run {candidate_id}")
        case_receipt = _load_object(case_path, label=f"live-menu case {candidate_id}")
        injection_payload = _load_object(
            injection_path,
            label=f"live-menu injection plan {candidate_id}",
        )
        plan = _validate_frozen_injection_plan(
            payload=injection_payload,
            case=case,
            max_episode_steps=max_episode_steps,
        )
        if (
            run_receipt.get("schema") != "lh_alfworld_live_menu_constraint_population_case_run/v1"
            or run_receipt.get("status") != "completed"
            or run_receipt.get("scientific_result") is not False
            or run_receipt.get("execution_role") != "formal_population_case"
            or run_receipt.get("experiment_mode") != "live_menu_constraint"
            or run_receipt.get("treatment_component") != treatment_component
            or run_receipt.get("git_head") != FROZEN_LIVE_MENU_FORMAL_EXECUTION_REVISION
            or run_receipt.get("candidate_id") != candidate_id
            or run_receipt.get("pilot_manifest_file_sha256")
            != FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_FILE_SHA256
            or run_receipt.get("pilot_manifest_content_sha256")
            != FROZEN_ACTOR_CONTROL_POPULATION_MANIFEST_CONTENT_SHA256
            or run_receipt.get("source_trajectory_file_sha256") != case.get("source_file_sha256")
            or run_receipt.get("causal_gold_file_sha256") != case.get("causal_gold_file_sha256")
            or run_receipt.get("task_manifest_file_sha256")
            != FROZEN_ACTOR_CONTROL_TASK_MANIFEST_FILE_SHA256
            or run_receipt.get("source_collection_result_file_sha256")
            != pilot_inputs.get("source_collection_result_file_sha256")
            or run_receipt.get("system_cell_sha256") != actor_identity.get("system_cell_sha256")
            or run_receipt.get("policy_checkpoint_sha256") != policy_checkpoint_sha256
            or run_receipt.get("case_receipt_file_sha256")
            != files["live_menu_constraint_case.json"]
            or run_receipt.get("case_receipt_content_sha256") != case_receipt.get("content_sha256")
            or run_receipt.get("T_suffix_success") is not True
            or run_receipt.get("paired_seed_count") != len(sampling_seeds)
            or case_receipt.get("schema") != "lh_alfworld_live_menu_constraint_case/v1"
            or case_receipt.get("task_id") != case.get("task_id")
            or case_receipt.get("traj_uid") != case.get("traj_uid")
            or case_receipt.get("injection_plan_sha256") != case.get("injection_plan_sha256")
            or case_receipt.get("gold_step_idx") != case.get("gold_step_idx")
            or case_receipt.get("max_episode_steps") != max_episode_steps
            or case_receipt.get("paired_sampling_seeds") != list(sampling_seeds)
            or case_receipt.get("treatment_component_sha256")
            != treatment_component["component_sha256"]
            or case_receipt.get("claim_scope")
            != (
                "effect_of_response_contract_aligned_live_menu_constraint_after_"
                "known_correct_historical_action"
            )
            or case_receipt.get("out_of_menu_mediation_tested") is not False
            or case_receipt.get("tracer_accuracy_tested") is not False
        ):
            raise AlfWorldActorControlError(
                f"formal live-menu run binding changed for {candidate_id}"
            )
        canonical_case = dict(case_receipt)
        claimed_content_sha256 = canonical_case.pop("content_sha256", None)
        if claimed_content_sha256 != "sha256:" + canonical_sha256(canonical_case):
            raise AlfWorldActorControlError(
                f"formal live-menu case content hash changed for {candidate_id}"
            )
        _validate_live_menu_case_semantics(
            receipt=case_receipt,
            plan=plan,
            sampling_seeds=sampling_seeds,
            remaining_interaction_budget=int(case["remaining_interaction_budget"]),
            policy_checkpoint_sha256=policy_checkpoint_sha256,
        )
        case_receipts[candidate_id] = case_receipt
        for name, digest in (*files.items(), ("SHA256SUMS", _file_sha256(sums_path))):
            input_files.append(
                {
                    "candidate_id": candidate_id,
                    "kind": name,
                    "path": str(directory / name),
                    "file_sha256": digest,
                }
            )

    summary = build_alfworld_live_menu_constraint_population_summary(
        pilot_manifest=pilot,
        case_receipts_by_candidate=case_receipts,
        bootstrap_samples=FROZEN_ACTOR_CONTROL_BOOTSTRAP_SAMPLES,
        bootstrap_seed=FROZEN_ACTOR_CONTROL_BOOTSTRAP_SEED,
    )
    receipt: dict[str, Any] = {
        "schema": LIVE_MENU_CONSTRAINT_POPULATION_AGGREGATION_SCHEMA,
        "status": "completed",
        "scientific_result": True,
        "aggregation_git_head": aggregation_revision,
        "formal_case_execution_revision": FROZEN_LIVE_MENU_FORMAL_EXECUTION_REVISION,
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_content_sha256": preregistration.get("content_sha256"),
        "pilot_manifest_path": str(pilot_path),
        "pilot_manifest_file_sha256": pilot_file_sha256,
        "pilot_manifest_content_sha256": pilot.get("content_sha256"),
        "input_files": input_files,
        "summary": summary,
    }
    receipt["content_sha256"] = "sha256:" + canonical_sha256(receipt)
    atomic_write_json(output_path.resolve(), receipt, overwrite=False)
    return receipt


__all__ = [
    "ACTOR_CONTROL_POPULATION_AGGREGATION_SCHEMA",
    "FROZEN_ACTOR_CONTROL_FORMAL_EXECUTION_REVISION",
    "FROZEN_LIVE_MENU_FORMAL_EXECUTION_REVISION",
    "FROZEN_LIVE_MENU_PREREGISTRATION_CONTENT_SHA256",
    "FROZEN_LIVE_MENU_PREREGISTRATION_FILE_SHA256",
    "LIVE_MENU_CONSTRAINT_POPULATION_AGGREGATION_SCHEMA",
    "run_frozen_alfworld_actor_control_population_aggregation",
    "run_frozen_alfworld_live_menu_constraint_population_aggregation",
]
