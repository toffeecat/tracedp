"""Action-only counterfactual search over failed ALFWorld trajectories.

The module separates candidate planning from environment execution.  A search
can discover a sufficient repair, but it deliberately cannot promote that
repair to a root cause without matched-sham and reverse-injection evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from lh_platform.training.agentic_rl_runtime import PolicyActionRequest, TrainablePolicy
from lh_platform.training.alfworld_agentic_suite import AlfWorldBranch
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.g2po.actor_protocol import (
    G2POActorHistoryStep,
    build_alfworld_actor_protocol,
)
from lh_platform.training.model_io import ConversationIR

ALFWORLD_ACTION_COUNTERFACTUAL_PLAN_SCHEMA = "lh_alfworld_action_counterfactual_plan/v1"
ALFWORLD_ACTION_COUNTERFACTUAL_SEARCH_SCHEMA = "lh_alfworld_action_counterfactual_search/v1"
ALFWORLD_STAGED_ACTION_COUNTERFACTUAL_SEARCH_SCHEMA = (
    "lh_alfworld_staged_action_counterfactual_search/v1"
)
ALFWORLD_ROOT_CAUSE_EVIDENCE_SCHEMA = "lh_intervention_verified_root_cause/v1"
ALFWORLD_CAUSAL_VALIDATION_SCHEMA = "lh_alfworld_causal_validation/v1"
ALFWORLD_SUFFICIENT_CAUSE_EVIDENCE_SCHEMA = "lh_intervention_verified_sufficient_cause/v1"
ALFWORLD_THREE_ARM_CAUSAL_VALIDATION_SCHEMA = "lh_alfworld_three_arm_causal_validation/v2"
ALFWORLD_SINGLE_ACTION_ROOT_CAUSE_ADMISSION_SCHEMA = (
    "lh_alfworld_single_action_root_cause_admission/v1"
)
ALFWORLD_SOURCE_SEED_SCHEDULE = "base_plus_trajectory_ordinal_times_horizon_plus_step"

ContinuationMode = Literal["open_loop", "closed_loop"]
CausalArm = Literal["intervention", "matched_sham", "reverse_injection"]


class AlfWorldCounterfactualError(RuntimeError):
    """Raised when a counterfactual arm is not scientifically comparable."""


def _matched_sham_placebo_action(
    admissible_actions: tuple[str, ...],
    *,
    source_action: str,
    replacement_action: str,
) -> str | None:
    """Return the frozen-menu placebo action for the matched sham arm.

    The sham must not replay the fault action (that is the baseline trajectory)
    or the proposed repair (that is the treatment).  It replaces the candidate
    step with the first admissible action that is neither, exercising the same
    state and the same continuation protocol.  Menu order is frozen in the plan,
    so the choice is reproducible.  Returns ``None`` when the frozen menu cannot
    supply a placebo, in which case the sham arm fails closed instead of
    degenerating into a baseline replay.
    """

    for action in admissible_actions:
        if action != source_action and action != replacement_action:
            return action
    return None


@dataclass(frozen=True)
class AlfWorldSourceStep:
    step_idx: int
    source_action: str
    admissible_actions: tuple[str, ...]
    source_action_admissible: bool
    state_snapshot_sha256: str
    state_snapshot_artifact_path: str
    post_state_snapshot_sha256: str
    post_state_snapshot_artifact_path: str
    observation: str
    next_observation: str
    sampling_seed: int


@dataclass(frozen=True)
class AlfWorldSourceTrajectory:
    """Validated terminal ALFWorld trajectory before any failure-only operation."""

    records: tuple[TrainingTrajectoryRecord, ...]
    task_id: str
    traj_uid: str
    split: str
    environment_seed: int
    task_description: str
    actor_protocol_id: str
    model_id: str
    model_io_profile_sha256: str
    response_contract_sha256: str
    task_manifest_lock_sha256: str
    chat_template_sha256: str
    sampling_seed_base: int
    sampling_seed_schedule: str
    sampling_temperature: float
    success: bool
    terminal_reason: str
    source_trajectory_sha256: str
    steps: tuple[AlfWorldSourceStep, ...]


@dataclass(frozen=True)
class AlfWorldSingleActionRootCauseAdmission:
    """Source-only suitability decision for a single-action causal claim."""

    source_trajectory_sha256: str
    task_id: str
    traj_uid: str
    terminal_reason: str
    source_step_count: int
    observation_only_step_count: int
    unique_source_actions: tuple[str, ...]
    decision: Literal["admitted", "rejected"]
    blockers: tuple[str, ...]
    schema: str = ALFWORLD_SINGLE_ACTION_ROOT_CAUSE_ADMISSION_SCHEMA

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "task_id": self.task_id,
            "traj_uid": self.traj_uid,
            "terminal_reason": self.terminal_reason,
            "source_step_count": self.source_step_count,
            "observation_only_step_count": self.observation_only_step_count,
            "unique_source_actions": list(self.unique_source_actions),
            "decision": self.decision,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class SingleActionIntervention:
    intervention_id: str
    step_idx: int
    source_action: str
    replacement_action: str
    source_pre_state_snapshot_sha256: str


@dataclass(frozen=True)
class CausalArmBinding:
    plan_sha256: str
    source_trajectory_sha256: str
    intervention_id: str
    step_idx: int
    source_pre_state_snapshot_sha256: str
    source_action: str
    replacement_action: str
    continuation_mode: ContinuationMode
    controller_identity_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("plan_sha256", self.plan_sha256),
            ("source_trajectory_sha256", self.source_trajectory_sha256),
            (
                "source_pre_state_snapshot_sha256",
                self.source_pre_state_snapshot_sha256,
            ),
            ("controller_identity_sha256", self.controller_identity_sha256),
        ):
            _require_sha256(name, value)
        if not self.intervention_id or self.step_idx < 0:
            raise AlfWorldCounterfactualError("causal arm identity is invalid")
        if not self.source_action or not self.replacement_action:
            raise AlfWorldCounterfactualError("causal arm actions are required")
        if self.continuation_mode != "closed_loop":
            raise AlfWorldCounterfactualError("causal validation requires closed-loop continuation")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlfWorldSingleActionCounterfactualPlan:
    task_id: str
    traj_uid: str
    split: str
    environment_seed: int
    task_description: str
    actor_protocol_id: str
    model_id: str
    model_io_profile_sha256: str
    response_contract_sha256: str
    task_manifest_lock_sha256: str
    chat_template_sha256: str
    sampling_seed_base: int
    sampling_seed_schedule: str
    sampling_temperature: float
    source_terminal_reason: str
    source_trajectory_sha256: str
    steps: tuple[AlfWorldSourceStep, ...]
    candidates: tuple[SingleActionIntervention, ...]
    source_action_outside_admissible_steps: tuple[int, ...]
    schema: str = ALFWORLD_ACTION_COUNTERFACTUAL_PLAN_SCHEMA

    @property
    def candidate_count(self) -> int:
        """Number of single-action replacements, excluding the baseline arm."""

        return len(self.candidates)

    @property
    def planned_replay_arm_count(self) -> int:
        return 1 + self.candidate_count

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "traj_uid": self.traj_uid,
            "split": self.split,
            "environment_seed": self.environment_seed,
            "task_description": self.task_description,
            "actor_protocol_id": self.actor_protocol_id,
            "model_id": self.model_id,
            "model_io_profile_sha256": self.model_io_profile_sha256,
            "response_contract_sha256": self.response_contract_sha256,
            "task_manifest_lock_sha256": self.task_manifest_lock_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "sampling_seed_base": self.sampling_seed_base,
            "sampling_seed_schedule": self.sampling_seed_schedule,
            "sampling_temperature": self.sampling_temperature,
            "source_terminal_reason": self.source_terminal_reason,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "steps": [asdict(step) for step in self.steps],
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "source_action_outside_admissible_steps": list(
                self.source_action_outside_admissible_steps
            ),
            "candidate_count": self.candidate_count,
            "planned_replay_arm_count": self.planned_replay_arm_count,
        }


@dataclass(frozen=True)
class ReplayArmOutcome:
    success: bool
    terminal_reason: str
    pre_state_identity_verified: bool
    replacement_admissible: bool
    suffix_diverged: bool
    executed_actions: tuple[str, ...]
    final_state_snapshot_sha256: str
    physical_call_counts: Mapping[str, Any]
    outcome_evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.terminal_reason:
            raise AlfWorldCounterfactualError("replay arm terminal_reason is required")
        if self.final_state_snapshot_sha256:
            _require_sha256("final_state_snapshot_sha256", self.final_state_snapshot_sha256)
        if self.suffix_diverged and self.success:
            raise AlfWorldCounterfactualError(
                "a suffix-diverged open-loop arm cannot be labeled successful"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["physical_call_counts"] = dict(self.physical_call_counts)
        payload["outcome_evidence"] = dict(self.outcome_evidence or {})
        return payload


class CounterfactualReplayBackend(Protocol):
    """One uniform replay seam for real ALFWorld and deterministic test doubles."""

    def replay(
        self,
        *,
        plan: AlfWorldSingleActionCounterfactualPlan,
        intervention: SingleActionIntervention | None,
        continuation_mode: ContinuationMode,
    ) -> ReplayArmOutcome: ...


class ClosedLoopContinuationPolicy(Protocol):
    """Select one new semantic action after an intervention."""

    def choose_action(
        self,
        *,
        arm_id: str,
        task_id: str,
        step_idx: int,
        task_description: str,
        current_observation: str,
        admissible_actions: tuple[str, ...],
        history: tuple[tuple[str, str], ...],
        remaining_budget: int,
        sampling_seed: int,
        sampling_temperature: float,
    ) -> tuple[str, Mapping[str, Any]]: ...


class FrozenAlfWorldActorContinuationPolicy:
    """Adapt the platform policy seam to closed-loop counterfactual continuation."""

    def __init__(
        self,
        *,
        policy: TrainablePolicy,
        actor_protocol_id: str,
        history_length: int,
        show_admissible_actions: bool,
        model_io_profile_id: str,
        action_selection_mode: str = "free_generation",
    ) -> None:
        if action_selection_mode not in {
            "free_generation",
            "serialized_live_menu_trie",
        }:
            raise AlfWorldCounterfactualError("unsupported ALFWorld actor action-selection mode")
        self._policy = policy
        self._protocol = build_alfworld_actor_protocol(
            protocol_id=actor_protocol_id,
            history_length=history_length,
            show_admissible_actions=show_admissible_actions,
        )
        self._model_io_profile_id = model_io_profile_id
        self._action_selection_mode = action_selection_mode

    def choose_action(
        self,
        *,
        arm_id: str,
        task_id: str,
        step_idx: int,
        task_description: str,
        current_observation: str,
        admissible_actions: tuple[str, ...],
        history: tuple[tuple[str, str], ...],
        remaining_budget: int,
        sampling_seed: int,
        sampling_temperature: float,
    ) -> tuple[str, Mapping[str, Any]]:
        actor_history = tuple(
            G2POActorHistoryStep(observation=observation, environment_action=action)
            for observation, action in history
        )
        prompt = self._protocol.render_prompt(
            task_description=task_description,
            current_observation=current_observation,
            admissible_actions=admissible_actions,
            history=actor_history,
        )
        constrained_response_candidates = (
            tuple(
                self._protocol.response_contract.serialize_supervision(action).content
                for action in admissible_actions
            )
            if self._action_selection_mode == "serialized_live_menu_trie"
            else ()
        )
        request = PolicyActionRequest(
            traj_uid=f"counterfactual:{task_id}:{arm_id}",
            step_idx=step_idx,
            conversation=ConversationIR.from_legacy_user_prompt(prompt),
            model_io_profile_id=self._model_io_profile_id,
            response_contract_id=self._protocol.response_contract.id,
            response_contract_sha256=self._protocol.response_contract.descriptor.sha256,
            admissible_actions=(),
            sampling_temperature=sampling_temperature,
            sampling_seed=sampling_seed,
            response_stop_texts=self._protocol.response_contract.stop_texts,
        )
        if constrained_response_candidates:
            sampler = getattr(self._policy, "sample_serialized_response", None)
            if not callable(sampler):
                raise AlfWorldCounterfactualError(
                    "policy does not implement serialized-response trie sampling"
                )
            sample = sampler(
                request,
                constrained_response_candidates=constrained_response_candidates,
                seed=sampling_seed,
            )
        else:
            samples = self._policy.sample_actions((request,), seed=sampling_seed)
            if len(samples) != 1:
                raise AlfWorldCounterfactualError(
                    "closed-loop policy did not return exactly one action sample"
                )
            sample = samples[0]
        if sample.policy_checkpoint_sha256 != self._policy.checkpoint_sha256:
            raise AlfWorldCounterfactualError("closed-loop policy checkpoint changed")
        projection = self._protocol.project_response(
            sample.action_text,
            admissible_actions=admissible_actions,
        )
        if constrained_response_candidates and (
            not projection.is_action_valid
            or projection.environment_action not in admissible_actions
        ):
            raise AlfWorldCounterfactualError(
                "serialized live-menu constraint returned an invalid response"
            )
        return projection.environment_action, {
            "arm_id": arm_id,
            "step_idx": step_idx,
            "remaining_budget": remaining_budget,
            "actor_visible_input_sha256": "sha256:"
            + _canonical_sha256({"rendered_prompt": prompt}),
            "sampling_seed": sampling_seed,
            "sampling_temperature": sampling_temperature,
            "policy_checkpoint_sha256": sample.policy_checkpoint_sha256,
            "action_selection_mode": self._action_selection_mode,
            "live_admissible_actions_sha256": "sha256:"
            + _canonical_sha256(list(admissible_actions)),
            "live_admissible_actions": list(admissible_actions),
            "constrained_response_candidate_count": len(constrained_response_candidates),
            "serialized_response_candidates_sha256": "sha256:"
            + _canonical_sha256(list(constrained_response_candidates)),
            "serialized_response_candidates": list(constrained_response_candidates),
            "model_response": sample.action_text,
            "environment_action": projection.environment_action,
            "format_valid": projection.is_action_valid,
            "format_invalid_reasons": list(projection.invalid_reasons),
            "action_token_ids": list(sample.action_token_ids),
            "action_token_log_probs": list(sample.old_action_token_log_probs),
            "allowed_action_token_ids": [
                list(values) for values in getattr(sample, "allowed_action_token_ids", ())
            ],
            "finish_reason": getattr(sample, "finish_reason", "unknown"),
        }


class AlfWorldBranchReplayBackend:
    """Execute counterfactual arms on fresh, fixed-game ALFWorld branches.

    Prefixes and suffixes use the branch's direct step when available.  The
    intervention step uses the verified public ``step`` seam, while the saved
    replay-recipe hash is checked immediately before intervention.
    """

    def __init__(
        self,
        *,
        branch_factory: Callable[[], AlfWorldBranch],
        closed_loop_policy: ClosedLoopContinuationPolicy | None = None,
    ) -> None:
        self._branch_factory = branch_factory
        self._closed_loop_policy = closed_loop_policy

    def replay(
        self,
        *,
        plan: AlfWorldSingleActionCounterfactualPlan,
        intervention: SingleActionIntervention | None,
        continuation_mode: ContinuationMode,
        causal_arm: CausalArm = "intervention",
    ) -> ReplayArmOutcome:
        if continuation_mode not in {"open_loop", "closed_loop"}:
            raise AlfWorldCounterfactualError("unsupported continuation mode")
        if continuation_mode == "closed_loop" and self._closed_loop_policy is None:
            raise AlfWorldCounterfactualError(
                "closed-loop replay requires an explicit continuation policy"
            )
        if causal_arm not in {"intervention", "matched_sham", "reverse_injection"}:
            raise AlfWorldCounterfactualError("unsupported causal arm")
        if causal_arm != "intervention" and (
            intervention is None or continuation_mode != "closed_loop"
        ):
            raise AlfWorldCounterfactualError(
                "causal control arms require one closed-loop intervention candidate"
            )
        if intervention is not None:
            if intervention.step_idx >= len(plan.steps):
                raise AlfWorldCounterfactualError("intervention step is outside the source trace")
            source_step = plan.steps[intervention.step_idx]
            if (
                intervention.source_action != source_step.source_action
                or intervention.source_pre_state_snapshot_sha256
                != source_step.state_snapshot_sha256
                or intervention.replacement_action not in source_step.admissible_actions
                or intervention.replacement_action == source_step.source_action
            ):
                raise AlfWorldCounterfactualError("intervention differs from the frozen plan")

        branch = self._branch_factory()
        executed: list[str] = []
        actor_history: list[tuple[str, str]] = []
        pre_state_verified = True
        replacement_admissible = intervention is None
        suffix_diverged = False
        last_step: Any = None
        outcome_details: dict[str, Any] = {}
        try:
            current = branch.reset()
            if intervention is None:
                for source_step in plan.steps:
                    checkpoint = branch.capture_checkpoint()
                    pre_state_verified = pre_state_verified and (
                        checkpoint.snapshot.state_snapshot_sha256
                        == source_step.state_snapshot_sha256
                        and tuple(current.admissible_actions) == source_step.admissible_actions
                    )
                    actor_history.append((current.text, source_step.source_action))
                    last_step = self._direct_step(branch, source_step.source_action)
                    executed.append(source_step.source_action)
                    current = last_step.observation
                    if last_step.won or last_step.done:
                        break
                return self._finish(
                    branch=branch,
                    last_step=last_step,
                    success=bool(last_step is not None and last_step.won),
                    terminal_reason=(
                        "environment_won"
                        if last_step is not None and last_step.won
                        else "environment_done"
                        if last_step is not None and last_step.done
                        else plan.source_terminal_reason or "source_failure"
                    ),
                    pre_state_verified=pre_state_verified,
                    replacement_admissible=True,
                    suffix_diverged=False,
                    executed=executed,
                    outcome_details={"arm": "baseline"},
                )

            for source_step in plan.steps[: intervention.step_idx]:
                checkpoint = branch.capture_checkpoint()
                pre_state_verified = pre_state_verified and (
                    checkpoint.snapshot.state_snapshot_sha256 == source_step.state_snapshot_sha256
                    and tuple(current.admissible_actions) == source_step.admissible_actions
                )
                actor_history.append((current.text, source_step.source_action))
                last_step = self._direct_step(branch, source_step.source_action)
                executed.append(source_step.source_action)
                current = last_step.observation
                if last_step.won or last_step.done:
                    return self._finish(
                        branch=branch,
                        last_step=last_step,
                        success=False,
                        terminal_reason="prefix_terminated_before_intervention",
                        pre_state_verified=False,
                        replacement_admissible=False,
                        suffix_diverged=False,
                        executed=executed,
                        outcome_details={"arm": "intervention"},
                    )

            source_step = plan.steps[intervention.step_idx]
            checkpoint = branch.capture_checkpoint()
            pre_state_verified = pre_state_verified and (
                checkpoint.snapshot.state_snapshot_sha256
                == intervention.source_pre_state_snapshot_sha256
                and tuple(current.admissible_actions) == source_step.admissible_actions
            )
            sham_placebo_action: str | None = None
            if causal_arm == "intervention":
                applied_action = intervention.replacement_action
            elif causal_arm == "matched_sham":
                sham_placebo_action = _matched_sham_placebo_action(
                    source_step.admissible_actions,
                    source_action=intervention.source_action,
                    replacement_action=intervention.replacement_action,
                )
                applied_action = (
                    sham_placebo_action
                    if sham_placebo_action is not None
                    else intervention.source_action
                )
            else:
                # The reverse-injection arm re-applies the fault action under the
                # repair arm's continuation protocol (treatment rollback).
                applied_action = intervention.source_action
            replacement_admissible = (
                applied_action in current.admissible_actions
                and applied_action in source_step.admissible_actions
            )
            sham_not_constructible = (
                causal_arm == "matched_sham" and sham_placebo_action is None
            )
            if (
                not pre_state_verified
                or not replacement_admissible
                or sham_not_constructible
            ):
                return self._finish(
                    branch=branch,
                    last_step=last_step,
                    success=False,
                    terminal_reason=(
                        "source_pre_state_identity_mismatch"
                        if not pre_state_verified
                        else "matched_sham_not_constructible"
                        if sham_not_constructible
                        else "treatment_action_not_admissible"
                    ),
                    pre_state_verified=pre_state_verified,
                    replacement_admissible=replacement_admissible,
                    suffix_diverged=False,
                    executed=executed,
                    outcome_details={"arm": causal_arm},
                )

            intervention_observation = current.text
            last_step = branch.step(applied_action)
            executed.append(applied_action)
            actor_history.append((intervention_observation, applied_action))
            current = last_step.observation
            outcome_details = {
                "arm": causal_arm,
                "intervention_id": intervention.intervention_id,
                "applied_action": applied_action,
                "intervention_execution_mode": "verified_branch_step",
                "intervention_restore_receipt_sha256": last_step.restore_receipt_sha256,
                "intervention_restore_receipt_path": last_step.restore_receipt_path,
            }
            if causal_arm == "matched_sham":
                outcome_details["matched_sham_placebo_action"] = applied_action
            if last_step.won or last_step.done:
                return self._finish(
                    branch=branch,
                    last_step=last_step,
                    success=bool(last_step.won),
                    terminal_reason=("environment_won" if last_step.won else "environment_done"),
                    pre_state_verified=pre_state_verified,
                    replacement_admissible=replacement_admissible,
                    suffix_diverged=False,
                    executed=executed,
                    outcome_details=outcome_details,
                )

            if continuation_mode == "open_loop":
                for suffix_step in plan.steps[intervention.step_idx + 1 :]:
                    if suffix_step.source_action not in current.admissible_actions:
                        suffix_diverged = True
                        break
                    last_step = self._direct_step(branch, suffix_step.source_action)
                    executed.append(suffix_step.source_action)
                    current = last_step.observation
                    if last_step.won or last_step.done:
                        break
            else:
                assert self._closed_loop_policy is not None
                while len(executed) < len(plan.steps) and not (last_step.won or last_step.done):
                    next_index = len(executed)
                    action, policy_evidence = self._closed_loop_policy.choose_action(
                        arm_id=f"{intervention.intervention_id}:{causal_arm}",
                        task_id=plan.task_id,
                        step_idx=next_index,
                        task_description=plan.task_description,
                        current_observation=current.text,
                        admissible_actions=tuple(current.admissible_actions),
                        history=tuple(actor_history),
                        remaining_budget=len(plan.steps) - len(executed),
                        sampling_seed=plan.steps[next_index].sampling_seed,
                        sampling_temperature=plan.sampling_temperature,
                    )
                    policy_record = dict(policy_evidence)
                    policy_record["action_in_live_admissible_menu"] = (
                        action in current.admissible_actions
                    )
                    outcome_details.setdefault("closed_loop_policy_evidence", []).append(
                        policy_record
                    )
                    before_text = current.text
                    last_step = self._direct_step(branch, action)
                    executed.append(action)
                    actor_history.append((before_text, action))
                    current = last_step.observation

            return self._finish(
                branch=branch,
                last_step=last_step,
                success=bool(last_step is not None and last_step.won and not suffix_diverged),
                terminal_reason=(
                    "suffix_diverged"
                    if suffix_diverged
                    else "environment_won"
                    if last_step is not None and last_step.won
                    else "environment_done"
                    if last_step is not None and last_step.done
                    else "budget_exhausted"
                ),
                pre_state_verified=pre_state_verified,
                replacement_admissible=replacement_admissible,
                suffix_diverged=suffix_diverged,
                executed=executed,
                outcome_details=outcome_details,
            )
        finally:
            branch.close()

    @staticmethod
    def _direct_step(branch: AlfWorldBranch, action: str) -> Any:
        direct = getattr(branch, "step_unverified", None)
        return direct(action) if callable(direct) else branch.step(action)

    @staticmethod
    def _finish(
        *,
        branch: AlfWorldBranch,
        last_step: Any,
        success: bool,
        terminal_reason: str,
        pre_state_verified: bool,
        replacement_admissible: bool,
        suffix_diverged: bool,
        executed: list[str],
        outcome_details: Mapping[str, Any],
    ) -> ReplayArmOutcome:
        final_checkpoint = branch.capture_checkpoint()
        evidence = dict(outcome_details)
        if last_step is not None:
            evidence.update(
                {
                    "environment_done": bool(last_step.done),
                    "environment_won": bool(last_step.won),
                    "reward": float(last_step.reward),
                }
            )
        return ReplayArmOutcome(
            success=success,
            terminal_reason=terminal_reason,
            pre_state_identity_verified=pre_state_verified,
            replacement_admissible=replacement_admissible,
            suffix_diverged=suffix_diverged,
            executed_actions=tuple(executed),
            final_state_snapshot_sha256=(final_checkpoint.snapshot.state_snapshot_sha256),
            physical_call_counts=dict(branch.physical_call_counters()),
            outcome_evidence=evidence,
        )


@dataclass(frozen=True)
class CounterfactualCandidateResult:
    intervention: SingleActionIntervention
    outcome: ReplayArmOutcome
    repair_admitted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention": asdict(self.intervention),
            "outcome": self.outcome.to_dict(),
            "repair_admitted": self.repair_admitted,
        }


@dataclass(frozen=True)
class AlfWorldSingleActionCounterfactualSearchReceipt:
    plan_sha256: str
    continuation_mode: ContinuationMode
    baseline: ReplayArmOutcome
    candidate_results: tuple[CounterfactualCandidateResult, ...]
    baseline_reproduced_failure: bool
    root_cause_verified: bool = False
    schema: str = ALFWORLD_ACTION_COUNTERFACTUAL_SEARCH_SCHEMA

    @property
    def repair_candidates(self) -> tuple[CounterfactualCandidateResult, ...]:
        return tuple(result for result in self.candidate_results if result.repair_admitted)

    @property
    def suffix_divergence_count(self) -> int:
        return sum(result.outcome.suffix_diverged for result in self.candidate_results)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_sha256": self.plan_sha256,
            "continuation_mode": self.continuation_mode,
            "baseline": self.baseline.to_dict(),
            "candidate_results": [result.to_dict() for result in self.candidate_results],
            "baseline_reproduced_failure": self.baseline_reproduced_failure,
            "repair_candidate_count": len(self.repair_candidates),
            "suffix_divergence_count": self.suffix_divergence_count,
            "root_cause_verified": self.root_cause_verified,
        }


@dataclass(frozen=True)
class CounterfactualStageResult:
    step_idx: int
    planned_candidate_count: int
    candidate_results: tuple[CounterfactualCandidateResult, ...]

    def __post_init__(self) -> None:
        if self.step_idx < 0 or self.planned_candidate_count < 0:
            raise AlfWorldCounterfactualError("counterfactual stage identity is invalid")
        if len(self.candidate_results) != self.planned_candidate_count:
            raise AlfWorldCounterfactualError(
                "counterfactual stage stopped before its frozen candidate set completed"
            )
        if any(result.intervention.step_idx != self.step_idx for result in self.candidate_results):
            raise AlfWorldCounterfactualError(
                "counterfactual stage contains a candidate from another step"
            )

    @property
    def repair_candidates(self) -> tuple[CounterfactualCandidateResult, ...]:
        return tuple(result for result in self.candidate_results if result.repair_admitted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "planned_candidate_count": self.planned_candidate_count,
            "executed_candidate_count": len(self.candidate_results),
            "candidate_results": [result.to_dict() for result in self.candidate_results],
            "repair_candidate_count": len(self.repair_candidates),
            "stage_completed": True,
        }


@dataclass(frozen=True)
class AlfWorldStagedSingleActionCounterfactualSearchReceipt:
    plan_sha256: str
    continuation_mode: ContinuationMode
    baseline: ReplayArmOutcome
    stages: tuple[CounterfactualStageResult, ...]
    baseline_reproduced_failure: bool
    total_planned_candidate_count: int
    root_cause_verified: bool = False
    schema: str = ALFWORLD_STAGED_ACTION_COUNTERFACTUAL_SEARCH_SCHEMA

    @property
    def candidate_results(self) -> tuple[CounterfactualCandidateResult, ...]:
        return tuple(result for stage in self.stages for result in stage.candidate_results)

    @property
    def repair_candidates(self) -> tuple[CounterfactualCandidateResult, ...]:
        return tuple(result for result in self.candidate_results if result.repair_admitted)

    @property
    def completed_steps(self) -> tuple[int, ...]:
        return tuple(stage.step_idx for stage in self.stages)

    @property
    def stopping_step(self) -> int | None:
        repair_steps = tuple(stage.step_idx for stage in self.stages if stage.repair_candidates)
        return repair_steps[0] if repair_steps else None

    @property
    def executed_candidate_count(self) -> int:
        return len(self.candidate_results)

    @property
    def full_search_exhausted(self) -> bool:
        return self.executed_candidate_count == self.total_planned_candidate_count

    @property
    def stopping_rule_satisfied(self) -> bool:
        if not self.stages:
            return self.total_planned_candidate_count == 0
        repair_stages = tuple(stage for stage in self.stages if stage.repair_candidates)
        if repair_stages:
            return len(repair_stages) == 1 and repair_stages[0] is self.stages[-1]
        return self.full_search_exhausted

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_sha256": self.plan_sha256,
            "continuation_mode": self.continuation_mode,
            "baseline": self.baseline.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "completed_steps": list(self.completed_steps),
            "stopping_step": self.stopping_step,
            "stopping_rule_satisfied": self.stopping_rule_satisfied,
            "total_planned_candidate_count": self.total_planned_candidate_count,
            "executed_candidate_count": self.executed_candidate_count,
            "repair_candidate_count": len(self.repair_candidates),
            "full_search_exhausted": self.full_search_exhausted,
            "baseline_reproduced_failure": self.baseline_reproduced_failure,
            "root_cause_verified": self.root_cause_verified,
        }


@dataclass(frozen=True)
class InterventionVerifiedRootCauseEvidence:
    verified: bool
    missing_requirements: tuple[str, ...]
    baseline_failed: bool
    intervention_repaired: bool
    matched_sham_failed: bool
    reverse_injection_failed: bool
    schema: str = ALFWORLD_ROOT_CAUSE_EVIDENCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterventionVerifiedSufficientCauseEvidence:
    verified: bool
    missing_requirements: tuple[str, ...]
    baseline_failed: bool
    intervention_repaired: bool
    matched_sham_failed: bool
    schema: str = ALFWORLD_SUFFICIENT_CAUSE_EVIDENCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlfWorldCausalValidationReceipt:
    binding: CausalArmBinding
    baseline: ReplayArmOutcome
    intervention: ReplayArmOutcome
    matched_sham: ReplayArmOutcome
    reverse_injection: ReplayArmOutcome
    verdict: InterventionVerifiedRootCauseEvidence
    schema: str = ALFWORLD_CAUSAL_VALIDATION_SCHEMA

    @property
    def root_cause_verified(self) -> bool:
        return self.verdict.verified

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": self.binding.to_dict(),
            "binding_sha256": self.binding.sha256,
            "baseline": self.baseline.to_dict(),
            "intervention": self.intervention.to_dict(),
            "matched_sham": self.matched_sham.to_dict(),
            "reverse_injection": self.reverse_injection.to_dict(),
            "verdict": self.verdict.to_dict(),
            "root_cause_verified": self.root_cause_verified,
        }


@dataclass(frozen=True)
class AlfWorldThreeArmCausalValidationReceipt:
    binding: CausalArmBinding
    baseline: ReplayArmOutcome
    intervention: ReplayArmOutcome
    matched_sham: ReplayArmOutcome
    verdict: InterventionVerifiedSufficientCauseEvidence
    schema: str = ALFWORLD_THREE_ARM_CAUSAL_VALIDATION_SCHEMA

    @property
    def sufficient_cause_verified(self) -> bool:
        return self.verdict.verified

    @property
    def causal_gold_admitted(self) -> bool:
        return self.verdict.verified

    @property
    def root_cause_verified(self) -> bool:
        """Do not promote a local sufficient cause to a unique global root."""

        return False

    @property
    def claim_scope(self) -> str:
        return "single_action_sufficient_cause_under_frozen_controller"

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": self.binding.to_dict(),
            "binding_sha256": self.binding.sha256,
            "baseline": self.baseline.to_dict(),
            "intervention": self.intervention.to_dict(),
            "matched_sham": self.matched_sham.to_dict(),
            "verdict": self.verdict.to_dict(),
            "sufficient_cause_verified": self.sufficient_cause_verified,
            "causal_gold_admitted": self.causal_gold_admitted,
            "root_cause_verified": self.root_cause_verified,
            "claim_scope": self.claim_scope,
        }


def build_alfworld_source_trajectory(
    records: Sequence[TrainingTrajectoryRecord],
) -> AlfWorldSourceTrajectory:
    """Validate and project one terminal trace without assuming failure."""

    ordered = tuple(sorted(records, key=lambda record: record.step_idx))
    if not ordered:
        raise AlfWorldCounterfactualError("source projection requires one trajectory")
    identity = {
        (record.benchmark, record.task_id, record.traj_uid, record.split, record.seed)
        for record in ordered
    }
    if len(identity) != 1 or ordered[0].benchmark != "alfworld_native":
        raise AlfWorldCounterfactualError("records do not describe one ALFWorld trajectory")
    if tuple(record.step_idx for record in ordered) != tuple(range(len(ordered))):
        raise AlfWorldCounterfactualError("source trajectory steps must be contiguous from zero")
    if any(record.terminal for record in ordered[:-1]) or not ordered[-1].terminal:
        raise AlfWorldCounterfactualError("source trajectory must have exactly one final terminal")
    if not ordered[-1].success and any(record.success for record in ordered[:-1]):
        raise AlfWorldCounterfactualError(
            "source trajectory reports success before its terminal step"
        )
    if ordered[0].seed is None:
        raise AlfWorldCounterfactualError("source trajectory lacks an environment seed")
    actor_protocol_ids = {str(record.extra.get("actor_protocol_id", "")) for record in ordered}
    sampling_seed_bases = {record.extra.get("sampling_seed_base") for record in ordered}
    sampling_seed_schedules = {
        str(record.extra.get("sampling_seed_schedule", "")) for record in ordered
    }
    sampling_temperatures = {record.extra.get("sampling_temperature") for record in ordered}
    model_ids = {record.model_id for record in ordered}
    frozen_actor_hashes = {
        name: {str(record.extra.get(name, "")) for record in ordered}
        for name in (
            "model_io_profile_sha256",
            "response_contract_sha256",
            "task_manifest_lock_sha256",
            "chat_template_sha256",
        )
    }
    if (
        len(actor_protocol_ids) != 1
        or not next(iter(actor_protocol_ids))
        or len(sampling_seed_bases) != 1
        or len(sampling_seed_schedules) != 1
        or next(iter(sampling_seed_schedules)) != ALFWORLD_SOURCE_SEED_SCHEDULE
        or len(sampling_temperatures) != 1
        or len(model_ids) != 1
        or any(len(values) != 1 for values in frozen_actor_hashes.values())
    ):
        raise AlfWorldCounterfactualError("source actor identity or sampling schedule changed")
    for name, values in frozen_actor_hashes.items():
        value = next(iter(values))
        try:
            _require_sha256(name, value)
        except AlfWorldCounterfactualError as error:
            raise AlfWorldCounterfactualError("source actor identity is incomplete") from error
    sampling_seed_base = next(iter(sampling_seed_bases))
    sampling_temperature = next(iter(sampling_temperatures))
    if (
        isinstance(sampling_seed_base, bool)
        or not isinstance(sampling_seed_base, int)
        or sampling_seed_base < 0
        or isinstance(sampling_temperature, bool)
        or not isinstance(sampling_temperature, (int, float))
        or sampling_temperature <= 0
    ):
        raise AlfWorldCounterfactualError("source sampling schedule is invalid")

    source_steps: list[AlfWorldSourceStep] = []
    for record in ordered:
        action = _source_action(record)
        raw_menu = record.extra.get("alfworld_admissible_actions")
        if not isinstance(raw_menu, (list, tuple)) or not raw_menu:
            raise AlfWorldCounterfactualError(
                f"source step {record.step_idx} lacks admissible actions"
            )
        menu = tuple(str(value).strip() for value in raw_menu)
        if any(not value or "\n" in value for value in menu) or len(set(menu)) != len(menu):
            raise AlfWorldCounterfactualError(
                f"source step {record.step_idx} has a malformed admissible-action menu"
            )
        pre_hash = record.state_snapshot_sha256
        post_hash = str(record.extra.get("post_state_snapshot_sha256", ""))
        _require_sha256("state_snapshot_sha256", pre_hash)
        _require_sha256("post_state_snapshot_sha256", post_hash)
        pre_path = str(record.extra.get("state_snapshot_artifact_path", ""))
        post_path = str(record.extra.get("post_state_snapshot_artifact_path", ""))
        if not pre_path or not post_path:
            raise AlfWorldCounterfactualError(
                f"source step {record.step_idx} lacks state artifact paths"
            )
        action_admissible = action in menu
        sampling_seed = record.extra.get("sampling_seed")
        if (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, int)
            or sampling_seed != sampling_seed_base + record.step_idx
        ):
            raise AlfWorldCounterfactualError(
                f"source step {record.step_idx} sampling seed changed"
            )
        source_steps.append(
            AlfWorldSourceStep(
                step_idx=record.step_idx,
                source_action=action,
                admissible_actions=menu,
                source_action_admissible=action_admissible,
                state_snapshot_sha256=pre_hash,
                state_snapshot_artifact_path=pre_path,
                post_state_snapshot_sha256=post_hash,
                post_state_snapshot_artifact_path=post_path,
                observation=record.observation,
                next_observation=record.next_observation,
                sampling_seed=sampling_seed,
            )
        )
    source_hash = _canonical_sha256([record.to_dict() for record in ordered])
    return AlfWorldSourceTrajectory(
        records=ordered,
        task_id=ordered[0].task_id,
        traj_uid=ordered[0].traj_uid,
        split=ordered[0].split,
        environment_seed=int(ordered[0].seed),
        task_description=str(
            ordered[0].extra.get("actor_task_description") or ordered[0].instruction
        ),
        actor_protocol_id=next(iter(actor_protocol_ids)),
        model_id=next(iter(model_ids)),
        model_io_profile_sha256=next(iter(frozen_actor_hashes["model_io_profile_sha256"])),
        response_contract_sha256=next(iter(frozen_actor_hashes["response_contract_sha256"])),
        task_manifest_lock_sha256=next(iter(frozen_actor_hashes["task_manifest_lock_sha256"])),
        chat_template_sha256=next(iter(frozen_actor_hashes["chat_template_sha256"])),
        sampling_seed_base=sampling_seed_base,
        sampling_seed_schedule=next(iter(sampling_seed_schedules)),
        sampling_temperature=float(sampling_temperature),
        success=ordered[-1].success,
        terminal_reason=ordered[-1].terminal_reason,
        source_trajectory_sha256=source_hash,
        steps=tuple(source_steps),
    )


def build_alfworld_single_action_counterfactual_plan(
    records: Sequence[TrainingTrajectoryRecord],
) -> AlfWorldSingleActionCounterfactualPlan:
    """Freeze all state-dependent one-action replacements for one failed trace.

    The exact candidate count is ``sum(|A(s_t) \\ {a_t}|)``.  It reduces to
    ``sum(|A(s_t)| - 1)`` only when every source action belongs to its retained
    admissible-action set.
    """

    source = build_alfworld_source_trajectory(records)
    if source.success:
        raise AlfWorldCounterfactualError(
            "counterfactual search requires a failed source trajectory"
        )

    candidates: list[SingleActionIntervention] = []
    outside: list[int] = []
    for step in source.steps:
        if not step.source_action_admissible:
            outside.append(step.step_idx)
        for replacement in step.admissible_actions:
            if replacement == step.source_action:
                continue
            candidate_identity = {
                "traj_uid": source.traj_uid,
                "step_idx": step.step_idx,
                "source_action": step.source_action,
                "replacement_action": replacement,
                "source_pre_state_snapshot_sha256": step.state_snapshot_sha256,
            }
            candidates.append(
                SingleActionIntervention(
                    intervention_id=(
                        f"step-{step.step_idx:04d}-{_canonical_sha256(candidate_identity)[:16]}"
                    ),
                    step_idx=step.step_idx,
                    source_action=step.source_action,
                    replacement_action=replacement,
                    source_pre_state_snapshot_sha256=step.state_snapshot_sha256,
                )
            )

    return AlfWorldSingleActionCounterfactualPlan(
        task_id=source.task_id,
        traj_uid=source.traj_uid,
        split=source.split,
        environment_seed=source.environment_seed,
        task_description=source.task_description,
        actor_protocol_id=source.actor_protocol_id,
        model_id=source.model_id,
        model_io_profile_sha256=source.model_io_profile_sha256,
        response_contract_sha256=source.response_contract_sha256,
        task_manifest_lock_sha256=source.task_manifest_lock_sha256,
        chat_template_sha256=source.chat_template_sha256,
        sampling_seed_base=source.sampling_seed_base,
        sampling_seed_schedule=source.sampling_seed_schedule,
        sampling_temperature=source.sampling_temperature,
        source_terminal_reason=source.terminal_reason,
        source_trajectory_sha256=source.source_trajectory_sha256,
        steps=source.steps,
        candidates=tuple(candidates),
        source_action_outside_admissible_steps=tuple(outside),
    )


def assess_alfworld_single_action_root_cause_admission(
    records: Sequence[TrainingTrajectoryRecord],
) -> AlfWorldSingleActionRootCauseAdmission:
    """Reject source failures that cannot support a single-action root-cause claim.

    This gate uses source-only facts.  In particular, an episode that consumes
    its whole interaction budget using only observation actions is a diffuse
    non-progress failure, not evidence that one executed action damaged task
    state.  Such an episode may be studied as a loop, but not scored as a
    single-action causal-localization case.
    """

    source = build_alfworld_source_trajectory(records)
    observation_only_verbs = {"examine", "inventory", "look"}
    observation_only_step_count = sum(
        step.source_action.strip().lower().split(maxsplit=1)[0] in observation_only_verbs
        for step in source.steps
        if step.source_action.strip()
    )
    blockers: list[str] = []
    if source.success:
        blockers.append("source_trajectory_succeeded")
    if source.terminal_reason == "max_episode_steps" and observation_only_step_count == len(
        source.steps
    ):
        blockers.append("diffuse_observation_only_horizon_exhaustion")
    return AlfWorldSingleActionRootCauseAdmission(
        source_trajectory_sha256=source.source_trajectory_sha256,
        task_id=source.task_id,
        traj_uid=source.traj_uid,
        terminal_reason=source.terminal_reason,
        source_step_count=len(source.steps),
        observation_only_step_count=observation_only_step_count,
        unique_source_actions=tuple(sorted({step.source_action for step in source.steps})),
        decision="rejected" if blockers else "admitted",
        blockers=tuple(blockers),
    )


def require_alfworld_single_action_root_cause_admission(
    admission: AlfWorldSingleActionRootCauseAdmission,
) -> None:
    """Fail before dispatch when a source cannot carry the causal claim."""

    if admission.decision != "admitted":
        raise AlfWorldCounterfactualError(
            "single-action root-cause source admission rejected: " + ", ".join(admission.blockers)
        )


def run_alfworld_single_action_counterfactual_search(
    plan: AlfWorldSingleActionCounterfactualPlan,
    *,
    replay_backend: CounterfactualReplayBackend,
    continuation_mode: ContinuationMode,
) -> AlfWorldSingleActionCounterfactualSearchReceipt:
    """Run one baseline plus every frozen single-action replacement arm."""

    if continuation_mode not in {"open_loop", "closed_loop"}:
        raise AlfWorldCounterfactualError("unsupported continuation mode")
    baseline = _run_verified_baseline(
        plan=plan,
        replay_backend=replay_backend,
        continuation_mode=continuation_mode,
    )

    results: list[CounterfactualCandidateResult] = []
    for candidate in plan.candidates:
        outcome = replay_backend.replay(
            plan=plan,
            intervention=candidate,
            continuation_mode=continuation_mode,
        )
        repair_admitted = bool(
            outcome.success
            and outcome.pre_state_identity_verified
            and outcome.replacement_admissible
            and not outcome.suffix_diverged
        )
        results.append(
            CounterfactualCandidateResult(
                intervention=candidate,
                outcome=outcome,
                repair_admitted=repair_admitted,
            )
        )
    return AlfWorldSingleActionCounterfactualSearchReceipt(
        plan_sha256=plan.sha256,
        continuation_mode=continuation_mode,
        baseline=baseline,
        candidate_results=tuple(results),
        baseline_reproduced_failure=True,
        root_cause_verified=False,
    )


def _run_verified_baseline(
    *,
    plan: AlfWorldSingleActionCounterfactualPlan,
    replay_backend: CounterfactualReplayBackend,
    continuation_mode: ContinuationMode,
) -> ReplayArmOutcome:
    baseline = replay_backend.replay(
        plan=plan,
        intervention=None,
        continuation_mode=continuation_mode,
    )
    baseline_failed = (
        not baseline.success
        and baseline.pre_state_identity_verified
        and not baseline.suffix_diverged
        and baseline.executed_actions == tuple(step.source_action for step in plan.steps)
        and baseline.terminal_reason == plan.source_terminal_reason
        and baseline.final_state_snapshot_sha256 == plan.steps[-1].post_state_snapshot_sha256
    )
    if not baseline_failed:
        raise AlfWorldCounterfactualError(
            "baseline did not reproduce the source failure from verified states"
        )
    return baseline


def run_alfworld_staged_single_action_counterfactual_search(
    plan: AlfWorldSingleActionCounterfactualPlan,
    *,
    replay_backend: CounterfactualReplayBackend,
    continuation_mode: ContinuationMode,
    baseline_observer: Callable[[ReplayArmOutcome], None] | None = None,
    stage_observer: Callable[[CounterfactualStageResult], None] | None = None,
) -> AlfWorldStagedSingleActionCounterfactualSearchReceipt:
    """Run frozen candidates by ascending step and stop after one complete repair stage."""

    if continuation_mode not in {"open_loop", "closed_loop"}:
        raise AlfWorldCounterfactualError("unsupported continuation mode")
    baseline = _run_verified_baseline(
        plan=plan,
        replay_backend=replay_backend,
        continuation_mode=continuation_mode,
    )
    if baseline_observer is not None:
        baseline_observer(baseline)
    stages: list[CounterfactualStageResult] = []
    for step in plan.steps:
        stage_candidates = tuple(
            candidate for candidate in plan.candidates if candidate.step_idx == step.step_idx
        )
        results: list[CounterfactualCandidateResult] = []
        for candidate in stage_candidates:
            outcome = replay_backend.replay(
                plan=plan,
                intervention=candidate,
                continuation_mode=continuation_mode,
            )
            results.append(
                CounterfactualCandidateResult(
                    intervention=candidate,
                    outcome=outcome,
                    repair_admitted=bool(
                        outcome.success
                        and outcome.pre_state_identity_verified
                        and outcome.replacement_admissible
                        and not outcome.suffix_diverged
                    ),
                )
            )
        stage = CounterfactualStageResult(
            step_idx=step.step_idx,
            planned_candidate_count=len(stage_candidates),
            candidate_results=tuple(results),
        )
        stages.append(stage)
        if stage_observer is not None:
            stage_observer(stage)
        if stage.repair_candidates:
            break
    receipt = AlfWorldStagedSingleActionCounterfactualSearchReceipt(
        plan_sha256=plan.sha256,
        continuation_mode=continuation_mode,
        baseline=baseline,
        stages=tuple(stages),
        baseline_reproduced_failure=True,
        total_planned_candidate_count=len(plan.candidates),
    )
    if not receipt.stopping_rule_satisfied:
        raise AlfWorldCounterfactualError("staged stopping rule was not satisfied")
    return receipt


def assess_intervention_verified_root_cause(
    *,
    binding: CausalArmBinding,
    baseline: ReplayArmOutcome | None,
    intervention: ReplayArmOutcome | None,
    matched_sham: ReplayArmOutcome | None,
    reverse_injection: ReplayArmOutcome | None,
) -> InterventionVerifiedRootCauseEvidence:
    """Require all four causal arms before labeling an event a root cause.

    Arm roles: the baseline replays the fault trajectory, the intervention
    applies the proposed repair, the matched sham applies a frozen placebo
    action under the repair protocol, and the reverse injection re-applies the
    fault action under the repair protocol.  Every arm must bind to the same
    causal binding, and the reverse injection must be independently executed
    rather than a duplicate of another arm's trajectory.
    """

    def binding_matches(outcome: ReplayArmOutcome | None, arm: str) -> bool:
        evidence = outcome.outcome_evidence if outcome is not None else None
        return bool(
            isinstance(evidence, Mapping)
            and evidence.get("causal_arm") == arm
            and evidence.get("causal_binding_sha256") == binding.sha256
        )

    def verified_intervention_replay(outcome: ReplayArmOutcome | None) -> bool:
        evidence = outcome.outcome_evidence if outcome is not None else None
        if not isinstance(evidence, Mapping):
            return False
        receipt_sha256 = evidence.get("intervention_restore_receipt_sha256")
        return bool(
            evidence.get("intervention_execution_mode") == "verified_branch_step"
            and isinstance(receipt_sha256, str)
            and len(receipt_sha256) == 64
            and all(character in "0123456789abcdef" for character in receipt_sha256)
        )

    checks = {
        "baseline_failure": bool(
            baseline is not None
            and not baseline.success
            and baseline.pre_state_identity_verified
            and not baseline.suffix_diverged
        ),
        "intervention_repair": bool(
            intervention is not None
            and intervention.success
            and intervention.pre_state_identity_verified
            and intervention.replacement_admissible
            and not intervention.suffix_diverged
        ),
        "matched_sham": bool(
            matched_sham is not None
            and not matched_sham.success
            and matched_sham.pre_state_identity_verified
            and not matched_sham.suffix_diverged
        ),
        "matched_sham_placebo": bool(
            matched_sham is not None
            and isinstance(matched_sham.outcome_evidence, Mapping)
            and matched_sham.outcome_evidence.get("matched_sham_placebo_action")
            not in {None, binding.source_action, binding.replacement_action}
        ),
        "reverse_injection": bool(
            reverse_injection is not None
            and not reverse_injection.success
            and reverse_injection.pre_state_identity_verified
            and not reverse_injection.suffix_diverged
        ),
        "baseline_binding": binding_matches(baseline, "baseline"),
        "intervention_binding": binding_matches(intervention, "intervention"),
        "matched_sham_binding": binding_matches(matched_sham, "matched_sham"),
        "reverse_injection_binding": binding_matches(reverse_injection, "reverse_injection"),
        "intervention_matched_replay": verified_intervention_replay(intervention),
        "matched_sham_matched_replay": verified_intervention_replay(matched_sham),
        "reverse_injection_matched_replay": verified_intervention_replay(reverse_injection),
        "nonredundant_reverse_injection": bool(
            reverse_injection is not None
            and isinstance(reverse_injection.outcome_evidence, Mapping)
            and reverse_injection.outcome_evidence.get("reverse_injection_independent") is True
        ),
    }
    missing = tuple(name for name, passed in checks.items() if not passed)
    return InterventionVerifiedRootCauseEvidence(
        verified=not missing,
        missing_requirements=missing,
        baseline_failed=checks["baseline_failure"],
        intervention_repaired=checks["intervention_repair"],
        matched_sham_failed=checks["matched_sham"],
        reverse_injection_failed=checks["reverse_injection"],
    )


def assess_intervention_verified_sufficient_cause(
    *,
    binding: CausalArmBinding,
    baseline: ReplayArmOutcome | None,
    intervention: ReplayArmOutcome | None,
    matched_sham: ReplayArmOutcome | None,
) -> InterventionVerifiedSufficientCauseEvidence:
    """Assess the preregistered three-arm sufficient-cause contrast.

    The matched sham applies a frozen placebo action (neither the fault action
    nor the proposed repair) through the same public verified step and frozen
    controller.  This is a local sufficiency claim, not uniqueness or global
    necessity.
    """

    def binding_matches(outcome: ReplayArmOutcome | None, arm: str) -> bool:
        evidence = outcome.outcome_evidence if outcome is not None else None
        return bool(
            isinstance(evidence, Mapping)
            and evidence.get("causal_arm") == arm
            and evidence.get("causal_binding_sha256") == binding.sha256
        )

    def verified_treatment_step(
        outcome: ReplayArmOutcome | None,
        *,
        expected_action: str,
    ) -> bool:
        evidence = outcome.outcome_evidence if outcome is not None else None
        if not isinstance(evidence, Mapping):
            return False
        receipt_sha256 = evidence.get("intervention_restore_receipt_sha256")
        return bool(
            evidence.get("intervention_execution_mode") == "verified_branch_step"
            and evidence.get("applied_action") == expected_action
            and isinstance(receipt_sha256, str)
            and len(receipt_sha256) == 64
            and all(character in "0123456789abcdef" for character in receipt_sha256)
        )

    def verified_placebo_step(outcome: ReplayArmOutcome | None) -> bool:
        evidence = outcome.outcome_evidence if outcome is not None else None
        if not isinstance(evidence, Mapping):
            return False
        receipt_sha256 = evidence.get("intervention_restore_receipt_sha256")
        applied_action = evidence.get("applied_action")
        return bool(
            evidence.get("intervention_execution_mode") == "verified_branch_step"
            and isinstance(applied_action, str)
            and applied_action not in {binding.source_action, binding.replacement_action}
            and isinstance(receipt_sha256, str)
            and len(receipt_sha256) == 64
            and all(character in "0123456789abcdef" for character in receipt_sha256)
        )

    checks = {
        "baseline_failure": bool(
            baseline is not None
            and not baseline.success
            and baseline.pre_state_identity_verified
            and not baseline.suffix_diverged
        ),
        "intervention_repair": bool(
            intervention is not None
            and intervention.success
            and intervention.pre_state_identity_verified
            and intervention.replacement_admissible
            and not intervention.suffix_diverged
        ),
        "matched_sham_failure": bool(
            matched_sham is not None
            and not matched_sham.success
            and matched_sham.pre_state_identity_verified
            and not matched_sham.suffix_diverged
        ),
        "baseline_binding": binding_matches(baseline, "baseline"),
        "intervention_binding": binding_matches(intervention, "intervention"),
        "matched_sham_binding": binding_matches(matched_sham, "matched_sham"),
        "intervention_verified_step": verified_treatment_step(
            intervention,
            expected_action=binding.replacement_action,
        ),
        "matched_sham_verified_step": verified_placebo_step(matched_sham),
    }
    missing = tuple(name for name, passed in checks.items() if not passed)
    return InterventionVerifiedSufficientCauseEvidence(
        verified=not missing,
        missing_requirements=missing,
        baseline_failed=checks["baseline_failure"],
        intervention_repaired=checks["intervention_repair"],
        matched_sham_failed=checks["matched_sham_failure"],
    )


def run_alfworld_single_action_causal_validation(
    plan: AlfWorldSingleActionCounterfactualPlan,
    *,
    intervention: SingleActionIntervention,
    replay_backend: AlfWorldBranchReplayBackend,
    controller_identity_sha256: str,
) -> AlfWorldCausalValidationReceipt:
    """Run and bind baseline, repair, sham, and reverse-injection arms."""

    if intervention not in plan.candidates:
        raise AlfWorldCounterfactualError("causal candidate is not in the frozen plan")
    binding = CausalArmBinding(
        plan_sha256=plan.sha256,
        source_trajectory_sha256=plan.source_trajectory_sha256,
        intervention_id=intervention.intervention_id,
        step_idx=intervention.step_idx,
        source_pre_state_snapshot_sha256=(intervention.source_pre_state_snapshot_sha256),
        source_action=intervention.source_action,
        replacement_action=intervention.replacement_action,
        continuation_mode="closed_loop",
        controller_identity_sha256=controller_identity_sha256,
    )
    baseline = replay_backend.replay(
        plan=plan,
        intervention=None,
        continuation_mode="closed_loop",
    )
    baseline_reproduced = bool(
        not baseline.success
        and baseline.pre_state_identity_verified
        and not baseline.suffix_diverged
        and baseline.executed_actions == tuple(step.source_action for step in plan.steps)
        and baseline.terminal_reason == plan.source_terminal_reason
        and baseline.final_state_snapshot_sha256 == plan.steps[-1].post_state_snapshot_sha256
    )
    if not baseline_reproduced:
        raise AlfWorldCounterfactualError(
            "causal validation baseline did not reproduce the frozen failure"
        )
    repaired = replay_backend.replay(
        plan=plan,
        intervention=intervention,
        continuation_mode="closed_loop",
    )
    if not (
        repaired.success
        and repaired.pre_state_identity_verified
        and repaired.replacement_admissible
        and not repaired.suffix_diverged
    ):
        raise AlfWorldCounterfactualError(
            "causal validation candidate did not reproduce its repair"
        )
    matched_sham = replay_backend.replay(
        plan=plan,
        intervention=intervention,
        continuation_mode="closed_loop",
        causal_arm="matched_sham",
    )
    reverse_injection = replay_backend.replay(
        plan=plan,
        intervention=intervention,
        continuation_mode="closed_loop",
        causal_arm="reverse_injection",
    )

    def bound(
        outcome: ReplayArmOutcome,
        arm: str,
        *,
        reverse_injection_independent: bool | None = None,
    ) -> ReplayArmOutcome:
        evidence = dict(outcome.outcome_evidence or {})
        evidence.update(
            {
                "causal_arm": arm,
                "causal_binding_sha256": binding.sha256,
            }
        )
        if reverse_injection_independent is not None:
            evidence["reverse_injection_independent"] = reverse_injection_independent
        return replace(outcome, outcome_evidence=evidence)

    # The reverse-injection arm only adds evidence when it executed a trajectory
    # that no other arm executed; a duplicate would double-count one replay.
    other_trajectories = {
        (outcome.executed_actions, outcome.final_state_snapshot_sha256)
        for outcome in (baseline, repaired, matched_sham)
    }
    reverse_injection_independent = (
        reverse_injection.executed_actions,
        reverse_injection.final_state_snapshot_sha256,
    ) not in other_trajectories
    bound_baseline = bound(baseline, "baseline")
    bound_repaired = bound(repaired, "intervention")
    bound_sham = bound(matched_sham, "matched_sham")
    bound_reverse = bound(
        reverse_injection,
        "reverse_injection",
        reverse_injection_independent=reverse_injection_independent,
    )
    verdict = assess_intervention_verified_root_cause(
        binding=binding,
        baseline=bound_baseline,
        intervention=bound_repaired,
        matched_sham=bound_sham,
        reverse_injection=bound_reverse,
    )
    return AlfWorldCausalValidationReceipt(
        binding=binding,
        baseline=bound_baseline,
        intervention=bound_repaired,
        matched_sham=bound_sham,
        reverse_injection=bound_reverse,
        verdict=verdict,
    )


def run_alfworld_single_action_three_arm_validation(
    plan: AlfWorldSingleActionCounterfactualPlan,
    *,
    intervention: SingleActionIntervention,
    replay_backend: AlfWorldBranchReplayBackend,
    controller_identity_sha256: str,
) -> AlfWorldThreeArmCausalValidationReceipt:
    """Run the preregistered baseline, intervention, and matched-sham arms."""

    if intervention not in plan.candidates:
        raise AlfWorldCounterfactualError("causal candidate is not in the frozen plan")
    binding = CausalArmBinding(
        plan_sha256=plan.sha256,
        source_trajectory_sha256=plan.source_trajectory_sha256,
        intervention_id=intervention.intervention_id,
        step_idx=intervention.step_idx,
        source_pre_state_snapshot_sha256=(intervention.source_pre_state_snapshot_sha256),
        source_action=intervention.source_action,
        replacement_action=intervention.replacement_action,
        continuation_mode="closed_loop",
        controller_identity_sha256=controller_identity_sha256,
    )
    baseline = replay_backend.replay(
        plan=plan,
        intervention=None,
        continuation_mode="closed_loop",
    )
    baseline_reproduced = bool(
        not baseline.success
        and baseline.pre_state_identity_verified
        and not baseline.suffix_diverged
        and baseline.executed_actions == tuple(step.source_action for step in plan.steps)
        and baseline.terminal_reason == plan.source_terminal_reason
        and baseline.final_state_snapshot_sha256 == plan.steps[-1].post_state_snapshot_sha256
    )
    if not baseline_reproduced:
        raise AlfWorldCounterfactualError(
            "three-arm validation baseline did not reproduce the frozen failure"
        )
    repaired = replay_backend.replay(
        plan=plan,
        intervention=intervention,
        continuation_mode="closed_loop",
    )
    if not (
        repaired.success
        and repaired.pre_state_identity_verified
        and repaired.replacement_admissible
        and not repaired.suffix_diverged
    ):
        raise AlfWorldCounterfactualError(
            "three-arm validation candidate did not reproduce its repair"
        )
    matched_sham = replay_backend.replay(
        plan=plan,
        intervention=intervention,
        continuation_mode="closed_loop",
        causal_arm="matched_sham",
    )

    def bound(outcome: ReplayArmOutcome, arm: str) -> ReplayArmOutcome:
        evidence = dict(outcome.outcome_evidence or {})
        evidence.update(
            {
                "causal_arm": arm,
                "causal_binding_sha256": binding.sha256,
            }
        )
        return replace(outcome, outcome_evidence=evidence)

    bound_baseline = bound(baseline, "baseline")
    bound_repaired = bound(repaired, "intervention")
    bound_sham = bound(matched_sham, "matched_sham")
    verdict = assess_intervention_verified_sufficient_cause(
        binding=binding,
        baseline=bound_baseline,
        intervention=bound_repaired,
        matched_sham=bound_sham,
    )
    return AlfWorldThreeArmCausalValidationReceipt(
        binding=binding,
        baseline=bound_baseline,
        intervention=bound_repaired,
        matched_sham=bound_sham,
        verdict=verdict,
    )


def _source_action(record: TrainingTrajectoryRecord) -> str:
    action = record.parsed_action.get("command")
    if (
        not isinstance(action, str)
        or not action
        or action != action.strip()
        or "\n" in action
        or record.extra.get("environment_action") != action
    ):
        raise AlfWorldCounterfactualError(
            f"source step {record.step_idx} lacks one canonical semantic action"
        )
    return action


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_file_sha256(path: Path, expected_sha256: str, *, label: str) -> str:
    """Fail closed when a preregistered input file has drifted."""

    _require_sha256(f"expected {label} SHA-256", expected_sha256)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AlfWorldCounterfactualError(f"cannot read {label}: {error}") from error
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise AlfWorldCounterfactualError(f"{label} SHA-256 changed")
    return actual


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise AlfWorldCounterfactualError(f"{name} must be a lowercase SHA-256")
