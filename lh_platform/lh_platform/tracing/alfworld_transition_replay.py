"""Bridge ALFWorld branch replay into transition-candidate causal evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Protocol

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_action_counterfactual import (
    AlfWorldSingleActionCounterfactualPlan,
    AlfWorldSourceStep,
    ReplayArmOutcome,
    SingleActionIntervention,
)
from lh_platform.tracing.alfworld_controlled_fault_injection import (
    ControlledFaultInjectionArmV1,
)
from lh_platform.tracing.replay_receipts import (
    replay_outcome_sha256,
    restore_receipt_sha256,
)
from lh_platform.tracing.types import (
    CounterfactualReplayEvidenceV1,
    ReplayArmIdentityV1,
    ReplayOutcomeV1,
    TraceCaseV1,
    TraceInterventionV1,
)
from lh_platform.training.alfworld_agentic_suite import AlfWorldBranch


def _sha256(value: object) -> str:
    return "sha256:" + canonical_sha256(value)


def _canonical_digest(value: str, *, label: str) -> str:
    digest = value[7:] if value.startswith("sha256:") else value
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} is not a canonical SHA-256")
    return "sha256:" + digest


class AlfWorldCausalReplayBackend(Protocol):
    def replay(
        self,
        *,
        plan: AlfWorldSingleActionCounterfactualPlan,
        intervention: SingleActionIntervention | None,
        continuation_mode: str,
        causal_arm: str = "intervention",
    ) -> ReplayArmOutcome: ...


class ControlledFaultReplayAuthority(Protocol):
    controller_id: str

    def evaluate(
        self,
        case: TraceCaseV1,
        intervention: TraceInterventionV1,
    ) -> CounterfactualReplayEvidenceV1: ...


class AlfWorldControlledFaultReplayAuthority:
    """Execute blind candidates with a frozen failure-prefix/source-suffix controller."""

    def __init__(
        self,
        *,
        failure_arm: ControlledFaultInjectionArmV1,
        source_steps: tuple[AlfWorldSourceStep, ...],
        suffix_controller_sha256: str,
        branch_factory: Callable[[], AlfWorldBranch],
    ) -> None:
        if failure_arm.arm != "injected_failure" or failure_arm.success:
            raise ValueError("controlled-fault replay authority requires one failure arm")
        if not failure_arm.steps or len(source_steps) < len(failure_arm.steps):
            raise ValueError("controlled-fault source controller is shorter than the failure")
        if any(step.step_idx != index for index, step in enumerate(source_steps)):
            raise ValueError("controlled-fault source controller steps are not contiguous")
        if any(
            step.step_idx != index for index, step in enumerate(failure_arm.steps)
        ):
            raise ValueError("controlled-fault failure steps are not contiguous")
        self._failure_arm = failure_arm
        self._source_steps = source_steps
        self._suffix_controller_sha256 = _canonical_digest(
            suffix_controller_sha256,
            label="controlled-fault suffix controller",
        )
        if self._suffix_controller_sha256 != failure_arm.suffix_controller_sha256:
            raise ValueError("controlled-fault suffix controller binding changed")
        self._branch_factory = branch_factory
        self.controller_id = (
            "alfworld-controlled-fault-replay/v1:"
            + self._suffix_controller_sha256.removeprefix("sha256:")
        )

    def _run_arm(
        self,
        *,
        target_step_idx: int,
        replacement_action: str,
        sham_action: str | None,
        arm: str,
    ) -> dict[str, Any]:
        branch = self._branch_factory()
        executed: list[str] = []
        pre_state_verified = True
        treatment_restore_receipt_sha256: str | None = None
        terminal_reason = "budget_exhausted"
        last_step: Any = None
        try:
            current = branch.reset()
            for index, source_step in enumerate(self._source_steps):
                failure_step = (
                    self._failure_arm.steps[index]
                    if index < len(self._failure_arm.steps)
                    else None
                )
                checkpoint = branch.capture_checkpoint()
                if failure_step is not None and (
                    arm == "fault_baseline" or index <= target_step_idx
                ):
                    pre_state_verified = pre_state_verified and (
                        checkpoint.snapshot.state_snapshot_sha256
                        == failure_step.pre_state_snapshot_sha256
                        and tuple(current.admissible_actions)
                        == failure_step.admissible_actions
                    )
                observed_action = (
                    failure_step.action
                    if failure_step is not None
                    else source_step.source_action
                )
                action = observed_action
                if arm == "predicted_repair" and index == target_step_idx:
                    action = replacement_action
                elif arm == "matched_sham" and index == target_step_idx:
                    if sham_action is None:
                        raise ValueError("matched sham action is not constructible")
                    action = sham_action
                action_admissible = action in current.admissible_actions
                treatment_step = index == target_step_idx and arm in {
                    "predicted_repair",
                    "matched_sham",
                }
                continuation_requires_admissible = (
                    failure_step is None
                    or (arm != "fault_baseline" and index > target_step_idx)
                    or treatment_step
                )
                if not action_admissible and continuation_requires_admissible:
                    terminal_reason = (
                        "treatment_action_inadmissible"
                        if treatment_step
                        else "suffix_action_inadmissible"
                    )
                    break
                if index == target_step_idx and arm in {
                    "predicted_repair",
                    "matched_sham",
                }:
                    last_step = branch.step(action)
                    treatment_restore_receipt_sha256 = _canonical_digest(
                        str(last_step.restore_receipt_sha256),
                        label=f"controlled-fault {arm} restore receipt",
                    )
                else:
                    direct_step = getattr(branch, "step_unverified", None)
                    last_step = (
                        direct_step(action)
                        if callable(direct_step)
                        else branch.step(action)
                    )
                executed.append(action)
                current = last_step.observation
                if last_step.won:
                    terminal_reason = "environment_won"
                    break
                if last_step.done:
                    terminal_reason = "environment_done"
                    break
            final_checkpoint = branch.capture_checkpoint()
            return {
                "arm": arm,
                "success": bool(last_step is not None and last_step.won),
                "terminal_reason": terminal_reason,
                "pre_state_identity_verified": pre_state_verified,
                "treatment_restore_receipt_sha256": (
                    treatment_restore_receipt_sha256
                ),
                "executed_actions": tuple(executed),
                "final_state_snapshot_sha256": _canonical_digest(
                    final_checkpoint.snapshot.state_snapshot_sha256,
                    label=f"controlled-fault {arm} final state",
                ),
                "physical_call_counts": dict(branch.physical_call_counters()),
            }
        finally:
            branch.close()

    def evaluate(
        self,
        case: TraceCaseV1,
        intervention: TraceInterventionV1,
    ) -> CounterfactualReplayEvidenceV1:
        sequence_by_id = {event.event_id: event.sequence for event in case.events}
        try:
            target_step_idx = sequence_by_id[intervention.target_event_id]
            failure_step = self._failure_arm.steps[target_step_idx]
        except (IndexError, KeyError) as error:
            raise ValueError("controlled-fault replay target is outside the failure") from error
        # The matched sham must be a placebo rather than a second baseline replay:
        # use the first frozen-menu action that is neither the fault action nor the
        # proposed repair, and fail closed when the menu cannot supply one.
        sham_action = next(
            (
                candidate
                for candidate in failure_step.admissible_actions
                if candidate != failure_step.action
                and candidate != intervention.replacement_content
            ),
            None,
        )
        if sham_action is None:
            raise ValueError("matched sham action is not constructible from the frozen menu")
        raw_outcomes = {
            "fault_baseline": self._run_arm(
                target_step_idx=target_step_idx,
                replacement_action=intervention.replacement_content,
                sham_action=None,
                arm="fault_baseline",
            ),
            "predicted_repair": self._run_arm(
                target_step_idx=target_step_idx,
                replacement_action=intervention.replacement_content,
                sham_action=None,
                arm="predicted_repair",
            ),
            "matched_sham": self._run_arm(
                target_step_idx=target_step_idx,
                replacement_action=intervention.replacement_content,
                sham_action=sham_action,
                arm="matched_sham",
            ),
        }
        baseline = raw_outcomes["fault_baseline"]
        expected_failure_actions = self._failure_arm.executed_actions
        expected_final_state = _canonical_digest(
                self._failure_arm.final_state_snapshot_sha256,
                label="controlled-fault frozen failure final state",
            )
        baseline_mismatches = tuple(
            name
            for name, mismatch in {
                "unexpected_success": baseline["success"],
                "pre_state_identity": not baseline["pre_state_identity_verified"],
                "executed_actions": (
                    baseline["executed_actions"] != expected_failure_actions
                ),
                "final_state": (
                    baseline["final_state_snapshot_sha256"] != expected_final_state
                ),
            }.items()
            if mismatch
        )
        if baseline_mismatches:
            raise ValueError(
                "controlled-fault baseline did not reproduce the failure: "
                + ", ".join(baseline_mismatches)
            )

        source_snapshot_sha256 = _canonical_digest(
            failure_step.pre_state_snapshot_sha256,
            label="controlled-fault source snapshot",
        )
        target_actions = {
            "fault_baseline": failure_step.action,
            "predicted_repair": intervention.replacement_content,
            "matched_sham": sham_action,
        }
        outcomes: dict[str, ReplayOutcomeV1] = {}
        identities: list[ReplayArmIdentityV1] = []
        baseline_protocol = _sha256(
            {
                "mode": "blind_failure_replay",
                "failure_trajectory_sha256": self._failure_arm.trajectory_sha256,
            }
        )
        suffix_seed = _sha256(
            {
                "mode": "deterministic_source_suffix",
                "suffix_controller_sha256": self._suffix_controller_sha256,
            }
        )
        for arm, raw in raw_outcomes.items():
            restored = (
                source_snapshot_sha256
                if raw["pre_state_identity_verified"]
                else _sha256(
                    {
                        "arm": arm,
                        "restore_failed": True,
                        "source_snapshot_sha256": source_snapshot_sha256,
                    }
                )
            )
            unhashed = ReplayOutcomeV1(
                success=bool(raw["success"]),
                terminal_reason=(
                    "verifier_accepted" if raw["success"] else "verifier_rejected"
                ),
                target_event_id=intervention.target_event_id,
                input_content_sha256=_sha256(target_actions[arm]),
                restored_state_sha256=restored,
                terminal_state_sha256=str(raw["final_state_snapshot_sha256"]),
                outcome_receipt_sha256="",
            )
            outcomes[arm] = replace(
                unhashed,
                outcome_receipt_sha256=replay_outcome_sha256(
                    controller_id=self.controller_id,
                    case_id=case.case_id,
                    arm=arm,
                    source_snapshot_sha256=source_snapshot_sha256,
                    outcome=unhashed,
                ),
            )
            restore_protocol_receipt = raw[
                "treatment_restore_receipt_sha256"
            ] or _sha256(
                {
                    "arm": arm,
                    "failure_trajectory_sha256": self._failure_arm.trajectory_sha256,
                    "source_snapshot_sha256": source_snapshot_sha256,
                    "pre_state_identity_verified": raw[
                        "pre_state_identity_verified"
                    ],
                }
            )
            identities.append(
                ReplayArmIdentityV1(
                    arm=arm,
                    target_event_id=intervention.target_event_id,
                    target_action_sha256=_sha256(target_actions[arm]),
                    continuation_policy_sha256=(
                        baseline_protocol
                        if arm == "fault_baseline"
                        else self._suffix_controller_sha256
                    ),
                    seed_schedule_sha256=(
                        baseline_protocol if arm == "fault_baseline" else suffix_seed
                    ),
                    restore_protocol_receipt_sha256=str(
                        restore_protocol_receipt
                    ),
                    outcome_evidence_sha256=_sha256(raw),
                )
            )
        preliminary = CounterfactualReplayEvidenceV1(
            controller_id=self.controller_id,
            source_trace_sha256=case.sha256,
            source_snapshot_sha256=source_snapshot_sha256,
            restore_verified=all(
                bool(raw["pre_state_identity_verified"])
                for raw in raw_outcomes.values()
            ),
            restore_receipt_sha256="sha256:" + "0" * 64,
            baseline=outcomes["fault_baseline"],
            repair=outcomes["predicted_repair"],
            sham=outcomes["matched_sham"],
            replay_calls=3,
            arm_identities=tuple(identities),
        )
        return replace(
            preliminary,
            restore_receipt_sha256=restore_receipt_sha256(
                case=case,
                evidence=preliminary,
            ),
        )


class AlfWorldControlledFaultTransitionReplayBackend:
    """Keep blind candidate enumeration separate from privileged replay authority."""

    def __init__(
        self,
        *,
        case: TraceCaseV1,
        failure_arm: ControlledFaultInjectionArmV1,
        replay_authority: ControlledFaultReplayAuthority,
    ) -> None:
        if failure_arm.arm != "injected_failure" or failure_arm.success:
            raise ValueError("controlled-fault transition backend requires one failure arm")
        if not replay_authority.controller_id:
            raise ValueError("controlled-fault replay authority identity is empty")
        if len(case.events) != len(failure_arm.steps):
            raise ValueError("controlled-fault failure arm does not cover the trace")
        failure_trajectory_sha256 = failure_arm.trajectory_sha256
        for event, step in zip(case.events, failure_arm.steps, strict=True):
            metadata = dict(event.metadata)
            if (
                event.sequence != step.step_idx
                or metadata.get("actor_action") != step.action
                or metadata.get("state_snapshot_sha256")
                != step.pre_state_snapshot_sha256
                or metadata.get("source_trajectory_sha256")
                != failure_trajectory_sha256
            ):
                raise ValueError("controlled-fault blind trace binding changed")
        self._source_trace_sha256 = case.sha256
        self._steps_by_event_id = {
            event.event_id: step
            for event, step in zip(case.events, failure_arm.steps, strict=True)
        }
        self._replay_authority = replay_authority
        self.controller_id = replay_authority.controller_id

    def interventions_for_event(
        self,
        case: TraceCaseV1,
        event_id: str,
    ) -> tuple[TraceInterventionV1, ...]:
        if case.sha256 != self._source_trace_sha256:
            raise ValueError("controlled-fault candidate request changed the trace")
        try:
            step = self._steps_by_event_id[event_id]
        except KeyError as error:
            raise ValueError("controlled-fault candidate event is outside the trace") from error
        replacements = tuple(
            dict.fromkeys(
                action
                for action in step.admissible_actions
                if action != step.action
            )
        )
        return tuple(
            TraceInterventionV1(
                target_event_id=event_id,
                replacement_content=action,
            )
            for action in replacements
        )

    def evaluate(
        self,
        case: TraceCaseV1,
        intervention: TraceInterventionV1,
    ) -> CounterfactualReplayEvidenceV1:
        allowed = self.interventions_for_event(case, intervention.target_event_id)
        if intervention not in allowed:
            raise ValueError("controlled-fault intervention is not a blind candidate")
        evidence = self._replay_authority.evaluate(case, intervention)
        if evidence.controller_id != self.controller_id:
            raise ValueError("controlled-fault replay authority identity changed")
        return evidence


class AlfWorldTransitionCandidateReplayBackend:
    """Enumerate frozen ALFWorld actions and retain continuation-level arm identity."""

    def __init__(
        self,
        *,
        plan: AlfWorldSingleActionCounterfactualPlan,
        replay_backend: AlfWorldCausalReplayBackend,
        controller_identity_sha256: str,
    ) -> None:
        self._plan = plan
        self._replay_backend = replay_backend
        self._controller_identity_sha256 = _canonical_digest(
            controller_identity_sha256,
            label="ALFWorld controller identity",
        )
        self.controller_id = (
            "alfworld-transition-candidate/v1:"
            + self._controller_identity_sha256.removeprefix("sha256:")
        )

    def _validate_case_binding(self, case: TraceCaseV1) -> None:
        source_hashes = {
            dict(event.metadata).get("source_trajectory_sha256", "")
            for event in case.events
        }
        if source_hashes != {self._plan.source_trajectory_sha256}:
            raise ValueError("ALFWorld transition plan is not bound to the trace")
        if len(case.events) != len(self._plan.steps) or any(
            event.sequence != step.step_idx
            for event, step in zip(case.events, self._plan.steps, strict=True)
        ):
            raise ValueError("ALFWorld transition plan event identity changed")

    def interventions_for_event(
        self,
        case: TraceCaseV1,
        event_id: str,
    ) -> tuple[TraceInterventionV1, ...]:
        self._validate_case_binding(case)
        sequence_by_id = {event.event_id: event.sequence for event in case.events}
        if event_id not in sequence_by_id:
            raise ValueError("ALFWorld transition candidate is outside the trace")
        step_idx = sequence_by_id[event_id]
        return tuple(
            TraceInterventionV1(
                target_event_id=event_id,
                replacement_content=candidate.replacement_action,
            )
            for candidate in self._plan.candidates
            if candidate.step_idx == step_idx
        )

    def evaluate(
        self,
        case: TraceCaseV1,
        intervention: TraceInterventionV1,
    ) -> CounterfactualReplayEvidenceV1:
        self._validate_case_binding(case)
        sequence_by_id = {event.event_id: event.sequence for event in case.events}
        try:
            step_idx = sequence_by_id[intervention.target_event_id]
        except KeyError as error:
            raise ValueError("ALFWorld intervention target is outside the trace") from error
        matches = tuple(
            candidate
            for candidate in self._plan.candidates
            if candidate.step_idx == step_idx
            and candidate.replacement_action == intervention.replacement_content
        )
        if len(matches) != 1:
            raise ValueError("ALFWorld intervention is not one frozen plan candidate")
        candidate = matches[0]
        baseline = self._replay_backend.replay(
            plan=self._plan,
            intervention=None,
            continuation_mode="closed_loop",
        )
        repair = self._replay_backend.replay(
            plan=self._plan,
            intervention=candidate,
            continuation_mode="closed_loop",
        )
        sham = self._replay_backend.replay(
            plan=self._plan,
            intervention=candidate,
            continuation_mode="closed_loop",
            causal_arm="matched_sham",
        )
        expected_source_actions = tuple(step.source_action for step in self._plan.steps)
        baseline_exact = (
            not baseline.success
            and baseline.pre_state_identity_verified
            and not baseline.suffix_diverged
            and baseline.executed_actions == expected_source_actions
            and baseline.terminal_reason == self._plan.source_terminal_reason
            and baseline.final_state_snapshot_sha256
            == self._plan.steps[-1].post_state_snapshot_sha256
        )
        if not baseline_exact:
            raise ValueError("ALFWorld baseline did not reproduce the frozen failure")

        source_snapshot_sha256 = _canonical_digest(
            candidate.source_pre_state_snapshot_sha256,
            label="ALFWorld source snapshot",
        )
        raw_outcomes = {
            "fault_baseline": baseline,
            "predicted_repair": repair,
            "matched_sham": sham,
        }
        # The matched sham uses the first frozen-menu action that is neither the
        # fault action nor the proposed repair, so the arm cannot degenerate into
        # a second baseline replay.  Fail closed when the menu cannot supply one.
        sham_action = next(
            (
                action
                for action in self._plan.steps[step_idx].admissible_actions
                if action != candidate.source_action
                and action != candidate.replacement_action
            ),
            None,
        )
        if sham_action is None:
            raise ValueError("matched sham action is not constructible from the frozen menu")
        target_actions = {
            "fault_baseline": candidate.source_action,
            "predicted_repair": candidate.replacement_action,
            "matched_sham": sham_action,
        }
        outcomes: dict[str, ReplayOutcomeV1] = {}
        identities: list[ReplayArmIdentityV1] = []
        source_continuation = _sha256(
            {
                "mode": "frozen_source_suffix_replay",
                "actor_protocol_id": self._plan.actor_protocol_id,
                "plan_sha256": self._plan.sha256,
            }
        )
        frozen_continuation = self._controller_identity_sha256
        source_seed = _sha256(
            {
                "mode": "source_schedule",
                "base": self._plan.sampling_seed_base,
                "schedule": self._plan.sampling_seed_schedule,
                "temperature": self._plan.sampling_temperature,
            }
        )
        frozen_seed = _sha256(
            {
                "mode": "counterfactual_closed_loop_schedule",
                "base": self._plan.sampling_seed_base,
                "schedule": self._plan.sampling_seed_schedule,
                "temperature": self._plan.sampling_temperature,
                "intervention_id": candidate.intervention_id,
            }
        )
        for arm, raw in raw_outcomes.items():
            restored = (
                source_snapshot_sha256
                if raw.pre_state_identity_verified
                else _sha256(
                    {
                        "arm": arm,
                        "restore_failed": True,
                        "source_snapshot_sha256": source_snapshot_sha256,
                    }
                )
            )
            unhashed = ReplayOutcomeV1(
                success=raw.success,
                terminal_reason=(
                    "verifier_accepted" if raw.success else "verifier_rejected"
                ),
                target_event_id=intervention.target_event_id,
                input_content_sha256=_sha256(target_actions[arm]),
                restored_state_sha256=restored,
                terminal_state_sha256=_canonical_digest(
                    raw.final_state_snapshot_sha256,
                    label=f"ALFWorld {arm} terminal state",
                ),
                outcome_receipt_sha256="",
            )
            outcomes[arm] = replace(
                unhashed,
                outcome_receipt_sha256=replay_outcome_sha256(
                    controller_id=self.controller_id,
                    case_id=case.case_id,
                    arm=arm,
                    source_snapshot_sha256=source_snapshot_sha256,
                    outcome=unhashed,
                ),
            )
            raw_evidence = raw.outcome_evidence
            if not isinstance(raw_evidence, Mapping):
                raise ValueError(f"ALFWorld {arm} outcome evidence is missing")
            if arm == "fault_baseline":
                restore_protocol_receipt = _sha256(
                    {
                        "arm": arm,
                        "plan_sha256": self._plan.sha256,
                        "source_snapshot_sha256": source_snapshot_sha256,
                        "pre_state_identity_verified": raw.pre_state_identity_verified,
                        "executed_actions": list(raw.executed_actions),
                    }
                )
            else:
                restore_protocol_receipt = _canonical_digest(
                    str(raw_evidence.get("intervention_restore_receipt_sha256", "")),
                    label=f"ALFWorld {arm} restore receipt",
                )
            identities.append(
                ReplayArmIdentityV1(
                    arm=arm,
                    target_event_id=intervention.target_event_id,
                    target_action_sha256=_sha256(target_actions[arm]),
                    continuation_policy_sha256=(
                        source_continuation
                        if arm == "fault_baseline"
                        else frozen_continuation
                    ),
                    seed_schedule_sha256=(
                        source_seed if arm == "fault_baseline" else frozen_seed
                    ),
                    restore_protocol_receipt_sha256=restore_protocol_receipt,
                    outcome_evidence_sha256=_sha256(raw.to_dict()),
                )
            )

        preliminary = CounterfactualReplayEvidenceV1(
            controller_id=self.controller_id,
            source_trace_sha256=case.sha256,
            source_snapshot_sha256=source_snapshot_sha256,
            restore_verified=all(
                raw.pre_state_identity_verified for raw in raw_outcomes.values()
            ),
            restore_receipt_sha256="sha256:" + "0" * 64,
            baseline=outcomes["fault_baseline"],
            repair=outcomes["predicted_repair"],
            sham=outcomes["matched_sham"],
            replay_calls=3,
            arm_identities=tuple(identities),
        )
        return replace(
            preliminary,
            restore_receipt_sha256=restore_receipt_sha256(
                case=case,
                evidence=preliminary,
            ),
        )


__all__ = [
    "AlfWorldControlledFaultReplayAuthority",
    "AlfWorldControlledFaultTransitionReplayBackend",
    "AlfWorldTransitionCandidateReplayBackend",
]
