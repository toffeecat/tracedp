"""Controlled single-action fault injection over a successful ALFWorld trace."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_action_counterfactual import (
    AlfWorldSourceStep,
    build_alfworld_source_trajectory,
)
from lh_platform.tracing.types import (
    TraceCaseV1,
    TraceContextV1,
    TraceEventV1,
    TraceOutcomeV1,
)
from lh_platform.training.alfworld_agentic_suite import (
    AlfWorldBranch,
    AlfWorldBranchObservation,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord

CONTROLLED_FAULT_INJECTION_PLAN_SCHEMA = "lh_alfworld_controlled_fault_injection_plan/v1"
SOURCE_SUFFIX_CONTROLLER_SCHEMA = "lh_alfworld_source_open_loop_suffix/v1"
CONTROLLED_FAULT_INJECTION_CAUSAL_SCHEMA = (
    "lh_alfworld_controlled_fault_injection_causal_validation/v1"
)
CONTROLLED_FAULT_INJECTION_CLAIM_SCOPE = "controlled_single_action_sufficient_failure_cause"

FaultInjectionArmName = Literal[
    "baseline",
    "injected_failure",
    "matched_control",
]


class ControlledFaultInjectionError(RuntimeError):
    """Raised when a controlled injection cannot support the frozen protocol."""


def _sha256(value: object) -> str:
    return "sha256:" + canonical_sha256(value)


def _require_raw_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ControlledFaultInjectionError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class ControlledActionInjectionV1:
    injection_id: str
    step_idx: int
    source_action: str
    injected_action: str
    source_pre_state_snapshot_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlfWorldControlledFaultInjectionPlanV1:
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
    source_trajectory_sha256: str
    source_terminal_reason: str
    steps: tuple[AlfWorldSourceStep, ...]
    injection: ControlledActionInjectionV1
    suffix_controller_sha256: str
    schema: str = CONTROLLED_FAULT_INJECTION_PLAN_SCHEMA

    @property
    def source_trajectory_success(self) -> bool:
        return True

    @property
    def source_trajectory_step_count(self) -> int:
        return len(self.steps)

    @property
    def post_injection_source_action_count(self) -> int:
        return len(self.steps) - self.injection.step_idx - 1

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

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
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "source_terminal_reason": self.source_terminal_reason,
            "source_trajectory_success": self.source_trajectory_success,
            "source_trajectory_step_count": self.source_trajectory_step_count,
            "post_injection_source_action_count": (self.post_injection_source_action_count),
            "steps": [asdict(step) for step in self.steps],
            "injection": self.injection.to_dict(),
            "suffix_controller_sha256": self.suffix_controller_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> AlfWorldControlledFaultInjectionPlanV1:
        """Rebuild a persisted plan through its typed contract."""

        if payload.get("schema") != CONTROLLED_FAULT_INJECTION_PLAN_SCHEMA:
            raise ControlledFaultInjectionError("persisted controlled-fault plan schema is invalid")
        raw_steps = payload.get("steps")
        raw_injection = payload.get("injection")
        if not isinstance(raw_steps, list) or not isinstance(raw_injection, Mapping):
            raise ControlledFaultInjectionError("persisted controlled-fault plan is incomplete")
        try:
            steps = tuple(
                AlfWorldSourceStep(
                    step_idx=int(step["step_idx"]),
                    source_action=str(step["source_action"]),
                    admissible_actions=tuple(step["admissible_actions"]),
                    source_action_admissible=bool(step["source_action_admissible"]),
                    state_snapshot_sha256=str(step["state_snapshot_sha256"]),
                    state_snapshot_artifact_path=str(step["state_snapshot_artifact_path"]),
                    post_state_snapshot_sha256=str(step["post_state_snapshot_sha256"]),
                    post_state_snapshot_artifact_path=str(
                        step["post_state_snapshot_artifact_path"]
                    ),
                    observation=str(step["observation"]),
                    next_observation=str(step["next_observation"]),
                    sampling_seed=int(step["sampling_seed"]),
                )
                for step in raw_steps
                if isinstance(step, Mapping)
            )
            injection = ControlledActionInjectionV1(
                injection_id=str(raw_injection["injection_id"]),
                step_idx=int(raw_injection["step_idx"]),
                source_action=str(raw_injection["source_action"]),
                injected_action=str(raw_injection["injected_action"]),
                source_pre_state_snapshot_sha256=str(
                    raw_injection["source_pre_state_snapshot_sha256"]
                ),
            )
            plan = cls(
                task_id=str(payload["task_id"]),
                traj_uid=str(payload["traj_uid"]),
                split=str(payload["split"]),
                environment_seed=int(payload["environment_seed"]),
                task_description=str(payload["task_description"]),
                actor_protocol_id=str(payload["actor_protocol_id"]),
                model_id=str(payload["model_id"]),
                model_io_profile_sha256=str(payload["model_io_profile_sha256"]),
                response_contract_sha256=str(payload["response_contract_sha256"]),
                task_manifest_lock_sha256=str(payload["task_manifest_lock_sha256"]),
                chat_template_sha256=str(payload["chat_template_sha256"]),
                sampling_seed_base=int(payload["sampling_seed_base"]),
                sampling_seed_schedule=str(payload["sampling_seed_schedule"]),
                sampling_temperature=float(payload["sampling_temperature"]),
                source_trajectory_sha256=str(payload["source_trajectory_sha256"]),
                source_terminal_reason=str(payload["source_terminal_reason"]),
                steps=steps,
                injection=injection,
                suffix_controller_sha256=str(payload["suffix_controller_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ControlledFaultInjectionError(
                f"persisted controlled-fault plan is invalid: {error}"
            ) from error
        if len(steps) != len(raw_steps) or tuple(step.step_idx for step in steps) != tuple(
            range(len(steps))
        ):
            raise ControlledFaultInjectionError("persisted controlled-fault plan steps are invalid")
        if injection.step_idx >= len(steps):
            raise ControlledFaultInjectionError(
                "persisted controlled-fault injection is outside the plan"
            )
        return plan


@dataclass(frozen=True)
class CapturedFaultInjectionStepV1:
    step_idx: int
    action: str
    action_admissible: bool
    observation_before: str
    observation_after: str
    admissible_actions: tuple[str, ...]
    pre_state_snapshot_sha256: str
    post_state_snapshot_sha256: str

    def __post_init__(self) -> None:
        if self.step_idx < 0 or not self.action or not self.admissible_actions:
            raise ControlledFaultInjectionError("captured fault-injection step is invalid")
        if not isinstance(self.action_admissible, bool):
            raise ControlledFaultInjectionError("captured action admissibility must be explicit")
        if self.action_admissible != (self.action in self.admissible_actions):
            raise ControlledFaultInjectionError(
                "captured action admissibility contradicts the live menu"
            )
        _require_raw_sha256("captured pre-state SHA-256", self.pre_state_snapshot_sha256)
        _require_raw_sha256("captured post-state SHA-256", self.post_state_snapshot_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "action": self.action,
            "action_admissible": self.action_admissible,
            "observation_before": self.observation_before,
            "observation_after": self.observation_after,
            "admissible_actions": list(self.admissible_actions),
            "pre_state_snapshot_sha256": self.pre_state_snapshot_sha256,
            "post_state_snapshot_sha256": self.post_state_snapshot_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CapturedFaultInjectionStepV1:
        try:
            raw_actions = payload["admissible_actions"]
            if not isinstance(raw_actions, list) or not all(
                isinstance(action, str) for action in raw_actions
            ):
                raise TypeError("admissible_actions must be a string list")
            action_admissible = payload["action_admissible"]
            if not isinstance(action_admissible, bool):
                raise TypeError("action_admissible must be boolean")
            return cls(
                step_idx=int(payload["step_idx"]),
                action=str(payload["action"]),
                action_admissible=action_admissible,
                observation_before=str(payload["observation_before"]),
                observation_after=str(payload["observation_after"]),
                admissible_actions=tuple(raw_actions),
                pre_state_snapshot_sha256=str(payload["pre_state_snapshot_sha256"]),
                post_state_snapshot_sha256=str(payload["post_state_snapshot_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ControlledFaultInjectionError(
                f"persisted controlled-fault step is invalid: {error}"
            ) from error


@dataclass(frozen=True)
class ControlledFaultInjectionArmV1:
    arm: FaultInjectionArmName
    success: bool
    terminal_reason: str
    pre_state_identity_verified: bool
    treatment_action_admissible: bool
    treatment_execution_mode: str | None
    treatment_restore_receipt_sha256: str | None
    suffix_controller_sha256: str
    steps: tuple[CapturedFaultInjectionStepV1, ...]
    final_state_snapshot_sha256: str
    physical_call_counts: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.arm not in {"baseline", "injected_failure", "matched_control"}:
            raise ControlledFaultInjectionError("fault-injection arm name is invalid")
        if not self.terminal_reason or not self.steps:
            raise ControlledFaultInjectionError("fault-injection arm is incomplete")
        if tuple(step.step_idx for step in self.steps) != tuple(range(len(self.steps))):
            raise ControlledFaultInjectionError(
                "fault-injection arm steps must be contiguous from zero"
            )
        if not self.suffix_controller_sha256.startswith("sha256:"):
            raise ControlledFaultInjectionError("suffix controller identity is invalid")
        _require_raw_sha256("fault-injection final-state SHA-256", self.final_state_snapshot_sha256)
        if self.treatment_restore_receipt_sha256 is not None:
            _require_raw_sha256(
                "treatment restore receipt SHA-256",
                self.treatment_restore_receipt_sha256,
            )

    @property
    def executed_actions(self) -> tuple[str, ...]:
        return tuple(step.action for step in self.steps)

    @property
    def trajectory_sha256(self) -> str:
        return canonical_sha256([step.to_dict() for step in self.steps])

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "success": self.success,
            "terminal_reason": self.terminal_reason,
            "pre_state_identity_verified": self.pre_state_identity_verified,
            "treatment_action_admissible": self.treatment_action_admissible,
            "treatment_execution_mode": self.treatment_execution_mode,
            "treatment_restore_receipt_sha256": (self.treatment_restore_receipt_sha256),
            "suffix_controller_sha256": self.suffix_controller_sha256,
            "steps": [step.to_dict() for step in self.steps],
            "final_state_snapshot_sha256": self.final_state_snapshot_sha256,
            "physical_call_counts": dict(self.physical_call_counts),
            "trajectory_sha256": self.trajectory_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControlledFaultInjectionArmV1:
        raw_steps = payload.get("steps")
        raw_counts = payload.get("physical_call_counts")
        if not isinstance(raw_steps, list) or not isinstance(raw_counts, Mapping):
            raise ControlledFaultInjectionError("persisted controlled-fault arm is incomplete")
        try:
            pre_state_verified = payload["pre_state_identity_verified"]
            treatment_admissible = payload["treatment_action_admissible"]
            success = payload["success"]
            if not all(
                isinstance(value, bool)
                for value in (pre_state_verified, treatment_admissible, success)
            ):
                raise TypeError("arm boolean fields must be boolean")
            arm = cls(
                arm=str(payload["arm"]),  # type: ignore[arg-type]
                success=success,
                terminal_reason=str(payload["terminal_reason"]),
                pre_state_identity_verified=pre_state_verified,
                treatment_action_admissible=treatment_admissible,
                treatment_execution_mode=(
                    None
                    if payload["treatment_execution_mode"] is None
                    else str(payload["treatment_execution_mode"])
                ),
                treatment_restore_receipt_sha256=(
                    None
                    if payload["treatment_restore_receipt_sha256"] is None
                    else str(payload["treatment_restore_receipt_sha256"])
                ),
                suffix_controller_sha256=str(payload["suffix_controller_sha256"]),
                steps=tuple(
                    CapturedFaultInjectionStepV1.from_dict(step)
                    for step in raw_steps
                    if isinstance(step, Mapping)
                ),
                final_state_snapshot_sha256=str(payload["final_state_snapshot_sha256"]),
                physical_call_counts=dict(raw_counts),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ControlledFaultInjectionError(
                f"persisted controlled-fault arm is invalid: {error}"
            ) from error
        if len(arm.steps) != len(raw_steps):
            raise ControlledFaultInjectionError("persisted controlled-fault arm steps are invalid")
        expected = payload.get("trajectory_sha256")
        if expected is not None and expected != arm.trajectory_sha256:
            raise ControlledFaultInjectionError(
                "persisted controlled-fault arm trajectory SHA-256 changed"
            )
        return arm


@dataclass(frozen=True)
class ControlledFaultInjectionCausalBindingV1:
    source_trajectory_sha256: str
    failure_trajectory_sha256: str
    source_trace_sha256: str
    injection_step_idx: int
    source_action: str
    injected_action: str
    repair_action: str
    source_pre_state_snapshot_sha256: str
    suffix_controller_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledFaultInjectionCausalReceiptV1:
    binding: ControlledFaultInjectionCausalBindingV1
    baseline: ControlledFaultInjectionArmV1
    injected_failure: ControlledFaultInjectionArmV1
    matched_control: ControlledFaultInjectionArmV1
    causal_gold_admitted: bool
    missing_requirements: tuple[str, ...]
    claim_scope: str = CONTROLLED_FAULT_INJECTION_CLAIM_SCOPE
    root_cause_verified: bool = False
    schema: str = CONTROLLED_FAULT_INJECTION_CAUSAL_SCHEMA

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": self.binding.to_dict(),
            "baseline": self.baseline.to_dict(),
            "injected_failure": self.injected_failure.to_dict(),
            "matched_control": self.matched_control.to_dict(),
            "causal_gold_admitted": self.causal_gold_admitted,
            "sufficient_failure_cause_verified": self.causal_gold_admitted,
            "root_cause_verified": self.root_cause_verified,
            "missing_requirements": list(self.missing_requirements),
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True)
class ControlledFaultInjectionArtifactsV1:
    trace_case: TraceCaseV1
    causal_receipt: ControlledFaultInjectionCausalReceiptV1


@dataclass(frozen=True)
class ControlledFaultInjectionArmsV1:
    baseline: ControlledFaultInjectionArmV1
    injected_failure: ControlledFaultInjectionArmV1
    matched_control: ControlledFaultInjectionArmV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "injected_failure": self.injected_failure.to_dict(),
            "matched_control": self.matched_control.to_dict(),
        }


def _build_blind_failure_case(
    plan: AlfWorldControlledFaultInjectionPlanV1,
    injected_failure: ControlledFaultInjectionArmV1,
) -> TraceCaseV1:
    events: list[TraceEventV1] = []
    parent: str | None = None
    for step in injected_failure.steps:
        event_id = f"step-{step.step_idx:04d}"
        events.append(
            TraceEventV1(
                event_id=event_id,
                sequence=step.step_idx,
                kind="environment_action",
                content=(
                    f"Step {step.step_idx}\n"
                    f"Observation before action: {step.observation_before}\n"
                    f"Admissible actions: "
                    f"{json.dumps(step.admissible_actions, ensure_ascii=False)}\n"
                    f"Actor response: <action>{step.action}</action>\n"
                    f"Environment action: {step.action}\n"
                    f"Observation after action: {step.observation_after}"
                ),
                parent_event_ids=() if parent is None else (parent,),
                metadata=tuple(
                    sorted(
                        {
                            "actor_action": step.action,
                            "actor_reasoning": "",
                            "observation_after": step.observation_after,
                            "observation_before": step.observation_before,
                            "source_trajectory_sha256": (injected_failure.trajectory_sha256),
                            "state_snapshot_sha256": (step.pre_state_snapshot_sha256),
                        }.items()
                    )
                ),
            )
        )
        parent = event_id
    initial_state = json.dumps(
        {
            "visible_observation": injected_failure.steps[0].observation_before,
            "state_snapshot_sha256": (injected_failure.steps[0].pre_state_snapshot_sha256),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    evaluator_failure = json.dumps(
        {
            "benchmark_won": False,
            "terminal_reason": injected_failure.terminal_reason,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return TraceCaseV1(
        case_id=(
            f"alfworld-controlled-fault:{plan.task_id}:{injected_failure.trajectory_sha256[:16]}"
        ),
        task_id=plan.task_id,
        events=tuple(events),
        interruption_event_id=events[-1].event_id,
        observed_outcome=TraceOutcomeV1(
            success=False,
            terminal_reason=injected_failure.terminal_reason,
            score=0.0,
        ),
        context=TraceContextV1(
            task_specification=plan.task_description,
            initial_state=initial_state,
            evaluator_failure=evaluator_failure,
        ),
    )


def build_alfworld_controlled_fault_injection_artifacts(
    plan: AlfWorldControlledFaultInjectionPlanV1,
    *,
    baseline: ControlledFaultInjectionArmV1,
    injected_failure: ControlledFaultInjectionArmV1,
    matched_control: ControlledFaultInjectionArmV1,
    minimum_post_injection_actions: int,
) -> ControlledFaultInjectionArtifactsV1:
    """Admit causal gold and build a label-free TraceCase through one seam."""

    if minimum_post_injection_actions < 0:
        raise ControlledFaultInjectionError(
            "minimum post-injection action count must be nonnegative"
        )
    injection = plan.injection
    source_actions = tuple(step.source_action for step in plan.steps)
    source_final_state = plan.steps[-1].post_state_snapshot_sha256

    def treatment_verified(
        arm: ControlledFaultInjectionArmV1,
        *,
        expected_action: str,
    ) -> bool:
        return bool(
            len(arm.steps) > injection.step_idx
            and arm.steps[injection.step_idx].action == expected_action
            and arm.steps[injection.step_idx].pre_state_snapshot_sha256
            == injection.source_pre_state_snapshot_sha256
            and arm.treatment_action_admissible
            and arm.treatment_execution_mode == "verified_branch_step"
            and arm.treatment_restore_receipt_sha256
        )

    post_injection_actions = sum(
        step.step_idx > injection.step_idx for step in injected_failure.steps
    )
    checks = {
        "arm_names": (
            baseline.arm == "baseline"
            and injected_failure.arm == "injected_failure"
            and matched_control.arm == "matched_control"
        ),
        "controller_binding": all(
            arm.suffix_controller_sha256 == plan.suffix_controller_sha256
            for arm in (baseline, injected_failure, matched_control)
        ),
        "pre_state_identity": all(
            arm.pre_state_identity_verified for arm in (baseline, injected_failure, matched_control)
        ),
        "baseline_success": (
            baseline.success
            and baseline.executed_actions == source_actions
            and baseline.final_state_snapshot_sha256 == source_final_state
        ),
        "injected_failure": (
            not injected_failure.success
            and injected_failure.executed_actions[: injection.step_idx]
            == source_actions[: injection.step_idx]
            and treatment_verified(
                injected_failure,
                expected_action=injection.injected_action,
            )
        ),
        "post_injection_depth": (post_injection_actions >= minimum_post_injection_actions),
        "matched_control_post_state": (
            len(matched_control.steps) > injection.step_idx
            and matched_control.steps[injection.step_idx].post_state_snapshot_sha256
            == plan.steps[injection.step_idx].post_state_snapshot_sha256
        ),
        "matched_control_success": (
            matched_control.success
            and matched_control.executed_actions == source_actions
            and matched_control.final_state_snapshot_sha256 == source_final_state
            and treatment_verified(
                matched_control,
                expected_action=injection.source_action,
            )
        ),
    }
    missing = tuple(name for name, passed in checks.items() if not passed)
    if missing:
        raise ControlledFaultInjectionError(
            "controlled fault-injection causal gold gate failed: " + ", ".join(missing)
        )
    trace_case = _build_blind_failure_case(plan, injected_failure)
    binding = ControlledFaultInjectionCausalBindingV1(
        source_trajectory_sha256=plan.source_trajectory_sha256,
        failure_trajectory_sha256=injected_failure.trajectory_sha256,
        source_trace_sha256=trace_case.sha256,
        injection_step_idx=injection.step_idx,
        source_action=injection.source_action,
        injected_action=injection.injected_action,
        repair_action=injection.source_action,
        source_pre_state_snapshot_sha256=(injection.source_pre_state_snapshot_sha256),
        suffix_controller_sha256=plan.suffix_controller_sha256,
    )
    receipt = ControlledFaultInjectionCausalReceiptV1(
        binding=binding,
        baseline=baseline,
        injected_failure=injected_failure,
        matched_control=matched_control,
        causal_gold_admitted=True,
        missing_requirements=(),
    )
    return ControlledFaultInjectionArtifactsV1(
        trace_case=trace_case,
        causal_receipt=receipt,
    )


def _run_fault_injection_arm(
    plan: AlfWorldControlledFaultInjectionPlanV1,
    *,
    branch_factory: Callable[[], AlfWorldBranch],
    arm: FaultInjectionArmName,
    step_observer: Callable[
        [int, AlfWorldBranchObservation, AlfWorldBranchObservation, str, str],
        None,
    ]
    | None = None,
) -> ControlledFaultInjectionArmV1:
    branch = branch_factory()
    injection = plan.injection
    captured: list[CapturedFaultInjectionStepV1] = []
    pre_state_verified = True
    treatment_admissible = arm == "baseline"
    treatment_mode: str | None = None
    treatment_receipt: str | None = None
    terminal_reason = "budget_exhausted"
    last_step: Any = None
    try:
        current = branch.reset()
        for source_step in plan.steps:
            checkpoint = branch.capture_checkpoint()
            pre_sha256 = checkpoint.snapshot.state_snapshot_sha256
            if arm != "injected_failure" or source_step.step_idx <= injection.step_idx:
                pre_state_verified = pre_state_verified and (
                    pre_sha256 == source_step.state_snapshot_sha256
                )
            action = source_step.source_action
            if arm == "injected_failure" and source_step.step_idx == injection.step_idx:
                action = injection.injected_action
            action_admissible = action in current.admissible_actions
            is_treatment = (
                arm in {"injected_failure", "matched_control"}
                and source_step.step_idx == injection.step_idx
            )
            if is_treatment and not action_admissible:
                terminal_reason = "treatment_action_inadmissible"
                break
            if (
                arm == "injected_failure"
                and source_step.step_idx > injection.step_idx
                and not action_admissible
            ):
                terminal_reason = "suffix_action_inadmissible"
                break
            observation_before = current.text
            if is_treatment:
                treatment_admissible = action_admissible
                treatment_mode = "verified_branch_step"
                last_step = branch.step(action)
                treatment_receipt = last_step.restore_receipt_sha256
            else:
                direct_step = getattr(branch, "step_unverified", None)
                last_step = direct_step(action) if callable(direct_step) else branch.step(action)
            current = last_step.observation
            post_checkpoint = branch.capture_checkpoint()
            post_sha256 = post_checkpoint.snapshot.state_snapshot_sha256
            captured.append(
                CapturedFaultInjectionStepV1(
                    step_idx=source_step.step_idx,
                    action=action,
                    action_admissible=action_admissible,
                    observation_before=observation_before,
                    observation_after=current.text,
                    admissible_actions=tuple(checkpoint.observation.admissible_actions),
                    pre_state_snapshot_sha256=pre_sha256,
                    post_state_snapshot_sha256=post_sha256,
                )
            )
            if step_observer is not None:
                step_observer(
                    source_step.step_idx,
                    checkpoint.observation,
                    current,
                    pre_sha256,
                    post_sha256,
                )
            if last_step.won:
                terminal_reason = "environment_won"
                break
            if last_step.done:
                terminal_reason = "environment_done"
                break
        final_checkpoint = branch.capture_checkpoint()
        return ControlledFaultInjectionArmV1(
            arm=arm,
            success=bool(last_step is not None and last_step.won),
            terminal_reason=terminal_reason,
            pre_state_identity_verified=pre_state_verified,
            treatment_action_admissible=treatment_admissible,
            treatment_execution_mode=treatment_mode,
            treatment_restore_receipt_sha256=treatment_receipt,
            suffix_controller_sha256=plan.suffix_controller_sha256,
            steps=tuple(captured),
            final_state_snapshot_sha256=(final_checkpoint.snapshot.state_snapshot_sha256),
            physical_call_counts=dict(branch.physical_call_counters()),
        )
    finally:
        branch.close()


def execute_alfworld_controlled_fault_injection_arms(
    plan: AlfWorldControlledFaultInjectionPlanV1,
    *,
    branch_factory: Callable[[], AlfWorldBranch],
) -> ControlledFaultInjectionArmsV1:
    """Execute and return all frozen arms before applying the causal gate."""

    baseline = _run_fault_injection_arm(
        plan,
        branch_factory=branch_factory,
        arm="baseline",
    )
    injected_failure = _run_fault_injection_arm(
        plan,
        branch_factory=branch_factory,
        arm="injected_failure",
    )
    matched_control = _run_fault_injection_arm(
        plan,
        branch_factory=branch_factory,
        arm="matched_control",
    )
    return ControlledFaultInjectionArmsV1(
        baseline=baseline,
        injected_failure=injected_failure,
        matched_control=matched_control,
    )


def run_alfworld_controlled_fault_injection(
    plan: AlfWorldControlledFaultInjectionPlanV1,
    *,
    branch_factory: Callable[[], AlfWorldBranch],
    minimum_post_injection_actions: int,
) -> ControlledFaultInjectionArtifactsV1:
    """Execute the frozen arms and admit only a verified causal case."""

    arms = execute_alfworld_controlled_fault_injection_arms(
        plan,
        branch_factory=branch_factory,
    )
    return build_alfworld_controlled_fault_injection_artifacts(
        plan,
        baseline=arms.baseline,
        injected_failure=arms.injected_failure,
        matched_control=arms.matched_control,
        minimum_post_injection_actions=minimum_post_injection_actions,
    )


def build_alfworld_controlled_fault_injection_plan(
    records: tuple[TrainingTrajectoryRecord, ...] | list[TrainingTrajectoryRecord],
    *,
    step_idx: int,
    injected_action: str,
) -> AlfWorldControlledFaultInjectionPlanV1:
    """Freeze one preselected injection without reading any replay outcome."""

    source = build_alfworld_source_trajectory(records)
    if not source.success:
        raise ControlledFaultInjectionError(
            "controlled fault injection requires a successful source trajectory"
        )
    if step_idx < 0 or step_idx >= len(source.steps) - 1:
        raise ControlledFaultInjectionError(
            "controlled injection step must be nonterminal and inside the source trace"
        )
    step = source.steps[step_idx]
    if injected_action not in step.admissible_actions:
        raise ControlledFaultInjectionError(
            "controlled injected action must be source-state admissible"
        )
    if injected_action == step.source_action:
        raise ControlledFaultInjectionError(
            "controlled injected action must differ from the source action"
        )
    identity = {
        "source_trajectory_sha256": source.source_trajectory_sha256,
        "step_idx": step_idx,
        "source_action": step.source_action,
        "injected_action": injected_action,
        "source_pre_state_snapshot_sha256": step.state_snapshot_sha256,
    }
    injection = ControlledActionInjectionV1(
        injection_id=f"step-{step_idx:04d}-{canonical_sha256(identity)[:16]}",
        step_idx=step_idx,
        source_action=step.source_action,
        injected_action=injected_action,
        source_pre_state_snapshot_sha256=step.state_snapshot_sha256,
    )
    suffix_controller_sha256 = _sha256(
        {
            "schema": SOURCE_SUFFIX_CONTROLLER_SCHEMA,
            "source_trajectory_sha256": source.source_trajectory_sha256,
            "injection_step_idx": step_idx,
            "suffix_actions": [item.source_action for item in source.steps[step_idx + 1 :]],
        }
    )
    return AlfWorldControlledFaultInjectionPlanV1(
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
        source_trajectory_sha256=source.source_trajectory_sha256,
        source_terminal_reason=source.terminal_reason,
        steps=source.steps,
        injection=injection,
        suffix_controller_sha256=suffix_controller_sha256,
    )


__all__ = [
    "AlfWorldControlledFaultInjectionPlanV1",
    "CapturedFaultInjectionStepV1",
    "ControlledActionInjectionV1",
    "ControlledFaultInjectionArmV1",
    "ControlledFaultInjectionArmsV1",
    "ControlledFaultInjectionArtifactsV1",
    "ControlledFaultInjectionCausalReceiptV1",
    "ControlledFaultInjectionError",
    "build_alfworld_controlled_fault_injection_artifacts",
    "build_alfworld_controlled_fault_injection_plan",
    "execute_alfworld_controlled_fault_injection_arms",
    "run_alfworld_controlled_fault_injection",
]
