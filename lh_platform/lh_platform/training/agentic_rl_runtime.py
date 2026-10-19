"""Platform-native runtime boundary for comparable agentic-RL experiments.

The runtime owns orchestration and evidence, while five replaceable components
own their domain details: trainable policy, task suite, decomposition, credit / RL
method, and evaluation.  Baseline repositories are semantic oracles only; no
upstream trainer is imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from lh_platform.core.atomic_io import atomic_write_json, atomic_write_jsonl
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.trajectory_v2 import validate_active_trajectory_records

if TYPE_CHECKING:
    from lh_platform.training.dynamic_sampling import DynamicSamplingConfig
    from lh_platform.training.experience_bridge import ExperienceRecord
    from lh_platform.training.model_io import ConversationIR


AGENTIC_RL_RUN_RECEIPT_SCHEMA = "lh_agentic_rl_run_receipt/v1"
AGENTIC_RL_RUN_EVENT_SCHEMA = "lh_agentic_rl_run_event/v1"
PAIRED_RUNTIME_LOCK_SCHEMA = "lh_agentic_rl_paired_runtime_lock/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TREATMENT_FIELDS = frozenset({"decomposition", "credit_assigner"})
_ADVANTAGE_SIGNAL_EPSILON = 1e-12


class AgenticRLContractError(ValueError):
    """Raised before a non-comparable or non-on-policy update can run."""


class RunEventLogger:
    """Append-only structured lifecycle log for one experiment invocation.

    The logger deliberately accepts explicit payloads only. It never captures
    process environment variables, model parameters, prompts, or action text.
    """

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        if not str(run_id).strip():
            raise AgenticRLContractError("run event log requires a non-empty run_id")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self._sequence = 0
        self._lock = threading.Lock()
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fd = os.open(self.path, flags, 0o600)
        except FileExistsError as error:
            raise AgenticRLContractError(f"run event log already exists: {self.path}") from error
        os.close(fd)

    def emit(self, event: str, **payload: Any) -> None:
        if not str(event).strip():
            raise AgenticRLContractError("run event name must be non-empty")
        with self._lock:
            row = {
                "schema": AGENTIC_RL_RUN_EVENT_SCHEMA,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "event": str(event),
                "payload": payload,
            }
            encoded = (
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            ).encode("utf-8")
            flags = (
                os.O_WRONLY
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(self.path, flags)
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("failed to append run event")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            self._sequence += 1


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Bind a prompt to the exact rollout tokenizer IDs, not re-tokenized text."""

    return canonical_sha256([int(token) for token in token_ids])


@dataclass(frozen=True)
class ComponentRef:
    """Content-addressed implementation/config identity for one runtime role."""

    role: str
    name: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.role.strip() or not self.name.strip() or not self.revision.strip():
            raise AgenticRLContractError("component role, name, and revision are required")
        _require_sha256(self.sha256, f"component {self.role!r} sha256")


@dataclass(frozen=True)
class ExperimentArm:
    """Everything capable of changing an arm's result, with treatment explicit."""

    arm_id: str
    policy_backend: ComponentRef
    initial_checkpoint_sha256: str
    train_task_suite: ComponentRef
    train_manifest_sha256: str
    eval_task_suite: ComponentRef
    eval_manifest_sha256: str
    rl_method: ComponentRef
    optimizer_config_sha256: str
    rollout_config_sha256: str
    prompt_template_sha256: str
    seed_schedule_sha256: str
    decomposition: ComponentRef
    credit_assigner: ComponentRef

    def __post_init__(self) -> None:
        if not self.arm_id.strip():
            raise AgenticRLContractError("arm_id is required")
        for field_name in (
            "initial_checkpoint_sha256",
            "train_manifest_sha256",
            "eval_manifest_sha256",
            "optimizer_config_sha256",
            "rollout_config_sha256",
            "prompt_template_sha256",
            "seed_schedule_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        expected_roles = {
            "policy_backend": "policy_backend",
            "train_task_suite": "train_task_suite",
            "eval_task_suite": "eval_task_suite",
            "rl_method": "rl_method",
            "decomposition": "decomposition",
            "credit_assigner": "credit_assigner",
        }
        for field_name, role in expected_roles.items():
            if getattr(self, field_name).role != role:
                raise AgenticRLContractError(f"{field_name} must have component role {role!r}")


@dataclass(frozen=True)
class PairedRuntimeLock:
    schema: str
    control_arm_id: str
    treatment_arm_id: str
    fixed_factors_sha256: str
    treatment_delta: tuple[str, ...]


def build_paired_runtime_lock(
    control: ExperimentArm,
    treatment: ExperimentArm,
) -> PairedRuntimeLock:
    """Fail closed unless the two arms differ only in decomposition / credit."""

    if control.arm_id == treatment.arm_id:
        raise AgenticRLContractError("paired experiment requires distinct arm IDs")
    control_values = _arm_values(control)
    treatment_values = _arm_values(treatment)
    drift = tuple(
        name
        for name in sorted(control_values)
        if name not in {"arm_id", *_TREATMENT_FIELDS}
        and control_values[name] != treatment_values[name]
    )
    if drift:
        raise AgenticRLContractError("paired runtime fixed factor drift: " + ", ".join(drift))
    delta = tuple(
        name for name in sorted(_TREATMENT_FIELDS) if control_values[name] != treatment_values[name]
    )
    if not delta:
        raise AgenticRLContractError(
            "paired experiment has no decomposition or credit treatment delta"
        )
    fixed = {
        name: value
        for name, value in control_values.items()
        if name not in {"arm_id", *_TREATMENT_FIELDS}
    }
    return PairedRuntimeLock(
        schema=PAIRED_RUNTIME_LOCK_SCHEMA,
        control_arm_id=control.arm_id,
        treatment_arm_id=treatment.arm_id,
        fixed_factors_sha256=canonical_sha256(fixed),
        treatment_delta=delta,
    )


@dataclass(frozen=True)
class OnPolicyActionSample:
    """Policy-owned evidence for one exact action-token span."""

    traj_uid: str
    step_idx: int
    action_text: str
    prompt_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    old_action_token_log_probs: tuple[float, ...]
    policy_checkpoint_sha256: str
    rollout_seed: int
    response_mask: tuple[int, ...] = ()
    model_io_profile_sha256: str | None = None
    response_contract_sha256: str | None = None
    normalized_messages_sha256: str | None = None
    chat_template_sha256: str | None = None
    matched_stop_text: str | None = None
    finish_reason: str = "unknown"
    sampling_temperature: float = 1.0
    allowed_action_token_ids: tuple[tuple[int, ...], ...] = ()
    sampling_mode: str = "policy"


@dataclass(frozen=True)
class PolicyActionRequest:
    """One task-suite prompt bound to a trajectory step."""

    traj_uid: str
    step_idx: int
    prompt_text: str | None = None
    conversation: ConversationIR | None = None
    model_io_profile_id: str | None = None
    response_contract_id: str | None = None
    response_contract_sha256: str | None = None
    admissible_actions: tuple[str, ...] = ()
    sampling_temperature: float | None = None
    sampling_seed: int | None = None
    response_stop_texts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.prompt_text is None) == (self.conversation is None):
            raise AgenticRLContractError(
                "policy request requires exactly one of prompt_text or conversation"
            )
        if self.prompt_text is not None and not self.prompt_text:
            raise AgenticRLContractError("policy request prompt_text is empty")
        if self.response_contract_sha256 is not None:
            _require_sha256(
                self.response_contract_sha256,
                "policy request response contract",
            )
        if (self.response_contract_id is None) != (self.response_contract_sha256 is None):
            raise AgenticRLContractError(
                "response contract ID and hash must be configured together"
            )
        if self.sampling_seed is not None and (
            isinstance(self.sampling_seed, bool)
            or not isinstance(self.sampling_seed, int)
            or self.sampling_seed < 0
        ):
            raise AgenticRLContractError(
                "policy request sampling_seed must be a non-negative integer"
            )
        if any(not str(value) for value in self.response_stop_texts):
            raise AgenticRLContractError("response stop texts must be non-empty")
        if len(set(self.response_stop_texts)) != len(self.response_stop_texts):
            raise AgenticRLContractError("response stop texts must be unique")


@dataclass(frozen=True)
class AgenticRolloutBatch:
    records: tuple[TrainingTrajectoryRecord, ...]
    policy_samples: tuple[OnPolicyActionSample, ...]
    task_manifest_sha256: str
    policy_checkpoint_sha256: str

    def validate(self) -> AgenticRolloutBatch:
        _require_sha256(self.task_manifest_sha256, "task_manifest_sha256")
        _require_sha256(self.policy_checkpoint_sha256, "policy_checkpoint_sha256")
        try:
            rows = validate_active_trajectory_records(self.records, require_v2=True)
        except ValueError as error:
            raise AgenticRLContractError(f"invalid rollout trajectory: {error}") from error
        samples = _index_unique(
            self.policy_samples,
            key=lambda item: (item.traj_uid, item.step_idx),
            label="policy sample",
        )
        row_keys = {(row.traj_uid, row.step_idx) for row in rows}
        if set(samples) != row_keys:
            raise AgenticRLContractError(
                "every active transition requires exactly one current-policy sample"
            )
        for row in rows:
            sample = samples[(row.traj_uid, row.step_idx)]
            _validate_sample(sample)
            if sample.policy_checkpoint_sha256 != self.policy_checkpoint_sha256:
                raise AgenticRLContractError(
                    "rollout mixes samples from a different policy checkpoint"
                )
            if row.action_raw != sample.action_text:
                raise AgenticRLContractError(
                    "trajectory action differs from the policy-owned sampled action"
                )
            if row.action_token_count != len(sample.action_token_ids):
                raise AgenticRLContractError(
                    "action_token_count differs from exact rollout action token IDs"
                )
            if row.prompt_hash != token_ids_sha256(sample.prompt_token_ids):
                raise AgenticRLContractError(
                    "prompt_hash differs from exact rollout prompt token IDs"
                )
            token_evidence = (
                row.prompt_ids,
                row.response_ids,
                row.response_mask,
                row.response_old_logprobs,
            )
            if token_evidence != (
                sample.prompt_token_ids,
                sample.action_token_ids,
                (1,) * len(sample.action_token_ids),
                sample.old_action_token_log_probs,
            ):
                raise AgenticRLContractError(
                    "trajectory exact-token evidence differs from policy sample"
                )
            if row.extra.get("task_manifest_lock_sha256") != self.task_manifest_sha256:
                raise AgenticRLContractError(
                    "trajectory task manifest lock differs from the collected suite"
                )
        return self


@dataclass(frozen=True)
class PolicyTarget:
    traj_uid: str
    step_idx: int
    advantage: float
    actor_token_weight: float


@dataclass(frozen=True)
class PreparedPolicyUpdate:
    method: ComponentRef
    credit_assigner: ComponentRef
    objective: str
    targets: tuple[PolicyTarget, ...]
    diagnostics: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PolicyUpdateResult:
    optimizer_step: int
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    trainable_parameter_delta_l2: float
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class CheckpointArtifact:
    path: str
    sha256: str


@dataclass(frozen=True)
class EvaluationResult:
    task_suite: ComponentRef
    task_manifest_sha256: str
    policy_checkpoint_sha256: str
    task_count: int
    success_count: int
    mean_action_steps: float
    success_by_horizon: Mapping[str, Mapping[str, int]]
    sampling_temperature: float | None = None
    episodes_per_task: int = 1
    stochastic_sampling: bool = False
    sampling_uncertainty_interval_available: bool = False
    evaluation_selection_method: str = ""


@dataclass(frozen=True)
class RolloutRequest:
    update_index: int
    seed: int
    group_size: int
    max_episode_steps: int
    rollout_config_sha256: str
    sampling_seed_base: int | None = None

    def __post_init__(self) -> None:
        if self.sampling_seed_base is not None and (
            isinstance(self.sampling_seed_base, bool)
            or not isinstance(self.sampling_seed_base, int)
            or self.sampling_seed_base < 0
        ):
            raise AgenticRLContractError(
                "rollout sampling_seed_base must be a non-negative integer"
            )


@dataclass(frozen=True)
class EvaluationRequest:
    seed: int
    max_episode_steps: int


@dataclass(frozen=True)
class ArmRunPlan:
    arm: ExperimentArm
    updates: int
    rollout_seeds: tuple[int, ...]
    eval_seed: int
    group_size: int
    max_episode_steps: int

    def __post_init__(self) -> None:
        if self.updates < 1 or len(self.rollout_seeds) != self.updates:
            raise AgenticRLContractError("rollout_seeds must contain exactly one seed per update")
        if any(type(seed) is not int or seed < 0 for seed in self.rollout_seeds):
            raise AgenticRLContractError("rollout seeds must be non-negative integers")
        if type(self.eval_seed) is not int or self.eval_seed < 0:
            raise AgenticRLContractError("eval seed must be a non-negative integer")
        if self.group_size < 1 or self.max_episode_steps < 1:
            raise AgenticRLContractError("rollout group size and horizon must be positive")
        if canonical_sha256(list(self.rollout_seeds)) != self.arm.seed_schedule_sha256:
            raise AgenticRLContractError(
                "run plan seed schedule differs from the arm's locked schedule"
            )


@dataclass(frozen=True)
class ArmRunResult:
    arm_id: str
    updates_completed: int
    update_results: tuple[PolicyUpdateResult, ...]
    update_diagnostics: tuple[Mapping[str, Any], ...]
    checkpoint: CheckpointArtifact
    evaluation: EvaluationResult
    receipt_path: str


class TrainablePolicy(Protocol):
    component: ComponentRef
    checkpoint_sha256: str

    def sampling_cursor(self) -> int: ...

    def sampling_events_since(self, cursor: int) -> tuple[OnPolicyActionSample, ...]: ...

    def sample_actions(
        self,
        requests: tuple[PolicyActionRequest, ...],
        *,
        seed: int,
    ) -> tuple[OnPolicyActionSample, ...]: ...

    def apply_experience_records(
        self,
        records: Sequence[ExperienceRecord],
    ) -> PolicyUpdateResult: ...

    def materialize_checkpoint(self, output_dir: Path) -> CheckpointArtifact: ...


class AgenticTaskSuite(Protocol):
    component: ComponentRef
    manifest_sha256: str

    def collect(
        self,
        policy: TrainablePolicy,
        request: RolloutRequest,
    ) -> AgenticRolloutBatch: ...

    def evaluate(
        self,
        policy: TrainablePolicy,
        request: EvaluationRequest,
    ) -> EvaluationResult: ...


class DecompositionProvider(Protocol):
    component: ComponentRef

    def annotate(
        self,
        records: tuple[TrainingTrajectoryRecord, ...],
    ) -> tuple[TrainingTrajectoryRecord, ...]: ...


class RLMethod(Protocol):
    component: ComponentRef
    credit_assigner: ComponentRef

    def prepare_update(
        self,
        records: tuple[TrainingTrajectoryRecord, ...],
    ) -> PreparedPolicyUpdate: ...


def collect_on_policy_rollout(
    *,
    policy: TrainablePolicy,
    train_suite: AgenticTaskSuite,
    request: RolloutRequest,
    expected_manifest_sha256: str,
    expected_checkpoint_sha256: str,
) -> AgenticRolloutBatch:
    """Collect once and bind every sample to the policy-owned ledger."""

    cursor = policy.sampling_cursor()
    rollout = train_suite.collect(policy, request).validate()
    _validate_on_policy_rollout(
        rollout,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    if rollout.policy_samples != policy.sampling_events_since(cursor):
        raise AgenticRLContractError(
            "rollout samples do not match the policy-owned sampling ledger"
        )
    return rollout


def _validate_on_policy_rollout(
    rollout: AgenticRolloutBatch,
    *,
    expected_manifest_sha256: str,
    expected_checkpoint_sha256: str,
) -> None:
    if rollout.task_manifest_sha256 != expected_manifest_sha256:
        raise AgenticRLContractError("collected rollout used a different train manifest")
    if rollout.policy_checkpoint_sha256 != expected_checkpoint_sha256:
        raise AgenticRLContractError("collected rollout is not from the current checkpoint")


class AgenticRLExperimentRunner:
    """Collect, annotate, credit, update, checkpoint, and evaluate one arm."""

    def run_arm(
        self,
        plan: ArmRunPlan,
        *,
        policy: TrainablePolicy,
        train_suite: AgenticTaskSuite,
        eval_suite: AgenticTaskSuite,
        decomposition: DecompositionProvider,
        method: RLMethod,
        output_dir: Path,
        event_logger: RunEventLogger | None = None,
        training_backend: object | None = None,
        rollout_overrides: Mapping[int, AgenticRolloutBatch] | None = None,
        dynamic_sampling_config: DynamicSamplingConfig | None = None,
    ) -> ArmRunResult:
        if event_logger is None:
            event_logger = RunEventLogger(
                Path(output_dir) / "run_events.jsonl",
                run_id=uuid.uuid4().hex,
            )
        _emit_event(
            event_logger,
            "arm_started",
            arm=asdict(plan.arm),
            plan={
                "updates": plan.updates,
                "rollout_seeds": list(plan.rollout_seeds),
                "eval_seed": plan.eval_seed,
                "group_size": plan.group_size,
                "max_episode_steps": plan.max_episode_steps,
            },
            output_dir=str(Path(output_dir).resolve()),
        )
        try:
            return self._run_arm(
                plan,
                policy=policy,
                train_suite=train_suite,
                eval_suite=eval_suite,
                decomposition=decomposition,
                method=method,
                output_dir=output_dir,
                event_logger=event_logger,
                training_backend=training_backend,
                rollout_overrides=rollout_overrides,
                dynamic_sampling_config=dynamic_sampling_config,
            )
        except Exception as error:
            _emit_event(
                event_logger,
                "arm_failed",
                arm_id=plan.arm.arm_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise

    def _run_arm(
        self,
        plan: ArmRunPlan,
        *,
        policy: TrainablePolicy,
        train_suite: AgenticTaskSuite,
        eval_suite: AgenticTaskSuite,
        decomposition: DecompositionProvider,
        method: RLMethod,
        output_dir: Path,
        event_logger: RunEventLogger | None,
        training_backend: object | None,
        rollout_overrides: Mapping[int, AgenticRolloutBatch] | None,
        dynamic_sampling_config: DynamicSamplingConfig | None = None,
    ) -> ArmRunResult:
        from lh_platform.training.backends.local_transformers import (
            LocalTransformersBackend,
        )
        from lh_platform.training.dynamic_sampling import (
            DynamicSamplingConfig as _DynamicSamplingConfig,
        )
        from lh_platform.training.dynamic_sampling import (
            filter_zero_variance_groups,
        )
        from lh_platform.training.experience_bridge import (
            build_experience_records,
            write_experience_artifact,
        )

        arm = plan.arm
        output_dir = Path(output_dir)
        _require_component(policy.component, arm.policy_backend, "policy backend")
        _require_component(train_suite.component, arm.train_task_suite, "train task suite")
        _require_component(eval_suite.component, arm.eval_task_suite, "eval task suite")
        _require_component(decomposition.component, arm.decomposition, "decomposition")
        _require_component(method.component, arm.rl_method, "RL method")
        _require_component(method.credit_assigner, arm.credit_assigner, "credit assigner")
        if policy.checkpoint_sha256 != arm.initial_checkpoint_sha256:
            raise AgenticRLContractError(
                "policy initial checkpoint differs from the arm's locked checkpoint"
            )
        if train_suite.manifest_sha256 != arm.train_manifest_sha256:
            raise AgenticRLContractError("train task manifest differs from the arm lock")
        if eval_suite.manifest_sha256 != arm.eval_manifest_sha256:
            raise AgenticRLContractError("eval task manifest differs from the arm lock")

        update_results: list[PolicyUpdateResult] = []
        update_diagnostics: list[Mapping[str, Any]] = []
        applied_update_count = 0
        if training_backend is None:
            training_backend = LocalTransformersBackend(policy)
        elif training_backend is not policy:
            raise AgenticRLContractError(
                "an injected training backend must also be the synchronous rollout policy"
            )
        rollout_artifacts: list[Mapping[str, Any]] = []
        experience_artifacts: list[Mapping[str, Any]] = []
        overrides = dict(rollout_overrides or {})
        if set(overrides) - set(range(plan.updates)):
            raise AgenticRLContractError(
                "rollout overrides contain an update outside this arm plan"
            )
        for update_index, seed in enumerate(plan.rollout_seeds):
            checkpoint_before = policy.checkpoint_sha256
            _emit_event(
                event_logger,
                "rollout_started",
                arm_id=arm.arm_id,
                update_index=update_index,
                seed=seed,
                policy_checkpoint_sha256=checkpoint_before,
            )
            request = RolloutRequest(
                update_index=update_index,
                seed=seed,
                group_size=plan.group_size,
                max_episode_steps=plan.max_episode_steps,
                rollout_config_sha256=arm.rollout_config_sha256,
            )
            if update_index in overrides:
                rollout = overrides[update_index].validate()
                _validate_on_policy_rollout(
                    rollout,
                    expected_manifest_sha256=arm.train_manifest_sha256,
                    expected_checkpoint_sha256=checkpoint_before,
                )
                rollout_source = "precollected_frozen_batch"
            else:
                rollout = collect_on_policy_rollout(
                    policy=policy,
                    train_suite=train_suite,
                    request=request,
                    expected_manifest_sha256=arm.train_manifest_sha256,
                    expected_checkpoint_sha256=checkpoint_before,
                )
                rollout_source = "live_policy_collection"
            ds_config = dynamic_sampling_config or _DynamicSamplingConfig()
            if ds_config.enabled and update_index not in overrides:
                rollout, ds_receipt = filter_zero_variance_groups(
                    rollout,
                    require_terminal_variation=ds_config.require_terminal_variation,
                )
                _emit_event(
                    event_logger,
                    "dynamic_sampling_applied",
                    arm_id=arm.arm_id,
                    update_index=update_index,
                    groups_collected=ds_receipt.groups_collected,
                    groups_informative=ds_receipt.groups_informative,
                    groups_filtered=ds_receipt.groups_filtered,
                    transitions_kept=ds_receipt.transitions_kept,
                    transitions_discarded=ds_receipt.transitions_discarded,
                )
                if not rollout.records:
                    _emit_event(
                        event_logger,
                        "dynamic_sampling_skipped_update",
                        arm_id=arm.arm_id,
                        update_index=update_index,
                        reason="zero informative groups after filtering",
                    )
                    continue
            raw_rollout_path = output_dir / "rollouts" / f"update-{update_index:04d}-raw.jsonl"
            raw_rollout_sha256 = _write_rollout_artifact(
                raw_rollout_path,
                rollout,
            )
            _emit_event(
                event_logger,
                "rollout_collected",
                arm_id=arm.arm_id,
                update_index=update_index,
                transition_count=len(rollout.records),
                trajectory_count=len({row.traj_uid for row in rollout.records}),
                task_count=len({row.task_id for row in rollout.records}),
                terminal_count=sum(row.terminal for row in rollout.records),
                success_count=sum(row.terminal and row.success for row in rollout.records),
                action_token_count=sum(row.action_token_count for row in rollout.records),
                task_manifest_sha256=rollout.task_manifest_sha256,
                policy_checkpoint_sha256=rollout.policy_checkpoint_sha256,
                rollout_artifact_path=str(raw_rollout_path),
                rollout_artifact_sha256=raw_rollout_sha256,
                rollout_source=rollout_source,
            )

            annotated = tuple(decomposition.annotate(rollout.records))
            _validate_decomposition_projection(rollout.records, annotated)
            _emit_event(
                event_logger,
                "decomposition_applied",
                arm_id=arm.arm_id,
                update_index=update_index,
                decomposition=asdict(decomposition.component),
                transition_count=len(annotated),
                annotated_transition_count=sum(
                    bool(
                        row.extra.get("active_subproblem_id")
                        or row.extra.get("goal_annotation_verified") is True
                    )
                    for row in annotated
                ),
            )
            annotated_rollout = AgenticRolloutBatch(
                records=annotated,
                policy_samples=rollout.policy_samples,
                task_manifest_sha256=rollout.task_manifest_sha256,
                policy_checkpoint_sha256=rollout.policy_checkpoint_sha256,
            ).validate()
            annotated_rollout_path = (
                output_dir / "rollouts" / f"update-{update_index:04d}-annotated.jsonl"
            )
            annotated_rollout_sha256 = _write_rollout_artifact(
                annotated_rollout_path,
                annotated_rollout,
            )
            rollout_artifacts.append(
                {
                    "update_index": update_index,
                    "raw_path": str(raw_rollout_path),
                    "raw_sha256": raw_rollout_sha256,
                    "annotated_path": str(annotated_rollout_path),
                    "annotated_sha256": annotated_rollout_sha256,
                }
            )
            _emit_event(
                event_logger,
                "annotated_rollout_written",
                arm_id=arm.arm_id,
                update_index=update_index,
                path=str(annotated_rollout_path),
                sha256=annotated_rollout_sha256,
            )
            prepared = method.prepare_update(annotated)
            _require_component(prepared.method, arm.rl_method, "prepared RL method")
            _require_component(
                prepared.credit_assigner,
                arm.credit_assigner,
                "prepared credit assigner",
            )
            _validate_policy_targets(annotated, prepared.targets)
            prepared_diagnostics = dict(prepared.diagnostics or {})
            try:
                json.dumps(prepared_diagnostics, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise AgenticRLContractError(
                    "prepared update diagnostics must be JSON serializable"
                ) from error
            _emit_event(
                event_logger,
                "update_prepared",
                arm_id=arm.arm_id,
                update_index=update_index,
                method=asdict(prepared.method),
                credit_assigner=asdict(prepared.credit_assigner),
                objective=prepared.objective,
                target_count=len(prepared.targets),
                diagnostics=prepared_diagnostics,
            )
            # A batch with no advantage signal must be a true no-op: the
            # reference-KL gradient would otherwise move the policy without any
            # learning signal, which the update validator rejects.
            has_learning_signal = any(
                abs(target.advantage) > _ADVANTAGE_SIGNAL_EPSILON
                for target in prepared.targets
            )
            if not has_learning_signal:
                _emit_event(
                    event_logger,
                    "update_skipped_no_signal",
                    arm_id=arm.arm_id,
                    update_index=update_index,
                    target_count=len(prepared.targets),
                    diagnostics=prepared_diagnostics,
                )
                continue
            experience_records = build_experience_records(
                annotated_rollout,
                prepared,
            )
            experience_path = output_dir / f"update_{update_index:04d}" / "experience_records.jsonl"
            experience_artifact_sha256 = write_experience_artifact(
                experience_path,
                experience_records,
            )
            experience_receipt = training_backend.submit(experience_records)
            experience_artifacts.append(
                {
                    "update_index": update_index,
                    "path": str(experience_path),
                    "artifact_sha256": experience_artifact_sha256,
                    **asdict(experience_receipt),
                }
            )
            _emit_event(
                event_logger,
                "experience_submitted",
                arm_id=arm.arm_id,
                update_index=update_index,
                **asdict(experience_receipt),
            )
            result = training_backend.update()
            # Only signal-carrying batches reach the optimizer, so the expected
            # step counts applied updates; skipped updates never step it.
            _validate_update_result(
                result,
                expected_step=applied_update_count + 1,
                checkpoint_before=checkpoint_before,
                checkpoint_after=policy.checkpoint_sha256,
                has_learning_signal=True,
            )
            update_results.append(result)
            update_diagnostics.append(prepared_diagnostics)
            applied_update_count += 1
            _emit_event(
                event_logger,
                "update_applied",
                arm_id=arm.arm_id,
                update_index=update_index,
                **asdict(result),
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = training_backend.checkpoint(output_dir)
        _require_sha256(checkpoint.sha256, "checkpoint artifact sha256")
        if checkpoint.sha256 != policy.checkpoint_sha256:
            raise AgenticRLContractError("materialized checkpoint identity differs from policy")
        if not Path(checkpoint.path).is_file():
            raise AgenticRLContractError("materialized checkpoint artifact does not exist")
        _emit_event(
            event_logger,
            "checkpoint_materialized",
            arm_id=arm.arm_id,
            **asdict(checkpoint),
        )
        evaluation = eval_suite.evaluate(
            policy,
            EvaluationRequest(seed=plan.eval_seed, max_episode_steps=plan.max_episode_steps),
        )
        _validate_evaluation(evaluation, arm=arm, policy=policy)
        _emit_event(
            event_logger,
            "evaluation_completed",
            arm_id=arm.arm_id,
            **asdict(evaluation),
        )

        receipt_path = output_dir / "agentic_rl_run_receipt.json"
        receipt = {
            "schema": AGENTIC_RL_RUN_RECEIPT_SCHEMA,
            "arm": asdict(arm),
            "updates_completed": len(update_results),
            "update_results": [asdict(item) for item in update_results],
            "update_diagnostics": list(update_diagnostics),
            "rollout_artifacts": rollout_artifacts,
            "experience_artifacts": experience_artifacts,
            "training_backend": asdict(training_backend.identity),
            "checkpoint": asdict(checkpoint),
            "evaluation": asdict(evaluation),
        }
        # Publish the receipt atomically: a crash mid-write must leave either no
        # receipt or a complete one, never a truncated file.
        atomic_write_json(receipt_path, receipt, overwrite=True)
        _emit_event(
            event_logger,
            "arm_completed",
            arm_id=arm.arm_id,
            updates_completed=len(update_results),
            receipt_path=str(receipt_path),
            checkpoint_sha256=checkpoint.sha256,
        )
        return ArmRunResult(
            arm_id=arm.arm_id,
            updates_completed=len(update_results),
            update_results=tuple(update_results),
            update_diagnostics=tuple(update_diagnostics),
            checkpoint=checkpoint,
            evaluation=evaluation,
            receipt_path=str(receipt_path),
        )


def _emit_event(
    event_logger: RunEventLogger | None,
    event: str,
    **payload: Any,
) -> None:
    if event_logger is not None:
        event_logger.emit(event, **payload)


def _write_rollout_artifact(
    path: Path,
    rollout: AgenticRolloutBatch,
) -> str:
    """Persist reviewable prompts/responses/actions and exact policy evidence."""

    samples = {(sample.traj_uid, sample.step_idx): sample for sample in rollout.policy_samples}
    atomic_write_jsonl(
        path,
        (
            {
                "schema": "lh_agentic_rl_rollout_transition/v1",
                "record": asdict(record),
                "policy_sample": asdict(samples[(record.traj_uid, record.step_idx)]),
            }
            for record in rollout.records
        ),
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_agentic_rollout_artifact(
    path: str | Path,
    rollout: AgenticRolloutBatch,
) -> str:
    """Persist one validated frozen rollout with exact policy-token evidence."""

    return _write_rollout_artifact(Path(path), rollout.validate())


def load_agentic_rollout_artifact(path: str | Path) -> AgenticRolloutBatch:
    """Reload a persisted rollout without dropping old-policy sample evidence."""

    source = Path(path).resolve()
    records: list[TrainingTrajectoryRecord] = []
    samples: list[OnPolicyActionSample] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AgenticRLContractError(f"cannot read rollout artifact: {error}") from error
    if not lines:
        raise AgenticRLContractError("rollout artifact is empty")
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AgenticRLContractError(
                f"rollout artifact line {line_number} is not JSON"
            ) from error
        if not isinstance(row, Mapping) or row.get("schema") != (
            "lh_agentic_rl_rollout_transition/v1"
        ):
            raise AgenticRLContractError(
                f"rollout artifact line {line_number} has an unsupported schema"
            )
        record = row.get("record")
        sample = row.get("policy_sample")
        if not isinstance(record, dict) or not isinstance(sample, Mapping):
            raise AgenticRLContractError(
                f"rollout artifact line {line_number} lacks record/sample evidence"
            )
        records.append(TrainingTrajectoryRecord.from_dict(record))
        samples.append(
            OnPolicyActionSample(
                traj_uid=str(sample.get("traj_uid", "")),
                step_idx=int(sample.get("step_idx", -1)),
                action_text=str(sample.get("action_text", "")),
                prompt_token_ids=tuple(int(token) for token in sample.get("prompt_token_ids", ())),
                action_token_ids=tuple(int(token) for token in sample.get("action_token_ids", ())),
                old_action_token_log_probs=tuple(
                    float(value) for value in sample.get("old_action_token_log_probs", ())
                ),
                policy_checkpoint_sha256=str(sample.get("policy_checkpoint_sha256", "")),
                rollout_seed=int(sample.get("rollout_seed", 0)),
                response_mask=tuple(
                    int(value)
                    for value in sample.get(
                        "response_mask",
                        (1,) * len(sample.get("action_token_ids", ())),
                    )
                ),
                sampling_temperature=float(sample.get("sampling_temperature", 1.0)),
                allowed_action_token_ids=tuple(
                    tuple(int(token) for token in allowed)
                    for allowed in sample.get("allowed_action_token_ids", ())
                ),
                sampling_mode=str(sample.get("sampling_mode", "policy")),
            )
        )
    manifest_locks = {str(record.extra.get("task_manifest_lock_sha256", "")) for record in records}
    checkpoint_locks = {sample.policy_checkpoint_sha256 for sample in samples}
    if len(manifest_locks) != 1 or len(checkpoint_locks) != 1:
        raise AgenticRLContractError("rollout artifact mixes task manifests or policy checkpoints")
    return AgenticRolloutBatch(
        records=tuple(records),
        policy_samples=tuple(samples),
        task_manifest_sha256=next(iter(manifest_locks)),
        policy_checkpoint_sha256=next(iter(checkpoint_locks)),
    ).validate()


def _arm_values(arm: ExperimentArm) -> dict[str, Any]:
    return asdict(arm)


def _require_sha256(value: str, label: str) -> None:
    if not _SHA256.fullmatch(str(value or "")):
        raise AgenticRLContractError(f"{label} must be a lowercase SHA-256 digest")


def _require_component(actual: ComponentRef, expected: ComponentRef, label: str) -> None:
    if actual != expected:
        raise AgenticRLContractError(f"{label} differs from the arm's locked component")


def _index_unique(items: Iterable[Any], *, key: Any, label: str) -> dict[Any, Any]:
    indexed: dict[Any, Any] = {}
    for item in items:
        item_key = key(item)
        if item_key in indexed:
            raise AgenticRLContractError(f"duplicate {label} key: {item_key!r}")
        indexed[item_key] = item
    return indexed


def _validate_sample(sample: OnPolicyActionSample) -> None:
    if sample.sampling_mode != "policy":
        raise AgenticRLContractError("expert/replay actions cannot enter an on-policy update")
    if not sample.action_text.strip():
        raise AgenticRLContractError("sampled policy action must be observable")
    if not sample.prompt_token_ids or not sample.action_token_ids:
        raise AgenticRLContractError("policy sample requires exact prompt and action token IDs")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in (*sample.prompt_token_ids, *sample.action_token_ids)
    ):
        raise AgenticRLContractError("policy token IDs must be non-negative integers")
    if len(sample.old_action_token_log_probs) != len(sample.action_token_ids):
        raise AgenticRLContractError(
            "old-policy log probabilities must align exactly with action tokens"
        )
    if any(not math.isfinite(value) for value in sample.old_action_token_log_probs):
        raise AgenticRLContractError("old-policy action log probabilities must be finite")
    if not math.isfinite(sample.sampling_temperature) or sample.sampling_temperature <= 0:
        raise AgenticRLContractError("policy sampling temperature must be finite and positive")
    if type(sample.rollout_seed) is not int or sample.rollout_seed < 0:
        raise AgenticRLContractError("policy rollout seed must be a non-negative integer")
    for value, label in (
        (sample.model_io_profile_sha256, "sample model I/O profile"),
        (sample.response_contract_sha256, "sample response contract"),
        (sample.normalized_messages_sha256, "sample normalized messages"),
        (sample.chat_template_sha256, "sample chat template"),
    ):
        if value is not None:
            _require_sha256(value, label)
    if sample.allowed_action_token_ids:
        if len(sample.allowed_action_token_ids) != len(sample.action_token_ids):
            raise AgenticRLContractError(
                "constrained-policy token sets must align with action tokens"
            )
        for selected, allowed in zip(
            sample.action_token_ids,
            sample.allowed_action_token_ids,
            strict=True,
        ):
            if not allowed or selected not in allowed:
                raise AgenticRLContractError(
                    "sampled action token is outside its recorded policy support"
                )
    _require_sha256(sample.policy_checkpoint_sha256, "sample policy checkpoint")


def validate_action_sample_for_request(
    request: PolicyActionRequest,
    sample: OnPolicyActionSample,
    *,
    expected_model_io_profile_sha256: str | None = None,
) -> None:
    """Fail closed when a backend returns evidence for a different I/O contract."""

    _validate_sample(sample)
    if (sample.traj_uid, sample.step_idx) != (request.traj_uid, request.step_idx):
        raise AgenticRLContractError("policy sample identity differs from request")
    if sample.response_contract_sha256 != request.response_contract_sha256:
        raise AgenticRLContractError("policy sample response contract differs from request")
    if request.sampling_seed is not None and sample.rollout_seed != request.sampling_seed:
        raise AgenticRLContractError("policy sample seed differs from request")
    if expected_model_io_profile_sha256 is not None:
        _require_sha256(
            expected_model_io_profile_sha256,
            "expected model I/O profile",
        )
        if sample.model_io_profile_sha256 != expected_model_io_profile_sha256:
            raise AgenticRLContractError(
                "policy sample model I/O profile differs from loaded policy"
            )
        if sample.normalized_messages_sha256 is None or sample.chat_template_sha256 is None:
            raise AgenticRLContractError(
                "explicit model I/O sample omitted message/template evidence"
            )
    elif request.model_io_profile_id is not None:
        raise AgenticRLContractError("model-I/O-bound request was served by a legacy policy")
    if sample.matched_stop_text is not None and (
        sample.matched_stop_text not in request.response_stop_texts
    ):
        raise AgenticRLContractError("policy sample matched a stop outside the response contract")


def _validate_decomposition_projection(
    before: Sequence[TrainingTrajectoryRecord],
    after: Sequence[TrainingTrajectoryRecord],
) -> None:
    if len(before) != len(after):
        raise AgenticRLContractError("decomposition cannot add or remove transitions")
    immutable_fields = tuple(
        field.name for field in fields(TrainingTrajectoryRecord) if field.name != "extra"
    )
    for original, annotated in zip(before, after, strict=True):
        if any(getattr(original, name) != getattr(annotated, name) for name in immutable_fields):
            raise AgenticRLContractError(
                "decomposition may annotate extra fields only; it cannot mutate rollout facts"
            )
        for key, value in original.extra.items():
            if key not in annotated.extra or annotated.extra[key] != value:
                raise AgenticRLContractError(
                    "decomposition cannot delete or rewrite existing verifier evidence"
                )


def _validate_policy_targets(
    records: Sequence[TrainingTrajectoryRecord],
    targets: Sequence[PolicyTarget],
) -> None:
    rows = _index_unique(
        records,
        key=lambda item: (item.traj_uid, item.step_idx),
        label="trajectory",
    )
    indexed = _index_unique(
        targets,
        key=lambda item: (item.traj_uid, item.step_idx),
        label="policy target",
    )
    if set(rows) != set(indexed):
        raise AgenticRLContractError("RL method must emit exactly one target per transition")
    mass_by_trajectory: dict[str, float] = {}
    for key, target in indexed.items():
        row = rows[key]
        if not math.isfinite(target.advantage):
            raise AgenticRLContractError("policy target advantage must be finite")
        if not math.isfinite(target.actor_token_weight) or target.actor_token_weight <= 0:
            raise AgenticRLContractError("actor token weight must be finite and positive")
        mass_by_trajectory[row.traj_uid] = mass_by_trajectory.get(row.traj_uid, 0.0) + (
            target.actor_token_weight * row.action_token_count
        )
    for traj_uid, mass in mass_by_trajectory.items():
        if not math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AgenticRLContractError(
                f"trajectory {traj_uid!r} gradient mass is {mass}, expected 1"
            )


def _validate_update_result(
    result: PolicyUpdateResult,
    *,
    expected_step: int,
    checkpoint_before: str,
    checkpoint_after: str,
    has_learning_signal: bool,
) -> None:
    if result.optimizer_step != expected_step:
        raise AgenticRLContractError("policy optimizer step is not contiguous")
    if result.checkpoint_before_sha256 != checkpoint_before:
        raise AgenticRLContractError("policy update used a different starting checkpoint")
    if result.checkpoint_after_sha256 != checkpoint_after:
        raise AgenticRLContractError("policy update result does not bind the live checkpoint")
    _require_sha256(result.checkpoint_after_sha256, "updated checkpoint sha256")
    if (
        not math.isfinite(result.trainable_parameter_delta_l2)
        or result.trainable_parameter_delta_l2 < 0
    ):
        raise AgenticRLContractError(
            "trainable-parameter update norm must be finite and non-negative"
        )
    if has_learning_signal and result.trainable_parameter_delta_l2 == 0:
        raise AgenticRLContractError(
            "optimizer did not produce a measurable trainable-parameter update"
        )
    if not has_learning_signal and result.trainable_parameter_delta_l2 > 0:
        raise AgenticRLContractError(
            "optimizer changed trainable parameters without an advantage signal"
        )
    if any(not math.isfinite(value) for value in result.metrics.values()):
        raise AgenticRLContractError("policy update metrics must be finite")


def _validate_evaluation(
    evaluation: EvaluationResult,
    *,
    arm: ExperimentArm,
    policy: TrainablePolicy,
) -> None:
    _require_component(evaluation.task_suite, arm.eval_task_suite, "evaluation task suite")
    if evaluation.task_manifest_sha256 != arm.eval_manifest_sha256:
        raise AgenticRLContractError("evaluation used a different task manifest")
    if evaluation.policy_checkpoint_sha256 != policy.checkpoint_sha256:
        raise AgenticRLContractError("evaluation did not use the materialized trained checkpoint")
    if evaluation.task_count < 1 or not 0 <= evaluation.success_count <= evaluation.task_count:
        raise AgenticRLContractError("evaluation task/success counts are invalid")
    if not math.isfinite(evaluation.mean_action_steps) or evaluation.mean_action_steps < 0:
        raise AgenticRLContractError("evaluation mean action steps are invalid")
    if evaluation.episodes_per_task < 1:
        raise AgenticRLContractError("evaluation episodes_per_task must be positive")
    temperature = evaluation.sampling_temperature
    if temperature is not None and (not math.isfinite(temperature) or temperature < 0.0):
        raise AgenticRLContractError("evaluation sampling temperature is invalid")
    expected_stochastic = temperature is not None and temperature > 0.0
    if evaluation.stochastic_sampling != expected_stochastic:
        raise AgenticRLContractError(
            "evaluation stochastic-sampling flag disagrees with temperature"
        )
    horizon_tasks = 0
    horizon_successes = 0
    for counts in evaluation.success_by_horizon.values():
        tasks = int(counts.get("tasks", -1))
        successes = int(counts.get("successes", -1))
        if tasks < 0 or not 0 <= successes <= tasks:
            raise AgenticRLContractError("evaluation horizon counts are invalid")
        horizon_tasks += tasks
        horizon_successes += successes
    if horizon_tasks != evaluation.task_count or horizon_successes != evaluation.success_count:
        raise AgenticRLContractError("evaluation horizon strata do not sum to the reported totals")
