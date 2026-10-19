"""Bind dataset semantics to one shared, ordered Runtime action lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    ActionExecutionSummary,
    ActionPhase,
    EvidenceState,
    PortableRuntimeKernel,
    ResourceObservation,
    RuntimeAtomKnowledge,
    RuntimeStateAtom,
    RuntimeStateFrame,
    StateTransition,
    TaskNode,
)

RUNTIME_ADAPTER_LIFECYCLE_IMPLEMENTATION_STATUS = "implementation_unverified"


class RuntimeCapabilityKind(StrEnum):
    """Name environment operations without implying unsupported replay semantics."""

    OBSERVE_PUBLIC_STATE = "observe_public_state"
    CONTINUE_SESSION = "continue_session"
    INTERCEPT_ACTION = "intercept_action"
    EXECUTE_ACTION = "execute_action"
    CANCEL_ACTION = "cancel_action"
    RETAIN_STATE = "retain_state"
    CLONE_STATE = "clone_state"
    RESTORE_STATE = "restore_state"
    COMPENSATE = "compensate"


class RuntimeFailureSemantics(StrEnum):
    """Describe what Runtime may conclude after a failed capability call."""

    NO_EFFECT = "no_effect"
    PARTIAL_EFFECT = "partial_effect"
    UNKNOWN_EFFECT = "unknown_effect"


class RuntimeNativeExecutionState(StrEnum):
    """Describe the receipt available from the actual environment boundary."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECEIPT_MISSING = "receipt_missing"


@dataclass(frozen=True)
class RuntimeCapabilityGrant:
    """Declare one supported operation with its scope, conditions, and failure meaning."""

    capability_id: str
    kind: RuntimeCapabilityKind
    scope_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    cost_units: int
    failure_semantics: RuntimeFailureSemantics

    def __post_init__(self) -> None:
        if not self.capability_id or not self.scope_ids or not self.conditions:
            raise ValueError("runtime capability grant is incomplete")
        if not all(isinstance(item, str) and item for item in self.scope_ids):
            raise ValueError("runtime capability scope identities are invalid")
        if not all(isinstance(item, str) and item for item in self.conditions):
            raise ValueError("runtime capability conditions are invalid")
        if len(self.scope_ids) != len(set(self.scope_ids)):
            raise ValueError("runtime capability scope identities repeat")
        if self.cost_units < 0:
            raise ValueError("runtime capability cost cannot be negative")


@dataclass(frozen=True)
class RuntimeTaskInstanceSpec:
    """Bind a public task instance to exactly one dataset adapter."""

    adapter_id: str
    dataset_family: str
    instance_id: str
    environment_session_id: str
    public_instruction: str
    declared_task_nodes: tuple[TaskNode, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.adapter_id
            or not self.dataset_family
            or not self.instance_id
            or not self.environment_session_id
            or not self.public_instruction.strip()
        ):
            raise ValueError("runtime task instance identity is incomplete")


@dataclass(frozen=True)
class InstantiatedRuntimeView:
    """Return the complete Kernel initialization view produced by one adapter."""

    spec: RuntimeTaskInstanceSpec
    task_nodes: tuple[TaskNode, ...]
    initial_observations: tuple[ResourceObservation, ...]
    state_frame: RuntimeStateFrame
    capabilities: tuple[RuntimeCapabilityGrant, ...]

    def __post_init__(self) -> None:
        task_ids = [node.task_id for node in self.task_nodes]
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("instantiated runtime task identities are invalid")
        atom_subjects = {atom.subject_id for atom in self.state_frame.atoms}
        resource_ids = {item.resource_id for item in self.initial_observations}
        if not atom_subjects.issuperset(resource_ids):
            raise ValueError("runtime frame omits a declared resource subject")
        capability_ids = [item.capability_id for item in self.capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("runtime capability identities repeat")


@dataclass(frozen=True)
class RuntimeActionIntent:
    """Name one proposed action and the exact capability scope it requests."""

    action_id: str
    tool_name: str
    arguments: Mapping[str, object]
    capability_id: str
    scope_ids: tuple[str, ...]
    sequence: int
    actor: str = "actor"

    def __post_init__(self) -> None:
        if (
            not self.action_id
            or not self.tool_name
            or not self.capability_id
            or not self.actor
            or self.sequence < 0
            or not self.scope_ids
            or len(self.scope_ids) != len(set(self.scope_ids))
        ):
            raise ValueError("runtime action intent is incomplete")


@dataclass(frozen=True)
class RuntimeNativePreflight:
    """Carry the existing Gateway or environment preflight result without executing."""

    allowed: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool or self.allowed == bool(self.reason):
            raise ValueError("runtime native preflight result is inconsistent")


@dataclass(frozen=True)
class RuntimeActionAdmission:
    """Return the adapter's decision without mutating the Kernel or environment."""

    action_id: str
    capability_id: str
    scope_ids: tuple[str, ...]
    allowed: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.action_id
            or not self.capability_id
            or not self.scope_ids
            or len(self.scope_ids) != len(set(self.scope_ids))
            or type(self.allowed) is not bool
            or self.allowed == bool(self.reason)
        ):
            raise ValueError("runtime action admission is inconsistent")


@dataclass(frozen=True)
class RuntimeNativeActionReceipt:
    """Describe the actual execution result before dataset-specific interpretation."""

    action_id: str
    state: RuntimeNativeExecutionState
    environment_step_executed: bool | None
    effect_scope_complete: bool
    reason: str | None = None
    execution_summary: ActionExecutionSummary | None = None

    def __post_init__(self) -> None:
        if (
            not self.action_id
            or type(self.effect_scope_complete) is not bool
            or type(self.environment_step_executed) not in {bool, type(None)}
        ):
            raise ValueError("runtime native action receipt is invalid")
        if self.state is RuntimeNativeExecutionState.COMPLETED and self.reason is not None:
            raise ValueError("completed native receipts cannot carry a failure reason")
        if self.state in {
            RuntimeNativeExecutionState.COMPLETED,
            RuntimeNativeExecutionState.FAILED,
        } and self.environment_step_executed is None:
            raise ValueError("completed and failed receipts require an execution decision")
        if self.state is not RuntimeNativeExecutionState.COMPLETED and not self.reason:
            raise ValueError("non-completed native receipts require a reason")
        if self.state is RuntimeNativeExecutionState.RECEIPT_MISSING and (
            self.environment_step_executed is not None or self.effect_scope_complete
        ):
            raise ValueError("a missing receipt cannot claim execution or effect coverage")


@dataclass(frozen=True)
class RuntimePredicateObservation:
    """Bind one explicit check result to current resources or neutral atoms."""

    evidence_id: str
    task_id: str
    proposition: str
    state: EvidenceState
    resource_ids: tuple[str, ...] = ()
    atom_ids: tuple[str, ...] = ()
    predicate_id: str | None = None
    check_id: str | None = None
    incident_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.evidence_id
            or not self.task_id
            or not self.proposition
            or self.state is EvidenceState.STALE
            or (not self.resource_ids and not self.atom_ids)
            or len(self.resource_ids) != len(set(self.resource_ids))
            or len(self.atom_ids) != len(set(self.atom_ids))
            or (self.predicate_id is None) != (self.check_id is None)
            or self.predicate_id is not None
            and self.incident_id is not None
        ):
            raise ValueError("runtime predicate observation is invalid")


@dataclass(frozen=True)
class RuntimeActionInterpretation:
    """Return one complete post-action view and at most one atomic predicate result."""

    action_id: str
    state_frame: RuntimeStateFrame
    resource_observations: tuple[ResourceObservation, ...] = ()
    predicate_observations: tuple[RuntimePredicateObservation, ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id or len(self.predicate_observations) > 1:
            raise ValueError("runtime action interpretation is invalid")


@dataclass(frozen=True)
class RuntimeManagedTransition:
    """Expose the Kernel transition caused by one adapter interpretation."""

    action_id: str
    phase: ActionPhase
    changed_resource_ids: tuple[str, ...]
    changed_atom_ids: tuple[str, ...]
    stale_evidence_ids: tuple[str, ...]
    affected_task_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeFinalProjection:
    """Expose Runtime state only; benchmark reward and success are intentionally absent."""

    adapter_id: str
    dataset_family: str
    instance_id: str
    environment_session_id: str
    state_frame: RuntimeStateFrame
    kernel_state: Mapping[str, object]
    implementation_status: str = RUNTIME_ADAPTER_LIFECYCLE_IMPLEMENTATION_STATUS

    def __post_init__(self) -> None:
        if self.implementation_status != RUNTIME_ADAPTER_LIFECYCLE_IMPLEMENTATION_STATUS:
            raise ValueError("runtime adapter lifecycle is not verified for formal use")


class RuntimeDatasetAdapter(Protocol):
    """Translate one dataset's public facts across the complete Runtime lifecycle."""

    adapter_id: str
    dataset_family: str

    def instantiate(
        self,
        *,
        spec: RuntimeTaskInstanceSpec,
        current_frame: RuntimeStateFrame,
        native_state: object,
    ) -> InstantiatedRuntimeView: ...

    def admit_action(
        self,
        *,
        spec: RuntimeTaskInstanceSpec,
        current_frame: RuntimeStateFrame,
        intent: RuntimeActionIntent,
        native_preflight: RuntimeNativePreflight,
    ) -> RuntimeActionAdmission: ...

    def interpret_action(
        self,
        *,
        spec: RuntimeTaskInstanceSpec,
        admission: RuntimeActionAdmission,
        receipt: RuntimeNativeActionReceipt,
        current_frame: RuntimeStateFrame,
        native_state: object,
        next_sequence: int,
    ) -> RuntimeActionInterpretation: ...

    def project_final(
        self,
        *,
        spec: RuntimeTaskInstanceSpec,
        current_frame: RuntimeStateFrame,
        native_state: object,
        sequence: int,
    ) -> RuntimeStateFrame: ...


class RuntimeAdapterLifecycle:
    """Enforce the shared action state machine while adapters remain pure translators."""

    def __init__(
        self,
        *,
        adapter: RuntimeDatasetAdapter,
        view: InstantiatedRuntimeView,
        kernel: PortableRuntimeKernel,
    ) -> None:
        if adapter.adapter_id != view.spec.adapter_id:
            raise ValueError("runtime lifecycle adapter identity differs")
        if adapter.dataset_family != view.spec.dataset_family:
            raise ValueError("runtime lifecycle dataset family differs")
        if kernel.session_id != view.spec.environment_session_id:
            raise ValueError("runtime lifecycle Kernel session differs")
        self.adapter = adapter
        self.view = view
        self.kernel = kernel
        self._capabilities = {item.capability_id: item for item in view.capabilities}
        self._active_admission: RuntimeActionAdmission | None = None
        if kernel.state_frame_sequence is None:
            kernel.initialize_state_frame(view.state_frame)
        else:
            current_ids = set(kernel.state_atoms)
            view_ids = {atom.atom_id for atom in view.state_frame.atoms}
            if not view_ids <= current_ids:
                raise ValueError("runtime lifecycle resumed with a different atom scope")

    @property
    def current_frame(self) -> RuntimeStateFrame:
        """Return current Kernel-owned facts without granting adapters version authority."""

        if self.kernel.state_frame_sequence is None:
            raise ValueError("runtime lifecycle has no initialized state frame")
        return RuntimeStateFrame(
            frame_id=f"kernel-frame-{self.kernel.state_frame_sequence}",
            environment_session_id=self.kernel.session_id,
            sequence=self.kernel.state_frame_sequence,
            atoms=tuple(self.kernel.state_atoms.values()),
            observation_complete=bool(self.kernel.state_frame_observation_complete),
        )

    def begin_action(
        self,
        intent: RuntimeActionIntent,
        *,
        native_preflight: RuntimeNativePreflight,
    ) -> RuntimeActionAdmission:
        """Record exactly one proposal, then block it or mark the execution boundary."""

        if self._active_admission is not None:
            raise ValueError("runtime lifecycle already has an active action")
        if intent.sequence != self.current_frame.sequence:
            raise ValueError("runtime action intent sequence differs from current state")
        capability = self._capabilities.get(intent.capability_id)
        if capability is None or not set(intent.scope_ids) <= set(capability.scope_ids):
            raise ValueError("runtime action requests an unsupported capability scope")
        admission = self.adapter.admit_action(
            spec=self.view.spec,
            current_frame=self.current_frame,
            intent=intent,
            native_preflight=native_preflight,
        )
        if (
            admission.action_id != intent.action_id
            or admission.capability_id != intent.capability_id
            or admission.scope_ids != intent.scope_ids
            or (not native_preflight.allowed and admission.allowed)
        ):
            raise ValueError("runtime adapter admission differs from its input")
        self.kernel.propose_action(
            action_id=intent.action_id,
            tool_name=intent.tool_name,
            arguments=intent.arguments,
            actor=intent.actor,
        )
        if not admission.allowed:
            self.kernel.block_action(intent.action_id, reason=str(admission.reason))
            return admission
        self.kernel.start_action(intent.action_id)
        self._active_admission = admission
        return admission

    def complete_action(
        self,
        *,
        receipt: RuntimeNativeActionReceipt,
        native_state: object,
    ) -> RuntimeManagedTransition:
        """Interpret one real receipt, update Kernel state, then admit explicit evidence."""

        admission = self._active_admission
        if admission is None or receipt.action_id != admission.action_id:
            raise ValueError("runtime action receipt has no matching active admission")
        next_sequence = self.current_frame.sequence + 1
        interpretation = self.adapter.interpret_action(
            spec=self.view.spec,
            admission=admission,
            receipt=receipt,
            current_frame=self.current_frame,
            native_state=native_state,
            next_sequence=next_sequence,
        )
        if (
            interpretation.action_id != receipt.action_id
            or interpretation.state_frame.environment_session_id != self.kernel.session_id
            or interpretation.state_frame.sequence != next_sequence
        ):
            raise ValueError("runtime action interpretation identity differs")
        changed_resources = tuple(
            item.resource_id
            for item in interpretation.resource_observations
            if self.kernel.observations.get(item.resource_id) != item
        )
        current_atoms = self.kernel.state_atoms
        changed_atoms = tuple(
            atom.atom_id
            for atom in interpretation.state_frame.atoms
            if atom.atom_id not in current_atoms
            or atom.knowledge != current_atoms[atom.atom_id].knowledge
            or atom.value != current_atoms[atom.atom_id].value
        )
        observed_effect = bool(changed_resources or changed_atoms)
        observation_complete = (
            receipt.effect_scope_complete
            and interpretation.state_frame.observation_complete
            and all(item.observed for item in interpretation.resource_observations)
        )
        phase = _action_phase(
            receipt.state,
            observed_effect=observed_effect,
            observation_complete=observation_complete,
        )
        transition = self.kernel.finish_action(
            receipt.action_id,
            phase=phase,
            observations=interpretation.resource_observations,
            state_frame=interpretation.state_frame,
            reason=receipt.reason,
            execution_summary=receipt.execution_summary,
        )
        self._active_admission = None
        evidence_ids: list[str] = []
        for observation in interpretation.predicate_observations:
            self.kernel.record_evidence(
                evidence_id=observation.evidence_id,
                task_id=observation.task_id,
                proposition=observation.proposition,
                state=observation.state,
                resource_ids=observation.resource_ids,
                atom_ids=observation.atom_ids,
                source_action_id=receipt.action_id,
                predicate_id=observation.predicate_id,
                check_id=observation.check_id,
                incident_id=observation.incident_id,
            )
            evidence_ids.append(observation.evidence_id)
        return _managed_transition(receipt.action_id, phase, transition, evidence_ids)

    def final_projection(self, *, native_state: object) -> RuntimeFinalProjection:
        """Verify the adapter's final public view without inferring official outcome."""

        if self._active_admission is not None or self.kernel.unfinished_action_ids:
            raise ValueError("runtime lifecycle cannot finalize an unfinished action")
        frame = self.adapter.project_final(
            spec=self.view.spec,
            current_frame=self.current_frame,
            native_state=native_state,
            sequence=self.current_frame.sequence,
        )
        if frame != self.current_frame:
            raise ValueError("runtime final state was not committed through the lifecycle")
        return RuntimeFinalProjection(
            adapter_id=self.adapter.adapter_id,
            dataset_family=self.adapter.dataset_family,
            instance_id=self.view.spec.instance_id,
            environment_session_id=self.kernel.session_id,
            state_frame=frame,
            kernel_state=self.kernel.public_projection(),
        )


def _action_phase(
    state: RuntimeNativeExecutionState,
    *,
    observed_effect: bool,
    observation_complete: bool,
) -> ActionPhase:
    """Apply the definition-based effect table independently of adapter labels."""

    if state is RuntimeNativeExecutionState.COMPLETED:
        return ActionPhase.COMPLETED
    if state in {
        RuntimeNativeExecutionState.CANCELLED,
        RuntimeNativeExecutionState.RECEIPT_MISSING,
    }:
        return ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
    if observed_effect:
        return ActionPhase.FAILED_AFTER_PARTIAL_EFFECT
    if observation_complete:
        return ActionPhase.FAILED_WITHOUT_EFFECT
    return ActionPhase.FAILED_WITH_UNKNOWN_EFFECT


def _managed_transition(
    action_id: str,
    phase: ActionPhase,
    transition: StateTransition,
    evidence_ids: Sequence[str],
) -> RuntimeManagedTransition:
    """Copy only durable Kernel results into the caller-facing transition."""

    return RuntimeManagedTransition(
        action_id=action_id,
        phase=phase,
        changed_resource_ids=transition.changed_resource_ids,
        changed_atom_ids=transition.changed_atom_ids,
        stale_evidence_ids=transition.stale_evidence_ids,
        affected_task_ids=transition.affected_task_ids,
        evidence_ids=tuple(evidence_ids),
    )


def instantiate_runtime_view(
    *,
    adapter: RuntimeDatasetAdapter,
    spec: RuntimeTaskInstanceSpec,
    native_state: object,
) -> InstantiatedRuntimeView:
    """Validate the adapter identity and its complete abstract initialization result."""

    if adapter.adapter_id != spec.adapter_id:
        raise ValueError("runtime task instance adapter identity differs")
    if adapter.dataset_family != spec.dataset_family:
        raise ValueError("runtime task instance dataset family differs")
    view = adapter.instantiate(spec=spec, native_state=native_state)
    if view.spec != spec:
        raise ValueError("runtime adapter returned a different task instance")
    if view.state_frame.environment_session_id != spec.environment_session_id:
        raise ValueError("runtime adapter returned a different environment session")
    if spec.declared_task_nodes and view.task_nodes != spec.declared_task_nodes:
        raise ValueError("runtime adapter changed the declared task graph")
    _validate_task_graph(view.task_nodes)
    return view


def _validate_task_graph(task_nodes: Sequence[TaskNode]) -> None:
    """Require one connected acyclic task tree before Kernel construction."""

    by_id = {node.task_id: node for node in task_nodes}
    roots = [node for node in task_nodes if node.parent_id is None]
    if len(roots) != 1:
        raise ValueError("runtime task graph must have exactly one root")
    for node in task_nodes:
        if node.parent_id is not None and node.parent_id not in by_id:
            raise ValueError("runtime task graph parent is missing")
        visited: set[str] = set()
        current = node
        while current.parent_id is not None:
            if current.task_id in visited:
                raise ValueError("runtime task graph contains a cycle")
            visited.add(current.task_id)
            current = by_id[current.parent_id]


__all__ = [
    "InstantiatedRuntimeView",
    "RUNTIME_ADAPTER_LIFECYCLE_IMPLEMENTATION_STATUS",
    "RuntimeActionAdmission",
    "RuntimeActionIntent",
    "RuntimeActionInterpretation",
    "RuntimeAdapterLifecycle",
    "RuntimeAtomKnowledge",
    "RuntimeCapabilityGrant",
    "RuntimeCapabilityKind",
    "RuntimeDatasetAdapter",
    "RuntimeFailureSemantics",
    "RuntimeFinalProjection",
    "RuntimeManagedTransition",
    "RuntimeNativeActionReceipt",
    "RuntimeNativeExecutionState",
    "RuntimeNativePreflight",
    "RuntimePredicateObservation",
    "RuntimeStateAtom",
    "RuntimeStateFrame",
    "RuntimeTaskInstanceSpec",
    "instantiate_runtime_view",
]
