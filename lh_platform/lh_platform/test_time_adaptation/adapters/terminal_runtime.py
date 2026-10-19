"""Instantiate the abstract Runtime view from Terminal resource observations."""

from __future__ import annotations

from dataclasses import dataclass

from lh_platform.benchmarks.runtime_task_contract_compiler import (
    compile_initial_resource_task_graph,
)
from lh_platform.benchmarks.terminal_resource_driver import ResourceSpec
from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    ResourceObservation,
)
from lh_platform.test_time_adaptation.runtime_dataset_adapter import (
    InstantiatedRuntimeView,
    RuntimeActionAdmission,
    RuntimeActionIntent,
    RuntimeActionInterpretation,
    RuntimeAtomKnowledge,
    RuntimeCapabilityGrant,
    RuntimeCapabilityKind,
    RuntimeFailureSemantics,
    RuntimeNativeActionReceipt,
    RuntimeNativePreflight,
    RuntimeStateAtom,
    RuntimeStateFrame,
    RuntimeTaskInstanceSpec,
)

TERMINAL_RUNTIME_ADAPTER_V1 = "terminal_managed_runtime_adapter_v1"


@dataclass(frozen=True)
class TerminalRuntimeNativeState:
    """Carry the exact Terminal inputs already consumed by the task compiler."""

    resources: tuple[ResourceSpec, ...]
    observations: tuple[ResourceObservation, ...]
    compiler_id: str


class TerminalRuntimeDatasetAdapter:
    """Project Terminal state without changing the existing compiler or Kernel types."""

    adapter_id = TERMINAL_RUNTIME_ADAPTER_V1
    dataset_family = "terminal"

    @staticmethod
    def _state_frame(
        *,
        spec: RuntimeTaskInstanceSpec,
        native_state: TerminalRuntimeNativeState,
        sequence: int,
    ) -> RuntimeStateFrame:
        """Project the same fixed atom scope at every Terminal observation boundary."""

        atoms: list[RuntimeStateAtom] = []
        for observation in sorted(
            native_state.observations, key=lambda item: item.resource_id
        ):
            knowledge = (
                RuntimeAtomKnowledge.OBSERVED
                if observation.observed
                else RuntimeAtomKnowledge.UNKNOWN
            )
            atoms.extend(
                (
                    RuntimeStateAtom(
                        atom_id=f"resource:{observation.resource_id}:exists",
                        subject_id=observation.resource_id,
                        predicate="resource_exists",
                        knowledge=knowledge,
                        value=observation.exists if observation.observed else None,
                        source="terminal_resource_observation",
                    ),
                    RuntimeStateAtom(
                        atom_id=f"resource:{observation.resource_id}:kind",
                        subject_id=observation.resource_id,
                        predicate="resource_kind",
                        knowledge=RuntimeAtomKnowledge.OBSERVED,
                        value=observation.kind.value,
                        source="terminal_resource_configuration",
                    ),
                    RuntimeStateAtom(
                        atom_id=f"resource:{observation.resource_id}:size-bytes",
                        subject_id=observation.resource_id,
                        predicate="resource_size_bytes",
                        knowledge=(
                            RuntimeAtomKnowledge.OBSERVED
                            if observation.observed and observation.size_bytes is not None
                            else RuntimeAtomKnowledge.UNKNOWN
                        ),
                        value=(
                            observation.size_bytes
                            if observation.observed
                            and observation.size_bytes is not None
                            else None
                        ),
                        source="terminal_resource_observation",
                    ),
                )
            )
        return RuntimeStateFrame(
            frame_id=f"terminal-observation-{sequence}",
            environment_session_id=spec.environment_session_id,
            sequence=sequence,
            atoms=tuple(atoms),
            observation_complete=all(item.observed for item in native_state.observations),
        )

    def instantiate(
        self,
        *,
        spec: RuntimeTaskInstanceSpec,
        native_state: object,
    ) -> InstantiatedRuntimeView:
        """Compile the task graph and a neutral state frame from public resources."""

        if not isinstance(native_state, TerminalRuntimeNativeState):
            raise TypeError("Terminal adapter requires TerminalRuntimeNativeState")
        task_nodes = compile_initial_resource_task_graph(
            instruction=spec.public_instruction,
            resources=native_state.resources,
            initial_observations=native_state.observations,
            compiler_id=native_state.compiler_id,
        )
        resource_scope = tuple(
            sorted(observation.resource_id for observation in native_state.observations)
        )
        session_scope = (spec.environment_session_id,)
        capabilities = (
            RuntimeCapabilityGrant(
                capability_id="terminal-observe-configured-resources",
                kind=RuntimeCapabilityKind.OBSERVE_PUBLIC_STATE,
                scope_ids=resource_scope,
                conditions=("resource is in the configured public observation scope",),
                cost_units=0,
                failure_semantics=RuntimeFailureSemantics.UNKNOWN_EFFECT,
            ),
            RuntimeCapabilityGrant(
                capability_id="terminal-continue-live-session",
                kind=RuntimeCapabilityKind.CONTINUE_SESSION,
                scope_ids=session_scope,
                conditions=("the Harbor environment instance remains active",),
                cost_units=0,
                failure_semantics=RuntimeFailureSemantics.UNKNOWN_EFFECT,
            ),
            RuntimeCapabilityGrant(
                capability_id="terminal-intercept-actor-action",
                kind=RuntimeCapabilityKind.INTERCEPT_ACTION,
                scope_ids=session_scope,
                conditions=("the action crosses the managed Terminal tool gateway",),
                cost_units=0,
                failure_semantics=RuntimeFailureSemantics.NO_EFFECT,
            ),
            RuntimeCapabilityGrant(
                capability_id="terminal-execute-actor-action",
                kind=RuntimeCapabilityKind.EXECUTE_ACTION,
                scope_ids=session_scope,
                conditions=("the Harbor environment accepts the delegated tool call",),
                cost_units=0,
                failure_semantics=RuntimeFailureSemantics.UNKNOWN_EFFECT,
            ),
        )
        return InstantiatedRuntimeView(
            spec=spec,
            task_nodes=task_nodes,
            initial_observations=native_state.observations,
            state_frame=self._state_frame(spec=spec, native_state=native_state, sequence=0),
            capabilities=capabilities,
        )

    def admit_action(
        self,
        *,
        spec: RuntimeTaskInstanceSpec,
        current_frame: RuntimeStateFrame,
        intent: RuntimeActionIntent,
        native_preflight: RuntimeNativePreflight,
    ) -> RuntimeActionAdmission:
        """Preserve the Gateway preflight result without executing or widening scope."""

        del spec, current_frame
        return RuntimeActionAdmission(
            action_id=intent.action_id,
            capability_id=intent.capability_id,
            scope_ids=intent.scope_ids,
            allowed=native_preflight.allowed,
            reason=native_preflight.reason,
        )

    def interpret_action(
        self,
        *,
        spec: RuntimeTaskInstanceSpec,
        admission: RuntimeActionAdmission,
        receipt: RuntimeNativeActionReceipt,
        current_frame: RuntimeStateFrame,
        native_state: object,
        next_sequence: int,
    ) -> RuntimeActionInterpretation:
        """Project post-action resources; shared lifecycle derives the action phase."""

        del current_frame
        if not isinstance(native_state, TerminalRuntimeNativeState):
            raise TypeError("Terminal adapter requires TerminalRuntimeNativeState")
        if receipt.action_id != admission.action_id:
            raise ValueError("Terminal action receipt identity differs")
        return RuntimeActionInterpretation(
            action_id=receipt.action_id,
            state_frame=self._state_frame(
                spec=spec,
                native_state=native_state,
                sequence=next_sequence,
            ),
            resource_observations=native_state.observations,
        )

    def project_final(
        self,
        *,
        spec: RuntimeTaskInstanceSpec,
        current_frame: RuntimeStateFrame,
        native_state: object,
        sequence: int,
    ) -> RuntimeStateFrame:
        """Return the Runtime-visible Terminal state without an official result."""

        if not isinstance(native_state, TerminalRuntimeNativeState):
            raise TypeError("Terminal adapter requires TerminalRuntimeNativeState")
        del current_frame
        frame = self._state_frame(spec=spec, native_state=native_state, sequence=sequence)
        return RuntimeStateFrame(
            frame_id=f"kernel-frame-{sequence}",
            environment_session_id=frame.environment_session_id,
            sequence=frame.sequence,
            atoms=frame.atoms,
            observation_complete=frame.observation_complete,
        )


__all__ = [
    "TERMINAL_RUNTIME_ADAPTER_V1",
    "TerminalRuntimeDatasetAdapter",
    "TerminalRuntimeNativeState",
]
