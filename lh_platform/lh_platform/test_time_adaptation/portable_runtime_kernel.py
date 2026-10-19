"""Environment-neutral task, evidence, resource, and action state maintenance."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from lh_platform.test_time_adaptation.runtime_acceptance_predicate import (
    AcceptancePredicate,
    AcceptancePredicateAggregation,
    AcceptancePredicateProvenance,
    AcceptancePredicateState,
    acceptance_predicate_payload,
    decode_acceptance_predicate,
    derive_actor_inferred_predicate,
)

_SCHEMA = "portable_runtime_event_v1"
ACTION_EXECUTION_HEAD_CHARACTERS = 256
ACTION_EXECUTION_TAIL_CHARACTERS = 1024
RECENT_ACTOR_EXECUTION_WINDOW = 4


class ResourceKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    DATABASE = "database"
    DATABASE_SIDECAR = "database_sidecar"
    PROCESS = "process"
    SERVICE = "service"
    OTHER = "other"


class EvidenceState(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    STALE = "stale"


class RuntimeAtomKnowledge(StrEnum):
    """State whether a neutral environment value was observed or is unknown."""

    OBSERVED = "observed"
    UNKNOWN = "unknown"


RuntimeAtomValue = bool | int | float | str | tuple[str, ...] | None


@dataclass(frozen=True)
class RuntimeStateAtom:
    """Represent one public fact whose version is assigned only by the Kernel."""

    atom_id: str
    subject_id: str
    predicate: str
    knowledge: RuntimeAtomKnowledge
    value: RuntimeAtomValue
    source: str
    version: int | None = None

    def __post_init__(self) -> None:
        if not self.atom_id or not self.subject_id or not self.predicate or not self.source:
            raise ValueError("runtime state atom identity is incomplete")
        if not isinstance(self.value, (bool, int, float, str, tuple, type(None))):
            raise ValueError("runtime state atom value must be immutable and scalar")
        if self.knowledge is RuntimeAtomKnowledge.UNKNOWN and self.value is not None:
            raise ValueError("unknown runtime atoms cannot carry a value")
        if self.version is not None and self.version < 0:
            raise ValueError("runtime state atom version cannot be negative")
        if isinstance(self.value, tuple) and not all(
            isinstance(item, str) and item for item in self.value
        ):
            raise ValueError("runtime state atom tuple values must contain strings")


@dataclass(frozen=True)
class RuntimeStateFrame:
    """Carry one complete dataset-public observation boundary into the Kernel."""

    frame_id: str
    environment_session_id: str
    sequence: int
    atoms: tuple[RuntimeStateAtom, ...]
    observation_complete: bool = True

    def __post_init__(self) -> None:
        if (
            not self.frame_id
            or not self.environment_session_id
            or self.sequence < 0
            or type(self.observation_complete) is not bool
        ):
            raise ValueError("runtime state frame identity is incomplete")
        atom_ids = [atom.atom_id for atom in self.atoms]
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError("runtime state frame atom identities repeat")
        if any(atom.version is not None for atom in self.atoms):
            raise ValueError("dataset adapters cannot assign runtime atom versions")


class DependencyRole(StrEnum):
    CONFIRMED = "confirmed"
    HYPOTHESIZED = "hypothesized"
    ABSENT = "absent"


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    AWAITING_EVALUATION = "awaiting_evaluation"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskNodeSource(StrEnum):
    PUBLIC_CONTRACT = "public_contract"
    INFERRED_PLAN = "inferred_plan"


class OutputRequirement(StrEnum):
    ALL = "all"
    ANY = "any"


class ActionPhase(StrEnum):
    PROPOSED = "proposed"
    BLOCKED = "blocked"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED_WITHOUT_EFFECT = "failed_without_effect"
    FAILED_AFTER_PARTIAL_EFFECT = "failed_after_partial_effect"
    FAILED_WITH_UNKNOWN_EFFECT = "failed_with_unknown_effect"
    CANCELLED_WITH_UNKNOWN_EFFECT = "cancelled_with_unknown_effect"


_TERMINAL_ACTION_PHASES = {
    ActionPhase.BLOCKED,
    ActionPhase.COMPLETED,
    ActionPhase.FAILED_WITHOUT_EFFECT,
    ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
    ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
    ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT,
}


@dataclass(frozen=True)
class ResourceObservation:
    """One scoped fact about a resource, independent of retention or restore."""

    resource_id: str
    kind: ResourceKind
    observed: bool
    exists: bool | None
    size_bytes: int | None = None
    modified_time: int | None = None
    modified_time_detail: str | None = None
    change_time: int | None = None
    change_time_detail: str | None = None
    inode: int | None = None
    file_type: str | None = None
    mode: str | None = None
    group_id: str | None = None
    descendant_entry_count: int | None = None
    descendant_regular_file_count: int | None = None
    descendant_total_file_bytes: int | None = None
    descendant_latest_modified_time: str | None = None
    descendant_latest_change_time: str | None = None
    descendant_observation_complete: bool | None = None

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource identity is required")
        if not self.observed and self.exists is not None:
            raise ValueError("an unobserved resource cannot have an existence result")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("resource size cannot be negative")
        descendant_counts = (
            self.descendant_entry_count,
            self.descendant_regular_file_count,
            self.descendant_total_file_bytes,
        )
        if any(value is not None and value < 0 for value in descendant_counts):
            raise ValueError("directory descendant counts cannot be negative")
        descendant_values = (
            *descendant_counts,
            self.descendant_latest_modified_time,
            self.descendant_latest_change_time,
            self.descendant_observation_complete,
        )
        if any(value is not None for value in descendant_values) and any(
            value is None for value in descendant_values
        ):
            raise ValueError("directory descendant summary must be complete")


@dataclass(frozen=True)
class TaskNode:
    """One fixed public goal or revisable inferred child in the task graph."""

    task_id: str
    parent_id: str | None
    title: str
    prerequisite_resource_ids: tuple[str, ...] = ()
    hypothesized_prerequisite_resource_ids: tuple[str, ...] = ()
    required_output_resource_ids: tuple[str, ...] = ()
    output_requirement: OutputRequirement = OutputRequirement.ALL
    public_requirement: bool = False
    source: TaskNodeSource = TaskNodeSource.INFERRED_PLAN

    def __post_init__(self) -> None:
        if not self.task_id or not self.title:
            raise ValueError("task identity and title are required")
        if len(set(self.prerequisite_resource_ids)) != len(
            self.prerequisite_resource_ids
        ):
            raise ValueError("task prerequisites must be unique")
        if len(set(self.hypothesized_prerequisite_resource_ids)) != len(
            self.hypothesized_prerequisite_resource_ids
        ):
            raise ValueError("hypothesized task prerequisites must be unique")
        if set(self.prerequisite_resource_ids).intersection(
            self.hypothesized_prerequisite_resource_ids
        ):
            raise ValueError("confirmed and hypothesized prerequisites must differ")
        if len(set(self.required_output_resource_ids)) != len(
            self.required_output_resource_ids
        ):
            raise ValueError("task output obligations must be unique")
        if not self.required_output_resource_ids and self.output_requirement is not OutputRequirement.ALL:
            raise ValueError("tasks without outputs cannot use alternative output semantics")
        if self.public_requirement != (self.source is TaskNodeSource.PUBLIC_CONTRACT):
            raise ValueError("public task requirements must come from the public contract")


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    task_id: str
    proposition: str
    state: EvidenceState
    resource_versions: tuple[tuple[str, int], ...]
    atom_versions: tuple[tuple[str, int], ...] = ()
    dependency_claims: tuple[tuple[str, DependencyRole], ...] = ()
    dependency_claims_version_bound: bool = True
    source_action_id: str | None = None
    predicate_id: str | None = None
    check_id: str | None = None


@dataclass(frozen=True)
class PredicateIncident:
    """Keep a failed predicate's history distinct from its latest recheck."""

    incident_id: str
    task_id: str
    proposition: str
    opening_evidence_id: str
    latest_evidence_id: str
    status: str
    predicate_id: str | None = None


@dataclass(frozen=True)
class TaskRevisionRecord:
    """One evidence-bound delta applied over the immutable initial task graph."""

    task_id: str
    evidence_id: str
    dependency_role_changes: tuple[tuple[str, DependencyRole], ...]


@dataclass(frozen=True)
class ActionExecutionSummary:
    """Retain bounded Actor-visible execution evidence without hidden state."""

    content_head: str
    content_tail: str
    content_length_chars: int
    content_truncated: bool
    is_error: bool
    environment_step_executed: bool
    return_code: int | None
    requested_timeout_seconds: int | None
    effective_timeout_seconds: int | None
    process_group_containment: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.content_head, str) or not isinstance(
            self.content_tail, str
        ):
            raise ValueError("action execution summary content is invalid")
        retained = len(self.content_head) + len(self.content_tail)
        optional_integers = (
            self.return_code,
            self.requested_timeout_seconds,
            self.effective_timeout_seconds,
        )
        if (
            len(self.content_head) > ACTION_EXECUTION_HEAD_CHARACTERS
            or len(self.content_tail) > ACTION_EXECUTION_TAIL_CHARACTERS
            or type(self.content_length_chars) is not int
            or self.content_length_chars < 0
            or retained > self.content_length_chars
            or type(self.content_truncated) is not bool
            or self.content_truncated != (retained < self.content_length_chars)
            or self.content_truncated
            and len(self.content_tail) != ACTION_EXECUTION_TAIL_CHARACTERS
            or len(self.content_head) < min(
                self.content_length_chars, ACTION_EXECUTION_HEAD_CHARACTERS
            )
            or type(self.is_error) is not bool
            or type(self.environment_step_executed) is not bool
            or any(
                value is not None and type(value) is not int
                for value in optional_integers
            )
            or any(
                value is not None and value < 0
                for value in (
                    self.requested_timeout_seconds,
                    self.effective_timeout_seconds,
                )
            )
            or self.process_group_containment is not None
            and (
                not isinstance(self.process_group_containment, str)
                or not self.process_group_containment
            )
        ):
            raise ValueError("action execution summary is invalid")


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    actor: str
    phase: ActionPhase
    reason: str | None = None
    execution_summary: ActionExecutionSummary | None = None


@dataclass(frozen=True)
class ActorFailureBoundary:
    """Preserve one observed Actor failure without claiming a task root cause."""

    action_id: str
    failure_count_since_resource_change: int
    resource_versions: tuple[tuple[str, int], ...]
    candidate_task_ids: tuple[str, ...]
    association_evidence_id: str


@dataclass(frozen=True)
class StateTransition:
    changed_resource_ids: tuple[str, ...]
    stale_evidence_ids: tuple[str, ...]
    affected_task_ids: tuple[str, ...]
    changed_atom_ids: tuple[str, ...] = ()


def _execution_summary_payload(value: ActionExecutionSummary) -> dict[str, object]:
    """Serialize every admitted summary field for event replay and finalization."""

    return {
        "content_head": value.content_head,
        "content_tail": value.content_tail,
        "content_length_chars": value.content_length_chars,
        "content_truncated": value.content_truncated,
        "is_error": value.is_error,
        "environment_step_executed": value.environment_step_executed,
        "return_code": value.return_code,
        "requested_timeout_seconds": value.requested_timeout_seconds,
        "effective_timeout_seconds": value.effective_timeout_seconds,
        "process_group_containment": value.process_group_containment,
    }


def _decode_execution_summary(value: object) -> ActionExecutionSummary:
    """Reject replayed summaries with unknown, missing, or coerced fields."""

    keys = {
        "content_head",
        "content_tail",
        "content_length_chars",
        "content_truncated",
        "is_error",
        "environment_step_executed",
        "return_code",
        "requested_timeout_seconds",
        "effective_timeout_seconds",
        "process_group_containment",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("action execution summary payload is invalid")
    return ActionExecutionSummary(
        content_head=value["content_head"],
        content_tail=value["content_tail"],
        content_length_chars=value["content_length_chars"],
        content_truncated=value["content_truncated"],
        is_error=value["is_error"],
        environment_step_executed=value["environment_step_executed"],
        return_code=value["return_code"],
        requested_timeout_seconds=value["requested_timeout_seconds"],
        effective_timeout_seconds=value["effective_timeout_seconds"],
        process_group_containment=value["process_group_containment"],
    )


def _observation_payload(value: ResourceObservation) -> dict[str, object]:
    payload: dict[str, object] = {
        "resource_id": value.resource_id,
        "kind": value.kind.value,
        "observed": value.observed,
        "exists": value.exists,
        "size_bytes": value.size_bytes,
        "modified_time": value.modified_time,
        "modified_time_detail": value.modified_time_detail,
        "change_time": value.change_time,
        "change_time_detail": value.change_time_detail,
        "inode": value.inode,
        "file_type": value.file_type,
        "mode": value.mode,
        "group_id": value.group_id,
    }
    if value.descendant_entry_count is not None:
        payload.update(
            {
                "descendant_entry_count": value.descendant_entry_count,
                "descendant_regular_file_count": value.descendant_regular_file_count,
                "descendant_total_file_bytes": value.descendant_total_file_bytes,
                "descendant_latest_modified_time": value.descendant_latest_modified_time,
                "descendant_latest_change_time": value.descendant_latest_change_time,
                "descendant_observation_complete": value.descendant_observation_complete,
            }
        )
    return payload


def _decode_observation(value: object) -> ResourceObservation:
    if not isinstance(value, dict):
        raise ValueError("resource observation must be an object")
    return ResourceObservation(
        resource_id=str(value.get("resource_id", "")),
        kind=ResourceKind(value.get("kind")),
        observed=value.get("observed") is True,
        exists=value.get("exists") if type(value.get("exists")) is bool else None,
        size_bytes=(
            int(value["size_bytes"])
            if isinstance(value.get("size_bytes"), int)
            else None
        ),
        modified_time=(
            int(value["modified_time"])
            if isinstance(value.get("modified_time"), int)
            else None
        ),
        modified_time_detail=(
            str(value["modified_time_detail"])
            if value.get("modified_time_detail") is not None
            else None
        ),
        change_time=(
            int(value["change_time"])
            if isinstance(value.get("change_time"), int)
            else None
        ),
        change_time_detail=(
            str(value["change_time_detail"])
            if value.get("change_time_detail") is not None
            else None
        ),
        inode=(
            int(value["inode"])
            if isinstance(value.get("inode"), int)
            else None
        ),
        file_type=(
            str(value["file_type"]) if value.get("file_type") is not None else None
        ),
        mode=str(value["mode"]) if value.get("mode") is not None else None,
        group_id=(
            str(value["group_id"]) if value.get("group_id") is not None else None
        ),
        descendant_entry_count=(
            int(value["descendant_entry_count"])
            if isinstance(value.get("descendant_entry_count"), int)
            else None
        ),
        descendant_regular_file_count=(
            int(value["descendant_regular_file_count"])
            if isinstance(value.get("descendant_regular_file_count"), int)
            else None
        ),
        descendant_total_file_bytes=(
            int(value["descendant_total_file_bytes"])
            if isinstance(value.get("descendant_total_file_bytes"), int)
            else None
        ),
        descendant_latest_modified_time=(
            str(value["descendant_latest_modified_time"])
            if value.get("descendant_latest_modified_time") is not None
            else None
        ),
        descendant_latest_change_time=(
            str(value["descendant_latest_change_time"])
            if value.get("descendant_latest_change_time") is not None
            else None
        ),
        descendant_observation_complete=(
            value.get("descendant_observation_complete")
            if type(value.get("descendant_observation_complete")) is bool
            else None
        ),
    )


def _state_atom_payload(value: RuntimeStateAtom) -> dict[str, object]:
    """Serialize the adapter observation without inventing a dataset-specific field."""

    return {
        "atom_id": value.atom_id,
        "subject_id": value.subject_id,
        "predicate": value.predicate,
        "knowledge": value.knowledge.value,
        "value": list(value.value) if isinstance(value.value, tuple) else value.value,
        "source": value.source,
    }


def _decode_state_atom(value: object) -> RuntimeStateAtom:
    """Decode the exact neutral atom shape used by event replay."""

    expected = {"atom_id", "subject_id", "predicate", "knowledge", "value", "source"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("runtime state atom payload is invalid")
    raw_value = value["value"]
    atom_value: RuntimeAtomValue
    if isinstance(raw_value, list):
        if any(not isinstance(item, str) or not item for item in raw_value):
            raise ValueError("runtime state atom list value is invalid")
        atom_value = tuple(raw_value)
    elif isinstance(raw_value, (bool, int, float, str, type(None))):
        atom_value = raw_value
    else:
        raise ValueError("runtime state atom value is invalid")
    return RuntimeStateAtom(
        atom_id=str(value["atom_id"]),
        subject_id=str(value["subject_id"]),
        predicate=str(value["predicate"]),
        knowledge=RuntimeAtomKnowledge(value["knowledge"]),
        value=atom_value,
        source=str(value["source"]),
    )


def _state_frame_payload(value: RuntimeStateFrame) -> dict[str, object]:
    """Persist a complete frame so replay assigns the same versions independently."""

    return {
        "frame_id": value.frame_id,
        "environment_session_id": value.environment_session_id,
        "sequence": value.sequence,
        "observation_complete": value.observation_complete,
        "atoms": [_state_atom_payload(atom) for atom in value.atoms],
    }


def _decode_state_frame(value: object) -> RuntimeStateFrame:
    """Reject partial frame records before replay can alter version state."""

    expected = {
        "frame_id",
        "environment_session_id",
        "sequence",
        "observation_complete",
        "atoms",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("runtime state frame payload is invalid")
    raw_atoms = value["atoms"]
    if not isinstance(raw_atoms, list):
        raise ValueError("runtime state frame atoms are invalid")
    return RuntimeStateFrame(
        frame_id=str(value["frame_id"]),
        environment_session_id=str(value["environment_session_id"]),
        sequence=value["sequence"],
        observation_complete=value["observation_complete"],
        atoms=tuple(_decode_state_atom(item) for item in raw_atoms),
    )


class PortableRuntimeKernel:
    """Maintain one session's resource versions and dependent task state.

    The kernel accepts observations and declared dependencies. It deliberately
    does not infer a benchmark root cause or choose a repair from resource names.
    """

    def __init__(
        self,
        *,
        session_id: str,
        event_log_path: Path,
        task_nodes: Sequence[TaskNode],
        acceptance_predicates: Sequence[AcceptancePredicate] = (),
        acceptance_predicate_enabled: bool = False,
    ) -> None:
        if not session_id:
            raise ValueError("runtime session identity is required")
        self.session_id = session_id
        self.event_log_path = event_log_path
        self._initial_task_nodes = {node.task_id: node for node in task_nodes}
        self.task_nodes = dict(self._initial_task_nodes)
        if len(self.task_nodes) != len(task_nodes):
            raise ValueError("task ids must be unique")
        self._validate_task_graph()
        self._initial_acceptance_predicates = {
            item.predicate_id: item for item in acceptance_predicates
        }
        self.acceptance_predicates = dict(self._initial_acceptance_predicates)
        self.acceptance_predicate_enabled = acceptance_predicate_enabled
        self._acceptance_predicate_incident_contract = (
            "atomic_evidence_event_v1" if acceptance_predicate_enabled else None
        )
        self._pending_replayed_predicate_incident: tuple[str, str] | None = None
        if len(self.acceptance_predicates) != len(acceptance_predicates):
            raise ValueError("acceptance predicate ids must be unique")
        if any(
            item.task_id not in self.task_nodes
            or item.provenance is not AcceptancePredicateProvenance.PUBLIC_CONTRACT
            or item.state is not AcceptancePredicateState.NEVER_CHECKED
            for item in acceptance_predicates
        ):
            raise ValueError("initial acceptance predicates must be fixed public conditions")
        if acceptance_predicates and not acceptance_predicate_enabled:
            raise ValueError("acceptance predicates require their feature schema")
        self.acceptance_predicate_check_bindings: dict[str, str] = {}
        self.observations: dict[str, ResourceObservation] = {}
        self.resource_versions: dict[str, int] = {}
        self.state_atoms: dict[str, RuntimeStateAtom] = {}
        self.state_atom_versions: dict[str, int] = {}
        self.state_frame_sequence: int | None = None
        self.state_frame_observation_complete: bool | None = None
        self.evidence: dict[str, EvidenceRecord] = {}
        self.predicate_incidents: dict[str, PredicateIncident] = {}
        self.task_revisions: list[TaskRevisionRecord] = []
        self.actions: dict[str, ActionRecord] = {}
        self.actor_failure_boundaries: dict[str, ActorFailureBoundary] = {}
        self.task_statuses: dict[str, TaskStatus] = {}
        self.actor_session_closed = False
        self.official_evaluation: dict[str, object] | None = None
        self.runtime_supervision_disabled_reason: str | None = None
        self.failed_runtime_intervention_versions: set[tuple[str, int]] = set()
        self._inline_runtime_failure_markers = True
        self._structured_dependency_evidence = True
        self._event_count = 0

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        event_log_path: Path,
        task_nodes: Sequence[TaskNode],
        initial_observations: Sequence[ResourceObservation],
        acceptance_predicates: Sequence[AcceptancePredicate] = (),
        acceptance_predicate_enabled: bool = False,
    ) -> PortableRuntimeKernel:
        """Create a new durable session from one initial observation boundary."""

        if event_log_path.exists():
            raise FileExistsError(f"runtime event log already exists: {event_log_path}")
        kernel = cls(
            session_id=session_id,
            event_log_path=event_log_path,
            task_nodes=task_nodes,
            acceptance_predicates=acceptance_predicates,
            acceptance_predicate_enabled=acceptance_predicate_enabled,
        )
        kernel._apply_observations(initial_observations)
        kernel._validate_acceptance_predicate_resources()
        kernel._refresh_task_statuses()
        kernel._append_event(
            "initialized",
            {
                "session_id": session_id,
                "runtime_failure_markers": "inline_v1",
                "structured_dependency_evidence": "v1",
                "task_nodes": [
                    {
                        "task_id": node.task_id,
                        "parent_id": node.parent_id,
                        "title": node.title,
                        "prerequisite_resource_ids": list(
                            node.prerequisite_resource_ids
                        ),
                        "hypothesized_prerequisite_resource_ids": list(
                            node.hypothesized_prerequisite_resource_ids
                        ),
                        "required_output_resource_ids": list(
                            node.required_output_resource_ids
                        ),
                        "output_requirement": node.output_requirement.value,
                        "public_requirement": node.public_requirement,
                        "source": node.source.value,
                    }
                    for node in task_nodes
                ],
                **(
                    {
                        "acceptance_predicate_schema": "runtime_acceptance_predicate_v1",
                        "acceptance_predicate_incident_contract": (
                            "atomic_evidence_event_v1"
                        ),
                        "acceptance_predicates": [
                            acceptance_predicate_payload(item)
                            for item in acceptance_predicates
                        ],
                    }
                    if acceptance_predicate_enabled
                    else {}
                ),
                "observations": [
                    _observation_payload(item) for item in initial_observations
                ],
            },
        )
        return kernel

    def _validate_task_graph(self) -> None:
        """Reject missing parents and cycles before any state becomes durable."""

        for node in self.task_nodes.values():
            if node.parent_id is not None and node.parent_id not in self.task_nodes:
                raise ValueError(f"task parent is missing: {node.parent_id}")
            seen = {node.task_id}
            parent_id = node.parent_id
            while parent_id is not None:
                if parent_id in seen:
                    raise ValueError("task graph contains a cycle")
                seen.add(parent_id)
                parent_id = self.task_nodes[parent_id].parent_id

    def _validate_acceptance_predicate_resources(self) -> None:
        """Require every predicate scope to name an adapter-observed resource."""

        known_resources = set(self.resource_versions)
        if any(
            not set(item.resource_ids) <= known_resources
            for item in self.acceptance_predicates.values()
        ):
            raise ValueError("acceptance predicate references an unobserved resource")

    def _acceptance_residual_status(self, task_id: str) -> str | None:
        """Aggregate required predicates without manufacturing official completion."""

        predicates = [
            item
            for item in self.acceptance_predicates.values()
            if item.task_id == task_id
            and item.aggregation is not AcceptancePredicateAggregation.ADVISORY
        ]
        if not predicates:
            return None
        conjunctive = [
            item
            for item in predicates
            if item.aggregation is AcceptancePredicateAggregation.ALL
        ]
        alternatives = [
            item
            for item in predicates
            if item.aggregation is AcceptancePredicateAggregation.ANY
        ]
        if any(
            item.state is AcceptancePredicateState.CURRENT_REFUTED
            for item in conjunctive
        ) or alternatives and all(
            item.state is AcceptancePredicateState.CURRENT_REFUTED
            for item in alternatives
        ):
            return "blocked"
        all_conjunctive_supported = all(
            item.state is AcceptancePredicateState.CURRENT_SUPPORTED
            for item in conjunctive
        )
        alternative_supported = not alternatives or any(
            item.state is AcceptancePredicateState.CURRENT_SUPPORTED
            for item in alternatives
        )
        if all_conjunctive_supported and alternative_supported:
            return "awaiting_official_evaluation"
        return "unresolved"

    def register_or_bind_actor_check_predicate(
        self,
        *,
        registration_action_id: str,
        task_id: str,
        proposition: str,
        source_excerpt: str,
        resource_ids: Sequence[str],
    ) -> str:
        """Bind one completed Actor check without changing the public contract."""

        action = self.actions.get(registration_action_id)
        node = self.task_nodes.get(task_id)
        if (
            action is None
            or action.actor != "actor"
            or action.tool_name != "register_runtime_check"
            or action.phase is not ActionPhase.COMPLETED
            or node is None
            or not node.public_requirement
        ):
            raise ValueError("Actor predicate requires one completed public registration")
        if (
            action.arguments.get("task_id") != task_id
            or action.arguments.get("proposition") != proposition
            or action.arguments.get("source_excerpt") != source_excerpt
            or action.arguments.get("resource_ids", []) != list(resource_ids)
        ):
            raise ValueError("Actor predicate differs from its registration action")
        predicate_id, inferred = self.preview_actor_check_predicate(
            registration_action_id=registration_action_id,
            task_id=task_id,
            proposition=proposition,
            source_excerpt=source_excerpt,
            resource_ids=resource_ids,
        )
        if inferred is not None:
            self.acceptance_predicates[predicate_id] = inferred
            self._append_event(
                "acceptance_predicate_registered",
                {
                    "registration_action_id": registration_action_id,
                    "predicate": acceptance_predicate_payload(inferred),
                },
            )
        self.bind_acceptance_predicate_check(
            check_id=f"actor-check:{registration_action_id}",
            predicate_id=predicate_id,
            task_id=task_id,
            proposition=proposition,
            resource_ids=resource_ids,
            registration_action_id=registration_action_id,
        )
        return predicate_id

    def preview_actor_check_predicate(
        self,
        *,
        registration_action_id: str,
        task_id: str,
        proposition: str,
        source_excerpt: str,
        resource_ids: Sequence[str],
    ) -> tuple[str, AcceptancePredicate | None]:
        """Validate an Actor proposal before its registration action is committed."""

        node = self.task_nodes.get(task_id)
        public_roots = [
            item
            for item in self.task_nodes.values()
            if item.public_requirement and item.parent_id is None
        ]
        if node is None or not node.public_requirement or len(public_roots) != 1:
            raise ValueError("Actor predicate must bind one public task")
        inferred = derive_actor_inferred_predicate(
            registration_action_id=registration_action_id,
            task_id=task_id,
            public_instruction=public_roots[0].title,
            proposition=proposition,
            source_excerpt=source_excerpt,
            resource_ids=resource_ids,
        )
        matches = [
            item
            for item in self.acceptance_predicates.values()
            if item.provenance is AcceptancePredicateProvenance.PUBLIC_CONTRACT
            and item.task_id == inferred.task_id
            and item.proposition == inferred.proposition
            and item.source_start == inferred.source_start
            and item.source_end == inferred.source_end
            and item.source_excerpt == inferred.source_excerpt
            and item.resource_ids == inferred.resource_ids
        ]
        if len(matches) > 1:
            raise ValueError("Actor check matches multiple public predicates")
        if matches:
            return matches[0].predicate_id, None
        existing = self.acceptance_predicates.get(inferred.predicate_id)
        if existing is not None:
            if existing != inferred:
                raise ValueError("Actor predicate identity is already registered")
            # Resume may observe the predicate event before its check-binding event.
            return existing.predicate_id, None
        return inferred.predicate_id, inferred

    def bind_acceptance_predicate_check(
        self,
        *,
        check_id: str,
        predicate_id: str,
        task_id: str,
        proposition: str,
        resource_ids: Sequence[str],
        registration_action_id: str | None = None,
    ) -> None:
        """Persist a check binding only when its public contract matches exactly."""

        if not check_id or check_id in self.acceptance_predicate_check_bindings:
            raise ValueError("acceptance check identity must be new")
        predicate = self.acceptance_predicates.get(predicate_id)
        if predicate is None:
            raise ValueError("acceptance check references an unknown predicate")
        if (
            task_id != predicate.task_id
            or proposition != predicate.proposition
            or tuple(resource_ids) != predicate.resource_ids
        ):
            raise ValueError("acceptance check contract differs from its predicate")
        if registration_action_id is not None:
            action = self.actions.get(registration_action_id)
            if (
                action is None
                or action.actor != "actor"
                or action.tool_name != "register_runtime_check"
                or action.phase is not ActionPhase.COMPLETED
            ):
                raise ValueError("dynamic acceptance check lacks its registration action")
        self.acceptance_predicate_check_bindings[check_id] = predicate_id
        self._append_event(
            "acceptance_predicate_check_bound",
            {
                "check_id": check_id,
                "predicate_id": predicate_id,
                "task_id": task_id,
                "proposition": proposition,
                "resource_ids": list(resource_ids),
                "registration_action_id": registration_action_id,
            },
        )

    def predicate_id_for_check(self, check_id: str) -> str | None:
        """Return the durable condition verified by one check program."""

        return self.acceptance_predicate_check_bindings.get(check_id)

    def _append_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Append one ordered event without content hashes or hidden state."""

        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": _SCHEMA,
            "sequence": self._event_count,
            "event_type": event_type,
            "payload": dict(payload),
        }
        with self.event_log_path.open("a") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._event_count += 1

    def _open_actions(self) -> tuple[str, ...]:
        return tuple(
            action_id
            for action_id, action in self.actions.items()
            if action.phase not in _TERMINAL_ACTION_PHASES
        )

    @property
    def unfinished_action_ids(self) -> tuple[str, ...]:
        """Expose unresolved action identities for explicit interruption handling."""

        return self._open_actions()

    @property
    def initial_acceptance_predicates(self) -> tuple[AcceptancePredicate, ...]:
        """Expose the immutable public predicate definitions for finalization."""

        return tuple(self._initial_acceptance_predicates.values())

    def _apply_observations(
        self, observations: Sequence[ResourceObservation]
    ) -> tuple[str, ...]:
        resource_ids = [item.resource_id for item in observations]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("one observation boundary contains duplicate resources")
        changed: list[str] = []
        for observation in observations:
            previous = self.observations.get(observation.resource_id)
            if previous != observation:
                self.resource_versions[observation.resource_id] = (
                    self.resource_versions.get(observation.resource_id, 0) + 1
                )
                self.observations[observation.resource_id] = observation
                changed.append(observation.resource_id)
        return tuple(changed)

    def _validate_state_frame(self, frame: RuntimeStateFrame) -> None:
        """Reject an invalid frame before any resource or atom state can mutate."""

        if frame.environment_session_id != self.session_id:
            raise ValueError("runtime state frame belongs to a different session")
        expected_sequence = 0 if self.state_frame_sequence is None else self.state_frame_sequence + 1
        if frame.sequence != expected_sequence:
            raise ValueError("runtime state frame sequence is not contiguous")
        incoming = {atom.atom_id: atom for atom in frame.atoms}
        if self.state_atoms and not set(self.state_atoms) <= set(incoming):
            raise ValueError("runtime state frame omitted a previously observed atom")
        for atom_id, atom in incoming.items():
            previous = self.state_atoms.get(atom_id)
            if previous is not None and (
                previous.subject_id != atom.subject_id
                or previous.predicate != atom.predicate
            ):
                raise ValueError("runtime state atom identity changed across frames")

    def _apply_state_frame(self, frame: RuntimeStateFrame) -> tuple[str, ...]:
        """Assign atom versions from one ordered, complete dataset observation frame."""

        self._validate_state_frame(frame)
        incoming = {atom.atom_id: atom for atom in frame.atoms}
        changed: list[str] = []
        for atom_id, atom in incoming.items():
            previous = self.state_atoms.get(atom_id)
            if previous is None or (
                previous.knowledge != atom.knowledge or previous.value != atom.value
            ):
                self.state_atom_versions[atom_id] = (
                    self.state_atom_versions.get(atom_id, 0) + 1
                )
                changed.append(atom_id)
            self.state_atoms[atom_id] = atom
        self.state_frame_sequence = frame.sequence
        self.state_frame_observation_complete = frame.observation_complete
        return tuple(changed)

    def initialize_state_frame(self, frame: RuntimeStateFrame) -> tuple[str, ...]:
        """Persist the first neutral frame after the existing resource initialization."""

        if self.state_frame_sequence is not None or self.actions or self.evidence:
            raise ValueError("runtime state frame must initialize before action or evidence state")
        changed = self._apply_state_frame(frame)
        self._append_event("state_frame_initialized", {"state_frame": _state_frame_payload(frame)})
        return changed

    def refresh_state_frame(self, frame: RuntimeStateFrame, *, source: str) -> StateTransition:
        """Apply a public frame observed outside an executing action boundary."""

        if not source:
            raise ValueError("state frame refresh source is required")
        changed_atoms = self._apply_state_frame(frame)
        stale = self._invalidate_evidence((), changed_atoms)
        self._refresh_task_statuses()
        transition = StateTransition((), stale, (), changed_atoms)
        self._append_event(
            "state_frame_refreshed",
            {"source": source, "state_frame": _state_frame_payload(frame)},
        )
        return transition

    def _refresh_task_statuses(self) -> None:
        """Propagate resource availability from children toward public parents."""

        children: dict[str, list[str]] = {task_id: [] for task_id in self.task_nodes}
        for node in self.task_nodes.values():
            if node.parent_id is not None:
                children[node.parent_id].append(node.task_id)

        statuses: dict[str, TaskStatus] = {}

        def resolve(task_id: str) -> TaskStatus:
            node = self.task_nodes[task_id]
            child_statuses = [resolve(child_id) for child_id in children[task_id]]
            prerequisites = [
                self.observations.get(resource_id)
                for resource_id in node.prerequisite_resource_ids
            ]
            hypotheses = [
                self.observations.get(resource_id)
                for resource_id in node.hypothesized_prerequisite_resource_ids
            ]
            outputs = [
                self.observations.get(resource_id)
                for resource_id in node.required_output_resource_ids
            ]
            if any(
                observation is not None
                and observation.observed
                and observation.exists is False
                for observation in prerequisites
            ) or any(status is TaskStatus.BLOCKED for status in child_statuses):
                status = TaskStatus.BLOCKED
            elif any(
                observation is None or not observation.observed
                for observation in prerequisites
            ) or any(
                observation is None
                or not observation.observed
                or observation.exists is False
                for observation in hypotheses
            ) or any(status is TaskStatus.PENDING for status in child_statuses):
                status = TaskStatus.PENDING
            elif outputs and node.output_requirement is OutputRequirement.ANY and any(
                observation is not None
                and observation.observed
                and observation.exists is True
                for observation in outputs
            ):
                status = TaskStatus.AWAITING_EVALUATION
            elif outputs and any(
                observation is None or not observation.observed for observation in outputs
            ):
                status = TaskStatus.PENDING
            elif outputs and any(
                observation.exists is False for observation in outputs
            ):
                status = TaskStatus.READY
            elif outputs:
                status = TaskStatus.AWAITING_EVALUATION
            elif child_statuses and all(
                status is TaskStatus.COMPLETED for status in child_statuses
            ):
                status = TaskStatus.COMPLETED
            elif child_statuses and all(
                status in {TaskStatus.COMPLETED, TaskStatus.AWAITING_EVALUATION}
                for status in child_statuses
            ):
                status = TaskStatus.AWAITING_EVALUATION
            elif any(status is TaskStatus.RUNNING for status in child_statuses):
                status = TaskStatus.RUNNING
            else:
                status = TaskStatus.READY
            if node.public_requirement:
                residual = self._acceptance_residual_status(task_id)
                if residual == "blocked":
                    status = TaskStatus.BLOCKED
                elif residual == "awaiting_official_evaluation" and status in {
                    TaskStatus.READY,
                    TaskStatus.COMPLETED,
                    TaskStatus.AWAITING_EVALUATION,
                }:
                    status = TaskStatus.AWAITING_EVALUATION
                elif residual == "unresolved" and status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.AWAITING_EVALUATION,
                }:
                    status = TaskStatus.READY
            statuses[task_id] = status
            return status

        for task_id in self.task_nodes:
            resolve(task_id)
        public_output_scope: set[str] = set()
        for node in self.task_nodes.values():
            if not node.public_requirement or not node.required_output_resource_ids:
                continue
            current_id: str | None = node.task_id
            while current_id is not None:
                public_output_scope.add(current_id)
                current_id = self.task_nodes[current_id].parent_id
        for task_id, node in self.task_nodes.items():
            if not node.public_requirement:
                continue
            if self.official_evaluation is not None:
                statuses[task_id] = (
                    TaskStatus.COMPLETED
                    if self.official_evaluation["succeeded"] is True
                    else TaskStatus.FAILED
                )
            # Preserve a visible missing-output state after the Actor stops.
            elif (
                self.actor_session_closed
                and task_id not in public_output_scope
                and statuses[task_id] is not TaskStatus.BLOCKED
            ):
                statuses[task_id] = TaskStatus.AWAITING_EVALUATION
        self.task_statuses = statuses

    def _affected_tasks(self, changed_resources: Sequence[str]) -> tuple[str, ...]:
        changed = set(changed_resources)
        affected: list[str] = []
        for node in self.task_nodes.values():
            if changed.intersection(
                set(node.prerequisite_resource_ids).union(
                    node.hypothesized_prerequisite_resource_ids,
                    node.required_output_resource_ids,
                )
            ):
                current_id: str | None = node.task_id
                while current_id is not None:
                    if current_id not in affected:
                        affected.append(current_id)
                    current_id = self.task_nodes[current_id].parent_id
        return tuple(affected)

    def _invalidate_evidence(
        self,
        changed_resources: Sequence[str],
        changed_atoms: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Expire evidence bound to an earlier resource or neutral atom version."""

        stale: list[str] = []
        changed_set = set(changed_resources)
        changed_atom_set = set(changed_atoms)
        for evidence_id, evidence in tuple(self.evidence.items()):
            if evidence.state is EvidenceState.STALE:
                continue
            if changed_set.intersection(
                dict(evidence.resource_versions)
            ) or changed_atom_set.intersection(dict(evidence.atom_versions)):
                self.evidence[evidence_id] = replace(
                    evidence, state=EvidenceState.STALE
                )
                if evidence.predicate_id is not None:
                    predicate = self.acceptance_predicates[evidence.predicate_id]
                    if predicate.latest_evidence_id == evidence_id:
                        self.acceptance_predicates[evidence.predicate_id] = (
                            predicate.with_evidence(
                                state=AcceptancePredicateState.STALE,
                                evidence_id=evidence_id,
                            )
                        )
                stale.append(evidence_id)
        if stale:
            self._rebuild_task_nodes_from_current_evidence()
        return tuple(stale)

    def _rebuild_task_nodes_from_current_evidence(self) -> None:
        """Derive the live graph from the initial graph and non-stale revisions."""

        self.task_nodes = dict(self._initial_task_nodes)
        for revision in self.task_revisions:
            evidence = self.evidence.get(revision.evidence_id)
            if evidence is None or evidence.state is EvidenceState.STALE:
                continue
            node = self.task_nodes[revision.task_id]
            confirmed = list(node.prerequisite_resource_ids)
            hypothesized = list(node.hypothesized_prerequisite_resource_ids)
            for resource_id, role in revision.dependency_role_changes:
                if resource_id in confirmed:
                    confirmed.remove(resource_id)
                if resource_id in hypothesized:
                    hypothesized.remove(resource_id)
                if role is DependencyRole.CONFIRMED:
                    confirmed.append(resource_id)
                elif role is DependencyRole.HYPOTHESIZED:
                    hypothesized.append(resource_id)
            self.task_nodes[revision.task_id] = replace(
                node,
                prerequisite_resource_ids=tuple(confirmed),
                hypothesized_prerequisite_resource_ids=tuple(hypothesized),
            )

    def refresh_observations(
        self,
        observations: Sequence[ResourceObservation],
        *,
        source: str,
    ) -> StateTransition:
        """Apply changes observed outside the currently executing action boundary."""

        if not source:
            raise ValueError("observation refresh source is required")
        changed = self._apply_observations(observations)
        stale = self._invalidate_evidence(changed)
        self._refresh_task_statuses()
        transition = StateTransition(
            changed_resource_ids=changed,
            stale_evidence_ids=stale,
            affected_task_ids=self._affected_tasks(changed),
        )
        self._append_event(
            "observations_refreshed",
            {
                "source": source,
                "observations": [_observation_payload(item) for item in observations],
            },
        )
        return transition

    def record_evidence(
        self,
        *,
        evidence_id: str,
        task_id: str,
        proposition: str,
        state: EvidenceState,
        resource_ids: Sequence[str],
        atom_ids: Sequence[str] = (),
        dependency_claims: Mapping[str, DependencyRole | str] | None = None,
        dependency_claims_version_bound: bool = True,
        source_action_id: str | None = None,
        predicate_id: str | None = None,
        check_id: str | None = None,
        incident_id: str | None = None,
    ) -> None:
        """Bind runtime evidence to versions, or mark public contract claims static."""

        if not evidence_id or evidence_id in self.evidence or not proposition:
            raise ValueError("evidence id must be new and non-empty")
        if task_id not in self.task_nodes:
            raise ValueError("evidence task does not exist")
        self._validate_evidence_source_action(source_action_id)
        versions = []
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("evidence resources must be unique")
        for resource_id in resource_ids:
            if resource_id not in self.resource_versions:
                raise ValueError(f"evidence resource is unobserved: {resource_id}")
            versions.append((resource_id, self.resource_versions[resource_id]))
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError("evidence atoms must be unique")
        atom_versions = []
        for atom_id in atom_ids:
            if atom_id not in self.state_atom_versions:
                raise ValueError(f"evidence atom is unobserved: {atom_id}")
            atom_versions.append((atom_id, self.state_atom_versions[atom_id]))
        claims = tuple(
            sorted(
                (
                    str(resource_id),
                    role if isinstance(role, DependencyRole) else DependencyRole(role),
                )
                for resource_id, role in (dependency_claims or {}).items()
            )
        )
        claimed_resource_ids = {resource_id for resource_id, _ in claims}
        if not claimed_resource_ids <= set(self.resource_versions):
            raise ValueError("dependency claims reference unobserved resources")
        versioned_resource_ids = {resource_id for resource_id, _ in versions}
        if dependency_claims_version_bound:
            if not claimed_resource_ids <= versioned_resource_ids:
                raise ValueError(
                    "version-bound dependency claims require matching resource versions"
                )
        elif versions:
            raise ValueError("static dependency evidence cannot depend on resource versions")
        record = EvidenceRecord(
            evidence_id=evidence_id,
            task_id=task_id,
            proposition=proposition,
            state=state,
            resource_versions=tuple(versions),
            atom_versions=tuple(atom_versions),
            dependency_claims=claims,
            dependency_claims_version_bound=dependency_claims_version_bound,
            source_action_id=source_action_id,
            predicate_id=predicate_id,
            check_id=check_id,
        )
        incident_id = self._incident_id_for_predicate_evidence(
            record,
            requested_incident_id=incident_id,
        )
        self._apply_acceptance_predicate_evidence(record)
        self.evidence[evidence_id] = record
        incident_transition = None
        if incident_id is not None:
            self._apply_incident_evidence(incident_id, evidence_id)
            incident_transition = {
                "incident_id": incident_id,
                "evidence_id": evidence_id,
                "status": self.predicate_incidents[incident_id].status,
            }
        self._refresh_task_statuses()
        self._append_event(
            "evidence_recorded",
            {
                "evidence_id": evidence_id,
                "task_id": task_id,
                "proposition": proposition,
                "state": state.value,
                "resource_versions": [list(item) for item in versions],
                **(
                    {"atom_versions": [list(item) for item in atom_versions]}
                    if atom_versions
                    else {}
                ),
                "dependency_claims": {
                    resource_id: role.value for resource_id, role in claims
                },
                "dependency_claims_version_bound": dependency_claims_version_bound,
                **({"source_action_id": source_action_id} if source_action_id is not None else {}),
                **({"predicate_id": predicate_id} if predicate_id is not None else {}),
                **({"check_id": check_id} if check_id is not None else {}),
                **(
                    {"predicate_incident_transition": incident_transition}
                    if incident_transition is not None
                    else {}
                ),
            },
        )

    def _apply_acceptance_predicate_evidence(self, record: EvidenceRecord) -> None:
        """Update one predicate from evidence after validating its complete binding."""

        if record.predicate_id is None:
            if record.check_id is not None:
                raise ValueError("unscoped evidence cannot cite an acceptance check")
            return
        if record.check_id is None or record.source_action_id is None:
            raise ValueError("predicate evidence requires a bound executed check")
        if self.acceptance_predicate_check_bindings.get(record.check_id) != (
            record.predicate_id
        ):
            raise ValueError("predicate evidence check binding differs")
        predicate = self.acceptance_predicates.get(record.predicate_id)
        if predicate is None:
            raise ValueError("evidence references an unknown acceptance predicate")
        if (
            record.task_id != predicate.task_id
            or record.proposition != predicate.proposition
            or tuple(resource_id for resource_id, _ in record.resource_versions)
            != predicate.resource_ids
            or record.atom_versions
        ):
            raise ValueError("evidence differs from its acceptance predicate")
        source_action = self.actions[record.source_action_id]
        if source_action.phase not in {
            ActionPhase.COMPLETED,
            ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
            ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
        }:
            raise ValueError("predicate evidence source action has no check outcome")
        if (
            record.state is EvidenceState.SUPPORTED
            and source_action.phase is not ActionPhase.COMPLETED
        ):
            raise ValueError("supported predicate evidence requires a completed check")
        if source_action.actor == "actor" and source_action.tool_name != "bash":
            raise ValueError("predicate evidence Actor source is not a check command")
        if source_action.actor not in {"actor", "runtime"}:
            raise ValueError("predicate evidence source actor is unsupported")
        if source_action.actor == "runtime":
            # A targeted repair owns its registered postcondition check, so the
            # completed repair action is the durable source for that evidence.
            if (
                source_action.phase is not ActionPhase.COMPLETED
                or source_action.tool_name
                not in {
                    "runtime_registered_check_execution",
                    "runtime_targeted_repair_execution",
                }
                or source_action.arguments.get("check_id") != record.check_id
                or source_action.arguments.get("predicate_id") != record.predicate_id
            ):
                raise ValueError("predicate evidence runtime action differs from its check")
        state = {
            EvidenceState.SUPPORTED: AcceptancePredicateState.CURRENT_SUPPORTED,
            EvidenceState.REFUTED: AcceptancePredicateState.CURRENT_REFUTED,
            EvidenceState.UNKNOWN: AcceptancePredicateState.CURRENT_UNKNOWN,
            EvidenceState.STALE: AcceptancePredicateState.STALE,
        }[record.state]
        self.acceptance_predicates[record.predicate_id] = predicate.with_evidence(
            state=state,
            evidence_id=record.evidence_id,
        )

    def _incident_id_for_predicate_evidence(
        self,
        record: EvidenceRecord,
        *,
        requested_incident_id: str | None = None,
    ) -> str | None:
        """Choose the one incident transition that must share the evidence event."""

        if record.predicate_id is None:
            if requested_incident_id is None:
                return None
            if not requested_incident_id:
                raise ValueError("predicate incident identity is empty")
            existing = self.predicate_incidents.get(requested_incident_id)
            if existing is None and record.state is not EvidenceState.REFUTED:
                raise ValueError("a new incident must open from refuted evidence")
            return requested_incident_id
        if requested_incident_id is not None:
            raise ValueError("acceptance predicate incidents choose their own identity")
        if record.state is EvidenceState.STALE:
            raise ValueError("executed predicate evidence cannot be born stale")
        active = [
            item
            for item in self.predicate_incidents.values()
            if item.predicate_id == record.predicate_id and item.status != "resolved"
        ]
        if len(active) > 1:
            raise ValueError("acceptance predicate has multiple active incidents")
        if active:
            return active[0].incident_id
        if record.state is EvidenceState.REFUTED:
            return f"incident:{record.evidence_id}"
        return None

    def update_predicate_incident(self, incident_id: str, evidence_id: str) -> None:
        """Persist an incident transition justified by already admitted check evidence."""
        evidence = self.evidence.get(evidence_id)
        if evidence is not None and evidence.predicate_id is not None:
            raise ValueError("acceptance predicate incidents are atomic with evidence")
        self._apply_incident_evidence(incident_id, evidence_id)
        self._append_event("predicate_incident_updated", {
            "incident_id": incident_id, "evidence_id": evidence_id,
        })

    def _apply_incident_evidence(self, incident_id: str, evidence_id: str) -> None:
        """Apply the same evidence-bound transition during execution and replay."""
        if not self.evidence or next(reversed(self.evidence)) != evidence_id:
            raise ValueError("incident must consume the newest admitted evidence")
        evidence = self.evidence[evidence_id]
        previous = self.predicate_incidents.get(incident_id)
        if evidence.source_action_id is None or evidence.state is EvidenceState.STALE:
            raise ValueError("incident requires current executed check evidence")
        if previous is None:
            if not incident_id or evidence.state is not EvidenceState.REFUTED:
                raise ValueError("incident must open with a refuted predicate")
            opening = evidence_id
        else:
            first = self.evidence[previous.opening_evidence_id]
            if evidence.predicate_id is not None or first.predicate_id is not None:
                same_predicate = (
                    evidence.predicate_id is not None
                    and evidence.predicate_id == first.predicate_id
                )
            else:
                same_predicate = (
                    evidence.task_id,
                    evidence.proposition,
                    tuple(dict(evidence.resource_versions)),
                    tuple(dict(evidence.atom_versions)),
                ) == (
                    first.task_id,
                    first.proposition,
                    tuple(dict(first.resource_versions)),
                    tuple(dict(first.atom_versions)),
                )
            if not same_predicate:
                raise ValueError("incident evidence predicate or scope differs")
            if previous.status == "resolved":
                raise ValueError("a resolved incident requires a new recurrence identity")
            opening = previous.opening_evidence_id
        status = {
            EvidenceState.SUPPORTED: "resolved",
            EvidenceState.REFUTED: "open",
            EvidenceState.UNKNOWN: "unknown",
        }[evidence.state]
        self.predicate_incidents[incident_id] = PredicateIncident(
            incident_id,
            evidence.task_id,
            evidence.proposition,
            opening,
            evidence_id,
            status,
            evidence.predicate_id,
        )
        if evidence.predicate_id is not None:
            predicate = self.acceptance_predicates[evidence.predicate_id]
            self.acceptance_predicates[evidence.predicate_id] = predicate.with_incident(
                incident_id
            )
        self._refresh_task_statuses()

    def _validate_evidence_source_action(self, action_id: str | None) -> None:
        """Require an observed terminal action without treating its outcome as truth."""

        if action_id is None:
            return
        action = self.actions.get(action_id)
        if action is None or action.phase not in {
            ActionPhase.COMPLETED,
            ActionPhase.FAILED_WITHOUT_EFFECT,
            ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
            ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
            ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT,
        }:
            raise ValueError("evidence source action has no execution outcome")

    def revise_inferred_task_prerequisites(
        self,
        *,
        task_id: str,
        prerequisite_resource_ids: Sequence[str],
        hypothesized_prerequisite_resource_ids: Sequence[str],
        evidence_id: str,
    ) -> None:
        """Revise one inferred dependency claim while preserving the public goal."""

        required = tuple(prerequisite_resource_ids)
        hypothesized = tuple(hypothesized_prerequisite_resource_ids)
        revised, changes = self._validated_prerequisite_revision(
            task_id=task_id,
            prerequisite_resource_ids=required,
            hypothesized_prerequisite_resource_ids=hypothesized,
            evidence_id=evidence_id,
        )
        self.task_revisions.append(
            TaskRevisionRecord(
                task_id=task_id,
                evidence_id=evidence_id,
                dependency_role_changes=changes,
            )
        )
        self._rebuild_task_nodes_from_current_evidence()
        if self.task_nodes[task_id] != revised:
            raise RuntimeError("rebuilt task revision differs from validated state")
        self._refresh_task_statuses()
        self._append_event(
            "task_prerequisites_revised",
            {
                "task_id": task_id,
                "prerequisite_resource_ids": list(required),
                "hypothesized_prerequisite_resource_ids": list(hypothesized),
                "evidence_id": evidence_id,
                "dependency_role_changes": {
                    resource_id: role.value for resource_id, role in changes
                },
            },
        )

    def _validated_prerequisite_revision(
        self,
        *,
        task_id: str,
        prerequisite_resource_ids: Sequence[str],
        hypothesized_prerequisite_resource_ids: Sequence[str],
        evidence_id: str,
    ) -> tuple[TaskNode, tuple[tuple[str, DependencyRole], ...]]:
        """Validate that decisive evidence covers every changed dependency claim."""

        node = self.task_nodes[task_id]
        evidence = self.evidence.get(evidence_id)
        if node.source is not TaskNodeSource.INFERRED_PLAN:
            raise ValueError("public task requirements cannot be revised")
        if (
            evidence is None
            or evidence.task_id != task_id
            or evidence.state not in {EvidenceState.SUPPORTED, EvidenceState.REFUTED}
        ):
            raise ValueError("task revision requires current decisive evidence")
        required = tuple(prerequisite_resource_ids)
        hypothesized = tuple(hypothesized_prerequisite_resource_ids)
        known_resources = set(self.observations)
        if not set(required).union(hypothesized) <= known_resources:
            raise ValueError("task revision references an unobserved resource")

        def role(resource_id: str, confirmed: tuple[str, ...], uncertain: tuple[str, ...]) -> str:
            if resource_id in confirmed:
                return "confirmed"
            if resource_id in uncertain:
                return "hypothesized"
            return "absent"

        compared_resources = set(node.prerequisite_resource_ids).union(
            node.hypothesized_prerequisite_resource_ids,
            required,
            hypothesized,
        )
        changed_resources = {
            resource_id
            for resource_id in compared_resources
            if role(
                resource_id,
                node.prerequisite_resource_ids,
                node.hypothesized_prerequisite_resource_ids,
            )
            != role(resource_id, required, hypothesized)
        }
        evidence_resources = {item[0] for item in evidence.resource_versions}
        if not evidence.dependency_claims_version_bound:
            evidence_resources.update(
                resource_id for resource_id, _ in evidence.dependency_claims
            )
        if not changed_resources:
            raise ValueError("task revision does not change a dependency claim")
        if not changed_resources <= evidence_resources:
            raise ValueError("task revision evidence does not cover changed resources")
        target_roles = {
            resource_id: DependencyRole(
                role(resource_id, required, hypothesized)
            )
            for resource_id in changed_resources
        }
        evidence_claims = dict(evidence.dependency_claims)
        if self._structured_dependency_evidence and any(
            evidence_claims.get(resource_id) is not target_role
            for resource_id, target_role in target_roles.items()
        ):
            raise ValueError("task revision role differs from structured evidence")
        return (
            replace(
                node,
                prerequisite_resource_ids=required,
                hypothesized_prerequisite_resource_ids=hypothesized,
            ),
            tuple(sorted(target_roles.items())),
        )

    def propose_action(
        self,
        *,
        action_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        actor: str = "actor",
    ) -> None:
        """Record a proposal before any environment operation occurs."""

        if not action_id or action_id in self.actions:
            raise ValueError("action id must be new and non-empty")
        if self.actor_session_closed:
            raise ValueError("actor session is already closed")
        if self._open_actions():
            raise ValueError("serial runtime already has an unfinished action")
        record = ActionRecord(
            action_id=action_id,
            tool_name=tool_name,
            arguments=dict(arguments),
            actor=actor,
            phase=ActionPhase.PROPOSED,
        )
        self.actions[action_id] = record
        self._append_event(
            "action_proposed",
            {
                "action_id": action_id,
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "actor": actor,
            },
        )

    def _expire_unscoped_action_evidence(self) -> tuple[str, ...]:
        """Limit unversioned dynamic observations to their Actor boundary."""
        expired = []
        for evidence_id, evidence in tuple(self.evidence.items()):
            if (evidence.source_action_id is not None and not evidence.resource_versions
                    and evidence.state is not EvidenceState.STALE):
                self.evidence[evidence_id] = replace(evidence, state=EvidenceState.STALE)
                expired.append(evidence_id)
        if expired:
            self._rebuild_task_nodes_from_current_evidence()
            self._refresh_task_statuses()
        return tuple(expired)

    def start_action(self, action_id: str) -> None:
        """Mark the exact boundary immediately before environment execution."""

        action = self.actions[action_id]
        if action.phase is not ActionPhase.PROPOSED:
            raise ValueError("only a proposed action can start")
        self.actions[action_id] = replace(action, phase=ActionPhase.STARTED)
        expired = self._expire_unscoped_action_evidence() if action.actor == "actor" else ()
        self._append_event("action_started", {
            "action_id": action_id,
            **({"expired_unscoped_evidence_ids": list(expired)} if expired else {}),
        })

    def block_action(self, action_id: str, *, reason: str) -> None:
        """Terminate a proposal that never reached the environment."""

        action = self.actions[action_id]
        if action.phase is not ActionPhase.PROPOSED or not reason:
            raise ValueError("only a proposed action can be blocked with a reason")
        self.actions[action_id] = replace(
            action,
            phase=ActionPhase.BLOCKED,
            reason=reason,
        )
        payload: dict[str, object] = {"action_id": action_id, "reason": reason}
        marker = self._runtime_failure_marker(action, ActionPhase.BLOCKED)
        if marker is not None:
            payload["runtime_failure"] = marker
        self._append_event("action_blocked", payload)

    def record_blocked_proposal(
        self,
        *,
        action_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        reason: str,
        actor: str = "actor",
    ) -> None:
        """Record a rejected concurrent proposal without opening another action."""

        if not action_id or action_id in self.actions or not reason:
            raise ValueError("blocked proposal identity and reason are required")
        self.actions[action_id] = ActionRecord(
            action_id=action_id,
            tool_name=tool_name,
            arguments=dict(arguments),
            actor=actor,
            phase=ActionPhase.BLOCKED,
            reason=reason,
        )
        self._append_event(
            "action_blocked_proposal",
            {
                "action_id": action_id,
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "actor": actor,
                "reason": reason,
            },
        )

    def finish_action(
        self,
        action_id: str,
        *,
        phase: ActionPhase,
        observations: Sequence[ResourceObservation],
        state_frame: RuntimeStateFrame | None = None,
        reason: str | None = None,
        execution_summary: ActionExecutionSummary | None = None,
    ) -> StateTransition:
        """Consume a terminal execution receipt and its post-action observations."""

        if phase not in {
            ActionPhase.COMPLETED,
            ActionPhase.FAILED_WITHOUT_EFFECT,
            ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
            ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
            ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT,
        }:
            raise ValueError("finish phase must describe a terminal execution outcome")
        action = self.actions[action_id]
        if action.phase is not ActionPhase.STARTED:
            raise ValueError("only a started action can finish")
        if execution_summary is not None and action.actor != "actor":
            raise ValueError("only Actor actions may retain an execution summary")
        if state_frame is not None:
            self._validate_state_frame(state_frame)
        changed = self._apply_observations(observations)
        changed_atoms = self._apply_state_frame(state_frame) if state_frame is not None else ()
        stale = self._invalidate_evidence(changed, changed_atoms)
        self.actions[action_id] = replace(
            action,
            phase=phase,
            reason=reason,
            execution_summary=execution_summary,
        )
        self._refresh_task_statuses()
        transition = StateTransition(
            changed_resource_ids=changed,
            stale_evidence_ids=stale,
            affected_task_ids=self._affected_tasks(changed),
            changed_atom_ids=changed_atoms,
        )
        payload: dict[str, object] = {
            "action_id": action_id,
            "phase": phase.value,
            "reason": reason,
            "observations": [_observation_payload(item) for item in observations],
            **(
                {"state_frame": _state_frame_payload(state_frame)}
                if state_frame is not None
                else {}
            ),
        }
        marker = self._runtime_failure_marker(action, phase)
        if marker is not None:
            payload["runtime_failure"] = marker
        if execution_summary is not None:
            payload["execution_summary"] = _execution_summary_payload(
                execution_summary
            )
        self._append_event("action_finished", payload)
        return transition

    def _runtime_failure_marker(
        self, action: ActionRecord, phase: ActionPhase
    ) -> dict[str, object] | None:
        """Atomically bind a failed runtime action to its resulting resource version."""

        if action.actor != "runtime" or phase is ActionPhase.COMPLETED:
            return None
        resource_id = action.arguments.get("resource_id")
        if not isinstance(resource_id, str) or resource_id not in self.resource_versions:
            raise ValueError("runtime intervention lacks a versioned resource")
        resource_version = self.resource_versions[resource_id]
        self.failed_runtime_intervention_versions.add((resource_id, resource_version))
        return {
            "resource_id": resource_id,
            "resource_version": resource_version,
        }

    def record_actor_failure_boundary(
        self,
        *,
        action_id: str,
        failure_count_since_resource_change: int,
        candidate_task_ids: Sequence[str],
        association_evidence_id: str,
    ) -> None:
        """Bind every executed Actor failure to its resulting resource versions."""

        action = self.actions.get(action_id)
        task_ids = tuple(sorted(candidate_task_ids))
        if (
            action is None
            or action.actor != "actor"
            or action.phase not in {
                ActionPhase.FAILED_WITHOUT_EFFECT,
                ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
                ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
            }
            or action_id in self.actor_failure_boundaries
            or failure_count_since_resource_change < 1
            or len(task_ids) != len(set(task_ids))
            or any(task_id not in self.task_nodes for task_id in task_ids)
            or not association_evidence_id
        ):
            raise ValueError("Actor failure boundary is invalid")
        boundary = ActorFailureBoundary(
            action_id=action_id,
            failure_count_since_resource_change=failure_count_since_resource_change,
            resource_versions=tuple(sorted(self.resource_versions.items())),
            candidate_task_ids=task_ids,
            association_evidence_id=association_evidence_id,
        )
        self.actor_failure_boundaries[action_id] = boundary
        self._append_event(
            "actor_failure_boundary_recorded",
            {
                "action_id": action_id,
                "failure_count_since_resource_change": (
                    failure_count_since_resource_change
                ),
                "resource_versions": [list(item) for item in boundary.resource_versions],
                "candidate_task_ids": list(task_ids),
                "association_evidence_id": association_evidence_id,
            },
        )

    def record_runtime_intervention_failure(
        self, *, resource_id: str, resource_version: int
    ) -> None:
        """Persist retry suppression for one failed repair at one resource version."""

        if self.resource_versions.get(resource_id) != resource_version:
            raise ValueError("runtime intervention failure version is not current")
        key = (resource_id, resource_version)
        if key in self.failed_runtime_intervention_versions:
            return
        if self._inline_runtime_failure_markers:
            raise ValueError(
                "runtime intervention failure must originate from its terminal action"
            )
        self.failed_runtime_intervention_versions.add(key)
        self._append_event(
            "runtime_intervention_failure_recorded",
            {"resource_id": resource_id, "resource_version": resource_version},
        )

    def disable_runtime_supervision(self, *, reason: str) -> None:
        """Persist a supervisor subsystem failure across Actor-process recovery."""

        if not reason:
            raise ValueError("runtime supervision disable reason is required")
        if self.runtime_supervision_disabled_reason is not None:
            return
        self.runtime_supervision_disabled_reason = reason
        self._append_event("runtime_supervision_disabled", {"reason": reason})

    def public_projection(self) -> dict[str, object]:
        """Expose method-visible state without retained bytes or private evaluator data."""

        return {
            "schema_version": (
                "portable_runtime_public_projection_v7"
                if self.state_frame_sequence is not None
                else "portable_runtime_public_projection_v6"
            ),
            "session_id": self.session_id,
            **(
                {
                    "acceptance_predicate_schema": "runtime_acceptance_predicate_v1",
                    "acceptance_predicates": [
                        acceptance_predicate_payload(item)
                        for item in self.acceptance_predicates.values()
                    ],
                    "acceptance_predicate_check_bindings": [
                        {"check_id": check_id, "predicate_id": predicate_id}
                        for check_id, predicate_id in (
                            self.acceptance_predicate_check_bindings.items()
                        )
                    ],
                    "task_acceptance_summary": {
                        task_id: self._acceptance_residual_status(task_id)
                        for task_id in self.task_nodes
                        if self._acceptance_residual_status(task_id) is not None
                    },
                }
                if self.acceptance_predicate_enabled
                else {}
            ),
            **({"predicate_incidents": [
                {
                    "incident_id": item.incident_id, "task_id": item.task_id,
                    "proposition": item.proposition,
                    "opening_evidence_id": item.opening_evidence_id,
                    "latest_evidence_id": item.latest_evidence_id,
                    "status": "recheck_required"
                    if self.evidence[item.latest_evidence_id].state is EvidenceState.STALE
                    else item.status,
                    **(
                        {"predicate_id": item.predicate_id}
                        if item.predicate_id is not None
                        else {}
                    ),
                } for item in self.predicate_incidents.values()
            ]} if self.predicate_incidents else {}),
            "actor_session_closed": self.actor_session_closed,
            "official_evaluation": self.official_evaluation,
            "runtime_supervision_disabled_reason": (
                self.runtime_supervision_disabled_reason
            ),
            "failed_runtime_intervention_versions": [
                [resource_id, version]
                for resource_id, version in sorted(
                    self.failed_runtime_intervention_versions
                )
            ],
            "resource_versions": dict(sorted(self.resource_versions.items())),
            **(
                {
                    "state_frame_sequence": self.state_frame_sequence,
                    "state_frame_observation_complete": (
                        self.state_frame_observation_complete
                    ),
                    "state_atom_versions": dict(
                        sorted(self.state_atom_versions.items())
                    ),
                    "state_atoms": [
                        {
                            **_state_atom_payload(self.state_atoms[atom_id]),
                            "version": self.state_atom_versions[atom_id],
                        }
                        for atom_id in sorted(self.state_atoms)
                    ],
                }
                if self.state_frame_sequence is not None
                else {}
            ),
            "resources": [
                {
                    **_observation_payload(self.observations[resource_id]),
                    "version": self.resource_versions[resource_id],
                }
                for resource_id in sorted(self.observations)
            ],
            "task_statuses": {
                task_id: self.task_statuses[task_id].value
                for task_id in sorted(self.task_statuses)
            },
            "task_nodes": [
                {
                    "task_id": node.task_id,
                    "parent_id": node.parent_id,
                    "title": node.title,
                    "public_requirement": node.public_requirement,
                    "source": node.source.value,
                    "prerequisite_resource_ids": list(
                        node.prerequisite_resource_ids
                    ),
                    "hypothesized_prerequisite_resource_ids": list(
                        node.hypothesized_prerequisite_resource_ids
                    ),
                    "required_output_resource_ids": list(
                        node.required_output_resource_ids
                    ),
                    "output_requirement": node.output_requirement.value,
                    "status": self.task_statuses[node.task_id].value,
                }
                for node in self.task_nodes.values()
            ],
            "evidence": [
                {
                    "evidence_id": record.evidence_id,
                    "task_id": record.task_id,
                    "proposition": record.proposition,
                    "state": record.state.value,
                    "resource_versions": [list(item) for item in record.resource_versions],
                    **(
                        {"atom_versions": [list(item) for item in record.atom_versions]}
                        if record.atom_versions
                        else {}
                    ),
                    "dependency_claims": {
                        resource_id: role.value
                        for resource_id, role in record.dependency_claims
                    },
                    "dependency_claims_version_bound": (
                        record.dependency_claims_version_bound
                    ),
                    **({"source_action_id": record.source_action_id}
                       if record.source_action_id is not None else {}),
                    **(
                        {"predicate_id": record.predicate_id}
                        if record.predicate_id is not None
                        else {}
                    ),
                    **(
                        {"check_id": record.check_id}
                        if record.check_id is not None
                        else {}
                    ),
                }
                for record in self.evidence.values()
            ],
            "task_revisions": [
                {
                    "task_id": revision.task_id,
                    "evidence_id": revision.evidence_id,
                    "dependency_role_changes": {
                        resource_id: role.value
                        for resource_id, role in revision.dependency_role_changes
                    },
                    "active": self.evidence[revision.evidence_id].state
                    is not EvidenceState.STALE,
                }
                for revision in self.task_revisions
            ],
            "actions": [
                {
                    "action_id": record.action_id,
                    "tool_name": record.tool_name,
                    "arguments": dict(record.arguments),
                    "actor": record.actor,
                    "phase": record.phase.value,
                    "reason": record.reason,
                }
                for record in self.actions.values()
            ],
            "recent_actor_execution_summaries": [
                {
                    "action_id": record.action_id,
                    "tool_name": record.tool_name,
                    "phase": record.phase.value,
                    "execution_summary": _execution_summary_payload(
                        record.execution_summary
                    ),
                }
                for record in tuple(self.actions.values())
                if record.actor == "actor" and record.execution_summary is not None
            ][-RECENT_ACTOR_EXECUTION_WINDOW:],
        }

    def close_actor_session(self) -> None:
        """Close action generation while keeping the official result unresolved."""

        if self.actor_session_closed or self._open_actions():
            raise ValueError("actor session cannot close in its current state")
        self.actor_session_closed = True
        self._refresh_task_statuses()
        self._append_event("actor_session_closed", {})

    def resolve_interrupted_action(
        self,
        *,
        fresh_observations: Sequence[ResourceObservation],
        fresh_state_frame: RuntimeStateFrame | None = None,
    ) -> None:
        """Close one in-flight action after observing the surviving environment."""

        open_actions = self._open_actions()
        if len(open_actions) != 1:
            raise ValueError("interrupted recovery requires exactly one unfinished action")
        action_id = open_actions[0]
        action = self.actions[action_id]
        if action.phase is ActionPhase.PROPOSED:
            # A durable proposal is still before the environment boundary. Record
            # any independently observed drift, then preserve that exact lifecycle.
            self.refresh_observations(
                fresh_observations,
                source="interrupted_before_environment_execution",
            )
            if fresh_state_frame is not None:
                self.refresh_state_frame(
                    fresh_state_frame,
                    source="interrupted_before_environment_execution",
                )
            self.block_action(
                action_id,
                reason="runtime stopped before environment execution started",
            )
            return
        self.finish_action(
            action_id,
            phase=ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT,
            observations=fresh_observations,
            state_frame=fresh_state_frame,
            reason="runtime stopped before an execution receipt was durable",
        )

    def record_official_evaluation(self, *, score: float, succeeded: bool) -> None:
        """Bind the independent terminal evaluator after the actor session closes."""

        if (
            not self.actor_session_closed
            or self.official_evaluation is not None
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError("official evaluation cannot be recorded in its current state")
        self.official_evaluation = {
            "score": float(score),
            "succeeded": succeeded,
        }
        self._refresh_task_statuses()
        self._append_event(
            "official_evaluation_recorded",
            {"score": float(score), "succeeded": succeeded},
        )

    @classmethod
    def _read_rows(cls, event_log_path: Path) -> list[dict[str, Any]]:
        rows = []
        for expected_sequence, line in enumerate(event_log_path.read_text().splitlines()):
            value = json.loads(line)
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != _SCHEMA
                or value.get("sequence") != expected_sequence
                or not isinstance(value.get("payload"), dict)
            ):
                raise ValueError("runtime event sequence is not contiguous")
            rows.append(value)
        if not rows or rows[0].get("event_type") != "initialized":
            raise ValueError("runtime event log lacks initialization")
        return rows

    @classmethod
    def _replay(cls, event_log_path: Path) -> PortableRuntimeKernel:
        """Rebuild state from explicit events without appending duplicate records."""

        rows = cls._read_rows(event_log_path)
        initial = rows[0]["payload"]
        raw_nodes = initial.get("task_nodes")
        raw_observations = initial.get("observations")
        if not isinstance(raw_nodes, list) or not isinstance(raw_observations, list):
            raise ValueError("runtime initialization payload is invalid")
        nodes = tuple(
            TaskNode(
                task_id=str(row["task_id"]),
                parent_id=(
                    str(row["parent_id"])
                    if row.get("parent_id") is not None
                    else None
                ),
                title=str(row["title"]),
                prerequisite_resource_ids=tuple(row["prerequisite_resource_ids"]),
                hypothesized_prerequisite_resource_ids=tuple(
                    row.get("hypothesized_prerequisite_resource_ids", ())
                ),
                required_output_resource_ids=tuple(
                    row.get("required_output_resource_ids", ())
                ),
                output_requirement=OutputRequirement(
                    row.get("output_requirement", OutputRequirement.ALL.value)
                ),
                public_requirement=row.get("public_requirement") is True,
                source=TaskNodeSource(row["source"]),
            )
            for row in raw_nodes
        )
        raw_predicates = initial.get("acceptance_predicates", [])
        predicate_schema = initial.get("acceptance_predicate_schema")
        incident_contract = initial.get("acceptance_predicate_incident_contract")
        if (
            not isinstance(raw_predicates, list)
            or predicate_schema not in {None, "runtime_acceptance_predicate_v1"}
            or (predicate_schema is None and raw_predicates)
            or incident_contract not in {None, "atomic_evidence_event_v1"}
            or (incident_contract is not None and predicate_schema is None)
        ):
            raise ValueError("runtime acceptance predicate initialization is invalid")
        predicates = tuple(decode_acceptance_predicate(row) for row in raw_predicates)
        kernel = cls(
            session_id=str(initial.get("session_id", "")),
            event_log_path=event_log_path,
            task_nodes=nodes,
            acceptance_predicates=predicates,
            acceptance_predicate_enabled=(
                predicate_schema == "runtime_acceptance_predicate_v1"
            ),
        )
        kernel._inline_runtime_failure_markers = (
            initial.get("runtime_failure_markers") == "inline_v1"
        )
        kernel._structured_dependency_evidence = (
            initial.get("structured_dependency_evidence") == "v1"
        )
        kernel._acceptance_predicate_incident_contract = incident_contract
        kernel._apply_observations(tuple(map(_decode_observation, raw_observations)))
        kernel._validate_acceptance_predicate_resources()
        kernel._refresh_task_statuses()
        kernel._event_count = 1
        for row in rows[1:]:
            event_type = str(row["event_type"])
            payload = row["payload"]
            pending = kernel._pending_replayed_predicate_incident
            if pending is not None and (
                event_type != "predicate_incident_updated"
                or payload.get("incident_id") != pending[0]
                or payload.get("evidence_id") != pending[1]
            ):
                raise ValueError(
                    "legacy predicate evidence lacks its adjacent incident transition"
                )
            kernel._apply_replayed_event(event_type, payload)
            kernel._event_count += 1
        if kernel._pending_replayed_predicate_incident is not None:
            raise ValueError(
                "legacy predicate evidence lacks its adjacent incident transition"
            )
        return kernel

    def _apply_replayed_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Apply one already-validated event without writing it again."""

        action_id = str(payload.get("action_id", ""))
        if event_type == "state_frame_initialized":
            if set(payload) != {"state_frame"} or self.state_frame_sequence is not None:
                raise ValueError("replayed initial state frame is invalid")
            self._apply_state_frame(_decode_state_frame(payload["state_frame"]))
        elif event_type == "state_frame_refreshed":
            if set(payload) != {"source", "state_frame"} or not payload.get("source"):
                raise ValueError("replayed state frame refresh is invalid")
            changed_atoms = self._apply_state_frame(
                _decode_state_frame(payload["state_frame"])
            )
            self._invalidate_evidence((), changed_atoms)
            self._refresh_task_statuses()
        elif event_type == "evidence_recorded":
            source_action_id = payload.get("source_action_id")
            self._validate_evidence_source_action(source_action_id)
            evidence_id = str(payload.get("evidence_id", ""))
            task_id = str(payload.get("task_id", ""))
            proposition = str(payload.get("proposition", ""))
            raw_versions = payload.get("resource_versions")
            if (
                not evidence_id
                or evidence_id in self.evidence
                or task_id not in self.task_nodes
                or not proposition
                or not isinstance(raw_versions, list)
            ):
                raise ValueError("replayed evidence identity or payload is invalid")
            resource_versions = tuple(
                (str(item[0]), int(item[1])) for item in raw_versions
            )
            raw_atom_versions = payload.get("atom_versions", [])
            if not isinstance(raw_atom_versions, list):
                raise ValueError("replayed evidence atom versions are invalid")
            atom_versions = tuple(
                (str(item[0]), int(item[1])) for item in raw_atom_versions
            )
            raw_claims = payload.get("dependency_claims", {})
            if not isinstance(raw_claims, dict):
                raise ValueError("replayed structured dependency claims are invalid")
            dependency_claims = tuple(
                sorted(
                    (str(resource_id), DependencyRole(role))
                    for resource_id, role in raw_claims.items()
                )
            )
            claims_version_bound = payload.get(
                "dependency_claims_version_bound",
                bool(resource_versions),
            )
            if not isinstance(claims_version_bound, bool):
                raise ValueError("replayed dependency claim scope is invalid")
            if len(resource_versions) != len(
                {resource_id for resource_id, _ in resource_versions}
            ) or any(
                self.resource_versions.get(resource_id) != version
                for resource_id, version in resource_versions
            ):
                raise ValueError("replayed evidence resource version is not current")
            if len(atom_versions) != len(
                {atom_id for atom_id, _ in atom_versions}
            ) or any(
                self.state_atom_versions.get(atom_id) != version
                for atom_id, version in atom_versions
            ):
                raise ValueError("replayed evidence atom version is not current")
            if not {
                resource_id for resource_id, _ in dependency_claims
            } <= set(self.resource_versions):
                raise ValueError("replayed dependency claims reference unknown resources")
            claimed_resource_ids = {
                resource_id for resource_id, _ in dependency_claims
            }
            versioned_resource_ids = {
                resource_id for resource_id, _ in resource_versions
            }
            if claims_version_bound:
                if not claimed_resource_ids <= versioned_resource_ids:
                    raise ValueError(
                        "replayed version-bound claims lack resource versions"
                    )
            elif resource_versions:
                raise ValueError(
                    "replayed static dependency evidence has resource versions"
                )
            record = EvidenceRecord(
                evidence_id=evidence_id,
                task_id=task_id,
                proposition=proposition,
                state=EvidenceState(payload["state"]),
                resource_versions=resource_versions,
                atom_versions=atom_versions,
                dependency_claims=dependency_claims,
                dependency_claims_version_bound=claims_version_bound,
                source_action_id=source_action_id,
                predicate_id=(
                    str(payload["predicate_id"])
                    if payload.get("predicate_id") is not None
                    else None
                ),
                check_id=(
                    str(payload["check_id"])
                    if payload.get("check_id") is not None
                    else None
                ),
            )
            raw_incident = payload.get("predicate_incident_transition")
            requested_incident_id = (
                str(raw_incident.get("incident_id", ""))
                if isinstance(raw_incident, dict) and record.predicate_id is None
                else None
            )
            incident_id = self._incident_id_for_predicate_evidence(
                record,
                requested_incident_id=requested_incident_id,
            )
            if raw_incident is not None and (
                not isinstance(raw_incident, dict)
                or set(raw_incident) != {"incident_id", "evidence_id", "status"}
                or raw_incident.get("incident_id") != incident_id
                or raw_incident.get("evidence_id") != evidence_id
            ):
                raise ValueError("replayed atomic predicate incident is invalid")
            if (
                incident_id is not None
                and raw_incident is None
                and self._acceptance_predicate_incident_contract
                == "atomic_evidence_event_v1"
            ):
                raise ValueError("atomic predicate evidence lacks its incident transition")
            if incident_id is None and raw_incident is not None:
                raise ValueError("predicate evidence has a stray incident transition")
            self._apply_acceptance_predicate_evidence(record)
            self.evidence[record.evidence_id] = record
            if incident_id is not None and raw_incident is not None:
                self._apply_incident_evidence(incident_id, evidence_id)
                if (
                    self.predicate_incidents[incident_id].status
                    != raw_incident.get("status")
                ):
                    raise ValueError("replayed atomic predicate incident status differs")
            elif incident_id is not None:
                self._pending_replayed_predicate_incident = (
                    incident_id,
                    evidence_id,
                )
            self._refresh_task_statuses()
        elif event_type == "predicate_incident_updated":
            evidence = self.evidence.get(str(payload.get("evidence_id", "")))
            if (
                evidence is not None
                and evidence.predicate_id is not None
                and self._acceptance_predicate_incident_contract
                == "atomic_evidence_event_v1"
            ):
                raise ValueError("atomic predicate incident was persisted separately")
            self._apply_incident_evidence(payload["incident_id"], payload["evidence_id"])
            if self._pending_replayed_predicate_incident is not None:
                self._pending_replayed_predicate_incident = None
        elif event_type == "acceptance_predicate_registered":
            registration_action_id = str(payload.get("registration_action_id", ""))
            predicate = decode_acceptance_predicate(payload.get("predicate"))
            action = self.actions.get(registration_action_id)
            node = self.task_nodes.get(predicate.task_id)
            public_roots = [
                item
                for item in self.task_nodes.values()
                if item.public_requirement and item.parent_id is None
            ]
            arguments = action.arguments if action is not None else {}
            if (
                action is None
                or action.actor != "actor"
                or action.tool_name != "register_runtime_check"
                or action.phase is not ActionPhase.COMPLETED
                or node is None
                or not node.public_requirement
                or len(public_roots) != 1
                or predicate
                != derive_actor_inferred_predicate(
                    registration_action_id=registration_action_id,
                    task_id=str(arguments.get("task_id", "")),
                    public_instruction=public_roots[0].title,
                    proposition=str(arguments.get("proposition", "")),
                    source_excerpt=str(arguments.get("source_excerpt", "")),
                    resource_ids=tuple(arguments.get("resource_ids", ())),
                )
                or predicate.predicate_id in self.acceptance_predicates
            ):
                raise ValueError("replayed Actor acceptance predicate is invalid")
            self.acceptance_predicates[predicate.predicate_id] = predicate
        elif event_type == "acceptance_predicate_check_bound":
            check_id = str(payload.get("check_id", ""))
            predicate_id = str(payload.get("predicate_id", ""))
            task_id = str(payload.get("task_id", ""))
            proposition = str(payload.get("proposition", ""))
            raw_resource_ids = payload.get("resource_ids")
            registration_action_id = payload.get("registration_action_id")
            predicate = self.acceptance_predicates.get(predicate_id)
            if (
                set(payload)
                != {
                    "check_id",
                    "predicate_id",
                    "task_id",
                    "proposition",
                    "resource_ids",
                    "registration_action_id",
                }
                or not check_id
                or check_id in self.acceptance_predicate_check_bindings
                or predicate is None
                or not isinstance(raw_resource_ids, list)
                or task_id != predicate.task_id
                or proposition != predicate.proposition
                or tuple(raw_resource_ids) != predicate.resource_ids
            ):
                raise ValueError("replayed acceptance check binding is invalid")
            if registration_action_id is not None:
                action = self.actions.get(str(registration_action_id))
                if (
                    action is None
                    or action.actor != "actor"
                    or action.tool_name != "register_runtime_check"
                    or action.phase is not ActionPhase.COMPLETED
                    or check_id != f"actor-check:{registration_action_id}"
                ):
                    raise ValueError("replayed dynamic acceptance check is invalid")
                arguments = action.arguments
                public_roots = [
                    item
                    for item in self.task_nodes.values()
                    if item.public_requirement and item.parent_id is None
                ]
                if len(public_roots) != 1:
                    raise ValueError("replayed acceptance contract lacks one public root")
                inferred = derive_actor_inferred_predicate(
                    registration_action_id=str(registration_action_id),
                    task_id=str(arguments.get("task_id", "")),
                    public_instruction=public_roots[0].title,
                    proposition=str(arguments.get("proposition", "")),
                    source_excerpt=str(arguments.get("source_excerpt", "")),
                    resource_ids=tuple(arguments.get("resource_ids", ())),
                )
                matches = [
                    item.predicate_id
                    for item in self.acceptance_predicates.values()
                    if item.provenance is AcceptancePredicateProvenance.PUBLIC_CONTRACT
                    and item.task_id == inferred.task_id
                    and item.proposition == inferred.proposition
                    and item.source_start == inferred.source_start
                    and item.source_end == inferred.source_end
                    and item.source_excerpt == inferred.source_excerpt
                    and item.resource_ids == inferred.resource_ids
                ]
                expected_predicate_id = matches[0] if matches else inferred.predicate_id
                if len(matches) > 1 or predicate_id != expected_predicate_id:
                    raise ValueError("replayed dynamic acceptance binding differs")
            self.acceptance_predicate_check_bindings[check_id] = predicate_id
        elif event_type == "observations_refreshed":
            observations = tuple(
                _decode_observation(item) for item in payload["observations"]
            )
            changed = self._apply_observations(observations)
            self._invalidate_evidence(changed)
            self._refresh_task_statuses()
        elif event_type == "task_prerequisites_revised":
            task_id = str(payload["task_id"])
            revised, changes = self._validated_prerequisite_revision(
                task_id=task_id,
                prerequisite_resource_ids=tuple(payload["prerequisite_resource_ids"]),
                hypothesized_prerequisite_resource_ids=tuple(
                    payload["hypothesized_prerequisite_resource_ids"]
                ),
                evidence_id=str(payload.get("evidence_id", "")),
            )
            raw_changes = payload.get("dependency_role_changes")
            if self._structured_dependency_evidence:
                expected_changes = {
                    resource_id: role.value for resource_id, role in changes
                }
                if raw_changes != expected_changes:
                    raise ValueError("replayed task revision roles differ")
            self.task_revisions.append(
                TaskRevisionRecord(
                    task_id=task_id,
                    evidence_id=str(payload.get("evidence_id", "")),
                    dependency_role_changes=changes,
                )
            )
            self._rebuild_task_nodes_from_current_evidence()
            if self.task_nodes[task_id] != revised:
                raise ValueError("replayed task revision does not rebuild exactly")
            self._refresh_task_statuses()
        elif event_type == "action_proposed":
            self.actions[action_id] = ActionRecord(
                action_id=action_id,
                tool_name=str(payload["tool_name"]),
                arguments=dict(payload["arguments"]),
                actor=str(payload["actor"]),
                phase=ActionPhase.PROPOSED,
            )
        elif event_type == "action_blocked_proposal":
            self.actions[action_id] = ActionRecord(
                action_id=action_id,
                tool_name=str(payload["tool_name"]),
                arguments=dict(payload["arguments"]),
                actor=str(payload["actor"]),
                phase=ActionPhase.BLOCKED,
                reason=str(payload["reason"]),
            )
        elif event_type == "action_started":
            self.actions[action_id] = replace(
                self.actions[action_id], phase=ActionPhase.STARTED
            )
            if "expired_unscoped_evidence_ids" in payload:
                if self.actions[action_id].actor != "actor":
                    raise ValueError("non-Actor boundary cannot expire unscoped evidence")
                expired = self._expire_unscoped_action_evidence()
                if list(expired) != payload["expired_unscoped_evidence_ids"]:
                    raise ValueError("replayed unscoped evidence expiration differs")
        elif event_type == "action_blocked":
            action = self.actions[action_id]
            phase = ActionPhase.BLOCKED
            self._apply_replayed_runtime_failure_marker(
                action=action,
                phase=phase,
                payload=payload,
            )
            self.actions[action_id] = replace(
                action,
                phase=phase,
                reason=str(payload["reason"]),
            )
        elif event_type == "action_finished":
            observations = tuple(
                _decode_observation(item) for item in payload["observations"]
            )
            changed = self._apply_observations(observations)
            changed_atoms = (
                self._apply_state_frame(_decode_state_frame(payload["state_frame"]))
                if "state_frame" in payload
                else ()
            )
            self._invalidate_evidence(changed, changed_atoms)
            action = self.actions[action_id]
            phase = ActionPhase(payload["phase"])
            execution_summary = (
                _decode_execution_summary(payload["execution_summary"])
                if "execution_summary" in payload
                else None
            )
            if execution_summary is not None and action.actor != "actor":
                raise ValueError("replayed runtime action contains an Actor summary")
            self._apply_replayed_runtime_failure_marker(
                action=action,
                phase=phase,
                payload=payload,
            )
            self.actions[action_id] = replace(
                action,
                phase=phase,
                reason=(
                    str(payload["reason"])
                    if payload.get("reason") is not None
                    else None
                ),
                execution_summary=execution_summary,
            )
            self._refresh_task_statuses()
        elif event_type == "actor_failure_boundary_recorded":
            action = self.actions.get(action_id)
            raw_versions = payload.get("resource_versions")
            raw_task_ids = payload.get("candidate_task_ids")
            failure_count = payload.get("failure_count_since_resource_change")
            evidence_id = payload.get("association_evidence_id")
            if (
                action is None
                or action.actor != "actor"
                or action.phase not in {
                    ActionPhase.FAILED_WITHOUT_EFFECT,
                    ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
                    ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
                }
                or action_id in self.actor_failure_boundaries
                or isinstance(failure_count, bool)
                or not isinstance(failure_count, int)
                or failure_count < 1
                or not isinstance(raw_versions, list)
                or not isinstance(raw_task_ids, list)
                or not isinstance(evidence_id, str)
                or not evidence_id
            ):
                raise ValueError("replayed Actor failure boundary is invalid")
            resource_versions = tuple(
                (str(item[0]), int(item[1])) for item in raw_versions
            )
            task_ids = tuple(str(item) for item in raw_task_ids)
            if (
                resource_versions != tuple(sorted(self.resource_versions.items()))
                or task_ids != tuple(sorted(task_ids))
                or len(task_ids) != len(set(task_ids))
                or any(task_id not in self.task_nodes for task_id in task_ids)
            ):
                raise ValueError("replayed Actor failure boundary state differs")
            self.actor_failure_boundaries[action_id] = ActorFailureBoundary(
                action_id=action_id,
                failure_count_since_resource_change=failure_count,
                resource_versions=resource_versions,
                candidate_task_ids=task_ids,
                association_evidence_id=evidence_id,
            )
        elif event_type == "actor_session_closed":
            self.actor_session_closed = True
            self._refresh_task_statuses()
        elif event_type == "official_evaluation_recorded":
            self.official_evaluation = {
                "score": float(payload["score"]),
                "succeeded": payload["succeeded"] is True,
            }
            self._refresh_task_statuses()
        elif event_type == "runtime_intervention_failure_recorded":
            if self._inline_runtime_failure_markers:
                raise ValueError(
                    "standalone runtime failure events are unsupported in this log"
                )
            resource_id = str(payload["resource_id"])
            resource_version = int(payload["resource_version"])
            if self.resource_versions.get(resource_id) != resource_version:
                raise ValueError("replayed runtime failure version is not current")
            self.failed_runtime_intervention_versions.add(
                (resource_id, resource_version)
            )
        elif event_type == "runtime_supervision_disabled":
            reason = str(payload.get("reason", ""))
            if not reason or self.runtime_supervision_disabled_reason is not None:
                raise ValueError("replayed runtime supervision disable event is invalid")
            self.runtime_supervision_disabled_reason = reason
            self._refresh_task_statuses()
        else:
            raise ValueError(f"unsupported runtime event type: {event_type}")

    def _apply_replayed_runtime_failure_marker(
        self,
        *,
        action: ActionRecord,
        phase: ActionPhase,
        payload: Mapping[str, Any],
    ) -> None:
        """Validate and restore an action-terminal retry marker from one event."""

        raw_marker = payload.get("runtime_failure")
        expects_marker = action.actor == "runtime" and phase is not ActionPhase.COMPLETED
        if raw_marker is None:
            if self._inline_runtime_failure_markers and expects_marker:
                raise ValueError("runtime failure event lacks its inline version marker")
            return
        if not expects_marker or not isinstance(raw_marker, dict):
            raise ValueError("runtime failure event contains an invalid inline marker")
        resource_id = action.arguments.get("resource_id")
        if not isinstance(resource_id, str) or resource_id not in self.resource_versions:
            raise ValueError("runtime failure action lacks a versioned resource")
        expected = {
            "resource_id": resource_id,
            "resource_version": self.resource_versions[resource_id],
        }
        if raw_marker != expected:
            raise ValueError("runtime failure marker differs from the action result")
        self.failed_runtime_intervention_versions.add(
            (resource_id, self.resource_versions[resource_id])
        )

    @classmethod
    def load(cls, event_log_path: Path) -> PortableRuntimeKernel:
        """Load a closed event prefix and reject ambiguous in-flight actions."""

        kernel = cls._replay(event_log_path)
        if kernel._open_actions():
            raise ValueError("runtime event log contains an unfinished action")
        return kernel

    @classmethod
    def load_interrupted(
        cls,
        event_log_path: Path,
        *,
        fresh_observations: Sequence[ResourceObservation],
        fresh_state_frame: RuntimeStateFrame | None = None,
    ) -> PortableRuntimeKernel:
        """Close one interrupted action only after a fresh environment observation."""

        kernel = cls._replay(event_log_path)
        kernel.resolve_interrupted_action(
            fresh_observations=fresh_observations,
            fresh_state_frame=fresh_state_frame,
        )
        return kernel

    @classmethod
    def load_for_resume(
        cls,
        event_log_path: Path,
        *,
        fresh_observations: Sequence[ResourceObservation],
        fresh_state_frame: RuntimeStateFrame | None = None,
    ) -> PortableRuntimeKernel:
        """Load an open actor session and resolve at most one interrupted action."""

        kernel = cls._replay(event_log_path)
        if kernel.actor_session_closed or kernel.official_evaluation is not None:
            raise ValueError("a closed or evaluated runtime session cannot resume")
        if kernel._open_actions():
            kernel.resolve_interrupted_action(
                fresh_observations=fresh_observations,
                fresh_state_frame=fresh_state_frame,
            )
        return kernel


__all__ = [
    "ACTION_EXECUTION_HEAD_CHARACTERS",
    "ACTION_EXECUTION_TAIL_CHARACTERS",
    "AcceptancePredicate",
    "AcceptancePredicateAggregation",
    "AcceptancePredicateProvenance",
    "AcceptancePredicateState",
    "ActionPhase",
    "ActionExecutionSummary",
    "ActionRecord",
    "ActorFailureBoundary",
    "DependencyRole",
    "EvidenceRecord",
    "EvidenceState",
    "PortableRuntimeKernel",
    "ResourceKind",
    "ResourceObservation",
    "RuntimeAtomKnowledge",
    "RuntimeStateAtom",
    "RuntimeStateFrame",
    "StateTransition",
    "TaskNode",
    "TaskNodeSource",
    "TaskRevisionRecord",
    "TaskStatus",
]
