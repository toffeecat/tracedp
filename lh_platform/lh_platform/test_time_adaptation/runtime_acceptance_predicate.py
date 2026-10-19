"""Durable acceptance conditions kept separate from their check programs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

ACCEPTANCE_PREDICATE_IMPLEMENTATION_STATUS = "implementation_unverified"
EXPOSED_DEVELOPMENT_CANARY_RUN_CLASS = "exposed_development_canary"
FORMAL_EXPERIMENT_RUN_CLASS = "formal_experiment"


class AcceptancePredicateProvenance(StrEnum):
    """State whether a condition is fixed by the task or proposed by the Actor."""

    PUBLIC_CONTRACT = "public_contract"
    ACTOR_INFERRED = "actor_inferred"


class AcceptancePredicateAggregation(StrEnum):
    """Describe how a predicate contributes to its owning task's residual state."""

    ALL = "all"
    ANY = "any"
    ADVISORY = "advisory"


class AcceptancePredicateState(StrEnum):
    """Track current evidence maturity without asserting official task success."""

    NEVER_CHECKED = "never_checked"
    CURRENT_SUPPORTED = "current_supported"
    CURRENT_REFUTED = "current_refuted"
    CURRENT_UNKNOWN = "current_unknown"
    STALE = "stale"


class _TaskNodeLike(Protocol):
    task_id: str
    title: str
    public_requirement: bool


@dataclass(frozen=True)
class AcceptancePredicate:
    """One stable residual condition and its latest evidence/incident pointers."""

    predicate_id: str
    task_id: str
    proposition: str
    provenance: AcceptancePredicateProvenance
    source_start: int
    source_end: int
    source_excerpt: str
    resource_ids: tuple[str, ...]
    aggregation: AcceptancePredicateAggregation
    state: AcceptancePredicateState = AcceptancePredicateState.NEVER_CHECKED
    latest_evidence_id: str | None = None
    latest_incident_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.predicate_id
            or not self.task_id
            or not self.proposition.strip()
            or not self.source_excerpt
            or type(self.source_start) is not int
            or type(self.source_end) is not int
            or self.source_start < 0
            or self.source_end <= self.source_start
            or self.source_end - self.source_start != len(self.source_excerpt)
        ):
            raise ValueError("acceptance predicate identity or source span is invalid")
        if not self.resource_ids:
            raise ValueError("acceptance predicate requires an observable resource")
        if len(self.resource_ids) != len(set(self.resource_ids)):
            raise ValueError("acceptance predicate resources must be unique")
        if self.provenance is AcceptancePredicateProvenance.ACTOR_INFERRED and (
            self.aggregation is not AcceptancePredicateAggregation.ADVISORY
        ):
            raise ValueError("Actor-inferred predicates cannot redefine required work")
        if self.state is AcceptancePredicateState.NEVER_CHECKED and (
            self.latest_evidence_id is not None or self.latest_incident_id is not None
        ):
            raise ValueError("unchecked acceptance predicate cannot cite evidence")
        if self.state is not AcceptancePredicateState.NEVER_CHECKED and (
            not self.latest_evidence_id
        ):
            raise ValueError("checked acceptance predicate requires latest evidence")

    def with_evidence(
        self,
        *,
        state: AcceptancePredicateState,
        evidence_id: str,
    ) -> AcceptancePredicate:
        """Advance maturity while leaving the fixed condition definition unchanged."""

        if state is AcceptancePredicateState.NEVER_CHECKED or not evidence_id:
            raise ValueError("predicate evidence transition is invalid")
        return replace(self, state=state, latest_evidence_id=evidence_id)

    def with_incident(self, incident_id: str) -> AcceptancePredicate:
        """Attach the incident that currently explains a refuted condition."""

        if not incident_id:
            raise ValueError("predicate incident identity is required")
        return replace(self, latest_incident_id=incident_id)


_PUBLIC_MANIFEST_KEYS = {
    "predicate_id",
    "task_id",
    "proposition",
    "source_start",
    "source_end",
    "source_excerpt",
    "resource_ids",
    "aggregation",
}


def acceptance_predicate_payload(value: AcceptancePredicate) -> dict[str, object]:
    """Serialize every predicate field so replay can reject silent schema drift."""

    return {
        "predicate_id": value.predicate_id,
        "task_id": value.task_id,
        "proposition": value.proposition,
        "provenance": value.provenance.value,
        "source_start": value.source_start,
        "source_end": value.source_end,
        "source_excerpt": value.source_excerpt,
        "resource_ids": list(value.resource_ids),
        "aggregation": value.aggregation.value,
        "state": value.state.value,
        "latest_evidence_id": value.latest_evidence_id,
        "latest_incident_id": value.latest_incident_id,
    }


def decode_acceptance_predicate(value: object) -> AcceptancePredicate:
    """Decode the exact durable shape rather than accepting partial state."""

    expected = _PUBLIC_MANIFEST_KEYS | {
        "provenance",
        "state",
        "latest_evidence_id",
        "latest_incident_id",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("acceptance predicate payload is invalid")
    resource_ids = value["resource_ids"]
    if not isinstance(resource_ids, list) or any(
        not isinstance(item, str) or not item for item in resource_ids
    ):
        raise ValueError("acceptance predicate resources are invalid")
    return AcceptancePredicate(
        predicate_id=value["predicate_id"],
        task_id=value["task_id"],
        proposition=value["proposition"],
        provenance=AcceptancePredicateProvenance(value["provenance"]),
        source_start=value["source_start"],
        source_end=value["source_end"],
        source_excerpt=value["source_excerpt"],
        resource_ids=tuple(resource_ids),
        aggregation=AcceptancePredicateAggregation(value["aggregation"]),
        state=AcceptancePredicateState(value["state"]),
        latest_evidence_id=value["latest_evidence_id"],
        latest_incident_id=value["latest_incident_id"],
    )


def compile_public_acceptance_manifest(
    *,
    instruction: str,
    task_nodes: Sequence[_TaskNodeLike],
    resource_ids: Sequence[str],
    manifest: Sequence[Mapping[str, object]],
) -> tuple[AcceptancePredicate, ...]:
    """Compile only explicit, source-positioned public predicates.

    Arbitrary prose is deliberately not interpreted. Exact coordinates prove
    where a declared condition came from, while semantic correctness remains a
    task-author and independent-review responsibility.
    """

    public_tasks = {
        node.task_id: node for node in task_nodes if node.public_requirement
    }
    known_resources = set(resource_ids)
    compiled: list[AcceptancePredicate] = []
    for row in manifest:
        if not isinstance(row, Mapping) or set(row) != _PUBLIC_MANIFEST_KEYS:
            raise ValueError("public acceptance manifest schema is invalid")
        raw_resources = row["resource_ids"]
        if (
            not isinstance(row["predicate_id"], str)
            or not isinstance(row["task_id"], str)
            or not isinstance(row["proposition"], str)
            or type(row["source_start"]) is not int
            or type(row["source_end"]) is not int
            or not isinstance(row["source_excerpt"], str)
            or not isinstance(row["aggregation"], str)
            or not isinstance(raw_resources, list)
            or any(not isinstance(item, str) or not item for item in raw_resources)
        ):
            raise ValueError("public acceptance manifest fields are invalid")
        predicate = AcceptancePredicate(
            predicate_id=row["predicate_id"],
            task_id=row["task_id"],
            proposition=row["proposition"],
            provenance=AcceptancePredicateProvenance.PUBLIC_CONTRACT,
            source_start=row["source_start"],
            source_end=row["source_end"],
            source_excerpt=row["source_excerpt"],
            resource_ids=tuple(raw_resources),
            aggregation=AcceptancePredicateAggregation(row["aggregation"]),
        )
        if predicate.task_id not in public_tasks:
            raise ValueError("public acceptance predicate must own a public task")
        if instruction[predicate.source_start : predicate.source_end] != (
            predicate.source_excerpt
        ):
            raise ValueError("public acceptance predicate source span differs")
        if not set(predicate.resource_ids) <= known_resources:
            raise ValueError("public acceptance predicate references unknown resources")
        compiled.append(predicate)
    identities = [item.predicate_id for item in compiled]
    if len(identities) != len(set(identities)):
        raise ValueError("public acceptance predicate ids must be unique")
    definitions = [
        (
            item.task_id,
            item.proposition,
            item.source_start,
            item.source_end,
            item.source_excerpt,
            item.resource_ids,
        )
        for item in compiled
    ]
    if len(definitions) != len(set(definitions)):
        raise ValueError("public acceptance predicate definitions must be unique")
    return tuple(compiled)


def derive_actor_inferred_predicate(
    *,
    registration_action_id: str,
    task_id: str,
    public_instruction: str,
    proposition: str,
    source_excerpt: str,
    resource_ids: Sequence[str],
) -> AcceptancePredicate:
    """Ground one Actor proposal in a unique public span without upgrading it."""

    source_start = public_instruction.find(source_excerpt)
    if (
        source_start < 0
        or public_instruction.find(source_excerpt, source_start + 1) >= 0
    ):
        raise ValueError("Actor predicate source excerpt is absent or ambiguous")
    return AcceptancePredicate(
        predicate_id=f"actor-predicate:{registration_action_id}",
        task_id=task_id,
        proposition=proposition,
        provenance=AcceptancePredicateProvenance.ACTOR_INFERRED,
        source_start=source_start,
        source_end=source_start + len(source_excerpt),
        source_excerpt=source_excerpt,
        resource_ids=tuple(resource_ids),
        aggregation=AcceptancePredicateAggregation.ADVISORY,
    )


__all__ = [
    "ACCEPTANCE_PREDICATE_IMPLEMENTATION_STATUS",
    "AcceptancePredicate",
    "AcceptancePredicateAggregation",
    "AcceptancePredicateProvenance",
    "AcceptancePredicateState",
    "EXPOSED_DEVELOPMENT_CANARY_RUN_CLASS",
    "FORMAL_EXPERIMENT_RUN_CLASS",
    "acceptance_predicate_payload",
    "compile_public_acceptance_manifest",
    "decode_acceptance_predicate",
    "derive_actor_inferred_predicate",
]
