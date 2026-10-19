"""Environment-verifiable goal graph and transition-label contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

GOAL_GRAPH_SCHEMA = "lh_goal_graph/v1"
GOAL_VECTOR_SCHEMA = "lh_goal_vector/v1"
TRANSITION_GOAL_LABEL_SCHEMA = "lh_transition_goal_label/v1"
ACTOR_STATE_EXPOSURE_SCHEMA = "lh_actor_state_exposure/v1"
ACTOR_AWARENESS_SCHEMA = "lh_actor_goal_awareness/v1"


class GoalGraphError(ValueError):
    """Raised when a goal graph cannot be independently verified."""


class ActorVisibility(StrEnum):
    PUBLIC = "public"
    TRAINING_ONLY = "training_only"
    AUDIT_ONLY = "audit_only"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GoalAtom:
    goal_id: str
    verifier_id: str
    invariant: bool = False
    weight: float = 1.0
    actor_visibility: ActorVisibility = ActorVisibility.PUBLIC

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_visibility", ActorVisibility(self.actor_visibility))
        if not self.goal_id.strip() or not self.verifier_id.strip():
            raise GoalGraphError("goal atom requires goal_id and verifier_id")
        if self.weight <= 0:
            raise GoalGraphError("goal atom weight must be positive")


@dataclass(frozen=True)
class SubproblemNode:
    node_id: str
    parent_id: str | None
    precondition_goal_ids: tuple[str, ...]
    effect_goal_ids: tuple[str, ...]
    label: str
    source: str = "verifier_dependency"

    def __post_init__(self) -> None:
        if not self.node_id.strip() or not self.label.strip() or not self.source.strip():
            raise GoalGraphError("subproblem node requires stable identity, label, and source")
        if not self.effect_goal_ids:
            raise GoalGraphError("subproblem node requires at least one verifiable effect")


@dataclass(frozen=True)
class GoalGraph:
    task_id: str
    root_id: str
    atoms: tuple[GoalAtom, ...]
    nodes: tuple[SubproblemNode, ...]
    edges: tuple[tuple[str, str], ...]
    schema: str = GOAL_GRAPH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GOAL_GRAPH_SCHEMA or not self.task_id.strip():
            raise GoalGraphError("unsupported or unidentified goal graph")
        atom_ids = [atom.goal_id for atom in self.atoms]
        node_ids = [node.node_id for node in self.nodes]
        if not atom_ids or len(atom_ids) != len(set(atom_ids)):
            raise GoalGraphError("goal graph atom IDs must be non-empty and unique")
        if not node_ids or len(node_ids) != len(set(node_ids)):
            raise GoalGraphError("goal graph node IDs must be non-empty and unique")
        if self.root_id not in node_ids:
            raise GoalGraphError("goal graph root is missing")
        atom_set = set(atom_ids)
        node_set = set(node_ids)
        parent_edges: list[tuple[str, str]] = []
        for node in self.nodes:
            if node.parent_id is not None and node.parent_id not in node_set:
                raise GoalGraphError("subproblem parent is missing")
            if node.node_id == self.root_id and node.parent_id is not None:
                raise GoalGraphError("goal graph root cannot have a parent")
            if node.node_id != self.root_id and node.parent_id is None:
                raise GoalGraphError("every non-root subproblem requires a parent")
            if node.parent_id is not None:
                parent_edges.append((node.parent_id, node.node_id))
            if not set((*node.precondition_goal_ids, *node.effect_goal_ids)) <= atom_set:
                raise GoalGraphError("subproblem references an unknown goal atom")
        for source, target in self.edges:
            if source not in node_set or target not in node_set or source == target:
                raise GoalGraphError("goal graph edge references invalid nodes")
        _assert_acyclic(node_set, self.edges)
        _assert_acyclic(node_set, tuple(parent_edges))

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "root_id": self.root_id,
            "atoms": sorted(
                (
                    {
                        "goal_id": atom.goal_id,
                        "verifier_id": atom.verifier_id,
                        "invariant": atom.invariant,
                        "weight": atom.weight,
                        "actor_visibility": atom.actor_visibility.value,
                    }
                    for atom in self.atoms
                ),
                key=lambda row: row["goal_id"],
            ),
            "nodes": sorted(
                (
                    {
                        "node_id": node.node_id,
                        "parent_id": node.parent_id,
                        "precondition_goal_ids": list(node.precondition_goal_ids),
                        "effect_goal_ids": list(node.effect_goal_ids),
                        "label": node.label,
                        "source": node.source,
                    }
                    for node in self.nodes
                ),
                key=lambda row: row["node_id"],
            ),
            "edges": sorted([list(edge) for edge in self.edges]),
        }


@dataclass(frozen=True)
class GoalVector:
    truth: tuple[tuple[str, bool], ...]
    achievement_latches: tuple[str, ...]
    regressed_goal_ids: tuple[str, ...]
    graph_sha256: str
    schema: str = GOAL_VECTOR_SCHEMA

    @classmethod
    def from_truth(
        cls,
        graph: GoalGraph,
        truth: Mapping[str, bool],
        *,
        previous: GoalVector | None = None,
    ) -> GoalVector:
        expected = {atom.goal_id for atom in graph.atoms}
        if set(truth) != expected or any(not isinstance(value, bool) for value in truth.values()):
            raise GoalGraphError("goal vector truth must cover every graph atom with booleans")
        if previous is not None and previous.graph_sha256 != graph.sha256:
            raise GoalGraphError("goal vector history belongs to a different graph")
        ordered = tuple(sorted((goal_id, bool(value)) for goal_id, value in truth.items()))
        before = previous.truth_map if previous is not None else {}
        regressed = tuple(
            goal_id for goal_id, value in ordered if before.get(goal_id, False) and not value
        )
        latches = set(previous.achievement_latches if previous is not None else ())
        latches.update(goal_id for goal_id, value in ordered if value)
        return cls(
            truth=ordered,
            achievement_latches=tuple(sorted(latches)),
            regressed_goal_ids=regressed,
            graph_sha256=graph.sha256,
        )

    @property
    def truth_map(self) -> dict[str, bool]:
        return dict(self.truth)

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class TransitionGoalLabel:
    action: str
    achieved_goal_ids_before: tuple[str, ...]
    achieved_goal_ids_after: tuple[str, ...]
    newly_satisfied_goal_ids: tuple[str, ...]
    regressed_goal_ids: tuple[str, ...]
    residual_goal_ids: tuple[str, ...]
    residual_goal_signature: str
    active_frontier_ids: tuple[str, ...]
    frontier_signature: str
    graph_sha256: str
    annotation_sha256: str
    schema: str = TRANSITION_GOAL_LABEL_SCHEMA


@dataclass(frozen=True)
class VerificationReceipt:
    verified: bool
    compiler_id: str
    annotation_sha256: str
    verifier_input_sha256: str
    source: str = "environment_contract"


@dataclass(frozen=True)
class ActorStateExposureProjection:
    parent_task: str
    subproblems: tuple[Mapping[str, Any], ...]
    residual_goal_ids: tuple[str, ...]
    active_frontier_ids: tuple[str, ...]
    remaining_budget: int
    schema: str = ACTOR_STATE_EXPOSURE_SCHEMA

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class SubproblemCompiler(Protocol):
    compiler_id: str

    def compile_task(self, task: Any) -> GoalGraph: ...

    def project_state(self, task: Any, environment_state: Any) -> GoalVector: ...

    def frontier(self, graph: GoalGraph, vector: GoalVector) -> tuple[str, ...]: ...

    def label_transition(
        self,
        graph: GoalGraph,
        before: GoalVector,
        action: Any,
        after: GoalVector,
    ) -> TransitionGoalLabel: ...

    def verify_annotation(self, annotation: TransitionGoalLabel) -> VerificationReceipt: ...


def frontier(graph: GoalGraph, vector: GoalVector) -> tuple[str, ...]:
    _require_vector(graph, vector)
    node_map = {node.node_id: node for node in graph.nodes}
    truth = vector.truth_map
    incoming: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    for source, target in graph.edges:
        incoming[target].add(source)

    def node_achieved(node: SubproblemNode) -> bool:
        return all(_goal_satisfied(graph, vector, goal_id) for goal_id in node.effect_goal_ids)

    def node_currently_satisfied(node: SubproblemNode) -> bool:
        return all(truth[goal_id] for goal_id in node.effect_goal_ids)

    root = node_map[graph.root_id]
    if node_currently_satisfied(root):
        return ()

    def regressed_effect_is_needed(node: SubproblemNode) -> bool:
        """Return whether a previously achieved node must be repaired now.

        Achievement latches record that a milestone happened; they are not live
        action preconditions.  A regressed producer is reopened only when its
        currently-false effect is required by the unfinished root or by an
        unfinished consumer.  This avoids both unsafe unlocks and unnecessary
        repair work.
        """

        false_effects = {goal_id for goal_id in node.effect_goal_ids if not truth[goal_id]}
        if false_effects & set(root.effect_goal_ids):
            return True
        return any(
            consumer.node_id != node.node_id
            and not node_currently_satisfied(consumer)
            and bool(false_effects & set(consumer.precondition_goal_ids))
            for consumer in graph.nodes
        )

    ready: list[str] = []
    for node in sorted(graph.nodes, key=lambda item: item.node_id):
        if node.node_id == graph.root_id or node_currently_satisfied(node):
            continue
        if node_achieved(node) and not regressed_effect_is_needed(node):
            continue
        if not all(truth[goal_id] for goal_id in node.precondition_goal_ids):
            continue
        if not all(node_achieved(node_map[source]) for source in incoming[node.node_id]):
            continue
        ready.append(node.node_id)
    return tuple(ready)


def label_transition(
    graph: GoalGraph,
    before: GoalVector,
    action: Any,
    after: GoalVector,
) -> TransitionGoalLabel:
    _require_vector(graph, before)
    _require_vector(graph, after)
    before_truth = before.truth_map
    after_truth = after.truth_map
    newly_satisfied = tuple(
        sorted(
            goal_id for goal_id, value in after_truth.items() if value and not before_truth[goal_id]
        )
    )
    residual = _residual_goal_ids(graph, after)
    active = frontier(graph, after)
    body = {
        "schema": TRANSITION_GOAL_LABEL_SCHEMA,
        "action": str(action),
        "achieved_goal_ids_before": list(before.achievement_latches),
        "achieved_goal_ids_after": list(after.achievement_latches),
        "newly_satisfied_goal_ids": list(newly_satisfied),
        "regressed_goal_ids": list(after.regressed_goal_ids),
        "residual_goal_ids": list(residual),
        "residual_goal_signature": canonical_sha256(list(residual)),
        "active_frontier_ids": list(active),
        "frontier_signature": canonical_sha256(list(active)),
        "graph_sha256": graph.sha256,
    }
    return TransitionGoalLabel(
        action=str(action),
        achieved_goal_ids_before=before.achievement_latches,
        achieved_goal_ids_after=after.achievement_latches,
        newly_satisfied_goal_ids=newly_satisfied,
        regressed_goal_ids=after.regressed_goal_ids,
        residual_goal_ids=residual,
        residual_goal_signature=body["residual_goal_signature"],
        active_frontier_ids=active,
        frontier_signature=body["frontier_signature"],
        graph_sha256=graph.sha256,
        annotation_sha256=canonical_sha256(body),
    )


def project_actor_state_exposure(
    graph: GoalGraph,
    vector: GoalVector,
    *,
    remaining_budget: int,
) -> ActorStateExposureProjection:
    _require_vector(graph, vector)
    if isinstance(remaining_budget, bool) or remaining_budget < 0:
        raise GoalGraphError("remaining budget must be a non-negative integer")
    atom_map = {atom.goal_id: atom for atom in graph.atoms}
    visible_nodes = {
        node.node_id: node
        for node in graph.nodes
        if all(
            atom_map[goal_id].actor_visibility is ActorVisibility.PUBLIC
            for goal_id in (*node.precondition_goal_ids, *node.effect_goal_ids)
        )
    }
    active = set(frontier(graph, vector))
    subproblems: list[Mapping[str, Any]] = []
    for node in sorted(visible_nodes.values(), key=lambda item: item.node_id):
        complete = all(
            vector.truth_map[goal_id]
            if node.node_id == graph.root_id
            else _goal_satisfied(graph, vector, goal_id)
            for goal_id in node.effect_goal_ids
        )
        subproblems.append(
            {
                "id": node.node_id,
                "label": node.label,
                "status": "frontier"
                if node.node_id in active
                else "achieved"
                if complete
                else "blocked",
                "requires": [
                    goal_id
                    for goal_id in node.precondition_goal_ids
                    if atom_map[goal_id].actor_visibility is ActorVisibility.PUBLIC
                ],
            }
        )
    public_residual = tuple(
        goal_id
        for goal_id in _residual_goal_ids(graph, vector)
        if atom_map[goal_id].actor_visibility is ActorVisibility.PUBLIC
    )
    public_frontier = tuple(
        node_id for node_id in frontier(graph, vector) if node_id in visible_nodes
    )
    root = next(node for node in graph.nodes if node.node_id == graph.root_id)
    return ActorStateExposureProjection(
        parent_task=root.label,
        subproblems=tuple(subproblems),
        residual_goal_ids=public_residual,
        active_frontier_ids=public_frontier,
        remaining_budget=remaining_budget,
    )


@dataclass(frozen=True)
class ActorAwarenessProjection(ActorStateExposureProjection):
    """Historical prompt contract retained for artifact compatibility."""

    schema: str = ACTOR_AWARENESS_SCHEMA


def project_actor_awareness(
    graph: GoalGraph,
    vector: GoalVector,
    *,
    remaining_budget: int,
) -> ActorAwarenessProjection:
    projection = project_actor_state_exposure(
        graph,
        vector,
        remaining_budget=remaining_budget,
    )
    return ActorAwarenessProjection(
        parent_task=projection.parent_task,
        subproblems=projection.subproblems,
        residual_goal_ids=projection.residual_goal_ids,
        active_frontier_ids=projection.active_frontier_ids,
        remaining_budget=projection.remaining_budget,
    )


def goal_state_projection(
    graph: GoalGraph,
    vector: GoalVector,
) -> dict[str, Any]:
    """Return the canonical current-state fields used by DPPO grouping."""

    _require_vector(graph, vector)
    residual = _residual_goal_ids(graph, vector)
    active = frontier(graph, vector)
    return {
        "residual_goal_ids": list(residual),
        "residual_goal_signature": canonical_sha256(list(residual)),
        "active_frontier_ids": list(active),
        "frontier_signature": canonical_sha256(list(active)),
    }


def _goal_satisfied(graph: GoalGraph, vector: GoalVector, goal_id: str) -> bool:
    atom = next(atom for atom in graph.atoms if atom.goal_id == goal_id)
    if atom.invariant:
        return vector.truth_map[goal_id]
    return goal_id in vector.achievement_latches


def _residual_goal_ids(graph: GoalGraph, vector: GoalVector) -> tuple[str, ...]:
    root = next(node for node in graph.nodes if node.node_id == graph.root_id)
    truth = vector.truth_map
    if all(truth[goal_id] for goal_id in root.effect_goal_ids):
        return ()
    root_regressions = {goal_id for goal_id in root.effect_goal_ids if not truth[goal_id]}
    return tuple(
        sorted(
            atom.goal_id
            for atom in graph.atoms
            if atom.goal_id in root_regressions or not _goal_satisfied(graph, vector, atom.goal_id)
        )
    )


def _require_vector(graph: GoalGraph, vector: GoalVector) -> None:
    if vector.schema != GOAL_VECTOR_SCHEMA or vector.graph_sha256 != graph.sha256:
        raise GoalGraphError("goal vector does not belong to this graph")


def _assert_acyclic(nodes: set[str], edges: tuple[tuple[str, str], ...]) -> None:
    adjacency = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in set(edges):
        adjacency[source].append(target)
        indegree[target] += 1
    pending = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        for target in adjacency[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                pending.append(target)
    if visited != len(nodes):
        raise GoalGraphError("goal graph contains a cycle")
