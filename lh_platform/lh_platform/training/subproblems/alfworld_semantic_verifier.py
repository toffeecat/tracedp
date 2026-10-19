"""Independent semantic audits for ALFWorld parent/subproblem graphs.

The compiler in :mod:`alfworld` constructs a graph.  This module deliberately
does not call that compiler, its frontier implementation, or its phase selector.
It uses a second, declarative task specification to catch systematic graph
mutations and independently evaluates canonical environment facts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from lh_platform.training.subproblems.base import (
    ActorVisibility,
    GoalGraph,
    GoalVector,
    canonical_sha256,
)

ALFWORLD_GRAPH_SEMANTIC_AUDIT_SCHEMA = "lh_alfworld_goal_graph_semantic_audit/v3"
ALFWORLD_SNAPSHOT_COMPOSITION_AUDIT_SCHEMA = "lh_alfworld_snapshot_composition_audit/v3"
ALFWORLD_SEMANTIC_VERIFIER_ID = "alfworld_independent_declarative_task_spec_v3"


@dataclass(frozen=True)
class AlfWorldGoalGraphSemanticReceipt:
    verified: bool
    task_type: str
    graph_sha256: str
    task_contract_sha256: str
    passed_checks: tuple[str, ...]
    violations: tuple[str, ...]
    verifier_id: str = ALFWORLD_SEMANTIC_VERIFIER_ID
    schema: str = ALFWORLD_GRAPH_SEMANTIC_AUDIT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verifier_id": self.verifier_id,
            "verified": self.verified,
            "task_type": self.task_type,
            "graph_sha256": self.graph_sha256,
            "task_contract_sha256": self.task_contract_sha256,
            "passed_checks": list(self.passed_checks),
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class AlfWorldSnapshotCompositionReceipt:
    verified: bool
    task_type: str
    graph_sha256: str
    goal_vector_sha256: str
    facts_sha256: str
    independent_goal_truth: tuple[tuple[str, bool], ...]
    parent_contract_passed: bool
    graph_root_contract_passed: bool
    all_required_children_passed: bool
    child_to_parent_implication_passed: bool
    violations: tuple[str, ...]
    verifier_id: str = ALFWORLD_SEMANTIC_VERIFIER_ID
    schema: str = ALFWORLD_SNAPSHOT_COMPOSITION_AUDIT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verifier_id": self.verifier_id,
            "verified": self.verified,
            "task_type": self.task_type,
            "graph_sha256": self.graph_sha256,
            "goal_vector_sha256": self.goal_vector_sha256,
            "facts_sha256": self.facts_sha256,
            "independent_goal_truth": {
                goal_id: value for goal_id, value in self.independent_goal_truth
            },
            "parent_contract_passed": self.parent_contract_passed,
            "graph_root_contract_passed": self.graph_root_contract_passed,
            "all_required_children_passed": self.all_required_children_passed,
            "child_to_parent_implication_passed": (self.child_to_parent_implication_passed),
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class _TaskContract:
    task_type: str
    object_type: str
    receptacle_type: str
    toggle_type: str

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "task_type": self.task_type,
                "object_type": self.object_type,
                "receptacle_type": self.receptacle_type,
                "toggle_type": self.toggle_type,
            }
        )


@dataclass(frozen=True)
class _ExpectedAtom:
    verifier_id: str
    invariant: bool
    actor_visibility: ActorVisibility = ActorVisibility.PUBLIC


@dataclass(frozen=True)
class _ExpectedNode:
    parent_id: str | None
    preconditions: tuple[str, ...]
    effects: tuple[str, ...]


@dataclass(frozen=True)
class _ExpectedGraph:
    atoms: Mapping[str, _ExpectedAtom]
    nodes: Mapping[str, _ExpectedNode]
    edges: frozenset[tuple[str, str]]


def audit_alfworld_goal_graph_semantics(
    task: Mapping[str, Any],
    graph: GoalGraph,
) -> AlfWorldGoalGraphSemanticReceipt:
    """Compare a compiled graph with an independent declarative task spec."""

    contract = _parse_task_contract(task)
    expected = _expected_graph(contract)
    violations: list[str] = []
    passed: list[str] = []

    if graph.root_id != "root":
        violations.append("root_id_mismatch")
    else:
        passed.append("root_identity")

    actual_atoms = {atom.goal_id: atom for atom in graph.atoms}
    if set(actual_atoms) != set(expected.atoms):
        violations.append("atom_set_mismatch")
    else:
        passed.append("atom_set")
    for goal_id in sorted(set(actual_atoms) & set(expected.atoms)):
        actual = actual_atoms[goal_id]
        spec = expected.atoms[goal_id]
        if (
            actual.verifier_id != spec.verifier_id
            or actual.invariant is not spec.invariant
            or actual.actor_visibility is not spec.actor_visibility
        ):
            violations.append(f"atom_contract_mismatch:{goal_id}")
        else:
            passed.append(f"atom_contract:{goal_id}")

    actual_nodes = {node.node_id: node for node in graph.nodes}
    if set(actual_nodes) != set(expected.nodes):
        violations.append("node_set_mismatch")
    else:
        passed.append("node_set")
    for node_id in sorted(set(actual_nodes) & set(expected.nodes)):
        actual = actual_nodes[node_id]
        spec = expected.nodes[node_id]
        if (
            actual.parent_id != spec.parent_id
            or tuple(actual.precondition_goal_ids) != spec.preconditions
            or tuple(actual.effect_goal_ids) != spec.effects
            or actual.source != "verifier_dependency"
        ):
            violations.append(f"node_contract_mismatch:{node_id}")
        else:
            passed.append(f"node_contract:{node_id}")

    if frozenset(graph.edges) != expected.edges:
        violations.append("dependency_edge_mismatch")
    else:
        passed.append("dependency_edges")

    root = actual_nodes.get(graph.root_id)
    child_effects = {
        goal_id
        for node_id, node in actual_nodes.items()
        if node_id != graph.root_id
        for goal_id in node.effect_goal_ids
    }
    if root is None or not set(root.effect_goal_ids) <= child_effects:
        violations.append("root_effect_not_covered_by_children")
    else:
        passed.append("root_effect_coverage")

    referenced_atoms = {
        goal_id
        for node in actual_nodes.values()
        for goal_id in (*node.precondition_goal_ids, *node.effect_goal_ids)
    }
    if set(actual_atoms) != referenced_atoms:
        violations.append("orphan_or_unknown_goal_atom")
    else:
        passed.append("goal_atom_coverage")

    ordered_violations = tuple(sorted(set(violations)))
    return AlfWorldGoalGraphSemanticReceipt(
        verified=not ordered_violations,
        task_type=contract.task_type,
        graph_sha256=graph.sha256,
        task_contract_sha256=contract.sha256,
        passed_checks=tuple(sorted(set(passed))),
        violations=ordered_violations,
    )


def audit_alfworld_snapshot_composition(
    task: Mapping[str, Any],
    graph: GoalGraph,
    vector: GoalVector,
    canonical_facts: Iterable[Mapping[str, Any]],
) -> AlfWorldSnapshotCompositionReceipt:
    """Audit fact truth and child-to-parent composition without compiler calls.

    ``canonical_facts`` is the retained ``[{name, args}, ...]`` verifier input.
    Parsing raw TextWorld strings is intentionally outside this second semantic
    path, so formatting normalization and task semantics remain separate risks.
    """

    contract = _parse_task_contract(task)
    graph_receipt = audit_alfworld_goal_graph_semantics(task, graph)
    facts = _canonical_fact_tuples(canonical_facts)
    independent_truth, parent_passed = _evaluate_contract(contract, facts)
    independent_truth_map = dict(independent_truth)
    violations = [f"graph_semantics:{violation}" for violation in graph_receipt.violations]

    if vector.graph_sha256 != graph.sha256:
        violations.append("goal_vector_graph_mismatch")
    for goal_id in sorted(set(vector.truth_map) | set(independent_truth_map)):
        if vector.truth_map.get(goal_id) != independent_truth_map.get(goal_id):
            violations.append(f"goal_vector_truth_mismatch:{goal_id}")

    root = next((node for node in graph.nodes if node.node_id == graph.root_id), None)
    root_passed = bool(root) and all(
        independent_truth_map.get(goal_id, False) for goal_id in root.effect_goal_ids
    )
    if root_passed != parent_passed:
        violations.append("graph_root_parent_contract_disagreement")

    atom_by_id = {atom.goal_id: atom for atom in graph.atoms}

    def child_effect_passed(goal_id: str) -> bool:
        atom = atom_by_id.get(goal_id)
        if atom is None:
            return False
        if atom.invariant:
            return independent_truth_map.get(goal_id, False)
        return goal_id in vector.achievement_latches

    children = [node for node in graph.nodes if node.node_id != graph.root_id]
    all_children_passed = bool(children) and all(
        all(child_effect_passed(goal_id) for goal_id in node.effect_goal_ids) for node in children
    )
    implication_passed = (not all_children_passed) or parent_passed
    if not implication_passed:
        violations.append("all_required_children_do_not_imply_parent")

    ordered_violations = tuple(sorted(set(violations)))
    return AlfWorldSnapshotCompositionReceipt(
        verified=not ordered_violations,
        task_type=contract.task_type,
        graph_sha256=graph.sha256,
        goal_vector_sha256=vector.sha256,
        facts_sha256=canonical_sha256([{"name": name, "args": list(args)} for name, args in facts]),
        independent_goal_truth=independent_truth,
        parent_contract_passed=parent_passed,
        graph_root_contract_passed=root_passed,
        all_required_children_passed=all_children_passed,
        child_to_parent_implication_passed=implication_passed,
        violations=ordered_violations,
    )


def _parse_task_contract(task: Mapping[str, Any]) -> _TaskContract:
    if not isinstance(task, Mapping):
        raise ValueError("ALFWorld semantic task must be a mapping")
    task_type = str(task.get("task_type", "")).strip()
    supported = {
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
    }
    if task_type not in supported:
        raise ValueError(f"unsupported ALFWorld semantic task: {task_type!r}")
    params = task.get("pddl_params")
    if not isinstance(params, Mapping):
        raise ValueError("ALFWorld semantic task lacks pddl_params")
    object_type = str(params.get("object_target") or "").strip().lower()
    receptacle_type = str(params.get("parent_target") or "").strip().lower()
    toggle_type = str(params.get("toggle_target") or "").strip().lower()
    if not object_type:
        raise ValueError("ALFWorld semantic task lacks object target")
    if task_type == "look_at_obj_in_light":
        if not toggle_type:
            raise ValueError("ALFWorld semantic task lacks toggle target")
    elif not receptacle_type:
        raise ValueError("ALFWorld semantic task lacks receptacle target")
    return _TaskContract(
        task_type=task_type,
        object_type=object_type,
        receptacle_type=receptacle_type,
        toggle_type=toggle_type,
    )


def _expected_graph(contract: _TaskContract) -> _ExpectedGraph:
    acquired = _ExpectedAtom(
        f"holds({contract.object_type})",
        invariant=contract.task_type == "look_at_obj_in_light",
    )
    if contract.task_type == "pick_two_obj_and_place":
        return _ExpectedGraph(
            atoms={
                "acquired": acquired,
                "placed_first": _ExpectedAtom(
                    "same_receptacle_count_in("
                    f"{contract.object_type},{contract.receptacle_type})>=1",
                    invariant=False,
                ),
                "placed_second": _ExpectedAtom(
                    "same_receptacle_count_in("
                    f"{contract.object_type},{contract.receptacle_type})>=2",
                    invariant=True,
                ),
            },
            nodes={
                "root": _ExpectedNode(None, (), ("placed_second",)),
                "acquire_first": _ExpectedNode("root", (), ("acquired",)),
                "place_first": _ExpectedNode("root", ("acquired",), ("placed_first",)),
                "acquire_and_place_second": _ExpectedNode(
                    "root", ("placed_first",), ("placed_second",)
                ),
            },
            edges=frozenset(
                {
                    ("acquire_first", "place_first"),
                    ("place_first", "acquire_and_place_second"),
                }
            ),
        )
    if contract.task_type == "look_at_obj_in_light":
        return _ExpectedGraph(
            atoms={
                "acquired": acquired,
                "light_on": _ExpectedAtom(f"istoggled({contract.toggle_type})", invariant=True),
            },
            nodes={
                "root": _ExpectedNode(None, (), ("acquired", "light_on")),
                "acquire_target": _ExpectedNode("root", (), ("acquired",)),
                "toggle_light": _ExpectedNode("root", ("acquired",), ("light_on",)),
            },
            edges=frozenset({("acquire_target", "toggle_light")}),
        )

    transform = {
        "pick_clean_then_place_in_recep": ("isclean", "transform_clean"),
        "pick_heat_then_place_in_recep": ("ishot", "transform_heat"),
        "pick_cool_then_place_in_recep": ("iscool", "transform_cool"),
    }.get(contract.task_type)
    atoms: dict[str, _ExpectedAtom] = {"acquired": acquired}
    nodes: dict[str, _ExpectedNode] = {
        "root": _ExpectedNode(None, (), ("placed",)),
        "acquire_target": _ExpectedNode("root", (), ("acquired",)),
    }
    if transform is None:
        atoms["placed"] = _ExpectedAtom(
            f"inreceptacle({contract.object_type},{contract.receptacle_type})",
            invariant=True,
        )
        nodes["place_target"] = _ExpectedNode("root", ("acquired",), ("placed",))
        edges = frozenset({("acquire_target", "place_target")})
    else:
        predicate, transform_node = transform
        atoms["transformed"] = _ExpectedAtom(
            f"{predicate}({contract.object_type})", invariant=False
        )
        atoms["placed"] = _ExpectedAtom(
            (
                f"and({predicate}({contract.object_type}),"
                f"inreceptacle({contract.object_type},{contract.receptacle_type}))"
            ),
            invariant=True,
        )
        nodes[transform_node] = _ExpectedNode("root", ("acquired",), ("transformed",))
        nodes["place_target"] = _ExpectedNode("root", ("acquired", "transformed"), ("placed",))
        edges = frozenset(
            {
                ("acquire_target", transform_node),
                (transform_node, "place_target"),
            }
        )
    return _ExpectedGraph(atoms=atoms, nodes=nodes, edges=edges)


def _canonical_fact_tuples(
    canonical_facts: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rows: list[tuple[str, tuple[str, ...]]] = []
    for fact in canonical_facts:
        if not isinstance(fact, Mapping):
            raise ValueError("semantic audit requires canonical fact mappings")
        name = fact.get("name")
        args = fact.get("args")
        if not isinstance(name, str) or not isinstance(args, (list, tuple)):
            raise ValueError("semantic audit fact requires name and args")
        rows.append(
            (
                name.strip().lower(),
                tuple(str(argument).strip().lower() for argument in args),
            )
        )
    return tuple(sorted(rows))


def _evaluate_contract(
    contract: _TaskContract,
    facts: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[tuple[str, bool], ...], bool]:
    object_instances = _typed_instances(facts, "objecttype", f"{contract.object_type}type")
    receptacle_instances = _typed_instances(
        facts, "receptacletype", f"{contract.receptacle_type}type"
    )
    toggle_instances = _typed_instances(facts, "objecttype", f"{contract.toggle_type}type")
    held = {
        args[-1]
        for name, args in facts
        if name == "holds" and args and args[-1] in object_instances
    }
    placements = {
        (args[0], args[1])
        for name, args in facts
        if name == "inreceptacle"
        and len(args) >= 2
        and args[0] in object_instances
        and args[1] in receptacle_instances
    }
    placed = {object_id for object_id, _receptacle_id in placements}

    if contract.task_type == "pick_two_obj_and_place":
        placed_by_receptacle: dict[str, set[str]] = {}
        for object_id, receptacle_id in placements:
            placed_by_receptacle.setdefault(receptacle_id, set()).add(object_id)
        truth = {
            "acquired": bool(held),
            "placed_first": len(placed) >= 1,
            "placed_second": max(
                (len(members) for members in placed_by_receptacle.values()),
                default=0,
            )
            >= 2,
        }
        parent = truth["placed_second"]
    elif contract.task_type == "look_at_obj_in_light":
        light_on = any(
            name == "istoggled" and args and args[0] in toggle_instances for name, args in facts
        )
        truth = {"acquired": bool(held), "light_on": light_on}
        parent = truth["acquired"] and truth["light_on"]
    else:
        predicate = {
            "pick_clean_then_place_in_recep": "isclean",
            "pick_heat_then_place_in_recep": "ishot",
            "pick_cool_then_place_in_recep": "iscool",
        }.get(contract.task_type)
        transformed = (
            {
                args[0]
                for name, args in facts
                if name == predicate and args and args[0] in object_instances
            }
            if predicate is not None
            else set()
        )
        placed_complete = placed if predicate is None else placed & transformed
        truth = {"acquired": bool(held), "placed": bool(placed_complete)}
        if predicate is not None:
            truth["transformed"] = bool(transformed)
        parent = bool(placed_complete)
    return tuple(sorted(truth.items())), bool(parent)


def _typed_instances(
    facts: tuple[tuple[str, tuple[str, ...]], ...],
    predicate: str,
    expected_type: str,
) -> set[str]:
    if not expected_type or expected_type == "type":
        return set()
    return {
        args[0]
        for name, args in facts
        if name == predicate and len(args) >= 2 and args[-1] == expected_type
    }
