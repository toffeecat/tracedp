"""ALFWorld goal graphs compiled from task metadata and live TextWorld facts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from lh_platform.studies.alfworld_subproblems import (
    _held_targets,
    _instances_of_type,
    _normalize_facts,
    _parse_task_metadata,
    _placed_targets,
    _placed_targets_by_receptacle,
    _unary_targets,
)
from lh_platform.training.subproblems.base import (
    ActorVisibility,
    GoalAtom,
    GoalGraph,
    GoalVector,
    SubproblemNode,
    TransitionGoalLabel,
    VerificationReceipt,
    canonical_sha256,
    frontier,
    label_transition,
)


class AlfWorldSubproblemCompiler:
    compiler_id = "alfworld_fact_goal_graph_v3"

    def compile_task(self, task: Mapping[str, Any]) -> GoalGraph:
        task_type, targets = _parse_task_metadata(task)
        task_id = str(task.get("task_id") or canonical_sha256({"task_type": task_type, **targets}))
        object_name = targets["object"]
        receptacle = targets.get("receptacle", "")
        common_atoms = [
            GoalAtom(
                "acquired",
                f"holds({object_name})",
                invariant=task_type == "look_at_obj_in_light",
                actor_visibility=ActorVisibility.PUBLIC,
            )
        ]
        nodes = [
            SubproblemNode(
                "acquire_target",
                "root",
                (),
                ("acquired",),
                f"acquire the target {object_name}",
            )
        ]
        edges: list[tuple[str, str]] = []

        if task_type == "pick_two_obj_and_place":
            atoms = (
                *common_atoms,
                GoalAtom(
                    "placed_first",
                    f"same_receptacle_count_in({object_name},{receptacle})>=1",
                ),
                GoalAtom(
                    "placed_second",
                    f"same_receptacle_count_in({object_name},{receptacle})>=2",
                    invariant=True,
                ),
            )
            nodes = [
                SubproblemNode("root", None, (), ("placed_second",), "place two target objects"),
                SubproblemNode(
                    "acquire_first", "root", (), ("acquired",), "acquire the first target"
                ),
                SubproblemNode(
                    "place_first",
                    "root",
                    ("acquired",),
                    ("placed_first",),
                    f"place the first target in {receptacle}",
                ),
                SubproblemNode(
                    "acquire_and_place_second",
                    "root",
                    ("placed_first",),
                    ("placed_second",),
                    f"acquire and place the second target in {receptacle}",
                ),
            ]
            edges = [
                ("acquire_first", "place_first"),
                ("place_first", "acquire_and_place_second"),
            ]
        elif task_type == "look_at_obj_in_light":
            toggle = targets["toggle"]
            atoms = (
                *common_atoms,
                GoalAtom("light_on", f"istoggled({toggle})", invariant=True),
            )
            nodes = [
                SubproblemNode(
                    "root", None, (), ("acquired", "light_on"), "inspect the target under light"
                ),
                *nodes,
                SubproblemNode(
                    "toggle_light",
                    "root",
                    ("acquired",),
                    ("light_on",),
                    f"turn on {toggle}",
                ),
            ]
            edges = [("acquire_target", "toggle_light")]
        else:
            transform = {
                "pick_clean_then_place_in_recep": ("clean", "transform_clean", "clean"),
                "pick_heat_then_place_in_recep": ("hot", "transform_heat", "heat"),
                "pick_cool_then_place_in_recep": ("cool", "transform_cool", "cool"),
            }.get(task_type)
            atoms_list = list(common_atoms)
            if transform is not None:
                fact_name, node_id, verb = transform
                atoms_list.append(GoalAtom("transformed", f"is{fact_name}({object_name})"))
                nodes.append(
                    SubproblemNode(
                        node_id,
                        "root",
                        ("acquired",),
                        ("transformed",),
                        f"{verb} the target {object_name}",
                    )
                )
                place_preconditions = ("acquired", "transformed")
                edges.extend((("acquire_target", node_id), (node_id, "place_target")))
            else:
                place_preconditions = ("acquired",)
                edges.append(("acquire_target", "place_target"))
            placed_verifier_id = f"inreceptacle({object_name},{receptacle})"
            if transform is not None:
                placed_verifier_id = (
                    f"and(is{fact_name}({object_name}),inreceptacle({object_name},{receptacle}))"
                )
            atoms_list.append(
                GoalAtom(
                    "placed",
                    placed_verifier_id,
                    invariant=True,
                )
            )
            nodes = [
                SubproblemNode("root", None, (), ("placed",), f"place the target in {receptacle}"),
                *nodes,
                SubproblemNode(
                    "place_target",
                    "root",
                    place_preconditions,
                    ("placed",),
                    f"place the target {object_name} in {receptacle}",
                ),
            ]
            atoms = tuple(atoms_list)

        return GoalGraph(
            task_id=task_id,
            root_id="root",
            atoms=tuple(atoms),
            nodes=tuple(nodes),
            edges=tuple(edges),
        )

    def project_state(
        self,
        task: Mapping[str, Any],
        environment_state: Iterable[Any],
        *,
        previous: GoalVector | None = None,
    ) -> GoalVector:
        graph = self.compile_task(task)
        task_type, targets = _parse_task_metadata(task)
        facts = _normalize_facts(environment_state)
        objects = _instances_of_type(facts, "objecttype", targets["object"])
        receptacles = _instances_of_type(facts, "receptacletype", targets.get("receptacle", ""))
        toggles = _instances_of_type(facts, "objecttype", targets.get("toggle", ""))
        held = _held_targets(facts, objects)
        placed = _placed_targets(facts, objects, receptacles)
        placed_by_receptacle = _placed_targets_by_receptacle(
            facts,
            objects,
            receptacles,
        )
        truth: dict[str, bool]
        if task_type == "pick_two_obj_and_place":
            truth = {
                "acquired": bool(held),
                "placed_first": len(placed) >= 1,
                "placed_second": max(
                    (len(members) for members in placed_by_receptacle.values()),
                    default=0,
                )
                >= 2,
            }
        elif task_type == "look_at_obj_in_light":
            truth = {
                "acquired": bool(held),
                "light_on": bool(_unary_targets(facts, "istoggled", toggles)),
            }
        else:
            transform_fact = {
                "pick_clean_then_place_in_recep": "isclean",
                "pick_heat_then_place_in_recep": "ishot",
                "pick_cool_then_place_in_recep": "iscool",
            }.get(task_type)
            transformed = (
                _unary_targets(facts, transform_fact, objects)
                if transform_fact is not None
                else frozenset()
            )
            placed_complete = placed if transform_fact is None else placed & transformed
            truth = {
                # Acquisition is an episode event: it becomes latched only after
                # the agent actually holds a target.  A target that happens to
                # start in the destination (or already transformed) is not
                # evidence that this rollout acquired it.
                "acquired": bool(held),
                "placed": bool(placed_complete),
            }
            if transform_fact is not None:
                truth["transformed"] = bool(transformed)
        return GoalVector.from_truth(graph, truth, previous=previous)

    def frontier(self, graph: GoalGraph, vector: GoalVector) -> tuple[str, ...]:
        return frontier(graph, vector)

    def label_transition(
        self,
        graph: GoalGraph,
        before: GoalVector,
        action: Any,
        after: GoalVector,
    ) -> TransitionGoalLabel:
        return label_transition(graph, before, action, after)

    def verify_annotation(self, annotation: TransitionGoalLabel) -> VerificationReceipt:
        body = asdict(annotation)
        declared = body.pop("annotation_sha256")
        body.pop("schema")
        expected = canonical_sha256({"schema": annotation.schema, **body})
        if expected != declared:
            raise ValueError("ALFWorld goal annotation hash mismatch")
        return VerificationReceipt(
            verified=True,
            compiler_id=self.compiler_id,
            annotation_sha256=declared,
            verifier_input_sha256=canonical_sha256(body),
        )
