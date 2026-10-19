"""AppWorld goal graphs derived from official train/dev evaluator requirements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from lh_platform.training.subproblems.base import (
    ActorVisibility,
    GoalAtom,
    GoalGraph,
    GoalGraphError,
    GoalVector,
    SubproblemNode,
    TransitionGoalLabel,
    VerificationReceipt,
    canonical_sha256,
    frontier,
    label_transition,
)


@dataclass(frozen=True)
class AppWorldTaskContract:
    task_id: str
    split: str
    instruction: str
    evaluator_requirements: tuple[str, ...]
    required_apps: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.instruction.strip():
            raise GoalGraphError("AppWorld task requires stable identity and instruction")
        if self.split not in {"train", "dev"}:
            raise GoalGraphError("AppWorld oracle goal compilation is restricted to train/dev")
        if not self.evaluator_requirements or any(
            not requirement.strip() for requirement in self.evaluator_requirements
        ):
            raise GoalGraphError("AppWorld task requires non-empty evaluator requirements")
        if len(set(self.evaluator_requirements)) != len(self.evaluator_requirements):
            raise GoalGraphError("AppWorld evaluator requirements must be unique")


class AppWorldSubproblemCompiler:
    """Compile a flat verified goal set without reading solution/API-call traces."""

    compiler_id = "appworld_official_evaluator_goal_graph_v1"

    def compile_task(self, task: AppWorldTaskContract) -> GoalGraph:
        # Re-run dataclass validation for callers crossing an untyped adapter boundary.
        if task.split not in {"train", "dev"}:
            raise GoalGraphError("AppWorld oracle goal compilation is restricted to train/dev")
        atoms = tuple(
            GoalAtom(
                goal_id=f"requirement_{index:03d}",
                verifier_id=("appworld_evaluator_requirement:" + canonical_sha256(requirement)),
                invariant=True,
                actor_visibility=ActorVisibility.TRAINING_ONLY,
            )
            for index, requirement in enumerate(task.evaluator_requirements)
        )
        nodes = (
            SubproblemNode(
                node_id="root",
                parent_id=None,
                precondition_goal_ids=(),
                effect_goal_ids=tuple(atom.goal_id for atom in atoms),
                label=task.instruction,
                source="official_evaluator",
            ),
            *(
                SubproblemNode(
                    node_id=f"satisfy_{index:03d}",
                    parent_id="root",
                    precondition_goal_ids=(),
                    effect_goal_ids=(atom.goal_id,),
                    label=f"private evaluator milestone {index + 1}",
                    source="official_evaluator",
                )
                for index, (atom, requirement) in enumerate(
                    zip(atoms, task.evaluator_requirements, strict=True)
                )
            ),
        )
        return GoalGraph(
            task_id=task.task_id,
            root_id="root",
            atoms=atoms,
            nodes=nodes,
            edges=(),
        )

    def project_state(
        self,
        task: AppWorldTaskContract,
        environment_state: Mapping[str, Any],
        *,
        previous: GoalVector | None = None,
    ) -> GoalVector:
        graph = self.compile_task(task)
        passes = environment_state.get("passes")
        failures = environment_state.get("failures")
        if not isinstance(passes, list) or not isinstance(failures, list):
            raise GoalGraphError("AppWorld evaluator state requires passes and failures")
        passed_requirements = _requirements(passes)
        failed_requirements = _requirements(failures)
        if passed_requirements & failed_requirements:
            raise GoalGraphError("AppWorld evaluator marked one requirement pass and fail")
        observed = passed_requirements | failed_requirements
        expected = set(task.evaluator_requirements)
        if observed != expected:
            raise GoalGraphError("AppWorld evaluator state does not cover every requirement")
        truth = {
            f"requirement_{index:03d}": requirement in passed_requirements
            for index, requirement in enumerate(task.evaluator_requirements)
        }
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
            raise GoalGraphError("AppWorld goal annotation hash mismatch")
        return VerificationReceipt(
            verified=True,
            compiler_id=self.compiler_id,
            annotation_sha256=declared,
            verifier_input_sha256=canonical_sha256(body),
        )


def _requirements(rows: list[Any]) -> set[str]:
    requirements: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("requirement"), str):
            raise GoalGraphError("AppWorld evaluator result contains an invalid requirement")
        requirements.append(str(row["requirement"]))
    if len(requirements) != len(set(requirements)):
        raise GoalGraphError("AppWorld evaluator result contains duplicate requirements")
    return set(requirements)
