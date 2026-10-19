"""DecompositionProvider adapters: verifier-backed trajectory annotation.

Satisfies ``agentic_rl_runtime.DecompositionProvider``. Credit / RLMethod
adapters must not live here — editing credit must not open annotate paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace

from lh_platform.alfworld_state import build_verified_subproblem_annotation
from lh_platform.studies.alfworld_subproblems import derive_alfworld_active_subproblem
from lh_platform.training.agentic_rl_components.errors import AgenticRLComponentError
from lh_platform.training.agentic_rl_components.identity import builtin_component
from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.alfworld_semantic_verifier import (
    audit_alfworld_goal_graph_semantics,
    audit_alfworld_snapshot_composition,
)
from lh_platform.training.subproblems.appworld import (
    AppWorldSubproblemCompiler,
    AppWorldTaskContract,
)
from lh_platform.training.subproblems.base import goal_state_projection

ALFWORLD_DECOMPOSITION_RESERVED_FIELDS = frozenset(
    {
        "active_subproblem_id",
        "subproblem_contract",
        "subproblem_contract_sha256",
        "subproblem_contract_verified",
        "subproblem_source",
        "subproblem_verifier_input",
        "goal_graph",
        "goal_graph_sha256",
        "goal_vector_before",
        "goal_vector_after",
        "achieved_goal_ids_before",
        "achieved_goal_ids_after",
        "newly_satisfied_goal_ids",
        "regressed_goal_ids",
        "residual_goal_ids_after",
        "residual_goal_ids",
        "residual_goal_signature",
        "residual_goal_signature_before",
        "residual_goal_signature_after",
        "active_frontier_ids",
        "frontier_signature",
        "active_frontier_ids_after",
        "frontier_signature_after",
        "goal_annotation_sha256",
        "goal_annotation_verified",
        "goal_verifier_input",
        "goal_graph_semantic_verified",
        "goal_graph_semantic_verifier_id",
        "goal_graph_semantic_receipt_sha256",
        "goal_state_semantic_verified",
        "goal_state_semantic_verifier_id",
        "goal_state_before_semantic_receipt_sha256",
        "goal_state_after_semantic_receipt_sha256",
    }
)


class NoDecomposition:
    """Control component: preserve the exact verified rollout unchanged."""

    component = builtin_component(role="decomposition", name="none")

    def annotate(
        self,
        records: tuple[TrainingTrajectoryRecord, ...],
    ) -> tuple[TrainingTrajectoryRecord, ...]:
        return tuple(records)


class AlfWorldFactDecomposition:
    """Derive current child contracts only from task metadata and live facts."""

    component = builtin_component(
        role="decomposition",
        name="alfworld_fact_phases",
        revision="v3",
        config={
            "compiler_id": AlfWorldSubproblemCompiler.compiler_id,
            "task_metadata_field": "alfworld_task_metadata",
            "facts_field": "alfworld_facts",
            "reference_plan_allowed": False,
            "actor_prompt_projection": False,
        },
    )
    _RESERVED = ALFWORLD_DECOMPOSITION_RESERVED_FIELDS

    def annotate(
        self,
        records: tuple[TrainingTrajectoryRecord, ...],
    ) -> tuple[TrainingTrajectoryRecord, ...]:
        annotated: list[TrainingTrajectoryRecord] = []
        compiler = AlfWorldSubproblemCompiler()
        previous_by_trajectory = {}
        ordered = sorted(records, key=lambda row: (row.traj_uid, row.step_idx))
        for record in ordered:
            overlap = self._RESERVED.intersection(record.extra)
            if overlap:
                raise AgenticRLComponentError(
                    "rollout already contains reserved decomposition fields: "
                    + ", ".join(sorted(overlap))
                )
            task_metadata = record.extra.get("alfworld_task_metadata")
            facts = record.extra.get("alfworld_facts")
            next_facts = record.extra.get("alfworld_next_facts")
            if not isinstance(task_metadata, Mapping):
                raise AgenticRLComponentError(
                    "ALFWorld decomposition requires verifier-owned alfworld_task_metadata"
                )
            if not isinstance(facts, (list, tuple)):
                raise AgenticRLComponentError(
                    "ALFWorld decomposition requires verifier-owned alfworld_facts"
                )
            if not isinstance(next_facts, (list, tuple)):
                raise AgenticRLComponentError(
                    "ALFWorld decomposition requires verifier-owned alfworld_next_facts"
                )
            phase = derive_alfworld_active_subproblem(task_metadata, facts)
            receipt = build_verified_subproblem_annotation(**phase.to_builder_args())
            graph = compiler.compile_task(task_metadata)
            before = compiler.project_state(
                task_metadata,
                facts,
                previous=previous_by_trajectory.get(record.traj_uid),
            )
            after = compiler.project_state(
                task_metadata,
                next_facts,
                previous=before,
            )
            label = compiler.label_transition(graph, before, record.action_raw, after)
            verification = compiler.verify_annotation(label)
            next_phase = derive_alfworld_active_subproblem(task_metadata, next_facts)
            before_canonical_facts = phase.verifier_input.get("facts")
            after_canonical_facts = next_phase.verifier_input.get("facts")
            if not isinstance(before_canonical_facts, list) or not isinstance(
                after_canonical_facts, list
            ):
                raise AgenticRLComponentError(
                    "ALFWorld decomposition did not retain canonical verifier facts"
                )
            graph_semantics = audit_alfworld_goal_graph_semantics(task_metadata, graph)
            before_semantics = audit_alfworld_snapshot_composition(
                task_metadata,
                graph,
                before,
                before_canonical_facts,
            )
            after_semantics = audit_alfworld_snapshot_composition(
                task_metadata,
                graph,
                after,
                after_canonical_facts,
            )
            if not (
                graph_semantics.verified and before_semantics.verified and after_semantics.verified
            ):
                violations = sorted(
                    {
                        *graph_semantics.violations,
                        *before_semantics.violations,
                        *after_semantics.violations,
                    }
                )
                raise AgenticRLComponentError(
                    "independent ALFWorld semantic verification failed: " + ", ".join(violations)
                )
            previous_by_trajectory[record.traj_uid] = after
            graph_receipt = _goal_graph_receipt(
                graph=graph,
                before=before,
                after=after,
                label=label,
                verified=verification.verified,
                verifier_input={
                    "schema": "lh_alfworld_goal_state_verifier_input/v3",
                    "task_metadata": dict(task_metadata),
                    "facts": list(facts),
                },
            )
            graph_receipt.update(
                {
                    "goal_graph_semantic_verified": graph_semantics.verified,
                    "goal_graph_semantic_verifier_id": graph_semantics.verifier_id,
                    "goal_graph_semantic_receipt_sha256": canonical_sha256(
                        graph_semantics.to_dict()
                    ),
                    "goal_state_semantic_verified": (
                        before_semantics.verified and after_semantics.verified
                    ),
                    "goal_state_semantic_verifier_id": before_semantics.verifier_id,
                    "goal_state_before_semantic_receipt_sha256": canonical_sha256(
                        before_semantics.to_dict()
                    ),
                    "goal_state_after_semantic_receipt_sha256": canonical_sha256(
                        after_semantics.to_dict()
                    ),
                }
            )
            annotated.append(replace(record, extra={**record.extra, **receipt, **graph_receipt}))
        by_key = {(row.traj_uid, row.step_idx): row for row in annotated}
        return tuple(by_key[(row.traj_uid, row.step_idx)] for row in records)


class AppWorldEvaluatorDecomposition:
    """Derive DPPO state only from AppWorld's official train/dev evaluator."""

    component = builtin_component(
        role="decomposition",
        name="appworld_official_evaluator_goals",
        config={
            "test_splits_allowed": False,
            "solution_code_allowed": False,
            "api_call_trace_allowed": False,
            "actor_prompt_projection": False,
        },
    )

    def annotate(
        self,
        records: tuple[TrainingTrajectoryRecord, ...],
    ) -> tuple[TrainingTrajectoryRecord, ...]:
        compiler = AppWorldSubproblemCompiler()
        annotated: list[TrainingTrajectoryRecord] = []
        previous_by_trajectory = {}
        for record in sorted(records, key=lambda row: (row.traj_uid, row.step_idx)):
            raw_contract = record.extra.get("appworld_task_contract")
            evaluator_before = record.extra.get("appworld_evaluator_before")
            evaluator_after = record.extra.get("appworld_evaluator_after")
            if not isinstance(raw_contract, Mapping):
                raise AgenticRLComponentError(
                    "AppWorld decomposition requires appworld_task_contract"
                )
            if not isinstance(evaluator_before, Mapping) or not isinstance(
                evaluator_after, Mapping
            ):
                raise AgenticRLComponentError(
                    "AppWorld decomposition requires official evaluator before/after states"
                )
            task = AppWorldTaskContract(
                task_id=str(raw_contract.get("task_id", "")),
                split=str(raw_contract.get("split", "")),
                instruction=str(raw_contract.get("instruction", "")),
                evaluator_requirements=tuple(
                    str(value) for value in raw_contract.get("evaluator_requirements", ())
                ),
                required_apps=tuple(str(value) for value in raw_contract.get("required_apps", ())),
            )
            graph = compiler.compile_task(task)
            before = compiler.project_state(
                task,
                evaluator_before,
                previous=previous_by_trajectory.get(record.traj_uid),
            )
            after = compiler.project_state(
                task,
                evaluator_after,
                previous=before,
            )
            label = compiler.label_transition(graph, before, record.action_raw, after)
            verification = compiler.verify_annotation(label)
            previous_by_trajectory[record.traj_uid] = after
            receipt = _goal_graph_receipt(
                graph=graph,
                before=before,
                after=after,
                label=label,
                verified=verification.verified,
                verifier_input={
                    "schema": "lh_appworld_goal_state_verifier_input/v1",
                    "task_contract": asdict(task),
                    "evaluator_state": dict(evaluator_before),
                },
            )
            overlap = set(receipt).intersection(record.extra)
            if overlap:
                raise AgenticRLComponentError(
                    "rollout already contains reserved AppWorld decomposition fields: "
                    + ", ".join(sorted(overlap))
                )
            annotated.append(replace(record, extra={**record.extra, **receipt}))
        by_key = {(row.traj_uid, row.step_idx): row for row in annotated}
        return tuple(by_key[(row.traj_uid, row.step_idx)] for row in records)


def _goal_graph_receipt(
    *,
    graph,
    before,
    after,
    label,
    verified: bool,
    verifier_input: Mapping[str, object],
) -> dict[str, object]:
    before_projection = goal_state_projection(graph, before)
    return {
        "goal_graph": graph.to_dict(),
        "goal_graph_sha256": graph.sha256,
        "goal_verifier_input": dict(verifier_input),
        "goal_vector_before": asdict(before),
        "goal_vector_after": asdict(after),
        "achieved_goal_ids_before": list(label.achieved_goal_ids_before),
        "achieved_goal_ids_after": list(label.achieved_goal_ids_after),
        "newly_satisfied_goal_ids": list(label.newly_satisfied_goal_ids),
        "regressed_goal_ids": list(label.regressed_goal_ids),
        **before_projection,
        "residual_goal_ids_after": list(label.residual_goal_ids),
        "residual_goal_signature_before": before_projection["residual_goal_signature"],
        "residual_goal_signature_after": label.residual_goal_signature,
        "active_frontier_ids_after": list(label.active_frontier_ids),
        "frontier_signature_after": label.frontier_signature,
        "goal_annotation_sha256": label.annotation_sha256,
        "goal_annotation_verified": verified,
        "subproblem_source": "environment_contract",
    }
