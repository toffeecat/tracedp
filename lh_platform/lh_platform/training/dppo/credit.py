"""Verifier-Refined G2PO credit identity over shared G2PO-TN statistics.

``DPPO*`` symbols are retained as compatibility imports for historical run
artifacts.  This module does not implement child-boundary credit or dynamic
programming and must not publish the future DPPO method identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any

from lh_platform.studies.alfworld_subproblems import (
    AlfWorldSubproblemError,
    verify_alfworld_subproblem_annotation,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.credit_estimators.g2po_bridge import (
    assign_verifier_refined_g2po_credit,
)
from lh_platform.training.g2po.credit import (
    G2POCreditAssignment,
    G2POCreditConfig,
    G2POCreditMode,
    G2POTransition,
    validate_g2po_transitions,
)
from lh_platform.training.g2po.credit import (
    transitions_from_v2_records as g2po_transitions_from_v2_records,
)
from lh_platform.training.state_identity import VerifiedStateIdentity, verified_state_identity
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.alfworld_semantic_verifier import (
    audit_alfworld_goal_graph_semantics,
    audit_alfworld_snapshot_composition,
)
from lh_platform.training.subproblems.appworld import (
    AppWorldSubproblemCompiler,
    AppWorldTaskContract,
)
from lh_platform.training.subproblems.base import (
    GoalVector,
    canonical_sha256,
    goal_state_projection,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DPPOCreditMode(StrEnum):
    """Legacy enum name for verifier-refined G2PO implementation modes."""

    VERIFIED_STATE = "verified_state_dp"
    ORACLE_PHASE_REFINED_TN = "oracle_phase_refined_g2po_tn"


@dataclass(frozen=True)
class DPPOTransition(G2POTransition):
    """G2PO transition plus verifier-owned residual-state evidence.

    The class name is an artifact-compatibility boundary, not a method claim.
    """

    state_identity: VerifiedStateIdentity | None = None
    active_subproblem_id: str | None = None
    subproblem_contract: Mapping[str, Any] | None = None
    subproblem_contract_sha256: str | None = None
    subproblem_contract_verified: bool = False
    subproblem_source: str | None = None
    subproblem_verifier_input: Mapping[str, Any] | None = None
    goal_graph: Mapping[str, Any] | None = None
    goal_graph_sha256: str | None = None
    residual_goal_ids: tuple[str, ...] | None = None
    residual_goal_signature: str | None = None
    active_frontier_ids: tuple[str, ...] | None = None
    frontier_signature: str | None = None
    goal_verifier_input: Mapping[str, Any] | None = None
    goal_graph_semantic_verified: bool = False
    goal_graph_semantic_verifier_id: str | None = None
    goal_graph_semantic_receipt_sha256: str | None = None
    goal_state_semantic_verified: bool = False
    goal_state_semantic_verifier_id: str | None = None
    goal_state_before_semantic_receipt_sha256: str | None = None
    goal_state_after_semantic_receipt_sha256: str | None = None


def transitions_from_v2_records(
    records: Iterable[TrainingTrajectoryRecord],
    *,
    mode: DPPOCreditMode | str,
) -> tuple[DPPOTransition, ...]:
    """Project active records into verifier-refined G2PO transitions."""

    selected_mode = DPPOCreditMode(mode)
    source_records = tuple(records)
    base_rows = g2po_transitions_from_v2_records(
        source_records,
        mode=G2POCreditMode.TRAJECTORY_NORMALIZED,
    )
    rows: list[DPPOTransition] = []
    records_by_key = {
        (row.group_uid, row.traj_uid, row.step_idx): row for row in source_records if row.active
    }
    for base in base_rows:
        record = records_by_key[(base.group_uid, base.trajectory_id, base.step_index)]
        rows.append(
            DPPOTransition(
                **{field.name: getattr(base, field.name) for field in fields(G2POTransition)},
                state_identity=(
                    verified_state_identity(record)
                    if selected_mode is DPPOCreditMode.VERIFIED_STATE
                    else None
                ),
                active_subproblem_id=_optional_string(record.extra, "active_subproblem_id"),
                subproblem_contract=_optional_mapping(record.extra, "subproblem_contract"),
                subproblem_contract_sha256=_optional_string(
                    record.extra, "subproblem_contract_sha256"
                ),
                subproblem_contract_verified=bool(
                    record.extra.get("subproblem_contract_verified", False)
                ),
                subproblem_source=_optional_string(record.extra, "subproblem_source"),
                subproblem_verifier_input=_optional_mapping(
                    record.extra, "subproblem_verifier_input"
                ),
                goal_graph=_optional_mapping(record.extra, "goal_graph"),
                goal_graph_sha256=_optional_string(record.extra, "goal_graph_sha256"),
                residual_goal_ids=_optional_string_tuple(record.extra, "residual_goal_ids"),
                residual_goal_signature=_optional_string(record.extra, "residual_goal_signature"),
                active_frontier_ids=_optional_string_tuple(record.extra, "active_frontier_ids"),
                frontier_signature=_optional_string(record.extra, "frontier_signature"),
                goal_verifier_input=_optional_mapping(record.extra, "goal_verifier_input"),
                goal_graph_semantic_verified=bool(
                    record.extra.get("goal_graph_semantic_verified", False)
                ),
                goal_graph_semantic_verifier_id=_optional_string(
                    record.extra, "goal_graph_semantic_verifier_id"
                ),
                goal_graph_semantic_receipt_sha256=_optional_string(
                    record.extra, "goal_graph_semantic_receipt_sha256"
                ),
                goal_state_semantic_verified=bool(
                    record.extra.get("goal_state_semantic_verified", False)
                ),
                goal_state_semantic_verifier_id=_optional_string(
                    record.extra, "goal_state_semantic_verifier_id"
                ),
                goal_state_before_semantic_receipt_sha256=_optional_string(
                    record.extra, "goal_state_before_semantic_receipt_sha256"
                ),
                goal_state_after_semantic_receipt_sha256=_optional_string(
                    record.extra, "goal_state_after_semantic_receipt_sha256"
                ),
            )
        )
    return validate_dppo_transitions(rows, mode=selected_mode)


def validate_dppo_transitions(
    transitions: Iterable[DPPOTransition],
    *,
    mode: DPPOCreditMode | str,
) -> tuple[DPPOTransition, ...]:
    """Fail closed unless the refined state identity is independently verified."""

    selected_mode = DPPOCreditMode(mode)
    rows = validate_g2po_transitions(
        transitions,
        mode=G2POCreditMode.TRAJECTORY_NORMALIZED,
    )
    if selected_mode is DPPOCreditMode.VERIFIED_STATE:
        for row in rows:
            if row.state_identity is None:
                raise ValueError("verified_state_dp fails closed without a verified state identity")
        return rows

    by_trajectory: dict[tuple[str, str], list[DPPOTransition]] = {}
    for row in rows:
        by_trajectory.setdefault((row.group_uid, row.trajectory_id), []).append(row)
    for trajectory in by_trajectory.values():
        previous: GoalVector | None = None
        for row in sorted(trajectory, key=lambda item: item.step_index):
            verifier_schema, previous = _verify_goal_state_projection(row, previous=previous)
            if verifier_schema in {
                "lh_alfworld_goal_state_verifier_input/v1",
                "lh_alfworld_goal_state_verifier_input/v2",
                "lh_alfworld_goal_state_verifier_input/v3",
            }:
                _verify_legacy_alfworld_phase_receipt(row)
    return rows


def assign_dppo_credit(
    transitions: Iterable[DPPOTransition],
    *,
    mode: DPPOCreditMode | str,
    config: G2POCreditConfig | None = None,
) -> tuple[G2POCreditAssignment, ...]:
    """Validate refined identity, then route through ``g2po_bridge``.

    Credit assignment locality is ``credit_estimators.g2po_bridge`` (online
    Verifier-Refined → G2PO-TN).  This module only owns fail-closed validation
    of verifier / goal-graph evidence before that seam.
    """

    selected_mode = DPPOCreditMode(mode)
    rows = validate_dppo_transitions(transitions, mode=selected_mode)
    return assign_verifier_refined_g2po_credit(
        rows,
        mode=selected_mode.value,
        config=config,
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_legacy_alfworld_phase_receipt(row: DPPOTransition) -> None:
    if not row.active_subproblem_id or not row.active_subproblem_id.strip():
        raise ValueError("oracle subproblem credit requires active_subproblem_id")
    if not _SHA256.fullmatch(row.subproblem_contract_sha256 or ""):
        raise ValueError("oracle subproblem credit requires subproblem_contract_sha256")
    if not row.subproblem_contract_verified:
        raise ValueError("oracle subproblem credit requires an environment-verified contract")
    if row.subproblem_source != "environment_contract":
        raise ValueError("oracle subproblem credit only accepts environment_contract annotations")
    if not isinstance(row.subproblem_contract, Mapping):
        raise ValueError("oracle subproblem credit requires the retained contract payload")
    if _canonical_sha256(row.subproblem_contract) != row.subproblem_contract_sha256:
        raise ValueError("oracle subproblem contract payload hash mismatch")
    if not isinstance(row.subproblem_verifier_input, Mapping):
        raise ValueError("oracle subproblem credit requires verifier input")
    try:
        verify_alfworld_subproblem_annotation(
            active_subproblem_id=row.active_subproblem_id,
            contract=row.subproblem_contract,
            verifier_input=row.subproblem_verifier_input,
        )
    except AlfWorldSubproblemError as error:
        raise ValueError(f"oracle subproblem verification failed: {error}") from error


def _verify_goal_state_projection(
    row: DPPOTransition,
    *,
    previous: GoalVector | None,
) -> tuple[str, GoalVector]:
    if not isinstance(row.goal_graph, Mapping):
        raise ValueError("DPPO credit requires the retained goal graph")
    if not _SHA256.fullmatch(row.goal_graph_sha256 or ""):
        raise ValueError("DPPO credit requires goal_graph_sha256")
    if canonical_sha256(row.goal_graph) != row.goal_graph_sha256:
        raise ValueError("DPPO goal graph payload hash mismatch")
    if row.residual_goal_ids is None or row.active_frontier_ids is None:
        raise ValueError("DPPO credit requires residual goals and active frontier")
    if canonical_sha256(list(row.residual_goal_ids)) != row.residual_goal_signature:
        raise ValueError("DPPO residual goal signature mismatch")
    if canonical_sha256(list(row.active_frontier_ids)) != row.frontier_signature:
        raise ValueError("DPPO frontier signature mismatch")
    verifier_input = row.goal_verifier_input
    if not isinstance(verifier_input, Mapping):
        raise ValueError("DPPO credit requires retained goal-state verifier input")
    schema = str(verifier_input.get("schema", ""))
    if schema in {
        "lh_alfworld_goal_state_verifier_input/v1",
        "lh_alfworld_goal_state_verifier_input/v2",
        "lh_alfworld_goal_state_verifier_input/v3",
    }:
        task_metadata = verifier_input.get("task_metadata")
        facts = verifier_input.get("facts")
        if not isinstance(task_metadata, Mapping) or not isinstance(facts, (list, tuple)):
            raise ValueError("DPPO ALFWorld verifier input is incomplete")
        compiler = AlfWorldSubproblemCompiler()
        graph = compiler.compile_task(task_metadata)
        vector = compiler.project_state(task_metadata, facts, previous=previous)
        if schema.endswith(("/v2", "/v3")):
            _verify_independent_alfworld_semantics(
                row,
                task_metadata=task_metadata,
                graph=graph,
                vector=vector,
            )
    elif schema == "lh_appworld_goal_state_verifier_input/v1":
        raw_task = verifier_input.get("task_contract")
        evaluator_state = verifier_input.get("evaluator_state")
        if not isinstance(raw_task, Mapping) or not isinstance(evaluator_state, Mapping):
            raise ValueError("DPPO AppWorld verifier input is incomplete")
        task = AppWorldTaskContract(
            task_id=str(raw_task.get("task_id", "")),
            split=str(raw_task.get("split", "")),
            instruction=str(raw_task.get("instruction", "")),
            evaluator_requirements=tuple(
                str(v) for v in raw_task.get("evaluator_requirements", ())
            ),
            required_apps=tuple(str(v) for v in raw_task.get("required_apps", ())),
        )
        compiler = AppWorldSubproblemCompiler()
        graph = compiler.compile_task(task)
        vector = compiler.project_state(task, evaluator_state, previous=previous)
    else:
        raise ValueError("DPPO goal-state verifier input schema is unsupported")
    expected = goal_state_projection(graph, vector)
    if graph.to_dict() != dict(row.goal_graph):
        raise ValueError("DPPO goal graph does not match verifier input")
    if tuple(expected["residual_goal_ids"]) != tuple(row.residual_goal_ids or ()):
        raise ValueError("DPPO residual goals do not match verifier input")
    if tuple(expected["active_frontier_ids"]) != tuple(row.active_frontier_ids or ()):
        raise ValueError("DPPO frontier does not match verifier input")
    return schema, vector


def _verify_independent_alfworld_semantics(
    row: DPPOTransition,
    *,
    task_metadata: Mapping[str, Any],
    graph,
    vector: GoalVector,
) -> None:
    """Recompute versioned semantic receipts before refined credit consumes them."""

    if not row.goal_graph_semantic_verified or not row.goal_state_semantic_verified:
        raise ValueError(
            "DPPO versioned ALFWorld credit requires independent semantic verification"
        )
    verifier_input = row.subproblem_verifier_input
    canonical_facts = verifier_input.get("facts") if isinstance(verifier_input, Mapping) else None
    if not isinstance(canonical_facts, list):
        raise ValueError("DPPO versioned ALFWorld credit requires canonical semantic facts")
    graph_receipt = audit_alfworld_goal_graph_semantics(task_metadata, graph)
    state_receipt = audit_alfworld_snapshot_composition(
        task_metadata,
        graph,
        vector,
        canonical_facts,
    )
    if not graph_receipt.verified or not state_receipt.verified:
        raise ValueError("DPPO versioned ALFWorld independent semantic verification failed")
    if row.goal_graph_semantic_verifier_id != graph_receipt.verifier_id:
        raise ValueError("DPPO graph semantic verifier identity mismatch")
    if row.goal_state_semantic_verifier_id != state_receipt.verifier_id:
        raise ValueError("DPPO state semantic verifier identity mismatch")
    if canonical_sha256(graph_receipt.to_dict()) != (row.goal_graph_semantic_receipt_sha256):
        raise ValueError("DPPO graph semantic receipt hash mismatch")
    if canonical_sha256(state_receipt.to_dict()) != (row.goal_state_before_semantic_receipt_sha256):
        raise ValueError("DPPO state semantic receipt hash mismatch")
    if not _SHA256.fullmatch(row.goal_state_after_semantic_receipt_sha256 or ""):
        raise ValueError("DPPO versioned ALFWorld credit requires after-state semantic evidence")


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if value is not None else None


def _optional_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else None


def _optional_string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...] | None:
    value = payload.get(key)
    return tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else None


__all__ = [
    "DPPOCreditMode",
    "DPPOTransition",
    "assign_dppo_credit",
    "transitions_from_v2_records",
    "validate_dppo_transitions",
]
