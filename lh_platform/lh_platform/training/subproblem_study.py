"""Fail-closed contract for the P0 oracle-subproblem credit ablation.

This is deliberately narrower than a hierarchical-agent experiment.  It keeps
the actor prompt, rollout harness, and actor-token objective fixed, then tests
whether environment-verified phase boundaries change credit quality and later
long-horizon learning relative to G2PO-TN.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.dppo.credit import DPPOCreditMode, transitions_from_v2_records
from lh_platform.training.dppo.study import load_g2po_experiment_config
from lh_platform.training.trajectory_v2 import validate_active_trajectory_records

ORACLE_SUBPROBLEM_STUDY_SCHEMA = "lh_oracle_subproblem_credit_ablation/v1"


class OracleSubproblemStudyError(ValueError):
    """Raised when the P0 ablation confounds decomposition with harness changes."""


@dataclass(frozen=True)
class OracleSubproblemStudyPreflight:
    config_path: Path
    base_study_path: Path
    treatment_ids: tuple[str, ...]


@dataclass(frozen=True)
class OracleSubproblemTraceGate:
    """Evidence that the P0 state-splitting mechanism was actually exercised.

    A cross-goal-state observation group is deliberately only a *credit-collision
    candidate*: equal text observations at different verifier-owned phases do
    not by themselves prove a latent-state false merge.  They do prove that
    DPPO's residual/frontier key can differ from G2PO-TN on this trace.  The
    legacy ``cross_phase`` field names remain stable for artifact readers.
    """

    active_transitions: int
    annotation_coverage: float
    cross_phase_observation_groups: int
    cross_phase_transitions: int
    cross_phase_tasks: int
    cross_phase_candidate_transition_fraction: float
    phase_transition_counts: Mapping[str, int]
    legacy_phase_frontier_disagreement_count: int
    frontier_live_precondition_audit_coverage: float
    frontier_live_precondition_violation_count: int
    passed: bool
    blocked_by: tuple[str, ...]


def load_oracle_subproblem_study_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OracleSubproblemStudyError(
            f"cannot read oracle-subproblem study config: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise OracleSubproblemStudyError("oracle-subproblem study config must be a mapping")
    _validate_oracle_subproblem_config(raw, config_path=config_path)
    return raw


def preflight_oracle_subproblem_study(
    path: str | Path,
) -> OracleSubproblemStudyPreflight:
    config_path = Path(path).resolve()
    raw = load_oracle_subproblem_study_config(config_path)
    base_relative = raw["claim_boundary"]["base_study"]
    base_path = (config_path.parent.parent.parent / base_relative).resolve()
    try:
        base = load_g2po_experiment_config(base_path)
    except Exception as error:
        raise OracleSubproblemStudyError(f"base G2PO study is invalid: {error}") from error
    if raw["actor_loss"] != base["actor_loss"]:
        raise OracleSubproblemStudyError(
            "oracle-subproblem study must use the base actor-loss contract exactly"
        )
    return OracleSubproblemStudyPreflight(
        config_path=config_path,
        base_study_path=base_path,
        treatment_ids=tuple(item["id"] for item in raw["treatments"]),
    )


def assess_oracle_subproblem_trace(
    records: Iterable[TrainingTrajectoryRecord],
    *,
    min_cross_phase_observation_groups: int = 1,
    min_cross_phase_tasks: int = 1,
    min_cross_phase_candidate_transition_fraction: float = 0.0,
) -> OracleSubproblemTraceGate:
    """Fail closed unless a trace both carries and exercises P0 annotations."""

    rows = validate_active_trajectory_records(records, require_v2=True)
    annotated = [
        row
        for row in rows
        if row.extra.get("goal_graph_sha256")
        and row.extra.get("residual_goal_signature")
        and row.extra.get("frontier_signature")
        and row.extra.get("goal_annotation_verified") is True
        and row.extra.get("subproblem_source") == "environment_contract"
    ]
    coverage = len(annotated) / len(rows)
    frontier_audit_rows = [row for row in rows if has_frontier_audit_receipt(row)]
    frontier_audit_coverage = len(frontier_audit_rows) / len(rows)
    blocked: list[str] = []
    if coverage != 1.0:
        blocked.append("missing_or_non_environment_subproblem_annotation")
    if frontier_audit_coverage != 1.0:
        blocked.append("missing_frontier_live_precondition_audit_receipt")
    if coverage == 1.0 and frontier_audit_coverage == 1.0:
        # Run the same strict field validation used by the credit path, not a
        # weaker metrics-only reconstruction.
        transitions_from_v2_records(
            rows,
            mode=DPPOCreditMode.ORACLE_PHASE_REFINED_TN,
        )

    by_observation: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    phase_counts: Counter[str] = Counter()
    legacy_frontier_disagreements = 0
    live_precondition_violations = count_frontier_live_precondition_violations(frontier_audit_rows)
    for row in annotated:
        residual = str(row.extra["residual_goal_signature"])
        active_frontier = str(row.extra["frontier_signature"])
        by_observation[(row.group_uid, row.observation)].add((residual, active_frontier))
        frontier_ids = row.extra.get("active_frontier_ids")
        phase_counts[
            ",".join(str(value) for value in frontier_ids)
            if isinstance(frontier_ids, (list, tuple))
            else active_frontier
        ] += 1
        frontier = (
            tuple(str(value) for value in frontier_ids)
            if isinstance(frontier_ids, (list, tuple))
            else ()
        )
        legacy = row.extra.get("active_subproblem_id")
        if legacy is not None and not (
            str(legacy) in frontier or (str(legacy) == "complete" and not frontier)
        ):
            legacy_frontier_disagreements += 1
    cross_phase = {
        group_observation: identities
        for group_observation, identities in by_observation.items()
        if len(identities) > 1
    }
    cross_phase_transitions = sum(
        1 for row in annotated if (row.group_uid, row.observation) in cross_phase
    )
    cross_phase_tasks = {
        row.task_id for row in annotated if (row.group_uid, row.observation) in cross_phase
    }
    candidate_fraction = cross_phase_transitions / len(rows)
    if legacy_frontier_disagreements:
        blocked.append("legacy_phase_frontier_disagreement")
    if live_precondition_violations:
        blocked.append("frontier_live_precondition_violation")
    if len(cross_phase) < min_cross_phase_observation_groups:
        blocked.append("mechanism_not_triggered_no_cross_phase_observation_group")
    elif len(cross_phase_tasks) < min_cross_phase_tasks:
        blocked.append("mechanism_not_triggered_too_few_tasks")
    elif candidate_fraction < min_cross_phase_candidate_transition_fraction:
        blocked.append("mechanism_not_triggered_too_few_cross_phase_candidate_transitions")
    return OracleSubproblemTraceGate(
        active_transitions=len(rows),
        annotation_coverage=coverage,
        cross_phase_observation_groups=len(cross_phase),
        cross_phase_transitions=cross_phase_transitions,
        cross_phase_tasks=len(cross_phase_tasks),
        cross_phase_candidate_transition_fraction=candidate_fraction,
        phase_transition_counts=dict(sorted(phase_counts.items())),
        legacy_phase_frontier_disagreement_count=legacy_frontier_disagreements,
        frontier_live_precondition_audit_coverage=frontier_audit_coverage,
        frontier_live_precondition_violation_count=live_precondition_violations,
        passed=not blocked,
        blocked_by=tuple(blocked),
    )


def has_frontier_audit_receipt(row: TrainingTrajectoryRecord) -> bool:
    active = row.extra.get("active_frontier_ids")
    graph = row.extra.get("goal_graph")
    vector = row.extra.get("goal_vector_before")
    if not isinstance(active, (list, tuple)):
        return False
    if not isinstance(graph, Mapping) or not isinstance(vector, Mapping):
        return False
    nodes = graph.get("nodes")
    truth_rows = vector.get("truth")
    if not isinstance(nodes, (list, tuple)) or not isinstance(truth_rows, (list, tuple)):
        return False
    node_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("node_id"), str):
            return False
        if not isinstance(node.get("precondition_goal_ids"), (list, tuple)):
            return False
        node_ids.add(str(node["node_id"]))
    for truth_row in truth_rows:
        if (
            not isinstance(truth_row, (list, tuple))
            or len(truth_row) != 2
            or not isinstance(truth_row[0], str)
            or not isinstance(truth_row[1], bool)
        ):
            return False
    return all(isinstance(node_id, str) and node_id in node_ids for node_id in active)


def count_frontier_live_precondition_violations(
    records: Iterable[TrainingTrajectoryRecord],
) -> int:
    """Count transitions whose declared frontier is not executable now.

    This check intentionally reads the retained graph and live truth vector
    directly instead of trusting the frontier compiler that produced the
    annotation.  Historical achievement latches may explain progress, but may
    never satisfy a live action precondition.
    """

    violations = 0
    for row in records:
        active = row.extra.get("active_frontier_ids")
        graph = row.extra.get("goal_graph")
        vector = row.extra.get("goal_vector_before")
        if not isinstance(active, (list, tuple)):
            raise OracleSubproblemStudyError("frontier audit requires active_frontier_ids")
        if not isinstance(graph, Mapping) or not isinstance(vector, Mapping):
            raise OracleSubproblemStudyError(
                "frontier audit requires retained goal graph and live vector"
            )
        nodes = graph.get("nodes")
        truth_rows = vector.get("truth")
        if not isinstance(nodes, (list, tuple)) or not isinstance(truth_rows, (list, tuple)):
            raise OracleSubproblemStudyError(
                "frontier audit received malformed graph or live truth"
            )
        node_map = {str(node.get("node_id")): node for node in nodes if isinstance(node, Mapping)}
        try:
            truth = {
                str(goal_id): value for goal_id, value in truth_rows if isinstance(value, bool)
            }
        except (TypeError, ValueError) as error:
            raise OracleSubproblemStudyError(
                "frontier audit received malformed live truth rows"
            ) from error
        transition_violated = False
        for node_id in active:
            node = node_map.get(str(node_id))
            if not isinstance(node, Mapping):
                raise OracleSubproblemStudyError("frontier audit references an unknown node")
            preconditions = node.get("precondition_goal_ids")
            if not isinstance(preconditions, (list, tuple)):
                raise OracleSubproblemStudyError(
                    "frontier audit received malformed node preconditions"
                )
            if any(truth.get(str(goal_id)) is not True for goal_id in preconditions):
                transition_violated = True
                break
        violations += int(transition_violated)
    return violations


def _validate_oracle_subproblem_config(raw: Mapping[str, Any], *, config_path: Path) -> None:
    if raw.get("schema") != ORACLE_SUBPROBLEM_STUDY_SCHEMA:
        raise OracleSubproblemStudyError(f"expected schema {ORACLE_SUBPROBLEM_STUDY_SCHEMA!r}")
    claim = raw.get("claim_boundary")
    if not isinstance(claim, Mapping) or not isinstance(claim.get("base_study"), str):
        raise OracleSubproblemStudyError("claim_boundary.base_study is required")
    base = (config_path.parent.parent.parent / claim["base_study"]).resolve()
    root = config_path.parent.parent.parent.resolve()
    if root not in base.parents or not base.is_file():
        raise OracleSubproblemStudyError(
            "claim_boundary.base_study must be an existing workspace-relative file"
        )

    harness = raw.get("fixed_harness")
    expected_harness = {
        "actor_prompt_includes_active_subproblem": False,
        "planner_enabled": False,
        "controller_enabled": False,
        "context_policy": "identical_to_baseline",
        "evaluator_schedule": "identical_to_baseline",
        "rollout_budget": "identical_to_baseline",
    }
    if harness != expected_harness:
        raise OracleSubproblemStudyError(
            "P0 must keep prompt, planner, controller, and fixed harness identical to baseline"
        )

    oracle = raw.get("oracle_decomposition")
    expected_oracle = {
        "annotation_source": "environment_contract",
        "active_subproblem_only": True,
        "contract_must_be_environment_verified": True,
        "reference_action_sequences_allowed": False,
        "hidden_reasoning_allowed": False,
        "required_trace_fields": [
            "active_subproblem_id",
            "subproblem_contract",
            "subproblem_contract_sha256",
            "subproblem_contract_verified",
            "subproblem_source",
            "subproblem_verifier_input",
        ],
    }
    if oracle != expected_oracle:
        raise OracleSubproblemStudyError(
            "P0 oracle decomposition must use only environment-verified active contracts"
        )

    treatments = raw.get("treatments")
    expected_treatments = (
        ("g2po_tn", "g2po_trajectory_normalized", "source_observation_grouping"),
        (
            "dppo",
            "oracle_phase_refined_g2po_tn",
            "observation_plus_active_environment_verified_subproblem",
        ),
    )
    observed = (
        tuple(
            (item.get("id"), item.get("credit_mode"), item.get("state_merge"))
            for item in treatments
        )
        if isinstance(treatments, list)
        else ()
    )
    if observed != expected_treatments:
        raise OracleSubproblemStudyError(
            "P0 must compare exactly G2PO-TN and its oracle-subproblem credit variant"
        )

    gates = raw.get("gates")
    annotation_gate = gates.get("annotation") if isinstance(gates, Mapping) else None
    if annotation_gate != {
        "required_source": "environment_contract",
        "reject_reference_action_annotations": True,
        "on_missing": "fail_closed",
    }:
        raise OracleSubproblemStudyError(
            "P0 must fail closed on missing or non-environment subproblem annotations"
        )
    mechanism_gate = gates.get("mechanism") if isinstance(gates, Mapping) else None
    if mechanism_gate != {
        "min_annotation_coverage": 1.0,
        "min_cross_phase_observation_groups": 1,
        "on_not_triggered": "mechanism_not_triggered",
    }:
        raise OracleSubproblemStudyError(
            "P0 requires an annotation-coverage and cross-phase mechanism gate"
        )
