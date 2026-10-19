"""Retrospective natural-fragment anatomy for frozen ALFWorld policy traces.

This module asks a deliberately narrower question than trajectory composition:
do already-collected *failed* policy trajectories contain independently verified
child effects?  A positive result is support anatomy only.  Historical traces
without recoverable pre/post snapshots cannot become ``FragmentRecordV1`` and
cannot support a stitching, replay, training, or policy-improvement claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.studies.runtime import AlfworldTask, AlfworldTaskManifest
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.alfworld_semantic_verifier import (
    ALFWORLD_SEMANTIC_VERIFIER_ID,
    audit_alfworld_snapshot_composition,
)
from lh_platform.training.subproblems.base import GoalGraph, GoalVector, canonical_sha256

NATURAL_FRAGMENT_CANDIDATE_SCHEMA = "lh_natural_fragment_candidate/v1"
NATURAL_SUPPORT_TASK_RECEIPT_SCHEMA = "lh_natural_support_task_receipt/v1"
NATURAL_SUPPORT_REPORT_SCHEMA = "lh_natural_support_anatomy_report/v1"
NATURAL_SUPPORT_DECISION_PRESENT = (
    "RETROSPECTIVE_NATURAL_CHILD_EFFECT_SUPPORT_PRESENT_NOT_COMPOSABILITY"
)
NATURAL_SUPPORT_DECISION_ABSENT = "RETROSPECTIVE_NATURAL_CHILD_EFFECT_SUPPORT_NOT_ESTABLISHED"


class NaturalSupportError(RuntimeError):
    """Raised when historical trace evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class NaturalSupportGateConfig:
    min_failed_trajectories: int
    min_failed_with_verified_fragment: int
    min_task_types_with_verified_fragment: int
    min_wilson_lower_bound: float
    require_one_trajectory_per_task: bool = True

    def __post_init__(self) -> None:
        if self.min_failed_trajectories < 1:
            raise NaturalSupportError("minimum failed-trajectory denominator must be positive")
        if self.min_failed_with_verified_fragment < 1:
            raise NaturalSupportError("minimum fragment-support count must be positive")
        if self.min_task_types_with_verified_fragment < 1:
            raise NaturalSupportError("minimum fragment-support task types must be positive")
        if not 0.0 <= self.min_wilson_lower_bound <= 1.0:
            raise NaturalSupportError("Wilson lower-bound threshold must be in [0, 1]")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_failed_trajectories": self.min_failed_trajectories,
            "min_failed_with_verified_fragment": self.min_failed_with_verified_fragment,
            "min_task_types_with_verified_fragment": (self.min_task_types_with_verified_fragment),
            "min_wilson_lower_bound": self.min_wilson_lower_bound,
            "require_one_trajectory_per_task": self.require_one_trajectory_per_task,
        }


@dataclass(frozen=True)
class NaturalFragmentCandidateV1:
    task_id: str
    traj_uid: str
    source_policy_checkpoint_sha256: str
    boundary_step_idx: int
    child_goal_id: str
    child_node_ids: tuple[str, ...]
    prefix_action_count: int
    prefix_action_trace_sha256: str
    boundary_action: str
    persistent_at_trajectory_end: bool
    source_parent_success: bool
    independent_verifier_id: str
    independent_verifier_receipt_sha256: str
    replay_ready: bool = False
    training_eligible: bool = False
    schema: str = NATURAL_FRAGMENT_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NATURAL_FRAGMENT_CANDIDATE_SCHEMA:
            raise NaturalSupportError("unsupported natural-fragment candidate schema")
        for name, value in (
            ("task_id", self.task_id),
            ("traj_uid", self.traj_uid),
            ("child_goal_id", self.child_goal_id),
            ("boundary_action", self.boundary_action),
        ):
            if not value or value != value.strip():
                raise NaturalSupportError(f"natural fragment requires canonical {name}")
        for name, value in (
            ("source_policy_checkpoint_sha256", self.source_policy_checkpoint_sha256),
            ("prefix_action_trace_sha256", self.prefix_action_trace_sha256),
            (
                "independent_verifier_receipt_sha256",
                self.independent_verifier_receipt_sha256,
            ),
        ):
            _require_sha256(name, value)
        if self.boundary_step_idx < 0 or self.prefix_action_count != self.boundary_step_idx + 1:
            raise NaturalSupportError("fragment prefix boundary is inconsistent")
        if not self.child_node_ids:
            raise NaturalSupportError("candidate effect is not owned by a child node")
        if self.replay_ready or self.training_eligible:
            raise NaturalSupportError(
                "historical candidates without snapshots/replay cannot enter training"
            )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "traj_uid": self.traj_uid,
            "source_policy_checkpoint_sha256": self.source_policy_checkpoint_sha256,
            "boundary_step_idx": self.boundary_step_idx,
            "child_goal_id": self.child_goal_id,
            "child_node_ids": list(self.child_node_ids),
            "prefix_action_count": self.prefix_action_count,
            "prefix_action_trace_sha256": self.prefix_action_trace_sha256,
            "boundary_action": self.boundary_action,
            "persistent_at_trajectory_end": self.persistent_at_trajectory_end,
            "source_parent_success": self.source_parent_success,
            "independent_verifier_id": self.independent_verifier_id,
            "independent_verifier_receipt_sha256": (self.independent_verifier_receipt_sha256),
            "replay_ready": self.replay_ready,
            "training_eligible": self.training_eligible,
        }


@dataclass(frozen=True)
class NaturalSupportTaskReceiptV1:
    task_id: str
    group_uid: str
    task_type: str
    horizon_bucket: str
    traj_uid: str
    trajectory_steps: int
    source_parent_success: bool
    parent_verifier_agrees: bool
    verified_fragment_candidates: tuple[NaturalFragmentCandidateV1, ...]
    final_verified_goal_ids: tuple[str, ...]
    final_parent_verifier_receipt_sha256: str
    schema: str = NATURAL_SUPPORT_TASK_RECEIPT_SCHEMA

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "group_uid": self.group_uid,
            "task_type": self.task_type,
            "horizon_bucket": self.horizon_bucket,
            "traj_uid": self.traj_uid,
            "trajectory_steps": self.trajectory_steps,
            "source_parent_success": self.source_parent_success,
            "parent_verifier_agrees": self.parent_verifier_agrees,
            "verified_fragment_candidates": [
                candidate.to_dict() for candidate in self.verified_fragment_candidates
            ],
            "final_verified_goal_ids": list(self.final_verified_goal_ids),
            "final_parent_verifier_receipt_sha256": (self.final_parent_verifier_receipt_sha256),
        }


def analyze_historical_alfworld_natural_support(
    *,
    trace_path: str | Path,
    result_path: str | Path,
    manifest_path: str | Path,
    data_root: str | Path,
    expected_trace_sha256: str,
    expected_result_sha256: str,
    expected_manifest_sha256: str,
    expected_trace_rows: int,
    expected_task_count: int,
    expected_policy_checkpoint_sha256: str,
    expected_sampling_temperature: float,
    gate: NaturalSupportGateConfig,
    expected_split: str = "valid_unseen",
) -> dict[str, Any]:
    """Analyze every frozen trajectory without selecting on effects or outcomes."""

    trace = Path(trace_path).resolve()
    result = Path(result_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    root = Path(data_root).resolve()
    for name, path, expected in (
        ("trace", trace, expected_trace_sha256),
        ("result", result, expected_result_sha256),
        ("manifest", manifest_file, expected_manifest_sha256),
    ):
        actual = _sha256_file(path)
        if actual != expected:
            raise NaturalSupportError(f"historical {name} SHA-256 changed: {actual}")
    source_result = _read_json(result)
    _validate_source_result(
        source_result,
        expected_trace_sha256=expected_trace_sha256,
        expected_trace_rows=expected_trace_rows,
        expected_task_count=expected_task_count,
        expected_policy_checkpoint_sha256=expected_policy_checkpoint_sha256,
    )
    manifest = AlfworldTaskManifest.read(manifest_file)
    issues = manifest.validate(root)
    if issues:
        raise NaturalSupportError(
            "historical ALFWorld manifest/data invalid: "
            + "; ".join(f"{issue.code}:{issue.message}" for issue in issues[:10])
        )
    if len(manifest.tasks) != expected_task_count:
        raise NaturalSupportError("historical manifest task count changed")
    task_by_id = {task.task_id: task for task in manifest.tasks}
    rows_by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_count = 0
    with trace.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise NaturalSupportError(f"blank trace row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise NaturalSupportError(
                    f"invalid trace JSON at line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise NaturalSupportError("historical trace row must be a mapping")
            _validate_trace_row(
                row,
                expected_policy_checkpoint_sha256=expected_policy_checkpoint_sha256,
                expected_sampling_temperature=expected_sampling_temperature,
                expected_split=expected_split,
            )
            rows_by_traj[str(row["traj_uid"])].append(row)
            row_count += 1
    if row_count != expected_trace_rows:
        raise NaturalSupportError(
            f"historical trace row count changed: {row_count} != {expected_trace_rows}"
        )
    tasks_by_traj: dict[str, str] = {}
    for traj_uid, rows in rows_by_traj.items():
        task_ids = {str(row["task_id"]) for row in rows}
        if len(task_ids) != 1:
            raise NaturalSupportError(f"trajectory {traj_uid!r} spans multiple tasks")
        tasks_by_traj[traj_uid] = next(iter(task_ids))
    trajectory_count_by_task = Counter(tasks_by_traj.values())
    if gate.require_one_trajectory_per_task and any(
        count != 1 for count in trajectory_count_by_task.values()
    ):
        raise NaturalSupportError("N0 requires exactly one historical trajectory per task")
    if set(trajectory_count_by_task) != set(task_by_id):
        raise NaturalSupportError("historical trace does not cover the frozen manifest exactly")

    compiler = AlfWorldSubproblemCompiler()
    task_receipts: list[NaturalSupportTaskReceiptV1] = []
    for traj_uid in sorted(rows_by_traj):
        task_id = tasks_by_traj[traj_uid]
        task = task_by_id.get(task_id)
        if task is None:
            raise NaturalSupportError(f"trace task is outside manifest: {task_id}")
        metadata = _load_task_metadata(task, root)
        task_receipts.append(
            analyze_natural_support_trajectory(
                task=task,
                traj_uid=traj_uid,
                rows=tuple(rows_by_traj[traj_uid]),
                metadata=metadata,
                compiler=compiler,
                expected_policy_checkpoint_sha256=expected_policy_checkpoint_sha256,
            )
        )

    failures = tuple(item for item in task_receipts if not item.source_parent_success)
    failures_with_fragment = tuple(item for item in failures if item.verified_fragment_candidates)
    failures_with_persistent = tuple(
        item
        for item in failures
        if any(
            candidate.persistent_at_trajectory_end
            for candidate in item.verified_fragment_candidates
        )
    )
    failed_support_types = tuple(sorted({item.task_type for item in failures_with_fragment}))
    failed_support_horizons = tuple(
        sorted({item.horizon_bucket for item in failures_with_fragment})
    )
    wilson = (
        wilson_interval(len(failures_with_fragment), len(failures))
        if failures
        else (0.0, 1.0)
    )
    gates = {
        "failed_denominator": len(failures) >= gate.min_failed_trajectories,
        "failed_fragment_count": (
            len(failures_with_fragment) >= gate.min_failed_with_verified_fragment
        ),
        "task_type_diversity": (
            len(failed_support_types) >= gate.min_task_types_with_verified_fragment
        ),
        "wilson_lower_bound": wilson[0] >= gate.min_wilson_lower_bound,
        "one_trajectory_per_task": all(count == 1 for count in trajectory_count_by_task.values()),
        "parent_verifier_agreement": all(item.parent_verifier_agrees for item in task_receipts),
        "no_replay_or_training_upgrade": all(
            not candidate.replay_ready and not candidate.training_eligible
            for item in task_receipts
            for candidate in item.verified_fragment_candidates
        ),
    }
    gate_passed = all(gates.values())
    effect_counts = Counter(
        candidate.child_goal_id
        for item in failures
        for candidate in item.verified_fragment_candidates
    )
    failed_by_type = _stratified_counts(failures, failures_with_fragment, "task_type")
    failed_by_horizon = _stratified_counts(
        failures,
        failures_with_fragment,
        "horizon_bucket",
    )
    report = {
        "schema": NATURAL_SUPPORT_REPORT_SCHEMA,
        "scientific_result": False,
        "retrospective_exploratory": True,
        "claim_scope": "natural_failed_trajectory_child_effect_anatomy_only",
        "source": {
            "trace_path": str(trace),
            "trace_sha256": expected_trace_sha256,
            "trace_rows": row_count,
            "result_path": str(result),
            "result_sha256": expected_result_sha256,
            "manifest_path": str(manifest_file),
            "manifest_sha256": expected_manifest_sha256,
            "policy_checkpoint_sha256": expected_policy_checkpoint_sha256,
            "sampling_temperature": expected_sampling_temperature,
            "split": expected_split,
            "trajectory_count": len(task_receipts),
            "one_trajectory_per_task": True,
        },
        "gate_config": gate.to_dict(),
        "gate_config_sha256": gate.sha256,
        "metrics": {
            "task_count": len(task_receipts),
            "success_count": sum(item.source_parent_success for item in task_receipts),
            "failed_trajectory_count": len(failures),
            "failed_with_verified_fragment_count": len(failures_with_fragment),
            "failed_with_persistent_fragment_count": len(failures_with_persistent),
            "failed_fragment_rate": (
                len(failures_with_fragment) / len(failures) if failures else None
            ),
            "failed_fragment_rate_wilson_95": list(wilson),
            "failed_fragment_task_types": list(failed_support_types),
            "failed_fragment_horizons": list(failed_support_horizons),
            "failed_effect_counts": dict(sorted(effect_counts.items())),
            "failed_by_task_type": failed_by_type,
            "failed_by_horizon": failed_by_horizon,
        },
        "gates": gates,
        "gate_passed": gate_passed,
        "decision": (
            NATURAL_SUPPORT_DECISION_PRESENT if gate_passed else NATURAL_SUPPORT_DECISION_ABSENT
        ),
        "task_receipts": [item.to_dict() for item in task_receipts],
        "task_receipt_sha256s": {item.task_id: item.sha256 for item in task_receipts},
        "non_claims": [
            "same_instance_multi_trajectory_composability",
            "exact_or_e1_junction_validity",
            "full_parent_replay_success",
            "natural_fragment_population_rate",
            "training_data_eligibility",
            "training_or_credit_effect",
            "policy_improvement",
        ],
        "next_gate": (
            "prospective_same_instance_multi_rollout_collection_with_snapshots_and_replay"
            if gate_passed
            else "stop_cvfc_natural_line_and_return_to_actor_context_harness_support"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def analyze_natural_support_trajectory(
    *,
    task: AlfworldTask,
    traj_uid: str,
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    compiler: AlfWorldSubproblemCompiler,
    expected_policy_checkpoint_sha256: str,
) -> NaturalSupportTaskReceiptV1:
    """Analyze one trajectory while preserving its complete failure denominator."""

    if not rows:
        raise NaturalSupportError("natural-support trajectory is empty")
    ordered = tuple(sorted(rows, key=lambda item: int(item["step_idx"])))
    if [int(row["step_idx"]) for row in ordered] != list(range(len(ordered))):
        raise NaturalSupportError(f"trajectory {traj_uid!r} steps are not contiguous")
    terminal = [index for index, row in enumerate(ordered) if bool(row["terminal"])]
    if terminal != [len(ordered) - 1]:
        raise NaturalSupportError(f"trajectory {traj_uid!r} requires one terminal final transition")
    task_ids = {str(row["task_id"]) for row in ordered}
    trajectory_ids = {str(row["traj_uid"]) for row in ordered}
    policy_ids = {str(row["policy_checkpoint_sha256"]) for row in ordered}
    if task_ids != {task.task_id} or trajectory_ids != {traj_uid}:
        raise NaturalSupportError("trajectory identity changed within rows")
    if policy_ids != {expected_policy_checkpoint_sha256}:
        raise NaturalSupportError("trajectory policy identity changed")

    graph = compiler.compile_task(metadata)
    initial_vector = compiler.project_state(metadata, ordered[0]["environment_facts"])
    vector = initial_vector
    actions: list[str] = []
    first_candidate_by_goal: dict[str, NaturalFragmentCandidateV1] = {}
    source_parent_success = any(bool(row["success"]) for row in ordered)
    for index, row in enumerate(ordered):
        current = compiler.project_state(
            metadata,
            row["environment_facts"],
            previous=vector if index else None,
        )
        if current.truth_map != vector.truth_map:
            raise NaturalSupportError(
                f"trajectory {traj_uid!r} before-facts disagree at step {index}"
            )
        action = str(row["environment_action"]).strip()
        actions.append(action)
        after = compiler.project_state(
            metadata,
            row["next_environment_facts"],
            previous=current,
        )
        label = compiler.label_transition(graph, current, action, after)
        compiler_receipt = compiler.verify_annotation(label)
        semantic = audit_alfworld_snapshot_composition(
            metadata,
            graph,
            after,
            row["next_environment_facts"],
        )
        if not compiler_receipt.verified or not semantic.verified:
            raise NaturalSupportError(
                f"independent child verifier failed for {traj_uid!r} step {index}"
            )
        semantic_payload = semantic.to_dict()
        for goal_id in label.newly_satisfied_goal_ids:
            child_nodes = tuple(
                sorted(
                    node.node_id
                    for node in graph.nodes
                    if node.node_id != graph.root_id and goal_id in node.effect_goal_ids
                )
            )
            if not child_nodes or goal_id in first_candidate_by_goal:
                continue
            first_candidate_by_goal[goal_id] = NaturalFragmentCandidateV1(
                task_id=task.task_id,
                traj_uid=traj_uid,
                source_policy_checkpoint_sha256=expected_policy_checkpoint_sha256,
                boundary_step_idx=index,
                child_goal_id=goal_id,
                child_node_ids=child_nodes,
                prefix_action_count=len(actions),
                prefix_action_trace_sha256=canonical_sha256({"canonical_actions": list(actions)}),
                boundary_action=action,
                persistent_at_trajectory_end=False,
                source_parent_success=source_parent_success,
                independent_verifier_id=ALFWORLD_SEMANTIC_VERIFIER_ID,
                independent_verifier_receipt_sha256=canonical_sha256(
                    {
                        "compiler_receipt": {
                            "verified": compiler_receipt.verified,
                            "compiler_id": compiler_receipt.compiler_id,
                            "annotation_sha256": compiler_receipt.annotation_sha256,
                            "verifier_input_sha256": (compiler_receipt.verifier_input_sha256),
                            "source": compiler_receipt.source,
                        },
                        "semantic_receipt": semantic_payload,
                    }
                ),
            )
        vector = after
    final_semantic = audit_alfworld_snapshot_composition(
        metadata,
        graph,
        vector,
        ordered[-1]["next_environment_facts"],
    )
    if not final_semantic.verified:
        raise NaturalSupportError("final independent parent verifier failed")
    candidates = tuple(
        _with_persistence(candidate, graph=graph, final_vector=vector)
        for candidate in sorted(
            first_candidate_by_goal.values(),
            key=lambda item: (item.boundary_step_idx, item.child_goal_id),
        )
    )
    parent_agrees = bool(final_semantic.parent_contract_passed) == source_parent_success
    if not parent_agrees:
        raise NaturalSupportError(
            f"environment and independent parent verifier disagree for {task.task_id}"
        )
    return NaturalSupportTaskReceiptV1(
        task_id=task.task_id,
        group_uid=task.group_uid,
        task_type=task.task_type,
        horizon_bucket=task.horizon_bucket,
        traj_uid=traj_uid,
        trajectory_steps=len(ordered),
        source_parent_success=source_parent_success,
        parent_verifier_agrees=parent_agrees,
        verified_fragment_candidates=candidates,
        final_verified_goal_ids=_verified_goal_ids(graph, vector),
        final_parent_verifier_receipt_sha256=canonical_sha256(final_semantic.to_dict()),
    )


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval without optional dependencies."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise NaturalSupportError("Wilson counts must be integers")
    if successes < 0 or total < 1 or successes > total:
        raise NaturalSupportError("invalid Wilson count inputs")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def _with_persistence(
    candidate: NaturalFragmentCandidateV1,
    *,
    graph: GoalGraph,
    final_vector: GoalVector,
) -> NaturalFragmentCandidateV1:
    atom = next(atom for atom in graph.atoms if atom.goal_id == candidate.child_goal_id)
    persistent = (
        final_vector.truth_map[candidate.child_goal_id]
        if atom.invariant
        else candidate.child_goal_id in final_vector.achievement_latches
    )
    return NaturalFragmentCandidateV1(
        **{
            **candidate.__dict__,
            "persistent_at_trajectory_end": persistent,
        }
    )


def _verified_goal_ids(graph: GoalGraph, vector: GoalVector) -> tuple[str, ...]:
    return tuple(
        sorted(
            atom.goal_id
            for atom in graph.atoms
            if (
                vector.truth_map[atom.goal_id]
                if atom.invariant
                else atom.goal_id in vector.achievement_latches
            )
        )
    )


def _stratified_counts(
    failures: Sequence[NaturalSupportTaskReceiptV1],
    supported: Sequence[NaturalSupportTaskReceiptV1],
    field: str,
) -> dict[str, dict[str, int]]:
    denominator = Counter(str(getattr(item, field)) for item in failures)
    numerator = Counter(str(getattr(item, field)) for item in supported)
    return {
        key: {
            "failed_trajectories": denominator[key],
            "failed_with_verified_fragment": numerator[key],
        }
        for key in sorted(denominator)
    }


def _validate_source_result(
    payload: Mapping[str, Any],
    *,
    expected_trace_sha256: str,
    expected_trace_rows: int,
    expected_task_count: int,
    expected_policy_checkpoint_sha256: str,
) -> None:
    if payload.get("schema") != "lh_agentic_rl_full_evaluation/v1":
        raise NaturalSupportError("unsupported historical evaluation result schema")
    expected = {
        "evaluation_trace_sha256": expected_trace_sha256,
        "evaluation_trace_rows": expected_trace_rows,
        "policy_checkpoint_sha256": expected_policy_checkpoint_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise NaturalSupportError(f"historical result field changed: {key}")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("task_count") != expected_task_count:
        raise NaturalSupportError("historical evaluation task count changed")
    if payload.get("claim_scope") != "baseline_evaluation_only_no_training":
        raise NaturalSupportError("historical source was not a frozen no-training evaluation")


def _validate_trace_row(
    row: Mapping[str, Any],
    *,
    expected_policy_checkpoint_sha256: str,
    expected_sampling_temperature: float,
    expected_split: str,
) -> None:
    required = {
        "schema",
        "task_id",
        "traj_uid",
        "step_idx",
        "split",
        "environment_facts",
        "next_environment_facts",
        "environment_action",
        "terminal",
        "success",
        "policy_checkpoint_sha256",
        "sampling_temperature",
        "action_token_ids",
        "action_token_log_probs",
        "horizon_bucket",
        "reference_steps",
    }
    missing = sorted(required - set(row))
    if missing:
        raise NaturalSupportError(f"historical trace row missing fields: {missing}")
    if row.get("schema") != "lh_alfworld_evaluation_transition/v1":
        raise NaturalSupportError("unsupported historical transition schema")
    if expected_split not in {"valid_seen", "valid_unseen"}:
        raise NaturalSupportError("support anatomy split is unsupported")
    if row.get("split") != expected_split:
        raise NaturalSupportError("historical support anatomy split changed")
    if row.get("policy_checkpoint_sha256") != expected_policy_checkpoint_sha256:
        raise NaturalSupportError("historical trace policy checkpoint changed")
    if float(row.get("sampling_temperature")) != expected_sampling_temperature:
        raise NaturalSupportError("historical trace sampling temperature changed")
    token_ids = row.get("action_token_ids")
    logprobs = row.get("action_token_log_probs")
    if (
        not isinstance(token_ids, list)
        or not isinstance(logprobs, list)
        or not token_ids
        or len(token_ids) != len(logprobs)
    ):
        raise NaturalSupportError("historical trace token/logprob provenance is invalid")
    if not isinstance(row.get("environment_facts"), list) or not isinstance(
        row.get("next_environment_facts"), list
    ):
        raise NaturalSupportError("historical trace lacks canonical fact lists")
    if not str(row.get("environment_action", "")).strip():
        raise NaturalSupportError("historical transition lacks a semantic action")


def _load_task_metadata(task: AlfworldTask, data_root: Path) -> dict[str, Any]:
    path = (data_root / task.trajectory_path).resolve()
    if not path.is_relative_to(data_root):
        raise NaturalSupportError("task metadata path escapes ALFWorld data root")
    raw = _read_json(path)
    params = raw.get("pddl_params")
    if not isinstance(params, Mapping):
        raise NaturalSupportError("ALFWorld task metadata lacks pddl_params")
    if raw.get("task_type") != task.task_type:
        raise NaturalSupportError("manifest and task metadata task_type differ")
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "pddl_params": {
            "object_target": params.get("object_target", ""),
            "parent_target": params.get("parent_target", ""),
            "toggle_target": params.get("toggle_target", ""),
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NaturalSupportError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise NaturalSupportError(f"JSON artifact is not a mapping: {path}")
    return value


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise NaturalSupportError(f"{name} must be a lowercase SHA-256")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
