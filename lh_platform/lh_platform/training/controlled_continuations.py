"""Analyze frozen continuation programs from verified ALFWorld state pairs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.controlled_aliasing import (
    analyze_controlled_aliasing_artifact,
    validate_alfworld_replay_evidence,
)
from lh_platform.training.dppo.analysis import load_rollout_artifact
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.base import goal_state_projection

CONTROLLED_CONTINUATION_ANALYSIS_SCHEMA = "lh_dppo_controlled_continuation_analysis/v1"
CONTROLLED_CONTINUATION_MATRIX_SCHEMA = "lh_alfworld_controlled_continuation_matrix/v1"
ALFWORLD_CONTROLLED_CONTINUATION_OUTCOME_SCHEMA = "lh_alfworld_controlled_continuation_outcome/v1"
ALFWORLD_CONTROLLED_CONTINUATION_OUTCOME_ARTIFACT_SCHEMA = (
    "lh_alfworld_controlled_continuation_outcome_artifact/v1"
)
CONTROLLED_CONTINUATION_EVIDENCE_CLAIM_SCOPE = (
    "controlled_continuation_score_heterogeneity_not_policy_effectiveness"
)
CONTROLLED_CONTINUATION_STATS_CLAIM_SCOPE = (
    "controlled_continuation_statistics_only_not_evidence_admission"
)
CONTROLLED_CONTINUATION_CONTROLLER_SOURCE = "preregistered_environment_intervention_program"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ControlledContinuationAnalysisError(ValueError):
    """Raised when a continuation matrix violates its controlled design."""


@dataclass(frozen=True)
class ControlledContinuationOutcomeEvidence:
    """Benchmark-owned result plus an independent fact-goal verification."""

    benchmark_score: float
    official_won: bool
    environment_done: bool
    verifier_complete: bool
    residual_goal_ids: tuple[str, ...]
    artifact_path: str = ""
    artifact_sha256: str = ""
    benchmark_source: str = "official_alfworld_infos"
    verifier_source: str = "environment_fact_goal_compiler"
    schema: str = ALFWORLD_CONTROLLED_CONTINUATION_OUTCOME_SCHEMA

    @classmethod
    def from_mapping(cls, value: object) -> ControlledContinuationOutcomeEvidence:
        if not isinstance(value, Mapping):
            raise ControlledContinuationAnalysisError(
                "controlled continuation requires structured Outcome Evidence"
            )
        if value.get("schema") != ALFWORLD_CONTROLLED_CONTINUATION_OUTCOME_SCHEMA:
            raise ControlledContinuationAnalysisError(
                "unsupported controlled continuation Outcome Evidence schema"
            )
        try:
            score = float(value.get("benchmark_score"))
        except (TypeError, ValueError) as error:
            raise ControlledContinuationAnalysisError(
                "ALFWorld benchmark score must be numeric"
            ) from error
        if not math.isfinite(score) or score < 0.0:
            raise ControlledContinuationAnalysisError(
                "ALFWorld benchmark score must be finite and non-negative"
            )
        won = value.get("official_won")
        done = value.get("environment_done")
        complete = value.get("verifier_complete")
        if not all(isinstance(item, bool) for item in (won, done, complete)):
            raise ControlledContinuationAnalysisError(
                "official won, environment done, and verifier complete must be booleans"
            )
        residual = _string_sequence(
            value.get("residual_goal_ids"),
            "residual_goal_ids",
            allow_empty=True,
        )
        if value.get("benchmark_source") != "official_alfworld_infos":
            raise ControlledContinuationAnalysisError(
                "Outcome Evidence is not bound to official ALFWorld infos"
            )
        if value.get("verifier_source") != "environment_fact_goal_compiler":
            raise ControlledContinuationAnalysisError(
                "Outcome Evidence is not bound to the fact-goal compiler"
            )
        if complete != (not residual):
            raise ControlledContinuationAnalysisError(
                "verifier completion disagrees with residual goals"
            )
        if won != complete:
            raise ControlledContinuationAnalysisError(
                "official ALFWorld outcome disagrees with the independent verifier"
            )
        if won != (score > 0.0):
            raise ControlledContinuationAnalysisError(
                "ALFWorld benchmark score disagrees with official won"
            )
        return cls(
            benchmark_score=score,
            official_won=won,
            environment_done=done,
            verifier_complete=complete,
            residual_goal_ids=residual,
            artifact_path=str(value.get("artifact_path", "")).strip(),
            artifact_sha256=str(value.get("artifact_sha256", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "benchmark_source": self.benchmark_source,
            "benchmark_score": self.benchmark_score,
            "official_won": self.official_won,
            "environment_done": self.environment_done,
            "verifier_source": self.verifier_source,
            "verifier_complete": self.verifier_complete,
            "residual_goal_ids": list(self.residual_goal_ids),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class ControlledContinuationCell:
    """One state-by-program cell in the preregistered continuation matrix."""

    state_id: str
    frontier_ids: tuple[str, ...]
    state_snapshot_sha256: str
    program_id: str
    program_actions: tuple[str, ...]
    executed_actions: tuple[str, ...]
    first_action_state_snapshot_sha256: str
    first_action_restore_verified: bool
    outcome_evidence: ControlledContinuationOutcomeEvidence
    termination: str = ""
    step_evidence: tuple[Mapping[str, Any], ...] = ()
    controller_source: str = ""
    model_policy_used: bool | None = None

    @classmethod
    def from_mapping(cls, row: object) -> ControlledContinuationCell:
        if not isinstance(row, Mapping):
            raise ControlledContinuationAnalysisError(
                "controlled continuation row must be a mapping"
            )
        state_id = _required_text(row, "state_id")
        program_id = _required_text(row, "program_id")
        frontier_ids = _string_sequence(row.get("frontier_ids"), "frontier_ids")
        program_actions = _string_sequence(row.get("program_actions"), "program_actions")
        executed_actions = _string_sequence(row.get("executed_actions"), "executed_actions")
        if program_actions != executed_actions:
            raise ControlledContinuationAnalysisError(
                f"continuation program was not executed exactly: {state_id}/{program_id}"
            )
        snapshot = _sha256(row.get("state_snapshot_sha256"), "state snapshot")
        first_snapshot = _sha256(
            row.get("first_action_state_snapshot_sha256"),
            "first-action state snapshot",
        )
        if snapshot != first_snapshot:
            raise ControlledContinuationAnalysisError(
                f"continuation did not start from the admitted state: {state_id}/{program_id}"
            )
        first_restore = row.get("first_action_restore_verified")
        if first_restore is not True:
            raise ControlledContinuationAnalysisError(
                f"first continuation action lacks restore evidence: {state_id}/{program_id}"
            )
        evidence_rows = row.get("step_evidence", ())
        if not isinstance(evidence_rows, (list, tuple)) or not all(
            isinstance(item, Mapping) for item in evidence_rows
        ):
            raise ControlledContinuationAnalysisError(
                "step_evidence must be a sequence of mappings"
            )
        model_policy_used = row.get("model_policy_used")
        if model_policy_used is not None and not isinstance(model_policy_used, bool):
            raise ControlledContinuationAnalysisError(
                "model_policy_used must be boolean when present"
            )
        return cls(
            state_id=state_id,
            frontier_ids=frontier_ids,
            state_snapshot_sha256=snapshot,
            program_id=program_id,
            program_actions=program_actions,
            executed_actions=executed_actions,
            first_action_state_snapshot_sha256=first_snapshot,
            first_action_restore_verified=True,
            outcome_evidence=ControlledContinuationOutcomeEvidence.from_mapping(
                row.get("outcome_evidence")
            ),
            termination=str(row.get("termination", "")).strip(),
            step_evidence=tuple(dict(item) for item in evidence_rows),
            controller_source=str(row.get("controller_source", "")).strip(),
            model_policy_used=model_policy_used,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "frontier_ids": list(self.frontier_ids),
            "state_snapshot_sha256": self.state_snapshot_sha256,
            "program_id": self.program_id,
            "program_actions": list(self.program_actions),
            "executed_actions": list(self.executed_actions),
            "first_action_state_snapshot_sha256": (self.first_action_state_snapshot_sha256),
            "first_action_restore_verified": self.first_action_restore_verified,
            "outcome_evidence": self.outcome_evidence.to_dict(),
            "termination": self.termination,
            "step_evidence": [dict(item) for item in self.step_evidence],
            "controller_source": self.controller_source,
            "model_policy_used": self.model_policy_used,
        }


def analyze_controlled_continuation_artifact(
    matrix_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Independently admit one matrix only after revalidating all evidence."""

    source = Path(matrix_path).resolve()
    payload = _read_matrix(source)
    cells = tuple(ControlledContinuationCell.from_mapping(row) for row in payload["rows"])
    evidence = _validate_artifact_evidence(payload, cells)
    analysis = build_controlled_continuation_analysis(cells, gamma=payload["gamma"])
    analysis["evidence"] = evidence
    analysis["source_matrix"] = {
        "path": str(source),
        "sha256": _sha256_file(source),
    }
    analysis["admitted"] = analysis["mechanism_triggered"]
    analysis["blocked_by"] = (
        []
        if analysis["mechanism_triggered"]
        else ["no_phase_conditioned_continuation_score_heterogeneity"]
    )
    analysis["claim_scope"] = CONTROLLED_CONTINUATION_EVIDENCE_CLAIM_SCOPE
    if output_path is not None:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return analysis


def build_controlled_continuation_analysis(
    rows: Iterable[Mapping[str, Any] | ControlledContinuationCell],
    *,
    gamma: float,
) -> dict[str, Any]:
    """Compute statistics only; this function cannot admit evidence."""

    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ControlledContinuationAnalysisError("continuation gamma must be between zero and one")
    cells = tuple(
        row
        if isinstance(row, ControlledContinuationCell)
        else ControlledContinuationCell.from_mapping(row)
        for row in rows
    )
    if not cells:
        raise ControlledContinuationAnalysisError("controlled continuation matrix is empty")

    by_state: dict[str, list[ControlledContinuationCell]] = defaultdict(list)
    by_program: dict[str, list[ControlledContinuationCell]] = defaultdict(list)
    matrix_cells: set[tuple[str, str]] = set()
    for cell in cells:
        identity = (cell.state_id, cell.program_id)
        if identity in matrix_cells:
            raise ControlledContinuationAnalysisError(
                f"duplicate continuation matrix cell: {identity[0]}/{identity[1]}"
            )
        matrix_cells.add(identity)
        by_state[cell.state_id].append(cell)
        by_program[cell.program_id].append(cell)

    if len(by_state) != 2:
        raise ControlledContinuationAnalysisError(
            "controlled continuation analysis requires exactly two states"
        )
    if len(by_program) < 2:
        raise ControlledContinuationAnalysisError(
            "controlled continuation analysis requires at least two programs"
        )
    expected_cells = {(state_id, program_id) for state_id in by_state for program_id in by_program}
    if matrix_cells != expected_cells:
        raise ControlledContinuationAnalysisError(
            "continuation matrix must execute every program from every state"
        )

    state_snapshots: dict[str, str] = {}
    state_frontiers: dict[str, tuple[str, ...]] = {}
    for state_id, state_cells in by_state.items():
        snapshots = {cell.state_snapshot_sha256 for cell in state_cells}
        frontiers = {cell.frontier_ids for cell in state_cells}
        if len(snapshots) != 1 or len(frontiers) != 1:
            raise ControlledContinuationAnalysisError(
                f"state identity drifted across programs: {state_id}"
            )
        state_snapshots[state_id] = snapshots.pop()
        state_frontiers[state_id] = frontiers.pop()
    if len(set(state_snapshots.values())) != len(state_snapshots):
        raise ControlledContinuationAnalysisError(
            "controlled states must have distinct recoverable snapshots"
        )
    if len(set(state_frontiers.values())) != len(state_frontiers):
        raise ControlledContinuationAnalysisError(
            "controlled states must have different verified frontiers"
        )

    program_definitions: list[dict[str, Any]] = []
    for program_id, program_cells in sorted(by_program.items()):
        action_variants = {cell.program_actions for cell in program_cells}
        if len(action_variants) != 1:
            raise ControlledContinuationAnalysisError(
                f"program actions drifted across states: {program_id}"
            )
        actions = action_variants.pop()
        program_definitions.append(
            {
                "program_id": program_id,
                "actions": list(actions),
                "program_sha256": canonical_sha256(
                    {"program_id": program_id, "actions": list(actions)}
                ),
            }
        )

    returns_by_cell = {
        (cell.state_id, cell.program_id): (gamma ** len(cell.executed_actions))
        * cell.outcome_evidence.benchmark_score
        for cell in cells
    }
    all_returns = list(returns_by_cell.values())
    coarse_score = _mean(all_returns)
    coarse_mse = _mean([(value - coarse_score) ** 2 for value in all_returns])
    refined_scores: list[dict[str, Any]] = []
    squared_errors: list[float] = []
    for state_id, state_cells in sorted(by_state.items()):
        returns = [returns_by_cell[(cell.state_id, cell.program_id)] for cell in state_cells]
        score = _mean(returns)
        state_errors = [(value - score) ** 2 for value in returns]
        squared_errors.extend(state_errors)
        refined_scores.append(
            {
                "state_id": state_id,
                "state_snapshot_sha256": state_snapshots[state_id],
                "frontier_ids": list(state_frontiers[state_id]),
                "sample_count": len(returns),
                "continuation_score": score,
                "empirical_mse": _mean(state_errors),
                "coarse_score_bias": coarse_score - score,
                "returns": returns,
            }
        )
    refined_mse = _mean(squared_errors)
    score_values = [float(row["continuation_score"]) for row in refined_scores]
    score_range = max(score_values) - min(score_values)
    mechanism_triggered = score_range > 0.0
    blocked_by = ["evidence_not_verified_by_statistical_builder"]
    if not mechanism_triggered:
        blocked_by.append("no_phase_conditioned_continuation_score_heterogeneity")
    return {
        "schema": CONTROLLED_CONTINUATION_ANALYSIS_SCHEMA,
        "gamma": gamma,
        "state_count": len(by_state),
        "program_count": len(by_program),
        "matrix_cell_count": len(cells),
        "matrix_complete": True,
        "program_distribution": "uniform_same_programs_per_verified_state",
        "programs": program_definitions,
        "coarse_state_key": "official_g2po_anchor",
        "refined_state_key": "official_g2po_anchor_plus_verified_frontier",
        "coarse_score": {
            "sample_count": len(all_returns),
            "continuation_score": coarse_score,
            "empirical_mse": coarse_mse,
        },
        "refined_scores": refined_scores,
        "refined_empirical_mse": refined_mse,
        "empirical_mse_reduction": coarse_mse - refined_mse,
        "continuation_score_range": score_range,
        "mechanism_triggered": mechanism_triggered,
        "admitted": False,
        "blocked_by": blocked_by,
        "rows": [cell.to_dict() for cell in cells],
        "claim_scope": CONTROLLED_CONTINUATION_STATS_CLAIM_SCOPE,
    }


def _read_matrix(source: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlledContinuationAnalysisError(
            f"cannot read controlled continuation matrix: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ControlledContinuationAnalysisError(
            "controlled continuation matrix must be a mapping"
        )
    if payload.get("schema") != CONTROLLED_CONTINUATION_MATRIX_SCHEMA:
        raise ControlledContinuationAnalysisError(
            "unsupported controlled continuation matrix schema"
        )
    if not isinstance(payload.get("rows"), list):
        raise ControlledContinuationAnalysisError(
            "controlled continuation matrix rows must be a list"
        )
    try:
        gamma = float(payload.get("gamma", -1.0))
    except (TypeError, ValueError) as error:
        raise ControlledContinuationAnalysisError(
            "controlled continuation gamma must be numeric"
        ) from error
    if not 0.0 <= gamma <= 1.0:
        raise ControlledContinuationAnalysisError("continuation gamma must be between zero and one")
    return {**payload, "gamma": gamma}


def _validate_artifact_evidence(
    payload: Mapping[str, Any],
    cells: tuple[ControlledContinuationCell, ...],
) -> dict[str, Any]:
    recipe = _validate_recipe_evidence(payload.get("recipe"), cells, payload["gamma"])
    collision, task_metadata = _validate_collision_evidence(payload.get("collision"), cells)
    _validate_physical_cell_evidence(cells, task_metadata=task_metadata)
    return {
        "recipe": recipe,
        "collision": collision,
        "physical_cell_count": len(cells),
        "outcome_evidence_revalidated": True,
        "restore_evidence_revalidated": True,
        "model_policy_used": False,
    }


def _validate_recipe_evidence(
    value: object,
    cells: tuple[ControlledContinuationCell, ...],
    gamma: float,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlledContinuationAnalysisError("matrix lacks recipe evidence")
    path = _required_path(value, "path")
    file_sha256 = _sha256(value.get("file_sha256"), "recipe file")
    if not path.is_file() or _sha256_file(path) != file_sha256:
        raise ControlledContinuationAnalysisError("recipe file evidence mismatch")
    binding = value.get("repository_binding")
    if not isinstance(binding, Mapping):
        raise ControlledContinuationAnalysisError("recipe repository binding is missing")
    root = _required_path(binding, "repository_root")
    revision = _required_text(binding, "repository_revision")
    relative = _required_text(binding, "repository_relative_path")
    if path != (root / relative).resolve():
        raise ControlledContinuationAnalysisError("recipe path is not repository-bound")
    if binding.get("matches_head") is not True:
        raise ControlledContinuationAnalysisError("recipe was not committed unchanged")
    if binding.get("working_file_sha256") != file_sha256:
        raise ControlledContinuationAnalysisError("recipe working hash mismatch")
    try:
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ControlledContinuationAnalysisError(
            f"cannot revalidate committed recipe: {error}"
        ) from error
    committed_sha256 = hashlib.sha256(committed).hexdigest()
    if committed_sha256 != file_sha256 or binding.get("head_blob_sha256") != committed_sha256:
        raise ControlledContinuationAnalysisError("committed recipe hash mismatch")

    from lh_platform.training.alfworld_controlled_aliasing import (
        load_alfworld_controlled_aliasing_recipe,
    )

    recipe = load_alfworld_controlled_aliasing_recipe(path)
    if value.get("semantic_sha256") != recipe.sha256:
        raise ControlledContinuationAnalysisError("recipe semantic hash mismatch")
    if recipe.continuation_gamma != gamma:
        raise ControlledContinuationAnalysisError("matrix gamma differs from recipe")
    expected_programs = {
        program.program_id: program.actions for program in recipe.continuation_programs
    }
    actual_programs: dict[str, tuple[str, ...]] = {}
    for cell in cells:
        existing = actual_programs.setdefault(cell.program_id, cell.program_actions)
        if existing != cell.program_actions:
            raise ControlledContinuationAnalysisError("matrix program definition drifted")
    if actual_programs != expected_programs:
        raise ControlledContinuationAnalysisError(
            "matrix programs differ from the preregistered recipe"
        )
    return {
        "path": str(path),
        "file_sha256": file_sha256,
        "semantic_sha256": recipe.sha256,
        "repository_revision": revision,
        "preregistered_programs_revalidated": True,
    }


def _validate_collision_evidence(
    value: object,
    cells: tuple[ControlledContinuationCell, ...],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ControlledContinuationAnalysisError("matrix lacks collision evidence")
    rollout_path = _required_path(value, "rollout_path")
    rollout_sha256 = _sha256(value.get("rollout_sha256"), "collision rollout")
    if not rollout_path.is_file() or _sha256_file(rollout_path) != rollout_sha256:
        raise ControlledContinuationAnalysisError("collision rollout evidence mismatch")
    manifest = analyze_controlled_aliasing_artifact(rollout_path)
    if manifest.get("admitted") is not True:
        raise ControlledContinuationAnalysisError(
            "source collision rollout failed independent admission"
        )
    task_id = _required_text(value, "task_id")
    group_uid = _required_text(value, "group_uid")
    official_anchor = _required_exact_text(value, "official_anchor")
    anchor_sha256 = _sha256(value.get("official_anchor_sha256"), "official anchor")
    if canonical_sha256(official_anchor) != anchor_sha256:
        raise ControlledContinuationAnalysisError("official anchor hash mismatch")
    groups = [
        group
        for group in manifest.get("collision_groups", ())
        if isinstance(group, Mapping)
        and group.get("task_id") == task_id
        and group.get("group_uid") == group_uid
        and group.get("official_anchor") == official_anchor
        and group.get("official_anchor_sha256") == anchor_sha256
    ]
    if len(groups) != 1:
        raise ControlledContinuationAnalysisError(
            "matrix is not bound to one independently admitted collision"
        )
    expected_states: set[tuple[str, tuple[str, ...]]] = set()
    for state in groups[0].get("verified_frontier_states", ()):
        if not isinstance(state, Mapping):
            continue
        frontier_ids = _string_sequence(state.get("active_frontier_ids"), "active_frontier_ids")
        for member in state.get("members", ()):
            if isinstance(member, Mapping):
                snapshot = _sha256(member.get("state_snapshot_sha256"), "collision state snapshot")
                expected_states.add((snapshot, frontier_ids))
    actual_states = {(cell.state_snapshot_sha256, cell.frontier_ids) for cell in cells}
    if actual_states != expected_states:
        raise ControlledContinuationAnalysisError(
            "matrix states differ from the admitted collision states"
        )
    source_records = tuple(load_rollout_artifact(rollout_path))
    matching_metadata = {
        canonical_sha256(record.extra.get("alfworld_task_metadata")): (
            record.extra.get("alfworld_task_metadata")
        )
        for record in source_records
        if record.task_id == task_id
        and record.group_uid == group_uid
        and record.observation == official_anchor
        and isinstance(record.extra.get("alfworld_task_metadata"), Mapping)
    }
    if len(matching_metadata) != 1:
        raise ControlledContinuationAnalysisError(
            "collision rollout does not expose one canonical task metadata object"
        )
    task_metadata = next(iter(matching_metadata.values()))
    summary = {
        "rollout_path": str(rollout_path),
        "rollout_sha256": rollout_sha256,
        "task_id": task_id,
        "group_uid": group_uid,
        "official_anchor_sha256": anchor_sha256,
        "collision_revalidated": True,
    }
    return summary, task_metadata


def _validate_physical_cell_evidence(
    cells: tuple[ControlledContinuationCell, ...],
    *,
    task_metadata: Mapping[str, Any],
) -> None:
    for cell in cells:
        identity = f"{cell.state_id}/{cell.program_id}"
        if cell.controller_source != CONTROLLED_CONTINUATION_CONTROLLER_SOURCE:
            raise ControlledContinuationAnalysisError(
                f"continuation controller is not preregistered: {identity}"
            )
        if cell.model_policy_used is not False:
            raise ControlledContinuationAnalysisError(
                f"continuation matrix ambiguously used a model policy: {identity}"
            )
        if len(cell.step_evidence) != len(cell.executed_actions):
            raise ControlledContinuationAnalysisError(
                f"continuation step evidence is incomplete: {identity}"
            )
        for index, (action, evidence) in enumerate(
            zip(cell.executed_actions, cell.step_evidence, strict=True)
        ):
            if evidence.get("action") != action or evidence.get("restore_verified") is not True:
                raise ControlledContinuationAnalysisError(
                    f"continuation step evidence action mismatch: {identity}/{index}"
                )
            snapshot_sha256 = evidence.get("state_snapshot_sha256")
            reasons = validate_alfworld_replay_evidence(
                snapshot_sha256=snapshot_sha256,
                snapshot_path=_optional_path(evidence.get("state_snapshot_artifact_path")),
                receipt_sha256=evidence.get("restore_receipt_sha256"),
                receipt_path=_optional_path(evidence.get("restore_receipt_path")),
                action=action,
            )
            if reasons:
                raise ControlledContinuationAnalysisError(
                    f"continuation restore evidence failed for {identity}/{index}: "
                    + ", ".join(reasons)
                )
            if index == 0 and snapshot_sha256 != cell.state_snapshot_sha256:
                raise ControlledContinuationAnalysisError(
                    f"continuation first step is not collision-bound: {identity}"
                )
        _validate_outcome_artifact(cell, task_metadata=task_metadata)


def _validate_outcome_artifact(
    cell: ControlledContinuationCell,
    *,
    task_metadata: Mapping[str, Any],
) -> None:
    identity = f"{cell.state_id}/{cell.program_id}"
    evidence = cell.outcome_evidence
    path = _optional_path(evidence.artifact_path)
    artifact_sha256 = _sha256(evidence.artifact_sha256, "outcome artifact")
    if path is None or not path.is_file() or _sha256_file(path) != artifact_sha256:
        raise ControlledContinuationAnalysisError(
            f"controlled continuation outcome artifact mismatch: {identity}"
        )
    if path.name != f"{artifact_sha256}.json":
        raise ControlledContinuationAnalysisError(
            f"controlled continuation outcome artifact path is noncanonical: {identity}"
        )
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlledContinuationAnalysisError(
            f"cannot read controlled continuation outcome artifact: {identity}"
        ) from error
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema") != ALFWORLD_CONTROLLED_CONTINUATION_OUTCOME_ARTIFACT_SCHEMA
    ):
        raise ControlledContinuationAnalysisError(
            f"unsupported controlled continuation outcome artifact: {identity}"
        )
    if artifact.get("state_id") != cell.state_id or artifact.get("program_id") != cell.program_id:
        raise ControlledContinuationAnalysisError(
            f"controlled continuation outcome artifact identity mismatch: {identity}"
        )
    retained_metadata = artifact.get("task_metadata")
    final_facts = artifact.get("final_facts")
    replay_outcome = artifact.get("replay_outcome")
    if (
        not isinstance(retained_metadata, Mapping)
        or canonical_sha256(retained_metadata) != canonical_sha256(task_metadata)
        or not isinstance(final_facts, list)
        or not isinstance(replay_outcome, Mapping)
    ):
        raise ControlledContinuationAnalysisError(
            f"controlled continuation outcome verifier input mismatch: {identity}"
        )
    if _canonical_fact_rows(final_facts) != _canonical_fact_rows(replay_outcome.get("facts")):
        raise ControlledContinuationAnalysisError(
            f"controlled continuation final facts are not step-bound: {identity}"
        )
    final_receipt_path = _optional_path(cell.step_evidence[-1].get("restore_receipt_path"))
    try:
        final_receipt = json.loads(
            final_receipt_path.read_text(encoding="utf-8") if final_receipt_path is not None else ""
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ControlledContinuationAnalysisError(
            f"cannot bind outcome to final restore receipt: {identity}"
        ) from error
    replay_outcome_sha256 = canonical_sha256(replay_outcome)
    if (
        not isinstance(final_receipt, Mapping)
        or final_receipt.get("replay_outcome_sha256") != replay_outcome_sha256
        or final_receipt.get("first_outcome_sha256") != replay_outcome_sha256
    ):
        raise ControlledContinuationAnalysisError(
            f"controlled continuation outcome is not restore-replay-bound: {identity}"
        )
    compiler = AlfWorldSubproblemCompiler()
    graph = compiler.compile_task(retained_metadata)
    vector = compiler.project_state(retained_metadata, final_facts)
    projection = goal_state_projection(graph, vector)
    residual_goal_ids = tuple(projection["residual_goal_ids"])
    verifier_complete = not residual_goal_ids
    benchmark_score = 10.0 if verifier_complete else 0.0
    official_outcome = artifact.get("official_outcome")
    if not isinstance(official_outcome, Mapping):
        raise ControlledContinuationAnalysisError(
            f"controlled continuation official outcome is missing: {identity}"
        )
    final_receipt_sha256 = cell.step_evidence[-1].get("restore_receipt_sha256")
    matches = bool(
        artifact.get("final_step_restore_receipt_sha256") == final_receipt_sha256
        and artifact.get("verifier_projection") == projection
        and official_outcome.get("won") is replay_outcome.get("won")
        and bool(replay_outcome.get("won")) is verifier_complete
        and official_outcome.get("benchmark_score") == benchmark_score
        and official_outcome.get("source_reward") == replay_outcome.get("reward")
        and official_outcome.get("done") is replay_outcome.get("done")
        and bool(replay_outcome.get("done")) is evidence.environment_done
        and evidence.official_won is verifier_complete
        and evidence.verifier_complete is verifier_complete
        and evidence.residual_goal_ids == residual_goal_ids
        and evidence.benchmark_score == benchmark_score
    )
    if not matches:
        raise ControlledContinuationAnalysisError(
            f"controlled continuation outcome failed independent verification: {identity}"
        )


def _canonical_fact_rows(value: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, (list, tuple)):
        raise ControlledContinuationAnalysisError(
            "controlled continuation facts must be a sequence"
        )
    rows: list[tuple[str, tuple[str, ...]]] = []
    for item in value:
        if isinstance(item, Mapping):
            name = item.get("name")
            args = item.get("args")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            name, args = item
        else:
            raise ControlledContinuationAnalysisError("controlled continuation fact row is invalid")
        if not isinstance(name, str) or not isinstance(args, (list, tuple)):
            raise ControlledContinuationAnalysisError("controlled continuation fact row is invalid")
        rows.append((name, tuple(str(argument) for argument in args)))
    return tuple(sorted(rows))


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ControlledContinuationAnalysisError(f"{key} must be non-empty text")
    return value.strip()


def _required_exact_text(row: Mapping[str, Any], key: str) -> str:
    """Return an identity-bearing string without changing its byte content."""

    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ControlledContinuationAnalysisError(f"{key} must be non-empty text")
    return value


def _string_sequence(
    value: object,
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or (not allow_empty and not value)
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        qualifier = "string sequence" if allow_empty else "non-empty string sequence"
        raise ControlledContinuationAnalysisError(f"{field} must be a {qualifier}")
    return tuple(str(item) for item in value)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ControlledContinuationAnalysisError(f"{field} SHA256 is invalid")
    return value


def _required_path(row: Mapping[str, Any], key: str) -> Path:
    path = _optional_path(row.get(key))
    if path is None:
        raise ControlledContinuationAnalysisError(f"{key} must be a non-empty path")
    return path


def _optional_path(value: object) -> Path | None:
    return Path(value).resolve() if isinstance(value, str) and value else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
