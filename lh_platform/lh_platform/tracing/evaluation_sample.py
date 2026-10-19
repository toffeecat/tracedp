"""Experiment contract separating actor execution, blind tracing, and recovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracing.types import TraceCaseV1


class FailureOrigin(StrEnum):
    NATURAL = "natural_actor_failure"
    CONTROLLED = "controlled_hidden_root"


class RootVerificationContinuation(StrEnum):
    FROZEN_SOURCE_SUFFIX = "frozen_source_suffix"
    FROZEN_SOURCE_PROGRAM = "frozen_source_program"
    SAME_ACTOR_REPLANNING = "same_actor_closed_loop_replanning"


class RecoveryContinuation(StrEnum):
    NOT_RUN = "not_run"
    SAME_ACTOR_REPLANNING = "same_actor_closed_loop_replanning"


@dataclass(frozen=True)
class HiddenRootGold:
    """Scoring-only root evidence that must never enter a tracer prompt."""

    event_ids: tuple[str, ...]
    causal_distance: int
    counterfactually_verified: bool

    def __post_init__(self) -> None:
        if not self.event_ids:
            raise ValueError("hidden root gold requires at least one event")
        if self.causal_distance < 0:
            raise ValueError("causal distance must be non-negative")


@dataclass(frozen=True)
class TraceEvaluationSample:
    """One frozen failed episode plus scoring-only evidence and execution policy."""

    blind_trace: TraceCaseV1
    failure_origin: FailureOrigin
    source_actor_closed_loop: bool
    root_verification_continuation: RootVerificationContinuation
    hidden_gold: HiddenRootGold
    recovery_continuation: RecoveryContinuation = RecoveryContinuation.NOT_RUN
    recovery_evidence_available: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.hidden_gold.event_ids) - self.blind_trace.event_ids
        if unknown:
            raise ValueError("hidden root must reference events in the blind trace")

    def tracer_input(self) -> dict[str, object]:
        """Return exactly the task, failed trace, and public failure signal."""

        return self.blind_trace.to_dict()

    @property
    def supports_blind_localization(self) -> bool:
        return self.hidden_gold.counterfactually_verified

    @property
    def supports_end_to_end_recovery(self) -> bool:
        return (
            self.hidden_gold.counterfactually_verified
            and self.source_actor_closed_loop
            and self.recovery_continuation is RecoveryContinuation.SAME_ACTOR_REPLANNING
            and self.recovery_evidence_available
        )

    @property
    def distance_stratum(self) -> str:
        distance = self.hidden_gold.causal_distance
        if distance == 0:
            return "distance_0"
        if distance <= 3:
            return "distance_1_3"
        if distance <= 9:
            return "distance_4_9"
        return "distance_ge_10"

    @property
    def claim_boundary(self) -> str:
        if self.supports_end_to_end_recovery:
            return "blind_localize_verify_repair_resume"
        if self.failure_origin is FailureOrigin.CONTROLLED:
            return "controlled_blind_localization_only"
        return "natural_failure_blind_localization_only"


def validate_actor_closed_loop_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_task_ids: frozenset[str],
) -> None:
    """Verify the persisted model-response-to-environment transition chain."""

    rows_by_task: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in expected_task_ids:
            rows_by_task.setdefault(task_id, []).append(row)
    if set(rows_by_task) != expected_task_ids:
        raise ValueError("source actor evidence does not cover the evaluation tasks")

    for task_id, task_rows in rows_by_task.items():
        ordered = sorted(task_rows, key=lambda row: int(row["step_index"]))
        if [int(row["step_index"]) for row in ordered] != list(range(len(ordered))):
            raise ValueError(f"source actor steps are not continuous for {task_id}")
        for index, row in enumerate(ordered):
            if not str(row.get("model_response", "")).strip():
                raise ValueError(f"source model response is missing for {task_id}")
            if row.get("parser_valid") is not True:
                raise ValueError(f"source parser rejected an action for {task_id}")
            if row.get("action_in_admissible_menu") is not True:
                raise ValueError(f"source action is outside the public menu for {task_id}")
            if not str(row.get("environment_action", "")).strip():
                raise ValueError(f"source environment action is missing for {task_id}")
            if row.get("environment_step_executed") is not True:
                raise ValueError(f"source environment step did not execute for {task_id}")
            if "next_observation" not in row:
                raise ValueError(f"source next observation is missing for {task_id}")
            if index + 1 < len(ordered):
                next_row = ordered[index + 1]
                if row["next_observation"] != next_row.get("observation"):
                    raise ValueError(f"source observation chain is broken for {task_id}")


__all__ = [
    "FailureOrigin",
    "HiddenRootGold",
    "RecoveryContinuation",
    "RootVerificationContinuation",
    "TraceEvaluationSample",
    "validate_actor_closed_loop_rows",
]
