"""Contracts for trainable method evidence."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

TRAINING_DATASET_SCHEMA = "lh_training_trajectory/v2"
LEGACY_TRAINING_DATASET_SCHEMA = "lh_training_trajectory/v1"


@dataclass(frozen=True)
class TrainingTrajectoryRecord:
    """One canonical transition record for training-data preparation.

    Reward semantics (v3):
    - ``reward``: legacy field, terminal outcome reward (e.g. 10.0 success / 0.0 fail).
    - ``dense_step_reward``: optional per-step shaping signal; None means no
      dense reward was assigned for this transition.
    - ``terminal_outcome_reward``: explicit terminal-only reward; non-None only
      on the terminal step.  When set, takes precedence over ``reward`` for
      credit assignment.  Legacy v2 data leaves this as None.
    - ``success``: independent boolean, never inferred from reward magnitude.
    """

    dataset_id: str
    benchmark: str
    task_id: str
    split: str
    seed: int | None
    traj_uid: str
    group_uid: str
    step_idx: int
    instruction: str
    observation: str
    action_raw: str
    action_type: str
    parsed_action: dict[str, Any]
    next_observation: str
    reward: float
    terminal: bool
    success: bool
    outcome_evidence: dict[str, Any]
    action_language: str
    method_source: str
    model_id: str
    # v2 is an environment-transition contract.  The fields below deliberately
    # describe observable runtime state and never retain hidden planner thought.
    trajectory_length: int | None = None
    dense_step_reward: float | None = None
    terminal_outcome_reward: float | None = None
    terminal_reason: str = ""
    decision_role: str = "executor"
    state_snapshot_sha256: str = ""
    goal_hash: str = ""
    remaining_budget: int | None = None
    action_affordance_hash: str = ""
    rollout_config_hash: str = ""
    active: bool = True
    action_token_count: int = 1
    prompt_hash: str = ""
    # Exact rollout-token evidence.  These fields are additive for stored v2
    # compatibility, but are mandatory before an on-policy backend update.
    prompt_ids: tuple[int, ...] = ()
    response_ids: tuple[int, ...] = ()
    response_mask: tuple[int, ...] = ()
    response_old_logprobs: tuple[float, ...] = ()
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRAINING_DATASET_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingTrajectoryRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        values = {k: v for k, v in data.items() if k in known}
        for name in (
            "prompt_ids",
            "response_ids",
            "response_mask",
            "response_old_logprobs",
        ):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    @property
    def effective_terminal_reward(self) -> float:
        """Return the authoritative terminal reward for credit assignment."""

        if self.terminal_outcome_reward is not None:
            return self.terminal_outcome_reward
        return self.reward


def validate_reward_semantics(records: Iterable[TrainingTrajectoryRecord]) -> list[str]:
    """Check reward field consistency within a trajectory batch.

    Returns a list of violation descriptions (empty = valid).
    """

    violations: list[str] = []
    by_trajectory: dict[str, list[TrainingTrajectoryRecord]] = {}
    for record in records:
        by_trajectory.setdefault(record.traj_uid, []).append(record)
    for traj_uid, trajectory in by_trajectory.items():
        ordered = sorted(trajectory, key=lambda r: r.step_idx)
        terminal_rows = [r for r in ordered if r.terminal]
        if len(terminal_rows) != 1:
            violations.append(f"{traj_uid}: expected exactly 1 terminal row, got {len(terminal_rows)}")
            continue
        terminal_row = terminal_rows[0]
        non_terminal = [r for r in ordered if not r.terminal]
        for r in non_terminal:
            if r.terminal_outcome_reward is not None:
                violations.append(
                    f"{traj_uid}:step_{r.step_idx}: non-terminal row has terminal_outcome_reward"
                )
        if terminal_row.terminal_outcome_reward is not None:
            if terminal_row.success and terminal_row.terminal_outcome_reward <= 0:
                violations.append(
                    f"{traj_uid}: success=True but terminal_outcome_reward<=0"
                )
            if not terminal_row.success and terminal_row.terminal_outcome_reward > 0:
                violations.append(
                    f"{traj_uid}: success=False but terminal_outcome_reward>0"
                )
    return violations


@dataclass(frozen=True)
class TrainingDatasetManifest:
    dataset_id: str
    schema_version: str
    path: str
    sha256: str
    record_count: int
    benchmarks: tuple[str, ...]
    methods: tuple[str, ...]
    splits: tuple[str, ...]
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    method: str
    record_count: int
    missing_features: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedTrainingData:
    method: str
    dataset_manifest: TrainingDatasetManifest
    out_dir: str
    files: tuple[str, ...]
    validation: ValidationReport
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunEvidence:
    method: str
    execution_path: str
    dry_run: bool
    command: tuple[str, ...]
    config_paths: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    dataset_sha256: str
    upstream_repo: str
    upstream_commit: str | None = None
    status: str = "prepared"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MethodArtifact:
    artifact_id: str
    method: str
    artifact_type: str
    path: str
    training_dataset_sha256: str
    training_evidence_path: str
    execution_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
