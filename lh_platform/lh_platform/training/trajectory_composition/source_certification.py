"""Selective restore certification for snapshot-only frozen-policy transitions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.alfworld_state import (
    ALFWORLD_STATE_CAPABILITY_SCHEMA,
    AlfWorldPhysicalCallCounters,
    AlfWorldReplayStateController,
    AlfWorldStateSnapshot,
    project_scalar_textworld_outcome,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.trajectory_composition.contracts import canonical_sha256

SNAPSHOT_TRANSITION_CERTIFICATION_SCHEMA = "lh_snapshot_transition_certification/v1"


class SnapshotTransitionCertificationError(RuntimeError):
    """Raised when a snapshot-only transition cannot be replay-certified."""


@dataclass(frozen=True)
class SnapshotTransitionCertificationV1:
    task_id: str
    task_instance_sha256: str
    source_traj_uid: str
    step_idx: int
    canonical_action: str
    source_pre_state_snapshot_sha256: str
    source_post_state_snapshot_sha256: str
    replay_post_state_snapshot_sha256: str
    restore_receipt_sha256: str
    restore_receipt_path: str
    independent_outcome_receipt_sha256: str
    physical_call_delta: Mapping[str, Any]
    physical_call_cumulative: Mapping[str, Any]
    physical_call_delta_sha256: str
    physical_call_cumulative_sha256: str
    post_state_identity_verified: bool
    outcome_verified: bool
    restore_verified: bool
    training_eligible: bool = False
    schema: str = SNAPSHOT_TRANSITION_CERTIFICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SNAPSHOT_TRANSITION_CERTIFICATION_SCHEMA:
            raise SnapshotTransitionCertificationError("unsupported certification schema")
        if not self.task_id or not self.source_traj_uid or self.step_idx < 0:
            raise SnapshotTransitionCertificationError("certification identity is incomplete")
        if (
            not self.canonical_action
            or self.canonical_action != self.canonical_action.strip()
            or "\n" in self.canonical_action
        ):
            raise SnapshotTransitionCertificationError("certification action is not canonical")
        for name, value in (
            ("task_instance_sha256", self.task_instance_sha256),
            ("source_pre_state_snapshot_sha256", self.source_pre_state_snapshot_sha256),
            ("source_post_state_snapshot_sha256", self.source_post_state_snapshot_sha256),
            ("replay_post_state_snapshot_sha256", self.replay_post_state_snapshot_sha256),
            ("restore_receipt_sha256", self.restore_receipt_sha256),
            ("independent_outcome_receipt_sha256", self.independent_outcome_receipt_sha256),
            ("physical_call_delta_sha256", self.physical_call_delta_sha256),
            ("physical_call_cumulative_sha256", self.physical_call_cumulative_sha256),
        ):
            _require_sha256(name, value)
        delta = AlfWorldPhysicalCallCounters.from_dict(self.physical_call_delta)
        cumulative = AlfWorldPhysicalCallCounters.from_dict(self.physical_call_cumulative)
        if delta.sha256 != self.physical_call_delta_sha256:
            raise SnapshotTransitionCertificationError("certification call-delta hash mismatch")
        if cumulative.sha256 != self.physical_call_cumulative_sha256:
            raise SnapshotTransitionCertificationError(
                "certification cumulative-call hash mismatch"
            )
        if delta.logical_semantic_action_calls != 0:
            raise SnapshotTransitionCertificationError(
                "selective certification probes are not source logical actions"
            )
        if (
            delta.candidate_probe_first_worker_step_calls != 1
            or delta.candidate_probe_replay_worker_step_calls != 1
        ):
            raise SnapshotTransitionCertificationError(
                "selective certification requires first and replay probe calls"
            )
        if not (
            self.post_state_identity_verified and self.outcome_verified and self.restore_verified
        ):
            raise SnapshotTransitionCertificationError(
                "failed transition cannot receive a certification receipt"
            )
        if self.source_post_state_snapshot_sha256 != self.replay_post_state_snapshot_sha256:
            raise SnapshotTransitionCertificationError("source/replay post-state identity differs")
        if self.training_eligible:
            raise SnapshotTransitionCertificationError(
                "transition certification cannot admit training data"
            )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "task_instance_sha256": self.task_instance_sha256,
            "source_traj_uid": self.source_traj_uid,
            "step_idx": self.step_idx,
            "canonical_action": self.canonical_action,
            "source_pre_state_snapshot_sha256": self.source_pre_state_snapshot_sha256,
            "source_post_state_snapshot_sha256": self.source_post_state_snapshot_sha256,
            "replay_post_state_snapshot_sha256": self.replay_post_state_snapshot_sha256,
            "restore_receipt_sha256": self.restore_receipt_sha256,
            "restore_receipt_path": self.restore_receipt_path,
            "independent_outcome_receipt_sha256": self.independent_outcome_receipt_sha256,
            "physical_call_delta": dict(self.physical_call_delta),
            "physical_call_cumulative": dict(self.physical_call_cumulative),
            "physical_call_delta_sha256": self.physical_call_delta_sha256,
            "physical_call_cumulative_sha256": self.physical_call_cumulative_sha256,
            "post_state_identity_verified": self.post_state_identity_verified,
            "outcome_verified": self.outcome_verified,
            "restore_verified": self.restore_verified,
            "training_eligible": self.training_eligible,
        }


def certify_snapshot_only_actor_transition(
    record: TrainingTrajectoryRecord,
    *,
    task_instance_sha256: str,
    artifact_root: str | Path,
    controller: AlfWorldReplayStateController,
) -> SnapshotTransitionCertificationV1:
    """Replay one retained source transition twice and bind its exact post-state."""

    _require_sha256("task_instance_sha256", task_instance_sha256)
    extra = record.extra
    if extra.get("transition_evidence_mode") != "snapshot_only":
        raise SnapshotTransitionCertificationError(
            "selective certification requires a snapshot_only source row"
        )
    if extra.get("restore_verified") is not False:
        raise SnapshotTransitionCertificationError(
            "snapshot-only source row must not claim restore verification"
        )
    if extra.get("restore_probe_receipt_sha256") or extra.get("restore_probe_receipt_path"):
        raise SnapshotTransitionCertificationError(
            "snapshot-only source row already claims a restore receipt"
        )
    action = _canonical_action(record)
    pre_hash = record.state_snapshot_sha256
    post_hash = str(extra.get("post_state_snapshot_sha256", ""))
    _require_sha256("source_pre_state_snapshot_sha256", pre_hash)
    _require_sha256("source_post_state_snapshot_sha256", post_hash)
    root = Path(artifact_root).resolve()
    pre_payload = _read_bound_payload(
        str(extra.get("state_snapshot_artifact_path", "")),
        expected_sha256=pre_hash,
        root=root,
    )
    _read_bound_payload(
        str(extra.get("post_state_snapshot_artifact_path", "")),
        expected_sha256=post_hash,
        root=root,
    )
    snapshot = AlfWorldStateSnapshot(
        schema=ALFWORLD_STATE_CAPABILITY_SCHEMA,
        state_snapshot_sha256=pre_hash,
        payload=pre_payload,
        wrapper_fields=controller.wrapper_fields,
    )
    verified = controller.probe_snapshot(
        snapshot,
        action,
        project_outcome=project_scalar_textworld_outcome,
    )
    if verified.post_snapshot_artifact is None:
        raise SnapshotTransitionCertificationError("probe did not materialize a post-state")
    replay_post_hash = verified.post_snapshot_artifact.state_snapshot_sha256
    post_identity_verified = replay_post_hash == post_hash
    source_outcome = extra.get("alfworld_environment_outcome")
    if not isinstance(source_outcome, Mapping):
        raise SnapshotTransitionCertificationError(
            "snapshot-only source lacks its raw environment outcome"
        )
    expected_outcome = {
        "observation": str(source_outcome.get("observation", "")),
        "reward": float(source_outcome.get("reward")),
        "done": bool(source_outcome.get("done")),
        "won": bool(source_outcome.get("won")),
        "facts": _canonical_facts(source_outcome.get("facts")),
        "admissible_actions": tuple(
            str(value) for value in source_outcome.get("admissible_actions") or ()
        ),
    }
    if expected_outcome["facts"] != _canonical_facts(extra.get("alfworld_next_facts")):
        raise SnapshotTransitionCertificationError(
            "source environment outcome and next-facts record differ"
        )
    if expected_outcome["admissible_actions"] != tuple(
        str(value) for value in extra.get("alfworld_next_admissible_actions") or ()
    ):
        raise SnapshotTransitionCertificationError(
            "source environment outcome and next-admissible record differ"
        )
    actual_outcome = {
        "observation": str(verified.outcome["observation"]),
        "reward": float(verified.outcome["reward"]),
        "done": bool(verified.outcome["done"]),
        "won": bool(verified.outcome["won"]),
        "facts": _canonical_facts(verified.outcome["facts"]),
        "admissible_actions": tuple(str(value) for value in verified.outcome["admissible_actions"]),
    }
    outcome_verified = actual_outcome == expected_outcome
    if not post_identity_verified:
        raise SnapshotTransitionCertificationError(
            "selective replay post-state differs from source collection"
        )
    if not outcome_verified:
        raise SnapshotTransitionCertificationError(
            "selective replay outcome differs from source collection"
        )
    if verified.physical_call_delta is None or verified.physical_call_cumulative is None:
        raise SnapshotTransitionCertificationError("probe lacks physical-call accounting")
    return SnapshotTransitionCertificationV1(
        task_id=record.task_id,
        task_instance_sha256=task_instance_sha256,
        source_traj_uid=record.traj_uid,
        step_idx=record.step_idx,
        canonical_action=action,
        source_pre_state_snapshot_sha256=pre_hash,
        source_post_state_snapshot_sha256=post_hash,
        replay_post_state_snapshot_sha256=replay_post_hash,
        restore_receipt_sha256=verified.restore_receipt_sha256,
        restore_receipt_path=verified.restore_receipt_path,
        independent_outcome_receipt_sha256=canonical_sha256(
            {
                "expected": expected_outcome,
                "actual": actual_outcome,
            }
        ),
        physical_call_delta=verified.physical_call_delta.to_dict(),
        physical_call_cumulative=verified.physical_call_cumulative.to_dict(),
        physical_call_delta_sha256=verified.physical_call_delta.sha256,
        physical_call_cumulative_sha256=verified.physical_call_cumulative.sha256,
        post_state_identity_verified=True,
        outcome_verified=True,
        restore_verified=verified.restore_receipt.restore_verified,
    )


def _canonical_action(record: TrainingTrajectoryRecord) -> str:
    value = record.parsed_action.get("command")
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\n" in value
        or record.extra.get("environment_action") != value
    ):
        raise SnapshotTransitionCertificationError("source semantic action is invalid")
    return value


def _canonical_facts(value: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, (list, tuple)):
        raise SnapshotTransitionCertificationError("source/replay facts are unavailable")
    rows: list[tuple[str, tuple[str, ...]]] = []
    for fact in value:
        if isinstance(fact, Mapping):
            name = str(fact.get("name", ""))
            args = tuple(str(item) for item in fact.get("args") or ())
        elif isinstance(fact, (list, tuple)) and len(fact) == 2:
            name = str(fact[0])
            raw_args = fact[1]
            if not isinstance(raw_args, (list, tuple)):
                raise SnapshotTransitionCertificationError("replay fact args are invalid")
            args = tuple(str(item) for item in raw_args)
        else:
            raise SnapshotTransitionCertificationError("source/replay fact shape is invalid")
        rows.append((name, args))
    return tuple(sorted(rows))


def _read_bound_payload(path_value: str, *, expected_sha256: str, root: Path) -> bytes:
    path = Path(path_value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SnapshotTransitionCertificationError("snapshot artifact escapes or is unavailable")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SnapshotTransitionCertificationError("snapshot artifact SHA-256 mismatch")
    return payload


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SnapshotTransitionCertificationError(f"{name} must be a lowercase SHA-256")
