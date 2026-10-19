"""Verifier-gated, environment-state identity for safe cross-trajectory merges.

Observation equality is useful for auditing a source method but is not enough
to justify sharing value: hidden wrapper state, a different goal, remaining
budget, or action affordances can change the continuation problem.  This
module only emits an identity after a restore probe has verified the snapshot.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from lh_platform.training.contracts import TrainingTrajectoryRecord


class StateIdentityError(ValueError):
    """Raised when a record cannot safely participate in a state merge."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VerifiedStateIdentity:
    snapshot_sha256: str
    goal_hash: str
    remaining_budget: int
    action_affordance_hash: str
    state_snapshot_artifact_sha256: str
    restore_probe_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_hash("snapshot_sha256", self.snapshot_sha256)
        _require_hash("goal_hash", self.goal_hash)
        _require_hash("action_affordance_hash", self.action_affordance_hash)
        _require_hash("restore_probe_receipt_sha256", self.restore_probe_receipt_sha256)
        if self.state_snapshot_artifact_sha256 != self.snapshot_sha256:
            raise StateIdentityError(
                "verified state identity must point to the same content-addressed snapshot artifact"
            )
        if self.remaining_budget < 0:
            raise StateIdentityError(
                "verified state identity requires non-negative remaining_budget"
            )

    @property
    def key(self) -> str:
        payload = {
            "action_affordance_hash": self.action_affordance_hash,
            "goal_hash": self.goal_hash,
            "remaining_budget": self.remaining_budget,
            "state_snapshot_sha256": self.snapshot_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RestoreProbeCandidate:
    """A possible shared state that is *not yet allowed* to merge.

    Snapshot capture is cheap for the TextWorld reset/replay backend, whereas
    a restore probe is expensive.  We therefore identify duplicate candidate
    states first and run restore/replay only for states that could actually
    contribute a cross-trajectory value merge.  ``key`` has the same state
    variables as a verified identity, but no credit code may consume this
    class until a receipt upgrades each participating row.
    """

    key: str
    snapshot_sha256: str
    goal_hash: str
    remaining_budget: int
    action_affordance_hash: str
    record_refs: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ObservationMergeAudit:
    transitions: int
    observation_groups: int
    false_merge_groups: int
    false_merge_pairs: int
    false_merge_rate: float
    verified_transitions: int
    verified_state_coverage: float
    shared_verified_state_groups: int
    shared_verified_state_ratio: float


def verified_state_identity(record: TrainingTrajectoryRecord) -> VerifiedStateIdentity:
    """Build an identity only when all state variables and restore proof exist."""

    if not bool(record.extra.get("restore_verified", False)):
        raise StateIdentityError("verified_state_dp requires a successful restore probe")
    if record.extra.get("remaining_budget_verified") is not True:
        raise StateIdentityError("verified_state_dp requires environment-verified remaining_budget")
    if record.extra.get("action_affordance_verified") is not True:
        raise StateIdentityError(
            "verified_state_dp requires environment-verified action affordances"
        )
    _require_hash("state_snapshot_sha256", record.state_snapshot_sha256)
    if record.extra.get("state_snapshot_artifact_sha256") != record.state_snapshot_sha256:
        raise StateIdentityError(
            "verified_state_dp requires a content-addressed state snapshot artifact"
        )
    _require_hash(
        "restore_probe_receipt_sha256", record.extra.get("restore_probe_receipt_sha256", "")
    )
    _require_hash("goal_hash", record.goal_hash)
    _require_hash("action_affordance_hash", record.action_affordance_hash)
    if record.remaining_budget is None or record.remaining_budget < 0:
        raise StateIdentityError("verified_state_dp requires non-negative remaining_budget")
    return VerifiedStateIdentity(
        snapshot_sha256=record.state_snapshot_sha256,
        goal_hash=record.goal_hash,
        remaining_budget=record.remaining_budget,
        action_affordance_hash=record.action_affordance_hash,
        state_snapshot_artifact_sha256=str(record.extra["state_snapshot_artifact_sha256"]),
        restore_probe_receipt_sha256=str(record.extra["restore_probe_receipt_sha256"]),
    )


def select_restore_probe_candidates(
    records: Iterable[TrainingTrajectoryRecord], *, min_visits: int = 2
) -> tuple[RestoreProbeCandidate, ...]:
    """Find potentially shared state snapshots before spending probe budget.

    Candidate identity deliberately excludes receipt status; it must never be
    passed to graph credit.  Each selected candidate has at least two visible
    visits with the same snapshot, goal, remaining budget, and affordance set,
    so a successful restore probe can improve an actual merge rather than
    being paid for every single rollout step.
    """

    if min_visits < 2:
        raise ValueError("restore probe candidates require min_visits >= 2")
    grouped: dict[tuple[str, str, int, str], list[TrainingTrajectoryRecord]] = defaultdict(list)
    for record in records:
        try:
            key = _candidate_components(record)
        except StateIdentityError:
            continue
        grouped[key].append(record)

    candidates: list[RestoreProbeCandidate] = []
    for (snapshot_sha256, goal_hash, remaining_budget, affordance_hash), members in grouped.items():
        if len(members) < min_visits:
            continue
        payload = {
            "action_affordance_hash": affordance_hash,
            "goal_hash": goal_hash,
            "remaining_budget": remaining_budget,
            "state_snapshot_sha256": snapshot_sha256,
        }
        state_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        candidates.append(
            RestoreProbeCandidate(
                key=state_key,
                snapshot_sha256=snapshot_sha256,
                goal_hash=goal_hash,
                remaining_budget=remaining_budget,
                action_affordance_hash=affordance_hash,
                record_refs=tuple(sorted((member.traj_uid, member.step_idx) for member in members)),
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.key))


def audit_observation_merges(
    records: Iterable[TrainingTrajectoryRecord],
) -> ObservationMergeAudit:
    """Measure how often source-style observation grouping aliases real states."""

    rows = tuple(records)
    by_observation: dict[tuple[str, str], list[TrainingTrajectoryRecord]] = defaultdict(list)
    identities: list[str] = []
    for row in rows:
        by_observation[(row.group_uid, row.observation)].append(row)
        try:
            identities.append(verified_state_identity(row).key)
        except StateIdentityError:
            continue

    false_groups = 0
    false_pairs = 0
    for group_rows in by_observation.values():
        keys: list[str] = []
        for row in group_rows:
            try:
                keys.append(verified_state_identity(row).key)
            except StateIdentityError:
                continue
        if len(set(keys)) > 1:
            false_groups += 1
            counts = Counter(keys)
            total_pairs = len(keys) * (len(keys) - 1) // 2
            same_pairs = sum(value * (value - 1) // 2 for value in counts.values())
            false_pairs += total_pairs - same_pairs

    identity_counts = Counter(identities)
    shared = sum(1 for count in identity_counts.values() if count > 1)
    observation_groups = len(by_observation)
    return ObservationMergeAudit(
        transitions=len(rows),
        observation_groups=observation_groups,
        false_merge_groups=false_groups,
        false_merge_pairs=false_pairs,
        false_merge_rate=(false_groups / observation_groups) if observation_groups else 0.0,
        verified_transitions=len(identities),
        verified_state_coverage=(len(identities) / len(rows)) if rows else 0.0,
        shared_verified_state_groups=shared,
        shared_verified_state_ratio=(shared / len(identity_counts)) if identity_counts else 0.0,
    )


def _require_hash(field: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise StateIdentityError(f"verified_state_dp requires a lowercase SHA-256 {field}")


def _candidate_components(record: TrainingTrajectoryRecord) -> tuple[str, str, int, str]:
    """Validate non-receipt identity fields for selective probe scheduling."""

    if record.extra.get("state_snapshot_artifact_sha256") != record.state_snapshot_sha256:
        raise StateIdentityError("restore candidate requires a content-addressed snapshot artifact")
    if record.extra.get("remaining_budget_verified") is not True:
        raise StateIdentityError("restore candidate requires verified remaining_budget")
    if record.extra.get("action_affordance_verified") is not True:
        raise StateIdentityError("restore candidate requires verified action affordances")
    _require_hash("state_snapshot_sha256", record.state_snapshot_sha256)
    _require_hash("goal_hash", record.goal_hash)
    _require_hash("action_affordance_hash", record.action_affordance_hash)
    if record.remaining_budget is None or record.remaining_budget < 0:
        raise StateIdentityError("restore candidate requires non-negative remaining_budget")
    return (
        record.state_snapshot_sha256,
        record.goal_hash,
        record.remaining_budget,
        record.action_affordance_hash,
    )
