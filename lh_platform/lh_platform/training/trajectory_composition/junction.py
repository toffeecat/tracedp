"""Independent E1 junction receipts for replayed fragment transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lh_platform.alfworld_state import AlfWorldPhysicalCallCounters, AlfWorldStateError
from lh_platform.training.trajectory_composition.contracts import (
    CompositionContractError,
    FragmentRecordV1,
    canonical_sha256,
)

E1_JUNCTION_REPLAY_RECEIPT_SCHEMA = "lh_e1_junction_replay_receipt/v1"


@dataclass(frozen=True)
class E1JunctionReplayReceiptV1:
    receipt_uid: str
    task_instance_sha256: str
    parent_contract_sha256: str
    left_fragment_uid: str
    left_fragment_sha256: str
    right_fragment_uid: str
    right_fragment_sha256: str
    left_source_traj_uid: str
    right_source_traj_uid: str
    left_replay_post_state_snapshot_sha256: str
    actual_junction_pre_state_snapshot_sha256: str
    right_source_pre_state_snapshot_sha256: str
    actual_right_post_state_snapshot_sha256: str
    actual_junction_pre_goal_ids: tuple[str, ...]
    right_required_goal_ids: tuple[str, ...]
    right_expected_added_goal_ids: tuple[str, ...]
    actual_right_post_goal_ids: tuple[str, ...]
    observed_added_goal_ids: tuple[str, ...]
    observed_deleted_goal_ids: tuple[str, ...]
    invariant_goal_ids: tuple[str, ...]
    independent_effect_verifier_receipt_sha256: str
    right_transition_restore_receipt_sha256s: tuple[str, ...]
    right_action_trace_sha256: str
    physical_call_delta: dict[str, Any]
    physical_call_delta_sha256: str
    exact_state_junction: bool
    precondition_verified: bool
    expected_effect_transferred: bool
    invariants_preserved: bool
    effect_transfer_verified: bool
    requires_full_parent_replay: bool
    candidate_only: bool
    source_logprobs_combined: bool
    training_eligible: bool = False
    schema: str = E1_JUNCTION_REPLAY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != E1_JUNCTION_REPLAY_RECEIPT_SCHEMA:
            raise CompositionContractError("unsupported E1 junction receipt schema")
        if not self.receipt_uid or not self.left_fragment_uid or not self.right_fragment_uid:
            raise CompositionContractError("E1 junction identity is incomplete")
        for name, value in (
            ("task_instance_sha256", self.task_instance_sha256),
            ("parent_contract_sha256", self.parent_contract_sha256),
            ("left_fragment_sha256", self.left_fragment_sha256),
            ("right_fragment_sha256", self.right_fragment_sha256),
            (
                "left_replay_post_state_snapshot_sha256",
                self.left_replay_post_state_snapshot_sha256,
            ),
            (
                "actual_junction_pre_state_snapshot_sha256",
                self.actual_junction_pre_state_snapshot_sha256,
            ),
            (
                "right_source_pre_state_snapshot_sha256",
                self.right_source_pre_state_snapshot_sha256,
            ),
            (
                "actual_right_post_state_snapshot_sha256",
                self.actual_right_post_state_snapshot_sha256,
            ),
            (
                "independent_effect_verifier_receipt_sha256",
                self.independent_effect_verifier_receipt_sha256,
            ),
            ("right_action_trace_sha256", self.right_action_trace_sha256),
            ("physical_call_delta_sha256", self.physical_call_delta_sha256),
        ):
            _require_sha256(name, value)
        for digest in self.right_transition_restore_receipt_sha256s:
            _require_sha256("right_transition_restore_receipt_sha256", digest)
        if self.left_source_traj_uid == self.right_source_traj_uid:
            raise CompositionContractError("E1 junction requires independently sourced fragments")
        if self.left_replay_post_state_snapshot_sha256 != (
            self.actual_junction_pre_state_snapshot_sha256
        ):
            raise CompositionContractError("left replay post-state is not the actual junction")
        if self.right_source_pre_state_snapshot_sha256 == (
            self.actual_junction_pre_state_snapshot_sha256
        ):
            raise CompositionContractError("exact-state junction must be classified as E0, not E1")
        if self.exact_state_junction:
            raise CompositionContractError("E1 receipt cannot claim an exact-state junction")
        if not (
            self.precondition_verified
            and self.expected_effect_transferred
            and self.invariants_preserved
            and self.effect_transfer_verified
        ):
            raise CompositionContractError("failed E1 transfer cannot receive a verified receipt")
        if not self.requires_full_parent_replay or not self.candidate_only:
            raise CompositionContractError("E1 junction remains candidate-only until full replay")
        if self.source_logprobs_combined or self.training_eligible:
            raise CompositionContractError("E1 junction cannot become on-policy training evidence")
        try:
            calls = AlfWorldPhysicalCallCounters.from_dict(self.physical_call_delta)
        except AlfWorldStateError as error:
            raise CompositionContractError(
                f"E1 physical-call receipt is invalid: {error}"
            ) from error
        if calls.sha256 != self.physical_call_delta_sha256:
            raise CompositionContractError("E1 physical-call receipt hash mismatch")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "receipt_uid": self.receipt_uid,
            "task_instance_sha256": self.task_instance_sha256,
            "parent_contract_sha256": self.parent_contract_sha256,
            "left_fragment_uid": self.left_fragment_uid,
            "left_fragment_sha256": self.left_fragment_sha256,
            "right_fragment_uid": self.right_fragment_uid,
            "right_fragment_sha256": self.right_fragment_sha256,
            "left_source_traj_uid": self.left_source_traj_uid,
            "right_source_traj_uid": self.right_source_traj_uid,
            "left_replay_post_state_snapshot_sha256": (self.left_replay_post_state_snapshot_sha256),
            "actual_junction_pre_state_snapshot_sha256": (
                self.actual_junction_pre_state_snapshot_sha256
            ),
            "right_source_pre_state_snapshot_sha256": (self.right_source_pre_state_snapshot_sha256),
            "actual_right_post_state_snapshot_sha256": (
                self.actual_right_post_state_snapshot_sha256
            ),
            "actual_junction_pre_goal_ids": list(self.actual_junction_pre_goal_ids),
            "right_required_goal_ids": list(self.right_required_goal_ids),
            "right_expected_added_goal_ids": list(self.right_expected_added_goal_ids),
            "actual_right_post_goal_ids": list(self.actual_right_post_goal_ids),
            "observed_added_goal_ids": list(self.observed_added_goal_ids),
            "observed_deleted_goal_ids": list(self.observed_deleted_goal_ids),
            "invariant_goal_ids": list(self.invariant_goal_ids),
            "independent_effect_verifier_receipt_sha256": (
                self.independent_effect_verifier_receipt_sha256
            ),
            "right_transition_restore_receipt_sha256s": list(
                self.right_transition_restore_receipt_sha256s
            ),
            "right_action_trace_sha256": self.right_action_trace_sha256,
            "physical_call_delta": dict(self.physical_call_delta),
            "physical_call_delta_sha256": self.physical_call_delta_sha256,
            "exact_state_junction": self.exact_state_junction,
            "precondition_verified": self.precondition_verified,
            "expected_effect_transferred": self.expected_effect_transferred,
            "invariants_preserved": self.invariants_preserved,
            "effect_transfer_verified": self.effect_transfer_verified,
            "requires_full_parent_replay": self.requires_full_parent_replay,
            "candidate_only": self.candidate_only,
            "source_logprobs_combined": self.source_logprobs_combined,
            "training_eligible": self.training_eligible,
        }


def certify_e1_junction_replay(
    *,
    left: FragmentRecordV1,
    right: FragmentRecordV1,
    left_replay_post_state_snapshot_sha256: str,
    actual_junction_pre_state_snapshot_sha256: str,
    actual_right_post_state_snapshot_sha256: str,
    actual_junction_pre_goal_ids: tuple[str, ...],
    actual_right_post_goal_ids: tuple[str, ...],
    independent_effect_verifier_receipt_sha256: str,
    right_transition_restore_receipt_sha256s: tuple[str, ...],
    physical_call_delta: AlfWorldPhysicalCallCounters,
) -> E1JunctionReplayReceiptV1:
    """Certify one transferred right fragment at an actual non-exact junction."""

    if left.task_instance_sha256 != right.task_instance_sha256:
        raise CompositionContractError("E1 fragments belong to different task instances")
    if left.parent_contract_sha256 != right.parent_contract_sha256:
        raise CompositionContractError("E1 fragments use different parent contracts")
    if left.source_traj_uid == right.source_traj_uid:
        raise CompositionContractError("E1 fragments require independent source trajectories")
    if left_replay_post_state_snapshot_sha256 != actual_junction_pre_state_snapshot_sha256:
        raise CompositionContractError("left replay did not terminate at the declared junction")
    if right.pre_state_snapshot_sha256 == actual_junction_pre_state_snapshot_sha256:
        raise CompositionContractError("junction is exact-state E0 rather than transferred E1")
    if len(right_transition_restore_receipt_sha256s) != right.transition_count:
        raise CompositionContractError("every transferred right action requires restore evidence")
    for digest in (
        left_replay_post_state_snapshot_sha256,
        actual_junction_pre_state_snapshot_sha256,
        actual_right_post_state_snapshot_sha256,
        independent_effect_verifier_receipt_sha256,
        *right_transition_restore_receipt_sha256s,
    ):
        _require_sha256("E1 replay digest", digest)
    pre = frozenset(_canonical_goals(actual_junction_pre_goal_ids))
    post = frozenset(_canonical_goals(actual_right_post_goal_ids))
    required = frozenset(right.required_goal_ids)
    expected_added = frozenset(right.added_goal_ids)
    invariants = frozenset(right.invariant_goal_ids)
    observed_added = post - pre
    observed_deleted = pre - post
    precondition_verified = required <= pre
    expected_effect_transferred = bool(expected_added) and expected_added <= observed_added
    invariants_preserved = not bool(observed_deleted & invariants)
    if not precondition_verified:
        raise CompositionContractError("E1 right-fragment precondition failed at junction")
    if not expected_effect_transferred:
        raise CompositionContractError("E1 expected effect did not transfer")
    if not invariants_preserved:
        raise CompositionContractError("E1 transfer regressed an invariant")
    if physical_call_delta.logical_semantic_action_calls != right.transition_count:
        raise CompositionContractError("E1 logical action count differs from right fragment")
    if (
        physical_call_delta.rollout_first_worker_step_calls != right.transition_count
        or physical_call_delta.rollout_verification_worker_step_calls != right.transition_count
    ):
        raise CompositionContractError("E1 transferred actions lack physical replay accounting")
    body = {
        "task_instance_sha256": left.task_instance_sha256,
        "left_fragment_uid": left.fragment_uid,
        "right_fragment_uid": right.fragment_uid,
        "actual_junction_pre_state_snapshot_sha256": (actual_junction_pre_state_snapshot_sha256),
        "actual_right_post_state_snapshot_sha256": actual_right_post_state_snapshot_sha256,
        "independent_effect_verifier_receipt_sha256": (independent_effect_verifier_receipt_sha256),
        "physical_call_delta_sha256": physical_call_delta.sha256,
    }
    return E1JunctionReplayReceiptV1(
        receipt_uid=f"e1-junction-{canonical_sha256(body)[:24]}",
        task_instance_sha256=left.task_instance_sha256,
        parent_contract_sha256=left.parent_contract_sha256,
        left_fragment_uid=left.fragment_uid,
        left_fragment_sha256=left.sha256,
        right_fragment_uid=right.fragment_uid,
        right_fragment_sha256=right.sha256,
        left_source_traj_uid=left.source_traj_uid,
        right_source_traj_uid=right.source_traj_uid,
        left_replay_post_state_snapshot_sha256=left_replay_post_state_snapshot_sha256,
        actual_junction_pre_state_snapshot_sha256=(actual_junction_pre_state_snapshot_sha256),
        right_source_pre_state_snapshot_sha256=right.pre_state_snapshot_sha256,
        actual_right_post_state_snapshot_sha256=actual_right_post_state_snapshot_sha256,
        actual_junction_pre_goal_ids=tuple(sorted(pre)),
        right_required_goal_ids=tuple(sorted(required)),
        right_expected_added_goal_ids=tuple(sorted(expected_added)),
        actual_right_post_goal_ids=tuple(sorted(post)),
        observed_added_goal_ids=tuple(sorted(observed_added)),
        observed_deleted_goal_ids=tuple(sorted(observed_deleted)),
        invariant_goal_ids=tuple(sorted(invariants)),
        independent_effect_verifier_receipt_sha256=(independent_effect_verifier_receipt_sha256),
        right_transition_restore_receipt_sha256s=tuple(right_transition_restore_receipt_sha256s),
        right_action_trace_sha256=right.action_trace_sha256,
        physical_call_delta=physical_call_delta.to_dict(),
        physical_call_delta_sha256=physical_call_delta.sha256,
        exact_state_junction=False,
        precondition_verified=True,
        expected_effect_transferred=True,
        invariants_preserved=True,
        effect_transfer_verified=True,
        requires_full_parent_replay=True,
        candidate_only=True,
        source_logprobs_combined=False,
    )


def _canonical_goals(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if any(not value or value != value.strip() for value in normalized):
        raise CompositionContractError("E1 goal IDs must be canonical")
    if len(normalized) != len(set(normalized)):
        raise CompositionContractError("E1 goal IDs must be unique")
    return tuple(sorted(normalized))


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CompositionContractError(f"{name} must be a lowercase SHA-256")
