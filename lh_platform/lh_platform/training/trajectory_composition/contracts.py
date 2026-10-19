"""Fail-closed contracts for replay-certified trajectory composition.

A composed path is a multi-source data artifact, never a sampled trajectory from
one behavior policy.  These contracts intentionally keep fragment provenance,
semantic environment actions, search output, and full replay evidence separate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

FRAGMENT_RECORD_SCHEMA = "lh_fragment_record/v1"
COMPOSITION_PLAN_SCHEMA = "lh_composition_plan/v1"
REPLAY_EVIDENCE_SCHEMA = "lh_composition_replay_evidence/v1"
COMPOSITION_RECEIPT_SCHEMA = "lh_composition_receipt/v1"
_SHA256_HEX = frozenset("0123456789abcdef")


class CompositionContractError(ValueError):
    """Raised when fragment or composition evidence is incomplete."""


class CompositionLevel(StrEnum):
    """Composition claims in increasing order of transfer risk."""

    E0_EXACT_STATE = "e0_exact_state"
    E1_VERIFIED_EFFECT = "e1_verified_effect_replay_required"


class FragmentSourceKind(StrEnum):
    """How the source fragment was obtained."""

    BEHAVIOR_ROLLOUT = "behavior_rollout"
    CHILD_TASK_ROLLOUT = "child_task_rollout"
    CONTROLLED_ORACLE = "controlled_oracle"
    FIXTURE = "fixture"


class CompositionTrainingUse(StrEnum):
    """Permitted use after successful full replay."""

    NONE = "none"
    SUPERVISED_OR_OEC = "supervised_or_oec_only"


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON and reject non-serializable evidence."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CompositionContractError(
            f"composition evidence is not canonical JSON: {error!r}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise CompositionContractError(f"{name} must be a lowercase SHA-256")


def _require_identifier(name: str, value: str) -> None:
    if not value or value != value.strip() or any(character in value for character in "\r\n"):
        raise CompositionContractError(f"{name} must be a non-empty single-line identifier")


def _ordered_unique(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if any(not value or value != value.strip() for value in normalized):
        raise CompositionContractError(f"{name} contains an empty or non-canonical value")
    if len(normalized) != len(set(normalized)):
        raise CompositionContractError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class FragmentRecordV1:
    """One independently sourced semantic-action fragment.

    ``required_goal_ids`` are applicability requirements for composition.  An
    E1 source may demonstrate an effect without satisfying those requirements
    in its own pre-state; only full replay can then admit the composed path.
    """

    fragment_uid: str
    task_instance_sha256: str
    parent_contract_sha256: str
    split: str
    source_traj_uid: str
    source_kind: FragmentSourceKind
    source_policy_id: str
    source_policy_sha256: str
    source_parent_success: bool
    source_started_from_parent_initial: bool
    pre_state_snapshot_sha256: str
    post_state_snapshot_sha256: str
    pre_goal_ids: tuple[str, ...]
    post_goal_ids: tuple[str, ...]
    required_goal_ids: tuple[str, ...]
    added_goal_ids: tuple[str, ...]
    deleted_goal_ids: tuple[str, ...]
    invariant_goal_ids: tuple[str, ...]
    canonical_actions: tuple[str, ...]
    action_trace_sha256: str
    source_action_provenance_sha256: str
    effect_verifier_id: str
    effect_verifier_receipt_sha256: str
    restore_receipt_sha256s: tuple[str, ...]
    transition_count: int
    independent_effect_verified: bool
    training_eligible: bool = False
    schema: str = FRAGMENT_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FRAGMENT_RECORD_SCHEMA:
            raise CompositionContractError("unsupported fragment-record schema")
        _require_identifier("fragment_uid", self.fragment_uid)
        _require_identifier("source_traj_uid", self.source_traj_uid)
        _require_identifier("source_policy_id", self.source_policy_id)
        _require_identifier("effect_verifier_id", self.effect_verifier_id)
        if not self.split.strip():
            raise CompositionContractError("fragment split is required")
        object.__setattr__(self, "source_kind", FragmentSourceKind(self.source_kind))
        for name, value in (
            ("task_instance_sha256", self.task_instance_sha256),
            ("parent_contract_sha256", self.parent_contract_sha256),
            ("source_policy_sha256", self.source_policy_sha256),
            ("pre_state_snapshot_sha256", self.pre_state_snapshot_sha256),
            ("post_state_snapshot_sha256", self.post_state_snapshot_sha256),
            ("action_trace_sha256", self.action_trace_sha256),
            ("source_action_provenance_sha256", self.source_action_provenance_sha256),
            ("effect_verifier_receipt_sha256", self.effect_verifier_receipt_sha256),
        ):
            _require_sha256(name, value)
        for receipt in self.restore_receipt_sha256s:
            _require_sha256("restore_receipt_sha256", receipt)
        if self.transition_count < 1:
            raise CompositionContractError("fragment transition_count must be positive")
        if self.transition_count != len(self.canonical_actions):
            raise CompositionContractError(
                "fragment transition_count must equal semantic-action count"
            )
        if self.transition_count != len(self.restore_receipt_sha256s):
            raise CompositionContractError(
                "every fragment transition requires one restore/replay receipt"
            )
        actions = tuple(str(action) for action in self.canonical_actions)
        if any(
            not action or action != action.strip() or "\n" in action or "\r" in action
            for action in actions
        ):
            raise CompositionContractError(
                "fragment actions must be non-empty canonical semantic actions"
            )
        if self.action_trace_sha256 != canonical_sha256({"actions": list(actions)}):
            raise CompositionContractError("fragment semantic-action trace hash mismatch")
        object.__setattr__(self, "canonical_actions", actions)
        pre = frozenset(_ordered_unique(self.pre_goal_ids, name="pre_goal_ids"))
        post = frozenset(_ordered_unique(self.post_goal_ids, name="post_goal_ids"))
        required = frozenset(_ordered_unique(self.required_goal_ids, name="required_goal_ids"))
        added = frozenset(_ordered_unique(self.added_goal_ids, name="added_goal_ids"))
        deleted = frozenset(_ordered_unique(self.deleted_goal_ids, name="deleted_goal_ids"))
        invariants = frozenset(_ordered_unique(self.invariant_goal_ids, name="invariant_goal_ids"))
        if added != post - pre:
            raise CompositionContractError("fragment added goals differ from verified states")
        if deleted != pre - post:
            raise CompositionContractError("fragment deleted goals differ from verified states")
        if added & deleted:
            raise CompositionContractError("fragment cannot add and delete the same goal")
        if deleted & invariants:
            raise CompositionContractError("fragment deletes a declared invariant")
        if not added and not deleted:
            raise CompositionContractError("fragment has no independently verified goal effect")
        if not required and self.source_kind is FragmentSourceKind.CHILD_TASK_ROLLOUT:
            raise CompositionContractError("child-task fragments require an explicit contract")
        if not self.independent_effect_verified:
            raise CompositionContractError("unverified effects cannot become fragments")
        if self.training_eligible:
            raise CompositionContractError("raw fragments are never directly training eligible")
        object.__setattr__(self, "pre_goal_ids", tuple(sorted(pre)))
        object.__setattr__(self, "post_goal_ids", tuple(sorted(post)))
        object.__setattr__(self, "required_goal_ids", tuple(sorted(required)))
        object.__setattr__(self, "added_goal_ids", tuple(sorted(added)))
        object.__setattr__(self, "deleted_goal_ids", tuple(sorted(deleted)))
        object.__setattr__(self, "invariant_goal_ids", tuple(sorted(invariants)))

    @classmethod
    def build(
        cls,
        *,
        task_instance_sha256: str,
        parent_contract_sha256: str,
        split: str,
        source_traj_uid: str,
        source_kind: FragmentSourceKind,
        source_policy_id: str,
        source_policy_sha256: str,
        source_parent_success: bool,
        source_started_from_parent_initial: bool,
        pre_state_snapshot_sha256: str,
        post_state_snapshot_sha256: str,
        pre_goal_ids: tuple[str, ...],
        post_goal_ids: tuple[str, ...],
        required_goal_ids: tuple[str, ...],
        invariant_goal_ids: tuple[str, ...],
        canonical_actions: tuple[str, ...],
        source_action_provenance_sha256: str,
        effect_verifier_id: str,
        effect_verifier_receipt_sha256: str,
        restore_receipt_sha256s: tuple[str, ...],
    ) -> FragmentRecordV1:
        action_trace_sha256 = canonical_sha256({"actions": list(canonical_actions)})
        body = {
            "task_instance_sha256": task_instance_sha256,
            "parent_contract_sha256": parent_contract_sha256,
            "split": split,
            "source_traj_uid": source_traj_uid,
            "source_kind": FragmentSourceKind(source_kind).value,
            "source_started_from_parent_initial": bool(source_started_from_parent_initial),
            "pre_state_snapshot_sha256": pre_state_snapshot_sha256,
            "post_state_snapshot_sha256": post_state_snapshot_sha256,
            "required_goal_ids": sorted(required_goal_ids),
            "action_trace_sha256": action_trace_sha256,
            "effect_verifier_receipt_sha256": effect_verifier_receipt_sha256,
        }
        return cls(
            fragment_uid=f"fragment-{canonical_sha256(body)[:24]}",
            task_instance_sha256=task_instance_sha256,
            parent_contract_sha256=parent_contract_sha256,
            split=split,
            source_traj_uid=source_traj_uid,
            source_kind=source_kind,
            source_policy_id=source_policy_id,
            source_policy_sha256=source_policy_sha256,
            source_parent_success=bool(source_parent_success),
            source_started_from_parent_initial=bool(source_started_from_parent_initial),
            pre_state_snapshot_sha256=pre_state_snapshot_sha256,
            post_state_snapshot_sha256=post_state_snapshot_sha256,
            pre_goal_ids=tuple(pre_goal_ids),
            post_goal_ids=tuple(post_goal_ids),
            required_goal_ids=tuple(required_goal_ids),
            added_goal_ids=tuple(sorted(set(post_goal_ids) - set(pre_goal_ids))),
            deleted_goal_ids=tuple(sorted(set(pre_goal_ids) - set(post_goal_ids))),
            invariant_goal_ids=tuple(invariant_goal_ids),
            canonical_actions=tuple(canonical_actions),
            action_trace_sha256=action_trace_sha256,
            source_action_provenance_sha256=source_action_provenance_sha256,
            effect_verifier_id=effect_verifier_id,
            effect_verifier_receipt_sha256=effect_verifier_receipt_sha256,
            restore_receipt_sha256s=tuple(restore_receipt_sha256s),
            transition_count=len(canonical_actions),
            independent_effect_verified=True,
        )

    @property
    def source_preconditions_satisfied(self) -> bool:
        return set(self.required_goal_ids) <= set(self.pre_goal_ids)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fragment_uid": self.fragment_uid,
            "task_instance_sha256": self.task_instance_sha256,
            "parent_contract_sha256": self.parent_contract_sha256,
            "split": self.split,
            "source_traj_uid": self.source_traj_uid,
            "source_kind": self.source_kind.value,
            "source_policy_id": self.source_policy_id,
            "source_policy_sha256": self.source_policy_sha256,
            "source_parent_success": self.source_parent_success,
            "source_started_from_parent_initial": self.source_started_from_parent_initial,
            "pre_state_snapshot_sha256": self.pre_state_snapshot_sha256,
            "post_state_snapshot_sha256": self.post_state_snapshot_sha256,
            "pre_goal_ids": list(self.pre_goal_ids),
            "post_goal_ids": list(self.post_goal_ids),
            "required_goal_ids": list(self.required_goal_ids),
            "source_preconditions_satisfied": self.source_preconditions_satisfied,
            "added_goal_ids": list(self.added_goal_ids),
            "deleted_goal_ids": list(self.deleted_goal_ids),
            "invariant_goal_ids": list(self.invariant_goal_ids),
            "canonical_actions": list(self.canonical_actions),
            "action_trace_sha256": self.action_trace_sha256,
            "source_action_provenance_sha256": self.source_action_provenance_sha256,
            "effect_verifier_id": self.effect_verifier_id,
            "effect_verifier_receipt_sha256": self.effect_verifier_receipt_sha256,
            "restore_receipt_sha256s": list(self.restore_receipt_sha256s),
            "transition_count": self.transition_count,
            "independent_effect_verified": self.independent_effect_verified,
            "training_eligible": self.training_eligible,
        }


@dataclass(frozen=True)
class CompositionPlanV1:
    """Deterministic search output awaiting complete environment replay."""

    plan_uid: str
    task_instance_sha256: str
    parent_contract_sha256: str
    composition_level: CompositionLevel
    initial_state_snapshot_sha256: str
    initial_goal_ids: tuple[str, ...]
    target_goal_ids: tuple[str, ...]
    fragment_uids: tuple[str, ...]
    source_traj_uids: tuple[str, ...]
    canonical_actions: tuple[str, ...]
    total_action_cost: int
    all_source_parents_failed: bool
    exact_state_junctions: bool
    requires_full_replay: bool
    candidate_only: bool
    synthetic_behavior_policy: bool
    planner_config_sha256: str
    schema: str = COMPOSITION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COMPOSITION_PLAN_SCHEMA:
            raise CompositionContractError("unsupported composition-plan schema")
        _require_identifier("plan_uid", self.plan_uid)
        object.__setattr__(self, "composition_level", CompositionLevel(self.composition_level))
        for name, value in (
            ("task_instance_sha256", self.task_instance_sha256),
            ("parent_contract_sha256", self.parent_contract_sha256),
            ("initial_state_snapshot_sha256", self.initial_state_snapshot_sha256),
            ("planner_config_sha256", self.planner_config_sha256),
        ):
            _require_sha256(name, value)
        if not self.fragment_uids or len(self.fragment_uids) != len(set(self.fragment_uids)):
            raise CompositionContractError("composition requires unique fragments")
        if len(set(self.source_traj_uids)) < 2:
            raise CompositionContractError("composition requires at least two source trajectories")
        if self.total_action_cost != len(self.canonical_actions) or self.total_action_cost < 1:
            raise CompositionContractError("composition action cost is inconsistent")
        if not set(self.target_goal_ids):
            raise CompositionContractError("composition requires parent target goals")
        if not self.all_source_parents_failed:
            raise CompositionContractError(
                "support-recovery plans cannot consume a successful parent source"
            )
        if not self.requires_full_replay or not self.candidate_only:
            raise CompositionContractError("a searched composition is candidate-only until replay")
        if self.synthetic_behavior_policy:
            raise CompositionContractError(
                "multi-source composition cannot declare a synthetic behavior policy"
            )
        if self.composition_level is CompositionLevel.E0_EXACT_STATE:
            if not self.exact_state_junctions:
                raise CompositionContractError("E0 requires exact-state junctions")
        elif self.exact_state_junctions:
            raise CompositionContractError("E1 must not be mislabeled exact-state")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_uid": self.plan_uid,
            "task_instance_sha256": self.task_instance_sha256,
            "parent_contract_sha256": self.parent_contract_sha256,
            "composition_level": self.composition_level.value,
            "initial_state_snapshot_sha256": self.initial_state_snapshot_sha256,
            "initial_goal_ids": list(self.initial_goal_ids),
            "target_goal_ids": list(self.target_goal_ids),
            "fragment_uids": list(self.fragment_uids),
            "source_traj_uids": list(self.source_traj_uids),
            "canonical_actions": list(self.canonical_actions),
            "total_action_cost": self.total_action_cost,
            "all_source_parents_failed": self.all_source_parents_failed,
            "exact_state_junctions": self.exact_state_junctions,
            "requires_full_replay": self.requires_full_replay,
            "candidate_only": self.candidate_only,
            "synthetic_behavior_policy": self.synthetic_behavior_policy,
            "planner_config_sha256": self.planner_config_sha256,
        }


@dataclass(frozen=True)
class ReplayEvidenceV1:
    """Observed result of replaying every planned action from the parent start."""

    plan_uid: str
    task_instance_sha256: str
    parent_contract_sha256: str
    initial_state_snapshot_sha256: str
    final_state_snapshot_sha256: str
    executed_canonical_actions: tuple[str, ...]
    final_goal_ids: tuple[str, ...]
    parent_contract_verified: bool
    replay_completed: bool
    transition_restore_receipt_sha256s: tuple[str, ...]
    independent_parent_verifier_id: str
    independent_parent_verifier_receipt_sha256: str
    environment_identity_sha256: str
    replay_seed: int
    schema: str = REPLAY_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPLAY_EVIDENCE_SCHEMA:
            raise CompositionContractError("unsupported replay-evidence schema")
        _require_identifier("plan_uid", self.plan_uid)
        _require_identifier("independent_parent_verifier_id", self.independent_parent_verifier_id)
        for name, value in (
            ("task_instance_sha256", self.task_instance_sha256),
            ("parent_contract_sha256", self.parent_contract_sha256),
            ("initial_state_snapshot_sha256", self.initial_state_snapshot_sha256),
            ("final_state_snapshot_sha256", self.final_state_snapshot_sha256),
            (
                "independent_parent_verifier_receipt_sha256",
                self.independent_parent_verifier_receipt_sha256,
            ),
            ("environment_identity_sha256", self.environment_identity_sha256),
        ):
            _require_sha256(name, value)
        for receipt in self.transition_restore_receipt_sha256s:
            _require_sha256("transition_restore_receipt_sha256", receipt)
        if len(self.executed_canonical_actions) != len(self.transition_restore_receipt_sha256s):
            raise CompositionContractError(
                "full replay requires one restore receipt per executed action"
            )
        if isinstance(self.replay_seed, bool):
            raise CompositionContractError("replay seed must be an integer")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_uid": self.plan_uid,
            "task_instance_sha256": self.task_instance_sha256,
            "parent_contract_sha256": self.parent_contract_sha256,
            "initial_state_snapshot_sha256": self.initial_state_snapshot_sha256,
            "final_state_snapshot_sha256": self.final_state_snapshot_sha256,
            "executed_canonical_actions": list(self.executed_canonical_actions),
            "final_goal_ids": list(self.final_goal_ids),
            "parent_contract_verified": self.parent_contract_verified,
            "replay_completed": self.replay_completed,
            "transition_restore_receipt_sha256s": list(self.transition_restore_receipt_sha256s),
            "independent_parent_verifier_id": self.independent_parent_verifier_id,
            "independent_parent_verifier_receipt_sha256": (
                self.independent_parent_verifier_receipt_sha256
            ),
            "environment_identity_sha256": self.environment_identity_sha256,
            "replay_seed": self.replay_seed,
        }


@dataclass(frozen=True)
class CompositionReceiptV1:
    """Final admission decision; successful compositions remain off-policy data."""

    plan_uid: str
    plan_sha256: str
    replay_evidence_sha256: str
    admitted: bool
    violations: tuple[str, ...]
    training_use: CompositionTrainingUse
    on_policy_eligible: bool
    scientific_result: bool
    schema: str = COMPOSITION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COMPOSITION_RECEIPT_SCHEMA:
            raise CompositionContractError("unsupported composition-receipt schema")
        _require_identifier("plan_uid", self.plan_uid)
        _require_sha256("plan_sha256", self.plan_sha256)
        _require_sha256("replay_evidence_sha256", self.replay_evidence_sha256)
        object.__setattr__(self, "training_use", CompositionTrainingUse(self.training_use))
        if self.admitted != (not self.violations):
            raise CompositionContractError("composition admission and violations disagree")
        expected_use = (
            CompositionTrainingUse.SUPERVISED_OR_OEC
            if self.admitted
            else CompositionTrainingUse.NONE
        )
        if self.training_use is not expected_use:
            raise CompositionContractError("composition training use disagrees with admission")
        if self.on_policy_eligible:
            raise CompositionContractError("composed paths are never on-policy samples")
        if self.scientific_result:
            raise CompositionContractError(
                "a replay receipt alone cannot be labeled a scientific result"
            )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_uid": self.plan_uid,
            "plan_sha256": self.plan_sha256,
            "replay_evidence_sha256": self.replay_evidence_sha256,
            "admitted": self.admitted,
            "violations": list(self.violations),
            "training_use": self.training_use.value,
            "on_policy_eligible": self.on_policy_eligible,
            "scientific_result": self.scientific_result,
        }


def certify_composition(
    plan: CompositionPlanV1,
    evidence: ReplayEvidenceV1,
) -> CompositionReceiptV1:
    """Admit only byte-bound, complete, independently verified parent replay."""

    violations: list[str] = []
    if evidence.plan_uid != plan.plan_uid:
        violations.append("plan_uid_mismatch")
    if evidence.task_instance_sha256 != plan.task_instance_sha256:
        violations.append("task_instance_mismatch")
    if evidence.parent_contract_sha256 != plan.parent_contract_sha256:
        violations.append("parent_contract_mismatch")
    if evidence.initial_state_snapshot_sha256 != plan.initial_state_snapshot_sha256:
        violations.append("parent_initial_state_mismatch")
    if evidence.executed_canonical_actions != plan.canonical_actions:
        violations.append("semantic_action_trace_mismatch")
    if len(evidence.transition_restore_receipt_sha256s) != plan.total_action_cost:
        violations.append("incomplete_transition_restore_evidence")
    if not evidence.replay_completed:
        violations.append("full_replay_incomplete")
    if not evidence.parent_contract_verified:
        violations.append("parent_contract_not_verified")
    if not set(plan.target_goal_ids) <= set(evidence.final_goal_ids):
        violations.append("target_goals_not_verified")
    ordered = tuple(sorted(set(violations)))
    admitted = not ordered
    return CompositionReceiptV1(
        plan_uid=plan.plan_uid,
        plan_sha256=plan.sha256,
        replay_evidence_sha256=evidence.sha256,
        admitted=admitted,
        violations=ordered,
        training_use=(
            CompositionTrainingUse.SUPERVISED_OR_OEC if admitted else CompositionTrainingUse.NONE
        ),
        on_policy_eligible=False,
        scientific_result=False,
    )
