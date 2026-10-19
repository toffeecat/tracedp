"""Evidence-gated controlled aliasing fixtures for DPPO mechanism tests.

This module does not manufacture observations or subproblem labels.  It only
admits already-collected ALFWorld transitions when the pinned G2PO credit key
(``record.observation``) collides across independently verified goal frontiers
and both pre-action states have auditable snapshot/restore evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from lh_platform.alfworld_state import (
    ALFWORLD_REPLAY_BACKEND,
    ALFWORLD_RESTORE_RECEIPT_SCHEMA,
    ALFWORLD_STATE_CAPABILITY_SCHEMA,
    REQUIRED_RESTORE_COMPARISONS,
)
from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.dppo.analysis import load_rollout_artifact
from lh_platform.training.g2po.actor_protocol import (
    g2po_anchor_tracker_source_sha256,
)
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.base import goal_state_projection

CONTROLLED_ALIASING_MANIFEST_SCHEMA = "lh_dppo_controlled_aliasing_manifest/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TRANSITION_SOURCES = frozenset({"on_policy", "controlled_environment_intervention"})
_PINNED_G2PO_REVISION = "b8ffaf56ad241ce73507908e6ed30209cafd324d"
_PINNED_ACTOR_PROTOCOL = "official_g2po_alfworld_actor_b8ffaf56"
_CONTROLLED_RESTORE_COMPARISONS = (
    *REQUIRED_RESTORE_COMPARISONS,
    "facts",
    "admissible_actions",
)


def build_controlled_aliasing_manifest(
    records: Iterable[TrainingTrajectoryRecord],
    *,
    source_rollout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Admit only official-anchor collisions backed by real replay evidence.

    The input may be a natural on-policy rollout or an explicitly labelled
    environment-intervention fixture.  A raw TextWorld observation collision
    is never substituted for the source G2PO anchor.
    """

    rows = tuple(records)
    verifier_errors = _independent_goal_verifier_errors(rows)
    eligible: list[tuple[TrainingTrajectoryRecord, dict[str, Any]]] = []
    ineligible: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reasons, evidence = _verify_record(
            row,
            goal_verifier_error=verifier_errors.get((row.traj_uid, row.step_idx)),
        )
        if reasons:
            reason_counts.update(reasons)
            ineligible.append(
                {
                    "record_ref": _record_ref(row),
                    "reasons": list(reasons),
                }
            )
        else:
            eligible.append((row, evidence))

    by_anchor: dict[tuple[str, str, str], list[tuple[TrainingTrajectoryRecord, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for row, evidence in eligible:
        by_anchor[(row.task_id, row.group_uid, row.observation)].append((row, evidence))

    collision_groups: list[dict[str, Any]] = []
    candidate_pair_count = 0
    candidate_transition_pair_count = 0
    for (task_id, group_uid, official_anchor), members in sorted(by_anchor.items()):
        by_goal_state: dict[
            tuple[str, str], list[tuple[TrainingTrajectoryRecord, dict[str, Any]]]
        ] = defaultdict(list)
        for row, evidence in members:
            by_goal_state[
                (
                    str(row.extra["residual_goal_signature"]),
                    str(row.extra["frontier_signature"]),
                )
            ].append((row, evidence))
        state_items = sorted(by_goal_state.items())
        state_pairs: list[dict[str, Any]] = []
        for left_index, (left_identity, left_members) in enumerate(state_items):
            for right_identity, right_members in state_items[left_index + 1 :]:
                if left_identity[1] == right_identity[1]:
                    continue
                left = sorted(left_members, key=_member_sort_key)[0]
                right = sorted(right_members, key=_member_sort_key)[0]
                if left[0].state_snapshot_sha256 == right[0].state_snapshot_sha256:
                    # A single recoverable state cannot legitimately have two
                    # verifier-owned frontiers.  Treat it as annotation drift,
                    # not as evidence for the mechanism.
                    continue
                if not _pair_controls_match(left[0], right[0]):
                    # Equal G2PO anchors are necessary but not sufficient for
                    # a controlled fixture.  Keep raw text, affordances, and
                    # budget fixed so the residual frontier is the intended
                    # varying factor.
                    continue
                state_pairs.append(
                    {
                        "left_record_ref": _record_ref(left[0]),
                        "right_record_ref": _record_ref(right[0]),
                        "left_frontier_signature": left_identity[1],
                        "right_frontier_signature": right_identity[1],
                    }
                )
                candidate_pair_count += 1
                candidate_transition_pair_count += len(left_members) * len(right_members)
        if not state_pairs:
            continue
        collision_groups.append(
            {
                "task_id": task_id,
                "group_uid": group_uid,
                "official_anchor": official_anchor,
                "official_anchor_sha256": canonical_sha256(official_anchor),
                "verified_transition_count": len(members),
                "verified_frontier_states": [
                    _goal_state_payload(identity, state_members)
                    for identity, state_members in state_items
                ],
                "candidate_state_pairs": state_pairs,
            }
        )

    blocked_by: list[str] = []
    if ineligible:
        blocked_by.append("ineligible_evidence_records_present")
    if not collision_groups:
        blocked_by.append("no_verified_official_anchor_cross_frontier_collision")
    source_payload = _source_rollout_payload(source_rollout_path)
    return {
        "schema": CONTROLLED_ALIASING_MANIFEST_SCHEMA,
        "source_rollout": source_payload,
        "record_count": len(rows),
        "eligible_record_count": len(eligible),
        "ineligible_record_count": len(ineligible),
        "ineligible_reason_counts": dict(sorted(reason_counts.items())),
        "ineligible_records": ineligible,
        "official_anchor_group_count": len(by_anchor),
        "collision_group_count": len(collision_groups),
        "candidate_pair_count": candidate_pair_count,
        "candidate_transition_pair_count": candidate_transition_pair_count,
        "collision_groups": collision_groups,
        "admitted": not blocked_by,
        "blocked_by": blocked_by,
        "claim_scope": (
            "controlled_cross_frontier_collision_candidate_not_credit_or_policy_effectiveness"
        ),
    }


def analyze_controlled_aliasing_artifact(
    rollout_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load one persisted rollout and write its controlled-fixture admission."""

    source = Path(rollout_path).resolve()
    manifest = build_controlled_aliasing_manifest(
        load_rollout_artifact(source),
        source_rollout_path=source,
    )
    if output_path is not None:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def _verify_record(
    row: TrainingTrajectoryRecord,
    *,
    goal_verifier_error: str | None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    reasons: list[str] = []
    if row.benchmark != "alfworld_native":
        reasons.append("unsupported_benchmark")
    if row.method_source not in _ALLOWED_TRANSITION_SOURCES:
        reasons.append("unsupported_transition_source")
    if row.extra.get("actor_protocol_id") != _PINNED_ACTOR_PROTOCOL:
        reasons.append("official_anchor_protocol_unbound")
    if row.extra.get("g2po_source_revision") != _PINNED_G2PO_REVISION:
        reasons.append("official_anchor_source_revision_unbound")
    if row.extra.get("official_anchor_projection_verified") is not True:
        reasons.append("official_anchor_projection_unverified")
    if row.extra.get("anchor_tracker_implementation_sha256") != g2po_anchor_tracker_source_sha256():
        reasons.append("official_anchor_projection_implementation_unbound")
    if not row.observation:
        reasons.append("official_anchor_missing")
    raw_observation = row.extra.get("alfworld_raw_observation")
    if not isinstance(raw_observation, str) or not raw_observation:
        reasons.append("raw_environment_observation_missing")
    if not _SHA256.fullmatch(row.goal_hash or ""):
        reasons.append("goal_hash_invalid")
    if not _SHA256.fullmatch(row.action_affordance_hash or ""):
        reasons.append("action_affordance_hash_invalid")
    elif isinstance(row.extra.get("alfworld_admissible_actions"), (list, tuple)) and (
        canonical_sha256(list(row.extra["alfworld_admissible_actions"]))
        != row.action_affordance_hash
    ):
        reasons.append("action_affordance_hash_mismatch")
    if row.remaining_budget is None or row.remaining_budget < 0:
        reasons.append("remaining_budget_invalid")
    if (
        row.extra.get("goal_annotation_verified") is not True
        or row.extra.get("subproblem_source") != "environment_contract"
    ):
        reasons.append("environment_goal_annotation_unverified")
    if goal_verifier_error is not None:
        reasons.append(goal_verifier_error)
    frontier = _string_sequence(row.extra.get("active_frontier_ids"))
    residual = _string_sequence(row.extra.get("residual_goal_ids"))
    if frontier is None or canonical_sha256(list(frontier)) != row.extra.get("frontier_signature"):
        reasons.append("frontier_signature_invalid")
    if residual is None or canonical_sha256(list(residual)) != row.extra.get(
        "residual_goal_signature"
    ):
        reasons.append("residual_goal_signature_invalid")
    if row.extra.get("restore_verified") is not True:
        reasons.append("restore_not_verified")
    if not _SHA256.fullmatch(row.state_snapshot_sha256 or ""):
        reasons.append("state_snapshot_sha256_invalid")

    snapshot_path = _path(row.extra.get("state_snapshot_artifact_path"))
    declared_artifact_sha256 = row.extra.get("state_snapshot_artifact_sha256")
    if declared_artifact_sha256 is not None and (
        declared_artifact_sha256 != row.state_snapshot_sha256
    ):
        reasons.append("state_snapshot_artifact_identity_mismatch")

    receipt_sha256 = row.extra.get("restore_probe_receipt_sha256")
    receipt_path = _path(row.extra.get("restore_probe_receipt_path"))
    reasons.extend(
        validate_alfworld_replay_evidence(
            snapshot_sha256=row.state_snapshot_sha256,
            snapshot_path=snapshot_path,
            receipt_sha256=receipt_sha256,
            receipt_path=receipt_path,
            action=row.extra.get("environment_action"),
        )
    )
    receipt_revalidated = not any(reason.startswith("restore_receipt") for reason in reasons)

    return tuple(dict.fromkeys(reasons)), {
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else "",
        "receipt_path": str(receipt_path) if receipt_path is not None else "",
        "receipt_sha256": str(receipt_sha256 or ""),
        "restore_receipt_revalidated": receipt_revalidated,
    }


def validate_alfworld_replay_evidence(
    *,
    snapshot_sha256: object,
    snapshot_path: Path | None,
    receipt_sha256: object,
    receipt_path: Path | None,
    action: object,
    expected_replay_outcome: Mapping[str, Any] | None = None,
    expected_gamefile: str | None = None,
    expected_action_history: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Revalidate one physical ALFWorld snapshot/restore evidence pair.

    This public evidence seam is shared by collision and continuation audits so
    neither path can admit self-reported restore booleans.
    """

    reasons: list[str] = []
    if not isinstance(snapshot_sha256, str) or not _SHA256.fullmatch(snapshot_sha256):
        reasons.append("state_snapshot_sha256_invalid")
    if snapshot_path is None or not snapshot_path.is_file():
        reasons.append("state_snapshot_artifact_missing")
    elif not isinstance(snapshot_sha256, str):
        reasons.append("state_snapshot_artifact_sha256_mismatch")
    elif _sha256_file(snapshot_path) != snapshot_sha256:
        reasons.append("state_snapshot_artifact_sha256_mismatch")
    elif snapshot_path.name != f"{snapshot_sha256}.pickle":
        reasons.append("state_snapshot_artifact_path_noncanonical")
    elif not _valid_replay_snapshot(snapshot_path):
        reasons.append("state_snapshot_payload_invalid")
    elif expected_gamefile is not None or expected_action_history is not None:
        payload = _load_replay_snapshot_payload(snapshot_path)
        if payload is None:
            reasons.append("state_snapshot_payload_invalid")
        else:
            if expected_gamefile is not None and payload.get("gamefiles") != (expected_gamefile,):
                reasons.append("state_snapshot_gamefile_mismatch")
            if (
                expected_action_history is not None
                and payload.get("action_history") != expected_action_history
            ):
                reasons.append("state_snapshot_action_history_mismatch")

    if not isinstance(receipt_sha256, str) or not _SHA256.fullmatch(receipt_sha256):
        reasons.append("restore_receipt_sha256_invalid")
    receipt: Mapping[str, Any] | None = None
    if receipt_path is None or not receipt_path.is_file():
        reasons.append("restore_receipt_missing")
    elif isinstance(receipt_sha256, str) and _sha256_file(receipt_path) != receipt_sha256:
        reasons.append("restore_receipt_sha256_mismatch")
    else:
        try:
            parsed = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt = parsed if isinstance(parsed, Mapping) else None
        except (OSError, json.JSONDecodeError):
            receipt = None
        if receipt is None:
            reasons.append("restore_receipt_invalid")
        elif not _valid_restore_receipt_fields(
            snapshot_sha256=snapshot_sha256,
            action=action,
            receipt=receipt,
        ):
            reasons.append("restore_receipt_content_mismatch")
        elif isinstance(snapshot_sha256, str) and isinstance(action, str):
            expected = _expected_receipt_filename_for(snapshot_sha256, action)
            if receipt_path.name != expected:
                reasons.append("restore_receipt_path_noncanonical")
        if receipt is not None and expected_replay_outcome is not None:
            expected_outcome_sha256 = canonical_sha256(expected_replay_outcome)
            if (
                receipt.get("replay_outcome_sha256") != expected_outcome_sha256
                or receipt.get("first_outcome_sha256") != expected_outcome_sha256
            ):
                reasons.append("restore_receipt_replay_outcome_mismatch")
    return tuple(dict.fromkeys(reasons))


def _valid_restore_receipt(row: TrainingTrajectoryRecord, receipt: Mapping[str, Any]) -> bool:
    return _valid_restore_receipt_fields(
        snapshot_sha256=row.state_snapshot_sha256,
        action=row.extra.get("environment_action"),
        receipt=receipt,
    )


def _valid_restore_receipt_fields(
    *,
    snapshot_sha256: object,
    action: object,
    receipt: Mapping[str, Any],
) -> bool:
    comparisons = receipt.get("comparisons")
    return bool(
        receipt.get("schema") == ALFWORLD_RESTORE_RECEIPT_SCHEMA
        and receipt.get("state_snapshot_sha256") == snapshot_sha256
        and receipt.get("restore_verified") is True
        and isinstance(comparisons, Mapping)
        and all(comparisons.get(name) is True for name in _CONTROLLED_RESTORE_COMPARISONS)
        and (not isinstance(action, str) or receipt.get("action") == action)
    )


def _valid_replay_snapshot(path: Path) -> bool:
    payload = _load_replay_snapshot_payload(path)
    if payload is None:
        return False
    gamefiles = payload.get("gamefiles")
    action_history = payload.get("action_history")
    wrapper_state = payload.get("wrapper_state")
    return bool(
        payload.get("schema") == ALFWORLD_STATE_CAPABILITY_SCHEMA
        and payload.get("backend") == ALFWORLD_REPLAY_BACKEND
        and isinstance(gamefiles, tuple)
        and len(gamefiles) == 1
        and all(isinstance(item, str) and item for item in gamefiles)
        and isinstance(action_history, tuple)
        and all(isinstance(item, str) and item for item in action_history)
        and isinstance(wrapper_state, Mapping)
    )


def _load_replay_snapshot_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = _RestrictedUnpickler(io.BytesIO(path.read_bytes())).load()
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"snapshot payload may not load global {module}.{name}")


def _expected_receipt_filename(row: TrainingTrajectoryRecord) -> str:
    action = str(row.extra.get("environment_action", ""))
    return _expected_receipt_filename_for(row.state_snapshot_sha256, action)


def _expected_receipt_filename_for(snapshot_sha256: str, action: str) -> str:
    action_digest = hashlib.sha256(action.encode("utf-8")).hexdigest()[:16]
    return f"{snapshot_sha256}_{action_digest}.json"


def _independent_goal_verifier_errors(
    rows: tuple[TrainingTrajectoryRecord, ...],
) -> dict[tuple[str, int], str]:
    compiler = AlfWorldSubproblemCompiler()
    errors: dict[tuple[str, int], str] = {}
    previous_by_trajectory: dict[str, Any] = {}
    for row in sorted(rows, key=lambda item: (item.traj_uid, item.step_idx)):
        key = (row.traj_uid, row.step_idx)
        verifier_input = row.extra.get("goal_verifier_input")
        if not isinstance(verifier_input, Mapping) or verifier_input.get("schema") != (
            "lh_alfworld_goal_state_verifier_input/v1"
        ):
            errors[key] = "goal_verifier_input_missing"
            continue
        metadata = verifier_input.get("task_metadata")
        facts = verifier_input.get("facts")
        if not isinstance(metadata, Mapping) or not isinstance(facts, (list, tuple)):
            errors[key] = "goal_verifier_input_invalid"
            continue
        try:
            graph = compiler.compile_task(metadata)
            before = compiler.project_state(
                metadata,
                facts,
                previous=previous_by_trajectory.get(row.traj_uid),
            )
            projection = goal_state_projection(graph, before)
            retained_graph = row.extra.get("goal_graph")
            matches = bool(
                isinstance(retained_graph, Mapping)
                and canonical_sha256(retained_graph) == graph.sha256
                and row.extra.get("goal_graph_sha256") == graph.sha256
                and list(projection["residual_goal_ids"])
                == list(row.extra.get("residual_goal_ids", ()))
                and projection["residual_goal_signature"]
                == row.extra.get("residual_goal_signature")
                and list(projection["active_frontier_ids"])
                == list(row.extra.get("active_frontier_ids", ()))
                and projection["frontier_signature"] == row.extra.get("frontier_signature")
            )
            if not matches:
                errors[key] = "independent_goal_verifier_mismatch"
            next_facts = row.extra.get("alfworld_next_facts")
            if isinstance(next_facts, (list, tuple)):
                previous_by_trajectory[row.traj_uid] = compiler.project_state(
                    metadata,
                    next_facts,
                    previous=before,
                )
            else:
                previous_by_trajectory[row.traj_uid] = before
        except (TypeError, ValueError, KeyError):
            errors[key] = "independent_goal_verifier_failed"
    return errors


def _goal_state_payload(
    identity: tuple[str, str],
    members: list[tuple[TrainingTrajectoryRecord, dict[str, Any]]],
) -> dict[str, Any]:
    ordered = sorted(members, key=_member_sort_key)
    exemplar = ordered[0][0]
    return {
        "residual_goal_signature": identity[0],
        "frontier_signature": identity[1],
        "residual_goal_ids": list(exemplar.extra["residual_goal_ids"]),
        "active_frontier_ids": list(exemplar.extra["active_frontier_ids"]),
        "member_count": len(ordered),
        "members": [
            {
                **_record_ref(row),
                "state_snapshot_artifact_path": evidence["snapshot_path"],
                "restore_probe_receipt_path": evidence["receipt_path"],
                "restore_probe_receipt_sha256": evidence["receipt_sha256"],
                "restore_receipt_revalidated": evidence["restore_receipt_revalidated"],
            }
            for row, evidence in ordered
        ],
    }


def _record_ref(row: TrainingTrajectoryRecord) -> dict[str, Any]:
    return {
        "traj_uid": row.traj_uid,
        "step_idx": row.step_idx,
        "state_snapshot_sha256": row.state_snapshot_sha256,
    }


def _pair_controls_match(left: TrainingTrajectoryRecord, right: TrainingTrajectoryRecord) -> bool:
    return bool(
        left.extra.get("alfworld_raw_observation") == right.extra.get("alfworld_raw_observation")
        and left.action_affordance_hash == right.action_affordance_hash
        and left.remaining_budget == right.remaining_budget
        and left.goal_hash == right.goal_hash
    )


def _member_sort_key(
    item: tuple[TrainingTrajectoryRecord, dict[str, Any]],
) -> tuple[str, int, str]:
    row = item[0]
    return row.traj_uid, row.step_idx, row.state_snapshot_sha256


def _string_sequence(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _path(value: object) -> Path | None:
    return Path(value).resolve() if isinstance(value, str) and value else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_rollout_payload(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path).resolve()
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
    }
