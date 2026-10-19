"""Canonical training trajectory dataset helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from lh_platform.training.contracts import (
    TRAINING_DATASET_SCHEMA,
    TrainingDatasetManifest,
    TrainingTrajectoryRecord,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(records: Iterable[TrainingTrajectoryRecord], path: Path) -> TrainingDatasetManifest:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(records)
    with path.open("w", encoding="utf-8") as fh:
        for record in materialized:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return build_manifest(path, materialized)


def read_jsonl(path: Path) -> list[TrainingTrajectoryRecord]:
    out: list[TrainingTrajectoryRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            out.append(TrainingTrajectoryRecord.from_dict(json.loads(line)))
    return out


def build_manifest(
    path: Path, records: list[TrainingTrajectoryRecord] | None = None
) -> TrainingDatasetManifest:
    path = Path(path)
    if records is None:
        records = read_jsonl(path)
    dataset_id = records[0].dataset_id if records else path.stem
    return TrainingDatasetManifest(
        dataset_id=dataset_id,
        schema_version=TRAINING_DATASET_SCHEMA,
        path=str(path),
        sha256=sha256_file(path),
        record_count=len(records),
        benchmarks=tuple(sorted({r.benchmark for r in records})),
        methods=tuple(sorted({r.method_source for r in records})),
        splits=tuple(sorted({r.split for r in records})),
    )


def write_manifest(manifest: TrainingDatasetManifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def canonical_transition_sha256(record: TrainingTrajectoryRecord) -> str:
    """Hash the source-derived portion of one canonical training transition.

    Collection provenance is intentionally excluded: it describes where the
    transition came from, rather than the transition itself. ``timestamp`` is
    also excluded because it is assigned while materializing JSONL and is not
    present in the runner trajectory. Everything that affects training --
    task, observations, action, reward, outcome, and parsed metadata -- stays
    inside the fingerprint.
    """

    payload = record.to_dict()
    payload.pop("timestamp", None)
    extra = dict(payload.get("extra") or {})
    extra.pop("collection_evidence", None)
    payload["extra"] = extra
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def records_from_trajectory_dict(
    trajectory: dict[str, Any],
    *,
    dataset_id: str,
    split: str = "train",
    action_language: str = "unknown",
    collection_evidence: Mapping[str, Any] | None = None,
    traj_uid_override: str | None = None,
    group_uid_override: str | None = None,
) -> list[TrainingTrajectoryRecord]:
    steps = list(trajectory.get("steps") or [])
    instruction = ""
    for step in steps:
        if step.get("role") == "system":
            instruction = str(step.get("content") or "")
            break

    outcome = trajectory.get("info", {}).get("outcome_evidence", {})
    traj_uid = str(
        traj_uid_override
        or trajectory.get("info", {}).get("traj_uid")
        or trajectory.get("task_id")
        or "trajectory"
    )
    group_uid = str(
        group_uid_override
        or trajectory.get("info", {}).get("group_uid")
        or trajectory.get("task_id")
        or traj_uid
    )
    seed = None
    action_steps = [step for step in steps if step.get("action")]
    trajectory_length = len(action_steps)
    derived_goal_hash = _stable_sha256(
        {"task_id": trajectory.get("task_id") or "", "instruction": instruction}
    )
    derived_rollout_config_hash = _stable_sha256(
        {
            "benchmark": trajectory.get("benchmark") or "",
            "method": trajectory.get("method") or "",
            "model": trajectory.get("model") or "",
            "rollout_config": trajectory.get("info", {}).get("rollout_config") or {},
        }
    )
    records: list[TrainingTrajectoryRecord] = []
    prev_observation = ""
    for step in steps:
        info = step.get("info") or {}
        if seed is None and info.get("seed") is not None:
            seed = int(info["seed"])
        obs = step.get("observation") or {}
        if step.get("action"):
            action = step["action"]
            next_obs = str(obs.get("content") or "")
            obs_info = obs.get("info") or {}
            terminal = bool(obs_info.get("done"))
            success = (
                bool(obs_info.get("success"))
                if terminal
                else bool(trajectory.get("success", False))
            )
            reward = _reward_from_observation(obs, success=success, terminal=terminal)
            raw_step_idx = step.get("step_idx")
            step_idx = int(raw_step_idx) if raw_step_idx is not None else len(records)
            budget_value = obs_info.get("remaining_budget", info.get("remaining_budget"))
            budget_verified = bool(
                obs_info.get(
                    "remaining_budget_verified",
                    info.get("remaining_budget_verified", budget_value is not None),
                )
            )
            remaining_budget = (
                int(budget_value) if budget_verified else max(trajectory_length - step_idx - 1, 0)
            )
            affordances = obs_info.get("action_affordances", info.get("action_affordances"))
            affordance_verified = bool(
                obs_info.get(
                    "action_affordance_verified",
                    info.get("action_affordance_verified", affordances is not None),
                )
            )
            action_raw = str(action.get("raw") or step.get("content") or "")
            token_count = info.get("action_token_count")
            token_count_verified = bool(
                info.get("action_token_count_verified", token_count is not None)
            )
            normalized_token_count = (
                int(token_count) if token_count_verified else max(1, len(action_raw.split()))
            )
            decision_role = str(
                info.get("decision_role") or action.get("decision_role") or "executor"
            )
            terminal_reason = str(
                obs_info.get("terminal_reason")
                or info.get("terminal_reason")
                or (
                    "success"
                    if terminal and success
                    else "environment_done"
                    if terminal
                    else "in_progress"
                )
            )
            explicit_trajectory_length = obs_info.get(
                "trajectory_length", info.get("trajectory_length")
            )
            explicit_goal_hash = obs_info.get("goal_hash", info.get("goal_hash"))
            explicit_affordance_hash = obs_info.get(
                "action_affordance_hash", info.get("action_affordance_hash")
            )
            explicit_rollout_config_hash = obs_info.get(
                "rollout_config_hash", info.get("rollout_config_hash")
            )
            record = TrainingTrajectoryRecord(
                dataset_id=dataset_id,
                benchmark=str(trajectory.get("benchmark") or ""),
                task_id=str(trajectory.get("task_id") or ""),
                split=split,
                seed=seed,
                traj_uid=traj_uid,
                group_uid=group_uid,
                step_idx=step_idx,
                instruction=instruction,
                observation=prev_observation,
                action_raw=action_raw,
                action_type=str(action.get("action_type") or "text"),
                parsed_action=action,
                next_observation=next_obs,
                reward=reward,
                terminal=terminal,
                success=success,
                outcome_evidence=outcome if isinstance(outcome, dict) else {},
                action_language=action_language,
                method_source=str(trajectory.get("method") or ""),
                model_id=str(trajectory.get("model") or ""),
                trajectory_length=(
                    int(explicit_trajectory_length)
                    if explicit_trajectory_length is not None
                    else trajectory_length
                ),
                terminal_reason=terminal_reason,
                decision_role=decision_role,
                state_snapshot_sha256=str(
                    obs_info.get("state_snapshot_sha256") or info.get("state_snapshot_sha256") or ""
                ),
                goal_hash=str(explicit_goal_hash or derived_goal_hash),
                remaining_budget=remaining_budget,
                action_affordance_hash=str(
                    explicit_affordance_hash
                    or _stable_sha256(
                        affordances if affordance_verified else {"unknown_affordances": True}
                    )
                ),
                rollout_config_hash=str(
                    explicit_rollout_config_hash or derived_rollout_config_hash
                ),
                active=not bool(info.get("padding") or info.get("post_terminal")),
                action_token_count=normalized_token_count,
                extra={
                    "step_info": info,
                    "observation_info": obs_info,
                    "is_action_valid": bool(
                        obs_info.get("is_action_valid", info.get("is_action_valid", True))
                    ),
                    "restore_verified": bool(
                        obs_info.get("restore_verified", info.get("restore_verified", False))
                    ),
                    "state_snapshot_artifact_sha256": str(
                        obs_info.get("state_snapshot_artifact_sha256")
                        or info.get("state_snapshot_artifact_sha256")
                        or ""
                    ),
                    "state_snapshot_artifact_path": str(
                        obs_info.get("state_snapshot_artifact_path")
                        or info.get("state_snapshot_artifact_path")
                        or ""
                    ),
                    "restore_probe_receipt_sha256": str(
                        obs_info.get("restore_probe_receipt_sha256")
                        or info.get("restore_probe_receipt_sha256")
                        or ""
                    ),
                    "restore_probe_receipt_path": str(
                        obs_info.get("restore_probe_receipt_path")
                        or info.get("restore_probe_receipt_path")
                        or ""
                    ),
                    "task_manifest_lock_sha256": str(
                        obs_info.get("task_manifest_lock_sha256")
                        or info.get("task_manifest_lock_sha256")
                        or ""
                    ),
                    "remaining_budget_verified": budget_verified,
                    "action_affordance_verified": affordance_verified,
                    "action_token_count_verified": token_count_verified,
                    "goal_hash_verified": explicit_goal_hash is not None,
                    "rollout_config_hash_verified": explicit_rollout_config_hash is not None,
                    "v2_runtime_fields_verified": bool(
                        obs_info.get(
                            "v2_runtime_fields_verified",
                            info.get("v2_runtime_fields_verified", False),
                        )
                    ),
                    "state_snapshot_provenance": "environment"
                    if (obs_info.get("state_snapshot_sha256") or info.get("state_snapshot_sha256"))
                    else "missing",
                },
            )
            if collection_evidence is not None:
                reserved = {
                    "source_task_id",
                    "source_step_idx",
                    "source_transition_sha256",
                }
                overlap = reserved.intersection(collection_evidence)
                if overlap:
                    raise ValueError(
                        "collection_evidence must not predeclare source transition fields: "
                        + ", ".join(sorted(overlap))
                    )
                source_link = {
                    **dict(collection_evidence),
                    "source_task_id": record.task_id,
                    "source_step_idx": record.step_idx,
                    "source_transition_sha256": canonical_transition_sha256(record),
                }
                record = replace(
                    record,
                    extra={**record.extra, "collection_evidence": source_link},
                )
            records.append(record)
        if obs.get("content"):
            prev_observation = str(obs.get("content") or "")
    return records


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reward_from_observation(obs: dict[str, Any], *, success: bool, terminal: bool) -> float:
    info = obs.get("info") or {}
    if info.get("reward") is not None:
        return float(info["reward"])
    if bool(obs.get("is_error")):
        return -1.0
    if terminal:
        return 1.0 if success else -1.0
    return 0.0
