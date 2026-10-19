"""Real-ALFWorld infrastructure smoke for selective snapshot certification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lh_platform.alfworld_state import AlfWorldPhysicalCallCounters
from lh_platform.studies.runtime import AlfworldTask
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.trajectory_composition.alfworld_pilot import (
    AlfWorldCompositionPilotError,
    _InstalledPilotBranch,
    _load_task_metadata,
    installed_alfworld_runtime_identity,
)
from lh_platform.training.trajectory_composition.contracts import canonical_sha256
from lh_platform.training.trajectory_composition.source_certification import (
    certify_snapshot_only_actor_transition,
)

SELECTIVE_CERTIFICATION_SMOKE_SCHEMA = "lh_alfworld_selective_certification_smoke/v1"
SELECTIVE_CERTIFICATION_SMOKE_DECISION = "REAL_ALFWORLD_SELECTIVE_CERTIFICATION_INFRASTRUCTURE_PASS"


class SelectiveCertificationSmokeError(RuntimeError):
    """Raised when the real ALFWorld selective-certification smoke is invalid."""


def run_selective_certification_smoke(
    *,
    task: AlfworldTask,
    data_root: str | Path,
    output_dir: str | Path,
    seed: int,
    action: str,
    expected_alfworld_version: str,
    expected_textworld_version: str,
) -> Path:
    """Execute one source action and certify it in an independent fresh branch."""

    if task.split not in {"valid_seen", "valid_unseen"}:
        raise SelectiveCertificationSmokeError(
            "selective-certification smoke requires an evaluation-split task"
        )
    if not action or action != action.strip() or "\n" in action:
        raise SelectiveCertificationSmokeError("smoke action must be canonical")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SelectiveCertificationSmokeError("smoke seed must be non-negative")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = Path(data_root).resolve()
    game_path = (data / task.game_path).resolve()
    trajectory_path = (data / task.trajectory_path).resolve()
    if (
        not game_path.is_relative_to(data)
        or not trajectory_path.is_relative_to(data)
        or not game_path.is_file()
        or not trajectory_path.is_file()
    ):
        raise SelectiveCertificationSmokeError("smoke task artifacts are unavailable")
    metadata = _load_task_metadata(task, data)
    runtime = installed_alfworld_runtime_identity()
    if runtime.version != expected_alfworld_version:
        raise SelectiveCertificationSmokeError("installed ALFWorld version changed")
    if runtime.textworld_version != expected_textworld_version:
        raise SelectiveCertificationSmokeError("installed TextWorld version changed")
    task_instance_sha256 = canonical_sha256(
        {
            "task": asdict(task),
            "game_sha256": _sha256_file(game_path),
            "trajectory_sha256": _sha256_file(trajectory_path),
        }
    )

    source: _InstalledPilotBranch | None = None
    certification_branch: _InstalledPilotBranch | None = None
    try:
        source = _InstalledPilotBranch(
            task=task,
            data_root=data,
            seed=seed,
            output_dir=root / "source",
            expected_alfworld_version=expected_alfworld_version,
        )
        source_observation, _unused_expert_plan, _initial = source.reset()
        if action not in source_observation["admissible_actions"]:
            raise SelectiveCertificationSmokeError(
                f"authorized smoke action is not initially admissible: {action}"
            )
        next_observation, source_evidence = source.step_snapshot_only(action)
        source_calls = _physical_calls(source_evidence["physical_call_cumulative"])
        if (
            source_calls.source_worker_reset_calls != 1
            or source_calls.snapshot_only_worker_step_calls != 1
            or source_calls.logical_semantic_action_calls != 1
            or source_calls.environment_step_calls != 1
        ):
            raise SelectiveCertificationSmokeError(
                "source snapshot-only physical-call accounting changed"
            )
        outcome = source_evidence["environment_outcome"]
        if not isinstance(outcome, dict):
            raise SelectiveCertificationSmokeError("source environment outcome is missing")
        record = TrainingTrajectoryRecord(
            dataset_id="infrastructure:selective-certification-smoke",
            benchmark="alfworld_native",
            task_id=task.task_id,
            split=task.split,
            seed=seed,
            traj_uid=f"selective-certification-smoke:{task.task_id}:{seed}",
            group_uid=task.group_uid,
            step_idx=0,
            instruction="infrastructure-only deterministic semantic action",
            observation=str(source_observation["text"]),
            action_raw=action,
            action_type="controlled_infrastructure_semantic_action",
            parsed_action={"command": action},
            next_observation=str(next_observation["text"]),
            reward=float(outcome["reward"]),
            terminal=bool(outcome["done"]),
            success=bool(outcome["won"]),
            outcome_evidence={
                "won": bool(outcome["won"]),
                "step_won": bool(outcome["won"]),
                "environment_done": bool(outcome["done"]),
            },
            action_language="en",
            method_source="controlled_infrastructure",
            model_id="none",
            trajectory_length=1,
            terminal_reason=("environment_done" if bool(outcome["done"]) else "smoke_cap"),
            state_snapshot_sha256=str(source_evidence["pre_state_snapshot_sha256"]),
            goal_hash=canonical_sha256(metadata),
            remaining_budget=1,
            action_affordance_hash=canonical_sha256(list(source_observation["admissible_actions"])),
            rollout_config_hash=canonical_sha256(
                {
                    "mode": "snapshot_only_then_fresh_branch_certification",
                    "action": action,
                }
            ),
            action_token_count=0,
            extra={
                "transition_evidence_mode": "snapshot_only",
                "restore_verified": False,
                "restore_probe_receipt_sha256": "",
                "restore_probe_receipt_path": "",
                "state_snapshot_artifact_path": source_evidence["pre_state_snapshot_artifact_path"],
                "post_state_snapshot_sha256": source_evidence["post_state_snapshot_sha256"],
                "post_state_snapshot_artifact_path": source_evidence[
                    "post_state_snapshot_artifact_path"
                ],
                "alfworld_physical_call_delta": source_evidence["physical_call_delta"],
                "alfworld_physical_call_cumulative": source_evidence["physical_call_cumulative"],
                "alfworld_trajectory_physical_call_totals": source_evidence[
                    "physical_call_cumulative"
                ],
                "alfworld_task_metadata": metadata,
                "alfworld_facts": list(source_observation["facts"]),
                "alfworld_next_facts": list(next_observation["facts"]),
                "alfworld_next_admissible_actions": list(next_observation["admissible_actions"]),
                "alfworld_environment_outcome": dict(outcome),
                "alfworld_raw_next_observation": next_observation["text"],
                "environment_action": action,
            },
        )

        certification_branch = _InstalledPilotBranch(
            task=task,
            data_root=data,
            seed=seed,
            output_dir=root / "certification",
            expected_alfworld_version=expected_alfworld_version,
        )
        certification_branch.reset()
        certification = certify_snapshot_only_actor_transition(
            record,
            task_instance_sha256=task_instance_sha256,
            artifact_root=root,
            controller=certification_branch.controller,
        )
        certification_calls = _physical_calls(certification.physical_call_cumulative)
        if (
            certification_calls.source_worker_reset_calls != 1
            or certification_calls.state_restore_calls != 3
            or certification_calls.candidate_probe_first_worker_step_calls != 1
            or certification_calls.candidate_probe_replay_worker_step_calls != 1
            or certification_calls.logical_semantic_action_calls != 0
            or certification_calls.environment_reset_calls != 4
            or certification_calls.environment_step_calls != 2
        ):
            raise SelectiveCertificationSmokeError(
                "fresh-branch certification physical-call accounting changed"
            )
        combined = source_calls.plus(certification_calls)
        receipt = {
            "schema": SELECTIVE_CERTIFICATION_SMOKE_SCHEMA,
            "scientific_result": False,
            "claim_scope": (
                "real_alfworld_snapshot_post_identity_and_selective_restore_"
                "certification_infrastructure_only"
            ),
            "decision": SELECTIVE_CERTIFICATION_SMOKE_DECISION,
            "task": asdict(task),
            "seed": seed,
            "authorized_action": action,
            "uses_expert_actions": False,
            "task_instance_sha256": task_instance_sha256,
            "runtime": asdict(runtime),
            "source": {
                "pre_state_snapshot_sha256": source_evidence["pre_state_snapshot_sha256"],
                "post_state_snapshot_sha256": source_evidence["post_state_snapshot_sha256"],
                "environment_outcome_sha256": canonical_sha256(outcome),
                "physical_call_totals": source_calls.to_dict(),
                "physical_call_totals_sha256": source_calls.sha256,
            },
            "certification": certification.to_dict(),
            "certification_sha256": certification.sha256,
            "post_state_identity_verified": certification.post_state_identity_verified,
            "outcome_verified": certification.outcome_verified,
            "restore_verified": certification.restore_verified,
            "combined_physical_call_totals": combined.to_dict(),
            "combined_physical_call_totals_sha256": combined.sha256,
            "execution_counts": {
                "model_samples": 0,
                "logical_semantic_actions": combined.logical_semantic_action_calls,
                "environment_resets": combined.environment_reset_calls,
                "environment_steps": combined.environment_step_calls,
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "training_examples_admitted": 0,
            },
            "training_eligible": False,
            "n1_execution_authorized": False,
            "non_claims": [
                "natural_actor_fragment_support",
                "same_instance_complementarity",
                "e1_junction_transfer",
                "full_parent_replay",
                "physical_call_efficiency_advantage",
                "training_or_policy_improvement",
            ],
        }
        receipt["receipt_content_sha256"] = canonical_sha256(receipt)
        receipt_path = root / "smoke_receipt.json"
        _write_json(receipt_path, receipt)
        return receipt_path
    except AlfWorldCompositionPilotError as error:
        raise SelectiveCertificationSmokeError(str(error)) from error
    finally:
        if certification_branch is not None:
            certification_branch.close()
        if source is not None:
            source.close()


def _physical_calls(value: Any) -> AlfWorldPhysicalCallCounters:
    if not isinstance(value, dict):
        raise SelectiveCertificationSmokeError("physical-call receipt is missing")
    try:
        return AlfWorldPhysicalCallCounters.from_dict(value)
    except Exception as error:
        raise SelectiveCertificationSmokeError(f"invalid physical-call receipt: {error}") from error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SELECTIVE_CERTIFICATION_SMOKE_DECISION",
    "SELECTIVE_CERTIFICATION_SMOKE_SCHEMA",
    "SelectiveCertificationSmokeError",
    "run_selective_certification_smoke",
]
