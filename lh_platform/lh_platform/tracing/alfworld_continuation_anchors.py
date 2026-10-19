"""Source-only replay anchors for natural ALFWorld actor continuations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_action_counterfactual import (
    AlfWorldSourceStep,
    build_alfworld_source_trajectory,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord

ALFWORLD_CONTINUATION_ANCHOR_SCHEMA = "lh_alfworld_continuation_anchor/v1"
ALFWORLD_TASK_DISJOINT_CONTINUATION_ANCHOR_MANIFEST_SCHEMA = (
    "lh_alfworld_task_disjoint_continuation_anchor_manifest/v1"
)
ALFWORLD_CONTINUATION_ANCHOR_SELECTION_RULE = (
    "lexicographically_first_replayable_failed_source_per_task_at_fixed_step"
)

_ACTOR_IDENTITY_FIELDS = frozenset(
    {
        "system_cell_path",
        "system_cell_file_sha256",
        "system_cell_id",
        "system_cell_sha256",
        "parameter_checkpoint_sha256",
        "runtime_policy_identity_sha256",
        "model_io_profile_sha256",
        "actor_protocol_id",
        "response_contract_sha256",
        "chat_template_sha256",
        "actor_history_length",
        "show_admissible_actions",
        "sampling_temperature",
    }
)
_ANCHOR_CASE_FIELDS = frozenset(
    {
        "case_ordinal",
        "task_id",
        "traj_uid",
        "source_path",
        "source_file_sha256",
        "anchor_path",
        "anchor_file_sha256",
        "anchor_sha256",
        "anchor_step_idx",
        "sampling_seed_base",
    }
)


class AlfWorldContinuationAnchorError(ValueError):
    """Raised when a source prefix cannot support exact actor continuation."""


def _raw_sha256(label: str, value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise AlfWorldContinuationAnchorError(f"{label} must be a raw SHA-256")
    return value


@dataclass(frozen=True)
class AlfWorldContinuationAnchorV1:
    """One failed-or-successful source prefix with explicit terminal provenance."""

    anchor_id: str
    task_id: str
    traj_uid: str
    split: str
    environment_seed: int
    task_description: str
    actor_protocol_id: str
    model_id: str
    model_io_profile_sha256: str
    response_contract_sha256: str
    task_manifest_lock_sha256: str
    chat_template_sha256: str
    source_sampling_seed_base: int
    source_sampling_seed_schedule: str
    sampling_temperature: float
    source_trajectory_sha256: str
    source_trajectory_success: bool
    source_terminal_reason: str
    steps: tuple[AlfWorldSourceStep, ...]
    anchor_step_idx: int
    schema: str = ALFWORLD_CONTINUATION_ANCHOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ALFWORLD_CONTINUATION_ANCHOR_SCHEMA or not self.anchor_id:
            raise AlfWorldContinuationAnchorError("continuation anchor identity is invalid")
        if (
            not self.task_id
            or not self.traj_uid
            or not self.source_terminal_reason
            or self.anchor_step_idx < 0
            or self.anchor_step_idx >= len(self.steps) - 1
            or tuple(step.step_idx for step in self.steps) != tuple(range(len(self.steps)))
        ):
            raise AlfWorldContinuationAnchorError("continuation anchor source binding is invalid")
        for label, value in (
            ("model I/O profile", self.model_io_profile_sha256),
            ("response contract", self.response_contract_sha256),
            ("task manifest", self.task_manifest_lock_sha256),
            ("chat template", self.chat_template_sha256),
            ("source trajectory", self.source_trajectory_sha256),
        ):
            _raw_sha256(label, value)
        if not all(step.source_action_admissible for step in self.replayed_steps):
            raise AlfWorldContinuationAnchorError(
                "continuation anchor prefix contains an out-of-menu source action"
            )

    @property
    def replayed_steps(self) -> tuple[AlfWorldSourceStep, ...]:
        return self.steps[: self.anchor_step_idx + 1]

    @property
    def replayed_source_action_count(self) -> int:
        return len(self.replayed_steps)

    @property
    def anchor_source_action(self) -> str:
        return self.steps[self.anchor_step_idx].source_action

    @property
    def anchor_pre_state_snapshot_sha256(self) -> str:
        return self.steps[self.anchor_step_idx].state_snapshot_sha256

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "anchor_id": self.anchor_id,
            "task_id": self.task_id,
            "traj_uid": self.traj_uid,
            "split": self.split,
            "environment_seed": self.environment_seed,
            "task_description": self.task_description,
            "actor_protocol_id": self.actor_protocol_id,
            "model_id": self.model_id,
            "model_io_profile_sha256": self.model_io_profile_sha256,
            "response_contract_sha256": self.response_contract_sha256,
            "task_manifest_lock_sha256": self.task_manifest_lock_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "source_sampling_seed_base": self.source_sampling_seed_base,
            "source_sampling_seed_schedule": self.source_sampling_seed_schedule,
            "sampling_temperature": self.sampling_temperature,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "source_trajectory_success": self.source_trajectory_success,
            "source_terminal_reason": self.source_terminal_reason,
            "source_trajectory_step_count": len(self.steps),
            "steps": [
                {
                    "step_idx": step.step_idx,
                    "source_action": step.source_action,
                    "admissible_actions": list(step.admissible_actions),
                    "source_action_admissible": step.source_action_admissible,
                    "state_snapshot_sha256": step.state_snapshot_sha256,
                    "state_snapshot_artifact_path": step.state_snapshot_artifact_path,
                    "post_state_snapshot_sha256": step.post_state_snapshot_sha256,
                    "post_state_snapshot_artifact_path": step.post_state_snapshot_artifact_path,
                    "observation": step.observation,
                    "next_observation": step.next_observation,
                    "sampling_seed": step.sampling_seed,
                }
                for step in self.steps
            ],
            "anchor_step_idx": self.anchor_step_idx,
            "anchor_source_action": self.anchor_source_action,
            "anchor_pre_state_snapshot_sha256": self.anchor_pre_state_snapshot_sha256,
            "replayed_source_action_count": self.replayed_source_action_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AlfWorldContinuationAnchorV1:
        raw_steps = payload.get("steps")
        if payload.get("schema") != ALFWORLD_CONTINUATION_ANCHOR_SCHEMA or not isinstance(
            raw_steps, list
        ):
            raise AlfWorldContinuationAnchorError("persisted continuation anchor is invalid")
        try:
            steps = tuple(
                AlfWorldSourceStep(
                    step_idx=int(step["step_idx"]),
                    source_action=str(step["source_action"]),
                    admissible_actions=tuple(step["admissible_actions"]),
                    source_action_admissible=bool(step["source_action_admissible"]),
                    state_snapshot_sha256=str(step["state_snapshot_sha256"]),
                    state_snapshot_artifact_path=str(step["state_snapshot_artifact_path"]),
                    post_state_snapshot_sha256=str(step["post_state_snapshot_sha256"]),
                    post_state_snapshot_artifact_path=str(
                        step["post_state_snapshot_artifact_path"]
                    ),
                    observation=str(step["observation"]),
                    next_observation=str(step["next_observation"]),
                    sampling_seed=int(step["sampling_seed"]),
                )
                for step in raw_steps
                if isinstance(step, Mapping)
            )
            source_success = payload["source_trajectory_success"]
            if not isinstance(source_success, bool):
                raise TypeError("source_trajectory_success must be boolean")
            anchor = cls(
                schema=str(payload["schema"]),
                anchor_id=str(payload["anchor_id"]),
                task_id=str(payload["task_id"]),
                traj_uid=str(payload["traj_uid"]),
                split=str(payload["split"]),
                environment_seed=int(payload["environment_seed"]),
                task_description=str(payload["task_description"]),
                actor_protocol_id=str(payload["actor_protocol_id"]),
                model_id=str(payload["model_id"]),
                model_io_profile_sha256=str(payload["model_io_profile_sha256"]),
                response_contract_sha256=str(payload["response_contract_sha256"]),
                task_manifest_lock_sha256=str(payload["task_manifest_lock_sha256"]),
                chat_template_sha256=str(payload["chat_template_sha256"]),
                source_sampling_seed_base=int(payload["source_sampling_seed_base"]),
                source_sampling_seed_schedule=str(payload["source_sampling_seed_schedule"]),
                sampling_temperature=float(payload["sampling_temperature"]),
                source_trajectory_sha256=str(payload["source_trajectory_sha256"]),
                source_trajectory_success=source_success,
                source_terminal_reason=str(payload["source_terminal_reason"]),
                steps=steps,
                anchor_step_idx=int(payload["anchor_step_idx"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlfWorldContinuationAnchorError(
                f"persisted continuation anchor is invalid: {error}"
            ) from error
        if len(steps) != len(raw_steps):
            raise AlfWorldContinuationAnchorError("persisted continuation steps are invalid")
        computed = anchor.to_dict()
        for field in (
            "source_trajectory_step_count",
            "anchor_source_action",
            "anchor_pre_state_snapshot_sha256",
            "replayed_source_action_count",
        ):
            if payload.get(field) != computed[field]:
                raise AlfWorldContinuationAnchorError(
                    f"persisted continuation anchor derived field changed: {field}"
                )
        return anchor


@dataclass(frozen=True)
class AlfWorldContinuationAnchorSourceV1:
    """One immutable trajectory file offered to the source-only selector."""

    source_path: str
    source_file_sha256: str
    records: tuple[TrainingTrajectoryRecord, ...]

    def __post_init__(self) -> None:
        if not self.source_path or not self.records:
            raise AlfWorldContinuationAnchorError("continuation anchor source is incomplete")
        _raw_sha256("continuation anchor source file", self.source_file_sha256)


@dataclass(frozen=True)
class AlfWorldContinuationAnchorSelectionV1:
    """Selected source file and its one target-blind continuation anchor."""

    source_path: str
    source_file_sha256: str
    anchor: AlfWorldContinuationAnchorV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_file_sha256": self.source_file_sha256,
            "anchor_id": self.anchor.anchor_id,
            "anchor_sha256": self.anchor.sha256,
            "task_id": self.anchor.task_id,
            "traj_uid": self.anchor.traj_uid,
            "anchor_step_idx": self.anchor.anchor_step_idx,
        }


@dataclass(frozen=True)
class AlfWorldContinuationAnchorArtifactV1:
    """Materialized selected anchor before actor generation."""

    selection: AlfWorldContinuationAnchorSelectionV1
    anchor_path: str
    anchor_file_sha256: str

    def __post_init__(self) -> None:
        if not self.anchor_path:
            raise AlfWorldContinuationAnchorError("continuation anchor path is empty")
        _raw_sha256("continuation anchor file", self.anchor_file_sha256)


def build_alfworld_continuation_anchor(
    records: Sequence[TrainingTrajectoryRecord],
    *,
    anchor_step_idx: int,
) -> AlfWorldContinuationAnchorV1:
    """Freeze one source prefix without requiring terminal source success."""

    source = build_alfworld_source_trajectory(records)
    if anchor_step_idx < 0 or anchor_step_idx >= len(source.steps) - 1:
        raise AlfWorldContinuationAnchorError(
            "continuation anchor must be nonterminal and inside the source trace"
        )
    identity = {
        "source_trajectory_sha256": source.source_trajectory_sha256,
        "anchor_step_idx": anchor_step_idx,
        "anchor_source_action": source.steps[anchor_step_idx].source_action,
        "anchor_pre_state_snapshot_sha256": source.steps[anchor_step_idx].state_snapshot_sha256,
    }
    return AlfWorldContinuationAnchorV1(
        anchor_id=f"step-{anchor_step_idx:04d}-{canonical_sha256(identity)[:16]}",
        task_id=source.task_id,
        traj_uid=source.traj_uid,
        split=source.split,
        environment_seed=source.environment_seed,
        task_description=source.task_description,
        actor_protocol_id=source.actor_protocol_id,
        model_id=source.model_id,
        model_io_profile_sha256=source.model_io_profile_sha256,
        response_contract_sha256=source.response_contract_sha256,
        task_manifest_lock_sha256=source.task_manifest_lock_sha256,
        chat_template_sha256=source.chat_template_sha256,
        source_sampling_seed_base=source.sampling_seed_base,
        source_sampling_seed_schedule=source.sampling_seed_schedule,
        sampling_temperature=source.sampling_temperature,
        source_trajectory_sha256=source.source_trajectory_sha256,
        source_trajectory_success=source.success,
        source_terminal_reason=source.terminal_reason,
        steps=source.steps,
        anchor_step_idx=anchor_step_idx,
    )


def select_alfworld_failed_source_continuation_anchors(
    sources: Iterable[AlfWorldContinuationAnchorSourceV1],
    *,
    excluded_task_ids: Sequence[str],
    anchor_step_idx: int,
    minimum_task_count: int,
) -> tuple[AlfWorldContinuationAnchorSelectionV1, ...]:
    """Select the first replayable failed source per non-excluded task."""

    excluded = tuple(excluded_task_ids)
    if (
        tuple(sorted(set(excluded))) != excluded
        or any(not task_id for task_id in excluded)
        or isinstance(anchor_step_idx, bool)
        or anchor_step_idx < 0
        or isinstance(minimum_task_count, bool)
        or minimum_task_count < 1
    ):
        raise AlfWorldContinuationAnchorError("continuation anchor selection contract is invalid")
    excluded_set = frozenset(excluded)
    eligible: list[
        tuple[str, AlfWorldContinuationAnchorSourceV1, AlfWorldContinuationAnchorV1]
    ] = []
    for source_input in sources:
        source = build_alfworld_source_trajectory(source_input.records)
        if source.task_id in excluded_set or source.success:
            continue
        if anchor_step_idx >= len(source.steps) - 1 or not all(
            step.source_action_admissible for step in source.steps[: anchor_step_idx + 1]
        ):
            continue
        eligible.append(
            (
                source.task_id,
                source_input,
                build_alfworld_continuation_anchor(
                    source_input.records,
                    anchor_step_idx=anchor_step_idx,
                ),
            )
        )

    selected: dict[
        str, tuple[AlfWorldContinuationAnchorSourceV1, AlfWorldContinuationAnchorV1]
    ] = {}
    for task_id, source_input, anchor in sorted(
        eligible,
        key=lambda item: (item[0], item[1].source_path),
    ):
        selected.setdefault(task_id, (source_input, anchor))
    if len(selected) < minimum_task_count:
        raise AlfWorldContinuationAnchorError(
            "replayable failed-source task coverage is below the frozen minimum"
        )
    return tuple(
        AlfWorldContinuationAnchorSelectionV1(
            source_path=source_input.source_path,
            source_file_sha256=source_input.source_file_sha256,
            anchor=anchor,
        )
        for _, (source_input, anchor) in sorted(selected.items())
    )


def _validate_actor_identity(actor_identity: Mapping[str, Any]) -> None:
    if set(actor_identity) != _ACTOR_IDENTITY_FIELDS:
        raise AlfWorldContinuationAnchorError("continuation actor identity is incomplete")
    for field in (
        "system_cell_file_sha256",
        "system_cell_sha256",
        "parameter_checkpoint_sha256",
        "runtime_policy_identity_sha256",
        "model_io_profile_sha256",
        "response_contract_sha256",
        "chat_template_sha256",
    ):
        _raw_sha256(f"continuation actor {field}", str(actor_identity.get(field, "")))
    if (
        not isinstance(actor_identity.get("system_cell_path"), str)
        or not actor_identity["system_cell_path"]
        or not isinstance(actor_identity.get("system_cell_id"), str)
        or not actor_identity["system_cell_id"]
        or not isinstance(actor_identity.get("actor_protocol_id"), str)
        or not actor_identity["actor_protocol_id"]
        or actor_identity.get("actor_history_length") != 8
        or actor_identity.get("show_admissible_actions") is not True
        or actor_identity.get("sampling_temperature") != 0.4
    ):
        raise AlfWorldContinuationAnchorError("continuation actor contract changed")


def build_alfworld_task_disjoint_continuation_anchor_manifest(
    artifacts: Sequence[AlfWorldContinuationAnchorArtifactV1],
    *,
    task_manifest_path: str,
    source_collection_result_file_sha256: str,
    task_manifest_file_sha256: str,
    source_task_count: int,
    excluded_prior_task_ids: Sequence[str],
    actor_identity: Mapping[str, Any],
    anchor_step_idx: int,
    sampling_seed_base: int,
    max_episode_steps: int,
    minimum_independent_tasks_for_replay_spend: int,
) -> dict[str, Any]:
    """Freeze one source-only actor opportunity per independent task."""

    _raw_sha256(
        "source collection result file",
        source_collection_result_file_sha256,
    )
    _raw_sha256("task manifest file", task_manifest_file_sha256)
    _validate_actor_identity(actor_identity)
    excluded = tuple(excluded_prior_task_ids)
    if (
        not task_manifest_path
        or tuple(sorted(set(excluded))) != excluded
        or isinstance(source_task_count, bool)
        or source_task_count < 1
        or isinstance(anchor_step_idx, bool)
        or anchor_step_idx < 0
        or isinstance(sampling_seed_base, bool)
        or sampling_seed_base < 0
        or isinstance(max_episode_steps, bool)
        or max_episode_steps <= anchor_step_idx + 1
        or isinstance(minimum_independent_tasks_for_replay_spend, bool)
        or minimum_independent_tasks_for_replay_spend < 1
    ):
        raise AlfWorldContinuationAnchorError("continuation manifest shape is invalid")
    ordered = tuple(sorted(artifacts, key=lambda item: item.selection.anchor.task_id))
    task_ids = tuple(item.selection.anchor.task_id for item in ordered)
    if (
        not ordered
        or len(ordered) < minimum_independent_tasks_for_replay_spend
        or len(task_ids) != len(set(task_ids))
        or set(task_ids).intersection(excluded)
        or len(task_ids) > source_task_count
    ):
        raise AlfWorldContinuationAnchorError(
            "continuation population is not independently covered"
        )
    cases = []
    for ordinal, artifact in enumerate(ordered):
        selection = artifact.selection
        anchor = selection.anchor
        if (
            anchor.source_trajectory_success is not False
            or anchor.anchor_step_idx != anchor_step_idx
            or anchor.actor_protocol_id != actor_identity["actor_protocol_id"]
            or anchor.model_io_profile_sha256 != actor_identity["model_io_profile_sha256"]
            or anchor.response_contract_sha256 != actor_identity["response_contract_sha256"]
            or anchor.chat_template_sha256 != actor_identity["chat_template_sha256"]
            or anchor.sampling_temperature != actor_identity["sampling_temperature"]
        ):
            raise AlfWorldContinuationAnchorError(
                "continuation anchor differs from the frozen actor contract"
            )
        cases.append(
            {
                "case_ordinal": ordinal,
                "task_id": anchor.task_id,
                "traj_uid": anchor.traj_uid,
                "source_path": selection.source_path,
                "source_file_sha256": selection.source_file_sha256,
                "anchor_path": artifact.anchor_path,
                "anchor_file_sha256": artifact.anchor_file_sha256,
                "anchor_sha256": anchor.sha256,
                "anchor_step_idx": anchor.anchor_step_idx,
                "sampling_seed_base": sampling_seed_base + ordinal * max_episode_steps,
            }
        )
    manifest: dict[str, Any] = {
        "schema": ALFWORLD_TASK_DISJOINT_CONTINUATION_ANCHOR_MANIFEST_SCHEMA,
        "status": "frozen_before_actor_generation",
        "source_collection_result_file_sha256": (source_collection_result_file_sha256),
        "task_manifest_path": task_manifest_path,
        "task_manifest_file_sha256": task_manifest_file_sha256,
        "source_task_count": source_task_count,
        "excluded_prior_task_ids": list(excluded),
        "excluded_prior_task_count": len(excluded),
        "selection_rule": ALFWORLD_CONTINUATION_ANCHOR_SELECTION_RULE,
        "selection_uses_new_actor_outcomes": False,
        "new_actor_outcome_fields_present": False,
        "anchor_step_idx": anchor_step_idx,
        "selected_case_count": len(cases),
        "selected_task_count": len(cases),
        "opportunity_count": len(cases),
        "sampling_seed_base": sampling_seed_base,
        "sampling_seed_schedule": "base_plus_task_ordinal_times_horizon",
        "max_episode_steps": max_episode_steps,
        "stop_rule": "first_actor_proposal_outside_its_live_menu",
        "execute_after_first_out_of_menu_proposal": False,
        "minimum_independent_tasks_for_replay_spend": (minimum_independent_tasks_for_replay_spend),
        "actor_identity": dict(actor_identity),
        "cases": cases,
    }
    manifest["content_sha256"] = "sha256:" + canonical_sha256(manifest)
    return manifest


def validate_alfworld_task_disjoint_continuation_anchor_manifest(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Validate the complete pre-generation population and seed schedule."""

    canonical = dict(manifest)
    claimed_content_sha256 = canonical.pop("content_sha256", None)
    raw_cases = manifest.get("cases")
    raw_excluded = manifest.get("excluded_prior_task_ids")
    raw_actor = manifest.get("actor_identity")
    if (
        manifest.get("schema") != ALFWORLD_TASK_DISJOINT_CONTINUATION_ANCHOR_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen_before_actor_generation"
        or claimed_content_sha256 != "sha256:" + canonical_sha256(canonical)
        or manifest.get("selection_rule") != ALFWORLD_CONTINUATION_ANCHOR_SELECTION_RULE
        or manifest.get("selection_uses_new_actor_outcomes") is not False
        or manifest.get("new_actor_outcome_fields_present") is not False
        or manifest.get("sampling_seed_schedule") != "base_plus_task_ordinal_times_horizon"
        or manifest.get("stop_rule") != "first_actor_proposal_outside_its_live_menu"
        or manifest.get("execute_after_first_out_of_menu_proposal") is not False
        or not isinstance(raw_cases, list)
        or not isinstance(raw_excluded, list)
        or not isinstance(raw_actor, Mapping)
        or not isinstance(manifest.get("task_manifest_path"), str)
        or not manifest.get("task_manifest_path")
    ):
        raise AlfWorldContinuationAnchorError("continuation manifest contract changed")
    _validate_actor_identity(raw_actor)
    _raw_sha256(
        "source collection result file",
        str(manifest.get("source_collection_result_file_sha256", "")),
    )
    _raw_sha256(
        "task manifest file",
        str(manifest.get("task_manifest_file_sha256", "")),
    )
    integer_fields = (
        "source_task_count",
        "excluded_prior_task_count",
        "anchor_step_idx",
        "selected_case_count",
        "selected_task_count",
        "opportunity_count",
        "sampling_seed_base",
        "max_episode_steps",
        "minimum_independent_tasks_for_replay_spend",
    )
    if any(
        isinstance(manifest.get(field), bool)
        or not isinstance(manifest.get(field), int)
        or int(manifest[field]) < 1
        for field in integer_fields
    ):
        raise AlfWorldContinuationAnchorError("continuation manifest dimensions changed")
    excluded = tuple(raw_excluded)
    if tuple(sorted(set(excluded))) != excluded or manifest["excluded_prior_task_count"] != len(
        excluded
    ):
        raise AlfWorldContinuationAnchorError("continuation exclusions changed")
    anchor_step_idx = int(manifest["anchor_step_idx"])
    max_episode_steps = int(manifest["max_episode_steps"])
    seed_base = int(manifest["sampling_seed_base"])
    cases: list[Mapping[str, Any]] = []
    for ordinal, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping) or set(raw_case) != _ANCHOR_CASE_FIELDS:
            raise AlfWorldContinuationAnchorError("continuation case contains future actor fields")
        if (
            raw_case.get("case_ordinal") != ordinal
            or raw_case.get("anchor_step_idx") != anchor_step_idx
            or raw_case.get("sampling_seed_base") != seed_base + ordinal * max_episode_steps
            or any(
                not isinstance(raw_case.get(field), str) or not raw_case.get(field)
                for field in (
                    "task_id",
                    "traj_uid",
                    "source_path",
                    "anchor_path",
                )
            )
        ):
            raise AlfWorldContinuationAnchorError("continuation case identity changed")
        for field in (
            "source_file_sha256",
            "anchor_file_sha256",
        ):
            _raw_sha256(f"continuation case {field}", str(raw_case.get(field, "")))
        anchor_sha256 = raw_case.get("anchor_sha256")
        if not isinstance(anchor_sha256, str) or not anchor_sha256.startswith("sha256:"):
            raise AlfWorldContinuationAnchorError("continuation anchor SHA-256 changed")
        _raw_sha256("continuation anchor", anchor_sha256.removeprefix("sha256:"))
        cases.append(raw_case)
    task_ids = tuple(str(case["task_id"]) for case in cases)
    if (
        len(cases) != manifest["selected_case_count"]
        or len(cases) != manifest["selected_task_count"]
        or len(cases) != manifest["opportunity_count"]
        or len(cases) < manifest["minimum_independent_tasks_for_replay_spend"]
        or len(task_ids) != len(set(task_ids))
        or set(task_ids).intersection(excluded)
        or len(task_ids) > manifest["source_task_count"]
    ):
        raise AlfWorldContinuationAnchorError(
            "continuation population is not independently covered"
        )
    return tuple(cases)


__all__ = [
    "ALFWORLD_CONTINUATION_ANCHOR_SELECTION_RULE",
    "ALFWORLD_CONTINUATION_ANCHOR_SCHEMA",
    "ALFWORLD_TASK_DISJOINT_CONTINUATION_ANCHOR_MANIFEST_SCHEMA",
    "AlfWorldContinuationAnchorArtifactV1",
    "AlfWorldContinuationAnchorError",
    "AlfWorldContinuationAnchorSelectionV1",
    "AlfWorldContinuationAnchorSourceV1",
    "AlfWorldContinuationAnchorV1",
    "build_alfworld_continuation_anchor",
    "build_alfworld_task_disjoint_continuation_anchor_manifest",
    "select_alfworld_failed_source_continuation_anchors",
    "validate_alfworld_task_disjoint_continuation_anchor_manifest",
]
