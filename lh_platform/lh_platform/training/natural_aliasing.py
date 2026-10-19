"""Materialize paired-prefix natural aliasing rows from frozen collections."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.training.dataset import sha256_file

LEGACY_NATURAL_ROWS_SCHEMA = "lh_natural_aliasing_horizon_rows/v1"
NATURAL_ROWS_SCHEMA = "lh_natural_aliasing_horizon_rows/v2"
COLLECTION_SCHEMA = "lh_two_dataset_credit_collection/v1"


class NaturalAliasingError(ValueError):
    """Raised when a frozen collection cannot support paired-prefix analysis."""


def materialize_natural_aliasing_rows(
    collection_result_path: str | Path,
    *,
    horizons: Iterable[int],
    output_path: str | Path,
) -> dict[str, Any]:
    """Compute exposure-normalized aliasing burden at paired trajectory prefixes.

    A strict context is exposed when the same official anchor, raw observation,
    verified affordance set and original-rollout remaining budget are visited
    by different trajectories with different verifier-derived residual state.
    Contexts, rather than transitions, are the counting unit.
    """

    source = Path(collection_result_path).resolve()
    collection = _load_mapping(source)
    if collection.get("schema") != COLLECTION_SCHEMA:
        raise NaturalAliasingError("unsupported frozen collection schema")
    if collection.get("policy_update_executed") is not False:
        raise NaturalAliasingError("natural aliasing rows require a frozen policy")

    actor_screen_path = Path(str(collection.get("actor_competence_screen_path", "")))
    if not actor_screen_path.is_file():
        raise NaturalAliasingError("collection lacks actor competence screen")
    expected_screen_sha = str(collection.get("actor_competence_screen_sha256", ""))
    if sha256_file(actor_screen_path) != expected_screen_sha:
        raise NaturalAliasingError("actor competence screen hash mismatch")
    actor_screen = _load_mapping(actor_screen_path)
    if actor_screen.get("lineage_verified") is not True:
        raise NaturalAliasingError("actor collection lineage is not verified")
    actor_competence = _actor_competence_by_benchmark(actor_screen)

    prefix_horizons = tuple(sorted(set(int(value) for value in horizons)))
    if len(prefix_horizons) < 3 or prefix_horizons[0] <= 0:
        raise NaturalAliasingError("paired-prefix analysis requires three positive horizons")
    collection_config = collection.get("config")
    if not isinstance(collection_config, Mapping):
        raise NaturalAliasingError("collection lacks rollout configuration")
    source_rollout_horizon = int(collection_config.get("max_episode_steps", -1))
    if source_rollout_horizon < prefix_horizons[-1]:
        raise NaturalAliasingError("paired prefixes exceed the frozen source rollout horizon")

    datasets = collection.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise NaturalAliasingError("collection has no datasets")

    records_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    artifact_bindings = []
    for benchmark, receipt in sorted(datasets.items()):
        if not isinstance(receipt, Mapping):
            raise NaturalAliasingError("dataset receipt is invalid")
        records = []
        for split in ("source", "target"):
            binding = receipt.get(f"{split}_artifact")
            if not isinstance(binding, Mapping):
                raise NaturalAliasingError(f"{benchmark} lacks {split} artifact")
            path = Path(str(binding.get("path", ""))).resolve()
            expected_sha = str(binding.get("sha256", ""))
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise NaturalAliasingError(f"{benchmark} {split} artifact hash mismatch")
            loaded = _load_annotated_records(path)
            records.extend(loaded)
            artifact_bindings.append(
                {
                    "benchmark": str(benchmark),
                    "split": split,
                    "path": str(path),
                    "sha256": expected_sha,
                    "record_count": len(loaded),
                }
            )
        records_by_benchmark[str(benchmark)] = records

    rows = []
    summaries = []
    for benchmark, records in sorted(records_by_benchmark.items()):
        by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_trajectory[str(record["traj_uid"])].append(record)
        for trajectory in by_trajectory.values():
            trajectory.sort(key=lambda item: int(item["step_idx"]))

        for horizon in prefix_horizons:
            prefix = [
                record
                for trajectory in by_trajectory.values()
                for record in trajectory
                if int(record["step_idx"]) < horizon
            ]
            strict_contexts: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
            for record in prefix:
                strict_contexts[_strict_context_identity(benchmark, record)].append(record)
            alias_detection_steps = {
                identity: detection_step
                for identity, members in strict_contexts.items()
                if (detection_step := _first_alias_detection_step(members)) is not None
            }
            aliased_contexts = set(alias_detection_steps)
            snapshot_verdict = _snapshot_consistency_verdict(strict_contexts)
            repeat_contexts = {
                identity
                for identity, members in strict_contexts.items()
                if len({str(member["traj_uid"]) for member in members}) > 1
            }
            at_risk_count = 0
            for trajectory_id, trajectory in sorted(by_trajectory.items()):
                members = [record for record in trajectory if int(record["step_idx"]) < horizon]
                if not members:
                    continue
                observed_length = int(trajectory[-1]["step_idx"]) + 1
                at_risk = observed_length >= horizon
                at_risk_count += int(at_risk)
                trajectory_contexts = {
                    _strict_context_identity(benchmark, record) for record in members
                }
                trajectory_first_visit = {
                    identity: min(
                        int(record["step_idx"]) + 1
                        for record in members
                        if _strict_context_identity(benchmark, record) == identity
                    )
                    for identity in trajectory_contexts
                }
                trajectory_aliased = trajectory_contexts & aliased_contexts
                trajectory_repeated = trajectory_contexts & repeat_contexts
                exposed = len(trajectory_aliased)
                repeat_opportunities = len(trajectory_repeated)
                aliased_steps = [
                    event_step
                    for identity in trajectory_aliased
                    if (
                        event_step := max(
                            trajectory_first_visit[identity],
                            alias_detection_steps[identity],
                        )
                    )
                    <= observed_length
                ]
                rows.append(
                    {
                        "benchmark": benchmark,
                        "task_id": f"{benchmark}:{members[0]['task_id']}",
                        "trajectory_id": f"{benchmark}:{trajectory_id}",
                        "horizon": horizon,
                        "aliasing_burden": (
                            exposed / repeat_opportunities if repeat_opportunities else 0.0
                        ),
                        "aliasing_exposure_rate": exposed / len(trajectory_contexts),
                        "aliased_context_count": exposed,
                        "strict_context_count": len(trajectory_contexts),
                        "observed_transition_count": len(members),
                        "first_aliased_step": min(aliased_steps) if aliased_steps else None,
                        "strict_context_ids": sorted(
                            canonical_sha256(list(identity)) for identity in trajectory_contexts
                        ),
                        "repeat_context_ids": sorted(
                            canonical_sha256(list(identity)) for identity in trajectory_repeated
                        ),
                        "aliased_context_ids": sorted(
                            canonical_sha256(list(identity)) for identity in trajectory_aliased
                        ),
                        "repeat_opportunity_count": repeat_opportunities,
                        "conditional_aliasing_burden": (
                            exposed / repeat_opportunities if repeat_opportunities else 0.0
                        ),
                        "at_risk_at_horizon": at_risk,
                        "terminal_censored_before_horizon": not at_risk,
                    }
                )
            summaries.append(
                {
                    "benchmark": benchmark,
                    "horizon": horizon,
                    "trajectory_count": len(by_trajectory),
                    "at_risk_trajectory_count": at_risk_count,
                    "terminal_censored_trajectory_count": (len(by_trajectory) - at_risk_count),
                    "strict_context_count": len(strict_contexts),
                    "repeat_opportunity_context_count": len(repeat_contexts),
                    "collision_context_count": len(aliased_contexts),
                    "collision_context_rate": (
                        len(aliased_contexts) / len(strict_contexts) if strict_contexts else 0.0
                    ),
                    "transition_count": len(prefix),
                    "aliased_context_count": len(aliased_contexts),
                    "repeat_opportunity_count": len(repeat_contexts),
                    "aliasing_exposure_rate": (
                        len(aliased_contexts) / len(strict_contexts) if strict_contexts else 0.0
                    ),
                    "conditional_aliasing_burden": (
                        len(aliased_contexts) / len(repeat_contexts) if repeat_contexts else 0.0
                    ),
                    "snapshot_consistency": snapshot_verdict,
                }
            )

    result = {
        "schema": NATURAL_ROWS_SCHEMA,
        "claim_scope": (
            "frozen_policy_paired_prefix_natural_aliasing_prevalence_not_outcome_or_method_effect"
        ),
        "metric_definition": (
            "fraction_of_repeated_strict_contexts_whose_cross_trajectory_visits_"
            "contain_multiple_verifier_residual_frontier_identities"
        ),
        "collection_binding": {
            "path": str(source),
            "sha256": sha256_file(source),
            "actor_competence_screen_path": str(actor_screen_path),
            "actor_competence_screen_sha256": expected_screen_sha,
            "model_checkpoint_sha256": collection.get("model_checkpoint_sha256"),
            "source_rollout_horizon": source_rollout_horizon,
            "actor_competence_by_benchmark": actor_competence,
        },
        "artifact_bindings": artifact_bindings,
        "horizons": list(prefix_horizons),
        "summaries": summaries,
        "rows": rows,
    }
    atomic_write_json(output_path, result)
    return result


def _actor_competence_by_benchmark(
    actor_screen: Mapping[str, Any],
) -> dict[str, dict[str, float | int]]:
    benchmarks = actor_screen.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or not benchmarks:
        raise NaturalAliasingError("actor competence screen lacks benchmark outcome counts")
    output: dict[str, dict[str, float | int]] = {}
    for benchmark, splits in benchmarks.items():
        if not isinstance(splits, Mapping) or not splits:
            raise NaturalAliasingError("actor competence benchmark is malformed")
        successes = 0
        trajectories = 0
        for split in splits.values():
            if not isinstance(split, Mapping):
                raise NaturalAliasingError("actor competence split is malformed")
            successes += int(split.get("successful_trajectory_count", -1))
            trajectories += int(split.get("trajectory_count", -1))
        if successes < 0 or trajectories <= 0 or successes > trajectories:
            raise NaturalAliasingError("actor competence outcome counts are invalid")
        output[str(benchmark)] = {
            "successful_trajectory_count": successes,
            "trajectory_count": trajectories,
            "success_rate": successes / trajectories,
        }
    return output


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise NaturalAliasingError(f"artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise NaturalAliasingError(f"artifact is not a mapping: {path}")
    return value


def _load_annotated_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        record = payload.get("record") if isinstance(payload, Mapping) else None
        if not isinstance(record, Mapping):
            raise NaturalAliasingError(f"invalid annotated record at {path}:{line_number}")
        extra = record.get("extra")
        if not isinstance(extra, Mapping):
            raise NaturalAliasingError(f"record lacks verifier metadata at {path}:{line_number}")
        if extra.get("goal_annotation_verified") is not True:
            raise NaturalAliasingError(f"unverified goal annotation at {path}:{line_number}")
        if extra.get("remaining_budget_verified") is not True:
            raise NaturalAliasingError(f"unverified remaining budget at {path}:{line_number}")
        records.append(dict(record))
    if not records:
        raise NaturalAliasingError(f"annotated artifact is empty: {path}")
    return records


def _strict_context_identity(benchmark: str, record: Mapping[str, Any]) -> tuple[object, ...]:
    extra = record["extra"]
    raw_observation = (
        extra.get("alfworld_raw_observation")
        if benchmark == "alfworld"
        else extra.get("raw_observation", record.get("observation"))
    )
    if not isinstance(raw_observation, str) or not raw_observation:
        raise NaturalAliasingError("record lacks actor-visible raw observation")
    affordance_hash = str(record.get("action_affordance_hash", ""))
    remaining_budget = record.get("remaining_budget")
    if (
        extra.get("action_affordance_verified") is not True
        or len(affordance_hash) != 64
        or isinstance(remaining_budget, bool)
        or not isinstance(remaining_budget, int)
        or remaining_budget < 0
    ):
        raise NaturalAliasingError(
            "strict context requires verified affordance and remaining budget"
        )
    return (
        benchmark,
        str(record["group_uid"]),
        str(record["observation"]),
        raw_observation,
        affordance_hash,
        remaining_budget,
    )


def _fine_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    extra = record["extra"]
    residual = str(extra.get("residual_goal_signature", ""))
    frontier = str(extra.get("frontier_signature", ""))
    if not residual or not frontier:
        raise NaturalAliasingError("record lacks residual/frontier identity")
    return residual, frontier


def _first_alias_detection_step(
    members: Iterable[Mapping[str, Any]],
) -> int | None:
    """Return the first prefix where a cross-trajectory alias is observable.

    An early visit is not retrospectively marked as an event merely because a
    later trajectory eventually reaches the same strict context with a
    different residual state.  The event occurs only when both incompatible
    visits are present in the frozen prefix.

    Same snapshot + different fine identity is verifier inconsistency, not
    aliasing — those pairs are excluded.
    """

    seen: list[tuple[str, tuple[str, str], str]] = []
    ordered = sorted(members, key=lambda member: int(member["step_idx"]))
    for member in ordered:
        trajectory_id = str(member["traj_uid"])
        fine_identity = _fine_identity(member)
        snapshot = str(member.get("state_snapshot_sha256", ""))
        if any(
            prior_trajectory != trajectory_id
            and prior_fine != fine_identity
            and not (prior_snapshot and prior_snapshot == snapshot)
            for prior_trajectory, prior_fine, prior_snapshot in seen
        ):
            return int(member["step_idx"]) + 1
        seen.append((trajectory_id, fine_identity, snapshot))
    return None


def _snapshot_consistency_verdict(
    members_by_context: Mapping[tuple[object, ...], list[Mapping[str, Any]]],
) -> dict[str, int]:
    """Classify cross-trajectory context visits by snapshot/label agreement."""

    same_snapshot_same_label = 0
    same_snapshot_different_label = 0
    different_snapshot_same_label = 0
    different_snapshot_different_label = 0
    for members in members_by_context.values():
        trajectories = {str(m["traj_uid"]) for m in members}
        if len(trajectories) < 2:
            continue
        ordered = sorted(members, key=lambda m: (str(m["traj_uid"]), int(m["step_idx"])))
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                if str(left["traj_uid"]) == str(right["traj_uid"]):
                    continue
                left_fine = _fine_identity(left)
                right_fine = _fine_identity(right)
                left_snap = str(left.get("state_snapshot_sha256", ""))
                right_snap = str(right.get("state_snapshot_sha256", ""))
                same_snap = left_snap == right_snap and left_snap != ""
                same_label = left_fine == right_fine
                if same_snap and same_label:
                    same_snapshot_same_label += 1
                elif same_snap and not same_label:
                    same_snapshot_different_label += 1
                elif not same_snap and same_label:
                    different_snapshot_same_label += 1
                else:
                    different_snapshot_different_label += 1
    return {
        "same_snapshot_same_label": same_snapshot_same_label,
        "same_snapshot_different_label": same_snapshot_different_label,
        "different_snapshot_same_label": different_snapshot_same_label,
        "different_snapshot_different_label": different_snapshot_different_label,
        "verifier_inconsistency_count": same_snapshot_different_label,
        "aliasing_candidate_count": different_snapshot_different_label,
    }
