"""Dynamic sampling: reject zero-variance groups and re-collect until informative.

Adapted from DAPO (https://arxiv.org/abs/2503.14476) and G2PO's
``filter_group_data``.  A group whose trajectories all share the same terminal
outcome provides no contrastive credit signal; updating on it wastes compute and
injects noise.  This module filters such groups and optionally re-collects until
a minimum number of informative groups is reached.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    AgenticRolloutBatch,
    OnPolicyActionSample,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord


@dataclass(frozen=True)
class DynamicSamplingConfig:
    enabled: bool = False
    min_informative_groups: int = 1
    max_retries: int = 3
    require_terminal_variation: bool = True


@dataclass(frozen=True)
class DynamicSamplingReceipt:
    enabled: bool
    attempts: int
    groups_collected: int
    groups_informative: int
    groups_filtered: int
    transitions_kept: int
    transitions_discarded: int
    informative_group_uids: tuple[str, ...]
    filtered_group_uids: tuple[str, ...]


def group_outcome_variance(
    records: tuple[TrainingTrajectoryRecord, ...],
) -> dict[str, bool]:
    """Return per-group_uid boolean: True if the group has outcome variation."""
    by_group: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        if row.terminal:
            by_group[row.group_uid].append(row.success)
    result: dict[str, bool] = {}
    for group_uid, outcomes in by_group.items():
        result[group_uid] = len(set(outcomes)) > 1
    for row in records:
        if row.group_uid not in result:
            result[row.group_uid] = False
    return result


def filter_zero_variance_groups(
    rollout: AgenticRolloutBatch,
    *,
    require_terminal_variation: bool = True,
) -> tuple[AgenticRolloutBatch, DynamicSamplingReceipt]:
    """Remove groups where all trajectories share the same terminal outcome.

    Returns the filtered rollout and a receipt describing what was removed.
    If no groups are filtered, the original rollout is returned unchanged.
    """
    variance = group_outcome_variance(rollout.records)
    informative = {uid for uid, has_var in variance.items() if has_var}
    filtered = {uid for uid, has_var in variance.items() if not has_var}

    if not filtered:
        return rollout, DynamicSamplingReceipt(
            enabled=True,
            attempts=1,
            groups_collected=len(variance),
            groups_informative=len(informative),
            groups_filtered=0,
            transitions_kept=len(rollout.records),
            transitions_discarded=0,
            informative_group_uids=tuple(sorted(informative)),
            filtered_group_uids=(),
        )

    kept_records = tuple(
        row for row in rollout.records if row.group_uid in informative
    )
    kept_keys = {(row.traj_uid, row.step_idx) for row in kept_records}
    kept_samples = tuple(
        sample
        for sample in rollout.policy_samples
        if (sample.traj_uid, sample.step_idx) in kept_keys
    )

    filtered_rollout = AgenticRolloutBatch(
        records=kept_records,
        policy_samples=kept_samples,
        task_manifest_sha256=rollout.task_manifest_sha256,
        policy_checkpoint_sha256=rollout.policy_checkpoint_sha256,
    )

    receipt = DynamicSamplingReceipt(
        enabled=True,
        attempts=1,
        groups_collected=len(variance),
        groups_informative=len(informative),
        groups_filtered=len(filtered),
        transitions_kept=len(kept_records),
        transitions_discarded=len(rollout.records) - len(kept_records),
        informative_group_uids=tuple(sorted(informative)),
        filtered_group_uids=tuple(sorted(filtered)),
    )
    return filtered_rollout, receipt


def collect_with_dynamic_sampling(
    *,
    collect_fn,
    config: DynamicSamplingConfig,
    base_seed: int,
    **collect_kwargs: Any,
) -> tuple[AgenticRolloutBatch, DynamicSamplingReceipt]:
    """Collect rollouts, filtering zero-variance groups and retrying if needed.

    ``collect_fn`` must accept a ``seed`` keyword argument and return an
    ``AgenticRolloutBatch``.  On each retry the seed is incremented to produce
    a fresh set of trajectories.

    The final batch is the union of informative groups across all attempts.
    """
    if not config.enabled:
        rollout = collect_fn(seed=base_seed, **collect_kwargs)
        return rollout, DynamicSamplingReceipt(
            enabled=False,
            attempts=1,
            groups_collected=len(
                {row.group_uid for row in rollout.records}
            ),
            groups_informative=len(
                {row.group_uid for row in rollout.records}
            ),
            groups_filtered=0,
            transitions_kept=len(rollout.records),
            transitions_discarded=0,
            informative_group_uids=tuple(
                sorted({row.group_uid for row in rollout.records})
            ),
            filtered_group_uids=(),
        )

    all_informative_records: list[TrainingTrajectoryRecord] = []
    all_informative_samples: list[OnPolicyActionSample] = []
    all_informative_uids: set[str] = set()
    all_filtered_uids: set[str] = set()
    total_groups_collected = 0
    manifest_sha256 = ""
    checkpoint_sha256 = ""

    for attempt in range(config.max_retries):
        seed = base_seed + attempt * 1000
        rollout = collect_fn(seed=seed, **collect_kwargs)
        if not manifest_sha256:
            manifest_sha256 = rollout.task_manifest_sha256
            checkpoint_sha256 = rollout.policy_checkpoint_sha256

        variance = group_outcome_variance(rollout.records)
        total_groups_collected += len(variance)
        informative = {uid for uid, v in variance.items() if v}
        filtered = {uid for uid, v in variance.items() if not v}
        all_filtered_uids |= filtered

        kept_keys: set[tuple[str, int]] = set()
        for row in rollout.records:
            if row.group_uid in informative and row.group_uid not in all_informative_uids:
                all_informative_records.append(row)
                kept_keys.add((row.traj_uid, row.step_idx))
        for sample in rollout.policy_samples:
            if (sample.traj_uid, sample.step_idx) in kept_keys:
                all_informative_samples.append(sample)
        all_informative_uids |= informative

        if len(all_informative_uids) >= config.min_informative_groups:
            break

    if not all_informative_records:
        raise AgenticRLContractError(
            f"dynamic sampling found zero informative groups after "
            f"{config.max_retries} attempts (min_informative_groups="
            f"{config.min_informative_groups})"
        )

    final_rollout = AgenticRolloutBatch(
        records=tuple(all_informative_records),
        policy_samples=tuple(all_informative_samples),
        task_manifest_sha256=manifest_sha256,
        policy_checkpoint_sha256=checkpoint_sha256,
    )
    receipt = DynamicSamplingReceipt(
        enabled=True,
        attempts=attempt + 1,
        groups_collected=total_groups_collected,
        groups_informative=len(all_informative_uids),
        groups_filtered=len(all_filtered_uids - all_informative_uids),
        transitions_kept=len(all_informative_records),
        transitions_discarded=0,
        informative_group_uids=tuple(sorted(all_informative_uids)),
        filtered_group_uids=tuple(sorted(all_filtered_uids - all_informative_uids)),
    )
    return final_rollout, receipt
