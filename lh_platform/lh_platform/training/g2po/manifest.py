"""Frozen ALFWorld task-manifest construction for the G2PO comparison.

The generic study runtime uses fixed 20/50-step buckets.  They leave only six
``valid_unseen`` tasks in the short stratum of the official ALFWorld cache,
which makes a three-stratum mechanism result mostly a reporting artefact.  The
registered G2PO study instead freezes *evaluation-split* reference-plan
terciles before any treatment runs, then applies those exact thresholds to both
the train and evaluation manifests.

This module creates and validates the two manifests without changing the
generic GraphGPO/ALFWorld runtime contract.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from lh_platform.studies.runtime import AlfworldTaskManifest, build_alfworld_task_manifest

G2PO_HORIZON_STRATIFICATION = "frozen_valid_unseen_reference_steps_terciles"
_REQUIRED_SPLITS = ("train", "valid_unseen")
_HORIZON_BUCKETS = frozenset({"short", "medium", "long"})

__all__ = [
    "G2PO_HORIZON_STRATIFICATION",
    "G2POManifestError",
    "build_g2po_alfworld_manifest_pair",
    "frozen_valid_unseen_tercile_policy",
    "validate_g2po_alfworld_manifest_pair",
    "load_and_validate_g2po_alfworld_manifest_pair",
]


class G2POManifestError(ValueError):
    """Raised when a registered G2PO train/evaluation manifest is invalid."""


def build_g2po_alfworld_manifest_pair(
    data_root: str | Path,
    *,
    dataset_root_label: str = "ALFWORLD_DATA",
) -> tuple[AlfworldTaskManifest, AlfworldTaskManifest]:
    """Build split-only manifests with thresholds frozen from valid_unseen.

    The thresholds are derived once from the immutable official evaluation split
    and are not recomputed per seed, treatment, or run.  This makes the three
    reported horizon strata comparable while avoiding the sparse short stratum
    produced by the generic 20/50 policy.
    """

    source = build_alfworld_task_manifest(data_root)
    policy = frozen_valid_unseen_tercile_policy(source)
    train = _partition_manifest(
        source,
        split="train",
        policy=policy,
        dataset_root_label=dataset_root_label,
    )
    evaluation = _partition_manifest(
        source,
        split="valid_unseen",
        policy=policy,
        dataset_root_label=dataset_root_label,
    )
    validate_g2po_alfworld_manifest_pair(train, evaluation)
    return train, evaluation


def frozen_valid_unseen_tercile_policy(
    manifest: AlfworldTaskManifest,
) -> dict[str, Any]:
    """Return deterministic upper-bound tertiles from valid_unseen lengths."""

    reference_steps = sorted(
        task.reference_steps for task in manifest.tasks if task.split == "valid_unseen"
    )
    if len(reference_steps) < 3:
        raise G2POManifestError(
            "valid_unseen needs at least three tasks to freeze three horizon strata"
        )
    short_max = reference_steps[math.ceil(len(reference_steps) / 3) - 1]
    medium_max = reference_steps[math.ceil(2 * len(reference_steps) / 3) - 1]
    if short_max >= medium_max:
        raise G2POManifestError(
            "valid_unseen reference-plan tertiles do not form three distinct strata"
        )
    counts = Counter(_horizon_bucket(step, short_max, medium_max) for step in reference_steps)
    if set(counts) != _HORIZON_BUCKETS:
        raise G2POManifestError("valid_unseen tertiles did not populate all horizon strata")
    return {
        "stratification": G2PO_HORIZON_STRATIFICATION,
        "frozen_from_split": "valid_unseen",
        "short_max_reference_steps": short_max,
        "medium_max_reference_steps": medium_max,
        "valid_unseen_stratum_counts": dict(sorted(counts.items())),
    }


def validate_g2po_alfworld_manifest_pair(
    train: AlfworldTaskManifest,
    evaluation: AlfworldTaskManifest,
) -> None:
    """Reject a pair that cannot define the registered comparison population."""

    _validate_partition(train, expected_split="train")
    _validate_partition(evaluation, expected_split="valid_unseen")
    if dict(train.horizon_policy) != dict(evaluation.horizon_policy):
        raise G2POManifestError("train/evaluation manifests use different horizon policies")
    policy = dict(evaluation.horizon_policy)
    if policy.get("stratification") != G2PO_HORIZON_STRATIFICATION:
        raise G2POManifestError("manifest pair does not use frozen valid_unseen tertiles")
    expected_counts = Counter(task.horizon_bucket for task in evaluation.tasks)
    if dict(policy.get("valid_unseen_stratum_counts") or {}) != dict(
        sorted(expected_counts.items())
    ):
        raise G2POManifestError("valid_unseen horizon counts do not match frozen policy")


def load_and_validate_g2po_alfworld_manifest_pair(
    train_path: str | Path,
    eval_path: str | Path,
) -> tuple[AlfworldTaskManifest, AlfworldTaskManifest]:
    """Load the registered pair without requiring a local ALFWorld cache."""

    train_file = Path(train_path)
    eval_file = Path(eval_path)
    missing = [str(path) for path in (train_file, eval_file) if not path.is_file()]
    if missing:
        raise G2POManifestError("registered task manifest missing: " + ", ".join(missing))
    try:
        train = AlfworldTaskManifest.read(train_file)
        evaluation = AlfworldTaskManifest.read(eval_file)
    except (OSError, TypeError, ValueError) as error:
        raise G2POManifestError(f"invalid registered task manifest: {error}") from error
    validate_g2po_alfworld_manifest_pair(train, evaluation)
    return train, evaluation


def _partition_manifest(
    source: AlfworldTaskManifest,
    *,
    split: str,
    policy: Mapping[str, Any],
    dataset_root_label: str,
) -> AlfworldTaskManifest:
    if split not in _REQUIRED_SPLITS:
        raise G2POManifestError(f"unsupported G2PO manifest split: {split!r}")
    short_max = int(policy["short_max_reference_steps"])
    medium_max = int(policy["medium_max_reference_steps"])
    tasks = tuple(
        replace(
            task,
            horizon_bucket=_horizon_bucket(task.reference_steps, short_max, medium_max),
        )
        for task in source.tasks
        if task.split == split
    )
    return AlfworldTaskManifest(
        schema=source.schema,
        benchmark=source.benchmark,
        dataset_root=dataset_root_label,
        horizon_policy=dict(policy),
        tasks=tasks,
    )


def _validate_partition(manifest: AlfworldTaskManifest, *, expected_split: str) -> None:
    if manifest.schema != "lh_task_manifest/v1" or manifest.benchmark != "alfworld_native":
        raise G2POManifestError("registered manifest is not an ALFWorld task manifest")
    if not manifest.tasks:
        raise G2POManifestError(f"{expected_split} manifest is empty")
    actual_splits = {task.split for task in manifest.tasks}
    if actual_splits != {expected_split}:
        raise G2POManifestError(
            f"manifest must contain only {expected_split}, got {sorted(actual_splits)}"
        )
    buckets = {task.horizon_bucket for task in manifest.tasks}
    if buckets != _HORIZON_BUCKETS:
        raise G2POManifestError(
            f"{expected_split} manifest must contain short, medium, and long tasks"
        )
    task_ids = [task.task_id for task in manifest.tasks]
    if len(task_ids) != len(set(task_ids)):
        raise G2POManifestError(f"{expected_split} manifest contains duplicate task ids")


def _horizon_bucket(reference_steps: int, short_max: int, medium_max: int) -> str:
    if reference_steps <= short_max:
        return "short"
    if reference_steps <= medium_max:
        return "medium"
    return "long"
