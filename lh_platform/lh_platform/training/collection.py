"""Fail-closed construction of canonical train/eval trajectory collections.

The training dataset is represented at transition level, while scientific
independence lives at task/group level.  Splitting JSONL rows directly can put
two steps from one trajectory, or two rollouts of one task, into both train and
evaluation.  This module makes that impossible: it partitions connected
components of ``task_id`` and ``group_uid`` and emits a manifest proving the
resulting sets are disjoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from lh_platform.training.contracts import TrainingTrajectoryRecord

TRAIN_EVAL_COLLECTION_SCHEMA = "lh_train_eval_collection/v1"
DEFAULT_TRAIN_SPLITS = ("train",)
DEFAULT_EVAL_SPLITS = (
    "val",
    "valid",
    "validation",
    "valid_seen",
    "valid_unseen",
    "eval",
    "test",
)


class CollectionIntegrityError(ValueError):
    """Raised when a train/eval collection cannot be made independent."""


@dataclass(frozen=True)
class CollectionPartition:
    """Content-addressed description of one collection partition."""

    name: str
    source_splits: tuple[str, ...]
    record_count: int
    trajectory_count: int
    group_count: int
    task_count: int
    records_sha256: str
    trajectories_sha256: str
    groups_sha256: str
    tasks_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainEvalCollectionManifest:
    """Immutable provenance and leakage proof for a collection."""

    schema: str
    collection_id: str
    dataset_id: str
    source_record_count: int
    source_records_sha256: str
    split_strategy: str
    holdout_fraction: float | None
    identity_basis: str
    train: CollectionPartition
    evaluation: CollectionPartition
    trajectory_overlap_count: int
    group_overlap_count: int
    task_overlap_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainEvalCollection:
    """A group-safe canonical collection plus its auditable manifest."""

    train_records: tuple[TrainingTrajectoryRecord, ...]
    eval_records: tuple[TrainingTrajectoryRecord, ...]
    manifest: TrainEvalCollectionManifest

    def write(
        self,
        output_dir: str | Path,
        *,
        train_filename: str = "train.jsonl",
        eval_filename: str = "eval.jsonl",
        manifest_filename: str = "train_eval_collection_manifest.json",
    ) -> Mapping[str, Path]:
        """Materialize canonical rows and a manifest without changing their content."""

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        train_path = root / train_filename
        eval_path = root / eval_filename
        manifest_path = root / manifest_filename
        _write_records(train_path, self.train_records)
        _write_records(eval_path, self.eval_records)
        manifest_path.write_text(
            json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"train": train_path, "eval": eval_path, "manifest": manifest_path}


def build_train_eval_collection(
    records: Sequence[TrainingTrajectoryRecord] | Iterable[TrainingTrajectoryRecord],
    *,
    holdout_fraction: float = 0.2,
    train_splits: Sequence[str] = DEFAULT_TRAIN_SPLITS,
    eval_splits: Sequence[str] = DEFAULT_EVAL_SPLITS,
) -> TrainEvalCollection:
    """Build a disjoint train/eval collection from canonical trajectory rows.

    Explicit source splits take precedence when both train and evaluation labels
    are present.  Otherwise an all-train source gets a stable hash-based
    holdout over connected task/group components.  A component is the smallest
    unit that keeps *both* task and group identities in one partition.
    """

    # Canonical order is part of the collection contract.  The same source
    # rows must produce byte-stable collection artifacts even when an upstream
    # loader changes its iteration order.
    rows = tuple(sorted(records, key=_record_sort_key))
    if not rows:
        raise CollectionIntegrityError("cannot build a train/eval collection from zero records")
    if not 0.0 < holdout_fraction < 1.0:
        raise CollectionIntegrityError("holdout_fraction must be strictly between 0 and 1")

    _validate_records(rows)
    dataset_ids = {row.dataset_id for row in rows}
    if len(dataset_ids) != 1:
        raise CollectionIntegrityError(
            "a train/eval collection must bind exactly one dataset_id; "
            f"received {sorted(dataset_ids)!r}"
        )
    dataset_id = next(iter(dataset_ids))
    train_labels = {str(split) for split in train_splits}
    eval_labels = {str(split) for split in eval_splits}
    if not train_labels or not eval_labels or train_labels & eval_labels:
        raise CollectionIntegrityError(
            "train_splits and eval_splits must be non-empty and disjoint"
        )

    components, component_by_record = _task_group_components(rows)
    source_splits = {row.split for row in rows}
    unknown_splits = source_splits - train_labels - eval_labels
    if unknown_splits:
        raise CollectionIntegrityError(
            "collection contains unrecognized source splits: " + ", ".join(sorted(unknown_splits))
        )

    has_train = bool(source_splits & train_labels)
    has_eval = bool(source_splits & eval_labels)
    if has_train and has_eval:
        strategy = "source_split"
        train_component_ids, eval_component_ids = _source_split_components(
            rows,
            components,
            train_labels=train_labels,
            eval_labels=eval_labels,
        )
        used_holdout_fraction: float | None = None
    elif has_train:
        strategy = "deterministic_task_group_holdout"
        train_component_ids, eval_component_ids = _deterministic_holdout_components(
            components,
            dataset_id=dataset_id,
            holdout_fraction=holdout_fraction,
        )
        used_holdout_fraction = holdout_fraction
    else:
        raise CollectionIntegrityError(
            "collection needs a train source split, or an all-train source for deterministic holdout"
        )

    train_rows = tuple(
        row for index, row in enumerate(rows) if component_by_record[index] in train_component_ids
    )
    eval_rows = tuple(
        row for index, row in enumerate(rows) if component_by_record[index] in eval_component_ids
    )
    if not train_rows or not eval_rows:
        raise CollectionIntegrityError(
            "train/eval collection must contain at least one record in each partition"
        )

    train_trajectories = {row.traj_uid for row in train_rows}
    eval_trajectories = {row.traj_uid for row in eval_rows}
    train_groups = {row.group_uid for row in train_rows}
    eval_groups = {row.group_uid for row in eval_rows}
    train_tasks = {row.task_id for row in train_rows}
    eval_tasks = {row.task_id for row in eval_rows}
    trajectory_overlap = train_trajectories & eval_trajectories
    group_overlap = train_groups & eval_groups
    task_overlap = train_tasks & eval_tasks
    if trajectory_overlap or group_overlap or task_overlap:
        details: list[str] = []
        if trajectory_overlap:
            details.append("trajectories=" + ",".join(sorted(trajectory_overlap)[:5]))
        if group_overlap:
            details.append("groups=" + ",".join(sorted(group_overlap)[:5]))
        if task_overlap:
            details.append("tasks=" + ",".join(sorted(task_overlap)[:5]))
        raise CollectionIntegrityError(
            "trajectory/task/group leakage remains after collection split: " + "; ".join(details)
        )

    source_hash = records_sha256(rows)
    train_partition = _partition("train", train_rows)
    eval_partition = _partition("evaluation", eval_rows)
    collection_id = hashlib.sha256(
        (
            f"{dataset_id}\0{source_hash}\0{strategy}\0"
            f"{train_partition.records_sha256}\0{eval_partition.records_sha256}"
        ).encode()
    ).hexdigest()[:16]
    manifest = TrainEvalCollectionManifest(
        schema=TRAIN_EVAL_COLLECTION_SCHEMA,
        collection_id=collection_id,
        dataset_id=dataset_id,
        source_record_count=len(rows),
        source_records_sha256=source_hash,
        split_strategy=strategy,
        holdout_fraction=used_holdout_fraction,
        identity_basis="declared_task_group_trajectory_ids",
        train=train_partition,
        evaluation=eval_partition,
        trajectory_overlap_count=0,
        group_overlap_count=0,
        task_overlap_count=0,
    )
    return TrainEvalCollection(
        train_records=train_rows,
        eval_records=eval_rows,
        manifest=manifest,
    )


def _validate_records(rows: tuple[TrainingTrajectoryRecord, ...]) -> None:
    for index, row in enumerate(rows):
        missing = [
            name
            for name, value in (
                ("dataset_id", row.dataset_id),
                ("task_id", row.task_id),
                ("group_uid", row.group_uid),
                ("traj_uid", row.traj_uid),
                ("split", row.split),
            )
            if not str(value).strip()
        ]
        if missing:
            raise CollectionIntegrityError(
                f"record {index} lacks collection identity fields: {', '.join(missing)}"
            )


def _task_group_components(
    rows: tuple[TrainingTrajectoryRecord, ...],
) -> tuple[dict[str, tuple[int, ...]], tuple[str, ...]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    for row in rows:
        group_node = f"group\0{row.group_uid}"
        union(group_node, f"task\0{row.task_id}")
        # The same rollout must never appear in both partitions, even when an
        # upstream exporter gives its steps inconsistent task/group labels.
        union(group_node, f"trajectory\0{row.traj_uid}")

    indices_by_component: dict[str, list[int]] = {}
    component_by_record: list[str] = []
    for index, row in enumerate(rows):
        component_id = find(f"group\0{row.group_uid}")
        component_by_record.append(component_id)
        indices_by_component.setdefault(component_id, []).append(index)
    return (
        {key: tuple(value) for key, value in indices_by_component.items()},
        tuple(component_by_record),
    )


def _source_split_components(
    rows: tuple[TrainingTrajectoryRecord, ...],
    components: Mapping[str, tuple[int, ...]],
    *,
    train_labels: set[str],
    eval_labels: set[str],
) -> tuple[set[str], set[str]]:
    train_components: set[str] = set()
    eval_components: set[str] = set()
    for component, indices in components.items():
        labels = {rows[index].split for index in indices}
        has_train = bool(labels & train_labels)
        has_eval = bool(labels & eval_labels)
        if has_train and has_eval:
            raise CollectionIntegrityError(
                "source split leakage: one task/group component appears in both train and evaluation"
            )
        if has_train:
            train_components.add(component)
        elif has_eval:
            eval_components.add(component)
        else:  # Defensive: source labels were checked before this function.
            raise CollectionIntegrityError(
                "source split leakage: task/group component has no recognized partition"
            )
    if not train_components or not eval_components:
        raise CollectionIntegrityError(
            "source_split collection requires at least one task/group component in each partition"
        )
    return train_components, eval_components


def _deterministic_holdout_components(
    components: Mapping[str, tuple[int, ...]],
    *,
    dataset_id: str,
    holdout_fraction: float,
) -> tuple[set[str], set[str]]:
    if len(components) < 2:
        raise CollectionIntegrityError(
            "cannot create a group-safe train/eval collection: at least two task/group components are required"
        )
    ordered = sorted(
        components,
        key=lambda component: hashlib.sha256(f"{dataset_id}\0{component}".encode()).hexdigest(),
    )
    eval_count = max(1, min(len(ordered) - 1, round(len(ordered) * holdout_fraction)))
    eval_components = set(ordered[:eval_count])
    return set(ordered) - eval_components, eval_components


def _partition(name: str, rows: tuple[TrainingTrajectoryRecord, ...]) -> CollectionPartition:
    trajectories = tuple(sorted({row.traj_uid for row in rows}))
    groups = tuple(sorted({row.group_uid for row in rows}))
    tasks = tuple(sorted({row.task_id for row in rows}))
    return CollectionPartition(
        name=name,
        source_splits=tuple(sorted({row.split for row in rows})),
        record_count=len(rows),
        trajectory_count=len(trajectories),
        group_count=len(groups),
        task_count=len(tasks),
        records_sha256=records_sha256(rows),
        trajectories_sha256=_strings_sha256(trajectories),
        groups_sha256=_strings_sha256(groups),
        tasks_sha256=_strings_sha256(tasks),
    )


def records_sha256(rows: Sequence[TrainingTrajectoryRecord]) -> str:
    """Return the order-independent content hash used by collection manifests.

    The hash is intentionally public: downstream benchmark contracts must be
    able to prove that a training partition is exactly the one materialized by
    a collection, without trusting its JSONL row order or filename.
    """

    canonical_rows = sorted(
        json.dumps(row.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in rows
    )
    return hashlib.sha256("\n".join(canonical_rows).encode("utf-8")).hexdigest()


def _record_sort_key(row: TrainingTrajectoryRecord) -> tuple[object, ...]:
    """Keep collection membership and serialized row order reproducible."""

    return (
        row.dataset_id,
        row.benchmark,
        row.task_id,
        row.group_uid,
        row.traj_uid,
        row.step_idx,
        json.dumps(row.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    )


def _strings_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _write_records(path: Path, rows: Sequence[TrainingTrajectoryRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
