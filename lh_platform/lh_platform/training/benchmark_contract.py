"""Bind ALFWorld training partitions to source-backed benchmark evidence.

``TrainEvalCollection`` prevents declared task/group/trajectory leakage.  It
does not establish that those declarations refer to the actual ALFWorld files,
or that the source owns a ``won`` outcome.  This module creates an immutable
bundle that proves the input-side contract only.  It deliberately does *not*
claim that a native benchmark outcome was executed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from lh_platform.studies.runtime import (
    ALFWORLD_SPLITS,
    BENCHMARK_AUDIT_SCHEMA,
    TASK_MANIFEST_SCHEMA,
    AlfworldTask,
    AlfworldTaskManifest,
    BenchmarkAuditReport,
    build_alfworld_task_manifest,
    sha256_file,
)
from lh_platform.training.collection import (
    TRAIN_EVAL_COLLECTION_SCHEMA,
    TrainEvalCollection,
    records_sha256,
)
from lh_platform.training.contracts import TrainingDatasetManifest, TrainingTrajectoryRecord
from lh_platform.training.dataset import read_jsonl

ALFWORLD_TRAINING_BENCHMARK_BINDING_SCHEMA = "lh_alfworld_training_benchmark_binding/v2"
ALFWORLD_TRAINING_BUNDLE_SCHEMA = "lh_alfworld_training_bundle/v1"
ALFWORLD_REQUIRED_AUDIT_COMPONENTS = (
    "user",
    "environment",
    "ground_truth",
    "evaluation",
)
ALFWORLD_REQUIRED_LOGIC_FILES = (
    "logic/alfred.pddl",
    "logic/alfred.twl2",
)


class BenchmarkContractError(ValueError):
    """Raised when a training partition lacks source-backed benchmark identity."""


@dataclass(frozen=True)
class AlfworldTrainingBenchmarkBinding:
    """Immutable evidence for a source-bound ALFWorld training bundle.

    ``source_evaluator_contract_bound`` means that task identity, selected
    environment files, source split, and the source-owned evaluator *contract*
    were verified.  It never means that the records' outcomes were rerun or
    accepted as an official ALFWorld score.
    """

    schema: str
    benchmark: str
    collection_id: str
    dataset_id: str
    collection_split_strategy: str
    collection_holdout_fraction: float | None
    source_records_sha256: str
    train_records_sha256: str
    evaluation_records_sha256: str
    train_record_count: int
    evaluation_record_count: int
    task_manifest_path: str
    task_manifest_sha256: str
    task_manifest_task_count: int
    alfworld_data_root: str | None
    alfworld_data_artifacts_sha256: str | None
    alfworld_data_verified: bool
    collection_record_count: int
    bound_task_count: int
    identity_contract: str
    train_source_splits: tuple[str, ...]
    evaluation_source_splits: tuple[str, ...]
    source_split_evaluation_bound: bool
    benchmark_audit_path: str | None
    benchmark_audit_sha256: str | None
    benchmark_audit_source_commit: str | None
    benchmark_audit_verified: bool
    source_evaluator_contract_bound: bool
    outcome_owner: str | None
    outcome_field: str | None
    outcome_evidence_quality: str | None
    native_outcomes_verified: bool
    outcome_contract_status: str
    claim_scope: str = "plumbing"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def read(cls, path: str | Path) -> AlfworldTrainingBenchmarkBinding:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("ALFWorld training binding must be a JSON object")
        try:
            return cls(
                schema=str(raw["schema"]),
                benchmark=str(raw["benchmark"]),
                collection_id=str(raw["collection_id"]),
                dataset_id=str(raw["dataset_id"]),
                collection_split_strategy=str(raw["collection_split_strategy"]),
                collection_holdout_fraction=(
                    float(raw["collection_holdout_fraction"])
                    if raw.get("collection_holdout_fraction") is not None
                    else None
                ),
                source_records_sha256=str(raw["source_records_sha256"]),
                train_records_sha256=str(raw["train_records_sha256"]),
                evaluation_records_sha256=str(raw["evaluation_records_sha256"]),
                train_record_count=int(raw["train_record_count"]),
                evaluation_record_count=int(raw["evaluation_record_count"]),
                task_manifest_path=str(raw["task_manifest_path"]),
                task_manifest_sha256=str(raw["task_manifest_sha256"]),
                task_manifest_task_count=int(raw["task_manifest_task_count"]),
                alfworld_data_root=(
                    str(raw["alfworld_data_root"])
                    if raw.get("alfworld_data_root") is not None
                    else None
                ),
                alfworld_data_artifacts_sha256=(
                    str(raw["alfworld_data_artifacts_sha256"])
                    if raw.get("alfworld_data_artifacts_sha256") is not None
                    else None
                ),
                alfworld_data_verified=bool(raw["alfworld_data_verified"]),
                collection_record_count=int(raw["collection_record_count"]),
                bound_task_count=int(raw["bound_task_count"]),
                identity_contract=str(raw["identity_contract"]),
                train_source_splits=tuple(str(value) for value in raw["train_source_splits"]),
                evaluation_source_splits=tuple(
                    str(value) for value in raw["evaluation_source_splits"]
                ),
                source_split_evaluation_bound=bool(raw["source_split_evaluation_bound"]),
                benchmark_audit_path=(
                    str(raw["benchmark_audit_path"])
                    if raw.get("benchmark_audit_path") is not None
                    else None
                ),
                benchmark_audit_sha256=(
                    str(raw["benchmark_audit_sha256"])
                    if raw.get("benchmark_audit_sha256") is not None
                    else None
                ),
                benchmark_audit_source_commit=(
                    str(raw["benchmark_audit_source_commit"])
                    if raw.get("benchmark_audit_source_commit") is not None
                    else None
                ),
                benchmark_audit_verified=bool(raw["benchmark_audit_verified"]),
                source_evaluator_contract_bound=bool(raw["source_evaluator_contract_bound"]),
                outcome_owner=(str(raw["outcome_owner"]) if raw.get("outcome_owner") else None),
                outcome_field=(str(raw["outcome_field"]) if raw.get("outcome_field") else None),
                outcome_evidence_quality=(
                    str(raw["outcome_evidence_quality"])
                    if raw.get("outcome_evidence_quality")
                    else None
                ),
                native_outcomes_verified=bool(raw["native_outcomes_verified"]),
                outcome_contract_status=str(raw["outcome_contract_status"]),
                claim_scope=str(raw.get("claim_scope", "plumbing")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid ALFWorld training binding: {error}") from error


def bind_alfworld_train_eval_collection(
    collection: TrainEvalCollection,
    *,
    task_manifest_path: str | Path,
    benchmark_audit_path: str | Path | None = None,
    alfworld_data_root: str | Path | None = None,
    require_source_evaluator_contract: bool = False,
) -> AlfworldTrainingBenchmarkBinding:
    """Bind a group-safe collection to manifest, data artifacts, and audit evidence.

    A deterministic holdout over ``train`` tasks remains useful for plumbing,
    but cannot become a source-split benchmark comparison.  Requiring the
    source evaluator contract additionally requires a re-scanned ALFWorld data
    root; static manifest declarations alone are insufficient.
    """

    if collection.manifest.schema != TRAIN_EVAL_COLLECTION_SCHEMA:
        raise BenchmarkContractError(
            "unsupported train/eval collection schema: "
            f"expected {TRAIN_EVAL_COLLECTION_SCHEMA}, got {collection.manifest.schema!r}"
        )
    manifest_file, manifest = _load_task_manifest(task_manifest_path)
    manifest_sha256 = sha256_file(manifest_file)
    _validate_collection_identity(collection, manifest)

    data_root: Path | None = None
    data_artifacts_sha256: str | None = None
    if alfworld_data_root is not None:
        data_root, data_artifacts_sha256 = _verify_alfworld_data_root(
            manifest,
            alfworld_data_root,
        )

    audit_file: Path | None = None
    audit: BenchmarkAuditReport | None = None
    if benchmark_audit_path is not None:
        audit_file, audit = _load_verified_benchmark_audit(
            benchmark_audit_path,
            expected_manifest_sha256=manifest_sha256,
        )

    train_source_splits = tuple(sorted({record.split for record in collection.train_records}))
    evaluation_source_splits = tuple(sorted({record.split for record in collection.eval_records}))
    source_split_evaluation_bound = (
        collection.manifest.split_strategy == "source_split"
        and train_source_splits == ("train",)
        and bool(evaluation_source_splits)
        and set(evaluation_source_splits).issubset({"valid_seen", "valid_unseen"})
    )
    benchmark_audit_verified = audit is not None
    source_evaluator_contract_bound = (
        source_split_evaluation_bound
        and benchmark_audit_verified
        and data_root is not None
        and data_artifacts_sha256 is not None
    )
    if require_source_evaluator_contract and not source_evaluator_contract_bound:
        reasons: list[str] = []
        if not source_split_evaluation_bound:
            reasons.append("collection is not source-aligned train versus valid_seen/valid_unseen")
        if not benchmark_audit_verified:
            reasons.append("a verified benchmark audit was not supplied")
        if data_root is None:
            reasons.append("a re-scanned --alfworld-data root was not supplied")
        raise BenchmarkContractError(
            "source evaluator contract requires source-aligned evaluation, a verified "
            "benchmark audit, and revalidated ALFWorld data: " + "; ".join(reasons)
        )

    records = collection.train_records + collection.eval_records
    return AlfworldTrainingBenchmarkBinding(
        schema=ALFWORLD_TRAINING_BENCHMARK_BINDING_SCHEMA,
        benchmark=manifest.benchmark,
        collection_id=collection.manifest.collection_id,
        dataset_id=collection.manifest.dataset_id,
        collection_split_strategy=collection.manifest.split_strategy,
        collection_holdout_fraction=collection.manifest.holdout_fraction,
        source_records_sha256=collection.manifest.source_records_sha256,
        train_records_sha256=collection.manifest.train.records_sha256,
        evaluation_records_sha256=collection.manifest.evaluation.records_sha256,
        train_record_count=collection.manifest.train.record_count,
        evaluation_record_count=collection.manifest.evaluation.record_count,
        task_manifest_path=str(manifest_file),
        task_manifest_sha256=manifest_sha256,
        task_manifest_task_count=len(manifest.tasks),
        alfworld_data_root=str(data_root) if data_root else None,
        alfworld_data_artifacts_sha256=data_artifacts_sha256,
        alfworld_data_verified=data_root is not None and data_artifacts_sha256 is not None,
        collection_record_count=len(records),
        bound_task_count=len({record.task_id for record in records}),
        identity_contract=(
            "exact(task_id, group_uid, split) against lh_task_manifest/v1; "
            "task files and ALFWorld logic are content-addressed when data is supplied"
        ),
        train_source_splits=train_source_splits,
        evaluation_source_splits=evaluation_source_splits,
        source_split_evaluation_bound=source_split_evaluation_bound,
        benchmark_audit_path=str(audit_file) if audit_file else None,
        benchmark_audit_sha256=sha256_file(audit_file) if audit_file else None,
        benchmark_audit_source_commit=audit.source_commit if audit else None,
        benchmark_audit_verified=benchmark_audit_verified,
        source_evaluator_contract_bound=source_evaluator_contract_bound,
        outcome_owner=audit.outcome_owner if audit else None,
        outcome_field=audit.outcome_field if audit else None,
        outcome_evidence_quality=audit.outcome_evidence_quality if audit else None,
        native_outcomes_verified=False,
        outcome_contract_status="not_executed",
    )


def materialize_alfworld_training_bundle(
    collection: TrainEvalCollection,
    binding: AlfworldTrainingBenchmarkBinding,
    *,
    source_dataset_manifest: TrainingDatasetManifest,
    output_root: str | Path,
) -> Mapping[str, Path]:
    """Atomically publish a uniquely named bound collection bundle.

    A failed bind must never leave a stale ``train.jsonl`` or binding that can
    be mistaken for the current invocation.  Each successful collection is
    published beneath its content-derived ``collection_id``; existing bundles
    are preserved as evidence and never overwritten.
    """

    root = Path(output_root).expanduser().resolve()
    _validate_collection_matches_binding(collection, binding)
    if root.exists() and not root.is_dir():
        raise BenchmarkContractError(f"ALFWorld bundle output root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    legacy_names = {
        "train.jsonl",
        "eval.jsonl",
        "train_eval_collection_manifest.json",
        "alfworld_training_benchmark_binding.json",
    }
    legacy = sorted(name for name in legacy_names if (root / name).exists())
    if legacy:
        raise BenchmarkContractError(
            "output root contains legacy direct artifacts; choose a fresh bundle root "
            "to preserve their provenance: " + ", ".join(legacy)
        )

    target = root / binding.collection_id
    if target.exists():
        raise BenchmarkContractError(
            f"bound collection already exists and will not be overwritten: {target}"
        )
    staging = root / f".{binding.collection_id}.staging-{uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        paths = dict(collection.write(staging))
        source_manifest_path = staging / "source_training_dataset_manifest.json"
        source_manifest_path.write_text(
            json.dumps(source_dataset_manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        binding_path = binding.write(staging / "alfworld_training_benchmark_binding.json")
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in (
                paths["train"],
                paths["eval"],
                paths["manifest"],
                source_manifest_path,
                binding_path,
            )
        }
        bundle_manifest_path = staging / "alfworld_training_bundle_manifest.json"
        bundle_manifest_path.write_text(
            json.dumps(
                {
                    "schema": ALFWORLD_TRAINING_BUNDLE_SCHEMA,
                    "collection_id": binding.collection_id,
                    "binding_schema": binding.schema,
                    "artifact_sha256": artifact_hashes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "bundle": target,
        "train": target / "train.jsonl",
        "eval": target / "eval.jsonl",
        "manifest": target / "train_eval_collection_manifest.json",
        "source_dataset_manifest": target / "source_training_dataset_manifest.json",
        "binding": target / "alfworld_training_benchmark_binding.json",
        "bundle_manifest": target / "alfworld_training_bundle_manifest.json",
    }


def require_alfworld_training_bundle_if_needed(
    dataset_path: str | Path,
    *,
    binding_path: str | Path | None,
) -> AlfworldTrainingBenchmarkBinding | None:
    """Require a verified bound bundle only when a dataset declares ALFWorld.

    This lets generic ToolBench plumbing remain usable while preventing the
    user-facing training dispatchers from accepting raw ALFWorld rollout JSONL
    as if it were a source-bound scientific training partition.
    """

    dataset = Path(dataset_path)
    if not dataset.is_file():
        raise BenchmarkContractError(f"training dataset not found: {dataset}")
    benchmark_values = _declared_benchmarks(dataset)
    if "alfworld_native" not in benchmark_values:
        return None
    if benchmark_values != {"alfworld_native"}:
        raise BenchmarkContractError(
            "ALFWorld source-bound training cannot mix benchmark identities in one dataset"
        )
    if binding_path is None:
        raise BenchmarkContractError(
            "ALFWorld training requires --alfworld-binding from "
            "bind-alfworld-training-collection; raw rollout JSONL is blocked"
        )
    return verify_alfworld_training_partition(dataset, binding_path)


def verify_alfworld_training_partition(
    dataset_path: str | Path,
    binding_path: str | Path,
) -> AlfworldTrainingBenchmarkBinding:
    """Verify that ``dataset_path`` is the published train partition of a bundle."""

    dataset = Path(dataset_path).expanduser().resolve()
    binding_file = Path(binding_path).expanduser().resolve()
    if not binding_file.is_file():
        raise BenchmarkContractError(f"ALFWorld training binding not found: {binding_file}")
    try:
        binding = AlfworldTrainingBenchmarkBinding.read(binding_file)
    except (OSError, TypeError, ValueError) as error:
        raise BenchmarkContractError(f"invalid ALFWorld training binding: {error}") from error
    if binding.schema != ALFWORLD_TRAINING_BENCHMARK_BINDING_SCHEMA:
        raise BenchmarkContractError(
            "unsupported ALFWorld training binding schema: "
            f"expected {ALFWORLD_TRAINING_BENCHMARK_BINDING_SCHEMA}, got {binding.schema!r}"
        )
    if binding.claim_scope != "plumbing" or binding.native_outcomes_verified:
        raise BenchmarkContractError(
            "training binding has an invalid outcome claim; input binding cannot verify native outcomes"
        )
    if binding.outcome_contract_status != "not_executed":
        raise BenchmarkContractError(
            "training binding must declare outcome_contract_status=not_executed"
        )
    if not binding.source_evaluator_contract_bound:
        raise BenchmarkContractError(
            "ALFWorld training binding lacks a verified source evaluator contract"
        )

    bundle_root = binding_file.parent
    expected_train = bundle_root / "train.jsonl"
    expected_eval = bundle_root / "eval.jsonl"
    if dataset != expected_train.resolve():
        raise BenchmarkContractError(
            "ALFWorld training dataset must be the bundle's published train.jsonl: "
            f"expected {expected_train}, got {dataset}"
        )
    if not expected_eval.is_file():
        raise BenchmarkContractError(f"ALFWorld bundle is missing eval partition: {expected_eval}")
    _verify_bundle_manifest(bundle_root, binding)
    _verify_collection_manifest(bundle_root, binding)

    train_records = read_jsonl(dataset)
    eval_records = read_jsonl(expected_eval)
    _verify_partition_records(
        train_records,
        expected_count=binding.train_record_count,
        expected_sha256=binding.train_records_sha256,
        partition_name="train",
        expected_dataset_id=binding.dataset_id,
    )
    _verify_partition_records(
        eval_records,
        expected_count=binding.evaluation_record_count,
        expected_sha256=binding.evaluation_records_sha256,
        partition_name="evaluation",
        expected_dataset_id=binding.dataset_id,
    )
    manifest = _verify_published_binding_evidence(binding)
    _validate_record_identity(train_records, manifest, partition_name="published train")
    _validate_record_identity(eval_records, manifest, partition_name="published evaluation")
    return binding


def _load_task_manifest(path: str | Path) -> tuple[Path, AlfworldTaskManifest]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise BenchmarkContractError(f"ALFWorld task manifest not found: {manifest_path}")
    try:
        manifest = AlfworldTaskManifest.read(manifest_path)
    except (OSError, TypeError, ValueError) as error:
        raise BenchmarkContractError(f"invalid ALFWorld task manifest: {error}") from error

    if manifest.schema != TASK_MANIFEST_SCHEMA:
        raise BenchmarkContractError(
            "unsupported ALFWorld task manifest schema: "
            f"expected {TASK_MANIFEST_SCHEMA}, got {manifest.schema!r}"
        )
    if manifest.benchmark != "alfworld_native":
        raise BenchmarkContractError(
            "ALFWorld task manifest benchmark mismatch: "
            f"expected 'alfworld_native', got {manifest.benchmark!r}"
        )
    if not manifest.tasks:
        raise BenchmarkContractError("ALFWorld task manifest is empty")

    seen_task_ids: set[str] = set()
    splits_by_group: dict[str, set[str]] = {}
    for task in manifest.tasks:
        _validate_manifest_task(task)
        if task.task_id in seen_task_ids:
            raise BenchmarkContractError(
                f"ALFWorld task manifest has duplicate task_id: {task.task_id!r}"
            )
        seen_task_ids.add(task.task_id)
        splits_by_group.setdefault(task.group_uid, set()).add(task.split)
    leaked_groups = {
        group_uid: sorted(splits)
        for group_uid, splits in splits_by_group.items()
        if len(splits) > 1
    }
    if leaked_groups:
        sample = ", ".join(
            f"{group_uid}={splits}" for group_uid, splits in sorted(leaked_groups.items())[:5]
        )
        raise BenchmarkContractError(f"ALFWorld task manifest has group split leakage: {sample}")
    return manifest_path, manifest


def _validate_manifest_task(task: AlfworldTask) -> None:
    if not task.task_id:
        raise BenchmarkContractError("ALFWorld task manifest has an empty task_id")
    if not task.group_uid:
        raise BenchmarkContractError(
            f"ALFWorld task manifest task {task.task_id!r} has an empty group_uid"
        )
    if task.split not in ALFWORLD_SPLITS:
        raise BenchmarkContractError(
            f"ALFWorld task manifest task {task.task_id!r} has unsupported split {task.split!r}"
        )


def _validate_collection_identity(
    collection: TrainEvalCollection,
    manifest: AlfworldTaskManifest,
) -> None:
    _validate_record_identity(collection.train_records, manifest, partition_name="train")
    _validate_record_identity(collection.eval_records, manifest, partition_name="evaluation")


def _validate_collection_matches_binding(
    collection: TrainEvalCollection,
    binding: AlfworldTrainingBenchmarkBinding,
) -> None:
    """Reject a hand-assembled binding that does not describe this collection."""

    manifest_file, manifest = _load_task_manifest(binding.task_manifest_path)
    if sha256_file(manifest_file) != binding.task_manifest_sha256:
        raise BenchmarkContractError(
            "collection binding refers to a changed ALFWorld task manifest"
        )
    _validate_collection_identity(collection, manifest)

    expected = {
        "benchmark": "alfworld_native",
        "collection_id": collection.manifest.collection_id,
        "dataset_id": collection.manifest.dataset_id,
        "collection_split_strategy": collection.manifest.split_strategy,
        "collection_holdout_fraction": collection.manifest.holdout_fraction,
        "source_records_sha256": collection.manifest.source_records_sha256,
        "train_records_sha256": collection.manifest.train.records_sha256,
        "evaluation_records_sha256": collection.manifest.evaluation.records_sha256,
        "train_record_count": collection.manifest.train.record_count,
        "evaluation_record_count": collection.manifest.evaluation.record_count,
        "collection_record_count": collection.manifest.source_record_count,
    }
    actual = {
        "benchmark": binding.benchmark,
        "collection_id": binding.collection_id,
        "dataset_id": binding.dataset_id,
        "collection_split_strategy": binding.collection_split_strategy,
        "collection_holdout_fraction": binding.collection_holdout_fraction,
        "source_records_sha256": binding.source_records_sha256,
        "train_records_sha256": binding.train_records_sha256,
        "evaluation_records_sha256": binding.evaluation_records_sha256,
        "train_record_count": binding.train_record_count,
        "evaluation_record_count": binding.evaluation_record_count,
        "collection_record_count": binding.collection_record_count,
    }
    mismatches = [name for name, value in expected.items() if actual[name] != value]
    if mismatches:
        raise BenchmarkContractError(
            "collection and ALFWorld binding disagree on: " + ", ".join(mismatches)
        )


def _validate_record_identity(
    records: tuple[TrainingTrajectoryRecord, ...] | list[TrainingTrajectoryRecord],
    manifest: AlfworldTaskManifest,
    *,
    partition_name: str,
) -> None:
    tasks_by_id = {task.task_id: task for task in manifest.tasks}
    for record in records:
        if record.benchmark != manifest.benchmark:
            raise BenchmarkContractError(
                f"{partition_name} record {record.task_id!r} benchmark mismatch: "
                f"expected {manifest.benchmark!r}, got {record.benchmark!r}"
            )
        expected = tasks_by_id.get(record.task_id)
        if expected is None:
            raise BenchmarkContractError(
                f"{partition_name} record has unknown task_id: {record.task_id!r}"
            )
        if record.group_uid != expected.group_uid:
            raise BenchmarkContractError(
                f"{partition_name} record {record.task_id!r} group_uid mismatch: "
                f"expected {expected.group_uid!r}, got {record.group_uid!r}"
            )
        if record.split != expected.split:
            raise BenchmarkContractError(
                f"{partition_name} record {record.task_id!r} split mismatch: "
                f"expected {expected.split!r}, got {record.split!r}"
            )


def _verify_alfworld_data_root(
    manifest: AlfworldTaskManifest,
    data_root: str | Path,
) -> tuple[Path, str]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise BenchmarkContractError(f"ALFWorld data root not found: {root}")
    issues = manifest.validate(root)
    if issues:
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in issues[:8])
        raise BenchmarkContractError(f"ALFWorld manifest does not match data root: {details}")
    missing_logic = [
        relative for relative in ALFWORLD_REQUIRED_LOGIC_FILES if not (root / relative).is_file()
    ]
    if missing_logic:
        raise BenchmarkContractError(
            "ALFWorld data root is missing official logic artifacts: " + ", ".join(missing_logic)
        )
    try:
        short_max_steps = int(manifest.horizon_policy["short_max_reference_steps"])
        medium_max_steps = int(manifest.horizon_policy["medium_max_reference_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkContractError(
            "ALFWorld task manifest has an invalid horizon policy"
        ) from error
    rescanned = build_alfworld_task_manifest(
        root,
        short_max_steps=short_max_steps,
        medium_max_steps=medium_max_steps,
    )
    if rescanned.tasks != manifest.tasks:
        raise BenchmarkContractError(
            "ALFWorld data scan does not reproduce the task manifest; task identity or "
            "environment files changed"
        )
    return root, _alfworld_data_artifacts_sha256(manifest, root)


def _alfworld_data_artifacts_sha256(manifest: AlfworldTaskManifest, root: Path) -> str:
    relative_paths = set(ALFWORLD_REQUIRED_LOGIC_FILES)
    for task in manifest.tasks:
        relative_paths.add(task.game_path)
        relative_paths.add(task.trajectory_path)
    digest = hashlib.sha256()
    digest.update(b"lh_alfworld_data_artifacts/v1\0")
    for relative in sorted(relative_paths):
        artifact = root / relative
        if not artifact.is_file():  # Defensive: manifest.validate should have caught task files.
            raise BenchmarkContractError(f"ALFWorld data artifact is missing: {artifact}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(artifact).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_verified_benchmark_audit(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[Path, BenchmarkAuditReport]:
    audit_path = Path(path).expanduser().resolve()
    if not audit_path.is_file():
        raise BenchmarkContractError(f"benchmark audit not found: {audit_path}")
    try:
        audit = BenchmarkAuditReport.read(audit_path)
    except (OSError, TypeError, ValueError) as error:
        raise BenchmarkContractError(f"invalid benchmark audit: {error}") from error

    if audit.schema != BENCHMARK_AUDIT_SCHEMA:
        raise BenchmarkContractError(
            "unsupported benchmark audit schema: "
            f"expected {BENCHMARK_AUDIT_SCHEMA}, got {audit.schema!r}"
        )
    if audit.benchmark != "alfworld_native":
        raise BenchmarkContractError(
            "benchmark audit benchmark mismatch: "
            f"expected 'alfworld_native', got {audit.benchmark!r}"
        )
    if audit.task_manifest_sha256 != expected_manifest_sha256:
        raise BenchmarkContractError(
            "benchmark audit task-manifest hash mismatch; audit is not bound to this collection manifest"
        )
    if not audit.source_commit:
        raise BenchmarkContractError("benchmark audit source_commit is empty")
    if (
        audit.outcome_owner != "benchmark"
        or audit.outcome_field != "won"
        or audit.outcome_evidence_quality != "strong"
    ):
        raise BenchmarkContractError(
            "benchmark audit does not prove a strong benchmark-owned won outcome contract"
        )

    missing_components = [
        name for name in ALFWORLD_REQUIRED_AUDIT_COMPONENTS if name not in audit.components
    ]
    if missing_components:
        raise BenchmarkContractError(
            "benchmark audit is missing required components: " + ", ".join(missing_components)
        )
    for name in ALFWORLD_REQUIRED_AUDIT_COMPONENTS:
        component = audit.components[name]
        if not component.passed:
            raise BenchmarkContractError(f"benchmark audit component did not pass: {name}")
        if not component.evidence:
            raise BenchmarkContractError(
                f"benchmark audit component has no source evidence: {name}"
            )
        for evidence in component.evidence:
            evidence_path = Path(evidence.path)
            if not evidence_path.is_absolute():
                evidence_path = audit_path.parent / evidence_path
            if not evidence.verified:
                raise BenchmarkContractError(
                    f"benchmark audit evidence was not verified: {evidence_path}"
                )
            if not evidence_path.is_file():
                raise BenchmarkContractError(
                    f"benchmark audit evidence file is missing: {evidence_path}"
                )
            if sha256_file(evidence_path) != evidence.sha256:
                raise BenchmarkContractError(
                    f"benchmark audit evidence hash mismatch: {evidence_path}"
                )
    return audit_path, audit


def _verify_bundle_manifest(
    bundle_root: Path,
    binding: AlfworldTrainingBenchmarkBinding,
) -> None:
    bundle_manifest_path = bundle_root / "alfworld_training_bundle_manifest.json"
    if not bundle_manifest_path.is_file():
        raise BenchmarkContractError(f"ALFWorld bundle manifest is missing: {bundle_manifest_path}")
    try:
        raw = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise BenchmarkContractError(f"invalid ALFWorld bundle manifest: {error}") from error
    if raw.get("schema") != ALFWORLD_TRAINING_BUNDLE_SCHEMA:
        raise BenchmarkContractError("unsupported ALFWorld training bundle manifest schema")
    if raw.get("collection_id") != binding.collection_id:
        raise BenchmarkContractError("ALFWorld bundle collection_id does not match binding")
    expected_hashes = raw.get("artifact_sha256")
    if not isinstance(expected_hashes, dict):
        raise BenchmarkContractError("ALFWorld bundle manifest has no artifact hashes")
    for name, expected_hash in expected_hashes.items():
        artifact = bundle_root / str(name)
        if not artifact.is_file():
            raise BenchmarkContractError(f"ALFWorld bundle artifact is missing: {artifact}")
        if sha256_file(artifact) != expected_hash:
            raise BenchmarkContractError(f"ALFWorld bundle artifact hash mismatch: {artifact}")


def _verify_collection_manifest(
    bundle_root: Path,
    binding: AlfworldTrainingBenchmarkBinding,
) -> None:
    path = bundle_root / "train_eval_collection_manifest.json"
    if not path.is_file():
        raise BenchmarkContractError(f"ALFWorld collection manifest is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise BenchmarkContractError(f"invalid ALFWorld collection manifest: {error}") from error
    if raw.get("schema") != TRAIN_EVAL_COLLECTION_SCHEMA:
        raise BenchmarkContractError("unsupported ALFWorld collection manifest schema")
    expected = {
        "collection_id": binding.collection_id,
        "dataset_id": binding.dataset_id,
        "source_record_count": binding.collection_record_count,
        "source_records_sha256": binding.source_records_sha256,
        "split_strategy": binding.collection_split_strategy,
        "holdout_fraction": binding.collection_holdout_fraction,
    }
    mismatches = [name for name, value in expected.items() if raw.get(name) != value]
    train = raw.get("train") if isinstance(raw.get("train"), dict) else {}
    evaluation = raw.get("evaluation") if isinstance(raw.get("evaluation"), dict) else {}
    if train.get("record_count") != binding.train_record_count:
        mismatches.append("train.record_count")
    if train.get("records_sha256") != binding.train_records_sha256:
        mismatches.append("train.records_sha256")
    if evaluation.get("record_count") != binding.evaluation_record_count:
        mismatches.append("evaluation.record_count")
    if evaluation.get("records_sha256") != binding.evaluation_records_sha256:
        mismatches.append("evaluation.records_sha256")
    if mismatches:
        raise BenchmarkContractError(
            "ALFWorld collection manifest and binding disagree on: " + ", ".join(mismatches)
        )


def _verify_partition_records(
    records: list[TrainingTrajectoryRecord],
    *,
    expected_count: int,
    expected_sha256: str,
    partition_name: str,
    expected_dataset_id: str,
) -> None:
    if len(records) != expected_count:
        raise BenchmarkContractError(
            f"ALFWorld {partition_name} partition count mismatch: expected {expected_count}, got {len(records)}"
        )
    if {record.dataset_id for record in records} != {expected_dataset_id}:
        raise BenchmarkContractError(
            f"ALFWorld {partition_name} partition dataset_id does not match binding"
        )
    if {record.benchmark for record in records} != {"alfworld_native"}:
        raise BenchmarkContractError(
            f"ALFWorld {partition_name} partition contains a non-ALFWorld benchmark record"
        )
    if records_sha256(records) != expected_sha256:
        raise BenchmarkContractError(
            f"ALFWorld {partition_name} partition content hash does not match binding"
        )


def _verify_published_binding_evidence(
    binding: AlfworldTrainingBenchmarkBinding,
) -> AlfworldTaskManifest:
    manifest_file, manifest = _load_task_manifest(binding.task_manifest_path)
    manifest_sha256 = sha256_file(manifest_file)
    if manifest_sha256 != binding.task_manifest_sha256:
        raise BenchmarkContractError("published ALFWorld task manifest hash changed after binding")
    if len(manifest.tasks) != binding.task_manifest_task_count:
        raise BenchmarkContractError("published ALFWorld task manifest count changed after binding")
    if not binding.alfworld_data_root or not binding.alfworld_data_artifacts_sha256:
        raise BenchmarkContractError("published ALFWorld binding lacks data-root artifact evidence")
    _, current_data_hash = _verify_alfworld_data_root(manifest, binding.alfworld_data_root)
    if current_data_hash != binding.alfworld_data_artifacts_sha256:
        raise BenchmarkContractError("published ALFWorld data artifacts changed after binding")
    if not binding.benchmark_audit_path or not binding.benchmark_audit_sha256:
        raise BenchmarkContractError("published ALFWorld binding lacks benchmark audit evidence")
    audit_file, audit = _load_verified_benchmark_audit(
        binding.benchmark_audit_path,
        expected_manifest_sha256=manifest_sha256,
    )
    if sha256_file(audit_file) != binding.benchmark_audit_sha256:
        raise BenchmarkContractError("published benchmark audit changed after binding")
    if audit.source_commit != binding.benchmark_audit_source_commit:
        raise BenchmarkContractError(
            "published benchmark audit source commit changed after binding"
        )
    return manifest


def _declared_benchmarks(dataset: Path) -> set[str]:
    values: set[str] = set()
    try:
        with dataset.open("r", encoding="utf-8") as handle:
            for _line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("row is not a JSON object")
                benchmark = raw.get("benchmark")
                if benchmark is not None:
                    values.add(str(benchmark))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BenchmarkContractError(
            f"cannot inspect training dataset benchmark identities at {dataset}: {error}"
        ) from error
    return values
