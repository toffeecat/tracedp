"""Fail-closed bridge from LH runner trajectories to the official SWE-bench harness.

The generic :mod:`swebench_lite` adapter deliberately does not execute tests.
That is correct: a unified diff being present, or even applying cleanly, is not
evidence that an issue was fixed.  This module owns the boundary that the
official project owns:

``runner trajectory -> official predictions.jsonl -> run_evaluation receipt``

It does not manufacture a pass rate.  A receipt is marked ``real_native`` only
after the upstream ``swebench.harness.run_evaluation`` process returns zero
*and* writes a complete official run report for the exact prediction roster.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any

from lh_platform.benchmarks.swebench_lite import downloaded_manifest_path, validate_full_download
from lh_platform.core.artifact_redaction import redact_artifact
from lh_platform.core.evidence_contract import (
    EXPERIMENT_EVIDENCE_SCHEMA,
    RUNNER_TRAJECTORY_MANIFEST_SCHEMA,
    canonical_sha256,
)
from lh_platform.core.generation_authority import GENERATION_AUTHORITY_SCHEMA
from lh_platform.models.base import (
    MODEL_GENERATION_EVENT_SCHEMA,
    generation_event_sha256,
    sha256_text,
)

# The current upstream SWE-bench README and maintained Hugging Face collection
# identify Lite under the SWE-bench organization.  The final evaluator normally
# consumes our frozen JSONL, but this label remains the canonical public
# reference for diagnostic subset runs and for every evidence lock.
OFFICIAL_DATASET_NAME = "SWE-bench/SWE-bench_Lite"
OFFICIAL_SPLIT = "test"
OFFICIAL_HARNESS_MODULE = "swebench.harness.run_evaluation"
OFFICIAL_PREDICTION_FIELDS = (
    "instance_id",
    "model_name_or_path",
    "model_patch",
)
PREDICTION_BUNDLE_SCHEMA = "lh_swebench_prediction_bundle/v1"
OFFICIAL_EVALUATION_RECEIPT_SCHEMA = "lh_swebench_official_evaluation_receipt/v1"
ACTION_MODEL_GENERATION_SCHEMA = "lh_action_model_generation/v3"
MODEL_RESPONSE_BINDING_SCHEMA = "lh_model_response_binding/v1"
FULL_ROSTER_LOCK_SCHEMA = "lh_swebench_full_roster_lock/v1"
HARNESS_IDENTITY_LOCK_SCHEMA = "lh_swebench_harness_identity_lock/v1"
CANONICAL_DATASET_ANCHOR_SCHEMA = "lh_swebench_canonical_dataset_anchor/v1"
SWE_BENCH_LITE_EXPECTED_INSTANCES = 300
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FENCED_DIFF_RE = re.compile(
    r"^\s*```(?:diff|patch)?\s*\n(?P<patch>.*?)\n?```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_DATASET_ANCHOR_RELATIVE_PATH = Path("configs/locks/swebench_lite_canonical_v1.json")


class SWEbenchOfficialBridgeError(RuntimeError):
    """Base error for an invalid official-evaluation bridge artifact."""


class SWEbenchPredictionExportError(SWEbenchOfficialBridgeError):
    """Raised when a runner artifact cannot be honestly exported as a patch."""


@dataclass(frozen=True)
class ExtractedPatch:
    """One patch plus the runner-owned action that produced it."""

    patch: str
    action: Mapping[str, Any]
    source: str


@dataclass(frozen=True)
class PredictionBundle:
    """The official-format predictions and provenance sidecar produced from one run."""

    predictions_path: Path
    receipt_path: Path
    prediction_count: int
    prediction_sha256: str
    source_run_id: str
    source_bundle_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["predictions_path"] = str(self.predictions_path)
        payload["receipt_path"] = str(self.receipt_path)
        payload["source_bundle_path"] = str(self.source_bundle_path)
        return payload


@dataclass(frozen=True)
class VerifiedPredictionInput:
    """Source binding available for one predictions file at evaluation time."""

    receipt_path: Path | None
    input_provenance: str
    source_contract: Mapping[str, Any] | None
    source_run_dir: Path | None


@dataclass(frozen=True)
class EvaluationScope:
    """Whether a requested roster is a comparison-grade full Lite evaluation."""

    scope: str
    comparison_eligible: bool
    reason: str
    canonical_roster_lock_path: Path | None = None
    canonical_roster_sha256: str | None = None
    canonical_roster_lock_sha256: str | None = None
    canonical_dataset_checksum: str | None = None
    canonical_source_revision: str | None = None
    canonical_dataset_anchor_path: Path | None = None
    canonical_dataset_anchor_sha256: str | None = None
    canonical_dataset_anchor_status: str = "not_checked"


@dataclass(frozen=True)
class CanonicalDatasetAnchorVerification:
    """Trust status of the repository-reviewed full Lite snapshot."""

    status: str
    path: Path
    reason: str
    sha256: str | None = None


@dataclass(frozen=True)
class FrozenOfficialDataset:
    """A checksum-verified full Lite JSONL copied into the evaluation host."""

    data_path: Path
    manifest_path: Path
    dataset_checksum: str
    dataset_revision: str
    dataset_manifest_sha256: str


@dataclass(frozen=True)
class HarnessIdentityVerification:
    """Audited identity facts for the exact evaluator process being launched."""

    status: str
    lock_path: Path | None
    lock_sha256: str | None = None
    source_commit: str | None = None


@dataclass(frozen=True)
class OfficialEvaluationReceipt:
    """Observed outcome of one requested upstream SWE-bench evaluation."""

    receipt_path: Path
    status: str
    execution_tier: str
    real_native_completed: bool
    command: tuple[str, ...]
    source_predictions_path: Path
    predictions_path: Path
    prediction_sha256: str
    evaluator_dataset_path: Path | None
    evaluator_dataset_manifest_path: Path | None
    input_provenance: str
    prediction_bundle_receipt_path: Path | None
    evaluation_scope: str
    comparison_eligible: bool
    scope_reason: str
    canonical_roster_lock_path: Path | None
    canonical_roster_sha256: str | None
    canonical_roster_lock_sha256: str | None
    canonical_dataset_checksum: str | None
    canonical_source_revision: str | None
    canonical_dataset_anchor_path: Path | None
    canonical_dataset_anchor_sha256: str | None
    canonical_dataset_anchor_status: str
    harness_identity_status: str
    harness_identity: Mapping[str, Any] | None
    harness_identity_lock_path: Path | None
    harness_identity_lock_sha256: str | None
    harness_source_commit: str | None
    source_generation: Mapping[str, Any] | None
    official_harness_completed: bool
    run_id: str
    output_dir: Path
    preflight: Mapping[str, Any]
    returncode: int | None = None
    report_path: Path | None = None
    report_sha256: str | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    report_summary: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "receipt_path",
            "source_predictions_path",
            "predictions_path",
            "evaluator_dataset_path",
            "evaluator_dataset_manifest_path",
            "prediction_bundle_receipt_path",
            "canonical_roster_lock_path",
            "canonical_dataset_anchor_path",
            "harness_identity_lock_path",
            "output_dir",
            "report_path",
            "stdout_path",
            "stderr_path",
        ):
            value = payload.get(key)
            if value is not None:
                payload[key] = str(value)
        payload["command"] = list(self.command)
        payload["schema"] = OFFICIAL_EVALUATION_RECEIPT_SCHEMA
        payload = redact_artifact(payload)
        payload["artifact_sha256"] = canonical_sha256(payload)
        return payload


def export_predictions_from_run(
    run_dir: str | Path,
    predictions_path: str | Path,
) -> PredictionBundle:
    """Export every manifested SWE-bench task in a completed runner run.

    The exporter intentionally fails when even one manifested trajectory lacks
    a valid unified diff.  The official harness accepts partial prediction
    files, but silently dropping failed task generations would change the
    evaluation denominator and make an agent look better than it is.
    """

    source_root = Path(run_dir).expanduser().resolve()
    target = Path(predictions_path).expanduser().resolve()
    contract = _load_completed_swebench_contract(source_root)
    entries = _runner_trajectory_entries(contract, source_root)
    _validate_prediction_export_target(source_root, target, entries)
    model_name = _resolve_model_name(contract)

    predictions: list[dict[str, str]] = []
    missing: list[str] = []
    for entry in entries:
        trajectory = _load_verified_trajectory(source_root, entry)
        extracted = _extract_patch(trajectory)
        if extracted is None:
            missing.append(str(entry["task_id"]))
            continue
        _validate_task_patch_generation(
            trajectory,
            extracted,
            task_id=str(entry["task_id"]),
            model_name=model_name,
        )
        predictions.append(
            {
                "instance_id": str(entry["task_id"]),
                "model_name_or_path": model_name,
                "model_patch": extracted.patch,
            }
        )
    if missing:
        raise SWEbenchPredictionExportError(
            "missing valid patch for manifested SWE-bench task(s): " + ", ".join(sorted(missing))
        )

    _validate_prediction_rows(predictions)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    source_bundle_path = target.with_suffix(target.suffix + ".sources")
    _validate_prediction_bundle_destination(target, receipt_path, source_bundle_path)
    staged_bundle = _materialize_source_bundle(source_root, entries, source_bundle_path)
    published_bundle = False
    try:
        _write_jsonl_atomic(target, predictions, overwrite=False)
        # The receipt is the publish marker: until it exists, evaluators reject
        # this artifact.  Publishing the bundle after winning the prediction
        # path prevents a second exporter from replacing its source evidence.
        os.replace(staged_bundle, source_bundle_path)
        published_bundle = True
        digest = _sha256_file(target)
        bundle_receipt = {
            "schema": PREDICTION_BUNDLE_SCHEMA,
            "created_at": time.time(),
            "source_run_id": str(contract["run_id"]),
            "source_bundle_path": source_bundle_path.name,
            "source_evidence_contract": "evidence_contract.json",
            "source_evidence_contract_sha256": str(contract["artifact_sha256"]),
            "source_trajectory_manifest_schema": RUNNER_TRAJECTORY_MANIFEST_SCHEMA,
            "source_trajectory_entries": [
                {
                    "path": str(entry["path"]),
                    "task_id": str(entry["task_id"]),
                    "sha256": str(entry["sha256"]),
                }
                for entry in entries
            ],
            "predictions_path": str(target),
            "prediction_sha256": digest,
            "prediction_count": len(predictions),
            "instance_ids": [row["instance_id"] for row in predictions],
            "model_name_or_path": model_name,
            "source_model_identity": _source_model_identity(contract, model_name),
            "missing_instance_ids": [],
            "claim_scope": "prediction_artifact_only",
            "note": (
                "This bundle proves only a local trajectory-to-prediction export. "
                "It contains no SWE-bench pass rate; use run_official_evaluation "
                "and preserve its official receipt. The sibling .sources directory "
                "is required when moving this export to an evaluation host."
            ),
        }
        bundle_receipt["artifact_sha256"] = canonical_sha256(bundle_receipt)
        _write_json_atomic(receipt_path, bundle_receipt, overwrite=False)
    except Exception:
        # A prediction file without its receipt is intentionally unusable by
        # default.  Do not leave a half-export that another run could mistake
        # for a complete provenance bundle.
        target.unlink(missing_ok=True)
        if published_bundle:
            shutil.rmtree(source_bundle_path, ignore_errors=True)
        else:
            shutil.rmtree(staged_bundle, ignore_errors=True)
        raise
    return PredictionBundle(
        predictions_path=target,
        receipt_path=receipt_path,
        prediction_count=len(predictions),
        prediction_sha256=digest,
        source_run_id=str(contract["run_id"]),
        source_bundle_path=source_bundle_path,
    )


def validate_official_prediction_file(path: str | Path) -> list[dict[str, str]]:
    """Validate a homogeneous official-format predictions JSONL file.

    The upstream evaluator can accept arbitrary extra fields and partial task
    sets.  The platform refuses both here so the resulting receipt states one
    exact model and one exact evaluated roster.
    """

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise SWEbenchOfficialBridgeError(f"predictions file not found: {target}")
    rows: list[dict[str, str]] = []
    for line_no, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise SWEbenchOfficialBridgeError(
                f"predictions JSONL line {line_no} is invalid JSON"
            ) from error
        if not isinstance(raw, Mapping):
            raise SWEbenchOfficialBridgeError(
                f"predictions JSONL line {line_no} must be a JSON object"
            )
        keys = set(raw)
        expected = set(OFFICIAL_PREDICTION_FIELDS)
        if keys != expected:
            raise SWEbenchOfficialBridgeError(
                f"predictions JSONL line {line_no} must contain exactly "
                f"{sorted(expected)}; received {sorted(keys)}"
            )
        rows.append({key: str(raw[key]) for key in OFFICIAL_PREDICTION_FIELDS})
    _validate_prediction_rows(rows)
    return rows


def _verify_prediction_bundle_receipt(
    predictions_path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    allow_unbound_predictions: bool,
) -> VerifiedPredictionInput:
    """Bind an evaluation request to the runner export unless opt-out is explicit.

    An upstream evaluator can validly score an external predictions file, but
    an experiment platform must not imply that such a file came from the
    configured model/method.  Requiring the export receipt by default keeps
    those two cases distinct while preserving an explicit audit lane for
    externally generated submissions.
    """

    receipt_path = predictions_path.with_suffix(predictions_path.suffix + ".receipt.json")
    if not receipt_path.is_file():
        if allow_unbound_predictions:
            return VerifiedPredictionInput(
                receipt_path=None,
                input_provenance="external_unbound",
                source_contract=None,
                source_run_dir=None,
            )
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt is required for platform evaluation; "
            "use export-swebench-predictions or explicitly allow unbound predictions"
        )
    raw = _read_json_mapping(receipt_path, label="prediction bundle receipt")
    if raw.get("schema") != PREDICTION_BUNDLE_SCHEMA:
        raise SWEbenchOfficialBridgeError("unexpected prediction bundle receipt schema")
    declared = str(raw.get("artifact_sha256") or "")
    calculated = canonical_sha256(
        {key: value for key, value in raw.items() if key != "artifact_sha256"}
    )
    if declared != calculated:
        raise SWEbenchOfficialBridgeError("prediction bundle receipt artifact_sha256 mismatch")
    if str(raw.get("prediction_sha256") or "") != _sha256_file(predictions_path):
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt does not match predictions bytes"
        )
    expected_ids = [str(row["instance_id"]) for row in rows]
    if raw.get("prediction_count") != len(rows):
        raise SWEbenchOfficialBridgeError("prediction bundle receipt prediction_count mismatch")
    if raw.get("instance_ids") != expected_ids:
        raise SWEbenchOfficialBridgeError("prediction bundle receipt instance_ids mismatch")
    if raw.get("missing_instance_ids") != []:
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt declares missing task predictions"
        )
    models = {str(row["model_name_or_path"]) for row in rows}
    if raw.get("model_name_or_path") not in models:
        raise SWEbenchOfficialBridgeError("prediction bundle receipt model_name_or_path mismatch")
    if not str(raw.get("source_run_id") or "").strip():
        raise SWEbenchOfficialBridgeError("prediction bundle receipt lacks source_run_id")
    if raw.get("source_evidence_contract") != "evidence_contract.json":
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt has an unexpected source evidence-contract path"
        )
    if len(str(raw.get("source_evidence_contract_sha256") or "")) != 64:
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt lacks source evidence-contract hash"
        )
    if raw.get("source_trajectory_manifest_schema") != RUNNER_TRAJECTORY_MANIFEST_SCHEMA:
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt has an unexpected source trajectory-manifest schema"
        )
    entries = raw.get("source_trajectory_entries")
    if not isinstance(entries, list) or len(entries) != len(rows):
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt trajectory roster does not match predictions"
        )
    if not all(
        isinstance(entry, Mapping)
        and str(entry.get("path") or "")
        and str(entry.get("task_id") or "")
        and len(str(entry.get("sha256") or "")) == 64
        for entry in entries
    ):
        raise SWEbenchOfficialBridgeError("prediction bundle receipt trajectory entry is malformed")
    receipt_ids = [str(entry.get("task_id") or "") for entry in entries]
    if receipt_ids != expected_ids:
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt trajectory task IDs do not match predictions"
        )
    bundle_name = _safe_source_bundle_name(raw)
    source_run_dir = (predictions_path.parent / bundle_name).resolve()
    if source_run_dir.is_symlink() or not source_run_dir.is_dir():
        raise SWEbenchOfficialBridgeError(
            "prediction bundle portable source bundle is missing beside predictions"
        )
    try:
        source_contract = _load_completed_swebench_contract(source_run_dir)
        source_entries = _runner_trajectory_entries(source_contract, source_run_dir)
        source_patches: list[dict[str, str]] = []
        source_model = _resolve_model_name(source_contract)
        for source_entry in source_entries:
            trajectory = _load_verified_trajectory(source_run_dir, source_entry)
            extracted = _extract_patch(trajectory)
            if extracted is None:
                raise SWEbenchOfficialBridgeError(
                    "source trajectory no longer contains a valid unified-diff patch: "
                    + str(source_entry["task_id"])
                )
            try:
                _validate_task_patch_generation(
                    trajectory,
                    extracted,
                    task_id=str(source_entry["task_id"]),
                    model_name=source_model,
                )
            except SWEbenchPredictionExportError as error:
                raise SWEbenchOfficialBridgeError(
                    "source trajectory generation evidence is invalid: " + str(error)
                ) from error
            source_patches.append(
                {
                    "instance_id": str(source_entry["task_id"]),
                    "model_name_or_path": source_model,
                    "model_patch": extracted.patch,
                }
            )
    except SWEbenchOfficialBridgeError as error:
        raise SWEbenchOfficialBridgeError(
            "prediction bundle source run cannot be re-verified: " + str(error)
        ) from error
    if str(source_contract.get("artifact_sha256") or "") != str(
        raw.get("source_evidence_contract_sha256") or ""
    ):
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt source evidence-contract hash no longer matches source run"
        )
    expected_entries = [
        {
            "path": str(entry["path"]),
            "task_id": str(entry["task_id"]),
            "sha256": str(entry["sha256"]),
        }
        for entry in source_entries
    ]
    if raw.get("source_trajectory_entries") != expected_entries:
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt trajectory manifest no longer matches source run"
        )
    if source_model != str(raw.get("model_name_or_path") or ""):
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt model label no longer matches source run"
        )
    if list(rows) != source_patches:
        raise SWEbenchOfficialBridgeError(
            "prediction rows do not exactly match patches re-extracted from source trajectories"
        )
    return VerifiedPredictionInput(
        receipt_path=receipt_path,
        input_provenance="runner_bound",
        source_contract=source_contract,
        source_run_dir=source_run_dir,
    )


def _freeze_prediction_input_bundle(
    source_predictions_path: Path,
    output_root: Path,
    *,
    allow_unbound_predictions: bool,
) -> Path:
    """Copy the minimum evaluation input into a fresh immutable work area.

    The copied bundle, rather than the caller-owned source path, is later
    provenance-verified and supplied to the upstream evaluator.  Copying only
    the receipt-declared files also rejects symlink traversal and prevents a
    source directory from smuggling unrelated state into the evaluation host.
    """

    input_root = output_root / "input"
    input_root.mkdir(exist_ok=False)
    frozen_predictions = input_root / "predictions.jsonl"
    _copy_plain_file(source_predictions_path, frozen_predictions, label="predictions")

    source_receipt = source_predictions_path.with_suffix(
        source_predictions_path.suffix + ".receipt.json"
    )
    if not source_receipt.is_file():
        if allow_unbound_predictions:
            return frozen_predictions
        # Let the canonical verifier produce the public error wording after
        # freezing the user-provided prediction bytes.
        return frozen_predictions

    frozen_receipt = frozen_predictions.with_suffix(frozen_predictions.suffix + ".receipt.json")
    _copy_plain_file(source_receipt, frozen_receipt, label="prediction bundle receipt")
    receipt = _read_json_mapping(frozen_receipt, label="prediction bundle receipt")
    bundle_name = _safe_source_bundle_name(receipt)
    source_bundle = (source_predictions_path.parent / bundle_name).resolve()
    if source_bundle.is_symlink() or not source_bundle.is_dir():
        raise SWEbenchOfficialBridgeError(
            "prediction bundle portable source bundle is missing or unsafe"
        )
    entries = receipt.get("source_trajectory_entries")
    if not isinstance(entries, list):
        raise SWEbenchOfficialBridgeError(
            "prediction bundle receipt lacks a safe source trajectory manifest"
        )
    frozen_bundle = input_root / bundle_name
    frozen_bundle.mkdir(exist_ok=False)
    source_contract = source_bundle / "evidence_contract.json"
    _copy_plain_file(
        source_contract, frozen_bundle / "evidence_contract.json", label="source evidence contract"
    )
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SWEbenchOfficialBridgeError(
                "prediction bundle receipt source trajectory entry is malformed"
            )
        path = str(entry.get("path") or "")
        source_trajectory = _safe_relative_task_path(source_bundle, path)
        _copy_plain_file(
            source_trajectory,
            _safe_relative_task_path(frozen_bundle, path),
            label="source trajectory",
        )
    return frozen_predictions


def _safe_source_bundle_name(receipt: Mapping[str, Any]) -> str:
    source_bundle_raw = str(receipt.get("source_bundle_path") or "").strip()
    bundle_relative = PurePosixPath(source_bundle_raw)
    if (
        not source_bundle_raw
        or bundle_relative.is_absolute()
        or ".." in bundle_relative.parts
        or len(bundle_relative.parts) != 1
        or not bundle_relative.name.endswith(".sources")
    ):
        raise SWEbenchOfficialBridgeError("prediction bundle receipt source bundle path is unsafe")
    return bundle_relative.name


def _copy_plain_file(source: Path, target: Path, *, label: str) -> None:
    """Copy one regular file without following an attacker-controlled symlink."""

    if source.is_symlink() or not source.is_file():
        raise SWEbenchOfficialBridgeError(f"{label} is missing or unsafe: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if target.is_symlink() or not target.is_file():
        raise SWEbenchOfficialBridgeError(f"failed to freeze {label}: {source}")


def _freeze_official_dataset_input(
    source_path_raw: str | Path | None,
    source_manifest_raw: str | Path | None,
    output_root: Path,
) -> FrozenOfficialDataset | None:
    """Freeze an audited full Lite download for the upstream evaluator.

    Upstream SWE-bench accepts local JSONL files.  For a comparison-grade run
    that is necessary: the Docker evaluator must grade the exact bytes that
    the generating run and full-roster lock describe, rather than whatever a
    mutable Hugging Face alias happens to resolve to on evaluation day.
    """

    if source_path_raw is None:
        return None
    source_path = Path(source_path_raw).expanduser().resolve()
    source_manifest = (
        Path(source_manifest_raw).expanduser().resolve()
        if source_manifest_raw is not None
        else downloaded_manifest_path(source_path)
    )
    valid, reason, metadata = validate_full_download(source_path, source_manifest)
    if not valid or metadata is None:
        raise SWEbenchOfficialBridgeError(
            "official dataset path is not an audited full SWE-bench Lite download: " + reason
        )
    input_root = output_root / "input"
    frozen_path = input_root / "swe-bench-lite.jsonl"
    frozen_manifest = downloaded_manifest_path(frozen_path)
    _copy_plain_file(source_path, frozen_path, label="official SWE-bench Lite JSONL")
    _copy_plain_file(source_manifest, frozen_manifest, label="official SWE-bench Lite manifest")
    copied_valid, copied_reason, copied_metadata = validate_full_download(
        frozen_path,
        frozen_manifest,
    )
    if not copied_valid or copied_metadata is None:
        raise SWEbenchOfficialBridgeError(
            "frozen official dataset failed its integrity check: " + copied_reason
        )
    return FrozenOfficialDataset(
        data_path=frozen_path,
        manifest_path=frozen_manifest,
        dataset_checksum=str(copied_metadata["dataset_checksum"]),
        dataset_revision=str(copied_metadata["dataset_revision"]),
        dataset_manifest_sha256=str(copied_metadata["dataset_manifest_sha256"]),
    )


def build_official_evaluation_command(
    predictions_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str,
    max_workers: int = 1,
    timeout_s: int = 1_800,
    dataset_name: str = OFFICIAL_DATASET_NAME,
    split: str = OFFICIAL_SPLIT,
    python_executable: str | Path | None = None,
    _allow_frozen_dataset: bool = False,
) -> tuple[str, ...]:
    """Build the upstream CLI exactly, pinning its denominator to prediction IDs."""

    executable = _require_canonical_interpreter(python_executable)
    _validate_command_dataset_name(
        dataset_name=dataset_name,
        split=split,
        output_dir=output_dir,
        allow_frozen_dataset=_allow_frozen_dataset,
    )
    rows = validate_official_prediction_file(predictions_path)
    _validate_run_id(run_id)
    if max_workers < 1:
        raise SWEbenchOfficialBridgeError("max_workers must be >= 1")
    if timeout_s < 1:
        raise SWEbenchOfficialBridgeError("timeout_s must be >= 1")
    prediction_target = Path(predictions_path).expanduser().resolve()
    report_root = Path(output_dir).expanduser().resolve()
    instance_ids = [row["instance_id"] for row in rows]
    return tuple(
        [
            executable,
            "-m",
            OFFICIAL_HARNESS_MODULE,
            "--dataset_name",
            str(dataset_name),
            "--split",
            OFFICIAL_SPLIT,
            "--predictions_path",
            str(prediction_target),
            "--max_workers",
            str(max_workers),
            "--timeout",
            str(timeout_s),
            "--run_id",
            run_id,
            "--report_dir",
            str(report_root),
            "--instance_ids",
            *instance_ids,
        ]
    )


def _require_canonical_public_target(
    *,
    dataset_name: str,
    split: str,
    python_executable: str | Path | None,
) -> str:
    """Reject API-level substitutions that could masquerade as SWE-bench Lite.

    The upstream command deliberately supports custom datasets and arbitrary
    Python environments.  This bridge does not: its ``real_native`` label is
    reserved for the public Lite test split and for the interpreter whose
    evaluator bytes are fingerprinted by the harness identity lock.
    """

    if dataset_name != OFFICIAL_DATASET_NAME or split != OFFICIAL_SPLIT:
        raise SWEbenchOfficialBridgeError(
            "official bridge only evaluates "
            f"dataset_name={OFFICIAL_DATASET_NAME!r}, split={OFFICIAL_SPLIT!r}"
        )
    return _require_canonical_interpreter(python_executable)


def _require_canonical_interpreter(python_executable: str | Path | None) -> str:
    """Return only the process interpreter whose package bytes are locked."""

    current = Path(sys.executable).expanduser().resolve()
    if python_executable is not None:
        requested = Path(python_executable).expanduser().resolve()
        if requested != current:
            raise SWEbenchOfficialBridgeError(
                "python_executable override is forbidden because the harness identity "
                "lock fingerprints the current interpreter"
            )
    return str(current)


def _validate_command_dataset_name(
    *,
    dataset_name: str,
    split: str,
    output_dir: str | Path,
    allow_frozen_dataset: bool,
) -> None:
    """Permit only the public Lite name or our freshly frozen local copy."""

    if split != OFFICIAL_SPLIT:
        raise SWEbenchOfficialBridgeError(
            f"official bridge only evaluates split={OFFICIAL_SPLIT!r}"
        )
    if dataset_name == OFFICIAL_DATASET_NAME:
        return
    expected = Path(output_dir).expanduser().resolve() / "input" / "swe-bench-lite.jsonl"
    candidate = Path(dataset_name).expanduser().resolve()
    if not allow_frozen_dataset or candidate != expected:
        raise SWEbenchOfficialBridgeError(
            "official bridge may only pass its own frozen SWE-bench Lite JSONL to the harness"
        )


def run_official_evaluation(
    predictions_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str,
    max_workers: int = 1,
    timeout_s: int = 1_800,
    dataset_name: str = OFFICIAL_DATASET_NAME,
    split: str = OFFICIAL_SPLIT,
    execute: bool = False,
    allow_unbound_predictions: bool = False,
    full_roster_lock_path: str | Path | None = None,
    harness_identity_lock_path: str | Path | None = None,
    official_dataset_path: str | Path | None = None,
    official_dataset_manifest_path: str | Path | None = None,
    process_timeout_s: float | None = None,
    python_executable: str | Path | None = None,
) -> OfficialEvaluationReceipt:
    """Plan or execute the official upstream evaluator and write a receipt.

    ``execute=False`` is intentionally the default: local macOS development
    and the current VAST container have no supported Docker evaluation setup.
    It still validates the exact prediction artifact and writes the command a
    Docker-capable host must execute.  ``execute=True`` never falls back to a
    heuristic evaluator when preflight fails.
    """

    _require_canonical_public_target(
        dataset_name=dataset_name,
        split=split,
        python_executable=python_executable,
    )
    source_prediction_target = Path(predictions_path).expanduser().resolve()
    if not source_prediction_target.is_file():
        raise SWEbenchOfficialBridgeError(f"predictions file not found: {source_prediction_target}")
    output_root = Path(output_dir).expanduser().resolve()
    _prepare_fresh_output_dir(output_root, reserve_execution=execute)
    # Do not hand the upstream harness a mutable caller-owned file.  Freeze
    # the prediction bundle inside the fresh evaluation directory first, then
    # verify *that copy* and use it as the only command input.  This closes the
    # check-to-use window in which a valid local patch file could be replaced
    # with a different patch after provenance verification.
    prediction_target = _freeze_prediction_input_bundle(
        source_prediction_target,
        output_root,
        allow_unbound_predictions=allow_unbound_predictions,
    )
    rows = validate_official_prediction_file(prediction_target)
    binding = _verify_prediction_bundle_receipt(
        prediction_target,
        rows,
        allow_unbound_predictions=allow_unbound_predictions,
    )
    if official_dataset_path is None and binding.source_contract is not None:
        source_benchmark = binding.source_contract.get("benchmark")
        if isinstance(source_benchmark, Mapping) and source_benchmark.get("dataset_path"):
            candidate = Path(str(source_benchmark["dataset_path"])).expanduser()
            if candidate.is_file():
                official_dataset_path = candidate
                if official_dataset_manifest_path is None:
                    manifest_candidate = source_benchmark.get("dataset_manifest_path")
                    official_dataset_manifest_path = (
                        Path(str(manifest_candidate)).expanduser()
                        if manifest_candidate
                        else downloaded_manifest_path(candidate)
                    )
    frozen_dataset = _freeze_official_dataset_input(
        official_dataset_path,
        official_dataset_manifest_path,
        output_root,
    )
    scope = _classify_evaluation_scope(
        rows,
        binding,
        frozen_dataset=frozen_dataset,
        full_roster_lock_path=full_roster_lock_path,
    )
    _validate_run_id(run_id)
    command = build_official_evaluation_command(
        prediction_target,
        output_dir=output_root,
        run_id=run_id,
        max_workers=max_workers,
        timeout_s=timeout_s,
        dataset_name=(
            str(frozen_dataset.data_path) if frozen_dataset is not None else OFFICIAL_DATASET_NAME
        ),
        split=OFFICIAL_SPLIT,
        python_executable=None,
        _allow_frozen_dataset=frozen_dataset is not None,
    )
    receipt_path = output_root / "official_evaluation_receipt.json"
    prediction_sha256 = _sha256_file(prediction_target)
    common = {
        "receipt_path": receipt_path,
        "command": command,
        "source_predictions_path": source_prediction_target,
        "predictions_path": prediction_target,
        "prediction_sha256": prediction_sha256,
        "evaluator_dataset_path": (
            frozen_dataset.data_path if frozen_dataset is not None else None
        ),
        "evaluator_dataset_manifest_path": (
            frozen_dataset.manifest_path if frozen_dataset is not None else None
        ),
        "input_provenance": binding.input_provenance,
        "prediction_bundle_receipt_path": binding.receipt_path,
        "evaluation_scope": scope.scope,
        "comparison_eligible": scope.comparison_eligible,
        "scope_reason": scope.reason,
        "canonical_roster_lock_path": scope.canonical_roster_lock_path,
        "canonical_roster_sha256": scope.canonical_roster_sha256,
        "canonical_roster_lock_sha256": scope.canonical_roster_lock_sha256,
        "canonical_dataset_checksum": scope.canonical_dataset_checksum,
        "canonical_source_revision": scope.canonical_source_revision,
        "canonical_dataset_anchor_path": scope.canonical_dataset_anchor_path,
        "canonical_dataset_anchor_sha256": scope.canonical_dataset_anchor_sha256,
        "canonical_dataset_anchor_status": scope.canonical_dataset_anchor_status,
        "source_generation": (
            _source_generation_identity(binding.source_contract)
            if binding.source_contract is not None
            else None
        ),
        "harness_identity_lock_sha256": None,
        "harness_source_commit": None,
        "run_id": run_id,
        "output_dir": output_root,
    }

    def write_receipt(**kwargs: Any) -> OfficialEvaluationReceipt:
        receipt = OfficialEvaluationReceipt(**{**common, **kwargs})
        _write_json_atomic(receipt_path, receipt.to_dict(), overwrite=False)
        return receipt

    if not execute:
        return write_receipt(
            status="planned",
            execution_tier="not_evaluated",
            real_native_completed=False,
            official_harness_completed=False,
            harness_identity_status="not_checked",
            harness_identity=None,
            harness_identity_lock_path=(
                Path(harness_identity_lock_path).expanduser().resolve()
                if harness_identity_lock_path is not None
                else None
            ),
            preflight={"ok": None, "missing": [], "skipped": "execute=false"},
        )

    preflight = _official_runtime_preflight()
    harness_identity = _observed_harness_identity(preflight)
    identity = _verify_harness_identity_lock(
        harness_identity,
        harness_identity_lock_path,
    )
    if not bool(preflight.get("ok")):
        return write_receipt(
            status="preflight_failed",
            execution_tier="not_evaluated",
            real_native_completed=False,
            official_harness_completed=False,
            harness_identity_status=identity.status,
            harness_identity=harness_identity,
            harness_identity_lock_path=identity.lock_path,
            harness_identity_lock_sha256=identity.lock_sha256,
            harness_source_commit=identity.source_commit,
            preflight=preflight,
        )
    if identity.status == "mismatch":
        return write_receipt(
            status="harness_identity_mismatch",
            execution_tier="official_unverified",
            real_native_completed=False,
            official_harness_completed=False,
            harness_identity_status=identity.status,
            harness_identity=harness_identity,
            harness_identity_lock_path=identity.lock_path,
            harness_identity_lock_sha256=identity.lock_sha256,
            harness_source_commit=identity.source_commit,
            preflight=preflight,
        )

    stdout_path = output_root / "official_harness.stdout.log"
    stderr_path = output_root / "official_harness.stderr.log"
    try:
        completed = subprocess.run(
            command,
            cwd=str(output_root),
            capture_output=True,
            text=True,
            timeout=process_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        _write_text_atomic(stdout_path, _coerce_process_text(error.stdout), overwrite=False)
        _write_text_atomic(stderr_path, _coerce_process_text(error.stderr), overwrite=False)
        return write_receipt(
            status="harness_timeout",
            execution_tier="not_evaluated",
            real_native_completed=False,
            official_harness_completed=False,
            harness_identity_status=identity.status,
            harness_identity=harness_identity,
            harness_identity_lock_path=identity.lock_path,
            harness_identity_lock_sha256=identity.lock_sha256,
            harness_source_commit=identity.source_commit,
            preflight=preflight,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    except OSError as error:
        _write_text_atomic(stderr_path, repr(error) + "\n", overwrite=False)
        return write_receipt(
            status="harness_launch_failed",
            execution_tier="not_evaluated",
            real_native_completed=False,
            official_harness_completed=False,
            harness_identity_status=identity.status,
            harness_identity=harness_identity,
            harness_identity_lock_path=identity.lock_path,
            harness_identity_lock_sha256=identity.lock_sha256,
            harness_source_commit=identity.source_commit,
            preflight=preflight,
            stderr_path=stderr_path,
        )

    _write_text_atomic(stdout_path, completed.stdout or "", overwrite=False)
    _write_text_atomic(stderr_path, completed.stderr or "", overwrite=False)
    report_path = _official_report_path(output_root, rows[0]["model_name_or_path"], run_id)
    status = "harness_failed"
    tier = "not_evaluated"
    real_native = False
    official_completed = False
    summary: Mapping[str, Any] | None = None
    report_sha256: str | None = None
    if completed.returncode == 0 and not report_path.is_file():
        status = "result_artifact_missing"
    elif completed.returncode == 0:
        try:
            summary = _validate_official_report(report_path, [row["instance_id"] for row in rows])
        except SWEbenchOfficialBridgeError:
            status = "result_artifact_invalid"
        else:
            report_sha256 = _sha256_file(report_path)
            if not _is_complete_official_report(summary, [row["instance_id"] for row in rows]):
                status = "harness_incomplete"
                tier = "official_incomplete"
            else:
                official_completed = True
                if binding.input_provenance != "runner_bound":
                    status = "completed_external_unbound"
                    tier = "official_external_unbound"
                elif scope.scope == "official_full_local_snapshot":
                    status = "completed_official_full_local_snapshot"
                    tier = "official_full_local_snapshot"
                elif not scope.comparison_eligible:
                    status = "completed_official_subset"
                    tier = "official_subset"
                elif identity.status != "verified":
                    status = "completed_official_unverified"
                    tier = "official_unverified"
                else:
                    status = "completed"
                    tier = "real_native"
                    real_native = True
    return write_receipt(
        status=status,
        execution_tier=tier,
        real_native_completed=real_native,
        official_harness_completed=official_completed,
        harness_identity_status=identity.status,
        harness_identity=harness_identity,
        harness_identity_lock_path=identity.lock_path,
        harness_identity_lock_sha256=identity.lock_sha256,
        harness_source_commit=identity.source_commit,
        preflight=preflight,
        returncode=completed.returncode,
        report_path=report_path if report_path.is_file() else None,
        report_sha256=report_sha256,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        report_summary=summary,
    )


def _load_completed_swebench_contract(run_dir: Path) -> dict[str, Any]:
    contract_path = run_dir / "evidence_contract.json"
    raw = _read_json_mapping(contract_path, label="runner evidence contract")
    if raw.get("schema") != EXPERIMENT_EVIDENCE_SCHEMA:
        raise SWEbenchPredictionExportError(
            f"unexpected runner evidence schema: {raw.get('schema')!r}"
        )
    if raw.get("run_status") != "completed":
        raise SWEbenchPredictionExportError(
            f"runner run must be completed; received run_status={raw.get('run_status')!r}"
        )
    declared = str(raw.get("artifact_sha256") or "")
    calculated = canonical_sha256(
        {key: value for key, value in raw.items() if key != "artifact_sha256"}
    )
    if declared != calculated:
        raise SWEbenchPredictionExportError("runner evidence contract artifact_sha256 mismatch")
    benchmark = raw.get("benchmark")
    if not isinstance(benchmark, Mapping) or benchmark.get("id") != "swebench_lite":
        raise SWEbenchPredictionExportError(
            "runner evidence contract must belong to benchmark='swebench_lite'"
        )
    if not str(raw.get("run_id") or "").strip():
        raise SWEbenchPredictionExportError("runner evidence contract lacks run_id")
    _validate_source_generation_credibility(raw)
    return raw


def _validate_source_generation_credibility(contract: Mapping[str, Any]) -> None:
    """Require evidence that the declared model, not a fallback, made patches.

    A completed trajectory is not automatically a model execution.  The
    runner already records strict runtime facts; the official bridge must
    consume them before labelling a patch file with ``model_name_or_path``.
    Otherwise a dummy fallback or failed endpoint could be exported as Qwen.
    """

    claim = contract.get("claim")
    if not isinstance(claim, Mapping):
        raise SWEbenchPredictionExportError("runner evidence contract lacks claim evidence")
    claim_scope = str(claim.get("scope") or "")
    if claim_scope in {"", "pending", "plumbing", "invalid"}:
        raise SWEbenchPredictionExportError(
            f"runner source is not a credible model-generation run: claim.scope={claim_scope!r}"
        )

    model = contract.get("model")
    if not isinstance(model, Mapping):
        raise SWEbenchPredictionExportError("runner evidence contract lacks model evidence")
    provider = str(model.get("provider") or "")
    if provider in {"", "dummy"}:
        raise SWEbenchPredictionExportError(
            "runner source uses no credible non-dummy model provider"
        )
    method = contract.get("method")
    if not isinstance(method, Mapping):
        raise SWEbenchPredictionExportError("runner evidence contract lacks method evidence")
    if method.get("strict_proxy_compatible") is not True:
        raise SWEbenchPredictionExportError(
            "runner source method is not declared strict_proxy_compatible"
        )
    label = _resolve_model_name(contract)
    preflight = model.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("passed") is not True:
        raise SWEbenchPredictionExportError(
            "runner source model preflight did not prove the requested runtime live"
        )
    observed = model.get("observed")
    if not isinstance(observed, Mapping):
        raise SWEbenchPredictionExportError("runner source lacks observed model runtime evidence")
    if observed.get("model_instantiated") is not True:
        raise SWEbenchPredictionExportError(
            "runner source did not instantiate its configured model"
        )
    if bool(observed.get("dummy_model_used")) or bool(observed.get("dummy_fallback_used")):
        raise SWEbenchPredictionExportError("runner source used a dummy model or dummy fallback")
    if int(observed.get("remote_error_count", 0) or 0) > 0:
        raise SWEbenchPredictionExportError("runner source recorded remote model-call errors")
    if int(observed.get("remote_call_count", 0) or 0) < 1:
        raise SWEbenchPredictionExportError(
            "runner source lacks positive remote model-call evidence"
        )
    requested_models = observed.get("requested_models")
    if not isinstance(requested_models, list) or requested_models != [label]:
        raise SWEbenchPredictionExportError(
            "runner source observed model identity does not exactly match its official label"
        )


def _source_generation_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Carry the source runtime and adapter facts into the final receipt."""

    model = contract.get("model")
    method = contract.get("method")
    claim = contract.get("claim")
    model_map = dict(model) if isinstance(model, Mapping) else {}
    observed = model_map.get("observed")
    preflight = model_map.get("preflight")
    method_map = dict(method) if isinstance(method, Mapping) else {}
    claim_map = dict(claim) if isinstance(claim, Mapping) else {}
    return {
        "official_model_label": _resolve_model_name(contract),
        "model_provider": model_map.get("provider"),
        "runtime_preflight_passed": (
            preflight.get("passed") if isinstance(preflight, Mapping) else None
        ),
        "observed_requested_models": (
            observed.get("requested_models") if isinstance(observed, Mapping) else None
        ),
        "remote_call_count": (
            observed.get("remote_call_count") if isinstance(observed, Mapping) else None
        ),
        "remote_error_count": (
            observed.get("remote_error_count") if isinstance(observed, Mapping) else None
        ),
        "dummy_fallback_used": (
            observed.get("dummy_fallback_used") if isinstance(observed, Mapping) else None
        ),
        "method_id": method_map.get("id"),
        "method_integration_level": method_map.get("integration_level"),
        "method_strict_proxy_compatible": method_map.get("strict_proxy_compatible"),
        "source_claim_scope": claim_map.get("scope"),
    }


def _runner_trajectory_entries(contract: Mapping[str, Any], run_dir: Path) -> list[dict[str, str]]:
    artifacts = contract.get("artifacts")
    manifest = artifacts.get("trajectory_manifest") if isinstance(artifacts, Mapping) else None
    if not isinstance(manifest, Mapping):
        raise SWEbenchPredictionExportError("runner evidence contract lacks trajectory_manifest")
    if manifest.get("schema") != RUNNER_TRAJECTORY_MANIFEST_SCHEMA:
        raise SWEbenchPredictionExportError("unexpected runner trajectory manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SWEbenchPredictionExportError("runner trajectory manifest has no entries")
    out: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    seen_tasks: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SWEbenchPredictionExportError(
                "runner trajectory manifest contains a non-object entry"
            )
        path = str(entry.get("path") or "")
        task_id = str(entry.get("task_id") or "")
        digest = str(entry.get("sha256") or "")
        if not path or not task_id or len(digest) != 64:
            raise SWEbenchPredictionExportError("runner trajectory manifest entry is incomplete")
        _safe_relative_task_path(run_dir, path)
        if path in seen_paths or task_id in seen_tasks:
            raise SWEbenchPredictionExportError(
                "runner trajectory manifest has duplicate path or task_id"
            )
        seen_paths.add(path)
        seen_tasks.add(task_id)
        out.append({"path": path, "task_id": task_id, "sha256": digest})
    return sorted(out, key=lambda item: item["task_id"])


def _load_verified_trajectory(run_dir: Path, entry: Mapping[str, str]) -> Mapping[str, Any]:
    target = _safe_relative_task_path(run_dir, entry["path"])
    if not target.is_file():
        raise SWEbenchPredictionExportError(f"manifested trajectory is missing: {entry['path']}")
    actual_digest = _sha256_file(target)
    if actual_digest != entry["sha256"]:
        raise SWEbenchPredictionExportError(
            f"manifested trajectory sha256 mismatch: {entry['path']}"
        )
    trajectory = _read_json_mapping(target, label="manifested trajectory")
    if str(trajectory.get("task_id") or "") != entry["task_id"]:
        raise SWEbenchPredictionExportError(
            f"manifested trajectory task_id mismatch: {entry['path']}"
        )
    if trajectory.get("benchmark") != "swebench_lite":
        raise SWEbenchPredictionExportError(
            f"manifested trajectory benchmark is not swebench_lite: {entry['path']}"
        )
    return trajectory


def _extract_patch(trajectory: Mapping[str, Any]) -> ExtractedPatch | None:
    """Return a patch only when it is attached to a persisted action.

    ``Step.content`` alone is not an admissible prediction source: it has no
    action-level generation receipt.  The runner records the latter after
    each ``predict`` invocation, which lets this bridge bind a patch to the
    exact task and model call that produced it.
    """

    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return None
    for step in reversed(steps):
        if not isinstance(step, Mapping):
            continue
        action = step.get("action")
        if not isinstance(action, Mapping):
            continue
        info = action.get("info")
        if isinstance(info, Mapping) and isinstance(info.get("patch_text"), str):
            normalized = _normalize_unified_patch(str(info["patch_text"]))
            if normalized:
                return ExtractedPatch(
                    patch=normalized,
                    action=action,
                    source="action.info.patch_text",
                )
        raw = action.get("raw")
        if isinstance(raw, str):
            normalized = _normalize_unified_patch(raw)
            if normalized:
                return ExtractedPatch(
                    patch=normalized,
                    action=action,
                    source="action.raw",
                )
    return None


def _require_nonnegative_int(
    value: Any,
    *,
    context: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SWEbenchPredictionExportError(f"{context} is malformed")
    return value


def _validate_task_patch_generation(
    trajectory: Mapping[str, Any],
    extracted: ExtractedPatch,
    *,
    task_id: str,
    model_name: str,
) -> None:
    """Require task- and action-level evidence for one exported patch.

    Run-level telemetry can only say that *some* task used Qwen.  The official
    prediction contract is stricter: every patch must come from a trajectory
    whose own model snapshot is healthy *and* be byte-bound to a successful
    model response emitted during that action's ``predict`` call.  A remote
    call counter alone is deliberately insufficient: an adapter could call a
    model, discard its reply, and return a static diff.  The trajectory file
    is checksum-bound by the runner manifest, so this evidence is revalidated
    both at export and when a portable prediction bundle is consumed on
    another host.
    """

    trajectory_info = trajectory.get("info")
    runtime = trajectory_info.get("model_runtime") if isinstance(trajectory_info, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise SWEbenchPredictionExportError(
            f"task {task_id} lacks task-level model runtime evidence"
        )
    if runtime.get("model_instantiated") is not True:
        raise SWEbenchPredictionExportError(
            f"task {task_id} did not instantiate its configured model"
        )
    if bool(runtime.get("dummy_model_used")) or bool(runtime.get("dummy_fallback_used")):
        raise SWEbenchPredictionExportError(f"task {task_id} used a dummy model or fallback")
    if (
        _require_nonnegative_int(
            runtime.get("remote_error_count", 0),
            context=f"task {task_id} remote_error_count",
        )
        != 0
    ):
        raise SWEbenchPredictionExportError(f"task {task_id} recorded remote model-call errors")
    task_remote_calls = _require_nonnegative_int(
        runtime.get("remote_call_count", 0),
        context=f"task {task_id} remote_call_count",
    )
    if task_remote_calls < 1:
        raise SWEbenchPredictionExportError(
            f"task {task_id} lacks positive remote model-call evidence"
        )
    if runtime.get("requested_model") != model_name:
        raise SWEbenchPredictionExportError(
            f"task {task_id} model identity does not match the official label"
        )

    action = extracted.action
    action_info = action.get("info")
    generation = action_info.get("model_generation") if isinstance(action_info, Mapping) else None
    if not isinstance(generation, Mapping):
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action lacks runner-owned generation evidence"
        )
    if generation.get("schema") != ACTION_MODEL_GENERATION_SCHEMA:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action generation evidence has an unexpected schema"
        )
    cursor_start, cursor_end = _validate_generation_authority(
        generation,
        task_id=task_id,
    )
    raw = action.get("raw")
    if not isinstance(raw, str):
        raise SWEbenchPredictionExportError(f"task {task_id} patch action raw text is malformed")
    expected_action_hash = sha256_text(raw)
    if generation.get("action_sha256") != expected_action_hash:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action generation evidence hash mismatch"
        )
    if generation.get("model_instantiated_after") is not True:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action did not instantiate its configured model"
        )
    if generation.get("requested_model") != model_name:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action model identity does not match the official label"
        )
    remote_delta = _require_nonnegative_int(
        generation.get("remote_call_delta", 0),
        context=f"task {task_id} patch action remote_call_delta",
    )
    if remote_delta < 1:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action lacks positive remote model-call evidence"
        )
    if remote_delta > task_remote_calls:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action remote-call evidence exceeds task runtime evidence"
        )
    event_count = _require_nonnegative_int(
        generation.get("generation_event_count", 0),
        context=f"task {task_id} patch action generation_event_count",
    )
    if event_count != cursor_end - cursor_start:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action generation-event count does not match its cursor window"
        )
    if (
        _require_nonnegative_int(
            generation.get("remote_error_delta", 0),
            context=f"task {task_id} patch action remote_error_delta",
        )
        != 0
    ):
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action recorded remote model-call errors"
        )
    if (
        _require_nonnegative_int(
            generation.get("dummy_fallback_delta", 0),
            context=f"task {task_id} patch action dummy_fallback_delta",
        )
        != 0
        or bool(generation.get("dummy_model_used"))
        or bool(generation.get("dummy_fallback_used"))
    ):
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action used a dummy model or fallback"
        )
    if extracted.source == "action.info.patch_text":
        patch_text = action_info.get("patch_text") if isinstance(action_info, Mapping) else None
        expected_patch_hash = sha256_text(patch_text) if isinstance(patch_text, str) else None
        if (
            expected_patch_hash is None
            or generation.get("patch_text_sha256") != expected_patch_hash
        ):
            raise SWEbenchPredictionExportError(
                f"task {task_id} patch_text lacks generation evidence hash"
            )
        _validate_model_response_binding(
            generation,
            binding_key="patch_text_response_binding",
            content=patch_text,
            task_id=task_id,
            model_name=model_name,
            cursor_start=cursor_start,
            cursor_end=cursor_end,
        )
    else:
        _validate_model_response_binding(
            generation,
            binding_key="raw_response_binding",
            content=raw,
            task_id=task_id,
            model_name=model_name,
            cursor_start=cursor_start,
            cursor_end=cursor_end,
        )


def _validate_generation_authority(
    generation: Mapping[str, Any],
    *,
    task_id: str,
) -> tuple[int, int]:
    """Require a Runner-controlled process authority and one action window."""

    authority = generation.get("authority")
    if not isinstance(authority, Mapping):
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action lacks process-isolated generation authority"
        )
    if authority.get("schema") != GENERATION_AUTHORITY_SCHEMA:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action generation authority has an unexpected schema"
        )
    if authority.get("mode") != "process_isolated":
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action was not generated through a process-isolated authority"
        )
    if authority.get("model_channel") != "generate_only":
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action generation authority exposed an invalid model channel"
        )
    if authority.get("method_execution") != "isolated_json_rpc":
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action was not produced by an isolated JSON-RPC method worker"
        )
    if authority.get("strict_proxy_compatible") is not True:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action was produced by a method without strict proxy admission"
        )
    cursor_start = _require_nonnegative_int(
        authority.get("generation_cursor_start"),
        context=f"task {task_id} patch action generation_cursor_start",
    )
    cursor_end = _require_nonnegative_int(
        authority.get("generation_cursor_end"),
        context=f"task {task_id} patch action generation_cursor_end",
    )
    if cursor_end < cursor_start:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch action generation cursor window is inverted"
        )
    return cursor_start, cursor_end


def _validate_model_response_binding(
    generation: Mapping[str, Any],
    *,
    binding_key: str,
    content: str,
    task_id: str,
    model_name: str,
    cursor_start: int,
    cursor_end: int,
) -> None:
    """Verify that an exported patch is exactly a recorded remote response."""

    binding = generation.get(binding_key)
    if not isinstance(binding, Mapping):
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch lacks a runner-bound model response"
        )
    if binding.get("schema") != MODEL_RESPONSE_BINDING_SCHEMA:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch response binding has an unexpected schema"
        )
    if binding.get("status") != "bound":
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch is not exactly bound to a successful model response"
        )
    expected_content_hash = sha256_text(content)
    if binding.get("content_sha256") != expected_content_hash:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch response binding content hash mismatch"
        )
    event = binding.get("generation_event")
    if not isinstance(event, Mapping):
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch response binding lacks a model event"
        )
    if event.get("schema") != MODEL_GENERATION_EVENT_SCHEMA:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch model event has an unexpected schema"
        )
    if event.get("event_sha256") != generation_event_sha256(event):
        raise SWEbenchPredictionExportError(f"task {task_id} patch model event digest mismatch")
    if event.get("response_sha256") != expected_content_hash:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch does not equal the recorded model response"
        )
    if event.get("outcome") != "remote_success":
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch model event is not a successful remote generation"
        )
    if event.get("requested_model") != model_name:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch model-event identity does not match the official label"
        )
    generation_id = event.get("generation_id")
    sequence = event.get("sequence")
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
    ):
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch model event identity is malformed"
        )
    if not cursor_start < sequence <= cursor_end:
        raise SWEbenchPredictionExportError(
            f"task {task_id} patch model event falls outside its action generation window"
        )


def _normalize_unified_patch(value: str) -> str | None:
    text = value.strip()
    fenced = _FENCED_DIFF_RE.match(text)
    if fenced:
        text = fenced.group("patch").strip()
    if not text.startswith("diff --git "):
        return None
    # A single header is not a patch. Require at least one hunk marker as a
    # cheap structural floor without pretending to reimplement git-apply.
    if "\n@@ " not in text and "\n@@@ " not in text:
        return None
    return text + "\n"


def _resolve_model_name(contract: Mapping[str, Any]) -> str:
    """Return the model label recorded before a run generated its patches.

    This bridge deliberately has no post-hoc model-name override.  An upstream
    predictions file uses this label in the report name, so letting a CLI
    rewrite it would make a Qwen trajectory look like a different model.  If
    an endpoint alias needs a different public label, declare that label in the
    experiment configuration before collecting the runner trajectory.
    """

    model = contract.get("model")
    if isinstance(model, Mapping):
        requested = str(model.get("requested_model") or "").strip()
        if requested:
            return requested
        observed = model.get("observed")
        if isinstance(observed, Mapping):
            names = observed.get("requested_models")
            if isinstance(names, list) and len(names) == 1 and str(names[0]).strip():
                return str(names[0]).strip()
    raise SWEbenchPredictionExportError("runner evidence contract lacks a model_name_or_path")


def _source_model_identity(contract: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    """Preserve the contract facts behind the immutable official model label."""
    identity = _source_generation_identity(contract)
    identity["official_model_label"] = model_name
    return identity


def _validate_prediction_export_target(
    source_root: Path,
    target: Path,
    entries: Sequence[Mapping[str, str]],
) -> None:
    """Allow only a new, non-core artifact below the completed source run.

    A completed evidence contract is the trust anchor for export.  Letting an
    arbitrary ``--output`` overwrite it (or a manifested trajectory) turns an
    export command into an evidence-corruption tool.  Keeping generated
    predictions under ``<run>/artifacts/`` also makes the provenance boundary
    inspectable without scanning unrelated directories.
    """

    artifact_root = (source_root / "artifacts").resolve()
    try:
        target.relative_to(artifact_root)
    except ValueError as error:
        raise SWEbenchPredictionExportError(
            "predictions output must be a new file below <run_dir>/artifacts/"
        ) from error
    forbidden = {
        (source_root / "evidence_contract.json").resolve(),
        *((source_root / str(entry["path"])).resolve() for entry in entries),
    }
    if target in forbidden:
        raise SWEbenchPredictionExportError(
            "predictions output may not overwrite runner evidence or a manifested trajectory"
        )
    if target.suffix != ".jsonl":
        raise SWEbenchPredictionExportError("predictions output must use the .jsonl suffix")
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    source_bundle_path = target.with_suffix(target.suffix + ".sources")
    if target.exists() or receipt_path.exists() or source_bundle_path.exists():
        raise SWEbenchPredictionExportError(
            "predictions output or its provenance bundle already exists; choose a new artifact path"
        )


def _validate_prediction_bundle_destination(
    target: Path,
    receipt_path: Path,
    source_bundle_path: Path,
) -> None:
    """Recheck all publish targets immediately before materializing a bundle."""

    if target.exists() or receipt_path.exists() or source_bundle_path.exists():
        raise SWEbenchPredictionExportError(
            "predictions output or its provenance bundle already exists; choose a new artifact path"
        )


def _materialize_source_bundle(
    source_root: Path,
    entries: Sequence[Mapping[str, str]],
    destination: Path,
) -> Path:
    """Copy the signed runner evidence required to verify an export elsewhere.

    The returned directory is a same-filesystem staging path.  It is published
    only after the exporter atomically claims the prediction destination.
    Keeping this bundle next to predictions lets a local author rsync one
    directory to a Docker server without depending on stale absolute paths.
    """

    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent)))
    try:
        contract = source_root / "evidence_contract.json"
        _copy_file_with_hash(contract, staged / "evidence_contract.json", _sha256_file(contract))
        for entry in entries:
            source = _safe_relative_task_path(source_root, str(entry["path"]))
            _copy_file_with_hash(source, staged / str(entry["path"]), str(entry["sha256"]))
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return staged


def _copy_file_with_hash(source: Path, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if _sha256_file(target) != expected_sha256:
        raise SWEbenchPredictionExportError(
            f"portable source bundle copy hash mismatch: {source.name}"
        )


def _validate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SWEbenchOfficialBridgeError("predictions must contain at least one row")
    seen_ids: set[str] = set()
    models: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != set(OFFICIAL_PREDICTION_FIELDS):
            raise SWEbenchOfficialBridgeError(
                f"prediction row {index} does not match official field contract"
            )
        instance_id = str(row.get("instance_id") or "").strip()
        model = str(row.get("model_name_or_path") or "").strip()
        patch = _normalize_unified_patch(str(row.get("model_patch") or ""))
        if not instance_id:
            raise SWEbenchOfficialBridgeError(f"prediction row {index} lacks instance_id")
        if instance_id in seen_ids:
            raise SWEbenchOfficialBridgeError(f"duplicate prediction instance_id: {instance_id}")
        if not model:
            raise SWEbenchOfficialBridgeError(f"prediction row {index} lacks model_name_or_path")
        if patch is None:
            raise SWEbenchOfficialBridgeError(
                f"prediction row {index} model_patch is not a valid unified diff"
            )
        seen_ids.add(instance_id)
        models.add(model)
    if len(models) != 1:
        raise SWEbenchOfficialBridgeError(
            "one official evaluation receipt must contain exactly one model_name_or_path"
        )


def _official_runtime_preflight() -> dict[str, Any]:
    """Check the prerequisites that the upstream Docker harness actually needs."""

    missing: list[str] = []
    swebench_available = importlib.util.find_spec("swebench") is not None
    if not swebench_available:
        missing.append("python_package:swebench")
    docker_cli = shutil.which("docker")
    docker_daemon = False
    docker_error: str | None = None
    if not docker_cli:
        missing.append("docker_cli")
    else:
        try:
            probe = subprocess.run(
                [docker_cli, "info"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            docker_error = type(error).__name__
            missing.append("docker_daemon")
        else:
            docker_daemon = probe.returncode == 0
            if not docker_daemon:
                docker_error = (probe.stderr or probe.stdout or "docker info failed").strip()[:300]
                missing.append("docker_daemon")
    return {
        "ok": not missing,
        "missing": missing,
        "swebench_importable": swebench_available,
        "docker_cli": docker_cli,
        "docker_daemon": docker_daemon,
        "docker_error": docker_error,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _repository_head_blob(path: Path) -> bytes | None:
    """Return the checked-out HEAD bytes for one repository-owned lock.

    A file generated beside an experiment is evidence *about that experiment*,
    not a canonical dataset fact.  The full Lite trust root must first be
    reviewed and committed at a fixed repository path; comparing against the
    HEAD blob prevents a just-created local candidate from self-authorizing a
    benchmark claim.
    """

    try:
        repository_root = _REPOSITORY_ROOT.resolve()
        relative = path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "show", f"HEAD:{relative}"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _verify_repository_canonical_dataset_anchor(
    roster_lock: Mapping[str, Any],
    *,
    roster_lock_sha256: str,
) -> CanonicalDatasetAnchorVerification:
    """Verify the one repository-reviewed full-data trust anchor.

    ``download manifest -> roster lock`` only proves self-consistency: the
    same operator can generate all three artifacts from a fabricated 300-row
    JSONL.  A comparison claim therefore requires an independently reviewed
    anchor committed to the repository.  This function intentionally has no
    arbitrary-path override; external candidates remain local snapshots until
    a maintainer audits and commits the canonical anchor.
    """

    path = (_REPOSITORY_ROOT / _CANONICAL_DATASET_ANCHOR_RELATIVE_PATH).resolve()
    if not path.is_file() or path.is_symlink():
        return CanonicalDatasetAnchorVerification(
            status="missing",
            path=path,
            reason="repository-reviewed canonical dataset anchor is not installed",
        )
    head_bytes = _repository_head_blob(path)
    if head_bytes is None or head_bytes != path.read_bytes():
        return CanonicalDatasetAnchorVerification(
            status="untrusted",
            path=path,
            reason="canonical dataset anchor is not identical to the checked-out HEAD blob",
        )
    try:
        anchor = _read_json_mapping(path, label="repository canonical SWE-bench dataset anchor")
    except SWEbenchOfficialBridgeError as error:
        return CanonicalDatasetAnchorVerification(
            status="malformed",
            path=path,
            reason=str(error),
        )
    if anchor.get("schema") != CANONICAL_DATASET_ANCHOR_SCHEMA:
        return CanonicalDatasetAnchorVerification(
            status="malformed",
            path=path,
            reason="canonical dataset anchor schema mismatch",
        )
    if anchor.get("candidate_only") is True or anchor.get("review_status") != "approved":
        return CanonicalDatasetAnchorVerification(
            status="untrusted",
            path=path,
            reason="canonical dataset anchor has not been explicitly approved after review",
        )
    review = anchor.get("review")
    if not isinstance(review, Mapping):
        return CanonicalDatasetAnchorVerification(
            status="malformed",
            path=path,
            reason="canonical dataset anchor lacks a structured review record",
        )
    declared = str(anchor.get("artifact_sha256") or "")
    calculated = canonical_sha256(
        {key: value for key, value in anchor.items() if key != "artifact_sha256"}
    )
    if declared != calculated:
        return CanonicalDatasetAnchorVerification(
            status="malformed",
            path=path,
            reason="canonical dataset anchor artifact_sha256 mismatch",
        )
    expected_keys = (
        "official_repository",
        "dataset_name",
        "split",
        "source_revision",
        "expected_n_instances",
        "instance_ids",
        "roster_sha256",
        "dataset_checksum",
        "dataset_manifest_sha256",
    )
    for key in expected_keys:
        if anchor.get(key) != roster_lock.get(key):
            return CanonicalDatasetAnchorVerification(
                status="mismatch",
                path=path,
                reason=f"canonical dataset anchor does not match local roster lock field {key}",
                sha256=declared,
            )
    if not re.fullmatch(r"[0-9a-f]{40}", str(anchor.get("source_revision") or "")):
        return CanonicalDatasetAnchorVerification(
            status="malformed",
            path=path,
            reason="canonical dataset anchor must record a full 40-hex immutable source revision",
            sha256=declared,
        )
    if review.get("source_revision") != anchor.get("source_revision"):
        return CanonicalDatasetAnchorVerification(
            status="mismatch",
            path=path,
            reason="canonical dataset review record source_revision mismatch",
            sha256=declared,
        )
    source_url = str(review.get("source_url") or "")
    allowed_prefixes = (
        "https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite/",
        "https://github.com/SWE-bench/SWE-bench/",
    )
    if not source_url.startswith(allowed_prefixes):
        return CanonicalDatasetAnchorVerification(
            status="malformed",
            path=path,
            reason="canonical dataset review record source_url is not an official immutable source",
            sha256=declared,
        )
    if anchor.get("local_roster_lock_sha256") != roster_lock_sha256:
        return CanonicalDatasetAnchorVerification(
            status="mismatch",
            path=path,
            reason="canonical dataset anchor does not bind the supplied local roster lock",
            sha256=declared,
        )
    return CanonicalDatasetAnchorVerification(
        status="verified",
        path=path,
        reason="repository-reviewed canonical dataset anchor matches frozen full snapshot",
        sha256=declared,
    )


def _classify_evaluation_scope(
    rows: Sequence[Mapping[str, str]],
    binding: VerifiedPredictionInput,
    *,
    frozen_dataset: FrozenOfficialDataset | None,
    full_roster_lock_path: str | Path | None,
) -> EvaluationScope:
    """Permit a full benchmark claim only with an audited canonical roster.

    The upstream CLI accepts ``--instance_ids`` and correctly reports that
    *requested subset*.  That is useful for debugging but it is not a
    SWE-bench Lite score.  A platform-level full claim therefore additionally
    needs a versioned roster lock, the complete locally recorded source data,
    and an exact ID match.
    """

    if full_roster_lock_path is None:
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="no canonical SWE-bench Lite roster lock was supplied",
        )
    lock_path = Path(full_roster_lock_path).expanduser().resolve()
    raw = _read_json_mapping(lock_path, label="SWE-bench full roster lock")
    if raw.get("schema") != FULL_ROSTER_LOCK_SCHEMA:
        raise SWEbenchOfficialBridgeError("unexpected SWE-bench full roster lock schema")
    declared = str(raw.get("artifact_sha256") or "")
    calculated = canonical_sha256(
        {key: value for key, value in raw.items() if key != "artifact_sha256"}
    )
    if declared != calculated:
        raise SWEbenchOfficialBridgeError("SWE-bench full roster lock artifact_sha256 mismatch")
    if raw.get("dataset_name") != OFFICIAL_DATASET_NAME or raw.get("split") != OFFICIAL_SPLIT:
        raise SWEbenchOfficialBridgeError("SWE-bench full roster lock dataset/split mismatch")
    if raw.get("official_repository") != "https://github.com/SWE-bench/SWE-bench":
        raise SWEbenchOfficialBridgeError(
            "SWE-bench full roster lock must identify the official repository"
        )
    source_revision = str(raw.get("source_revision") or "")
    if not re.fullmatch(r"[0-9a-f]{7,64}", source_revision):
        raise SWEbenchOfficialBridgeError(
            "SWE-bench full roster lock must record an immutable audited dataset source_revision"
        )
    roster = raw.get("instance_ids")
    if not isinstance(roster, list) or not all(isinstance(item, str) and item for item in roster):
        raise SWEbenchOfficialBridgeError("SWE-bench full roster lock instance_ids is malformed")
    if len(set(roster)) != len(roster):
        raise SWEbenchOfficialBridgeError(
            "SWE-bench full roster lock contains duplicate instance_ids"
        )
    if raw.get("expected_n_instances") != SWE_BENCH_LITE_EXPECTED_INSTANCES:
        raise SWEbenchOfficialBridgeError(
            "SWE-bench full roster lock must explicitly declare the 300-instance Lite denominator"
        )
    if len(roster) != SWE_BENCH_LITE_EXPECTED_INSTANCES:
        raise SWEbenchOfficialBridgeError(
            "SWE-bench full roster lock does not contain 300 instance_ids"
        )
    locked_dataset_checksum = str(raw.get("dataset_checksum") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", locked_dataset_checksum):
        raise SWEbenchOfficialBridgeError(
            "SWE-bench full roster lock must bind the audited full dataset checksum"
        )
    locked_manifest_sha256 = str(raw.get("dataset_manifest_sha256") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", locked_manifest_sha256):
        raise SWEbenchOfficialBridgeError(
            "SWE-bench full roster lock must bind the audited downloaded-data manifest"
        )
    roster_sha256 = canonical_sha256(sorted(roster))
    if raw.get("roster_sha256") != roster_sha256:
        raise SWEbenchOfficialBridgeError("SWE-bench full roster lock roster_sha256 mismatch")
    if binding.input_provenance != "runner_bound" or binding.source_contract is None:
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="external or unverifiable predictions cannot make a platform full-benchmark claim",
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    benchmark = binding.source_contract.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="source run lacks benchmark evidence",
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    if benchmark.get("data_source") != "swebench_lite_full":
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="source run did not use the full downloaded SWE-bench Lite data",
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    if benchmark.get("dataset_split") != OFFICIAL_SPLIT or not benchmark.get("dataset_checksum"):
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="source run lacks full-dataset split/checksum evidence",
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    if benchmark.get("dataset_checksum") != locked_dataset_checksum:
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="source run dataset checksum does not match the local roster lock",
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    if benchmark.get("dataset_revision") != source_revision:
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="source run dataset revision does not match the local roster lock",
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    if benchmark.get("dataset_manifest_sha256") != locked_manifest_sha256:
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="source run dataset manifest does not match the local roster lock",
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    if frozen_dataset is None:
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason=(
                "no audited local full dataset was frozen for the official harness; "
                "a moving Hugging Face alias cannot support real_native"
            ),
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    if (
        frozen_dataset.dataset_checksum != locked_dataset_checksum
        or frozen_dataset.dataset_revision != source_revision
        or frozen_dataset.dataset_manifest_sha256 != locked_manifest_sha256
    ):
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="frozen harness dataset does not match the canonical full-roster lock",
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    predicted_ids = sorted(str(row["instance_id"]) for row in rows)
    if predicted_ids != sorted(roster):
        return EvaluationScope(
            scope="official_subset",
            comparison_eligible=False,
            reason="prediction roster is not the canonical complete 300-instance Lite roster",
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
        )
    anchor = _verify_repository_canonical_dataset_anchor(
        raw,
        roster_lock_sha256=declared,
    )
    if anchor.status != "verified":
        return EvaluationScope(
            scope="official_full_local_snapshot",
            comparison_eligible=False,
            reason=(
                "full local 300-instance snapshot is self-consistent but not a "
                f"repository-reviewed canonical dataset: {anchor.reason}"
            ),
            canonical_roster_lock_path=lock_path,
            canonical_roster_sha256=roster_sha256,
            canonical_roster_lock_sha256=declared,
            canonical_dataset_checksum=locked_dataset_checksum,
            canonical_source_revision=source_revision,
            canonical_dataset_anchor_path=anchor.path,
            canonical_dataset_anchor_sha256=anchor.sha256,
            canonical_dataset_anchor_status=anchor.status,
        )
    return EvaluationScope(
        scope="official_full",
        comparison_eligible=True,
        reason=(
            "runner-bound full data, repository-reviewed canonical 300-instance "
            "snapshot, and exact prediction roster match"
        ),
        canonical_roster_lock_path=lock_path,
        canonical_roster_sha256=roster_sha256,
        canonical_roster_lock_sha256=declared,
        canonical_dataset_checksum=locked_dataset_checksum,
        canonical_source_revision=source_revision,
        canonical_dataset_anchor_path=anchor.path,
        canonical_dataset_anchor_sha256=anchor.sha256,
        canonical_dataset_anchor_status=anchor.status,
    )


def _observed_harness_identity(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint the installed evaluator instead of trusting an import name."""

    identity: dict[str, Any] = {
        "schema": "lh_swebench_harness_identity_observation/v1",
        "swebench_distribution_version": None,
        "module_sha256": {},
        "docker_server": None,
    }
    try:
        identity["swebench_distribution_version"] = importlib_metadata.version("swebench")
    except importlib_metadata.PackageNotFoundError:
        pass
    for module_name in ("swebench.harness.run_evaluation", "swebench.harness.reporting"):
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        origin = spec.origin if spec is not None else None
        if origin and Path(origin).is_file():
            identity["module_sha256"][module_name] = _sha256_file(Path(origin))
    docker_cli = preflight.get("docker_cli")
    if docker_cli and bool(preflight.get("docker_daemon")):
        try:
            result = subprocess.run(
                [str(docker_cli), "version", "--format", "{{json .Server}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                parsed = json.loads(result.stdout)
                if isinstance(parsed, Mapping):
                    identity["docker_server"] = dict(parsed)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
    identity["artifact_sha256"] = canonical_sha256(identity)
    return identity


def _verify_harness_identity_lock(
    observed: Mapping[str, Any],
    lock_path_raw: str | Path | None,
) -> HarnessIdentityVerification:
    """Compare observed evaluator bytes against an audited, versioned lock."""

    if lock_path_raw is None:
        return HarnessIdentityVerification(status="unverified", lock_path=None)
    lock_path = Path(lock_path_raw).expanduser().resolve()
    raw = _read_json_mapping(lock_path, label="SWE-bench harness identity lock")
    if raw.get("schema") != HARNESS_IDENTITY_LOCK_SCHEMA:
        raise SWEbenchOfficialBridgeError("unexpected SWE-bench harness identity lock schema")
    declared = str(raw.get("artifact_sha256") or "")
    calculated = canonical_sha256(
        {key: value for key, value in raw.items() if key != "artifact_sha256"}
    )
    if declared != calculated:
        raise SWEbenchOfficialBridgeError(
            "SWE-bench harness identity lock artifact_sha256 mismatch"
        )
    if raw.get("official_repository") != "https://github.com/SWE-bench/SWE-bench":
        raise SWEbenchOfficialBridgeError(
            "SWE-bench harness identity lock must identify the official repository"
        )
    source_commit = str(raw.get("source_commit") or "")
    if not re.fullmatch(r"[0-9a-f]{7,64}", source_commit):
        raise SWEbenchOfficialBridgeError(
            "SWE-bench harness identity lock must record an audited source_commit"
        )
    expected = raw.get("observed_identity")
    if not isinstance(expected, Mapping):
        raise SWEbenchOfficialBridgeError("SWE-bench harness identity lock lacks observed_identity")
    if dict(expected) != dict(observed):
        return HarnessIdentityVerification(
            status="mismatch",
            lock_path=lock_path,
            lock_sha256=declared,
            source_commit=source_commit,
        )
    return HarnessIdentityVerification(
        status="verified",
        lock_path=lock_path,
        lock_sha256=declared,
        source_commit=source_commit,
    )


def _prepare_fresh_output_dir(output_root: Path, *, reserve_execution: bool) -> None:
    """Reserve a unique harness working directory before any output exists.

    The upstream evaluator can skip previously completed instance logs.  A
    reused directory could therefore bind a new prediction file to an old
    report.  Fresh, immutable output directories remove that failure mode.
    """

    if output_root.exists():
        if not output_root.is_dir():
            raise SWEbenchOfficialBridgeError("official evaluation output_dir is not a directory")
        if any(output_root.iterdir()):
            raise SWEbenchOfficialBridgeError(
                "official evaluation output_dir must be a fresh empty directory"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    if reserve_execution:
        marker = output_root / ".lh_swebench_execution_reservation.json"
        try:
            _write_json_atomic(
                marker,
                {"schema": "lh_swebench_execution_reservation/v1", "created_at": time.time()},
                overwrite=False,
            )
        except FileExistsError as error:
            raise SWEbenchOfficialBridgeError(
                "official evaluation output_dir is already reserved"
            ) from error


def _official_report_path(output_dir: Path, model_name_or_path: str, run_id: str) -> Path:
    # This naming exactly follows swebench.harness.reporting.make_run_report.
    safe_model = model_name_or_path.replace("/", "__")
    return output_dir / f"{safe_model}.{run_id}.json"


def _validate_official_report(path: Path, requested_ids: Iterable[str]) -> dict[str, Any]:
    report = _read_json_mapping(path, label="official SWE-bench report")
    required = {
        "schema_version",
        "total_instances",
        "submitted_instances",
        "completed_instances",
        "resolved_instances",
        "unresolved_instances",
        "empty_patch_instances",
        "error_instances",
        "completed_ids",
        "incomplete_ids",
        "empty_patch_ids",
        "submitted_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
    }
    missing = sorted(required - set(report))
    if missing:
        raise SWEbenchOfficialBridgeError(
            "official SWE-bench report lacks fields: " + ", ".join(missing)
        )
    if report["schema_version"] != 2:
        raise SWEbenchOfficialBridgeError(
            f"unexpected official SWE-bench report schema_version: {report['schema_version']!r}"
        )
    requested = sorted(set(str(item) for item in requested_ids))
    for field in (
        "completed_ids",
        "incomplete_ids",
        "empty_patch_ids",
        "submitted_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
    ):
        if not isinstance(report[field], list) or not all(
            isinstance(item, str) for item in report[field]
        ):
            raise SWEbenchOfficialBridgeError(
                f"official SWE-bench report field {field} is malformed"
            )
    for field in (
        "total_instances",
        "submitted_instances",
        "completed_instances",
        "resolved_instances",
        "unresolved_instances",
        "empty_patch_instances",
        "error_instances",
    ):
        if (
            not isinstance(report[field], int)
            or isinstance(report[field], bool)
            or report[field] < 0
        ):
            raise SWEbenchOfficialBridgeError(
                f"official SWE-bench report field {field} is malformed"
            )
    id_sets: dict[str, set[str]] = {}
    for field in (
        "completed_ids",
        "incomplete_ids",
        "empty_patch_ids",
        "submitted_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
    ):
        values = list(report[field])
        if len(set(values)) != len(values):
            raise SWEbenchOfficialBridgeError(
                f"official SWE-bench report field {field} contains duplicate IDs"
            )
        ids = set(values)
        unknown = ids - set(requested)
        if unknown:
            raise SWEbenchOfficialBridgeError(
                f"official SWE-bench report field {field} contains IDs outside the requested roster"
            )
        id_sets[field] = ids
    submitted = id_sets["submitted_ids"]
    completed = id_sets["completed_ids"]
    incomplete = id_sets["incomplete_ids"]
    empty = id_sets["empty_patch_ids"]
    resolved = id_sets["resolved_ids"]
    unresolved = id_sets["unresolved_ids"]
    errors = id_sets["error_ids"]
    requested_set = set(requested)
    if submitted != requested_set:
        raise SWEbenchOfficialBridgeError(
            "official report submitted_ids do not match requested prediction roster"
        )
    if incomplete != requested_set - submitted:
        raise SWEbenchOfficialBridgeError(
            "official report incomplete_ids are inconsistent with submitted_ids"
        )
    if not empty <= submitted or not completed <= submitted or not errors <= submitted:
        raise SWEbenchOfficialBridgeError(
            "official report completion/empty/error IDs must be subsets of submitted_ids"
        )
    if not resolved <= completed or not unresolved <= completed:
        raise SWEbenchOfficialBridgeError(
            "official report resolved/unresolved IDs must be subsets of completed_ids"
        )
    if resolved & unresolved or resolved & errors or unresolved & errors:
        raise SWEbenchOfficialBridgeError(
            "official report resolved, unresolved, and error IDs must be disjoint"
        )
    if empty & (resolved | unresolved | errors):
        raise SWEbenchOfficialBridgeError(
            "official report empty-patch IDs cannot also have a result or error"
        )
    if completed - (resolved | unresolved | errors):
        raise SWEbenchOfficialBridgeError(
            "official report completed_ids include IDs without a result classification"
        )
    if submitted != empty | resolved | unresolved | errors:
        raise SWEbenchOfficialBridgeError(
            "official report submitted IDs are not completely classified"
        )
    expected_counts = {
        "total_instances": len(requested_set),
        "submitted_instances": len(submitted),
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": len(empty),
        "error_instances": len(errors),
    }
    for field, expected in expected_counts.items():
        if report[field] != expected:
            raise SWEbenchOfficialBridgeError(
                f"official SWE-bench report {field} does not match its ID list"
            )
    return report


def _is_complete_official_report(report: Mapping[str, Any], requested_ids: Iterable[str]) -> bool:
    requested = sorted(set(str(item) for item in requested_ids))
    if report.get("total_instances") != len(requested):
        return False
    if report.get("submitted_instances") != len(requested):
        return False
    if report.get("completed_instances") != len(requested):
        return False
    if sorted(report.get("completed_ids", [])) != requested:
        return False
    if report.get("error_instances") != 0 or report.get("empty_patch_instances") != 0:
        return False
    if report.get("error_ids") or report.get("empty_patch_ids") or report.get("incomplete_ids"):
        return False
    resolved = report.get("resolved_ids", [])
    unresolved = report.get("unresolved_ids", [])
    if set(resolved) | set(unresolved) != set(requested):
        return False
    if set(resolved) & set(unresolved):
        return False
    if report.get("resolved_instances") != len(resolved):
        return False
    if report.get("unresolved_instances") != len(unresolved):
        return False
    return True


def _safe_relative_task_path(run_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1 or pure.suffix != ".json":
        raise SWEbenchPredictionExportError(f"invalid manifested trajectory path: {relative!r}")
    target = (run_dir / pure.name).resolve()
    if target.parent != run_dir.resolve():
        raise SWEbenchPredictionExportError(
            f"manifested trajectory escapes run directory: {relative!r}"
        )
    return target


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SWEbenchOfficialBridgeError(f"{label} not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SWEbenchOfficialBridgeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise SWEbenchOfficialBridgeError(f"{label} must be a JSON object: {path}")
    return raw


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.fullmatch(str(run_id)):
        raise SWEbenchOfficialBridgeError("run_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = True,
) -> None:
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _write_text_atomic(path, content, overwrite=overwrite)


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = True,
) -> None:
    _write_text_atomic(
        path,
        json.dumps(redact_artifact(dict(payload)), indent=2, ensure_ascii=False) + "\n",
        overwrite=overwrite,
    )


def _write_text_atomic(path: Path, content: str, *, overwrite: bool = True) -> None:
    """Durably write one artifact without sharing a fixed staging filename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(staged, path)
        else:
            # ``link`` is an atomic create-if-absent operation on the same
            # filesystem.  It avoids the TOCTOU race in exists()+replace().
            os.link(staged, path)
            staged.unlink()
        try:
            parent_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            parent_fd = None
        if parent_fd is not None:
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def _coerce_process_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
