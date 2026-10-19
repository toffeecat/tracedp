"""Fail-closed admission for frozen self-trace comparison runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.core.system_cell import HashBoundComponentV1, SystemCellManifestV1
from lh_platform.tracing.self_trace_datasets import SelfTraceDatasetV1, self_trace_dataset
from lh_platform.tracing.self_trace_methods import SelfTraceMethodV1, self_trace_method

SOURCE_ROOT = Path(__file__).resolve().parents[1]
RUN_ADMISSION_MANIFEST_PATH = (
    SOURCE_ROOT.parents[1]
    / "manifests/experiments/controlled_self_trace_v3_run_admission.json"
)
RUN_ADMISSION_SCHEMA = "lh_self_trace_run_admission/v1"
RUN_ADMISSION_PREREGISTRATION_ID = "controlled-self-trace-v3-2026-07-24"

SELF_TRACE_ACTOR_SOURCE_PATHS = {
    "self_trace_controlled_source_sha256": "tracing/self_trace_controlled.py",
    "self_trace_unseen_source_sha256": "tracing/self_trace_unseen.py",
}
SELF_TRACE_ENVIRONMENT_SOURCE_PATHS = {
    **SELF_TRACE_ACTOR_SOURCE_PATHS,
    "self_trace_datasets_source_sha256": "tracing/self_trace_datasets.py",
}
SELF_TRACE_RUNTIME_SOURCE_PATHS = {
    "policy_source_sha256": "training/transformers_peft_policy.py",
    "chat_adapter_source_sha256": "models/transformers_peft_chat.py",
    "self_trace_runner_source_sha256": "tracing/self_trace.py",
    "self_trace_experiment_source_sha256": "tracing/self_trace_experiment.py",
    "trace_adapter_source_sha256": "tracing/adapters.py",
    "self_trace_methods_source_sha256": "tracing/self_trace_methods.py",
    "core_registry_source_sha256": "core/registry.py",
    "tracer_registry_instance_source_sha256": "tracing/registry.py",
    "trace_investigation_runner_source_sha256": "tracing/runner.py",
    "trace_types_source_sha256": "tracing/types.py",
    "replay_receipts_source_sha256": "tracing/replay_receipts.py",
    "model_io_profiles_source_sha256": "training/model_io/profiles.py",
    "model_io_renderer_source_sha256": "training/model_io/renderer.py",
}
SELF_TRACE_METHOD_SOURCE_PATHS = {
    "trace_adapter_source_sha256": "tracing/adapters.py",
    "self_trace_methods_source_sha256": "tracing/self_trace_methods.py",
    "core_registry_source_sha256": "core/registry.py",
    "tracer_registry_instance_source_sha256": "tracing/registry.py",
    "trace_investigation_runner_source_sha256": "tracing/runner.py",
    "trace_types_source_sha256": "tracing/types.py",
    "replay_receipts_source_sha256": "tracing/replay_receipts.py",
}


@dataclass(frozen=True)
class PreregisteredSelfTraceRunV1:
    dataset_id: str
    model_id: str
    trace_method_id: str
    system_cell_id: str
    system_cell_sha256: str
    sampling_seed: int


@dataclass(frozen=True)
class SelfTraceRunAdmissionV1:
    dataset: SelfTraceDatasetV1
    method: SelfTraceMethodV1
    preregistered_run: PreregisteredSelfTraceRunV1 | None
    run_admission_sha256: str | None


def _source_sha256(relative_path: str, *, source_root: Path = SOURCE_ROOT) -> str:
    return hashlib.sha256((source_root / relative_path).read_bytes()).hexdigest()


def _validate_component_source_lock(
    component: HashBoundComponentV1,
    expected_paths: Mapping[str, str],
    *,
    label: str,
    source_root: Path = SOURCE_ROOT,
) -> None:
    raw_lock = component.config.get("implementation_lock")
    if not isinstance(raw_lock, Mapping) or set(raw_lock) != set(expected_paths):
        raise ValueError(f"System Cell {label} implementation lock is incomplete")
    expected_lock = {
        key: _source_sha256(relative_path, source_root=source_root)
        for key, relative_path in expected_paths.items()
    }
    if dict(raw_lock) != expected_lock:
        raise ValueError(f"System Cell {label} source lock does not match the checkout")
    if component.config.get("implementation_lock_sha256") != canonical_sha256(expected_lock):
        raise ValueError(f"System Cell {label} implementation lock hash is invalid")


def validate_self_trace_source_locks(
    system_cell: SystemCellManifestV1,
    *,
    source_root: Path = SOURCE_ROOT,
) -> None:
    """Recompute every preregistered source lock against the active checkout."""

    _validate_component_source_lock(
        system_cell.actor_protocol,
        SELF_TRACE_ACTOR_SOURCE_PATHS,
        label="actor protocol",
        source_root=source_root,
    )
    _validate_component_source_lock(
        system_cell.environment,
        SELF_TRACE_ENVIRONMENT_SOURCE_PATHS,
        label="environment",
        source_root=source_root,
    )
    _validate_component_source_lock(
        system_cell.runtime,
        SELF_TRACE_RUNTIME_SOURCE_PATHS,
        label="runtime",
        source_root=source_root,
    )
    _validate_component_source_lock(
        system_cell.action_objective,
        SELF_TRACE_METHOD_SOURCE_PATHS,
        label="trace method",
        source_root=source_root,
    )


def _run_from_mapping(row: Mapping[str, object]) -> PreregisteredSelfTraceRunV1:
    expected_keys = {
        "dataset_id",
        "model_id",
        "trace_method_id",
        "system_cell_id",
        "system_cell_sha256",
        "sampling_seed",
    }
    if set(row) != expected_keys:
        raise ValueError("self-trace run-admission row has invalid fields")
    sampling_seed = row["sampling_seed"]
    if (
        isinstance(sampling_seed, bool)
        or not isinstance(sampling_seed, int)
        or sampling_seed < 0
    ):
        raise ValueError("self-trace run-admission seed must be a non-negative integer")
    system_cell_sha256 = str(row["system_cell_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", system_cell_sha256) is None:
        raise ValueError("self-trace run-admission Cell hash is invalid")
    return PreregisteredSelfTraceRunV1(
        dataset_id=str(row["dataset_id"]),
        model_id=str(row["model_id"]),
        trace_method_id=str(row["trace_method_id"]),
        system_cell_id=str(row["system_cell_id"]),
        system_cell_sha256=system_cell_sha256,
        sampling_seed=sampling_seed,
    )


def validate_self_trace_run_admission_manifest(
    expected_sha256: str,
    *,
    manifest_path: Path = RUN_ADMISSION_MANIFEST_PATH,
    source_root: Path = SOURCE_ROOT,
) -> tuple[dict[str, object], str]:
    """Verify the independent admission trust root and its own implementation locks."""

    if re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("self-trace run-admission SHA-256 is invalid")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read self-trace run-admission manifest") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "preregistration_id",
        "dataset_id",
        "dataset_sha256",
        "source_locks",
        "runs",
    }:
        raise ValueError("self-trace run-admission manifest fields are invalid")
    actual_sha256 = "sha256:" + canonical_sha256(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError("self-trace run-admission manifest hash mismatch")
    if payload["schema"] != RUN_ADMISSION_SCHEMA:
        raise ValueError("self-trace run-admission schema is unsupported")
    if payload["preregistration_id"] != RUN_ADMISSION_PREREGISTRATION_ID:
        raise ValueError("self-trace run-admission preregistration id is invalid")
    if not isinstance(payload["dataset_id"], str) or not payload["dataset_id"]:
        raise ValueError("self-trace run-admission dataset id is invalid")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", str(payload["dataset_sha256"])) is None:
        raise ValueError("self-trace run-admission dataset SHA-256 is invalid")
    source_locks = payload["source_locks"]
    expected_source_paths = {
        "self_trace_admission_source_sha256": "tracing/self_trace_admission.py",
        "core_registry_source_sha256": "core/registry.py",
    }
    if not isinstance(source_locks, Mapping) or set(source_locks) != set(
        expected_source_paths
    ):
        raise ValueError("self-trace run-admission source locks are incomplete")
    expected_source_locks = {
        key: _source_sha256(relative_path, source_root=source_root)
        for key, relative_path in expected_source_paths.items()
    }
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in source_locks.values()
    ):
        raise ValueError("self-trace run-admission source lock is invalid")
    if dict(source_locks) != expected_source_locks:
        raise ValueError("self-trace run-admission source lock does not match the checkout")
    rows = payload["runs"]
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("self-trace run-admission manifest must contain four runs")
    runs: list[PreregisteredSelfTraceRunV1] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("self-trace run-admission row must be an object")
        runs.append(_run_from_mapping(row))
    if any(run.dataset_id != payload["dataset_id"] for run in runs):
        raise ValueError("self-trace run-admission row dataset identity mismatch")
    run_keys = {(run.model_id, run.trace_method_id) for run in runs}
    if len(run_keys) != len(runs):
        raise ValueError("self-trace run-admission contains duplicate run keys")
    models = {run.model_id for run in runs}
    methods = {run.trace_method_id for run in runs}
    if len(models) != 2 or methods != {"all_at_once", "counterfactual_replay"}:
        raise ValueError("self-trace run-admission must define the frozen 2x2 matrix")
    if run_keys != {(model_id, method_id) for model_id in models for method_id in methods}:
        raise ValueError("self-trace run-admission 2x2 matrix is incomplete")
    return payload, actual_sha256


def _preregistered_run(
    rows: object,
    *,
    dataset_id: str,
    model_id: str,
    trace_method_id: str,
) -> PreregisteredSelfTraceRunV1:
    if not isinstance(rows, list):
        raise ValueError("self-trace run-admission runs must be a list")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("self-trace run-admission row must be an object")
        run = _run_from_mapping(row)
        if (
            run.dataset_id == dataset_id
            and run.model_id == model_id
            and run.trace_method_id == trace_method_id
        ):
            return run
    raise ValueError("self-trace model/dataset/method cell was not preregistered")


def admit_self_trace_run(
    *,
    system_cell: SystemCellManifestV1,
    model_id: str,
    dataset_id: str,
    trace_method_id: str,
    sampling_seed: int,
    expected_run_admission_sha256: str | None = None,
) -> SelfTraceRunAdmissionV1:
    """Resolve descriptors and reject drift before a model is loaded or called."""

    dataset = self_trace_dataset(dataset_id)
    method = self_trace_method(trace_method_id)
    environment = system_cell.environment.config
    if environment.get("benchmark") != dataset.dataset_id:
        raise ValueError("System Cell does not bind the requested self-trace dataset")
    preregistration = dataset.manifest.get("preregistration")
    is_preregistered = isinstance(preregistration, Mapping) and (
        preregistration.get("frozen_before_model_execution") is True
    )
    bound_dataset_sha256 = environment.get("dataset_sha256")
    if is_preregistered and bound_dataset_sha256 is None:
        raise ValueError("preregistered System Cell is missing the self-trace dataset hash")
    if bound_dataset_sha256 is not None and bound_dataset_sha256 != dataset.sha256:
        raise ValueError("System Cell self-trace dataset hash does not match the registry")
    versions = environment.get("versions")
    if not isinstance(versions, Mapping) or versions.get("replay_controller") != (
        dataset.replay_controller_id
    ):
        raise ValueError("System Cell does not bind the requested self-trace replay controller")
    objective = system_cell.action_objective.config
    if objective.get("trace_method_id") != method.trace_method_id:
        raise ValueError("System Cell does not bind the requested self-trace trace method")
    if tuple(objective.get("trace_method_capabilities", ())) != method.capabilities:
        raise ValueError("System Cell self-trace method capabilities do not match the descriptor")
    if objective.get("trace_method_evidence_ceiling") != method.evidence_ceiling.value:
        raise ValueError("System Cell self-trace evidence ceiling does not match the descriptor")

    run: PreregisteredSelfTraceRunV1 | None = None
    run_admission_sha256: str | None = None
    if is_preregistered:
        validate_self_trace_source_locks(system_cell)
        if expected_run_admission_sha256 is None:
            raise ValueError("preregistered self-trace requires a run-admission hash")
        admission_manifest, run_admission_sha256 = (
            validate_self_trace_run_admission_manifest(expected_run_admission_sha256)
        )
        if (
            admission_manifest["dataset_id"] != dataset.dataset_id
            or admission_manifest["dataset_sha256"] != dataset.sha256
        ):
            raise ValueError("self-trace run-admission dataset identity mismatch")
        run = _preregistered_run(
            admission_manifest["runs"],
            dataset_id=dataset.dataset_id,
            model_id=model_id,
            trace_method_id=method.trace_method_id,
        )
        if sampling_seed != run.sampling_seed:
            raise ValueError("self-trace sampling seed differs from the preregistration")
        if system_cell.system_cell_id != run.system_cell_id:
            raise ValueError("self-trace System Cell id differs from the preregistration")
        if system_cell.sha256 != run.system_cell_sha256:
            raise ValueError("self-trace System Cell hash differs from the preregistration")
    return SelfTraceRunAdmissionV1(
        dataset=dataset,
        method=method,
        preregistered_run=run,
        run_admission_sha256=run_admission_sha256,
    )


__all__ = [
    "SELF_TRACE_ACTOR_SOURCE_PATHS",
    "SELF_TRACE_ENVIRONMENT_SOURCE_PATHS",
    "SELF_TRACE_METHOD_SOURCE_PATHS",
    "SELF_TRACE_RUNTIME_SOURCE_PATHS",
    "RUN_ADMISSION_MANIFEST_PATH",
    "RUN_ADMISSION_PREREGISTRATION_ID",
    "RUN_ADMISSION_SCHEMA",
    "PreregisteredSelfTraceRunV1",
    "SelfTraceRunAdmissionV1",
    "admit_self_trace_run",
    "validate_self_trace_run_admission_manifest",
    "validate_self_trace_source_locks",
]
