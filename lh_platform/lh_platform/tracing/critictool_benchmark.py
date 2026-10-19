"""Frozen, gold-blind CriticTool official-data evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.integrations._provenance import require_clean_source_revision
from lh_platform.tracing.integrations.critictool import CRITICTOOL_ERROR_TYPES
from lh_platform.tracing.types import TraceCaseV1, TraceEventV1, TraceOutcomeV1
from lh_platform.tracing.unified import (
    TraceDiagnosticEventV1,
    TraceDiagnosticStage,
    TraceDiagnosticStatus,
    TraceFindingKind,
    TraceMethodPlatform,
    TraceMethodRequestV1,
    TraceMethodRunReceiptV2,
    TraceRunStatus,
)
from lh_platform.tracing.unified_benchmark import (
    TraceBenchmarkRunSpecV1,
    TraceBenchmarkRuntimeAttestationV1,
)

CRITICTOOL_BENCHMARK_MANIFEST_SCHEMA = "lh_critictool_trace_benchmark_manifest/v1"
CRITICTOOL_SELECTION_ALGORITHM = (
    "per_error_type_lowest_sha256(seed:error_type:source_key)_internal_base_failures"
)
CRITICTOOL_SOURCE_PATH = "internal_error/base_data_wo_cot.json"
CRITICTOOL_BENCHMARK_AUTHORIZATION_SCHEMA = "lh_critictool_trace_benchmark_authorization/v1"
CRITICTOOL_BENCHMARK_RESULT_SCHEMA = "lh_critictool_trace_benchmark_result/v4"
PREVIOUS_CRITICTOOL_BENCHMARK_RESULT_SCHEMA = "lh_critictool_trace_benchmark_result/v3"
CRITICTOOL_BENCHMARK_GATE_AUDIT_SCHEMA = "lh_critictool_trace_benchmark_gate_audit/v1"
CRITICTOOL_FROZEN_MANIFEST_SHA256 = (
    "sha256:b0889d5e75a6c69486270a528bf5a2eac04bfe5dd045e6cc95d00095646276a9"
)
CRITICTOOL_V2_FROZEN_MANIFEST_SHA256 = (
    "sha256:d49eaf5f2d44a4ea32dffc9810bb84e7026b10807166d6304eeef1cd67af3a56"
)
CRITICTOOL_V3_FROZEN_MANIFEST_SHA256 = (
    "sha256:6df63de5185883cc3cd65f2d7e58da9d47abcdf73dcf8adcbe38e65260eb0a70"
)
CRITICTOOL_FROZEN_MANIFEST_SHA256S = (
    CRITICTOOL_FROZEN_MANIFEST_SHA256,
    CRITICTOOL_V2_FROZEN_MANIFEST_SHA256,
    CRITICTOOL_V3_FROZEN_MANIFEST_SHA256,
)
CRITICTOOL_FROZEN_SOURCE_REVISION = "601defa22646f25bc1c71bd04cc8ac90a8d8838c"
CRITICTOOL_FROZEN_MODEL_IDS = (
    "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
    "langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld@895c9fe1db239ae6140ffec7801a51e157292f83",
)
CRITICTOOL_FROZEN_MAX_TOKENS = 512
CRITICTOOL_FROZEN_TIMEOUT_SECONDS = 180
CRITICTOOL_FROZEN_TEMPERATURE = 0.0
CRITICTOOL_FROZEN_GENERATION_SEED = 0
CRITICTOOL_FROZEN_MINIMUM_COMPLETED_PER_MODEL = 10
CRITICTOOL_LEGACY_COT_METHOD_DESCRIPTOR = MappingProxyType(
    {
        "method_id": "critictool",
        "display_name": "CriticTool Self-Critique",
        "family": "reflection",
        "capabilities": ("error_classification", "recovery_proposal"),
        "input_modalities": ("text_trajectory",),
        "output_kinds": ("error_classification", "recovery_proposal"),
        "integration_level": "upstream_native_port",
        "execution_path": "isolated_upstream_subprocess",
        "evidence_ceiling": "L2_typed",
        "source_relpath": "code/CriticTool",
        "source_revision": CRITICTOOL_FROZEN_SOURCE_REVISION,
        "unavailable_reason": None,
        "corpus_method_id": "critictool",
    }
)
CRITICTOOL_FROZEN_METHOD_DESCRIPTOR = MappingProxyType(
    {
        **dict(CRITICTOOL_LEGACY_COT_METHOD_DESCRIPTOR),
        "execution_path": "isolated_upstream_subprocess:llm_no_cot",
    }
)
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _canonical_arguments(value: object) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("CriticTool ground-truth arguments must be an object")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_records(dataset_root: Path) -> tuple[Path, Mapping[str, Any]]:
    source = (dataset_root / CRITICTOOL_SOURCE_PATH).resolve()
    if not source.is_relative_to(dataset_root) or not source.is_file():
        raise ValueError("CriticTool benchmark source file is missing or escaped root")
    records = _mapping(
        json.loads(source.read_text(encoding="utf-8")),
        label="CriticTool source dataset",
    )
    if not records:
        raise ValueError("CriticTool source dataset is empty")
    return source, records


def _selected_source_keys(
    *,
    records: Mapping[str, Any],
    seed: str,
    per_error_type_count: int,
    excluded_source_keys: tuple[str, ...],
) -> tuple[str, ...]:
    if not seed:
        raise ValueError("CriticTool selection seed must be non-empty")
    if (
        isinstance(per_error_type_count, bool)
        or not isinstance(per_error_type_count, int)
        or per_error_type_count < 1
    ):
        raise ValueError("CriticTool per-error-type count must be positive")
    if len(excluded_source_keys) != len(set(excluded_source_keys)):
        raise ValueError("CriticTool exclusion keys must be unique")
    unknown_exclusions = set(excluded_source_keys) - set(records)
    if unknown_exclusions:
        raise ValueError(
            f"CriticTool exclusions are absent from source: {sorted(unknown_exclusions)}"
        )
    excluded = set(excluded_source_keys)
    selected: list[str] = []
    for error_type in CRITICTOOL_ERROR_TYPES:
        candidates: list[tuple[str, str]] = []
        for source_key, raw_record in records.items():
            if not isinstance(source_key, str) or source_key in excluded:
                continue
            record = _mapping(raw_record, label=f"CriticTool record {source_key}")
            if record.get("error_type") != error_type:
                continue
            rank = hashlib.sha256(f"{seed}:{error_type}:{source_key}".encode()).hexdigest()
            candidates.append((rank, source_key))
        if len(candidates) < per_error_type_count:
            raise ValueError(
                f"CriticTool error type {error_type} has fewer than "
                f"{per_error_type_count} eligible records"
            )
        selected.extend(source_key for _, source_key in sorted(candidates)[:per_error_type_count])
    return tuple(selected)


def _case_id(source_key: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", source_key).strip("-")
    if not normalized:
        raise ValueError("CriticTool source key cannot form a case id")
    return f"critictool-{normalized}"


def _manifest_case(source_key: str, raw_record: object) -> dict[str, object]:
    record = _mapping(raw_record, label=f"CriticTool record {source_key}")
    error_type = record.get("error_type")
    if error_type not in CRITICTOOL_ERROR_TYPES:
        raise ValueError(f"CriticTool record {source_key} has unsupported error type")
    ground_truth = _mapping(
        record.get("ground_truth"),
        label=f"CriticTool ground truth {source_key}",
    )
    recovery_tool = ground_truth.get("name")
    if not isinstance(recovery_tool, str) or not recovery_tool:
        raise ValueError(f"CriticTool record {source_key} has no recovery tool")
    return {
        "case_id": _case_id(source_key),
        "source_key": source_key,
        "source_record_sha256": "sha256:" + canonical_sha256(record),
        "error_type": error_type,
        "recovery_tool": recovery_tool,
        "recovery_arguments_json": _canonical_arguments(ground_truth.get("args")),
    }


def _build_manifest(
    *,
    dataset_root: Path,
    dataset_id: str,
    seed: str,
    per_error_type_count: int,
    excluded_source_keys: tuple[str, ...],
) -> dict[str, object]:
    if not dataset_id:
        raise ValueError("CriticTool benchmark dataset id must be non-empty")
    revision = require_clean_source_revision(
        dataset_root,
        method_name="CriticTool benchmark dataset",
    )
    source, records = _source_records(dataset_root)
    selected_keys = _selected_source_keys(
        records=records,
        seed=seed,
        per_error_type_count=per_error_type_count,
        excluded_source_keys=excluded_source_keys,
    )
    payload: dict[str, object] = {
        "schema": CRITICTOOL_BENCHMARK_MANIFEST_SCHEMA,
        "dataset_id": dataset_id,
        "source_revision": revision,
        "source_path": CRITICTOOL_SOURCE_PATH,
        "source_sha256": _sha256_file(source),
        "selection": {
            "algorithm": CRITICTOOL_SELECTION_ALGORITHM,
            "seed": seed,
            "per_error_type_count": per_error_type_count,
            "excluded_source_keys": list(excluded_source_keys),
        },
        "cases": [_manifest_case(source_key, records[source_key]) for source_key in selected_keys],
    }
    payload["manifest_sha256"] = "sha256:" + canonical_sha256(payload)
    return payload


def prepare_critictool_trace_benchmark_manifest(
    *,
    dataset_root: str | Path,
    output_path: str | Path,
    dataset_id: str,
    seed: str,
    per_error_type_count: int,
    excluded_source_keys: tuple[str, ...] = (),
) -> dict[str, object]:
    root = Path(dataset_root).resolve()
    manifest = _build_manifest(
        dataset_root=root,
        dataset_id=dataset_id,
        seed=seed,
        per_error_type_count=per_error_type_count,
        excluded_source_keys=excluded_source_keys,
    )
    atomic_write_json(output_path, manifest, overwrite=False)
    return manifest


@dataclass(frozen=True)
class CriticToolBenchmarkGoldV1:
    error_type: str
    recovery_tool: str
    recovery_arguments_json: str

    def __post_init__(self) -> None:
        if self.error_type not in CRITICTOOL_ERROR_TYPES:
            raise ValueError("CriticTool gold error type is unsupported")
        if not self.recovery_tool:
            raise ValueError("CriticTool gold recovery tool is empty")
        if _canonical_arguments(json.loads(self.recovery_arguments_json)) != (
            self.recovery_arguments_json
        ):
            raise ValueError("CriticTool gold recovery arguments are not canonical")


@dataclass(frozen=True)
class CriticToolBenchmarkCaseV1:
    request: TraceMethodRequestV1
    gold: CriticToolBenchmarkGoldV1
    source_key: str
    source_record_sha256: str

    def __post_init__(self) -> None:
        if not self.source_key or not _SHA256_PATTERN.fullmatch(self.source_record_sha256):
            raise ValueError("CriticTool benchmark case provenance is invalid")


@dataclass(frozen=True)
class CriticToolBenchmarkDatasetV1:
    dataset_id: str
    source_revision: str
    source_sha256: str
    manifest_sha256: str
    selection: tuple[tuple[str, object], ...]
    cases: tuple[CriticToolBenchmarkCaseV1, ...]


def _runtime_attestation_from_mapping(
    value: object,
) -> TraceBenchmarkRuntimeAttestationV1:
    raw = _mapping(value, label="CriticTool runtime attestation")
    advertised = raw.get("advertised_model_ids")
    if not isinstance(advertised, list) or not all(isinstance(item, str) for item in advertised):
        raise ValueError("CriticTool advertised model ids are invalid")
    return TraceBenchmarkRuntimeAttestationV1(
        requested_model_id=str(raw.get("requested_model_id", "")),
        advertised_model_ids=tuple(advertised),
        endpoint_catalog_sha256=str(raw.get("endpoint_catalog_sha256", "")),
        attestation_level=str(raw.get("attestation_level", "")),
    )


def _run_spec_from_mapping(value: object) -> TraceBenchmarkRunSpecV1:
    raw = _mapping(value, label="CriticTool run spec")
    method_ids = raw.get("method_ids")
    if not isinstance(method_ids, list) or not all(isinstance(item, str) for item in method_ids):
        raise ValueError("CriticTool run spec method ids are invalid")
    return TraceBenchmarkRunSpecV1(
        dataset_manifest_sha256=str(raw.get("dataset_manifest_sha256", "")),
        model_id=str(raw.get("model_id", "")),
        method_ids=tuple(method_ids),
        method_descriptors_sha256=str(raw.get("method_descriptors_sha256", "")),
        platform_revision=str(raw.get("platform_revision", "")),
        endpoint_base_url_sha256=str(raw.get("endpoint_base_url_sha256", "")),
        python_executable_sha256=str(raw.get("python_executable_sha256", "")),
        python_environment_sha256=str(raw.get("python_environment_sha256", "")),
        max_tokens=int(raw.get("max_tokens", 0)),
        timeout_seconds=int(raw.get("timeout_seconds", 0)),
        temperature=float(raw.get("temperature", -1.0)),
        generation_seed=int(raw.get("generation_seed", -1)),
        max_model_calls=(
            None
            if raw.get("max_model_calls") is None
            else int(raw["max_model_calls"])
        ),
        run_spec_sha256=str(raw.get("run_spec_sha256", "")),
    )


def _method_descriptor_sha256(descriptor: Mapping[str, object]) -> str:
    return "sha256:" + canonical_sha256({"method_descriptors": [dict(descriptor)]})


def validate_critictool_formal_method_descriptor(
    descriptor: Mapping[str, object],
    *,
    dataset_manifest_sha256: str | None = None,
) -> None:
    if descriptor.get("method_id") != "critictool":
        raise ValueError("CriticTool authorization permits only critictool")
    if descriptor.get("source_revision") != CRITICTOOL_FROZEN_SOURCE_REVISION:
        raise ValueError("CriticTool formal source revision mismatch")
    observed = dict(descriptor)
    for key in ("capabilities", "input_modalities", "output_kinds"):
        value = observed.get(key)
        if isinstance(value, (list, tuple)):
            observed[key] = tuple(value)
    # CriticTool v1 predates the shared outcome-scope field; its fixed behavior
    # is observed-failure-only, so accept the canonical default without changing
    # the frozen descriptor key set.
    if observed.get("outcome_scope", "observed_failure") != "observed_failure":
        raise ValueError("CriticTool formal method descriptor mismatch: outcome_scope")
    observed.pop("outcome_scope", None)
    if dataset_manifest_sha256 in {
        CRITICTOOL_FROZEN_MANIFEST_SHA256,
        CRITICTOOL_V2_FROZEN_MANIFEST_SHA256,
    }:
        candidates = (CRITICTOOL_LEGACY_COT_METHOD_DESCRIPTOR,)
    elif dataset_manifest_sha256 == CRITICTOOL_V3_FROZEN_MANIFEST_SHA256:
        candidates = (CRITICTOOL_FROZEN_METHOD_DESCRIPTOR,)
    elif dataset_manifest_sha256 is None:
        candidates = (
            CRITICTOOL_LEGACY_COT_METHOD_DESCRIPTOR,
            CRITICTOOL_FROZEN_METHOD_DESCRIPTOR,
        )
    else:
        candidates = (CRITICTOOL_FROZEN_METHOD_DESCRIPTOR,)
    expected_keys = set(candidates[0])
    if set(observed) != expected_keys:
        raise ValueError("CriticTool formal method descriptor key set mismatch")
    for candidate in candidates:
        if all(observed[key] == candidate[key] for key in candidate):
            return
    mismatched = tuple(
        sorted(
            key
            for key in expected_keys
            if all(observed[key] != candidate[key] for candidate in candidates)
        )
    )
    raise ValueError(
        "CriticTool formal method descriptor mismatch: "
        + ", ".join(mismatched)
    )


@dataclass(frozen=True)
class CriticToolBenchmarkAuthorizationV1:
    dataset_manifest_sha256: str
    platform_revision: str
    model_id: str
    execution_id: str
    method_descriptor: Mapping[str, object]
    runtime_attestation: TraceBenchmarkRuntimeAttestationV1
    run_spec: TraceBenchmarkRunSpecV1
    authorization_sha256: str
    schema: str = CRITICTOOL_BENCHMARK_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CRITICTOOL_BENCHMARK_AUTHORIZATION_SCHEMA:
            raise ValueError("unsupported CriticTool authorization schema")
        validate_critictool_formal_method_descriptor(
            self.method_descriptor,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
        )
        if not re.fullmatch(r"[0-9a-f]{40}", self.platform_revision):
            raise ValueError("CriticTool authorization platform revision is invalid")
        if not self.model_id or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.execution_id
        ):
            raise ValueError("CriticTool authorization identity is invalid")
        expected_run_identity = (
            self.dataset_manifest_sha256,
            self.platform_revision,
            self.model_id,
            ("critictool",),
            _method_descriptor_sha256(self.method_descriptor),
        )
        observed_run_identity = (
            self.run_spec.dataset_manifest_sha256,
            self.run_spec.platform_revision,
            self.run_spec.model_id,
            self.run_spec.method_ids,
            self.run_spec.method_descriptors_sha256,
        )
        if observed_run_identity != expected_run_identity:
            raise ValueError("CriticTool authorization run spec mismatch")
        if self.runtime_attestation.requested_model_id != self.model_id:
            raise ValueError("CriticTool authorization runtime model mismatch")
        expected_hash = "sha256:" + canonical_sha256(self.to_dict(include_sha256=False))
        if self.authorization_sha256 != expected_hash:
            raise ValueError("CriticTool authorization hash mismatch")

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "platform_revision": self.platform_revision,
            "model_id": self.model_id,
            "execution_id": self.execution_id,
            "method_descriptor": dict(self.method_descriptor),
            "runtime_attestation": self.runtime_attestation.to_dict(),
            "run_spec": self.run_spec.to_dict(),
        }
        if include_sha256:
            payload["authorization_sha256"] = self.authorization_sha256
        return payload


def build_critictool_trace_benchmark_authorization(
    *,
    dataset: CriticToolBenchmarkDatasetV1,
    model_id: str,
    execution_id: str,
    platform_revision: str,
    method_descriptor: Mapping[str, object],
    runtime_attestation: TraceBenchmarkRuntimeAttestationV1,
    run_spec: TraceBenchmarkRunSpecV1,
) -> CriticToolBenchmarkAuthorizationV1:
    unhashed: dict[str, object] = {
        "schema": CRITICTOOL_BENCHMARK_AUTHORIZATION_SCHEMA,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "platform_revision": platform_revision,
        "model_id": model_id,
        "execution_id": execution_id,
        "method_descriptor": dict(method_descriptor),
        "runtime_attestation": runtime_attestation.to_dict(),
        "run_spec": run_spec.to_dict(),
    }
    return CriticToolBenchmarkAuthorizationV1(
        dataset_manifest_sha256=dataset.manifest_sha256,
        platform_revision=platform_revision,
        model_id=model_id,
        execution_id=execution_id,
        method_descriptor=dict(method_descriptor),
        runtime_attestation=runtime_attestation,
        run_spec=run_spec,
        authorization_sha256="sha256:" + canonical_sha256(unhashed),
    )


def write_critictool_trace_benchmark_authorization(
    authorization: CriticToolBenchmarkAuthorizationV1,
    output_path: str | Path,
) -> None:
    atomic_write_json(output_path, authorization.to_dict(), overwrite=False)


def load_critictool_trace_benchmark_authorization(
    path: str | Path,
) -> CriticToolBenchmarkAuthorizationV1:
    raw = _mapping(
        json.loads(Path(path).read_text(encoding="utf-8")),
        label="CriticTool benchmark authorization",
    )
    return CriticToolBenchmarkAuthorizationV1(
        dataset_manifest_sha256=str(raw.get("dataset_manifest_sha256", "")),
        platform_revision=str(raw.get("platform_revision", "")),
        model_id=str(raw.get("model_id", "")),
        execution_id=str(raw.get("execution_id", "")),
        method_descriptor=_mapping(
            raw.get("method_descriptor"),
            label="CriticTool authorized method descriptor",
        ),
        runtime_attestation=_runtime_attestation_from_mapping(raw.get("runtime_attestation")),
        run_spec=_run_spec_from_mapping(raw.get("run_spec")),
        authorization_sha256=str(raw.get("authorization_sha256", "")),
        schema=str(raw.get("schema", "")),
    )


def _request_from_record(
    *,
    source_key: str,
    case_id: str,
    raw_record: object,
) -> TraceMethodRequestV1:
    record = _mapping(raw_record, label=f"CriticTool record {source_key}")
    raw_prompt = record.get("query_prompt")
    if not isinstance(raw_prompt, list) or len(raw_prompt) < 2:
        raise ValueError(f"CriticTool record {source_key} has invalid query prompt")
    events: list[TraceEventV1] = []
    for sequence, raw_message in enumerate(raw_prompt):
        message = _mapping(raw_message, label="CriticTool query message")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise ValueError(f"CriticTool record {source_key} has invalid message")
        events.append(
            TraceEventV1(
                event_id=f"event-{sequence:04d}",
                sequence=sequence,
                kind=role,
                content=content,
                parent_event_ids=() if sequence == 0 else (f"event-{sequence - 1:04d}",),
                metadata=(("role", role),),
            )
        )
    case = TraceCaseV1(
        case_id=case_id,
        task_id=source_key,
        events=tuple(events),
        interruption_event_id=events[-1].event_id,
        observed_outcome=TraceOutcomeV1(
            success=False,
            terminal_reason="critictool_labeled_internal_error",
        ),
    )
    return TraceMethodRequestV1(case=case)


def load_critictool_trace_benchmark_manifest(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path,
    expected_manifest_sha256: str,
) -> CriticToolBenchmarkDatasetV1:
    raw = _mapping(
        json.loads(Path(manifest_path).read_text(encoding="utf-8")),
        label="CriticTool benchmark manifest",
    )
    if raw.get("schema") != CRITICTOOL_BENCHMARK_MANIFEST_SCHEMA:
        raise ValueError("unsupported CriticTool benchmark manifest schema")
    manifest = dict(raw)
    observed_hash = manifest.pop("manifest_sha256", None)
    if not _SHA256_PATTERN.fullmatch(expected_manifest_sha256):
        raise ValueError("expected CriticTool manifest hash is invalid")
    if observed_hash != expected_manifest_sha256 or observed_hash != (
        "sha256:" + canonical_sha256(manifest)
    ):
        raise ValueError("CriticTool benchmark manifest hash mismatch")
    root = Path(dataset_root).resolve()
    selection = _mapping(
        manifest.get("selection"),
        label="CriticTool benchmark selection",
    )
    if selection.get("algorithm") != CRITICTOOL_SELECTION_ALGORITHM:
        raise ValueError("unsupported CriticTool selection algorithm")
    excluded = selection.get("excluded_source_keys", [])
    if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
        raise ValueError("CriticTool exclusion list is invalid")
    recomputed = _build_manifest(
        dataset_root=root,
        dataset_id=str(manifest.get("dataset_id", "")),
        seed=str(selection.get("seed", "")),
        per_error_type_count=selection.get("per_error_type_count"),
        excluded_source_keys=tuple(excluded),
    )
    if recomputed != raw:
        raise ValueError("CriticTool benchmark manifest does not match frozen source")
    _, records = _source_records(root)
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("CriticTool benchmark cases are invalid")
    cases: list[CriticToolBenchmarkCaseV1] = []
    for raw_case in raw_cases:
        entry = _mapping(raw_case, label="CriticTool benchmark case")
        source_key = str(entry.get("source_key", ""))
        gold = CriticToolBenchmarkGoldV1(
            error_type=str(entry.get("error_type", "")),
            recovery_tool=str(entry.get("recovery_tool", "")),
            recovery_arguments_json=str(entry.get("recovery_arguments_json", "")),
        )
        cases.append(
            CriticToolBenchmarkCaseV1(
                request=_request_from_record(
                    source_key=source_key,
                    case_id=str(entry.get("case_id", "")),
                    raw_record=records[source_key],
                ),
                gold=gold,
                source_key=source_key,
                source_record_sha256=str(entry.get("source_record_sha256", "")),
            )
        )
    return CriticToolBenchmarkDatasetV1(
        dataset_id=str(manifest.get("dataset_id", "")),
        source_revision=str(manifest.get("source_revision", "")),
        source_sha256=str(manifest.get("source_sha256", "")),
        manifest_sha256=expected_manifest_sha256,
        selection=tuple(sorted(selection.items())),
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class CriticToolBenchmarkScoreV1:
    error_type_exact: bool | None
    recovery_tool_exact: bool | None
    recovery_arguments_exact: bool | None
    full_recovery_exact: bool | None

    def __post_init__(self) -> None:
        if any(
            value is not None and not isinstance(value, bool)
            for value in (
                self.error_type_exact,
                self.recovery_tool_exact,
                self.recovery_arguments_exact,
                self.full_recovery_exact,
            )
        ):
            raise ValueError("CriticTool benchmark scores must be boolean or null")

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _gold(value: CriticToolBenchmarkGoldV1 | Mapping[str, object]) -> CriticToolBenchmarkGoldV1:
    if isinstance(value, CriticToolBenchmarkGoldV1):
        return value
    return CriticToolBenchmarkGoldV1(
        error_type=str(value.get("error_type", "")),
        recovery_tool=str(value.get("recovery_tool", "")),
        recovery_arguments_json=str(value.get("recovery_arguments_json", "")),
    )


def score_critictool_trace_receipt(
    receipt: TraceMethodRunReceiptV2,
    gold: CriticToolBenchmarkGoldV1 | Mapping[str, object],
) -> CriticToolBenchmarkScoreV1:
    expected = _gold(gold)
    if receipt.status is not TraceRunStatus.COMPLETED:
        return CriticToolBenchmarkScoreV1(None, None, None, None)
    classification_findings = tuple(
        finding
        for finding in receipt.findings
        if finding.kind is TraceFindingKind.ERROR_CLASSIFICATION
    )
    recovery_findings = tuple(
        finding
        for finding in receipt.findings
        if finding.kind is TraceFindingKind.RECOVERY_PROPOSAL
    )
    error_exact = (
        any(finding.label == expected.error_type for finding in classification_findings)
        if TraceFindingKind.ERROR_CLASSIFICATION in receipt.descriptor.output_kinds
        else None
    )
    tool_exact = (
        any(finding.label == expected.recovery_tool for finding in recovery_findings)
        if TraceFindingKind.RECOVERY_PROPOSAL in receipt.descriptor.output_kinds
        else None
    )
    arguments_exact = (
        any(
            dict(finding.attributes).get("recovery_arguments_json")
            == expected.recovery_arguments_json
            for finding in recovery_findings
        )
        if TraceFindingKind.RECOVERY_PROPOSAL in receipt.descriptor.output_kinds
        else None
    )
    full_recovery = (
        any(
            finding.label == expected.recovery_tool
            and dict(finding.attributes).get("recovery_arguments_json")
            == expected.recovery_arguments_json
            for finding in recovery_findings
        )
        if recovery_findings
        else (False if tool_exact is not None else None)
    )
    return CriticToolBenchmarkScoreV1(
        error_type_exact=error_exact,
        recovery_tool_exact=tool_exact,
        recovery_arguments_exact=arguments_exact,
        full_recovery_exact=full_recovery,
    )


@dataclass(frozen=True)
class CriticToolBenchmarkPredictionV1:
    kind: str
    label: str
    recovery_arguments_json: str | None

    def __post_init__(self) -> None:
        if self.kind not in {
            TraceFindingKind.ERROR_CLASSIFICATION.value,
            TraceFindingKind.RECOVERY_PROPOSAL.value,
        }:
            raise ValueError("CriticTool benchmark prediction kind is unsupported")
        if not self.label:
            raise ValueError("CriticTool benchmark prediction label is empty")

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class CriticToolBenchmarkCellV1:
    case_id: str
    source_key: str
    source_record_sha256: str
    request_sha256: str
    method_id: str
    status: str
    response_model_id: str | None
    receipt_sha256: str
    score: CriticToolBenchmarkScoreV1
    predictions: tuple[CriticToolBenchmarkPredictionV1, ...]
    blockers: tuple[str, ...]
    diagnostics: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id or not self.source_key or not self.method_id:
            raise ValueError("CriticTool benchmark cell identity is incomplete")
        if self.status not in {status.value for status in TraceRunStatus}:
            raise ValueError("CriticTool benchmark cell status is invalid")
        if self.status == TraceRunStatus.COMPLETED.value and not (self.response_model_id):
            raise ValueError("completed CriticTool cell lacks response model id")
        if not all(
            _SHA256_PATTERN.fullmatch(value)
            for value in (
                self.source_record_sha256,
                self.request_sha256,
                self.receipt_sha256,
            )
        ):
            raise ValueError("CriticTool benchmark cell hash is invalid")
        for event in self.diagnostics:
            TraceDiagnosticEventV1.from_dict(event)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "source_key": self.source_key,
            "source_record_sha256": self.source_record_sha256,
            "request_sha256": self.request_sha256,
            "method_id": self.method_id,
            "status": self.status,
            "response_model_id": self.response_model_id,
            "receipt_sha256": self.receipt_sha256,
            "score": self.score.to_dict(),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "blockers": list(self.blockers),
            "diagnostics": [dict(event) for event in self.diagnostics],
        }


@dataclass(frozen=True)
class CriticToolBenchmarkAggregateV1:
    method_id: str
    case_count: int
    completed_count: int
    blocked_count: int
    invalid_count: int
    error_type_evaluable_count: int
    error_type_exact_count: int
    recovery_tool_evaluable_count: int
    recovery_tool_exact_count: int
    recovery_arguments_evaluable_count: int
    recovery_arguments_exact_count: int
    full_recovery_evaluable_count: int
    full_recovery_exact_count: int

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = self.__dict__.copy()
        # Fixed-denominator rates divide by every admitted case so non-completed
        # cells count as misses; without them a partially executed method could
        # be scored on its survivors alone.
        for name in (
            "error_type",
            "recovery_tool",
            "recovery_arguments",
            "full_recovery",
        ):
            exact = payload[f"{name}_exact_count"]
            payload[f"{name}_fixed_denominator_rate"] = (
                exact / self.case_count if self.case_count else None
            )
        return payload


@dataclass(frozen=True)
class CriticToolBenchmarkResultV1:
    dataset_id: str
    dataset_manifest_sha256: str
    dataset_source_revision: str
    dataset_source_sha256: str
    platform_revision: str
    model_id: str
    execution_id: str
    authorization_sha256: str
    method_ids: tuple[str, ...]
    method_descriptors: tuple[Mapping[str, object], ...]
    runtime_attestation: TraceBenchmarkRuntimeAttestationV1
    run_spec: TraceBenchmarkRunSpecV1
    cells: tuple[CriticToolBenchmarkCellV1, ...]
    aggregates: tuple[CriticToolBenchmarkAggregateV1, ...]
    result_sha256: str
    schema: str = CRITICTOOL_BENCHMARK_RESULT_SCHEMA

    def __post_init__(self) -> None:
        expected = "sha256:" + canonical_sha256(self.to_dict(include_sha256=False))
        if self.result_sha256 != expected:
            raise ValueError("CriticTool benchmark result hash mismatch")

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_source_revision": self.dataset_source_revision,
            "dataset_source_sha256": self.dataset_source_sha256,
            "platform_revision": self.platform_revision,
            "model_id": self.model_id,
            "execution_id": self.execution_id,
            "authorization_sha256": self.authorization_sha256,
            "method_ids": list(self.method_ids),
            "method_descriptors": [dict(item) for item in self.method_descriptors],
            "runtime_attestation": self.runtime_attestation.to_dict(),
            "run_spec": self.run_spec.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
            "aggregates": [aggregate.to_dict() for aggregate in self.aggregates],
        }
        if include_sha256:
            payload["result_sha256"] = self.result_sha256
        return payload


def _prediction_summaries(
    receipt: TraceMethodRunReceiptV2,
) -> tuple[CriticToolBenchmarkPredictionV1, ...]:
    summaries: list[CriticToolBenchmarkPredictionV1] = []
    for finding in receipt.findings:
        if finding.kind not in {
            TraceFindingKind.ERROR_CLASSIFICATION,
            TraceFindingKind.RECOVERY_PROPOSAL,
        }:
            continue
        summaries.append(
            CriticToolBenchmarkPredictionV1(
                kind=finding.kind.value,
                label=finding.label,
                recovery_arguments_json=(
                    dict(finding.attributes).get("recovery_arguments_json")
                    if finding.kind is TraceFindingKind.RECOVERY_PROPOSAL
                    else None
                ),
            )
        )
    return tuple(summaries)


def _receipt_response_model_id(receipt: TraceMethodRunReceiptV2) -> str | None:
    response_model_ids = {
        value
        for finding in receipt.findings
        for key, value in finding.attributes
        if key == "response_model_id" and value
    }
    if receipt.status is TraceRunStatus.COMPLETED and len(response_model_ids) != 1:
        raise ValueError("completed CriticTool receipt must attest exactly one response model")
    if len(response_model_ids) > 1:
        raise ValueError("CriticTool receipt contains conflicting response models")
    return next(iter(response_model_ids), None)


def _aggregate_cells(
    method_id: str,
    cells: tuple[CriticToolBenchmarkCellV1, ...],
) -> CriticToolBenchmarkAggregateV1:
    selected = tuple(cell for cell in cells if cell.method_id == method_id)
    names = (
        "error_type",
        "recovery_tool",
        "recovery_arguments",
        "full_recovery",
    )
    score_values = {
        name: tuple(
            value
            for cell in selected
            if (value := getattr(cell.score, f"{name}_exact")) is not None
        )
        for name in names
    }
    return CriticToolBenchmarkAggregateV1(
        method_id=method_id,
        case_count=len(selected),
        completed_count=sum(cell.status == TraceRunStatus.COMPLETED.value for cell in selected),
        blocked_count=sum(cell.status == TraceRunStatus.BLOCKED.value for cell in selected),
        invalid_count=sum(cell.status == TraceRunStatus.INVALID.value for cell in selected),
        error_type_evaluable_count=len(score_values["error_type"]),
        error_type_exact_count=sum(score_values["error_type"]),
        recovery_tool_evaluable_count=len(score_values["recovery_tool"]),
        recovery_tool_exact_count=sum(score_values["recovery_tool"]),
        recovery_arguments_evaluable_count=len(score_values["recovery_arguments"]),
        recovery_arguments_exact_count=sum(score_values["recovery_arguments"]),
        full_recovery_evaluable_count=len(score_values["full_recovery"]),
        full_recovery_exact_count=sum(score_values["full_recovery"]),
    )


def run_critictool_trace_benchmark(
    *,
    platform: TraceMethodPlatform,
    dataset: CriticToolBenchmarkDatasetV1,
    authorization: CriticToolBenchmarkAuthorizationV1,
    runtime_attestation: TraceBenchmarkRuntimeAttestationV1,
    run_spec: TraceBenchmarkRunSpecV1,
    progress_callback: (Callable[[CriticToolBenchmarkCellV1, int, int], None] | None) = None,
) -> CriticToolBenchmarkResultV1:
    model_id = authorization.model_id
    execution_id = authorization.execution_id
    method_ids = ("critictool",)
    platform_revision = authorization.platform_revision
    if authorization.dataset_manifest_sha256 != dataset.manifest_sha256:
        raise ValueError("CriticTool benchmark authorization dataset mismatch")
    if runtime_attestation.to_dict() != authorization.runtime_attestation.to_dict():
        raise ValueError("CriticTool benchmark runtime is not authorized")
    if run_spec.to_dict() != authorization.run_spec.to_dict():
        raise ValueError("CriticTool benchmark run spec is not authorized")
    descriptors = {item.method_id: item for item in platform.catalog()}
    unknown = set(method_ids) - descriptors.keys()
    if unknown:
        raise ValueError(f"CriticTool benchmark methods are unknown: {sorted(unknown)}")
    expected_identity = (
        dataset.manifest_sha256,
        model_id,
        method_ids,
        platform_revision,
    )
    observed_identity = (
        run_spec.dataset_manifest_sha256,
        run_spec.model_id,
        run_spec.method_ids,
        run_spec.platform_revision,
    )
    if observed_identity != expected_identity:
        raise ValueError("CriticTool benchmark run spec identity mismatch")
    if runtime_attestation.requested_model_id != model_id:
        raise ValueError("CriticTool benchmark runtime model mismatch")
    method_descriptors = tuple(descriptors[method_id].to_dict() for method_id in method_ids)
    validate_critictool_formal_method_descriptor(
        method_descriptors[0],
        dataset_manifest_sha256=dataset.manifest_sha256,
    )
    if method_descriptors != (dict(authorization.method_descriptor),):
        raise ValueError("CriticTool benchmark method descriptor is not authorized")
    expected_descriptor_hash = "sha256:" + canonical_sha256(
        {"method_descriptors": [dict(item) for item in method_descriptors]}
    )
    if run_spec.method_descriptors_sha256 != expected_descriptor_hash:
        raise ValueError("CriticTool benchmark method provenance mismatch")

    cells: list[CriticToolBenchmarkCellV1] = []
    total = len(dataset.cases) * len(method_ids)
    for item in dataset.cases:
        for method_id in method_ids:
            receipt = platform.run(method_id, item.request)
            cell = CriticToolBenchmarkCellV1(
                case_id=item.request.case.case_id,
                source_key=item.source_key,
                source_record_sha256=item.source_record_sha256,
                request_sha256=item.request.sha256,
                method_id=method_id,
                status=receipt.status.value,
                response_model_id=_receipt_response_model_id(receipt),
                receipt_sha256=receipt.receipt_sha256,
                score=score_critictool_trace_receipt(receipt, item.gold),
                predictions=_prediction_summaries(receipt),
                blockers=receipt.blockers,
                diagnostics=tuple(event.to_dict() for event in receipt.diagnostics),
            )
            cells.append(cell)
            if progress_callback is not None:
                progress_callback(cell, len(cells), total)
    frozen_cells = tuple(cells)
    aggregates = tuple(_aggregate_cells(method_id, frozen_cells) for method_id in method_ids)
    unhashed: dict[str, object] = {
        "schema": CRITICTOOL_BENCHMARK_RESULT_SCHEMA,
        "dataset_id": dataset.dataset_id,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_source_revision": dataset.source_revision,
        "dataset_source_sha256": dataset.source_sha256,
        "platform_revision": platform_revision,
        "model_id": model_id,
        "execution_id": execution_id,
        "authorization_sha256": authorization.authorization_sha256,
        "method_ids": list(method_ids),
        "method_descriptors": [dict(item) for item in method_descriptors],
        "runtime_attestation": runtime_attestation.to_dict(),
        "run_spec": run_spec.to_dict(),
        "cells": [cell.to_dict() for cell in frozen_cells],
        "aggregates": [aggregate.to_dict() for aggregate in aggregates],
    }
    return CriticToolBenchmarkResultV1(
        dataset_id=dataset.dataset_id,
        dataset_manifest_sha256=dataset.manifest_sha256,
        dataset_source_revision=dataset.source_revision,
        dataset_source_sha256=dataset.source_sha256,
        platform_revision=platform_revision,
        model_id=model_id,
        execution_id=execution_id,
        authorization_sha256=authorization.authorization_sha256,
        method_ids=method_ids,
        method_descriptors=method_descriptors,
        runtime_attestation=runtime_attestation,
        run_spec=run_spec,
        cells=frozen_cells,
        aggregates=aggregates,
        result_sha256="sha256:" + canonical_sha256(unhashed),
    )


def write_critictool_trace_benchmark_result(
    result: CriticToolBenchmarkResultV1,
    output_path: str | Path,
) -> None:
    atomic_write_json(output_path, result.to_dict(), overwrite=False)


def _verified_critictool_result(path: str | Path) -> Mapping[str, Any]:
    raw = dict(
        _mapping(
            json.loads(Path(path).read_text(encoding="utf-8")),
            label="CriticTool benchmark result",
        )
    )
    if raw.get("schema") not in {
        CRITICTOOL_BENCHMARK_RESULT_SCHEMA,
        PREVIOUS_CRITICTOOL_BENCHMARK_RESULT_SCHEMA,
    }:
        raise ValueError("unsupported CriticTool benchmark result schema")
    observed_hash = raw.get("result_sha256")
    unhashed = dict(raw)
    unhashed.pop("result_sha256", None)
    if observed_hash != "sha256:" + canonical_sha256(unhashed):
        raise ValueError("CriticTool benchmark result hash mismatch")
    run_spec = dict(_mapping(raw.get("run_spec"), label="CriticTool benchmark run spec"))
    observed_run_spec_hash = run_spec.pop("run_spec_sha256", None)
    if observed_run_spec_hash != "sha256:" + canonical_sha256(run_spec):
        raise ValueError("CriticTool benchmark run spec hash mismatch")
    return raw


def _critictool_cell_diagnostic_failures(
    cell: Mapping[str, Any],
    *,
    result_path: Path,
) -> tuple[str, ...]:
    raw_diagnostics = cell.get("diagnostics")
    if not isinstance(raw_diagnostics, list) or not all(
        isinstance(event, Mapping) for event in raw_diagnostics
    ):
        return ("diagnostics_invalid",)
    try:
        diagnostics = tuple(
            TraceDiagnosticEventV1.from_dict(event) for event in raw_diagnostics
        )
    except (TypeError, ValueError):
        return ("diagnostics_invalid",)
    failures: list[str] = []
    stages = tuple(event.stage for event in diagnostics)
    if len(stages) != len(set(stages)):
        failures.append("diagnostic_stage_duplicate")
    failed = tuple(
        event for event in diagnostics
        if event.status is TraceDiagnosticStatus.FAILED
    )
    status = str(cell.get("status", ""))
    if status == TraceRunStatus.COMPLETED.value:
        required = {
            TraceDiagnosticStage.WORKER_LAUNCH,
            TraceDiagnosticStage.MODEL_TRANSPORT,
            TraceDiagnosticStage.MODEL_RESPONSE,
            TraceDiagnosticStage.UPSTREAM_PARSE,
            TraceDiagnosticStage.UPSTREAM_VALIDATION,
            TraceDiagnosticStage.ARTIFACT_PUBLICATION,
            TraceDiagnosticStage.ADAPTER_VALIDATION,
            TraceDiagnosticStage.PLATFORM_VALIDATION,
        }
        if failed:
            failures.append("completed_cell_has_failed_diagnostic")
        if not required.issubset(stages):
            failures.append("completed_cell_diagnostic_stage_missing")
    elif status == TraceRunStatus.INVALID.value and len(failed) != 1:
        failures.append("invalid_cell_failure_stage_count_mismatch")
    elif status == TraceRunStatus.BLOCKED.value and (
        len(failed) != 1 or failed[0].stage is not TraceDiagnosticStage.ADMISSION
    ):
        failures.append("blocked_cell_admission_diagnostic_mismatch")

    response_events = tuple(
        event
        for event in diagnostics
        if event.stage is TraceDiagnosticStage.MODEL_RESPONSE
        and event.status is TraceDiagnosticStatus.COMPLETED
    )
    if len(response_events) > 1:
        failures.append("model_response_diagnostic_count_mismatch")
    for event in response_events:
        raw_output_sha256 = dict(event.attributes).get("raw_output_sha256")
        artifacts = tuple(
            artifact
            for artifact in event.artifacts
            if artifact.artifact_id == "model_response"
        )
        if len(artifacts) != 1:
            failures.append("model_response_artifact_count_mismatch")
            continue
        artifact = artifacts[0]
        if artifact.content_sha256 != raw_output_sha256:
            failures.append("model_response_artifact_hash_binding_mismatch")
            continue
        uri = Path(artifact.uri)
        if uri.is_absolute() or ".." in uri.parts:
            failures.append("model_response_artifact_uri_unsafe")
            continue
        parent = result_path.resolve().parent
        target = (parent / uri).resolve()
        if not target.is_relative_to(parent) or not target.is_file():
            failures.append("model_response_artifact_missing")
            continue
        if _sha256_file(target) != artifact.content_sha256:
            failures.append("model_response_artifact_content_mismatch")
        if target.stat().st_size != artifact.byte_count:
            failures.append("model_response_artifact_size_mismatch")
    return tuple(sorted(set(failures)))


def build_critictool_trace_benchmark_gate_audit(
    result_paths: Sequence[str | Path],
    *,
    authorization_paths: Sequence[str | Path],
    dataset: CriticToolBenchmarkDatasetV1,
    expected_model_ids: tuple[str, ...],
    expected_platform_revision: str,
    minimum_completed_per_model: int = 10,
) -> dict[str, object]:
    if len(expected_model_ids) < 2 or len(expected_model_ids) != len(set(expected_model_ids)):
        raise ValueError("CriticTool gate requires at least two unique models")
    if len(result_paths) != len(expected_model_ids) or len(
        authorization_paths
    ) != len(expected_model_ids):
        raise ValueError(
            "CriticTool gate requires exactly one result and authorization per model"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", expected_platform_revision):
        raise ValueError("CriticTool gate platform revision is invalid")
    if minimum_completed_per_model < 1 or minimum_completed_per_model > len(dataset.cases):
        raise ValueError("CriticTool gate minimum completed count is invalid")

    expected_cases = {
        item.request.case.case_id: (
            item.source_key,
            item.source_record_sha256,
            item.request.sha256,
        )
        for item in dataset.cases
    }
    resolved_result_paths = tuple(Path(path).resolve() for path in result_paths)
    results = tuple(_verified_critictool_result(path) for path in resolved_result_paths)
    authorizations = tuple(
        load_critictool_trace_benchmark_authorization(path)
        for path in authorization_paths
    )
    observed_models = tuple(str(result.get("model_id", "")) for result in results)
    if set(observed_models) != set(expected_model_ids) or len(observed_models) != len(
        set(observed_models)
    ):
        raise ValueError("CriticTool gate result models do not match frozen models")
    authorization_models = tuple(item.model_id for item in authorizations)
    if set(authorization_models) != set(expected_model_ids) or len(
        authorization_models
    ) != len(set(authorization_models)):
        raise ValueError(
            "CriticTool gate authorization models do not match frozen models"
        )

    model_audits: list[dict[str, object]] = []
    all_failures: list[str] = []
    completed_total = 0
    for model_id in expected_model_ids:
        result = next(item for item in results if str(item.get("model_id", "")) == model_id)
        result_path = next(
            path
            for path, item in zip(resolved_result_paths, results, strict=True)
            if str(item.get("model_id", "")) == model_id
        )
        authorization = next(
            item for item in authorizations if item.model_id == model_id
        )
        failures: list[str] = []
        if (
            dataset.manifest_sha256
            in {
                CRITICTOOL_V2_FROZEN_MANIFEST_SHA256,
                CRITICTOOL_V3_FROZEN_MANIFEST_SHA256,
            }
            and result.get("schema") != CRITICTOOL_BENCHMARK_RESULT_SCHEMA
        ):
            failures.append("diagnostic_result_schema_mismatch")
        if "@" not in model_id:
            failures.append("model_id_not_revision_qualified")
        if result.get("dataset_manifest_sha256") != dataset.manifest_sha256:
            failures.append("dataset_manifest_mismatch")
        if result.get("platform_revision") != expected_platform_revision:
            failures.append("platform_revision_mismatch")
        if result.get("authorization_sha256") != authorization.authorization_sha256:
            failures.append("authorization_hash_mismatch")
        if authorization.dataset_manifest_sha256 != dataset.manifest_sha256:
            failures.append("authorization_dataset_mismatch")
        if authorization.platform_revision != expected_platform_revision:
            failures.append("authorization_platform_mismatch")
        if authorization.execution_id != result.get("execution_id"):
            failures.append("authorization_execution_mismatch")
        if result.get("method_ids") != ["critictool"]:
            failures.append("method_contract_mismatch")
        descriptors = result.get("method_descriptors")
        if not isinstance(descriptors, list) or len(descriptors) != 1:
            failures.append("method_descriptor_count_mismatch")
            descriptors = []
        elif not isinstance(descriptors[0], Mapping) or (
            descriptors[0].get("method_id") != "critictool"
        ):
            failures.append("method_descriptor_identity_mismatch")
        else:
            try:
                validate_critictool_formal_method_descriptor(
                    descriptors[0],
                    dataset_manifest_sha256=dataset.manifest_sha256,
                )
            except ValueError:
                failures.append("formal_method_descriptor_mismatch")

        run_spec = _mapping(result.get("run_spec"), label="CriticTool run spec")
        if run_spec.get("model_id") != model_id:
            failures.append("run_spec_model_mismatch")
        if run_spec.get("dataset_manifest_sha256") != dataset.manifest_sha256:
            failures.append("run_spec_dataset_mismatch")
        if run_spec.get("platform_revision") != expected_platform_revision:
            failures.append("run_spec_platform_mismatch")
        if run_spec.get("method_ids") != ["critictool"]:
            failures.append("run_spec_method_mismatch")
        if dict(run_spec) != authorization.run_spec.to_dict():
            failures.append("authorization_run_spec_mismatch")
        observed_settings = (
            run_spec.get("max_tokens"),
            run_spec.get("timeout_seconds"),
            run_spec.get("temperature"),
            run_spec.get("generation_seed"),
        )
        frozen_settings = (
            CRITICTOOL_FROZEN_MAX_TOKENS,
            CRITICTOOL_FROZEN_TIMEOUT_SECONDS,
            CRITICTOOL_FROZEN_TEMPERATURE,
            CRITICTOOL_FROZEN_GENERATION_SEED,
        )
        if observed_settings != frozen_settings:
            failures.append("frozen_generation_settings_mismatch")
        if descriptors and run_spec.get("method_descriptors_sha256") != (
            "sha256:" + canonical_sha256({"method_descriptors": [dict(descriptors[0])]})
        ):
            failures.append("run_spec_method_descriptor_hash_mismatch")
        if descriptors and dict(descriptors[0]) != dict(
            authorization.method_descriptor
        ):
            failures.append("authorization_method_descriptor_mismatch")

        attestation = _mapping(
            result.get("runtime_attestation"),
            label="CriticTool runtime attestation",
        )
        if attestation.get("requested_model_id") != model_id:
            failures.append("runtime_attestation_model_mismatch")
        if dict(attestation) != authorization.runtime_attestation.to_dict():
            failures.append("authorization_runtime_attestation_mismatch")

        raw_cells = result.get("cells")
        if not isinstance(raw_cells, list):
            raise ValueError("CriticTool gate result cells are invalid")
        cells_by_id: dict[str, Mapping[str, Any]] = {}
        for raw_cell in raw_cells:
            cell = _mapping(raw_cell, label="CriticTool benchmark cell")
            case_id = str(cell.get("case_id", ""))
            if not case_id or case_id in cells_by_id:
                failures.append("duplicate_or_empty_case_id")
                continue
            cells_by_id[case_id] = cell
        if set(cells_by_id) != set(expected_cases):
            failures.append("frozen_case_set_mismatch")

        completed_count = 0
        for case_id, cell in cells_by_id.items():
            expected_case = expected_cases.get(case_id)
            if expected_case is None:
                continue
            source_identity = (
                cell.get("source_key"),
                cell.get("source_record_sha256"),
                cell.get("request_sha256"),
            )
            if source_identity != expected_case:
                failures.append(f"{case_id}:case_provenance_mismatch")
            if cell.get("method_id") != "critictool":
                failures.append(f"{case_id}:method_mismatch")
            if dataset.manifest_sha256 in {
                CRITICTOOL_V2_FROZEN_MANIFEST_SHA256,
                CRITICTOOL_V3_FROZEN_MANIFEST_SHA256,
            }:
                failures.extend(
                    f"{case_id}:{failure}"
                    for failure in _critictool_cell_diagnostic_failures(
                        cell,
                        result_path=result_path,
                    )
                )
            status = cell.get("status")
            if status != TraceRunStatus.COMPLETED.value:
                continue
            completed_count += 1
            if cell.get("response_model_id") != model_id:
                failures.append(f"{case_id}:response_model_mismatch")
            predictions = cell.get("predictions")
            if not isinstance(predictions, list):
                failures.append(f"{case_id}:predictions_invalid")
                continue
            classifications = [
                item
                for item in predictions
                if isinstance(item, Mapping)
                and item.get("kind") == TraceFindingKind.ERROR_CLASSIFICATION.value
            ]
            recoveries = [
                item
                for item in predictions
                if isinstance(item, Mapping)
                and item.get("kind") == TraceFindingKind.RECOVERY_PROPOSAL.value
            ]
            if len(classifications) != 1 or (
                classifications[0].get("label") not in CRITICTOOL_ERROR_TYPES
            ):
                failures.append(f"{case_id}:taxonomy_classification_invalid")
            if len(recoveries) != 1 or not recoveries[0].get("label"):
                failures.append(f"{case_id}:structured_recovery_invalid")
            else:
                raw_arguments = recoveries[0].get("recovery_arguments_json")
                try:
                    parsed_arguments = json.loads(str(raw_arguments))
                    canonical_arguments = _canonical_arguments(parsed_arguments)
                except (json.JSONDecodeError, ValueError):
                    failures.append(f"{case_id}:structured_recovery_invalid")
                else:
                    if raw_arguments != canonical_arguments:
                        failures.append(f"{case_id}:structured_recovery_not_canonical")
        completed_total += completed_count
        if completed_count < minimum_completed_per_model:
            failures.append("minimum_completed_cells_not_met")
        all_failures.extend(f"{model_id}:{reason}" for reason in failures)
        model_audits.append(
            {
                "model_id": model_id,
                "execution_id": result.get("execution_id"),
                "authorization_sha256": authorization.authorization_sha256,
                "result_sha256": result.get("result_sha256"),
                "expected_cell_count": len(dataset.cases),
                "completed_cell_count": completed_count,
                "completed_coverage_rate": (
                    completed_count / len(dataset.cases) if dataset.cases else None
                ),
                "gate_passed": not failures,
                "failure_reasons": failures,
            }
        )

    unhashed: dict[str, object] = {
        "schema": CRITICTOOL_BENCHMARK_GATE_AUDIT_SCHEMA,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "platform_revision": expected_platform_revision,
        "expected_model_ids": list(expected_model_ids),
        "minimum_completed_per_model": minimum_completed_per_model,
        "expected_cell_count": len(dataset.cases) * len(expected_model_ids),
        "completed_cell_count": completed_total,
        "aggregate_denominator_policy": (
            "per-metric fixed_denominator_rate divides by all admitted cases so "
            "non-completed cells count as misses; minimum_completed_per_model still "
            "gates publication below full coverage"
        ),
        "gate_passed": not all_failures,
        "failure_reasons": all_failures,
        "models": model_audits,
    }
    return {
        **unhashed,
        "gate_audit_sha256": "sha256:" + canonical_sha256(unhashed),
    }


def write_critictool_trace_benchmark_gate_audit(
    audit: Mapping[str, object],
    output_path: str | Path,
) -> None:
    if audit.get("schema") != CRITICTOOL_BENCHMARK_GATE_AUDIT_SCHEMA:
        raise ValueError("unsupported CriticTool gate audit schema")
    observed_hash = audit.get("gate_audit_sha256")
    unhashed = dict(audit)
    unhashed.pop("gate_audit_sha256", None)
    if observed_hash != "sha256:" + canonical_sha256(unhashed):
        raise ValueError("CriticTool gate audit hash mismatch")
    atomic_write_json(output_path, dict(audit), overwrite=False)


__all__ = [
    "CRITICTOOL_BENCHMARK_AUTHORIZATION_SCHEMA",
    "CRITICTOOL_BENCHMARK_GATE_AUDIT_SCHEMA",
    "CRITICTOOL_BENCHMARK_MANIFEST_SCHEMA",
    "CRITICTOOL_BENCHMARK_RESULT_SCHEMA",
    "CRITICTOOL_ERROR_TYPES",
    "CRITICTOOL_FROZEN_GENERATION_SEED",
    "CRITICTOOL_FROZEN_MANIFEST_SHA256",
    "CRITICTOOL_FROZEN_MANIFEST_SHA256S",
    "CRITICTOOL_FROZEN_MAX_TOKENS",
    "CRITICTOOL_FROZEN_METHOD_DESCRIPTOR",
    "CRITICTOOL_LEGACY_COT_METHOD_DESCRIPTOR",
    "CRITICTOOL_FROZEN_MINIMUM_COMPLETED_PER_MODEL",
    "CRITICTOOL_FROZEN_MODEL_IDS",
    "CRITICTOOL_V2_FROZEN_MANIFEST_SHA256",
    "CRITICTOOL_V3_FROZEN_MANIFEST_SHA256",
    "CRITICTOOL_FROZEN_SOURCE_REVISION",
    "CRITICTOOL_FROZEN_TEMPERATURE",
    "CRITICTOOL_FROZEN_TIMEOUT_SECONDS",
    "CRITICTOOL_SELECTION_ALGORITHM",
    "CriticToolBenchmarkCaseV1",
    "CriticToolBenchmarkDatasetV1",
    "CriticToolBenchmarkGoldV1",
    "CriticToolBenchmarkScoreV1",
    "CriticToolBenchmarkAggregateV1",
    "CriticToolBenchmarkAuthorizationV1",
    "CriticToolBenchmarkCellV1",
    "CriticToolBenchmarkPredictionV1",
    "CriticToolBenchmarkResultV1",
    "build_critictool_trace_benchmark_authorization",
    "build_critictool_trace_benchmark_gate_audit",
    "load_critictool_trace_benchmark_authorization",
    "load_critictool_trace_benchmark_manifest",
    "prepare_critictool_trace_benchmark_manifest",
    "run_critictool_trace_benchmark",
    "score_critictool_trace_receipt",
    "validate_critictool_formal_method_descriptor",
    "write_critictool_trace_benchmark_authorization",
    "write_critictool_trace_benchmark_gate_audit",
    "write_critictool_trace_benchmark_result",
]
