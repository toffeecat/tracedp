"""Auditable, prompt-free case inventory for unified tracer benchmark results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.unified_benchmark import (
    LEGACY_TRACE_BENCHMARK_RESULT_SCHEMA,
    PREVIOUS_TRACE_BENCHMARK_RESULT_SCHEMA,
    TRACE_BENCHMARK_RESULT_SCHEMA,
    TraceBenchmarkDatasetV1,
)

TRACE_BENCHMARK_CASE_INVENTORY_SCHEMA = "lh_trace_benchmark_case_inventory/v1"
SAFE_FINDING_ATTRIBUTE_KEYS = frozenset(
    {
        "agent_name",
        "model_calls",
        "observation_reading",
        "port",
        "raw_output_sha256",
        "response_model_id",
        "resolved_by",
        "score_caller",
        "score_planner",
        "score_selector",
        "score_synthesizer",
        "self_verification",
        "strategic_backtracking",
        "task_decomposition",
        "taxonomy",
        "total_tokens",
        "upstream_variant",
    }
)


@dataclass(frozen=True)
class FrozenTraceBenchmarkCaseIdentityV1:
    case_id: str
    source_path: str
    source_sha256: str
    request_sha256: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.source_path:
            raise ValueError("frozen trace benchmark case identity is incomplete")
        for value in (self.source_sha256, self.request_sha256):
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise ValueError("frozen trace benchmark case hash is invalid")


def frozen_trace_benchmark_case_identities(
    dataset: TraceBenchmarkDatasetV1,
) -> tuple[FrozenTraceBenchmarkCaseIdentityV1, ...]:
    return tuple(
        FrozenTraceBenchmarkCaseIdentityV1(
            case_id=item.request.case.case_id,
            source_path=item.source_path,
            source_sha256=item.source_sha256,
            request_sha256=item.request.sha256,
        )
        for item in dataset.cases
    )


def validate_frozen_trace_benchmark_result(
    result: Mapping[str, Any],
    *,
    expected_dataset_manifest_sha256: str,
    expected_model_ids: tuple[str, ...],
    expected_method_ids: tuple[str, ...],
    expected_cases: tuple[FrozenTraceBenchmarkCaseIdentityV1, ...],
) -> None:
    if result.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError("trace benchmark result does not match frozen manifest")
    if result.get("model_id") not in expected_model_ids:
        raise ValueError("trace benchmark result does not match frozen models")
    if result.get("method_ids") != list(expected_method_ids):
        raise ValueError("trace benchmark result does not match frozen methods")
    cells = result.get("cells")
    if not isinstance(cells, list):
        raise ValueError("trace benchmark result cells are invalid")
    observed: dict[tuple[str, str], tuple[str, str, str]] = {}
    for raw_cell in cells:
        cell = _mapping(raw_cell, label="trace benchmark frozen cell")
        key = (str(cell.get("case_id", "")), str(cell.get("method_id", "")))
        observed[key] = (
            str(cell.get("source_path", "")),
            str(cell.get("source_sha256", "")),
            str(cell.get("request_sha256", "")),
        )
    expected = {
        (case.case_id, method_id): (
            case.source_path,
            case.source_sha256,
            case.request_sha256,
        )
        for case in expected_cases
        for method_id in expected_method_ids
    }
    if observed != expected:
        raise ValueError("trace benchmark result does not match frozen case identities")


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _load_verified_result(path: str | Path) -> Mapping[str, Any]:
    payload = _mapping(
        json.loads(Path(path).read_text(encoding="utf-8")),
        label="trace benchmark result",
    )
    if payload.get("schema") not in {
        LEGACY_TRACE_BENCHMARK_RESULT_SCHEMA,
        PREVIOUS_TRACE_BENCHMARK_RESULT_SCHEMA,
        TRACE_BENCHMARK_RESULT_SCHEMA,
    }:
        raise ValueError("unsupported trace benchmark result schema")
    claimed = payload.get("result_sha256")
    unhashed = dict(payload)
    unhashed.pop("result_sha256", None)
    actual = "sha256:" + canonical_sha256(unhashed)
    if claimed != actual:
        raise ValueError("trace benchmark result hash mismatch")
    method_ids = payload.get("method_ids")
    cells = payload.get("cells")
    if (
        not isinstance(method_ids, list)
        or not method_ids
        or not all(isinstance(item, str) and item for item in method_ids)
        or len(method_ids) != len(set(method_ids))
        or not isinstance(cells, list)
    ):
        raise ValueError("trace benchmark result matrix identity is invalid")
    pairs: list[tuple[str, str]] = []
    for raw_cell in cells:
        cell = _mapping(raw_cell, label="trace benchmark result cell")
        pairs.append((str(cell.get("case_id", "")), str(cell.get("method_id", ""))))
    case_ids = sorted({case_id for case_id, _ in pairs})
    expected_pairs = {
        (case_id, method_id) for case_id in case_ids for method_id in method_ids
    }
    if (
        not case_ids
        or len(pairs) != len(set(pairs))
        or set(pairs) != expected_pairs
    ):
        raise ValueError("trace benchmark result matrix is incomplete")
    if payload.get("schema") in {
        PREVIOUS_TRACE_BENCHMARK_RESULT_SCHEMA,
        TRACE_BENCHMARK_RESULT_SCHEMA,
    }:
        run_spec = _mapping(payload.get("run_spec"), label="trace benchmark run spec")
        run_spec_unhashed = dict(run_spec)
        claimed_run_spec = run_spec_unhashed.pop("run_spec_sha256", None)
        if claimed_run_spec != "sha256:" + canonical_sha256(run_spec_unhashed):
            raise ValueError("trace benchmark run spec hash mismatch")
        method_descriptors = payload.get("method_descriptors")
        descriptor_hash = "sha256:" + canonical_sha256(
            {"method_descriptors": method_descriptors}
        )
        expected_identity = {
            "dataset_manifest_sha256": payload.get("dataset_manifest_sha256"),
            "model_id": payload.get("model_id"),
            "method_ids": method_ids,
            "platform_revision": payload.get("platform_revision"),
            "method_descriptors_sha256": descriptor_hash,
        }
        if any(run_spec.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("trace benchmark run spec provenance mismatch")
        if not isinstance(payload.get("execution_id"), str) or not payload.get(
            "execution_id"
        ):
            raise ValueError("trace benchmark execution id is missing")
        if payload.get("schema") == TRACE_BENCHMARK_RESULT_SCHEMA:
            python_environment_sha256 = run_spec.get(
                "python_environment_sha256"
            )
            if (
                not isinstance(python_environment_sha256, str)
                or not python_environment_sha256.startswith("sha256:")
            ):
                raise ValueError(
                    "trace benchmark Python environment attestation is missing"
                )
    return payload


def _blocker_categories(blockers: object) -> list[str]:
    if not isinstance(blockers, list) or not all(
        isinstance(blocker, str) for blocker in blockers
    ):
        raise ValueError("trace benchmark blockers must be a string list")
    categories: set[str] = set()
    for blocker in blockers:
        normalized = blocker.casefold()
        if "contextwindowexceeded" in normalized or "maximum context length" in normalized:
            categories.add("context_window_exceeded")
        elif "out of range" in normalized:
            categories.add("out_of_range")
        elif "did not contain agent and step" in normalized:
            categories.add("missing_required_fields")
        elif "no decisive errors found" in normalized:
            categories.add("no_decisive_error")
        elif "upstream process failed" in normalized:
            categories.add("upstream_process_failed")
        else:
            categories.add("unclassified")
    return sorted(categories)


def _sanitize_finding(finding: Mapping[str, Any]) -> dict[str, object]:
    attributes = _mapping(
        finding.get("attributes"),
        label="trace benchmark finding attributes",
    )
    unknown = set(attributes) - SAFE_FINDING_ATTRIBUTE_KEYS
    unsafe_unknown = {
        key for key in unknown if not str(key).startswith("justification_")
    }
    if unsafe_unknown:
        raise ValueError(
            f"unreviewed trace benchmark finding attributes: {sorted(unsafe_unknown)}"
        )
    return {
        "kind": finding.get("kind"),
        "event_ids": finding.get("event_ids"),
        "label": finding.get("label"),
        "score": finding.get("score"),
        "attributes": {
            key: value
            for key, value in attributes.items()
            if key in SAFE_FINDING_ATTRIBUTE_KEYS
        },
    }


def _sanitize_cell(cell: object, *, model_id: str) -> dict[str, object]:
    payload = _mapping(cell, label="trace benchmark cell")
    findings = payload.get("findings")
    if not isinstance(findings, list) or not all(
        isinstance(finding, Mapping) for finding in findings
    ):
        raise ValueError("trace benchmark result lacks structured findings")
    return {
        "model_id": model_id,
        "case_id": payload.get("case_id"),
        "method_id": payload.get("method_id"),
        "source_path": payload.get("source_path"),
        "source_sha256": payload.get("source_sha256"),
        "request_sha256": payload.get("request_sha256"),
        "receipt_sha256": payload.get("receipt_sha256"),
        "status": payload.get("status"),
        "findings": [_sanitize_finding(finding) for finding in findings],
        "score": payload.get("score"),
        "blocker_categories": _blocker_categories(payload.get("blockers")),
    }


def build_trace_benchmark_case_inventory(
    result_paths: Sequence[str | Path],
    *,
    expected_dataset_manifest_sha256: str,
    expected_model_ids: tuple[str, ...],
    expected_method_ids: tuple[str, ...],
    expected_cases: tuple[FrozenTraceBenchmarkCaseIdentityV1, ...],
) -> dict[str, object]:
    """Verify multiple model results and strip free-text blockers from committed evidence."""

    if len(result_paths) < 2:
        raise ValueError("trace benchmark inventory requires at least two results")
    results = tuple(_load_verified_result(path) for path in result_paths)
    for result in results:
        validate_frozen_trace_benchmark_result(
            result,
            expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
            expected_model_ids=expected_model_ids,
            expected_method_ids=expected_method_ids,
            expected_cases=expected_cases,
        )
    identity_keys = (
        "dataset_id",
        "dataset_manifest_sha256",
        "dataset_source_revision",
        "platform_revision",
        "method_ids",
        "method_descriptors",
        "high_process_score_on_failure_threshold",
    )
    reference = results[0]
    for result in results[1:]:
        if any(result.get(key) != reference.get(key) for key in identity_keys):
            raise ValueError("trace benchmark result identities do not match")
    model_ids = tuple(str(result.get("model_id", "")) for result in results)
    if any(not model_id for model_id in model_ids) or len(model_ids) != len(set(model_ids)):
        raise ValueError("trace benchmark inventory model ids must be unique")
    runs = [
        {
            "model_id": model_id,
            "execution_id": result.get("execution_id"),
            "result_sha256": result.get("result_sha256"),
            "runtime_attestation": result.get("runtime_attestation"),
            "run_spec": result.get("run_spec"),
        }
        for model_id, result in zip(model_ids, results, strict=True)
    ]
    cells = [
        _sanitize_cell(cell, model_id=model_id)
        for model_id, result in zip(model_ids, results, strict=True)
        for cell in result.get("cells", [])
    ]
    unhashed: dict[str, object] = {
        "schema": TRACE_BENCHMARK_CASE_INVENTORY_SCHEMA,
        "dataset_id": reference.get("dataset_id"),
        "dataset_manifest_sha256": reference.get("dataset_manifest_sha256"),
        "dataset_source_revision": reference.get("dataset_source_revision"),
        "platform_revision": reference.get("platform_revision"),
        "method_ids": reference.get("method_ids"),
        "method_descriptors": reference.get("method_descriptors"),
        "high_process_score_on_failure_threshold": reference.get(
            "high_process_score_on_failure_threshold"
        ),
        "runs": runs,
        "cells": cells,
    }
    return {
        **unhashed,
        "inventory_sha256": "sha256:" + canonical_sha256(unhashed),
    }


def write_trace_benchmark_case_inventory(
    inventory: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    atomic_write_json(output_path, inventory, overwrite=False)


__all__ = [
    "SAFE_FINDING_ATTRIBUTE_KEYS",
    "TRACE_BENCHMARK_CASE_INVENTORY_SCHEMA",
    "FrozenTraceBenchmarkCaseIdentityV1",
    "build_trace_benchmark_case_inventory",
    "frozen_trace_benchmark_case_identities",
    "validate_frozen_trace_benchmark_result",
    "write_trace_benchmark_case_inventory",
]
