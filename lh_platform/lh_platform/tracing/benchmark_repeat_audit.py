"""Prompt-free reproducibility audit for unified tracer benchmark repeats."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lh_platform.core.atomic_io import atomic_write_json
from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.benchmark_inventory import (
    FrozenTraceBenchmarkCaseIdentityV1,
    _load_verified_result,
    _mapping,
    validate_frozen_trace_benchmark_result,
)
from lh_platform.tracing.unified_benchmark import TRACE_BENCHMARK_RESULT_SCHEMA

TRACE_BENCHMARK_REPEAT_AUDIT_SCHEMA = "lh_trace_benchmark_repeat_audit/v1"


def _cell_index(result: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    raw_cells = result.get("cells")
    if not isinstance(raw_cells, list):
        raise ValueError("trace benchmark repeat cells must be a list")
    cells: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_cell in raw_cells:
        cell = _mapping(raw_cell, label="trace benchmark repeat cell")
        key = (str(cell.get("case_id", "")), str(cell.get("method_id", "")))
        if not all(key) or key in cells:
            raise ValueError("trace benchmark repeat cell identity is invalid")
        cells[key] = cell
    return cells


def _semantic_findings(cell: Mapping[str, Any]) -> object:
    findings = cell.get("findings")
    if not isinstance(findings, list):
        raise ValueError("trace benchmark repeat findings must be a list")
    sanitized: list[dict[str, object]] = []
    for raw_finding in findings:
        finding = _mapping(raw_finding, label="trace benchmark repeat finding")
        attributes = _mapping(
            finding.get("attributes"), label="trace benchmark repeat attributes"
        )
        sanitized.append(
            {
                "kind": finding.get("kind"),
                "event_ids": finding.get("event_ids"),
                "label": finding.get("label"),
                "score": finding.get("score"),
                "attributes": {
                    key: value
                    for key, value in attributes.items()
                    if key != "raw_output_sha256"
                    and not str(key).startswith("justification_")
                },
            }
        )
    return sanitized


def _raw_output_hashes(cell: Mapping[str, Any]) -> tuple[str, ...]:
    findings = cell.get("findings")
    if not isinstance(findings, list):
        return ()
    hashes = {
        str(attributes["raw_output_sha256"])
        for finding in findings
        if isinstance(finding, Mapping)
        and isinstance((attributes := finding.get("attributes")), Mapping)
        and "raw_output_sha256" in attributes
    }
    return tuple(sorted(hashes))


def _verify_response_models(result: Mapping[str, Any]) -> int:
    model_id = str(result.get("model_id", ""))
    completed = 0
    for cell in _cell_index(result).values():
        if cell.get("status") != "completed":
            continue
        completed += 1
        findings = cell.get("findings")
        response_ids = {
            attributes.get("response_model_id")
            for finding in findings
            if isinstance(finding, Mapping)
            and isinstance((attributes := finding.get("attributes")), Mapping)
        }
        if response_ids != {model_id}:
            raise ValueError(
                "completed trace benchmark cell lacks exact response model provenance"
            )
    return completed


def build_trace_benchmark_repeat_audit(
    result_paths: Sequence[str | Path],
    *,
    expected_model_ids: tuple[str, ...],
    expected_method_ids: tuple[str, ...],
    expected_execution_ids: Mapping[str, tuple[str, ...]],
    expected_dataset_manifest_sha256: str,
    expected_cases: tuple[FrozenTraceBenchmarkCaseIdentityV1, ...],
    expected_platform_revision: str,
    expected_method_descriptors_sha256: str,
    expected_max_tokens: int,
    expected_timeout_seconds: int,
    expected_temperature: float,
    expected_generation_seed: int,
    expected_python_environment_sha256: str,
    expected_repeats_per_model: int = 2,
) -> dict[str, object]:
    """Require the externally frozen complete matrix and independent repeats."""

    results = tuple(_load_verified_result(path) for path in result_paths)
    if (
        len(expected_model_ids) < 2
        or len(expected_model_ids) != len(set(expected_model_ids))
        or not expected_method_ids
        or len(expected_method_ids) != len(set(expected_method_ids))
        or not expected_cases
        or expected_repeats_per_model != 2
    ):
        raise ValueError("repeat audit expected matrix is invalid")
    expected_result_count = len(expected_model_ids) * expected_repeats_per_model
    if len(results) != expected_result_count or any(
        result.get("schema") != TRACE_BENCHMARK_RESULT_SCHEMA for result in results
    ):
        raise ValueError("repeat audit result count or schema is invalid")
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
    if any(
        any(result.get(key) != reference.get(key) for key in identity_keys)
        for result in results[1:]
    ):
        raise ValueError("trace benchmark repeat identities do not match")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        validate_frozen_trace_benchmark_result(
            result,
            expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
            expected_model_ids=expected_model_ids,
            expected_method_ids=expected_method_ids,
            expected_cases=expected_cases,
        )
        if result.get("platform_revision") != expected_platform_revision:
            raise ValueError("repeat audit platform revision does not match frozen protocol")
        run_spec = _mapping(result.get("run_spec"), label="repeat audit run spec")
        expected_run_projection = {
            "dataset_manifest_sha256": expected_dataset_manifest_sha256,
            "platform_revision": expected_platform_revision,
            "method_descriptors_sha256": expected_method_descriptors_sha256,
            "max_tokens": expected_max_tokens,
            "timeout_seconds": expected_timeout_seconds,
            "temperature": expected_temperature,
            "generation_seed": expected_generation_seed,
            "python_environment_sha256": expected_python_environment_sha256,
        }
        if any(run_spec.get(key) != value for key, value in expected_run_projection.items()):
            raise ValueError("repeat audit run spec does not match frozen protocol")
        raw_descriptors = result.get("method_descriptors")
        if not isinstance(raw_descriptors, list):
            raise ValueError("repeat audit method descriptors are invalid")
        for raw_descriptor in raw_descriptors:
            descriptor = _mapping(raw_descriptor, label="repeat audit method descriptor")
            if (
                descriptor.get("integration_level")
                not in {"upstream_native", "upstream_native_port"}
                or descriptor.get("unavailable_reason") is not None
                or not descriptor.get("source_revision")
            ):
                raise ValueError("repeat audit contains a non-native frozen method")
        if any(cell.get("status") == "blocked" for cell in _cell_index(result).values()):
            raise ValueError("repeat audit contains blocked frozen method cells")
        grouped[str(result.get("model_id", ""))].append(result)
    if set(grouped) != set(expected_model_ids) or any(
        len(repeats) != expected_repeats_per_model for repeats in grouped.values()
    ):
        raise ValueError("repeat audit models or repeat counts do not match frozen matrix")
    execution_ids = [str(result.get("execution_id", "")) for result in results]
    if any(not value for value in execution_ids) or len(execution_ids) != len(
        set(execution_ids)
    ):
        raise ValueError("repeat audit requires unique independent execution ids")
    if set(expected_execution_ids) != set(expected_model_ids) or any(
        len(ids) != expected_repeats_per_model or len(ids) != len(set(ids))
        for ids in expected_execution_ids.values()
    ):
        raise ValueError("repeat audit expected execution ids are invalid")
    if set(execution_ids) != {
        execution_id
        for model_ids in expected_execution_ids.values()
        for execution_id in model_ids
    }:
        raise ValueError("repeat audit execution ids do not match authorization")

    model_audits: list[dict[str, object]] = []
    for model_id in sorted(grouped):
        repeats = sorted(
            grouped[model_id], key=lambda result: str(result.get("execution_id", ""))
        )
        observed_model_execution_ids = tuple(
            str(result.get("execution_id", "")) for result in repeats
        )
        if set(observed_model_execution_ids) != set(expected_execution_ids[model_id]):
            raise ValueError("repeat audit model/execution authorization mismatch")
        first, second = repeats
        if first.get("run_spec") != second.get("run_spec"):
            raise ValueError("same-model repeats have different run specs")
        completed_counts = (
            _verify_response_models(first),
            _verify_response_models(second),
        )
        first_cells = _cell_index(first)
        second_cells = _cell_index(second)
        if first_cells.keys() != second_cells.keys():
            raise ValueError("same-model repeats have different cells")
        comparisons: list[dict[str, object]] = []
        for case_id, method_id in sorted(first_cells):
            a = first_cells[(case_id, method_id)]
            b = second_cells[(case_id, method_id)]
            raw_a = _raw_output_hashes(a)
            raw_b = _raw_output_hashes(b)
            comparisons.append(
                {
                    "case_id": case_id,
                    "method_id": method_id,
                    "status_a": a.get("status"),
                    "status_b": b.get("status"),
                    "status_equal": a.get("status") == b.get("status"),
                    "score_equal": a.get("score") == b.get("score"),
                    "semantic_findings_equal": _semantic_findings(a)
                    == _semantic_findings(b),
                    "raw_output_hash_evaluable": bool(raw_a and raw_b),
                    "raw_output_hash_equal": raw_a == raw_b if raw_a and raw_b else None,
                }
            )
        model_audits.append(
            {
                "model_id": model_id,
                "execution_ids": list(observed_model_execution_ids),
                "result_sha256": [
                    first.get("result_sha256"),
                    second.get("result_sha256"),
                ],
                "run_spec": first.get("run_spec"),
                "completed_cells_with_response_model_provenance": list(
                    completed_counts
                ),
                "cell_count": len(comparisons),
                "status_agreement_count": sum(
                    bool(item["status_equal"]) for item in comparisons
                ),
                "score_agreement_count": sum(
                    bool(item["score_equal"]) for item in comparisons
                ),
                "semantic_finding_agreement_count": sum(
                    bool(item["semantic_findings_equal"]) for item in comparisons
                ),
                "raw_output_hash_evaluable_count": sum(
                    bool(item["raw_output_hash_evaluable"]) for item in comparisons
                ),
                "raw_output_hash_agreement_count": sum(
                    item["raw_output_hash_equal"] is True for item in comparisons
                ),
                "cells": comparisons,
            }
        )
    unhashed: dict[str, object] = {
        "schema": TRACE_BENCHMARK_REPEAT_AUDIT_SCHEMA,
        "dataset_id": reference.get("dataset_id"),
        "dataset_manifest_sha256": reference.get("dataset_manifest_sha256"),
        "dataset_source_revision": reference.get("dataset_source_revision"),
        "platform_revision": reference.get("platform_revision"),
        "method_ids": reference.get("method_ids"),
        "models": model_audits,
    }
    return {
        **unhashed,
        "repeat_audit_sha256": "sha256:" + canonical_sha256(unhashed),
    }


def write_trace_benchmark_repeat_audit(
    audit: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    atomic_write_json(output_path, audit, overwrite=False)


__all__ = [
    "TRACE_BENCHMARK_REPEAT_AUDIT_SCHEMA",
    "build_trace_benchmark_repeat_audit",
    "write_trace_benchmark_repeat_audit",
]
