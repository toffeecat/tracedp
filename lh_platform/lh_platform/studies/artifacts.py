"""Materialize immutable inputs to a compiled study run."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from lh_platform.studies.protocol import CompiledStudy, StudyProtocol


def write_study_artifacts(
    protocol: StudyProtocol,
    compiled: CompiledStudy,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Write the evidence, comparison, trial, and validity manifests."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    evidence_path = root / "evidence_manifest.json"
    comparison_path = root / "comparison_manifest.json"
    trials_path = root / "trial_manifest.jsonl"
    validity_path = root / "validity_report.json"

    _write_json(
        evidence_path,
        {
            "schema": "lh_study_evidence/v1",
            "study_id": protocol.id,
            "study_source": str(protocol.source_path) if protocol.source_path else None,
            "declared_evidence": [asdict(item) for item in protocol.evidence],
            "resolved_evidence": [asdict(item) for item in compiled.resolved_evidence],
        },
    )
    _write_json(
        comparison_path,
        {
            "schema": "lh_study_comparison/v1",
            "study_id": compiled.study_id,
            **asdict(compiled.comparison),
        },
    )
    trials_path.write_text(
        "".join(json.dumps(asdict(trial), sort_keys=True) + "\n" for trial in compiled.trials),
        encoding="utf-8",
    )
    _write_json(
        validity_path,
        {
            "schema": "lh_study_validity/v1",
            "study_id": protocol.id,
            "ok": True,
            "validation_scope": "protocol_only",
            "run_ready": False,
            "run_readiness_reason": (
                "Phase 0 validates declared study intent and fairness; runtime artifacts "
                "and benchmark availability are not yet verified"
            ),
            "checks": [
                "purpose",
                "treatment_differences",
                "budget_parity",
                "primary_source_evidence",
                "immutable_source_resolution",
                "requested_execution_paths",
                "replication",
                "required_outcome_evidence",
                "required_benchmark_audit_components",
                "requested_data_and_state_integrity",
                "study_kind_requirements",
            ],
            "pending_runtime_checks": [
                "actual_execution_path_and_dispatch",
                "benchmark_owned_outcome_evidence",
                "task_group_split_leakage",
                "state_reset_between_trials",
                "benchmark_component_audit_results",
            ],
        },
    )
    return evidence_path, comparison_path, trials_path, validity_path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
