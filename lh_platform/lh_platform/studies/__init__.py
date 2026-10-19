"""Study APIs with lazy imports so independent contracts stay testable."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from lh_platform.studies.graphgpo import ROOT_PUBLIC_EXPORTS as GRAPHGPO_ROOT_EXPORTS

_EXPORTS = {
    "CompiledStudy": "lh_platform.studies.protocol",
    "ComparisonPlan": "lh_platform.studies.protocol",
    "EvidenceRegistry": "lh_platform.studies.evidence",
    "ResolvedEvidence": "lh_platform.studies.evidence",
    "require_verified_source_root": "lh_platform.studies.evidence",
    "ALFWORLD_OFFICIAL_LAYOUT_QUOTES": "lh_platform.studies.alfworld_evidence",
    "AlfWorldRuntimeEvidence": "lh_platform.studies.alfworld_evidence",
    "alfworld_official_layout_missing": "lh_platform.studies.alfworld_evidence",
    "build_alfworld_runtime_evidence": "lh_platform.studies.alfworld_evidence",
    "StudyProtocol": "lh_platform.studies.protocol",
    "StudyConfigError": "lh_platform.studies.load",
    "StudyValidationError": "lh_platform.studies.protocol",
    "TrialPlan": "lh_platform.studies.protocol",
    "AlfworldTaskManifest": "lh_platform.studies.runtime",
    "GraphGPOBenchmarkAuditor": "lh_platform.studies.runtime",
    "GraphGPOStateRestoreProbe": "lh_platform.studies.state_restore_probe",
    "GraphGPOStateRestoreProbeReport": "lh_platform.studies.state_restore_probe",
    "GraphGPOStateSnapshotProvider": "lh_platform.studies.state_snapshot",
    "GraphGPOTraceArtifacts": "lh_platform.studies.trace",
    "GraphGPOTraceSidecar": "lh_platform.studies.trace_sidecar",
    "MechanismReport": "lh_platform.studies.mechanism",
    "MechanismProbePlan": "lh_platform.studies.mechanism",
    "ReplicationResult": "lh_platform.studies.results",
    "RuntimeInputs": "lh_platform.studies.runtime",
    "RuntimePreflight": "lh_platform.studies.runtime",
    "RuntimePreflightReport": "lh_platform.studies.runtime",
    "SourceContractReport": "lh_platform.studies.source_contract",
    "StudyExecutor": "lh_platform.studies.execution",
    "TrialExecutionResult": "lh_platform.studies.execution",
    "RUN_AUDIT_SCHEMA": "lh_platform.studies.run_audit",
    "OutcomeSidecarBundle": "lh_platform.studies.outcome_sidecar",
    "build_alfworld_task_manifest": "lh_platform.studies.runtime",
    "build_graphgpo_mechanism_report": "lh_platform.studies.mechanism",
    "build_graphgpo_mechanism_probe_plan": "lh_platform.studies.mechanism",
    "build_graphgpo_state_restore_probe": "lh_platform.studies.state_restore_probe",
    "build_graphgpo_state_restore_probe_report": "lh_platform.studies.state_restore_probe",
    "build_graphgpo_state_snapshot_provider": "lh_platform.studies.state_snapshot",
    "build_graphgpo_trace_artifacts": "lh_platform.studies.trace",
    "build_graphgpo_trace_sidecar": "lh_platform.studies.trace_sidecar",
    "build_replication_result": "lh_platform.studies.results",
    "build_study_run_audit": "lh_platform.studies.run_audit",
    "build_outcome_sidecar_patch": "lh_platform.studies.outcome_sidecar",
    "load_study": "lh_platform.studies.load",
    "materialize_alfworld_data_view": "lh_platform.studies.runtime",
    "run_graphgpo_reset_probe": "lh_platform.studies.runtime",
    "verify_source_contract": "lh_platform.studies.source_contract",
    "emit_native_outcome": "lh_platform.studies.outcome_sidecar",
    "set_evaluation_step": "lh_platform.studies.outcome_sidecar",
    "write_study_artifacts": "lh_platform.studies.artifacts",
    "AlfWorldStateError": "lh_platform.alfworld_state",
    "AlfWorldStateSnapshot": "lh_platform.alfworld_state",
    "AlfWorldRestoreReceipt": "lh_platform.alfworld_state",
    "AlfWorldStateArtifact": "lh_platform.alfworld_state",
    "AlfWorldPhysicalCallCounters": "lh_platform.alfworld_state",
    "AlfWorldVerifiedStep": "lh_platform.alfworld_state",
    "AlfWorldReplayStateController": "lh_platform.alfworld_state",
    "project_scalar_textworld_outcome": "lh_platform.alfworld_state",
    "build_verified_v2_transition_annotations": "lh_platform.alfworld_state",
    "build_verified_subproblem_annotation": "lh_platform.alfworld_state",
    "snapshot_alfworld_worker": "lh_platform.alfworld_state",
    "restore_alfworld_worker": "lh_platform.alfworld_state",
    "probe_alfworld_restore": "lh_platform.alfworld_state",
    "write_alfworld_state_artifact": "lh_platform.alfworld_state",
    "read_alfworld_state_artifact": "lh_platform.alfworld_state",
    "write_alfworld_restore_receipt": "lh_platform.alfworld_state",
    "write_alfworld_state_artifact_manifest": "lh_platform.alfworld_state",
    "read_alfworld_state_artifact_manifest": "lh_platform.alfworld_state",
    "step_with_verified_state": "lh_platform.alfworld_state",
    "state_capability_manifest": "lh_platform.alfworld_state",
    "ALFWORLD_SUBPROBLEM_CONTRACT_SCHEMA": "lh_platform.studies.alfworld_subproblems",
    "AlfWorldSubproblemAnnotation": "lh_platform.studies.alfworld_subproblems",
    "AlfWorldSubproblemError": "lh_platform.studies.alfworld_subproblems",
    "derive_alfworld_active_subproblem": "lh_platform.studies.alfworld_subproblems",
}

_EXPORTS.update({name: "lh_platform.studies.graphgpo" for name in GRAPHGPO_ROOT_EXPORTS})


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
