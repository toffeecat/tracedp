"""Verified ALFWorld layout and runtime-evidence assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lh_platform.studies.protocol import CompiledStudy, StudyProtocol
from lh_platform.studies.runtime import (
    AlfworldTaskManifest,
    GraphGPOBenchmarkAuditor,
    RuntimeInputs,
    RuntimePreflight,
    RuntimePreflightReport,
    materialize_alfworld_data_view,
    run_graphgpo_reset_probe,
    sha256_file,
)

ALFWORLD_OFFICIAL_LAYOUT_QUOTES = (
    {
        "component": "train_split",
        "source": "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
        "quote": "data_path: '$ALFWORLD_DATA/json_2.1.1/train'",
        "required_path": "json_2.1.1/train",
    },
    {
        "component": "valid_seen_split",
        "source": "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
        "quote": "eval_id_data_path: '$ALFWORLD_DATA/json_2.1.1/valid_seen'",
        "required_path": "json_2.1.1/valid_seen",
    },
    {
        "component": "valid_unseen_split",
        "source": "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
        "quote": "eval_ood_data_path: '$ALFWORLD_DATA/json_2.1.1/valid_unseen'",
        "required_path": "json_2.1.1/valid_unseen",
    },
    {
        "component": "pddl_domain",
        "source": "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
        "quote": "domain: '$ALFWORLD_DATA/logic/alfred.pddl'",
        "required_path": "logic/alfred.pddl",
    },
    {
        "component": "textworld_grammar",
        "source": "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
        "quote": "grammar: '$ALFWORLD_DATA/logic/alfred.twl2'",
        "required_path": "logic/alfred.twl2",
    },
)


@dataclass(frozen=True)
class AlfWorldRuntimeEvidence:
    runtime_data: Path
    benchmark_audit: Path
    state_reset_probe: Path | None
    reset_probe_returncode: int | None
    runtime_report: RuntimePreflightReport
    view_error: str | None


def alfworld_official_layout_missing(data_root: Path) -> list[str]:
    """Return missing official paths in source-contract order."""
    return [
        relative.as_posix()
        for item in ALFWORLD_OFFICIAL_LAYOUT_QUOTES
        if not (data_root / (relative := Path(str(item["required_path"])))).exists()
    ]


def build_alfworld_runtime_evidence(
    protocol: StudyProtocol,
    compiled: CompiledStudy,
    *,
    manifest_path: Path,
    source_data_root: Path,
    source_root: Path,
    source_commit: str,
    output_root: Path,
    python_executable: str,
    source_contract: Path | None,
    state_reset_probe: Path | None = None,
    run_reset_probe: bool = False,
    allow_view_failure: bool = False,
    resource_profile: str = "",
    required_gpus_override: int | None = None,
) -> AlfWorldRuntimeEvidence:
    """Assemble source-bound runtime evidence and write its preflight report."""
    runtime_data = source_data_root
    view_error: str | None = None
    try:
        manifest = AlfworldTaskManifest.read(manifest_path)
        runtime_data = materialize_alfworld_data_view(
            manifest,
            source_data_root,
            output_root / "alfworld_data_view",
        ).root
    except (OSError, TypeError, ValueError) as error:
        if not allow_view_failure:
            raise
        view_error = str(error)

    benchmark_audit = output_root / "benchmark_audit.json"
    GraphGPOBenchmarkAuditor(source_root).audit(
        task_manifest_sha256=sha256_file(manifest_path),
        source_commit=source_commit,
    ).write(benchmark_audit)

    probe_path = state_reset_probe
    reset_returncode: int | None = None
    if run_reset_probe:
        probe_path = output_root / "state_reset_probe.json"
        completed = run_graphgpo_reset_probe(
            manifest_path=manifest_path,
            source_data_root=source_data_root,
            source_root=source_root,
            source_commit=source_commit,
            output_path=probe_path,
            python_executable=python_executable,
        )
        reset_returncode = completed.returncode

    runtime_report = RuntimePreflight().check(
        protocol,
        compiled,
        RuntimeInputs(
            task_manifest=manifest_path,
            benchmark_audit=benchmark_audit,
            state_reset_probe=probe_path,
            alfworld_data=runtime_data,
            python_executable=python_executable,
            source_contract=source_contract,
            resource_profile=resource_profile,
            required_gpus_override=required_gpus_override,
        ),
    )
    runtime_report.write(output_root / "runtime_validity_report.json")
    return AlfWorldRuntimeEvidence(
        runtime_data=runtime_data,
        benchmark_audit=benchmark_audit,
        state_reset_probe=probe_path,
        reset_probe_returncode=reset_returncode,
        runtime_report=runtime_report,
        view_error=view_error,
    )


__all__ = [
    "ALFWORLD_OFFICIAL_LAYOUT_QUOTES",
    "AlfWorldRuntimeEvidence",
    "alfworld_official_layout_missing",
    "build_alfworld_runtime_evidence",
]
