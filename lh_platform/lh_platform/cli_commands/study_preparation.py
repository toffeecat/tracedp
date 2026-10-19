"""Auditable GraphGPO study-preparation command."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console

from lh_platform.studies.evidence import require_verified_source_root

__all__ = ["register_study_preparation_command"]


def register_study_preparation_command(
    app: typer.Typer,
    console: Console,
) -> None:
    """Register the study-preparation command on the public flat CLI."""

    @app.command(name="prepare-study")
    def prepare_study(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 file"),
        output_dir: Path | None = typer.Option(None, "--output-dir"),
        task_manifest: Path | None = typer.Option(None, "--task-manifest"),
        alfworld_data: Path | None = typer.Option(None, "--alfworld-data"),
        state_reset_probe: Path | None = typer.Option(
            None,
            "--state-reset-probe",
        ),
        mechanism_report: Path | None = typer.Option(
            None,
            "--mechanism-report",
            help="Phase 2 mechanism_report.json used to gate Phase 3 research expansion",
        ),
        run_reset_probe: bool = typer.Option(
            False,
            "--run-reset-probe",
            help="run the live official ALFWorld reset probe before preflight",
        ),
        python_executable: str = typer.Option(
            sys.executable,
            "--python",
            help="upstream GraphGPO Python executable",
        ),
        require_ready: bool = typer.Option(
            False,
            "--require-ready",
            help="exit non-zero when runtime checks block launch",
        ),
        resource_profile: str | None = typer.Option(
            None,
            "--resource-profile",
            help=(
                "resource-constrained override profile, "
                "e.g. single_5090_small, single_5090_group2, single_5090_group8, "
                "or single_5090_full_rollout_group8"
            ),
        ),
        resource_training_steps: int | None = typer.Option(
            None,
            "--resource-training-steps",
            min=1,
            help="training step budget baked into a resource-constrained prepared command",
        ),
        resource_model_path: str | None = typer.Option(
            None,
            "--resource-model-path",
            help=(
                "model checkpoint for an explicitly resource-constrained capacity probe; "
                "does not permit a formal reproduction claim"
            ),
        ),
    ) -> None:
        """Prepare auditable official GraphGPO commands without launching training."""
        from lh_platform.studies.graphgpo.runner import (
            ALLOWED_RESOURCE_PROFILES,
            PAPER_REPRODUCTION_CLAIM_SCOPE,
            RESOURCE_CONSTRAINED_CLAIM_SCOPE,
            resource_profile_required_gpus,
        )

        selected_resource_profile = resource_profile or ""
        if selected_resource_profile and selected_resource_profile not in ALLOWED_RESOURCE_PROFILES:
            raise typer.BadParameter(f"unknown resource profile: {selected_resource_profile}")
        if resource_training_steps is not None and not selected_resource_profile:
            raise typer.BadParameter("--resource-training-steps requires --resource-profile")
        if resource_model_path and not selected_resource_profile:
            raise typer.BadParameter("--resource-model-path requires --resource-profile")

        from lh_platform.studies import (
            GraphGPOStudyRunner,
            RuntimeInputs,
            RuntimePreflight,
            build_alfworld_runtime_evidence,
            build_graphgpo_mechanism_probe_plan,
            load_study,
            write_study_artifacts,
        )
        from lh_platform.studies.evidence import find_workspace_root

        protocol = load_study(study_path)
        compiled = protocol.compile()
        workspace_root = find_workspace_root(protocol.source_path)
        target = output_dir or Path("results") / "studies" / protocol.id / "prepare"
        target.mkdir(parents=True, exist_ok=True)
        write_study_artifacts(protocol, compiled, target)

        declared_manifest = task_manifest or Path(protocol.design.task_manifest)
        manifest_path = (
            declared_manifest.resolve()
            if declared_manifest.is_absolute()
            else (workspace_root / declared_manifest).resolve()
        )
        env_data = os.environ.get("ALFWORLD_DATA")
        source_data = alfworld_data or (Path(env_data) if env_data else None)
        if source_data is not None:
            source_data = source_data.expanduser().resolve()
        source_root = require_verified_source_root(compiled)
        contract_value = protocol.design.fixed_factors.get("source_contract")
        contract_path = (workspace_root / str(contract_value)).resolve() if contract_value else None
        resource_required_gpus = (
            resource_profile_required_gpus(selected_resource_profile)
            if selected_resource_profile
            else None
        )

        runtime_data = source_data
        audit_path: Path | None = None
        runtime_report = None
        if manifest_path.is_file() and source_data and source_data.is_dir():
            evidence = build_alfworld_runtime_evidence(
                protocol,
                compiled,
                manifest_path=manifest_path,
                source_data_root=source_data,
                source_root=source_root,
                source_commit=str(protocol.design.fixed_factors["source_commit"]),
                output_root=target,
                python_executable=python_executable,
                source_contract=contract_path,
                state_reset_probe=state_reset_probe,
                run_reset_probe=run_reset_probe,
                allow_view_failure=True,
                resource_profile=selected_resource_profile,
                required_gpus_override=resource_required_gpus,
            )
            runtime_data = evidence.runtime_data
            audit_path = evidence.benchmark_audit
            probe_path = evidence.state_reset_probe
            runtime_report = evidence.runtime_report
            if evidence.view_error:
                console.print(
                    f"[yellow]manifest view not materialized: {evidence.view_error}[/yellow]"
                )
            if evidence.reset_probe_returncode not in (None, 0):
                console.print(f"[yellow]live state-reset probe failed: {probe_path}[/yellow]")
        else:
            probe_path = state_reset_probe

        inputs = RuntimeInputs(
            task_manifest=manifest_path,
            benchmark_audit=audit_path,
            state_reset_probe=probe_path,
            alfworld_data=runtime_data,
            python_executable=python_executable,
            source_contract=contract_path,
            resource_profile=selected_resource_profile,
            required_gpus_override=resource_required_gpus,
        )
        if runtime_report is None:
            runtime_report = RuntimePreflight().check(protocol, compiled, inputs)
            runtime_report.write(target / "runtime_validity_report.json")
        selected_claim_scope = (
            RESOURCE_CONSTRAINED_CLAIM_SCOPE
            if selected_resource_profile
            else PAPER_REPRODUCTION_CLAIM_SCOPE
        )
        prepared = GraphGPOStudyRunner().prepare(
            protocol,
            compiled,
            runtime_report,
            source_root=source_root,
            alfworld_data=runtime_data or target / "missing_alfworld_data",
            output_root=target,
            python_executable=python_executable,
            claim_scope=selected_claim_scope,
            resource_profile=selected_resource_profile,
            resource_training_steps=resource_training_steps,
            resource_model_path=resource_model_path or "",
        )
        prepared.write(target)
        build_graphgpo_mechanism_probe_plan(
            protocol,
            compiled,
            prepared,
            runtime_report=runtime_report,
            source_root=source_root,
        ).write(target)
        color = "green" if runtime_report.run_ready else "yellow"
        console.print(
            f"[bold {color}]study prepared: {protocol.id}[/bold {color}]\n"
            f"  trials={len(prepared.trials)}\n"
            f"  run_ready={str(runtime_report.run_ready).lower()}\n"
            f"  blocked_by={list(runtime_report.blocked_by)}\n"
            f"  artifacts={target}"
        )
        if require_ready and not runtime_report.run_ready:
            raise typer.Exit(3)
