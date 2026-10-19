"""GraphGPO mechanism, trace, snapshot, and restore evidence commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from lh_platform.studies.evidence import require_verified_source_root

__all__ = ["register_study_evidence_commands"]


def register_study_evidence_commands(
    app: typer.Typer,
    console: Console,
) -> None:
    """Register GraphGPO evidence-building commands on the public flat CLI."""

    @app.command(name="build-mechanism-report")
    def build_mechanism_report(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 file"),
        edge_credit_trace: Path = typer.Option(..., "--edge-credit-trace"),
        counterfactual_continuations: Path = typer.Option(
            ...,
            "--counterfactual-continuations",
        ),
        output: Path = typer.Option(
            Path("mechanism_report.json"),
            "--output",
            help="Phase 4 mechanism_report.json output path",
        ),
        require_stable: bool = typer.Option(
            False,
            "--require-stable",
            help="exit non-zero when mechanism metrics do not all pass",
        ),
    ) -> None:
        """Build a falsifiable mechanism report from edge and counterfactual traces."""
        from lh_platform.studies import build_graphgpo_mechanism_report, load_study

        protocol = load_study(study_path)
        report = build_graphgpo_mechanism_report(
            protocol,
            edge_credit_trace=edge_credit_trace,
            counterfactual_continuations=counterfactual_continuations,
        )
        path = report.write(output)
        color = "green" if report.credit_assignment_stable else "yellow"
        console.print(
            f"[bold {color}]mechanism report built: {path}[/bold {color}]\n"
            f"  credit_assignment_stable="
            f"{str(report.credit_assignment_stable).lower()}\n"
            f"  blocked_by={list(report.blocked_by)}"
        )
        if require_stable and not report.credit_assignment_stable:
            raise typer.Exit(3)

    @app.command(name="build-trace-artifacts")
    def build_trace_artifacts(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 file"),
        task_manifest: Path = typer.Option(..., "--task-manifest"),
        raw_trace: list[Path] = typer.Option(
            ...,
            "--raw-trace",
            help="raw GraphGPO trace JSONL; may be passed repeatedly",
        ),
        output_dir: Path = typer.Option(
            Path("trace_artifacts"),
            "--output-dir",
            help=("directory for edge_credit_trace.jsonl and counterfactual_continuations.jsonl"),
        ),
        require_ready: bool = typer.Option(
            False,
            "--require-ready",
            help="exit non-zero when trace ingestion is blocked",
        ),
    ) -> None:
        """Build canonical Phase 4 trace inputs from raw upstream GraphGPO events."""
        from lh_platform.studies import build_graphgpo_trace_artifacts, load_study

        protocol = load_study(study_path)
        compiled = protocol.compile()
        bundle = build_graphgpo_trace_artifacts(
            protocol,
            task_manifest=task_manifest,
            raw_trace=tuple(raw_trace),
            output_dir=output_dir,
            source_root=require_verified_source_root(compiled),
        )
        color = "green" if bundle.ingestion_ready else "yellow"
        console.print(
            f"[bold {color}]trace artifacts built: "
            f"{bundle.manifest_path}[/bold {color}]\n"
            f"  ingestion_ready={str(bundle.ingestion_ready).lower()}\n"
            f"  row_counts={dict(bundle.row_counts)}\n"
            f"  blocked_by={list(bundle.blocked_by)}"
        )
        if require_ready and not bundle.ingestion_ready:
            raise typer.Exit(3)

    @app.command(name="build-trace-sidecar")
    def build_trace_sidecar(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 file"),
        output_dir: Path = typer.Option(
            Path("trace_sidecar"),
            "--output-dir",
            help="directory for sidecar emitter module and instrumentation patch",
        ),
        require_ready: bool = typer.Option(
            False,
            "--require-ready",
            help="exit non-zero when sidecar source evidence is not verified",
        ),
    ) -> None:
        """Build a non-invasive sidecar package for emitting raw GraphGPO events."""
        from lh_platform.studies import build_graphgpo_trace_sidecar, load_study

        protocol = load_study(study_path)
        compiled = protocol.compile()
        bundle = build_graphgpo_trace_sidecar(
            protocol,
            source_root=require_verified_source_root(compiled),
            output_dir=output_dir,
        )
        color = "green" if bundle.sidecar_ready else "yellow"
        console.print(
            f"[bold {color}]trace sidecar built: "
            f"{bundle.manifest_path}[/bold {color}]\n"
            f"  sidecar_ready={str(bundle.sidecar_ready).lower()}\n"
            f"  blocked_by={list(bundle.blocked_by)}\n"
            f"  module={bundle.sidecar_module}\n"
            f"  patch={bundle.instrumentation_patch}"
        )
        if require_ready and not bundle.sidecar_ready:
            raise typer.Exit(3)

    @app.command(name="build-state-snapshot-provider")
    def build_state_snapshot_provider(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 file"),
        output_dir: Path = typer.Option(
            Path("state_snapshot_provider"),
            "--output-dir",
            help="directory for exact-state provider module and instrumentation patch",
        ),
        require_ready: bool = typer.Option(
            False,
            "--require-ready",
            help="exit non-zero until a live restore probe verifies the provider",
        ),
    ) -> None:
        """Build a non-invasive exact-state snapshot provider for GraphGPO ALFWorld."""
        from lh_platform.studies import (
            build_graphgpo_state_snapshot_provider,
            load_study,
        )

        protocol = load_study(study_path)
        compiled = protocol.compile()
        provider = build_graphgpo_state_snapshot_provider(
            protocol,
            source_root=require_verified_source_root(compiled),
            output_dir=output_dir,
        )
        color = "green" if provider.provider_ready else "yellow"
        console.print(
            f"[bold {color}]state snapshot provider built: "
            f"{provider.manifest_path}[/bold {color}]\n"
            f"  provider_ready={str(provider.provider_ready).lower()}\n"
            f"  blocked_by={list(provider.blocked_by)}\n"
            f"  module={provider.provider_module}\n"
            f"  patch={provider.instrumentation_patch}"
        )
        if require_ready and not provider.provider_ready:
            raise typer.Exit(3)

    @app.command(name="build-state-restore-probe")
    def build_state_restore_probe(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 file"),
        output_dir: Path = typer.Option(
            Path("state_restore_probe"),
            "--output-dir",
            help="directory for live restore probe module and manifest",
        ),
        require_ready: bool = typer.Option(
            False,
            "--require-ready",
            help="exit non-zero until a live restore probe result verifies restore",
        ),
    ) -> None:
        """Build a live restore probe sidecar for GraphGPO ALFWorld snapshots."""
        from lh_platform.studies import build_graphgpo_state_restore_probe, load_study

        protocol = load_study(study_path)
        compiled = protocol.compile()
        probe = build_graphgpo_state_restore_probe(
            protocol,
            source_root=require_verified_source_root(compiled),
            output_dir=output_dir,
        )
        color = "green" if probe.probe_ready else "yellow"
        console.print(
            f"[bold {color}]state restore probe built: "
            f"{probe.manifest_path}[/bold {color}]\n"
            f"  probe_ready={str(probe.probe_ready).lower()}\n"
            f"  blocked_by={list(probe.blocked_by)}\n"
            f"  module={probe.probe_module}"
        )
        if require_ready and not probe.probe_ready:
            raise typer.Exit(3)

    @app.command(name="build-state-restore-probe-report")
    def build_state_restore_probe_report(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 file"),
        result_path: Path = typer.Option(
            ...,
            "--result-path",
            help="JSON result produced by lh_graphgpo_state_restore_probe.py",
        ),
        probe_manifest_path: Path = typer.Option(
            ...,
            "--probe-manifest-path",
            help="state_restore_probe_manifest.json generated with the sidecar",
        ),
        output_path: Path = typer.Option(
            Path("state_restore_probe_report.json"),
            "--output-path",
            help="path for the evaluated restore probe report",
        ),
        require_ready: bool = typer.Option(
            False,
            "--require-ready",
            help="exit non-zero when the probe result does not verify restore",
        ),
    ) -> None:
        """Evaluate a live restore probe result and fail closed on any mismatch."""
        from lh_platform.studies import (
            build_graphgpo_state_restore_probe_report,
            load_study,
        )

        protocol = load_study(study_path)
        report = build_graphgpo_state_restore_probe_report(
            protocol,
            result_path=result_path,
            probe_manifest_path=probe_manifest_path,
            output_path=output_path,
        )
        color = "green" if report.restore_verified else "red"
        console.print(
            f"[bold {color}]state restore probe report: "
            f"{report.report_path}[/bold {color}]\n"
            f"  restore_verified={str(report.restore_verified).lower()}\n"
            f"  blocked_by={list(report.blocked_by)}"
        )
        if require_ready and not report.restore_verified:
            raise typer.Exit(3)
