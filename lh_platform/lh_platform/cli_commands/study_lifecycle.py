"""Study validation, execution, audit, status, and analysis commands."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from lh_platform.studies.evidence import require_verified_source_root

__all__ = [
    "register_study_lifecycle_commands",
    "register_study_validation_command",
]


def register_study_validation_command(
    app: typer.Typer,
    console: Console,
) -> None:
    """Register protocol-only study validation at its legacy CLI position."""

    @app.command(name="validate-study")
    def validate_study(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 YAML/JSON file"),
        output_dir: Path | None = typer.Option(
            None,
            "--output-dir",
            help="directory for evidence/comparison/trial/validity manifests",
        ),
    ) -> None:
        """Compile a purpose-driven study without running any trials."""
        from lh_platform.studies import (
            StudyConfigError,
            StudyValidationError,
            load_study,
            write_study_artifacts,
        )

        try:
            protocol = load_study(study_path)
            compiled = protocol.compile()
        except StudyValidationError as error:
            console.print("[bold red]study validation failed[/bold red]")
            for issue in error.issues:
                console.print(
                    f"  [{issue.code}] {issue.path}: {issue.message}",
                    markup=False,
                )
            raise typer.Exit(2) from error
        except StudyConfigError as error:
            console.print(f"[bold red]study config error: {error}[/bold red]")
            raise typer.Exit(2) from error

        target = output_dir or Path("results") / "studies" / protocol.id / "plan"
        paths = write_study_artifacts(protocol, compiled, target)
        console.print(
            f"[bold green]study protocol valid: {protocol.id}[/bold green]\n"
            f"  {len(compiled.trials)} trial plans\n"
            "  run_ready=false (Phase 0 does not verify runtime artifacts)\n"
            f"  artifacts={target}"
        )
        for path in paths:
            console.print(f"  - {path.name}")


def register_study_lifecycle_commands(
    app: typer.Typer,
    console: Console,
) -> None:
    """Register prepared-study lifecycle commands on the public flat CLI."""

    def print_study_run_audit(audit: Mapping[str, Any]) -> None:
        paths = audit["artifact_paths"]
        color = "green" if audit["decision"] == "ready_for_outcome_analysis" else "yellow"
        console.print(
            f"[bold {color}]study run audit: {audit['decision']}[/bold {color}]\n"
            f"  blocked_by={audit['blocked_by']}\n"
            f"  json={paths['json']}\n"
            f"  markdown={paths['markdown']}"
        )

    @app.command(name="run-study")
    def run_prepared_study(
        prepared_dir: Path = typer.Argument(
            ...,
            help="directory produced by prepare-study",
        ),
        trial_id: list[str] = typer.Option(
            [],
            "--trial-id",
            help="specific prepared trial; repeatable",
        ),
        resume: bool = typer.Option(
            False,
            "--resume",
            help="skip completed trials and resume the study",
        ),
        training_steps: int | None = typer.Option(
            None,
            "--training-steps",
            min=1,
            help="calibration-only step override",
        ),
        calibration_small: bool = typer.Option(
            False,
            "--calibration-small",
            help="reduce batch, validation, and rollout concurrency for smoke-only calibration",
        ),
        require_audit_ready: bool = typer.Option(
            False,
            "--require-audit-ready",
            help="exit non-zero unless the generated run audit is ready for outcome analysis",
        ),
    ) -> None:
        """Run prepared official commands with durable logs, metrics and SVG curves."""
        from lh_platform.studies import StudyExecutor, build_study_run_audit

        results = StudyExecutor(prepared_dir).run(
            trial_ids=trial_id,
            resume=resume,
            training_steps=training_steps,
            calibration_small=calibration_small,
        )
        for result in results:
            color = "green" if result.status == "completed" else "red"
            console.print(
                f"[bold {color}]{result.trial_id}: {result.status}[/bold {color}] "
                f"elapsed={result.elapsed_s:.1f}s metrics={result.parsed_metric_events} "
                f"warnings={result.postrun_warning_count} "
                f"output={result.output_dir}"
            )
        audit = build_study_run_audit(prepared_dir)
        print_study_run_audit(audit)
        if any(result.status != "completed" for result in results):
            raise typer.Exit(3)
        if require_audit_ready and audit["decision"] != "ready_for_outcome_analysis":
            raise typer.Exit(3)

    @app.command(name="audit-study-run")
    def audit_study_run(
        prepared_dir: Path = typer.Argument(
            ...,
            help="directory produced by prepare-study",
        ),
        output: Path | None = typer.Option(
            None,
            "--output",
            help="JSON audit output; default: <prepared_dir>/study_run_audit.json",
        ),
        markdown: Path | None = typer.Option(
            None,
            "--markdown",
            help="Markdown audit output; default: <prepared_dir>/study_run_audit.md",
        ),
    ) -> None:
        """Build a fail-closed audit report from run artifacts."""
        from lh_platform.studies import build_study_run_audit

        audit = build_study_run_audit(
            prepared_dir,
            output_path=output,
            markdown_path=markdown,
        )
        print_study_run_audit(audit)

    @app.command(name="backfill-study-metrics")
    def backfill_study_metrics(
        prepared_dir: Path = typer.Argument(
            ...,
            help="directory produced by prepare-study",
        ),
        trial_id: list[str] = typer.Option(
            [],
            "--trial-id",
            help="specific completed trial; repeatable",
        ),
    ) -> None:
        """Reparse durable upstream stdout and regenerate CSV/SVG metric artifacts."""
        from lh_platform.studies.execution import backfill_study_metrics as _backfill

        results = _backfill(prepared_dir, trial_ids=trial_id)
        for result in results:
            console.print(
                f"[bold]metrics backfilled[/bold] {result['trial_id']} "
                f"reparsed={result['parsed_metric_event_count']} "
                f"csv={result['csv']} svg={result['svg']}"
            )

    @app.command(name="graphgpo-progress")
    def graphgpo_progress(
        prepared_dir: Path = typer.Argument(
            ...,
            help="directory produced by prepare-study",
        ),
        trial_id: str | None = typer.Option(
            None,
            "--trial-id",
            help="prepared GraphGPO trial to inspect",
        ),
        output: Path | None = typer.Option(
            None,
            "--output",
            help="JSON output; default: <prepared_dir>/graphgpo_progress.json",
        ),
        markdown: Path | None = typer.Option(
            None,
            "--markdown",
            help="Markdown output; default: <prepared_dir>/graphgpo_progress.md",
        ),
    ) -> None:
        """Report GraphGPO validation, rollout, and training progress without changing a run."""
        from lh_platform.studies import build_graphgpo_progress_snapshot

        snapshot = build_graphgpo_progress_snapshot(
            prepared_dir,
            trial_id=trial_id,
            output_path=output,
            markdown_path=markdown,
        )
        console.print(
            f"[bold]GraphGPO progress[/bold] trials={snapshot['trial_count']} "
            f"source_contract={str(snapshot['source_contract']['verified']).lower()}\n"
            f"  json={snapshot['artifact_paths']['json']}\n"
            f"  markdown={snapshot['artifact_paths']['markdown']}"
        )

    @app.command(name="graphgpo-parity-audit")
    def graphgpo_parity_audit(
        prepared_dir: Path = typer.Argument(
            ...,
            help="directory produced by prepare-study",
        ),
        source_root: Path = typer.Option(
            ...,
            "--source-root",
            help="required: path to the official GraphGPO/verl-agent source mirror (no default)",
        ),
        trial_id: str = typer.Option(
            ...,
            "--trial-id",
            help="prepared trial id to audit",
        ),
        output: Path | None = typer.Option(
            None,
            "--output",
            help="JSON output; default: <prepared_dir>/graphgpo_parity_audit.json",
        ),
        markdown: Path | None = typer.Option(
            None,
            "--markdown",
            help="Markdown output; default: <prepared_dir>/graphgpo_parity_audit.md",
        ),
    ) -> None:
        """Build a source-backed GraphGPO parity audit without rerunning training."""
        from lh_platform.studies import build_graphgpo_parity_audit

        audit = build_graphgpo_parity_audit(
            prepared_dir,
            source_root=source_root,
            trial_id=trial_id,
            output_path=output,
            markdown_path=markdown,
        )
        color = "green" if audit["formal_claim_supported"] else "yellow"
        paths = audit["artifact_paths"]
        console.print(
            f"[bold {color}]graphgpo parity audit: {audit['decision']}[/bold {color}]\n"
            f"  formal_claim_supported={audit['formal_claim_supported']}\n"
            f"  blocked_by={audit['blocked_by']}\n"
            f"  json={paths['json']}\n"
            f"  markdown={paths['markdown']}"
        )

    @app.command(name="build-native-outcome-sidecar")
    def build_native_outcome_sidecar(
        study_path: Path = typer.Argument(...),
        output_dir: Path = typer.Option(
            Path("native_outcome_sidecar"),
            "--output-dir",
        ),
    ) -> None:
        """Build the audited patch that emits task-level native ALFWorld outcomes."""
        from lh_platform.studies import build_outcome_sidecar_patch, load_study

        protocol = load_study(study_path)
        compiled = protocol.compile()
        bundle = build_outcome_sidecar_patch(
            source_root=require_verified_source_root(compiled),
            source_commit=str(protocol.design.fixed_factors["source_commit"]),
            output_dir=output_dir,
        )
        console.print(
            f"[bold green]native outcome sidecar built[/bold green]\n"
            f"  patch={bundle.patch_path}\n  source_commit={bundle.source_commit}"
        )

    @app.command(name="study-status")
    def study_status(prepared_dir: Path = typer.Argument(...)) -> None:
        """Print the durable state of a prepared study."""
        from lh_platform.studies import StudyExecutor

        console.print_json(data=dict(StudyExecutor(prepared_dir).status()))

    @app.command(name="analyze-study")
    def analyze_study(
        outcomes: Path = typer.Argument(
            ...,
            help="native task-level outcome JSONL",
        ),
        output: Path = typer.Option(
            Path("study_result.json"),
            "--output",
        ),
    ) -> None:
        """Evaluate the preregistered paper-near and superiority thresholds."""
        from lh_platform.studies import build_replication_result

        rows = [
            json.loads(line)
            for line in outcomes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result = build_replication_result(rows)
        result.write(output)
        color = "green" if result.verdict == "replicated" else "yellow"
        console.print(
            f"[bold {color}]study verdict: {result.verdict}[/bold {color}]\n"
            f"  GraphGPO={result.graphgpo_success_rate:.4f}\n"
            f"  GRPO={result.grpo_success_rate:.4f}\n"
            f"  paired_difference={result.paired_difference:.4f}\n"
            f"  ci95={result.confidence_interval_95}\n"
            f"  artifact={output}"
        )
