"""CLI adapters for staging and invoking official benchmark boundaries."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console


def register_official_benchmark_commands(
    app: typer.Typer,
    console: Console,
) -> None:
    """Register official corpus, prediction-export, and evaluator commands."""

    @app.command(name="stage-toolbench-full")
    def stage_toolbench_full(
        data_root: Path = typer.Argument(
            ...,
            help=(
                "downloaded upstream ToolBench data/ directory; it must contain all six "
                "test sets and toolenv/tools"
            ),
        ),
        source_reference: str = typer.Option(
            ...,
            "--source-reference",
            help=(
                "immutable upstream archive/revision reference recorded in the manifest; "
                "this is provenance, not an evaluator score"
            ),
        ),
        manifest_path: Path | None = typer.Option(
            None,
            "--manifest-path",
            help=(
                "manifest destination; default is "
                "<data-root>/toolbench_official_corpus_manifest.v1.json"
            ),
        ),
        overwrite: bool = typer.Option(
            False,
            "--overwrite",
            help="deliberately replace an existing manifest after the data was changed",
        ),
    ) -> None:
        """Freeze a ToolBench corpus against the repository-owned trust root.

        A run becomes ``real_loader`` only when the corpus matches the reviewed
        ``configs/locks/toolbench_canonical_source_lock.v1.json`` file committed
        in the checked-out repository HEAD.  The command intentionally accepts no
        arbitrary lock path: a data-adjacent lock could self-authorize fabricated
        bytes.  It never executes RapidAPI/replay or ToolEval, so it never emits a
        ToolBench pass rate.
        """

        from lh_platform.benchmarks.toolbench_official import (
            ToolBenchOfficialDataError,
            load_official_corpus,
            stage_official_corpus,
        )

        try:
            staged_manifest = stage_official_corpus(
                data_root,
                source_reference=source_reference,
                manifest_path=manifest_path,
                overwrite=overwrite,
            )
            corpus = load_official_corpus(
                data_root,
                manifest_path=staged_manifest,
            )
        except ToolBenchOfficialDataError as error:
            console.print(f"[bold red]ToolBench corpus staging rejected: {error}[/bold red]")
            raise typer.Exit(2) from error

        counts = {
            test_set: len(corpus.rosters_by_test_set[test_set]) for test_set in corpus.test_sets
        }
        console.print(
            "[bold green]ToolBench full input corpus staged[/bold green]\n"
            f"  data_root={corpus.data_root}\n"
            f"  manifest={corpus.manifest_path}\n"
            f"  manifest_sha256={corpus.manifest_sha256}\n"
            f"  source_reference={corpus.source_reference}\n"
            f"  roster_counts={counts}\n"
            f"  source_verified={corpus.source_verified}\n"
            "  validity=real_loader only when source_verified=true; "
            "no native tool execution or ToolEval score"
        )

    @app.command(name="stage-scienceworld-official")
    def stage_scienceworld_official(
        roster: Path = typer.Argument(
            ...,
            help=(
                "human-selected test-only ScienceWorld roster JSON; it contains task name, "
                "variation, simplification, and max_steps but no gold path"
            ),
        ),
        task_manifest: Path = typer.Option(
            ...,
            "--task-manifest",
            help=(
                "repository-relative server data destination for the generated public-task "
                "manifest; must be inside this checkout"
            ),
        ),
        candidate_lock: Path = typer.Option(
            ...,
            "--candidate-lock",
            help=(
                "non-authorizing candidate lock destination; it must NOT be the fixed "
                "reviewed lock path"
            ),
        ),
        upstream_revision: str = typer.Option(
            ...,
            "--upstream-revision",
            help="audited full 40-hex allenai/ScienceWorld source revision",
        ),
        overwrite: bool = typer.Option(
            False,
            "--overwrite",
            help="deliberately replace an existing staged manifest/candidate after review",
        ),
    ) -> None:
        """Stage an official ScienceWorld test roster without self-authorizing it.

        This starts the installed official Python/JAR API only to verify each
        selected variation belongs to the upstream test split and to record public
        reset fields.  The emitted candidate is *not* accepted by the evaluator.
        A human must review it, create an approved lock at the one fixed path, and
        commit that lock before a config can pass ``real_native`` admission.
        """

        from lh_platform.benchmarks.scienceworld_official import (
            ScienceWorldOfficialError,
            repository_environment_lock_path,
        )
        from lh_platform.benchmarks.scienceworld_official import (
            stage_scienceworld_official as stage,
        )

        try:
            manifest, candidate = stage(
                roster_path=roster,
                task_manifest_path=task_manifest,
                candidate_lock_path=candidate_lock,
                upstream_revision=upstream_revision,
                overwrite=overwrite,
            )
        except ScienceWorldOfficialError as error:
            console.print(f"[bold red]ScienceWorld staging rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            "[bold yellow]ScienceWorld test manifest staged; not yet authorized[/bold yellow]\n"
            f"  manifest={manifest}\n"
            f"  candidate_lock={candidate}\n"
            f"  fixed_reviewed_lock={repository_environment_lock_path()}\n"
            "  next=independently review source revision/module/JAR/task hashes; then create "
            "an approved, HEAD-tracked fixed lock\n"
            "  validity=unverified_loader until that separate review is committed"
        )

    @app.command(name="export-swebench-predictions")
    def export_swebench_predictions(
        run_dir: Path = typer.Argument(
            ..., help="completed lh_platform swebench_lite run directory"
        ),
        output: Path = typer.Option(
            ..., "--output", help="official-format predictions.jsonl destination"
        ),
    ) -> None:
        """Export one complete runner roster as upstream SWE-bench predictions.jsonl.

        This only creates a validated prediction artifact.  It never reports a
        pass rate; use ``evaluate-swebench-official`` on a Docker-capable host for
        the upstream evaluation contract.
        """

        from lh_platform.benchmarks.swebench_official import (
            SWEbenchOfficialBridgeError,
            export_predictions_from_run,
        )

        try:
            bundle = export_predictions_from_run(
                run_dir,
                output,
            )
        except SWEbenchOfficialBridgeError as error:
            console.print(f"[bold red]SWE-bench prediction export failed: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            "[bold green]official-format prediction artifact written[/bold green]\n"
            f"  predictions={bundle.predictions_path}\n"
            f"  receipt={bundle.receipt_path}\n"
            f"  source_bundle={bundle.source_bundle_path}\n"
            f"  count={bundle.prediction_count}\n"
            "  claim=prediction_artifact_only (no pass rate yet)"
        )

    @app.command(name="evaluate-swebench-official")
    def evaluate_swebench_official(
        predictions: Path = typer.Argument(..., help="official-format predictions.jsonl"),
        output_dir: Path = typer.Option(
            ..., "--output-dir", help="isolated upstream harness working directory"
        ),
        run_id: str = typer.Option(..., "--run-id", help="safe upstream harness run id"),
        max_workers: int = typer.Option(1, "--max-workers", min=1),
        timeout_s: int = typer.Option(1800, "--timeout-s", min=1),
        execute: bool = typer.Option(
            False,
            "--execute",
            help=(
                "actually invoke the upstream Docker harness; default only writes a plan receipt"
            ),
        ),
        allow_unbound_predictions: bool = typer.Option(
            False,
            "--allow-unbound-predictions",
            help=(
                "allow a predictions.jsonl without the platform export receipt; "
                "the official receipt will be marked external_unbound"
            ),
        ),
        full_roster_lock: Path | None = typer.Option(
            None,
            "--full-roster-lock",
            help=(
                "local 300-instance SWE-bench Lite snapshot lock; it must additionally "
                "match the repository-reviewed canonical dataset anchor before real_native"
            ),
        ),
        harness_identity_lock: Path | None = typer.Option(
            None,
            "--harness-identity-lock",
            help=(
                "audited installed SWE-bench harness fingerprint; without it a completed "
                "full run remains official_unverified"
            ),
        ),
        official_dataset: Path | None = typer.Option(
            None,
            "--official-dataset",
            help=(
                "audited full SWE-bench Lite JSONL on this evaluation host; it is copied "
                "into output_dir/input before the Docker harness starts"
            ),
        ),
        official_dataset_manifest: Path | None = typer.Option(
            None,
            "--official-dataset-manifest",
            help=(
                "paired immutable manifest for --official-dataset; defaults to "
                "<official-dataset>.manifest.json"
            ),
        ),
    ) -> None:
        """Plan or run the official SWE-bench Docker harness without fallbacks."""

        from lh_platform.benchmarks.swebench_official import (
            SWEbenchOfficialBridgeError,
            run_official_evaluation,
        )

        try:
            receipt = run_official_evaluation(
                predictions,
                output_dir=output_dir,
                run_id=run_id,
                max_workers=max_workers,
                timeout_s=timeout_s,
                execute=execute,
                allow_unbound_predictions=allow_unbound_predictions,
                full_roster_lock_path=full_roster_lock,
                harness_identity_lock_path=harness_identity_lock,
                official_dataset_path=official_dataset,
                official_dataset_manifest_path=official_dataset_manifest,
            )
        except SWEbenchOfficialBridgeError as error:
            console.print(f"[bold red]SWE-bench official evaluation rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        color = "green" if receipt.real_native_completed else "yellow"
        console.print(
            f"[bold {color}]SWE-bench official evaluation: {receipt.status}[/bold {color}]\n"
            f"  receipt={receipt.receipt_path}\n"
            f"  execution_tier={receipt.execution_tier}\n"
            f"  input_provenance={receipt.input_provenance}\n"
            f"  evaluation_scope={receipt.evaluation_scope}\n"
            f"  evaluator_dataset={receipt.evaluator_dataset_path}\n"
            f"  official_harness_completed={receipt.official_harness_completed}\n"
            f"  real_native_completed={receipt.real_native_completed}"
        )
        if execute and not receipt.real_native_completed:
            raise typer.Exit(2)


__all__ = ["register_official_benchmark_commands"]
