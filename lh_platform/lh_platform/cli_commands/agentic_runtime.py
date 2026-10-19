"""CLI entry points for structured agentic-RL execution and suite preflight."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console


def register_agentic_runtime_commands(app: typer.Typer, console: Console) -> None:
    """Register agentic-RL execution and AppWorld environment preflight."""

    @app.command(name="agentic-rl-p0", hidden=True)
    @app.command(name="agentic-rl-run")
    def run_agentic_rl_p0(
        config_path: Path = typer.Argument(
            ...,
            help="structured agentic-RL experiment composition YAML",
        ),
        model_path: Path = typer.Option(
            ...,
            "--model-path",
            help="local base-model checkpoint directory",
        ),
        source_root: Path = typer.Option(
            ...,
            "--source-root",
            help="pinned official G2PO source checkout",
        ),
        data_root: Path = typer.Option(
            ...,
            "--data-root",
            help="local ALFWORLD_DATA directory",
        ),
        output_dir: Path = typer.Option(
            ...,
            "--output-dir",
            help="run artifacts and append-only structured log directory",
        ),
        device: str | None = typer.Option(
            None,
            "--device",
            help="override configured device; use cpu for CPU-only preflight",
        ),
        preflight_only: bool = typer.Option(
            False,
            "--preflight-only",
            help="hash and validate all inputs without loading the model or GPU",
        ),
        allow_shared_gpu: bool = typer.Option(
            False,
            "--allow-shared-gpu",
            help="explicitly permit a GPU launch while another compute process is active",
        ),
        generation_micro_batch_size: int = typer.Option(
            16,
            "--generation-micro-batch-size",
            min=1,
            help=(
                "single-device Transformers generation batch size; logged as an execution factor"
            ),
        ),
        training_backend: str = typer.Option(
            "local_transformers",
            "--training-backend",
            help="optimizer/rollout authority: local_transformers or gated verl",
        ),
        verl_source_root: Path | None = typer.Option(
            None,
            "--verl-source-root",
            help="runtime-only pinned veRL checkout; never stored in portable YAML",
        ),
        verl_parity_receipt: Path | None = typer.Option(
            None,
            "--verl-parity-receipt",
            help="passed full-model frozen-update parity receipt required for live veRL",
        ),
    ) -> None:
        """Resolve and run a structured agentic-RL experiment composition."""

        from lh_platform.training.agentic_rl_p0 import (
            AgenticRLP0ConfigError,
            namespace_from_experiment_config,
            run_p0,
        )

        try:
            args = namespace_from_experiment_config(
                config_path,
                model_path=model_path,
                source_root=source_root,
                data_root=data_root,
                output_dir=output_dir,
                device=device,
                preflight_only=preflight_only,
                allow_shared_gpu=allow_shared_gpu,
                training_backend=training_backend,
                verl_source_root=verl_source_root,
                verl_parity_receipt=verl_parity_receipt,
            )
            args.full_eval_only = args.execution_mode == "full_eval_only"
            args.generation_micro_batch_size = generation_micro_batch_size
            result_path = run_p0(args)
        except AgenticRLP0ConfigError as error:
            console.print(f"[bold red]agentic-RL config error: {error}[/bold red]")
            raise typer.Exit(2) from error
        except Exception as error:
            console.print(
                f"[bold red]agentic-RL run failed: {type(error).__name__}: {error}[/bold red]"
            )
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]agentic-RL artifact written: {result_path}[/bold green]\n"
            f"  structured log={output_dir.resolve() / 'run_events.jsonl'}"
        )

    @app.command(name="appworld-agentic-preflight")
    def appworld_agentic_preflight(
        config_path: Path = typer.Argument(
            ...,
            help="structured native AppWorld train/dev suite YAML",
        ),
        source_root: Path = typer.Option(
            ...,
            "--source-root",
            help="pinned official AppWorld source checkout imported by the environment",
        ),
        data_root: Path = typer.Option(
            ...,
            "--data-root",
            help="official AppWorld data root",
        ),
        output_dir: Path = typer.Option(
            ...,
            "--output-dir",
            help="preflight receipt directory",
        ),
    ) -> None:
        """Verify AppWorld source/data/splits without loading a model or GPU."""

        from lh_platform.training.appworld_agentic_suite import (
            AppWorldSuiteError,
            preflight_official_appworld_suite_pair,
        )

        try:
            result_path = preflight_official_appworld_suite_pair(
                config_path,
                source_root=source_root,
                data_root=data_root,
                output_dir=output_dir,
            )
        except AppWorldSuiteError as error:
            console.print(f"[bold red]AppWorld preflight rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]AppWorld native suite preflight written: "
            f"{result_path}[/bold green]\n"
            "  claim_scope=native_suite_preflight_only_no_training_or_score"
        )


__all__ = ["register_agentic_runtime_commands"]
