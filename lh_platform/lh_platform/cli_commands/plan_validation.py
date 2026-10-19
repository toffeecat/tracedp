"""CPU-only typed prospective-plan validation command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console


def register_plan_validation_command(app: typer.Typer, console: Console) -> None:
    """Register fail-closed plan expansion without execution imports."""

    @app.command(name="plan")
    def plan(
        plan_path: Path = typer.Argument(
            ...,
            help="prospective typed plan YAML/JSON inside workspace_root",
        ),
        validate: bool = typer.Option(
            False,
            "--validate",
            help="expand hashes and construct a synthetic non-runnable launch receipt",
        ),
        workspace_root: Path = typer.Option(
            Path("."),
            "--workspace-root",
            help="repository root used to resolve hash-bound plan references",
        ),
        data_root: Path = typer.Option(
            ...,
            "--data-root",
            help="ALFWorld data root used only to verify task artifacts",
        ),
        calibration_result: Path | None = typer.Option(
            None,
            "--calibration-result",
            help=(
                "already-collected calibration result to validate and select "
                "without target execution"
            ),
        ),
        output: Path = typer.Option(
            ...,
            "--output",
            help="JSON path for the validation or system-cell selection receipt",
        ),
    ) -> None:
        """Validate a typed prospective plan without launching models or environments."""

        if not validate:
            console.print(
                "[bold red]plan rejected: only CPU-only --validate mode exists[/bold red]"
            )
            raise typer.Exit(2)
        from lh_platform.training.trajectory_composition.n1_plan import (
            N1PlanValidationError,
            validate_n1_design_plan,
            write_n1_plan_validation_receipt,
        )

        try:
            validated = validate_n1_design_plan(
                plan_path,
                workspace_root=workspace_root,
                data_root=data_root,
            )
            if calibration_result is None:
                receipt_path = write_n1_plan_validation_receipt(validated, output)
                decision = "prospective plan valid but not runnable"
            else:
                from lh_platform.training.trajectory_composition.system_cell_calibration import (
                    SystemCellCalibrationError,
                    select_system_cell_from_calibration,
                    write_system_cell_selection_receipt,
                )

                try:
                    selection = select_system_cell_from_calibration(
                        validated,
                        calibration_result,
                        workspace_root=workspace_root,
                    )
                except SystemCellCalibrationError as error:
                    console.print(f"[bold red]system-cell selection failed: {error}[/bold red]")
                    raise typer.Exit(2) from error
                receipt_path = write_system_cell_selection_receipt(selection, output)
                decision = str(selection.receipt["decision"])
        except N1PlanValidationError as error:
            console.print(f"[bold red]plan validation failed: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]{decision}[/bold green]\n"
            f"  receipt={receipt_path}\n"
            "  run_ready=false\n"
            "  validation/selection-time model/environment/optimizer calls=0"
        )


__all__ = ["register_plan_validation_command"]
