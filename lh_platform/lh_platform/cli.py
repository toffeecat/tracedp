"""CLI entry point for lh_platform."""

from __future__ import annotations

import typer
from rich.console import Console

from lh_platform.cli_commands.agentic_runtime import (
    register_agentic_runtime_commands,
)
from lh_platform.cli_commands.alfworld_evidence import (
    register_alfworld_evidence_commands,
)
from lh_platform.cli_commands.controlled_aliasing import (
    register_controlled_aliasing_commands,
)
from lh_platform.cli_commands.official_benchmarks import (
    register_official_benchmark_commands,
)
from lh_platform.cli_commands.plan_validation import register_plan_validation_command
from lh_platform.cli_commands.runtime_control import (
    register_model_check_command,
    register_runtime_commands,
)
from lh_platform.cli_commands.state_use import register_state_use_commands
from lh_platform.cli_commands.study_evidence import register_study_evidence_commands
from lh_platform.cli_commands.study_lifecycle import (
    register_study_lifecycle_commands,
    register_study_validation_command,
)
from lh_platform.cli_commands.study_preparation import (
    register_study_preparation_command,
)
from lh_platform.cli_commands.traceback import register_traceback_commands

app = typer.Typer(help="lh_platform - Unified Long-Horizon Agent Experiment Platform")
console = Console()


register_runtime_commands(app, console)


register_agentic_runtime_commands(app, console)


register_controlled_aliasing_commands(app, console)
register_official_benchmark_commands(app, console)


register_study_validation_command(app, console)


register_plan_validation_command(app, console)


register_alfworld_evidence_commands(app, console)


register_study_evidence_commands(app, console)


register_study_preparation_command(app, console)


register_study_lifecycle_commands(app, console)


register_model_check_command(app, console)


register_state_use_commands(app, console)


register_traceback_commands(app, console)


if __name__ == "__main__":
    app()
