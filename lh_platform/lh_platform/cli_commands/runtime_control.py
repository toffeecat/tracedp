"""Flat CLI commands for registry inspection and ordinary experiment runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from lh_platform.config.schema import (
    INTEGRATION_RANK,
    VALID_INTEGRATION_CHOICES_TEXT,
    VALIDITY_RANK,
)
from lh_platform.core.model_runtime import ModelRuntimePolicy
from lh_platform.core.registry import (
    BENCHMARKS,
    EVALUATORS,
    METHODS,
    MODELS,
    bootstrap_default_registries,
    bootstrap_runtime_registries,
)
from lh_platform.core.run_admission import resolve_method_admission
from lh_platform.core.runner import Runner


def _validate_cli_pair(
    benchmark: str,
    method: str,
    min_eval_validity: str,
    min_integration_level: str,
    allow_smoke_only: bool,
    *,
    strict_method_boundary: bool = False,
    method_kwargs: Mapping[str, Any] | None = None,
) -> str:
    if not BENCHMARKS.has(benchmark):
        return f"benchmark not registered: {benchmark}"
    bench_cls = BENCHMARKS._items[BENCHMARKS.resolve(benchmark)]
    validity = getattr(bench_cls, "eval_validity", "smoke_only")
    required_action_contract = str(getattr(bench_cls, "required_action_contract", "") or "")
    try:
        method_admission = resolve_method_admission(
            method_id=method,
            benchmark_id=benchmark,
            benchmark_required_action_contract=required_action_contract,
            strict=strict_method_boundary,
            method_kwargs=method_kwargs,
        )
    except KeyError as error:
        return str(error)
    if method_admission.reason:
        return method_admission.reason
    integration = method_admission.integration_level
    dynamic_validity = bool(getattr(bench_cls, "runtime_eval_validity_dynamic", False))
    if dynamic_validity:
        ceiling = getattr(bench_cls, "runtime_eval_validity_max", validity)
        if VALIDITY_RANK.get(ceiling, -1) < VALIDITY_RANK.get(min_eval_validity, 0):
            return (
                f"benchmark max_runtime_eval_validity={ceiling!r} < "
                f"min_eval_validity={min_eval_validity!r}"
            )
    else:
        if validity == "smoke_only" and not allow_smoke_only and min_eval_validity != "smoke_only":
            return (
                f"benchmark '{benchmark}' is smoke_only while "
                f"min_eval_validity={min_eval_validity!r}"
            )
        if VALIDITY_RANK.get(validity, -1) < VALIDITY_RANK.get(min_eval_validity, 0):
            return f"benchmark eval_validity={validity!r} < min_eval_validity={min_eval_validity!r}"
    if INTEGRATION_RANK.get(integration, -1) < INTEGRATION_RANK.get(min_integration_level, 0):
        return (
            f"method integration_level={integration!r} < "
            f"min_integration_level={min_integration_level!r}"
        )
    return ""


def register_runtime_commands(app: typer.Typer, console: Console) -> None:
    """Register registry inspection and ordinary single/matrix run commands."""

    @app.command()
    def check_env() -> None:
        """Print registered adapters and check local-repo availability."""
        bootstrap_default_registries()
        console.print("[bold green]=== Registries ===[/bold green]")
        for name, registry in [
            ("benchmark", BENCHMARKS),
            ("method", METHODS),
            ("model", MODELS),
            ("evaluator", EVALUATORS),
        ]:
            console.print(f"{name:>10}: {registry.available()}")
            if registry.aliases():
                console.print(f"{name + ' alias':>10}: {registry.aliases()}")

        from lh_platform.methods._local import has_repo

        console.print("\n[bold green]=== Local method repos ===[/bold green]")
        for slug in [
            "SeeAct",
            "CodeAct",
            "SWE-agent",
            "Agentless",
            "AutoCodeRover",
            "HiAgent",
        ]:
            console.print(f"  {slug:<14} -> {'OK' if has_repo(slug) else 'MISSING'}")

    @app.command()
    def run(
        benchmark: str = typer.Option("webvoyager", help="benchmark id"),
        method: str = typer.Option("noop", help="method id"),
        model: str = typer.Option(
            "dummy", help="model id or alias (e.g. gpt-4o-mini, openai_chat)"
        ),
        evaluator: str = typer.Option("long_horizon", help="evaluator id"),
        n_tasks: int = typer.Option(3, help="number of tasks to run"),
        seed: int | None = typer.Option(
            None,
            help="random seed for this single run; use run-config for a seed sweep",
        ),
        results_root: Path = typer.Option(Path("results"), help="results dir"),
        run_id: str | None = typer.Option(None, help="run id; default: auto"),
        model_config: str | None = typer.Option(
            None,
            "--model-config",
            help='JSON string of model kwargs, e.g. \'{"model":"gpt-4o-mini","temperature":0.0}\'',
        ),
        model_config_file: Path | None = typer.Option(
            None,
            "--model-config-file",
            help=("path to a JSON file with model kwargs; takes precedence over --model-config"),
        ),
        method_config: str | None = typer.Option(
            None,
            "--method-config",
            help='JSON string of method kwargs, e.g. \'{"mode":"native"}\'',
        ),
        method_config_file: Path | None = typer.Option(
            None,
            "--method-config-file",
            help="path to a JSON file with method kwargs",
        ),
        min_integration_level: str = typer.Option(
            "mock",
            help=(f"minimum method integration level: {VALID_INTEGRATION_CHOICES_TEXT}"),
        ),
        min_eval_validity: str = typer.Option(
            "smoke_only",
            help=(
                "minimum benchmark validity level: smoke_only|fixture_only|"
                "unverified_loader|real_loader|real_partial|real_native"
            ),
        ),
        allow_smoke_only: bool = typer.Option(
            False,
            "--allow-smoke-only",
            help=("allow smoke_only benchmarks even when min_eval_validity is higher"),
        ),
    ) -> None:
        """Run a single (benchmark, method, model) configuration."""
        model_kwargs: dict = {}
        if model_config_file is not None:
            if not model_config_file.is_file():
                console.print(
                    f"[bold red]--model-config-file not found: {model_config_file}[/bold red]"
                )
                raise typer.Exit(1)
            try:
                model_kwargs = json.loads(model_config_file.read_text())
                if not isinstance(model_kwargs, dict):
                    raise ValueError("must be a JSON object")
            except Exception as error:
                console.print(f"[bold red]invalid --model-config-file JSON: {error}[/bold red]")
                raise typer.Exit(1) from error
        elif model_config:
            try:
                model_kwargs = json.loads(model_config)
                if not isinstance(model_kwargs, dict):
                    raise ValueError("must be a JSON object")
            except Exception as error:
                console.print(f"[bold red]invalid --model-config JSON: {error}[/bold red]")
                raise typer.Exit(1) from error
        profile_model_id = str(model_kwargs.get("model_id") or "").strip()
        if profile_model_id:
            if model != "dummy" and model != profile_model_id:
                console.print(
                    "[bold red]--model conflicts with model_id declared in "
                    f"profile: {model!r} != {profile_model_id!r}[/bold red]"
                )
                raise typer.Exit(1)
            model = profile_model_id

        from lh_platform.core.model_runtime import split_model_kwargs_and_runtime

        try:
            model_kwargs, model_runtime = split_model_kwargs_and_runtime(model_kwargs)
        except ValueError as error:
            console.print(f"[bold red]invalid model runtime config: {error}[/bold red]")
            raise typer.Exit(1) from error

        method_kwargs: dict = {}
        if method_config_file is not None:
            if not method_config_file.is_file():
                console.print(
                    f"[bold red]--method-config-file not found: {method_config_file}[/bold red]"
                )
                raise typer.Exit(1)
            try:
                method_kwargs = json.loads(method_config_file.read_text())
                if not isinstance(method_kwargs, dict):
                    raise ValueError("must be a JSON object")
            except Exception as error:
                console.print(f"[bold red]invalid --method-config-file JSON: {error}[/bold red]")
                raise typer.Exit(1) from error
        elif method_config:
            try:
                method_kwargs = json.loads(method_config)
                if not isinstance(method_kwargs, dict):
                    raise ValueError("must be a JSON object")
            except Exception as error:
                console.print(f"[bold red]invalid --method-config JSON: {error}[/bold red]")
                raise typer.Exit(1) from error

        config_extra = {
            "model_kwargs": model_kwargs,
            "model_runtime": model_runtime,
            "method_kwargs": method_kwargs,
            "gates": {
                "allow_smoke_only": allow_smoke_only,
                "min_eval_validity": min_eval_validity,
                "min_integration_level": min_integration_level,
                "leaderboard_eligible": False,
            },
        }

        bootstrap_runtime_registries()
        try:
            resolved_provider = MODELS.resolve(model)
            runtime_policy = ModelRuntimePolicy.from_mapping(
                model_runtime,
                default_expected_model=(str(model_kwargs.get("model") or "").strip() or None),
            )
        except Exception as error:
            console.print(f"[bold red]invalid model runtime config: {error}[/bold red]")
            raise typer.Exit(1) from error
        strict_method_boundary = (
            resolved_provider == "openai_chat"
            and runtime_policy.require_live_server
            and not runtime_policy.allow_dummy_fallback
        )
        gate_reason = _validate_cli_pair(
            benchmark=benchmark,
            method=method,
            min_eval_validity=min_eval_validity,
            min_integration_level=min_integration_level,
            allow_smoke_only=allow_smoke_only,
            strict_method_boundary=strict_method_boundary,
            method_kwargs=method_kwargs,
        )
        if gate_reason:
            console.print(f"[bold red]refusing to run: {gate_reason}[/bold red]")
            raise typer.Exit(1)

        runner = Runner(
            benchmark=benchmark,
            method=method,
            model=model,
            evaluator=evaluator,
            n_tasks=n_tasks,
            results_root=results_root,
            run_id=run_id,
            config_extra=config_extra,
            seed=seed,
        )
        metrics = runner.run()
        console.print(f"[bold green]run_id={runner.run_id}[/bold green]")
        console.print(json.dumps(metrics.get("aggregate", {}), indent=2))
        if "per_task" in metrics and metrics["per_task"]:
            first_task = metrics["per_task"][0]
            console.print(
                "[dim]eval_validity="
                f"{first_task.get('benchmark_eval_validity', '?')}, "
                "method_integration_level="
                f"{first_task.get('method_integration_level', '?')}[/dim]"
            )

    @app.command(name="run-config")
    def run_config(
        config_path: Path = typer.Argument(..., help="path to experiment YAML/JSON config"),
        results_root: Path | None = typer.Option(
            None, "--results-root", help="override results_root from config"
        ),
    ) -> None:
        """Run a full experiment from a YAML/JSON config file.

        The config file describes a (benchmark x method) matrix with model
        profile, gates, and run settings. See configs/experiments/*.yaml.
        """
        from lh_platform.config.load import load_config
        from lh_platform.config.schema import ConfigError
        from lh_platform.scripts.run_experiment import run_experiment

        try:
            config = load_config(config_path)
        except ConfigError as error:
            console.print(f"[bold red]config error: {error}[/bold red]")
            raise typer.Exit(2) from error

        if results_root is not None:
            config.run.results_root = str(results_root)

        console.print(
            f"[bold green]=== {config.name} ===[/bold green]\n"
            f"  benchmarks={config.matrix.benchmarks}\n"
            f"  methods={config.matrix.methods}\n"
            f"  model={config.model.provider}\n"
            f"  results_root={config.run.results_root}\n"
            f"  gates={config.gates.min_eval_validity}/"
            f"{config.gates.min_integration_level} "
            f"allow_smoke_only={config.gates.allow_smoke_only}"
        )
        summaries = run_experiment(config)
        failed = sum(1 for summary in summaries if summary.get("error"))
        skipped = sum(1 for summary in summaries if summary.get("skipped"))
        console.print(
            f"\n[bold]done: {len(summaries)} pairs, {failed} errors, {skipped} skipped[/bold]"
        )
        raise typer.Exit(1 if failed else 0)


def register_model_check_command(app: typer.Typer, console: Console) -> None:
    """Register the model endpoint diagnostic at its legacy flat-CLI position."""

    @app.command()
    def model_check(
        model_config_file: Path = typer.Option(
            ..., "--model-config-file", help="path to a JSON/YAML model profile"
        ),
    ) -> None:
        """Check if a local model endpoint is reachable and functional."""
        import time

        bootstrap_runtime_registries()

        if not model_config_file.is_file():
            console.print(f"[bold red]File not found: {model_config_file}[/bold red]")
            raise typer.Exit(1)

        try:
            text = model_config_file.read_text()
            if model_config_file.suffix in {".yaml", ".yml"}:
                import yaml

                profile = yaml.safe_load(text)
            else:
                profile = json.loads(text)
        except Exception as error:
            console.print(f"[bold red]Failed to parse profile: {error}[/bold red]")
            raise typer.Exit(1) from error

        model_name = profile.get("model", "unknown")
        base_url = profile.get("base_url", "")
        api_key = profile.get("api_key", "dummy")

        console.print(f"[bold]Model profile: {model_config_file.name}[/bold]")
        console.print(f"  model: {model_name}")
        console.print(f"  base_url: {base_url}")
        console.print(f"  api_key: {'***' if api_key and api_key != 'dummy' else 'dummy'}")

        if not base_url:
            console.print("[bold red]No base_url set — cannot check endpoint.[/bold red]")
            raise typer.Exit(1)

        console.print("\n[bold cyan]Check 1: endpoint reachable[/bold cyan]")
        try:
            import urllib.request

            request = urllib.request.Request(f"{base_url}/models", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read())
            models = [item.get("id", "?") for item in data.get("data", [])]
            console.print(f"  OK — {len(models)} model(s) available: {models}")
        except Exception as error:
            console.print(f"  FAIL — {error}")
            raise typer.Exit(1) from error

        console.print("[bold cyan]Check 2: completion works[/bold cyan]")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
            started_at = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=8,
                temperature=0.0,
            )
            latency = time.time() - started_at
            response_text = response.choices[0].message.content or ""
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
            completion_tokens = (
                getattr(response.usage, "completion_tokens", 0) if response.usage else 0
            )
            console.print(
                f"  OK — response={response_text.strip()[:50]!r}, "
                f"latency={latency:.2f}s, "
                f"tokens={prompt_tokens}+{completion_tokens}"
            )
        except Exception as error:
            console.print(f"  FAIL — {error}")
            raise typer.Exit(1) from error

        console.print("\n[bold green]All checks passed.[/bold green]")


__all__ = ["register_model_check_command", "register_runtime_commands"]
