"""Parent-child state-use and frozen-credit CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console


def register_state_use_commands(app: typer.Typer, console: Console) -> None:
    """Register flat public commands without changing their CLI names."""

    @app.command(name="run-alfworld-controlled-actor-factorial")
    def run_alfworld_controlled_actor_factorial_command(
        config_path: Path = typer.Argument(
            ...,
            help="controlled actor-state-exposure x credit factorial YAML",
        ),
        model_path: Path = typer.Option(..., "--model-path"),
        source_root: Path = typer.Option(..., "--source-root"),
        data_root: Path = typer.Option(..., "--data-root"),
        output_dir: Path = typer.Option(..., "--output-dir"),
        device: str = typer.Option("cuda", "--device"),
        preflight_only: bool = typer.Option(False, "--preflight-only"),
        allow_shared_gpu: bool = typer.Option(False, "--allow-shared-gpu"),
        generation_micro_batch_size: int = typer.Option(
            16,
            "--generation-micro-batch-size",
            min=1,
        ),
    ) -> None:
        """Collect real model rollouts once per exposure, then compare frozen credit."""

        from lh_platform.training.controlled_actor_factorial_run import (
            ControlledActorFactorialRunError,
            run_controlled_actor_factorial,
        )

        try:
            result = run_controlled_actor_factorial(
                config_path,
                model_path=model_path,
                source_root=source_root,
                data_root=data_root,
                output_dir=output_dir,
                device=device,
                preflight_only=preflight_only,
                allow_shared_gpu=allow_shared_gpu,
                generation_micro_batch_size=generation_micro_batch_size,
            )
        except (ControlledActorFactorialRunError, OSError, ValueError) as error:
            console.print(f"[bold red]controlled actor factorial rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]controlled actor factorial artifact written: {result}[/bold green]\n"
            "  claim_scope=controlled_actor_behavior_and_frozen_credit_diagnostics_"
            "not_policy_improvement"
        )

    @app.command(name="audit-controlled-actor-old-logprobs")
    def audit_controlled_actor_old_logprobs_command(
        config_path: Path = typer.Argument(
            ...,
            help="preregistered teacher-forced old-logprob audit YAML",
        ),
        source_result: Path = typer.Option(
            ...,
            "--source-result",
            help="immutable controlled actor factorial result JSON",
        ),
        model_path: Path = typer.Option(..., "--model-path"),
        source_root: Path = typer.Option(..., "--source-root"),
        data_root: Path = typer.Option(..., "--data-root"),
        output_path: Path = typer.Option(..., "--output-path"),
        device: str = typer.Option("cuda", "--device"),
        preflight_only: bool = typer.Option(False, "--preflight-only"),
        allow_shared_gpu: bool = typer.Option(False, "--allow-shared-gpu"),
    ) -> None:
        """Recompute behavior logprobs at the frozen initial policy checkpoint."""

        from lh_platform.training.teacher_forced_logprob_audit import (
            TeacherForcedLogprobAuditError,
            run_teacher_forced_old_logprob_audit,
        )

        try:
            result = run_teacher_forced_old_logprob_audit(
                config_path,
                source_result=source_result,
                model_path=model_path,
                source_root=source_root,
                data_root=data_root,
                output_path=output_path,
                device=device,
                preflight_only=preflight_only,
                allow_shared_gpu=allow_shared_gpu,
            )
        except (TeacherForcedLogprobAuditError, OSError, ValueError) as error:
            console.print(
                f"[bold red]teacher-forced old-logprob audit rejected: {error}[/bold red]"
            )
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]teacher-forced old-logprob artifact written: {result}[/bold green]\n"
            "  policy_update_executed=False\n"
            "  claim_scope=old_policy_probability_consistency_not_policy_improvement"
        )

    @app.command(name="run-alfworld-functional-state-use")
    def run_alfworld_functional_state_use_command(
        config_path: Path = typer.Argument(
            ...,
            help="frozen ALFWorld functional residual-state benchmark YAML",
        ),
        model_path: Path = typer.Option(..., "--model-path"),
        source_root: Path = typer.Option(..., "--source-root"),
        data_root: Path = typer.Option(..., "--data-root"),
        output_dir: Path = typer.Option(..., "--output-dir"),
        device: str = typer.Option("cuda", "--device"),
        preflight_only: bool = typer.Option(False, "--preflight-only"),
        collect_only: bool = typer.Option(
            False,
            "--collect-only",
            help="collect and hash environment evidence without loading the model",
        ),
        manifest_path: Path | None = typer.Option(
            None,
            "--manifest-path",
            help="score an existing hash-bound environment manifest without replay",
        ),
        adapter_path: Path | None = typer.Option(
            None,
            "--adapter-path",
            help="hash-bound PEFT adapter checkpoint to score",
        ),
        allow_shared_gpu: bool = typer.Option(False, "--allow-shared-gpu"),
        scoring_micro_batch_size: int = typer.Option(
            32,
            "--scoring-micro-batch-size",
            min=1,
        ),
    ) -> None:
        """Score all admissible actions at verified child-status state pairs."""

        from lh_platform.training.parent_child_state_use.runtime import (
            FunctionalStateUseRunError,
            run_functional_state_use_benchmark,
        )

        try:
            result = run_functional_state_use_benchmark(
                config_path,
                model_path=model_path,
                source_root=source_root,
                data_root=data_root,
                output_dir=output_dir,
                device=device,
                preflight_only=preflight_only,
                allow_shared_gpu=allow_shared_gpu,
                scoring_micro_batch_size=scoring_micro_batch_size,
                collect_only=collect_only,
                manifest_path=manifest_path,
                adapter_path=adapter_path,
            )
        except (FunctionalStateUseRunError, OSError, ValueError) as error:
            console.print(f"[bold red]functional state-use benchmark rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]functional state-use artifact written: {result}[/bold green]\n"
            "  policy_update_executed=False\n"
            "  claim_scope=controlled_frozen_candidate_action_prompt_sensitivity_"
            "not_endogenous_awareness_or_policy_improvement"
        )

    @app.command(name="analyze-functional-state-use")
    def analyze_functional_state_use_command(
        result_paths: list[Path] = typer.Argument(
            ...,
            help="frozen functional state-use result JSON files",
        ),
        output_path: Path = typer.Option(..., "--output-path"),
    ) -> None:
        """Aggregate preregistered pair directions without an RL claim."""

        from lh_platform.training.evidence_artifacts import write_json_exclusive
        from lh_platform.training.parent_child_state_use.analysis import (
            FunctionalStateUseAnalysisError,
            analyze_functional_state_use_result_files,
        )

        try:
            result = analyze_functional_state_use_result_files(result_paths)
            write_json_exclusive(output_path.resolve(), result)
        except (FunctionalStateUseAnalysisError, OSError, ValueError) as error:
            console.print(f"[bold red]functional state-use analysis rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]functional state-use analysis written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  models={result['model_count']}\n"
            f"  pairs={result['unique_environment_pair_count']}\n"
            "  claim_scope=descriptive_preregistered_controlled_pairs_"
            "not_population_awareness_or_policy_improvement"
        )

    @app.command(name="run-alfworld-state-exposure-rollout")
    def run_alfworld_state_exposure_rollout_command(
        config_path: Path = typer.Argument(
            ...,
            help="frozen four-condition ALFWorld rollout YAML",
        ),
        model_path: Path = typer.Option(..., "--model-path"),
        source_root: Path = typer.Option(..., "--source-root"),
        data_root: Path = typer.Option(..., "--data-root"),
        output_dir: Path = typer.Option(..., "--output-dir"),
        device: str = typer.Option("cuda", "--device"),
        preflight_only: bool = typer.Option(False, "--preflight-only"),
        adapter_path: Path | None = typer.Option(None, "--adapter-path"),
        allow_shared_gpu: bool = typer.Option(False, "--allow-shared-gpu"),
    ) -> None:
        """Compare normal, flat, graph, and corrupted-graph full rollouts."""

        from lh_platform.training.parent_child_state_use.rollout import (
            StateExposureRolloutError,
            run_alfworld_state_exposure_rollout,
        )

        try:
            result = run_alfworld_state_exposure_rollout(
                config_path,
                model_path=model_path,
                source_root=source_root,
                data_root=data_root,
                output_dir=output_dir,
                device=device,
                preflight_only=preflight_only,
                allow_shared_gpu=allow_shared_gpu,
                adapter_path=adapter_path,
            )
        except (StateExposureRolloutError, OSError, ValueError) as error:
            console.print(f"[bold red]state-exposure rollout rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        claim_scope = "state_exposure_rollout_preflight_only"
        try:
            artifact = json.loads(result.read_text(encoding="utf-8"))
            claim_scope = str(artifact["claim_scope"])
        except (OSError, ValueError, KeyError, TypeError):
            if not preflight_only:
                claim_scope = (
                    "frozen_full_rollout_state_exposure_diagnostic_"
                    "not_policy_improvement_or_endogenous_awareness"
                )
        console.print(
            f"[bold green]state-exposure rollout written: {result}[/bold green]\n"
            "  policy_update_executed=False\n"
            f"  claim_scope={claim_scope}"
        )

    @app.command(name="audit-frozen-credit-pairing")
    def audit_frozen_credit_pairing_command(
        rollout: Path = typer.Argument(
            ...,
            help="frozen model-policy rollout JSONL with exact token evidence",
        ),
        config_path: Path = typer.Option(
            ...,
            "--config-path",
            help="committed controlled factorial config used to resolve the recipe",
        ),
        data_root: Path = typer.Option(
            ...,
            "--data-root",
            help="official ALFWorld data root used to rebuild task metadata",
        ),
        output_path: Path = typer.Option(..., "--output-path"),
    ) -> None:
        """Independently recompute G2PO-TN and DPPO targets from one rollout."""

        from lh_platform.training.agentic_rl_config import (
            load_controlled_actor_factorial_manifest,
        )
        from lh_platform.training.agentic_rl_runtime import (
            load_agentic_rollout_artifact,
        )
        from lh_platform.training.alfworld_controlled_aliasing import (
            load_alfworld_controlled_aliasing_recipe,
        )
        from lh_platform.training.controlled_actor_factorial_run import (
            controlled_factorial_repository_bindings,
            resolve_controlled_task_binding,
        )
        from lh_platform.training.dppo.analysis import build_frozen_credit_pairing

        try:
            controlled = load_controlled_actor_factorial_manifest(config_path)
            recipe = load_alfworld_controlled_aliasing_recipe(
                controlled.workspace_root / controlled.payload["controlled_recipe"]["path"]
            )
            task, metadata, _, task_manifest_sha256 = resolve_controlled_task_binding(
                recipe,
                data_root,
            )
            preregistration_bindings = controlled_factorial_repository_bindings(controlled)
            result = build_frozen_credit_pairing(
                load_agentic_rollout_artifact(rollout),
                expected_recipe=recipe,
                expected_task_metadata=metadata,
                expected_task_manifest_sha256=task_manifest_sha256,
                expected_gamefile=str((data_root.resolve() / task.game_path).resolve()),
                preregistration_bindings=preregistration_bindings,
            )
            output_path = output_path.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except (OSError, ValueError) as error:
            console.print(f"[bold red]frozen credit pairing audit rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]frozen credit pairing audit written: {output_path}[/bold green]\n"
            f"  difference_count={result['credit_target_difference_count']}\n"
            f"  credit_difference_gate_passed={result['credit_difference_gate_passed']}\n"
            "  paired_policy_update_eligible=False\n"
            "  claim_scope=frozen_rollout_credit_difference_not_policy_improvement"
        )


__all__ = ["register_state_use_commands"]
