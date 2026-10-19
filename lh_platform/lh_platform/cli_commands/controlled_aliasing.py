"""Controlled-aliasing collection and offline credit-diagnostic commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

__all__ = ["register_controlled_aliasing_commands"]


def register_controlled_aliasing_commands(
    app: typer.Typer,
    console: Console,
) -> None:
    """Register controlled-aliasing commands on the public flat CLI."""

    @app.command(name="analyze-agentic-rl-rollout")
    def analyze_agentic_rl_rollout(
        annotated_rollout: Path = typer.Argument(
            ...,
            help="annotated rollout JSONL emitted by the paired runner",
        ),
        paired_rollout: Path | None = typer.Option(
            None,
            "--paired-rollout",
            help="control/raw rollout used to verify a credit-only paired trace",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="structured mechanism/collection analysis JSON",
        ),
    ) -> None:
        """Audit collection quality and counterfactual G2PO/DPPO credit offline."""

        from lh_platform.training.dppo.analysis import (
            DPPORolloutAnalysisError,
            analyze_rollout_artifact,
        )

        try:
            result = analyze_rollout_artifact(
                annotated_rollout,
                paired_path=paired_rollout,
                output_path=output_path,
            )
        except (DPPORolloutAnalysisError, ValueError) as error:
            console.print(f"[bold red]rollout analysis rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        mechanism = result.get("mechanism") or {}
        counterfactual = result.get("counterfactual_credit") or {}
        collection = result["collection"]
        console.print(
            f"[bold green]rollout analysis written: {output_path.resolve()}[/bold green]\n"
            f"  valid_action_rate={collection['valid_action_rate']:.3f}\n"
            f"  successful_trajectories={collection['successful_trajectory_count']}\n"
            "  phase_progression_trajectories="
            f"{collection['trajectories_with_phase_progression_count']}\n"
            f"  mechanism_passed={mechanism.get('passed', False)}\n"
            "  coarse_groups_split_by_refinement="
            f"{counterfactual.get('coarse_state_groups_split_by_refinement', 0)}\n"
            "  claim_scope=collection_and_credit_diagnostics_not_policy_effectiveness"
        )

    @app.command(name="audit-controlled-aliasing")
    def audit_controlled_aliasing(
        rollout: Path = typer.Argument(
            ...,
            help="annotated ALFWorld rollout JSONL emitted by the paired runner",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="evidence-gated controlled-aliasing admission manifest",
        ),
    ) -> None:
        """Require a real official-anchor collision before spending GPU time."""

        from lh_platform.training.controlled_aliasing import (
            analyze_controlled_aliasing_artifact,
        )
        from lh_platform.training.dppo.analysis import DPPORolloutAnalysisError

        try:
            result = analyze_controlled_aliasing_artifact(
                rollout,
                output_path=output_path,
            )
        except (DPPORolloutAnalysisError, OSError, ValueError) as error:
            console.print(f"[bold red]controlled-aliasing audit rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        status = "green" if result["admitted"] else "yellow"
        console.print(
            f"[bold {status}]controlled-aliasing audit written: "
            f"{output_path.resolve()}[/bold {status}]\n"
            f"  eligible_records={result['eligible_record_count']}\n"
            f"  collision_groups={result['collision_group_count']}\n"
            f"  candidate_pairs={result['candidate_pair_count']}\n"
            f"  admitted={result['admitted']}\n"
            f"  blocked_by={','.join(result['blocked_by']) or 'none'}\n"
            "  claim_scope=controlled_cross_frontier_collision_candidate_"
            "not_credit_or_policy_effectiveness"
        )

    @app.command(name="collect-alfworld-controlled-aliasing")
    def collect_alfworld_controlled_aliasing_command(
        recipe: Path = typer.Argument(
            ...,
            help="structured controlled-aliasing recipe YAML",
        ),
        source_root: Path = typer.Option(
            ...,
            "--source-root",
            help="pinned original G2PO checkout",
        ),
        data_root: Path = typer.Option(
            ...,
            "--data-root",
            help="official ALFWORLD_DATA root",
        ),
        output_dir: Path = typer.Option(
            ...,
            "--output-dir",
            help="snapshot, restore, rollout, and admission artifact directory",
        ),
    ) -> None:
        """Collect a CPU-only, restore-verified official-anchor collision pair."""

        from lh_platform.training.alfworld_controlled_aliasing import (
            AlfWorldControlledAliasingError,
            collect_alfworld_controlled_aliasing,
        )
        from lh_platform.training.controlled_continuations import (
            CONTROLLED_CONTINUATION_EVIDENCE_CLAIM_SCOPE,
        )

        try:
            receipt = collect_alfworld_controlled_aliasing(
                recipe,
                source_root=source_root,
                data_root=data_root,
                output_dir=output_dir,
            )
        except (AlfWorldControlledAliasingError, OSError, ValueError) as error:
            console.print(f"[bold red]controlled ALFWorld collection rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]controlled ALFWorld evidence admitted: {receipt}[/bold green]\n"
            "  gpu_required=false\n"
            f"  claim_scope={CONTROLLED_CONTINUATION_EVIDENCE_CLAIM_SCOPE}"
        )

    @app.command(name="audit-controlled-continuations")
    def audit_controlled_continuations(
        matrix: Path = typer.Argument(
            ...,
            help="controlled continuation matrix emitted by the ALFWorld collector",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="independently recomputed continuation-score analysis",
        ),
    ) -> None:
        """Recompute coarse/refined scores from one frozen program matrix."""

        from lh_platform.training.controlled_continuations import (
            CONTROLLED_CONTINUATION_EVIDENCE_CLAIM_SCOPE,
            ControlledContinuationAnalysisError,
            analyze_controlled_continuation_artifact,
        )

        try:
            result = analyze_controlled_continuation_artifact(
                matrix,
                output_path=output_path,
            )
        except (ControlledContinuationAnalysisError, OSError, ValueError) as error:
            console.print(f"[bold red]controlled continuation audit rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        status = "green" if result["admitted"] else "yellow"
        refined = ",".join(
            f"{row['state_id']}={row['continuation_score']:.6f}" for row in result["refined_scores"]
        )
        console.print(
            f"[bold {status}]controlled continuation audit written: "
            f"{output_path.resolve()}[/bold {status}]\n"
            f"  matrix_cells={result['matrix_cell_count']}\n"
            f"  coarse_score={result['coarse_score']['continuation_score']:.6f}\n"
            f"  refined_scores={refined}\n"
            f"  score_range={result['continuation_score_range']:.6f}\n"
            f"  empirical_mse_reduction={result['empirical_mse_reduction']:.6f}\n"
            f"  admitted={result['admitted']}\n"
            f"  claim_scope={CONTROLLED_CONTINUATION_EVIDENCE_CLAIM_SCOPE}"
        )

    @app.command(name="audit-residual-credit-calibration")
    def audit_residual_credit_calibration(
        config_path: Path = typer.Argument(
            ...,
            help="structured oracle hard-split calibration diagnostic YAML",
        ),
        rollout: list[Path] = typer.Option(
            ...,
            "--rollout",
            help="verified normal-actor annotated rollout; repeat per pair",
        ),
        output_path: Path = typer.Option(..., "--output-path"),
    ) -> None:
        """Compare held-out coarse, refined, random, and exact-state estimators."""

        from lh_platform.training.config_provenance import (
            repository_config_tree_bindings,
        )
        from lh_platform.training.dataset import sha256_file
        from lh_platform.training.residual_credit.calibration import (
            ResidualCreditCalibrationError,
            analyze_residual_credit_calibration_artifacts,
            load_residual_credit_calibration_config,
            residual_calibration_recipe_contracts,
        )

        try:
            resolved = load_residual_credit_calibration_config(config_path)
            estimator = resolved.payload["estimator"]
            admission = resolved.payload["admission"]
            bindings = repository_config_tree_bindings(resolved)
            result = analyze_residual_credit_calibration_artifacts(
                rollout,
                gamma=float(estimator["gamma"]),
                success_reward=float(estimator["success_reward"]),
                random_split_seeds=tuple(estimator["matched_random_split_seeds"]),
                minimum_pair_count=int(admission["formal_minimum_pair_count"]),
                minimum_loto_coverage=float(admission["minimum_loto_coverage"]),
                recipe_contracts=residual_calibration_recipe_contracts(resolved),
                preregistration={
                    "config_path": str(resolved.config_path),
                    "config_file_sha256": sha256_file(resolved.config_path),
                    "config_sha256": resolved.sha256,
                    "bindings": [dict(binding) for binding in bindings],
                },
                output_path=output_path,
            )
        except (ResidualCreditCalibrationError, OSError, ValueError) as error:
            console.print(f"[bold red]residual calibration rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        status = "green" if result["calibration_gate_passed"] else "yellow"
        console.print(
            f"[bold {status}]residual calibration written: "
            f"{output_path.resolve()}[/bold {status}]\n"
            f"  pair_count={result['pair_count']}\n"
            f"  formal_gate_passed={result['calibration_gate_passed']}\n"
            f"  blocked_by={','.join(result['blocked_by']) or 'none'}\n"
            "  policy_update_executed=False\n"
            f"  claim_scope={result['claim_scope']}"
        )

    @app.command(name="run-alfworld-residual-credit-calibration")
    def run_alfworld_residual_credit_calibration(
        config_path: Path = typer.Argument(
            ...,
            help="structured oracle hard-split calibration diagnostic YAML",
        ),
        model_path: Path = typer.Option(..., "--model-path"),
        source_root: Path = typer.Option(..., "--source-root"),
        data_root: Path = typer.Option(..., "--data-root"),
        output_dir: Path = typer.Option(..., "--output-dir"),
        device: str = typer.Option("cuda", "--device"),
        preflight_only: bool = typer.Option(False, "--preflight-only"),
        allow_shared_gpu: bool = typer.Option(False, "--allow-shared-gpu"),
    ) -> None:
        """Collect privileged-pair continuations for an oracle diagnostic."""

        from lh_platform.training.residual_credit.calibration import (
            ResidualCreditCalibrationError,
        )
        from lh_platform.training.residual_credit.runtime import (
            ResidualCreditCalibrationRunError,
            run_residual_credit_calibration_pilot,
        )

        try:
            result = run_residual_credit_calibration_pilot(
                config_path,
                model_path=model_path,
                source_root=source_root,
                data_root=data_root,
                output_dir=output_dir,
                device=device,
                preflight_only=preflight_only,
                allow_shared_gpu=allow_shared_gpu,
            )
        except (
            ResidualCreditCalibrationError,
            ResidualCreditCalibrationRunError,
            OSError,
            ValueError,
        ) as error:
            console.print(f"[bold red]oracle diagnostic rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]oracle diagnostic written: {result.resolve()}[/bold green]\n"
            "  policy_update_executed=False\n"
            "  claim_scope=development_heldout_calibration_not_recursive_dp"
        )

    @app.command(name="run-controlled-credit-aliasing-sweep")
    def run_controlled_credit_aliasing_sweep_command(
        config_path: Path = typer.Argument(
            ...,
            help="CPU-only controlled credit-aliasing sweep YAML",
        ),
        output_path: Path = typer.Option(..., "--output-path"),
    ) -> None:
        """Identify estimator bias/variance before model or environment runs."""

        from lh_platform.training.controlled_credit_aliasing_sweep import (
            run_controlled_credit_aliasing_sweep,
        )
        from lh_platform.training.residual_credit.calibration import (
            ResidualCreditCalibrationError,
        )

        try:
            result = run_controlled_credit_aliasing_sweep(
                config_path,
                output_path=output_path,
            )
        except (ResidualCreditCalibrationError, OSError, ValueError) as error:
            console.print(
                f"[bold red]controlled credit-aliasing sweep rejected: {error}[/bold red]"
            )
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]controlled sweep written: {output_path.resolve()}[/bold green]\n"
            f"  cells={result['cell_count']}\n"
            "  gpu_required=false\n"
            f"  claim_scope={result['claim_scope']}"
        )

    @app.command(name="analyze-aliasing-horizon-evidence")
    def analyze_aliasing_horizon_evidence_command(
        sweep_path: Path = typer.Argument(
            ...,
            help="controlled estimator-ladder artifact",
        ),
        pilot_registry_path: Path = typer.Argument(
            ...,
            help="structured registry of prior natural and controlled pilots",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="claim-gated aliasing/horizon evidence JSON",
        ),
    ) -> None:
        """Separate controlled aliasing effects from natural horizon prevalence."""

        from lh_platform.training.aliasing_horizon_evidence import (
            AliasingHorizonEvidenceError,
            build_aliasing_horizon_evidence,
        )

        try:
            result = build_aliasing_horizon_evidence(
                sweep_path,
                pilot_registry_path,
                output_path=output_path,
            )
        except (AliasingHorizonEvidenceError, OSError, ValueError) as error:
            console.print(f"[bold red]aliasing/horizon evidence rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        gates = result["claim_gates"]
        console.print(
            f"[bold green]aliasing/horizon evidence written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  controlled_effect_observed="
            f"{str(gates['controlled_aliasing_effect_observed']).lower()}\n"
            f"  natural_correlation_established="
            f"{str(gates['natural_aliasing_horizon_correlation_established']).lower()}\n"
            f"  claim_scope={result['claim_scope']}"
        )

    @app.command(name="materialize-natural-aliasing-rows")
    def materialize_natural_aliasing_rows_command(
        collection_result_path: Path = typer.Argument(
            ...,
            help="frozen two-dataset collection_result.json",
        ),
        output_path: Path = typer.Option(..., "--output-path"),
        horizons: str = typer.Option("4,8,12,16", "--horizons"),
    ) -> None:
        """Build paired-prefix natural aliasing rows without updating a policy."""

        from lh_platform.training.natural_aliasing import (
            NaturalAliasingError,
            materialize_natural_aliasing_rows,
        )

        try:
            parsed_horizons = tuple(int(value) for value in horizons.split(","))
            result = materialize_natural_aliasing_rows(
                collection_result_path,
                horizons=parsed_horizons,
                output_path=output_path,
            )
        except (NaturalAliasingError, OSError, ValueError) as error:
            console.print(f"[bold red]natural aliasing rows rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]natural aliasing rows written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  rows={len(result['rows'])}\n"
            f"  horizons={result['horizons']}\n"
            f"  claim_scope={result['claim_scope']}"
        )

    @app.command(name="analyze-paired-prefix-horizon")
    def analyze_paired_prefix_horizon_command(
        config_path: Path = typer.Argument(
            ...,
            help="Three-checkpoint paired-prefix G1 YAML",
        ),
        output_path: Path = typer.Option(..., "--output-path"),
    ) -> None:
        """Audit horizon association without sampling or updating a policy."""

        from lh_platform.training.paired_prefix_horizon import (
            PairedPrefixHorizonError,
            analyze_paired_prefix_horizon_study,
        )

        try:
            result_path, svg_path = analyze_paired_prefix_horizon_study(
                config_path,
                output_path=output_path,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (PairedPrefixHorizonError, OSError, ValueError) as error:
            console.print(f"[bold red]paired-prefix G1 rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]paired-prefix G1 written: {result_path}[/bold green]\n"
            f"  chart={svg_path}\n"
            f"  supported={str(result['decision']['correlation_supported']).lower()}\n"
            f"  claim_scope={result['claim_scope']}"
        )

    @app.command(name="audit-alfworld-eval-aliasing")
    def audit_alfworld_eval_aliasing(
        full_evaluation_result: Path = typer.Argument(
            ...,
            help="completed full_evaluation_result.json",
        ),
        manifest_path: Path = typer.Option(
            ...,
            "--manifest-path",
            help="SHA-bound ALFWorld evaluation manifest",
        ),
        data_root: Path = typer.Option(
            ...,
            "--data-root",
            help="ALFWorld data root containing traj_data metadata",
        ),
        output_path: Path = typer.Option(..., "--output-path"),
    ) -> None:
        """Measure residual/frontier aliasing without changing the frozen actor."""

        from lh_platform.training.alfworld_eval_aliasing import (
            AlfWorldEvaluationAliasingError,
            analyze_alfworld_evaluation_aliasing,
        )

        try:
            result = analyze_alfworld_evaluation_aliasing(
                full_evaluation_result,
                manifest_path=manifest_path,
                data_root=data_root,
                output_path=output_path,
            )
        except (AlfWorldEvaluationAliasingError, OSError, ValueError) as error:
            console.print(
                f"[bold red]ALFWorld evaluation aliasing audit rejected: {error}[/bold red]"
            )
            raise typer.Exit(2) from error
        aliasing = result["aliasing"]
        association = result["association"]
        console.print(
            f"[bold green]ALFWorld evaluation aliasing audit written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  candidate_collision_groups="
            f"{aliasing['candidate_collision_group_count']}\n"
            f"  cross_trajectory_collision_groups="
            f"{aliasing['cross_trajectory_collision_group_count']}\n"
            f"  reference_steps_spearman="
            f"{association['reference_steps_vs_candidate_burden_spearman']}\n"
            f"  claim_scope={result['claim_scope']}"
        )
