"""Commands for post-interruption root-cause traceback experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

__all__ = ["register_traceback_commands"]


def _required_env_path(console: Console, name: str) -> Path:
    """Return the local path named by an environment variable that has no default."""

    value = os.environ.get(name, "").strip()
    if not value:
        console.print(f"[bold red]traceback command rejected: {name} is not set[/bold red]")
        raise typer.Exit(2)
    return Path(value)


def register_traceback_commands(app: typer.Typer, console: Console) -> None:
    @app.command(name="aggregate-frozen-alfworld-live-menu-constraint-population")
    def aggregate_frozen_alfworld_live_menu_constraint_population_command(
        case_root: Path = typer.Option(..., "--case-root"),
        output_path: Path = typer.Option(..., "--output-path"),
        expected_aggregation_revision: str = typer.Option(
            ...,
            "--expected-aggregation-revision",
        ),
        repo_root: Path = typer.Option(
            Path(__file__).resolve().parents[3],
            "--repo-root",
        ),
    ) -> None:
        """Aggregate the exact frozen ten-task live-menu comparison."""

        from lh_platform.tracing.alfworld_actor_control import AlfWorldActorControlError
        from lh_platform.tracing.alfworld_actor_control_population import (
            run_frozen_alfworld_live_menu_constraint_population_aggregation,
        )

        try:
            receipt = run_frozen_alfworld_live_menu_constraint_population_aggregation(
                preregistration_path=(
                    repo_root / "manifests/experiments/"
                    "qwen3_valid_unseen_live_menu_constraint_population_v1_"
                    "preregistration.json"
                ),
                pilot_manifest_path=(
                    repo_root / "manifests/qwen3_valid_unseen_gt_actor_control_pilot_v1.json"
                ),
                case_root=case_root,
                output_path=output_path,
                repo_root=repo_root,
                expected_aggregation_revision=expected_aggregation_revision,
            )
        except (AlfWorldActorControlError, OSError, subprocess.SubprocessError) as error:
            console.print(f"[bold red]live-menu aggregation rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        summary = receipt["summary"]
        console.print(
            f"[bold green]live-menu aggregation written: {output_path.resolve()}[/bold green]\n"
            f"  tasks={summary['task_count']} pairs={summary['paired_seed_count']}\n"
            f"  summary_sha256={summary['content_sha256']}"
        )

    @app.command(name="run-frozen-alfworld-live-menu-constraint-case")
    def run_frozen_alfworld_live_menu_constraint_case_command(
        candidate_id: str = typer.Option(..., "--candidate-id"),
        output_dir: Path = typer.Option(..., "--output-dir"),
        expected_platform_revision: str = typer.Option(
            ...,
            "--expected-platform-revision",
        ),
        cpu_preflight_only: bool = typer.Option(False, "--cpu-preflight-only"),
        preflight_receipt: Path | None = typer.Option(None, "--preflight-receipt"),
        repo_root: Path = typer.Option(
            Path(__file__).resolve().parents[3],
            "--repo-root",
        ),
    ) -> None:
        """Run one exact frozen live-menu population case."""

        if cpu_preflight_only != (preflight_receipt is not None):
            console.print(
                "[bold red]live-menu case rejected: CPU preflight and receipt "
                "must be requested together[/bold red]"
            )
            raise typer.Exit(2)
        workspace_root = _required_env_path(console, "LONGH_WORKSPACE_ROOT")
        gpu_lock_path = _required_env_path(console, "LONGH_GPU_LOCK_PATH")
        command = [
            sys.executable,
            str(repo_root / "lh_platform/scripts/run_alfworld_actor_control_pilot.py"),
            "--pilot-manifest",
            str(repo_root / "manifests/qwen3_valid_unseen_gt_actor_control_pilot_v1.json"),
            "--expected-pilot-manifest-sha256",
            "6ebd346a0b14a614b6ddc90d2cf0a8980ecc5a0f45fb80c5f2ea353dce83744c",
            "--task-manifest",
            str(repo_root / "manifests/alfworld_traceback_valid_unseen_exhaustive_47_v1.json"),
            "--expected-task-manifest-sha256",
            "93db7840fd8a0f07e89981068c107c1191c1a1df379132080fbcc7f8a26948a5",
            "--source-collection-result",
            str(
                workspace_root
                / "runs/traceback/qwen3_valid_unseen_exhaustive_47_source_collection_20260726_v3/source_collection_result.json"
            ),
            "--system-cell",
            str(repo_root / "manifests/system_cells/qwen3-action-only-h8-normal-v5.json"),
            "--experiment",
            str(
                repo_root
                / "configs/agentic_rl/experiments/alfworld_qwen3_4b_n1_actor_protocol_screen_v1.yaml"
            ),
            "--model-path",
            str(
                workspace_root
                / "models/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
            ),
            "--source-root",
            str(workspace_root / "external/G2PO"),
            "--data-root",
            str(workspace_root / "datasets/alfworld"),
            "--output-dir",
            str(output_dir),
            "--execution-role",
            "formal_population_case",
            "--experiment-mode",
            "live_menu_constraint",
            "--candidate-id",
            candidate_id,
            "--expected-platform-revision",
            expected_platform_revision,
            "--gpu-lock-path",
            str(gpu_lock_path),
            "--repo-root",
            str(repo_root),
        ]
        if cpu_preflight_only and preflight_receipt is not None:
            command.extend(
                [
                    "--cpu-preflight-only",
                    "--preflight-receipt",
                    str(preflight_receipt),
                ]
            )
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            console.print(f"[bold red]live-menu case rejected: {error}[/bold red]")
            raise typer.Exit(2) from error

    @app.command(name="aggregate-frozen-alfworld-actor-control-population")
    def aggregate_frozen_alfworld_actor_control_population_command(
        case_root: Path = typer.Option(..., "--case-root"),
        output_path: Path = typer.Option(..., "--output-path"),
        expected_aggregation_revision: str = typer.Option(
            ...,
            "--expected-aggregation-revision",
        ),
        repo_root: Path = typer.Option(
            Path(__file__).resolve().parents[3],
            "--repo-root",
        ),
    ) -> None:
        """Aggregate all ten frozen formal actor-control cases by task."""

        from lh_platform.tracing.alfworld_actor_control import (
            AlfWorldActorControlError,
        )
        from lh_platform.tracing.alfworld_actor_control_population import (
            run_frozen_alfworld_actor_control_population_aggregation,
        )

        try:
            receipt = run_frozen_alfworld_actor_control_population_aggregation(
                pilot_manifest_path=(
                    repo_root / "manifests/qwen3_valid_unseen_gt_actor_control_pilot_v1.json"
                ),
                case_root=case_root,
                output_path=output_path,
                repo_root=repo_root,
                expected_aggregation_revision=expected_aggregation_revision,
            )
        except (AlfWorldActorControlError, OSError, subprocess.SubprocessError) as error:
            console.print(f"[bold red]actor-control aggregation rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        summary = receipt["summary"]
        console.print(
            f"[bold green]actor-control population aggregation written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  tasks={summary['task_count']} pairs={summary['paired_seed_count']}\n"
            f"  summary_sha256={summary['content_sha256']}"
        )

    @app.command(name="run-frozen-alfworld-actor-control-case")
    def run_frozen_alfworld_actor_control_case_command(
        candidate_id: str = typer.Option(..., "--candidate-id"),
        output_dir: Path = typer.Option(..., "--output-dir"),
        expected_platform_revision: str = typer.Option(
            ...,
            "--expected-platform-revision",
        ),
        cpu_preflight_only: bool = typer.Option(
            False,
            "--cpu-preflight-only",
        ),
        preflight_receipt: Path | None = typer.Option(
            None,
            "--preflight-receipt",
        ),
        repo_root: Path = typer.Option(
            Path(__file__).resolve().parents[3],
            "--repo-root",
        ),
    ) -> None:
        """Run one case from the exact frozen formal 10-task population."""

        if cpu_preflight_only != (preflight_receipt is not None):
            console.print(
                "[bold red]actor-control case rejected: CPU preflight and receipt "
                "must be requested together[/bold red]"
            )
            raise typer.Exit(2)
        workspace_root = _required_env_path(console, "LONGH_WORKSPACE_ROOT")
        gpu_lock_path = _required_env_path(console, "LONGH_GPU_LOCK_PATH")
        command = [
            sys.executable,
            str(repo_root / "lh_platform/scripts/run_alfworld_actor_control_pilot.py"),
            "--pilot-manifest",
            str(repo_root / "manifests/qwen3_valid_unseen_gt_actor_control_pilot_v1.json"),
            "--expected-pilot-manifest-sha256",
            "6ebd346a0b14a614b6ddc90d2cf0a8980ecc5a0f45fb80c5f2ea353dce83744c",
            "--task-manifest",
            str(repo_root / "manifests/alfworld_traceback_valid_unseen_exhaustive_47_v1.json"),
            "--expected-task-manifest-sha256",
            "93db7840fd8a0f07e89981068c107c1191c1a1df379132080fbcc7f8a26948a5",
            "--source-collection-result",
            str(
                workspace_root
                / "runs/traceback/qwen3_valid_unseen_exhaustive_47_source_collection_20260726_v3/source_collection_result.json"
            ),
            "--system-cell",
            str(repo_root / "manifests/system_cells/qwen3-action-only-h8-normal-v5.json"),
            "--experiment",
            str(
                repo_root
                / "configs/agentic_rl/experiments/alfworld_qwen3_4b_n1_actor_protocol_screen_v1.yaml"
            ),
            "--model-path",
            str(
                workspace_root
                / "models/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
            ),
            "--source-root",
            str(workspace_root / "external/G2PO"),
            "--data-root",
            str(workspace_root / "datasets/alfworld"),
            "--output-dir",
            str(output_dir),
            "--execution-role",
            "formal_population_case",
            "--candidate-id",
            candidate_id,
            "--expected-platform-revision",
            expected_platform_revision,
            "--gpu-lock-path",
            str(gpu_lock_path),
            "--repo-root",
            str(repo_root),
        ]
        if cpu_preflight_only and preflight_receipt is not None:
            command.extend(
                [
                    "--cpu-preflight-only",
                    "--preflight-receipt",
                    str(preflight_receipt),
                ]
            )
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            console.print(f"[bold red]actor-control case rejected: {error}[/bold red]")
            raise typer.Exit(2) from error

    @app.command(name="audit-alfworld-causal-scope")
    def audit_alfworld_causal_scope_command(
        causal_evaluation_manifest: Path = typer.Option(
            ...,
            "--causal-evaluation-manifest",
            help="frozen valid-unseen causal evaluation manifest",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="new immutable scope-audit receipt JSON path",
        ),
        expected_platform_revision: str = typer.Option(
            ...,
            "--expected-platform-revision",
        ),
        repo_root: Path = typer.Option(
            Path(__file__).resolve().parents[3],
            "--repo-root",
        ),
    ) -> None:
        """Audit the fixed-controller boundary of the frozen 57 causal cases."""

        from lh_platform.tracing.alfworld_causal_scope_audit import (
            AlfWorldCausalScopeAuditError,
            run_frozen_valid_unseen_causal_scope_audit,
        )

        try:
            receipt = run_frozen_valid_unseen_causal_scope_audit(
                causal_evaluation_manifest_path=causal_evaluation_manifest,
                output_path=output_path,
                repo_root=repo_root,
                expected_platform_revision=expected_platform_revision,
            )
        except (AlfWorldCausalScopeAuditError, OSError, subprocess.SubprocessError) as error:
            console.print(f"[bold red]causal scope audit rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        audit = receipt["scope_audit"]
        console.print(
            f"[bold green]causal scope audit written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  cases={audit['case_count']} tasks={audit['task_count']}\n"
            f"  on_policy_actor_claim_eligible="
            f"{audit['on_policy_actor_claim_eligible']}\n"
            f"  audit_sha256={audit['content_sha256']}"
        )

    @app.command(name="export-trace-benchmark-repeat-audit")
    def export_trace_benchmark_repeat_audit_command(
        result_path: list[Path] = typer.Argument(
            ...,
            help="two same-spec results for each of at least two models",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="new prompt-free repeat audit JSON path",
        ),
        expected_model: list[str] = typer.Option(
            [],
            "--expected-model",
            help="frozen model id; repeat once per expected model",
        ),
        expected_method: list[str] = typer.Option(
            [],
            "--expected-method",
            help="frozen method id; repeat for every expected method",
        ),
        expected_execution: list[str] = typer.Option(
            [],
            "--expected-execution",
            help="authorized MODEL_ID=EXECUTION_ID pair; repeat for all executions",
        ),
        manifest_path: Path = typer.Option(
            ...,
            "--manifest-path",
        ),
        corpus_root: Path = typer.Option(..., "--corpus-root"),
        expected_manifest_sha256: str = typer.Option(..., "--expected-manifest-sha256"),
        expected_platform_revision: str = typer.Option(..., "--expected-platform-revision"),
        expected_method_descriptors_sha256: str = typer.Option(
            ..., "--expected-method-descriptors-sha256"
        ),
        expected_max_tokens: int = typer.Option(..., "--expected-max-tokens", min=1),
        expected_timeout_seconds: int = typer.Option(..., "--expected-timeout-seconds", min=1),
        expected_temperature: float = typer.Option(..., "--expected-temperature", min=0.0, max=2.0),
        expected_generation_seed: int = typer.Option(..., "--expected-generation-seed", min=0),
        expected_python_environment_sha256: str = typer.Option(
            ..., "--expected-python-environment-sha256"
        ),
    ) -> None:
        """Verify repeated matrices and publish cell-level stability evidence."""

        from lh_platform.tracing.benchmark_inventory import (
            frozen_trace_benchmark_case_identities,
        )
        from lh_platform.tracing.benchmark_repeat_audit import (
            build_trace_benchmark_repeat_audit,
            write_trace_benchmark_repeat_audit,
        )
        from lh_platform.tracing.unified_benchmark import load_trace_benchmark_manifest

        try:
            expected_execution_ids: dict[str, list[str]] = {}
            for binding in expected_execution:
                bound_model, separator, bound_execution = binding.partition("=")
                if not separator or not bound_model or not bound_execution:
                    raise ValueError("expected execution must use MODEL_ID=EXECUTION_ID")
                expected_execution_ids.setdefault(bound_model, []).append(bound_execution)
            dataset = load_trace_benchmark_manifest(
                manifest_path,
                corpus_root=corpus_root,
                expected_manifest_sha256=expected_manifest_sha256,
            )
            audit = build_trace_benchmark_repeat_audit(
                result_path,
                expected_model_ids=tuple(expected_model),
                expected_method_ids=tuple(expected_method),
                expected_execution_ids={
                    model_id: tuple(execution_ids)
                    for model_id, execution_ids in expected_execution_ids.items()
                },
                expected_dataset_manifest_sha256=expected_manifest_sha256,
                expected_cases=frozen_trace_benchmark_case_identities(dataset),
                expected_platform_revision=expected_platform_revision,
                expected_method_descriptors_sha256=(expected_method_descriptors_sha256),
                expected_max_tokens=expected_max_tokens,
                expected_timeout_seconds=expected_timeout_seconds,
                expected_temperature=expected_temperature,
                expected_generation_seed=expected_generation_seed,
                expected_python_environment_sha256=(expected_python_environment_sha256),
            )
            write_trace_benchmark_repeat_audit(audit, output_path)
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]trace repeat audit rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]trace repeat audit written: {output_path.resolve()}[/bold green]\n"
            f"  models={len(audit['models'])}\n"
            f"  repeat_audit_sha256={audit['repeat_audit_sha256']}"
        )

    @app.command(name="export-trace-benchmark-inventory")
    def export_trace_benchmark_inventory_command(
        result_path: list[Path] = typer.Argument(
            ...,
            help="two or more immutable trace benchmark result JSON files",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="new prompt-free case inventory JSON path",
        ),
        manifest_path: Path = typer.Option(..., "--manifest-path"),
        corpus_root: Path = typer.Option(..., "--corpus-root"),
        expected_manifest_sha256: str = typer.Option(..., "--expected-manifest-sha256"),
        expected_model: list[str] = typer.Option([], "--expected-model"),
        expected_method: list[str] = typer.Option([], "--expected-method"),
    ) -> None:
        """Verify result hashes and publish a branch-auditable case inventory."""

        from lh_platform.tracing.benchmark_inventory import (
            build_trace_benchmark_case_inventory,
            frozen_trace_benchmark_case_identities,
            write_trace_benchmark_case_inventory,
        )
        from lh_platform.tracing.unified_benchmark import load_trace_benchmark_manifest

        try:
            dataset = load_trace_benchmark_manifest(
                manifest_path,
                corpus_root=corpus_root,
                expected_manifest_sha256=expected_manifest_sha256,
            )
            inventory = build_trace_benchmark_case_inventory(
                result_path,
                expected_dataset_manifest_sha256=expected_manifest_sha256,
                expected_model_ids=tuple(expected_model),
                expected_method_ids=tuple(expected_method),
                expected_cases=frozen_trace_benchmark_case_identities(dataset),
            )
            write_trace_benchmark_case_inventory(inventory, output_path)
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]trace inventory rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]trace inventory written: {output_path.resolve()}[/bold green]\n"
            f"  cells={len(inventory['cells'])}\n"
            f"  inventory_sha256={inventory['inventory_sha256']}"
        )

    @app.command(name="list-trace-methods")
    def list_trace_methods_command() -> None:
        """List the complete supplied tracer corpus through one descriptor schema."""

        from lh_platform.tracing.integrations import build_catalog_trace_method_platform

        methods = build_catalog_trace_method_platform().catalog()
        payload = {
            "schema": "lh_trace_method_catalog/v1",
            "corpus_method_count": len({item.corpus_method_id for item in methods}),
            "executable_method_count": sum(
                item.integration_level != "catalog_only" for item in methods
            ),
            "methods": [item.to_dict() for item in methods],
        }
        console.print_json(data=payload)

    @app.command(name="audit-trace-method-coverage")
    def audit_trace_method_coverage_command(
        corpus_root: Path = typer.Argument(
            ...,
            help="root of the supplied agent_tracer corpus",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="new immutable native coverage JSON path",
        ),
        python_executable: str = typer.Option(
            sys.executable,
            "--python",
            help="Python executable used for dependency admission",
        ),
    ) -> None:
        """Separate native admission from source, catalog and paper-only coverage."""

        from lh_platform.tracing.integrations import build_native_trace_method_platform
        from lh_platform.tracing.native_coverage import (
            build_native_trace_coverage,
            write_native_trace_coverage,
        )

        try:
            platform = build_native_trace_method_platform(
                corpus_root=corpus_root,
                model_id="coverage-audit-no-inference",
                base_url="http://127.0.0.1:1/v1",
                python_executable=python_executable,
            )
            report = build_native_trace_coverage(
                corpus_root=corpus_root,
                platform=platform,
            )
            write_native_trace_coverage(report, output_path)
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]trace coverage audit rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]trace coverage audit written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  method_ids={report.method_count}\n"
            f"  corpus_methods={report.corpus_method_count}\n"
            f"  status_counts={dict(report.status_counts)}\n"
            f"  coverage_sha256={report.coverage_sha256}"
        )

    @app.command(name="prepare-critictool-trace-benchmark")
    def prepare_critictool_trace_benchmark_command(
        dataset_root: Path = typer.Argument(
            ...,
            help="clean checkout of the official CriticTool dataset",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="new immutable benchmark manifest JSON path",
        ),
        dataset_id: str = typer.Option(..., "--dataset-id"),
        seed: str = typer.Option(..., "--seed"),
        per_error_type_count: int = typer.Option(
            3,
            "--per-error-type-count",
            min=1,
        ),
        excluded_source_key: list[str] = typer.Option(
            [],
            "--excluded-source-key",
            help="source key already inspected during development; repeatable",
        ),
    ) -> None:
        """Freeze a stratified official CriticTool subset before model execution."""

        from lh_platform.tracing.critictool_benchmark import (
            prepare_critictool_trace_benchmark_manifest,
        )

        try:
            manifest = prepare_critictool_trace_benchmark_manifest(
                dataset_root=dataset_root,
                output_path=output_path,
                dataset_id=dataset_id,
                seed=seed,
                per_error_type_count=per_error_type_count,
                excluded_source_keys=tuple(excluded_source_key),
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]CriticTool manifest rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]CriticTool manifest written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  cases={len(manifest['cases'])}\n"
            f"  source_revision={manifest['source_revision']}\n"
            f"  manifest_sha256={manifest['manifest_sha256']}"
        )

    def _critictool_preflight(
        *,
        manifest_path: Path,
        dataset_root: Path,
        corpus_root: Path,
        model: str,
        base_url: str,
        expected_manifest_sha256: str,
        api_key: str,
        python_executable: str,
        max_tokens: int,
        timeout_seconds: int,
        temperature: float,
        generation_seed: int,
        diagnostic_artifact_root: Path | None = None,
    ) -> tuple[object, object, object, object, str, dict[str, object]]:
        from lh_platform.tracing.critictool_benchmark import (
            CRITICTOOL_FROZEN_GENERATION_SEED,
            CRITICTOOL_FROZEN_MANIFEST_SHA256,
            CRITICTOOL_FROZEN_MANIFEST_SHA256S,
            CRITICTOOL_FROZEN_MAX_TOKENS,
            CRITICTOOL_FROZEN_MODEL_IDS,
            CRITICTOOL_FROZEN_TEMPERATURE,
            CRITICTOOL_FROZEN_TIMEOUT_SECONDS,
            CRITICTOOL_V2_FROZEN_MANIFEST_SHA256,
            load_critictool_trace_benchmark_manifest,
            validate_critictool_formal_method_descriptor,
        )
        from lh_platform.tracing.integrations import build_native_trace_method_platform
        from lh_platform.tracing.integrations._provenance import (
            require_clean_source_revision,
        )
        from lh_platform.tracing.integrations.critictool import (
            CRITICTOOL_PROMPT_PROFILE_COT,
            CRITICTOOL_PROMPT_PROFILE_NO_COT,
        )
        from lh_platform.tracing.unified_benchmark import (
            attest_openai_model_endpoint,
            build_trace_benchmark_run_spec,
        )

        if expected_manifest_sha256 not in CRITICTOOL_FROZEN_MANIFEST_SHA256S:
            raise ValueError("CriticTool formal manifest is not a frozen protocol manifest")
        if model not in CRITICTOOL_FROZEN_MODEL_IDS:
            raise ValueError("CriticTool formal model is not in the frozen v1 pair")
        observed_settings = (
            max_tokens,
            timeout_seconds,
            temperature,
            generation_seed,
        )
        frozen_settings = (
            CRITICTOOL_FROZEN_MAX_TOKENS,
            CRITICTOOL_FROZEN_TIMEOUT_SECONDS,
            CRITICTOOL_FROZEN_TEMPERATURE,
            CRITICTOOL_FROZEN_GENERATION_SEED,
        )
        if observed_settings != frozen_settings:
            raise ValueError("CriticTool formal settings differ from frozen v1")
        dataset = load_critictool_trace_benchmark_manifest(
            manifest_path,
            dataset_root=dataset_root,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        platform_revision = require_clean_source_revision(
            Path(__file__).resolve().parents[3],
            method_name="CriticTool benchmark platform",
        )
        runtime_attestation = attest_openai_model_endpoint(
            base_url=base_url,
            api_key=api_key,
            requested_model_id=model,
            timeout_seconds=timeout_seconds,
        )
        prompt_profile = (
            CRITICTOOL_PROMPT_PROFILE_COT
            if expected_manifest_sha256
            in {
                CRITICTOOL_FROZEN_MANIFEST_SHA256,
                CRITICTOOL_V2_FROZEN_MANIFEST_SHA256,
            }
            else CRITICTOOL_PROMPT_PROFILE_NO_COT
        )
        platform = build_native_trace_method_platform(
            corpus_root=corpus_root,
            model_id=model,
            base_url=base_url,
            api_key=api_key,
            python_executable=python_executable,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            generation_seed=generation_seed,
            diagnostic_artifact_root=diagnostic_artifact_root,
            critictool_prompt_profile=prompt_profile,
        )
        descriptors = {descriptor.method_id: descriptor for descriptor in platform.catalog()}
        descriptor = descriptors.get("critictool")
        if descriptor is None:
            raise ValueError("CriticTool native method is not admitted")
        method_descriptor = descriptor.to_dict()
        validate_critictool_formal_method_descriptor(
            method_descriptor,
            dataset_manifest_sha256=dataset.manifest_sha256,
        )
        run_spec = build_trace_benchmark_run_spec(
            dataset_manifest_sha256=dataset.manifest_sha256,
            model_id=model,
            method_ids=("critictool",),
            method_descriptors=(method_descriptor,),
            platform_revision=platform_revision,
            base_url=base_url,
            python_executable=python_executable,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            generation_seed=generation_seed,
        )
        return (
            dataset,
            platform,
            runtime_attestation,
            run_spec,
            platform_revision,
            method_descriptor,
        )

    @app.command(name="authorize-critictool-trace-benchmark")
    def authorize_critictool_trace_benchmark_command(
        manifest_path: Path = typer.Argument(
            ...,
            help="frozen lh_critictool_trace_benchmark_manifest/v1 JSON",
        ),
        dataset_root: Path = typer.Option(..., "--dataset-root"),
        corpus_root: Path = typer.Option(..., "--corpus-root"),
        model: str = typer.Option(..., "--model"),
        execution_id: str = typer.Option(..., "--execution-id"),
        base_url: str = typer.Option(..., "--base-url"),
        output_path: Path = typer.Option(..., "--output-path"),
        expected_manifest_sha256: str = typer.Option(
            ...,
            "--expected-manifest-sha256",
        ),
        api_key_env: str = typer.Option("OPENAI_API_KEY", "--api-key-env"),
        python_executable: str = typer.Option(sys.executable, "--python"),
        max_tokens: int = typer.Option(512, "--max-tokens", min=1),
        timeout_seconds: int = typer.Option(180, "--timeout-seconds", min=1),
        temperature: float = typer.Option(0.0, "--temperature", min=0.0, max=2.0),
        generation_seed: int = typer.Option(0, "--generation-seed", min=0),
    ) -> None:
        """Create a pre-execution authorization; this command runs no Cell."""

        from lh_platform.tracing.critictool_benchmark import (
            build_critictool_trace_benchmark_authorization,
            write_critictool_trace_benchmark_authorization,
        )

        if output_path.exists():
            console.print(
                f"[bold red]CriticTool authorization rejected: output already exists: "
                f"{output_path}[/bold red]"
            )
            raise typer.Exit(2)
        try:
            api_key = os.environ.get(api_key_env, "").strip() or "EMPTY"
            (
                dataset,
                _platform,
                runtime_attestation,
                run_spec,
                platform_revision,
                method_descriptor,
            ) = _critictool_preflight(
                manifest_path=manifest_path,
                dataset_root=dataset_root,
                corpus_root=corpus_root,
                model=model,
                base_url=base_url,
                expected_manifest_sha256=expected_manifest_sha256,
                api_key=api_key,
                python_executable=python_executable,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                generation_seed=generation_seed,
                diagnostic_artifact_root=None,
            )
            authorization = build_critictool_trace_benchmark_authorization(
                dataset=dataset,
                model_id=model,
                execution_id=execution_id,
                platform_revision=platform_revision,
                method_descriptor=method_descriptor,
                runtime_attestation=runtime_attestation,
                run_spec=run_spec,
            )
            write_critictool_trace_benchmark_authorization(
                authorization,
                output_path,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]CriticTool authorization rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]CriticTool authorization written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  execution_id={authorization.execution_id}\n"
            f"  run_spec_sha256={authorization.run_spec.run_spec_sha256}\n"
            f"  authorization_sha256={authorization.authorization_sha256}"
        )

    @app.command(name="run-critictool-trace-benchmark")
    def run_critictool_trace_benchmark_command(
        manifest_path: Path = typer.Argument(
            ...,
            help="frozen lh_critictool_trace_benchmark_manifest/v1 JSON",
        ),
        dataset_root: Path = typer.Option(..., "--dataset-root"),
        corpus_root: Path = typer.Option(..., "--corpus-root"),
        authorization_path: Path = typer.Option(..., "--authorization-path"),
        model: str = typer.Option(..., "--model"),
        execution_id: str = typer.Option(..., "--execution-id"),
        base_url: str = typer.Option(..., "--base-url"),
        output_path: Path = typer.Option(..., "--output-path"),
        expected_manifest_sha256: str = typer.Option(
            ...,
            "--expected-manifest-sha256",
        ),
        api_key_env: str = typer.Option("OPENAI_API_KEY", "--api-key-env"),
        python_executable: str = typer.Option(sys.executable, "--python"),
        max_tokens: int = typer.Option(512, "--max-tokens", min=1),
        timeout_seconds: int = typer.Option(180, "--timeout-seconds", min=1),
        temperature: float = typer.Option(0.0, "--temperature", min=0.0, max=2.0),
        generation_seed: int = typer.Option(0, "--generation-seed", min=0),
    ) -> None:
        """Run kind-specific CriticTool scoring through the unified method seam."""

        from lh_platform.tracing.critictool_benchmark import (
            load_critictool_trace_benchmark_authorization,
            run_critictool_trace_benchmark,
            write_critictool_trace_benchmark_result,
        )

        if output_path.exists():
            console.print(
                f"[bold red]CriticTool benchmark rejected: output already exists: "
                f"{output_path}[/bold red]"
            )
            raise typer.Exit(2)
        diagnostic_artifact_root = output_path.with_name(
            f"{output_path.stem}-diagnostics"
        ).resolve()
        if diagnostic_artifact_root.exists():
            console.print(
                f"[bold red]CriticTool benchmark rejected: diagnostic artifact root "
                f"already exists: {diagnostic_artifact_root}[/bold red]"
            )
            raise typer.Exit(2)
        try:
            api_key = os.environ.get(api_key_env, "").strip() or "EMPTY"
            authorization = load_critictool_trace_benchmark_authorization(authorization_path)
            if authorization.model_id != model or authorization.execution_id != execution_id:
                raise ValueError("CriticTool CLI identity does not match authorization")
            (
                dataset,
                platform,
                runtime_attestation,
                run_spec,
                _platform_revision,
                _method_descriptor,
            ) = _critictool_preflight(
                manifest_path=manifest_path,
                dataset_root=dataset_root,
                corpus_root=corpus_root,
                model=model,
                base_url=base_url,
                expected_manifest_sha256=expected_manifest_sha256,
                api_key=api_key,
                python_executable=python_executable,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                generation_seed=generation_seed,
                diagnostic_artifact_root=diagnostic_artifact_root,
            )

            def record_progress(cell: object, completed: int, total: int) -> None:
                console.print(
                    f"  [{completed}/{total}] {cell.case_id} {cell.method_id} status={cell.status}"
                )

            result = run_critictool_trace_benchmark(
                platform=platform,
                dataset=dataset,
                authorization=authorization,
                runtime_attestation=runtime_attestation,
                run_spec=run_spec,
                progress_callback=record_progress,
            )
            write_critictool_trace_benchmark_result(result, output_path)
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]CriticTool benchmark rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]CriticTool benchmark written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  dataset_id={result.dataset_id}\n"
            f"  model_id={result.model_id}\n"
            f"  cells={len(result.cells)}\n"
            f"  run_spec_sha256={result.run_spec.run_spec_sha256}\n"
            f"  result_sha256={result.result_sha256}"
        )

    @app.command(name="export-critictool-trace-benchmark-gate-audit")
    def export_critictool_trace_benchmark_gate_audit_command(
        result_path: list[Path] = typer.Argument(
            ...,
            help="one immutable CriticTool result per frozen model",
        ),
        authorization_path: list[Path] = typer.Option(
            ...,
            "--authorization-path",
            help="matching immutable authorization; repeat for each model",
        ),
        manifest_path: Path = typer.Option(..., "--manifest-path"),
        dataset_root: Path = typer.Option(..., "--dataset-root"),
        expected_manifest_sha256: str = typer.Option(
            ...,
            "--expected-manifest-sha256",
        ),
        expected_model: list[str] = typer.Option(
            ...,
            "--expected-model",
            help="revision-qualified frozen model id; repeat for each model",
        ),
        expected_platform_revision: str = typer.Option(
            ...,
            "--expected-platform-revision",
        ),
        output_path: Path = typer.Option(..., "--output-path"),
    ) -> None:
        """Recompute the two-model engineering merge gate from immutable results."""

        from lh_platform.tracing.critictool_benchmark import (
            CRITICTOOL_FROZEN_MANIFEST_SHA256S,
            CRITICTOOL_FROZEN_MINIMUM_COMPLETED_PER_MODEL,
            CRITICTOOL_FROZEN_MODEL_IDS,
            build_critictool_trace_benchmark_gate_audit,
            load_critictool_trace_benchmark_manifest,
            write_critictool_trace_benchmark_gate_audit,
        )

        if output_path.exists():
            console.print(
                f"[bold red]CriticTool gate audit rejected: output already exists: "
                f"{output_path}[/bold red]"
            )
            raise typer.Exit(2)
        try:
            if expected_manifest_sha256 not in CRITICTOOL_FROZEN_MANIFEST_SHA256S:
                raise ValueError("CriticTool gate manifest is not a frozen protocol manifest")
            if set(expected_model) != set(CRITICTOOL_FROZEN_MODEL_IDS):
                raise ValueError("CriticTool gate models differ from frozen v1")
            dataset = load_critictool_trace_benchmark_manifest(
                manifest_path,
                dataset_root=dataset_root,
                expected_manifest_sha256=expected_manifest_sha256,
            )
            audit = build_critictool_trace_benchmark_gate_audit(
                result_path,
                authorization_paths=authorization_path,
                dataset=dataset,
                expected_model_ids=CRITICTOOL_FROZEN_MODEL_IDS,
                expected_platform_revision=expected_platform_revision,
                minimum_completed_per_model=(CRITICTOOL_FROZEN_MINIMUM_COMPLETED_PER_MODEL),
            )
            write_critictool_trace_benchmark_gate_audit(audit, output_path)
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]CriticTool gate audit rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]CriticTool gate audit written: "
            f"{output_path.resolve()}[/bold green]\n"
            f"  completed_cells={audit['completed_cell_count']}/"
            f"{audit['expected_cell_count']}\n"
            f"  gate_passed={audit['gate_passed']}\n"
            f"  gate_audit_sha256={audit['gate_audit_sha256']}"
        )
        if not audit["gate_passed"]:
            raise typer.Exit(1)

    @app.command(name="run-trace-method")
    def run_trace_method_command(
        method_id: str = typer.Argument(..., help="unified trace method id"),
        request_path: Path = typer.Argument(
            ...,
            help=("lh_trace_case/v1, lh_trace_episode/v1, or lh_trace_method_request/v1 JSON"),
        ),
        corpus_root: Path = typer.Option(
            ...,
            "--corpus-root",
            help="root of the supplied agent_tracer corpus",
        ),
        model: str = typer.Option(..., "--model", help="served diagnostician model id"),
        base_url: str = typer.Option(
            ...,
            "--base-url",
            help="OpenAI-compatible base URL ending in /v1",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="new immutable unified receipt JSON path",
        ),
        api_key_env: str = typer.Option("OPENAI_API_KEY", "--api-key-env"),
        python_executable: str = typer.Option(
            sys.executable,
            "--python",
            help="Python executable used by isolated upstream workers",
        ),
        max_tokens: int = typer.Option(512, "--max-tokens", min=1),
        timeout_seconds: int = typer.Option(180, "--timeout-seconds", min=1),
        temperature: float = typer.Option(0.0, "--temperature", min=0.0, max=2.0),
        generation_seed: int = typer.Option(0, "--generation-seed", min=0),
        critictool_prompt_profile: str = typer.Option(
            "llm_no_cot",
            "--critictool-prompt-profile",
            help="CriticTool upstream profile: llm_no_cot or llm_cot",
        ),
    ) -> None:
        """Run one admitted native tracer and emit a hash-bound normalized receipt."""

        from lh_platform.tracing.integrations import build_native_trace_method_platform
        from lh_platform.tracing.types import (
            TRACE_CASE_SCHEMA,
            TRACE_EPISODE_SCHEMA,
            trace_input_from_dict,
        )
        from lh_platform.tracing.unified import (
            TRACE_METHOD_REQUEST_SCHEMA,
            TraceMethodRequestV1,
            TraceRunStatus,
        )

        if output_path.exists():
            console.print(
                f"[bold red]trace method rejected: output already exists: {output_path}[/bold red]"
            )
            raise typer.Exit(2)
        try:
            raw = json.loads(request_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("trace request must be a JSON object")
            schema = raw.get("schema")
            if schema in {TRACE_CASE_SCHEMA, TRACE_EPISODE_SCHEMA}:
                request = TraceMethodRequestV1(case=trace_input_from_dict(raw))
            elif schema == TRACE_METHOD_REQUEST_SCHEMA:
                request = TraceMethodRequestV1.from_dict(raw)
            else:
                raise ValueError(f"unsupported trace request schema: {schema}")
            platform = build_native_trace_method_platform(
                corpus_root=corpus_root,
                model_id=model,
                base_url=base_url,
                api_key=os.environ.get(api_key_env, "").strip() or "EMPTY",
                python_executable=python_executable,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                generation_seed=generation_seed,
                critictool_prompt_profile=critictool_prompt_profile,
            )
            receipt = platform.run(method_id, request)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    receipt.to_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]trace method rejected: {error}[/bold red]")
            raise typer.Exit(2) from error

        color = "green" if receipt.status is TraceRunStatus.COMPLETED else "yellow"
        console.print(
            f"[bold {color}]trace method receipt written: {output_path.resolve()}[/bold {color}]\n"
            f"  method_id={receipt.method_id}\n"
            f"  status={receipt.status.value}\n"
            f"  findings={len(receipt.findings)}\n"
            f"  blockers={','.join(receipt.blockers) or 'none'}\n"
            f"  receipt_sha256={receipt.receipt_sha256}"
        )
        if receipt.status is TraceRunStatus.BLOCKED:
            raise typer.Exit(3)
        if receipt.status is TraceRunStatus.INVALID:
            raise typer.Exit(4)

    @app.command(name="run-trace-benchmark")
    def run_trace_benchmark_command(
        manifest_path: Path = typer.Argument(
            ...,
            help="frozen lh_trace_benchmark_manifest/v1 JSON",
        ),
        corpus_root: Path = typer.Option(
            ...,
            "--corpus-root",
            help="root of the supplied agent_tracer corpus",
        ),
        model: str = typer.Option(..., "--model", help="served diagnostician model id"),
        execution_id: str = typer.Option(
            ...,
            "--execution-id",
            help="unique immutable identifier for this independent execution",
        ),
        base_url: str = typer.Option(
            ...,
            "--base-url",
            help="OpenAI-compatible base URL ending in /v1",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="new immutable benchmark result JSON path",
        ),
        expected_manifest_sha256: str = typer.Option(
            ...,
            "--expected-manifest-sha256",
            help="externally preregistered canonical manifest hash",
        ),
        checkpoint_path: Path | None = typer.Option(
            None,
            "--checkpoint-path",
            help="durable resumable checkpoint; defaults beside output",
        ),
        method: list[str] = typer.Option(
            [],
            "--method",
            help="unified method id; repeatable; defaults to all text-native methods",
        ),
        api_key_env: str = typer.Option("OPENAI_API_KEY", "--api-key-env"),
        python_executable: str = typer.Option(
            sys.executable,
            "--python",
            help="Python executable used by isolated upstream workers",
        ),
        max_tokens: int = typer.Option(512, "--max-tokens", min=1),
        timeout_seconds: int = typer.Option(180, "--timeout-seconds", min=1),
        temperature: float = typer.Option(0.0, "--temperature", min=0.0, max=2.0),
        generation_seed: int = typer.Option(0, "--generation-seed", min=0),
    ) -> None:
        """Run a frozen, gold-blind tracer matrix with kind-specific scoring."""

        from lh_platform.tracing.integrations import build_native_trace_method_platform
        from lh_platform.tracing.integrations._provenance import (
            require_clean_source_revision,
        )
        from lh_platform.tracing.unified_benchmark import (
            DEFAULT_TEXT_TRACE_METHOD_IDS,
            TraceBenchmarkCellV1,
            TraceBenchmarkCheckpointV1,
            attest_openai_model_endpoint,
            build_trace_benchmark_run_spec,
            load_trace_benchmark_checkpoint,
            load_trace_benchmark_manifest,
            run_trace_benchmark,
            write_trace_benchmark_checkpoint,
            write_trace_benchmark_result,
        )

        if output_path.exists():
            console.print(
                f"[bold red]trace benchmark rejected: output already exists: "
                f"{output_path}[/bold red]"
            )
            raise typer.Exit(2)
        method_ids = tuple(method) or DEFAULT_TEXT_TRACE_METHOD_IDS
        resolved_checkpoint = checkpoint_path or output_path.with_name(
            output_path.name + ".checkpoint.json"
        )
        if resolved_checkpoint.resolve() == output_path.resolve():
            console.print("[bold red]trace benchmark rejected: checkpoint equals output[/bold red]")
            raise typer.Exit(2)
        try:
            api_key = os.environ.get(api_key_env, "").strip() or "EMPTY"
            dataset = load_trace_benchmark_manifest(
                manifest_path,
                corpus_root=corpus_root,
                expected_manifest_sha256=expected_manifest_sha256,
            )
            platform_revision = require_clean_source_revision(
                Path(__file__).resolve().parents[3],
                method_name="trace benchmark platform",
            )
            runtime_attestation = attest_openai_model_endpoint(
                base_url=base_url,
                api_key=api_key,
                requested_model_id=model,
                timeout_seconds=timeout_seconds,
            )
            platform = build_native_trace_method_platform(
                corpus_root=corpus_root,
                model_id=model,
                base_url=base_url,
                api_key=api_key,
                python_executable=python_executable,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                generation_seed=generation_seed,
            )
            descriptors_by_id = {
                descriptor.method_id: descriptor for descriptor in platform.catalog()
            }
            unknown_method_ids = set(method_ids) - descriptors_by_id.keys()
            if unknown_method_ids:
                raise ValueError(
                    f"trace benchmark contains unknown methods: {sorted(unknown_method_ids)}"
                )
            run_spec = build_trace_benchmark_run_spec(
                dataset_manifest_sha256=dataset.manifest_sha256,
                model_id=model,
                method_ids=method_ids,
                method_descriptors=tuple(
                    descriptors_by_id[method_id].to_dict() for method_id in method_ids
                ),
                platform_revision=platform_revision,
                base_url=base_url,
                python_executable=python_executable,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                generation_seed=generation_seed,
            )
            if resolved_checkpoint.exists():
                checkpoint = load_trace_benchmark_checkpoint(
                    resolved_checkpoint,
                    dataset_manifest_sha256=dataset.manifest_sha256,
                    model_id=model,
                    execution_id=execution_id,
                    method_ids=method_ids,
                    platform_revision=platform_revision,
                    runtime_attestation=runtime_attestation,
                    run_spec=run_spec,
                )
                completed_cells = list(checkpoint.cells)
                console.print(
                    f"  resuming {len(completed_cells)}/"
                    f"{len(dataset.cases) * len(method_ids)} cells"
                )
            else:
                completed_cells = []
                write_trace_benchmark_checkpoint(
                    TraceBenchmarkCheckpointV1(
                        dataset_manifest_sha256=dataset.manifest_sha256,
                        model_id=model,
                        execution_id=execution_id,
                        method_ids=method_ids,
                        platform_revision=platform_revision,
                        runtime_attestation=runtime_attestation,
                        run_spec=run_spec,
                        cells=(),
                    ),
                    resolved_checkpoint,
                    overwrite=False,
                )

            def record_progress(
                cell: TraceBenchmarkCellV1,
                completed: int,
                total: int,
            ) -> None:
                completed_cells.append(cell)
                write_trace_benchmark_checkpoint(
                    TraceBenchmarkCheckpointV1(
                        dataset_manifest_sha256=dataset.manifest_sha256,
                        model_id=model,
                        execution_id=execution_id,
                        method_ids=method_ids,
                        platform_revision=platform_revision,
                        runtime_attestation=runtime_attestation,
                        run_spec=run_spec,
                        cells=tuple(completed_cells),
                    ),
                    resolved_checkpoint,
                    overwrite=True,
                )
                console.print(
                    f"  [{completed}/{total}] {cell.case_id} {cell.method_id} status={cell.status}"
                )

            result = run_trace_benchmark(
                platform=platform,
                dataset=dataset,
                model_id=model,
                execution_id=execution_id,
                method_ids=method_ids,
                platform_revision=platform_revision,
                runtime_attestation=runtime_attestation,
                run_spec=run_spec,
                resume_cells=tuple(completed_cells),
                progress_callback=record_progress,
            )
            write_trace_benchmark_result(result, output_path)
            resolved_checkpoint.unlink(missing_ok=True)
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]trace benchmark rejected: {error}[/bold red]")
            raise typer.Exit(2) from error

        console.print(
            f"[bold green]trace benchmark written: {output_path.resolve()}[/bold green]\n"
            f"  dataset_id={result.dataset_id}\n"
            f"  model_id={result.model_id}\n"
            f"  cells={len(result.cells)}\n"
            f"  run_spec_sha256={result.run_spec.run_spec_sha256}\n"
            f"  result_sha256={result.result_sha256}"
        )
        for aggregate in result.aggregates:
            console.print(
                f"  {aggregate.method_id}: completed={aggregate.completed_count}/"
                f"{aggregate.case_count}, step_exact={aggregate.step_exact_count}/"
                f"{aggregate.step_evaluable_count}, component_exact="
                f"{aggregate.component_exact_count}/{aggregate.component_evaluable_count}"
            )

    @app.command(name="run-controlled-traceback-p0")
    def run_controlled_traceback_p0_command(
        model: str = typer.Option(..., "--model", help="served diagnostician model id"),
        base_url: str = typer.Option(
            ...,
            "--base-url",
            help="OpenAI-compatible base URL ending in /v1",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="new immutable P0 result JSON path",
        ),
        api_key_env: str = typer.Option(
            "OPENAI_API_KEY",
            "--api-key-env",
            help="environment variable containing the endpoint API key",
        ),
        seed: int = typer.Option(0, "--seed"),
    ) -> None:
        """Compare static localization with intervention-verified replay."""

        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            console.print(f"[bold red]traceback P0 rejected: {api_key_env} is not set[/bold red]")
            raise typer.Exit(2)

        from lh_platform.models.openai_chat import OpenAIChatModel
        from lh_platform.tracing import write_controlled_traceback_p0

        diagnostician = OpenAIChatModel(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=1024,
            seed=seed,
            allow_dummy_fallback=False,
        )
        try:
            result = write_controlled_traceback_p0(
                model=diagnostician,
                output_path=output_path,
            )
        except (FileExistsError, OSError, ValueError) as error:
            console.print(f"[bold red]traceback P0 rejected: {error}[/bold red]")
            raise typer.Exit(2) from error

        replay = next(
            method for method in result.methods if method.tracer_id == "counterfactual_replay"
        )
        color = "green" if result.effect_gate_passed else "yellow"
        console.print(
            f"[bold {color}]traceback P0 written: {output_path.resolve()}[/bold {color}]\n"
            f"  exact_localization_rate={replay.exact_localization_rate:.3f}\n"
            "  intervention_verified_root_cause_rate="
            f"{replay.intervention_verified_root_cause_rate:.3f}\n"
            f"  invalid_output_count={replay.invalid_output_count}\n"
            f"  effect_gate_passed={result.effect_gate_passed}\n"
            f"  blocked_by={','.join(result.blocked_by) or 'none'}\n"
            f"  claim_scope={result.claim_scope}"
        )
        if not result.effect_gate_passed:
            raise typer.Exit(3)

    @app.command(name="run-controlled-self-trace-p1")
    def run_controlled_self_trace_p1_command(
        model: str = typer.Option(..., "--model", help="actor/diagnostician model identity"),
        execution_backend: str = typer.Option(
            "transformers-peft",
            "--execution-backend",
            help="checkpoint-attested backend; currently transformers-peft only",
        ),
        expected_served_model: str = typer.Option(
            ...,
            "--expected-served-model",
            help="exact model identity expected in endpoint responses",
        ),
        checkpoint_path: Path = typer.Option(
            ...,
            "--checkpoint-path",
            help="local checkpoint directory served by the endpoint",
        ),
        system_cell_path: Path = typer.Option(
            ...,
            "--system-cell-path",
            help="frozen actor System Cell manifest",
        ),
        output_path: Path = typer.Option(
            ...,
            "--output-path",
            help="new immutable P1 result JSON path",
        ),
        dataset: str = typer.Option(
            "controlled_self_trace_heldout_v2",
            "--dataset",
            help="registered frozen self-trace dataset id",
        ),
        trace_method: str = typer.Option(
            "counterfactual_replay",
            "--trace-method",
            help="registered self-trace diagnostic method id",
        ),
        run_admission_sha256: str | None = typer.Option(
            None,
            "--run-admission-sha256",
            help="frozen run-admission manifest SHA-256 required by preregistered datasets",
        ),
        device: str = typer.Option("cuda", "--device", help="in-process policy device"),
        seed: int = typer.Option(0, "--seed"),
    ) -> None:
        """Run controlled interruptions and restart diagnosis with the same checkpoint."""

        if execution_backend != "transformers-peft":
            console.print(
                "[bold red]self-trace P1 rejected: backend lacks checkpoint attestation; "
                "use transformers-peft[/bold red]"
            )
            raise typer.Exit(2)

        from lh_platform.core.system_cell import load_system_cell_manifest
        from lh_platform.tracing import (
            SelfTraceRuntimeBindingV1,
            write_controlled_self_trace_p1,
        )
        from lh_platform.tracing.self_trace_admission import admit_self_trace_run
        from lh_platform.training.transformers_peft_policy import checkpoint_tree_sha256

        in_process_runtime = None
        try:
            if model != expected_served_model:
                raise ValueError("in-process requested and expected model identities differ")
            system_cell = load_system_cell_manifest(system_cell_path)
            admit_self_trace_run(
                system_cell=system_cell,
                model_id=expected_served_model,
                dataset_id=dataset,
                trace_method_id=trace_method,
                sampling_seed=seed,
                expected_run_admission_sha256=run_admission_sha256,
            )
            checkpoint_sha256 = checkpoint_tree_sha256(checkpoint_path)
            runtime_binding = SelfTraceRuntimeBindingV1(
                system_cell=system_cell,
                checkpoint_tree_sha256=checkpoint_sha256,
                expected_served_model_id=expected_served_model,
                sampling_seed=seed,
                run_admission_sha256=run_admission_sha256,
            )

            from lh_platform.models.transformers_peft_chat import FrozenTransformersChatRuntime

            in_process_runtime = FrozenTransformersChatRuntime.load(
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                system_cell=system_cell,
                model_id=model,
                device=device,
                seed=seed,
            )
            model_factory = in_process_runtime.new_session

            result = write_controlled_self_trace_p1(
                actor_model_factory=model_factory,
                diagnostician_model_factory=model_factory,
                runtime_binding=runtime_binding,
                output_path=output_path,
                dataset_id=dataset,
                trace_method_id=trace_method,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            console.print(f"[bold red]self-trace P1 rejected: {error}[/bold red]")
            raise typer.Exit(2) from error
        finally:
            if in_process_runtime is not None:
                in_process_runtime.close()

        color = "green" if result.effect_gate_passed else "yellow"
        console.print(
            f"[bold {color}]self-trace P1 written: {output_path.resolve()}[/bold {color}]\n"
            f"  dataset_id={result.dataset_id}\n"
            f"  trace_method_id={result.trace_method_id}\n"
            f"  exact_localization_rate={result.exact_localization_rate:.3f}\n"
            "  intervention_verified_root_cause_rate="
            f"{result.intervention_verified_root_cause_rate:.3f}\n"
            f"  self_trace_binding_rate={result.self_trace_binding_rate:.3f}\n"
            f"  invalid_output_count={result.invalid_output_count}\n"
            f"  effect_gate_passed={result.effect_gate_passed}\n"
            f"  blocked_by={','.join(result.blocked_by) or 'none'}\n"
            f"  claim_scope={result.claim_scope}"
        )
        if not result.effect_gate_passed:
            raise typer.Exit(3)
