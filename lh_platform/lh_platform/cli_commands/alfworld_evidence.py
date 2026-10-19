"""ALFWorld manifest, collection binding, evidence, and reset-probe commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from lh_platform.studies.evidence import require_verified_source_root

__all__ = ["register_alfworld_evidence_commands"]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def register_alfworld_evidence_commands(
    app: typer.Typer,
    console: Console,
) -> None:
    """Register ALFWorld environment-evidence commands on the public flat CLI."""

    @app.command(name="build-alfworld-manifest")
    def build_alfworld_manifest(
        alfworld_data: Path = typer.Argument(..., help="real ALFWORLD_DATA root"),
        output: Path = typer.Option(
            Path("manifests/alfworld_eval_v1.json"),
            "--output",
            help="task and horizon manifest path",
        ),
        short_max_steps: int = typer.Option(
            20,
            help="largest expert plan in short bucket",
        ),
        medium_max_steps: int = typer.Option(
            50,
            help="largest expert plan in medium bucket",
        ),
    ) -> None:
        """Build a group-safe manifest from real official ALFWorld artifacts."""
        from lh_platform.studies import build_alfworld_task_manifest

        manifest = build_alfworld_task_manifest(
            alfworld_data,
            short_max_steps=short_max_steps,
            medium_max_steps=medium_max_steps,
        )
        issues = manifest.validate(alfworld_data, group_key="group_uid")
        manifest.write(output)
        if issues:
            console.print(f"[bold red]manifest invalid: {output}[/bold red]")
            for issue in issues:
                console.print(f"  [{issue.code}] {issue.path}: {issue.message}")
            raise typer.Exit(2)
        counts = {
            split: sum(task.split == split for task in manifest.tasks)
            for split in ("train", "valid_seen", "valid_unseen")
        }
        console.print(
            f"[bold green]ALFWorld manifest built: {output}[/bold green]\n"
            f"  tasks={len(manifest.tasks)} splits={counts}\n"
            "  split leakage=passed; horizon buckets=recorded"
        )

    @app.command(name="bind-alfworld-training-collection")
    def bind_alfworld_training_collection(
        dataset: Path = typer.Argument(
            ...,
            help="canonical trajectory JSONL to partition and bind",
        ),
        task_manifest: Path = typer.Option(
            ...,
            "--task-manifest",
            help="audited ALFWorld task manifest",
        ),
        output_dir: Path = typer.Option(
            ...,
            "--output-dir",
            help="bundle root; the command atomically creates a collection_id subdirectory",
        ),
        benchmark_audit: Path | None = typer.Option(
            None,
            "--benchmark-audit",
            help=(
                "source-backed benchmark audit; required for --require-source-evaluator-contract"
            ),
        ),
        alfworld_data: Path | None = typer.Option(
            None,
            "--alfworld-data",
            help=("ALFWORLD_DATA root; re-scanned and content-addressed for source-bound training"),
        ),
        holdout_fraction: float = typer.Option(
            0.2,
            "--holdout-fraction",
            min=0.000001,
            max=0.999999,
            help="deterministic holdout fraction when the source has only train records",
        ),
        require_source_evaluator_contract: bool = typer.Option(
            False,
            "--require-source-evaluator-contract",
            help=(
                "require train/valid source split alignment, a verified evaluator audit, "
                "and a re-scanned ALFWORLD_DATA root"
            ),
        ),
    ) -> None:
        """Materialize a group-safe collection bound to ALFWorld task and evaluator evidence.

        This command verifies input identity, selected environment files, and the
        source-owned evaluator contract. It does not execute ALFWorld or emit a
        benchmark success rate, so its artifact remains plumbing evidence.
        """
        from lh_platform.training.benchmark_contract import (
            BenchmarkContractError,
            bind_alfworld_train_eval_collection,
            materialize_alfworld_training_bundle,
        )
        from lh_platform.training.collection import (
            CollectionIntegrityError,
            build_train_eval_collection,
        )
        from lh_platform.training.dataset import build_manifest, read_jsonl

        if not dataset.is_file():
            console.print(f"[bold red]dataset not found: {dataset}[/bold red]")
            raise typer.Exit(2)
        try:
            records = read_jsonl(dataset)
            collection = build_train_eval_collection(
                records,
                holdout_fraction=holdout_fraction,
            )
            binding = bind_alfworld_train_eval_collection(
                collection,
                task_manifest_path=task_manifest,
                benchmark_audit_path=benchmark_audit,
                alfworld_data_root=alfworld_data,
                require_source_evaluator_contract=require_source_evaluator_contract,
            )
        except (
            CollectionIntegrityError,
            BenchmarkContractError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            console.print(f"[bold red]ALFWorld training collection blocked: {error}[/bold red]")
            raise typer.Exit(2) from error

        source_manifest = build_manifest(dataset, records)
        try:
            paths = materialize_alfworld_training_bundle(
                collection,
                binding,
                source_dataset_manifest=source_manifest,
                output_root=output_dir,
            )
        except (BenchmarkContractError, OSError, TypeError, ValueError) as error:
            console.print(
                f"[bold red]ALFWorld training collection not published: {error}[/bold red]"
            )
            raise typer.Exit(2) from error
        console.print(
            f"[bold green]ALFWorld training collection bound: "
            f"{paths['bundle']}[/bold green]\n"
            f"  collection_id={binding.collection_id}\n"
            f"  train={paths['train']}\n"
            f"  eval={paths['eval']}\n"
            f"  binding={paths['binding']}\n"
            "  source_split_evaluation_bound="
            f"{str(binding.source_split_evaluation_bound).lower()}\n"
            "  source_evaluator_contract_bound="
            f"{str(binding.source_evaluator_contract_bound).lower()}\n"
            f"  native_outcomes_verified="
            f"{str(binding.native_outcomes_verified).lower()}\n"
            "  claim_scope=plumbing (no native ALFWorld outcome has been executed)"
        )

    @app.command(name="prepare-alfworld-evidence")
    def prepare_alfworld_evidence(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 file"),
        alfworld_data: Path = typer.Option(
            ...,
            "--alfworld-data",
            help="official ALFWORLD_DATA root",
        ),
        output_dir: Path = typer.Option(
            Path("results/studies/graph_credit_alfworld_v1/alfworld_evidence"),
            "--output-dir",
            help="evidence bundle output directory",
        ),
        run_reset_probe: bool = typer.Option(
            False,
            "--run-reset-probe",
            help=("run the live official ALFWorld reset probe and include it in preflight"),
        ),
        python_executable: str = typer.Option(
            sys.executable,
            "--python",
            help="upstream GraphGPO Python executable",
        ),
        require_ready: bool = typer.Option(
            False,
            "--require-ready",
            help="exit non-zero when runtime preflight blocks launch",
        ),
    ) -> None:
        """Build the official ALFWorld evidence bundle needed before GraphGPO training."""
        from lh_platform.studies import (
            build_alfworld_runtime_evidence,
            build_alfworld_task_manifest,
            load_study,
            write_study_artifacts,
        )
        from lh_platform.studies.alfworld_evidence import (
            ALFWORLD_OFFICIAL_LAYOUT_QUOTES,
            alfworld_official_layout_missing,
        )
        from lh_platform.studies.evidence import find_workspace_root

        protocol = load_study(study_path)
        compiled = protocol.compile()
        workspace_root = find_workspace_root(protocol.source_path)
        source_root = require_verified_source_root(compiled)
        source_commit = str(protocol.design.fixed_factors["source_commit"])
        target = output_dir.resolve()
        target.mkdir(parents=True, exist_ok=True)
        write_study_artifacts(protocol, compiled, target)

        data_root = alfworld_data.expanduser().resolve()
        bundle_path = target / "alfworld_evidence_bundle.json"
        missing_layout = alfworld_official_layout_missing(data_root)
        if missing_layout:
            _write_json(
                bundle_path,
                {
                    "schema": "lh_alfworld_evidence_bundle/v1",
                    "study_id": protocol.id,
                    "ready": False,
                    "blocked_by": ["alfworld_official_layout_missing"],
                    "alfworld_data": str(data_root),
                    "missing_paths": missing_layout,
                    "source_quotes": list(ALFWORLD_OFFICIAL_LAYOUT_QUOTES),
                },
            )
            console.print(
                f"[bold red]official ALFWorld layout missing: "
                f"{data_root}[/bold red]\n"
                f"  missing={missing_layout}\n"
                f"  evidence_bundle={bundle_path}"
            )
            raise typer.Exit(2)

        manifest_path = target / "task_manifest.json"
        manifest = build_alfworld_task_manifest(data_root)
        manifest.write(manifest_path)
        manifest_issues = manifest.validate(data_root, group_key="group_uid")
        if manifest_issues:
            _write_json(
                bundle_path,
                {
                    "schema": "lh_alfworld_evidence_bundle/v1",
                    "study_id": protocol.id,
                    "ready": False,
                    "blocked_by": ["alfworld_task_manifest_invalid"],
                    "alfworld_data": str(data_root),
                    "task_manifest": str(manifest_path),
                    "issues": [issue.to_dict() for issue in manifest_issues],
                    "source_quotes": list(ALFWORLD_OFFICIAL_LAYOUT_QUOTES),
                },
            )
            console.print(f"[bold red]ALFWorld task manifest invalid: {manifest_path}[/bold red]")
            for issue in manifest_issues:
                console.print(f"  [{issue.code}] {issue.path}: {issue.message}")
            raise typer.Exit(2)

        contract_value = protocol.design.fixed_factors.get("source_contract")
        contract_path = (workspace_root / str(contract_value)).resolve() if contract_value else None
        evidence = build_alfworld_runtime_evidence(
            protocol,
            compiled,
            manifest_path=manifest_path,
            source_data_root=data_root,
            source_root=source_root,
            source_commit=source_commit,
            output_root=target,
            python_executable=python_executable,
            source_contract=contract_path,
            run_reset_probe=run_reset_probe,
        )
        probe_path = evidence.state_reset_probe
        if evidence.reset_probe_returncode not in (None, 0):
            console.print(f"[yellow]live state-reset probe failed: {probe_path}[/yellow]")
        runtime_report = evidence.runtime_report
        benchmark_audit_path = evidence.benchmark_audit
        runtime_report_path = target / "runtime_validity_report.json"

        prepare_command = [
            "python",
            "-m",
            "lh_platform.cli",
            "prepare-study",
            str(study_path),
            "--task-manifest",
            str(manifest_path),
            "--alfworld-data",
            str(data_root),
            "--output-dir",
            str(target / "prepared"),
        ]
        if probe_path is not None:
            prepare_command.extend(["--state-reset-probe", str(probe_path)])
        _write_json(
            bundle_path,
            {
                "schema": "lh_alfworld_evidence_bundle/v1",
                "study_id": protocol.id,
                "ready": runtime_report.run_ready,
                "blocked_by": list(runtime_report.blocked_by),
                "alfworld_data": str(data_root),
                "source_root": str(source_root),
                "source_commit": source_commit,
                "source_quotes": list(ALFWORLD_OFFICIAL_LAYOUT_QUOTES),
                "task_manifest": str(manifest_path),
                "task_count": len(manifest.tasks),
                "data_view": str(evidence.runtime_data),
                "benchmark_audit": str(benchmark_audit_path),
                "state_reset_probe": str(probe_path) if probe_path else None,
                "runtime_validity_report": str(runtime_report_path),
                "next_prepare_command": prepare_command,
            },
        )
        color = "green" if runtime_report.run_ready else "yellow"
        console.print(
            f"[bold {color}]ALFWorld evidence bundle built: "
            f"{bundle_path}[/bold {color}]\n"
            f"  tasks={len(manifest.tasks)}\n"
            f"  run_ready={str(runtime_report.run_ready).lower()}\n"
            f"  blocked_by={list(runtime_report.blocked_by)}"
        )
        if require_ready and not runtime_report.run_ready:
            raise typer.Exit(3)

    @app.command(name="probe-alfworld-reset")
    def probe_alfworld_reset(
        study_path: Path = typer.Argument(..., help="path to an lh_study/v2 file"),
        task_manifest: Path = typer.Option(..., "--task-manifest"),
        alfworld_data: Path = typer.Option(..., "--alfworld-data"),
        output: Path = typer.Option(..., "--output"),
        python_executable: str = typer.Option(
            sys.executable,
            "--python",
            help="upstream GraphGPO Python executable",
        ),
    ) -> None:
        """Probe live state isolation through the pinned official ALFWorld worker."""
        from lh_platform.studies import load_study
        from lh_platform.studies.runtime import run_graphgpo_reset_probe

        protocol = load_study(study_path)
        compiled = protocol.compile()
        source_root = require_verified_source_root(compiled)
        completed = run_graphgpo_reset_probe(
            manifest_path=task_manifest,
            source_data_root=alfworld_data,
            source_root=source_root,
            source_commit=str(protocol.design.fixed_factors["source_commit"]),
            output_path=output,
            python_executable=python_executable,
        )
        if completed.returncode != 0:
            console.print(f"[bold red]state-reset probe failed: {output}[/bold red]")
            if completed.stderr.strip():
                console.print(completed.stderr.strip())
            raise typer.Exit(2)
        console.print(f"[bold green]state-reset probe passed: {output}[/bold green]")
