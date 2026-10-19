"""Runtime evidence and fail-closed preflight for purpose-driven studies.

Phase 0 proves that a study design is coherent.  This module proves that the
declared ALFWorld tasks, official GraphGPO source, native outcome, state reset,
budgets, and machine resources can actually support that design.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lh_platform.studies.evidence import find_workspace_root
from lh_platform.studies.protocol import CompiledStudy, StudyProtocol

TASK_MANIFEST_SCHEMA = "lh_task_manifest/v1"
BENCHMARK_AUDIT_SCHEMA = "lh_benchmark_audit/v1"
STATE_RESET_PROBE_SCHEMA = "lh_state_reset_probe/v1"
RUNTIME_REPORT_SCHEMA = "lh_study_runtime_validity/v1"

ALFWORLD_SPLITS = ("train", "valid_seen", "valid_unseen")
ALFWORLD_EXCLUDED_PATH_MARKERS = ("movable", "Sliced")
ALFWORLD_TASK_FILTER_POLICY_ID = "alfworld_supported_six_task_types_v1"
ALFWORLD_TASK_TYPES = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
}

GRAPHGPO_TRAIN_BATCH_SIZE = 16
GRAPHGPO_VAL_BATCH_SIZE = 128
GRAPHGPO_MAX_EPISODE_STEPS = 50
GRAPHGPO_MAX_RESPONSE_LENGTH = 512
GRAPHGPO_REQUIRED_GPUS = 2
GRAPHGPO_REQUIRED_IMPORTS = ("torch", "ray", "textworld", "alfworld", "verl")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RuntimeIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class AlfworldTask:
    task_id: str
    group_uid: str
    split: str
    game_path: str
    trajectory_path: str
    task_type: str
    reference_steps: int
    horizon_bucket: str


@dataclass(frozen=True)
class AlfworldTaskManifest:
    schema: str
    benchmark: str
    dataset_root: str
    horizon_policy: Mapping[str, Any]
    tasks: tuple[AlfworldTask, ...]
    scan_policy: Mapping[str, Any] = field(default_factory=dict)
    scan_summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_keys(self) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            sorted(
                (task.split, task.task_id, task.group_uid, task.game_path) for task in self.tasks
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "benchmark": self.benchmark,
            "dataset_root": self.dataset_root,
            "horizon_policy": dict(self.horizon_policy),
            "tasks": [asdict(task) for task in self.tasks],
        }
        if self.scan_policy:
            payload["scan_policy"] = dict(self.scan_policy)
        if self.scan_summary:
            payload["scan_summary"] = dict(self.scan_summary)
        return payload

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, self.to_dict())
        return target

    @classmethod
    def read(cls, path: str | Path) -> AlfworldTaskManifest:
        raw = _read_json(Path(path))
        tasks = raw.get("tasks") or ()
        if not isinstance(tasks, list):
            raise ValueError("task manifest tasks must be a list")
        return cls(
            schema=str(raw.get("schema", "")),
            benchmark=str(raw.get("benchmark", "")),
            dataset_root=str(raw.get("dataset_root", "")),
            horizon_policy=dict(raw.get("horizon_policy") or {}),
            tasks=tuple(AlfworldTask(**task) for task in tasks),
            scan_policy=dict(raw.get("scan_policy") or {}),
            scan_summary=dict(raw.get("scan_summary") or {}),
        )

    def validate(
        self,
        data_root: str | Path,
        *,
        group_key: str = "group_uid",
    ) -> tuple[RuntimeIssue, ...]:
        root = Path(data_root).resolve()
        issues: list[RuntimeIssue] = []
        if self.schema != TASK_MANIFEST_SCHEMA:
            issues.append(
                RuntimeIssue(
                    "unsupported_task_manifest_schema",
                    "task_manifest.schema",
                    f"expected {TASK_MANIFEST_SCHEMA}",
                )
            )
        if self.benchmark != "alfworld_native":
            issues.append(
                RuntimeIssue(
                    "task_manifest_benchmark_mismatch",
                    "task_manifest.benchmark",
                    "expected alfworld_native",
                )
            )
        if self.scan_policy:
            expected_policy = {
                "id": ALFWORLD_TASK_FILTER_POLICY_ID,
                "included_task_types": sorted(ALFWORLD_TASK_TYPES),
                "excluded_path_markers": list(ALFWORLD_EXCLUDED_PATH_MARKERS),
                "path_marker_match": "case_sensitive_substring_on_task_root",
                "exclusions_are_part_of_benchmark_contract": True,
            }
            if dict(self.scan_policy) != expected_policy:
                issues.append(
                    RuntimeIssue(
                        "task_filter_policy_mismatch",
                        "task_manifest.scan_policy",
                        "ALFWorld scan policy differs from the supported-task contract",
                    )
                )
            if self.scan_summary.get("included_task_count") != len(self.tasks):
                issues.append(
                    RuntimeIssue(
                        "task_scan_summary_mismatch",
                        "task_manifest.scan_summary.included_task_count",
                        "scan summary does not match manifest task count",
                    )
                )
        if not self.tasks:
            issues.append(
                RuntimeIssue(
                    "empty_task_manifest", "task_manifest.tasks", "no ALFWorld tasks found"
                )
            )
            return tuple(issues)

        task_ids: dict[str, int] = {}
        groups: dict[str, set[str]] = {}
        for index, task in enumerate(self.tasks):
            prefix = f"task_manifest.tasks[{index}]"
            task_ids[task.task_id] = task_ids.get(task.task_id, 0) + 1
            group_value = getattr(task, group_key, "")
            if not group_value:
                issues.append(
                    RuntimeIssue(
                        "task_group_key_missing",
                        f"{prefix}.{group_key}",
                        f"group key {group_key!r} is missing",
                    )
                )
            else:
                groups.setdefault(str(group_value), set()).add(task.split)
            if task.split not in ALFWORLD_SPLITS:
                issues.append(
                    RuntimeIssue(
                        "unknown_task_split",
                        f"{prefix}.split",
                        f"unsupported ALFWorld split {task.split!r}",
                    )
                )
            if task.horizon_bucket not in {"short", "medium", "long"}:
                issues.append(
                    RuntimeIssue(
                        "horizon_bucket_missing",
                        f"{prefix}.horizon_bucket",
                        "task must be assigned to short, medium, or long",
                    )
                )
            if task.reference_steps <= 0:
                issues.append(
                    RuntimeIssue(
                        "reference_horizon_missing",
                        f"{prefix}.reference_steps",
                        "horizon bucket requires a positive expert-plan length",
                    )
                )
            for name, relative in (
                ("game_path", task.game_path),
                ("trajectory_path", task.trajectory_path),
            ):
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    issues.append(
                        RuntimeIssue(
                            "task_path_escape",
                            f"{prefix}.{name}",
                            f"task path escapes ALFWORLD_DATA: {relative}",
                        )
                    )
                    continue
                resolved = root / relative_path
                if not resolved.is_file():
                    issues.append(
                        RuntimeIssue(
                            "task_artifact_not_found",
                            f"{prefix}.{name}",
                            f"task artifact not found: {resolved}",
                        )
                    )

        duplicates = sorted(task_id for task_id, count in task_ids.items() if count > 1)
        if duplicates:
            issues.append(
                RuntimeIssue(
                    "duplicate_task_id",
                    "task_manifest.tasks",
                    f"task ids are not unique: {', '.join(duplicates[:5])}",
                )
            )
        leaked = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
        if leaked:
            sample = ", ".join(f"{group}={splits}" for group, splits in sorted(leaked.items())[:5])
            issues.append(
                RuntimeIssue(
                    "task_group_split_leakage",
                    f"task_manifest.tasks.{group_key}",
                    f"groups cross dataset splits: {sample}",
                )
            )
        return tuple(issues)


def build_alfworld_task_manifest(
    data_root: str | Path,
    *,
    short_max_steps: int = 20,
    medium_max_steps: int = 50,
) -> AlfworldTaskManifest:
    """Scan the same artifacts accepted by the official AlfredTWEnv loader."""
    root = Path(data_root).resolve()
    tasks: list[AlfworldTask] = []
    summary: dict[str, int] = {
        "candidate_trajectory_count": 0,
        "included_task_count": 0,
        "excluded_path_marker_count": 0,
        "excluded_multiple_path_markers_count": 0,
        "excluded_movable_marker_count": 0,
        "excluded_Sliced_marker_count": 0,
        "excluded_missing_game_count": 0,
        "excluded_unreadable_artifact_count": 0,
        "excluded_unsupported_task_type_count": 0,
        "excluded_unsolvable_count": 0,
    }
    split_counts: dict[str, dict[str, int]] = {}
    for split in ALFWORLD_SPLITS:
        split_root = root / "json_2.1.1" / split
        if not split_root.is_dir():
            continue
        split_summary = {
            "candidate_trajectory_count": 0,
            "included_task_count": 0,
            "excluded_path_marker_count": 0,
        }
        split_counts[split] = split_summary
        for trajectory_path in sorted(split_root.rglob("traj_data.json")):
            summary["candidate_trajectory_count"] += 1
            split_summary["candidate_trajectory_count"] += 1
            task_root = trajectory_path.parent
            task_root_text = str(task_root)
            matched_markers = [
                marker for marker in ALFWORLD_EXCLUDED_PATH_MARKERS if marker in task_root_text
            ]
            if matched_markers:
                summary["excluded_path_marker_count"] += 1
                split_summary["excluded_path_marker_count"] += 1
                if len(matched_markers) > 1:
                    summary["excluded_multiple_path_markers_count"] += 1
                for marker in matched_markers:
                    summary[f"excluded_{marker}_marker_count"] += 1
                continue
            game_path = task_root / "game.tw-pddl"
            if not game_path.is_file():
                summary["excluded_missing_game_count"] += 1
                continue
            try:
                trajectory = _read_json(trajectory_path)
                game = _read_json(game_path)
            except (OSError, ValueError):
                summary["excluded_unreadable_artifact_count"] += 1
                continue
            task_type = str(trajectory.get("task_type", ""))
            if task_type not in ALFWORLD_TASK_TYPES:
                summary["excluded_unsupported_task_type_count"] += 1
                continue
            if game.get("solvable") is not True:
                summary["excluded_unsolvable_count"] += 1
                continue
            low_actions = (trajectory.get("plan") or {}).get("low_actions") or ()
            reference_steps = len(low_actions)
            if reference_steps <= short_max_steps:
                bucket = "short"
            elif reference_steps <= medium_max_steps:
                bucket = "medium"
            else:
                bucket = "long"
            relative_game = game_path.relative_to(root).as_posix()
            relative_trajectory = trajectory_path.relative_to(root).as_posix()
            explicit_group = trajectory.get("group_uid")
            fallback_group = f"{split}:{task_root.parent.name or task_root.name}"
            tasks.append(
                AlfworldTask(
                    task_id=str(trajectory.get("task_id") or relative_game),
                    group_uid=str(explicit_group or fallback_group),
                    split=split,
                    game_path=relative_game,
                    trajectory_path=relative_trajectory,
                    task_type=task_type,
                    reference_steps=reference_steps,
                    horizon_bucket=bucket,
                )
            )
            summary["included_task_count"] += 1
            split_summary["included_task_count"] += 1
    return AlfworldTaskManifest(
        schema=TASK_MANIFEST_SCHEMA,
        benchmark="alfworld_native",
        dataset_root=str(root),
        horizon_policy={
            "short_max_reference_steps": short_max_steps,
            "medium_max_reference_steps": medium_max_steps,
        },
        tasks=tuple(tasks),
        scan_policy={
            "id": ALFWORLD_TASK_FILTER_POLICY_ID,
            "included_task_types": sorted(ALFWORLD_TASK_TYPES),
            "excluded_path_markers": list(ALFWORLD_EXCLUDED_PATH_MARKERS),
            "path_marker_match": "case_sensitive_substring_on_task_root",
            "exclusions_are_part_of_benchmark_contract": True,
        },
        scan_summary={**summary, "by_split": split_counts},
    )


@dataclass(frozen=True)
class MaterializedAlfworldView:
    root: Path
    task_count: int
    manifest_bound: bool
    source_manifest_sha256: str


def materialize_alfworld_data_view(
    manifest: AlfworldTaskManifest,
    source_root: str | Path,
    output_root: str | Path,
) -> MaterializedAlfworldView:
    """Create the directory view consumed by official GraphGPO from a manifest.

    The upstream runner scans ``$ALFWORLD_DATA`` and has no manifest argument.
    File-level symlinks preserve its native loader while binding the scanned task
    set exactly to the audited manifest.
    """
    source = Path(source_root).resolve()
    target = Path(output_root).resolve()
    issues = manifest.validate(source)
    if issues:
        raise ValueError("; ".join(f"{issue.code}: {issue.message}" for issue in issues))
    manifest_json = json.dumps(manifest.to_dict(), sort_keys=True).encode("utf-8")
    source_manifest_sha256 = hashlib.sha256(manifest_json).hexdigest()
    if target.exists() and any(target.iterdir()):
        metadata_path = target / "dataset_view_manifest.json"
        if metadata_path.is_file():
            metadata = _read_json(metadata_path)
            rebound = build_alfworld_task_manifest(
                target,
                short_max_steps=int(manifest.horizon_policy["short_max_reference_steps"]),
                medium_max_steps=int(manifest.horizon_policy["medium_max_reference_steps"]),
            )
            reusable = (
                metadata.get("source_manifest_sha256") == source_manifest_sha256
                and metadata.get("manifest_bound") is True
                and rebound.task_keys == manifest.task_keys
            )
            if reusable:
                return MaterializedAlfworldView(
                    root=target,
                    task_count=len(rebound.tasks),
                    manifest_bound=True,
                    source_manifest_sha256=source_manifest_sha256,
                )
        raise FileExistsError(f"refusing to replace non-empty ALFWorld view: {target}")
    target.mkdir(parents=True, exist_ok=True)

    source_logic = source / "logic"
    target_logic = target / "logic"
    if not source_logic.is_dir():
        raise FileNotFoundError(f"ALFWorld logic directory not found: {source_logic}")
    target_logic.symlink_to(source_logic, target_is_directory=True)

    copied_roots: set[tuple[Path, Path]] = set()
    for task in manifest.tasks:
        source_task = (source / task.game_path).parent
        target_task = (target / task.game_path).parent
        key = (source_task, target_task)
        if key in copied_roots:
            continue
        copied_roots.add(key)
        target_task.mkdir(parents=True, exist_ok=True)
        for artifact in source_task.iterdir():
            destination = target_task / artifact.name
            if artifact.is_file() and not destination.exists():
                destination.symlink_to(artifact.resolve())

    rebound = build_alfworld_task_manifest(
        target,
        short_max_steps=int(manifest.horizon_policy["short_max_reference_steps"]),
        medium_max_steps=int(manifest.horizon_policy["medium_max_reference_steps"]),
    )
    view = MaterializedAlfworldView(
        root=target,
        task_count=len(rebound.tasks),
        manifest_bound=rebound.task_keys == manifest.task_keys,
        source_manifest_sha256=source_manifest_sha256,
    )
    _write_json(
        target / "dataset_view_manifest.json",
        {
            "schema": "lh_alfworld_data_view/v1",
            "task_count": view.task_count,
            "manifest_bound": view.manifest_bound,
            "source_manifest_sha256": view.source_manifest_sha256,
        },
    )
    return view


def run_graphgpo_reset_probe(
    *,
    manifest_path: str | Path,
    source_data_root: str | Path,
    source_root: str | Path,
    source_commit: str,
    output_path: str | Path,
    python_executable: str = sys.executable,
    seed: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Run a live same-task reset probe through the official GraphGPO worker."""
    manifest_file = Path(manifest_path).resolve()
    manifest = AlfworldTaskManifest.read(manifest_file)
    train_tasks = tuple(task for task in manifest.tasks if task.split == "train")
    if not train_tasks:
        raise ValueError("state-reset probe requires at least one train task")
    probe_manifest = AlfworldTaskManifest(
        schema=manifest.schema,
        benchmark=manifest.benchmark,
        dataset_root=manifest.dataset_root,
        horizon_policy=manifest.horizon_policy,
        tasks=(train_tasks[0],),
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parents[2]
    upstream = Path(source_root).resolve()
    with tempfile.TemporaryDirectory(prefix="lh-reset-probe-", dir=output.parent) as tmp:
        view = materialize_alfworld_data_view(
            probe_manifest, source_data_root, Path(tmp) / "alfworld"
        )
        env = {
            **dict(os.environ),
            "ALFWORLD_DATA": str(view.root),
            "PYTHONPATH": os.pathsep.join((str(package_root), str(upstream))),
        }
        command = (
            python_executable,
            "-m",
            "lh_platform.studies.graphgpo.reset_probe",
            "--source-root",
            str(upstream),
            "--task-id",
            train_tasks[0].task_id,
            "--source-commit",
            source_commit,
            "--task-manifest-sha256",
            sha256_file(manifest_file),
            "--output",
            str(output),
            "--seed",
            str(seed),
        )
        return subprocess.run(
            command,
            cwd=upstream,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )


@dataclass(frozen=True)
class SourceAuditEvidence:
    path: str
    sha256: str
    source_quotes: tuple[str, ...]
    verified: bool


@dataclass(frozen=True)
class BenchmarkAuditComponent:
    name: str
    passed: bool
    claim: str
    evidence: tuple[SourceAuditEvidence, ...]


@dataclass(frozen=True)
class BenchmarkAuditReport:
    schema: str
    benchmark: str
    source_commit: str
    task_manifest_sha256: str
    outcome_owner: str
    outcome_field: str
    outcome_evidence_quality: str
    components: Mapping[str, BenchmarkAuditComponent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "benchmark": self.benchmark,
            "source_commit": self.source_commit,
            "task_manifest_sha256": self.task_manifest_sha256,
            "outcome_owner": self.outcome_owner,
            "outcome_field": self.outcome_field,
            "outcome_evidence_quality": self.outcome_evidence_quality,
            "components": {
                name: {
                    "name": component.name,
                    "passed": component.passed,
                    "claim": component.claim,
                    "evidence": [asdict(item) for item in component.evidence],
                }
                for name, component in self.components.items()
            },
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, self.to_dict())
        return target

    @classmethod
    def read(cls, path: str | Path) -> BenchmarkAuditReport:
        raw = _read_json(Path(path))
        components = {
            name: BenchmarkAuditComponent(
                name=str(value.get("name", name)),
                passed=bool(value.get("passed", False)),
                claim=str(value.get("claim", "")),
                evidence=tuple(SourceAuditEvidence(**item) for item in value.get("evidence") or ()),
            )
            for name, value in (raw.get("components") or {}).items()
        }
        return cls(
            schema=str(raw.get("schema", "")),
            benchmark=str(raw.get("benchmark", "")),
            source_commit=str(raw.get("source_commit", "")),
            task_manifest_sha256=str(raw.get("task_manifest_sha256", "")),
            outcome_owner=str(raw.get("outcome_owner", "")),
            outcome_field=str(raw.get("outcome_field", "")),
            outcome_evidence_quality=str(raw.get("outcome_evidence_quality", "")),
            components=components,
        )


class GraphGPOBenchmarkAuditor:
    """Verify the four benchmark components against the pinned official source."""

    def __init__(self, source_root: str | Path) -> None:
        self.source_root = Path(source_root).resolve()

    def audit(self, *, task_manifest_sha256: str, source_commit: str) -> BenchmarkAuditReport:
        specs = {
            "user": (
                "ALFWorld reset returns benchmark feedback and the benchmark game identity",
                (
                    (
                        "agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py",
                        ("state = super().reset", 'state["extra.gamefile"] = self._gamefile'),
                    ),
                ),
            ),
            "environment": (
                "Official AlfredTWEnv loads solvable native TextWorld games from ALFWORLD_DATA",
                (
                    (
                        "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
                        (
                            "$ALFWORLD_DATA/json_2.1.1/train",
                            "$ALFWORLD_DATA/json_2.1.1/valid_seen",
                            "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
                        ),
                    ),
                    (
                        "agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py",
                        ("traj_data.json", "game.tw-pddl", "solvable"),
                    ),
                ),
            ),
            "ground_truth": (
                "TextWorld is explicitly requested to expose benchmark-owned won evidence",
                (
                    (
                        "agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py",
                        ("textworld.EnvInfos(won=True",),
                    ),
                ),
            ),
            "evaluation": (
                "GraphGPO reward and success aggregation consume benchmark info['won']",
                (
                    (
                        "recipe/GraphGPO/alfworld_envs.py",
                        ("float(info['won'])",),
                    ),
                    (
                        "agent_system/environments/base.py",
                        ("won_value = float(info['won'])",),
                    ),
                ),
            ),
        }
        components: dict[str, BenchmarkAuditComponent] = {}
        for name, (claim, evidence_specs) in specs.items():
            evidence: list[SourceAuditEvidence] = []
            for relative, quotes in evidence_specs:
                path = self.source_root / relative
                text = path.read_text(encoding="utf-8") if path.is_file() else ""
                verified = path.is_file() and all(quote in text for quote in quotes)
                evidence.append(
                    SourceAuditEvidence(
                        path=str(path),
                        sha256=sha256_file(path) if path.is_file() else "",
                        source_quotes=tuple(quotes),
                        verified=verified,
                    )
                )
            components[name] = BenchmarkAuditComponent(
                name=name,
                passed=bool(evidence) and all(item.verified for item in evidence),
                claim=claim,
                evidence=tuple(evidence),
            )
        return BenchmarkAuditReport(
            schema=BENCHMARK_AUDIT_SCHEMA,
            benchmark="alfworld_native",
            source_commit=source_commit,
            task_manifest_sha256=task_manifest_sha256,
            outcome_owner="benchmark",
            outcome_field="won",
            outcome_evidence_quality="strong",
            components=components,
        )


@dataclass(frozen=True)
class RuntimeResourceEvidence:
    python_executable: str
    imports_ok: bool
    required_imports: tuple[str, ...]
    available_gpus: int
    required_gpus: int
    details: str

    @property
    def passed(self) -> bool:
        return self.imports_ok and self.available_gpus >= self.required_gpus


@dataclass(frozen=True)
class RuntimeInputs:
    task_manifest: Path | None = None
    benchmark_audit: Path | None = None
    state_reset_probe: Path | None = None
    alfworld_data: Path | None = None
    python_executable: str = sys.executable
    source_contract: Path | None = None
    resource_profile: str = ""
    required_gpus_override: int | None = None


@dataclass(frozen=True)
class RuntimeCheck:
    code: str
    passed: bool
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimePreflightReport:
    schema: str
    study_id: str
    run_ready: bool
    planned_execution_path: str | None
    checks: tuple[RuntimeCheck, ...]

    @property
    def blocked_by(self) -> tuple[str, ...]:
        return tuple(check.code for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "study_id": self.study_id,
            "run_ready": self.run_ready,
            "planned_execution_path": self.planned_execution_path,
            "blocked_by": list(self.blocked_by),
            "checks": [asdict(check) for check in self.checks],
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, self.to_dict())
        return target

    @classmethod
    def ready_for_test(cls, study_id: str) -> RuntimePreflightReport:
        return cls(
            schema=RUNTIME_REPORT_SCHEMA,
            study_id=study_id,
            run_ready=True,
            planned_execution_path="upstream_native_runner",
            checks=(RuntimeCheck("test_runtime_ready", True, "test fixture"),),
        )

    @classmethod
    def blocked_for_test(cls, study_id: str, code: str) -> RuntimePreflightReport:
        return cls(
            schema=RUNTIME_REPORT_SCHEMA,
            study_id=study_id,
            run_ready=False,
            planned_execution_path=None,
            checks=(RuntimeCheck(code, False, "test fixture"),),
        )


ResourceProbe = Callable[[str, int], RuntimeResourceEvidence]


class RuntimePreflight:
    def __init__(self, *, resource_probe: ResourceProbe | None = None) -> None:
        self.resource_probe = resource_probe or _probe_runtime_resources

    def check(
        self,
        protocol: StudyProtocol,
        compiled: CompiledStudy,
        inputs: RuntimeInputs,
    ) -> RuntimePreflightReport:
        checks: list[RuntimeCheck] = []
        workspace_root = find_workspace_root(protocol.source_path)
        source_root, source_commit = _resolved_source(compiled)
        expected_commit = str(protocol.design.fixed_factors.get("source_commit", ""))
        entrypoint = source_root / "recipe/GraphGPO/main_graphgpo.py" if source_root else None
        source_ok = bool(
            source_root
            and source_root.is_dir()
            and source_commit == expected_commit
            and entrypoint
            and entrypoint.is_file()
        )
        checks.append(
            RuntimeCheck(
                "upstream_source_verified",
                source_ok,
                "official GraphGPO entrypoint is pinned to the declared commit"
                if source_ok
                else "official GraphGPO entrypoint or pinned commit is unavailable",
                {
                    "source_root": str(source_root) if source_root else None,
                    "source_commit": source_commit,
                    "expected_commit": expected_commit,
                    "entrypoint": str(entrypoint) if entrypoint else None,
                },
            )
        )

        contract_path = _resolve_input_path(inputs.source_contract, workspace_root)
        if contract_path and source_root:
            from lh_platform.studies.source_contract import verify_source_contract

            try:
                contract = verify_source_contract(contract_path, source_root=source_root)
                contract_ok = contract.verified
                contract_evidence = {
                    "path": str(contract_path),
                    "coverage": contract.coverage,
                    "blocked_by": list(contract.blocked_by),
                }
            except (OSError, TypeError, ValueError) as error:
                contract_ok = False
                contract_evidence = {"path": str(contract_path), "error": str(error)}
        else:
            contract_ok = protocol.schema != "lh_study/v3"
            contract_evidence = {"path": str(contract_path) if contract_path else None}
        checks.append(
            RuntimeCheck(
                "source_contract_verified" if contract_ok else "source_contract_unverified",
                contract_ok,
                "primary-source contract covers every scientific field"
                if contract_ok
                else "confirmatory v3 execution requires a verified primary-source contract",
                contract_evidence,
            )
        )

        manifest_path = _resolve_input_path(
            inputs.task_manifest or Path(protocol.design.task_manifest), workspace_root
        )
        manifest: AlfworldTaskManifest | None = None
        manifest_sha256 = ""
        data_root = _resolve_input_path(inputs.alfworld_data, workspace_root)
        if not manifest_path or not manifest_path.is_file():
            checks.append(
                RuntimeCheck(
                    "task_manifest_not_found",
                    False,
                    f"task manifest not found: {manifest_path}",
                )
            )
        elif not data_root or not data_root.is_dir():
            checks.append(
                RuntimeCheck(
                    "alfworld_data_not_found",
                    False,
                    f"ALFWORLD_DATA not found: {data_root}",
                )
            )
        else:
            try:
                manifest = AlfworldTaskManifest.read(manifest_path)
                manifest_sha256 = sha256_file(manifest_path)
                manifest_issues = manifest.validate(
                    data_root, group_key=protocol.integrity_requirements.group_key
                )
            except (OSError, TypeError, ValueError) as error:
                manifest_issues = (
                    RuntimeIssue("task_manifest_invalid", "task_manifest", str(error)),
                )
            if manifest_issues:
                checks.extend(
                    RuntimeCheck(issue.code, False, issue.message, {"path": issue.path})
                    for issue in manifest_issues
                )
            else:
                checks.append(
                    RuntimeCheck(
                        "task_manifest_verified",
                        True,
                        "task ids, group-level splits, paths, and horizon buckets are valid",
                        {
                            "path": str(manifest_path),
                            "sha256": manifest_sha256,
                            "task_count": len(manifest.tasks),
                        },
                    )
                )
                rebound = build_alfworld_task_manifest(
                    data_root,
                    short_max_steps=int(manifest.horizon_policy["short_max_reference_steps"]),
                    medium_max_steps=int(manifest.horizon_policy["medium_max_reference_steps"]),
                )
                bound = rebound.task_keys == manifest.task_keys
                checks.append(
                    RuntimeCheck(
                        "task_manifest_bound_to_upstream",
                        bound,
                        "ALFWORLD_DATA scan exactly matches the task manifest"
                        if bound
                        else "official ALFWorld directory scan does not match the task manifest",
                        {
                            "manifest_tasks": len(manifest.tasks),
                            "scanned_tasks": len(rebound.tasks),
                        },
                    )
                )
                counts = {
                    split: sum(task.split == split for task in manifest.tasks)
                    for split in ALFWORLD_SPLITS
                }
                capacity_ok = (
                    counts["train"] >= GRAPHGPO_TRAIN_BATCH_SIZE
                    and counts["valid_seen"] >= GRAPHGPO_VAL_BATCH_SIZE
                )
                checks.append(
                    RuntimeCheck(
                        "task_manifest_capacity",
                        capacity_ok,
                        "manifest covers the official train and validation batch sizes"
                        if capacity_ok
                        else "manifest is smaller than the official 16/128 train/validation batches",
                        counts,
                    )
                )

        self._check_benchmark_audit(
            checks,
            protocol,
            inputs.benchmark_audit,
            workspace_root,
            manifest_sha256,
            source_commit,
        )
        self._check_state_reset(
            checks,
            inputs.state_reset_probe,
            workspace_root,
            manifest_sha256,
            source_commit,
        )
        self._check_budget(checks, protocol)

        formal_required_gpus = int(
            protocol.design.fixed_factors.get("num_gpus", GRAPHGPO_REQUIRED_GPUS)
        )
        required_gpus = int(inputs.required_gpus_override or formal_required_gpus)
        if not data_root or not data_root.is_dir() or not source_ok:
            resources = RuntimeResourceEvidence(
                python_executable=inputs.python_executable,
                imports_ok=False,
                required_imports=GRAPHGPO_REQUIRED_IMPORTS,
                available_gpus=0,
                required_gpus=required_gpus,
                details="resource probe skipped until source and ALFWORLD_DATA exist",
            )
        else:
            resources = self.resource_probe(inputs.python_executable, required_gpus)
        resource_evidence = {
            **asdict(resources),
            "formal_required_gpus": formal_required_gpus,
            "effective_required_gpus": required_gpus,
            "resource_profile": inputs.resource_profile or None,
        }
        checks.append(
            RuntimeCheck(
                "runtime_resources_verified"
                if resources.passed
                else "runtime_resources_unavailable",
                resources.passed,
                "required upstream imports and GPUs are available"
                if resources.passed
                else resources.details,
                resource_evidence,
            )
        )

        run_ready = all(check.passed for check in checks)
        return RuntimePreflightReport(
            schema=RUNTIME_REPORT_SCHEMA,
            study_id=protocol.id,
            run_ready=run_ready,
            planned_execution_path="upstream_native_runner" if source_ok else None,
            checks=tuple(checks),
        )

    @staticmethod
    def _check_benchmark_audit(
        checks: list[RuntimeCheck],
        protocol: StudyProtocol,
        audit_input: Path | None,
        workspace_root: Path,
        manifest_sha256: str,
        source_commit: str,
    ) -> None:
        path = _resolve_input_path(audit_input, workspace_root)
        if not path or not path.is_file():
            checks.append(
                RuntimeCheck(
                    "benchmark_audit_not_found", False, f"benchmark audit not found: {path}"
                )
            )
            checks.append(
                RuntimeCheck(
                    "native_success_signal_unverified",
                    False,
                    "benchmark-owned ALFWorld success signal is not audited",
                )
            )
            return
        try:
            audit = BenchmarkAuditReport.read(path)
        except (OSError, TypeError, ValueError) as error:
            checks.append(RuntimeCheck("benchmark_audit_invalid", False, str(error)))
            return
        required = set(protocol.validity.required_benchmark_audit_components)
        components_ok = required.issubset(audit.components) and all(
            audit.components[name].passed for name in required if name in audit.components
        )
        hashes_ok = all(
            item.verified and Path(item.path).is_file() and sha256_file(item.path) == item.sha256
            for name in required
            for item in audit.components.get(
                name, BenchmarkAuditComponent(name, False, "", ())
            ).evidence
        )
        binding_ok = (
            audit.schema == BENCHMARK_AUDIT_SCHEMA
            and audit.benchmark == protocol.design.benchmark
            and bool(manifest_sha256)
            and audit.task_manifest_sha256 == manifest_sha256
            and audit.source_commit == source_commit
        )
        audit_ok = components_ok and hashes_ok and binding_ok
        checks.append(
            RuntimeCheck(
                "benchmark_audit_verified",
                audit_ok,
                "four benchmark components are source- and manifest-bound"
                if audit_ok
                else "benchmark audit components, hashes, manifest, or commit do not match",
                {"path": str(path), "required_components": sorted(required)},
            )
        )
        outcome_ok = (
            audit_ok
            and audit.outcome_owner == "benchmark"
            and audit.outcome_field == "won"
            and audit.outcome_evidence_quality == "strong"
        )
        checks.append(
            RuntimeCheck(
                "native_success_signal_verified",
                outcome_ok,
                "ALFWorld info['won'] is the audited native success signal"
                if outcome_ok
                else "native ALFWorld success signal is absent or insufficient",
            )
        )

    @staticmethod
    def _check_state_reset(
        checks: list[RuntimeCheck],
        probe_input: Path | None,
        workspace_root: Path,
        manifest_sha256: str,
        source_commit: str,
    ) -> None:
        path = _resolve_input_path(probe_input, workspace_root)
        if not path or not path.is_file():
            checks.append(
                RuntimeCheck(
                    "state_reset_probe_not_found",
                    False,
                    f"live state-reset probe not found: {path}",
                )
            )
            return
        try:
            raw = _read_json(path)
            restored = (
                raw.get("schema") == STATE_RESET_PROBE_SCHEMA
                and raw.get("benchmark") == "alfworld_native"
                and raw.get("source_commit") == source_commit
                and bool(manifest_sha256)
                and raw.get("task_manifest_sha256") == manifest_sha256
                and raw.get("probe_kind") == "live_alfworld_same_task"
                and raw.get("actual_execution_path") == "upstream_native_runner"
                and raw.get("exit_code") == 0
                and raw.get("task_id_before") == raw.get("task_id_after")
                and raw.get("initial_state_sha256") == raw.get("reset_state_sha256")
                and raw.get("initial_state_sha256") != raw.get("mutated_state_sha256")
                and not raw.get("residual_state_fields")
            )
        except (OSError, ValueError):
            restored = False
        checks.append(
            RuntimeCheck(
                "state_reset_probe_verified" if restored else "state_reset_probe_failed",
                restored,
                "live same-task probe restored initial state without residual fields"
                if restored
                else "live reset did not restore the same task to its initial state",
                {"path": str(path)},
            )
        )

    @staticmethod
    def _check_budget(checks: list[RuntimeCheck], protocol: StudyProtocol) -> None:
        budget = protocol.design.budget
        training_steps = int(budget.get("training_steps", 0))
        group_size = int(protocol.design.fixed_factors.get("rollout_group_size", 0))
        episodes = training_steps * GRAPHGPO_TRAIN_BATCH_SIZE * group_size
        environment_budget = int(budget.get("environment_steps", 0))
        generated_token_budget = int(budget.get("generated_tokens", 0))
        required_environment_steps = episodes * GRAPHGPO_MAX_EPISODE_STEPS
        # The official rollout loop calls generate_sequences inside the
        # env.max_steps loop. max_response_length is therefore a per-step cap.
        required_generated_tokens = (
            episodes * GRAPHGPO_MAX_EPISODE_STEPS * GRAPHGPO_MAX_RESPONSE_LENGTH
        )
        environment_ok = episodes > 0 and environment_budget >= required_environment_steps
        token_ok = episodes > 0 and generated_token_budget >= required_generated_tokens
        checks.append(
            RuntimeCheck(
                "environment_budget_guard_verified"
                if environment_ok
                else "environment_budget_infeasible",
                environment_ok,
                "environment budget covers the official env.max_steps horizon"
                if environment_ok
                else "environment budget cannot cover the official env.max_steps horizon",
                {
                    "scheduled_rollouts": episodes,
                    "official_max_episode_steps": GRAPHGPO_MAX_EPISODE_STEPS,
                    "declared_environment_steps": environment_budget,
                    "required_environment_steps": required_environment_steps,
                },
            )
        )
        checks.append(
            RuntimeCheck(
                "generated_token_budget_guard_verified"
                if token_ok
                else "generated_token_budget_infeasible",
                token_ok,
                "token budget covers every official multi-turn generation step"
                if token_ok
                else "token budget cannot cover the official data.max_response_length",
                {
                    "scheduled_rollouts": episodes,
                    "official_max_episode_steps": GRAPHGPO_MAX_EPISODE_STEPS,
                    "official_max_response_length": GRAPHGPO_MAX_RESPONSE_LENGTH,
                    "declared_generated_tokens": generated_token_budget,
                    "required_generated_tokens": required_generated_tokens,
                },
            )
        )


def _probe_runtime_resources(python_executable: str, required_gpus: int) -> RuntimeResourceEvidence:
    imports = ", ".join(GRAPHGPO_REQUIRED_IMPORTS)
    code = "import torch, ray, textworld, alfworld, verl; print(torch.cuda.device_count())"
    try:
        completed = subprocess.run(
            [python_executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        available = (
            int(completed.stdout.strip().splitlines()[-1]) if completed.returncode == 0 else 0
        )
        imports_ok = completed.returncode == 0
        details = (
            f"imports ok; {available} GPUs detected"
            if imports_ok
            else completed.stderr.strip() or f"required imports failed: {imports}"
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        available = 0
        imports_ok = False
        details = str(error)
    return RuntimeResourceEvidence(
        python_executable=python_executable,
        imports_ok=imports_ok,
        required_imports=GRAPHGPO_REQUIRED_IMPORTS,
        available_gpus=available,
        required_gpus=required_gpus,
        details=details,
    )


def _resolved_source(compiled: CompiledStudy) -> tuple[Path | None, str]:
    for evidence in compiled.resolved_evidence:
        if evidence.source_root and evidence.source_commit_verified:
            return Path(evidence.source_root), str(evidence.source_commit or "")
    return None, ""


def _resolve_input_path(value: Path | None, workspace_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON object required: {path}")
    return raw


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
