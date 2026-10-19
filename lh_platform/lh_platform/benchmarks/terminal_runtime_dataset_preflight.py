"""Controller-owned Harbor dataset admission before backend or Actor probing.

This local design is implementation_unverified. It is separate from the planned
TraceFix DatasetPreflightReceipt and never authorizes a formal experiment.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

_SCHEMA = "terminal_runtime_dataset_preflight_v1"
_RECEIPT_NAME = "controller-dataset-preflight.json"
_FROZEN_NAMESPACE_SCHEMA = "terminal_runtime_frozen_namespace_v1"
_FROZEN_PARENT_NAME = ".managed-runtime-frozen"
_FROZEN_PARENT_MARKER_NAME = ".longh-controller-namespace.json"
_MANAGED_KEYS = (
    "managed_runtime_backend",
    "managed_runtime_storage",
    "managed_runtime_actor",
    "managed_runtime_dataset_preflight",
)


class DatasetPreflightError(ValueError):
    """Expose only the frozen dataset failure code to the controller."""

    def __init__(self, code: str, *, receipt_owned: bool = True):
        self.code = code
        self.receipt_owned = receipt_owned
        super().__init__(code)


def _persist_json(path: Path, value: dict[str, object]) -> None:
    """Publish one complete file and sync both data and its directory entry."""

    staging = path.with_name(path.name + ".writing")
    with staging.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sync_tree(root: Path) -> None:
    """Sync copied task files and directory entries before accepting the launch."""

    directories = [root]
    for path in root.rglob("*"):
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _sync_directory(path: Path) -> None:
    """Persist one directory entry boundary after an exclusive child create."""

    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _persist_frozen_namespace_marker(path: Path, value: dict[str, object]) -> None:
    """Create one immutable-by-convention namespace marker without replacement."""

    # Exclusive create is the no-clobber step; it works on filesystems without hard links,
    # such as the run storage the Docker VM mounts. A crash mid-write leaves a partial marker
    # that the content comparison in _create_frozen_root rejects.
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        mode = stat.S_IMODE(os.fstat(stream.fileno()).st_mode) & ~0o222
        os.fchmod(stream.fileno(), mode)
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def filesystem_enforces_modes(directory: Path) -> bool:
    """Report whether removing write bits is visible on this filesystem."""

    descriptor, name = tempfile.mkstemp(dir=directory, prefix=".longh-mode-probe-")
    probe = Path(name)
    try:
        os.close(descriptor)
        os.chmod(probe, 0o444)
        return not stat.S_IMODE(probe.stat().st_mode) & 0o222
    finally:
        os.chmod(probe, 0o644)
        probe.unlink()


def _create_frozen_root(job_dir: Path) -> Path:
    """Create or verify the locally marked frozen namespace before use."""

    jobs_dir = job_dir.parent
    frozen_parent = jobs_dir / _FROZEN_PARENT_NAME
    created_parent = False
    try:
        frozen_parent.mkdir(exist_ok=False)
        created_parent = True
        _sync_directory(jobs_dir)
    except FileExistsError:
        pass
    if frozen_parent.is_symlink() or not frozen_parent.is_dir():
        raise ValueError("managed frozen parent must be a real directory")
    if frozen_parent.resolve(strict=True) != frozen_parent:
        raise ValueError("managed frozen parent escaped the jobs directory")

    marker_path = frozen_parent / _FROZEN_PARENT_MARKER_NAME
    expected_marker: dict[str, object] = {
        "schema_version": _FROZEN_NAMESPACE_SCHEMA,
        "owner": "lh_platform.terminal_runtime_dataset_preflight",
        "jobs_dir": str(jobs_dir),
    }
    if created_parent:
        _persist_frozen_namespace_marker(marker_path, expected_marker)
    marker_staging_path = marker_path.with_name(marker_path.name + ".writing")
    if marker_staging_path.exists() or marker_staging_path.is_symlink():
        raise ValueError("managed frozen parent has an incomplete controller marker")
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("managed frozen parent lacks its controller marker")
    if marker_path.resolve(strict=True).parent != frozen_parent:
        raise ValueError("managed frozen parent marker escaped its namespace")
    # Read-only modes are a convention the verifier byte contract backs up; a filesystem
    # that ignores modes cannot express it, and the preflight receipt records that.
    if filesystem_enforces_modes(frozen_parent) and stat.S_IMODE(
        marker_path.stat().st_mode
    ) & 0o222:
        raise ValueError("managed frozen parent marker must not be writable")
    if json.loads(marker_path.read_text()) != expected_marker:
        raise ValueError("managed frozen parent marker does not match this jobs directory")

    frozen_root = frozen_parent / job_dir.name
    if frozen_root.exists() or frozen_root.is_symlink():
        raise FileExistsError("frozen launch tree already exists")
    frozen_root.mkdir(exist_ok=False)
    _sync_directory(frozen_parent)
    if frozen_root.is_symlink() or frozen_root.resolve(strict=True).parent != frozen_parent:
        raise ValueError("managed frozen root escaped its parent")
    return frozen_root


def _reject_tree_symlinks(root: Path) -> None:
    """Reject task links because copy semantics would change or escape content."""

    for directory, child_directories, filenames in os.walk(root, followlinks=False):
        for name in (*child_directories, *filenames):
            if (Path(directory) / name).is_symlink():
                raise ValueError("task trees with symbolic links are unsupported")


def _fully_read_task(task: Any) -> tuple[str, ...]:
    """Read every multi-step instruction that Harbor defers until trial execution."""

    config = task.config
    steps = config.steps or ()
    return tuple(task.step_instruction(step.name) for step in steps)


def _validate_step_names(config: Any) -> None:
    """Reject step names that Harbor would resolve outside the task step root."""

    names: list[str] = []
    for step in config.steps or ():
        component = Path(step.name)
        if (
            not step.name
            or component.is_absolute()
            or component.name != step.name
            or step.name in {".", ".."}
            or "/" in step.name
            or "\\" in step.name
        ):
            raise ValueError("task step name must be one path component")
        names.append(step.name)
    if len(names) != len(set(names)):
        raise ValueError("task step names must be unique")


def _load_task(
    path: Path,
    *,
    task_type: Any,
    task_config_type: Any,
    extra_instruction_paths: list[Path],
    disable_verification: bool,
) -> tuple[Any, tuple[str, ...]]:
    """Validate path-bearing task config before Harbor resolves any step path."""

    definition = task_config_type.model_validate_toml((path / "task.toml").read_text())
    _validate_step_names(definition)
    task = task_type(
        task_dir=path,
        extra_instruction_paths=extra_instruction_paths,
        disable_verification=disable_verification,
    )
    if task.config != definition:
        raise ValueError("task config changed while it was being loaded")
    return task, _fully_read_task(task)


def _expected_roster(config: dict[str, object]) -> list[dict[str, str | None]]:
    """Read an independent ordered roster without deriving it from discovery."""

    declaration = config.get("managed_runtime_dataset_preflight")
    roster = declaration.get("expected_roster") if isinstance(declaration, dict) else None
    if not isinstance(roster, list) or not roster:
        raise DatasetPreflightError("dataset_config_invalid")
    expected: list[dict[str, str | None]] = []
    for item in roster:
        if not isinstance(item, dict) or set(item) != {
            "task_id",
            "source",
            "canonical_path",
        }:
            raise DatasetPreflightError("dataset_config_invalid")
        task_id, source = item["task_id"], item["source"]
        canonical_path = item["canonical_path"]
        if not isinstance(task_id, str) or not task_id:
            raise DatasetPreflightError("dataset_config_invalid")
        if source is not None:
            raise DatasetPreflightError("dataset_config_invalid")
        if not isinstance(canonical_path, str) or not canonical_path:
            raise DatasetPreflightError("dataset_config_invalid")
        source_path = Path(canonical_path).expanduser()
        if not source_path.is_absolute() or str(source_path) != canonical_path:
            raise DatasetPreflightError("dataset_config_invalid")
        expected.append(
            {
                "task_id": task_id,
                "source": source,
                "canonical_path": canonical_path,
            }
        )
    if len({item["task_id"] for item in expected}) != len(expected):
        raise DatasetPreflightError("dataset_config_invalid")
    return expected


def _mark_failure(receipt_path: Path, receipt: dict[str, object], code: str) -> None:
    """Persist a fixed code without serializing upstream exceptions or secrets."""

    receipt["status"] = "rejected"
    receipt["failure_code"] = code
    _persist_json(receipt_path, receipt)
    raise DatasetPreflightError(code)


def prepare_terminal_runtime_dataset_launch(
    config: dict[str, object],
    job_dir: Path,
    *,
    on_job_owned: Callable[[], None] | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    """Resolve, fully load, freeze and reload Harbor tasks before launch probes."""

    configured_jobs_dir = config.get("jobs_dir")
    configured_job_name = config.get("job_name")
    if (
        not isinstance(configured_jobs_dir, str)
        or not isinstance(configured_job_name, str)
        or Path(configured_jobs_dir).expanduser().resolve(strict=True) != job_dir.parent
        or configured_job_name != job_dir.name
    ):
        raise DatasetPreflightError("dataset_config_invalid", receipt_owned=False)

    receipt_path = job_dir / _RECEIPT_NAME
    try:
        # The job directory is the controller ownership claim. Only one process
        # can create it, so later progress updates cannot overwrite another run.
        job_dir.mkdir(exist_ok=False)
        if on_job_owned is not None:
            on_job_owned()
        _sync_directory(job_dir.parent)
    except FileExistsError:
        raise DatasetPreflightError("dataset_config_invalid", receipt_owned=False) from None
    receipt: dict[str, object] = {
        "schema_version": _SCHEMA,
        "design_status": "locally_designed",
        "implementation_status": "implementation_unverified",
        "status": "started",
        "failure_code": None,
        "expected_roster": None,
        "resolved_roster": None,
        "expected_count": None,
        "resolved_count": None,
        "materialized_count": None,
        "full_loaded_count": None,
        "frozen_copied_count": None,
        "frozen_reloaded_count": None,
        "harbor_version": None,
        "harbor_module_root": None,
        "harbor_distribution_root": None,
    }
    _persist_json(receipt_path, receipt)

    try:
        # Import Harbor only after a durable controller receipt exists.
        import harbor
        from harbor import JobPlan
        from harbor.models.job.config import JobConfig
        from harbor.models.task.config import TaskConfig as TaskDefinitionConfig
        from harbor.models.task.task import Task
        from harbor.models.trial.config import TaskConfig, VerifierConfig

        if harbor.__version__ != "0.21.0":
            raise ValueError("controller dataset preflight requires Harbor 0.21.0")
        module_file = Path(harbor.__file__).resolve(strict=True)
        distribution_root = Path(distribution("harbor").locate_file("")).resolve(strict=True)
        if module_file.parent.parent != distribution_root:
            raise ValueError("loaded Harbor module is outside its installed distribution root")
        receipt["harbor_version"] = harbor.__version__
        receipt["harbor_module_root"] = str(module_file.parent)
        receipt["harbor_distribution_root"] = str(distribution_root)
        _persist_json(receipt_path, receipt)
    except Exception:
        _mark_failure(receipt_path, receipt, "runtime_dependency_incompatible")

    try:
        expected = _expected_roster(config)
        receipt["expected_roster"] = expected
        receipt["expected_count"] = len(expected)
        raw_tasks = config.get("tasks", [])
        raw_datasets = config.get("datasets", [])
        raw_metrics = config.get("metrics", [])
        raw_verifier = config.get("verifier", {})
        if (
            not isinstance(raw_tasks, list)
            or not raw_tasks
            or not isinstance(raw_datasets, list)
            or raw_datasets
            or not isinstance(raw_metrics, list)
            or not isinstance(raw_verifier, dict)
            or config.get("source_jobs", []) != []
            or config.get("extra_instruction_paths", []) != []
        ):
            raise ValueError("unsupported dataset launch shape")
        install_only = config.get("install_only", False)
        if set(raw_verifier) - {"disable"}:
            raise ValueError("managed scored runs require the default official verifier")
        if "disable" in raw_verifier and type(raw_verifier["disable"]) is not bool:
            raise ValueError("managed verifier disable flag must be boolean")
        verifier_config = VerifierConfig.model_validate(raw_verifier)
        if not isinstance(install_only, bool):
            raise ValueError("invalid verifier mode")
        if verifier_config.disable or install_only:
            raise ValueError("managed scored runs require the official verifier")
        # Validate only the Harbor inputs used by dataset resolution. Full job
        # validation belongs to Harbor and must not classify Actor errors here.
        harbor_config = JobConfig.model_construct(
            tasks=[TaskConfig.model_validate(item) for item in raw_tasks],
            datasets=[],
            verifier=verifier_config,
            extra_instruction_paths=[],
        )
        if any(
            task.path is None
            or task.source is not None
            or task.is_git_task()
            or task.is_package_task()
            for task in harbor_config.tasks
        ):
            raise ValueError("only ungrouped explicit local task paths are admitted in this slice")
        if any(task.overwrite for task in harbor_config.tasks):
            raise ValueError("cache overwrite has no source ownership proof")
        from harbor.models.metric.config import MetricConfig
        from harbor.models.metric.type import MetricType

        if any(
            not isinstance(item, dict) or not set(item) <= {"type", "kwargs"}
            for item in raw_metrics
        ):
            raise ValueError("metric config contains unsupported fields")
        metrics = [MetricConfig.model_validate(item) for item in raw_metrics]
        if any(metric.type is MetricType.UV_SCRIPT or metric.kwargs for metric in metrics):
            raise ValueError("external or parameterized metrics are unsupported")
    except Exception:
        _mark_failure(receipt_path, receipt, "dataset_config_invalid")
    _persist_json(receipt_path, receipt)

    try:
        # Harbor resolves the accepted explicit-task roster through its own path.
        task_configs = asyncio.run(JobPlan.resolve_task_configs(harbor_config))
        resolved = [
            {
                "task_id": task.get_task_id().get_name(),
                "source": task.source,
                "canonical_path": str(task.path.expanduser().resolve(strict=True)),
            }
            for task in task_configs
        ]
    except Exception:
        _mark_failure(receipt_path, receipt, "dataset_resolution_failed")
    receipt["resolved_roster"] = resolved
    receipt["resolved_count"] = len(resolved)
    _persist_json(receipt_path, receipt)
    if resolved != expected:
        _mark_failure(receipt_path, receipt, "dataset_roster_mismatch")

    try:
        # Use Harbor's own materialization result and input-order reconstruction.
        downloads = asyncio.run(JobPlan.cache_tasks(task_configs))
    except Exception:
        _mark_failure(receipt_path, receipt, "task_download_failed")

    materialized: list[Path] = []
    try:
        for task, roster_entry in zip(task_configs, resolved, strict=True):
            result = downloads[task.get_task_id()]
            path = result.path.expanduser().resolve(strict=True)
            if not path.is_dir():
                raise ValueError("download result is not a directory")
            if str(path) != roster_entry["canonical_path"]:
                raise ValueError("materialized task path differs from resolved roster")
            materialized.append(path)
    except Exception:
        receipt["materialized_count"] = len(materialized)
        _mark_failure(receipt_path, receipt, "task_materialization_failed")
    receipt["materialized_count"] = len(materialized)
    _persist_json(receipt_path, receipt)

    loaded: list[tuple[Any, tuple[str, ...]]] = []
    try:
        for path in materialized:
            _reject_tree_symlinks(path)
            loaded.append(
                _load_task(
                    path,
                    task_type=Task,
                    task_config_type=TaskDefinitionConfig,
                    extra_instruction_paths=harbor_config.extra_instruction_paths,
                    disable_verification=harbor_config.verifier.disable,
                )
            )
    except Exception:
        receipt["full_loaded_count"] = len(loaded)
        _mark_failure(receipt_path, receipt, "task_full_load_failed")
    receipt["full_loaded_count"] = len(loaded)
    _persist_json(receipt_path, receipt)

    if len({path.name for path in materialized}) != len(materialized):
        _mark_failure(receipt_path, receipt, "dataset_roster_mismatch")

    frozen_paths: list[Path] = []
    try:
        # Existing managed paths may belong to an interrupted launch; preserve them.
        frozen_root = _create_frozen_root(job_dir)
        receipt["frozen_mode_enforcement"] = filesystem_enforces_modes(frozen_root.parent)
        (frozen_root / "tasks").mkdir(exist_ok=False)
        for source in materialized:
            _reject_tree_symlinks(source)
            target = frozen_root / "tasks" / source.name
            shutil.copytree(source, target, symlinks=True)
            _reject_tree_symlinks(target)
            for path in sorted(target.rglob("*"), reverse=True):
                os.chmod(path, stat.S_IMODE(path.stat().st_mode) & ~0o222)
            os.chmod(target, stat.S_IMODE(target.stat().st_mode) & ~0o222)
            frozen_paths.append(target.resolve())
        _sync_tree(frozen_root)
    except Exception:
        receipt["frozen_copied_count"] = len(frozen_paths)
        _mark_failure(receipt_path, receipt, "frozen_copy_failed")
    receipt["frozen_copied_count"] = len(frozen_paths)
    _persist_json(receipt_path, receipt)

    reloaded_count = 0
    try:
        for (original, original_step_instructions), path in zip(loaded, frozen_paths, strict=True):
            reloaded, reloaded_step_instructions = _load_task(
                path,
                task_type=Task,
                task_config_type=TaskDefinitionConfig,
                extra_instruction_paths=harbor_config.extra_instruction_paths,
                disable_verification=harbor_config.verifier.disable,
            )
            if (
                original.name != reloaded.name
                or original.config != reloaded.config
                or original.instruction != reloaded.instruction
                or original_step_instructions != reloaded_step_instructions
            ):
                raise ValueError("frozen task differs from materialized task")
            reloaded_count += 1
    except Exception:
        receipt["frozen_reloaded_count"] = reloaded_count
        _mark_failure(receipt_path, receipt, "frozen_task_reload_failed")
    receipt["frozen_reloaded_count"] = reloaded_count
    _persist_json(receipt_path, receipt)

    # Keep all other Harbor options, but replace inputs with local frozen paths.
    launch = {key: value for key, value in config.items() if key not in _MANAGED_KEYS}
    launch["jobs_dir"] = str(job_dir.parent)
    launch["job_name"] = job_dir.name
    launch["datasets"] = []
    # Do not carry a custom verifier block past the scored-run admission gate.
    launch["verifier"] = {"disable": False}
    launch["tasks"] = [
        {"path": str(path), "source": task.source}
        for path, task in zip(frozen_paths, task_configs, strict=True)
    ]
    launch_path = frozen_root / "launch-config.json"
    try:
        _persist_json(launch_path, launch)
        os.chmod(launch_path, stat.S_IMODE(launch_path.stat().st_mode) & ~0o222)
        tasks_root = frozen_root / "tasks"
        os.chmod(tasks_root, stat.S_IMODE(tasks_root.stat().st_mode) & ~0o222)
        os.chmod(frozen_root, stat.S_IMODE(frozen_root.stat().st_mode) & ~0o222)
        _sync_tree(frozen_root)
    except Exception:
        _mark_failure(receipt_path, receipt, "frozen_copy_failed")

    receipt["status"] = "accepted"
    _persist_json(receipt_path, receipt)
    return launch_path, tuple(frozen_paths)


__all__ = ["DatasetPreflightError", "prepare_terminal_runtime_dataset_launch"]
