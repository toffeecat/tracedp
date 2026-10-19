"""Run Harbor with the required managed-runtime plugin and post-run audit."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from lh_platform.benchmarks.terminal_managed_runtime_plugin import (
    finalize_terminal_managed_runtime_job,
)
from lh_platform.benchmarks.terminal_runtime_dataset_preflight import (
    DatasetPreflightError,
    prepare_terminal_runtime_dataset_launch,
)
from lh_platform.benchmarks.terminal_runtime_identity import resolve_host_pi_runtime
from lh_platform.benchmarks.terminal_verifier_contract import (
    write_job_verifier_contract,
)
from lh_platform.core.run_manifest import write_run_manifest

_PLUGIN_IMPORT = (
    "lh_platform.benchmarks.terminal_managed_runtime_plugin:ManagedRuntimeFinalizationPlugin"
)
_MANAGED_AGENT_OUTER_RESERVE_SECONDS = 60
_BACKEND_RECEIPT_SCHEMA = "terminal_managed_runtime_backend_v1"
_STORAGE_RECEIPT_SCHEMA = "terminal_managed_runtime_storage_v1"
_ACTOR_RECEIPT_SCHEMA = "terminal_managed_runtime_actor_v1"
_CONTROLLER_ONLY_CONFIG_KEYS = (
    "managed_runtime_backend",
    "managed_runtime_storage",
    "managed_runtime_actor",
    "managed_runtime_dataset_preflight",
)


def _validate_managed_actor_timeout(
    config: dict[str, object], task_paths: tuple[Path, ...]
) -> None:
    """Mirror Harbor's per-phase timeout resolution for every admitted task."""

    from harbor.models.job.config import JobConfig
    from harbor.models.task.task import Task

    launch = {
        key: value for key, value in config.items() if key not in _CONTROLLER_ONLY_CONFIG_KEYS
    }
    launch["datasets"] = []
    launch["tasks"] = [{"path": str(path), "source": None} for path in task_paths]
    launch["verifier"] = {"disable": False}
    harbor_config = JobConfig.model_validate(launch)
    if len(harbor_config.agents) != 1:
        raise ValueError("managed Harbor launch requires exactly one agent")
    agent = harbor_config.agents[0]
    host_timeout = agent.kwargs.get("host_pi_timeout_seconds")
    if isinstance(host_timeout, bool) or not isinstance(host_timeout, int) or host_timeout < 1:
        raise ValueError("managed Harbor host Pi timeout is invalid")
    multiplier = harbor_config.agent_timeout_multiplier
    if multiplier is None:
        multiplier = harbor_config.timeout_multiplier
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("managed Harbor agent timeout multiplier is invalid")
    for task_path in task_paths:
        task = Task(
            task_dir=task_path,
            extra_instruction_paths=harbor_config.extra_instruction_paths,
            disable_verification=harbor_config.verifier.disable,
        )
        phase_defaults = (
            tuple(
                step.agent.timeout_sec
                if step.agent.timeout_sec is not None
                else task.config.agent.timeout_sec
                for step in task.config.steps
            )
            if task.config.steps
            else (task.config.agent.timeout_sec,)
        )
        for default_timeout in phase_defaults:
            base_timeout = agent.override_timeout_sec or default_timeout
            if base_timeout is None:
                continue
            outer_timeout = min(
                float(base_timeout),
                float(agent.max_timeout_sec) if agent.max_timeout_sec else math.inf,
            ) * float(multiplier)
            if not math.isfinite(outer_timeout) or outer_timeout <= 0:
                raise ValueError("managed Harbor agent phase timeout must be finite and positive")
            if host_timeout + _MANAGED_AGENT_OUTER_RESERVE_SECONDS > outer_timeout:
                raise ValueError(
                    "host Pi timeout must leave 60 seconds for managed initialization, "
                    f"cleanup, and Harbor cancellation in every phase of {task_path.name}"
                )


def _read_launch_config(path: Path) -> dict[str, object]:
    """Read one in-process snapshot for every controller decision."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("managed Harbor launch config must be a JSON object")
    return value


def _read_launch_destination(config: dict[str, object]) -> tuple[Path, str]:
    """Resolve an owned Harbor output directory from the shared snapshot."""

    jobs_dir = config.get("jobs_dir")
    job_name = config.get("job_name")
    if not isinstance(jobs_dir, str) or not isinstance(job_name, str) or not job_name:
        raise ValueError("managed Harbor launch config lacks jobs_dir or job_name")
    job_component = Path(job_name)
    if (
        job_component.is_absolute()
        or job_component.name != job_name
        or job_name in {".", "..", ".managed-runtime-frozen"}
        or "/" in job_name
        or "\\" in job_name
    ):
        raise ValueError("managed Harbor job_name must be one path component")
    jobs_root = Path(jobs_dir).expanduser().resolve(strict=True)
    if not jobs_root.is_dir():
        raise ValueError("managed Harbor jobs_dir must be an existing directory")
    return jobs_root, job_name


def _require_current_interpreter(python_executable: str) -> str:
    """Keep preflight imports and Harbor CLI execution in one environment."""

    requested = Path(python_executable).expanduser()
    current = Path(sys.executable).expanduser()
    if not requested.is_absolute() or requested.absolute() != current.absolute():
        raise ValueError("managed Harbor python_executable must equal the controller interpreter")
    return str(current.absolute())


def _read_backend_contract(config: dict[str, object]) -> dict[str, str]:
    """Require an explicit Docker daemon identity for every managed launch."""

    backend = config.get("managed_runtime_backend")
    if not isinstance(backend, dict):
        raise ValueError("managed Harbor launch lacks managed_runtime_backend")
    if backend.get("type") != "docker_context":
        raise ValueError("managed runtime backend type must be docker_context")
    context = backend.get("context")
    architecture = backend.get("expected_architecture")
    if not isinstance(context, str) or not context:
        raise ValueError("managed runtime Docker context is invalid")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("managed runtime Docker architecture is invalid")
    return {"context": context, "expected_architecture": architecture}


def _read_storage_contract(config: dict[str, object]) -> Path:
    """Require private runtime storage to be explicit and independent of shell state."""

    storage = config.get("managed_runtime_storage")
    root = storage.get("root") if isinstance(storage, dict) else None
    if not isinstance(root, str) or not root:
        raise ValueError("managed Harbor launch lacks managed_runtime_storage.root")
    runtime_root = Path(root).expanduser()
    if not runtime_root.is_absolute():
        raise ValueError("managed runtime storage root must be absolute")
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root.resolve()


def _read_actor_contract(config: dict[str, object]) -> dict[str, str]:
    """Require every host-side Actor dependency that changes launch identity."""

    actor = config.get("managed_runtime_actor")
    if not isinstance(actor, dict):
        raise ValueError("managed Harbor launch lacks managed_runtime_actor")
    executable = actor.get("pi_executable")
    version = actor.get("pi_version")
    credential_env = actor.get("credential_env")
    credential_source = actor.get("credential_source")
    if not all(
        isinstance(item, str) and item
        for item in (executable, version, credential_env, credential_source)
    ):
        raise ValueError("managed runtime Actor contract is incomplete")
    if credential_source not in {"environment", "pi_auth_store"}:
        raise ValueError("managed runtime credential source is invalid")
    agents = config.get("agents")
    if not isinstance(agents, list) or len(agents) != 1 or not isinstance(agents[0], dict):
        raise ValueError("managed runtime Actor identity requires exactly one agent")
    model_name = agents[0].get("model_name")
    kwargs = agents[0].get("kwargs")
    if not isinstance(model_name, str) or "/" not in model_name or not isinstance(kwargs, dict):
        raise ValueError("managed runtime Actor model identity is incomplete")
    provider, model = model_name.split("/", maxsplit=1)
    thinking = kwargs.get("thinking")
    configured_pi_version = kwargs.get("pi_version")
    if not isinstance(thinking, str) or configured_pi_version != version:
        raise ValueError("managed runtime Actor version or thinking identity differs")
    identity = resolve_host_pi_runtime(
        provider=provider,
        model=model,
        thinking=thinking,
    )
    if credential_env != identity.credential_env:
        raise ValueError(
            "managed runtime credential binding differs: "
            f"expected {identity.credential_env}, declared {credential_env}"
        )
    executable_path = Path(executable).expanduser()
    if not executable_path.is_absolute():
        raise ValueError("managed runtime Pi executable must be absolute")
    return {
        "pi_executable": str(executable_path.resolve()),
        "pi_version": version,
        "credential_env": credential_env,
        "credential_source": credential_source,
        "provider": provider,
        "model": model,
    }


def _probe_actor_runtime(
    contract: dict[str, str], *, command_runner: Callable[..., Any]
) -> tuple[dict[str, object], str]:
    """Resolve the current credential and return a secret-free runtime receipt."""

    executable = Path(contract["pi_executable"])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("managed runtime Pi executable is unavailable")
    version_result = command_runner(
        (str(executable), "--version"),
        check=False,
        capture_output=True,
        text=True,
    )
    observed_version = str(version_result.stdout or "").strip()
    if int(version_result.returncode) != 0 or observed_version != contract["pi_version"]:
        raise ValueError(
            "managed runtime Pi version differs: "
            f"expected {contract['pi_version']}, observed {observed_version or 'unavailable'}"
        )
    credential_source = contract["credential_source"]
    if credential_source == "pi_auth_store":
        credential_result = command_runner(
            (
                str(executable),
                "auth",
                "print-api-key",
                "--provider",
                contract["provider"],
                "--model",
                contract["model"],
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        credential = str(credential_result.stdout or "").strip()
        if int(credential_result.returncode) != 0 or not credential:
            raise ValueError("managed runtime Pi credential resolution failed")
    else:
        credential = os.environ.get(contract["credential_env"], "")
        if not credential:
            raise ValueError(
                f"managed runtime credential is unavailable: {contract['credential_env']}"
            )
    return (
        {
            "schema_version": _ACTOR_RECEIPT_SCHEMA,
            "pi_executable": str(executable),
            "expected_pi_version": contract["pi_version"],
            "observed_pi_version": observed_version,
            "credential_env": contract["credential_env"],
            "credential_source": credential_source,
            "credential_present": True,
        },
        credential,
    )


def _probe_docker_backend(
    contract: dict[str, str], *, command_runner: Callable[..., Any]
) -> dict[str, object]:
    """Resolve the declared context and verify the daemon before task materialization."""

    context = contract["context"]
    endpoint_result = command_runner(
        (
            "docker",
            "context",
            "inspect",
            context,
            "--format",
            "{{json .Endpoints.docker.Host}}",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if int(endpoint_result.returncode) != 0:
        raise ValueError(f"managed runtime Docker context is unavailable: {context}")
    try:
        endpoint = json.loads(endpoint_result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("managed runtime Docker endpoint is invalid") from error
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("managed runtime Docker endpoint is invalid")

    info_result = command_runner(
        (
            "docker",
            "--context",
            context,
            "info",
            "--format",
            "{{json .}}",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if int(info_result.returncode) != 0:
        raise ValueError(f"managed runtime Docker daemon is unavailable: {context}")
    try:
        info = json.loads(info_result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("managed runtime Docker info is invalid") from error
    architecture = info.get("Architecture") if isinstance(info, dict) else None
    if architecture != contract["expected_architecture"]:
        raise ValueError(
            "managed runtime Docker architecture differs: "
            f"expected {contract['expected_architecture']}, observed {architecture}"
        )
    return {
        "schema_version": _BACKEND_RECEIPT_SCHEMA,
        "backend_type": "docker_context",
        "context": context,
        "endpoint": endpoint,
        "expected_architecture": contract["expected_architecture"],
        "observed_architecture": architecture,
        "daemon_name": info.get("Name"),
        "driver": info.get("Driver"),
        "docker_root_dir": info.get("DockerRootDir"),
        "server_version": info.get("ServerVersion"),
        "operating_system": info.get("OperatingSystem"),
    }


def _write_json_receipt(path: Path, payload: dict[str, object]) -> None:
    """Publish one complete controller receipt and sync its directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.writing")
    with staging.open("w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_exit_receipt(job_dir: Path, payload: dict[str, object]) -> None:
    """Persist controller and scientific-finalization outcomes as separate fields."""

    _write_json_receipt(job_dir / "controller-exit.json", payload)


def run_terminal_managed_runtime_harbor(
    config_path: Path,
    *,
    python_executable: str = sys.executable,
    command_runner: Callable[..., Any] = subprocess.run,
    backend_runner: Callable[..., Any] = subprocess.run,
    actor_runner: Callable[..., Any] = subprocess.run,
) -> int:
    """Require both Harbor success and an independently repeated job finalization."""

    config_path = config_path.expanduser().resolve()
    config = _read_launch_config(config_path)
    jobs_dir, job_name = _read_launch_destination(config)
    python_executable = _require_current_interpreter(python_executable)
    job_dir = jobs_dir / job_name
    preflight_job_owned = False

    def mark_preflight_job_owned() -> None:
        nonlocal preflight_job_owned
        preflight_job_owned = True

    try:
        # Admit the complete frozen dataset before inspecting Docker or Pi.
        executed_config_path, frozen_task_paths = prepare_terminal_runtime_dataset_launch(
            config,
            job_dir,
            on_job_owned=mark_preflight_job_owned,
        )
        _validate_managed_actor_timeout(config, frozen_task_paths)
        write_job_verifier_contract(job_dir, frozen_task_paths)
        backend_contract = _read_backend_contract(config)
        runtime_root = _read_storage_contract(config)
        actor_contract = _read_actor_contract(config)
        backend_receipt = _probe_docker_backend(backend_contract, command_runner=backend_runner)
        actor_receipt, actor_credential = _probe_actor_runtime(
            actor_contract, command_runner=actor_runner
        )
        _write_json_receipt(job_dir / "managed-runtime-backend-receipt.json", backend_receipt)
        _write_json_receipt(
            job_dir / "managed-runtime-storage-receipt.json",
            {
                "schema_version": _STORAGE_RECEIPT_SCHEMA,
                "root": str(runtime_root),
                "private_logs_root": str(runtime_root / "managed-runtime-private"),
            },
        )
        _write_json_receipt(job_dir / "managed-runtime-actor-receipt.json", actor_receipt)
        # Bind the run to the code revision and launch config before Harbor starts.
        write_run_manifest(
            job_dir,
            entry="terminal_managed_runtime_harbor",
            config_path=config_path,
            run_root=jobs_dir,
        )
    except (KeyboardInterrupt, SystemExit):
        if preflight_job_owned:
            try:
                _write_exit_receipt(
                    job_dir,
                    {
                        "schema_version": "terminal_managed_runtime_controller_exit_v2",
                        "harbor_exit_code": None,
                        "finalization_status": "interrupted",
                        "exit_code": None,
                    },
                )
            except Exception:
                pass
        raise
    except Exception as error:
        exit_receipt: dict[str, object] = {
            "schema_version": "terminal_managed_runtime_controller_exit_v2",
            "harbor_exit_code": None,
            "finalization_status": "preflight_rejected",
            "finalization_error_type": type(error).__name__,
            "finalization_reason": str(error),
            "exit_code": 2,
        }
        if isinstance(error, DatasetPreflightError):
            failure_field = (
                "runtime_failure_code"
                if error.code == "runtime_dependency_incompatible"
                else "dataset_failure_code"
            )
            exit_receipt[failure_field] = error.code
        if preflight_job_owned:
            _write_exit_receipt(job_dir, exit_receipt)
        return 2
    command: Sequence[str] = (
        python_executable,
        "-c",
        "from harbor.cli.main import app; app()",
        "run",
        "--config",
        str(executed_config_path),
        "--plugin",
        _PLUGIN_IMPORT,
        "--yes",
    )
    launch_environment = os.environ.copy()
    # A stale DOCKER_HOST overrides context routing in Docker clients.
    launch_environment.pop("DOCKER_HOST", None)
    launch_environment["DOCKER_CONTEXT"] = backend_contract["context"]
    launch_environment["LONGH_RUNTIME_ROOT"] = str(runtime_root)
    launch_environment["LONGH_PI_EXECUTABLE"] = actor_contract["pi_executable"]
    launch_environment[actor_contract["credential_env"]] = actor_credential
    try:
        completed = command_runner(command, check=False, env=launch_environment)
        harbor_exit_code = int(completed.returncode)
    except (KeyboardInterrupt, SystemExit):
        try:
            # Preserve the original interruption if the best-effort receipt fails.
            _write_exit_receipt(
                job_dir,
                {
                    "schema_version": "terminal_managed_runtime_controller_exit_v2",
                    "harbor_exit_code": None,
                    "finalization_status": "interrupted",
                    "exit_code": None,
                },
            )
        except Exception:
            pass
        raise
    except Exception as error:
        _write_exit_receipt(
            job_dir,
            {
                "schema_version": "terminal_managed_runtime_controller_exit_v2",
                "harbor_exit_code": None,
                "finalization_status": "launch_failed",
                "finalization_error_type": type(error).__name__,
                "finalization_reason": str(error),
                "exit_code": 2,
            },
        )
        return 2
    if harbor_exit_code != 0:
        _write_exit_receipt(
            job_dir,
            {
                "schema_version": "terminal_managed_runtime_controller_exit_v2",
                "harbor_exit_code": harbor_exit_code,
                "finalization_status": "not_run",
                "exit_code": harbor_exit_code,
            },
        )
        return harbor_exit_code

    try:
        job_receipt = finalize_terminal_managed_runtime_job(job_dir)
    except Exception as error:
        _write_exit_receipt(
            job_dir,
            {
                "schema_version": "terminal_managed_runtime_controller_exit_v2",
                "harbor_exit_code": 0,
                "finalization_status": "rejected",
                "finalization_error_type": type(error).__name__,
                "finalization_reason": str(error),
                "exit_code": 2,
            },
        )
        return 2

    _write_exit_receipt(
        job_dir,
        {
            "schema_version": "terminal_managed_runtime_controller_exit_v2",
            "harbor_exit_code": 0,
            "finalization_status": "completed",
            "job_receipt_schema": job_receipt["schema_version"],
            "exit_code": 0,
        },
    )
    return 0


__all__ = ["run_terminal_managed_runtime_harbor"]
