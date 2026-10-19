"""Clone a quiescent Docker filesystem for development-only candidate trials.

This provider does not clone process memory, mounted data, or remote services.
It reuses Harbor's real Docker environment for execution and file transfer.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkPolicy, TaskOS
from harbor.models.trial.paths import TrialPaths

IMPLEMENTATION_STATUS = "implementation_unverified"
_LABEL = "org.longh.candidate-lease"
_LOG_TARGETS = {"/logs/agent", "/logs/verifier", "/logs/artifacts"}
_IDLE_COMMANDS = {
    "sleep infinity",
    "sh -c sleep infinity",
    "/bin/sh -c sleep infinity",
    "tail -f /dev/null",
    "/bin/sh -c tail -f /dev/null",
}


@dataclass(frozen=True)
class DockerCandidateConfiguration:
    """Freeze the local daemon and infrastructure deadline for one Host run."""

    docker_context: str
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if not self.docker_context or self.timeout_seconds < 1:
            raise ValueError("candidate Docker context and timeout are required")


def declared_mount_targets(mounts: object) -> set[str]:
    """Read the container paths declared by Harbor's mount entries.

    Harbor annotates ``mounts`` as ``list[ServiceVolumeConfig]``, and
    ``ServiceVolumeConfig`` is a ``TypedDict``.  Each entry is therefore a plain
    mapping with a ``target`` key at runtime, not an object with a ``target``
    attribute.  The shape is validated here so a Harbor change fails loudly
    instead of silently producing an empty set that skips the mount audit.
    """

    if not isinstance(mounts, Sequence) or isinstance(mounts, (str, bytes)):
        raise ValueError("Harbor mount declarations must be a sequence")
    targets: set[str] = set()
    for index, mount in enumerate(mounts):
        if not isinstance(mount, Mapping):
            raise ValueError(f"Harbor mount declaration {index} is not a mapping")
        target = mount.get("target")
        if not isinstance(target, str) or not target:
            raise ValueError(f"Harbor mount declaration {index} lacks a target path")
        targets.add(target)
    return targets


def audit_source_mounts(
    inspect: dict[str, Any], *, declared_log_targets: set[str]
) -> list[dict[str, object]]:
    """Reject data omitted by commit, allowing only explicitly excluded logs."""
    if inspect["Config"].get("Volumes"):
        raise ValueError("image VOLUME declarations cannot be cloned by this provider")
    excluded = []
    for mount in inspect.get("Mounts", []):
        target = mount["Destination"]
        if target not in _LOG_TARGETS or target not in declared_log_targets:
            raise ValueError(f"current-state clone cannot retain mounted input: {target}")
        excluded.append({"target": target, "writable": bool(mount["RW"])})
    return excluded


def audit_quiescent_processes(output: str) -> list[str]:
    """Require idle keepalive or zombie processes after the source is paused."""
    commands = []
    for row in output.splitlines()[1:]:
        fields = row.strip().split(None, 2)
        if not fields:
            continue
        _, state, command = fields
        if not state.startswith("Z") and command not in _IDLE_COMMANDS:
            raise ValueError(f"live process state cannot be cloned: {command}")
        commands.append(command)
    if not commands:
        raise ValueError("source process inventory is empty")
    return commands


class _CandidateDockerEnvironment(DockerEnvironment):
    """Use upstream operations while making the one-use image part of cleanup."""

    def __init__(self, *, provider: DockerCurrentStateCandidateProvider, lease: dict, **kwargs):
        self.provider = provider
        self.lease = lease
        super().__init__(**kwargs)

    @property
    def environment_id(self) -> str:
        return self.session_id

    def _compose_env_vars(self, include_os_env: bool = True) -> dict[str, str]:
        values = super()._compose_env_vars(include_os_env=include_os_env)
        values.pop("DOCKER_HOST", None)
        values["DOCKER_CONTEXT"] = self.provider.configuration.docker_context
        return values

    async def exec(self, *args, **kwargs):
        """Destroy the trial on timeout/cancel so child processes cannot linger."""
        try:
            return await super().exec(*args, **kwargs)
        except BaseException:
            await self.stop(delete=True)
            raise

    async def stop(self, delete: bool = True) -> None:
        """Candidate environments are always disposable, including their image."""
        if not delete:
            raise ValueError("candidate environment cannot be retained for reuse")
        try:
            if self.lease["phase"] != "closed":
                await super().stop(delete=True)
        finally:
            await self.provider._cleanup(self.lease)


class DockerCurrentStateCandidateProvider:
    """Own a serialized Docker clone lifecycle and recover abandoned resources."""

    def __init__(
        self,
        environment: DockerEnvironment,
        *,
        configuration: DockerCandidateConfiguration,
        state_dir: Path,
    ) -> None:
        if not isinstance(environment, DockerEnvironment):
            raise ValueError("current-state provider requires Harbor DockerEnvironment")
        if environment.task_env_config.os != TaskOS.LINUX:
            raise ValueError("current-state provider supports Linux containers only")
        if os.environ.get("DOCKER_CONTEXT") != configuration.docker_context or os.environ.get(
            "DOCKER_HOST"
        ):
            raise ValueError("Host Docker daemon binding differs from candidate configuration")
        self.environment = environment
        self.configuration = configuration
        self.state_dir = state_dir
        self._lock_file = None
        self._leases: dict[str, dict] = {}
        self._clone_lock = asyncio.Lock()

    def _persist(self, lease: dict, phase: str) -> None:
        """Write ownership before the corresponding external state transition."""
        lease["phase"] = phase
        path = self.state_dir / f"{lease['lease_id']}.json"
        temporary = path.with_suffix(".pending")
        temporary.write_text(json.dumps(lease, indent=2) + "\n")
        temporary.replace(path)

    async def _docker(self, *arguments: str, check: bool = True) -> str:
        """Pin every Docker operation to the frozen local daemon."""
        process = await asyncio.create_subprocess_exec(
            "docker", "--context", self.configuration.docker_context, *arguments,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), self.configuration.timeout_seconds
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if process.returncode:
            error = stderr.decode(errors="replace")
            if not check and ("No such image:" in error or "No such container:" in error):
                return ""
            raise RuntimeError(
                f"candidate Docker {arguments[0]} failed: {error}"
            )
        return stdout.decode()

    async def open(self) -> None:
        """Recover prior crashes only after acquiring exclusive lifecycle ownership."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.state_dir / "provider.lock").open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config_path = self.state_dir / "provider.json"
            expected = asdict(self.configuration)
            if config_path.exists() and json.loads(config_path.read_text()) != expected:
                raise ValueError("candidate provider configuration changed during resume")
            config_path.write_text(json.dumps(expected, indent=2) + "\n")
            endpoint = json.loads(await self._docker("context", "inspect"))[0]["Endpoints"][
                "docker"
            ]["Host"]
            if not endpoint.startswith("unix://"):
                raise ValueError("candidate provider requires a local Unix Docker socket")
            for path in self.state_dir.glob("lease-*.json"):
                lease = json.loads(path.read_text())
                if lease["phase"] != "closed":
                    await self._cleanup(lease)
        except BaseException:
            self._lock_file.close()
            self._lock_file = None
            raise

    async def close(self) -> None:
        """Clean every owned trial without stopping the main environment."""
        try:
            for lease in self._leases.values():
                if lease["phase"] != "closed":
                    await self._cleanup(lease)
        finally:
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None

    async def _unpause_source(self, lease: dict) -> None:
        """Only undo this lease's pause on the same source container incarnation."""
        if not lease.get("source_pause_owned", False):
            return
        raw = await self._docker("container", "inspect", lease["source_container"], check=False)
        if raw:
            state = json.loads(raw)[0]["State"]
            if state["StartedAt"] == lease["source_started_at"] and state["Paused"]:
                await self._docker("unpause", lease["source_container"])
        lease["source_pause_owned"] = False
        self._persist(lease, lease["phase"])

    async def _cleanup(self, lease: dict) -> None:
        """Delete only the lease-labelled project/image and verify absence."""
        started = time.monotonic()
        await self._unpause_source(lease)
        project = lease["project_name"]
        containers = (await self._docker(
            "ps", "-aq", "--filter", f"label={_LABEL}={lease['lease_id']}"
        )).split()
        if containers:
            await self._docker("rm", "--force", "--volumes", *containers)
        if (await self._docker(
            "ps", "-aq", "--filter", f"label={_LABEL}={lease['lease_id']}"
        )).strip():
            raise RuntimeError("candidate container cleanup did not finish")
        networks = (await self._docker(
            "network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"
        )).split()
        if networks:
            await self._docker("network", "rm", *networks)
        await self._docker("image", "rm", lease["image_tag"], check=False)
        remaining = await self._docker("image", "inspect", lease["image_tag"], check=False)
        if remaining:
            raise RuntimeError("candidate image cleanup did not finish")
        lease["cleanup_duration_ms"] = int((time.monotonic() - started) * 1000)
        self._persist(lease, "closed")

    async def create_isolated_candidate_environment(self, *, session_id: str):
        """Commit the current quiescent layer, then start a network-free trial."""
        if self._lock_file is None:
            raise RuntimeError("candidate provider must be opened before cloning")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,100}", session_id):
            raise ValueError("candidate session_id must be a valid unique compose name")
        if session_id == self.environment.session_id or (await self._docker(
            "ps", "-aq", "--filter", f"label=com.docker.compose.project={session_id}"
        )).strip():
            raise ValueError("candidate session_id is already used by an environment")
        async with self._clone_lock:
            return await self._create(session_id)

    async def _create(self, session_id: str):
        started = time.monotonic()
        services = await self.environment._run_docker_compose_command(["config", "--services"])
        if (services.stdout or "").split() != ["main"]:
            raise ValueError("current-state clone does not support sidecar services")
        result = await self.environment._run_docker_compose_command(
            ["ps", "--format", "{{.Names}}", "main"]
        )
        names = (result.stdout or "").split()
        if len(names) != 1:
            raise ValueError("current-state clone requires one running main container")
        source = json.loads(await self._docker("container", "inspect", names[0]))[0]
        if not source["State"]["Running"] or source["State"]["Paused"]:
            raise ValueError("source main container must be running and unpaused")
        declared_logs = declared_mount_targets(self.environment._mounts)
        excluded_mounts = audit_source_mounts(source, declared_log_targets=declared_logs)
        lease_id = f"lease-{uuid4().hex}"
        lease = {
            "lease_id": lease_id, "project_name": session_id,
            "source_container": names[0], "source_started_at": source["State"]["StartedAt"],
            "image_tag": f"longh-candidate:{lease_id}",
            "filesystem_scope": "entire_container_layer_except_mounts",
            "excluded_mounts": excluded_mounts,
            "process_memory_copied": False, "network_mode": "none",
            "shared_writable_mounts": False,
            "source_pause_owned": True,
        }
        self._leases[lease_id] = lease
        self._persist(lease, "pause_requested")
        try:
            await self._docker("pause", names[0])
            self._persist(lease, "paused")
            processes = await self._docker("top", names[0], "-eo", "pid,stat,args")
            lease["quiescent_processes"] = audit_quiescent_processes(processes)
            self._persist(lease, "commit_requested")
            await self._docker("commit", "--pause=false", names[0], lease["image_tag"])
            self._persist(lease, "committed")
            await self._unpause_source(lease)
            directory = self.state_dir / lease_id
            directory.mkdir()
            # Never reuse task compose: it could reinstate volumes, services or startup work.
            (directory / "docker-compose.yaml").write_text(json.dumps({"services": {
                "main": {
                    "entrypoint": ["/bin/sh", "-c"], "command": ["sleep infinity"],
                    "network_mode": "none", "healthcheck": {"disable": True},
                    "labels": {_LABEL: lease_id}, "restart": "no",
                }
            }}))
            child = _CandidateDockerEnvironment(
                provider=self, lease=lease, environment_dir=directory,
                environment_name=session_id, session_id=session_id,
                trial_paths=TrialPaths(trial_dir=directory / "logs"),
                task_env_config=self.environment.task_env_config.model_copy(update={
                    "docker_image": lease["image_tag"],
                    "workdir": source["Config"]["WorkingDir"] or None,
                    "env": {},
                }),
                persistent_env=deepcopy(self.environment._persistent_env),
                mounts=[], network_policy=NetworkPolicy(),
                override_cpus=self.environment._effective_cpus,
                override_memory_mb=self.environment._effective_memory_mb,
            )
            child.default_user = self.environment.default_user
            self._persist(lease, "start_requested")
            await asyncio.wait_for(child.start(force_build=False), self.configuration.timeout_seconds)
            child_names = await self._docker(
                "ps", "--format", "{{.Names}}", "--filter", f"label={_LABEL}={lease_id}"
            )
            clone = json.loads(await self._docker("container", "inspect", child_names.strip()))[0]
            if clone["Mounts"] or clone["HostConfig"]["NetworkMode"] != "none":
                raise ValueError("candidate started with a mount or network access")
            lease["candidate_container"] = child_names.strip()
            lease["creation_duration_ms"] = int((time.monotonic() - started) * 1000)
            self._persist(lease, "ready")
            return child
        except BaseException:
            await self._cleanup(lease)
            raise


class DockerCandidateEnvironmentBridge:
    """Preserve the main environment API and add one explicit optional capability."""

    def __init__(self, provider: DockerCurrentStateCandidateProvider) -> None:
        self.provider = provider

    def __getattr__(self, name: str):
        return getattr(self.provider.environment, name)

    async def create_isolated_candidate_environment(self, *, session_id: str):
        return await self.provider.create_isolated_candidate_environment(session_id=session_id)
