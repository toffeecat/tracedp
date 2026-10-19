"""Start a historical continuation from a verified container snapshot instead of a replay.

Replaying a source prefix re-executes every recorded tool call. A step whose outcome
depends on the outside world (``pip install`` timing out in the source, succeeding later)
then makes the replay diverge, and the exact-parity check rightly refuses the run.

Instead, a cell's prefix is replayed once; only if that replay matches the source call by
call is the container committed to an image. Every arm then starts from that image, so
all arms provably begin in one identical failure state and no run replays anything.

At start an arm checks two things before continuing: the container runs the committed
image, and every public path observes exactly as it did when the snapshot was taken.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lh_platform.benchmarks.terminal_public_state import public_file_observation_payload

SNAPSHOT_RECEIPT_SCHEMA = "terminal_prefix_snapshot_v1"
DEFAULT_DOCKER_CONTEXT = "default"


def docker_command(*args: str) -> list[str]:
    """Docker CLI bound to the experiment daemon, never the user's current context."""

    context = os.environ.get("LONGH_DOCKER_CONTEXT") or os.environ.get("DOCKER_CONTEXT") or DEFAULT_DOCKER_CONTEXT
    return ["docker", "--context", context, *args]


async def _run(args: list[str], *, timeout: float = 900.0) -> str:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"docker command timed out: {' '.join(args[:4])}") from None
    if process.returncode != 0:
        raise RuntimeError(f"docker command failed ({process.returncode}): {stderr.decode()[:400]}")
    return stdout.decode().strip()


async def container_id(environment: Any) -> str:
    """The running container's id, read from inside it (Docker sets the hostname to it)."""

    result = await environment.exec(command="cat /etc/hostname", timeout_sec=30)
    if getattr(result, "return_code", 1) != 0:
        raise RuntimeError("cannot read the container hostname")
    hostname = str(result.stdout or "").strip()
    if not hostname or any(c not in "0123456789abcdef" for c in hostname):
        raise RuntimeError(f"container hostname is not a docker id: {hostname!r}")
    return hostname


# `docker commit` copies the container's labels into the image. The Compose project label
# then makes `docker compose down --rmi local` delete the snapshot when its job ends.
# Rebinding the
# Compose labels detaches the image from every job, so no teardown can claim it.
SNAPSHOT_COMPOSE_PROJECT = "longh-prefix-snapshot"


async def commit_container(container: str, image_tag: str, *, extra_changes: Sequence[str] = ()) -> str:
    """Commit the container's filesystem as ``image_tag``; return the image id.

    ``extra_changes`` are further ``--change`` arguments (a materialized ladder rung
    restores its base image's entrypoint and command this way).
    """

    changes = list(extra_changes)
    for label, value in (
        ("com.docker.compose.project", SNAPSHOT_COMPOSE_PROJECT),
        ("com.docker.compose.service", "snapshot"),
        ("com.docker.compose.project.working_dir", "/"),
        ("com.docker.compose.project.config_files", ""),
    ):
        changes += ["--change", f'LABEL {label}="{value}"']
    await _run(docker_command("commit", *changes, container, image_tag))
    project = await _run(docker_command(
        "image", "inspect", "--format", '{{index .Config.Labels "com.docker.compose.project"}}', image_tag
    ))
    if project != SNAPSHOT_COMPOSE_PROJECT:
        raise RuntimeError("snapshot image still carries its job's Compose project label")
    return await _run(docker_command("image", "inspect", "--format", "{{.Id}}", image_tag))


async def running_image_id(container: str) -> str:
    return await _run(docker_command("inspect", "--format", "{{.Image}}", container), timeout=60)


def public_state_payload(observations: Sequence[Any]) -> list[dict[str, object]]:
    """The observer's own serialization of a public-state snapshot, for exact comparison."""

    return [
        dict(item) if isinstance(item, Mapping) else public_file_observation_payload(item)
        for item in observations
    ]


def load_snapshot_receipt(path: str | Path) -> dict[str, object]:
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt.get("schema_version") != SNAPSHOT_RECEIPT_SCHEMA:
        raise ValueError("prefix snapshot receipt has an unknown schema")
    for key in ("image_tag", "image_id", "public_state", "history", "parity_policy"):
        if key not in receipt:
            raise ValueError(f"prefix snapshot receipt lacks {key}")
    if receipt["parity_policy"] != "exact_v1":
        raise ValueError("a prefix snapshot must come from an exact-parity replay")
    return receipt


async def verify_snapshot_start(
    *, environment: Any, gateway: Any, receipt: Mapping[str, object]
) -> dict[str, object]:
    """Refuse to continue unless this container is the snapshot, observed unchanged."""

    container = await container_id(environment)
    image = await running_image_id(container)
    if image != receipt["image_id"]:
        raise ValueError("container does not run the prefix snapshot image")
    observed = public_state_payload(await gateway._snapshot(0))
    if observed != receipt["public_state"]:
        raise ValueError("public state differs from the prefix snapshot")
    return {"container": container, "image_id": image, "public_paths_compared": len(observed)}


async def restore_prefix_from_snapshot(
    *,
    environment: Any,
    gateway: Any,
    receipt_path: str,
    prefix_tool_calls: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Verify the snapshot start and return the history its replay produced.

    The history must be the replay of exactly this prefix: same calls, same arguments, same
    error status as the source. The caller then treats it as the replay it skipped.
    """

    receipt = load_snapshot_receipt(receipt_path)
    history = tuple(dict(row) for row in receipt["history"])  # type: ignore[union-attr]
    rollback = receipt.get("rollback_after_call")
    if rollback is not None:
        # A rollback snapshot is the state before call `rollback + 1`; the Actor keeps its
        # full history, the environment is the earlier state.
        if receipt.get("full_prefix_length") != len(prefix_tool_calls) or len(history) != rollback:
            raise ValueError("rollback snapshot does not belong to this prefix")
        prefix_tool_calls = prefix_tool_calls[:rollback]
    if len(history) != len(prefix_tool_calls):
        raise ValueError("prefix snapshot covers a different number of calls")
    for row, call in zip(history, prefix_tool_calls, strict=True):
        if (
            row.get("tool_name") != call.get("tool_name")
            or row.get("arguments") != call.get("arguments")
            or bool(row.get("is_error")) != bool(call.get("source_is_error"))
        ):
            raise ValueError("prefix snapshot history differs from the source prefix")
    verification = await verify_snapshot_start(environment=environment, gateway=gateway, receipt=receipt)
    return history, {
        "schema_version": "terminal_prefix_snapshot_start_v1",
        "receipt": str(receipt_path),
        "image_tag": receipt["image_tag"],
        **({"rollback_after_call": rollback} if rollback is not None else {}),
        **verification,
    }


# --- Closed-loop supervisor ------------------------------


async def commit_probe_image(container: str, image_tag: str) -> str:
    """Freeze the live container into a throwaway image, so probes never touch the run."""

    return await commit_container(container, image_tag)


async def remove_image(image_tag: str) -> None:
    await _run(docker_command("image", "rm", "-f", image_tag), timeout=300)


async def restore_files(container: str, source_image: str, paths: Sequence[str], *, staging: Path) -> list[str]:
    """Copy ``paths`` from ``source_image`` into the live container, leaving every other file.

    A local repair of the lost inputs only (knowledge base ch. 5: repair the current
    residual state; widen to a rollback only when a local repair cannot restore it).
    """

    source = await _run(docker_command("create", "--entrypoint", "sh", source_image, "-c", "true"), timeout=300)
    restored = []
    try:
        staging.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(paths):
            local = staging / f"{index:03d}_{Path(path).name}"
            await _run(docker_command("cp", f"{source}:{path}", str(local)), timeout=600)
            await _run(docker_command("cp", str(local), f"{container}:{path}"), timeout=600)
            restored.append(path)
    finally:
        await _run(docker_command("rm", "-f", source), timeout=120)
    return restored
