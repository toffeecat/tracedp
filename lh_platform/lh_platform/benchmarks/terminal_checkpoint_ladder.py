"""Checkpoint ladder: a verified filesystem state every ``k`` calls of a zero-model replay.

Long cells (150+ calls) cannot be rolled back or bisected by replaying from the first call
every time. The ladder records the replayed state at every ``k``-th call, so a rollback or
bisection replays at most ``k`` calls from the nearest rung.

Rungs are deltas, not images: a committed image per rung would fill the experiment disk.
A rung is the set of paths the
container added or changed relative to its base image (``docker diff``), streamed out of
``docker export`` into a tar on experiment storage, plus the list of deleted paths and a
digest of every kept file. The base image itself is saved once per image id. A rung is
turned back into an image only when a run needs it, and its digests are checked then.

Like the prefix snapshot, a rung is filesystem only: running processes are not part of it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from lh_platform.benchmarks.terminal_prefix_snapshot import (
    SNAPSHOT_RECEIPT_SCHEMA,
    _run,
    commit_container,
    docker_command,
)

RUNG_SCHEMA = "terminal_checkpoint_rung_v1"
LADDER_SCHEMA = "terminal_checkpoint_ladder_v1"
WRITE_TRACE_SCHEMA = "terminal_call_write_trace_v1"
_SKIPPED_ROOTS = ("/proc/", "/sys/", "/dev/")


def ladder_points(call_count: int, every: int) -> tuple[int, ...]:
    """0-based call indices after which a rung is taken: every ``every``-th call and the last."""

    if every < 1:
        raise ValueError("a ladder needs a positive rung interval")
    if call_count < 1:
        return ()
    points = list(range(every - 1, call_count, every))
    if points[-1:] != [call_count - 1]:
        points.append(call_count - 1)
    return tuple(points)


def nearest_rung(points: Sequence[int], call: int) -> int | None:
    """The highest rung at or before ``call`` (the state after that call), if any."""

    eligible = [point for point in points if point <= call]
    return max(eligible) if eligible else None


def parse_docker_diff(text: str) -> dict[str, str]:
    """``docker diff`` lines (``A /x``, ``C /y``, ``D /z``) as path -> kind."""

    changes: dict[str, str] = {}
    for line in text.splitlines():
        kind, _, path = line.strip().partition(" ")
        if kind in ("A", "C", "D") and path.startswith("/"):
            changes[path] = kind
    return changes


def _member_path(name: str) -> str:
    return "/" + name.lstrip("./").rstrip("/")


class _HashingReader:
    def __init__(self, stream: IO[bytes]) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def filter_export(stream: IO[bytes], keep: set[str], out: IO[bytes]) -> dict[str, object]:
    """Copy the members of an exported filesystem tar whose path is in ``keep``.

    Returns counts and a sha256 per kept regular file. A hard link whose target is not
    kept is stored as a regular file copy of nothing and counted, never silently dropped.
    """

    digests: dict[str, str] = {}
    counts = {"scanned": 0, "files": 0, "dirs": 0, "links": 0, "other": 0, "bytes": 0, "unresolved_hardlinks": 0}
    written: set[str] = set()
    with tarfile.open(fileobj=stream, mode="r|") as source, tarfile.open(fileobj=out, mode="w|") as target:
        for member in source:
            counts["scanned"] += 1
            path = _member_path(member.name)
            if path not in keep or path.startswith(_SKIPPED_ROOTS):
                continue
            if member.isreg():
                reader = _HashingReader(source.extractfile(member))  # type: ignore[arg-type]
                target.addfile(member, reader)
                digests[path] = reader.digest.hexdigest()
                counts["files"] += 1
                counts["bytes"] += member.size
            elif member.islnk():
                if _member_path(member.linkname) not in written:
                    counts["unresolved_hardlinks"] += 1
                    continue
                target.addfile(member)
                counts["links"] += 1
            elif member.isdir():
                target.addfile(member)
                counts["dirs"] += 1
            elif member.issym():
                target.addfile(member)
                counts["links"] += 1
            else:
                counts["other"] += 1
                continue
            written.add(path)
    return {"counts": counts, "digests": digests, "kept_paths": sorted(written)}


def _export_filtered(container: str, keep: set[str], out_path: Path) -> dict[str, object]:
    process = subprocess.Popen(docker_command("export", container), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with out_path.open("wb") as out:
            result = filter_export(process.stdout, keep, out)  # type: ignore[arg-type]
        # Drain whatever the filter did not consume so export can finish.
        if process.stdout is not None:
            while process.stdout.read(1 << 20):
                pass
        _, stderr = process.communicate(timeout=600)
    finally:
        if process.poll() is None:
            process.kill()
    if process.returncode != 0:
        raise RuntimeError(f"docker export failed ({process.returncode}): {stderr.decode()[:400]}")
    return result


async def registry_digest(image_id: str) -> str | None:
    """A ``repo@sha256:...`` reference that re-pulls exactly this image, if it has one."""

    raw = await _run(docker_command("image", "inspect", "--format", "{{json .RepoDigests}}", image_id), timeout=120)
    digests = json.loads(raw or "null") or []
    return str(digests[0]) if digests else None


async def record_base_image(image_id: str, cache_dir: Path) -> dict[str, object]:
    """How the base image can be had again: a registry digest, else a saved tar.

    Official task images come from a registry, so their digest is enough and nothing is
    copied; only a locally built base is saved (``docker save`` through the socket).
    """

    started = time.monotonic()
    digest = await registry_digest(image_id)
    if digest is not None:
        return {"image_id": image_id, "repo_digest": digest, "saved_tar": None, "saved_tar_bytes": 0,
                "save_seconds": round(time.monotonic() - started, 3)}
    target = await save_base_image(image_id, cache_dir)
    return {"image_id": image_id, "repo_digest": None, "saved_tar": str(target),
            "saved_tar_bytes": target.stat().st_size, "save_seconds": round(time.monotonic() - started, 3)}


def _tar_filtered(container: str, keep: set[str], out_path: Path) -> dict[str, object] | None:
    """Pack only ``keep`` with the container's own GNU tar; ``None`` when that is not possible.

    Streams only the changed paths instead of the whole filesystem. Any non-zero tar exit
    falls back to the export.
    """

    paths = sorted(path.lstrip("/") for path in keep if "\n" not in path)
    if not paths:
        return None
    # stderr goes to a file: a pipe nobody reads can fill and block tar. stdin is written
    # and closed before reading, so only `wait` is called afterwards (`communicate` would
    # flush the closed stdin and raise).
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            docker_command("exec", "-i", container, "tar", "-C", "/", "-cf", "-", "--no-recursion", "-T", "-"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(("\n".join(paths) + "\n").encode())
            process.stdin.close()
            with out_path.open("wb") as out:
                result = filter_export(process.stdout, keep, out)
            while process.stdout.read(1 << 20):
                pass
            process.wait(timeout=600)
        except (tarfile.TarError, subprocess.TimeoutExpired, OSError, ValueError):
            process.kill()
            process.wait()
            return None
    if process.returncode != 0:
        return None
    return result


async def save_base_image(image_id: str, cache_dir: Path) -> Path:
    """``docker save`` the base image once per id onto experiment storage."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (image_id.replace(":", "_") + ".tar")
    if target.is_file():
        return target
    partial = target.with_suffix(".tar.partial")
    await _run(docker_command("save", "-o", str(partial), image_id), timeout=3600)
    partial.rename(target)
    return target


async def capture_rung(
    container: str,
    *,
    base_image_id: str,
    out_dir: Path,
    after_call: int,
    public_state: Sequence[Mapping[str, object]],
    history: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Store the container's delta against the base image.

    The changed set comes from ``docker diff``; the contents are packed by the container's
    own tar when it can (between replayed calls nothing of the Actor runs, but a service it
    started might), otherwise from a paused full export.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    changes = parse_docker_diff(await _run(docker_command("diff", container), timeout=600))
    keep = {path for path, kind in changes.items() if kind in ("A", "C")}
    deleted = sorted(path for path, kind in changes.items() if kind == "D")
    tar_path = out_dir / f"rung_{after_call:03d}.tar"
    exported = await asyncio.to_thread(_tar_filtered, container, keep, tar_path)
    method = "in_container_tar"
    if exported is None:
        method = "paused_export"
        await _run(docker_command("pause", container), timeout=120)
        try:
            exported = await asyncio.to_thread(_export_filtered, container, keep, tar_path)
        finally:
            await _run(docker_command("unpause", container), timeout=120)
    receipt = {
        "schema_version": RUNG_SCHEMA,
        "after_call": after_call,
        "calls_kept": after_call + 1,
        "base_image_id": base_image_id,
        "delta_tar": str(tar_path),
        "delta_tar_bytes": tar_path.stat().st_size,
        "changed_paths": len(keep),
        "deleted_paths": deleted,
        "digests": exported["digests"],
        "counts": exported["counts"],
        "public_state": [dict(item) for item in public_state],
        "history": [dict(row) for row in history],
        "capture_seconds": round(time.monotonic() - started, 3),
        "capture_method": method,
    }
    (out_dir / f"rung_{after_call:03d}.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


async def _image_config(image: str) -> dict[str, object]:
    raw = await _run(docker_command("image", "inspect", "--format", "{{json .Config}}", image), timeout=120)
    return json.loads(raw or "{}")


async def ensure_base_image(base_image_id: str, base_tar: Path | None, repo_digest: str | None = None) -> None:
    try:
        await _run(docker_command("image", "inspect", "--format", "{{.Id}}", base_image_id), timeout=120)
        return
    except RuntimeError:
        pass
    if repo_digest:
        await _run(docker_command("pull", repo_digest), timeout=3600)
    elif base_tar is not None and base_tar.is_file():
        await _run(docker_command("load", "-i", str(base_tar)), timeout=3600)
    else:
        raise RuntimeError("the rung's base image is gone and there is no digest or saved copy")
    found = await _run(docker_command("image", "inspect", "--format", "{{.Id}}", base_image_id), timeout=120)
    if found != base_image_id:
        raise RuntimeError("the restored base image is not the one the rung was captured on")


async def materialize_rung(
    rung: Mapping[str, object],
    *,
    image_tag: str,
    base_tar: Path | None = None,
    repo_digest: str | None = None,
    verify: bool = True,
) -> dict[str, object]:
    """Rebuild the rung's state as image ``image_tag``; optionally check every digest.

    The image keeps the base image's entrypoint and command, so a derived task built on it
    starts exactly like one built on a committed prefix snapshot.
    """

    base = str(rung["base_image_id"])
    started = time.monotonic()
    await ensure_base_image(base, base_tar, repo_digest)
    config = await _image_config(base)
    entrypoint = config.get("Entrypoint")
    # `docker commit --change 'ENTRYPOINT []'` does not clear an entrypoint (the image kept
    # ["sleep"] and Harbor's `sh -c "sleep infinity"` failed).
    # So the base entrypoint is overridden only when it exists, and restored non-empty.
    if entrypoint:
        create = ("create", "--entrypoint", "sleep", base, "3600")
    else:
        create = ("create", base, "sleep", "3600")
    container = await _run(docker_command(*create), timeout=300)
    try:
        await _run(docker_command("start", container), timeout=120)
        deleted = [str(path) for path in rung.get("deleted_paths", [])]  # type: ignore[union-attr]
        for start in range(0, len(deleted), 200):
            await _run(docker_command("exec", container, "rm", "-rf", "--", *deleted[start : start + 200]), timeout=600)
        with open(str(rung["delta_tar"]), "rb") as delta:
            process = await asyncio.create_subprocess_exec(
                *docker_command("cp", "-a", "-", f"{container}:/"),
                stdin=delta,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"docker cp into the rung container failed: {stderr.decode()[:400]}")
        await _run(docker_command("stop", "-t", "1", container), timeout=120)
        changes = []
        if entrypoint:
            changes += ["--change", f"ENTRYPOINT {json.dumps(entrypoint)}"]
        if config.get("Cmd"):
            changes += ["--change", f"CMD {json.dumps(config['Cmd'])}"]
        image_id = await commit_container(container, image_tag, extra_changes=changes)
    finally:
        await _run(docker_command("rm", "-f", container), timeout=120)
    result: dict[str, object] = {
        "image_tag": image_tag,
        "image_id": image_id,
        "materialize_seconds": round(time.monotonic() - started, 3),
    }
    if verify:
        result["verification"] = await verify_materialized(image_tag, rung)
    return result


async def verify_materialized(image_tag: str, rung: Mapping[str, object]) -> dict[str, object]:
    """Every kept file has its captured digest and every deleted path is absent."""

    container = await _run(docker_command("create", "--entrypoint", "sleep", image_tag, "1"), timeout=300)
    try:
        expected: dict[str, str] = dict(rung["digests"])  # type: ignore[arg-type]
        deleted = set(rung.get("deleted_paths", []))  # type: ignore[arg-type]

        def scan() -> dict[str, object]:
            process = subprocess.Popen(docker_command("export", container), stdout=subprocess.PIPE)
            found: dict[str, str] = {}
            present_deleted: list[str] = []
            with tarfile.open(fileobj=process.stdout, mode="r|") as source:
                for member in source:
                    path = _member_path(member.name)
                    if path in deleted:
                        present_deleted.append(path)
                    if member.isreg() and path in expected:
                        reader = _HashingReader(source.extractfile(member))  # type: ignore[arg-type]
                        while reader.read(1 << 20):
                            pass
                        found[path] = reader.digest.hexdigest()
            process.wait(timeout=600)
            mismatched = sorted(p for p, d in expected.items() if found.get(p) != d)
            return {"files_checked": len(expected), "mismatched": mismatched, "deleted_still_present": present_deleted}

        report = await asyncio.to_thread(scan)
    finally:
        await _run(docker_command("rm", "-f", container), timeout=120)
    report["exact"] = not report["mismatched"] and not report["deleted_still_present"]
    return report


def snapshot_receipt_from_rung(
    rung: Mapping[str, object],
    materialized: Mapping[str, object],
    *,
    full_prefix_length: int,
    cell_id: str,
    task: str,
    public_paths: Sequence[str],
    ladder_receipt: str,
) -> dict[str, object]:
    """A ``terminal_prefix_snapshot_v1`` rollback receipt, so arms can start from a rung."""

    verification = materialized.get("verification")
    if not isinstance(verification, Mapping) or verification.get("exact") is not True:
        raise ValueError("only a rung whose digests were all re-verified can start a run")
    kept = int(rung["calls_kept"])  # type: ignore[arg-type]
    return {
        "schema_version": SNAPSHOT_RECEIPT_SCHEMA,
        "cell_id": cell_id,
        "task": task,
        "image_tag": materialized["image_tag"],
        "image_id": materialized["image_id"],
        "public_paths": list(public_paths),
        "public_state": rung["public_state"],
        "history": rung["history"],
        "parity_policy": "exact_v1",
        "rollback_after_call": kept,
        "full_prefix_length": full_prefix_length,
        "ladder_receipt": ladder_receipt,
        "limits": "filesystem only: running processes and memory are not part of the snapshot",
    }


async def write_trace_since(environment: Any, container: str, epoch: str, before: Mapping[str, str]) -> dict[str, object]:
    """Paths one call changed: inode-change time after ``epoch`` plus ``docker diff`` deltas.

    Both probes only read. ``ctime`` catches files rewritten in place (``mtime`` can be set
    back by ``tar`` or ``cp -p``); the diff delta catches additions and deletions.
    """

    command = (
        "find / -xdev \\( -path /proc -o -path /sys -o -path /dev \\) -prune -o "
        f"-newerct @{epoch} -print 2>/dev/null | head -n {_TRACE_LIMIT + 1}"
    )
    result = await environment.exec(command=command, timeout_sec=300)
    changed = sorted(line for line in str(getattr(result, "stdout", "") or "").splitlines() if line.startswith("/"))
    truncated = len(changed) > _TRACE_LIMIT
    after = parse_docker_diff(await _run(docker_command("diff", container), timeout=600))
    diff_delta = sorted(path for path, kind in after.items() if before.get(path) != kind)
    return {"ctime_changed": changed[:_TRACE_LIMIT], "ctime_truncated": truncated, "diff_delta": diff_delta, "diff": after}


_TRACE_LIMIT = 5000
