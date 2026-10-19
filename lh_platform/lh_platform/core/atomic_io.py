"""Crash-consistent publication of immutable platform artifacts."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import stat
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4


def _atomic_write_chunks(
    path: str | Path,
    chunks: Iterable[bytes],
    *,
    overwrite: bool = False,
) -> None:
    """Stream bytes into a staged file, then publish it atomically.

    The staging file lives beside the destination, so publication is atomic on
    the destination filesystem.  ``overwrite=False`` uses a hard-link claim
    instead of a racy ``exists()`` check.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    published = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(staged, flags, 0o600)
        try:
            buffered = os.fdopen(
                fd,
                "wb",
                buffering=1024 * 1024,
                closefd=False,
            )
            try:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("atomic payload chunks must be bytes")
                    buffered.write(chunk)
                buffered.flush()
            finally:
                buffered.close()
            os.fsync(fd)
        finally:
            os.close(fd)

        if overwrite:
            os.replace(staged, target)
            published = True
        else:
            try:
                os.link(staged, target, follow_symlinks=False)
            except OSError as error:
                unsupported = {
                    errno.EPERM,
                    errno.EOPNOTSUPP,
                    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
                }
                if error.errno not in unsupported:
                    raise
                # FAT-family volumes do not support hard links. The kernel's
                # exclusive rename preserves the same no-overwrite contract.
                _rename_noreplace(staged, target)
                published = True
            else:
                published = True
                staged.unlink()

        parent_fd = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published or staged.exists():
            try:
                staged.unlink()
            except FileNotFoundError:
                pass


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one regular file only after its complete bytes reach storage."""

    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    _atomic_write_chunks(path, (payload,), overwrite=overwrite)


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(text, str):
        raise TypeError("atomic text payload must be a string")
    atomic_write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, encoded, overwrite=overwrite)


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
    ensure_ascii: bool = True,
    default: Callable[[Any], Any] | None = None,
) -> None:
    def encoded_rows() -> Iterable[bytes]:
        for row in rows:
            yield (
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=ensure_ascii,
                    default=default,
                )
                + "\n"
            ).encode("utf-8")

    _atomic_write_chunks(path, encoded_rows(), overwrite=overwrite)


def _fsync_directory(path: Path) -> None:
    fd = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    """Flush a staged regular-file tree before publishing its directory."""

    if not root.is_dir() or root.is_symlink():
        raise ValueError("atomic directory staging root must be a real directory")
    directories = [root]
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in dirnames:
            child = current_path / name
            if child.is_symlink():
                raise ValueError("atomic directory trees cannot contain symlinks")
            directories.append(child)
        for name in filenames:
            child = current_path / name
            metadata = child.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("atomic directory trees may contain only regular files")
            fd = os.open(
                child,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _rename_noreplace(staged: Path, destination: Path) -> None:
    """Atomically rename a path while refusing an existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(staged)
    target = os.fsencode(destination)
    if sys.platform == "darwin":
        rename_exclusive = 0x00000004  # RENAME_EXCL from <sys/stdio.h>
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise NotImplementedError("kernel no-replace rename is unavailable")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source, target, rename_exclusive)
    elif sys.platform.startswith("linux"):
        at_fdcwd = -100
        rename_noreplace = 1
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise NotImplementedError("kernel no-replace rename is unavailable")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            at_fdcwd,
            source,
            at_fdcwd,
            target,
            rename_noreplace,
        )
    elif os.name == "nt":
        # Windows MoveFile semantics already refuse an existing destination.
        os.rename(staged, destination)
        return
    else:
        raise NotImplementedError("kernel no-replace rename is unavailable")

    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def atomic_publish_directory(staging: str | Path, target: str | Path) -> None:
    """Publish an immutable directory after exclusively claiming its name.

    Both paths must be siblings.  The lock is deliberately fail-closed: a
    process crash can leave a lock for operator inspection, but a concurrent
    platform publisher can never replace an existing checkpoint.
    """

    staged = Path(staging)
    destination = Path(target)
    if staged.parent != destination.parent:
        raise ValueError("atomic directory publication requires sibling paths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(f".{destination.name}.publish.lock")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    lock_fd = os.open(lock, flags, 0o600)
    published = False
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        _fsync_directory(destination.parent)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        _fsync_tree(staged)
        # The lock serializes platform publishers; the kernel no-replace flag
        # also rejects an uncooperative external process winning the race.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        _rename_noreplace(staged, destination)
        published = True
        _fsync_directory(destination.parent)
    finally:
        os.close(lock_fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(destination.parent)
    if not published:
        raise OSError("atomic directory publication did not complete")


__all__ = [
    "atomic_publish_directory",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
]
