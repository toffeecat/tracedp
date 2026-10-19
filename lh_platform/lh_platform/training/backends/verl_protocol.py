"""Bounded JSON-lines RPC for the isolated veRL process."""

from __future__ import annotations

import json
import selectors
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MAX_WORKER_MESSAGE_BYTES = 64 * 1024 * 1024


class VerLWorkerError(RuntimeError):
    """The isolated worker failed or returned an invalid receipt."""


class VerLWorkerClient:
    def __init__(
        self,
        command: Sequence[str],
        *,
        startup_timeout_seconds: float,
        request_timeout_seconds: float,
    ) -> None:
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("veRL worker command must be non-empty text")
        self._request_timeout_seconds = request_timeout_seconds
        self._next_request_id = 1
        self._closed = False
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="lh-verl-worker-")
        self.stderr_path = Path(self._temporary_directory.name) / "stderr.log"
        self._stderr = self.stderr_path.open("wb")
        self._process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=False,
            bufsize=0,
        )
        self._selector = selectors.DefaultSelector()
        assert self._process.stdout is not None
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        self.startup_timeout_seconds = startup_timeout_seconds

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, Any]:
        if self._closed:
            raise VerLWorkerError("veRL worker client is closed")
        if self._process.poll() is not None:
            raise VerLWorkerError(self._failure_message("veRL worker exited"))
        request_id = self._next_request_id
        self._next_request_id += 1
        message = (
            json.dumps(
                {
                    "request_id": request_id,
                    "operation": operation,
                    "payload": dict(payload),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        if len(message) > MAX_WORKER_MESSAGE_BYTES:
            raise VerLWorkerError("veRL worker request exceeds the bounded protocol")
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(message)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise VerLWorkerError(self._failure_message("cannot write to veRL worker")) from error
        response = self._read_response(
            timeout_seconds=(
                self._request_timeout_seconds if timeout_seconds is None else timeout_seconds
            )
        )
        if response.get("request_id") != request_id:
            raise VerLWorkerError("veRL worker response request ID drifted")
        if response.get("ok") is not True:
            detail = str(response.get("error_message") or "worker error")
            raise VerLWorkerError(detail)
        result = response.get("result")
        if not isinstance(result, dict):
            raise VerLWorkerError("veRL worker result must be an object")
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                try:
                    self.request("close", {}, timeout_seconds=5.0)
                except VerLWorkerError:
                    self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        finally:
            self._closed = True
            self._selector.close()
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
            self._stderr.close()
            self._temporary_directory.cleanup()

    def _read_response(self, *, timeout_seconds: float) -> Mapping[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        chunks = bytearray()
        assert self._process.stdout is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VerLWorkerError("veRL worker response timed out")
            events = self._selector.select(remaining)
            if not events:
                continue
            chunk = self._process.stdout.read(1)
            if not chunk:
                raise VerLWorkerError(self._failure_message("veRL worker closed stdout"))
            if chunk == b"\n":
                break
            chunks.extend(chunk)
            if len(chunks) > MAX_WORKER_MESSAGE_BYTES:
                raise VerLWorkerError("veRL worker response exceeds the bounded protocol")
        try:
            value = json.loads(chunks.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerLWorkerError("veRL worker emitted invalid JSON") from error
        if not isinstance(value, dict):
            raise VerLWorkerError("veRL worker response must be an object")
        return value

    def _failure_message(self, prefix: str) -> str:
        self._stderr.flush()
        detail = self.stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        return f"{prefix}: {detail.strip()}" if detail.strip() else prefix


__all__ = ["MAX_WORKER_MESSAGE_BYTES", "VerLWorkerClient", "VerLWorkerError"]
