"""Zero-model replay that also records a checkpoint ladder and a per-call write trace.

It is the plain controlled replay with three hooks
filled in; with neither option set it behaves exactly like ``HostTerminalReplayAgent``.

* ``ladder_every=k``: after every ``k``-th call (and the last) a rung is captured, see
  :mod:`terminal_checkpoint_ladder`. The base image is saved once to
  ``ladder_base_cache``.
* ``trace_writes=True``: before each call the container clock is read; after it, the
  paths whose inode changed since then and the ``docker diff`` delta are recorded. This is
  the exact write set the command-text heuristic of the dependency graph only guesses.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from lh_platform.benchmarks.harbor_opencode_go_pi import HostTerminalReplayAgent
from lh_platform.benchmarks.terminal_checkpoint_ladder import (
    LADDER_SCHEMA,
    WRITE_TRACE_SCHEMA,
    capture_rung,
    ladder_points,
    parse_docker_diff,
    record_base_image,
    write_trace_since,
)
from lh_platform.benchmarks.terminal_prefix_snapshot import (
    _run,
    container_id,
    docker_command,
    public_state_payload,
    running_image_id,
)


class HostTerminalLadderReplayAgent(HostTerminalReplayAgent):
    def __init__(
        self,
        *args,
        ladder_every: int | None = None,
        ladder_base_cache: str | None = None,
        trace_writes: bool = False,
        source_era_bash: bool = False,
        batched_public_snapshot: bool = False,
        **kwargs,
    ) -> None:
        if (ladder_every is None) != (ladder_base_cache is None):
            raise ValueError("a ladder needs both a rung interval and a base-image cache")
        super().__init__(*args, **kwargs)
        self.ladder_every = ladder_every
        self.ladder_base_cache = ladder_base_cache
        self.trace_writes = trace_writes
        self.source_era_bash = source_era_bash
        # Read by the base replay when it builds its gateway: one exec per public snapshot,
        # self-checked byte for byte against the per-path read on its first use.
        self.batched_public_snapshot = batched_public_snapshot
        self._container: str | None = None
        self._points: frozenset[int] = frozenset()
        self._rungs: list[dict[str, object]] = []
        self._segment_started = 0.0
        self._epoch: str | None = None
        self._diff: dict[str, str] = {}
        self._base: dict[str, object] = {}

    async def _before_replay_calls(self, environment, gateway) -> None:
        if self.source_era_bash:
            gateway.source_era_bash = True
            gateway.source_text_refusals = True
        if self.ladder_every is None and not self.trace_writes:
            return
        self._container = await container_id(environment)
        if self.ladder_every is not None:
            image = await running_image_id(self._container)
            self._base = await record_base_image(image, Path(str(self.ladder_base_cache)))
            self._points = frozenset(ladder_points(len(self.tool_calls), self.ladder_every))
        if self.trace_writes:
            self._diff = parse_docker_diff(await _run(docker_command("diff", self._container), timeout=600))
        self._segment_started = time.monotonic()

    async def _before_replay_call(self, environment, gateway, index) -> None:
        if not self.trace_writes:
            return
        result = await environment.exec(command="date +%s.%N", timeout_sec=30)
        stamp = str(getattr(result, "stdout", "") or "").strip()
        # One second earlier: ctime has coarse granularity on some filesystems, and an
        # extra path is harmless while a missed write is not.
        self._epoch = f"{float(stamp) - 1.0:.3f}" if stamp else None

    async def _after_replay_call(self, environment, gateway, index, history) -> None:
        if self.trace_writes and self._container is not None:
            row: dict[str, object] = {"schema_version": WRITE_TRACE_SCHEMA, "sequence": index}
            if self._epoch is None:
                row["error"] = "container clock unreadable"
            else:
                trace = await write_trace_since(environment, self._container, self._epoch, self._diff)
                self._diff = trace.pop("diff")  # type: ignore[assignment]
                row.update(trace)
            with (self.logs_dir / "terminal-call-write-trace.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self._container is not None and index in self._points:
            segment = round(time.monotonic() - self._segment_started, 3)
            public_state = public_state_payload(await gateway._snapshot(len(history)))
            rung = await capture_rung(
                self._container,
                base_image_id=str(self._base["image_id"]),
                out_dir=self.logs_dir / "ladder",
                after_call=index,
                public_state=public_state,
                history=list(history),
            )
            self._rungs.append(
                {
                    "after_call": index,
                    "receipt": str(self.logs_dir / "ladder" / f"rung_{index:03d}.json"),
                    "segment_seconds": segment,
                    "capture_seconds": rung["capture_seconds"],
                    "delta_tar_bytes": rung["delta_tar_bytes"],
                    "changed_paths": rung["changed_paths"],
                    "deleted_paths": len(rung["deleted_paths"]),  # type: ignore[arg-type]
                }
            )
            self._segment_started = time.monotonic()

    async def run(self, instruction, environment, context) -> None:
        await super().run(instruction, environment, context)
        if self.ladder_every is not None:
            if sorted(self._points) != [rung["after_call"] for rung in self._rungs]:
                raise RuntimeError("the checkpoint ladder is incomplete")
            (self.logs_dir / "terminal-checkpoint-ladder.json").write_text(
                json.dumps(
                    {
                        "schema_version": LADDER_SCHEMA,
                        "every": self.ladder_every,
                        "call_count": len(self.tool_calls),
                        "base": self._base,
                        "rungs": self._rungs,
                        # Raw: source admission accepts a ladder only when both replays of
                        # the source match it call by call; the trial artifact stays untouched.
                        "acceptance": "recorded by source admission, not here",
                        "limits": "filesystem only: running processes and memory are not part of a rung",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        metadata = dict(context.metadata or {})
        metadata.update({
            "ladder_rungs": len(self._rungs),
            "write_trace": self.trace_writes,
            "bash_semantics": "source_era_pre_2026_09_08" if self.source_era_bash else "contained",
        })
        context.metadata = metadata
