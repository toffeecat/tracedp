"""TraceDP: the supervisor re-enters at the Actor's natural settlements.

At each settlement (at most ``MAX_REENTRIES`` per run) the supervisor freezes the live
container into a throwaway image, re-probes it with zero model calls (input inventory
against the plan stage's initial inventory; clean-context runs of required scripts), and
lets ``supervisor_dp.reentry_decision`` choose:

* ``restore_inputs``: copy only the lost inputs back from the latest verified snapshot in
  which they are intact, and tell the Actor which files were lost and restored;
* ``redeliver_env_evidence``: deliver the clean-context evidence again;
* ``defer``: return ``None`` so the v2 runtime makes its usual settlement decision.

Probe and restore time is charged to the arm's shared ledger (role ORCHESTRATION); every
re-entry appends one line to the configured log.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from lh_platform.benchmarks import terminal_prefix_snapshot as snapshot
from lh_platform.tracefix_bench import supervisor_probes as probes
from lh_platform.tracefix_bench.scorer_contracts import ResourceRole
from lh_platform.tracefix_bench.shared_budget import FileSharedBudgetLedger
from lh_platform.tracefix_bench.supervisor_dp import MAX_REENTRIES, ReentryEvidence, reentry_decision

SCHEMA = "tracedp_supervisor_reentry_v1"
CONFIG_SCHEMA = "tracedp_supervisor_reentry_config_v1"


def restore_notice(paths: list[str], intervention: str | None) -> str:
    files = ", ".join(paths)
    text = (
        f"The checked condition is not satisfied: {files} existed at the start of the task and no "
        "longer exists after this attempt. It has been restored to its original content from a "
        "verified earlier state. Inspect the cause and continue while preserving the restored file."
    )
    if intervention:
        text += f" Guidance written earlier for the step that lost it: {intervention}"
    return text


class SupervisorReentry:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        container_id: Callable[[Any], Awaitable[str]] = snapshot.container_id,
        commit: Callable[[str, str], Awaitable[str]] = snapshot.commit_probe_image,
        remove: Callable[[str], Awaitable[None]] = snapshot.remove_image,
        run_in_image: Callable[..., tuple[int, str]] = probes.run_in_image,
        restore: Callable[..., Awaitable[list[str]]] = snapshot.restore_files,
    ) -> None:
        if config.get("schema_version") != CONFIG_SCHEMA:
            raise ValueError("supervisor re-entry config schema differs")
        self.config = config
        self.reentries = 0
        self.restores = 0
        self.env_deliveries = 1 if config.get("env_evidence_delivered") else 0
        self._container_id = container_id
        self._commit = commit
        self._remove = remove
        self._run_in_image = run_in_image
        self._restore = restore

    @classmethod
    def from_path(cls, path: str | Path, **kwargs: Any) -> SupervisorReentry:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)

    async def _probe(self, image: str) -> tuple[list[str], bool, dict[str, Any]]:
        context = str(self.config["docker_context"])
        _code, out = await asyncio.to_thread(
            self._run_in_image, image, probes.inventory_command(self.config["directories"]), docker_context=context
        )
        now = probes.parse_inventory(out)
        initial = self.config["initial_inventory"]
        missing = probes.resource_debt(initial, now, self.config["outputs"])["missing"]
        clean = {}
        for script in self.config["scripts"]:
            _code, out = await asyncio.to_thread(
                self._run_in_image, image, probes.clean_context_command(script), docker_context=context
            )
            clean[script] = probes.clean_context_verdict(out)
        failing = any(result["verdict"] == "fail" for result in clean.values())
        return missing, failing, {"clean_context": clean}

    async def at_settlement(self, environment: Any) -> str | None:
        """The notice for the Actor, or ``None`` to defer to the v2 runtime."""

        if self.reentries >= MAX_REENTRIES:
            return None
        self.reentries += 1
        started = time.monotonic()
        record: dict[str, Any] = {"schema_version": SCHEMA, "reentry": self.reentries}
        notice = None
        try:
            container = await self._container_id(environment)
            image = f"longh-reentry/{self.config['run_label']}:{self.reentries}"
            await self._commit(container, image)
            try:
                missing, failing, details = await self._probe(image)
            finally:
                await self._remove(image)
            evidence = ReentryEvidence(
                missing_inputs=tuple(missing), clean_context_failing=failing,
                rolled_back=bool(self.config.get("rolled_back")),
                intervention_delivered=bool(self.config.get("intervention_text")),
                env_evidence_delivered=self.env_deliveries, restores=self.restores,
            )
            choice, values = reentry_decision(evidence)
            record.update(evidence=evidence.__dict__, values=values, choice=choice, **details)
            if choice == "restore_inputs":
                restored = await self._restore(
                    container, str(self.config["restore_image"]), list(missing),
                    staging=Path(self.config["staging_dir"]) / f"reentry_{self.reentries}",
                )
                self.restores += 1
                record["restored"] = restored
                notice = restore_notice(restored, self.config.get("intervention_text"))
            elif choice == "redeliver_env_evidence" and self.config.get("env_evidence_text"):
                self.env_deliveries += 1
                notice = str(self.config["env_evidence_text"])
        except Exception as error:  # a failed probe defers; it never ends the run
            record["error"] = f"{type(error).__name__}: {str(error)[:400]}"
            notice = None
        seconds = time.monotonic() - started
        record["seconds"] = seconds
        FileSharedBudgetLedger(Path(self.config["ledger_path"])).record_wall_time(
            ResourceRole.ORCHESTRATION, wall_time_seconds=seconds
        )
        with Path(self.config["log_path"]).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return notice


__all__ = ["CONFIG_SCHEMA", "SupervisorReentry", "restore_notice"]
