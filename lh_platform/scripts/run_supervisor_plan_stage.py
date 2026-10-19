#!/usr/bin/env python3
"""TraceDP v3 supervisor planning stage.

Runs before the Actor resumes. At each step the supervisor DP
(``lh_platform.tracefix_bench.supervisor_dp``) picks a meta-action from the evidence
gathered so far; this script executes it and records the observed outcome:

* ``probe_inventory``: input files present in the initial task image and missing at the
  failure snapshot (resource debt);
* ``probe_clean_context``: required scripts that run under ``python3`` but fail in a
  fresh interpreter;
* ``bisect_rollback``: binary search over verified prefix replays for the last call after
  which every debt file is still intact; writes ``<request>.rollback.json``;
* ``intervene``: the DoVer intervention at the step that
  destroyed the debt files, with that evidence as its reason;
* ``resume``: stop.

Independently of the DP (it carries no measured value), the dependency probe runs the
frozen public checks at the failure snapshot and writes the last writer of the earliest
refuted subject to ``<request>.hypothesis.json`` for the check priors.

All probe and replay time is charged to the arm's shared ledger (role ORCHESTRATION);
the DoVer call is charged through the metering proxy (role LOCATOR). Outputs:
``<request>.plan.json`` and, when chosen, ``.rollback.json``, ``.intervention.{json,txt}``,
``.env_evidence.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from run_binary_bridge_arm import _ledger, _public_case  # noqa: E402
from run_controlled_injection_pairs import SNAPSHOT_VERSION, wait_for_replay  # noqa: E402
from run_dover_intervention_stage import (  # noqa: E402
    UPSTREAM_SCRIPT,
    _audited_chat,
    generate_intervention,
)
from terminal_historical_managed_launch import read_json  # noqa: E402

from lh_platform.test_time_adaptation.dover_official_intervention_public import (  # noqa: E402
    DoVerOfficialInterventionPublicAdapter,
)
from lh_platform.tracefix_bench import supervisor_probes as probes  # noqa: E402
from lh_platform.tracefix_bench.scorer_contracts import ResourceRole  # noqa: E402
from lh_platform.tracefix_bench.shared_budget import FileSharedBudgetLedger  # noqa: E402
from lh_platform.tracefix_bench.supervisor_dp import (  # noqa: E402
    SUPERVISOR_DP_ID,
    PlanContext,
    PlanState,
    best_action,
)

SCHEMA = "tracedp_supervisor_plan_v1"
ARM_ID = "tracedp-v3"


class Planner:
    def __init__(self, args, matrix: dict, terminal_root: Path) -> None:
        self.args = args
        self.matrix = matrix
        self.task = str(matrix["task_instance"]["task_name"])
        self.instruction = (terminal_root / "tasks" / self.task / "instruction.md").read_text(encoding="utf-8")
        self.docker = str(matrix["shared_environment"]["docker_context"])
        self.initial_image = tomllib.loads((terminal_root / "tasks" / self.task / "task.toml").read_text())[
            "environment"]["docker_image"]
        snapshot = read_json(Path(matrix["prefix_snapshot"]["receipt"]))
        self.failure_image = str(snapshot["image_tag"])
        self.prefix_calls = len(snapshot["history"])
        self.ledger_path = args.request.with_suffix(".ledger.json")
        self.ledger = _ledger(self.ledger_path, matrix, args.arm_id)
        self.directories = probes.instruction_directories(self.instruction)
        self.scripts = probes.required_scripts(self.instruction)
        self.outputs = list(matrix["public_paths"])
        storage = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback" / "controlled_injection_v1"
        self.storage = storage
        self.visible_cell = storage / args.cell.relative_to(args.cell.parents[3])
        self.initial_inventory: dict[str, str] = {}
        self.debt: list[str] = []
        self.record: dict = {"steps": [], "wall_time_seconds": {}}

    def _charge(self, label: str, started: float) -> float:
        seconds = time.monotonic() - started
        self.ledger.record_wall_time(ResourceRole.ORCHESTRATION, wall_time_seconds=seconds)
        self.record["wall_time_seconds"][label] = self.record["wall_time_seconds"].get(label, 0.0) + seconds
        return seconds

    def _inventory(self, image: str) -> dict[str, str]:
        _code, out = probes.run_in_image(image, probes.inventory_command(self.directories), docker_context=self.docker)
        return probes.parse_inventory(out)

    # -- meta-actions -------------------------------------------------------------------
    def probe_inventory(self) -> str:
        started = time.monotonic()
        self.initial_inventory = self._inventory(self.initial_image)
        debt = probes.resource_debt(self.initial_inventory, self._inventory(self.failure_image), self.outputs)
        self.debt = debt["missing"]
        self.record["inventory"] = {"directories": self.directories, "initial_files": len(self.initial_inventory), **debt}
        # The closed loop compares later states against this inventory.
        with self.args.request.with_suffix(".initial_inventory.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"directories": self.directories, "scripts": self.scripts,
                                     "outputs": self.outputs, "files": self.initial_inventory}, indent=2) + "\n")
        self._charge("probe_inventory", started)
        return "debt" if self.debt else "clear"

    def probe_clean_context(self) -> str:
        started = time.monotonic()
        results = {}
        for script in self.scripts:
            _code, out = probes.run_in_image(self.failure_image, probes.clean_context_command(script),
                                             docker_context=self.docker)
            results[script] = probes.clean_context_verdict(out)
        self.record["clean_context"] = results
        self._charge("probe_clean_context", started)
        failing = [s for s, r in results.items() if r["verdict"] == "fail"]
        if failing:
            lines = [
                f"The checked condition is not satisfied: {script} runs successfully with python3 in this "
                f"environment but fails in a fresh Python virtual environment without additional packages "
                f"(exit {results[script]['clean_exit']}: {results[script]['clean_stderr'][-200:]!r}). "
                "Inspect the cause and repair the affected prerequisite while preserving other work. "
                f"Then rerun this exact check: python3 -m venv --without-pip /tmp/lh_clean && "
                f"/tmp/lh_clean/bin/python {script}. "
                "A successful repair command alone does not resolve the incident."
                for script in failing
            ]
            with self.args.request.with_suffix(".env_evidence.txt").open("x", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n")
            return "fail"
        return "pass"

    def _snapshot(self, k: int) -> dict | None:
        receipt = self.visible_cell.parent / f"prefix_snapshot_upto{k:03d}.json"
        if not receipt.is_file():
            job = self.storage / "prefix_snapshots" / f"snapshot_{self.visible_cell.parent.name}_upto{k:03d}_{SNAPSHOT_VERSION}"
            if wait_for_replay(job) == "timeout":
                return None
            subprocess.run([sys.executable, str(HERE.parent / "controlled_injection.py"), "snapshot-cells",
                            str(self.visible_cell), "--job-root", str(self.storage / "prefix_snapshots"),
                            "--tasks-root", str(self.storage / "snapshot_tasks"), "--after-call", str(k)],
                           check=False)
        return {"path": receipt, **read_json(receipt)} if receipt.is_file() else None

    def _intact(self, image: str) -> bool:
        inventory = self._inventory(image)
        return all(inventory.get(p) == self.initial_inventory[p] for p in self.debt)

    def bisect_rollback(self) -> str:
        started = time.monotonic()
        lo, hi, probes_run = 0, self.prefix_calls, []
        outcome = "done"
        while hi - lo > 1:
            mid = (lo + hi) // 2
            snapshot = self._snapshot(mid)
            if snapshot is None:
                outcome = "failed"
                probes_run.append({"after_call": mid, "snapshot": "not_accepted"})
                break
            intact = self._intact(str(snapshot["image_tag"]))
            probes_run.append({"after_call": mid, "intact": intact})
            lo, hi = (mid, hi) if intact else (lo, mid)
        self.record["bisection"] = {"debt": self.debt, "probes": probes_run, "last_intact_after_call": lo}
        if outcome == "done" and lo == 0:
            outcome = "failed"  # the first call destroyed the debt: no snapshot before it
            self.record["bisection"]["reason"] = "debt arises at the first call"
        if outcome == "done":
            snapshot = self._snapshot(lo)
            if snapshot is None:
                outcome = "failed"
            else:
                self.hypothesis_event = f"step-{lo + 1:04d}"
                self.rollback_image = str(snapshot["image_tag"])
                self.hypothesis_reason = (
                    f"{', '.join(self.debt)} existed in the initial environment and was still intact after "
                    f"call {lo}; after this step it no longer exists, and nothing restores it later."
                )
                with self.args.request.with_suffix(".rollback.json").open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "receipt": str(snapshot["path"]), "derived_task_path": snapshot["derived_task_path"],
                        "rollback_after_call": lo, "hypothesis_event": self.hypothesis_event,
                        "source": "supervisor_bisection",
                    }, indent=2) + "\n")
        self._charge("bisect_rollback", started)
        return outcome

    def intervene(self) -> str:
        terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
        case = _public_case(self.matrix, self.args.source_session, terminal_root)
        adapter = DoVerOfficialInterventionPublicAdapter(self.args.corpus_root / UPSTREAM_SCRIPT)
        model = str(self.matrix["shared_actor"]["model"])
        audit, chat = _audited_chat(model=model, max_tokens=int(self.matrix["shared_budget"]["per_call_output_tokens"]),
                                    ledger_path=self.ledger_path)
        hypothesis = {"finding": {"event_ids": [self.hypothesis_event], "explanation": self.hypothesis_reason}}
        with audit:
            result = generate_intervention(case, hypothesis, task_instruction=self.instruction, adapter=adapter, chat=chat)
            if "raw_response" in result:
                result["model_calls"] = audit.snapshot().model_calls
        with self.args.request.with_suffix(".intervention.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema_version": "tracedp_dover_intervention_stage_v1",
                                     "source": "supervisor_plan", **result}, ensure_ascii=False, indent=2) + "\n")
        if result["status"] == "completed":
            with self.args.request.with_suffix(".intervention.txt").open("x", encoding="utf-8") as stream:
                stream.write(str(result["replacement_text"]) + "\n")
            return "done"
        return "invalid"

    def probe_dependency(self) -> None:
        """Last writer of the earliest refuted subject, for the check priors only."""

        started = time.monotonic()
        lines = []
        checks = [c for c in self.matrix["public_acceptance_checks"] if c.get("verdict_protocol") == "exit_status_v1"]
        for check in checks:
            command = check["arguments"]["command"].replace("'", "'\\''")
            lines.append(f"timeout {int(check['arguments'].get('timeout', 60))} sh -c '{command}' >/dev/null 2>&1; "
                         f"echo \"LH_CHECK {check['check_id']} $?\"")
        # Evidence is bound to the state it was taken in: after a rollback, probe the
        # rolled-back state, not the failure state.
        image = getattr(self, "rollback_image", None) or self.failure_image
        _code, out = probes.run_in_image(image, "; ".join(lines) or "true", docker_context=self.docker)
        verdicts = {parts[1]: int(parts[2]) for parts in (line.split() for line in out.splitlines())
                    if len(parts) == 3 and parts[0] == "LH_CHECK"}
        refuted = [c for c in checks if verdicts.get(c["check_id"]) == 1]
        subjects = sorted({r for c in refuted for r in c["resource_ids"]})
        observer = self.args.source_session.parents[2] / "terminal-public-state.jsonl"
        rows = [json.loads(line) for line in observer.read_text().splitlines() if line.strip()]
        writers = probes.dependency_hypotheses(rows, subjects)
        self.record["dependency"] = {"refuted_checks": [c["check_id"] for c in refuted], "writers": writers,
                                     "probed_image": image}
        if writers:
            earliest = min(writers.values())
            with self.args.request.with_suffix(".hypothesis.json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps({"event_id": earliest, "source": "dependency_last_writer",
                                         "writers": writers}, indent=2) + "\n")
        self._charge("probe_dependency", started)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--source-session", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--arm-id", default=ARM_ID, help="the arm whose shared ledger the plan is charged to")
    parser.add_argument("--dry-run", action="store_true",
                        help="zero model calls: record an intervention the DP chose instead of making it")
    args, _unknown = parser.parse_known_args()
    matrix = read_json(args.matrix)
    planner = Planner(args, matrix, Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"]))
    context = PlanContext(prefix_calls=planner.prefix_calls, script_required=bool(planner.scripts))
    state = PlanState(clean="unknown" if planner.scripts else "na")
    field = {"probe_inventory": "inventory", "probe_clean_context": "clean",
             "bisect_rollback": "rollback", "intervene": "intervention"}
    for _ in range(8):
        action, values = best_action(state, context)
        step = {"state": state.__dict__.copy(), "values": values, "action": action}
        planner.record["steps"].append(step)
        if action == "resume":
            break
        if action == "intervene" and args.dry_run:
            step["outcome"] = "skipped_dry_run"
            planner.record["would_intervene"] = {"event": planner.hypothesis_event, "reason": planner.hypothesis_reason}
            break
        outcome = getattr(planner, action)()
        step["outcome"] = outcome
        state = PlanState(**{**state.__dict__, field[action]: outcome})
    planner.probe_dependency()
    planner.record.update(schema_version=SCHEMA, selector=SUPERVISOR_DP_ID, final_state=state.__dict__,
                          context=context.__dict__, ledger=FileSharedBudgetLedger(planner.ledger_path).receipt())
    with args.request.with_suffix(".plan.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(planner.record, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"plan": [s["action"] for s in planner.record["steps"]], "final": state.__dict__}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
