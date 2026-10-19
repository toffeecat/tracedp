#!/usr/bin/env python3
"""Summarize controlled-injection paired runs from their manifests, ledgers and job dirs.

Reads only what the runs left on disk: the manifest rows (one per attempt), each job's
official reward and verifier report, the per-arm shared-budget ledger, and for managed arms
the decision log. A run counts only with a verifier report; the last reported attempt of a
(cell, arm, repeat) is its result. Nothing here re-scores or re-labels a run.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

RUNS = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
CI = RUNS / "controlled_injection_v1"
ARM_ORDER = (
    "no-localization", "check-feedback-only", "info-gain-reserve", "tracedp", "tracedp-guided", "tracedp-v2", "tracedp-v2h", "tracedp-v2hr", "tracedp-v2hri", "tracedp-v3", "tracedp-v3c", "tracedp-v3o", "tracedp-v3of", "tracedp-v4", "agentrewind-local",
    "binary-bridge", "whowhen-all-at-once-bridge", "whowhen-step-by-step-bridge",
    "traceelephant-all-at-once-bridge", "traceelephant-step-by-step-bridge",
    "traceelephant-binary-search-bridge",
)


def _job_dir(run_id: str) -> Path | None:
    for root in (RUNS / "managed_controller_jobs", RUNS):
        if (root / run_id).is_dir():
            return root / run_id
    return None


def _trial(job: Path) -> Path | None:
    trials = [t for t in job.glob("*__*") if not t.name.startswith("._") and (t / "result.json").is_file()]
    return trials[0] if trials else None


def _decisions(trial: Path) -> list[str]:
    path = trial / "agent" / "runtime-decisions.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["event_type"] == "decision_selected":
            rows.append(str(row["payload"]["selected_action_type"]).replace("run_", "").replace("_", "-"))
    return rows


def _ledger(request: str) -> dict:
    path = Path(request).with_suffix(".ledger.json")
    if not path.is_file():
        return {}
    ledger = json.loads(path.read_text())
    return {"calls": ledger["calls"], "in": ledger["input_tokens"], "out": ledger["output_tokens"],
            "complete": ledger["metering_complete"]}


_BACKFILL = CI / "method_failure_backfill.jsonl"
_BACKFILLED = (
    {json.loads(line)["run_id"]: json.loads(line)["method_failure"] for line in _BACKFILL.read_text().splitlines()}
    if _BACKFILL.is_file() else {}
)


# Run ids invalidated for infrastructure reasons (reused ids, rollback races, phantom charges,
# interrupted attempts). They are infrastructure-invalid whatever they scored.
_CONTAMINATED = {
    json.loads(line)["run_id"]
    for name in ("reused_run_id_contamination.jsonl", "rollback_race_invalid.jsonl", "phantom_charge_invalid.jsonl",
                 "interrupted_attempts.jsonl")
    if (CI / name).is_file()
    for line in (CI / name).read_text().splitlines() if line.strip()
}


def _locator_method_failure(request: str) -> str | None:
    # Failures from launchers older than the method-failure rule are recorded, with
    # their evidence, in method_failure_backfill.jsonl instead of a receipt.
    backfilled = _BACKFILLED.get(Path(request).stem)
    if backfilled:
        return backfilled
    receipt = Path(request).with_suffix(".locate.json")
    if not receipt.is_file():
        return None
    data = json.loads(receipt.read_text())
    return data.get("method_failure") if data.get("status") == "method_failure" else None


def load(manifests: list[Path]) -> list[dict]:
    # A (cell, arm, repeat) is answered by its last reported attempt, in manifest order;
    # only if no attempt has a report does the last unreported one stand (as invalid).
    # Attempt numbers are per launch, so a later snapshot-start retry can carry a smaller
    # number than an earlier invalid replay-start attempt of the same repeat.
    # A locator method failure ends its (cell, arm, repeat) at the first attempt that shows
    # it; any later retry (launched by a runner older than that rule) is not counted.
    last: dict[tuple, dict] = {}
    final: set[tuple] = set()
    for manifest in manifests:
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            key = (row["cell_id"], row["arm"], row["repeat"])
            if key in final or row["run_id"] in _CONTAMINATED:
                continue
            failure = row.get("method_failure") or _locator_method_failure(row["request"])
            if failure:
                last[key] = {**row, "method_failure": failure}
                final.add(key)
                continue
            previous = last.get(key)
            if previous is None or row["verifier_report"] or not previous["verifier_report"]:
                last[key] = row
    runs = []
    for _key, row in sorted(last.items()):
        job = _job_dir(row["run_id"])
        trial = _trial(job) if job else None
        result = json.loads((trial / "result.json").read_text()) if trial else {}
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        verify_only = row.get("verify_only") or {}
        if not reward and verify_only.get("valid"):
            # The official verifier rerun on the committed post-Actor state (network loss).
            reward = verify_only.get("reward")
        # A method failure (budget exhausted, invalid locator output) is a scored 0.
        valid = bool(row["verifier_report"]) or bool(row.get("method_failure"))
        if row.get("method_failure"):
            reward = 0.0
        runs.append({
            **row,
            "valid": valid,
            "reward": reward if valid else None,
            "decisions": _decisions(trial) if trial else [],
            "ledger": _ledger(row["request"]),
        })
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    runs = load(args.manifests)
    if args.json:
        args.json.write_text(json.dumps(runs, indent=2) + "\n")
    by_arm = collections.defaultdict(list)
    for run in runs:
        by_arm[run["arm"]].append(run)
    print(f"{'arm':22s} {'runs':>4s} {'valid':>5s} {'resolved':>8s} {'calls(mean)':>11s}  per-cell resolved/valid")
    for arm in ARM_ORDER:
        rows = by_arm.get(arm, [])
        if not rows:
            continue
        valid = [r for r in rows if r["valid"]]
        won = sum(1 for r in valid if r["reward"] == 1.0)
        calls = [r["ledger"].get("calls") for r in rows if r["ledger"]]
        cells = collections.defaultdict(lambda: [0, 0])
        for r in valid:
            cells[r["cell_id"].split("__")[-1] + "@" + r["task"][:6]][1] += 1
            cells[r["cell_id"].split("__")[-1] + "@" + r["task"][:6]][0] += r["reward"] == 1.0
        per_cell = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(cells.items()))
        mean_calls = f"{sum(calls) / len(calls):.1f}" if calls else "-"
        print(f"{arm:22s} {len(rows):4d} {len(valid):5d} {won:8d} {mean_calls:>11s}  {per_cell}")
    print()
    for run in runs:
        trace = " > ".join(run["decisions"]) or "-"
        print(f"{run['cell_id'][-22:]:22s} r{run['repeat']} {run['arm']:20s} reward={run['reward']} try{run['attempt']} {trace[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
