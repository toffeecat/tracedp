#!/usr/bin/env python3
"""Paid paired runs on controlled injection cells (CONTROLLED_INJECTION_SPEC_V1 §9).

For every cell: build its v5 matrix if missing, then for each repeat run every arm once in
an order shuffled by a frozen seed. Managed arms go through the managed controller
(run_historical_managed_arm.py); the binary arm runs locate, prepare and run. Every launch
appends one row to an append-only run manifest; nothing is retried automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from controlled_injection import SNAPSHOT_VERSION, cell_matrix_command  # noqa: E402

MANAGED_ARMS = (
    "no-localization", "check-feedback-only", "info-gain-reserve", "tracedp", "tracedp-guided", "tracedp-v2",
    "tracedp-v2h", "tracedp-v2hr", "tracedp-v2hri", "tracedp-v3", "tracedp-v3c", "tracedp-v3o", "tracedp-v3of", "tracedp-v4",
)
BINARY_ARM = "binary-bridge"
METHOD_FAILURE_EXIT = 3  # run_binary_bridge_arm.METHOD_FAILURE_EXIT
# Every author-code locator arm runs through the same locate / prepare / run launcher.
LOCATOR_ARMS = (
    "binary-bridge", "whowhen-all-at-once-bridge", "whowhen-step-by-step-bridge",
    "traceelephant-all-at-once-bridge", "traceelephant-step-by-step-bridge",
    "traceelephant-binary-search-bridge",
)


def _arm_order(arms: list[str], *, seed: int, cell_id: str, repeat: int) -> list[str]:
    order = list(arms)
    random.Random(seed + repeat + zlib.crc32(cell_id.encode())).shuffle(order)
    return order


RUNS_ROOT = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
MAX_TRIES = 3


def _run(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def has_verifier_report(run_id: str, runs_root: Path = RUNS_ROOT) -> bool:
    """True when the official verifier left its CTRF report for this run.

    Managed arms run under ``managed_controller_jobs``; the binary arm runs directly in the
    traceback runs root. A run without the report scored a default 0, not a verdict.
    """

    for root in (runs_root / "managed_controller_jobs", runs_root):
        job = root / run_id
        if job.is_dir():
            return any(
                (trial / "verifier" / "ctrf.json").is_file()
                for trial in job.glob("*__*")
                if not trial.name.startswith("._")
            )
    return False


REPLAY_WAIT_SECONDS = 2 * 3600


def wait_for_replay(job: Path, *, timeout: float = REPLAY_WAIT_SECONDS, poll: float = 30.0) -> str:
    """Wait while another launcher's replay of the same prefix is still running.

    Returns ``"absent"`` (no replay), ``"finished"`` (its Harbor job recorded
    ``finished_at``) or ``"timeout"``.
    """

    import time

    deadline = time.monotonic() + timeout
    while True:
        result = job / "result.json"
        if not job.is_dir():
            return "absent"
        if result.is_file():
            try:
                if json.loads(result.read_text()).get("finished_at"):
                    return "finished"
            except json.JSONDecodeError:
                pass  # being written
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(poll)


def _skip_rollback(request: Path, upto: int, reason: str) -> None:
    with request.with_suffix(".rollback_skipped.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"rollback_after_call": upto, "reason": reason}, indent=2) + "\n")


def _prepare_rollback(cell_path: Path, request: Path, python: str, scripts: Path) -> None:
    """Roll back to just before the hypothesized step, when a verified snapshot exists.

    Writes ``<request>.rollback.json`` for the launcher; without it the arm starts from the
    failure snapshot as every other arm does (final-turn hypotheses, step 1, or a rollback
    snapshot that does not replay exactly).
    """

    receipt = json.loads(request.with_suffix(".locate.json").read_text())
    events = (receipt.get("finding") or {}).get("event_ids") or []
    if receipt.get("status") == "method_failure" or not events or not str(events[0]).startswith("step-"):
        return
    upto = int(str(events[0]).split("-")[1]) - 1
    if upto < 1:
        return
    root = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback" / "controlled_injection_v1"
    visible_cell = root / cell_path.relative_to(cell_path.parents[3])
    snapshot = visible_cell.parent / f"prefix_snapshot_upto{upto:03d}.json"
    if not snapshot.is_file():
        # A parallel repeat may be replaying the same prefix: wait for it rather than give
        # this repeat a different treatment.
        job = root / "prefix_snapshots" / f"snapshot_{visible_cell.parent.name}_upto{upto:03d}_{SNAPSHOT_VERSION}"
        waited = wait_for_replay(job)
        if waited == "timeout":
            _skip_rollback(request, upto, "concurrent_build_timeout")
            return
        _run([python, str(scripts / "controlled_injection.py"), "snapshot-cells", str(visible_cell),
              "--job-root", str(root / "prefix_snapshots"), "--tasks-root", str(root / "snapshot_tasks"),
              "--after-call", str(upto)])
    if not snapshot.is_file():
        _skip_rollback(request, upto, "replay_not_accepted")
        return
    data = json.loads(snapshot.read_text())
    with request.with_suffix(".rollback.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "receipt": str(snapshot), "derived_task_path": data["derived_task_path"],
            "rollback_after_call": upto, "hypothesis_event": events[0],
        }, indent=2) + "\n")


QUOTA_EXHAUSTED_EXIT = 4
# Run ids of attempts that failed because the provider quota was exhausted.
QUOTA_BACKFILL = RUNS_ROOT / "controlled_injection_v1" / "quota_failure_backfill.jsonl"
# Runs invalidated for infrastructure reasons after they scored (reused ids, rollback races,
# phantom charges, interrupted attempts).
INVALIDATED = tuple(
    RUNS_ROOT / "controlled_injection_v1" / name
    for name in ("reused_run_id_contamination.jsonl", "rollback_race_invalid.jsonl", "phantom_charge_invalid.jsonl",
                 "interrupted_attempts.jsonl")
)


def invalidated_run_ids() -> set[str]:
    return {
        json.loads(line)["run_id"]
        for path in INVALIDATED if path.is_file()
        for line in path.read_text().splitlines() if line.strip()
    }
QUOTA_MARKER = "(2056)"  # MiniMax quota error; its server text is "已达到 Token Plan 用量上限" (Token Plan usage limit reached)


def provider_quota_exhausted() -> bool:
    """One 1-token probe after a failed run: is the provider's plan quota used up?

    A spent plan quota makes every later attempt fail with HTTP 429 and QUOTA_MARKER;
    this probe detects that state.
    The probe is outside every arm and is not charged to any arm's ledger.
    """

    import urllib.error
    import urllib.request

    key = os.environ.get("MINIMAX_CN_API_KEY")
    if not key:
        return False
    request = urllib.request.Request(
        "https://api.minimaxi.com/v1/chat/completions",
        data=json.dumps({"model": "MiniMax-M3", "messages": [{"role": "user", "content": "hi"}],
                         "max_tokens": 1}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=60).close()
    except urllib.error.HTTPError as error:
        return error.code == 429 and QUOTA_MARKER in error.read().decode("utf-8", "replace")
    except OSError:
        return False
    return False


def _job_exists(run_id: str, runs_root: Path = RUNS_ROOT) -> bool:
    return any((root / run_id).exists() for root in (runs_root / "managed_controller_jobs", runs_root))


# The official test.sh needs uv (astral.sh, github.com), pytest (pypi.org) and curl (Ubuntu archive).
_VERIFIER_HOSTS = ("https://astral.sh", "https://github.com", "https://pypi.org", "https://archive.ubuntu.com")


def network_reachable(docker_context: str) -> bool:
    """Every verifier host answers from inside a throwaway container on the VM."""

    probe = "; ".join(f"wget -q -T 8 -O /dev/null {host} || exit 1" for host in _VERIFIER_HOSTS)
    try:
        done = subprocess.run(
            ["docker", "--context", docker_context, "run", "--rm", "alpine:3.19", "sh", "-c", probe],
            capture_output=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


def wait_for_network(docker_context: str, *, max_wait_seconds: int, poll_seconds: int = 60) -> float:
    """Wait (bounded) until the verifier's hosts are reachable; return the seconds waited."""

    started = time.monotonic()
    while not network_reachable(docker_context):
        waited = time.monotonic() - started
        if waited >= max_wait_seconds:
            print(f"network still unreachable after {waited:.0f} s: starting anyway", file=sys.stderr, flush=True)
            break
        print(f"verifier hosts unreachable from the VM; waiting ({waited:.0f} s so far)", file=sys.stderr, flush=True)
        time.sleep(poll_seconds)
    return time.monotonic() - started


def _trial_dirs(run_id: str) -> list[Path]:
    return [t for root in (RUNS_ROOT / "managed_controller_jobs", RUNS_ROOT)
            for t in sorted((root / run_id).glob("*__*")) if not t.name.startswith("._")]


def settle_post_actor_image(run_id: str, request: Path, *, task: str, report: bool, python: str,
                            scripts: Path, docker_context: str) -> dict | None:
    """Rerun only the official verifier on the kept post-Actor state when its report was lost.

    Runs only when the Actor phase finished cleanly (the image was committed and the trial
    raised nothing) and no CTRF report exists. The kept image is removed either way.
    Returns the verify-only summary, or ``None`` when no rerun was made.
    """

    trials = _trial_dirs(run_id)
    receipt_path = next((t / "agent" / "post-actor-image.json" for t in trials
                         if (t / "agent" / "post-actor-image.json").is_file()), None)
    if receipt_path is None:
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "committed":
        return None
    summary = None
    keep = False
    try:
        result = json.loads((trials[0] / "result.json").read_text()) if (trials[0] / "result.json").is_file() else {}
        if not report and not result.get("exception_info"):
            out = request.with_suffix(".verify_only.json")
            _run([python, str(scripts / "verify_only.py"), "--task", task, "--image", str(receipt["image_tag"]),
                  "--label", run_id, "--job-root", str(RUNS_ROOT / "controlled_injection_v1" / "verify_only"),
                  "--out", str(out)])
            if out.is_file():
                data = json.loads(out.read_text())
                summary = {"report": str(out), "valid": bool(data.get("valid")), "reward": data.get("reward"),
                           "attempts": len(data.get("attempts", [])),
                           "unscorable_cause": data.get("unscorable_cause")}
                # A network-lost verdict keeps the state: the Actor's work is not repeated later.
                keep = not summary["valid"] and summary["unscorable_cause"] is None
                if keep:
                    summary["kept_image"] = str(receipt["image_tag"])
    finally:
        if not keep:
            subprocess.run(["docker", "--context", docker_context, "image", "rm", "-f", str(receipt["image_tag"])],
                           capture_output=True)
    return summary


def managed_budget_exhausted(run_id: str, request: Path) -> bool:
    """A managed run interrupted by a shared-budget refusal while metering was complete.

    Complete metering means every charge was a real provider usage report; charges at the
    reservation (no usage returned) clear it and mark an infrastructure fault instead.
    """

    ledger = request.with_suffix(".ledger.json")
    if not ledger.is_file() or not json.loads(ledger.read_text()).get("metering_complete"):
        return False
    for root in (RUNS_ROOT / "managed_controller_jobs", RUNS_ROOT):
        for result in (root / run_id).glob("*__*/result.json"):
            if result.parent.name.startswith("._"):
                continue
            message = str((json.loads(result.read_text()).get("exception_info") or {}).get("exception_message"))
            if "shared_budget_exhausted" in message:
                return True
    return False


def run_id_taken(run_id: str, requests_root: Path, runs_root: Path = RUNS_ROOT) -> bool:
    """An id is spent once anything was written under it, and is never reused.

    Any sidecar counts, not only the request: a locate that failed before writing its
    receipt still left a ledger holding that attempt's reservation.
    """

    return any(requests_root.glob(f"{run_id}.*")) or _job_exists(run_id, runs_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cells", nargs="+", type=Path, help="cell.json files")
    parser.add_argument("--menus-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--requests-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--continuation", choices=("informed", "neutral_v1"), default="informed")
    parser.add_argument("--repeat-start", type=int, default=1,
                        help="first repeat number; evaluation runs start after the screening repeats")
    parser.add_argument("--prefix-state", choices=("replay", "snapshot"), default="replay",
                        help="snapshot: every arm starts from the cell's verified container snapshot")
    parser.add_argument("--estimates", choices=("uniform", "observed_existence_v1"), default="uniform",
                        help="check-outcome estimates written into each cell matrix")
    parser.add_argument("--network-wait-seconds", type=int, default=0,
                        help="before each attempt, wait up to this long for the verifier hosts")
    parser.add_argument("--budget-gate", action="store_true",
                        help="repairs that would leave the Actor's next call unfunded are not chosen")
    parser.add_argument("--verify-only-fallback", action="store_true",
                        help="keep each run's post-Actor state; rerun only the verifier on a lost report")
    parser.add_argument("--batched-snapshot", action="store_true",
                        help="every arm's public observer reads all paths in one exec (matrix observer block)")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--arms", nargs="+", default=[*MANAGED_ARMS, BINARY_ARM])
    parser.add_argument("--corpus-root", type=Path, required=True, help="reference corpus directory (not shipped with this demo)")
    args = parser.parse_args()
    python = sys.executable
    scripts = HERE.parent
    args.requests_root.mkdir(parents=True, exist_ok=True)
    args.matrix_root.mkdir(parents=True, exist_ok=True)
    # Resume: a (cell, arm, repeat) with a scored row is done; launched run ids are never reused.
    scored: set[tuple[str, str, int]] = set()
    # Attempts lost to an exhausted provider quota do not count against MAX_TRIES.
    quota_lost: set[str] = set()
    if QUOTA_BACKFILL.is_file():
        quota_lost |= {json.loads(line)["run_id"] for line in QUOTA_BACKFILL.read_text().splitlines() if line.strip()}
    invalid = invalidated_run_ids()
    quota_lost |= invalid  # an invalidated attempt does not use one of the tries either
    if args.manifest.is_file():
        for line in args.manifest.read_text().splitlines():
            row = json.loads(line)
            if row["verifier_report"] and row["run_id"] not in invalid:
                scored.add((row["cell_id"], row["arm"], row["repeat"]))
            if row.get("infrastructure_failure") == "provider_quota_exhausted":
                quota_lost.add(row["run_id"])
    for cell_path in args.cells:
        cell = json.loads(cell_path.read_text())
        cell_id = cell["cell_id"]
        task = cell_path.parent.parent.name
        condition = "" if args.continuation == "informed" else f"__{args.continuation}"
        snap_tag = "__snap" if args.prefix_state == "snapshot" else ""
        est_tag = "" if args.estimates == "uniform" else "__oe1"
        # Matrix version tag, keyed on the newest TraceDP arm the matrix carries.
        version_tag = (
            "__m10" if "tracedp-v4" in args.arms
            else "__m9" if "tracedp-v3of" in args.arms
            else "__m8" if "tracedp-v3o" in args.arms
            else "__m7" if "tracedp-v3c" in args.arms
            else "__m6" if "tracedp-v3" in args.arms
            else "__m5" if "tracedp-v2hri" in args.arms
            else "__m4" if "tracedp-v2hr" in args.arms
            else "__m3" if "tracedp-v2h" in args.arms
            else "__m2" if "tracedp-v2" in args.arms else ""
        )
        obs_tag = (("__obsb" if args.batched_snapshot else "") + ("__vo" if args.verify_only_fallback else "")
                   + ("__bg" if args.budget_gate else ""))
        matrix = args.matrix_root / f"{cell_id}{condition}{snap_tag}{est_tag}{version_tag}{obs_tag}.json"
        if not matrix.exists():
            try:
                cell_matrix_command(
                argparse.Namespace(
                    cell=cell_path,
                    menu=args.menus_root / task / "frozen_menu.json",
                    template=args.template,
                    continuation=args.continuation,
                    prefix_state=args.prefix_state,
                    estimates=args.estimates,
                    batched_snapshot=args.batched_snapshot,
                    verify_only_fallback=args.verify_only_fallback,
                    budget_gate=args.budget_gate,
                    out=matrix,
                )
                )
            except FileExistsError:
                pass  # a parallel launcher for the same cell wrote it first
        matrix_payload = json.loads(matrix.read_text())
        matrix_arms = matrix_payload["arms"]
        docker_context = str(matrix_payload["shared_environment"]["docker_context"])
        declared = {arm["arm_id"] for arm in matrix_arms}
        hypothesis_locators = {
            arm["arm_id"]: arm["hypothesis_locator"] for arm in matrix_arms if arm.get("hypothesis_locator")
        }
        rollback_arms = {arm["arm_id"] for arm in matrix_arms if arm.get("rollback_on_hypothesis")}
        intervention_arms = {arm["arm_id"] for arm in matrix_arms if arm.get("dover_intervention")}
        plan_arms = {arm["arm_id"] for arm in matrix_arms if arm.get("supervisor_plan")}
        missing = [arm for arm in args.arms if arm not in declared and arm not in LOCATOR_ARMS]
        if missing:
            raise ValueError(f"{matrix.name} does not declare arms: {', '.join(missing)}")
        for repeat in range(args.repeat_start, args.repeat_start + args.repeats):
            for arm in _arm_order(args.arms, seed=args.seed, cell_id=cell_id, repeat=repeat):
                if (cell_id, arm, repeat) in scored:
                    continue
                used = 0
                for attempt in range(1, 100):
                    if used >= MAX_TRIES:
                        break
                    tag = ("" if args.continuation == "informed" else "_n") + (
                        "_snap" if args.prefix_state == "snapshot" else ""
                    ) + ("" if args.estimates == "uniform" else "_oe1")
                    run_id = f"ci_{cell_id}_{arm.replace('-', '_')}{tag}_r{repeat}" + (f"_try{attempt}" if attempt > 1 else "")
                    request = args.requests_root / f"{run_id}.json"
                    if run_id_taken(run_id, args.requests_root):
                        # Left behind by an earlier launch: keep it, take the next id.
                        used += run_id not in quota_lost
                        continue
                    used += 1
                    if args.network_wait_seconds:
                        wait_for_network(docker_context, max_wait_seconds=args.network_wait_seconds)
                    common = ["--matrix", str(matrix), "--source-session", cell["session"], "--run-id", run_id,
                              "--request", str(request)]
                    if arm in LOCATOR_ARMS:
                        bridge = [python, str(scripts / "run_binary_bridge_arm.py"), "--arm-id", arm]
                        code = _run([*bridge, "--stage", "locate", *common, "--corpus-root", str(args.corpus_root)])
                        if code == 0:
                            code = _run([*bridge, "--stage", "prepare", *common])
                        if code == 0:
                            code = _run([*bridge, "--stage", "run", *common])
                    else:
                        locator = hypothesis_locators.get(arm)
                        code = 0
                        if locator:
                            # TraceDP's hypothesis: the author locator, charged to this arm.
                            code = _run([python, str(scripts / "run_binary_bridge_arm.py"), "--arm-id", locator,
                                         "--budget-arm-id", arm, "--stage", "locate", *common,
                                         "--corpus-root", str(args.corpus_root)])
                            code = 0 if code in (0, METHOD_FAILURE_EXIT) else code
                            if code == 0 and arm in rollback_arms:
                                _prepare_rollback(cell_path, request, python, scripts)
                            if code == 0 and arm in intervention_arms:
                                # DoVer's intervention at the hypothesized step.
                                code = _run([python, str(scripts / "run_dover_intervention_stage.py"), *common,
                                             "--corpus-root", str(args.corpus_root)])
                        if code == 0 and arm in plan_arms:
                            # The supervisor DP chooses the hand-off before the Actor resumes.
                            code = _run([python, str(scripts / "run_supervisor_plan_stage.py"), *common,
                                         "--corpus-root", str(args.corpus_root), "--cell", str(cell_path),
                                         "--arm-id", arm])
                        if code == 0:
                            code = _run([python, str(scripts / "run_historical_managed_arm.py"), "--arm-id", arm,
                                         "--source-session-cwd", cell["session_cwd"], "--mode", "run", *common])
                    report = has_verifier_report(run_id)
                    verified = settle_post_actor_image(
                        run_id, request, task=task, report=report, python=python, scripts=scripts,
                        docker_context=docker_context,
                    )
                    if verified is not None and verified.get("valid"):
                        report = True
                    row = {"cell_id": cell_id, "task": task, "motif": cell["motif"], "arm": arm,
                           "repeat": repeat, "attempt": attempt, "run_id": run_id, "exit_code": code,
                           "verifier_report": report, "request": str(request)}
                    if verified is not None:
                        row["verify_only"] = verified
                    if arm in LOCATOR_ARMS and code == METHOD_FAILURE_EXIT:
                        # The locator failed as a method (budget or invalid output): a scored
                        # 0, never retried (run_binary_bridge_arm.method_failure_reason).
                        receipt = json.loads(request.with_suffix(".locate.json").read_text())
                        row["method_failure"] = receipt["method_failure"]
                    elif arm not in LOCATOR_ARMS and not report and managed_budget_exhausted(run_id, request):
                        # The Actor exhausted the shared pool under complete metering: a method
                        # failure (scored 0, never retried), not an unreported infrastructure run.
                        row["method_failure"] = "budget_exhausted"
                    quota_out = not report and not row.get("method_failure") and provider_quota_exhausted()
                    if quota_out:
                        row["infrastructure_failure"] = "provider_quota_exhausted"
                    with args.manifest.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                    if quota_out:
                        print("provider quota exhausted: stopping this runner", file=sys.stderr, flush=True)
                        return QUOTA_EXHAUSTED_EXIT
                    if report or row.get("method_failure"):
                        break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
