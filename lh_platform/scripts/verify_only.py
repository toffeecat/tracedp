#!/usr/bin/env python3
"""Run only the official verifier on a frozen filesystem state, retrying on network loss.

The Terminal-Bench 2.1 ``tests/test.sh`` installs curl, uv and pytest from the network
while it scores. When that download fails the run has no CTRF report and its reward is
not a verdict, but the Actor (or replay) phase was fine. Repeating that phase to get a
verdict wastes a long replay and, for a paid run, changes the run. Instead, the state is
taken as an image (a materialized ladder rung, or a container committed after the Actor
phase), a derived task points ``docker_image`` at it, and a zero-call job runs the
unchanged official verifier on it, up to ``--attempts`` times.

Filesystem only: a task whose tests need processes the Actor left running (servers)
cannot be re-verified this way and is refused by ``--service-tasks``.

    python lh_platform/scripts/verify_only.py --task build-pmars --image longh-rung/x:p0 \
        --label build_pmars_final --job-root <storage>/verify_only --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import controlled_injection as ci  # noqa: E402
from lh_platform.benchmarks.terminal_prefix_snapshot import docker_command  # noqa: E402

SCHEMA = "terminal_verify_only_v1"
# Tasks whose official tests talk to a service the Actor started (spec v1 source table).
DEFAULT_SERVICE_TASKS = ("configure-git-webserver", "headless-terminal", "nginx-request-logging", "pypi-server")


# Why an official verifier produced no report. Network loss is transient and worth a retry;
# a broken package state inside the kept image is not, and retrying it only repeats the failure
# (for example, an interrupted dpkg in the kept image makes every retry fail the same way).
CAUSE_NETWORK = "network"
CAUSE_STATE = "state_unscorable"
CAUSE_UNKNOWN = "unknown"
RETRYABLE_CAUSES = frozenset({CAUSE_NETWORK, CAUSE_UNKNOWN})
_NETWORK_MARKERS = (
    "Could not resolve host", "Failed to connect", "failed to download", "Couldn't connect to server",
    "Connection timed out", "Temporary failure in name resolution", "connection closed prematurely",
)
_STATE_MARKERS = ("dpkg was interrupted", "dpkg --configure -a")


def classify_verifier_failure(trial: Path) -> str:
    """The cause of a missing report, read from the verifier's own output.

    State markers are checked first: a broken package state also produces download-style
    messages later in the output, and that state is what must be reported.
    """

    stdout = trial / "verifier" / "test-stdout.txt"
    text = stdout.read_text(errors="replace") if stdout.is_file() else ""
    if any(marker in text for marker in _STATE_MARKERS):
        return CAUSE_STATE
    if any(marker in text for marker in _NETWORK_MARKERS):
        return CAUSE_NETWORK
    return CAUSE_UNKNOWN


def derived_task_root(official_root: Path, derived_root: Path, task: str, image: str) -> Path:
    """``<derived_root>/tasks/<task>``: the official task with ``docker_image`` = ``image``."""

    target = derived_root / "tasks" / task
    if not target.exists():
        shutil.copytree(official_root / "tasks" / task, target)
        toml = target / "task.toml"
        lines = toml.read_text(encoding="utf-8").splitlines()
        hits = [i for i, line in enumerate(lines) if line.strip().startswith("docker_image")]
        if len(hits) != 1:
            raise ValueError(f"{task}: expected exactly one docker_image line")
        lines[hits[0]] = f'docker_image = "{image}"'
        toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    text = (target / "task.toml").read_text(encoding="utf-8")
    if f'docker_image = "{image}"' not in text:
        raise ValueError(f"derived task {target} points at another image")
    return derived_root


def image_id_of(image: str) -> str:
    """The image ID behind a tag, recorded so a verdict names exactly the image it ran on.

    Tags are reused across attempts (two different IDs were once seen under one tag), so a
    verdict without the ID cannot be traced back to its image.
    """

    completed = subprocess.run(
        docker_command("image", "inspect", "--format", "{{.Id}}", image),
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ValueError(f"image not found: {image}")
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--image", required=True, help="tag of the frozen state")
    parser.add_argument("--label", required=True, help="unique job label")
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--public-paths", nargs="*", help="defaults to the census source's public paths")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--service-tasks", nargs="*", default=list(DEFAULT_SERVICE_TASKS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.task in args.service_tasks:
        raise SystemExit(f"{args.task}: its tests need live processes; a filesystem state cannot be re-verified")
    official = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    public_paths = args.public_paths
    if not public_paths:
        runs_root = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
        config = json.loads((runs_root / ci.CENSUS_RUN.format(slug=args.task.replace("-", "_")) / "config.json").read_text())
        public_paths = list(config["agents"][0]["kwargs"]["public_paths"])
    image_id = image_id_of(args.image)
    derived = derived_task_root(official, args.job_root / "derived" / args.label, args.task, args.image)
    previous = os.environ["LONGH_TERMINAL_BENCH_ROOT"]
    os.environ["LONGH_TERMINAL_BENCH_ROOT"] = str(derived)
    try:
        attempts = []
        result = None
        for attempt in range(1, args.attempts + 1):
            result = ci._run_replay_once(
                job_root=args.job_root, run_id=f"verify_only_{args.label}_a{attempt}",
                task=args.task, public_paths=public_paths, chain=[],
            )
            cause = None if result["verifier_report"] else classify_verifier_failure(Path(result["trial"]))
            attempts.append({"run_id": result["run_id"], "verifier_report": result["verifier_report"],
                             "reward": result["reward"], "exception": result["exception"], "cause": cause})
            if result["verifier_report"] or cause not in RETRYABLE_CAUSES:
                break
    finally:
        os.environ["LONGH_TERMINAL_BENCH_ROOT"] = previous
    report = {
        "schema_version": SCHEMA,
        "task": args.task,
        "image": args.image,
        "image_id": image_id,
        "attempts": attempts,
        "valid": bool(result and result["verifier_report"]),
        "reward": result["reward"] if result and result["verifier_report"] else None,
        "unscorable_cause": attempts[-1]["cause"] if attempts and attempts[-1]["cause"] == CAUSE_STATE else None,
        "limits": "filesystem only: processes the Actor left running are not part of the state",
    }
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
