#!/usr/bin/env python3
"""Rescore runs whose verifier was lost to the network, on the kept post-Actor image.

A row written by run_controlled_injection_pairs with ``verify_only.kept_image`` and no valid
verdict has its Actor state still in Docker. Once the network is back, the official test.sh
is rerun on that image (verify_only.py); a valid verdict appends a rescored row, which the
summary reads as the attempt's verdict. The image is removed after a valid verdict, or after
a state-unscorable one. The Actor is never rerun.

usage: rescore_kept_images.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

CI = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback" / "controlled_injection_v1"
DOCKER_CONTEXT = os.environ.get("LONGH_DOCKER_CONTEXT", "default")


def pending_rows(manifest: Path) -> list[dict]:
    """Rows whose kept image still awaits a valid verdict, with the last one per attempt."""

    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    kept = {}
    for row in rows:
        summary = row.get("verify_only") or {}
        if summary.get("kept_image") and not summary.get("valid"):
            kept[(row["run_id"], row["attempt"])] = row
    return list(kept.values())


def rescored_row(row: dict, verdict: dict, out: Path) -> dict:
    """The original row with the verdict, appended so the summary reads it as this attempt's."""

    return {
        **row,
        "verifier_report": bool(verdict.get("valid")),
        "rescored": True,
        "verify_only": {
            "report": str(out), "valid": bool(verdict.get("valid")), "reward": verdict.get("reward"),
            "attempts": len(verdict.get("attempts", [])), "unscorable_cause": verdict.get("unscorable_cause"),
            "kept_image": None if verdict.get("valid") or verdict.get("unscorable_cause") else row["verify_only"]["kept_image"],
        },
    }


def _rescore_one(row: dict, manifest: Path, dry_run: bool) -> str:
    image = row["verify_only"]["kept_image"]
    out = Path(row["request"]).with_suffix(".rescore.json")
    if dry_run:
        return f"would rescore {row['run_id']} on {image}"
    if subprocess.run(["docker", "--context", DOCKER_CONTEXT, "image", "inspect", image],
                      capture_output=True).returncode:
        return f"image gone, cannot rescore {row['run_id']}"
    subprocess.run([sys.executable, str(HERE.parent / "verify_only.py"), "--task", row["task"], "--image", image,
                    "--label", row["run_id"], "--job-root", str(CI / "verify_only"), "--out", str(out)],
                   capture_output=True)
    if not out.is_file():
        return f"no verdict written for {row['run_id']}"
    verdict = json.loads(out.read_text())
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(rescored_row(row, verdict, out)) + "\n")
    if verdict.get("valid") or verdict.get("unscorable_cause"):
        subprocess.run(["docker", "--context", DOCKER_CONTEXT, "image", "rm", "-f", image], capture_output=True)
    return f"{row['run_id']}: valid={verdict.get('valid')} reward={verdict.get('reward')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for manifest in sorted(CI.glob("pair_manifest_*.jsonl")):
        for row in pending_rows(manifest):
            print(_rescore_one(row, manifest, args.dry_run), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
