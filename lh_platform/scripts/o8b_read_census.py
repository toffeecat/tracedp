"""O8b read-variant census: read->write value relays, and whether the read file is a task input.

Zero model calls. For each census source it loads the frozen trajectory with the cell builder's own
loader, finds value relays (``output_relay_edges`` + ``relay_sites``, the same rule as O8b), and keeps
the ones whose producer is a ``read`` call. One JSON line per source is printed; that output is the
receipt cited by the O8b read-variant preregistration.

Two deliberate choices, both recorded in each line:
- ``allow_tool_use_ending=True`` is passed for every source, so the census sees the same sources as the
  admission receipts (cobol-modernization needs it). It does not admit anything.
- When the observer log is one row short of the calls and the last call was executed, one empty row is
  appended (``padded_final_row``). Relay detection reads paths and write targets from the arguments, so the
  padding cannot create or hide a read relay.

Usage (from lh_platform/, with LONGH_STORAGE_ROOT and LONGH_TERMINAL_BENCH_ROOT set):
    python3 scripts/o8b_read_census.py [TASK ...] > census.jsonl
With no TASK arguments, every census-pool source is read.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import controlled_injection as ci

from lh_platform.tracefix_bench.dependency_graph import (
    call_accesses,
    output_relay_edges,
    relay_sites,
)

CENSUS_PREFIX = "terminal_minimax_m3_token_plan_source_census_full89_v2_attempt1_"
_LINE_LABEL = re.compile(r"^\d+: ", re.MULTILINE)


def content_occurrences(read_output: str, token: str) -> int:
    """How often ``token`` occurs in a numbered read output, line labels excluded."""

    return _LINE_LABEL.sub("", read_output).count(token)


def _tool_outputs(session: Path, prefix: list[dict]) -> list[str]:
    texts: dict[str, str] = {}
    with session.open() as handle:
        for line in handle:
            row = json.loads(line)
            message = row.get("message", {})
            if row.get("type") == "message" and message.get("role") == "toolResult":
                texts[str(message.get("toolCallId"))] = "".join(
                    str(part.get("text", "")) for part in message.get("content", []) if isinstance(part, dict)
                )
    return [texts.get(str(call.get("source_tool_call_id")), "") for call in prefix]


def _input_names(terminal_root: Path, task: str) -> set[str]:
    inputs = terminal_root / "tasks" / task / "environment" / "inputs"
    return {path.name for path in inputs.glob("*")} if inputs.is_dir() else set()


def census_one(task: str, runs: Path, terminal_root: Path, storage_root: Path) -> dict:
    record: dict = {
        "task": task,
        "admitted": (storage_root / "runs/traceback/controlled_injection_v1/phase0/receipts"
                     / f"source_admission_{task}_h.json").is_file(),
    }
    try:
        source = ci.load_census_source(runs, terminal_root, task, allow_tool_use_ending=True)
    except Exception as error:  # the loader's own refusal is the evidence; keep the reason
        return {**record, "status": "load_failed", "error": str(error)[:200]}
    prefix = source["prefix"]
    rows = ci._read_jsonl(source["session"].parent.parent.parent / "terminal-public-state.jsonl")
    padded = len(rows) == len(prefix) - 1 and bool(prefix[-1].get("environment_step_executed"))
    if padded:
        rows = rows + [{"before": [], "after": [], "tool_result": {"content": ""}}]
    if len(rows) != len(prefix):
        return {**record, "status": "row_mismatch", "calls": len(prefix), "rows": len(rows)}
    outputs = _tool_outputs(source["session"], prefix)
    edges = output_relay_edges(prefix, outputs, instruction=source["instruction"])
    relays = relay_sites(call_accesses(prefix, rows, cwd="/app"), edges)
    inputs = _input_names(terminal_root, task)
    read_relays = []
    for relay in relays:
        producer, consumer = int(relay["producer"]), int(relay["consumer"])
        if prefix[producer]["tool_name"] != "read":
            continue
        path = str(prefix[producer]["arguments"].get("path", ""))
        token = str(relay["token"])
        read_relays.append({
            "producer": producer,
            "path": path,
            "consumer": consumer,
            "consumer_tool": prefix[consumer]["tool_name"],
            "distance": consumer - producer,
            "token": token,
            "occurrences_in_read": content_occurrences(outputs[producer], token),
            "input_file": Path(path).name in inputs,
        })
    return {
        **record,
        "status": "analysed",
        "calls": len(prefix),
        "padded_final_row": padded,
        "relays_any": len(relays),
        "read_relays": read_relays,
    }


def main(argv: list[str]) -> int:
    storage_root = Path(os.environ["LONGH_STORAGE_ROOT"])
    runs = storage_root / "runs" / "traceback"
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    tasks_dir = terminal_root / "tasks"
    by_slug = {path.name.replace("-", "_"): path.name for path in tasks_dir.iterdir() if path.is_dir()}
    if argv:
        tasks = argv
    else:
        slugs = {d.name[len(CENSUS_PREFIX):].rsplit("_attempt1", 1)[0] for d in runs.glob(CENSUS_PREFIX + "*")}
        tasks = [by_slug[slug] for slug in sorted(slugs) if slug in by_slug]
    for task in tasks:
        print(json.dumps(census_one(task, runs, terminal_root, storage_root), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
