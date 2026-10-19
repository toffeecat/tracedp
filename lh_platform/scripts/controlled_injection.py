#!/usr/bin/env python3
"""Build and qualify controlled injection cells (CONTROLLED_INJECTION_SPEC_V1).

Every stage here is zero-model: frozen tool chains are replayed by
``HostTerminalReplayAgent`` inside the official task image, and the official verifier
scores the result. Each replay job gets a verifier byte contract before Harbor starts.

Stages:
``admit-sources``  S1-S3 for successful census trajectories (two replays each).
``menu-brief``     the blind public-tier menu brief for one task (instruction + environment/).
``menu-freeze``    validate an author's menu against the brief and freeze contracts + receipt.
``build-cells``    menu sanity on the success state, then seeded single/double-fault cells with
                   Q2-Q5 and motif classification, packaged as continuation sources.
``cell-matrix``    a v5 recovery matrix for one cell: public menu, mechanical public repairs
                   (spec §7), uniform estimates, the cell budget for every arm.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import html
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from terminal_historical_managed_launch import (  # noqa: E402
    frozen_source_cwd,
    prefix_from_frozen_source,
)

from lh_platform.benchmarks.terminal_observed_runtime import (  # noqa: E402
    ObservedTerminalRunSpec,
    build_controlled_terminal_replay_harbor_config,
)
from lh_platform.benchmarks.terminal_pi_invocation import neutral_actor_cwd  # noqa: E402
from lh_platform.benchmarks.terminal_prefix_parity import classify_prefix_parity  # noqa: E402
from lh_platform.benchmarks.terminal_runtime_check import (  # noqa: E402
    RUNTIME_CHECK_PREDICATE_KINDS,
    SUBJECT_REFUTABLE_PRESENCE_POLICY,
    RuntimeCheckContract,
)
from lh_platform.benchmarks.terminal_verifier_contract import (  # noqa: E402
    write_job_verifier_contract,
)
from lh_platform.benchmarks.terminal_prefix_snapshot import SNAPSHOT_RECEIPT_SCHEMA, docker_command
from lh_platform.tracefix_bench.cell_funnel import FunnelLedger  # noqa: E402
from lh_platform.tracefix_bench.controlled_injection import (  # noqa: E402
    SPEC_ID,
    _compatible,
    apply_sites,
    cell_budget,
    classify_motif,
    dumps_jsonl,
    enumerate_sites,
    menu_check_calls,
    parse_menu_verdicts,
    refuted,
    replay_history,
    rewrite_observer,
    rewrite_session,
)

CENSUS_RUN = "terminal_minimax_m3_token_plan_source_census_full89_v2_attempt1_{slug}_attempt1"
DEFAULT_SUCCESS_TASKS = (
    "build-pmars", "build-pov-ray", "constraints-scheduling", "fix-git", "git-leak-recovery",
    "log-summary-date-ranges", "multi-source-data-merger", "nginx-request-logging",
    "prove-plus-comm", "pypi-server", "regex-log",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _only(paths: list[Path], label: str) -> Path:
    paths = [path for path in paths if not path.name.startswith("._")]
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(paths)}")
    return paths[0]


def terminal_stop_reason(session: Path) -> str | None:
    """``stopReason`` of the session's last assistant message."""

    reason = None
    for row in _read_jsonl(session):
        message = row.get("message", {})
        if row.get("type") == "message" and message.get("role") == "assistant":
            reason = message.get("stopReason")
    return reason


def load_census_source(
    runs_root: Path, terminal_root: Path, task: str, *, allow_tool_use_ending: bool = False
) -> dict:
    """The frozen successful trajectory of ``task`` and everything its replay needs.

    A source whose last assistant turn ended in ``toolUse`` (the run stopped after a
    completed tool result) has a fully determined prefix too; S3 admits it only when asked.
    """

    run = runs_root / CENSUS_RUN.format(slug=task.replace("-", "_"))
    config = json.loads((run / "config.json").read_text())
    trial = _only(list(run.glob("*__*")), "census trial")
    session = _only(list((trial / "agent/pi/sessions").glob("*.jsonl")), "Pi session")
    agent = config["agents"][0]
    provider, model = str(agent["model_name"]).split("/", maxsplit=1)
    instruction = (terminal_root / "tasks" / task / "instruction.md").read_text(encoding="utf-8")
    ending = terminal_stop_reason(session)
    stop_reason = "toolUse" if allow_tool_use_ending and ending == "toolUse" else "stop"
    prefix, message_count, session_id = prefix_from_frozen_source(
        source_session=session,
        source_session_cwd=frozen_source_cwd(session),
        task_instruction=instruction,
        provider=provider,
        model=model,
        thinking=str(agent["kwargs"]["thinking"]),
        expected_terminal_stop_reason=stop_reason,
    )
    return {
        "task": task,
        "terminal_stop_reason": stop_reason,
        "trial": trial,
        "session": session,
        "instruction": instruction,
        "public_paths": list(agent["kwargs"]["public_paths"]),
        "prefix": prefix,
        "message_count": message_count,
        "session_id": session_id,
    }


def require_vm_visible(path: Path) -> None:
    root = os.environ["LONGH_STORAGE_ROOT"]
    if not str(path).startswith(root.rstrip("/") + "/"):
        raise ValueError(f"{path} must use the VM-mounted storage spelling {root}")


def _run_replay_once(
    *, job_root: Path, run_id: str, task: str, public_paths: list[str], chain: list[dict],
    snapshot: dict | None = None, ladder: dict | None = None, agent_timeout_multiplier: float | None = None,
) -> dict:
    """One Harbor job for a frozen chain (or its finished result read back from disk)."""

    require_vm_visible(job_root)
    storage_root = Path(os.environ["LONGH_STORAGE_ROOT"])
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    config = build_controlled_terminal_replay_harbor_config(
        ObservedTerminalRunSpec(run_id=run_id, task=task, public_paths=tuple(public_paths)),
        storage_root=storage_root,
        terminal_bench_root=terminal_root,
        tool_calls=tuple(chain),
        container_proxy=os.environ.get("LONGH_CONTAINER_PROXY") or None,
        snapshot=snapshot,
        **({"ladder": ladder} if ladder is not None else {}),
        **({"agent_timeout_multiplier": agent_timeout_multiplier} if agent_timeout_multiplier is not None else {}),
    )
    job_root.mkdir(parents=True, exist_ok=True)
    config["jobs_dir"] = str(job_root)
    job_dir = job_root / run_id
    config_path = job_root / f"{run_id}.harbor.json"
    finished = [
        t for t in job_dir.glob("*__*") if not t.name.startswith("._") and (t / "result.json").is_file()
    ] if job_dir.exists() else []
    returncode = None
    if finished:
        # Resume: a completed replay is read back, never re-run. The frozen config must be
        # the one this call would have written, or the stored result answers another chain.
        if json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
            raise ValueError(f"completed replay {run_id} was run with a different chain")
    else:
        if job_dir.exists():
            raise FileExistsError(f"incomplete replay job already exists: {job_dir}")
        write_job_verifier_contract(job_dir, (Path(config["tasks"][0]["path"]),))
        with config_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        environment = {**os.environ, "DOCKER_CONTEXT": os.environ.get("LONGH_DOCKER_CONTEXT", "default")}
        environment.pop("DOCKER_HOST", None)
        completed = subprocess.run(
            [os.environ["LONGH_HARBOR"], "run", "--config", str(config_path), "--yes"],
            check=False,
            env=environment,
            capture_output=True,
            text=True,
        )
        returncode = completed.returncode
        (job_root / f"{run_id}.harbor.log").write_text(completed.stdout + completed.stderr)
    trial = _only(list(job_dir.glob("*__*")), "replay trial")
    result = json.loads((trial / "result.json").read_text())
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    exception = (result.get("exception_info") or {}).get("exception_message")
    session = trial / "agent/pi/sessions/controlled-replay.jsonl"
    history = replay_history(_read_jsonl(session)) if session.is_file() else []
    observer = trial / "agent/terminal-public-state.jsonl"
    # The verifier downloads uv at test time; without its CTRF report the reward is the
    # default 0, not a verdict on the chain (observed on 13 of the first replays).
    report = (trial / "verifier" / "ctrf.json").is_file()
    return {
        "run_id": run_id,
        "trial": str(trial),
        "agent_execution": result.get("agent_execution"),
        "verifier_report": report,
        "harbor_exit_code": returncode,
        "resumed_from_disk": returncode is None,
        "reward": reward,
        "exception": exception,
        "history": history,
        "observer_rows": _read_jsonl(observer) if observer.is_file() else [],
    }


REPLAY_ATTEMPTS = 3


def run_replay(
    *, job_root: Path, run_id: str, task: str, public_paths: list[str], chain: list[dict],
    snapshot: dict | None = None, ladder: dict | None = None, agent_timeout_multiplier: float | None = None,
) -> dict:
    """Replay a chain; a reward counts only with a verifier report, else retry under a new id.

    Unreported attempts stay on disk as evidence and are listed in ``attempts``. If every
    attempt lacks a report the result has ``valid = False`` and ``reward = None``: an
    infrastructure failure, never an official verdict.
    """

    attempts = []
    for attempt in range(1, REPLAY_ATTEMPTS + 1):
        suffix_id = run_id if attempt == 1 else f"{run_id}_try{attempt}"
        replay = _run_replay_once(
            job_root=job_root, run_id=suffix_id, task=task, public_paths=public_paths, chain=chain,
            **({"snapshot": snapshot} if snapshot is not None else {}),
            **({"ladder": ladder} if ladder is not None else {}),
            **({"agent_timeout_multiplier": agent_timeout_multiplier} if agent_timeout_multiplier is not None else {}),
        )
        attempts.append({"run_id": suffix_id, "verifier_report": replay["verifier_report"], "reward": replay["reward"]})
        if replay["verifier_report"]:
            return {**replay, "valid": True, "attempts": attempts}
    return {**replay, "reward": None, "valid": False, "attempts": attempts}


def _source_results(session: Path) -> dict[str, str]:
    texts = {}
    for row in _read_jsonl(session):
        message = row.get("message", {})
        if row.get("type") == "message" and message.get("role") == "toolResult":
            texts[str(message["toolCallId"])] = "".join(
                str(part.get("text", "")) for part in message.get("content", []) if isinstance(part, dict)
            )
    return texts


def _span_seconds(span: dict | None) -> float | None:
    """Seconds between a Harbor phase's ``started_at`` and ``finished_at``."""

    from datetime import datetime

    try:
        started = datetime.fromisoformat(str(span["started_at"]).replace("Z", "+00:00"))  # type: ignore[index]
        finished = datetime.fromisoformat(str(span["finished_at"]).replace("Z", "+00:00"))  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None
    return round((finished - started).total_seconds(), 1)


def _parity_stats(expected: list[dict], history: list[dict]) -> dict:
    """Per-call error-status agreement, the quantity long sources lose first."""

    pairs = list(zip(expected, history, strict=False))
    agree = [bool(call.get("source_is_error")) == bool(row.get("is_error")) for call, row in pairs]
    return {
        "replayed_calls": len(history),
        "error_status_agreement": round(sum(agree) / len(expected), 4) if expected else None,
        "first_error_status_difference": next((i for i, ok in enumerate(agree) if not ok), None),
    }


def _ladder_summary(trial: str) -> dict | None:
    path = Path(trial) / "agent" / "terminal-checkpoint-ladder.json"
    if not path.is_file():
        return None
    ladder = json.loads(path.read_text())
    rungs = ladder["rungs"]
    return {
        "receipt": str(path),
        "every": ladder["every"],
        "rungs": len(rungs),
        "base_image_id": ladder["base"]["image_id"],
        "base_repo_digest": ladder["base"].get("repo_digest"),
        "base_tar_bytes": ladder["base"].get("saved_tar_bytes"),
        "base_save_seconds": ladder["base"].get("save_seconds"),
        "segment_seconds": [rung["segment_seconds"] for rung in rungs],
        "capture_seconds": [rung["capture_seconds"] for rung in rungs],
        "delta_tar_bytes": [rung["delta_tar_bytes"] for rung in rungs],
    }


def admit_sources(args) -> int:
    """S1-S3 for each source; optionally a checkpoint ladder and write trace on replay 1.

    Every gate outcome also goes to the funnel ledger when ``--funnel`` is given, so a
    rejected long source (timeout, parity) stays in the denominator with its reason.
    """

    runs_root = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    job_root = args.job_root
    funnel = FunnelLedger(args.funnel) if getattr(args, "funnel", None) else None
    ladder = None
    if getattr(args, "ladder_every", None):
        if not getattr(args, "ladder_base_cache", None):
            raise ValueError("--ladder-every needs --ladder-base-cache")
        ladder = {"every": args.ladder_every, "base_cache": str(args.ladder_base_cache)}
    if getattr(args, "trace_writes", False):
        ladder = {**(ladder or {}), "trace_writes": True}
    # Bash semantics belong to the source, so both replays use them; the ladder and the
    # write trace stay on replay 1 only.
    era = {"source_era_bash": True} if getattr(args, "source_era_bash", False) else None
    if getattr(args, "batched_public_snapshot", False):
        # Observation speed only; applies to both replays like the Bash semantics.
        era = {**(era or {}), "batched_public_snapshot": True}
    if era is not None:
        ladder = {**(ladder or {}), **era}
    multiplier = getattr(args, "agent_timeout_multiplier", None)

    def mark(task: str, gate: str, passed: bool | None, reason: str | None = None, **detail) -> None:
        if funnel is not None:
            funnel.record(candidate_id=f"source:{task}", source_task=task, gate=gate, passed=passed,
                          reason=reason, detail=detail or None)

    rows = []
    for task in args.tasks:
        row: dict = {"task": task}
        try:
            source = load_census_source(
                runs_root, terminal_root, task, allow_tool_use_ending=getattr(args, "allow_tool_use_ending", False)
            )
        except Exception as error:  # S3 and source-integrity failures exclude the task
            rows.append({**row, "admitted": False, "stage": "S3", "reason": f"{type(error).__name__}: {error}"})
            mark(task, "S3_source_load", False, rows[-1]["reason"][:300])
            print(json.dumps(rows[-1]), flush=True)
            continue
        mark(task, "S3_source_load", True, calls=len(source["prefix"]), terminal_stop_reason=source["terminal_stop_reason"])
        source_texts = _source_results(source["session"])
        replays = []
        failure = None
        for attempt in (1, 2):
            try:
                replay = run_replay(
                    job_root=job_root,
                    run_id=f"source_admission_{task.replace('-', '_')}{'_' + args.tag if args.tag else ''}_r{attempt}",
                    task=task,
                    public_paths=source["public_paths"],
                    chain=source["prefix"],
                    **({"ladder": ladder} if ladder is not None and attempt == 1
                       else {"ladder": era} if era is not None else {}),
                    **({"agent_timeout_multiplier": multiplier} if multiplier is not None else {}),
                )
            except Exception as error:  # a task that cannot be replayed is excluded, not fatal
                failure = f"{type(error).__name__}: {error}"
                break
            history = [
                {**item, "source_tool_call_id": call["source_tool_call_id"]}
                for item, call in zip(replay["history"], source["prefix"], strict=False)
            ]
            exemptions, mismatch = classify_prefix_parity(source["prefix"], history)
            identical = sum(
                1
                for item, call in zip(replay["history"], source["prefix"], strict=False)
                if item.get("content") == source_texts.get(str(call["source_tool_call_id"]))
            )
            replays.append(
                {
                    "run_id": replay["run_id"],
                    "trial": replay["trial"],
                    "valid": replay["valid"],
                    "reward": replay["reward"],
                    "exception": replay["exception"],
                    "parity_mismatch": mismatch,
                    "parity_exemptions": len(exemptions),
                    "output_identical_to_source": identical,
                    "output_identical_rate": round(identical / len(source["prefix"]), 4) if source["prefix"] else None,
                    "call_count": len(source["prefix"]),
                    "agent_seconds": _span_seconds(replay.get("agent_execution")),
                    **_parity_stats(source["prefix"], history),
                }
            )
        # An injected cell starts from a success; a natural-failure cell must reproduce
        # the source's official failure, with a report, on both replays.
        if failure is not None:
            rows.append({**row, "admitted": False, "stage": "S1", "reason": failure})
            # The replay machinery raised: an infrastructure outcome, not a verdict on the source.
            mark(task, "S1_source_replay", None, failure[:300])
            print(json.dumps(rows[-1]), flush=True)
            continue
        wanted = 0 if getattr(args, "natural", False) else 1
        s1 = all(r["valid"] and r["reward"] == wanted and (wanted == 1 or r["exception"] is None) for r in replays)
        s2 = all(r["parity_mismatch"] is None for r in replays)
        if not all(r["valid"] for r in replays):
            mark(task, "S1_source_replay", None, "no verifier report on a replay")
        else:
            reasons = sorted({str(r["exception"] or f"reward {r['reward']}") for r in replays if r["reward"] != wanted})
            mark(task, "S1_source_replay", s1, "; ".join(reasons)[:300] if not s1 else None,
                 rewards=[r["reward"] for r in replays], agent_seconds=[r["agent_seconds"] for r in replays])
        mark(task, "S2_source_parity", s2,
             None if s2 else str(next(r["parity_mismatch"] for r in replays if r["parity_mismatch"] is not None))[:300],
             agreement=[r["error_status_agreement"] for r in replays])
        admitted = s1 and s2
        ladder_row = _ladder_summary(replays[0]["trial"]) if ladder is not None and ladder.get("every") else None
        if ladder is not None and ladder.get("every"):
            mark(task, "ladder", (ladder_row is not None) and admitted,
                 None if (ladder_row is not None and admitted) else
                 ("ladder not written" if ladder_row is None else "source not admitted, ladder not accepted"),
                 **({k: v for k, v in ladder_row.items() if k != "receipt"} if ladder_row else {}))
        rows.append(
            {
                **row,
                "admitted": admitted,
                "stage": "S1-S2",
                "call_count": len(source["prefix"]),
                "terminal_stop_reason": source["terminal_stop_reason"],
                "session": str(source["session"]),
                "replays": replays,
                **({"ladder": {**ladder_row, "accepted": admitted}} if ladder_row is not None else {}),
            }
        )
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "session"}), flush=True)
    receipt = {"schema_version": "controlled_injection_source_admission_v1", "spec_id": SPEC_ID, "rows": rows}
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return 0


MENU_BRIEF_SCHEMA = "controlled_injection_menu_brief_v1"
_BRIEF_FILE_LIMIT = 6000
_FORBIDDEN_COMMAND_TOKENS = ("/tests", "tests/", "solution", "test_outputs", "pytest")
MENU_RULES = """\
You write read-only acceptance checks for a terminal task, using ONLY the task instruction
and the environment files below. You have not seen, and must not guess at, the task's
hidden tests or reference solution.

Return one JSON object {"checks": [...]} and nothing else. Each check has:
  check_id            short kebab-case id, unique
  scope               "broad" (one cheap check covering several requirements) or "narrow"
                      (exactly one requirement)
  proposition         one sentence: what holds when the check passes
  instruction_excerpt an EXACT, contiguous substring of the instruction this check enforces
  subject_paths       the public paths it reads; each must be in PUBLIC PATHS below
  predicate_kind      one of: existence, shape, consistency, behavioral_discriminator
  command             a bash command run from /app that exits 0 when the proposition holds
                      and 1 when it does not; it must not modify anything, must not reference
                      tests or solutions, and must finish within 60 seconds
Rules: at least 1 broad and at least 2 narrow checks, at most 8 checks in total. Use only
facts stated in the instruction or visible in the environment files; if a requirement
cannot be checked from those, skip it.
"""


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def menu_brief(args) -> int:
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    runs_root = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
    task_dir = terminal_root / "tasks" / args.task
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    run = runs_root / CENSUS_RUN.format(slug=args.task.replace("-", "_"))
    public_paths = json.loads((run / "config.json").read_text())["agents"][0]["kwargs"]["public_paths"]
    parts = [MENU_RULES, "PUBLIC PATHS:\n" + "\n".join(public_paths), "INSTRUCTION:\n" + instruction]
    for path in sorted((task_dir / "environment").rglob("*")):
        if not path.is_file() or path.name.startswith("._"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f"<binary, {path.stat().st_size} bytes>"
        relative = path.relative_to(task_dir)
        parts.append(f"ENVIRONMENT FILE {relative}:\n{text[:_BRIEF_FILE_LIMIT]}")
    brief = "\n\n".join(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(brief)
    print(json.dumps({"task": args.task, "brief": str(args.out), "brief_sha256": _sha256(brief)}))
    return 0


def _expand_check(item: dict, *, instruction: str, public_paths: list[str]) -> tuple[dict, dict]:
    """One author check as a bound contract and its predicate; raise on any rule breach."""

    excerpt = str(item.get("instruction_excerpt", ""))
    start = instruction.find(excerpt)
    if not excerpt or start < 0:
        raise ValueError("excerpt is not an exact instruction substring")
    subjects = [str(p) for p in item.get("subject_paths", [])]
    if not subjects or set(subjects) - set(public_paths):
        raise ValueError("subject paths must be nonempty public paths")
    # The agent transport HTML-escapes &, < and >; the raw menu hash keeps the original.
    command = html.unescape(str(item.get("command", ""))).strip()
    if not command or any(token in command for token in _FORBIDDEN_COMMAND_TOKENS):
        raise ValueError("command is empty or references tests/solutions")
    if item.get("predicate_kind") not in RUNTIME_CHECK_PREDICATE_KINDS:
        raise ValueError("predicate kind is unsupported")
    if item.get("scope") not in ("broad", "narrow"):
        raise ValueError("scope is neither broad nor narrow")
    check_id = f"pub-{item['check_id']}"
    predicate_id = f"pred-{item['check_id']}"
    contract = {
        "task_id": "public-task",
        "proposition": str(item["proposition"]),
        "arguments": {"command": f"cd /app && {command}", "timeout": 60},
        "supported_output": "exit_0_acceptance_condition_holds",
        "refuted_output": "exit_1_acceptance_condition_fails",
        "resource_ids": subjects,
        "check_id": check_id,
        "predicate_id": predicate_id,
        "verdict_protocol": "exit_status_v1",
        "presence_policy": SUBJECT_REFUTABLE_PRESENCE_POLICY,
        "prerequisite_resource_ids": [],
        "predicate_kind": item["predicate_kind"],
        "read_only": True,
    }
    RuntimeCheckContract(**{**contract, "resource_ids": tuple(subjects)})
    predicate = {
        "predicate_id": predicate_id,
        "task_id": "public-task",
        "proposition": str(item["proposition"]),
        "source_start": start,
        "source_end": start + len(excerpt),
        "source_excerpt": excerpt,
        "resource_ids": subjects,
        "aggregation": "all",
    }
    return {**contract, "menu_scope": item["scope"]}, predicate


def freeze_menu(*, task: str, instruction: str, public_paths: list[str], menu: dict) -> dict:
    """Expand an author menu; drop checks that break a rule, then require the structure."""

    checks = menu.get("checks")
    if not isinstance(checks, list) or not 1 <= len(checks) <= 8:
        raise ValueError("a menu has 1 to 8 checks")
    ids = [str(item.get("check_id", "")) for item in checks]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("check ids must be unique and nonempty")
    contracts, predicates, rejected = [], [], []
    for item in checks:
        try:
            contract, predicate = _expand_check(item, instruction=instruction, public_paths=public_paths)
        except (KeyError, ValueError) as error:
            rejected.append({"check_id": item.get("check_id"), "reason": str(error)})
            continue
        contracts.append(contract)
        predicates.append(predicate)
    scopes = [contract["menu_scope"] for contract in contracts]
    if scopes.count("broad") < 1 or scopes.count("narrow") < 2:
        raise ValueError(
            "after dropping rule breaches the menu needs one broad and two narrow checks: "
            + json.dumps(rejected)
        )
    return {
        "task": task,
        "checks": contracts,
        "predicates": predicates,
        "rejected_checks": rejected,
        "menu_information_tiers": {c["check_id"]: "public_instruction" for c in contracts},
    }


def menu_freeze(args) -> int:
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    runs_root = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
    instruction = (terminal_root / "tasks" / args.task / "instruction.md").read_text(encoding="utf-8")
    run = runs_root / CENSUS_RUN.format(slug=args.task.replace("-", "_"))
    public_paths = json.loads((run / "config.json").read_text())["agents"][0]["kwargs"]["public_paths"]
    brief = args.brief.read_text(encoding="utf-8")
    raw = args.menu.read_text(encoding="utf-8")
    frozen = freeze_menu(task=args.task, instruction=instruction, public_paths=public_paths, menu=json.loads(raw))
    receipt = {
        "schema_version": "controlled_injection_public_menu_v1",
        "spec_id": SPEC_ID,
        "author": args.author,
        "author_inputs": "instruction.md and environment/ only (menu brief)",
        "command_normalization": "html.unescape of transport-escaped &, <, >",
        "brief_sha256": _sha256(brief),
        "raw_menu_sha256": _sha256(raw),
        **frozen,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task": args.task,
                "checks": [c["check_id"] for c in frozen["checks"]],
                "rejected": frozen["rejected_checks"],
            }
        )
    )
    return 0


PAIR_MOTIFS = ("D1_independent", "D2_masked", "D3_shared_symptom")


def _replay_with_checks(*, job_root, run_id, source, chain, checks) -> dict:
    replay = run_replay(
        job_root=job_root,
        run_id=run_id,
        task=source["task"],
        public_paths=source["public_paths"],
        chain=chain + menu_check_calls(checks),
    )
    history = replay["history"]
    chain_history = [
        {**item, "source_tool_call_id": call["source_tool_call_id"]}
        for item, call in zip(history[: len(chain)], chain, strict=False)
    ]
    exemptions, mismatch = classify_prefix_parity(chain, chain_history)
    verdicts = parse_menu_verdicts(history, checks) if len(history) == len(chain) + len(checks) else {}
    return {
        **replay,
        "chain_history": history[: len(chain)],
        "chain_observer": replay["observer_rows"][: len(chain)],
        "valid": replay["valid"],
        "attempts": replay["attempts"],
        "silent": mismatch is None,
        "parity_mismatch": mismatch,
        "parity_exemptions": len(exemptions),
        "verdicts": verdicts,
    }


def _summary(replay: dict) -> dict:
    return {
        "run_id": replay["run_id"],
        "trial": replay["trial"],
        "reward": replay["reward"],
        "valid": replay["valid"],
        "tries": len(replay["attempts"]),
        "silent": replay["silent"],
        "refuted": sorted(refuted(replay["verdicts"])),
    }


def _package_cell(*, cell_dir: Path, source: dict, sites, replay: dict) -> dict:
    """Write the injected session and observer, then prove the extractor accepts them."""

    chain = apply_sites(source["prefix"], sites)
    entries = rewrite_session(
        _read_jsonl(source["session"]),
        prefix=source["prefix"],
        sites=sites,
        replay_results=replay["chain_history"],
    )
    agent_dir = cell_dir / "agent"
    sessions = agent_dir / "pi" / "sessions"
    sessions.mkdir(parents=True, exist_ok=False)
    cwd = neutral_actor_cwd(agent_dir)
    for entry in entries:
        if entry.get("type") == "session":
            entry["cwd"] = str(cwd)
    session_path = sessions / source["session"].name
    session_path.write_text(dumps_jsonl(entries), encoding="utf-8")
    observer = rewrite_observer(replay["chain_observer"], injected_chain=chain)
    (agent_dir / "terminal-public-state.jsonl").write_text(dumps_jsonl(observer), encoding="utf-8")
    agent = json.loads(
        (Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs/traceback"
         / CENSUS_RUN.format(slug=source["task"].replace("-", "_")) / "config.json").read_text()
    )["agents"][0]
    provider, model = str(agent["model_name"]).split("/", maxsplit=1)
    extracted, _, _ = prefix_from_frozen_source(
        source_session=session_path,
        source_session_cwd=cwd,
        task_instruction=source["instruction"],
        provider=provider,
        model=model,
        thinking=str(agent["kwargs"]["thinking"]),
        expected_terminal_stop_reason="stop",
    )
    if [c["arguments"] for c in extracted] != [c["arguments"] for c in chain]:
        raise ValueError("packaged session does not extract to the injected chain")
    return {"session": str(session_path), "session_cwd": str(cwd), "chain_length": len(chain)}


def build_cells(args) -> int:
    runs_root = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    source = load_census_source(runs_root, terminal_root, args.task)
    menu = json.loads(args.menu.read_text())
    slug = f"{args.task.replace('-', '_')}_{args.tag}"
    job_root = args.job_root
    task_cells = args.cells_dir / args.task
    task_cells.mkdir(parents=True, exist_ok=True)
    log = []

    sanity = _replay_with_checks(
        job_root=job_root, run_id=f"cells_{slug}_sanity", source=source,
        chain=list(source["prefix"]), checks=menu["checks"],
    )
    valid = [c for c in menu["checks"] if sanity["verdicts"].get(c["check_id"], {}).get("verdict") == "supported"]
    dropped = sorted({c["check_id"] for c in menu["checks"]} - {c["check_id"] for c in valid})
    scopes = [c["menu_scope"] for c in valid]
    receipt = {
        "schema_version": "controlled_injection_cells_v1",
        "spec_id": SPEC_ID,
        "task": args.task,
        "menu": str(args.menu),
        "sanity": {**_summary(sanity), "verdicts": sanity["verdicts"], "dropped_checks": dropped},
        "cells": [],
        "attempts": log,
    }
    if not sanity["valid"]:
        receipt["excluded"] = "infrastructure: verifier produced no report on the success state"
    elif sanity["reward"] != 1 or not sanity["silent"] or scopes.count("broad") < 1 or scopes.count("narrow") < 2:
        receipt["excluded"] = "menu sanity failed on the success state"
    if receipt.get("excluded"):
        (task_cells / f"cells-receipt-{args.tag}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"task": args.task, "excluded": receipt["excluded"], "dropped": dropped}))
        return 0
    check_costs = {c["check_id"]: sanity["verdicts"][c["check_id"]]["elapsed_ms"] for c in valid}
    sites = enumerate_sites(
        source["prefix"],
        instruction=source["instruction"],
        observer_rows=_read_jsonl(source["session"].parent.parent.parent / "terminal-public-state.jsonl"),
        session_entries=_read_jsonl(source["session"]),
    )
    if len(sites) > args.max_sites:
        # A seeded subset keeps large tasks bounded; small tasks are covered exhaustively.
        sites = sorted(random.Random(args.seed).sample(sites, args.max_sites), key=lambda x: (x.call_index, x.operator))
    receipt["site_count"] = len(sites)
    table = receipt["site_table"] = []
    singles = {}
    for index, site in enumerate(sites):
        replay = _replay_with_checks(
            job_root=job_root, run_id=f"cells_{slug}_site{index:03d}", source=source,
            chain=apply_sites(source["prefix"], [site]), checks=valid,
        )
        singles[index] = replay
        row = {"site": site.to_dict(), **_summary(replay), "verifier_valid": replay["valid"]}
        # How the public menu and the official verifier relate on this fault (spec §6, Q4).
        row["class"] = (
            "infra_invalid" if not replay["valid"]
            else "not_silent" if not replay["silent"]
            else "official_pass_menu_alarm" if replay["reward"] == 1 and refuted(replay["verdicts"])
            else "official_pass_menu_quiet" if replay["reward"] == 1
            else "official_fail_menu_blind" if not refuted(replay["verdicts"])
            else "official_fail_menu_sees"
        )
        table.append(row)
    qualified = [i for i, row in enumerate(table) if row["class"] == "official_fail_menu_sees"]
    receipt["site_classes"] = dict(collections.Counter(row["class"] for row in table))

    accepted: dict[str, int] = {"D0_single": 0, **dict.fromkeys(PAIR_MOTIFS, 0)}
    quotas = {"D0_single": args.singles, **{motif: 1 for motif in PAIR_MOTIFS}}

    def accept(motif, chosen, replays):
        index = accepted[motif]
        accepted[motif] += 1
        cell_id = f"{slug}__{motif.split('_')[0]}_{index}"
        existing = task_cells / cell_id / "cell.json"
        if existing.is_file():
            cell = json.loads(existing.read_text())
            if cell["gold_sites"] != [site.to_dict() for site in chosen] or cell["motif"] != motif:
                raise ValueError(f"packaged cell {cell_id} holds a different injection")
            receipt["cells"].append(cell)
            return
        packaging = _package_cell(cell_dir=task_cells / cell_id, source=source, sites=chosen, replay=replays[0])
        cell = {
            "cell_id": cell_id,
            "motif": motif,
            "gold_sites": [site.to_dict() for site in chosen],
            "replays": [_summary(r) for r in replays],
            "menu_check_ids": [c["check_id"] for c in valid],
            "check_costs_ms": check_costs,
            "budget": cell_budget(fault_count=len(chosen), check_count=len(valid)),
            **packaging,
        }
        (task_cells / cell_id / "cell.json").write_text(json.dumps(cell, indent=2) + "\n")
        receipt["cells"].append(cell)

    rng = random.Random(args.seed)
    for i in rng.sample(qualified, min(len(qualified), args.singles)):
        accept("D0_single", [sites[i]], [singles[i]])

    pairs = [
        (a, b)
        for a, b in itertools.combinations(qualified, 2)
        if _compatible((sites[a], sites[b]), distinct_calls=True)
    ]
    rng.shuffle(pairs)
    for a, b in pairs[: args.pair_attempts]:
        if all(accepted[m] >= quotas[m] for m in PAIR_MOTIFS):
            break
        chosen = [sites[a], sites[b]]
        both = _replay_with_checks(
            job_root=job_root, run_id=f"cells_{slug}_pair{a:03d}_{b:03d}", source=source,
            chain=apply_sites(source["prefix"], chosen), checks=valid,
        )
        row = {
            "kind": "pair", "sites": [s.to_dict() for s in chosen], "both": _summary(both),
            "alone": [table[a]["refuted"], table[b]["refuted"]],
        }
        log.append(row)
        if not both["valid"]:
            row["result"] = "infra_invalid_verifier_report_missing"
            continue
        if both["reward"] != 0 or not both["silent"] or not refuted(both["verdicts"]):
            row["result"] = "rejected_pair_does_not_fail_visibly"
            continue
        motif = classify_motif(
            refuted(both["verdicts"]), frozenset(table[a]["refuted"]), frozenset(table[b]["refuted"])
        )
        row["motif"] = motif
        if motif in quotas and accepted[motif] < quotas[motif]:
            row["result"] = "accepted"
            accept(motif, chosen, [both, singles[a], singles[b]])
        else:
            row["result"] = "motif_full_or_rejected"
    # Multi-fault cells (k >= 3): the more independent visible faults, the less likely the
    # Actor repairs them all after being told the state failed (self-rescue, contract R6).
    for k in range(3, args.max_faults + 1):
        motif = f"DK{k}_multi"
        quotas[motif] = args.multi
        accepted[motif] = 0
        combos = [
            c for c in itertools.combinations(qualified, k)
            if _compatible(tuple(sites[i] for i in c), distinct_calls=True)
        ]
        rng.shuffle(combos)
        for combo in combos[: args.multi_attempts]:
            if accepted[motif] >= quotas[motif]:
                break
            chosen = [sites[i] for i in combo]
            both = _replay_with_checks(
                job_root=job_root, run_id=f"cells_{slug}_k{k}_" + "_".join(f"{i:03d}" for i in combo),
                source=source, chain=apply_sites(source["prefix"], chosen), checks=valid,
            )
            row = {"kind": f"k{k}", "sites": [x.to_dict() for x in chosen], "both": _summary(both),
                   "alone": [table[i]["refuted"] for i in combo]}
            log.append(row)
            if not both["valid"]:
                row["result"] = "infra_invalid_verifier_report_missing"
            elif both["reward"] != 0 or not both["silent"] or not refuted(both["verdicts"]):
                row["result"] = "rejected_multi_does_not_fail_visibly"
            else:
                row["result"] = "accepted"
                accept(motif, chosen, [both, *[singles[i] for i in combo]])
    receipt["accepted"] = accepted
    (task_cells / f"cells-receipt-{args.tag}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"task": args.task, "accepted": accepted, "sites": len(sites),
                      "classes": receipt["site_classes"], "dropped_checks": dropped}))
    return 0


SNAPSHOT_VERSION = "v2"  # v1 images were deleted by their job's teardown (Compose labels)


def snapshot_image_tag(cell_id: str) -> str:
    return f"longh-snap/{cell_id.lower()}:{SNAPSHOT_VERSION}"


def snapshot_cells(args) -> int:
    """Replay each cell's prefix once, commit it, and keep the snapshot only if it is exact.

    Accepted when the replay matches the packaged session call by call with no exemption
    and the official verifier, with a report, scores the snapshot state as the failure the
    cell is about (reward 0). The verified receipt and a derived task whose image is the
    snapshot are written next to the cell; arms then start from it without replaying.
    """

    runs_root = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    upto = getattr(args, "after_call", None)
    for cell_path in args.cells:
        cell = json.loads(cell_path.read_text())
        task = cell_path.parent.parent.name
        cell_id = cell["cell_id"]
        # A rollback snapshot keeps only the first `upto` calls: the state just before
        # call `upto + 1`, the step a hypothesis blames.
        label = cell_id if upto is None else f"{cell_id}_upto{upto:03d}"
        out = cell_path.parent / ("prefix_snapshot.json" if upto is None else f"prefix_snapshot_upto{upto:03d}.json")
        if out.is_file():
            print(json.dumps({"cell": cell_id, "status": "exists"}))
            continue
        source = load_census_source(runs_root, terminal_root, task)
        agent = json.loads((runs_root / CENSUS_RUN.format(slug=task.replace("-", "_")) / "config.json").read_text())["agents"][0]
        provider, model = str(agent["model_name"]).split("/", maxsplit=1)
        chain, _, _ = prefix_from_frozen_source(
            source_session=Path(cell["session"]),
            source_session_cwd=Path(cell["session_cwd"]),
            task_instruction=source["instruction"],
            provider=provider,
            model=model,
            thinking=str(agent["kwargs"]["thinking"]),
            expected_terminal_stop_reason="stop",
        )
        full_length = len(chain)
        if upto is not None:
            if not 1 <= upto < full_length:
                raise ValueError(f"rollback point {upto} is outside the prefix of {full_length} calls")
            chain = chain[:upto]
        tag = snapshot_image_tag(label)
        # A step whose outcome depends on the outside world can diverge on one replay and
        # match on the next; only an exact replay is ever committed as the snapshot.
        for attempt in range(1, args.parity_attempts + 1):
            replay = run_replay(
                job_root=args.job_root,
                run_id=f"snapshot_{label}_{SNAPSHOT_VERSION}" + (f"_a{attempt}" if attempt > 1 else ""),
                task=task, public_paths=source["public_paths"], chain=list(chain),
                snapshot={"image_tag": tag, "after_call": len(chain) - 1},
            )
            history = [
                {**item, "source_tool_call_id": call["source_tool_call_id"]}
                for item, call in zip(replay["history"], chain, strict=False)
            ]
            exemptions, mismatch = classify_prefix_parity(list(chain), history)
            if replay["valid"] and mismatch is None and not exemptions:
                break
        capture_path = Path(replay["trial"]) / "agent" / "terminal-prefix-snapshot-capture.json"
        row = {"cell": cell_id, "run_id": replay["run_id"], "reward": replay["reward"],
               "valid": replay["valid"], "parity_mismatch": mismatch, "exemptions": len(exemptions)}
        # The full prefix must reproduce the failure; a rolled-back state need not fail.
        reward_ok = replay["reward"] == 0 if upto is None else replay["reward"] is not None
        if not replay["valid"] or not reward_ok or mismatch is not None or exemptions or not capture_path.is_file():
            row["status"] = "rejected"
            print(json.dumps(row), flush=True)
            continue
        capture = json.loads(capture_path.read_text())
        surviving = subprocess.run(
            docker_command("image", "inspect", "--format", "{{.Id}}", tag),
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        if surviving != capture["image_id"]:
            # The job's own teardown must not be able to remove the snapshot.
            print(json.dumps({**row, "status": "rejected", "reason": "snapshot image did not survive its job"}))
            continue
        derived = args.tasks_root / label / task
        if not derived.exists():
            shutil.copytree(terminal_root / "tasks" / task, derived)
            toml = derived / "task.toml"
            text = toml.read_text(encoding="utf-8")
            lines = text.splitlines()
            hits = [i for i, line in enumerate(lines) if line.strip().startswith("docker_image")]
            if len(hits) != 1:
                raise ValueError(f"{task}: expected exactly one docker_image line")
            lines[hits[0]] = f'docker_image = "{tag}"'
            toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        receipt = {
            "schema_version": SNAPSHOT_RECEIPT_SCHEMA,
            "cell_id": cell_id,
            "task": task,
            "image_tag": tag,
            "image_id": capture["image_id"],
            "public_paths": capture["public_paths"],
            "public_state": capture["public_state"],
            "history": capture["history"],
            "parity_policy": "exact_v1",
            "snapshot_replay_run": replay["run_id"],
            "snapshot_replay_trial": replay["trial"],
            "official_reward_of_snapshot_state": replay["reward"],
            "derived_task_path": str(derived),
            "source_task_path": str(terminal_root / "tasks" / task),
            "limits": "filesystem only: running processes and memory are not part of the snapshot",
            **({"rollback_after_call": upto, "full_prefix_length": full_length} if upto is not None else {}),
        }
        out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({**row, "status": "accepted", "image_id": capture["image_id"]}), flush=True)
    return 0


def build_natural_cells(args) -> int:
    """Package a naturally failed source as a cell: no injection, the failure is the cell.

    The menu cannot be sanity-checked against a success state here (there is none), so a
    check stays when it reaches a decisive verdict on the failed state. Whether the menu
    refutes anything is recorded as the cell's visibility class, not used to select it.
    """

    runs_root = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    source = load_census_source(runs_root, terminal_root, args.task)
    menu = json.loads(args.menu.read_text())
    slug = f"{args.task.replace('-', '_')}_{args.tag}"
    task_cells = args.cells_dir / args.task
    task_cells.mkdir(parents=True, exist_ok=True)
    replay = _replay_with_checks(
        job_root=args.job_root, run_id=f"cells_{slug}_natural", source=source,
        chain=list(source["prefix"]), checks=menu["checks"],
    )
    receipt = {
        "schema_version": "controlled_injection_natural_cells_v1",
        "spec_id": SPEC_ID,
        "task": args.task,
        "menu": str(args.menu),
        "replay": {**_summary(replay), "verdicts": replay["verdicts"]},
    }
    decisive = [
        c for c in menu["checks"]
        if replay["verdicts"].get(c["check_id"], {}).get("verdict") in {"supported", "refuted"}
    ]
    if not replay["valid"]:
        receipt["excluded"] = "infrastructure: verifier produced no report"
    elif replay["reward"] != 0 or not replay["silent"]:
        receipt["excluded"] = "the replay did not reproduce the natural official failure"
    elif len(decisive) < 3:
        receipt["excluded"] = "fewer than three checks reached a decisive verdict"
    if receipt.get("excluded"):
        (task_cells / f"cells-receipt-{args.tag}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"task": args.task, "excluded": receipt["excluded"]}))
        return 0
    cell_id = f"{slug}__N0_0"
    existing = task_cells / cell_id / "cell.json"
    if existing.is_file():
        cell = json.loads(existing.read_text())
    else:
        packaging = _package_cell(cell_dir=task_cells / cell_id, source=source, sites=[], replay=replay)
        cell = {
            "cell_id": cell_id,
            "motif": "N0_natural",
            "gold_sites": [],
            "replays": [_summary(replay)],
            "menu_check_ids": [c["check_id"] for c in decisive],
            "check_costs_ms": {c["check_id"]: replay["verdicts"][c["check_id"]]["elapsed_ms"] for c in decisive},
            "budget": cell_budget(fault_count=1, check_count=len(decisive)),
            **packaging,
        }
        existing.write_text(json.dumps(cell, indent=2) + "\n")
    seen = sorted(refuted(replay["verdicts"]))
    receipt["cell"] = cell
    receipt["visibility"] = "menu_sees" if seen else "menu_blind"
    (task_cells / f"cells-receipt-{args.tag}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"task": args.task, "cell": cell_id, "visibility": receipt["visibility"], "refuted": seen,
                      "checks": len(decisive)}))
    return 0


_CONTRACT_KEYS = {
    "task_id", "proposition", "arguments", "supported_output", "refuted_output", "resource_ids",
    "check_id", "predicate_id", "verdict_protocol", "presence_policy", "prerequisite_resource_ids",
    "predicate_kind", "read_only",
}


def public_repairs(checks: list[dict], predicates: list[dict]) -> list[dict]:
    """Spec §7: one guidance repair per check, built only from its proposition and excerpt."""

    excerpts = {p["predicate_id"]: p["source_excerpt"] for p in predicates}
    repairs = []
    for check in checks:
        subjects = ", ".join(check["resource_ids"])
        repairs.append(
            {
                "repair_id": "repair-" + check["check_id"].removeprefix("pub-"),
                "task_id": "public-task",
                "proposition": check["proposition"],
                "arguments": {
                    "candidate_form": "actor_guidance",
                    "guidance_text": (
                        f"A public acceptance check does not hold: {check['proposition']} "
                        f'The task instruction requires: "{excerpts[check["predicate_id"]]}". '
                        f"Inspect the current state of {subjects}, find why this requirement is "
                        "not met, and fix it without breaking the other requirements."
                    ),
                    "fixer_identity": "public_menu_mechanical_repair_v1",
                    "allowed_tools": "bash,read,write",
                },
                "resource_ids": list(check["resource_ids"]),
                "time_limit_seconds": 300,
            }
        )
    return repairs


def _uniform_rule(rule_id: str, action_type: str, contract_id: str, predicate: str, duration_ms: int) -> dict:
    third = 1.0 / 3.0
    return {
        "rule_id": rule_id,
        "action_type": action_type,
        "contract_id": contract_id,
        "target_predicate": predicate,
        "duration_ms": duration_ms,
        "outcomes": [{"outcome_id": o, "probability": third} for o in ("supported", "refuted", "unknown")],
        "provenance_id": "uniform-prior:controlled-injection-v1",
        "provenance_detail": "no task-disjoint calibration data; uniform over supported/refuted/unknown",
    }


GUIDED_METHOD = "tracedp_guidance_phase_option_dp_dev_v1"
OBSERVED_EXISTENCE_ESTIMATES = "observed_existence_v1"


def guided_completion_configuration(
    *, cell: dict, checks: list[dict], repairs: list[dict], protocol: str, actor_cell_id: str,
    resolved_probability: float, provenance: str,
) -> dict:
    """Two joint outcomes per repair: every check supported, or only its target refuted."""

    recheck_ms = sum(max(1000, int(cell["check_costs_ms"].get(c["check_id"]) or 1000)) for c in checks)
    rules = []
    for repair in repairs:
        target = "pub-" + repair["repair_id"].removeprefix("repair-")
        rules.append(
            {
                "rule_id": f"guided-{repair['repair_id']}",
                "repair_contract_id": f"{repair['repair_id']}|{protocol}",
                "task_id": repair["task_id"],
                "target_predicate": repair["proposition"],
                "actor_duration_ms": 120_000,
                "recheck_duration_ms": recheck_ms,
                "outcomes": [
                    {
                        "outcome_id": "all_public_checks_supported",
                        "probability": resolved_probability,
                        "check_verdicts": [{"check_id": c["check_id"], "verdict": "supported"} for c in checks],
                    },
                    {
                        "outcome_id": "target_still_refuted",
                        "probability": 1.0 - resolved_probability,
                        "check_verdicts": [
                            {"check_id": c["check_id"], "verdict": "refuted" if c["check_id"] == target else "supported"}
                            for c in checks
                        ],
                    },
                ],
                "provenance_id": provenance,
                "provenance_detail": "controlled injection guided-completion prior",
            }
        )
    return {
        "schema_version": "runtime_guided_completion_configuration_v1",
        "configuration_id": f"{cell['cell_id']}-guided-v1",
        "actor_cell_id": actor_cell_id,
        "prediction_scope": "main_actor_and_registered_checks_given_admitted_guidance",
        "implementation_status": "implementation_unverified",
        "estimate_status": "development_prior_unverified",
        "rules": rules,
    }


NEUTRAL_CONDITION = "neutral_v1"
NEUTRAL_TRIGGER = "Continue the original task."


def cell_matrix(
    *, template: dict, cell: dict, menu: dict, task: str, terminal_root: Path, public_paths: list[str],
    continuation: str = "informed",
    guided_resolved_probability: float = 0.5,
    guided_provenance: str = "uniform-prior:no-calibration-cells-yet",
    prefix_snapshot: dict | None = None,
    estimates: str = "uniform",
    start_public_state: list | None = None,
) -> dict:
    from lh_platform.benchmarks.terminal_runtime_check import runtime_check_configuration_payload
    from lh_platform.test_time_adaptation.runtime_owned_option_live import (
        RUNTIME_ATOMIC_ESTIMATE_CALIBRATION_STATUS,
    )

    valid = set(cell["menu_check_ids"])
    checks = [{k: v for k, v in c.items() if k in _CONTRACT_KEYS} for c in menu["checks"] if c["check_id"] in valid]
    predicate_ids = {c["predicate_id"] for c in checks}
    predicates = [p for p in menu["predicates"] if p["predicate_id"] in predicate_ids]
    repairs = public_repairs(checks, predicates)
    protocol = template["candidate_trial_protocol"]
    instruction = (terminal_root / "tasks" / task / "instruction.md").read_text(encoding="utf-8")
    costs = cell["check_costs_ms"]
    rules = [r for r in template["atomic_estimate_configuration"]["rules"]
             if r["action_type"] in ("continue_actor", "submit_current_state")]
    observed = {row["path"]: row for row in (start_public_state or [])}
    if estimates == OBSERVED_EXISTENCE_ESTIMATES and not observed:
        raise ValueError("observed-existence estimates need the starting public state")
    for c in checks:
        rule = _uniform_rule(f"uniform-{c['check_id']}", "run_registered_check", c["check_id"], c["proposition"],
                             max(1000, int(costs.get(c["check_id"]) or 1000)))
        if (
            estimates == OBSERVED_EXISTENCE_ESTIMATES
            and c.get("predicate_kind") == "existence"
            and c["resource_ids"]
            and all(observed.get(r, {}).get("exists") is True for r in c["resource_ids"])
        ):
            # An existence check on paths the starting state already shows as present
            # cannot refute unless the Actor later deletes them; it carries no information.
            rule.update(
                rule_id=f"observed-exists-{c['check_id']}",
                outcomes=[
                    {"outcome_id": "supported", "probability": 0.98},
                    {"outcome_id": "refuted", "probability": 0.01},
                    {"outcome_id": "unknown", "probability": 0.01},
                ],
                provenance_id="observed-existence-v1:start-public-state",
                provenance_detail="all subject paths observed present in the verified starting snapshot",
            )
        elif estimates not in ("uniform", OBSERVED_EXISTENCE_ESTIMATES):
            raise ValueError(f"unknown estimates mode: {estimates}")
        rules.append(rule)
    rules += [
        _uniform_rule(f"uniform-{r['repair_id']}", "run_targeted_repair", f"{r['repair_id']}|{protocol}",
                      r["proposition"], 300_000)
        for r in repairs
    ]
    matrix = copy.deepcopy(template)
    budget = cell["budget"]["budget_units"]
    matrix.update(
        {
            "matrix_version": "v5_controlled_injection",
            "menu_tier": "public_only",
            "menu_information_tiers": {c["check_id"]: "public_instruction" for c in checks},
            "menu_tier_evidence": f"blind public menu {menu.get('brief_sha256')} by {menu.get('author')}",
            "task_instance": {
                "task_name": task,
                "task_path": str(terminal_root / "tasks" / task),
                "public_instruction_source": f"terminal-bench-2.1/tasks/{task}/instruction.md",
                "failure_state_identity": "controlled-injection-packaged-session",
                "official_evaluator": f"terminal-bench {task} evaluator",
            },
            "public_paths": list(public_paths),
            "resource_specs": [
                {"resource_id": p, "kind": "file", "provenance": "public_instruction", "group_id": None}
                for p in public_paths
            ],
            "public_acceptance_checks": checks,
            "public_acceptance_predicates": predicates,
            "fixer_candidates": repairs,
            "admitted_repair_candidates": [],
            "repair_admission_policy": "runtime_candidate_trial_gate_v1",
            "admission_evidence": {
                "policy": (
                    "spec §7: mechanical public repairs are grounded by construction; the "
                    "isolated candidate trial at selection time delivers guidance only if it "
                    "resolves, so no per-cell A0 admission is run"
                )
            },
            "controlled_injection": {
                "spec_id": SPEC_ID,
                "cell_id": cell["cell_id"],
                "motif": cell["motif"],
                "gold_sites": cell["gold_sites"],
                "source_session": cell["session"],
                "source_session_cwd": cell["session_cwd"],
                "budget": cell["budget"],
            },
        }
    )
    review = matrix["public_semantic_review"]
    review.update(
        {
            "reviewer": f"controlled_injection_blind_menu:{menu.get('author')}",
            # The host compares this byte-for-byte with the instruction file it is handed.
            "public_instruction": instruction,
            "predicate_ids": [p["predicate_id"] for p in predicates],
            "reviewed_manifest": copy.deepcopy(predicates),
            "reviewed_check_contracts": runtime_check_configuration_payload(
                tuple(RuntimeCheckContract(**{**c, "resource_ids": tuple(c["resource_ids"])}) for c in checks),
                diagnosis_enabled=False,
            )["contracts"],
            "hidden_evaluator_or_reference_answer_used": False,
        }
    )
    configuration = matrix["atomic_estimate_configuration"]
    for key in ("calibration_id", "source_task_ids", "excluded_task_id", "smoothing"):
        configuration.pop(key, None)
    configuration.update(
        {
            "calibration_status": RUNTIME_ATOMIC_ESTIMATE_CALIBRATION_STATUS,
            "configuration_id": f"{cell['cell_id']}-{'uniform-v1' if estimates == 'uniform' else estimates}",
            "rules": rules,
        }
    )
    repair_ids = [r["repair_id"] for r in repairs]
    for arm in matrix["arms"]:
        if arm["runtime_intervention_budget"]:
            arm["runtime_intervention_budget"] = budget
        if arm["arm_id"] != "check-feedback-only" and arm.get("runtime_method") != "managed_observation_only_v1":
            arm["repair_ids"] = list(repair_ids)
            arm["runtime_acceptance_candidate_trials"] = True
    actor = matrix["shared_actor"]
    matrix["guided_completion_configuration"] = guided_completion_configuration(
        cell=cell, checks=checks, repairs=repairs, protocol=protocol,
        actor_cell_id=f"{actor['provider']}/{actor['model']}|thinking={actor['thinking']}|transport=host_pi_unix_rpc",
        resolved_probability=guided_resolved_probability, provenance=guided_provenance,
    )
    tracedp = next(arm for arm in matrix["arms"] if arm["arm_id"] == "tracedp")
    guided = {k: copy.deepcopy(v) for k, v in tracedp.items() if k != "runtime_planning_horizon_cap"}
    guided.update(
        arm_id="tracedp-guided",
        runtime_method=GUIDED_METHOD,
        interpretation=(
            "TraceDP with the guidance phase priced explicitly (delivery, Actor, full public "
            "recheck): horizon 2 at replans and follow-ups, 3 otherwise."
        ),
    )
    matrix["arms"].append(guided)
    # TraceDP v2: the information-gain arm's exact configuration, plus recovery lookahead.
    info_gain = next(arm for arm in matrix["arms"] if arm["arm_id"] == "info-gain-reserve")
    v2 = copy.deepcopy(info_gain)
    v2.update(
        arm_id="tracedp-v2",
        runtime_method="tracedp_recovery_lookahead_dev_v2",
        interpretation=(
            "TraceDP v2: the adaptive information-gain base with repair reserve, plus a "
            "bounded recovery-value lookahead that overrides the base only when strictly better."
        ),
    )
    matrix["arms"].append(v2)
    # TraceDP v2 with a step hypothesis from the TraceElephant all-at-once locator.
    v2h = copy.deepcopy(v2)
    v2h.update(
        arm_id="tracedp-v2h",
        hypothesis_locator="traceelephant-all-at-once-bridge",
        interpretation=(
            "TraceDP v2 whose check priors follow TraceElephant all-at-once's step hypothesis "
            "(suspect public paths), the locator charged to this arm's budget."
        ),
    )
    matrix["arms"].append(v2h)
    # Step 5: also roll the environment back to just before the hypothesized step.
    v2hr = copy.deepcopy(v2h)
    v2hr.update(
        arm_id="tracedp-v2hr",
        rollback_on_hypothesis=True,
        interpretation=(
            "TraceDP v2h that starts from a verified snapshot of the state just before the "
            "hypothesized step; the Actor keeps its full history and the same neutral trigger."
        ),
    )
    matrix["arms"].append(v2hr)
    # DoVer's intervention at the hypothesized step, delivered like a bridge lead.
    v2hri = copy.deepcopy(v2hr)
    v2hri.update(
        arm_id="tracedp-v2hri",
        dover_intervention=True,
        interpretation=(
            "TraceDP v2hr whose Actor receives DoVer's intervention for the hypothesized step "
            "(author prompt and parser, one call charged to this arm), wrapped exactly as a "
            "bridge lead; without a valid intervention it is tracedp-v2hr."
        ),
    )
    matrix["arms"].append(v2hri)
    # The supervisor DP decides which components to use, from zero-call evidence.
    v3 = copy.deepcopy(v2)
    v3.update(
        arm_id="tracedp-v3",
        supervisor_plan=True,
        interpretation=(
            "TraceDP v3: a supervisor DP over meta-actions (input inventory, clean-context probe, "
            "bisected rollback on verified replays, DoVer intervention) chooses the hand-off, "
            "then the tracedp-v2 runtime continues."
        ),
    )
    matrix["arms"].append(v3)
    # The supervisor re-enters at the Actor's natural settlements.
    v3c = copy.deepcopy(v3)
    v3c.update(
        arm_id="tracedp-v3c",
        supervisor_closed_loop=True,
        interpretation=(
            "TraceDP v3 whose supervisor re-enters at each natural settlement (at most twice): "
            "re-probes the live state, restores lost inputs or redelivers clean-context evidence, "
            "otherwise defers to the v2 runtime."
        ),
    )
    matrix["arms"].append(v3c)
    # v3's planning stage, with the obligation-value runtime instead of v2's.
    v3o = copy.deepcopy(v3)
    v3o.update(
        arm_id="tracedp-v3o",
        runtime_method="tracedp_obligation_value_dev_v1",
        interpretation=(
            "TraceDP v3 whose runtime values the residual public obligations (supported, "
            "refuted once, refuted again after a remedy, unknown) with frozen obligation-level "
            "recovery rates, and compares checks, repairs and the hand-back at every boundary."
        ),
    )
    matrix["arms"].append(v3o)
    # v3o with the Runtime's observation overlapped with the public
    # snapshot at each tool boundary; every decision and observed fact is unchanged.
    v3of = copy.deepcopy(v3o)
    v3of.update(
        arm_id="tracedp-v3of",
        concurrent_runtime_observation=True,
        interpretation=(
            "TraceDP v3o whose Runtime resource observation runs concurrently with the public "
            "snapshot at each tool boundary (implementation efficiency only)."
        ),
    )
    matrix["arms"].append(v3of)
    # The composition carried to the hard cells: v3's planning,
    # v3c's closed-loop re-entry and v3o's obligation-value runtime; no new mechanism.
    v4 = copy.deepcopy(v3o)
    v4.update(
        arm_id="tracedp-v4",
        supervisor_closed_loop=True,
        interpretation=(
            "TraceDP v4: v3 supervisor planning, v3c re-entry at natural settlements, and the "
            "v3o obligation-value runtime (the composition to be evaluated on the hard cells)."
        ),
    )
    matrix["arms"].append(v4)
    matrix["continuation_condition"] = continuation
    if continuation == NEUTRAL_CONDITION:
        # One shared trigger for every arm; none of them is told the state failed.
        for arm in matrix["arms"]:
            if arm["arm_id"] != "binary-bridge":
                arm["continuation_instruction"] = NEUTRAL_TRIGGER
        matrix["continuation_opening"] = NEUTRAL_TRIGGER
    elif continuation != "informed":
        raise ValueError(f"unknown continuation condition: {continuation}")
    if prefix_snapshot is not None:
        # Every arm starts from the same verified container image; nothing is replayed.
        matrix["prefix_snapshot"] = prefix_snapshot
    matrix["check_estimates"] = estimates
    matrix.pop("semantic_review_correction", None)
    matrix["launch_ready"] = False
    return matrix


def _prefix_snapshot_binding(args) -> dict | None:
    if getattr(args, "prefix_state", "replay") != "snapshot":
        return None
    receipt_path = args.cell.parent / "prefix_snapshot.json"
    receipt = json.loads(receipt_path.read_text())
    return {
        "receipt": str(receipt_path),
        "derived_task_path": receipt["derived_task_path"],
        "image_tag": receipt["image_tag"],
        "image_id": receipt["image_id"],
    }


def _start_public_state(args) -> list | None:
    receipt = args.cell.parent / "prefix_snapshot.json"
    if getattr(args, "estimates", "uniform") == "uniform" or not receipt.is_file():
        return None
    return json.loads(receipt.read_text())["public_state"]


def cell_matrix_command(args) -> int:
    cell = json.loads(args.cell.read_text())
    menu = json.loads(args.menu.read_text())
    template = json.loads(args.template.read_text())
    runs_root = Path(os.environ["LONGH_STORAGE_ROOT"]) / "runs" / "traceback"
    run = runs_root / CENSUS_RUN.format(slug=menu["task"].replace("-", "_"))
    public_paths = json.loads((run / "config.json").read_text())["agents"][0]["kwargs"]["public_paths"]
    matrix = cell_matrix(
        template=template, cell=cell, menu=menu, task=menu["task"],
        terminal_root=Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"]), public_paths=public_paths,
        continuation=getattr(args, "continuation", "informed"),
        prefix_snapshot=_prefix_snapshot_binding(args),
        estimates=getattr(args, "estimates", "uniform"),
        start_public_state=_start_public_state(args),
    )
    observer = {}
    if getattr(args, "batched_snapshot", False):
        # One harness setting for every arm of the matrix: the public snapshot
        # reads all paths in one exec; the facts read are byte-identical.
        observer["batched_public_snapshot"] = True
    if getattr(args, "budget_gate", False):
        # Repairs that would leave the Actor's next call unfunded are not chosen.
        observer["budget_affordability_gate"] = True
    if getattr(args, "verify_only_fallback", False):
        # Keep each run's post-Actor filesystem so a verifier lost to the network is rerun
        # on that exact state (verify_only.py) instead of repeating the Actor.
        observer["verify_only_fallback"] = True
    if observer:
        matrix["observer"] = observer
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n")
    from build_recovery_matrix_v3 import matrix_launch_blockers

    print(json.dumps({"out": str(args.out), "launch_blockers": matrix_launch_blockers(matrix)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    admit = sub.add_parser("admit-sources")
    admit.add_argument("--tasks", nargs="+", default=list(DEFAULT_SUCCESS_TASKS))
    admit.add_argument("--job-root", type=Path, required=True)
    admit.add_argument("--out", type=Path, required=True)
    admit.add_argument("--tag", default="", help="batch tag for re-admission; keeps job names unique")
    admit.add_argument(
        "--natural", action="store_true",
        help="admit naturally failed sources (the replay must reproduce the official failure)",
    )
    admit.add_argument("--allow-tool-use-ending", action="store_true",
                       help="S3: admit sources whose last turn ended in toolUse")
    admit.add_argument("--agent-timeout-multiplier", type=float,
                       help="let a zero-model replay of a long source outlast the task's agent timeout")
    admit.add_argument("--ladder-every", type=int, help="capture a checkpoint rung every N calls on replay 1")
    admit.add_argument("--ladder-base-cache", type=Path, help="where base images are saved once per image id")
    admit.add_argument("--trace-writes", action="store_true", help="record each call's exact write set on replay 1")
    admit.add_argument("--funnel", type=Path, help="append every gate outcome to this funnel ledger")
    admit.add_argument("--batched-public-snapshot", action="store_true",
                       help="read all public paths in one exec per snapshot (self-checked, falls back)")
    admit.add_argument("--source-era-bash", action="store_true",
                       help="replay Bash with the gateway semantics that older recorded sources were captured under")
    brief = sub.add_parser("menu-brief")
    brief.add_argument("--task", required=True)
    brief.add_argument("--out", type=Path, required=True)
    freeze = sub.add_parser("menu-freeze")
    freeze.add_argument("--task", required=True)
    freeze.add_argument("--brief", type=Path, required=True)
    freeze.add_argument("--menu", type=Path, required=True)
    freeze.add_argument("--author", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    snap = sub.add_parser("snapshot-cells")
    snap.add_argument("cells", nargs="+", type=Path)
    snap.add_argument("--job-root", type=Path, required=True)
    snap.add_argument("--tasks-root", type=Path, required=True)
    snap.add_argument("--parity-attempts", type=int, default=4)
    snap.add_argument("--after-call", type=int, help="rollback snapshot: keep only the first N calls")
    natural = sub.add_parser("build-natural-cells")
    natural.add_argument("--task", required=True)
    natural.add_argument("--menu", type=Path, required=True)
    natural.add_argument("--cells-dir", type=Path, required=True)
    natural.add_argument("--job-root", type=Path, required=True)
    natural.add_argument("--tag", required=True)
    cells = sub.add_parser("build-cells")
    cells.add_argument("--task", required=True)
    cells.add_argument("--menu", type=Path, required=True)
    cells.add_argument("--job-root", type=Path, required=True)
    cells.add_argument("--cells-dir", type=Path, required=True)
    cells.add_argument("--seed", type=int, default=20261014)
    cells.add_argument("--tag", required=True, help="batch tag; keeps replay job names unique")
    cells.add_argument("--singles", type=int, default=2)
    cells.add_argument("--pair-attempts", type=int, default=8)
    cells.add_argument("--max-sites", type=int, default=20)
    cells.add_argument("--max-faults", type=int, default=2, help="largest fault count to build (>=3 adds DKk_multi cells)")
    cells.add_argument("--multi", type=int, default=2, help="multi-fault cells wanted per fault count")
    cells.add_argument("--multi-attempts", type=int, default=6)
    matrix = sub.add_parser("cell-matrix")
    matrix.add_argument("--cell", type=Path, required=True)
    matrix.add_argument("--menu", type=Path, required=True)
    matrix.add_argument("--template", type=Path, required=True)
    matrix.add_argument("--out", type=Path, required=True)
    matrix.add_argument("--prefix-state", choices=("replay", "snapshot"), default="replay")
    matrix.add_argument("--estimates", choices=("uniform", OBSERVED_EXISTENCE_ESTIMATES), default="uniform")
    args = parser.parse_args()
    if args.stage == "cell-matrix":
        return cell_matrix_command(args)
    if args.stage == "build-cells":
        return build_cells(args)
    if args.stage == "build-natural-cells":
        return build_natural_cells(args)
    if args.stage == "snapshot-cells":
        return snapshot_cells(args)
    if args.stage == "admit-sources":
        return admit_sources(args)
    if args.stage == "menu-brief":
        return menu_brief(args)
    if args.stage == "menu-freeze":
        return menu_freeze(args)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
