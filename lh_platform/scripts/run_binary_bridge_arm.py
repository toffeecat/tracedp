#!/usr/bin/env python3
"""Who&When binary search + shared guidance bridge + same actor, under one budget pool.

The binary-bridge arm of a contract-v0.2 recovery matrix (R1/R2):

``locate``  (paid, tracer) build the public trace of the frozen source failure exactly as
            the census population does (canonical public projection), run the author's
            binary search through MiniMax with thinking disabled, and charge every
            tracer call to the arm's shared ledger;
``prepare`` (no calls) compile the attributed event with the shared method-agnostic
            bridge and freeze a historical-session continuation that replays the same
            prefix as every other arm, with the same actor, timeout, output cap and
            ledger;
``run``     (paid, actor) execute the frozen Harbor config.

Each stage writes its receipt next to ``--request`` and refuses to overwrite one.
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

from build_recovery_matrix_v3 import matrix_launch_blockers  # noqa: E402
from run_historical_managed_arm import task_agent_timeout_seconds  # noqa: E402
from terminal_historical_managed_launch import (  # noqa: E402
    frozen_source_cwd,
    prefix_from_frozen_source,
    read_json,
)

from lh_platform.benchmarks.terminal_attribution_guidance import (  # noqa: E402
    BRIDGE_TREATMENT_ID,
    compile_public_step_guidance,
)
from lh_platform.benchmarks.terminal_observed_runtime import (  # noqa: E402
    ObservedTerminalRunSpec,
    build_terminal_historical_session_rescue_harbor_config,
)
from lh_platform.benchmarks.terminal_runtime_identity import (  # noqa: E402
    resolve_host_pi_runtime,
)
from lh_platform.test_time_adaptation.terminal_log_attribution import (  # noqa: E402
    build_terminal_log_trace_case,
)
from lh_platform.tracefix_bench.scorer_contracts import ResourceRole  # noqa: E402
from lh_platform.tracefix_bench.shared_budget import (  # noqa: E402
    FileSharedBudgetLedger,
    SharedRoleBudgetV1,
)
from lh_platform.tracing.integrations._runtime import NativeTraceRuntimeV1  # noqa: E402
from lh_platform.tracing.integrations.traceelephant import native_traceelephant_adapters  # noqa: E402
from lh_platform.tracing.integrations.whowhen import native_whowhen_adapters  # noqa: E402
from lh_platform.tracing.unified import TraceMethodRequestV1  # noqa: E402

ARM_ID = "binary-bridge"
# Author-code locators of the baseline papers, each a separate arm with the same bridge,
# actor, continuation and shared budget.
LOCATOR_ARMS = {
    "binary-bridge": "who_when.binary_search",
    "whowhen-all-at-once-bridge": "who_when.all_at_once",
    "whowhen-step-by-step-bridge": "who_when.step_by_step",
    "traceelephant-all-at-once-bridge": "traceelephant.all_at_once",
    "traceelephant-step-by-step-bridge": "traceelephant.step_by_step",
    "traceelephant-binary-search-bridge": "traceelephant.binary_search",
}
MINIMAX_OPENAI_BASE_URL = "https://api.minimaxi.com/v1"
FAILURE_SUMMARY = "The official Terminal-Bench verifier returned reward 0."



def apply_prefix_snapshot(config: dict, matrix: dict) -> None:
    """Same snapshot start as the managed arms (run_historical_managed_arm)."""

    snapshot = matrix.get("prefix_snapshot")
    if not snapshot:
        return
    config["tasks"][0]["path"] = str(snapshot["derived_task_path"])
    config["agents"][0]["kwargs"]["prefix_snapshot_receipt"] = str(snapshot["receipt"])


# A locator that cannot finish inside the shared budget, or returns an output its own
# contract rejects (BASELINE_ADAPTER_DESIGN_V0_1 §3: out of range, missing fields), has
# failed as a method. It scores 0 and is not retried: retrying would buy it extra draws.
METHOD_FAILURE_EXIT = 3


def method_failure_reason(error: BaseException) -> str | None:
    text = f"{type(error).__name__}: {error}"
    if "BudgetRefusedError" in text or "budget" in text.lower() and "cannot cover" in text:
        return "budget_exhausted"
    if "out of range" in text or "method_output_invalid" in text or "lacks a valid" in text:
        return "method_output_invalid"
    return None


def _write_new(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _read_jsonl(path: Path) -> tuple[dict, ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _public_case(matrix: dict, source_session: Path, terminal_root: Path):
    """The canonical public trace, built exactly as the census population builds it."""

    task = str(matrix["task_instance"]["task_name"])
    instruction = (terminal_root / "tasks" / task / "instruction.md").read_text(encoding="utf-8")
    trial = source_session.parent.parent.parent.parent
    observer = trial / "agent" / "terminal-public-state.jsonl"
    compilation = build_terminal_log_trace_case(
        case_id=task,
        task_id=task,
        task_specification=instruction,
        failure_summary=FAILURE_SUMMARY,
        terminal_reason="source_failure",
        official_reward=0.0,
        session_records=_read_jsonl(source_session),
        observer_records=_read_jsonl(observer) if observer.is_file() else (),
    )
    return compilation.canonical_public_projection.bind_trace_case(compilation.trace_case)


def _ledger(path: Path, matrix: dict, arm_id: str = ARM_ID) -> FileSharedBudgetLedger:
    if path.exists():
        return FileSharedBudgetLedger(path)
    return FileSharedBudgetLedger(
        path,
        SharedRoleBudgetV1(
            budget_id=f"{matrix['task_instance']['task_name']}:{arm_id}",
            roles=(ResourceRole.ACTOR, ResourceRole.LOCATOR, ResourceRole.ORCHESTRATION),
            **matrix["shared_budget"],
        ),
    )


def locate(args, matrix: dict, terminal_root: Path, corpus_root: Path) -> None:
    receipt_path = args.request.with_suffix(".locate.json")
    case = _public_case(matrix, args.source_session, terminal_root)
    ledger_path = args.request.with_suffix(".ledger.json")
    _ledger(ledger_path, matrix, args.budget_arm_id or args.arm_id)
    runtime = NativeTraceRuntimeV1(
        model_id=str(matrix["shared_actor"]["model"]),
        base_url=MINIMAX_OPENAI_BASE_URL,
        api_key=os.environ["MINIMAX_CN_API_KEY"],
        max_tokens=int(matrix["shared_budget"]["per_call_output_tokens"]),
        timeout_seconds=600,
        disable_thinking=True,
        budget_ledger_path=str(ledger_path),
        budget_role=ResourceRole.LOCATOR.value,
    )
    method_id = LOCATOR_ARMS[args.arm_id]
    family = (
        native_traceelephant_adapters if method_id.startswith("traceelephant.") else native_whowhen_adapters
    )
    adapter = next(
        item
        for item in family(corpus_root=corpus_root, runtime=runtime)
        if item.descriptor.method_id == method_id
    )
    try:
        output = adapter.execute(TraceMethodRequestV1(case))
    except Exception as error:  # classified below; provider and network errors re-raise
        reason = method_failure_reason(error)
        if reason is None:
            raise
        _write_new(
            receipt_path,
            {
                "schema_version": "binary_bridge_locate_v1",
                "arm_id": args.arm_id,
                "locator_method_id": LOCATOR_ARMS[args.arm_id],
                "status": "method_failure",
                "method_failure": reason,
                "error": f"{type(error).__name__}: {str(error)[-800:]}",
                "ledger_after_locate": FileSharedBudgetLedger(ledger_path).receipt(),
            },
        )
        raise SystemExit(METHOD_FAILURE_EXIT)
    finding = output.findings[0]
    if len(finding.event_ids) != 1:
        # The locator abstained (no single step): a method result, scored 0, not retried;
        # TraceDP's hypothesis arms fall back.
        _write_new(
            receipt_path,
            {
                "schema_version": "binary_bridge_locate_v1",
                "arm_id": args.arm_id,
                "locator_method_id": LOCATOR_ARMS[args.arm_id],
                "status": "method_failure",
                "method_failure": "abstained",
                "finding": finding.to_dict(),
                "ledger_after_locate": FileSharedBudgetLedger(ledger_path).receipt(),
            },
        )
        raise SystemExit(METHOD_FAILURE_EXIT)
    guidance = compile_public_step_guidance(case, finding)
    guidance_path = args.request.with_suffix(".guidance.txt")
    with guidance_path.open("x", encoding="utf-8") as stream:
        stream.write(guidance + "\n")
    _write_new(
        receipt_path,
        {
            "schema_version": "binary_bridge_locate_v1",
            "arm_id": args.arm_id,
            "locator_method_id": LOCATOR_ARMS[args.arm_id],
            "fixer_treatment_id": BRIDGE_TREATMENT_ID,
            "source_session": str(args.source_session),
            "public_event_count": len(case.events),
            "finding": finding.to_dict(),
            "guidance_path": str(guidance_path),
            "tracer_runtime": {
                "model": runtime.model_id,
                "base_url": runtime.base_url,
                "max_tokens": runtime.max_tokens,
                "disable_thinking": True,
            },
            "ledger_after_locate": FileSharedBudgetLedger(ledger_path).receipt(),
        },
    )
    print(json.dumps({"finding": finding.to_dict(), "guidance": str(guidance_path)}, indent=2))


def prepare(args, matrix: dict, storage_root: Path, terminal_root: Path) -> None:
    guidance_path = args.request.with_suffix(".guidance.txt")
    if not guidance_path.is_file():
        raise ValueError("run the locate stage first")
    actor = matrix["shared_actor"]
    runtime = resolve_host_pi_runtime(
        provider=str(actor["provider"]), model=str(actor["model"]), thinking=str(actor["thinking"])
    )
    task = str(matrix["task_instance"]["task_name"])
    source_cwd = frozen_source_cwd(args.source_session)
    instruction = (terminal_root / "tasks" / task / "instruction.md").read_text(encoding="utf-8")
    prefix, message_count, session_id = prefix_from_frozen_source(
        source_session=args.source_session,
        source_session_cwd=source_cwd,
        task_instruction=instruction,
        provider=runtime.provider,
        model=runtime.model,
        thinking=runtime.thinking,
        expected_terminal_stop_reason=str(matrix["expected_terminal_stop_reason"]),
    )
    config = build_terminal_historical_session_rescue_harbor_config(
        ObservedTerminalRunSpec(run_id=args.run_id, task=task, public_paths=tuple(matrix["public_paths"])),
        storage_root=storage_root,
        terminal_bench_root=terminal_root,
        source_session=args.source_session,
        source_session_cwd=source_cwd,
        prefix_tool_calls=tuple(prefix),
        source_actor={
            "provider": actor["provider"],
            "model": actor["model"],
            "transport": "host_pi_unix_rpc",
            "thinking": actor["thinking"],
            "max_output_tokens": actor["source_max_output_tokens"],
        },
        expected_terminal_stop_reason=str(matrix["expected_terminal_stop_reason"]),
        container_proxy=os.environ.get("LONGH_CONTAINER_PROXY") or None,
        continuation_treatment="targeted_guidance",
        targeted_guidance_file=guidance_path,
        continuation_opening=matrix.get("continuation_opening"),
        host_pi_timeout_seconds=args.host_pi_timeout_seconds,
        host_pi_max_output_tokens=int(actor["max_output_tokens"]),
        runtime=runtime,
    )
    # Same uniform actor bound as the managed arms (run_historical_managed_arm.prepare).
    task_timeout = task_agent_timeout_seconds(terminal_root, task)
    if task_timeout is not None and args.host_pi_timeout_seconds / task_timeout > 1.0:
        config["agent_timeout_multiplier"] = args.host_pi_timeout_seconds / task_timeout
    kwargs = config["agents"][0]["kwargs"]
    apply_prefix_snapshot(config, matrix)
    kwargs["actor_budget_ledger_path"] = str(args.request.with_suffix(".ledger.json"))
    if (matrix.get("observer") or {}).get("batched_public_snapshot"):
        # Harness knob shared by every arm of the matrix, never per arm.
        kwargs["batched_public_snapshot"] = True
    if (matrix.get("observer") or {}).get("verify_only_fallback"):
        kwargs["post_actor_image_tag"] = f"longh-postactor/{args.run_id.lower()}:final"
    kwargs["actor_upstream_base_url"] = str(actor["upstream_base_url"])
    _write_new(
        args.request,
        {
            "schema_version": "binary_bridge_arm_request_v1",
            "arm_id": args.arm_id,
            "fixer_treatment_id": BRIDGE_TREATMENT_ID,
            "provider_calls_performed_by_this_step": 0,
            "source_session": str(args.source_session),
            "source_session_id": session_id,
            "source_message_count": message_count,
            "frozen_prefix_tool_calls": len(prefix),
            "host_pi_timeout_seconds": args.host_pi_timeout_seconds,
            "docker_context": str(matrix["shared_environment"]["docker_context"]),
            "harbor_config": config,
        },
    )
    print(json.dumps({"request": str(args.request), "frozen_prefix_tool_calls": len(prefix)}))


def run(args) -> int:
    manifest = read_json(args.request)
    config_path = args.request.with_suffix(".harbor.json")
    _write_new(config_path, manifest["harbor_config"])
    environment = {**os.environ, "DOCKER_CONTEXT": str(manifest["docker_context"])}
    environment.pop("DOCKER_HOST", None)
    return subprocess.run(
        [os.environ["LONGH_HARBOR"], "run", "--config", str(config_path), "--yes"],
        check=False,
        env=environment,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("locate", "prepare", "run"), required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--source-session", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--host-pi-timeout-seconds", type=int, default=3600)
    parser.add_argument("--arm-id", choices=sorted(LOCATOR_ARMS), default=ARM_ID)
    parser.add_argument(
        "--budget-arm-id",
        help="charge the locator to this arm's ledger (a TraceDP arm taking the locator's hypothesis)",
    )
    args = parser.parse_args()
    args.source_session = args.source_session.resolve()
    args.request = args.request.resolve()
    matrix = read_json(args.matrix)
    if matrix.get("contract_version") != "v0.2":
        raise ValueError("binary-bridge arm requires a contract-v0.2 matrix")
    blockers = [item for item in matrix_launch_blockers(matrix, ARM_ID) if "arm " in item]
    if blockers:
        raise ValueError("matrix launch blockers: " + "; ".join(blockers))
    storage_root = Path(os.environ["LONGH_STORAGE_ROOT"]).resolve()
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"]).resolve()
    args.request.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "locate":
        if args.corpus_root is None:
            raise ValueError("locate needs --corpus-root (the agent_tracer reference corpus)")
        locate(args, matrix, terminal_root, args.corpus_root.resolve())
        return 0
    if args.stage == "prepare":
        prepare(args, matrix, storage_root, terminal_root)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
