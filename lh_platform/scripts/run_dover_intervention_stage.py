#!/usr/bin/env python3
"""DoVer hypothesis intervention for TraceDP v2hri.

Reads the arm's hypothesis (``<request>.locate.json``, TraceElephant all-at-once), builds
the DoVer public trial at the blamed event, and runs the author's intervention prompt and
parser (``DoVer_official_snapshot/scripts/intervention_recommender_trial.py``) through one
audited MiniMax call charged to the arm's shared ledger. Writes
``<request>.intervention.json`` and, when the intervention is valid,
``<request>.intervention.txt`` with the replacement text the launcher delivers.

Without a hypothesis, or when DoVer's output is invalid or the pool cannot pay for the
call, no text is written and the arm behaves as ``tracedp-v2hr``; the receipt records why.
Provider and network errors re-raise (an infrastructure failure, not a method result).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from run_binary_bridge_arm import (  # noqa: E402
    FAILURE_SUMMARY,
    MINIMAX_OPENAI_BASE_URL,
    _public_case,
    method_failure_reason,
)
from terminal_historical_managed_launch import read_json  # noqa: E402

from lh_platform.test_time_adaptation.dover_official_intervention_public import (  # noqa: E402
    DOVER_OFFICIAL_INTERVENTION_ADAPTED_PUBLIC_METHOD_ID,
    DOVER_OFFICIAL_INTERVENTION_SYSTEM_PROMPT,
    DoVerInterventionCompatibilityError,
    DoVerOfficialInterventionPublicAdapter,
    build_dover_public_trial_from_case,
)
from lh_platform.tracefix_bench.scorer_contracts import ResourceRole  # noqa: E402
from lh_platform.tracefix_bench.shared_budget import FileSharedBudgetLedger  # noqa: E402

SCHEMA = "tracedp_dover_intervention_stage_v1"
UPSTREAM_SCRIPT = Path("code/DoVer_official_snapshot/scripts/intervention_recommender_trial.py")
HYPOTHESIS_SOURCE = "traceelephant.all_at_once"


def _write_new(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _budget_refused(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}"
    return "local_budget_exceeded" in text or "BudgetRefusedError" in text or method_failure_reason(error) == "budget_exhausted"


def generate_intervention(
    case,
    locate_receipt: Mapping[str, object],
    *,
    task_instruction: str,
    adapter: DoVerOfficialInterventionPublicAdapter,
    chat: Callable[[list[dict[str, str]]], str],
) -> dict[str, object]:
    """One DoVer intervention at the hypothesized event; never raises for a method result."""

    finding = locate_receipt.get("finding") or {}
    events = list(finding.get("event_ids") or [])  # type: ignore[union-attr]
    if locate_receipt.get("status") == "method_failure" or not events:
        return {"status": "skipped", "reason": "no_hypothesis"}
    trial = build_dover_public_trial_from_case(
        case,
        task_instruction=task_instruction,
        failure_summary=FAILURE_SUMMARY,
        failed_event_id=str(events[0]),
        diagnosed_reason=str(finding.get("explanation") or ""),
        diagnosis_source=HYPOTHESIS_SOURCE,
    )
    prompt = adapter.build_prompt(trial)
    messages = [
        {"role": "system", "content": DOVER_OFFICIAL_INTERVENTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    record: dict[str, object] = {"hypothesis_event": events[0], "failed_step_index": trial.failed_step_index,
                                 "prompt": prompt}
    try:
        raw = chat(messages)
    except Exception as error:
        if _budget_refused(error):
            return {**record, "status": "method_failure", "method_failure": "budget_exhausted",
                    "error": f"{type(error).__name__}: {str(error)[-400:]}"}
        raise
    record["raw_response"] = raw
    try:
        category, replacement = adapter.parse_response(raw)
    except (DoVerInterventionCompatibilityError, ValueError) as error:
        return {**record, "status": "method_failure", "method_failure": "method_output_invalid",
                "error": f"{type(error).__name__}: {error}"}
    return {**record, "status": "completed", "category": category, "replacement_text": replacement}


def _audited_chat(*, model: str, max_tokens: int, ledger_path: Path):
    """The author's call shape (system + user, one completion) behind the metering proxy."""

    from openai import OpenAI

    from lh_platform.tracing.integrations._model_audit_proxy import AuditedOpenAIEndpoint

    audit = AuditedOpenAIEndpoint(
        upstream_base_url=MINIMAX_OPENAI_BASE_URL, expected_model_id=model, max_tokens=max_tokens,
        max_model_calls=1, temperature=0.0, generation_seed=0, timeout_seconds=600,
        request_overrides={"thinking": {"type": "disabled"}},
        budget_ledger_path=str(ledger_path), budget_role=ResourceRole.LOCATOR.value,
    )

    def chat(messages: list[dict[str, str]]) -> str:
        client = OpenAI(api_key=os.environ["MINIMAX_CN_API_KEY"], base_url=audit.base_url, max_retries=0)
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return response.choices[0].message.content or ""

    return audit, chat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--source-session", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    args, _unknown = parser.parse_known_args()
    matrix = read_json(args.matrix)
    terminal_root = Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"])
    task = str(matrix["task_instance"]["task_name"])
    instruction = (terminal_root / "tasks" / task / "instruction.md").read_text(encoding="utf-8")
    receipt_path = args.request.with_suffix(".intervention.json")
    ledger_path = args.request.with_suffix(".ledger.json")
    locate_receipt = read_json(args.request.with_suffix(".locate.json"))
    adapter = DoVerOfficialInterventionPublicAdapter(args.corpus_root / UPSTREAM_SCRIPT)
    case = _public_case(matrix, args.source_session, terminal_root)
    model = str(matrix["shared_actor"]["model"])
    audit, chat = _audited_chat(
        model=model, max_tokens=int(matrix["shared_budget"]["per_call_output_tokens"]), ledger_path=ledger_path,
    )
    with audit:
        result = generate_intervention(case, locate_receipt, task_instruction=instruction, adapter=adapter, chat=chat)
        if "raw_response" in result:
            # Attests the served model id and the provider's usage for the one call.
            summary = audit.snapshot()
            result["model_calls"] = summary.model_calls
    _write_new(receipt_path, {
        "schema_version": SCHEMA,
        "method_id": DOVER_OFFICIAL_INTERVENTION_ADAPTED_PUBLIC_METHOD_ID,
        "upstream_script": str(adapter.upstream_script),
        "model": model,
        **result,
        "ledger_after_intervention": FileSharedBudgetLedger(ledger_path).receipt(),
    })
    if result["status"] == "completed":
        with args.request.with_suffix(".intervention.txt").open("x", encoding="utf-8") as stream:
            stream.write(str(result["replacement_text"]) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "hypothesis_event", "category", "method_failure")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
