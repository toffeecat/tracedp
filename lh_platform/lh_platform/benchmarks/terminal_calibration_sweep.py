"""Calibration sweep: environment/action outcomes from one frozen failure entry.

This is NOT a debugging method.  It runs after the historical prefix replay and before any
continuation, DP, JIT or actor turn, and it measures

    P(y | z0, u)   for every registered check u, and one isolated trial per repair candidate,

where `z0` is one frozen entry state: every action is executed in its own fresh clone of the
same entry, so no action can contaminate another action's measurement.  The receipts are the
calibration input and the reachability evidence for downstream gates.

Field names are frozen here so calibration never needs a schema change later.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lh_platform.benchmarks.terminal_runtime_check import (
    CHECKER_PREREQUISITE_MISSING_EXIT_CODE,
    CHECKER_PREREQUISITE_MISSING_MARKER,
    RuntimeCheckContract,
    RuntimePredicateRepairContract,
    check_prerequisites_available,
    interpret_runtime_check_result,
)
from lh_platform.test_time_adaptation.portable_runtime_kernel import EvidenceState

CALIBRATION_SCHEMA = "runtime_calibration_a0_v1"

# Injected seams: the sweep never talks to Harbor directly, so it is testable without Docker.
ManifestCapture = Callable[[str], Awaitable[Mapping[str, object]]]
IsolatedEnvironment = Any
IsolatedEnvironmentFactory = Callable[[str], Awaitable[IsolatedEnvironment]]
IsolatedRestore = Callable[[Mapping[str, object], IsolatedEnvironment], Awaitable[None]]
IsolatedBashExecutor = Callable[[IsolatedEnvironment, Mapping[str, Any]], Awaitable[Mapping[str, Any]]]
RepairTrialRunner = Callable[..., Awaitable[Any]]


def _digest(payload: bytes) -> str:
    """Provenance digest for one calibrated artifact."""

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def entry_state_digest(manifest: Mapping[str, object]) -> str:
    """Digest the entry state identity so every action can prove it ran from the same z0."""

    rows = []
    for path in sorted(manifest):
        row = manifest[path]
        if not isinstance(row, Mapping):
            rows.append(f"{path}:{row}")
            continue
        if row.get("kind") == "file":
            rows.append(f"{path}:file:{_digest(bytes(row.get('content') or b''))}")
        elif row.get("kind") == "directory":
            files = row.get("files") or ()
            inner = ",".join(
                f"{name}:{_digest(bytes(content))}" for name, content in files
            )
            rows.append(f"{path}:dir:[{inner}]")
        else:
            rows.append(f"{path}:{row.get('kind')}")
    return _digest("\n".join(rows).encode("utf-8"))


def check_contract_digest(contract: RuntimeCheckContract) -> str:
    """Digest the frozen check contract so calibration can be tied to one spec revision."""

    payload = {
        "check_id": contract.check_id,
        "predicate_id": contract.predicate_id,
        "predicate_kind": contract.predicate_kind,
        "proposition": contract.proposition,
        "subject_resource_ids": list(contract.resource_ids),
        "prerequisite_resource_ids": list(contract.prerequisite_resource_ids),
        "presence_policy": contract.presence_policy,
        "verdict_protocol": contract.verdict_protocol,
        "command": contract.arguments.get("command"),
        "timeout": contract.arguments.get("timeout"),
    }
    return _digest(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _check_row(
    *,
    contract: RuntimeCheckContract,
    source_task_id: str,
    entry_digest: str,
    prerequisites_available: bool,
) -> dict[str, object]:
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "source_task_id": source_task_id,
        "failure_state_id": contract.task_id,
        "entry_state_digest": entry_digest,
        "check_id": contract.check_id,
        "predicate_id": contract.predicate_id,
        "predicate_kind": contract.predicate_kind,
        "subject_resource_ids": list(contract.resource_ids),
        "prerequisite_resource_ids": list(contract.prerequisite_resource_ids),
        "check_contract_digest": check_contract_digest(contract),
        "prerequisites_available": prerequisites_available,
        "executed": False,
        "exit_code": None,
        "verdict": None,
        "reason": None,
        "duration_ms": 0,
        "resource_versions_before": {},
        "resource_versions_after": {},
        "state_mutated": None,
        "stdout_digest": None,
        "stderr_digest": None,
    }


def _resource_versions(environment: Any) -> dict[str, int]:
    """Best-effort public-resource version snapshot, when the environment exposes one."""

    kernel = getattr(environment, "runtime_kernel", None)
    versions = getattr(kernel, "resource_versions", None)
    return dict(versions) if isinstance(versions, Mapping) else {}


def _state(value: object) -> str | None:
    """Normalize an `EvidenceState` (or a plain string) to its wire value."""

    return str(getattr(value, "value", value)) if value is not None else None


def repair_trial_row(
    *,
    repair: RuntimePredicateRepairContract,
    receipt: object,
    check_contracts: Sequence[RuntimeCheckContract],
    entry_rows: Sequence[Mapping[str, object]],
    source_task_id: str,
    entry_digest: str,
    duration_ms: int,
) -> dict[str, object]:
    """Translate one isolated trial receipt into a calibration row.

    The real `TerminalIsolatedTrialReceipt` exposes one post-trial observation per registered
    check plus a combined verdict; the target check is the unique check whose proposition
    matches the repair proposition (the platform's unambiguous-repair invariant).  Entry
    verdicts come from this sweep's own check rows for the same entry state, so
    `target_before -> target_after` is measured against z0 rather than against a guess.
    """

    bound = [check for check in check_contracts if check.proposition == repair.proposition]
    target = bound[0] if len(bound) == 1 else None
    target_check_id = str(target.check_id) if target is not None else None
    entry_by_check = {
        str(row.get("check_id")): row for row in entry_rows if row.get("kind") == "check"
    }
    before_all = {check_id: _state(row.get("verdict")) for check_id, row in entry_by_check.items()}
    after_all = {
        str(getattr(observation, "check_id", "")): _state(
            getattr(observation, "evidence_state", None)
        )
        for observation in (getattr(receipt, "observations", ()) or ())
    }
    target_before = before_all.get(target_check_id) if target_check_id else None
    target_after = (
        after_all.get(target_check_id)
        if target_check_id
        else _state(getattr(receipt, "verdict", None))
    )
    collateral_before = {
        check_id: state for check_id, state in before_all.items() if check_id != target_check_id
    }
    collateral_after = {
        check_id: state for check_id, state in after_all.items() if check_id != target_check_id
    }
    regression = any(
        collateral_before.get(check_id) == "supported"
        and collateral_after.get(check_id) == "refuted"
        for check_id in collateral_before
    )
    return {
        "kind": "repair_trial",
        "schema_version": CALIBRATION_SCHEMA,
        "source_task_id": source_task_id,
        "entry_state_digest": entry_digest,
        "repair_id": repair.repair_id,
        "target_predicate_id": getattr(target, "predicate_id", None),
        "repair_type": repair.arguments.get("candidate_form"),
        "trial_executed": True,
        "delivery_mode": "isolated_clone",
        "target_before": target_before,
        "target_after": target_after,
        "collateral_predicates_before": collateral_before,
        "collateral_predicates_after": collateral_after,
        "local_resolved": (
            target_before == "refuted" and target_after == "supported"
        ),
        "regression_detected": regression,
        "combined_verdict": _state(getattr(receipt, "verdict", None)),
        "repair_return_code": getattr(receipt, "repair_return_code", None),
        "main_state_unchanged": getattr(receipt, "main_state_unchanged", None),
        "isolated_environment_session_id": getattr(
            receipt, "isolated_environment_session_id", None
        ),
        "duration_ms": duration_ms,
        "cost": {
            "transferred_bytes": getattr(receipt, "transferred_bytes", None),
            "filesystem_transfer_calls": getattr(receipt, "filesystem_transfer_calls", None),
            "environment_create_calls": getattr(receipt, "environment_create_calls", None),
            "environment_delete_calls": getattr(receipt, "environment_delete_calls", None),
            "environment_exec_calls": getattr(receipt, "environment_exec_calls", None),
            "total_elapsed_ms": getattr(receipt, "total_elapsed_ms", None),
        },
    }


def run_calibration_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The sanity table summarizing calibration outcomes."""

    checks = [row for row in rows if row.get("kind") != "repair_trial"]
    repairs = [row for row in rows if row.get("kind") == "repair_trial"]
    verdicts = [str(row.get("verdict")) for row in checks]
    durations = [int(row.get("duration_ms") or 0) for row in checks if row.get("executed")]
    return {
        "checks": len(checks),
        "supported": verdicts.count("supported"),
        "refuted": verdicts.count("refuted"),
        "unknown": verdicts.count("unknown"),
        "mean_check_duration_ms": (
            round(sum(durations) / len(durations)) if durations else 0
        ),
        "repairs_tried": len(repairs),
        "local_resolves": sum(1 for row in repairs if row.get("local_resolved")),
        "regressions": sum(1 for row in repairs if row.get("regression_detected")),
    }


async def run_calibration_sweep(
    *,
    source_task_id: str,
    check_contracts: Sequence[RuntimeCheckContract],
    repair_contracts: Sequence[RuntimePredicateRepairContract],
    capture_entry_manifest: ManifestCapture,
    create_isolated_environment: IsolatedEnvironmentFactory,
    restore_isolated: IsolatedRestore,
    execute_isolated_bash: IsolatedBashExecutor,
    stop_isolated_environment: Callable[[IsolatedEnvironment], Awaitable[None]],
    run_repair_trial: RepairTrialRunner | None,
    kernel_for_prerequisites: Any | None,
    output_path: Path,
    negative_control: bool = False,
) -> dict[str, object]:
    """Execute every check (fresh clone each) and one isolated trial per repair candidate."""

    rows: list[dict[str, object]] = []
    # The entry manifest is captured once and replayed into every clone, which is what makes
    # each row an independent sample of P(y | z0, u).
    manifest = await capture_entry_manifest("a0-entry")
    digest = entry_state_digest(manifest)

    for index, contract in enumerate(check_contracts):
        row = _check_row(
            contract=contract,
            source_task_id=source_task_id,
            entry_digest=digest,
            prerequisites_available=True,
        )
        row["kind"] = "check"
        environment = await create_isolated_environment(f"a0-check-{index}")
        try:
            await restore_isolated(manifest, environment)
            prerequisites_available = check_prerequisites_available(
                kernel_for_prerequisites, contract
            ) if kernel_for_prerequisites is not None else True
            row["prerequisites_available"] = prerequisites_available
            row["resource_versions_before"] = _resource_versions(environment)
            if not prerequisites_available:
                # A blocked checker is UNKNOWN, never a refutation; the command is replaced by
                # the declared "cannot run" status exactly like the live runtime does.
                row["executed"] = False
                row["verdict"] = EvidenceState.UNKNOWN.value
                row["reason"] = "checker_prerequisite_missing"
                row["exit_code"] = CHECKER_PREREQUISITE_MISSING_EXIT_CODE
                row["stdout_digest"] = _digest(
                    f"{CHECKER_PREREQUISITE_MISSING_MARKER}:{','.join(contract.prerequisite_resource_ids)}".encode()
                )
            else:
                started = time.monotonic()
                result = await execute_isolated_bash(
                    environment, dict(contract.arguments)
                )
                row["duration_ms"] = int((time.monotonic() - started) * 1000)
                content = str(result.get("content") or "")
                return_code = result.get("return_code")
                row["executed"] = bool(result.get("environment_step_executed", True))
                row["exit_code"] = return_code
                state = interpret_runtime_check_result(
                    contract,
                    content=content,
                    return_code=return_code,
                    is_error=result.get("is_error"),
                )
                row["verdict"] = state.value
                if (
                    return_code == CHECKER_PREREQUISITE_MISSING_EXIT_CODE
                    or CHECKER_PREREQUISITE_MISSING_MARKER in content
                ):
                    # The command reported the declared "checker could not run" status: the
                    # verdict stays unknown and the reason is explicit, never a refutation.
                    row["executed"] = False
                    row["prerequisites_available"] = False
                    row["reason"] = "checker_prerequisite_missing"
                else:
                    row["reason"] = (
                        None if state is not EvidenceState.UNKNOWN else "ambiguous_outcome"
                    )
                row["stdout_digest"] = _digest(content.encode("utf-8", "replace"))
            row["resource_versions_after"] = _resource_versions(environment)
            before, after = row["resource_versions_before"], row["resource_versions_after"]
            row["state_mutated"] = (
                None if not before or not after else before != after
            )
        finally:
            await stop_isolated_environment(environment)
        rows.append(row)

    if run_repair_trial is not None and not negative_control:
        for repair in repair_contracts:
            started = time.monotonic()
            receipt = await run_repair_trial(repair)
            rows.append(
                repair_trial_row(
                    repair=repair,
                    receipt=receipt,
                    check_contracts=check_contracts,
                    entry_rows=rows,
                    source_task_id=source_task_id,
                    entry_digest=digest,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            )

    summary = run_calibration_summary(rows)
    payload = {
        "schema_version": CALIBRATION_SCHEMA,
        "source_task_id": source_task_id,
        "entry_state_digest": digest,
        "source_role": "negative_source_control" if negative_control else "failure_source",
        "expected_refuted_predicates": 0 if negative_control else None,
        "expected_repair_trials": 0 if negative_control else len(repair_contracts),
        "rows": rows,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload
