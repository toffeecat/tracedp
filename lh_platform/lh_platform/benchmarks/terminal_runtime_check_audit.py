"""Cross-check predicate evidence against declarations and executed tool receipts."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from lh_platform.benchmarks.terminal_runtime_check import (
    SUBJECT_REFUTABLE_PRESENCE_POLICY,
)

_REGISTERED_CHECK_EXECUTION_PREFIX_V2 = r"""
set -E -o pipefail
trap 'status=$?; trap - ERR; exit "$status"' ERR
""".strip()


def _verdict_protocol(contract: dict[str, object]) -> str:
    """Decode old exact-output contracts without consulting production helpers."""

    return str(contract.get("verdict_protocol", "exact_output_v1"))


def _execution_contract(
    contract: dict[str, object], *, execution_wrapper: str = "bash_pipefail_v1"
) -> tuple[str, str]:
    """Independently derive the action and receipt identities for one protocol."""

    if execution_wrapper not in {"bash_pipefail_v1", "bash_pipefail_errtrap_v2"}:
        raise ValueError("unsupported runtime check execution wrapper")
    mode_prefix = (
        "bash_pipefail_errtrap"
        if execution_wrapper == "bash_pipefail_errtrap_v2"
        else "bash_pipefail"
    )
    if _verdict_protocol(contract) == "exit_status_v1":
        return (
            f"{mode_prefix}_exit_status_{'v2' if mode_prefix.endswith('errtrap') else 'v1'}",
            "return_0_supported_1_refuted_other_unknown_v1",
        )
    return (
        f"{mode_prefix}_exact_output_{'v2' if mode_prefix.endswith('errtrap') else 'v1'}",
        "exit_0_exact_declared_output_other_unknown_v1",
    )


def _prerequisite_argument_fields(
    contract: Mapping[str, object],
    observations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Rebuild the prerequisite fields every live registered-check action records.

    An empty prerequisite list is available. Subject existence is ignored here:
    a missing subject is the predicate failing, not a blocked checker.
    """

    raw_ids = contract.get("prerequisite_resource_ids", [])
    available = all(
        observations.get(resource_id, {}).get("observed") is True
        and observations.get(resource_id, {}).get("exists") is True
        for resource_id in raw_ids
    )
    fields: dict[str, object] = {
        "prerequisite_status": "available" if available else "missing",
    }
    if raw_ids:
        fields["prerequisite_resource_ids"] = list(raw_ids)
    return fields


def _subject_observable(
    contract: Mapping[str, object],
    observations: Mapping[str, Mapping[str, object]],
) -> bool:
    """Mirror the runtime's legality rule for running a check on its subjects.

    Every subject must be observed. Only the legacy presence policy also requires the
    subject to exist; a subject-refutable (v3) check runs on a missing subject and the
    absence refutes its predicate.
    """

    legacy = contract.get("presence_policy") != SUBJECT_REFUTABLE_PRESENCE_POLICY
    return all(
        observations.get(resource_id, {}).get("observed") is True
        and (not legacy or observations.get(resource_id, {}).get("exists") is True)
        for resource_id in contract["resource_ids"]
    )


def _registered_check_arguments_match(args: object, expected: object) -> bool:
    """Match current prerequisite fields, or a pre-field log with none to record.

    Actions recorded before prerequisite_status existed omit that key. That older
    shape is still valid only when the checker had no prerequisites and was
    therefore available. A recorded status must equal the derived one.
    """

    if args == expected:
        return True
    if not isinstance(args, dict) or not isinstance(expected, dict):
        return False
    if "prerequisite_status" in args or "prerequisite_resource_ids" in args:
        return False
    if expected.get("prerequisite_status") != "available":
        return False
    if "prerequisite_resource_ids" in expected:
        return False
    historical = {
        key: value for key, value in expected.items() if key != "prerequisite_status"
    }
    return args == historical


def _expected_executed_arguments_v2(contract: dict[str, object]) -> dict[str, object]:
    """Rebuild the versioned Bash input without importing the runtime wrapper."""

    declared = contract.get("arguments")
    if not isinstance(declared, dict) or not isinstance(declared.get("command"), str):
        raise ValueError("runtime check contract lacks executable arguments")
    return {
        **declared,
        "command": _REGISTERED_CHECK_EXECUTION_PREFIX_V2 + "\n" + declared["command"],
    }


def _interpret_verdict(
    contract: dict[str, object], *, content: str, return_code: object, is_error: object
) -> str:
    """Recompute verdicts from the documented contract, not runtime code."""

    if isinstance(return_code, bool) or not isinstance(return_code, int):
        return "unknown"
    if _verdict_protocol(contract) == "exit_status_v1":
        return "supported" if return_code == 0 else "refuted" if return_code == 1 else "unknown"
    if is_error is False and return_code == 0:
        if content == contract["supported_output"]:
            return "supported"
        if content == contract["refuted_output"]:
            return "refuted"
    return "unknown"


def _blocking_fact(verdict: str, *, content: str, return_code: object) -> dict | None:
    """Rebuild the diagnostic payload expected for non-supporting outcomes."""

    if verdict == "supported":
        return None
    return {
        "return_code": (
            return_code
            if isinstance(return_code, int) and not isinstance(return_code, bool)
            else None
        ),
        "output": content,
    }


def audit_runtime_check_evidence(
    *, configuration_path: Path, event_log_path: Path, observer_path: Path,
    runtime_execution_path: Path | None = None,
    execution_wrapper: str = "bash_pipefail_v1",
    targeted_repair_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object] | None:
    """Independently derive initial verdicts; never infer official task success."""
    if execution_wrapper not in {"bash_pipefail_v1", "bash_pipefail_errtrap_v2"}:
        raise ValueError("unsupported runtime check execution wrapper")
    events = [json.loads(line) for line in event_log_path.read_text().splitlines() if line]
    recorded = [
        row["payload"] for row in events
        if row["event_type"] == "evidence_recorded"
        and row["payload"]["evidence_id"].startswith("runtime-check:")
    ]
    completed_ids = {
        row["payload"]["action_id"] for row in events
        if row["event_type"] == "action_finished" and row["payload"]["phase"] == "completed"
    }
    registrations = [
        row for row in events if row["event_type"] == "action_proposed"
        and row["payload"]["tool_name"] == "register_runtime_check"
        and row["payload"]["action_id"] in completed_ids
    ]
    phase_proposals = [
        row
        for row in events
        if row["event_type"] == "action_proposed"
        and row["payload"]["tool_name"] == "complete_runtime_check_registration"
    ]
    runtime_execution_proposals = [
        row
        for row in events
        if row["event_type"] == "action_proposed"
        and row["payload"]["tool_name"] == "runtime_registered_check_execution"
    ]
    runtime_execution_rows = (
        [json.loads(line) for line in runtime_execution_path.read_text().splitlines() if line]
        if runtime_execution_path is not None and runtime_execution_path.is_file()
        else []
    )
    if not configuration_path.exists():
        if recorded or registrations or phase_proposals or runtime_execution_proposals:
            raise ValueError("runtime checks lack their frozen configuration")
        if runtime_execution_rows:
            raise ValueError("runtime check executions lack their frozen configuration")
        return None
    configuration = json.loads(configuration_path.read_text())
    if configuration["schema_version"] not in {
        "runtime_check_configuration_v1",
        "runtime_check_configuration_v2",
        "runtime_check_configuration_v3",
    }:
        raise ValueError("unsupported runtime check audit configuration")
    contracts = list(configuration["contracts"])
    subject_refutable = [
        contract.get("presence_policy") == SUBJECT_REFUTABLE_PRESENCE_POLICY
        for contract in contracts
    ]
    # v3 is exactly the all-subject-refutable configuration the runtime emits.
    if (configuration["schema_version"] == "runtime_check_configuration_v3") != (
        bool(contracts) and all(subject_refutable)
    ) or (any(subject_refutable) and not all(subject_refutable)):
        raise ValueError("runtime check audit configuration presence policies differ")
    rows = (
        [json.loads(line) for line in observer_path.read_text().splitlines() if line]
        if observer_path.exists() else []
    )
    receipts = {row["tool_call_id"]: row for row in rows}
    if len(receipts) != len(rows):
        raise ValueError("duplicate runtime check tool receipt")
    runtime_receipts = {
        str(row.get("runtime_action_id")): row for row in runtime_execution_rows
    }
    pending_targeted_repairs = dict(targeted_repair_bindings or {})
    expected_receipt_schema = (
        "runtime_check_execution_receipt_v2"
        if execution_wrapper == "bash_pipefail_errtrap_v2"
        else "runtime_check_execution_receipt_v1"
    )
    if len(runtime_receipts) != len(runtime_execution_rows) or any(
        row.get("schema_version") != expected_receipt_schema
        or row.get("sequence") != index
        for index, row in enumerate(runtime_execution_rows)
    ):
        raise ValueError("runtime check execution receipt sequence is invalid")
    public_nodes = [
        node for node in events[0]["payload"]["task_nodes"] if node["public_requirement"]
    ]
    acceptance_enabled = (
        events[0]["payload"].get("acceptance_predicate_schema")
        == "runtime_acceptance_predicate_v1"
    )
    public_roots = [node for node in public_nodes if node.get("parent_id") is None]
    if len(public_roots) != 1:
        raise ValueError("runtime check audit lacks one public instruction root")
    public_instruction = public_roots[0]["title"]
    public_task_ids = {node.get("task_id") for node in public_nodes}
    phase_audit = None
    if configuration.get("actor_registration_phase_required") is True:
        completed_phase_proposals = [
            row for row in phase_proposals if row["payload"]["action_id"] in completed_ids
        ]
        if len(completed_phase_proposals) != 1:
            raise ValueError("required check registration phase lacks one completed action")
        phase_proposal = completed_phase_proposals[0]
        phase_action_id = phase_proposal["payload"]["action_id"]
        arguments = phase_proposal["payload"]["arguments"]
        if (
            configuration.get("actor_registration_enabled") is not True
            or phase_proposal["payload"]["actor"] != "actor"
            or arguments.get("task_id") not in public_task_ids
            or arguments.get("status")
            not in {"checks_registered", "no_public_check_available"}
            or not isinstance(arguments.get("reason"), str)
            or not arguments["reason"].strip()
        ):
            raise ValueError("check registration phase completion is invalid")
        phase_receipt = receipts.get(phase_action_id)
        result = phase_receipt.get("tool_result") if phase_receipt else None
        if (
            phase_receipt is None
            or phase_receipt.get("tool_name") != "complete_runtime_check_registration"
            or phase_receipt.get("input") != arguments
            or phase_receipt.get("environment_step_executed") is not False
            or not isinstance(result, dict)
            or result.get("is_error") is not False
            or result.get("environment_step_executed") is not False
            or result.get("details", {}).get("check_registration_phase") != "completed"
            or result.get("details", {}).get("status") != arguments["status"]
            or result.get("details", {}).get("task_id") != arguments["task_id"]
        ):
            raise ValueError("check registration phase lacks its exact receipt")
        completion_index = next(
            index for index, row in enumerate(events)
            if row["event_type"] == "action_finished"
            and row["payload"]["action_id"] == phase_action_id
            and row["payload"]["phase"] == "completed"
        )
        proposed_actions = {
            row["payload"]["action_id"]: row["payload"]
            for row in events if row["event_type"] == "action_proposed"
        }
        early_side_effects = [
            row["payload"]["action_id"]
            for index, row in enumerate(events[:completion_index])
            if row["event_type"] == "action_started"
            and proposed_actions[row["payload"]["action_id"]]["actor"] == "actor"
            and proposed_actions[row["payload"]["action_id"]]["tool_name"] in {"bash", "write"}
        ]
        completed_registration_ids = {
            row["payload"]["action_id"] for row in registrations
            if events.index(row) < completion_index
        }
        if early_side_effects:
            raise ValueError("Actor side effect started before check registration completed")
        initial_contracts = configuration.get("contracts", [])
        if (
            arguments["status"] == "checks_registered"
            and not initial_contracts
            and not completed_registration_ids
        ):
            raise ValueError("checks_registered lacks an earlier completed declaration")
        if (
            arguments["status"] == "no_public_check_available"
            and (initial_contracts or completed_registration_ids)
        ):
            raise ValueError("no-check phase completion contradicts an earlier declaration")
        phase_audit = {
            "required": True,
            "source_action_id": phase_action_id,
            "task_id": arguments["task_id"],
            "status": arguments["status"],
            "reason": arguments["reason"],
            "evidence_state": "actor_judgment_only",
        }
    elif phase_proposals:
        raise ValueError("check registration phase action was not configured")
    actions: dict[str, dict] = {}
    expected: dict[str, dict] = {}
    verified_evidence_ids: set[str] = set()
    checked = []
    declarations = []
    observations: dict[str, dict] = {}
    versions: dict[str, int] = {}
    live_checks: dict[str, dict] = {}
    evidence_contracts: dict[str, dict] = {}
    incident_latest: dict[str, str] = {}
    incident_updates: list[dict[str, object]] = []
    handled_incidents: set[str] = set()
    guidance_bindings = []
    check_guidance_bindings = []
    runtime_execution_bindings = []
    guided_check_versions = set()
    proposed_runtime_checks: dict[str, dict[str, object]] = {}
    latest_actor_finished = None
    for event in events:
        payload = event["payload"]
        if event["event_type"] in {"initialized", "observations_refreshed", "action_finished"}:
            for observation in payload.get("observations", []):
                resource_id = observation["resource_id"]
                if observations.get(resource_id) != observation:
                    observations[resource_id] = observation
                    versions[resource_id] = versions.get(resource_id, 0) + 1
        if event["event_type"] == "action_proposed":
            actions[payload["action_id"]] = payload
            if payload["tool_name"] == "runtime_registered_check_execution":
                args = payload["arguments"]
                check_index = args.get("check_index")
                contract = (
                    contracts[check_index]
                    if type(check_index) is int and 0 <= check_index < len(contracts)
                    else None
                )
                expected_versions = (
                    [[resource_id, versions.get(resource_id)] for resource_id in contract["resource_ids"]]
                    if contract is not None
                    else None
                )
                execution_mode, verdict_interpretation = (
                    _execution_contract(
                        contract, execution_wrapper=execution_wrapper
                    )
                    if contract is not None
                    else (None, None)
                )
                expected_arguments = (
                    {
                        "check_index": check_index,
                        **(
                            {
                                "check_id": contract["check_id"],
                                "predicate_id": contract["predicate_id"],
                            }
                            if "check_id" in contract
                            else {}
                        ),
                        "resource_id": contract["resource_ids"][0],
                        "resource_version": expected_versions[0][1],
                        "task_id": contract["task_id"],
                        "proposition": contract["proposition"],
                        "check_arguments": contract["arguments"],
                        "resource_ids": contract["resource_ids"],
                        "resource_versions": expected_versions,
                        "execution_mode": execution_mode,
                        "verdict_interpretation": verdict_interpretation,
                        **_prerequisite_argument_fields(contract, observations),
                        **(
                            {"verdict_protocol": "exit_status_v1"}
                            if contract is not None
                            and _verdict_protocol(contract) == "exit_status_v1"
                            else {}
                        ),
                    }
                    if contract is not None and expected_versions is not None
                    else None
                )
                if (
                    payload["actor"] != "runtime"
                    or contract is None
                    or not contract["resource_ids"]
                    or expected_versions is None
                    or not all(type(item[1]) is int for item in expected_versions)
                    or not _subject_observable(contract, observations)
                    or not _registered_check_arguments_match(args, expected_arguments)
                ):
                    raise ValueError("runtime check execution differs from live audited state")
                proposed_runtime_checks[payload["action_id"]] = {
                    "contract": contract,
                    "check_index": check_index,
                    "trigger_action_id": latest_actor_finished,
                    "versions_before": expected_versions,
                }
            elif payload["tool_name"] == "runtime_registered_check_guidance":
                args = payload["arguments"]
                check_index = args.get("check_index")
                contract = (
                    contracts[check_index]
                    if type(check_index) is int and 0 <= check_index < len(contracts)
                    else None
                )
                expected_versions = None
                version_key = None
                if contract is not None and contract["resource_ids"]:
                    candidate_versions = [
                        [resource_id, versions.get(resource_id)]
                        for resource_id in contract["resource_ids"]
                    ]
                    if all(type(item[1]) is int for item in candidate_versions):
                        expected_versions = candidate_versions
                        version_key = (
                            check_index,
                            tuple(
                                (str(item[0]), int(item[1]))
                                for item in candidate_versions
                            ),
                        )
                current_evidence = any(
                    (
                        contract.get("predicate_id") is not None
                        and row.get("predicate_id") == contract["predicate_id"]
                        or contract.get("predicate_id") is None
                        and row["task_id"] == contract["task_id"]
                        and row["proposition"] == contract["proposition"]
                    )
                    and row["resource_versions"] == expected_versions
                    and (
                        contract.get("predicate_id") is None
                        or row["state"] in {"supported", "refuted"}
                    )
                    for row in live_checks.values()
                ) if contract is not None else False
                if (
                    payload["actor"] != "runtime"
                    or contract is None
                    or expected_versions is None
                    or version_key is None
                    or not _subject_observable(contract, observations)
                    or current_evidence
                    or version_key in guided_check_versions
                    or args.get("resource_id") != contract["resource_ids"][0]
                    or args.get("resource_version") != expected_versions[0][1]
                    or args.get("check_id") != contract.get("check_id")
                    or args.get("predicate_id") != contract.get("predicate_id")
                    or args.get("task_id") != contract["task_id"]
                    or args.get("proposition") != contract["proposition"]
                    or args.get("check_arguments") != contract["arguments"]
                    or args.get("supported_output") != contract["supported_output"]
                    or args.get("refuted_output") != contract["refuted_output"]
                    or args.get("verdict_protocol", "exact_output_v1")
                    != _verdict_protocol(contract)
                    or args.get("resource_versions") != expected_versions
                    or args.get("effect_status") != "awaiting_exact_execution"
                ):
                    raise ValueError("registered check guidance differs from live audited state")
                instance_id = (
                    f"request-registered-check:{check_index}:"
                    + ",".join(str(item[1]) for item in expected_versions)
                )
                check_guidance_bindings.append({
                    "runtime_action_id": payload["action_id"],
                    "trigger_action_id": latest_actor_finished,
                    "instance_action_id": instance_id,
                    "arguments": args,
                    "scope": "prepared_exact_check_request_only",
                })
                if payload["action_id"] in completed_ids:
                    guided_check_versions.add(version_key)
            elif payload["tool_name"] == "runtime_predicate_guidance":
                args = payload["arguments"]
                evidence_id = args.get("evidence_id")
                evidence = live_checks.get(evidence_id)
                contract = evidence_contracts.get(evidence_id)
                incident_id = args.get("incident_id")
                if (
                    configuration.get("predicate_diagnosis_enabled") is not True
                    or payload["actor"] != "runtime"
                    or evidence is None or contract is None
                    or evidence["state"] != "refuted"
                    or any(versions.get(key) != version for key, version in evidence["resource_versions"])
                    or incident_latest.get(incident_id) != evidence_id
                    or incident_id in handled_incidents
                    or args.get("source_action_id") != evidence["source_action_id"]
                    or args.get("task_id") != contract["task_id"]
                    or args.get("proposition") != contract["proposition"]
                    or args.get("check_arguments") != contract["arguments"]
                    or args.get("supported_output") != contract["supported_output"]
                    or args.get("refuted_output") != contract["refuted_output"]
                    or args.get("verdict_protocol", "exact_output_v1")
                    != _verdict_protocol(contract)
                    or args.get("resource_ids") != contract["resource_ids"]
                    or args.get("effect_status") != "awaiting_recheck"
                ):
                    raise ValueError("predicate guidance differs from live audited check")
                guidance_bindings.append({
                    "runtime_action_id": payload["action_id"],
                    "trigger_action_id": latest_actor_finished,
                    "instance_action_id": f"diagnose-predicate:{incident_id}:{evidence_id}",
                    "arguments": args,
                    "scope": "prepared_guidance_source_binding_only",
                })
                if payload["action_id"] in completed_ids:
                    handled_incidents.add(incident_id)
        elif event["event_type"] == "action_started":
            if actions[payload["action_id"]]["actor"] == "actor" and expected:
                raise ValueError("runtime check evidence was deferred past the next Actor action")
            if actions[payload["action_id"]]["actor"] == "actor":
                # Unscoped facts cannot survive a new action with unknown effects.
                live_checks = {key: row for key, row in live_checks.items() if row["resource_versions"]}
        elif event["event_type"] == "action_finished":
            action_id = payload["action_id"]
            action = actions[action_id]
            if action["actor"] == "runtime" and action["tool_name"] == (
                "runtime_targeted_repair_execution"
            ):
                binding = pending_targeted_repairs.pop(action_id, None)
                if binding is None or binding.get("arguments") != action["arguments"]:
                    raise ValueError("targeted repair check lacks its independent audit")
                evidence_payload = binding.get("evidence_payload")
                contract = binding.get("contract")
                if evidence_payload is not None:
                    if not isinstance(evidence_payload, Mapping) or not isinstance(
                        contract, Mapping
                    ):
                        raise ValueError("targeted repair check binding is invalid")
                    evidence_id = evidence_payload.get("evidence_id")
                    if not isinstance(evidence_id, str):
                        raise ValueError("targeted repair check evidence identity is invalid")
                    expected[evidence_id] = dict(evidence_payload)
                    evidence_contracts[evidence_id] = dict(contract)
                    checked.append(
                        {
                            **dict(evidence_payload),
                            "arguments": contract["arguments"],
                            "observed_output": binding["receipt"]["check_result"]["content"],
                            "return_code": binding["receipt"]["check_result"]["details"].get(
                                "return_code"
                            ),
                            "executor": "runtime_targeted_repair",
                        }
                    )
                continue
            if action["actor"] == "runtime" and action["tool_name"] == (
                "runtime_registered_check_execution"
            ):
                proposed = proposed_runtime_checks.pop(action_id, None)
                receipt = runtime_receipts.pop(action_id, None)
                if proposed is None or receipt is None:
                    raise ValueError("runtime check execution lacks its independent receipt")
                contract = proposed["contract"]
                if not isinstance(contract, dict):
                    raise ValueError("runtime check execution lacks its contract")
                versions_after = [
                    [resource_id, versions.get(resource_id)]
                    for resource_id in contract["resource_ids"]
                ]
                execution_mode, _ = _execution_contract(
                    contract, execution_wrapper=execution_wrapper
                )
                expected_executed_arguments = (
                    _expected_executed_arguments_v2(contract)
                    if execution_wrapper == "bash_pipefail_errtrap_v2"
                    else None
                )
                if (
                    receipt.get("runtime_action_id") != action_id
                    or receipt.get("trigger_action_id") != proposed["trigger_action_id"]
                    or receipt.get("check_index") != proposed["check_index"]
                    or receipt.get("declared_arguments") != contract["arguments"]
                    or (
                        execution_wrapper == "bash_pipefail_errtrap_v2"
                        and receipt.get("executed_arguments")
                        != expected_executed_arguments
                    )
                    or (
                        execution_wrapper == "bash_pipefail_v1"
                        and "executed_arguments" in receipt
                    )
                    or receipt.get("execution_mode") != execution_mode
                    or receipt.get("resource_versions_before") != proposed["versions_before"]
                    or receipt.get("resource_versions_after") != versions_after
                    or type(receipt.get("observation_complete")) is not bool
                    or receipt.get("environment_step_executed") not in {True, False, None}
                    or not isinstance(receipt.get("content"), str)
                    or (
                        receipt.get("execution_error") is not None
                        and not isinstance(receipt.get("execution_error"), str)
                    )
                    or (
                        receipt.get("environment_step_executed") is None
                    ) != isinstance(receipt.get("execution_error"), str)
                    or (
                        receipt.get("environment_step_executed") is True
                        and (
                            isinstance(receipt.get("return_code"), bool)
                            or not isinstance(receipt.get("return_code"), int)
                        )
                    )
                ):
                    raise ValueError("runtime check execution receipt differs from Kernel")
                verdict = _interpret_verdict(
                    contract,
                    content=receipt["content"],
                    return_code=receipt.get("return_code"),
                    is_error=receipt.get("return_code") != 0,
                )
                expected_blocking_fact = (
                    {
                        "return_code": None,
                        "output": receipt["execution_error"],
                    }
                    if isinstance(receipt.get("execution_error"), str)
                    else _blocking_fact(
                        verdict,
                        content=receipt["content"],
                        return_code=receipt.get("return_code"),
                    )
                )
                evidence_id = (
                    f"runtime-check:{action_id}:{proposed['check_index']}"
                    if receipt.get("environment_step_executed") is True
                    and receipt.get("observation_complete") is True
                    else None
                )
                if (
                    receipt.get("evidence_id") != evidence_id
                    or receipt.get("evidence_state") != (verdict if evidence_id else None)
                    or (
                        (
                            "blocking_fact" in receipt
                            or _verdict_protocol(contract) == "exit_status_v1"
                        )
                        and receipt.get("blocking_fact") != expected_blocking_fact
                    )
                    or (payload["phase"] == "completed")
                    != (
                        receipt.get("environment_step_executed") is True
                        and receipt.get("observation_complete") is True
                    )
                ):
                    raise ValueError("runtime check execution verdict differs from receipt")
                runtime_execution_bindings.append({
                    "runtime_action_id": action_id,
                    "trigger_action_id": proposed["trigger_action_id"],
                    "instance_action_id": (
                        f"run-registered-check:{proposed['check_index']}:"
                        + ",".join(str(item[1]) for item in proposed["versions_before"])
                    ),
                    "arguments": action["arguments"],
                    "receipt": receipt,
                })
                if evidence_id is not None:
                    expected[evidence_id] = {
                        "evidence_id": evidence_id,
                        "source_action_id": action_id,
                        "task_id": contract["task_id"],
                        "proposition": contract["proposition"],
                        "state": verdict,
                        "resource_versions": versions_after,
                        **(
                            {
                                "check_id": contract["check_id"],
                                "predicate_id": contract["predicate_id"],
                            }
                            if "predicate_id" in contract
                            else {}
                        ),
                    }
                    evidence_contracts[evidence_id] = contract
                    checked.append({
                        **expected[evidence_id],
                        "arguments": contract["arguments"],
                        "observed_output": receipt["content"],
                        "return_code": receipt["return_code"],
                        "executor": "runtime",
                    })
                continue
            if action["actor"] == "actor_bridge":
                latest_actor_finished = action_id
                continue
            if action["actor"] != "actor":
                continue
            latest_actor_finished = action_id
            row = receipts[action_id]
            result = row.get("tool_result")
            if row["tool_name"] != action["tool_name"] or row["input"] != action["arguments"]:
                raise ValueError("runtime check receipt differs from action inputs")
            if action["tool_name"] == "register_runtime_check":
                if payload["phase"] != "completed":
                    continue
                args = action["arguments"]
                excerpt = args["source_excerpt"]
                if (
                    configuration.get("actor_registration_enabled") is not True
                    or not excerpt
                    or public_instruction.count(excerpt) != 1
                    or result["is_error"] is not False
                    or result["environment_step_executed"] is not False
                    or result["details"].get("check_provenance") != "actor_inferred"
                    or result["details"].get("source_excerpt") != excerpt
                    or (
                        acceptance_enabled
                        and (
                            result["details"].get("check_id")
                            != f"actor-check:{action_id}"
                            or not isinstance(
                                result["details"].get("predicate_id"), str
                            )
                        )
                    )
                    or result["details"].get(
                        "verdict_protocol", "exact_output_v1"
                    )
                    != args.get("verdict_protocol", "exact_output_v1")
                ):
                    raise ValueError("runtime check registration lacks public-source receipt")
                protocol = args.get("verdict_protocol", "exact_output_v1")
                contracts.append({
                    "task_id": args["task_id"], "proposition": args["proposition"],
                    "arguments": {"command": args["command"], "timeout": args.get("timeout", 120)},
                    "supported_output": (
                        "return_code_0" if protocol == "exit_status_v1"
                        else args["supported_output"]
                    ),
                    "refuted_output": (
                        "return_code_1" if protocol == "exit_status_v1"
                        else args["refuted_output"]
                    ),
                    "resource_ids": args.get("resource_ids", []),
                    **(
                        {
                            "check_id": result["details"]["check_id"],
                            "predicate_id": result["details"]["predicate_id"],
                        }
                        if acceptance_enabled
                        else {}
                    ),
                    **(
                        {"verdict_protocol": protocol}
                        if protocol != "exact_output_v1" else {}
                    ),
                })
                declarations.append({"source_action_id": action_id, "source_excerpt": excerpt})
                continue
            if action["tool_name"] != "bash" or not result:
                continue
            if row["observation_complete"] is not True or row["environment_step_executed"] is not True:
                continue
            for index, contract in enumerate(contracts):
                if action["arguments"] != contract["arguments"]:
                    continue
                boundary = row.get("pre_supervision_content_chars")
                content = result["content"]
                if type(boundary) is not int or not 0 <= boundary <= len(content):
                    raise ValueError("runtime check audit lacks tool content boundary")
                # The finalizer separately checks the suffix against delivered guidance.
                raw_content = content[:boundary]
                if contract["supported_output"] == contract["refuted_output"]:
                    raise ValueError("runtime check audit has ambiguous verdict outputs")
                verdict = _interpret_verdict(
                    contract,
                    content=raw_content,
                    return_code=result["details"].get("return_code"),
                    is_error=result["is_error"],
                )
                evidence_id = f"runtime-check:{action_id}:{index}"
                # Later supervision may change resources before the receipt is delivered.
                # Bind evidence to the source action's terminal observation instead.
                expected[evidence_id] = {
                    "evidence_id": evidence_id, "source_action_id": action_id,
                    "task_id": contract["task_id"], "proposition": contract["proposition"],
                    "state": verdict,
                    "resource_versions": [[key, versions[key]] for key in contract["resource_ids"]],
                    **(
                        {
                            "check_id": contract["check_id"],
                            "predicate_id": contract["predicate_id"],
                        }
                        if "predicate_id" in contract
                        else {}
                    ),
                }
                evidence_contracts[evidence_id] = contract
                checked.append({
                    **expected[evidence_id], "arguments": contract["arguments"],
                    "observed_output": raw_content,
                    "return_code": result["details"].get("return_code"),
                })
        elif event["event_type"] == "evidence_recorded":
            evidence_id = payload["evidence_id"]
            if not evidence_id.startswith("runtime-check:"):
                if payload.get("predicate_id") is not None or payload.get("check_id") is not None:
                    raise ValueError("predicate evidence lacks an audited runtime check")
                continue
            expected_evidence = expected.pop(evidence_id, None)
            if expected_evidence is None or any(
                payload.get(key) != value for key, value in expected_evidence.items()
            ):
                raise ValueError("runtime check evidence differs from executed check")
            verified_evidence_ids.add(evidence_id)
            live_checks[evidence_id] = expected_evidence
            incident_transition = payload.get("predicate_incident_transition")
            if incident_transition is not None:
                expected_incident_status = {
                    "supported": "resolved",
                    "refuted": "open",
                    "unknown": "unknown",
                }.get(str(expected_evidence["state"]))
                if (
                    not isinstance(incident_transition, dict)
                    or set(incident_transition)
                    != {"incident_id", "evidence_id", "status"}
                    or incident_transition.get("evidence_id") != evidence_id
                    or not isinstance(incident_transition.get("incident_id"), str)
                    or incident_transition.get("status") != expected_incident_status
                ):
                    raise ValueError("atomic predicate incident receipt is invalid")
                incident_id = str(incident_transition["incident_id"])
                incident_latest[incident_id] = evidence_id
                incident_updates.append(dict(incident_transition))
        elif event["event_type"] == "predicate_incident_updated":
            if payload["evidence_id"] not in verified_evidence_ids:
                raise ValueError("predicate incident lacks an audited check")
            incident_latest[payload["incident_id"]] = payload["evidence_id"]
            incident_updates.append(dict(payload))
    if expected:
        raise ValueError("executed runtime check lacks recorded evidence")
    if proposed_runtime_checks or runtime_receipts or pending_targeted_repairs:
        raise ValueError("runtime check execution receipts differ from proposed actions")
    return {
        "schema_version": "runtime_check_evidence_audit_v1",
        "scope": "declaration_and_tool_receipt_consistency_only",
        "semantic_validity": "implementation_unverified",
        "actor_declarations": declarations,
        **({"registration_phase": phase_audit} if phase_audit is not None else {}),
        "checks": checked,
        **({"guidance_bindings": guidance_bindings} if guidance_bindings else {}),
        **(
            {"check_guidance_bindings": check_guidance_bindings}
            if check_guidance_bindings
            else {}
        ),
        **(
            {"runtime_execution_bindings": runtime_execution_bindings}
            if runtime_execution_bindings
            else {}
        ),
        "incident_updates": incident_updates,
    }
