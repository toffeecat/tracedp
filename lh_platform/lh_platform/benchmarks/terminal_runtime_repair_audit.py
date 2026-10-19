"""Independently bind configured predicate repairs to their durable execution chain."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

_CHECK_PREFIX_V2 = r"""
set -E -o pipefail
trap 'status=$?; trap - ERR; exit "$status"' ERR
""".strip()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a complete JSONL stream and reject non-object rows."""

    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"targeted repair artifact contains a non-object row: {path.name}")
    return rows


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _check_verdict(contract: Mapping[str, object], result: Mapping[str, object]) -> str:
    """Recompute the registered-check verdict from its public protocol."""

    details = result.get("details")
    if not isinstance(details, Mapping):
        return "unknown"
    return_code = details.get("return_code")
    if type(return_code) is not int:
        return "unknown"
    if contract.get("verdict_protocol", "exact_output_v1") == "exit_status_v1":
        return "supported" if return_code == 0 else "refuted" if return_code == 1 else "unknown"
    if result.get("is_error") is False and return_code == 0:
        content = result.get("content")
        if content == contract.get("supported_output"):
            return "supported"
        if content == contract.get("refuted_output"):
            return "refuted"
    return "unknown"


def _valid_native_result(value: object) -> bool:
    """Require the raw fields needed to distinguish execution from delivery."""

    if not isinstance(value, Mapping):
        return False
    details = value.get("details")
    executed = value.get("environment_step_executed")
    return_code = details.get("return_code") if isinstance(details, Mapping) else None
    return (
        isinstance(value.get("content"), str)
        and type(value.get("is_error")) is bool
        and type(executed) is bool
        and isinstance(details, Mapping)
        and (executed is False or type(return_code) is int)
        and (executed is False or value.get("is_error") is (return_code != 0))
    )


def _configured_contracts(
    repair_configuration: object,
    check_configuration: object,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    """Validate the two frozen contract sets without using runtime constructors."""

    if not isinstance(repair_configuration, Mapping) or (
        repair_configuration.get("schema_version")
        != "runtime_predicate_repair_configuration_v1"
        or repair_configuration.get("implementation_status") != "implementation_unverified"
        or set(repair_configuration) != {
            "schema_version",
            "implementation_status",
            "contracts",
        }
        or not isinstance(repair_configuration.get("contracts"), list)
    ):
        raise ValueError("targeted repair frozen configuration is invalid")
    if not isinstance(check_configuration, Mapping) or not isinstance(
        check_configuration.get("contracts"), list
    ):
        raise ValueError("targeted repair lacks frozen registered checks")
    repairs = list(repair_configuration["contracts"])
    checks = list(check_configuration["contracts"])
    if not repairs or any(not isinstance(item, Mapping) for item in repairs):
        raise ValueError("targeted repair frozen contracts are invalid")
    repair_ids = [item.get("repair_id") for item in repairs]
    if any(
        not isinstance(item, str) or not item or ":" in item for item in repair_ids
    ) or len(
        repair_ids
    ) != len(set(repair_ids)):
        raise ValueError("targeted repair frozen identities repeat or are invalid")
    if any(
        not isinstance(check, Mapping)
        or not isinstance(check.get("task_id"), str)
        or not isinstance(check.get("proposition"), str)
        or not isinstance(check.get("arguments"), Mapping)
        or not isinstance(check["arguments"].get("command"), str)
        or not _positive_int(check["arguments"].get("timeout", 120))
        or not isinstance(check.get("resource_ids"), list)
        or not check["resource_ids"]
        or any(not isinstance(item, str) or not item for item in check["resource_ids"])
        or check.get("supported_output") == check.get("refuted_output")
        for check in checks
    ):
        raise ValueError("targeted repair registered check fields are invalid")
    for repair in repairs:
        arguments = repair.get("arguments")
        bash_arguments = (
            isinstance(arguments, Mapping)
            and set(arguments) == {"command"}
            and isinstance(arguments.get("command"), str)
            and arguments["command"]
        )
        guidance_arguments = (
            isinstance(arguments, Mapping)
            and set(arguments)
            == {"candidate_form", "guidance_text", "fixer_identity", "allowed_tools"}
            and arguments.get("candidate_form") == "actor_guidance"
            and isinstance(arguments.get("guidance_text"), str)
            and arguments["guidance_text"].strip()
            and isinstance(arguments.get("fixer_identity"), str)
            and arguments["fixer_identity"].strip()
            and arguments.get("allowed_tools") == "bash,read,write"
        )
        if (
            set(repair) != {
                "task_id",
                "repair_id",
                "proposition",
                "arguments",
                "resource_ids",
                "time_limit_seconds",
            }
            or not isinstance(repair.get("task_id"), str)
            or not isinstance(repair.get("proposition"), str)
            or not (bash_arguments or guidance_arguments)
            or not isinstance(repair.get("resource_ids"), list)
            or not repair["resource_ids"]
            or any(
                not isinstance(item, str) or not item
                for item in repair["resource_ids"]
            )
            or not _positive_int(repair.get("time_limit_seconds"))
        ):
            raise ValueError("targeted repair frozen contract fields are invalid")
        matches = [
            check
            for check in checks
            if isinstance(check, Mapping)
            and check.get("task_id") == repair["task_id"]
            and check.get("proposition") == repair["proposition"]
            and check.get("resource_ids") == repair["resource_ids"]
        ]
        if len(matches) != 1:
            raise ValueError("targeted repair lacks one frozen registered check")
    return repairs, checks


def _kernel_version_boundaries(
    rows: list[dict[str, object]], targeted_ids: set[str]
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Rebuild resource versions from observations around each targeted action."""

    if not rows or rows[0].get("event_type") != "initialized":
        raise ValueError("targeted repair Kernel history lacks initialization")
    initial = rows[0].get("payload")
    observations = initial.get("observations") if isinstance(initial, Mapping) else None
    if not isinstance(observations, list):
        raise ValueError("targeted repair Kernel initialization is invalid")
    current = {
        str(item["resource_id"]): dict(item)
        for item in observations
        if isinstance(item, Mapping) and isinstance(item.get("resource_id"), str)
    }
    # Kernel assigns version one when the initial observation is first admitted.
    versions = {resource_id: 1 for resource_id in current}
    before: dict[str, dict[str, int]] = {}
    after: dict[str, dict[str, int]] = {}
    for row in rows[1:]:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        action_id = payload.get("action_id")
        if row.get("event_type") == "action_started" and action_id in targeted_ids:
            before[str(action_id)] = dict(versions)
        if row.get("event_type") != "action_finished":
            continue
        terminal_observations = payload.get("observations")
        if not isinstance(terminal_observations, list):
            raise ValueError("targeted repair Kernel observations are invalid")
        for item in terminal_observations:
            if not isinstance(item, Mapping) or not isinstance(item.get("resource_id"), str):
                raise ValueError("targeted repair Kernel observation is invalid")
            resource_id = str(item["resource_id"])
            normalized = dict(item)
            if resource_id not in current:
                versions[resource_id] = 1
            elif current[resource_id] != normalized:
                versions[resource_id] += 1
            current[resource_id] = normalized
        if action_id in targeted_ids:
            after[str(action_id)] = dict(versions)
    if set(before) != targeted_ids or set(after) != targeted_ids:
        raise ValueError("targeted repair Kernel start or finish boundary is missing")
    return before, after


def _selected_repair_bindings(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Join targeted selections, starts, and outcomes by explicit policy identity."""

    selected: dict[str, dict[str, object]] = {}
    action_to_decision: dict[str, str] = {}
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("targeted repair joint-policy payload is invalid")
        event_type = row.get("event_type")
        if event_type == "decision_selected" and payload.get("selected_action_type") == (
            "run_targeted_repair"
        ):
            decision_id = payload.get("decision_id")
            slate = payload.get("slate")
            candidates = slate.get("candidates") if isinstance(slate, Mapping) else None
            matches = [
                item
                for item in candidates or []
                if isinstance(item, Mapping)
                and item.get("instance_action_id")
                == payload.get("selected_instance_action_id")
            ]
            if (
                not isinstance(decision_id, str)
                or decision_id in selected
                or len(matches) != 1
                or matches[0].get("action_type") != "run_targeted_repair"
                or not isinstance(slate.get("trigger_action_id"), str)
            ):
                raise ValueError("targeted repair policy selection is invalid")
            selected[decision_id] = {
                "candidate": matches[0],
                "trigger_action_id": slate["trigger_action_id"],
                "executed_action_id": None,
                "outcome": None,
            }
        elif event_type == "execution_started" and payload.get("action_type") == (
            "run_targeted_repair"
        ):
            decision_id = payload.get("decision_id")
            action_id = payload.get("executed_action_id")
            binding = selected.get(str(decision_id))
            if (
                binding is None
                or not isinstance(action_id, str)
                or action_id in action_to_decision
                or payload.get("instance_action_id")
                != binding["candidate"].get("instance_action_id")
            ):
                raise ValueError("targeted repair policy start is invalid")
            binding["executed_action_id"] = action_id
            action_to_decision[action_id] = str(decision_id)
        elif event_type == "execution_outcome" and payload.get("action_type") == (
            "run_targeted_repair"
        ):
            decision_id = payload.get("decision_id")
            binding = selected.get(str(decision_id))
            if (
                binding is None
                or binding["outcome"] is not None
                or payload.get("executed_action_id") != binding["executed_action_id"]
                or payload.get("instance_action_id")
                != binding["candidate"].get("instance_action_id")
                or payload.get("status")
                not in {"completed", "failed", "cancelled_with_unknown_effect"}
            ):
                raise ValueError("targeted repair policy outcome is invalid")
            binding["outcome"] = payload
    if any(
        binding["executed_action_id"] is None or binding["outcome"] is None
        for binding in selected.values()
    ):
        raise ValueError("targeted repair policy lifecycle is incomplete")
    return {str(binding["executed_action_id"]): binding for binding in selected.values()}


def _is_actor_guidance_repair(repair: Mapping[str, object]) -> bool:
    arguments = repair.get("arguments")
    return isinstance(arguments, Mapping) and arguments.get("candidate_form") == "actor_guidance"


def _audit_actor_guidance_execution(
    *,
    action_id: str,
    repair: Mapping[str, object],
    check: Mapping[str, object],
    repair_index: int,
    check_index: int,
    args: Mapping[str, object],
    candidate: Mapping[str, object],
    execution: Mapping[str, object],
    operation: Mapping[str, object],
    compiled: object,
    receipt: Mapping[str, object],
    binding: Mapping[str, object],
    finished: Mapping[str, object],
    versions_before: Mapping[str, object],
    versions_after: Mapping[str, object],
    expected_instance: str,
) -> dict[str, object]:
    """Authenticate isolated Actor-guidance trial plus live delivery-only receipt."""

    if not isinstance(compiled, Mapping):
        raise ValueError("actor guidance compiled arguments are invalid")
    if compiled.get("candidate_trial_protocol") != (
        "harbor_separate_environment_filesystem_clone_v1"
    ):
        raise ValueError("actor guidance repair was selected without an isolated trial")
    check_timeout = check.get("arguments", {}).get("timeout", 120)
    base_limit = int(repair["time_limit_seconds"]) * 1000 + int(check_timeout) * 1000
    versions = args.get("resource_versions")
    expected_compiled = {
        "repair_id": repair.get("repair_id"),
        "repair_arguments": dict(repair["arguments"]),
        "postcondition_check_arguments": dict(check["arguments"]),
        "check_index": check_index,
        "incident_id": args.get("incident_id"),
        "candidate_trial_protocol": compiled["candidate_trial_protocol"],
    }
    extra_limit = args.get("time_limit_ms")
    expected_action = {
        "decision_action_id": expected_instance,
        "repair_id": repair.get("repair_id"),
        "repair_index": repair_index,
        "check_index": check_index,
        "incident_id": args.get("incident_id"),
        "resource_id": versions[0][0] if isinstance(versions, list) and versions else None,
        "resource_version": versions[0][1] if isinstance(versions, list) and versions else None,
        "task_id": repair["task_id"],
        "proposition": repair["proposition"],
        "repair_arguments": dict(repair["arguments"]),
        "check_arguments": dict(check["arguments"]),
        "resource_ids": list(repair["resource_ids"]),
        "resource_versions": versions,
        "time_limit_ms": extra_limit,
        "candidate_trial_protocol": compiled["candidate_trial_protocol"],
    }
    if check.get("check_id") is not None:
        expected_action.update(
            check_id=check.get("check_id"), predicate_id=check.get("predicate_id")
        )
    if (
        dict(compiled) != expected_compiled
        or dict(args) != expected_action
        or candidate.get("instance_action_id") != expected_instance
        or candidate.get("action_type") != "run_targeted_repair"
        or execution.get("delivery_only") is not True
        or operation.get("operation")
        != "trial_in_separate_environment_then_deliver_frozen_guidance_if_admitted"
        or candidate.get("transition_model_id")
        != "separate_environment_trial_then_frozen_guidance_delivery_v1"
        or type(candidate.get("cost_units")) is not int
        or candidate.get("cost_units") < 2
        or type(extra_limit) is not int
        or extra_limit <= base_limit
    ):
        raise ValueError("actor guidance selection differs from frozen contracts")
    expected_repair_arguments = {
        **dict(repair["arguments"]),
        "timeout": repair["time_limit_seconds"],
    }
    phase = finished.get("phase")
    outcome = binding["outcome"]
    expected_status = (
        "completed"
        if phase == "completed"
        else "cancelled_with_unknown_effect"
        if phase == "cancelled_with_unknown_effect"
        else "failed"
    )
    elapsed = (
        receipt.get("repair_elapsed_ms"),
        receipt.get("check_elapsed_ms"),
        receipt.get("total_elapsed_ms"),
    )
    if (
        receipt.get("schema_version") != "runtime_targeted_repair_execution_receipt_v1"
        or receipt.get("repair_arguments") != expected_repair_arguments
        or receipt.get("phase") != phase
        or outcome.get("status") != expected_status
        or any(not _nonnegative_int(item) for item in elapsed)
        or receipt.get("total_elapsed_ms")
        < receipt.get("repair_elapsed_ms") + receipt.get("check_elapsed_ms")
        or receipt.get("evidence_id") is not None
        or receipt.get("evidence_state") is not None
        or finished.get("reason") != receipt.get("execution_error")
        or any(
            versions_before.get(resource_id) != versions_after.get(resource_id)
            for resource_id in repair["resource_ids"]
        )
    ):
        raise ValueError("actor guidance native receipt differs from live delivery")
    repair_result = receipt.get("repair_result")
    if not isinstance(repair_result, Mapping) or not _valid_native_result(repair_result):
        raise ValueError("actor guidance native repair result is invalid")
    details = repair_result.get("details")
    trial = details.get("candidate_trial_receipt") if isinstance(details, Mapping) else None
    if not isinstance(trial, Mapping) or trial.get("main_state_unchanged") is not True:
        raise ValueError("actor guidance isolated trial receipt is missing")
    if phase == "completed":
        if (
            repair_result.get("environment_step_executed") is not False
            or repair_result.get("is_error") is not False
            or receipt.get("check_result") is not None
            or receipt.get("execution_error") is not None
            or trial.get("verdict") != "supported"
            or details.get("candidate_form") != "actor_guidance"
            or details.get("effect_status") != "awaiting_recheck"
            or repair_result.get("content") != repair["arguments"]["guidance_text"]
        ):
            raise ValueError("actor guidance live delivery claimed a main-session repair")
    elif phase == "failed_without_effect":
        if (
            repair_result.get("environment_step_executed") is not False
            or repair_result.get("is_error") is not True
            or receipt.get("check_result") is not None
            or trial.get("verdict") == "supported"
            or not isinstance(receipt.get("execution_error"), str)
        ):
            raise ValueError("rejected actor guidance must not mutate the live environment")
    elif phase != "cancelled_with_unknown_effect":
        raise ValueError("actor guidance phase is unsupported")
    return {
        "runtime_action_id": action_id,
        "repair_id": repair.get("repair_id"),
        "incident_id": args.get("incident_id"),
        "phase": phase,
        "evidence_state": None,
        "repair_elapsed_ms": receipt["repair_elapsed_ms"],
        "check_elapsed_ms": receipt["check_elapsed_ms"],
        "total_elapsed_ms": receipt["total_elapsed_ms"],
    }


def audit_runtime_targeted_repairs(
    *,
    repair_configuration: object,
    check_configuration: object,
    event_log_path: Path,
    joint_policy_path: Path,
    receipt_path: Path,
) -> dict[str, object] | None:
    """Authenticate repair selections, Kernel actions, native results, and evidence."""

    event_rows = _read_jsonl(event_log_path)
    decision_rows = _read_jsonl(joint_policy_path)
    receipt_rows = _read_jsonl(receipt_path)
    targeted_proposals = {
        str(row["payload"].get("action_id")): row["payload"]
        for row in event_rows
        if row.get("event_type") == "action_proposed"
        and isinstance(row.get("payload"), Mapping)
        and row["payload"].get("actor") == "runtime"
        and row["payload"].get("tool_name") == "runtime_targeted_repair_execution"
    }
    targeted_selected = any(
        row.get("event_type") == "decision_selected"
        and isinstance(row.get("payload"), Mapping)
        and row["payload"].get("selected_action_type") == "run_targeted_repair"
        for row in decision_rows
    )
    if repair_configuration is None:
        if targeted_proposals or targeted_selected or receipt_rows:
            raise ValueError("targeted repair execution lacks frozen configuration")
        return None

    repairs, checks = _configured_contracts(repair_configuration, check_configuration)
    policy_bindings = _selected_repair_bindings(decision_rows)
    if set(policy_bindings) != set(targeted_proposals):
        raise ValueError("targeted repair policy and Kernel actions differ")
    finished = {
        str(row["payload"].get("action_id")): row["payload"]
        for row in event_rows
        if row.get("event_type") == "action_finished"
        and isinstance(row.get("payload"), Mapping)
        and str(row["payload"].get("action_id")) in targeted_proposals
    }
    if set(finished) != set(targeted_proposals):
        raise ValueError("targeted repair Kernel lifecycle is incomplete")
    versions_before_by_action, versions_after_by_action = _kernel_version_boundaries(
        event_rows, set(targeted_proposals)
    )
    if any(row.get("sequence") != index for index, row in enumerate(receipt_rows)):
        raise ValueError("targeted repair native receipt sequence differs")
    native = {str(row.get("runtime_action_id")): row for row in receipt_rows}
    if len(native) != len(receipt_rows) or set(native) != set(targeted_proposals):
        raise ValueError("targeted repair native receipts are missing or duplicate")

    evidence_rows = {
        str(row["payload"].get("evidence_id")): row["payload"]
        for row in event_rows
        if row.get("event_type") == "evidence_recorded"
        and isinstance(row.get("payload"), Mapping)
    }
    legacy_incident_updates = [
        row["payload"]
        for row in event_rows
        if row.get("event_type") == "predicate_incident_updated"
        and isinstance(row.get("payload"), Mapping)
    ]
    audited = []
    evidence_bindings = []
    for action_id, proposal in targeted_proposals.items():
        args = proposal.get("arguments")
        if not isinstance(args, Mapping):
            raise ValueError("targeted repair Kernel arguments are invalid")
        repair_index = args.get("repair_index")
        check_index = args.get("check_index")
        if (
            type(repair_index) is not int
            or not 0 <= repair_index < len(repairs)
            or type(check_index) is not int
            or not 0 <= check_index < len(checks)
        ):
            raise ValueError("targeted repair contract index is invalid")
        repair = repairs[repair_index]
        check = checks[check_index]
        binding = policy_bindings[action_id]
        candidate = binding["candidate"]
        execution = candidate.get("execution_contract")
        if not isinstance(execution, Mapping):
            raise ValueError("targeted repair selection lacks an execution contract")
        operation = execution.get("proposed_check_or_repair")
        if not isinstance(operation, Mapping):
            raise ValueError("targeted repair selection operation is invalid")
        compiled = operation.get("arguments") if isinstance(operation, Mapping) else None
        versions = args.get("resource_versions")
        incident_id = args.get("incident_id")
        repair_id = repair.get("repair_id")
        expected_instance = (
            f"run-targeted-repair:{repair_id}:{incident_id}:"
            + ",".join(str(item[1]) for item in versions or [])
        )
        if _is_actor_guidance_repair(repair):
            audited.append(
                _audit_actor_guidance_execution(
                    action_id=action_id,
                    repair=repair,
                    check=check,
                    repair_index=repair_index,
                    check_index=check_index,
                    args=args,
                    candidate=candidate,
                    execution=execution,
                    operation=operation,
                    compiled=compiled,
                    receipt=native[action_id],
                    binding=binding,
                    finished=finished[action_id],
                    versions_before=versions_before_by_action[action_id],
                    versions_after=versions_after_by_action[action_id],
                    expected_instance=expected_instance,
                )
            )
            evidence_bindings.append(
                {
                    "runtime_action_id": action_id,
                    "trigger_action_id": binding["trigger_action_id"],
                    "instance_action_id": expected_instance,
                    "arguments": dict(args),
                    "evidence_payload": None,
                    "contract": dict(check),
                    "receipt": dict(native[action_id]),
                }
            )
            continue
        check_timeout = check.get("arguments", {}).get("timeout", 120)
        expected_limit = int(repair["time_limit_seconds"]) * 1000 + int(check_timeout) * 1000
        expected_compiled = {
            "repair_id": repair_id,
            "repair_arguments": dict(repair["arguments"]),
            "postcondition_check_arguments": dict(check["arguments"]),
            "check_index": check_index,
            "incident_id": incident_id,
        }
        expected_action = {
            "decision_action_id": expected_instance,
            "repair_id": repair_id,
            "repair_index": repair_index,
            "check_index": check_index,
            "incident_id": incident_id,
            "resource_id": versions[0][0] if isinstance(versions, list) and versions else None,
            "resource_version": versions[0][1] if isinstance(versions, list) and versions else None,
            "task_id": repair["task_id"],
            "proposition": repair["proposition"],
            "repair_arguments": dict(repair["arguments"]),
            "check_arguments": dict(check["arguments"]),
            "resource_ids": list(repair["resource_ids"]),
            "resource_versions": versions,
            "time_limit_ms": expected_limit,
        }
        if check.get("check_id") is not None:
            expected_action.update(
                check_id=check.get("check_id"), predicate_id=check.get("predicate_id")
            )
        target = candidate.get("target_binding")
        expected_versions_before = [
            [resource_id, versions_before_by_action[action_id].get(resource_id)]
            for resource_id in repair["resource_ids"]
        ]
        legality = evidence_rows.get(str(candidate.get("legality_evidence_id")))
        legality_transition = (
            legality.get("predicate_incident_transition")
            if isinstance(legality, Mapping)
            else None
        )
        legality_legacy_updates = [
            row
            for row in legacy_incident_updates
            if row.get("incident_id") == incident_id
            and row.get("evidence_id") == candidate.get("legality_evidence_id")
        ]
        if (
            not isinstance(versions, list)
            or not versions
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or type(item[1]) is not int
                for item in versions
            )
            or list(repair["resource_ids"]) != [item[0] for item in versions]
            or versions != expected_versions_before
            or args != expected_action
            or candidate.get("instance_action_id") != expected_instance
            or candidate.get("action_type") != "run_targeted_repair"
            or candidate.get("affected_task_ids") != [repair["task_id"]]
            or candidate.get("cost_units") != 1
            or candidate.get("cost_bytes") != 0
            or candidate.get("transition_model_id")
            != "configured_repair_then_registered_recheck_v1"
            or candidate.get("legality_evidence_id")
            not in execution.get("supporting_evidence", [])
            or not isinstance(target, Mapping)
            or target.get("resource_id") != versions[0][0]
            or target.get("resource_version") != versions[0][1]
            or target.get("task_id") != repair["task_id"]
            or execution.get("target_predicate") != repair["proposition"]
            or execution.get("scope")
            != [repair["task_id"], *repair["resource_ids"]]
            or execution.get("success_postcondition")
            != "registered check observes predicate supported: "
            + str(repair["proposition"])
            or execution.get("failure_branch")
            != "record refuted or unknown postcondition evidence and keep the incident open"
            or operation.get("operation")
            != "execute_targeted_repair_then_registered_check"
            or compiled != expected_compiled
            or execution.get("time_limit_ms") != expected_limit
            or execution.get("delivery_only") is not False
            or not isinstance(legality, Mapping)
            or legality.get("state") not in {"refuted", "unknown"}
            or legality.get("resource_versions") != versions
            or not (
                isinstance(legality_transition, Mapping)
                and legality_transition.get("incident_id") == incident_id
                or len(legality_legacy_updates) == 1
            )
        ):
            raise ValueError("targeted repair selection differs from frozen contracts")

        receipt = native[action_id]
        expected_repair_arguments = {
            **dict(repair["arguments"]),
            "timeout": repair["time_limit_seconds"],
        }
        expected_check_arguments = {
            **dict(check["arguments"]),
            "command": _CHECK_PREFIX_V2 + "\n" + str(check["arguments"]["command"]),
        }
        phase = finished[action_id].get("phase")
        outcome = binding["outcome"]
        expected_status = (
            "completed"
            if phase == "completed"
            else "cancelled_with_unknown_effect"
            if phase == "cancelled_with_unknown_effect"
            else "failed"
        )
        elapsed = (
            receipt.get("repair_elapsed_ms"),
            receipt.get("check_elapsed_ms"),
            receipt.get("total_elapsed_ms"),
        )
        if (
            receipt.get("schema_version")
            != "runtime_targeted_repair_execution_receipt_v1"
            or receipt.get("trigger_action_id") != binding["trigger_action_id"]
            or receipt.get("repair_index") != repair_index
            or receipt.get("check_index") != check_index
            or receipt.get("incident_id") != incident_id
            or receipt.get("repair_arguments") != expected_repair_arguments
            or receipt.get("check_arguments") != expected_check_arguments
            or receipt.get("phase") != phase
            or outcome.get("status") != expected_status
            or any(not _nonnegative_int(item) for item in elapsed)
            or receipt.get("total_elapsed_ms")
            < receipt.get("repair_elapsed_ms") + receipt.get("check_elapsed_ms")
            or type(receipt.get("observation_complete")) is not bool
            or finished[action_id].get("reason") != receipt.get("execution_error")
            or outcome.get("reason") != (
                None if expected_status == "completed" else receipt.get("execution_error")
            )
        ):
            raise ValueError("targeted repair native receipt differs from execution")

        repair_result = receipt.get("repair_result")
        check_result = receipt.get("check_result")
        if repair_result is not None and not _valid_native_result(repair_result):
            raise ValueError("targeted repair native repair result is invalid")
        if check_result is not None and not _valid_native_result(check_result):
            raise ValueError("targeted repair native check result is invalid")
        repair_succeeded = isinstance(repair_result, Mapping) and (
            repair_result.get("environment_step_executed") is True
            and repair_result.get("is_error") is False
        )
        repair_details = repair_result.get("details") if isinstance(repair_result, Mapping) else None
        check_details = check_result.get("details") if isinstance(check_result, Mapping) else None
        observed_change = any(
            versions_before_by_action[action_id].get(resource_id)
            != versions_after_by_action[action_id].get(resource_id)
            for resource_id in repair["resource_ids"]
        )
        if phase != "cancelled_with_unknown_effect":
            if not isinstance(repair_result, Mapping):
                expected_phase_from_raw = (
                    "failed_after_partial_effect"
                    if receipt.get("observation_complete") is True and observed_change
                    else "failed_with_unknown_effect"
                )
            elif repair_result.get("environment_step_executed") is not True:
                expected_phase_from_raw = "failed_without_effect"
            elif repair_result.get("is_error") is True:
                expected_phase_from_raw = (
                    "failed_after_partial_effect"
                    if receipt.get("observation_complete") is True and observed_change
                    else "failed_with_unknown_effect"
                )
            else:
                expected_phase_from_raw = (
                    "completed"
                    if isinstance(check_result, Mapping)
                    and check_result.get("environment_step_executed") is True
                    and receipt.get("observation_complete") is True
                    else "failed_with_unknown_effect"
                )
            if phase != expected_phase_from_raw:
                raise ValueError("targeted repair phase differs from raw execution")
        if (
            isinstance(check_result, Mapping) and not repair_succeeded
            or phase == "completed"
            and (not repair_succeeded or not isinstance(check_result, Mapping))
            or isinstance(repair_details, Mapping)
            and repair_result.get("environment_step_executed") is True
            and repair_details.get("effective_command_timeout_seconds")
            != repair["time_limit_seconds"]
            or isinstance(check_details, Mapping)
            and check_result.get("environment_step_executed") is True
            and check_details.get("effective_command_timeout_seconds") != check_timeout
            or phase == "completed" and receipt.get("observation_complete") is not True
        ):
            raise ValueError("targeted repair mandatory recheck boundary differs")
        verdict = _check_verdict(check, check_result) if isinstance(check_result, Mapping) else None
        expected_evidence_id = (
            f"runtime-check:{action_id}:{check_index}"
            if phase == "completed" and isinstance(check_result, Mapping)
            else None
        )
        if (
            receipt.get("evidence_id") != expected_evidence_id
            or receipt.get("evidence_state")
            != (verdict if expected_evidence_id is not None else None)
            or (verdict == "supported" and phase != "completed")
            or (phase == "completed" and receipt.get("execution_error") is not None)
            or (phase != "completed" and not isinstance(receipt.get("execution_error"), str))
        ):
            raise ValueError("targeted repair check verdict differs from evidence receipt")
        evidence_payload = None
        if expected_evidence_id is not None:
            evidence_payload = evidence_rows.get(expected_evidence_id)
            expected_incident_status = {
                "supported": "resolved",
                "refuted": "open",
                "unknown": "unknown",
            }[str(verdict)]
            expected_evidence = {
                "evidence_id": expected_evidence_id,
                "task_id": check["task_id"],
                "proposition": check["proposition"],
                "state": verdict,
                "resource_versions": [
                    [resource_id, versions_after_by_action[action_id].get(resource_id)]
                    for resource_id in check["resource_ids"]
                ],
                "dependency_claims": {},
                "dependency_claims_version_bound": True,
                "source_action_id": action_id,
                **(
                    {
                        "predicate_id": check["predicate_id"],
                        "check_id": check["check_id"],
                    }
                    if check.get("predicate_id") is not None
                    else {}
                ),
            }
            if not isinstance(evidence_payload, Mapping) or any(
                evidence_payload.get(key) != value for key, value in expected_evidence.items()
            ):
                raise ValueError("targeted repair supported evidence differs from Kernel")
            transition = evidence_payload.get("predicate_incident_transition")
            legacy_updates = [
                row
                for row in legacy_incident_updates
                if row.get("incident_id") == incident_id
                and row.get("evidence_id") == expected_evidence_id
            ]
            atomic_transition = (
                isinstance(transition, Mapping)
                and transition.get("incident_id") == incident_id
                and transition.get("evidence_id") == expected_evidence_id
                and transition.get("status") == expected_incident_status
            )
            if not atomic_transition and len(legacy_updates) != 1:
                raise ValueError("targeted repair incident transition differs from verdict")
        evidence_bindings.append(
            {
                "runtime_action_id": action_id,
                "trigger_action_id": binding["trigger_action_id"],
                "instance_action_id": expected_instance,
                "arguments": dict(args),
                "evidence_payload": dict(evidence_payload) if evidence_payload else None,
                "contract": dict(check),
                "receipt": dict(receipt),
            }
        )
        audited.append(
            {
                "runtime_action_id": action_id,
                "repair_id": repair_id,
                "incident_id": incident_id,
                "phase": phase,
                "evidence_state": verdict,
                "repair_elapsed_ms": receipt["repair_elapsed_ms"],
                "check_elapsed_ms": receipt["check_elapsed_ms"],
                "total_elapsed_ms": receipt["total_elapsed_ms"],
            }
        )
    return {
        "schema_version": "runtime_targeted_repair_evidence_audit_v1",
        "scope": "public_predicate_repair_execution_consistency_only",
        "official_evaluator_result_inferred": False,
        "implementation_status": "implementation_unverified",
        "executions": audited,
        "check_bindings": evidence_bindings,
    }


__all__ = ["audit_runtime_targeted_repairs"]
