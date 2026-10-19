"""Separate prepared runtime guidance from Pi delivery and observed follow-up."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from lh_platform.benchmarks.terminal_pi_stream import load_pi_stream_receipt
from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    ActionPhase,
    PortableRuntimeKernel,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL receipt: {path}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL receipt is not an object: {path}")
        rows.append(row)
    return rows


def _gateway_predicate_deliveries(agent_dir: Path) -> dict[str, tuple[str, str]]:
    """Map each prepared guidance action to the Actor tool result carrying it."""
    deliveries: dict[str, tuple[str, str]] = {}
    path = agent_dir / "terminal-public-state.jsonl"
    rows = _read_jsonl(path) if path.is_file() else []
    for row in rows:
        trigger_action_id = row.get("tool_call_id")
        result = row.get("tool_result")
        details = result.get("details") if isinstance(result, Mapping) else None
        interventions = details.get("runtime_interventions") if isinstance(details, Mapping) else None
        if interventions is None:
            continue
        if not isinstance(interventions, list):
            raise ValueError("runtime intervention gateway receipt is invalid")
        for intervention in interventions:
            if not isinstance(intervention, Mapping):
                raise ValueError("runtime intervention gateway receipt is invalid")
            kind = intervention.get("intervention_kind")
            if kind not in {
                "runtime_predicate_guidance",
                "runtime_registered_check_guidance",
            }:
                continue
            runtime_action_id = intervention.get("action_id")
            if (
                not isinstance(runtime_action_id, str)
                or not isinstance(trigger_action_id, str)
                or runtime_action_id in deliveries
            ):
                raise ValueError("predicate guidance gateway delivery is invalid")
            deliveries[runtime_action_id] = (trigger_action_id, str(kind))
    return deliveries


def _pi_timeline(
    agent_dir: Path, actor_runtime: Mapping[str, object],
) -> tuple[
    dict[str, tuple[int, int]], dict[str, tuple[int, int]], list[tuple[int, int]]
]:
    """Recover ordered assistant requests and tool-result deliveries across attempts."""
    provider = actor_runtime.get("provider")
    model = actor_runtime.get("model")
    if not isinstance(provider, str) or not isinstance(model, str):
        raise ValueError("managed Actor model identity is missing")
    event_path = agent_dir / "actor-session-events.jsonl"
    events = _read_jsonl(event_path)
    attempt_count = sum(row.get("event_type") == "actor_attempt_started" for row in events)
    call_positions: dict[str, tuple[int, int]] = {}
    execution_positions: dict[str, tuple[int, int]] = {}
    assistant_positions: list[tuple[int, int]] = []
    for attempt_index in range(attempt_count):
        receipt = load_pi_stream_receipt(
            agent_dir / f"pi-attempt-{attempt_index:03d}.txt",
            expected_provider=provider,
            expected_model=model,
        )
        assistant_positions.extend(
            (attempt_index, sequence)
            for sequence in receipt.actionable_assistant_event_sequences
        )
        for call in receipt.tool_calls:
            if call.tool_call_id in call_positions:
                raise ValueError("managed Actor Pi tool call identities are duplicated")
            call_positions[call.tool_call_id] = (attempt_index, call.event_sequence)
        for execution in receipt.tool_executions:
            if execution.tool_call_id in execution_positions:
                raise ValueError("managed Actor Pi tool execution identities are duplicated")
            execution_positions[execution.tool_call_id] = (
                attempt_index, execution.event_sequence,
            )
    return call_positions, execution_positions, assistant_positions


def audit_runtime_predicate_guidance_followup(
    *,
    agent_dir: Path,
    kernel: PortableRuntimeKernel,
    actor_runtime: Mapping[str, object],
    check_evidence_audit: Mapping[str, object],
) -> dict[str, object]:
    """Audit Pi delivery, later Actor turns, exact rechecks, and incident resolution."""
    raw_bindings = check_evidence_audit.get("guidance_bindings", [])
    raw_check_bindings = check_evidence_audit.get("check_guidance_bindings", [])
    raw_checks = check_evidence_audit.get("checks", [])
    if (
        not isinstance(raw_bindings, list)
        or not isinstance(raw_check_bindings, list)
        or not isinstance(raw_checks, list)
    ):
        raise ValueError("runtime check audit guidance data is invalid")
    deliveries = _gateway_predicate_deliveries(agent_dir)
    call_positions, execution_positions, assistant_positions = _pi_timeline(
        agent_dir, actor_runtime,
    )
    action_order = {action_id: index for index, action_id in enumerate(kernel.actions)}
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for binding in raw_bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("runtime guidance binding is invalid")
        runtime_action_id = binding.get("runtime_action_id")
        trigger_action_id = binding.get("trigger_action_id")
        arguments = binding.get("arguments")
        if (
            not isinstance(runtime_action_id, str)
            or runtime_action_id in seen
            or not isinstance(trigger_action_id, str)
            or not isinstance(arguments, Mapping)
        ):
            raise ValueError("runtime guidance binding identity is invalid")
        seen.add(runtime_action_id)
        runtime_action = kernel.actions.get(runtime_action_id)
        if (
            runtime_action is None
            or runtime_action.actor != "runtime"
            or runtime_action.tool_name != "runtime_predicate_guidance"
            or runtime_action.phase is not ActionPhase.COMPLETED
            or deliveries.get(runtime_action_id)
            != (trigger_action_id, "runtime_predicate_guidance")
        ):
            raise ValueError("runtime guidance lacks its exact gateway delivery")
        incident_id = arguments.get("incident_id")
        incident = kernel.predicate_incidents.get(str(incident_id))
        if not isinstance(incident_id, str) or incident is None:
            raise ValueError("runtime guidance lacks its predicate incident")

        execution_position = execution_positions.get(trigger_action_id)
        delivered_to_pi = execution_position is not None
        later_assistant = bool(
            delivered_to_pi
            and any(position > execution_position for position in assistant_positions)
        )
        later_calls = [
            (position, action_id)
            for action_id, position in call_positions.items()
            if delivered_to_pi
            and position > execution_position
            and action_id in action_order
            and action_order[action_id] > action_order[runtime_action_id]
            and kernel.actions[action_id].actor == "actor"
        ]
        later_calls.sort()
        post_delivery_action_ids = [action_id for _, action_id in later_calls]
        post_delivery_set = set(post_delivery_action_ids)
        exact_rechecks: list[str] = []
        supported_recheck = False
        for check in raw_checks:
            if not isinstance(check, Mapping):
                raise ValueError("runtime check audit row is invalid")
            if (
                check.get("source_action_id") not in post_delivery_set
                or check.get("task_id") != arguments.get("task_id")
                or check.get("proposition") != arguments.get("proposition")
                or check.get("arguments") != arguments.get("check_arguments")
            ):
                continue
            evidence_id = check.get("evidence_id")
            if not isinstance(evidence_id, str):
                raise ValueError("runtime recheck lacks evidence identity")
            exact_rechecks.append(evidence_id)
            supported_recheck = supported_recheck or check.get("state") == "supported"
        output.append({
            "runtime_action_id": runtime_action_id,
            "trigger_action_id": trigger_action_id,
            "incident_id": incident_id,
            "delivery_status": "delivered_to_pi" if delivered_to_pi else "gateway_only",
            "actor_continuation_after_delivery_observed": later_assistant,
            "post_delivery_actor_action_ids": post_delivery_action_ids,
            "exact_rechecks": exact_rechecks,
            "final_incident_status": incident.status,
            "resolved_by_post_delivery_recheck": (
                supported_recheck and incident.status == "resolved"
            ),
        })
    check_requests: list[dict[str, object]] = []
    for binding in raw_check_bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("runtime check guidance binding is invalid")
        runtime_action_id = binding.get("runtime_action_id")
        trigger_action_id = binding.get("trigger_action_id")
        arguments = binding.get("arguments")
        if (
            not isinstance(runtime_action_id, str)
            or runtime_action_id in seen
            or not isinstance(trigger_action_id, str)
            or not isinstance(arguments, Mapping)
        ):
            raise ValueError("runtime check guidance binding identity is invalid")
        seen.add(runtime_action_id)
        runtime_action = kernel.actions.get(runtime_action_id)
        if (
            runtime_action is None
            or runtime_action.actor != "runtime"
            or runtime_action.tool_name != "runtime_registered_check_guidance"
            or runtime_action.phase is not ActionPhase.COMPLETED
            or deliveries.get(runtime_action_id)
            != (trigger_action_id, "runtime_registered_check_guidance")
        ):
            raise ValueError("runtime check guidance lacks its exact gateway delivery")
        execution_position = execution_positions.get(trigger_action_id)
        delivered_to_pi = execution_position is not None
        later_assistant = bool(
            delivered_to_pi
            and any(position > execution_position for position in assistant_positions)
        )
        later_calls = [
            (position, action_id)
            for action_id, position in call_positions.items()
            if delivered_to_pi
            and position > execution_position
            and action_id in action_order
            and action_order[action_id] > action_order[runtime_action_id]
            and kernel.actions[action_id].actor == "actor"
        ]
        later_calls.sort()
        post_delivery_action_ids = [action_id for _, action_id in later_calls]
        post_delivery_set = set(post_delivery_action_ids)
        exact_checks = [
            str(check["evidence_id"])
            for check in raw_checks
            if isinstance(check, Mapping)
            and check.get("source_action_id") in post_delivery_set
            and check.get("task_id") == arguments.get("task_id")
            and check.get("proposition") == arguments.get("proposition")
            and check.get("arguments") == arguments.get("check_arguments")
            and isinstance(check.get("evidence_id"), str)
        ]
        check_requests.append({
            "runtime_action_id": runtime_action_id,
            "trigger_action_id": trigger_action_id,
            "delivery_status": "delivered_to_pi" if delivered_to_pi else "gateway_only",
            "actor_continuation_after_delivery_observed": later_assistant,
            "post_delivery_actor_action_ids": post_delivery_action_ids,
            "exact_check_evidence_ids": exact_checks,
        })
    if set(deliveries) != seen:
        raise ValueError("predicate guidance gateway deliveries differ from audited bindings")
    return {
        "schema_version": "runtime_predicate_guidance_followup_audit_v1",
        "scope": "pi_delivery_and_observed_actor_followup_only",
        "semantic_compliance": "implementation_unverified",
        "guidance": output,
        **({"check_requests": check_requests} if check_requests else {}),
    }
