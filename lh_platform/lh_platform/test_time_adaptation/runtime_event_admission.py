"""Admit expensive runtime planning only at durable management events.

The gate is intentionally separate from the Kernel update loop. Every Actor
boundary still reaches the Kernel and slate compiler; this module decides only
whether that already-compiled boundary warrants a provider call.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

from lh_platform.test_time_adaptation.portable_runtime_kernel import ActionPhase
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)

V8_SEMANTIC_REVIEW_THRESHOLD = 2
V8_CUMULATIVE_PLANNER_BUDGET_MS = 180_000
_CHECK_EVIDENCE_SOURCE_PHASE_VALUES = frozenset(
    phase.value
    for phase in {
        ActionPhase.COMPLETED,
        ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
        ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
    }
)
_CHECK_EXECUTION_TERMINAL_PHASE_VALUES = frozenset(
    phase.value
    for phase in {
        ActionPhase.COMPLETED,
        ActionPhase.FAILED_WITHOUT_EFFECT,
        ActionPhase.FAILED_AFTER_PARTIAL_EFFECT,
        ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
        ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT,
    }
)

_EVENT_ACTION_KINDS = {
    RuntimeJointActionType.RESTORE_RESOURCE: "restore_available",
    RuntimeJointActionType.REQUEST_OUTPUT_COMPLETION: "output_completion",
    RuntimeJointActionType.REQUEST_ACTOR_CONVERGENCE: "actor_convergence",
    RuntimeJointActionType.SUBMIT_CURRENT_STATE: "actor_convergence",
    RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS: "failure_incident",
    RuntimeJointActionType.REQUEST_PREDICATE_DIAGNOSIS: "predicate_incident",
    RuntimeJointActionType.REQUEST_REGISTERED_CHECK: "predicate_check",
    RuntimeJointActionType.RUN_REGISTERED_CHECK: "predicate_check",
    RuntimeJointActionType.REQUEST_SEMANTIC_CHECK: "semantic_review",
}


def _boundary(slate: RuntimeJointActionSlate) -> dict[str, object]:
    """Bind every no-call or admission decision to the executable boundary."""

    return {
        "session_id": slate.session_id,
        "trigger_action_id": slate.trigger_action_id,
        "resource_versions": [list(item) for item in slate.resource_versions],
        "intervention_budget": slate.budget_before,
        "actor_time_remaining_ms": slate.actor_time_remaining_ms,
        "no_progress_actor_actions": slate.no_progress_actor_actions,
    }


def _validate_projection(slate: RuntimeJointActionSlate, projection: Mapping) -> None:
    """Reject a gate decision made from a different Kernel boundary."""

    actions = projection.get("actions")
    if (
        projection.get("schema_version") != "portable_runtime_public_projection_v6"
        or projection.get("session_id") != slate.session_id
        or projection.get("resource_versions") != dict(slate.resource_versions)
        or not isinstance(actions, list)
        or not actions
        or not isinstance(actions[-1], Mapping)
        or actions[-1].get("action_id") != slate.trigger_action_id
    ):
        raise ValueError("v8 event gate differs from the live Kernel boundary")


def _check_source_matches_contract(
    action: Mapping,
    contract: Mapping,
    *,
    check_index: int,
    allowed_phase_values: frozenset[str],
) -> bool:
    """Match evidence to the exact check action and interpretation contract."""

    arguments = action.get("arguments")
    check_arguments = contract.get("arguments")
    if (
        not isinstance(arguments, Mapping)
        or not isinstance(check_arguments, Mapping)
        or action.get("phase") not in allowed_phase_values
    ):
        return False
    if action.get("actor") == "actor":
        return action.get("tool_name") == "bash" and dict(arguments) == dict(
            check_arguments
        )
    if (
        action.get("actor") != "runtime"
        or action.get("tool_name") != "runtime_registered_check_execution"
    ):
        return False
    return (
        arguments.get("check_index") == check_index
        and arguments.get("check_arguments") == dict(check_arguments)
        and arguments.get("task_id") == contract.get("task_id")
        and arguments.get("proposition") == contract.get("proposition")
        and arguments.get("resource_ids") == contract.get("resource_ids")
        and arguments.get("verdict_protocol", "exact_output_v1")
        == contract.get("verdict_protocol", "exact_output_v1")
    )


def _registered_check_epochs(
    projection: Mapping,
    configured_check_contracts: Iterable[Mapping],
) -> tuple[dict[str, object], ...]:
    """Bind each check index to its latest real execution or evidence epoch."""

    actions = projection.get("actions")
    evidence = projection.get("evidence", [])
    if not isinstance(actions, list) or not isinstance(evidence, list):
        raise ValueError("v8 registered check history is invalid")
    contracts = [dict(item) for item in configured_check_contracts]
    action_positions: dict[str, int] = {}
    for position, action in enumerate(actions):
        if not isinstance(action, Mapping) or not isinstance(action.get("action_id"), str):
            raise ValueError("v8 action history is invalid")
        action_positions[str(action["action_id"])] = position
        if (
            action.get("actor") == "actor"
            and action.get("tool_name") == "register_runtime_check"
            and action.get("phase") == "completed"
        ):
            arguments = action.get("arguments")
            if not isinstance(arguments, Mapping):
                raise ValueError("v8 registered contract action is invalid")
            # Actor registration stores public grounding fields, while the
            # executed bash action contains only the normalized tool arguments.
            contracts.append(
                {
                    "task_id": arguments.get("task_id"),
                    "proposition": arguments.get("proposition"),
                    "resource_ids": arguments.get("resource_ids"),
                    "arguments": {
                        "command": arguments.get("command"),
                        "timeout": arguments.get("timeout", 120),
                    },
                    "verdict_protocol": arguments.get(
                        "verdict_protocol", "exact_output_v1"
                    ),
                }
            )

    rows = []
    for check_index, contract in enumerate(contracts):
        task_id = contract.get("task_id")
        proposition = contract.get("proposition")
        resource_ids = contract.get("resource_ids")
        check_arguments = contract.get("arguments")
        verdict_protocol = contract.get("verdict_protocol", "exact_output_v1")
        if (
            not isinstance(task_id, str)
            or not isinstance(proposition, str)
            or not isinstance(resource_ids, list)
            or any(not isinstance(item, str) for item in resource_ids)
            or not isinstance(check_arguments, Mapping)
            or not isinstance(verdict_protocol, str)
        ):
            raise ValueError("v8 registered check contract is invalid")
        matching_evidence = []
        for record in evidence:
            if not isinstance(record, Mapping):
                raise ValueError("v8 evidence history is invalid")
            versions = record.get("resource_versions")
            if not isinstance(versions, list) or any(
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or type(item[1]) is not int
                for item in versions
            ):
                raise ValueError("v8 evidence resource versions are invalid")
            if (
                record.get("task_id") == task_id
                and record.get("proposition") == proposition
                and [item[0] for item in versions] == resource_ids
            ):
                source_action_id = record.get("source_action_id")
                if (
                    not isinstance(record.get("evidence_id"), str)
                    or record.get("state")
                    not in {"supported", "refuted", "unknown", "stale"}
                    or not isinstance(source_action_id, str)
                    or source_action_id not in action_positions
                ):
                    raise ValueError("v8 check evidence identity or source action is invalid")
                source_position = action_positions[source_action_id]
                if (
                    record["evidence_id"]
                    == f"runtime-check:{source_action_id}:{check_index}"
                    and _check_source_matches_contract(
                        actions[source_position],
                        contract,
                        check_index=check_index,
                        allowed_phase_values=_CHECK_EVIDENCE_SOURCE_PHASE_VALUES,
                    )
                ):
                    matching_evidence.append((source_position, record))
        latest_evidence = max(matching_evidence, default=None, key=lambda item: item[0])
        executions = [
            (position, action)
            for position, action in enumerate(actions)
            if action.get("actor") == "runtime"
            and action.get("tool_name") == "runtime_registered_check_execution"
            and _check_source_matches_contract(
                action,
                contract,
                check_index=check_index,
                allowed_phase_values=_CHECK_EXECUTION_TERMINAL_PHASE_VALUES,
            )
        ]
        latest_execution = max(executions, default=None, key=lambda item: item[0])
        if latest_evidence is not None and (
            latest_execution is None or latest_evidence[0] >= latest_execution[0]
        ):
            position, record = latest_evidence
            source_action = actions[position]
            rows.append(
                {
                    "check_index": check_index,
                    "epoch_kind": "evidence",
                    "source_action_id": record["source_action_id"],
                    "source_actor": source_action.get("actor"),
                    "source_tool_name": source_action.get("tool_name"),
                    "source_action_phase": source_action.get("phase"),
                    "evidence_id": record.get("evidence_id"),
                    "evidence_state": record.get("state"),
                    "evidence_resource_versions": record.get("resource_versions"),
                }
            )
        elif latest_execution is not None:
            _, action = latest_execution
            rows.append(
                {
                    "check_index": check_index,
                    "epoch_kind": "terminal_execution_without_evidence",
                    "source_action_id": action["action_id"],
                    "source_actor": "runtime",
                    "source_tool_name": "runtime_registered_check_execution",
                    "source_action_phase": action.get("phase"),
                    "evidence_id": None,
                    "evidence_state": None,
                    "evidence_resource_versions": None,
                }
            )
        else:
            rows.append(
                {
                    "check_index": check_index,
                    "epoch_kind": "not_executed",
                    "source_action_id": None,
                    "source_actor": None,
                    "source_tool_name": None,
                    "source_action_phase": None,
                    "evidence_id": None,
                    "evidence_state": None,
                    "evidence_resource_versions": None,
                }
            )
    return tuple(rows)


def _candidate_event_rows(
    slate: RuntimeJointActionSlate,
    registered_check_epochs: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    """Map Runtime-owned candidates to stable event identities."""

    rows = []
    for candidate in slate.candidates:
        kind = _EVENT_ACTION_KINDS.get(candidate.action_type)
        if kind is None:
            continue
        target_binding = candidate.target_binding.to_dict()
        legality_evidence_id = candidate.legality_evidence_id
        instance_coordinate = None
        if candidate.action_type in {
            RuntimeJointActionType.REQUEST_REGISTERED_CHECK,
            RuntimeJointActionType.RUN_REGISTERED_CHECK,
        }:
            # A registered predicate is one method capability. Ordinary source
            # edits change its executable version but do not create a new
            # metareasoning event before an actual check verdict exists.
            parts = candidate.instance_action_id.split(":", 2)
            instance_coordinate = ":".join(parts[:2])
            try:
                check_index = int(parts[1])
                check_epoch = registered_check_epochs[check_index]
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError("v8 registered check candidate is invalid") from error
            target_binding = {
                **target_binding,
                "resource_version": None,
                "obligation_version_key": None,
            }
            legality_evidence_id = "registered_check_pending"
        else:
            check_epoch = None
        coordinate = {
            "event_kind": kind,
            "action_type": candidate.action_type.value,
            "target_binding": target_binding,
            "transition_model_id": candidate.transition_model_id,
            "legality_evidence_id": legality_evidence_id,
            # Incident identities are intentionally action-bound; other event
            # families use their semantic target/version coordinates instead.
            "incident_action_id": (
                candidate.instance_action_id
                if candidate.action_type
                in {
                    RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS,
                    RuntimeJointActionType.REQUEST_PREDICATE_DIAGNOSIS,
                }
                else None
            ),
            "instance_coordinate": instance_coordinate,
            "registered_check_epoch": check_epoch,
        }
        rows.append(
            {
                "event_key": json.dumps(
                    coordinate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "event_kind": kind,
                "instance_action_id": candidate.instance_action_id,
            }
        )
    return rows


def planner_event_admission_v8(
    slate: RuntimeJointActionSlate,
    projection: Mapping,
    *,
    previously_admitted_event_keys: Iterable[str] = (),
    semantic_review_threshold: int = V8_SEMANTIC_REVIEW_THRESHOLD,
    configured_check_contracts: Iterable[Mapping] = (),
) -> dict[str, object]:
    """Return a literal admission/no-call record for one live V8 boundary.

    Candidate identities include the incident, evidence, resource, obligation,
    and resource-version bindings produced by the shared slate compiler. A
    repeated candidate is therefore suppressed, while a changed incident or
    restored resource version becomes eligible again.
    """

    if semantic_review_threshold < 1:
        raise ValueError("v8 semantic review threshold must be positive")
    _validate_projection(slate, projection)
    previous = tuple(sorted(set(previously_admitted_event_keys)))
    if any(not item for item in previous):
        raise ValueError("v8 prior event identity is invalid")
    check_epochs = _registered_check_epochs(projection, configured_check_contracts)
    common = {
        "schema_version": "runtime_planner_event_admission_v1",
        "boundary": _boundary(slate),
        "semantic_review_threshold": semantic_review_threshold,
        "previously_admitted_event_keys": list(previous),
        "registered_check_epochs": list(check_epochs),
    }
    if len(slate.candidates) == 1:
        return {
            **common,
            "decision": "do_not_call_provider",
            "reason": "single_candidate_deterministic",
            "event_rows": [],
            "new_event_keys": [],
            "provider_called": False,
        }

    rows = _candidate_event_rows(slate, check_epochs)
    threshold_crossing = (
        slate.no_progress_actor_actions == semantic_review_threshold
        and any(
            item.action_type is RuntimeJointActionType.REQUEST_SEMANTIC_CHECK
            for item in slate.candidates
        )
    )
    if threshold_crossing:
        threshold_key = (
            "semantic_review_threshold_crossed:"
            + ",".join(
                item.instance_action_id
                for item in slate.candidates
                if item.action_type is RuntimeJointActionType.REQUEST_SEMANTIC_CHECK
            )
        )
        rows.append(
            {
                "event_key": threshold_key,
                "event_kind": "semantic_review_threshold_crossed",
                "instance_action_id": slate.trigger_action_id,
            }
        )
    prior = set(previous)
    new_keys = sorted({row["event_key"] for row in rows} - prior)
    if not new_keys:
        return {
            **common,
            "decision": "do_not_call_provider",
            "reason": "event_gate_suppressed",
            "event_rows": rows,
            "new_event_keys": [],
            "provider_called": False,
        }
    return {
        **common,
        "decision": "call_provider",
        "reason": "new_management_event",
        "event_rows": rows,
        "new_event_keys": new_keys,
        "provider_called": True,
    }


def validate_planner_event_admission_v8(
    slate: RuntimeJointActionSlate,
    evidence: Mapping,
) -> dict[str, object]:
    """Recompute saved gate evidence without trusting its derived fields."""

    previous = evidence.get("previously_admitted_event_keys")
    threshold = evidence.get("semantic_review_threshold")
    check_epochs = evidence.get("registered_check_epochs")
    if (
        not isinstance(previous, list)
        or any(not isinstance(item, str) for item in previous)
        or type(threshold) is not int
        or not isinstance(check_epochs, list)
        or any(not isinstance(item, Mapping) for item in check_epochs)
    ):
        raise ValueError("v8 event admission replay inputs are invalid")
    # Resolver/load validates the saved epoch inputs internally. The finalizer
    # separately rebuilds them from the complete Kernel projection and contracts.
    common = {
        "schema_version": "runtime_planner_event_admission_v1",
        "boundary": _boundary(slate),
        "semantic_review_threshold": threshold,
        "previously_admitted_event_keys": sorted(set(previous)),
        "registered_check_epochs": [dict(item) for item in check_epochs],
    }
    if len(slate.candidates) == 1:
        rebuilt = {
            **common,
            "decision": "do_not_call_provider",
            "reason": "single_candidate_deterministic",
            "event_rows": [],
            "new_event_keys": [],
            "provider_called": False,
        }
    else:
        rows = _candidate_event_rows(slate, tuple(dict(item) for item in check_epochs))
        threshold_crossing = (
            slate.no_progress_actor_actions == threshold
            and any(
                item.action_type is RuntimeJointActionType.REQUEST_SEMANTIC_CHECK
                for item in slate.candidates
            )
        )
        if threshold_crossing:
            rows.append(
                {
                    "event_key": "semantic_review_threshold_crossed:"
                    + ",".join(
                        item.instance_action_id
                        for item in slate.candidates
                        if item.action_type
                        is RuntimeJointActionType.REQUEST_SEMANTIC_CHECK
                    ),
                    "event_kind": "semantic_review_threshold_crossed",
                    "instance_action_id": slate.trigger_action_id,
                }
            )
        new_keys = sorted(
            {row["event_key"] for row in rows} - set(common["previously_admitted_event_keys"])
        )
        rebuilt = {
            **common,
            "decision": "call_provider" if new_keys else "do_not_call_provider",
            "reason": "new_management_event" if new_keys else "event_gate_suppressed",
            "event_rows": rows,
            "new_event_keys": new_keys,
            "provider_called": bool(new_keys),
        }
    if dict(evidence) != rebuilt:
        raise ValueError("v8 event admission evidence differs from the live slate")
    return rebuilt


def planner_cumulative_budget_gate_v8(
    slate: RuntimeJointActionSlate,
    *,
    cumulative_planner_elapsed_ms: int,
    planner_ceiling_ms: int,
    provider_timeout_overrun_margin_ms: int,
    cumulative_planner_budget_ms: int = V8_CUMULATIVE_PLANNER_BUDGET_MS,
) -> dict[str, object] | None:
    """Reserve a full provider ceiling before starting another paid call."""

    if (
        cumulative_planner_elapsed_ms < 0
        or planner_ceiling_ms < 1
        or provider_timeout_overrun_margin_ms < 0
        or cumulative_planner_budget_ms < planner_ceiling_ms
        + provider_timeout_overrun_margin_ms
    ):
        raise ValueError("v8 cumulative planner budget contract is invalid")
    reserved_call_ms = planner_ceiling_ms + provider_timeout_overrun_margin_ms
    required = cumulative_planner_elapsed_ms + reserved_call_ms
    if required <= cumulative_planner_budget_ms:
        return None
    return {
        "schema_version": "runtime_planner_cumulative_budget_no_call_v1",
        "boundary": _boundary(slate),
        "reason": "cumulative_planner_budget_exhausted",
        "cumulative_planner_elapsed_ms": cumulative_planner_elapsed_ms,
        "planner_ceiling_ms": planner_ceiling_ms,
        "provider_timeout_overrun_margin_ms": provider_timeout_overrun_margin_ms,
        "reserved_call_ms": reserved_call_ms,
        "cumulative_planner_budget_ms": cumulative_planner_budget_ms,
        "required_budget_ms": required,
        "provider_called": False,
    }


def planner_time_reserve_gate_v8(
    slate: RuntimeJointActionSlate,
    *,
    planner_ceiling_ms: int,
    actor_convergence_reserve_ms: int,
    provider_timeout_overrun_margin_ms: int,
) -> dict[str, object] | None:
    """Keep V7's reserve inequality under a distinct V8 evidence identity."""

    remaining = slate.actor_time_remaining_ms
    if (
        remaining is None
        or planner_ceiling_ms < 1
        or actor_convergence_reserve_ms < 1
        or provider_timeout_overrun_margin_ms < 0
    ):
        raise ValueError("v8 planner time reserve contract is invalid")
    required = (
        planner_ceiling_ms
        + actor_convergence_reserve_ms
        + provider_timeout_overrun_margin_ms
    )
    if remaining >= required:
        return None
    return {
        "schema_version": "runtime_planner_time_reserve_no_call_v1",
        "boundary": _boundary(slate),
        "reason": "precall_reserve",
        "actor_time_remaining_ms": remaining,
        "planner_ceiling_ms": planner_ceiling_ms,
        "actor_convergence_reserve_ms": actor_convergence_reserve_ms,
        "provider_timeout_overrun_margin_ms": provider_timeout_overrun_margin_ms,
        "required_time_ms": required,
        "provider_called": False,
    }


_RESOURCE_GATE_EVENT_SCHEMA_V8 = {
    "runtime_planner_cumulative_budget_no_call_v1": (
        "runtime_planner_cumulative_budget_no_call_v2"
    ),
    "runtime_planner_time_reserve_no_call_v1": (
        "runtime_planner_time_reserve_no_call_v2"
    ),
}


def bind_event_admission_to_resource_gate_v8(
    slate: RuntimeJointActionSlate,
    resource_gate: Mapping,
    event_admission: Mapping,
) -> dict[str, object]:
    """Preserve the admitted event when a later resource gate skips provider I/O.

    The legacy V1 resource-gate receipts remain valid replay inputs. New online
    decisions use V2 so an event discovered before the budget checks is not
    erased merely because the provider call cannot be afforded.
    """

    schema = resource_gate.get("schema_version")
    upgraded_schema = _RESOURCE_GATE_EVENT_SCHEMA_V8.get(schema)
    if upgraded_schema is None:
        raise ValueError("v8 resource gate schema cannot carry event evidence")
    admission = validate_planner_event_admission_v8(slate, event_admission)
    if admission["decision"] != "call_provider":
        raise ValueError("v8 resource gate requires an admitted event")
    if (
        resource_gate.get("boundary") != _boundary(slate)
        or resource_gate.get("provider_called") is not False
    ):
        raise ValueError("v8 resource gate differs from the live boundary")
    return {
        **dict(resource_gate),
        "schema_version": upgraded_schema,
        "event_admission": admission,
    }


def durable_event_consumptions_v8(
    selected_decisions: Iterable[Mapping],
    terminal_provider_receipts: Iterable[Mapping],
) -> tuple[dict[str, object], ...]:
    """Pair event consumption with an actual terminal provider receipt.

    An embedded admission records what the event gate wanted to do. Only a
    completed provider attempt consumes that event. Resource gates and pure
    no-call decisions therefore retain their evidence without entering the
    durable cooldown set.
    """

    terminal_models: list[tuple[int, dict[str, object]]] = []
    for receipt_index, receipt in enumerate(terminal_provider_receipts):
        status = receipt.get("status")
        model = (
            receipt.get("planning_model")
            if status == "prediction_admitted_uncalibrated"
            else receipt.get("planning_failure")
            if status in {"prediction_failed", "model_abstained"}
            else None
        )
        if not isinstance(model, Mapping):
            raise ValueError("v8 terminal provider receipt lacks its planning result")
        terminal_models.append((receipt_index, dict(model)))

    consumed_receipts: set[int] = set()
    rows: list[dict[str, object]] = []
    consuming_schemas = {
        "runtime_planner_prediction_failure_v1",
        "runtime_bounded_prediction_v7",
    }
    for decision in selected_decisions:
        model = decision.get("planning_model")
        if not isinstance(model, Mapping) or model.get("schema_version") not in consuming_schemas:
            continue
        decision_id = decision.get("decision_id")
        if not isinstance(decision_id, str):
            raise ValueError("v8 provider decision lacks a durable identity")
        admission = model.get("event_admission")
        if (
            model.get("provider_called", True) is not True
            or not isinstance(admission, Mapping)
            or admission.get("decision") != "call_provider"
            or admission.get("provider_called") is not True
        ):
            raise ValueError("v8 provider decision lacks an admitted event")
        matches = [
            receipt_index
            for receipt_index, terminal_model in terminal_models
            if receipt_index not in consumed_receipts and terminal_model == dict(model)
        ]
        if len(matches) != 1:
            raise ValueError("v8 provider decision lacks one matching terminal receipt")
        receipt_index = matches[0]
        consumed_receipts.add(receipt_index)
        new_keys = admission.get("new_event_keys")
        if not isinstance(new_keys, list) or any(
            not isinstance(item, str) or not item for item in new_keys
        ):
            raise ValueError("v8 consumed event keys are invalid")
        rows.append(
            {
                "decision_id": decision_id,
                "terminal_receipt_index": receipt_index,
                "new_event_keys": list(new_keys),
            }
        )
    return tuple(rows)


__all__ = [
    "V8_CUMULATIVE_PLANNER_BUDGET_MS",
    "V8_SEMANTIC_REVIEW_THRESHOLD",
    "bind_event_admission_to_resource_gate_v8",
    "durable_event_consumptions_v8",
    "planner_cumulative_budget_gate_v8",
    "planner_event_admission_v8",
    "planner_time_reserve_gate_v8",
    "validate_planner_event_admission_v8",
]
