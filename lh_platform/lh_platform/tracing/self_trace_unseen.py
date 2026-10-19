"""Preregistered unseen controlled tasks for same-checkpoint postmortems."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from lh_platform.tracing.self_trace_controlled import (
    ControlledSelfTraceCell,
    SelfTraceCaseTemplateV1,
    SelfTraceHiddenLabelV1,
    SelfTraceReplayProgramV1,
    controlled_self_trace_dataset_manifest_from_cells,
)
from lh_platform.tracing.types import (
    TraceCauseType,
    TraceComponent,
    TraceInterventionV1,
)

CONTROLLED_SELF_TRACE_UNSEEN_DATASET_ID = "controlled_self_trace_unseen_v3"
CONTROLLED_SELF_TRACE_UNSEEN_REPLAY_CONTROLLER_ID = (
    "controlled_self_trace_executable_replay/v3"
)


def _hotel_execute(state: dict[str, Any], content: str) -> None:
    check_in = re.search(r"check_in\s*=\s*['\"]([^'\"]+)['\"]", content)
    check_out = re.search(r"check_out\s*=\s*['\"]([^'\"]+)['\"]", content)
    state["transition_valid"] = check_in is not None and check_out is not None
    state["check_in"] = check_in.group(1) if check_in else None
    state["check_out"] = check_out.group(1) if check_out else None


def _train_execute(state: dict[str, Any], content: str) -> None:
    train = re.search(r"\b(IC\d+)\b", content, flags=re.IGNORECASE)
    platform = re.search(r"platform\s+(\d+)", content, flags=re.IGNORECASE)
    observed_at = re.search(r"observed at\s+(\d{1,2}:\d{2})", content, flags=re.IGNORECASE)
    state["transition_valid"] = all((train, platform, observed_at))
    state["train"] = train.group(1).upper() if train else None
    state["platform"] = platform.group(1) if platform else None
    state["observed_at"] = observed_at.group(1) if observed_at else None


def _inventory_execute(state: dict[str, Any], content: str) -> None:
    lowered = content.lower()
    accepted = "accepted" in lowered
    rejected = "rejected" in lowered
    available = re.search(r"available_units\s*=\s*(\d+)", content)
    reserved = re.search(r"reserved_units\s*=\s*(\d+)", content)
    state["transition_valid"] = (accepted ^ rejected) and available is not None and reserved is not None
    state["reservation_created"] = accepted if state["transition_valid"] else None
    state["available_units"] = int(available.group(1)) if available else None
    state["reserved_units"] = int(reserved.group(1)) if reserved else None


def _phone_execute(state: dict[str, Any], content: str) -> None:
    if "phone unavailable" in content.lower():
        state["transition_valid"] = True
        state["phone_digits"] = None
        return
    match = re.search(r"(?:\+?\d[\d\s-]{8,}\d)", content)
    state["transition_valid"] = match is not None
    state["phone_digits"] = re.sub(r"\D", "", match.group(0)) if match else None


def _timezone_router_execute(state: dict[str, Any], content: str) -> None:
    tool = re.search(r"tool\s*=\s*['\"]([^'\"]+)['\"]", content)
    request = re.search(r"['\"]request['\"]\s*:\s*['\"]([^'\"]+)['\"]", content)
    state["transition_valid"] = tool is not None and request is not None
    state["routed_tool"] = tool.group(1) if tool else None
    state["request"] = request.group(1) if request else None


def _share_execute(state: dict[str, Any], content: str) -> None:
    is_share = bool(re.search(r"\bshare\.send\s*\(", content, flags=re.IGNORECASE))
    recipient = re.search(r"to\s*=\s*['\"]([^'\"]+)['\"]", content)
    filename = re.search(r"file\s*=\s*['\"]([^'\"]+)['\"]", content)
    state["transition_valid"] = is_share and recipient is not None and filename is not None
    state["shared_to"] = recipient.group(1).lower() if recipient else None
    state["shared_file"] = filename.group(1) if filename else None


def controlled_self_trace_unseen_cells() -> tuple[ControlledSelfTraceCell, ...]:
    """Return six frozen tasks that are disjoint from the v2 development set."""

    rows = (
        (
            SelfTraceCaseTemplateV1(
                case_id="unseen-a17",
                task_id="unseen-hotel-november-017",
                task_prompt=(
                    "Reserve a hotel room with check-in on 2026-11-14 and check-out on "
                    "2026-11-16."
                ),
                fault_event_kind="orchestrator",
                fault_event_content=(
                    'hotel.reserve(check_in="2026-11-13", check_out="2026-11-16")'
                ),
                downstream_kind="tool_result",
                downstream_content="Reservation confirmed for 2026-11-13 through 2026-11-16.",
                interruption_content="Verifier rejected the reservation because check-in is one day early.",
                terminal_reason="verifier_rejected",
                decoy_events=(
                    ("tool_result", "Loyalty lookup timed out, then recovered without changing the rate."),
                    ("memory", "Bed preference is unavailable; no bed preference was requested."),
                ),
                fault_insert_index=0,
            ),
            TraceCauseType.TOOL_ARGUMENT_CORRUPTION,
            TraceComponent.ORCHESTRATOR,
            'hotel.reserve(check_in="2026-11-14", check_out="2026-11-16")',
            SelfTraceReplayProgramV1(
                sham_content=(
                    'hotel.reserve(check_in="2026-11-14", check_out="2026-11-17")'
                ),
                source_snapshot={
                    "required_check_in": "2026-11-14",
                    "required_check_out": "2026-11-16",
                },
                execute=_hotel_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("check_in") == state.get("required_check_in")
                    and state.get("check_out") == state.get("required_check_out")
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="unseen-b29",
                task_id="unseen-train-ic742-029",
                task_prompt="At 18:20, report the live departure platform for train IC742.",
                fault_event_kind="environment_observation",
                fault_event_content="Cached departure board: IC742 platform 5 (observed at 17:40).",
                downstream_kind="action",
                downstream_content="final_answer(IC742 departs from platform 5.)",
                interruption_content="Verifier rejected the stale platform; the live board shows platform 7.",
                terminal_reason="stale_observation",
                decoy_events=(
                    ("memory", "Ticket status loaded: valid for IC742."),
                    ("tool_result", "Station cafe availability lookup failed; it is unrelated."),
                ),
                fault_insert_index=1,
            ),
            TraceCauseType.OBSERVATION_STALENESS,
            TraceComponent.ENVIRONMENT_OBSERVATION,
            "Live departure board: IC742 platform 7 (observed at 18:20).",
            SelfTraceReplayProgramV1(
                sham_content="Live departure board: IC742 platform 6 (observed at 18:20).",
                source_snapshot={
                    "required_train": "IC742",
                    "required_platform": "7",
                    "required_observed_at": "18:20",
                },
                execute=_train_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("train") == state.get("required_train")
                    and state.get("platform") == state.get("required_platform")
                    and state.get("observed_at") == state.get("required_observed_at")
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="unseen-c41",
                task_id="unseen-inventory-k19-041",
                task_prompt="Reserve 3 units of SKU K-19 only if at least 3 units are available.",
                fault_event_kind="tool_result",
                fault_event_content=(
                    "inventory.reserve result: accepted; available_units=0; reserved_units=3."
                ),
                downstream_kind="orchestrator",
                downstream_content="Runtime committed the accepted inventory reservation.",
                interruption_content="Inventory verifier found a reservation with no available stock.",
                terminal_reason="tool_contract_violation",
                decoy_events=(
                    ("observation", "Price cache expired; price is irrelevant to the stock condition."),
                    ("memory", "Preferred warehouse loaded: west; it also has zero units."),
                ),
                fault_insert_index=2,
            ),
            TraceCauseType.TOOL_RESPONSE_CORRUPTION,
            TraceComponent.TOOL,
            "inventory.reserve result: rejected; available_units=0; reserved_units=0.",
            SelfTraceReplayProgramV1(
                sham_content=(
                    "inventory.reserve result: accepted; available_units=0; reserved_units=1."
                ),
                source_snapshot={
                    "required_available_units": 0,
                    "required_reserved_units": 0,
                    "reservation_created": False,
                },
                execute=_inventory_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("available_units") == state.get("required_available_units")
                    and state.get("reserved_units") == state.get("required_reserved_units")
                    and state.get("reservation_created") is False
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="unseen-d53",
                task_id="unseen-courier-phone-053",
                task_prompt=(
                    "After pickup confirmation, notify the courier contact at +44 20 7946 0958."
                ),
                fault_event_kind="memory",
                fault_event_content="Context compaction completed; courier phone unavailable.",
                downstream_kind="action",
                downstream_content="courier.notify(phone='unknown')",
                interruption_content="Verifier rejected dispatch because the courier contact was not notified.",
                terminal_reason="required_history_missing",
                decoy_events=(
                    ("tool_result", "Pickup confirmation succeeded for parcel CX-81."),
                    ("observation", "Traffic route loaded successfully; no route change is needed."),
                ),
                fault_insert_index=0,
            ),
            TraceCauseType.HISTORY_MEMORY_DELETION,
            TraceComponent.MEMORY,
            "Memory restored: courier phone +44 20 7946 0958.",
            SelfTraceReplayProgramV1(
                sham_content="Memory restored: courier phone +44 20 7946 0959.",
                source_snapshot={"required_phone_digits": "442079460958"},
                execute=_phone_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("phone_digits") == state.get("required_phone_digits")
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="unseen-e67",
                task_id="unseen-timezone-paris-seoul-067",
                task_prompt="Convert 09:00 in Europe/Paris to Asia/Seoul.",
                fault_event_kind="orchestrator",
                fault_event_content=(
                    "route(tool='calculator.divide', args={'request':'09:00 Europe/Paris to "
                    "Asia/Seoul'})"
                ),
                downstream_kind="tool_result",
                downstream_content="Calculator rejected the timezone identifiers.",
                interruption_content="Execution stopped because a non-timezone tool received the request.",
                terminal_reason="wrong_tool_routed",
                decoy_events=(
                    ("memory", "Calendar locale loaded: en-GB; it does not select the tool."),
                    ("observation", "Timezone database warning recovered before routing."),
                ),
                fault_insert_index=1,
            ),
            TraceCauseType.ORCHESTRATOR_ROUTING_ERROR,
            TraceComponent.ORCHESTRATOR,
            (
                "route(tool='timezone.convert', args={'request':'09:00 Europe/Paris to "
                "Asia/Seoul'})"
            ),
            SelfTraceReplayProgramV1(
                sham_content=(
                    "route(tool='calendar.convert', args={'request':'09:00 Europe/Paris to "
                    "Asia/Seoul'})"
                ),
                source_snapshot={
                    "required_tool": "timezone.convert",
                    "required_request": "09:00 Europe/Paris to Asia/Seoul",
                },
                execute=_timezone_router_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("routed_tool") == state.get("required_tool")
                    and state.get("request") == state.get("required_request")
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="unseen-f79",
                task_id="unseen-upload-share-079",
                task_prompt="Upload audit.pdf and send its share link to reviewer@example.com.",
                fault_event_kind="orchestrator",
                fault_event_content="Runtime terminated after upload; share notification was omitted.",
                downstream_kind="action",
                downstream_content="final_answer(audit.pdf uploaded.)",
                interruption_content="Verifier rejected completion because no share link was sent.",
                terminal_reason="required_action_omitted",
                decoy_events=(
                    ("tool_result", "Upload completed successfully for audit.pdf."),
                    ("observation", "Virus scan completed successfully; the file is safe."),
                ),
                fault_insert_index=2,
            ),
            TraceCauseType.PREMATURE_TERMINATION,
            TraceComponent.ORCHESTRATOR,
            "share.send(to='reviewer@example.com', file='audit.pdf')",
            SelfTraceReplayProgramV1(
                sham_content="share.send(to='archive@example.com', file='audit.pdf')",
                source_snapshot={
                    "file_uploaded": True,
                    "required_recipient": "reviewer@example.com",
                    "required_file": "audit.pdf",
                },
                execute=_share_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("file_uploaded") is True
                    and state.get("shared_to") == state.get("required_recipient")
                    and state.get("shared_file") == state.get("required_file")
                ),
            ),
        ),
    )
    return tuple(
        ControlledSelfTraceCell(
            template=template,
            label=SelfTraceHiddenLabelV1(
                case_id=template.case_id,
                fault_type=cause_type,
                causal_event_ids=(template.fault_event_id,),
                component=component,
                intervention=TraceInterventionV1(
                    target_event_id=template.fault_event_id,
                    replacement_content=replacement,
                ),
            ),
            replay_program=replace(program, target_event_id=template.fault_event_id),
        )
        for template, cause_type, component, replacement, program in rows
    )


def controlled_self_trace_unseen_dataset_manifest() -> dict[str, Any]:
    """Return the preregistered v3 contract frozen before any v3 model run."""

    return controlled_self_trace_dataset_manifest_from_cells(
        schema="lh_controlled_self_trace_dataset/v3",
        dataset_id=CONTROLLED_SELF_TRACE_UNSEEN_DATASET_ID,
        split="preregistered_unseen_task_holdout",
        cells=controlled_self_trace_unseen_cells(),
        replay_controller_id=CONTROLLED_SELF_TRACE_UNSEEN_REPLAY_CONTROLLER_ID,
        preregistration={
            "frozen_on": "2026-07-24",
            "frozen_before_model_execution": True,
            "prior_development_dataset_ids": ["controlled_self_trace_heldout_v2"],
            "task_overlap_with_prior": 0,
            "adaptive_changes_after_first_model_output_forbidden": True,
        },
    )


__all__ = [
    "CONTROLLED_SELF_TRACE_UNSEEN_DATASET_ID",
    "CONTROLLED_SELF_TRACE_UNSEEN_REPLAY_CONTROLLER_ID",
    "controlled_self_trace_unseen_cells",
    "controlled_self_trace_unseen_dataset_manifest",
]
