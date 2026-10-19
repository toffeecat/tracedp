"""Small hidden-label fault-injection set for the traceback P0."""

from __future__ import annotations

from dataclasses import dataclass

from lh_platform.tracing.types import (
    TraceCaseV1,
    TraceCauseType,
    TraceComponent,
    TraceEventV1,
    TraceInterventionV1,
    TraceOutcomeV1,
)

CONTROLLED_TRACEBACK_DATASET_ID = "controlled_traceback_v3"


@dataclass(frozen=True)
class ControlledTracebackLabelV1:
    case_id: str
    fault_type: str
    causal_event_ids: tuple[str, ...]
    component: TraceComponent
    cause_type: TraceCauseType
    intervention: TraceInterventionV1


@dataclass(frozen=True)
class ControlledTracebackCell:
    case: TraceCaseV1
    label: ControlledTracebackLabelV1


def _case(
    *,
    case_id: str,
    task_id: str,
    events: tuple[TraceEventV1, ...],
    interruption_event_id: str,
    terminal_reason: str,
    fault_type: str,
    causal_event_ids: tuple[str, ...],
    component: TraceComponent,
    target_event_id: str,
    replacement_content: str,
) -> ControlledTracebackCell:
    trace_case = TraceCaseV1(
        case_id=case_id,
        task_id=task_id,
        events=events,
        interruption_event_id=interruption_event_id,
        observed_outcome=TraceOutcomeV1(
            success=False,
            terminal_reason=terminal_reason,
        ),
    )
    return ControlledTracebackCell(
        case=trace_case,
        label=ControlledTracebackLabelV1(
            case_id=case_id,
            fault_type=fault_type,
            causal_event_ids=causal_event_ids,
            component=component,
            cause_type=TraceCauseType(fault_type),
            intervention=TraceInterventionV1(
                target_event_id=target_event_id,
                replacement_content=replacement_content,
            ),
        ),
    )


def controlled_traceback_cells() -> tuple[ControlledTracebackCell, ...]:
    """Return six paired cases while keeping labels outside tracer input."""

    return (
        _case(
            case_id="tool-argument-corruption-001",
            task_id="controlled-calendar-001",
            events=(
                TraceEventV1(
                    "event-0",
                    0,
                    "observation",
                    "The meeting must be created for 2026-07-25.",
                ),
                TraceEventV1(
                    "event-1",
                    1,
                    "action",
                    'calendar.create(date="2026-07-24")',
                    ("event-0",),
                ),
                TraceEventV1(
                    "event-2",
                    2,
                    "tool_result",
                    "Meeting created for 2026-07-24.",
                    ("event-1",),
                ),
                TraceEventV1(
                    "event-3",
                    3,
                    "interruption",
                    "Verifier rejected the final state: wrong meeting date.",
                    ("event-2",),
                ),
            ),
            interruption_event_id="event-3",
            terminal_reason="verifier_rejected",
            fault_type="tool_argument_corruption",
            causal_event_ids=("event-1",),
            component=TraceComponent.ACTOR,
            target_event_id="event-1",
            replacement_content='calendar.create(date="2026-07-25")',
        ),
        _case(
            case_id="observation-staleness-001",
            task_id="controlled-weather-001",
            events=(
                TraceEventV1(
                    "event-0",
                    0,
                    "observation",
                    "Report the current Paris weather; current time is 10:05.",
                ),
                TraceEventV1(
                    "event-1",
                    1,
                    "observation",
                    "Cached weather: 24C and clear (fetched at 08:00).",
                    ("event-0",),
                ),
                TraceEventV1(
                    "event-2",
                    2,
                    "action",
                    "final_answer(Paris is 24C and clear.)",
                    ("event-1",),
                ),
                TraceEventV1(
                    "event-3",
                    3,
                    "interruption",
                    "Verifier rejected stale data; live weather was 18C and rain.",
                    ("event-2",),
                ),
            ),
            interruption_event_id="event-3",
            terminal_reason="stale_observation",
            fault_type="observation_staleness",
            causal_event_ids=("event-1",),
            component=TraceComponent.ENVIRONMENT_OBSERVATION,
            target_event_id="event-1",
            replacement_content="Current weather: 18C and rain (fresh at 10:05).",
        ),
        _case(
            case_id="tool-response-corruption-001",
            task_id="controlled-bank-001",
            events=(
                TraceEventV1(
                    "event-0",
                    0,
                    "observation",
                    "Transfer $80 only if the account balance is sufficient.",
                ),
                TraceEventV1(
                    "event-1",
                    1,
                    "tool_result",
                    "Balance lookup returned $50.",
                    ("event-0",),
                ),
                TraceEventV1(
                    "event-2",
                    2,
                    "tool_result",
                    "Transfer result: accepted; new balance $20.",
                    ("event-1",),
                ),
                TraceEventV1(
                    "event-3",
                    3,
                    "interruption",
                    "Ledger verifier found the transfer executed despite insufficient funds.",
                    ("event-2",),
                ),
            ),
            interruption_event_id="event-3",
            terminal_reason="tool_contract_violation",
            fault_type="tool_response_corruption",
            causal_event_ids=("event-2",),
            component=TraceComponent.TOOL,
            target_event_id="event-2",
            replacement_content="Transfer result: rejected; insufficient funds.",
        ),
        _case(
            case_id="history-memory-deletion-001",
            task_id="controlled-shipping-001",
            events=(
                TraceEventV1(
                    "event-0",
                    0,
                    "observation",
                    "Ship the parcel to 10 Cedar Street after confirming inventory.",
                ),
                TraceEventV1(
                    "event-1",
                    1,
                    "tool_result",
                    "Inventory confirmed: one parcel available.",
                    ("event-0",),
                ),
                TraceEventV1(
                    "event-2",
                    2,
                    "memory",
                    "Context compaction completed; shipping address unavailable.",
                    ("event-1",),
                ),
                TraceEventV1(
                    "event-3",
                    3,
                    "action",
                    "shipping.create(address='unknown')",
                    ("event-2",),
                ),
                TraceEventV1(
                    "event-4",
                    4,
                    "interruption",
                    "Shipping verifier rejected an unknown destination.",
                    ("event-3",),
                ),
            ),
            interruption_event_id="event-4",
            terminal_reason="required_history_missing",
            fault_type="history_memory_deletion",
            causal_event_ids=("event-2",),
            component=TraceComponent.MEMORY,
            target_event_id="event-2",
            replacement_content="Memory restored: ship to 10 Cedar Street.",
        ),
        _case(
            case_id="orchestrator-routing-error-001",
            task_id="controlled-router-001",
            events=(
                TraceEventV1(
                    "event-0",
                    0,
                    "observation",
                    "Look up the weather for Paris.",
                ),
                TraceEventV1(
                    "event-1",
                    1,
                    "action",
                    "request(tool='weather.lookup', city='Paris')",
                    ("event-0",),
                ),
                TraceEventV1(
                    "event-2",
                    2,
                    "orchestrator",
                    "route(tool='calendar.lookup', args={'city':'Paris'})",
                    ("event-1",),
                ),
                TraceEventV1(
                    "event-3",
                    3,
                    "interruption",
                    "Calendar rejected unknown argument city.",
                    ("event-2",),
                ),
            ),
            interruption_event_id="event-3",
            terminal_reason="wrong_tool_routed",
            fault_type="orchestrator_routing_error",
            causal_event_ids=("event-2",),
            component=TraceComponent.ORCHESTRATOR,
            target_event_id="event-2",
            replacement_content="route(tool='weather.lookup', args={'city':'Paris'})",
        ),
        _case(
            case_id="premature-termination-001",
            task_id="controlled-email-001",
            events=(
                TraceEventV1(
                    "event-0",
                    0,
                    "observation",
                    "Create the report and then email ops@example.com.",
                ),
                TraceEventV1(
                    "event-1",
                    1,
                    "tool_result",
                    "Report created successfully at /reports/daily.pdf.",
                    ("event-0",),
                ),
                TraceEventV1(
                    "event-2",
                    2,
                    "action",
                    "final_answer(Report creation complete.)",
                    ("event-1",),
                ),
                TraceEventV1(
                    "event-3",
                    3,
                    "interruption",
                    "Verifier rejected completion: required email was never sent.",
                    ("event-2",),
                ),
            ),
            interruption_event_id="event-3",
            terminal_reason="required_action_omitted",
            fault_type="premature_termination",
            causal_event_ids=("event-2",),
            component=TraceComponent.ACTOR,
            target_event_id="event-2",
            replacement_content="email.send(to='ops@example.com', subject='Report ready')",
        ),
    )
