"""Frozen held-out fault programs for controlled same-checkpoint postmortems."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.adapters import TraceInvestigationError
from lh_platform.tracing.replay_receipts import replay_outcome_sha256, restore_receipt_sha256
from lh_platform.tracing.types import (
    CounterfactualReplayEvidenceV1,
    ReplayOutcomeV1,
    TraceCaseV1,
    TraceCauseType,
    TraceComponent,
    TraceEventV1,
    TraceInterventionV1,
    TraceOutcomeV1,
)

CONTROLLED_SELF_TRACE_DATASET_ID = "controlled_self_trace_heldout_v2"


def _sha256(value: object) -> str:
    return "sha256:" + canonical_sha256(value)


def _copy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(state, sort_keys=True))


@dataclass(frozen=True)
class SelfTraceCaseTemplateV1:
    case_id: str
    task_id: str
    task_prompt: str
    fault_event_kind: str
    fault_event_content: str
    downstream_kind: str
    downstream_content: str
    interruption_content: str
    terminal_reason: str
    decoy_events: tuple[tuple[str, str], ...] = ()
    fault_insert_index: int = 0

    def __post_init__(self) -> None:
        if not self.decoy_events:
            raise ValueError("controlled self-trace cases require at least one decoy event")
        if not 0 <= self.fault_insert_index <= len(self.decoy_events):
            raise ValueError("fault_insert_index must address the decoy event sequence")
        if any(not kind or not content for kind, content in self.decoy_events):
            raise ValueError("controlled self-trace decoy events must be non-empty")

    @property
    def decoy_event_contents(self) -> tuple[str, ...]:
        return tuple(content for _, content in self.decoy_events)

    @property
    def fault_event_id(self) -> str:
        return f"event-{2 + self.fault_insert_index}"

    def actor_messages(self) -> tuple[str, str]:
        return (
            "Act on the task as an execution agent. Return only the next concrete action; "
            "do not diagnose or discuss possible failures.",
            self.task_prompt,
        )

    def materialize(
        self,
        *,
        actor_output: str,
        actor_metadata: Mapping[str, str],
    ) -> TraceCaseV1:
        events: list[TraceEventV1] = [
            TraceEventV1("event-0", 0, "observation", self.task_prompt),
            TraceEventV1(
                "event-1",
                1,
                "actor",
                actor_output,
                ("event-0",),
                tuple(sorted(actor_metadata.items())),
            ),
        ]
        staged_events = list(self.decoy_events)
        staged_events.insert(
            self.fault_insert_index,
            (self.fault_event_kind, self.fault_event_content),
        )
        staged_events.append((self.downstream_kind, self.downstream_content))
        parent_event_id = "event-1"
        for kind, content in staged_events:
            sequence = len(events)
            event_id = f"event-{sequence}"
            events.append(
                TraceEventV1(
                    event_id,
                    sequence,
                    kind,
                    content,
                    (parent_event_id,),
                )
            )
            parent_event_id = event_id
        interruption_event_id = f"event-{len(events)}"
        events.append(
            TraceEventV1(
                interruption_event_id,
                len(events),
                "interruption",
                self.interruption_content,
                (parent_event_id,),
            )
        )
        return TraceCaseV1(
            case_id=self.case_id,
            task_id=self.task_id,
            events=tuple(events),
            interruption_event_id=interruption_event_id,
            observed_outcome=TraceOutcomeV1(
                success=False,
                terminal_reason=self.terminal_reason,
            ),
        )


@dataclass(frozen=True)
class SelfTraceHiddenLabelV1:
    case_id: str
    fault_type: TraceCauseType
    causal_event_ids: tuple[str, ...]
    component: TraceComponent
    intervention: TraceInterventionV1


@dataclass(frozen=True)
class SelfTraceReplayProgramV1:
    sham_content: str
    source_snapshot: Mapping[str, Any]
    execute: Callable[[dict[str, Any], str], None]
    verify: Callable[[dict[str, Any]], bool]
    target_event_id: str = "event-2"


@dataclass(frozen=True)
class ControlledSelfTraceCell:
    template: SelfTraceCaseTemplateV1
    label: SelfTraceHiddenLabelV1
    replay_program: SelfTraceReplayProgramV1

    @property
    def case_id(self) -> str:
        return self.template.case_id


def _calendar_execute(state: dict[str, Any], content: str) -> None:
    match = re.search(r"calendar\.create\(\s*date\s*=\s*['\"]([^'\"]+)['\"]", content)
    state["transition_valid"] = match is not None
    state["created_date"] = match.group(1) if match else None


def _weather_execute(state: dict[str, Any], content: str) -> None:
    temperature = re.search(r"(-?\d+)\s*c\b", content, flags=re.IGNORECASE)
    condition = re.search(r"\b(rain|clear|cloudy)\b", content, flags=re.IGNORECASE)
    observed_at = re.search(
        r"(?:fresh|fetched)(?:\s+at)?\s+(\d{1,2}:\d{2})",
        content,
        flags=re.IGNORECASE,
    )
    state["transition_valid"] = all((temperature, condition, observed_at))
    if state["transition_valid"]:
        state["temperature_c"] = int(temperature.group(1))
        state["condition"] = condition.group(1).lower()
        state["observed_at"] = observed_at.group(1)


def _bank_execute(state: dict[str, Any], content: str) -> None:
    lowered = content.lower()
    accepted = "accepted" in lowered
    rejected = "rejected" in lowered
    state["transition_valid"] = accepted ^ rejected
    if state["transition_valid"]:
        state["transfer_executed"] = accepted


def _memory_execute(state: dict[str, Any], content: str) -> None:
    lowered = content.lower()
    if "address unavailable" in lowered:
        state["transition_valid"] = True
        state["address"] = None
        return
    address = re.search(r"\b(\d+\s+birch\s+road)\b", content, flags=re.IGNORECASE)
    state["transition_valid"] = address is not None
    state["address"] = address.group(1).title() if address else None


def _router_execute(state: dict[str, Any], content: str) -> None:
    tool = re.search(r"tool\s*=\s*['\"]([^'\"]+)['\"]", content)
    symbol = re.search(r"['\"]symbol['\"]\s*:\s*['\"]([^'\"]+)['\"]", content)
    state["transition_valid"] = tool is not None and symbol is not None
    state["routed_tool"] = tool.group(1) if tool else None
    state["symbol"] = symbol.group(1) if symbol else None


def _email_execute(state: dict[str, Any], content: str) -> None:
    is_email_call = bool(re.search(r"\bemail(?:\.send)?\s*\(", content, flags=re.IGNORECASE))
    recipient = re.search(r"(?:to\s*=\s*['\"])?([\w.+-]+@[\w.-]+)", content)
    state["transition_valid"] = is_email_call and recipient is not None
    state["email_sent_to"] = recipient.group(1).lower() if recipient else None


def controlled_self_trace_cells() -> tuple[ControlledSelfTraceCell, ...]:
    """Return six task-held-out cells; hidden labels never enter model prompts."""

    rows = (
        (
            SelfTraceCaseTemplateV1(
                case_id="self-trace-tool-argument-heldout-001",
                task_id="heldout-calendar-august-001",
                task_prompt="Create the release meeting for 2026-08-02.",
                fault_event_kind="orchestrator",
                fault_event_content='calendar.create(date="2026-08-01")',
                downstream_kind="tool_result",
                downstream_content="Meeting created for 2026-08-01.",
                interruption_content="Verifier rejected the final state: release meeting date is wrong.",
                terminal_reason="verifier_rejected",
                decoy_events=(
                    (
                        "observation",
                        "Directory warning: attendee lookup unavailable; no attendees were requested.",
                    ),
                    (
                        "memory",
                        "Prior room preference is unavailable; the task has no room constraint.",
                    ),
                ),
                fault_insert_index=1,
            ),
            TraceCauseType.TOOL_ARGUMENT_CORRUPTION,
            TraceComponent.ORCHESTRATOR,
            'calendar.create(date="2026-08-02")',
            SelfTraceReplayProgramV1(
                sham_content='calendar.create(date="2026-08-03")',
                source_snapshot={"required_date": "2026-08-02", "created_date": None},
                execute=_calendar_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("created_date") == state.get("required_date")
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="self-trace-observation-staleness-heldout-001",
                task_id="heldout-weather-tokyo-001",
                task_prompt="Report current Tokyo weather at 15:30.",
                fault_event_kind="environment_observation",
                fault_event_content="Cached Tokyo weather: 27C and clear (fetched at 12:00).",
                downstream_kind="action",
                downstream_content="final_answer(Tokyo is 27C and clear.)",
                interruption_content="Verifier rejected stale data; live weather was 21C and cloudy.",
                terminal_reason="stale_observation",
                decoy_events=(
                    (
                        "tool_result",
                        "Weather provider latency warning recovered before this response.",
                    ),
                    ("memory", "Display unit preference loaded: Celsius."),
                ),
                fault_insert_index=0,
            ),
            TraceCauseType.OBSERVATION_STALENESS,
            TraceComponent.ENVIRONMENT_OBSERVATION,
            "Current Tokyo weather: 21C and cloudy (fresh at 15:30).",
            SelfTraceReplayProgramV1(
                sham_content="Current Tokyo weather: 27C and clear (fresh at 15:30).",
                source_snapshot={
                    "required_temperature_c": 21,
                    "required_condition": "cloudy",
                    "required_observed_at": "15:30",
                },
                execute=_weather_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("temperature_c") == state.get("required_temperature_c")
                    and state.get("condition") == state.get("required_condition")
                    and state.get("observed_at") == state.get("required_observed_at")
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="self-trace-tool-response-heldout-001",
                task_id="heldout-bank-transfer-001",
                task_prompt="Transfer $60 only if the account balance of $40 is sufficient.",
                fault_event_kind="tool_result",
                fault_event_content="Transfer result: accepted despite insufficient funds.",
                downstream_kind="orchestrator",
                downstream_content="Runtime committed the accepted transfer response.",
                interruption_content="Ledger verifier found an overdraft caused by the accepted transfer.",
                terminal_reason="tool_contract_violation",
                decoy_events=(
                    ("observation", "Transfer memo is blank; the memo field is optional."),
                    ("memory", "Fraud-screen status is clear for this account."),
                ),
                fault_insert_index=2,
            ),
            TraceCauseType.TOOL_RESPONSE_CORRUPTION,
            TraceComponent.TOOL,
            "Transfer result: rejected; insufficient funds.",
            SelfTraceReplayProgramV1(
                sham_content="Transfer result: accepted; override enabled.",
                source_snapshot={"balance": 40, "required_amount": 60, "transfer_executed": False},
                execute=_bank_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("transfer_executed") is False
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="self-trace-memory-deletion-heldout-001",
                task_id="heldout-shipping-birch-001",
                task_prompt="Ship the parcel to 44 Birch Road after inventory confirmation.",
                fault_event_kind="memory",
                fault_event_content="Context compaction completed; shipping address unavailable.",
                downstream_kind="action",
                downstream_content="shipping.create(address='unknown')",
                interruption_content="Shipping verifier rejected the unknown destination.",
                terminal_reason="required_history_missing",
                decoy_events=(
                    ("tool_result", "Inventory confirmation succeeded for the parcel."),
                    (
                        "observation",
                        "Packaging color is unavailable; no packaging color was requested.",
                    ),
                ),
                fault_insert_index=1,
            ),
            TraceCauseType.HISTORY_MEMORY_DELETION,
            TraceComponent.MEMORY,
            "Memory restored: ship to 44 Birch Road.",
            SelfTraceReplayProgramV1(
                sham_content="Memory restored: ship to 19 Oak Lane.",
                source_snapshot={"required_address": "44 Birch Road", "address": "44 Birch Road"},
                execute=_memory_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("address") == state.get("required_address")
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="self-trace-routing-heldout-001",
                task_id="heldout-stock-aapl-001",
                task_prompt="Look up the current AAPL stock quote.",
                fault_event_kind="orchestrator",
                fault_event_content="route(tool='currency.lookup', args={'symbol':'AAPL'})",
                downstream_kind="tool_result",
                downstream_content="Currency tool rejected unknown symbol AAPL.",
                interruption_content="Execution interrupted because the wrong tool received the request.",
                terminal_reason="wrong_tool_routed",
                decoy_events=(
                    ("memory", "A previous MSFT quote expired; the current request is AAPL."),
                    ("observation", "News-service warning: headlines are temporarily unavailable."),
                ),
                fault_insert_index=0,
            ),
            TraceCauseType.ORCHESTRATOR_ROUTING_ERROR,
            TraceComponent.ORCHESTRATOR,
            "route(tool='stock.lookup', args={'symbol':'AAPL'})",
            SelfTraceReplayProgramV1(
                sham_content="route(tool='news.lookup', args={'symbol':'AAPL'})",
                source_snapshot={"required_tool": "stock.lookup", "required_symbol": "AAPL"},
                execute=_router_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("routed_tool") == state.get("required_tool")
                    and state.get("symbol") == state.get("required_symbol")
                ),
            ),
        ),
        (
            SelfTraceCaseTemplateV1(
                case_id="self-trace-premature-termination-heldout-001",
                task_id="heldout-finance-report-001",
                task_prompt="Create the budget report and then email finance@example.com.",
                fault_event_kind="orchestrator",
                fault_event_content="Runtime terminated after report creation; notification step omitted.",
                downstream_kind="action",
                downstream_content="final_answer(Budget report created.)",
                interruption_content="Verifier rejected completion because finance was never emailed.",
                terminal_reason="required_action_omitted",
                decoy_events=(
                    ("tool_result", "Budget report file was created successfully."),
                    (
                        "observation",
                        "Printer service is offline; printing was not requested.",
                    ),
                ),
                fault_insert_index=2,
            ),
            TraceCauseType.PREMATURE_TERMINATION,
            TraceComponent.ORCHESTRATOR,
            "email.send(to='finance@example.com', subject='Budget report ready')",
            SelfTraceReplayProgramV1(
                sham_content="email.send(to='sales@example.com', subject='Budget report ready')",
                source_snapshot={
                    "report_created": True,
                    "required_recipient": "finance@example.com",
                },
                execute=_email_execute,
                verify=lambda state: (
                    state.get("transition_valid") is True
                    and state.get("report_created") is True
                    and state.get("email_sent_to") == state.get("required_recipient")
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


def controlled_self_trace_dataset_manifest_from_cells(
    *,
    schema: str,
    dataset_id: str,
    split: str,
    cells: tuple[ControlledSelfTraceCell, ...],
    replay_controller_id: str | None,
    preregistration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one frozen dataset without exposing executable callables."""

    manifest: dict[str, Any] = {
        "schema": schema,
        "dataset_id": dataset_id,
        "split": split,
        "cases": [
            {
                "template": {
                    "case_id": cell.template.case_id,
                    "task_id": cell.template.task_id,
                    "task_prompt": cell.template.task_prompt,
                    "fault_event_kind": cell.template.fault_event_kind,
                    "fault_event_content": cell.template.fault_event_content,
                    "downstream_kind": cell.template.downstream_kind,
                    "downstream_content": cell.template.downstream_content,
                    "interruption_content": cell.template.interruption_content,
                    "terminal_reason": cell.template.terminal_reason,
                    "decoy_events": [
                        {"kind": kind, "content": content}
                        for kind, content in cell.template.decoy_events
                    ],
                    "fault_insert_index": cell.template.fault_insert_index,
                },
                "hidden_label": {
                    "fault_type": cell.label.fault_type.value,
                    "causal_event_ids": list(cell.label.causal_event_ids),
                    "component": cell.label.component.value,
                    "intervention": cell.label.intervention.to_dict(),
                },
                "replay": {
                    "target_event_id": cell.replay_program.target_event_id,
                    "sham_content": cell.replay_program.sham_content,
                    "source_snapshot": dict(cell.replay_program.source_snapshot),
                },
            }
            for cell in cells
        ],
    }
    if replay_controller_id is not None:
        manifest["replay_controller_id"] = replay_controller_id
    if preregistration is not None:
        manifest["preregistration"] = dict(preregistration)
    return manifest


def controlled_self_trace_dataset_manifest() -> dict[str, Any]:
    """Return the frozen, hashable case/label/replay contract used by P1."""

    return controlled_self_trace_dataset_manifest_from_cells(
        schema="lh_controlled_self_trace_dataset/v2",
        dataset_id=CONTROLLED_SELF_TRACE_DATASET_ID,
        split="heldout_from_controlled_traceback_v3_development_cases",
        cells=controlled_self_trace_cells(),
        replay_controller_id=None,
    )


class SelfTraceReplayController:
    """Execute baseline, proposed repair, and matched sham for one materialized trace."""

    def __init__(
        self,
        cell: ControlledSelfTraceCell,
        source_trace: TraceCaseV1,
        *,
        controller_id: str = "controlled_self_trace_executable_replay/v2",
    ) -> None:
        if source_trace.case_id != cell.case_id:
            raise ValueError("self-trace replay case mismatch")
        if not controller_id:
            raise ValueError("self-trace replay controller_id must be non-empty")
        self._cell = cell
        self._source_trace_sha256 = source_trace.sha256
        self.controller_id = controller_id

    def _run_arm(
        self,
        *,
        arm: str,
        content: str,
        target_event_id: str,
        source_snapshot_sha256: str,
        original_target_content: str,
    ) -> tuple[ReplayOutcomeV1, str]:
        program = self._cell.replay_program
        state = _copy_state(program.source_snapshot)
        restored_state_sha256 = _sha256(state)
        effective_content = content
        if target_event_id != program.target_event_id:
            effective_content = original_target_content
        program.execute(state, effective_content)
        success = program.verify(state)
        unhashed = ReplayOutcomeV1(
            success=success,
            terminal_reason="verifier_accepted" if success else "verifier_rejected",
            target_event_id=target_event_id,
            input_content_sha256=_sha256(content),
            restored_state_sha256=restored_state_sha256,
            terminal_state_sha256=_sha256(state),
            outcome_receipt_sha256="",
        )
        return (
            ReplayOutcomeV1(
                success=unhashed.success,
                terminal_reason=unhashed.terminal_reason,
                target_event_id=unhashed.target_event_id,
                input_content_sha256=unhashed.input_content_sha256,
                restored_state_sha256=unhashed.restored_state_sha256,
                terminal_state_sha256=unhashed.terminal_state_sha256,
                outcome_receipt_sha256=replay_outcome_sha256(
                    controller_id=self.controller_id,
                    case_id=self._cell.case_id,
                    arm=arm,
                    source_snapshot_sha256=source_snapshot_sha256,
                    outcome=unhashed,
                ),
            ),
            restored_state_sha256,
        )

    def evaluate(
        self,
        case: TraceCaseV1,
        intervention: TraceInterventionV1,
    ) -> CounterfactualReplayEvidenceV1:
        if case.case_id != self._cell.case_id or case.sha256 != self._source_trace_sha256:
            raise TraceInvestigationError("self-trace replay frozen trace mismatch")
        program = self._cell.replay_program
        event_by_id = {event.event_id: event for event in case.events}
        try:
            original = event_by_id[program.target_event_id].content
        except KeyError as error:
            raise TraceInvestigationError("self-trace replay target missing") from error
        snapshot_sha256 = _sha256(program.source_snapshot)
        baseline, baseline_restore = self._run_arm(
            arm="fault_baseline",
            content=original,
            target_event_id=program.target_event_id,
            source_snapshot_sha256=snapshot_sha256,
            original_target_content=original,
        )
        repair, repair_restore = self._run_arm(
            arm="predicted_repair",
            content=intervention.replacement_content,
            target_event_id=intervention.target_event_id,
            source_snapshot_sha256=snapshot_sha256,
            original_target_content=original,
        )
        sham, sham_restore = self._run_arm(
            arm="matched_sham",
            content=program.sham_content,
            target_event_id=program.target_event_id,
            source_snapshot_sha256=snapshot_sha256,
            original_target_content=original,
        )
        unhashed = CounterfactualReplayEvidenceV1(
            controller_id=self.controller_id,
            source_trace_sha256=case.sha256,
            source_snapshot_sha256=snapshot_sha256,
            restore_verified=all(
                value == snapshot_sha256
                for value in (baseline_restore, repair_restore, sham_restore)
            ),
            restore_receipt_sha256="",
            baseline=baseline,
            repair=repair,
            sham=sham,
            replay_calls=3,
        )
        return CounterfactualReplayEvidenceV1(
            controller_id=unhashed.controller_id,
            source_trace_sha256=unhashed.source_trace_sha256,
            source_snapshot_sha256=unhashed.source_snapshot_sha256,
            restore_verified=unhashed.restore_verified,
            restore_receipt_sha256=restore_receipt_sha256(case=case, evidence=unhashed),
            baseline=unhashed.baseline,
            repair=unhashed.repair,
            sham=unhashed.sham,
            replay_calls=unhashed.replay_calls,
        )
