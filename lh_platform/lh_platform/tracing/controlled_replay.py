"""Executable deterministic replay programs for the controlled traceback set."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.adapters import TraceInvestigationError
from lh_platform.tracing.controlled import controlled_traceback_cells
from lh_platform.tracing.replay_receipts import (
    replay_outcome_sha256,
    restore_receipt_sha256,
)
from lh_platform.tracing.types import (
    CounterfactualReplayEvidenceV1,
    ReplayOutcomeV1,
    TraceCaseV1,
    TraceInterventionV1,
)


def _sha256(value: object) -> str:
    return "sha256:" + canonical_sha256(value)


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(state, sort_keys=True))


@dataclass(frozen=True)
class _ReplayProgram:
    target_event_id: str
    sham_content: str
    source_snapshot: dict[str, Any]
    execute: Callable[[dict[str, Any], str], None]
    verify: Callable[[dict[str, Any]], bool]


def _calendar_execute(state: dict[str, Any], content: str) -> None:
    match = re.search(r"calendar\.create\(\s*date\s*=\s*['\"]([^'\"]+)['\"]", content)
    state["transition_valid"] = match is not None
    state["created_date"] = match.group(1) if match else None


def _weather_execute(state: dict[str, Any], content: str) -> None:
    temperature = re.search(r"(-?\d+)\s*c\b", content, flags=re.IGNORECASE)
    condition = re.search(r"\b(rain|clear)\b", content, flags=re.IGNORECASE)
    observed_at = re.search(
        r"(?:fresh|fetched)(?:\s+at)?\s+(\d{1,2}:\d{2})",
        content,
        flags=re.IGNORECASE,
    )
    valid = temperature is not None and condition is not None and observed_at is not None
    state["transition_valid"] = valid
    if valid:
        state["temperature_c"] = int(temperature.group(1))
        state["condition"] = condition.group(1).lower()
        state["observed_at"] = observed_at.group(1)


def _bank_execute(state: dict[str, Any], content: str) -> None:
    lowered = content.lower()
    accepted = "accepted" in lowered
    rejected = "rejected" in lowered
    state["transition_valid"] = accepted ^ rejected
    if not state["transition_valid"]:
        return
    state["transfer_executed"] = accepted
    balance = re.search(r"(?:new\s+)?balance\s+\$?(\d+)", content, flags=re.IGNORECASE)
    if balance:
        state["balance"] = int(balance.group(1))


def _memory_execute(state: dict[str, Any], content: str) -> None:
    lowered = content.lower()
    if "address unavailable" in lowered:
        state["transition_valid"] = True
        state["address"] = None
        return
    address = re.search(r"\b(\d+\s+cedar\s+street)\b", content, flags=re.IGNORECASE)
    state["transition_valid"] = address is not None
    state["address"] = address.group(1).title() if address else None


def _router_execute(state: dict[str, Any], content: str) -> None:
    tool = re.search(r"tool\s*=\s*['\"]([^'\"]+)['\"]", content)
    city = re.search(r"['\"]city['\"]\s*:\s*['\"]([^'\"]+)['\"]", content)
    state["transition_valid"] = tool is not None and city is not None
    state["routed_tool"] = tool.group(1) if tool else None
    state["city"] = city.group(1) if city else None


def _email_execute(state: dict[str, Any], content: str) -> None:
    is_email_call = bool(re.search(r"\bemail(?:\.send)?\s*\(", content, flags=re.IGNORECASE))
    recipient = re.search(r"(?:to\s*=\s*['\"])?([\w.+-]+@[\w.-]+)", content)
    state["transition_valid"] = is_email_call and recipient is not None
    state["email_sent_to"] = recipient.group(1).lower() if recipient else None


def _programs() -> dict[str, _ReplayProgram]:
    return {
        "tool-argument-corruption-001": _ReplayProgram(
            target_event_id="event-1",
            sham_content='calendar.create(date="2026-07-26")',
            source_snapshot={
                "required_date": "2026-07-25",
                "created_date": None,
                "transition_valid": True,
            },
            execute=_calendar_execute,
            verify=lambda state: (
                state["transition_valid"] and state["created_date"] == state["required_date"]
            ),
        ),
        "observation-staleness-001": _ReplayProgram(
            target_event_id="event-1",
            sham_content="Current weather: 24C and clear (fresh at 10:05).",
            source_snapshot={
                "required_temperature_c": 18,
                "required_condition": "rain",
                "required_observed_at": "10:05",
                "temperature_c": None,
                "condition": None,
                "observed_at": None,
                "transition_valid": True,
            },
            execute=_weather_execute,
            verify=lambda state: (
                state["transition_valid"]
                and state["temperature_c"] == state["required_temperature_c"]
                and state["condition"] == state["required_condition"]
                and state["observed_at"] == state["required_observed_at"]
            ),
        ),
        "tool-response-corruption-001": _ReplayProgram(
            target_event_id="event-2",
            sham_content="Transfer result: accepted; new balance $10.",
            source_snapshot={
                "balance": 50,
                "required_amount": 80,
                "transfer_executed": False,
                "transition_valid": True,
            },
            execute=_bank_execute,
            verify=lambda state: (
                state["transition_valid"]
                and state["balance"] == 50
                and state["transfer_executed"] is False
            ),
        ),
        "history-memory-deletion-001": _ReplayProgram(
            target_event_id="event-2",
            sham_content="Memory restored: ship to 99 Pine Avenue.",
            source_snapshot={
                "required_address": "10 Cedar Street",
                "address": "10 Cedar Street",
                "transition_valid": True,
            },
            execute=_memory_execute,
            verify=lambda state: (
                state["transition_valid"] and state["address"] == state["required_address"]
            ),
        ),
        "orchestrator-routing-error-001": _ReplayProgram(
            target_event_id="event-2",
            sham_content="route(tool='news.lookup', args={'city':'Paris'})",
            source_snapshot={
                "required_tool": "weather.lookup",
                "required_city": "Paris",
                "routed_tool": None,
                "city": None,
                "transition_valid": True,
            },
            execute=_router_execute,
            verify=lambda state: (
                state["transition_valid"]
                and state["routed_tool"] == state["required_tool"]
                and state["city"] == state["required_city"]
            ),
        ),
        "premature-termination-001": _ReplayProgram(
            target_event_id="event-2",
            sham_content="email.send(to='sales@example.com', subject='Report ready')",
            source_snapshot={
                "report_created": True,
                "required_recipient": "ops@example.com",
                "email_sent_to": None,
                "transition_valid": True,
            },
            execute=_email_execute,
            verify=lambda state: (
                state["transition_valid"]
                and state["report_created"]
                and state["email_sent_to"] == state["required_recipient"]
            ),
        ),
    }


class ControlledReplayController:
    """Restore one frozen state and execute baseline, repair, and matched sham."""

    controller_id = "controlled_executable_replay/v3"

    def __init__(self, case_id: str) -> None:
        try:
            self._program = _programs()[case_id]
        except KeyError as error:
            raise ValueError(f"unsupported controlled replay case: {case_id}") from error
        self._case_id = case_id
        # Freeze the source snapshot hash once. Every arm compares its working state
        # against this reference, so a state that diverges from the declared frozen
        # snapshot (tampering, or a lossy state copy) fails restore verification
        # instead of being compared against a hash of itself.
        self._source_snapshot_sha256 = _sha256(self._program.source_snapshot)
        self._source_trace_sha256 = {
            cell.case.case_id: cell.case.sha256 for cell in controlled_traceback_cells()
        }[case_id]

    def _run_arm(
        self,
        *,
        arm: str,
        replacement_content: str,
        target_event_id: str,
        source_snapshot_sha256: str,
        original_target_content: str,
    ) -> tuple[ReplayOutcomeV1, str]:
        state = _copy_state(self._program.source_snapshot)
        restored_state_sha256 = _sha256(state)
        # This program can only replay its own frozen target event. An intervention on a
        # different event is not executable, so the arm re-runs the original action and
        # the outcome reports the executed event and content, not the intervention's
        # intent; the runner's binding blockers then reject the mismatched claim.
        replayed_event_id = target_event_id
        effective_content = replacement_content
        if target_event_id != self._program.target_event_id:
            replayed_event_id = self._program.target_event_id
            effective_content = original_target_content
        self._program.execute(state, effective_content)
        success = self._program.verify(state)
        terminal_reason = "verifier_accepted" if success else "verifier_rejected"
        unhashed = ReplayOutcomeV1(
            success=success,
            terminal_reason=terminal_reason,
            target_event_id=replayed_event_id,
            input_content_sha256=_sha256(effective_content),
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
                    case_id=self._case_id,
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
        if case.case_id != self._case_id:
            raise TraceInvestigationError("controlled replay case mismatch")
        if case.sha256 != self._source_trace_sha256:
            raise TraceInvestigationError("controlled replay frozen trace hash mismatch")
        event_by_id = {event.event_id: event for event in case.events}
        try:
            original_target_content = event_by_id[self._program.target_event_id].content
        except KeyError as error:
            raise TraceInvestigationError("controlled replay target missing from trace") from error

        source_snapshot_sha256 = self._source_snapshot_sha256
        baseline, baseline_restore = self._run_arm(
            arm="fault_baseline",
            replacement_content=original_target_content,
            target_event_id=self._program.target_event_id,
            source_snapshot_sha256=source_snapshot_sha256,
            original_target_content=original_target_content,
        )
        repair, repair_restore = self._run_arm(
            arm="predicted_repair",
            replacement_content=intervention.replacement_content,
            target_event_id=intervention.target_event_id,
            source_snapshot_sha256=source_snapshot_sha256,
            original_target_content=original_target_content,
        )
        sham, sham_restore = self._run_arm(
            arm="matched_sham",
            replacement_content=self._program.sham_content,
            target_event_id=self._program.target_event_id,
            source_snapshot_sha256=source_snapshot_sha256,
            original_target_content=original_target_content,
        )
        restore_verified = all(
            restored == source_snapshot_sha256
            for restored in (baseline_restore, repair_restore, sham_restore)
        )
        unhashed = CounterfactualReplayEvidenceV1(
            controller_id=self.controller_id,
            source_trace_sha256=case.sha256,
            source_snapshot_sha256=source_snapshot_sha256,
            restore_verified=restore_verified,
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
