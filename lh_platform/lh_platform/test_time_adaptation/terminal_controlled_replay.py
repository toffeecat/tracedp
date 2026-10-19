"""Frozen public-state interventions over observed Terminal trajectories."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from lh_platform.benchmarks.terminal_observed_trajectory import TerminalObservedTransition
from lh_platform.benchmarks.terminal_public_state import (
    PublicFileObservation,
    compact_public_file_state,
    decode_exact_public_file_state,
)


@dataclass(frozen=True)
class TerminalReplayIntervention:
    """One exact public-state edit scheduled after a frozen tool call."""

    arm: Literal["replacement", "sham"]
    after_call_sequence: int
    target_state: tuple[PublicFileObservation, ...]


def compile_terminal_replay_interventions(
    payloads: tuple[dict[str, Any], ...],
    *,
    tool_call_count: int,
) -> tuple[TerminalReplayIntervention, ...]:
    """Validate and order a same-arm multi-checkpoint replay plan."""

    if tool_call_count < 1:
        raise ValueError("terminal replay requires at least one frozen tool call")
    interventions = []
    for payload in payloads:
        sequence = payload.get("after_call_sequence")
        arm = payload.get("arm")
        if (
            type(sequence) is not int
            or not 0 <= sequence < tool_call_count
            or arm not in {"replacement", "sham"}
        ):
            raise ValueError("terminal intervention identity is invalid")
        interventions.append(
            TerminalReplayIntervention(
                arm=arm,
                after_call_sequence=sequence,
                target_state=decode_exact_public_file_state(payload.get("target_state")),
            )
        )
    sequences = [item.after_call_sequence for item in interventions]
    if len(sequences) != len(set(sequences)):
        raise ValueError("terminal replay cannot apply two interventions after one call")
    if len({item.arm for item in interventions}) > 1:
        raise ValueError("terminal grouped interventions must use one matched arm")
    seen_paths: set[str] = set()
    for item in interventions:
        paths = {state.path for state in item.target_state}
        # Later snapshots must not silently undo an earlier counterfactual edit.
        if seen_paths.intersection(paths):
            raise ValueError("terminal grouped interventions require disjoint public paths")
        seen_paths.update(paths)
    return tuple(sorted(interventions, key=lambda item: item.after_call_sequence))


def replace_once_in_tool_call(
    call: dict[str, object],
    *,
    tool_name: str,
    argument_name: str,
    before: str,
    after: str,
) -> dict[str, object]:
    """Return one frozen call with exactly one textual intervention."""
    if call.get("tool_name") != tool_name:
        raise ValueError("frozen root tool does not match the intervention contract")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("frozen root tool arguments must be an object")
    value = arguments.get(argument_name)
    if not isinstance(value, str) or value.count(before) != 1:
        raise ValueError("frozen root argument changed from the intervention contract")
    return {
        **call,
        "tool_name": tool_name,
        "arguments": {**arguments, argument_name: value.replace(before, after)},
    }


def require_exact_prefix_and_root_state(
    source: tuple[TerminalObservedTransition, ...],
    replay: tuple[TerminalObservedTransition, ...],
    *,
    root_sequence: int,
) -> None:
    """Bind the intervention to the exact source-visible state at the root."""
    if len(source) != len(replay):
        raise ValueError("controlled replay tool-call population changed")
    if not 0 <= root_sequence < len(source):
        raise ValueError("controlled root sequence is outside the source trajectory")
    for expected, actual in zip(source[:root_sequence], replay[:root_sequence], strict=True):
        if replace(expected, tool_call_id=actual.tool_call_id) != actual:
            raise ValueError("controlled replay diverged before the hidden root")
    if source[root_sequence].before != replay[root_sequence].before:
        raise ValueError("controlled replay root pre-action state changed")


def validate_terminal_grouped_replay(
    *,
    payloads: tuple[dict[str, Any], ...],
    original: tuple[TerminalObservedTransition, ...],
    replay: tuple[TerminalObservedTransition, ...],
    receipts: tuple[dict[str, Any], ...],
    original_tool_results: tuple[object, ...],
    replay_tool_results: tuple[object, ...],
) -> None:
    """Validate frozen calls and every exact readback in a grouped replay."""

    plan = compile_terminal_replay_interventions(
        payloads,
        tool_call_count=len(original),
    )
    if len(plan) < 2:
        raise ValueError("grouped replay requires at least two interventions")
    if len(original) != len(replay) or len(original_tool_results) != len(original):
        raise ValueError("grouped replay tool-call population changed")
    if len(replay_tool_results) != len(replay) or len(receipts) != len(plan):
        raise ValueError("grouped replay evidence population changed")
    for expected, actual in zip(original, replay, strict=True):
        if (
            expected.sequence,
            expected.tool_name,
            expected.tool_input,
            expected.environment_step_executed,
        ) != (
            actual.sequence,
            actual.tool_name,
            actual.tool_input,
            actual.environment_step_executed,
        ):
            raise ValueError("grouped replay frozen tool chain changed")
    first_sequence = plan[0].after_call_sequence
    require_exact_prefix_and_root_state(
        original,
        replay,
        root_sequence=first_sequence,
    )
    if original[first_sequence].after != replay[first_sequence].after:
        raise ValueError("grouped replay first root call changed before intervention")
    if original_tool_results[: first_sequence + 1] != replay_tool_results[: first_sequence + 1]:
        raise ValueError("grouped replay first root tool output changed")

    for scheduled, receipt in zip(plan, receipts, strict=True):
        sequence = scheduled.after_call_sequence
        target = compact_public_file_state(scheduled.target_state)
        before = compact_public_file_state(replay[sequence].after)
        target_by_path = {str(row["path"]): row for row in target}
        before_paths = {str(row["path"]) for row in before}
        if not set(target_by_path) <= before_paths:
            raise ValueError("grouped replay target path is absent from checkpoint state")
        expected_after = [target_by_path.get(str(row["path"]), row) for row in before]
        if (
            receipt.get("schema_version") != "terminal_public_state_intervention_v1"
            or receipt.get("arm") != scheduled.arm
            or receipt.get("after_call_sequence") != sequence
            or receipt.get("target_state") != target
            or receipt.get("before") != before
            or receipt.get("after") != expected_after
            or receipt.get("exact_readback") is not True
            or receipt.get("model_called") is not False
        ):
            raise ValueError("grouped replay intervention receipt differs")
        # No state transition exists between an intervention and the next call;
        # this binds the replacement to the later checkpoint instead of merely
        # validating an isolated receipt.
        if (
            sequence < len(replay) - 1
            and compact_public_file_state(replay[sequence + 1].before) != receipt["after"]
        ):
            raise ValueError("grouped replay state continuity differs")

    intervention_sequences = {item.after_call_sequence for item in plan}
    for sequence in range(len(replay) - 1):
        if sequence in intervention_sequences:
            continue
        if compact_public_file_state(replay[sequence + 1].before) != compact_public_file_state(
            replay[sequence].after
        ):
            raise ValueError("grouped replay state continuity differs")


__all__ = [
    "TerminalReplayIntervention",
    "compile_terminal_replay_interventions",
    "replace_once_in_tool_call",
    "require_exact_prefix_and_root_state",
    "validate_terminal_grouped_replay",
]
