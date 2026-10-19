"""Public-input contracts for replay-optional Terminal forward recovery."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lh_platform.benchmarks.terminal_observed_trajectory import (
    load_observed_terminal_transitions,
)
from lh_platform.benchmarks.terminal_pi_stream import load_pi_stream_receipt
from lh_platform.benchmarks.terminal_public_state import (
    public_file_observation_payload,
)
from lh_platform.benchmarks.terminal_runtime_identity import TerminalRuntimeIdentity

TERMINAL_FORWARD_DIRECT_PROMPT = "direct_continue"
TERMINAL_FORWARD_CRITIQUE_PROMPT = "critique_then_replan"
TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT = "adapted_targeted_guidance"
_TERMINAL_FORWARD_PROMPT_MODES = frozenset(
    {
        TERMINAL_FORWARD_DIRECT_PROMPT,
        TERMINAL_FORWARD_CRITIQUE_PROMPT,
        TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT,
    }
)

_HISTORY_FIELDS = frozenset(
    {
        "sequence",
        "tool_name",
        "tool_input",
        "tool_output",
        "is_error",
        "environment_step_executed",
    }
)
_SOURCE_ACTOR_FIELDS = (
    "provider",
    "model",
    "transport",
    "thinking",
    "max_output_tokens",
)
_STATE_COVERAGE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_id",
        "review_status",
        "audit_packet_file",
        "audit_decision_file",
        "audit_id",
        "auditor_id",
        "benchmark_id",
        "task_name",
        "source_trial_name",
        "source_trial_relative_path",
        "source_map_relative_path",
        "task_contract_receipt_relative_path",
        "public_paths",
        "path_compilation_truncated",
        "reconstruction_scope",
        "unsupported_state_outcome_determining",
        "assessed_state_domains",
        "evidence",
    }
)
_STATE_COVERAGE_DOMAINS = frozenset(
    {
        "declared_public_files",
        "unlisted_files",
        "processes",
        "services",
        "network",
        "database",
        "environment_variables",
        "permissions",
        "clock",
    }
)
_STATE_COVERAGE_EVIDENCE_FIELDS = frozenset(
    {"evidence_id", "source", "finding"}
)
_STATE_COVERAGE_ACTOR_FIELDS = frozenset(
    {
        "schema_version",
        "producer_id",
        "review_status",
        "audit_id",
        "auditor_id",
        "benchmark_id",
        "task_name",
        "source_trial_name",
        "public_paths",
        "path_compilation_truncated",
        "reconstruction_scope",
        "unsupported_state_outcome_determining",
        "assessed_state_domains",
        "reviewed_evidence_ids",
    }
)


def normalize_terminal_forward_prompt_mode(value: object) -> str:
    """Accept only the two preregistered same-actor continuation treatments."""

    if not isinstance(value, str) or value not in _TERMINAL_FORWARD_PROMPT_MODES:
        raise ValueError("Terminal forward continuation prompt mode is invalid")
    return value


def build_terminal_forward_continuation_instruction(
    *,
    task_instruction: str,
    history: tuple[dict[str, object], ...],
    prompt_mode: str,
    targeted_guidance: str | None = None,
) -> str:
    """Compile one root-free continuation treatment from public source evidence."""

    prompt_mode = normalize_terminal_forward_prompt_mode(prompt_mode)
    normalized_guidance = targeted_guidance.strip() if targeted_guidance is not None else None
    if prompt_mode == TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT:
        if not normalized_guidance:
            raise ValueError("targeted continuation requires method-generated guidance")
    elif normalized_guidance is not None:
        raise ValueError("untargeted continuation cannot carry method-generated guidance")
    if prompt_mode == TERMINAL_FORWARD_DIRECT_PROMPT:
        treatment = (
            "Continue the original task from its observed failed final state. Inspect the "
            "current sandbox, correct any remaining problem, and finish the task. Do not "
            "assume the prior attempt's final answer is correct."
        )
    elif prompt_mode == TERMINAL_FORWARD_CRITIQUE_PROMPT:
        treatment = (
            "Before taking any tool action, write a concise visible critique of why the "
            "completed attempt failed and a recovery plan grounded only in the original "
            "task and public history below. Then inspect the current sandbox, execute that "
            "plan, and finish the original task. Do not assume the prior attempt's final "
            "answer is correct."
        )
    else:
        # Historical-message fixers need an explicit adaptation when the backend can only
        # continue from the current state; the receipt records that semantic difference.
        treatment = (
            "Continue the original task from its observed failed final state. Apply the "
            "following method-generated targeted guidance, verify its postcondition in the "
            "current sandbox, and finish the task:\n"
            f"{normalized_guidance}\n"
            "Do not assume the guidance or the prior attempt's final answer is correct."
        )
    return (
        f"{treatment}\n\n"
        f"ORIGINAL TASK:\n{task_instruction}\n\n"
        "PUBLIC EXECUTED HISTORY:\n"
        f"{json.dumps(history, ensure_ascii=False)}"
    )


@dataclass(frozen=True)
class TerminalForwardRecoverySource:
    """Evaluator-independent inputs reconstructed from one failed Harbor trial."""

    task: str
    failed_public_file_state: tuple[dict[str, object], ...]
    public_history: tuple[dict[str, object], ...]
    source_actor: dict[str, object]


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"forward recovery source must contain an object: {path.name}")
    return value


def _single_session(source_trial: Path) -> Path:
    sessions = tuple(
        path
        for path in (source_trial / "agent" / "pi" / "sessions").glob("*.jsonl")
        if not path.name.startswith("._")
    )
    if len(sessions) != 1:
        raise ValueError("forward recovery source must contain one Pi session")
    return sessions[0]


def _source_observer(source_trial: Path) -> Path:
    """Resolve the current census observer path with legacy-run compatibility."""

    current = source_trial / "agent" / "terminal-public-state.jsonl"
    if current.is_file():
        return current
    legacy = source_trial / "agent" / "pi" / "terminal-public-state.jsonl"
    if legacy.is_file():
        return legacy
    raise ValueError("forward recovery source public-state observer is missing")


def _source_actor_identity(
    source_trial: Path,
    *,
    result: dict[str, object],
    actor_timed_out: bool,
) -> dict[str, object]:
    """Recover a timeout-truncated actor identity from its config and Pi receipt."""

    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    if isinstance(metadata, dict):
        return {field: metadata.get(field) for field in _SOURCE_ACTOR_FIELDS}
    if not actor_timed_out:
        raise ValueError("forward recovery source actor metadata is missing")

    config = result.get("config")
    agent = config.get("agent") if isinstance(config, dict) else None
    kwargs = agent.get("kwargs") if isinstance(agent, dict) else None
    agent_info = result.get("agent_info")
    model_info = agent_info.get("model_info") if isinstance(agent_info, dict) else None
    provider = model_info.get("provider") if isinstance(model_info, dict) else None
    model = model_info.get("name") if isinstance(model_info, dict) else None
    if (
        not isinstance(agent, dict)
        or agent.get("name")
        != "lh_platform.benchmarks.harbor_opencode_go_pi:HostDelegatedOpenCodeGoPi"
        or not isinstance(kwargs, dict)
        or not isinstance(provider, str)
        or not provider
        or not isinstance(model, str)
        or not model
        or agent.get("model_name") != f"{provider}/{model}"
        or kwargs.get("thinking") != "off"
        or type(kwargs.get("host_pi_max_output_tokens")) is not int
    ):
        raise ValueError("forward recovery timeout actor identity is incomplete")
    receipt = load_pi_stream_receipt(
        source_trial / "agent" / "pi.txt",
        expected_provider=provider,
        expected_model=model,
    )
    try:
        receipt.require_valid(return_code=0)
    except RuntimeError as error:
        raise ValueError("forward recovery timeout Pi receipt is invalid") from error
    return {
        "provider": provider,
        "model": model,
        "transport": "host_pi_unix_rpc",
        "thinking": "off",
        "max_output_tokens": int(kwargs["host_pi_max_output_tokens"]),
    }


def _public_tool_outputs(session: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for line in session.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("forward recovery source session row differs")
        message = row.get("message")
        if row.get("type") != "message" or not isinstance(message, dict):
            continue
        if message.get("role") != "toolResult":
            continue
        tool_call_id = str(message.get("toolCallId", ""))
        if not tool_call_id or tool_call_id in outputs:
            raise ValueError("forward recovery source tool result identity differs")
        blocks = message.get("content") or []
        if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
            raise ValueError("forward recovery source tool result content differs")
        outputs[tool_call_id] = "\n".join(
            str(block.get("text", "")) for block in blocks if block.get("type") == "text"
        )
    return outputs


def load_terminal_forward_recovery_source(
    source_trial: Path,
    *,
    public_paths: tuple[str, ...],
    expected_task: str,
) -> TerminalForwardRecoverySource:
    """Compile actor-visible recovery input from an official failed Harbor trial."""

    result = _read_object(source_trial / "result.json")
    task_name = str(result.get("task_name", ""))
    if task_name not in {expected_task, f"terminal-bench/{expected_task}"}:
        raise ValueError("forward recovery source task differs")
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    exception = result.get("exception_info")
    actor_timed_out = (
        isinstance(exception, dict) and exception.get("exception_type") == "AgentTimeoutError"
    )
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or float(reward) != 0.0
        or (exception is not None and not actor_timed_out)
    ):
        raise ValueError("forward recovery source must be an official natural failure")
    session = _single_session(source_trial)
    observer = _source_observer(source_trial)
    transitions = load_observed_terminal_transitions(
        session_path=session,
        observer_path=observer,
        public_paths=public_paths,
    )
    complete_transitions = tuple(transition for transition in transitions if transition.after)
    if not complete_transitions:
        raise ValueError("forward recovery source has no complete public final state")
    tool_outputs = _public_tool_outputs(session)
    history = normalize_terminal_public_history(
        [
            {
                "sequence": transition.sequence,
                "tool_name": transition.tool_name,
                "tool_input": transition.tool_input,
                "tool_output": tool_outputs[transition.tool_call_id],
                "is_error": transition.is_error,
                "environment_step_executed": transition.environment_step_executed,
            }
            for transition in transitions
        ]
    )
    source_actor = _source_actor_identity(
        source_trial,
        result=result,
        actor_timed_out=actor_timed_out,
    )
    return TerminalForwardRecoverySource(
        task=expected_task,
        failed_public_file_state=tuple(
            public_file_observation_payload(item) for item in complete_transitions[-1].after
        ),
        public_history=history,
        source_actor=source_actor,
    )


def normalize_terminal_public_history(rows: object) -> tuple[dict[str, object], ...]:
    """Accept only tool history that was visible to the original actor."""

    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("forward recovery requires non-empty public tool history")
    normalized: list[dict[str, object]] = []
    previous_sequence = -1
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _HISTORY_FIELDS:
            raise ValueError("forward recovery public history schema differs")
        sequence = row["sequence"]
        if type(sequence) is not int or sequence <= previous_sequence:
            raise ValueError("forward recovery public history order differs")
        tool_name = row["tool_name"]
        tool_input = row["tool_input"]
        tool_output = row["tool_output"]
        is_error = row["is_error"]
        executed = row["environment_step_executed"]
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("forward recovery public history tool name is invalid")
        if not isinstance(tool_input, Mapping) or not isinstance(tool_output, str):
            raise ValueError("forward recovery public history tool payload is invalid")
        if type(is_error) is not bool or type(executed) is not bool:
            raise ValueError("forward recovery public history status is invalid")
        normalized.append(
            {
                "sequence": sequence,
                "tool_name": tool_name,
                "tool_input": dict(tool_input),
                "tool_output": tool_output,
                "is_error": is_error,
                "environment_step_executed": executed,
            }
        )
        previous_sequence = sequence
    return tuple(normalized)


def normalize_terminal_source_actor(
    value: object,
    *,
    runtime: TerminalRuntimeIdentity,
    max_output_tokens: int,
) -> dict[str, object]:
    """Bind recovery to the actor contract that produced the failed episode."""

    if not isinstance(value, Mapping) or set(value) != set(_SOURCE_ACTOR_FIELDS):
        raise ValueError("forward recovery source actor schema differs")
    expected: dict[str, object] = {
        "provider": runtime.provider,
        "model": runtime.model,
        "transport": runtime.transport,
        "thinking": runtime.thinking,
        "max_output_tokens": max_output_tokens,
    }
    normalized = dict(value)
    if normalized != expected:
        raise ValueError("forward recovery source actor differs from continuation actor")
    return normalized


def normalize_terminal_state_coverage(
    value: object,
    *,
    expected_task: str | None = None,
    expected_source_trial_name: str | None = None,
    expected_source_trial_relative_path: str | None = None,
    expected_source_map_relative_path: str | None = None,
    expected_task_contract_receipt_relative_path: str | None = None,
    expected_public_paths: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Validate a task-bound audit receipt for public-projection recovery."""

    if not isinstance(value, Mapping) or set(value) != _STATE_COVERAGE_FIELDS:
        raise ValueError("forward recovery state coverage schema differs")
    evidence = value["evidence"]
    domains = value["assessed_state_domains"]
    public_paths = value["public_paths"]
    identities = (
        value["audit_id"],
        value["auditor_id"],
        value["task_name"],
        value["source_trial_name"],
        value["source_trial_relative_path"],
        value["source_map_relative_path"],
        value["task_contract_receipt_relative_path"],
    )
    if (
        value["schema_version"] != "terminal_forward_state_coverage_receipt_v2"
        or value["producer_id"] != "longh_terminal_state_coverage_packet_builder_v1"
        or value["review_status"] != "approved"
        or value["audit_packet_file"] != "audit_packet.json"
        or value["audit_decision_file"] != "audit_decision.json"
        or value["benchmark_id"] != "terminal-bench-2.1"
        or any(not isinstance(item, str) or not item for item in identities)
        or not isinstance(public_paths, (list, tuple))
        or not public_paths
        or any(not isinstance(item, str) or not item for item in public_paths)
        or len(set(public_paths)) != len(public_paths)
        or type(value["path_compilation_truncated"]) is not bool
        or value["reconstruction_scope"] != "declared_public_files"
        or value["unsupported_state_outcome_determining"] is not False
        or not isinstance(domains, Mapping)
        or set(domains) != _STATE_COVERAGE_DOMAINS
        or domains.get("declared_public_files") != "restored"
        or any(
            domains.get(domain) != "not_outcome_determining"
            for domain in _STATE_COVERAGE_DOMAINS - {"declared_public_files"}
        )
        or not isinstance(evidence, (list, tuple))
        or not evidence
        or any(
            not isinstance(item, Mapping)
            or set(item) != _STATE_COVERAGE_EVIDENCE_FIELDS
            or any(
                not isinstance(item[field], str) or not item[field]
                for field in _STATE_COVERAGE_EVIDENCE_FIELDS
            )
            for item in evidence
        )
    ):
        raise ValueError("public file projection is insufficient for forward recovery")
    if expected_task is not None and value["task_name"] != expected_task:
        raise ValueError("state coverage receipt belongs to a different task")
    if (
        expected_source_trial_name is not None
        and value["source_trial_name"] != expected_source_trial_name
    ):
        raise ValueError("state coverage receipt belongs to a different source trial")
    expected_bindings = (
        ("source trial", expected_source_trial_relative_path, value["source_trial_relative_path"]),
        ("source map", expected_source_map_relative_path, value["source_map_relative_path"]),
        (
            "task contract",
            expected_task_contract_receipt_relative_path,
            value["task_contract_receipt_relative_path"],
        ),
    )
    for label, expected, actual in expected_bindings:
        if expected is not None and actual != expected:
            raise ValueError(f"state coverage receipt belongs to a different {label}")
    if expected_public_paths is not None and tuple(public_paths) != expected_public_paths:
        raise ValueError("state coverage receipt belongs to different public paths")
    return {
        **dict(value),
        "public_paths": list(public_paths),
        "assessed_state_domains": dict(domains),
        "evidence": [dict(item) for item in evidence],
    }


def load_terminal_state_coverage_receipt(
    path: Path,
    *,
    storage_root: Path,
    terminal_bench_root: Path,
    expected_task: str,
    expected_source_trial: Path,
    expected_source_map_receipt: Path,
    expected_task_contract_receipt: Path,
    expected_public_paths: tuple[str, ...],
) -> dict[str, object]:
    """Load one durable audit artifact instead of trusting launch-time prose."""

    storage_root = storage_root.resolve()
    source_trial_relative = expected_source_trial.resolve().relative_to(storage_root).as_posix()
    source_map_relative = (
        expected_source_map_receipt.resolve().relative_to(storage_root).as_posix()
    )
    task_contract_relative = (
        expected_task_contract_receipt.resolve().relative_to(storage_root).as_posix()
    )
    normalized = normalize_terminal_state_coverage(
        _read_object(path),
        expected_task=expected_task,
        expected_source_trial_name=expected_source_trial.name,
        expected_source_trial_relative_path=source_trial_relative,
        expected_source_map_relative_path=source_map_relative,
        expected_task_contract_receipt_relative_path=task_contract_relative,
        expected_public_paths=expected_public_paths,
    )
    # Rebuild the claimed receipt from its separate evidence and review artifacts.
    from lh_platform.benchmarks.terminal_state_coverage_audit import (
        verify_terminal_state_coverage_bundle,
    )

    verify_terminal_state_coverage_bundle(
        path,
        receipt=normalized,
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
    )
    return normalized


def terminal_state_coverage_actor_attestation(
    receipt: object,
) -> dict[str, object]:
    """Remove evaluator-only source locations before configuring the actor."""

    normalized = normalize_terminal_state_coverage(receipt)
    return {
        "schema_version": "terminal_forward_state_coverage_actor_attestation_v1",
        "producer_id": normalized["producer_id"],
        "review_status": normalized["review_status"],
        "audit_id": normalized["audit_id"],
        "auditor_id": normalized["auditor_id"],
        "benchmark_id": normalized["benchmark_id"],
        "task_name": normalized["task_name"],
        "source_trial_name": normalized["source_trial_name"],
        "public_paths": list(normalized["public_paths"]),
        "path_compilation_truncated": normalized["path_compilation_truncated"],
        "reconstruction_scope": normalized["reconstruction_scope"],
        "unsupported_state_outcome_determining": normalized[
            "unsupported_state_outcome_determining"
        ],
        "assessed_state_domains": dict(normalized["assessed_state_domains"]),
        "reviewed_evidence_ids": [
            str(item["evidence_id"]) for item in normalized["evidence"]
        ],
    }


def normalize_terminal_state_coverage_actor_attestation(
    value: object,
) -> dict[str, object]:
    """Accept only the minimal reviewed state claim exposed to the actor."""

    if not isinstance(value, Mapping) or set(value) != _STATE_COVERAGE_ACTOR_FIELDS:
        raise ValueError("forward recovery actor state coverage schema differs")
    domains = value["assessed_state_domains"]
    public_paths = value["public_paths"]
    evidence_ids = value["reviewed_evidence_ids"]
    identities = (
        value["audit_id"],
        value["auditor_id"],
        value["task_name"],
        value["source_trial_name"],
    )
    if (
        value["schema_version"]
        != "terminal_forward_state_coverage_actor_attestation_v1"
        or value["producer_id"] != "longh_terminal_state_coverage_packet_builder_v1"
        or value["review_status"] != "approved"
        or value["benchmark_id"] != "terminal-bench-2.1"
        or any(not isinstance(item, str) or not item for item in identities)
        or not isinstance(public_paths, (list, tuple))
        or not public_paths
        or any(not isinstance(item, str) or not item for item in public_paths)
        or len(set(public_paths)) != len(public_paths)
        or type(value["path_compilation_truncated"]) is not bool
        or value["reconstruction_scope"] != "declared_public_files"
        or value["unsupported_state_outcome_determining"] is not False
        or not isinstance(domains, Mapping)
        or set(domains) != _STATE_COVERAGE_DOMAINS
        or domains.get("declared_public_files") != "restored"
        or any(
            domains.get(domain) != "not_outcome_determining"
            for domain in _STATE_COVERAGE_DOMAINS - {"declared_public_files"}
        )
        or not isinstance(evidence_ids, (list, tuple))
        or not evidence_ids
        or any(not isinstance(item, str) or not item for item in evidence_ids)
        or len(set(evidence_ids)) != len(evidence_ids)
    ):
        raise ValueError("public file projection is insufficient for forward recovery")
    return {
        **dict(value),
        "public_paths": list(public_paths),
        "assessed_state_domains": dict(domains),
        "reviewed_evidence_ids": list(evidence_ids),
    }


__all__ = [
    "TERMINAL_FORWARD_CRITIQUE_PROMPT",
    "TERMINAL_FORWARD_DIRECT_PROMPT",
    "TerminalForwardRecoverySource",
    "build_terminal_forward_continuation_instruction",
    "load_terminal_state_coverage_receipt",
    "load_terminal_forward_recovery_source",
    "normalize_terminal_public_history",
    "normalize_terminal_forward_prompt_mode",
    "normalize_terminal_source_actor",
    "normalize_terminal_state_coverage",
    "normalize_terminal_state_coverage_actor_attestation",
    "terminal_state_coverage_actor_attestation",
]
