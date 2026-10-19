"""Harbor Pi agents configured for OpenCode Go credentials.

Harbor 0.21 knows the ``opencode`` provider but not Pi's distinct
``opencode-go`` provider slug.  The stock Pi agent therefore cannot forward the
already configured ``OPENCODE_API_KEY`` when the model is
``opencode-go/<model>``.  This adapter changes only that credential-resolution
contract; installation, tool execution, trajectory capture, and scoring remain
Harbor's native Pi implementation.
"""

import asyncio
import contextlib
import json
import os
import shutil
import signal
import tempfile
import time
import uuid
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import override

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.pi import Pi
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName

from lh_platform.benchmarks.runtime_task_contract_compiler import (
    SUPPORTED_TASK_CONTRACT_COMPILERS,
    TASK_CONTRACT_COMPILER_V3,
    apply_public_instruction_dependency_evidence,
)
from lh_platform.benchmarks.shared_budget_proxy import SharedBudgetProviderProxy
from lh_platform.benchmarks.terminal_attribution_guidance import (
    historical_investigation_continuation,
)
from lh_platform.benchmarks.terminal_calibration_sweep import (
    run_calibration_sweep,
)
from lh_platform.benchmarks.terminal_delegated_tools import (
    ACTOR_CONVERGENCE_RESERVE_SECONDS,
    ACTOR_FAILURE_DIAGNOSIS_THRESHOLD,
    ACTOR_TOOL_RECEIPT_SCHEMA,
    OUTPUT_PROGRESS_ACTION_THRESHOLD,
    OUTPUT_SEMANTIC_STABILITY_ACTION_THRESHOLD,
    ActorNaturalSettlementPrompt,
    TerminalEnvironmentToolGateway,
    TerminalToolRpcServer,
    probe_bash_containment_capability,
)
from lh_platform.benchmarks.terminal_forward_recovery_contract import (
    TERMINAL_FORWARD_CRITIQUE_PROMPT,
    TERMINAL_FORWARD_DIRECT_PROMPT,
    TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT,
    build_terminal_forward_continuation_instruction,
    normalize_terminal_forward_prompt_mode,
    normalize_terminal_public_history,
    normalize_terminal_source_actor,
    normalize_terminal_state_coverage_actor_attestation,
)
from lh_platform.benchmarks.terminal_live_acceptance_portfolio import (
    TerminalSeparateEnvironmentPortfolioGate,
    candidate_environment_session_id,
    isolated_actor_guidance_instruction,
    isolated_guidance_execution,
)
from lh_platform.benchmarks.terminal_managed_runtime_identity import (
    EFFECT_LEDGER_RUNTIME_METHODS,
    INTERVENTION_BUDGET_METHODS,
    JOINT_RUNTIME_METHODS,
    OUTPUT_PROGRESS_RUNTIME_METHODS,
    POLICY_RUNTIME_METHODS,
    SHARED_INPUT_PROTECTION_METHOD,
    SUPPORTED_RUNTIME_METHODS,
)
from lh_platform.benchmarks.terminal_pi_invocation import (
    host_pi_command as _host_pi_command,
)
from lh_platform.benchmarks.terminal_pi_invocation import (
    host_pi_rpc_command as _host_pi_rpc_command,
)
from lh_platform.benchmarks.terminal_pi_invocation import (
    neutral_actor_cwd,
)
from lh_platform.benchmarks.terminal_pi_invocation import (
    send_host_pi_instruction as _send_host_pi_instruction,
)
from lh_platform.benchmarks.terminal_pi_invocation import (
    send_host_pi_rpc_command as _send_host_pi_rpc_command,
)
from lh_platform.benchmarks.terminal_pi_invocation import (
    write_host_pi_models_config as _write_host_pi_models_config,
)
from lh_platform.benchmarks.terminal_pi_session_continuation import (
    active_session_message_count,
    load_frozen_pi_continuation_source,
    stage_pi_continuation_session,
    terminal_instructions_are_equivalent,
)
from lh_platform.benchmarks.terminal_pi_stream import (
    PiStreamReceipt,
    load_pi_stream_receipt,
)
from lh_platform.benchmarks.terminal_prefix_parity import (  # noqa: E402
    PREFIX_PARITY_POLICY_EXACT,
    PREFIX_PARITY_POLICY_EXEMPTIONS,
    classify_prefix_parity,
)
from lh_platform.benchmarks.terminal_provider_http_capture import (
    require_terminal_provider_http_capture,
)
from lh_platform.benchmarks.terminal_prefix_snapshot import (
    commit_container,
    container_id,
    public_state_payload,
    restore_prefix_from_snapshot,
)
from lh_platform.benchmarks.terminal_public_state import (
    compact_public_file_state,
    decode_exact_public_file_state,
    public_file_observation_payload,
)
from lh_platform.benchmarks.terminal_resource_driver import (
    TERMINAL_RESOURCE_OBSERVATION_V1,
    TERMINAL_RESOURCE_OBSERVATION_V2,
    ResourceSpec,
    TerminalResourceDriver,
    expand_sqlite_resource_family_specs,
)
from lh_platform.benchmarks.terminal_runtime_check import (
    RuntimeCheckContract,
    RuntimePredicateRepairContract,
    is_actor_guidance_repair_arguments,
    runtime_check_configuration_payload,
    runtime_repair_configuration_payload,
)
from lh_platform.benchmarks.terminal_runtime_identity import (
    resolve_host_pi_runtime,
)
from lh_platform.benchmarks.terminal_source_census import MAX_HOST_PI_TIMEOUT_SECONDS
from lh_platform.test_time_adaptation.adapters.terminal_runtime import (
    TERMINAL_RUNTIME_ADAPTER_V1,
    TerminalRuntimeDatasetAdapter,
    TerminalRuntimeNativeState,
)
from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    ActionPhase,
    PortableRuntimeKernel,
    ResourceKind,
)
from lh_platform.test_time_adaptation.runtime_acceptance_predicate import (
    ACCEPTANCE_PREDICATE_IMPLEMENTATION_STATUS,
    EXPOSED_DEVELOPMENT_CANARY_RUN_CLASS,
    FORMAL_EXPERIMENT_RUN_CLASS,
    compile_public_acceptance_manifest,
)
from lh_platform.test_time_adaptation.runtime_bounded_trace_dp import (
    RuntimeBoundedTraceDPResolver,
    RuntimeBoundedTraceDPV4Resolver,
    RuntimeBoundedTraceDPV6Resolver,
    RuntimeBoundedTraceDPV7Resolver,
    RuntimeEventTraceDPV8Resolver,
)
from lh_platform.test_time_adaptation.runtime_compact_prediction import (
    V7_PROVIDER_TIMEOUT_OVERRUN_MARGIN_MS,
)
from lh_platform.test_time_adaptation.runtime_dataset_adapter import (
    RuntimeTaskInstanceSpec,
    instantiate_runtime_view,
)
from lh_platform.test_time_adaptation.runtime_event_admission import (
    V8_CUMULATIVE_PLANNER_BUDGET_MS,
    V8_SEMANTIC_REVIEW_THRESHOLD,
    planner_event_admission_v8,
)
from lh_platform.test_time_adaptation.runtime_guidance_phase_dp import (
    GuidedCompletionConfiguration,
    RuntimeGuidancePhaseLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_information_gain_selector import (
    RuntimeInformationGainLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_recovery_lookahead_selector import (
    RuntimeRecoveryLookaheadLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_obligation_value_selector import (
    RuntimeObligationValueLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_jit_action_advantage import (
    FrozenRuntimeJITMemory,
)
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    ACTOR_RESERVE_JOINT_METHODS,
    ADAPTIVE_INFORMATION_GAIN_RUNTIME_DEV_METHOD,
    TRACEDP_RECOVERY_LOOKAHEAD_DEV_METHOD,
    TRACEDP_OBLIGATION_VALUE_DEV_METHOD,
    BOUNDED_TRACEDP_DEV_METHODS,
    BOUNDED_TRACEDP_DEV_V4_METHOD,
    BOUNDED_TRACEDP_DEV_V6_METHOD,
    BOUNDED_TRACEDP_DEV_V7_METHOD,
    EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS,
    FAILURE_DIAGNOSIS_JOINT_METHODS,
    FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD,
    JIT_JOINT_METHODS,
    PLANNED_JOINT_METHODS,
    ROTATING_SOURCE_JOINT_METHODS,
    RPC_JOINT_METHODS,
    RUNTIME_OPTION_CHAIN_METHODS,
    RUNTIME_OWNED_OPTION_DP_DEV_METHODS,
    SEMANTIC_STABILITY_JOINT_METHODS,
    SETTLEMENT_REVIEW_JOINT_METHODS,
    TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
    TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD,
    TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD,
    TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_DEV_METHOD,
    TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_V2_METHOD,
    RuntimeJointSlatePolicy,
)
from lh_platform.test_time_adaptation.runtime_option_jit import (
    FrozenRuntimeOptionJITMemory,
    JITV2EstimateAdapter,
    RuntimeOptionJITContinuationContract,
    RuntimeOwnedOptionJITLiveResolver,
    RuntimeOwnedOptionJITV2LiveResolver,
)
from lh_platform.test_time_adaptation.runtime_option_jit_v2 import (
    FrozenRuntimeOptionJITMemoryV2,
)
from lh_platform.test_time_adaptation.runtime_output_obligation import (
    RuntimeInterventionEffectLedger,
)
from lh_platform.test_time_adaptation.runtime_owned_option_live import (
    RuntimeFrozenAtomicEstimateConfiguration,
    RuntimeOwnedOptionLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_public_prediction import (
    RUNTIME_PUBLIC_PREDICTION_SYSTEM_PROMPT,
    PublicRuntimePredictionProvider,
)
from lh_platform.test_time_adaptation.runtime_residual_jit import (
    RuntimeResidualJITResolver,
)
from lh_platform.test_time_adaptation.runtime_settlement_trace_dp import (
    RuntimeSettlementTraceDPResolver,
)
from lh_platform.test_time_adaptation.runtime_trace_dp import (
    DynamicTraceDPRuntimePolicy,
)
from lh_platform.test_time_adaptation.terminal_controlled_replay import (
    compile_terminal_replay_interventions,
)
from lh_platform.tracefix_bench.scorer_contracts import ResourceRole
from lh_platform.training.pi_alfworld_runtime import PiAuthenticatedOpenAiActor

_OBSERVER_REMOTE_PATH = "/tmp/longh-terminal-public-state-observer.ts"
# Pi emits each structured tool result as one JSONL record; image payloads can
# exceed the default buffer while the surrounding session remains bounded.
# asyncio's StreamReader.readline() aborts with "Separator is not found, and chunk
# exceed the limit" once a single line passes this bound.  A resumed long-horizon session
# can replay a conversation state whose JSON exceeds 32 MiB (the dna-insert Full run at
# B=12 died exactly that way), so keep generous headroom.
_PI_STDOUT_STREAM_LIMIT = 256 * 1024 * 1024
# Match Terminal-Bench's official per-task actor budget; this bounds a whole Pi session.
_DEFAULT_HOST_PI_TIMEOUT_SECONDS = 900
# Bound one response so a single verbose turn cannot consume the full task budget.
_DEFAULT_HOST_PI_MAX_OUTPUT_TOKENS = 16384
_ACTOR_SESSION_EVENT_SCHEMA = "terminal_managed_actor_session_event_v1"
_MANAGED_AGENT_IMPORT = "lh_platform.benchmarks.harbor_opencode_go_pi:HostManagedOpenCodeGoPi"
_FAILURE_DIAGNOSIS_RUNTIME_METHODS = FAILURE_DIAGNOSIS_JOINT_METHODS
_MANAGED_RUNTIME_CONFIG_V1 = "terminal_managed_runtime_config_v1"
_MANAGED_RUNTIME_CONFIG_V2 = "terminal_managed_runtime_config_v2"
# One shared continuation trigger keeps direct / delayed / strong-fixer / TraceDP
# comparable: only the Runtime's action choice may differ between those arms.
HISTORICAL_MANAGED_CONTINUATION_INSTRUCTION = (
    "Continue the original task from its observed failed final state. Inspect the "
    "current sandbox, correct any remaining problem, and finish the task. Do not "
    "assume the prior attempt's final answer is correct."
)
HISTORICAL_MANAGED_REPLAY_SCHEMA = "terminal_historical_managed_prefix_replay_v1"

HISTORICAL_MANAGED_CONTINUATION_SCHEMA = (
    "terminal_historical_managed_pi_session_continuation_v1"
)


def _terminal_runtime_adapter_config() -> dict[str, str]:
    """Return the explicit adapter identity introduced by config schema V2."""

    return {
        "adapter_id": TERMINAL_RUNTIME_ADAPTER_V1,
        "dataset_family": "terminal",
        "instantiation_scope": "initial_kernel_inputs_v1",
    }


def _actor_failure_diagnosis_threshold(runtime_method: str) -> int | None:
    """Bind each diagnosis source identity to its frozen failure threshold."""

    if runtime_method not in _FAILURE_DIAGNOSIS_RUNTIME_METHODS:
        return None
    if runtime_method in EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS:
        return 1
    return ACTOR_FAILURE_DIAGNOSIS_THRESHOLD


class HostPiSessionInterrupted(RuntimeError):
    """Report a retryable Pi subprocess failure while Harbor's environment is live."""


def _runtime_prediction_accounting(path: Path) -> dict[str, object]:
    """Aggregate completed planning calls and preserve unknown interrupted cost."""

    if not path.is_file():
        return {
            "model_call_count": 0,
            "admitted_prediction_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "elapsed_ms": 0,
            "token_accounting_complete": True,
            "ambiguous_started_call": False,
        }
    pending = False
    terminal_rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("runtime prediction receipt is not an object")
        if row.get("status") == "started":
            if pending:
                raise ValueError("runtime prediction calls overlap")
            pending = True
            continue
        if not pending:
            raise ValueError("runtime prediction terminal receipt lacks a start")
        terminal_rows.append(row)
        pending = False
    usages = [row.get("model_usage") for row in terminal_rows]
    complete = all(isinstance(usage, dict) for usage in usages) and not pending

    def total(field: str) -> int:
        return sum(
            int(usage.get(field, 0))
            for usage in usages
            if isinstance(usage, dict)
        )

    return {
        "model_call_count": len(terminal_rows),
        "admitted_prediction_count": sum(
            row.get("status") == "prediction_admitted_uncalibrated"
            for row in terminal_rows
        ),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "elapsed_ms": sum(int(row.get("elapsed_ms", 0)) for row in terminal_rows),
        "token_accounting_complete": complete,
        "ambiguous_started_call": pending,
    }


def _build_bounded_runtime_resolver(
    *,
    runtime_method: str,
    planner_actor: object,
    kernel: PortableRuntimeKernel,
    receipt_path: Path,
    configured_check_contracts: tuple[RuntimeCheckContract, ...] = (),
) -> (
    RuntimeBoundedTraceDPResolver
    | RuntimeBoundedTraceDPV4Resolver
    | RuntimeBoundedTraceDPV6Resolver
    | RuntimeBoundedTraceDPV7Resolver
    | RuntimeEventTraceDPV8Resolver
):
    """Select the saved-model compiler matching one bounded method identity."""

    provider = PublicRuntimePredictionProvider(
        actor=planner_actor,
        public_state=lambda: kernel.public_projection(),
        receipt_path=receipt_path,
        forecast_contract=(
            "compact_root_verdict_terminal_aggregate_event_v8"
            if runtime_method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
            else "compact_root_verdict_terminal_aggregate_v7"
            if runtime_method == BOUNDED_TRACEDP_DEV_V7_METHOD
            else (
                "flat_parent_coordinates_v6"
                if runtime_method == BOUNDED_TRACEDP_DEV_V6_METHOD
                else (
                    "verdict_actor_time_v4"
                    if runtime_method == BOUNDED_TRACEDP_DEV_V4_METHOD
                    else "runtime_accounted_v3"
                )
            )
        ),
        configured_check_contracts=tuple(
            runtime_check_configuration_payload(
                configured_check_contracts,
                diagnosis_enabled=False,
            )["contracts"]
        ),
        actor_convergence_reserve_ms=(
            ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000
            if runtime_method in {
                BOUNDED_TRACEDP_DEV_V7_METHOD,
                TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
            }
            else None
        ),
        provider_timeout_overrun_margin_ms=(
            V7_PROVIDER_TIMEOUT_OVERRUN_MARGIN_MS
            if runtime_method in {
                BOUNDED_TRACEDP_DEV_V7_METHOD,
                TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
            }
            else 0
        ),
        event_decision_path=(
            receipt_path.with_name("runtime-decisions.jsonl")
            if runtime_method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
            else None
        ),
        cumulative_planner_budget_ms=(
            V8_CUMULATIVE_PLANNER_BUDGET_MS
            if runtime_method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
            else None
        ),
        semantic_review_threshold=(
            V8_SEMANTIC_REVIEW_THRESHOLD
            if runtime_method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
            else None
        ),
    )
    if runtime_method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD:
        return RuntimeEventTraceDPV8Resolver(provider)
    if runtime_method == BOUNDED_TRACEDP_DEV_V7_METHOD:
        return RuntimeBoundedTraceDPV7Resolver(provider)
    if runtime_method == BOUNDED_TRACEDP_DEV_V6_METHOD:
        return RuntimeBoundedTraceDPV6Resolver(provider)
    if runtime_method == BOUNDED_TRACEDP_DEV_V4_METHOD:
        return RuntimeBoundedTraceDPV4Resolver(provider)
    if runtime_method in BOUNDED_TRACEDP_DEV_METHODS:
        return RuntimeBoundedTraceDPResolver(provider)
    raise ValueError("bounded runtime resolver requires a bounded method")


def _read_actor_session_events(path: Path) -> list[dict[str, object]]:
    """Read a contiguous actor-attempt prefix before appending recovery events."""

    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    open_attempt: int | None = None
    completed = False
    expected_actor_session_id: str | None = None
    expected_environment_session_id: str | None = None
    attempt_count = 0
    for sequence, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        payload = row.get("payload") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != _ACTOR_SESSION_EVENT_SCHEMA
            or row.get("sequence") != sequence
            or not isinstance(payload, dict)
        ):
            raise ValueError("managed Actor session event log is invalid")
        actor_session_id = payload.get("actor_session_id")
        environment_session_id = payload.get("environment_session_id")
        attempt_index = payload.get("attempt_index")
        if (
            not isinstance(actor_session_id, str)
            or not actor_session_id
            or not isinstance(environment_session_id, str)
            or not environment_session_id
            or (
                expected_actor_session_id is not None
                and actor_session_id != expected_actor_session_id
            )
            or (
                expected_environment_session_id is not None
                and environment_session_id != expected_environment_session_id
            )
        ):
            raise ValueError("managed Actor session event identity differs")
        expected_actor_session_id = actor_session_id
        expected_environment_session_id = environment_session_id
        event_type = row.get("event_type")
        if event_type == "actor_attempt_started":
            if completed or open_attempt is not None or attempt_index != attempt_count:
                raise ValueError("managed Actor attempt start sequence differs")
            open_attempt = attempt_count
            attempt_count += 1
        elif event_type in {
            "actor_attempt_interrupted",
            "actor_attempt_completed",
            "actor_session_budget_exhausted",
        }:
            if open_attempt is None or attempt_index != open_attempt:
                raise ValueError("managed Actor attempt terminal event differs")
            open_attempt = None
            completed = event_type in {
                "actor_attempt_completed",
                "actor_session_budget_exhausted",
            }
        else:
            raise ValueError("managed Actor session event type is unsupported")
        rows.append(row)
    return rows


def _append_actor_session_event(path: Path, *, event_type: str, payload: dict[str, object]) -> None:
    """Persist one Pi-attempt lifecycle event before changing the next boundary."""

    sequence = len(_read_actor_session_events(path))
    with path.open("a") as output:
        output.write(
            json.dumps(
                {
                    "schema_version": _ACTOR_SESSION_EVENT_SCHEMA,
                    "sequence": sequence,
                    "event_type": event_type,
                    "payload": payload,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _reconcile_interrupted_actor_receipts(path: Path, *, kernel: PortableRuntimeKernel) -> None:
    """Join completion-ordered receipts to proposed actions, then close gaps."""

    existing = (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.is_file()
        else []
    )
    actions = [action for action in kernel.actions.values() if action.actor == "actor"]
    actions_by_id = {action.action_id: action for action in actions}
    if len(existing) > len(actions):
        raise ValueError("terminal observer history exceeds Kernel actor actions")
    observed_ids: set[str] = set()
    for sequence, row in enumerate(existing):
        action_id = row.get("tool_call_id") if isinstance(row, dict) else None
        action = actions_by_id.get(action_id) if isinstance(action_id, str) else None
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != ACTOR_TOOL_RECEIPT_SCHEMA
            or row.get("sequence") != sequence
            or action is None
            or action_id in observed_ids
            or row.get("tool_name") != action.tool_name
            or row.get("input") != dict(action.arguments)
        ):
            raise ValueError("terminal observer history differs from Kernel actor actions")
        observed_ids.add(action_id)
    # Parallel proposals are ordered when Pi emits them, while gateway receipts
    # are ordered when execution or rejection completes. Join them by call ID.
    missing = [action for action in actions if action.action_id not in observed_ids]
    if any(
        action.phase not in {ActionPhase.BLOCKED, ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT}
        for action in missing
    ):
        raise ValueError("completed Kernel Actor action lacks its gateway receipt")
    if not missing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        for action in missing:
            sequence = len(existing)
            output.write(
                json.dumps(
                    {
                        "schema_version": ACTOR_TOOL_RECEIPT_SCHEMA,
                        "sequence": sequence,
                        "tool_call_id": action.action_id,
                        "tool_name": action.tool_name,
                        "input": dict(action.arguments),
                        "overlapping_tool_calls": False,
                        "environment_step_executed": (
                            False if action.phase is ActionPhase.BLOCKED else None
                        ),
                        "observation_complete": False,
                        "before": [],
                        "after": [],
                        "is_error": True,
                        "latency_ms": {},
                        "tool_result": None,
                        "error_stage": "host_runtime_interruption",
                        "error_type": "HostRuntimeRestart",
                        "recovery_reason": action.reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            existing.append({"tool_call_id": action.action_id})


async def _stop_async_process_group(process: asyncio.subprocess.Process) -> None:
    """Stop Pi and every provider subprocess in its isolated process group."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.returncode is None:
            await process.wait()
        return
    if process.returncode is None:
        try:
            async with asyncio.timeout(1):
                await process.wait()
        except TimeoutError:
            pass
    # The Pi leader may exit before a provider child. Kill the isolated group
    # even after the leader has a return code so background work cannot leak.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.returncode is None:
        await process.wait()


def _terminal_rpc_parent() -> Path:
    """Keep ephemeral Unix sockets below the macOS path-length limit."""

    root = Path(tempfile.gettempdir()) / "longh-terminal-rpc"
    root.mkdir(parents=True, exist_ok=True)
    return root


class OpenCodeGoPi(Pi):
    """Pi with an explicit OpenCode Go API-key environment contract."""

    MODEL_CONNECTION = ModelConnectionSpec(
        api_key_envs=("OPENCODE_API_KEY",),
        passthrough=True,
    )


class ObservedOpenCodeGoPi(OpenCodeGoPi):
    """Pi with native tools preserved and public filesystem observation enabled."""

    CLI_FLAGS = [
        *Pi.CLI_FLAGS,
        CliFlag(
            "public_state_observer",
            cli="--extension",
            type="str",
            default=_OBSERVER_REMOTE_PATH,
        ),
    ]

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        source = Path(__file__).with_name("terminal_public_state_observer.ts")
        await self._upload_agent_owned_file(environment, source, _OBSERVER_REMOTE_PATH)


class HostDelegatedOpenCodeGoPi(BaseAgent):
    """Run Pi on the host and delegate only terminal tools to the task container."""

    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)

    def __init__(
        self,
        *args,
        public_paths: list[str],
        pi_version: str = "0.84.1",
        thinking: str = "off",
        host_pi_timeout_seconds: int = _DEFAULT_HOST_PI_TIMEOUT_SECONDS,
        host_pi_max_output_tokens: int = _DEFAULT_HOST_PI_MAX_OUTPUT_TOKENS,
        observe_public_state: bool = True,
        security_probe_only: bool = False,
        neutral_actor_cwd: bool = False,
        provider_http_capture: bool = False,
        actor_budget_ledger_path: str | None = None,
        actor_upstream_base_url: str | None = None,
        batched_public_snapshot: bool = False,
        post_actor_image_tag: str | None = None,
        budget_affordability_gate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Harness knob shared by every arm of a comparison (matrix "observer" block).
        self.batched_public_snapshot = batched_public_snapshot
        # Repairs that would leave the Actor's next call unfunded are not chosen.
        self.budget_affordability_gate = budget_affordability_gate
        # Harness: the post-Actor filesystem is kept as this image, so a verifier lost to
        # the network can be rerun on the exact state instead of repeating the Actor.
        self.post_actor_image_tag = post_actor_image_tag
        if type(neutral_actor_cwd) is not bool:
            raise ValueError("neutral_actor_cwd must be boolean")
        # Pi reports its working directory to the actor.  With the default (the host
        # trial log directory) the actor sees run and task identity and addresses host
        # paths that do not exist in the container; new configs should enable this.
        self.neutral_actor_cwd = neutral_actor_cwd
        # Route the actor through the arm's shared cross-role budget pool.  The real
        # upstream must be explicit because the proxy replaces Pi's provider URL.
        if (actor_budget_ledger_path is None) != (actor_upstream_base_url is None):
            raise ValueError("actor budget needs both a ledger path and the upstream base URL")
        self.actor_budget_ledger_path = actor_budget_ledger_path
        self.actor_upstream_base_url = actor_upstream_base_url
        if type(provider_http_capture) is not bool:
            raise ValueError("provider_http_capture must be boolean")
        self.provider_http_capture = provider_http_capture
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("host Pi agent requires an explicit provider/model name")
        provider, model = self.model_name.split("/", maxsplit=1)
        self.runtime = resolve_host_pi_runtime(
            provider=provider,
            model=model,
            thinking=thinking,
        )
        if provider != self.runtime.provider:
            raise ValueError("host Pi agent provider does not match its frozen runtime")
        self.public_paths = tuple(public_paths)
        self.pi_version = pi_version
        self.thinking = thinking
        if not 1 <= host_pi_timeout_seconds <= MAX_HOST_PI_TIMEOUT_SECONDS:
            raise ValueError("host Pi timeout is outside the supported range")
        if host_pi_max_output_tokens < 1:
            raise ValueError("host Pi max output tokens must be positive")
        self.host_pi_timeout_seconds = host_pi_timeout_seconds
        self.host_pi_max_output_tokens = host_pi_max_output_tokens
        self.observe_public_state = observe_public_state
        self.security_probe_only = security_probe_only

    @staticmethod
    @override
    def name() -> str:
        return AgentName.PI.value

    @override
    def version(self) -> str:
        return self.pi_version

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    def _actor_cwd(self) -> Path:
        """Return the directory Pi runs in, which Pi reports to the actor.

        The neutral directory is stable per trial (so session lookup by id keeps
        working across resumes) and its name carries no run or task identity.
        """

        if not self.neutral_actor_cwd:
            return self.logs_dir
        return neutral_actor_cwd(self.logs_dir)

    def _pi_executable(self) -> Path:
        configured = os.environ.get("LONGH_PI_EXECUTABLE")
        resolved = configured or shutil.which("pi")
        if not resolved:
            raise RuntimeError("LONGH_PI_EXECUTABLE or pi on PATH is required")
        return Path(resolved).resolve()

    async def _security_probe(
        self,
        gateway: TerminalEnvironmentToolGateway,
        context: AgentContext,
    ) -> None:
        command = """
found=0
for file in /proc/[0-9]*/environ; do
  [ -r "$file" ] || continue
  if tr '\\0' '\\n' < "$file" | grep -q '^OPENCODE_API_KEY='; then found=1; fi
done
if [ "$found" -eq 0 ]; then
  echo CREDENTIAL_ISOLATED
else
  echo CREDENTIAL_LEAK
  exit 42
fi
""".strip()
        result = await gateway.execute(
            tool_call_id="credential-isolation-probe",
            tool_name="bash",
            arguments={"command": command, "timeout": 30},
        )
        passed = result.details.get("return_code") == 0 and "CREDENTIAL_ISOLATED" in result.content
        receipt = {
            "status": "passed" if passed else "failed",
            "model_called": False,
            "task_container_has_opencode_key": not passed,
            "tool_transport": "host_pi_unix_rpc",
        }
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "credential-isolation.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        context.metadata = receipt
        if not passed:
            raise RuntimeError("task container can observe the OpenCode Go credential")

    async def _commit_post_actor_state(self, environment: BaseEnvironment) -> None:
        """Keep the post-Actor filesystem as an image; never fails the run."""

        tag = getattr(self, "post_actor_image_tag", None)
        if not tag:
            return
        from lh_platform.benchmarks.terminal_prefix_snapshot import commit_container, container_id

        started = time.monotonic()
        receipt: dict[str, object] = {"schema_version": "terminal_post_actor_image_v1", "image_tag": tag}
        try:
            receipt["image_id"] = await commit_container(await container_id(environment), tag)
            receipt["status"] = "committed"
        except Exception as error:  # the run's own result must not depend on this harness step
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        receipt["seconds"] = round(time.monotonic() - started, 3)
        (self.logs_dir / "post-actor-image.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        bash_capability = await probe_bash_containment_capability(environment)
        (self.logs_dir / "bash-containment-capability.json").write_text(
            json.dumps(bash_capability.payload(), ensure_ascii=False, indent=2) + "\n"
        )
        self._managed_gateway = gateway = TerminalEnvironmentToolGateway(
            environment=environment,
            batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
            public_paths=self.public_paths,
            observer_path=self.logs_dir / "terminal-public-state.jsonl",
            scratch_root=self.logs_dir / "delegated-scratch",
            observe_public_state=self.observe_public_state,
            bash_containment_capability=bash_capability,
        )
        if self.security_probe_only:
            await self._security_probe(gateway, context)
            return
        await self._continue_with_gateway(
            instruction=instruction,
            gateway=gateway,
            context=context,
        )

    async def _continue_with_gateway(
        self,
        *,
        instruction: str,
        gateway: TerminalEnvironmentToolGateway,
        context: AgentContext,
    ) -> None:
        """Expose one prepared task environment to a bounded host Pi session."""

        attempt_deadline = getattr(
            self,
            "_managed_session_deadline_monotonic",
            time.monotonic() + self.host_pi_timeout_seconds,
        )
        if attempt_deadline <= time.monotonic():
            raise HostPiSessionInterrupted("managed Actor session budget was exhausted")
        gateway.begin_actor_attempt(deadline_monotonic=attempt_deadline)
        with tempfile.TemporaryDirectory(dir=_terminal_rpc_parent()) as rpc_directory:
            socket_path = Path(rpc_directory) / "tools.sock"
            server = TerminalToolRpcServer(socket_path=socket_path, gateway=gateway)
            await server.start()
            try:
                call = {
                    "instruction": instruction,
                    "socket_path": socket_path,
                    "context": context,
                    "gateway": gateway,
                }
                if hasattr(self, "_managed_actor_session_id"):
                    call["session_id"] = self._managed_actor_session_id
                if hasattr(self, "_managed_actor_session_file"):
                    call["session_file"] = self._managed_actor_session_file
                    # Bind every attempt to the file's CURRENT pre-prompt content: a
                    # resumed attempt sees the messages earlier attempts appended, and
                    # the frozen source count aborted it with
                    # "host Pi RPC session identity differs" (dna-insert Full, P5b).
                    call["expected_session_message_count"] = active_session_message_count(
                        self._managed_actor_session_file
                    )
                elif hasattr(self, "_managed_actor_expected_message_count"):
                    call["expected_session_message_count"] = (
                        self._managed_actor_expected_message_count
                    )
                await self._run_host_pi(**call)
            finally:
                await server.close()
                gateway.end_actor_attempt()
                if gateway.fatal_containment_error is not None:
                    raise gateway.fatal_containment_error

    async def _run_isolated_actor_guidance(
        self,
        environment,
        arguments,
        timeout_seconds: int,
        *,
        context: AgentContext,
    ):
        """Run clone-only guidance and charge its received usage to the owning execution."""

        if getattr(self, "_isolated_actor_trial_active", False):
            raise RuntimeError("isolated actor guidance cannot nest")
        public_instruction = getattr(self, "_managed_public_instruction", "")
        guidance_text = arguments.get("guidance_text")
        if not isinstance(public_instruction, str) or not public_instruction.strip():
            raise RuntimeError("isolated actor guidance requires the live public instruction")
        if not isinstance(guidance_text, str) or not guidance_text.strip():
            raise RuntimeError("isolated actor guidance requires frozen guidance text")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or timeout_seconds < 1
        ):
            raise RuntimeError("isolated actor guidance requires a positive time limit")
        session_id = str(getattr(environment, "session_id", "") or "isolated")
        main_logs = self.logs_dir
        trial_logs = main_logs / "isolated-guidance-trials" / candidate_environment_session_id(
            session_id, "guidance"
        )
        trial_logs.mkdir(parents=True, exist_ok=True)
        instruction = isolated_actor_guidance_instruction(
            public_instruction=public_instruction,
            guidance_text=guidance_text,
        )
        saved_deadline = getattr(self, "_managed_session_deadline_monotonic", None)
        isolated_deadline = time.monotonic() + timeout_seconds
        if saved_deadline is not None:
            isolated_deadline = min(saved_deadline, isolated_deadline)
        saved_session_attrs = {}
        for attr in (
            "_managed_actor_session_id",
            "_managed_actor_session_file",
            "_managed_actor_expected_message_count",
        ):
            if hasattr(self, attr):
                saved_session_attrs[attr] = getattr(self, attr)
                delattr(self, attr)
        bash_capability = await probe_bash_containment_capability(environment)
        gateway = TerminalEnvironmentToolGateway(
            environment=environment,
            batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
            public_paths=self.public_paths,
            observer_path=trial_logs / "terminal-public-state.jsonl",
            scratch_root=trial_logs / "delegated-scratch",
            observe_public_state=True,
            bash_containment_capability=bash_capability,
        )
        self._isolated_actor_trial_active = True
        self.logs_dir = trial_logs
        self._managed_session_deadline_monotonic = isolated_deadline
        trial_context = AgentContext()
        execution_status = "interrupted"
        try:
            await self._continue_with_gateway(
                instruction=instruction,
                gateway=gateway,
                context=trial_context,
            )
            execution_status = "completed"
            actor_tool_calls = 0
            observer = trial_logs / "terminal-public-state.jsonl"
            if observer.is_file():
                for line in observer.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("tool_name") in {"bash", "read", "write"}:
                        actor_tool_calls += 1
            return isolated_guidance_execution(
                content="isolated actor guidance completed",
                is_error=False,
                environment_step_executed=True,
                return_code=0,
                actor_tool_calls=actor_tool_calls,
                actor_session_id=session_id,
            )
        except Exception as error:
            execution_status = "failed"
            return isolated_guidance_execution(
                content=f"{type(error).__name__}: {error}",
                is_error=True,
                environment_step_executed=False,
                return_code=None,
                actor_tool_calls=0,
                actor_session_id=session_id,
            )
        finally:
            self.logs_dir = main_logs
            if saved_deadline is None:
                delattr(self, "_managed_session_deadline_monotonic")
            else:
                self._managed_session_deadline_monotonic = saved_deadline
            for attr, value in saved_session_attrs.items():
                setattr(self, attr, value)
            self._isolated_actor_trial_active = False
            # A timeout or cancellation can still have completed provider messages.
            for field in ("n_input_tokens", "n_output_tokens", "n_cache_tokens"):
                value = getattr(trial_context, field)
                if value is not None:
                    setattr(context, field, (getattr(context, field) or 0) + value)
            usage_complete = (
                execution_status == "completed"
                and all(
                    getattr(trial_context, field) is not None
                    for field in ("n_input_tokens", "n_output_tokens", "n_cache_tokens")
                )
            )
            if not usage_complete:
                self._managed_token_accounting_complete = False
            usage = {
                "schema_version": "isolated_guidance_usage_v1",
                "scope": "current_host_invocation",
                "actor_session_id": session_id,
                "logs_dir": str(trial_logs),
                "execution_status": execution_status,
                "input_tokens_including_cache": trial_context.n_input_tokens,
                "cache_read_tokens": trial_context.n_cache_tokens,
                "output_tokens": trial_context.n_output_tokens,
                "cost_usd": trial_context.cost_usd,
                "token_accounting_complete": usage_complete,
            }
            metadata = dict(context.metadata or {})
            metadata["isolated_guidance_usage"] = [
                *metadata.get("isolated_guidance_usage", []), usage
            ]
            context.metadata = metadata
            # Append one current-call receipt; old receipts are never recharged on resume.
            with (main_logs / "isolated-guidance-usage.jsonl").open("a") as stream:
                stream.write(json.dumps(usage, ensure_ascii=False) + "\n")

    async def _consume_host_pi_rpc(
        self,
        *,
        process: asyncio.subprocess.Process,
        output: object,
        receipt: PiStreamReceipt,
        instruction: str,
        gateway: TerminalEnvironmentToolGateway,
        attempt_index: int,
        deadline: float,
        session_id: str,
    ) -> int:
        """Drive Pi RPC until natural settlement or a verified v4 submission."""

        assert process.stdin is not None
        assert process.stdout is not None
        if self.budget_affordability_gate and self.actor_budget_ledger_path is not None:
            from lh_platform.tracefix_bench.budget_affordability import next_reservation_estimate
            from lh_platform.tracefix_bench.scorer_contracts import ResourceRole
            from lh_platform.tracefix_bench.shared_budget import FileSharedBudgetLedger

            ledger = FileSharedBudgetLedger(Path(str(self.actor_budget_ledger_path)))

            def budget_gate(visible_chars: int) -> tuple[int, int]:
                # Room is the ledger's own admission quantity; the Actor's next request is the
                # largest settled Actor request so far plus what the last exchange appended.
                last_output = receipt.usage_events[-1].output_tokens if receipt.usage_events else 0
                return ledger.input_room(), next_reservation_estimate(
                    ledger.max_settled_input(ResourceRole.ACTOR), visible_chars, last_output
                )

            gateway.attach_budget_gate(
                budget_gate, log_path=self.logs_dir / "budget-gate-events.jsonl"
            )
        prompt_request_id = f"actor-prompt-{attempt_index:03d}"
        abort_request_id = f"runtime-submit-{attempt_index:03d}"
        state_request_id = f"actor-state-{attempt_index:03d}"
        await _send_host_pi_rpc_command(
            process.stdin,
            request_id=state_request_id,
            command_type="get_state",
        )
        await _send_host_pi_rpc_command(
            process.stdin,
            request_id=prompt_request_id,
            command_type="prompt",
            message=instruction,
        )
        trigger_at = deadline - ACTOR_CONVERGENCE_RESERVE_SECONDS
        timer_task: asyncio.Task[None] | None = asyncio.create_task(
            asyncio.sleep(max(0.0, trigger_at - time.monotonic()))
        )
        read_task: asyncio.Task[bytes] | None = asyncio.create_task(process.stdout.readline())
        submission = None
        submission_closed = False
        stdin_closed = False
        reserve_open = False
        boundary_index = 0
        settlement_review = None
        settlement_review_attempted = False
        settlement_review_closed = False
        settlement_review_request_id = f"runtime-settlement-review-{attempt_index:03d}"
        settlement_review_enabled = (
            getattr(self, "runtime_method", None) in SETTLEMENT_REVIEW_JOINT_METHODS
        )
        # Option-chain methods decide again where the Actor settles on its own, so
        # the run does not end with unspent budget at the point a check matters most.
        natural_decision_enabled = (
            getattr(self, "runtime_method", None) in RUNTIME_OPTION_CHAIN_METHODS
        )
        natural_decision_round = 0
        natural_decision_pending: str | None = None
        natural_decision_max_rounds = 6

        async def request_submission() -> None:
            """Recheck state only at the timer or a completed tool event."""

            nonlocal submission, boundary_index
            submission = await gateway.begin_actor_state_submission(
                attempt_index=attempt_index,
                boundary_index=boundary_index,
                actor_time_budget_ms=self.host_pi_timeout_seconds * 1000,
            )
            boundary_index += 1
            if submission is not None:
                await _send_host_pi_rpc_command(
                    process.stdin,
                    request_id=abort_request_id,
                    command_type="abort",
                )

        try:
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                while read_task is not None:
                    awaited = {read_task}
                    if timer_task is not None:
                        awaited.add(timer_task)
                    done, _ = await asyncio.wait(awaited, return_when=asyncio.FIRST_COMPLETED)
                    if timer_task is not None and timer_task in done:
                        reserve_open = True
                        await request_submission()
                        timer_task = None
                    if read_task not in done:
                        continue
                    line = read_task.result()
                    if not line:
                        read_task = None
                        break
                    output.write(line)
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        event = None
                    if isinstance(event, dict):
                        receipt.observe(event)
                        if (
                            settlement_review is not None
                            and not settlement_review_closed
                            and event.get("type") == "response"
                            and event.get("id") == settlement_review_request_id
                        ):
                            accepted = (
                                event.get("command") == "prompt" and event.get("success") is True
                            )
                            await gateway.finish_actor_settlement_review(
                                settlement_review,
                                accepted=accepted,
                                reason=(
                                    None if accepted else "Pi rejected the settlement review prompt"
                                ),
                            )
                            settlement_review_closed = True
                            if not accepted and not stdin_closed:
                                process.stdin.close()
                                await process.stdin.wait_closed()
                                stdin_closed = True
                        if (
                            reserve_open
                            and submission is None
                            and (
                                getattr(
                                    getattr(gateway, "joint_runtime_policy", None),
                                    "remaining_budget_units",
                                    1,
                                )
                                > 0
                            )
                            and event.get("type") == "tool_execution_end"
                        ):
                            await request_submission()
                    if (
                        submission is not None
                        and not submission_closed
                        and receipt.controlled_abort_is_settled(
                            prompt_request_id=prompt_request_id,
                            abort_request_id=abort_request_id,
                        )
                    ):
                        await gateway.finish_actor_state_submission(submission, completed=True)
                        submission_closed = True
                        # A verified abort and settled receipt is the terminal
                        # submission boundary, even for methods that also support review.
                        if not stdin_closed:
                            process.stdin.close()
                            await process.stdin.wait_closed()
                            stdin_closed = True
                    if (
                        isinstance(event, dict)
                        and event.get("type") == "agent_settled"
                        and submission is None
                        and settlement_review_enabled
                    ):
                        if not settlement_review_attempted:
                            settlement_review_attempted = True
                            settlement_review = await gateway.begin_actor_settlement_review(
                                attempt_index=attempt_index,
                                review_index=0,
                            )
                            if settlement_review is not None:
                                await _send_host_pi_rpc_command(
                                    process.stdin,
                                    request_id=settlement_review_request_id,
                                    command_type="prompt",
                                    message=settlement_review.notice,
                                )
                            elif not stdin_closed:
                                process.stdin.close()
                                await process.stdin.wait_closed()
                                stdin_closed = True
                        elif settlement_review_closed and not stdin_closed:
                            process.stdin.close()
                            await process.stdin.wait_closed()
                            stdin_closed = True
                    if (
                        isinstance(event, dict)
                        and event.get("type") == "response"
                        and natural_decision_pending is not None
                        and event.get("id") == natural_decision_pending
                    ):
                        natural_decision_pending = None
                        if not (event.get("command") == "prompt" and event.get("success") is True):
                            if not stdin_closed:
                                process.stdin.close()
                                await process.stdin.wait_closed()
                                stdin_closed = True
                    if (
                        isinstance(event, dict)
                        and event.get("type") == "agent_settled"
                        and submission is None
                        and natural_decision_enabled
                        and not stdin_closed
                    ):
                        prompt = None
                        if natural_decision_round < natural_decision_max_rounds:
                            reentry = getattr(self, "supervisor_reentry", None)
                            reentry_notice = (
                                await reentry.at_settlement(gateway.environment)
                                if reentry is not None
                                else None
                            )
                            if reentry_notice is not None:
                                # The supervisor acted (restored inputs, or evidence again);
                                # the v2 runtime decides at the next settlement.
                                prompt = ActorNaturalSettlementPrompt(
                                    trigger_action_id=(
                                        f"supervisor-reentry-{attempt_index:03d}"
                                        f"-{natural_decision_round:03d}"
                                    ),
                                    notice=reentry_notice,
                                    interventions=(),
                                )
                            else:
                                prompt = await gateway.decide_at_natural_settlement(
                                    attempt_index=attempt_index,
                                    round_index=natural_decision_round,
                                )
                            natural_decision_round += 1
                        if prompt is not None:
                            natural_decision_pending = (
                                f"runtime-natural-settlement-{attempt_index:03d}"
                                f"-{natural_decision_round - 1:03d}"
                            )
                            await _send_host_pi_rpc_command(
                                process.stdin,
                                request_id=natural_decision_pending,
                                command_type="prompt",
                                message=prompt.notice,
                            )
                        else:
                            process.stdin.close()
                            await process.stdin.wait_closed()
                            stdin_closed = True
                    prompt_accepted = any(
                        item.request_id == prompt_request_id
                        and item.command == "prompt"
                        and item.success
                        for item in receipt.rpc_responses
                    )
                    if (
                        not stdin_closed
                        and not settlement_review_enabled
                        and not natural_decision_enabled
                        and prompt_accepted
                        and receipt.agent_settled_sequences
                        and (submission is None or submission_closed)
                    ):
                        process.stdin.close()
                        await process.stdin.wait_closed()
                        stdin_closed = True
                    read_task = asyncio.create_task(process.stdout.readline())
                return await process.wait()
        except TimeoutError:
            if settlement_review is not None and not settlement_review_closed:
                await gateway.finish_actor_settlement_review(
                    settlement_review,
                    accepted=False,
                    reason="Pi did not acknowledge the settlement review prompt",
                )
                settlement_review_closed = True
            if submission is not None and not submission_closed:
                await gateway.finish_actor_state_submission(
                    submission,
                    completed=False,
                    reason="Pi did not return a complete abort and settled receipt",
                )
                submission_closed = True
            raise
        finally:
            for task in (timer_task, read_task):
                if task is not None and not task.done():
                    task.cancel()

    async def _run_host_pi(
        self,
        *,
        instruction: str,
        socket_path: Path,
        context: AgentContext,
        session_id: str | None = None,
        session_file: Path | None = None,
        expected_session_message_count: int | None = None,
        gateway: TerminalEnvironmentToolGateway | None = None,
    ) -> None:
        executable = self._pi_executable()
        version_process = await asyncio.create_subprocess_exec(
            str(executable),
            "--version",
            stdout=asyncio.subprocess.PIPE,
        )
        version_stdout, _ = await version_process.communicate()
        if version_process.returncode != 0 or version_stdout.decode().strip() != self.pi_version:
            raise RuntimeError(f"host Pi must be version {self.pi_version}")

        extension = Path(__file__).with_name("terminal_delegating_tools.ts")
        session_dir = self.logs_dir / "pi" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        if not environment.get(self.runtime.credential_env):
            raise RuntimeError(
                f"host Pi requires {self.runtime.credential_env} for {self.runtime.model_name}"
            )
        # Pi persists discovered credentials under PI_CODING_AGENT_DIR. Keep that
        # directory off the storage volume so experiment artifacts never contain API secrets.
        with contextlib.ExitStack() as budget_scope, tempfile.TemporaryDirectory(
            prefix="longh-pi-config-"
        ) as config_directory:
            budget_proxy = (
                budget_scope.enter_context(
                    SharedBudgetProviderProxy(
                        upstream_base_url=str(self.actor_upstream_base_url),
                        ledger_path=Path(str(self.actor_budget_ledger_path)),
                        role=ResourceRole.ACTOR,
                    )
                )
                if self.actor_budget_ledger_path is not None
                else None
            )
            _write_host_pi_models_config(
                config_dir=Path(config_directory),
                max_output_tokens=self.host_pi_max_output_tokens,
                runtime=self.runtime,
                **({"base_url": budget_proxy.base_url} if budget_proxy is not None else {}),
            )
            environment["PI_CODING_AGENT_DIR"] = config_directory
            environment["LH_TERMINAL_TOOL_SOCKET"] = str(socket_path)
            environment["LH_TERMINAL_PROVIDER_REQUEST_AUDIT"] = str(
                self.logs_dir / "provider-request-audit.jsonl"
            )
            environment["LH_TERMINAL_PROVIDER_REQUEST_ATTEMPT"] = str(
                getattr(self, "_managed_actor_attempt_index", 0)
            )
            if self.provider_http_capture:
                capture_path = self.logs_dir / "provider-http-capture.jsonl"
                capture_start_offset = capture_path.stat().st_size if capture_path.exists() else 0
                environment["LH_TERMINAL_PROVIDER_HTTP_CAPTURE"] = str(capture_path)
            else:
                environment.pop("LH_TERMINAL_PROVIDER_HTTP_CAPTURE", None)
            if gateway is not None and gateway.allow_check_registration:
                environment["LH_TERMINAL_REGISTER_CHECKS"] = "1"
                assert gateway.runtime_kernel is not None
                environment["LH_TERMINAL_RUNTIME_RESOURCE_IDS"] = json.dumps(
                    list(gateway.runtime_kernel.resource_versions)
                )
            else:
                environment.pop("LH_TERMINAL_REGISTER_CHECKS", None)
                environment.pop("LH_TERMINAL_RUNTIME_RESOURCE_IDS", None)
            if gateway is not None and gateway.require_check_registration_phase:
                environment["LH_TERMINAL_REQUIRE_CHECK_REGISTRATION_PHASE"] = "1"
            else:
                environment.pop("LH_TERMINAL_REQUIRE_CHECK_REGISTRATION_PHASE", None)
            rpc_submission = (
                getattr(self, "runtime_method", None) in RPC_JOINT_METHODS
                and not getattr(self, "_isolated_actor_trial_active", False)
            )
            if rpc_submission and gateway is None:
                raise RuntimeError("RPC Actor Bridge requires the live tool gateway")
            if rpc_submission and session_id is None:
                raise RuntimeError("RPC Actor Bridge requires the Pi session identity")
            command_builder = _host_pi_rpc_command if rpc_submission else _host_pi_command
            command = command_builder(
                executable=executable,
                extension=extension,
                session_dir=session_dir,
                runtime=self.runtime,
                session_id=None if session_file is not None else session_id,
                session_file=session_file,
                allow_check_registration=gateway is not None and gateway.allow_check_registration,
                require_check_registration_phase=(
                    gateway is not None and gateway.require_check_registration_phase
                ),
            )
            attempt_index = getattr(self, "_managed_actor_attempt_index", None)
            stream_stem = (
                f"pi-attempt-{attempt_index:03d}" if isinstance(attempt_index, int) else "pi"
            )
            stderr_path = self.logs_dir / f"{stream_stem}.stderr.txt"
            output_path = self.logs_dir / f"{stream_stem}.txt"
            with stderr_path.open("wb") as stderr:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=self._actor_cwd(),
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=stderr,
                    limit=_PI_STDOUT_STREAM_LIMIT,
                    start_new_session=True,
                )
                assert process.stdin is not None
                assert process.stdout is not None
                receipt = PiStreamReceipt(
                    expected_provider=self.runtime.provider,
                    expected_model=self.runtime.model,
                )
                try:
                    with output_path.open("wb") as output:
                        try:
                            # Bound the complete agent session so an upstream request cannot
                            # leave Harbor and a scientific run waiting indefinitely.
                            deadline = getattr(
                                self,
                                "_managed_session_deadline_monotonic",
                                time.monotonic() + self.host_pi_timeout_seconds,
                            )
                            remaining_seconds = deadline - time.monotonic()
                            if remaining_seconds <= 0:
                                raise HostPiSessionInterrupted(
                                    "managed Actor session budget was exhausted"
                                )
                            async with asyncio.timeout(remaining_seconds):
                                if rpc_submission:
                                    assert gateway is not None
                                    return_code = await self._consume_host_pi_rpc(
                                        process=process,
                                        output=output,
                                        receipt=receipt,
                                        instruction=instruction,
                                        gateway=gateway,
                                        attempt_index=(
                                            attempt_index if isinstance(attempt_index, int) else 0
                                        ),
                                        deadline=deadline,
                                        session_id=session_id,
                                    )
                                else:
                                    # Print mode consumes one opaque prompt and then exits.
                                    await _send_host_pi_instruction(process.stdin, instruction)
                                    async for line in process.stdout:
                                        output.write(line)
                                        try:
                                            event = json.loads(line)
                                        except json.JSONDecodeError:
                                            continue
                                        if isinstance(event, dict):
                                            receipt.observe(event)
                                    return_code = await process.wait()
                        except TimeoutError as error:
                            if hasattr(self, "_managed_token_accounting_complete"):
                                self._managed_token_accounting_complete = False
                            # Report the attempt's own deadline: an isolated guidance trial
                            # runs on a shorter bound than the host budget.
                            raise HostPiSessionInterrupted(
                                f"host Pi session exceeded its {remaining_seconds:.0f}s "
                                f"attempt deadline (host budget "
                                f"{self.host_pi_timeout_seconds}s)"
                            ) from error
                finally:
                    # Failed and timed-out calls still consumed model tokens.
                    context.n_input_tokens = (context.n_input_tokens or 0) + (
                        receipt.input_tokens + receipt.cache_tokens
                    )
                    context.n_output_tokens = (context.n_output_tokens or 0) + (
                        receipt.output_tokens
                    )
                    context.n_cache_tokens = (context.n_cache_tokens or 0) + (receipt.cache_tokens)
                    await _stop_async_process_group(process)
        if return_code != 0:
            if hasattr(self, "_managed_token_accounting_complete"):
                self._managed_token_accounting_complete = False
            raise HostPiSessionInterrupted(
                f"host Pi exited with code {return_code}: {stderr_path.read_text()}"
            )
        try:
            controlled_abort = rpc_submission and any(
                item.command == "abort" and item.success for item in receipt.rpc_responses
            )
            if controlled_abort:
                receipt.require_controlled_abort(
                    prompt_request_id=f"actor-prompt-{attempt_index or 0:03d}",
                    abort_request_id=f"runtime-submit-{attempt_index or 0:03d}",
                )
            if rpc_submission:
                assert session_id is not None
                receipt.require_rpc_session_identity(
                    request_id=f"actor-state-{attempt_index or 0:03d}",
                    expected_session_id=session_id,
                    expected_message_count=expected_session_message_count,
                    expected_session_file=session_file,
                )
            receipt.require_valid(
                return_code=return_code,
                controlled_abort_verified=controlled_abort,
            )
            if self.provider_http_capture:
                require_terminal_provider_http_capture(capture_path, start_offset=capture_start_offset)
        except (OSError, RuntimeError, ValueError) as error:
            if hasattr(self, "_managed_token_accounting_complete"):
                self._managed_token_accounting_complete = False
            raise HostPiSessionInterrupted(f"host Pi receipt is incomplete: {error}") from error
        context.metadata = {
            **dict(context.metadata or {}),
            "provider": self.runtime.provider,
            "model": self.runtime.model,
            "pi_version": self.pi_version,
            "thinking": self.thinking,
            "transport": self.runtime.transport,
            "reasoning_tokens": receipt.reasoning_tokens,
            "max_output_tokens": self.host_pi_max_output_tokens,
            "credential_location": "host_only",
            "public_state_observation_enabled": self.observe_public_state,
            "public_path_count": len(self.public_paths),
        }


class HostManagedOpenCodeGoPi(HostDelegatedOpenCodeGoPi):
    """Run one Pi session through the shared resource and task-state kernel."""

    def __init__(
        self,
        *args,
        resource_specs: list[dict[str, object]],
        retention_budget_bytes: int,
        restore_missing_prerequisites: bool,
        runtime_method: str,
        runtime_intervention_budget: int = 0,
        runtime_planning_horizon_cap: int | None = None,
        runtime_residual_structure: str = "task_linked",
        runtime_experience_mode: str = "none",
        runtime_progress_trigger_threshold: int | None = None,
        joint_source_arm_index: int | None = None,
        jit_memory_path: str | None = None,
        jit_test_task_id: str | None = None,
        actor_resume_limit: int = 1,
        discover_database_sidecars: bool = False,
        task_contract_compiler: str = TASK_CONTRACT_COMPILER_V3,
        runtime_check_contracts: list[dict[str, object]] | None = None,
        runtime_predicate_repair_contracts: list[dict[str, object]] | None = None,
        runtime_acceptance_predicate_manifest: list[dict[str, object]] | None = None,
        runtime_acceptance_predicate_semantic_review: dict[str, object] | None = None,
        run_class: str | None = None,
        predicate_diagnosis_enabled: bool = False,
        allow_check_registration: bool = False,
        concurrent_runtime_observation: bool = False,
        require_check_registration_phase: bool = False,
        runtime_planner_base_url: str | None = None,
        runtime_planner_max_output_tokens: int = 4096,
        runtime_planner_timeout_seconds: int = 180,
        runtime_atomic_estimate_configuration: dict[str, object] | None = None,
        runtime_guided_completion_configuration: dict[str, object] | None = None,
        runtime_acceptance_candidate_trials: bool = False,
        runtime_docker_candidate_provider: dict[str, object] | None = None,
        # A0 calibration: after the frozen prefix replay, execute every registered check and one
        # isolated trial per repair candidate, then stop -- never staging a continuation.
        runtime_calibration_mode: bool = False,
        runtime_calibration_negative_control: bool = False,
        runtime_calibration_source_task_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if retention_budget_bytes < 1:
            raise ValueError("managed runtime retention budget must be positive")
        if runtime_method not in SUPPORTED_RUNTIME_METHODS:
            raise ValueError("managed runtime method identity is unsupported")
        if restore_missing_prerequisites != (runtime_method == SHARED_INPUT_PROTECTION_METHOD):
            raise ValueError("managed runtime method and repair policy differ")
        if runtime_method in INTERVENTION_BUDGET_METHODS:
            if runtime_intervention_budget < 1:
                raise ValueError("managed runtime policy requires a positive budget")
        elif runtime_intervention_budget != 0:
            raise ValueError("only a managed policy may receive an intervention budget")
        planning_horizons = (1, 2, 3) if runtime_method == TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD else (1, 2)
        if runtime_planning_horizon_cap is not None and runtime_planning_horizon_cap not in planning_horizons:
            raise ValueError("runtime planning horizon cap is unsupported")
        if runtime_residual_structure not in ("task_linked", "flat"):
            raise ValueError("runtime residual structure mode is unsupported")
        if runtime_experience_mode not in ("none", "jit_live", "memory_prompt"):
            raise ValueError("runtime experience mode is unsupported")
        if runtime_progress_trigger_threshold is not None and (
            int(runtime_progress_trigger_threshold) < 1
        ):
            raise ValueError("progress trigger threshold must be positive")
        if runtime_experience_mode == "jit_live" and not jit_memory_path:
            raise ValueError("jit_live experience mode requires frozen JIT memory")
        if runtime_method in ROTATING_SOURCE_JOINT_METHODS or runtime_method == (
            FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD
        ):
            if (
                isinstance(joint_source_arm_index, bool)
                or not isinstance(joint_source_arm_index, int)
                or joint_source_arm_index < 0
            ):
                raise ValueError("rotating joint source requires an arm index")
        elif joint_source_arm_index is not None:
            raise ValueError("only a rotating joint source may receive an arm index")
        jit_method = runtime_method in JIT_JOINT_METHODS
        if jit_method:
            source_memory = Path(jit_memory_path or "")
            if (
                not source_memory.is_absolute()
                or not source_memory.is_file()
                or not isinstance(jit_test_task_id, str)
                or not jit_test_task_id
            ):
                raise ValueError("TraceDP JIT requires frozen memory and task identity")
            self.jit_memory_source_path = source_memory
            self.jit_test_task_id = jit_test_task_id
        elif jit_memory_path is not None or jit_test_task_id is not None:
            raise ValueError("only TraceDP JIT may receive frozen memory settings")
        else:
            self.jit_memory_source_path = None
            self.jit_test_task_id = None
        bounded_planner = runtime_method in BOUNDED_TRACEDP_DEV_METHODS
        if bounded_planner:
            if not isinstance(runtime_planner_base_url, str) or not runtime_planner_base_url:
                raise ValueError("bounded TraceDP requires an explicit planner endpoint")
            if runtime_planner_max_output_tokens < 1 or runtime_planner_timeout_seconds < 1:
                raise ValueError("bounded TraceDP planner limits must be positive")
        elif runtime_planner_base_url is not None:
            raise ValueError("only bounded TraceDP may receive a planner endpoint")
        self.runtime_planner_base_url = runtime_planner_base_url
        self.runtime_planner_max_output_tokens = runtime_planner_max_output_tokens
        self.runtime_planner_timeout_seconds = runtime_planner_timeout_seconds
        if runtime_method in RUNTIME_OWNED_OPTION_DP_DEV_METHODS:
            if not isinstance(runtime_atomic_estimate_configuration, dict):
                raise ValueError(
                    "Runtime-owned option DP requires frozen atomic estimates"
                )
            self.runtime_atomic_estimate_configuration = (
                RuntimeFrozenAtomicEstimateConfiguration.from_dict(
                    deepcopy(runtime_atomic_estimate_configuration)
                )
            )
        elif runtime_atomic_estimate_configuration is not None:
            raise ValueError(
                "only Runtime-owned option DP may receive atomic estimates"
            )
        else:
            self.runtime_atomic_estimate_configuration = None
        self.runtime_guided_completion_configuration = None
        if runtime_method == TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD:
            if run_class != "exposed_development_canary" or not isinstance(runtime_guided_completion_configuration, dict):
                raise ValueError("guidance phase planning requires explicit development continuation estimates")
            guided_configuration = GuidedCompletionConfiguration.from_dict(deepcopy(runtime_guided_completion_configuration))
            actor_cell_id = f"{self.model_name}|thinking={self.thinking}|transport={self.runtime.transport}"
            if guided_configuration.actor_cell_id != actor_cell_id:
                raise ValueError("guided continuation estimate differs from the frozen Actor identity")
            self.runtime_guided_completion_configuration = guided_configuration
        elif runtime_guided_completion_configuration is not None:
            raise ValueError("only guidance phase planning may receive continuation estimates")
        if type(runtime_acceptance_candidate_trials) is not bool:
            raise ValueError("candidate trial flag must be boolean")
        if runtime_acceptance_candidate_trials and (
            runtime_method not in JOINT_RUNTIME_METHODS
            or not runtime_predicate_repair_contracts
        ):
            # Isolated trial execution only needs a joint selector to own the
            # candidate action, so every joint method may use it.  Requiring a
            # Runtime-owned option DP here would make the delay and strong-fixer
            # comparators unable to run the same candidate as TraceDP.
            raise ValueError(
                "candidate trials require a joint Runtime selector and repair contracts"
            )
        self.runtime_acceptance_candidate_trials = runtime_acceptance_candidate_trials
        self.runtime_docker_candidate_provider = deepcopy(runtime_docker_candidate_provider)
        self.runtime_calibration_mode = runtime_calibration_mode
        self.runtime_calibration_negative_control = runtime_calibration_negative_control
        self.runtime_calibration_source_task_id = runtime_calibration_source_task_id
        if runtime_docker_candidate_provider is not None:
            from lh_platform.benchmarks.terminal_docker_candidate_provider import (
                DockerCandidateConfiguration,
            )

            if run_class != EXPOSED_DEVELOPMENT_CANARY_RUN_CLASS:
                raise ValueError("unverified Docker candidate provider is development-only")
            DockerCandidateConfiguration(**runtime_docker_candidate_provider)
        if actor_resume_limit < 0:
            raise ValueError("managed runtime actor resume limit cannot be negative")
        if type(discover_database_sidecars) is not bool:
            raise ValueError("database sidecar discovery flag must be boolean")
        if task_contract_compiler not in SUPPORTED_TASK_CONTRACT_COMPILERS:
            raise ValueError("task contract compiler identity is unsupported")
        specs = []
        for row in resource_specs:
            if set(row) != {"resource_id", "kind", "provenance", "group_id"}:
                raise ValueError("managed runtime resource schema differs")
            specs.append(
                ResourceSpec(
                    resource_id=str(row["resource_id"]),
                    kind=ResourceKind(row["kind"]),
                    provenance=str(row["provenance"]),
                    group_id=(str(row["group_id"]) if row["group_id"] is not None else None),
                )
            )
        if not specs:
            raise ValueError("managed runtime requires resource declarations")
        self.resource_specs = (
            expand_sqlite_resource_family_specs(specs)
            if discover_database_sidecars
            else tuple(specs)
        )
        if discover_database_sidecars:
            # Observation scope must include discovered metadata even when exact
            # retention or later restoration is unavailable.
            self.public_paths = tuple(
                dict.fromkeys(
                    (*self.public_paths, *(item.resource_id for item in self.resource_specs))
                )
            )
        self.discover_database_sidecars = discover_database_sidecars
        self.task_contract_compiler = task_contract_compiler
        self.runtime_dataset_adapter = TerminalRuntimeDatasetAdapter()
        self.retention_budget_bytes = retention_budget_bytes
        self.restore_missing_prerequisites = restore_missing_prerequisites
        self.concurrent_runtime_observation = concurrent_runtime_observation
        self.runtime_method = runtime_method
        self.runtime_intervention_budget = runtime_intervention_budget
        self.runtime_planning_horizon_cap = runtime_planning_horizon_cap
        self.runtime_residual_structure = runtime_residual_structure
        self.runtime_experience_mode = runtime_experience_mode
        self.runtime_progress_trigger_threshold = runtime_progress_trigger_threshold
        self.joint_source_arm_index = joint_source_arm_index
        self.actor_resume_limit = actor_resume_limit
        if (
            type(predicate_diagnosis_enabled) is not bool
            or type(allow_check_registration) is not bool
            or type(require_check_registration_phase) is not bool
        ):
            raise ValueError("managed runtime check flags must be boolean")
        if require_check_registration_phase and not allow_check_registration:
            raise ValueError("required check registration phase needs Actor registration")
        if predicate_diagnosis_enabled and runtime_method not in JOINT_RUNTIME_METHODS:
            raise ValueError("predicate diagnosis requires a joint runtime selector")
        self.runtime_check_contracts = tuple(
            RuntimeCheckContract(
                **{
                    **deepcopy(row),
                    "resource_ids": tuple(row.get("resource_ids", ())),
                }
            )
            for row in (runtime_check_contracts or [])
        )
        self.runtime_predicate_repair_contracts = tuple(
            RuntimePredicateRepairContract(
                **{
                    **deepcopy(row),
                    "resource_ids": tuple(row.get("resource_ids", ())),
                }
            )
            for row in (runtime_predicate_repair_contracts or [])
        )
        repair_ids = [
            item.repair_id for item in self.runtime_predicate_repair_contracts
        ]
        if len(repair_ids) != len(set(repair_ids)):
            raise ValueError("targeted repair contract identities repeat")
        if self.runtime_predicate_repair_contracts and (
            runtime_method not in JOINT_RUNTIME_METHODS
            or runtime_intervention_budget < 1
        ):
            raise ValueError("targeted repairs require a budgeted joint selector")
        if any(
            is_actor_guidance_repair_arguments(item.arguments)
            for item in self.runtime_predicate_repair_contracts
        ) and not runtime_acceptance_candidate_trials:
            raise ValueError("actor guidance repairs require isolated candidate trials")
        check_scopes = [
            (item.task_id, item.proposition, item.resource_ids)
            for item in self.runtime_check_contracts
        ]
        if any(
            check_scopes.count((item.task_id, item.proposition, item.resource_ids)) != 1
            for item in self.runtime_predicate_repair_contracts
        ):
            raise ValueError("targeted repair lacks one unambiguous configured check")
        self.runtime_repair_configuration = (
            runtime_repair_configuration_payload(
                self.runtime_predicate_repair_contracts
            )
            if self.runtime_predicate_repair_contracts
            else None
        )
        if runtime_acceptance_predicate_manifest is not None and not isinstance(
            runtime_acceptance_predicate_manifest, list
        ):
            raise ValueError("runtime acceptance predicate manifest must be a list")
        self.runtime_acceptance_predicate_manifest = (
            deepcopy(runtime_acceptance_predicate_manifest)
            if runtime_acceptance_predicate_manifest is not None
            else None
        )
        if self.runtime_acceptance_predicate_manifest is not None:
            if run_class not in {
                EXPOSED_DEVELOPMENT_CANARY_RUN_CLASS,
                FORMAL_EXPERIMENT_RUN_CLASS,
            }:
                raise ValueError("acceptance predicate run class is required")
            if (
                run_class == FORMAL_EXPERIMENT_RUN_CLASS
                and ACCEPTANCE_PREDICATE_IMPLEMENTATION_STATUS
                != "implementation_verified"
            ):
                raise ValueError(
                    "unverified acceptance predicates cannot enter a formal experiment"
                )
            review = runtime_acceptance_predicate_semantic_review
            predicate_ids = [
                row.get("predicate_id")
                for row in self.runtime_acceptance_predicate_manifest
                if isinstance(row, dict)
            ]
            reviewed_check_contracts = runtime_check_configuration_payload(
                self.runtime_check_contracts,
                diagnosis_enabled=False,
            )["contracts"]
            if (
                not isinstance(review, dict)
                or set(review)
                != {
                    "schema_version",
                    "review_scope",
                    "reviewer",
                    "public_instruction",
                    "predicate_ids",
                    "reviewed_manifest",
                    "reviewed_check_contracts",
                    "semantic_entailment_reviewed",
                    "hidden_evaluator_or_reference_answer_used",
                    "baseline_access_policy",
                }
                or review.get("schema_version")
                != "runtime_acceptance_predicate_public_semantic_review_v2"
                or review.get("review_scope")
                != "public_instruction_and_explicit_manifest_only"
                or not isinstance(review.get("reviewer"), str)
                or not str(review["reviewer"]).strip()
                or not isinstance(review.get("public_instruction"), str)
                or review.get("predicate_ids") != predicate_ids
                or review.get("reviewed_manifest")
                != self.runtime_acceptance_predicate_manifest
                or review.get("reviewed_check_contracts")
                != reviewed_check_contracts
                or review.get("semantic_entailment_reviewed") is not True
                or review.get("hidden_evaluator_or_reference_answer_used") is not False
                or review.get("baseline_access_policy")
                != "same_manifest_and_checks_for_all_methods"
            ):
                raise ValueError(
                    "acceptance predicate canary lacks its public semantic review"
                )
        elif run_class is not None:
            raise ValueError("acceptance predicate run class requires a manifest")
        elif runtime_acceptance_predicate_semantic_review is not None:
            raise ValueError("acceptance predicate semantic review requires a manifest")
        self.run_class = run_class
        self.runtime_acceptance_predicate_semantic_review = (
            deepcopy(runtime_acceptance_predicate_semantic_review)
            if runtime_acceptance_predicate_semantic_review is not None
            else None
        )
        if self.runtime_acceptance_predicate_manifest is not None and any(
            contract.check_id is None for contract in self.runtime_check_contracts
        ):
            raise ValueError(
                "acceptance predicate runs require bound configured checks"
            )
        if self.runtime_acceptance_predicate_manifest is not None:
            manifest_by_id = {
                row.get("predicate_id"): row
                for row in self.runtime_acceptance_predicate_manifest
                if isinstance(row, dict)
            }
            if any(
                not isinstance(row, dict)
                or not isinstance(row.get("predicate_id"), str)
                for row in self.runtime_acceptance_predicate_manifest
            ) or len(manifest_by_id) != len(self.runtime_acceptance_predicate_manifest):
                raise ValueError("acceptance predicate manifest identities are invalid")
            for contract in self.runtime_check_contracts:
                predicate = manifest_by_id.get(contract.predicate_id)
                if (
                    predicate is None
                    or contract.task_id != predicate.get("task_id")
                    or contract.proposition != predicate.get("proposition")
                    or list(contract.resource_ids) != predicate.get("resource_ids")
                ):
                    raise ValueError(
                        "configured check contract differs from acceptance predicate"
                    )
        self.predicate_diagnosis_enabled = predicate_diagnosis_enabled
        self.allow_check_registration = allow_check_registration
        self.require_check_registration_phase = require_check_registration_phase
        self.runtime_check_configuration = (
            runtime_check_configuration_payload(
                self.runtime_check_contracts, diagnosis_enabled=predicate_diagnosis_enabled,
                registration_enabled=allow_check_registration,
                registration_phase_required=require_check_registration_phase,
            )
            if (
                self.runtime_check_contracts
                or predicate_diagnosis_enabled
                or allow_check_registration
                or require_check_registration_phase
            )
            else None
        )

    def _actor_runtime_identity(self) -> dict[str, object]:
        """Freeze every setting that determines the actual host Pi execution path."""

        identity = {
            "agent_import": _MANAGED_AGENT_IMPORT,
            "model_name": self.model_name,
            "provider": self.runtime.provider,
            "model": self.runtime.model,
            "thinking": self.thinking,
            "transport": self.runtime.transport,
            "pi_version": self.pi_version,
            "host_pi_timeout_seconds": self.host_pi_timeout_seconds,
            "host_pi_max_output_tokens": self.host_pi_max_output_tokens,
            "observe_public_state": self.observe_public_state,
            "public_path_count": len(self.public_paths),
        }
        if self.provider_http_capture:
            identity["provider_http_capture"] = "terminal_provider_http_capture_v1"
        if self.runtime_method in RPC_JOINT_METHODS:
            identity["actor_control_mode"] = "pi_jsonl_rpc"
        if self.runtime_method in BOUNDED_TRACEDP_DEV_METHODS:
            identity["runtime_planner"] = {
                "provider": self.runtime.provider,
                "model": self.runtime.model,
                "transport": "pi_authenticated_openai_chat_completion",
                "base_url": self.runtime_planner_base_url,
                "maximum_output_tokens": self.runtime_planner_max_output_tokens,
                "timeout_seconds": self.runtime_planner_timeout_seconds,
                "thinking": "off",
                "response_format": "json_object",
                **(
                    {
                        "precall_actor_convergence_reserve_ms": (
                            ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000
                        ),
                        "precall_provider_timeout_overrun_margin_ms": (
                            V7_PROVIDER_TIMEOUT_OVERRUN_MARGIN_MS
                        ),
                        "timeout_is_hard_wall_clock": False,
                    }
                    if self.runtime_method in {
                        BOUNDED_TRACEDP_DEV_V7_METHOD,
                        TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD,
                    }
                    else {}
                ),
                **(
                    {
                        "event_admission_schema": "runtime_planner_event_admission_v1",
                        "cumulative_planner_budget_ms": (
                            V8_CUMULATIVE_PLANNER_BUDGET_MS
                        ),
                        "semantic_review_threshold": (
                            V8_SEMANTIC_REVIEW_THRESHOLD
                        ),
                    }
                    if self.runtime_method == TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
                    else {}
                ),
            }
        return identity

    def _reconcile_open_runtime_decision(
        self,
        *,
        kernel: PortableRuntimeKernel,
        runtime_supervisor: DynamicTraceDPRuntimePolicy | None,
    ) -> None:
        """Close one interrupted policy decision against its exact Kernel action."""

        if runtime_supervisor is None or not runtime_supervisor.has_open_decision:
            return
        resource_id = runtime_supervisor.open_resource_id
        open_cost_units = runtime_supervisor.open_cost_units
        assert resource_id is not None and open_cost_units is not None
        executed_action_id = runtime_supervisor.open_executed_action_id
        if executed_action_id is None:
            runtime_supervisor.abandon_interrupted_before_execution(
                reason="runtime process restarted before environment execution started"
            )
            return
        action = kernel.actions.get(executed_action_id)
        if action is None:
            runtime_supervisor.abandon_interrupted_before_execution(
                reason=("runtime process restarted before the reserved Kernel action was durable")
            )
            return
        if action.actor != "runtime" or action.arguments.get("resource_id") != resource_id:
            raise ValueError("runtime decision execution differs from Kernel history")
        resumed_status = (
            "completed"
            if action.phase is ActionPhase.COMPLETED
            else "cancelled_with_unknown_effect"
            if action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
            else "failed"
        )
        runtime_supervisor.record_outcome(
            action_id=executed_action_id,
            resource_id=resource_id,
            cost_units=open_cost_units,
            status=resumed_status,
            reason=(
                None
                if resumed_status == "completed"
                else action.reason or "runtime ended without a durable policy outcome"
            ),
            resource_version_after=kernel.resource_versions[resource_id],
        )

    @staticmethod
    def _reconcile_open_joint_runtime_decision(
        *,
        kernel: PortableRuntimeKernel,
        joint_runtime_policy: RuntimeJointSlatePolicy | None,
    ) -> None:
        """Close an interrupted joint choice from its durable Kernel phase."""

        if joint_runtime_policy is None or not joint_runtime_policy.has_open_decision:
            return
        action_id = joint_runtime_policy.open_executed_action_id
        if action_id is None or action_id not in kernel.actions:
            joint_runtime_policy.abandon_before_environment_effect(
                reason="runtime process restarted before a Kernel action was durable"
            )
            return
        action = kernel.actions[action_id]
        expected_tool = {
            RuntimeJointActionType.RESTORE_RESOURCE: "restore_resource",
            RuntimeJointActionType.REQUEST_OUTPUT_COMPLETION: "runtime_guidance",
            RuntimeJointActionType.REQUEST_ACTOR_CONVERGENCE: ("runtime_convergence_guidance"),
            RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS: (
                "runtime_failure_diagnosis_guidance"
            ),
            RuntimeJointActionType.SUBMIT_CURRENT_STATE: "submit_current_state",
            RuntimeJointActionType.REQUEST_SEMANTIC_CHECK: ("runtime_semantic_guidance"),
            RuntimeJointActionType.REQUEST_REGISTERED_CHECK: (
                "runtime_registered_check_guidance"
            ),
            RuntimeJointActionType.RUN_REGISTERED_CHECK: (
                "runtime_registered_check_execution"
            ),
            RuntimeJointActionType.RUN_TARGETED_REPAIR: (
                "runtime_targeted_repair_execution"
            ),
            RuntimeJointActionType.REQUEST_PREDICATE_DIAGNOSIS: (
                "runtime_predicate_guidance"
            ),
        }.get(joint_runtime_policy.open_action_type)
        if action.actor != "runtime" or action.tool_name != expected_tool:
            raise ValueError("joint runtime execution differs from Kernel history")
        status = (
            "completed"
            if action.phase is ActionPhase.COMPLETED
            else "cancelled_with_unknown_effect"
            if action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
            else "failed"
        )
        joint_runtime_policy.record_outcome(
            status=status,
            reason=(
                None
                if status == "completed"
                else action.reason or "runtime ended without a durable policy outcome"
            ),
        )

    @staticmethod
    def _reconcile_uncommitted_v8_prediction(
        *,
        kernel: PortableRuntimeKernel,
        joint_runtime_policy: RuntimeJointSlatePolicy | None,
    ) -> None:
        """Commit a paid V8 result before any resumed Actor boundary can replace it."""

        if (
            joint_runtime_policy is None
            or joint_runtime_policy.method_id != TRACEDP_EVENT_RUNTIME_DEV_V8_METHOD
        ):
            return
        resolver = joint_runtime_policy.decision_resolver
        if not isinstance(resolver, RuntimeEventTraceDPV8Resolver) or not isinstance(
            resolver.model_provider, PublicRuntimePredictionProvider
        ):
            raise ValueError("v8 runtime policy lacks its public prediction provider")
        provider = resolver.model_provider
        recovered = provider.recover_uncommitted_v8_prediction()
        if recovered is None:
            return
        if joint_runtime_policy.has_open_decision:
            raise ValueError("v8 uncommitted prediction overlaps an open decision")
        slate, model = recovered
        admission = model.get("event_admission")
        expected = planner_event_admission_v8(
            slate,
            kernel.public_projection(),
            previously_admitted_event_keys=provider._previous_v8_event_keys(),
            semantic_review_threshold=int(provider.semantic_review_threshold),
            configured_check_contracts=provider.configured_check_contracts,
        )
        if admission != expected or expected.get("decision") != "call_provider":
            raise ValueError("v8 uncommitted prediction differs from the live Kernel boundary")
        joint_runtime_policy.select(slate)

    @staticmethod
    def _reconcile_intervention_effect_ledger(
        *,
        kernel: PortableRuntimeKernel,
        intervention_effect_ledger: RuntimeInterventionEffectLedger | None,
        decision_log_path: Path,
        actor_event_log_path: Path,
        pi_stream_root: Path,
        expected_provider: str,
        expected_model: str,
    ) -> None:
        """Represent recovered v4 interventions without inventing Pi delivery."""

        if intervention_effect_ledger is None:
            return
        actor_events = _read_actor_session_events(actor_event_log_path)
        interrupted_attempts = {
            int(row["payload"]["attempt_index"])
            for row in actor_events
            if row["event_type"] in {"actor_attempt_interrupted", "actor_session_budget_exhausted"}
        }
        call_attempt: dict[str, int] = {}
        execution_ids: set[str] = set()
        terminal_uncompleted_ids: set[str] = set()
        attempt_count = sum(row["event_type"] == "actor_attempt_started" for row in actor_events)
        for attempt_index in range(attempt_count):
            path = pi_stream_root / f"pi-attempt-{attempt_index:03d}.txt"
            if not path.is_file():
                continue
            receipt = load_pi_stream_receipt(
                path,
                expected_provider=expected_provider,
                expected_model=expected_model,
            )
            call_attempt.update({item.tool_call_id: attempt_index for item in receipt.tool_calls})
            execution_ids.update(item.tool_call_id for item in receipt.tool_executions)
            terminal_uncompleted_ids.update(
                item.tool_call_id
                for item in receipt.tool_calls
                if receipt.is_terminal_uncompleted_tool_call(item.tool_call_id)
            )

        trigger_by_execution: dict[str, str] = {}
        pending_trigger: str | None = None
        if decision_log_path.is_file():
            for line in decision_log_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                payload = row.get("payload") if isinstance(row, dict) else None
                if not isinstance(payload, dict):
                    continue
                if row.get("event_type") == "decision_planned":
                    trigger = payload.get("trigger_action_id")
                    pending_trigger = trigger if isinstance(trigger, str) else None
                elif row.get("event_type") == "decision_selected":
                    slate = payload.get("slate")
                    trigger = slate.get("trigger_action_id") if isinstance(slate, dict) else None
                    pending_trigger = trigger if isinstance(trigger, str) else None
                elif row.get("event_type") in {
                    "decision_execution_started",
                    "execution_started",
                }:
                    executed = payload.get("executed_action_id")
                    if isinstance(executed, str) and pending_trigger is not None:
                        trigger_by_execution[executed] = pending_trigger
                elif row.get("event_type") in {
                    "decision_outcome",
                    "decision_abandoned",
                    "execution_outcome",
                }:
                    pending_trigger = None

        recorded = {item.action_id: item for item in intervention_effect_ledger.records}
        for action_id, record in tuple(recorded.items()):
            if record.delivery_status.value != "delivered":
                continue
            trigger_action_id = trigger_by_execution.get(action_id)
            attempt_index = (
                call_attempt.get(trigger_action_id) if trigger_action_id is not None else None
            )
            if (
                trigger_action_id is not None
                and attempt_index in interrupted_attempts
                and trigger_action_id not in execution_ids
                and trigger_action_id in terminal_uncompleted_ids
            ):
                intervention_effect_ledger.correct_interrupted_delivery_without_pi_receipt(
                    action_id=action_id,
                    trigger_action_id=trigger_action_id,
                )
        recorded_ids = {item.action_id for item in intervention_effect_ledger.records}
        for action in kernel.actions.values():
            if action.actor != "runtime" or action.action_id in recorded_ids:
                continue
            if action.tool_name not in {
                "runtime_convergence_guidance",
                "runtime_failure_diagnosis_guidance",
                "runtime_semantic_guidance",
                "runtime_guidance",
            }:
                continue
            obligation_id = action.arguments.get("obligation_id")
            # v3 progress guidance predates the effect-ledger obligation contract.
            if action.tool_name == "runtime_guidance" and obligation_id is None:
                continue
            if not isinstance(obligation_id, str):
                raise ValueError("recovered runtime intervention lacks an obligation")
            intervention_effect_ledger.record_recovered_unknown_delivery(
                action_id=action.action_id,
                obligation_id=obligation_id,
                obligation_version_key=action.arguments.get("obligation_version_key"),
            )

    async def _environment_session_id(self, environment: BaseEnvironment) -> str:
        """Read the live container identity used across every managed action."""

        result = await environment.exec("cat /etc/hostname", timeout_sec=30)
        session_id = str(result.stdout or "").strip()
        if result.return_code != 0 or not session_id:
            raise RuntimeError("managed runtime cannot establish environment session identity")
        return session_id

    def _managed_private_logs_dir(self) -> Path:
        """Place supervisor state outside Harbor's Actor-visible log bind mount."""

        runtime_root = Path(os.environ.get("LONGH_RUNTIME_ROOT", ""))
        if not runtime_root.is_absolute():
            raise RuntimeError("managed runtime requires an absolute LONGH_RUNTIME_ROOT")
        return (
            runtime_root
            / "managed-runtime-private"
            / self.logs_dir.parent.parent.name
            / self.logs_dir.parent.name
            / "agent"
        )

    def _managed_actor_session_identity(self) -> str:
        """Choose the durable Actor session identity of a newly started trial.

        A fresh task starts a new Pi session.  A managed continuation of one
        historical failure must instead reuse the identity its Pi context file
        already carries, because Harbor's Actor receipt validators join the Pi
        stream session id to this durable runtime configuration value.
        """

        return str(uuid.uuid4())

    @override
    async def _open_candidate_environment(
        self, environment: BaseEnvironment
    ) -> tuple[object, BaseEnvironment]:
        """Wrap the live environment when the arm enables isolated candidate trials.

        Returns ``(provider, wrapped_environment)``; the provider is ``None`` when the arm
        disables candidate trials, and the caller closes a non-None provider in `finally`.
        The A0 calibration branch needs the same wrap as the live path because the isolated
        clone factory is the provider's capability.
        """

        if self.runtime_docker_candidate_provider is None:
            return None, environment
        from lh_platform.benchmarks.terminal_docker_candidate_provider import (
            DockerCandidateConfiguration,
            DockerCandidateEnvironmentBridge,
            DockerCurrentStateCandidateProvider,
        )

        provider = DockerCurrentStateCandidateProvider(
            environment,
            configuration=DockerCandidateConfiguration(
                **self.runtime_docker_candidate_provider
            ),
            state_dir=self.logs_dir / "docker-candidate-leases",
        )
        await provider.open()
        return provider, DockerCandidateEnvironmentBridge(provider)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Keep supervisor artifacts private until the Actor and environment close."""

        if (
            self.runtime_acceptance_predicate_semantic_review is not None
            and (
                self.runtime_acceptance_predicate_semantic_review.get(
                    "public_instruction"
                )
                != instruction
                or self.runtime_acceptance_predicate_semantic_review.get(
                    "reviewed_manifest"
                )
                != self.runtime_acceptance_predicate_manifest
                or self.runtime_acceptance_predicate_semantic_review.get(
                    "reviewed_check_contracts"
                )
                != runtime_check_configuration_payload(
                    self.runtime_check_contracts,
                    diagnosis_enabled=False,
                )["contracts"]
            )
        ):
            raise ValueError(
                "acceptance predicate semantic review differs at run entry"
            )
        public_logs_dir = self.logs_dir
        private_logs_dir = self._managed_private_logs_dir()
        private_logs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = private_logs_dir
        candidate_provider = None
        try:
            candidate_provider, environment = await self._open_candidate_environment(
                environment
            )
            await self._run_with_private_logs(instruction, environment, context)
            await self._commit_post_actor_state(environment)
        finally:
            if candidate_provider is not None:
                await candidate_provider.close()
            # Harbor's bind mount is Actor-visible, so publish only after Actor exit.
            public_logs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(private_logs_dir, public_logs_dir, dirs_exist_ok=True)
            self.logs_dir = public_logs_dir

    async def _build_candidate_trial_gate(
        self, environment: BaseEnvironment, session_id: str
    ) -> TerminalSeparateEnvironmentPortfolioGate | None:
        """Create the isolated current-state clone gate when candidate trials are enabled."""

        if not self.runtime_acceptance_candidate_trials:
            return None
        provider = getattr(environment, "create_isolated_candidate_environment", None)
        if not callable(provider):
            raise RuntimeError(
                "Harbor BaseEnvironment exposes no isolated current-state clone; "
                "candidate trials require a reviewed environment provider"
            )

        async def create_isolated_environment(trial_id: str):
            """Create a fresh disposable Harbor session for exactly one trial."""

            return await provider(
                session_id=candidate_environment_session_id(session_id, trial_id)
            )

        return TerminalSeparateEnvironmentPortfolioGate(
            main_environment=environment,
            isolated_environment_factory=create_isolated_environment,
            public_paths=self.public_paths,
            durable_root=self.logs_dir / "runtime-candidate-portfolio",
        )

    async def _run_with_private_logs(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Initialize, manage, persist, and close one continuous actor session."""

        # Managed retries share one Actor budget; initialization and recovery do not
        # reset the deadline or create extra model time.
        self._managed_public_instruction = instruction
        self._managed_session_deadline_monotonic = time.monotonic() + self.host_pi_timeout_seconds
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        bash_capability = await probe_bash_containment_capability(environment)
        session_id = await self._environment_session_id(environment)
        candidate_trial_gate = await self._build_candidate_trial_gate(
            environment, session_id
        )
        # Kept on self because the historical continuation subclass runs the A0 calibration
        # sweep from its own run() and needs the same clone gate and tool gateway.
        self._managed_candidate_trial_gate = candidate_trial_gate
        event_log_path = self.logs_dir / "runtime-events.jsonl"
        config_path = self.logs_dir / "runtime-config.json"
        retention_root = self.logs_dir / "runtime-retention"
        retention_receipt_path = self.logs_dir / "runtime-retention.jsonl"
        decision_log_path = self.logs_dir / "runtime-decisions.jsonl"
        effect_log_path = self.logs_dir / "runtime-intervention-effects.jsonl"
        jit_memory_artifact_path = self.logs_dir / "runtime-jit-memory.json"
        actor_event_log_path = self.logs_dir / "actor-session-events.jsonl"
        observer_path = self.logs_dir / "terminal-public-state.jsonl"
        resuming = config_path.is_file()
        if not resuming:
            # Configuration is the commit marker written before the Actor starts.
            # A prefix without it can only contain agent-owned initialization data.
            if event_log_path.is_file():
                partial_kernel = PortableRuntimeKernel.load(event_log_path)
                if partial_kernel.actions or partial_kernel.actor_session_closed:
                    raise ValueError("managed runtime history exists without its configuration")
            event_log_path.unlink(missing_ok=True)
            retention_receipt_path.unlink(missing_ok=True)
            decision_log_path.unlink(missing_ok=True)
            effect_log_path.unlink(missing_ok=True)
            jit_memory_artifact_path.unlink(missing_ok=True)
            actor_event_log_path.unlink(missing_ok=True)
            observer_path.unlink(missing_ok=True)
            config_path.with_suffix(".json.initializing").unlink(missing_ok=True)
            shutil.rmtree(retention_root, ignore_errors=True)
        jit_memory = None
        jit_memory_payload = None
        if self.runtime_method in JIT_JOINT_METHODS:
            if resuming:
                if not jit_memory_artifact_path.is_file():
                    raise ValueError("managed runtime JIT memory artifact is missing")
            else:
                assert self.jit_memory_source_path is not None
                jit_memory_payload = json.loads(self.jit_memory_source_path.read_text())
                jit_memory_artifact_path.write_text(
                    json.dumps(
                        jit_memory_payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            if self.runtime_method == TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_V2_METHOD:
                jit_memory = FrozenRuntimeOptionJITMemoryV2.load(
                    jit_memory_artifact_path
                )
            elif self.runtime_method == TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_DEV_METHOD:
                jit_memory = FrozenRuntimeOptionJITMemory.load(jit_memory_artifact_path)
            else:
                jit_memory = FrozenRuntimeJITMemory.load(jit_memory_artifact_path)
        planner_actor = None
        if self.runtime_method in BOUNDED_TRACEDP_DEV_METHODS:
            planner_raw_path = self.logs_dir / "runtime-public-prediction-provider.jsonl"

            def record_planner_raw(payload: dict[str, object], latency: float) -> None:
                """Persist the provider response before prediction parsing."""

                with planner_raw_path.open("a", encoding="utf-8") as output:
                    output.write(
                        json.dumps(
                            {"response": payload, "latency_seconds": latency},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            planner_actor = PiAuthenticatedOpenAiActor(
                executable=self._pi_executable(),
                provider=self.runtime.provider,
                model=self.runtime.model,
                base_url=str(self.runtime_planner_base_url),
                max_tokens=self.runtime_planner_max_output_tokens,
                timeout_seconds=self.runtime_planner_timeout_seconds,
                system_prompt=RUNTIME_PUBLIC_PREDICTION_SYSTEM_PROMPT,
                session_id=f"longh-runtime-planner-{session_id}",
                json_object_output=True,
                raw_response_recorder=record_planner_raw,
            )
        def build_joint_decision_resolver(runtime_kernel: PortableRuntimeKernel):
            """Bind live public state only after the durable Kernel is available."""

            if self.runtime_method == TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_V2_METHOD:
                # Verified-transition adaptation of the DP estimate model.  The
                # wrapper records both the base and the adapted Bellman passes so the receipt
                # proves the memory was loaded and moved the estimates.
                assert self.runtime_atomic_estimate_configuration is not None
                if not isinstance(jit_memory, FrozenRuntimeOptionJITMemoryV2):
                    raise ValueError("JIT v2 memory type differs")
                adapter = JITV2EstimateAdapter(
                    memory=jit_memory,
                    test_task_id=str(self.jit_test_task_id),
                    configured_checks=self.runtime_check_contracts,
                    repair_types={
                        str(item.repair_id): str(
                            item.arguments.get("candidate_form", "actor_guidance")
                        )
                        for item in self.runtime_predicate_repair_contracts
                    },
                )
                return RuntimeOwnedOptionJITV2LiveResolver(
                    base_resolver=RuntimeOwnedOptionLiveResolver(
                        configuration=self.runtime_atomic_estimate_configuration,
                        configured_checks=self.runtime_check_contracts,
                        public_projection=runtime_kernel.public_projection,
                        planning_horizon_cap=self.runtime_planning_horizon_cap,
                        residual_structure=self.runtime_residual_structure,
                        estimate_adapter=adapter,
                    ),
                    memory=jit_memory,
                    test_task_id=str(self.jit_test_task_id),
                    method_id=self.runtime_method,
                )
            if (
                self.runtime_method
                == TRACEDP_RUNTIME_OWNED_OPTION_DP_JIT_DEV_METHOD
            ) or self.runtime_experience_mode == "jit_live":
                assert self.runtime_atomic_estimate_configuration is not None
                if not isinstance(jit_memory, FrozenRuntimeOptionJITMemory):
                    raise ValueError("Runtime-owned option JIT memory type differs")
                continuation = RuntimeOptionJITContinuationContract(
                    continuation_policy_id=(
                        "tracedp_runtime_owned_option_dp_dev_v1"
                    ),
                    actor_cell_id=(
                        f"{self.model_name}|thinking={self.thinking}|"
                        f"transport={self.runtime.transport}"
                    ),
                    trigger_contract_id="runtime_option_chain_v1",
                    action_domain_contract_id="runtime_option_action_domain_v1",
                    objective_id=(
                        "residual_completion_then_unblocked_then_time_then_false_"
                        "interventions_then_intervention_cost_v1"
                    ),
                    budget_semantics_id="runtime_option_chain_budget_v1",
                )
                return RuntimeOwnedOptionJITLiveResolver(
                    base_resolver=RuntimeOwnedOptionLiveResolver(
                        configuration=self.runtime_atomic_estimate_configuration,
                        configured_checks=self.runtime_check_contracts,
                        public_projection=runtime_kernel.public_projection,
                        planning_horizon_cap=self.runtime_planning_horizon_cap,
                        residual_structure=self.runtime_residual_structure,
                    ),
                    memory=jit_memory,
                    test_task_id=str(self.jit_test_task_id),
                    continuation=continuation,
                )
            if self.runtime_method == TRACEDP_RECOVERY_LOOKAHEAD_DEV_METHOD:
                assert self.runtime_atomic_estimate_configuration is not None
                return RuntimeRecoveryLookaheadLiveResolver(
                    configuration=self.runtime_atomic_estimate_configuration,
                    configured_checks=self.runtime_check_contracts,
                    public_projection=runtime_kernel.public_projection,
                    residual_structure=self.runtime_residual_structure,
                )
            if self.runtime_method == TRACEDP_OBLIGATION_VALUE_DEV_METHOD:
                assert self.runtime_atomic_estimate_configuration is not None
                return RuntimeObligationValueLiveResolver(
                    configuration=self.runtime_atomic_estimate_configuration,
                    configured_checks=self.runtime_check_contracts,
                    public_projection=runtime_kernel.public_projection,
                    residual_structure=self.runtime_residual_structure,
                )
            if self.runtime_method == ADAPTIVE_INFORMATION_GAIN_RUNTIME_DEV_METHOD:
                assert self.runtime_atomic_estimate_configuration is not None
                return RuntimeInformationGainLiveResolver(
                    configuration=self.runtime_atomic_estimate_configuration,
                    configured_checks=self.runtime_check_contracts,
                    public_projection=runtime_kernel.public_projection,
                    residual_structure=self.runtime_residual_structure,
                )
            if self.runtime_method == TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD:
                return RuntimeGuidancePhaseLiveResolver(
                    configuration=self.runtime_atomic_estimate_configuration,
                    guided_configuration=self.runtime_guided_completion_configuration,
                    configured_checks=self.runtime_check_contracts,
                    public_projection=runtime_kernel.public_projection,
                    planning_horizon_cap=self.runtime_planning_horizon_cap,
                    residual_structure=self.runtime_residual_structure,
                )
            if self.runtime_method in RUNTIME_OWNED_OPTION_DP_DEV_METHODS:
                assert self.runtime_atomic_estimate_configuration is not None
                return RuntimeOwnedOptionLiveResolver(
                    configuration=self.runtime_atomic_estimate_configuration,
                    configured_checks=self.runtime_check_contracts,
                    public_projection=runtime_kernel.public_projection,
                    planning_horizon_cap=self.runtime_planning_horizon_cap,
                    residual_structure=self.runtime_residual_structure,
                )
            if self.runtime_method in BOUNDED_TRACEDP_DEV_METHODS:
                return _build_bounded_runtime_resolver(
                    runtime_method=self.runtime_method,
                    planner_actor=planner_actor,
                    kernel=runtime_kernel,
                    receipt_path=self.logs_dir / "runtime-public-predictions.jsonl",
                    configured_check_contracts=self.runtime_check_contracts,
                )
            if self.runtime_method == TRACEDP_JIT_RESIDUAL_RUNTIME_METHOD:
                return RuntimeResidualJITResolver(
                    method_id=self.runtime_method,
                    jit_memory=jit_memory,
                    test_task_id=str(self.jit_test_task_id),
                )
            if self.runtime_method in PLANNED_JOINT_METHODS:
                return RuntimeSettlementTraceDPResolver(
                    method_id=self.runtime_method,
                    jit_memory=jit_memory,
                    test_task_id=self.jit_test_task_id,
                )
            return None
        runtime_config = json.loads(config_path.read_text()) if resuming else None
        observation_schema = (
            str(
                runtime_config.get(
                    "resource_observation_schema",
                    TERMINAL_RESOURCE_OBSERVATION_V1,
                )
            )
            if isinstance(runtime_config, dict)
            else TERMINAL_RESOURCE_OBSERVATION_V2
        )
        driver = TerminalResourceDriver(
            environment=environment,
            resources=self.resource_specs,
            retention_root=retention_root,
            receipt_path=retention_receipt_path,
            retention_budget_bytes=self.retention_budget_bytes,
            observation_schema=observation_schema,
        )
        if resuming:
            assert isinstance(runtime_config, dict)
            recorded_schema = runtime_config.get("schema_version")
            recorded_adapter = runtime_config.get("runtime_dataset_adapter")
            recorded_adapter_migration = runtime_config.get(
                "runtime_dataset_adapter_migration"
            )
            legacy_adapter_config = (
                recorded_schema == _MANAGED_RUNTIME_CONFIG_V1
                and "runtime_dataset_adapter" not in runtime_config
            )
            current_adapter_config = (
                recorded_schema == _MANAGED_RUNTIME_CONFIG_V2
                and recorded_adapter == _terminal_runtime_adapter_config()
                and "runtime_dataset_adapter_migration" in runtime_config
                and recorded_adapter_migration
                in (
                    None,
                    {
                        "from_schema": _MANAGED_RUNTIME_CONFIG_V1,
                        "to_schema": _MANAGED_RUNTIME_CONFIG_V2,
                        "reason": "bind_existing_terminal_kernel_to_explicit_adapter",
                    },
                )
            )
            expected_resources = [
                {
                    "resource_id": spec.resource_id,
                    "kind": spec.kind.value,
                    "provenance": spec.provenance,
                    "group_id": spec.group_id,
                }
                for spec in self.resource_specs
            ]
            recorded_resources = [
                {key: row.get(key) for key in ("resource_id", "kind", "provenance", "group_id")}
                for row in runtime_config.get("resources", [])
                if isinstance(row, dict)
            ]
            if (
                not (legacy_adapter_config or current_adapter_config)
                or observation_schema
                != runtime_config.get(
                    "resource_observation_schema",
                    TERMINAL_RESOURCE_OBSERVATION_V1,
                )
                or runtime_config.get("runtime_method") != self.runtime_method
                or runtime_config.get("environment_session_id") != session_id
                or runtime_config.get("restore_missing_prerequisites")
                != self.restore_missing_prerequisites
                or runtime_config.get("retention_budget_bytes") != self.retention_budget_bytes
                or runtime_config.get("runtime_intervention_budget")
                != self.runtime_intervention_budget
                or runtime_config.get("joint_source_arm_index") != self.joint_source_arm_index
                or runtime_config.get("runtime_check_configuration") != self.runtime_check_configuration
                or runtime_config.get("runtime_repair_configuration")
                != self.runtime_repair_configuration
                or runtime_config.get("runtime_acceptance_predicate_manifest")
                != self.runtime_acceptance_predicate_manifest
                or runtime_config.get("runtime_acceptance_predicate_semantic_review")
                != self.runtime_acceptance_predicate_semantic_review
                or runtime_config.get("run_class") != self.run_class
                or runtime_config.get("runtime_docker_candidate_provider")
                != self.runtime_docker_candidate_provider
                or runtime_config.get("runtime_atomic_estimate_configuration")
                != (
                    self.runtime_atomic_estimate_configuration.to_dict()
                    if self.runtime_atomic_estimate_configuration is not None
                    else None
                )
                or runtime_config.get("runtime_acceptance_candidate_trials")
                != self.runtime_acceptance_candidate_trials
                or runtime_config.get("runtime_guided_completion_configuration")
                != (
                    self.runtime_guided_completion_configuration.to_dict()
                    if self.runtime_guided_completion_configuration is not None
                    else None
                )
                or runtime_config.get("jit_test_task_id") != self.jit_test_task_id
                or runtime_config.get("jit_memory_source_path")
                != (
                    str(self.jit_memory_source_path)
                    if self.jit_memory_source_path is not None
                    else None
                )
                or runtime_config.get("jit_memory_artifact")
                != (
                    "runtime-jit-memory.json"
                    if self.runtime_method in JIT_JOINT_METHODS
                    else None
                )
                or runtime_config.get("jit_memory_payload")
                != (
                    json.loads(jit_memory_artifact_path.read_text())
                    if self.runtime_method in JIT_JOINT_METHODS
                    else None
                )
                or runtime_config.get("joint_policy_schema")
                != (
                    RuntimeJointSlatePolicy.schema_version
                    if self.runtime_method in JOINT_RUNTIME_METHODS
                    else None
                )
                or runtime_config.get("output_progress_action_threshold")
                != (
                    OUTPUT_PROGRESS_ACTION_THRESHOLD
                    if self.runtime_method in OUTPUT_PROGRESS_RUNTIME_METHODS
                    else None
                )
                or runtime_config.get("actor_failure_diagnosis_threshold")
                != _actor_failure_diagnosis_threshold(self.runtime_method)
                or runtime_config.get("output_semantic_stability_action_threshold")
                != (
                    OUTPUT_SEMANTIC_STABILITY_ACTION_THRESHOLD
                    if self.runtime_method in SEMANTIC_STABILITY_JOINT_METHODS
                    else None
                )
                or runtime_config.get("actor_convergence_reserve_seconds")
                != (
                    ACTOR_CONVERGENCE_RESERVE_SECONDS
                    if self.runtime_method in ACTOR_RESERVE_JOINT_METHODS
                    else None
                )
                or runtime_config.get("actor_resume_limit") != self.actor_resume_limit
                or runtime_config.get("database_sidecar_discovery")
                != ("sqlite_conventional_sidecars_v2" if self.discover_database_sidecars else None)
                or runtime_config.get("public_paths") != list(self.public_paths)
                or runtime_config.get("public_instruction") != instruction.strip()
                or runtime_config.get("task_contract_compiler") != self.task_contract_compiler
                or runtime_config.get("intervention_effect_schema")
                != (
                    "runtime_intervention_effect_event_v1"
                    if self.runtime_method in EFFECT_LEDGER_RUNTIME_METHODS
                    else None
                )
                or runtime_config.get("actor_session_event_schema") != _ACTOR_SESSION_EVENT_SCHEMA
                or runtime_config.get("actor_runtime") != self._actor_runtime_identity()
                or runtime_config.get("bash_process_group_containment") != bash_capability.payload()
                or recorded_resources != expected_resources
            ):
                raise ValueError("managed runtime resume configuration differs")
            if legacy_adapter_config:
                runtime_config = {
                    **runtime_config,
                    "schema_version": _MANAGED_RUNTIME_CONFIG_V2,
                    "runtime_dataset_adapter": _terminal_runtime_adapter_config(),
                    "runtime_dataset_adapter_migration": {
                        "from_schema": _MANAGED_RUNTIME_CONFIG_V1,
                        "to_schema": _MANAGED_RUNTIME_CONFIG_V2,
                        "reason": "bind_existing_terminal_kernel_to_explicit_adapter",
                    },
                }
                migration_path = config_path.with_suffix(".json.migrating")
                migration_path.write_text(
                    json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n"
                )
                migration_path.replace(config_path)
            actor_session_id = runtime_config.get("actor_session_id")
            if not isinstance(actor_session_id, str) or not actor_session_id:
                raise ValueError("managed runtime actor session identity is missing")
            retention = driver.load_initial_retention()
            try:
                kernel = PortableRuntimeKernel.load(event_log_path)
            except ValueError as error:
                if str(error) != "runtime event log contains an unfinished action":
                    raise
                fresh_observations = await driver.observe()
                kernel = PortableRuntimeKernel.load_for_resume(
                    event_log_path,
                    fresh_observations=fresh_observations,
                )
            if kernel.session_id != session_id:
                raise ValueError("managed runtime environment session differs")
            expected_predicates = compile_public_acceptance_manifest(
                instruction=instruction.strip(),
                task_nodes=tuple(kernel.task_nodes.values()),
                resource_ids=tuple(item.resource_id for item in self.resource_specs),
                manifest=self.runtime_acceptance_predicate_manifest or (),
            )
            expected_predicates_by_id = {
                predicate.predicate_id: predicate for predicate in expected_predicates
            }
            if (
                kernel.acceptance_predicate_enabled
                != (self.runtime_acceptance_predicate_manifest is not None)
                or kernel.initial_acceptance_predicates != expected_predicates
                or any(
                    contract.check_id is not None
                    and kernel.predicate_id_for_check(contract.check_id)
                    != contract.predicate_id
                    for contract in self.runtime_check_contracts
                )
                or self.runtime_acceptance_predicate_manifest is not None
                and any(
                    contract.predicate_id not in expected_predicates_by_id
                    or contract.task_id
                    != expected_predicates_by_id[str(contract.predicate_id)].task_id
                    or contract.proposition
                    != expected_predicates_by_id[str(contract.predicate_id)].proposition
                    or contract.resource_ids
                    != expected_predicates_by_id[
                        str(contract.predicate_id)
                    ].resource_ids
                    for contract in self.runtime_check_contracts
                )
            ):
                raise ValueError("managed runtime acceptance predicate state differs")
            runtime_supervisor = (
                DynamicTraceDPRuntimePolicy.load(
                    budget_units=self.runtime_intervention_budget,
                    decision_log_path=decision_log_path,
                    policy_id=self.runtime_method,
                )
                if self.runtime_method in POLICY_RUNTIME_METHODS
                else None
            )
            joint_runtime_policy = (
                RuntimeJointSlatePolicy.load(
                    method_id=self.runtime_method,
                    budget_units=self.runtime_intervention_budget,
                    event_log_path=decision_log_path,
                    source_arm_index=self.joint_source_arm_index,
                    decision_resolver=build_joint_decision_resolver(kernel),
                )
                if self.runtime_method in JOINT_RUNTIME_METHODS
                else None
            )
            intervention_effect_ledger = (
                RuntimeInterventionEffectLedger(effect_log_path, load_existing=True)
                if self.runtime_method in EFFECT_LEDGER_RUNTIME_METHODS
                else None
            )
            self._reconcile_open_runtime_decision(
                kernel=kernel,
                runtime_supervisor=runtime_supervisor,
            )
            self._reconcile_uncommitted_v8_prediction(
                kernel=kernel,
                joint_runtime_policy=joint_runtime_policy,
            )
            self._reconcile_open_joint_runtime_decision(
                kernel=kernel,
                joint_runtime_policy=joint_runtime_policy,
            )
        else:
            retention_started = time.monotonic()
            retention = await driver.retain_initial()
            retention_ms = round((time.monotonic() - retention_started) * 1000)
            observation_started = time.monotonic()
            initial_observations = await driver.observe()
            initial_observation_ms = round((time.monotonic() - observation_started) * 1000)
            retention = driver.bind_initial_observations(initial_observations)
            runtime_view = instantiate_runtime_view(
                adapter=self.runtime_dataset_adapter,
                spec=RuntimeTaskInstanceSpec(
                    adapter_id=TERMINAL_RUNTIME_ADAPTER_V1,
                    dataset_family="terminal",
                    instance_id=session_id,
                    environment_session_id=session_id,
                    public_instruction=instruction,
                ),
                native_state=TerminalRuntimeNativeState(
                    resources=self.resource_specs,
                    observations=initial_observations,
                    compiler_id=self.task_contract_compiler,
                ),
            )
            task_nodes = runtime_view.task_nodes
            initial_observations = runtime_view.initial_observations
            acceptance_predicates = compile_public_acceptance_manifest(
                instruction=instruction.strip(),
                task_nodes=task_nodes,
                resource_ids=tuple(item.resource_id for item in self.resource_specs),
                manifest=self.runtime_acceptance_predicate_manifest or (),
            )
            acceptance_ids = {
                predicate.predicate_id for predicate in acceptance_predicates
            }
            acceptance_by_id = {
                predicate.predicate_id: predicate
                for predicate in acceptance_predicates
            }
            bound_check_ids = [
                contract.check_id
                for contract in self.runtime_check_contracts
                if contract.check_id is not None
            ]
            invalid_check_binding = (
                len(bound_check_ids) != len(set(bound_check_ids))
                or any(
                    contract.predicate_id not in acceptance_ids
                    for contract in self.runtime_check_contracts
                )
                or any(
                    contract.predicate_id not in acceptance_by_id
                    or contract.task_id
                    != acceptance_by_id[str(contract.predicate_id)].task_id
                    or contract.proposition
                    != acceptance_by_id[str(contract.predicate_id)].proposition
                    or contract.resource_ids
                    != acceptance_by_id[str(contract.predicate_id)].resource_ids
                    for contract in self.runtime_check_contracts
                )
            )
            if (
                self.runtime_acceptance_predicate_manifest is not None
                and invalid_check_binding
            ):
                raise ValueError("configured check differs from acceptance manifest")
            kernel = PortableRuntimeKernel.create(
                session_id=session_id,
                event_log_path=event_log_path,
                task_nodes=task_nodes,
                initial_observations=initial_observations,
                acceptance_predicates=acceptance_predicates,
                acceptance_predicate_enabled=(
                    self.runtime_acceptance_predicate_manifest is not None
                ),
            )
            for contract in self.runtime_check_contracts:
                if contract.check_id is None:
                    continue
                kernel.bind_acceptance_predicate_check(
                    check_id=contract.check_id,
                    predicate_id=str(contract.predicate_id),
                    task_id=contract.task_id,
                    proposition=contract.proposition,
                    resource_ids=contract.resource_ids,
                )
            public_dependency_evidence_ids = apply_public_instruction_dependency_evidence(
                kernel=kernel,
                instruction=instruction,
                resources=self.resource_specs,
                initial_observations=initial_observations,
                compiler_id=self.task_contract_compiler,
            )
            runtime_supervisor = (
                DynamicTraceDPRuntimePolicy(
                    budget_units=self.runtime_intervention_budget,
                    decision_log_path=decision_log_path,
                    policy_id=self.runtime_method,
                )
                if self.runtime_method in POLICY_RUNTIME_METHODS
                else None
            )
            joint_runtime_policy = (
                RuntimeJointSlatePolicy(
                    method_id=self.runtime_method,
                    budget_units=self.runtime_intervention_budget,
                    event_log_path=decision_log_path,
                    source_arm_index=self.joint_source_arm_index,
                    decision_resolver=build_joint_decision_resolver(kernel),
                )
                if self.runtime_method in JOINT_RUNTIME_METHODS
                else None
            )
            if runtime_supervisor is not None or joint_runtime_policy is not None:
                # An empty journal is the durable initial state. It lets a host
                # interruption before the first tool call replay as zero decisions.
                decision_log_path.touch()
            intervention_effect_ledger = (
                RuntimeInterventionEffectLedger(effect_log_path)
                if self.runtime_method in EFFECT_LEDGER_RUNTIME_METHODS
                else None
            )
            actor_session_id = self._managed_actor_session_identity()
            capabilities = driver.restoration_capabilities()
            runtime_config = {
                "schema_version": _MANAGED_RUNTIME_CONFIG_V2,
                "managed_state_visibility": "host_private_until_actor_exit_v1",
                "managed_private_artifact_root": str(self.logs_dir.resolve()),
                "runtime_method": self.runtime_method,
                "environment_session_id": session_id,
                "actor_session_id": actor_session_id,
                "public_instruction": instruction.strip(),
                "task_contract_compiler": self.task_contract_compiler,
                "runtime_dataset_adapter": _terminal_runtime_adapter_config(),
                "runtime_dataset_adapter_migration": None,
                "resource_observation_schema": observation_schema,
                "intervention_effect_schema": (
                    "runtime_intervention_effect_event_v1"
                    if self.runtime_method in EFFECT_LEDGER_RUNTIME_METHODS
                    else None
                ),
                "actor_session_event_schema": _ACTOR_SESSION_EVENT_SCHEMA,
                "provider_request_audit_schema": "terminal_provider_request_audit_v1",
                "runtime_capabilities": {
                    "pi_subprocess_resume_in_live_environment": True,
                    "harbor_host_process_resume": False,
                },
                "bash_process_group_containment": bash_capability.payload(),
                "actor_runtime": self._actor_runtime_identity(),
                "public_dependency_evidence_ids": list(public_dependency_evidence_ids),
                "restore_missing_prerequisites": (self.restore_missing_prerequisites),
                "public_paths": list(self.public_paths),
                "retention_budget_bytes": self.retention_budget_bytes,
                "runtime_intervention_budget": self.runtime_intervention_budget,
                "joint_source_arm_index": self.joint_source_arm_index,
                "runtime_check_configuration": self.runtime_check_configuration,
                "runtime_repair_configuration": self.runtime_repair_configuration,
                "runtime_acceptance_predicate_manifest": (
                    self.runtime_acceptance_predicate_manifest
                ),
                "runtime_acceptance_predicate_semantic_review": (
                    self.runtime_acceptance_predicate_semantic_review
                ),
                "run_class": self.run_class,
                "runtime_docker_candidate_provider": (
                    self.runtime_docker_candidate_provider
                ),
                "runtime_atomic_estimate_configuration": (
                    self.runtime_atomic_estimate_configuration.to_dict()
                    if self.runtime_atomic_estimate_configuration is not None
                    else None
                ),
                "runtime_acceptance_candidate_trials": (
                    self.runtime_acceptance_candidate_trials
                ),
                **({"runtime_guided_completion_configuration": self.runtime_guided_completion_configuration.to_dict()}
                   if self.runtime_guided_completion_configuration is not None else {}),
                "jit_test_task_id": self.jit_test_task_id,
                "jit_memory_source_path": (
                    str(self.jit_memory_source_path)
                    if self.jit_memory_source_path is not None
                    else None
                ),
                "jit_memory_artifact": (
                    "runtime-jit-memory.json"
                    if self.runtime_method in JIT_JOINT_METHODS
                    else None
                ),
                "jit_memory_payload": jit_memory_payload,
                "joint_policy_schema": (
                    RuntimeJointSlatePolicy.schema_version
                    if self.runtime_method in JOINT_RUNTIME_METHODS
                    else None
                ),
                "output_progress_action_threshold": (
                    OUTPUT_PROGRESS_ACTION_THRESHOLD
                    if self.runtime_method in OUTPUT_PROGRESS_RUNTIME_METHODS
                    else None
                ),
                "actor_failure_diagnosis_threshold": (
                    _actor_failure_diagnosis_threshold(self.runtime_method)
                ),
                "output_semantic_stability_action_threshold": (
                    OUTPUT_SEMANTIC_STABILITY_ACTION_THRESHOLD
                    if self.runtime_method in SEMANTIC_STABILITY_JOINT_METHODS
                    else None
                ),
                "actor_convergence_reserve_seconds": (
                    ACTOR_CONVERGENCE_RESERVE_SECONDS
                    if self.runtime_method in ACTOR_RESERVE_JOINT_METHODS
                    else None
                ),
                "actor_resume_limit": self.actor_resume_limit,
                "database_sidecar_discovery": (
                    "sqlite_conventional_sidecars_v2" if self.discover_database_sidecars else None
                ),
                "initialization_latency_ms": {
                    "retention": retention_ms,
                    "post_retention_observation": initial_observation_ms,
                },
                "resources": [
                    {
                        "resource_id": spec.resource_id,
                        "kind": spec.kind.value,
                        "provenance": spec.provenance,
                        "group_id": spec.group_id,
                        "retention_status": driver.retention_records[spec.resource_id].status.value,
                        "restore_supported": capabilities[spec.resource_id].supported,
                        "restore_scope": capabilities[spec.resource_id].scope,
                        "restore_conditions": list(capabilities[spec.resource_id].conditions),
                    }
                    for spec in self.resource_specs
                ],
            }
            config_staging_path = config_path.with_suffix(".json.initializing")
            config_staging_path.write_text(
                json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n"
            )
            config_staging_path.replace(config_path)
        self._managed_actor_session_id = actor_session_id
        _reconcile_interrupted_actor_receipts(observer_path, kernel=kernel)
        gateway = TerminalEnvironmentToolGateway(
            environment=environment,
            batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
            public_paths=self.public_paths,
            observer_path=observer_path,
            scratch_root=self.logs_dir / "delegated-scratch",
            observe_public_state=self.observe_public_state,
            resource_driver=driver,
            runtime_kernel=kernel,
            restore_missing_prerequisites=self.restore_missing_prerequisites,
            runtime_supervisor=runtime_supervisor,
            concurrent_runtime_observation=self.concurrent_runtime_observation,
            joint_runtime_policy=joint_runtime_policy,
            progress_trigger_threshold=self.runtime_progress_trigger_threshold,
            intervention_effect_ledger=intervention_effect_ledger,
            bash_containment_capability=bash_capability,
            actor_time_budget_seconds=(
                self.host_pi_timeout_seconds
                if self.runtime_method in ACTOR_RESERVE_JOINT_METHODS
                else None
            ),
            selector_time_observation_enabled=(
                self.runtime_method
                in (BOUNDED_TRACEDP_DEV_METHODS | RUNTIME_OPTION_CHAIN_METHODS)
            ),
            initial_sequence=sum(action.actor == "actor" for action in kernel.actions.values()),
            check_contracts=self.runtime_check_contracts,
            repair_contracts=self.runtime_predicate_repair_contracts,
            predicate_diagnosis_enabled=self.predicate_diagnosis_enabled,
            allow_check_registration=self.allow_check_registration,
            require_check_registration_phase=self.require_check_registration_phase,
            targeted_repair_trial_gate=candidate_trial_gate,
            isolated_guidance_executor=partial(self._run_isolated_actor_guidance, context=context),
        )
        actor_events = _read_actor_session_events(actor_event_log_path)
        for row in actor_events:
            payload = row["payload"]
            assert isinstance(payload, dict)
            if (
                payload.get("actor_session_id") != actor_session_id
                or payload.get("environment_session_id") != session_id
            ):
                raise ValueError("managed Actor event session differs from runtime")
        actor_attempt_count = sum(
            row["event_type"] == "actor_attempt_started" for row in actor_events
        )
        self._managed_token_accounting_complete = not any(
            row["event_type"] in {"actor_attempt_interrupted", "actor_session_budget_exhausted"}
            for row in actor_events
        )
        actor_completed = bool(
            actor_events
            and actor_events[-1]["event_type"]
            in {"actor_attempt_completed", "actor_session_budget_exhausted"}
        )
        if kernel.actor_session_closed != actor_completed:
            if kernel.actor_session_closed or kernel.unfinished_action_ids:
                raise ValueError("managed Actor and Kernel terminal states differ")
        if actor_events and actor_events[-1]["event_type"] == "actor_attempt_started":
            _append_actor_session_event(
                actor_event_log_path,
                event_type="actor_attempt_interrupted",
                payload={
                    "actor_session_id": actor_session_id,
                    "environment_session_id": session_id,
                    "attempt_index": actor_attempt_count - 1,
                    "reason": "host runtime restarted before attempt closure",
                },
            )
        self._reconcile_intervention_effect_ledger(
            kernel=kernel,
            intervention_effect_ledger=intervention_effect_ledger,
            decision_log_path=decision_log_path,
            actor_event_log_path=actor_event_log_path,
            pi_stream_root=self.logs_dir,
            expected_provider=self.runtime.provider,
            expected_model=self.runtime.model,
        )
        try:
            while not actor_completed:
                actor_resume_count = actor_attempt_count
                if actor_resume_count > self.actor_resume_limit:
                    raise HostPiSessionInterrupted("managed Actor resume budget was exhausted")
                attempt_index = actor_attempt_count
                self._managed_actor_attempt_index = attempt_index
                # Create the durable stream slots before the start event so a crash
                # at this boundary remains a complete, auditable empty attempt.
                for suffix in (".txt", ".stderr.txt"):
                    (self.logs_dir / f"pi-attempt-{attempt_index:03d}{suffix}").touch()
                _append_actor_session_event(
                    actor_event_log_path,
                    event_type="actor_attempt_started",
                    payload={
                        "actor_session_id": actor_session_id,
                        "environment_session_id": session_id,
                        "attempt_index": attempt_index,
                    },
                )
                actor_attempt_count += 1
                try:
                    await self._continue_with_gateway(
                        instruction=(
                            instruction
                            if attempt_index == 0
                            else (
                                "Continue the same task in the current live environment. "
                                "Re-check any action whose effect was uncertain after "
                                "the interruption."
                            )
                        ),
                        gateway=gateway,
                        context=context,
                    )
                except HostPiSessionInterrupted as error:
                    self._managed_token_accounting_complete = False
                    budget_exhausted = self._managed_session_deadline_monotonic <= time.monotonic()
                    _append_actor_session_event(
                        actor_event_log_path,
                        event_type=(
                            "actor_session_budget_exhausted"
                            if budget_exhausted
                            else "actor_attempt_interrupted"
                        ),
                        payload={
                            "actor_session_id": actor_session_id,
                            "environment_session_id": session_id,
                            "attempt_index": attempt_index,
                            "reason": str(error),
                        },
                    )
                    if kernel.unfinished_action_ids:
                        try:
                            surviving_observations = await driver.observe()
                        except Exception:
                            surviving_observations = driver.unknown_observations()
                        kernel.resolve_interrupted_action(fresh_observations=surviving_observations)
                    if budget_exhausted:
                        actor_completed = True
                        break
                    if attempt_index >= self.actor_resume_limit:
                        raise
                    fresh_observations = await driver.observe()
                    kernel = PortableRuntimeKernel.load_for_resume(
                        event_log_path,
                        fresh_observations=fresh_observations,
                    )
                    runtime_supervisor = (
                        DynamicTraceDPRuntimePolicy.load(
                            budget_units=self.runtime_intervention_budget,
                            decision_log_path=decision_log_path,
                            policy_id=self.runtime_method,
                        )
                        if self.runtime_method in POLICY_RUNTIME_METHODS
                        else None
                    )
                    joint_runtime_policy = (
                        RuntimeJointSlatePolicy.load(
                            method_id=self.runtime_method,
                            budget_units=self.runtime_intervention_budget,
                            event_log_path=decision_log_path,
                            source_arm_index=self.joint_source_arm_index,
                            decision_resolver=build_joint_decision_resolver(kernel),
                        )
                        if self.runtime_method in JOINT_RUNTIME_METHODS
                        else None
                    )
                    intervention_effect_ledger = (
                        RuntimeInterventionEffectLedger(effect_log_path, load_existing=True)
                        if self.runtime_method in EFFECT_LEDGER_RUNTIME_METHODS
                        else None
                    )
                    self._reconcile_open_runtime_decision(
                        kernel=kernel,
                        runtime_supervisor=runtime_supervisor,
                    )
                    self._reconcile_uncommitted_v8_prediction(
                        kernel=kernel,
                        joint_runtime_policy=joint_runtime_policy,
                    )
                    self._reconcile_open_joint_runtime_decision(
                        kernel=kernel,
                        joint_runtime_policy=joint_runtime_policy,
                    )
                    self._reconcile_intervention_effect_ledger(
                        kernel=kernel,
                        intervention_effect_ledger=intervention_effect_ledger,
                        decision_log_path=decision_log_path,
                        actor_event_log_path=actor_event_log_path,
                        pi_stream_root=self.logs_dir,
                        expected_provider=self.runtime.provider,
                        expected_model=self.runtime.model,
                    )
                    _reconcile_interrupted_actor_receipts(observer_path, kernel=kernel)
                    gateway = TerminalEnvironmentToolGateway(
                        environment=environment,
                        batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
                        public_paths=self.public_paths,
                        observer_path=observer_path,
                        scratch_root=self.logs_dir / "delegated-scratch",
                        observe_public_state=self.observe_public_state,
                        resource_driver=driver,
                        runtime_kernel=kernel,
                        restore_missing_prerequisites=(self.restore_missing_prerequisites),
                        runtime_supervisor=runtime_supervisor,
                        concurrent_runtime_observation=self.concurrent_runtime_observation,
                        joint_runtime_policy=joint_runtime_policy,
            progress_trigger_threshold=self.runtime_progress_trigger_threshold,
                        intervention_effect_ledger=intervention_effect_ledger,
                        bash_containment_capability=bash_capability,
                        actor_time_budget_seconds=(
                            self.host_pi_timeout_seconds
                            if self.runtime_method in ACTOR_RESERVE_JOINT_METHODS
                            else None
                        ),
                        selector_time_observation_enabled=(
                            self.runtime_method
                            in (
                                BOUNDED_TRACEDP_DEV_METHODS
                                | RUNTIME_OPTION_CHAIN_METHODS
                            )
                        ),
                        initial_sequence=sum(
                            action.actor == "actor" for action in kernel.actions.values()
                        ),
                        check_contracts=self.runtime_check_contracts,
                        repair_contracts=self.runtime_predicate_repair_contracts,
                        predicate_diagnosis_enabled=self.predicate_diagnosis_enabled,
                        allow_check_registration=self.allow_check_registration,
                        require_check_registration_phase=(
                            self.require_check_registration_phase
                        ),
                        targeted_repair_trial_gate=candidate_trial_gate,
                        isolated_guidance_executor=partial(
                            self._run_isolated_actor_guidance, context=context
                        ),
                    )
                    continue
                except asyncio.CancelledError:
                    # Harbor enforces the outer Agent timeout by cancellation.
                    # Close the durable attempt before propagating that timeout.
                    self._managed_token_accounting_complete = False
                    _append_actor_session_event(
                        actor_event_log_path,
                        event_type="actor_attempt_interrupted",
                        payload={
                            "actor_session_id": actor_session_id,
                            "environment_session_id": session_id,
                            "attempt_index": attempt_index,
                            "reason": "Harbor cancelled the active Actor attempt",
                        },
                    )
                    raise
                _append_actor_session_event(
                    actor_event_log_path,
                    event_type="actor_attempt_completed",
                    payload={
                        "actor_session_id": actor_session_id,
                        "environment_session_id": session_id,
                        "attempt_index": attempt_index,
                    },
                )
                actor_completed = True
                break
        finally:
            if kernel.unfinished_action_ids:
                try:
                    surviving_observations = await driver.observe()
                except Exception:
                    surviving_observations = driver.unknown_observations()
                kernel.resolve_interrupted_action(fresh_observations=surviving_observations)
            if actor_completed and not kernel.actor_session_closed:
                kernel.close_actor_session()
            final_actor_events = _read_actor_session_events(actor_event_log_path)
            terminal_event_type = (
                final_actor_events[-1]["event_type"] if final_actor_events else None
            )
            final_state = kernel.public_projection()
            (self.logs_dir / "runtime-final-state.json").write_text(
                json.dumps(final_state, ensure_ascii=False, indent=2) + "\n"
            )
            runtime_actions = [
                action for action in kernel.actions.values() if action.actor == "runtime"
            ]
            completed_interventions = [
                action for action in runtime_actions if action.phase is ActionPhase.COMPLETED
            ]
            cancelled_interventions = [
                action
                for action in runtime_actions
                if action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
            ]
            planner_accounting = _runtime_prediction_accounting(
                self.logs_dir / "runtime-public-predictions.jsonl"
            )
            if self.runtime_method in BOUNDED_TRACEDP_DEV_METHODS:
                context.n_input_tokens = (context.n_input_tokens or 0) + int(
                    planner_accounting["input_tokens"]
                )
                context.n_output_tokens = (context.n_output_tokens or 0) + int(
                    planner_accounting["output_tokens"]
                )
            complete_token_accounting = (
                self._managed_token_accounting_complete
                and planner_accounting["token_accounting_complete"] is True
            )
            metadata = dict(context.metadata or {})
            actor_resume_count = max(0, actor_attempt_count - 1)
            actor_runtime = self._actor_runtime_identity()
            for key in (
                "provider",
                "model",
                "pi_version",
                "thinking",
                "transport",
                "max_output_tokens",
                "public_state_observation_enabled",
                "public_path_count",
            ):
                source_key = {
                    "max_output_tokens": "host_pi_max_output_tokens",
                    "public_state_observation_enabled": "observe_public_state",
                }.get(key, key)
                metadata.setdefault(key, actor_runtime[source_key])
            metadata.setdefault("credential_location", "host_only")
            metadata.setdefault("reasoning_tokens", 0)
            if self.runtime_method in BOUNDED_TRACEDP_DEV_METHODS:
                metadata["reasoning_tokens"] = int(metadata["reasoning_tokens"]) + int(
                    planner_accounting["reasoning_tokens"]
                )
            if self.runtime_method in RPC_JOINT_METHODS:
                metadata["actor_control_mode"] = "pi_jsonl_rpc"
            metadata.update(
                {
                    "continuous_environment_session": session_id,
                    "continuous_actor_session": actor_completed,
                    "actor_session_terminal_reason": (
                        "budget_exhausted"
                        if terminal_event_type == "actor_session_budget_exhausted"
                        else "natural_completion"
                        if actor_completed
                        else None
                    ),
                    "actor_session_id": actor_session_id,
                    "actor_session_resumed": actor_resume_count > 0,
                    "actor_session_resume_count": actor_resume_count,
                    "actor_session_resume_limit": self.actor_resume_limit,
                    "actor_session_resumable": not actor_completed,
                    "pi_subprocess_resume_in_live_environment": True,
                    "harbor_host_process_resume": False,
                    "bash_process_group_containment": bash_capability.payload(),
                    "runtime_method": self.runtime_method,
                    "runtime_intervention_count": len(completed_interventions),
                    "runtime_intervention_attempt_count": len(runtime_actions),
                    "runtime_intervention_completed_count": len(completed_interventions),
                    "runtime_intervention_failure_count": (
                        len(runtime_actions)
                        - len(completed_interventions)
                        - len(cancelled_interventions)
                    ),
                    "runtime_intervention_cancelled_count": len(cancelled_interventions),
                    "runtime_intervention_budget": self.runtime_intervention_budget,
                    "actor_check_registration_phase_required": (
                        self.require_check_registration_phase
                    ),
                    "actor_check_registration_phase_status": (
                        gateway.check_registration_phase_status
                    ),
                    "output_progress_action_threshold": (
                        OUTPUT_PROGRESS_ACTION_THRESHOLD
                        if self.runtime_method in OUTPUT_PROGRESS_RUNTIME_METHODS
                        else None
                    ),
                    "actor_failure_diagnosis_threshold": (
                        _actor_failure_diagnosis_threshold(self.runtime_method)
                    ),
                    "output_semantic_stability_action_threshold": (
                        OUTPUT_SEMANTIC_STABILITY_ACTION_THRESHOLD
                        if self.runtime_method in SEMANTIC_STABILITY_JOINT_METHODS
                        else None
                    ),
                    "actor_convergence_reserve_seconds": (
                        ACTOR_CONVERGENCE_RESERVE_SECONDS
                        if self.runtime_method in ACTOR_RESERVE_JOINT_METHODS
                        else None
                    ),
                    "runtime_intervention_budget_remaining": (
                        runtime_supervisor.remaining_budget_units
                        if runtime_supervisor is not None
                        else joint_runtime_policy.remaining_budget_units
                        if joint_runtime_policy is not None
                        else None
                    ),
                    "runtime_retained_bytes": sum(item.bytes_retained for item in retention),
                    "runtime_resource_count": len(self.resource_specs),
                    "runtime_planner_accounting": (
                        planner_accounting
                        if self.runtime_method in BOUNDED_TRACEDP_DEV_METHODS
                        else None
                    ),
                    "token_accounting_complete": complete_token_accounting,
                    "token_accounting_semantics": (
                        "complete_actor_and_runtime_planner_usage"
                        if complete_token_accounting
                        and self.runtime_method in BOUNDED_TRACEDP_DEV_METHODS
                        else "complete_message_end_usage"
                        if complete_token_accounting
                        else "lower_bound_with_unknown_actor_or_runtime_planner_usage"
                        if self.runtime_method in BOUNDED_TRACEDP_DEV_METHODS
                        else "lower_bound_from_received_message_end_usage"
                    ),
                }
            )
            context.metadata = metadata
            if planner_actor is not None:
                planner_actor.close()


class HostTerminalHistoricalManagedOpenCodeGoPi(HostManagedOpenCodeGoPi):
    """Manage the continuation of one exact historical failed Actor session.

    The frozen public prefix is replayed before the Runtime kernel exists, so every
    managed observation, registered public acceptance check, isolated repair trial,
    and delivery decision describes the real post-failure state.  The Actor then
    continues its exact persisted Pi context; its history is never reconstructed
    inside a new prompt.

    This is the only seam where "the Runtime takes over a failed task" is real: a
    plain managed run always starts from a fresh task state, and the historical
    continuation host owns no kernel, checks, trials, or selector.
    """

    def __init__(
        self,
        *args,
        source_session_path: str,
        source_session_id: str,
        source_session_cwd: str,
        source_actor: dict[str, object],
        expected_terminal_stop_reason: str,
        task_instruction: str,
        prefix_tool_calls: list[dict[str, object]],
        source_actor_max_output_tokens: int,
        continuation_instruction: str | None = None,
        prefix_snapshot_receipt: str | None = None,
        supervisor_reentry_config: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prefix_snapshot_receipt = prefix_snapshot_receipt
        # The supervisor re-enters at natural settlements.
        self.supervisor_reentry = None
        if supervisor_reentry_config is not None:
            from lh_platform.benchmarks.supervisor_reentry import SupervisorReentry

            self.supervisor_reentry = SupervisorReentry.from_path(supervisor_reentry_config)
        if not self.runtime_check_contracts or any(
            contract.check_id is None for contract in self.runtime_check_contracts
        ):
            raise ValueError(
                "historical managed continuation requires bound executable public checks"
            )
        if self.runtime_acceptance_predicate_manifest is None:
            raise ValueError(
                "historical managed continuation requires public acceptance predicates"
            )
        if self.runtime_method not in SUPPORTED_RUNTIME_METHODS:
            raise ValueError(
                "historical managed continuation requires a supported Runtime method"
            )
        self.source_session_path = Path(source_session_path).resolve()
        # The source episode was produced under its own recorded output budget.
        # Recovery arms may be given a larger continuation budget, so the two
        # numbers stay separate: the source identity must match the recorded
        # episode, while `host_pi_max_output_tokens` only bounds the continuation.
        if (
            isinstance(source_actor_max_output_tokens, bool)
            or not isinstance(source_actor_max_output_tokens, int)
            or source_actor_max_output_tokens < 1
        ):
            raise ValueError("historical managed source actor output budget is invalid")
        self.source_actor_max_output_tokens = source_actor_max_output_tokens
        self.continuation_max_output_tokens = self.host_pi_max_output_tokens
        self.source_actor = normalize_terminal_source_actor(
            source_actor,
            runtime=self.runtime,
            max_output_tokens=self.source_actor_max_output_tokens,
        )
        self.source_session = load_frozen_pi_continuation_source(
            self.source_session_path,
            expected_instruction=task_instruction,
            expected_provider=self.runtime.provider,
            expected_model=self.runtime.model,
            expected_thinking=self.runtime.thinking,
            expected_terminal_stop_reason=expected_terminal_stop_reason,
            expected_cwd=Path(source_session_cwd),
        )
        if self.source_session.session_id != source_session_id:
            raise ValueError("historical managed Pi source session ID differs")
        self.prefix_tool_calls = tuple(prefix_tool_calls)
        calls = self.source_session.tool_receipt.tool_calls
        executions = {
            item.tool_call_id: item
            for item in self.source_session.tool_receipt.tool_executions
        }
        if not self.prefix_tool_calls or len(calls) != len(self.prefix_tool_calls):
            raise ValueError("historical managed Pi source call count differs")
        for call, expected in zip(calls, self.prefix_tool_calls, strict=True):
            execution = executions.get(call.tool_call_id)
            if (
                expected.get("source_tool_call_id") != call.tool_call_id
                or expected.get("tool_name") != call.tool_name
                or expected.get("arguments") != call.arguments
                or execution is None
                or expected.get("source_is_error") is not execution.is_error
            ):
                raise ValueError("historical managed Pi source call contract differs")
        normalized_trigger = (
            str(continuation_instruction).strip()
            if continuation_instruction is not None
            else HISTORICAL_MANAGED_CONTINUATION_INSTRUCTION
        )
        if not normalized_trigger:
            raise ValueError("historical managed continuation trigger is empty")
        self.continuation_instruction = normalized_trigger
        self._historical_continuation_armed = False
        self._staged_session: Path | None = None
        self._historical_source_bytes = b""

    @override
    def _managed_actor_session_identity(self) -> str:
        """Reuse the identity the staged historical Pi context file already carries."""

        return self.source_session.session_id

    async def _restore_historical_prefix_snapshot(
        self, environment: BaseEnvironment, gateway: TerminalEnvironmentToolGateway
    ) -> tuple[dict[str, object], ...]:
        """Start from the verified snapshot of this prefix instead of replaying it."""

        history, verification = await restore_prefix_from_snapshot(
            environment=environment,
            gateway=gateway,
            receipt_path=str(self.prefix_snapshot_receipt),
            prefix_tool_calls=self.prefix_tool_calls,
        )
        with (self.logs_dir / "terminal-historical-replay.jsonl").open("w", encoding="utf-8") as session:
            for row in history:
                session.write(json.dumps(row, ensure_ascii=False) + "\n")
        (self.logs_dir / "terminal-historical-replay.json").write_text(
            json.dumps(
                {
                    "schema_version": HISTORICAL_MANAGED_REPLAY_SCHEMA,
                    "replayed_tool_call_count": len(history),
                    "execution_and_error_parity": True,
                    "parity_policy": PREFIX_PARITY_POLICY_EXACT,
                    "parity_exemptions": [],
                    "model_called_during_replay": False,
                    "official_verifier_executed": False,
                    "status": "failure_state_restored_from_verified_snapshot",
                    "prefix_snapshot": verification,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return history

    async def _replay_historical_prefix(
        self, environment: BaseEnvironment
    ) -> tuple[dict[str, object], ...]:
        """Rebuild the observed failure state before the Runtime observes anything."""

        gateway = TerminalEnvironmentToolGateway(
            environment=environment,
            batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
            public_paths=self.public_paths,
            observer_path=self.logs_dir / "terminal-historical-replay-state.jsonl",
            scratch_root=self.logs_dir / "historical-replay-scratch",
            observe_public_state=self.observe_public_state,
        )
        if self.prefix_snapshot_receipt is not None:
            return await self._restore_historical_prefix_snapshot(environment, gateway)
        history: list[dict[str, object]] = []
        replay_path = self.logs_dir / "terminal-historical-replay.jsonl"
        with replay_path.open("w", encoding="utf-8") as session:
            for index, call in enumerate(self.prefix_tool_calls):
                tool_name = str(call["tool_name"])
                arguments = call.get("arguments")
                if not isinstance(arguments, dict):
                    raise ValueError("historical managed prefix arguments must be an object")
                call_id = f"historical-managed-prefix-{index:03d}"
                if call.get("environment_step_executed", True) is False:
                    result = gateway.record_parallel_rejection(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                else:
                    result = await gateway.execute(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                history.append(
                    {
                        "sequence": index,
                        "source_tool_call_id": call["source_tool_call_id"],
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "is_error": result.is_error,
                        "result": result.content,
                    }
                )
                session.write(json.dumps(history[-1], ensure_ascii=False) + "\n")
                session.flush()
        parity_exemptions, parity_mismatch = classify_prefix_parity(
            list(self.prefix_tool_calls), history
        )
        if parity_mismatch is not None:
            raise ValueError("historical managed prefix replay error parity differs")
        (self.logs_dir / "terminal-historical-replay.json").write_text(
            json.dumps(
                {
                    "schema_version": HISTORICAL_MANAGED_REPLAY_SCHEMA,
                    "replayed_tool_call_count": len(history),
                    "execution_and_error_parity": True,
                    "parity_policy": (
                        PREFIX_PARITY_POLICY_EXACT
                        if not parity_exemptions
                        else PREFIX_PARITY_POLICY_EXEMPTIONS
                    ),
                    "parity_exemptions": parity_exemptions,
                    "model_called_during_replay": False,
                    "official_verifier_executed": False,
                    "status": "failure_state_replayed",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return tuple(history)

    @override
    async def _continue_with_gateway(
        self,
        *,
        instruction: str,
        gateway: TerminalEnvironmentToolGateway,
        context: AgentContext,
    ) -> None:
        """Bind the staged historical context to the first live Actor attempt only."""

        if self._historical_continuation_armed:
            if getattr(self, "_isolated_actor_trial_active", False):
                raise RuntimeError(
                    "historical continuation cannot start inside an isolated trial"
                )
            if gateway.runtime_kernel is None or not gateway.check_contracts:
                raise RuntimeError(
                    "historical managed continuation requires the managed runtime gateway"
                )
            if self._staged_session is None:
                raise RuntimeError("historical managed continuation lacks a staged session")
            self._historical_continuation_armed = False
            self._managed_actor_session_id = self.source_session.session_id
            self._managed_actor_session_file = self._staged_session
            self._managed_actor_expected_message_count = self.source_session.message_count
            instruction = self.continuation_instruction
        await super()._continue_with_gateway(
            instruction=instruction,
            gateway=gateway,
            context=context,
        )

    def _verify_historical_continuation_appended(self) -> bool:
        """Require the continued Pi session to extend its staged historical copy.

        A real continuation must append at least one entry; otherwise the Actor
        never actually continued and the trial result would be meaningless.
        """

        assert self._staged_session is not None
        continued = self._staged_session.read_bytes()
        if len(continued) <= len(self._historical_source_bytes) or not continued.startswith(
            self._historical_source_bytes
        ):
            raise RuntimeError(
                "managed Pi continuation did not append to the staged source context"
            )
        return True

    def _continuation_truncation_accounting(self) -> dict[str, object]:
        """Summarize how often the continuation Actor hit its output budget.

        Reporting `output_tokens_used / max_output_tokens / truncated` is required
        to separate a method failure from a downstream execution-capability
        ceiling, so the numbers are read back from the real Pi attempt streams
        rather than inferred from the terminal outcome.
        """

        attempts: list[dict[str, object]] = []
        for path in sorted(self.logs_dir.glob("pi-attempt-*.txt")):
            if path.name.endswith(".stderr.txt"):
                continue
            receipt = load_pi_stream_receipt(
                path,
                expected_provider=self.runtime.provider,
                expected_model=self.runtime.model,
            )
            attempts.append(
                {
                    "stream": path.name,
                    "output_tokens_used": receipt.output_tokens,
                    "max_output_tokens": self.continuation_max_output_tokens,
                    "truncated_assistant_messages": (
                        receipt.truncated_assistant_messages
                    ),
                    "last_assistant_stop_reason": receipt.last_assistant_stop_reason,
                }
            )
        return {
            "schema_version": "terminal_continuation_output_budget_accounting_v1",
            "continuation_max_output_tokens": self.continuation_max_output_tokens,
            "source_actor_max_output_tokens": self.source_actor_max_output_tokens,
            "attempts": attempts,
            "output_tokens_used": sum(
                int(row["output_tokens_used"]) for row in attempts
            ),
            "truncated_assistant_messages": sum(
                int(row["truncated_assistant_messages"]) for row in attempts
            ),
            "truncated": any(
                int(row["truncated_assistant_messages"]) > 0 for row in attempts
            ),
        }

    async def _prerequisite_probe_kernel(self, environment: BaseEnvironment) -> object:
        """Observe every checker prerequisite once at the frozen entry.

        A0 builds no runtime kernel, so without this probe the sweep would treat every
        prerequisite as available; a missing checker tool (e.g. dna-insert's oligotm) would
        then crash the check body and be recorded as refuted instead of the declared
        blocked-checker UNKNOWN.
        """

        resource_ids = sorted(
            {
                resource_id
                for contract in self.runtime_check_contracts
                for resource_id in contract.prerequisite_resource_ids
            }
        )
        observations = {}
        for resource_id in resource_ids:
            # `test -e -- PATH` is malformed for the shell builtin (extra operand, exit 2),
            # which silently marked every prerequisite missing on the first probe version;
            # the quoted-path form is the one that works.
            result = await environment.exec(
                f"test -e {json.dumps(resource_id)}", timeout_sec=30
            )
            observations[resource_id] = SimpleNamespace(
                observed=True, exists=result.return_code == 0
            )
        return SimpleNamespace(observations=observations)

    async def _run_a0_calibration_sweep(
        self, *, gateway: object, trial_gate: object, prerequisite_kernel: object,
        context: AgentContext,
    ) -> dict[str, object]:
        """Execute the A0 calibration sweep and stop before any continuation or DP.

        Every check and every repair candidate runs in its own fresh clone of the frozen entry
        state, so the rows measure P(y | z0, u) rather than a sequence of contaminated states.
        """

        if not self.runtime_check_contracts or not self.runtime_predicate_repair_contracts:
            raise ValueError("A0 calibration requires configured checks and repair candidates")

        async def capture_entry_manifest(label: str) -> dict[str, object]:
            manifest, _transferred, _calls = await trial_gate._capture_main(label)
            return manifest

        async def create_isolated_environment(label: str):
            return await trial_gate.isolated_environment_factory(label)

        async def restore_isolated(manifest, isolated_environment) -> None:
            await trial_gate._restore_isolated(
                manifest, isolated_environment=isolated_environment
            )

        async def execute_isolated_bash(isolated_environment, arguments) -> dict[str, object]:
            executor = gateway._isolated_bash_executor_factory(isolated_environment)
            result = await executor(dict(arguments), True)
            return {
                "content": result.content,
                # `DelegatedToolResult` carries the process status inside `details`, exactly
                # like the isolated trial gate reads it.
                "return_code": result.details.get("return_code"),
                "is_error": result.is_error,
                "environment_step_executed": result.environment_step_executed,
            }

        async def stop_isolated(isolated_environment) -> None:
            await isolated_environment.stop(delete=True)

        async def run_repair_trial(repair):
            return await trial_gate.run_trial(
                trial_id=f"a0-repair-{repair.repair_id}",
                failure_state_id=repair.task_id,
                repair=repair,
                checks=self.runtime_check_contracts,
                execute_isolated_bash_factory=gateway._isolated_bash_executor_factory,
                execute_isolated_guidance=partial(self._run_isolated_actor_guidance, context=context),
            )

        return await run_calibration_sweep(
            source_task_id=(
                self.runtime_calibration_source_task_id or self.logs_dir.name
            ),
            check_contracts=self.runtime_check_contracts,
            repair_contracts=self.runtime_predicate_repair_contracts,
            capture_entry_manifest=capture_entry_manifest,
            create_isolated_environment=create_isolated_environment,
            restore_isolated=restore_isolated,
            execute_isolated_bash=execute_isolated_bash,
            stop_isolated_environment=stop_isolated,
            run_repair_trial=run_repair_trial,
            kernel_for_prerequisites=prerequisite_kernel,
            output_path=self.logs_dir / "a0-calibration.json",
            negative_control=self.runtime_calibration_negative_control,
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Replay the failure, continue its exact Pi context, then score officially."""

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        instruction_receipt = {
            "schema_version": "terminal_historical_instruction_identity_v1",
            "source_instruction": self.source_session.task_instruction,
            "received_instruction": instruction,
            "raw_equal": instruction == self.source_session.task_instruction,
            "boundary_normalized_equal": terminal_instructions_are_equivalent(
                self.source_session.task_instruction, instruction
            ),
            "normalization": "remove_at_most_one_terminal_line_ending",
        }
        (self.logs_dir / "terminal-historical-instruction-identity.json").write_text(
            json.dumps(instruction_receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not instruction_receipt["boundary_normalized_equal"]:
            raise ValueError("historical managed runtime task instruction differs")
        history = await self._replay_historical_prefix(environment)
        if self.runtime_calibration_mode:
            # A0 stops here on purpose: no continuation staging, no DP, no JIT, no actor turn.
            # The sweep runs before super().run(), so the candidate-environment wrap, the
            # shared clone gate, the public instruction (needed by the isolated guidance
            # trials) and a minimal tool gateway are built here.
            self._managed_public_instruction = instruction
            candidate_provider = None
            try:
                candidate_provider, candidate_environment = (
                    await self._open_candidate_environment(environment)
                )
                session_id = await self._environment_session_id(candidate_environment)
                trial_gate = await self._build_candidate_trial_gate(
                    candidate_environment, session_id
                )
                if trial_gate is None:
                    raise RuntimeError(
                        "A0 calibration requires the isolated candidate trial gate"
                    )
                self._managed_candidate_trial_gate = trial_gate
                bash_capability = await probe_bash_containment_capability(
                    candidate_environment
                )
                gateway = TerminalEnvironmentToolGateway(
                    environment=candidate_environment,
                    batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
                    public_paths=self.public_paths,
                    observer_path=self.logs_dir / "terminal-public-state.jsonl",
                    scratch_root=self.logs_dir / "delegated-scratch",
                    observe_public_state=self.observe_public_state,
                    bash_containment_capability=bash_capability,
                )
                self._managed_gateway = gateway
                prerequisite_kernel = await self._prerequisite_probe_kernel(
                    candidate_environment
                )
                await self._run_a0_calibration_sweep(
                    gateway=gateway,
                    trial_gate=trial_gate,
                    prerequisite_kernel=prerequisite_kernel,
                    context=context,
                )
            finally:
                if candidate_provider is not None:
                    await candidate_provider.close()
            return
        self._historical_source_bytes = self.source_session_path.read_bytes()
        self._staged_session = stage_pi_continuation_session(
            self.source_session_path,
            self.logs_dir / "pi" / "sessions",
        )
        receipt_path = self.logs_dir / "terminal-historical-session-continuation.json"
        receipt = {
            "schema_version": HISTORICAL_MANAGED_CONTINUATION_SCHEMA,
            "source_actor": self.source_actor,
            "source_session_id": self.source_session.session_id,
            "source_session_version": self.source_session.session_version,
            "source_session_cwd": self.source_session.session_cwd,
            "source_leaf_id": self.source_session.leaf_id,
            "source_message_count": self.source_session.message_count,
            "source_terminal_stop_reason": self.source_session.terminal_stop_reason,
            "replayed_tool_call_count": len(history),
            "execution_and_error_parity": True,
            "context_load_mode": "pi_explicit_session_file",
            "history_reconstructed_in_prompt": False,
            "continuation_instruction": self.continuation_instruction,
            "continuation_instruction_source": "shared_frozen_historical_trigger_v1",
            # The two budgets are recorded separately so a report can prove the
            # continuation budget was a shared execution-capability decision and
            # not a per-arm advantage.
            "source_actor_max_output_tokens": self.source_actor_max_output_tokens,
            "continuation_max_output_tokens": self.continuation_max_output_tokens,
            "continuation_budget_override": (
                self.continuation_max_output_tokens
                != self.source_actor_max_output_tokens
            ),
            "runtime_method": self.runtime_method,
            "runtime_owns_failure_state": True,
            "model_called_during_replay_or_staging": False,
            "official_verifier_executed": False,
            "status": "runtime_failure_state_armed",
        }
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._historical_continuation_armed = True
        await super().run(instruction, environment, context)
        if self.source_session_path.read_bytes() != self._historical_source_bytes:
            raise RuntimeError("historical managed Pi source session changed during continuation")
        appended = self._verify_historical_continuation_appended()
        truncation = self._continuation_truncation_accounting()
        (self.logs_dir / "terminal-continuation-output-budget.json").write_text(
            json.dumps(truncation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        receipt.update(
            {
                "status": "managed_actor_continuation_completed",
                "source_session_unchanged": True,
                "staged_session_appended": appended,
                "output_budget_accounting": truncation,
            }
        )
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        context.metadata = {
            **dict(context.metadata or {}),
            "historical_continuation_identity": "exact_pi_session_context_branch",
            "historical_prefix_tool_call_count": len(self.prefix_tool_calls),
            "historical_source_pi_session_id": self.source_session.session_id,
            "historical_source_pi_message_count": self.source_session.message_count,
            "history_reconstructed_in_prompt": False,
            "runtime_owns_failure_state": True,
            "source_actor_max_output_tokens": self.source_actor_max_output_tokens,
            "continuation_max_output_tokens": self.continuation_max_output_tokens,
            "continuation_output_tokens_used": truncation["output_tokens_used"],
            "continuation_truncated": truncation["truncated"],
            "continuation_truncated_assistant_messages": truncation[
                "truncated_assistant_messages"
            ],
        }


class HostTerminalHistoricalManagedZeroModelProbe(
    HostTerminalHistoricalManagedOpenCodeGoPi
):
    """Run the historical managed path with zero provider calls.

    A paid trial launches Pi, which costs money and depends on a provider.  This
    probe keeps every other part of the path real: the frozen public prefix is
    replayed inside the official task environment, the managed kernel initializes
    on the resulting failure state, the Runtime compiles and ranks its real action
    slate, and the reviewed registered checks execute in the same container.  Only
    the Actor's token generation is replaced by a frozen list of read-only tool
    calls.

    It therefore proves nothing about task success.  It exists to falsify three
    configuration assumptions before spending money:

    1. every action in the real slate has exactly one frozen atomic estimate rule,
       which the Runtime-owned DP resolver enforces by raising otherwise;
    2. the replayed failure state in the real container matches the frozen public
       checks, which must come back refuted;
    3. the reviewed check commands are executable as declared.

    The probe refuses to execute an isolated candidate trial, because a guidance
    trial needs a real Actor; that refusal is recorded instead of silently
    pretending a trial happened.
    """

    def __init__(
        self,
        *args,
        zero_model_actor_commands: list[str],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.run_class != EXPOSED_DEVELOPMENT_CANARY_RUN_CLASS:
            raise ValueError("the zero-model probe is development-only")
        commands = tuple(zero_model_actor_commands)
        if not commands or any(
            not isinstance(item, str) or not item.strip() for item in commands
        ):
            raise ValueError("the zero-model probe needs non-empty read-only commands")
        self.zero_model_actor_commands = commands
        self._probe_actor_calls: list[dict[str, object]] = []

    @override
    async def _run_host_pi(
        self,
        *,
        instruction: str,
        socket_path: Path,
        context: AgentContext,
        session_id: str | None = None,
        session_file: Path | None = None,
        expected_session_message_count: int | None = None,
        gateway: TerminalEnvironmentToolGateway | None = None,
    ) -> None:
        """Execute the frozen probe actions instead of launching a model."""

        del instruction, socket_path, context
        if gateway is None:
            raise RuntimeError("the zero-model probe requires the live tool gateway")
        if getattr(self, "_isolated_actor_trial_active", False):
            self._probe_actor_calls.append({"refused": "isolated_candidate_trial"})
            raise RuntimeError(
                "the zero-model probe cannot run an isolated Actor guidance trial"
            )
        for index, command in enumerate(self.zero_model_actor_commands):
            result = await gateway.execute(
                tool_call_id=f"zero-model-probe-{index:03d}",
                tool_name="bash",
                arguments={"command": command},
            )
            self._probe_actor_calls.append(
                {
                    "sequence": index,
                    "command": command,
                    "is_error": result.is_error,
                    "return_code": result.details.get("return_code"),
                    "result_head": result.content[:512],
                    "runtime_interventions": len(
                        result.details.get("runtime_interventions") or ()
                    ),
                }
            )

    @override
    def _verify_historical_continuation_appended(self) -> bool:
        """The probe replaces the Actor, so no real Pi append can be observed.

        Returning False keeps the receipt truthful: nothing here may be read as a
        continued Actor session.
        """

        return False

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Replay and manage the failure state, then publish the probe receipt."""

        await super().run(instruction, environment, context)
        receipt_path = self.logs_dir / "terminal-historical-managed-zero-model-probe.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": "terminal_historical_managed_zero_model_probe_v1",
                    "implementation_status": "implementation_unverified",
                    "provider_call_count": 0,
                    "official_evaluator_consumed": False,
                    "purpose": (
                        "qualify estimate coverage, failure-state replay parity, and "
                        "check executability without spending provider tokens"
                    ),
                    "claim_boundary": (
                        "This run says nothing about task success or about any TraceDP "
                        "advantage; the Actor never generated a token."
                    ),
                    "runtime_method": self.runtime_method,
                    "actor_commands": list(self.zero_model_actor_commands),
                    "actor_calls": self._probe_actor_calls,
                    "staged_session_appended": False,
                    "actor_session_continued": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class HostTerminalReplayAgent(BaseAgent):
    """Replay a frozen terminal tool chain without invoking a model."""

    async def _capture_prefix_snapshot(
        self,
        environment: BaseEnvironment,
        gateway: TerminalEnvironmentToolGateway,
        history: list[dict[str, object]],
    ) -> None:
        """Commit the container right after the prefix, before any menu check runs.

        This receipt is raw: the snapshot stage accepts it only after it has checked the
        replay against the source call by call and the official verifier's verdict.
        """

        container = await container_id(environment)
        public_state = public_state_payload(await gateway._snapshot(len(history)))
        image_id = await commit_container(container, str(self.snapshot_image_tag))
        (self.logs_dir / "terminal-prefix-snapshot-capture.json").write_text(
            json.dumps(
                {
                    "schema_version": "terminal_prefix_snapshot_capture_v1",
                    "image_tag": self.snapshot_image_tag,
                    "image_id": image_id,
                    "container": container,
                    "after_call": self.snapshot_after_call,
                    "public_paths": list(self.public_paths),
                    "public_state": public_state,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    # Hooks for subclasses (the checkpoint ladder); the plain replay does nothing here.
    async def _before_replay_calls(
        self, environment: BaseEnvironment, gateway: TerminalEnvironmentToolGateway
    ) -> None:
        return None

    async def _before_replay_call(
        self, environment: BaseEnvironment, gateway: TerminalEnvironmentToolGateway, index: int
    ) -> None:
        return None

    async def _after_replay_call(
        self,
        environment: BaseEnvironment,
        gateway: TerminalEnvironmentToolGateway,
        index: int,
        history: list[dict[str, object]],
    ) -> None:
        return None

    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)

    def __init__(
        self,
        *args,
        public_paths: list[str],
        tool_calls: list[dict[str, object]],
        intervention: dict[str, object] | None = None,
        interventions: list[dict[str, object]] | None = None,
        snapshot_image_tag: str | None = None,
        snapshot_after_call: int | None = None,
        **kwargs,
    ) -> None:
        if intervention is not None and interventions is not None:
            raise ValueError("provide either one intervention or a group, not both")
        if (snapshot_image_tag is None) != (snapshot_after_call is None):
            raise ValueError("a prefix snapshot needs both an image tag and its last call")
        super().__init__(*args, **kwargs)
        self.public_paths = tuple(public_paths)
        self.tool_calls = tuple(tool_calls)
        self.snapshot_image_tag = snapshot_image_tag
        self.snapshot_after_call = snapshot_after_call
        self.interventions = tuple(
            interventions
            if interventions is not None
            else ([intervention] if intervention is not None else [])
        )

    @staticmethod
    @override
    def name() -> str:
        return AgentName.PI.value

    @override
    def version(self) -> str:
        return "controlled-replay-v1"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del instruction
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        session_dir = self.logs_dir / "pi" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / "controlled-replay.jsonl"
        gateway = TerminalEnvironmentToolGateway(
            environment=environment,
            batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
            public_paths=self.public_paths,
            observer_path=self.logs_dir / "terminal-public-state.jsonl",
            scratch_root=self.logs_dir / "delegated-scratch",
        )
        plan = (
            compile_terminal_replay_interventions(
                self.interventions,
                tool_call_count=len(self.tool_calls),
            )
            if self.interventions
            else ()
        )
        interventions_by_sequence = {item.after_call_sequence: item for item in plan}
        intervention_arm = plan[0].arm if plan else None
        applied_sequences: list[int] = []
        intervention_receipts_path = self.logs_dir / "terminal-interventions.jsonl"
        if plan:
            # Each replay attempt owns one receipt stream; never mix stale retries.
            intervention_receipts_path.write_text("", encoding="utf-8")
        snapshot_history: list[dict[str, object]] = []
        await self._before_replay_calls(environment, gateway)
        with session_path.open("w") as session:
            for index, call in enumerate(self.tool_calls):
                await self._before_replay_call(environment, gateway, index)
                tool_call_id = f"controlled-call-{index:03d}"
                tool_name = str(call["tool_name"])
                arguments = call["arguments"]
                if not isinstance(arguments, dict):
                    raise ValueError("controlled replay arguments must be an object")
                session.write(
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "toolCall",
                                        "id": tool_call_id,
                                        "name": tool_name,
                                        "arguments": arguments,
                                    }
                                ],
                            },
                        }
                    )
                    + "\n"
                )
                if call.get("environment_step_executed", True) is False:
                    result = gateway.record_parallel_rejection(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        source_text=call.get("source_error_text"),
                    )
                else:
                    result = await gateway.execute(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                session.write(
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "toolResult",
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                                "isError": result.is_error,
                                "content": [{"type": "text", "text": result.content}],
                            },
                        }
                    )
                    + "\n"
                )
                session.flush()
                snapshot_history.append(
                    {
                        "sequence": index,
                        "source_tool_call_id": call.get("source_tool_call_id"),
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "is_error": result.is_error,
                        "result": result.content,
                    }
                )
                if self.snapshot_image_tag is not None and index == self.snapshot_after_call:
                    await self._capture_prefix_snapshot(environment, gateway, snapshot_history)
                await self._after_replay_call(environment, gateway, index, snapshot_history)
                scheduled = interventions_by_sequence.get(index)
                if scheduled is not None:
                    intervention_before, intervention_after = await gateway.restore_public_state(
                        scheduled.target_state
                    )
                    receipt = {
                        "schema_version": "terminal_public_state_intervention_v1",
                        "arm": scheduled.arm,
                        "after_call_sequence": index,
                        "target_state": compact_public_file_state(scheduled.target_state),
                        "before": compact_public_file_state(intervention_before),
                        "after": compact_public_file_state(intervention_after),
                        "exact_readback": True,
                        "model_called": False,
                    }
                    with intervention_receipts_path.open("a", encoding="utf-8") as receipts:
                        receipts.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                        receipts.flush()
                    # Preserve the single-checkpoint receipt consumed by v1 probes.
                    if len(plan) == 1:
                        (self.logs_dir / "terminal-intervention.json").write_text(
                            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
                        )
                    applied_sequences.append(index)
        if len(applied_sequences) != len(plan):
            raise RuntimeError("terminal public-state intervention plan was not fully applied")
        if self.snapshot_image_tag is not None and not (
            self.logs_dir / "terminal-prefix-snapshot-capture.json"
        ).is_file():
            raise RuntimeError("the prefix snapshot was not captured")
        context.n_input_tokens = 0
        context.n_output_tokens = 0
        context.n_cache_tokens = 0
        context.metadata = {
            "transport": "host_terminal_controlled_replay",
            "model_called": False,
            "tool_call_count": len(self.tool_calls),
            "intervention_applied": bool(plan),
            "intervention_count": len(plan),
            "intervention_arm": intervention_arm,
            "intervention_after_call_sequence": (
                plan[0].after_call_sequence if len(plan) == 1 else None
            ),
            "intervention_after_call_sequences": applied_sequences,
        }


class HostTerminalRecoveryOpenCodeGoPi(HostDelegatedOpenCodeGoPi):
    """Replay a public prefix, repair its final checkpoint, then resume with MiMo."""

    def __init__(
        self,
        *args,
        public_paths: list[str],
        prefix_tool_calls: list[dict[str, object]],
        causal_arm: str,
        intervention: dict[str, object] | None,
        **kwargs,
    ) -> None:
        super().__init__(*args, public_paths=public_paths, **kwargs)
        self.prefix_tool_calls = tuple(prefix_tool_calls)
        if causal_arm not in {"original", "replacement", "matched_sham"}:
            raise ValueError("Terminal recovery causal arm is invalid")
        if (causal_arm == "original") != (intervention is None):
            raise ValueError("only the original arm may omit the recovery intervention")
        plans = (
            compile_terminal_replay_interventions(
                (intervention,),
                tool_call_count=len(self.prefix_tool_calls),
            )
            if intervention is not None
            else ()
        )
        if plans and plans[0].after_call_sequence != len(self.prefix_tool_calls) - 1:
            raise ValueError("Terminal recovery repair must end the frozen prefix")
        self.causal_arm = causal_arm
        self.intervention = plans[0] if plans else None

    async def _replay_public_prefix(
        self,
        gateway: TerminalEnvironmentToolGateway,
    ) -> tuple[dict[str, object], ...]:
        """Execute the frozen public prefix and persist the exact repair readback."""

        history: list[dict[str, object]] = []
        session_dir = self.logs_dir / "controlled-prefix"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / "replay.jsonl"
        with session_path.open("w", encoding="utf-8") as session:
            for index, call in enumerate(self.prefix_tool_calls):
                tool_name = str(call["tool_name"])
                arguments = call.get("arguments")
                if not isinstance(arguments, dict):
                    raise ValueError("Terminal recovery prefix arguments must be an object")
                call_id = f"recovery-prefix-{index:03d}"
                if call.get("environment_step_executed", True) is False:
                    result = gateway.record_parallel_rejection(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                else:
                    result = await gateway.execute(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                row = {
                    "sequence": index,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "is_error": result.is_error,
                    "result": result.content,
                }
                history.append(row)
                session.write(json.dumps(row, ensure_ascii=False) + "\n")
                session.flush()

        checkpoint = await gateway.snapshot_public_state()
        if self.intervention is None:
            before = after = checkpoint
        else:
            before, after = await gateway.restore_public_state(self.intervention.target_state)
        receipt = {
            "schema_version": "terminal_recovery_intervention_v1",
            "arm": self.causal_arm,
            "after_call_sequence": len(self.prefix_tool_calls) - 1,
            "checkpoint": compact_public_file_state(checkpoint),
            # Exact actor-visible bytes are needed to prove cross-arm checkpoint parity.
            "checkpoint_exact": [public_file_observation_payload(item) for item in checkpoint],
            "target_state": (
                [public_file_observation_payload(item) for item in self.intervention.target_state]
                if self.intervention is not None
                else None
            ),
            "before": compact_public_file_state(before),
            "after": compact_public_file_state(after),
            "after_exact": [public_file_observation_payload(item) for item in after],
            "exact_readback": True,
            "model_called": False,
        }
        (self.logs_dir / "terminal-recovery-intervention.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return tuple(history)

    @staticmethod
    def _continuation_instruction(
        *, task_instruction: str, history: tuple[dict[str, object], ...]
    ) -> str:
        """Expose only the original task and public executed history to the resumed actor."""

        return (
            "Continue the same task from a restored recovery checkpoint. The listed tool "
            "actions already happened in this sandbox; inspect the current state and finish "
            "the original task. Do not assume the prior attempt's final answer is correct.\n\n"
            f"ORIGINAL TASK:\n{task_instruction}\n\n"
            "PUBLIC EXECUTED HISTORY:\n"
            f"{json.dumps(history, ensure_ascii=False)}"
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        gateway = TerminalEnvironmentToolGateway(
            environment=environment,
            batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
            public_paths=self.public_paths,
            observer_path=self.logs_dir / "terminal-public-state.jsonl",
            scratch_root=self.logs_dir / "delegated-scratch",
        )
        history = await self._replay_public_prefix(gateway)
        await self._continue_with_gateway(
            instruction=self._continuation_instruction(
                task_instruction=instruction,
                history=history,
            ),
            gateway=gateway,
            context=context,
        )
        context.metadata = {
            **context.metadata,
            "recovery_prefix_tool_call_count": len(self.prefix_tool_calls),
            "recovery_intervention_applied": self.intervention is not None,
            "recovery_intervention_arm": self.causal_arm,
            "recovery_exact_readback": True,
            "continuation_mode": "same_actor_closed_loop",
        }


class HostTerminalHistoricalSessionContinuationPi(HostTerminalRecoveryOpenCodeGoPi):
    """Replay a natural failure and continue its exact persisted Pi context."""

    def __init__(
        self,
        *args,
        source_session_path: str,
        source_session_id: str,
        source_session_cwd: str,
        source_actor: dict[str, object],
        expected_terminal_stop_reason: str,
        task_instruction: str,
        continuation_treatment: str = "direct_continue",
        targeted_guidance: str | None = None,
        targeted_guidance_source: str | None = None,
        continuation_opening: str | None = None,
        prefix_snapshot_receipt: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, causal_arm="original", intervention=None, **kwargs)
        self.prefix_snapshot_receipt = prefix_snapshot_receipt
        self.continuation_opening = (
            continuation_opening.strip() if continuation_opening else None
        )
        self.source_session_path = Path(source_session_path).resolve()
        self.source_actor = normalize_terminal_source_actor(
            source_actor,
            runtime=self.runtime,
            max_output_tokens=self.host_pi_max_output_tokens,
        )
        self.source_session = load_frozen_pi_continuation_source(
            self.source_session_path,
            expected_instruction=task_instruction,
            expected_provider=self.runtime.provider,
            expected_model=self.runtime.model,
            expected_thinking=self.runtime.thinking,
            expected_terminal_stop_reason=expected_terminal_stop_reason,
            expected_cwd=Path(source_session_cwd),
        )
        if self.source_session.session_id != source_session_id:
            raise ValueError("historical Pi source session ID differs")
        if continuation_treatment not in {"direct_continue", "targeted_guidance"}:
            raise ValueError("historical Pi continuation treatment is invalid")
        normalized_guidance = targeted_guidance.strip() if targeted_guidance else None
        if continuation_treatment == "targeted_guidance":
            if not normalized_guidance or not targeted_guidance_source:
                raise ValueError("targeted historical continuation requires frozen guidance")
        elif normalized_guidance is not None or targeted_guidance_source is not None:
            raise ValueError("direct historical continuation cannot carry guidance")
        self.continuation_treatment = continuation_treatment
        self.targeted_guidance = normalized_guidance
        self.targeted_guidance_source = targeted_guidance_source
        calls = self.source_session.tool_receipt.tool_calls
        executions = {
            item.tool_call_id: item
            for item in self.source_session.tool_receipt.tool_executions
        }
        if len(calls) != len(self.prefix_tool_calls):
            raise ValueError("historical Pi source call count differs")
        for call, expected in zip(calls, self.prefix_tool_calls, strict=True):
            execution = executions.get(call.tool_call_id)
            if (
                expected.get("source_tool_call_id") != call.tool_call_id
                or expected.get("tool_name") != call.tool_name
                or expected.get("arguments") != call.arguments
                or execution is None
                or expected.get("source_is_error") is not execution.is_error
            ):
                raise ValueError("historical Pi source call contract differs")

    def _continuation_instruction(self) -> str:
        """Trigger the next turn without reconstructing history in a new prompt."""

        if self.continuation_treatment == "direct_continue":
            return (
                "Continue the original task from its observed failed final state. Inspect "
                "the current sandbox, correct any remaining problem, and finish the task. "
                "Do not assume the prior attempt's final answer is correct."
            )
        return historical_investigation_continuation(
            self.targeted_guidance or "", opening=self.continuation_opening
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        instruction_receipt = {
            "schema_version": "terminal_historical_instruction_identity_v1",
            "source_instruction": self.source_session.task_instruction,
            "received_instruction": instruction,
            "raw_equal": instruction == self.source_session.task_instruction,
            "boundary_normalized_equal": terminal_instructions_are_equivalent(
                self.source_session.task_instruction, instruction
            ),
            "normalization": "remove_at_most_one_terminal_line_ending",
        }
        (self.logs_dir / "terminal-historical-instruction-identity.json").write_text(
            json.dumps(instruction_receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not instruction_receipt["boundary_normalized_equal"]:
            raise ValueError("historical Pi runtime task instruction differs")
        gateway = TerminalEnvironmentToolGateway(
            environment=environment,
            batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
            public_paths=self.public_paths,
            observer_path=self.logs_dir / "terminal-public-state.jsonl",
            scratch_root=self.logs_dir / "delegated-scratch",
        )
        if self.prefix_snapshot_receipt is not None:
            history, verification = await restore_prefix_from_snapshot(
                environment=environment,
                gateway=gateway,
                receipt_path=str(self.prefix_snapshot_receipt),
                prefix_tool_calls=self.prefix_tool_calls,
            )
            (self.logs_dir / "terminal-prefix-snapshot-start.json").write_text(
                json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        else:
            history = await self._replay_public_prefix(gateway)
        expected_errors = tuple(
            bool(call["source_is_error"]) for call in self.prefix_tool_calls
        )
        replay_errors = tuple(bool(row["is_error"]) for row in history)
        if replay_errors != expected_errors:
            raise ValueError("historical Pi rescue replay error parity differs")

        source_bytes = self.source_session_path.read_bytes()
        staged_session = stage_pi_continuation_session(
            self.source_session_path,
            self.logs_dir / "pi" / "sessions",
        )
        self._managed_actor_session_id = self.source_session.session_id
        self._managed_actor_session_file = staged_session
        self._managed_actor_expected_message_count = self.source_session.message_count
        receipt_path = self.logs_dir / "terminal-historical-session-continuation.json"
        receipt = {
            "schema_version": "terminal_historical_pi_session_continuation_v1",
            "source_actor": self.source_actor,
            "source_session_id": self.source_session.session_id,
            "source_session_version": self.source_session.session_version,
            "source_session_cwd": self.source_session.session_cwd,
            "source_leaf_id": self.source_session.leaf_id,
            "source_message_count": self.source_session.message_count,
            "source_terminal_stop_reason": self.source_session.terminal_stop_reason,
            "replayed_tool_call_count": len(history),
            "execution_and_error_parity": True,
            "context_load_mode": "pi_explicit_session_file",
            "history_reconstructed_in_prompt": False,
            "continuation_treatment": self.continuation_treatment,
            "targeted_guidance": self.targeted_guidance,
            "targeted_guidance_source": self.targeted_guidance_source,
            "model_called_during_replay_or_staging": False,
            "official_verifier_executed": False,
            "status": "continuation_ready",
        }
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        await self._continue_with_gateway(
            instruction=self._continuation_instruction(),
            gateway=gateway,
            context=context,
        )
        if self.source_session_path.read_bytes() != source_bytes:
            raise RuntimeError("historical Pi source session changed during continuation")
        continued_bytes = staged_session.read_bytes()
        if not continued_bytes.startswith(source_bytes) or len(continued_bytes) <= len(source_bytes):
            raise RuntimeError("Pi continuation did not append to the staged source context")
        await self._commit_post_actor_state(environment)
        receipt.update(
            {
                "status": "actor_continuation_completed",
                "source_session_unchanged": True,
                "staged_session_appended": True,
            }
        )
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        context.metadata = {
            **dict(context.metadata or {}),
            "recovery_prefix_tool_call_count": len(self.prefix_tool_calls),
            "recovery_intervention_applied": False,
            "recovery_intervention_arm": "original",
            "recovery_exact_readback": True,
            "source_actor_matched": True,
            "source_pi_session_id": self.source_session.session_id,
            "source_pi_message_count": self.source_session.message_count,
            "pi_session_context_loaded": True,
            "history_reconstructed_in_prompt": False,
            "continuation_treatment": self.continuation_treatment,
            "targeted_guidance_source": self.targeted_guidance_source,
            "continuation_mode": "exact_pi_session_context_branch",
        }


class HostTerminalForwardRecoveryOpenCodeGoPi(HostDelegatedOpenCodeGoPi):
    """Restore an observed failed final state and let the same actor continue."""

    def __init__(
        self,
        *args,
        public_paths: list[str],
        failed_final_state: list[dict[str, object]],
        public_history: list[dict[str, object]],
        repair_state: list[dict[str, object]] | None,
        source_actor: dict[str, object],
        state_coverage: dict[str, object],
        source_trial_name: str,
        continuation_prompt_mode: str = TERMINAL_FORWARD_DIRECT_PROMPT,
        targeted_guidance: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, public_paths=public_paths, **kwargs)
        self.failed_final_state = decode_exact_public_file_state(failed_final_state)
        if {item.path for item in self.failed_final_state} != set(self.public_paths):
            raise ValueError("forward recovery final state does not cover public paths")
        self.public_history = normalize_terminal_public_history(public_history)
        self.source_actor = normalize_terminal_source_actor(
            source_actor,
            runtime=self.runtime,
            max_output_tokens=self.host_pi_max_output_tokens,
        )
        self.state_coverage = normalize_terminal_state_coverage_actor_attestation(state_coverage)
        if not source_trial_name or any(character in source_trial_name for character in "/\\"):
            raise ValueError("forward recovery source trial name is invalid")
        self.source_trial_name = source_trial_name
        self.continuation_prompt_mode = normalize_terminal_forward_prompt_mode(
            continuation_prompt_mode
        )
        self.targeted_guidance = (
            targeted_guidance.strip() if targeted_guidance is not None else None
        )
        if self.continuation_prompt_mode == TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT:
            if not self.targeted_guidance:
                raise ValueError("targeted continuation requires method-generated guidance")
        elif self.targeted_guidance is not None:
            raise ValueError("untargeted continuation cannot carry method-generated guidance")
        self.repair_state = (
            decode_exact_public_file_state(repair_state) if repair_state is not None else None
        )
        if (
            self.continuation_prompt_mode == TERMINAL_FORWARD_CRITIQUE_PROMPT
            and self.repair_state is not None
        ):
            raise ValueError("end-of-trace critique cannot include a pre-applied repair")
        if (
            self.continuation_prompt_mode == TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT
            and self.repair_state is not None
        ):
            raise ValueError("targeted continuation cannot include a pre-applied repair")
        if self.repair_state is not None and not {item.path for item in self.repair_state} <= set(
            self.public_paths
        ):
            raise ValueError("forward repair escaped the observed public state")

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        gateway = TerminalEnvironmentToolGateway(
            environment=environment,
            batched_public_snapshot=getattr(self, "batched_public_snapshot", False),
            public_paths=self.public_paths,
            observer_path=self.logs_dir / "terminal-public-state.jsonl",
            scratch_root=self.logs_dir / "delegated-scratch",
        )
        initial_state = await gateway.snapshot_public_state()
        # Reconstruct the admitted public projection before a policy-selected repair.
        _, failed_after = await gateway.restore_public_state(self.failed_final_state)
        repair_after = failed_after
        if self.repair_state is not None:
            _, repair_after = await gateway.restore_public_state(self.repair_state)
        receipt = {
            "schema_version": "terminal_forward_recovery_restore_v2",
            "source_actor": self.source_actor,
            "source_trial_name": self.source_trial_name,
            "continuation_prompt_mode": self.continuation_prompt_mode,
            "targeted_guidance": self.targeted_guidance,
            "state_coverage_attestation": self.state_coverage,
            "initial_state": [public_file_observation_payload(item) for item in initial_state],
            "failed_public_file_state": [
                public_file_observation_payload(item) for item in self.failed_final_state
            ],
            "failed_state_readback": [
                public_file_observation_payload(item) for item in failed_after
            ],
            "repair_state": (
                [public_file_observation_payload(item) for item in self.repair_state]
                if self.repair_state is not None
                else None
            ),
            "continuation_start_state": [
                public_file_observation_payload(item) for item in repair_after
            ],
            "exact_readback": True,
            "model_called_during_restore": False,
        }
        (self.logs_dir / "terminal-forward-recovery-restore.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        await self._continue_with_gateway(
            instruction=build_terminal_forward_continuation_instruction(
                task_instruction=instruction,
                history=self.public_history,
                prompt_mode=self.continuation_prompt_mode,
                targeted_guidance=self.targeted_guidance,
            ),
            gateway=gateway,
            context=context,
        )
        context.metadata = {
            **context.metadata,
            "forward_failed_public_projection_restored": True,
            "forward_repair_applied": self.repair_state is not None,
            "forward_recovery_exact_readback": True,
            "source_actor_matched": True,
            "evidence_fidelity": "live_fresh_sandbox_from_public_projection",
            "replay_scope": "public_projection_restore",
            "restored_state_scope": "declared_public_files",
            "unsupported_state_outcome_determining": False,
            "causal_claim_admissible": False,
            "continuation_mode": "same_actor_from_failed_public_projection",
            "continuation_prompt_mode": self.continuation_prompt_mode,
            "targeted_guidance_supplied": self.targeted_guidance is not None,
        }
