"""Delegate Terminal tools without placing the model credential in the task container."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import shlex
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from lh_platform.benchmarks.terminal_live_acceptance_portfolio import (
    IsolatedGuidanceExecutor,
    TerminalSeparateEnvironmentPortfolioGate,
)
from lh_platform.benchmarks.terminal_managed_runtime_identity import (
    EFFECT_LEDGER_RUNTIME_METHODS,
    FIXED_TWO_STAGE_RUNTIME_METHOD,
    OUTPUT_PROGRESS_RUNTIME_METHODS,
    OUTPUT_SEMANTIC_RUNTIME_METHODS,
)
from lh_platform.benchmarks.terminal_public_state import (
    PublicFileObservation,
    is_exact_restorable_public_file,
    public_file_observation_payload,
    validate_actor_visible_path,
)
from lh_platform.benchmarks.terminal_resource_driver import TerminalResourceDriver
from lh_platform.benchmarks.terminal_runtime_check import (
    CHECKER_PREREQUISITE_MISSING_EXIT_CODE,
    CHECKER_PREREQUISITE_MISSING_MARKER,
    EXACT_OUTPUT_VERDICT_PROTOCOL,
    EXIT_STATUS_VERDICT_PROTOCOL,
    RuntimeCheckContract,
    RuntimePredicateRepairContract,
    bind_runtime_check_configuration,
    check_prerequisites_available,
    conditional_runtime_repair_options,
    interpret_runtime_check_result,
    is_actor_guidance_repair_arguments,
    pending_runtime_check_candidates,
    pending_runtime_repair_candidates,
    record_runtime_check,
    record_runtime_check_evidence,
    runtime_check_blocking_fact,
)
from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    ACTION_EXECUTION_HEAD_CHARACTERS,
    ACTION_EXECUTION_TAIL_CHARACTERS,
    ActionExecutionSummary,
    ActionPhase,
    EvidenceState,
    PortableRuntimeKernel,
    ResourceObservation,
    TaskStatus,
)
from lh_platform.tracefix_bench.budget_affordability import repair_admissible
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionCandidate,
    RuntimeJointActionType,
    compile_runtime_joint_action_slate,
    runtime_failure_diagnosis_incident_key,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    BOUNDED_TRACEDP_DEV_METHODS,
    EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS,
    FAILURE_DIAGNOSIS_JOINT_METHODS,
    FAILURE_DIAGNOSIS_ONLY_JOINT_METHODS,
    FIXED_JOINT_SLATE_V3_METHOD,
    FIXED_JOINT_SLATE_V5_METHOD,
    RUNTIME_OPTION_CHAIN_METHODS,
    RUNTIME_OPTION_FOLLOWUP_TRIGGER,
    SEMANTIC_STABILITY_JOINT_METHODS,
    SETTLEMENT_ONLY_JOINT_METHODS,
    SETTLEMENT_REVIEW_JOINT_METHODS,
    TIME_RESERVE_JOINT_METHODS,
    TRACEDP_RESIDUAL_METHODS,
    TRACEDP_RESIDUAL_RUNTIME_METHOD,
    TRACEDP_RESIDUAL_RUNTIME_V2_METHOD,
    TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
    RuntimeJointSlatePolicy,
    RuntimeOptionChain,
    project_runtime_option_action_domain,
)
from lh_platform.test_time_adaptation.runtime_output_obligation import (
    RuntimeInterventionDeliveryStatus,
    RuntimeInterventionEffectLedger,
    RuntimeInterventionEffectStatus,
    RuntimeOutputObligation,
    compile_runtime_output_obligations,
)
from lh_platform.test_time_adaptation.runtime_trace_dp import (
    DynamicTraceDPRuntimePolicy,
    RuntimeRepairCapability,
    compile_runtime_progress_candidates,
    compile_runtime_semantic_candidates,
)

_PUBLIC_STATE_OBSERVATION_TIMEOUT_SECONDS = 60
# Files above this size keep the per-path download in a batched snapshot.
_BATCHED_SNAPSHOT_MAX_BYTES = 8 * 1024 * 1024
_PUBLIC_STATE_OBSERVER_CONCURRENCY = 2
_BASH_PROCESS_GROUP_KILL_AFTER_SECONDS = 5
_BASH_OUTER_TIMEOUT_MARGIN_SECONDS = 30
_BASH_SESSION_COMPLETION_RESERVE_SECONDS = 5
_BASH_CONTAINMENT_MODE = "timeout_process_group_procfs_pipefail_v5"
_BASH_PIPELINE_EXIT_STATUS = "gnu_bash_rightmost_nonzero_v1"
_REGISTERED_CHECK_EXECUTION_PREFIX = r"""
set -E -o pipefail
trap 'status=$?; trap - ERR; exit "$status"' ERR
""".strip()
OUTPUT_PROGRESS_ACTION_THRESHOLD = 8
OUTPUT_SEMANTIC_STABILITY_ACTION_THRESHOLD = 2
ACTOR_FAILURE_DIAGNOSIS_THRESHOLD = 2
ACTOR_CONVERGENCE_RESERVE_SECONDS = 120
ACTOR_TOOL_RECEIPT_SCHEMA = "terminal_actor_tool_receipt_v1"


class TerminalToolEnvironment(Protocol):
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any: ...

    async def is_file(self, path: str, user: str | int | None = None) -> bool: ...

    async def download_file(self, source_path: str, target_path: Path | str) -> None: ...

    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...


@dataclass(frozen=True)
class DelegatedToolResult:
    content: str
    is_error: bool
    environment_step_executed: bool
    details: dict[str, Any]


def _actor_execution_summary(result: DelegatedToolResult) -> ActionExecutionSummary:
    """Bound the exact pre-supervision tool result to diagnostic head and tail text."""

    content_length = len(result.content)
    content_head = result.content[:ACTION_EXECUTION_HEAD_CHARACTERS]
    remaining = result.content[len(content_head):]
    content_tail = (
        remaining
        if len(remaining) <= ACTION_EXECUTION_TAIL_CHARACTERS
        else result.content[-ACTION_EXECUTION_TAIL_CHARACTERS:]
    )

    def integer_detail(name: str) -> int | None:
        value = result.details.get(name)
        return value if type(value) is int else None

    containment = result.details.get("process_group_containment")
    return ActionExecutionSummary(
        content_head=content_head,
        content_tail=content_tail,
        content_length_chars=content_length,
        content_truncated=len(content_head) + len(content_tail) < content_length,
        is_error=result.is_error,
        environment_step_executed=result.environment_step_executed,
        return_code=integer_detail("return_code"),
        requested_timeout_seconds=integer_detail(
            "requested_command_timeout_seconds"
        ),
        effective_timeout_seconds=integer_detail(
            "effective_command_timeout_seconds"
        ),
        process_group_containment=(
            containment if isinstance(containment, str) and containment else None
        ),
    )


@dataclass(frozen=True)
class ActorStateSubmission:
    """Bind one time-triggered joint choice to its later Pi RPC receipt."""

    action_id: str
    decision_id: str
    candidate: RuntimeJointActionCandidate
    actor_time_budget_ms: int
    actor_time_remaining_ms: int
    convergence_reserve_ms: int


@dataclass(frozen=True)
class ActorSettlementReview:
    """Bind one natural-settlement choice to the later Pi prompt response."""

    action_id: str
    decision_id: str
    candidate: RuntimeJointActionCandidate
    notice: str


@dataclass(frozen=True)
class ActorNaturalSettlementPrompt:
    """What an option-chain method asks of the Actor after it settled on its own."""

    trigger_action_id: str
    notice: str
    interventions: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class BashContainmentCapability:
    """Describe the exact container primitives required by the Bash supervisor."""

    mode: str
    bash_version: str
    setsid_version: str
    kill_after_seconds: int
    outer_timeout_margin_seconds: int
    process_table_source: str
    pipeline_exit_status: str

    def payload(self) -> dict[str, object]:
        """Return the machine-readable capability bound into runtime receipts."""

        return {
            "mode": self.mode,
            "supported": True,
            "bash_version": self.bash_version,
            "setsid_version": self.setsid_version,
            "kill_after_seconds": self.kill_after_seconds,
            "outer_timeout_margin_seconds": self.outer_timeout_margin_seconds,
            "process_table_source": self.process_table_source,
            "pipeline_exit_status": self.pipeline_exit_status,
            "scope": "over_budget_same_process_group_descendants",
            "external_cancellation": "unsupported_session_becomes_non_resumable",
        }


def _observation_payload(observation: PublicFileObservation) -> dict[str, object]:
    return public_file_observation_payload(observation)


_BASH_SUPERVISOR_SCRIPT = r"""
set +e
live_process_group_rows() {
  target_pgid=$1
  for stat_path in /proc/[0-9]*/stat; do
    [ -r "$stat_path" ] || continue
    IFS= read -r stat_line < "$stat_path" || continue
    stat_tail=${stat_line##*) }
    read -r process_state process_ppid process_pgid process_rest <<< "$stat_tail"
    if [ "$process_pgid" = "$target_pgid" ] && [ "$process_state" != "Z" ]; then
      printf "%s %s\n" "${stat_path#/proc/}" "$process_state"
    fi
  done
}
actor_command=$1
timeout_seconds=$2
kill_after_seconds=$3
marker_path=$(mktemp /tmp/longh-bash-expired.XXXXXX)
setsid bash -o pipefail -c "$actor_command" &
actor_pgid=$!
setsid bash -c '
  sleep "$1"
  printf "expired\n" > "$2"
  kill -TERM -- "-$3" 2>/dev/null
  sleep "$4"
  kill -KILL -- "-$3" 2>/dev/null
' _ "$timeout_seconds" "$marker_path" "$actor_pgid" "$kill_after_seconds" &
watchdog_pgid=$!
wait "$actor_pgid"
actor_return_code=$?
if [ -s "$marker_path" ]; then
  wait "$watchdog_pgid" 2>/dev/null
  live_rows=$(live_process_group_rows "$actor_pgid")
  rm -f "$marker_path"
  if [ -n "$live_rows" ]; then
    printf "Bash containment left live process-group members:\n%s\n" "$live_rows" >&2
    exit 125
  fi
  exit 124
fi
kill -TERM -- "-$watchdog_pgid" 2>/dev/null
wait "$watchdog_pgid" 2>/dev/null
rm -f "$marker_path"
exit "$actor_return_code"
""".strip()


def _contained_bash_command(
    command: str,
    *,
    timeout_seconds: int,
) -> str:
    """Supervise one Actor Bash in a separately addressable process group."""

    return " ".join(
        (
            "bash -c",
            shlex.quote(_BASH_SUPERVISOR_SCRIPT),
            "longh-bash-supervisor",
            shlex.quote(command),
            str(timeout_seconds),
            str(_BASH_PROCESS_GROUP_KILL_AFTER_SECONDS),
        )
    )


async def probe_bash_containment_capability(
    environment: TerminalToolEnvironment,
) -> BashContainmentCapability:
    """Reject an image before Actor execution when supervisor primitives are absent."""

    command = r"""
set -e
for executable in bash setsid mktemp sleep rm; do
  command -v "$executable" >/dev/null
done
type kill >/dev/null
test -r /proc/self/stat
if bash -o pipefail -c 'exit 7 | :'; then
  exit 1
else
  test "$?" -eq 7
fi
bash --version | head -n 1
setsid --version | head -n 1
""".strip()
    result = await environment.exec(command, cwd="/app", timeout_sec=30)
    lines = [line.strip() for line in str(result.stdout or "").splitlines() if line.strip()]
    if result.return_code != 0 or len(lines) != 2:
        raise RuntimeError("task image lacks the Bash process-group containment capability")
    if not lines[0].startswith("GNU bash, version ") or not lines[1].startswith(
        "setsid from util-linux "
    ):
        raise RuntimeError("task image returned unsupported Bash supervisor versions")
    return BashContainmentCapability(
        mode=_BASH_CONTAINMENT_MODE,
        bash_version=lines[0],
        setsid_version=lines[1],
        kill_after_seconds=_BASH_PROCESS_GROUP_KILL_AFTER_SECONDS,
        outer_timeout_margin_seconds=_BASH_OUTER_TIMEOUT_MARGIN_SECONDS,
        process_table_source="linux_proc_stat",
        pipeline_exit_status=_BASH_PIPELINE_EXIT_STATUS,
    )


def _parse_batched_snapshot(
    stdout: str, count: int
) -> dict[int, tuple[str, bytes | None]] | None:
    """Index -> (stat header or ``@@MISSING``, bytes or ``None`` for a per-path read)."""

    entries: dict[int, tuple[str, bytes | None]] = {}
    index: int | None = None
    header = ""
    lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("@@LH "):
            if index is not None:
                return None
            head, _, rest = line[5:].partition(" ")
            if not head.isdigit():
                return None
            index, header, lines = int(head), rest, []
        elif line.startswith("@@END "):
            if index is None or line[6:] != str(index) or index in entries:
                return None
            payload: bytes | None
            if header == "@@MISSING" or not header.startswith("regular"):
                payload = None
            elif lines in (["@@BIG"], ["@@B64FAIL"]):
                payload = None
            else:
                try:
                    payload = base64.b64decode("".join(lines), validate=True)
                except (binascii.Error, ValueError):
                    return None
            entries[index] = (header, payload)
            index = None
        elif index is not None:
            lines.append(line)
        else:
            return None
    if index is not None or set(entries) != set(range(count)):
        return None
    return entries


async def _discard_task(task: asyncio.Future | None) -> None:
    """Cancel an overlapped read that an earlier failure made pointless."""

    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


class TerminalEnvironmentToolGateway:
    """Execute one sequential native tool call in Harbor's task environment."""

    # Zero-model replays of sources collected before process-group containment set this:
    # Bash then runs exactly as it did when those sources were recorded.
    source_era_bash: bool = False
    # Replay only: a call the source did not execute returns the source's own result text
    # (a tool-layer refusal such as "Tool edit not found"), not a replay-generated one.
    source_text_refusals: bool = False

    def __init__(
        self,
        *,
        environment: TerminalToolEnvironment,
        public_paths: tuple[str, ...],
        observer_path: Path,
        scratch_root: Path,
        observe_public_state: bool = True,
        resource_driver: TerminalResourceDriver | None = None,
        runtime_kernel: PortableRuntimeKernel | None = None,
        restore_missing_prerequisites: bool = False,
        progress_trigger_threshold: int | None = None,
        runtime_supervisor: DynamicTraceDPRuntimePolicy | None = None,
        joint_runtime_policy: RuntimeJointSlatePolicy | None = None,
        intervention_effect_ledger: RuntimeInterventionEffectLedger | None = None,
        bash_containment_capability: BashContainmentCapability | None = None,
        actor_time_budget_seconds: int | None = None,
        selector_time_observation_enabled: bool = False,
        initial_sequence: int = 0,
        check_contracts: tuple[RuntimeCheckContract, ...] = (),
        repair_contracts: tuple[RuntimePredicateRepairContract, ...] = (),
        targeted_repair_trial_gate: TerminalSeparateEnvironmentPortfolioGate | None = None,
        isolated_guidance_executor: IsolatedGuidanceExecutor | None = None,
        predicate_diagnosis_enabled: bool = False,
        allow_check_registration: bool = False,
        require_check_registration_phase: bool = False,
        concurrent_runtime_observation: bool = False,
        batched_public_snapshot: bool = False,
    ) -> None:
        self.environment = environment
        # One exec reads every public path (type, mode, bytes) instead of a stat and a
        # download per path; the first batched snapshot is checked against the per-path
        # read and batching turns itself off on any difference.
        self.batched_public_snapshot = batched_public_snapshot
        self._batched_snapshot_verified = False
        self.batched_snapshot_events: list[dict[str, object]] = []
        # The Runtime's resource observation and the public
        # snapshot read the same state at each tool boundary; when on, they overlap
        # instead of running one after the other. Observed facts are unchanged.
        self.concurrent_runtime_observation = concurrent_runtime_observation

        self.public_paths = tuple(validate_actor_visible_path(path) for path in public_paths)
        if len(self.public_paths) != len(set(self.public_paths)):
            raise ValueError("public paths must be unique")
        if observe_public_state and not self.public_paths:
            raise ValueError("public paths are required when public state observation is enabled")
        self.observe_public_state = observe_public_state
        self.observer_path = observer_path
        self.scratch_root = scratch_root
        if (resource_driver is None) != (runtime_kernel is None):
            raise ValueError("resource driver and runtime kernel must be configured together")
        if resource_driver is not None and resource_driver.environment is not environment:
            raise ValueError("resource driver must use the gateway environment session")
        self.resource_driver = resource_driver
        self.runtime_kernel = runtime_kernel
        if check_contracts and runtime_kernel is None:
            raise ValueError("check contracts require the managed runtime")
        self.check_contracts = tuple(check_contracts)
        # Pool-affordability gate, attached per attempt by the host.
        self._budget_gate = None
        self._last_visible_chars = 0
        self.budget_gate_events: list[dict[str, object]] = []
        self.repair_contracts = tuple(repair_contracts)
        self.targeted_repair_trial_gate = targeted_repair_trial_gate
        self._isolated_guidance_executor = isolated_guidance_executor
        if targeted_repair_trial_gate is not None and not repair_contracts:
            raise ValueError("candidate trial gate requires targeted repair contracts")
        if (
            targeted_repair_trial_gate is not None
            and targeted_repair_trial_gate.main_environment is not environment
        ):
            raise ValueError("candidate trial gate must protect this Gateway environment")
        if any(is_actor_guidance_repair_arguments(item.arguments) for item in self.repair_contracts):
            if targeted_repair_trial_gate is None or isolated_guidance_executor is None:
                raise ValueError(
                    "actor guidance repairs require isolated trial and an isolated Actor runner"
                )
        if repair_contracts and (
            runtime_kernel is None or joint_runtime_policy is None
        ):
            raise ValueError("targeted repairs require the managed joint runtime")
        repair_ids = [item.repair_id for item in self.repair_contracts]
        if len(repair_ids) != len(set(repair_ids)):
            raise ValueError("targeted repair contract identities repeat")
        self.allow_check_registration = allow_check_registration
        if allow_check_registration and runtime_kernel is None:
            raise ValueError("Actor check registration requires the managed runtime")
        if require_check_registration_phase and not allow_check_registration:
            raise ValueError("required check registration phase needs Actor registration")
        self.require_check_registration_phase = require_check_registration_phase
        completed_phase_actions = [
            action
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "actor"
            and action.tool_name == "complete_runtime_check_registration"
            and action.phase is ActionPhase.COMPLETED
        ]
        if len(completed_phase_actions) > 1:
            raise ValueError("check registration phase was completed more than once")
        self._check_registration_phase_completion = (
            completed_phase_actions[0] if completed_phase_actions else None
        )
        if predicate_diagnosis_enabled and ((not check_contracts and not allow_check_registration) or joint_runtime_policy is None):
            raise ValueError("predicate diagnosis requires checks and a joint selector")
        self.predicate_diagnosis_enabled = predicate_diagnosis_enabled
        self._diagnosed_predicate_incident_ids = {
            str(action.arguments["incident_id"])
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "runtime" and action.tool_name == "runtime_predicate_guidance"
            and action.phase is ActionPhase.COMPLETED
        }
        if restore_missing_prerequisites and runtime_kernel is None:
            raise ValueError("prerequisite restoration requires the managed runtime")
        if runtime_supervisor is not None and runtime_kernel is None:
            raise ValueError("runtime supervisor requires the managed runtime")
        if joint_runtime_policy is not None and runtime_kernel is None:
            raise ValueError("joint runtime policy requires the managed runtime")
        if (
            sum(
                (
                    bool(restore_missing_prerequisites),
                    runtime_supervisor is not None,
                    joint_runtime_policy is not None,
                )
            )
            > 1
        ):
            raise ValueError("runtime intervention policies are mutually exclusive")
        self.restore_missing_prerequisites = restore_missing_prerequisites
        if progress_trigger_threshold is not None and int(progress_trigger_threshold) < 1:
            raise ValueError("progress trigger threshold must be positive")
        self.progress_trigger_threshold = (
            OUTPUT_PROGRESS_ACTION_THRESHOLD
            if progress_trigger_threshold is None
            else int(progress_trigger_threshold)
        )
        self.runtime_supervisor = runtime_supervisor
        self.joint_runtime_policy = joint_runtime_policy
        self.intervention_effect_ledger = intervention_effect_ledger
        if (
            runtime_supervisor is not None
            and runtime_supervisor.policy_id in EFFECT_LEDGER_RUNTIME_METHODS
            and intervention_effect_ledger is None
        ):
            raise ValueError("output-supervision policy requires an intervention effect ledger")
        if joint_runtime_policy is not None and intervention_effect_ledger is None:
            raise ValueError("joint runtime policy requires an intervention effect ledger")
        self.bash_containment_capability = bash_containment_capability
        if actor_time_budget_seconds is not None and actor_time_budget_seconds < 1:
            raise ValueError("Actor time budget must be positive")
        if (
            joint_runtime_policy is not None
            and joint_runtime_policy.method_id == FIXED_JOINT_SLATE_V3_METHOD
            and actor_time_budget_seconds is None
        ):
            raise ValueError("time-aware joint runtime requires the Actor time budget")
        self.actor_time_budget_seconds = actor_time_budget_seconds
        bounded_selector = (
            joint_runtime_policy is not None
            and joint_runtime_policy.method_id in BOUNDED_TRACEDP_DEV_METHODS
        )
        option_chain_selector = (
            joint_runtime_policy is not None
            and joint_runtime_policy.method_id in RUNTIME_OPTION_CHAIN_METHODS
        )
        if (
            bounded_selector or option_chain_selector
        ) and actor_time_budget_seconds is None:
            raise ValueError("time-aware selector requires the Actor time budget")
        # A planner whose state includes remaining time must receive that state on every slate.
        self.selector_time_observation_enabled = (
            selector_time_observation_enabled
            or bounded_selector
            or option_chain_selector
        )
        self._actor_attempt_deadline: float | None = None
        self._fatal_containment_error: RuntimeError | None = None
        self._failed_runtime_intervention_versions: set[tuple[str, int]] = set(
            (
                runtime_supervisor.failed_resource_versions
                if runtime_supervisor is not None
                else set()
            ).union(
                runtime_kernel.failed_runtime_intervention_versions
                if runtime_kernel is not None
                else set()
            )
        )
        self._lock = asyncio.Lock()
        if initial_sequence < 0:
            raise ValueError("terminal gateway sequence cannot be negative")
        self._sequence = initial_sequence
        existing_observer_rows = (
            sum(bool(line.strip()) for line in observer_path.read_text().splitlines())
            if observer_path.is_file()
            else 0
        )
        if runtime_kernel is not None and existing_observer_rows != initial_sequence:
            raise ValueError("terminal observer history differs from Kernel actor actions")
        self._observer_sequence = existing_observer_rows
        self.runtime_check_execution_path = (
            self.observer_path.parent / "runtime-check-executions.jsonl"
        )
        self.runtime_repair_execution_path = (
            self.observer_path.parent / "runtime-targeted-repair-executions.jsonl"
        )
        existing_check_execution_rows = (
            sum(
                bool(line.strip())
                for line in self.runtime_check_execution_path.read_text().splitlines()
            )
            if self.runtime_check_execution_path.is_file()
            else 0
        )
        recorded_check_execution_actions = [
            action
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "runtime"
            and action.tool_name == "runtime_registered_check_execution"
        ]
        if existing_check_execution_rows != len(recorded_check_execution_actions):
            raise ValueError("runtime check execution receipts differ from Kernel history")
        self._runtime_check_execution_sequence = existing_check_execution_rows
        existing_repair_execution_rows = (
            sum(
                bool(line.strip())
                for line in self.runtime_repair_execution_path.read_text().splitlines()
            )
            if self.runtime_repair_execution_path.is_file()
            else 0
        )
        finished_repair_actions = [
            action
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "runtime"
            and action.tool_name == "runtime_targeted_repair_execution"
            and action.phase not in {ActionPhase.PROPOSED, ActionPhase.STARTED}
        ]
        if existing_repair_execution_rows != len(finished_repair_actions):
            raise ValueError("runtime targeted repair receipts differ from Kernel history")
        self._runtime_repair_execution_sequence = existing_repair_execution_rows
        self._handled_targeted_repair_action_ids = {
            str(action.arguments["decision_action_id"])
            for action in finished_repair_actions
            if isinstance(action.arguments.get("decision_action_id"), str)
        }
        self._runtime_supervision_disabled_reason = (
            runtime_kernel.runtime_supervision_disabled_reason
            if runtime_kernel is not None
            else None
        )
        self._output_resource_ids = frozenset(
            resource_id
            for node in (runtime_kernel.task_nodes.values() if runtime_kernel else ())
            for resource_id in node.required_output_resource_ids
        )
        self._guided_output_versions = {
            (str(action.arguments["resource_id"]), int(action.arguments["resource_version"]))
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "runtime"
            and action.tool_name == "runtime_guidance"
            and isinstance(action.arguments.get("resource_id"), str)
            and isinstance(action.arguments.get("resource_version"), int)
        }
        self._actor_actions_since_output_progress = (
            self._replay_actor_output_progress() if runtime_kernel is not None else 0
        )
        self._semantic_checked_obligation_versions = {
            self._decode_obligation_key(action.arguments.get("obligation_version_key"))
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "runtime" and action.tool_name == "runtime_semantic_guidance"
        }
        self._actor_convergence_requested = any(
            action.actor == "runtime"
            and action.tool_name == "runtime_convergence_guidance"
            and action.phase is ActionPhase.COMPLETED
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
        )
        latest_failure = (
            tuple(runtime_kernel.actor_failure_boundaries.values())[-1]
            if runtime_kernel is not None and runtime_kernel.actor_failure_boundaries
            else None
        )
        self._actor_failures_since_resource_change = (
            latest_failure.failure_count_since_resource_change
            if latest_failure is not None
            and latest_failure.resource_versions
            == tuple(sorted(runtime_kernel.resource_versions.items()))
            else 0
        )
        self._diagnosed_failure_action_ids = {
            str(action.arguments["failure_action_id"])
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "runtime"
            and action.tool_name == "runtime_failure_diagnosis_guidance"
            and action.phase is ActionPhase.COMPLETED
            and isinstance(action.arguments.get("failure_action_id"), str)
        }
        self._diagnosed_failure_incident_keys = {
            runtime_failure_diagnosis_incident_key(
                resource_versions=self._decode_resource_versions(
                    action.arguments.get("failure_resource_versions")
                ),
                obligation_version_key=self._decode_obligation_key(
                    action.arguments.get("obligation_version_key")
                ),
            )
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "runtime"
            and action.tool_name == "runtime_failure_diagnosis_guidance"
            and action.phase is ActionPhase.COMPLETED
        }
        self._diagnosed_obligation_versions = {
            self._decode_obligation_key(action.arguments.get("obligation_version_key"))
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "runtime"
            and action.tool_name == "runtime_failure_diagnosis_guidance"
            and action.phase is ActionPhase.COMPLETED
        }

        bind_runtime_check_configuration(
            self.observer_path.parent / "runtime-check-contracts.json", self.check_contracts,
            diagnosis_enabled=self.predicate_diagnosis_enabled,
            registration_enabled=self.allow_check_registration,
            registration_phase_required=self.require_check_registration_phase,
        )
        if runtime_kernel is not None:
            for action in runtime_kernel.actions.values():
                if (action.actor == "actor" and action.tool_name == "register_runtime_check"
                        and action.phase is ActionPhase.COMPLETED):
                    contract = self._parse_actor_check(action.arguments)
                    if runtime_kernel.acceptance_predicate_enabled:
                        check_id = f"actor-check:{action.action_id}"
                        predicate_id = runtime_kernel.predicate_id_for_check(check_id)
                        if predicate_id is None:
                            predicate_id = (
                                runtime_kernel.register_or_bind_actor_check_predicate(
                                    registration_action_id=action.action_id,
                                    task_id=contract.task_id,
                                    proposition=contract.proposition,
                                    source_excerpt=str(
                                        action.arguments["source_excerpt"]
                                    ),
                                    resource_ids=contract.resource_ids,
                                )
                            )
                        contract = contract.with_predicate_binding(
                            check_id=check_id,
                            predicate_id=predicate_id,
                        )
                    self.check_contracts += (contract,)
        self._handled_registered_check_versions = {
            (
                int(action.arguments["check_index"]),
                tuple(
                    (str(item[0]), int(item[1]))
                    for item in action.arguments["resource_versions"]
                ),
            )
            for action in (runtime_kernel.actions.values() if runtime_kernel else ())
            if action.actor == "runtime"
            and action.tool_name
            in {
                "runtime_registered_check_guidance",
                "runtime_registered_check_execution",
            }
            and action.phase not in {ActionPhase.PROPOSED, ActionPhase.STARTED}
        }

    @property
    def check_registration_phase_status(self) -> str | None:
        """Expose the durable Actor judgment that opened side-effect execution."""

        if self._check_registration_phase_completion is None:
            return None
        return str(self._check_registration_phase_completion.arguments["status"])

    @property
    def _runtime_policy_id(self) -> str:
        """Return the configured intervention identity for receipts and notices."""

        if self.joint_runtime_policy is not None:
            return self.joint_runtime_policy.method_id
        if self.runtime_supervisor is not None:
            return self.runtime_supervisor.policy_id
        return "shared_input_protection_v1"

    @staticmethod
    def _decode_obligation_key(
        raw: object,
    ) -> tuple[str, str, tuple[tuple[str, int], ...], str]:
        """Decode the exact task-level version key stored on a Kernel action."""

        if not isinstance(raw, list) or len(raw) != 4 or not isinstance(raw[2], list):
            raise ValueError("runtime semantic obligation key is invalid")
        versions = tuple((str(item[0]), int(item[1])) for item in raw[2])
        return (str(raw[0]), str(raw[1]), versions, str(raw[3]))

    @staticmethod
    def _encode_obligation_key(
        key: tuple[str, str, tuple[tuple[str, int], ...], str],
    ) -> list[object]:
        return [key[0], key[1], [list(item) for item in key[2]], key[3]]

    @staticmethod
    def _decode_resource_versions(raw: object) -> tuple[tuple[str, int], ...]:
        """Decode the complete observed-state version vector from a Kernel action."""

        if not isinstance(raw, list):
            raise ValueError("runtime failure resource versions are invalid")
        versions: list[tuple[str, int]] = []
        for item in raw:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
            ):
                raise ValueError("runtime failure resource versions are invalid")
            versions.append((item[0], item[1]))
        if tuple(versions) != tuple(sorted(versions)):
            raise ValueError("runtime failure resource versions are not canonical")
        return tuple(versions)

    def _compile_failure_diagnosis_context(
        self, *, candidate_task_ids: tuple[str, ...], failure_action_id: str
    ) -> dict[str, object]:
        """Project the versioned residual graph without inventing a root cause."""

        if self.runtime_kernel is None:
            raise RuntimeError("failure diagnosis requires the runtime kernel")
        action = self.runtime_kernel.actions.get(failure_action_id)
        if action is None:
            raise RuntimeError("failure diagnosis action is missing")
        task_rows: list[dict[str, object]] = []
        referenced_resources: set[str] = set()
        for task_id in candidate_task_ids:
            node = self.runtime_kernel.task_nodes.get(task_id)
            if node is None:
                raise RuntimeError("failure diagnosis task is missing")
            referenced_resources.update(node.prerequisite_resource_ids)
            referenced_resources.update(node.hypothesized_prerequisite_resource_ids)
            referenced_resources.update(node.required_output_resource_ids)
            task_rows.append(
                {
                    "task_id": task_id,
                    "parent_id": node.parent_id,
                    "status": self.runtime_kernel.task_statuses[task_id].value,
                    "source": node.source.value,
                    "confirmed_prerequisite_resource_ids": list(node.prerequisite_resource_ids),
                    "hypothesized_prerequisite_resource_ids": list(
                        node.hypothesized_prerequisite_resource_ids
                    ),
                    "required_output_resource_ids": list(node.required_output_resource_ids),
                }
            )
        resource_rows = []
        for resource_id in sorted(referenced_resources):
            observation = self.runtime_kernel.observations[resource_id]
            resource_rows.append(
                {
                    "resource_id": resource_id,
                    "version": self.runtime_kernel.resource_versions[resource_id],
                    "observed": observation.observed,
                    "exists": observation.exists,
                    "kind": observation.kind.value,
                }
            )
        evidence_rows = [
            {
                "evidence_id": evidence.evidence_id,
                "task_id": evidence.task_id,
                "state": evidence.state.value,
                "dependency_claims": [
                    [resource_id, role.value] for resource_id, role in evidence.dependency_claims
                ],
            }
            for evidence in self.runtime_kernel.evidence.values()
            if evidence.task_id in candidate_task_ids
        ]
        return {
            "failed_transition": {
                "action_id": action.action_id,
                "tool_name": action.tool_name,
                "phase": action.phase.value,
                "reason": action.reason,
            },
            "candidate_tasks": task_rows,
            "resource_facts": resource_rows,
            "evidence": evidence_rows,
        }

    def _replay_actor_output_progress(self) -> int:
        """Reconstruct the no-progress counter from durable observations on resume."""

        if self.runtime_kernel is None or not self._output_resource_ids:
            return 0
        rows = [
            json.loads(line) for line in self.runtime_kernel.event_log_path.read_text().splitlines()
        ]
        observations: dict[str, object] = {}
        actor_by_action: dict[str, bool] = {}
        counter = 0

        def apply(raw: object) -> bool:
            if not isinstance(raw, list):
                return False
            changed = False
            for item in raw:
                if not isinstance(item, dict):
                    continue
                resource_id = item.get("resource_id")
                if resource_id in self._output_resource_ids:
                    if observations.get(str(resource_id)) != item:
                        changed = True
                    observations[str(resource_id)] = item
            return changed

        for row in rows:
            payload = row.get("payload") if isinstance(row, dict) else None
            if not isinstance(payload, dict):
                continue
            event_type = row.get("event_type")
            if event_type == "initialized":
                apply(payload.get("observations"))
            elif event_type == "action_proposed":
                actor_by_action[str(payload.get("action_id", ""))] = payload.get("actor") == "actor"
            elif event_type == "observations_refreshed":
                apply(payload.get("observations"))
            elif event_type == "action_finished":
                changed = apply(payload.get("observations"))
                action_id = str(payload.get("action_id", ""))
                if actor_by_action.get(action_id):
                    counter = 0 if changed else counter + 1
                elif (
                    self.runtime_kernel.actions.get(action_id) is not None
                    and self.runtime_kernel.actions[action_id].tool_name == "runtime_guidance"
                ):
                    counter = 0
        return counter

    def begin_actor_attempt(self, *, deadline_monotonic: float) -> None:
        """Bind tool deadlines to the enclosing Pi attempt before it can call tools."""

        if deadline_monotonic <= time.monotonic():
            raise ValueError("Actor attempt deadline must be in the future")
        self._actor_attempt_deadline = deadline_monotonic

    def _live_actor_time_remaining_ms(self) -> int:
        """Read the active attempt clock at the exact decision boundary."""

        if self._actor_attempt_deadline is None:
            raise RuntimeError("Actor time state requires an active attempt")
        return max(0, round((self._actor_attempt_deadline - time.monotonic()) * 1000))

    def end_actor_attempt(self) -> None:
        """Clear the attempt deadline after every RPC handler has closed."""

        self._actor_attempt_deadline = None

    async def begin_actor_state_submission(
        self,
        *,
        attempt_index: int,
        boundary_index: int = 0,
        actor_time_budget_ms: int,
    ) -> ActorStateSubmission | None:
        """Select a submission using the live clock after pending tools release the lock."""

        if (
            self.joint_runtime_policy is None
            or self.joint_runtime_policy.method_id not in TIME_RESERVE_JOINT_METHODS
            or self.runtime_kernel is None
            or self.resource_driver is None
        ):
            return None
        async with self._lock:
            await self._refresh_runtime(source="actor_bridge_time_reserve")
            failure_source_submission = (
                self.joint_runtime_policy.method_id
                in FAILURE_DIAGNOSIS_ONLY_JOINT_METHODS
            )
            capabilities = (
                {} if failure_source_submission else await self._policy_capabilities()
            )
            # Sample after every awaited state read so the slate uses the true
            # decision-boundary clock rather than the start of reconciliation.
            actor_time_remaining_ms = self._live_actor_time_remaining_ms()
            trigger_action_id = (
                f"actor-bridge-time-reserve-{attempt_index:03d}-{boundary_index:03d}"
            )
            self.runtime_kernel.propose_action(
                action_id=trigger_action_id,
                tool_name="actor_time_reserve_trigger",
                arguments={
                    "actor_time_budget_ms": actor_time_budget_ms,
                    "actor_time_remaining_ms": actor_time_remaining_ms,
                    "convergence_reserve_ms": ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000,
                },
                actor="actor_bridge",
            )
            self.runtime_kernel.start_action(trigger_action_id)
            self.runtime_kernel.finish_action(
                trigger_action_id,
                phase=ActionPhase.COMPLETED,
                observations=(),
            )
            slate = compile_runtime_joint_action_slate(
                kernel=self.runtime_kernel,
                capabilities=capabilities,
                trigger_action_id=trigger_action_id,
                budget_before=self.joint_runtime_policy.remaining_budget_units,
                no_progress_actor_actions=self._actor_actions_since_output_progress,
                progress_trigger_threshold=self.progress_trigger_threshold,
                excluded_resource_versions=frozenset(self._failed_runtime_intervention_versions),
                excluded_progress_resource_versions=frozenset(),
                excluded_semantic_obligation_versions=frozenset(
                    self._semantic_checked_obligation_versions
                ),
                output_semantics_enabled=(
                    False
                    if failure_source_submission
                    else self._actor_actions_since_output_progress
                    >= OUTPUT_SEMANTIC_STABILITY_ACTION_THRESHOLD
                ),
                output_progress_enabled=not failure_source_submission,
                actor_time_budget_ms=actor_time_budget_ms,
                actor_time_remaining_ms=actor_time_remaining_ms,
                actor_convergence_reserve_ms=(ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000),
                actor_submission_enabled=True,
            )
            if not any(
                item.action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE
                for item in slate.candidates
            ):
                return None
            slate = self._gated_slate(slate)
            action_domain_projection = None
            if (
                self.joint_runtime_policy.method_id
                in RUNTIME_OPTION_CHAIN_METHODS
            ):
                # The reserve slate is a real decision boundary, so an option-chain
                # selector must consume the same declared sub-domain it consumes at
                # every other boundary.  Without the projection `select()` rejects
                # the slate, which aborted a real run the first time the reserve
                # timer fired.
                slate, action_domain_projection = (
                    project_runtime_option_action_domain(slate)
                )
            selection = self.joint_runtime_policy.select(
                slate,
                action_domain_projection=action_domain_projection,
            )
            candidate = selection.selected
            if candidate.action_type is RuntimeJointActionType.CONTINUE_ACTOR:
                return None
            if candidate.action_type is not RuntimeJointActionType.SUBMIT_CURRENT_STATE:
                raise RuntimeError("v4 Actor Bridge selected a non-submission action")
            action_id = f"runtime-joint-submit-at-{attempt_index:03d}"
            self.joint_runtime_policy.record_execution_started(action_id=action_id)
            # The latch rejects any tool proposal produced concurrently with abort.
            self._actor_convergence_requested = True
            return ActorStateSubmission(
                action_id=action_id,
                decision_id=selection.decision_id,
                candidate=candidate,
                actor_time_budget_ms=actor_time_budget_ms,
                actor_time_remaining_ms=actor_time_remaining_ms,
                convergence_reserve_ms=ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000,
            )

    async def begin_actor_settlement_review(
        self, *, attempt_index: int, review_index: int
    ) -> ActorSettlementReview | None:
        """Select one frozen semantic-review arm at Pi's natural idle boundary."""

        if (
            self.joint_runtime_policy is None
            or self.joint_runtime_policy.method_id not in SETTLEMENT_REVIEW_JOINT_METHODS
            or self.runtime_kernel is None
            or self.resource_driver is None
        ):
            return None
        async with self._lock:
            await self._refresh_runtime(source="actor_bridge_natural_settlement")
            actor_time_budget_ms = None
            actor_time_remaining_ms = None
            if self.selector_time_observation_enabled:
                if self._actor_attempt_deadline is None or self.actor_time_budget_seconds is None:
                    raise RuntimeError("time-aware settlement lacks an active time boundary")
                actor_time_budget_ms = self.actor_time_budget_seconds * 1000
                actor_time_remaining_ms = max(
                    0, round((self._actor_attempt_deadline - time.monotonic()) * 1000)
                )
            trigger_action_id = (
                f"actor-bridge-natural-settlement-{attempt_index:03d}-{review_index:03d}"
            )
            self.runtime_kernel.propose_action(
                action_id=trigger_action_id,
                tool_name="actor_natural_settlement_trigger",
                arguments={"review_index": review_index},
                actor="actor_bridge",
            )
            self.runtime_kernel.start_action(trigger_action_id)
            self.runtime_kernel.finish_action(
                trigger_action_id,
                phase=ActionPhase.COMPLETED,
                observations=(),
            )
            slate = compile_runtime_joint_action_slate(
                kernel=self.runtime_kernel,
                # Settlement source arms compare only accept versus one review.
                capabilities={},
                trigger_action_id=trigger_action_id,
                budget_before=self.joint_runtime_policy.remaining_budget_units,
                no_progress_actor_actions=self._actor_actions_since_output_progress,
                progress_trigger_threshold=self.progress_trigger_threshold,
                excluded_resource_versions=frozenset(self._failed_runtime_intervention_versions),
                excluded_progress_resource_versions=frozenset(self._guided_output_versions),
                excluded_semantic_obligation_versions=frozenset(
                    self._semantic_checked_obligation_versions
                ),
                output_semantics_enabled=True,
                output_progress_enabled=(
                    self.joint_runtime_policy.method_id
                    in {
                        TRACEDP_RESIDUAL_RUNTIME_METHOD,
                        TRACEDP_RESIDUAL_RUNTIME_V2_METHOD,
                        TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
                    }
                ),
                settlement_output_completion_enabled=(
                    self.joint_runtime_policy.method_id
                    in {
                        TRACEDP_RESIDUAL_RUNTIME_METHOD,
                        TRACEDP_RESIDUAL_RUNTIME_V2_METHOD,
                        TRACEDP_RESIDUAL_RUNTIME_V3_METHOD,
                    }
                ),
                actor_time_budget_ms=actor_time_budget_ms,
                actor_time_remaining_ms=actor_time_remaining_ms,
            )
            slate = self._gated_slate(slate)
            selection = self.joint_runtime_policy.select(slate)
            candidate = selection.selected
            if candidate.action_type is RuntimeJointActionType.CONTINUE_ACTOR:
                return None
            if candidate.action_type not in {
                RuntimeJointActionType.REQUEST_SEMANTIC_CHECK,
                RuntimeJointActionType.REQUEST_OUTPUT_COMPLETION,
            }:
                raise RuntimeError("settlement review policy selected a non-semantic action")
            obligation = self._joint_output_obligation(candidate)
            action_id = f"runtime-joint-settlement-review-{attempt_index:03d}"
            self.joint_runtime_policy.record_execution_started(action_id=action_id)
            semantic = candidate.action_type is RuntimeJointActionType.REQUEST_SEMANTIC_CHECK
            intervention = {
                "status": "completed",
                "resource_id": candidate.target_binding.resource_id,
                "policy": self.joint_runtime_policy.method_id,
                "intervention_kind": (
                    "runtime_semantic_guidance" if semantic else "runtime_guidance"
                ),
                "output_resource_ids": list(obligation.output_resource_ids),
                "output_requirement": obligation.output_requirement.value,
            }
            if not semantic:
                intervention["no_progress_actor_actions"] = (
                    self._actor_actions_since_output_progress
                )
            notice = self._runtime_notice([intervention]).removeprefix("\n\n")
            return ActorSettlementReview(
                action_id=action_id,
                decision_id=selection.decision_id,
                candidate=candidate,
                notice=notice,
            )

    async def decide_at_natural_settlement(
        self, *, attempt_index: int, round_index: int
    ) -> ActorNaturalSettlementPrompt | None:
        """Give option-chain methods a decision where the Actor would otherwise end.

        Option-chain methods decide only after an Actor tool call. An Actor that
        declares completion without (more) tool use therefore ends the run with the
        budget unspent, which is exactly the stopping point where a registered check
        is most informative. This runs one ordinary decision (with the usual
        same-boundary follow-ups) at that boundary. The Actor is prompted again only
        when the outcome needs its action: a check that was not supported, a failed
        runtime step, or a guidance candidate. A supported check leaves it settled.
        """

        if (
            self.joint_runtime_policy is None
            or self.joint_runtime_policy.method_id not in RUNTIME_OPTION_CHAIN_METHODS
            or self.runtime_kernel is None
            or self.resource_driver is None
            or self._runtime_supervision_disabled_reason is not None
            or self.joint_runtime_policy.remaining_budget_units <= 0
        ):
            return None
        async with self._lock:
            await self._refresh_runtime(source="actor_bridge_natural_settlement")
            sequence = self._sequence
            self._sequence += 1
            trigger_action_id = (
                f"actor-bridge-natural-settlement-{attempt_index:03d}-{round_index:03d}"
            )
            self.runtime_kernel.propose_action(
                action_id=trigger_action_id,
                tool_name="actor_natural_settlement_trigger",
                arguments={"review_index": round_index},
                actor="actor_bridge",
            )
            self.runtime_kernel.start_action(trigger_action_id)
            self.runtime_kernel.finish_action(
                trigger_action_id,
                phase=ActionPhase.COMPLETED,
                observations=(),
            )
            try:
                interventions = await self._apply_runtime_supervision(
                    sequence=sequence,
                    trigger_action_id=trigger_action_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                self._runtime_supervision_disabled_reason = reason
                self.runtime_kernel.disable_runtime_supervision(reason=reason)
                return None
        if not interventions or not any(
            self._intervention_needs_actor(item) for item in interventions
        ):
            return None
        notice = self._runtime_notice(interventions).removeprefix("\n\n")
        if not notice:
            return None
        return ActorNaturalSettlementPrompt(
            trigger_action_id=trigger_action_id,
            notice=notice,
            interventions=tuple(interventions),
        )

    @staticmethod
    def _intervention_needs_actor(intervention: Mapping[str, object]) -> bool:
        """True unless the intervention only confirmed a condition the Actor can keep."""

        kind = intervention.get("intervention_kind")
        if kind in {"runtime_registered_check_execution", "runtime_targeted_repair_execution"}:
            if (
                kind == "runtime_targeted_repair_execution"
                and intervention.get("candidate_form") == "actor_guidance"
            ):
                return intervention.get("effect_status") == "awaiting_recheck"
            return not (
                intervention.get("status") == "completed"
                and intervention.get("evidence_state") == EvidenceState.SUPPORTED.value
            )
        return True

    async def finish_actor_settlement_review(
        self, review: ActorSettlementReview, *, accepted: bool, reason: str | None = None
    ) -> None:
        """Record delivery only after Pi confirms that the follow-up prompt was accepted."""

        if self.joint_runtime_policy is None:
            raise RuntimeError("Actor settlement review runtime is unavailable")
        async with self._lock:
            if not accepted:
                self.joint_runtime_policy.abandon_before_environment_effect(
                    reason=reason or "Pi rejected the settlement review prompt"
                )
                return
            self._execute_joint_guidance(
                sequence=self._sequence,
                selection_id=review.decision_id,
                candidate=review.candidate,
                action_id=review.action_id,
                execution_already_started=True,
            )

    async def finish_actor_state_submission(
        self,
        submission: ActorStateSubmission,
        *,
        completed: bool,
        reason: str | None = None,
    ) -> None:
        """Close the selected submission only after the Pi control receipt is known."""

        if self.joint_runtime_policy is None or self.runtime_kernel is None:
            raise RuntimeError("Actor state submission runtime is unavailable")
        candidate = submission.candidate
        binding = candidate.target_binding
        if (
            binding.resource_id is None
            or binding.resource_version is None
            or binding.obligation_version_key is None
        ):
            raise RuntimeError("Actor state submission lacks an output binding")
        async with self._lock:
            self.runtime_kernel.propose_action(
                action_id=submission.action_id,
                tool_name="submit_current_state",
                arguments={
                    "resource_id": binding.resource_id,
                    "resource_version": binding.resource_version,
                    "task_id": binding.task_id,
                    "obligation_id": binding.obligation_id,
                    "obligation_version_key": self._encode_obligation_key(
                        binding.obligation_version_key
                    ),
                    "actor_time_budget_ms": submission.actor_time_budget_ms,
                    "actor_time_remaining_ms": submission.actor_time_remaining_ms,
                    "convergence_reserve_ms": submission.convergence_reserve_ms,
                    "joint_decision_id": submission.decision_id,
                },
                actor="runtime",
            )
            self.runtime_kernel.start_action(submission.action_id)
            phase = (
                ActionPhase.COMPLETED if completed else ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
            )
            self.runtime_kernel.finish_action(
                submission.action_id,
                phase=phase,
                observations=(),
                reason=reason,
            )
            self.joint_runtime_policy.record_outcome(
                status="completed" if completed else "cancelled_with_unknown_effect",
                reason=reason,
            )

    @property
    def fatal_containment_error(self) -> RuntimeError | None:
        """Expose a cleanup failure that makes same-environment resume unsafe."""

        return self._fatal_containment_error

    def _parse_actor_check(self, arguments: Mapping[str, Any]) -> RuntimeCheckContract:
        """Ground a proposed check in public text without certifying its correctness."""
        if not self.allow_check_registration or self.runtime_kernel is None:
            raise ValueError("Actor check registration is disabled")
        task_id = arguments["task_id"]
        excerpt = arguments["source_excerpt"]
        if task_id not in self.runtime_kernel.task_nodes:
            raise ValueError("check task is not in the current public graph")
        public_roots = [
            node
            for node in self.runtime_kernel.task_nodes.values()
            if node.public_requirement and node.parent_id is None
        ]
        if (
            not isinstance(excerpt, str)
            or not excerpt.strip()
            or len(public_roots) != 1
            or excerpt not in public_roots[0].title
        ):
            raise ValueError("check source excerpt is absent from the public task")
        proposition = arguments["proposition"]
        command = arguments["command"]
        verdict_protocol = arguments.get(
            "verdict_protocol", EXACT_OUTPUT_VERDICT_PROTOCOL
        )
        if any(
            not isinstance(value, str) or not value
            for value in (proposition, command, verdict_protocol)
        ):
            raise ValueError("check proposition, command and protocol must be nonempty text")
        if verdict_protocol not in {
            EXACT_OUTPUT_VERDICT_PROTOCOL,
            EXIT_STATUS_VERDICT_PROTOCOL,
        }:
            raise ValueError("runtime check verdict protocol is unsupported")
        if verdict_protocol == EXIT_STATUS_VERDICT_PROTOCOL:
            supported, refuted = "return_code_0", "return_code_1"
        else:
            supported = arguments["supported_output"]
            refuted = arguments["refuted_output"]
            if any(
                not isinstance(value, str) or not value
                for value in (supported, refuted)
            ):
                raise ValueError("check verdict outputs must be nonempty text")
        resources = tuple(arguments.get("resource_ids", ()))
        if not resources:
            raise ValueError("check must bind at least one observed resource")
        if any(item not in self.runtime_kernel.resource_versions for item in resources):
            raise ValueError("check resource scope is not observed")
        timeout = arguments.get("timeout", 120)
        if type(timeout) is not int or not 1 <= timeout <= 600:
            raise ValueError("check timeout must be between 1 and 600 seconds")
        return RuntimeCheckContract(
            task_id,
            proposition,
            {"command": command, "timeout": timeout},
            supported,
            refuted,
            resources,
            verdict_protocol,
        )

    def _preflight_error(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        tool_call_id: str = "preflight-registration",
    ) -> DelegatedToolResult | None:
        """Validate syntax before marking an action as started in the environment."""

        if self._actor_convergence_requested:
            return DelegatedToolResult(
                "Runtime supervisor blocked this tool because the declared output "
                "already exists and the Actor session is inside its finalization "
                "reserve. Stop calling tools and return the final response now.",
                True,
                False,
                {"runtime_convergence_gate": True},
            )

        if tool_name == "register_runtime_check":
            try:
                contract = self._parse_actor_check(arguments)
                if any(
                    item.task_id == contract.task_id
                    and item.proposition == contract.proposition
                    and dict(item.arguments) == dict(contract.arguments)
                    and item.verdict_protocol == contract.verdict_protocol
                       for item in self.check_contracts):
                    raise ValueError("identical check program is already registered")
                if self.runtime_kernel is not None and (
                    self.runtime_kernel.acceptance_predicate_enabled
                ):
                    self.runtime_kernel.preview_actor_check_predicate(
                        registration_action_id=tool_call_id,
                        task_id=contract.task_id,
                        proposition=contract.proposition,
                        source_excerpt=str(arguments["source_excerpt"]),
                        resource_ids=contract.resource_ids,
                    )
            except (KeyError, TypeError, ValueError) as error:
                return DelegatedToolResult(str(error), True, False, {})
            return None
        if tool_name == "complete_runtime_check_registration":
            if not self.require_check_registration_phase or self.runtime_kernel is None:
                return DelegatedToolResult(
                    "required check registration phase is disabled", True, False, {}
                )
            if self._check_registration_phase_completion is not None:
                return DelegatedToolResult(
                    "check registration phase is already complete", True, False, {}
                )
            task_id = arguments.get("task_id")
            status = arguments.get("status")
            reason = arguments.get("reason")
            node = self.runtime_kernel.task_nodes.get(str(task_id))
            if node is None or not node.public_requirement:
                return DelegatedToolResult(
                    "phase completion task is not a public task node", True, False, {}
                )
            if status not in {"checks_registered", "no_public_check_available"}:
                return DelegatedToolResult(
                    "check registration phase status is invalid", True, False, {}
                )
            if not isinstance(reason, str) or not reason.strip():
                return DelegatedToolResult(
                    "check registration phase reason is required", True, False, {}
                )
            if status == "checks_registered" and not self.check_contracts:
                return DelegatedToolResult(
                    "checks_registered requires at least one registered check",
                    True,
                    False,
                    {},
                )
            if status == "no_public_check_available" and self.check_contracts:
                return DelegatedToolResult(
                    "no_public_check_available contradicts a registered check",
                    True,
                    False,
                    {},
                )
            return None
        if (
            self.require_check_registration_phase
            and self._check_registration_phase_completion is None
            and tool_name in {"bash", "write"}
        ):
            return DelegatedToolResult(
                "Complete the required public-task check-registration phase before "
                f"calling {tool_name}.",
                True,
                False,
                {"check_registration_phase": "open"},
            )
        if tool_name == "bash":
            command = arguments.get("command")
            timeout = arguments.get("timeout", 120)
            if not isinstance(command, str) or not command:
                return DelegatedToolResult("bash command is required", True, False, {})
            if not isinstance(timeout, int) or timeout <= 0:
                return DelegatedToolResult(
                    "bash timeout must be a positive integer", True, False, {}
                )
            if (
                self._actor_attempt_deadline is not None
                and self._actor_attempt_deadline - time.monotonic()
                <= (_BASH_OUTER_TIMEOUT_MARGIN_SECONDS + _BASH_SESSION_COMPLETION_RESERVE_SECONDS)
            ):
                return DelegatedToolResult(
                    "Bash was not started because the Actor session is closing.",
                    True,
                    False,
                    {},
                )
            return None
        if tool_name == "read":
            raw_path = arguments.get("path")
            offset = arguments.get("offset", 1)
            limit = arguments.get("limit", 2000)
            if not isinstance(raw_path, str):
                return DelegatedToolResult("read path is required", True, False, {})
            if not isinstance(offset, int) or offset < 1:
                return DelegatedToolResult("read offset must be at least 1", True, False, {})
            if not isinstance(limit, int) or limit < 1:
                return DelegatedToolResult("read limit must be positive", True, False, {})
            try:
                validate_actor_visible_path(raw_path)
            except ValueError as error:
                return DelegatedToolResult(str(error), True, False, {})
            return None
        if tool_name == "write":
            raw_path = arguments.get("path")
            content = arguments.get("content")
            if not isinstance(raw_path, str) or not isinstance(content, str):
                return DelegatedToolResult("write path and content are required", True, False, {})
            try:
                validate_actor_visible_path(raw_path)
            except ValueError as error:
                return DelegatedToolResult(str(error), True, False, {})
            return None
        return DelegatedToolResult(f"unsupported terminal tool: {tool_name}", True, False, {})

    async def _observe_file(
        self, path: str, sequence: int, path_index: int
    ) -> PublicFileObservation:
        stat_result = await self.environment.exec(
            f"stat -c '%F|%a' -- {shlex.quote(path)}",
            timeout_sec=_PUBLIC_STATE_OBSERVATION_TIMEOUT_SECONDS,
        )
        if stat_result.return_code != 0:
            return PublicFileObservation(
                path=path,
                exists=False,
                file_type=None,
                mode=None,
                content=None,
            )
        stat_parts = str(stat_result.stdout or "").strip().split("|", maxsplit=1)
        if len(stat_parts) != 2:
            raise RuntimeError(f"cannot observe public path type and mode: {path}")
        file_type, mode = stat_parts
        if file_type == "regular empty file":
            # GNU stat names zero-length files separately (e.g. a WAL truncated by sqlite3);
            # the empty content below still records the state exactly.
            file_type = "regular file"
        if file_type == "directory":
            return PublicFileObservation(
                path=path,
                exists=True,
                file_type=file_type,
                mode=mode,
                content=None,
            )
        if file_type != "regular file":
            raise RuntimeError(f"unsupported public path type {file_type}: {path}")
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        local_path = self.scratch_root / f"observe-{sequence}-{path_index}.bin"
        await self.environment.download_file(path, local_path)
        try:
            content = local_path.read_bytes()
        finally:
            local_path.unlink(missing_ok=True)
        return PublicFileObservation(
            path=path,
            exists=True,
            file_type=file_type,
            mode=mode,
            content=content,
        )

    async def _snapshot(self, sequence: int) -> tuple[PublicFileObservation, ...]:
        """Read current public files, batched when enabled and verified, else per path."""

        if not self.batched_public_snapshot or not self.public_paths:
            return await self._snapshot_per_path(sequence)
        batched = await self._snapshot_batched(sequence)
        if batched is None:
            return await self._snapshot_per_path(sequence)
        if not self._batched_snapshot_verified:
            reference = await self._snapshot_per_path(sequence)
            same = reference == batched
            self.batched_snapshot_events.append({"sequence": sequence, "verified_equal": same})
            if not same:
                self.batched_public_snapshot = False
                return reference
            self._batched_snapshot_verified = True
        return batched

    async def _snapshot_batched(
        self, sequence: int
    ) -> tuple[PublicFileObservation, ...] | None:
        """One exec for every public path; ``None`` asks for the per-path read instead."""

        script = "; ".join(
            f"printf '@@LH {index} '; "
            f"if s=$(stat -c '%F|%a|%s' -- {shlex.quote(path)} 2>/dev/null); then printf '%s\\n' \"$s\"; "
            f"case \"$s\" in 'regular file|'*|'regular empty file|'*) "
            f"if [ \"${{s##*|}}\" -le {_BATCHED_SNAPSHOT_MAX_BYTES} ]; then "
            f"base64 {shlex.quote(path)} 2>/dev/null || printf '@@B64FAIL\\n'; "
            f"else printf '@@BIG\\n'; fi;; esac; "
            f"else printf '@@MISSING\\n'; fi; printf '@@END {index}\\n'"
            for index, path in enumerate(self.public_paths)
        )
        try:
            result = await self.environment.exec(
                script, timeout_sec=_PUBLIC_STATE_OBSERVATION_TIMEOUT_SECONDS
            )
        except Exception:
            return None
        entries = _parse_batched_snapshot(str(result.stdout or ""), len(self.public_paths))
        if result.return_code != 0 or entries is None:
            return None
        observations = []
        for index, path in enumerate(self.public_paths):
            header, payload = entries[index]
            if header == "@@MISSING":
                observations.append(
                    PublicFileObservation(path=path, exists=False, file_type=None, mode=None, content=None)
                )
                continue
            file_type, mode, _size = header.rsplit("|", 2)
            if file_type == "regular empty file":
                file_type = "regular file"
            if file_type == "directory":
                observations.append(
                    PublicFileObservation(path=path, exists=True, file_type=file_type, mode=mode, content=None)
                )
                continue
            if file_type != "regular file":
                raise RuntimeError(f"unsupported public path type {file_type}: {path}")
            if payload is None:
                observations.append(await self._observe_file(path, sequence, index))
                continue
            observations.append(
                PublicFileObservation(path=path, exists=True, file_type=file_type, mode=mode, content=payload)
            )
        return tuple(observations)

    async def _snapshot_per_path(self, sequence: int) -> tuple[PublicFileObservation, ...]:
        """Read current public files with bounded concurrency, without atomicity guarantees."""

        limiter = asyncio.Semaphore(_PUBLIC_STATE_OBSERVER_CONCURRENCY)

        async def observe(path: str, path_index: int) -> PublicFileObservation:
            async with limiter:
                return await self._observe_file(path, sequence, path_index)

        # gather preserves the frozen public-path order required by replay parity.
        return tuple(
            await asyncio.gather(
                *(observe(path, path_index) for path_index, path in enumerate(self.public_paths))
            )
        )

    async def restore_public_state(
        self,
        target_state: tuple[PublicFileObservation, ...],
    ) -> tuple[
        tuple[PublicFileObservation, ...],
        tuple[PublicFileObservation, ...],
    ]:
        """Restore an exact, bounded subset of the observed public filesystem.

        This is a verifier-side intervention, not an actor tool call. Directory
        trees are deliberately unsupported because the observer records only a
        directory's type and mode, not its recursive contents.
        """

        if not target_state:
            raise ValueError("public-state intervention must restore at least one path")
        target_paths = tuple(observation.path for observation in target_state)
        if len(target_paths) != len(set(target_paths)):
            raise ValueError("public-state intervention paths must be unique")
        if not set(target_paths) <= set(self.public_paths) or "/app" in target_paths:
            raise ValueError("public-state intervention escaped the observed path set")
        if any(not is_exact_restorable_public_file(item) for item in target_state):
            raise ValueError("public-state intervention supports exact regular files only")

        # Actor serialization does not stop background writers between operations.
        before = await self._snapshot(self._sequence)
        before_by_path = {observation.path: observation for observation in before}
        if any(
            before_by_path[path].exists and before_by_path[path].file_type == "directory"
            for path in target_paths
        ):
            raise ValueError("public-state intervention cannot replace a directory")

        self.scratch_root.mkdir(parents=True, exist_ok=True)
        for index, observation in enumerate(target_state):
            quoted_path = shlex.quote(observation.path)
            if not observation.exists:
                result = await self.environment.exec(
                    f"rm -f -- {quoted_path}",
                    timeout_sec=_PUBLIC_STATE_OBSERVATION_TIMEOUT_SECONDS,
                )
            else:
                local_path = self.scratch_root / f"restore-{self._sequence}-{index}.bin"
                local_path.write_bytes(observation.content or b"")
                try:
                    await self.environment.upload_file(local_path, observation.path)
                finally:
                    local_path.unlink(missing_ok=True)
                result = await self.environment.exec(
                    f"chmod {observation.mode} -- {quoted_path}",
                    timeout_sec=_PUBLIC_STATE_OBSERVATION_TIMEOUT_SECONDS,
                )
            if result.return_code != 0:
                raise RuntimeError(f"cannot restore public path: {observation.path}")

        after = await self._snapshot(self._sequence)
        after_by_path = {observation.path: observation for observation in after}
        if any(
            after_by_path[path] != target
            for path, target in zip(target_paths, target_state, strict=True)
        ):
            raise RuntimeError("public-state intervention failed exact readback")
        return before, after

    async def snapshot_public_state(self) -> tuple[PublicFileObservation, ...]:
        """Capture the current allowlisted state without creating a tool transition."""

        return await self._snapshot(self._sequence)

    def _append_observer(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        before: tuple[PublicFileObservation, ...],
        after: tuple[PublicFileObservation, ...],
        is_error: bool,
        environment_step_executed: bool | None,
        overlapping_tool_calls: bool = False,
        error_stage: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        latency_ms: Mapping[str, int] | None = None,
        tool_result: DelegatedToolResult | None = None,
        pre_supervision_content: str | None = None,
    ) -> None:
        """Persist the exact gateway receipt joined to one Kernel Actor action."""

        # Size of what this exchange appends to the Actor's next request.
        self._last_visible_chars = len(json.dumps(dict(arguments), default=str)) + (
            len(tool_result.content) if tool_result is not None else 0
        )
        # A content offset separates tool text from our appended notice, even
        # when the tool itself prints a supervisor-looking marker.
        if pre_supervision_content is not None and (
            tool_result is None or not tool_result.content.startswith(pre_supervision_content)
        ):
            raise ValueError("pre-supervision content is not the delivered prefix")
        self.observer_path.parent.mkdir(parents=True, exist_ok=True)
        with self.observer_path.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "schema_version": ACTOR_TOOL_RECEIPT_SCHEMA,
                        "sequence": self._observer_sequence,
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "input": dict(arguments),
                        "overlapping_tool_calls": overlapping_tool_calls,
                        "environment_step_executed": environment_step_executed,
                        "observation_complete": self.observe_public_state and error_stage is None,
                        "before": [_observation_payload(observation) for observation in before],
                        "after": [_observation_payload(observation) for observation in after],
                        "is_error": is_error,
                        "latency_ms": dict(latency_ms or {}),
                        "tool_result": (
                            {
                                "content": tool_result.content,
                                "is_error": tool_result.is_error,
                                "environment_step_executed": (
                                    tool_result.environment_step_executed
                                ),
                                "details": tool_result.details,
                            }
                            if tool_result is not None
                            else None
                        ),
                        **(
                            {"pre_supervision_content_chars": len(pre_supervision_content)}
                            if pre_supervision_content is not None else {}
                        ),
                        **({"error_stage": error_stage} if error_stage else {}),
                        **({"error_type": error_type} if error_type else {}),
                        **({"error_message": error_message} if error_message is not None else {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._observer_sequence += 1

    def _append_runtime_check_execution(
        self,
        *,
        action_id: str,
        trigger_action_id: str,
        check_index: int,
        declared_arguments: Mapping[str, Any],
        executed_arguments: Mapping[str, Any],
        result: DelegatedToolResult | None,
        resource_versions_before: tuple[tuple[str, int], ...],
        resource_versions_after: tuple[tuple[str, int], ...],
        evidence_id: str | None,
        evidence_state: EvidenceState | None,
        observation_complete: bool,
        execution_mode: str,
        blocking_fact: Mapping[str, object] | None,
        execution_error: str | None = None,
    ) -> None:
        """Persist the environment result independently from policy and evidence logs."""

        self.runtime_check_execution_path.parent.mkdir(parents=True, exist_ok=True)
        with self.runtime_check_execution_path.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "schema_version": "runtime_check_execution_receipt_v2",
                        "sequence": self._runtime_check_execution_sequence,
                        "runtime_action_id": action_id,
                        "trigger_action_id": trigger_action_id,
                        "check_index": check_index,
                        "declared_arguments": dict(declared_arguments),
                        # Preserve the exact Bash input separately from the public
                        # check declaration so audits can verify the wrapper ran.
                        "executed_arguments": dict(executed_arguments),
                        "execution_mode": execution_mode,
                        "return_code": (
                            result.details.get("return_code") if result is not None else None
                        ),
                        "content": result.content if result is not None else "",
                        "environment_step_executed": (
                            result.environment_step_executed if result is not None else None
                        ),
                        "execution_error": execution_error,
                        "observation_complete": observation_complete,
                        "resource_versions_before": [
                            list(item) for item in resource_versions_before
                        ],
                        "resource_versions_after": [
                            list(item) for item in resource_versions_after
                        ],
                        "evidence_id": evidence_id,
                        "evidence_state": (
                            evidence_state.value if evidence_state is not None else None
                        ),
                        "blocking_fact": (
                            dict(blocking_fact) if blocking_fact is not None else None
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._runtime_check_execution_sequence += 1

    def _append_runtime_repair_execution(
        self,
        *,
        action_id: str,
        trigger_action_id: str,
        repair_index: int,
        check_index: int,
        incident_id: str,
        repair_arguments: Mapping[str, Any],
        check_arguments: Mapping[str, Any],
        repair_result: DelegatedToolResult | None,
        check_result: DelegatedToolResult | None,
        phase: ActionPhase,
        repair_elapsed_ms: int,
        check_elapsed_ms: int,
        total_elapsed_ms: int,
        observation_complete: bool,
        evidence_id: str | None,
        evidence_state: EvidenceState | None,
        execution_error: str | None,
    ) -> None:
        """Persist repair and mandatory recheck outputs independently from policy logs."""

        self.runtime_repair_execution_path.parent.mkdir(parents=True, exist_ok=True)
        with self.runtime_repair_execution_path.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "schema_version": "runtime_targeted_repair_execution_receipt_v1",
                        "sequence": self._runtime_repair_execution_sequence,
                        "runtime_action_id": action_id,
                        "trigger_action_id": trigger_action_id,
                        "repair_index": repair_index,
                        "check_index": check_index,
                        "incident_id": incident_id,
                        "repair_arguments": dict(repair_arguments),
                        "check_arguments": dict(check_arguments),
                        "repair_result": (
                            {
                                "content": repair_result.content,
                                "is_error": repair_result.is_error,
                                "environment_step_executed": (
                                    repair_result.environment_step_executed
                                ),
                                "details": repair_result.details,
                            }
                            if repair_result is not None
                            else None
                        ),
                        "check_result": (
                            {
                                "content": check_result.content,
                                "is_error": check_result.is_error,
                                "environment_step_executed": (
                                    check_result.environment_step_executed
                                ),
                                "details": check_result.details,
                            }
                            if check_result is not None
                            else None
                        ),
                        "phase": phase.value,
                        "repair_elapsed_ms": repair_elapsed_ms,
                        "check_elapsed_ms": check_elapsed_ms,
                        "total_elapsed_ms": total_elapsed_ms,
                        "observation_complete": observation_complete,
                        "evidence_id": evidence_id,
                        "evidence_state": (
                            evidence_state.value if evidence_state is not None else None
                        ),
                        "execution_error": execution_error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._runtime_repair_execution_sequence += 1

    async def _bash_in_environment(
        self,
        arguments: Mapping[str, Any],
        *,
        environment: TerminalToolEnvironment,
        containment_capability: BashContainmentCapability,
        apply_actor_deadline: bool,
    ) -> DelegatedToolResult:
        """Apply the same process containment to main and candidate environments."""

        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            return DelegatedToolResult("bash command is required", True, False, {})
        timeout = arguments.get("timeout", 120)
        if not isinstance(timeout, int) or timeout <= 0:
            return DelegatedToolResult("bash timeout must be a positive integer", True, False, {})
        requested_timeout = min(timeout, 600)
        effective_timeout = requested_timeout
        if apply_actor_deadline and self._actor_attempt_deadline is not None:
            remaining = self._actor_attempt_deadline - time.monotonic()
            effective_timeout = min(
                effective_timeout,
                math.floor(
                    remaining
                    - _BASH_OUTER_TIMEOUT_MARGIN_SECONDS
                    - _BASH_SESSION_COMPLETION_RESERVE_SECONDS
                ),
            )
        if effective_timeout < 1:
            return DelegatedToolResult(
                "Bash was not started because the Actor session is closing.",
                True,
                False,
                {
                    "requested_command_timeout_seconds": requested_timeout,
                    "process_group_containment": containment_capability.mode,
                },
            )
        try:
            result = await environment.exec(
                _contained_bash_command(
                    command,
                    timeout_seconds=effective_timeout,
                ),
                cwd="/app",
                env=None,
                # The outer host timer may fire only after container cleanup finishes.
                timeout_sec=effective_timeout + _BASH_OUTER_TIMEOUT_MARGIN_SECONDS,
            )
        except asyncio.CancelledError:
            # Harbor's portable exec API exposes no host-owned execution handle.
            # Never trust Actor-writable container state to authorize later cleanup.
            self._fatal_containment_error = RuntimeError(
                "external Bash cancellation makes this environment non-resumable"
            )
            raise
        output = "".join(part for part in (result.stdout, result.stderr) if part)
        if result.return_code != 0:
            output += f"\nCommand exited with code {result.return_code}"
        return DelegatedToolResult(
            output,
            result.return_code != 0,
            True,
            {
                "return_code": result.return_code,
                "requested_command_timeout_seconds": requested_timeout,
                "effective_command_timeout_seconds": effective_timeout,
                "process_group_containment": containment_capability.mode,
            },
        )

    async def _source_era_bash(self, arguments: Mapping[str, Any]) -> DelegatedToolResult:
        """Bash as the gateway ran it before process-group containment.

        No supervisor and no ``setsid`` requirement. A timeout ends the wait, not the
        command: the process keeps running in the container, as it did in the recorded
        source. The result text is the one the source recorded.
        """

        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            return DelegatedToolResult("bash command is required", True, False, {})
        timeout = arguments.get("timeout", 120)
        if not isinstance(timeout, int) or timeout <= 0:
            return DelegatedToolResult("bash timeout must be a positive integer", True, False, {})
        try:
            result = await self.environment.exec(command, cwd="/app", env=None, timeout_sec=min(timeout, 600))
        except RuntimeError as error:
            if "timed out" not in str(error):
                raise
            return DelegatedToolResult(
                f"RuntimeError: {error}", True, True, {"return_code": None, "bash_semantics": "source_era"}
            )
        output = "".join(part for part in (result.stdout, result.stderr) if part)
        if result.return_code != 0:
            output += f"\nCommand exited with code {result.return_code}"
        return DelegatedToolResult(
            output, result.return_code != 0, True, {"return_code": result.return_code, "bash_semantics": "source_era"}
        )

    async def _bash(self, arguments: Mapping[str, Any]) -> DelegatedToolResult:
        """Execute one contained Bash command in the protected live environment."""

        if self.source_era_bash:
            return await self._source_era_bash(arguments)
        if self.bash_containment_capability is None:
            self.bash_containment_capability = await probe_bash_containment_capability(
                self.environment
            )
        return await self._bash_in_environment(
            arguments,
            environment=self.environment,
            containment_capability=self.bash_containment_capability,
            apply_actor_deadline=True,
        )

    def _isolated_bash_executor_factory(self, environment: TerminalToolEnvironment):
        """Build an executor that preserves live Gateway Bash and check semantics."""

        containment_capability: BashContainmentCapability | None = None

        async def execute(
            arguments: Mapping[str, Any], registered_check: bool
        ) -> DelegatedToolResult:
            nonlocal containment_capability
            if containment_capability is None:
                containment_capability = await probe_bash_containment_capability(
                    environment
                )
            executed_arguments = dict(arguments)
            if registered_check:
                executed_arguments["command"] = (
                    _REGISTERED_CHECK_EXECUTION_PREFIX
                    + "\n"
                    + str(arguments["command"])
                )
            return await self._bash_in_environment(
                executed_arguments,
                environment=environment,
                containment_capability=containment_capability,
                apply_actor_deadline=True,
            )

        return execute

    async def _read(self, arguments: Mapping[str, Any], sequence: int) -> DelegatedToolResult:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str):
            return DelegatedToolResult("read path is required", True, False, {})
        offset = arguments.get("offset", 1)
        limit = arguments.get("limit", 2000)
        if not isinstance(offset, int) or offset < 1:
            return DelegatedToolResult("read offset must be at least 1", True, False, {})
        if not isinstance(limit, int) or limit < 1:
            return DelegatedToolResult("read limit must be positive", True, False, {})
        try:
            path = validate_actor_visible_path(raw_path)
        except ValueError as error:
            return DelegatedToolResult(str(error), True, False, {})
        if not await self.environment.is_file(path):
            return DelegatedToolResult(f"file not found: {path}", True, True, {})
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        local_path = self.scratch_root / f"read-{sequence}.bin"
        await self.environment.download_file(path, local_path)
        try:
            text = local_path.read_text()
        except UnicodeDecodeError:
            return DelegatedToolResult(f"file is not UTF-8 text: {path}", True, True, {})
        finally:
            local_path.unlink(missing_ok=True)
        lines = text.splitlines()
        selected = lines[offset - 1 : offset - 1 + limit]
        rendered = "\n".join(
            f"{line_number}: {line}" for line_number, line in enumerate(selected, start=offset)
        )
        return DelegatedToolResult(rendered, False, True, {"path": path})

    async def _write(self, arguments: Mapping[str, Any], sequence: int) -> DelegatedToolResult:
        raw_path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(raw_path, str) or not isinstance(content, str):
            return DelegatedToolResult("write path and content are required", True, False, {})
        try:
            path = validate_actor_visible_path(raw_path)
        except ValueError as error:
            return DelegatedToolResult(str(error), True, False, {})
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        local_path = self.scratch_root / f"write-{sequence}.txt"
        local_path.write_text(content)
        try:
            await self.environment.upload_file(local_path, path)
        finally:
            local_path.unlink(missing_ok=True)
        return DelegatedToolResult(
            f"Wrote {len(content.encode())} bytes to {path}", False, True, {}
        )

    async def _refresh_runtime(self, *, source: str) -> int:
        """Observe managed resources without coupling them to replay snapshots."""

        if self.resource_driver is None or self.runtime_kernel is None:
            return 0
        started = time.monotonic()
        observations = await self.resource_driver.observe()
        self.runtime_kernel.refresh_observations(observations, source=source)
        return round((time.monotonic() - started) * 1000)

    async def _observe_runtime_after_uncertain_effect(
        self,
    ) -> tuple[tuple[ResourceObservation, ...], bool]:
        """Return fresh facts, or explicit unknowns when observation itself fails."""

        if self.resource_driver is None:
            return (), False
        try:
            return await self.resource_driver.observe(), True
        except Exception:
            return self.resource_driver.unknown_observations(), False

    def _record_actor_failure_boundary(
        self, *, action_id: str, resource_change_observed: bool
    ) -> None:
        """Open one version-bound incident for every completed Actor failure."""

        assert self.runtime_kernel is not None
        if resource_change_observed:
            self._actor_failures_since_resource_change = 0
        self._actor_failures_since_resource_change += 1
        candidate_task_ids = tuple(
            sorted(
                task_id
                for task_id, status in self.runtime_kernel.task_statuses.items()
                if status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}
            )
        )
        self.runtime_kernel.record_actor_failure_boundary(
            action_id=action_id,
            failure_count_since_resource_change=self._actor_failures_since_resource_change,
            candidate_task_ids=candidate_task_ids,
            association_evidence_id="session_open_task_scope_v1",
        )

    async def _finish_runtime_action(
        self,
        *,
        action_id: str,
        result: DelegatedToolResult,
        observation: asyncio.Future | None = None,
    ) -> tuple[dict[str, object] | None, int]:
        """Close one action from its real receipt and post-action resource facts."""

        if self.resource_driver is None or self.runtime_kernel is None:
            return None, 0
        started = time.monotonic()
        execution_summary = _actor_execution_summary(result)
        observations = await (observation if observation is not None else self.resource_driver.observe())
        changed = any(
            self.runtime_kernel.observations.get(item.resource_id) != item for item in observations
        )
        # Resource observations do not cover every effect of an arbitrary tool.
        phase = (
            ActionPhase.FAILED_AFTER_PARTIAL_EFFECT
            if result.is_error and changed
            else ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
            if result.is_error
            else ActionPhase.COMPLETED
        )
        transition = self.runtime_kernel.finish_action(
            action_id,
            phase=phase,
            observations=observations,
            reason=("tool returned an error" if result.is_error else None),
            execution_summary=execution_summary,
        )
        if result.is_error:
            self._record_actor_failure_boundary(
                action_id=action_id,
                resource_change_observed=changed,
            )
        elif changed:
            self._actor_failures_since_resource_change = 0
        return (
            {
                "changed_resource_ids": list(transition.changed_resource_ids),
                "stale_evidence_ids": list(transition.stale_evidence_ids),
                "affected_task_ids": list(transition.affected_task_ids),
                "action_phase": phase.value,
                "actor_failures_since_resource_change": (
                    self._actor_failures_since_resource_change
                ),
            },
            round((time.monotonic() - started) * 1000),
        )

    async def _apply_shared_input_protection(self, *, sequence: int) -> list[dict[str, object]]:
        """Restore retained missing prerequisites without task-specific rules."""

        if (
            not self.restore_missing_prerequisites
            or self.resource_driver is None
            or self.runtime_kernel is None
        ):
            return []
        capabilities = self.resource_driver.restoration_capabilities()
        candidates: list[str] = []
        for node in self.runtime_kernel.task_nodes.values():
            for resource_id in (
                *node.prerequisite_resource_ids,
                *node.hypothesized_prerequisite_resource_ids,
            ):
                observation = self.runtime_kernel.observations.get(resource_id)
                capability = capabilities.get(resource_id)
                if (
                    observation is not None
                    and observation.observed
                    and observation.exists is False
                    and capability is not None
                    and capability.supported
                    and resource_id not in candidates
                ):
                    candidates.append(resource_id)
        interventions = []
        for index, resource_id in enumerate(candidates):
            resource_version = self.runtime_kernel.resource_versions[resource_id]
            failure_key = (resource_id, resource_version)
            if failure_key in self._failed_runtime_intervention_versions:
                continue
            action_id = f"runtime-restore-after-{sequence:06d}-{index:02d}"
            try:
                await self._restore_resource_unlocked(
                    action_id=action_id,
                    resource_id=resource_id,
                )
            except asyncio.CancelledError:
                # Shared protection has no policy decision to close. The restore
                # helper has already closed the Kernel action as cancelled.
                current_version = self.runtime_kernel.resource_versions[resource_id]
                self._failed_runtime_intervention_versions.add((resource_id, current_version))
                self.runtime_kernel.record_runtime_intervention_failure(
                    resource_id=resource_id,
                    resource_version=current_version,
                )
                raise
            except Exception as error:
                # A supervisor failure must not rewrite an already completed actor receipt.
                self._failed_runtime_intervention_versions.add(failure_key)
                self.runtime_kernel.record_runtime_intervention_failure(
                    resource_id=resource_id,
                    resource_version=resource_version,
                )
                interventions.append(
                    {
                        "action_id": action_id,
                        "resource_id": resource_id,
                        "policy": "shared_input_protection_v1",
                        "status": "failed",
                        "reason": str(error),
                    }
                )
            else:
                interventions.append(
                    {
                        "action_id": action_id,
                        "resource_id": resource_id,
                        "policy": "shared_input_protection_v1",
                        "status": "completed",
                    }
                )
        return interventions

    async def _policy_capabilities(self) -> dict[str, RuntimeRepairCapability]:
        """Project driver abilities into the narrow information visible to TraceDP."""

        if self.resource_driver is None or self.runtime_kernel is None:
            return {}
        missing_resource_ids = tuple(
            sorted(
                resource_id
                for node in self.runtime_kernel.task_nodes.values()
                for resource_id in (
                    *node.prerequisite_resource_ids,
                    *node.hypothesized_prerequisite_resource_ids,
                )
                if (
                    (observation := self.runtime_kernel.observations.get(resource_id)) is not None
                    and observation.observed
                    and observation.exists is False
                )
            )
        )
        live_capabilities = await self.resource_driver.current_restoration_capabilities(
            missing_resource_ids
        )
        confirmed_resources = {
            resource_id
            for node in self.runtime_kernel.task_nodes.values()
            for resource_id in node.prerequisite_resource_ids
        }
        return {
            resource_id: RuntimeRepairCapability(
                resource_id=resource_id,
                supported=capability.supported,
                cost_units=1,
                cost_bytes=capability.cost_bytes,
                success_probability=1.0,
                probability_evidence_id="terminal_exact_retention_contract_v1",
                necessity_probability=(1.0 if resource_id in confirmed_resources else 0.5),
                necessity_evidence_id=(
                    "confirmed_task_prerequisite_v1"
                    if resource_id in confirmed_resources
                    else "locally_designed_beta_1_1_initial_resource_prior_v1"
                ),
                support_reason=capability.reason,
            )
            for resource_id, capability in (live_capabilities.items())
        }

    async def _apply_dynamic_tracedp(
        self, *, sequence: int, trigger_action_id: str
    ) -> list[dict[str, object]]:
        """Choose, execute, observe, and replan repairs over the live task graph."""

        if (
            self.runtime_supervisor is None
            or self.resource_driver is None
            or self.runtime_kernel is None
        ):
            return []
        interventions: list[dict[str, object]] = []
        capabilities = await self._policy_capabilities()
        while True:
            decision = self.runtime_supervisor.plan(
                kernel=self.runtime_kernel,
                capabilities=capabilities,
                trigger_action_id=trigger_action_id,
                excluded_resource_versions=frozenset(self._failed_runtime_intervention_versions),
            )
            if decision is None or decision.action_id is None:
                break
            resource_id = decision.resource_id
            if resource_id is None:
                raise RuntimeError("TraceDP chose a repair without a resource")
            capability = capabilities[resource_id]
            resource_version = self.runtime_kernel.resource_versions[resource_id]
            action_prefix = (
                "runtime-fixed"
                if self.runtime_supervisor.policy_id == FIXED_TWO_STAGE_RUNTIME_METHOD
                else "runtime-tracedp"
            )
            action_id = f"{action_prefix}-after-{sequence:06d}-{len(interventions):02d}"
            self.runtime_supervisor.record_execution_started(action_id=action_id)
            try:
                await self._restore_resource_unlocked(
                    action_id=action_id,
                    resource_id=resource_id,
                )
            except asyncio.CancelledError:
                current_version = self.runtime_kernel.resource_versions[resource_id]
                self._failed_runtime_intervention_versions.update(
                    {(resource_id, resource_version), (resource_id, current_version)}
                )
                self.runtime_kernel.record_runtime_intervention_failure(
                    resource_id=resource_id,
                    resource_version=current_version,
                )
                self.runtime_supervisor.record_outcome(
                    action_id=action_id,
                    resource_id=resource_id,
                    cost_units=capability.cost_units,
                    status="cancelled_with_unknown_effect",
                    reason="runtime repair was cancelled before a durable receipt",
                    resource_version_after=current_version,
                )
                raise
            except Exception as error:
                reason = str(error)
                current_version = self.runtime_kernel.resource_versions[resource_id]
                self._failed_runtime_intervention_versions.update(
                    {(resource_id, resource_version), (resource_id, current_version)}
                )
                self.runtime_kernel.record_runtime_intervention_failure(
                    resource_id=resource_id,
                    resource_version=current_version,
                )
                self.runtime_supervisor.record_outcome(
                    action_id=action_id,
                    resource_id=resource_id,
                    cost_units=capability.cost_units,
                    status="failed",
                    reason=reason,
                    resource_version_after=current_version,
                )
                interventions.append(
                    {
                        "action_id": action_id,
                        "decision_action_id": decision.action_id,
                        "resource_id": resource_id,
                        "policy": self.runtime_supervisor.policy_id,
                        "status": "failed",
                        "reason": reason,
                    }
                )
                continue
            current_version = self.runtime_kernel.resource_versions[resource_id]
            self.runtime_supervisor.record_outcome(
                action_id=action_id,
                resource_id=resource_id,
                cost_units=capability.cost_units,
                status="completed",
                reason=None,
                resource_version_after=current_version,
            )
            interventions.append(
                {
                    "action_id": action_id,
                    "decision_action_id": decision.action_id,
                    "resource_id": resource_id,
                    "policy": self.runtime_supervisor.policy_id,
                    "status": "completed",
                }
            )
        if self.runtime_supervisor.policy_id in OUTPUT_SEMANTIC_RUNTIME_METHODS:
            interventions.extend(
                self._apply_dynamic_output_semantics(
                    sequence=sequence,
                    trigger_action_id=trigger_action_id,
                )
            )
        if self.runtime_supervisor.policy_id in OUTPUT_PROGRESS_RUNTIME_METHODS:
            interventions.extend(
                self._apply_dynamic_output_progress(
                    sequence=sequence,
                    trigger_action_id=trigger_action_id,
                )
            )
        return interventions

    def _apply_dynamic_output_semantics(
        self, *, sequence: int, trigger_action_id: str
    ) -> list[dict[str, object]]:
        """Request one independent semantic check when a complete output appears."""

        if (
            self.runtime_supervisor is None
            or self.runtime_kernel is None
            or self.intervention_effect_ledger is None
            or self.runtime_supervisor.policy_id not in OUTPUT_SEMANTIC_RUNTIME_METHODS
            or self.runtime_supervisor.remaining_budget_units < 1
        ):
            return []
        issues, candidates, _, obligations = compile_runtime_semantic_candidates(
            self.runtime_kernel,
            excluded_obligation_versions=frozenset(self._semantic_checked_obligation_versions),
        )
        if not candidates:
            return []
        decision = self.runtime_supervisor.plan_output_semantics(
            kernel=self.runtime_kernel,
            trigger_action_id=trigger_action_id,
            excluded_obligation_versions=frozenset(self._semantic_checked_obligation_versions),
        )
        if decision.action_id is None or decision.resource_id is None:
            return []
        candidate_payloads = decision.diagnostics.get("candidates")
        candidate_payload = (
            candidate_payloads.get(decision.action_id)
            if isinstance(candidate_payloads, dict)
            else None
        )
        if (
            not isinstance(candidate_payload, dict)
            or candidate_payload.get("action_kind") != "runtime_semantic_guidance"
        ):
            raise RuntimeError("TraceDP semantic decision lacks candidate evidence")
        issue_id = candidate_payload.get("issue_id")
        if not isinstance(issue_id, str) or issue_id not in issues:
            raise RuntimeError("TraceDP semantic decision lacks an output obligation")
        obligation = obligations[issue_id]
        resource_id = decision.resource_id
        resource_version = self.runtime_kernel.resource_versions[resource_id]
        action_id = f"runtime-semantic-guidance-after-{sequence:06d}"
        arguments = {
            "resource_id": resource_id,
            "resource_version": resource_version,
            "task_id": obligation.task_id,
            "obligation_id": obligation.obligation_id,
            "obligation_version_key": self._encode_obligation_key(obligation.version_key),
            "output_resource_ids": list(obligation.output_resource_ids),
            "output_requirement": obligation.output_requirement.value,
            "affected_task_ids": list(obligation.affected_task_ids),
            "phase": obligation.phase.value,
        }
        self.runtime_supervisor.record_execution_started(action_id=action_id)
        self.runtime_kernel.propose_action(
            action_id=action_id,
            tool_name="runtime_semantic_guidance",
            arguments=arguments,
            actor="runtime",
        )
        self.runtime_kernel.start_action(action_id)
        self.runtime_kernel.finish_action(
            action_id,
            phase=ActionPhase.COMPLETED,
            observations=(),
        )
        self.runtime_supervisor.record_outcome(
            action_id=action_id,
            resource_id=resource_id,
            cost_units=int(candidate_payload["cost_units"]),
            status="completed",
            reason=None,
            resource_version_after=resource_version,
        )
        self.intervention_effect_ledger.record_delivery(
            action_id=action_id,
            obligation=obligation,
            status=RuntimeInterventionDeliveryStatus.DELIVERED,
        )
        self._semantic_checked_obligation_versions.add(obligation.version_key)
        # Close this trigger with a durable abstention after the one intervention.
        self.runtime_supervisor.plan_output_semantics(
            kernel=self.runtime_kernel,
            trigger_action_id=trigger_action_id,
            excluded_obligation_versions=frozenset(self._semantic_checked_obligation_versions),
        )
        return [
            {
                "action_id": action_id,
                "decision_action_id": decision.action_id,
                "resource_id": resource_id,
                "policy": self.runtime_supervisor.policy_id,
                "intervention_kind": "runtime_semantic_guidance",
                "status": "completed",
                "task_id": obligation.task_id,
                "obligation_id": obligation.obligation_id,
                "obligation_version_key": arguments["obligation_version_key"],
                "output_resource_ids": list(obligation.output_resource_ids),
                "output_requirement": obligation.output_requirement.value,
                "affected_task_ids": list(obligation.affected_task_ids),
            }
        ]

    def _record_pending_intervention_effects(self, *, evidence_action_id: str) -> None:
        """Close a delivered intervention only after later output evidence changes."""

        if self.intervention_effect_ledger is None or self.runtime_kernel is None:
            return
        current = {
            item.obligation_id: item
            for item in compile_runtime_output_obligations(self.runtime_kernel)
        }
        for record in self.intervention_effect_ledger.records:
            if record.effect_status is not RuntimeInterventionEffectStatus.PENDING:
                continue
            obligation = current.get(record.obligation_id)
            if obligation is not None and obligation.version_key != record.obligation_version_key:
                self.intervention_effect_ledger.record_effect(
                    action_id=record.action_id,
                    status=RuntimeInterventionEffectStatus.OUTPUT_CHANGED,
                    evidence_id=evidence_action_id,
                )

    def _apply_dynamic_output_progress(
        self, *, sequence: int, trigger_action_id: str
    ) -> list[dict[str, object]]:
        """Emit one auditable reminder when declared outputs remain unchanged."""

        if (
            self.runtime_supervisor is None
            or self.runtime_kernel is None
            or self.runtime_supervisor.policy_id
            not in {
                "dynamic_tracedp_runtime_v2",
                "dynamic_tracedp_runtime_v3",
                "dynamic_tracedp_runtime_v4",
                FIXED_TWO_STAGE_RUNTIME_METHOD,
            }
            or self._actor_actions_since_output_progress < OUTPUT_PROGRESS_ACTION_THRESHOLD
            or self.runtime_supervisor.remaining_budget_units < 1
        ):
            return []
        _, available, _ = compile_runtime_progress_candidates(
            self.runtime_kernel,
            excluded_resource_versions=frozenset(self._guided_output_versions),
        )
        if not available:
            self._actor_actions_since_output_progress = 0
            return []
        no_progress_actions = self._actor_actions_since_output_progress
        decision = self.runtime_supervisor.plan_output_progress(
            kernel=self.runtime_kernel,
            trigger_action_id=trigger_action_id,
            excluded_resource_versions=frozenset(self._guided_output_versions),
            no_progress_actor_actions=no_progress_actions,
            trigger_threshold=OUTPUT_PROGRESS_ACTION_THRESHOLD,
        )
        if decision.action_id is None or decision.resource_id is None:
            self._actor_actions_since_output_progress = 0
            return []
        candidates = decision.diagnostics.get("candidates")
        candidate = candidates.get(decision.action_id) if isinstance(candidates, dict) else None
        if not isinstance(candidate, dict) or candidate.get("action_kind") != ("runtime_guidance"):
            raise RuntimeError("TraceDP progress decision lacks guidance evidence")
        task_id = decision.action_id.removeprefix("guide-output:")
        task = self.runtime_kernel.task_nodes.get(task_id)
        if task is None or not task.required_output_resource_ids:
            raise RuntimeError("TraceDP progress decision lacks a public output task")
        obligation = None
        if self.runtime_supervisor.policy_id in {
            "dynamic_tracedp_runtime_v4",
            FIXED_TWO_STAGE_RUNTIME_METHOD,
        }:
            obligation = next(
                (
                    item
                    for item in compile_runtime_output_obligations(self.runtime_kernel)
                    if item.task_id == task_id
                ),
                None,
            )
            if obligation is None:
                raise RuntimeError("TraceDP progress decision lacks an output obligation")
        resource_id = decision.resource_id
        resource_version = self.runtime_kernel.resource_versions[resource_id]
        action_id = f"runtime-guidance-after-{sequence:06d}"
        arguments = {
            "resource_id": resource_id,
            "resource_version": resource_version,
            "task_id": task_id,
            "output_resource_ids": list(task.required_output_resource_ids),
            "output_requirement": task.output_requirement.value,
            "no_progress_actor_actions": no_progress_actions,
            "trigger_threshold": OUTPUT_PROGRESS_ACTION_THRESHOLD,
        }
        if obligation is not None:
            arguments.update(
                {
                    "obligation_id": obligation.obligation_id,
                    "obligation_version_key": self._encode_obligation_key(obligation.version_key),
                    "affected_task_ids": list(obligation.affected_task_ids),
                    "phase": obligation.phase.value,
                }
            )
        self.runtime_supervisor.record_execution_started(action_id=action_id)
        self.runtime_kernel.propose_action(
            action_id=action_id,
            tool_name="runtime_guidance",
            arguments=arguments,
            actor="runtime",
        )
        self.runtime_kernel.start_action(action_id)
        self.runtime_kernel.finish_action(
            action_id,
            phase=ActionPhase.COMPLETED,
            observations=(),
        )
        self.runtime_supervisor.record_outcome(
            action_id=action_id,
            resource_id=resource_id,
            cost_units=int(candidate["cost_units"]),
            status="completed",
            reason=None,
            resource_version_after=resource_version,
        )
        if obligation is not None:
            if self.intervention_effect_ledger is None:
                raise RuntimeError("TraceDP v4 progress guidance lacks an effect ledger")
            self.intervention_effect_ledger.record_delivery(
                action_id=action_id,
                obligation=obligation,
                status=RuntimeInterventionDeliveryStatus.DELIVERED,
            )
        self._guided_output_versions.add((resource_id, resource_version))
        self._actor_actions_since_output_progress = 0

        # Close the trigger with a durable stop decision after the one intervention.
        current_candidates = {
            (
                item.resource_id,
                self.runtime_kernel.resource_versions[item.resource_id],
            )
            for item in available
        }
        self.runtime_supervisor.plan_output_progress(
            kernel=self.runtime_kernel,
            trigger_action_id=trigger_action_id,
            excluded_resource_versions=frozenset(
                self._guided_output_versions.union(current_candidates)
            ),
            no_progress_actor_actions=no_progress_actions,
            trigger_threshold=OUTPUT_PROGRESS_ACTION_THRESHOLD,
        )
        intervention = {
            "action_id": action_id,
            "decision_action_id": decision.action_id,
            "resource_id": resource_id,
            "policy": self.runtime_supervisor.policy_id,
            "intervention_kind": "runtime_guidance",
            "status": "completed",
            "task_id": task_id,
            "output_resource_ids": list(task.required_output_resource_ids),
            "output_requirement": task.output_requirement.value,
            "no_progress_actor_actions": no_progress_actions,
        }
        if obligation is not None:
            intervention.update(
                {
                    "obligation_id": obligation.obligation_id,
                    "obligation_version_key": arguments["obligation_version_key"],
                    "affected_task_ids": list(obligation.affected_task_ids),
                    "phase": obligation.phase.value,
                }
            )
        return [intervention]

    def _joint_output_obligation(
        self, candidate: RuntimeJointActionCandidate
    ) -> RuntimeOutputObligation:
        """Resolve the exact obligation frozen into one joint action instance."""

        if self.runtime_kernel is None:
            raise RuntimeError("joint runtime Kernel is unavailable")
        binding = candidate.target_binding
        obligation = next(
            (
                item
                for item in compile_runtime_output_obligations(self.runtime_kernel)
                if item.obligation_id == binding.obligation_id
                and item.task_id == binding.task_id
                and item.version_key == binding.obligation_version_key
            ),
            None,
        )
        if obligation is None:
            raise RuntimeError("joint runtime output obligation changed after selection")
        return obligation

    def _execute_predicate_guidance(
        self, *, sequence: int, selection_id: str, candidate: RuntimeJointActionCandidate,
    ) -> dict[str, object]:
        """Deliver a check-and-repair contract; completion here means delivery only."""
        assert self.runtime_kernel is not None and self.joint_runtime_policy is not None
        kernel = self.runtime_kernel
        incident = next((item for item in kernel.predicate_incidents.values()
                         if candidate.instance_action_id ==
                         f"diagnose-predicate:{item.incident_id}:{item.latest_evidence_id}"), None)
        if (incident is None or incident.status != "open"
                or incident.incident_id in self._diagnosed_predicate_incident_ids
                or kernel.evidence[incident.latest_evidence_id].state.value != "refuted"):
            raise RuntimeError("predicate diagnosis changed after selection")
        source = kernel.evidence[incident.latest_evidence_id]
        source_action = kernel.actions[source.source_action_id]
        source_check_arguments = (
            source_action.arguments
            if source_action.actor == "actor"
            else source_action.arguments.get("check_arguments")
        )
        contract = next(
            (
                item
                for item in self.check_contracts
                if item.task_id == incident.task_id
                and item.proposition == incident.proposition
                and isinstance(source_check_arguments, Mapping)
                and dict(item.arguments) == dict(source_check_arguments)
                and item.resource_ids == tuple(dict(source.resource_versions))
            ),
            None,
        )
        if contract is None:
            raise RuntimeError("predicate diagnosis source differs from its check contract")
        arguments = {
            "incident_id": incident.incident_id, "task_id": incident.task_id,
            "proposition": incident.proposition, "evidence_id": source.evidence_id,
            "source_action_id": source.source_action_id,
            "check_arguments": dict(contract.arguments),
            "supported_output": contract.supported_output,
            "refuted_output": contract.refuted_output,
            **(
                {"verdict_protocol": contract.verdict_protocol}
                if contract.verdict_protocol != EXACT_OUTPUT_VERDICT_PROTOCOL
                else {}
            ),
            "resource_ids": list(contract.resource_ids),
            "effect_status": "awaiting_recheck",
        }
        action_id = f"runtime-predicate-guidance-after-{sequence:06d}"
        self.joint_runtime_policy.record_execution_started(action_id=action_id)
        kernel.propose_action(action_id=action_id, tool_name="runtime_predicate_guidance",
                              arguments=arguments, actor="runtime")
        kernel.start_action(action_id)
        kernel.finish_action(action_id, phase=ActionPhase.COMPLETED, observations=())
        self.joint_runtime_policy.record_outcome(status="completed", reason=None)
        self._diagnosed_predicate_incident_ids.add(incident.incident_id)
        return {
            **arguments, "action_id": action_id, "decision_action_id": candidate.instance_action_id,
            "joint_decision_id": selection_id, "resource_id": None,
            "policy": self.joint_runtime_policy.method_id,
            "intervention_kind": "runtime_predicate_guidance", "status": "completed",
        }

    def _execute_registered_check_guidance(
        self, *, sequence: int, selection_id: str, candidate: RuntimeJointActionCandidate,
    ) -> dict[str, object]:
        """Ask the same Actor for one exact check without treating delivery as evidence."""

        assert self.runtime_kernel is not None and self.joint_runtime_policy is not None
        prefix = "request-registered-check:"
        if not candidate.instance_action_id.startswith(prefix):
            raise RuntimeError("registered check selection identity is invalid")
        try:
            check_index = int(candidate.instance_action_id[len(prefix):].split(":", 1)[0])
            contract = self.check_contracts[check_index]
        except (IndexError, TypeError, ValueError) as error:
            raise RuntimeError("registered check selection is unavailable") from error
        versions = tuple(
            (resource_id, self.runtime_kernel.resource_versions[resource_id])
            for resource_id in contract.resource_ids
        )
        available = pending_runtime_check_candidates(
            self.runtime_kernel,
            self.check_contracts,
            excluded_versions=frozenset(self._handled_registered_check_versions),
        )
        if candidate not in available:
            raise RuntimeError("registered check state changed after selection")
        arguments = {
            "check_index": check_index,
            **(
                {
                    "check_id": contract.check_id,
                    "predicate_id": contract.predicate_id,
                }
                if contract.check_id is not None
                else {}
            ),
            "resource_id": versions[0][0],
            "resource_version": versions[0][1],
            "task_id": contract.task_id,
            "proposition": contract.proposition,
            "check_arguments": dict(contract.arguments),
            "supported_output": contract.supported_output,
            "refuted_output": contract.refuted_output,
            **(
                {"verdict_protocol": contract.verdict_protocol}
                if contract.verdict_protocol != EXACT_OUTPUT_VERDICT_PROTOCOL
                else {}
            ),
            "resource_versions": [list(item) for item in versions],
            "effect_status": "awaiting_exact_execution",
        }
        action_id = f"runtime-registered-check-guidance-after-{sequence:06d}"
        self.joint_runtime_policy.record_execution_started(action_id=action_id)
        self.runtime_kernel.propose_action(
            action_id=action_id,
            tool_name="runtime_registered_check_guidance",
            arguments=arguments,
            actor="runtime",
        )
        self.runtime_kernel.start_action(action_id)
        self.runtime_kernel.finish_action(
            action_id,
            phase=ActionPhase.COMPLETED,
            observations=(),
        )
        self.joint_runtime_policy.record_outcome(status="completed", reason=None)
        self._handled_registered_check_versions.add((check_index, versions))
        return {
            **arguments,
            "action_id": action_id,
            "decision_action_id": candidate.instance_action_id,
            "joint_decision_id": selection_id,
            "resource_id": candidate.target_binding.resource_id,
            "policy": self.joint_runtime_policy.method_id,
            "intervention_kind": "runtime_registered_check_guidance",
            "status": "completed",
        }

    async def _execute_registered_check(
        self,
        *,
        sequence: int,
        selection_id: str,
        trigger_action_id: str,
        candidate: RuntimeJointActionCandidate,
    ) -> dict[str, object]:
        """Execute one selected check in the live environment and record its verdict."""

        assert self.runtime_kernel is not None and self.joint_runtime_policy is not None
        prefix = "run-registered-check:"
        if not candidate.instance_action_id.startswith(prefix):
            raise RuntimeError("runtime check selection identity is invalid")
        try:
            check_index = int(candidate.instance_action_id[len(prefix):].split(":", 1)[0])
            contract = self.check_contracts[check_index]
        except (IndexError, TypeError, ValueError) as error:
            raise RuntimeError("runtime check selection is unavailable") from error
        available = pending_runtime_check_candidates(
            self.runtime_kernel,
            self.check_contracts,
            excluded_versions=frozenset(self._handled_registered_check_versions),
        )
        if candidate not in available:
            raise RuntimeError("runtime check state changed after selection")
        versions_before = tuple(
            (resource_id, self.runtime_kernel.resource_versions[resource_id])
            for resource_id in contract.resource_ids
        )
        # v3 subject/prerequisite split: the check stays executable when its SUBJECT is
        # missing (that is the predicate failing), but when a checker PREREQUISITE is missing
        # the command is replaced by a declared non-0/1 exit so the verdict is UNKNOWN rather
        # than a refutation of the task predicate.
        prerequisites_available = check_prerequisites_available(
            self.runtime_kernel, contract
        )
        exit_status_protocol = (
            contract.verdict_protocol == EXIT_STATUS_VERDICT_PROTOCOL
        )
        execution_mode = (
            "bash_pipefail_errtrap_exit_status_v2"
            if exit_status_protocol
            else "bash_pipefail_errtrap_exact_output_v2"
        )
        verdict_interpretation = (
            "return_0_supported_1_refuted_other_unknown_v1"
            if exit_status_protocol
            else "exit_0_exact_declared_output_other_unknown_v1"
        )
        arguments = {
            "check_index": check_index,
            **(
                {
                    "check_id": contract.check_id,
                    "predicate_id": contract.predicate_id,
                }
                if contract.check_id is not None
                else {}
            ),
            "resource_id": versions_before[0][0],
            "resource_version": versions_before[0][1],
            "task_id": contract.task_id,
            "proposition": contract.proposition,
            "check_arguments": dict(contract.arguments),
            "resource_ids": list(contract.resource_ids),
            "resource_versions": [list(item) for item in versions_before],
            "execution_mode": execution_mode,
            "verdict_interpretation": verdict_interpretation,
            "prerequisite_status": (
                "available" if prerequisites_available else "missing"
            ),
            **(
                {"prerequisite_resource_ids": list(contract.prerequisite_resource_ids)}
                if contract.prerequisite_resource_ids
                else {}
            ),
            **(
                {"verdict_protocol": contract.verdict_protocol}
                if exit_status_protocol
                else {}
            ),
        }
        action_id = self._runtime_action_id("runtime-registered-check", sequence)
        self.joint_runtime_policy.record_execution_started(action_id=action_id)
        self.runtime_kernel.propose_action(
            action_id=action_id,
            tool_name="runtime_registered_check_execution",
            arguments=arguments,
            actor="runtime",
        )
        self.runtime_kernel.start_action(action_id)
        command = (
            str(contract.arguments["command"])
            if prerequisites_available
            else (
                f"echo {CHECKER_PREREQUISITE_MISSING_MARKER}: "
                + ",".join(contract.prerequisite_resource_ids)
                + f"; exit {CHECKER_PREREQUISITE_MISSING_EXIT_CODE}"
            )
        )
        executed_arguments = {
            **dict(contract.arguments),
            # ERR preserves an unhandled failure before a later `;` command while
            # Bash still exempts explicit `if`, `&&`, `||`, and `!` control flow.
            "command": _REGISTERED_CHECK_EXECUTION_PREFIX + "\n" + command,
        }
        try:
            result = await self._bash(executed_arguments)
            observations, observation_complete = (
                await self._observe_runtime_after_uncertain_effect()
            )
        except asyncio.CancelledError:
            observations, observation_complete = (
                await self._observe_runtime_after_uncertain_effect()
            )
            self.runtime_kernel.finish_action(
                action_id,
                phase=ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT,
                observations=observations,
                reason="runtime check execution was cancelled",
            )
            self.joint_runtime_policy.record_outcome(
                status="cancelled_with_unknown_effect",
                reason="runtime check execution was cancelled",
            )
            versions_after = tuple(
                (resource_id, self.runtime_kernel.resource_versions[resource_id])
                for resource_id in contract.resource_ids
            )
            cancellation_reason = "CancelledError: runtime check execution was cancelled"
            self._append_runtime_check_execution(
                action_id=action_id,
                trigger_action_id=trigger_action_id,
                check_index=check_index,
                declared_arguments=contract.arguments,
                executed_arguments=executed_arguments,
                result=None,
                resource_versions_before=versions_before,
                resource_versions_after=versions_after,
                evidence_id=None,
                evidence_state=None,
                observation_complete=observation_complete,
                execution_mode=execution_mode,
                blocking_fact={
                    "return_code": None,
                    "output": cancellation_reason,
                },
                execution_error=cancellation_reason,
            )
            self._handled_registered_check_versions.add((check_index, versions_before))
            raise
        except Exception as error:
            observations, observation_complete = (
                await self._observe_runtime_after_uncertain_effect()
            )
            reason = f"{type(error).__name__}: {error}"
            self.runtime_kernel.finish_action(
                action_id,
                phase=ActionPhase.FAILED_WITH_UNKNOWN_EFFECT,
                observations=observations,
                reason=reason,
            )
            versions_after = tuple(
                (resource_id, self.runtime_kernel.resource_versions[resource_id])
                for resource_id in contract.resource_ids
            )
            self._append_runtime_check_execution(
                action_id=action_id,
                trigger_action_id=trigger_action_id,
                check_index=check_index,
                declared_arguments=contract.arguments,
                executed_arguments=executed_arguments,
                result=None,
                resource_versions_before=versions_before,
                resource_versions_after=versions_after,
                evidence_id=None,
                evidence_state=None,
                observation_complete=observation_complete,
                execution_mode=execution_mode,
                blocking_fact={"return_code": None, "output": reason},
                execution_error=reason,
            )
            self.joint_runtime_policy.record_outcome(status="failed", reason=reason)
            self._handled_registered_check_versions.add((check_index, versions_before))
            return {
                "action_id": action_id,
                "decision_action_id": candidate.instance_action_id,
                "joint_decision_id": selection_id,
                "resource_id": candidate.target_binding.resource_id,
                "policy": self.joint_runtime_policy.method_id,
                "intervention_kind": "runtime_registered_check_execution",
                "status": "failed",
                "reason": reason,
                "task_id": contract.task_id,
                "proposition": contract.proposition,
                "predicate_id": contract.predicate_id,
                "check_id": contract.check_id,
                "return_code": None,
                "evidence_id": None,
                "evidence_state": None,
                "content": "",
                "blocking_fact": {"return_code": None, "output": reason},
                "resource_versions_before": [list(item) for item in versions_before],
                "resource_versions_after": [list(item) for item in versions_after],
            }
        if result.environment_step_executed is not True:
            phase = ActionPhase.FAILED_WITHOUT_EFFECT
            reason = "runtime check was not executed"
        elif not observation_complete:
            phase = ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
            reason = "runtime check post-observation failed"
        else:
            # A refuted predicate is a successful check action, not a tool failure.
            phase = ActionPhase.COMPLETED
            reason = None
        self.runtime_kernel.finish_action(
            action_id,
            phase=phase,
            observations=observations,
            reason=reason,
        )
        return_code = result.details.get("return_code")
        evidence_state = None
        evidence_id = None
        interpreted_state = interpret_runtime_check_result(
            contract,
            content=result.content,
            return_code=return_code,
            is_error=result.is_error,
        )
        blocking_fact = runtime_check_blocking_fact(
            interpreted_state,
            content=result.content,
            return_code=return_code,
        )
        if result.environment_step_executed is True and observation_complete:
            evidence_state = interpreted_state
            evidence_id = f"runtime-check:{action_id}:{check_index}"
            record_runtime_check_evidence(
                self.runtime_kernel,
                contract,
                evidence_id=evidence_id,
                source_action_id=action_id,
                state=evidence_state,
            )
        versions_after = tuple(
            (resource_id, self.runtime_kernel.resource_versions[resource_id])
            for resource_id in contract.resource_ids
        )
        self._append_runtime_check_execution(
            action_id=action_id,
            trigger_action_id=trigger_action_id,
            check_index=check_index,
            declared_arguments=contract.arguments,
            executed_arguments=executed_arguments,
            result=result,
            resource_versions_before=versions_before,
            resource_versions_after=versions_after,
            evidence_id=evidence_id,
            evidence_state=evidence_state,
            observation_complete=observation_complete,
            execution_mode=execution_mode,
            blocking_fact=blocking_fact,
            execution_error=None,
        )
        policy_status = "completed" if phase is ActionPhase.COMPLETED else "failed"
        self.joint_runtime_policy.record_outcome(status=policy_status, reason=reason)
        self._handled_registered_check_versions.add((check_index, versions_before))
        return {
            "action_id": action_id,
            "decision_action_id": candidate.instance_action_id,
            "joint_decision_id": selection_id,
            "resource_id": candidate.target_binding.resource_id,
            "policy": self.joint_runtime_policy.method_id,
            "intervention_kind": "runtime_registered_check_execution",
            "status": policy_status,
            "task_id": contract.task_id,
            "proposition": contract.proposition,
            "predicate_id": contract.predicate_id,
            "check_id": contract.check_id,
            "return_code": return_code,
            "evidence_id": evidence_id,
            "evidence_state": (
                evidence_state.value if evidence_state is not None else None
            ),
            "content": result.content,
            "blocking_fact": blocking_fact,
            "resource_versions_before": [list(item) for item in versions_before],
            "resource_versions_after": [list(item) for item in versions_after],
        }

    async def _execute_targeted_repair(
        self,
        *,
        sequence: int,
        selection_id: str,
        trigger_action_id: str,
        candidate: RuntimeJointActionCandidate,
    ) -> dict[str, object]:
        """Run one configured repair and require its registered postcondition check."""

        assert self.runtime_kernel is not None and self.joint_runtime_policy is not None
        prefix = "run-targeted-repair:"
        if not candidate.instance_action_id.startswith(prefix):
            raise RuntimeError("targeted repair selection identity is invalid")
        try:
            repair_id = candidate.instance_action_id[len(prefix):].split(":", 1)[0]
            repair_index, repair = next(
                (index, item)
                for index, item in enumerate(self.repair_contracts)
                if item.repair_id == repair_id
            )
            execution = candidate.execution_contract
            assert execution is not None
            compiled_arguments = execution.proposed_check_or_repair["arguments"]
            assert isinstance(compiled_arguments, Mapping)
            check_index = int(compiled_arguments["check_index"])
            incident_id = str(compiled_arguments["incident_id"])
            check = self.check_contracts[check_index]
        except (AssertionError, IndexError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError("targeted repair selection is unavailable") from error
        available = pending_runtime_repair_candidates(
            self.runtime_kernel,
            self.repair_contracts,
            self.check_contracts,
            excluded_action_instance_ids=frozenset(
                self._handled_targeted_repair_action_ids
            ),
        )
        if self.targeted_repair_trial_gate is not None:
            available = tuple(
                self.targeted_repair_trial_gate.bind_planning_action(item)
                for item in available
            )
        if candidate not in available:
            raise RuntimeError("targeted repair state changed after selection")
        expected_compiled_arguments = {
            "repair_id": repair.repair_id,
            "repair_arguments": dict(repair.arguments),
            "postcondition_check_arguments": dict(check.arguments),
            "check_index": check_index,
            "incident_id": incident_id,
        }
        if self.targeted_repair_trial_gate is not None:
            expected_compiled_arguments["candidate_trial_protocol"] = (
                self.targeted_repair_trial_gate.protocol_id
            )
        if compiled_arguments != expected_compiled_arguments:
            raise RuntimeError("targeted repair execution contract changed after selection")
        versions_before = tuple(
            (resource_id, self.runtime_kernel.resource_versions[resource_id])
            for resource_id in repair.resource_ids
        )
        action_id = self._runtime_action_id("runtime-targeted-repair", sequence)
        action_arguments = {
            "decision_action_id": candidate.instance_action_id,
            "repair_id": repair.repair_id,
            "repair_index": repair_index,
            "check_index": check_index,
            "incident_id": incident_id,
            "resource_id": versions_before[0][0],
            "resource_version": versions_before[0][1],
            "task_id": repair.task_id,
            "proposition": repair.proposition,
            "repair_arguments": dict(repair.arguments),
            "check_arguments": dict(check.arguments),
            "resource_ids": list(repair.resource_ids),
            "resource_versions": [list(item) for item in versions_before],
            "time_limit_ms": execution.time_limit_ms,
        }
        if check.check_id is not None:
            action_arguments.update(
                check_id=check.check_id,
                predicate_id=check.predicate_id,
            )
        if self.targeted_repair_trial_gate is not None:
            action_arguments["candidate_trial_protocol"] = (
                self.targeted_repair_trial_gate.protocol_id
            )
        self.joint_runtime_policy.record_execution_started(action_id=action_id)
        self.runtime_kernel.propose_action(
            action_id=action_id,
            tool_name="runtime_targeted_repair_execution",
            arguments=action_arguments,
            actor="runtime",
        )
        self.runtime_kernel.start_action(action_id)
        total_started = time.monotonic()
        repair_started = total_started
        repair_result: DelegatedToolResult | None = None
        check_result: DelegatedToolResult | None = None
        repair_elapsed_ms = 0
        check_elapsed_ms = 0
        evidence_id = None
        evidence_state = None
        isolated_trial = None
        observation_complete = False
        execution_error = None
        phase: ActionPhase | None = None
        guidance_repair = is_actor_guidance_repair_arguments(repair.arguments)
        if guidance_repair and (
            self.targeted_repair_trial_gate is None
            or self._isolated_guidance_executor is None
        ):
            raise RuntimeError(
                "actor guidance candidate requires isolated trial and Actor runner"
            )
        repair_arguments = {
            **dict(repair.arguments),
            "timeout": repair.time_limit_seconds,
        }
        check_arguments = {
            **dict(check.arguments),
            "command": _REGISTERED_CHECK_EXECUTION_PREFIX
            + "\n"
            + str(check.arguments["command"]),
        }
        try:
            if self.targeted_repair_trial_gate is not None:
                failure_state_id = (
                    f"{incident_id}:"
                    + ",".join(
                        f"{resource_id}={version}"
                        for resource_id, version in versions_before
                    )
                )
                isolated_trial = await self.targeted_repair_trial_gate.run_trial(
                    trial_id=action_id,
                    failure_state_id=failure_state_id,
                    repair=repair,
                    checks=(check,),
                    execute_isolated_bash_factory=(
                        self._isolated_bash_executor_factory
                    ),
                    execute_isolated_guidance=self._isolated_guidance_executor,
                )
                if isolated_trial.verdict is not EvidenceState.SUPPORTED:
                    observations, observation_complete = (
                        await self._observe_runtime_after_uncertain_effect()
                    )
                    phase = (
                        ActionPhase.FAILED_WITHOUT_EFFECT
                        if isolated_trial.main_state_unchanged and observation_complete
                        else ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
                    )
                    execution_error = (
                        "isolated candidate was not admitted: "
                        f"{isolated_trial.verdict.value}"
                    )
                    repair_result = DelegatedToolResult(
                        content=execution_error,
                        is_error=True,
                        environment_step_executed=False,
                        details={
                            "candidate_trial_receipt": isolated_trial.to_dict(),
                            "return_code": isolated_trial.repair_return_code,
                            **(
                                {"candidate_form": "actor_guidance"}
                                if guidance_repair
                                else {}
                            ),
                        },
                    )
                elif guidance_repair:
                    observations, observation_complete = (
                        await self._observe_runtime_after_uncertain_effect()
                    )
                    if isolated_trial.main_state_unchanged and observation_complete:
                        phase = ActionPhase.COMPLETED
                        repair_result = DelegatedToolResult(
                            content=str(repair.arguments["guidance_text"]),
                            is_error=False,
                            environment_step_executed=False,
                            details={
                                "candidate_trial_receipt": isolated_trial.to_dict(),
                                "candidate_form": "actor_guidance",
                                "effect_status": "awaiting_recheck",
                                "return_code": isolated_trial.repair_return_code,
                            },
                        )
                    else:
                        phase = ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
                        execution_error = (
                            "isolated actor guidance trial disturbed the live environment"
                        )
                        repair_result = DelegatedToolResult(
                            content=execution_error,
                            is_error=True,
                            environment_step_executed=False,
                            details={
                                "candidate_trial_receipt": isolated_trial.to_dict(),
                                "candidate_form": "actor_guidance",
                                "return_code": isolated_trial.repair_return_code,
                            },
                        )
                else:
                    repair_result = await self._bash(repair_arguments)
            else:
                repair_result = await self._bash(repair_arguments)
            repair_elapsed_ms = round((time.monotonic() - repair_started) * 1000)
            if phase is not None:
                pass
            elif repair_result.environment_step_executed is not True or repair_result.is_error:
                observations, observation_complete = (
                    await self._observe_runtime_after_uncertain_effect()
                )
                changed = observation_complete and any(
                    self.runtime_kernel.observations.get(item.resource_id) != item
                    for item in observations
                )
                phase = (
                    ActionPhase.FAILED_WITHOUT_EFFECT
                    if repair_result.environment_step_executed is not True
                    else ActionPhase.FAILED_AFTER_PARTIAL_EFFECT
                    if changed
                    else ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
                )
                execution_error = repair_result.content
            else:
                check_started = time.monotonic()
                check_result = await self._bash(check_arguments)
                check_elapsed_ms = round((time.monotonic() - check_started) * 1000)
                observations, observation_complete = (
                    await self._observe_runtime_after_uncertain_effect()
                )
                phase = (
                    ActionPhase.COMPLETED
                    if check_result.environment_step_executed is True
                    and observation_complete
                    else ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
                )
                if phase is not ActionPhase.COMPLETED:
                    execution_error = "targeted repair postcondition check was incomplete"
        except asyncio.CancelledError:
            repair_elapsed_ms = repair_elapsed_ms or round(
                (time.monotonic() - repair_started) * 1000
            )
            observations, observation_complete = (
                await self._observe_runtime_after_uncertain_effect()
            )
            phase = ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
            execution_error = "CancelledError: targeted repair execution was cancelled"
            self.runtime_kernel.finish_action(
                action_id,
                phase=phase,
                observations=observations,
                reason=execution_error,
            )
            # Independently rounded phase durations must not exceed the total receipt.
            total_elapsed_ms = max(
                round((time.monotonic() - total_started) * 1000),
                repair_elapsed_ms + check_elapsed_ms,
            )
            self._append_runtime_repair_execution(
                action_id=action_id,
                trigger_action_id=trigger_action_id,
                repair_index=repair_index,
                check_index=check_index,
                incident_id=incident_id,
                repair_arguments=repair_arguments,
                check_arguments=check_arguments,
                repair_result=repair_result,
                check_result=check_result,
                phase=phase,
                repair_elapsed_ms=repair_elapsed_ms,
                check_elapsed_ms=check_elapsed_ms,
                total_elapsed_ms=total_elapsed_ms,
                observation_complete=observation_complete,
                evidence_id=None,
                evidence_state=None,
                execution_error=execution_error,
            )
            self._handled_targeted_repair_action_ids.add(candidate.instance_action_id)
            self.joint_runtime_policy.record_outcome(
                status="cancelled_with_unknown_effect", reason=execution_error
            )
            raise
        except Exception as error:
            repair_elapsed_ms = repair_elapsed_ms or round(
                (time.monotonic() - repair_started) * 1000
            )
            observations, observation_complete = (
                await self._observe_runtime_after_uncertain_effect()
            )
            changed = observation_complete and any(
                self.runtime_kernel.observations.get(item.resource_id) != item
                for item in observations
            )
            phase = (
                ActionPhase.FAILED_AFTER_PARTIAL_EFFECT
                if changed
                else ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
            )
            execution_error = f"{type(error).__name__}: {error}"
        assert phase is not None
        self.runtime_kernel.finish_action(
            action_id,
            phase=phase,
            observations=observations,
            reason=(None if phase is ActionPhase.COMPLETED else execution_error),
        )
        if phase is ActionPhase.COMPLETED and check_result is not None:
            evidence_state = interpret_runtime_check_result(
                check,
                content=check_result.content,
                return_code=check_result.details.get("return_code"),
                is_error=check_result.is_error,
            )
            evidence_id = f"runtime-check:{action_id}:{check_index}"
            record_runtime_check_evidence(
                self.runtime_kernel,
                check,
                evidence_id=evidence_id,
                source_action_id=action_id,
                state=evidence_state,
            )
        total_elapsed_ms = max(
            round((time.monotonic() - total_started) * 1000),
            repair_elapsed_ms + check_elapsed_ms,
        )
        self._append_runtime_repair_execution(
            action_id=action_id,
            trigger_action_id=trigger_action_id,
            repair_index=repair_index,
            check_index=check_index,
            incident_id=incident_id,
            repair_arguments=repair_arguments,
            check_arguments=check_arguments,
            repair_result=repair_result,
            check_result=check_result,
            phase=phase,
            repair_elapsed_ms=repair_elapsed_ms,
            check_elapsed_ms=check_elapsed_ms,
            total_elapsed_ms=total_elapsed_ms,
            observation_complete=observation_complete,
            evidence_id=evidence_id,
            evidence_state=evidence_state,
            execution_error=execution_error,
        )
        self._handled_targeted_repair_action_ids.add(candidate.instance_action_id)
        policy_status = (
            "completed"
            if phase is ActionPhase.COMPLETED
            else "cancelled_with_unknown_effect"
            if phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
            else "failed"
        )
        self.joint_runtime_policy.record_outcome(
            status=policy_status,
            reason=(None if policy_status == "completed" else execution_error),
        )
        incident = self.runtime_kernel.predicate_incidents[incident_id]
        return {
            "action_id": action_id,
            "decision_action_id": candidate.instance_action_id,
            "joint_decision_id": selection_id,
            "resource_id": candidate.target_binding.resource_id,
            "policy": self.joint_runtime_policy.method_id,
            "intervention_kind": "runtime_targeted_repair_execution",
            "status": policy_status,
            "task_id": repair.task_id,
            "proposition": repair.proposition,
            "incident_id": incident_id,
            "incident_status": incident.status,
            "evidence_id": evidence_id,
            "evidence_state": (
                evidence_state.value if evidence_state is not None else None
            ),
            "repair_elapsed_ms": repair_elapsed_ms,
            "check_elapsed_ms": check_elapsed_ms,
            "total_elapsed_ms": total_elapsed_ms,
            "action_phase": phase.value,
            "reason": execution_error,
            "candidate_form": "actor_guidance" if guidance_repair else "bash_command",
            "effect_status": (
                "awaiting_recheck"
                if guidance_repair and phase is ActionPhase.COMPLETED
                else None
            ),
            "guidance_text": (
                str(repair.arguments["guidance_text"])
                if guidance_repair and phase is ActionPhase.COMPLETED
                else None
            ),
            "candidate_trial_receipt": (
                isolated_trial.to_dict() if isolated_trial is not None else None
            ),
        }

    def _execute_joint_guidance(
        self,
        *,
        sequence: int,
        selection_id: str,
        candidate: RuntimeJointActionCandidate,
        action_id: str | None = None,
        execution_already_started: bool = False,
    ) -> dict[str, object]:
        """Deliver exactly one selected output intervention and close its lifecycle."""

        if (
            self.runtime_kernel is None
            or self.joint_runtime_policy is None
            or self.intervention_effect_ledger is None
        ):
            raise RuntimeError("joint output intervention is not configured")
        obligation = self._joint_output_obligation(candidate)
        binding = candidate.target_binding
        resource_id = binding.resource_id
        resource_version = binding.resource_version
        if resource_id is None or resource_version is None:
            raise RuntimeError("joint output intervention lacks a resource binding")
        semantic = candidate.action_type is RuntimeJointActionType.REQUEST_SEMANTIC_CHECK
        failure_diagnosis = (
            candidate.action_type is RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS
        )
        convergence = candidate.action_type is RuntimeJointActionType.REQUEST_ACTOR_CONVERGENCE
        action_kind = (
            "runtime_convergence_guidance"
            if convergence
            else "runtime_failure_diagnosis_guidance"
            if failure_diagnosis
            else "runtime_semantic_guidance"
            if semantic
            else "runtime_guidance"
        )
        action_id = action_id or f"runtime-joint-{action_kind}-after-{sequence:06d}"
        arguments = {
            "resource_id": resource_id,
            "resource_version": resource_version,
            "task_id": obligation.task_id,
            "obligation_id": obligation.obligation_id,
            "obligation_version_key": self._encode_obligation_key(obligation.version_key),
            "output_resource_ids": list(obligation.output_resource_ids),
            "output_requirement": obligation.output_requirement.value,
            "affected_task_ids": list(obligation.affected_task_ids),
            "phase": obligation.phase.value,
        }
        if failure_diagnosis:
            boundary = (
                tuple(self.runtime_kernel.actor_failure_boundaries.values())[-1]
                if self.runtime_kernel.actor_failure_boundaries
                else None
            )
            if (
                boundary is None
                or boundary.action_id in self._diagnosed_failure_action_ids
                or boundary.resource_versions
                != tuple(sorted(self.runtime_kernel.resource_versions.items()))
                or runtime_failure_diagnosis_incident_key(
                    resource_versions=boundary.resource_versions,
                    obligation_version_key=obligation.version_key,
                )
                in self._diagnosed_failure_incident_keys
                or candidate.instance_action_id
                != (
                    f"diagnose-failure:{boundary.action_id}:"
                    f"{boundary.failure_count_since_resource_change}:"
                    f"{obligation.obligation_id}"
                )
            ):
                raise RuntimeError("failure diagnosis boundary changed after selection")
            arguments.update(
                {
                    "failure_action_id": boundary.action_id,
                    "failure_count_since_resource_change": (
                        boundary.failure_count_since_resource_change
                    ),
                    "failure_resource_versions": [
                        list(item) for item in boundary.resource_versions
                    ],
                    "failure_candidate_task_ids": list(boundary.candidate_task_ids),
                    "failure_association_evidence_id": (boundary.association_evidence_id),
                }
            )
            if self.joint_runtime_policy.method_id in TRACEDP_RESIDUAL_METHODS:
                arguments["failure_diagnosis_context"] = self._compile_failure_diagnosis_context(
                    candidate_task_ids=boundary.candidate_task_ids,
                    failure_action_id=boundary.action_id,
                )
        elif convergence:
            if self._actor_attempt_deadline is None or self.actor_time_budget_seconds is None:
                raise RuntimeError("convergence guidance lacks an active time boundary")
            arguments.update(
                {
                    "actor_time_budget_ms": self.actor_time_budget_seconds * 1000,
                    "actor_time_remaining_ms": max(
                        0,
                        round((self._actor_attempt_deadline - time.monotonic()) * 1000),
                    ),
                    "convergence_reserve_ms": ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000,
                }
            )
        elif not semantic:
            arguments.update(
                {
                    "no_progress_actor_actions": self._actor_actions_since_output_progress,
                    "trigger_threshold": OUTPUT_PROGRESS_ACTION_THRESHOLD,
                }
            )
        if not execution_already_started:
            self.joint_runtime_policy.record_execution_started(action_id=action_id)
        try:
            self.runtime_kernel.propose_action(
                action_id=action_id,
                tool_name=action_kind,
                arguments=arguments,
                actor="runtime",
            )
            self.runtime_kernel.start_action(action_id)
            self.runtime_kernel.finish_action(
                action_id,
                phase=ActionPhase.COMPLETED,
                observations=(),
            )
            self.intervention_effect_ledger.record_delivery(
                action_id=action_id,
                obligation=obligation,
                status=RuntimeInterventionDeliveryStatus.DELIVERED,
            )
        except Exception as error:
            self.joint_runtime_policy.record_outcome(
                status="failed", reason=f"{type(error).__name__}: {error}"
            )
            raise
        self.joint_runtime_policy.record_outcome(status="completed", reason=None)
        if semantic:
            self._semantic_checked_obligation_versions.add(obligation.version_key)
        elif failure_diagnosis:
            self._diagnosed_failure_action_ids.add(str(arguments["failure_action_id"]))
            self._diagnosed_failure_incident_keys.add(
                runtime_failure_diagnosis_incident_key(
                    resource_versions=self._decode_resource_versions(
                        arguments["failure_resource_versions"]
                    ),
                    obligation_version_key=obligation.version_key,
                )
            )
            self._diagnosed_obligation_versions.add(obligation.version_key)
        elif convergence:
            self._actor_convergence_requested = True
        else:
            self._guided_output_versions.add((resource_id, resource_version))
            self._actor_actions_since_output_progress = 0
        intervention = {
            "action_id": action_id,
            "decision_action_id": candidate.instance_action_id,
            "joint_decision_id": selection_id,
            "resource_id": resource_id,
            "policy": self.joint_runtime_policy.method_id,
            "intervention_kind": action_kind,
            "status": "completed",
            "task_id": obligation.task_id,
            "obligation_id": obligation.obligation_id,
            "obligation_version_key": arguments["obligation_version_key"],
            "output_resource_ids": list(obligation.output_resource_ids),
            "output_requirement": obligation.output_requirement.value,
            "affected_task_ids": list(obligation.affected_task_ids),
            "phase": obligation.phase.value,
        }
        if failure_diagnosis:
            intervention.update(
                {
                    "failure_action_id": arguments["failure_action_id"],
                    "failure_count_since_resource_change": arguments[
                        "failure_count_since_resource_change"
                    ],
                    "failure_resource_versions": arguments["failure_resource_versions"],
                    "failure_candidate_task_ids": arguments["failure_candidate_task_ids"],
                    "failure_association_evidence_id": arguments["failure_association_evidence_id"],
                }
            )
            if "failure_diagnosis_context" in arguments:
                intervention["failure_diagnosis_context"] = arguments["failure_diagnosis_context"]
        elif not semantic and not convergence:
            intervention["no_progress_actor_actions"] = arguments["no_progress_actor_actions"]
        if convergence:
            intervention.update(
                {
                    "actor_time_budget_ms": arguments["actor_time_budget_ms"],
                    "actor_time_remaining_ms": arguments["actor_time_remaining_ms"],
                    "convergence_reserve_ms": arguments["convergence_reserve_ms"],
                }
            )
        return intervention

    def attach_budget_gate(self, provider, log_path: Path | None = None) -> None:
        """Harness hook: ``provider(visible_chars) -> (room, next_estimate)``.

        Each dropped repair is also appended to ``log_path`` (an audit of the gate).
        """

        self._budget_gate = provider
        self._budget_gate_log = log_path

    def _gated_slate(self, slate: RuntimeJointActionSlate) -> RuntimeJointActionSlate:
        """Drop pool-consuming repairs the affordability gate refuses; checks stay."""

        provider = getattr(self, "_budget_gate", None)
        if provider is None:
            return slate
        room, next_estimate = provider(self._last_visible_chars)
        if repair_admissible(room, next_estimate):
            return slate
        kept = tuple(
            candidate for candidate in slate.candidates
            if candidate.action_type is not RuntimeJointActionType.RUN_TARGETED_REPAIR
        )
        options = tuple(
            option for option in slate.conditional_future_options
            if option.action_type is not RuntimeJointActionType.RUN_TARGETED_REPAIR
        )
        event = {
            "room": room, "next_estimate": next_estimate,
            "dropped_repairs": len(slate.candidates) - len(kept),
        }
        self.budget_gate_events.append(event)
        if getattr(self, "_budget_gate_log", None) is not None:
            self._budget_gate_log.parent.mkdir(parents=True, exist_ok=True)
            with self._budget_gate_log.open("a") as output:
                output.write(json.dumps(event) + "\n")
        return replace(slate, candidates=kept, conditional_future_options=options)

    async def _apply_joint_runtime_policy(
        self,
        *,
        sequence: int,
        trigger_action_id: str,
        output_semantics_enabled: bool | None = None,
        option_chain_ids: frozenset[str] | None = None,
    ) -> list[dict[str, object]]:
        """Execute one selection, plus one evidence-gated option replan when enabled."""

        if (
            self.joint_runtime_policy is None
            or self.resource_driver is None
            or self.runtime_kernel is None
        ):
            return []
        settlement_only = self.joint_runtime_policy.method_id in SETTLEMENT_ONLY_JOINT_METHODS
        failure_diagnosis_enabled = (
            self.joint_runtime_policy.method_id in FAILURE_DIAGNOSIS_JOINT_METHODS
        )
        diagnosis_only = (
            self.joint_runtime_policy.method_id in FAILURE_DIAGNOSIS_ONLY_JOINT_METHODS
        )
        capabilities = (
            {}
            if settlement_only or diagnosis_only or option_chain_ids is not None
            else await self._policy_capabilities()
        )
        actor_time_budget_ms = None
        actor_time_remaining_ms = None
        actor_convergence_reserve_ms = None
        if (
            self.joint_runtime_policy.method_id == FIXED_JOINT_SLATE_V3_METHOD
            or self.selector_time_observation_enabled
        ):
            if self._actor_attempt_deadline is None or self.actor_time_budget_seconds is None:
                raise RuntimeError("time-aware joint runtime lacks an active time boundary")
            actor_time_budget_ms = self.actor_time_budget_seconds * 1000
            actor_time_remaining_ms = max(
                0, round((self._actor_attempt_deadline - time.monotonic()) * 1000)
            )
            if self.joint_runtime_policy.method_id == FIXED_JOINT_SLATE_V3_METHOD:
                actor_convergence_reserve_ms = ACTOR_CONVERGENCE_RESERVE_SECONDS * 1000
        registered_check_candidates = (
            ()
            if option_chain_ids is not None
            else pending_runtime_check_candidates(
                self.runtime_kernel,
                self.check_contracts,
                excluded_versions=frozenset(
                    self._handled_registered_check_versions
                ),
            )
        )
        targeted_repair_candidates = pending_runtime_repair_candidates(
            self.runtime_kernel,
            self.repair_contracts,
            self.check_contracts,
            excluded_action_instance_ids=frozenset(
                self._handled_targeted_repair_action_ids
            ),
        )
        if self.targeted_repair_trial_gate is not None:
            targeted_repair_candidates = tuple(
                self.targeted_repair_trial_gate.bind_planning_action(item)
                for item in targeted_repair_candidates
            )
        if option_chain_ids is not None:
            # Deferred objects never reach the executor. Recompilation must find
            # the exact now-executable candidate corresponding to the first plan.
            targeted_repair_candidates = tuple(
                item
                for item in targeted_repair_candidates
                if item.conditional_future_option_id in option_chain_ids
            )
            if not targeted_repair_candidates:
                raise RuntimeError(
                    "runtime option replan has no executable contract-bound repair"
                )
        conditional_future_options = (
            ()
            if option_chain_ids is not None
            else conditional_runtime_repair_options(
                self.runtime_kernel,
                self.repair_contracts,
                self.check_contracts,
                registered_check_candidates,
                budget_before=self.joint_runtime_policy.remaining_budget_units,
                actor_time_remaining_ms=actor_time_remaining_ms,
            )
        )
        if self.targeted_repair_trial_gate is not None:
            conditional_future_options = tuple(
                self.targeted_repair_trial_gate.bind_planning_action(item)
                for item in conditional_future_options
            )
            prerequisite_costs = {
                item.instance_action_id: item.cost_units
                for item in registered_check_candidates
            }
            conditional_future_options = tuple(
                item
                for item in conditional_future_options
                if prerequisite_costs[item.prerequisite_action_id] + item.cost_units
                <= self.joint_runtime_policy.remaining_budget_units
            )
        slate = compile_runtime_joint_action_slate(
            kernel=self.runtime_kernel,
            capabilities=capabilities,
            trigger_action_id=trigger_action_id,
            budget_before=self.joint_runtime_policy.remaining_budget_units,
            no_progress_actor_actions=self._actor_actions_since_output_progress,
            progress_trigger_threshold=self.progress_trigger_threshold,
            excluded_resource_versions=frozenset(self._failed_runtime_intervention_versions),
            # The counter is the cooldown. A prior reminder must not suppress the
            # same still-missing output after another full no-progress interval.
            excluded_progress_resource_versions=(
                frozenset()
                if self.joint_runtime_policy.method_id in SEMANTIC_STABILITY_JOINT_METHODS
                else frozenset(self._guided_output_versions)
            ),
            excluded_semantic_obligation_versions=frozenset(
                self._semantic_checked_obligation_versions
                | (
                    self._diagnosed_obligation_versions
                    if self.joint_runtime_policy.method_id
                    == TRACEDP_RESIDUAL_RUNTIME_V3_METHOD
                    else set()
                )
            ),
            output_semantics_enabled=(
                False
                if option_chain_ids is not None
                else
                output_semantics_enabled
                if output_semantics_enabled is not None
                else False
                if (
                    settlement_only
                    or diagnosis_only
                    or self.joint_runtime_policy.method_id == FIXED_JOINT_SLATE_V5_METHOD
                )
                else self._actor_actions_since_output_progress
                >= OUTPUT_SEMANTIC_STABILITY_ACTION_THRESHOLD
                if self.joint_runtime_policy.method_id in SEMANTIC_STABILITY_JOINT_METHODS
                else True
            ),
            output_progress_enabled=(
                option_chain_ids is None and not settlement_only and not diagnosis_only
            ),
            failure_diagnosis_enabled=(
                option_chain_ids is None and failure_diagnosis_enabled
            ),
            failure_diagnosis_threshold=(
                1
                if self.joint_runtime_policy.method_id
                in EARLY_FAILURE_DIAGNOSIS_JOINT_METHODS
                else ACTOR_FAILURE_DIAGNOSIS_THRESHOLD
            ),
            excluded_failure_action_ids=frozenset(self._diagnosed_failure_action_ids),
            excluded_failure_incident_keys=frozenset(
                self._diagnosed_failure_incident_keys
            ),
            predicate_diagnosis_enabled=(
                option_chain_ids is None and self.predicate_diagnosis_enabled
            ),
            excluded_predicate_incident_ids=frozenset(self._diagnosed_predicate_incident_ids),
            registered_check_candidates=registered_check_candidates,
            targeted_repair_candidates=targeted_repair_candidates,
            conditional_future_options=conditional_future_options,
            actor_time_budget_ms=actor_time_budget_ms,
            actor_time_remaining_ms=actor_time_remaining_ms,
            actor_convergence_reserve_ms=actor_convergence_reserve_ms,
        )
        slate = self._gated_slate(slate)
        action_domain_projection = None
        if self.joint_runtime_policy.method_id in RUNTIME_OPTION_CHAIN_METHODS:
            # Fixed and Bellman selectors consume the same declared sub-domain.
            slate, action_domain_projection = project_runtime_option_action_domain(
                slate
            )
        selection = self.joint_runtime_policy.select(
            slate,
            action_domain_projection=action_domain_projection,
        )
        candidate = selection.selected
        if candidate.action_type is RuntimeJointActionType.CONTINUE_ACTOR:
            return []
        if candidate.action_type is RuntimeJointActionType.REQUEST_PREDICATE_DIAGNOSIS:
            return [self._execute_predicate_guidance(
                sequence=sequence, selection_id=selection.decision_id, candidate=candidate,
            )]
        if candidate.action_type is RuntimeJointActionType.REQUEST_REGISTERED_CHECK:
            return [self._execute_registered_check_guidance(
                sequence=sequence, selection_id=selection.decision_id, candidate=candidate,
            )]
        if candidate.action_type is RuntimeJointActionType.RUN_REGISTERED_CHECK:
            matching_options = tuple(
                item
                for item in slate.conditional_future_options
                if item.prerequisite_action_id == candidate.instance_action_id
            )
            if (
                self.joint_runtime_policy.method_id in RUNTIME_OPTION_CHAIN_METHODS
                and option_chain_ids is None
                and matching_options
            ):
                self.joint_runtime_policy.open_option_chain(
                    selection=selection,
                    expected_check_action_id=self._runtime_action_id(
                        "runtime-registered-check", sequence
                    ),
                    conditional_future_option_ids=tuple(
                        item.option_id for item in matching_options
                    ),
                )
            intervention = await self._execute_registered_check(
                sequence=sequence,
                selection_id=selection.decision_id,
                trigger_action_id=trigger_action_id,
                candidate=candidate,
            )
            chain = self.joint_runtime_policy.open_option_chain_state
            followup: list[dict[str, object]] = []
            replanned = False
            if chain is not None:
                replan_id = self._runtime_option_replan_action_id(chain)
                followup = await self._complete_runtime_option_chain(
                    sequence=sequence,
                    intervention=intervention,
                )
                replanned = replan_id in self.runtime_kernel.actions
            if not replanned:
                followup = [
                    *followup,
                    *await self._runtime_option_followup(
                        sequence=sequence,
                        check_action_id=str(intervention["action_id"]),
                    ),
                ]
            return [intervention, *followup]
        if candidate.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR:
            return [await self._execute_targeted_repair(
                sequence=sequence,
                selection_id=selection.decision_id,
                trigger_action_id=trigger_action_id,
                candidate=candidate,
            )]
        if candidate.action_type in {
            RuntimeJointActionType.REQUEST_OUTPUT_COMPLETION,
            RuntimeJointActionType.REQUEST_ACTOR_CONVERGENCE,
            RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS,
            RuntimeJointActionType.REQUEST_SEMANTIC_CHECK,
        }:
            return [
                self._execute_joint_guidance(
                    sequence=sequence,
                    selection_id=selection.decision_id,
                    candidate=candidate,
                )
            ]
        binding = candidate.target_binding
        resource_id = binding.resource_id
        resource_version = binding.resource_version
        if resource_id is None or resource_version is None:
            raise RuntimeError("joint restore lacks a resource binding")
        action_id = f"runtime-joint-restore-after-{sequence:06d}"
        self.joint_runtime_policy.record_execution_started(action_id=action_id)
        try:
            await self._restore_resource_unlocked(
                action_id=action_id,
                resource_id=resource_id,
            )
        except asyncio.CancelledError:
            current_version = self.runtime_kernel.resource_versions[resource_id]
            self._failed_runtime_intervention_versions.update(
                {(resource_id, resource_version), (resource_id, current_version)}
            )
            self.runtime_kernel.record_runtime_intervention_failure(
                resource_id=resource_id,
                resource_version=current_version,
            )
            self.joint_runtime_policy.record_outcome(
                status="cancelled_with_unknown_effect",
                reason="runtime repair was cancelled before a durable receipt",
            )
            raise
        except Exception as error:
            current_version = self.runtime_kernel.resource_versions[resource_id]
            self._failed_runtime_intervention_versions.update(
                {(resource_id, resource_version), (resource_id, current_version)}
            )
            self.runtime_kernel.record_runtime_intervention_failure(
                resource_id=resource_id,
                resource_version=current_version,
            )
            reason = f"{type(error).__name__}: {error}"
            self.joint_runtime_policy.record_outcome(status="failed", reason=reason)
            return [
                {
                    "action_id": action_id,
                    "decision_action_id": candidate.instance_action_id,
                    "joint_decision_id": selection.decision_id,
                    "resource_id": resource_id,
                    "policy": self.joint_runtime_policy.method_id,
                    "status": "failed",
                    "reason": reason,
                }
            ]
        self.joint_runtime_policy.record_outcome(status="completed", reason=None)
        return [
            {
                "action_id": action_id,
                "decision_action_id": candidate.instance_action_id,
                "joint_decision_id": selection.decision_id,
                "resource_id": resource_id,
                "policy": self.joint_runtime_policy.method_id,
                "status": "completed",
            }
        ]

    def _runtime_action_id(self, prefix: str, sequence: int) -> str:
        """One Runtime action id per execution; later ones at a boundary get a suffix.

        Same-boundary follow-up decisions can run several checks after one Actor tool
        call. The suffix is derived from the durable Kernel, so a resumed process cannot
        reuse an id; the first action keeps the historical id.
        """

        assert self.runtime_kernel is not None
        base = f"{prefix}-after-{sequence:06d}"
        if base not in self.runtime_kernel.actions:
            return base
        index = 1
        while f"{base}-f{index:02d}" in self.runtime_kernel.actions:
            index += 1
        return f"{base}-f{index:02d}"

    async def _runtime_option_followup(
        self, *, sequence: int, check_action_id: str
    ) -> list[dict[str, object]]:
        """Decide again at the same boundary after a check that did not open a replan.

        Without this, a supported check ends the boundary and the next decision waits
        for another Actor tool call; an Actor that settles quickly then leaves the
        budget unused. The follow-up slate is the full
        slate, and the loop ends when the selector continues the Actor or the budget is
        spent.
        """

        if (
            self.joint_runtime_policy is None
            or self.runtime_kernel is None
            or self.joint_runtime_policy.method_id not in RUNTIME_OPTION_CHAIN_METHODS
            or self.joint_runtime_policy.remaining_budget_units <= 0
        ):
            return []
        trigger_action_id = "actor-bridge-runtime-option-followup:" + check_action_id
        if trigger_action_id in self.runtime_kernel.actions:
            return []
        self.runtime_kernel.propose_action(
            action_id=trigger_action_id,
            tool_name=RUNTIME_OPTION_FOLLOWUP_TRIGGER,
            arguments={"after_check_action_id": check_action_id},
            actor="actor_bridge",
        )
        self.runtime_kernel.start_action(trigger_action_id)
        self.runtime_kernel.finish_action(
            trigger_action_id,
            phase=ActionPhase.COMPLETED,
            observations=(),
        )
        return await self._apply_joint_runtime_policy(
            sequence=sequence,
            trigger_action_id=trigger_action_id,
        )

    @staticmethod
    def _runtime_option_replan_action_id(chain: RuntimeOptionChain) -> str:
        """Derive the one stable bridge identity owned by a persisted option chain."""

        return "actor-bridge-runtime-option-replan:" + chain.chain_id

    def _joint_decision_exists_for_trigger(self, trigger_action_id: str) -> bool:
        """Read the durable policy log without inferring a decision from Kernel names."""

        assert self.joint_runtime_policy is not None
        if not self.joint_runtime_policy.event_log_path.is_file():
            return False
        for line in self.joint_runtime_policy.event_log_path.read_text().splitlines():
            row = json.loads(line)
            payload = row.get("payload") if isinstance(row, dict) else None
            slate = payload.get("slate") if isinstance(payload, dict) else None
            if (
                row.get("event_type") == "decision_selected"
                and isinstance(slate, dict)
                and slate.get("trigger_action_id") == trigger_action_id
            ):
                return True
        return False

    async def _complete_runtime_option_chain(
        self,
        *,
        sequence: int,
        intervention: Mapping[str, object],
    ) -> list[dict[str, object]]:
        """Close or replan one durable chain before any later Actor tool executes."""

        assert self.joint_runtime_policy is not None
        assert self.runtime_kernel is not None
        chain = self.joint_runtime_policy.open_option_chain_state
        if chain is None:
            return []
        evidence_state = intervention.get("evidence_state")
        if evidence_state not in {
            EvidenceState.REFUTED.value,
            EvidenceState.UNKNOWN.value,
        }:
            self.joint_runtime_policy.close_option_chain(
                evidence_state=(evidence_state if isinstance(evidence_state, str) else None),
                reason="check_did_not_require_repair_replan",
                replan_trigger_action_id=None,
            )
            return []
        if self.joint_runtime_policy.remaining_budget_units <= 0:
            self.joint_runtime_policy.close_option_chain(
                evidence_state=evidence_state,
                reason="repair_replan_budget_exhausted",
                replan_trigger_action_id=None,
            )
            return []
        replan_action_id = self._runtime_option_replan_action_id(chain)
        existing_trigger = self.runtime_kernel.actions.get(replan_action_id)
        if existing_trigger is None:
            self.runtime_kernel.propose_action(
                action_id=replan_action_id,
                tool_name="runtime_option_replan_trigger",
                arguments={
                    "option_chain_id": chain.chain_id,
                    "parent_trigger_action_id": chain.parent_trigger_action_id,
                    "prerequisite_decision_id": chain.prerequisite_decision_id,
                    "prerequisite_instance_action_id": (
                        chain.prerequisite_instance_action_id
                    ),
                    "prerequisite_evidence_state": evidence_state,
                    "conditional_future_option_ids": list(
                        chain.conditional_future_option_ids
                    ),
                },
                actor="actor_bridge",
            )
            self.runtime_kernel.start_action(replan_action_id)
            self.runtime_kernel.finish_action(
                replan_action_id,
                phase=ActionPhase.COMPLETED,
                observations=(),
            )
        elif (
            existing_trigger.actor != "actor_bridge"
            or existing_trigger.tool_name != "runtime_option_replan_trigger"
            or existing_trigger.phase is not ActionPhase.COMPLETED
            or existing_trigger.arguments.get("option_chain_id") != chain.chain_id
        ):
            raise RuntimeError("persisted runtime option replan trigger differs")
        followup = []
        if not self._joint_decision_exists_for_trigger(replan_action_id):
            followup = await self._apply_joint_runtime_policy(
                sequence=sequence,
                trigger_action_id=replan_action_id,
                output_semantics_enabled=False,
                option_chain_ids=frozenset(chain.conditional_future_option_ids),
            )
        if self.joint_runtime_policy.has_open_decision:
            raise RuntimeError("runtime option replan left an execution decision open")
        self.joint_runtime_policy.close_option_chain(
            evidence_state=evidence_state,
            reason="evidence_gated_replan_completed",
            replan_trigger_action_id=replan_action_id,
        )
        return followup

    async def _resume_runtime_option_chain(
        self, *, sequence: int
    ) -> list[dict[str, object]]:
        """Resume one interrupted check-to-replan chain before a new Actor boundary."""

        if self.joint_runtime_policy is None or self.runtime_kernel is None:
            return []
        chain = self.joint_runtime_policy.open_option_chain_state
        if chain is None:
            return []
        replan_action_id = self._runtime_option_replan_action_id(chain)
        replan_action = self.runtime_kernel.actions.get(replan_action_id)
        if (
            replan_action is not None
            and self._joint_decision_exists_for_trigger(replan_action_id)
        ):
            evidence_state = replan_action.arguments.get(
                "prerequisite_evidence_state"
            )
            if evidence_state not in {
                EvidenceState.REFUTED.value,
                EvidenceState.UNKNOWN.value,
            }:
                raise RuntimeError("persisted runtime option replan evidence differs")
            self.joint_runtime_policy.close_option_chain(
                evidence_state=evidence_state,
                reason="recovered_completed_evidence_gated_replan",
                replan_trigger_action_id=replan_action_id,
            )
            return []
        action = self.runtime_kernel.actions.get(chain.expected_check_action_id)
        if action is None:
            if self.joint_runtime_policy.has_open_decision:
                self.joint_runtime_policy.abandon_before_environment_effect(
                    reason=(
                        "runtime resumed before the option check crossed the "
                        "durable Kernel boundary"
                    )
                )
            self.joint_runtime_policy.close_option_chain(
                evidence_state=None,
                reason="interrupted_before_check_action_became_durable",
                replan_trigger_action_id=None,
            )
            return []
        if (
            action.actor != "runtime"
            or action.tool_name != "runtime_registered_check_execution"
        ):
            raise RuntimeError("persisted runtime option check action differs")
        if self.joint_runtime_policy.has_open_decision:
            if action.phase in {ActionPhase.PROPOSED, ActionPhase.STARTED}:
                raise RuntimeError("runtime option check still has an unfinished Kernel phase")
            policy_status = (
                "completed"
                if action.phase is ActionPhase.COMPLETED
                else "cancelled_with_unknown_effect"
                if action.phase is ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT
                else "failed"
            )
            self.joint_runtime_policy.record_outcome(
                status=policy_status,
                reason=(None if policy_status == "completed" else action.reason),
            )
        evidence = [
            item
            for item in self.runtime_kernel.evidence.values()
            if item.source_action_id == chain.expected_check_action_id
            and item.check_id is not None
        ]
        if len(evidence) > 1:
            raise RuntimeError("runtime option check produced ambiguous evidence")
        evidence_state = evidence[0].state.value if evidence else None
        intervention = {
            "evidence_state": evidence_state,
            "action_id": chain.expected_check_action_id,
            "intervention_kind": "runtime_registered_check_execution",
            "status": (
                "completed" if action.phase is ActionPhase.COMPLETED else "failed"
            ),
        }
        return await self._complete_runtime_option_chain(
            sequence=sequence,
            intervention=intervention,
        )

    async def _apply_runtime_supervision(
        self, *, sequence: int, trigger_action_id: str
    ) -> list[dict[str, object]]:
        """Dispatch exactly one frozen runtime intervention policy."""

        if self._runtime_supervision_disabled_reason is not None:
            return []
        if self.joint_runtime_policy is not None:
            return await self._apply_joint_runtime_policy(
                sequence=sequence,
                trigger_action_id=trigger_action_id,
            )
        if self.runtime_supervisor is not None:
            return await self._apply_dynamic_tracedp(
                sequence=sequence,
                trigger_action_id=trigger_action_id,
            )
        return await self._apply_shared_input_protection(sequence=sequence)

    @staticmethod
    def _runtime_notice(interventions: list[dict[str, object]]) -> str:
        """Expose supervisor state changes to the actor without hiding tool output."""

        notices = []
        resolved_repair_predicates = {
            intervention.get("proposition")
            for intervention in interventions
            if intervention.get("intervention_kind")
            == "runtime_targeted_repair_execution"
            and intervention.get("evidence_state") == EvidenceState.SUPPORTED.value
        }
        for intervention in interventions:
            if intervention["status"] == "supervisor_disabled":
                notices.append(
                    "Runtime supervision was disabled after an internal failure: "
                    f"{intervention['reason']}"
                )
                continue
            if intervention.get("intervention_kind") == "runtime_predicate_guidance":
                if intervention.get("verdict_protocol") == EXIT_STATUS_VERDICT_PROTOCOL:
                    verdict_instruction = (
                        "The exact recheck uses exit status 0 for supported, 1 for "
                        "refuted, and any other status for unknown."
                    )
                else:
                    verdict_instruction = (
                        f"Its passing output is {intervention['supported_output']!r}; "
                        f"its failing output is {intervention['refuted_output']!r}."
                    )
                notices.append(
                    f"The checked condition is not satisfied: {intervention['proposition']}. "
                    f"Evidence action: {intervention['source_action_id']}. "
                    "Inspect the cause and repair the affected prerequisite while preserving other work. "
                    f"Then rerun this exact check: {json.dumps(intervention['check_arguments'])}. "
                    f"{verdict_instruction} "
                    "A successful repair command alone does not resolve the incident. "
                    "If the check is inconclusive, retain uncertainty and revise the plan."
                )
                continue
            if intervention.get("intervention_kind") == "runtime_registered_check_guidance":
                if intervention.get("verdict_protocol") == EXIT_STATUS_VERDICT_PROTOCOL:
                    verdict_instruction = (
                        "The command must return 0 only when supported, 1 only when "
                        "refuted, and another status when execution is inconclusive."
                    )
                else:
                    verdict_instruction = (
                        "The command must finish the actual check, exit successfully, and "
                        f"print exactly {intervention['supported_output']!r} when supported "
                        f"or {intervention['refuted_output']!r} when refuted."
                    )
                notices.append(
                    f"The public condition still lacks current evidence: "
                    f"{intervention['proposition']}. Run this exact check now: "
                    f"{json.dumps(intervention['check_arguments'])}. {verdict_instruction} "
                    "Do not treat this "
                    "request as evidence. If the command cannot produce a declared "
                    "verdict, retain the condition as unknown and revise the check contract."
                )
                continue
            if intervention.get("intervention_kind") == "runtime_registered_check_execution":
                state = intervention.get("evidence_state")
                proposition = intervention.get("proposition")
                if proposition in resolved_repair_predicates:
                    # The later event is the current state; keep the stale verdict in logs.
                    continue
                if intervention.get("status") != "completed":
                    notices.append(
                        "Runtime could not complete the registered public check for "
                        f"{proposition}: {intervention.get('reason')}. Retain the condition "
                        "as unknown and choose another bounded check or repair path."
                    )
                elif state == EvidenceState.SUPPORTED.value:
                    notices.append(
                        "Runtime executed the registered public check and supported: "
                        f"{proposition}. This verifies only this condition; it does not show "
                        "that the task is complete."
                    )
                elif state == EvidenceState.REFUTED.value:
                    notices.append(
                        "Runtime executed the registered public check and refuted: "
                        f"{proposition}. Blocking fact: "
                        f"{json.dumps(intervention.get('blocking_fact'))}. Repair "
                        "this prerequisite while preserving completed work, then rerun "
                        "the same check before claiming completion."
                    )
                else:
                    notices.append(
                        "Runtime executed the registered public check for "
                        f"{proposition}, but the result was inconclusive. Blocking fact: "
                        f"{json.dumps(intervention.get('blocking_fact'))}. Inspect the raw "
                        "check output and revise the check or plan without treating the "
                        "predicate as true."
                    )
                continue
            if intervention.get("intervention_kind") == "runtime_targeted_repair_execution":
                if (
                    intervention.get("candidate_form") == "actor_guidance"
                    and intervention.get("effect_status") == "awaiting_recheck"
                    and intervention.get("status") == "completed"
                ):
                    notices.append(
                        "An isolated copy of the failed state passed the registered public "
                        "checks after applying this frozen guidance. The live environment is "
                        "unchanged. Apply the same guidance now, then rerun the registered "
                        "public checks. Frozen guidance:\n"
                        f"{intervention['guidance_text']}"
                    )
                    continue
                if (
                    intervention.get("candidate_form") == "actor_guidance"
                    and intervention.get("status") != "completed"
                ):
                    notices.append(
                        "Isolated trial of the frozen Actor guidance was not admitted: "
                        f"{intervention.get('reason')}. The live environment was not changed."
                    )
                    continue
                state = intervention.get("evidence_state")
                proposition = intervention.get("proposition")
                if state == EvidenceState.SUPPORTED.value:
                    notices.append(
                        "Runtime executed the configured repair and its registered check "
                        f"supported: {proposition}. This verifies only this condition; it does "
                        "not show that the task is complete."
                    )
                elif state == EvidenceState.REFUTED.value:
                    notices.append(
                        "Runtime executed the configured repair, but its registered check "
                        f"still refuted: {proposition}. The incident remains open."
                    )
                else:
                    notices.append(
                        "Runtime could not verify the configured repair for "
                        f"{proposition}: {intervention.get('reason')}. Treat its effect as "
                        "unknown and keep the incident unresolved."
                    )
                continue
            resource_id = intervention["resource_id"]
            if intervention.get("intervention_kind") == "runtime_guidance":
                output_paths = ", ".join(intervention["output_resource_ids"])
                requirement = (
                    "at least one of these outputs"
                    if intervention["output_requirement"] == "any"
                    else "all of these outputs"
                )
                notices.append(
                    "Runtime supervisor observed "
                    f"{intervention['no_progress_actor_actions']} Actor actions without "
                    "a change to the declared public outputs. Produce "
                    f"{requirement}: {output_paths}. Then validate the result using "
                    "publicly available checks."
                )
                continue
            if intervention.get("intervention_kind") == "runtime_failure_diagnosis_guidance":
                diagnosis_context = intervention.get("failure_diagnosis_context")
                if isinstance(diagnosis_context, dict):
                    failed = diagnosis_context.get("failed_transition")
                    tasks = diagnosis_context.get("candidate_tasks")
                    resources = diagnosis_context.get("resource_facts")
                    if (
                        not isinstance(failed, dict)
                        or not isinstance(tasks, list)
                        or not isinstance(resources, list)
                    ):
                        raise RuntimeError("TraceDP failure diagnosis context is malformed")
                    task_lines = [
                        "- {task_id} (parent={parent_id}, status={status}, source={source}); "
                        "confirmed inputs={confirmed}; hypothesized inputs={hypothesized}; "
                        "required outputs={outputs}".format(
                            task_id=row["task_id"],
                            parent_id=row["parent_id"],
                            status=row["status"],
                            source=row["source"],
                            confirmed=row["confirmed_prerequisite_resource_ids"],
                            hypothesized=row["hypothesized_prerequisite_resource_ids"],
                            outputs=row["required_output_resource_ids"],
                        )
                        for row in tasks
                    ]
                    resource_lines = [
                        "- {resource_id}: version={version}, observed={observed}, "
                        "exists={exists}, kind={kind}".format(**row)
                        for row in resources
                    ]
                    notices.append(
                        "TraceDP residual-state diagnosis follows the observed failed "
                        f"transition {failed['action_id']} (tool={failed['tool_name']}, "
                        f"phase={failed['phase']}, reason={failed['reason']}). No managed "
                        "resource version changed at this boundary, so the failure refutes "
                        "that execution attempt, not the public task.\nOpen task graph:\n"
                        + "\n".join(task_lines)
                        + "\nCurrent resource facts:\n"
                        + "\n".join(resource_lines)
                        + "\nUse this versioned graph to choose a different next check or "
                        "fix. Treat hypothesized inputs and absent observations as uncertain; "
                        "do not claim a root cause without new evidence."
                    )
                    continue
                notices.append(
                    "Runtime supervisor observed "
                    f"{intervention['failure_count_since_resource_change']} failed "
                    "Actor tool executions since the last managed resource change. "
                    "Reassess the current approach before repeating it. Use only public "
                    "task instructions, current tool results, and available environment "
                    "capabilities to identify which assumption or validation path failed; "
                    "choose a behaviorally different next check or fix. Do not claim the "
                    "task requirement is false merely because a tool is unavailable."
                )
                continue
            if intervention.get("intervention_kind") == "runtime_semantic_guidance":
                output_paths = ", ".join(intervention["output_resource_ids"])
                if intervention.get("policy") == TRACEDP_RESIDUAL_RUNTIME_V3_METHOD:
                    notices.append(
                        "Runtime supervisor observed a declared output before official "
                        f"evaluation: {output_paths}. Re-read the public task and verify "
                        "the output through its required external consumption path. "
                        "Existence alone does not prove completion. For a package or "
                        "directory, check the required install, import, build, invocation, "
                        "and behavior from outside its source tree as applicable. For a "
                        "file or data artifact, check its public schema, boundaries, "
                        "serialization, and behavior as applicable. Use a behaviorally "
                        "distinct check, complete any missing delivery step, and correct "
                        "the output when public evidence supports a change."
                    )
                    continue
                notice = (
                    "Runtime supervisor observed a complete declared output before "
                    "official evaluation. Perform a behaviorally distinct public "
                    "semantic check of: "
                    f"{output_paths}. Inspect selection, aggregation units, boundaries "
                    "between fields or records, library defaults, and serialization. "
                    "Repeating the same calculation is not an independent check. "
                )
                if intervention.get("policy") in OUTPUT_SEMANTIC_RUNTIME_METHODS:
                    notice += (
                        "When plausible interpretations give different results, "
                        "enumerate them and cite public task, schema, or documentation "
                        "evidence for the chosen interpretation. Do not keep the current "
                        "value merely because it was written first. Preserve independently "
                        "declared field and record boundaries unless a public source "
                        "explicitly defines a joined serialization. Correct the output "
                        "when the evidence supports a different result."
                    )
                else:
                    notice += "Correct the output if the independent result differs."
                notices.append(notice)
                continue
            if intervention.get("intervention_kind") == "runtime_convergence_guidance":
                output_paths = ", ".join(intervention["output_resource_ids"])
                notices.append(
                    "Runtime supervisor observed that the declared output exists and "
                    "the Actor session entered its finalization reserve. Perform only "
                    "a quick public check if it can finish immediately, then stop "
                    f"calling tools and return the final response. Output: {output_paths}."
                )
                continue
            if intervention["status"] == "completed":
                notices.append(
                    "Runtime supervisor restored protected prerequisite "
                    f"{resource_id} in the current environment."
                )
            else:
                notices.append(
                    "Runtime supervisor left the current state unchanged because it "
                    f"could not safely restore {resource_id}: {intervention['reason']}"
                )
        if not notices:
            return ""
        return "\n\n[Runtime supervisor]\n" + "\n".join(notices)

    async def execute(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> DelegatedToolResult:
        """Serialize one tool call and bind fresh observations to its actual execution."""

        if not tool_call_id:
            raise ValueError("tool call ID is required")
        if self._lock.locked():
            return self.record_parallel_rejection(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
            )

        async with self._lock:
            call_started = time.monotonic()
            sequence = self._sequence
            self._sequence += 1
            # A persisted check outcome must finish its replan branch before the
            # next Actor action can observe or mutate the environment.
            resumed_interventions = await self._resume_runtime_option_chain(
                sequence=sequence
            )
            early_snapshot_started = time.monotonic()
            early_snapshot = (
                asyncio.ensure_future(self._snapshot(sequence))
                if self.concurrent_runtime_observation and self.runtime_kernel is not None
                else None
            )
            if self.runtime_kernel is not None:
                self.runtime_kernel.propose_action(
                    action_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                )
                try:
                    runtime_before_ms = await self._refresh_runtime(source="pre_action_observation")
                except Exception as error:
                    await _discard_task(early_snapshot)
                    self.runtime_kernel.block_action(
                        tool_call_id,
                        reason="resource observation failed before execution",
                    )
                    self._append_observer(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        before=(),
                        after=(),
                        is_error=True,
                        environment_step_executed=False,
                        error_stage="resource_before_snapshot",
                        error_type=type(error).__name__,
                        error_message=str(error),
                        latency_ms={"total": round((time.monotonic() - call_started) * 1000)},
                    )
                    raise
            else:
                runtime_before_ms = 0
            before_snapshot_started = (
                early_snapshot_started if early_snapshot is not None else time.monotonic()
            )
            try:
                # The gateway lock cannot prevent external changes since the last tool.
                before = await (early_snapshot if early_snapshot is not None else self._snapshot(sequence))
                before_snapshot_ms = round((time.monotonic() - before_snapshot_started) * 1000)
            except Exception as error:
                if self.runtime_kernel is not None:
                    self.runtime_kernel.block_action(
                        tool_call_id,
                        reason="public observation failed before execution",
                    )
                self._append_observer(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    before=(),
                    after=(),
                    is_error=True,
                    environment_step_executed=False,
                    error_stage="before_snapshot",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    latency_ms={
                        "before_snapshot": round(
                            (time.monotonic() - before_snapshot_started) * 1000
                        ),
                        "total": round((time.monotonic() - call_started) * 1000),
                    },
                )
                raise
            preflight_error = self._preflight_error(
                tool_name,
                arguments,
                tool_call_id=tool_call_id,
            )
            if preflight_error is not None:
                if self.runtime_kernel is not None:
                    self.runtime_kernel.block_action(
                        tool_call_id,
                        reason=preflight_error.content,
                    )
                self._append_observer(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    before=before,
                    after=before,
                    is_error=True,
                    environment_step_executed=False,
                    latency_ms={
                        "before_snapshot": before_snapshot_ms,
                        "tool_execution": 0,
                        "after_snapshot": 0,
                        "total": round((time.monotonic() - call_started) * 1000),
                    },
                    tool_result=preflight_error,
                )
                return preflight_error
            if self.runtime_kernel is not None:
                self.runtime_kernel.start_action(tool_call_id)
            tool_started = time.monotonic()
            try:
                if tool_name == "register_runtime_check":
                    contract = self._parse_actor_check(arguments)
                    self.check_contracts += (contract,)
                    result = DelegatedToolResult(
                        "Registered an Actor-inferred check, not task success. Runtime may "
                        "execute this exact command through the shared environment adapter: "
                        + json.dumps(dict(contract.arguments)), False, False,
                        {
                            "check_provenance": "actor_inferred",
                            "source_excerpt": arguments["source_excerpt"],
                            "verdict_protocol": contract.verdict_protocol,
                        },
                    )
                elif tool_name == "complete_runtime_check_registration":
                    result = DelegatedToolResult(
                        "Completed the public-task check-registration phase. This records "
                        "an Actor judgment, not task evidence or success.",
                        False,
                        False,
                        {
                            "check_registration_phase": "completed",
                            "status": arguments["status"],
                            "task_id": arguments["task_id"],
                        },
                    )
                elif tool_name == "bash":
                    result = await self._bash(arguments)
                elif tool_name == "read":
                    result = await self._read(arguments, sequence)
                elif tool_name == "write":
                    result = await self._write(arguments, sequence)
                else:
                    result = DelegatedToolResult(
                        f"unsupported terminal tool: {tool_name}", True, False, {}
                    )
                tool_execution_ms = round((time.monotonic() - tool_started) * 1000)
            except asyncio.CancelledError as error:
                if self.runtime_kernel is not None:
                    observations, _ = await self._observe_runtime_after_uncertain_effect()
                    self.runtime_kernel.finish_action(
                        tool_call_id,
                        phase=ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT,
                        observations=observations,
                        reason=f"{type(error).__name__}: execution was cancelled",
                    )
                self._append_observer(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    before=before,
                    after=(),
                    is_error=True,
                    environment_step_executed=None,
                    error_stage="tool_execution",
                    error_type=type(error).__name__,
                    latency_ms={
                        "before_snapshot": before_snapshot_ms,
                        "tool_execution": round((time.monotonic() - tool_started) * 1000),
                        "total": round((time.monotonic() - call_started) * 1000),
                    },
                )
                raise
            except Exception as error:
                if self.runtime_kernel is not None:
                    (
                        observations,
                        observation_succeeded,
                    ) = await self._observe_runtime_after_uncertain_effect()
                    changed = any(
                        self.runtime_kernel.observations.get(item.resource_id) != item
                        for item in observations
                    )
                    failure_resource_change_observed = observation_succeeded and changed
                    self.runtime_kernel.finish_action(
                        tool_call_id,
                        phase=(
                            ActionPhase.FAILED_AFTER_PARTIAL_EFFECT
                            if failure_resource_change_observed
                            else ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
                        ),
                        observations=observations,
                        reason=f"{type(error).__name__}: execution receipt unavailable",
                    )
                    self._record_actor_failure_boundary(
                        action_id=tool_call_id,
                        resource_change_observed=failure_resource_change_observed,
                    )
                self._append_observer(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    before=before,
                    after=(),
                    is_error=True,
                    environment_step_executed=None,
                    error_stage="tool_execution",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    latency_ms={
                        "before_snapshot": before_snapshot_ms,
                        "tool_execution": round((time.monotonic() - tool_started) * 1000),
                        "total": round((time.monotonic() - call_started) * 1000),
                    },
                )
                raise
            early_observation = (
                asyncio.ensure_future(self.resource_driver.observe())
                if self.concurrent_runtime_observation
                and self.resource_driver is not None
                and self.runtime_kernel is not None
                else None
            )
            if result.environment_step_executed:
                after_snapshot_started = time.monotonic()
                post_observation_error_stage: str | None = None
                post_observation_error_type: str | None = None
                try:
                    after = await self._snapshot(sequence)
                    after_snapshot_ms = round((time.monotonic() - after_snapshot_started) * 1000)
                except Exception as error:
                    after = ()
                    after_snapshot_ms = round((time.monotonic() - after_snapshot_started) * 1000)
                    post_observation_error_stage = "after_snapshot"
                    post_observation_error_type = type(error).__name__
            else:
                after = before
                after_snapshot_ms = 0
                post_observation_error_stage = None
                post_observation_error_type = None
            try:
                runtime_transition, runtime_after_ms = await self._finish_runtime_action(
                    action_id=tool_call_id,
                    result=result,
                    observation=early_observation,
                )
            except Exception as error:
                if self.runtime_kernel is not None:
                    action = self.runtime_kernel.actions.get(tool_call_id)
                    if action is not None and action.phase is ActionPhase.STARTED:
                        observations = self.resource_driver.unknown_observations()  # type: ignore[union-attr]
                        transition = self.runtime_kernel.finish_action(
                            tool_call_id,
                            phase=(
                                ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
                                if result.is_error
                                else ActionPhase.COMPLETED
                            ),
                            observations=observations,
                            reason=(
                                "tool returned an error and resource state is unknown"
                                if result.is_error
                                else None
                            ),
                            execution_summary=_actor_execution_summary(result),
                        )
                        runtime_transition = {
                            "changed_resource_ids": list(transition.changed_resource_ids),
                            "stale_evidence_ids": list(transition.stale_evidence_ids),
                            "affected_task_ids": list(transition.affected_task_ids),
                            "action_phase": self.runtime_kernel.actions[tool_call_id].phase.value,
                        }
                        if result.is_error:
                            self._record_actor_failure_boundary(
                                action_id=tool_call_id,
                                resource_change_observed=False,
                            )
                    else:
                        # A bookkeeping failure may happen after finish_action commits;
                        # never issue a second terminal transition for that action.
                        runtime_transition = None
                    reason = f"{type(error).__name__}: {error}"
                    self._runtime_supervision_disabled_reason = reason
                    self.runtime_kernel.disable_runtime_supervision(reason=reason)
                runtime_after_ms = round((time.monotonic() - call_started) * 1000)
                post_observation_error_stage = (
                    "after_snapshot_and_resource_after_snapshot"
                    if post_observation_error_stage is not None
                    else "resource_after_snapshot"
                )
                post_observation_error_type = type(error).__name__
            if (
                tool_name == "register_runtime_check"
                and not result.is_error
                and self.runtime_kernel is not None
                and self.runtime_kernel.acceptance_predicate_enabled
            ):
                contract = self.check_contracts[-1]
                predicate_id = self.runtime_kernel.register_or_bind_actor_check_predicate(
                    registration_action_id=tool_call_id,
                    task_id=contract.task_id,
                    proposition=contract.proposition,
                    source_excerpt=str(arguments["source_excerpt"]),
                    resource_ids=contract.resource_ids,
                )
                check_id = f"actor-check:{tool_call_id}"
                self.check_contracts = (
                    *self.check_contracts[:-1],
                    contract.with_predicate_binding(
                        check_id=check_id,
                        predicate_id=predicate_id,
                    ),
                )
                result = DelegatedToolResult(
                    result.content,
                    result.is_error,
                    result.environment_step_executed,
                    {
                        **result.details,
                        "check_id": check_id,
                        "predicate_id": predicate_id,
                    },
                )
            if (
                tool_name == "complete_runtime_check_registration"
                and not result.is_error
                and self.runtime_kernel is not None
            ):
                self._check_registration_phase_completion = self.runtime_kernel.actions[
                    tool_call_id
                ]
            if self.runtime_kernel is not None and self._output_resource_ids:
                changed_outputs = set(
                    runtime_transition.get("changed_resource_ids", ())
                    if runtime_transition is not None
                    else ()
                ).intersection(self._output_resource_ids)
                self._actor_actions_since_output_progress = (
                    0 if changed_outputs else self._actor_actions_since_output_progress + 1
                )
            # Admit check outcomes before supervision can change their state scope.
            if self.check_contracts and post_observation_error_stage is None:
                assert self.runtime_kernel is not None
                for index, contract in enumerate(self.check_contracts):
                    if tool_name != "bash" or dict(contract.arguments) != dict(arguments):
                        continue
                    record_runtime_check(
                        self.runtime_kernel, contract,
                        {
                            "tool_call_id": tool_call_id, "tool_name": tool_name,
                            "input": dict(arguments),
                            "environment_step_executed": result.environment_step_executed,
                            "pre_supervision_content_chars": len(result.content),
                            "tool_result": {
                                "content": result.content, "is_error": result.is_error,
                                "details": {
                                    **result.details,
                                    "runtime_state": self.runtime_kernel.public_projection(),
                                },
                            },
                        },
                        evidence_id=f"runtime-check:{tool_call_id}:{index}",
                    )
            self._record_pending_intervention_effects(evidence_action_id=tool_call_id)
            intervention_started = time.monotonic()
            if post_observation_error_stage in {
                "resource_after_snapshot",
                "after_snapshot_and_resource_after_snapshot",
            }:
                runtime_interventions = [
                    {
                        "action_id": None,
                        "resource_id": None,
                        "policy": (self._runtime_policy_id),
                        "status": "supervisor_disabled",
                        "reason": reason,
                    }
                ]
            else:
                try:
                    runtime_interventions = await self._apply_runtime_supervision(
                        sequence=sequence,
                        trigger_action_id=tool_call_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # The actor receipt is already durable. Disable the broken
                    # supervisor without rewriting a successful tool execution.
                    reason = f"{type(error).__name__}: {error}"
                    self._runtime_supervision_disabled_reason = reason
                    assert self.runtime_kernel is not None
                    self.runtime_kernel.disable_runtime_supervision(reason=reason)
                    runtime_interventions = [
                        {
                            "action_id": None,
                            "resource_id": None,
                            "policy": (self._runtime_policy_id),
                            "status": "supervisor_disabled",
                            "reason": reason,
                        }
                    ]
            runtime_interventions = [
                *resumed_interventions,
                *runtime_interventions,
            ]
            runtime_intervention_ms = round((time.monotonic() - intervention_started) * 1000)
            returned_result = result
            if runtime_transition is not None and self.runtime_kernel is not None:
                returned_result = DelegatedToolResult(
                    content=result.content + self._runtime_notice(runtime_interventions),
                    is_error=result.is_error,
                    environment_step_executed=result.environment_step_executed,
                    details={
                        **result.details,
                        "runtime_transition": runtime_transition,
                        "runtime_interventions": runtime_interventions,
                        "runtime_state": self.runtime_kernel.public_projection(),
                    },
                )
            # Persist the exact value returned through the Pi tool boundary.
            self._append_observer(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                before=before,
                after=after,
                is_error=result.is_error,
                environment_step_executed=result.environment_step_executed,
                error_stage=post_observation_error_stage,
                error_type=post_observation_error_type,
                latency_ms={
                    "before_snapshot": before_snapshot_ms,
                    "tool_execution": tool_execution_ms,
                    "after_snapshot": after_snapshot_ms,
                    **(
                        {
                            "runtime_before_observation": runtime_before_ms,
                            "runtime_after_observation": runtime_after_ms,
                            "runtime_intervention": runtime_intervention_ms,
                        }
                        if self.runtime_kernel is not None
                        else {}
                    ),
                    "total": round((time.monotonic() - call_started) * 1000),
                },
                tool_result=returned_result,
                pre_supervision_content=result.content,
            )
            return returned_result

    async def _restore_resource_unlocked(
        self,
        *,
        action_id: str,
        resource_id: str,
    ) -> None:
        """Apply one repair while the caller owns the gateway execution lock."""

        if self.resource_driver is None or self.runtime_kernel is None:
            raise RuntimeError("managed resource restore is not configured")
        self.runtime_kernel.propose_action(
            action_id=action_id,
            tool_name="restore_resource",
            arguments={"resource_id": resource_id},
            actor="runtime",
        )
        try:
            await self._refresh_runtime(source="pre_restore_observation")
        except asyncio.CancelledError:
            self.runtime_kernel.block_action(
                action_id,
                reason="restore was cancelled before environment execution",
            )
            raise
        except Exception:
            # No restore reached the environment, so close the proposal as blocked.
            self.runtime_kernel.block_action(
                action_id,
                reason="resource observation failed before restore execution",
            )
            raise
        self.runtime_kernel.start_action(action_id)
        try:
            await self.resource_driver.restore_missing_prerequisite(resource_id)
            observations = await self.resource_driver.observe()
        except asyncio.CancelledError:
            observations, _ = await self._observe_runtime_after_uncertain_effect()
            self.runtime_kernel.finish_action(
                action_id,
                phase=ActionPhase.CANCELLED_WITH_UNKNOWN_EFFECT,
                observations=observations,
                reason="runtime repair was cancelled with unknown effect",
            )
            raise
        except Exception as error:
            (
                observations,
                observation_succeeded,
            ) = await self._observe_runtime_after_uncertain_effect()
            changed = any(
                self.runtime_kernel.observations.get(item.resource_id) != item
                for item in observations
            )
            self.runtime_kernel.finish_action(
                action_id,
                phase=(
                    ActionPhase.FAILED_AFTER_PARTIAL_EFFECT
                    if observation_succeeded and changed
                    else ActionPhase.FAILED_WITH_UNKNOWN_EFFECT
                ),
                observations=observations,
                reason=f"{type(error).__name__}: resource restore failed",
            )
            raise
        self.runtime_kernel.finish_action(
            action_id,
            phase=ActionPhase.COMPLETED,
            observations=observations,
        )

    async def restore_resource(
        self,
        *,
        action_id: str,
        resource_id: str,
    ) -> None:
        """Apply one explicit runtime repair inside this live gateway session."""

        async with self._lock:
            await self._restore_resource_unlocked(
                action_id=action_id,
                resource_id=resource_id,
            )

    def record_parallel_rejection(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        source_text: str | None = None,
    ) -> DelegatedToolResult:
        """Reproduce one source-observed refusal without environment access.

        With ``source_text_refusals`` on, the source's own result text is returned, so the
        Actor reads what it read in the source. Otherwise the parallel-rejection text is used.
        """
        from_source = bool(self.source_text_refusals and source_text)
        reason = source_text if from_source else "parallel terminal tool calls are not supported"
        result = DelegatedToolResult(
            reason,
            True,
            False,
            {"text_source": "source" if from_source else "replay"},
        )
        if self.runtime_kernel is not None:
            self.runtime_kernel.record_blocked_proposal(
                action_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                reason="parallel terminal tool calls are not supported",
            )
        self._append_observer(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            before=(),
            after=(),
            is_error=True,
            environment_step_executed=False,
            overlapping_tool_calls=True,
            error_stage="parallel_rejection",
            tool_result=result,
        )
        return result


class TerminalToolRpcServer:
    """One-request-per-connection Unix socket for the host Pi extension."""

    def __init__(
        self,
        *,
        socket_path: Path,
        gateway: TerminalEnvironmentToolGateway,
    ) -> None:
        self.socket_path = socket_path
        self.gateway = gateway
        self._server: asyncio.AbstractServer | None = None
        self._handlers: set[asyncio.Task[object]] = set()

    async def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle,
            path=str(self.socket_path),
        )

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        current = asyncio.current_task()
        handlers = tuple(task for task in self._handlers if task is not current)
        for task in handlers:
            task.cancel()
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)
        self.socket_path.unlink(missing_ok=True)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Track each in-flight request so shutdown waits for cancellation cleanup."""

        current = asyncio.current_task()
        if current is not None:
            self._handlers.add(current)
        try:
            try:
                row = json.loads((await reader.readline()).decode())
                result = await self.gateway.execute(
                    tool_call_id=str(row["tool_call_id"]),
                    tool_name=str(row["tool_name"]),
                    arguments=row["arguments"],
                )
                response = {
                    "ok": True,
                    "content": result.content,
                    "is_error": result.is_error,
                    "details": result.details,
                }
            except Exception as error:
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
            try:
                await writer.drain()
            except ConnectionError:
                # The host Pi can time out while the environment finishes a
                # side-effecting call; the durable gateway receipt remains authoritative.
                pass
        finally:
            try:
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass
            finally:
                if current is not None:
                    self._handlers.discard(current)
