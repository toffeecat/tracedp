"""Run reviewed Terminal repair candidates in a separate Harbor environment.

The main Gateway owns selection and live-state mutation.  This adapter only copies
the declared public resources into an already-started, distinct environment, runs
one configured repair plus its registered public checks there, and persists enough
phase evidence to resume without inventing an execution result.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from lh_platform.benchmarks.terminal_runtime_check import (
    RuntimeCheckContract,
    RuntimePredicateRepairContract,
    interpret_runtime_check_result,
    is_actor_guidance_repair_arguments,
)
from lh_platform.test_time_adaptation.portable_runtime_kernel import EvidenceState
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeConditionalFutureOption,
    RuntimeJointActionCandidate,
)

_POSIX_MODE = re.compile(r"[0-7]{3,4}")


def candidate_environment_session_id(main_session_id: str, trial_id: str) -> str:
    """Build a Compose-safe name for one disposable candidate environment.

    Harbor session and action identities may contain characters that Docker
    Compose rejects.  The mapping is injective on the sanitized alphabet so a
    resumed trial keeps the same project name.
    """

    token = f"{main_session_id}-ct-{trial_id}".lower()
    token = re.sub(r"[^a-z0-9_-]+", "-", token).strip("-")
    if not token or token[0] not in "abcdefghijklmnopqrstuvwxyz0123456789":
        token = f"c{token}"
    token = token[:101].rstrip("-")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,100}", token):
        raise ValueError("candidate session_id must be a valid unique compose name")
    return token


class TerminalCloneEnvironment(Protocol):
    """Expose the Harbor operations required for an exact filesystem transfer."""

    session_id: str

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any: ...

    async def is_file(self, path: str, user: str | int | None = None) -> bool: ...

    async def is_dir(self, path: str, user: str | int | None = None) -> bool: ...

    async def download_file(self, source_path: str, target_path: Path | str) -> None: ...

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None: ...

    async def upload_file(self, source_path: Path | str, target_path: str) -> None: ...

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None: ...

    async def stop(self, delete: bool) -> None: ...


class TerminalTrialBashResult(Protocol):
    """Expose the Gateway result fields consumed by the isolated trial."""

    content: str
    is_error: bool
    environment_step_executed: bool
    details: dict[str, Any]


TerminalTrialBashExecutor = Callable[
    [Mapping[str, Any], bool], Awaitable[TerminalTrialBashResult]
]
TerminalCloneEnvironmentFactory = Callable[[str], Awaitable[TerminalCloneEnvironment]]
TerminalTrialBashExecutorFactory = Callable[
    [TerminalCloneEnvironment], TerminalTrialBashExecutor
]


@dataclass(frozen=True)
class IsolatedGuidanceExecution:
    """Preserve one isolated Actor-guidance attempt without claiming live repair."""

    content: str
    is_error: bool
    environment_step_executed: bool
    details: dict[str, Any]


IsolatedGuidanceExecutor = Callable[
    [TerminalCloneEnvironment, Mapping[str, Any], int],
    Awaitable[IsolatedGuidanceExecution],
]


def isolated_actor_guidance_instruction(
    *, public_instruction: str, guidance_text: str
) -> str:
    """Give the isolated Actor the public task plus frozen guidance, nothing else."""

    if not public_instruction.strip() or not guidance_text.strip():
        raise ValueError("isolated actor guidance requires public instruction and frozen text")
    return (
        f"{public_instruction.rstrip()}\n\n"
        "This is an isolated copy of the failed live state. Apply the following frozen "
        "guidance, then verify the public postconditions. This copy is discarded if those "
        "postconditions fail. Do not assume the guidance is correct.\n"
        f"{guidance_text.strip()}\n"
    )


def isolated_guidance_execution(
    *,
    content: str,
    is_error: bool,
    environment_step_executed: bool,
    return_code: int | None,
    actor_tool_calls: int,
    actor_session_id: str,
) -> IsolatedGuidanceExecution:
    """Build the isolated Actor receipt consumed by the public-check gate."""

    if (
        isinstance(actor_tool_calls, bool)
        or not isinstance(actor_tool_calls, int)
        or actor_tool_calls < 0
        or (return_code is not None and type(return_code) is not int)
    ):
        raise ValueError("isolated actor guidance receipt is incomplete")
    return IsolatedGuidanceExecution(
        content=content,
        is_error=is_error,
        environment_step_executed=environment_step_executed,
        details={
            "return_code": return_code,
            "candidate_form": "actor_guidance",
            "actor_tool_calls": actor_tool_calls,
            "actor_session_id": actor_session_id,
        },
    )


@dataclass(frozen=True)
class TerminalTrialCheckObservation:
    """Preserve one isolated registered-check result and its interpretation."""

    check_id: str
    evidence_state: EvidenceState
    return_code: int | None
    output: str
    elapsed_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "evidence_state": self.evidence_state.value,
            "return_code": self.return_code,
            "output": self.output,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class TerminalIsolatedTrialReceipt:
    """Describe one isolated execution without claiming a live repair."""

    trial_id: str
    failure_state_id: str
    repair_id: str
    main_environment_session_id: str
    isolated_environment_session_id: str
    request_binding: Mapping[str, object]
    main_state_unchanged: bool
    repair_return_code: int | None
    repair_output: str
    observations: tuple[TerminalTrialCheckObservation, ...]
    verdict: EvidenceState
    transferred_bytes: int
    filesystem_transfer_calls: int
    environment_create_calls: int
    environment_delete_calls: int
    clone_cleanup_elapsed_ms: int
    environment_exec_calls: int
    total_elapsed_ms: int

    schema_version = "terminal_isolated_candidate_trial_receipt_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "implementation_status": "implementation_unverified",
            "trial_id": self.trial_id,
            "failure_state_id": self.failure_state_id,
            "repair_id": self.repair_id,
            "main_environment_session_id": self.main_environment_session_id,
            "isolated_environment_session_id": self.isolated_environment_session_id,
            "request_binding": dict(self.request_binding),
            "main_state_unchanged": self.main_state_unchanged,
            "repair_return_code": self.repair_return_code,
            "repair_output": self.repair_output,
            "observations": [item.to_dict() for item in self.observations],
            "verdict": self.verdict.value,
            "cost": {
                "transferred_bytes": self.transferred_bytes,
                "filesystem_transfer_calls": self.filesystem_transfer_calls,
                "environment_create_calls": self.environment_create_calls,
                "environment_delete_calls": self.environment_delete_calls,
                "clone_cleanup_elapsed_ms": self.clone_cleanup_elapsed_ms,
                "environment_exec_calls": self.environment_exec_calls,
                "total_elapsed_ms": self.total_elapsed_ms,
                "provider_calls": 0,
                "actor_tokens": 0,
            },
        }


def _combined_verdict(
    observations: tuple[TerminalTrialCheckObservation, ...],
) -> EvidenceState:
    """Require every public check and preserve unknown transport outcomes."""

    states = {item.evidence_state for item in observations}
    if EvidenceState.REFUTED in states:
        return EvidenceState.REFUTED
    if EvidenceState.UNKNOWN in states:
        return EvidenceState.UNKNOWN
    return EvidenceState.SUPPORTED


class TerminalSeparateEnvironmentPortfolioGate:
    """Clone public resources and trial a candidate in a distinct Harbor session."""

    protocol_id = "harbor_separate_environment_filesystem_clone_v1"

    def __init__(
        self,
        *,
        main_environment: TerminalCloneEnvironment,
        isolated_environment_factory: TerminalCloneEnvironmentFactory,
        public_paths: tuple[str, ...],
        durable_root: Path,
        selection_cost_units: int = 1,
        selection_time_limit_ms: int = 240_000,
    ) -> None:
        main_session = getattr(main_environment, "session_id", "")
        if not main_session:
            raise ValueError("candidate trials require a main Harbor session identity")
        if not public_paths or len(public_paths) != len(set(public_paths)):
            raise ValueError("candidate trials require unique public paths")
        if any(not path.startswith("/app/") for path in public_paths):
            raise ValueError("candidate trial resources must remain below /app")
        if (
            isinstance(selection_cost_units, bool)
            or not isinstance(selection_cost_units, int)
            or selection_cost_units < 1
            or isinstance(selection_time_limit_ms, bool)
            or not isinstance(selection_time_limit_ms, int)
            or selection_time_limit_ms < 1
        ):
            raise ValueError("candidate trial selection cost and time must be positive")
        self.main_environment = main_environment
        self.isolated_environment_factory = isolated_environment_factory
        self.public_paths = public_paths
        self.durable_root = durable_root
        self.selection_cost_units = selection_cost_units
        self.selection_time_limit_ms = selection_time_limit_ms
        self.event_log_path = durable_root / "terminal-candidate-trials.jsonl"
        self.transfer_root = durable_root / "candidate-transfer"
        self._events = self._load_events()

    def bind_planning_action(
        self,
        action: RuntimeJointActionCandidate | RuntimeConditionalFutureOption,
    ) -> RuntimeJointActionCandidate | RuntimeConditionalFutureOption:
        """Expose clone, trial, and live delivery as the action the selector values."""

        contract = action.execution_contract
        if contract is None:
            raise ValueError("candidate trial action lacks an execution contract")
        operation = contract.proposed_check_or_repair
        arguments = operation.get("arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("candidate trial action arguments are invalid")
        repair_arguments = arguments.get("repair_arguments")
        guidance = is_actor_guidance_repair_arguments(
            repair_arguments if isinstance(repair_arguments, Mapping) else {}
        )
        bound_contract = replace(
            contract,
            proposed_check_or_repair={
                "operation": (
                    "trial_in_separate_environment_then_deliver_frozen_guidance_if_admitted"
                    if guidance
                    else "trial_in_separate_environment_then_live_repair_and_recheck"
                ),
                "arguments": {
                    **dict(arguments),
                    "candidate_trial_protocol": self.protocol_id,
                },
            },
            time_limit_ms=contract.time_limit_ms + self.selection_time_limit_ms,
            delivery_only=True if guidance else contract.delivery_only,
            failure_branch=(
                "reject refuted or unknown isolated trial without main delivery; "
                "otherwise use the declared live repair failure branch"
            ),
        )
        return replace(
            action,
            cost_units=action.cost_units + self.selection_cost_units,
            transition_model_id=(
                "separate_environment_trial_then_frozen_guidance_delivery_v1"
                if guidance
                else "separate_environment_trial_then_configured_repair_recheck_v1"
            ),
            execution_contract=bound_contract,
        )

    def _load_events(self) -> list[dict[str, object]]:
        """Load only a contiguous durable prefix; malformed recovery must fail closed."""

        if not self.event_log_path.is_file():
            return []
        rows = [json.loads(line) for line in self.event_log_path.read_text().splitlines()]
        if any(
            not isinstance(row, dict) or row.get("sequence") != index
            for index, row in enumerate(rows)
        ):
            raise ValueError("candidate trial event history is not contiguous")
        return rows

    def _append(self, event_type: str, payload: Mapping[str, object]) -> None:
        """Append one phase boundary before any later irreversible operation."""

        self.durable_root.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": "terminal_candidate_trial_event_v1",
            "sequence": len(self._events),
            "event_type": event_type,
            "payload": dict(payload),
        }
        with self.event_log_path.open("a") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._events.append(row)

    async def _path_mode(
        self, environment: TerminalCloneEnvironment, path: str
    ) -> str | None:
        """Read one declared path's POSIX mode.

        The snapshot transfer goes through a volume that cannot store POSIX modes, so the
        mode must be recorded from the container and re-applied after every restore.
        """

        result = await environment.exec(
            f"stat -c %a -- {json.dumps(path)}", timeout_sec=30
        )
        mode = str(result.stdout or "").strip()
        if result.return_code != 0 or not _POSIX_MODE.fullmatch(mode):
            return None
        return mode

    async def _apply_mode(
        self,
        environment: TerminalCloneEnvironment,
        path: str,
        mode: object,
    ) -> None:
        """Restore one recorded mode after an upload created the path with default bits."""

        if mode is None:
            return
        if not isinstance(mode, str) or not _POSIX_MODE.fullmatch(mode):
            raise ValueError(f"candidate snapshot mode is invalid: {mode!r}")
        result = await environment.exec(
            f"chmod {mode} -- {json.dumps(path)}", timeout_sec=30
        )
        if result.return_code != 0:
            raise RuntimeError(f"cannot restore mode {mode} for {path}")

    async def _capture_environment(
        self,
        environment: TerminalCloneEnvironment,
        storage_id: str,
    ) -> tuple[dict[str, object], int, int]:
        """Copy the declared resource set from one concrete environment."""

        target = self.transfer_root / storage_id
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        manifest: dict[str, object] = {}
        transferred = 0
        transfer_calls = 0
        for index, path in enumerate(self.public_paths):
            local = target / f"resource-{index}"
            if await environment.is_file(path):
                await environment.download_file(path, local)
                transfer_calls += 1
                content = local.read_bytes()
                transferred += len(content)
                manifest[path] = {
                    "kind": "file",
                    "local": str(local),
                    "content": content,
                    "mode": await self._path_mode(environment, path),
                }
            elif await environment.is_dir(path):
                await environment.download_dir(path, local)
                transfer_calls += 1
                file_rows = []
                for child in sorted(item for item in local.rglob("*") if item.is_file()):
                    content = child.read_bytes()
                    transferred += len(content)
                    file_rows.append((str(child.relative_to(local)), content))
                manifest[path] = {
                    "kind": "directory",
                    "local": str(local),
                    "files": file_rows,
                    "mode": await self._path_mode(environment, path),
                }
            else:
                manifest[path] = {"kind": "missing", "local": str(local)}
        return manifest, transferred, transfer_calls

    async def _capture_main(
        self, trial_id: str
    ) -> tuple[dict[str, object], int, int]:
        """Copy the complete declared resource set to durable host storage."""

        return await self._capture_environment(self.main_environment, trial_id)

    async def _restore_isolated(
        self,
        manifest: Mapping[str, object],
        *,
        isolated_environment: TerminalCloneEnvironment,
    ) -> tuple[int, int, int]:
        """Replace every declared path in the separate environment from the snapshot."""

        calls = 0
        transfer_calls = 0
        uploaded_bytes = 0
        for path, raw in manifest.items():
            assert isinstance(raw, Mapping)
            result = await isolated_environment.exec(
                f"rm -rf -- {json.dumps(path)}", timeout_sec=120
            )
            calls += 1
            if result.return_code != 0:
                raise RuntimeError(f"cannot clear isolated resource: {path}")
            kind = raw["kind"]
            local = Path(str(raw["local"]))
            if kind == "file":
                await isolated_environment.upload_file(local, path)
                transfer_calls += 1
                uploaded_bytes += len(raw["content"])
                await self._apply_mode(isolated_environment, path, raw.get("mode"))
            elif kind == "directory":
                await isolated_environment.upload_dir(local, path)
                transfer_calls += 1
                uploaded_bytes += sum(len(content) for _, content in raw["files"])
                await self._apply_mode(isolated_environment, path, raw.get("mode"))
            elif kind != "missing":
                raise ValueError("candidate snapshot resource kind is invalid")
        return calls, transfer_calls, uploaded_bytes

    async def _main_matches(
        self, manifest: Mapping[str, object]
    ) -> tuple[bool, int, int]:
        """Compare exact bytes after the trial instead of trusting isolation metadata."""

        probe_id = f"main-readback-{len(self._events):06d}"
        observed, _, transfer_calls = await self._capture_main(probe_id)
        try:
            for path, expected_raw in manifest.items():
                actual_raw = observed[path]
                assert isinstance(expected_raw, Mapping) and isinstance(actual_raw, Mapping)
                if expected_raw["kind"] != actual_raw["kind"]:
                    return False, 0, transfer_calls
                if expected_raw["kind"] == "file" and expected_raw["content"] != actual_raw["content"]:
                    return False, 0, transfer_calls
                if expected_raw["kind"] == "directory" and expected_raw["files"] != actual_raw["files"]:
                    return False, 0, transfer_calls
                if expected_raw.get("mode") != actual_raw.get("mode"):
                    return False, 0, transfer_calls
            readback_bytes = sum(
                len(raw.get("content", b""))
                if raw["kind"] == "file"
                else sum(len(content) for _, content in raw.get("files", ()))
                if raw["kind"] == "directory"
                else 0
                for raw in observed.values()
                if isinstance(raw, Mapping)
            )
            return True, readback_bytes, transfer_calls
        finally:
            shutil.rmtree(self.transfer_root / probe_id, ignore_errors=True)

    async def _isolated_matches(
        self,
        environment: TerminalCloneEnvironment,
        manifest: Mapping[str, object],
        *,
        trial_id: str,
    ) -> tuple[bool, int, int]:
        """Verify clone readback before exposing the candidate to the session."""

        probe_id = f"{trial_id}-isolated-readback"
        observed, readback_bytes, transfer_calls = await self._capture_environment(
            environment, probe_id
        )
        try:
            for path, expected_raw in manifest.items():
                actual_raw = observed[path]
                assert isinstance(expected_raw, Mapping) and isinstance(
                    actual_raw, Mapping
                )
                if expected_raw["kind"] != actual_raw["kind"]:
                    return False, readback_bytes, transfer_calls
                if (
                    expected_raw["kind"] == "file"
                    and expected_raw["content"] != actual_raw["content"]
                ):
                    return False, readback_bytes, transfer_calls
                if (
                    expected_raw["kind"] == "directory"
                    and expected_raw["files"] != actual_raw["files"]
                ):
                    return False, readback_bytes, transfer_calls
                if expected_raw.get("mode") != actual_raw.get("mode"):
                    return False, readback_bytes, transfer_calls
            return True, readback_bytes, transfer_calls
        finally:
            shutil.rmtree(self.transfer_root / probe_id, ignore_errors=True)

    def _request_binding(
        self,
        *,
        failure_state_id: str,
        repair: RuntimePredicateRepairContract,
        checks: tuple[RuntimeCheckContract, ...],
    ) -> dict[str, object]:
        """Bind recovery to the complete candidate, check, resource, and session request."""

        return {
            "protocol_id": self.protocol_id,
            "failure_state_id": failure_state_id,
            "repair": {
                "repair_id": repair.repair_id,
                "task_id": repair.task_id,
                "proposition": repair.proposition,
                "arguments": dict(repair.arguments),
                "resource_ids": list(repair.resource_ids),
                "time_limit_seconds": repair.time_limit_seconds,
            },
            "checks": [
                {
                    "check_id": check.check_id,
                    "predicate_id": check.predicate_id,
                    "task_id": check.task_id,
                    "proposition": check.proposition,
                    "arguments": dict(check.arguments),
                    "supported_output": check.supported_output,
                    "refuted_output": check.refuted_output,
                    "resource_ids": list(check.resource_ids),
                    "verdict_protocol": check.verdict_protocol,
                }
                for check in checks
            ],
            "public_paths": list(self.public_paths),
            "main_environment_session_id": self.main_environment.session_id,
            "selection_cost_units": self.selection_cost_units,
            "selection_time_limit_ms": self.selection_time_limit_ms,
        }

    def completed_receipt(
        self,
        trial_id: str,
        *,
        expected_request_binding: Mapping[str, object],
    ) -> TerminalIsolatedTrialReceipt | None:
        """Recover only a fully persisted receipt; incomplete phases are never inferred."""

        rows = [
            row for row in self._events
            if row.get("event_type") == "trial_completed"
            and isinstance(row.get("payload"), dict)
            and row["payload"].get("trial_id") == trial_id
        ]
        if not rows:
            return None
        payload = rows[-1]["payload"]["receipt"]
        assert isinstance(payload, Mapping)
        request_binding = payload.get("request_binding")
        if request_binding != expected_request_binding:
            raise ValueError("completed candidate trial differs from resumed request")
        observations_raw = payload["observations"]
        assert isinstance(observations_raw, list)
        observations = tuple(
            TerminalTrialCheckObservation(
                check_id=str(item["check_id"]),
                evidence_state=EvidenceState(str(item["evidence_state"])),
                return_code=item["return_code"] if isinstance(item["return_code"], int) else None,
                output=str(item["output"]),
                elapsed_ms=int(item["elapsed_ms"]),
            )
            for item in observations_raw
        )
        cost = payload["cost"]
        assert isinstance(cost, Mapping)
        return TerminalIsolatedTrialReceipt(
            trial_id=str(payload["trial_id"]),
            failure_state_id=str(payload["failure_state_id"]),
            repair_id=str(payload["repair_id"]),
            main_environment_session_id=str(payload["main_environment_session_id"]),
            isolated_environment_session_id=str(payload["isolated_environment_session_id"]),
            request_binding=dict(expected_request_binding),
            main_state_unchanged=payload["main_state_unchanged"] is True,
            repair_return_code=(payload["repair_return_code"] if isinstance(payload["repair_return_code"], int) else None),
            repair_output=str(payload["repair_output"]),
            observations=observations,
            verdict=EvidenceState(str(payload["verdict"])),
            transferred_bytes=int(cost["transferred_bytes"]),
            filesystem_transfer_calls=int(cost["filesystem_transfer_calls"]),
            environment_create_calls=int(cost["environment_create_calls"]),
            environment_delete_calls=int(cost["environment_delete_calls"]),
            clone_cleanup_elapsed_ms=int(cost["clone_cleanup_elapsed_ms"]),
            environment_exec_calls=int(cost["environment_exec_calls"]),
            total_elapsed_ms=int(cost["total_elapsed_ms"]),
        )

    async def run_trial(
        self,
        *,
        trial_id: str,
        failure_state_id: str,
        repair: RuntimePredicateRepairContract,
        checks: tuple[RuntimeCheckContract, ...],
        execute_isolated_bash_factory: TerminalTrialBashExecutorFactory,
        execute_isolated_guidance: IsolatedGuidanceExecutor | None = None,
    ) -> TerminalIsolatedTrialReceipt:
        """Execute one candidate once; a durable completed receipt is replayed on resume."""

        request_binding = self._request_binding(
            failure_state_id=failure_state_id,
            repair=repair,
            checks=checks,
        )
        cached = self.completed_receipt(
            trial_id,
            expected_request_binding=request_binding,
        )
        if cached is not None:
            return cached
        incomplete = [
            row
            for row in self._events
            if isinstance(row.get("payload"), dict)
            and row["payload"].get("trial_id") == trial_id
        ]
        if incomplete:
            # Every pre-completion phase touches only the disposable environment.
            # Restoring the full public snapshot makes a retry safe and explicit.
            self._append(
                "incomplete_trial_restarted",
                {
                    "trial_id": trial_id,
                    "last_event_type": incomplete[-1]["event_type"],
                },
            )
        started = time.monotonic()
        self._append(
            "clone_started",
            {"trial_id": trial_id, "failure_state_id": failure_state_id, "repair_id": repair.repair_id},
        )
        manifest, transferred, transfer_calls = await self._capture_main(trial_id)
        cleanup_elapsed_ms = 0
        isolated_environment: TerminalCloneEnvironment | None = None
        try:
            attempt_index = sum(
                row.get("event_type") == "isolated_environment_created"
                and isinstance(row.get("payload"), dict)
                and row["payload"].get("trial_id") == trial_id
                for row in self._events
            )
            isolated_environment = await self.isolated_environment_factory(
                f"{trial_id}__attempt-{attempt_index}"
            )
            if (
                isolated_environment is self.main_environment
                or not isolated_environment.session_id
                or isolated_environment.session_id == self.main_environment.session_id
            ):
                raise RuntimeError(
                    "candidate trial factory did not create a distinct Harbor session"
                )
            self._append(
                "isolated_environment_created",
                {
                    "trial_id": trial_id,
                    "isolated_environment_session_id": isolated_environment.session_id,
                },
            )
            execute_isolated_bash = execute_isolated_bash_factory(
                isolated_environment
            )
            exec_calls, upload_calls, uploaded_bytes = await self._restore_isolated(
                manifest,
                isolated_environment=isolated_environment,
            )
            transfer_calls += upload_calls
            transferred += uploaded_bytes
            clone_matches, isolated_readback_bytes, isolated_readback_calls = (
                await self._isolated_matches(
                isolated_environment,
                manifest,
                trial_id=trial_id,
                )
            )
            transfer_calls += isolated_readback_calls
            transferred += isolated_readback_bytes
            if not clone_matches:
                raise RuntimeError(
                    "isolated environment does not match the declared failure state"
                )
            self._append(
                "clone_ready",
                {"trial_id": trial_id, "transferred_bytes": transferred},
            )
            self._append("candidate_started", {"trial_id": trial_id})
            guidance = is_actor_guidance_repair_arguments(repair.arguments)
            if guidance:
                if execute_isolated_guidance is None:
                    raise RuntimeError(
                        "actor guidance trial requires an isolated Actor runner"
                    )
                result = await execute_isolated_guidance(
                    isolated_environment,
                    dict(repair.arguments),
                    repair.time_limit_seconds,
                )
                actor_calls = result.details.get("actor_tool_calls")
                exec_calls += (
                    actor_calls
                    if type(actor_calls) is int and actor_calls > 0
                    else 1
                )
            else:
                result = await execute_isolated_bash(
                    {
                        **dict(repair.arguments),
                        "timeout": repair.time_limit_seconds,
                    },
                    False,
                )
                exec_calls += 1
            repair_output = result.content
            repair_return_code = result.details.get("return_code")
            if isinstance(repair_return_code, bool) or not isinstance(
                repair_return_code, int
            ):
                repair_return_code = None
            self._append(
                "candidate_finished",
                {
                    "trial_id": trial_id,
                    "return_code": repair_return_code,
                    "output": repair_output,
                },
            )
            observations = []
            if (
                result.environment_step_executed is True
                and result.is_error is False
                and repair_return_code == 0
            ):
                for check in checks:
                    check_started = time.monotonic()
                    checked = await execute_isolated_bash(
                        dict(check.arguments),
                        True,
                    )
                    exec_calls += 1
                    output = checked.content
                    observations.append(
                        TerminalTrialCheckObservation(
                            check_id=check.check_id or f"unbound:{len(observations)}",
                            evidence_state=interpret_runtime_check_result(
                                check,
                                content=output,
                                return_code=checked.details.get("return_code"),
                                is_error=checked.is_error,
                            ),
                            return_code=(
                                checked.details.get("return_code")
                                if type(checked.details.get("return_code")) is int
                                else None
                            ),
                            output=output,
                            elapsed_ms=round(
                                (time.monotonic() - check_started) * 1000
                            ),
                        )
                    )
            else:
                observations = [
                    TerminalTrialCheckObservation(
                        check_id=check.check_id or f"unbound:{index}",
                        evidence_state=EvidenceState.UNKNOWN,
                        return_code=None,
                        output="candidate failed before this check",
                        elapsed_ms=0,
                    )
                    for index, check in enumerate(checks)
                ]
            unchanged, main_readback_bytes, main_readback_calls = (
                await self._main_matches(manifest)
            )
            transfer_calls += main_readback_calls
            verdict = _combined_verdict(tuple(observations))
            if not unchanged:
                verdict = EvidenceState.UNKNOWN
        finally:
            cleanup_started = time.monotonic()
            if isolated_environment is not None:
                await isolated_environment.stop(delete=True)
                self._append(
                    "isolated_environment_deleted",
                    {
                        "trial_id": trial_id,
                        "isolated_environment_session_id": (
                            isolated_environment.session_id
                        ),
                    },
                )
            shutil.rmtree(self.transfer_root / trial_id, ignore_errors=True)
            cleanup_elapsed_ms = round((time.monotonic() - cleanup_started) * 1000)
        assert isolated_environment is not None
        receipt = TerminalIsolatedTrialReceipt(
            trial_id=trial_id,
            failure_state_id=failure_state_id,
            repair_id=repair.repair_id,
            main_environment_session_id=self.main_environment.session_id,
            isolated_environment_session_id=isolated_environment.session_id,
            request_binding=request_binding,
            main_state_unchanged=unchanged,
            repair_return_code=repair_return_code,
            repair_output=repair_output,
            observations=tuple(observations),
            verdict=verdict,
            transferred_bytes=transferred + main_readback_bytes,
            filesystem_transfer_calls=transfer_calls,
            environment_create_calls=1,
            environment_delete_calls=1,
            clone_cleanup_elapsed_ms=cleanup_elapsed_ms,
            environment_exec_calls=exec_calls,
            total_elapsed_ms=round((time.monotonic() - started) * 1000),
        )
        self._append("trial_completed", {"trial_id": trial_id, "receipt": receipt.to_dict()})
        return receipt


__all__ = [
    "candidate_environment_session_id",
    "isolated_actor_guidance_instruction",
    "isolated_guidance_execution",
    "IsolatedGuidanceExecution",
    "IsolatedGuidanceExecutor",
    "TerminalCloneEnvironment",
    "TerminalIsolatedTrialReceipt",
    "TerminalSeparateEnvironmentPortfolioGate",
    "TerminalTrialBashExecutor",
    "TerminalTrialBashResult",
    "TerminalTrialCheckObservation",
]
