"""Runner-side lifecycle for isolated generation and method processes."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
from collections.abc import Mapping
from multiprocessing.connection import Connection
from typing import Any

from lh_platform.core._generation_protocol import (
    GENERATION_AUTHORITY_SCHEMA,
    AuthorityModelProxy,
    GenerationAuthorityError,
    _action_from_wire,
    _safe_mapping,
    _wire_recv,
    _wire_send,
)
from lh_platform.core._generation_workers import _authority_worker, _method_worker
from lh_platform.core.method_capabilities import MethodCapabilities
from lh_platform.core.run_admission import StrictMethodLaunch
from lh_platform.core.types import Action, Observation, Task, Trajectory

logger = logging.getLogger(__name__)


class IsolatedMethodProxy:
    """Runner-side proxy for a method executed in a separate process.

    The proxy accepts and returns only JSON data.  In particular, Runner never
    unpickles a method-controlled ``Action`` object; it validates the returned
    JSON action before attaching Runner-owned model-generation evidence.
    """

    def __init__(
        self,
        launch: StrictMethodLaunch,
        model_channel: Connection,
        *,
        startup_timeout_s: float = 30.0,
        request_timeout_s: float = 300.0,
    ) -> None:
        self.method_id = str(launch.method_id)
        self._launch = launch
        self._model_channel = model_channel
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self._ctx = mp.get_context("spawn")
        self._command_parent: Connection | None = None
        self._process: mp.Process | None = None
        self._started = False
        self._broken = False
        self._runtime_capabilities: MethodCapabilities | None = None

    @property
    def runtime_capabilities(self) -> MethodCapabilities | None:
        return self._runtime_capabilities

    @property
    def broken(self) -> bool:
        """Whether a transport timeout/EOF makes this worker non-reusable."""

        return self._broken

    def _mark_broken(self) -> None:
        self._broken = True

    def start(self) -> None:
        if self._started:
            return
        command_parent, command_child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_method_worker,
            args=(
                command_child,
                self._model_channel,
                self._launch,
            ),
            name="lh-isolated-method",
            daemon=True,
        )
        try:
            process.start()
            command_child.close()
            # The method process is now the only owner of this client endpoint.
            self._model_channel.close()
            if not command_parent.poll(self.startup_timeout_s):
                raise GenerationAuthorityError("isolated method startup timed out")
            response = _wire_recv(command_parent)
            if response.get("ok") is not True:
                raise GenerationAuthorityError(
                    "isolated method failed to start: "
                    + str(response.get("error") or "unknown error")
                )
            runtime_capabilities = response.get("runtime_capabilities")
            if not isinstance(runtime_capabilities, Mapping):
                raise GenerationAuthorityError(
                    "isolated method omitted runtime capability observation"
                )
            self._runtime_capabilities = MethodCapabilities.from_mapping(runtime_capabilities)
            self._command_parent = command_parent
            self._process = process
            self._started = True
            self._broken = False
        except Exception:
            command_parent.close()
            try:
                command_child.close()
            except OSError:
                pass
            try:
                self._model_channel.close()
            except OSError:
                pass
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            try:
                process.close()
            except (OSError, ValueError):
                pass
            raise

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self._started or self._command_parent is None:
            raise GenerationAuthorityError("isolated method is not started")
        if self._broken:
            raise GenerationAuthorityError("isolated method worker is unavailable")
        _wire_send(self._command_parent, dict(payload))
        if not self._command_parent.poll(self.request_timeout_s):
            self._mark_broken()
            raise GenerationAuthorityError("isolated method request timed out")
        try:
            response = _wire_recv(self._command_parent)
        except (EOFError, OSError, GenerationAuthorityError):
            self._mark_broken()
            raise
        if response.get("ok") is not True:
            error_type = str(response.get("error_type") or "GenerationAuthorityError")
            detail = str(response.get("error") or "unknown method error")
            raise GenerationAuthorityError(f"{error_type}: {detail}")
        return response

    def reset(self, task: Task) -> None:
        self._request({"op": "reset", "task": task.to_dict()})

    def setup(self, task: Task) -> None:
        self._request({"op": "setup", "task": task.to_dict()})

    def predict(self, task: Task, trajectory: Trajectory) -> Action:
        response = self._request(
            {
                "op": "predict",
                "task": task.to_dict(),
                "trajectory": trajectory.to_dict(),
            }
        )
        return _action_from_wire(response.get("action"))

    def update(
        self,
        trajectory: Trajectory,
        action: Action,
        observation: Observation,
    ) -> None:
        self._request(
            {
                "op": "update",
                "trajectory": trajectory.to_dict(),
                "action": action.to_dict(),
                "observation": observation.to_dict(),
            }
        )

    def close(self) -> None:
        command = self._command_parent
        process = self._process
        try:
            if self._started and command is not None:
                try:
                    _wire_send(command, {"op": "shutdown"})
                    if command.poll(min(self.request_timeout_s, 3.0)):
                        _wire_recv(command)
                except Exception:
                    logger.exception("generation: IsolatedMethodProxy shutdown handshake failed")
        finally:
            if command is not None:
                try:
                    command.close()
                except OSError:
                    pass
            if process is not None:
                process.join(timeout=3.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3.0)
                # ``join`` reaps the child but leaves the parent's sentinel
                # descriptor open. Explicitly close the Process handle so a
                # long matrix of strict local-model runs does not leak one
                # file descriptor per task.
                try:
                    process.close()
                except (OSError, ValueError):
                    pass
            self._command_parent = None
            self._process = None
            self._started = False
            self._broken = False


class GenerationAuthority:
    """Runner-owned process authority for one task's model generations."""

    def __init__(
        self,
        model_id: str,
        model_kwargs: Mapping[str, Any],
        *,
        startup_timeout_s: float = 30.0,
        request_timeout_s: float = 300.0,
        control_timeout_s: float = 5.0,
    ) -> None:
        self.model_id = str(model_id)
        self.model_kwargs = copy.deepcopy(dict(model_kwargs))
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.control_timeout_s = float(control_timeout_s)
        self._ctx = mp.get_context("spawn")
        self._method_parent: Connection | None = None
        self._control_parent: Connection | None = None
        self._process: mp.Process | None = None
        self._proxy: AuthorityModelProxy | None = None
        self._started = False
        self._broken = False
        self._method_channel_transferred = False

    def __enter__(self) -> GenerationAuthority:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def model_proxy(self) -> AuthorityModelProxy:
        if self._method_channel_transferred:
            raise GenerationAuthorityError(
                "generation authority method channel was transferred to an isolated worker"
            )
        if self._proxy is None:
            raise GenerationAuthorityError("generation authority is not started")
        return self._proxy

    @property
    def broken(self) -> bool:
        """Whether a control/model transport failure makes this authority unsafe to reuse."""

        return self._broken

    def _mark_broken(self) -> None:
        self._broken = True

    def start(self) -> None:
        if self._started:
            return
        method_parent, method_child = self._ctx.Pipe(duplex=True)
        control_parent, control_child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_authority_worker,
            args=(method_child, control_child, self.model_id, self.model_kwargs),
            name="lh-generation-authority",
            daemon=True,
        )
        try:
            process.start()
            # The parent must not retain the child endpoints; otherwise an EOF
            # cannot prove the child exited.
            method_child.close()
            control_child.close()
            if not control_parent.poll(self.startup_timeout_s):
                raise GenerationAuthorityError("generation authority startup timed out")
            response = _wire_recv(control_parent)
            if not isinstance(response, Mapping) or response.get("ok") is not True:
                detail = (
                    response.get("error")
                    if isinstance(response, Mapping)
                    else "malformed startup response"
                )
                raise GenerationAuthorityError(f"generation authority failed to start: {detail}")
            self._method_parent = method_parent
            self._control_parent = control_parent
            self._process = process
            self._proxy = AuthorityModelProxy(
                method_parent,
                model_name=str(self.model_kwargs.get("model") or self.model_id),
                timeout_s=self.request_timeout_s,
                on_transport_failure=self._mark_broken,
            )
            self._started = True
            self._broken = False
        except Exception:
            method_parent.close()
            control_parent.close()
            try:
                method_child.close()
            except OSError:
                pass
            try:
                control_child.close()
            except OSError:
                pass
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            try:
                process.close()
            except (OSError, ValueError):
                pass
            raise

    def take_method_channel(self) -> Connection:
        """Transfer the generate-only endpoint to an isolated method worker.

        A strict Runner must call this exactly once before any action.  Once
        transferred, the controller cannot make model requests through the
        method channel; it retains only the independent control endpoint.
        """

        if not self._started or self._method_parent is None:
            raise GenerationAuthorityError("generation authority is not started")
        if self._method_channel_transferred:
            raise GenerationAuthorityError(
                "generation authority method channel already transferred"
            )
        channel = self._method_parent
        self._method_parent = None
        self._proxy = None
        self._method_channel_transferred = True
        return channel

    def _control(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self._started or self._control_parent is None:
            raise GenerationAuthorityError("generation authority is not started")
        if self._broken:
            raise GenerationAuthorityError(
                "generation authority is unavailable after transport failure"
            )
        _wire_send(self._control_parent, dict(payload))
        if not self._control_parent.poll(min(self.request_timeout_s, self.control_timeout_s)):
            self._mark_broken()
            raise GenerationAuthorityError("generation authority control request timed out")
        try:
            response = _wire_recv(self._control_parent)
        except (EOFError, OSError, GenerationAuthorityError):
            self._mark_broken()
            raise
        if not isinstance(response, Mapping):
            raise GenerationAuthorityError(
                "generation authority control returned malformed response"
            )
        if response.get("ok") is not True:
            error_type = str(response.get("error_type") or "GenerationAuthorityError")
            detail = str(response.get("error") or "unknown authority error")
            raise GenerationAuthorityError(f"{error_type}: {detail}")
        return copy.deepcopy(dict(response))

    def cursor(self) -> int:
        value = self._control({"op": "cursor"}).get("cursor")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GenerationAuthorityError("generation authority returned malformed cursor")
        return value

    def events_since(self, cursor: int) -> tuple[dict[str, Any], ...]:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise GenerationAuthorityError("generation authority cursor is malformed")
        raw_events = self._control({"op": "events_since", "cursor": cursor}).get("events")
        if not isinstance(raw_events, (list, tuple)):
            raise GenerationAuthorityError("generation authority returned malformed events")
        return tuple(
            copy.deepcopy(dict(event)) for event in raw_events if isinstance(event, Mapping)
        )

    def runtime_evidence(self) -> dict[str, Any]:
        runtime = self._control({"op": "runtime"}).get("runtime")
        evidence = _safe_mapping(runtime)
        evidence["generation_authority"] = {
            "schema": GENERATION_AUTHORITY_SCHEMA,
            "mode": "process_isolated",
            "model_channel": "generate_only",
            "method_execution": (
                "isolated_json_rpc"
                if self._method_channel_transferred
                else "direct_proxy_test_only"
            ),
        }
        return evidence

    def action_authority_evidence(self, *, cursor_start: int, cursor_end: int) -> dict[str, Any]:
        if (
            isinstance(cursor_start, bool)
            or not isinstance(cursor_start, int)
            or cursor_start < 0
            or isinstance(cursor_end, bool)
            or not isinstance(cursor_end, int)
            or cursor_end < cursor_start
        ):
            raise GenerationAuthorityError("generation authority action cursor window is malformed")
        return {
            "schema": GENERATION_AUTHORITY_SCHEMA,
            "mode": "process_isolated",
            "model_channel": "generate_only",
            "method_execution": (
                "isolated_json_rpc"
                if self._method_channel_transferred
                else "direct_proxy_test_only"
            ),
            "generation_cursor_start": cursor_start,
            "generation_cursor_end": cursor_end,
        }

    def close(self) -> None:
        control = self._control_parent
        process = self._process
        try:
            if self._started and control is not None and not self._broken:
                try:
                    self._control({"op": "shutdown"})
                except Exception:
                    logger.exception("generation: IsolatedGenerationAuthority shutdown failed")
        finally:
            for connection in (self._method_parent, self._control_parent):
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass
            if process is not None:
                process.join(timeout=3.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3.0)
                # See IsolatedMethodProxy.close(): a joined multiprocessing
                # Process still owns a local sentinel FD until close().
                try:
                    process.close()
                except (OSError, ValueError):
                    pass
            self._method_parent = None
            self._control_parent = None
            self._process = None
            self._proxy = None
            self._started = False
            self._broken = False
            self._method_channel_transferred = False
