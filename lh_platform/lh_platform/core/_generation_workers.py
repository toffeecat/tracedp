"""Child-process entry points for authoritative model and method execution."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from multiprocessing.connection import Connection, wait
from typing import Any

from lh_platform.core._generation_protocol import (
    AuthorityModelProxy,
    GenerationAuthorityError,
    _action_from_wire,
    _generation_cursor,
    _generation_events,
    _messages_from_wire,
    _observation_from_wire,
    _runtime_snapshot,
    _safe_mapping,
    _scrub_strict_worker_environment,
    _send_error,
    _serialize_output,
    _strict_method_constructor_kwargs,
    _task_from_wire,
    _trajectory_from_wire,
    _wire_recv,
    _wire_send,
)
from lh_platform.core.method_capabilities import MethodCapabilities
from lh_platform.core.registry import (
    METHODS,
    MODELS,
    bootstrap_method_registry,
    bootstrap_model_registry,
)
from lh_platform.core.run_admission import StrictMethodLaunch
from lh_platform.core.types import Action


def _authority_worker(
    method_channel: Connection,
    control_channel: Connection,
    model_id: str,
    model_kwargs: dict[str, Any],
) -> None:
    """Run model generation and the immutable ledger outside method code."""

    model: object | None = None
    try:
        # The authority is the only process that owns model-event evidence.
        # Importing method plugins here would let module-level adapter code
        # monkeypatch its helpers before the model is created.
        bootstrap_model_registry()
        model = MODELS.create(model_id, **dict(model_kwargs))
        _wire_send(
            control_channel,
            {
                "ok": True,
                "kind": "ready",
                "runtime": _runtime_snapshot(model),
            },
        )
    except Exception as error:
        _send_error(control_channel, error)
        method_channel.close()
        control_channel.close()
        return

    try:
        method_channel_open = True
        while True:
            channels = [control_channel]
            if method_channel_open:
                channels.insert(0, method_channel)
            ready = wait(channels)
            for channel in ready:
                try:
                    request = _wire_recv(channel)
                except EOFError:
                    if channel is method_channel:
                        # The isolated method process has ended.  The Runner
                        # still owns the control channel and must be able to
                        # read final event/runtime facts before shutdown.
                        method_channel_open = False
                        method_channel.close()
                        continue
                    return
                if not isinstance(request, Mapping):
                    _send_error(
                        channel,
                        GenerationAuthorityError("authority request is malformed"),
                    )
                    continue
                operation = request.get("op")
                if channel is method_channel:
                    # The untrusted method-facing channel intentionally accepts
                    # exactly one operation.  It cannot append or query ledger
                    # entries, even if an adapter reaches into proxy internals.
                    if operation != "generate":
                        _send_error(
                            method_channel,
                            GenerationAuthorityError("method channel only permits generate"),
                        )
                        continue
                    try:
                        messages = _messages_from_wire(request.get("messages"))
                        kwargs = _safe_mapping(request.get("kwargs"))
                        output = model.generate(messages, **kwargs)  # type: ignore[union-attr]
                        _wire_send(
                            method_channel,
                            {
                                "ok": True,
                                "output": _serialize_output(output),
                                # This cache is compatibility-only.  Runner
                                # obtains its own authoritative snapshot via
                                # the control channel below.
                                "runtime": _runtime_snapshot(model),
                            },
                        )
                    except Exception as error:  # model error remains evidence
                        _send_error(method_channel, error)
                    continue

                if operation == "cursor":
                    try:
                        _wire_send(
                            control_channel,
                            {"ok": True, "cursor": _generation_cursor(model)},
                        )
                    except Exception as error:
                        _send_error(control_channel, error)
                elif operation == "events_since":
                    try:
                        raw_cursor = request.get("cursor")
                        if isinstance(raw_cursor, bool) or not isinstance(raw_cursor, int):
                            raise GenerationAuthorityError("events_since cursor is malformed")
                        _wire_send(
                            control_channel,
                            {
                                "ok": True,
                                "events": _generation_events(model, raw_cursor),
                            },
                        )
                    except Exception as error:
                        _send_error(control_channel, error)
                elif operation == "runtime":
                    _wire_send(
                        control_channel,
                        {"ok": True, "runtime": _runtime_snapshot(model)},
                    )
                elif operation == "shutdown":
                    _wire_send(control_channel, {"ok": True, "shutdown": True})
                    return
                else:
                    _send_error(
                        control_channel,
                        GenerationAuthorityError("unknown authority control operation"),
                    )
    finally:
        method_channel.close()
        control_channel.close()


def _method_worker(
    command_channel: Connection,
    model_channel: Connection,
    launch: StrictMethodLaunch,
) -> None:
    """Run one method adapter outside Runner's provenance-control process."""

    try:
        # The worker imports exactly one declared adapter.  Its registry does
        # not load model implementations: the only permitted model access is
        # the generate-only authority proxy bound below.  The only method
        # options crossing this seam were explicitly allowlisted by the
        # Method Inventory when Run Admission built ``launch``.
        _scrub_strict_worker_environment()
        bootstrap_method_registry(launch.method_id)
        constructor_kwargs = _strict_method_constructor_kwargs(launch.constructor_kwargs_dict())
        method = METHODS.create(launch.method_id, **constructor_kwargs)
        if launch.seed is not None:
            # Match the non-strict Runner contract without assuming every
            # adapter constructor accepts ``seed``.
            method.seed = launch.seed
        binder = getattr(method, "bind_model_proxy", None)
        if not callable(binder):
            raise GenerationAuthorityError("isolated method must expose bind_model_proxy")
        binder(
            AuthorityModelProxy(
                model_channel,
                model_name="generation_authority_proxy",
            )
        )
        _wire_send(
            command_channel,
            {
                "ok": True,
                "kind": "ready",
                "runtime_capabilities": MethodCapabilities.observe(method).to_dict(),
            },
        )
    except Exception as error:
        _send_error(command_channel, error)
        command_channel.close()
        model_channel.close()
        return

    try:
        while True:
            try:
                request = _wire_recv(command_channel)
            except EOFError:
                return
            operation = request.get("op")
            try:
                if operation == "reset":
                    method.reset(copy.deepcopy(_task_from_wire(request.get("task"))))
                    _wire_send(command_channel, {"ok": True})
                elif operation == "setup":
                    method.setup(copy.deepcopy(_task_from_wire(request.get("task"))))
                    _wire_send(command_channel, {"ok": True})
                elif operation == "predict":
                    action = method.predict(
                        copy.deepcopy(_task_from_wire(request.get("task"))),
                        copy.deepcopy(_trajectory_from_wire(request.get("trajectory"))),
                    )
                    if not isinstance(action, Action):
                        raise GenerationAuthorityError("method.predict did not return Action")
                    _wire_send(
                        command_channel,
                        {"ok": True, "action": action.to_dict()},
                    )
                elif operation == "update":
                    method.update(
                        copy.deepcopy(_trajectory_from_wire(request.get("trajectory"))),
                        copy.deepcopy(_action_from_wire(request.get("action"))),
                        copy.deepcopy(_observation_from_wire(request.get("observation"))),
                    )
                    _wire_send(command_channel, {"ok": True})
                elif operation == "shutdown":
                    _wire_send(command_channel, {"ok": True, "shutdown": True})
                    return
                else:
                    raise GenerationAuthorityError("unknown isolated-method operation")
            except Exception as error:
                _send_error(command_channel, error)
    finally:
        command_channel.close()
        model_channel.close()
