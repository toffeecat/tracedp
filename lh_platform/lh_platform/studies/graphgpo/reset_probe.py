"""Subprocess entrypoint for a live official GraphGPO ALFWorld reset probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import traceback
from pathlib import Path
from typing import Any

__all__ = ["main"]


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _snapshot(worker: Any, observation: Any, infos: Any) -> dict[str, Any]:
    return {
        "observation": observation,
        "gamefile": infos.get("extra.gamefile"),
        "won": infos.get("won"),
        "history_items": copy.deepcopy(worker.history_items),
        "location": worker.location,
        "holding": worker.holding,
        "item_location": copy.deepcopy(worker.item_location),
        "history_list": copy.deepcopy(worker.history_list),
        "done": worker.done,
    }


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    from lh_platform.methods.lifecycle import resolve_repo_context

    context = resolve_repo_context(
        slug="graphgpo-reset-probe",
        explicit_repo=source_root,
        import_path=source_root,
        probe_import=lambda: importlib.import_module("recipe.GraphGPO.alfworld_envs") is not None,
        supported_paths=["upstream_native_runner"],
    )
    if not context.probe_result.success:
        raise RuntimeError(context.probe_result.error)

    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (  # type: ignore[import-not-found]
        get_environment,
    )
    from recipe.GraphGPO.alfworld_envs import (  # type: ignore[import-not-found]
        AlfworldWorker,
        load_config_file,
    )

    config_path = (
        source_root / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    )
    config = load_config_file(config_path)
    config["general"]["use_cuda"] = False
    base_env = get_environment(config["env"]["type"])(config, train_eval="train")
    worker = AlfworldWorker(config, args.seed, base_env)
    initial_obs, initial_infos = worker.reset()
    task_before = str(_first(initial_infos.get("extra.gamefile")) or args.task_id)
    initial = _snapshot(worker, initial_obs, initial_infos)

    commands = _first(initial_infos.get("admissible_commands")) or []
    if isinstance(commands, str):
        commands = [commands]
    action = next(
        (command for command in commands if command not in {"look", "inventory"}),
        commands[0] if commands else "look",
    )
    mutated_obs, _, _, mutated_infos = worker.step(action)
    # Poison every adapter-owned mutable field so reset completeness is observable
    # even if the selected environment action happens to be a semantic no-op.
    worker.history_items["__lh_reset_probe__"] = {"dirty": True}
    worker.location = "__lh_reset_probe__"
    worker.holding = "__lh_reset_probe__"
    worker.item_location["__lh_reset_probe__"] = {"dirty": True}
    worker.history_list.append(("__lh_reset_probe__",))
    worker.done = True
    mutated = _snapshot(worker, mutated_obs, mutated_infos)

    reset_obs, reset_infos = worker.reset()
    task_after = str(_first(reset_infos.get("extra.gamefile")) or args.task_id)
    reset = _snapshot(worker, reset_obs, reset_infos)
    residual_fields = [
        key
        for key in (
            "history_items",
            "location",
            "holding",
            "item_location",
            "history_list",
            "done",
        )
        if reset[key] != initial[key]
    ]
    close = getattr(worker.env, "close", None)
    if callable(close):
        close()
    return {
        "schema": "lh_state_reset_probe/v1",
        "benchmark": "alfworld_native",
        "source_commit": args.source_commit,
        "task_manifest_sha256": args.task_manifest_sha256,
        "probe_kind": "live_alfworld_same_task",
        "actual_execution_path": "upstream_native_runner",
        "exit_code": 0,
        "task_id_before": task_before,
        "task_id_after": task_after,
        "declared_task_id": args.task_id,
        "action": action,
        "initial_state_sha256": _stable_hash(initial),
        "mutated_state_sha256": _stable_hash(mutated),
        "reset_state_sha256": _stable_hash(reset),
        "residual_state_fields": residual_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--task-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        payload = _run(args)
        return_code = 0
    except Exception as error:  # the failure artifact is part of audit evidence
        payload = {
            "schema": "lh_state_reset_probe/v1",
            "benchmark": "alfworld_native",
            "source_commit": args.source_commit,
            "task_manifest_sha256": args.task_manifest_sha256,
            "probe_kind": "live_alfworld_same_task",
            "actual_execution_path": "upstream_native_runner",
            "exit_code": 1,
            "task_id_before": args.task_id,
            "task_id_after": None,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "residual_state_fields": ["probe_failed"],
        }
        return_code = 1
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
