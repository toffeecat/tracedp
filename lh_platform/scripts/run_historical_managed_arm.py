#!/usr/bin/env python3
"""Run one paid historical managed arm for a frozen recovery-composition matrix.

Unlike the zero-model probe, this script launches the real MiniMax Actor, so it
consumes provider tokens.  It always replays the frozen failure prefix first, so
the Runtime manages the real post-failure state rather than a fresh task.

Use `--mode prepare` to freeze the Harbor configuration without spending anything,
and `--mode run` to execute it.  A guidance candidate is never delivered to the
live Actor unless the isolated trial makes the registered public checks supported.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from terminal_historical_managed_launch import (  # noqa: E402
    _resolve_jit_memory,
    frozen_source_cwd,
    prefix_from_frozen_source,
    read_json,
)

# `storage_root` and `terminal_root` are not rewritten to a host-side spelling:
# they become Harbor bind mounts that Docker resolves inside the Linux VM, which
# may expose a different spelling than the host.  See `prepare()` for the full
# explanation.
from lh_platform.benchmarks.terminal_attribution_guidance import (  # noqa: E402
    historical_investigation_continuation,
)
from lh_platform.benchmarks.terminal_observed_runtime import (  # noqa: E402
    ObservedTerminalRunSpec,
    build_terminal_historical_managed_rescue_harbor_config,
)
from lh_platform.benchmarks.terminal_runtime_identity import (  # noqa: E402
    resolve_host_pi_runtime,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (  # noqa: E402
    RUNTIME_OWNED_OPTION_DP_DEV_METHODS,
    TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD,
)

MATRIX_SCHEMA = "recovery_composition_matrix_v1"




def hypothesis_estimates(matrix: dict, arm: dict, output: Path) -> dict:
    """The arm's estimates; an arm with a hypothesis locator reads its locate receipt.

    The driver runs the author locator first and charges it to this arm's ledger. A failed
    locator leaves the frozen estimates unchanged: TraceDP continues without the hypothesis.
    """

    base = matrix["atomic_estimate_configuration"]
    if arm.get("supervisor_plan"):
        # TraceDP v3: the dependency axis (last writer of a refuted subject), when found.
        hypothesis_path = output.with_suffix(".hypothesis.json")
        if not hypothesis_path.is_file():
            return base
        events = [json.loads(hypothesis_path.read_text())["event_id"]]
    elif not arm.get("hypothesis_locator"):
        return base
    else:
        receipt = json.loads(output.with_suffix(".locate.json").read_text())
        finding = receipt.get("finding") or {}
        events = finding.get("event_ids") or []
        if receipt.get("status") == "method_failure" or not events:
            return base
    from lh_platform.tracefix_bench.hypothesis_check_priors import (
        hypothesis_adjusted_configuration,
    )

    source = Path(matrix["controlled_injection"]["source_session"])
    observer = source.parents[2] / "terminal-public-state.jsonl"
    rows = [json.loads(line) for line in observer.read_text().splitlines() if line.strip()]
    order = [c["check_id"] for c in matrix["public_acceptance_checks"]]
    return hypothesis_adjusted_configuration(
        base, checks=matrix["public_acceptance_checks"], observer_rows=rows,
        event_id=str(events[0]), check_order=order,
    )


def rollback_snapshot(output: Path) -> dict | None:
    """A per-run rollback snapshot chosen by the driver from this arm's hypothesis."""

    path = output.with_suffix(".rollback.json")
    return json.loads(path.read_text()) if path.is_file() else None


def dover_intervention_text(output: Path) -> str | None:
    """DoVer's replacement text for this run (run_dover_intervention_stage), if valid."""

    path = output.with_suffix(".intervention.txt")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def write_reentry_config(
    output: Path, *, matrix: dict, budget_ledger: Path, rollback: dict | None, intervention: str | None,
    env_evidence_delivered: bool,
) -> Path:
    """TraceDP v3c: what the supervisor needs at the Actor's settlements."""

    import re

    from lh_platform.benchmarks.supervisor_reentry import CONFIG_SCHEMA

    inventory = json.loads(output.with_suffix(".initial_inventory.json").read_text())
    restore_receipt = rollback["receipt"] if rollback is not None else matrix["prefix_snapshot"]["receipt"]
    env_evidence = output.with_suffix(".env_evidence.txt")
    config = {
        "schema_version": CONFIG_SCHEMA,
        "run_label": re.sub(r"[^a-z0-9_.-]", "_", output.stem.lower())[:100],
        "docker_context": str(matrix["shared_environment"]["docker_context"]),
        "directories": inventory["directories"],
        "scripts": inventory["scripts"],
        "outputs": inventory["outputs"],
        "initial_inventory": inventory["files"],
        "restore_image": json.loads(Path(restore_receipt).read_text())["image_tag"],
        "rolled_back": rollback is not None,
        "intervention_text": intervention,
        "env_evidence_text": env_evidence.read_text(encoding="utf-8").strip() if env_evidence.is_file() else None,
        "env_evidence_delivered": env_evidence_delivered,
        "ledger_path": str(budget_ledger),
        "log_path": str(output.with_suffix(".reentry.jsonl")),
        "staging_dir": str(output.with_suffix(".reentry_staging")),
    }
    path = output.with_suffix(".reentry_config.json")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    return path


def apply_prefix_snapshot(config: dict, matrix: dict) -> None:
    """Start from the cell's verified snapshot: its derived task and its receipt."""

    snapshot = matrix.get("prefix_snapshot")
    if not snapshot:
        return
    config["tasks"][0]["path"] = str(snapshot["derived_task_path"])
    config["agents"][0]["kwargs"]["prefix_snapshot_receipt"] = str(snapshot["receipt"])


def arm_planning_horizon_cap(arm: dict) -> int | None:
    """Read an arm's planning-horizon cap under either historical spelling.

    The matrix builder emits ``runtime_planning_horizon_cap`` (the agent kwarg name) while older
    matrices used ``planning_horizon_cap``.  Reading only one spelling silently dropped the knob,
    so horizon ablations executed identically.
    """

    for key in ("runtime_planning_horizon_cap", "planning_horizon_cap"):
        value = arm.get(key)
        if value is not None:
            return int(value)
    return None


MANAGED_CONTROLLER_JOBS_DIRNAME = "managed_controller_jobs"


def bind_managed_controller(
    config: dict, *, runtime_root: Path, docker_context: str, expected_architecture: str
) -> None:
    """Add the contracts the managed controller requires before Harbor starts.

    Runs launched through ``run_terminal_managed_runtime_harbor`` get Docker, Pi and
    credential preflight, a task/verifier byte contract frozen before the run, and
    per-trial finalization (decision replay, check and repair receipts, token
    accounting).
    """

    agent = config["agents"][0]
    provider, model = str(agent["model_name"]).split("/", maxsplit=1)
    identity = resolve_host_pi_runtime(
        provider=provider, model=model, thinking=str(agent["kwargs"]["thinking"])
    )
    pi_executable = os.environ.get("LONGH_PI_EXECUTABLE") or shutil.which("pi")
    if not pi_executable:
        raise ValueError("managed controller needs the host Pi executable")
    if len(config["tasks"]) != 1:
        raise ValueError("a historical managed arm launches exactly one task")
    task_path = str(config["tasks"][0]["path"])
    # The shared runs/traceback root holds an unmarked frozen namespace from September
    # that the controller refuses to adopt; controller jobs get their own jobs root.
    jobs_root = Path(config["jobs_dir"]) / MANAGED_CONTROLLER_JOBS_DIRNAME
    jobs_root.mkdir(parents=True, exist_ok=True)
    config["jobs_dir"] = str(jobs_root)
    config["python_executable"] = sys.executable
    config["managed_runtime_backend"] = {
        "type": "docker_context",
        "context": docker_context,
        "expected_architecture": expected_architecture,
    }
    config["managed_runtime_storage"] = {"root": str(runtime_root)}
    config["managed_runtime_actor"] = {
        "pi_executable": str(Path(pi_executable).resolve()),
        "pi_version": str(agent["kwargs"]["pi_version"]),
        "credential_env": identity.credential_env,
        "credential_source": "environment",
    }
    config["managed_runtime_dataset_preflight"] = {
        "expected_roster": [
            {"task_id": Path(task_path).name, "source": None, "canonical_path": task_path}
        ]
    }


def assert_arm_knobs_reached_request(config: dict, arm: dict) -> None:
    """Fail closed when a declared arm knob did not reach the Harbor agent kwargs.

    A matrix whose arm claims an ablation must not run with the knob missing: that produced
    "same name, identical execution" rows before.
    """

    kwargs = config["agents"][0]["kwargs"]
    expected = {
        "runtime_planning_horizon_cap": arm_planning_horizon_cap(arm),
        "runtime_residual_structure": arm.get("residual_structure"),
        "runtime_experience_mode": arm.get("experience_mode"),
        "runtime_progress_trigger_threshold": arm.get("progress_trigger_threshold"),
    }
    if arm.get("supervisor_closed_loop") and not kwargs.get("supervisor_reentry_config"):
        raise ValueError("declared arm knob did not reach the request: supervisor_reentry_config")
    if arm.get("concurrent_runtime_observation") and kwargs.get("concurrent_runtime_observation") is not True:
        raise ValueError("declared arm knob did not reach the request: concurrent_runtime_observation")
    for key, value in expected.items():
        if value is None:
            continue
        if kwargs.get(key) != value:
            raise ValueError(
                "declared arm knob did not reach the request: "
                f"{key}={value!r} got {kwargs.get(key)!r}"
            )


def task_agent_timeout_seconds(terminal_root: Path, task: str) -> float | None:
    """Read the task's own agent-phase timeout from its contract."""

    task_toml = terminal_root / "tasks" / task / "task.toml"
    if not task_toml.is_file():
        return None
    payload = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    timeout = (payload.get("agent") or {}).get("timeout_sec")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        return None
    return float(timeout)


def assert_calibration_knobs_reached_request(
    config: dict,
    *,
    calibration_mode: bool,
    calibration_negative_control: bool,
    calibration_source_task_id: str | None,
) -> None:
    """Fail closed when A0 calibration mode did not reach the Harbor agent kwargs.

    A calibration run that silently executes as a normal continuation would spend provider
    tokens on the wrong behavior and produce no `a0-calibration.json`; the launch is refused
    instead.
    """

    kwargs = config["agents"][0]["kwargs"]
    if kwargs.get("runtime_calibration_mode", False) != calibration_mode:
        raise ValueError(
            "calibration mode did not reach the request: "
            f"declared {calibration_mode!r} got {kwargs.get('runtime_calibration_mode')!r}"
        )
    if not calibration_mode:
        return
    if (
        kwargs.get("runtime_calibration_negative_control")
        != calibration_negative_control
    ):
        raise ValueError("calibration negative-control flag did not reach the request")
    if kwargs.get("runtime_calibration_source_task_id") != calibration_source_task_id:
        raise ValueError("calibration source task id did not reach the request")


def prepare(
    *,
    matrix: dict[str, object],
    matrix_path: Path,
    arm_id: str,
    storage_root: Path,
    terminal_root: Path,
    source_session: Path,
    source_session_cwd: Path,
    run_id: str,
    output: Path,
    host_pi_timeout_seconds: int,
    calibration_mode: bool = False,
    calibration_negative_control: bool = False,
    calibration_source_task_id: str | None = None,
    runtime_root: Path | None = None,
) -> dict[str, object]:
    """Freeze one paid arm configuration without calling any provider."""

    arms = {row["arm_id"]: row for row in matrix["arms"]}
    if arm_id not in arms:
        raise ValueError(f"unknown arm: {arm_id}")
    arm = arms[arm_id]
    if matrix.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("paid arm requires a frozen recovery-composition matrix")
    if matrix.get("contract_version") == "v0.2":
        # v0.2 matrices state prerequisites the launcher can check itself.
        from build_recovery_matrix_v3 import matrix_launch_blockers

        blockers = matrix_launch_blockers(matrix, arm_id)
        if calibration_mode:
            # A0 calibration *is* the isolated admission trial, so only that blocker is waived.
            blockers = [item for item in blockers if "without an isolated admission trial" not in item]
        if blockers:
            raise ValueError("matrix launch blockers: " + "; ".join(blockers))
    expected_stop = matrix.get("expected_terminal_stop_reason")
    if not isinstance(expected_stop, str) or not expected_stop:
        raise ValueError("recovery-composition matrix requires expected_terminal_stop_reason")
    # These paths go into Harbor's `jobs_dir` and `tasks[].path`, which Docker
    # bind-mounts from inside the Linux VM.  The VM shares the storage volume under
    # a single path spelling, and a case variant of that path may exist on a
    # case-insensitive host but not in the VM.  A wrong case therefore makes the
    # daemon create an empty VM-local directory, so the verifier writes to
    # `/logs/verifier` and the host never sees them.  The environment-provided
    # spelling is authoritative and must not be rewritten.
    source_session_cwd = frozen_source_cwd(source_session)
    actor = matrix["shared_actor"]
    runtime = resolve_host_pi_runtime(
        provider=str(actor["provider"]),
        model=str(actor["model"]),
        thinking=str(actor["thinking"]),
    )
    task = str(matrix["task_instance"]["task_name"])
    jit_memory = _resolve_jit_memory(arm)
    continuation_instruction = (
        str(arm["continuation_instruction"])
        if arm.get("continuation_instruction") is not None
        else None
    )
    intervention = (
        dover_intervention_text(output) if arm.get("dover_intervention") or arm.get("supervisor_plan") else None
    )
    env_evidence = output.with_suffix(".env_evidence.txt")
    if arm.get("supervisor_plan") and intervention is None and env_evidence.is_file():
        # The clean-context finding reaches the Actor worded as a refuted check.
        opening = matrix.get("continuation_opening") or continuation_instruction or ""
        continuation_instruction = f"{opening} {env_evidence.read_text(encoding='utf-8').strip()}".strip()
    if intervention is not None:
        # The same wrapper and opening as every bridge lead.
        continuation_instruction = historical_investigation_continuation(
            intervention, opening=matrix.get("continuation_opening") or continuation_instruction
        )
    task_instruction = (terminal_root / "tasks" / task / "instruction.md").read_text(
        encoding="utf-8"
    )
    prefix, message_count, session_id = prefix_from_frozen_source(
        source_session=source_session,
        source_session_cwd=source_session_cwd,
        task_instruction=task_instruction,
        provider=runtime.provider,
        model=runtime.model,
        thinking=runtime.thinking,
        expected_terminal_stop_reason=str(matrix["expected_terminal_stop_reason"]),
    )
    selected_repairs = [
        row
        for row in matrix["fixer_candidates"]
        if row["repair_id"] in arm["repair_ids"]
    ]
    if len(selected_repairs) != len(arm["repair_ids"]):
        raise ValueError("arm references an unknown repair candidate")
    config = build_terminal_historical_managed_rescue_harbor_config(
        ObservedTerminalRunSpec(
            run_id=run_id,
            task=task,
            public_paths=tuple(matrix["public_paths"]),
        ),
        storage_root=storage_root,
        terminal_bench_root=terminal_root,
        runtime=runtime,
        source_session=source_session,
        source_session_cwd=source_session_cwd,
        prefix_tool_calls=tuple(prefix),
        source_actor={
            "provider": actor["provider"],
            "model": actor["model"],
            "transport": "host_pi_unix_rpc",
            "thinking": actor["thinking"],
            "max_output_tokens": actor["source_max_output_tokens"],
        },
        # The source episode's own budget; the continuation budget is passed as
        # host_pi_max_output_tokens below so the two can differ deliberately.
        source_actor_max_output_tokens=int(actor["source_max_output_tokens"]),
        expected_terminal_stop_reason=str(matrix["expected_terminal_stop_reason"]),
        container_proxy=os.environ.get("LONGH_CONTAINER_PROXY") or None,
        resource_specs=list(matrix["resource_specs"]),
        retention_budget_bytes=int(matrix["retention_budget_bytes"]),
        runtime_method=str(arm["runtime_method"]),
        runtime_intervention_budget=int(arm["runtime_intervention_budget"]),
        planning_horizon_cap=arm_planning_horizon_cap(arm),
        residual_structure=(
            str(arm["residual_structure"])
            if arm.get("residual_structure") is not None
            else None
        ),
        experience_mode=(
            str(arm["experience_mode"])
            if arm.get("experience_mode") is not None
            else None
        ),
        progress_trigger_threshold=(
            int(arm["progress_trigger_threshold"])
            if arm.get("progress_trigger_threshold") is not None
            else None
        ),
        jit_memory_path=str(jit_memory) if jit_memory is not None else None,
        jit_test_task_id=task if jit_memory is not None else None,
        continuation_instruction=continuation_instruction,
        runtime_check_contracts=list(matrix["public_acceptance_checks"]),
        runtime_predicate_repair_contracts=list(selected_repairs),
        runtime_acceptance_predicate_manifest=list(
            matrix["public_acceptance_predicates"]
        ),
        runtime_acceptance_predicate_semantic_review=matrix["public_semantic_review"],
        runtime_acceptance_candidate_trials=bool(
            arm["runtime_acceptance_candidate_trials"]
        ),
        # Every Runtime-owned option-DP arm needs the frozen estimate configuration, including
        # the JIT v2 method: the hardcoded non-JIT id silently starved the Full arm and the
        # agent refused to construct the resolver (`Runtime-owned option DP requires frozen
        # atomic estimates`).
        runtime_atomic_estimate_configuration=(
            hypothesis_estimates(matrix, arm, output)
            if arm["runtime_method"] in RUNTIME_OWNED_OPTION_DP_DEV_METHODS
            else None
        ),
        runtime_docker_candidate_provider=(
            matrix["shared_environment"]["candidate_provider"]
            if arm["runtime_acceptance_candidate_trials"]
            else None
        ),
        runtime_guided_completion_configuration=(
            matrix["guided_completion_configuration"]
            if arm["runtime_method"] == TRACEDP_GUIDANCE_PHASE_DP_DEV_METHOD
            else None
        ),
        calibration_mode=calibration_mode,
        calibration_negative_control=calibration_negative_control,
        calibration_source_task_id=calibration_source_task_id,
        run_class=str(matrix["run_class"]),
        actor_resume_limit=1,
        host_pi_timeout_seconds=host_pi_timeout_seconds,
        host_pi_max_output_tokens=int(actor["max_output_tokens"]),
    )
    # Uniform phase bound: the same 3600 s Actor bound for every arm.  Some
    # tasks declare a shorter agent timeout (cancel 900 s, dna-insert 1800 s) and with the
    # budget-12 loop a run legitimately needs longer, so the bound is raised uniformly for
    # every arm; a per-task bound would truncate some arms and confound the comparison.
    # The multiplier is frozen into every request and recorded in its phase-bound block.
    phase_bound_task_timeout = task_agent_timeout_seconds(terminal_root, task)
    if phase_bound_task_timeout is not None:
        # The controller requires the Harbor phase timeout to exceed the Actor bound by
        # its init/cleanup/cancellation reserve; the Actor bound itself stays uniform.
        from lh_platform.benchmarks.terminal_managed_runtime_controller import (
            _MANAGED_AGENT_OUTER_RESERVE_SECONDS,
        )

        multiplier = (
            host_pi_timeout_seconds + _MANAGED_AGENT_OUTER_RESERVE_SECONDS
        ) / phase_bound_task_timeout
        if multiplier > 1.0:
            config["agent_timeout_multiplier"] = multiplier
    budget_ledger = None
    if matrix.get("shared_budget") is not None:
        # Contract v0.2 R3: every arm draws its actor calls from one frozen pool.
        from lh_platform.tracefix_bench.scorer_contracts import ResourceRole
        from lh_platform.tracefix_bench.shared_budget import (
            FileSharedBudgetLedger,
            SharedRoleBudgetV1,
        )

        budget_ledger = output.with_suffix(".ledger.json")
        budget = SharedRoleBudgetV1(
            budget_id=f"{matrix['task_instance']['task_name']}:{arm_id}",
            roles=(ResourceRole.ACTOR, ResourceRole.LOCATOR, ResourceRole.ORCHESTRATION),
            **matrix["shared_budget"],
        )
        if (arm.get("hypothesis_locator") or arm.get("supervisor_plan")) and budget_ledger.exists():
            # The hypothesis locator already drew from this arm's pool: keep one ledger.
            if FileSharedBudgetLedger(budget_ledger).budget() != budget:
                raise ValueError("hypothesis locator charged a different budget than this arm's")
        else:
            FileSharedBudgetLedger(budget_ledger, budget)
        agent_kwargs = config["agents"][0]["kwargs"]
        apply_prefix_snapshot(config, matrix)
        override = (
            rollback_snapshot(output) if arm.get("rollback_on_hypothesis") or arm.get("supervisor_plan") else None
        )
        if override is not None:
            config["tasks"][0]["path"] = str(override["derived_task_path"])
            agent_kwargs["prefix_snapshot_receipt"] = str(override["receipt"])
        agent_kwargs["actor_budget_ledger_path"] = str(budget_ledger)
        agent_kwargs["actor_upstream_base_url"] = str(matrix["shared_actor"]["upstream_base_url"])
        if arm.get("concurrent_runtime_observation"):
            agent_kwargs["concurrent_runtime_observation"] = True
        if (matrix.get("observer") or {}).get("batched_public_snapshot"):
            # Harness knob shared by every arm of the matrix, never per arm.
            agent_kwargs["batched_public_snapshot"] = True
        if (matrix.get("observer") or {}).get("budget_affordability_gate"):
            # Harness knob, every arm of the matrix alike.
            agent_kwargs["budget_affordability_gate"] = True
        if (matrix.get("observer") or {}).get("verify_only_fallback"):
            agent_kwargs["post_actor_image_tag"] = f"longh-postactor/{run_id.lower()}:final"
        if arm.get("supervisor_closed_loop"):
            agent_kwargs["supervisor_reentry_config"] = str(write_reentry_config(
                output, matrix=matrix, budget_ledger=budget_ledger, rollback=override,
                intervention=intervention,
                env_evidence_delivered=bool(intervention is None and env_evidence.is_file()),
            ))
    bind_managed_controller(
        config,
        runtime_root=runtime_root or Path(os.environ["LONGH_RUNTIME_ROOT"]),
        docker_context=str(matrix["shared_environment"]["docker_context"]),
        expected_architecture=str(
            matrix["shared_environment"].get("expected_architecture", "x86_64")
        ),
    )
    assert_arm_knobs_reached_request(config, arm)
    assert_calibration_knobs_reached_request(
        config,
        calibration_mode=calibration_mode,
        calibration_negative_control=calibration_negative_control,
        calibration_source_task_id=calibration_source_task_id,
    )
    manifest = {
        "schema_version": "terminal_historical_managed_arm_request_v1",
        "implementation_status": "implementation_unverified",
        "scientific_result": False,
        "provider_calls_performed_by_this_step": 0,
        "arm_id": arm_id,
        "runtime_method": arm["runtime_method"],
        "source_matrix": str(matrix_path),
        "source_session": str(source_session),
        "source_session_id": session_id,
        "source_message_count": message_count,
        "frozen_prefix_tool_calls": len(prefix),
        "executed_prefix_tool_calls": sum(
            bool(row["environment_step_executed"]) for row in prefix
        ),
        "host_pi_timeout_seconds": host_pi_timeout_seconds,
        "shared_budget_ledger": str(budget_ledger) if budget_ledger is not None else None,
        "dover_intervention_delivered": intervention is not None,
        "env_evidence_delivered": bool(
            arm.get("supervisor_plan") and intervention is None and output.with_suffix(".env_evidence.txt").is_file()
        ),
        "docker_context": str(matrix["shared_environment"]["docker_context"]),
        "phase_bound": {
            "task_agent_timeout_seconds": phase_bound_task_timeout,
            "agent_timeout_multiplier": config.get("agent_timeout_multiplier"),
        },
        "calibration": {
            "mode": calibration_mode,
            "negative_control": calibration_negative_control,
            "source_task_id": calibration_source_task_id,
            "agent_timeout_multiplier": config.get("agent_timeout_multiplier"),
            "task_agent_timeout_seconds": phase_bound_task_timeout,
        },
        "harbor_config": config,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"arm request already exists: {output}")
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--source-session", type=Path, required=True)
    parser.add_argument("--source-session-cwd", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "run"), default="prepare")
    # The task's own contract declares a 3600 s agent timeout, so the canary uses
    # the same bound instead of a shorter local default.
    parser.add_argument("--host-pi-timeout-seconds", type=int, default=3600)
    # A0 calibration: replay the frozen prefix, run every registered check and one isolated
    # repair trial per candidate, then stop (no live continuation; each repair trial runs one
    # short isolated Actor session against a disposable clone, so it does spend provider tokens).
    parser.add_argument("--calibration-mode", action="store_true")
    parser.add_argument("--calibration-negative-control", action="store_true")
    parser.add_argument("--calibration-source-task-id", default=None)
    args = parser.parse_args()

    request_path = args.request.resolve()
    if args.mode == "run" and request_path.is_file():
        # A paid run must execute the frozen request that was reviewed.
        manifest = read_json(request_path)
        if manifest.get("schema_version") != (
            "terminal_historical_managed_arm_request_v1"
        ):
            raise ValueError("existing arm request has an unsupported schema")
    else:
        manifest = prepare(
            matrix=read_json(args.matrix),
            matrix_path=args.matrix.resolve(),
            arm_id=args.arm_id,
            storage_root=Path(os.environ["LONGH_STORAGE_ROOT"]).resolve(),
            terminal_root=Path(os.environ["LONGH_TERMINAL_BENCH_ROOT"]).resolve(),
            source_session=args.source_session.resolve(),
            source_session_cwd=args.source_session_cwd.resolve(),
            run_id=args.run_id,
            output=request_path,
            host_pi_timeout_seconds=args.host_pi_timeout_seconds,
            calibration_mode=args.calibration_mode,
            calibration_negative_control=args.calibration_negative_control,
            calibration_source_task_id=args.calibration_source_task_id,
        )
    if args.mode == "prepare":
        print(
            json.dumps(
                {key: value for key, value in manifest.items() if key != "harbor_config"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config_path = request_path.with_suffix(".harbor.json")
    config_path.write_text(
        json.dumps(manifest["harbor_config"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if "managed_runtime_backend" in manifest["harbor_config"]:
        from lh_platform.benchmarks.terminal_managed_runtime_controller import (
            run_terminal_managed_runtime_harbor,
        )

        exit_code = run_terminal_managed_runtime_harbor(config_path)
        print(f"managed controller exit code: {exit_code}")
        return exit_code
    # Requests frozen before the controller binding replay through Harbor directly.
    docker_context = str(
        manifest.get("docker_context")
        or manifest["harbor_config"]["shared_environment"]["docker_context"]
    )
    run_environment = {**os.environ, "DOCKER_CONTEXT": docker_context}
    run_environment.pop("DOCKER_HOST", None)
    completed = subprocess.run(
        [os.environ["LONGH_HARBOR"], "run", "--config", str(config_path), "--yes"],
        check=False,
        env=run_environment,
    )
    print(f"harbor exit code: {completed.returncode}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
