"""Portable Harbor configuration for observed Terminal-Bench runtime cells."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from lh_platform.benchmarks.terminal_forward_recovery_contract import (
    TERMINAL_FORWARD_CRITIQUE_PROMPT,
    TERMINAL_FORWARD_DIRECT_PROMPT,
    TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT,
    load_terminal_forward_recovery_source,
    load_terminal_state_coverage_receipt,
    normalize_terminal_forward_prompt_mode,
    normalize_terminal_public_history,
    normalize_terminal_source_actor,
    terminal_state_coverage_actor_attestation,
)
from lh_platform.benchmarks.terminal_managed_runtime_identity import (
    DYNAMIC_RUNTIME_METHODS,
    INTERVENTION_BUDGET_METHODS,
    SHARED_INPUT_PROTECTION_METHOD,
    SUPPORTED_RUNTIME_METHODS,
)
from lh_platform.benchmarks.terminal_pi_invocation import is_neutral_actor_cwd_of
from lh_platform.benchmarks.terminal_pi_session_continuation import (
    load_frozen_pi_continuation_source,
)
from lh_platform.benchmarks.terminal_public_state import (
    decode_exact_public_file_state,
    validate_actor_visible_path,
)
from lh_platform.benchmarks.terminal_replay_population import (
    compile_terminal_task_public_paths,
)
from lh_platform.benchmarks.terminal_resource_driver import (
    compile_terminal_resource_specs,
)
from lh_platform.benchmarks.terminal_runtime_identity import (
    HOST_PI_RUNTIME,
    TerminalRuntimeIdentity,
)
from lh_platform.benchmarks.terminal_source_census import MAX_HOST_PI_TIMEOUT_SECONDS
from lh_platform.benchmarks.terminal_task_contract import (
    resolve_terminal_source_task_contract,
    verify_terminal_task_contract,
)


def _same_directory(left: Path, right: Path) -> bool:
    """Compare two paths by filesystem identity rather than by spelling."""

    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _is_under(child: Path, root: Path) -> bool:
    """True when `child` is `root` or lies inside it, by filesystem identity.

    The check walks the child's ancestors and compares each one with
    `os.path.samefile`.  Raw string containment cannot be used here because the
    storage volume is reached through case variants (for example ``/Volumes/Data`` and
    ``/Volumes/data``) that name the same directory on macOS but are distinct paths in the
    container VM.
    """

    candidate = child
    while True:
        if _same_directory(candidate, root):
            return True
        if candidate.parent == candidate:
            return False
        candidate = candidate.parent


@dataclass(frozen=True)
class ObservedTerminalRunSpec:
    run_id: str
    task: str
    public_paths: tuple[str, ...]


def build_observed_terminal_harbor_config(
    spec: ObservedTerminalRunSpec,
    *,
    storage_root: Path,
    terminal_bench_root: Path,
    container_proxy: str | None,
    security_probe_only: bool = False,
    runtime: TerminalRuntimeIdentity = HOST_PI_RUNTIME,
) -> dict[str, object]:
    """Bind one official task to a frozen host-Pi actor and public state."""
    if not spec.run_id or any(character in spec.run_id for character in "/\\"):
        raise ValueError("run ID must be a non-empty path segment")
    public_paths = tuple(validate_actor_visible_path(path) for path in spec.public_paths)
    if not public_paths or len(public_paths) != len(set(public_paths)):
        raise ValueError("public paths must be non-empty and unique")

    task_dir = terminal_bench_root / "tasks" / spec.task
    instruction = (task_dir / "instruction.md").read_text()
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
    provenance = instruction + "\n" + dockerfile
    missing = [
        path
        for path in public_paths
        if path not in provenance and path.removeprefix("/app/") not in provenance
    ]
    if missing:
        raise ValueError(f"public paths lack official task provenance: {missing}")

    environment: dict[str, object] = {"type": "docker"}
    if container_proxy:
        environment["env"] = {
            "http_proxy": container_proxy,
            "https_proxy": container_proxy,
        }
    agent: dict[str, object] = {
        "name": ("lh_platform.benchmarks.harbor_opencode_go_pi:HostDelegatedOpenCodeGoPi"),
        "kwargs": {
            "pi_version": "0.84.1",
            "thinking": runtime.thinking,
            "public_paths": list(public_paths),
            **({"security_probe_only": True} if security_probe_only else {}),
        },
    }
    if not security_probe_only:
        agent["model_name"] = runtime.model_name
    return {
        "job_name": spec.run_id,
        "jobs_dir": str(storage_root / "runs" / "traceback"),
        "n_concurrent_trials": 1,
        "environment": environment,
        "agents": [agent],
        "tasks": [{"path": str(task_dir)}],
    }


def build_controlled_terminal_replay_harbor_config(
    spec: ObservedTerminalRunSpec,
    *,
    storage_root: Path,
    terminal_bench_root: Path,
    tool_calls: tuple[dict[str, object], ...],
    container_proxy: str | None,
    intervention: dict[str, object] | None = None,
    interventions: tuple[dict[str, object], ...] | None = None,
    snapshot: dict[str, object] | None = None,
    ladder: dict[str, object] | None = None,
    agent_timeout_multiplier: float | None = None,
) -> dict[str, object]:
    """Bind a frozen tool chain to the same official task and verifier.

    ``ladder`` (``every``, ``base_cache``, ``trace_writes``) switches to the replay agent
    that also records a checkpoint ladder and a per-call write trace; without it the
    config is byte-identical to before. ``agent_timeout_multiplier`` lets a zero-model
    replay of a long source outlast the task's agent timeout (build-pmars: 900 s).
    """
    if intervention is not None and interventions is not None:
        raise ValueError("provide either one intervention or a group, not both")
    config = build_observed_terminal_harbor_config(
        spec,
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
        container_proxy=container_proxy,
    )
    config["agents"] = [
        {
            "name": ("lh_platform.benchmarks.harbor_opencode_go_pi:HostTerminalReplayAgent"),
            "kwargs": {
                "public_paths": list(spec.public_paths),
                "tool_calls": list(tool_calls),
                **({"intervention": intervention} if intervention is not None else {}),
                **({"interventions": list(interventions)} if interventions is not None else {}),
                **(
                    {
                        "snapshot_image_tag": snapshot["image_tag"],
                        "snapshot_after_call": snapshot["after_call"],
                    }
                    if snapshot is not None
                    else {}
                ),
            },
        }
    ]
    if ladder is not None:
        agent = config["agents"][0]  # type: ignore[index]
        agent["name"] = "lh_platform.benchmarks.terminal_ladder_replay_agent:HostTerminalLadderReplayAgent"
        if ladder.get("every") is not None:
            agent["kwargs"]["ladder_every"] = int(ladder["every"])
            agent["kwargs"]["ladder_base_cache"] = str(ladder["base_cache"])
        if ladder.get("trace_writes"):
            agent["kwargs"]["trace_writes"] = True
        if ladder.get("source_era_bash"):
            agent["kwargs"]["source_era_bash"] = True
        if ladder.get("batched_public_snapshot"):
            agent["kwargs"]["batched_public_snapshot"] = True
    if agent_timeout_multiplier is not None:
        if not agent_timeout_multiplier > 0:
            raise ValueError("agent timeout multiplier must be positive")
        config["agent_timeout_multiplier"] = float(agent_timeout_multiplier)
    return config


def build_terminal_recovery_harbor_config(
    spec: ObservedTerminalRunSpec,
    *,
    storage_root: Path,
    terminal_bench_root: Path,
    prefix_tool_calls: tuple[dict[str, object], ...],
    causal_arm: str,
    intervention: dict[str, object] | None,
    container_proxy: str | None,
    host_pi_timeout_seconds: int = 900,
    host_pi_max_output_tokens: int = 16384,
) -> dict[str, object]:
    """Replay one public prefix, apply its repair, then resume with the frozen actor."""

    if not prefix_tool_calls:
        raise ValueError("Terminal recovery requires a non-empty public prefix")
    if causal_arm not in {"original", "replacement", "matched_sham"}:
        raise ValueError("Terminal recovery causal arm is invalid")
    if (causal_arm == "original") != (intervention is None):
        raise ValueError("only the original arm may omit the recovery intervention")
    if not 1 <= host_pi_timeout_seconds <= MAX_HOST_PI_TIMEOUT_SECONDS:
        raise ValueError("Terminal recovery host Pi timeout is outside the supported range")
    if host_pi_max_output_tokens < 1:
        raise ValueError("Terminal recovery host Pi max output tokens must be positive")
    if (
        intervention is not None
        and intervention.get("after_call_sequence") != len(prefix_tool_calls) - 1
    ):
        raise ValueError("Terminal recovery intervention must end the frozen prefix")
    config = build_observed_terminal_harbor_config(
        spec,
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
        container_proxy=container_proxy,
    )
    config["agents"] = [
        {
            "name": (
                "lh_platform.benchmarks.harbor_opencode_go_pi:HostTerminalRecoveryOpenCodeGoPi"
            ),
            "model_name": HOST_PI_RUNTIME.model_name,
            "kwargs": {
                "pi_version": "0.84.1",
                "thinking": HOST_PI_RUNTIME.thinking,
                "host_pi_timeout_seconds": host_pi_timeout_seconds,
                "host_pi_max_output_tokens": host_pi_max_output_tokens,
                "public_paths": list(spec.public_paths),
                "prefix_tool_calls": list(prefix_tool_calls),
                "causal_arm": causal_arm,
                "intervention": intervention,
            },
        }
    ]
    return config


def build_terminal_historical_session_rescue_harbor_config(
    spec: ObservedTerminalRunSpec,
    *,
    storage_root: Path,
    terminal_bench_root: Path,
    source_session: Path,
    source_session_cwd: Path,
    prefix_tool_calls: tuple[dict[str, object], ...],
    source_actor: dict[str, object],
    expected_terminal_stop_reason: str,
    container_proxy: str | None,
    continuation_treatment: str = "direct_continue",
    targeted_guidance_file: Path | None = None,
    continuation_opening: str | None = None,
    host_pi_timeout_seconds: int = 900,
    host_pi_max_output_tokens: int = 16384,
    runtime: TerminalRuntimeIdentity = HOST_PI_RUNTIME,
) -> dict[str, object]:
    """Replay one failure and let Pi append to its exact historical context."""

    source_session = source_session.resolve()
    source_session_cwd = source_session_cwd.resolve()
    trial_agent_dir = source_session.parent.parent.parent
    neutral_cwd = is_neutral_actor_cwd_of(source_session_cwd, trial_agent_dir)
    try:
        source_session.relative_to(storage_root.resolve())
        if not neutral_cwd:
            source_session_cwd.relative_to(storage_root.resolve())
    except ValueError as error:
        raise ValueError("historical Pi session must be under the storage root") from error
    if trial_agent_dir != source_session_cwd and not neutral_cwd:
        raise ValueError("historical Pi session cwd does not match its source trial")
    if not prefix_tool_calls:
        raise ValueError("historical Pi rescue requires a non-empty replay prefix")
    if not 1 <= host_pi_timeout_seconds <= MAX_HOST_PI_TIMEOUT_SECONDS:
        raise ValueError("historical Pi rescue timeout is outside the supported range")
    if host_pi_max_output_tokens < 1:
        raise ValueError("historical Pi rescue output budget must be positive")
    if continuation_treatment not in {"direct_continue", "targeted_guidance"}:
        raise ValueError("historical Pi continuation treatment is invalid")
    guidance: str | None = None
    guidance_source: str | None = None
    if continuation_treatment == "targeted_guidance":
        if targeted_guidance_file is None:
            raise ValueError("targeted historical continuation requires a guidance file")
        guidance_path = targeted_guidance_file.resolve()
        try:
            guidance_source = str(guidance_path.relative_to(storage_root.resolve()))
        except ValueError as error:
            raise ValueError("historical Pi guidance must be under the storage root") from error
        if not guidance_path.is_file():
            raise ValueError("historical Pi guidance file is missing")
        guidance = guidance_path.read_text(encoding="utf-8").strip()
        if not guidance:
            raise ValueError("historical Pi guidance file is empty")
    elif targeted_guidance_file is not None:
        raise ValueError("direct historical continuation cannot carry guidance")
    normalized_actor = normalize_terminal_source_actor(
        source_actor,
        runtime=runtime,
        max_output_tokens=host_pi_max_output_tokens,
    )
    task_dir = terminal_bench_root / "tasks" / spec.task
    task_instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    source = load_frozen_pi_continuation_source(
        source_session,
        expected_instruction=task_instruction,
        expected_provider=runtime.provider,
        expected_model=runtime.model,
        expected_thinking=runtime.thinking,
        expected_terminal_stop_reason=expected_terminal_stop_reason,
        expected_cwd=source_session_cwd,
    )
    if len(source.tool_receipt.tool_calls) != len(prefix_tool_calls):
        raise ValueError("historical Pi rescue prefix count differs")
    config = build_observed_terminal_harbor_config(
        spec,
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
        container_proxy=container_proxy,
        runtime=runtime,
    )
    config["agents"] = [
        {
            "name": (
                "lh_platform.benchmarks.harbor_opencode_go_pi:"
                "HostTerminalHistoricalSessionContinuationPi"
            ),
            "model_name": runtime.model_name,
            "kwargs": {
                "pi_version": "0.84.1",
                "thinking": runtime.thinking,
                "host_pi_timeout_seconds": host_pi_timeout_seconds,
                "host_pi_max_output_tokens": host_pi_max_output_tokens,
                "public_paths": list(spec.public_paths),
                "prefix_tool_calls": list(prefix_tool_calls),
                "source_session_path": str(source_session),
                "source_session_id": source.session_id,
                "source_session_cwd": str(source_session_cwd),
                "source_actor": normalized_actor,
                "expected_terminal_stop_reason": expected_terminal_stop_reason,
                "task_instruction": task_instruction,
                "continuation_treatment": continuation_treatment,
                **(
                    {"continuation_opening": continuation_opening}
                    if continuation_opening is not None
                    else {}
                ),
                "targeted_guidance": guidance,
                "targeted_guidance_source": guidance_source,
            },
        }
    ]
    return config


def build_terminal_historical_managed_rescue_harbor_config(
    spec: ObservedTerminalRunSpec,
    *,
    storage_root: Path,
    terminal_bench_root: Path,
    source_session: Path,
    source_session_cwd: Path,
    prefix_tool_calls: tuple[dict[str, object], ...],
    source_actor: dict[str, object],
    source_actor_max_output_tokens: int,
    expected_terminal_stop_reason: str,
    container_proxy: str | None,
    resource_specs: list[dict[str, object]],
    retention_budget_bytes: int,
    runtime_method: str,
    runtime_intervention_budget: int,
    planning_horizon_cap: int | None = None,
    residual_structure: str | None = None,
    experience_mode: str | None = None,
    progress_trigger_threshold: int | None = None,
    jit_memory_path: str | None = None,
    jit_test_task_id: str | None = None,
    runtime_check_contracts: list[dict[str, object]],
    runtime_predicate_repair_contracts: list[dict[str, object]],
    runtime_acceptance_predicate_manifest: list[dict[str, object]],
    runtime_acceptance_predicate_semantic_review: dict[str, object],
    runtime_acceptance_candidate_trials: bool,
    run_class: str,
    runtime_atomic_estimate_configuration: dict[str, object] | None = None,
    runtime_docker_candidate_provider: dict[str, object] | None = None,
    runtime_guided_completion_configuration: dict[str, object] | None = None,
    calibration_mode: bool = False,
    calibration_negative_control: bool = False,
    calibration_source_task_id: str | None = None,
    actor_resume_limit: int = 1,
    continuation_instruction: str | None = None,
    host_pi_timeout_seconds: int = 900,
    host_pi_max_output_tokens: int = 16384,
    runtime: TerminalRuntimeIdentity = HOST_PI_RUNTIME,
) -> dict[str, object]:
    """Let the managed Runtime own the continuation of one historical failure.

    The generated configuration differs from the historical continuation builder
    in exactly one way: it binds the managed Host, so the same replayed failure
    state is observed by a Runtime kernel with registered public acceptance checks,
    isolated repair trials, and one shared selector for every comparison arm.

    `source_actor_max_output_tokens` describes the episode that produced the
    failure, while `host_pi_max_output_tokens` is the shared continuation budget.
    They are allowed to differ so a downstream execution-capability ceiling can be
    lifted uniformly for every recovery arm without rewriting the source episode.

    A0 calibration mode (`calibration_mode`) stops after the frozen prefix replay and runs
    the register-only calibration sweep: every check and one isolated trial per repair
    candidate, each in a fresh clone of the same entry.  It never stages a continuation.
    The three calibration knobs are only emitted when the mode is requested, so every
    non-A0 request stays byte-identical to the legacy builder output.
    """

    if (calibration_negative_control or calibration_source_task_id is not None) and not (
        calibration_mode
    ):
        raise ValueError("calibration options require calibration mode")

    source_session = source_session.resolve()
    source_session_cwd = source_session_cwd.resolve()
    # Containment and equality are decided by filesystem identity, not raw strings.
    # macOS is case-insensitive but case-preserving, while the container VM shares the
    # storage volume under a single spelling, so `/Volumes/Data/...` and
    # `/Volumes/data/...` name one directory yet never compare equal as text.
    trial_agent_dir = source_session.parent.parent.parent
    neutral_cwd = is_neutral_actor_cwd_of(source_session_cwd, trial_agent_dir)
    if not _is_under(source_session, storage_root) or not (
        neutral_cwd or _is_under(source_session_cwd, storage_root)
    ):
        raise ValueError("historical managed Pi session must be under the storage root")
    if not (neutral_cwd or _same_directory(trial_agent_dir, source_session_cwd)):
        raise ValueError("historical managed Pi session cwd does not match its source trial")
    if not prefix_tool_calls:
        raise ValueError("historical managed rescue requires a non-empty replay prefix")
    if not 1 <= host_pi_timeout_seconds <= MAX_HOST_PI_TIMEOUT_SECONDS:
        raise ValueError("historical managed rescue timeout is outside the supported range")
    if host_pi_max_output_tokens < 1:
        raise ValueError("historical managed rescue output budget must be positive")
    if runtime_method not in SUPPORTED_RUNTIME_METHODS:
        raise ValueError("historical managed rescue runtime method is unsupported")
    # Mirror the Host's own budget rule instead of demanding a positive budget from
    # every method: observation-only methods must carry exactly zero interventions,
    # and requiring more would make the `direct` control arm unbuildable.
    if runtime_method in INTERVENTION_BUDGET_METHODS:
        if runtime_intervention_budget < 1:
            raise ValueError(
                "historical managed rescue requires a positive intervention budget"
            )
    elif runtime_intervention_budget != 0:
        raise ValueError(
            "only a budgeted Runtime method may receive an intervention budget"
        )
    if not runtime_check_contracts:
        raise ValueError("historical managed rescue requires executable public checks")
    if not runtime_acceptance_predicate_manifest:
        raise ValueError("historical managed rescue requires public acceptance predicates")
    if type(runtime_acceptance_candidate_trials) is not bool:
        raise ValueError("historical managed rescue candidate trial flag must be boolean")
    if runtime_docker_candidate_provider is not None and (
        run_class != "exposed_development_canary"
    ):
        raise ValueError("historical managed rescue candidate clones are development-only")
    if not resource_specs:
        raise ValueError("historical managed rescue requires declared resources")
    for row in resource_specs:
        if set(row) != {"resource_id", "kind", "provenance", "group_id"}:
            raise ValueError("historical managed rescue resource schema differs")
    public_paths = tuple(validate_actor_visible_path(path) for path in spec.public_paths)
    if not public_paths or len(public_paths) != len(set(public_paths)):
        raise ValueError("historical managed rescue public paths must be non-empty and unique")
    if actor_resume_limit < 0:
        raise ValueError("historical managed rescue actor resume limit cannot be negative")
    normalized_actor = normalize_terminal_source_actor(
        source_actor,
        runtime=runtime,
        max_output_tokens=source_actor_max_output_tokens,
    )
    task_dir = terminal_bench_root / "tasks" / spec.task
    task_instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    # Reuse the frozen loader so the source contract is validated here too instead
    # of being discovered only after Harbor has already started a trial.
    source = load_frozen_pi_continuation_source(
        source_session,
        expected_instruction=task_instruction,
        expected_provider=runtime.provider,
        expected_model=runtime.model,
        expected_thinking=runtime.thinking,
        expected_terminal_stop_reason=expected_terminal_stop_reason,
        expected_cwd=source_session_cwd,
    )
    if len(source.tool_receipt.tool_calls) != len(prefix_tool_calls):
        raise ValueError("historical managed rescue prefix count differs")
    config = build_observed_terminal_harbor_config(
        spec,
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
        container_proxy=container_proxy,
        runtime=runtime,
    )
    config["agents"] = [
        {
            "name": (
                "lh_platform.benchmarks.harbor_opencode_go_pi:"
                "HostTerminalHistoricalManagedOpenCodeGoPi"
            ),
            "model_name": runtime.model_name,
            "kwargs": {
                "pi_version": "0.84.1",
                "thinking": runtime.thinking,
                "host_pi_timeout_seconds": host_pi_timeout_seconds,
                "host_pi_max_output_tokens": host_pi_max_output_tokens,
                "public_paths": list(public_paths),
                # Historical failure state
                "source_session_path": str(source_session),
                "source_session_id": source.session_id,
                "source_session_cwd": str(source_session_cwd),
                "source_actor": normalized_actor,
                "source_actor_max_output_tokens": source_actor_max_output_tokens,
                "expected_terminal_stop_reason": expected_terminal_stop_reason,
                "task_instruction": task_instruction,
                "prefix_tool_calls": list(prefix_tool_calls),
                **(
                    {"continuation_instruction": continuation_instruction}
                    if continuation_instruction is not None
                    else {}
                ),
                # Managed Runtime
                "resource_specs": list(resource_specs),
                "retention_budget_bytes": retention_budget_bytes,
                "restore_missing_prerequisites": False,
                "runtime_method": runtime_method,
                "runtime_intervention_budget": runtime_intervention_budget,
                **(
                    {"runtime_planning_horizon_cap": planning_horizon_cap}
                    if planning_horizon_cap is not None
                    else {}
                ),
                **(
                    {"runtime_residual_structure": residual_structure}
                    if residual_structure is not None
                    else {}
                ),
                **(
                    {"runtime_experience_mode": experience_mode}
                    if experience_mode is not None
                    else {}
                ),
                **(
                    {"runtime_progress_trigger_threshold": progress_trigger_threshold}
                    if progress_trigger_threshold is not None
                    else {}
                ),
                **(
                    {"jit_memory_path": jit_memory_path}
                    if jit_memory_path is not None
                    else {}
                ),
                **(
                    {"jit_test_task_id": jit_test_task_id}
                    if jit_test_task_id is not None
                    else {}
                ),
                "runtime_check_contracts": list(runtime_check_contracts),
                "runtime_predicate_repair_contracts": list(
                    runtime_predicate_repair_contracts
                ),
                "runtime_acceptance_predicate_manifest": list(
                    runtime_acceptance_predicate_manifest
                ),
                "runtime_acceptance_predicate_semantic_review": (
                    runtime_acceptance_predicate_semantic_review
                ),
                "runtime_acceptance_candidate_trials": (
                    runtime_acceptance_candidate_trials
                ),
                "runtime_atomic_estimate_configuration": (
                    runtime_atomic_estimate_configuration
                ),
                "runtime_docker_candidate_provider": (
                    runtime_docker_candidate_provider
                ),
                # Emitted only for the guidance-phase planner, so every other request
                # stays byte-identical to the legacy builder output.
                **(
                    {
                        "runtime_guided_completion_configuration": (
                            runtime_guided_completion_configuration
                        )
                    }
                    if runtime_guided_completion_configuration is not None
                    else {}
                ),
                **(
                    {
                        "runtime_calibration_mode": True,
                        "runtime_calibration_negative_control": (
                            calibration_negative_control
                        ),
                        "runtime_calibration_source_task_id": (
                            calibration_source_task_id
                        ),
                    }
                    if calibration_mode
                    else {}
                ),
                "actor_resume_limit": actor_resume_limit,
                "run_class": run_class,
            },
        }
    ]
    return config


def build_terminal_historical_managed_probe_harbor_config(
    spec: ObservedTerminalRunSpec,
    *,
    probe_actor_commands: tuple[str, ...],
    **managed_kwargs: object,
) -> dict[str, object]:
    """Bind the same managed failure state to the zero-model probe Host.

    The probe reuses the managed builder unchanged, so the kernel, checks, repairs,
    estimates, and resource declarations cannot drift between a probe run and the
    real arm it qualifies.  Only the Actor's token generation is replaced.
    """

    if not probe_actor_commands:
        raise ValueError("the zero-model probe requires at least one Actor command")
    config = build_terminal_historical_managed_rescue_harbor_config(
        spec,
        **managed_kwargs,  # type: ignore[arg-type]
    )
    agent = config["agents"][0]
    assert isinstance(agent, dict)
    kwargs = agent["kwargs"]
    assert isinstance(kwargs, dict)
    agent["name"] = (
        "lh_platform.benchmarks.harbor_opencode_go_pi:"
        "HostTerminalHistoricalManagedZeroModelProbe"
    )
    kwargs["zero_model_actor_commands"] = list(probe_actor_commands)
    if kwargs.get("run_class") != "exposed_development_canary":
        raise ValueError("the zero-model probe is development-only")
    return config


def build_terminal_forward_recovery_harbor_config(
    spec: ObservedTerminalRunSpec,
    *,
    storage_root: Path,
    terminal_bench_root: Path,
    source_trial: Path,
    repair_state: tuple[dict[str, object], ...] | None,
    state_coverage_receipt: Path,
    source_map_receipt: Path,
    container_proxy: str | None,
    continuation_prompt_mode: str = TERMINAL_FORWARD_DIRECT_PROMPT,
    targeted_guidance: str | None = None,
    host_pi_timeout_seconds: int = 900,
    host_pi_max_output_tokens: int = 16384,
    runtime: TerminalRuntimeIdentity = HOST_PI_RUNTIME,
) -> dict[str, object]:
    """Restore one admitted public failure projection, then continue its actor."""

    task_dir = terminal_bench_root / "tasks" / spec.task
    task_contract_receipt = resolve_terminal_source_task_contract(
        source_map_receipt,
        storage_root=storage_root,
        source_trial=source_trial,
        expected_task=spec.task,
        expected_public_paths=spec.public_paths,
    )
    verify_terminal_task_contract(
        task_dir,
        task_contract_receipt,
        expected_task=spec.task,
    )
    source = load_terminal_forward_recovery_source(
        source_trial,
        public_paths=spec.public_paths,
        expected_task=spec.task,
    )
    failed_final_state = source.failed_public_file_state
    final_state = decode_exact_public_file_state(list(failed_final_state))
    final_paths = {item.path for item in final_state}
    if final_paths != set(spec.public_paths):
        raise ValueError("Terminal forward recovery requires every declared public path")
    continuation_prompt_mode = normalize_terminal_forward_prompt_mode(
        continuation_prompt_mode
    )
    normalized_guidance = targeted_guidance.strip() if targeted_guidance is not None else None
    if continuation_prompt_mode == TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT:
        if not normalized_guidance:
            raise ValueError("targeted continuation requires method-generated guidance")
    elif normalized_guidance is not None:
        raise ValueError("untargeted continuation cannot carry method-generated guidance")
    if repair_state is not None:
        repaired = decode_exact_public_file_state(list(repair_state))
        if not {item.path for item in repaired} <= final_paths:
            raise ValueError("Terminal forward repair escaped the observed public state")
    if (
        continuation_prompt_mode == TERMINAL_FORWARD_CRITIQUE_PROMPT
        and repair_state is not None
    ):
        raise ValueError("end-of-trace critique cannot include a pre-applied repair")
    if (
        continuation_prompt_mode == TERMINAL_FORWARD_TARGETED_GUIDANCE_PROMPT
        and repair_state is not None
    ):
        raise ValueError("targeted continuation cannot include a pre-applied repair")
    if not 1 <= host_pi_timeout_seconds <= MAX_HOST_PI_TIMEOUT_SECONDS:
        raise ValueError(
            "Terminal forward recovery host Pi timeout is outside the supported range"
        )
    if host_pi_max_output_tokens < 1:
        raise ValueError("Terminal forward recovery host Pi max output tokens must be positive")
    normalized_history = normalize_terminal_public_history(source.public_history)
    normalized_actor = normalize_terminal_source_actor(
        source.source_actor,
        runtime=runtime,
        max_output_tokens=host_pi_max_output_tokens,
    )
    normalized_coverage = load_terminal_state_coverage_receipt(
        state_coverage_receipt,
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
        expected_task=spec.task,
        expected_source_trial=source_trial,
        expected_source_map_receipt=source_map_receipt,
        expected_task_contract_receipt=task_contract_receipt,
        expected_public_paths=spec.public_paths,
    )
    config = build_observed_terminal_harbor_config(
        spec,
        storage_root=storage_root,
        terminal_bench_root=terminal_bench_root,
        container_proxy=container_proxy,
        runtime=runtime,
    )
    config["agents"] = [
        {
            "name": (
                "lh_platform.benchmarks.harbor_opencode_go_pi:"
                "HostTerminalForwardRecoveryOpenCodeGoPi"
            ),
            "model_name": runtime.model_name,
            "kwargs": {
                "pi_version": "0.84.1",
                "thinking": runtime.thinking,
                "host_pi_timeout_seconds": host_pi_timeout_seconds,
                "host_pi_max_output_tokens": host_pi_max_output_tokens,
                "public_paths": list(spec.public_paths),
                "failed_final_state": list(failed_final_state),
                "public_history": list(normalized_history),
                "repair_state": list(repair_state) if repair_state is not None else None,
                "source_actor": normalized_actor,
                "state_coverage": terminal_state_coverage_actor_attestation(
                    normalized_coverage
                ),
                "source_trial_name": source_trial.name,
                "continuation_prompt_mode": continuation_prompt_mode,
                "targeted_guidance": normalized_guidance,
            },
        }
    ]
    return config


def build_terminal_source_census_harbor_config(
    *,
    run_id: str,
    task: str,
    storage_root: Path,
    terminal_bench_root: Path,
    container_proxy: str | None,
    maximum_public_paths: int = 24,
    host_pi_timeout_seconds: int = 900,
    host_pi_max_output_tokens: int = 16384,
    runtime: TerminalRuntimeIdentity = HOST_PI_RUNTIME,
    observe_public_state: bool | None = None,
) -> dict[str, object]:
    """Run one official task with outcome-blind, replayable public observations.

    Paths come only from the task instruction and Dockerfile. This keeps source
    collection blind to both the actor outcome and the private verifier while
    preserving the transitions required for a later zero-model replay.
    """
    if not run_id or any(character in run_id for character in "/\\"):
        raise ValueError("run ID must be a non-empty path segment")
    if maximum_public_paths < 1:
        raise ValueError("maximum public paths must be positive")
    if not 1 <= host_pi_timeout_seconds <= MAX_HOST_PI_TIMEOUT_SECONDS:
        raise ValueError("source census host Pi timeout is outside the supported range")
    if host_pi_max_output_tokens < 1:
        raise ValueError("source census host Pi max output tokens must be positive")
    task_dir = terminal_bench_root / "tasks" / task
    if not (task_dir / "task.toml").is_file():
        raise ValueError(f"official Terminal task is missing: {task}")
    compilation = compile_terminal_task_public_paths(
        instruction=(task_dir / "instruction.md").read_text(),
        dockerfile=(task_dir / "environment" / "Dockerfile").read_text(),
        maximum_replay_paths=maximum_public_paths,
    )
    observation_enabled = (
        bool(compilation.replay_paths)
        if observe_public_state is None
        else observe_public_state
    )
    if observation_enabled and not compilation.replay_paths:
        raise ValueError("public-state observation cannot be forced without public paths")
    observed_paths = compilation.replay_paths if observation_enabled else ()

    environment: dict[str, object] = {"type": "docker"}
    if container_proxy:
        environment["env"] = {
            "http_proxy": container_proxy,
            "https_proxy": container_proxy,
        }
    return {
        "job_name": run_id,
        "jobs_dir": str(storage_root / "runs" / "traceback"),
        "n_concurrent_trials": 1,
        "environment": environment,
        "agents": [
            {
                "name": ("lh_platform.benchmarks.harbor_opencode_go_pi:HostDelegatedOpenCodeGoPi"),
                "model_name": runtime.model_name,
                "kwargs": {
                    "pi_version": "0.84.1",
                    "thinking": runtime.thinking,
                    "host_pi_timeout_seconds": host_pi_timeout_seconds,
                    "host_pi_max_output_tokens": host_pi_max_output_tokens,
                    "public_paths": list(observed_paths),
                    "observe_public_state": observation_enabled,
                },
            }
        ],
        "tasks": [{"path": str(task_dir)}],
    }


def build_terminal_managed_runtime_harbor_config(
    *,
    run_id: str,
    task: str,
    storage_root: Path,
    terminal_bench_root: Path,
    container_proxy: str | None,
    retention_budget_bytes: int,
    maximum_public_paths: int = 24,
    host_pi_timeout_seconds: int = 960,
    host_pi_max_output_tokens: int = 16384,
    runtime: TerminalRuntimeIdentity = HOST_PI_RUNTIME,
    runtime_method: str = "shared_input_protection_v1",
    runtime_intervention_budget: int = 4,
) -> dict[str, object]:
    """Build one continuous Pi/Harbor cell with independent resource handling."""

    if not run_id or any(character in run_id for character in "/\\"):
        raise ValueError("run ID must be a non-empty path segment")
    if retention_budget_bytes < 1:
        raise ValueError("managed runtime retention budget must be positive")
    if not 1 <= host_pi_timeout_seconds <= MAX_HOST_PI_TIMEOUT_SECONDS:
        raise ValueError("managed runtime host Pi timeout is outside the supported range")
    if host_pi_max_output_tokens < 1:
        raise ValueError("managed runtime host Pi max output tokens must be positive")
    if runtime_method not in SUPPORTED_RUNTIME_METHODS:
        raise ValueError("managed runtime method identity is unsupported")
    if (
        runtime_method in DYNAMIC_RUNTIME_METHODS
        and runtime_intervention_budget < 1
    ):
        raise ValueError("dynamic TraceDP runtime requires a positive budget")
    task_dir = terminal_bench_root / "tasks" / task
    if not (task_dir / "task.toml").is_file():
        raise ValueError(f"official Terminal task is missing: {task}")
    compilation = compile_terminal_task_public_paths(
        instruction=(task_dir / "instruction.md").read_text(),
        dockerfile=(task_dir / "environment" / "Dockerfile").read_text(),
        maximum_replay_paths=maximum_public_paths,
    )
    resource_specs = compile_terminal_resource_specs(compilation)
    if not resource_specs:
        raise ValueError("managed runtime requires public resource candidates")
    environment: dict[str, object] = {"type": "docker"}
    if container_proxy:
        environment["env"] = {
            "http_proxy": container_proxy,
            "https_proxy": container_proxy,
        }
    return {
        "job_name": run_id,
        "jobs_dir": str(storage_root / "runs" / "traceback"),
        "n_concurrent_trials": 1,
        "environment": environment,
        "agents": [
            {
                "name": (
                    "lh_platform.benchmarks.harbor_opencode_go_pi:"
                    "HostManagedOpenCodeGoPi"
                ),
                "model_name": runtime.model_name,
                "kwargs": {
                    "pi_version": "0.84.1",
                    "thinking": runtime.thinking,
                    "host_pi_timeout_seconds": host_pi_timeout_seconds,
                    "host_pi_max_output_tokens": host_pi_max_output_tokens,
                    "public_paths": list(compilation.replay_paths),
                    "observe_public_state": bool(compilation.replay_paths),
                    "resource_specs": [
                        {
                            "resource_id": spec.resource_id,
                            "kind": spec.kind.value,
                            "provenance": spec.provenance,
                            "group_id": spec.group_id,
                        }
                        for spec in resource_specs
                    ],
                    "retention_budget_bytes": retention_budget_bytes,
                    "restore_missing_prerequisites": (
                        runtime_method == SHARED_INPUT_PROTECTION_METHOD
                    ),
                    "runtime_method": runtime_method,
                    "runtime_intervention_budget": (
                        runtime_intervention_budget
                        if runtime_method in DYNAMIC_RUNTIME_METHODS
                        else 0
                    ),
                },
            }
        ],
        "tasks": [{"path": str(task_dir)}],
    }
