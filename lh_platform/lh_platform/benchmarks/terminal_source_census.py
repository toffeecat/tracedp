"""Frozen Terminal-Bench source-census population."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

TERMINAL_CENSUS_CLASSIFIER_VERSION = "terminal_census_case_v2"
HOST_PI_OUTER_TIMEOUT_GRACE_SECONDS = 60
MAX_HOST_PI_TIMEOUT_SECONDS = 14_400


class TerminalCaseClassification(StrEnum):
    """Scientific/operational status of one frozen Terminal case."""

    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    SOURCE_FAILURE = "source_failure"
    SOURCE_SUCCESS = "source_success"


class TerminalCaseTermination(StrEnum):
    """Observable point at which one Terminal case ended."""

    ACTOR_TIMEOUT = "actor_timeout"
    IDENTITY_INVALID = "identity_invalid"
    MISSING_REWARD = "missing_reward"
    OFFICIAL_REWARD = "official_reward"
    RUNNER_ERROR = "runner_error"
    RUNTIME_ERROR = "runtime_error"


class TerminalFailureScope(StrEnum):
    """Whether a failure is local to one task or shared by concurrent tasks."""

    NONE = "none"
    TASK = "task"
    SHARED_ENVIRONMENT = "shared_environment"
    SHARED_PROVIDER = "shared_provider"


class TerminalFailureCause(StrEnum):
    """Stable failure categories used by orchestration policy and audit logs."""

    NONE = "none"
    ACTOR_TIMEOUT = "actor_timeout"
    DOCKER_NETWORK_POOL_EXHAUSTED = "docker_network_pool_exhausted"
    DOCKER_REGISTRY_UNAVAILABLE = "docker_registry_unavailable"
    ENVIRONMENT_TIMEOUT = "environment_timeout"
    IDENTITY_INVALID = "identity_invalid"
    MODEL_TRUNCATION = "model_truncation"
    PROVIDER_CONNECTION = "provider_connection"
    PROVIDER_OTHER = "provider_other"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RUNNER_ERROR = "runner_error"
    RUNTIME_ERROR = "runtime_error"
    VERIFIER_TIMEOUT = "verifier_timeout"


@dataclass(frozen=True)
class TerminalFailureDiagnosis:
    """Typed root cause consumed by concurrency and shared-outage policy."""

    scope: TerminalFailureScope
    cause: TerminalFailureCause


def is_pre_actor_infrastructure_invalid(record: Mapping[str, object]) -> bool:
    """Recognize a case receipt produced before an actor could start."""

    identity_fields = ("provider", "model", "transport", "thinking", "reasoning_tokens")
    return (
        record.get("classification") == TerminalCaseClassification.INFRASTRUCTURE_INVALID
        and record.get("tool_call_count") in {None, 0}
        and all(record.get(field) is None for field in identity_fields)
        and record.get("token_usage") is None
        and isinstance(record.get("runtime_error"), str)
        and bool(record["runtime_error"])
    )


def source_host_pi_timeout_seconds(
    *,
    task_agent_timeout_sec: int,
    configured_minimum_seconds: int,
    grace_seconds: int = HOST_PI_OUTER_TIMEOUT_GRACE_SECONDS,
) -> int:
    """Keep the host safety timeout outside the official Harbor actor timeout."""

    if min(task_agent_timeout_sec, configured_minimum_seconds, grace_seconds) < 1:
        raise ValueError("Terminal timeout inputs must be positive")
    timeout = max(
        configured_minimum_seconds,
        task_agent_timeout_sec + grace_seconds,
    )
    if timeout > MAX_HOST_PI_TIMEOUT_SECONDS:
        raise ValueError("Terminal host Pi timeout exceeds the supported safety bound")
    return timeout


def diagnose_terminal_failure(
    *,
    exception_type: str | None,
    exception_message: str | None,
    provider_failure_terminal: bool,
    provider_failure_kind: str | None,
) -> TerminalFailureDiagnosis:
    """Translate source-specific evidence once before orchestration decisions."""

    if provider_failure_terminal:
        provider_causes = {
            "connection": TerminalFailureCause.PROVIDER_CONNECTION,
            "model_truncation": TerminalFailureCause.MODEL_TRUNCATION,
            "rate_limit": TerminalFailureCause.PROVIDER_RATE_LIMIT,
            "service_unavailable": TerminalFailureCause.PROVIDER_UNAVAILABLE,
            "timeout": TerminalFailureCause.PROVIDER_TIMEOUT,
        }
        cause = provider_causes.get(
            provider_failure_kind or "",
            TerminalFailureCause.PROVIDER_OTHER,
        )
        shared = cause in {
            TerminalFailureCause.PROVIDER_CONNECTION,
            TerminalFailureCause.PROVIDER_RATE_LIMIT,
            TerminalFailureCause.PROVIDER_TIMEOUT,
            TerminalFailureCause.PROVIDER_UNAVAILABLE,
        }
        return TerminalFailureDiagnosis(
            TerminalFailureScope.SHARED_PROVIDER if shared else TerminalFailureScope.TASK,
            cause,
        )

    lowered = (exception_message or "").lower()
    if "registry-1.docker.io" in lowered and any(
        marker in lowered
        for marker in ("service unavailable", "tls handshake timeout", "toomanyrequests")
    ):
        return TerminalFailureDiagnosis(
            TerminalFailureScope.SHARED_ENVIRONMENT,
            TerminalFailureCause.DOCKER_REGISTRY_UNAVAILABLE,
        )
    if "all predefined address pools have been fully subnetted" in lowered:
        # Docker cannot allocate any new sandbox network, so later tasks would fail too.
        return TerminalFailureDiagnosis(
            TerminalFailureScope.SHARED_ENVIRONMENT,
            TerminalFailureCause.DOCKER_NETWORK_POOL_EXHAUSTED,
        )
    if exception_type == "AgentTimeoutError":
        return TerminalFailureDiagnosis(
            TerminalFailureScope.TASK,
            TerminalFailureCause.ACTOR_TIMEOUT,
        )
    if exception_type == "EnvironmentStartTimeoutError":
        return TerminalFailureDiagnosis(
            TerminalFailureScope.TASK,
            TerminalFailureCause.ENVIRONMENT_TIMEOUT,
        )
    if exception_type == "VerifierTimeoutError":
        return TerminalFailureDiagnosis(
            TerminalFailureScope.TASK,
            TerminalFailureCause.VERIFIER_TIMEOUT,
        )
    if exception_type:
        return TerminalFailureDiagnosis(
            TerminalFailureScope.TASK,
            TerminalFailureCause.RUNTIME_ERROR,
        )
    return TerminalFailureDiagnosis(TerminalFailureScope.NONE, TerminalFailureCause.NONE)


@dataclass(frozen=True)
class TerminalCensusTask:
    name: str
    difficulty: str
    category: str
    cpus: int
    memory_mb: int
    agent_timeout_sec: int

    def record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TerminalCensusCaseDecision:
    """Separate budgeted actor outcomes from failures of the evaluation machinery."""

    classification: TerminalCaseClassification
    termination: TerminalCaseTermination


def classify_terminal_census_case(
    *,
    return_code: int | None,
    exception_type: str | None,
    reward: float | None,
    identity_valid: bool,
    terminal_provider_failure: bool = False,
) -> TerminalCensusCaseDecision:
    """Classify a case using official reward plus an explicit actor-timeout outcome."""
    if return_code not in (0, None):
        return TerminalCensusCaseDecision(
            TerminalCaseClassification.INFRASTRUCTURE_INVALID,
            TerminalCaseTermination.RUNNER_ERROR,
        )
    if terminal_provider_failure:
        return TerminalCensusCaseDecision(
            TerminalCaseClassification.INFRASTRUCTURE_INVALID,
            TerminalCaseTermination.RUNTIME_ERROR,
        )
    if exception_type == "AgentTimeoutError":
        if not identity_valid:
            return TerminalCensusCaseDecision(
                TerminalCaseClassification.INFRASTRUCTURE_INVALID,
                TerminalCaseTermination.IDENTITY_INVALID,
            )
        classification = (
            TerminalCaseClassification.SOURCE_SUCCESS
            if reward == 1.0
            else TerminalCaseClassification.SOURCE_FAILURE
        )
        return TerminalCensusCaseDecision(
            classification,
            TerminalCaseTermination.ACTOR_TIMEOUT,
        )
    if exception_type is not None:
        return TerminalCensusCaseDecision(
            TerminalCaseClassification.INFRASTRUCTURE_INVALID,
            TerminalCaseTermination.RUNTIME_ERROR,
        )
    if not identity_valid:
        return TerminalCensusCaseDecision(
            TerminalCaseClassification.INFRASTRUCTURE_INVALID,
            TerminalCaseTermination.IDENTITY_INVALID,
        )
    if reward is None:
        return TerminalCensusCaseDecision(
            TerminalCaseClassification.INFRASTRUCTURE_INVALID,
            TerminalCaseTermination.MISSING_REWARD,
        )
    classification = (
        TerminalCaseClassification.SOURCE_SUCCESS
        if reward == 1.0
        else TerminalCaseClassification.SOURCE_FAILURE
    )
    return TerminalCensusCaseDecision(
        classification,
        TerminalCaseTermination.OFFICIAL_REWARD,
    )


def load_terminal_census_population(
    terminal_bench_root: Path,
    *,
    expected_task_count: int,
) -> tuple[TerminalCensusTask, ...]:
    """Freeze the complete official task directory before any model call."""
    task_root = terminal_bench_root / "tasks"
    task_directories = tuple(
        sorted(
            (path for path in task_root.iterdir() if (path / "task.toml").is_file()),
            key=lambda path: path.name,
        )
    )
    if len(task_directories) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} Terminal tasks, found {len(task_directories)}"
        )

    population = []
    for task_directory in task_directories:
        config = tomllib.loads((task_directory / "task.toml").read_text())
        metadata = config["metadata"]
        environment = config["environment"]
        agent = config["agent"]
        population.append(
            TerminalCensusTask(
                name=task_directory.name,
                difficulty=str(metadata["difficulty"]),
                category=str(metadata["category"]),
                cpus=int(environment["cpus"]),
                memory_mb=int(environment["memory_mb"]),
                agent_timeout_sec=int(agent["timeout_sec"]),
            )
        )
    return tuple(population)


def select_terminal_census_population(
    population: tuple[TerminalCensusTask, ...],
    *,
    task_names: tuple[str, ...],
) -> tuple[TerminalCensusTask, ...]:
    """Select an explicit smoke subset without changing the frozen source population."""
    if not task_names:
        return population
    if len(set(task_names)) != len(task_names):
        raise ValueError("Terminal smoke task names must be unique")
    requested = set(task_names)
    selected = tuple(task for task in population if task.name in requested)
    missing = requested.difference(task.name for task in selected)
    if missing:
        raise ValueError(f"unknown Terminal task: {sorted(missing)[0]}")
    return selected
