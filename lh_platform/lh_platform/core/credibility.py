"""Run Credibility primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from lh_platform.core.execution_paths import (
    EXECUTION_AUTO,
    EXECUTION_BENCHMARK_STEP_LOOP,
    EXECUTION_MIXED,
    EXECUTION_PROMPT_WRAPPER,
    EXECUTION_UPSTREAM_NATIVE_RUNNER,
)
from lh_platform.core.execution_paths import (
    EXECUTION_TEST_TIME_OPTIMIZER as EXECUTION_TEST_TIME_OPTIMIZER,
)
from lh_platform.core.execution_paths import (
    EXECUTION_TRAINED_ARTIFACT_EVAL as EXECUTION_TRAINED_ARTIFACT_EVAL,
)
from lh_platform.core.execution_paths import (
    EXECUTION_UPSTREAM_TRAINING_RECIPE as EXECUTION_UPSTREAM_TRAINING_RECIPE,
)
from lh_platform.core.method_capabilities import canonical_integration_level
from lh_platform.core.outcome import EVIDENCE_STRONG, OutcomeEvidence
from lh_platform.core.types import Role, Trajectory


@dataclass
class RunCredibility:
    """Explanation of how trustworthy a task run is as comparison evidence."""

    benchmark_static_capability: str
    method_static_capability: str
    requested_execution: str
    actual_dispatch: str
    outcome_evidence_quality: str
    execution_failed: bool
    comparable: bool
    run_failed: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def requested_execution_from_kwargs(method_kwargs: dict) -> str:
    requested = str(method_kwargs.get("execution_path") or "").strip()
    if requested:
        return requested
    mode = str(method_kwargs.get("mode") or "").strip().lower()
    if mode == "native":
        return EXECUTION_UPSTREAM_NATIVE_RUNNER
    if mode == "prompt":
        return EXECUTION_PROMPT_WRAPPER
    return EXECUTION_AUTO


def actual_dispatch_from_trajectory(method_integration_level: str, trajectory: Trajectory) -> str:
    method_integration_level = canonical_integration_level(method_integration_level)
    declared_paths = _declared_execution_paths(trajectory)
    if len(declared_paths) > 1:
        return EXECUTION_MIXED
    if declared_paths:
        return next(iter(declared_paths))
    # Static capability is only a compatibility fallback for legacy native
    # adapters that predate per-action execution receipts.  It must never
    # overwrite evidence emitted by the actual trajectory.
    if method_integration_level == "native_runner":
        return EXECUTION_UPSTREAM_NATIVE_RUNNER
    action_types = _action_types(trajectory)
    if not action_types or action_types == {"noop"}:
        return EXECUTION_BENCHMARK_STEP_LOOP
    return EXECUTION_PROMPT_WRAPPER


def execution_failure_from_trajectory(trajectory: Trajectory) -> bool:
    """Whether Runner recorded a method/benchmark execution interruption.

    An environment score only has its intended meaning after the policy loop
    finished normally.  A method crash after a non-terminal environment action
    is not an authoritative score-0 failure sample; it is a failed execution.
    Both the structured trajectory marker and legacy ERROR steps are checked
    so previously implemented runner paths cannot silently bypass this rule.
    """

    marker = trajectory.info.get("execution_failure")
    if isinstance(marker, dict) and marker:
        return True
    return any(step.role == Role.ERROR for step in trajectory.steps)


def build_run_credibility(
    *,
    benchmark_eval_validity: str,
    method_integration_level: str,
    requested_execution: str,
    actual_dispatch: str,
    outcome_evidence: OutcomeEvidence,
    benchmark_finalization_failed: bool = False,
    execution_failed: bool = False,
) -> RunCredibility:
    normalized_method_integration_level = canonical_integration_level(method_integration_level)
    run_failed = (
        bool(benchmark_finalization_failed)
        or bool(execution_failed)
        or actual_dispatch == EXECUTION_MIXED
        or _explicit_execution_mismatch(requested_execution, actual_dispatch)
    )
    comparable = (
        not run_failed
        and outcome_evidence.evidence_quality == EVIDENCE_STRONG
        and benchmark_eval_validity == "real_native"
        and normalized_method_integration_level == "native_runner"
    )
    reason = _reason(
        benchmark_eval_validity=benchmark_eval_validity,
        method_integration_level=normalized_method_integration_level,
        requested_execution=requested_execution,
        actual_dispatch=actual_dispatch,
        outcome_evidence=outcome_evidence,
        comparable=comparable,
        run_failed=run_failed,
        benchmark_finalization_failed=benchmark_finalization_failed,
        execution_failed=execution_failed,
    )
    return RunCredibility(
        benchmark_static_capability=benchmark_eval_validity,
        method_static_capability=normalized_method_integration_level,
        requested_execution=requested_execution,
        actual_dispatch=actual_dispatch,
        outcome_evidence_quality=outcome_evidence.evidence_quality,
        execution_failed=bool(execution_failed),
        comparable=comparable,
        run_failed=run_failed,
        reason=reason,
    )


def _action_types(trajectory: Trajectory) -> set[str]:
    out: set[str] = set()
    for step in trajectory.steps:
        if step.action is not None and step.action.action_type:
            out.add(str(step.action.info.get("original_action_type") or step.action.action_type))
    return out


def _declared_execution_paths(trajectory: Trajectory) -> set[str]:
    out: set[str] = set()
    for step in trajectory.steps:
        if step.action is None:
            continue
        path = step.action.info.get("execution_path")
        if path:
            out.add(str(path))
    return out


def _explicit_execution_mismatch(requested_execution: str, actual_dispatch: str) -> bool:
    if requested_execution == EXECUTION_AUTO:
        return False
    return requested_execution != actual_dispatch


def _reason(
    *,
    benchmark_eval_validity: str,
    method_integration_level: str,
    requested_execution: str,
    actual_dispatch: str,
    outcome_evidence: OutcomeEvidence,
    comparable: bool,
    run_failed: bool,
    benchmark_finalization_failed: bool,
    execution_failed: bool,
) -> str:
    if benchmark_finalization_failed:
        return "benchmark finalization failed; run evidence is invalid"
    if execution_failed:
        return "task execution failed before an authoritative terminal outcome"
    if actual_dispatch == EXECUTION_MIXED:
        return "mixed execution paths were recorded in one trajectory"
    if run_failed:
        return (
            f"requested execution {requested_execution!r}, "
            f"but actual dispatch was {actual_dispatch!r}"
        )
    if benchmark_eval_validity == "smoke_only":
        return "not comparable: benchmark is smoke_only"
    if method_integration_level == "mock":
        return "not comparable: method integration_level is mock"
    if outcome_evidence.evidence_quality != EVIDENCE_STRONG:
        return (
            f"not comparable: Outcome Evidence is not strong ({outcome_evidence.evidence_quality})"
        )
    if benchmark_eval_validity != "real_native":
        return f"not comparable: benchmark is not real_native ({benchmark_eval_validity})"
    if method_integration_level != "native_runner":
        return (
            "not comparable: method integration_level is not native_runner "
            f"({method_integration_level})"
        )
    if comparable:
        return "comparable under recorded static capability, dispatch, and Outcome Evidence"
    return "not comparable under recorded credibility evidence"
