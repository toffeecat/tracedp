"""Compile Runtime-owned two-step options from atomic outcome estimates.

The estimator may value named outcomes for actions in an immutable slate. It
does not create states, actions, or follow-up branches. Runtime derives those
objects from public predicate, incident, and task-dependency contracts before
calling the existing time-aware Bellman planner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeConditionalFutureOption,
    RuntimeJointActionCandidate,
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    RuntimeJointResolvedChoice,
)
from lh_platform.test_time_adaptation.runtime_joint_trace_dp import (
    RuntimeJointTransitionOutcome,
)
from lh_platform.test_time_adaptation.runtime_time_aware_trace_dp import (
    RuntimeTimeAwarePolicyValue,
    RuntimeTimeAwareTraceDPPlanner,
    RuntimeTimedActionTransition,
    RuntimeTimedPlanningState,
)

_PREDICATE_OUTCOMES = frozenset({"supported", "refuted", "unknown"})
_CONTINUE_OUTCOMES = frozenset(
    {"acceptance_supported", "progress", "no_progress"}
)
_SUBMIT_OUTCOMES = frozenset({"submitted"})
_SUPPORTED_ACTION_TYPES = frozenset(
    {
        RuntimeJointActionType.CONTINUE_ACTOR,
        RuntimeJointActionType.RUN_REGISTERED_CHECK,
        RuntimeJointActionType.RUN_TARGETED_REPAIR,
        RuntimeJointActionType.SUBMIT_CURRENT_STATE,
    }
)
_UNBLOCKED_TASK_STATUSES = frozenset(
    {"ready", "running", "awaiting_evaluation", "completed"}
)


@dataclass(frozen=True)
class RuntimeOptionAcceptancePredicate:
    """Describe one public residual predicate and its current incident."""

    predicate_id: str
    task_id: str
    proposition: str
    evidence_state: str
    incident_id: str | None
    incident_status: str | None

    def __post_init__(self) -> None:
        if (
            not self.predicate_id
            or not self.task_id
            or not self.proposition.strip()
            or self.evidence_state not in {"supported", "refuted", "unknown", "stale"}
        ):
            raise ValueError("runtime option acceptance predicate is invalid")
        if (self.incident_id is None) != (self.incident_status is None):
            raise ValueError("runtime option incident identity is incomplete")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> RuntimeOptionAcceptancePredicate:
        expected = {
            "predicate_id",
            "task_id",
            "proposition",
            "evidence_state",
            "incident_id",
            "incident_status",
        }
        if set(payload) != expected:
            raise ValueError("runtime option acceptance predicate fields changed")
        return cls(
            predicate_id=str(payload["predicate_id"]),
            task_id=str(payload["task_id"]),
            proposition=str(payload["proposition"]),
            evidence_state=str(payload["evidence_state"]),
            incident_id=(
                str(payload["incident_id"])
                if payload["incident_id"] is not None
                else None
            ),
            incident_status=(
                str(payload["incident_status"])
                if payload["incident_status"] is not None
                else None
            ),
        )


@dataclass(frozen=True)
class RuntimeOptionTaskDependency:
    """Retain the public parent relation and current task status."""

    task_id: str
    parent_task_id: str | None
    status: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.status or self.parent_task_id == self.task_id:
            raise ValueError("runtime option task dependency is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeOptionTaskDependency:
        if set(payload) != {"task_id", "parent_task_id", "status"}:
            raise ValueError("runtime option task dependency fields changed")
        return cls(
            task_id=str(payload["task_id"]),
            parent_task_id=(
                str(payload["parent_task_id"])
                if payload["parent_task_id"] is not None
                else None
            ),
            status=str(payload["status"]),
        )


@dataclass(frozen=True)
class RuntimeOwnedOptionProjection:
    """Bind planning inputs to one public Runtime boundary."""

    session_id: str
    resource_versions: tuple[tuple[str, int], ...]
    acceptance_predicates: tuple[RuntimeOptionAcceptancePredicate, ...]
    task_dependencies: tuple[RuntimeOptionTaskDependency, ...]

    schema_version = "runtime_owned_option_projection_v1"

    def __post_init__(self) -> None:
        predicate_ids = [item.predicate_id for item in self.acceptance_predicates]
        task_ids = [item.task_id for item in self.task_dependencies]
        if (
            not self.session_id
            or not self.acceptance_predicates
            or len(predicate_ids) != len(set(predicate_ids))
            or len(task_ids) != len(set(task_ids))
            or tuple(sorted(self.resource_versions)) != self.resource_versions
        ):
            raise ValueError("runtime-owned option projection is invalid")
        known_tasks = set(task_ids)
        if any(
            item.task_id not in known_tasks for item in self.acceptance_predicates
        ) or any(
            item.parent_task_id is not None
            and item.parent_task_id not in known_tasks
            for item in self.task_dependencies
        ):
            raise ValueError("runtime-owned option projection references an unknown task")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "resource_versions": [list(item) for item in self.resource_versions],
            "acceptance_predicates": [
                item.to_dict() for item in self.acceptance_predicates
            ],
            "task_dependencies": [item.to_dict() for item in self.task_dependencies],
        }

    @classmethod
    def from_public_projection(
        cls,
        projection: Mapping[str, object],
        *,
        residual_acceptance_state: tuple[Mapping[str, object], ...],
    ) -> RuntimeOwnedOptionProjection:
        """Compile typed planning state from the Kernel's public projection."""

        if projection.get("schema_version") != "portable_runtime_public_projection_v6":
            raise ValueError("runtime-owned option requires public projection v6")
        raw_versions = projection.get("resource_versions")
        raw_tasks = projection.get("task_nodes")
        raw_incidents = projection.get("predicate_incidents", [])
        if (
            not isinstance(raw_versions, Mapping)
            or not isinstance(raw_tasks, list)
            or not isinstance(raw_incidents, list)
            or any(not isinstance(item, Mapping) for item in raw_tasks)
            or any(not isinstance(item, Mapping) for item in raw_incidents)
        ):
            raise ValueError("runtime-owned option public collections are invalid")
        predicates = []
        for row in residual_acceptance_state:
            task_id = row.get("task_id")
            proposition = row.get("proposition")
            predicate_id = row.get("predicate_id")
            evidence_state = row.get("latest_evidence_state") or "unknown"
            latest_evidence_id = row.get("latest_evidence_id")
            if not all(
                isinstance(value, str) and value
                for value in (task_id, proposition, predicate_id, evidence_state)
            ):
                raise ValueError("runtime-owned option residual predicate is invalid")
            incidents = [
                item
                for item in raw_incidents
                if item.get("task_id") == task_id
                and item.get("proposition") == proposition
                and (
                    latest_evidence_id is None
                    or item.get("latest_evidence_id") == latest_evidence_id
                )
            ]
            if len(incidents) > 1:
                raise ValueError("runtime-owned option predicate has multiple incidents")
            incident = incidents[0] if incidents else None
            predicates.append(
                RuntimeOptionAcceptancePredicate(
                    predicate_id=predicate_id,
                    task_id=task_id,
                    proposition=proposition,
                    evidence_state=evidence_state,
                    incident_id=(
                        str(incident["incident_id"]) if incident is not None else None
                    ),
                    incident_status=(
                        str(incident["status"]) if incident is not None else None
                    ),
                )
            )
        return cls(
            session_id=str(projection["session_id"]),
            resource_versions=tuple(
                sorted((str(key), int(value)) for key, value in raw_versions.items())
            ),
            acceptance_predicates=tuple(predicates),
            task_dependencies=tuple(
                RuntimeOptionTaskDependency(
                    task_id=str(item["task_id"]),
                    parent_task_id=(
                        str(item["parent_id"])
                        if item.get("parent_id") is not None
                        else None
                    ),
                    status=str(item["status"]),
                )
                for item in raw_tasks
            ),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeOwnedOptionProjection:
        expected = {
            "schema_version",
            "session_id",
            "resource_versions",
            "acceptance_predicates",
            "task_dependencies",
        }
        predicates = payload.get("acceptance_predicates")
        dependencies = payload.get("task_dependencies")
        versions = payload.get("resource_versions")
        if (
            set(payload) != expected
            or payload.get("schema_version") != cls.schema_version
            or not isinstance(predicates, list)
            or not isinstance(dependencies, list)
            or not isinstance(versions, list)
            or any(not isinstance(item, Mapping) for item in predicates)
            or any(not isinstance(item, Mapping) for item in dependencies)
        ):
            raise ValueError("runtime-owned option projection payload is invalid")
        return cls(
            session_id=str(payload["session_id"]),
            resource_versions=tuple((str(item[0]), int(item[1])) for item in versions),
            acceptance_predicates=tuple(
                RuntimeOptionAcceptancePredicate.from_dict(item)
                for item in predicates
            ),
            task_dependencies=tuple(
                RuntimeOptionTaskDependency.from_dict(item) for item in dependencies
            ),
        )


@dataclass(frozen=True)
class RuntimeAtomicOutcomeEstimate:
    """Assign probability to one action-type-specific atomic outcome."""

    outcome_id: str
    probability: float

    def __post_init__(self) -> None:
        if (
            not self.outcome_id
            or not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError("runtime atomic outcome estimate is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeAtomicActionEstimate:
    """Value named outcomes without predicting a state or future action."""

    instance_action_id: str
    duration_ms: int
    provenance_id: str
    provenance_detail: str
    outcomes: tuple[RuntimeAtomicOutcomeEstimate, ...]

    def __post_init__(self) -> None:
        outcome_ids = [item.outcome_id for item in self.outcomes]
        if (
            not self.instance_action_id
            or isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 1
            or not self.provenance_id
            or not self.provenance_detail.strip()
            or not self.outcomes
            or len(outcome_ids) != len(set(outcome_ids))
            or not math.isclose(
                sum(item.probability for item in self.outcomes),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("runtime atomic action estimate is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_action_id": self.instance_action_id,
            "duration_ms": self.duration_ms,
            "provenance_id": self.provenance_id,
            "provenance_detail": self.provenance_detail,
            "outcomes": [item.to_dict() for item in self.outcomes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeAtomicActionEstimate:
        expected = {
            "instance_action_id",
            "duration_ms",
            "provenance_id",
            "provenance_detail",
            "outcomes",
        }
        outcomes = payload.get("outcomes")
        if (
            set(payload) != expected
            or not isinstance(outcomes, list)
            or any(not isinstance(item, Mapping) for item in outcomes)
        ):
            raise ValueError("runtime atomic action estimate payload is invalid")
        return cls(
            instance_action_id=str(payload["instance_action_id"]),
            duration_ms=int(payload["duration_ms"]),
            provenance_id=str(payload["provenance_id"]),
            provenance_detail=str(payload["provenance_detail"]),
            outcomes=tuple(
                RuntimeAtomicOutcomeEstimate(
                    outcome_id=str(item["outcome_id"]),
                    probability=float(item["probability"]),
                )
                for item in outcomes
            ),
        )


@dataclass(frozen=True)
class _ProjectedNode:
    state: RuntimeTimedPlanningState
    predicate_states: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _CompiledOptionModel:
    states: tuple[RuntimeTimedPlanningState, ...]
    transitions: tuple[RuntimeTimedActionTransition, ...]
    transition_rows: tuple[dict[str, object], ...]
    immediate_baseline_action_id: str


_PlanningAction = RuntimeJointActionCandidate | RuntimeConditionalFutureOption


def _allowed_outcomes(action_type: RuntimeJointActionType) -> frozenset[str]:
    if action_type in {
        RuntimeJointActionType.RUN_REGISTERED_CHECK,
        RuntimeJointActionType.RUN_TARGETED_REPAIR,
    }:
        return _PREDICATE_OUTCOMES
    if action_type is RuntimeJointActionType.CONTINUE_ACTOR:
        return _CONTINUE_OUTCOMES
    if action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE:
        return _SUBMIT_OUTCOMES
    raise ValueError(f"runtime-owned option action type is unsupported: {action_type}")


def _predicate_for_candidate(
    candidate: _PlanningAction,
    projection: RuntimeOwnedOptionProjection,
) -> RuntimeOptionAcceptancePredicate | None:
    if candidate.action_type not in {
        RuntimeJointActionType.RUN_REGISTERED_CHECK,
        RuntimeJointActionType.RUN_TARGETED_REPAIR,
    }:
        return None
    contract = candidate.execution_contract
    task_id = candidate.target_binding.task_id
    if contract is None or task_id is None:
        raise ValueError("predicate action lacks its execution contract")
    matches = [
        item
        for item in projection.acceptance_predicates
        if item.task_id == task_id and item.proposition == contract.target_predicate
    ]
    if len(matches) != 1:
        raise ValueError("predicate action does not bind exactly one acceptance predicate")
    predicate = matches[0]
    if (
        candidate.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR
        and isinstance(candidate, RuntimeJointActionCandidate)
    ):
        arguments = contract.proposed_check_or_repair["arguments"]
        if (
            predicate.incident_id is None
            or predicate.incident_status not in {"open", "recheck_required"}
            or arguments.get("incident_id") != predicate.incident_id
        ):
            raise ValueError("targeted repair differs from the current predicate incident")
    return predicate


def _dependency_progress_fraction(
    dependencies: tuple[RuntimeOptionTaskDependency, ...],
    predicates: tuple[RuntimeOptionAcceptancePredicate, ...],
    predicate_states: tuple[tuple[str, str], ...],
) -> float:
    """Propagate predicted predicate progress through the public parent graph."""

    if not dependencies:
        return 0.0
    states = dict(predicate_states)
    predicates_by_task: dict[str, list[str]] = {
        item.task_id: [] for item in dependencies
    }
    children: dict[str, list[str]] = {item.task_id: [] for item in dependencies}
    dependency_by_id = {item.task_id: item for item in dependencies}
    for predicate in predicates:
        predicates_by_task[predicate.task_id].append(predicate.predicate_id)
    for dependency in dependencies:
        if dependency.parent_task_id is not None:
            children[dependency.parent_task_id].append(dependency.task_id)

    cache: dict[str, float] = {}

    def resolve(task_id: str) -> float:
        cached = cache.get(task_id)
        if cached is not None:
            return cached
        child_ids = children[task_id]
        predicate_ids = predicates_by_task[task_id]
        if predicate_ids:
            own = sum(states[item] == "supported" for item in predicate_ids) / len(
                predicate_ids
            )
        elif child_ids:
            own = 1.0
        else:
            own = float(
                dependency_by_id[task_id].status in _UNBLOCKED_TASK_STATUSES
            )
        child_progress = (
            sum(resolve(child_id) for child_id in child_ids) / len(child_ids)
            if child_ids
            else 1.0
        )
        cache[task_id] = own * child_progress
        return cache[task_id]

    return sum(resolve(item.task_id) for item in dependencies) / len(dependencies)


def _outcome_node(
    *,
    source: _ProjectedNode,
    candidate: _PlanningAction,
    outcome_id: str,
    state_id: str,
    duration_ms: int,
    projection: RuntimeOwnedOptionProjection,
) -> _ProjectedNode:
    predicate_states = dict(source.predicate_states)
    predicate = _predicate_for_candidate(candidate, projection)
    if predicate is not None:
        predicate_states[predicate.predicate_id] = outcome_id
    residual_complete = bool(predicate_states) and all(
        state == "supported" for state in predicate_states.values()
    )
    if (
        candidate.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        and outcome_id == "acceptance_supported"
    ):
        # Runtime interprets this fixed outcome as all public residual predicates
        # becoming supported; the estimator cannot provide an arbitrary next state.
        predicate_states = {key: "supported" for key in predicate_states}
        residual_complete = bool(predicate_states)
    if candidate.action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE:
        # Submission hands state to the independent evaluator. This terminal flag
        # records only whether public acceptance was already supported.
        residual_complete = bool(predicate_states) and all(
            state == "supported" for state in predicate_states.values()
        )
    actor_step_cost = int(
        candidate.action_type is not RuntimeJointActionType.SUBMIT_CURRENT_STATE
    )
    return _ProjectedNode(
        state=RuntimeTimedPlanningState(
            state_id=state_id,
            public_goal_complete=residual_complete,
            unblocked_fraction=max(
                source.state.unblocked_fraction,
                _dependency_progress_fraction(
                    projection.task_dependencies,
                    projection.acceptance_predicates,
                    tuple(predicate_states.items()),
                ),
            ),
            remaining_intervention_budget=(
                source.state.remaining_intervention_budget - candidate.cost_units
            ),
            remaining_actor_steps=source.state.remaining_actor_steps - actor_step_cost,
            remaining_time_ms=source.state.remaining_time_ms - duration_ms,
            residual_predicate_states=tuple(predicate_states.items()),
            outcome_phase=(
                "submitted_to_independent_evaluator"
                if candidate.action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE
                else outcome_id
            ),
        ),
        predicate_states=tuple(predicate_states.items()),
    )


def _matching_repairs(
    *,
    candidate: RuntimeJointActionCandidate,
    slate: RuntimeJointActionSlate,
) -> tuple[_PlanningAction, ...]:
    contract = candidate.execution_contract
    if contract is None:
        return ()
    executable = tuple(
        item
        for item in slate.candidates
        if item.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR
        and item.instance_action_id != candidate.instance_action_id
        and item.target_binding.task_id == candidate.target_binding.task_id
        and item.execution_contract is not None
        and item.execution_contract.target_predicate == contract.target_predicate
    )
    conditional = tuple(
        item
        for item in slate.conditional_future_options
        if item.prerequisite_action_id == candidate.instance_action_id
        and item.target_binding.task_id == candidate.target_binding.task_id
        and item.execution_contract.target_predicate == contract.target_predicate
    )
    return (*executable, *conditional)


def _followups(
    *,
    root_candidate: RuntimeJointActionCandidate,
    root_outcome_id: str,
    slate: RuntimeJointActionSlate,
) -> tuple[_PlanningAction, ...]:
    continuation = next(
        item
        for item in slate.candidates
        if item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
    )
    if root_candidate.action_type is RuntimeJointActionType.RUN_REGISTERED_CHECK:
        if root_outcome_id not in {"refuted", "unknown"}:
            return (continuation,)
        repairs = tuple(
            item
            for item in _matching_repairs(candidate=root_candidate, slate=slate)
            if not isinstance(item, RuntimeConditionalFutureOption)
            or root_outcome_id in item.prerequisite_outcomes
        )
        return (continuation, *repairs)
    if root_candidate.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR:
        if root_outcome_id in {"refuted", "unknown"}:
            return (continuation, *_matching_repairs(candidate=root_candidate, slate=slate))
        return (continuation,)
    return ()


def _immediate_value(
    transition: RuntimeTimedActionTransition,
    states: Mapping[str, RuntimeTimedPlanningState],
) -> RuntimeTimeAwarePolicyValue:
    """Calculate a one-action baseline directly from Runtime-generated leaves."""

    branches = tuple(
        (outcome.probability, states[outcome.next_state_id])
        for outcome in transition.outcomes
    )
    return RuntimeTimeAwarePolicyValue(
        public_success_probability=sum(
            probability * float(state.public_goal_complete)
            for probability, state in branches
        ),
        expected_unblocked_fraction=sum(
            probability * state.unblocked_fraction
            for probability, state in branches
        ),
        expected_duration_ms=float(transition.duration_ms),
        expected_false_interventions=0.0,
        expected_intervention_cost=float(transition.intervention_cost_units),
    )


def _compile_model(
    *,
    slate: RuntimeJointActionSlate,
    projection: RuntimeOwnedOptionProjection,
    estimates: tuple[RuntimeAtomicActionEstimate, ...],
    horizon: int = 2,
) -> _CompiledOptionModel:
    if horizon not in {1, 2}:
        raise ValueError("runtime-owned option horizon is unsupported")
    if (
        projection.session_id != slate.session_id
        or projection.resource_versions != slate.resource_versions
        or slate.actor_time_remaining_ms is None
    ):
        raise ValueError("runtime-owned option projection differs from the live slate")
    candidates = {item.instance_action_id: item for item in slate.candidates}
    options = {item.option_id: item for item in slate.conditional_future_options}
    estimate_by_id = {item.instance_action_id: item for item in estimates}
    universe = {**candidates, **options}
    if len(estimate_by_id) != len(estimates) or set(estimate_by_id) != set(universe):
        raise ValueError("atomic estimates must cover the exact immutable candidate universe")
    for candidate in universe.values():
        if candidate.action_type not in _SUPPORTED_ACTION_TYPES:
            raise ValueError("runtime-owned option slate contains an unsupported action")
        if (
            isinstance(candidate, RuntimeJointActionCandidate)
            and candidate.cost_units > slate.budget_before
        ):
            raise ValueError("runtime-owned option slate contains an over-budget root action")
        outcome_ids = {item.outcome_id for item in estimate_by_id[candidate.instance_action_id].outcomes}
        if outcome_ids != _allowed_outcomes(candidate.action_type):
            raise ValueError("atomic outcome names differ from the action contract")
        _predicate_for_candidate(candidate, projection)

    root_predicates = tuple(
        (item.predicate_id, item.evidence_state)
        for item in projection.acceptance_predicates
    )
    root = _ProjectedNode(
        state=RuntimeTimedPlanningState(
            state_id="root",
            public_goal_complete=False,
            unblocked_fraction=_dependency_progress_fraction(
                projection.task_dependencies,
                projection.acceptance_predicates,
                root_predicates,
            ),
            remaining_intervention_budget=slate.budget_before,
            remaining_actor_steps=horizon,
            remaining_time_ms=slate.actor_time_remaining_ms,
            residual_predicate_states=root_predicates,
            outcome_phase="current_runtime_boundary",
        ),
        predicate_states=root_predicates,
    )
    states = [root.state]
    transitions: list[RuntimeTimedActionTransition] = []
    rows: list[dict[str, object]] = []

    def add_transition(
        *,
        source: _ProjectedNode,
        candidate: _PlanningAction,
        estimate: RuntimeAtomicActionEstimate,
        depth: int,
        source_outcome: str,
        ordinal: int,
    ) -> tuple[_ProjectedNode, ...]:
        if (
            candidate.cost_units > source.state.remaining_intervention_budget
            or estimate.duration_ms > source.state.remaining_time_ms
            or (
                candidate.action_type is not RuntimeJointActionType.SUBMIT_CURRENT_STATE
                and source.state.remaining_actor_steps < 1
            )
        ):
            return ()
        targets = tuple(
            _outcome_node(
                source=source,
                candidate=candidate,
                outcome_id=outcome.outcome_id,
                state_id=(
                    f"d{depth}:{source.state.state_id}:{ordinal}:"
                    f"{candidate.instance_action_id}:"
                    f"{outcome.outcome_id}"
                ),
                duration_ms=estimate.duration_ms,
                projection=projection,
            )
            for outcome in estimate.outcomes
        )
        states.extend(item.state for item in targets)
        transition = RuntimeTimedActionTransition(
            state_id=source.state.state_id,
            instance_action_id=candidate.instance_action_id,
            action_type=candidate.action_type,
            intervention_cost_units=candidate.cost_units,
            expected_false_interventions=0.0,
            duration_ms=estimate.duration_ms,
            prediction_evidence_id=estimate.provenance_id,
            outcomes=tuple(
                RuntimeJointTransitionOutcome(
                    probability=outcome.probability,
                    next_state_id=target.state.state_id,
                    outcome_id=outcome.outcome_id,
                )
                for outcome, target in zip(estimate.outcomes, targets, strict=True)
            ),
        )
        transitions.append(transition)
        rows.append(
            {
                **asdict(transition),
                "action_type": transition.action_type.value,
                "outcomes": [asdict(item) for item in transition.outcomes],
                "depth": depth,
                "source_outcome": source_outcome,
                "provenance_detail": estimate.provenance_detail,
            }
        )
        return targets

    for root_index, candidate in enumerate(slate.candidates):
        estimate = estimate_by_id[candidate.instance_action_id]
        root_targets = add_transition(
            source=root,
            candidate=candidate,
            estimate=estimate,
            depth=1,
            source_outcome="root",
            ordinal=root_index,
        )
        root_outcomes = (
            zip(estimate.outcomes, root_targets, strict=True)
            if horizon == 2
            else ()
        )
        for outcome_index, (outcome, target) in enumerate(root_outcomes):
            for followup_index, followup in enumerate(
                _followups(
                    root_candidate=candidate,
                    root_outcome_id=outcome.outcome_id,
                    slate=slate,
                )
            ):
                followup_estimate = estimate_by_id[followup.instance_action_id]
                add_transition(
                    source=target,
                    candidate=followup,
                    estimate=followup_estimate,
                    depth=2,
                    source_outcome=(
                        f"{candidate.instance_action_id}:{outcome.outcome_id}"
                    ),
                    ordinal=(root_index * 100 + outcome_index * 10 + followup_index),
                )

    root_transitions = [item for item in transitions if item.state_id == "root"]
    if not root_transitions:
        raise ValueError("runtime-owned option slate has no feasible root action")
    states_by_id = {item.state_id: item for item in states}
    baseline = root_transitions[0]
    baseline_value = _immediate_value(baseline, states_by_id)
    for transition in root_transitions[1:]:
        value = _immediate_value(transition, states_by_id)
        if value.priority > baseline_value.priority:
            baseline = transition
            baseline_value = value
    return _CompiledOptionModel(
        states=tuple(states),
        transitions=tuple(transitions),
        transition_rows=tuple(rows),
        immediate_baseline_action_id=baseline.instance_action_id,
    )


def _value_payload(value: RuntimeTimeAwarePolicyValue) -> dict[str, float]:
    """Rename the planner's terminal field to preserve evaluator independence."""

    return {
        "residual_completion_proxy_probability": value.public_success_probability,
        "expected_unblocked_fraction": value.expected_unblocked_fraction,
        "expected_duration_ms": value.expected_duration_ms,
        "expected_false_interventions": value.expected_false_interventions,
        "expected_intervention_cost": value.expected_intervention_cost,
    }


def _state_payload(state: RuntimeTimedPlanningState) -> dict[str, object]:
    """Expose the internal terminal flag only under its actual proxy semantics."""

    payload = asdict(state)
    # Persist nested pairs as JSON-native lists so durable replay compares the
    # same representation produced after reading the JSONL decision log.
    payload["residual_predicate_states"] = [
        list(item) for item in state.residual_predicate_states
    ]
    payload["residual_completion_proxy"] = payload.pop("public_goal_complete")
    return payload


class RuntimeOwnedOptionDPResolver:
    """Build and replay a Runtime-owned option tree without a model-generated tree."""

    schema_version = "runtime_owned_two_step_option_model_v2"
    selector_identity = "runtime_owned_two_step_option_dp_v2"

    def resolve(
        self,
        slate: RuntimeJointActionSlate,
        projection: RuntimeOwnedOptionProjection,
        estimates: tuple[RuntimeAtomicActionEstimate, ...],
        *,
        horizon: int = 2,
    ) -> RuntimeJointResolvedChoice:
        """Compile one model from atomic estimates, then run existing Bellman code."""

        compiled = _compile_model(
            slate=slate,
            projection=projection,
            estimates=estimates,
            horizon=horizon,
        )
        model = {
            "schema_version": self.schema_version,
            "implementation_status": "implementation_unverified",
            "selector_identity": self.selector_identity,
            "objective_identity": (
                "residual_completion_then_unblocked_then_time_then_false_"
                "interventions_then_intervention_cost_v1"
            ),
            "horizon": horizon,
            "slate": slate.to_dict(),
            "projection": projection.to_dict(),
            "atomic_estimates": [item.to_dict() for item in estimates],
            "generated_states": [_state_payload(item) for item in compiled.states],
            "generated_transitions": list(compiled.transition_rows),
            "immediate_baseline_action_id": compiled.immediate_baseline_action_id,
            "official_evaluator_consumed": False,
            "terminal_value_semantics": "public_residual_completion_proxy_only",
        }
        return self._solve(slate=slate, model=model, compiled=compiled)

    def resolve_recorded(
        self,
        slate: RuntimeJointActionSlate,
        model: Mapping[str, object],
    ) -> RuntimeJointResolvedChoice:
        """Rebuild Runtime branches from saved inputs instead of trusting saved states."""

        if (
            model.get("schema_version") != self.schema_version
            or model.get("implementation_status") != "implementation_unverified"
            or model.get("selector_identity") != self.selector_identity
            or model.get("horizon") not in {1, 2}
            or model.get("slate") != slate.to_dict()
            or model.get("official_evaluator_consumed") is not False
            or model.get("terminal_value_semantics")
            != "public_residual_completion_proxy_only"
        ):
            raise ValueError("recorded runtime-owned option model identity differs")
        raw_projection = model.get("projection")
        raw_estimates = model.get("atomic_estimates")
        if not isinstance(raw_projection, Mapping) or not isinstance(raw_estimates, list):
            raise ValueError("recorded runtime-owned option inputs are invalid")
        projection = RuntimeOwnedOptionProjection.from_dict(raw_projection)
        estimates = tuple(
            RuntimeAtomicActionEstimate.from_dict(item)
            for item in raw_estimates
            if isinstance(item, Mapping)
        )
        if len(estimates) != len(raw_estimates):
            raise ValueError("recorded runtime-owned option estimate is invalid")
        compiled = _compile_model(
            slate=slate,
            projection=projection,
            estimates=estimates,
            horizon=int(model["horizon"]),
        )
        if (
            model.get("generated_states")
            != [_state_payload(item) for item in compiled.states]
            or model.get("generated_transitions") != list(compiled.transition_rows)
            or model.get("immediate_baseline_action_id")
            != compiled.immediate_baseline_action_id
        ):
            raise ValueError("recorded runtime-owned branches differ from reconstruction")
        return self._solve(slate=slate, model=dict(model), compiled=compiled)

    def _solve(
        self,
        *,
        slate: RuntimeJointActionSlate,
        model: Mapping[str, object],
        compiled: _CompiledOptionModel,
    ) -> RuntimeJointResolvedChoice:
        planner = RuntimeTimeAwareTraceDPPlanner(
            states=compiled.states,
            transitions=compiled.transitions,
            horizon=int(model["horizon"]),
        )
        decision = planner.plan("root")
        candidates = {item.instance_action_id: item for item in slate.candidates}
        selected = candidates.get(decision.instance_action_id)
        if selected is None:
            raise ValueError("runtime-owned option planner returned no executable action")
        return RuntimeJointResolvedChoice(
            selected=selected,
            rule_id=(
                "runtime_owned_two_step_option_bellman_v2"
                if model["horizon"] == 2
                else "runtime_owned_one_step_replan_bellman_v1"
            ),
            bellman_values={
                action_id: _value_payload(value)
                for action_id, value in decision.q_values
            },
            jit_advantages=None,
            planning_model=dict(model),
        )


__all__ = [
    "RuntimeAtomicActionEstimate",
    "RuntimeAtomicOutcomeEstimate",
    "RuntimeOptionAcceptancePredicate",
    "RuntimeOptionTaskDependency",
    "RuntimeOwnedOptionDPResolver",
    "RuntimeOwnedOptionProjection",
]
