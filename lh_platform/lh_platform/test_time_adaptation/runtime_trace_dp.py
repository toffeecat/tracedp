"""Finite-horizon TraceDP policy over live runtime prerequisite repairs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    OutputRequirement,
    PortableRuntimeKernel,
    TaskNode,
)
from lh_platform.test_time_adaptation.runtime_output_obligation import (
    RuntimeOutputObligation,
    RuntimeOutputPhase,
    compile_runtime_output_obligations,
)


@dataclass(frozen=True)
class RuntimeRepairCapability:
    """Expose only the policy-visible part of one environment repair ability."""

    resource_id: str
    supported: bool
    cost_units: int
    cost_bytes: int
    success_probability: float
    probability_evidence_id: str
    necessity_probability: float
    necessity_evidence_id: str
    support_reason: str | None = None
    action_kind: str = "restore_resource"

    def __post_init__(self) -> None:
        if not self.resource_id or self.cost_units < 1 or self.cost_bytes < 0:
            raise ValueError("runtime repair capability identity or cost is invalid")
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("runtime repair probability must be in [0, 1]")
        if not 0.0 <= self.necessity_probability <= 1.0:
            raise ValueError("runtime prerequisite necessity must be in [0, 1]")
        if not self.probability_evidence_id:
            raise ValueError("runtime repair probability provenance is required")
        if not self.necessity_evidence_id:
            raise ValueError("runtime prerequisite necessity provenance is required")
        if self.action_kind not in {
            "restore_resource",
            "runtime_guidance",
            "runtime_semantic_guidance",
        }:
            raise ValueError("runtime management action kind is unsupported")


@dataclass(frozen=True)
class RuntimeRepairCandidate:
    """Bind one legal restore action to the tasks whose prerequisites use it."""

    action_id: str
    resource_id: str
    affected_task_ids: tuple[str, ...]
    cost_units: int
    cost_bytes: int
    success_probability: float
    probability_evidence_id: str
    necessity_probability: float
    necessity_evidence_id: str
    action_kind: str = "restore_resource"
    issue_id: str | None = None

    @property
    def planning_issue_id(self) -> str:
        """Separate the Bellman issue identity from one accounting resource."""

        return self.issue_id or self.resource_id


@dataclass(frozen=True)
class RuntimePolicyValue:
    """Keep the runtime objective vector explicit instead of tuning one scalar."""

    preconditions_recovered_probability: float
    expected_unblocked_fraction: float
    expected_false_interventions: float
    expected_cost_units: float

    @property
    def priority(self) -> tuple[float, float, float, float]:
        return (
            self.preconditions_recovered_probability,
            self.expected_unblocked_fraction,
            -self.expected_false_interventions,
            -self.expected_cost_units,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "preconditions_recovered_probability": (
                self.preconditions_recovered_probability
            ),
            "expected_unblocked_fraction": self.expected_unblocked_fraction,
            "expected_false_interventions": self.expected_false_interventions,
            "expected_cost_units": self.expected_cost_units,
        }


@dataclass(frozen=True)
class RuntimePlanDecision:
    action_id: str | None
    resource_id: str | None
    value: RuntimePolicyValue
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _PlanningState:
    missing_resource_ids: frozenset[str]
    attempted_action_ids: frozenset[str]
    remaining_budget: int


@dataclass(frozen=True)
class _OpenRuntimeDecision:
    decision_id: str
    planned_action_id: str
    resource_id: str
    cost_units: int
    executed_action_id: str | None = None


def _children_by_parent(task_nodes: Sequence[TaskNode]) -> dict[str, tuple[str, ...]]:
    children: dict[str, list[str]] = {node.task_id: [] for node in task_nodes}
    for node in task_nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node.task_id)
    return {task_id: tuple(values) for task_id, values in children.items()}


def _task_resource_ids(node: TaskNode) -> frozenset[str]:
    """Combine confirmed and explicitly uncertain dependency claims for planning."""

    return frozenset(node.prerequisite_resource_ids).union(
        node.hypothesized_prerequisite_resource_ids
    )


def _blocked_tasks(
    task_nodes: Sequence[TaskNode], missing_resource_ids: frozenset[str]
) -> frozenset[str]:
    """Evaluate blocking directly from the declared parent and prerequisite graph."""

    nodes = {node.task_id: node for node in task_nodes}
    children = _children_by_parent(task_nodes)
    blocked: set[str] = set()

    def visit(task_id: str) -> bool:
        node = nodes[task_id]
        # Visit every sibling before aggregation; short-circuiting would omit
        # blocked branches from both the state and the planner objective.
        child_states = tuple(visit(child_id) for child_id in children[task_id])
        child_blocked = any(child_states)
        own_blocked = bool(
            missing_resource_ids.intersection(_task_resource_ids(node))
        )
        if own_blocked or child_blocked:
            blocked.add(task_id)
            return True
        return False

    for task_id, node in nodes.items():
        if node.parent_id is None:
            visit(task_id)
    return frozenset(blocked)


def _blocking_resources_by_task(
    task_nodes: Sequence[TaskNode], missing_resource_ids: frozenset[str]
) -> dict[str, frozenset[str]]:
    """Collect missing resources from each task's complete descendant subtree."""

    nodes = {node.task_id: node for node in task_nodes}
    children = _children_by_parent(task_nodes)
    resources: dict[str, frozenset[str]] = {}

    def visit(task_id: str) -> frozenset[str]:
        node = nodes[task_id]
        own = missing_resource_ids.intersection(_task_resource_ids(node))
        descendant = frozenset(
            resource_id
            for child_id in children[task_id]
            for resource_id in visit(child_id)
        )
        resources[task_id] = frozenset(own) | descendant
        return resources[task_id]

    for task_id, node in nodes.items():
        if node.parent_id is None:
            visit(task_id)
    return resources


def _none_required_probability(
    resource_ids: frozenset[str], necessity_by_resource: Mapping[str, float]
) -> float:
    """Use the declared independent Bernoulli premise for unresolved prerequisites."""

    return math.prod(1.0 - necessity_by_resource[item] for item in resource_ids)


def compile_runtime_repair_candidates(
    kernel: PortableRuntimeKernel,
    capabilities: Mapping[str, RuntimeRepairCapability],
    *,
    excluded_resource_versions: frozenset[tuple[str, int]] = frozenset(),
) -> tuple[frozenset[str], tuple[RuntimeRepairCandidate, ...]]:
    """Compile missing prerequisites without inferring task semantics from paths."""

    missing = frozenset(
        resource_id
        for node in kernel.task_nodes.values()
        for resource_id in _task_resource_ids(node)
        if (
            (observation := kernel.observations.get(resource_id)) is not None
            and observation.observed
            and observation.exists is False
        )
    )
    candidates = []
    nodes = kernel.task_nodes
    for resource_id in sorted(missing):
        capability = capabilities.get(resource_id)
        if capability is None or not capability.supported:
            continue
        if (resource_id, kernel.resource_versions[resource_id]) in excluded_resource_versions:
            continue
        affected_ids: set[str] = set()
        for node in nodes.values():
            if resource_id not in _task_resource_ids(node):
                continue
            current: TaskNode | None = node
            while current is not None:
                affected_ids.add(current.task_id)
                current = nodes.get(current.parent_id) if current.parent_id else None
        affected = tuple(sorted(affected_ids))
        candidates.append(
            RuntimeRepairCandidate(
                action_id=f"restore:{resource_id}",
                resource_id=resource_id,
                affected_task_ids=affected,
                cost_units=capability.cost_units,
                cost_bytes=capability.cost_bytes,
                success_probability=capability.success_probability,
                probability_evidence_id=capability.probability_evidence_id,
                necessity_probability=capability.necessity_probability,
                necessity_evidence_id=capability.necessity_evidence_id,
                action_kind=capability.action_kind,
            )
        )
    return missing, tuple(candidates)


def compile_runtime_progress_candidates(
    kernel: PortableRuntimeKernel,
    *,
    excluded_resource_versions: frozenset[tuple[str, int]] = frozenset(),
) -> tuple[
    frozenset[str],
    tuple[RuntimeRepairCandidate, ...],
    dict[str, tuple[str, ...]],
]:
    """Compile unresolved public outputs into generic Actor-guidance candidates."""

    candidates: list[RuntimeRepairCandidate] = []
    affected_by_issue: dict[str, tuple[str, ...]] = {}
    nodes = kernel.task_nodes
    for node in nodes.values():
        if not node.required_output_resource_ids:
            continue
        observations = [
            kernel.observations.get(resource_id)
            for resource_id in node.required_output_resource_ids
        ]
        if any(item is None or not item.observed for item in observations):
            continue
        exists = [item.exists is True for item in observations if item is not None]
        satisfied = (
            any(exists)
            if node.output_requirement is OutputRequirement.ANY
            else all(exists)
        )
        if satisfied:
            continue
        missing = [
            resource_id
            for resource_id, observation in zip(
                node.required_output_resource_ids, observations, strict=True
            )
            if observation is not None and observation.exists is False
        ]
        if not missing:
            continue
        anchor_resource_id = missing[0]
        if (
            anchor_resource_id,
            kernel.resource_versions[anchor_resource_id],
        ) in excluded_resource_versions:
            continue
        affected_ids: list[str] = []
        current: TaskNode | None = node
        while current is not None:
            affected_ids.append(current.task_id)
            current = nodes.get(current.parent_id) if current.parent_id else None
        affected = tuple(sorted(affected_ids))
        issue_id = f"public-output-obligation:{node.task_id}"
        affected_by_issue[issue_id] = affected
        candidates.append(
            RuntimeRepairCandidate(
                action_id=f"guide-output:{node.task_id}",
                resource_id=anchor_resource_id,
                affected_task_ids=affected,
                cost_units=1,
                cost_bytes=0,
                success_probability=0.5,
                probability_evidence_id=(
                    "locally_designed_beta_1_1_output_guidance_prior_v1"
                ),
                necessity_probability=1.0,
                necessity_evidence_id="explicit_public_output_obligation_v1",
                action_kind="runtime_guidance",
                issue_id=issue_id,
            )
        )
    return frozenset(affected_by_issue), tuple(candidates), affected_by_issue


def compile_runtime_semantic_candidates(
    kernel: PortableRuntimeKernel,
    *,
    excluded_obligation_versions: frozenset[
        tuple[str, str, tuple[tuple[str, int], ...], str]
    ] = frozenset(),
) -> tuple[
    frozenset[str],
    tuple[RuntimeRepairCandidate, ...],
    dict[str, tuple[str, ...]],
    dict[str, RuntimeOutputObligation],
]:
    """Compile complete present-output obligations into semantic-check actions."""

    obligations = {
        item.obligation_id: item
        for item in compile_runtime_output_obligations(kernel)
        if item.phase is RuntimeOutputPhase.OUTPUT_PRESENT_UNVERIFIED
        and item.version_key not in excluded_obligation_versions
    }
    candidates = tuple(
        RuntimeRepairCandidate(
            action_id=f"check-output-semantics:{obligation.task_id}",
            resource_id=obligation.output_resource_ids[0],
            affected_task_ids=obligation.affected_task_ids,
            cost_units=1,
            cost_bytes=0,
            success_probability=0.5,
            probability_evidence_id=(
                "locally_designed_beta_1_1_semantic_guidance_prior_v1"
            ),
            necessity_probability=1.0,
            necessity_evidence_id="public_output_requires_independent_evaluation_v1",
            action_kind="runtime_semantic_guidance",
            issue_id=obligation.obligation_id,
        )
        for obligation in obligations.values()
    )
    affected = {
        obligation.obligation_id: obligation.affected_task_ids
        for obligation in obligations.values()
    }
    return frozenset(obligations), candidates, affected, obligations


class RuntimeTraceDPPlanner:
    """Plan prerequisite restores by exact finite-horizon Bellman recursion.

    This is a locally designed runtime specialization of the project's
    Dynamic TraceDP objective. It plans only over declared restoration
    capabilities and never interprets a resource name as a benchmark root.
    """

    def __init__(
        self,
        *,
        task_nodes: Sequence[TaskNode],
        initial_missing_resource_ids: frozenset[str],
        candidates: Sequence[RuntimeRepairCandidate],
        necessity_probabilities: Mapping[str, float] | None = None,
        affected_task_ids_by_resource: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.task_nodes = tuple(task_nodes)
        self.initial_missing_resource_ids = initial_missing_resource_ids
        self.candidates = tuple(candidates)
        action_ids = tuple(candidate.action_id for candidate in self.candidates)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("runtime repair candidates must be unique")
        self._candidate_map = {
            candidate.action_id: candidate for candidate in self.candidates
        }
        derived_necessities = {
            candidate.planning_issue_id: candidate.necessity_probability
            for candidate in self.candidates
        }
        if necessity_probabilities is not None:
            derived_necessities.update(necessity_probabilities)
        if set(self.initial_missing_resource_ids) != set(derived_necessities):
            raise ValueError("every missing prerequisite requires a necessity probability")
        self._necessity_by_resource = derived_necessities
        if affected_task_ids_by_resource is None:
            self._initial_blocked = _blocked_tasks(
                self.task_nodes, self.initial_missing_resource_ids
            )
            self._blocking_resources = _blocking_resources_by_task(
                self.task_nodes, self.initial_missing_resource_ids
            )
        else:
            if set(affected_task_ids_by_resource) != set(
                self.initial_missing_resource_ids
            ):
                raise ValueError("runtime issue impact map is incomplete")
            self._initial_blocked = frozenset(
                task_id
                for task_ids in affected_task_ids_by_resource.values()
                for task_id in task_ids
            )
            self._blocking_resources = {
                task_id: frozenset(
                    resource_id
                    for resource_id, task_ids in affected_task_ids_by_resource.items()
                    if task_id in task_ids
                )
                for task_id in self._initial_blocked
            }
        self._cache: dict[_PlanningState, tuple[str | None, RuntimePolicyValue]] = {}

    def _stop_value(self, state: _PlanningState) -> RuntimePolicyValue:
        denominator = max(1, len(self._initial_blocked))
        all_required_absent_probability = _none_required_probability(
            state.missing_resource_ids,
            self._necessity_by_resource,
        )
        return RuntimePolicyValue(
            preconditions_recovered_probability=all_required_absent_probability,
            expected_unblocked_fraction=(
                sum(
                    _none_required_probability(
                        self._blocking_resources[task_id].intersection(
                            state.missing_resource_ids
                        ),
                        self._necessity_by_resource,
                    )
                    for task_id in self._initial_blocked
                )
                / denominator
            ),
            expected_false_interventions=0.0,
            expected_cost_units=0.0,
        )

    def _action_value(
        self, state: _PlanningState, candidate: RuntimeRepairCandidate
    ) -> RuntimePolicyValue:
        attempted = state.attempted_action_ids | {candidate.action_id}
        success_state = _PlanningState(
            missing_resource_ids=(
                state.missing_resource_ids - {candidate.planning_issue_id}
            ),
            attempted_action_ids=attempted,
            remaining_budget=state.remaining_budget - candidate.cost_units,
        )
        failure_state = _PlanningState(
            missing_resource_ids=state.missing_resource_ids,
            attempted_action_ids=attempted,
            remaining_budget=state.remaining_budget - candidate.cost_units,
        )
        success_value = self._best(success_state)[1]
        failure_value = self._best(failure_state)[1]
        p_success = candidate.success_probability
        return RuntimePolicyValue(
            preconditions_recovered_probability=(
                p_success * success_value.preconditions_recovered_probability
                + (1.0 - p_success)
                * failure_value.preconditions_recovered_probability
            ),
            expected_unblocked_fraction=(
                p_success * success_value.expected_unblocked_fraction
                + (1.0 - p_success) * failure_value.expected_unblocked_fraction
            ),
            expected_false_interventions=(
                (1.0 - candidate.necessity_probability)
                + p_success * success_value.expected_false_interventions
                + (1.0 - p_success)
                * failure_value.expected_false_interventions
            ),
            expected_cost_units=(
                candidate.cost_units
                + p_success * success_value.expected_cost_units
                + (1.0 - p_success) * failure_value.expected_cost_units
            ),
        )

    def _best(self, state: _PlanningState) -> tuple[str | None, RuntimePolicyValue]:
        cached = self._cache.get(state)
        if cached is not None:
            return cached
        best = (None, self._stop_value(state))
        legal = tuple(
            candidate
            for candidate in self.candidates
            if candidate.planning_issue_id in state.missing_resource_ids
            and candidate.action_id not in state.attempted_action_ids
            and candidate.cost_units <= state.remaining_budget
        )
        for candidate in legal:
            value = self._action_value(state, candidate)
            if value.priority > best[1].priority:
                best = (candidate.action_id, value)
        self._cache[state] = best
        return best

    def plan(self, *, remaining_budget: int) -> RuntimePlanDecision:
        """Return one action and the full Q/V table for the current live state."""

        if remaining_budget < 0:
            raise ValueError("runtime intervention budget cannot be negative")
        state = _PlanningState(
            missing_resource_ids=self.initial_missing_resource_ids,
            attempted_action_ids=frozenset(),
            remaining_budget=remaining_budget,
        )
        action_id, value = self._best(state)
        stop_value = self._stop_value(state)
        q_values = {
            "abstain": stop_value,
            **{
                candidate.action_id: self._action_value(state, candidate)
                for candidate in self.candidates
                if candidate.cost_units <= remaining_budget
            },
        }
        value_payload = value.to_dict()
        diagnostics = {
            "objective": "runtime_precondition_recovery_lexicographic_v1",
            "remaining_budget": remaining_budget,
            "unresolved_issue_ids": sorted(self.initial_missing_resource_ids),
            "chosen_action_id": action_id or "abstain",
            "v_star": value_payload,
            "q_values": {
                candidate_id: candidate_value.to_dict()
                for candidate_id, candidate_value in q_values.items()
            },
            "priority_advantages": {
                candidate_id: {
                    "preconditions_recovered_probability": (
                        candidate_value.preconditions_recovered_probability
                        - value.preconditions_recovered_probability
                    ),
                    "expected_unblocked_fraction": (
                        candidate_value.expected_unblocked_fraction
                        - value.expected_unblocked_fraction
                    ),
                    "expected_false_interventions": (
                        value.expected_false_interventions
                        - candidate_value.expected_false_interventions
                    ),
                    "expected_cost_units": (
                        value.expected_cost_units
                        - candidate_value.expected_cost_units
                    ),
                }
                for candidate_id, candidate_value in q_values.items()
            },
            "candidates": {
                candidate.action_id: {
                    "resource_id": candidate.resource_id,
                    "issue_id": candidate.planning_issue_id,
                    "affected_task_ids": list(candidate.affected_task_ids),
                    "cost_units": candidate.cost_units,
                    "cost_bytes": candidate.cost_bytes,
                    "success_probability": candidate.success_probability,
                    "necessity_probability": candidate.necessity_probability,
                    "action_kind": candidate.action_kind,
                }
                for candidate in self.candidates
            },
            "prediction_provenance": {
                candidate.action_id: candidate.probability_evidence_id
                for candidate in self.candidates
            },
            "necessity_provenance": {
                candidate.action_id: candidate.necessity_evidence_id
                for candidate in self.candidates
            },
        }
        candidate = self._candidate_map.get(action_id) if action_id else None
        return RuntimePlanDecision(
            action_id=action_id,
            resource_id=candidate.resource_id if candidate is not None else None,
            value=value,
            diagnostics=diagnostics,
        )


class DynamicTraceDPRuntimePolicy:
    """Persist proposals and outcomes for audited event-triggered policies."""

    _SUPPORTED_POLICY_IDS = {
        "dynamic_tracedp_runtime_v1",
        "dynamic_tracedp_runtime_v2",
        "dynamic_tracedp_runtime_v3",
        "dynamic_tracedp_runtime_v4",
        "fixed_two_stage_runtime_v1",
    }

    def __init__(
        self,
        *,
        budget_units: int,
        decision_log_path: Path,
        policy_id: str = "dynamic_tracedp_runtime_v1",
    ) -> None:
        if budget_units < 1:
            raise ValueError("dynamic runtime TraceDP requires a positive budget")
        if policy_id not in self._SUPPORTED_POLICY_IDS:
            raise ValueError("dynamic runtime TraceDP policy identity is unsupported")
        if decision_log_path.exists():
            raise FileExistsError(f"runtime decision log already exists: {decision_log_path}")
        self.policy_id = policy_id
        self.total_budget_units = budget_units
        self.remaining_budget_units = budget_units
        self.decision_log_path = decision_log_path
        self._sequence = 0
        self._open_decision: _OpenRuntimeDecision | None = None
        self.failed_resource_versions: set[tuple[str, int]] = set()
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.decision_log_path.touch(exist_ok=False)

    def _fixed_rule_decision(
        self,
        *,
        issue_ids: frozenset[str],
        candidates: Sequence[RuntimeRepairCandidate],
    ) -> RuntimePlanDecision:
        """Select the first legal candidate without Bellman values or learned scores."""

        legal = tuple(
            candidate
            for candidate in sorted(candidates, key=lambda item: item.action_id)
            if candidate.cost_units <= self.remaining_budget_units
        )
        chosen = legal[0] if legal else None
        value = RuntimePolicyValue(0.0, 0.0, 0.0, 0.0)
        diagnostics: dict[str, object] = {
            "objective": "fixed_first_legal_candidate_v1",
            "decision_rule": "fixed_first_legal_candidate_v1",
            "remaining_budget": self.remaining_budget_units,
            "unresolved_issue_ids": sorted(issue_ids),
            "chosen_action_id": chosen.action_id if chosen is not None else "abstain",
            "v_star": value.to_dict(),
            "q_values": {},
            "priority_advantages": {},
            "candidates": {
                candidate.action_id: {
                    "resource_id": candidate.resource_id,
                    "issue_id": candidate.planning_issue_id,
                    "affected_task_ids": list(candidate.affected_task_ids),
                    "cost_units": candidate.cost_units,
                    "cost_bytes": candidate.cost_bytes,
                    "success_probability": candidate.success_probability,
                    "necessity_probability": candidate.necessity_probability,
                    "action_kind": candidate.action_kind,
                }
                for candidate in candidates
            },
            "prediction_provenance": {
                candidate.action_id: candidate.probability_evidence_id
                for candidate in candidates
            },
            "necessity_provenance": {
                candidate.action_id: candidate.necessity_evidence_id
                for candidate in candidates
            },
        }
        return RuntimePlanDecision(
            action_id=chosen.action_id if chosen is not None else None,
            resource_id=chosen.resource_id if chosen is not None else None,
            value=value,
            diagnostics=diagnostics,
        )

    @classmethod
    def load(
        cls,
        *,
        budget_units: int,
        decision_log_path: Path,
        policy_id: str = "dynamic_tracedp_runtime_v1",
    ) -> DynamicTraceDPRuntimePolicy:
        """Replay a durable decision prefix without resetting budget or failures."""

        if budget_units < 1:
            raise ValueError("dynamic runtime TraceDP requires a positive budget")
        if policy_id not in cls._SUPPORTED_POLICY_IDS:
            raise ValueError("dynamic runtime TraceDP policy identity is unsupported")
        if not decision_log_path.is_file():
            raise FileNotFoundError(decision_log_path)
        policy = cls.__new__(cls)
        policy.policy_id = policy_id
        policy.total_budget_units = budget_units
        policy.remaining_budget_units = budget_units
        policy.decision_log_path = decision_log_path
        policy._sequence = 0
        policy._open_decision = None
        policy.failed_resource_versions = set()
        for expected_sequence, raw_line in enumerate(decision_log_path.read_text().splitlines()):
            row = json.loads(raw_line)
            if (
                not isinstance(row, dict)
                or row.get("schema_version")
                != "dynamic_tracedp_runtime_decision_v1"
                or row.get("sequence") != expected_sequence
            ):
                raise ValueError("runtime decision log sequence or schema differs")
            payload = row.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("runtime decision payload must be an object")
            event_type = row.get("event_type")
            if event_type == "decision_planned":
                if policy._open_decision is not None:
                    raise ValueError("runtime decision log contains overlapping decisions")
                action_id = payload.get("action_id")
                if payload.get("policy_id") != policy_id:
                    raise ValueError("runtime decision policy identity differs")
                if action_id is not None:
                    diagnostics = payload.get("diagnostics")
                    candidates = (
                        diagnostics.get("candidates")
                        if isinstance(diagnostics, dict)
                        else None
                    )
                    candidate = (
                        candidates.get(action_id)
                        if isinstance(candidates, dict)
                        else None
                    )
                    if not isinstance(candidate, dict):
                        raise ValueError("planned runtime action lacks candidate evidence")
                    policy._open_decision = _OpenRuntimeDecision(
                        decision_id=str(payload["decision_id"]),
                        planned_action_id=str(action_id),
                        resource_id=str(payload["resource_id"]),
                        cost_units=int(candidate["cost_units"]),
                    )
            elif event_type == "decision_execution_started":
                open_decision = policy._open_decision
                executed_action_id = payload.get("executed_action_id")
                if (
                    open_decision is None
                    or open_decision.executed_action_id is not None
                    or payload.get("decision_id") != open_decision.decision_id
                    or payload.get("planned_action_id")
                    != open_decision.planned_action_id
                    or payload.get("resource_id") != open_decision.resource_id
                    or not isinstance(executed_action_id, str)
                    or not executed_action_id
                ):
                    raise ValueError("runtime decision execution start is invalid")
                policy._open_decision = _OpenRuntimeDecision(
                    decision_id=open_decision.decision_id,
                    planned_action_id=open_decision.planned_action_id,
                    resource_id=open_decision.resource_id,
                    cost_units=open_decision.cost_units,
                    executed_action_id=executed_action_id,
                )
            elif event_type == "decision_outcome":
                open_decision = policy._open_decision
                if (
                    open_decision is None
                    or open_decision.executed_action_id is None
                    or payload.get("decision_id") != open_decision.decision_id
                    or payload.get("resource_id") != open_decision.resource_id
                    or payload.get("cost_units") != open_decision.cost_units
                    or payload.get("executed_action_id")
                    != open_decision.executed_action_id
                ):
                    raise ValueError("runtime decision outcome does not close its proposal")
                expected_remaining = (
                    policy.remaining_budget_units - open_decision.cost_units
                )
                if payload.get("remaining_budget") != expected_remaining:
                    raise ValueError("runtime decision budget history differs")
                policy.remaining_budget_units = expected_remaining
                if payload.get("status") != "completed":
                    policy.failed_resource_versions.add(
                        (
                            open_decision.resource_id,
                            int(payload["resource_version_after"]),
                        )
                    )
                policy._open_decision = None
            elif event_type == "decision_abandoned":
                open_decision = policy._open_decision
                if (
                    open_decision is None
                    or payload.get("decision_id") != open_decision.decision_id
                    or payload.get("planned_action_id")
                    != open_decision.planned_action_id
                    or payload.get("resource_id") != open_decision.resource_id
                    or payload.get("reserved_action_id")
                    != open_decision.executed_action_id
                ):
                    raise ValueError(
                        "abandoned runtime decision does not close its proposal"
                    )
                policy._open_decision = None
            else:
                raise ValueError("runtime decision event type is unsupported")
            policy._sequence += 1
        return policy

    @property
    def has_open_decision(self) -> bool:
        """Expose whether recovery must resolve an interrupted intervention."""

        return self._open_decision is not None

    @property
    def open_resource_id(self) -> str | None:
        return (
            self._open_decision.resource_id
            if self._open_decision is not None
            else None
        )

    @property
    def open_executed_action_id(self) -> str | None:
        """Return the exact Kernel action bound before environment execution."""

        return (
            self._open_decision.executed_action_id
            if self._open_decision is not None
            else None
        )

    @property
    def open_cost_units(self) -> int | None:
        """Expose the durable planned cost for exact resume reconciliation."""

        return (
            self._open_decision.cost_units
            if self._open_decision is not None
            else None
        )

    def record_execution_started(self, *, action_id: str) -> None:
        """Bind one planned decision to its exact Kernel action before execution."""

        if self._open_decision is None:
            raise RuntimeError("runtime policy has no open decision")
        if self._open_decision.executed_action_id is not None:
            raise RuntimeError("runtime decision execution already started")
        if not action_id:
            raise ValueError("runtime execution action identity is required")
        open_decision = self._open_decision
        self._append(
            "decision_execution_started",
            {
                "decision_id": open_decision.decision_id,
                "planned_action_id": open_decision.planned_action_id,
                "resource_id": open_decision.resource_id,
                "executed_action_id": action_id,
            },
        )
        self._open_decision = _OpenRuntimeDecision(
            decision_id=open_decision.decision_id,
            planned_action_id=open_decision.planned_action_id,
            resource_id=open_decision.resource_id,
            cost_units=open_decision.cost_units,
            executed_action_id=action_id,
        )

    def record_interrupted_outcome(
        self,
        *,
        action_id: str,
        resource_version_after: int,
        reason: str,
    ) -> None:
        """Close an in-flight decision conservatively during process recovery."""

        if self._open_decision is None:
            raise RuntimeError("runtime policy has no interrupted decision")
        self.record_outcome(
            action_id=action_id,
            resource_id=self._open_decision.resource_id,
            cost_units=self._open_decision.cost_units,
            status="cancelled_with_unknown_effect",
            reason=reason,
            resource_version_after=resource_version_after,
        )

    def abandon_interrupted_before_execution(self, *, reason: str) -> None:
        """Close a durable plan that never reached a Kernel execution action."""

        if self._open_decision is None:
            raise RuntimeError("runtime policy has no interrupted decision")
        open_decision = self._open_decision
        self._append(
            "decision_abandoned",
            {
                "decision_id": open_decision.decision_id,
                "planned_action_id": open_decision.planned_action_id,
                "resource_id": open_decision.resource_id,
                "reason": reason,
                "cost_units": 0,
                "remaining_budget": self.remaining_budget_units,
                "environment_execution_started": False,
                "reserved_action_id": open_decision.executed_action_id,
            },
        )
        self._open_decision = None

    def _append(self, event_type: str, payload: Mapping[str, object]) -> None:
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.decision_log_path.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "schema_version": "dynamic_tracedp_runtime_decision_v1",
                        "sequence": self._sequence,
                        "event_type": event_type,
                        "payload": dict(payload),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._sequence += 1

    def plan(
        self,
        *,
        kernel: PortableRuntimeKernel,
        capabilities: Mapping[str, RuntimeRepairCapability],
        trigger_action_id: str,
        excluded_resource_versions: frozenset[tuple[str, int]],
    ) -> RuntimePlanDecision | None:
        """Compile the current graph and durably record one Bellman decision."""

        if self._open_decision is not None:
            raise RuntimeError("runtime policy already has an unresolved decision")
        missing, candidates = compile_runtime_repair_candidates(
            kernel,
            capabilities,
            excluded_resource_versions=excluded_resource_versions,
        )
        if self.policy_id == "fixed_two_stage_runtime_v1":
            decision = self._fixed_rule_decision(
                issue_ids=missing,
                candidates=candidates,
            )
        else:
            planner = RuntimeTraceDPPlanner(
                task_nodes=tuple(kernel.task_nodes.values()),
                initial_missing_resource_ids=missing,
                candidates=candidates,
                necessity_probabilities={
                    resource_id: capabilities[resource_id].necessity_probability
                    for resource_id in missing
                },
            )
            decision = planner.plan(remaining_budget=self.remaining_budget_units)
        decision.diagnostics["capability_assessments"] = {
            resource_id: {
                "supported": capabilities[resource_id].supported,
                "support_reason": capabilities[resource_id].support_reason,
                "cost_units": capabilities[resource_id].cost_units,
                "cost_bytes": capabilities[resource_id].cost_bytes,
            }
            for resource_id in sorted(missing)
        }
        decision_id = f"runtime-decision-{self._sequence:06d}"
        self._append(
            "decision_planned",
            {
                "decision_id": decision_id,
                "policy_id": self.policy_id,
                "trigger_action_id": trigger_action_id,
                "resource_versions": dict(sorted(kernel.resource_versions.items())),
                "action_id": decision.action_id,
                "resource_id": decision.resource_id,
                "diagnostics": decision.diagnostics,
            },
        )
        if decision.action_id is not None:
            candidate = next(
                item for item in candidates if item.action_id == decision.action_id
            )
            self._open_decision = _OpenRuntimeDecision(
                decision_id=decision_id,
                planned_action_id=candidate.action_id,
                resource_id=candidate.resource_id,
                cost_units=candidate.cost_units,
            )
        return decision

    def plan_output_progress(
        self,
        *,
        kernel: PortableRuntimeKernel,
        trigger_action_id: str,
        excluded_resource_versions: frozenset[tuple[str, int]],
        no_progress_actor_actions: int,
        trigger_threshold: int,
    ) -> RuntimePlanDecision:
        """Plan one public-output guidance action after evidence of stalled progress."""

        if self._open_decision is not None:
            raise RuntimeError("runtime policy already has an unresolved decision")
        if no_progress_actor_actions < trigger_threshold or trigger_threshold < 1:
            raise ValueError("runtime progress trigger evidence is insufficient")
        missing, candidates, affected = compile_runtime_progress_candidates(
            kernel,
            excluded_resource_versions=excluded_resource_versions,
        )
        if self.policy_id == "fixed_two_stage_runtime_v1":
            decision = self._fixed_rule_decision(
                issue_ids=missing,
                candidates=candidates,
            )
        else:
            planner = RuntimeTraceDPPlanner(
                task_nodes=tuple(kernel.task_nodes.values()),
                initial_missing_resource_ids=missing,
                candidates=candidates,
                necessity_probabilities={
                    candidate.planning_issue_id: candidate.necessity_probability
                    for candidate in candidates
                },
                affected_task_ids_by_resource=affected,
            )
            decision = planner.plan(remaining_budget=self.remaining_budget_units)
        decision.diagnostics["objective"] = "runtime_output_progress_lexicographic_v1"
        decision.diagnostics["trigger_evidence"] = {
            "no_progress_actor_actions": no_progress_actor_actions,
            "trigger_threshold": trigger_threshold,
            "progress_definition": "declared_public_output_observation_changed",
        }
        decision.diagnostics["capability_assessments"] = {
            candidate.planning_issue_id: {
                "supported": True,
                "support_reason": "public output remains absent after the progress trigger",
                "cost_units": candidate.cost_units,
                "cost_bytes": candidate.cost_bytes,
            }
            for candidate in candidates
        }
        decision_id = f"runtime-decision-{self._sequence:06d}"
        self._append(
            "decision_planned",
            {
                "decision_id": decision_id,
                "policy_id": self.policy_id,
                "trigger_action_id": trigger_action_id,
                "resource_versions": dict(sorted(kernel.resource_versions.items())),
                "action_id": decision.action_id,
                "resource_id": decision.resource_id,
                "diagnostics": decision.diagnostics,
            },
        )
        if decision.action_id is not None:
            candidate = next(
                item for item in candidates if item.action_id == decision.action_id
            )
            self._open_decision = _OpenRuntimeDecision(
                decision_id=decision_id,
                planned_action_id=candidate.action_id,
                resource_id=candidate.resource_id,
                cost_units=candidate.cost_units,
            )
        return decision

    def plan_output_semantics(
        self,
        *,
        kernel: PortableRuntimeKernel,
        trigger_action_id: str,
        excluded_obligation_versions: frozenset[
            tuple[str, str, tuple[tuple[str, int], ...], str]
        ],
    ) -> RuntimePlanDecision:
        """Plan one semantic-check request over a complete public-output obligation."""

        if self._open_decision is not None:
            raise RuntimeError("runtime policy already has an unresolved decision")
        issues, candidates, affected, obligations = compile_runtime_semantic_candidates(
            kernel,
            excluded_obligation_versions=excluded_obligation_versions,
        )
        if self.policy_id == "fixed_two_stage_runtime_v1":
            decision = self._fixed_rule_decision(
                issue_ids=issues,
                candidates=candidates,
            )
        else:
            planner = RuntimeTraceDPPlanner(
                task_nodes=tuple(kernel.task_nodes.values()),
                initial_missing_resource_ids=issues,
                candidates=candidates,
                necessity_probabilities={
                    candidate.planning_issue_id: candidate.necessity_probability
                    for candidate in candidates
                },
                affected_task_ids_by_resource=affected,
            )
            decision = planner.plan(remaining_budget=self.remaining_budget_units)
        decision.diagnostics["objective"] = "runtime_output_semantic_check_lexicographic_v1"
        decision.diagnostics["trigger_evidence"] = {
            "trigger": "complete_public_output_became_present_before_official_evaluation",
            "obligations": {
                key: value.to_dict() for key, value in obligations.items()
            },
        }
        decision_id = f"runtime-decision-{self._sequence:06d}"
        self._append(
            "decision_planned",
            {
                "decision_id": decision_id,
                "policy_id": self.policy_id,
                "trigger_action_id": trigger_action_id,
                "resource_versions": dict(sorted(kernel.resource_versions.items())),
                "action_id": decision.action_id,
                "resource_id": decision.resource_id,
                "diagnostics": decision.diagnostics,
            },
        )
        if decision.action_id is not None:
            candidate = next(
                item for item in candidates if item.action_id == decision.action_id
            )
            self._open_decision = _OpenRuntimeDecision(
                decision_id=decision_id,
                planned_action_id=candidate.action_id,
                resource_id=candidate.resource_id,
                cost_units=candidate.cost_units,
            )
        return decision

    def record_outcome(
        self,
        *,
        action_id: str,
        resource_id: str,
        cost_units: int,
        status: str,
        reason: str | None,
        resource_version_after: int,
    ) -> None:
        """Charge an attempted intervention and close its durable decision."""

        if self._open_decision is None:
            raise RuntimeError("runtime policy has no open decision")
        open_decision = self._open_decision
        if open_decision.executed_action_id is None:
            raise RuntimeError("runtime outcome cannot precede execution start")
        if (
            action_id != open_decision.executed_action_id
        ):
            raise ValueError("runtime outcome action differs from execution start")
        if resource_id != open_decision.resource_id:
            raise ValueError("runtime outcome resource differs from the planned decision")
        if cost_units != open_decision.cost_units:
            raise ValueError("runtime outcome cost differs from the planned decision")
        if cost_units < 1 or cost_units > self.remaining_budget_units:
            raise ValueError("runtime intervention cost is outside the remaining budget")
        if status not in {"completed", "failed", "cancelled_with_unknown_effect"}:
            raise ValueError("runtime intervention outcome is invalid")
        if status != "completed" and not reason:
            raise ValueError("non-completed runtime intervention requires a reason")
        remaining_budget = self.remaining_budget_units - cost_units
        self._append(
            "decision_outcome",
            {
                "decision_id": open_decision.decision_id,
                "planned_action_id": open_decision.planned_action_id,
                "executed_action_id": action_id,
                "resource_id": resource_id,
                "status": status,
                "reason": reason,
                "cost_units": cost_units,
                "remaining_budget": remaining_budget,
                "resource_version_after": resource_version_after,
                "external_cancellation": status == "cancelled_with_unknown_effect",
                "probability_model_assessment": "not_assessed_online",
            },
        )
        self.remaining_budget_units = remaining_budget
        if status != "completed":
            self.failed_resource_versions.add(
                (resource_id, resource_version_after)
            )
        self._open_decision = None


__all__ = [
    "DynamicTraceDPRuntimePolicy",
    "RuntimePlanDecision",
    "RuntimePolicyValue",
    "RuntimeRepairCandidate",
    "RuntimeRepairCapability",
    "RuntimeTraceDPPlanner",
    "compile_runtime_repair_candidates",
    "compile_runtime_progress_candidates",
    "compile_runtime_semantic_candidates",
]
