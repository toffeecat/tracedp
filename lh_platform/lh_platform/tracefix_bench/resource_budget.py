"""Pure, locally designed resource-budget decisions for TraceFix-Bench v0.1.

The evaluator compares already persisted usage with one embedded budget vector.
It does not collect usage or prove that the supplied budget matches the formal
registry/run-plan snapshot, so every decision remains unbound and
implementation_unverified.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.tracefix_bench.contracts import UsageValueStatus
from lh_platform.tracefix_bench.run_contracts import (
    RUN_CONTRACT_IMPLEMENTATION_STATUS,
    ResourceCostV1,
    RunBudgetV1,
)
from lh_platform.tracefix_bench.scorer_contracts import (
    BudgetScope,
    ResourceBudgetContractV1,
    ResourceDimension,
    ResourceLimitV1,
    ResourceRole,
)

RESOURCE_BUDGET_LIMIT_DECISION_SCHEMA = (
    "longhorizon_tracefix_bench/resource_budget_limit_decision/v1"
)
RESOURCE_BUDGET_DECISION_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/resource_budget_decision_receipt/v1"
)
WHOLE_RUN_COVERAGE_ENTRY_SCHEMA = (
    "longhorizon_tracefix_bench/whole_run_coverage_entry/v1"
)
WHOLE_RUN_COVERAGE_RECEIPT_SCHEMA = (
    "longhorizon_tracefix_bench/whole_run_coverage_receipt/v1"
)
RESOURCE_BUDGET_DESIGN_ORIGIN = "locally_designed"
RESOURCE_BUDGET_BINDING_STATUS = "unbound"
_IDENTIFIER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_COUNT_DIMENSIONS = frozenset(
    {
        ResourceDimension.MODEL_CALLS,
        ResourceDimension.INPUT_TOKENS,
        ResourceDimension.OUTPUT_TOKENS,
        ResourceDimension.CACHE_READ_TOKENS,
        ResourceDimension.CACHE_WRITE_TOKENS,
        ResourceDimension.TOOL_CALLS,
        ResourceDimension.ENVIRONMENT_ACTIONS,
        ResourceDimension.PROBE_ACTIONS,
        ResourceDimension.CHECK_ACTIONS,
        ResourceDimension.REPAIR_ACTIONS,
        ResourceDimension.REPLAY_ATTEMPTS,
    }
)


class BudgetSubjectKind(StrEnum):
    """Distinguish per-work-item and independently measured whole-run usage."""

    WORK_ITEM = "work_item"
    RUN = "run"


class BudgetNumericRelation(StrEnum):
    """Preserve arithmetic comparison separately from evidence quality."""

    WITHIN_LIMIT = "within_limit"
    EXCEEDED = "exceeded"
    UNAVAILABLE = "unavailable"


class BudgetOverallDecision(StrEnum):
    """Only observed usage can prove compliance or an overrun."""

    WITHIN_LIMIT = "within_limit"
    EXCEEDED = "exceeded"
    UNVERIFIED = "unverified"


_WHOLE_RUN_COMPONENT_ROLES = frozenset(
    {
        ResourceRole.SCORER,
        ResourceRole.RECOVERY,
        ResourceRole.PERSISTENCE,
        ResourceRole.ORCHESTRATION,
    }
)


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _identifier(value: str, label: str) -> None:
    """Require a portable identity shared with registry and run contracts."""

    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable non-empty identifier")


def _finite(value: int | float, label: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")


@dataclass(frozen=True)
class ResourceKeyV1:
    """Serialize the exact three-part key shared by limits and actual usage."""

    role: ResourceRole
    scope: BudgetScope
    dimension: ResourceDimension

    def __post_init__(self) -> None:
        if not isinstance(self.role, ResourceRole):
            raise ValueError("resource key role must use the closed enum")
        if not isinstance(self.scope, BudgetScope):
            raise ValueError("resource key scope must use the closed enum")
        if not isinstance(self.dimension, ResourceDimension):
            raise ValueError("resource key dimension must use the closed enum")

    def to_dict(self) -> dict[str, str]:
        """Return the complete key without dropping its counting scope."""

        return {
            "role": self.role.value,
            "scope": self.scope.value,
            "dimension": self.dimension.value,
        }


@dataclass(frozen=True)
class WholeRunCoverageEntryV1:
    """Tie one required overhead component to one evidenced whole-run cost."""

    component_role: ResourceRole
    cost_key: ResourceKeyV1
    evidence_ref: str
    schema: str = WHOLE_RUN_COVERAGE_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WHOLE_RUN_COVERAGE_ENTRY_SCHEMA:
            raise ValueError("unsupported whole-run coverage entry schema")
        if self.component_role not in _WHOLE_RUN_COMPONENT_ROLES:
            raise ValueError("whole-run coverage component role is not required overhead")
        if type(self.cost_key) is not ResourceKeyV1:
            raise ValueError("whole-run coverage requires a typed resource key")
        if self.cost_key.role is not self.component_role:
            raise ValueError("whole-run coverage role must match its cost key")
        if self.cost_key.scope is not BudgetScope.WHOLE_RUN:
            raise ValueError("whole-run coverage requires a whole-run cost key")
        _text(self.evidence_ref, "whole-run coverage evidence_ref")

    def to_dict(self) -> dict[str, object]:
        """Serialize the component, exact cost key, and supporting evidence."""

        return {
            "schema": self.schema,
            "component_role": self.component_role.value,
            "cost_key": self.cost_key.to_dict(),
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class WholeRunCoverageReceiptV1:
    """Prove explicit accounting for scorer, recovery, persistence, and orchestration."""

    usage_aggregation_receipt_ref: str
    entries: tuple[WholeRunCoverageEntryV1, ...]
    schema: str = WHOLE_RUN_COVERAGE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WHOLE_RUN_COVERAGE_RECEIPT_SCHEMA:
            raise ValueError("unsupported whole-run coverage receipt schema")
        _text(
            self.usage_aggregation_receipt_ref,
            "whole-run coverage aggregation receipt ref",
        )
        if type(self.entries) is not tuple or any(
            type(entry) is not WholeRunCoverageEntryV1 for entry in self.entries
        ):
            raise ValueError("whole-run coverage entries must be typed records")
        keys = tuple(entry.cost_key for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise ValueError("whole-run coverage cannot repeat a cost key")
        if {entry.component_role for entry in self.entries} != _WHOLE_RUN_COMPONENT_ROLES:
            raise ValueError(
                "whole-run coverage requires scorer, recovery, persistence, and orchestration"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize every required overhead cost without a completeness boolean."""

        return {
            "schema": self.schema,
            "usage_aggregation_receipt_ref": self.usage_aggregation_receipt_ref,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class ResourceBudgetLimitDecisionV1:
    """Record one numeric comparison together with its usage evidence quality."""

    role: ResourceRole
    scope: BudgetScope
    dimension: ResourceDimension
    max_value: int | float
    usage_status: UsageValueStatus
    usage_value: int | float | None
    numeric_relation: BudgetNumericRelation
    usage_receipt_ref: str | None
    unavailable_reason: str | None
    unavailable_evidence_ref: str | None
    design_origin: str = RESOURCE_BUDGET_DESIGN_ORIGIN
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = RESOURCE_BUDGET_LIMIT_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESOURCE_BUDGET_LIMIT_DECISION_SCHEMA:
            raise ValueError("unsupported resource budget limit decision schema")
        ResourceKeyV1(self.role, self.scope, self.dimension)
        _finite(self.max_value, "resource limit max_value")
        if self.dimension in _COUNT_DIMENSIONS and type(self.max_value) is not int:
            raise ValueError("count and token limits must be integers")
        if not isinstance(self.usage_status, UsageValueStatus):
            raise ValueError("usage_status must use the closed enum")
        if not isinstance(self.numeric_relation, BudgetNumericRelation):
            raise ValueError("numeric_relation must use the closed enum")
        if self.usage_status is UsageValueStatus.UNAVAILABLE:
            if self.usage_value is not None:
                raise ValueError("unavailable usage must have a null value")
            if self.numeric_relation is not BudgetNumericRelation.UNAVAILABLE:
                raise ValueError("unavailable usage cannot claim a numeric relation")
            _text(self.unavailable_reason, "unavailable_reason")
            if self.usage_receipt_ref is not None:
                _text(self.usage_receipt_ref, "usage_receipt_ref")
            if self.unavailable_evidence_ref is not None:
                _text(self.unavailable_evidence_ref, "unavailable_evidence_ref")
        else:
            _finite(self.usage_value, "resource usage value")
            if self.dimension in _COUNT_DIMENSIONS and type(self.usage_value) is not int:
                raise ValueError("count and token usage must be integers")
            _text(self.usage_receipt_ref, "usage_receipt_ref")
            expected_relation = (
                BudgetNumericRelation.EXCEEDED
                if self.usage_value > self.max_value
                else BudgetNumericRelation.WITHIN_LIMIT
            )
            if self.numeric_relation is not expected_relation:
                raise ValueError("numeric relation does not match usage and limit")
            if self.unavailable_reason is not None or self.unavailable_evidence_ref is not None:
                raise ValueError("available usage cannot carry unavailability evidence")
        if self.design_origin != RESOURCE_BUDGET_DESIGN_ORIGIN:
            raise ValueError("resource budget decision must remain locally_designed")
        if self.implementation_status != RUN_CONTRACT_IMPLEMENTATION_STATUS:
            raise ValueError("resource budget decision cannot promote implementation status")

    def key(self) -> ResourceKeyV1:
        """Return the exact limit key used during matching."""

        return ResourceKeyV1(self.role, self.scope, self.dimension)

    def to_dict(self) -> dict[str, object]:
        """Serialize the limit decision without collapsing estimated evidence."""

        return {
            "schema": self.schema,
            "role": self.role.value,
            "scope": self.scope.value,
            "dimension": self.dimension.value,
            "max_value": self.max_value,
            "usage_status": self.usage_status.value,
            "usage_value": self.usage_value,
            "numeric_relation": self.numeric_relation.value,
            "usage_receipt_ref": self.usage_receipt_ref,
            "unavailable_reason": self.unavailable_reason,
            "unavailable_evidence_ref": self.unavailable_evidence_ref,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


@dataclass(frozen=True)
class ResourceBudgetDecisionReceiptV1:
    """Bind one complete budget decision to its subject and usage receipt."""

    budget_id: str
    budget_contract_ref: str
    subject_kind: BudgetSubjectKind
    subject_id: str
    usage_aggregation_receipt_ref: str
    budget_limits: tuple[ResourceLimitV1, ...]
    actual_costs: tuple[ResourceCostV1, ...]
    limit_decisions: tuple[ResourceBudgetLimitDecisionV1, ...]
    unbudgeted_usage_costs: tuple[ResourceCostV1, ...]
    usage_receipt_refs: tuple[str, ...]
    overall_decision: BudgetOverallDecision
    whole_run_coverage: WholeRunCoverageReceiptV1 | None
    binding_status: str = RESOURCE_BUDGET_BINDING_STATUS
    design_origin: str = RESOURCE_BUDGET_DESIGN_ORIGIN
    implementation_status: str = RUN_CONTRACT_IMPLEMENTATION_STATUS
    schema: str = RESOURCE_BUDGET_DECISION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESOURCE_BUDGET_DECISION_RECEIPT_SCHEMA:
            raise ValueError("unsupported resource budget decision receipt schema")
        _identifier(self.budget_id, "budget_id")
        _text(self.budget_contract_ref, "budget_contract_ref")
        _identifier(self.subject_id, "subject_id")
        _text(self.usage_aggregation_receipt_ref, "usage aggregation receipt ref")
        if not isinstance(self.subject_kind, BudgetSubjectKind):
            raise ValueError("subject_kind must use the closed enum")
        if type(self.budget_limits) is not tuple or not self.budget_limits or any(
            type(item) is not ResourceLimitV1 for item in self.budget_limits
        ):
            raise ValueError("budget_limits must be a nonempty typed limit vector")
        if type(self.actual_costs) is not tuple or any(
            type(item) is not ResourceCostV1 for item in self.actual_costs
        ):
            raise ValueError("actual_costs must be a typed cost vector")
        budget_keys = tuple(_resource_key(item) for item in self.budget_limits)
        actual_keys = tuple(_resource_key(item) for item in self.actual_costs)
        if len(set(budget_keys)) != len(budget_keys):
            raise ValueError("embedded budget limits cannot repeat a resource key")
        if len(set(actual_keys)) != len(actual_keys):
            raise ValueError("embedded actual costs cannot repeat a resource key")
        if type(self.limit_decisions) is not tuple or not self.limit_decisions:
            raise ValueError("budget receipt requires a nonempty tuple of limit decisions")
        if any(type(item) is not ResourceBudgetLimitDecisionV1 for item in self.limit_decisions):
            raise ValueError("limit_decisions must contain only decision records")
        limit_keys = tuple(item.key() for item in self.limit_decisions)
        if len(set(limit_keys)) != len(limit_keys):
            raise ValueError("budget receipt cannot repeat a resource limit key")
        if limit_keys != budget_keys:
            raise ValueError("limit decisions must preserve the embedded budget vector order")
        actual_by_key = dict(zip(actual_keys, self.actual_costs, strict=True))
        expected_decisions = tuple(
            _decision_for_limit(limit, actual_by_key.get(_resource_key(limit)))
            for limit in self.budget_limits
        )
        if self.limit_decisions != expected_decisions:
            raise ValueError("limit decisions do not match the embedded limits and costs")
        if type(self.unbudgeted_usage_costs) is not tuple or any(
            type(item) is not ResourceCostV1 for item in self.unbudgeted_usage_costs
        ):
            raise ValueError("unbudgeted_usage_costs must contain typed costs")
        expected_unbudgeted = tuple(
            cost
            for key, cost in zip(actual_keys, self.actual_costs, strict=True)
            if key not in set(budget_keys)
        )
        if self.unbudgeted_usage_costs != expected_unbudgeted:
            raise ValueError("unbudgeted usage must preserve every full extra cost record")
        if type(self.usage_receipt_refs) is not tuple:
            raise ValueError("usage_receipt_refs must be a tuple")
        for ref in self.usage_receipt_refs:
            _text(ref, "usage receipt ref")
        if len(set(self.usage_receipt_refs)) != len(self.usage_receipt_refs):
            raise ValueError("usage receipt refs must be unique")
        decision_refs = {
            ref
            for item in self.limit_decisions
            for ref in (item.usage_receipt_ref, item.unavailable_evidence_ref)
            if ref is not None
        }
        cost_refs = {
            ref
            for item in self.actual_costs
            for ref in (item.receipt_ref, item.unavailable_evidence_ref)
            if ref is not None
        }
        if self.usage_aggregation_receipt_ref not in self.usage_receipt_refs:
            raise ValueError("usage receipt refs must include the aggregation receipt")
        if not (decision_refs | cost_refs) <= set(self.usage_receipt_refs):
            raise ValueError("usage receipt refs must include every actual-cost evidence ref")
        if not isinstance(self.overall_decision, BudgetOverallDecision):
            raise ValueError("overall_decision must use the closed enum")
        expected_overall = _overall_decision(
            self.limit_decisions,
            self.unbudgeted_usage_costs,
        )
        if self.overall_decision is not expected_overall:
            raise ValueError("overall decision does not match per-limit evidence")
        if self.subject_kind is BudgetSubjectKind.RUN:
            if any(key.scope is not BudgetScope.WHOLE_RUN for key in budget_keys + actual_keys):
                raise ValueError("run decisions require only whole-run limits and costs")
            self._validate_whole_run_coverage(actual_by_key)
        else:
            if any(key.scope is BudgetScope.WHOLE_RUN for key in budget_keys + actual_keys):
                raise ValueError("work-item decisions cannot embed whole-run limits or costs")
            if self.whole_run_coverage is not None:
                raise ValueError("work-item receipt cannot carry whole-run coverage")
        if self.binding_status != RESOURCE_BUDGET_BINDING_STATUS:
            raise ValueError("pure resource budget decision must remain unbound")
        if self.design_origin != RESOURCE_BUDGET_DESIGN_ORIGIN:
            raise ValueError("resource budget receipt must remain locally_designed")
        if self.implementation_status != RUN_CONTRACT_IMPLEMENTATION_STATUS:
            raise ValueError("resource budget receipt cannot promote implementation status")

    def _validate_whole_run_coverage(
        self,
        actual_by_key: dict[ResourceKeyV1, ResourceCostV1],
    ) -> None:
        """Close typed overhead coverage against the embedded actual-cost vector."""

        if type(self.whole_run_coverage) is not WholeRunCoverageReceiptV1:
            raise ValueError("run decision requires a typed whole-run coverage receipt")
        if (
            self.whole_run_coverage.usage_aggregation_receipt_ref
            != self.usage_aggregation_receipt_ref
        ):
            raise ValueError("whole-run coverage and decision aggregation receipts must match")
        coverage_by_key = {
            entry.cost_key: entry for entry in self.whole_run_coverage.entries
        }
        required_actual_keys = {
            key for key in actual_by_key if key.role in _WHOLE_RUN_COMPONENT_ROLES
        }
        if set(coverage_by_key) != required_actual_keys:
            raise ValueError("whole-run coverage must include every required-component cost")
        for key, entry in coverage_by_key.items():
            cost = actual_by_key[key]
            evidence_refs = {
                ref
                for ref in (cost.receipt_ref, cost.unavailable_evidence_ref)
                if ref is not None
            }
            if entry.evidence_ref not in evidence_refs:
                raise ValueError("whole-run coverage evidence must match the actual cost")

    def to_dict(self) -> dict[str, object]:
        """Serialize embedded vectors, derived decisions, and full extra costs."""

        return {
            "schema": self.schema,
            "budget_id": self.budget_id,
            "budget_contract_ref": self.budget_contract_ref,
            "subject_kind": self.subject_kind.value,
            "subject_id": self.subject_id,
            "usage_aggregation_receipt_ref": self.usage_aggregation_receipt_ref,
            "budget_limits": [item.to_dict() for item in self.budget_limits],
            "actual_costs": [item.to_dict() for item in self.actual_costs],
            "limit_decisions": [item.to_dict() for item in self.limit_decisions],
            "unbudgeted_usage_costs": [
                item.to_dict() for item in self.unbudgeted_usage_costs
            ],
            "usage_receipt_refs": list(self.usage_receipt_refs),
            "overall_decision": self.overall_decision.value,
            "whole_run_coverage": (
                self.whole_run_coverage.to_dict()
                if self.whole_run_coverage is not None
                else None
            ),
            "binding_status": self.binding_status,
            "design_origin": self.design_origin,
            "implementation_status": self.implementation_status,
        }


def _resource_key(value: ResourceLimitV1 | ResourceCostV1) -> ResourceKeyV1:
    return ResourceKeyV1(value.role, value.scope, value.dimension)


def _decision_for_limit(
    limit: ResourceLimitV1,
    cost: ResourceCostV1 | None,
) -> ResourceBudgetLimitDecisionV1:
    """Derive one limit decision solely from the embedded limit and cost."""

    if cost is None:
        return ResourceBudgetLimitDecisionV1(
            role=limit.role,
            scope=limit.scope,
            dimension=limit.dimension,
            max_value=limit.max_value,
            usage_status=UsageValueStatus.UNAVAILABLE,
            usage_value=None,
            numeric_relation=BudgetNumericRelation.UNAVAILABLE,
            usage_receipt_ref=None,
            unavailable_reason="exact scoped usage key is missing",
            unavailable_evidence_ref=None,
        )
    if cost.status is UsageValueStatus.UNAVAILABLE:
        relation = BudgetNumericRelation.UNAVAILABLE
    else:
        if cost.value is None:
            raise ValueError("available resource cost cannot have a null value")
        relation = (
            BudgetNumericRelation.EXCEEDED
            if cost.value > limit.max_value
            else BudgetNumericRelation.WITHIN_LIMIT
        )
    return ResourceBudgetLimitDecisionV1(
        role=limit.role,
        scope=limit.scope,
        dimension=limit.dimension,
        max_value=limit.max_value,
        usage_status=cost.status,
        usage_value=cost.value,
        numeric_relation=relation,
        usage_receipt_ref=cost.receipt_ref,
        unavailable_reason=cost.unavailable_reason,
        unavailable_evidence_ref=cost.unavailable_evidence_ref,
    )


def _overall_decision(
    decisions: tuple[ResourceBudgetLimitDecisionV1, ...],
    unbudgeted_usage_costs: tuple[ResourceCostV1, ...],
) -> BudgetOverallDecision:
    """Apply the preregistered evidence rule without promoting estimates."""

    if any(
        item.usage_status is UsageValueStatus.OBSERVED
        and item.numeric_relation is BudgetNumericRelation.EXCEEDED
        for item in decisions
    ):
        return BudgetOverallDecision.EXCEEDED
    if unbudgeted_usage_costs:
        return BudgetOverallDecision.UNVERIFIED
    if all(
        item.usage_status is UsageValueStatus.OBSERVED
        and item.numeric_relation is BudgetNumericRelation.WITHIN_LIMIT
        for item in decisions
    ):
        return BudgetOverallDecision.WITHIN_LIMIT
    return BudgetOverallDecision.UNVERIFIED


def evaluate_resource_budget_v1(
    *,
    budget: ResourceBudgetContractV1 | RunBudgetV1,
    subject_kind: BudgetSubjectKind,
    subject_id: str,
    costs: tuple[ResourceCostV1, ...],
    budget_contract_ref: str,
    usage_aggregation_receipt_ref: str,
    whole_run_coverage: WholeRunCoverageReceiptV1 | None = None,
) -> ResourceBudgetDecisionReceiptV1:
    """Compare an exact scoped usage vector with a frozen budget.

    Whole-run callers must supply independently aggregated scorer, recovery,
    persistence, and orchestration costs. Summing work-item costs alone does
    not satisfy that contract. The returned receipt remains formally unbound.
    """

    if subject_kind is BudgetSubjectKind.WORK_ITEM:
        if type(budget) is not ResourceBudgetContractV1:
            raise ValueError("work-item decisions require a ResourceBudgetContractV1")
        if whole_run_coverage is not None:
            raise ValueError("work-item decisions cannot claim whole-run overhead")
    elif subject_kind is BudgetSubjectKind.RUN:
        if type(budget) is not RunBudgetV1:
            raise ValueError("run decisions require a RunBudgetV1")
        if type(whole_run_coverage) is not WholeRunCoverageReceiptV1:
            raise ValueError("run decisions require typed whole-run overhead coverage")
    else:
        raise ValueError("subject_kind must use the closed enum")
    _identifier(subject_id, "subject_id")
    _text(budget_contract_ref, "budget contract ref")
    _text(usage_aggregation_receipt_ref, "usage aggregation receipt ref")
    if type(costs) is not tuple or any(type(item) is not ResourceCostV1 for item in costs):
        raise ValueError("costs must be a tuple of ResourceCostV1 records")
    if subject_kind is BudgetSubjectKind.WORK_ITEM and any(
        item.scope is BudgetScope.WHOLE_RUN for item in costs
    ):
        raise ValueError("work-item decisions cannot consume whole-run costs")
    if subject_kind is BudgetSubjectKind.RUN and any(
        item.scope is not BudgetScope.WHOLE_RUN for item in costs
    ):
        raise ValueError("run decisions require whole-run scoped actual costs")

    costs_by_key: dict[ResourceKeyV1, ResourceCostV1] = {}
    for cost in costs:
        key = _resource_key(cost)
        if key in costs_by_key:
            raise ValueError("costs cannot repeat a role, scope, and dimension")
        costs_by_key[key] = cost

    limit_decisions: list[ResourceBudgetLimitDecisionV1] = []
    limit_keys: set[ResourceKeyV1] = set()
    for limit in budget.limits:
        key = _resource_key(limit)
        limit_keys.add(key)
        cost = costs_by_key.get(key)
        limit_decisions.append(_decision_for_limit(limit, cost))

    unbudgeted_costs = tuple(
        cost for key, cost in costs_by_key.items() if key not in limit_keys
    )
    usage_receipt_refs: list[str] = [usage_aggregation_receipt_ref]
    for cost in costs:
        for ref in (cost.receipt_ref, cost.unavailable_evidence_ref):
            if ref is not None and ref not in usage_receipt_refs:
                usage_receipt_refs.append(ref)
    decisions = tuple(limit_decisions)
    return ResourceBudgetDecisionReceiptV1(
        budget_id=budget.budget_id,
        budget_contract_ref=budget_contract_ref,
        subject_kind=subject_kind,
        subject_id=subject_id,
        usage_aggregation_receipt_ref=usage_aggregation_receipt_ref,
        budget_limits=budget.limits,
        actual_costs=costs,
        limit_decisions=decisions,
        unbudgeted_usage_costs=unbudgeted_costs,
        usage_receipt_refs=tuple(usage_receipt_refs),
        overall_decision=_overall_decision(decisions, unbudgeted_costs),
        whole_run_coverage=whole_run_coverage,
    )


__all__ = [
    "RESOURCE_BUDGET_DECISION_RECEIPT_SCHEMA",
    "RESOURCE_BUDGET_DESIGN_ORIGIN",
    "RESOURCE_BUDGET_BINDING_STATUS",
    "RESOURCE_BUDGET_LIMIT_DECISION_SCHEMA",
    "WHOLE_RUN_COVERAGE_ENTRY_SCHEMA",
    "WHOLE_RUN_COVERAGE_RECEIPT_SCHEMA",
    "BudgetNumericRelation",
    "BudgetOverallDecision",
    "BudgetSubjectKind",
    "ResourceBudgetDecisionReceiptV1",
    "ResourceBudgetLimitDecisionV1",
    "ResourceKeyV1",
    "WholeRunCoverageEntryV1",
    "WholeRunCoverageReceiptV1",
    "evaluate_resource_budget_v1",
]
