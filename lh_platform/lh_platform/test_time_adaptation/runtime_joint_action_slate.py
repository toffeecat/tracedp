"""Compile one immutable cross-task action slate at a runtime decision boundary.

This locally designed compiler joins existing resource, output-progress, and
output-semantic candidates before any intervention executes. It does not choose
an action or assign a scalar value; those responsibilities belong to the fixed,
TraceDP, and JIT policies that consume the same slate.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    PortableRuntimeKernel,
)
from lh_platform.test_time_adaptation.runtime_output_obligation import (
    RuntimeOutputObligation,
    compile_runtime_output_obligations,
)
from lh_platform.test_time_adaptation.runtime_trace_dp import (
    RuntimeRepairCandidate,
    RuntimeRepairCapability,
    compile_runtime_progress_candidates,
    compile_runtime_repair_candidates,
    compile_runtime_semantic_candidates,
)


def decoded_slate_int(value: object, *, field: str) -> int:
    """Decode one integer slate field without accepting a coerced scalar.

    `int(True)` and `int(1.0)` both succeed, and Python then treats `True == 1`
    and `1.0 == 1` as equal, so a plain dict-equality check on a replayed payload
    cannot detect a silently coerced scalar.  Rejecting bool and float here keeps
    every integer field in this module consistent with
    `RuntimeActionTargetBinding.from_dict`, which already guards its version.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"runtime joint action slate {field} is invalid")
    return value


def decoded_resource_versions(value: object) -> tuple[tuple[str, int], ...]:
    """Decode `resource_versions` as exactly `[str, int]` pairs."""

    if not isinstance(value, list):
        raise ValueError("runtime joint action slate resource versions are invalid")
    decoded: list[tuple[str, int]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("runtime joint action slate resource version entry is invalid")
        decoded.append(
            (
                str(item[0]),
                decoded_slate_int(item[1], field="resource version"),
            )
        )
    return tuple(decoded)

RuntimeResourceVersionKey = tuple[tuple[str, int], ...]
RuntimeObligationVersionKey = tuple[str, str, RuntimeResourceVersionKey, str]
RuntimeFailureDiagnosisIncidentKey = tuple[
    RuntimeResourceVersionKey, RuntimeObligationVersionKey
]


def runtime_failure_diagnosis_incident_key(
    *,
    resource_versions: RuntimeResourceVersionKey,
    obligation_version_key: RuntimeObligationVersionKey,
) -> RuntimeFailureDiagnosisIncidentKey:
    """Identify one execution-failure incident at one residual task state."""

    return resource_versions, obligation_version_key


class RuntimeJointActionType(StrEnum):
    """Name one behavior that remains stable when task-local targets change."""

    CONTINUE_ACTOR = "continue_actor"
    RESTORE_RESOURCE = "restore_resource"
    REQUEST_OUTPUT_COMPLETION = "request_output_completion"
    REQUEST_ACTOR_CONVERGENCE = "request_actor_convergence"
    REQUEST_FAILURE_DIAGNOSIS = "request_failure_diagnosis"
    REQUEST_REGISTERED_CHECK = "request_registered_check"
    REQUEST_PREDICATE_DIAGNOSIS = "request_predicate_diagnosis"
    RUN_REGISTERED_CHECK = "run_registered_check"
    RUN_TARGETED_REPAIR = "run_targeted_repair"
    SUBMIT_CURRENT_STATE = "submit_current_state"
    REQUEST_SEMANTIC_CHECK = "request_semantic_check"


@dataclass(frozen=True)
class RuntimeActionTargetBinding:
    """Bind a transferable action type to this task's resource and graph state."""

    resource_id: str | None
    resource_version: int | None
    task_id: str | None = None
    obligation_id: str | None = None
    obligation_version_key: tuple[str, str, tuple[tuple[str, int], ...], str] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "resource_id": self.resource_id,
            "resource_version": self.resource_version,
            "task_id": self.task_id,
            "obligation_id": self.obligation_id,
            "obligation_version_key": (
                [
                    self.obligation_version_key[0],
                    self.obligation_version_key[1],
                    [list(item) for item in self.obligation_version_key[2]],
                    self.obligation_version_key[3],
                ]
                if self.obligation_version_key is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeActionTargetBinding:
        """Decode a target without accepting omitted version or identity fields."""

        expected = {
            "resource_id",
            "resource_version",
            "task_id",
            "obligation_id",
            "obligation_version_key",
        }
        if set(payload) != expected:
            raise ValueError("runtime action target binding fields changed")
        raw_key = payload.get("obligation_version_key")
        key = None
        if raw_key is not None:
            if (
                not isinstance(raw_key, list)
                or len(raw_key) != 4
                or not isinstance(raw_key[2], list)
            ):
                raise ValueError("runtime action obligation binding is invalid")
            key = (
                str(raw_key[0]),
                str(raw_key[1]),
                tuple((str(item[0]), int(item[1])) for item in raw_key[2]),
                str(raw_key[3]),
            )
        resource_version = payload.get("resource_version")
        if resource_version is not None and (
            isinstance(resource_version, bool) or not isinstance(resource_version, int)
        ):
            raise ValueError("runtime action resource version is invalid")
        return cls(
            resource_id=(
                str(payload["resource_id"]) if payload.get("resource_id") is not None else None
            ),
            resource_version=resource_version,
            task_id=str(payload["task_id"]) if payload.get("task_id") is not None else None,
            obligation_id=(
                str(payload["obligation_id"]) if payload.get("obligation_id") is not None else None
            ),
            obligation_version_key=key,
        )


@dataclass(frozen=True)
class RuntimeActionExecutionContract:
    """Bind one candidate to an executable operation and observable outcome."""

    target_predicate: str
    supporting_evidence: tuple[str, ...]
    scope: tuple[str, ...]
    proposed_check_or_repair: Mapping[str, object]
    time_limit_ms: int
    success_postcondition: str
    failure_branch: str
    delivery_only: bool = False

    def __post_init__(self) -> None:
        operation = self.proposed_check_or_repair.get("operation")
        arguments = self.proposed_check_or_repair.get("arguments")
        if (
            not self.target_predicate.strip()
            or not self.supporting_evidence
            or any(not item for item in self.supporting_evidence)
            or len(self.supporting_evidence) != len(set(self.supporting_evidence))
            or not self.scope
            or any(not item for item in self.scope)
            or len(self.scope) != len(set(self.scope))
            or set(self.proposed_check_or_repair) != {"operation", "arguments"}
            or not isinstance(operation, str)
            or not operation
            or not isinstance(arguments, Mapping)
            or isinstance(self.time_limit_ms, bool)
            or not isinstance(self.time_limit_ms, int)
            or self.time_limit_ms < 1
            or not self.success_postcondition.strip()
            or not self.failure_branch.strip()
            or type(self.delivery_only) is not bool
        ):
            raise ValueError("runtime action execution contract is incomplete")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_predicate": self.target_predicate,
            "supporting_evidence": list(self.supporting_evidence),
            "scope": list(self.scope),
            "proposed_check_or_repair": {
                "operation": self.proposed_check_or_repair["operation"],
                "arguments": dict(self.proposed_check_or_repair["arguments"]),
            },
            "time_limit_ms": self.time_limit_ms,
            "success_postcondition": self.success_postcondition,
            "failure_branch": self.failure_branch,
            "delivery_only": self.delivery_only,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeActionExecutionContract:
        """Decode the complete action contract without inferring omitted behavior."""

        expected = {
            "target_predicate",
            "supporting_evidence",
            "scope",
            "proposed_check_or_repair",
            "time_limit_ms",
            "success_postcondition",
            "failure_branch",
            "delivery_only",
        }
        evidence = payload.get("supporting_evidence")
        scope = payload.get("scope")
        operation = payload.get("proposed_check_or_repair")
        if (
            set(payload) != expected
            or not isinstance(evidence, list)
            or not isinstance(scope, list)
            or not isinstance(operation, Mapping)
            or isinstance(payload.get("time_limit_ms"), bool)
            or not isinstance(payload.get("time_limit_ms"), int)
            or type(payload.get("delivery_only")) is not bool
        ):
            raise ValueError("runtime action execution contract payload is invalid")
        return cls(
            target_predicate=str(payload["target_predicate"]),
            supporting_evidence=tuple(str(item) for item in evidence),
            scope=tuple(str(item) for item in scope),
            proposed_check_or_repair=operation,
            time_limit_ms=payload["time_limit_ms"],
            success_postcondition=str(payload["success_postcondition"]),
            failure_branch=str(payload["failure_branch"]),
            delivery_only=payload["delivery_only"],
        )


@dataclass(frozen=True)
class RuntimeJointActionCandidate:
    """Describe one executable action instance without collapsing its target."""

    instance_action_id: str
    action_type: RuntimeJointActionType
    target_binding: RuntimeActionTargetBinding
    affected_task_ids: tuple[str, ...]
    cost_units: int
    cost_bytes: int
    transition_model_id: str
    legality_evidence_id: str
    execution_contract: RuntimeActionExecutionContract | None = None
    conditional_future_option_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.instance_action_id
            or not isinstance(self.action_type, RuntimeJointActionType)
            or self.cost_units < 0
            or self.cost_bytes < 0
            or not self.transition_model_id
            or not self.legality_evidence_id
            or self.conditional_future_option_id == ""
        ):
            raise ValueError("runtime joint action candidate is incomplete")
        if self.action_type is RuntimeJointActionType.CONTINUE_ACTOR:
            if self.cost_units != 0 or self.target_binding.resource_id is not None:
                raise ValueError("continue_actor must be the unique zero-cost action")
        elif self.action_type in {
            RuntimeJointActionType.REQUEST_REGISTERED_CHECK,
            RuntimeJointActionType.REQUEST_PREDICATE_DIAGNOSIS,
            RuntimeJointActionType.RUN_REGISTERED_CHECK,
            RuntimeJointActionType.RUN_TARGETED_REPAIR,
        }:
            if self.cost_units < 1 or self.target_binding.task_id is None:
                raise ValueError("predicate action requires a task target and cost")
        elif self.cost_units < 1 or self.target_binding.resource_id is None:
            raise ValueError("runtime intervention requires a bound target and cost")
        if (
            self.conditional_future_option_id is not None
            and self.action_type is not RuntimeJointActionType.RUN_TARGETED_REPAIR
        ):
            raise ValueError("only a targeted repair can correspond to a future option")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "instance_action_id": self.instance_action_id,
            "action_type": self.action_type.value,
            "target_binding": self.target_binding.to_dict(),
            "affected_task_ids": list(self.affected_task_ids),
            "cost_units": self.cost_units,
            "cost_bytes": self.cost_bytes,
            "transition_model_id": self.transition_model_id,
            "legality_evidence_id": self.legality_evidence_id,
        }
        if self.execution_contract is not None:
            payload["execution_contract"] = self.execution_contract.to_dict()
        if self.conditional_future_option_id is not None:
            payload["conditional_future_option_id"] = self.conditional_future_option_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeJointActionCandidate:
        """Decode one exact action instance for durable policy replay."""

        expected = {
            "instance_action_id",
            "action_type",
            "target_binding",
            "affected_task_ids",
            "cost_units",
            "cost_bytes",
            "transition_model_id",
            "legality_evidence_id",
        }
        if "execution_contract" in payload:
            expected.add("execution_contract")
        if "conditional_future_option_id" in payload:
            expected.add("conditional_future_option_id")
        target = payload.get("target_binding")
        affected = payload.get("affected_task_ids")
        raw_contract = payload.get("execution_contract")
        if (
            set(payload) != expected
            or not isinstance(target, Mapping)
            or not isinstance(affected, list)
            or ("execution_contract" in payload and not isinstance(raw_contract, Mapping))
        ):
            raise ValueError("runtime joint action candidate payload is invalid")
        return cls(
            instance_action_id=str(payload["instance_action_id"]),
            action_type=RuntimeJointActionType(str(payload["action_type"])),
            target_binding=RuntimeActionTargetBinding.from_dict(target),
            affected_task_ids=tuple(str(item) for item in affected),
            cost_units=int(payload["cost_units"]),
            cost_bytes=int(payload["cost_bytes"]),
            transition_model_id=str(payload["transition_model_id"]),
            legality_evidence_id=str(payload["legality_evidence_id"]),
            execution_contract=(
                RuntimeActionExecutionContract.from_dict(raw_contract)
                if isinstance(raw_contract, Mapping)
                else None
            ),
            conditional_future_option_id=(
                str(payload["conditional_future_option_id"])
                if payload.get("conditional_future_option_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class RuntimeConditionalFutureOption:
    """Describe a configured repair that becomes legal after a named check outcome."""

    option_id: str
    action_type: RuntimeJointActionType
    target_binding: RuntimeActionTargetBinding
    affected_task_ids: tuple[str, ...]
    cost_units: int
    cost_bytes: int
    transition_model_id: str
    prerequisite_action_id: str
    prerequisite_outcomes: tuple[str, ...]
    repair_contract_id: str
    check_contract_id: str
    resource_versions: RuntimeResourceVersionKey
    execution_contract: RuntimeActionExecutionContract

    def __post_init__(self) -> None:
        if (
            not self.option_id
            or self.action_type is not RuntimeJointActionType.RUN_TARGETED_REPAIR
            or self.target_binding.task_id is None
            or not self.affected_task_ids
            or self.cost_units < 1
            or self.cost_bytes < 0
            or not self.transition_model_id
            or not self.prerequisite_action_id
            or self.prerequisite_outcomes != ("refuted", "unknown")
            or not self.repair_contract_id
            or not self.check_contract_id
            or not self.resource_versions
            or tuple(sorted(self.resource_versions)) != self.resource_versions
        ):
            raise ValueError("runtime conditional future option is incomplete")

    @property
    def instance_action_id(self) -> str:
        """Expose a planning identity without making this option root-executable."""

        return self.option_id

    def to_dict(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "action_type": self.action_type.value,
            "target_binding": self.target_binding.to_dict(),
            "affected_task_ids": list(self.affected_task_ids),
            "cost_units": self.cost_units,
            "cost_bytes": self.cost_bytes,
            "transition_model_id": self.transition_model_id,
            "prerequisite_action_id": self.prerequisite_action_id,
            "prerequisite_outcomes": list(self.prerequisite_outcomes),
            "repair_contract_id": self.repair_contract_id,
            "check_contract_id": self.check_contract_id,
            "resource_versions": [list(item) for item in self.resource_versions],
            "execution_contract": self.execution_contract.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeConditionalFutureOption:
        """Restore a deferred option without promoting it to an executable candidate."""

        expected = {
            "option_id", "action_type", "target_binding", "affected_task_ids",
            "cost_units", "cost_bytes", "transition_model_id",
            "prerequisite_action_id", "prerequisite_outcomes",
            "repair_contract_id", "check_contract_id", "resource_versions",
            "execution_contract",
        }
        target = payload.get("target_binding")
        affected = payload.get("affected_task_ids")
        outcomes = payload.get("prerequisite_outcomes")
        versions = payload.get("resource_versions")
        contract = payload.get("execution_contract")
        if (
            set(payload) != expected
            or not isinstance(target, Mapping)
            or not isinstance(affected, list)
            or not isinstance(outcomes, list)
            or not isinstance(versions, list)
            or not isinstance(contract, Mapping)
        ):
            raise ValueError("runtime conditional future option payload is invalid")
        return cls(
            option_id=str(payload["option_id"]),
            action_type=RuntimeJointActionType(str(payload["action_type"])),
            target_binding=RuntimeActionTargetBinding.from_dict(target),
            affected_task_ids=tuple(str(item) for item in affected),
            cost_units=int(payload["cost_units"]),
            cost_bytes=int(payload["cost_bytes"]),
            transition_model_id=str(payload["transition_model_id"]),
            prerequisite_action_id=str(payload["prerequisite_action_id"]),
            prerequisite_outcomes=tuple(str(item) for item in outcomes),
            repair_contract_id=str(payload["repair_contract_id"]),
            check_contract_id=str(payload["check_contract_id"]),
            resource_versions=tuple((str(item[0]), int(item[1])) for item in versions),
            execution_contract=RuntimeActionExecutionContract.from_dict(contract),
        )


@dataclass(frozen=True)
class RuntimeJointActionSlate:
    """Freeze all actions seen by every selector at one Actor trigger."""

    session_id: str
    trigger_action_id: str
    resource_versions: tuple[tuple[str, int], ...]
    budget_before: int
    no_progress_actor_actions: int
    progress_trigger_threshold: int
    candidates: tuple[RuntimeJointActionCandidate, ...]
    conditional_future_options: tuple[RuntimeConditionalFutureOption, ...] = ()
    actor_time_budget_ms: int | None = None
    actor_time_remaining_ms: int | None = None

    schema_version = "runtime_joint_action_slate_v3"
    timed_legacy_schema_version = "runtime_joint_action_slate_v2"
    legacy_schema_version = "runtime_joint_action_slate_v1"

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or not self.trigger_action_id
            or self.budget_before < 0
            or self.no_progress_actor_actions < 0
            or self.progress_trigger_threshold < 1
        ):
            raise ValueError("runtime joint action slate boundary is invalid")
        time_values = (self.actor_time_budget_ms, self.actor_time_remaining_ms)
        if (time_values[0] is None) != (time_values[1] is None):
            raise ValueError("runtime joint action slate time boundary is incomplete")
        if time_values[0] is not None and (
            time_values[0] < 1
            or time_values[1] < 0  # type: ignore[operator]
            or time_values[1] > time_values[0]  # type: ignore[operator]
        ):
            raise ValueError("runtime joint action slate time boundary is invalid")
        instance_ids = [item.instance_action_id for item in self.candidates]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("runtime joint action slate repeats an action instance")
        continue_actions = [
            item
            for item in self.candidates
            if item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        ]
        if len(continue_actions) != 1:
            raise ValueError("runtime joint action slate requires one continue_actor")
        expected_order = tuple(sorted(self.candidates, key=_candidate_order_key))
        if self.candidates != expected_order:
            raise ValueError("runtime joint action slate ordering differs")
        option_ids = [item.option_id for item in self.conditional_future_options]
        if (
            len(option_ids) != len(set(option_ids))
            or set(option_ids) & set(instance_ids)
            or self.conditional_future_options
            != tuple(sorted(self.conditional_future_options, key=lambda item: item.option_id))
            or any(
                item.prerequisite_action_id not in set(instance_ids)
                for item in self.conditional_future_options
            )
        ):
            raise ValueError("runtime conditional future option universe differs")

    @property
    def capability_signature(self) -> str:
        """Describe transferable action-type multiplicity without task-local paths."""

        counts = Counter(item.action_type.value for item in self.candidates)
        return "|".join(f"{key}:{counts[key]}" for key in sorted(counts))

    @property
    def nonempty_action_count(self) -> int:
        return sum(
            item.action_type is not RuntimeJointActionType.CONTINUE_ACTOR
            for item in self.candidates
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": (
                self.schema_version
                if self.conditional_future_options
                else self.timed_legacy_schema_version
                if self.actor_time_budget_ms is not None
                else self.legacy_schema_version
            ),
            "session_id": self.session_id,
            "trigger_action_id": self.trigger_action_id,
            "resource_versions": [list(item) for item in self.resource_versions],
            "budget_before": self.budget_before,
            "no_progress_actor_actions": self.no_progress_actor_actions,
            "progress_trigger_threshold": self.progress_trigger_threshold,
            "capability_signature": self.capability_signature,
            "nonempty_action_count": self.nonempty_action_count,
            "candidates": [item.to_dict() for item in self.candidates],
        }
        if self.actor_time_budget_ms is not None:
            payload.update(
                {
                    "actor_time_budget_ms": self.actor_time_budget_ms,
                    "actor_time_remaining_ms": self.actor_time_remaining_ms,
                }
            )
        if self.conditional_future_options:
            payload["conditional_future_options"] = [
                item.to_dict() for item in self.conditional_future_options
            ]
            payload.setdefault("actor_time_budget_ms", None)
            payload.setdefault("actor_time_remaining_ms", None)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeJointActionSlate:
        """Load a slate and independently check its two derived summaries."""

        common_fields = {
            "schema_version",
            "session_id",
            "trigger_action_id",
            "resource_versions",
            "budget_before",
            "no_progress_actor_actions",
            "progress_trigger_threshold",
            "capability_signature",
            "nonempty_action_count",
            "candidates",
        }
        schema_version = payload.get("schema_version")
        expected = set(common_fields)
        if schema_version in {cls.schema_version, cls.timed_legacy_schema_version}:
            expected.update({"actor_time_budget_ms", "actor_time_remaining_ms"})
        if schema_version == cls.schema_version:
            expected.add("conditional_future_options")
        versions = payload.get("resource_versions")
        candidates = payload.get("candidates")
        options = payload.get("conditional_future_options", [])
        if (
            set(payload) != expected
            or schema_version not in {
                cls.schema_version,
                cls.timed_legacy_schema_version,
                cls.legacy_schema_version,
            }
            or not isinstance(versions, list)
            or not isinstance(candidates, list)
            or any(not isinstance(item, Mapping) for item in candidates)
            or not isinstance(options, list)
            or any(not isinstance(item, Mapping) for item in options)
        ):
            raise ValueError("runtime joint action slate payload is invalid")
        slate = cls(
            session_id=str(payload["session_id"]),
            trigger_action_id=str(payload["trigger_action_id"]),
            resource_versions=decoded_resource_versions(versions),
            budget_before=decoded_slate_int(
                payload["budget_before"], field="budget before"
            ),
            no_progress_actor_actions=decoded_slate_int(
                payload["no_progress_actor_actions"],
                field="no-progress action count",
            ),
            progress_trigger_threshold=decoded_slate_int(
                payload["progress_trigger_threshold"],
                field="progress trigger threshold",
            ),
            candidates=tuple(RuntimeJointActionCandidate.from_dict(item) for item in candidates),
            conditional_future_options=tuple(
                RuntimeConditionalFutureOption.from_dict(item) for item in options
            ),
            actor_time_budget_ms=(
                decoded_slate_int(
                    payload["actor_time_budget_ms"], field="Actor time budget"
                )
                if schema_version in {cls.schema_version, cls.timed_legacy_schema_version}
                and payload["actor_time_budget_ms"] is not None
                else None
            ),
            actor_time_remaining_ms=(
                decoded_slate_int(
                    payload["actor_time_remaining_ms"], field="Actor time remaining"
                )
                if schema_version in {cls.schema_version, cls.timed_legacy_schema_version}
                and payload["actor_time_remaining_ms"] is not None
                else None
            ),
        )
        if (
            payload.get("capability_signature") != slate.capability_signature
            or payload.get("nonempty_action_count") != slate.nonempty_action_count
        ):
            raise ValueError("runtime joint action slate summary differs")
        # Reject scalar coercion during durable replay; true and 1.0 must not silently become 1.
        if json.dumps(
            dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) != json.dumps(
            slate.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ):
            raise ValueError("runtime joint action slate payload is not canonical")
        return slate


def _candidate_order_key(
    candidate: RuntimeJointActionCandidate,
) -> tuple[int, str, str]:
    return (
        0 if candidate.action_type is RuntimeJointActionType.CONTINUE_ACTOR else 1,
        candidate.action_type.value,
        candidate.instance_action_id,
    )


def _obligation_by_issue(
    kernel: PortableRuntimeKernel,
) -> dict[str, RuntimeOutputObligation]:
    return {item.obligation_id: item for item in compile_runtime_output_obligations(kernel)}


def _bound_candidate(
    *,
    candidate: RuntimeRepairCandidate,
    action_type: RuntimeJointActionType,
    kernel: PortableRuntimeKernel,
    transition_model_id: str,
    legality_evidence_id: str,
    obligation: RuntimeOutputObligation | None = None,
) -> RuntimeJointActionCandidate:
    """Preserve the local candidate while separating type from target identity."""

    if action_type is RuntimeJointActionType.RESTORE_RESOURCE:
        contract = RuntimeActionExecutionContract(
            target_predicate=f"required resource {candidate.resource_id} exists",
            supporting_evidence=(
                candidate.necessity_evidence_id,
                candidate.probability_evidence_id,
            ),
            scope=(candidate.resource_id, *candidate.affected_task_ids),
            proposed_check_or_repair={
                "operation": "restore_retained_resource",
                "arguments": {
                    "resource_id": candidate.resource_id,
                    "resource_version": kernel.resource_versions[candidate.resource_id],
                },
            },
            time_limit_ms=120_000,
            success_postcondition=(
                f"resource {candidate.resource_id} is observed present at a newer version"
            ),
            failure_branch=(
                "keep the prerequisite unresolved, record the restore failure, and replan"
            ),
        )
    else:
        if obligation is None:
            raise ValueError("runtime guidance requires an output obligation")
        operation = {
            RuntimeJointActionType.REQUEST_OUTPUT_COMPLETION: "deliver_output_completion_guidance",
            RuntimeJointActionType.REQUEST_SEMANTIC_CHECK: "deliver_semantic_check_guidance",
        }.get(action_type)
        if operation is None:
            raise ValueError("runtime bound action type lacks a contract")
        contract = RuntimeActionExecutionContract(
            target_predicate=(
                f"public output obligation {obligation.obligation_id} reaches a verified state"
            ),
            supporting_evidence=(legality_evidence_id,),
            scope=(obligation.task_id, *obligation.output_resource_ids),
            proposed_check_or_repair={
                "operation": operation,
                "arguments": {
                    "obligation_id": obligation.obligation_id,
                    "phase": obligation.phase.value,
                },
            },
            time_limit_ms=30_000,
            success_postcondition=(
                "guidance is delivered; the target predicate remains unverified"
            ),
            failure_branch="record delivery failure without changing task evidence",
            delivery_only=True,
        )

    return RuntimeJointActionCandidate(
        instance_action_id=candidate.action_id,
        action_type=action_type,
        target_binding=RuntimeActionTargetBinding(
            resource_id=candidate.resource_id,
            resource_version=kernel.resource_versions[candidate.resource_id],
            task_id=obligation.task_id if obligation is not None else None,
            obligation_id=obligation.obligation_id if obligation is not None else None,
            obligation_version_key=(obligation.version_key if obligation is not None else None),
        ),
        affected_task_ids=candidate.affected_task_ids,
        cost_units=candidate.cost_units,
        cost_bytes=candidate.cost_bytes,
        transition_model_id=transition_model_id,
        legality_evidence_id=legality_evidence_id,
        execution_contract=contract,
    )


def _task_lineages_overlap(
    kernel: PortableRuntimeKernel,
    left_task_ids: tuple[str, ...],
    right_task_ids: tuple[str, ...],
) -> bool:
    """Return whether either task set contains an ancestor of the other."""

    def lineage(task_id: str) -> set[str]:
        chain = {task_id}
        current = kernel.task_nodes.get(task_id)
        while current is not None and current.parent_id is not None:
            chain.add(current.parent_id)
            current = kernel.task_nodes.get(current.parent_id)
        return chain

    left_lineages = {task_id: lineage(task_id) for task_id in left_task_ids}
    right_lineages = {task_id: lineage(task_id) for task_id in right_task_ids}
    return any(
        left_id in right_chain or right_id in left_chain
        for left_id, left_chain in left_lineages.items()
        for right_id, right_chain in right_lineages.items()
    )


def compile_runtime_joint_action_slate(
    *,
    kernel: PortableRuntimeKernel,
    capabilities: dict[str, RuntimeRepairCapability],
    trigger_action_id: str,
    budget_before: int,
    no_progress_actor_actions: int,
    progress_trigger_threshold: int,
    excluded_resource_versions: frozenset[tuple[str, int]] = frozenset(),
    excluded_progress_resource_versions: frozenset[tuple[str, int]] = frozenset(),
    excluded_semantic_obligation_versions: frozenset[
        tuple[str, str, tuple[tuple[str, int], ...], str]
    ] = frozenset(),
    output_progress_enabled: bool = True,
    settlement_output_completion_enabled: bool = False,
    output_semantics_enabled: bool = True,
    actor_time_budget_ms: int | None = None,
    actor_time_remaining_ms: int | None = None,
    actor_convergence_reserve_ms: int | None = None,
    actor_submission_enabled: bool = False,
    failure_diagnosis_enabled: bool = False,
    failure_diagnosis_threshold: int = 2,
    excluded_failure_action_ids: frozenset[str] = frozenset(),
    excluded_failure_incident_keys: frozenset[
        RuntimeFailureDiagnosisIncidentKey
    ] = frozenset(),
    predicate_diagnosis_enabled: bool = False,
    excluded_predicate_incident_ids: frozenset[str] = frozenset(),
    registered_check_candidates: tuple[RuntimeJointActionCandidate, ...] = (),
    targeted_repair_candidates: tuple[RuntimeJointActionCandidate, ...] = (),
    conditional_future_options: tuple[RuntimeConditionalFutureOption, ...] = (),
) -> RuntimeJointActionSlate:
    """Compile every legal action from one pre-intervention Kernel snapshot."""

    if budget_before < 0 or no_progress_actor_actions < 0:
        raise ValueError("runtime joint action budget or progress count is invalid")
    if progress_trigger_threshold < 1:
        raise ValueError("runtime joint action progress threshold must be positive")
    if failure_diagnosis_threshold < 1:
        raise ValueError("runtime failure diagnosis threshold must be positive")
    if any(candidate.execution_contract is None for candidate in registered_check_candidates):
        raise ValueError("registered check candidate lacks an execution contract")
    if any(candidate.execution_contract is None for candidate in targeted_repair_candidates):
        raise ValueError("targeted repair candidate lacks an execution contract")
    if actor_convergence_reserve_ms is not None and (
        actor_time_budget_ms is None
        or actor_time_remaining_ms is None
        or actor_convergence_reserve_ms < 1
        or actor_convergence_reserve_ms > actor_time_budget_ms
    ):
        raise ValueError("runtime actor convergence time boundary is invalid")
    continue_time_limit_ms = (
        max(1, actor_time_remaining_ms)
        if actor_time_remaining_ms is not None
        else 120_000
    )
    actions = [
        RuntimeJointActionCandidate(
            instance_action_id=RuntimeJointActionType.CONTINUE_ACTOR.value,
            action_type=RuntimeJointActionType.CONTINUE_ACTOR,
            target_binding=RuntimeActionTargetBinding(None, None),
            affected_task_ids=(),
            cost_units=0,
            cost_bytes=0,
            transition_model_id="actor_continuation_pending_observation_v1",
            legality_evidence_id="continuous_actor_session_capability_v1",
            execution_contract=RuntimeActionExecutionContract(
                target_predicate="the residual public task remains unfinished",
                supporting_evidence=("continuous_actor_session_capability_v1",),
                scope=(kernel.session_id,),
                proposed_check_or_repair={
                    "operation": "continue_same_actor",
                    "arguments": {"trigger_action_id": trigger_action_id},
                },
                time_limit_ms=continue_time_limit_ms,
                success_postcondition=(
                    "the same Actor reaches the next observable tool or submission boundary"
                ),
                failure_branch="return to Runtime with the latest observable state",
            ),
        )
    ]
    _, repair_candidates = compile_runtime_repair_candidates(
        kernel,
        capabilities,
        excluded_resource_versions=excluded_resource_versions,
    )
    actions.extend(
        _bound_candidate(
            candidate=candidate,
            action_type=RuntimeJointActionType.RESTORE_RESOURCE,
            kernel=kernel,
            transition_model_id="exact_retention_restore_then_reobserve_v1",
            legality_evidence_id=candidate.probability_evidence_id,
        )
        for candidate in repair_candidates
        if candidate.cost_units <= budget_before
    )
    actions.extend(
        candidate
        for candidate in targeted_repair_candidates
        if candidate.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR
        and candidate.cost_units <= budget_before
    )
    actions.extend(
        candidate
        for candidate in registered_check_candidates
        if candidate.action_type
        in {
            RuntimeJointActionType.REQUEST_REGISTERED_CHECK,
            RuntimeJointActionType.RUN_REGISTERED_CHECK,
        }
        and candidate.cost_units <= budget_before
    )

    obligations = _obligation_by_issue(kernel)
    if output_progress_enabled and (
        no_progress_actor_actions >= progress_trigger_threshold
        or settlement_output_completion_enabled
    ):
        _, progress_candidates, _ = compile_runtime_progress_candidates(
            kernel,
            excluded_resource_versions=excluded_progress_resource_versions,
        )
        actions.extend(
            _bound_candidate(
                candidate=candidate,
                action_type=RuntimeJointActionType.REQUEST_OUTPUT_COMPLETION,
                kernel=kernel,
                transition_model_id="guidance_delivery_then_effect_pending_v1",
                legality_evidence_id=(
                    "natural_settlement_with_incomplete_public_output_v1"
                    if settlement_output_completion_enabled
                    else "public_output_progress_trigger_v1"
                ),
                obligation=obligations[candidate.planning_issue_id],
            )
            for candidate in progress_candidates
            if candidate.cost_units <= budget_before
        )
    if output_semantics_enabled:
        _, semantic_candidates, _, semantic_obligations = compile_runtime_semantic_candidates(
            kernel,
            excluded_obligation_versions=(excluded_semantic_obligation_versions),
        )
        concrete_check_task_ids = tuple(
            candidate.target_binding.task_id
            for candidate in registered_check_candidates
            if candidate.action_type
            in {
                RuntimeJointActionType.REQUEST_REGISTERED_CHECK,
                RuntimeJointActionType.RUN_REGISTERED_CHECK,
            }
            and candidate.target_binding.task_id is not None
        )
        bound_semantic_candidates = (
            _bound_candidate(
                candidate=candidate,
                action_type=RuntimeJointActionType.REQUEST_SEMANTIC_CHECK,
                kernel=kernel,
                transition_model_id="semantic_guidance_then_effect_pending_v1",
                legality_evidence_id=("complete_public_output_requires_independent_check_v1"),
                obligation=semantic_obligations[candidate.planning_issue_id],
            )
            for candidate in semantic_candidates
            if candidate.cost_units <= budget_before
        )
        # Prefer an executable public check over generic guidance for the same lineage.
        actions.extend(
            candidate
            for candidate in bound_semantic_candidates
            if not _task_lineages_overlap(
                kernel,
                (candidate.target_binding.task_id,),
                concrete_check_task_ids,
            )
        )
    if predicate_diagnosis_enabled and budget_before >= 1:
        # A failed predicate is not an execution error or a missing output file.
        for incident in kernel.predicate_incidents.values():
            evidence = kernel.evidence[incident.latest_evidence_id]
            if (
                incident.status != "open"
                or evidence.state.value != "refuted"
                or incident.incident_id in excluded_predicate_incident_ids
            ):
                continue
            actions.append(RuntimeJointActionCandidate(
                instance_action_id=(
                    f"diagnose-predicate:{incident.incident_id}:{incident.latest_evidence_id}"
                ),
                action_type=RuntimeJointActionType.REQUEST_PREDICATE_DIAGNOSIS,
                target_binding=RuntimeActionTargetBinding(None, None, task_id=incident.task_id),
                affected_task_ids=(incident.task_id,),
                cost_units=1, cost_bytes=0,
                transition_model_id="predicate_diagnosis_effect_pending_v1",
                legality_evidence_id=incident.latest_evidence_id,
                execution_contract=RuntimeActionExecutionContract(
                    target_predicate=incident.proposition,
                    supporting_evidence=(incident.latest_evidence_id,),
                    scope=(incident.task_id,),
                    proposed_check_or_repair={
                        "operation": "deliver_predicate_diagnosis_guidance",
                        "arguments": {
                            "incident_id": incident.incident_id,
                            "evidence_id": incident.latest_evidence_id,
                        },
                    },
                    time_limit_ms=30_000,
                    success_postcondition=(
                        "guidance is delivered; the predicate remains refuted until rechecked"
                    ),
                    failure_branch="record delivery failure and keep the incident open",
                    delivery_only=True,
                ),
            ))
    if failure_diagnosis_enabled and kernel.actor_failure_boundaries:
        boundary = tuple(kernel.actor_failure_boundaries.values())[-1]
        if (
            boundary.failure_count_since_resource_change >= failure_diagnosis_threshold
            and boundary.action_id not in excluded_failure_action_ids
            and boundary.resource_versions == tuple(sorted(kernel.resource_versions.items()))
        ):
            actions.extend(
                RuntimeJointActionCandidate(
                    instance_action_id=(
                        f"diagnose-failure:{boundary.action_id}:"
                        f"{boundary.failure_count_since_resource_change}:"
                        f"{obligation.obligation_id}"
                    ),
                    action_type=RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS,
                    target_binding=RuntimeActionTargetBinding(
                        resource_id=obligation.output_resource_ids[0],
                        resource_version=kernel.resource_versions[
                            obligation.output_resource_ids[0]
                        ],
                        task_id=obligation.task_id,
                        obligation_id=obligation.obligation_id,
                        obligation_version_key=obligation.version_key,
                    ),
                    affected_task_ids=boundary.candidate_task_ids,
                    cost_units=1,
                    cost_bytes=0,
                    transition_model_id=("failure_diagnosis_guidance_then_effect_pending_v1"),
                    legality_evidence_id=("versioned_no_effect_failure_boundary_v1"),
                    execution_contract=RuntimeActionExecutionContract(
                        target_predicate=(
                            f"failure boundary {boundary.action_id} no longer blocks "
                            f"{obligation.task_id}"
                        ),
                        supporting_evidence=(
                            boundary.association_evidence_id,
                            "versioned_no_effect_failure_boundary_v1",
                        ),
                        scope=(obligation.task_id, *obligation.output_resource_ids),
                        proposed_check_or_repair={
                            "operation": "deliver_failure_diagnosis_guidance",
                            "arguments": {
                                "failure_action_id": boundary.action_id,
                                "obligation_id": obligation.obligation_id,
                            },
                        },
                        time_limit_ms=30_000,
                        success_postcondition=(
                            "guidance is delivered; failure resolution remains unverified"
                        ),
                        failure_branch=(
                            "record delivery failure and keep the failure boundary unresolved"
                        ),
                        delivery_only=True,
                    ),
                )
                for obligation in obligations.values()
                if (
                    obligation.task_id in boundary.candidate_task_ids
                    and budget_before >= 1
                    and runtime_failure_diagnosis_incident_key(
                        resource_versions=boundary.resource_versions,
                        obligation_version_key=obligation.version_key,
                    )
                    not in excluded_failure_incident_keys
                )
            )
    if (
        actor_convergence_reserve_ms is not None
        and actor_time_remaining_ms is not None
        and actor_time_remaining_ms <= actor_convergence_reserve_ms
    ):
        # A convergence request is legal only for an already complete public output.
        _, _, _, convergence_obligations = compile_runtime_semantic_candidates(
            kernel,
            excluded_obligation_versions=frozenset(),
        )
        timed_action_type = (
            RuntimeJointActionType.SUBMIT_CURRENT_STATE
            if actor_submission_enabled
            else RuntimeJointActionType.REQUEST_ACTOR_CONVERGENCE
        )
        timed_action_prefix = (
            "submit-current-state" if actor_submission_enabled else "request-actor-convergence"
        )
        actions.extend(
            RuntimeJointActionCandidate(
                instance_action_id=(
                    f"{timed_action_prefix}:{obligation.obligation_id}:"
                    f"{obligation.output_resource_ids[0]}:"
                    f"{kernel.resource_versions[obligation.output_resource_ids[0]]}"
                ),
                action_type=timed_action_type,
                target_binding=RuntimeActionTargetBinding(
                    resource_id=obligation.output_resource_ids[0],
                    resource_version=kernel.resource_versions[obligation.output_resource_ids[0]],
                    task_id=obligation.task_id,
                    obligation_id=obligation.obligation_id,
                    obligation_version_key=obligation.version_key,
                ),
                affected_task_ids=obligation.affected_task_ids,
                cost_units=1,
                cost_bytes=0,
                transition_model_id=(
                    "pi_rpc_abort_then_harbor_verifier_v1"
                    if actor_submission_enabled
                    else "convergence_guidance_then_actor_tool_gate_v1"
                ),
                legality_evidence_id=(
                    "complete_public_output_idle_gateway_and_time_reserve_v1"
                    if actor_submission_enabled
                    else "complete_public_output_and_time_reserve_v1"
                ),
                execution_contract=RuntimeActionExecutionContract(
                    target_predicate=(
                        f"public output obligation {obligation.obligation_id} is ready "
                        "for official evaluation"
                    ),
                    supporting_evidence=(
                        "complete_public_output_idle_gateway_and_time_reserve_v1"
                        if actor_submission_enabled
                        else "complete_public_output_and_time_reserve_v1",
                    ),
                    scope=(obligation.task_id, *obligation.output_resource_ids),
                    proposed_check_or_repair={
                        "operation": (
                            "stop_actor_and_submit_current_state"
                            if actor_submission_enabled
                            else "deliver_actor_convergence_guidance"
                        ),
                        "arguments": {"obligation_id": obligation.obligation_id},
                    },
                    time_limit_ms=max(1, actor_time_remaining_ms),
                    success_postcondition=(
                        "current state is handed to the official evaluator; task success "
                        "remains evaluator-owned"
                        if actor_submission_enabled
                        else "guidance is delivered; task completion remains unverified"
                    ),
                    failure_branch=(
                        "record submission failure without claiming task success"
                        if actor_submission_enabled
                        else "record delivery failure without changing task evidence"
                    ),
                    delivery_only=not actor_submission_enabled,
                ),
            )
            for obligation in convergence_obligations.values()
            if budget_before >= 1
        )
    return RuntimeJointActionSlate(
        session_id=kernel.session_id,
        trigger_action_id=trigger_action_id,
        resource_versions=tuple(sorted(kernel.resource_versions.items())),
        budget_before=budget_before,
        no_progress_actor_actions=no_progress_actor_actions,
        progress_trigger_threshold=progress_trigger_threshold,
        candidates=tuple(sorted(actions, key=_candidate_order_key)),
        conditional_future_options=conditional_future_options,
        actor_time_budget_ms=actor_time_budget_ms,
        actor_time_remaining_ms=actor_time_remaining_ms,
    )


__all__ = [
    "RuntimeActionExecutionContract",
    "RuntimeActionTargetBinding",
    "RuntimeConditionalFutureOption",
    "RuntimeJointActionCandidate",
    "RuntimeFailureDiagnosisIncidentKey",
    "RuntimeJointActionSlate",
    "RuntimeJointActionType",
    "compile_runtime_joint_action_slate",
    "runtime_failure_diagnosis_incident_key",
]
