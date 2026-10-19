"""Bind frozen atomic estimates to Runtime-owned live option planning.

The estimate configuration can value only an action already present in the
immutable Runtime slate. Runtime still owns predicates, states, legal actions,
conditional successors, and transitions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from lh_platform.benchmarks.terminal_runtime_check import RuntimeCheckContract
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeConditionalFutureOption,
    RuntimeJointActionCandidate,
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    RuntimeJointResolvedChoice,
)
from lh_platform.test_time_adaptation.runtime_owned_option_dp import (
    RuntimeAtomicActionEstimate,
    RuntimeAtomicOutcomeEstimate,
    RuntimeOwnedOptionDPResolver,
    RuntimeOwnedOptionProjection,
)
from lh_platform.test_time_adaptation.runtime_planning_horizon import (
    SUPPORTED_PLANNING_HORIZONS,
    effective_planning_horizon,
)
from lh_platform.test_time_adaptation.runtime_public_prediction import (
    compile_residual_acceptance_state,
    flatten_residual_acceptance_state,
)

RUNTIME_ATOMIC_ESTIMATE_PROVIDER_ID = (
    "locally_designed_frozen_atomic_estimate_provider_v1"
)
RUNTIME_ATOMIC_ESTIMATE_IMPLEMENTATION_STATUS = "implementation_unverified"
# The legacy global identity stays the default for every frozen artifact.  Calibration is a
# property of one estimate configuration, so a calibrated
# config carries its own status and provenance instead of changing this constant.
RUNTIME_ATOMIC_ESTIMATE_CALIBRATION_STATUS = "uncalibrated_development_canary"
RUNTIME_ATOMIC_ESTIMATE_SOURCE_CALIBRATED_STATUS = "source_calibrated_v2"
_RUNTIME_ESTIMATE_CALIBRATION_METADATA_KEYS = (
    "calibration_id",
    "source_task_ids",
    "excluded_task_id",
    "smoothing",
)


@dataclass(frozen=True)
class RuntimeFrozenAtomicEstimateRule:
    """Value one exact action contract without defining a successor state."""

    rule_id: str
    action_type: RuntimeJointActionType
    contract_id: str
    target_predicate: str | None
    duration_ms: int
    outcomes: tuple[RuntimeAtomicOutcomeEstimate, ...]
    provenance_id: str
    provenance_detail: str

    def __post_init__(self) -> None:
        if (
            not self.rule_id
            or not self.contract_id
            or self.target_predicate == ""
            or isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 1
            or not self.provenance_id
            or not self.provenance_detail.strip()
        ):
            raise ValueError("runtime frozen atomic estimate rule is incomplete")
        RuntimeAtomicActionEstimate(
            instance_action_id="configuration-validation",
            duration_ms=self.duration_ms,
            provenance_id=self.provenance_id,
            provenance_detail=self.provenance_detail,
            outcomes=self.outcomes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "action_type": self.action_type.value,
            "contract_id": self.contract_id,
            "target_predicate": self.target_predicate,
            "duration_ms": self.duration_ms,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "provenance_id": self.provenance_id,
            "provenance_detail": self.provenance_detail,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> RuntimeFrozenAtomicEstimateRule:
        expected = {
            "rule_id",
            "action_type",
            "contract_id",
            "target_predicate",
            "duration_ms",
            "outcomes",
            "provenance_id",
            "provenance_detail",
        }
        outcomes = payload.get("outcomes")
        if (
            set(payload) != expected
            or not isinstance(outcomes, list)
            or any(not isinstance(item, Mapping) for item in outcomes)
            or isinstance(payload.get("duration_ms"), bool)
            or not isinstance(payload.get("duration_ms"), int)
        ):
            raise ValueError("runtime frozen atomic estimate rule payload is invalid")
        return cls(
            rule_id=str(payload["rule_id"]),
            action_type=RuntimeJointActionType(str(payload["action_type"])),
            contract_id=str(payload["contract_id"]),
            target_predicate=(
                str(payload["target_predicate"])
                if payload["target_predicate"] is not None
                else None
            ),
            duration_ms=payload["duration_ms"],
            outcomes=tuple(
                RuntimeAtomicOutcomeEstimate(
                    outcome_id=str(item["outcome_id"]),
                    probability=float(item["probability"]),
                )
                for item in outcomes
            ),
            provenance_id=str(payload["provenance_id"]),
            provenance_detail=str(payload["provenance_detail"]),
        )


@dataclass(frozen=True)
class RuntimeFrozenAtomicEstimateConfiguration:
    """Freeze the zero-model rules used by one canary cell.

    A calibrated configuration is the same artifact with its own
    `calibration_status = "source_calibrated_v2"` and a provenance block
    (`calibration_id`, `source_task_ids`, `excluded_task_id`, `smoothing`); legacy payloads
    keep the global uncalibrated identity and stay byte-identical on load.
    """

    configuration_id: str
    rules: tuple[RuntimeFrozenAtomicEstimateRule, ...]
    calibration_status: str = RUNTIME_ATOMIC_ESTIMATE_CALIBRATION_STATUS
    calibration_id: str | None = None
    source_task_ids: tuple[str, ...] | None = None
    excluded_task_id: str | None = None
    smoothing: Mapping[str, object] | None = None

    schema_version = "runtime_frozen_atomic_estimate_configuration_v1"

    def __post_init__(self) -> None:
        rule_ids = [item.rule_id for item in self.rules]
        if (
            not self.configuration_id
            or not self.rules
            or len(rule_ids) != len(set(rule_ids))
        ):
            raise ValueError("runtime frozen atomic estimate configuration is invalid")
        if self.calibration_status == RUNTIME_ATOMIC_ESTIMATE_CALIBRATION_STATUS:
            if (
                self.calibration_id is not None
                or self.source_task_ids is not None
                or self.excluded_task_id is not None
                or self.smoothing is not None
            ):
                raise ValueError(
                    "uncalibrated estimate configuration cannot carry calibration metadata"
                )
            return
        if self.calibration_status != (
            RUNTIME_ATOMIC_ESTIMATE_SOURCE_CALIBRATED_STATUS
        ):
            raise ValueError("estimate calibration status is unsupported")
        if (
            not isinstance(self.calibration_id, str)
            or not self.calibration_id.strip()
            or not self.source_task_ids
            or len(set(self.source_task_ids)) != len(self.source_task_ids)
            or any(
                not isinstance(task_id, str) or not task_id
                for task_id in self.source_task_ids
            )
            or (
                self.excluded_task_id is not None
                and not isinstance(self.excluded_task_id, str)
            )
            or not isinstance(self.smoothing, Mapping)
            or not self.smoothing
        ):
            raise ValueError("calibrated estimate configuration provenance is incomplete")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "provider_id": RUNTIME_ATOMIC_ESTIMATE_PROVIDER_ID,
            "implementation_status": RUNTIME_ATOMIC_ESTIMATE_IMPLEMENTATION_STATUS,
            "calibration_status": self.calibration_status,
            "configuration_id": self.configuration_id,
            "rules": [item.to_dict() for item in self.rules],
        }
        if self.calibration_status != RUNTIME_ATOMIC_ESTIMATE_CALIBRATION_STATUS:
            payload["calibration_id"] = self.calibration_id
            payload["source_task_ids"] = list(self.source_task_ids or ())
            payload["excluded_task_id"] = self.excluded_task_id
            payload["smoothing"] = dict(self.smoothing or {})
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> RuntimeFrozenAtomicEstimateConfiguration:
        base = {
            "schema_version",
            "provider_id",
            "implementation_status",
            "calibration_status",
            "configuration_id",
            "rules",
        }
        calibrated = base | set(_RUNTIME_ESTIMATE_CALIBRATION_METADATA_KEYS)
        rules = payload.get("rules")
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("provider_id") != RUNTIME_ATOMIC_ESTIMATE_PROVIDER_ID
            or payload.get("implementation_status")
            != RUNTIME_ATOMIC_ESTIMATE_IMPLEMENTATION_STATUS
            or not isinstance(rules, list)
            or any(not isinstance(item, Mapping) for item in rules)
        ):
            raise ValueError("runtime frozen atomic estimate configuration differs")
        keys = set(payload)
        status = payload.get("calibration_status")
        if keys == base:
            if status != RUNTIME_ATOMIC_ESTIMATE_CALIBRATION_STATUS:
                raise ValueError("runtime frozen atomic estimate configuration differs")
            configuration = cls(
                configuration_id=str(payload["configuration_id"]),
                rules=tuple(
                    RuntimeFrozenAtomicEstimateRule.from_dict(item) for item in rules
                ),
            )
        elif keys == calibrated:
            if status != RUNTIME_ATOMIC_ESTIMATE_SOURCE_CALIBRATED_STATUS:
                raise ValueError("runtime frozen atomic estimate configuration differs")
            source_task_ids = payload.get("source_task_ids")
            smoothing = payload.get("smoothing")
            if (
                not isinstance(source_task_ids, list)
                or not isinstance(smoothing, Mapping)
            ):
                raise ValueError("runtime frozen atomic estimate configuration differs")
            configuration = cls(
                configuration_id=str(payload["configuration_id"]),
                rules=tuple(
                    RuntimeFrozenAtomicEstimateRule.from_dict(item) for item in rules
                ),
                calibration_status=str(status),
                calibration_id=str(payload["calibration_id"]),
                source_task_ids=tuple(str(item) for item in source_task_ids),
                excluded_task_id=(
                    str(payload["excluded_task_id"])
                    if payload["excluded_task_id"] is not None
                    else None
                ),
                smoothing=dict(smoothing),
            )
        else:
            raise ValueError("runtime frozen atomic estimate configuration differs")
        if configuration.to_dict() != dict(payload):
            raise ValueError("runtime frozen atomic estimate configuration changed on load")
        return configuration


_PlanningAction = RuntimeJointActionCandidate | RuntimeConditionalFutureOption


def _action_contract_identity(
    action: _PlanningAction,
    checks: tuple[RuntimeCheckContract, ...],
) -> tuple[str, str | None]:
    """Resolve a stable configured contract identity from an instantiated action."""

    contract = action.execution_contract
    if action.action_type is RuntimeJointActionType.CONTINUE_ACTOR:
        return RuntimeJointActionType.CONTINUE_ACTOR.value, None
    if action.action_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE:
        # Submitting is a terminal action whose value does not depend on which
        # output obligation it names, exactly like continuation.  Keeping the
        # obligation text out of the identity is what makes the action priceable by
        # a frozen configuration, because the obligation id only exists at runtime.
        return RuntimeJointActionType.SUBMIT_CURRENT_STATE.value, None
    if contract is None:
        raise ValueError("runtime-owned option action lacks an execution contract")
    operation_arguments = contract.proposed_check_or_repair.get("arguments")
    if not isinstance(operation_arguments, Mapping):
        raise ValueError("runtime-owned option action arguments are invalid")
    trial_protocol = operation_arguments.get("candidate_trial_protocol")
    if trial_protocol is not None and (
        not isinstance(trial_protocol, str) or not trial_protocol
    ):
        raise ValueError("runtime-owned option candidate trial protocol is invalid")
    if isinstance(action, RuntimeConditionalFutureOption):
        contract_id = action.repair_contract_id
        if trial_protocol is not None:
            contract_id = f"{contract_id}|{trial_protocol}"
        return contract_id, contract.target_predicate
    arguments = contract.proposed_check_or_repair["arguments"]
    if action.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR:
        repair_id = arguments.get("repair_id")
        if not isinstance(repair_id, str) or not repair_id:
            raise ValueError("targeted repair estimate lacks its configured identity")
        contract_id = repair_id
        if trial_protocol is not None:
            contract_id = f"{contract_id}|{trial_protocol}"
        return contract_id, contract.target_predicate
    if action.action_type is RuntimeJointActionType.RUN_REGISTERED_CHECK:
        matches = [
            check
            for check in checks
            if check.task_id == action.target_binding.task_id
            and check.proposition == contract.target_predicate
            and check.resource_ids == contract.scope[1:]
            and dict(check.arguments) == dict(arguments)
        ]
        if len(matches) != 1 or matches[0].check_id is None:
            raise ValueError("registered check estimate lacks one configured identity")
        return matches[0].check_id, contract.target_predicate
    raise ValueError("runtime-owned option action is outside the atomic estimate contract")


RESIDUAL_STRUCTURE_TASK_LINKED = "task_linked"
RESIDUAL_STRUCTURE_FLAT = "flat"
SUPPORTED_RESIDUAL_STRUCTURES = (
    RESIDUAL_STRUCTURE_TASK_LINKED,
    RESIDUAL_STRUCTURE_FLAT,
)

# Sentinel so `resolve(estimate_adapter=None)` can explicitly mean "no adapter" while the
# default keeps the constructor-installed adapter (JIT v2 uses both passes for its receipt).
_DEFAULT_ESTIMATE_ADAPTER = object()


class RuntimeOwnedOptionLiveResolver:
    """Project live Kernel state and apply frozen estimates at the selector seam."""

    receipt_schema_version = "runtime_atomic_estimate_receipt_v1"
    supported_planning_horizons = SUPPORTED_PLANNING_HORIZONS

    def __init__(
        self,
        *,
        configuration: RuntimeFrozenAtomicEstimateConfiguration,
        configured_checks: tuple[RuntimeCheckContract, ...],
        public_projection: Callable[[], Mapping[str, object]] | None = None,
        planning_horizon_cap: int | None = None,
        residual_structure: str = RESIDUAL_STRUCTURE_TASK_LINKED,
        estimate_adapter: object | None = None,
    ) -> None:
        if (
            planning_horizon_cap is not None
            and planning_horizon_cap not in self.supported_planning_horizons
        ):
            raise ValueError("runtime planning horizon cap is unsupported")
        if residual_structure not in SUPPORTED_RESIDUAL_STRUCTURES:
            raise ValueError("runtime residual structure mode is unsupported")
        self.configuration = configuration
        self.configured_checks = configured_checks
        self.public_projection = public_projection
        self.planning_horizon_cap = planning_horizon_cap
        self.residual_structure = residual_structure
        # Optional JIT v2 seam: called per action after the frozen rule is bound, and its
        # returned record is persisted in the estimate receipt for audit.
        self.estimate_adapter = estimate_adapter
        self._resolver = RuntimeOwnedOptionDPResolver()

    def _native_horizon(self, trigger_tool_name: str) -> int:
        """Preserve the original window while allowing separately identified stage models."""
        return 1 if trigger_tool_name == "runtime_option_replan_trigger" else 2

    def _effective_horizon(self, native: int) -> int:
        """Apply the original cap contract for the legacy resolver."""
        return effective_planning_horizon(native, self.planning_horizon_cap)

    def _estimates(
        self,
        slate: RuntimeJointActionSlate,
        projection: RuntimeOwnedOptionProjection,
        estimate_adapter: object | None,
    ) -> tuple[
        tuple[RuntimeAtomicActionEstimate, ...],
        tuple[dict[str, str], ...],
        list[dict[str, object]],
    ]:
        """Instantiate exact receipts only after Runtime freezes the action universe."""

        universe: tuple[_PlanningAction, ...] = (
            *slate.candidates,
            *slate.conditional_future_options,
        )
        estimates = []
        matches = []
        adaptations: list[dict[str, object]] = []
        for action in universe:
            contract_id, target_predicate = _action_contract_identity(
                action, self.configured_checks
            )
            rules = [
                rule
                for rule in self.configuration.rules
                if rule.action_type is action.action_type
                and rule.contract_id == contract_id
                and rule.target_predicate == target_predicate
            ]
            if len(rules) != 1:
                raise ValueError(
                    "runtime action lacks one exact frozen atomic estimate rule"
                )
            rule = rules[0]
            estimate = RuntimeAtomicActionEstimate(
                instance_action_id=action.instance_action_id,
                duration_ms=rule.duration_ms,
                provenance_id=rule.provenance_id,
                provenance_detail=rule.provenance_detail,
                outcomes=rule.outcomes,
            )
            if estimate_adapter is not None:
                estimate, adaptation = estimate_adapter(
                    slate, projection, action, rule, estimate
                )
                adaptations.append(dict(adaptation))
            estimates.append(estimate)
            matches.append(
                {
                    "instance_action_id": action.instance_action_id,
                    "rule_id": rule.rule_id,
                    "contract_id": contract_id,
                }
            )
        return tuple(estimates), tuple(matches), adaptations

    def resolve(
        self,
        slate: RuntimeJointActionSlate,
        *,
        estimate_adapter: object = _DEFAULT_ESTIMATE_ADAPTER,
    ) -> RuntimeJointResolvedChoice:
        """Run the real Bellman resolver from current public state and exact estimates."""

        if self.public_projection is None:
            raise RuntimeError("live runtime-owned option resolver lacks Kernel state")
        public_projection = self.public_projection()
        configured = tuple(
            {
                "task_id": item.task_id,
                "proposition": item.proposition,
                "resource_ids": list(item.resource_ids),
                **(
                    {"check_id": item.check_id, "predicate_id": item.predicate_id}
                    if item.check_id is not None
                    else {}
                ),
            }
            for item in self.configured_checks
        )
        residual = compile_residual_acceptance_state(
            public_projection,
            configured_check_contracts=configured,
        )
        if self.residual_structure == RESIDUAL_STRUCTURE_FLAT:
            # Full - S: same predicate facts, no task hierarchy / incident linkage.
            residual = flatten_residual_acceptance_state(residual)
        projection = RuntimeOwnedOptionProjection.from_public_projection(
            public_projection,
            residual_acceptance_state=tuple(residual),
        )
        adapter = (
            self.estimate_adapter
            if estimate_adapter is _DEFAULT_ESTIMATE_ADAPTER
            else estimate_adapter
        )
        estimates, matches, adaptations = self._estimates(slate, projection, adapter)
        actions = public_projection.get("actions")
        if not isinstance(actions, list):
            raise ValueError("runtime atomic estimate projection lacks actions")
        trigger_rows = [
            item
            for item in actions
            if isinstance(item, Mapping)
            and item.get("action_id") == slate.trigger_action_id
        ]
        if len(trigger_rows) != 1:
            raise ValueError("runtime atomic estimate trigger is absent from Kernel state")
        native_horizon = self._native_horizon(str(trigger_rows[0].get("tool_name")))
        # The cap limits only how far each planning call expands; the closed loop
        # still acts at every later boundary (Full - D configures cap=1).
        horizon = self._effective_horizon(native_horizon)
        choice = self._select(slate, projection, estimates, horizon)
        receipt = {
            "schema_version": self.receipt_schema_version,
            "configuration": self.configuration.to_dict(),
            "slate": slate.to_dict(),
            "projection": projection.to_dict(),
            "planning_horizon_cap": self.planning_horizon_cap,
            "residual_structure": self.residual_structure,
            "native_planning_horizon": native_horizon,
            "rule_matches": list(matches),
            "atomic_estimates": [item.to_dict() for item in estimates],
        }
        if adaptations:
            # Only present when a JIT v2 adapter ran, so every legacy receipt keeps its
            # exact key set and replay path.
            receipt["transition_adaptations"] = list(adaptations)
        return replace(
            choice,
            planning_model={
                **dict(choice.planning_model or {}),
                "atomic_estimate_receipt": receipt,
            },
        )

    def resolve_recorded(
        self,
        slate: RuntimeJointActionSlate,
        model: Mapping[str, object],
    ) -> RuntimeJointResolvedChoice:
        """Replay only a model whose exact estimate receipt matches its inputs."""

        receipt = model.get("atomic_estimate_receipt")
        # The frozen configuration carries the planning horizon and residual structure,
        # and `resolve` records them with the receipt; replay must accept the fields the
        # record path actually writes, or every resumed option-DP decision aborts.
        legacy_keys = {
            "schema_version",
            "configuration",
            "slate",
            "projection",
            "planning_horizon_cap",
            "residual_structure",
            "native_planning_horizon",
            "rule_matches",
            "atomic_estimates",
        }
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) not in (legacy_keys, legacy_keys | {"transition_adaptations"})
            or receipt.get("schema_version") != self.receipt_schema_version
            or receipt.get("configuration") != self.configuration.to_dict()
            or receipt.get("slate") != slate.to_dict()
            or receipt.get("projection") != model.get("projection")
            or receipt.get("atomic_estimates") != model.get("atomic_estimates")
        ):
            raise ValueError("runtime atomic estimate receipt differs")
        projection_payload = receipt.get("projection")
        if not isinstance(projection_payload, Mapping):
            raise ValueError("runtime atomic estimate receipt projection is missing")
        projection = RuntimeOwnedOptionProjection.from_dict(projection_payload)
        estimates, matches, adaptations = self._estimates(
            slate, projection, self.estimate_adapter
        )
        if (
            receipt.get("rule_matches") != list(matches)
            or receipt.get("atomic_estimates")
            != [item.to_dict() for item in estimates]
        ):
            raise ValueError("runtime atomic estimate rule binding differs")
        recorded_adaptations = receipt.get("transition_adaptations")
        if (recorded_adaptations is not None) != bool(adaptations) or (
            recorded_adaptations is not None
            and list(recorded_adaptations) != list(adaptations)
        ):
            raise ValueError("runtime atomic estimate transition adaptations differ")
        choice = self._select_recorded(slate, model, estimates)
        return replace(choice, planning_model=dict(model))

    def _select(
        self,
        slate: RuntimeJointActionSlate,
        projection: RuntimeOwnedOptionProjection,
        estimates: tuple[RuntimeAtomicActionEstimate, ...],
        horizon: int,
    ) -> RuntimeJointResolvedChoice:
        """Choose from bound estimates; selectors other than Bellman override this."""

        return self._resolver.resolve(slate, projection, estimates, horizon=horizon)

    def _select_recorded(
        self,
        slate: RuntimeJointActionSlate,
        model: Mapping[str, object],
        estimates: tuple[RuntimeAtomicActionEstimate, ...],
    ) -> RuntimeJointResolvedChoice:
        """Replay a recorded choice after its estimate receipt has been verified."""

        del estimates
        return self._resolver.resolve_recorded(slate, model)


__all__ = [
    "RUNTIME_ATOMIC_ESTIMATE_CALIBRATION_STATUS",
    "RUNTIME_ATOMIC_ESTIMATE_IMPLEMENTATION_STATUS",
    "RUNTIME_ATOMIC_ESTIMATE_PROVIDER_ID",
    "RuntimeFrozenAtomicEstimateConfiguration",
    "RuntimeFrozenAtomicEstimateRule",
    "RuntimeOwnedOptionLiveResolver",
]
