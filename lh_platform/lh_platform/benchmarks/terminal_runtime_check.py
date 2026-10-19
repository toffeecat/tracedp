"""Admit explicit check outcomes to the existing runtime evidence ledger."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    EvidenceState,
    PortableRuntimeKernel,
)

EXACT_OUTPUT_VERDICT_PROTOCOL = "exact_output_v1"
EXIT_STATUS_VERDICT_PROTOCOL = "exit_status_v1"
RUNTIME_CHECK_VERDICT_PROTOCOLS = frozenset(
    {EXACT_OUTPUT_VERDICT_PROTOCOL, EXIT_STATUS_VERDICT_PROTOCOL}
)
ACTOR_GUIDANCE_REPAIR_FORM = "actor_guidance"
ACTOR_GUIDANCE_ALLOWED_TOOLS = "bash,read,write"
ACTOR_GUIDANCE_REPAIR_ARGUMENT_KEYS = frozenset(
    {"candidate_form", "guidance_text", "fixer_identity", "allowed_tools"}
)
BASH_TARGETED_REPAIR_OPERATION = "execute_targeted_repair_then_registered_check"
ACTOR_GUIDANCE_REPAIR_OPERATION = "isolate_actor_guidance_then_deliver_if_admitted"

# Check eligibility semantics (v3).  A check verifies a SUBJECT resource -- the deliverable
# its proposition is about -- and may need PREREQUISITE resources only so the checker can
# run at all.  A missing subject is a failed predicate (refuted); a missing prerequisite is
# an unusable checker (unknown), never a failed predicate.  Legacy v1/v2 payloads keep the
# old rule (every listed resource must exist) so frozen artifacts keep their exact reachable
# action set.
LEGACY_RESOURCE_PRESENCE_POLICY = "legacy_all_resources_must_exist"
SUBJECT_REFUTABLE_PRESENCE_POLICY = "subject_refutable_prerequisite_required"
RUNTIME_CHECK_PRESENCE_POLICIES = frozenset(
    {LEGACY_RESOURCE_PRESENCE_POLICY, SUBJECT_REFUTABLE_PRESENCE_POLICY}
)
# Predicate kinds are frozen in the spec BEFORE any calibration run observes outcomes; they
# are the cross-task calibration key and must never be assigned after seeing verdicts.
RUNTIME_CHECK_PREDICATE_KINDS = frozenset(
    {"shape", "existence", "behavioral_discriminator", "consistency"}
)
# Exit status reserved for "the checker could not run"; the exit-status verdict protocol maps
# any status other than 0/1 to unknown, so this never refutes the task predicate.
CHECKER_PREREQUISITE_MISSING_EXIT_CODE = 70
CHECKER_PREREQUISITE_MISSING_MARKER = "runtime_checker_prerequisite_missing"
BASH_TARGETED_REPAIR_TRANSITION = "configured_repair_then_registered_recheck_v1"
ACTOR_GUIDANCE_REPAIR_TRANSITION = "configured_isolated_guidance_then_live_delivery_v1"

if TYPE_CHECKING:
    from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
        RuntimeConditionalFutureOption,
        RuntimeJointActionCandidate,
    )


@dataclass(frozen=True)
class RuntimeCheckContract:
    """Declare one reviewed check and its explicit result interpretation."""

    task_id: str
    proposition: str
    arguments: Mapping[str, Any]
    supported_output: str
    refuted_output: str
    resource_ids: tuple[str, ...]
    verdict_protocol: str = EXACT_OUTPUT_VERDICT_PROTOCOL
    check_id: str | None = None
    predicate_id: str | None = None
    # Defaults reproduce the legacy presence semantics exactly, so every existing construction
    # site (audit rebuilds, actor registration, v1/v2 payload loads) behaves as before.
    presence_policy: str = LEGACY_RESOURCE_PRESENCE_POLICY
    prerequisite_resource_ids: tuple[str, ...] = ()
    predicate_kind: str | None = None
    read_only: bool = False

    def __post_init__(self) -> None:
        if self.verdict_protocol not in RUNTIME_CHECK_VERDICT_PROTOCOLS:
            raise ValueError("runtime check verdict protocol is unsupported")
        if self.supported_output == self.refuted_output:
            raise ValueError("check outcomes must be distinguishable")
        if (self.check_id is None) != (self.predicate_id is None):
            raise ValueError("check and predicate identities must be bound together")
        if self.check_id is not None and (not self.check_id or not self.predicate_id):
            raise ValueError("check and predicate identities must be nonempty")
        if self.presence_policy not in RUNTIME_CHECK_PRESENCE_POLICIES:
            raise ValueError("runtime check presence policy is unsupported")
        # JSON payloads deliver these as lists; normalize once so comparisons and scope
        # construction stay tuple-based like resource_ids.
        if not isinstance(self.prerequisite_resource_ids, tuple):
            object.__setattr__(
                self, "prerequisite_resource_ids", tuple(self.prerequisite_resource_ids)
            )
        if any(
            not isinstance(resource_id, str) or not resource_id
            for resource_id in self.prerequisite_resource_ids
        ):
            raise ValueError("runtime check prerequisites must be nonempty resource ids")
        if set(self.prerequisite_resource_ids) & set(self.resource_ids):
            raise ValueError("a subject resource cannot also be a checker prerequisite")
        if (
            self.presence_policy == SUBJECT_REFUTABLE_PRESENCE_POLICY
            and not self.resource_ids
        ):
            raise ValueError("subject-refutable checks require at least one subject resource")
        if self.predicate_kind is not None and (
            self.predicate_kind not in RUNTIME_CHECK_PREDICATE_KINDS
        ):
            raise ValueError("runtime check predicate kind is unsupported")

    def with_predicate_binding(
        self, *, check_id: str, predicate_id: str
    ) -> RuntimeCheckContract:
        """Attach durable identities after an Actor registration is committed."""

        return replace(self, check_id=check_id, predicate_id=predicate_id)


def is_actor_guidance_repair_arguments(arguments: Mapping[str, Any]) -> bool:
    """True only for the frozen Actor-guidance repair form, never inferred."""

    return arguments.get("candidate_form") == ACTOR_GUIDANCE_REPAIR_FORM


def validate_runtime_repair_arguments(arguments: Mapping[str, Any]) -> None:
    """Admit a bash command or frozen Actor guidance; never compile prose to shell."""

    if set(arguments) == {"command"}:
        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("runtime predicate repair contract is incomplete")
        return
    if set(arguments) != ACTOR_GUIDANCE_REPAIR_ARGUMENT_KEYS:
        raise ValueError("runtime predicate repair contract is incomplete")
    guidance = arguments.get("guidance_text")
    fixer = arguments.get("fixer_identity")
    if (
        arguments.get("candidate_form") != ACTOR_GUIDANCE_REPAIR_FORM
        or not isinstance(guidance, str)
        or not guidance.strip()
        or not isinstance(fixer, str)
        or not fixer.strip()
    ):
        raise ValueError("runtime predicate repair contract is incomplete")
    if arguments.get("allowed_tools") != ACTOR_GUIDANCE_ALLOWED_TOOLS:
        raise ValueError("isolated actor guidance may only use the shared Terminal Actor tools")


def targeted_repair_execution_shape(
    repair: RuntimePredicateRepairContract,
) -> tuple[str, bool, str, str]:
    """Name the live operation without compiling a task-specific payload."""

    if is_actor_guidance_repair_arguments(repair.arguments):
        return (
            ACTOR_GUIDANCE_REPAIR_OPERATION,
            True,
            ACTOR_GUIDANCE_REPAIR_TRANSITION,
            (
                "isolated registered checks support the predicate; live Actor "
                "receives frozen guidance and must recheck"
            ),
        )
    return (
        BASH_TARGETED_REPAIR_OPERATION,
        False,
        BASH_TARGETED_REPAIR_TRANSITION,
        f"registered check observes predicate supported: {repair.proposition}",
    )


@dataclass(frozen=True)
class RuntimePredicateRepairContract:
    """Declare one executable repair for a refuted registered predicate."""

    repair_id: str
    task_id: str
    proposition: str
    arguments: Mapping[str, Any]
    resource_ids: tuple[str, ...]
    time_limit_seconds: int

    def __post_init__(self) -> None:
        if (
            not self.repair_id
            or ":" in self.repair_id
            or not self.task_id
            or not self.proposition
            or not self.resource_ids
            or isinstance(self.time_limit_seconds, bool)
            or not isinstance(self.time_limit_seconds, int)
            or self.time_limit_seconds < 1
        ):
            raise ValueError("runtime predicate repair contract is incomplete")
        validate_runtime_repair_arguments(self.arguments)


def interpret_runtime_check_result(
    contract: RuntimeCheckContract,
    *,
    content: str,
    return_code: object,
    is_error: object,
) -> EvidenceState:
    """Interpret only declared shell semantics; transport failures remain unknown."""

    if isinstance(return_code, bool) or not isinstance(return_code, int):
        return EvidenceState.UNKNOWN
    if contract.verdict_protocol == EXIT_STATUS_VERDICT_PROTOCOL:
        if return_code == 0:
            return EvidenceState.SUPPORTED
        if return_code == 1:
            return EvidenceState.REFUTED
        return EvidenceState.UNKNOWN
    if is_error is False and return_code == 0:
        if content == contract.supported_output:
            return EvidenceState.SUPPORTED
        if content == contract.refuted_output:
            return EvidenceState.REFUTED
    return EvidenceState.UNKNOWN


def runtime_check_blocking_fact(
    state: EvidenceState, *, content: str, return_code: object
) -> dict[str, object] | None:
    """Preserve the observed failure signal without turning diagnostics into truth."""

    if state is EvidenceState.SUPPORTED:
        return None
    return {
        "return_code": (
            return_code
            if isinstance(return_code, int) and not isinstance(return_code, bool)
            else None
        ),
        "output": content,
    }


def _runtime_check_time_limit_ms(contract: RuntimeCheckContract) -> int:
    """Convert the reviewed whole-second limit without accepting boolean values."""

    seconds = contract.arguments.get("timeout", 120)
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 1:
        raise ValueError("runtime check timeout must be a positive integer")
    return seconds * 1000


def conditional_runtime_repair_option_id(
    repair: RuntimePredicateRepairContract,
    *,
    check_index: int,
    versions: tuple[tuple[str, int], ...],
) -> str:
    """Name one repair contract while live candidates retain exact state versions."""

    del versions
    return (
        f"conditional-targeted-repair:{repair.repair_id}:check-{check_index}"
    )


def conditional_runtime_repair_options(
    kernel: PortableRuntimeKernel,
    repair_contracts: tuple[RuntimePredicateRepairContract, ...],
    check_contracts: tuple[RuntimeCheckContract, ...],
    registered_check_candidates: tuple[RuntimeJointActionCandidate, ...],
    *,
    budget_before: int,
    actor_time_remaining_ms: int | None,
) -> tuple[RuntimeConditionalFutureOption, ...]:
    """Compile configured repairs as non-executable successors to pending checks."""

    from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
        RuntimeActionExecutionContract,
        RuntimeActionTargetBinding,
        RuntimeConditionalFutureOption,
        RuntimeJointActionType,
    )

    options = []
    for check_candidate in registered_check_candidates:
        contract = check_candidate.execution_contract
        if (
            check_candidate.action_type is not RuntimeJointActionType.RUN_REGISTERED_CHECK
            or contract is None
        ):
            continue
        matching_checks = [
            (index, check)
            for index, check in enumerate(check_contracts)
            if check.task_id == check_candidate.target_binding.task_id
            and check.proposition == contract.target_predicate
            and check.resource_ids == contract.scope[1:]
            and dict(check.arguments)
            == dict(contract.proposed_check_or_repair["arguments"])
        ]
        if len(matching_checks) != 1:
            continue
        check_index, check = matching_checks[0]
        versions = tuple(
            (resource_id, kernel.resource_versions[resource_id])
            for resource_id in check.resource_ids
        )
        for repair in repair_contracts:
            if (
                repair.task_id != check.task_id
                or repair.proposition != check.proposition
                or repair.resource_ids != check.resource_ids
            ):
                continue
            future_duration_ms = (
                repair.time_limit_seconds * 1000
                + _runtime_check_time_limit_ms(check)
            )
            if (
                check_candidate.cost_units + 1 > budget_before
                or (
                    actor_time_remaining_ms is not None
                    and contract.time_limit_ms + future_duration_ms
                    > actor_time_remaining_ms
                )
            ):
                continue
            option_id = conditional_runtime_repair_option_id(
                repair, check_index=check_index, versions=versions
            )
            primary_resource_id, primary_version = versions[0]
            operation, delivery_only, transition_model_id, success_postcondition = (
                targeted_repair_execution_shape(repair)
            )
            options.append(
                RuntimeConditionalFutureOption(
                    option_id=option_id,
                    action_type=RuntimeJointActionType.RUN_TARGETED_REPAIR,
                    target_binding=RuntimeActionTargetBinding(
                        primary_resource_id,
                        primary_version,
                        task_id=repair.task_id,
                    ),
                    affected_task_ids=(repair.task_id,),
                    cost_units=1,
                    cost_bytes=0,
                    transition_model_id=transition_model_id,
                    prerequisite_action_id=check_candidate.instance_action_id,
                    prerequisite_outcomes=("refuted", "unknown"),
                    repair_contract_id=repair.repair_id,
                    check_contract_id=check.check_id or f"configured-check-{check_index}",
                    resource_versions=tuple(sorted(versions)),
                    execution_contract=RuntimeActionExecutionContract(
                        target_predicate=repair.proposition,
                        supporting_evidence=(
                            f"frozen-repair-contract:{repair.repair_id}",
                            f"frozen-check-contract:{check.check_id or check_index}",
                        ),
                        scope=(repair.task_id, *repair.resource_ids),
                        proposed_check_or_repair={
                            "operation": operation,
                            "arguments": {
                                "repair_id": repair.repair_id,
                                "repair_arguments": dict(repair.arguments),
                                "postcondition_check_arguments": dict(check.arguments),
                                "check_index": check_index,
                            },
                        },
                        time_limit_ms=future_duration_ms,
                        success_postcondition=success_postcondition,
                        failure_branch=(
                            "record refuted or unknown postcondition evidence and keep "
                            "the incident open"
                        ),
                        delivery_only=delivery_only,
                    ),
                )
            )
    return tuple(sorted(options, key=lambda item: item.option_id))


def check_prerequisites_available(
    kernel: PortableRuntimeKernel, contract: RuntimeCheckContract
) -> bool:
    """True only when every checker prerequisite resource is observed and physically present.

    Subject resources are deliberately NOT tested here: a missing subject is the predicate's
    own failure and must stay executable so the verdict is `refuted`.
    """

    return all(
        resource_id in kernel.observations
        and kernel.observations[resource_id].observed
        and kernel.observations[resource_id].exists is True
        for resource_id in contract.prerequisite_resource_ids
    )


def pending_runtime_check_candidates(
    kernel: PortableRuntimeKernel,
    contracts: tuple[RuntimeCheckContract, ...],
    *,
    excluded_versions: frozenset[tuple[int, tuple[tuple[str, int], ...]]] = frozenset(),
) -> tuple[RuntimeJointActionCandidate, ...]:
    """Compile version-bound checks that still lack a current executed verdict."""

    # Import here to keep the evidence model independent from the joint policy module.
    from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
        RuntimeActionExecutionContract,
        RuntimeActionTargetBinding,
        RuntimeJointActionCandidate,
        RuntimeJointActionType,
    )

    candidates = []
    for index, contract in enumerate(contracts):
        # Legacy static contracts can be unscoped evidence readers, but they
        # cannot become version-bound runtime actions.
        if not contract.resource_ids:
            continue
        # Subject and prerequisite resources must both be registered so the verdict can bind
        # to subject versions; only the legacy policy additionally requires the subject to
        # exist before a check may run.
        required_resource_ids = (
            *contract.resource_ids,
            *contract.prerequisite_resource_ids,
        )
        if any(
            resource_id not in kernel.resource_versions
            or resource_id not in kernel.observations
            for resource_id in required_resource_ids
        ):
            continue
        versions = tuple(
            (resource_id, kernel.resource_versions[resource_id])
            for resource_id in contract.resource_ids
        )
        if (index, versions) in excluded_versions:
            continue
        if contract.presence_policy == LEGACY_RESOURCE_PRESENCE_POLICY and any(
            not kernel.observations[resource_id].observed
            or kernel.observations[resource_id].exists is not True
            for resource_id in contract.resource_ids
        ):
            continue
        if any(
            (
                contract.predicate_id is not None
                and evidence.predicate_id == contract.predicate_id
                or contract.predicate_id is None
                and evidence.task_id == contract.task_id
                and evidence.proposition == contract.proposition
            )
            and evidence.resource_versions == versions
            and (
                evidence.state in {EvidenceState.SUPPORTED, EvidenceState.REFUTED}
                if contract.predicate_id is not None
                else evidence.state is not EvidenceState.STALE
            )
            for evidence in kernel.evidence.values()
        ):
            continue
        primary_resource_id, primary_version = versions[0]
        candidates.append(
            RuntimeJointActionCandidate(
                instance_action_id=(
                    f"run-registered-check:{index}:"
                    + ",".join(str(version) for _, version in versions)
                ),
                action_type=RuntimeJointActionType.RUN_REGISTERED_CHECK,
                target_binding=RuntimeActionTargetBinding(
                    primary_resource_id,
                    primary_version,
                    task_id=contract.task_id,
                ),
                affected_task_ids=(contract.task_id,),
                cost_units=1,
                cost_bytes=0,
                transition_model_id="runtime_exec_then_versioned_evidence_v1",
                legality_evidence_id=(
                    "registered_check_pending_at_resource_versions_v1:"
                    + (
                        f"{contract.check_id}:"
                        if contract.check_id is not None
                        else ""
                    )
                    + ",".join(f"{resource_id}@{version}" for resource_id, version in versions)
                ),
                execution_contract=RuntimeActionExecutionContract(
                    target_predicate=contract.proposition,
                    supporting_evidence=(
                        "registered_check_pending_at_resource_versions_v1:"
                        + ",".join(
                            f"{resource_id}@{version}" for resource_id, version in versions
                        ),
                    ),
                    scope=(contract.task_id, *contract.resource_ids),
                    proposed_check_or_repair={
                        "operation": "execute_registered_check",
                        "arguments": dict(contract.arguments),
                    },
                    time_limit_ms=_runtime_check_time_limit_ms(contract),
                    success_postcondition=(
                        f"predicate is observed supported: {contract.proposition}"
                    ),
                    failure_branch=(
                        "record refuted or unknown evidence and open or update its incident"
                    ),
                ),
            )
        )
    return tuple(candidates)


def pending_runtime_repair_candidates(
    kernel: PortableRuntimeKernel,
    repair_contracts: tuple[RuntimePredicateRepairContract, ...],
    check_contracts: tuple[RuntimeCheckContract, ...],
    *,
    excluded_action_instance_ids: frozenset[str] = frozenset(),
) -> tuple[RuntimeJointActionCandidate, ...]:
    """Offer only configured repairs with an exact registered postcondition check."""

    from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
        RuntimeActionExecutionContract,
        RuntimeActionTargetBinding,
        RuntimeJointActionCandidate,
        RuntimeJointActionType,
    )

    candidates = []
    for repair in repair_contracts:
        incident = next(
            (
                item
                for item in kernel.predicate_incidents.values()
                if item.status in {"open", "unknown"}
                and item.task_id == repair.task_id
                and item.proposition == repair.proposition
            ),
            None,
        )
        if incident is None:
            continue
        evidence = kernel.evidence[incident.latest_evidence_id]
        versions = tuple(evidence.resource_versions)
        # Only a current, version-bound REFUTED subject predicate unlocks a targeted repair.
        # An unknown / infra / stale verdict may keep its incident for diagnosis but must
        # never motivate an intervention on the task predicate.
        if (
            evidence.state is not EvidenceState.REFUTED
            or tuple(resource_id for resource_id, _ in versions) != repair.resource_ids
            or any(
                kernel.resource_versions.get(resource_id) != version
                for resource_id, version in versions
            )
        ):
            continue
        matching_check_indices = [
            index
            for index, check in enumerate(check_contracts)
            if check.task_id == repair.task_id
            and check.proposition == repair.proposition
            and check.resource_ids == repair.resource_ids
            and (
                evidence.check_id is None
                or check.check_id == evidence.check_id
            )
        ]
        if len(matching_check_indices) != 1:
            continue
        check_index = matching_check_indices[0]
        check = check_contracts[check_index]
        primary_resource_id, primary_version = versions[0]
        instance_action_id = (
            f"run-targeted-repair:{repair.repair_id}:{incident.incident_id}:"
            + ",".join(str(version) for _, version in versions)
        )
        if instance_action_id in excluded_action_instance_ids:
            continue
        operation, delivery_only, transition_model_id, success_postcondition = (
            targeted_repair_execution_shape(repair)
        )
        candidates.append(
            RuntimeJointActionCandidate(
                instance_action_id=instance_action_id,
                action_type=RuntimeJointActionType.RUN_TARGETED_REPAIR,
                target_binding=RuntimeActionTargetBinding(
                    primary_resource_id,
                    primary_version,
                    task_id=repair.task_id,
                ),
                affected_task_ids=(repair.task_id,),
                cost_units=1,
                cost_bytes=0,
                transition_model_id=transition_model_id,
                legality_evidence_id=incident.latest_evidence_id,
                execution_contract=RuntimeActionExecutionContract(
                    target_predicate=repair.proposition,
                    supporting_evidence=(incident.latest_evidence_id,),
                    scope=(repair.task_id, *repair.resource_ids),
                    proposed_check_or_repair={
                        "operation": operation,
                        "arguments": {
                            "repair_id": repair.repair_id,
                            "repair_arguments": dict(repair.arguments),
                            "postcondition_check_arguments": dict(check.arguments),
                            "check_index": check_index,
                            "incident_id": incident.incident_id,
                        },
                    },
                    time_limit_ms=(
                        repair.time_limit_seconds * 1000
                        + _runtime_check_time_limit_ms(check)
                    ),
                    success_postcondition=success_postcondition,
                    failure_branch=(
                        "record refuted or unknown postcondition evidence and keep the incident open"
                    ),
                    delivery_only=delivery_only,
                ),
                conditional_future_option_id=conditional_runtime_repair_option_id(
                    repair, check_index=check_index, versions=versions
                ),
            )
        )
    return tuple(candidates)


def runtime_repair_configuration_payload(
    contracts: tuple[RuntimePredicateRepairContract, ...],
) -> dict[str, object]:
    """Persist the exact configured repairs without deriving task-specific commands."""

    return {
        "schema_version": "runtime_predicate_repair_configuration_v1",
        "implementation_status": "implementation_unverified",
        "contracts": [
            {
                "task_id": item.task_id,
                "repair_id": item.repair_id,
                "proposition": item.proposition,
                "arguments": dict(item.arguments),
                "resource_ids": list(item.resource_ids),
                "time_limit_seconds": item.time_limit_seconds,
            }
            for item in contracts
        ],
    }


def runtime_check_configuration_payload(
    contracts: tuple[RuntimeCheckContract, ...], *, diagnosis_enabled: bool,
    registration_enabled: bool = False,
    registration_phase_required: bool = False,
) -> dict[str, object]:
    """Encode the initial check contract shared by host and gateway configuration."""
    if registration_phase_required and not registration_enabled:
        raise ValueError("required check registration phase needs Actor registration")
    bound = [item.check_id is not None for item in contracts]
    if bound and any(bound) and not all(bound):
        raise ValueError("runtime check configuration mixes legacy and bound contracts")
    subject_refutable = [
        item.presence_policy == SUBJECT_REFUTABLE_PRESENCE_POLICY for item in contracts
    ]
    if any(subject_refutable) and not all(subject_refutable):
        raise ValueError(
            "runtime check configuration mixes legacy and subject-refutable presence policies"
        )
    if all(subject_refutable) and contracts and not all(bound):
        raise ValueError("subject-refutable checks must carry durable identities")
    if all(subject_refutable) and contracts:
        schema_version = "runtime_check_configuration_v3"
    elif bound and all(bound):
        schema_version = "runtime_check_configuration_v2"
    else:
        schema_version = "runtime_check_configuration_v1"
    return {
        "schema_version": schema_version,
        "predicate_diagnosis_enabled": diagnosis_enabled,
        **({"actor_registration_enabled": True} if registration_enabled else {}),
        **(
            {"actor_registration_phase_required": True}
            if registration_phase_required else {}
        ),
        "contracts": [
            {
                "task_id": item.task_id, "proposition": item.proposition,
                "arguments": dict(item.arguments), "supported_output": item.supported_output,
                "refuted_output": item.refuted_output, "resource_ids": list(item.resource_ids),
                **(
                    {"check_id": item.check_id, "predicate_id": item.predicate_id}
                    if item.check_id is not None
                    else {}
                ),
                **(
                    {"verdict_protocol": item.verdict_protocol}
                    if item.verdict_protocol != EXACT_OUTPUT_VERDICT_PROTOCOL
                    else {}
                ),
                **(
                    {
                        "presence_policy": item.presence_policy,
                        **(
                            {
                                "prerequisite_resource_ids": list(
                                    item.prerequisite_resource_ids
                                )
                            }
                            if item.prerequisite_resource_ids
                            else {}
                        ),
                        **(
                            {"predicate_kind": item.predicate_kind}
                            if item.predicate_kind is not None
                            else {}
                        ),
                        **({"read_only": True} if item.read_only else {}),
                    }
                    if item.presence_policy == SUBJECT_REFUTABLE_PRESENCE_POLICY
                    else {}
                ),
            } for item in contracts
        ],
    }


def bind_runtime_check_configuration(
    path: Path, contracts: tuple[RuntimeCheckContract, ...], *, diagnosis_enabled: bool,
    registration_enabled: bool = False,
    registration_phase_required: bool = False,
) -> None:
    """Freeze complete check declarations and reject semantic drift on resume."""
    payload = runtime_check_configuration_payload(
        contracts, diagnosis_enabled=diagnosis_enabled, registration_enabled=registration_enabled,
        registration_phase_required=registration_phase_required,
    )
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("runtime check configuration differs on resume")
    elif contracts or diagnosis_enabled or registration_enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_runtime_check_configuration(path: Path) -> tuple[tuple[RuntimeCheckContract, ...], bool]:
    """Restore the same declared checks, including interpreter arguments and scope."""
    payload = json.loads(path.read_text())
    if payload["schema_version"] not in {
        "runtime_check_configuration_v1",
        "runtime_check_configuration_v2",
        "runtime_check_configuration_v3",
    }:
        raise ValueError("unsupported runtime check configuration")
    contracts = tuple(
        RuntimeCheckContract(
            **{
                **item,
                "resource_ids": tuple(item["resource_ids"]),
                "prerequisite_resource_ids": tuple(
                    item.get("prerequisite_resource_ids", ())
                ),
            }
        )
        for item in payload["contracts"]
    )
    enabled = payload["predicate_diagnosis_enabled"]
    bind_runtime_check_configuration(
        path, contracts, diagnosis_enabled=enabled,
        registration_enabled=payload.get("actor_registration_enabled", False),
        registration_phase_required=payload.get(
            "actor_registration_phase_required", False
        ),
    )
    return contracts, enabled


def record_runtime_check(
    kernel: PortableRuntimeKernel,
    contract: RuntimeCheckContract,
    receipt: Mapping[str, Any],
    *,
    evidence_id: str,
) -> EvidenceState:
    """Record the latest executed check, never an install or supervisor notice."""
    action_id = receipt.get("tool_call_id")
    actors = [action for action in kernel.actions.values() if action.actor == "actor"]
    if not actors or actors[-1].action_id != action_id:
        raise ValueError("check receipt must reference the latest Actor action")
    action = actors[-1]
    if (
        action.tool_name != "bash"
        or receipt.get("tool_name") != "bash"
        or dict(action.arguments) != dict(contract.arguments)
        or receipt.get("input") != dict(contract.arguments)
        or receipt.get("environment_step_executed") is not True
    ):
        raise ValueError("executed action does not match check contract")
    result = receipt.get("tool_result")
    if not isinstance(result, Mapping) or not isinstance(result.get("content"), str):
        raise ValueError("check receipt lacks delivered content")
    boundary = receipt.get("pre_supervision_content_chars")
    if type(boundary) is not int or not 0 <= boundary <= len(result["content"]):
        raise ValueError("check receipt lacks a valid tool content boundary")
    content = result["content"][:boundary]
    details = result.get("details", {})
    if boundary != len(result["content"]):
        raise ValueError("check content contains an unverified appended suffix")
    if details.get("runtime_interventions"):
        raise ValueError("check evidence requires a boundary without intervening supervision")
    versions = details.get("runtime_state", {}).get("resource_versions", {})
    if any(versions.get(key) != kernel.resource_versions.get(key) for key in contract.resource_ids):
        raise ValueError("check evidence resource versions changed after execution")
    state = interpret_runtime_check_result(
        contract,
        content=content,
        return_code=details.get("return_code"),
        is_error=result.get("is_error"),
    )
    record_runtime_check_evidence(
        kernel,
        contract,
        evidence_id=evidence_id,
        source_action_id=str(action_id),
        state=state,
    )
    return state


def record_runtime_check_evidence(
    kernel: PortableRuntimeKernel,
    contract: RuntimeCheckContract,
    *,
    evidence_id: str,
    source_action_id: str,
    state: EvidenceState,
) -> None:
    """Bind one executed Actor or Runtime check to evidence and its incident."""

    action = kernel.actions.get(source_action_id)
    if action is None or action.phase.value not in {
        "completed",
        "failed_after_partial_effect",
        "failed_with_unknown_effect",
    }:
        raise ValueError("runtime check evidence lacks a finished source action")
    if state is EvidenceState.SUPPORTED and action.phase.value != "completed":
        raise ValueError("supported evidence requires a completed check action")
    actor_check = action.actor == "actor" and (
        action.tool_name == "bash" and dict(action.arguments) == dict(contract.arguments)
    )
    runtime_check = action.actor == "runtime" and (
        action.phase.value == "completed"
        and action.tool_name
        in {
            "runtime_registered_check_execution",
            "runtime_targeted_repair_execution",
        }
        and action.arguments.get("check_arguments") == dict(contract.arguments)
        and action.arguments.get("task_id") == contract.task_id
        and action.arguments.get("proposition") == contract.proposition
        and action.arguments.get("resource_ids") == list(contract.resource_ids)
        and (
            contract.check_id is None
            or action.arguments.get("check_id") == contract.check_id
            and action.arguments.get("predicate_id") == contract.predicate_id
        )
    )
    if not actor_check and not runtime_check:
        raise ValueError("runtime check evidence source differs from its contract")
    kernel.record_evidence(
        evidence_id=evidence_id,
        task_id=contract.task_id,
        proposition=contract.proposition,
        state=state,
        resource_ids=contract.resource_ids,
        source_action_id=source_action_id,
        predicate_id=contract.predicate_id,
        check_id=contract.check_id,
    )
    if contract.predicate_id is not None:
        # Predicate evidence and its incident transition share one durable event.
        return
    matching = [
        item for item in kernel.predicate_incidents.values()
        if (
            contract.predicate_id is not None
            and item.predicate_id == contract.predicate_id
            or contract.predicate_id is None
            and item.task_id == contract.task_id
            and item.proposition == contract.proposition
        )
        and tuple(dict(kernel.evidence[item.opening_evidence_id].resource_versions))
        == contract.resource_ids
        and (
            contract.predicate_id is not None
            or dict(
                kernel.actions[
                    kernel.evidence[item.opening_evidence_id].source_action_id
                ].arguments.get(
                    "check_arguments",
                    kernel.actions[
                        kernel.evidence[item.opening_evidence_id].source_action_id
                    ].arguments,
                )
            )
            == dict(contract.arguments)
        )
    ]
    current = matching[-1] if matching else None
    if current is not None and current.status != "resolved":
        kernel.update_predicate_incident(current.incident_id, evidence_id)
    elif state is EvidenceState.REFUTED:
        kernel.update_predicate_incident(f"incident:{evidence_id}", evidence_id)
