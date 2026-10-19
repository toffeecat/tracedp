"""Compile complete public-output obligations from one live runtime kernel."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    OutputRequirement,
    PortableRuntimeKernel,
    TaskNode,
)


class RuntimeOutputPhase(StrEnum):
    OUTPUT_UNKNOWN = "output_unknown"
    OUTPUT_ABSENT = "output_absent"
    OUTPUT_PRESENT_UNVERIFIED = "output_present_unverified"
    AWAITING_OFFICIAL_EVALUATION = "awaiting_official_evaluation"
    OFFICIALLY_FAILED = "officially_failed"
    OFFICIALLY_SUCCEEDED = "officially_succeeded"


@dataclass(frozen=True)
class RuntimeOutputObligation:
    """Bind one task to every declared output and its complete live version vector."""

    obligation_id: str
    task_id: str
    output_requirement: OutputRequirement
    output_resource_ids: tuple[str, ...]
    resource_versions: tuple[tuple[str, int], ...]
    resource_existence: tuple[tuple[str, bool | None], ...]
    affected_task_ids: tuple[str, ...]
    phase: RuntimeOutputPhase

    @property
    def version_key(self) -> tuple[
        str, str, tuple[tuple[str, int], ...], str
    ]:
        """Return the retry/effect identity without privileging one output path."""

        return (
            self.task_id,
            self.output_requirement.value,
            self.resource_versions,
            self.phase.value,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "obligation_id": self.obligation_id,
            "task_id": self.task_id,
            "output_requirement": self.output_requirement.value,
            "output_resource_ids": list(self.output_resource_ids),
            "resource_versions": [list(item) for item in self.resource_versions],
            "resource_existence": [list(item) for item in self.resource_existence],
            "affected_task_ids": list(self.affected_task_ids),
            "phase": self.phase.value,
        }


def _affected_task_ids(node: TaskNode, kernel: PortableRuntimeKernel) -> tuple[str, ...]:
    affected = []
    current: TaskNode | None = node
    while current is not None:
        affected.append(current.task_id)
        current = (
            kernel.task_nodes.get(current.parent_id)
            if current.parent_id is not None
            else None
        )
    return tuple(sorted(affected))


def compile_runtime_output_obligations(
    kernel: PortableRuntimeKernel,
) -> tuple[RuntimeOutputObligation, ...]:
    """Project live observations into order-independent, task-level obligations."""

    obligations = []
    official = kernel.official_evaluation
    for node in kernel.task_nodes.values():
        if not node.public_requirement or not node.required_output_resource_ids:
            continue
        # Canonical path order prevents declaration order from becoming a policy tie-break.
        resources = tuple(sorted(node.required_output_resource_ids))
        observations = tuple(kernel.observations.get(resource_id) for resource_id in resources)
        existence = tuple(
            (
                resource_id,
                observation.exists
                if observation is not None and observation.observed
                else None,
            )
            for resource_id, observation in zip(resources, observations, strict=True)
        )
        present_values = tuple(exists for _, exists in existence)
        if official is not None:
            phase = (
                RuntimeOutputPhase.OFFICIALLY_SUCCEEDED
                if official["succeeded"] is True
                else RuntimeOutputPhase.OFFICIALLY_FAILED
            )
        elif node.output_requirement is OutputRequirement.ANY:
            if any(value is True for value in present_values):
                phase = RuntimeOutputPhase.OUTPUT_PRESENT_UNVERIFIED
            elif all(value is False for value in present_values):
                phase = RuntimeOutputPhase.OUTPUT_ABSENT
            else:
                phase = RuntimeOutputPhase.OUTPUT_UNKNOWN
        else:
            if all(value is True for value in present_values):
                phase = RuntimeOutputPhase.OUTPUT_PRESENT_UNVERIFIED
            elif any(value is False for value in present_values):
                phase = RuntimeOutputPhase.OUTPUT_ABSENT
            else:
                phase = RuntimeOutputPhase.OUTPUT_UNKNOWN
        if (
            official is None
            and kernel.actor_session_closed
            and phase is RuntimeOutputPhase.OUTPUT_PRESENT_UNVERIFIED
        ):
            phase = RuntimeOutputPhase.AWAITING_OFFICIAL_EVALUATION
        obligations.append(
            RuntimeOutputObligation(
                obligation_id=f"public-output-obligation:{node.task_id}",
                task_id=node.task_id,
                output_requirement=node.output_requirement,
                output_resource_ids=resources,
                resource_versions=tuple(
                    (resource_id, kernel.resource_versions[resource_id])
                    for resource_id in resources
                ),
                resource_existence=existence,
                affected_task_ids=_affected_task_ids(node, kernel),
                phase=phase,
            )
        )
    return tuple(sorted(obligations, key=lambda item: item.obligation_id))


class RuntimeInterventionDeliveryStatus(StrEnum):
    PROPOSED = "proposed"
    STARTED = "started"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED_UNKNOWN = "cancelled_unknown"


class RuntimeInterventionEffectStatus(StrEnum):
    PENDING = "pending"
    OUTPUT_CHANGED = "output_changed"
    EVIDENCE_ADDED = "evidence_added"
    OFFICIAL_SUCCEEDED_AFTER_DELIVERY = "official_succeeded_after_delivery"
    OFFICIAL_FAILED_AFTER_DELIVERY = "official_failed_after_delivery"
    NO_OBSERVED_EFFECT = "no_observed_effect"
    INVALIDATED = "invalidated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RuntimeInterventionEffectRecord:
    """Keep message delivery separate from any later observed task effect."""

    action_id: str
    obligation_id: str
    obligation_version_key: tuple[str, str, tuple[tuple[str, int], ...], str]
    delivery_status: RuntimeInterventionDeliveryStatus
    effect_status: RuntimeInterventionEffectStatus
    effect_evidence_id: str | None = None
    delivery_correction_trigger_action_id: str | None = None


class RuntimeInterventionEffectLedger:
    """Append and replay intervention effects without treating delivery as success."""

    schema_version = "runtime_intervention_effect_event_v1"

    def __init__(self, path: Path, *, load_existing: bool = False) -> None:
        self.path = path
        self._records: dict[str, RuntimeInterventionEffectRecord] = {}
        self._sequence = 0
        if load_existing:
            self._load()
        else:
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    @property
    def records(self) -> tuple[RuntimeInterventionEffectRecord, ...]:
        return tuple(self._records.values())

    def _append(self, event_type: str, payload: dict[str, object]) -> None:
        with self.path.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "schema_version": self.schema_version,
                        "sequence": self._sequence,
                        "event_type": event_type,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._sequence += 1

    @staticmethod
    def _key_payload(
        key: tuple[str, str, tuple[tuple[str, int], ...], str]
    ) -> list[object]:
        return [key[0], key[1], [list(item) for item in key[2]], key[3]]

    @staticmethod
    def _decode_key(raw: object) -> tuple[str, str, tuple[tuple[str, int], ...], str]:
        if (
            not isinstance(raw, list)
            or len(raw) != 4
            or not isinstance(raw[0], str)
            or not isinstance(raw[1], str)
            or not isinstance(raw[2], list)
            or not isinstance(raw[3], str)
        ):
            raise ValueError("runtime intervention obligation key is invalid")
        versions = []
        for item in raw[2]:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
                or item[1] < 0
            ):
                raise ValueError("runtime intervention resource version is invalid")
            versions.append((item[0], item[1]))
        try:
            phase = RuntimeOutputPhase(raw[3])
            requirement = OutputRequirement(raw[1])
        except ValueError as exc:
            raise ValueError("runtime intervention obligation key enum is invalid") from exc
        return (raw[0], requirement.value, tuple(versions), phase.value)

    def record_delivery(
        self,
        *,
        action_id: str,
        obligation: RuntimeOutputObligation,
        status: RuntimeInterventionDeliveryStatus,
    ) -> None:
        terminal_delivery_statuses = {
            RuntimeInterventionDeliveryStatus.DELIVERED,
            RuntimeInterventionDeliveryStatus.FAILED,
            RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN,
        }
        if (
            not action_id
            or action_id in self._records
            or status not in terminal_delivery_statuses
        ):
            raise ValueError("runtime intervention action identity must be new")
        effect = (
            RuntimeInterventionEffectStatus.PENDING
            if status is RuntimeInterventionDeliveryStatus.DELIVERED
            else RuntimeInterventionEffectStatus.UNKNOWN
        )
        record = RuntimeInterventionEffectRecord(
            action_id=action_id,
            obligation_id=obligation.obligation_id,
            obligation_version_key=obligation.version_key,
            delivery_status=status,
            effect_status=effect,
        )
        self._records[action_id] = record
        self._append(
            "delivery_recorded",
            {
                "action_id": action_id,
                "obligation_id": obligation.obligation_id,
                "obligation_version_key": self._key_payload(obligation.version_key),
                "delivery_status": status.value,
                "effect_status": effect.value,
            },
        )

    def record_recovered_unknown_delivery(
        self,
        *,
        action_id: str,
        obligation_id: str,
        obligation_version_key: object,
    ) -> None:
        """Close a cross-WAL crash gap without claiming that Pi received the message."""

        key = self._decode_key(obligation_version_key)
        if (
            not action_id
            or action_id in self._records
            or not obligation_id
            or obligation_id != f"public-output-obligation:{key[0]}"
        ):
            raise ValueError("recovered runtime intervention identity is invalid")
        record = RuntimeInterventionEffectRecord(
            action_id=action_id,
            obligation_id=obligation_id,
            obligation_version_key=key,
            delivery_status=RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN,
            effect_status=RuntimeInterventionEffectStatus.UNKNOWN,
        )
        self._records[action_id] = record
        self._append(
            "delivery_recorded",
            {
                "action_id": action_id,
                "obligation_id": obligation_id,
                "obligation_version_key": self._key_payload(key),
                "delivery_status": (
                    RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN.value
                ),
                "effect_status": RuntimeInterventionEffectStatus.UNKNOWN.value,
            },
        )

    def correct_interrupted_delivery_without_pi_receipt(
        self, *, action_id: str, trigger_action_id: str
    ) -> None:
        """Downgrade a prepared notice when an interrupted Pi attempt never received it."""

        record = self._records[action_id]
        if (
            record.delivery_status is not RuntimeInterventionDeliveryStatus.DELIVERED
            or record.effect_status is not RuntimeInterventionEffectStatus.PENDING
            or record.effect_evidence_id is not None
            or not trigger_action_id
        ):
            raise ValueError("runtime intervention delivery correction is invalid")
        updated = RuntimeInterventionEffectRecord(
            action_id=record.action_id,
            obligation_id=record.obligation_id,
            obligation_version_key=record.obligation_version_key,
            delivery_status=RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN,
            effect_status=RuntimeInterventionEffectStatus.UNKNOWN,
            delivery_correction_trigger_action_id=trigger_action_id,
        )
        self._records[action_id] = updated
        self._append(
            "delivery_corrected",
            {
                "action_id": action_id,
                "trigger_action_id": trigger_action_id,
                "delivery_status": RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN.value,
                "effect_status": RuntimeInterventionEffectStatus.UNKNOWN.value,
                "reason": "interrupted Pi attempt lacks tool_execution_end",
            },
        )

    def record_effect(
        self,
        *,
        action_id: str,
        status: RuntimeInterventionEffectStatus,
        evidence_id: str,
    ) -> None:
        record = self._records[action_id]
        if (
            record.delivery_status is not RuntimeInterventionDeliveryStatus.DELIVERED
            or record.effect_status is not RuntimeInterventionEffectStatus.PENDING
            or status is RuntimeInterventionEffectStatus.PENDING
            or not evidence_id
        ):
            raise ValueError("runtime intervention effect transition is invalid")
        updated = RuntimeInterventionEffectRecord(
            action_id=record.action_id,
            obligation_id=record.obligation_id,
            obligation_version_key=record.obligation_version_key,
            delivery_status=record.delivery_status,
            effect_status=status,
            effect_evidence_id=evidence_id,
            delivery_correction_trigger_action_id=(
                record.delivery_correction_trigger_action_id
            ),
        )
        self._records[action_id] = updated
        self._append(
            "effect_recorded",
            {
                "action_id": action_id,
                "effect_status": status.value,
                "effect_evidence_id": evidence_id,
            },
        )

    def _load(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line]
        for sequence, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != self.schema_version
                or row.get("sequence") != sequence
                or not isinstance(row.get("payload"), dict)
            ):
                raise ValueError("runtime intervention effect log is invalid")
            payload = row["payload"]
            action_id = payload.get("action_id")
            if not isinstance(action_id, str):
                raise ValueError("runtime intervention effect action is invalid")
            if row.get("event_type") == "delivery_recorded":
                if action_id in self._records:
                    raise ValueError("runtime intervention delivery is duplicated")
                obligation_id = payload.get("obligation_id")
                if not isinstance(obligation_id, str):
                    raise ValueError("runtime intervention obligation is invalid")
                key = self._decode_key(payload.get("obligation_version_key"))
                delivery = RuntimeInterventionDeliveryStatus(
                    payload.get("delivery_status")
                )
                effect = RuntimeInterventionEffectStatus(payload.get("effect_status"))
                expected_effect = (
                    RuntimeInterventionEffectStatus.PENDING
                    if delivery is RuntimeInterventionDeliveryStatus.DELIVERED
                    else RuntimeInterventionEffectStatus.UNKNOWN
                )
                if (
                    not obligation_id
                    or obligation_id != f"public-output-obligation:{key[0]}"
                    or delivery
                    not in {
                        RuntimeInterventionDeliveryStatus.DELIVERED,
                        RuntimeInterventionDeliveryStatus.FAILED,
                        RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN,
                    }
                    or effect is not expected_effect
                ):
                    raise ValueError("runtime intervention delivery state is invalid")
                self._records[action_id] = RuntimeInterventionEffectRecord(
                    action_id=action_id,
                    obligation_id=obligation_id,
                    obligation_version_key=key,
                    delivery_status=delivery,
                    effect_status=effect,
                )
            elif row.get("event_type") == "effect_recorded":
                evidence_id = payload.get("effect_evidence_id")
                prior = self._records.get(action_id)
                status = RuntimeInterventionEffectStatus(payload.get("effect_status"))
                if (
                    not isinstance(evidence_id, str)
                    or prior is None
                    or prior.delivery_status
                    is not RuntimeInterventionDeliveryStatus.DELIVERED
                    or prior.effect_status is not RuntimeInterventionEffectStatus.PENDING
                    or status is RuntimeInterventionEffectStatus.PENDING
                ):
                    raise ValueError("runtime intervention effect evidence is invalid")
                self._records[action_id] = RuntimeInterventionEffectRecord(
                    action_id=action_id,
                    obligation_id=prior.obligation_id,
                    obligation_version_key=prior.obligation_version_key,
                    delivery_status=prior.delivery_status,
                    effect_status=status,
                    effect_evidence_id=evidence_id,
                    delivery_correction_trigger_action_id=(
                        prior.delivery_correction_trigger_action_id
                    ),
                )
            elif row.get("event_type") == "delivery_corrected":
                prior = self._records.get(action_id)
                if (
                    prior is None
                    or prior.delivery_status
                    is not RuntimeInterventionDeliveryStatus.DELIVERED
                    or prior.effect_status is not RuntimeInterventionEffectStatus.PENDING
                    or prior.effect_evidence_id is not None
                    or payload.get("delivery_status")
                    != RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN.value
                    or payload.get("effect_status")
                    != RuntimeInterventionEffectStatus.UNKNOWN.value
                    or not isinstance(payload.get("trigger_action_id"), str)
                    or not payload.get("trigger_action_id")
                    or payload.get("reason")
                    != "interrupted Pi attempt lacks tool_execution_end"
                ):
                    raise ValueError("runtime intervention delivery correction is invalid")
                self._records[action_id] = RuntimeInterventionEffectRecord(
                    action_id=action_id,
                    obligation_id=prior.obligation_id,
                    obligation_version_key=prior.obligation_version_key,
                    delivery_status=RuntimeInterventionDeliveryStatus.CANCELLED_UNKNOWN,
                    effect_status=RuntimeInterventionEffectStatus.UNKNOWN,
                    delivery_correction_trigger_action_id=payload[
                        "trigger_action_id"
                    ],
                )
            else:
                raise ValueError("runtime intervention effect event type is invalid")
            self._sequence = sequence + 1


__all__ = [
    "RuntimeInterventionDeliveryStatus",
    "RuntimeInterventionEffectLedger",
    "RuntimeInterventionEffectRecord",
    "RuntimeInterventionEffectStatus",
    "RuntimeOutputObligation",
    "RuntimeOutputPhase",
    "compile_runtime_output_obligations",
]
