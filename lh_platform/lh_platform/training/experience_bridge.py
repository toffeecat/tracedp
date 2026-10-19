"""Exact-token seam between LongHorizon credit and training backends.

LongHorizon owns rollout, grouping, and advantage semantics.  A backend may
only consume these immutable records, apply an optimizer update, and emit a
checkpoint.  In particular, it must not decode and retokenize actions or
recompute reward/advantage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from lh_platform.core.atomic_io import atomic_write_jsonl
from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    AgenticRolloutBatch,
    CheckpointArtifact,
    PolicyUpdateResult,
    PreparedPolicyUpdate,
    canonical_sha256,
)
from lh_platform.training.dataset import sha256_file

if TYPE_CHECKING:
    from lh_platform.training.backends.local_transformers import (
        LocalTransformersBackend,
    )


EXPERIENCE_SUBMISSION_SCHEMA = "lh_experience_submission/v1"
TRAINING_BACKEND_IDENTITY_SCHEMA = "lh_training_backend_identity/v1"


@dataclass(frozen=True)
class BackendIdentity:
    """Requested and observed optimizer backend identity for one run."""

    schema: str
    requested_backend: str
    actual_backend: str
    adapter_revision: str
    execution_mode: str
    source_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != TRAINING_BACKEND_IDENTITY_SCHEMA:
            raise AgenticRLContractError("unknown training backend identity schema")
        for label, value in (
            ("requested backend", self.requested_backend),
            ("actual backend", self.actual_backend),
            ("backend adapter revision", self.adapter_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise AgenticRLContractError(f"{label} must be non-empty text")
        if self.execution_mode not in {"in_process", "isolated_process"}:
            raise AgenticRLContractError("unsupported training backend execution mode")
        if self.source_contract_sha256 is not None and (
            len(self.source_contract_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_contract_sha256)
        ):
            raise AgenticRLContractError("training backend source contract must be sha256")


@dataclass(frozen=True)
class ExperienceRecord:
    """One exact policy step crossing the control-plane/backend boundary."""

    traj_uid: str
    group_uid: str
    step_idx: int
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_old_logprobs: tuple[float, ...]
    step_advantage: float
    actor_token_weight: float
    policy_checkpoint_sha256: str
    rollout_seed: int
    sampling_temperature: float
    allowed_response_token_ids: tuple[tuple[int, ...], ...] = ()
    is_last_episode: bool = False

    @property
    def key(self) -> tuple[str, int]:
        return (self.traj_uid, self.step_idx)

    @property
    def active_token_count(self) -> int:
        return sum(self.response_mask)


@dataclass(frozen=True)
class ExperienceSubmissionReceipt:
    schema: str
    batch_sha256: str
    record_count: int
    active_token_count: int
    policy_checkpoint_sha256: str
    backend: BackendIdentity


class ExperienceBridge(Protocol):
    """Backend contract; semantics are fixed before ``submit`` is called."""

    @property
    def identity(self) -> BackendIdentity: ...

    def submit(self, records: Sequence[ExperienceRecord]) -> ExperienceSubmissionReceipt: ...

    def update(self) -> PolicyUpdateResult: ...

    def checkpoint(self, path: Path) -> CheckpointArtifact: ...


def build_experience_records(
    rollout: AgenticRolloutBatch,
    update: PreparedPolicyUpdate,
) -> tuple[ExperienceRecord, ...]:
    """Bind exact rollout tokens to externally prepared policy targets."""

    if update.objective != "clipped_policy_gradient":
        raise AgenticRLContractError(
            f"experience bridge does not implement objective {update.objective!r}"
        )
    rollout = rollout.validate()
    rows = {(row.traj_uid, row.step_idx): row for row in rollout.records}
    samples = {(sample.traj_uid, sample.step_idx): sample for sample in rollout.policy_samples}
    targets = {(target.traj_uid, target.step_idx): target for target in update.targets}
    if len(targets) != len(update.targets):
        raise AgenticRLContractError("duplicate experience policy target key")
    if set(rows) != set(targets):
        raise AgenticRLContractError(
            "experience bridge requires exactly one target per rollout transition"
        )

    records = []
    for key in sorted(rows):
        row = rows[key]
        sample = samples[key]
        target = targets[key]
        expected_token_evidence = (
            sample.prompt_token_ids,
            sample.action_token_ids,
            (1,) * len(sample.action_token_ids),
            sample.old_action_token_log_probs,
        )
        if (
            row.prompt_ids,
            row.response_ids,
            row.response_mask,
            row.response_old_logprobs,
        ) != expected_token_evidence:
            raise AgenticRLContractError(
                "canonical trajectory lacks matching exact-token policy evidence"
            )
        records.append(
            ExperienceRecord(
                traj_uid=row.traj_uid,
                group_uid=row.group_uid,
                step_idx=row.step_idx,
                prompt_ids=sample.prompt_token_ids,
                response_ids=sample.action_token_ids,
                response_mask=(1,) * len(sample.action_token_ids),
                response_old_logprobs=sample.old_action_token_log_probs,
                step_advantage=target.advantage,
                actor_token_weight=target.actor_token_weight,
                policy_checkpoint_sha256=sample.policy_checkpoint_sha256,
                rollout_seed=sample.rollout_seed,
                sampling_temperature=sample.sampling_temperature,
                allowed_response_token_ids=sample.allowed_action_token_ids,
                is_last_episode=row.terminal,
            )
        )
    return validate_experience_records(records)


def validate_experience_records(
    records: Sequence[ExperienceRecord],
) -> tuple[ExperienceRecord, ...]:
    """Fail closed on token drift, policy mixing, or invalid gradient mass."""

    frozen = tuple(records)
    if not frozen:
        raise AgenticRLContractError("experience submission must not be empty")
    keys: set[tuple[str, int]] = set()
    checkpoints: set[str] = set()
    mass_by_trajectory: dict[str, float] = {}
    for row in frozen:
        if not isinstance(row, ExperienceRecord):
            raise AgenticRLContractError(
                "experience submission contains a non-ExperienceRecord value"
            )
        if row.key in keys:
            raise AgenticRLContractError(f"duplicate experience key: {row.key!r}")
        keys.add(row.key)
        if (
            not isinstance(row.traj_uid, str)
            or not row.traj_uid.strip()
            or not isinstance(row.group_uid, str)
            or not row.group_uid.strip()
            or type(row.step_idx) is not int
            or row.step_idx < 0
        ):
            raise AgenticRLContractError("experience identity fields are invalid")
        if type(row.rollout_seed) is not int or row.rollout_seed < 0:
            raise AgenticRLContractError("experience rollout seed must be a non-negative integer")
        if type(row.is_last_episode) is not bool:
            raise AgenticRLContractError("experience terminal flag must be boolean")
        _validate_token_ids(row.prompt_ids, label="prompt")
        _validate_token_ids(row.response_ids, label="response")
        if (
            not isinstance(row.response_mask, tuple)
            or len(row.response_mask) != len(row.response_ids)
            or any(type(value) is not int or value not in (0, 1) for value in row.response_mask)
        ):
            raise AgenticRLContractError(
                "response mask must align with response IDs and contain only 0/1"
            )
        if row.active_token_count < 1:
            raise AgenticRLContractError("experience requires at least one active token")
        if (
            not isinstance(row.response_old_logprobs, tuple)
            or len(row.response_old_logprobs) != len(row.response_ids)
            or any(not _is_finite_number(value) for value in row.response_old_logprobs)
        ):
            raise AgenticRLContractError(
                "old logprobs must align exactly with response IDs and be finite"
            )
        if not _is_finite_number(row.step_advantage):
            raise AgenticRLContractError("step advantage must be finite")
        if not _is_finite_number(row.actor_token_weight) or row.actor_token_weight <= 0:
            raise AgenticRLContractError("actor token weight must be finite and positive")
        if not _is_finite_number(row.sampling_temperature) or row.sampling_temperature <= 0:
            raise AgenticRLContractError("sampling temperature must be finite and positive")
        if row.allowed_response_token_ids:
            if not isinstance(row.allowed_response_token_ids, tuple) or len(
                row.allowed_response_token_ids
            ) != len(row.response_ids):
                raise AgenticRLContractError("allowed token sets must align with response IDs")
            for selected, allowed in zip(
                row.response_ids,
                row.allowed_response_token_ids,
                strict=True,
            ):
                _validate_token_ids(allowed, label="allowed response")
                if len(set(allowed)) != len(allowed):
                    raise AgenticRLContractError("allowed response token IDs must be unique")
                if selected not in allowed:
                    raise AgenticRLContractError(
                        "response token is outside recorded policy support"
                    )
        if not isinstance(row.policy_checkpoint_sha256, str):
            raise AgenticRLContractError("experience policy checkpoint must be sha256")
        checkpoints.add(row.policy_checkpoint_sha256)
        mass_by_trajectory[row.traj_uid] = (
            mass_by_trajectory.get(row.traj_uid, 0.0)
            + row.actor_token_weight * row.active_token_count
        )
    if len(checkpoints) != 1:
        raise AgenticRLContractError("experience submission mixes policy checkpoints")
    checkpoint = next(iter(checkpoints))
    if len(checkpoint) != 64 or any(char not in "0123456789abcdef" for char in checkpoint):
        raise AgenticRLContractError("experience policy checkpoint must be sha256")
    for traj_uid, mass in mass_by_trajectory.items():
        if not math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AgenticRLContractError(
                f"trajectory {traj_uid!r} experience gradient mass is {mass}, expected 1"
            )
    return frozen


def experience_batch_sha256(records: Sequence[ExperienceRecord]) -> str:
    return canonical_sha256([asdict(row) for row in records])


def write_experience_artifact(
    path: Path,
    records: Sequence[ExperienceRecord],
) -> str:
    """Persist immutable credit-to-optimizer inputs and return file SHA-256."""

    validated = validate_experience_records(records)
    target = Path(path)
    atomic_write_jsonl(target, (asdict(row) for row in validated))
    return sha256_file(target)


def _validate_token_ids(values: Sequence[int], *, label: str) -> None:
    if (
        not isinstance(values, tuple)
        or not values
        or any(type(token) is not int or token < 0 for token in values)
    ):
        raise AgenticRLContractError(f"{label} token IDs must be non-empty non-negative integers")


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def __getattr__(name: str) -> object:
    """Keep the Phase-1 import path while backend modules deepen independently."""

    if name == "LocalTransformersBackend":
        from lh_platform.training.backends.local_transformers import (
            LocalTransformersBackend,
        )

        return LocalTransformersBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EXPERIENCE_SUBMISSION_SCHEMA",
    "TRAINING_BACKEND_IDENTITY_SCHEMA",
    "BackendIdentity",
    "ExperienceBridge",
    "ExperienceRecord",
    "ExperienceSubmissionReceipt",
    "LocalTransformersBackend",
    "build_experience_records",
    "experience_batch_sha256",
    "validate_experience_records",
    "write_experience_artifact",
]
