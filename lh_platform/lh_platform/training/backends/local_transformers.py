"""Single-process Transformers/PEFT backend adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    CheckpointArtifact,
    PolicyUpdateResult,
)
from lh_platform.training.experience_bridge import (
    EXPERIENCE_SUBMISSION_SCHEMA,
    TRAINING_BACKEND_IDENTITY_SCHEMA,
    BackendIdentity,
    ExperienceRecord,
    ExperienceSubmissionReceipt,
    experience_batch_sha256,
    validate_experience_records,
)


class LocalTransformersBackend:
    """Synchronous adapter for the existing local Transformers/PEFT policy."""

    def __init__(self, policy: object) -> None:
        self._policy = policy
        self._pending: tuple[ExperienceRecord, ...] | None = None
        self._failed = False
        self._identity = BackendIdentity(
            schema=TRAINING_BACKEND_IDENTITY_SCHEMA,
            requested_backend="local_transformers",
            actual_backend="local_transformers",
            adapter_revision="local_transformers/v1",
            execution_mode="in_process",
        )

    @property
    def identity(self) -> BackendIdentity:
        return self._identity

    def submit(
        self,
        records: Sequence[ExperienceRecord],
    ) -> ExperienceSubmissionReceipt:
        if self._failed:
            raise AgenticRLContractError("experience backend is failed and must be reconstructed")
        if self._pending is not None:
            raise AgenticRLContractError("experience backend already has an unconsumed submission")
        validated = validate_experience_records(records)
        checkpoint = validated[0].policy_checkpoint_sha256
        if getattr(self._policy, "checkpoint_sha256", None) != checkpoint:
            raise AgenticRLContractError(
                "experience checkpoint differs from current backend policy"
            )
        self._pending = validated
        return ExperienceSubmissionReceipt(
            schema=EXPERIENCE_SUBMISSION_SCHEMA,
            batch_sha256=experience_batch_sha256(validated),
            record_count=len(validated),
            active_token_count=sum(row.active_token_count for row in validated),
            policy_checkpoint_sha256=checkpoint,
            backend=self.identity,
        )

    def update(self) -> PolicyUpdateResult:
        if self._failed:
            raise AgenticRLContractError("experience backend is failed and must be reconstructed")
        if self._pending is None:
            raise AgenticRLContractError("experience backend has no pending submission")
        consumer = getattr(self._policy, "apply_experience_records", None)
        if not callable(consumer):
            raise AgenticRLContractError(
                "local policy does not implement exact-token experience updates"
            )
        pending = self._pending
        try:
            result = consumer(pending)
        except Exception:
            self._failed = True
            raise
        self._pending = None
        return result

    def checkpoint(self, path: Path) -> CheckpointArtifact:
        if self._failed:
            raise AgenticRLContractError("failed experience backend cannot publish a checkpoint")
        if self._pending is not None:
            raise AgenticRLContractError(
                "experience backend cannot checkpoint an unconsumed submission"
            )
        materialize = getattr(self._policy, "materialize_checkpoint", None)
        if not callable(materialize):
            raise AgenticRLContractError("local policy cannot materialize checkpoints")
        return materialize(Path(path))


__all__ = ["LocalTransformersBackend"]
