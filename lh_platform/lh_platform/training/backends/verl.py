"""Fail-closed isolated veRL optimizer backend."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    CheckpointArtifact,
    ComponentRef,
    OnPolicyActionSample,
    PolicyActionRequest,
    PolicyUpdateResult,
    validate_action_sample_for_request,
)
from lh_platform.training.backends.verl_protocol import (
    VerLWorkerClient,
    VerLWorkerError,
)
from lh_platform.training.backends.verl_source import (
    VERL_PINNED_SOURCE_CONTRACT,
    GitSourceContract,
    verify_git_source,
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

logger = logging.getLogger(__name__)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class VerLBackend:
    """Own one veRL policy/optimizer process behind exact-token JSON RPC."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        policy_payload: Mapping[str, Any],
        source_contract: GitSourceContract = VERL_PINNED_SOURCE_CONTRACT,
        python_executable: str = sys.executable,
        startup_timeout_seconds: float = 900.0,
        request_timeout_seconds: float = 1800.0,
        _worker_command: Sequence[str] | None = None,
    ) -> None:
        if source_contract.backend_id != "verl":
            raise AgenticRLContractError("veRL backend requires a veRL source contract")
        if not isinstance(policy_payload, Mapping):
            raise AgenticRLContractError("veRL policy payload must be a mapping")
        verified_source = verify_git_source(source_root, source_contract)
        command = tuple(
            _worker_command
            or (
                python_executable,
                "-m",
                "lh_platform.training.backends.verl_worker",
            )
        )
        self._worker = VerLWorkerClient(
            command,
            startup_timeout_seconds=startup_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
        self._failed = False
        self._pending = False
        self._closed = False
        try:
            ready = self._worker.request(
                "hello",
                {
                    "source_root": verified_source.source_root,
                    "source_contract": asdict(source_contract),
                    "source_receipt_contract_sha256": verified_source.contract_sha256,
                    "policy": dict(policy_payload),
                },
                timeout_seconds=startup_timeout_seconds,
            )
        except Exception:
            self._failed = True
            self._worker.close()
            raise
        revision = ready.get("adapter_revision")
        checkpoint = ready.get("checkpoint_sha256")
        if not isinstance(revision, str) or not revision.strip():
            self.close()
            raise AgenticRLContractError("veRL worker omitted adapter revision")
        self._checkpoint_sha256 = _require_sha256(checkpoint, "veRL worker checkpoint")
        component_payload = ready.get("component")
        if not isinstance(component_payload, dict):
            self.close()
            raise AgenticRLContractError("veRL worker omitted policy component")
        self.component = ComponentRef(**component_payload)
        adapter_initial_delta_zero = ready.get("adapter_initial_delta_zero")
        if type(adapter_initial_delta_zero) is not bool:
            self.close()
            raise AgenticRLContractError("veRL worker omitted adapter initialization audit")
        self.adapter_initial_delta_zero = adapter_initial_delta_zero
        policy_config = policy_payload.get("config")
        expected_model_io_sha256 = (
            policy_config.get("model_io_profile_sha256")
            if isinstance(policy_config, Mapping)
            else None
        )
        actual_model_io_sha256 = ready.get("model_io_profile_sha256")
        if actual_model_io_sha256 != expected_model_io_sha256:
            self.close()
            raise AgenticRLContractError("veRL parent/child model I/O profile hash drifted")
        self.model_io_profile_sha256 = actual_model_io_sha256
        self._samples: list[OnPolicyActionSample] = []
        self._identity = BackendIdentity(
            schema=TRAINING_BACKEND_IDENTITY_SCHEMA,
            requested_backend="verl",
            actual_backend="verl",
            adapter_revision=revision,
            execution_mode="isolated_process",
            source_contract_sha256=source_contract.sha256,
        )

    @property
    def identity(self) -> BackendIdentity:
        return self._identity

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    def sampling_cursor(self) -> int:
        return len(self._samples)

    def sampling_events_since(self, cursor: int) -> tuple[OnPolicyActionSample, ...]:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            return ()
        return tuple(self._samples[cursor:])

    def sample_actions(
        self,
        requests: tuple[PolicyActionRequest, ...],
        *,
        seed: int,
    ) -> tuple[OnPolicyActionSample, ...]:
        self._require_healthy()
        if self._pending:
            raise AgenticRLContractError(
                "veRL backend cannot sample while an update batch is pending"
            )
        if type(seed) is not int or seed < 0:
            raise AgenticRLContractError("veRL sampling seed must be non-negative")
        if not isinstance(requests, tuple) or any(
            not isinstance(request, PolicyActionRequest) for request in requests
        ):
            raise AgenticRLContractError("veRL sampling requires PolicyActionRequest records")
        payload = self._rpc(
            "sample_actions",
            {
                "requests": [asdict(request) for request in requests],
                "seed": seed,
            },
        )
        rows = payload.get("samples")
        if not isinstance(rows, list) or len(rows) != len(requests):
            self._poison("veRL worker returned a different sample count")
        samples = tuple(_decode_action_sample(row) for row in rows)
        for request, sample in zip(requests, samples, strict=True):
            validate_action_sample_for_request(
                request,
                sample,
                expected_model_io_profile_sha256=self.model_io_profile_sha256,
            )
            if sample.policy_checkpoint_sha256 != self._checkpoint_sha256:
                self._poison("veRL worker sample identity drifted")
        self._samples.extend(samples)
        return samples

    def submit(
        self,
        records: Sequence[ExperienceRecord],
    ) -> ExperienceSubmissionReceipt:
        self._require_healthy()
        if self._pending:
            raise AgenticRLContractError("veRL backend already has an unconsumed submission")
        validated = validate_experience_records(records)
        checkpoint = validated[0].policy_checkpoint_sha256
        if checkpoint != self._checkpoint_sha256:
            raise AgenticRLContractError("experience checkpoint differs from current veRL policy")
        batch_sha256 = experience_batch_sha256(validated)
        result = self._rpc(
            "submit",
            {
                "batch_sha256": batch_sha256,
                "records": [asdict(row) for row in validated],
            },
        )
        if result.get("batch_sha256") != batch_sha256:
            self._poison("veRL worker accepted a different experience batch")
        self._pending = True
        return ExperienceSubmissionReceipt(
            schema=EXPERIENCE_SUBMISSION_SCHEMA,
            batch_sha256=batch_sha256,
            record_count=len(validated),
            active_token_count=sum(row.active_token_count for row in validated),
            policy_checkpoint_sha256=checkpoint,
            backend=self.identity,
        )

    def update(self) -> PolicyUpdateResult:
        self._require_healthy()
        if not self._pending:
            raise AgenticRLContractError("veRL backend has no pending submission")
        before = self._checkpoint_sha256
        try:
            payload = self._rpc("update", {})
            result = PolicyUpdateResult(
                optimizer_step=_require_nonnegative_int(
                    payload.get("optimizer_step"), "optimizer step"
                ),
                checkpoint_before_sha256=_require_sha256(
                    payload.get("checkpoint_before_sha256"),
                    "veRL checkpoint before",
                ),
                checkpoint_after_sha256=_require_sha256(
                    payload.get("checkpoint_after_sha256"),
                    "veRL checkpoint after",
                ),
                trainable_parameter_delta_l2=_require_nonnegative_float(
                    payload.get("trainable_parameter_delta_l2"),
                    "trainable parameter delta",
                ),
                metrics=_require_metrics(payload.get("metrics")),
            )
            if result.checkpoint_before_sha256 != before:
                self._poison("veRL worker updated a different checkpoint")
        except Exception:
            self._failed = True
            raise
        self._checkpoint_sha256 = result.checkpoint_after_sha256
        self._pending = False
        return result

    def checkpoint(self, path: Path) -> CheckpointArtifact:
        self._require_healthy()
        if self._pending:
            raise AgenticRLContractError("veRL backend cannot checkpoint an unconsumed submission")
        output_dir = Path(path).resolve()
        payload = self._rpc("checkpoint", {"output_dir": str(output_dir)})
        artifact = CheckpointArtifact(
            path=str(payload.get("path") or ""),
            sha256=_require_sha256(payload.get("sha256"), "veRL checkpoint artifact"),
        )
        artifact_path = Path(artifact.path).resolve()
        if not artifact_path.is_relative_to(output_dir) or not artifact_path.is_file():
            self._poison("veRL worker checkpoint escaped or was not published")
        if artifact.sha256 != self._checkpoint_sha256:
            self._poison("veRL worker checkpoint identity drifted")
        return artifact

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            logger.exception("verl backend: close() failed during __del__")

    def _rpc(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return self._worker.request(operation, payload)
        except VerLWorkerError as error:
            raise AgenticRLContractError(str(error)) from error

    def _require_healthy(self) -> None:
        if self._closed:
            raise AgenticRLContractError("veRL backend is closed")
        if self._failed:
            raise AgenticRLContractError("veRL backend is failed and must be reconstructed")

    def _poison(self, message: str) -> None:
        self._failed = True
        raise AgenticRLContractError(message)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AgenticRLContractError(f"{label} must be lowercase sha256")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AgenticRLContractError(f"{label} must be a non-negative integer")
    return value


def _require_nonnegative_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise AgenticRLContractError(f"{label} must be a non-negative number")
    return float(value)


def _require_metrics(value: object) -> Mapping[str, float]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or isinstance(metric, bool) or not isinstance(metric, (int, float))
        for key, metric in value.items()
    ):
        raise AgenticRLContractError("veRL update metrics must be numeric")
    return {key: float(metric) for key, metric in value.items()}


def _decode_action_sample(payload: object) -> OnPolicyActionSample:
    if not isinstance(payload, dict):
        raise AgenticRLContractError("veRL action sample must be an object")
    normalized = dict(payload)
    for field in (
        "prompt_token_ids",
        "action_token_ids",
        "response_mask",
        "old_action_token_log_probs",
        "allowed_action_token_ids",
    ):
        normalized[field] = tuple(normalized.get(field, ()))
    normalized["allowed_action_token_ids"] = tuple(
        tuple(row) for row in normalized["allowed_action_token_ids"]
    )
    sample = OnPolicyActionSample(**normalized)
    if (
        not sample.action_token_ids
        or len(sample.response_mask) != len(sample.action_token_ids)
        or any(value not in (0, 1) for value in sample.response_mask)
        or len(sample.old_action_token_log_probs) != len(sample.action_token_ids)
    ):
        raise AgenticRLContractError("veRL action sample token evidence drifted")
    return sample


__all__ = ["VerLBackend"]
