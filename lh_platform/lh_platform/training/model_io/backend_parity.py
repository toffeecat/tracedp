"""Backend-neutral receipts for exact model-I/O sample parity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from lh_platform.training.agentic_rl_runtime import OnPolicyActionSample
from lh_platform.training.model_io.fingerprints import canonical_sha256
from lh_platform.training.model_io.messages import ModelIOContractError

MODEL_IO_BACKEND_PARITY_SCHEMA = "lh_model_io_backend_parity/v1"


@dataclass(frozen=True)
class ModelIOBackendSampleProjection:
    traj_uid: str
    step_idx: int
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    old_logprob_width: int
    model_io_profile_sha256: str | None
    response_contract_sha256: str | None
    normalized_messages_sha256: str | None
    chat_template_sha256: str | None
    matched_stop_text: str | None
    finish_reason: str


def project_model_io_sample(
    sample: OnPolicyActionSample,
) -> ModelIOBackendSampleProjection:
    if len(sample.response_mask) != len(sample.action_token_ids) or any(
        value not in (0, 1) for value in sample.response_mask
    ):
        raise ModelIOContractError(
            "model-I/O backend sample omitted or corrupted its response mask"
        )
    return ModelIOBackendSampleProjection(
        traj_uid=sample.traj_uid,
        step_idx=sample.step_idx,
        prompt_token_ids=sample.prompt_token_ids,
        response_token_ids=sample.action_token_ids,
        response_mask=sample.response_mask,
        old_logprob_width=len(sample.old_action_token_log_probs),
        model_io_profile_sha256=sample.model_io_profile_sha256,
        response_contract_sha256=sample.response_contract_sha256,
        normalized_messages_sha256=sample.normalized_messages_sha256,
        chat_template_sha256=sample.chat_template_sha256,
        matched_stop_text=sample.matched_stop_text,
        finish_reason=sample.finish_reason,
    )


def build_model_io_backend_parity_receipt(
    local_samples: Sequence[OnPolicyActionSample],
    compared_samples: Sequence[OnPolicyActionSample],
    *,
    compared_backend: str,
) -> dict[str, Any]:
    """Compare only model-I/O facts; optimizer numerical parity remains separate."""

    if not local_samples or not compared_samples:
        raise ModelIOContractError("model-I/O backend parity requires non-empty samples")
    local = tuple(project_model_io_sample(sample) for sample in local_samples)
    compared = tuple(project_model_io_sample(sample) for sample in compared_samples)
    if len(local) != len(compared):
        raise ModelIOContractError("model-I/O backend parity sample counts differ")
    mismatches = []
    rows = []
    for index, (left, right) in enumerate(zip(local, compared, strict=True)):
        equal = left == right
        if not equal:
            mismatches.append(index)
        rows.append(
            {
                "sample_index": index,
                "equal": equal,
                "local": asdict(left),
                "compared": asdict(right),
            }
        )
    payload: dict[str, Any] = {
        "schema": MODEL_IO_BACKEND_PARITY_SCHEMA,
        "local_backend": "local_transformers",
        "compared_backend": str(compared_backend),
        "sample_count": len(rows),
        "mismatch_indices": mismatches,
        "passed": not mismatches,
        "rows": rows,
        "claim_scope": "model_io_serialization_and_transport_not_policy_capability",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def require_model_io_backend_parity(receipt: dict[str, Any]) -> None:
    if (
        receipt.get("schema") != MODEL_IO_BACKEND_PARITY_SCHEMA
        or receipt.get("passed") is not True
        or receipt.get("mismatch_indices") != []
    ):
        raise ModelIOContractError("model-I/O backend parity did not pass")
    expected = dict(receipt)
    actual_sha256 = expected.pop("receipt_sha256", None)
    if actual_sha256 != canonical_sha256(expected):
        raise ModelIOContractError("model-I/O backend parity receipt hash drifted")


__all__ = [
    "MODEL_IO_BACKEND_PARITY_SCHEMA",
    "ModelIOBackendSampleProjection",
    "build_model_io_backend_parity_receipt",
    "project_model_io_sample",
    "require_model_io_backend_parity",
]
