"""No-GPU attestation of checkpoint-native prompt and assistant tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lh_platform.training.model_io.fingerprints import canonical_sha256
from lh_platform.training.model_io.messages import ModelIOContractError
from lh_platform.training.model_io.receipts import ModelIOAuditFixture
from lh_platform.training.model_io.renderer import NativeChatRenderer


def audit_model_io_fixtures(
    renderer: NativeChatRenderer,
    fixtures: Sequence[ModelIOAuditFixture],
    *,
    compare_legacy: bool = False,
) -> dict[str, Any]:
    """Render immutable token fixtures without loading model weights or a GPU."""

    if not fixtures:
        raise ModelIOContractError("model I/O audit has no fixtures")
    rows = []
    seen = set()
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            raise ModelIOContractError(f"duplicate model I/O fixture: {fixture.fixture_id}")
        seen.add(fixture.fixture_id)
        turn = renderer.render_assistant_turn(
            fixture.conversation,
            fixture.target,
        )
        row: dict[str, Any] = {
            "fixture_id": fixture.fixture_id,
            "normalized_messages_sha256": turn.normalized_messages_sha256,
            "assistant_target_sha256": canonical_sha256(fixture.target.content),
            "prompt_token_ids": list(turn.prompt_ids),
            "full_turn_token_ids": list(turn.full_ids),
            "response_token_ids": list(turn.response_ids),
            "response_mask": list(turn.response_mask),
            "native_turn_end_ids": list(turn.native_turn_end_ids),
            "assistant_content_token_count": turn.assistant_content_token_count,
        }
        if compare_legacy:
            row.update(_legacy_split_comparison(renderer, fixture, turn))
        rows.append(row)
    payload = {
        "schema": "lh_model_io_audit/v1",
        "tokenizer_class": type(renderer.tokenizer).__name__,
        "chat_template_sha256": rows and turn.chat_template_sha256,
        "special_tokens_map_sha256": canonical_sha256(
            getattr(renderer.tokenizer, "special_tokens_map", {})
        ),
        "model_io_profile": renderer.profile.to_dict(),
        "model_io_profile_sha256": renderer.profile.sha256,
        "fixtures": rows,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def model_io_golden_projection(receipt: dict[str, Any]) -> dict[str, Any]:
    """Select the stable fields that must remain identical across reruns."""

    package_versions = dict(receipt.get("package_versions", {}))
    tokenizer_artifact = dict(receipt.get("tokenizer_artifact", {}))
    return {
        "schema": "lh_model_io_golden/v1",
        "model_reference": receipt.get("model_reference"),
        "model_revision": receipt.get("model_revision"),
        "transformers_version": package_versions.get("transformers"),
        "tokenizers_version": package_versions.get("tokenizers"),
        "chat_template_sha256": receipt.get("chat_template_sha256"),
        "tokenizer_artifact_sha256": tokenizer_artifact.get("artifact_sha256"),
        "model_io_profile_sha256": receipt.get("model_io_profile_sha256"),
        "fixtures": [
            {
                key: row[key]
                for key in (
                    "fixture_id",
                    "prompt_token_ids",
                    "response_token_ids",
                    "native_turn_end_ids",
                    "legacy_prompt_equal",
                    "legacy_response_equal",
                )
            }
            for row in receipt.get("fixtures", ())
        ],
    }


def verify_model_io_golden(
    receipt: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    actual = model_io_golden_projection(receipt)
    if actual != expected:
        raise ModelIOContractError("model I/O token receipt differs from the pinned golden fixture")


def _legacy_split_comparison(
    renderer: NativeChatRenderer,
    fixture: ModelIOAuditFixture,
    turn: Any,
) -> dict[str, Any]:
    conversation = fixture.conversation
    if (
        len(conversation.messages) != 1
        or conversation.messages[0].role != "user"
        or conversation.tools
        or conversation.documents
    ):
        raise ModelIOContractError("legacy split comparison requires one plain user message")
    tokenizer = renderer.tokenizer
    try:
        rendered_prompt = tokenizer.apply_chat_template(
            [conversation.messages[0].to_dict()],
            tokenize=False,
            add_generation_prompt=True,
            **dict(renderer.profile.template_kwargs),
        )
        legacy_prompt = tuple(
            int(value)
            for value in tokenizer(
                rendered_prompt,
                add_special_tokens=False,
            )["input_ids"]
        )
        legacy_response = tuple(
            int(value)
            for value in tokenizer(
                fixture.target.content,
                add_special_tokens=False,
            )["input_ids"]
        )
    except Exception as error:
        raise ModelIOContractError(f"legacy split-token audit failed: {error}") from error
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and (not legacy_response or legacy_response[-1] != int(eos)):
        legacy_response = (*legacy_response, int(eos))
    return {
        "legacy_prompt_token_ids": list(legacy_prompt),
        "legacy_response_token_ids": list(legacy_response),
        "legacy_prompt_equal": legacy_prompt == turn.prompt_ids,
        "legacy_response_equal": legacy_response == turn.response_ids,
    }


__all__ = [
    "audit_model_io_fixtures",
    "model_io_golden_projection",
    "verify_model_io_golden",
]
