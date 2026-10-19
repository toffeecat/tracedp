"""Exact-token rendering through the checkpoint-native chat template."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lh_platform.training.model_io.contracts import AssistantTarget
from lh_platform.training.model_io.fingerprints import (
    canonical_sha256,
    chat_template_sha256,
)
from lh_platform.training.model_io.messages import (
    ChatMessage,
    ConversationIR,
    ModelIOContractError,
)
from lh_platform.training.model_io.profiles import ModelIOProfile


@dataclass(frozen=True)
class RenderedPrompt:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    normalized_messages_sha256: str
    chat_template_sha256: str
    model_io_profile_sha256: str


@dataclass(frozen=True)
class RenderedAssistantTurn:
    prompt_ids: tuple[int, ...]
    full_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    native_turn_end_ids: tuple[int, ...]
    assistant_content_token_count: int
    normalized_messages_sha256: str
    chat_template_sha256: str
    model_io_profile_sha256: str


class NativeChatRenderer:
    def __init__(self, tokenizer: Any, profile: ModelIOProfile) -> None:
        self.tokenizer = tokenizer
        self.profile = profile
        try:
            self._chat_template_sha256 = chat_template_sha256(tokenizer)
        except ValueError as error:
            raise ModelIOContractError(str(error)) from error

    @property
    def chat_template_sha256(self) -> str:
        """Expose the exact native template identity used by every render."""

        return self._chat_template_sha256

    def normalize(self, conversation: ConversationIR) -> ConversationIR:
        messages = list(conversation.messages)
        if self.profile.system_role_policy == "fold_into_first_user":
            system_messages = [message for message in messages if message.role == "system"]
            if system_messages:
                first_non_system = next(
                    (index for index, message in enumerate(messages) if message.role != "system"),
                    None,
                )
                if first_non_system is None or messages[first_non_system].role != "user":
                    raise ModelIOContractError(
                        "system folding requires a following first user message"
                    )
                if any(message.role == "system" for message in messages[first_non_system:]):
                    raise ModelIOContractError("system messages must precede the first user")
                merged = "\n\n".join(
                    [
                        *(message.content for message in system_messages),
                        messages[first_non_system].content,
                    ]
                )
                messages = [
                    ChatMessage(
                        role="user",
                        content=merged,
                        name=messages[first_non_system].name,
                    ),
                    *messages[first_non_system + 1 :],
                ]
        elif self.profile.system_role_policy == "forbid" and any(
            message.role == "system" for message in messages
        ):
            raise ModelIOContractError("model I/O profile forbids system messages")
        unsupported = sorted(
            {message.role for message in messages}.difference(self.profile.supported_roles)
        )
        if unsupported:
            raise ModelIOContractError(
                "model I/O profile does not support role(s): " + ", ".join(unsupported)
            )
        return ConversationIR(
            messages=tuple(messages),
            tools=conversation.tools,
            documents=conversation.documents,
        )

    def render_prompt(self, conversation: ConversationIR) -> RenderedPrompt:
        normalized = self.normalize(conversation)
        rendered = self._render(normalized, add_generation_prompt=True)
        input_ids, attention_mask = _extract_ids_and_mask(rendered)
        return RenderedPrompt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalized_messages_sha256=canonical_sha256(normalized.to_dict()),
            chat_template_sha256=self._chat_template_sha256,
            model_io_profile_sha256=self.profile.sha256,
        )

    def render_assistant_turn(
        self,
        conversation: ConversationIR,
        target: AssistantTarget,
    ) -> RenderedAssistantTurn:
        normalized = self.normalize(conversation)
        prompt = self.render_prompt(normalized)
        full = ConversationIR(
            messages=(*normalized.messages, target.to_message()),
            tools=normalized.tools,
            documents=normalized.documents,
        )
        full_ids, _ = _extract_ids_and_mask(self._render(full, add_generation_prompt=False))
        _require_prefix(prompt.input_ids, full_ids, label="full assistant turn")
        response_ids = full_ids[len(prompt.input_ids) :]
        native_turn_end_ids = self.native_turn_end_ids(normalized)
        if len(response_ids) <= len(native_turn_end_ids) or (
            response_ids[-len(native_turn_end_ids) :] != native_turn_end_ids
        ):
            raise ModelIOContractError("assistant target lacks native turn end")
        return RenderedAssistantTurn(
            prompt_ids=prompt.input_ids,
            full_ids=full_ids,
            response_ids=response_ids,
            response_mask=(1,) * len(response_ids),
            native_turn_end_ids=native_turn_end_ids,
            assistant_content_token_count=len(response_ids) - len(native_turn_end_ids),
            normalized_messages_sha256=prompt.normalized_messages_sha256,
            chat_template_sha256=prompt.chat_template_sha256,
            model_io_profile_sha256=prompt.model_io_profile_sha256,
        )

    def native_turn_end_ids(self, conversation: ConversationIR) -> tuple[int, ...]:
        normalized = self.normalize(conversation)
        prompt = self.render_prompt(normalized)
        if self.profile.assistant_turn_end_mode == "infer_from_distinct_assistant_suffix":
            return self._stable_assistant_suffix(normalized, prompt.input_ids)
        empty = ConversationIR(
            messages=(*normalized.messages, ChatMessage(role="assistant", content="")),
            tools=normalized.tools,
            documents=normalized.documents,
        )
        empty_ids, _ = _extract_ids_and_mask(self._render(empty, add_generation_prompt=False))
        _require_prefix(prompt.input_ids, empty_ids, label="empty assistant turn")
        native_turn_end_ids = empty_ids[len(prompt.input_ids) :]
        if not native_turn_end_ids:
            raise ModelIOContractError("native assistant turn end is empty")
        return native_turn_end_ids

    def _stable_assistant_suffix(
        self,
        conversation: ConversationIR,
        prompt_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        suffixes = []
        for marker in (
            "LONGHORIZON_TURN_END_PROBE_ALPHA_X",
            "LONGHORIZON_TURN_END_PROBE_OMEGA_Z",
        ):
            content = self._assistant_probe_content(marker)
            probe = ConversationIR(
                messages=(
                    *conversation.messages,
                    ChatMessage(role="assistant", content=content),
                ),
                tools=conversation.tools,
                documents=conversation.documents,
            )
            probe_ids, _ = _extract_ids_and_mask(self._render(probe, add_generation_prompt=False))
            _require_prefix(prompt_ids, probe_ids, label="assistant turn-end probe")
            suffixes.append(probe_ids[len(prompt_ids) :])
        stable = _longest_common_suffix(*suffixes)
        empty = ConversationIR(
            messages=(
                *conversation.messages,
                ChatMessage(role="assistant", content=""),
            ),
            tools=conversation.tools,
            documents=conversation.documents,
        )
        empty_ids, _ = _extract_ids_and_mask(self._render(empty, add_generation_prompt=False))
        stable = _longest_common_suffix(stable, empty_ids)
        if not stable or stable in suffixes:
            raise ModelIOContractError(
                "native assistant turn end could not be isolated from content"
            )
        return stable

    def _assistant_probe_content(self, marker: str) -> str:
        delimiters = self.profile.generated_reasoning_delimiters
        if delimiters == "paired":
            return f"<think>{marker}_REASON</think>{marker}_FINAL"
        if delimiters == "closing_only":
            return f"{marker}_REASON</think>{marker}_FINAL"
        return marker

    def _render(
        self,
        conversation: ConversationIR,
        *,
        add_generation_prompt: bool,
    ) -> Any:
        kwargs = dict(self.profile.template_kwargs)
        if conversation.tools:
            kwargs["tools"] = [dict(value) for value in conversation.tools]
        if conversation.documents:
            kwargs["documents"] = [dict(value) for value in conversation.documents]
        try:
            return self.tokenizer.apply_chat_template(
                [message.to_dict() for message in conversation.messages],
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                **kwargs,
            )
        except Exception as error:
            raise ModelIOContractError(f"native chat template rendering failed: {error}") from error


def _require_prefix(
    prompt_ids: tuple[int, ...],
    full_ids: tuple[int, ...],
    *,
    label: str,
) -> None:
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ModelIOContractError(f"prompt rendering is not an exact prefix of the {label}")


def _longest_common_suffix(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    width = 0
    limit = min(len(left), len(right))
    while width < limit and left[-(width + 1)] == right[-(width + 1)]:
        width += 1
    return left[len(left) - width :] if width else ()


def _extract_ids_and_mask(rendered: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(rendered, Mapping):
        raw_ids = rendered.get("input_ids")
        raw_mask = rendered.get("attention_mask")
    else:
        raw_ids = rendered
        raw_mask = None
    ids = _flatten_single_sequence(raw_ids, label="input_ids")
    if not ids:
        raise ModelIOContractError("native chat template returned no token IDs")
    mask = (
        _flatten_single_sequence(raw_mask, label="attention_mask")
        if raw_mask is not None
        else (1,) * len(ids)
    )
    if len(mask) != len(ids):
        raise ModelIOContractError("native chat template mask width differs from token IDs")
    return ids, mask


def _flatten_single_sequence(value: Any, *, label: str) -> tuple[int, ...]:
    if value is None:
        raise ModelIOContractError(f"native chat template omitted {label}")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ModelIOContractError(f"native chat template returned batched {label}")
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ModelIOContractError(f"native chat template returned invalid {label}")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ModelIOContractError(f"native chat template returned non-integer {label}") from error


__all__ = ["NativeChatRenderer", "RenderedAssistantTurn", "RenderedPrompt"]
