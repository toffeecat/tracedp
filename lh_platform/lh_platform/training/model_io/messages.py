"""Backend-neutral conversation objects for exact model I/O rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

ChatRole = Literal["system", "user", "assistant", "tool"]
_ROLES = {"system", "user", "assistant", "tool"}


class ModelIOContractError(ValueError):
    """Raised when a model I/O contract cannot be rendered exactly."""


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ModelIOContractError(f"unsupported chat role: {self.role}")
        if not isinstance(self.content, str):
            raise ModelIOContractError("chat message content must be text")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ModelIOContractError("tool_call_id is only valid on tool messages")

    def to_dict(self) -> dict[str, str]:
        payload = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True)
class ConversationIR:
    messages: tuple[ChatMessage, ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    documents: tuple[Mapping[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.messages:
            raise ModelIOContractError("conversation has no messages")
        if any(not isinstance(message, ChatMessage) for message in self.messages):
            raise ModelIOContractError("conversation contains an invalid message")

    @classmethod
    def from_legacy_user_prompt(cls, prompt_text: str) -> ConversationIR:
        if not str(prompt_text).strip():
            raise ModelIOContractError("legacy actor prompt is empty")
        return cls(messages=(ChatMessage(role="user", content=str(prompt_text)),))

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "tools": [dict(tool) for tool in self.tools],
            "documents": [dict(document) for document in self.documents],
        }

    def to_legacy_user_prompt(self) -> str:
        if self.tools or self.documents or len(self.messages) != 1:
            raise ModelIOContractError(
                "structured conversation cannot be flattened to a legacy prompt"
            )
        message = self.messages[0]
        if message.role != "user" or message.name is not None:
            raise ModelIOContractError("legacy prompt conversion requires one unnamed user message")
        return message.content


__all__ = [
    "ChatMessage",
    "ChatRole",
    "ConversationIR",
    "ModelIOContractError",
]
