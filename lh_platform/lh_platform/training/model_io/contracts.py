"""Generic response-target contracts shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from lh_platform.training.model_io.fingerprints import canonical_sha256
from lh_platform.training.model_io.messages import ChatMessage, ModelIOContractError
from lh_platform.training.model_io.profiles import ModelIOProfile


@dataclass(frozen=True)
class AssistantTarget:
    """One semantic assistant turn before checkpoint-native serialization."""

    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ModelIOContractError("assistant target is empty")

    def to_message(self) -> ChatMessage:
        return ChatMessage(role="assistant", content=self.content, name=self.name)


@dataclass(frozen=True)
class ResponseContractDescriptor:
    """Stable experiment identity for one benchmark response language."""

    contract_id: str
    reasoning_requirement: str
    serializer_id: str
    parser_id: str
    stop_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(
            value
            for value in (
                self.contract_id,
                self.reasoning_requirement,
                self.serializer_id,
                self.parser_id,
            )
        ):
            raise ModelIOContractError("response contract descriptor is incomplete")
        if any(not value for value in self.stop_texts):
            raise ModelIOContractError("response contract stop text is empty")
        if len(set(self.stop_texts)) != len(self.stop_texts):
            raise ModelIOContractError("response contract stop texts are not unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "reasoning_requirement": self.reasoning_requirement,
            "serializer_id": self.serializer_id,
            "parser_id": self.parser_id,
            "stop_texts": list(self.stop_texts),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


class ResponseContract(Protocol):
    """One owner for task instructions, SFT targets, stops, and parsing."""

    descriptor: ResponseContractDescriptor

    @property
    def id(self) -> str: ...

    @property
    def stop_texts(self) -> tuple[str, ...]: ...

    def render_instructions(self) -> str: ...

    def serialize_supervision(self, semantic_action: Any) -> AssistantTarget: ...

    def parse_response(self, response_text: str, **context: Any) -> Any: ...

    def validate_model_profile(self, profile: ModelIOProfile) -> None: ...


__all__ = ["AssistantTarget", "ResponseContract", "ResponseContractDescriptor"]
