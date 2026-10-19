"""Portable inputs for no-GPU model I/O audit receipts."""

from __future__ import annotations

from dataclasses import dataclass

from lh_platform.training.model_io.contracts import AssistantTarget
from lh_platform.training.model_io.messages import ConversationIR, ModelIOContractError


@dataclass(frozen=True)
class ModelIOAuditFixture:
    fixture_id: str
    conversation: ConversationIR
    target: AssistantTarget

    def __post_init__(self) -> None:
        if not self.fixture_id:
            raise ModelIOContractError("model I/O audit fixture ID is empty")


__all__ = ["ModelIOAuditFixture"]
