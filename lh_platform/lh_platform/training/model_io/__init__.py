"""Checkpoint-native model I/O contracts shared by train and rollout."""

from lh_platform.training.model_io.audit import (
    audit_model_io_fixtures,
    model_io_golden_projection,
    verify_model_io_golden,
)
from lh_platform.training.model_io.backend_parity import (
    MODEL_IO_BACKEND_PARITY_SCHEMA,
    ModelIOBackendSampleProjection,
    build_model_io_backend_parity_receipt,
    project_model_io_sample,
    require_model_io_backend_parity,
)
from lh_platform.training.model_io.contracts import (
    AssistantTarget,
    ResponseContract,
    ResponseContractDescriptor,
)
from lh_platform.training.model_io.messages import (
    ChatMessage,
    ConversationIR,
    ModelIOContractError,
)
from lh_platform.training.model_io.profiles import (
    ModelIOProfile,
    gemma3_it_profile,
    model_io_profile_by_id,
    qwen2_5_instruct_profile,
    qwen2_5_prompt_reasoning_profile,
    qwen3_2504_profile,
    qwen3_instruct_2507_profile,
    qwen3_thinking_2507_profile,
)
from lh_platform.training.model_io.receipts import ModelIOAuditFixture
from lh_platform.training.model_io.renderer import (
    NativeChatRenderer,
    RenderedAssistantTurn,
    RenderedPrompt,
)

__all__ = [
    "AssistantTarget",
    "MODEL_IO_BACKEND_PARITY_SCHEMA",
    "ResponseContract",
    "ResponseContractDescriptor",
    "ChatMessage",
    "ConversationIR",
    "ModelIOContractError",
    "ModelIOAuditFixture",
    "ModelIOBackendSampleProjection",
    "ModelIOProfile",
    "NativeChatRenderer",
    "RenderedAssistantTurn",
    "RenderedPrompt",
    "audit_model_io_fixtures",
    "build_model_io_backend_parity_receipt",
    "model_io_golden_projection",
    "gemma3_it_profile",
    "model_io_profile_by_id",
    "qwen2_5_instruct_profile",
    "qwen2_5_prompt_reasoning_profile",
    "qwen3_2504_profile",
    "qwen3_instruct_2507_profile",
    "qwen3_thinking_2507_profile",
    "project_model_io_sample",
    "require_model_io_backend_parity",
    "verify_model_io_golden",
]
