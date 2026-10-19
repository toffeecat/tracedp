"""Exact-token, train-only supervision for actor competence warm-starts.

This module is intentionally separate from :mod:`experience_bridge`: supervised
protocol/API competence is not an RL advantage and must never be represented as
a synthetic policy-gradient record.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from lh_platform.training.model_io import (
    AssistantTarget,
    ConversationIR,
    NativeChatRenderer,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCES = {
    "alfworld_environment_expert_plan": "alfworld",
    "appworld_official_compiled_solution": "appworld",
    "appworld_public_api_descriptions_query": "appworld",
    "appworld_public_api_doc_query": "appworld",
    "appworld_doc_grounded_compiled_solution": "appworld",
    "appworld_doc_grounded_solution_prefix": "appworld",
    "appworld_train_evaluator_progress_action": "appworld",
    "appworld_train_sequential_solution_action": "appworld",
}
_FORBIDDEN_MARKERS = (
    "<goal_awareness>",
    "residual_goal",
    "frontier_signature",
    "active_frontier",
    "goal_graph",
    "contract_sha256",
    "environment_facts",
    "episode_reward",
    "step_advantage",
)


class SupervisedExperienceError(ValueError):
    """Raised when warm-start supervision could leak experiment-only state."""


@dataclass(frozen=True)
class SupervisedTextExample:
    """Ephemeral source text before exact-token materialization."""

    example_uid: str
    benchmark: str
    task_id: str
    dataset_split: str
    supervision_source: str
    prompt_text: str
    response_text: str

    def __post_init__(self) -> None:
        _validate_identity(
            example_uid=self.example_uid,
            benchmark=self.benchmark,
            task_id=self.task_id,
            dataset_split=self.dataset_split,
            supervision_source=self.supervision_source,
        )
        if not self.prompt_text.strip() or not self.response_text.strip():
            raise SupervisedExperienceError("supervised prompt/response text is empty")
        combined = f"{self.prompt_text}\n{self.response_text}".lower()
        leaked = tuple(marker for marker in _FORBIDDEN_MARKERS if marker in combined)
        if leaked:
            raise SupervisedExperienceError(
                "supervised text contains forbidden marker(s): " + ", ".join(leaked)
            )


@dataclass(frozen=True)
class SupervisedExperienceRecord:
    """Token-identity record consumed by the local supervised optimizer."""

    example_uid: str
    benchmark: str
    task_id: str
    dataset_split: str
    supervision_source: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    prompt_text_sha256: str
    response_text_sha256: str
    normalized_messages_sha256: str
    chat_template_sha256: str
    model_io_profile_sha256: str
    native_turn_end_ids: tuple[int, ...]
    assistant_content_token_count: int
    privileged_state_fields: tuple[str, ...] = ()
    schema: str = "lh_supervised_experience/v2"

    def __post_init__(self) -> None:
        if self.schema != "lh_supervised_experience/v2":
            raise SupervisedExperienceError("unsupported supervised experience schema")
        _validate_identity(
            example_uid=self.example_uid,
            benchmark=self.benchmark,
            task_id=self.task_id,
            dataset_split=self.dataset_split,
            supervision_source=self.supervision_source,
        )
        if not self.prompt_token_ids:
            raise SupervisedExperienceError("supervised record has no prompt tokens")
        if not self.response_token_ids:
            raise SupervisedExperienceError("supervised record has no response tokens")
        if not self.native_turn_end_ids:
            raise SupervisedExperienceError("supervised record has no native turn end")
        if any(
            isinstance(value, bool) or int(value) < 0
            for value in (
                *self.prompt_token_ids,
                *self.response_token_ids,
                *self.native_turn_end_ids,
            )
        ):
            raise SupervisedExperienceError("supervised token IDs must be non-negative ints")
        hashes = (
            self.prompt_text_sha256,
            self.response_text_sha256,
            self.normalized_messages_sha256,
            self.chat_template_sha256,
            self.model_io_profile_sha256,
        )
        if any(not _SHA256.fullmatch(value) for value in hashes):
            raise SupervisedExperienceError("supervised text hashes must be SHA256")
        if self.response_token_ids[-len(self.native_turn_end_ids) :] != (self.native_turn_end_ids):
            raise SupervisedExperienceError(
                "supervised response lacks its native assistant turn end"
            )
        expected_content_tokens = len(self.response_token_ids) - len(self.native_turn_end_ids)
        if (
            isinstance(self.assistant_content_token_count, bool)
            or self.assistant_content_token_count < 1
            or self.assistant_content_token_count != expected_content_tokens
        ):
            raise SupervisedExperienceError(
                "assistant content token count differs from native turn suffix"
            )
        if self.privileged_state_fields:
            raise SupervisedExperienceError(
                "supervised record contains privileged state fields: "
                + ", ".join(self.privileged_state_fields)
            )

    @property
    def active_token_count(self) -> int:
        return len(self.response_token_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize_supervised_example(
    example: SupervisedTextExample,
    *,
    renderer: NativeChatRenderer,
    conversation: ConversationIR,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> SupervisedExperienceRecord:
    """Materialize the checkpoint-native complete assistant-turn suffix."""

    if conversation != ConversationIR.from_legacy_user_prompt(example.prompt_text):
        raise SupervisedExperienceError(
            "supervised conversation differs from the declared legacy prompt"
        )
    turn = renderer.render_assistant_turn(
        conversation,
        AssistantTarget(content=example.response_text),
    )
    if len(turn.prompt_ids) > max_prompt_tokens:
        raise SupervisedExperienceError(
            f"supervised prompt exceeds token budget: {len(turn.prompt_ids)} > {max_prompt_tokens}"
        )
    if len(turn.response_ids) > max_response_tokens:
        raise SupervisedExperienceError(
            "supervised response exceeds token budget: "
            f"{len(turn.response_ids)} > {max_response_tokens}"
        )
    return SupervisedExperienceRecord(
        example_uid=example.example_uid,
        benchmark=example.benchmark,
        task_id=example.task_id,
        dataset_split=example.dataset_split,
        supervision_source=example.supervision_source,
        prompt_token_ids=turn.prompt_ids,
        response_token_ids=turn.response_ids,
        prompt_text_sha256=_text_sha256(example.prompt_text),
        response_text_sha256=_text_sha256(example.response_text),
        normalized_messages_sha256=turn.normalized_messages_sha256,
        chat_template_sha256=turn.chat_template_sha256,
        model_io_profile_sha256=turn.model_io_profile_sha256,
        native_turn_end_ids=turn.native_turn_end_ids,
        assistant_content_token_count=turn.assistant_content_token_count,
    )


def validate_supervised_experience_records(
    records: Iterable[SupervisedExperienceRecord],
) -> tuple[SupervisedExperienceRecord, ...]:
    normalized = tuple(records)
    if not normalized:
        raise SupervisedExperienceError("supervised batch is empty")
    seen: set[str] = set()
    for record in normalized:
        if not isinstance(record, SupervisedExperienceRecord):
            raise SupervisedExperienceError("supervised batch contains an invalid record")
        if record.example_uid in seen:
            raise SupervisedExperienceError(
                f"supervised batch contains duplicate example: {record.example_uid}"
            )
        seen.add(record.example_uid)
    return normalized


def assert_disjoint_task_ids(
    *,
    benchmark: str,
    train_task_ids: Iterable[str],
    eval_task_ids: Iterable[str],
) -> None:
    overlap = sorted(set(train_task_ids).intersection(eval_task_ids))
    if overlap:
        raise SupervisedExperienceError(
            f"{benchmark} train/eval task ID overlap: " + ", ".join(overlap[:10])
        )


def _validate_identity(
    *,
    example_uid: str,
    benchmark: str,
    task_id: str,
    dataset_split: str,
    supervision_source: str,
) -> None:
    if not example_uid or not task_id:
        raise SupervisedExperienceError("supervised example identity is empty")
    if dataset_split != "train":
        raise SupervisedExperienceError("actor warm-start supervision must use train split")
    expected = _SOURCES.get(supervision_source)
    if expected is None or benchmark != expected:
        raise SupervisedExperienceError("supervision source does not match benchmark")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "SupervisedExperienceError",
    "SupervisedExperienceRecord",
    "SupervisedTextExample",
    "assert_disjoint_task_ids",
    "tokenize_supervised_example",
    "validate_supervised_experience_records",
]
