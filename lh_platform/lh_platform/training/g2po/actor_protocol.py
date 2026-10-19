"""Source-aligned ALFWorld actor protocol for the pinned G2PO runtime."""

# Portions reproduce ALFWorld prompt templates from verl-agent (Apache-2.0), https://github.com/langfengQ/verl-agent. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lh_platform.training.model_io import (
    AssistantTarget,
    ModelIOContractError,
    ModelIOProfile,
    ResponseContractDescriptor,
)

_INITIAL_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.\x20
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

_HISTORY_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.\x20
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

_ACTION_ONLY_OUTPUT_CONTRACT = """
Now it's your turn to take an action.
Choose exactly one admissible action for the current step and present it within <action> </action> tags. Do not add an explanation.
"""

_MENU_INDEX_OUTPUT_CONTRACT = """
Now it's your turn to take an action.
Choose exactly one numbered admissible action. Return only its one-based index within <action_index> </action_index> tags.
"""

_BARE_MENU_INDEX_OUTPUT_CONTRACT = """
Now it's your turn to take an action.
Choose exactly one numbered admissible action. Return only the integer representing its one-based index. Do not add tags, punctuation, or an explanation.
"""

_INITIAL_PROMPT_NO_ACTIONS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think>  </think> tags.\x20
Once you've finished your reasoning, you should choose an action for current step and present it within <action> </action> tags.
"""

_HISTORY_PROMPT_NO_ACTIONS = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think>  </think> tags.\x20
Once you've finished your reasoning, you should choose an action for current step and present it within <action> </action> tags.
"""

_OFFICIAL_G2PO_OUTPUT_CONTRACT = """
Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.\x20
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

_PROMPT_INJECTED_THINKING_OUTPUT_CONTRACT = """
Now it's your turn to take an action.
The model chat template has already opened the reasoning block. Continue reasoning about the current situation, then close it with </think>.
Once you've finished your reasoning, choose an admissible action for the current step and present it within <action> </action> tags. Do not generate another <think> tag.
"""

_TASK_MARKER = "Your task is to: "
_OUTPUT_CONTRACT_MARKER = "\nNow it's your turn to take an action."
_INVISIBLE_OBSERVATION_PREFIXES = ("You heat", "You cool", "You clean", "You turn on")
CONTEXT_EXTENSION_PLACEMENT_REVISION = "before_source_output_contract/v1"
G2PO_ALFWORLD_ACTOR_PROTOCOL_ID = "official_g2po_alfworld_actor_b8ffaf56"
MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID = (
    "model_compatible_non_thinking_alfworld_actor_v1"
)
MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_V2_ID = (
    "model_compatible_non_thinking_alfworld_actor_v2"
)
UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID = (
    "unique_action_tag_non_thinking_alfworld_actor_v2"
)
UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID = (
    "unique_action_tag_non_thinking_alfworld_actor/v1"
)
MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID = (
    "model_compatible_non_thinking_alfworld_actor/v1"
)
MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID = "menu_index_non_thinking_alfworld_actor_v1"
MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID = "menu_index_non_thinking_alfworld_actor/v1"
BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID = (
    "bare_menu_index_non_thinking_alfworld_actor_v1"
)
BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID = (
    "bare_menu_index_non_thinking_alfworld_actor/v1"
)
NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID = (
    "native_menu_choice_non_thinking_alfworld_actor_v1"
)
NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID = (
    "native_menu_choice_non_thinking_alfworld_actor/v1"
)
MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_IDS = frozenset(
    {
        MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
        MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_V2_ID,
    }
)
QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID = (
    "qwen_prompt_injected_thinking_alfworld_actor_v1"
)
QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_FAMILY_ID = (
    "qwen_prompt_injected_thinking_alfworld_actor/v1"
)
SUPPORTED_ALFWORLD_ACTOR_PROTOCOL_IDS = frozenset(
    {
        G2PO_ALFWORLD_ACTOR_PROTOCOL_ID,
        *MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_IDS,
        UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
        MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
        BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
        NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
        QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID,
    }
)

__all__ = [
    "CONTEXT_EXTENSION_PLACEMENT_REVISION",
    "BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID",
    "BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID",
    "G2PO_ALFWORLD_ACTOR_PROTOCOL_ID",
    "MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID",
    "MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID",
    "MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_IDS",
    "MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_V2_ID",
    "MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID",
    "MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID",
    "NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID",
    "NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID",
    "QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_FAMILY_ID",
    "QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID",
    "SUPPORTED_ALFWORLD_ACTOR_PROTOCOL_IDS",
    "UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID",
    "UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID",
    "alfworld_actor_protocol_semantic_family_id",
    "build_alfworld_actor_protocol",
    "extract_alfworld_task_description",
    "unique_action_tag_parser_sha256",
    "parse_unique_action_tag_response",
    "G2POAnchorTracker",
    "g2po_anchor_tracker_source_sha256",
    "G2POActorHistoryStep",
    "G2POActionProjection",
    "AlfWorldResponseContract",
    "G2POAlfWorldActorProtocol",
]


def alfworld_actor_protocol_semantic_family_id(protocol_id: str) -> str:
    """Return the runtime semantics independently of config-schema revision."""

    if protocol_id == G2PO_ALFWORLD_ACTOR_PROTOCOL_ID:
        return G2PO_ALFWORLD_ACTOR_PROTOCOL_ID
    if protocol_id in MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_IDS:
        return MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID
    if protocol_id == UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
        return UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID
    if protocol_id == MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
        return MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID
    if protocol_id == BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
        return BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID
    if protocol_id == NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
        return NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID
    if protocol_id == QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
        return QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_FAMILY_ID
    raise ValueError(f"unsupported ALFWorld actor protocol: {protocol_id}")


def extract_alfworld_task_description(initial_observation: str) -> str:
    start = str(initial_observation).find(_TASK_MARKER)
    if start == -1:
        raise ValueError("ALFWorld reset observation lacks its task description")
    return str(initial_observation)[start + len(_TASK_MARKER) :].strip()


class G2POAnchorTracker:
    """Reproduce the pinned manager's post-step graph-anchor projection."""

    def __init__(self, *, initial_actor_prompt: str, show_admissible_actions: bool = True) -> None:
        self._history = [str(initial_actor_prompt)]
        self._show_admissible_actions = show_admissible_actions

    def project_next(
        self,
        *,
        raw_observation: str,
        admissible_actions: Sequence[str],
    ) -> str:
        if self._show_admissible_actions:
            actions = "\n ".join(f"'{action}'" for action in admissible_actions if action != "help")
            projected = f"{raw_observation}\nAdmissible actions:\n {actions}\n"
        else:
            projected = f"{raw_observation}\n"
        if not str(raw_observation).startswith("Nothing happens"):
            self._history.append(projected)
        invisible = len(self._history) >= 2 and self._history[-2].startswith(
            _INVISIBLE_OBSERVATION_PREFIXES
        )
        if invisible:
            return " ".join(self._history[-2:])
        return self._history[-1]


def g2po_anchor_tracker_source_sha256() -> str:
    """Bind collected anchors to the exact local source implementation."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class G2POActorHistoryStep:
    observation: str
    environment_action: str


@dataclass(frozen=True)
class G2POActionProjection:
    model_response: str
    environment_action: str
    is_action_valid: bool
    invalid_reasons: tuple[str, ...]


# Unique-action-tag (ALFWorld-v2) parser grammar. It reuses the exact v1
# menu-index tag grammar (case-insensitive, whitespace tolerant, plain digits)
# and only changes the extraction scope: the whole response is searched and
# exactly one well-formed tag must exist. Any tag-like marker that is not part
# of a well-formed tag makes the response invalid.
_UNIQUE_ACTION_TAG_RE = re.compile(r"<action_index>\s*(\d+)\s*</action_index>", flags=re.IGNORECASE)
_ACTION_INDEX_OPEN_MARKER_RE = re.compile(r"<action_index", flags=re.IGNORECASE)
_ACTION_INDEX_CLOSE_MARKER_RE = re.compile(r"</action_index>", flags=re.IGNORECASE)


def _unique_action_tag_scan(response_text: str) -> tuple[list[re.Match[str]], bool]:
    """Return well-formed tag matches and whether every tag marker is consumed.

    A marker that is not covered by a well-formed tag match is a malformed or
    partial tag; the second returned value is False whenever one exists.
    """

    matches = list(_UNIQUE_ACTION_TAG_RE.finditer(response_text))
    spans = tuple((match.start(), match.end()) for match in matches)

    def covered(position: int) -> bool:
        return any(start <= position < end for start, end in spans)

    for marker in _ACTION_INDEX_OPEN_MARKER_RE.finditer(response_text):
        if not covered(marker.start()):
            return matches, False
    for marker in _ACTION_INDEX_CLOSE_MARKER_RE.finditer(response_text):
        if not covered(marker.start()):
            return matches, False
    return matches, True


def parse_unique_action_tag_response(
    response_text: str,
    *,
    admissible_actions: Sequence[str] = (),
) -> G2POActionProjection:
    """Apply the ALFWorld-v2 unique-action-tag contract to one raw response.

    Acceptance rule: exactly one well-formed ``<action_index>N</action_index>``
    tag anywhere in the response, with N inside the displayed menu range.
    Surrounding planning prose is allowed and never participates in action
    extraction; the full raw response is preserved on every projection.
    """

    original = str(response_text)
    actions = tuple(action for action in admissible_actions if action != "help")
    matches, all_markers_consumed = _unique_action_tag_scan(original)
    if not all_markers_consumed:
        return G2POActionProjection(
            model_response=original,
            environment_action="",
            is_action_valid=False,
            invalid_reasons=("malformed_action_index_tag",),
        )
    if not matches:
        return G2POActionProjection(
            model_response=original,
            environment_action="",
            is_action_valid=False,
            invalid_reasons=("missing_action_index_tag",),
        )
    if len(matches) > 1:
        return G2POActionProjection(
            model_response=original,
            environment_action="",
            is_action_valid=False,
            invalid_reasons=("multiple_action_index_tags",),
        )
    index = int(matches[0].group(1))
    if index < 1 or index > len(actions):
        return G2POActionProjection(
            model_response=original,
            environment_action="",
            is_action_valid=False,
            invalid_reasons=("action_index_out_of_range",),
        )
    return G2POActionProjection(
        model_response=original,
        environment_action=actions[index - 1],
        is_action_valid=True,
        invalid_reasons=(),
    )


def unique_action_tag_parser_sha256() -> str:
    """Bind the v2 parser identity to every behavior-determining source element.

    The digest covers the entry-point function source, the marker-coverage
    helper source, and the three registered tag patterns with their flags, so a
    change to any element that can alter parse decisions changes the identity.
    """

    parts = (
        inspect.getsource(_unique_action_tag_scan),
        inspect.getsource(parse_unique_action_tag_response),
        _UNIQUE_ACTION_TAG_RE.pattern,
        str(_UNIQUE_ACTION_TAG_RE.flags),
        _ACTION_INDEX_OPEN_MARKER_RE.pattern,
        str(_ACTION_INDEX_OPEN_MARKER_RE.flags),
        _ACTION_INDEX_CLOSE_MARKER_RE.pattern,
        str(_ACTION_INDEX_CLOSE_MARKER_RE.flags),
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AlfWorldResponseContract:
    """Own the ALFWorld output language across SFT and rollout."""

    reasoning_requirement: str
    action_representation: str = "text"

    def __post_init__(self) -> None:
        if self.reasoning_requirement not in {
            "paired_in_generated_response",
            "closing_only_in_generated_response",
            "none",
        }:
            raise ModelIOContractError(
                f"unsupported ALFWorld reasoning response requirement: {self.reasoning_requirement}"
            )
        if self.action_representation not in {
            "text",
            "menu_index",
            "bare_menu_index",
            "native_menu_choice",
            "unique_action_tag",
        }:
            raise ModelIOContractError(
                f"unsupported ALFWorld action representation: {self.action_representation}"
            )
        if (
            self.action_representation
            in {"menu_index", "bare_menu_index", "native_menu_choice", "unique_action_tag"}
            and self.reasoning_requirement != "none"
        ):
            raise ModelIOContractError("menu-index actions require non-thinking inference")

    @classmethod
    def official_g2po(cls) -> AlfWorldResponseContract:
        return cls(reasoning_requirement="paired_in_generated_response")

    @classmethod
    def action_only(cls) -> AlfWorldResponseContract:
        return cls(reasoning_requirement="none")

    @classmethod
    def menu_index(cls) -> AlfWorldResponseContract:
        return cls(reasoning_requirement="none", action_representation="menu_index")

    @classmethod
    def unique_action_tag(cls) -> AlfWorldResponseContract:
        return cls(reasoning_requirement="none", action_representation="unique_action_tag")

    @classmethod
    def bare_menu_index(cls) -> AlfWorldResponseContract:
        return cls(reasoning_requirement="none", action_representation="bare_menu_index")

    @classmethod
    def native_menu_choice(cls) -> AlfWorldResponseContract:
        return cls(reasoning_requirement="none", action_representation="native_menu_choice")

    @classmethod
    def prompt_injected_thinking(cls) -> AlfWorldResponseContract:
        return cls(reasoning_requirement="closing_only_in_generated_response")

    @property
    def require_think_tags(self) -> bool:
        """Compatibility view for the source G2PO paired-tag contract."""

        return self.reasoning_requirement == "paired_in_generated_response"

    @property
    def descriptor(self) -> ResponseContractDescriptor:
        if self.action_representation == "unique_action_tag":
            return ResponseContractDescriptor(
                contract_id="alfworld_unique_action_tag_only/v1",
                reasoning_requirement="none",
                serializer_id="alfworld_unique_action_tag_serializer_v1",
                parser_id="alfworld_unique_action_tag_parser_v1",
                stop_texts=("</action_index>",),
            )
        if self.action_representation == "menu_index":
            return ResponseContractDescriptor(
                contract_id="alfworld_menu_index_only/v1",
                reasoning_requirement="none",
                serializer_id="alfworld_menu_index_serializer_v1",
                parser_id="alfworld_menu_index_parser_v1",
                stop_texts=("</action_index>",),
            )
        if self.action_representation == "bare_menu_index":
            return ResponseContractDescriptor(
                contract_id="alfworld_bare_menu_index_only/v1",
                reasoning_requirement="none",
                serializer_id="alfworld_bare_menu_index_serializer_v1",
                parser_id="alfworld_bare_menu_index_parser_v1",
                stop_texts=(),
            )
        if self.action_representation == "native_menu_choice":
            return ResponseContractDescriptor(
                contract_id="alfworld_native_menu_choice/v1",
                reasoning_requirement="none",
                serializer_id="alfworld_native_menu_choice_serializer_v1",
                parser_id="alfworld_native_menu_choice_parser_v1",
                stop_texts=(),
            )
        if self.reasoning_requirement == "paired_in_generated_response":
            return ResponseContractDescriptor(
                contract_id="official_g2po_think_action/v1",
                reasoning_requirement="paired_in_generated_response",
                serializer_id="official_g2po_think_action_serializer_v1",
                parser_id="official_g2po_think_action_parser_v1",
                stop_texts=(),
            )
        if self.reasoning_requirement == "closing_only_in_generated_response":
            return ResponseContractDescriptor(
                contract_id="qwen_prompt_injected_thinking_action/v1",
                reasoning_requirement="closing_only_in_generated_response",
                serializer_id="qwen_prompt_injected_thinking_action_serializer_v1",
                parser_id="qwen_prompt_injected_thinking_action_parser_v1",
                stop_texts=("</action>",),
            )
        return ResponseContractDescriptor(
            contract_id="alfworld_action_tag_only/v1",
            reasoning_requirement="none",
            serializer_id="alfworld_action_tag_serializer_v1",
            parser_id="alfworld_action_tag_parser_v1",
            stop_texts=("</action>",),
        )

    @property
    def id(self) -> str:
        return self.descriptor.contract_id

    @property
    def stop_texts(self) -> tuple[str, ...]:
        return self.descriptor.stop_texts

    def render_instructions(self) -> str:
        if self.action_representation in {"menu_index", "unique_action_tag"}:
            return _MENU_INDEX_OUTPUT_CONTRACT
        if self.action_representation == "bare_menu_index":
            return _BARE_MENU_INDEX_OUTPUT_CONTRACT
        if self.action_representation == "native_menu_choice":
            return (
                "Call the choose_action native tool exactly once with one integer "
                "index from the current admissible-action menu. Do not emit text."
            )
        if self.reasoning_requirement == "paired_in_generated_response":
            return _OFFICIAL_G2PO_OUTPUT_CONTRACT
        if self.reasoning_requirement == "closing_only_in_generated_response":
            return _PROMPT_INJECTED_THINKING_OUTPUT_CONTRACT
        return _ACTION_ONLY_OUTPUT_CONTRACT

    def serialize_supervision(self, semantic_action: object) -> AssistantTarget:
        action = str(semantic_action).strip()
        if not action:
            raise ModelIOContractError("ALFWorld supervision action is empty")
        if self.action_representation in {"menu_index", "unique_action_tag"}:
            if not action.isdigit() or int(action) < 1:
                raise ModelIOContractError("menu-index supervision requires a positive integer")
            return AssistantTarget(f"<action_index>{action}</action_index>")
        if self.action_representation == "bare_menu_index":
            if not action.isdigit() or int(action) < 1:
                raise ModelIOContractError("menu-index supervision requires a positive integer")
            return AssistantTarget(action)
        if self.action_representation == "native_menu_choice":
            if not action.isdigit() or int(action) < 1:
                raise ModelIOContractError("native menu choice requires a positive integer")
            return AssistantTarget(
                json.dumps(
                    {"tool": "choose_action", "arguments": {"index": int(action)}},
                    separators=(",", ":"),
                )
            )
        if self.reasoning_requirement == "paired_in_generated_response":
            return AssistantTarget(
                "<think>Select the admissible action that advances the task.</think>\n"
                f"<action>{action}</action>"
            )
        if self.reasoning_requirement == "closing_only_in_generated_response":
            return AssistantTarget(
                "Select the admissible action that advances the task.</think>\n"
                f"<action>{action}</action>"
            )
        return AssistantTarget(f"<action>{action}</action>")

    def parse_response(
        self,
        response_text: str,
        *,
        admissible_actions: Sequence[str] = (),
    ) -> G2POActionProjection:
        original = str(response_text)
        if self.action_representation == "unique_action_tag":
            return parse_unique_action_tag_response(
                original,
                admissible_actions=admissible_actions,
            )
        if self.action_representation == "menu_index":
            actions = tuple(action for action in admissible_actions if action != "help")
            match = re.fullmatch(
                r"\s*<action_index>\s*(\d+)\s*</action_index>\s*",
                original,
                flags=re.IGNORECASE,
            )
            if match is None:
                return G2POActionProjection(
                    model_response=original,
                    environment_action="",
                    is_action_valid=False,
                    invalid_reasons=("invalid_action_index_tags",),
                )
            index = int(match.group(1))
            if index < 1 or index > len(actions):
                return G2POActionProjection(
                    model_response=original,
                    environment_action="",
                    is_action_valid=False,
                    invalid_reasons=("action_index_out_of_range",),
                )
            return G2POActionProjection(
                model_response=original,
                environment_action=actions[index - 1],
                is_action_valid=True,
                invalid_reasons=(),
            )
        if self.action_representation == "bare_menu_index":
            actions = tuple(action for action in admissible_actions if action != "help")
            match = re.fullmatch(r"\s*(\d+)\s*", original)
            if match is None:
                return G2POActionProjection(
                    model_response=original,
                    environment_action="",
                    is_action_valid=False,
                    invalid_reasons=("invalid_bare_action_index",),
                )
            index = int(match.group(1))
            if index < 1 or index > len(actions):
                return G2POActionProjection(
                    model_response=original,
                    environment_action="",
                    is_action_valid=False,
                    invalid_reasons=("action_index_out_of_range",),
                )
            return G2POActionProjection(
                model_response=original,
                environment_action=actions[index - 1],
                is_action_valid=True,
                invalid_reasons=(),
            )
        if self.action_representation == "native_menu_choice":
            actions = tuple(action for action in admissible_actions if action != "help")
            try:
                payload = json.loads(original)
            except json.JSONDecodeError:
                payload = None
            if (
                not isinstance(payload, dict)
                or set(payload) != {"tool", "arguments"}
                or payload.get("tool") != "choose_action"
                or not isinstance(payload.get("arguments"), dict)
                or set(payload["arguments"]) != {"index"}
                or type(payload["arguments"]["index"]) is not int
            ):
                return G2POActionProjection(
                    model_response=original,
                    environment_action="",
                    is_action_valid=False,
                    invalid_reasons=("invalid_native_menu_choice",),
                )
            index = payload["arguments"]["index"]
            if index < 1 or index > len(actions):
                return G2POActionProjection(
                    model_response=original,
                    environment_action="",
                    is_action_valid=False,
                    invalid_reasons=("action_index_out_of_range",),
                )
            return G2POActionProjection(
                model_response=original,
                environment_action=actions[index - 1],
                is_action_valid=True,
                invalid_reasons=(),
            )

        lowered = original.lower()
        start_tag = "<action>"
        end_tag = "</action>"
        start = lowered.find(start_tag)
        end = lowered.find(end_tag)
        if start == -1 or end == -1:
            return G2POActionProjection(
                model_response=original,
                environment_action=lowered[-30:],
                is_action_valid=False,
                invalid_reasons=("missing_action_tags",),
            )

        environment_action = lowered[start + len(start_tag) : end].strip().lower()
        reasons: list[str] = []
        if self.reasoning_requirement == "paired_in_generated_response" and (
            original.find("<think>") == -1 or original.find("</think>") == -1
        ):
            reasons.append("missing_think_tags")
        if self.reasoning_requirement == "closing_only_in_generated_response":
            if original.find("<think>") != -1:
                reasons.append("unexpected_generated_think_start")
            if original.find("</think>") == -1:
                reasons.append("missing_generated_think_end")
        if re.search(r"[\u4e00-\u9fff]", original):
            reasons.append("contains_chinese_characters")
        return G2POActionProjection(
            model_response=original,
            environment_action=environment_action,
            is_action_valid=not reasons,
            invalid_reasons=tuple(reasons),
        )

    def validate_model_profile(self, profile: ModelIOProfile) -> None:
        if self.reasoning_requirement == "paired_in_generated_response":
            if profile.generated_reasoning_delimiters != "paired":
                raise ModelIOContractError(
                    "official G2PO response requires generated paired reasoning tags"
                )
            return
        if self.reasoning_requirement == "closing_only_in_generated_response":
            if (
                profile.selected_reasoning_mode != "on"
                or not profile.prompt_injects_reasoning_start
                or profile.generated_reasoning_delimiters != "closing_only"
            ):
                raise ModelIOContractError(
                    "prompt-injected thinking response requires a fixed-on model "
                    "profile with a template-injected opening delimiter and a "
                    "generated closing delimiter"
                )
            return
        if profile.selected_reasoning_mode != "off":
            raise ModelIOContractError("ALFWorld action-only response requires reasoning mode off")


@dataclass(frozen=True)
class G2POAlfWorldActorProtocol:
    """Render one registered ALFWorld actor contract.

    The default remains the source-exact G2PO contract.  The non-thinking
    profile deliberately changes only the response contract; environment
    context, history layout, action listing, and graph-anchor semantics remain
    source aligned.
    """

    history_length: int = 2
    protocol_id: str = G2PO_ALFWORLD_ACTOR_PROTOCOL_ID
    show_admissible_actions: bool = True

    def __post_init__(self) -> None:
        if self.history_length < 0:
            raise ValueError("G2PO actor history length must be non-negative")
        if self.protocol_id not in SUPPORTED_ALFWORLD_ACTOR_PROTOCOL_IDS:
            raise ValueError(f"unsupported ALFWorld actor protocol: {self.protocol_id}")

    @property
    def require_think_tags(self) -> bool:
        return self.protocol_id == G2PO_ALFWORLD_ACTOR_PROTOCOL_ID

    @property
    def response_contract(self) -> AlfWorldResponseContract:
        if self.require_think_tags:
            return AlfWorldResponseContract.official_g2po()
        if self.protocol_id == UNIQUE_ACTION_TAG_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
            return AlfWorldResponseContract.unique_action_tag()
        if self.protocol_id == MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
            return AlfWorldResponseContract.menu_index()
        if self.protocol_id == BARE_MENU_INDEX_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
            return AlfWorldResponseContract.bare_menu_index()
        if self.protocol_id == NATIVE_MENU_CHOICE_NON_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
            return AlfWorldResponseContract.native_menu_choice()
        if self.protocol_id == QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_PROTOCOL_ID:
            return AlfWorldResponseContract.prompt_injected_thinking()
        return AlfWorldResponseContract.action_only()

    def render_prompt(
        self,
        *,
        task_description: str,
        current_observation: str,
        admissible_actions: Sequence[str],
        history: Sequence[G2POActorHistoryStep],
        context_extension: str | None = None,
        step_count_override: int | None = None,
    ) -> str:
        if step_count_override is not None and (
            isinstance(step_count_override, bool) or step_count_override < len(history)
        ):
            raise ValueError("step_count_override must cover every supplied history step")
        visible_actions = tuple(action for action in admissible_actions if action != "help")
        if self.response_contract.action_representation in {
            "menu_index",
            "bare_menu_index",
            "native_menu_choice",
            "unique_action_tag",
        }:
            actions = "\n ".join(
                f"{index}: '{action}'" for index, action in enumerate(visible_actions, start=1)
            )
        else:
            actions = "\n ".join(f"'{action}'" for action in visible_actions)
        use_history_layout = bool(history) or step_count_override is not None
        if not use_history_layout or self.history_length == 0:
            if self.show_admissible_actions:
                prompt = _INITIAL_PROMPT.format(
                    current_observation=current_observation,
                    admissible_actions=actions,
                )
            else:
                prompt = _INITIAL_PROMPT_NO_ACTIONS.format(
                    current_observation=current_observation,
                )
        else:
            step_count = len(history) if step_count_override is None else step_count_override
            recent = tuple(history[-self.history_length :])
            start = step_count - len(recent) + 1
            action_history = "\n".join(
                "[Observation "
                f"{start + offset}: '{item.observation}', "
                f"Action {start + offset}: '{item.environment_action}']"
                for offset, item in enumerate(recent)
            )
            if self.show_admissible_actions:
                prompt = _HISTORY_PROMPT.format(
                    task_description=task_description,
                    step_count=step_count,
                    history_length=len(recent),
                    action_history=action_history,
                    current_step=step_count + 1,
                    current_observation=current_observation,
                    admissible_actions=actions,
                )
            else:
                prompt = _HISTORY_PROMPT_NO_ACTIONS.format(
                    task_description=task_description,
                    step_count=step_count,
                    history_length=len(recent),
                    action_history=action_history,
                    current_step=step_count + 1,
                    current_observation=current_observation,
                )
        if prompt.count(_OUTPUT_CONTRACT_MARKER) != 1:
            raise ValueError("ALFWorld prompt lacks a unique output-contract replacement point")
        prompt = prompt[: prompt.index(_OUTPUT_CONTRACT_MARKER)]
        prompt += self.response_contract.render_instructions()
        if not context_extension:
            return prompt
        if prompt.count(_OUTPUT_CONTRACT_MARKER) != 1:
            raise ValueError("G2PO prompt lacks a unique output-contract insertion point")
        return prompt.replace(
            _OUTPUT_CONTRACT_MARKER,
            f"\n{context_extension}\n{_OUTPUT_CONTRACT_MARKER}",
            1,
        )

    def project_response(
        self,
        model_response: str,
        *,
        admissible_actions: Sequence[str],
    ) -> G2POActionProjection:
        """Project model output with the declared protocol semantics.

        Text-action protocols preserve the pinned source semantics.  The
        menu-index protocol instead binds its declared index to the displayed
        menu, avoiding any text repair or synonym matching.
        """

        return self.response_contract.parse_response(
            model_response,
            admissible_actions=admissible_actions,
        )


def build_alfworld_actor_protocol(
    *,
    protocol_id: str,
    history_length: int,
    show_admissible_actions: bool = True,
) -> G2POAlfWorldActorProtocol:
    """Construct only registered actor protocols; never infer from model names."""

    return G2POAlfWorldActorProtocol(
        history_length=history_length,
        protocol_id=protocol_id,
        show_admissible_actions=show_admissible_actions,
    )
