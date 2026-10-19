"""ToolBench ReAct actor protocol: prompt rendering and Call/Finish response contract."""

# Portions reproduce the ToolBench Call/Finish grammar (Apache-2.0), https://github.com/OpenBMB/ToolBench. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lh_platform.training.model_io import (
    AssistantTarget,
    ModelIOContractError,
    ModelIOProfile,
    ResponseContractDescriptor,
)

_CALL_RE = re.compile(
    r"Call\s*\(\s*api_name\s*=\s*['\"]?([A-Za-z0-9_.]+)['\"]?",
    re.IGNORECASE,
)
_FINISH_RE = re.compile(
    r"Finish\s*\(\s*(.*?)\s*\)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_PARAM_JSON_RE = re.compile(
    r"parameters\s*=\s*(\{.*?\})\s*\)",
    re.IGNORECASE | re.DOTALL,
)

TOOLBENCH_SYSTEM_TEMPLATE = """\
You are a tool-using agent that solves tasks by calling APIs.

Available functions:
{function_vocabulary}

Rules:
- Output exactly ONE action per turn.
- Call(api_name=<exact function name>, parameters={{...}})
- Finish({{"return_type":"give_answer","final_answer":"..."}}) when you have the answer.
- Finish({{"return_type":"give_up_and_restart"}}) if you cannot solve it.
- Maximum {max_steps} steps allowed.
"""


@dataclass(frozen=True)
class ParsedToolBenchAction:
    name: str
    valid: bool
    api_name: str = ""
    parameters: dict[str, Any] | None = None
    return_type: str = ""
    final_answer: str = ""
    finish_error: str = ""
    error: str | None = None
    raw: str = ""


def _parse_upstream_finish_arguments(action_input: str) -> dict[str, Any]:
    try:
        decoded = json.loads(action_input, strict=False)
    except (TypeError, json.JSONDecodeError):
        decoded = {}
        if '"return_type"' in action_input:
            for rt in ("give_answer", "give_up_and_restart"):
                if f'"return_type": "{rt}"' in action_input or f'"return_type":"{rt}"' in action_input:
                    decoded["return_type"] = rt
                    break
        if '"final_answer"' in action_input:
            match = re.search(r'"final_answer"\s*:\s*"(.*?)"', action_input, re.DOTALL)
            if match:
                decoded["final_answer"] = match.group(1)
    if not isinstance(decoded, dict):
        decoded = {}
    return decoded


def render_function_vocabulary(available_tools: list[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for tool in available_tools:
        name = str(tool.get("name", ""))
        desc = str(tool.get("description", ""))[:120]
        params = tool.get("parameters", {})
        properties = params.get("properties", {}) if isinstance(params, Mapping) else {}
        required = params.get("required", []) if isinstance(params, Mapping) else []
        param_parts: list[str] = []
        for pname, pschema in properties.items():
            ptype = pschema.get("type", "string") if isinstance(pschema, Mapping) else "string"
            req_mark = " (required)" if pname in required else ""
            param_parts.append(f"{pname}: {ptype}{req_mark}")
        param_str = ", ".join(param_parts) if param_parts else "none"
        lines.append(f"- {name}({param_str}): {desc}")
    lines.append('- Finish({"return_type":"give_answer","final_answer":"..."}): give final answer')
    lines.append('- Finish({"return_type":"give_up_and_restart"}): give up')
    return "\n".join(lines)


def render_toolbench_prompt(
    *,
    query: str,
    available_tools: list[Mapping[str, Any]],
    history: list[Mapping[str, str]],
    step_index: int,
    max_steps: int,
) -> str:
    vocabulary = render_function_vocabulary(available_tools)
    system = TOOLBENCH_SYSTEM_TEMPLATE.format(
        function_vocabulary=vocabulary,
        max_steps=max_steps,
    )
    parts = [system, f"Task: {query}\n"]
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "action":
            parts.append(f"Action: {content}")
        elif role == "observation":
            parts.append(f"Observation: {content}")
    parts.append(f"Step {step_index + 1}/{max_steps}. Output your next action:")
    return "\n".join(parts)


@dataclass(frozen=True)
class ToolBenchCallFinishContract:
    """Response contract for ToolBench's Call/Finish action grammar."""

    descriptor: ResponseContractDescriptor = ResponseContractDescriptor(
        contract_id="toolbench_call_finish/v1",
        reasoning_requirement="none",
        serializer_id="toolbench_call_finish_serializer_v1",
        parser_id="toolbench_call_finish_parser_v1",
        stop_texts=("\nObservation:", "\nAction:", "\nStep "),
    )

    @property
    def id(self) -> str:
        return self.descriptor.contract_id

    @property
    def stop_texts(self) -> tuple[str, ...]:
        return self.descriptor.stop_texts

    def parse_response(self, response_text: str, **_context: object) -> ParsedToolBenchAction:
        text = str(response_text).strip()
        if not text:
            return ParsedToolBenchAction(
                name="invalid", valid=False, error="empty response", raw=text
            )

        finish_match = _FINISH_RE.search(text)
        if finish_match:
            finish_input = finish_match.group(1).strip()
            if not finish_input:
                return ParsedToolBenchAction(
                    name="finish",
                    valid=False,
                    error="Finish requires non-empty arguments",
                    raw=text,
                )
            if finish_input.startswith("{") or '"return_type"' in finish_input:
                parsed = _parse_upstream_finish_arguments(finish_input)
                return_type = str(parsed.get("return_type", "")).strip()
                if not return_type:
                    return ParsedToolBenchAction(
                        name="finish",
                        valid=False,
                        finish_error='must have "return_type"',
                        error='Finish missing "return_type"',
                        raw=text,
                    )
                if return_type not in {"give_answer", "give_up_and_restart"}:
                    return ParsedToolBenchAction(
                        name="finish",
                        valid=False,
                        return_type=return_type,
                        finish_error='"return_type" is not a valid choice',
                        error=f"invalid return_type: {return_type}",
                        raw=text,
                    )
                final_answer = str(parsed.get("final_answer", ""))
                if return_type == "give_answer" and "final_answer" not in parsed:
                    return ParsedToolBenchAction(
                        name="finish",
                        valid=False,
                        return_type=return_type,
                        finish_error='must have "final_answer"',
                        error='Finish missing "final_answer"',
                        raw=text,
                    )
                return ParsedToolBenchAction(
                    name="finish",
                    valid=True,
                    return_type=return_type,
                    final_answer=final_answer,
                    raw=text,
                )
            return ParsedToolBenchAction(
                name="finish",
                valid=True,
                return_type="give_answer",
                final_answer=finish_input,
                raw=text,
            )

        call_match = _CALL_RE.search(text)
        if call_match:
            api_name = call_match.group(1).strip()
            params_match = _PARAM_JSON_RE.search(text)
            parameters: dict[str, Any] = {}
            if params_match:
                try:
                    parameters = json.loads(params_match.group(1))
                except (json.JSONDecodeError, TypeError) as exc:
                    return ParsedToolBenchAction(
                        name="call",
                        valid=False,
                        api_name=api_name,
                        error=f"parameters must be valid JSON: {exc}",
                        raw=text,
                    )
            return ParsedToolBenchAction(
                name="call",
                valid=True,
                api_name=api_name,
                parameters=parameters,
                raw=text,
            )

        return ParsedToolBenchAction(
            name="invalid",
            valid=False,
            error="response does not contain Call(...) or Finish(...)",
            raw=text,
        )

    def serialize_supervision(self, semantic_action: object) -> AssistantTarget:
        text = str(semantic_action).strip()
        if not text:
            raise ModelIOContractError("ToolBench supervision action is empty")
        return AssistantTarget(text)

    def validate_model_profile(self, profile: ModelIOProfile) -> None:
        if profile.selected_reasoning_mode != "off":
            raise ModelIOContractError(
                "ToolBench Call/Finish contract requires reasoning mode off"
            )


TOOLBENCH_CALL_FINISH_CONTRACT = ToolBenchCallFinishContract()
