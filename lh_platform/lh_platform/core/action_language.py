"""Action Language primitives.

An Action Language belongs to a benchmark. It parses model-emitted raw text
into a structured action before benchmark execution while keeping the raw text
as audit evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from lh_platform.core.types import Action, Task


@dataclass
class ParsedAction:
    """Structured result of applying a benchmark Action Language."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    valid: bool = True
    terminal: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> ParsedAction:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parsed action requires a non-empty name")
        if not isinstance(self.args, dict) or not isinstance(self.raw, str):
            raise ValueError("parsed action args/raw have invalid types")
        if not isinstance(self.valid, bool) or not isinstance(self.terminal, bool):
            raise ValueError("parsed action flags must be boolean")
        if self.valid and self.name == "invalid":
            raise ValueError("valid parsed action cannot use the invalid name")
        if not self.valid and not str(self.error or "").strip():
            raise ValueError("invalid parsed action requires an error reason")
        return self


class ActionLanguage:
    """Base Action Language interface used by benchmark modules."""

    name = "base"
    contract_id = ""

    def parse(self, action: Action) -> ParsedAction:
        raise NotImplementedError

    def prompt_vocabulary(self, task: Task) -> str:
        raise NotImplementedError

    def conformance_examples(self, task: Task) -> Iterable[str]:
        """Return parser-valid examples used to pin prompt/parser compatibility."""

        return ()

    def validate_contract(self, task: Task) -> None:
        if not str(self.contract_id).strip():
            raise ValueError("action language requires a stable contract_id")
        if not str(self.prompt_vocabulary(task)).strip():
            raise ValueError("action language requires a non-empty prompt vocabulary")
        examples = tuple(self.conformance_examples(task))
        if not examples:
            raise ValueError("action language requires conformance examples")
        for raw in examples:
            parsed = self.parse(Action(raw=raw)).validate()
            if not parsed.valid:
                raise ValueError(f"action-language example is rejected by its parser: {raw!r}")

    def annotate(self, action: Action, parsed: ParsedAction) -> None:
        """Attach parsed evidence to the original Action without dropping raw."""
        action.info.setdefault("original_action_type", action.action_type)
        action.action_type = parsed.name if parsed.valid else "invalid"
        action.tool_name = parsed.name if parsed.valid else None
        action.tool_args = dict(parsed.args) if parsed.valid else None
        action.info.setdefault("action_language", self.name)
        action.info.setdefault("action_language_contract", self.contract_id)
        action.info["parsed_action"] = parsed.to_dict()


def parse_and_annotate(language: ActionLanguage, action: Action) -> ParsedAction:
    parsed = language.parse(action).validate()
    language.annotate(action, parsed)
    return parsed


def bind_task_action_language(task: Task, language: ActionLanguage) -> Task:
    """Attach the benchmark-owned executable grammar to one task."""

    language.validate_contract(task)
    binding = {
        "contract_id": language.contract_id,
        "name": language.name,
        "prompt_vocabulary": language.prompt_vocabulary(task),
    }
    existing = task.metadata.get("action_language")
    if existing is not None and existing != binding:
        raise ValueError("task action-language binding changed after construction")
    task.metadata["action_language"] = binding
    return task


__all__ = [
    "ActionLanguage",
    "ParsedAction",
    "bind_task_action_language",
    "parse_and_annotate",
]
