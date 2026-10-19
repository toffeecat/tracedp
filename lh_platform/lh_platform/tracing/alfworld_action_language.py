"""Single frozen parser for the ALFWorld textual action language."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedAlfWorldActionV1:
    schema: str | None
    arguments: dict[str, str]
    status: str
    reason: str | None = None


_ACTION_PATTERNS = (
    ("PickupObject", re.compile(r"^take (.+) from (.+)$"), ("object", "receptacle")),
    ("PutObject", re.compile(r"^move (.+) to (.+)$"), ("object", "receptacle")),
    ("CleanObject", re.compile(r"^clean (.+) with (.+)$"), ("object", "receptacle")),
    ("HeatObject", re.compile(r"^heat (.+) with (.+)$"), ("object", "receptacle")),
    ("CoolObject", re.compile(r"^cool (.+) with (.+)$"), ("object", "receptacle")),
    ("SliceObject", re.compile(r"^slice (.+) with (.+)$"), ("object", "tool")),
    ("ToggleObject", re.compile(r"^use (.+)$"), ("object",)),
    ("OpenObject", re.compile(r"^open (.+)$"), ("receptacle",)),
    ("CloseObject", re.compile(r"^close (.+)$"), ("receptacle",)),
    ("GotoLocation", re.compile(r"^go to (.+)$"), ("receptacle",)),
    ("ExamineEntity", re.compile(r"^examine (.+)$"), ("entity",)),
)


def parse_alfworld_action(action: str) -> ParsedAlfWorldActionV1:
    """Parse exactly one command from the frozen ALFWorld action grammar."""

    normalized = action.strip().lower()
    for schema, pattern, argument_names in _ACTION_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match is None:
            continue
        arguments = dict(zip(argument_names, match.groups(), strict=True))
        if schema == "PutObject" and " from " in arguments["object"]:
            return ParsedAlfWorldActionV1(
                schema=schema,
                arguments=arguments,
                status="unsupported",
                reason="PutObject template is 'move {object} to {receptacle}'",
            )
        return ParsedAlfWorldActionV1(
            schema=schema,
            arguments=arguments,
            status="matched",
        )
    return ParsedAlfWorldActionV1(
        schema=None,
        arguments={},
        status="unsupported",
        reason="no frozen action template matched",
    )


__all__ = ["ParsedAlfWorldActionV1", "parse_alfworld_action"]
