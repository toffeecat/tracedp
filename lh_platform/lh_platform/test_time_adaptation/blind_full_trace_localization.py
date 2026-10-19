"""Blind localization over an entire actor-visible trajectory."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from lh_platform.tracing.types import TraceCaseV1

BLIND_TRACE_METHOD_IDS = (
    "generic_causal",
    "who_when_decisive",
    "traceelephant_unrecoverable",
)

_METHOD_INSTRUCTIONS = {
    "generic_causal": (
        "Find the earliest historical action or transition that causally created the "
        "state that later blocked task completion. Do not select the final symptom merely "
        "because it is recent."
    ),
    "who_when_decisive": (
        "Use a Who&When-style decisive-step analysis. Find the earliest historical step "
        "whose correction, followed by appropriate later adjustment, could have changed "
        "the failed outcome. Prefer the cause over the final symptom."
    ),
    "traceelephant_unrecoverable": (
        "Use a TraceElephant-style full-trace analysis. Find the earliest responsible "
        "transition that created the persistent state debt or lost the relevant recovery "
        "opportunity. Prefer the cause over later manifestations."
    ),
}


@dataclass(frozen=True)
class BlindTraceRanking:
    ranked_event_ids: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class BlindTraceMetrics:
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float
    first_gold_rank: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_blind_method_prompt(
    trace_case: TraceCaseV1,
    *,
    method_id: str,
    ranking_count: int = 3,
) -> str:
    """Serialize one complete trace while allowing a smaller candidate slate."""

    try:
        method_instruction = _METHOD_INSTRUCTIONS[method_id]
    except KeyError as error:
        raise ValueError(f"unknown blind trace method: {method_id}") from error

    event_lines = []
    for event in trace_case.events:
        event_lines.append(f'<event id="{event.event_id}">\n{event.content}\n</event>')
    if ranking_count < 1 or ranking_count > len(trace_case.events):
        raise ValueError("ranking count must fit the visible events")
    return (
        "You are diagnosing a failed long-horizon agent trajectory.\n"
        f"{method_instruction} Rank exactly {ranking_count} event IDs from most to least likely.\n\n"
        f"Task:\n{trace_case.context.task_specification}\n\n"
        f"Observed failure:\n{trace_case.context.evaluator_failure}\n\n"
        "Full actor-visible trajectory:\n"
        + "\n\n".join(event_lines)
        + "\n\nReturn only one JSON object with exactly two fields: ranked_event_ids "
        f"must be an array of exactly {ranking_count} unique IDs. Copy only the id attribute value "
        "from each event, without XML tags or prefixes; explanation must be one brief "
        "string. Do not use markdown."
    )


def build_blind_full_trace_prompt(trace_case: TraceCaseV1) -> str:
    """Preserve the original generic full-trace prompt."""

    return build_blind_method_prompt(trace_case, method_id="generic_causal")


def parse_blind_trace_ranking(
    response: str,
    *,
    allowed_event_ids: tuple[str, ...],
    expected_count: int = 3,
) -> BlindTraceRanking:
    """Parse a strict ranking over events in the visible candidate slate."""

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        # Some providers wrap an otherwise valid JSON answer in one fenced block.
        fenced = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if len(fenced) != 1:
            raise ValueError("blind tracer response must be one JSON object") from error
        try:
            payload = json.loads(fenced[0])
        except json.JSONDecodeError as fenced_error:
            raise ValueError("blind tracer response must be one JSON object") from fenced_error
    if not isinstance(payload, dict) or set(payload) != {"ranked_event_ids", "explanation"}:
        raise ValueError("blind tracer response must contain only ranking and explanation")
    ranked = payload["ranked_event_ids"]
    explanation = payload["explanation"]
    if expected_count < 1 or expected_count > len(allowed_event_ids):
        raise ValueError("expected count must fit the visible candidate slate")
    if (
        not isinstance(ranked, list)
        or len(ranked) != expected_count
        or not all(isinstance(event_id, str) for event_id in ranked)
        or len(set(ranked)) != expected_count
    ):
        raise ValueError(
            f"blind tracer must return exactly {expected_count} unique event IDs"
        )
    if not set(ranked) <= set(allowed_event_ids):
        raise ValueError("blind tracer returned an event absent from the visible trace")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("blind tracer explanation must be non-empty")
    return BlindTraceRanking(tuple(ranked), explanation.strip())


def score_blind_trace_ranking(
    ranking: BlindTraceRanking,
    *,
    gold_event_ids: tuple[str, ...],
) -> BlindTraceMetrics:
    """Score the first matched member of a non-empty set-valued causal gold."""

    if not gold_event_ids:
        raise ValueError("blind trace evaluator requires at least one verified root")
    first_rank = next(
        (
            index
            for index, event_id in enumerate(ranking.ranked_event_ids, start=1)
            if event_id in gold_event_ids
        ),
        None,
    )
    return BlindTraceMetrics(
        hit_at_1=first_rank == 1,
        hit_at_3=first_rank is not None,
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        first_gold_rank=first_rank,
    )


__all__ = [
    "BLIND_TRACE_METHOD_IDS",
    "BlindTraceMetrics",
    "BlindTraceRanking",
    "build_blind_full_trace_prompt",
    "build_blind_method_prompt",
    "parse_blind_trace_ranking",
    "score_blind_trace_ranking",
]
