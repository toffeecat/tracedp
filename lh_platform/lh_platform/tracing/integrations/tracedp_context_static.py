"""Static attribution ablations over dependency-only and dual-view TraceDP context."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from lh_platform.tracing.integrations.longh_residual_causal import (
    LongHResidualCausalRankingV1,
    compile_public_causal_ledger,
)
from lh_platform.tracing.types import EvidenceLevel, TraceCaseV1
from lh_platform.tracing.unified import (
    TraceFindingKind,
    TraceFindingV1,
    TraceMethodDescriptorV1,
    TraceMethodOutputV1,
    TraceMethodRequestV1,
)

DEPENDENCY_ONLY_METHOD_ID = "baseline.dependency_only"
TRACEDP_CONTEXT_STATIC_METHOD_ID = "longh.tracedp_context_static"
STATIC_CONTEXT_METHOD_IDS = (
    DEPENDENCY_ONLY_METHOD_ID,
    TRACEDP_CONTEXT_STATIC_METHOD_ID,
)

_REPRESENTATION_BY_METHOD = {
    DEPENDENCY_ONLY_METHOD_ID: "public_dependency_graph_only_v1",
    TRACEDP_CONTEXT_STATIC_METHOD_ID: "public_full_trace_plus_residual_causal_context_v1",
}


def _output_contract(ranking_count: int) -> str:
    return (
        "Return only one JSON object with exactly two fields: ranked_event_ids must "
        f"contain exactly {ranking_count} unique IDs copied from event id attributes, "
        "and explanation must be one brief string. Do not use markdown."
    )


def _dependency_lines(case: TraceCaseV1, *, include_semantics: bool) -> list[str]:
    """Render one stable event ledger with or without action/result semantics."""

    lines: list[str] = []
    for entry in compile_public_causal_ledger(case).entries:
        fields = [
            f'<event id="{entry.event_id}" sequence="{entry.sequence}">',
            f"roles={','.join(entry.operation_roles)}; status={entry.result_status}; "
            f"tool={entry.tool_name or 'none'}",
            f"resources={','.join(entry.resources) or '<none>'}",
            f"writes={','.join(entry.write_resources) or '<none>'}",
            f"upstream_writers={','.join(entry.upstream_writer_event_ids) or '<none>'}",
        ]
        if include_semantics:
            fields.extend(
                (
                    f"command={entry.command_summary or '<none>'}",
                    f"actor={entry.actor_summary or '<none>'}",
                    f"result={entry.result_summary or '<empty>'}",
                )
            )
        fields.append("</event>")
        lines.append("\n".join(fields))
    return lines


def build_dependency_only_prompt(case: TraceCaseV1, *, ranking_count: int = 3) -> str:
    """Expose only the mechanically compiled dependency topology to the same ranker."""

    if case.context is None:
        raise ValueError("dependency-only localization requires public task context")
    if ranking_count < 1 or ranking_count > len(case.events):
        raise ValueError("ranking count must fit the visible events")
    return (
        "You are localizing a root cause from a dependency-only projection of a failed "
        "long-horizon run. The projection contains no action text, result text, or answer "
        "key. Trace dependencies backward from the failed task and rank the earliest "
        "event that could have created an unresolved prerequisite. Treat an error status "
        "as evidence, not automatic causation.\n"
        f"Rank exactly {ranking_count} event IDs.\n\n"
        f"Task:\n{case.context.task_specification}\n\n"
        f"Observed failure:\n{case.context.evaluator_failure}\n\n"
        "Dependency-only event graph:\n"
        + "\n\n".join(_dependency_lines(case, include_semantics=False))
        + "\n\n"
        + _output_contract(ranking_count)
    )


def build_tracedp_context_static_prompt(case: TraceCaseV1, *, ranking_count: int = 3) -> str:
    """Combine the full public trace with its mechanical causal projection in one call."""

    if case.context is None:
        raise ValueError("TraceDP-context localization requires public task context")
    if ranking_count < 1 or ranking_count > len(case.events):
        raise ValueError("ranking count must fit the visible events")
    raw_events = "\n\n".join(
        f'<event id="{event.event_id}">\n{event.content}\n</event>' for event in case.events
    )
    return (
        "You are the static localization head for TraceDP context. This is not a replay "
        "or a dynamic-programming run. You receive two lossless views of the same public "
        "trajectory: the complete event text and a mechanical causal ledger. No answer "
        "key is present.\n"
        "Trace the terminal symptom backward. Distinguish commands that actually executed "
        "from commands skipped after an earlier shell failure. Consider harmful writes, "
        "misleading observations that anchor later decisions, omitted prerequisites, wrong "
        "ordering, and lost recovery opportunities. Rank the earliest independently "
        "plausible decisive transition, not merely the first error or final symptom.\n"
        f"Rank exactly {ranking_count} event IDs.\n\n"
        f"Task:\n{case.context.task_specification}\n\n"
        f"Observed failure:\n{case.context.evaluator_failure}\n\n"
        "Complete public trajectory:\n"
        + raw_events
        + "\n\nMechanical residual-causal context:\n"
        + "\n\n".join(_dependency_lines(case, include_semantics=True))
        + "\n\n"
        + _output_contract(ranking_count)
    )


@dataclass(frozen=True)
class StaticContextPredictionV1:
    """One static ranking plus audited model cost and runtime evidence."""

    ranking: LongHResidualCausalRankingV1
    model_calls: int
    response_model_id: str
    reasoning_tokens: int
    reasoning_tokens_reported: bool
    raw_response_reused: bool
    prompt_character_count: int
    ledger_event_count: int
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class StaticContextRunner(Protocol):
    def run(
        self, *, request: TraceMethodRequestV1, method_id: str
    ) -> StaticContextPredictionV1: ...


def static_context_catalog_descriptors() -> tuple[TraceMethodDescriptorV1, ...]:
    """Describe both ablations without claiming dynamic TraceDP execution."""

    return (
        TraceMethodDescriptorV1(
            method_id=DEPENDENCY_ONLY_METHOD_ID,
            display_name="Dependency-only Static Localizer",
            family="failure_attribution",
            capabilities=("step_attribution", "ranked_candidates", "dependency_graph"),
            input_modalities=("text_trajectory",),
            output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
            integration_level="catalog_only",
            execution_path="public_dependency_graph_then_one_call_ranker",
            evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
            source_relpath="lh_platform/tracing/integrations/tracedp_context_static.py",
            source_revision=None,
            unavailable_reason="static context runtime has not passed admission",
            corpus_method_id="longh_style_baselines",
        ),
        TraceMethodDescriptorV1(
            method_id=TRACEDP_CONTEXT_STATIC_METHOD_ID,
            display_name="TraceDP Context Static Localizer",
            family="failure_attribution",
            capabilities=(
                "step_attribution",
                "ranked_candidates",
                "public_causal_ledger",
                "full_trace_dual_view",
            ),
            input_modalities=("text_trajectory",),
            output_kinds=(TraceFindingKind.STEP_ATTRIBUTION,),
            integration_level="catalog_only",
            execution_path="full_public_trace_plus_causal_ledger_then_one_call_ranker",
            evidence_ceiling=EvidenceLevel.L3_LOCALIZED,
            source_relpath="lh_platform/tracing/integrations/tracedp_context_static.py",
            source_revision=None,
            unavailable_reason="static context runtime has not passed admission",
            corpus_method_id="longh",
        ),
    )


class StaticContextTraceAdapter:
    """Run one audited localization call over a frozen static representation."""

    def __init__(self, *, method_id: str, runner: StaticContextRunner) -> None:
        if method_id not in STATIC_CONTEXT_METHOD_IDS:
            raise ValueError(f"unknown static context method: {method_id}")
        self._runner = runner
        descriptor = next(
            item for item in static_context_catalog_descriptors() if item.method_id == method_id
        )
        self.descriptor = replace(
            descriptor,
            integration_level="platform_native",
            source_revision="v1",
            unavailable_reason=None,
        )

    def execute(self, request: TraceMethodRequestV1) -> TraceMethodOutputV1:
        """Validate identity, budget, and cost before publishing the ranking."""

        if not isinstance(request.case, TraceCaseV1):
            raise ValueError("static context localization requires TraceCaseV1")
        prediction = self._runner.run(
            request=request,
            method_id=self.descriptor.method_id,
        )
        expected_count = min(3, len(request.case.events))
        if (
            prediction.model_calls != 1
            or not prediction.response_model_id
            or prediction.reasoning_tokens != 0
            or type(prediction.reasoning_tokens_reported) is not bool
            or len(prediction.ranking.ranked_event_ids) != expected_count
            or prediction.prompt_character_count < 1
            or prediction.ledger_event_count != len(request.case.events)
            or prediction.input_tokens < 0
            or prediction.output_tokens < 0
            or prediction.cache_read_tokens < 0
            or prediction.cache_write_tokens < 0
            or prediction.latency_seconds < 0
        ):
            raise ValueError("static context model audit failed")
        return TraceMethodOutputV1(
            findings=(
                TraceFindingV1(
                    kind=TraceFindingKind.STEP_ATTRIBUTION,
                    event_ids=prediction.ranking.ranked_event_ids,
                    label="ranked_root_candidates",
                    score=None,
                    explanation=prediction.ranking.explanation,
                    attributes=(
                        ("representation", _REPRESENTATION_BY_METHOD[self.descriptor.method_id]),
                        ("model_calls", "1"),
                        ("response_model_id", prediction.response_model_id),
                        ("reasoning_tokens", str(prediction.reasoning_tokens)),
                        (
                            "reasoning_tokens_reported",
                            str(prediction.reasoning_tokens_reported).lower(),
                        ),
                        ("raw_response_reused", str(prediction.raw_response_reused).lower()),
                        ("prompt_character_count", str(prediction.prompt_character_count)),
                        ("ledger_event_count", str(prediction.ledger_event_count)),
                        ("input_tokens", str(prediction.input_tokens)),
                        ("output_tokens", str(prediction.output_tokens)),
                        ("cache_read_tokens", str(prediction.cache_read_tokens)),
                        ("cache_write_tokens", str(prediction.cache_write_tokens)),
                        ("latency_seconds", f"{prediction.latency_seconds:.6f}"),
                        ("private_inputs_used", "false"),
                        ("dynamic_programming_executed", "false"),
                    ),
                ),
            ),
            evidence_level=EvidenceLevel.L3_LOCALIZED,
        )


__all__ = [
    "DEPENDENCY_ONLY_METHOD_ID",
    "STATIC_CONTEXT_METHOD_IDS",
    "TRACEDP_CONTEXT_STATIC_METHOD_ID",
    "StaticContextPredictionV1",
    "StaticContextTraceAdapter",
    "build_dependency_only_prompt",
    "build_tracedp_context_static_prompt",
    "static_context_catalog_descriptors",
]
