"""Adapt a public step attribution into investigation guidance, without inventing a fix."""

from lh_platform.tracing.types import TraceInputV1
from lh_platform.tracing.unified import TraceFindingKind, TraceFindingV1

# Treatment identifier for the public-step investigation lead.
BRIDGE_TREATMENT_ID = "public_step_investigation_lead_v2"


# Shared neutral continuation (condition C2): it does not tell the Actor that the final state
# failed, so recovery depends on what the Runtime shows it, not on the prompt.
NEUTRAL_CONTINUATION_OPENING = "Continue the original task."
INFORMED_CONTINUATION_OPENING = "Continue the original task from its observed failed final state."


def historical_investigation_continuation(guidance: str, *, opening: str | None = None) -> str:
    """Continue a historical actor from a step lead, without inventing a postcondition.

    The lead may be a successful tool call. The actor is asked to check it against
    the task's public requirements in the current sandbox, not to apply it as a repair.
    """

    text = guidance.strip()
    if not text:
        raise ValueError("historical investigation continuation requires guidance")
    return (
        f"{opening or INFORMED_CONTINUATION_OPENING} Use the "
        "following external text only as an investigation lead. It may point at a "
        "step whose own tool result succeeded, and it may not state a postcondition. "
        "Inspect the current sandbox, check the lead against the task's public "
        "requirements, and finish the task:\n"
        f"{text}\n"
        "Do not assume the guidance or the prior attempt's final answer is correct."
    )


def compile_public_step_guidance(case: TraceInputV1, finding: TraceFindingV1) -> str:
    """Expose the attributed public event to the same Actor for current-state recovery.

    The upstream attribution is an investigation lead, not a verified causal root.
    History position follows Who&When's zero-based event-array convention; it is
    independent of an event's original source sequence number. No repair command
    or evaluator information is inferred by this local adaptation. The lead also
    states that a successful tool result on the attributed step is not task success,
    because the selected event can be a write that itself returned no error.
    """
    if finding.kind is not TraceFindingKind.STEP_ATTRIBUTION or len(finding.event_ids) != 1:
        raise ValueError("forward step guidance requires exactly one attributed event")
    event_id = finding.event_ids[0]
    if event_id not in case.event_ids:
        raise ValueError("attributed event is outside the public recovery input")
    position, event = next(
        (index, event) for index, event in enumerate(case.events) if event.event_id == event_id
    )
    return (
        "Unverified step attribution to investigate.\n"
        "This is an investigation lead, not a repair, and it states no postcondition. "
        "A successful tool result on the attributed step does not mean the task succeeded.\n"
        f"Public history position (zero-based): {position}\n"
        f"Public event content:\n{event.content}\n\n"
        f"Attribution explanation:\n{finding.explanation}"
    )
