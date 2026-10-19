"""Loss-mass mapping from LongHorizon records to veRL token aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lh_platform.training.experience_bridge import (
    ExperienceRecord,
    validate_experience_records,
)

VERL_OBJECTIVE_MAPPING_REVISION = "lh_to_verl_token_mean/v1"


@dataclass(frozen=True)
class VerLObjectiveRows:
    """Ragged tensors before device padding.

    veRL's token-mean divides the complete batch by ``active_token_count``.
    Multiplying each step advantage by that count and LongHorizon's already
    audited ``actor_token_weight`` preserves the original total gradient mass.
    """

    old_logprobs: tuple[tuple[float, ...], ...]
    response_masks: tuple[tuple[int, ...], ...]
    token_advantages: tuple[tuple[float, ...], ...]
    active_token_count: int
    mapping_revision: str = VERL_OBJECTIVE_MAPPING_REVISION


def build_verl_objective_rows(
    records: Sequence[ExperienceRecord],
) -> VerLObjectiveRows:
    validated = validate_experience_records(records)
    active_token_count = sum(row.active_token_count for row in validated)
    return VerLObjectiveRows(
        old_logprobs=tuple(row.response_old_logprobs for row in validated),
        response_masks=tuple(row.response_mask for row in validated),
        token_advantages=tuple(
            tuple(
                row.step_advantage * row.actor_token_weight * active_token_count if active else 0.0
                for active in row.response_mask
            )
            for row in validated
        ),
        active_token_count=active_token_count,
    )


__all__ = [
    "VERL_OBJECTIVE_MAPPING_REVISION",
    "VerLObjectiveRows",
    "build_verl_objective_rows",
]
