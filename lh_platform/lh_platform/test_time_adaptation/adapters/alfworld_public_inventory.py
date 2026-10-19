"""Conservative inventory evidence derived from actor-visible ALFWorld text."""

from __future__ import annotations

import re

from lh_platform.test_time_adaptation.public_residual_sidecar import (
    PublicTransitionEvidence,
    PublicTransitionSidecar,
)

_PICKUP_CONFIRMATION = re.compile(r"\byou pick up\b", re.IGNORECASE)
_RELEASE_CONFIRMATION = re.compile(r"\byou (?:put|move)\b", re.IGNORECASE)


def _inventory_capacity_transition(
    *,
    action: str,
    next_observation: str,
) -> tuple[tuple[tuple[str, bool], ...], tuple[tuple[str, bool], ...]]:
    """Infer a capacity write only when the environment text explicitly confirms it."""

    normalized_action = action.strip().lower()
    if normalized_action.startswith("take ") and _PICKUP_CONFIRMATION.search(next_observation):
        return (
            (("inventory_capacity_available", True),),
            (("inventory_capacity_available", False),),
        )
    if normalized_action.startswith(("put ", "move ")) and _RELEASE_CONFIRMATION.search(
        next_observation
    ):
        return (
            (("inventory_capacity_available", False),),
            (("inventory_capacity_available", True),),
        )
    return (), ()


def build_public_inventory_sidecar(
    evidence: PublicTransitionEvidence,
) -> PublicTransitionSidecar:
    """Build a public sidecar without goal labels, facts, or future information.

    A transition alone does not identify which child goal is blocked.  That
    decision is deliberately deferred to the failure router, where absent
    public evidence must lead to abstention.
    """

    before, after = _inventory_capacity_transition(
        action=evidence.action,
        next_observation=evidence.next_observation,
    )
    return PublicTransitionSidecar(
        task_id=evidence.task_id,
        event_id=evidence.event_id,
        sequence=evidence.sequence,
        public_predicates_before=before,
        public_predicates_after=after,
    )


__all__ = ["build_public_inventory_sidecar"]
