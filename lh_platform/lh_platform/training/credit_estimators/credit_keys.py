"""Single factory for Credit State Identity keys used by frozen estimators.

Callers that need Anchor Observation, Verifier-Refined, or exact-snapshot
identities must build them here rather than inventing parallel coarse_key /
refined_key adapters.  Estimators consume the resulting Hashable values; they
do not reinterpret residual text themselves.

Two Verifier-Refined shapes are intentional and must stay distinct:

* ``verifier_refined_key`` — frozen / cross-fitted visits (includes remaining
  budget).  Used by ``GroupedMean(key=phase)`` and PhasePool via ``phase_key``.
* ``verifier_refined_online_key`` — online P0 full-pool G2PO-TN hard split
  (no budget).  Used by the ``g2po_bridge`` online Verifier-Refined path.
"""

from __future__ import annotations

from collections.abc import Hashable


def anchor_observation_key(*, pair_uid: str, observation: str) -> Hashable:
    """G2PO Anchor Observation identity scoped to one controlled pair."""

    if not pair_uid or not observation:
        raise ValueError("anchor observation key requires pair_uid and observation")
    return (pair_uid, observation)


def verifier_refined_key(
    *,
    coarse_key: Hashable,
    residual_signature: str,
    frontier_signature: str,
    remaining_budget: int,
) -> Hashable:
    """Frozen Verifier-Refined identity: coarse + residual + frontier + budget."""

    if not residual_signature or not frontier_signature:
        raise ValueError("verifier-refined key requires residual and frontier signatures")
    if isinstance(remaining_budget, bool) or remaining_budget < 0:
        raise ValueError("verifier-refined key requires a non-negative remaining_budget")
    return (coarse_key, residual_signature, frontier_signature, remaining_budget)


def verifier_refined_online_key(
    *,
    anchor_observation: Hashable,
    residual_signature: str,
    frontier_signature: str,
) -> Hashable:
    """Online Verifier-Refined G2PO-TN identity (legacy P0 hard split).

    Deliberately omits remaining budget so online full-pool TN partitioning
    stays receipt-compatible.  Cross-fitted estimators must use
    :func:`verifier_refined_key` instead.
    """

    if not residual_signature or not frontier_signature:
        raise ValueError("verifier-refined online key requires residual and frontier signatures")
    return (
        "oracle_residual_frontier",
        anchor_observation,
        residual_signature,
        frontier_signature,
    )


import hashlib
import json
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CreditStateIdentityContract:
    """Unified identity contract binding calibration and online credit keys.

    Every calibration report and online method component must emit
    ``credit_state_identity_contract_sha256``.  Online admission fails closed
    when the calibration contract hash differs from the online contract hash.
    """

    schema: str = "lh_credit_state_identity_contract/v1"
    scope: Literal["task_group", "task_contract"] = "task_group"
    anchor_projection_id: str = "official_anchor_observation"
    raw_observation_mode: str = "sha256_exact"
    affordance_mode: str = "canonical_sorted_sha256"
    budget_mode: Literal["none", "exact", "bucket_5"] = "exact"
    residual_mode: str = "verifier_residual_goal_signature"
    frontier_mode: str = "verifier_frontier_signature"
    verifier_component_sha256: str = ""

    def contract_sha256(self) -> str:
        payload = {
            "schema": self.schema,
            "scope": self.scope,
            "anchor_projection_id": self.anchor_projection_id,
            "raw_observation_mode": self.raw_observation_mode,
            "affordance_mode": self.affordance_mode,
            "budget_mode": self.budget_mode,
            "residual_mode": self.residual_mode,
            "frontier_mode": self.frontier_mode,
            "verifier_component_sha256": self.verifier_component_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


FROZEN_CALIBRATION_CONTRACT = CreditStateIdentityContract(budget_mode="exact")
LEGACY_ONLINE_CONTRACT = CreditStateIdentityContract(budget_mode="none")


def assert_contract_parity(
    calibration_contract: CreditStateIdentityContract,
    online_contract: CreditStateIdentityContract,
) -> None:
    """Fail closed if calibration and online identity contracts diverge."""

    if calibration_contract.contract_sha256() != online_contract.contract_sha256():
        raise ValueError(
            "credit state identity contract mismatch: "
            f"calibration={calibration_contract.contract_sha256()[:16]}... "
            f"online={online_contract.contract_sha256()[:16]}... "
            "Online admission requires exact contract parity."
        )


__all__ = [
    "FROZEN_CALIBRATION_CONTRACT",
    "LEGACY_ONLINE_CONTRACT",
    "CreditStateIdentityContract",
    "anchor_observation_key",
    "assert_contract_parity",
    "verifier_refined_key",
    "verifier_refined_online_key",
]
