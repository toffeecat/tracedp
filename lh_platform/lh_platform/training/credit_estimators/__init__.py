"""Frozen-data G2PO estimator family used by component ablations.

Owns Credit State Identity keys and fit_predict (GroupedMean / PhasePool /
MatchedRandom / A³). Matched aliasing scientific audits live in
``training.aliasing_audit``; modules under this package with those names are
compatibility re-exports only.
"""

from .a3 import A3Config, A3Estimator
from .contracts import (
    CreditEstimate,
    CreditEstimator,
    CreditEvidenceError,
    CreditMetrics,
    CreditPrediction,
    CreditVisit,
    FrozenBehaviorEvidence,
)
from .coverage_plan import build_loto_collection_plan
from .credit_keys import (
    FROZEN_CALIBRATION_CONTRACT,
    LEGACY_ONLINE_CONTRACT,
    CreditStateIdentityContract,
    anchor_observation_key,
    assert_contract_parity,
    verifier_refined_key,
    verifier_refined_online_key,
)
from .dataset import credit_visits_from_trajectory_records
from .grouped import GroupedMeanConfig, GroupedMeanEstimator
from .phasepool import PhasePoolConfig, PhasePoolEstimator, self_influence_diagnostic
from .random_split import MatchedRandomConfig, MatchedRandomEstimator

__all__ = [
    "A3Config",
    "A3Estimator",
    "CreditEstimate",
    "CreditEstimator",
    "CreditEvidenceError",
    "CreditMetrics",
    "CreditPrediction",
    "CreditStateIdentityContract",
    "CreditVisit",
    "FROZEN_CALIBRATION_CONTRACT",
    "FrozenBehaviorEvidence",
    "GroupedMeanConfig",
    "GroupedMeanEstimator",
    "LEGACY_ONLINE_CONTRACT",
    "MatchedRandomConfig",
    "MatchedRandomEstimator",
    "PhasePoolConfig",
    "PhasePoolEstimator",
    "anchor_observation_key",
    "assert_contract_parity",
    "build_loto_collection_plan",
    "credit_visits_from_trajectory_records",
    "self_influence_diagnostic",
    "verifier_refined_key",
    "verifier_refined_online_key",
]
