"""Stable data contracts for frozen parent-child state-use evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lh_platform.training.alfworld_agentic_suite import AlfWorldBranchObservation
from lh_platform.training.alfworld_controlled_aliasing import (
    ControlledAliasingBranchRecipe,
)
from lh_platform.training.g2po.actor_protocol import G2POActorHistoryStep
from lh_platform.training.subproblems.base import GoalGraph, GoalVector


class FunctionalStateUseError(ValueError):
    """Raised when a frozen behavioral comparison is not identifiable."""


@dataclass(frozen=True)
class ResolvedFunctionalStateUseBenchmark:
    config_path: Path
    workspace_root: Path
    payload: Mapping[str, Any]
    sha256: str


class FrozenActionScoringPolicy(Protocol):
    checkpoint_sha256: str

    def score_admissible_actions(
        self,
        prompt_text: str,
        admissible_actions: tuple[str, ...],
        *,
        temperature: float,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class VerifiedStateEvidence:
    state_snapshot_sha256: str
    state_snapshot_artifact_path: str
    state_restore_receipt_sha256: str
    state_restore_receipt_path: str
    observation_fingerprint_sha256: str
    semantic_graph_verification: Mapping[str, Any]
    semantic_snapshot_verification: Mapping[str, Any]

    def to_artifact_fields(self) -> dict[str, Any]:
        """Preserve the established flat JSON evidence schema."""

        return {
            "state_snapshot_sha256": self.state_snapshot_sha256,
            "state_snapshot_artifact_path": self.state_snapshot_artifact_path,
            "state_restore_receipt_sha256": self.state_restore_receipt_sha256,
            "state_restore_receipt_path": self.state_restore_receipt_path,
            "state_observation_fingerprint_sha256": (self.observation_fingerprint_sha256),
            "semantic_graph_verification": dict(self.semantic_graph_verification),
            "semantic_snapshot_verification": dict(self.semantic_snapshot_verification),
        }


@dataclass(frozen=True)
class ReconstructedState:
    recipe: ControlledAliasingBranchRecipe
    observation: AlfWorldBranchObservation
    task_description: str
    history: tuple[G2POActorHistoryStep, ...]
    graph: GoalGraph
    vector: GoalVector
    normal_prompt: str
    evidence: VerifiedStateEvidence

    @property
    def state_snapshot_sha256(self) -> str:
        return self.evidence.state_snapshot_sha256

    @property
    def observation_fingerprint_sha256(self) -> str:
        return self.evidence.observation_fingerprint_sha256


__all__ = [
    "FrozenActionScoringPolicy",
    "FunctionalStateUseError",
    "ReconstructedState",
    "ResolvedFunctionalStateUseBenchmark",
    "VerifiedStateEvidence",
]
