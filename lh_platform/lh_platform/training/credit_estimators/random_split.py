"""Matched-size sham phase split for falsifying mechanical split effects."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace

from .contracts import CreditEstimate, CreditVisit, Exclusion, validate_visits
from .phasepool import PhasePoolConfig, PhasePoolEstimator


@dataclass(frozen=True)
class MatchedRandomConfig:
    seed: int
    exclusion: Exclusion = "trajectory"


class MatchedRandomEstimator:
    estimator_id = "matched_random_split"

    def __init__(self, config: MatchedRandomConfig) -> None:
        self.config = config

    def fit_predict(self, visits: Sequence[CreditVisit]) -> CreditEstimate:
        rows = validate_visits(visits)
        rng = random.Random(self.config.seed)
        by_coarse: dict[Hashable, list[CreditVisit]] = defaultdict(list)
        for row in rows:
            by_coarse[row.coarse_key].append(row)
        randomized: dict[str, Hashable] = {}
        original_sizes = []
        for coarse_key, members in by_coarse.items():
            ordered = sorted(members, key=lambda row: (row.trajectory_uid, row.visit_uid))
            labels = [row.phase_key for row in ordered]
            original_sizes.extend(Counter(labels).values())
            rng.shuffle(labels)
            for row, label in zip(ordered, labels, strict=True):
                randomized[row.visit_uid] = (
                    "matched_random",
                    self.config.seed,
                    coarse_key,
                    label,
                )
        randomized_rows = tuple(
            replace(row, phase_key=randomized[row.visit_uid]) for row in rows
        )
        estimate = PhasePoolEstimator(
            PhasePoolConfig(pooling="hard", exclusion=self.config.exclusion)
        ).fit_predict(randomized_rows)
        return CreditEstimate(
            estimator_id=f"matched_random_seed_{self.config.seed}",
            estimator_role="matched_size_random_split_control",
            predictions=estimate.predictions,
            metrics=estimate.metrics,
            metadata=(
                ("seed", self.config.seed),
                ("exclusion", self.config.exclusion),
                ("original_group_sizes", tuple(sorted(original_sizes))),
            ),
        )


__all__ = ["MatchedRandomConfig", "MatchedRandomEstimator"]
