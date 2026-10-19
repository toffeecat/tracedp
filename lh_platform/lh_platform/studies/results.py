"""Paper-targeted analysis for paired GraphGPO/GRPO ALFWorld outcomes."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplicationResult:
    schema: str
    verdict: str
    graphgpo_success_rate: float
    grpo_success_rate: float
    paired_difference: float
    confidence_interval_95: tuple[float, float]
    paper_graphgpo_success_rate: float
    absolute_error: float
    equivalence_margin: float
    superiority_margin: float
    seed_count: int
    paired_task_count: int
    task_strata: tuple[str, ...]

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target


def build_replication_result(
    rows: Iterable[Mapping[str, Any]],
    *,
    paper_graphgpo_success_rate: float = 0.9271,
    equivalence_margin: float = 0.05,
    superiority_margin: float = 0.10,
    bootstrap_samples: int = 10_000,
    analysis_seed: int = 20_260_713,
) -> ReplicationResult:
    rows = list(rows)
    if any("checkpoint_step" in row for row in rows):
        latest: dict[tuple[str, int], int] = {}
        for row in rows:
            key = (str(row["variant_id"]), int(row["seed"]))
            latest[key] = max(latest.get(key, -1), int(row.get("checkpoint_step", -1)))
        rows = [
            row
            for row in rows
            if int(row.get("checkpoint_step", -1))
            == latest[(str(row["variant_id"]), int(row["seed"]))]
        ]
    by_variant_seed: dict[tuple[str, int], dict[str, bool]] = {}
    for row in rows:
        variant = str(row["variant_id"])
        seed = int(row["seed"])
        task = str(row["task_id"])
        won = row["won"]
        if not isinstance(won, bool):
            raise ValueError("won must be benchmark-owned boolean evidence")
        bucket = by_variant_seed.setdefault((variant, seed), {})
        if task in bucket:
            raise ValueError(f"duplicate outcome for {variant} seed={seed} task={task}")
        bucket[task] = won
    task_types = {
        (int(row["seed"]), str(row["task_id"])): str(row.get("task_type") or "all") for row in rows
    }
    seeds = sorted({seed for variant, seed in by_variant_seed if variant == "graphgpo"})
    if not seeds or set(seeds) != {seed for variant, seed in by_variant_seed if variant == "grpo"}:
        raise ValueError("paired seed sets are required")
    pairs: dict[int, list[tuple[str, int, int]]] = {}
    for seed in seeds:
        graph = by_variant_seed[("graphgpo", seed)]
        base = by_variant_seed[("grpo", seed)]
        if set(graph) != set(base):
            raise ValueError("paired task sets are required")
        pairs[seed] = [
            (task_types.get((seed, task), "all"), int(graph[task]), int(base[task]))
            for task in sorted(graph)
        ]
    if len(seeds) < 3:
        raise ValueError("at least three paired seeds are required")
    paired_task_count = sum(map(len, pairs.values()))
    if paired_task_count == 0:
        raise ValueError("paired task sets are empty")
    graph_rate = sum(g for values in pairs.values() for _, g, _ in values) / paired_task_count
    base_rate = sum(b for values in pairs.values() for _, _, b in values) / paired_task_count
    difference = graph_rate - base_rate
    rng = random.Random(analysis_seed)
    strata = tuple(sorted({task_type for values in pairs.values() for task_type, _, _ in values}))
    draws: list[float] = []
    for _ in range(max(1, bootstrap_samples)):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        sampled: list[tuple[int, int]] = []
        for seed in sampled_seeds:
            values = pairs[seed]
            for stratum in strata:
                stratum_values = [(g, b) for task_type, g, b in values if task_type == stratum]
                if not stratum_values:
                    continue
                sampled.extend(rng.choice(stratum_values) for _ in stratum_values)
        if not sampled:
            raise ValueError("bootstrap sample is empty")
        draws.append(sum(g - b for g, b in sampled) / len(sampled))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    error = abs(graph_rate - paper_graphgpo_success_rate)
    point_ok = error <= equivalence_margin and difference >= superiority_margin
    if point_ok and low > 0:
        verdict = "replicated"
    elif point_ok:
        verdict = "inconclusive"
    else:
        verdict = "not_replicated"
    return ReplicationResult(
        schema="lh_study_result/v1",
        verdict=verdict,
        graphgpo_success_rate=graph_rate,
        grpo_success_rate=base_rate,
        paired_difference=difference,
        confidence_interval_95=(low, high),
        paper_graphgpo_success_rate=paper_graphgpo_success_rate,
        absolute_error=error,
        equivalence_margin=equivalence_margin,
        superiority_margin=superiority_margin,
        seed_count=len(seeds),
        paired_task_count=paired_task_count,
        task_strata=strata,
    )
