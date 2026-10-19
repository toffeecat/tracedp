"""Matrix-runner: expand an ExperimentConfig into (benchmark, method) pairs
and execute them through the existing Runner, applying gates first.

Gate behaviour:
  - For each pair, look up the registered benchmark and method classes.
  - If `gates.allow_smoke_only` is False, SKIP any pair whose benchmark's
    `eval_validity == "smoke_only"`.
  - If `gates.min_eval_validity` ranks higher than the pair's eval_validity,
    SKIP the pair.
  - If `gates.min_integration_level` ranks higher than the pair's
    `integration_level`, SKIP the pair.
  - If the method declares `requires_benchmark` (e.g. webvoyager_native),
    SKIP the pair unless it matches the benchmark's registry key.

Seed sweep:
  - `cfg.run.seeds` is a list of int (or empty -> [None]).
  - We iterate over each seed, then for each (b, m) pair we build a
    Runner with `seed=seed`. Each pair's metrics.json gets its own
    `-s{seed}` suffix in `run_id`, so the parallel seed sweeps write
    to separate directories and can be aggregated by the summarize
    step (which groups by `(method, benchmark, seed)`).

The runner records, per run:
  - run.json:        full experiment config + gate decision + seed
  - metrics.json:    long-horizon + offline eval (ToolBench-style)
  - aggregate:       also includes the gate decision reason + seed

Notes:
  - This script does NOT redefine the runner. It composes `Runner` from
    `lh_platform/core/runner.py` and applies gates before dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import stdev
from typing import Any

from lh_platform.config.load import load_config
from lh_platform.config.schema import (
    INTEGRATION_RANK,
    VALIDITY_RANK,
    ExperimentConfig,
)
from lh_platform.core.artifact_redaction import redact_artifact, redact_text
from lh_platform.core.registry import (
    BENCHMARKS,
    bootstrap_runtime_registries,
)
from lh_platform.core.run_admission import resolve_method_admission


def _resolve_pair(
    benchmark_key: str,
    method_key: str,
    *,
    strict_method_boundary: bool = False,
    method_kwargs: dict[str, Any] | None = None,
) -> tuple[Any | None, Any | None, str]:
    """Return (benchmark_cls, method descriptor, reason)."""
    if not BENCHMARKS.has(benchmark_key):
        return None, None, f"benchmark not registered: {benchmark_key}"
    b_cls = BENCHMARKS._items[BENCHMARKS.resolve(benchmark_key)]
    required_action_contract = str(getattr(b_cls, "required_action_contract", "") or "")
    try:
        method_descriptor = resolve_method_admission(
            method_id=method_key,
            benchmark_id=benchmark_key,
            benchmark_required_action_contract=required_action_contract,
            strict=strict_method_boundary,
            method_kwargs=method_kwargs,
        )
    except KeyError as error:
        return None, None, str(error)
    if method_descriptor.reason:
        return b_cls, method_descriptor, method_descriptor.reason
    return b_cls, method_descriptor, ""


def _gated(
    benchmark_cls: Any,
    method_descriptor: Any,
    gates: ExperimentConfig.gates.__class__,
    *,
    method_key: str | None = None,
) -> str:
    """Return '' if the pair passes gates, otherwise the reason."""
    validity = getattr(benchmark_cls, "eval_validity", "smoke_only")
    dynamic_validity = bool(getattr(benchmark_cls, "runtime_eval_validity_dynamic", False))
    # Some adapters (ToolBench/SWE-bench local snapshots) resolve their data
    # root only when the concrete benchmark instance is constructed.  A
    # class-level fixture label cannot reject a full server corpus before its
    # manifest is even read.  Runner repeats this exact gate after instance
    # construction, before it creates a method or makes a model call.
    if dynamic_validity:
        # Dynamic adapters may improve after their concrete data root is
        # inspected, but only up to an explicit ceiling.  Deferring every
        # gate would accidentally admit a ToolBench loader into a
        # ``real_native`` study merely because its class starts as a fixture.
        max_validity = getattr(benchmark_cls, "runtime_eval_validity_max", validity)
        min_v = gates.min_eval_validity
        if VALIDITY_RANK.get(max_validity, -1) < VALIDITY_RANK.get(min_v, 0):
            return (
                f"benchmark max_runtime_eval_validity={max_validity!r} "
                f"< min_eval_validity={min_v!r}"
            )
    else:
        if validity == "smoke_only" and not gates.allow_smoke_only:
            return "allow_smoke_only=false"
        min_v = gates.min_eval_validity
        if VALIDITY_RANK.get(validity, -1) < VALIDITY_RANK.get(min_v, 0):
            return f"benchmark eval_validity={validity!r} < min_eval_validity={min_v!r}"
    integration = getattr(method_descriptor, "integration_level", "mock")
    min_l = gates.min_integration_level
    if INTEGRATION_RANK.get(integration, -1) < INTEGRATION_RANK.get(min_l, 0):
        return f"method integration_level={integration!r} < min_integration_level={min_l!r}"
    return ""


def _aggregate_across_seeds(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize declared run metrics across seeds with explicit n/std."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        if summary.get("skipped") or summary.get("error"):
            continue
        benchmark = summary.get("benchmark")
        method = summary.get("method")
        if isinstance(benchmark, str) and isinstance(method, str):
            groups[(benchmark, method)].append(summary)
    output: list[dict[str, Any]] = []
    for (benchmark, method), rows in sorted(groups.items()):
        metrics: dict[str, dict[str, float | int]] = {}
        keys = set().union(*(row.keys() for row in rows))
        for key in sorted(keys):
            if key != "success_rate" and not key.startswith("mean_"):
                continue
            values = []
            for row in rows:
                value = row.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                numeric = float(value)
                if math.isfinite(numeric):
                    values.append(numeric)
            if values:
                metrics[key] = {
                    "mean": sum(values) / len(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
        output.append(
            {
                "benchmark": benchmark,
                "method": method,
                "seed_count": len(rows),
                "seeds": sorted(row["seed"] for row in rows if isinstance(row.get("seed"), int)),
                "metrics": metrics,
            }
        )
    return output


def _run_one_pair(
    cfg: ExperimentConfig,
    benchmark_key: str,
    method_key: str,
    benchmark_cls: Any,
    method_descriptor: Any,
    gate_reason: str,
    seed: int | None,
) -> dict[str, Any]:
    """Build a Runner and execute a single (benchmark, method, model, seed)."""
    from lh_platform.core.runner import Runner

    # Build model_kwargs: experiment.model.kwargs + run.extra
    model_kwargs: dict[str, Any] = dict(cfg.model.kwargs)
    if cfg.run.extra:
        # run_extra gets merged but never overwrites an explicit model_kwarg
        for k, v in cfg.run.extra.items():
            model_kwargs.setdefault(k, v)

    method_kwargs: dict[str, Any] = dict(cfg.run.method_kwargs)
    if seed is not None:
        method_kwargs.setdefault("seed", seed)

    config_extra: dict[str, Any] = {
        "model_kwargs": model_kwargs,
        "model_runtime": cfg.model.runtime.to_dict(),
        "benchmark_kwargs": dict(cfg.run.benchmark_kwargs),
        "method_kwargs": method_kwargs,
        "experiment_name": cfg.name,
        "experiment_description": cfg.description,
        "config_source": str(cfg.source_path) if cfg.source_path else None,
        "schema_version": cfg.schema_version,
        "gates": {
            "allow_smoke_only": cfg.gates.allow_smoke_only,
            "min_eval_validity": cfg.gates.min_eval_validity,
            "min_integration_level": cfg.gates.min_integration_level,
            "leaderboard_eligible": cfg.gates.leaderboard_eligible,
        },
        "seeds": cfg.run.seeds,
        "gate_decision": {
            "benchmark": benchmark_key,
            "method": method_key,
            "reason": gate_reason,
            "passed": gate_reason == "",
        },
    }

    runner = Runner(
        benchmark=benchmark_key,
        method=method_key,
        model=cfg.model.provider,
        evaluator=cfg.run.evaluator,
        n_tasks=cfg.run.n_tasks,
        results_root=cfg.run.results_root,
        config_extra=config_extra,
        seed=seed,
    )
    metrics = runner.run()
    agg = metrics.get("aggregate", {})
    agg["experiment_name"] = cfg.name
    agg["experiment_description"] = cfg.description
    agg["benchmark"] = benchmark_key
    agg["method"] = method_key
    agg["model_provider"] = cfg.model.provider
    agg["seed"] = seed
    # The Runner replaces this with instance-level evidence after loading the
    # data. Keep the class value only as a diagnostic fallback; do not
    # overwrite a runtime downgrade (e.g. missing WebVoyager JSONL).
    agg.setdefault(
        "benchmark_eval_validity",
        getattr(benchmark_cls, "eval_validity", "smoke_only"),
    )
    agg["benchmark_static_eval_validity"] = getattr(benchmark_cls, "eval_validity", "smoke_only")
    agg["method_integration_level"] = getattr(
        method_descriptor,
        "integration_level",
        "mock",
    )
    # Config may request leaderboard admission, but the Runner's evidence
    # contract decides whether runtime/outcome evidence actually permits it.
    agg["leaderboard_requested"] = cfg.gates.leaderboard_eligible
    return agg


def run_experiment(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    """Execute the experiment: for each seed and (benchmark, method), gate then run."""
    # A strict local-Qwen config must not execute adapter module-level code in
    # its controller just to build a matrix. The worker will load the selected
    # manifest entry after Runner has established its isolation boundary.
    strict_method_boundary = (
        cfg.model.provider == "openai_chat"
        and cfg.model.runtime.require_live_server
        and not cfg.model.runtime.allow_dummy_fallback
    )
    bootstrap_runtime_registries()
    t0 = time.time()
    summaries: list[dict[str, Any]] = []
    seeds: list[int | None] = list(cfg.run.seeds) if cfg.run.seeds else [None]
    n_pairs = len(cfg.matrix.benchmarks) * len(cfg.matrix.methods) * len(seeds)
    print(
        f"[run_experiment] {cfg.name}: {n_pairs} pairs "
        f"(b={len(cfg.matrix.benchmarks)} x m={len(cfg.matrix.methods)} x seeds={len(seeds)})"
    )

    # Pre-resolve pairs so we only do it once even across seed sweeps.
    resolved_pairs = []
    for b in cfg.matrix.benchmarks:
        for m in cfg.matrix.methods:
            b_cls, method_descriptor, resolve_reason = _resolve_pair(
                b,
                m,
                strict_method_boundary=strict_method_boundary,
                method_kwargs=cfg.run.method_kwargs,
            )
            if resolve_reason:
                print(f"--- SKIP {b} × {m}: {resolve_reason}")
                summaries.append(
                    {
                        "benchmark": b,
                        "method": m,
                        "skipped": True,
                        "skip_reason": resolve_reason,
                    }
                )
                continue
            resolved_pairs.append((b, m, b_cls, method_descriptor))

    for b, m, b_cls, method_descriptor in resolved_pairs:
        gate_reason = _gated(
            b_cls,
            method_descriptor,
            cfg.gates,
            method_key=m,
        )
        if gate_reason:
            print(f"--- GATE-SKIP {b} × {m}: {gate_reason}")
            summaries.append(
                {
                    "benchmark": b,
                    "method": m,
                    "skipped": True,
                    "skip_reason": gate_reason,
                    "benchmark_eval_validity": getattr(b_cls, "eval_validity", "smoke_only"),
                    "method_integration_level": getattr(
                        method_descriptor,
                        "integration_level",
                        "mock",
                    ),
                }
            )
            continue

        for seed in seeds:
            seed_label = f"s{seed}" if seed is not None else "s:none"
            print(f"--- RUN {b} × {m} ({seed_label})")
            try:
                agg = _run_one_pair(
                    cfg,
                    b,
                    m,
                    b_cls,
                    method_descriptor,
                    gate_reason,
                    seed,
                )
                summaries.append(agg)
                printable = {
                    k: agg.get(k)
                    for k in (
                        "mean_success",
                        "success_rate",
                        "heuristic_plan_match_rate",
                        "seed",
                    )
                    if agg.get(k) is not None
                }
                print(f"  ok: {json.dumps(printable)}")
            except Exception as e:
                print(f"  ERROR: {redact_text(str(e))}")
                summaries.append(
                    {
                        "benchmark": b,
                        "method": m,
                        "seed": seed,
                        "error": str(e),
                        "trace": traceback.format_exc(),
                    }
                )

    elapsed = time.time() - t0
    print(f"\n[run_experiment] done in {elapsed:.1f}s ({len(summaries)} summaries)")

    # Publish one non-colliding summary per experiment. Keep the historical
    # fixed filename only as a create-once compatibility pointer.
    results_root = Path(cfg.run.results_root)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", cfg.name).strip("-_") or "unnamed"
    summary_path = results_root / f"experiment_summary.{safe_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = redact_artifact(
        {
            "experiment_name": cfg.name,
            "description": cfg.description,
            "model_provider": cfg.model.provider,
            "benchmarks": cfg.matrix.benchmarks,
            "methods": cfg.matrix.methods,
            "seeds": seeds,
            "n_tasks": cfg.run.n_tasks,
            "evaluator": cfg.run.evaluator,
            "gates": {
                "allow_smoke_only": cfg.gates.allow_smoke_only,
                "min_eval_validity": cfg.gates.min_eval_validity,
                "min_integration_level": cfg.gates.min_integration_level,
                "leaderboard_eligible": cfg.gates.leaderboard_eligible,
            },
            "elapsed_s": elapsed,
            "summaries": summaries,
            "seed_aggregates": _aggregate_across_seeds(summaries),
        }
    )
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    payload["receipt_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    summary_path.write_text(serialized)
    compatibility_path = results_root / "experiment_summary.json"
    if not compatibility_path.exists():
        compatibility_path.write_text(serialized)
    print(f"[run_experiment] wrote {summary_path}")
    return summaries


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_experiment.py <experiment.yaml>")
        return 2
    cfg = load_config(sys.argv[1])
    summaries = run_experiment(cfg)
    return 0 if all(s.get("error") is None for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
