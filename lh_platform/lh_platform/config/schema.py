"""Config schema for the lh_platform experiment runner.

A config file (YAML or JSON) describes one experiment: a model profile, a
matrix of (benchmark, method) pairs, run settings, and gating rules.

The schema separates model constructor kwargs from runtime proof. The runner
forwards `model.kwargs` to the registry, while `model.runtime` is validated
before the task loop. This prevents a local Qwen/vLLM experiment from being
silently recorded as successful after an unavailable server falls back to a
dummy response.

YAML example:

    experiment:
      name: local_qwen_webvoyager_debug
      description: 4090 local Qwen3-4B driving WebVoyager real_partial

    model:
      provider: openai_chat
      path: configs/models/qwen3_4b_vllm.yaml      # optional; otherwise inline `kwargs`
      kwargs:
        model: qwen3-4b
        base_url: http://127.0.0.1:8000/v1
        api_key: dummy
        temperature: 0.0
        max_tokens: 512
      runtime:
        require_live_server: true
        allow_dummy_fallback: false
        expected_model: qwen3-4b

    matrix:
      benchmarks: [webvoyager]
      methods:    [webvoyager_native, full_history]

    run:
      n_tasks: 3
      evaluator: long_horizon
      results_root: results/local_qwen_webvoyager
      seeds: [0]                                  # forwarded as config_extra seed

    gates:
      min_eval_validity: real_loader
      min_integration_level: mock
      allow_smoke_only: false
      leaderboard_eligible: false
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lh_platform.core.method_capabilities import (
    METHOD_INTEGRATION_LEVELS,
    METHOD_INTEGRATION_RANK,
)
from lh_platform.core.model_runtime import ModelRuntimePolicy

# YAML schema version. Bump when the file format changes incompatibly.
SCHEMA_VERSION = 1

VALID_EVAL_VALIDITIES = (
    "smoke_only",
    "fixture_only",
    "unverified_loader",
    "real_loader",
    "real_partial",
    "real_native",
)

VALID_INTEGRATION_LEVELS = METHOD_INTEGRATION_LEVELS
VALID_EVAL_VALIDITY_ORDER_TEXT = " < ".join(VALID_EVAL_VALIDITIES)
VALID_INTEGRATION_ORDER_TEXT = " < ".join(VALID_INTEGRATION_LEVELS)
VALID_INTEGRATION_CHOICES_TEXT = "|".join(VALID_INTEGRATION_LEVELS)


@dataclass
class ModelConfig:
    provider: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    runtime: ModelRuntimePolicy = field(default_factory=ModelRuntimePolicy)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        if not isinstance(data, dict):
            raise ConfigError(f"model: expected mapping, got {type(data).__name__}")
        provider = data.get("provider")
        if not provider:
            raise ConfigError("model.provider is required (e.g. 'openai_chat', 'dummy')")
        kwargs = data.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            raise ConfigError("model.kwargs must be a mapping")
        runtime_raw = data.get("runtime") or {}
        if not isinstance(runtime_raw, dict):
            raise ConfigError("model.runtime must be a mapping")
        try:
            runtime = ModelRuntimePolicy.from_mapping(
                runtime_raw,
                default_expected_model=str(kwargs.get("model") or "").strip() or None,
            )
        except ValueError as error:
            raise ConfigError(str(error)) from error
        return cls(provider=provider, kwargs=dict(kwargs), runtime=runtime)


@dataclass
class MatrixConfig:
    benchmarks: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatrixConfig:
        if not isinstance(data, dict):
            raise ConfigError(f"matrix: expected mapping, got {type(data).__name__}")
        benchmarks = data.get("benchmarks") or []
        methods = data.get("methods") or []
        if not isinstance(benchmarks, list) or not benchmarks:
            raise ConfigError("matrix.benchmarks must be a non-empty list")
        if not isinstance(methods, list) or not methods:
            raise ConfigError("matrix.methods must be a non-empty list")
        for n in benchmarks:
            if not isinstance(n, str) or not n:
                raise ConfigError(f"matrix.benchmarks: invalid entry {n!r}")
        for n in methods:
            if not isinstance(n, str) or not n:
                raise ConfigError(f"matrix.methods: invalid entry {n!r}")
        return cls(benchmarks=list(benchmarks), methods=list(methods))


@dataclass
class RunConfig:
    n_tasks: int = 3
    evaluator: str = "long_horizon"
    results_root: str = "results"
    seeds: list[int] = field(default_factory=list)
    benchmark_kwargs: dict[str, Any] = field(default_factory=dict)
    method_kwargs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigError(f"run: expected mapping, got {type(data).__name__}")
        seeds = data.get("seeds") or []
        if not isinstance(seeds, list):
            raise ConfigError("run.seeds must be a list")
        seeds = [int(s) for s in seeds]
        benchmark_kwargs = data.get("benchmark_kwargs") or {}
        if not isinstance(benchmark_kwargs, dict):
            raise ConfigError("run.benchmark_kwargs must be a mapping")
        method_kwargs = data.get("method_kwargs") or {}
        if not isinstance(method_kwargs, dict):
            raise ConfigError("run.method_kwargs must be a mapping")
        extra = {
            k: v
            for k, v in data.items()
            if k
            not in (
                "n_tasks",
                "evaluator",
                "results_root",
                "seeds",
                "benchmark_kwargs",
                "method_kwargs",
            )
        }
        return cls(
            n_tasks=int(data.get("n_tasks", 3)),
            evaluator=str(data.get("evaluator", "long_horizon")),
            results_root=str(data.get("results_root", "results")),
            seeds=seeds,
            benchmark_kwargs=dict(benchmark_kwargs),
            method_kwargs=method_kwargs,
            extra=extra,
        )


@dataclass
class GatesConfig:
    """Validity gates. Evaluated by the runner before any work begins.

    `allow_smoke_only`: when False (default), any (benchmark, method) pair
        whose benchmark.eval_validity == 'smoke_only' is skipped.
    `min_eval_validity`: minimum validity required to enter the run.
        When set, any pair whose benchmark.eval_validity ranks below this
        tier is skipped. The rank order is:
            smoke_only < fixture_only < unverified_loader < real_loader < real_partial < real_native
    `min_integration_level`: same idea, applied to the method's
        integration_level. Rank order:
            mock < paper_only < paper-faithful < repo_import < native_runner
    `leaderboard_eligible`: passed through into metrics.aggregate so
        downstream summary tables can filter on it.
    """

    allow_smoke_only: bool = False
    min_eval_validity: str = "smoke_only"
    min_integration_level: str = "mock"
    leaderboard_eligible: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GatesConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigError(f"gates: expected mapping, got {type(data).__name__}")
        mev = str(data.get("min_eval_validity", "smoke_only"))
        if mev not in VALID_EVAL_VALIDITIES:
            raise ConfigError(
                f"gates.min_eval_validity must be one of {VALID_EVAL_VALIDITIES}; got {mev!r}"
            )
        mil = str(data.get("min_integration_level", "mock"))
        if mil not in VALID_INTEGRATION_LEVELS:
            raise ConfigError(
                f"gates.min_integration_level must be one of {VALID_INTEGRATION_LEVELS}; got {mil!r}"
            )
        return cls(
            allow_smoke_only=bool(data.get("allow_smoke_only", False)),
            min_eval_validity=mev,
            min_integration_level=mil,
            leaderboard_eligible=bool(data.get("leaderboard_eligible", False)),
        )


@dataclass
class ExperimentConfig:
    name: str
    description: str
    model: ModelConfig
    matrix: MatrixConfig
    run: RunConfig
    gates: GatesConfig
    raw: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    schema_version: int = SCHEMA_VERSION

    def to_runner_kwargs(self) -> dict[str, Any]:
        """Serialize to kwargs that the existing Runner understands.

        Returns a dict that the runner can use to construct each
        (benchmark, method, model, evaluator) configuration. The runner
        uses these kwargs as the source of truth and never re-parses the
        config file.
        """
        out = {
            "experiment_name": self.name,
            "experiment_description": self.description,
            "model_provider": self.model.provider,
            "model_kwargs": dict(self.model.kwargs),
            "model_runtime": self.model.runtime.to_dict(),
            "benchmark_kwargs": dict(self.run.benchmark_kwargs),
            "method_kwargs": dict(self.run.method_kwargs),
            "benchmarks": list(self.matrix.benchmarks),
            "methods": list(self.matrix.methods),
            "n_tasks": self.run.n_tasks,
            "evaluator": self.run.evaluator,
            "results_root": self.run.results_root,
            "seeds": list(self.run.seeds),
            "gates": {
                "allow_smoke_only": self.gates.allow_smoke_only,
                "min_eval_validity": self.gates.min_eval_validity,
                "min_integration_level": self.gates.min_integration_level,
                "leaderboard_eligible": self.gates.leaderboard_eligible,
            },
        }
        if self.run.extra:
            out["run_extra"] = dict(self.run.extra)
        return out


class ConfigError(ValueError):
    """Raised when an experiment config file is malformed."""


# Validity / integration rank tables shared with the runner.
VALIDITY_RANK = {v: i for i, v in enumerate(VALID_EVAL_VALIDITIES)}
INTEGRATION_RANK = dict(METHOD_INTEGRATION_RANK)
