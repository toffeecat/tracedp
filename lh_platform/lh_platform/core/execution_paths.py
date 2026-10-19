"""Canonical vocabulary for requested and observed execution paths.

Execution paths describe *how work actually ran*.  They are distinct from a
method's static integration level and from a study's training path.
"""

from __future__ import annotations

from enum import StrEnum


class ExecutionPath(StrEnum):
    """Execution routes understood by run evidence and study protocols."""

    AUTO = "auto"
    BENCHMARK_STEP_LOOP = "benchmark_step_loop"
    PROMPT_WRAPPER = "prompt_wrapper"
    TEST_TIME_OPTIMIZER = "test_time_optimizer"
    UPSTREAM_NATIVE_RUNNER = "upstream_native_runner"
    UPSTREAM_TRAINING_RECIPE = "upstream_training_recipe"
    TRAINED_ARTIFACT_EVAL = "trained_artifact_eval"
    MIXED = "mixed_execution_paths"


EXECUTION_AUTO = ExecutionPath.AUTO.value
EXECUTION_BENCHMARK_STEP_LOOP = ExecutionPath.BENCHMARK_STEP_LOOP.value
EXECUTION_PROMPT_WRAPPER = ExecutionPath.PROMPT_WRAPPER.value
EXECUTION_TEST_TIME_OPTIMIZER = ExecutionPath.TEST_TIME_OPTIMIZER.value
EXECUTION_UPSTREAM_NATIVE_RUNNER = ExecutionPath.UPSTREAM_NATIVE_RUNNER.value
EXECUTION_UPSTREAM_TRAINING_RECIPE = ExecutionPath.UPSTREAM_TRAINING_RECIPE.value
EXECUTION_TRAINED_ARTIFACT_EVAL = ExecutionPath.TRAINED_ARTIFACT_EVAL.value
EXECUTION_MIXED = ExecutionPath.MIXED.value


__all__ = [
    "EXECUTION_AUTO",
    "EXECUTION_BENCHMARK_STEP_LOOP",
    "EXECUTION_MIXED",
    "EXECUTION_PROMPT_WRAPPER",
    "EXECUTION_TEST_TIME_OPTIMIZER",
    "EXECUTION_TRAINED_ARTIFACT_EVAL",
    "EXECUTION_UPSTREAM_NATIVE_RUNNER",
    "EXECUTION_UPSTREAM_TRAINING_RECIPE",
    "ExecutionPath",
]
