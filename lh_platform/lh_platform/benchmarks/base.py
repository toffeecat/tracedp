"""Benchmark base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lh_platform.core.outcome import OutcomeEvidence
from lh_platform.core.types import Action, Observation, Task

# Benchmark validity levels:
#   smoke_only   : step() is a mock, results cannot be used for SOTA comparison
#   fixture_only : local format/contract fixture, never released benchmark data
#   unverified_loader : complete local data staged, but its upstream-release
#                       provenance has not been checked against a canonical lock
#   real_loader  : tasks come from real data, but step() is still mocked
#   real_partial : tasks real AND step() partially talks to the real env
#                  (e.g. read-only or single-website subset)
#   real_native  : tasks real AND step() invokes upstream benchmark harness
EVAL_SMOKE = "smoke_only"
EVAL_FIXTURE = "fixture_only"
EVAL_UNVERIFIED_LOADER = "unverified_loader"
EVAL_REAL_LOADER = "real_loader"
EVAL_REAL_PARTIAL = "real_partial"
EVAL_REAL_NATIVE = "real_native"


class BenchmarkBase(ABC):
    name: str = "base"

    # Empty means that this adapter has no machine-checkable action grammar.
    # A non-empty value is a hard compatibility requirement for methods that
    # declare their own non-generic action space (for example CodeAct's
    # Python-expression policy).  It is intentionally separate from
    # ``TaskDomain``: two TOOL benchmarks can expose incompatible grammars.
    required_action_contract: str = ""

    # Validity flags -- subclasses override:
    eval_validity: str = EVAL_SMOKE
    has_real_data: bool = False  # tasks come from cloned repo data files
    supports_native_eval: bool = False  # step() talks to real env
    has_native_harness: bool = False  # has run.py / eval.py / docker harness
    is_mock: bool = True  # step() is currently a mock
    load_path_used: str = ""  # set by load()/load_seeded() for lineage

    @abstractmethod
    def load(self, n: int = 3) -> list[Task]:
        """Return the first N tasks."""
        ...

    def step(self, task: Task, action: Action) -> Observation:
        """Apply the action in the environment and return an Observation.

        The default mock implementation just echoes the action's raw text.
        Subclasses should override to talk to the real benchmark environment.
        """
        return Observation(
            content=f"[{self.name} mock] received action: {action.raw[:120]}",
            info={"done": False, "continue": True},
        )

    def close(self, artifacts_dir: Path | None = None) -> None:  # noqa: B027  # intentional no-op default cleanup hook; subclasses override when they hold resources
        """Clean up resources and optionally write artifacts to the run directory.

        Args:
            artifacts_dir: Path to the run's artifacts/ directory. Benchmarks
                that produce artifacts should write them here when provided.
        """
        pass

    def abort(self) -> None:
        """Release live resources without receiving an artifact filesystem path.

        ``close(artifacts_dir=...)`` mixes two responsibilities inherited from
        early adapters: process/browser/JVM shutdown and artifact export.  A
        run-directory path is intentionally withheld if its inode no longer
        matches Runner's held storage FD.  Native adapters that own a live
        resource must therefore make this no-artifact cleanup path safe and
        idempotent; adapters needing custom cleanup should override it.
        """

        self.close()

    def bind_run_artifacts(self, artifacts_dir: Path) -> None:
        """Bind the immutable artifact sink before benchmark admission.

        Most adapters do not need an early binding and keep this no-op.  A
        benchmark whose native claim depends on a run receipt may override it
        and refuse promotion until Runner has supplied the per-run sink.
        """

        del artifacts_dir

    def finalize_run_evidence(
        self,
        *,
        run_id: str,
        trajectory_manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Bind benchmark-native receipts to the completed runner artifacts.

        ``close()`` happens before Runner serializes its task-trajectory
        manifest.  Native benchmarks that emit a receipt can override this
        hook to add the immutable ``run_id`` and manifest hash afterwards.
        The return value must use the receipt envelope validated by Runner:

        ``{"receipts": [{"name", "schema", "path", "sha256",
        "run_id", "trajectory_manifest_sha256"}]}``.

        Runner verifies the descriptor's binding, opens and snapshots the
        declared receipt bytes without following symlinks, and records that
        snapshot in its contract.  It does not infer an adapter-specific
        receipt schema: an adapter must validate its own canonical receipt
        contents before returning the descriptor.

        The default makes legacy and smoke adapters no-ops; it deliberately
        does *not* manufacture provenance for them.
        """

        del run_id, trajectory_manifest
        return {}

    def outcome_evidence(self, task: Task) -> OutcomeEvidence:
        """Return benchmark-owned Outcome Evidence for a finished task."""
        return OutcomeEvidence.fallback(False)

    def task_metrics(self, task: Task, trajectory: object) -> dict[str, object]:
        """Return additional benchmark-owned metrics for one completed task.

        The Runner accepts only ``environment_*`` / ``benchmark_*`` keys from
        this hook, so an adapter cannot overwrite canonical success or
        credibility evidence.  The default is deliberately empty: a benchmark
        must opt in before it contributes environment-specific metrics.
        """

        del task, trajectory
        return {}
