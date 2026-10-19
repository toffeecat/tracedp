"""Bind managed-runtime result finalization to Harbor's job lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lh_platform.benchmarks.terminal_managed_runtime_result import (
    finalize_terminal_managed_runtime_result,
    load_harbor_ctrf_evaluation_validity,
)

_JOB_RECEIPT_SCHEMA = "terminal_managed_runtime_job_evaluation_v1"
_TRIAL_STATUS_SCHEMA = "terminal_managed_runtime_trial_finalization_v1"


def _read_object(path: Path) -> dict[str, Any]:
    """Read one required JSON object without accepting scalar or list payloads."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _write_object(path: Path, value: dict[str, object]) -> None:
    """Write one local audit artifact only after its complete value is available."""

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".writing")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    staging.replace(path)


def _trial_directory(job_dir: Path, trial_name: str) -> Path:
    """Resolve a Harbor-owned trial name without permitting path traversal."""

    if not trial_name or Path(trial_name).name != trial_name:
        raise ValueError("Harbor result contains an invalid trial name")
    trial = job_dir / trial_name
    if not trial.is_dir():
        raise ValueError(f"managed runtime trial directory is missing: {trial_name}")
    return trial


def finalize_terminal_managed_runtime_job(
    job_dir: Path,
    *,
    trial_results: Sequence[Mapping[str, object]] | None = None,
    allow_legacy_projection: bool = False,
) -> dict[str, object]:
    """Revalidate and finalize every Harbor trial before accepting a whole job."""

    job_dir = job_dir.resolve()
    job_result = _read_object(job_dir / "result.json")
    expected_count = job_result.get("n_total_trials")
    stats = job_result.get("stats")
    if (
        job_result.get("finished_at") is None
        or not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
        or not isinstance(stats, dict)
        or stats.get("n_completed_trials") != expected_count
        or stats.get("n_errored_trials") != 0
        or stats.get("n_running_trials") != 0
        or stats.get("n_pending_trials") != 0
        or stats.get("n_cancelled_trials") != 0
    ):
        raise ValueError("Harbor job result is incomplete")

    # Harbor deliberately excludes trial_results from the durable job result.
    # The live plugin supplies them; the standalone controller reconstructs
    # the same list only from direct child trial result files.
    if trial_results is None:
        reconstructed: list[dict[str, object]] = []
        for candidate in sorted(job_dir.iterdir()):
            if not candidate.is_dir() or candidate.name.startswith("."):
                continue
            result_path = candidate / "result.json"
            if not result_path.is_file():
                continue
            reconstructed.append(_read_object(result_path))
        trial_results = reconstructed
    if len(trial_results) != expected_count:
        raise ValueError("Harbor job trial result count differs")

    trial_names: list[str] = []
    for item in trial_results:
        if not isinstance(item, Mapping) or not isinstance(item.get("trial_name"), str):
            raise ValueError("Harbor job contains an invalid trial result")
        trial_names.append(str(item["trial_name"]))
    if len(trial_names) != len(set(trial_names)):
        raise ValueError("Harbor job contains duplicate trial names")

    receipts: list[dict[str, object]] = []
    for trial_name in trial_names:
        trial = _trial_directory(job_dir, trial_name)
        receipt = finalize_terminal_managed_runtime_result(
            trial,
            evaluation_validity=load_harbor_ctrf_evaluation_validity(trial),
            allow_legacy_projection=allow_legacy_projection,
        )
        receipts.append(
            {
                "trial_name": trial_name,
                "runtime_method": receipt["runtime_method"],
                "official_reward": receipt["official_reward"],
                "official_success": receipt["official_success"],
                "token_accounting_complete": receipt["token_accounting_complete"],
                "token_accounting_semantics": receipt[
                    "token_accounting_semantics"
                ],
                "scientific_result": receipt["scientific_result"],
                "formal_experiment_authorized": receipt[
                    "formal_experiment_authorized"
                ],
                "admissible_claim_scope": receipt["admissible_claim_scope"],
                "receipt_path": (
                    f"{trial_name}/agent/runtime-evaluation-receipt.json"
                ),
            }
        )

    job_receipt: dict[str, object] = {
        "schema_version": _JOB_RECEIPT_SCHEMA,
        "job_id": str(job_result.get("id", "")),
        "trial_count": expected_count,
        "all_trials_finalized": True,
        "allow_legacy_projection": allow_legacy_projection,
        "trials": receipts,
    }
    receipt_path = job_dir / "managed-runtime-evaluation-receipt.json"
    if receipt_path.exists() and _read_object(receipt_path) != job_receipt:
        raise ValueError("managed runtime job evaluation receipt differs")
    if not receipt_path.exists():
        _write_object(receipt_path, job_receipt)
    return job_receipt


class ManagedRuntimeFinalizationPlugin:
    """Finalize each trial at Harbor END and audit the complete job at job END."""

    def __init__(self, *, allow_legacy_projection: bool = False) -> None:
        self.allow_legacy_projection = allow_legacy_projection
        self._job_dir: Path | None = None

    async def on_job_start(self, job: Any) -> None:
        """Register through Harbor's public trial-ended hook before work starts."""

        self._job_dir = Path(job.job_dir).resolve()
        job.on_trial_ended(self._on_trial_ended)

    async def _on_trial_ended(self, event: Any) -> None:
        """Seal one completed Harbor trial and preserve any rejection as evidence."""

        if self._job_dir is None:
            raise RuntimeError("managed runtime plugin was not attached to a job")
        trial_name = str(event.trial_name)
        trial = _trial_directory(self._job_dir, trial_name)
        status_path = trial / "agent/runtime-finalization-status.json"
        try:
            receipt = finalize_terminal_managed_runtime_result(
                trial,
                evaluation_validity=load_harbor_ctrf_evaluation_validity(trial),
                allow_legacy_projection=self.allow_legacy_projection,
            )
        except Exception as error:
            _write_object(
                status_path,
                {
                    "schema_version": _TRIAL_STATUS_SCHEMA,
                    "trial_name": trial_name,
                    "status": "rejected",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
            )
            return
        _write_object(
            status_path,
            {
                "schema_version": _TRIAL_STATUS_SCHEMA,
                "trial_name": trial_name,
                "status": "finalized",
                "runtime_method": receipt["runtime_method"],
                "official_reward": receipt["official_reward"],
            },
        )

    async def on_job_end(self, job_result: Any) -> None:
        """Create the job receipt; the controller independently checks its presence."""

        if self._job_dir is None:
            raise RuntimeError("managed runtime plugin was not attached to a job")
        live_trial_results = [
            item.model_dump(mode="json")
            for item in job_result.trial_results
        ]
        finalize_terminal_managed_runtime_job(
            self._job_dir,
            trial_results=live_trial_results,
            allow_legacy_projection=self.allow_legacy_projection,
        )


__all__ = [
    "ManagedRuntimeFinalizationPlugin",
    "finalize_terminal_managed_runtime_job",
]
