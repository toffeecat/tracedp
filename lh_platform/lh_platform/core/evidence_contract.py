"""Fail-closed runtime and provenance contracts for experiment runs.

Static adapter labels are useful admission hints, but they cannot establish
what actually ran.  This module records the concrete evidence required to
interpret a run: resolved configuration, code revision, model-server identity,
dataset source, and the claim boundary supported by the observed outcome.

The design intentionally keeps secrets out of artifacts.  A run may keep an
API key in process memory, but never writes it to ``run.json`` or the evidence
contract.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from lh_platform.core.artifact_redaction import (
    redact_artifact,
)
from lh_platform.core.artifact_redaction import (
    redact_secrets as redact_secrets,
)
from lh_platform.core.artifact_redaction import (
    redact_text as redact_text,
)
from lh_platform.core.claim_scope import derive_claim_scope
from lh_platform.core.code_provenance import git_provenance as git_provenance
from lh_platform.core.model_runtime import (
    RUNTIME_CONTRACT_SCHEMA as RUNTIME_CONTRACT_SCHEMA,
)
from lh_platform.core.model_runtime import (
    ModelRuntimeEvidence as ModelRuntimeEvidence,
)
from lh_platform.core.model_runtime import (
    ModelRuntimePolicy as ModelRuntimePolicy,
)
from lh_platform.core.model_runtime import (
    ModelRuntimePreflightError as ModelRuntimePreflightError,
)
from lh_platform.core.model_runtime import (
    failed_preflight_evidence as failed_preflight_evidence,
)
from lh_platform.core.model_runtime import (
    preflight_model_runtime as preflight_model_runtime,
)
from lh_platform.core.model_runtime import (
    skipped_preflight_evidence as skipped_preflight_evidence,
)
from lh_platform.core.model_runtime import (
    split_model_kwargs_and_runtime as split_model_kwargs_and_runtime,
)

EXPERIMENT_EVIDENCE_SCHEMA = "lh_experiment_evidence_contract/v1"
# A completed runner contract binds the exact per-task trajectory files that
# were materialized before the contract was finalized.  Collection code must
# consume this manifest instead of discovering whatever files happen to be in
# a run directory later.
RUNNER_TRAJECTORY_MANIFEST_SCHEMA = "lh_runner_trajectory_manifest/v1"


class RuntimeAdmissionError(RuntimeError):
    """Raised when actual benchmark evidence violates a config admission gate."""


def canonical_sha256(value: Any) -> str:
    """Hash a normalized, redacted JSON value for reproducibility checks."""

    encoded = json.dumps(
        redact_artifact(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_runtime_evidence(benchmark: Any) -> dict[str, Any]:
    """Capture the data/environment facts a benchmark instance actually exposes."""

    dataset_path = getattr(benchmark, "dataset_path", None)
    resolved_dataset_path: Path | None = None
    if dataset_path:
        try:
            candidate = Path(dataset_path)
            if candidate.is_file():
                resolved_dataset_path = candidate
        except (TypeError, ValueError):
            resolved_dataset_path = None
    evidence = {
        "data_source": getattr(benchmark, "data_source", "not_declared"),
        "has_real_data": bool(getattr(benchmark, "has_real_data", False)),
        "supports_native_eval": bool(getattr(benchmark, "supports_native_eval", False)),
        "has_native_harness": bool(getattr(benchmark, "has_native_harness", False)),
        "is_mock": bool(getattr(benchmark, "is_mock", True)),
        "eval_validity": str(getattr(benchmark, "eval_validity", "smoke_only")),
        # Some official adapters begin as a non-promoted loader and can only
        # establish their final validity after a local API probe in load().
        # Runner uses this explicit ceiling to defer the *pre-load* gate
        # without ever allowing model preflight before the post-load gate.
        "runtime_eval_validity_dynamic": bool(
            getattr(benchmark, "runtime_eval_validity_dynamic", False)
        ),
        "runtime_eval_validity_max": getattr(benchmark, "runtime_eval_validity_max", None),
        "required_action_contract": str(getattr(benchmark, "required_action_contract", "") or ""),
        # These fields are deliberately explicit even when the generic
        # adapter cannot provide them. A missing field blocks comparison
        # claims instead of being silently mistaken for provenance.
        "dataset_path": str(resolved_dataset_path) if resolved_dataset_path else None,
        "dataset_checksum": (
            f"sha256:{_sha256_file(resolved_dataset_path)}" if resolved_dataset_path else None
        ),
        "dataset_checksum_source": "file_content" if resolved_dataset_path else None,
        "dataset_split": getattr(benchmark, "dataset_split", None),
        "task_selection_strategy": getattr(benchmark, "task_selection_strategy", "not_declared"),
        "task_selection_seed": getattr(benchmark, "task_selection_seed", None),
        # A full benchmark file needs more than a byte hash: its source
        # revision and downloaded-data manifest bind the file to the public
        # dataset release.  Generic adapters simply leave these as None.
        "dataset_revision": getattr(benchmark, "dataset_revision", None),
        # Some full-data adapters additionally bind local files to an
        # independently reviewed upstream source lock. Keep that fact in the
        # generic contract so a manifest checksum cannot be mistaken for
        # release provenance.
        "dataset_source_verified": getattr(benchmark, "dataset_source_verified", None),
        "dataset_source_verification_reason": getattr(
            benchmark, "dataset_source_verification_reason", None
        ),
        "dataset_source_lock_sha256": getattr(benchmark, "dataset_source_lock_sha256", None),
        # A ToolBench remote receipt is only comparable as an environment
        # interaction when the exact HTTPS proxy was independently reviewed.
        # Keep this generic so provenance cannot disappear into adapter-only
        # logs.
        "remote_proxy_identity_verified": getattr(
            benchmark, "remote_proxy_identity_verified", None
        ),
        "remote_proxy_identity_reason": getattr(benchmark, "remote_proxy_identity_reason", None),
        "remote_proxy_identity_lock_sha256": getattr(
            benchmark, "remote_proxy_identity_lock_sha256", None
        ),
        "dataset_manifest_path": (
            str(benchmark.dataset_manifest_path)
            if getattr(benchmark, "dataset_manifest_path", None)
            else None
        ),
        "dataset_manifest_sha256": getattr(benchmark, "dataset_manifest_sha256", None),
        "environment_fingerprint": getattr(benchmark, "environment_fingerprint", None),
        "container_digest": getattr(benchmark, "container_digest", None),
        "evaluator_version": getattr(benchmark, "evaluator_version", None),
    }
    return evidence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_experiment_evidence_contract(
    *,
    run_id: str,
    resolved_config: Mapping[str, Any],
    model_provider: str,
    model_runtime_policy: ModelRuntimePolicy,
    model_preflight: ModelRuntimeEvidence,
    benchmark_id: str,
    benchmark_evidence: Mapping[str, Any],
    method_id: str,
    method_integration_level: str,
    method_strict_proxy_compatible: bool = False,
    method_supported_action_contracts: tuple[str, ...] | None = None,
    evaluator_id: str,
    seed: int | None,
    leaderboard_requested: bool,
) -> dict[str, Any]:
    """Build the initial, auditable artifact for one concrete run."""

    sanitized_config = redact_artifact(resolved_config)
    return {
        "schema": EXPERIMENT_EVIDENCE_SCHEMA,
        "run_id": run_id,
        "generated_at": time.time(),
        "run_status": "started",
        "code": git_provenance(),
        "config": {
            "source_path": resolved_config.get("config_source"),
            "resolved_config_sha256": canonical_sha256(sanitized_config),
            "resolved_config": sanitized_config,
        },
        "model": {
            "provider": model_provider,
            "requested_model": model_preflight.requested_model,
            "runtime_policy": model_runtime_policy.to_dict(),
            "preflight": model_preflight.to_dict(),
            "observed": {},
        },
        "benchmark": {
            "id": benchmark_id,
            **dict(benchmark_evidence),
        },
        "method": {
            "id": method_id,
            "integration_level": method_integration_level,
            "strict_proxy_compatible": bool(method_strict_proxy_compatible),
            "supported_action_contracts": (
                list(method_supported_action_contracts)
                if method_supported_action_contracts is not None
                else None
            ),
        },
        "evaluator": {"id": evaluator_id},
        "seed": seed,
        "claim": {
            "leaderboard_requested": bool(leaderboard_requested),
            "leaderboard_eligible": False,
            "scope": "pending",
            "reason": "run has not completed",
        },
    }


def complete_experiment_evidence_contract(
    contract: MutableMapping[str, Any],
    *,
    model_observed: Mapping[str, Any],
    outcome_evidence_qualities: list[str],
    run_failed: bool,
    run_status: str | None = None,
    source_contract_verified: bool = False,
) -> dict[str, Any]:
    """Finalize observed evidence and derive the strongest allowed claim scope."""

    payload = dict(contract)
    raw_model = payload.get("model")
    model_is_mapping = raw_model is None or isinstance(raw_model, Mapping)
    model = dict(raw_model or {}) if model_is_mapping else {}
    sanitized_observed = redact_artifact(model_observed)
    model["observed"] = sanitized_observed
    payload["model"] = model
    raw_claim = payload.get("claim")
    if "claim" not in payload:
        leaderboard_requested = False
    elif isinstance(raw_claim, Mapping):
        leaderboard_requested = raw_claim.get("leaderboard_requested", False)
    else:
        leaderboard_requested = raw_claim
    raw_benchmark = payload.get("benchmark")
    benchmark = {} if raw_benchmark is None else raw_benchmark
    raw_method = payload.get("method")
    method = {} if raw_method is None else raw_method
    raw_runtime_policy = model.get("runtime_policy") if model_is_mapping else raw_model
    model_runtime_policy = {} if raw_runtime_policy is None else raw_runtime_policy
    claim = derive_claim_scope(
        benchmark=benchmark,
        method=method,
        model_provider=model.get("provider", "") if model_is_mapping else raw_model,
        model_runtime_policy=model_runtime_policy,
        model_observed=sanitized_observed,
        outcome_evidence_qualities=outcome_evidence_qualities,
        leaderboard_requested=leaderboard_requested,
        run_failed=run_failed,
        source_contract_verified=source_contract_verified,
    )
    payload["claim"] = claim
    payload["run_status"] = run_status or ("failed" if run_failed else "completed")
    payload["completed_at"] = time.time()
    payload["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    return payload
