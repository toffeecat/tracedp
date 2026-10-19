"""Derive the strongest claim supported by completed run evidence.

This module is a pure policy boundary: it reads already-collected facts and
returns a claim record.  It does not execute models, benchmarks, or storage.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_COMPARISON_MODEL_FIELDS = (
    "model_revision",
    "tokenizer",
    "chat_template",
    "server_version",
)
_COMPARISON_BENCHMARK_FIELDS = (
    "dataset_checksum",
    "dataset_split",
    "environment_fingerprint",
    "container_digest",
    "evaluator_version",
)


def derive_claim_scope(
    *,
    benchmark: Mapping[str, Any],
    method: Mapping[str, Any],
    model_provider: str,
    model_runtime_policy: Mapping[str, Any],
    model_observed: Mapping[str, Any],
    outcome_evidence_qualities: list[str],
    leaderboard_requested: bool,
    run_failed: bool,
    source_contract_verified: bool,
) -> dict[str, Any]:
    """Return the strongest claim scope justified by the supplied evidence."""

    malformed = _malformed_claim_inputs(
        benchmark=benchmark,
        method=method,
        model_provider=model_provider,
        model_runtime_policy=model_runtime_policy,
        model_observed=model_observed,
        outcome_evidence_qualities=outcome_evidence_qualities,
        leaderboard_requested=leaderboard_requested,
        run_failed=run_failed,
        source_contract_verified=source_contract_verified,
    )
    if malformed:
        return _claim(
            leaderboard_requested if type(leaderboard_requested) is bool else False,
            scope="invalid",
            reason="malformed claim evidence: " + ", ".join(malformed),
        )
    if run_failed:
        return _claim(
            leaderboard_requested,
            scope="invalid",
            reason="runtime or execution failure",
        )
    if model_provider == "dummy" or bool(model_observed.get("dummy_model_used")):
        return _claim(
            leaderboard_requested,
            scope="plumbing",
            reason="dummy model is only valid for plumbing",
        )
    if not bool(model_observed.get("model_instantiated")):
        return _claim(
            leaderboard_requested,
            scope="plumbing",
            reason="no model execution was observed",
        )
    if bool(model_observed.get("dummy_fallback_used")):
        return _claim(
            leaderboard_requested,
            scope="plumbing",
            reason="requested model fell back to dummy output",
        )
    if _remote_calls_failed(model_observed.get("remote_error_count", 0)):
        return _claim(
            leaderboard_requested,
            scope="invalid",
            reason="one or more remote model calls failed",
        )

    validity = str(benchmark.get("eval_validity") or "smoke_only")
    integration = str(method.get("integration_level") or "mock")
    if validity in {"smoke_only", "fixture_only"} or integration in {
        "mock",
        "paper_only",
    }:
        return _claim(
            leaderboard_requested,
            scope="plumbing",
            reason="benchmark or method is not comparison-grade",
        )

    strong_outcome = bool(outcome_evidence_qualities) and all(
        quality == "strong" for quality in outcome_evidence_qualities
    )
    if (
        leaderboard_requested
        and validity == "real_native"
        and integration == "native_runner"
        and strong_outcome
    ):
        if not source_contract_verified:
            return _claim(
                True,
                scope="exploratory",
                reason="generic runner lacks a source-locked comparison contract",
            )
        malformed_comparison = _malformed_comparison_provenance(
            benchmark=benchmark,
            model_runtime_policy=model_runtime_policy,
            model_observed=model_observed,
            model_provider=model_provider,
        )
        if malformed_comparison:
            return _claim(
                True,
                scope="invalid",
                reason="malformed comparison evidence: " + ", ".join(malformed_comparison),
            )
        missing = _missing_comparison_evidence(
            benchmark,
            model_runtime_policy,
            model_observed,
            model_provider,
        )
        if missing:
            claim = _claim(
                True,
                scope="exploratory",
                reason="missing comparison evidence: " + ", ".join(missing),
            )
            claim["missing_comparison_evidence"] = missing
            return claim
        return {
            "leaderboard_requested": True,
            "leaderboard_eligible": True,
            "scope": "comparison_candidate",
            "reason": "native method, native benchmark, and strong outcome evidence recorded",
        }

    return _claim(
        leaderboard_requested,
        scope="exploratory",
        reason="run is auditable but lacks the complete native strong-evidence contract",
    )


def _claim(
    leaderboard_requested: bool,
    *,
    scope: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "leaderboard_requested": leaderboard_requested,
        "leaderboard_eligible": False,
        "scope": scope,
        "reason": reason,
    }


def _malformed_claim_inputs(
    *,
    benchmark: Any,
    method: Any,
    model_provider: Any,
    model_runtime_policy: Any,
    model_observed: Any,
    outcome_evidence_qualities: Any,
    leaderboard_requested: Any,
    run_failed: Any,
    source_contract_verified: Any,
) -> list[str]:
    malformed: list[str] = []
    for name, value in (
        ("leaderboard_requested", leaderboard_requested),
        ("run_failed", run_failed),
        ("source_contract_verified", source_contract_verified),
    ):
        if type(value) is not bool:
            malformed.append(name)
    for name, value in (
        ("benchmark", benchmark),
        ("method", method),
        ("model_observed", model_observed),
        ("model_runtime_policy", model_runtime_policy),
    ):
        if not isinstance(value, Mapping):
            malformed.append(name)
    if not isinstance(model_provider, str):
        malformed.append("model_provider")
    if not isinstance(outcome_evidence_qualities, (list, tuple)) or not all(
        isinstance(item, str) for item in outcome_evidence_qualities
    ):
        malformed.append("outcome_evidence_qualities")
    if isinstance(model_observed, Mapping):
        for name in (
            "model_instantiated",
            "dummy_model_used",
            "dummy_fallback_used",
        ):
            if name in model_observed and type(model_observed[name]) is not bool:
                malformed.append(f"model_observed.{name}")
        if "remote_error_count" in model_observed:
            count = model_observed["remote_error_count"]
            if type(count) is not int or count < 0:
                malformed.append("model_observed.remote_error_count")
    return malformed


def _remote_calls_failed(value: Any) -> bool:
    return type(value) is int and value > 0


def _malformed_comparison_provenance(
    *,
    benchmark: Mapping[str, Any],
    model_runtime_policy: Mapping[str, Any],
    model_observed: Mapping[str, Any],
    model_provider: str,
) -> list[str]:
    malformed: list[str] = []
    if model_provider and not model_provider.strip():
        malformed.append("model.provider")
    for prefix, payload, fields in (
        ("model", model_runtime_policy, _COMPARISON_MODEL_FIELDS),
        ("benchmark", benchmark, _COMPARISON_BENCHMARK_FIELDS),
    ):
        for field in fields:
            value = payload.get(field)
            if value not in (None, "") and (not isinstance(value, str) or not value.strip()):
                malformed.append(f"{prefix}.{field}")
    requested_models = model_observed.get("requested_models")
    if requested_models not in (None, (), []) and (
        not isinstance(requested_models, (list, tuple))
        or not requested_models
        or not all(isinstance(item, str) and item.strip() for item in requested_models)
    ):
        malformed.append("model.observed_request")
    return malformed


def _missing_comparison_evidence(
    benchmark: Mapping[str, Any],
    model_runtime_policy: Mapping[str, Any],
    model_observed: Mapping[str, Any],
    model_provider: str,
) -> list[str]:
    """Return provenance fields required before comparison admission."""

    missing: list[str] = []
    if not model_provider:
        missing.append("model.provider")
    for field in _COMPARISON_MODEL_FIELDS:
        if not model_runtime_policy.get(field):
            missing.append(f"model.{field}")
    if not model_observed.get("requested_models"):
        missing.append("model.observed_request")
    for field in _COMPARISON_BENCHMARK_FIELDS:
        if not benchmark.get(field):
            missing.append(f"benchmark.{field}")
    return missing


__all__ = ["derive_claim_scope"]
