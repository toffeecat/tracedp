"""Configuration resolution and fail-closed admission for Runner."""

from __future__ import annotations

import time
from typing import Any

from lh_platform.core.artifact_redaction import redact_secrets
from lh_platform.core.evidence_contract import complete_experiment_evidence_contract
from lh_platform.core.model_runtime import (
    ModelRuntimePolicy,
    split_model_kwargs_and_runtime,
)
from lh_platform.core.storage import RunStorage


class RunnerAdmissionMixin:
    def _resolved_model_runtime(self) -> tuple[dict[str, Any], ModelRuntimePolicy]:
        """Normalize all config surfaces into one model/runtime contract."""

        raw_kwargs = dict(self.config_extra.get("model_kwargs", {}))
        model_kwargs, embedded_runtime = split_model_kwargs_and_runtime(raw_kwargs)
        if self.model_alias and self.model_alias != self.model_id and "model" not in model_kwargs:
            model_kwargs["model"] = self.model_alias
        runtime_raw = dict(embedded_runtime)
        configured_runtime = self.config_extra.get("model_runtime", {})
        if configured_runtime:
            if not isinstance(configured_runtime, dict):
                raise ValueError("config_extra.model_runtime must be a mapping")
            runtime_raw.update(configured_runtime)
        policy = ModelRuntimePolicy.from_mapping(
            runtime_raw,
            default_expected_model=str(model_kwargs.get("model") or "").strip() or None,
        )
        # The runtime policy must control the model implementation too.  This
        # prevents a strict preflight from passing while a later model call
        # quietly substitutes a dummy answer.
        if self.model_id == "openai_chat":
            model_kwargs["allow_dummy_fallback"] = policy.allow_dummy_fallback
        if self.seed is not None:
            model_kwargs.setdefault("seed", self.seed)
        return model_kwargs, policy

    def _resolved_config_snapshot(
        self,
        *,
        model_kwargs: dict[str, Any],
        runtime_policy: ModelRuntimePolicy,
    ) -> dict[str, Any]:
        """Return the complete configuration whose hash names this run."""

        return {
            "config_source": self.config_extra.get("config_source"),
            "schema_version": self.config_extra.get("schema_version"),
            "experiment_name": self.config_extra.get("experiment_name"),
            "experiment_description": self.config_extra.get("experiment_description"),
            "benchmark": self.benchmark_id,
            "method": self.method_id,
            "model_provider": self.model_id,
            "model_alias": self.model_alias,
            "model_kwargs": dict(model_kwargs),
            "model_runtime": runtime_policy.to_dict(),
            "benchmark_kwargs": dict(self.benchmark_kwargs),
            "evaluator": self.evaluator_id,
            "n_tasks": self.n_tasks,
            "seed": self.seed,
            "method_kwargs": dict(self.method_kwargs),
            "gates": dict(self.config_extra.get("gates", {})),
            "run_extra": redact_secrets(self.config_extra.get("run_extra", {})),
        }

    def _requires_process_generation_authority(
        self,
        runtime_policy: ModelRuntimePolicy,
    ) -> bool:
        """Return whether this run needs isolated model provenance.

        A strict OpenAI-compatible profile is how local vLLM/Qwen experiments
        declare that a live endpoint is required and dummy fallback is
        forbidden.  Those are the only runs that can later become an official
        SWE-bench prediction bundle, so they must not share a mutable model
        ledger with method/plugin code.
        """

        return (
            self.model_id == "openai_chat"
            and bool(runtime_policy.require_live_server)
            and not bool(runtime_policy.allow_dummy_fallback)
        )

    def _runtime_benchmark_gate_reason(
        self,
        evidence: dict[str, Any],
        *,
        defer_dynamic_admission: bool = False,
    ) -> str:
        """Re-apply config validity gates to the instantiated benchmark.

        Adapter classes express intended capability. A local clone can still
        be absent, causing an instance to downgrade from ``real_partial`` to
        ``smoke_only``. Config-driven experiments must not spend model budget
        on that fallback merely because the class-level label looked valid.
        Direct legacy ``Runner`` calls without a gates mapping keep their
        historical behavior.
        """

        gates = self.config_extra.get("gates")
        if not isinstance(gates, dict):
            return ""
        from lh_platform.config.schema import VALIDITY_RANK

        validity = str(evidence.get("eval_validity") or "smoke_only")
        allow_smoke = gates.get("allow_smoke_only", False)
        if isinstance(allow_smoke, str):
            allow_smoke = allow_smoke.strip().lower() in {"1", "true", "yes"}
        min_validity = str(gates.get("min_eval_validity") or "smoke_only")
        current_rank = VALIDITY_RANK.get(validity, -1)
        minimum_rank = VALIDITY_RANK.get(min_validity, 0)
        needs_promotion = (
            validity == "smoke_only" and not bool(allow_smoke)
        ) or current_rank < minimum_rank
        if needs_promotion and defer_dynamic_admission:
            dynamic = bool(evidence.get("runtime_eval_validity_dynamic", False))
            ceiling = str(evidence.get("runtime_eval_validity_max") or "")
            ceiling_rank = VALIDITY_RANK.get(ceiling, -1)
            if dynamic and ceiling_rank >= minimum_rank:
                # Crucially, this is only a load-time deferral.  The caller
                # performs the ordinary strict gate again immediately after
                # load(), before model preflight or task dispatch.
                return ""
        if validity == "smoke_only" and not bool(allow_smoke):
            return "runtime benchmark evidence is smoke_only while allow_smoke_only=false"
        if current_rank < minimum_rank:
            return (
                f"runtime benchmark eval_validity={validity!r} < min_eval_validity={min_validity!r}"
            )
        return ""

    def _abort_before_task_loop(
        self,
        *,
        storage: RunStorage,
        contract: dict[str, Any],
        reason: str,
        run_status: str,
        aggregate_flag: str,
    ) -> dict[str, Any]:
        """Persist a failed admission artifact without dispatching a task."""

        contract = complete_experiment_evidence_contract(
            contract,
            model_observed={
                "model_instantiated": False,
                "dummy_model_used": False,
                "dummy_fallback_used": False,
            },
            outcome_evidence_qualities=[],
            run_failed=True,
            run_status=run_status,
        )
        storage.write_evidence_contract(contract)
        storage.write_metrics(
            {
                "run_id": storage.run_id,
                "benchmark": self.benchmark_id,
                "method": self.method_id,
                "model": self.model_id,
                "evaluator": self.evaluator_id,
                "seed": self.seed,
                "n_tasks": 0,
                "aggregate": {
                    aggregate_flag: True,
                    "admission_reason": reason,
                    "claim_scope": contract["claim"]["scope"],
                    "leaderboard_eligible": False,
                },
                "per_task": [],
                "evidence_contract": {
                    "path": "evidence_contract.json",
                    "schema": contract["schema"],
                    "sha256": contract["artifact_sha256"],
                    "claim_scope": contract["claim"]["scope"],
                },
                "finished_at": time.time(),
            }
        )
        storage.log(f"{run_status}: {reason}")
        storage.close()
        return contract
