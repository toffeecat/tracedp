"""Synchronous experiment orchestration behind the public Runner facade."""

from __future__ import annotations

import copy
import inspect
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lh_platform.core._runner_admission import RunnerAdmissionMixin
from lh_platform.core._runner_metrics import (
    aggregate_task_metrics as _aggregate,
)
from lh_platform.core._runner_metrics import (
    benchmark_task_metrics as _benchmark_task_metrics,
)
from lh_platform.core._runner_task import RunnerTaskMixin
from lh_platform.core.artifact_redaction import redact_artifact, redact_secrets
from lh_platform.core.credibility import (
    actual_dispatch_from_trajectory,
    build_run_credibility,
    execution_failure_from_trajectory,
    requested_execution_from_kwargs,
)
from lh_platform.core.evidence_contract import (
    RuntimeAdmissionError,
    benchmark_runtime_evidence,
    build_experiment_evidence_contract,
    complete_experiment_evidence_contract,
)
from lh_platform.core.generation_authority import (
    GenerationAuthority,
    IsolatedMethodProxy,
)
from lh_platform.core.generation_evidence import (
    authority_runtime_evidence,
    method_model_runtime_evidence,
    summarize_model_runtime,
)
from lh_platform.core.method_capabilities import MethodCapabilities
from lh_platform.core.model_runtime import (
    ModelRuntimePreflightError,
    failed_preflight_evidence,
    preflight_model_runtime,
    skipped_preflight_evidence,
)
from lh_platform.core.outcome import EVIDENCE_STRONG, OutcomeEvidence
from lh_platform.core.receipt_verification import collect_benchmark_receipts
from lh_platform.core.registry import (
    BENCHMARKS,
    EVALUATORS,
    METHODS,
    MODELS,
    bootstrap_runtime_registries,
    is_builtin_method_id,
    loaded_method_adapter_modules,
)
from lh_platform.core.run_admission import (
    audit_runtime_capabilities,
    resolve_method_admission,
)
from lh_platform.core.storage import RunStorage
from lh_platform.core.task_execution import (
    TaskExecutionRecord,
    close_failed_task_trajectory,
)
from lh_platform.core.types import Task, Trajectory
from lh_platform.methods.inventory import get_method_inventory_entry


class Runner(RunnerAdmissionMixin, RunnerTaskMixin):
    def __init__(
        self,
        benchmark: str,
        method: str,
        model: str,
        evaluator: str = "long_horizon",
        n_tasks: int = 3,
        results_root: str | Path = "results",
        run_id: str | None = None,
        config_extra: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        # Strict provenance runs must not import arbitrary method modules in
        # the controller.  Load only models/benchmarks/evaluators here; the
        # selected method is loaded inside its worker later.
        bootstrap_runtime_registries()
        if not BENCHMARKS.has(benchmark):
            raise KeyError(f"benchmark not registered: {benchmark}")
        if (
            not METHODS.has(method)
            and not is_builtin_method_id(method)
            and get_method_inventory_entry(method) is None
        ):
            raise KeyError(f"method not registered: {method}")
        if not MODELS.has(model):
            raise KeyError(
                f"model not registered: {model}. Use a registry key (e.g. 'openai_chat') "
                f"or an alias (one of {MODELS.aliases()})."
            )
        if not EVALUATORS.has(evaluator):
            raise KeyError(f"evaluator not registered: {evaluator}")

        self.benchmark_id = benchmark
        self.method_id = method
        # Resolve model alias (e.g. 'gpt-4o-mini' -> 'openai_chat')
        self.model_id = MODELS.resolve(model)
        self.model_alias = model
        self.evaluator_id = evaluator
        self.n_tasks = n_tasks
        self.results_root = Path(results_root)
        self.seed = seed
        # Append seed suffix to caller-provided run_id so parallel seed
        # sweeps don't collide. Auto-generated run_ids (storage side)
        # are already unique per second; we append -s{seed} on top.
        if run_id is not None and seed is not None:
            run_id = f"{run_id}_s{seed}"
        self.run_id = run_id
        self.config_extra = config_extra or {}
        raw_benchmark_kwargs = self.config_extra.get("benchmark_kwargs", {})
        if not isinstance(raw_benchmark_kwargs, dict):
            raise ValueError("config_extra.benchmark_kwargs must be a mapping")
        self.benchmark_kwargs = dict(raw_benchmark_kwargs)
        self.method_kwargs = dict(self.config_extra.get("method_kwargs", {}))
        if seed is not None:
            self.method_kwargs.setdefault("seed", seed)

    def run(self) -> dict[str, Any]:
        # When seed is set but run_id is auto-generated, append `-s{seed}`
        # so parallel seed sweeps don't collide on the same second-based id.
        if self.seed is not None and self.run_id is None:
            self.run_id = f"run_s{self.seed}"
        # Validate configuration before allocating a result directory.  A
        # malformed runtime policy is a configuration error, not a failed
        # experiment, so it must not leave a deceptively runnable-looking
        # artifact behind.  A *reachable-but-wrong* server is handled below
        # after storage exists so its failed preflight remains auditable.
        model_kwargs, runtime_policy = self._resolved_model_runtime()
        resolved_config = self._resolved_config_snapshot(
            model_kwargs=model_kwargs,
            runtime_policy=runtime_policy,
        )
        storage = RunStorage(self.results_root, self.run_id)
        self.run_id = storage.run_id
        benchmark_cls = BENCHMARKS._items[BENCHMARKS.resolve(self.benchmark_id)]
        strict_generation_authority = self._requires_process_generation_authority(runtime_policy)
        benchmark_required_action_contract = str(
            getattr(benchmark_cls, "required_action_contract", "") or ""
        )
        method_admission = resolve_method_admission(
            method_id=self.method_id,
            benchmark_id=self.benchmark_id,
            benchmark_required_action_contract=benchmark_required_action_contract,
            strict=strict_generation_authority,
            preloaded_adapters=(
                loaded_method_adapter_modules() if strict_generation_authority else ()
            ),
            method_kwargs=self.method_kwargs,
        )
        method_admission_reason = method_admission.reason
        method_static_capability = method_admission.declared.integration_level
        method_supported_action_contracts = method_admission.declared.supported_action_contracts
        method_strict_proxy_compatible = (
            method_admission.strict_proxy_compatible if strict_generation_authority else False
        )
        leaderboard_requested = bool(
            (self.config_extra.get("gates") or {}).get("leaderboard_eligible", False)
        )
        if method_admission_reason:
            # Do not even healthcheck a model server for a pair whose action
            # languages are statically incompatible.  That would violate the
            # advertised "reject before model dispatch" contract and can
            # touch a live remote endpoint unnecessarily.
            runtime_preflight = skipped_preflight_evidence(
                provider=self.model_id,
                model_kwargs=model_kwargs,
                policy=runtime_policy,
                reason=method_admission_reason,
            )
        else:
            # A model healthcheck is itself a remote request.  It must wait
            # until the concrete benchmark has proved that its data, proxy,
            # credentials, and configured validity tier are admissible.
            runtime_preflight = skipped_preflight_evidence(
                provider=self.model_id,
                model_kwargs=model_kwargs,
                policy=runtime_policy,
                reason="benchmark runtime admission pending",
            )

        static_benchmark_evidence = {
            "data_source": "not_loaded",
            "has_real_data": bool(getattr(benchmark_cls, "has_real_data", False)),
            "supports_native_eval": bool(getattr(benchmark_cls, "supports_native_eval", False)),
            "has_native_harness": bool(getattr(benchmark_cls, "has_native_harness", False)),
            "is_mock": bool(getattr(benchmark_cls, "is_mock", True)),
            "eval_validity": str(getattr(benchmark_cls, "eval_validity", "smoke_only")),
            "required_action_contract": str(
                getattr(benchmark_cls, "required_action_contract", "") or ""
            ),
        }
        contract = build_experiment_evidence_contract(
            run_id=storage.run_id,
            resolved_config=resolved_config,
            model_provider=self.model_id,
            model_runtime_policy=runtime_policy,
            model_preflight=runtime_preflight,
            benchmark_id=self.benchmark_id,
            benchmark_evidence=static_benchmark_evidence,
            method_id=self.method_id,
            method_integration_level=method_static_capability,
            method_strict_proxy_compatible=method_strict_proxy_compatible,
            method_supported_action_contracts=method_supported_action_contracts,
            evaluator_id=self.evaluator_id,
            seed=self.seed,
            leaderboard_requested=leaderboard_requested,
        )
        run_record = {
            "requested_run_id": storage.requested_run_id,
            "resolved_run_id": storage.run_id,
            "benchmark": self.benchmark_id,
            "method": self.method_id,
            "model": self.model_id,
            "evaluator": self.evaluator_id,
            "n_tasks": self.n_tasks,
            "seed": self.seed,
            "config_extra": redact_artifact(self.config_extra),
            "resolved_config_sha256": contract["config"]["resolved_config_sha256"],
            "runtime_policy": runtime_policy.to_dict(),
            "runtime_preflight": redact_secrets(runtime_preflight.to_dict()),
            "evidence_contract_path": "evidence_contract.json",
            "started_at": time.time(),
        }
        storage.write_run(run_record)
        storage.write_evidence_contract(contract)
        storage.log(
            f"start benchmark={self.benchmark_id} method={self.method_id} "
            f"model={self.model_id} evaluator={self.evaluator_id} "
            f"n_tasks={self.n_tasks} seed={self.seed}"
        )
        if method_admission_reason:
            self._abort_before_task_loop(
                storage=storage,
                contract=contract,
                reason=method_admission_reason,
                run_status="method_admission_failed",
                aggregate_flag="method_admission_failed",
            )
            raise RuntimeAdmissionError(method_admission_reason)

        def persist_preflight(preflight: Any) -> None:
            """Keep run.json and the evidence contract synchronized by phase."""

            contract["model"]["requested_model"] = preflight.requested_model
            contract["model"]["preflight"] = preflight.to_dict()
            run_record["runtime_preflight"] = redact_secrets(preflight.to_dict())
            storage.write_run(run_record)
            storage.write_evidence_contract(contract)

        benchmark: Any | None = None
        try:
            benchmark = BENCHMARKS.create(self.benchmark_id, **self.benchmark_kwargs)
        except Exception as error:
            reason = f"benchmark construction failed before model preflight: {type(error).__name__}: {error}"
            runtime_preflight = skipped_preflight_evidence(
                provider=self.model_id,
                model_kwargs=model_kwargs,
                policy=runtime_policy,
                reason=reason,
            )
            persist_preflight(runtime_preflight)
            self._abort_before_task_loop(
                storage=storage,
                contract=contract,
                reason=reason,
                run_status="runtime_admission_failed",
                aggregate_flag="runtime_admission_failed",
            )
            raise RuntimeAdmissionError(reason) from error

        benchmark_finalization_errors: list[str] = []
        benchmark_finalization_status = "pending"

        def refresh_benchmark_contract() -> dict[str, Any]:
            """Re-read mutable benchmark evidence into the run trust anchor.

            Native adapters may downgrade themselves during ``close()`` when
            their required receipt cannot be written.  The contract must
            reflect that post-close state before per-task outcomes or the
            aggregate are derived; otherwise a stale ``real_native`` label
            could outlive its reproducibility evidence.
            """

            assert benchmark is not None
            evidence = {
                "id": self.benchmark_id,
                **benchmark_runtime_evidence(benchmark),
            }
            contract["benchmark"] = evidence
            if benchmark_finalization_errors:
                contract["benchmark"]["finalization"] = {
                    "status": "failed",
                    "errors": list(benchmark_finalization_errors),
                }
            else:
                contract["benchmark"]["finalization"] = {"status": benchmark_finalization_status}
            storage.write_evidence_contract(contract)
            return evidence

        def close_benchmark(*, phase: str) -> None:
            """Close a benchmark and always persist its final runtime state."""

            nonlocal benchmark_finalization_status
            assert benchmark is not None
            try:
                closer = getattr(benchmark, "close", None)
                if callable(closer):
                    try:
                        parameters = inspect.signature(closer).parameters.values()
                    except (TypeError, ValueError):
                        # An opaque callable cannot be safely introspected;
                        # preserve the modern artifact-aware contract.
                        closer(artifacts_dir=storage.external_artifacts_dir())
                    else:
                        artifacts_parameter = next(
                            (
                                parameter
                                for parameter in parameters
                                if parameter.name == "artifacts_dir"
                            ),
                            None,
                        )
                        accepts_keyword_artifacts = (
                            artifacts_parameter is not None
                            and artifacts_parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
                        ) or any(
                            parameter.kind is inspect.Parameter.VAR_KEYWORD
                            for parameter in parameters
                        )
                        if accepts_keyword_artifacts:
                            closer(artifacts_dir=storage.external_artifacts_dir())
                        elif (
                            artifacts_parameter is not None
                            and artifacts_parameter.kind is inspect.Parameter.POSITIONAL_ONLY
                        ):
                            # A few upstream wrappers use a positional-only
                            # cleanup sink.  Honor that explicit signature;
                            # passing it as a keyword would falsely mark an
                            # otherwise valid native run as a close failure.
                            closer(storage.external_artifacts_dir())
                        else:
                            # Legacy upstream wrappers often expose close()
                            # only.  Do not invent a positional contract or
                            # turn a successful native evaluator into a
                            # TypeError merely because it predates artifacts.
                            closer()
            except Exception as error:
                reason = f"benchmark.close {phase} failed: {type(error).__name__}: {error}"
                benchmark_finalization_errors.append(reason)
                storage.log(f"{reason}\n{traceback.format_exc()}")
            finally:
                benchmark_finalization_status = (
                    "failed" if benchmark_finalization_errors else "closed"
                )
                refresh_benchmark_contract()

        def finalize_benchmark_receipts(
            trajectory_manifest: Mapping[str, Any],
        ) -> list[dict[str, str]]:
            """Seal any benchmark-native receipt after trajectories exist.

            ``close`` owns resource cleanup and may create a preliminary
            receipt.  This later hook is the only place where a receipt can
            legally claim a particular Runner run and its immutable trajectory
            manifest.  Any failure invalidates every task result rather than
            leaving a stale ``real_native`` aggregate behind.
            """

            nonlocal benchmark_finalization_status
            assert benchmark is not None
            if benchmark_finalization_errors:
                benchmark_finalization_status = "failed"
                refresh_benchmark_contract()
                return []
            finalizer = getattr(benchmark, "finalize_run_evidence", None)
            if not callable(finalizer):
                benchmark_finalization_status = "completed"
                refresh_benchmark_contract()
                return []
            try:
                receipts = collect_benchmark_receipts(
                    finalizer,
                    storage=storage,
                    run_id=storage.run_id,
                    trajectory_manifest=trajectory_manifest,
                )
                benchmark_finalization_status = "completed"
                return receipts
            except Exception as error:
                reason = f"benchmark final evidence binding failed: {type(error).__name__}: {error}"
                benchmark_finalization_errors.append(reason)
                benchmark_finalization_status = "failed"
                storage.log(f"{reason}\n{traceback.format_exc()}")
                return []
            finally:
                refresh_benchmark_contract()

        try:
            binder = getattr(benchmark, "bind_run_artifacts", None)
            if callable(binder):
                binder(storage.external_artifacts_dir())
        except Exception as error:
            reason = (
                "benchmark artifact binding failed before model preflight: "
                f"{type(error).__name__}: {error}"
            )
            refresh_benchmark_contract()
            close_benchmark(phase="after artifact binding failure")
            runtime_preflight = skipped_preflight_evidence(
                provider=self.model_id,
                model_kwargs=model_kwargs,
                policy=runtime_policy,
                reason=reason,
            )
            persist_preflight(runtime_preflight)
            self._abort_before_task_loop(
                storage=storage,
                contract=contract,
                reason=reason,
                run_status="runtime_admission_failed",
                aggregate_flag="runtime_admission_failed",
            )
            raise RuntimeAdmissionError(reason) from error

        refresh_benchmark_contract()
        runtime_gate_reason = self._runtime_benchmark_gate_reason(
            contract["benchmark"],
            defer_dynamic_admission=True,
        )
        if runtime_gate_reason:
            close_benchmark(phase="during admission")
            runtime_preflight = skipped_preflight_evidence(
                provider=self.model_id,
                model_kwargs=model_kwargs,
                policy=runtime_policy,
                reason=(
                    "benchmark runtime admission failed before model preflight: "
                    f"{runtime_gate_reason}"
                ),
            )
            persist_preflight(runtime_preflight)
            self._abort_before_task_loop(
                storage=storage,
                contract=contract,
                reason=runtime_gate_reason,
                run_status="runtime_admission_failed",
                aggregate_flag="runtime_admission_failed",
            )
            raise RuntimeAdmissionError(runtime_gate_reason)

        try:
            seeded_loader = getattr(benchmark, "load_seeded", None)
            if self.seed is not None and callable(seeded_loader):
                tasks: Sequence[Task] = seeded_loader(n=self.n_tasks, seed=self.seed)
            else:
                tasks = benchmark.load(n=self.n_tasks)
        except Exception as e:
            storage.log(f"benchmark.load failed: {e}\n{traceback.format_exc()}")
            tasks = []

        # A dataset may exist but be empty or unreadable; several adapters
        # only downgrade their instance validity from inside ``load()``.
        # Refresh the contract and enforce the same gate before constructing a
        # method or issuing any model call.
        refresh_benchmark_contract()
        runtime_gate_reason = self._runtime_benchmark_gate_reason(contract["benchmark"])
        if not tasks:
            runtime_gate_reason = runtime_gate_reason or "benchmark.load yielded no tasks"
        if runtime_gate_reason:
            close_benchmark(phase="after load admission")
            runtime_preflight = skipped_preflight_evidence(
                provider=self.model_id,
                model_kwargs=model_kwargs,
                policy=runtime_policy,
                reason=(
                    "benchmark runtime admission failed before model preflight: "
                    f"{runtime_gate_reason}"
                ),
            )
            persist_preflight(runtime_preflight)
            self._abort_before_task_loop(
                storage=storage,
                contract=contract,
                reason=runtime_gate_reason,
                run_status="runtime_admission_failed",
                aggregate_flag="runtime_admission_failed",
            )
            raise RuntimeAdmissionError(runtime_gate_reason)

        try:
            runtime_preflight = preflight_model_runtime(
                provider=self.model_id,
                model_kwargs=model_kwargs,
                policy=runtime_policy,
            )
        except Exception as error:
            runtime_preflight = failed_preflight_evidence(
                provider=self.model_id,
                model_kwargs=model_kwargs,
                policy=runtime_policy,
                error=error,
            )
        persist_preflight(runtime_preflight)
        if not runtime_preflight.passed:
            close_benchmark(phase="after model preflight failure")
            self._abort_before_task_loop(
                storage=storage,
                contract=contract,
                reason=runtime_preflight.reason,
                run_status="runtime_preflight_failed",
                aggregate_flag="runtime_preflight_failed",
            )
            raise ModelRuntimePreflightError(runtime_preflight.reason)

        try:
            evaluator = EVALUATORS.create(self.evaluator_id)
        except Exception as error:
            # At this point the benchmark may own a live official environment
            # and a successfully admitted dataset.  Treat evaluator creation
            # as part of run startup, not as an unhandled exception: close the
            # benchmark, persist an invalid artifact, and never leave the
            # environment open merely because no task was dispatched.
            reason = (
                "evaluator construction failed after benchmark admission: "
                f"{type(error).__name__}: {error}"
            )
            close_benchmark(phase="after evaluator construction failure")
            self._abort_before_task_loop(
                storage=storage,
                contract=contract,
                reason=reason,
                run_status="evaluator_initialization_failed",
                aggregate_flag="evaluator_initialization_failed",
            )
            raise RuntimeAdmissionError(reason) from error

        task_records: list[TaskExecutionRecord] = []
        for idx, task in enumerate(tasks):
            storage.log(f"task {idx + 1}/{len(tasks)}: {task.task_id}")
            method_create_kwargs = {
                k: v for k, v in self.method_kwargs.items() if k not in {"execution_path"}
            }
            traj = Trajectory(
                task_id=task.task_id,
                method=self.method_id,
                model=self.model_id,
                benchmark=self.benchmark_id,
            )
            method: Any | None = None
            generation_authority: GenerationAuthority | None = None
            isolated_method: IsolatedMethodProxy | None = None
            observed: dict[str, Any] = {"model_instantiated": False}
            try:
                if strict_generation_authority:
                    generation_authority = GenerationAuthority(
                        self.model_id,
                        model_kwargs,
                    )
                    generation_authority.start()
                    # Do not construct or invoke the method in the Runner
                    # process. The method worker receives the authority's
                    # generate-only endpoint and no model endpoint/credential
                    # configuration; Runner retains only its independent
                    # cursor/events/runtime control endpoint.
                    if method_admission.strict_launch is None:
                        raise RuntimeAdmissionError(
                            "strict method launch plan missing after successful admission"
                        )
                    isolated_method = IsolatedMethodProxy(
                        method_admission.strict_launch,
                        generation_authority.take_method_channel(),
                    )
                    isolated_method.start()
                    method = isolated_method
                    observed_capabilities = isolated_method.runtime_capabilities
                    storage.log(
                        f"task {task.task_id}: isolated method + generation authority enabled"
                    )
                else:
                    method = METHODS.create(
                        self.method_id,
                        model_id=self.model_id,
                        model_registry=MODELS,
                        model_kwargs=model_kwargs,
                        **method_create_kwargs,
                    )
                    # Surface the seed on the method instance too, in case it
                    # was wiped by adapter init.
                    if self.seed is not None:
                        method.seed = self.seed
                    observed_capabilities = MethodCapabilities.observe(method)
                if observed_capabilities is None:
                    raise RuntimeError("method runtime capability observation is missing")
                capability_receipt = audit_runtime_capabilities(
                    method_admission.declared,
                    observed_capabilities,
                )
                traj.info["method_capability_receipt"] = capability_receipt.to_dict()
                if not capability_receipt.matches_declared:
                    storage.log(
                        f"task {task.task_id}: runtime method capability mismatch "
                        f"fields={list(capability_receipt.mismatches)!r}; "
                        "Method Inventory remains authoritative"
                    )
                self._run_one(
                    task,
                    method,
                    traj,
                    storage,
                    benchmark,
                    generation_authority=generation_authority,
                    strict_proxy_compatible=method_strict_proxy_compatible,
                )
                if generation_authority is not None:
                    observed = authority_runtime_evidence(generation_authority)
                else:
                    observed = method_model_runtime_evidence(method)
            except Exception as e:
                storage.log(f"task {task.task_id} failed: {e}\n{traceback.format_exc()}")
                traj.success = False
                traj.info["error"] = repr(e)
                close_failed_task_trajectory(
                    trajectory=traj,
                    storage=storage,
                    task=task,
                    error=e,
                )
                # A timed-out worker may still wait on a model call. Kill it
                # before attempting authority control so this task cannot
                # leave a live process or poison a later task.
                if isolated_method is not None and isolated_method.broken:
                    isolated_method.close()
                if generation_authority is not None:
                    observed = authority_runtime_evidence(
                        generation_authority,
                        cause=e,
                    )
                elif method is not None:
                    observed = method_model_runtime_evidence(method)
            finally:
                if isolated_method is not None:
                    isolated_method.close()
                if generation_authority is not None:
                    generation_authority.close()
            task_records.append(
                TaskExecutionRecord(
                    task=task,
                    trajectory=traj,
                    method_integration_level=method_static_capability,
                    observed_model_runtime=redact_secrets(dict(observed or {})),
                )
            )

        close_benchmark(phase="after task loop")

        per_task_metrics: list[dict[str, Any]] = []
        task_outcomes: list[OutcomeEvidence] = []
        outcome_qualities: list[str] = []
        observed_model_runtimes: list[dict[str, Any]] = []
        task_trajectory_paths: list[Path] = []
        for record in task_records:
            task = record.task
            traj = record.trajectory
            method_static_capability = record.method_integration_level
            observed_runtime = record.observed_model_runtime
            outcome = self._outcome_evidence(benchmark, task, traj)
            task_outcomes.append(outcome)
            outcome_qualities.append(outcome.evidence_quality)
            observed_model_runtimes.append(observed_runtime)
            benchmark_eval_validity = getattr(benchmark, "eval_validity", "smoke_only")
            requested_execution = requested_execution_from_kwargs(self.method_kwargs)
            actual_dispatch = actual_dispatch_from_trajectory(method_static_capability, traj)
            credibility = build_run_credibility(
                benchmark_eval_validity=benchmark_eval_validity,
                method_integration_level=method_static_capability,
                requested_execution=requested_execution,
                actual_dispatch=actual_dispatch,
                outcome_evidence=outcome,
                benchmark_finalization_failed=bool(benchmark_finalization_errors),
                execution_failed=execution_failure_from_trajectory(traj),
            )
            diagnostic_success = bool(outcome.success)
            verified_success = diagnostic_success and outcome.evidence_quality == EVIDENCE_STRONG
            traj.info["outcome_evidence"] = outcome.to_dict()
            traj.info["diagnostic_outcome_success"] = diagnostic_success
            traj.info["run_credibility"] = credibility.to_dict()
            traj.info["model_runtime"] = observed_runtime
            # A terminal answer, patch preflight, or heuristic match is a
            # diagnostic event, not benchmark success.  Only a strong
            # benchmark-owned verifier may set the canonical success metric.
            traj.success = verified_success
            if credibility.run_failed:
                traj.success = False
                traj.info["run_failed"] = credibility.reason
            if outcome.final_answer is not None:
                traj.final_answer = outcome.final_answer
            task_trajectory_paths.append(storage.write_task_trajectory(traj))
            try:
                m = evaluator.score(traj, task)
            except Exception as e:
                m = {"error": repr(e)}
            m["task_id"] = task.task_id
            m["steps"] = len(traj.steps)
            m["success"] = traj.success
            m["diagnostic_outcome_success"] = diagnostic_success
            m["seed"] = self.seed
            m["benchmark_eval_validity"] = benchmark_eval_validity
            m["method_integration_level"] = method_static_capability
            m["method_capability_receipt"] = copy.deepcopy(
                traj.info.get("method_capability_receipt")
            )
            m["outcome_evidence"] = outcome.to_dict()
            m["run_credibility"] = credibility.to_dict()
            m["model_runtime"] = observed_runtime
            benchmark_metrics, rejected_metric_keys = _benchmark_task_metrics(
                benchmark,
                task,
                traj,
            )
            m.update(benchmark_metrics)
            if rejected_metric_keys:
                m["benchmark_metric_rejected_keys"] = rejected_metric_keys
            per_task_metrics.append(m)

        # This is the trust anchor for downstream rollout collection.  It
        # binds the complete set of task trajectories written by this run;
        # a collector may not discover or silently omit files afterward.
        contract["artifacts"] = {
            "trajectory_manifest": storage.build_task_trajectory_manifest(task_trajectory_paths)
        }
        benchmark_receipts = finalize_benchmark_receipts(
            contract["artifacts"]["trajectory_manifest"]
        )
        if benchmark_finalization_errors:
            # Native receipt binding runs *after* the first trajectory write,
            # because the receipt must hash those exact files.  If that final
            # step fails, rewrite all task artifacts with failed credibility
            # and rebuild the manifest; no later collector can accidentally
            # train on a trajectory that the final run contract rejects.
            for index, record in enumerate(task_records):
                task = record.task
                traj = record.trajectory
                method_static_capability = record.method_integration_level
                observed_runtime = record.observed_model_runtime
                outcome = task_outcomes[index]
                credibility = build_run_credibility(
                    benchmark_eval_validity=getattr(benchmark, "eval_validity", "smoke_only"),
                    method_integration_level=method_static_capability,
                    requested_execution=requested_execution_from_kwargs(self.method_kwargs),
                    actual_dispatch=actual_dispatch_from_trajectory(method_static_capability, traj),
                    outcome_evidence=outcome,
                    benchmark_finalization_failed=True,
                    execution_failed=execution_failure_from_trajectory(traj),
                )
                traj.success = False
                traj.info["run_credibility"] = credibility.to_dict()
                traj.info["run_failed"] = credibility.reason
                traj.info["outcome_evidence"] = outcome.to_dict()
                traj.info["model_runtime"] = observed_runtime
                task_trajectory_paths[index] = storage.write_task_trajectory(traj)
                metric = per_task_metrics[index]
                metric["success"] = False
                metric["benchmark_eval_validity"] = getattr(
                    benchmark, "eval_validity", "smoke_only"
                )
                metric["run_credibility"] = credibility.to_dict()
            contract["artifacts"]["trajectory_manifest"] = storage.build_task_trajectory_manifest(
                task_trajectory_paths
            )
        if benchmark_receipts:
            contract["artifacts"]["benchmark_receipts"] = benchmark_receipts

        aggregate = _aggregate(per_task_metrics)
        aggregate["seed"] = self.seed
        model_observed = summarize_model_runtime(observed_model_runtimes)
        run_failed = any(
            bool((record.trajectory.info.get("run_credibility") or {}).get("run_failed"))
            for record in task_records
        )
        contract = complete_experiment_evidence_contract(
            contract,
            model_observed=model_observed,
            outcome_evidence_qualities=outcome_qualities,
            run_failed=run_failed,
        )
        storage.write_evidence_contract(contract)
        aggregate["claim_scope"] = contract["claim"]["scope"]
        aggregate["leaderboard_eligible"] = contract["claim"]["leaderboard_eligible"]
        aggregate["leaderboard_requested"] = contract["claim"]["leaderboard_requested"]
        aggregate["benchmark_eval_validity"] = contract["benchmark"]["eval_validity"]
        aggregate["benchmark_data_source"] = contract["benchmark"]["data_source"]
        metrics = {
            "run_id": storage.run_id,
            "benchmark": self.benchmark_id,
            "method": self.method_id,
            "model": self.model_id,
            "evaluator": self.evaluator_id,
            "seed": self.seed,
            "n_tasks": len(per_task_metrics),
            "aggregate": aggregate,
            "per_task": per_task_metrics,
            "evidence_contract": {
                "path": "evidence_contract.json",
                "schema": contract["schema"],
                "sha256": contract["artifact_sha256"],
                "claim_scope": contract["claim"]["scope"],
            },
            "finished_at": time.time(),
        }
        storage.write_metrics(metrics)
        storage.log(f"done; aggregate={aggregate}")
        offline = getattr(benchmark, "last_summary", None)
        if isinstance(offline, dict) and offline:
            aggregate.update(offline)
            metrics["aggregate"] = aggregate
            metrics["offline_eval"] = offline
            storage.write_metrics(metrics)
        storage.close()
        return metrics
