"""Single-task execution semantics shared by Runner orchestration."""

from __future__ import annotations

import copy
import time
from typing import Any

from lh_platform.core.generation_authority import GenerationAuthority
from lh_platform.core.generation_evidence import (
    action_model_generation_evidence,
    freeze_action,
    in_process_action_authority_evidence,
    merge_benchmark_execution_annotations,
    method_model_generation_cursor,
    method_model_generation_events_since,
    method_model_runtime_evidence,
)
from lh_platform.core.outcome import OutcomeEvidence
from lh_platform.core.storage import RunStorage
from lh_platform.core.task_execution import record_execution_failure
from lh_platform.core.types import Action, Observation, Role, Step, Task, Trajectory


class RunnerTaskMixin:
    def _outcome_evidence(self, benchmark: Any, task: Task, traj: Trajectory) -> OutcomeEvidence:
        producer = getattr(benchmark, "outcome_evidence", None)
        if producer is None:
            return OutcomeEvidence.fallback(traj.success)
        try:
            outcome = producer(task)
        except Exception as error:
            return OutcomeEvidence.producer_error(type(error).__name__)
        if isinstance(outcome, OutcomeEvidence):
            try:
                return outcome.validate()
            except ValueError as error:
                return OutcomeEvidence.producer_error(f"invalid evidence: {error}")
        if isinstance(outcome, dict):
            try:
                return OutcomeEvidence(**outcome).validate()
            except (TypeError, ValueError) as error:
                return OutcomeEvidence.producer_error(f"invalid evidence: {error}")
        return OutcomeEvidence.producer_error(f"unsupported type {type(outcome).__name__}")

    def _run_one(
        self,
        task: Task,
        method: Any,
        traj: Trajectory,
        storage: RunStorage,
        benchmark: Any,
        *,
        generation_authority: GenerationAuthority | None = None,
        strict_proxy_compatible: bool = False,
    ) -> None:
        """Run a single task.

        The contract is:
        1. The runner appends an initial system step.
        2. The method.predict is called up to task.max_steps times.
        3. The benchmark's step(action) returns an Observation.
        4. The runner appends the step to the trajectory and storage.
        5. The method.update(...) is called with action+observation.
        6. After max_steps or success, the trajectory is closed.
        """
        init_step = Step(
            step_idx=0,
            role=Role.SYSTEM,
            content=task.instruction,
            info={
                "task_id": task.task_id,
                "domain": task.domain.value,
                "seed": self.seed,
            },
        )
        traj.append(init_step)
        storage.append_step_task(init_step.to_dict(), task.task_id)

        try:
            method.reset(copy.deepcopy(task))
        except Exception as e:
            record_execution_failure(traj, stage="method.reset", error=e)
            err_step = Step(
                step_idx=1,
                role=Role.ERROR,
                content=f"method.reset failed: {e}",
                info={"stage": "method.reset", "task_id": task.task_id},
            )
            traj.append(err_step)
            storage.append_step_task(err_step.to_dict(), task.task_id)
            storage.log(f"method.reset failed: {e}")
            traj.close(success=False)
            return
        try:
            method.setup(copy.deepcopy(task))
        except Exception as e:
            record_execution_failure(traj, stage="method.setup", error=e)
            err_step = Step(
                step_idx=1,
                role=Role.ERROR,
                content=f"method.setup failed: {e}",
                info={"stage": "method.setup", "task_id": task.task_id},
            )
            traj.append(err_step)
            storage.append_step_task(err_step.to_dict(), task.task_id)
            storage.log(f"method.setup failed: {e}")
            traj.close(success=False)
            return

        success = False
        for i in range(1, task.max_steps + 1):
            if generation_authority is not None:
                generation_cursor_start = generation_authority.cursor()
                generation_before = generation_authority.runtime_evidence()
                authority_evidence = generation_authority.action_authority_evidence
            else:
                generation_cursor_start = method_model_generation_cursor(method)
                generation_before = method_model_runtime_evidence(method)
                authority_evidence = in_process_action_authority_evidence
            try:
                action_start = time.time()
                # Methods receive isolated snapshots.  In particular, a
                # later predict call cannot retroactively relabel an earlier
                # action's model event or patch binding in the persisted
                # trajectory.
                action: Action = method.predict(
                    copy.deepcopy(task),
                    copy.deepcopy(traj),
                )
                action_latency = time.time() - action_start
            except Exception as e:
                record_execution_failure(traj, stage="method.predict", error=e)
                err_step = Step(
                    step_idx=i,
                    role=Role.ERROR,
                    content=f"method.predict failed: {e}",
                    info={"stage": "method.predict", "task_id": task.task_id},
                )
                traj.append(err_step)
                storage.append_step_task(err_step.to_dict(), task.task_id)
                break

            if generation_authority is not None:
                generation_cursor_end = generation_authority.cursor()
                generation_after = generation_authority.runtime_evidence()
                generation_events = generation_authority.events_since(generation_cursor_start)
                action_authority = authority_evidence(
                    cursor_start=generation_cursor_start,
                    cursor_end=generation_cursor_end,
                )
                action_authority["strict_proxy_compatible"] = bool(strict_proxy_compatible)
            else:
                generation_cursor_end = method_model_generation_cursor(method)
                generation_after = method_model_runtime_evidence(method)
                generation_events = method_model_generation_events_since(
                    method,
                    generation_cursor_start,
                )
                action_authority = authority_evidence(
                    cursor_start=generation_cursor_start,
                    cursor_end=generation_cursor_end,
                )

            # A run-level count of model calls is not enough to prove that a
            # particular task's patch came from the configured model.  Bind
            # the action to a model-owned response event observed *during this
            # predict* call.  The evidence includes the exact response hash,
            # so a method cannot authenticate a static patch by making an
            # unrelated remote call and discarding its output.
            action.info = dict(action.info or {})
            action.info["model_generation"] = action_model_generation_evidence(
                before=generation_before,
                after=generation_after,
                generation_events=generation_events,
                authority=action_authority,
                action=action,
            )
            # Persist a deep-copy before the environment or Method.update can
            # observe it.  A method owns the Action it returned; it must not
            # be able to rewrite a persisted SWE patch or runner-owned
            # generation evidence after the runner has authenticated it.
            persisted_action = freeze_action(action)

            try:
                # Benchmarks receive their own copy as well.  Some native
                # harnesses annotate actions for execution diagnostics; those
                # annotations must not mutate the signed trajectory action.
                execution_action = freeze_action(persisted_action)
                obs: Observation = benchmark.step(task, execution_action)
                merge_benchmark_execution_annotations(
                    persisted_action,
                    execution_action,
                )
            except Exception as e:
                record_execution_failure(traj, stage="benchmark.step", error=e)
                err_step = Step(
                    step_idx=i,
                    role=Role.ERROR,
                    content=f"benchmark.step failed: {e}",
                    info={"stage": "benchmark.step", "task_id": task.task_id},
                )
                traj.append(err_step)
                storage.append_step_task(err_step.to_dict(), task.task_id)
                storage.log(f"benchmark.step failed: {e}")
                traj.close(success=False)
                return

            action_info = persisted_action.info or {}
            step = Step(
                step_idx=i,
                role=Role.ASSISTANT,
                content=persisted_action.raw,
                action=persisted_action,
                observation=obs,
                prompt_tokens=int(action_info.get("prompt_tokens", 0) or 0),
                completion_tokens=int(action_info.get("completion_tokens", 0) or 0),
                cost_usd=float(action_info.get("cost_usd", 0.0) or 0.0),
                latency_s=float(action_info.get("latency_s", action_latency)),
                info={
                    "model_id": self.model_id,
                    "benchmark_id": self.benchmark_id,
                    "task_id": task.task_id,
                    "seed": self.seed,
                },
            )
            traj.append(step)
            storage.append_step_task(step.to_dict(), task.task_id)

            try:
                # Do not hand adapters a mutable reference to the persisted
                # trajectory/action.  Update hooks are for method-internal
                # state, not for retroactively changing recorded evidence.
                method.update(
                    copy.deepcopy(traj),
                    freeze_action(persisted_action),
                    copy.deepcopy(obs),
                )
            except Exception as e:
                record_execution_failure(traj, stage="method.update", error=e)
                err_step = Step(
                    step_idx=i,
                    role=Role.ERROR,
                    content=f"method.update failed: {e}",
                    info={"stage": "method.update", "task_id": task.task_id},
                )
                traj.append(err_step)
                storage.append_step_task(err_step.to_dict(), task.task_id)
                storage.log(f"method.update failed: {e}")
                traj.close(success=False)
                return

            if obs.info.get("done") is True:
                final = obs.info.get("answer")
                success = bool(obs.info.get("success", False))
                traj.close(success=success, final_answer=final)
                return

        traj.close(success=success, final_answer=None)
