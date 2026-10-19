"""Native ToolBench task suite for agentic RL training with real API execution."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lh_platform.training.agentic_rl_components import builtin_component
from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    AgenticRolloutBatch,
    EvaluationRequest,
    EvaluationResult,
    PolicyActionRequest,
    RolloutRequest,
    TrainablePolicy,
    canonical_sha256,
    token_ids_sha256,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.model_io import ConversationIR
from lh_platform.training.toolbench_actor_protocol import (
    TOOLBENCH_CALL_FINISH_CONTRACT,
    ParsedToolBenchAction,
    render_toolbench_prompt,
)

logger = logging.getLogger(__name__)


class ToolBenchSuiteError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolBenchTaskRow:
    query_id: str
    test_set: str
    query: str
    available_tools: list[Mapping[str, Any]]
    relevant_function_names: tuple[str, ...]
    category: str = ""
    tool_name: str = ""
    api_invocations: tuple[Mapping[str, Any], ...] = ()


class ToolBenchRemoteExecutor(Protocol):
    def execute(
        self,
        *,
        category: str,
        tool_name: str,
        api_name: str,
        tool_input: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ToolBenchSuiteConfig:
    role: str
    data_root: str
    artifacts_dir: str
    test_sets: tuple[str, ...]
    tasks_per_update: int = 4
    eval_task_count: int = 16
    max_episode_steps: int = 8
    history_length: int = 8
    train_temperature: float = 1.0
    eval_temperature: float = 0.4
    model_io_profile_id: str | None = None
    format_reward_weight: float = 0.1
    api_recall_reward_weight: float = 0.3
    finish_reward_weight: float = 0.6
    rate_limit_delay_s: float = 2.0
    rate_limit_recovery_delay_s: float = 10.0
    max_observation_length: int = 1024
    max_consecutive_transport_failures: int = 3

    def __post_init__(self) -> None:
        if self.role not in {"train_task_suite", "eval_task_suite"}:
            raise ToolBenchSuiteError("ToolBench suite role must be train or eval")
        if not self.test_sets:
            raise ToolBenchSuiteError("ToolBench suite requires at least one test set")
        if self.tasks_per_update < 1 or self.eval_task_count < 1:
            raise ToolBenchSuiteError("ToolBench task budgets must be positive")
        if self.max_episode_steps < 1 or self.max_episode_steps > 16:
            raise ToolBenchSuiteError("ToolBench max_episode_steps must be 1-16")
        if self.history_length < 0:
            raise ToolBenchSuiteError("ToolBench history_length must be non-negative")


def _resolve_function_invocation(
    action: ParsedToolBenchAction,
    task_row: ToolBenchTaskRow,
) -> tuple[str, str, str]:
    """Resolve api_name to (category, tool_name, api_name) via suffix matching."""
    target = action.api_name
    for inv in task_row.api_invocations:
        fn_name = str(inv.get("function_name", ""))
        if fn_name == target or fn_name.endswith(target):
            return (
                str(inv.get("category", "")),
                str(inv.get("tool_name", "")),
                str(inv.get("api_name", target)),
            )
    for tool in task_row.available_tools:
        if str(tool.get("name", "")) == target:
            return (
                task_row.category,
                task_row.tool_name,
                target,
            )
    return (task_row.category, task_row.tool_name, target)


class ToolBenchNativeTaskSuite:
    def __init__(
        self,
        config: ToolBenchSuiteConfig,
        *,
        tasks: Sequence[ToolBenchTaskRow],
        executor: ToolBenchRemoteExecutor,
    ) -> None:
        self.config = config
        self._executor = executor
        self._tasks = list(tasks)
        if not self._tasks:
            raise ToolBenchSuiteError("ToolBench suite requires at least one task")
        task_ids = [t.query_id for t in self._tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ToolBenchSuiteError("ToolBench tasks must have unique query_ids")
        self.manifest_sha256 = canonical_sha256(
            {
                "schema": "lh_toolbench_manifest/v1",
                "test_sets": list(config.test_sets),
                "task_ids": task_ids,
                "data_root_sha256": hashlib.sha256(
                    config.data_root.encode()
                ).hexdigest(),
            }
        )
        self.dataset_content_sha256 = self.manifest_sha256
        self._artifacts_dir = Path(config.artifacts_dir).resolve()
        self.component = builtin_component(
            role=config.role,
            name="toolbench_native",
            config={
                "test_sets": list(config.test_sets),
                "manifest_sha256": self.manifest_sha256,
                "max_episode_steps": config.max_episode_steps,
                "response_contract_id": TOOLBENCH_CALL_FINISH_CONTRACT.id,
                "response_contract_sha256": TOOLBENCH_CALL_FINISH_CONTRACT.descriptor.sha256,
                "execution_mode": "upstream_remote",
                "evaluator": "tooleval_gpt35_majority_vote",
            },
        )

    def _rotated(self, seed: int, count: int) -> list[ToolBenchTaskRow]:
        n = len(self._tasks)
        offset = seed % n
        selected: list[ToolBenchTaskRow] = []
        for i in range(min(count, n)):
            selected.append(self._tasks[(offset + i) % n])
        return selected

    def collect(self, policy: TrainablePolicy, request: RolloutRequest) -> AgenticRolloutBatch:
        if self.config.role != "train_task_suite":
            raise ToolBenchSuiteError("evaluation suite cannot collect training rollouts")
        selected = self._rotated(request.seed, self.config.tasks_per_update)
        cursor = policy.sampling_cursor()
        records: list[TrainingTrajectoryRecord] = []
        for task_index, task_row in enumerate(selected):
            group_uid = f"toolbench:{task_row.query_id}:u{request.update_index}:s{request.seed}"
            for branch_index in range(request.group_size):
                records.extend(
                    self._run_episode(
                        policy,
                        task_row=task_row,
                        traj_uid=f"{group_uid}:r{branch_index}",
                        group_uid=group_uid,
                        policy_seed=(
                            request.seed + task_index * request.group_size + branch_index
                        ),
                        max_steps=request.max_episode_steps,
                        rollout_config_sha256=request.rollout_config_sha256,
                        temperature=self.config.train_temperature,
                    )
                )
        keys = {(row.traj_uid, row.step_idx) for row in records}
        samples = tuple(
            sample
            for sample in policy.sampling_events_since(cursor)
            if (sample.traj_uid, sample.step_idx) in keys
        )
        return AgenticRolloutBatch(
            records=tuple(records),
            policy_samples=samples,
            task_manifest_sha256=self.manifest_sha256,
            policy_checkpoint_sha256=policy.checkpoint_sha256,
        ).validate()

    def evaluate(self, policy: TrainablePolicy, request: EvaluationRequest) -> EvaluationResult:
        if self.config.role != "eval_task_suite":
            raise ToolBenchSuiteError("training suite cannot evaluate")
        selected = self._rotated(request.seed, self.config.eval_task_count)
        successes = 0
        lengths: list[int] = []
        per_test_set: dict[str, dict[str, int]] = {}
        for index, task_row in enumerate(selected):
            rows = self._run_episode(
                policy,
                task_row=task_row,
                traj_uid=f"eval:{task_row.query_id}:{index}",
                group_uid=f"eval:{task_row.query_id}",
                policy_seed=request.seed + index,
                max_steps=request.max_episode_steps,
                rollout_config_sha256=canonical_sha256(
                    {"evaluation": True, "max_steps": request.max_episode_steps}
                ),
                temperature=self.config.eval_temperature,
            )
            won = bool(rows[-1].success) if rows else False
            successes += int(won)
            lengths.append(len(rows))
            ts = task_row.test_set
            if ts not in per_test_set:
                per_test_set[ts] = {"successes": 0, "tasks": 0}
            per_test_set[ts]["tasks"] += 1
            per_test_set[ts]["successes"] += int(won)
        return EvaluationResult(
            task_suite=self.component,
            task_manifest_sha256=self.manifest_sha256,
            policy_checkpoint_sha256=policy.checkpoint_sha256,
            task_count=len(selected),
            success_count=successes,
            mean_action_steps=sum(lengths) / max(len(lengths), 1),
            success_by_horizon=per_test_set,
            sampling_temperature=self.config.eval_temperature,
            episodes_per_task=1,
            stochastic_sampling=self.config.eval_temperature > 0.0,
            sampling_uncertainty_interval_available=False,
        )

    def _run_episode(
        self,
        policy: TrainablePolicy,
        *,
        task_row: ToolBenchTaskRow,
        traj_uid: str,
        group_uid: str,
        policy_seed: int,
        max_steps: int,
        rollout_config_sha256: str,
        temperature: float,
    ) -> list[TrainingTrajectoryRecord]:
        history: list[dict[str, str]] = []
        records: list[TrainingTrajectoryRecord] = []
        called_apis: set[str] = set()
        terminal_reason = "max_episode_steps"
        won = False
        consecutive_transport_failures = 0

        for step_index in range(max_steps):
            remaining = max_steps - step_index
            prompt = render_toolbench_prompt(
                query=task_row.query,
                available_tools=task_row.available_tools,
                history=history[-self.config.history_length :],
                step_index=step_index,
                max_steps=max_steps,
            )
            request = PolicyActionRequest(
                traj_uid=traj_uid,
                step_idx=step_index,
                conversation=ConversationIR.from_legacy_user_prompt(prompt),
                model_io_profile_id=self.config.model_io_profile_id,
                response_contract_id=TOOLBENCH_CALL_FINISH_CONTRACT.id,
                response_contract_sha256=(
                    TOOLBENCH_CALL_FINISH_CONTRACT.descriptor.sha256
                ),
                admissible_actions=(),
                sampling_temperature=temperature,
                response_stop_texts=TOOLBENCH_CALL_FINISH_CONTRACT.stop_texts,
            )
            (sample,) = policy.sample_actions(
                (request,),
                seed=policy_seed + step_index * 100_000,
            )
            parsed = TOOLBENCH_CALL_FINISH_CONTRACT.parse_response(sample.action_text)

            observation = ""
            step_reward = 0.0
            terminal = False

            if parsed.name == "call" and parsed.valid:
                step_reward += self.config.format_reward_weight
                relevant = set(task_row.relevant_function_names)
                if parsed.api_name in relevant:
                    step_reward += self.config.api_recall_reward_weight
                called_apis.add(parsed.api_name)

                category, tool_name, api_name = _resolve_function_invocation(
                    parsed, task_row
                )
                tool_input = json.dumps(parsed.parameters or {}, ensure_ascii=False)
                try:
                    result = self._executor.execute(
                        category=category,
                        tool_name=tool_name,
                        api_name=api_name,
                        tool_input=tool_input,
                    )
                    status_code = int(result.get("status_code", 0))
                    observation = str(result.get("observation", ""))
                    if len(observation) > self.config.max_observation_length:
                        observation = (
                            observation[: self.config.max_observation_length] + "..."
                        )

                    if status_code == 10:
                        time.sleep(self.config.rate_limit_recovery_delay_s)
                        result = self._executor.execute(
                            category=category,
                            tool_name=tool_name,
                            api_name=api_name,
                            tool_input=tool_input,
                        )
                        status_code = int(result.get("status_code", 0))
                        observation = str(result.get("observation", ""))

                    if status_code == 12:
                        consecutive_transport_failures += 1
                        observation = f"[TRANSPORT_ERROR] {observation}"
                        if (
                            consecutive_transport_failures
                            >= self.config.max_consecutive_transport_failures
                        ):
                            terminal = True
                            terminal_reason = "transport_failure"
                    else:
                        consecutive_transport_failures = 0

                    if status_code == 1:
                        observation = f"[INVALID_FUNCTION] {observation}"

                except Exception as exc:
                    consecutive_transport_failures += 1
                    observation = f"[EXECUTION_ERROR] {exc}"
                    if (
                        consecutive_transport_failures
                        >= self.config.max_consecutive_transport_failures
                    ):
                        terminal = True
                        terminal_reason = "transport_failure"

                history.append({"role": "action", "content": sample.action_text})
                history.append({"role": "observation", "content": observation})

            elif parsed.name == "finish" and parsed.valid:
                step_reward += self.config.format_reward_weight
                if parsed.return_type == "give_answer":
                    step_reward += self.config.finish_reward_weight
                    all_relevant_called = set(
                        task_row.relevant_function_names
                    ).issubset(called_apis)
                    won = all_relevant_called
                    terminal_reason = (
                        "finish_give_answer_success"
                        if won
                        else "finish_give_answer_incomplete"
                    )
                else:
                    terminal_reason = "finish_give_up"
                terminal = True
                observation = f"[FINISH:{parsed.return_type}]"
                history.append({"role": "action", "content": sample.action_text})
                history.append({"role": "observation", "content": observation})

            else:
                observation = f"[INVALID_ACTION] {parsed.error or 'parse failure'}"
                history.append({"role": "action", "content": sample.action_text})
                history.append({"role": "observation", "content": observation})

            if step_index + 1 >= max_steps:
                terminal = True
                if terminal_reason == "max_episode_steps":
                    terminal_reason = "max_episode_steps"

            state_hash = canonical_sha256(
                {"history": history, "step": step_index}
            )

            records.append(
                TrainingTrajectoryRecord(
                    dataset_id=f"online:{self.manifest_sha256}",
                    benchmark="toolbench_native",
                    task_id=task_row.query_id,
                    split=task_row.test_set,
                    seed=policy_seed,
                    traj_uid=traj_uid,
                    group_uid=group_uid,
                    step_idx=step_index,
                    instruction=task_row.query,
                    observation=json.dumps(
                        {"last_observation": history[-1]["content"] if history else ""},
                        sort_keys=True,
                    ),
                    action_raw=sample.action_text,
                    action_type=parsed.name,
                    parsed_action={
                        "name": parsed.name,
                        "valid": parsed.valid,
                        "api_name": parsed.api_name,
                        "parameters": parsed.parameters,
                        "return_type": parsed.return_type,
                        "error": parsed.error,
                    },
                    next_observation=observation,
                    reward=step_reward,
                    terminal=terminal,
                    success=won if terminal else False,
                    outcome_evidence={
                        "terminal_reason": terminal_reason,
                        "called_apis": sorted(called_apis),
                        "relevant_apis": list(task_row.relevant_function_names),
                    },
                    action_language="toolbench_call",
                    method_source="on_policy",
                    model_id=policy.component.name,
                    trajectory_length=None,
                    terminal_reason="pending",
                    decision_role="executor",
                    state_snapshot_sha256=state_hash,
                    goal_hash=canonical_sha256(
                        {"query_id": task_row.query_id, "test_set": task_row.test_set}
                    ),
                    remaining_budget=remaining,
                    action_affordance_hash=canonical_sha256(
                        {"action_type": "toolbench_call_finish", "free_text": True}
                    ),
                    rollout_config_hash=rollout_config_sha256,
                    active=True,
                    action_token_count=len(sample.action_token_ids),
                    prompt_hash=token_ids_sha256(sample.prompt_token_ids),
                    prompt_ids=sample.prompt_token_ids,
                    response_ids=sample.action_token_ids,
                    response_mask=(1,) * len(sample.action_token_ids),
                    response_old_logprobs=sample.old_action_token_log_probs,
                    extra={
                        "task_manifest_lock_sha256": self.manifest_sha256,
                        "test_set": task_row.test_set,
                        "step_reward": step_reward,
                    },
                )
            )

            if terminal:
                break

        trajectory_length = len(records)
        finalized: list[TrainingTrajectoryRecord] = []
        for row in records:
            finalized.append(
                TrainingTrajectoryRecord(
                    **{
                        **row.__dict__,
                        "trajectory_length": trajectory_length,
                        "terminal_reason": terminal_reason,
                        "success": won,
                    }
                )
            )
        return finalized
