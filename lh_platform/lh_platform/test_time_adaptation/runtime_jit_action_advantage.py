"""Frozen task-disjoint JIT action advantages for live runtime decisions.

The return and advantage arithmetic follows the official JitRL Jericho code
(discounted return, per-action mean minus overall mean, max-positive normalization,
episode-count weight).  This adaptation uses a frozen exact-state population and sparse
official rewards; it never adds an action outside the current environment capability
slate.

It is not an official JitRL reproduction.  Registered deviations from the Jericho code:
- retrieval is an exact ``RuntimeManagementStateKey`` match, with no n-gram or vector
  similarity, no dynamic threshold and no top-k;
- the exploration bonus is removed, and a decision falls back to the base policy when
  any legal action lacks support;
- the memory is frozen before evaluation instead of growing across episodes.
Results must be reported as "JitRL-style advantage on exact states", not as JitRL.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    ROTATING_RESIDUAL_SOURCE_METHOD,
    ROTATING_SETTLEMENT_SOURCE_METHOD,
)

_OFFICIAL_RECEIPT_SCHEMA = "terminal_managed_runtime_evaluation_v2"
_OFFICIAL_TASK_PREFIX = "terminal-bench/"


def _replay_official_receipt(path: Path) -> dict[str, object]:
    """Re-run the trial finalizer instead of trusting a detached receipt file."""

    if path.name != "runtime-evaluation-receipt.json" or path.parent.name != "agent":
        raise ValueError("runtime JIT evidence is not a canonical trial receipt")
    trial = path.parent.parent
    from lh_platform.benchmarks.terminal_managed_runtime_result import (
        finalize_terminal_managed_runtime_result,
        load_harbor_ctrf_evaluation_validity,
    )

    return finalize_terminal_managed_runtime_result(
        trial,
        evaluation_validity=load_harbor_ctrf_evaluation_validity(trial),
    )


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime JIT {field} must be a non-empty string")
    return value


def _canonical_terminal_task_id(official_task_name: str) -> str:
    """Map Harbor's official namespace to the frozen Terminal task identity."""

    if not official_task_name.startswith(_OFFICIAL_TASK_PREFIX):
        raise ValueError("runtime JIT official task namespace differs")
    task_id = official_task_name.removeprefix(_OFFICIAL_TASK_PREFIX)
    if not task_id or "/" in task_id:
        raise ValueError("runtime JIT official task identity is not canonical")
    return task_id


@dataclass(frozen=True)
class RuntimeManagementStateKey:
    """Describe transferable public runtime state without task-specific paths."""

    environment_family: str
    parent_contract_type: str
    failed_child_type: str
    mechanism_type: str
    phase: str
    capability_signature: str
    remaining_budget_bucket: str

    def __post_init__(self) -> None:
        if any(not value for value in self.to_dict().values()):
            raise ValueError("runtime JIT state fields must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "environment_family": self.environment_family,
            "parent_contract_type": self.parent_contract_type,
            "failed_child_type": self.failed_child_type,
            "mechanism_type": self.mechanism_type,
            "phase": self.phase,
            "capability_signature": self.capability_signature,
            "remaining_budget_bucket": self.remaining_budget_bucket,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeManagementStateKey:
        expected = {
            "environment_family",
            "parent_contract_type",
            "failed_child_type",
            "mechanism_type",
            "phase",
            "capability_signature",
            "remaining_budget_bucket",
        }
        if set(payload) != expected:
            raise ValueError("runtime JIT state payload fields changed")
        return cls(**{field: _required_string(payload, field) for field in expected})


@dataclass(frozen=True)
class RuntimeJITTaskIdentity:
    task_id: str
    group_id: str
    split: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.group_id or not self.split:
            raise ValueError("runtime JIT task identity is incomplete")

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "group_id": self.group_id,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeJITTaskIdentity:
        if set(payload) != {"task_id", "group_id", "split"}:
            raise ValueError("runtime JIT task identity fields changed")
        return cls(
            task_id=_required_string(payload, "task_id"),
            group_id=_required_string(payload, "group_id"),
            split=_required_string(payload, "split"),
        )


@dataclass(frozen=True)
class RuntimeJITPopulationManifest:
    """Freeze canonical task and group identities for train/test exclusion."""

    population_id: str
    benchmark_id: str
    source_split: str
    evaluation_split: str
    tasks: tuple[RuntimeJITTaskIdentity, ...]

    def __post_init__(self) -> None:
        if (
            not self.population_id
            or not self.benchmark_id
            or not self.source_split
            or not self.evaluation_split
            or self.source_split == self.evaluation_split
            or not self.tasks
        ):
            raise ValueError("runtime JIT population identity is invalid")
        task_ids = [item.task_id for item in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("runtime JIT population repeats a canonical task")
        if any(
            item.split not in {self.source_split, self.evaluation_split}
            for item in self.tasks
        ):
            raise ValueError("runtime JIT population contains an undeclared split")
        source = [item for item in self.tasks if item.split == self.source_split]
        evaluation = [item for item in self.tasks if item.split == self.evaluation_split]
        if not source or not evaluation:
            raise ValueError("runtime JIT population requires source and evaluation tasks")
        if {item.group_id for item in source}.intersection(
            item.group_id for item in evaluation
        ):
            raise ValueError("runtime JIT population is not group-disjoint")
        object.__setattr__(
            self,
            "tasks",
            tuple(sorted(self.tasks, key=lambda item: (item.split, item.task_id))),
        )

    def identity(self, task_id: str, *, split: str) -> RuntimeJITTaskIdentity:
        matches = [
            item for item in self.tasks if item.task_id == task_id and item.split == split
        ]
        if len(matches) != 1:
            raise ValueError("runtime JIT task is absent from its frozen split")
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "population_id": self.population_id,
            "benchmark_id": self.benchmark_id,
            "source_split": self.source_split,
            "evaluation_split": self.evaluation_split,
            "tasks": [item.to_dict() for item in self.tasks],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeJITPopulationManifest:
        expected = {
            "population_id",
            "benchmark_id",
            "source_split",
            "evaluation_split",
            "tasks",
        }
        raw_tasks = payload.get("tasks")
        if set(payload) != expected or not isinstance(raw_tasks, list):
            raise ValueError("runtime JIT population payload is invalid")
        if any(not isinstance(item, Mapping) for item in raw_tasks):
            raise ValueError("runtime JIT population task payload is invalid")
        return cls(
            population_id=_required_string(payload, "population_id"),
            benchmark_id=_required_string(payload, "benchmark_id"),
            source_split=_required_string(payload, "source_split"),
            evaluation_split=_required_string(payload, "evaluation_split"),
            tasks=tuple(RuntimeJITTaskIdentity.from_dict(item) for item in raw_tasks),
        )


def _load_official_reward(path: Path, *, expected_task_id: str) -> float:
    """Admit official task reward independently from incomplete cost accounting."""

    if not path.is_absolute() or not path.is_file():
        raise ValueError("runtime JIT official evidence must be an absolute file path")
    payload = json.loads(path.read_text())
    validity = payload.get("evaluation_validity") if isinstance(payload, dict) else None
    reward = payload.get("official_reward") if isinstance(payload, dict) else None
    token_accounting_complete = (
        payload.get("token_accounting_complete") if isinstance(payload, dict) else None
    )
    token_accounting_semantics = (
        payload.get("token_accounting_semantics") if isinstance(payload, dict) else None
    )
    required_fields = {
        "schema_version",
        "task_name",
        "runtime_method",
        "environment_session_id",
        "actor_projection_schema",
        "legacy_actor_projection_accepted",
        "evaluation_validity",
        "actor_action_count",
        "runtime_intervention_attempt_count",
        "runtime_intervention_completed_count",
        "runtime_intervention_failed_count",
        "runtime_intervention_cancelled_count",
        "official_reward",
        "official_success",
        "trial_exception",
        "public_task_status",
        "token_accounting_complete",
        "token_accounting_semantics",
        "scientific_result",
        "formal_experiment_authorized",
        "admissible_claim_scope",
    }
    validity_fields = {
        "source_artifact",
        "trial_name",
        "official_tests_started",
        "evaluation_infrastructure_succeeded",
        "test_count",
        "passed_count",
        "failed_count",
        "skipped_count",
        "pending_count",
        "other_count",
        "invalid_reason",
    }
    count_fields = (
        "test_count",
        "passed_count",
        "failed_count",
        "skipped_count",
        "pending_count",
        "other_count",
    )
    valid_counts = (
        isinstance(validity, dict)
        and set(validity) == validity_fields
        and all(
            not isinstance(validity.get(field), bool)
            and isinstance(validity.get(field), int)
            and int(validity[field]) >= 0
            for field in count_fields
        )
        and validity["test_count"]
        == sum(int(validity[field]) for field in count_fields[1:])
        and validity["test_count"] > 0
    )
    if (
        not isinstance(payload, dict)
        or not required_fields <= set(payload)
        or payload.get("schema_version") != _OFFICIAL_RECEIPT_SCHEMA
        or _canonical_terminal_task_id(str(payload.get("task_name", "")))
        != expected_task_id
        or not valid_counts
        or validity.get("official_tests_started") is not True
        or validity.get("evaluation_infrastructure_succeeded") is not True
        or validity.get("invalid_reason") is not None
        or isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or not 0.0 <= float(reward) <= 1.0
        or payload.get("official_success") is not (float(reward) == 1.0)
        or payload.get("trial_exception") is not None
        or not isinstance(token_accounting_complete, bool)
        or token_accounting_semantics
        != (
            "complete_message_end_usage"
            if token_accounting_complete
            else "lower_bound_from_received_message_end_usage"
        )
        or payload.get("public_task_status")
        != ("completed" if float(reward) == 1.0 else "failed")
    ):
        raise ValueError("runtime JIT official evidence receipt is invalid")
    replayed = _replay_official_receipt(path)
    if replayed != payload:
        raise ValueError("runtime JIT official receipt differs from trial replay")
    return float(reward)


def load_official_runtime_reward(path: Path, *, expected_task_id: str) -> float:
    """Expose the replay-validated official reward to aligned JIT memories."""

    return _load_official_reward(path, expected_task_id=expected_task_id)


def replay_official_runtime_receipt(path: Path) -> dict[str, object]:
    """Expose canonical trial replay to continuation-aligned source loaders."""

    return _replay_official_receipt(path)


@dataclass(frozen=True)
class RuntimeJITExperience:
    """Store one historical action whose terminal reward has official evidence."""

    task_id: str
    episode_number: int
    state: RuntimeManagementStateKey
    action_type: RuntimeJointActionType
    future_rewards: tuple[float, ...]
    official_evidence_path: str

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or isinstance(self.episode_number, bool)
            or not isinstance(self.episode_number, int)
            or self.episode_number < 1
            or not isinstance(self.action_type, RuntimeJointActionType)
            or not self.future_rewards
            or not self.official_evidence_path
            or any(not math.isfinite(float(value)) for value in self.future_rewards)
        ):
            raise ValueError("runtime JIT experience identity or rewards are invalid")

    def discounted_return(self, gamma: float) -> float:
        return sum(
            (gamma**offset) * float(reward)
            for offset, reward in enumerate(self.future_rewards)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "episode_number": self.episode_number,
            "state": self.state.to_dict(),
            "action_type": self.action_type.value,
            "future_rewards": list(self.future_rewards),
            "official_evidence_path": self.official_evidence_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeJITExperience:
        expected = {
            "task_id",
            "episode_number",
            "state",
            "action_type",
            "future_rewards",
            "official_evidence_path",
        }
        state = payload.get("state")
        rewards = payload.get("future_rewards")
        episode = payload.get("episode_number")
        if (
            set(payload) != expected
            or not isinstance(state, Mapping)
            or not isinstance(rewards, list)
            or not rewards
            or isinstance(episode, bool)
            or not isinstance(episode, int)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in rewards
            )
        ):
            raise ValueError("runtime JIT experience payload is invalid")
        return cls(
            task_id=_required_string(payload, "task_id"),
            episode_number=episode,
            state=RuntimeManagementStateKey.from_dict(state),
            action_type=RuntimeJointActionType(
                _required_string(payload, "action_type")
            ),
            future_rewards=tuple(float(value) for value in rewards),
            official_evidence_path=_required_string(payload, "official_evidence_path"),
        )


def settlement_management_state(
    slate: RuntimeJointActionSlate,
) -> RuntimeManagementStateKey:
    """Project a shared settlement slate into one task-agnostic retrieval key."""

    action_types = tuple(item.action_type for item in slate.candidates)
    expected = (
        RuntimeJointActionType.CONTINUE_ACTOR,
        RuntimeJointActionType.REQUEST_SEMANTIC_CHECK,
    )
    semantic = next(
        (
            item
            for item in slate.candidates
            if item.action_type is RuntimeJointActionType.REQUEST_SEMANTIC_CHECK
        ),
        None,
    )
    version_key = (
        semantic.target_binding.obligation_version_key
        if semantic is not None
        else None
    )
    if action_types != expected or version_key is None:
        raise ValueError("runtime JIT source settlement slate differs")
    return RuntimeManagementStateKey(
        environment_family="terminal",
        parent_contract_type="produce_artifact",
        failed_child_type="verify_semantics",
        mechanism_type="natural_settlement_review",
        phase=version_key[-1],
        capability_signature=slate.capability_signature,
        remaining_budget_bucket=f"b{slate.budget_before}",
    )


def residual_management_state(
    slate: RuntimeJointActionSlate,
) -> RuntimeManagementStateKey:
    """Project one actionable midstream slate without task paths or hidden facts."""

    actionable = tuple(
        item
        for item in slate.candidates
        if item.action_type is not RuntimeJointActionType.CONTINUE_ACTOR
    )
    if not actionable:
        raise ValueError("runtime JIT residual slate has no management action")
    fresh_first_failure = any(
        item.action_type is RuntimeJointActionType.REQUEST_FAILURE_DIAGNOSIS
        and item.instance_action_id.split(":", 3)[:3]
        == ["diagnose-failure", slate.trigger_action_id, "1"]
        for item in actionable
    )
    phases = {
        item.target_binding.obligation_version_key[-1]
        for item in actionable
        if item.target_binding.obligation_version_key is not None
    }
    # A failure-management choice is caused by the observed transition. Output
    # existence is a separate fact and must not fragment equivalent decisions.
    phase = (
        "actor_action_failed_without_effect"
        if fresh_first_failure
        else next(iter(phases))
        if len(phases) == 1
        else "mixed_runtime_state"
    )
    action_types = "|".join(sorted({item.action_type.value for item in actionable}))
    return RuntimeManagementStateKey(
        environment_family="terminal",
        parent_contract_type="produce_artifact",
        failed_child_type=action_types,
        mechanism_type=(
            "fresh_first_failure_management"
            if fresh_first_failure
            else "midstream_residual_management"
        ),
        phase=phase,
        capability_signature=slate.capability_signature,
        remaining_budget_bucket=f"b{slate.budget_before}",
    )


def load_settlement_source_experience(
    *,
    official_evidence_path: Path,
    population: RuntimeJITPopulationManifest,
    episode_number: int,
) -> RuntimeJITExperience:
    """Admit one replay-valid frozen arm selected at the natural Pi boundary."""

    receipt = _replay_official_receipt(official_evidence_path)
    task_id = _canonical_terminal_task_id(_required_string(receipt, "task_name"))
    population.identity(task_id, split=population.source_split)
    if receipt.get("runtime_method") != ROTATING_SETTLEMENT_SOURCE_METHOD:
        raise ValueError("runtime JIT source method differs")
    trial = official_evidence_path.parent.parent
    runtime_config = json.loads((trial / "agent/runtime-config.json").read_text())
    if not isinstance(runtime_config, dict):
        raise ValueError("runtime JIT source arm identity differs")
    arm_index = runtime_config.get("joint_source_arm_index")
    if (
        runtime_config.get("runtime_method") != ROTATING_SETTLEMENT_SOURCE_METHOD
        or runtime_config.get("managed_state_visibility")
        != "host_private_until_actor_exit_v1"
        or isinstance(arm_index, bool)
        or not isinstance(arm_index, int)
        or arm_index < 0
    ):
        raise ValueError("runtime JIT source arm identity differs")
    decision_path = trial / "agent/runtime-decisions.jsonl"
    settlement_rows = []
    for line in decision_path.read_text().splitlines():
        row = json.loads(line)
        payload = row.get("payload") if isinstance(row, dict) else None
        slate_payload = payload.get("slate") if isinstance(payload, dict) else None
        if (
            row.get("event_type") == "decision_selected"
            and isinstance(slate_payload, Mapping)
            and str(slate_payload.get("trigger_action_id", "")).startswith(
                "actor-bridge-natural-settlement-"
            )
        ):
            settlement_rows.append((payload, RuntimeJointActionSlate.from_dict(slate_payload)))
    if len(settlement_rows) != 1:
        raise ValueError("runtime JIT source settlement decision differs")
    payload, slate = settlement_rows[0]
    state = settlement_management_state(slate)
    selected_index = next(
        (
            index
            for index, item in enumerate(slate.candidates)
            if item.instance_action_id == payload.get("selected_instance_action_id")
            and item.action_type.value == payload.get("selected_action_type")
        ),
        None,
    )
    if (
        payload.get("method_id") != ROTATING_SETTLEMENT_SOURCE_METHOD
        or payload.get("source_arm_index") != arm_index
        or payload.get("rule_id") != "frozen_rotating_joint_source_arm_v1"
        or selected_index is None
        or selected_index != arm_index % len(slate.candidates)
        or payload.get("bellman_values") is not None
        or payload.get("jit_advantages") is not None
    ):
        raise ValueError("runtime JIT frozen source selection differs")
    official_reward = _load_official_reward(
        official_evidence_path,
        expected_task_id=task_id,
    )
    return RuntimeJITExperience(
        task_id=task_id,
        episode_number=episode_number,
        state=state,
        action_type=slate.candidates[selected_index].action_type,
        future_rewards=(official_reward,),
        official_evidence_path=str(official_evidence_path.resolve()),
    )


def load_residual_source_experience(
    *,
    official_evidence_path: Path,
    population: RuntimeJITPopulationManifest,
    episode_number: int,
) -> RuntimeJITExperience:
    """Admit the first task-agnostic midstream choice from a one-action source run."""

    receipt = _replay_official_receipt(official_evidence_path)
    task_id = _canonical_terminal_task_id(_required_string(receipt, "task_name"))
    population.identity(task_id, split=population.source_split)
    if receipt.get("runtime_method") != ROTATING_RESIDUAL_SOURCE_METHOD:
        raise ValueError("runtime JIT residual source method differs")
    trial = official_evidence_path.parent.parent
    runtime_config = json.loads((trial / "agent/runtime-config.json").read_text())
    arm_index = runtime_config.get("joint_source_arm_index")
    if (
        runtime_config.get("runtime_method") != ROTATING_RESIDUAL_SOURCE_METHOD
        or runtime_config.get("managed_state_visibility")
        != "host_private_until_actor_exit_v1"
        or runtime_config.get("runtime_intervention_budget") != 1
        or isinstance(arm_index, bool)
        or not isinstance(arm_index, int)
        or arm_index < 0
    ):
        raise ValueError("runtime JIT residual source identity differs")
    eligible: list[tuple[dict[str, object], RuntimeJointActionSlate]] = []
    for line in (trial / "agent/runtime-decisions.jsonl").read_text().splitlines():
        row = json.loads(line)
        payload = row.get("payload") if isinstance(row, dict) else None
        slate_payload = payload.get("slate") if isinstance(payload, dict) else None
        if row.get("event_type") != "decision_selected" or not isinstance(
            slate_payload, Mapping
        ):
            continue
        slate = RuntimeJointActionSlate.from_dict(slate_payload)
        if (
            len(slate.candidates) > 1
            and not slate.trigger_action_id.startswith("actor-bridge-")
        ):
            eligible.append((payload, slate))
    if not eligible:
        raise ValueError("runtime JIT residual source has no actionable midstream slate")
    payload, slate = eligible[0]
    selected_index = next(
        (
            index
            for index, item in enumerate(slate.candidates)
            if item.instance_action_id == payload.get("selected_instance_action_id")
            and item.action_type.value == payload.get("selected_action_type")
        ),
        None,
    )
    if (
        payload.get("method_id") != ROTATING_RESIDUAL_SOURCE_METHOD
        or payload.get("source_arm_index") != arm_index
        or payload.get("rule_id") != "frozen_rotating_joint_source_arm_v1"
        or selected_index is None
        or selected_index != arm_index % len(slate.candidates)
        or payload.get("bellman_values") is not None
        or payload.get("jit_advantages") is not None
    ):
        raise ValueError("runtime JIT residual source selection differs")
    selected = slate.candidates[selected_index]
    later_selected_types = {
        str(item_payload.get("selected_action_type"))
        for item_payload, _ in eligible[1:]
    }
    if (
        selected.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        and later_selected_types - {RuntimeJointActionType.CONTINUE_ACTOR.value}
    ):
        raise ValueError("continued residual source later changed its exploration arm")
    official_reward = _load_official_reward(
        official_evidence_path,
        expected_task_id=task_id,
    )
    return RuntimeJITExperience(
        task_id=task_id,
        episode_number=episode_number,
        state=residual_management_state(slate),
        action_type=selected.action_type,
        future_rewards=(official_reward,),
        official_evidence_path=str(official_evidence_path.resolve()),
    )


@dataclass(frozen=True)
class RuntimeJITActionScore:
    instance_action_id: str
    action_type: RuntimeJointActionType
    base_score: float
    support_count: int
    task_support_count: int
    average_return: float | None
    raw_advantage: float
    normalized_advantage: float
    corrected_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "action_type": self.action_type.value,
        }


@dataclass(frozen=True)
class RuntimeLegalActionSlate:
    """Bind base scores to exact instances from one compiler-owned joint slate."""

    joint_slate: RuntimeJointActionSlate
    instance_base_scores: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.instance_base_scores:
            raise ValueError("runtime JIT legal action slate is incomplete")
        action_ids = [item[0] for item in self.instance_base_scores]
        expected_ids = {item.instance_action_id for item in self.joint_slate.candidates}
        if (
            len(action_ids) != len(set(action_ids))
            or set(action_ids) != expected_ids
            or any(
                not action_id
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
                for action_id, score in self.instance_base_scores
            )
        ):
            raise ValueError("runtime JIT legal action slate is invalid")
        object.__setattr__(
            self,
            "instance_base_scores",
            tuple(sorted((key, float(value)) for key, value in self.instance_base_scores)),
        )

    @property
    def capability_signature(self) -> str:
        return self.joint_slate.capability_signature


@dataclass(frozen=True)
class RuntimeJITDecision:
    chosen_instance_action_id: str
    chosen_action_type: RuntimeJointActionType
    admitted: bool
    fallback_reason: str | None
    population_id: str
    test_task_id: str
    test_group_id: str
    gamma: float
    minimum_task_support_per_action: int
    retrieved_experience_count: int
    retrieved_task_count: int
    overall_average_return: float | None
    frozen_episode_count: int
    episode_weight: float
    scores: tuple[RuntimeJITActionScore, ...]
    evidence_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                field: value
                for field, value in self.__dict__.items()
                if field not in {"scores", "evidence_paths"}
            },
            "scores": [score.to_dict() for score in self.scores],
            "evidence_paths": list(self.evidence_paths),
            "chosen_action_type": self.chosen_action_type.value,
        }


class FrozenRuntimeJITMemory:
    """Collect source-split outcomes, then expose immutable held-out retrieval."""

    schema_version = "frozen_runtime_jit_memory_v3"

    def __init__(
        self,
        *,
        population: RuntimeJITPopulationManifest,
        gamma: float,
        minimum_task_support_per_action: int,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("runtime JIT gamma must be in [0, 1]")
        if minimum_task_support_per_action < 1:
            raise ValueError("runtime JIT support gate must be positive")
        self.population = population
        self.gamma = float(gamma)
        self.minimum_task_support_per_action = minimum_task_support_per_action
        self._experiences: list[RuntimeJITExperience] = []
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def experiences(self) -> tuple[RuntimeJITExperience, ...]:
        return tuple(self._experiences)

    @property
    def frozen_episode_count(self) -> int:
        return len({item.episode_number for item in self._experiences})

    def record(self, experience: RuntimeJITExperience) -> None:
        """Validate canonical source identity and independent official reward evidence."""

        if self._frozen:
            raise RuntimeError("runtime JIT memory is frozen")
        self.population.identity(
            experience.task_id,
            split=self.population.source_split,
        )
        official_reward = _load_official_reward(
            Path(experience.official_evidence_path),
            expected_task_id=experience.task_id,
        )
        # This first adaptation uses sparse official reward only; intermediate
        # self-evaluations are not admitted as JitRL step rewards.
        if any(value != 0.0 for value in experience.future_rewards[:-1]) or not math.isclose(
            experience.future_rewards[-1],
            official_reward,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("runtime JIT reward sequence differs from official evidence")
        if experience in self._experiences:
            raise ValueError("runtime JIT experience is duplicated")
        reused_receipt_episodes = {
            item.episode_number
            for item in self._experiences
            if item.official_evidence_path == experience.official_evidence_path
        }
        if reused_receipt_episodes and reused_receipt_episodes != {
            experience.episode_number
        }:
            raise ValueError("one official receipt cannot support multiple episodes")
        self._experiences.append(experience)

    def freeze(self) -> None:
        if not self._experiences:
            raise ValueError("runtime JIT memory cannot freeze without experience")
        episodes = {item.episode_number for item in self._experiences}
        if episodes != set(range(1, len(episodes) + 1)):
            raise ValueError("runtime JIT episode numbers must be contiguous from one")
        tasks_by_episode: dict[int, set[str]] = {}
        for item in self._experiences:
            tasks_by_episode.setdefault(item.episode_number, set()).add(item.task_id)
        if any(len(tasks) != 1 for tasks in tasks_by_episode.values()):
            raise ValueError("one runtime JIT episode cannot combine different tasks")
        self._frozen = True

    def save(self, path: Path) -> None:
        if not self._frozen:
            raise RuntimeError("runtime JIT memory must be frozen before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "population": self.population.to_dict(),
                    "gamma": self.gamma,
                    "minimum_task_support_per_action": (
                        self.minimum_task_support_per_action
                    ),
                    "frozen": True,
                    "experiences": [item.to_dict() for item in self._experiences],
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> FrozenRuntimeJITMemory:
        payload = json.loads(path.read_text())
        expected = {
            "schema_version",
            "population",
            "gamma",
            "minimum_task_support_per_action",
            "frozen",
            "experiences",
        }
        population = payload.get("population") if isinstance(payload, dict) else None
        experiences = payload.get("experiences") if isinstance(payload, dict) else None
        gamma = payload.get("gamma") if isinstance(payload, dict) else None
        support = (
            payload.get("minimum_task_support_per_action")
            if isinstance(payload, dict)
            else None
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema_version") != cls.schema_version
            or payload.get("frozen") is not True
            or not isinstance(population, Mapping)
            or not isinstance(experiences, list)
            or isinstance(gamma, bool)
            or not isinstance(gamma, (int, float))
            or isinstance(support, bool)
            or not isinstance(support, int)
        ):
            raise ValueError("runtime JIT memory artifact is invalid")
        memory = cls(
            population=RuntimeJITPopulationManifest.from_dict(population),
            gamma=float(gamma),
            minimum_task_support_per_action=support,
        )
        for row in experiences:
            if not isinstance(row, Mapping):
                raise ValueError("runtime JIT memory experience is invalid")
            memory.record(RuntimeJITExperience.from_dict(row))
        memory.freeze()
        return memory

    def score_legal_actions(
        self,
        *,
        state: RuntimeManagementStateKey,
        test_task_id: str,
        legal_slate: RuntimeLegalActionSlate,
    ) -> RuntimeJITDecision:
        """Correct scores using only the frozen source split and current legal slate."""

        if not self._frozen:
            raise RuntimeError("runtime JIT memory must be frozen before retrieval")
        test_identity = self.population.identity(
            test_task_id,
            split=self.population.evaluation_split,
        )
        if state.capability_signature != legal_slate.capability_signature:
            raise ValueError("runtime JIT state and live capability slate differ")
        legal_base_scores = dict(legal_slate.instance_base_scores)
        candidates_by_id = {
            item.instance_action_id: item
            for item in legal_slate.joint_slate.candidates
        }
        legal_action_types = frozenset(
            item.action_type for item in legal_slate.joint_slate.candidates
        )
        matches = tuple(
            item
            for item in self._experiences
            if item.state == state and item.action_type in legal_action_types
        )
        returns_by_action = {action_type: [] for action_type in legal_action_types}
        tasks_by_action = {action_type: set() for action_type in legal_action_types}
        for item in matches:
            returns_by_action[item.action_type].append(
                item.discounted_return(self.gamma)
            )
            tasks_by_action[item.action_type].add(item.task_id)

        admitted = all(
            len(tasks_by_action[action_type]) >= self.minimum_task_support_per_action
            for action_type in legal_action_types
        )
        episode_weight = min(
            1.0 + (self.frozen_episode_count / 50.0) * 0.5,
            1.5,
        )
        fallback_reason = None
        overall_average: float | None = None
        average_returns: dict[str, float] = {}
        raw_advantages = {action_type: 0.0 for action_type in legal_action_types}
        normalized_advantages = {
            action_type: 0.0 for action_type in legal_action_types
        }
        if admitted:
            all_returns = [value for values in returns_by_action.values() for value in values]
            overall_average = sum(all_returns) / len(all_returns)
            average_returns = {
                action_type: sum(values) / len(values)
                for action_type, values in returns_by_action.items()
            }
            raw_advantages = {
                action_type: average_returns[action_type] - overall_average
                for action_type in legal_action_types
            }
            positive = [value for value in raw_advantages.values() if value > 0.0]
            denominator = (
                max(positive)
                if positive
                else abs(min(raw_advantages.values(), default=0.0))
            )
            if denominator > 0.0:
                normalized_advantages = {
                    action_type: value / denominator
                    for action_type, value in raw_advantages.items()
                }
        else:
            fallback_reason = "insufficient_task_disjoint_support_for_complete_legal_slate"

        scores = tuple(
            RuntimeJITActionScore(
                instance_action_id=instance_action_id,
                action_type=candidates_by_id[instance_action_id].action_type,
                base_score=float(base_score),
                support_count=len(
                    returns_by_action[candidates_by_id[instance_action_id].action_type]
                ),
                task_support_count=len(
                    tasks_by_action[candidates_by_id[instance_action_id].action_type]
                ),
                average_return=average_returns.get(
                    candidates_by_id[instance_action_id].action_type
                ),
                raw_advantage=raw_advantages[
                    candidates_by_id[instance_action_id].action_type
                ],
                normalized_advantage=normalized_advantages[
                    candidates_by_id[instance_action_id].action_type
                ],
                corrected_score=(
                    float(base_score)
                    + episode_weight
                    * normalized_advantages[
                        candidates_by_id[instance_action_id].action_type
                    ]
                ),
            )
            for instance_action_id, base_score in legal_base_scores.items()
        )
        chosen = max(scores, key=lambda item: item.corrected_score)
        return RuntimeJITDecision(
            chosen_instance_action_id=chosen.instance_action_id,
            chosen_action_type=chosen.action_type,
            admitted=admitted,
            fallback_reason=fallback_reason,
            population_id=self.population.population_id,
            test_task_id=test_identity.task_id,
            test_group_id=test_identity.group_id,
            gamma=self.gamma,
            minimum_task_support_per_action=self.minimum_task_support_per_action,
            retrieved_experience_count=len(matches),
            retrieved_task_count=len({item.task_id for item in matches}),
            overall_average_return=overall_average,
            frozen_episode_count=self.frozen_episode_count,
            episode_weight=episode_weight,
            scores=scores,
            evidence_paths=tuple(item.official_evidence_path for item in matches),
        )


__all__ = [
    "FrozenRuntimeJITMemory",
    "RuntimeJITActionScore",
    "RuntimeJITDecision",
    "RuntimeJITExperience",
    "RuntimeJITPopulationManifest",
    "RuntimeJITTaskIdentity",
    "RuntimeLegalActionSlate",
    "RuntimeManagementStateKey",
    "load_settlement_source_experience",
    "load_residual_source_experience",
    "load_official_runtime_reward",
    "replay_official_runtime_receipt",
    "residual_management_state",
    "settlement_management_state",
]
