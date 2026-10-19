"""Apply task-disjoint experience to the live Runtime-owned Bellman slate.

The official JitRL implementation motivates retrieval-time action-value
correction. This locally designed adaptation keeps the executable action slate
and Bellman values Runtime-owned, requires an identical continuation contract,
and uses bounded posterior shrinkage instead of max-normalizing any nonzero
sample difference to unit magnitude.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from lh_platform.test_time_adaptation.runtime_jit_action_advantage import (
    RuntimeJITPopulationManifest,
    load_official_runtime_reward,
    replay_official_runtime_receipt,
)
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import (
    FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD,
    TRACEDP_RUNTIME_OWNED_OPTION_DP_DEV_METHOD,
    RuntimeJointResolvedChoice,
)
from lh_platform.test_time_adaptation.runtime_option_jit_v2 import (
    SCHEMA_VERSION as JIT_V2_SCHEMA_VERSION,
)
from lh_platform.test_time_adaptation.runtime_option_jit_v2 import (
    FrozenRuntimeOptionJITMemoryV2,
    RecoveryActionKeyV2,
    RecoveryStateView,
    recovery_state_key_v2_from_view,
)
from lh_platform.test_time_adaptation.runtime_owned_option_dp import (
    RuntimeAtomicOutcomeEstimate,
    RuntimeOwnedOptionProjection,
)
from lh_platform.test_time_adaptation.runtime_owned_option_live import (
    RuntimeOwnedOptionLiveResolver,
)
from lh_platform.test_time_adaptation.runtime_time_aware_trace_dp import (
    RuntimeTimeAwarePolicyValue,
)

_VALUE_FIELDS = {
    "residual_completion_proxy_probability",
    "expected_unblocked_fraction",
    "expected_duration_ms",
    "expected_false_interventions",
    "expected_intervention_cost",
}


@dataclass(frozen=True)
class RuntimeOptionJITContinuationContract:
    """Identify the policy executed after a forced first Runtime option."""

    continuation_policy_id: str
    actor_cell_id: str
    trigger_contract_id: str
    action_domain_contract_id: str
    objective_id: str
    budget_semantics_id: str

    def __post_init__(self) -> None:
        if any(not str(value).strip() for value in asdict(self).values()):
            raise ValueError("runtime option JIT continuation contract is incomplete")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> RuntimeOptionJITContinuationContract:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("runtime option JIT continuation contract fields changed")
        return cls(**{key: str(value) for key, value in payload.items()})


@dataclass(frozen=True)
class RuntimeOptionJITStateKey:
    """Project transferable residual state without task ids or file paths."""

    environment_family: str
    capability_signature: str
    evidence_state_counts: tuple[tuple[str, int], ...]
    incident_status_counts: tuple[tuple[str, int], ...]
    task_status_counts: tuple[tuple[str, int], ...]
    remaining_budget_bucket: str

    def __post_init__(self) -> None:
        count_groups = (
            self.evidence_state_counts,
            self.incident_status_counts,
            self.task_status_counts,
        )
        if (
            not self.environment_family
            or not self.capability_signature
            or not self.remaining_budget_bucket
            or any(tuple(sorted(group)) != group for group in count_groups)
            or any(value < 0 for group in count_groups for _, value in group)
        ):
            raise ValueError("runtime option JIT state key is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "environment_family": self.environment_family,
            "capability_signature": self.capability_signature,
            "evidence_state_counts": [list(item) for item in self.evidence_state_counts],
            "incident_status_counts": [list(item) for item in self.incident_status_counts],
            "task_status_counts": [list(item) for item in self.task_status_counts],
            "remaining_budget_bucket": self.remaining_budget_bucket,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeOptionJITStateKey:
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("runtime option JIT state key fields changed")
        groups = {}
        for field in (
            "evidence_state_counts",
            "incident_status_counts",
            "task_status_counts",
        ):
            raw = payload.get(field)
            if not isinstance(raw, list) or any(
                not isinstance(item, list) or len(item) != 2 for item in raw
            ):
                raise ValueError("runtime option JIT state counts are invalid")
            groups[field] = tuple((str(item[0]), int(item[1])) for item in raw)
        return cls(
            environment_family=str(payload["environment_family"]),
            capability_signature=str(payload["capability_signature"]),
            remaining_budget_bucket=str(payload["remaining_budget_bucket"]),
            **groups,
        )


def option_jit_state_key(
    *, slate: RuntimeJointActionSlate, projection: RuntimeOwnedOptionProjection
) -> RuntimeOptionJITStateKey:
    """Compile the exact public state abstraction used by source and target."""

    return RuntimeOptionJITStateKey(
        environment_family="terminal",
        capability_signature=slate.capability_signature,
        evidence_state_counts=tuple(
            sorted(Counter(item.evidence_state for item in projection.acceptance_predicates).items())
        ),
        incident_status_counts=tuple(
            sorted(
                Counter(
                    item.incident_status or "no_incident"
                    for item in projection.acceptance_predicates
                ).items()
            )
        ),
        task_status_counts=tuple(
            sorted(Counter(item.status for item in projection.task_dependencies).items())
        ),
        remaining_budget_bucket=f"b{slate.budget_before}",
    )


@dataclass(frozen=True)
class RuntimeOptionJITExperience:
    """Store one forced first option followed by the declared continuation."""

    task_id: str
    episode_number: int
    state: RuntimeOptionJITStateKey
    action_type: RuntimeJointActionType
    continuation: RuntimeOptionJITContinuationContract
    official_reward: float
    official_evidence_path: str

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or isinstance(self.episode_number, bool)
            or not isinstance(self.episode_number, int)
            or self.episode_number < 1
            or not isinstance(self.action_type, RuntimeJointActionType)
            or self.official_reward not in {0.0, 1.0}
            or not self.official_evidence_path
        ):
            raise ValueError("runtime option JIT experience is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "episode_number": self.episode_number,
            "state": self.state.to_dict(),
            "action_type": self.action_type.value,
            "continuation": self.continuation.to_dict(),
            "official_reward": self.official_reward,
            "official_evidence_path": self.official_evidence_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeOptionJITExperience:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("runtime option JIT experience fields changed")
        state = payload.get("state")
        continuation = payload.get("continuation")
        reward = payload.get("official_reward")
        episode = payload.get("episode_number")
        if (
            not isinstance(state, Mapping)
            or not isinstance(continuation, Mapping)
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or isinstance(episode, bool)
            or not isinstance(episode, int)
        ):
            raise ValueError("runtime option JIT experience payload is invalid")
        return cls(
            task_id=str(payload["task_id"]),
            episode_number=episode,
            state=RuntimeOptionJITStateKey.from_dict(state),
            action_type=RuntimeJointActionType(str(payload["action_type"])),
            continuation=RuntimeOptionJITContinuationContract.from_dict(continuation),
            official_reward=float(reward),
            official_evidence_path=str(payload["official_evidence_path"]),
        )


def load_runtime_option_source_experience(
    *,
    official_evidence_path: Path,
    population: RuntimeJITPopulationManifest,
    episode_number: int,
) -> RuntimeOptionJITExperience:
    """Rebuild a forced option and its DP continuation from the canonical trial."""

    receipt = replay_official_runtime_receipt(official_evidence_path)
    task_name = receipt.get("task_name")
    if not isinstance(task_name, str) or not task_name.startswith("terminal-bench/"):
        raise ValueError("runtime option JIT source task namespace differs")
    task_id = task_name.removeprefix("terminal-bench/")
    population.identity(task_id, split=population.source_split)
    if receipt.get("runtime_method") != FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD:
        raise ValueError("runtime option JIT source method differs")
    trial = official_evidence_path.parent.parent
    config = json.loads((trial / "agent/runtime-config.json").read_text())
    actor = config.get("actor_runtime") if isinstance(config, dict) else None
    arm_index = config.get("joint_source_arm_index") if isinstance(config, dict) else None
    if (
        not isinstance(config, dict)
        or config.get("runtime_method") != FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD
        or isinstance(arm_index, bool)
        or not isinstance(arm_index, int)
        or arm_index < 0
        or not isinstance(actor, Mapping)
        or any(
            not isinstance(actor.get(field), str) or not actor[field]
            for field in ("provider", "model", "thinking", "transport")
        )
    ):
        raise ValueError("runtime option JIT source configuration differs")
    decisions = []
    for line in (trial / "agent/runtime-decisions.jsonl").read_text().splitlines():
        row = json.loads(line)
        payload = row.get("payload") if isinstance(row, dict) else None
        if row.get("event_type") == "decision_selected" and isinstance(payload, Mapping):
            decisions.append(payload)
    forced = [
        payload
        for payload in decisions
        if isinstance(payload.get("source_protocol"), Mapping)
        and payload["source_protocol"].get("phase") == "forced_first_action"
    ]
    if len(forced) != 1:
        raise ValueError("runtime option JIT source has no unique forced decision")
    payload = forced[0]
    slate_payload = payload.get("slate")
    model = payload.get("planning_model")
    if not isinstance(slate_payload, Mapping) or not isinstance(model, Mapping):
        raise ValueError("runtime option JIT source lacks its DP state")
    slate = RuntimeJointActionSlate.from_dict(slate_payload)
    if arm_index >= len(slate.candidates):
        raise ValueError("runtime option JIT source arm is outside the slate")
    selected = slate.candidates[arm_index]
    protocol = payload["source_protocol"]
    if (
        payload.get("method_id") != FORCED_FIRST_OPTION_DP_SOURCE_DEV_METHOD
        or payload.get("rule_id") != "forced_first_option_dp_source_arm_v1"
        or payload.get("selected_instance_action_id") != selected.instance_action_id
        or payload.get("selected_action_type") != selected.action_type.value
        or protocol.get("continuation_policy_id")
        != TRACEDP_RUNTIME_OWNED_OPTION_DP_DEV_METHOD
        or protocol.get("first_action_index") != arm_index
    ):
        raise ValueError("runtime option JIT forced source selection differs")
    later = decisions[decisions.index(payload) + 1 :]
    if any(
        not isinstance(item.get("source_protocol"), Mapping)
        or item["source_protocol"].get("phase") != "continuation"
        or item["source_protocol"].get("continuation_policy_id")
        != TRACEDP_RUNTIME_OWNED_OPTION_DP_DEV_METHOD
        or str(item.get("rule_id", "")).startswith("forced_first")
        for item in later
    ):
        raise ValueError("runtime option JIT source continuation policy differs")
    projection_payload = model.get("projection")
    if not isinstance(projection_payload, Mapping):
        raise ValueError("runtime option JIT source projection is missing")
    projection = RuntimeOwnedOptionProjection.from_dict(projection_payload)
    continuation = RuntimeOptionJITContinuationContract(
        continuation_policy_id=TRACEDP_RUNTIME_OWNED_OPTION_DP_DEV_METHOD,
        actor_cell_id=(
            f"{actor['provider']}/{actor['model']}|thinking={actor['thinking']}|"
            f"transport={actor['transport']}"
        ),
        trigger_contract_id="runtime_option_chain_v1",
        action_domain_contract_id="runtime_option_action_domain_v1",
        objective_id=(
            "residual_completion_then_unblocked_then_time_then_false_"
            "interventions_then_intervention_cost_v1"
        ),
        budget_semantics_id="runtime_option_chain_budget_v1",
    )
    return RuntimeOptionJITExperience(
        task_id=task_id,
        episode_number=episode_number,
        state=option_jit_state_key(slate=slate, projection=projection),
        action_type=selected.action_type,
        continuation=continuation,
        official_reward=load_official_runtime_reward(
            official_evidence_path, expected_task_id=task_id
        ),
        official_evidence_path=str(official_evidence_path.resolve()),
    )


@dataclass(frozen=True)
class RuntimeOptionJITCorrection:
    action_type: RuntimeJointActionType
    support_count: int
    task_support_count: int
    average_official_reward: float | None
    posterior_mean: float | None
    empirical_variance: float | None
    raw_advantage: float
    reliability: float
    applied_correction: float

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "action_type": self.action_type.value}


class FrozenRuntimeOptionJITMemory:
    """Freeze continuation-aligned source episodes and bounded correction settings."""

    schema_version = "frozen_runtime_option_jit_memory_v1"

    def __init__(
        self,
        *,
        population: RuntimeJITPopulationManifest,
        continuation: RuntimeOptionJITContinuationContract,
        minimum_task_support_per_action: int,
        prior_strength: float = 4.0,
        maximum_memory_weight: float = 0.5,
    ) -> None:
        if (
            minimum_task_support_per_action < 1
            or not math.isfinite(prior_strength)
            or prior_strength <= 0.0
            or not math.isfinite(maximum_memory_weight)
            or not 0.0 <= maximum_memory_weight <= 1.0
        ):
            raise ValueError("runtime option JIT memory settings are invalid")
        self.population = population
        self.continuation = continuation
        self.minimum_task_support_per_action = minimum_task_support_per_action
        self.prior_strength = float(prior_strength)
        self.maximum_memory_weight = float(maximum_memory_weight)
        self._experiences: list[RuntimeOptionJITExperience] = []
        self._frozen = False

    @property
    def experiences(self) -> tuple[RuntimeOptionJITExperience, ...]:
        return tuple(self._experiences)

    def record(self, experience: RuntimeOptionJITExperience) -> None:
        """Admit only replay-valid official rewards under the same continuation."""

        if self._frozen:
            raise RuntimeError("runtime option JIT memory is frozen")
        self.population.identity(experience.task_id, split=self.population.source_split)
        if experience.continuation != self.continuation:
            raise ValueError("runtime option JIT continuation policy differs")
        replayed = load_runtime_option_source_experience(
            official_evidence_path=Path(experience.official_evidence_path),
            population=self.population,
            episode_number=experience.episode_number,
        )
        if replayed != experience:
            raise ValueError("runtime option JIT experience differs from source replay")
        if experience in self._experiences:
            raise ValueError("runtime option JIT experience is duplicated")
        self._experiences.append(experience)

    def freeze(self) -> None:
        if not self._experiences:
            raise ValueError("runtime option JIT memory cannot freeze empty")
        episodes = {item.episode_number for item in self._experiences}
        if episodes != set(range(1, len(episodes) + 1)):
            raise ValueError("runtime option JIT episode numbers must be contiguous")
        if any(
            len([item for item in self._experiences if item.episode_number == episode])
            != 1
            for episode in episodes
        ):
            raise ValueError("one runtime option JIT episode must contain one forced action")
        self._frozen = True

    def save(self, path: Path) -> None:
        if not self._frozen:
            raise RuntimeError("runtime option JIT memory must be frozen before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "implementation_status": "implementation_unverified",
                    "population": self.population.to_dict(),
                    "continuation": self.continuation.to_dict(),
                    "minimum_task_support_per_action": self.minimum_task_support_per_action,
                    "prior_strength": self.prior_strength,
                    "maximum_memory_weight": self.maximum_memory_weight,
                    "experiences": [item.to_dict() for item in self._experiences],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> FrozenRuntimeOptionJITMemory:
        payload = json.loads(path.read_text())
        expected = {
            "schema_version",
            "implementation_status",
            "population",
            "continuation",
            "minimum_task_support_per_action",
            "prior_strength",
            "maximum_memory_weight",
            "experiences",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("runtime option JIT memory artifact fields changed")
        population = payload.get("population")
        continuation = payload.get("continuation")
        experiences = payload.get("experiences")
        if (
            payload.get("schema_version") != cls.schema_version
            or payload.get("implementation_status") != "implementation_unverified"
            or not isinstance(population, Mapping)
            or not isinstance(continuation, Mapping)
            or not isinstance(experiences, list)
        ):
            raise ValueError("runtime option JIT memory artifact is invalid")
        memory = cls(
            population=RuntimeJITPopulationManifest.from_dict(population),
            continuation=RuntimeOptionJITContinuationContract.from_dict(continuation),
            minimum_task_support_per_action=int(payload["minimum_task_support_per_action"]),
            prior_strength=float(payload["prior_strength"]),
            maximum_memory_weight=float(payload["maximum_memory_weight"]),
        )
        for row in experiences:
            if not isinstance(row, Mapping):
                raise ValueError("runtime option JIT experience row is invalid")
            memory.record(RuntimeOptionJITExperience.from_dict(row))
        memory.freeze()
        return memory

    def corrections(
        self,
        *,
        state: RuntimeOptionJITStateKey,
        test_task_id: str,
        legal_action_types: tuple[RuntimeJointActionType, ...],
        continuation: RuntimeOptionJITContinuationContract,
    ) -> tuple[bool, str | None, tuple[RuntimeOptionJITCorrection, ...]]:
        """Return raw-scale corrections or an exact no-evidence fallback."""

        if not self._frozen:
            raise RuntimeError("runtime option JIT memory must be frozen")
        self.population.identity(test_task_id, split=self.population.evaluation_split)
        if continuation != self.continuation:
            return False, "continuation_policy_mismatch", ()
        if len(legal_action_types) != len(set(legal_action_types)):
            return False, "multiple_live_instances_share_one_action_type", ()
        matches = tuple(item for item in self._experiences if item.state == state)
        values = {
            action_type: [
                item.official_reward for item in matches if item.action_type is action_type
            ]
            for action_type in legal_action_types
        }
        tasks = {
            action_type: {
                item.task_id for item in matches if item.action_type is action_type
            }
            for action_type in legal_action_types
        }
        if any(
            len(tasks[action_type]) < self.minimum_task_support_per_action
            for action_type in legal_action_types
        ):
            return False, "insufficient_task_disjoint_support_for_live_slate", ()
        means = {
            action_type: sum(action_values) / len(action_values)
            for action_type, action_values in values.items()
        }
        equal_action_baseline = sum(means.values()) / len(means)
        corrections = []
        for action_type in legal_action_types:
            action_values = values[action_type]
            count = len(action_values)
            mean = means[action_type]
            reliability = count / (count + self.prior_strength)
            variance = sum((value - mean) ** 2 for value in action_values) / count
            raw_advantage = mean - equal_action_baseline
            corrections.append(
                RuntimeOptionJITCorrection(
                    action_type=action_type,
                    support_count=count,
                    task_support_count=len(tasks[action_type]),
                    average_official_reward=mean,
                    posterior_mean=(
                        sum(action_values) + self.prior_strength * equal_action_baseline
                    )
                    / (count + self.prior_strength),
                    empirical_variance=variance,
                    raw_advantage=raw_advantage,
                    reliability=reliability,
                    applied_correction=(
                        self.maximum_memory_weight * reliability * raw_advantage
                    ),
                )
            )
        return True, None, tuple(corrections)


def _policy_value(payload: Mapping[str, object]) -> RuntimeTimeAwarePolicyValue:
    """Rebuild a typed Bellman value from the independently logged payload."""

    if set(payload) != _VALUE_FIELDS:
        raise ValueError("runtime option JIT Bellman value fields changed")
    values = {key: float(payload[key]) for key in _VALUE_FIELDS}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("runtime option JIT Bellman value is non-finite")
    return RuntimeTimeAwarePolicyValue(
        public_success_probability=values["residual_completion_proxy_probability"],
        expected_unblocked_fraction=values["expected_unblocked_fraction"],
        expected_duration_ms=values["expected_duration_ms"],
        expected_false_interventions=values["expected_false_interventions"],
        expected_intervention_cost=values["expected_intervention_cost"],
    )


def _value_payload(value: RuntimeTimeAwarePolicyValue) -> dict[str, float]:
    return {
        "residual_completion_proxy_probability": value.public_success_probability,
        "expected_unblocked_fraction": value.expected_unblocked_fraction,
        "expected_duration_ms": value.expected_duration_ms,
        "expected_false_interventions": value.expected_false_interventions,
        "expected_intervention_cost": value.expected_intervention_cost,
    }


class RuntimeOwnedOptionJITLiveResolver:
    """Correct the live DP completion coordinate without changing its slate."""

    receipt_schema_version = "runtime_owned_option_jit_decision_v1"

    def __init__(
        self,
        *,
        base_resolver: RuntimeOwnedOptionLiveResolver,
        memory: FrozenRuntimeOptionJITMemory,
        test_task_id: str,
        continuation: RuntimeOptionJITContinuationContract,
    ) -> None:
        self.base_resolver = base_resolver
        self.memory = memory
        self.test_task_id = test_task_id
        self.continuation = continuation

    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        base = self.base_resolver.resolve(slate)
        return self._apply(slate=slate, base=base, recorded_receipt=None)

    def resolve_recorded(
        self, slate: RuntimeJointActionSlate, model: Mapping[str, object]
    ) -> RuntimeJointResolvedChoice:
        recorded = model.get("option_jit_receipt")
        if not isinstance(recorded, Mapping):
            raise ValueError("recorded runtime option JIT receipt is missing")
        base = self.base_resolver.resolve_recorded(slate, model)
        return self._apply(slate=slate, base=base, recorded_receipt=recorded)

    def _apply(
        self,
        *,
        slate: RuntimeJointActionSlate,
        base: RuntimeJointResolvedChoice,
        recorded_receipt: Mapping[str, object] | None,
    ) -> RuntimeJointResolvedChoice:
        model = base.planning_model
        if not isinstance(model, Mapping) or not isinstance(base.bellman_values, Mapping):
            raise ValueError("runtime option JIT requires a real Bellman decision")
        projection_payload = model.get("projection")
        if not isinstance(projection_payload, Mapping):
            raise ValueError("runtime option JIT projection is missing")
        projection = RuntimeOwnedOptionProjection.from_dict(projection_payload)
        state = option_jit_state_key(slate=slate, projection=projection)
        action_types = tuple(item.action_type for item in slate.candidates)
        admitted, fallback_reason, corrections = self.memory.corrections(
            state=state,
            test_task_id=self.test_task_id,
            legal_action_types=action_types,
            continuation=self.continuation,
        )
        corrections_by_type = {item.action_type: item for item in corrections}
        candidates = {item.instance_action_id: item for item in slate.candidates}
        adjusted_values = {}
        for action_id, payload in base.bellman_values.items():
            if action_id not in candidates or not isinstance(payload, Mapping):
                raise ValueError("runtime option JIT Bellman action differs from slate")
            value = _policy_value(payload)
            correction = corrections_by_type.get(candidates[action_id].action_type)
            success = value.public_success_probability
            if admitted and correction is not None:
                success = min(1.0, max(0.0, success + correction.applied_correction))
            adjusted_values[action_id] = replace(value, public_success_probability=success)
        selected_id = max(
            adjusted_values,
            key=lambda action_id: adjusted_values[action_id].priority,
        )
        receipt = {
            "schema_version": self.receipt_schema_version,
            "implementation_status": "implementation_unverified",
            "test_task_id": self.test_task_id,
            "state": state.to_dict(),
            "continuation": self.continuation.to_dict(),
            "admitted": admitted,
            "fallback_reason": fallback_reason,
            "base_selected_instance_action_id": base.selected.instance_action_id,
            "selected_instance_action_id": selected_id,
            "corrections": [item.to_dict() for item in corrections],
            "source_evidence_paths": sorted(
                {
                    item.official_evidence_path
                    for item in self.memory.experiences
                    if item.state == state and item.action_type in action_types
                }
            ),
        }
        if recorded_receipt is not None and dict(recorded_receipt) != receipt:
            raise ValueError("recorded runtime option JIT correction differs")
        return RuntimeJointResolvedChoice(
            selected=candidates[selected_id],
            rule_id=(
                "runtime_owned_option_dp_jit_bounded_correction_v1"
                if admitted
                else "runtime_owned_option_dp_jit_exact_fallback_v1"
            ),
            bellman_values={
                action_id: _value_payload(value)
                for action_id, value in adjusted_values.items()
            },
            jit_advantages=receipt,
            planning_model={**dict(model), "option_jit_receipt": receipt},
        )


class JITV2EstimateAdapter:
    """Adapt one action's outcome estimate from the JIT v2 verified-transition memory.

    the task objective is untouched; the adapter only replaces the transition
    distribution the Bellman planner plans with, for actions whose ``(state, action)`` key has
    enough independent task-disjoint support.  Every call returns an auditable record (state
    and action key, support counts, weight, before/after distributions, evidence paths) which
    the live resolver persists in its estimate receipt.
    """

    def __init__(
        self,
        *,
        memory: FrozenRuntimeOptionJITMemoryV2,
        test_task_id: str,
        configured_checks: tuple,
        repair_types: Mapping[str, str],
    ) -> None:
        self.memory = memory
        self.test_task_id = test_task_id
        self._kind_by_check_id = {
            str(check.check_id): str(check.predicate_kind)
            for check in configured_checks
            if check.check_id is not None
        }
        self._kind_by_predicate_id = {
            str(check.predicate_id): str(check.predicate_kind) for check in configured_checks
        }
        self.repair_types = dict(repair_types)

    def state_view(
        self, slate: RuntimeJointActionSlate, projection: RuntimeOwnedOptionProjection
    ) -> RecoveryStateView:
        verdicts: list[tuple[str, str]] = []
        incident_kinds: list[str] = []
        active = 0
        for predicate in projection.acceptance_predicates:
            kind = self._kind_by_predicate_id.get(str(predicate.predicate_id))
            if kind is None:
                continue
            verdicts.append((kind, str(predicate.evidence_state)))
            if predicate.incident_id is not None and str(
                predicate.incident_status
            ) not in {"resolved", "closed"}:
                active += 1
                incident_kinds.append(kind)
        families = {
            action.action_type.value
            for action in (*slate.candidates, *slate.conditional_future_options)
        }
        return RecoveryStateView(
            legal_action_families=tuple(sorted(families)),
            check_verdicts=tuple(verdicts),
            active_incident_count=active,
            repairable_incident=any(
                action.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR
                for action in slate.candidates
            ),
            candidate_kinds=tuple(incident_kinds),
            # The effect ledger is not projected into the resolver yet; the frozen field stays
            # at its neutral value on both the live and the offline key paths.
            recent_intervention_status="none",
            remaining_budget_units=int(slate.budget_before),
        )

    def action_key(self, action, rule) -> RecoveryActionKeyV2:
        family = action.action_type.value
        if family == "run_registered_check":
            return RecoveryActionKeyV2.check(
                self._kind_by_check_id.get(str(rule.contract_id), "shape")
            )
        if family == "run_targeted_repair":
            repair_id = str(rule.contract_id).split("|", 1)[0]
            return RecoveryActionKeyV2.repair(
                self.repair_types.get(repair_id, "actor_guidance")
            )
        return RecoveryActionKeyV2.generic(family)

    def __call__(self, slate, projection, action, rule, estimate):
        state_key = recovery_state_key_v2_from_view(self.state_view(slate, projection))
        action_key = self.action_key(action, rule)
        adaptation = self.memory.adapt(
            test_task_id=self.test_task_id,
            state_key=state_key,
            action_key=action_key,
            before_outcomes=tuple(
                (item.outcome_id, item.probability) for item in estimate.outcomes
            ),
        )
        if adaptation.admitted:
            estimate = replace(
                estimate,
                outcomes=tuple(
                    RuntimeAtomicOutcomeEstimate(outcome_id=name, probability=value)
                    for name, value in adaptation.after_outcomes
                ),
            )
        record = {
            "instance_action_id": estimate.instance_action_id,
            "action_key": action_key.to_dict(),
            "state_key": state_key.to_dict(),
            "admitted": adaptation.admitted,
            "fallback_reason": adaptation.fallback_reason,
            "support_task_count": adaptation.support_task_count,
            "support_episode_count": adaptation.support_episode_count,
            "memory_weight": adaptation.memory_weight,
            "before_outcomes": [list(item) for item in adaptation.before_outcomes],
            "after_outcomes": [list(item) for item in adaptation.after_outcomes],
            "regression_rate": adaptation.regression_rate,
            "evidence_paths": list(adaptation.evidence_paths),
        }
        return estimate, record


class RuntimeOwnedOptionJITV2LiveResolver:
    """Run the DP twice with the JIT v2 estimate adaptation and record both passes.

    The base pass proves the unadapted Bellman values; the adapted pass is the one whose
    choice is returned, so the receipt can show the transition estimate change and the
    resulting value change without giving the JIT any authority beyond the estimate model.
    """

    receipt_schema_version = "runtime_owned_option_jit_v2_decision_v1"

    def __init__(
        self,
        *,
        base_resolver: RuntimeOwnedOptionLiveResolver,
        memory: FrozenRuntimeOptionJITMemoryV2,
        test_task_id: str,
        method_id: str,
    ) -> None:
        self.base_resolver = base_resolver
        self.memory = memory
        self.test_task_id = test_task_id
        self.method_id = method_id

    def _receipt(
        self,
        *,
        adapted: RuntimeJointResolvedChoice,
        base_values: Mapping[str, object] | None,
        base_selected: object,
    ) -> dict[str, object]:
        atomic = dict((adapted.planning_model or {}).get("atomic_estimate_receipt") or {})
        adaptations = list(atomic.get("transition_adaptations") or [])
        return {
            "schema_version": self.receipt_schema_version,
            "implementation_status": "implementation_unverified",
            "method_id": self.method_id,
            "test_task_id": self.test_task_id,
            "memory_schema_version": JIT_V2_SCHEMA_VERSION,
            "population_id": self.memory.population.population_id,
            "transition_count": len(self.memory.transitions),
            "adaptations": adaptations,
            "admitted_action_count": sum(
                1 for row in adaptations if row.get("admitted")
            ),
            "base_selected_instance_action_id": base_selected,
            "selected_instance_action_id": adapted.selected.instance_action_id,
            "base_bellman_values": dict(base_values) if base_values else {},
            "adapted_bellman_values": dict(adapted.bellman_values),
        }

    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        base = self.base_resolver.resolve(slate, estimate_adapter=None)
        adapted = self.base_resolver.resolve(slate)
        receipt = self._receipt(
            adapted=adapted,
            base_values=base.bellman_values,
            base_selected=base.selected.instance_action_id,
        )
        return replace(
            adapted,
            jit_advantages=receipt,
            planning_model={
                **dict(adapted.planning_model or {}),
                "option_jit_v2_receipt": receipt,
            },
        )

    def resolve_recorded(
        self,
        slate: RuntimeJointActionSlate,
        model: Mapping[str, object],
    ) -> RuntimeJointResolvedChoice:
        recorded = model.get("option_jit_v2_receipt")
        if not isinstance(recorded, Mapping):
            raise ValueError("recorded JIT v2 receipt is missing")
        adapted = self.base_resolver.resolve_recorded(slate, model)
        receipt = self._receipt(
            adapted=adapted,
            base_values=recorded.get("base_bellman_values"),
            base_selected=recorded.get("base_selected_instance_action_id"),
        )
        if dict(recorded) != receipt:
            raise ValueError("recorded JIT v2 correction differs")
        return replace(
            adapted,
            jit_advantages=receipt,
            planning_model={
                **dict(adapted.planning_model or {}),
                "option_jit_v2_receipt": receipt,
            },
        )


__all__ = [
    "FrozenRuntimeOptionJITMemory",
    "JITV2EstimateAdapter",
    "RuntimeOptionJITContinuationContract",
    "RuntimeOptionJITCorrection",
    "RuntimeOptionJITExperience",
    "RuntimeOptionJITStateKey",
    "RuntimeOwnedOptionJITLiveResolver",
    "RuntimeOwnedOptionJITV2LiveResolver",
    "load_runtime_option_source_experience",
    "option_jit_state_key",
]
