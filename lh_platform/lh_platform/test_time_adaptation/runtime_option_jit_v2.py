"""JIT v2: task-disjoint verified-transition memory for the Runtime-owned option DP.

The legacy v1 artifact
(`frozen_runtime_option_jit_memory_v1`, official-reward only, forced-first sources, exact
task keys) stays immutable.  This module is the replacement formulation for the recovery
composition line:

* the stored unit is a **verified transition** ``(z, u, y, z', cost)`` — a check verdict, a
  repair trial outcome, or a continuation progress outcome observed by the runtime — not a
  terminal reward credited backwards;
* the task objective is unchanged: the Bellman planner still defines action value from the
  task, the JIT only adapts the transition distribution it plans with;
* retrieval keys are cross-task by construction: `RecoveryStateKeyV2` is built from a small
  public state view and validated against frozen vocabularies (no task ids, no paths, no
  contract ids, no exact check/predicate counts), and `RecoveryActionKeyV2` is
  ``(action family, subtype)`` where the subtype is the predicate kind for checks and the
  repair type for repairs;
* support is measured in **independent source tasks** and the frozen gate is 2; an action
  without enough task-disjoint support abstains and the base estimate is used unchanged.

Everything here is offline arithmetic: construction, freeze, save/load, and the bounded
adaptation of one action's outcome distribution.  The runtime seam that calls `adapt` and
records the before/after evidence lives in the resolver (see `runtime_owned_option_jit_v2`
integration in the live resolver module).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lh_platform.test_time_adaptation.runtime_jit_action_advantage import (
    RuntimeJITPopulationManifest,
)
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionType,
)

SCHEMA_VERSION = "frozen_runtime_option_jit_memory_v2"

# The recovery-composition action families; every one is a real joint-slate action type.
SUPPORTED_ACTION_FAMILIES_V2 = tuple(
    action_type.value
    for action_type in (
        RuntimeJointActionType.CONTINUE_ACTOR,
        RuntimeJointActionType.RUN_REGISTERED_CHECK,
        RuntimeJointActionType.RUN_TARGETED_REPAIR,
        RuntimeJointActionType.SUBMIT_CURRENT_STATE,
    )
)
IMPLEMENTATION_STATUS = "implementation_unverified"
SUPPORTED_ENVIRONMENT_FAMILIES = ("terminal",)
RECOVERY_PHASES = (
    "unverified",
    "active_refutation",
    "post_repair",
    "apparently_resolved",
)
EVIDENCE_PROFILES = ("has_refuted", "has_unknown", "all_supported")
INCIDENT_BUCKETS = ("0", "1", "2+")
BUDGET_BUCKETS = ("low", "medium", "high")
INTERVENTION_STATUSES = ("none", "failed", "locally_resolved", "regressed")
CANDIDATE_SUBTYPE_GENERIC = "generic"
CANDIDATE_SUBTYPE_MIXED = "mixed"

CHECK_OUTCOMES = ("supported", "refuted", "unknown")
REPAIR_OUTCOMES = ("supported", "refuted", "unknown")
CONTINUE_OUTCOMES = ("progress", "no_progress", "self_repair", "new_incident")

# A full incident cycle is one check plus one repair trial plus one recheck.
REPAIR_CYCLE_COST_UNITS = 4
BUDGET_BUCKET_BOUNDS = ((4, "low"), (8, "medium"))  # budget < 4 -> low; < 8 -> medium; else high


def budget_bucket(budget_units: int) -> str:
    """Bucket remaining budget against one full probe -> repair -> recheck cycle."""

    if isinstance(budget_units, bool) or not isinstance(budget_units, int) or budget_units < 0:
        raise ValueError("JIT v2 budget units must be a non-negative integer")
    for bound, label in BUDGET_BUCKET_BOUNDS:
        if budget_units < bound:
            return label
    return "high"


def _validated(value: object, vocabulary: Sequence[str], label: str) -> str:
    if not isinstance(value, str) or value not in vocabulary:
        raise ValueError(f"JIT v2 {label} is outside its frozen vocabulary: {value!r}")
    return value


@dataclass(frozen=True)
class RecoveryActionKeyV2:
    """Cross-task action identity: stable family plus kind-level subtype."""

    family: str
    subtype: str

    def __post_init__(self) -> None:
        _validated(self.family, SUPPORTED_ACTION_FAMILIES_V2, "action family")
        if not isinstance(self.subtype, str) or not self.subtype:
            raise ValueError("JIT v2 action subtype is required")
        if self.family in {"continue_actor", "submit_current_state"}:
            if self.subtype != CANDIDATE_SUBTYPE_GENERIC:
                raise ValueError("JIT v2 generic actions require the generic subtype")
        if "/" in self.subtype or "_id" in self.subtype:
            raise ValueError("JIT v2 action subtype cannot carry task identities")

    @classmethod
    def check(cls, predicate_kind: str) -> RecoveryActionKeyV2:
        return cls(family="run_registered_check", subtype=predicate_kind)

    @classmethod
    def repair(cls, repair_type: str) -> RecoveryActionKeyV2:
        return cls(family="run_targeted_repair", subtype=repair_type)

    @classmethod
    def generic(cls, family: str) -> RecoveryActionKeyV2:
        return cls(family=family, subtype=CANDIDATE_SUBTYPE_GENERIC)

    def to_dict(self) -> dict[str, str]:
        return {"family": self.family, "subtype": self.subtype}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RecoveryActionKeyV2:
        if set(payload) != {"family", "subtype"}:
            raise ValueError("JIT v2 action key fields changed")
        return cls(family=str(payload["family"]), subtype=str(payload["subtype"]))


@dataclass(frozen=True)
class RecoveryStateView:
    """The small public snapshot a state key is computed from.

    Both the live resolver and the offline extractor build this view; kind-level verdict
    aggregation is deliberate, because per-predicate identity would leak task structure into
    the retrieval key.
    """

    legal_action_families: tuple[str, ...]
    check_verdicts: tuple[tuple[str, str], ...]  # (predicate_kind, verdict)
    active_incident_count: int
    repairable_incident: bool
    candidate_kinds: tuple[str, ...]
    recent_intervention_status: str
    remaining_budget_units: int

    def __post_init__(self) -> None:
        families = tuple(sorted(set(self.legal_action_families)))
        if not families or any(
            family not in SUPPORTED_ACTION_FAMILIES_V2 for family in families
        ):
            raise ValueError("JIT v2 state view action families are invalid")
        object.__setattr__(self, "legal_action_families", families)
        verdicts = tuple(sorted(self.check_verdicts))
        if any(verdict not in CHECK_OUTCOMES + ("stale",) for _, verdict in verdicts):
            raise ValueError("JIT v2 state view verdict is invalid")
        object.__setattr__(self, "check_verdicts", verdicts)
        if (
            isinstance(self.active_incident_count, bool)
            or not isinstance(self.active_incident_count, int)
            or self.active_incident_count < 0
        ):
            raise ValueError("JIT v2 state view incident count is invalid")
        kinds = tuple(sorted(set(self.candidate_kinds)))
        object.__setattr__(self, "candidate_kinds", kinds)
        _validated(self.recent_intervention_status, INTERVENTION_STATUSES, "intervention status")


def recovery_state_key_v2_from_view(view: RecoveryStateView) -> RecoveryStateKeyV2:
    """Project a public state view into the frozen cross-task retrieval key."""

    verdict_values = [verdict for _, verdict in view.check_verdicts]
    if "refuted" in verdict_values:
        evidence_profile = "has_refuted"
    elif not verdict_values or any(
        value in {"unknown", "stale"} for value in verdict_values
    ):
        # No verdict at all is not "all supported": nothing has been verified yet.
        evidence_profile = "has_unknown"
    else:
        evidence_profile = "all_supported"
    if view.active_incident_count > 0:
        phase = "active_refutation"
    elif view.recent_intervention_status in {"failed", "locally_resolved", "regressed"}:
        phase = "post_repair"
    elif evidence_profile == "all_supported":
        phase = "apparently_resolved"
    else:
        phase = "unverified"
    if view.active_incident_count == 0:
        incident_bucket = "0"
    elif view.active_incident_count == 1:
        incident_bucket = "1"
    else:
        incident_bucket = "2+"
    if not view.candidate_kinds:
        subtype = CANDIDATE_SUBTYPE_GENERIC
    elif len(view.candidate_kinds) == 1:
        subtype = view.candidate_kinds[0]
    else:
        subtype = CANDIDATE_SUBTYPE_MIXED
    return RecoveryStateKeyV2(
        environment_family="terminal",
        recovery_phase=phase,
        legal_action_families=view.legal_action_families,
        evidence_profile=evidence_profile,
        active_incident_count=incident_bucket,
        repairable_incident=bool(view.repairable_incident),
        candidate_subtype=subtype,
        remaining_budget_bucket=budget_bucket(view.remaining_budget_units),
        recent_intervention_status=view.recent_intervention_status,
    )


@dataclass(frozen=True)
class RecoveryStateKeyV2:
    """Frozen task-agnostic retrieval key."""

    environment_family: str
    recovery_phase: str
    legal_action_families: tuple[str, ...]
    evidence_profile: str
    active_incident_count: str
    repairable_incident: bool
    candidate_subtype: str
    remaining_budget_bucket: str
    recent_intervention_status: str

    def __post_init__(self) -> None:
        _validated(
            self.environment_family,
            SUPPORTED_ENVIRONMENT_FAMILIES,
            "environment family",
        )
        _validated(self.recovery_phase, RECOVERY_PHASES, "recovery phase")
        _validated(self.evidence_profile, EVIDENCE_PROFILES, "evidence profile")
        _validated(self.active_incident_count, INCIDENT_BUCKETS, "incident bucket")
        _validated(self.remaining_budget_bucket, BUDGET_BUCKETS, "budget bucket")
        _validated(
            self.recent_intervention_status,
            INTERVENTION_STATUSES,
            "intervention status",
        )
        families = tuple(sorted(set(self.legal_action_families)))
        if not families or any(
            family not in SUPPORTED_ACTION_FAMILIES_V2 for family in families
        ):
            raise ValueError("JIT v2 state key action families are invalid")
        object.__setattr__(self, "legal_action_families", families)
        if not isinstance(self.repairable_incident, bool):
            raise ValueError("JIT v2 state key repairable flag must be boolean")
        if not isinstance(self.candidate_subtype, str) or not self.candidate_subtype:
            raise ValueError("JIT v2 state key candidate subtype is required")
        # Frozen content guard: the key must never carry task structure.
        forbidden = ("/", "task-", "public-task", "terminal-bench")
        for field in (
            self.candidate_subtype,
            *self.legal_action_families,
        ):
            if any(token in field for token in forbidden):
                raise ValueError("JIT v2 state key carries task-specific content")

    def to_dict(self) -> dict[str, object]:
        return {
            "environment_family": self.environment_family,
            "recovery_phase": self.recovery_phase,
            "legal_action_families": list(self.legal_action_families),
            "evidence_profile": self.evidence_profile,
            "active_incident_count": self.active_incident_count,
            "repairable_incident": self.repairable_incident,
            "candidate_subtype": self.candidate_subtype,
            "remaining_budget_bucket": self.remaining_budget_bucket,
            "recent_intervention_status": self.recent_intervention_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RecoveryStateKeyV2:
        expected = {
            "environment_family",
            "recovery_phase",
            "legal_action_families",
            "evidence_profile",
            "active_incident_count",
            "repairable_incident",
            "candidate_subtype",
            "remaining_budget_bucket",
            "recent_intervention_status",
        }
        families = payload.get("legal_action_families")
        if set(payload) != expected or not isinstance(families, list):
            raise ValueError("JIT v2 state key payload fields changed")
        if type(payload.get("repairable_incident")) is not bool:
            raise ValueError("JIT v2 state key payload flag is invalid")
        return cls(
            environment_family=str(payload["environment_family"]),
            recovery_phase=str(payload["recovery_phase"]),
            legal_action_families=tuple(str(item) for item in families),
            evidence_profile=str(payload["evidence_profile"]),
            active_incident_count=str(payload["active_incident_count"]),
            repairable_incident=payload["repairable_incident"],
            candidate_subtype=str(payload["candidate_subtype"]),
            remaining_budget_bucket=str(payload["remaining_budget_bucket"]),
            recent_intervention_status=str(payload["recent_intervention_status"]),
        )


@dataclass(frozen=True)
class VerifiedTransitionV2:
    """One verified ``(z, u, y, z', cost)`` observation from a source episode."""

    task_id: str
    episode_number: int
    state_key: RecoveryStateKeyV2
    action_key: RecoveryActionKeyV2
    outcome_key: str
    next_state_key: RecoveryStateKeyV2 | None
    cost_units: int
    duration_ms: int
    regression: bool
    evidence_path: str

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or isinstance(self.episode_number, bool)
            or not isinstance(self.episode_number, int)
            or self.episode_number < 1
            or not self.outcome_key
            or isinstance(self.cost_units, bool)
            or not isinstance(self.cost_units, int)
            or self.cost_units < 0
            or isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
            or type(self.regression) is not bool
            or not self.evidence_path
        ):
            raise ValueError("JIT v2 transition identity or measurements are invalid")
        allowed = self.outcome_vocabulary()
        if self.outcome_key not in allowed:
            raise ValueError("JIT v2 transition outcome is outside its action vocabulary")

    def outcome_vocabulary(self) -> tuple[str, ...]:
        if self.action_key.family == "run_registered_check":
            return CHECK_OUTCOMES
        if self.action_key.family == "run_targeted_repair":
            return REPAIR_OUTCOMES
        if self.action_key.family == "continue_actor":
            return CONTINUE_OUTCOMES
        return ("submitted",)

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "episode_number": self.episode_number,
            "state_key": self.state_key.to_dict(),
            "action_key": self.action_key.to_dict(),
            "outcome_key": self.outcome_key,
            "next_state_key": (
                self.next_state_key.to_dict() if self.next_state_key is not None else None
            ),
            "cost_units": self.cost_units,
            "duration_ms": self.duration_ms,
            "regression": self.regression,
            "evidence_path": self.evidence_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> VerifiedTransitionV2:
        expected = {
            "task_id",
            "episode_number",
            "state_key",
            "action_key",
            "outcome_key",
            "next_state_key",
            "cost_units",
            "duration_ms",
            "regression",
            "evidence_path",
        }
        state = payload.get("state_key")
        action = payload.get("action_key")
        next_state = payload.get("next_state_key")
        if (
            set(payload) != expected
            or not isinstance(state, Mapping)
            or not isinstance(action, Mapping)
            or (next_state is not None and not isinstance(next_state, Mapping))
        ):
            raise ValueError("JIT v2 transition payload fields changed")
        return cls(
            task_id=str(payload["task_id"]),
            episode_number=int(payload["episode_number"]),
            state_key=RecoveryStateKeyV2.from_dict(state),
            action_key=RecoveryActionKeyV2.from_dict(action),
            outcome_key=str(payload["outcome_key"]),
            next_state_key=(
                RecoveryStateKeyV2.from_dict(next_state)
                if isinstance(next_state, Mapping)
                else None
            ),
            cost_units=int(payload["cost_units"]),
            duration_ms=int(payload["duration_ms"]),
            regression=bool(payload["regression"]),
            evidence_path=str(payload["evidence_path"]),
        )


@dataclass(frozen=True)
class JITTransitionAdaptation:
    """One bounded adaptation of an action's outcome distribution at a decision point."""

    state_key: RecoveryStateKeyV2
    action_key: RecoveryActionKeyV2
    admitted: bool
    fallback_reason: str | None
    support_task_count: int
    support_episode_count: int
    memory_weight: float
    before_outcomes: tuple[tuple[str, float], ...]
    after_outcomes: tuple[tuple[str, float], ...]
    regression_rate: float | None
    evidence_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "state_key": self.state_key.to_dict(),
            "action_key": self.action_key.to_dict(),
            "admitted": self.admitted,
            "fallback_reason": self.fallback_reason,
            "support_task_count": self.support_task_count,
            "support_episode_count": self.support_episode_count,
            "memory_weight": self.memory_weight,
            "before_outcomes": [list(item) for item in self.before_outcomes],
            "after_outcomes": [list(item) for item in self.after_outcomes],
            "regression_rate": self.regression_rate,
            "evidence_paths": list(self.evidence_paths),
        }


class FrozenRuntimeOptionJITMemoryV2:
    """Freeze verified transitions, then adapt one action's estimates at a time."""

    def __init__(
        self,
        *,
        population: RuntimeJITPopulationManifest,
        minimum_task_support_per_action: int = 2,
        prior_strength: float = 4.0,
        maximum_memory_weight: float = 0.5,
    ) -> None:
        if (
            isinstance(minimum_task_support_per_action, bool)
            or not isinstance(minimum_task_support_per_action, int)
            or minimum_task_support_per_action < 1
            or not math.isfinite(prior_strength)
            or prior_strength <= 0.0
            or not math.isfinite(maximum_memory_weight)
            or not 0.0 <= maximum_memory_weight <= 1.0
        ):
            raise ValueError("JIT v2 memory settings are invalid")
        self.population = population
        self.minimum_task_support_per_action = minimum_task_support_per_action
        self.prior_strength = float(prior_strength)
        self.maximum_memory_weight = float(maximum_memory_weight)
        self._transitions: list[VerifiedTransitionV2] = []
        self._frozen = False

    @property
    def transitions(self) -> tuple[VerifiedTransitionV2, ...]:
        return tuple(self._transitions)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def record(self, transition: VerifiedTransitionV2) -> None:
        """Admit one verified observation; repetition inside an episode is data.

        Unlike the forced-action v1 memory, one natural episode legitimately yields several
        transitions with the same key and outcome (for example three shape checks that refute
        at the same entry), and each is an independent observation for the outcome counts.
        """

        if self._frozen:
            raise RuntimeError("JIT v2 memory is frozen")
        self.population.identity(transition.task_id, split=self.population.source_split)
        self._transitions.append(transition)

    def freeze(self) -> None:
        if not self._transitions:
            raise ValueError("JIT v2 memory cannot freeze without transitions")
        episodes = {item.episode_number for item in self._transitions}
        if episodes != set(range(1, len(episodes) + 1)):
            raise ValueError("JIT v2 episode numbers must be contiguous from one")
        self._frozen = True

    def eligible(
        self,
        *,
        test_task_id: str,
        state_key: RecoveryStateKeyV2,
        action_key: RecoveryActionKeyV2,
    ) -> tuple[VerifiedTransitionV2, ...]:
        """Task-disjoint exact-key retrieval: the test task never retrieves itself."""

        return tuple(
            item
            for item in self._transitions
            if item.task_id != test_task_id
            and item.state_key == state_key
            and item.action_key == action_key
        )

    def adapt(
        self,
        *,
        test_task_id: str,
        state_key: RecoveryStateKeyV2,
        action_key: RecoveryActionKeyV2,
        before_outcomes: Sequence[tuple[str, float]],
    ) -> JITTransitionAdaptation:
        """Blend the memory outcome distribution into the base estimates of one action.

        The weight grows with the number of eligible episodes and is capped by
        `maximum_memory_weight`, so a two-task source population can move an estimate but can
        never overwrite the calibrated base.  Without the frozen independent-task support the
        base estimate is returned unchanged with an explicit fallback reason.
        """

        before = tuple((str(name), float(value)) for name, value in before_outcomes)
        if not before or abs(math.fsum(value for _, value in before) - 1.0) > 1e-9:
            raise ValueError("JIT v2 base outcome distribution must sum to one")
        eligible = self.eligible(
            test_task_id=test_task_id, state_key=state_key, action_key=action_key
        )
        task_count = len({item.task_id for item in eligible})
        if task_count < self.minimum_task_support_per_action:
            return JITTransitionAdaptation(
                state_key=state_key,
                action_key=action_key,
                admitted=False,
                fallback_reason="insufficient_task_disjoint_support",
                support_task_count=task_count,
                support_episode_count=len(eligible),
                memory_weight=0.0,
                before_outcomes=before,
                after_outcomes=before,
                regression_rate=None,
                evidence_paths=tuple(sorted({item.evidence_path for item in eligible})),
            )
        vocabulary = (
            eligible[0].outcome_vocabulary()
            if eligible
            else tuple(name for name, _ in before)
        )
        counts = {name: 0 for name in vocabulary}
        for item in eligible:
            counts[item.outcome_key] = counts.get(item.outcome_key, 0) + 1
        total = len(eligible)
        memory_distribution = {
            name: (1.0 + counts.get(name, 0)) / (len(vocabulary) + total)
            for name in vocabulary
        }
        weight = min(self.maximum_memory_weight, total / (total + self.prior_strength))
        after = []
        for name, value in before:
            memory_value = memory_distribution.get(name)
            if memory_value is None:
                after.append((name, value))
            else:
                after.append((name, (1.0 - weight) * value + weight * memory_value))
        # Re-normalize only for names the base did not carry, then correct the residual on the
        # largest entry so the frozen sum-to-one invariant holds exactly.
        after = tuple(after)
        residual = 1.0 - math.fsum(value for _, value in after)
        if abs(residual) > 1e-9:
            index = max(range(len(after)), key=lambda i: after[i][1])
            name, value = after[index]
            after = after[:index] + ((name, value + residual),) + after[index + 1 :]
        regression_rate = (
            sum(1 for item in eligible if item.regression) / len(eligible)
            if eligible
            else None
        )
        return JITTransitionAdaptation(
            state_key=state_key,
            action_key=action_key,
            admitted=True,
            fallback_reason=None,
            support_task_count=task_count,
            support_episode_count=len(eligible),
            memory_weight=weight,
            before_outcomes=before,
            after_outcomes=after,
            regression_rate=regression_rate,
            evidence_paths=tuple(sorted({item.evidence_path for item in eligible})),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "implementation_status": IMPLEMENTATION_STATUS,
            "population": self.population.to_dict(),
            "minimum_task_support_per_action": self.minimum_task_support_per_action,
            "prior_strength": self.prior_strength,
            "maximum_memory_weight": self.maximum_memory_weight,
            "transitions": [item.to_dict() for item in self._transitions],
        }

    def save(self, path: Path) -> None:
        if not self._frozen:
            raise RuntimeError("JIT v2 memory must be frozen before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> FrozenRuntimeOptionJITMemoryV2:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "implementation_status",
            "population",
            "minimum_task_support_per_action",
            "prior_strength",
            "maximum_memory_weight",
            "transitions",
        }
        transitions = payload.get("transitions")
        population = payload.get("population")
        if (
            set(payload) != expected
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("implementation_status") != IMPLEMENTATION_STATUS
            or not isinstance(population, Mapping)
            or not isinstance(transitions, list)
            or any(not isinstance(item, Mapping) for item in transitions)
        ):
            raise ValueError("JIT v2 memory payload differs")
        memory = cls(
            population=RuntimeJITPopulationManifest.from_dict(population),
            minimum_task_support_per_action=int(
                payload["minimum_task_support_per_action"]
            ),
            prior_strength=float(payload["prior_strength"]),
            maximum_memory_weight=float(payload["maximum_memory_weight"]),
        )
        for item in transitions:
            memory.record(VerifiedTransitionV2.from_dict(item))
        memory.freeze()
        if memory.to_dict() != dict(payload):
            raise ValueError("JIT v2 memory changed on load")
        return memory
