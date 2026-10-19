"""Fail-closed admission for task-disjoint transition-grounded populations."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_controlled_fault_injection import (
    AlfWorldControlledFaultInjectionPlanV1,
    ControlledFaultInjectionArmV1,
)
from lh_platform.tracing.alfworld_transition_grounded_traceback import (
    build_alfworld_transition_grounded_proposal,
)
from lh_platform.tracing.trace_observability import TraceSymbolicStateEvidenceV1
from lh_platform.tracing.types import TraceCaseV1

TRANSITION_GROUNDED_POPULATION_SCHEMA = "lh_alfworld_transition_grounded_population/v1"
TRANSITION_GROUNDED_BLIND_POPULATION_SCHEMA = "lh_alfworld_transition_grounded_blind_population/v1"
TRANSITION_GROUNDED_PREFLIGHT_SCHEMA = "lh_alfworld_transition_grounded_population_preflight/v1"
TRANSITION_GROUNDED_BLIND_PREFLIGHT_SCHEMA = (
    "lh_alfworld_transition_grounded_blind_population_preflight/v1"
)
TRANSITION_GROUNDED_TWO_STRATA_POPULATION_SCHEMA = (
    "lh_alfworld_transition_grounded_population/v2"
)
TRANSITION_GROUNDED_TWO_STRATA_BLIND_POPULATION_SCHEMA = (
    "lh_alfworld_transition_grounded_blind_population/v2"
)
TRANSITION_GROUNDED_TWO_STRATA_PREFLIGHT_SCHEMA = (
    "lh_alfworld_transition_grounded_population_preflight/v2"
)
TRANSITION_GROUNDED_TWO_STRATA_BLIND_PREFLIGHT_SCHEMA = (
    "lh_alfworld_transition_grounded_blind_population_preflight/v2"
)
TRANSITION_GROUNDED_V1_PROTOCOL_ID = "task_disjoint_three_strata_v1"
TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL_ID = "task_disjoint_two_strata_v2"
FORMAL_MAX_INTERVENTION_TRIALS = 12


@dataclass(frozen=True)
class TransitionGroundedPopulationProtocolV1:
    """One frozen denominator contract for a task-disjoint cohort.

    A protocol is deliberately not inferred from observed replay outcomes.  The
    expected schemas, strata, denominator, and per-case replay budget are all
    declared before the first causal replay is read.
    """

    protocol_id: str
    population_schema: str
    blind_population_schema: str
    population_preflight_schema: str
    blind_population_preflight_schema: str
    distance_strata: tuple[str, ...]
    cases_per_stratum: int
    requires_explicit_manifest_protocol: bool
    max_intervention_trials_per_case: int = FORMAL_MAX_INTERVENTION_TRIALS

    def __post_init__(self) -> None:
        if (
            not self.protocol_id
            or not self.population_schema
            or not self.blind_population_schema
            or not self.population_preflight_schema
            or not self.blind_population_preflight_schema
            or not self.distance_strata
            or len(self.distance_strata) != len(set(self.distance_strata))
            or self.cases_per_stratum < 1
            or self.max_intervention_trials_per_case < 1
        ):
            raise ValueError("transition-grounded protocol identity is invalid")

    @property
    def case_count(self) -> int:
        return len(self.distance_strata) * self.cases_per_stratum

    @property
    def expected_stratum_counts(self) -> dict[str, int]:
        return {stratum: self.cases_per_stratum for stratum in self.distance_strata}


TRANSITION_GROUNDED_V1_PROTOCOL = TransitionGroundedPopulationProtocolV1(
    protocol_id=TRANSITION_GROUNDED_V1_PROTOCOL_ID,
    population_schema=TRANSITION_GROUNDED_POPULATION_SCHEMA,
    blind_population_schema=TRANSITION_GROUNDED_BLIND_POPULATION_SCHEMA,
    population_preflight_schema=TRANSITION_GROUNDED_PREFLIGHT_SCHEMA,
    blind_population_preflight_schema=TRANSITION_GROUNDED_BLIND_PREFLIGHT_SCHEMA,
    distance_strata=(
        "terminal_immediate",
        "short_suffix_1_2",
        "long_suffix_ge_3",
    ),
    cases_per_stratum=5,
    requires_explicit_manifest_protocol=False,
)
TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL = TransitionGroundedPopulationProtocolV1(
    protocol_id=TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL_ID,
    population_schema=TRANSITION_GROUNDED_TWO_STRATA_POPULATION_SCHEMA,
    blind_population_schema=TRANSITION_GROUNDED_TWO_STRATA_BLIND_POPULATION_SCHEMA,
    population_preflight_schema=TRANSITION_GROUNDED_TWO_STRATA_PREFLIGHT_SCHEMA,
    blind_population_preflight_schema=(
        TRANSITION_GROUNDED_TWO_STRATA_BLIND_PREFLIGHT_SCHEMA
    ),
    distance_strata=("short_suffix_1_2", "long_suffix_ge_3"),
    cases_per_stratum=5,
    requires_explicit_manifest_protocol=True,
)
_PROTOCOLS_BY_ID = {
    protocol.protocol_id: protocol
    for protocol in (
        TRANSITION_GROUNDED_V1_PROTOCOL,
        TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL,
    )
}


def transition_grounded_population_protocol(
    protocol_id: str,
) -> TransitionGroundedPopulationProtocolV1:
    """Return a known protocol; never admit an ad-hoc stratum set."""

    try:
        return _PROTOCOLS_BY_ID[protocol_id]
    except KeyError as error:
        raise ValueError(f"unknown transition-grounded cohort protocol: {protocol_id}") from error


def transition_grounded_population_protocol_ids() -> tuple[str, ...]:
    return tuple(sorted(_PROTOCOLS_BY_ID))


# Legacy names remain pinned to the historical three-strata protocol.
DISTANCE_STRATA = TRANSITION_GROUNDED_V1_PROTOCOL.distance_strata
FORMAL_MIN_CASE_COUNT = TRANSITION_GROUNDED_V1_PROTOCOL.case_count
FORMAL_MIN_STRATUM_COUNT = TRANSITION_GROUNDED_V1_PROTOCOL.cases_per_stratum
ALFWORLD_SPLITS = frozenset({"train", "valid_seen", "valid_unseen"})
VISIBLE_ARTIFACT_KEYS = frozenset(
    {
        "trace_case",
        "symbolic_state_evidence",
        "benchmark_precondition_failure_evidence",
    }
)
REPLAY_ONLY_ARTIFACT_KEYS = frozenset({"controlled_injection_plan", "controlled_failure_arm"})
POPULATION_CASE_KEYS = frozenset(
    {
        "case_id",
        "task_id",
        "group_uid",
        "distance_stratum",
        "causal_distance",
        "tracer_visible_artifacts",
        "replay_only_artifacts",
    }
)
BLIND_POPULATION_CASE_KEYS = frozenset(
    {"case_id", "task_id", "group_uid", "tracer_visible_artifacts"}
)
_FORBIDDEN_VISIBLE_CAUSAL_KEYS = frozenset(
    {
        "causal_event_id",
        "causal_gold",
        "gold_event_id",
        "gold_step_idx",
        "fault_event_id",
        "causal_event_ids",
        "injected_action",
        "injection_id",
        "injection_step_idx",
        "root_cause_event_id",
        "repair_action",
        "source_action",
    }
)
_FAILURE_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "task_id",
        "source_pre_state_facts_sha256",
        "proposal",
        "action_schema",
        "grounded_arguments",
        "violation_labels",
        "violations",
        "pair_outcome_used",
        "tracer_output_used",
        "precondition_gold_assigned",
        "content_sha256",
    }
)


def canonical_alfworld_group_identity(group_uid: str) -> str:
    """Compare ALFWorld task families across split-qualified group UIDs."""

    split, separator, signature = group_uid.partition(":")
    if separator and split in ALFWORLD_SPLITS and signature:
        return signature
    return group_uid


def _canonical_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_raw_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolve_artifact(
    root: Path,
    descriptor: object,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "path",
        "file_sha256",
    }:
        raise ValueError(f"{label} descriptor fields changed")
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes the population root") from error
    expected = _require_raw_sha256(descriptor.get("file_sha256"), label=f"{label} file hash")
    if not path.is_file() or _canonical_file_sha256(path) != expected:
        raise ValueError(f"{label} file hash changed")
    return path, _load_object(path, label=label)


def _visible_causal_label_keys(value: object) -> tuple[str, ...]:
    leaks: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            if key in _FORBIDDEN_VISIBLE_CAUSAL_KEYS:
                leaks.append(key)
            if key in {"pair_outcome_used", "tracer_output_used"} and child is not False:
                leaks.append(key)
            leaks.extend(_visible_causal_label_keys(child))
    elif isinstance(value, list):
        for child in value:
            leaks.extend(_visible_causal_label_keys(child))
    return tuple(sorted(set(leaks)))


def _distance_stratum(distance: int) -> str:
    if distance == 0:
        return "terminal_immediate"
    if 1 <= distance <= 2:
        return "short_suffix_1_2"
    if distance >= 3:
        return "long_suffix_ge_3"
    raise ValueError("controlled injection occurs after the interruption")


def _event_admissible_actions(content: str) -> tuple[str, ...]:
    prefix = "Admissible actions: "
    for line in content.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            value = ast.literal_eval(line[len(prefix) :])
        except (SyntaxError, ValueError) as error:
            raise ValueError("trace event admissible-action menu is invalid") from error
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("trace event admissible-action menu is invalid")
        return tuple(value)
    raise ValueError("trace event lacks an admissible-action menu")


@dataclass(frozen=True)
class TransitionGroundedPopulationPreflightV1:
    platform_revision: str
    development_task_manifest_file_sha256: str
    population_manifest_file_sha256: str
    population_manifest_content_sha256: str
    case_count: int
    task_count: int
    task_overlap_count: int
    group_overlap_count: int
    distance_stratum_counts: tuple[tuple[str, int], ...]
    max_intervention_trials_per_case: int
    tracer_visible_causal_label_count: int
    proposal_candidate_count: int
    authorized: bool
    gpu_accessed: bool = False
    schema: str = TRANSITION_GROUNDED_PREFLIGHT_SCHEMA

    def __post_init__(self) -> None:
        strata = dict(self.distance_stratum_counts)
        if (
            self.schema != TRANSITION_GROUNDED_PREFLIGHT_SCHEMA
            or not self.platform_revision
            or self.case_count < FORMAL_MIN_CASE_COUNT
            or self.task_count != self.case_count
            or self.task_overlap_count < 0
            or self.group_overlap_count < 0
            or set(strata) != set(DISTANCE_STRATA)
            or any(value < FORMAL_MIN_STRATUM_COUNT for value in strata.values())
            or sum(strata.values()) != self.case_count
            or self.max_intervention_trials_per_case != FORMAL_MAX_INTERVENTION_TRIALS
            or self.tracer_visible_causal_label_count < 0
            or self.proposal_candidate_count < 0
            or self.gpu_accessed
            or self.authorized
            != (
                self.task_overlap_count == 0
                and self.group_overlap_count == 0
                and self.tracer_visible_causal_label_count == 0
            )
        ):
            raise ValueError("transition-grounded preflight identity is invalid")
        _require_raw_sha256(
            self.development_task_manifest_file_sha256,
            label="development task manifest file hash",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["distance_stratum_counts"] = dict(self.distance_stratum_counts)
        return payload

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TransitionGroundedPopulationPreflightV2:
    """Preflight receipt for the prospective short/long V2 cohort only."""

    cohort_protocol: str
    platform_revision: str
    development_task_manifest_file_sha256: str
    population_manifest_file_sha256: str
    population_manifest_content_sha256: str
    case_count: int
    task_count: int
    task_overlap_count: int
    group_overlap_count: int
    distance_stratum_counts: tuple[tuple[str, int], ...]
    max_intervention_trials_per_case: int
    tracer_visible_causal_label_count: int
    proposal_candidate_count: int
    authorized: bool
    gpu_accessed: bool = False
    schema: str = TRANSITION_GROUNDED_TWO_STRATA_PREFLIGHT_SCHEMA

    def __post_init__(self) -> None:
        protocol = TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL
        strata = dict(self.distance_stratum_counts)
        if (
            self.schema != protocol.population_preflight_schema
            or self.cohort_protocol != protocol.protocol_id
            or not self.platform_revision
            or self.case_count != protocol.case_count
            or self.task_count != self.case_count
            or self.task_overlap_count < 0
            or self.group_overlap_count < 0
            or strata != protocol.expected_stratum_counts
            or self.max_intervention_trials_per_case
            != protocol.max_intervention_trials_per_case
            or self.tracer_visible_causal_label_count < 0
            or self.proposal_candidate_count < 0
            or self.gpu_accessed
            or self.authorized
            != (
                self.task_overlap_count == 0
                and self.group_overlap_count == 0
                and self.tracer_visible_causal_label_count == 0
            )
        ):
            raise ValueError("transition-grounded two-strata preflight identity is invalid")
        _require_raw_sha256(
            self.development_task_manifest_file_sha256,
            label="development task manifest file hash",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["distance_stratum_counts"] = dict(self.distance_stratum_counts)
        return payload

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TransitionGroundedBlindPopulationPreflightV1:
    platform_revision: str
    development_task_manifest_file_sha256: str
    blind_population_manifest_file_sha256: str
    blind_population_manifest_content_sha256: str
    source_population_manifest_file_sha256: str
    case_count: int
    task_count: int
    task_overlap_count: int
    group_overlap_count: int
    tracer_visible_causal_label_count: int
    proposal_candidate_count: int
    replay_only_opened: bool
    authorized: bool
    gpu_accessed: bool = False
    schema: str = TRANSITION_GROUNDED_BLIND_PREFLIGHT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != TRANSITION_GROUNDED_BLIND_PREFLIGHT_SCHEMA
            or not self.platform_revision
            or self.case_count < FORMAL_MIN_CASE_COUNT
            or self.task_count != self.case_count
            or self.task_overlap_count < 0
            or self.group_overlap_count < 0
            or self.tracer_visible_causal_label_count < 0
            or self.proposal_candidate_count < 0
            or self.replay_only_opened
            or self.gpu_accessed
            or self.authorized
            != (
                self.task_overlap_count == 0
                and self.group_overlap_count == 0
                and self.tracer_visible_causal_label_count == 0
            )
        ):
            raise ValueError("transition-grounded blind preflight identity is invalid")
        _require_raw_sha256(
            self.development_task_manifest_file_sha256,
            label="development task manifest file hash",
        )
        _require_raw_sha256(
            self.source_population_manifest_file_sha256,
            label="source population manifest file hash",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TransitionGroundedBlindPopulationPreflightV2:
    """Gold-blind preflight receipt bound to the V2 two-strata source cohort."""

    cohort_protocol: str
    platform_revision: str
    development_task_manifest_file_sha256: str
    blind_population_manifest_file_sha256: str
    blind_population_manifest_content_sha256: str
    source_population_manifest_file_sha256: str
    case_count: int
    task_count: int
    task_overlap_count: int
    group_overlap_count: int
    tracer_visible_causal_label_count: int
    proposal_candidate_count: int
    replay_only_opened: bool
    authorized: bool
    gpu_accessed: bool = False
    schema: str = TRANSITION_GROUNDED_TWO_STRATA_BLIND_PREFLIGHT_SCHEMA

    def __post_init__(self) -> None:
        protocol = TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL
        if (
            self.schema != protocol.blind_population_preflight_schema
            or self.cohort_protocol != protocol.protocol_id
            or not self.platform_revision
            or self.case_count != protocol.case_count
            or self.task_count != self.case_count
            or self.task_overlap_count < 0
            or self.group_overlap_count < 0
            or self.tracer_visible_causal_label_count < 0
            or self.proposal_candidate_count < 0
            or self.replay_only_opened
            or self.gpu_accessed
            or self.authorized
            != (
                self.task_overlap_count == 0
                and self.group_overlap_count == 0
                and self.tracer_visible_causal_label_count == 0
            )
        ):
            raise ValueError("transition-grounded two-strata blind preflight identity is invalid")
        _require_raw_sha256(
            self.development_task_manifest_file_sha256,
            label="development task manifest file hash",
        )
        _require_raw_sha256(
            self.source_population_manifest_file_sha256,
            label="source population manifest file hash",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return "sha256:" + canonical_sha256(self.to_dict())


def preflight_transition_grounded_blind_population(
    blind_manifest_path: Path,
    *,
    expected_blind_manifest_file_sha256: str,
    expected_source_population_manifest_file_sha256: str,
    development_task_ids: frozenset[str],
    development_group_uids: frozenset[str],
    development_task_manifest_file_sha256: str,
    expected_case_count: int,
    platform_revision: str,
    cohort_protocol_id: str = TRANSITION_GROUNDED_V1_PROTOCOL_ID,
) -> (
    TransitionGroundedBlindPopulationPreflightV1
    | TransitionGroundedBlindPopulationPreflightV2
):
    """Admit only tracer-visible bytes; never open replay authority artifacts."""

    protocol = transition_grounded_population_protocol(cohort_protocol_id)
    blind_manifest_path = blind_manifest_path.resolve()
    expected_blind_manifest_file_sha256 = _require_raw_sha256(
        expected_blind_manifest_file_sha256, label="blind population manifest file hash"
    )
    expected_source_population_manifest_file_sha256 = _require_raw_sha256(
        expected_source_population_manifest_file_sha256,
        label="source population manifest file hash",
    )
    development_task_manifest_file_sha256 = _require_raw_sha256(
        development_task_manifest_file_sha256,
        label="development task manifest file hash",
    )
    if _canonical_file_sha256(blind_manifest_path) != expected_blind_manifest_file_sha256:
        raise ValueError("blind population manifest file hash changed")
    manifest = _load_object(blind_manifest_path, label="transition-grounded blind population")
    cases = manifest.get("cases")
    if (
        manifest.get("schema") != protocol.blind_population_schema
        or manifest.get("status") != "frozen"
        or manifest.get("population_role") != "primary"
        or (
            protocol.requires_explicit_manifest_protocol
            and manifest.get("cohort_protocol") != protocol.protocol_id
        )
        or (
            not protocol.requires_explicit_manifest_protocol
            and "cohort_protocol" in manifest
        )
        or manifest.get("source_population_manifest_file_sha256")
        != expected_source_population_manifest_file_sha256
        or isinstance(expected_case_count, bool)
        or expected_case_count != protocol.case_count
        or not isinstance(cases, list)
        or manifest.get("case_count") != expected_case_count
        or len(cases) != expected_case_count
        or not platform_revision
    ):
        raise ValueError("blind formal population contract changed")

    root = blind_manifest_path.parent.resolve()
    seen_cases: set[str] = set()
    seen_tasks: set[str] = set()
    seen_groups: set[str] = set()
    visible_leaks: set[str] = set()
    proposal_candidate_count = 0
    for raw_case in cases:
        if not isinstance(raw_case, Mapping) or set(raw_case) != BLIND_POPULATION_CASE_KEYS:
            raise ValueError("blind population case projection changed")
        case_id = raw_case.get("case_id")
        task_id = raw_case.get("task_id")
        group_uid = raw_case.get("group_uid")
        group_identity = (
            canonical_alfworld_group_identity(group_uid) if isinstance(group_uid, str) else ""
        )
        visible = raw_case.get("tracer_visible_artifacts")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen_cases
            or not isinstance(task_id, str)
            or not task_id
            or task_id in seen_tasks
            or not isinstance(group_uid, str)
            or not group_uid
            or group_identity in seen_groups
            or not isinstance(visible, Mapping)
            or set(visible) != VISIBLE_ARTIFACT_KEYS
        ):
            raise ValueError("blind population case identity changed")
        seen_cases.add(case_id)
        seen_tasks.add(task_id)
        seen_groups.add(group_identity)
        _, trace_raw = _resolve_artifact(root, visible["trace_case"], label=f"{case_id} trace")
        _, symbolic_raw = _resolve_artifact(
            root, visible["symbolic_state_evidence"], label=f"{case_id} symbolic"
        )
        _, failure = _resolve_artifact(
            root,
            visible["benchmark_precondition_failure_evidence"],
            label=f"{case_id} failure evidence",
        )
        case = TraceCaseV1.from_dict(trace_raw)
        symbolic = TraceSymbolicStateEvidenceV1.from_dict(symbolic_raw)
        for payload in (trace_raw, symbolic_raw, failure):
            visible_leaks.update(_visible_causal_label_keys(payload))
        if visible_leaks:
            raise ValueError(
                "tracer-visible failure evidence contains causal label keys: "
                + ", ".join(sorted(visible_leaks))
            )
        if (
            trace_raw != case.to_dict()
            or symbolic_raw != symbolic.to_dict()
            or set(failure) != _FAILURE_EVIDENCE_KEYS
            or case.case_id != case_id
            or case.task_id != task_id
            or failure.get("task_id") != task_id
            or symbolic.source_trace_sha256 != case.sha256
            or len(symbolic.events) != len(case.events)
        ):
            raise ValueError(f"{case_id} blind visible artifact binding changed")
        for event, state in zip(case.events, symbolic.events, strict=True):
            metadata = dict(event.metadata)
            if (
                event.event_id != state.event_id
                or event.sequence != state.sequence
                or metadata.get("state_snapshot_sha256") != state.pre_state_snapshot_sha256
            ):
                raise ValueError(f"{case_id} blind per-step binding changed")
            _event_admissible_actions(event.content)
        proposal_candidate_count += len(
            build_alfworld_transition_grounded_proposal(case, symbolic, failure).candidates
        )
    task_overlap = seen_tasks & development_task_ids
    development_group_identities = {
        canonical_alfworld_group_identity(group_uid) for group_uid in development_group_uids
    }
    group_overlap = seen_groups & development_group_identities
    if task_overlap or group_overlap:
        raise ValueError("blind population overlaps development task/group")
    shared = {
        "platform_revision": platform_revision,
        "development_task_manifest_file_sha256": development_task_manifest_file_sha256,
        "blind_population_manifest_file_sha256": expected_blind_manifest_file_sha256,
        "blind_population_manifest_content_sha256": "sha256:" + canonical_sha256(manifest),
        "source_population_manifest_file_sha256": expected_source_population_manifest_file_sha256,
        "case_count": len(seen_cases),
        "task_count": len(seen_tasks),
        "task_overlap_count": len(task_overlap),
        "group_overlap_count": len(group_overlap),
        "tracer_visible_causal_label_count": len(visible_leaks),
        "proposal_candidate_count": proposal_candidate_count,
        "replay_only_opened": False,
        "authorized": True,
    }
    if protocol.protocol_id == TRANSITION_GROUNDED_V1_PROTOCOL_ID:
        return TransitionGroundedBlindPopulationPreflightV1(**shared)
    if protocol.protocol_id == TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL_ID:
        return TransitionGroundedBlindPopulationPreflightV2(
            cohort_protocol=protocol.protocol_id,
            **shared,
        )
    raise AssertionError("known transition-grounded protocol is not dispatched")


def preflight_transition_grounded_population(
    manifest_path: Path,
    *,
    expected_manifest_file_sha256: str,
    development_task_ids: frozenset[str],
    development_group_uids: frozenset[str],
    development_task_manifest_file_sha256: str,
    expected_case_count: int,
    expected_stratum_counts: Mapping[str, int],
    expected_max_intervention_trials: int,
    platform_revision: str,
    cohort_protocol_id: str = TRANSITION_GROUNDED_V1_PROTOCOL_ID,
) -> TransitionGroundedPopulationPreflightV1 | TransitionGroundedPopulationPreflightV2:
    """Admit one immutable population without executing GPU or environment calls."""

    protocol = transition_grounded_population_protocol(cohort_protocol_id)
    manifest_path = manifest_path.resolve()
    expected_manifest_file_sha256 = _require_raw_sha256(
        expected_manifest_file_sha256,
        label="population manifest file hash",
    )
    if _canonical_file_sha256(manifest_path) != expected_manifest_file_sha256:
        raise ValueError("population manifest file hash changed")
    development_task_manifest_file_sha256 = _require_raw_sha256(
        development_task_manifest_file_sha256,
        label="development task manifest file hash",
    )
    manifest = _load_object(manifest_path, label="transition-grounded population")
    cases = manifest.get("cases")
    if (
        manifest.get("schema") != protocol.population_schema
        or manifest.get("status") != "frozen"
        or manifest.get("population_role") != "primary"
        or (
            protocol.requires_explicit_manifest_protocol
            and manifest.get("cohort_protocol") != protocol.protocol_id
        )
        or (
            not protocol.requires_explicit_manifest_protocol
            and "cohort_protocol" in manifest
        )
        or isinstance(expected_case_count, bool)
        or expected_case_count != protocol.case_count
        or not isinstance(cases, list)
        or manifest.get("case_count") != expected_case_count
        or len(cases) != expected_case_count
        or isinstance(expected_max_intervention_trials, bool)
        or expected_max_intervention_trials < 1
        or manifest.get("max_intervention_trials_per_case") != expected_max_intervention_trials
        or not platform_revision
    ):
        raise ValueError("formal population denominator or contract changed")
    if (
        expected_max_intervention_trials != protocol.max_intervention_trials_per_case
        or manifest.get("max_intervention_trials_per_case")
        != protocol.max_intervention_trials_per_case
    ):
        raise ValueError("formal replay budget changed")
    normalized_expected_strata = {key: value for key, value in expected_stratum_counts.items()}
    if (
        normalized_expected_strata != protocol.expected_stratum_counts
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != protocol.cases_per_stratum
            for value in normalized_expected_strata.values()
        )
        or sum(normalized_expected_strata.values()) != expected_case_count
    ):
        raise ValueError("expected distance-stratum contract is invalid")

    root = manifest_path.parent.resolve()
    seen_cases: set[str] = set()
    seen_tasks: set[str] = set()
    seen_groups: set[str] = set()
    observed_strata = {stratum: 0 for stratum in protocol.distance_strata}
    proposal_candidate_count = 0
    visible_leaks: set[str] = set()
    for raw_case in cases:
        if not isinstance(raw_case, Mapping) or set(raw_case) != POPULATION_CASE_KEYS:
            raise ValueError("population case must be an object")
        case_id = raw_case.get("case_id")
        task_id = raw_case.get("task_id")
        group_uid = raw_case.get("group_uid")
        group_identity = (
            canonical_alfworld_group_identity(group_uid) if isinstance(group_uid, str) else ""
        )
        visible = raw_case.get("tracer_visible_artifacts")
        replay_only = raw_case.get("replay_only_artifacts")
        declared_distance = raw_case.get("causal_distance")
        declared_stratum = raw_case.get("distance_stratum")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen_cases
            or not isinstance(task_id, str)
            or not task_id
            or task_id in seen_tasks
            or not isinstance(group_uid, str)
            or not group_uid
            or group_identity in seen_groups
            or not isinstance(visible, Mapping)
            or set(visible) != VISIBLE_ARTIFACT_KEYS
            or not isinstance(replay_only, Mapping)
            or set(replay_only) != REPLAY_ONLY_ARTIFACT_KEYS
            or isinstance(declared_distance, bool)
            or not isinstance(declared_distance, int)
            or declared_stratum not in protocol.distance_strata
        ):
            raise ValueError("population case identity or artifact partition changed")
        seen_cases.add(case_id)
        seen_tasks.add(task_id)
        seen_groups.add(group_identity)

        _, trace_raw = _resolve_artifact(root, visible["trace_case"], label=f"{case_id} trace case")
        _, symbolic_raw = _resolve_artifact(
            root,
            visible["symbolic_state_evidence"],
            label=f"{case_id} symbolic state evidence",
        )
        _, failure_evidence = _resolve_artifact(
            root,
            visible["benchmark_precondition_failure_evidence"],
            label=f"{case_id} benchmark precondition failure evidence",
        )
        _, plan_raw = _resolve_artifact(
            root,
            replay_only["controlled_injection_plan"],
            label=f"{case_id} controlled injection plan",
        )
        _, failure_arm_raw = _resolve_artifact(
            root,
            replay_only["controlled_failure_arm"],
            label=f"{case_id} controlled failure arm",
        )
        case = TraceCaseV1.from_dict(trace_raw)
        symbolic = TraceSymbolicStateEvidenceV1.from_dict(symbolic_raw)
        plan = AlfWorldControlledFaultInjectionPlanV1.from_dict(plan_raw)
        failure_arm = ControlledFaultInjectionArmV1.from_dict(failure_arm_raw)
        for tracer_visible_payload in (trace_raw, symbolic_raw, failure_evidence):
            visible_leaks.update(_visible_causal_label_keys(tracer_visible_payload))
        if visible_leaks:
            raise ValueError(
                "tracer-visible failure evidence contains causal label keys: "
                + ", ".join(sorted(visible_leaks))
            )
        if trace_raw != case.to_dict() or symbolic_raw != symbolic.to_dict():
            raise ValueError(f"{case_id} tracer-visible typed projection changed")
        if set(failure_evidence) != _FAILURE_EVIDENCE_KEYS:
            raise ValueError(f"{case_id} failure-evidence projection changed")
        if case.context.task_specification != plan.task_description:
            raise ValueError(f"{case_id} task specification binding changed")
        if (
            case.case_id != case_id
            or case.task_id != task_id
            or plan.task_id != task_id
            or failure_evidence.get("task_id") != task_id
            or failure_arm.arm != "injected_failure"
            or failure_arm.success
            or failure_arm.terminal_reason != case.observed_outcome.terminal_reason
            or plan.suffix_controller_sha256 != failure_arm.suffix_controller_sha256
            or symbolic.source_trace_sha256 != case.sha256
            or len(case.events) != len(failure_arm.steps)
            or len(plan.steps) <= len(failure_arm.steps)
            or len(symbolic.events) != len(case.events)
            or failure_arm.steps[plan.injection.step_idx].action != plan.injection.injected_action
            or failure_arm.steps[plan.injection.step_idx].pre_state_snapshot_sha256
            != plan.injection.source_pre_state_snapshot_sha256
        ):
            raise ValueError(f"{case_id} typed artifact binding changed")
        rejected_source_action = plan.steps[len(failure_arm.steps)].source_action
        if failure_evidence.get("proposal") != rejected_source_action:
            raise ValueError(f"{case_id} rejected source action binding changed")
        for event, state, step in zip(
            case.events,
            symbolic.events,
            failure_arm.steps,
            strict=True,
        ):
            metadata = dict(event.metadata)
            if (
                event.event_id != state.event_id
                or event.sequence != state.sequence
                or event.sequence != step.step_idx
                or metadata.get("actor_action") != step.action
                or metadata.get("observation_before") != step.observation_before
                or metadata.get("observation_after") != step.observation_after
                or metadata.get("source_trajectory_sha256") != failure_arm.trajectory_sha256
                or metadata.get("state_snapshot_sha256") != step.pre_state_snapshot_sha256
                or state.pre_state_snapshot_sha256 != step.pre_state_snapshot_sha256
                or state.post_state_snapshot_sha256 != step.post_state_snapshot_sha256
                or _event_admissible_actions(event.content) != step.admissible_actions
            ):
                raise ValueError(f"{case_id} per-step replay binding changed")
        interruption_sequence = next(
            event.sequence for event in case.events if event.event_id == case.interruption_event_id
        )
        observed_distance = interruption_sequence - plan.injection.step_idx
        observed_stratum = _distance_stratum(observed_distance)
        if declared_distance != observed_distance or declared_stratum != observed_stratum:
            raise ValueError(f"{case_id} causal distance binding changed")
        observed_strata[observed_stratum] += 1
        proposal = build_alfworld_transition_grounded_proposal(
            case,
            symbolic,
            failure_evidence,
        )
        proposal_candidate_count += len(proposal.candidates)

    if observed_strata != normalized_expected_strata:
        raise ValueError("population distance-stratum denominator changed")
    overlap = seen_tasks & development_task_ids
    if overlap:
        raise ValueError(
            "population task overlap with development tasks: " + ", ".join(sorted(overlap))
        )
    development_group_identities = {
        canonical_alfworld_group_identity(group_uid) for group_uid in development_group_uids
    }
    group_overlap = seen_groups & development_group_identities
    if group_overlap:
        raise ValueError(
            "population group overlap with development groups: " + ", ".join(sorted(group_overlap))
        )
    manifest_content_sha256 = "sha256:" + canonical_sha256(manifest)
    shared = {
        "platform_revision": platform_revision,
        "development_task_manifest_file_sha256": development_task_manifest_file_sha256,
        "population_manifest_file_sha256": expected_manifest_file_sha256,
        "population_manifest_content_sha256": manifest_content_sha256,
        "case_count": len(seen_cases),
        "task_count": len(seen_tasks),
        "task_overlap_count": len(overlap),
        "group_overlap_count": len(group_overlap),
        "distance_stratum_counts": tuple(sorted(observed_strata.items())),
        "max_intervention_trials_per_case": expected_max_intervention_trials,
        "tracer_visible_causal_label_count": len(visible_leaks),
        "proposal_candidate_count": proposal_candidate_count,
        "authorized": True,
    }
    if protocol.protocol_id == TRANSITION_GROUNDED_V1_PROTOCOL_ID:
        return TransitionGroundedPopulationPreflightV1(**shared)
    if protocol.protocol_id == TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL_ID:
        return TransitionGroundedPopulationPreflightV2(
            cohort_protocol=protocol.protocol_id,
            **shared,
        )
    raise AssertionError("known transition-grounded protocol is not dispatched")


__all__ = [
    "DISTANCE_STRATA",
    "FORMAL_MAX_INTERVENTION_TRIALS",
    "FORMAL_MIN_CASE_COUNT",
    "FORMAL_MIN_STRATUM_COUNT",
    "TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL",
    "TRANSITION_GROUNDED_TWO_STRATA_PROTOCOL_ID",
    "TRANSITION_GROUNDED_TWO_STRATA_POPULATION_SCHEMA",
    "TRANSITION_GROUNDED_TWO_STRATA_BLIND_POPULATION_SCHEMA",
    "TRANSITION_GROUNDED_TWO_STRATA_PREFLIGHT_SCHEMA",
    "TRANSITION_GROUNDED_TWO_STRATA_BLIND_PREFLIGHT_SCHEMA",
    "TRANSITION_GROUNDED_V1_PROTOCOL",
    "TRANSITION_GROUNDED_V1_PROTOCOL_ID",
    "TRANSITION_GROUNDED_POPULATION_SCHEMA",
    "TRANSITION_GROUNDED_BLIND_POPULATION_SCHEMA",
    "TRANSITION_GROUNDED_PREFLIGHT_SCHEMA",
    "TRANSITION_GROUNDED_BLIND_PREFLIGHT_SCHEMA",
    "TransitionGroundedPopulationProtocolV1",
    "TransitionGroundedBlindPopulationPreflightV1",
    "TransitionGroundedBlindPopulationPreflightV2",
    "TransitionGroundedPopulationPreflightV1",
    "TransitionGroundedPopulationPreflightV2",
    "canonical_alfworld_group_identity",
    "preflight_transition_grounded_blind_population",
    "preflight_transition_grounded_population",
    "transition_grounded_population_protocol",
    "transition_grounded_population_protocol_ids",
]
