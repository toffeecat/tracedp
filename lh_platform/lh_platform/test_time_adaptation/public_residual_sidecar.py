"""Contract for runtime-public residual/frontier sidecars.

The sidecar is written beside a trajectory, not inserted into the actor
prompt.  It must be produced by the environment adapter during execution and
must not contain evaluator or hidden-fact fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

FORBIDDEN_KEYS = frozenset(
    {"gold", "evaluator", "oracle", "hidden_facts", "future_state", "backend_state"}
)
PUBLIC_TRANSITION_KEYS = frozenset(
    {
        "kind",
        "task_id",
        "event_id",
        "sequence",
        "public_predicates_before",
        "public_predicates_after",
        "visibility",
    }
)


@dataclass(frozen=True)
class PublicTransitionEvidence:
    """The complete public input available to a live transition adapter."""

    task_id: str
    event_id: str
    sequence: int
    action: str
    observation: str
    next_observation: str
    admissible_actions: tuple[str, ...]
    next_admissible_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.event_id or self.sequence < 0 or not self.action:
            raise ValueError("public transition identity is invalid")


@dataclass(frozen=True)
class PublicTransitionSidecar:
    """A live sidecar that cannot carry goal labels or hidden state."""

    task_id: str
    event_id: str
    sequence: int
    public_predicates_before: tuple[tuple[str, bool], ...] = ()
    public_predicates_after: tuple[tuple[str, bool], ...] = ()
    kind: str = "public_transition_v1"
    visibility: str = "actor_visible_transition"

    def __post_init__(self) -> None:
        if not self.task_id or not self.event_id or self.sequence < 0:
            raise ValueError("public transition sidecar identity is invalid")
        if self.kind != "public_transition_v1" or self.visibility != "actor_visible_transition":
            raise ValueError("unsupported public transition sidecar")
        for predicates in (self.public_predicates_before, self.public_predicates_after):
            if any(not key or type(value) is not bool for key, value in predicates):
                raise ValueError("public transition predicates must be boolean pairs")
            if any(
                any(token in key.casefold() for token in FORBIDDEN_KEYS) for key, _ in predicates
            ):
                raise ValueError("public transition predicates contain hidden/evaluator labels")

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "task_id": self.task_id,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "public_predicates_before": [list(item) for item in self.public_predicates_before],
            "public_predicates_after": [list(item) for item in self.public_predicates_after],
            "visibility": self.visibility,
        }


@dataclass(frozen=True)
class PublicResidualSidecar:
    task_id: str
    event_id: str
    sequence: int
    residual_goal_ids: tuple[str, ...]
    active_frontier_ids: tuple[str, ...]
    blocked_child_ids: tuple[str, ...]
    regressed_goal_ids: tuple[str, ...]
    blocked_prerequisite_ids: tuple[str, ...] = ()
    public_predicates_before: tuple[tuple[str, bool], ...] = ()
    public_predicates_after: tuple[tuple[str, bool], ...] = ()
    visibility: str = "runtime_public_projection"

    def __post_init__(self) -> None:
        if not self.task_id or not self.event_id or self.sequence < 0:
            raise ValueError("sidecar identity is invalid")
        if self.visibility != "runtime_public_projection":
            raise ValueError("unsupported sidecar visibility")
        for values in (
            self.residual_goal_ids,
            self.active_frontier_ids,
            self.blocked_child_ids,
            self.blocked_prerequisite_ids,
            self.regressed_goal_ids,
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError("sidecar ids must be non-empty strings")
        for predicates in (self.public_predicates_before, self.public_predicates_after):
            if any(not key or type(value) is not bool for key, value in predicates):
                raise ValueError("sidecar public predicates must be boolean pairs")

    def to_record(self) -> dict[str, Any]:
        return asdict(self) | {
            "residual_goal_ids": list(self.residual_goal_ids),
            "active_frontier_ids": list(self.active_frontier_ids),
            "blocked_child_ids": list(self.blocked_child_ids),
            "blocked_prerequisite_ids": list(self.blocked_prerequisite_ids),
            "regressed_goal_ids": list(self.regressed_goal_ids),
            "public_predicates_before": [list(item) for item in self.public_predicates_before],
            "public_predicates_after": [list(item) for item in self.public_predicates_after],
        }


def validate_public_sidecar_record(record: Mapping[str, Any]) -> PublicResidualSidecar:
    lowered = {str(key).lower() for key in record}
    if any(any(token in key for token in FORBIDDEN_KEYS) for key in lowered):
        raise ValueError("sidecar contains forbidden hidden/evaluator field")

    def ids(name: str, *, optional: bool = False) -> tuple[str, ...]:
        value = record.get(name, () if optional else None)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"sidecar field {name} must be a sequence")
        return tuple(str(item) for item in value)

    def predicates(name: str) -> tuple[tuple[str, bool], ...]:
        value = record.get(name, ())
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"sidecar field {name} must be a sequence")
        pairs = tuple(tuple(item) for item in value)
        if any(
            len(item) != 2 or not isinstance(item[0], str) or type(item[1]) is not bool
            for item in pairs
        ):
            raise ValueError(f"sidecar field {name} must contain boolean pairs")
        return tuple((str(key), value) for key, value in pairs)

    return PublicResidualSidecar(
        task_id=str(record.get("task_id", "")),
        event_id=str(record.get("event_id", "")),
        sequence=int(record.get("sequence", -1)),
        residual_goal_ids=ids("residual_goal_ids"),
        active_frontier_ids=ids("active_frontier_ids"),
        blocked_child_ids=ids("blocked_child_ids"),
        blocked_prerequisite_ids=ids("blocked_prerequisite_ids", optional=True),
        regressed_goal_ids=ids("regressed_goal_ids"),
        public_predicates_before=predicates("public_predicates_before"),
        public_predicates_after=predicates("public_predicates_after"),
        visibility=str(record.get("visibility", "")),
    )


def validate_public_transition_sidecar_record(
    record: Mapping[str, Any],
) -> PublicTransitionSidecar:
    unknown = set(record) - PUBLIC_TRANSITION_KEYS
    if unknown:
        raise ValueError(f"public transition sidecar has unsupported fields: {sorted(unknown)}")
    lowered = {str(key).lower() for key in record}
    if any(any(token in key for token in FORBIDDEN_KEYS) for key in lowered):
        raise ValueError("public transition sidecar contains hidden/evaluator fields")

    def predicates(name: str) -> tuple[tuple[str, bool], ...]:
        value = record.get(name, ())
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"public transition field {name} must be a sequence")
        pairs = tuple(tuple(item) for item in value)
        if any(
            len(item) != 2 or not isinstance(item[0], str) or type(item[1]) is not bool
            for item in pairs
        ):
            raise ValueError(f"public transition field {name} must contain boolean pairs")
        return tuple((str(key), value) for key, value in pairs)

    return PublicTransitionSidecar(
        task_id=str(record.get("task_id", "")),
        event_id=str(record.get("event_id", "")),
        sequence=int(record.get("sequence", -1)),
        public_predicates_before=predicates("public_predicates_before"),
        public_predicates_after=predicates("public_predicates_after"),
        kind=str(record.get("kind", "")),
        visibility=str(record.get("visibility", "")),
    )


__all__ = [
    "PublicResidualSidecar",
    "PublicTransitionEvidence",
    "PublicTransitionSidecar",
    "validate_public_sidecar_record",
    "validate_public_transition_sidecar_record",
]
