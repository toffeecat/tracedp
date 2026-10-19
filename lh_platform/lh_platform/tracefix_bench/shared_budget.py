"""One frozen budget pool shared by every role of a compared arm, enforced at dispatch.

A paired comparison (for example Dynamic TraceDP versus the Who&When binary baseline)
is only fair when both arms draw tracing, prediction, planning, checks, repairs, actor
continuation and retries from the same pool.  ``SharedRoleBudgetV1`` freezes that pool
and a single per-call output cap; ``SharedBudgetLedger`` reserves the worst case before
each model dispatch, refuses a dispatch the pool cannot cover, and settles actual usage
afterwards.  Usage the provider did not report is charged at the reservation, and the
ledger records that metering was incomplete.

How an arm splits the pool between roles is the arm's own decision; the pool itself
must be identical across arms (``require_identical_budgets``).
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lh_platform.tracefix_bench.scorer_contracts import ResourceRole

SHARED_ROLE_BUDGET_SCHEMA = "longhorizon_tracefix_bench/shared_role_budget/v1"


class BudgetRefusedError(RuntimeError):
    """A dispatch was refused before it reached the provider."""


def _non_negative_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _non_negative_number(value: object, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")


@dataclass(frozen=True)
class SharedRoleBudgetV1:
    """The complete pool one arm may spend on one case."""

    budget_id: str
    per_call_output_tokens: int
    total_model_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_wall_time_seconds: float
    total_monetary_cost: float | None
    roles: tuple[ResourceRole, ...]
    schema: str = SHARED_ROLE_BUDGET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHARED_ROLE_BUDGET_SCHEMA:
            raise ValueError("unsupported shared role budget schema")
        if not isinstance(self.budget_id, str) or not self.budget_id.strip():
            raise ValueError("shared budget requires a budget_id")
        for label in (
            "per_call_output_tokens",
            "total_model_calls",
            "total_input_tokens",
            "total_output_tokens",
        ):
            _non_negative_int(getattr(self, label), label)
        if self.per_call_output_tokens < 1:
            raise ValueError("per_call_output_tokens must be positive")
        if self.per_call_output_tokens > self.total_output_tokens:
            raise ValueError("per-call output cap cannot exceed the output pool")
        _non_negative_number(self.total_wall_time_seconds, "total_wall_time_seconds")
        if self.total_monetary_cost is not None:
            _non_negative_number(self.total_monetary_cost, "total_monetary_cost")
        if not self.roles or any(not isinstance(role, ResourceRole) for role in self.roles):
            raise ValueError("shared budget roles must be a nonempty tuple of ResourceRole")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("shared budget roles must be unique")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["roles"] = [role.value for role in self.roles]
        return payload


def require_identical_budgets(*budgets: SharedRoleBudgetV1) -> SharedRoleBudgetV1:
    """Refuse a comparison whose arms were given different pools."""

    if not budgets:
        raise ValueError("at least one budget is required")
    first = budgets[0].to_dict()
    for other in budgets[1:]:
        if other.to_dict() != first:
            raise ValueError("compared arms must share one identical budget pool")
    return budgets[0]


@dataclass
class _Reservation:
    reservation_id: int
    role: ResourceRole
    input_tokens: int
    output_tokens: int
    settled: bool = False


@dataclass
class SharedBudgetLedger:
    """Reserve before dispatch, settle afterwards, never exceed the pool."""

    budget: SharedRoleBudgetV1
    _reservations: dict[int, _Reservation] = field(default_factory=dict)
    _next_id: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_seconds: float = 0.0
    monetary_cost: float = 0.0
    metering_complete: bool = True
    by_role: dict[str, dict[str, float]] = field(default_factory=dict)

    def _pending(self) -> tuple[int, int, int]:
        open_items = [item for item in self._reservations.values() if not item.settled]
        return (
            len(open_items),
            sum(item.input_tokens for item in open_items),
            sum(item.output_tokens for item in open_items),
        )

    def reserve(self, role: ResourceRole, *, input_tokens: int) -> int:
        """Reserve one model dispatch at its worst case, or refuse it."""

        if role not in self.budget.roles:
            raise BudgetRefusedError(f"role {role.value} is not part of this budget")
        _non_negative_int(input_tokens, "input_tokens")
        pending_calls, pending_input, pending_output = self._pending()
        output = self.budget.per_call_output_tokens
        if self.calls + pending_calls + 1 > self.budget.total_model_calls:
            raise BudgetRefusedError("model call pool exhausted")
        if self.input_tokens + pending_input + input_tokens > self.budget.total_input_tokens:
            raise BudgetRefusedError("input token pool cannot cover this dispatch")
        if self.output_tokens + pending_output + output > self.budget.total_output_tokens:
            raise BudgetRefusedError("output token pool cannot cover the per-call cap")
        if self.wall_time_seconds >= self.budget.total_wall_time_seconds:
            raise BudgetRefusedError("wall-time pool exhausted")
        if (
            self.budget.total_monetary_cost is not None
            and self.monetary_cost >= self.budget.total_monetary_cost
        ):
            raise BudgetRefusedError("monetary pool exhausted")
        reservation_id = self._next_id
        self._next_id += 1
        self._reservations[reservation_id] = _Reservation(
            reservation_id, role, input_tokens, output
        )
        return reservation_id

    def settle(
        self,
        reservation_id: int,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        wall_time_seconds: float,
        monetary_cost: float | None = None,
    ) -> None:
        """Record actual usage; unknown token counts are charged at the reservation."""

        item = self._reservations.get(reservation_id)
        if item is None or item.settled:
            raise ValueError("unknown or already settled reservation")
        _non_negative_number(wall_time_seconds, "wall_time_seconds")
        if input_tokens is None or output_tokens is None:
            self.metering_complete = False
        used_input = item.input_tokens if input_tokens is None else input_tokens
        used_output = item.output_tokens if output_tokens is None else output_tokens
        _non_negative_int(used_input, "input_tokens")
        _non_negative_int(used_output, "output_tokens")
        if used_output > self.budget.per_call_output_tokens:
            raise ValueError("provider returned more output than the per-call cap")
        if monetary_cost is None:
            if self.budget.total_monetary_cost is not None:
                self.metering_complete = False
        else:
            _non_negative_number(monetary_cost, "monetary_cost")
            self.monetary_cost += monetary_cost
        item.settled = True
        self.calls += 1
        self.input_tokens += used_input
        self.output_tokens += used_output
        self.wall_time_seconds += wall_time_seconds
        role_row = self.by_role.setdefault(
            item.role.value,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "wall_time_seconds": 0.0},
        )
        role_row["calls"] += 1
        role_row["input_tokens"] += used_input
        role_row["output_tokens"] += used_output
        role_row["wall_time_seconds"] += wall_time_seconds

    def record_wall_time(self, role: ResourceRole, *, wall_time_seconds: float) -> None:
        """Charge wall time spent without a model call (probes, replays) to the pool.

        No call is counted and no tokens move; a pool already past its wall-time cap
        refuses every later dispatch through ``reserve``.
        """

        if role not in self.budget.roles:
            raise BudgetRefusedError(f"role {role.value} is not part of this budget")
        _non_negative_number(wall_time_seconds, "wall_time_seconds")
        self.wall_time_seconds += wall_time_seconds
        role_row = self.by_role.setdefault(
            role.value,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "wall_time_seconds": 0.0},
        )
        role_row["wall_time_seconds"] += wall_time_seconds

    def max_settled_input(self, role: ResourceRole) -> int:
        """Largest settled reservation estimate for ``role`` (request bytes // 4).

        Context grows along a session, so this is the Actor's latest request size; isolated
        repair sessions reserve smaller requests, so the maximum never under-projects.
        """

        return max(
            (item.input_tokens for item in self._reservations.values()
             if item.settled and item.role is role),
            default=0,
        )

    def input_room(self) -> int:
        """Input tokens still admissible: the pool minus settled and pending input.

        A read-only view of the quantity ``reserve`` tests. Pending dispatches count at
        their reservation, so the view is exact between calls and never over-admits.
        """

        _pending_calls, pending_input, _pending_output = self._pending()
        return self.budget.total_input_tokens - self.input_tokens - pending_input

    def receipt(self) -> dict[str, object]:
        pending_calls, _, _ = self._pending()
        if pending_calls:
            raise ValueError("cannot issue a budget receipt with unsettled dispatches")
        return {
            "budget": self.budget.to_dict(),
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_time_seconds": self.wall_time_seconds,
            "monetary_cost": self.monetary_cost,
            "metering_complete": self.metering_complete,
            "by_role": self.by_role,
        }


def _ledger_state(ledger: SharedBudgetLedger) -> dict[str, object]:
    return {
        "budget": ledger.budget.to_dict(),
        "reservations": [asdict(item) | {"role": item.role.value} for item in ledger._reservations.values()],
        "next_id": ledger._next_id,
        "calls": ledger.calls,
        "input_tokens": ledger.input_tokens,
        "output_tokens": ledger.output_tokens,
        "wall_time_seconds": ledger.wall_time_seconds,
        "monetary_cost": ledger.monetary_cost,
        "metering_complete": ledger.metering_complete,
        "by_role": ledger.by_role,
    }


def _budget_from_dict(payload: dict) -> SharedRoleBudgetV1:
    values = dict(payload)
    values["roles"] = tuple(ResourceRole(role) for role in values["roles"])
    return SharedRoleBudgetV1(**values)


def _ledger_from_state(state: dict) -> SharedBudgetLedger:
    ledger = SharedBudgetLedger(_budget_from_dict(state["budget"]))
    for row in state["reservations"]:
        item = _Reservation(
            row["reservation_id"],
            ResourceRole(row["role"]),
            row["input_tokens"],
            row["output_tokens"],
            row["settled"],
        )
        ledger._reservations[item.reservation_id] = item
    ledger._next_id = state["next_id"]
    for key in (
        "calls",
        "input_tokens",
        "output_tokens",
        "wall_time_seconds",
        "monetary_cost",
        "metering_complete",
        "by_role",
    ):
        setattr(ledger, key, state[key])
    return ledger


class FileSharedBudgetLedger:
    """The same ledger persisted to one JSON file under an exclusive file lock.

    Separate processes (a tracer worker, the host actor, a planner) that point at the
    same file draw from one pool.  Input tokens are unknown before dispatch, so the
    input pool admits on settled usage plus the caller's estimate and can overshoot by
    at most one call's input; the output pool reserves the full per-call cap.
    """

    def __init__(self, path: Path, budget: SharedRoleBudgetV1 | None = None) -> None:
        self.path = Path(path)
        if budget is not None:
            with self._locked() as handle:
                if handle.read().strip():
                    raise ValueError("budget ledger file already initialized")
                self._write(handle, _ledger_state(SharedBudgetLedger(budget)))

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write(handle, state: dict[str, object]) -> None:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(state, ensure_ascii=False, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())

    def _update(self, action):
        with self._locked() as handle:
            text = handle.read()
            if not text.strip():
                raise ValueError("budget ledger file is not initialized")
            ledger = _ledger_from_state(json.loads(text))
            result = action(ledger)
            self._write(handle, _ledger_state(ledger))
            return result

    def reserve(self, role: ResourceRole, *, input_tokens: int) -> int:
        return self._update(lambda ledger: ledger.reserve(role, input_tokens=input_tokens))

    def settle(self, reservation_id: int, **usage) -> None:
        self._update(lambda ledger: ledger.settle(reservation_id, **usage))

    def record_wall_time(self, role: ResourceRole, *, wall_time_seconds: float) -> None:
        self._update(lambda ledger: ledger.record_wall_time(role, wall_time_seconds=wall_time_seconds))

    def receipt(self) -> dict[str, object]:
        return self._update(lambda ledger: ledger.receipt())

    def input_room(self) -> int:
        return self._update(lambda ledger: ledger.input_room())

    def max_settled_input(self, role: ResourceRole) -> int:
        return self._update(lambda ledger: ledger.max_settled_input(role))

    def budget(self) -> SharedRoleBudgetV1:
        return self._update(lambda ledger: ledger.budget)


__all__ = [
    "FileSharedBudgetLedger",
    "SHARED_ROLE_BUDGET_SCHEMA",
    "BudgetRefusedError",
    "SharedBudgetLedger",
    "SharedRoleBudgetV1",
    "require_identical_budgets",
]
