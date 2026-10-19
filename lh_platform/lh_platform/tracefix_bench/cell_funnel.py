"""Funnel ledger and cell schema v2.

Every candidate cell passes a fixed sequence of gates, from loading its source to final
admission. The ledger keeps one append-only row per candidate per gate, rejected or not,
so every reported population has an explicit denominator and every exclusion a reason
(knowledge base ch. 8: technical admission, difficulty strata and mechanism subsets are
separate decisions, and a hard case never leaves the main set for being hard).

Schema v2 binds a cell to its actors and execution semantics, carries the long-horizon
numbers of plan section 7, and keeps scorer-only fields out of the public record.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

FUNNEL_SCHEMA = "tracefix_cell_funnel_row_v1"
CELL_V2_SCHEMA = "tracefix_cell_v2"

# Gates in order. S* admit a source, Q* qualify an injected cell (spec v1 plus Q7-Q9),
# then technical admission, the two actor-relative difficulty gates and the final verdict.
GATES = (
    "S3_source_load",
    "S1_source_replay",
    "S2_source_parity",
    "ladder",
    "site_enumeration",
    "Q1_source_success",
    "Q2_injected_failure",
    "Q3_undo",
    "Q4_menu_visibility",
    "Q5_history_parity",
    "Q6_motif",
    "Q7_dormancy",
    "Q8_distance",
    "Q8b_misleading_last_writer",
    "Q9_irreversibility",
    "Q10_sticky_obligation",
    "visibility",
    "technical_admission",
    "self_rescue",
    "competence",
    "admitted",
)
# A failed gate here moves the cell to a stratum instead of excluding it.
STRATIFYING_GATES = frozenset(
    {"self_rescue", "competence", "Q7_dormancy", "Q8_distance", "Q10_sticky_obligation", "visibility"}
)
# Method-side properties (plan section 5.3) a cell is labelled with and reported by.
METHOD_PROPERTIES = (
    "P1_destroyed_input",
    "P2_distant_cause",
    "P3_environment_dependency",
    "P4_feedback_insufficient",
    "P5_tight_budget",
    "P6_costly_rollback",
    "P7_jit_task_coverage",
)
# A failure no supervisor can see: a refuted menu check, an inventory or observer change,
# or a failing clean-context probe. Cells visible through none stay out of the main table.
VISIBILITY_CHANNELS = ("menu_refutation", "inventory_or_observer", "clean_context")

EXECUTION_SEMANTICS = ("same_actor_continuation", "cross_actor_takeover", "fresh_retry")
SOURCES = ("R1_external_natural", "R2_injected", "R3_own_natural")
SCORER_ONLY_KEYS = frozenset(
    {
        "gold_sites", "root_call", "root_label", "undo_receipts", "external_labels", "three_arm_receipts",
        "last_writer_calls", "feedback_trials",
    }
)


class FunnelLedger:
    """Append-only jsonl: one row per (candidate, gate) evaluation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(
        self,
        *,
        candidate_id: str,
        source_task: str,
        gate: str,
        passed: bool | None,
        reason: str | None = None,
        detail: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """``passed=None`` marks an infrastructure outcome: no verdict, no denominator change."""

        if gate not in GATES:
            raise ValueError(f"unknown funnel gate: {gate}")
        if passed is False and not reason:
            raise ValueError("a rejection needs a reason")
        row = {
            "schema_version": FUNNEL_SCHEMA,
            "recorded_at": time.time(),
            "candidate_id": candidate_id,
            "source_task": source_task,
            "gate": gate,
            "passed": passed,
            **({"reason": reason} if reason else {}),
            **({"detail": dict(detail)} if detail else {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return row


def read_funnel(path: str | Path) -> list[dict[str, object]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("schema_version") != FUNNEL_SCHEMA:
                raise ValueError("funnel row has an unknown schema")
            rows.append(row)
    return rows


def funnel_summary(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Per gate: candidates evaluated, passed, rejected (with reasons), infrastructure-only.

    The latest verdict of a candidate at a gate wins, so a retried infrastructure failure
    that later passes is counted once, as passed.
    """

    latest: dict[tuple[str, str], Mapping[str, object]] = {}
    tasks: dict[str, str] = {}
    for row in rows:
        latest[(str(row["candidate_id"]), str(row["gate"]))] = row
        tasks[str(row["candidate_id"])] = str(row["source_task"])
    gates: dict[str, dict[str, object]] = {}
    for gate in GATES:
        verdicts = [row for (_, g), row in latest.items() if g == gate]
        if not verdicts:
            continue
        reasons = Counter(str(row.get("reason")) for row in verdicts if row["passed"] is False)
        gates[gate] = {
            "evaluated": sum(1 for row in verdicts if row["passed"] is not None),
            "passed": sum(1 for row in verdicts if row["passed"] is True),
            "rejected": sum(1 for row in verdicts if row["passed"] is False),
            "infrastructure_only": sum(1 for row in verdicts if row["passed"] is None),
            "stratifying": gate in STRATIFYING_GATES,
            "rejection_reasons": dict(reasons.most_common()),
        }
    admitted = sorted(cid for (cid, gate), row in latest.items() if gate == "admitted" and row["passed"] is True)
    return {
        "candidates": len(tasks),
        "source_tasks": len(set(tasks.values())),
        "admitted": len(admitted),
        "admitted_source_tasks": len({tasks[cid] for cid in admitted}),
        "gates": gates,
    }


@dataclass(frozen=True)
class LongHorizon:
    """Plan section 7. ``None`` means not measured, never zero."""

    l_pi: int
    l_star: int | None = None
    root_call: int | None = None
    # Per refuted subject, the call that last wrote it: with ``root_call`` this is the pair
    # that scores a hypothesis as exact or off by k without any model call (P2, Q8b).
    last_writer_calls: tuple[tuple[str, int], ...] = ()
    d_end: int | None = None
    h_c: int | None = None
    h_f: int | None = None
    d_lock: int | None = None
    rollback_cost_calls: int | None = None
    context_tokens: int | None = None


def last_writer_offsets(horizon: LongHorizon) -> dict[str, int]:
    """Signed distance from the true root to each subject's last writer (0 = the axis is right)."""

    if horizon.root_call is None:
        return {}
    return {subject: call - horizon.root_call for subject, call in horizon.last_writer_calls}


@dataclass(frozen=True)
class FeedbackTrial:
    """One isolated feedback trial (Q10): the Actor is told one refuted obligation."""

    cell_id: str
    obligation: str  # the refuted public check
    trial: int
    fixed: bool  # the obligation's check passes after the trial
    input_tokens: int | None = None
    # Actor calls after the feedback until it first changed the refuted subject (1 = the
    # very next call); None = it never did within ``calls_after_feedback`` calls. Lets the
    # method side estimate P(stall | not fixed) for a stall-escalation trigger.
    first_subject_change_after: int | None = None
    calls_after_feedback: int | None = None


def feedback_recovery_rates(trials: Iterable[FeedbackTrial]) -> dict[str, dict[str, object]]:
    """Per obligation: trials, fixed, rate, and whether it is sticky (fixed in at most 1 of 3)."""

    grouped: dict[str, list[FeedbackTrial]] = {}
    for trial in trials:
        grouped.setdefault(trial.obligation, []).append(trial)
    rates = {}
    for obligation, rows in sorted(grouped.items()):
        fixed = sum(1 for row in rows if row.fixed)
        unfixed = [row for row in rows if not row.fixed and row.calls_after_feedback is not None]
        stalled = [row for row in unfixed if row.first_subject_change_after is None]
        first_changes = sorted(
            row.first_subject_change_after for row in rows if row.first_subject_change_after is not None
        )
        rates[obligation] = {
            "trials": len(rows),
            "fixed": fixed,
            "rate": fixed / len(rows),
            "cells": sorted({row.cell_id for row in rows}),
            "sticky": len(rows) >= 3 and (len(rows) - fixed) >= 2,
            # Of the trials that did not fix it, how many never touched the subject at all.
            "stalled_given_unfixed": (len(stalled), len(unfixed)),
            "first_subject_change_after": first_changes,
        }
    return rates


def horizon_stratum(horizon: LongHorizon) -> str:
    """long: H_c >= 3 or D_end >= 50; mid: D_end 10-49; short: D_end < 10; else unknown."""

    if (horizon.h_c or 0) >= 3 or (horizon.d_end or 0) >= 50:
        return "long"
    if horizon.d_end is None:
        return "unknown"
    return "mid" if horizon.d_end >= 10 else "short"


@dataclass(frozen=True)
class CellIdentity:
    cell_id: str
    source_task: str
    environment: str
    source: str
    source_actor: str
    recovering_actor: str
    execution_semantics: str
    split: str


def cell_v2_records(
    identity: CellIdentity,
    horizon: LongHorizon,
    *,
    strata: Mapping[str, object],
    public: Mapping[str, object],
    scorer: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Public and scorer records of one cell; scorer-only fields may not leak to public."""

    if identity.execution_semantics not in EXECUTION_SEMANTICS:
        raise ValueError(f"unknown execution semantics: {identity.execution_semantics}")
    if identity.source not in SOURCES:
        raise ValueError(f"unknown cell source: {identity.source}")
    if identity.split not in ("development", "calibration", "confirmation"):
        raise ValueError(f"unknown split: {identity.split}")
    leaked = SCORER_ONLY_KEYS & (set(public) | set(strata))
    if leaked:
        raise ValueError(f"scorer-only fields in the public record: {sorted(leaked)}")
    unknown = set(strata.get("method_properties", ())) - set(METHOD_PROPERTIES)
    if unknown:
        raise ValueError(f"unknown method properties: {sorted(unknown)}")
    channels = set(strata.get("visibility", ()))
    if not channels <= set(VISIBILITY_CHANNELS):
        raise ValueError(f"unknown visibility channels: {sorted(channels - set(VISIBILITY_CHANNELS))}")
    horizon_payload = asdict(horizon)
    # D_end and the root position reveal where the root is; only the stratum is public.
    public_horizon = {"l_pi": horizon.l_pi, "stratum": horizon_stratum(horizon)}
    public_record = {
        "schema_version": CELL_V2_SCHEMA,
        "record": "public",
        **asdict(identity),
        "long_horizon": public_horizon,
        "strata": dict(strata),
        **dict(public),
    }
    scorer_record = {
        "schema_version": CELL_V2_SCHEMA,
        "record": "scorer",
        "cell_id": identity.cell_id,
        "long_horizon": {
            **horizon_payload,
            "last_writer_calls": dict(horizon.last_writer_calls),
            "last_writer_offsets": last_writer_offsets(horizon),
            "stratum": horizon_stratum(horizon),
        },
        **dict(scorer),
    }
    trials = scorer_record.get("feedback_trials")
    if trials is not None:
        scorer_record["feedback_trials"] = [
            asdict(trial) if isinstance(trial, FeedbackTrial) else dict(trial) for trial in trials  # type: ignore[union-attr]
        ]
    return public_record, scorer_record
