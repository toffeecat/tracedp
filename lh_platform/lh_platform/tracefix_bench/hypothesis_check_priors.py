"""Check-outcome priors informed by an author locator's step hypothesis.

A public check is more likely to refute when the step a locator suspects, or a later step,
wrote one of the check's subject paths.

The suspect paths come only from the public observer of the cell's starting trace: for
``step-k``, every path whose observation changed at step k or later; for a final assistant
``turn`` (the trace ended too early), every path that does not exist at the end plus the
paths changed by the last step. The prior is the locator-weighted mix

    P(refuted) = r * P(refuted | suspect or not) + (1 - r) * P_base(refuted),

and an existence check whose subject paths are observed present keeps its near-zero prior.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

HYPOTHESIS_PRIOR_ID = "traceelephant_all_at_once_suspect_paths_v1"
REFUTED_IF_SUSPECT = (63 + 1) / (81 + 2)
REFUTED_IF_NOT_SUSPECT = (3 + 1) / (16 + 2)
LOCATOR_RELIABILITY = (10 + 1) / (10 + 2)
UNKNOWN_PROBABILITY = 0.02  # checks almost never return an inconclusive verdict (0 of 116)


def _state(row: Mapping[str, object], key: str) -> dict[str, tuple]:
    return {
        item["path"]: (item.get("exists"), item.get("content_base64"))
        for item in row.get(key, [])  # type: ignore[union-attr]
    }


def suspect_paths(observer_rows: Sequence[Mapping[str, object]], event_id: str) -> set[str]:
    """Public paths a fault at ``event_id`` could have produced, from the public observer."""

    changed = [
        {p for p in set(_state(row, "before")) | set(_state(row, "after"))
         if _state(row, "before").get(p) != _state(row, "after").get(p)}
        for row in observer_rows
    ]
    if event_id.startswith("step-"):
        start = int(event_id.split("-")[1]) - 1
        return set().union(*changed[start:]) if changed[start:] else set()
    if event_id.startswith("turn-") and observer_rows:
        final = _state(observer_rows[-1], "after")
        missing = {p for p, (exists, _) in final.items() if not exists}
        return missing | (changed[-1] if changed else set())
    return set()


def hypothesis_adjusted_configuration(
    configuration: Mapping[str, object],
    *,
    checks: Sequence[Mapping[str, object]],
    observer_rows: Sequence[Mapping[str, object]],
    event_id: str,
    check_order: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return a copy of the frozen estimate configuration with hypothesis-informed checks."""

    suspect = suspect_paths(observer_rows, event_id)
    by_check = {c["check_id"]: c for c in checks}
    adjusted = copy.deepcopy(dict(configuration))
    for rule in adjusted["rules"]:  # type: ignore[union-attr]
        check = by_check.get(rule.get("contract_id"))
        if rule.get("action_type") != "run_registered_check" or check is None:
            continue
        if str(rule.get("provenance_id", "")).startswith("observed-existence"):
            continue  # an observed-present existence check stays uninformative
        base = next(o["probability"] for o in rule["outcomes"] if o["outcome_id"] == "refuted")
        hit = bool(set(check["resource_ids"]) & suspect)
        conditional = REFUTED_IF_SUSPECT if hit else REFUTED_IF_NOT_SUSPECT
        refuted = LOCATOR_RELIABILITY * conditional + (1.0 - LOCATOR_RELIABILITY) * base
        rule["outcomes"] = [
            {"outcome_id": "supported", "probability": 1.0 - refuted - UNKNOWN_PROBABILITY},
            {"outcome_id": "refuted", "probability": refuted},
            {"outcome_id": "unknown", "probability": UNKNOWN_PROBABILITY},
        ]
        rule["rule_id"] = f"hyp-{rule['rule_id']}"
        rule["provenance_id"] = f"{HYPOTHESIS_PRIOR_ID}:{event_id}:{'suspect' if hit else 'not_suspect'}"
        rule["provenance_detail"] = (
            "locator-weighted refutation prior from the suspect public paths of an author "
            "TraceElephant all-at-once step hypothesis"
        )
    adjusted["configuration_id"] = f"{adjusted['configuration_id']}+{HYPOTHESIS_PRIOR_ID}:{event_id}"
    return adjusted


__all__ = ["HYPOTHESIS_PRIOR_ID", "hypothesis_adjusted_configuration", "suspect_paths"]
