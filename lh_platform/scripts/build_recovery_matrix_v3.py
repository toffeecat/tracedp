#!/usr/bin/env python3
"""Derive a recovery matrix (v3) from a frozen v2 matrix.

The derived matrix:
- gives every arm the source actor's output cap (one per-call cap for all roles) and one
  frozen SharedRoleBudgetV1 pool;
- adds the ``no-localization`` lower-bound arm (same actor, Runtime observes only);
  no oracle arm unless the matrix names formal gold;
- keeps the one-step (``residual-greedy``) and two-step (``tracedp``) closed loops, so the
  DP comparison differs only in lookahead depth;
- drops JIT arms whose blockers are recorded in v2, with the reason;
- replaces the hand-maintained ``required_before_launch`` list with prerequisites that
  ``matrix_launch_blockers`` checks from the matrix itself.

The binary (Who&When) arm is launched by its own entry point and draws from the same
budget pool; it is listed in ``external_arms`` so the comparison block is complete.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

V3_SCHEMA_NOTE = "recovery_composition_matrix_v1 contract_v0_2"
RUNTIME_GATE_POLICY = "runtime_candidate_trial_gate_v1"
MECHANICAL_PUBLIC_REPAIR = "public_menu_mechanical_repair_v1"
DEFAULT_SHARED_BUDGET = {
    "per_call_output_tokens": 16384,
    "total_model_calls": 120,
    "total_input_tokens": 300_000,
    "total_output_tokens": 100_000,
    # Equals the launcher's preregistered uniform 3600 s actor bound for every arm.
    "total_wall_time_seconds": 3600.0,
    "total_monetary_cost": None,
}


def matrix_launch_blockers(matrix: dict, arm_id: str | None = None) -> list[str]:
    """Return unmet launch prerequisites; arm checks cover only ``arm_id`` when given."""

    blockers: list[str] = []
    rules = matrix["atomic_estimate_configuration"]["rules"]
    types = {(rule["action_type"], rule["contract_id"]) for rule in rules}
    for needed in (("continue_actor", "continue_actor"), ("submit_current_state", "submit_current_state")):
        if needed not in types:
            blockers.append(f"no estimate rule for {needed[0]}")
    for check in matrix["public_acceptance_checks"]:
        if ("run_registered_check", check["check_id"]) not in types:
            blockers.append(f"no estimate rule for check {check['check_id']}")
    protocol = matrix.get("candidate_trial_protocol")
    for repair in matrix["fixer_candidates"]:
        contract = f"{repair['repair_id']}|{protocol}" if protocol else repair["repair_id"]
        if ("run_targeted_repair", contract) not in types:
            blockers.append(f"no estimate rule for repair {repair['repair_id']}")
    cap = matrix["shared_budget"]["per_call_output_tokens"]
    if matrix["shared_actor"]["max_output_tokens"] != cap:
        blockers.append("actor output cap differs from the shared per-call cap")
    if matrix["output_budget_contract"]["continuation_max_output_tokens"] != cap:
        blockers.append("continuation output cap differs from the shared per-call cap")
    admitted = set(matrix.get("admitted_repair_candidates", []))
    # Spec §7 (controlled injection): mechanical public repairs are admitted at selection
    # time by the isolated candidate trial, which delivers guidance only if it resolves.
    runtime_gated = {
        repair["repair_id"]
        for repair in matrix["fixer_candidates"]
        if matrix.get("repair_admission_policy") == RUNTIME_GATE_POLICY
        and repair.get("arguments", {}).get("fixer_identity") == MECHANICAL_PUBLIC_REPAIR
    }
    for arm in matrix["arms"]:
        if arm_id is not None and arm["arm_id"] != arm_id:
            continue
        gated = runtime_gated if arm.get("runtime_acceptance_candidate_trials") else set()
        missing = [rid for rid in arm["repair_ids"] if rid not in admitted | gated]
        if arm.get("runtime_acceptance_candidate_trials") and missing:
            blockers.append(
                f"arm {arm['arm_id']}: repair candidates without an isolated admission trial: "
                + ", ".join(missing)
            )
    return blockers


def build_v3(v2: dict, *, shared_budget: dict) -> dict:
    matrix = copy.deepcopy(v2)
    cap = shared_budget["per_call_output_tokens"]
    matrix["contract_version"] = "v0.2"
    matrix["shared_budget"] = dict(shared_budget)
    matrix["shared_actor"]["max_output_tokens"] = cap
    # The budget proxy replaces Pi's provider URL, so the real upstream is frozen here.
    matrix["shared_actor"]["upstream_base_url"] = "https://api.minimaxi.com/anthropic"
    matrix["output_budget_contract"]["continuation_max_output_tokens"] = cap
    matrix["output_budget_contract"]["provider_acceptance_verified"] = False
    matrix["output_budget_contract"]["provider_acceptance_note"] = (
        "Cap lowered to the source actor's 16384; the paid source "
        "smoke run used this cap for the actor through the budget proxy."
    )
    kept = {"residual-greedy", "tracedp"}
    dropped = [arm["arm_id"] for arm in v2["arms"] if arm["arm_id"] not in kept]
    arms = [
        {
            "arm_id": "no-localization",
            "interpretation": (
                "Contract v0.2 R2 lower bound: the same actor continues from the same "
                "frozen failure state with no diagnosis, check or repair; the Runtime "
                "only observes."
            ),
            "runtime_method": "managed_observation_only_v1",
            "runtime_intervention_budget": 0,
            "runtime_acceptance_candidate_trials": False,
            "repair_ids": [],
            "continuation_instruction": None,
        }
    ]
    arms += [copy.deepcopy(arm) for arm in v2["arms"] if arm["arm_id"] in kept]
    for arm in arms:
        arm.pop("launch_ready", None)
    matrix["arms"] = arms
    matrix["external_arms"] = [
        {
            "arm_id": "binary-bridge",
            "interpretation": (
                "Who&When binary search on the public source trace, then the shared "
                "method-agnostic guidance bridge, then the same actor."
            ),
            "entry_point": "not yet implemented",
        }
    ]
    matrix["dropped_arms"] = {
        arm_id: "JIT arms stay blocked: no task-disjoint source episodes (v2 explicitly_blocked)"
        for arm_id in dropped
        if arm_id != "fresh-debugger"
    }
    if "fresh-debugger" in dropped:
        matrix["dropped_arms"]["fresh-debugger"] = (
            "replaced by no-localization: v0.2 R2 defines the lower bound as the same actor "
            "continuing without a fresh-debugger prompt"
        )
    matrix.setdefault("admitted_repair_candidates", [])
    matrix["required_before_launch"] = [
        "matrix_launch_blockers(matrix) must be empty (checked by the launcher)",
        "every arm replays the same frozen prefix; the launcher compares public file bytes",
    ]
    matrix["launch_ready"] = False
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    v3 = build_v3(json.loads(args.v2.read_text()), shared_budget=DEFAULT_SHARED_BUDGET)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(v3, ensure_ascii=False, indent=2) + "\n")
    blockers = matrix_launch_blockers(v3)
    print(json.dumps({"out": str(args.out), "launch_blockers": blockers}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
