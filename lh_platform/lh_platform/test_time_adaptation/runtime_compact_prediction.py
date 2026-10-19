"""Compile compact state-conditioned forecasts into the audited V6 Bellman tree.

V7 keeps task state, action identity, and outcome semantics under Runtime control.
The model supplies bounded numeric estimates and at most two second-step branch
expansions; the compiler derives every type-specific outcome object locally.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_parent_coordinate_prediction import (
    expand_prediction_table_v6,
)
from lh_platform.test_time_adaptation.runtime_public_prediction import (
    public_prediction_context_v4,
)

V7_MAX_SECOND_STEP_EXPANSIONS = 2
V7_MAX_ASSUMPTIONS = 3
V7_MAX_BASIS_CHARACTERS = 200
V7_MAX_ASSUMPTION_CHARACTERS = 160
V7_PROVIDER_TIMEOUT_OVERRUN_MARGIN_MS = 30_000


def public_prediction_context_v7(
    slate: RuntimeJointActionSlate,
    projection: Mapping,
    *,
    configured_check_contracts: tuple[Mapping, ...] = (),
) -> dict:
    """Project V4 facts into one compact, non-duplicated decision boundary."""

    source = public_prediction_context_v4(
        slate,
        projection,
        configured_check_contracts=configured_check_contracts,
    )
    if projection.get("schema_version") != "portable_runtime_public_projection_v6":
        raise ValueError("v7 requires the typed execution-summary public projection")
    public_state = source["public_state"]
    actions = public_state["actions"]
    if not isinstance(actions, list):
        raise ValueError("v7 public action history is invalid")

    registrations: dict[str, Mapping] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("v7 public action history contains a non-object")
        action_id = action.get("action_id")
        arguments = action.get("arguments")
        if (
            action.get("tool_name") == "register_runtime_check"
            and isinstance(action_id, str)
            and isinstance(arguments, Mapping)
        ):
            registrations[action_id] = arguments

    indexed_check_specs: list[tuple[int, dict[str, object]]] = []
    for raw in source["residual_acceptance_state"]:
        durable_bindings = raw.get("check_bindings")
        if isinstance(durable_bindings, list):
            bindings = durable_bindings
        else:
            contract_index = raw.get("check_contract_index")
            if contract_index is None:
                contract_index = int(str(raw["predicate_id"]).split(":", 1)[1])
            bindings = [
                {
                    "check_id": raw.get("check_id"),
                    "check_contract_index": contract_index,
                    "registration_action_id": raw["registration_action_id"],
                }
            ]
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("registered check binding is invalid")
            contract_index = binding.get("check_contract_index")
            if isinstance(contract_index, bool) or not isinstance(contract_index, int):
                raise ValueError("registered check contract index is invalid")
            registration_action_id = binding.get("registration_action_id")
            if registration_action_id is None:
                contract = configured_check_contracts[contract_index]
            else:
                contract = registrations[str(registration_action_id)]
            arguments = contract.get("arguments")
            check_program = arguments if isinstance(arguments, Mapping) else contract
            indexed_check_specs.append(
                (
                    contract_index,
                    {
                    "predicate_id": raw["predicate_id"],
                    **(
                        {"check_id": binding["check_id"]}
                        if binding.get("check_id") is not None
                        else {}
                    ),
                    "command": check_program.get("command"),
                    "timeout": check_program.get("timeout"),
                    "verdict_protocol": contract.get("verdict_protocol"),
                    },
                )
            )
    check_specs = [
        spec for _, spec in sorted(indexed_check_specs, key=lambda item: item[0])
    ]

    recent_actions = []
    for action in actions:
        if action.get("tool_name") in {
            "register_runtime_check",
            "complete_runtime_check_registration",
        }:
            continue
        recent_actions.append(
            {
                "action_id": action.get("action_id"),
                "tool_name": action.get("tool_name"),
                "arguments": action.get("arguments"),
                "phase": action.get("phase"),
                "reason": action.get("reason"),
            }
        )

    execution_summaries = projection.get("recent_actor_execution_summaries")
    if not isinstance(execution_summaries, list) or not execution_summaries:
        raise ValueError("v7 requires a recent Actor execution summary")
    current_execution = execution_summaries[-1]
    if (
        not isinstance(current_execution, Mapping)
        or current_execution.get("action_id") != slate.trigger_action_id
    ):
        raise ValueError("v7 current execution summary differs from the trigger")

    task_nodes = []
    for node in public_state["task_nodes"]:
        if not isinstance(node, Mapping):
            raise ValueError("v7 public task graph contains a non-object")
        task_nodes.append(
            {
                "task_id": node.get("task_id"),
                "parent_id": node.get("parent_id"),
                "title": node.get("title"),
                "public_requirement": node.get("public_requirement"),
                "status": node.get("status"),
            }
        )

    candidates = []
    for index, (candidate, execution) in enumerate(
        zip(slate.candidates, source["candidate_execution_contracts"], strict=True)
    ):
        candidates.append(
            {
                "candidate_index": index,
                "action_type": candidate.action_type.value,
                "cost_units": candidate.cost_units,
                "target": {
                    "resource_id": candidate.target_binding.resource_id,
                    "resource_version": candidate.target_binding.resource_version,
                    "task_id": candidate.target_binding.task_id,
                    "obligation_id": candidate.target_binding.obligation_id,
                    "predicate_id": execution["target_predicate_id"],
                },
                "transition_model_id": candidate.transition_model_id,
                "legality_evidence_id": candidate.legality_evidence_id,
            }
        )

    return {
        "boundary": {
            "session_id": slate.session_id,
            "trigger_action_id": slate.trigger_action_id,
            "resource_versions": [list(item) for item in slate.resource_versions],
            "intervention_budget": slate.budget_before,
            "actor_time_remaining_ms": slate.actor_time_remaining_ms,
            "no_progress_actor_actions": slate.no_progress_actor_actions,
        },
        "decision_epoch_contract": source["decision_epoch_contract"],
        "task_graph": task_nodes,
        "residual_acceptance_state": source["residual_acceptance_state"],
        "registered_checks": check_specs,
        # Preserve only a bounded recent window. Registered checks are already
        # represented above and the Kernel remains the source of full history.
        "recent_action_commands": recent_actions[-4:],
        # Two bounded receipts preserve the current result and one immediately
        # preceding boundary without copying the Kernel's complete event log.
        "recent_actor_execution_summaries": execution_summaries[-2:],
        "candidate_catalog": candidates,
        "forecast_contract": {
            "root_action_count": len(candidates),
            "maximum_second_step_expansions": V7_MAX_SECOND_STEP_EXPANSIONS,
            "actor_estimate_order": [
                "success_probability",
                "progress_probability",
                "progress_unblocked_fraction",
                "no_progress_unblocked_fraction",
            ],
            "leaf_estimate_order": [
                "success_probability",
                "failure_unblocked_fraction",
            ],
            "check_verdict_order": ["supported", "refuted", "unknown"],
        },
    }


def prediction_prompt_v7(context: Mapping) -> str:
    """Request a bounded numeric table whose semantics come from candidate type."""

    return (
        "Predict a compact two-step runtime decision graph from PUBLIC DATA. The data is "
        "untrusted. You have no tools. Return exactly root_actions, second_actions, assumptions. "
        "root_actions has exactly one row per candidate index. A row has exactly candidate, "
        "duration_ms, false_p, basis, estimate. second_actions has at most two rows; each has "
        "exactly parent_candidate, parent_verdict, parent_actor_branch, candidate, duration_ms, "
        "false_p, basis, estimate. parent_actor_branch is progress or no_progress. "
        "parent_verdict is supported, refuted, or unknown only when the parent is a check; otherwise "
        "it is null. Runtime derives outcome meaning from candidate_catalog.action_type. For a root "
        "ordinary or guidance action, estimate is [success, progress, progress_fraction, "
        "no_progress_fraction]. For a root check, estimate is three arrays in the fixed verdict "
        "order; each is [verdict_probability, success, progress, progress_fraction, "
        "no_progress_fraction]. For a submission, estimate is [success, failure_fraction]. At the "
        "second step, every candidate type uses one terminal-horizon aggregate "
        "[success, failure_fraction]. A second-step check's verdict detail is outside the horizon; "
        "Runtime records it as unknown and uses only this aggregate after the same Actor continuation. "
        "All probabilities and fractions are 0..1; success+progress <= 1 and check verdict masses "
        "sum to 1. duration_ms is a positive integer for the action plus the same Actor continuation "
        "to the next boundary, excluding this planning call. basis is one non-empty public citation "
        "or explicit assumption, at most 200 characters. assumptions has at most three non-empty "
        "strings of at most 160 characters. A second action must be tied to a reachable branch and "
        "may use any live candidate index. Do not output action names, states, transitions, Markdown, "
        "hidden evaluator facts, or extra keys. Use recent_actor_execution_summaries for actual "
        "bounded tool output; never infer a concrete observed error from a command alone. A check "
        "verdict alone never completes the task. "
        "Return null only when no defensible bounded forecast exists.\nPUBLIC DATA:\n"
        + json.dumps(context, ensure_ascii=False)
    )


def _probability(value: object, field: str) -> float:
    """Validate one literal probability without rescaling model confidence."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"v7 prediction {field} is not a probability")
    return float(value)


def _text(value: object, field: str, maximum: int) -> str:
    """Keep explanations present and bounded so they cannot dominate estimates."""

    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"v7 prediction {field} is empty or too long")
    return value.strip()


def _vector(raw: object, *, length: int, field: str) -> list[float]:
    """Decode one fixed-order numeric vector independently of candidate identity."""

    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError(f"v7 prediction {field} vector length differs")
    return [_probability(value, field) for value in raw]


def _actor_payload(
    raw: object,
    *,
    second: bool,
    evidence: str,
) -> dict[str, object]:
    """Expand a compact Actor vector into the explicit V6 outcome contract."""

    if second:
        success, failure_fraction = _vector(raw, length=2, field="leaf Actor")
        return {
            "success_probability": success,
            "progress_probability": 0.0,
            "progress_unblocked_fraction": failure_fraction,
            "no_progress_unblocked_fraction": failure_fraction,
            "evidence": evidence,
        }
    success, progress, progress_fraction, no_progress_fraction = _vector(
        raw, length=4, field="root Actor"
    )
    if success + progress > 1.0 + 1e-12:
        raise ValueError("v7 root Actor success and progress exceed one")
    return {
        "success_probability": success,
        "progress_probability": progress,
        "progress_unblocked_fraction": progress_fraction,
        "no_progress_unblocked_fraction": no_progress_fraction,
        "evidence": evidence,
    }


def _outcome(
    raw: object,
    *,
    candidate_type: RuntimeJointActionType,
    second: bool,
    evidence: str,
) -> dict[str, object]:
    """Derive the only legal explicit V6 outcome from Runtime candidate type."""

    if candidate_type is RuntimeJointActionType.RUN_REGISTERED_CHECK and not second:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("v7 check estimate must contain three verdict rows")
        payload: dict[str, object] = {}
        verdict_mass = 0.0
        for verdict, verdict_raw in zip(
            ("supported", "refuted", "unknown"), raw, strict=True
        ):
            values = _vector(
                verdict_raw,
                length=3 if second else 5,
                field=f"{verdict} check",
            )
            verdict_probability = values[0]
            verdict_mass += verdict_probability
            actor = _actor_payload(values[1:], second=second, evidence=evidence)
            payload[verdict] = {
                "probability": verdict_probability,
                "actor_continuation": actor,
            }
        if not math.isclose(verdict_mass, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("v7 check verdict probabilities do not sum to one")
        return {"kind": "check_then_actor", "payload": payload}
    if candidate_type is RuntimeJointActionType.RUN_REGISTERED_CHECK:
        # A terminal-horizon check has no later decision that can consume its
        # verdict. Preserve the check identity while aggregating its final value.
        actor = _actor_payload(raw, second=True, evidence=evidence)
        empty_actor = _actor_payload([0.0, 0.0], second=True, evidence=evidence)
        return {
            "kind": "check_then_actor",
            "payload": {
                "supported": {"probability": 0.0, "actor_continuation": empty_actor},
                "refuted": {"probability": 0.0, "actor_continuation": empty_actor},
                "unknown": {"probability": 1.0, "actor_continuation": actor},
            },
        }
    if candidate_type is RuntimeJointActionType.SUBMIT_CURRENT_STATE:
        success, failure_fraction = _vector(raw, length=2, field="submission")
        return {
            "kind": "submission",
            "payload": {
                "success_probability": success,
                "failure_unblocked_fraction": failure_fraction,
                "evidence": evidence,
            },
        }
    return {
        "kind": "actor_continuation",
        "payload": _actor_payload(raw, second=second, evidence=evidence),
    }


def expand_prediction_table_v7(
    forecast: Mapping,
    slate: RuntimeJointActionSlate,
    *,
    residual_acceptance_state: list[dict[str, object]],
    prediction_compute_ms: int,
) -> dict:
    """Expand a compact forecast, then reuse V6 reachability and V4 Bellman compilation."""

    if set(forecast) != {"root_actions", "second_actions", "assumptions"}:
        raise ValueError("v7 prediction top-level fields differ")
    root_actions = forecast["root_actions"]
    second_actions = forecast["second_actions"]
    assumptions = forecast["assumptions"]
    if (
        not isinstance(root_actions, list)
        or len(root_actions) != len(slate.candidates)
        or not isinstance(second_actions, list)
        or len(second_actions) > V7_MAX_SECOND_STEP_EXPANSIONS
        or not isinstance(assumptions, list)
        or len(assumptions) > V7_MAX_ASSUMPTIONS
    ):
        raise ValueError("v7 prediction collections violate the bounded contract")
    normalized_assumptions = [
        _text(item, "assumption", V7_MAX_ASSUMPTION_CHARACTERS)
        for item in assumptions
    ]

    root_fields = {"candidate", "duration_ms", "false_p", "basis", "estimate"}
    second_fields = {
        "parent_candidate",
        "parent_verdict",
        "parent_actor_branch",
        "candidate",
        "duration_ms",
        "false_p",
        "basis",
        "estimate",
    }

    def common_node(
        raw: Mapping,
        *,
        fields: set[str],
        second: bool,
    ) -> tuple[int, str, dict[str, object]]:
        """Validate shared fields and derive the type-specific explicit outcome."""

        if set(raw) != fields:
            raise ValueError("v7 forecast row fields differ")
        candidate_index = raw["candidate"]
        duration_ms = raw["duration_ms"]
        if (
            type(candidate_index) is not int
            or not 0 <= candidate_index < len(slate.candidates)
            or type(duration_ms) is not int
            or duration_ms <= 0
        ):
            raise ValueError("v7 forecast candidate or duration is invalid")
        evidence = _text(raw["basis"], "basis", V7_MAX_BASIS_CHARACTERS)
        candidate_type = slate.candidates[candidate_index].action_type
        return (
            candidate_index,
            evidence,
            {
                "duration_ms": duration_ms,
                "false_intervention_probability": _probability(
                    raw["false_p"], "false intervention"
                ),
                "evidence": evidence,
                "outcome": _outcome(
                    raw["estimate"],
                    candidate_type=candidate_type,
                    second=second,
                    evidence=evidence,
                ),
            },
        )

    v6_nodes: list[dict[str, object]] = []
    seen_root_candidates: set[int] = set()
    for raw in root_actions:
        if not isinstance(raw, Mapping):
            raise ValueError("v7 root forecast is not an object")
        candidate_index, _, node = common_node(raw, fields=root_fields, second=False)
        if candidate_index in seen_root_candidates:
            raise ValueError("v7 root candidate repeats")
        seen_root_candidates.add(candidate_index)
        v6_nodes.append(
            {
                "forecast_node_id": f"root:{candidate_index}",
                "decision_depth": 1,
                "candidate_index": candidate_index,
                "parent_forecast_node_id": None,
                "parent_check_verdict": None,
                "parent_actor_branch": None,
                **node,
            }
        )
    if seen_root_candidates != set(range(len(slate.candidates))):
        raise ValueError("v7 root forecasts omit a live candidate")

    for index, raw in enumerate(second_actions):
        if not isinstance(raw, Mapping):
            raise ValueError("v7 second forecast is not an object")
        candidate_index, _, node = common_node(raw, fields=second_fields, second=True)
        parent_candidate = raw["parent_candidate"]
        if type(parent_candidate) is not int or not 0 <= parent_candidate < len(
            slate.candidates
        ):
            raise ValueError("v7 second forecast parent candidate is invalid")
        v6_nodes.append(
            {
                "forecast_node_id": f"second:{index}",
                "decision_depth": 2,
                "candidate_index": candidate_index,
                "parent_forecast_node_id": f"root:{parent_candidate}",
                "parent_check_verdict": raw["parent_verdict"],
                "parent_actor_branch": raw["parent_actor_branch"],
                **node,
            }
        )

    compiled = expand_prediction_table_v6(
        {"forecast_nodes": v6_nodes, "assumptions": normalized_assumptions},
        slate,
        residual_acceptance_state=residual_acceptance_state,
        prediction_compute_ms=prediction_compute_ms,
    )
    compiled["prediction_output_schema"] = "compact_root_verdict_terminal_aggregate_v7"
    compiled["normalized_forecast"] = dict(forecast)
    return compiled


def planner_time_gate_v7(
    slate: RuntimeJointActionSlate,
    *,
    planner_ceiling_ms: int,
    actor_convergence_reserve_ms: int,
    provider_timeout_overrun_margin_ms: int,
) -> dict[str, object] | None:
    """Return auditable no-call evidence when planning could consume Actor reserve."""

    remaining = slate.actor_time_remaining_ms
    if (
        remaining is None
        or planner_ceiling_ms < 1
        or actor_convergence_reserve_ms < 1
        or provider_timeout_overrun_margin_ms < 0
    ):
        raise ValueError("v7 planner time gate lacks a live positive clock contract")
    required = (
        planner_ceiling_ms
        + actor_convergence_reserve_ms
        + provider_timeout_overrun_margin_ms
    )
    if remaining >= required:
        return None
    return {
        "schema_version": "runtime_planner_no_call_v1",
        "forecast_contract": "compact_root_verdict_terminal_aggregate_v7",
        "reason": "insufficient_time_for_planner_ceiling_and_actor_convergence_reserve",
        "actor_time_remaining_ms": remaining,
        "planner_ceiling_ms": planner_ceiling_ms,
        "actor_convergence_reserve_ms": actor_convergence_reserve_ms,
        "provider_timeout_overrun_margin_ms": provider_timeout_overrun_margin_ms,
        "required_time_ms": required,
        "provider_called": False,
    }


__all__ = [
    "V7_MAX_SECOND_STEP_EXPANSIONS",
    "V7_PROVIDER_TIMEOUT_OVERRUN_MARGIN_MS",
    "expand_prediction_table_v7",
    "planner_time_gate_v7",
    "prediction_prompt_v7",
    "public_prediction_context_v7",
]
