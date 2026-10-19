"""Bind an explicit two-step prediction model to a live runtime action slate."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict

from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionSlate,
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_policy import RuntimeJointResolvedChoice
from lh_platform.test_time_adaptation.runtime_joint_trace_dp import (
    RuntimeJointActionTransition,
    RuntimeJointPlanningState,
    RuntimeJointTraceDPPlanner,
    RuntimeJointTransitionOutcome,
)
from lh_platform.test_time_adaptation.runtime_time_aware_trace_dp import (
    RuntimeTimeAwareTraceDPPlanner,
    RuntimeTimedActionTransition,
    RuntimeTimedPlanningState,
)


class RuntimeBoundedTraceDPResolver:
    """Plan from supplied predictions, retaining the full model for durable replay."""

    def __init__(self, model_provider: Callable[[RuntimeJointActionSlate], Mapping | None]):
        self.model_provider = model_provider

    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        """Acquire predictions once; the policy writer persists them with the choice."""
        return self.resolve_recorded(slate, self.model_provider(slate))

    def resolve_recorded(
        self, slate: RuntimeJointActionSlate, model: Mapping | None,
    ) -> RuntimeJointResolvedChoice:
        """Recompute a stored choice without querying a changing prediction provider."""
        continuation = next(
            item for item in slate.candidates
            if item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        )
        if model is None:
            return RuntimeJointResolvedChoice(
                continuation, "bounded_dp_missing_prediction_continue_v1", {}, None,
            )
        if (
            model.get("schema_version") != "runtime_bounded_prediction_v1"
            or model.get("slate") != slate.to_dict()
            or model.get("prediction_status") != "locally_designed"
            or not model.get("prediction_source")
            or model.get("horizon") != 2
        ):
            raise ValueError("bounded DP prediction identity differs")
        states = tuple(
            RuntimeJointPlanningState(**{
                **{key: value for key, value in row.items() if key != "remaining_time_ms"},
                "missing_resource_types": tuple(row["missing_resource_types"]),
            }) for row in model["states"]
        )
        by_id = {state.state_id: state for state in states}
        times = {row["state_id"]: row["remaining_time_ms"] for row in model["states"]}
        if any(type(value) is not int or value < 0 for value in times.values()):
            raise ValueError("bounded DP state time is invalid")
        root = by_id[model["root_state_id"]]
        if (
            root.remaining_intervention_budget != slate.budget_before
            or slate.actor_time_remaining_ms is None
            or times[root.state_id] != slate.actor_time_remaining_ms
            or root.public_goal_complete
        ):
            raise ValueError("bounded DP root differs from live budget")
        candidates = {item.instance_action_id: item for item in slate.candidates}
        supported_types = {item.action_type for item in slate.candidates}
        root_actions = set()
        transitions = []
        for row in model["transitions"]:
            duration = row["duration_ms"]
            transition = RuntimeJointActionTransition(**{
                **{key: value for key, value in row.items() if key != "duration_ms"},
                "action_type": RuntimeJointActionType(row["action_type"]),
                "outcomes": tuple(RuntimeJointTransitionOutcome(**item) for item in row["outcomes"]),
            })
            source = by_id[transition.state_id]
            if type(duration) is not int or duration <= 0:
                raise ValueError("every planned action must consume positive time")
            if transition.action_type not in supported_types:
                raise ValueError("bounded DP predicts an unavailable action capability")
            if source.state_id == root.state_id:
                candidate = candidates.get(transition.instance_action_id)
                if (
                    candidate is None or candidate.action_type != transition.action_type
                    or candidate.cost_units != transition.intervention_cost_units
                    or transition.instance_action_id in root_actions
                ):
                    raise ValueError("bounded DP root action differs from slate")
                root_actions.add(transition.instance_action_id)
            if (
                transition.intervention_cost_units > source.remaining_intervention_budget
                or duration > times[source.state_id]
            ):
                # Infeasible future actions cannot contribute optimistic leaf value.
                continue
            for outcome in transition.outcomes:
                target = by_id[outcome.next_state_id]
                # Every non-submission edge includes the same Actor's next policy segment.
                actor_step_cost = int(
                    transition.action_type is not RuntimeJointActionType.SUBMIT_CURRENT_STATE
                )
                if (
                    target.remaining_intervention_budget
                    > source.remaining_intervention_budget - transition.intervention_cost_units
                    or times[target.state_id] > times[source.state_id] - duration
                    or target.remaining_actor_steps > source.remaining_actor_steps - actor_step_cost
                ):
                    raise ValueError("bounded DP prediction creates budget or time")
            transitions.append(transition)
        if root_actions != set(candidates):
            raise ValueError("bounded DP prediction omits a live candidate")
        decision = RuntimeJointTraceDPPlanner(
            states=states, transitions=tuple(transitions), horizon=2,
        ).plan(root.state_id)
        # The current executor has no implicit stop action. Expose this fallback.
        selected = candidates.get(decision.instance_action_id, continuation)
        return RuntimeJointResolvedChoice(
            selected,
            "bounded_dp_two_step_bellman_v1" if decision.instance_action_id is not None
            else "bounded_dp_implicit_stop_continue_fallback_v1",
            {action_id: asdict(value) for action_id, value in decision.q_values},
            None, dict(model),
        )


class RuntimeBoundedTraceDPV4Resolver:
    """Replay the verdict-conditioned, time-aware v4 planning model."""

    def __init__(self, model_provider: Callable[[RuntimeJointActionSlate], Mapping | None]):
        self.model_provider = model_provider

    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        """Acquire one prediction and preserve it with the selected action."""

        return self.resolve_recorded(slate, self.model_provider(slate))

    def resolve_recorded(
        self, slate: RuntimeJointActionSlate, model: Mapping | None
    ) -> RuntimeJointResolvedChoice:
        """Recompute a stored v4 choice without another provider call."""

        continuation = next(
            item
            for item in slate.candidates
            if item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        )
        if model is None:
            return RuntimeJointResolvedChoice(
                continuation,
                "bounded_dp_v4_missing_prediction_continue_v1",
                {},
                None,
            )
        if (
            model.get("schema_version") != "runtime_bounded_prediction_v2"
            or model.get("prediction_output_schema")
            != "fixed_two_step_verdict_actor_time_v4"
            or model.get("slate") != slate.to_dict()
            or model.get("prediction_status") != "locally_designed"
            or not model.get("prediction_source")
            or model.get("horizon") != 2
        ):
            raise ValueError("bounded DP v4 prediction identity differs")
        raw_states = model.get("states")
        raw_transitions = model.get("transitions")
        residual = model.get("residual_acceptance_state")
        if (
            not isinstance(raw_states, list)
            or not isinstance(raw_transitions, list)
            or not isinstance(residual, list)
        ):
            raise ValueError("bounded DP v4 prediction collections are invalid")
        states = tuple(
            RuntimeTimedPlanningState(
                **{
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "residual_predicate_states"
                    },
                    "residual_predicate_states": tuple(
                        (str(item[0]), str(item[1]))
                        for item in row["residual_predicate_states"]
                    ),
                }
            )
            for row in raw_states
        )
        by_id = {state.state_id: state for state in states}
        root = by_id.get(str(model.get("root_state_id")))
        expected_root_predicates = tuple(
            (str(item["predicate_id"]), str(item["verification_maturity"]))
            for item in residual
        )
        residual_by_id = {str(item["predicate_id"]): item for item in residual}
        if (
            root is None
            or root.remaining_intervention_budget != slate.budget_before
            or root.remaining_actor_steps != 2
            or slate.actor_time_remaining_ms is None
            or root.remaining_time_ms != slate.actor_time_remaining_ms
            or root.public_goal_complete
            or root.residual_predicate_states != expected_root_predicates
        ):
            raise ValueError("bounded DP v4 root differs from live state")
        candidates = {item.instance_action_id: item for item in slate.candidates}
        supported_types = {item.action_type for item in slate.candidates}
        root_actions: set[str] = set()
        transitions: list[RuntimeTimedActionTransition] = []
        for row in raw_transitions:
            if not isinstance(row, Mapping):
                raise ValueError("bounded DP v4 transition is invalid")
            transition = RuntimeTimedActionTransition(
                **{
                    **{key: value for key, value in row.items() if key not in {"action_type", "outcomes"}},
                    "action_type": RuntimeJointActionType(str(row["action_type"])),
                    "outcomes": tuple(
                        RuntimeJointTransitionOutcome(**item) for item in row["outcomes"]
                    ),
                }
            )
            if transition.action_type not in supported_types:
                raise ValueError("bounded DP v4 predicts an unavailable action capability")
            if transition.state_id == root.state_id:
                candidate = candidates.get(transition.instance_action_id)
                if (
                    candidate is None
                    or candidate.action_type is not transition.action_type
                    or candidate.cost_units != transition.intervention_cost_units
                    or transition.instance_action_id in root_actions
                ):
                    raise ValueError("bounded DP v4 root action differs from slate")
                if candidate.action_type is RuntimeJointActionType.RUN_REGISTERED_CHECK:
                    parts = candidate.instance_action_id.split(":", 2)
                    if len(parts) < 2 or not parts[1].isdigit():
                        raise ValueError("bounded DP v4 check identity is invalid")
                    check_index = int(parts[1])
                    durable_matches = [
                        item
                        for item in residual
                        if any(
                            binding.get("check_contract_index") == check_index
                            for binding in item.get("check_bindings", ())
                            if isinstance(binding, Mapping)
                        )
                    ]
                    if not durable_matches:
                        durable_matches = [
                            item
                            for item in residual
                            if item.get("check_contract_index") == check_index
                        ]
                    if len(durable_matches) > 1:
                        raise ValueError(
                            "bounded DP v4 check maps to multiple predicates"
                        )
                    predicate = (
                        durable_matches[0]
                        if durable_matches
                        else residual_by_id.get(f"registered-check:{check_index}")
                    )
                    if (
                        predicate is None
                        or predicate.get("task_id") != candidate.target_binding.task_id
                    ):
                        raise ValueError("bounded DP v4 check target differs")
                root_actions.add(transition.instance_action_id)
            transitions.append(transition)
        if root_actions != set(candidates):
            raise ValueError("bounded DP v4 prediction omits a live candidate")
        decision = RuntimeTimeAwareTraceDPPlanner(
            states=states,
            transitions=tuple(transitions),
            horizon=2,
        ).plan(root.state_id)
        selected = candidates.get(decision.instance_action_id, continuation)
        return RuntimeJointResolvedChoice(
            selected,
            (
                "bounded_dp_v4_verdict_actor_time_bellman_v1"
                if decision.instance_action_id is not None
                else "bounded_dp_v4_implicit_stop_continue_fallback_v1"
            ),
            {action_id: asdict(value) for action_id, value in decision.q_values},
            None,
            dict(model),
        )


class RuntimeBoundedTraceDPV6Resolver:
    """Replay a normalized v6 catalog through the audited v4 timed planner."""

    def __init__(self, model_provider: Callable[[RuntimeJointActionSlate], Mapping | None]):
        self.model_provider = model_provider

    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        """Acquire one normalized prediction and retain that exact model for replay."""

        return self.resolve_recorded(slate, self.model_provider(slate))

    def resolve_recorded(
        self, slate: RuntimeJointActionSlate, model: Mapping | None
    ) -> RuntimeJointResolvedChoice:
        """Validate v6 identity before delegating unchanged Bellman semantics."""

        continuation = next(
            item
            for item in slate.candidates
            if item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        )
        if model is None:
            return RuntimeJointResolvedChoice(
                continuation,
                "bounded_dp_v6_missing_prediction_continue_v1",
                {},
                None,
            )
        if (
            model.get("schema_version") != "runtime_bounded_prediction_v4"
            or model.get("prediction_output_schema")
            != "flat_parent_coordinates_verdict_actor_time_v6"
            or model.get("slate") != slate.to_dict()
            or model.get("prediction_status") != "locally_designed"
            or model.get("prediction_source")
            != "uncalibrated_public_state_parent_coordinate_model_v6"
            or model.get("horizon") != 2
        ):
            raise ValueError("bounded DP v6 prediction identity differs")
        catalog = model.get("candidate_catalog")
        expected_catalog = [
            {"candidate_index": index, "candidate": candidate.to_dict()}
            for index, candidate in enumerate(slate.candidates)
        ]
        if catalog != expected_catalog:
            raise ValueError("bounded DP v6 candidate catalog differs from the live slate")

        normalized_forecast = model.get("normalized_forecast")
        residual = model.get("residual_acceptance_state")
        prediction_compute_ms = model.get("prediction_compute_ms")
        if (
            not isinstance(normalized_forecast, Mapping)
            or not isinstance(residual, list)
            or type(prediction_compute_ms) is not int
            or prediction_compute_ms < 1
        ):
            raise ValueError("bounded DP v6 replay inputs are invalid")
        from lh_platform.test_time_adaptation.runtime_parent_coordinate_prediction import (
            expand_prediction_table_v6,
        )

        rebuilt = {
            **expand_prediction_table_v6(
                normalized_forecast,
                slate,
                residual_acceptance_state=residual,
                prediction_compute_ms=prediction_compute_ms,
            ),
            "schema_version": "runtime_bounded_prediction_v4",
            "slate": slate.to_dict(),
            "horizon": 2,
            "prediction_status": "locally_designed",
            "prediction_source": "uncalibrated_public_state_parent_coordinate_model_v6",
            "prediction_compute_ms": prediction_compute_ms,
        }
        if dict(model) != rebuilt:
            raise ValueError("bounded DP v6 saved model differs from normalized replay")

        # V6 changes model-facing normalization only. V4 still validates every
        # state, cost, root edge, residual predicate, and Bellman transition.
        v4_model = {
            **rebuilt,
            "schema_version": "runtime_bounded_prediction_v2",
            "prediction_output_schema": "fixed_two_step_verdict_actor_time_v4",
        }
        choice = RuntimeBoundedTraceDPV4Resolver(lambda _: None).resolve_recorded(
            slate, v4_model
        )
        return RuntimeJointResolvedChoice(
            choice.selected,
            (
                "bounded_dp_v6_normalized_verdict_actor_time_bellman_v1"
                if choice.bellman_values
                else "bounded_dp_v6_implicit_stop_continue_fallback_v1"
            ),
            choice.bellman_values,
            None,
            dict(model),
        )


class RuntimeBoundedTraceDPV7Resolver:
    """Replay compact V7 forecasts or an audited pre-call reserve decision."""

    def __init__(self, model_provider: Callable[[RuntimeJointActionSlate], Mapping | None]):
        self.model_provider = model_provider

    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        """Acquire one compact forecast without altering the live action slate."""

        return self.resolve_recorded(slate, self.model_provider(slate))

    def resolve_recorded(
        self, slate: RuntimeJointActionSlate, model: Mapping | None
    ) -> RuntimeJointResolvedChoice:
        """Validate normalized input before using unchanged V4 Bellman semantics."""

        continuation = next(
            item
            for item in slate.candidates
            if item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        )
        if model is None:
            return RuntimeJointResolvedChoice(
                continuation,
                "bounded_dp_v7_missing_prediction_continue_v1",
                {},
                None,
            )
        if model.get("schema_version") == "runtime_planner_no_call_v1":
            from lh_platform.test_time_adaptation.runtime_compact_prediction import (
                planner_time_gate_v7,
            )

            if (
                model.get("forecast_contract") != "compact_root_verdict_terminal_aggregate_v7"
                or model.get("provider_called") is not False
                or type(model.get("planner_ceiling_ms")) is not int
                or type(model.get("actor_convergence_reserve_ms")) is not int
                or type(model.get("provider_timeout_overrun_margin_ms")) is not int
            ):
                raise ValueError("bounded DP v7 no-call evidence identity differs")
            expected = planner_time_gate_v7(
                slate,
                planner_ceiling_ms=int(model["planner_ceiling_ms"]),
                actor_convergence_reserve_ms=int(
                    model["actor_convergence_reserve_ms"]
                ),
                provider_timeout_overrun_margin_ms=int(
                    model["provider_timeout_overrun_margin_ms"]
                ),
            )
            if expected is None or dict(model) != expected:
                raise ValueError("bounded DP v7 no-call evidence differs from live time")
            return RuntimeJointResolvedChoice(
                continuation,
                "bounded_dp_v7_precall_reserve_gate_continue_v1",
                {},
                None,
                dict(model),
            )
        if (
            model.get("schema_version") != "runtime_bounded_prediction_v6"
            or model.get("prediction_output_schema")
            != "compact_root_verdict_terminal_aggregate_v7"
            or model.get("slate") != slate.to_dict()
            or model.get("prediction_status") != "locally_designed"
            or model.get("prediction_source")
            != "uncalibrated_public_state_compact_terminal_aggregate_model_v7"
            or model.get("horizon") != 2
        ):
            raise ValueError("bounded DP v7 prediction identity differs")
        normalized_forecast = model.get("normalized_forecast")
        residual = model.get("residual_acceptance_state")
        prediction_compute_ms = model.get("prediction_compute_ms")
        if (
            not isinstance(normalized_forecast, Mapping)
            or not isinstance(residual, list)
            or type(prediction_compute_ms) is not int
            or prediction_compute_ms < 1
        ):
            raise ValueError("bounded DP v7 replay inputs are invalid")
        from lh_platform.test_time_adaptation.runtime_compact_prediction import (
            expand_prediction_table_v7,
        )

        rebuilt = {
            **expand_prediction_table_v7(
                normalized_forecast,
                slate,
                residual_acceptance_state=residual,
                prediction_compute_ms=prediction_compute_ms,
            ),
            "schema_version": "runtime_bounded_prediction_v6",
            "slate": slate.to_dict(),
            "horizon": 2,
            "prediction_status": "locally_designed",
            "prediction_source": "uncalibrated_public_state_compact_terminal_aggregate_model_v7",
            "prediction_compute_ms": prediction_compute_ms,
        }
        if dict(model) != rebuilt:
            raise ValueError("bounded DP v7 saved model differs from compact replay")
        v4_model = {
            **rebuilt,
            "schema_version": "runtime_bounded_prediction_v2",
            "prediction_output_schema": "fixed_two_step_verdict_actor_time_v4",
        }
        choice = RuntimeBoundedTraceDPV4Resolver(lambda _: None).resolve_recorded(
            slate, v4_model
        )
        return RuntimeJointResolvedChoice(
            choice.selected,
            (
                "bounded_dp_v7_compact_root_verdict_terminal_aggregate_bellman_v1"
                if choice.bellman_values
                else "bounded_dp_v7_implicit_stop_continue_fallback_v1"
            ),
            choice.bellman_values,
            None,
            dict(model),
        )


class RuntimeEventTraceDPV8Resolver:
    """Replay V8 admission evidence and delegate admitted forecasts to V7."""

    def __init__(self, model_provider: Callable[[RuntimeJointActionSlate], Mapping | None]):
        self.model_provider = model_provider

    def resolve(self, slate: RuntimeJointActionSlate) -> RuntimeJointResolvedChoice:
        """Acquire one event-gated model and preserve its exact evidence."""

        return self.resolve_recorded(slate, self.model_provider(slate))

    def resolve_recorded(
        self, slate: RuntimeJointActionSlate, model: Mapping | None
    ) -> RuntimeJointResolvedChoice:
        """Validate every V8 fallback separately before reusing V7 Bellman."""

        from lh_platform.test_time_adaptation.runtime_event_admission import (
            bind_event_admission_to_resource_gate_v8,
            planner_cumulative_budget_gate_v8,
            planner_time_reserve_gate_v8,
            validate_planner_event_admission_v8,
        )

        continuation = next(
            item
            for item in slate.candidates
            if item.action_type is RuntimeJointActionType.CONTINUE_ACTOR
        )
        if model is None:
            raise ValueError("v8 requires explicit planning or no-call evidence")
        schema = model.get("schema_version")
        if schema == "runtime_planner_event_admission_v1":
            evidence = validate_planner_event_admission_v8(slate, model)
            reason = evidence["reason"]
            if reason not in {
                "single_candidate_deterministic",
                "event_gate_suppressed",
            }:
                raise ValueError("v8 event no-call reason differs")
            return RuntimeJointResolvedChoice(
                continuation,
                (
                    "bounded_dp_v8_single_candidate_deterministic_continue_v1"
                    if reason == "single_candidate_deterministic"
                    else "bounded_dp_v8_event_gate_suppressed_continue_v1"
                ),
                {},
                None,
                dict(model),
            )
        if schema in {
            "runtime_planner_time_reserve_no_call_v1",
            "runtime_planner_time_reserve_no_call_v2",
        }:
            expected = planner_time_reserve_gate_v8(
                slate,
                planner_ceiling_ms=int(model.get("planner_ceiling_ms", 0)),
                actor_convergence_reserve_ms=int(
                    model.get("actor_convergence_reserve_ms", 0)
                ),
                provider_timeout_overrun_margin_ms=int(
                    model.get("provider_timeout_overrun_margin_ms", -1)
                ),
            )
            if schema == "runtime_planner_time_reserve_no_call_v2" and expected is not None:
                admission = model.get("event_admission")
                if not isinstance(admission, Mapping):
                    raise ValueError("v8 pre-call reserve lacks event admission")
                expected = bind_event_admission_to_resource_gate_v8(
                    slate, expected, admission
                )
            if expected is None or dict(model) != expected:
                raise ValueError("v8 pre-call reserve evidence differs")
            return RuntimeJointResolvedChoice(
                continuation,
                "bounded_dp_v8_precall_reserve_continue_v1",
                {},
                None,
                dict(model),
            )
        if schema in {
            "runtime_planner_cumulative_budget_no_call_v1",
            "runtime_planner_cumulative_budget_no_call_v2",
        }:
            expected = planner_cumulative_budget_gate_v8(
                slate,
                cumulative_planner_elapsed_ms=int(
                    model.get("cumulative_planner_elapsed_ms", -1)
                ),
                planner_ceiling_ms=int(model.get("planner_ceiling_ms", 0)),
                provider_timeout_overrun_margin_ms=int(
                    model.get("provider_timeout_overrun_margin_ms", -1)
                ),
                cumulative_planner_budget_ms=int(
                    model.get("cumulative_planner_budget_ms", 0)
                ),
            )
            if (
                schema == "runtime_planner_cumulative_budget_no_call_v2"
                and expected is not None
            ):
                admission = model.get("event_admission")
                if not isinstance(admission, Mapping):
                    raise ValueError("v8 cumulative budget gate lacks event admission")
                expected = bind_event_admission_to_resource_gate_v8(
                    slate, expected, admission
                )
            if expected is None or dict(model) != expected:
                raise ValueError("v8 cumulative planner budget evidence differs")
            return RuntimeJointResolvedChoice(
                continuation,
                "bounded_dp_v8_cumulative_planner_budget_exhausted_continue_v1",
                {},
                None,
                dict(model),
            )
        if schema == "runtime_planner_prediction_failure_v1":
            admission = model.get("event_admission")
            if (
                model.get("slate") != slate.to_dict()
                or model.get("provider_called") is not True
                or model.get("failure_kind")
                not in {
                    "contract_invalid",
                    "model_abstained",
                    "provider_error",
                    "response_parse_error",
                }
                or not isinstance(admission, Mapping)
                or validate_planner_event_admission_v8(slate, admission)["decision"]
                != "call_provider"
            ):
                raise ValueError("v8 provider failure evidence differs")
            return RuntimeJointResolvedChoice(
                continuation,
                (
                    "bounded_dp_v8_contract_invalid_prediction_fallback_continue_v1"
                    if model["failure_kind"] == "contract_invalid"
                    else "bounded_dp_v8_provider_prediction_failure_continue_v1"
                ),
                {},
                None,
                dict(model),
            )
        if (
            schema != "runtime_bounded_prediction_v7"
            or model.get("prediction_output_schema")
            != "compact_root_verdict_terminal_aggregate_v7"
            or model.get("slate") != slate.to_dict()
            or model.get("prediction_status") != "locally_designed"
            or model.get("prediction_source")
            != "uncalibrated_public_state_compact_terminal_aggregate_model_v7"
            or model.get("horizon") != 2
        ):
            raise ValueError("bounded DP v8 prediction identity differs")
        admission = model.get("event_admission")
        if (
            not isinstance(admission, Mapping)
            or validate_planner_event_admission_v8(slate, admission)["decision"]
            != "call_provider"
        ):
            raise ValueError("bounded DP v8 event admission differs")
        v7_model = {
            key: value for key, value in model.items() if key != "event_admission"
        }
        v7_model["schema_version"] = "runtime_bounded_prediction_v6"
        choice = RuntimeBoundedTraceDPV7Resolver(lambda _: None).resolve_recorded(
            slate, v7_model
        )
        return RuntimeJointResolvedChoice(
            choice.selected,
            (
                "bounded_dp_v8_event_compact_bellman_v1"
                if choice.bellman_values
                else "bounded_dp_v8_implicit_stop_continue_fallback_v1"
            ),
            choice.bellman_values,
            None,
            dict(model),
        )


__all__ = [
    "RuntimeBoundedTraceDPResolver",
    "RuntimeBoundedTraceDPV4Resolver",
    "RuntimeBoundedTraceDPV6Resolver",
    "RuntimeBoundedTraceDPV7Resolver",
    "RuntimeEventTraceDPV8Resolver",
]
