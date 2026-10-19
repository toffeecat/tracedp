"""Price delivery separately from a conditional Actor-and-recheck option.

This locally designed extension preserves the legacy graph/replay path. Named
future check results are separate forecasts, never evidence of live completion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from lh_platform.benchmarks.terminal_runtime_check import (
    runtime_check_configuration_payload,
)
from lh_platform.test_time_adaptation.runtime_joint_action_slate import (
    RuntimeJointActionType,
)
from lh_platform.test_time_adaptation.runtime_joint_trace_dp import RuntimeJointTransitionOutcome
from lh_platform.test_time_adaptation.runtime_owned_option_dp import (
    RuntimeAtomicActionEstimate,
    RuntimeOwnedOptionDPResolver,
    RuntimeOwnedOptionProjection,
    _compile_model,
    _CompiledOptionModel,
    _dependency_progress_fraction,
    _immediate_value,
    _outcome_node,
    _ProjectedNode,
    _state_payload,
)
from lh_platform.test_time_adaptation.runtime_owned_option_live import (
    RuntimeFrozenAtomicEstimateConfiguration,
    RuntimeOwnedOptionLiveResolver,
    _action_contract_identity,
)
from lh_platform.test_time_adaptation.runtime_time_aware_trace_dp import (
    RuntimeTimedActionTransition,
)

_VERDICTS = {"supported", "refuted", "unknown"}
_SCOPE = "main_actor_and_registered_checks_given_admitted_guidance"


@dataclass(frozen=True)
class GuidedCheckOutcome:
    """Describe a joint result of named public checks after the main Actor runs."""

    outcome_id: str
    probability: float
    check_verdicts: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if (
            not self.outcome_id
            or type(self.probability) not in (int, float)
            or not math.isfinite(self.probability)
            or not 0 <= self.probability <= 1
            or not self.check_verdicts
            or len({k for k, _ in self.check_verdicts}) != len(self.check_verdicts)
            or any(not k or v not in _VERDICTS for k, v in self.check_verdicts)
        ):
            raise ValueError("guided main-check outcome is invalid")

    def to_dict(self):
        """Keep forecast results explicitly keyed by the registered check identity."""
        return {
            "outcome_id": self.outcome_id,
            "probability": self.probability,
            "check_verdicts": [{"check_id": k, "verdict": v} for k, v in self.check_verdicts],
        }


@dataclass(frozen=True)
class GuidedCompletionRule:
    """Bind a conditional main-execution forecast to one exact guidance contract."""

    rule_id: str
    repair_contract_id: str
    task_id: str
    target_predicate: str
    actor_duration_ms: int
    recheck_duration_ms: int
    outcomes: tuple[GuidedCheckOutcome, ...]
    provenance_id: str
    provenance_detail: str

    def __post_init__(self):
        if (
            any(
                not s
                for s in (
                    self.rule_id,
                    self.repair_contract_id,
                    self.task_id,
                    self.target_predicate,
                    self.provenance_id,
                    self.provenance_detail,
                )
            )
            or type(self.actor_duration_ms) is not int
            or self.actor_duration_ms < 1
            or type(self.recheck_duration_ms) is not int
            or self.recheck_duration_ms < 1
            or not self.outcomes
            or len({o.outcome_id for o in self.outcomes}) != len(self.outcomes)
            or not math.isclose(
                sum(o.probability for o in self.outcomes), 1, abs_tol=1e-12, rel_tol=0
            )
        ):
            raise ValueError("guided main-execution rule is invalid")
        keys = {k for k, _ in self.outcomes[0].check_verdicts}
        if any({k for k, _ in o.check_verdicts} != keys for o in self.outcomes):
            raise ValueError("guided outcomes cover different check portfolios")

    @property
    def check_ids(self):
        """Expose the common portfolio without inferring it from outcome names."""
        return frozenset(k for k, _ in self.outcomes[0].check_verdicts)

    def to_dict(self):
        """Persist the scoped forecast and its two distinct duration components."""
        return {
            "rule_id": self.rule_id,
            "repair_contract_id": self.repair_contract_id,
            "task_id": self.task_id,
            "target_predicate": self.target_predicate,
            "actor_duration_ms": self.actor_duration_ms,
            "recheck_duration_ms": self.recheck_duration_ms,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "provenance_id": self.provenance_id,
            "provenance_detail": self.provenance_detail,
        }


@dataclass(frozen=True)
class GuidedCompletionConfiguration:
    """Keep conditional main forecasts separate from clone-admission estimates."""

    configuration_id: str
    actor_cell_id: str
    rules: tuple[GuidedCompletionRule, ...]
    schema_version = "runtime_guided_completion_configuration_v1"

    def __post_init__(self):
        keys = [(r.repair_contract_id, r.task_id, r.target_predicate) for r in self.rules]
        if not self.configuration_id or not self.actor_cell_id or len(keys) != len(set(keys)):
            raise ValueError("guided completion configuration identity is invalid")

    def to_dict(self):
        """Label these forecasts as unverified development priors, never calibration gold."""
        return {
            "schema_version": self.schema_version,
            "configuration_id": self.configuration_id,
            "actor_cell_id": self.actor_cell_id,
            "prediction_scope": _SCOPE,
            "implementation_status": "implementation_unverified",
            "estimate_status": "development_prior_unverified",
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, data):
        """Load the explicit scope; a legacy clone configuration is not interchangeable."""
        if (
            data.get("schema_version") != cls.schema_version
            or data.get("prediction_scope") != _SCOPE
            or data.get("implementation_status") != "implementation_unverified"
            or data.get("estimate_status") != "development_prior_unverified"
        ):
            raise ValueError("guided completion configuration scope differs")
        rules = tuple(
            GuidedCompletionRule(
                **{
                    k: r[k]
                    for k in (
                        "rule_id",
                        "repair_contract_id",
                        "task_id",
                        "target_predicate",
                        "actor_duration_ms",
                        "recheck_duration_ms",
                        "provenance_id",
                        "provenance_detail",
                    )
                },
                outcomes=tuple(
                    GuidedCheckOutcome(
                        o["outcome_id"],
                        o["probability"],
                        tuple((v["check_id"], v["verdict"]) for v in o["check_verdicts"]),
                    )
                    for o in r["outcomes"]
                ),
            )
            for r in data["rules"]
        )
        result = cls(data["configuration_id"], data["actor_cell_id"], rules)
        if result.to_dict() != data:
            raise ValueError("guided completion configuration fields differ")
        return result


class RuntimeGuidancePhaseDPResolver(RuntimeOwnedOptionDPResolver):
    """Extend existing option branches with a priced, conditional main verification phase."""

    schema_version = "runtime_guidance_phase_option_model_v1"
    selector_identity = "runtime_guidance_phase_option_dp_v1"

    def __init__(self, configuration, configured_checks):
        self.configuration = configuration
        self.configured_checks = configured_checks

    def _bind(self, action, projection):
        """Require the full public portfolio because main Actor writes are not scope-confined."""
        contract_id, target = _action_contract_identity(action, self.configured_checks)
        rules = [
            r
            for r in self.configuration.rules
            if r.repair_contract_id == contract_id
            and r.target_predicate == target
            and r.task_id == action.target_binding.task_id
        ]
        if len(rules) != 1:
            raise ValueError("guidance lacks an exact conditional main-execution rule")
        rule = rules[0]
        affected = self.configured_checks
        if rule.check_ids != {c.check_id for c in affected}:
            raise ValueError("guided completion must cover every public registered check")
        mapping = {}
        for check in affected:
            predicates = [
                p
                for p in projection.acceptance_predicates
                if p.task_id == check.task_id and p.proposition == check.proposition
            ]
            if len(predicates) != 1:
                raise ValueError("guided check does not bind one public predicate")
            mapping[check.check_id] = predicates[0].predicate_id
        if not any(
            c.task_id == rule.task_id and c.proposition == rule.target_predicate for c in affected
        ):
            raise ValueError("guided completion portfolio omits the target predicate")
        return rule, mapping

    def _compile(self, slate, projection, estimates, horizon):
        """Repair delivery leaves and replace their futures with explicitly priced phases."""
        if type(horizon) is not int or horizon not in (1, 2, 3):
            raise ValueError("guidance phase planning horizon is unsupported")
        original = _compile_model(
            slate=slate, projection=projection, estimates=estimates, horizon=min(horizon, 2)
        )
        actions = {
            a.instance_action_id: a for a in (*slate.candidates, *slate.conditional_future_options)
        }
        bound = {
            k: self._bind(a, projection)
            for k, a in actions.items()
            if a.action_type is RuntimeJointActionType.RUN_TARGETED_REPAIR
            and a.execution_contract.delivery_only
        }
        old_states = {s.state_id: s for s in original.states}
        outgoing = {}
        row_by_key = {}
        for t, row in zip(original.transitions, original.transition_rows, strict=True):
            outgoing.setdefault(t.state_id, []).append(t)
            row_by_key[t.state_id, t.instance_action_id] = row
        states, transitions, rows, pending = [], [], [], []

        def append_transition(t, extra=None):
            row = {
                "state_id": t.state_id,
                "instance_action_id": t.instance_action_id,
                "action_type": t.action_type.value,
                "intervention_cost_units": t.intervention_cost_units,
                "expected_false_interventions": t.expected_false_interventions,
                "duration_ms": t.duration_ms,
                "prediction_evidence_id": t.prediction_evidence_id,
                "outcomes": [
                    {
                        "probability": o.probability,
                        "next_state_id": o.next_state_id,
                        "outcome_id": o.outcome_id,
                    }
                    for o in t.outcomes
                ],
            }
            row.update(extra or {})
            transitions.append(t)
            rows.append(row)

        def guided_future(source, action_id, depth):
            """Charge main Actor time and public verification after admission, not at delivery."""
            rule, mapping = bound[action_id]
            pending.append(
                {
                    "state_id": source.state_id,
                    "guidance_action_id": action_id,
                    "rule_id": rule.rule_id,
                    "check_ids": sorted(mapping),
                }
            )
            if depth >= horizon:
                return
            # Each configured check costs one unit in the upstream slate builder.
            cost = len(mapping)
            duration = rule.actor_duration_ms + rule.recheck_duration_ms
            verified = (
                cost <= source.remaining_intervention_budget
                and duration <= source.remaining_time_ms
            )
            if not verified and rule.actor_duration_ms > source.remaining_time_ms:
                return
            targets = []
            outcomes = (
                rule.outcomes
                if verified
                else (
                    GuidedCheckOutcome(
                        "continued_without_affordable_recheck",
                        1.0,
                        tuple((k, "unknown") for k in sorted(mapping)),
                    ),
                )
            )
            charged_duration = duration if verified else rule.actor_duration_ms
            for o in outcomes:
                # Continuing the unrestricted Actor invalidates any un-rechecked predicate.
                predicates = {key: "stale" for key, _ in source.residual_predicate_states}
                for check_id, verdict in o.check_verdicts:
                    predicates[mapping[check_id]] = verdict if verified else "stale"
                target = replace(
                    source,
                    state_id=f"{source.state_id}:main:{o.outcome_id}",
                    public_goal_complete=all(v == "supported" for v in predicates.values()),
                    unblocked_fraction=_dependency_progress_fraction(
                        projection.task_dependencies,
                        projection.acceptance_predicates,
                        tuple(predicates.items()),
                    ),
                    remaining_intervention_budget=source.remaining_intervention_budget
                    - (cost if verified else 0),
                    remaining_actor_steps=source.remaining_actor_steps - 1,
                    remaining_time_ms=source.remaining_time_ms - charged_duration,
                    residual_predicate_states=tuple(predicates.items()),
                    outcome_phase="main_actor_and_registered_rechecks"
                    if verified
                    else "main_actor_without_affordable_recheck",
                )
                states.append(target)
                targets.append(
                    RuntimeJointTransitionOutcome(o.probability, target.state_id, o.outcome_id)
                )
            append_transition(
                RuntimeTimedActionTransition(
                    source.state_id,
                    f"guided-main:{action_id}",
                    RuntimeJointActionType.RUN_REGISTERED_CHECK
                    if verified
                    else RuntimeJointActionType.CONTINUE_ACTOR,
                    cost if verified else 0,
                    0.0,
                    charged_duration,
                    rule.provenance_id
                    if verified
                    else "public-budget:no-affordable-guided-recheck",
                    tuple(targets),
                ),
                {
                    "transition_kind": "conditional_main_actor_and_recheck_phase",
                    "rule_id": rule.rule_id,
                    "prediction_scope": _SCOPE,
                    "actor_duration_ms": rule.actor_duration_ms,
                    "recheck_duration_ms": rule.recheck_duration_ms if verified else 0,
                    "check_ids": sorted(mapping) if verified else [],
                    "root_executable": False,
                },
            )

        def visit(source, depth):
            """Rebuild descendants from corrected source state instead of copying old support."""
            for t in outgoing.get(source.state_id, ()):
                action = actions[t.instance_action_id]
                delivery = t.instance_action_id in bound
                children = []
                for o in t.outcomes:
                    if delivery:
                        child = replace(
                            old_states[o.next_state_id],
                            public_goal_complete=source.public_goal_complete,
                            unblocked_fraction=source.unblocked_fraction,
                            residual_predicate_states=source.residual_predicate_states,
                            remaining_actor_steps=horizon - depth - 1,
                            outcome_phase=f"guidance_delivery:{o.outcome_id}",
                        )
                    else:
                        child = _outcome_node(
                            source=_ProjectedNode(source, source.residual_predicate_states),
                            candidate=action,
                            outcome_id=o.outcome_id,
                            state_id=o.next_state_id,
                            duration_ms=t.duration_ms,
                            projection=projection,
                        ).state
                    states.append(child)
                    children.append((o, child))
                append_transition(t, row_by_key[t.state_id, t.instance_action_id])
                for o, child in children:
                    if delivery and o.outcome_id == "supported":
                        guided_future(child, t.instance_action_id, depth + 1)
                    else:
                        visit(child, depth + 1)

        root = replace(old_states["root"], remaining_actor_steps=horizon)
        states.append(root)
        visit(root, 0)
        by_id = {s.state_id: s for s in states}
        roots = [t for t in transitions if t.state_id == "root"]
        baseline = max(roots, key=lambda t: _immediate_value(t, by_id).priority)
        return _CompiledOptionModel(
            tuple(states), tuple(transitions), tuple(rows), baseline.instance_action_id
        ), pending

    def resolve(self, slate, projection, estimates, *, horizon=3):
        """Select only a live root action; future completion stages remain forecasts."""
        compiled, pending = self._compile(slate, projection, estimates, horizon)
        model = {
            "schema_version": self.schema_version,
            "implementation_status": "implementation_unverified",
            "selector_identity": self.selector_identity,
            "objective_identity": "residual_completion_then_unblocked_then_time_then_false_interventions_then_intervention_cost_v1",
            "horizon": horizon,
            "slate": slate.to_dict(),
            "projection": projection.to_dict(),
            "atomic_estimates": [e.to_dict() for e in estimates],
            "configured_checks": runtime_check_configuration_payload(
                self.configured_checks, diagnosis_enabled=False
            )["contracts"],
            "guided_completion_configuration": self.configuration.to_dict(),
            "generated_states": [_state_payload(s) for s in compiled.states],
            "generated_transitions": list(compiled.transition_rows),
            "pending_guidance_states": pending,
            "immediate_baseline_action_id": compiled.immediate_baseline_action_id,
            "lookahead_scope": "two_level_runtime_options_with_guided_completion_extension",
            "official_evaluator_consumed": False,
            "terminal_value_semantics": "public_residual_completion_proxy_only",
        }
        return self._solve(slate=slate, model=model, compiled=compiled)

    def _solve(self, **kwargs):
        """Reuse Bellman arithmetic while giving the new scope a distinct rule identity."""
        return replace(
            super()._solve(**kwargs), rule_id="runtime_guidance_phase_completion_bellman_v1"
        )

    def resolve_recorded(self, slate, model: Mapping):
        """Rebuild every phase from the bound rules and reject altered saved leaves."""
        projection = RuntimeOwnedOptionProjection.from_dict(model["projection"])
        estimates = tuple(
            RuntimeAtomicActionEstimate.from_dict(e) for e in model["atomic_estimates"]
        )
        choice = self.resolve(slate, projection, estimates, horizon=model["horizon"])
        expected = dict(choice.planning_model)
        # The live wrapper owns and separately checks its estimate receipt.
        actual = {k: v for k, v in model.items() if k != "atomic_estimate_receipt"}
        if actual != expected:
            raise ValueError("recorded guidance phase model differs from reconstruction")
        return replace(choice, planning_model=dict(model))


@dataclass(frozen=True)
class _PhaseEstimateBundle:
    """Freeze both probability scopes in the existing live receipt lifecycle."""

    atomic: RuntimeFrozenAtomicEstimateConfiguration
    guided: GuidedCompletionConfiguration

    @property
    def rules(self):
        return self.atomic.rules

    def to_dict(self):
        return {
            "schema_version": "runtime_guidance_phase_estimate_bundle_v1",
            "atomic": self.atomic.to_dict(),
            "guided": self.guided.to_dict(),
        }


class RuntimeGuidancePhaseLiveResolver(RuntimeOwnedOptionLiveResolver):
    """Bind public input and both scoped forecasts through the real live selector seam."""

    supported_planning_horizons = (1, 2, 3)
    receipt_schema_version = "runtime_guidance_phase_estimate_receipt_v1"

    def __init__(self, *, guided_configuration, **kwargs):
        cap = kwargs.get("planning_horizon_cap")
        if cap is not None and type(cap) is not int:
            raise ValueError("guidance phase planning horizon cap must be an integer")
        super().__init__(**kwargs)
        self.configuration = _PhaseEstimateBundle(self.configuration, guided_configuration)
        self._resolver = RuntimeGuidancePhaseDPResolver(
            guided_configuration, self.configured_checks
        )

    def _native_horizon(self, trigger_tool_name):
        """Allow the admitted-guidance extension after either decision boundary."""
        return 2 if trigger_tool_name == "runtime_option_replan_trigger" else 3

    def _effective_horizon(self, native):
        """Cap abstract phase epochs without limiting later real closed-loop actions."""
        return (
            min(native, self.planning_horizon_cap)
            if self.planning_horizon_cap is not None
            else native
        )

    def resolve_recorded(self, slate, model):
        """Bind the phase window and current scope before rebuilding saved forecasts."""
        receipt = model.get("atomic_estimate_receipt", {})
        native = receipt.get("native_planning_horizon")
        if (
            native not in (2, 3)
            or receipt.get("planning_horizon_cap") != self.planning_horizon_cap
            or receipt.get("residual_structure") != self.residual_structure
            or self._effective_horizon(native) != model.get("horizon")
        ):
            raise ValueError("recorded guidance phase window differs")
        if self.public_projection is not None:
            triggers = [
                r
                for r in self.public_projection()["actions"]
                if r["action_id"] == slate.trigger_action_id
            ]
            if len(triggers) != 1 or self._native_horizon(triggers[0]["tool_name"]) != native:
                raise ValueError("recorded guidance phase trigger differs")
        return super().resolve_recorded(slate, model)
