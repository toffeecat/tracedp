"""Derive ALFWorld action/precondition violation sets from frozen PDDL facts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_action_language import parse_alfworld_action
from lh_platform.tracing.alfworld_precondition_failure_replay import (
    ALFWORLD_CONTINUATION_ANCHOR_PRECONDITION_FAILURE_REPLAY_SCHEMA,
    ALFWORLD_CONTROLLED_TERMINAL_PRECONDITION_FAILURE_REPLAY_SCHEMA,
    ALFWORLD_PRECONDITION_FAILURE_REPLAY_SCHEMA,
)

ALFWORLD_PRECONDITION_FAILURE_GOLD_SCHEMA = (
    "lh_alfworld_precondition_failure_gold/v2"
)
FROZEN_ALFWORLD_PDDL_DOMAIN_SHA256 = (
    "0af55e7f92d12b09d28515aa730e4d03fddd17228ef1fd7b18a3ee8ba49b27b9"
)


class AlfWorldPreconditionGoldError(ValueError):
    """Raised when a proposed action cannot be bound to frozen PDDL evidence."""


def _require_mapping(label: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlfWorldPreconditionGoldError(f"{label} must be an object")
    return value


def _normalized_facts(raw_facts: object) -> set[tuple[str, tuple[str, ...]]]:
    if not isinstance(raw_facts, list) or not raw_facts:
        raise AlfWorldPreconditionGoldError("pre-state facts are missing")
    facts: set[tuple[str, tuple[str, ...]]] = set()
    for raw_fact in raw_facts:
        fact = _require_mapping("pre-state fact", raw_fact)
        name = fact.get("name")
        args = fact.get("args")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(args, list)
            or any(not isinstance(value, str) or not value.strip() for value in args)
        ):
            raise AlfWorldPreconditionGoldError("pre-state fact is malformed")
        facts.add(
            (
                name.strip().lower(),
                tuple(value.strip().lower() for value in args),
            )
        )
    return facts


def _fact(
    facts: set[tuple[str, tuple[str, ...]]],
    name: str,
    *args: str,
) -> bool:
    return (name, tuple(args)) in facts


def _values_for(
    facts: set[tuple[str, tuple[str, ...]]],
    name: str,
    first_arg: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(args[1] for fact_name, args in facts if fact_name == name and args[0] == first_arg)
    )


def _entity_names(
    facts: set[tuple[str, tuple[str, ...]]], name: str
) -> set[str]:
    return {args[0] for fact_name, args in facts if fact_name == name}


def _violation(label: str, **evidence: Any) -> dict[str, Any]:
    return {"label": label, "evidence": evidence}


def _resolve_argument(
    *,
    value: str,
    expected_kind: str,
    expected_names: set[str],
    other_names: set[str],
) -> tuple[str | None, dict[str, Any] | None]:
    if value in expected_names:
        return value, None
    if value in other_names:
        return None, _violation(
            "argument_type_mismatch",
            argument=value,
            expected_kind=expected_kind,
            observed_kind="object" if expected_kind == "receptacle" else "receptacle",
        )
    candidates = tuple(sorted(name for name in expected_names if name.startswith(value + " ")))
    if len(candidates) > 1:
        return None, _violation(
            "ambiguous_entity_reference",
            argument=value,
            expected_kind=expected_kind,
            canonical_candidates=list(candidates),
        )
    if len(candidates) == 1:
        return None, _violation(
            "noncanonical_entity_reference",
            argument=value,
            expected_kind=expected_kind,
            canonical_candidate=candidates[0],
        )
    return None, _violation(
        "unknown_entity_reference",
        argument=value,
        expected_kind=expected_kind,
        canonical_candidates=[],
    )


def _resolve_examine_argument(
    *,
    value: str,
    object_names: set[str],
    receptacle_names: set[str],
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    if value in object_names:
        return "ExamineObject", "object", None
    if value in receptacle_names:
        return "ExamineReceptacle", "receptacle", None
    candidates = tuple(
        sorted(name for name in object_names | receptacle_names if name.startswith(value + " "))
    )
    if len(candidates) > 1:
        return None, None, _violation(
            "ambiguous_entity_reference",
            argument=value,
            expected_kind="object_or_receptacle",
            canonical_candidates=list(candidates),
        )
    if len(candidates) == 1:
        return None, None, _violation(
            "noncanonical_entity_reference",
            argument=value,
            expected_kind="object_or_receptacle",
            canonical_candidate=candidates[0],
        )
    return None, None, _violation(
        "unknown_entity_reference",
        argument=value,
        expected_kind="object_or_receptacle",
        canonical_candidates=[],
    )


def _current_agent_location(
    facts: set[tuple[str, tuple[str, ...]]],
) -> str:
    locations = tuple(
        sorted(
            args[1]
            for name, args in facts
            if name == "atlocation" and args[0].strip() == "agent1"
        )
    )
    if len(locations) != 1:
        raise AlfWorldPreconditionGoldError("agent location is not unique")
    return locations[0]


def _receptacle_state_violations(
    *,
    facts: set[tuple[str, tuple[str, ...]]],
    receptacle: str,
    agent_location: str,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    locations = _values_for(facts, "receptacleatlocation", receptacle)
    if agent_location not in locations:
        violations.append(
            _violation(
                "agent_not_at_receptacle",
                receptacle=receptacle,
                agent_location=agent_location,
                receptacle_locations=list(locations),
                required_predicate="receptacleAtLocation(receptacle, agent_location)",
            )
        )
    if _fact(facts, "openable", receptacle) and not _fact(
        facts, "opened", receptacle
    ):
        violations.append(
            _violation(
                "receptacle_closed",
                receptacle=receptacle,
                observed_openable=True,
                observed_opened=False,
                required_disjunction="not openable(receptacle) or opened(receptacle)",
            )
        )
    return violations


def derive_alfworld_precondition_failure_gold(
    *,
    replay_receipt: Mapping[str, Any],
    pddl_domain: str,
) -> dict[str, Any]:
    """Return the full violated-precondition set for one rejected proposal.

    ALFWorld's PDDL environment dispatches only by exact membership in the
    current valid-command list. This function therefore separates failures
    that cannot ground a domain action from grounded actions whose frozen PDDL
    preconditions are false in the exact replay facts.
    """

    canonical_receipt = dict(replay_receipt)
    claimed_receipt_sha256 = canonical_receipt.pop("content_sha256", None)
    pre_state = _require_mapping("pre state", replay_receipt.get("pre_state"))
    post_state = _require_mapping("post state", replay_receipt.get("post_state"))
    proposal = replay_receipt.get("proposal")
    if (
        replay_receipt.get("schema")
        not in {
            ALFWORLD_PRECONDITION_FAILURE_REPLAY_SCHEMA,
            ALFWORLD_CONTINUATION_ANCHOR_PRECONDITION_FAILURE_REPLAY_SCHEMA,
            ALFWORLD_CONTROLLED_TERMINAL_PRECONDITION_FAILURE_REPLAY_SCHEMA,
        }
        or claimed_receipt_sha256
        != "sha256:" + canonical_sha256(canonical_receipt)
        or replay_receipt.get("proposal_in_pre_state_live_menu") is not False
        or replay_receipt.get("benchmark_facts_unchanged") is not True
        or replay_receipt.get("live_menu_unchanged") is not True
        or replay_receipt.get("environment_rejection_observed") is not True
        or replay_receipt.get("exact_transition_restore_replay_verified") is not True
        or replay_receipt.get("precondition_gold_assigned") is not False
        or not isinstance(proposal, str)
        or not proposal.strip()
        or post_state.get("observation") != "Nothing happens."
    ):
        raise AlfWorldPreconditionGoldError("replay receipt is not admitted P0 evidence")
    if not isinstance(pddl_domain, str) or not pddl_domain:
        raise AlfWorldPreconditionGoldError("PDDL domain is missing")
    pddl_domain_sha256 = hashlib.sha256(pddl_domain.encode()).hexdigest()
    if pddl_domain_sha256 != FROZEN_ALFWORLD_PDDL_DOMAIN_SHA256:
        raise AlfWorldPreconditionGoldError("PDDL domain identity changed")

    facts = _normalized_facts(pre_state.get("facts"))
    object_names = _entity_names(facts, "objecttype")
    receptacle_names = _entity_names(facts, "receptacletype")
    normalized_proposal = proposal.strip().lower()
    action_schema: str | None = None
    raw_arguments: dict[str, str] = {}
    unsupported_reason: str | None = None

    violations: list[dict[str, Any]] = []
    parsed_action = parse_alfworld_action(normalized_proposal)
    if parsed_action.status == "matched" and parsed_action.schema == "ExamineEntity":
        raw_value = parsed_action.arguments["entity"]
        schema, argument_name, resolution_violation = _resolve_examine_argument(
            value=raw_value,
            object_names=object_names,
            receptacle_names=receptacle_names,
        )
        if resolution_violation is not None:
            violations.append(resolution_violation)
        else:
            action_schema = schema
            raw_arguments = {str(argument_name): raw_value}

    elif parsed_action.status == "matched":
        action_schema = parsed_action.schema
        raw_arguments = dict(parsed_action.arguments)
    else:
        action_schema = parsed_action.schema
        raw_arguments = dict(parsed_action.arguments)
        unsupported_reason = parsed_action.reason

    grounded_arguments: dict[str, str] = {}
    if (action_schema is None and not violations) or unsupported_reason is not None:
        violations.append(
            _violation(
                "unsupported_action_form",
                proposal=normalized_proposal,
                aligned_schema=action_schema,
                reason=unsupported_reason or "no frozen action template matched",
            )
        )
    else:
        for argument_name, raw_value in raw_arguments.items():
            expected_kind = "object" if argument_name in {"object", "tool"} else "receptacle"
            expected_names = object_names if expected_kind == "object" else receptacle_names
            other_names = receptacle_names if expected_kind == "object" else object_names
            grounded, resolution_violation = _resolve_argument(
                value=raw_value,
                expected_kind=expected_kind,
                expected_names=expected_names,
                other_names=other_names,
            )
            if resolution_violation is not None:
                violations.append(resolution_violation)
            elif grounded is not None:
                grounded_arguments[argument_name] = grounded

    # An action instance is either fully grounded or not grounded. Retaining a
    # valid target beside an unresolved object would overstate what the exact
    # command dispatcher can identify.
    if violations:
        grounded_arguments = {}

    if not violations and action_schema is not None:
        agent_location = _current_agent_location(facts)
        receptacle = grounded_arguments.get("receptacle")
        if action_schema == "GotoLocation":
            # The only state preconditions are the current agent location and
            # the target's benchmark-owned receptacle location; both exist for
            # any grounded GotoLocation command generated by this domain.
            locations = _values_for(facts, "receptacleatlocation", str(receptacle))
            if not locations:
                violations.append(
                    _violation(
                        "receptacle_has_no_location",
                        receptacle=str(receptacle),
                        required_predicate="receptacleAtLocation(receptacle, location)",
                    )
                )
            elif agent_location in locations:
                violations.append(
                    _violation(
                        "action_effect_already_satisfied",
                        action_schema=action_schema,
                        receptacle=str(receptacle),
                        agent_location=agent_location,
                        receptacle_locations=list(locations),
                        dispatcher_stage="fast_downward_sas_operator_materialization",
                    )
                )
        elif action_schema == "ExamineReceptacle":
            locations = _values_for(facts, "receptacleatlocation", str(receptacle))
            if agent_location not in locations:
                violations.append(
                    _violation(
                        "agent_not_at_receptacle",
                        receptacle=str(receptacle),
                        agent_location=agent_location,
                        receptacle_locations=list(locations),
                        required_predicate="receptacleAtLocation(receptacle, agent_location)",
                    )
                )
            elif _fact(facts, "checked", str(receptacle)):
                violations.append(
                    _violation(
                        "action_effect_already_satisfied",
                        action_schema=action_schema,
                        entity=str(receptacle),
                        satisfied_effect="checked(entity)",
                        dispatcher_stage="fast_downward_sas_operator_materialization",
                    )
                )
        elif action_schema == "ExamineObject":
            obj = grounded_arguments["object"]
            held = _fact(facts, "holds", "agent1", obj)
            accessible_receptacles = tuple(
                sorted(
                    candidate
                    for candidate in receptacle_names
                    if _fact(facts, "inreceptacle", obj, candidate)
                    and agent_location
                    in _values_for(facts, "receptacleatlocation", candidate)
                    and (
                        not _fact(facts, "openable", candidate)
                        or _fact(facts, "opened", candidate)
                    )
                )
            )
            if not held and not accessible_receptacles:
                violations.append(
                    _violation(
                        "object_not_visible_or_held",
                        object=obj,
                        held=False,
                        accessible_receptacles=[],
                        required_disjunction=(
                            "holds(agent1, object) or object in an open local receptacle"
                        ),
                    )
                )
            elif _fact(facts, "checked", obj):
                violations.append(
                    _violation(
                        "action_effect_already_satisfied",
                        action_schema=action_schema,
                        entity=obj,
                        satisfied_effect="checked(entity)",
                        dispatcher_stage="fast_downward_sas_operator_materialization",
                    )
                )
        elif action_schema == "PickupObject":
            obj = grounded_arguments["object"]
            if not _fact(facts, "pickupable", obj):
                violations.append(
                    _violation(
                        "object_not_pickupable",
                        object=obj,
                        required_predicate="pickupable(object)",
                    )
                )
            if not _fact(facts, "inreceptacle", obj, receptacle):
                violations.append(
                    _violation(
                        "object_not_in_receptacle",
                        object=obj,
                        receptacle=receptacle,
                        required_predicate="inReceptacle(object, receptacle)",
                    )
                )
            if _fact(facts, "holdsany", "agent1"):
                held_objects = _values_for(facts, "holds", "agent1")
                violations.append(
                    _violation(
                        "agent_holds_other_object",
                        held_objects=list(held_objects),
                        violated_negative_precondition="not holdsAny(agent1)",
                    )
                )
            violations.extend(
                _receptacle_state_violations(
                    facts=facts,
                    receptacle=str(receptacle),
                    agent_location=agent_location,
                )
            )
        elif action_schema == "PutObject":
            obj = grounded_arguments["object"]
            if not _fact(facts, "holds", "agent1", obj):
                held_objects = _values_for(facts, "holds", "agent1")
                violations.append(
                    _violation(
                        "agent_does_not_hold_object",
                        object=obj,
                        held_objects=list(held_objects),
                        required_predicate="holds(agent1, object)",
                    )
                )
            violations.extend(
                _receptacle_state_violations(
                    facts=facts,
                    receptacle=str(receptacle),
                    agent_location=agent_location,
                )
            )
            object_types = _values_for(facts, "objecttype", obj)
            receptacle_types = _values_for(facts, "receptacletype", str(receptacle))
            if not any(
                _fact(facts, "cancontain", receptacle_type, object_type)
                for receptacle_type in receptacle_types
                for object_type in object_types
            ):
                violations.append(
                    _violation(
                        "receptacle_cannot_contain_object",
                        object=obj,
                        receptacle=str(receptacle),
                        object_types=list(object_types),
                        receptacle_types=list(receptacle_types),
                        required_predicate="canContain(receptacle_type, object_type)",
                    )
                )
        elif action_schema in {"CleanObject", "HeatObject", "CoolObject"}:
            obj = grounded_arguments["object"]
            capability_by_schema = {
                "CleanObject": ("cleanable", "object_not_cleanable"),
                "HeatObject": ("heatable", "object_not_heatable"),
                "CoolObject": ("coolable", "object_not_coolable"),
            }
            capability, missing_capability_label = capability_by_schema[action_schema]
            if not _fact(facts, capability, obj):
                violations.append(
                    _violation(
                        missing_capability_label,
                        object=obj,
                        required_predicate=f"{capability}(object)",
                    )
                )
            required_types_by_schema = {
                "CleanObject": {"sinktype", "sinkbasintype"},
                "HeatObject": {"microwavetype"},
                "CoolObject": {"fridgetype"},
            }
            receptacle_types = _values_for(facts, "receptacletype", str(receptacle))
            required_types = required_types_by_schema[action_schema]
            if not set(receptacle_types).intersection(required_types):
                violations.append(
                    _violation(
                        "receptacle_wrong_appliance_type",
                        receptacle=str(receptacle),
                        receptacle_types=list(receptacle_types),
                        required_types=sorted(required_types),
                    )
                )
            if not _fact(facts, "holds", "agent1", obj):
                held_objects = _values_for(facts, "holds", "agent1")
                violations.append(
                    _violation(
                        "agent_does_not_hold_object",
                        object=obj,
                        held_objects=list(held_objects),
                        required_predicate="holds(agent1, object)",
                    )
                )
            locations = _values_for(facts, "receptacleatlocation", str(receptacle))
            if agent_location not in locations:
                violations.append(
                    _violation(
                        "agent_not_at_receptacle",
                        receptacle=str(receptacle),
                        agent_location=agent_location,
                        receptacle_locations=list(locations),
                        required_predicate="receptacleAtLocation(receptacle, agent_location)",
                    )
                )

        elif action_schema == "OpenObject":
            if not _fact(facts, "openable", str(receptacle)):
                violations.append(
                    _violation(
                        "receptacle_not_openable",
                        receptacle=str(receptacle),
                        required_predicate="openable(receptacle)",
                    )
                )
            locations = _values_for(facts, "receptacleatlocation", str(receptacle))
            if agent_location not in locations:
                violations.append(
                    _violation(
                        "agent_not_at_receptacle",
                        receptacle=str(receptacle),
                        agent_location=agent_location,
                        receptacle_locations=list(locations),
                        required_predicate="receptacleAtLocation(receptacle, agent_location)",
                    )
                )
            if _fact(facts, "opened", str(receptacle)):
                violations.append(
                    _violation(
                        "receptacle_already_open",
                        receptacle=str(receptacle),
                        violated_negative_precondition="not opened(receptacle)",
                    )
                )
        elif action_schema == "CloseObject":
            if not _fact(facts, "openable", str(receptacle)):
                violations.append(
                    _violation(
                        "receptacle_not_openable",
                        receptacle=str(receptacle),
                        required_predicate="openable(receptacle)",
                    )
                )
            locations = _values_for(facts, "receptacleatlocation", str(receptacle))
            if agent_location not in locations:
                violations.append(
                    _violation(
                        "agent_not_at_receptacle",
                        receptacle=str(receptacle),
                        agent_location=agent_location,
                        receptacle_locations=list(locations),
                        required_predicate="receptacleAtLocation(receptacle, agent_location)",
                    )
                )
            if not _fact(facts, "opened", str(receptacle)):
                violations.append(
                    _violation(
                        "receptacle_already_closed",
                        receptacle=str(receptacle),
                        required_predicate="opened(receptacle)",
                    )
                )
        elif action_schema == "ToggleObject":
            obj = grounded_arguments["object"]
            if not _fact(facts, "toggleable", obj):
                violations.append(
                    _violation(
                        "object_not_toggleable",
                        object=obj,
                        required_predicate="toggleable(object)",
                    )
                )
            local_receptacles = tuple(
                sorted(
                    candidate
                    for candidate in receptacle_names
                    if _fact(facts, "inreceptacle", obj, candidate)
                    and agent_location
                    in _values_for(facts, "receptacleatlocation", candidate)
                )
            )
            if not local_receptacles:
                violations.append(
                    _violation(
                        "object_not_at_agent_location",
                        object=obj,
                        agent_location=agent_location,
                        local_receptacles=[],
                    )
                )
        elif action_schema == "SliceObject":
            obj = grounded_arguments["object"]
            tool = grounded_arguments["tool"]
            if not _fact(facts, "sliceable", obj):
                violations.append(
                    _violation(
                        "object_not_sliceable",
                        object=obj,
                        required_predicate="sliceable(object)",
                    )
                )
            tool_types = _values_for(facts, "objecttype", tool)
            if not set(tool_types).intersection({"knifetype", "butterknifetype"}):
                violations.append(
                    _violation(
                        "tool_wrong_type",
                        tool=tool,
                        tool_types=list(tool_types),
                        required_types=["butterknifetype", "knifetype"],
                    )
                )
            if not _fact(facts, "holds", "agent1", tool):
                violations.append(
                    _violation(
                        "agent_does_not_hold_tool",
                        tool=tool,
                        held_objects=list(_values_for(facts, "holds", "agent1")),
                    )
                )
            object_locations = _values_for(facts, "objectatlocation", obj)
            if agent_location not in object_locations:
                violations.append(
                    _violation(
                        "object_not_at_agent_location",
                        object=obj,
                        agent_location=agent_location,
                        object_locations=list(object_locations),
                    )
                )

    if not violations and action_schema is not None:
        violations.append(
            _violation(
                "compiled_dispatcher_omitted_grounded_action",
                action_schema=action_schema,
                proposal=normalized_proposal,
                domain_preconditions_reconstructed_as_satisfied=True,
                dispatcher_stage="fast_downward_sas_operator_materialization",
                evidence_boundary=(
                    "exact omission is observed; optimizer-specific pruning reason is not inferred"
                ),
            )
        )
    if not violations:
        raise AlfWorldPreconditionGoldError("rejected proposal has no derivable failure mode")
    violations = sorted(violations, key=lambda item: item["label"])
    labels = [item["label"] for item in violations]
    if len(labels) != len(set(labels)):
        raise AlfWorldPreconditionGoldError("violation labels are not unique")
    gold: dict[str, Any] = {
        "schema": ALFWORLD_PRECONDITION_FAILURE_GOLD_SCHEMA,
        "task_id": replay_receipt.get("task_id"),
        "traj_uid": replay_receipt.get("traj_uid"),
        "sampling_seed_base": replay_receipt.get("sampling_seed_base"),
        "source_replay_content_sha256": claimed_receipt_sha256,
        "source_pre_state_facts_sha256": pre_state.get("facts_sha256"),
        "source_pre_state_live_menu_sha256": pre_state.get(
            "live_admissible_actions_sha256"
        ),
        "proposal": proposal,
        "pddl_domain_sha256": pddl_domain_sha256,
        "dispatcher_semantics": "exact_membership_in_current_valid_commands",
        "action_schema": action_schema,
        "raw_arguments": raw_arguments,
        "grounded_arguments": grounded_arguments,
        "violation_labels": labels,
        "violations": violations,
        "violation_set_is_multi_valued": len(labels) > 1,
        "gold_source": (
            "frozen_pddl_domain_plus_exact_replay_facts_plus_compiled_dispatcher_menu"
        ),
        "pair_outcome_used": False,
        "tracer_output_used": False,
        "precondition_gold_assigned": True,
    }
    gold["content_sha256"] = "sha256:" + canonical_sha256(gold)
    return gold


__all__ = [
    "ALFWORLD_PRECONDITION_FAILURE_GOLD_SCHEMA",
    "FROZEN_ALFWORLD_PDDL_DOMAIN_SHA256",
    "AlfWorldPreconditionGoldError",
    "derive_alfworld_precondition_failure_gold",
]
