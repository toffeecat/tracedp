"""Environment-fact subproblem contracts for the ALFWorld P0 ablation.

The compiler deliberately consumes only the task's PDDL metadata and current
TextWorld facts.  It never reads ``plan.high_pddl`` or ``plan.low_actions``:
those are reference trajectories and would make a credit-only ablation leak
the answer into its annotations.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

ALFWORLD_SUBPROBLEM_CONTRACT_SCHEMA = "lh_alfworld_subproblem_contract/v2"
ALFWORLD_SUBPROBLEM_VERIFIER_INPUT_SCHEMA = "lh_alfworld_subproblem_verifier_input/v2"

_SUPPORTED_TASKS = frozenset(
    {
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
    }
)


class AlfWorldSubproblemError(ValueError):
    """Raised when an ALFWorld runtime cannot emit a verified P0 phase."""


@dataclass(frozen=True)
class AlfWorldSubproblemAnnotation:
    """One current phase derived from environment facts, not agent text."""

    active_subproblem_id: str
    contract: Mapping[str, Any]
    verifier_input: Mapping[str, Any]
    source: str = "environment_contract"

    def to_builder_args(self) -> dict[str, Any]:
        return {
            "active_subproblem_id": self.active_subproblem_id,
            "contract": dict(self.contract),
            "verifier_input": dict(self.verifier_input),
            "source": self.source,
        }


def derive_alfworld_active_subproblem(
    traj_data: Mapping[str, Any], facts: Iterable[Any]
) -> AlfWorldSubproblemAnnotation:
    """Return the first unfinished, fact-verifiable ALFWorld phase.

    This provides a privileged *training annotation* only.  P0 must not pass
    the returned phase ID or contract text to the actor prompt.  The returned
    contract has stable task/phase semantics, while fact truth is used only to
    select the active phase; this keeps a phase key stable across its states.
    """

    task_type, targets = _parse_task_metadata(traj_data)
    normalized_facts = _normalize_facts(facts)
    object_instances = _instances_of_type(normalized_facts, "objecttype", targets["object"])
    receptacle_instances = _instances_of_type(
        normalized_facts, "receptacletype", targets.get("receptacle", "")
    )
    toggle_instances = _instances_of_type(normalized_facts, "objecttype", targets.get("toggle", ""))

    held_targets = _held_targets(normalized_facts, object_instances)
    transformed = {
        "clean": _unary_targets(normalized_facts, "isclean", object_instances),
        "hot": _unary_targets(normalized_facts, "ishot", object_instances),
        "cool": _unary_targets(normalized_facts, "iscool", object_instances),
    }
    placed_targets = _placed_targets(normalized_facts, object_instances, receptacle_instances)
    placed_target_groups = _placed_targets_by_receptacle(
        normalized_facts, object_instances, receptacle_instances
    )
    toggle_ready = bool(_unary_targets(normalized_facts, "istoggled", toggle_instances))

    phase = _select_phase(
        task_type,
        held_targets=held_targets,
        transformed=transformed,
        placed_targets=placed_targets,
        max_targets_in_one_receptacle=max(
            (len(objects) for objects in placed_target_groups.values()),
            default=0,
        ),
        toggle_ready=toggle_ready,
    )
    contract = {
        "schema": ALFWORLD_SUBPROBLEM_CONTRACT_SCHEMA,
        "task_type": task_type,
        "phase": phase,
        "target_object_type": targets["object"],
        "target_receptacle_type": targets.get("receptacle", ""),
        "target_toggle_type": targets.get("toggle", ""),
    }
    verifier_input = {
        "schema": ALFWORLD_SUBPROBLEM_VERIFIER_INPUT_SCHEMA,
        "task_metadata": {
            "task_type": task_type,
            "pddl_params": {
                "object_target": targets["object"],
                "parent_target": targets.get("receptacle", ""),
                "toggle_target": targets.get("toggle", ""),
            },
        },
        "facts": [{"name": name, "args": list(args)} for name, args in sorted(normalized_facts)],
    }
    return AlfWorldSubproblemAnnotation(
        active_subproblem_id=phase,
        contract=contract,
        verifier_input=verifier_input,
    )


def verify_alfworld_subproblem_annotation(
    *,
    active_subproblem_id: str,
    contract: Mapping[str, Any],
    verifier_input: Mapping[str, Any],
) -> AlfWorldSubproblemAnnotation:
    """Recompute one phase from its retained task metadata and environment facts."""

    if verifier_input.get("schema") != ALFWORLD_SUBPROBLEM_VERIFIER_INPUT_SCHEMA:
        raise AlfWorldSubproblemError("unsupported ALFWorld subproblem verifier input")
    task_metadata = verifier_input.get("task_metadata")
    facts = verifier_input.get("facts")
    if not isinstance(task_metadata, Mapping) or not isinstance(facts, list):
        raise AlfWorldSubproblemError(
            "ALFWorld subproblem verifier input requires task_metadata and facts"
        )
    recomputed = derive_alfworld_active_subproblem(task_metadata, facts)
    if recomputed.active_subproblem_id != active_subproblem_id:
        raise AlfWorldSubproblemError("active subproblem does not match environment facts")
    if dict(recomputed.contract) != dict(contract):
        raise AlfWorldSubproblemError("subproblem contract does not match environment facts")
    if dict(recomputed.verifier_input) != dict(verifier_input):
        raise AlfWorldSubproblemError("subproblem verifier input is not canonical")
    return recomputed


def _parse_task_metadata(traj_data: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    if not isinstance(traj_data, Mapping):
        raise AlfWorldSubproblemError("ALFWorld task metadata must be a mapping")
    task_type = str(traj_data.get("task_type", "")).strip()
    if task_type not in _SUPPORTED_TASKS:
        raise AlfWorldSubproblemError(f"unsupported or non-P0 ALFWorld task type: {task_type!r}")
    params = traj_data.get("pddl_params")
    if not isinstance(params, Mapping):
        raise AlfWorldSubproblemError("ALFWorld task metadata lacks pddl_params")
    object_type = _type_name(params.get("object_target"))
    receptacle_type = _type_name(params.get("parent_target"))
    toggle_type = _type_name(params.get("toggle_target"))
    if not object_type:
        raise AlfWorldSubproblemError("ALFWorld task metadata lacks object_target")
    if task_type != "look_at_obj_in_light" and not receptacle_type:
        raise AlfWorldSubproblemError("ALFWorld placement task metadata lacks parent_target")
    if task_type == "look_at_obj_in_light" and not toggle_type:
        raise AlfWorldSubproblemError("ALFWorld light task metadata lacks toggle_target")
    return task_type, {
        "object": object_type,
        "receptacle": receptacle_type,
        "toggle": toggle_type,
    }


def _select_phase(
    task_type: str,
    *,
    held_targets: frozenset[str],
    transformed: Mapping[str, frozenset[str]],
    placed_targets: frozenset[str],
    max_targets_in_one_receptacle: int,
    toggle_ready: bool,
) -> str:
    if task_type == "pick_two_obj_and_place":
        placed_count = max_targets_in_one_receptacle
        if placed_count >= 2:
            return "complete"
        if placed_count == 1:
            return "acquire_and_place_second"
        return "place_first" if held_targets else "acquire_first"
    if task_type == "look_at_obj_in_light":
        if held_targets and toggle_ready:
            return "complete"
        if not held_targets:
            return "acquire_target"
        return "toggle_light"
    transform_phase = {
        "pick_clean_then_place_in_recep": ("clean", "transform_clean"),
        "pick_heat_then_place_in_recep": ("hot", "transform_heat"),
        "pick_cool_then_place_in_recep": ("cool", "transform_cool"),
    }.get(task_type)
    if transform_phase is not None:
        fact_name, phase = transform_phase
        transformed_targets = transformed[fact_name]
        if transformed_targets & placed_targets:
            return "complete"
        if not held_targets:
            return "acquire_target"
        if not (held_targets & transformed_targets):
            return phase
        return "place_target"
    if placed_targets:
        return "complete"
    return "place_target" if held_targets else "acquire_target"


def _normalize_facts(facts: Iterable[Any]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for fact in facts:
        if isinstance(fact, Mapping):
            name = fact.get("name")
            args = fact.get("args")
            if not isinstance(name, str) or not isinstance(args, list):
                raise AlfWorldSubproblemError(f"unsupported normalized TextWorld fact: {fact!r}")
            normalized.append(
                (name.strip().lower(), tuple(str(value).strip().lower() for value in args))
            )
            continue
        if isinstance(fact, str):
            name, args = _parse_text_fact(fact)
            normalized.append((name, args))
            continue
        else:
            name = getattr(fact, "name", None)
            names = getattr(fact, "names", None)
            if name is None:
                arguments = getattr(fact, "arguments", ())
                names = tuple(getattr(argument, "name", argument) for argument in arguments)
            if name is None:
                raise AlfWorldSubproblemError(f"unsupported TextWorld fact: {fact!r}")
            parts = [str(name).lower(), *(str(value).lower() for value in (names or ()))]
        if parts:
            normalized.append((parts[0], tuple(parts[1:])))
    return tuple(normalized)


def _parse_text_fact(fact: str) -> tuple[str, tuple[str, ...]]:
    """Normalize the string representation emitted by TextWorld facts.

    TextWorld's object names include an ordinal (for example ``apple 1``),
    while its printable facts are whitespace-delimited.  Splitting those facts
    naively turns the same entity into ``apple`` in type facts and ``apple 1``
    in relation facts, which makes the contract compiler silently miss every
    transition.  The native ``Proposition.names`` path already preserves these
    names; this parser gives the log/replay string path the same semantics.
    """

    pddl = re.fullmatch(r"\s*([^\s(]+)\((.*)\)\s*", fact.lower())
    if pddl is not None:
        values = tuple(
            argument.split(":", 1)[0].strip()
            for argument in pddl.group(2).split(",")
            if argument.strip()
        )
        return pddl.group(1), values
    parts = fact.lower().split()
    if not parts:
        raise AlfWorldSubproblemError("empty TextWorld fact")
    name, values = parts[0], parts[1:]
    if name in {"objecttype", "receptacletype"} and len(values) >= 2:
        return name, (" ".join(values[:-1]), values[-1])
    if name == "holds" and len(values) >= 2:
        return name, (values[0], " ".join(values[1:]))
    if name in {"isclean", "ishot", "iscool", "istoggled"}:
        return name, (" ".join(values),)
    if name == "inreceptacle" and values:
        # In ALFWorld's printed facts every concrete entity ends in its ordinal.
        # Keep the fallback whole value so unsupported formats fail closed later.
        ordinal = next((index for index, value in enumerate(values) if value.isdigit()), None)
        if ordinal is not None and ordinal + 1 < len(values):
            return name, (" ".join(values[: ordinal + 1]), " ".join(values[ordinal + 1 :]))
    return name, tuple(values)


def _instances_of_type(
    facts: Iterable[tuple[str, tuple[str, ...]]], predicate: str, type_name: str
) -> frozenset[str]:
    if not type_name:
        return frozenset()
    expected = f"{type_name}type"
    return frozenset(
        args[0]
        for name, args in facts
        if name == predicate and len(args) >= 2 and args[-1] == expected
    )


def _held_targets(
    facts: Iterable[tuple[str, tuple[str, ...]]], targets: frozenset[str]
) -> frozenset[str]:
    return frozenset(
        args[-1] for name, args in facts if name == "holds" and args and args[-1] in targets
    )


def _unary_targets(
    facts: Iterable[tuple[str, tuple[str, ...]]], predicate: str, targets: frozenset[str]
) -> frozenset[str]:
    return frozenset(
        args[0] for name, args in facts if name == predicate and args and args[0] in targets
    )


def _placed_targets(
    facts: Iterable[tuple[str, tuple[str, ...]]],
    objects: frozenset[str],
    receptacles: frozenset[str],
) -> frozenset[str]:
    return frozenset(
        args[0]
        for name, args in facts
        if name == "inreceptacle"
        and len(args) >= 2
        and args[0] in objects
        and args[1] in receptacles
    )


def _placed_targets_by_receptacle(
    facts: Iterable[tuple[str, tuple[str, ...]]],
    objects: frozenset[str],
    receptacles: frozenset[str],
) -> dict[str, frozenset[str]]:
    """Group target instances by the concrete receptacle that contains them.

    ALFWorld's ``pick_two_obj_and_place`` parent contract quantifies one
    receptacle instance and requires two distinct targets inside that same
    instance.  Counting targets across all receptacles of the requested type
    would accept a state that the official PDDL rejects.
    """

    grouped: dict[str, set[str]] = {}
    for name, args in facts:
        if (
            name == "inreceptacle"
            and len(args) >= 2
            and args[0] in objects
            and args[1] in receptacles
        ):
            grouped.setdefault(args[1], set()).add(args[0])
    return {receptacle: frozenset(members) for receptacle, members in grouped.items()}


def _type_name(value: Any) -> str:
    return str(value or "").strip().lower()
