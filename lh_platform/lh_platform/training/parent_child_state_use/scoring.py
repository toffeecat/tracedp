"""Pure prompt construction and candidate-action scoring contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lh_platform.training.g2po.actor_protocol import (
    G2POActorHistoryStep,
    G2POAlfWorldActorProtocol,
)
from lh_platform.training.parent_child_state_use.contracts import (
    FunctionalStateUseError,
    ReconstructedState,
)
from lh_platform.training.parent_child_state_use.exposure import (
    ActorStateExposureMode,
    render_actor_state_exposure_extension,
)


def render_condition_prompt(
    protocol: G2POAlfWorldActorProtocol,
    state: ReconstructedState,
    *,
    remaining_budget: int,
    mode: ActorStateExposureMode,
    corrupted_history: tuple[G2POActorHistoryStep, ...],
) -> tuple[str, Mapping[str, Any] | None]:
    history = state.history
    step_count_override = None
    extension = None
    exposure = None
    if mode is ActorStateExposureMode.LOCAL_ONLY:
        history = ()
        step_count_override = len(state.history)
    elif mode is ActorStateExposureMode.CORRUPTED_HISTORY:
        history = corrupted_history
        exposure = {
            "control": "swapped_pair_history",
            "history_source_state_id": "other_pair_member",
        }
    elif mode not in {
        ActorStateExposureMode.NORMAL,
        ActorStateExposureMode.LOCAL_ONLY,
        ActorStateExposureMode.CORRUPTED_HISTORY,
    }:
        extension, exposure = render_actor_state_exposure_extension(
            state.graph,
            state.vector,
            remaining_budget=remaining_budget,
            mode=mode,
        )
    prompt = protocol.render_prompt(
        task_description=state.task_description,
        current_observation=state.observation.text,
        admissible_actions=state.observation.admissible_actions,
        history=history,
        context_extension=extension,
        step_count_override=step_count_override,
    )
    return prompt, exposure


def validate_action_distribution(
    distribution: Mapping[str, Any],
    *,
    expected_checkpoint: str,
    expected_actions: tuple[str, ...],
) -> None:
    if distribution.get("policy_checkpoint_sha256") != expected_checkpoint:
        raise FunctionalStateUseError("action distribution belongs to another checkpoint")
    probabilities = probability_map(distribution)
    if set(probabilities) != set(expected_actions):
        raise FunctionalStateUseError("action distribution does not cover candidate set")
    if not all(probability >= 0.0 for probability in probabilities.values()):
        raise FunctionalStateUseError("action distribution contains negative mass")
    if abs(sum(probabilities.values()) - 1.0) > 1e-5:
        raise FunctionalStateUseError("action distribution is not normalized")
    rows = distribution.get("actions")
    if not isinstance(rows, list):
        raise FunctionalStateUseError("action distribution rows are missing")
    for row in rows:
        if not isinstance(row, Mapping):
            raise FunctionalStateUseError("action distribution row is invalid")
        action_tokens = row.get("action_token_ids")
        termination = row.get("termination_token_id")
        termination_tokens = row.get("termination_token_ids")
        scored_tokens = row.get("scored_token_ids")
        if termination_tokens is None:
            normalized_termination = (
                [termination]
                if isinstance(termination, int) and not isinstance(termination, bool)
                else None
            )
        elif (
            isinstance(termination_tokens, list)
            and termination_tokens
            and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in termination_tokens
            )
            and (
                termination is None
                or (len(termination_tokens) == 1 and termination == termination_tokens[0])
            )
        ):
            normalized_termination = termination_tokens
        else:
            normalized_termination = None
        if (
            not isinstance(action_tokens, list)
            or normalized_termination is None
            or scored_tokens != [*action_tokens, *normalized_termination]
            or row.get("scored_token_count") != len(scored_tokens)
            or row.get("action_token_count") != len(action_tokens)
        ):
            raise FunctionalStateUseError(
                "action distribution row lacks a replayable registered-termination token path"
            )


def probability_map(distribution: Mapping[str, Any]) -> dict[str, float]:
    rows = distribution.get("actions")
    if not isinstance(rows, list):
        raise FunctionalStateUseError("action distribution rows are missing")
    probabilities = {
        str(row["action"]): float(row["probability"]) for row in rows if isinstance(row, Mapping)
    }
    if len(probabilities) != len(rows):
        raise FunctionalStateUseError("action distribution contains duplicate actions")
    return probabilities


def pair_metrics(
    distributions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    target_actions: Mapping[str, tuple[str, ...]],
    modes: tuple[ActorStateExposureMode, ...],
) -> dict[str, Any]:
    left, right = tuple(distributions)
    left_targets = target_actions[left]
    right_targets = target_actions[right]
    metrics: dict[str, Any] = {}
    for mode in modes:
        left_probabilities = probability_map(distributions[left][mode.value])
        right_probabilities = probability_map(distributions[right][mode.value])
        sensitivity = 0.5 * (
            probability_mass(left_probabilities, left_targets)
            - probability_mass(right_probabilities, left_targets)
            + probability_mass(right_probabilities, right_targets)
            - probability_mass(left_probabilities, right_targets)
        )
        metrics[mode.value] = {
            "pair_sensitivity": sensitivity,
            "mean_target_action_mass": 0.5
            * (
                probability_mass(left_probabilities, left_targets)
                + probability_mass(right_probabilities, right_targets)
            ),
            "state_ids": [left, right],
        }
    return metrics


def probability_mass(
    probabilities: Mapping[str, float],
    actions: tuple[str, ...],
) -> float:
    return sum(float(probabilities.get(action, 0.0)) for action in actions)


__all__ = [
    "pair_metrics",
    "probability_map",
    "probability_mass",
    "render_condition_prompt",
    "validate_action_distribution",
]
