"""Frozen-policy behavioral test for residual-task state use in ALFWorld."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from lh_platform.studies.runtime import AlfworldTask
from lh_platform.training.agentic_rl_config import load_agentic_rl_experiment
from lh_platform.training.agentic_rl_runtime import canonical_sha256
from lh_platform.training.alfworld_agentic_suite import (
    BranchFactory,
)
from lh_platform.training.alfworld_controlled_aliasing import (
    AlfWorldControlledAliasingRecipe,
    load_alfworld_controlled_aliasing_recipe,
)
from lh_platform.training.g2po.actor_protocol import (
    G2POAlfWorldActorProtocol,
)
from lh_platform.training.parent_child_state_use.contracts import (
    FrozenActionScoringPolicy,
    FunctionalStateUseError,
    ResolvedFunctionalStateUseBenchmark,
)
from lh_platform.training.parent_child_state_use.exposure import (
    ActorStateExposureMode,
)
from lh_platform.training.parent_child_state_use.interventions import (
    evaluate_continuation_programs as _evaluate_continuation_programs,
)
from lh_platform.training.parent_child_state_use.interventions import (
    label_one_step_actions as _label_one_step_actions,
)
from lh_platform.training.parent_child_state_use.interventions import (
    shortest_successful_first_actions as _shortest_successful_first_actions,
)
from lh_platform.training.parent_child_state_use.scoring import (
    pair_metrics as _pair_metrics,
)
from lh_platform.training.parent_child_state_use.scoring import (
    probability_map as _probability_map,
)
from lh_platform.training.parent_child_state_use.scoring import (
    probability_mass as _mass,
)
from lh_platform.training.parent_child_state_use.scoring import (
    render_condition_prompt as _render_condition_prompt,
)
from lh_platform.training.parent_child_state_use.scoring import (
    validate_action_distribution as _validate_distribution,
)
from lh_platform.training.parent_child_state_use.state_pair import (
    reconstruct_state as _reconstruct_state,
)
from lh_platform.training.parent_child_state_use.state_pair import (
    validate_state_pair as _validate_state_pair,
)
from lh_platform.training.subproblems.alfworld import AlfWorldSubproblemCompiler
from lh_platform.training.subproblems.base import goal_state_projection

FUNCTIONAL_STATE_USE_SCHEMA = "lh_alfworld_functional_residual_state_use/v2"
FUNCTIONAL_STATE_USE_MANIFEST_SCHEMA = "lh_alfworld_functional_residual_state_use_manifest/v1"
FUNCTIONAL_STATE_USE_CLAIM_SCOPE = (
    "controlled_frozen_candidate_action_prompt_sensitivity_"
    "not_endogenous_awareness_or_policy_improvement"
)

__all__ = [
    "FUNCTIONAL_STATE_USE_SCHEMA",
    "FUNCTIONAL_STATE_USE_MANIFEST_SCHEMA",
    "FUNCTIONAL_STATE_USE_CLAIM_SCOPE",
    "FunctionalStateUseError",
    "ResolvedFunctionalStateUseBenchmark",
    "FrozenActionScoringPolicy",
    "load_functional_state_use_benchmark",
    "collect_alfworld_functional_state_use_manifest",
    "score_alfworld_functional_state_use_manifest",
    "collect_alfworld_functional_state_use",
]


def load_functional_state_use_benchmark(
    config_path: str | Path,
) -> ResolvedFunctionalStateUseBenchmark:
    path = Path(config_path).resolve()
    root = _workspace_root(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FunctionalStateUseError(
            f"cannot load functional state-use benchmark: {path}"
        ) from error
    if not isinstance(raw, Mapping):
        raise FunctionalStateUseError("functional state-use benchmark must be a mapping")
    expected_keys = {
        "schema",
        "id",
        "experiment_ref",
        "controlled_recipe_ref",
        "pair_type",
        "expected_pair_observability",
        "state_exposure_modes",
        "action_scoring",
        "execution",
        "claim_boundary",
    }
    if set(raw) != expected_keys or raw.get("schema") != (
        "lh_alfworld_functional_state_use_benchmark/v1"
    ):
        raise FunctionalStateUseError("functional state-use benchmark schema differs")
    modes = raw.get("state_exposure_modes")
    if modes != [mode.value for mode in ActorStateExposureMode]:
        raise FunctionalStateUseError("functional state-use exposure modes differ")
    if raw.get("pair_type") != "child_status_pair":
        raise FunctionalStateUseError("functional state-use pair type differs")
    expected_observability = str(raw.get("expected_pair_observability", ""))
    if expected_observability not in {
        "actor_prompt_distinguishable_candidate",
        "privileged_only_pair",
    }:
        raise FunctionalStateUseError("functional state-use expected pair observability differs")
    action_scoring = raw.get("action_scoring")
    if not isinstance(action_scoring, Mapping) or dict(action_scoring) != {
        "mode": "token_trie_constrained_candidate_distribution",
        "temperature": 1.0,
        "excluded_actions": ["help"],
        "target_action_source": ("shortest_successful_preregistered_environment_continuation"),
    }:
        raise FunctionalStateUseError("functional state-use action scoring differs")
    execution = raw.get("execution")
    if not isinstance(execution, Mapping) or dict(execution) != {
        "policy_update_allowed": False,
        "pair_count": 1,
        "require_same_current_observation": True,
        "require_same_admissible_actions": True,
        "require_distinct_target_actions": True,
    }:
        raise FunctionalStateUseError("functional state-use execution gate differs")
    claim = raw.get("claim_boundary")
    if not isinstance(claim, Mapping) or dict(claim) != {
        "functional_residual_state_use": False,
        "endogenous_awareness": False,
        "policy_improvement": False,
        "credit_assignment": False,
        "free_text_action_marginal": False,
    }:
        raise FunctionalStateUseError("functional state-use claim boundary differs")
    experiment_path = _workspace_reference(root, raw.get("experiment_ref"))
    recipe_path = _workspace_reference(root, raw.get("controlled_recipe_ref"))
    experiment = load_agentic_rl_experiment(experiment_path)
    recipe = load_alfworld_controlled_aliasing_recipe(recipe_path)
    if experiment.payload["runtime"]["mode"] != "controlled_diagnostic_only":
        raise FunctionalStateUseError("functional benchmark actor can execute updates")
    if experiment.payload["source"]["revision"] != recipe.source_revision:
        raise FunctionalStateUseError("functional benchmark source and recipe differ")
    payload = {
        "schema": raw["schema"],
        "id": str(raw["id"]),
        "experiment": {
            "path": experiment_path.relative_to(root).as_posix(),
            "sha256": experiment.sha256,
        },
        "controlled_recipe": {
            "path": recipe_path.relative_to(root).as_posix(),
            "sha256": recipe.sha256,
        },
        "pair_type": raw["pair_type"],
        "expected_pair_observability": expected_observability,
        "state_exposure_modes": list(modes),
        "action_scoring": dict(action_scoring),
        "execution": dict(execution),
        "claim_boundary": dict(claim),
    }
    return ResolvedFunctionalStateUseBenchmark(
        config_path=path,
        workspace_root=root,
        payload=payload,
        sha256=canonical_sha256(payload),
    )


def collect_alfworld_functional_state_use_manifest(
    recipe: AlfWorldControlledAliasingRecipe,
    *,
    task: AlfworldTask,
    task_metadata: Mapping[str, Any],
    instruction: str,
    branch_factory: BranchFactory,
    artifacts_dir: str | Path,
    state_exposure_modes: tuple[ActorStateExposureMode | str, ...],
    expected_pair_observability: str,
    collection_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect immutable environment evidence before loading a model."""

    if task.task_id != recipe.task_id:
        raise FunctionalStateUseError("functional state-use task differs from recipe")
    modes = tuple(ActorStateExposureMode(mode) for mode in state_exposure_modes)
    expected_modes = tuple(ActorStateExposureMode)
    if modes != expected_modes:
        raise FunctionalStateUseError(
            "functional state-use benchmark requires all preregistered exposure modes"
        )
    if len(recipe.branches) != 2:
        raise FunctionalStateUseError("functional state-use benchmark requires one pair")
    if not recipe.continuation_programs:
        raise FunctionalStateUseError(
            "functional state-use benchmark requires continuation programs"
        )

    root = Path(artifacts_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = G2POAlfWorldActorProtocol()
    compiler = AlfWorldSubproblemCompiler()
    graph = compiler.compile_task(task_metadata)
    states = tuple(
        _reconstruct_state(
            recipe,
            branch_recipe,
            task=task,
            task_metadata=task_metadata,
            branch_factory=branch_factory,
            artifacts_dir=root / "states" / branch_recipe.branch_id,
            protocol=protocol,
            compiler=compiler,
            graph=graph,
        )
        for branch_recipe in recipe.branches
    )
    _validate_state_pair(states)
    observability = (
        "actor_prompt_distinguishable_candidate"
        if len({state.normal_prompt for state in states}) == len(states)
        else "privileged_only_pair"
    )
    if observability != expected_pair_observability:
        raise FunctionalStateUseError(
            "reconstructed pair observability differs from preregistered benchmark"
        )

    state_rows: list[dict[str, Any]] = []
    target_actions_by_state: dict[str, tuple[str, ...]] = {}
    corrupted_history_by_state = {
        states[0].recipe.branch_id: states[1].history,
        states[1].recipe.branch_id: states[0].history,
    }
    for state in states:
        state_id = state.recipe.branch_id
        program_results = _evaluate_continuation_programs(
            recipe,
            state.recipe,
            task=task,
            task_metadata=task_metadata,
            branch_factory=branch_factory,
            artifacts_dir=root / "programs" / state_id,
            compiler=compiler,
            graph=graph,
            expected_state=state,
        )
        target_actions = _shortest_successful_first_actions(program_results)
        available_actions = tuple(
            action for action in state.observation.admissible_actions if action != "help"
        )
        if not set(target_actions) <= set(available_actions):
            raise FunctionalStateUseError(
                "shortest successful continuation starts outside admissible actions"
            )
        action_labels = _label_one_step_actions(
            recipe,
            state.recipe,
            task=task,
            task_metadata=task_metadata,
            branch_factory=branch_factory,
            artifacts_dir=root / "actions" / state_id,
            compiler=compiler,
            graph=graph,
            actions=available_actions,
            target_actions=target_actions,
            expected_state=state,
        )
        target_actions_by_state[state_id] = target_actions
        condition_rows: dict[str, Any] = {}
        for mode in modes:
            prompt, exposure = _render_condition_prompt(
                protocol,
                state,
                remaining_budget=recipe.remaining_budget,
                mode=mode,
                corrupted_history=corrupted_history_by_state[state_id],
            )
            condition_rows[mode.value] = {
                "state_exposure_mode": mode.value,
                "actor_prompt": prompt,
                "state_exposure": exposure,
            }
        state_rows.append(
            {
                "state_id": state_id,
                "raw_observation": state.observation.text,
                "admissible_actions": list(available_actions),
                **state.evidence.to_artifact_fields(),
                "state_binding_probe_action": recipe.probe_action,
                "active_frontier_ids": list(
                    goal_state_projection(graph, state.vector)["active_frontier_ids"]
                ),
                "target_actions": list(target_actions),
                "target_action_source": (
                    "shortest_successful_preregistered_environment_continuation"
                ),
                "continuation_programs": program_results,
                "action_labels": action_labels,
                "conditions": condition_rows,
            }
        )

    if len({tuple(actions) for actions in target_actions_by_state.values()}) != 2:
        raise FunctionalStateUseError("controlled pair does not require different target actions")
    target_sets = [set(actions) for actions in target_actions_by_state.values()]
    if target_sets[0] & target_sets[1]:
        raise FunctionalStateUseError("controlled pair target action sets must be disjoint")
    body = {
        "schema": FUNCTIONAL_STATE_USE_MANIFEST_SCHEMA,
        "recipe_sha256": recipe.sha256,
        "task_id": task.task_id,
        "instruction": str(instruction),
        "pair_type": "child_status_pair",
        "expected_pair_observability": expected_pair_observability,
        "pair_observability": observability,
        "state_exposure_modes": [mode.value for mode in modes],
        "states": state_rows,
        "collection_provenance": dict(collection_provenance or {}),
        "claim_scope": ("environment_manifest_only_no_model_load_or_policy_update"),
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def score_alfworld_functional_state_use_manifest(
    manifest: Mapping[str, Any],
    *,
    policy: FrozenActionScoringPolicy,
    sampling_temperature: float,
) -> dict[str, Any]:
    """Score a hash-bound environment manifest without environment access."""

    body = dict(manifest)
    declared_manifest_sha256 = str(body.pop("manifest_sha256", ""))
    if (
        body.get("schema") != FUNCTIONAL_STATE_USE_MANIFEST_SCHEMA
        or canonical_sha256(body) != declared_manifest_sha256
    ):
        raise FunctionalStateUseError("functional state-use manifest integrity check failed")
    if sampling_temperature <= 0:
        raise FunctionalStateUseError("sampling temperature must be positive")
    modes = tuple(ActorStateExposureMode(value) for value in body.get("state_exposure_modes", ()))
    if modes != tuple(ActorStateExposureMode):
        raise FunctionalStateUseError("functional state-use manifest exposure modes differ")
    raw_states = body.get("states")
    if not isinstance(raw_states, list) or len(raw_states) != 2:
        raise FunctionalStateUseError("functional state-use manifest requires one state pair")

    checkpoint_before = policy.checkpoint_sha256
    state_rows = deepcopy(raw_states)
    distributions: dict[str, dict[str, Mapping[str, Any]]] = {}
    target_actions_by_state: dict[str, tuple[str, ...]] = {}
    for state in state_rows:
        if not isinstance(state, dict):
            raise FunctionalStateUseError("functional manifest state is invalid")
        state_id = str(state.get("state_id", ""))
        available_actions = tuple(str(action) for action in state.get("admissible_actions", ()))
        target_actions = tuple(str(action) for action in state.get("target_actions", ()))
        action_labels = state.get("action_labels")
        conditions = state.get("conditions")
        if (
            not state_id
            or not available_actions
            or not target_actions
            or not isinstance(action_labels, Mapping)
            or not isinstance(conditions, dict)
            or set(conditions) != {mode.value for mode in modes}
        ):
            raise FunctionalStateUseError("functional state-use manifest state contract differs")
        target_actions_by_state[state_id] = target_actions
        distributions[state_id] = {}
        for mode in modes:
            condition = conditions[mode.value]
            if not isinstance(condition, dict):
                raise FunctionalStateUseError("functional manifest prompt condition is invalid")
            prompt = str(condition.get("actor_prompt", ""))
            if not prompt:
                raise FunctionalStateUseError("functional manifest prompt is missing")
            distribution = dict(
                policy.score_admissible_actions(
                    prompt,
                    available_actions,
                    temperature=sampling_temperature,
                )
            )
            _validate_distribution(
                distribution,
                expected_checkpoint=checkpoint_before,
                expected_actions=available_actions,
            )
            distributions[state_id][mode.value] = distribution
            probabilities = _probability_map(distribution)
            condition.update(
                {
                    "distribution": distribution,
                    "target_action_mass": _mass(probabilities, target_actions),
                    "one_step_progress_mass": _mass(
                        probabilities,
                        tuple(
                            action
                            for action, label in action_labels.items()
                            if isinstance(label, Mapping) and label.get("one_step_progress")
                        ),
                    ),
                    "regression_mass": _mass(
                        probabilities,
                        tuple(
                            action
                            for action, label in action_labels.items()
                            if isinstance(label, Mapping) and label.get("regresses_goal")
                        ),
                    ),
                    "no_detected_regression_or_immediate_failure_mass": _mass(
                        probabilities,
                        tuple(
                            action
                            for action, label in action_labels.items()
                            if isinstance(label, Mapping)
                            and label.get("no_detected_goal_regression_or_immediate_failure")
                        ),
                    ),
                    "top_action": max(probabilities, key=probabilities.get),
                }
            )

    if len(target_actions_by_state) != 2:
        raise FunctionalStateUseError("functional manifest state IDs are not unique")
    pair_metrics = _pair_metrics(distributions, target_actions_by_state, modes)
    normal_prompts = [state["conditions"]["normal"]["actor_prompt"] for state in state_rows]
    local_prompts = [state["conditions"]["local_only"]["actor_prompt"] for state in state_rows]
    corrupted_history_prompts = [
        state["conditions"]["corrupted_history"]["actor_prompt"] for state in state_rows
    ]
    prompt_factor_control_gates = {
        "local_only_pair_prompt_exact_match": len(set(local_prompts)) == 1,
        "normal_pair_prompt_distinct": len(set(normal_prompts)) == len(normal_prompts),
    }
    prompt_factor_control_gates["normal_vs_local_history_isolated_candidate"] = (
        prompt_factor_control_gates["local_only_pair_prompt_exact_match"]
        and prompt_factor_control_gates["normal_pair_prompt_distinct"]
    )
    pair_prompt_token_delta_by_state_exposure = {
        mode.value: (
            int(state_rows[0]["conditions"][mode.value]["distribution"]["prompt_token_count"])
            - int(state_rows[1]["conditions"][mode.value]["distribution"]["prompt_token_count"])
        )
        for mode in modes
    }
    prompt_factor_control_gates["normal_pair_exact_token_match"] = (
        pair_prompt_token_delta_by_state_exposure[ActorStateExposureMode.NORMAL.value] == 0
    )
    normal_token_counts = sorted(
        int(state["conditions"]["normal"]["distribution"]["prompt_token_count"])
        for state in state_rows
    )
    corrupted_history_token_counts = sorted(
        int(state["conditions"]["corrupted_history"]["distribution"]["prompt_token_count"])
        for state in state_rows
    )
    prompt_factor_control_gates["history_sham_control_present"] = all(
        normal != corrupted
        for normal, corrupted in zip(
            normal_prompts,
            corrupted_history_prompts,
            strict=True,
        )
    )
    prompt_factor_control_gates["normal_vs_corrupted_history_pair_token_multiset_match"] = (
        normal_token_counts == corrupted_history_token_counts
    )
    prompt_length_controls = {
        state["state_id"]: {
            "graph_vs_flat_token_delta": (
                state["conditions"]["graph_oracle"]["distribution"]["prompt_token_count"]
                - state["conditions"]["flat_oracle"]["distribution"]["prompt_token_count"]
            ),
            "correct_vs_corrupted_graph_token_delta": (
                state["conditions"]["graph_oracle"]["distribution"]["prompt_token_count"]
                - state["conditions"]["corrupted_graph"]["distribution"]["prompt_token_count"]
            ),
        }
        for state in state_rows
    }
    prompt_length_control_gates = {
        "correct_vs_corrupted_graph_exact_token_match": all(
            row["correct_vs_corrupted_graph_token_delta"] == 0
            for row in prompt_length_controls.values()
        ),
        "graph_vs_flat_exact_token_match": all(
            row["graph_vs_flat_token_delta"] == 0 for row in prompt_length_controls.values()
        ),
    }
    sensitivities = {
        mode.value: float(pair_metrics[mode.value]["pair_sensitivity"]) for mode in modes
    }
    behavioral_contrasts = {
        "normal_over_local_only_pair_sensitivity": (
            sensitivities[ActorStateExposureMode.NORMAL.value]
            - sensitivities[ActorStateExposureMode.LOCAL_ONLY.value]
        ),
        "normal_over_corrupted_history_pair_sensitivity": (
            sensitivities[ActorStateExposureMode.NORMAL.value]
            - sensitivities[ActorStateExposureMode.CORRUPTED_HISTORY.value]
        ),
        "flat_oracle_over_normal_pair_sensitivity": (
            sensitivities[ActorStateExposureMode.FLAT_ORACLE.value]
            - sensitivities[ActorStateExposureMode.NORMAL.value]
        ),
        "graph_oracle_over_flat_pair_sensitivity": (
            sensitivities[ActorStateExposureMode.GRAPH_ORACLE.value]
            - sensitivities[ActorStateExposureMode.FLAT_ORACLE.value]
        ),
        "correct_over_corrupted_graph_pair_sensitivity": (
            sensitivities[ActorStateExposureMode.GRAPH_ORACLE.value]
            - sensitivities[ActorStateExposureMode.CORRUPTED_GRAPH.value]
        ),
    }
    graph_edge_identifiable = prompt_length_control_gates["graph_vs_flat_exact_token_match"]
    payload_correctness_identifiable = prompt_length_control_gates[
        "correct_vs_corrupted_graph_exact_token_match"
    ]
    independent_environment_semantics_verified = all(
        state["semantic_graph_verification"]["verified"]
        and state["semantic_snapshot_verification"]["verified"]
        for state in state_rows
    )
    positive_normal_prompt_sensitivity = (
        prompt_factor_control_gates["normal_pair_prompt_distinct"]
        and sensitivities[ActorStateExposureMode.NORMAL.value] > 0.0
    )
    positive_history_contribution = (
        prompt_factor_control_gates["normal_vs_local_history_isolated_candidate"]
        and behavioral_contrasts["normal_over_local_only_pair_sensitivity"] > 0.0
    )
    history_content_effect_identifiable = (
        prompt_factor_control_gates["history_sham_control_present"]
        and prompt_factor_control_gates["normal_vs_corrupted_history_pair_token_multiset_match"]
    )
    positive_history_content_effect = (
        history_content_effect_identifiable
        and behavioral_contrasts["normal_over_corrupted_history_pair_sensitivity"] > 0.0
    )
    functional_test_identifiable = (
        independent_environment_semantics_verified
        and body["pair_observability"] == "actor_prompt_distinguishable_candidate"
        and history_content_effect_identifiable
    )
    evidence_gates = {
        "independent_environment_semantics_verified": (independent_environment_semantics_verified),
        "positive_normal_prompt_sensitivity": (positive_normal_prompt_sensitivity),
        "positive_history_contribution": positive_history_contribution,
        "history_content_effect_identifiable": (history_content_effect_identifiable),
        "positive_history_content_effect": positive_history_content_effect,
        "functional_residual_state_use_test_identifiable": (functional_test_identifiable),
        "functional_residual_state_use_supported_on_preregistered_pair": (
            functional_test_identifiable
            and positive_normal_prompt_sensitivity
            and positive_history_contribution
            and positive_history_content_effect
        ),
        "graph_edge_effect_identifiable": graph_edge_identifiable,
        "positive_graph_edge_effect": (
            graph_edge_identifiable
            and behavioral_contrasts["graph_oracle_over_flat_pair_sensitivity"] > 0.0
        ),
        "oracle_payload_correctness_effect_identifiable": (payload_correctness_identifiable),
        "positive_oracle_payload_correctness_effect": (
            payload_correctness_identifiable
            and behavioral_contrasts["correct_over_corrupted_graph_pair_sensitivity"] > 0.0
        ),
    }
    checkpoint_after = policy.checkpoint_sha256
    if checkpoint_after != checkpoint_before:
        raise FunctionalStateUseError("functional state-use scoring mutated policy")
    return {
        "schema": FUNCTIONAL_STATE_USE_SCHEMA,
        "manifest_sha256": declared_manifest_sha256,
        "recipe_sha256": body["recipe_sha256"],
        "task_id": body["task_id"],
        "instruction": body["instruction"],
        "pair_type": body["pair_type"],
        "expected_pair_observability": body["expected_pair_observability"],
        "pair_observability": body["pair_observability"],
        "state_exposure_modes": [mode.value for mode in modes],
        "action_distribution_semantics": (
            "token_trie_constrained_candidate_choice_not_free_text_marginal"
        ),
        "states": state_rows,
        "pair_metrics_by_state_exposure": pair_metrics,
        "prompt_length_controls": prompt_length_controls,
        "prompt_length_control_gates": prompt_length_control_gates,
        "prompt_factor_control_gates": prompt_factor_control_gates,
        "pair_prompt_token_delta_by_state_exposure": (pair_prompt_token_delta_by_state_exposure),
        "behavioral_contrasts": behavioral_contrasts,
        "evidence_gates": evidence_gates,
        "policy_checkpoint_before_sha256": checkpoint_before,
        "policy_checkpoint_after_sha256": checkpoint_after,
        "policy_update_executed": False,
        "claim_scope": FUNCTIONAL_STATE_USE_CLAIM_SCOPE,
    }


def collect_alfworld_functional_state_use(
    recipe: AlfWorldControlledAliasingRecipe,
    *,
    task: AlfworldTask,
    task_metadata: Mapping[str, Any],
    instruction: str,
    policy: FrozenActionScoringPolicy,
    branch_factory: BranchFactory,
    artifacts_dir: str | Path,
    state_exposure_modes: tuple[ActorStateExposureMode | str, ...],
    sampling_temperature: float,
    expected_pair_observability: str,
) -> dict[str, Any]:
    """Backward-compatible in-process collection followed by frozen scoring."""

    manifest = collect_alfworld_functional_state_use_manifest(
        recipe,
        task=task,
        task_metadata=task_metadata,
        instruction=instruction,
        branch_factory=branch_factory,
        artifacts_dir=artifacts_dir,
        state_exposure_modes=state_exposure_modes,
        expected_pair_observability=expected_pair_observability,
        collection_provenance=None,
    )
    return score_alfworld_functional_state_use_manifest(
        manifest,
        policy=policy,
        sampling_temperature=sampling_temperature,
    )


def _workspace_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "configs").is_dir() and (
            candidate / "lh_platform" / "pyproject.toml"
        ).is_file():
            return candidate.resolve()
    raise FunctionalStateUseError("cannot resolve functional benchmark workspace root")


def _workspace_reference(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FunctionalStateUseError("functional benchmark reference is missing")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FunctionalStateUseError("functional benchmark reference is invalid")
    return path
