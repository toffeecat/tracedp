"""Capture benchmark-owned state evidence at first out-of-menu proposals."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.tracing.alfworld_continuation_anchors import (
    AlfWorldContinuationAnchorV1,
)
from lh_platform.tracing.alfworld_controlled_fault_injection import (
    AlfWorldControlledFaultInjectionPlanV1,
    ControlledFaultInjectionArmV1,
)
from lh_platform.training.alfworld_agentic_suite import AlfWorldBranch

ALFWORLD_PRECONDITION_FAILURE_REPLAY_SCHEMA = "lh_alfworld_precondition_failure_replay/v1"
ALFWORLD_CONTINUATION_ANCHOR_PRECONDITION_FAILURE_REPLAY_SCHEMA = (
    "lh_alfworld_precondition_failure_replay/v2"
)
ALFWORLD_CONTROLLED_TERMINAL_PRECONDITION_FAILURE_REPLAY_SCHEMA = (
    "lh_alfworld_controlled_terminal_precondition_failure_replay/v1"
)


class AlfWorldPreconditionReplayError(ValueError):
    """Raised when an out-of-menu replay is not bound to its frozen source."""


def _require_mapping(label: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlfWorldPreconditionReplayError(f"{label} must be an object")
    return value


def _validate_actor_step(raw_step: object) -> tuple[Mapping[str, Any], str, tuple[str, ...]]:
    step = _require_mapping("T_raw actor step", raw_step)
    action = step.get("environment_action")
    raw_menu = step.get("live_admissible_actions")
    if (
        step.get("action_selection_mode") != "free_generation"
        or not isinstance(action, str)
        or not action
        or not isinstance(raw_menu, list)
        or not raw_menu
        or any(not isinstance(value, str) or not value for value in raw_menu)
        or len(raw_menu) != len(set(raw_menu))
        or step.get("live_admissible_actions_sha256") != "sha256:" + canonical_sha256(raw_menu)
        or step.get("action_in_live_admissible_menu") is not (action in raw_menu)
    ):
        raise AlfWorldPreconditionReplayError("T_raw actor-step menu binding changed")
    return step, action, tuple(raw_menu)


def _direct_step(branch: AlfWorldBranch, action: str) -> Any:
    direct = getattr(branch, "step_unverified", None)
    return direct(action) if callable(direct) else branch.step(action)


def _state_payload(
    *,
    observation: Any,
    state_snapshot_sha256: str,
) -> dict[str, Any]:
    # TextWorld exposes logical facts through a set-backed container. Their
    # iteration order can differ across processes even when the restored state
    # snapshot is identical, so evidence identity must not depend on that order.
    facts = sorted(
        (dict(value) for value in observation.facts),
        key=canonical_sha256,
    )
    menu = list(observation.admissible_actions)
    return {
        "state_snapshot_sha256": state_snapshot_sha256,
        "observation": observation.text,
        "facts": facts,
        "facts_sha256": "sha256:" + canonical_sha256(facts),
        "live_admissible_actions": menu,
        "live_admissible_actions_sha256": "sha256:" + canonical_sha256(menu),
        "gamefile": observation.gamefile,
    }


def collect_alfworld_precondition_failure_replay(
    *,
    plan: AlfWorldControlledFaultInjectionPlanV1,
    branch_factory: Callable[[], AlfWorldBranch],
    raw_arm: Mapping[str, Any],
    expected_sampling_seed: int,
) -> dict[str, Any]:
    """Replay to the first raw out-of-menu step and capture exact pre/post state.

    The receipt deliberately stops before assigning a failure label. Facts and
    exact snapshots are benchmark-owned evidence; a later classifier may derive
    precondition labels from them without using terminal continuation outcomes.
    """

    return _collect_alfworld_precondition_failure_replay(
        source_steps=plan.steps,
        target_step_idx=plan.injection.step_idx,
        source_action=plan.injection.source_action,
        source_pre_state_snapshot_sha256=(plan.injection.source_pre_state_snapshot_sha256),
        task_id=plan.task_id,
        traj_uid=plan.traj_uid,
        receipt_schema=ALFWORLD_PRECONDITION_FAILURE_REPLAY_SCHEMA,
        receipt_identity={"injection_plan_sha256": plan.sha256},
        branch_factory=branch_factory,
        raw_arm=raw_arm,
        expected_sampling_seed=expected_sampling_seed,
    )


def collect_alfworld_continuation_anchor_precondition_failure_replay(
    *,
    anchor: AlfWorldContinuationAnchorV1,
    branch_factory: Callable[[], AlfWorldBranch],
    raw_arm: Mapping[str, Any],
    expected_sampling_seed: int,
) -> dict[str, Any]:
    """Replay an unexecuted proposal from one failed-source continuation anchor."""

    return _collect_alfworld_precondition_failure_replay(
        source_steps=anchor.steps,
        target_step_idx=anchor.anchor_step_idx,
        source_action=anchor.anchor_source_action,
        source_pre_state_snapshot_sha256=anchor.anchor_pre_state_snapshot_sha256,
        task_id=anchor.task_id,
        traj_uid=anchor.traj_uid,
        receipt_schema=(ALFWORLD_CONTINUATION_ANCHOR_PRECONDITION_FAILURE_REPLAY_SCHEMA),
        receipt_identity={
            "continuation_anchor_id": anchor.anchor_id,
            "continuation_anchor_sha256": anchor.sha256,
            "source_trajectory_success": anchor.source_trajectory_success,
            "source_terminal_reason": anchor.source_terminal_reason,
        },
        branch_factory=branch_factory,
        raw_arm=raw_arm,
        expected_sampling_seed=expected_sampling_seed,
    )


def collect_alfworld_controlled_terminal_precondition_failure_replay(
    *,
    plan: AlfWorldControlledFaultInjectionPlanV1,
    failure_arm: ControlledFaultInjectionArmV1,
    branch_factory: Callable[[], AlfWorldBranch],
) -> dict[str, Any]:
    """Verify the first rejected source-suffix action after a controlled fault.

    The receipt exposes the rejected action and exact terminal state, but not an
    event id for where the rejection was observed or where the fault was injected.
    This keeps terminal failure semantics separate from historical localization.
    """

    if (
        failure_arm.arm != "injected_failure"
        or failure_arm.success
        or failure_arm.terminal_reason != "suffix_action_inadmissible"
        or not failure_arm.pre_state_identity_verified
        or failure_arm.suffix_controller_sha256 != plan.suffix_controller_sha256
        or not failure_arm.steps
        or len(failure_arm.steps) >= len(plan.steps)
    ):
        raise AlfWorldPreconditionReplayError(
            "controlled terminal replay requires a bound suffix rejection"
        )
    if tuple(step.step_idx for step in failure_arm.steps) != tuple(
        range(len(failure_arm.steps))
    ):
        raise AlfWorldPreconditionReplayError(
            "controlled failure steps are not contiguous"
        )

    proposal_step_idx = len(failure_arm.steps)
    proposal = plan.steps[proposal_step_idx].source_action
    branch = branch_factory()
    try:
        current = branch.reset()
        for expected in failure_arm.steps:
            checkpoint = branch.capture_checkpoint()
            if (
                checkpoint.snapshot.state_snapshot_sha256
                != expected.pre_state_snapshot_sha256
                or tuple(current.admissible_actions) != expected.admissible_actions
                or current.text != expected.observation_before
                or (expected.action in current.admissible_actions)
                is not expected.action_admissible
            ):
                raise AlfWorldPreconditionReplayError(
                    f"controlled failure identity changed at step {expected.step_idx}"
                )
            current = _direct_step(branch, expected.action).observation
            post_checkpoint = branch.capture_checkpoint()
            if (
                post_checkpoint.snapshot.state_snapshot_sha256
                != expected.post_state_snapshot_sha256
                or current.text != expected.observation_after
            ):
                raise AlfWorldPreconditionReplayError(
                    f"controlled failure outcome changed at step {expected.step_idx}"
                )

        pre_checkpoint = branch.capture_checkpoint()
        if (
            pre_checkpoint.snapshot.state_snapshot_sha256
            != failure_arm.final_state_snapshot_sha256
            or proposal in current.admissible_actions
        ):
            raise AlfWorldPreconditionReplayError(
                "controlled terminal rejection state changed"
            )
        pre_state = _state_payload(
            observation=current,
            state_snapshot_sha256=pre_checkpoint.snapshot.state_snapshot_sha256,
        )
        proposal_step = branch.step(proposal)
        post_checkpoint = branch.capture_checkpoint()
        if (
            proposal_step.state_snapshot_sha256
            != pre_checkpoint.snapshot.state_snapshot_sha256
            or proposal_step.post_state_snapshot_sha256
            != post_checkpoint.snapshot.state_snapshot_sha256
            or not proposal_step.restore_receipt_sha256
            or proposal_step.replay_outcome != proposal_step.environment_outcome
        ):
            raise AlfWorldPreconditionReplayError(
                "controlled terminal proposal restore/replay changed"
            )
        post_state = _state_payload(
            observation=proposal_step.observation,
            state_snapshot_sha256=post_checkpoint.snapshot.state_snapshot_sha256,
        )
        facts_unchanged = pre_state["facts_sha256"] == post_state["facts_sha256"]
        menu_unchanged = (
            pre_state["live_admissible_actions_sha256"]
            == post_state["live_admissible_actions_sha256"]
        )
        replayed_action_prefix = list(failure_arm.executed_actions)
        receipt: dict[str, Any] = {
            "schema": ALFWORLD_CONTROLLED_TERMINAL_PRECONDITION_FAILURE_REPLAY_SCHEMA,
            "task_id": plan.task_id,
            "traj_uid": plan.traj_uid,
            "source_failure_trace_sha256": failure_arm.trajectory_sha256,
            "plan_sha256": plan.sha256,
            "sampling_seed_base": plan.sampling_seed_base,
            "selection_rule": "first_frozen_source_suffix_action_outside_terminal_live_menu",
            "selection_uses_pair_outcome": False,
            "pair_outcome_used": False,
            "terminal_anchor_included": False,
            "source_step_idx": proposal_step_idx,
            "proposal": proposal,
            "proposal_in_pre_state_live_menu": False,
            "replayed_action_prefix": replayed_action_prefix,
            "replayed_action_prefix_sha256": "sha256:"
            + canonical_sha256(replayed_action_prefix),
            "pre_state": pre_state,
            "post_state": post_state,
            "benchmark_facts_unchanged": facts_unchanged,
            "live_menu_unchanged": menu_unchanged,
            "replay_state_identity_unchanged": (
                pre_state["state_snapshot_sha256"]
                == post_state["state_snapshot_sha256"]
            ),
            "environment_rejection_observed": bool(
                facts_unchanged
                and menu_unchanged
                and proposal_step.observation.text == "Nothing happens."
                and not proposal_step.won
                and not proposal_step.done
            ),
            "proposal_step_outcome": {
                "reward": proposal_step.reward,
                "won": proposal_step.won,
                "done": proposal_step.done,
                "environment_outcome": proposal_step.environment_outcome,
                "replay_outcome": proposal_step.replay_outcome,
                "pre_state_snapshot_sha256": proposal_step.state_snapshot_sha256,
                "post_state_snapshot_sha256": (
                    proposal_step.post_state_snapshot_sha256
                ),
                "restore_receipt_sha256": proposal_step.restore_receipt_sha256,
            },
            "exact_transition_restore_replay_verified": True,
            "physical_call_counts": dict(branch.physical_call_counters()),
            "precondition_gold_assigned": False,
            "tracer_accuracy_tested": False,
            "continuation_success_tested": False,
        }
        receipt["content_sha256"] = "sha256:" + canonical_sha256(receipt)
        return receipt
    finally:
        branch.close()


def _collect_alfworld_precondition_failure_replay(
    *,
    source_steps: tuple[Any, ...],
    target_step_idx: int,
    source_action: str,
    source_pre_state_snapshot_sha256: str,
    task_id: str,
    traj_uid: str,
    receipt_schema: str,
    receipt_identity: Mapping[str, Any],
    branch_factory: Callable[[], AlfWorldBranch],
    raw_arm: Mapping[str, Any],
    expected_sampling_seed: int,
) -> dict[str, Any]:
    """Shared exact-state transition verifier for frozen source prefixes."""

    if (
        isinstance(expected_sampling_seed, bool)
        or not isinstance(expected_sampling_seed, int)
        or expected_sampling_seed < 0
        or raw_arm.get("arm") != "T_raw"
        or raw_arm.get("sampling_seed_base") != expected_sampling_seed
    ):
        raise AlfWorldPreconditionReplayError("T_raw sampling identity changed")
    raw_actor_steps = raw_arm.get("actor_steps")
    raw_executed_actions = raw_arm.get("executed_actions")
    if (
        not isinstance(raw_actor_steps, list)
        or not raw_actor_steps
        or not isinstance(raw_executed_actions, list)
        or any(not isinstance(value, str) or not value for value in raw_executed_actions)
    ):
        raise AlfWorldPreconditionReplayError("T_raw action coverage changed")
    actor_steps = tuple(_validate_actor_step(value) for value in raw_actor_steps)
    prefix_action_count = target_step_idx + 1
    if (
        tuple(step.step_idx for step in source_steps) != tuple(range(len(source_steps)))
        or target_step_idx < 0
        or target_step_idx >= len(source_steps)
        or tuple(raw_executed_actions[:target_step_idx])
        != tuple(step.source_action for step in source_steps[:target_step_idx])
        or raw_executed_actions[target_step_idx] != source_action
        or source_steps[target_step_idx].source_action != source_action
    ):
        raise AlfWorldPreconditionReplayError("T_raw executed/actor action binding changed")

    first_out_of_menu_ordinal = next(
        (ordinal for ordinal, (_, action, menu) in enumerate(actor_steps) if action not in menu),
        None,
    )
    if first_out_of_menu_ordinal is None:
        raise AlfWorldPreconditionReplayError("T_raw has no out-of-menu proposal")
    prior_actor_actions = tuple(action for _, action, _ in actor_steps[:first_out_of_menu_ordinal])
    if (
        len(raw_executed_actions) != prefix_action_count + first_out_of_menu_ordinal
        or tuple(raw_executed_actions[prefix_action_count:]) != prior_actor_actions
    ):
        raise AlfWorldPreconditionReplayError("T_raw executed/actor action binding changed")
    if first_out_of_menu_ordinal != len(actor_steps) - 1 or any(
        action not in menu for _, action, menu in actor_steps[:first_out_of_menu_ordinal]
    ):
        raise AlfWorldPreconditionReplayError("first out-of-menu selection changed")

    branch = branch_factory()
    try:
        current = branch.reset()
        for source_step in source_steps[:target_step_idx]:
            checkpoint = branch.capture_checkpoint()
            if (
                checkpoint.snapshot.state_snapshot_sha256 != source_step.state_snapshot_sha256
                or tuple(current.admissible_actions) != source_step.admissible_actions
            ):
                raise AlfWorldPreconditionReplayError(
                    f"source prefix identity changed at step {source_step.step_idx}"
                )
            current = _direct_step(branch, source_step.source_action).observation

        target_source_step = source_steps[target_step_idx]
        target_checkpoint = branch.capture_checkpoint()
        if (
            target_checkpoint.snapshot.state_snapshot_sha256 != source_pre_state_snapshot_sha256
            or target_checkpoint.snapshot.state_snapshot_sha256
            != target_source_step.state_snapshot_sha256
            or tuple(current.admissible_actions) != target_source_step.admissible_actions
            or source_action not in current.admissible_actions
        ):
            raise AlfWorldPreconditionReplayError("intervention source identity changed")
        current = _direct_step(branch, source_action).observation

        for ordinal, (_, action, menu) in enumerate(actor_steps[:first_out_of_menu_ordinal]):
            if tuple(current.admissible_actions) != menu or action not in menu:
                raise AlfWorldPreconditionReplayError(
                    f"replayed actor state/menu changed at ordinal {ordinal}"
                )
            current = _direct_step(branch, action).observation

        source_step, proposal, source_menu = actor_steps[first_out_of_menu_ordinal]
        if tuple(current.admissible_actions) != source_menu or proposal in source_menu:
            raise AlfWorldPreconditionReplayError("first out-of-menu replay state changed")
        pre_checkpoint = branch.capture_checkpoint()
        pre_state = _state_payload(
            observation=current,
            state_snapshot_sha256=pre_checkpoint.snapshot.state_snapshot_sha256,
        )
        # The diagnostic transition is verified by the branch's exact-state
        # restore/replay contract. Prefix actions stay unverified because they
        # are already bound to the frozen source receipts above.
        proposal_step = branch.step(proposal)
        post_checkpoint = branch.capture_checkpoint()
        if (
            proposal_step.state_snapshot_sha256 != pre_checkpoint.snapshot.state_snapshot_sha256
            or proposal_step.post_state_snapshot_sha256
            != post_checkpoint.snapshot.state_snapshot_sha256
            or not proposal_step.restore_receipt_sha256
            or proposal_step.replay_outcome != proposal_step.environment_outcome
        ):
            raise AlfWorldPreconditionReplayError(
                "proposal transition restore/replay verification changed"
            )
        post_state = _state_payload(
            observation=proposal_step.observation,
            state_snapshot_sha256=post_checkpoint.snapshot.state_snapshot_sha256,
        )
        facts_unchanged = pre_state["facts_sha256"] == post_state["facts_sha256"]
        menu_unchanged = (
            pre_state["live_admissible_actions_sha256"]
            == post_state["live_admissible_actions_sha256"]
        )
        snapshot_unchanged = (
            pre_state["state_snapshot_sha256"] == post_state["state_snapshot_sha256"]
        )
        replayed_action_prefix = list(
            raw_executed_actions[: prefix_action_count + first_out_of_menu_ordinal]
        )
        receipt: dict[str, Any] = {
            "schema": receipt_schema,
            "task_id": task_id,
            "traj_uid": traj_uid,
            **dict(receipt_identity),
            "sampling_seed_base": expected_sampling_seed,
            "selection_rule": "first_T_raw_step_outside_its_recorded_live_menu",
            "selection_uses_pair_outcome": False,
            "source_actor_step_ordinal": first_out_of_menu_ordinal,
            "source_step_idx": source_step.get("step_idx"),
            "proposal": proposal,
            "proposal_in_pre_state_live_menu": False,
            "replayed_action_prefix": replayed_action_prefix,
            "replayed_action_prefix_sha256": "sha256:" + canonical_sha256(replayed_action_prefix),
            "pre_state": pre_state,
            "post_state": post_state,
            "benchmark_facts_unchanged": facts_unchanged,
            "live_menu_unchanged": menu_unchanged,
            "replay_state_identity_unchanged": snapshot_unchanged,
            "environment_rejection_observed": bool(
                facts_unchanged
                and menu_unchanged
                and not proposal_step.won
                and not proposal_step.done
            ),
            "proposal_step_outcome": {
                "reward": proposal_step.reward,
                "won": proposal_step.won,
                "done": proposal_step.done,
                "environment_outcome": proposal_step.environment_outcome,
                "replay_outcome": proposal_step.replay_outcome,
                "pre_state_snapshot_sha256": proposal_step.state_snapshot_sha256,
                "post_state_snapshot_sha256": (proposal_step.post_state_snapshot_sha256),
                "restore_receipt_sha256": proposal_step.restore_receipt_sha256,
            },
            "exact_transition_restore_replay_verified": True,
            "physical_call_counts": dict(branch.physical_call_counters()),
            "precondition_gold_assigned": False,
            "tracer_accuracy_tested": False,
            "continuation_success_tested": False,
        }
        receipt["content_sha256"] = "sha256:" + canonical_sha256(receipt)
        return receipt
    finally:
        branch.close()


__all__ = [
    "ALFWORLD_CONTINUATION_ANCHOR_PRECONDITION_FAILURE_REPLAY_SCHEMA",
    "ALFWORLD_CONTROLLED_TERMINAL_PRECONDITION_FAILURE_REPLAY_SCHEMA",
    "ALFWORLD_PRECONDITION_FAILURE_REPLAY_SCHEMA",
    "AlfWorldPreconditionReplayError",
    "collect_alfworld_continuation_anchor_precondition_failure_replay",
    "collect_alfworld_controlled_terminal_precondition_failure_replay",
    "collect_alfworld_precondition_failure_replay",
]
