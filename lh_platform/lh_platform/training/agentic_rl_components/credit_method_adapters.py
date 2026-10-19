"""RLMethod adapters: clipped-PG credit arms behind the G2PO actor seam.

Satisfies ``agentic_rl_runtime.RLMethod``. Decomposition annotate adapters must
not live here — changing credit must not open ALFWorld/AppWorld annotate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Literal

from lh_platform.training.agentic_rl_components.errors import AgenticRLComponentError
from lh_platform.training.agentic_rl_components.identity import builtin_component
from lh_platform.training.agentic_rl_runtime import PolicyTarget, PreparedPolicyUpdate
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.credit_estimators import (
    CreditEstimator,
    GroupedMeanConfig,
    GroupedMeanEstimator,
    PhasePoolConfig,
    PhasePoolEstimator,
    credit_visits_from_trajectory_records,
)
from lh_platform.training.credit_estimators.g2po_bridge import (
    assign_estimator_g2po_credit,
)
from lh_platform.training.dppo.credit import (
    DPPOCreditMode,
)
from lh_platform.training.dppo.credit import (
    transitions_from_v2_records as dppo_transitions_from_v2_records,
)
from lh_platform.training.g2po.credit import (
    SUCCESS_REWARD,
    G2POCreditConfig,
    G2POCreditMode,
)
from lh_platform.training.g2po.credit import (
    transitions_from_v2_records as g2po_transitions_from_v2_records,
)
from lh_platform.training.unified_credit import (
    UnifiedCreditTreatment,
    assign_unified_credit,
)


class UnifiedG2POMethod:
    """One fixed clipped-PG method with replaceable G2PO-family credit."""

    def __init__(
        self,
        treatment: UnifiedCreditTreatment | str,
        *,
        config: G2POCreditConfig | None = None,
        source_contract_sha256: str | None = None,
    ) -> None:
        self.treatment = UnifiedCreditTreatment(treatment)
        self.config = config or G2POCreditConfig()
        if source_contract_sha256 is not None and (
            len(source_contract_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_contract_sha256)
        ):
            raise AgenticRLComponentError(
                "G2PO source parity contract must be a lowercase SHA-256 digest"
            )
        shared_config = {
            "objective": "clipped_policy_gradient",
            "actor_mass": "trajectory_step_token_unit_mass",
            "credit_config": asdict(self.config),
            "source_credit_parity_contract_sha256": source_contract_sha256,
        }
        self.component = builtin_component(
            role="rl_method",
            name="g2po_family_clipped_pg",
            config=shared_config,
        )
        self.credit_assigner = builtin_component(
            role="credit_assigner",
            name=self.treatment.value,
            config={**shared_config, "treatment": self.treatment.value},
        )

    def prepare_update(
        self,
        records: tuple[TrainingTrajectoryRecord, ...],
    ) -> PreparedPolicyUpdate:
        g2po_mode = {
            UnifiedCreditTreatment.GRPO_U: G2POCreditMode.SOURCE,
            UnifiedCreditTreatment.G2PO_U: G2POCreditMode.SOURCE,
            UnifiedCreditTreatment.G2PO_TN: G2POCreditMode.TRAJECTORY_NORMALIZED,
        }.get(self.treatment)
        if g2po_mode is not None:
            transitions = g2po_transitions_from_v2_records(records, mode=g2po_mode)
        else:
            dppo_mode = {
                UnifiedCreditTreatment.VERIFIED_STATE_DP: DPPOCreditMode.VERIFIED_STATE,
                UnifiedCreditTreatment.ORACLE_PHASE_REFINED_G2PO_TN: (
                    DPPOCreditMode.ORACLE_PHASE_REFINED_TN
                ),
            }[self.treatment]
            transitions = dppo_transitions_from_v2_records(
                records,
                mode=dppo_mode,
            )
        assignments = assign_unified_credit(
            transitions,
            treatment=self.treatment,
            g2po_config=self.config,
        )
        targets = tuple(
            PolicyTarget(
                traj_uid=record.traj_uid,
                step_idx=record.step_idx,
                advantage=assignment.advantage,
                actor_token_weight=assignment.actor_token_weight,
            )
            for record, assignment in zip(records, assignments, strict=True)
        )
        diagnostics: dict[str, object] = {
            "treatment": self.treatment.value,
            "transition_count": len(transitions),
        }
        if self.treatment is UnifiedCreditTreatment.ORACLE_PHASE_REFINED_G2PO_TN:
            from lh_platform.training.subproblem_study import assess_oracle_subproblem_trace

            diagnostics["oracle_subproblem_trace_gate"] = asdict(
                assess_oracle_subproblem_trace(records)
            )
        return PreparedPolicyUpdate(
            method=self.component,
            credit_assigner=self.credit_assigner,
            objective="clipped_policy_gradient",
            targets=targets,
            diagnostics=diagnostics,
        )


class EstimatorG2POMethod:
    """Cross-fitted value estimator behind the unchanged G2PO actor seam.

    This method deliberately owns no environment semantics.  Residual/frontier
    labels must already have been produced by a verifier-backed decomposition
    component, and the estimator bridge must cover every transition before an
    optimizer update is admitted.
    """

    def __init__(
        self,
        estimator: CreditEstimator,
        *,
        node_identity: Literal["coarse", "phase"],
        credit_assigner_name: str,
        config: G2POCreditConfig | None = None,
        source_contract_sha256: str | None = None,
        require_outcome_variation: bool = True,
    ) -> None:
        if not credit_assigner_name.strip():
            raise AgenticRLComponentError("credit assigner name is required")
        if source_contract_sha256 is not None and (
            len(source_contract_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_contract_sha256)
        ):
            raise AgenticRLComponentError(
                "G2PO source parity contract must be a lowercase SHA-256 digest"
            )
        self.estimator = estimator
        self.node_identity = node_identity
        self.config = config or G2POCreditConfig()
        self.require_outcome_variation = require_outcome_variation
        raw_estimator_config = getattr(estimator, "config", None)
        estimator_config = (
            asdict(raw_estimator_config)
            if raw_estimator_config is not None and is_dataclass(raw_estimator_config)
            else {}
        )
        shared_config = {
            "objective": "clipped_policy_gradient",
            "actor_mass": "trajectory_step_token_unit_mass",
            "credit_config": asdict(self.config),
            "source_credit_parity_contract_sha256": source_contract_sha256,
        }
        self.component = builtin_component(
            role="rl_method",
            name="g2po_family_clipped_pg",
            config=shared_config,
        )
        self.credit_assigner = builtin_component(
            role="credit_assigner",
            name=credit_assigner_name,
            config={
                **shared_config,
                "estimator_id": estimator.estimator_id,
                "estimator_config": estimator_config,
                "node_identity": node_identity,
                "require_outcome_variation": require_outcome_variation,
            },
        )

    def prepare_update(
        self,
        records: tuple[TrainingTrajectoryRecord, ...],
    ) -> PreparedPolicyUpdate:
        terminal_outcomes_by_group: dict[str, set[bool]] = {}
        for row in records:
            if row.terminal:
                terminal_outcomes_by_group.setdefault(row.group_uid, set()).add(row.success)
        groups_without_variation = tuple(
            sorted(
                group_uid
                for group_uid, outcomes in terminal_outcomes_by_group.items()
                if outcomes != {False, True}
            )
        )
        if self.require_outcome_variation and groups_without_variation:
            raise AgenticRLComponentError(
                "cross-fitted credit training requires both success and failure "
                "terminal outcomes inside every task group: " + ", ".join(groups_without_variation)
            )
        transitions = g2po_transitions_from_v2_records(
            records,
            mode=G2POCreditMode.TRAJECTORY_NORMALIZED,
        )
        visits = credit_visits_from_trajectory_records(
            records,
            gamma=self.config.gamma,
            success_reward=SUCCESS_REWARD,
        )
        result = assign_estimator_g2po_credit(
            transitions,
            visits,
            estimator=self.estimator,
            node_identity=self.node_identity,
            mode=G2POCreditMode.TRAJECTORY_NORMALIZED,
            config=self.config,
        )
        from lh_platform.training.subproblem_study import assess_oracle_subproblem_trace

        trace_gate = assess_oracle_subproblem_trace(records)
        targets = tuple(
            PolicyTarget(
                traj_uid=record.traj_uid,
                step_idx=record.step_idx,
                advantage=assignment.advantage,
                actor_token_weight=assignment.actor_token_weight,
            )
            for record, assignment in zip(
                records,
                result.assignments,
                strict=True,
            )
        )
        # PhasePool one-update gate contract (assess_phasepool_one_update_decision):
        # require outcome_variation (bool) and credit_estimator.metrics.coverage
        # (float, must be 1.0). Missing keys block as "*_unknown".
        return PreparedPolicyUpdate(
            method=self.component,
            credit_assigner=self.credit_assigner,
            objective="clipped_policy_gradient",
            targets=targets,
            diagnostics={
                "transition_count": len(transitions),
                "outcome_variation": not groups_without_variation,
                "outcome_variation_group_count": len(terminal_outcomes_by_group),
                "node_identity": self.node_identity,
                "oracle_subproblem_trace_gate": asdict(trace_gate),
                "credit_estimator": {
                    "id": result.estimate.estimator_id,
                    "role": result.estimate.estimator_role,
                    "metrics": asdict(result.estimate.metrics),
                    "metadata": dict(result.estimate.metadata),
                },
            },
        )


_ESTIMATOR_CONFIG_FIELDS = frozenset(
    {
        "family",
        "key",
        "weighting",
        "exclusion",
        "pooling",
        "fixed_pooling_weight",
        "require_outcome_variation",
    }
)


def build_g2po_family_method(
    credit_assigner_id: str,
    *,
    estimator_config: Mapping[str, object] | None = None,
    config: G2POCreditConfig | None = None,
    source_contract_sha256: str | None = None,
):
    """Build one registered credit arm without branching the actor optimizer."""

    estimator_payload = dict(estimator_config or {})
    legacy_treatments = {
        "g2po_tn": UnifiedCreditTreatment.G2PO_TN,
        "oracle_phase_refined_g2po_tn": (UnifiedCreditTreatment.ORACLE_PHASE_REFINED_G2PO_TN),
    }
    if credit_assigner_id in legacy_treatments:
        if estimator_payload:
            raise AgenticRLComponentError(
                "legacy G2PO credit assigners do not accept estimator config"
            )
        return UnifiedG2POMethod(
            legacy_treatments[credit_assigner_id],
            config=config,
            source_contract_sha256=source_contract_sha256,
        )
    # Preferred Verifier-Refined path: GroupedMean(phase) + g2po_bridge.
    # Legacy online full-pool remains ``oracle_phase_refined_g2po_tn`` above.
    if credit_assigner_id not in {
        "g2po_tn_loto",
        "phasepool_g2po",
        "verifier_refined_g2po",
    }:
        raise AgenticRLComponentError(
            f"unsupported G2PO-family credit assigner {credit_assigner_id!r}"
        )
    if set(estimator_payload) != _ESTIMATOR_CONFIG_FIELDS:
        raise AgenticRLComponentError(
            "cross-fitted credit estimator config fields differ from schema"
        )
    common_expected = {
        "weighting": "trajectory",
        "exclusion": "trajectory",
        "require_outcome_variation": True,
    }
    if any(estimator_payload[key] != value for key, value in common_expected.items()):
        raise AgenticRLComponentError(
            "cross-fitted credit estimators require trajectory LOTO and outcome variation"
        )
    if credit_assigner_id == "g2po_tn_loto":
        expected = {
            "family": "grouped_mean",
            "key": "coarse",
            "pooling": "full",
        }
        if any(estimator_payload[key] != value for key, value in expected.items()):
            raise AgenticRLComponentError(
                "g2po_tn_loto must be the coarse trajectory-LOTO grouped mean"
            )
        estimator = GroupedMeanEstimator(
            GroupedMeanConfig(
                key="coarse",
                weighting="trajectory",
                exclusion="trajectory",
            )
        )
        node_identity: Literal["coarse", "phase"] = "coarse"
    elif credit_assigner_id == "verifier_refined_g2po":
        expected = {
            "family": "grouped_mean",
            "key": "phase",
            "pooling": "full",
        }
        if any(estimator_payload[key] != value for key, value in expected.items()):
            raise AgenticRLComponentError(
                "verifier_refined_g2po must be the phase trajectory-LOTO grouped mean"
            )
        estimator = GroupedMeanEstimator(
            GroupedMeanConfig(
                key="phase",
                weighting="trajectory",
                exclusion="trajectory",
            )
        )
        node_identity = "phase"
    elif credit_assigner_id == "phasepool_g2po":
        expected = {
            "family": "phasepool",
            "key": "phase",
        }
        if any(estimator_payload[key] != value for key, value in expected.items()):
            raise AgenticRLComponentError("phasepool_g2po must use verifier phase identity")
        estimator = PhasePoolEstimator(
            PhasePoolConfig(
                pooling=str(estimator_payload["pooling"]),
                fixed_pooling_weight=float(estimator_payload["fixed_pooling_weight"]),
                exclusion="trajectory",
            )
        )
        node_identity = "phase"
    return EstimatorG2POMethod(
        estimator,
        node_identity=node_identity,
        credit_assigner_name=credit_assigner_id,
        config=config,
        source_contract_sha256=source_contract_sha256,
        require_outcome_variation=True,
    )
