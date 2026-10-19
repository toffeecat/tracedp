"""Compile a research question into explicit comparison and trial plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from lh_platform.core.execution_paths import ExecutionPath
from lh_platform.studies.evidence import (
    EvidenceRegistry,
    EvidenceResolutionError,
    ResolvedEvidence,
    find_workspace_root,
)

_STUDY_EXECUTION_PATHS = frozenset(
    {
        ExecutionPath.BENCHMARK_STEP_LOOP,
        ExecutionPath.PROMPT_WRAPPER,
        ExecutionPath.UPSTREAM_NATIVE_RUNNER,
    }
)


class StudyKind(StrEnum):
    BASELINE_COMPARISON = "baseline_comparison"
    MECHANISM_TEST = "mechanism_test"
    ABLATION = "ablation"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    EXPLORATORY = "exploratory"


class ArtifactStatus(StrEnum):
    PAPER_ONLY = "paper_only"
    SOURCE_AVAILABLE = "source_available"


class TrainingPath(StrEnum):
    PROMPT_PROXY = "prompt_proxy"
    PAPER_REIMPLEMENTATION = "paper_reimplementation"
    UPSTREAM_TRAINING = "upstream_training"


class EvidenceQuality(StrEnum):
    ABSENT = "absent"
    WEAK = "weak"
    STRONG = "strong"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str


class StudyValidationError(ValueError):
    """Raised when a study cannot produce trustworthy comparison plans."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        summary = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        super().__init__(summary)


@dataclass(frozen=True)
class EvidenceRef:
    id: str
    paper: str
    quote: str
    locator: str
    paper_path: str
    paper_sha256: str
    source_url: str
    source_root: str = ""
    commit: str = ""
    source_paths: tuple[str, ...] = ()
    supports: tuple[str, ...] = ()
    artifact_status: ArtifactStatus = ArtifactStatus.PAPER_ONLY


@dataclass(frozen=True)
class VariantSpec:
    id: str
    method: str
    requested_training_path: TrainingPath
    requested_execution_path: ExecutionPath
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutcomeSpec:
    name: str
    direction: str
    required_outcome_evidence_quality: EvidenceQuality


@dataclass(frozen=True)
class StudyDesign:
    benchmark: str
    task_manifest: str
    control: VariantSpec
    treatment: VariantSpec
    treatment_factors: tuple[str, ...]
    fixed_factors: Mapping[str, Any]
    budget: Mapping[str, int]
    primary_outcome: OutcomeSpec
    mechanism_metrics: tuple[str, ...] = ()
    ablation_reference_parameters: Mapping[str, Any] = field(default_factory=dict)
    ablations: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ReplicationSpec:
    training_seeds: tuple[int, ...]
    paired_eval_tasks: bool


@dataclass(frozen=True)
class AnalysisSpec:
    estimator: str
    uncertainty: str
    stop_rule: str
    horizon_strata: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntegrityRequirements:
    split_unit: str
    group_key: str
    disjoint_splits: bool
    reset_state_between_trials: bool


@dataclass(frozen=True)
class ValiditySpec:
    require_upstream_training: bool
    require_upstream_native_runner: bool
    require_strong_outcome_evidence: bool
    require_benchmark_audit: bool
    required_benchmark_audit_components: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonPlan:
    control_id: str
    treatment_id: str
    treatment_factors: tuple[str, ...]
    differences: Mapping[str, tuple[Any, Any]]
    fixed_factors: Mapping[str, Any]
    budget: Mapping[str, int]
    primary_outcome: OutcomeSpec
    paired_eval_tasks: bool
    ablation_reference_parameters: Mapping[str, Any]


@dataclass(frozen=True)
class TrialPlan:
    study_id: str
    role: str
    variant_id: str
    seed: int
    benchmark: str
    task_manifest: str
    method: str
    requested_training_path: TrainingPath
    requested_execution_path: ExecutionPath
    parameters: Mapping[str, Any]
    fixed_factors: Mapping[str, Any]
    budget: Mapping[str, int]
    integrity_requirements: IntegrityRequirements


@dataclass(frozen=True)
class CompiledStudy:
    study_id: str
    comparison: ComparisonPlan
    trials: tuple[TrialPlan, ...]
    resolved_evidence: tuple[ResolvedEvidence, ...]


@dataclass(frozen=True)
class StudyProtocol:
    schema: str
    id: str
    question: str
    hypothesis: str
    kinds: tuple[StudyKind, ...]
    confirmatory: bool
    evidence: tuple[EvidenceRef, ...]
    design: StudyDesign
    replication: ReplicationSpec
    analysis: AnalysisSpec
    integrity_requirements: IntegrityRequirements
    validity: ValiditySpec
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StudyProtocol:
        study = data["study"]
        design = data["design"]
        replication = data["replication"]
        analysis = data["analysis"]
        integrity = data.get("integrity_requirements") or {}
        validity = data["validity"]

        if not isinstance(design.get("control"), Mapping):
            raise StudyValidationError(
                (
                    ValidationIssue(
                        code="missing_control",
                        path="design.control",
                        message="a comparison control is required",
                    ),
                )
            )

        return cls(
            schema=str(data.get("schema", "")),
            id=str(study.get("id", "")),
            question=str(study.get("question", "")),
            hypothesis=str(study.get("hypothesis", "")),
            kinds=tuple(StudyKind(kind) for kind in study.get("kinds") or ()),
            confirmatory=bool(study.get("confirmatory", False)),
            evidence=tuple(_evidence_ref(item) for item in data.get("evidence") or ()),
            design=StudyDesign(
                benchmark=str(design.get("benchmark", "")),
                task_manifest=str(design.get("task_manifest", "")),
                control=_variant(design["control"]),
                treatment=_variant(design["treatment"]),
                treatment_factors=tuple(design.get("treatment_factors") or ()),
                fixed_factors=dict(design.get("fixed_factors") or {}),
                budget=dict(design.get("budget") or {}),
                primary_outcome=_outcome(design.get("primary_outcome") or {}),
                mechanism_metrics=tuple(design.get("mechanism_metrics") or ()),
                ablation_reference_parameters=dict(
                    design.get("ablation_reference_parameters") or {}
                ),
                ablations=tuple(dict(item) for item in design.get("ablations") or ()),
            ),
            replication=ReplicationSpec(
                training_seeds=tuple(int(seed) for seed in replication.get("training_seeds") or ()),
                paired_eval_tasks=bool(replication.get("paired_eval_tasks", False)),
            ),
            analysis=AnalysisSpec(
                estimator=str(analysis.get("estimator", "")),
                uncertainty=str(analysis.get("uncertainty", "")),
                stop_rule=str(analysis.get("stop_rule", "")),
                horizon_strata=tuple(analysis.get("horizon_strata") or ()),
            ),
            integrity_requirements=IntegrityRequirements(
                split_unit=str(integrity.get("split_unit", "")),
                group_key=str(integrity.get("group_key", "")),
                disjoint_splits=bool(integrity.get("disjoint_splits", False)),
                reset_state_between_trials=bool(integrity.get("reset_state_between_trials", False)),
            ),
            validity=ValiditySpec(
                require_upstream_training=bool(validity.get("require_upstream_training", False)),
                require_upstream_native_runner=bool(
                    validity.get("require_upstream_native_runner", False)
                ),
                require_strong_outcome_evidence=bool(
                    validity.get("require_strong_outcome_evidence", False)
                ),
                require_benchmark_audit=bool(validity.get("require_benchmark_audit", False)),
                required_benchmark_audit_components=tuple(
                    validity.get("required_benchmark_audit_components") or ()
                ),
            ),
        )

    def compile(self) -> CompiledStudy:
        differences = _variant_differences(self.design.control, self.design.treatment)
        resolved_evidence, resolution_issues = _resolve_evidence(self)
        issues = _protocol_contract_issues(self)
        issues.extend(_purpose_issues(self))
        issues.extend(_treatment_difference_issues(self, differences))
        issues.extend(_budget_issues(self))
        issues.extend(_evidence_issues(self))
        issues.extend(resolution_issues)
        issues.extend(_fidelity_issues(self))
        issues.extend(_execution_identity_issues(self))
        issues.extend(_replication_issues(self))
        issues.extend(_result_validity_issues(self))
        issues.extend(_kind_issues(self))
        issues.extend(_ablation_issues(self))
        issues.extend(_integrity_issues(self))
        if issues:
            raise StudyValidationError(tuple(issues))

        comparison = ComparisonPlan(
            control_id=self.design.control.id,
            treatment_id=self.design.treatment.id,
            treatment_factors=self.design.treatment_factors,
            differences=differences,
            fixed_factors=dict(self.design.fixed_factors),
            budget=dict(self.design.budget),
            primary_outcome=self.design.primary_outcome,
            paired_eval_tasks=self.replication.paired_eval_tasks,
            ablation_reference_parameters=dict(self.design.ablation_reference_parameters),
        )
        effective_control = _with_reference_parameters(
            self.design.control, self.design.ablation_reference_parameters
        )
        effective_treatment = _with_reference_parameters(
            self.design.treatment, self.design.ablation_reference_parameters
        )
        trial_variants = (
            ("control", effective_control),
            ("treatment", effective_treatment),
            *(
                ("ablation", _ablation_variant(effective_treatment, ablation))
                for ablation in self.design.ablations
            ),
        )
        trials = tuple(
            _trial(self, role, variant, seed)
            for seed in self.replication.training_seeds
            for role, variant in trial_variants
        )
        return CompiledStudy(
            study_id=self.id,
            comparison=comparison,
            trials=trials,
            resolved_evidence=resolved_evidence,
        )


def _protocol_contract_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if protocol.schema not in {"lh_study/v2", "lh_study/v3"}:
        issues.append(
            ValidationIssue(
                code="unsupported_study_schema",
                path="schema",
                message="expected lh_study/v2 or lh_study/v3",
            )
        )
    if not protocol.kinds:
        issues.append(
            ValidationIssue(
                code="missing_study_kind",
                path="study.kinds",
                message="at least one study kind is required",
            )
        )
    if not protocol.design.primary_outcome.name.strip():
        issues.append(
            ValidationIssue(
                code="missing_primary_outcome",
                path="design.primary_outcome.name",
                message="a pre-registered primary outcome is required",
            )
        )
    if not protocol.analysis.estimator.strip():
        issues.append(
            ValidationIssue(
                code="missing_analysis_estimator",
                path="analysis.estimator",
                message="an analysis estimator is required",
            )
        )
    if not protocol.analysis.uncertainty.strip():
        issues.append(
            ValidationIssue(
                code="missing_uncertainty_method",
                path="analysis.uncertainty",
                message="an uncertainty method is required",
            )
        )
    if not protocol.analysis.stop_rule.strip():
        issues.append(
            ValidationIssue(
                code="missing_stop_rule",
                path="analysis.stop_rule",
                message="a pre-registered stopping rule is required",
            )
        )
    return issues


def _purpose_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not protocol.question.strip():
        issues.append(
            ValidationIssue(
                code="missing_question",
                path="study.question",
                message="a research question is required",
            )
        )
    if not protocol.hypothesis.strip():
        issues.append(
            ValidationIssue(
                code="missing_hypothesis",
                path="study.hypothesis",
                message="a falsifiable hypothesis is required",
            )
        )
    return issues


def _treatment_difference_issues(
    protocol: StudyProtocol,
    differences: Mapping[str, tuple[Any, Any]],
) -> list[ValidationIssue]:
    declared = set(protocol.design.treatment_factors)
    issues = [
        ValidationIssue(
            code="undeclared_treatment_difference",
            path=f"design.treatment.{factor}",
            message=f"{factor!r} differs between control and treatment but is not declared",
        )
        for factor in differences
        if factor not in declared
    ]
    issues.extend(
        ValidationIssue(
            code="inert_treatment_factor",
            path=f"design.treatment_factors.{factor}",
            message=f"declared treatment factor {factor!r} does not differ from the control",
        )
        for factor in protocol.design.treatment_factors
        if factor not in differences
    )
    return issues


def _budget_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    resource_keys = ("environment_steps", "generated_tokens", "training_steps")
    positive = {
        key
        for key in resource_keys
        if isinstance(protocol.design.budget.get(key), int) and protocol.design.budget[key] > 0
    }
    if not positive:
        return [
            ValidationIssue(
                code="missing_fixed_budget",
                path="design.budget",
                message="at least one positive shared resource budget is required",
            )
        ]
    if _requires_confirmatory_evidence(protocol):
        missing = [key for key in resource_keys if key not in positive]
        if missing:
            return [
                ValidationIssue(
                    code="incomplete_fixed_budget",
                    path="design.budget",
                    message=f"confirmatory training is missing shared limits: {', '.join(missing)}",
                )
            ]
    return []


def _evidence_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for index, evidence in enumerate(protocol.evidence):
        prefix = f"evidence[{index}]"
        if not evidence.quote.strip():
            issues.append(
                ValidationIssue(
                    code="missing_evidence_quote",
                    path=f"{prefix}.quote",
                    message="an exact primary-source quote is required",
                )
            )
        if not evidence.locator.strip():
            issues.append(
                ValidationIssue(
                    code="missing_evidence_locator",
                    path=f"{prefix}.locator",
                    message="the quote page or section is required",
                )
            )
        if not evidence.paper_path.strip():
            issues.append(
                ValidationIssue(
                    code="missing_paper_path",
                    path=f"{prefix}.paper_path",
                    message="the immutable local paper path is required",
                )
            )
        if not evidence.paper_sha256.strip():
            issues.append(
                ValidationIssue(
                    code="missing_paper_sha256",
                    path=f"{prefix}.paper_sha256",
                    message="the immutable paper SHA-256 is required",
                )
            )
        if not evidence.source_url.strip():
            issues.append(
                ValidationIssue(
                    code="missing_source_url",
                    path=f"{prefix}.source_url",
                    message="the primary source URL is required",
                )
            )
        if evidence.artifact_status == "source_available" and not evidence.source_root.strip():
            issues.append(
                ValidationIssue(
                    code="missing_source_root",
                    path=f"{prefix}.source_root",
                    message="source_available evidence requires a local source root",
                )
            )
        if evidence.artifact_status == "source_available" and not evidence.commit.strip():
            issues.append(
                ValidationIssue(
                    code="missing_source_commit",
                    path=f"{prefix}.commit",
                    message="source_available evidence requires an immutable commit",
                )
            )
        if evidence.artifact_status == "source_available" and not evidence.source_paths:
            issues.append(
                ValidationIssue(
                    code="missing_source_paths",
                    path=f"{prefix}.source_paths",
                    message="source_available evidence requires source-code paths",
                )
            )

    supported = {factor for evidence in protocol.evidence for factor in evidence.supports}
    issues.extend(
        ValidationIssue(
            code="missing_factor_evidence",
            path=f"design.treatment_factors.{factor}",
            message=f"no primary-source evidence supports treatment factor {factor!r}",
        )
        for factor in protocol.design.treatment_factors
        if factor not in supported
    )
    if _requires_confirmatory_evidence(protocol):
        source_supported = {
            factor
            for evidence in protocol.evidence
            if evidence.artifact_status == "source_available"
            for factor in evidence.supports
        }
        issues.extend(
            ValidationIssue(
                code="source_unavailable_for_confirmatory",
                path=f"design.treatment_factors.{factor}",
                message=f"confirmatory factor {factor!r} requires resolved source code",
            )
            for factor in protocol.design.treatment_factors
            if factor not in source_supported
        )
    return issues


def _resolve_evidence(
    protocol: StudyProtocol,
) -> tuple[tuple[ResolvedEvidence, ...], list[ValidationIssue]]:
    registry = EvidenceRegistry(find_workspace_root(protocol.source_path))
    resolved: list[ResolvedEvidence] = []
    issues: list[ValidationIssue] = []
    for index, evidence in enumerate(protocol.evidence):
        required = (
            evidence.quote,
            evidence.locator,
            evidence.paper_path,
            evidence.paper_sha256,
            evidence.source_url,
        )
        if not all(value.strip() for value in required):
            continue
        if evidence.artifact_status == "source_available" and not (
            evidence.source_root.strip() and evidence.commit.strip() and evidence.source_paths
        ):
            continue
        try:
            resolved.append(registry.resolve(evidence, index))
        except EvidenceResolutionError as error:
            issues.append(ValidationIssue(code=error.code, path=error.path, message=error.message))
    return tuple(resolved), issues


def _fidelity_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    confirmatory = _requires_confirmatory_evidence(protocol)
    for role, variant in (
        ("control", protocol.design.control),
        ("treatment", protocol.design.treatment),
    ):
        allowed_training = {TrainingPath.UPSTREAM_TRAINING}
        if confirmatory and not protocol.validity.require_upstream_training:
            allowed_training.add(TrainingPath.PAPER_REIMPLEMENTATION)
        if variant.requested_training_path not in allowed_training:
            issues.append(
                ValidationIssue(
                    code="insufficient_training_fidelity",
                    path=f"design.{role}.requested_training_path",
                    message=(
                        "confirmatory studies require a source-locked paper "
                        "reimplementation or upstream training"
                    ),
                )
            )
        allowed_execution = {ExecutionPath.UPSTREAM_NATIVE_RUNNER}
        if confirmatory and not protocol.validity.require_upstream_native_runner:
            allowed_execution.add(ExecutionPath.BENCHMARK_STEP_LOOP)
        if variant.requested_execution_path not in allowed_execution:
            issues.append(
                ValidationIssue(
                    code="insufficient_evaluation_fidelity",
                    path=f"design.{role}.requested_execution_path",
                    message=(
                        "confirmatory studies require a source-locked benchmark "
                        "step loop or upstream native runner"
                    ),
                )
            )
    return issues


def _execution_identity_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    """Bind generic step-loop studies to the adapters they actually invoke.

    ``upstream_native_runner`` studies deliberately do not consult these
    registries: their benchmark and method identities are source-locked by the
    upstream execution contract.  A ``benchmark_step_loop`` study, by contrast,
    will be executed by :class:`lh_platform.core.runner.Runner`; accepting an
    unregistered identity there would produce a trial plan that cannot run.
    """

    variants = (
        ("control", protocol.design.control),
        ("treatment", protocol.design.treatment),
    )
    generic_variants = tuple(
        (role, variant)
        for role, variant in variants
        if variant.requested_execution_path == ExecutionPath.BENCHMARK_STEP_LOOP
    )
    if not generic_variants:
        return []

    # Keep the study layer independent from adapter import order, while making
    # the execution contract fail before a trial is prepared.
    from lh_platform.core.registry import BENCHMARKS, METHODS, bootstrap_default_registries

    bootstrap_default_registries()
    issues: list[ValidationIssue] = []
    if not BENCHMARKS.has(protocol.design.benchmark):
        issues.append(
            ValidationIssue(
                code="benchmark_step_loop_benchmark_not_registered",
                path="design.benchmark",
                message=(
                    f"benchmark_step_loop requires a registered benchmark; "
                    f"{protocol.design.benchmark!r} is upstream-native metadata, not a step-loop adapter"
                ),
            )
        )
    for role, variant in generic_variants:
        if not METHODS.has(variant.method):
            issues.append(
                ValidationIssue(
                    code="benchmark_step_loop_method_not_registered",
                    path=f"design.{role}.method",
                    message=(
                        f"benchmark_step_loop requires a registered method; "
                        f"{variant.method!r} is not executable by the generic runner"
                    ),
                )
            )
    return issues


def _replication_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    if not _requires_confirmatory_evidence(protocol):
        return []
    issues: list[ValidationIssue] = []
    if len(set(protocol.replication.training_seeds)) < 3:
        issues.append(
            ValidationIssue(
                code="insufficient_replication",
                path="replication.training_seeds",
                message="confirmatory studies require at least three distinct training seeds",
            )
        )
    if not protocol.replication.paired_eval_tasks:
        issues.append(
            ValidationIssue(
                code="unpaired_evaluation",
                path="replication.paired_eval_tasks",
                message="confirmatory comparisons require paired evaluation tasks",
            )
        )
    return issues


def _result_validity_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if (
        _requires_confirmatory_evidence(protocol)
        or protocol.validity.require_strong_outcome_evidence
    ) and protocol.design.primary_outcome.required_outcome_evidence_quality != "strong":
        issues.append(
            ValidationIssue(
                code="insufficient_outcome_evidence",
                path="design.primary_outcome.required_outcome_evidence_quality",
                message="this study must require strong benchmark-owned Outcome Evidence",
            )
        )
    if _requires_confirmatory_evidence(protocol) or protocol.validity.require_benchmark_audit:
        required = {"user", "environment", "ground_truth", "evaluation"}
        missing = sorted(required - set(protocol.validity.required_benchmark_audit_components))
        if missing:
            issues.append(
                ValidationIssue(
                    code="incomplete_benchmark_audit_requirements",
                    path="validity.required_benchmark_audit_components",
                    message=f"missing required benchmark audit components: {', '.join(missing)}",
                )
            )
    return issues


def _kind_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    kinds = set(protocol.kinds)
    if "mechanism_test" in kinds and not protocol.design.mechanism_metrics:
        issues.append(
            ValidationIssue(
                code="missing_mechanism_metrics",
                path="design.mechanism_metrics",
                message="mechanism_test requires a falsifiable mechanism measurement",
            )
        )
    if "ablation" in kinds and not protocol.design.ablations:
        issues.append(
            ValidationIssue(
                code="missing_ablations",
                path="design.ablations",
                message="ablation studies require at least one declared ablation",
            )
        )
    return issues


def _ablation_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    if StudyKind.ABLATION not in protocol.kinds:
        return []
    issues: list[ValidationIssue] = []
    evidence_by_id = {evidence.id: evidence for evidence in protocol.evidence}
    evidence_ids = set(evidence_by_id)
    reference = protocol.design.ablation_reference_parameters
    ablation_ids = [str(ablation.get("id", "")).strip() for ablation in protocol.design.ablations]
    for index, ablation in enumerate(protocol.design.ablations):
        prefix = f"design.ablations[{index}]"
        ablation_id = ablation_ids[index]
        if not ablation_id:
            issues.append(
                ValidationIssue(
                    code="missing_ablation_id",
                    path=f"{prefix}.id",
                    message="each ablation requires a stable id",
                )
            )
        elif ablation_ids.count(ablation_id) > 1:
            issues.append(
                ValidationIssue(
                    code="duplicate_ablation_id",
                    path=f"{prefix}.id",
                    message=f"ablation id {ablation_id!r} is duplicated",
                )
            )
        parameters = ablation.get("parameters") or {}
        if not isinstance(parameters, Mapping) or not parameters:
            issues.append(
                ValidationIssue(
                    code="missing_ablation_parameters",
                    path=f"{prefix}.parameters",
                    message="an ablation must change at least one treatment parameter",
                )
            )
        else:
            unknown_parameters = sorted(set(parameters) - set(reference))
            if unknown_parameters:
                issues.append(
                    ValidationIssue(
                        code="unknown_ablation_parameter",
                        path=f"{prefix}.parameters",
                        message=(
                            "parameters absent from source-backed reference: "
                            f"{', '.join(unknown_parameters)}"
                        ),
                    )
                )
            elif all(reference[name] == value for name, value in parameters.items()):
                issues.append(
                    ValidationIssue(
                        code="inert_ablation",
                        path=f"{prefix}.parameters",
                        message="the ablation matches the source-backed treatment defaults",
                    )
                )
        evidence_refs = tuple(ablation.get("evidence_refs") or ())
        if not evidence_refs:
            issues.append(
                ValidationIssue(
                    code="missing_ablation_evidence",
                    path=f"{prefix}.evidence_refs",
                    message="each ablation must cite resolved primary-source evidence",
                )
            )
        else:
            unknown = sorted(set(evidence_refs) - evidence_ids)
            if unknown:
                issues.append(
                    ValidationIssue(
                        code="unknown_ablation_evidence",
                        path=f"{prefix}.evidence_refs",
                        message=f"unknown evidence refs: {', '.join(unknown)}",
                    )
                )
            elif isinstance(parameters, Mapping):
                for parameter in parameters:
                    factor = f"parameters.{parameter}"
                    if not any(
                        factor in evidence_by_id[evidence_id].supports
                        for evidence_id in evidence_refs
                    ):
                        issues.append(
                            ValidationIssue(
                                code="unsupported_ablation_parameter",
                                path=f"{prefix}.evidence_refs",
                                message=(
                                    f"no referenced evidence supports ablation factor {factor!r}"
                                ),
                            )
                        )
    return issues


def _integrity_issues(protocol: StudyProtocol) -> list[ValidationIssue]:
    if not _requires_confirmatory_evidence(protocol):
        return []
    issues: list[ValidationIssue] = []
    if (
        protocol.integrity_requirements.split_unit != "task_group"
        or not protocol.integrity_requirements.group_key.strip()
        or not protocol.integrity_requirements.disjoint_splits
    ):
        issues.append(
            ValidationIssue(
                code="data_leakage_risk",
                path="integrity_requirements",
                message=(
                    "confirmatory training requires disjoint task_group splits "
                    "with an explicit group key"
                ),
            )
        )
    if (
        not protocol.integrity_requirements.reset_state_between_trials
        and "state_persistence" not in protocol.design.treatment_factors
    ):
        issues.append(
            ValidationIssue(
                code="state_isolation_missing",
                path="integrity_requirements.reset_state_between_trials",
                message="state must reset between trials unless persistence is the treatment",
            )
        )
    return issues


def _requires_confirmatory_evidence(protocol: StudyProtocol) -> bool:
    return protocol.confirmatory or any(
        kind is not StudyKind.EXPLORATORY for kind in protocol.kinds
    )


def _evidence_ref(data: Mapping[str, Any]) -> EvidenceRef:
    return EvidenceRef(
        id=str(data.get("id", "")),
        paper=str(data.get("paper", "")),
        quote=str(data.get("quote", "")),
        locator=str(data.get("locator", "")),
        paper_path=str(data.get("paper_path", "")),
        paper_sha256=str(data.get("paper_sha256", "")),
        source_url=str(data.get("source_url", "")),
        source_root=str(data.get("source_root", "")),
        commit=str(data.get("commit", "")),
        source_paths=tuple(data.get("source_paths") or ()),
        supports=tuple(data.get("supports") or ()),
        artifact_status=ArtifactStatus(data.get("artifact_status", "paper_only")),
    )


def _variant(data: Mapping[str, Any]) -> VariantSpec:
    requested_execution_path = ExecutionPath(data.get("requested_execution_path", ""))
    if requested_execution_path not in _STUDY_EXECUTION_PATHS:
        allowed = ", ".join(sorted(path.value for path in _STUDY_EXECUTION_PATHS))
        raise ValueError("study requested_execution_path must be one of " + allowed)
    return VariantSpec(
        id=str(data.get("id", "")),
        method=str(data.get("method", "")),
        requested_training_path=TrainingPath(data.get("requested_training_path", "")),
        requested_execution_path=requested_execution_path,
        parameters=dict(data.get("parameters") or {}),
    )


def _outcome(data: Mapping[str, Any]) -> OutcomeSpec:
    return OutcomeSpec(
        name=str(data.get("name", "")),
        direction=str(data.get("direction", "")),
        required_outcome_evidence_quality=EvidenceQuality(
            data.get("required_outcome_evidence_quality", "")
        ),
    )


def _variant_differences(
    control: VariantSpec,
    treatment: VariantSpec,
) -> dict[str, tuple[Any, Any]]:
    differences: dict[str, tuple[Any, Any]] = {}
    for field_name in ("method", "requested_training_path", "requested_execution_path"):
        left = getattr(control, field_name)
        right = getattr(treatment, field_name)
        if left != right:
            differences[field_name] = (left, right)
    for name in sorted(set(control.parameters) | set(treatment.parameters)):
        left = control.parameters.get(name)
        right = treatment.parameters.get(name)
        if left != right:
            differences[f"parameters.{name}"] = (left, right)
    return differences


def _trial(
    protocol: StudyProtocol,
    role: str,
    variant: VariantSpec,
    seed: int,
) -> TrialPlan:
    return TrialPlan(
        study_id=protocol.id,
        role=role,
        variant_id=variant.id,
        seed=seed,
        benchmark=protocol.design.benchmark,
        task_manifest=protocol.design.task_manifest,
        method=variant.method,
        requested_training_path=variant.requested_training_path,
        requested_execution_path=variant.requested_execution_path,
        parameters=dict(variant.parameters),
        fixed_factors=dict(protocol.design.fixed_factors),
        budget=dict(protocol.design.budget),
        integrity_requirements=protocol.integrity_requirements,
    )


def _ablation_variant(
    treatment: VariantSpec,
    ablation: Mapping[str, Any],
) -> VariantSpec:
    parameters = dict(treatment.parameters)
    parameters.update(dict(ablation.get("parameters") or {}))
    return VariantSpec(
        id=str(ablation.get("id", "")),
        method=treatment.method,
        requested_training_path=treatment.requested_training_path,
        requested_execution_path=treatment.requested_execution_path,
        parameters=parameters,
    )


def _with_reference_parameters(
    variant: VariantSpec,
    reference_parameters: Mapping[str, Any],
) -> VariantSpec:
    parameters = dict(reference_parameters)
    parameters.update(variant.parameters)
    return VariantSpec(
        id=variant.id,
        method=variant.method,
        requested_training_path=variant.requested_training_path,
        requested_execution_path=variant.requested_execution_path,
        parameters=parameters,
    )
