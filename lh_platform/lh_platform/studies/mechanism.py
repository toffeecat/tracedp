"""Phase 2 mechanism-probe planning for GraphGPO studies.

The probe plan is intentionally pre-execution evidence.  It declares the exact
state and trace contracts required before a mechanism result can be trusted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lh_platform.studies.graphgpo.runner import GRAPHGPO_ALLOWED_ABLATIONS, PreparedStudy
from lh_platform.studies.protocol import CompiledStudy, StudyKind, StudyProtocol
from lh_platform.studies.runtime import RuntimePreflightReport, sha256_file

MECHANISM_PROBE_SCHEMA = "lh_mechanism_probe_plan/v1"
MECHANISM_REPORT_SCHEMA = "lh_mechanism_report/v1"
GRAPHGPO_REQUIRED_MECHANISM_METRICS = (
    "advantage_counterfactual_rank_correlation",
    "correct_action_retention",
    "effective_gradient_trajectory_ratio",
)
MIN_COUNTERFACTUAL_EDGE_COUNT = 3
MIN_RANK_CORRELATION = 0.5
MIN_RETENTION_RATE = 0.5
MIN_EFFECTIVE_TRAJECTORY_RATIO = 0.5


EdgeKey = tuple[str, str, int, str, str, int, str, str]


@dataclass(frozen=True)
class MechanismSourceEvidence:
    path: str
    sha256: str
    source_quotes: tuple[str, ...]
    decision: str
    verified: bool


@dataclass(frozen=True)
class MechanismProbeUnit:
    metric: str
    comparison: str
    sample_unit: str
    required_state_policy: str
    estimator: str
    required_fields: tuple[str, ...]


@dataclass(frozen=True)
class MechanismAblationPlan:
    ablation_id: str
    variant_ids: tuple[str, ...]
    parameters: Mapping[str, Any]
    purpose: str


@dataclass(frozen=True)
class MechanismMetricResult:
    passed: bool
    value: float
    threshold: float
    sample_count: int
    details: Mapping[str, Any]


@dataclass(frozen=True)
class MechanismReport:
    schema: str
    study_id: str
    credit_assignment_stable: bool
    source: str
    blocked_by: tuple[str, ...]
    metrics: Mapping[str, MechanismMetricResult]
    input_artifacts: Mapping[str, Mapping[str, Any]]

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        if target.suffix != ".json":
            target = target / "mechanism_report.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


@dataclass(frozen=True)
class EdgeTraceRow:
    key: EdgeKey
    task_id: str
    seed: int
    variant_id: str
    trajectory_id: str
    horizon_bucket: str
    edge_advantage: float
    advantages: tuple[float, ...]
    expert_action_match: bool
    trajectory_success: bool
    is_action_valid: bool


@dataclass(frozen=True)
class CounterfactualContinuationRow:
    key: EdgeKey
    won: bool


@dataclass(frozen=True)
class MechanismProbePlan:
    schema: str
    study_id: str
    plan_ready: bool
    blocked_by: tuple[str, ...]
    reference_treatment_variant: str
    exact_state_required: bool
    continuation_rollouts_per_edge: int
    counterfactual_success_signal: str
    source_evidence: tuple[MechanismSourceEvidence, ...]
    probe_units: tuple[MechanismProbeUnit, ...]
    ablations: tuple[MechanismAblationPlan, ...]
    required_artifacts: tuple[str, ...]

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "mechanism_probe_plan.json"
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_graphgpo_mechanism_probe_plan(
    protocol: StudyProtocol,
    compiled: CompiledStudy,
    prepared: PreparedStudy,
    *,
    runtime_report: RuntimePreflightReport,
    source_root: str | Path,
    continuation_rollouts_per_edge: int = 8,
) -> MechanismProbePlan:
    source = Path(source_root).resolve()
    source_evidence = _source_evidence(source)
    blocked_by = list(runtime_report.blocked_by)
    if not prepared.launch_allowed:
        blocked_by.extend(code for code in prepared.blocked_by if code not in blocked_by)
    if not all(evidence.verified for evidence in source_evidence):
        blocked_by.append("mechanism_source_unverified")
    if tuple(protocol.design.mechanism_metrics) != GRAPHGPO_REQUIRED_MECHANISM_METRICS:
        blocked_by.append("mechanism_metrics_not_preregistered")
    requires_ablations = any(str(kind) == "ablation" for kind in protocol.kinds)
    if requires_ablations and _ablation_ids(protocol) != GRAPHGPO_ALLOWED_ABLATIONS:
        blocked_by.append("ablation_set_not_preregistered")
    if not requires_ablations and _ablation_ids(protocol):
        blocked_by.append("unexpected_ablations_for_treatment_only_probe")

    ablations = tuple(
        MechanismAblationPlan(
            ablation_id=ablation_id,
            variant_ids=tuple(
                sorted(
                    {
                        trial.variant_id
                        for trial in compiled.trials
                        if trial.variant_id == ablation_id
                    }
                )
            ),
            parameters=dict(ablation.get("parameters") or {}),
            purpose=_ablation_purpose(ablation_id),
        )
        for ablation_id, ablation in zip(
            _ablation_ids(protocol), protocol.design.ablations, strict=False
        )
    )
    probe_units = (
        MechanismProbeUnit(
            metric="advantage_counterfactual_rank_correlation",
            comparison=(
                "assigned edge advantage vs empirical completion probability from "
                "the same restored ALFWorld state"
            ),
            sample_unit="task_id, seed, trajectory_id, edge_index, candidate_action",
            required_state_policy="restore exact pre-action environment state before each continuation",
            estimator="spearman_rank_correlation_by_horizon_bucket",
            required_fields=(
                "edge_advantage",
                "counterfactual_completion_probability",
                "horizon_bucket",
                "state_snapshot_sha256",
            ),
        ),
        MechanismProbeUnit(
            metric="correct_action_retention",
            comparison="verified correct failed-trajectory actions with positive edge credit",
            sample_unit="failed trajectory edge",
            required_state_policy="expert-plan alignment from manifest-bound ALFWorld task",
            estimator="retained_positive_credit_rate",
            required_fields=(
                "edge_advantage",
                "expert_action_match",
                "trajectory_success",
            ),
        ),
        MechanismProbeUnit(
            metric="effective_gradient_trajectory_ratio",
            comparison="non-zero useful-advantage trajectories across treatment and ablations",
            sample_unit="trajectory",
            required_state_policy="native rollout trace with upstream advantages",
            estimator="ratio_with_nonzero_valid_action_advantage",
            required_fields=(
                "advantages",
                "is_action_valid",
                "trajectory_success",
            ),
        ),
    )
    return MechanismProbePlan(
        schema=MECHANISM_PROBE_SCHEMA,
        study_id=protocol.id,
        plan_ready=not blocked_by,
        blocked_by=tuple(dict.fromkeys(blocked_by)),
        reference_treatment_variant=protocol.design.treatment.id,
        exact_state_required=True,
        continuation_rollouts_per_edge=continuation_rollouts_per_edge,
        counterfactual_success_signal="info['won']",
        source_evidence=source_evidence,
        probe_units=probe_units,
        ablations=ablations,
        required_artifacts=(
            "runtime_validity_report.json",
            "state_reset_probe.json",
            "planned_commands.jsonl",
            "edge_credit_trace.jsonl",
            "counterfactual_continuations.jsonl",
            "mechanism_report.json",
        ),
    )


def build_graphgpo_mechanism_report(
    protocol: StudyProtocol,
    *,
    edge_credit_trace: str | Path,
    counterfactual_continuations: str | Path,
) -> MechanismReport:
    edge_path = Path(edge_credit_trace)
    continuation_path = Path(counterfactual_continuations)
    raw_edges, edge_errors = _read_jsonl(edge_path)
    raw_continuations, continuation_errors = _read_jsonl(continuation_path)
    blockers = [*edge_errors, *continuation_errors]

    edge_rows, edge_schema_errors = _parse_edge_rows(raw_edges, protocol.id)
    continuation_rows, continuation_schema_errors = _parse_continuation_rows(
        raw_continuations, protocol.id
    )
    if edge_schema_errors:
        blockers.append("edge_credit_trace_schema_invalid")
    if continuation_schema_errors:
        blockers.append("counterfactual_continuation_schema_invalid")
    if not edge_rows:
        blockers.append("edge_credit_trace_empty")
    if not continuation_rows:
        blockers.append("counterfactual_continuations_empty")

    probabilities = _counterfactual_probabilities(continuation_rows)
    matched_pairs: list[tuple[float, float, str]] = []
    missing_counterfactual_edges = 0
    for edge in edge_rows:
        if edge.key not in probabilities:
            missing_counterfactual_edges += 1
            continue
        matched_pairs.append((edge.edge_advantage, probabilities[edge.key], edge.horizon_bucket))
    if missing_counterfactual_edges:
        blockers.append("counterfactual_edges_missing")

    metrics = {
        "advantage_counterfactual_rank_correlation": _rank_correlation_metric(
            matched_pairs,
            required_buckets=tuple(protocol.analysis.horizon_strata),
            missing_counterfactual_edges=missing_counterfactual_edges,
        ),
        "correct_action_retention": _correct_action_retention_metric(edge_rows),
        "effective_gradient_trajectory_ratio": _effective_gradient_metric(
            edge_rows,
            required_variants=(
                protocol.design.treatment.id,
                *(_ablation_ids(protocol) if StudyKind.ABLATION in protocol.kinds else ()),
            ),
        ),
    }
    for metric, result in metrics.items():
        if not result.passed:
            blockers.append(f"mechanism_metric_failed:{metric}")
    blockers = tuple(dict.fromkeys(blockers))
    return MechanismReport(
        schema=MECHANISM_REPORT_SCHEMA,
        study_id=protocol.id,
        credit_assignment_stable=not blockers,
        source="phase4_edge_counterfactual_artifacts",
        blocked_by=blockers,
        metrics=metrics,
        input_artifacts={
            "edge_credit_trace": _artifact_evidence(edge_path),
            "counterfactual_continuations": _artifact_evidence(continuation_path),
        },
    )


def _ablation_ids(protocol: StudyProtocol) -> tuple[str, ...]:
    return tuple(str(ablation.get("id", "")) for ablation in protocol.design.ablations)


def _read_jsonl(path: Path) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    if not path.is_file():
        return (), (f"artifact_not_found:{path.name}",)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                errors.append(f"artifact_json_invalid:{path.name}:{line_number}")
                continue
            if not isinstance(row, dict):
                errors.append(f"artifact_row_not_object:{path.name}:{line_number}")
                continue
            rows.append(row)
    except OSError:
        errors.append(f"artifact_not_readable:{path.name}")
    return tuple(rows), tuple(errors)


def _parse_edge_rows(
    rows: tuple[dict[str, Any], ...], study_id: str
) -> tuple[tuple[EdgeTraceRow, ...], tuple[str, ...]]:
    parsed: list[EdgeTraceRow] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        if row.get("study_id") != study_id:
            continue
        key = _edge_key(row)
        edge_advantage = _as_float(row.get("edge_advantage"))
        advantages = _as_float_tuple(row.get("advantages"))
        expert_action_match = _as_bool(row.get("expert_action_match"))
        trajectory_success = _as_bool(row.get("trajectory_success"))
        is_action_valid = _as_bool(row.get("is_action_valid"))
        horizon_bucket = _as_nonempty_str(row.get("horizon_bucket"))
        variant_id = _as_nonempty_str(row.get("variant_id"))
        if (
            key is None
            or edge_advantage is None
            or advantages is None
            or expert_action_match is None
            or trajectory_success is None
            or is_action_valid is None
            or horizon_bucket is None
            or variant_id is None
        ):
            errors.append(f"edge_credit_trace_schema_invalid:{index}")
            continue
        parsed.append(
            EdgeTraceRow(
                key=key,
                task_id=key[1],
                seed=key[2],
                variant_id=variant_id,
                trajectory_id=key[4],
                horizon_bucket=horizon_bucket,
                edge_advantage=edge_advantage,
                advantages=advantages,
                expert_action_match=expert_action_match,
                trajectory_success=trajectory_success,
                is_action_valid=is_action_valid,
            )
        )
    return tuple(parsed), tuple(errors)


def _parse_continuation_rows(
    rows: tuple[dict[str, Any], ...], study_id: str
) -> tuple[tuple[CounterfactualContinuationRow, ...], tuple[str, ...]]:
    parsed: list[CounterfactualContinuationRow] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        if row.get("study_id") != study_id:
            continue
        key = _edge_key(row)
        won = _as_bool(row.get("won"))
        if key is None or won is None:
            errors.append(f"counterfactual_continuation_schema_invalid:{index}")
            continue
        parsed.append(CounterfactualContinuationRow(key=key, won=won))
    return tuple(parsed), tuple(errors)


def _edge_key(row: Mapping[str, Any]) -> EdgeKey | None:
    required = (
        "study_id",
        "task_id",
        "seed",
        "variant_id",
        "trajectory_id",
        "edge_index",
        "candidate_action",
        "state_snapshot_sha256",
    )
    if any(key not in row for key in required):
        return None
    try:
        return (
            str(row["study_id"]),
            str(row["task_id"]),
            int(row["seed"]),
            str(row["variant_id"]),
            str(row["trajectory_id"]),
            int(row["edge_index"]),
            str(row["candidate_action"]),
            str(row["state_snapshot_sha256"]),
        )
    except (TypeError, ValueError):
        return None


def _counterfactual_probabilities(
    rows: tuple[CounterfactualContinuationRow, ...],
) -> dict[EdgeKey, float]:
    wins_by_key: dict[EdgeKey, list[bool]] = {}
    for row in rows:
        wins_by_key.setdefault(row.key, []).append(row.won)
    return {
        key: sum(1 for item in wins if item) / len(wins)
        for key, wins in wins_by_key.items()
        if wins
    }


def _rank_correlation_metric(
    pairs: list[tuple[float, float, str]],
    *,
    required_buckets: tuple[str, ...],
    missing_counterfactual_edges: int,
) -> MechanismMetricResult:
    required = required_buckets or tuple(sorted({bucket for _, _, bucket in pairs}))
    by_bucket: dict[str, dict[str, float | int | bool]] = {}
    for bucket in required:
        bucket_pairs = [
            (advantage, probability)
            for advantage, probability, item_bucket in pairs
            if item_bucket == bucket
        ]
        value = _spearman(
            [advantage for advantage, _ in bucket_pairs],
            [probability for _, probability in bucket_pairs],
        )
        by_bucket[bucket] = {
            "value": value,
            "sample_count": len(bucket_pairs),
            "passed": len(bucket_pairs) >= MIN_COUNTERFACTUAL_EDGE_COUNT
            and value >= MIN_RANK_CORRELATION,
        }
    value = min((float(item["value"]) for item in by_bucket.values()), default=0.0)
    passed = (
        bool(by_bucket)
        and all(bool(item["passed"]) for item in by_bucket.values())
        and missing_counterfactual_edges == 0
    )
    return MechanismMetricResult(
        passed=passed,
        value=value,
        threshold=MIN_RANK_CORRELATION,
        sample_count=len(pairs),
        details={
            "missing_counterfactual_edges": missing_counterfactual_edges,
            "by_horizon_bucket": by_bucket,
        },
    )


def _correct_action_retention_metric(rows: tuple[EdgeTraceRow, ...]) -> MechanismMetricResult:
    candidates = [row for row in rows if row.expert_action_match and not row.trajectory_success]
    retained = [row for row in candidates if row.edge_advantage > 0.0]
    value = len(retained) / len(candidates) if candidates else 0.0
    return MechanismMetricResult(
        passed=bool(candidates) and value >= MIN_RETENTION_RATE,
        value=value,
        threshold=MIN_RETENTION_RATE,
        sample_count=len(candidates),
        details={"retained_positive_credit_edges": len(retained)},
    )


def _effective_gradient_metric(
    rows: tuple[EdgeTraceRow, ...], *, required_variants: tuple[str, ...]
) -> MechanismMetricResult:
    trajectories: dict[tuple[str, str, int, str], list[EdgeTraceRow]] = {}
    for row in rows:
        key = (row.variant_id, row.task_id, row.seed, row.trajectory_id)
        trajectories.setdefault(key, []).append(row)
    by_variant: dict[str, dict[str, float | int | bool]] = {}
    for variant in required_variants:
        variant_trajectories = {
            key: value for key, value in trajectories.items() if key[0] == variant
        }
        effective = sum(
            1
            for trajectory_rows in variant_trajectories.values()
            if any(
                row.is_action_valid and any(abs(advantage) > 1e-12 for advantage in row.advantages)
                for row in trajectory_rows
            )
        )
        value = effective / len(variant_trajectories) if variant_trajectories else 0.0
        by_variant[variant] = {
            "value": value,
            "sample_count": len(variant_trajectories),
            "effective_trajectories": effective,
            "passed": bool(variant_trajectories) and value >= MIN_EFFECTIVE_TRAJECTORY_RATIO,
        }
    value = min((float(item["value"]) for item in by_variant.values()), default=0.0)
    passed = bool(by_variant) and all(bool(item["passed"]) for item in by_variant.values())
    return MechanismMetricResult(
        passed=passed,
        value=value,
        threshold=MIN_EFFECTIVE_TRAJECTORY_RATIO,
        sample_count=len(trajectories),
        details={"by_variant": by_variant},
    )


def _as_nonempty_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_float_tuple(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),)
    if not isinstance(value, list) or not value:
        return None
    parsed = tuple(_as_float(item) for item in value)
    if any(item is None for item in parsed):
        return None
    return tuple(float(item) for item in parsed if item is not None)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=False)
    )
    left_denominator = sum((a - left_mean) ** 2 for a in left_ranks) ** 0.5
    right_denominator = sum((b - right_mean) ** 2 for b in right_ranks) ** 0.5
    if left_denominator == 0.0 or right_denominator == 0.0:
        return 0.0
    return numerator / (left_denominator * right_denominator)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end) / 2.0
        for ordered_index in range(index, end + 1):
            ranks[ordered[ordered_index][0]] = average_rank
        index = end + 1
    return ranks


def _artifact_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _source_evidence(source: Path) -> tuple[MechanismSourceEvidence, ...]:
    trainer = source / "recipe/GraphGPO/graphgpo_ray_trainer.py"
    core_graph = source / "recipe/GraphGPO/core_graph.py"
    return (
        _quote_evidence(
            trainer,
            (
                "Gs = core_graph.ComplexGraph.from_data(",
                "step_rewards_tensor = core_graph.compute_graph_path_returns(",
                "batch.batch['step_rewards'] = step_rewards_tensor",
                "batch = compute_advantage(",
            ),
            "Mechanism probes must trace the same upstream graph construction, path-return, and advantage path used by training.",
        ),
        _quote_evidence(
            core_graph,
            (
                "step_group_uids = build_step_group(anchor_obs, index)",
                "step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)",
                "scores = step_advantage_w * step_advantages + episode_advantage_w * episode_advantages",
                "scores = step_advantage_w * step_advantages",
            ),
            "Mechanism metrics compare GraphGPO edge credit against exact-state counterfactual completion, and ablations isolate episode-only versus graph-only credit.",
        ),
    )


def _quote_evidence(path: Path, quotes: tuple[str, ...], decision: str) -> MechanismSourceEvidence:
    try:
        text = path.read_text(encoding="utf-8")
        digest = sha256_file(path)
        verified = all(quote in text for quote in quotes)
    except OSError:
        digest = ""
        verified = False
    return MechanismSourceEvidence(
        path=str(path),
        sha256=digest,
        source_quotes=quotes,
        decision=decision,
        verified=verified,
    )


def _ablation_purpose(ablation_id: str) -> str:
    purposes = {
        "episode_only": "isolate outcome-level episode credit without graph step credit",
        "graph_only": "isolate graph step credit without episode outcome credit",
        "similarity_merge": "test whether state canonicalization changes credit assignment",
    }
    return purposes.get(ablation_id, "unregistered ablation")
