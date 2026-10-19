"""Phase 5 raw GraphGPO trace ingestion.

This module is the seam between an upstream GraphGPO runner and the mechanism
report builder.  It does not run ALFWorld, Ray, or training.  It only converts
raw runner events into the canonical Phase 4 JSONL artifacts after checking
that the rows are study-bound, manifest-bound, and exact-state keyed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lh_platform.studies.protocol import StudyProtocol
from lh_platform.studies.runtime import AlfworldTaskManifest, sha256_file

RAW_TRACE_EVENT_SCHEMA = "lh_graphgpo_raw_trace_event/v1"
TRACE_ARTIFACT_BUNDLE_SCHEMA = "lh_graphgpo_trace_artifacts/v1"
EDGE_EVENT_TYPE = "edge_credit"
CONTINUATION_EVENT_TYPE = "counterfactual_continuation"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class TraceSourceEvidence:
    path: str
    sha256: str
    source_quotes: tuple[str, ...]
    decision: str
    verified: bool


@dataclass(frozen=True)
class GraphGPOTraceArtifacts:
    schema: str
    study_id: str
    ingestion_ready: bool
    source: str
    blocked_by: tuple[str, ...]
    raw_event_count: int
    row_counts: Mapping[str, int]
    input_artifacts: Mapping[str, Any]
    output_artifacts: Mapping[str, Mapping[str, Any]]
    source_evidence: tuple[TraceSourceEvidence, ...]
    edge_credit_trace: str
    counterfactual_continuations: str
    manifest_path: str


def build_graphgpo_trace_artifacts(
    protocol: StudyProtocol,
    *,
    task_manifest: str | Path,
    raw_trace: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    source_root: str | Path,
) -> GraphGPOTraceArtifacts:
    """Convert raw upstream GraphGPO trace events into Phase 4 artifacts."""

    raw_paths = _normalize_paths(raw_trace)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    edge_path = target / "edge_credit_trace.jsonl"
    continuation_path = target / "counterfactual_continuations.jsonl"
    manifest_path = target / "trace_artifact_manifest.json"

    blockers: list[str] = []
    source_evidence = graphgpo_trace_source_evidence(Path(source_root))
    if not all(evidence.verified for evidence in source_evidence):
        blockers.append("trace_source_unverified")

    horizon_by_task, manifest_errors = _load_task_horizons(Path(task_manifest))
    blockers.extend(manifest_errors)

    raw_rows, read_errors = _read_raw_trace_rows(raw_paths)
    blockers.extend(read_errors)
    if not raw_paths:
        blockers.append("raw_trace_not_provided")

    allowed_variants = {
        protocol.design.control.id,
        protocol.design.treatment.id,
        *tuple(str(item.get("id", "")) for item in protocol.design.ablations),
    }
    matched_rows = 0
    edge_rows: list[dict[str, Any]] = []
    continuation_rows: list[dict[str, Any]] = []
    edge_keys: set[tuple[Any, ...]] = set()
    continuation_keys: set[tuple[Any, ...]] = set()

    for row in raw_rows:
        row_study_id = _as_nonempty_str(row.get("study_id"))
        if row_study_id != protocol.id:
            blockers.append("raw_trace_study_id_mismatch")
            continue
        matched_rows += 1
        parsed = _parse_common_event(row)
        if parsed is None:
            blockers.append("raw_trace_schema_invalid")
            continue
        event_type = parsed["event_type"]
        task_id = parsed["task_id"]
        variant_id = parsed["variant_id"]
        if variant_id not in allowed_variants:
            blockers.append("raw_trace_variant_not_preregistered")
            continue
        if task_id not in horizon_by_task:
            blockers.append("raw_trace_task_not_in_manifest")
            continue
        horizon_bucket = horizon_by_task[task_id]
        raw_horizon = row.get("horizon_bucket")
        if raw_horizon is not None and raw_horizon != horizon_bucket:
            blockers.append("raw_trace_horizon_mismatch")
            continue

        if event_type == EDGE_EVENT_TYPE:
            edge = _parse_edge_event(row, parsed, horizon_bucket)
            if edge is None:
                blockers.append("raw_trace_schema_invalid")
                continue
            key = _canonical_edge_key(edge)
            if key in edge_keys:
                blockers.append("raw_trace_duplicate_edge_key")
                continue
            edge_keys.add(key)
            edge_rows.append(edge)
        elif event_type == CONTINUATION_EVENT_TYPE:
            continuation = _parse_continuation_event(row, parsed)
            if continuation is None:
                blockers.append("raw_trace_schema_invalid")
                continue
            key = (
                *_canonical_edge_key(continuation),
                continuation["rollout_index"],
            )
            if key in continuation_keys:
                blockers.append("raw_trace_duplicate_continuation_key")
                continue
            continuation_keys.add(key)
            continuation_rows.append(continuation)
        else:
            blockers.append("raw_trace_unknown_event_type")

    if matched_rows == 0:
        blockers.append("raw_trace_empty_for_study")
    if not edge_rows:
        blockers.append("edge_credit_trace_empty")
    if not continuation_rows:
        blockers.append("counterfactual_continuations_empty")

    blockers = list(dict.fromkeys(blockers))
    ingestion_ready = not blockers
    if not ingestion_ready:
        edge_rows = []
        continuation_rows = []

    _write_jsonl(edge_path, edge_rows)
    _write_jsonl(continuation_path, continuation_rows)

    bundle = GraphGPOTraceArtifacts(
        schema=TRACE_ARTIFACT_BUNDLE_SCHEMA,
        study_id=protocol.id,
        ingestion_ready=ingestion_ready,
        source="phase5_raw_graphgpo_trace_events",
        blocked_by=tuple(blockers),
        raw_event_count=matched_rows,
        row_counts={
            "edge_credit_trace": len(edge_rows),
            "counterfactual_continuations": len(continuation_rows),
        },
        input_artifacts={
            "task_manifest": _artifact_evidence(Path(task_manifest)),
            "raw_trace": [_artifact_evidence(path) for path in raw_paths],
        },
        output_artifacts={
            "edge_credit_trace": _artifact_evidence(edge_path),
            "counterfactual_continuations": _artifact_evidence(continuation_path),
        },
        source_evidence=source_evidence,
        edge_credit_trace=str(edge_path),
        counterfactual_continuations=str(continuation_path),
        manifest_path=str(manifest_path),
    )
    manifest_path.write_text(
        json.dumps(asdict(bundle), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def _normalize_paths(raw_trace: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(raw_trace, (str, Path)):
        return (Path(raw_trace),)
    return tuple(Path(path) for path in raw_trace)


def _load_task_horizons(path: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    if not path.is_file():
        return {}, (f"task_manifest_not_found:{path.name}",)
    try:
        manifest = AlfworldTaskManifest.read(path)
    except (OSError, TypeError, ValueError):
        return {}, (f"task_manifest_not_readable:{path.name}",)
    if manifest.schema != "lh_task_manifest/v1":
        return {}, ("unsupported_task_manifest_schema",)
    horizon_by_task: dict[str, str] = {}
    duplicates: set[str] = set()
    for task in manifest.tasks:
        if task.task_id in horizon_by_task:
            duplicates.add(task.task_id)
        horizon_by_task[task.task_id] = task.horizon_bucket
    if duplicates:
        return horizon_by_task, ("duplicate_task_id",)
    if not horizon_by_task:
        return horizon_by_task, ("empty_task_manifest",)
    return horizon_by_task, ()


def _read_raw_trace_rows(
    raw_paths: tuple[Path, ...],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in raw_paths:
        if not path.is_file():
            errors.append(f"raw_trace_not_found:{path.name}")
            continue
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    errors.append(f"raw_trace_json_invalid:{path.name}:{line_number}")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"raw_trace_row_not_object:{path.name}:{line_number}")
                    continue
                rows.append(row)
        except OSError:
            errors.append(f"raw_trace_not_readable:{path.name}")
    return tuple(rows), tuple(errors)


def _parse_common_event(row: Mapping[str, Any]) -> dict[str, Any] | None:
    schema = _as_nonempty_str(row.get("schema"))
    event_type = _as_nonempty_str(row.get("event_type"))
    study_id = _as_nonempty_str(row.get("study_id"))
    task_id = _as_nonempty_str(row.get("task_id"))
    seed = _as_int(row.get("seed"))
    variant_id = _as_nonempty_str(row.get("variant_id"))
    trajectory_id = _as_nonempty_str(row.get("trajectory_id"))
    edge_index = _as_int(row.get("edge_index"))
    candidate_action = _as_nonempty_str(row.get("candidate_action"))
    state_snapshot_sha256 = _as_sha256(row.get("state_snapshot_sha256"))
    if (
        schema != RAW_TRACE_EVENT_SCHEMA
        or event_type is None
        or study_id is None
        or task_id is None
        or seed is None
        or variant_id is None
        or trajectory_id is None
        or edge_index is None
        or candidate_action is None
        or state_snapshot_sha256 is None
    ):
        return None
    return {
        "event_type": event_type,
        "study_id": study_id,
        "task_id": task_id,
        "seed": seed,
        "variant_id": variant_id,
        "trajectory_id": trajectory_id,
        "edge_index": edge_index,
        "candidate_action": candidate_action,
        "state_snapshot_sha256": state_snapshot_sha256,
    }


def _parse_edge_event(
    row: Mapping[str, Any],
    common: Mapping[str, Any],
    horizon_bucket: str,
) -> dict[str, Any] | None:
    edge_advantage = _as_float(row.get("edge_advantage"))
    advantages = _as_float_list(row.get("advantages"))
    expert_action_match = _as_bool(row.get("expert_action_match"))
    trajectory_success = _as_bool(row.get("trajectory_success"))
    is_action_valid = _as_bool(row.get("is_action_valid"))
    if (
        edge_advantage is None
        or advantages is None
        or expert_action_match is None
        or trajectory_success is None
        or is_action_valid is None
    ):
        return None
    return {
        **{key: common[key] for key in _COMMON_CANONICAL_FIELDS},
        "horizon_bucket": horizon_bucket,
        "edge_advantage": edge_advantage,
        "advantages": advantages,
        "expert_action_match": expert_action_match,
        "trajectory_success": trajectory_success,
        "is_action_valid": is_action_valid,
    }


def _parse_continuation_event(
    row: Mapping[str, Any], common: Mapping[str, Any]
) -> dict[str, Any] | None:
    rollout_index = _as_int(row.get("rollout_index"))
    won = _as_bool(row.get("won"))
    if rollout_index is None or rollout_index < 0 or won is None:
        return None
    return {
        **{key: common[key] for key in _COMMON_CANONICAL_FIELDS},
        "rollout_index": rollout_index,
        "won": won,
    }


_COMMON_CANONICAL_FIELDS = (
    "study_id",
    "task_id",
    "seed",
    "variant_id",
    "trajectory_id",
    "edge_index",
    "candidate_action",
    "state_snapshot_sha256",
)


def _canonical_edge_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in _COMMON_CANONICAL_FIELDS)


def _as_nonempty_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _as_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not _SHA256_RE.fullmatch(stripped):
        return None
    return stripped.lower()


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    parsed = [_as_float(item) for item in value]
    if any(item is None for item in parsed):
        return None
    return [float(item) for item in parsed if item is not None]


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _artifact_evidence(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        evidence["sha256"] = sha256_file(path)
        evidence["size_bytes"] = path.stat().st_size
    return evidence


def graphgpo_trace_source_evidence(
    source_root: str | Path,
) -> tuple[TraceSourceEvidence, ...]:
    return _trace_source_evidence(Path(source_root))


def _trace_source_evidence(source_root: Path) -> tuple[TraceSourceEvidence, ...]:
    checks = (
        (
            "recipe/GraphGPO/graphgpo_ray_trainer.py",
            (
                "Gs = core_graph.ComplexGraph.from_data(",
                "step_rewards_tensor = core_graph.compute_graph_path_returns(",
                "batch = compute_advantage(",
            ),
            "raw trace events must be emitted after GraphGPO graph returns and advantages exist",
        ),
        (
            "recipe/GraphGPO/core_graph.py",
            (
                "step_group_uids = build_step_group(anchor_obs, index)",
                "scores = step_advantage_w * step_advantages + episode_advantage_w * episode_advantages",
                "is_action_valid_=batch.non_tensor_batch['is_action_valid']",
            ),
            "edge rows must expose state grouping, final advantages, and action validity",
        ),
        (
            "recipe/GraphGPO/alfworld_envs.py",
            ("reward = 10.0 * float(info['won'])",),
            "continuation rows must use the benchmark-owned ALFWorld win signal",
        ),
    )
    evidence: list[TraceSourceEvidence] = []
    for relative, quotes, decision in checks:
        path = source_root / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        verified = path.is_file() and all(quote in text for quote in quotes)
        evidence.append(
            TraceSourceEvidence(
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else "",
                source_quotes=quotes,
                decision=decision,
                verified=verified,
            )
        )
    return tuple(evidence)
