"""Build a non-invasive GraphGPO trace sidecar package.

The platform must not modify the official GraphGPO repository in-place.  Phase
6 therefore emits a sidecar module plus an auditable patch/instruction artifact
that a runner can add to its PYTHONPATH and apply in a controlled workspace.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lh_platform.studies.protocol import StudyProtocol
from lh_platform.studies.runtime import sha256_file
from lh_platform.studies.trace import (
    RAW_TRACE_EVENT_SCHEMA,
    TRACE_ARTIFACT_BUNDLE_SCHEMA,
    TraceSourceEvidence,
    graphgpo_trace_source_evidence,
)

TRACE_SIDECAR_SCHEMA = "lh_graphgpo_trace_sidecar/v1"
SIDECAR_MODULE_NAME = "lh_graphgpo_trace_sidecar.py"
INSTRUMENTATION_PATCH_NAME = "graphgpo_trace_instrumentation.patch"


@dataclass(frozen=True)
class GraphGPOTraceSidecar:
    schema: str
    study_id: str
    sidecar_ready: bool
    source: str
    blocked_by: tuple[str, ...]
    raw_trace_schema: str
    canonical_artifact_schema: str
    required_environment: Mapping[str, str]
    state_snapshot_contract: Mapping[str, Any]
    sidecar_module: str
    instrumentation_patch: str
    manifest_path: str
    source_evidence: tuple[TraceSourceEvidence, ...]
    output_artifacts: Mapping[str, Mapping[str, Any]]


def build_graphgpo_trace_sidecar(
    protocol: StudyProtocol,
    *,
    source_root: str | Path,
    output_dir: str | Path,
) -> GraphGPOTraceSidecar:
    """Write the sidecar emitter and a patch artifact for the official runner."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    module_path = target / SIDECAR_MODULE_NAME
    patch_path = target / INSTRUMENTATION_PATCH_NAME
    manifest_path = target / "trace_sidecar_manifest.json"

    source = Path(source_root)
    source_evidence = graphgpo_trace_source_evidence(source)
    state_snapshot_verified = _source_has_state_snapshot_key(source)
    blocked_by: list[str] = []
    if not all(item.verified for item in source_evidence):
        blocked_by.append("trace_source_unverified")
    if not state_snapshot_verified:
        blocked_by.append("state_snapshot_key_unverified")

    module_path.write_text(_SIDECAR_SOURCE, encoding="utf-8")
    patch_path.write_text(_instrumentation_patch(source), encoding="utf-8")

    bundle = GraphGPOTraceSidecar(
        schema=TRACE_SIDECAR_SCHEMA,
        study_id=protocol.id,
        sidecar_ready=not blocked_by,
        source="phase6_noninvasive_graphgpo_sidecar",
        blocked_by=tuple(blocked_by),
        raw_trace_schema=RAW_TRACE_EVENT_SCHEMA,
        canonical_artifact_schema=TRACE_ARTIFACT_BUNDLE_SCHEMA,
        required_environment={
            "LH_GRAPHGPO_RAW_TRACE_PATH": "absolute path to raw_trace.jsonl",
            "LH_STUDY_ID": protocol.id,
            "LH_VARIANT_ID": "prepared trial variant id",
            "LH_SEED": "prepared trial seed",
            "PYTHONPATH": "prepend directory containing lh_graphgpo_trace_sidecar.py",
        },
        state_snapshot_contract={
            "required_batch_field": "state_snapshot_sha256",
            "verified": state_snapshot_verified,
            "blocked_when_missing": "state_snapshot_key_unverified",
            "reason": (
                "exact-state mechanism probes must not use observation hashes as "
                "restorable environment state keys"
            ),
        },
        sidecar_module=str(module_path),
        instrumentation_patch=str(patch_path),
        manifest_path=str(manifest_path),
        source_evidence=source_evidence,
        output_artifacts={
            "sidecar_module": _artifact_evidence(module_path),
            "instrumentation_patch": _artifact_evidence(patch_path),
        },
    )
    manifest_path.write_text(
        json.dumps(asdict(bundle), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def _artifact_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _source_has_state_snapshot_key(source_root: Path) -> bool:
    candidates = (
        source_root / "recipe/GraphGPO/rollout_loop.py",
        source_root / "recipe/GraphGPO/graphgpo_ray_trainer.py",
    )
    for path in candidates:
        if path.is_file() and "state_snapshot_sha256" in path.read_text(encoding="utf-8"):
            return True
    return False


def _instrumentation_patch(source_root: Path) -> str:
    edits = {
        "recipe/GraphGPO/graphgpo_ray_trainer.py": _instrument_trainer,
        "recipe/GraphGPO/rollout_loop.py": _instrument_rollout_loop,
    }
    chunks: list[str] = []
    for relative, edit in edits.items():
        path = source_root / relative
        original = path.read_text(encoding="utf-8")
        modified = edit(original)
        chunks.extend(
            difflib.unified_diff(
                original.splitlines(),
                modified.splitlines(),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
                lineterm="",
                n=20,
            )
        )
    return "\n".join(chunks) + "\n"


def _instrument_trainer(original: str) -> str:
    import_anchor = "from . import core_graph\n"
    if "from lh_graphgpo_trace_sidecar import GraphGPOTraceEmitter\n" not in original:
        original = original.replace(
            import_anchor,
            import_anchor + "from lh_graphgpo_trace_sidecar import GraphGPOTraceEmitter\n",
            1,
        )
    advantage_anchor = (
        '                            gigpo_mode=self.config.algorithm.get("graphgpo", {}).get("mode", "mean_std_norm"),\n'
        "                            epsilon=1e-6,\n"
        "                        )\n"
    )
    emitter_call = (
        "                        GraphGPOTraceEmitter.from_env().emit_edge_credit_batch(batch)\n"
    )
    if emitter_call not in original:
        original = original.replace(
            advantage_anchor,
            advantage_anchor + emitter_call,
            1,
        )
    return original


def _instrument_rollout_loop(original: str) -> str:
    action_valid_anchor = (
        "            if 'is_action_valid' in infos[0]:\n"
        "                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)\n"
        "            else:\n"
        "                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)\n"
    )
    state_snapshot_block = (
        "\n"
        "            if 'state_snapshot_sha256' in infos[0]:\n"
        "                batch.non_tensor_batch['state_snapshot_sha256'] = np.array([info['state_snapshot_sha256'] for info in infos], dtype=object)\n"
    )
    if state_snapshot_block not in original:
        original = original.replace(
            action_valid_anchor,
            action_valid_anchor + state_snapshot_block,
            1,
        )
    return original


_SIDECAR_SOURCE = r'''"""Sidecar emitter for GraphGPO raw mechanism trace events.

This module intentionally has no dependency on lh_platform, torch, numpy, Ray,
or ALFWorld.  Put the file on PYTHONPATH for an instrumented GraphGPO run and
set LH_GRAPHGPO_RAW_TRACE_PATH, LH_STUDY_ID, LH_VARIANT_ID, and LH_SEED.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


RAW_TRACE_EVENT_SCHEMA = "lh_graphgpo_raw_trace_event/v1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class GraphGPOTraceEmitter:
    def __init__(
        self,
        output_path: str | Path,
        *,
        study_id: str,
        variant_id: str,
        seed: int,
    ) -> None:
        self.output_path = Path(output_path)
        self.study_id = study_id
        self.variant_id = variant_id
        self.seed = int(seed)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "GraphGPOTraceEmitter | _NullGraphGPOTraceEmitter":
        output_path = os.environ.get("LH_GRAPHGPO_RAW_TRACE_PATH")
        study_id = os.environ.get("LH_STUDY_ID")
        variant_id = os.environ.get("LH_VARIANT_ID")
        seed = os.environ.get("LH_SEED")
        if not output_path or not study_id or not variant_id or seed is None:
            return _NullGraphGPOTraceEmitter()
        try:
            seed_value = int(seed)
        except ValueError:
            return _NullGraphGPOTraceEmitter()
        return cls(
            output_path,
            study_id=study_id,
            variant_id=variant_id,
            seed=seed_value,
        )

    def emit_edge_credit_event(
        self,
        *,
        task_id: str,
        trajectory_id: str,
        edge_index: int,
        candidate_action: str,
        state_snapshot_sha256: str,
        edge_advantage: float,
        advantages: list[float],
        expert_action_match: bool,
        trajectory_success: bool,
        is_action_valid: bool,
    ) -> None:
        state_snapshot_sha256 = _valid_sha256_or_none(state_snapshot_sha256)
        if state_snapshot_sha256 is None:
            return
        self._write(
            {
                "schema": RAW_TRACE_EVENT_SCHEMA,
                "event_type": "edge_credit",
                "study_id": self.study_id,
                "task_id": task_id,
                "seed": self.seed,
                "variant_id": self.variant_id,
                "trajectory_id": trajectory_id,
                "edge_index": int(edge_index),
                "candidate_action": candidate_action,
                "state_snapshot_sha256": state_snapshot_sha256,
                "edge_advantage": float(edge_advantage),
                "advantages": [float(value) for value in advantages],
                "expert_action_match": bool(expert_action_match),
                "trajectory_success": bool(trajectory_success),
                "is_action_valid": bool(is_action_valid),
            }
        )

    def emit_counterfactual_continuation(
        self,
        *,
        task_id: str,
        trajectory_id: str,
        edge_index: int,
        candidate_action: str,
        state_snapshot_sha256: str,
        rollout_index: int,
        won: bool,
    ) -> None:
        state_snapshot_sha256 = _valid_sha256_or_none(state_snapshot_sha256)
        if state_snapshot_sha256 is None:
            return
        self._write(
            {
                "schema": RAW_TRACE_EVENT_SCHEMA,
                "event_type": "counterfactual_continuation",
                "study_id": self.study_id,
                "task_id": task_id,
                "seed": self.seed,
                "variant_id": self.variant_id,
                "trajectory_id": trajectory_id,
                "edge_index": int(edge_index),
                "candidate_action": candidate_action,
                "state_snapshot_sha256": state_snapshot_sha256,
                "rollout_index": int(rollout_index),
                "won": bool(won),
            }
        )

    def emit_edge_credit_batch(self, batch: Any) -> None:
        non_tensor = getattr(batch, "non_tensor_batch", {}) or {}
        tensor_batch = getattr(batch, "batch", {}) or {}
        state_snapshots = _sequence(non_tensor.get("state_snapshot_sha256"))
        if not state_snapshots:
            # Exact-state evidence is mandatory.  Do not emit observation-hash
            # proxies because Phase 5 treats state_snapshot_sha256 as the
            # restored-environment state key.
            return
        task_ids = _sequence(non_tensor.get("uid"))
        trajectory_ids = _sequence(non_tensor.get("traj_uid"))
        actions = _sequence(non_tensor.get("text_actions"))
        valid_actions = _sequence(non_tensor.get("is_action_valid"))
        episode_rewards = _sequence(non_tensor.get("episode_rewards"))
        advantages = _matrix(tensor_batch.get("advantages"))
        if not advantages:
            return
        row_count = min(
            len(advantages),
            len(task_ids),
            len(trajectory_ids),
            len(actions),
            len(state_snapshots),
        )
        for index in range(row_count):
            row_advantages = [float(value) for value in advantages[index]]
            if not row_advantages:
                continue
            self.emit_edge_credit_event(
                task_id=str(task_ids[index]),
                trajectory_id=str(trajectory_ids[index]),
                edge_index=index,
                candidate_action=str(actions[index]),
                state_snapshot_sha256=str(state_snapshots[index]),
                edge_advantage=float(row_advantages[0]),
                advantages=row_advantages,
                expert_action_match=False,
                trajectory_success=float(episode_rewards[index]) > 0.0
                if index < len(episode_rewards)
                else False,
                is_action_valid=bool(valid_actions[index])
                if index < len(valid_actions)
                else True,
            )

    def _write(self, row: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class _NullGraphGPOTraceEmitter:
    def emit_edge_credit_event(self, **_: Any) -> None:
        return None

    def emit_counterfactual_continuation(self, **_: Any) -> None:
        return None

    def emit_edge_credit_batch(self, _: Any) -> None:
        return None


def snapshot_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _valid_sha256_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not _SHA256_RE.fullmatch(stripped):
        return None
    return stripped.lower()


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _matrix(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    rows: list[list[float]] = []
    for row in value:
        if isinstance(row, list):
            rows.append([float(item) for item in row])
        else:
            rows.append([float(row)])
    return rows
'''
