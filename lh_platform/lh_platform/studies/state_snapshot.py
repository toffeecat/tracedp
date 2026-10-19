"""Build a non-invasive exact-state snapshot provider for GraphGPO ALFWorld.

The provider is deliberately conservative.  It can emit a 64-hex key only when
the worker state is serializable as a restorable payload.  The manifest remains
blocked until a live restore probe proves that those bytes can restore the real
ALFWorld environment, not merely hash observation text.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from lh_platform.studies.protocol import StudyProtocol
from lh_platform.studies.runtime import sha256_file
from lh_platform.studies.trace import TraceSourceEvidence

STATE_SNAPSHOT_PROVIDER_SCHEMA = "lh_graphgpo_state_snapshot_provider/v1"
STATE_SNAPSHOT_SCHEMA = "lh_graphgpo_exact_state_snapshot/v1"
PROVIDER_MODULE_NAME = "lh_graphgpo_state_snapshot_provider.py"
INSTRUMENTATION_PATCH_NAME = "graphgpo_state_snapshot_instrumentation.patch"


@dataclass(frozen=True)
class GraphGPOStateSnapshotProvider:
    schema: str
    study_id: str
    provider_ready: bool
    source: str
    blocked_by: tuple[str, ...]
    state_snapshot_schema: str
    exact_state_contract: Mapping[str, Any]
    required_environment: Mapping[str, str]
    provider_module: str
    instrumentation_patch: str
    manifest_path: str
    source_evidence: tuple[TraceSourceEvidence, ...]
    output_artifacts: Mapping[str, Mapping[str, Any]]


def build_graphgpo_state_snapshot_provider(
    protocol: StudyProtocol,
    *,
    source_root: str | Path,
    output_dir: str | Path,
) -> GraphGPOStateSnapshotProvider:
    """Write the exact-state provider and patch artifacts."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    module_path = target / PROVIDER_MODULE_NAME
    patch_path = target / INSTRUMENTATION_PATCH_NAME
    manifest_path = target / "state_snapshot_provider_manifest.json"

    source = Path(source_root)
    source_evidence = _state_snapshot_source_evidence(source)
    blocked_by: list[str] = []
    if not all(item.verified for item in source_evidence):
        blocked_by.append("state_snapshot_source_unverified")
    blocked_by.append("live_state_restore_probe_not_run")

    module_path.write_text(_PROVIDER_SOURCE, encoding="utf-8")
    patch_path.write_text(_instrumentation_patch(source), encoding="utf-8")

    provider = GraphGPOStateSnapshotProvider(
        schema=STATE_SNAPSHOT_PROVIDER_SCHEMA,
        study_id=protocol.id,
        provider_ready=not blocked_by,
        source="phase7_noninvasive_graphgpo_state_snapshot_provider",
        blocked_by=tuple(blocked_by),
        state_snapshot_schema=STATE_SNAPSHOT_SCHEMA,
        exact_state_contract={
            "key": "state_snapshot_sha256",
            "payload_source": "serialized AlfworldWorker.env plus wrapper state",
            "observation_hash_allowed": False,
            "restore_probe_required": True,
            "blocked_when_unproven": "live_state_restore_probe_not_run",
        },
        required_environment={
            "PYTHONPATH": ("prepend directory containing lh_graphgpo_state_snapshot_provider.py")
        },
        provider_module=str(module_path),
        instrumentation_patch=str(patch_path),
        manifest_path=str(manifest_path),
        source_evidence=source_evidence,
        output_artifacts={
            "provider_module": _artifact_evidence(module_path),
            "instrumentation_patch": _artifact_evidence(patch_path),
        },
    )
    provider = replace(
        provider,
        output_artifacts={
            **dict(provider.output_artifacts),
            "provider_manifest": {
                "path": str(manifest_path),
                "self_hash_omitted": True,
                "reason": "a manifest cannot contain a stable hash of itself",
            },
        },
    )
    manifest_path.write_text(
        json.dumps(asdict(provider), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provider


def _artifact_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _state_snapshot_source_evidence(
    source_root: Path,
) -> tuple[TraceSourceEvidence, ...]:
    checks = (
        (
            "recipe/GraphGPO/alfworld_envs.py",
            (
                "class AlfworldWorker:",
                "self.env = base_env.init_env(batch_size=1)",
                "obs, scores, dones, infos = self.env.step(actions)",
                "infos['observation_text'] = obs",
            ),
            "snapshot provider must run inside AlfworldWorker where the underlying environment and wrapper state coexist",
        ),
        (
            "recipe/GraphGPO/rollout_loop.py",
            (
                "next_obs, rewards, dones, infos = envs.step(text_actions)",
                "batch.non_tensor_batch['is_action_valid']",
            ),
            "rollout loop must propagate state_snapshot_sha256 from infos into the batch",
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


def _instrumentation_patch(source_root: Path) -> str:
    edits = {
        "recipe/GraphGPO/alfworld_envs.py": _instrument_alfworld_worker,
        "recipe/GraphGPO/rollout_loop.py": _instrument_rollout_loop,
    }
    chunks: list[str] = []
    for relative, edit in edits.items():
        path = source_root / relative
        if not path.is_file():
            continue
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
    return "\n".join(chunks) + ("\n" if chunks else "")


def _instrument_alfworld_worker(original: str) -> str:
    import_anchor = "import re\n"
    import_line = "from lh_graphgpo_state_snapshot_provider import worker_state_snapshot_sha256\n"
    if import_line not in original:
        original = original.replace(import_anchor, import_anchor + import_line, 1)

    step_anchor = "        actions = [action] \n"
    step_snapshot = (
        "        pre_action_state_snapshot_sha256 = worker_state_snapshot_sha256(self)\n"
    )
    if step_snapshot not in original:
        original = original.replace(
            step_anchor,
            step_anchor + step_snapshot,
            1,
        )

    step_info_anchor = (
        "        infos['item_location']=[self.item_location]\n"
        "        return obs, scores, dones, infos\n"
    )
    step_info_snapshot = (
        "        if pre_action_state_snapshot_sha256:\n"
        "            infos['state_snapshot_sha256'] = [pre_action_state_snapshot_sha256]\n"
    )
    if step_info_snapshot not in original:
        original = original.replace(
            step_info_anchor,
            "        infos['item_location']=[self.item_location]\n"
            + step_info_snapshot
            + "        return obs, scores, dones, infos\n",
            1,
        )

    reset_anchor = (
        "        infos['item_location']=[self.item_location]\n"
        "        infos['observation_text'] = obs\n"
        "        return obs, infos\n"
    )
    reset_snapshot = (
        "        state_snapshot_sha256 = worker_state_snapshot_sha256(self)\n"
        "        if state_snapshot_sha256:\n"
        "            infos['state_snapshot_sha256'] = [state_snapshot_sha256]\n"
    )
    if reset_snapshot not in original:
        original = original.replace(
            reset_anchor,
            "        infos['item_location']=[self.item_location]\n"
            "        infos['observation_text'] = obs\n"
            + reset_snapshot
            + "        return obs, infos\n",
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


_PROVIDER_SOURCE = r'''"""Exact-state snapshot helper for GraphGPO ALFWorld workers.

This sidecar refuses to hash observation text.  It only emits a key when the
underlying worker environment and wrapper state can be serialized together.
Live restore must still be probed before treating the key as launch-ready.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from typing import Any


STATE_SNAPSHOT_SCHEMA = "lh_graphgpo_exact_state_snapshot/v1"


def worker_state_snapshot_sha256(worker: Any) -> str | None:
    payload = worker_state_snapshot_payload(worker)
    if payload is None:
        return None
    return hashlib.sha256(payload).hexdigest()


def worker_state_snapshot_payload(worker: Any) -> bytes | None:
    try:
        snapshot = {
            "schema": STATE_SNAPSHOT_SCHEMA,
            "env": worker.env,
            "wrapper_state": {
                "history_items": copy.deepcopy(getattr(worker, "history_items")),
                "location": copy.deepcopy(getattr(worker, "location")),
                "holding": copy.deepcopy(getattr(worker, "holding")),
                "item_location": copy.deepcopy(getattr(worker, "item_location")),
                "history_list": copy.deepcopy(getattr(worker, "history_list")),
                "done": copy.deepcopy(getattr(worker, "done")),
            },
        }
        payload = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.loads(payload)
        return payload
    except Exception:
        return None
'''
