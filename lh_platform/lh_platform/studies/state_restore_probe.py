"""Build and evaluate a live restore probe for GraphGPO ALFWorld snapshots."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from lh_platform.studies.protocol import StudyProtocol
from lh_platform.studies.runtime import sha256_file
from lh_platform.studies.trace import TraceSourceEvidence

STATE_RESTORE_PROBE_MANIFEST_SCHEMA = "lh_graphgpo_state_restore_probe_manifest/v1"
STATE_RESTORE_PROBE_RESULT_SCHEMA = "lh_graphgpo_state_restore_probe_result/v1"
STATE_RESTORE_PROBE_REPORT_SCHEMA = "lh_graphgpo_state_restore_probe_report/v1"
PROBE_MODULE_NAME = "lh_graphgpo_state_restore_probe.py"
REQUIRED_COMPARISONS = ("observation", "reward", "done", "won", "wrapper_state")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REWARD_PROJECTION = "graphgpo_compute_reward"
REWARD_PROJECTION_SOURCE = "upstream_compute_reward"


@dataclass(frozen=True)
class GraphGPOStateRestoreProbe:
    schema: str
    study_id: str
    probe_ready: bool
    source: str
    blocked_by: tuple[str, ...]
    restore_probe_schema: str
    exact_restore_contract: Mapping[str, Any]
    required_environment: Mapping[str, str]
    probe_module: str
    manifest_path: str
    source_evidence: tuple[TraceSourceEvidence, ...]
    output_artifacts: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class GraphGPOStateRestoreProbeReport:
    schema: str
    study_id: str
    restore_verified: bool
    blocked_by: tuple[str, ...]
    result_path: str
    report_path: str
    result_summary: Mapping[str, Any]
    input_artifacts: Mapping[str, Mapping[str, Any]]


def build_graphgpo_state_restore_probe(
    protocol: StudyProtocol,
    *,
    source_root: str | Path,
    output_dir: str | Path,
) -> GraphGPOStateRestoreProbe:
    """Write the live restore probe sidecar and blocked manifest."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    module_path = target / PROBE_MODULE_NAME
    manifest_path = target / "state_restore_probe_manifest.json"

    source = Path(source_root)
    source_evidence = _state_restore_source_evidence(source)
    blocked_by: list[str] = []
    if not all(item.verified for item in source_evidence):
        blocked_by.append("state_restore_probe_source_unverified")
    blocked_by.append("live_state_restore_probe_not_run")

    module_path.write_text(_PROBE_SOURCE, encoding="utf-8")
    probe = GraphGPOStateRestoreProbe(
        schema=STATE_RESTORE_PROBE_MANIFEST_SCHEMA,
        study_id=protocol.id,
        probe_ready=not blocked_by,
        source="phase8_live_graphgpo_state_restore_probe",
        blocked_by=tuple(blocked_by),
        restore_probe_schema=STATE_RESTORE_PROBE_RESULT_SCHEMA,
        exact_restore_contract={
            "snapshot_payload": "serialized AlfworldWorker.env plus wrapper state",
            "verification": (
                "restore the pre-action snapshot and replay the same action; "
                "observation, reward, done, won, and wrapper_state must match"
            ),
            "comparison_fields": REQUIRED_COMPARISONS,
            "reward_projection": REWARD_PROJECTION,
            "observation_hash_allowed": False,
            "blocked_when_missing": "live_state_restore_probe_not_run",
        },
        required_environment={
            "PYTHONPATH": ("prepend directory containing lh_graphgpo_state_restore_probe.py")
        },
        probe_module=str(module_path),
        manifest_path=str(manifest_path),
        source_evidence=source_evidence,
        output_artifacts={"probe_module": _artifact_evidence(module_path)},
    )
    probe = replace(
        probe,
        output_artifacts={
            **dict(probe.output_artifacts),
            "probe_manifest": {
                "path": str(manifest_path),
                "self_hash_omitted": True,
                "reason": "a manifest cannot contain a stable hash of itself",
            },
        },
    )
    manifest_path.write_text(
        json.dumps(asdict(probe), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return probe


def build_graphgpo_state_restore_probe_report(
    protocol: StudyProtocol,
    *,
    result_path: str | Path,
    output_path: str | Path,
    probe_manifest_path: str | Path,
) -> GraphGPOStateRestoreProbeReport:
    """Evaluate a sidecar restore-probe result and write a fail-closed report."""

    result = _read_probe_result(Path(result_path))
    probe_manifest = _read_probe_manifest(Path(probe_manifest_path))
    blockers = _probe_result_blockers(protocol.id, result, probe_manifest)
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    restore_verified = not blockers
    summary = _probe_result_summary(result)
    report = GraphGPOStateRestoreProbeReport(
        schema=STATE_RESTORE_PROBE_REPORT_SCHEMA,
        study_id=protocol.id,
        restore_verified=restore_verified,
        blocked_by=tuple(blockers),
        result_path=str(result_path),
        report_path=str(report_path),
        result_summary=summary,
        input_artifacts={
            "probe_result": _artifact_evidence(Path(result_path))
            if Path(result_path).is_file()
            else {"path": str(result_path), "missing": True},
            "probe_manifest": _artifact_evidence(Path(probe_manifest_path))
            if Path(probe_manifest_path).is_file()
            else {"path": str(probe_manifest_path), "missing": True},
        },
    )
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _read_probe_result(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {
            "schema": STATE_RESTORE_PROBE_RESULT_SCHEMA,
            "restore_verified": False,
            "read_error": repr(error),
        }
    if not isinstance(payload, Mapping):
        return {
            "schema": STATE_RESTORE_PROBE_RESULT_SCHEMA,
            "restore_verified": False,
            "read_error": "probe result top-level is not a mapping",
        }
    return payload


def _read_probe_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"read_error": repr(error)}
    if not isinstance(payload, Mapping):
        return {"read_error": "probe manifest top-level is not a mapping"}
    return payload


def _probe_result_blockers(
    study_id: str,
    result: Mapping[str, Any],
    probe_manifest: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    expected_producer_sha256 = _expected_probe_module_sha256(probe_manifest)
    blockers.extend(_probe_manifest_blockers(study_id, probe_manifest))
    if expected_producer_sha256 is None:
        blockers.append("state_restore_probe_manifest_invalid")
    if result.get("schema") != STATE_RESTORE_PROBE_RESULT_SCHEMA:
        blockers.append("state_restore_probe_schema_mismatch")
    if result.get("producer") != PROBE_MODULE_NAME:
        blockers.append("state_restore_probe_producer_mismatch")
    if result.get("producer_sha256") != expected_producer_sha256:
        blockers.append("state_restore_probe_producer_hash_mismatch")
    if result.get("study_id") != study_id:
        blockers.append("state_restore_probe_study_mismatch")
    if result.get("reward_projection") != REWARD_PROJECTION:
        blockers.append("state_restore_probe_reward_projection_mismatch")
    if result.get("reward_projection_source") != REWARD_PROJECTION_SOURCE:
        blockers.append("state_restore_probe_reward_source_mismatch")
    if result.get("multi_modal_source") not in ("argument", "worker_attribute"):
        blockers.append("state_restore_probe_multi_modal_unbound")
    reported_blockers = result.get("blocked_by")
    if not isinstance(reported_blockers, list):
        blockers.append("state_restore_probe_blocked_by_missing")
    elif reported_blockers:
        blockers.append("state_restore_probe_reported_blockers")
    snapshot = result.get("pre_action_snapshot_sha256")
    if not isinstance(snapshot, str) or not _SHA256_RE.match(snapshot):
        blockers.append("state_restore_probe_snapshot_invalid")
    comparisons = result.get("comparisons")
    if not isinstance(comparisons, Mapping):
        blockers.append("state_restore_probe_comparisons_missing")
    else:
        missing = [key for key in REQUIRED_COMPARISONS if key not in comparisons]
        if missing:
            blockers.append("state_restore_probe_comparisons_missing")
        if any(comparisons.get(key) is not True for key in REQUIRED_COMPARISONS):
            blockers.append("state_restore_probe_failed")
    if result.get("restore_verified") is not True:
        blockers.append("state_restore_probe_failed")
    if result.get("read_error"):
        blockers.append("state_restore_probe_result_unreadable")
    return sorted(set(blockers))


def _probe_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = result.get("comparisons")
    return {
        "restore_verified": result.get("restore_verified") is True,
        "producer": result.get("producer"),
        "producer_sha256": result.get("producer_sha256"),
        "study_id": result.get("study_id"),
        "reward_projection": result.get("reward_projection"),
        "reward_projection_source": result.get("reward_projection_source"),
        "multi_modal": result.get("multi_modal"),
        "multi_modal_source": result.get("multi_modal_source"),
        "pre_action_snapshot_sha256": result.get("pre_action_snapshot_sha256"),
        "comparisons": dict(comparisons) if isinstance(comparisons, Mapping) else None,
        "blocked_by": list(result.get("blocked_by") or ()),
    }


def _artifact_evidence(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"path": str(path)}
    if path.is_file():
        evidence.update(
            {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    else:
        evidence["missing"] = True
    return evidence


def _expected_probe_module_sha256(probe_manifest: Mapping[str, Any]) -> str | None:
    artifacts = probe_manifest.get("output_artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    module = artifacts.get("probe_module")
    if not isinstance(module, Mapping):
        return None
    digest = module.get("sha256")
    if isinstance(digest, str) and _SHA256_RE.match(digest):
        return digest
    return None


def _probe_manifest_blockers(
    study_id: str,
    probe_manifest: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if probe_manifest.get("schema") != STATE_RESTORE_PROBE_MANIFEST_SCHEMA:
        blockers.append("state_restore_probe_manifest_schema_mismatch")
    if probe_manifest.get("study_id") != study_id:
        blockers.append("state_restore_probe_manifest_study_mismatch")
    if probe_manifest.get("restore_probe_schema") != STATE_RESTORE_PROBE_RESULT_SCHEMA:
        blockers.append("state_restore_probe_manifest_result_schema_mismatch")
    evidence = probe_manifest.get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        blockers.append("state_restore_probe_manifest_source_unverified")
    elif any(not isinstance(item, Mapping) or not item.get("verified") for item in evidence):
        blockers.append("state_restore_probe_manifest_source_unverified")
    if probe_manifest.get("read_error"):
        blockers.append("state_restore_probe_manifest_unreadable")
    return blockers


def _state_restore_source_evidence(
    source_root: Path,
) -> tuple[TraceSourceEvidence, ...]:
    checks = (
        (
            "recipe/GraphGPO/alfworld_envs.py",
            (
                "self.env = base_env.init_env(batch_size=1)",
                "obs, scores, dones, infos = self.env.step(actions)",
                "infos['observation_text'] = obs",
                "infos['items_info']=[self.history_items]",
                "def compute_reward(info, multi_modal=False):",
                "rewards_list.append(compute_reward(info, self.multi_modal))",
            ),
            "restore probe must run at the AlfworldWorker layer because step outcome depends on both env and wrapper state",
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


_PROBE_SOURCE = r'''"""Live restore probe helper for GraphGPO ALFWorld workers.

Import this module inside an isolated GraphGPO/ALFWorld process.  It verifies
that a serialized pre-action worker snapshot can be restored and replayed with
the same action to produce the same outcome.  It deliberately does not use an
observation hash as a state proxy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


STATE_SNAPSHOT_SCHEMA = "lh_graphgpo_exact_state_snapshot/v1"
STATE_RESTORE_PROBE_RESULT_SCHEMA = "lh_graphgpo_state_restore_probe_result/v1"
PROBE_MODULE_NAME = "lh_graphgpo_state_restore_probe.py"
REWARD_PROJECTION = "graphgpo_compute_reward"
REWARD_PROJECTION_SOURCE = "upstream_compute_reward"
FALLBACK_REWARD_PROJECTION_SOURCE = "local_fallback_compute_reward"
WRAPPER_FIELDS = (
    "history_items",
    "location",
    "holding",
    "item_location",
    "history_list",
    "done",
)
COMPARISON_FIELDS = ("observation", "reward", "done", "won", "wrapper_state")


try:
    from recipe.GraphGPO.alfworld_envs import compute_reward as _UPSTREAM_COMPUTE_REWARD
except Exception:
    _UPSTREAM_COMPUTE_REWARD = None


def worker_state_snapshot_payload(worker: Any) -> bytes | None:
    try:
        snapshot = {
            "schema": STATE_SNAPSHOT_SCHEMA,
            "env": worker.env,
            "wrapper_state": {
                field: copy.deepcopy(getattr(worker, field))
                for field in WRAPPER_FIELDS
            },
        }
        payload = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.loads(payload)
        return payload
    except Exception:
        return None


def worker_state_snapshot_sha256(worker: Any) -> str | None:
    payload = worker_state_snapshot_payload(worker)
    if payload is None:
        return None
    return hashlib.sha256(payload).hexdigest()


def restore_worker_state(worker: Any, payload: bytes) -> bool:
    try:
        snapshot = pickle.loads(payload)
        if snapshot.get("schema") != STATE_SNAPSHOT_SCHEMA:
            return False
        wrapper_state = snapshot["wrapper_state"]
        worker.env = snapshot["env"]
        for field in WRAPPER_FIELDS:
            setattr(worker, field, copy.deepcopy(wrapper_state[field]))
        return True
    except Exception:
        return False


def probe_worker_restore(
    worker: Any,
    action: Any,
    *,
    study_id: str | None = None,
    multi_modal: bool | None = None,
) -> dict[str, Any]:
    payload = worker_state_snapshot_payload(worker)
    if payload is None:
        return _blocked_result("snapshot_payload_unavailable", study_id=study_id)

    pre_action_snapshot_sha256 = hashlib.sha256(payload).hexdigest()
    multi_modal_value, multi_modal_source = _resolve_multi_modal(worker, multi_modal)
    first = _run_step_projection(worker, action, multi_modal=multi_modal_value)
    restored = restore_worker_state(worker, payload)
    if not restored:
        return _blocked_result(
            "snapshot_restore_failed",
            study_id=study_id,
            pre_action_snapshot_sha256=pre_action_snapshot_sha256,
        )

    second = _run_step_projection(worker, action, multi_modal=multi_modal_value)
    comparisons = {
        field: first[field] == second[field]
        for field in COMPARISON_FIELDS
    }
    if first.get("reward_error") or second.get("reward_error"):
        comparisons["reward"] = False
    reward_projection_source = first.get("reward_projection_source")
    if reward_projection_source != second.get("reward_projection_source"):
        reward_projection_source = "reward_projection_source_mismatch"
    restore_verified = all(comparisons.values())
    blocked_by = [] if restore_verified else ["state_restore_probe_failed"]
    result = {
        "schema": STATE_RESTORE_PROBE_RESULT_SCHEMA,
        "producer": PROBE_MODULE_NAME,
        "producer_sha256": _producer_sha256(),
        "reward_projection": REWARD_PROJECTION,
        "reward_projection_source": reward_projection_source,
        "multi_modal": multi_modal_value,
        "multi_modal_source": multi_modal_source,
        "restore_verified": restore_verified,
        "blocked_by": blocked_by,
        "pre_action_snapshot_sha256": pre_action_snapshot_sha256,
        "comparisons": comparisons,
        "first_step": first,
        "second_step": second,
    }
    if study_id is not None:
        result["study_id"] = study_id
    return result


def write_probe_result(
    worker: Any,
    action: Any,
    output_path: str | Path,
    *,
    study_id: str | None = None,
    multi_modal: bool | None = None,
) -> dict[str, Any]:
    result = probe_worker_restore(
        worker,
        action,
        study_id=study_id,
        multi_modal=multi_modal,
    )
    Path(output_path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _run_step_projection(worker: Any, action: Any, *, multi_modal: bool) -> dict[str, str]:
    obs, rewards, dones, infos = worker.step(action)
    reward, reward_error, reward_source = _graphgpo_reward(
        infos,
        multi_modal=multi_modal,
    )
    return {
        "observation": _canonical(obs),
        "raw_scores": _canonical(rewards),
        "reward": _canonical(reward),
        "reward_error": _canonical(reward_error) if reward_error else "",
        "reward_projection_source": reward_source,
        "done": _canonical(dones),
        "won": _canonical(_extract_won(infos)),
        "wrapper_state": _canonical(_wrapper_state_projection(worker)),
    }


def _resolve_multi_modal(worker: Any, multi_modal: bool | None) -> tuple[bool, str]:
    if multi_modal is not None:
        return bool(multi_modal), "argument"
    if hasattr(worker, "multi_modal"):
        return bool(getattr(worker, "multi_modal")), "worker_attribute"
    return False, "default"


def _extract_won(infos: Any) -> Any:
    return _info_value(infos, "won")


def _graphgpo_reward(
    infos: Any,
    *,
    multi_modal: bool,
) -> tuple[float | None, str | None, str]:
    info = _flatten_info(infos)
    if _UPSTREAM_COMPUTE_REWARD is not None:
        try:
            return (
                float(_UPSTREAM_COMPUTE_REWARD(info, multi_modal)),
                None,
                REWARD_PROJECTION_SOURCE,
            )
        except Exception as error:
            return None, repr(error), REWARD_PROJECTION_SOURCE
    try:
        won = info.get("won")
        if won is None:
            return None, "missing_won", FALLBACK_REWARD_PROJECTION_SOURCE
        reward = 10.0 * float(won)
        if multi_modal:
            goal_condition_success_rate = info.get("goal_condition_success_rate")
            if goal_condition_success_rate is None:
                return (
                    None,
                    "missing_goal_condition_success_rate",
                    FALLBACK_REWARD_PROJECTION_SOURCE,
                )
            reward += float(goal_condition_success_rate)
        return reward, None, FALLBACK_REWARD_PROJECTION_SOURCE
    except Exception as error:
        return None, repr(error), FALLBACK_REWARD_PROJECTION_SOURCE


def _info_value(infos: Any, key: str) -> Any:
    return _flatten_info(infos).get(key)


def _flatten_info(infos: Any) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(infos, dict):
        for key, value in infos.items():
            if isinstance(value, (list, tuple)) and value:
                flattened[key] = value[0]
            else:
                flattened[key] = value
    return flattened


def _producer_sha256() -> str | None:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return None


def _wrapper_state_projection(worker: Any) -> dict[str, Any]:
    return {
        field: copy.deepcopy(getattr(worker, field))
        for field in WRAPPER_FIELDS
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=repr)


def _blocked_result(
    reason: str,
    *,
    study_id: str | None = None,
    pre_action_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    result = {
        "schema": STATE_RESTORE_PROBE_RESULT_SCHEMA,
        "producer": PROBE_MODULE_NAME,
        "producer_sha256": _producer_sha256(),
        "reward_projection": REWARD_PROJECTION,
        "reward_projection_source": None,
        "multi_modal": None,
        "multi_modal_source": None,
        "restore_verified": False,
        "blocked_by": [reason],
        "pre_action_snapshot_sha256": pre_action_snapshot_sha256,
        "comparisons": {field: False for field in COMPARISON_FIELDS},
    }
    if study_id is not None:
        result["study_id"] = study_id
    return result
'''
