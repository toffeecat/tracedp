"""Descriptive aggregation for frozen functional residual-state pairs."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from lh_platform.training.dataset import sha256_file
from lh_platform.training.parent_child_state_use.benchmark import (
    FUNCTIONAL_STATE_USE_SCHEMA,
)

FUNCTIONAL_STATE_USE_ANALYSIS_SCHEMA = "lh_functional_state_use_analysis/v1"


class FunctionalStateUseAnalysisError(ValueError):
    """Raised when result artifacts are not comparable descriptive evidence."""


def analyze_functional_state_use_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate pair-level directions without claiming generalization."""

    if not artifacts:
        raise FunctionalStateUseAnalysisError(
            "functional state-use analysis requires at least one artifact"
        )
    rows = [_validated_row(artifact) for artifact in artifacts]
    seen = set()
    for row in rows:
        key = (row["policy_checkpoint_sha256"], row["environment_manifest_sha256"])
        if key in seen:
            raise FunctionalStateUseAnalysisError(
                "functional state-use analysis contains a duplicate model/pair"
            )
        seen.add(key)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["policy_checkpoint_sha256"]].append(row)
    model_summaries = []
    manifest_sets = []
    for policy_checkpoint_sha256, model_rows in sorted(grouped.items()):
        ordered = sorted(model_rows, key=lambda item: item["task_id"])
        base_checkpoints = {item["base_checkpoint_sha256"] for item in ordered}
        if len(base_checkpoints) != 1:
            raise FunctionalStateUseAnalysisError(
                "one policy checkpoint has conflicting base checkpoints"
            )
        adapter_identities = {
            (
                item["adapter_checkpoint_sha256"],
                item["adapter_artifact_sha256"],
            )
            for item in ordered
        }
        if len(adapter_identities) != 1:
            raise FunctionalStateUseAnalysisError(
                "one policy checkpoint has conflicting adapter provenance"
            )
        named_rows = [item for item in ordered if item["model_provenance_present"]]
        named_identities = {(item["model_id"], item["model_reference"]) for item in named_rows}
        if len(named_identities) > 1:
            raise FunctionalStateUseAnalysisError(
                "one checkpoint hash has conflicting model provenance"
            )
        identity_row = named_rows[0] if named_rows else ordered[0]
        manifest_sets.append({item["environment_manifest_sha256"] for item in ordered})
        model_summaries.append(
            {
                "policy_checkpoint_sha256": policy_checkpoint_sha256,
                "base_checkpoint_sha256": identity_row["base_checkpoint_sha256"],
                "adapter_checkpoint_sha256": identity_row["adapter_checkpoint_sha256"],
                "adapter_artifact_sha256": identity_row["adapter_artifact_sha256"],
                "model_id": identity_row["model_id"],
                "model_reference": identity_row["model_reference"],
                "pair_count": len(ordered),
                "identifiable_pair_count": sum(int(item["test_identifiable"]) for item in ordered),
                "supported_pair_count": sum(
                    int(item["supported_on_preregistered_pair"]) for item in ordered
                ),
                "positive_normal_pair_count": sum(
                    int(item["normal_pair_sensitivity"] > 0.0) for item in ordered
                ),
                "mean_normal_pair_sensitivity": fmean(
                    item["normal_pair_sensitivity"] for item in ordered
                ),
                "mean_normal_over_corrupted_history": fmean(
                    item["normal_over_corrupted_history"] for item in ordered
                ),
                "mean_correct_over_corrupted_graph": fmean(
                    item["correct_over_corrupted_graph"] for item in ordered
                ),
                "pairs": ordered,
            }
        )
    shared_manifests = set.intersection(*manifest_sets) if manifest_sets else set()
    all_manifests = set.union(*manifest_sets) if manifest_sets else set()
    return {
        "schema": FUNCTIONAL_STATE_USE_ANALYSIS_SCHEMA,
        "artifact_count": len(rows),
        "model_count": len(model_summaries),
        "unique_environment_pair_count": len(all_manifests),
        "shared_environment_pair_count_across_all_models": len(shared_manifests),
        "complete_cross_model_pairing": all(
            manifests == manifest_sets[0] for manifests in manifest_sets[1:]
        ),
        "models": model_summaries,
        "claim_scope": (
            "descriptive_preregistered_controlled_pairs_not_population_"
            "awareness_policy_improvement_or_significance"
        ),
    }


def analyze_functional_state_use_result_files(
    result_paths: Sequence[str | Path],
) -> dict[str, Any]:
    artifacts = []
    inputs = []
    for raw_path in result_paths:
        path = Path(raw_path).resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FunctionalStateUseAnalysisError(
                f"cannot load functional state-use result: {path}"
            ) from error
        if not isinstance(payload, Mapping):
            raise FunctionalStateUseAnalysisError("functional state-use result must be a mapping")
        artifacts.append(payload)
        inputs.append({"path": str(path), "file_sha256": sha256_file(path)})
    return {
        **analyze_functional_state_use_artifacts(artifacts),
        "inputs": inputs,
    }


def _validated_row(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if artifact.get("schema") != FUNCTIONAL_STATE_USE_SCHEMA:
        raise FunctionalStateUseAnalysisError("functional state-use result schema differs")
    if artifact.get("policy_update_executed") is not False or (
        artifact.get("policy_checkpoint_before_sha256")
        != artifact.get("policy_checkpoint_after_sha256")
    ):
        raise FunctionalStateUseAnalysisError(
            "functional state-use result is not a frozen-policy artifact"
        )
    model = artifact.get("model")
    if not isinstance(model, Mapping):
        model = {}
    adapter = artifact.get("adapter_binding")
    if not isinstance(adapter, Mapping):
        adapter = {}
    gates = artifact.get("evidence_gates")
    metrics = artifact.get("pair_metrics_by_state_exposure")
    contrasts = artifact.get("behavioral_contrasts")
    if not all(isinstance(value, Mapping) for value in (gates, metrics, contrasts)):
        raise FunctionalStateUseAnalysisError("functional state-use result metrics are missing")
    try:
        normal = float(metrics["normal"]["pair_sensitivity"])
        history = float(contrasts["normal_over_corrupted_history_pair_sensitivity"])
        graph = float(contrasts["correct_over_corrupted_graph_pair_sensitivity"])
    except (KeyError, TypeError, ValueError) as error:
        raise FunctionalStateUseAnalysisError(
            "functional state-use result directional metrics differ"
        ) from error
    checkpoint = str(artifact.get("base_checkpoint_sha256", ""))
    policy_checkpoint = str(artifact.get("policy_checkpoint_before_sha256", ""))
    manifest = str(artifact.get("environment_manifest_sha256", ""))
    if len(checkpoint) != 64 or len(policy_checkpoint) != 64 or len(manifest) != 64:
        raise FunctionalStateUseAnalysisError(
            "functional state-use result provenance hashes differ"
        )
    adapter_checkpoint = str(adapter.get("checkpoint_sha256", ""))
    adapter_artifact = str(adapter.get("adapter_artifact_sha256", ""))
    if adapter and (len(adapter_checkpoint) != 64 or len(adapter_artifact) != 64):
        raise FunctionalStateUseAnalysisError(
            "functional state-use adapter provenance hashes differ"
        )
    return {
        "task_id": str(artifact.get("task_id", "")),
        "pair_type": str(artifact.get("pair_type", "")),
        "recipe_sha256": str(artifact.get("recipe_sha256", "")),
        "environment_manifest_sha256": manifest,
        "base_checkpoint_sha256": checkpoint,
        "policy_checkpoint_sha256": policy_checkpoint,
        "adapter_checkpoint_sha256": adapter_checkpoint or None,
        "adapter_artifact_sha256": adapter_artifact or None,
        "model_id": str(model.get("id", checkpoint[:12])),
        "model_reference": str(model.get("reference", checkpoint[:12])),
        "model_provenance_present": bool(model.get("id") and model.get("reference")),
        "test_identifiable": bool(gates.get("functional_residual_state_use_test_identifiable")),
        "supported_on_preregistered_pair": bool(
            gates.get("functional_residual_state_use_supported_on_preregistered_pair")
        ),
        "normal_pair_sensitivity": normal,
        "normal_over_corrupted_history": history,
        "correct_over_corrupted_graph": graph,
    }


__all__ = [
    "FUNCTIONAL_STATE_USE_ANALYSIS_SCHEMA",
    "FunctionalStateUseAnalysisError",
    "analyze_functional_state_use_artifacts",
    "analyze_functional_state_use_result_files",
]
