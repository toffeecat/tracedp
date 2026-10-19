"""Source-backed GraphGPO parity audit for prepared upstream runs."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRAPHGPO_PARITY_AUDIT_SCHEMA = "lh_graphgpo_parity_audit/v1"
FORMAL_CLAIM_SCOPE = "paper_reproduction_formal"
RESOURCE_CLAIM_SCOPE = "resource_constrained_ablation"

__all__ = [
    "GRAPHGPO_PARITY_AUDIT_SCHEMA",
    "FORMAL_CLAIM_SCOPE",
    "RESOURCE_CLAIM_SCOPE",
    "SourceAnchor",
    "FormalTarget",
    "SOURCE_ANCHORS",
    "FORMAL_TARGETS",
    "RESOURCE_OVERRIDE_REASON_TEMPLATE",
    "build_graphgpo_parity_audit",
]


@dataclass(frozen=True)
class SourceAnchor:
    id: str
    path: str
    quotes: tuple[str, ...]
    supports: tuple[str, ...]


@dataclass(frozen=True)
class FormalTarget:
    field: str
    official: Any
    claim_impact: str
    source_anchor: str


SOURCE_ANCHORS = (
    SourceAnchor(
        id="data_preprocess_parquet",
        path="examples/data_preprocess/prepare.py",
        quotes=(
            "train_dataset = dataset['train'].select(range(args.train_data_size))",
            "test_dataset = dataset['test'].select(range(args.val_data_size))",
            "train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))",
            "test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))",
        ),
        supports=(
            "data.train_files",
            "data.val_files",
            "data.train_batch_size",
            "data.val_batch_size",
        ),
    ),
    SourceAnchor(
        id="official_alfworld_recipe",
        path="recipe/GraphGPO/run_qwen2.5_1.5b_alfworld_train.sh",
        quotes=(
            "train_data_size=16",
            "val_data_size=128",
            "group_size=8",
            "actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct",
            "data.train_batch_size=$train_data_size",
            "data.val_batch_size=$val_data_size",
            "data.max_response_length=512",
            "actor_rollout_ref.actor.ppo_mini_batch_size=256",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
            "actor_rollout_ref.rollout.enforce_eager=False",
            "actor_rollout_ref.rollout.free_cache_engine=False",
            "env.max_steps=50",
            "env.rollout.n=$group_size",
            "trainer.n_gpus_per_node=2",
            "trainer.test_freq=5",
            "trainer.total_epochs=150",
            "trainer.val_before_train=True",
        ),
        supports=(
            "actor_rollout_ref.model.path",
            "trainer.n_gpus_per_node",
            "env.rollout.n",
            "data.max_response_length",
            "data.train_batch_size",
            "data.val_batch_size",
            "actor_rollout_ref.actor.ppo_mini_batch_size",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.free_cache_engine",
            "env.max_steps",
            "trainer.test_freq",
            "trainer.total_epochs",
            "trainer.val_before_train",
        ),
    ),
    SourceAnchor(
        id="dataloader_worker_topology",
        path="recipe/GraphGPO/graphgpo_ray_trainer.py",
        quotes=('num_workers=self.config.data.get("dataloader_num_workers", 8),',),
        supports=("+data.dataloader_num_workers",),
    ),
    SourceAnchor(
        id="rollout_identity_and_success",
        path="recipe/GraphGPO/rollout_loop.py",
        quotes=(
            "gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)",
            "batch.non_tensor_batch['uid'] = uid_batch",
            "batch.non_tensor_batch['traj_uid'] = traj_uid",
            "data['anchor_obs']=to_hashable(data['anchor_obs'])",
            "data['next_obs']=to_hashable(data['next_obs'])",
            "data['next_obs']='success'",
        ),
        supports=(
            "env.rollout.n",
            "uid",
            "traj_uid",
            "anchor_obs",
            "next_obs",
            "success_terminal_state",
        ),
    ),
    SourceAnchor(
        id="reward_response_last_token",
        path="agent_system/reward_manager/episode.py",
        quotes=(
            "episode_rewards = data_item.non_tensor_batch['episode_rewards']",
            "reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)",
        ),
        supports=("episode_rewards", "token_level_scores", "reward_last_response_token"),
    ),
    SourceAnchor(
        id="alfworld_success_reward",
        path="recipe/GraphGPO/alfworld_envs.py",
        quotes=(
            "def compute_reward(info, multi_modal=False):",
            "reward = 10.0 * float(info['won'])",
            "rewards_list.append(compute_reward(info, self.multi_modal))",
        ),
        supports=("alfworld_task_reward", "episode_rewards", "won"),
    ),
    SourceAnchor(
        id="invalid_action_penalty",
        path="recipe/GraphGPO/graphgpo_ray_trainer.py",
        quotes=(
            "action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)",
            "reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids",
            "metrics = {'episode/valid_action_ratio': valid_action_ratio}",
        ),
        supports=(
            "is_action_valid",
            "actor_rollout_ref.actor.invalid_action_penalty_coef",
            "token_level_scores",
            "episode/valid_action_ratio",
        ),
    ),
    SourceAnchor(
        id="graphgpo_advantage_path",
        path="recipe/GraphGPO/graphgpo_ray_trainer.py",
        quotes=(
            "elif adv_estimator == AdvantageEstimator.GraphGPO:",
            "advantages, returns = core_graph.compute_graphgpo_outcome_advantage(",
            "step_rewards_tensor = core_graph.compute_graph_path_returns(",
            "batch.batch['step_rewards'] = step_rewards_tensor",
        ),
        supports=("algorithm.adv_estimator", "step_rewards", "advantages", "returns"),
    ),
    SourceAnchor(
        id="training_budget_binding",
        path="recipe/GraphGPO/graphgpo_ray_trainer.py",
        quotes=(
            "if self.config.trainer.total_training_steps is not None:",
            "total_training_steps = self.config.trainer.total_training_steps",
        ),
        supports=("trainer.total_training_steps", "trainer.total_epochs"),
    ),
    SourceAnchor(
        id="rollout_artifact_dump",
        path="recipe/GraphGPO/graphgpo_ray_trainer.py",
        quotes=(
            'rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)',
            'scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()',
            "self._dump_generations(",
            '"input": inputs,',
            '"output": outputs,',
            '"score": scores,',
            '"step": [self.global_steps] * n,',
        ),
        supports=(
            "trainer.rollout_data_dir",
            "token_level_scores",
            "rollout_artifact.input",
            "rollout_artifact.output",
            "rollout_artifact.score",
            "rollout_artifact.step",
        ),
    ),
    SourceAnchor(
        id="ppo_policy_loss",
        path="verl/trainer/ppo/core_algos.py",
        quotes=(
            "pg_losses1 = -advantages * ratio",
            "pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)",
            "pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)",
        ),
        supports=("actor/pg_loss", "advantages", "response_mask", "loss_agg_mode"),
    ),
)


FORMAL_TARGETS = (
    FormalTarget(
        "actor_rollout_ref.model.path",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "paper_claim_blocking",
        "official_alfworld_recipe",
    ),
    FormalTarget("trainer.n_gpus_per_node", 2, "paper_claim_blocking", "official_alfworld_recipe"),
    FormalTarget("env.rollout.n", 8, "paper_claim_blocking", "official_alfworld_recipe"),
    FormalTarget(
        "data.max_response_length", 512, "paper_claim_blocking", "official_alfworld_recipe"
    ),
    FormalTarget("data.train_batch_size", 16, "paper_claim_blocking", "official_alfworld_recipe"),
    FormalTarget("data.val_batch_size", 128, "paper_claim_blocking", "official_alfworld_recipe"),
    FormalTarget("env.max_steps", 50, "paper_claim_blocking", "official_alfworld_recipe"),
    FormalTarget(
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        256,
        "paper_claim_blocking",
        "official_alfworld_recipe",
    ),
    FormalTarget(
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        32,
        "paper_claim_blocking",
        "official_alfworld_recipe",
    ),
    FormalTarget(
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        2,
        "paper_claim_blocking",
        "official_alfworld_recipe",
    ),
    FormalTarget(
        "actor_rollout_ref.rollout.gpu_memory_utilization",
        0.5,
        "safe_runtime_override",
        "official_alfworld_recipe",
    ),
    FormalTarget(
        "actor_rollout_ref.rollout.free_cache_engine",
        False,
        "safe_runtime_override",
        "official_alfworld_recipe",
    ),
    FormalTarget("trainer.test_freq", 5, "paper_claim_blocking", "official_alfworld_recipe"),
    FormalTarget("trainer.total_epochs", 150, "paper_claim_blocking", "official_alfworld_recipe"),
    FormalTarget(
        "trainer.total_training_steps",
        "not_resource_capped",
        "paper_claim_blocking",
        "training_budget_binding",
    ),
    FormalTarget(
        "trainer.val_before_train", True, "paper_claim_blocking", "official_alfworld_recipe"
    ),
    FormalTarget(
        "actor_rollout_ref.rollout.enforce_eager",
        False,
        "safe_runtime_override",
        "official_alfworld_recipe",
    ),
)
RESOURCE_OVERRIDE_REASON_TEMPLATE = "{resource_profile} resource profile cannot carry the formal 1.5B/2-GPU/8-rollout/150-epoch recipe"


def build_graphgpo_parity_audit(
    prepared_dir: str | Path,
    *,
    source_root: str | Path,
    trial_id: str,
    output_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed source parity audit without mutating upstream code."""
    root = Path(prepared_dir).resolve()
    source = Path(source_root).resolve()
    trial = _find_planned_trial(root / "planned_commands.jsonl", trial_id)
    study_audit = _read_json(root / "study_run_audit.json")
    run_trial = _find_run_trial(study_audit, trial_id)
    source_contract = _verify_source_anchors(source)
    field_diffs = _field_diffs(trial)
    data_contract = _data_contract(trial)
    rollout_contract = _rollout_contract(trial, run_trial)
    reward_signal = _reward_signal_diagnosis(trial, run_trial, rollout_contract)
    metrics_contract = _metrics_contract(run_trial)
    baseline_comparison = _baseline_comparison(study_audit)

    blockers: list[str] = []
    if trial is None:
        blockers.append("trial_not_found")
    blockers.extend(source_contract["blocked_by"])
    blockers.extend(data_contract["blocked_by"])
    blockers.extend(rollout_contract["blocked_by"])
    blockers.extend(metrics_contract["blocked_by"])
    claim_scope = str((trial or {}).get("claim_scope", ""))
    if trial is not None and claim_scope not in {FORMAL_CLAIM_SCOPE, RESOURCE_CLAIM_SCOPE}:
        blockers.append("claim_scope_missing_or_unknown")
    if any(diff["claim_impact"] == "paper_claim_blocking" for diff in field_diffs):
        blockers.append("paper_claim_blocking_resource_diff")
    if claim_scope == FORMAL_CLAIM_SCOPE and field_diffs:
        blockers.append("formal_profile_uses_resource_constrained_overrides")
    if _has_stdout_warnings(run_trial, study_audit):
        blockers.append("stdout_postrun_warnings")
    blockers = list(dict.fromkeys(blockers))

    formal_claim_supported = not blockers and claim_scope == FORMAL_CLAIM_SCOPE
    if not source_contract["verified"] or metrics_contract["blocked_by"] or trial is None:
        decision = "requires_diagnosis"
    elif "claim_scope_missing_or_unknown" in blockers:
        decision = "requires_diagnosis"
    elif claim_scope == FORMAL_CLAIM_SCOPE and field_diffs:
        decision = "requires_diagnosis"
    elif claim_scope == RESOURCE_CLAIM_SCOPE or field_diffs:
        decision = "resource_constrained_only"
    else:
        decision = "formal_parity_ready"

    if output_path is None:
        output_path = root / "graphgpo_parity_audit.json"
    if markdown_path is None:
        markdown_path = root / "graphgpo_parity_audit.md"
    json_target = Path(output_path)
    md_target = Path(markdown_path)
    payload: dict[str, Any] = {
        "schema": GRAPHGPO_PARITY_AUDIT_SCHEMA,
        "generated_at": time.time(),
        "prepared_dir": str(root),
        "source_root": str(source),
        "trial_id": trial_id,
        "decision": decision,
        "formal_claim_supported": formal_claim_supported,
        "blocked_by": blockers,
        "artifact_paths": {
            "json": str(json_target),
            "markdown": str(md_target),
        },
        "source_contract": source_contract,
        "trial": _trial_summary(trial, run_trial),
        "field_diffs": field_diffs,
        "data_contract": data_contract,
        "rollout_contract": rollout_contract,
        "reward_signal": reward_signal,
        "metrics_contract": metrics_contract,
        "baseline_comparison": baseline_comparison,
    }
    json_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_target.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _verify_source_anchors(source_root: Path) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    blocked_by: list[str] = []
    for anchor in SOURCE_ANCHORS:
        path = source_root / anchor.path
        missing_quotes: list[str] = []
        exists = path.is_file()
        if exists:
            text = path.read_text(encoding="utf-8", errors="replace")
            missing_quotes = [quote for quote in anchor.quotes if quote not in text]
        else:
            missing_quotes = list(anchor.quotes)
        verified = exists and not missing_quotes
        if not verified:
            blocked_by.append(f"source_anchor_unverified:{anchor.id}")
        anchors.append(
            {
                "id": anchor.id,
                "path": str(path),
                "relative_path": anchor.path,
                "verified": verified,
                "quotes": list(anchor.quotes),
                "missing_quotes": missing_quotes,
                "supports": list(anchor.supports),
            }
        )
    return {
        "verified": not blocked_by,
        "blocked_by": blocked_by,
        "anchors": anchors,
    }


def _field_diffs(trial: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if trial is None:
        return []
    overrides = _trial_overrides(trial)
    environment = trial.get("environment", {})
    if isinstance(environment, Mapping) and "LH_ATTENTION_IMPLEMENTATION" in environment:
        overrides.setdefault(
            "+ray_init.runtime_env.env_vars.LH_ATTENTION_IMPLEMENTATION",
            environment.get("LH_ATTENTION_IMPLEMENTATION"),
        )
    claim_scope = str(trial.get("claim_scope", ""))
    diffs: list[dict[str, Any]] = []
    resource_profile = str(trial.get("resource_profile", ""))
    if resource_profile:
        diffs.append(
            {
                "field": "resource_profile",
                "official": FORMAL_CLAIM_SCOPE,
                "prepared": resource_profile,
                "claim_impact": "resource_only",
                "claim_scope": claim_scope,
                "source_anchor": "official_alfworld_recipe",
                "reason": "the prepared trial explicitly chooses a resource-constrained profile instead of the formal paper recipe",
            }
        )
    for target in FORMAL_TARGETS:
        prepared = overrides.get(target.field)
        if prepared is None:
            diffs.append(
                {
                    "field": target.field,
                    "official": target.official,
                    "prepared": "(missing)",
                    "claim_impact": target.claim_impact,
                    "claim_scope": claim_scope,
                    "source_anchor": target.source_anchor,
                    "reason": "prepared command does not explicitly bind this official GraphGPO recipe field",
                }
            )
            continue
        normalized = _coerce_value(prepared)
        if _matches_official(target.field, normalized, target.official):
            continue
        diffs.append(
            {
                "field": target.field,
                "official": target.official,
                "prepared": normalized,
                "claim_impact": target.claim_impact,
                "claim_scope": claim_scope,
                "source_anchor": target.source_anchor,
                "reason": _resource_override_reason(resource_profile)
                if claim_scope == RESOURCE_CLAIM_SCOPE
                else "prepared field differs from the official GraphGPO ALFWorld recipe",
            }
        )
    if (
        str(overrides.get("+ray_init.runtime_env.env_vars.LH_ATTENTION_IMPLEMENTATION", ""))
        == "eager"
    ):
        if not any(item["field"] == "actor_rollout_ref.rollout.enforce_eager" for item in diffs):
            diffs.append(
                {
                    "field": "actor_rollout_ref.rollout.enforce_eager",
                    "official": False,
                    "prepared": True,
                    "claim_impact": "safe_runtime_override",
                    "claim_scope": claim_scope,
                    "source_anchor": "official_alfworld_recipe",
                    "reason": "runtime attention fallback forced by the single-5090 compatibility profile",
                }
            )
    return diffs


def _resource_override_reason(resource_profile: str) -> str:
    profile = resource_profile or "resource-constrained"
    return RESOURCE_OVERRIDE_REASON_TEMPLATE.format(resource_profile=profile)


def _matches_official(field: str, prepared: Any, official: Any) -> bool:
    if field == "trainer.total_training_steps":
        return prepared in (None, "not_resource_capped")
    return prepared == official


def _metrics_contract(run_trial: Mapping[str, Any] | None) -> dict[str, Any]:
    metrics = run_trial.get("metric_summary", {}) if isinstance(run_trial, Mapping) else {}
    progress = run_trial.get("progress_summary", {}) if isinstance(run_trial, Mapping) else {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    if not isinstance(progress, Mapping):
        progress = {}
    groups = {
        "pg_loss": ("actor/pg_loss",),
        "kl_loss": ("actor/kl_loss",),
        "reward": ("critic/rewards/mean", "episode/reward/mean"),
        "success_rate": ("val/success_rate", "episode/success_rate"),
    }

    present: dict[str, float] = {}
    missing: list[str] = []
    for group, keys in groups.items():
        found = None
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found = (key, float(value))
                break
        if found is None:
            missing.append(group)
        else:
            present[found[0]] = found[1]
    validation_progress_present = any(
        isinstance(progress.get(key), (int, float)) and not isinstance(progress.get(key), bool)
        for key in (
            "validation_meta_info_events",
            "validation_generation_end_events",
            "validation_expected_events",
            "validation_completed_passes",
        )
    )
    if not validation_progress_present:
        missing.append("validation_progress")
    blocked_by = [f"metric_group_missing:{group}" for group in missing]
    return {
        "verified": not blocked_by,
        "blocked_by": blocked_by,
        "present": present,
        "progress_summary": dict(progress),
    }


def _data_contract(trial: Mapping[str, Any] | None) -> dict[str, Any]:
    """Verify the actual parquet files selected by the official recipe.

    The upstream preparer selects 16 train rows and 128 validation rows before
    writing ``train.parquet`` and ``test.parquet``. A prepared command can
    quote those paths while pointing at a different slice, so config parity
    alone is insufficient evidence.
    """
    overrides = _trial_overrides(trial or {})
    checks = (("train", "data.train_files", 16), ("val", "data.val_files", 128))
    blocked_by: list[str] = []
    result: dict[str, Any] = {}
    for label, field, expected_rows in checks:
        raw_path = overrides.get(field)
        entry: dict[str, Any] = {
            "field": field,
            "expected_rows": expected_rows,
            "path": str(raw_path) if raw_path is not None else None,
            "row_count": None,
        }
        if not raw_path:
            blocked_by.append(f"data_{label}_path_missing")
        elif not Path(str(raw_path)).is_file():
            blocked_by.append(f"data_{label}_file_missing")
        else:
            try:
                import pyarrow.parquet as pq

                entry["row_count"] = int(pq.ParquetFile(str(raw_path)).metadata.num_rows)
                if entry["row_count"] != expected_rows:
                    blocked_by.append(f"data_{label}_row_count_mismatch")
            except Exception as exc:
                entry["read_error"] = repr(exc)
                blocked_by.append(f"data_{label}_parquet_unreadable")
        result[label] = entry
    return {"verified": not blocked_by, "blocked_by": blocked_by, **result}


def _rollout_contract(
    trial: Mapping[str, Any] | None,
    run_trial: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Verify the upstream rollout dump that feeds the policy update.

    ``_dump_generations`` is the only official, persistent per-rollout artifact
    available without patching the upstream trainer.  It deliberately cannot
    prove GraphGPO's internal graph identity fields; those require the separate
    state-restore-gated trace sidecar.  It can, however, prove that actual
    token-level scores, decoded responses, and the completed global step were
    carried through to a durable artifact.
    """
    overrides = _trial_overrides(trial or {})
    raw_dir = overrides.get("trainer.rollout_data_dir")
    maximum_records_per_update = _rollout_record_bound(overrides)
    blocked_by: list[str] = []
    result: dict[str, Any] = {
        "path": str(raw_dir) if raw_dir is not None else None,
        "files": [],
        "record_count": 0,
        "observed_global_steps": [],
        "records_by_global_step": {},
        "maximum_records_per_update": maximum_records_per_update,
        "full_horizon_exhausted": None,
        "score_distribution": {
            "negative": 0,
            "zero": 0,
            "positive": 0,
        },
        "score_values": [],
        "all_scores_nonpositive": None,
        "dynamic_graph_identity": {
            "observed": False,
            "required_fields": ["uid", "traj_uid", "anchor_obs", "next_obs"],
            "reason": (
                "the official _dump_generations artifact contains input/output/score/step; "
                "GraphGPO identity fields require the state-restore-gated trace sidecar"
            ),
        },
    }
    if not raw_dir:
        blocked_by.append("rollout_artifact_path_missing")
        return {"verified": False, "blocked_by": blocked_by, **result}

    rollout_dir = Path(str(raw_dir))
    if not rollout_dir.is_dir():
        blocked_by.append("rollout_artifact_directory_missing")
        return {"verified": False, "blocked_by": blocked_by, **result}

    files = sorted(rollout_dir.glob("*.jsonl"), key=lambda path: path.name)
    result["files"] = [str(path) for path in files]
    if not files:
        blocked_by.append("rollout_artifact_missing")
        return {"verified": False, "blocked_by": blocked_by, **result}

    required_fields = ("input", "output", "score", "step")
    observed_steps: set[int] = set()
    records_by_step: dict[int, int] = {}
    score_counts = result["score_distribution"]
    assert isinstance(score_counts, dict)
    score_values: set[float] = set()
    identity_seen = False
    for path in files:
        try:
            rows = _read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result.setdefault("read_errors", []).append({"path": str(path), "error": repr(exc)})
            blocked_by.append("rollout_artifact_unreadable")
            continue
        if not rows:
            blocked_by.append("rollout_artifact_empty")
            continue
        for row in rows:
            result["record_count"] += 1
            missing = [field for field in required_fields if field not in row]
            if missing:
                blocked_by.extend(f"rollout_artifact_field_missing:{field}" for field in missing)
                continue
            step = row["step"]
            if isinstance(step, int) and not isinstance(step, bool):
                observed_steps.add(step)
                records_by_step[step] = records_by_step.get(step, 0) + 1
            else:
                blocked_by.append("rollout_artifact_step_invalid")
            score = row["score"]
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                blocked_by.append("rollout_artifact_score_invalid")
            elif score > 0:
                score_counts["positive"] += 1
                score_values.add(float(score))
            elif score < 0:
                score_counts["negative"] += 1
                score_values.add(float(score))
            else:
                score_counts["zero"] += 1
                score_values.add(float(score))
            identity_seen = identity_seen or all(
                field in row for field in result["dynamic_graph_identity"]["required_fields"]
            )

    result["observed_global_steps"] = sorted(observed_steps)
    result["records_by_global_step"] = {
        str(step): records_by_step[step] for step in sorted(records_by_step)
    }
    if maximum_records_per_update is not None and records_by_step:
        if any(count > maximum_records_per_update for count in records_by_step.values()):
            blocked_by.append("rollout_artifact_records_exceed_update_bound")
        result["full_horizon_exhausted"] = all(
            count == maximum_records_per_update for count in records_by_step.values()
        )
    result["score_values"] = sorted(score_values)
    result["all_scores_nonpositive"] = (
        bool(result["record_count"]) and score_counts["positive"] == 0
    )
    result["dynamic_graph_identity"]["observed"] = identity_seen

    metrics = run_trial.get("metric_summary", {}) if isinstance(run_trial, Mapping) else {}
    completed_step = metrics.get("training/global_step") if isinstance(metrics, Mapping) else None
    if isinstance(completed_step, (int, float)) and not isinstance(completed_step, bool):
        expected_step = int(completed_step)
        result["expected_completed_global_step"] = expected_step
        if expected_step > 0 and expected_step not in observed_steps:
            blocked_by.append("rollout_artifact_completed_step_missing")

    return {
        "verified": not blocked_by,
        "blocked_by": list(dict.fromkeys(blocked_by)),
        **result,
    }


def _rollout_record_bound(overrides: Mapping[str, Any]) -> int | None:
    fields = ("data.train_batch_size", "env.rollout.n", "env.max_steps")
    values: list[int] = []
    for field in fields:
        value = overrides.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return None
        values.append(value)
    return values[0] * values[1] * values[2]


def _reward_signal_diagnosis(
    trial: Mapping[str, Any] | None,
    run_trial: Mapping[str, Any] | None,
    rollout_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify task reward versus the optional invalid-action shaping signal.

    For text-only ALFWorld, the official environment emits 10 only when
    ``won`` is true.  The trainer then optionally subtracts the configured
    invalid-action penalty directly from token-level scores.  Keeping these
    pathways separate prevents an update driven only by invalid-action shaping
    from being misreported as task-success learning.
    """
    overrides = _trial_overrides(trial or {})
    metrics = run_trial.get("metric_summary", {}) if isinstance(run_trial, Mapping) else {}
    metrics = metrics if isinstance(metrics, Mapping) else {}
    episode_reward = metrics.get("episode/reward/mean")
    critic_reward = metrics.get("critic/rewards/mean")
    valid_action_ratio = metrics.get("episode/valid_action_ratio")
    penalty = overrides.get("actor_rollout_ref.actor.invalid_action_penalty_coef")
    use_penalty = overrides.get("actor_rollout_ref.actor.use_invalid_action_penalty", True)
    use_kl_in_reward = overrides.get("algorithm.use_kl_in_reward", False)
    score_distribution = rollout_contract.get("score_distribution", {})
    score_values = rollout_contract.get("score_values", [])
    negative_count = (
        score_distribution.get("negative", 0) if isinstance(score_distribution, Mapping) else 0
    )
    positive_count = (
        score_distribution.get("positive", 0) if isinstance(score_distribution, Mapping) else 0
    )
    numeric_scores = [
        float(value)
        for value in score_values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    numeric_penalty = (
        float(penalty)
        if isinstance(penalty, (int, float)) and not isinstance(penalty, bool)
        else None
    )
    negative_scores_match_penalty = (
        bool(negative_count)
        and bool(use_penalty)
        and numeric_penalty is not None
        and all(
            math.isclose(abs(score), numeric_penalty, rel_tol=1e-5, abs_tol=1e-5)
            for score in numeric_scores
            if score < 0
        )
    )
    episode_reward_zero = (
        isinstance(episode_reward, (int, float))
        and not isinstance(episode_reward, bool)
        and math.isclose(float(episode_reward), 0.0, abs_tol=1e-8)
    )
    task_success_signal_absent = episode_reward_zero and positive_count == 0
    if task_success_signal_absent and negative_scores_match_penalty and not use_kl_in_reward:
        diagnosis = "invalid_action_penalty_only"
    elif task_success_signal_absent:
        diagnosis = "task_success_signal_absent"
    elif positive_count:
        diagnosis = "positive_task_signal_observed"
    else:
        diagnosis = "insufficient_reward_evidence"
    return {
        "diagnosis": diagnosis,
        "episode_reward_mean": episode_reward,
        "critic_reward_mean": critic_reward,
        "valid_action_ratio": valid_action_ratio,
        "rollout_positive_score_count": positive_count,
        "rollout_negative_score_count": negative_count,
        "rollout_score_values": numeric_scores,
        "invalid_action_penalty_enabled": bool(use_penalty),
        "invalid_action_penalty_coef": numeric_penalty,
        "negative_scores_match_invalid_action_penalty": negative_scores_match_penalty,
        "use_kl_in_reward": bool(use_kl_in_reward),
        "task_success_signal_absent": task_success_signal_absent,
        "claim": (
            "All observed training scores are explained by the configured invalid-action penalty; "
            "no positive ALFWorld success reward reached this update."
            if diagnosis == "invalid_action_penalty_only"
            else "No positive task reward conclusion is emitted without matching upstream reward evidence."
        ),
    }


def _baseline_comparison(study_audit: Mapping[str, Any]) -> dict[str, Any]:
    comparison = study_audit.get("comparison_summary", {})
    if not isinstance(comparison, Mapping):
        return {"purpose": "unavailable", "paired_seeds": [], "pairwise_deltas": []}
    return {
        "purpose": comparison.get("purpose"),
        "paired_seeds": list(comparison.get("paired_seeds", []))
        if isinstance(comparison.get("paired_seeds"), list)
        else [],
        "pairwise_deltas": list(comparison.get("pairwise_deltas", []))
        if isinstance(comparison.get("pairwise_deltas"), list)
        else [],
    }


def _trial_summary(
    planned: Mapping[str, Any] | None,
    run_trial: Mapping[str, Any] | None,
) -> dict[str, Any]:
    planned = planned or {}
    run_trial = run_trial or {}
    return {
        "trial_id": planned.get("trial_id") or run_trial.get("trial_id"),
        "variant_id": planned.get("variant_id"),
        "seed": planned.get("seed"),
        "claim_scope": planned.get("claim_scope"),
        "resource_profile": planned.get("resource_profile"),
        "status": run_trial.get("status"),
        "returncode": run_trial.get("returncode"),
        "blockers": list(run_trial.get("blockers", []))
        if isinstance(run_trial.get("blockers"), list)
        else [],
    }


def _find_planned_trial(path: Path, trial_id: str) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    for row in _read_jsonl(path):
        if str(row.get("trial_id", "")) == trial_id:
            return row
    return None


def _find_run_trial(study_audit: Mapping[str, Any], trial_id: str) -> Mapping[str, Any] | None:
    trials = study_audit.get("trials", [])
    if not isinstance(trials, list):
        return None
    for trial in trials:
        if isinstance(trial, Mapping) and str(trial.get("trial_id", "")) == trial_id:
            return trial
    return None


def _has_stdout_warnings(
    run_trial: Mapping[str, Any] | None,
    study_audit: Mapping[str, Any],
) -> bool:
    blockers: list[Any] = []
    if isinstance(run_trial, Mapping) and isinstance(run_trial.get("blockers"), list):
        blockers.extend(run_trial.get("blockers", []))
    if isinstance(study_audit.get("blocked_by"), list):
        blockers.extend(study_audit.get("blocked_by", []))
    return "stdout_postrun_warnings" in {str(item) for item in blockers}


def _trial_overrides(trial: Mapping[str, Any]) -> dict[str, Any]:
    raw = trial.get("overrides", {})
    overrides = dict(raw) if isinstance(raw, Mapping) else {}
    argv = trial.get("argv", [])
    if isinstance(argv, list):
        for item in argv:
            text = str(item)
            if "=" not in text:
                continue
            key, value = text.split("=", 1)
            overrides.setdefault(key, _coerce_value(value))
    return overrides


def _coerce_value(value: Any) -> Any:
    if isinstance(value, str):
        if value == "True":
            return True
        if value == "False":
            return False
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if isinstance(raw, Mapping):
            rows.append(dict(raw))
    return rows


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# GraphGPO parity audit",
        "",
        f"- decision: `{payload['decision']}`",
        f"- formal claim supported: `{str(payload['formal_claim_supported']).lower()}`",
        f"- blocked by: `{', '.join(payload['blocked_by']) or '(none)'}`",
        f"- trial: `{payload['trial_id']}`",
        "",
        "## Source quote contract",
        "",
    ]
    source_contract = payload.get("source_contract", {})
    if isinstance(source_contract, Mapping):
        for anchor in source_contract.get("anchors", []):
            if not isinstance(anchor, Mapping):
                continue
            lines.extend(
                [
                    f"### {anchor.get('id')}",
                    "",
                    f"- path: `{anchor.get('relative_path')}`",
                    f"- verified: `{str(anchor.get('verified')).lower()}`",
                    f"- supports: `{', '.join(anchor.get('supports', []))}`",
                    "",
                ]
            )
            lines.extend(["#### Source quotes", ""])
            for quote in anchor.get("quotes", []):
                if isinstance(quote, str):
                    lines.append(f"- `{quote}`")
            missing = [
                str(item) for item in anchor.get("missing_quotes", []) if isinstance(item, str)
            ]
            if missing:
                lines.extend(["", "#### Missing quotes", ""])
                for quote in missing:
                    lines.append(f"- `{quote}`")
            lines.append("")
    lines.extend(
        [
            "## Prepared-field diff",
            "",
            "| field | official | prepared | impact |",
            "| --- | --- | --- | --- |",
        ]
    )
    for diff in payload.get("field_diffs", []):
        if not isinstance(diff, Mapping):
            continue
        lines.append(
            f"| `{diff.get('field')}` | `{diff.get('official')}` | `{diff.get('prepared')}` | `{diff.get('claim_impact')}` |"
        )
    if not payload.get("field_diffs"):
        lines.append("| `(none)` |  |  |  |")
    data = payload.get("data_contract", {})
    lines.extend(["## Prepared data contract", ""])
    if isinstance(data, Mapping):
        lines.extend(
            [
                f"- verified: `{str(data.get('verified')).lower()}`",
                f"- blocked by: `{', '.join(data.get('blocked_by', [])) or '(none)'}`",
                "",
                "| split | expected rows | actual rows | path |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for split in ("train", "val"):
            item = data.get(split, {})
            if isinstance(item, Mapping):
                lines.append(
                    f"| `{split}` | {item.get('expected_rows')} | {item.get('row_count')} | `{item.get('path')}` |"
                )
    lines.extend(["", "## Upstream rollout artifact contract", ""])
    rollout = payload.get("rollout_contract", {})
    if isinstance(rollout, Mapping):
        score_distribution = rollout.get("score_distribution", {})
        identity = rollout.get("dynamic_graph_identity", {})
        lines.extend(
            [
                f"- verified: `{str(rollout.get('verified')).lower()}`",
                f"- blocked by: `{', '.join(rollout.get('blocked_by', [])) or '(none)'}`",
                f"- path: `{rollout.get('path')}`",
                f"- records: `{rollout.get('record_count')}`",
                f"- observed global steps: `{', '.join(str(step) for step in rollout.get('observed_global_steps', [])) or '(none)'}`",
                f"- maximum records per update: `{rollout.get('maximum_records_per_update')}`",
                f"- all observed updates exhausted full horizon: `{str(rollout.get('full_horizon_exhausted')).lower()}`",
                f"- score distribution (negative/zero/positive): `{score_distribution.get('negative') if isinstance(score_distribution, Mapping) else None}/{score_distribution.get('zero') if isinstance(score_distribution, Mapping) else None}/{score_distribution.get('positive') if isinstance(score_distribution, Mapping) else None}`",
                f"- all scores nonpositive: `{str(rollout.get('all_scores_nonpositive')).lower()}`",
                f"- dynamic GraphGPO identity observed: `{str(identity.get('observed')).lower() if isinstance(identity, Mapping) else 'false'}`",
            ]
        )
    lines.extend(["", "## Reward-signal diagnosis", ""])
    reward_signal = payload.get("reward_signal", {})
    if isinstance(reward_signal, Mapping):
        lines.extend(
            [
                f"- diagnosis: `{reward_signal.get('diagnosis')}`",
                f"- episode reward mean: `{reward_signal.get('episode_reward_mean')}`",
                f"- critic reward mean: `{reward_signal.get('critic_reward_mean')}`",
                f"- valid-action ratio: `{reward_signal.get('valid_action_ratio')}`",
                f"- score values: `{', '.join(str(value) for value in reward_signal.get('rollout_score_values', [])) or '(none)'}`",
                f"- negative scores match invalid-action penalty: `{str(reward_signal.get('negative_scores_match_invalid_action_penalty')).lower()}`",
                f"- claim: {reward_signal.get('claim')}",
            ]
        )
    lines.extend(["", "## Metrics contract", ""])
    metrics = payload.get("metrics_contract", {})
    if isinstance(metrics, Mapping):
        lines.extend(
            [
                f"- verified: `{str(metrics.get('verified')).lower()}`",
                f"- blocked by: `{', '.join(metrics.get('blocked_by', [])) or '(none)'}`",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        present = metrics.get("present", {})
        if isinstance(present, Mapping) and present:
            for key, value in present.items():
                lines.append(
                    f"| `{key}` | {value:.6g} |"
                    if isinstance(value, float)
                    else f"| `{key}` | {value} |"
                )
        else:
            lines.append("| `(none)` |  |")
    comparison = payload.get("baseline_comparison", {})
    lines.extend(["", "## Baseline comparison passthrough", ""])
    if isinstance(comparison, Mapping):
        lines.extend(
            [
                f"- purpose: `{comparison.get('purpose')}`",
                f"- paired seeds: `{', '.join(str(seed) for seed in comparison.get('paired_seeds', [])) or '(none)'}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)
