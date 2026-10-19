"""Freeze the pinned G2PO valid-seen worker population into reviewable data."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from lh_platform.studies.runtime import (
    AlfworldTask,
    AlfworldTaskManifest,
    build_alfworld_task_manifest,
)

G2PO_EVALUATION_SCHEDULE_SCHEMA = "lh_g2po_evaluation_schedule/v1"
G2PO_VALID_SEEN_SEED_START = 1000
G2PO_VALID_SEEN_EPISODES = 128
G2PO_EVALUATION_SOURCE_FILES = (
    "agent_system/environments/env_manager.py",
    "agent_system/environments/env_package/alfworld/envs.py",
    "agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py",
)

__all__ = [
    "G2PO_EVALUATION_SCHEDULE_SCHEMA",
    "G2PO_VALID_SEEN_SEED_START",
    "G2PO_VALID_SEEN_EPISODES",
    "G2PO_EVALUATION_SOURCE_FILES",
    "G2POEvaluationScheduleAudit",
    "freeze_g2po_valid_seen_population",
    "audit_g2po_valid_seen_schedule",
    "build_g2po_valid_seen_population",
]


@dataclass(frozen=True)
class G2POEvaluationScheduleAudit:
    source_revision: str
    schedule_sha256: str
    episode_count: int
    unique_game_count: int
    source_file_sha256: Mapping[str, str]
    exact_source_reset_match: bool


def freeze_g2po_valid_seen_population(
    *,
    source_root: str | Path,
    data_root: str | Path,
    expected_revision: str,
    episode_count: int = G2PO_VALID_SEEN_EPISODES,
    seed_start: int = G2PO_VALID_SEEN_SEED_START,
) -> tuple[AlfworldTaskManifest, dict[str, Any]]:
    """Execute the pinned per-worker reset contract and freeze selected games."""

    source = Path(source_root).resolve()
    data = Path(data_root).resolve()
    actual_revision = _git_output(source, "rev-parse", "HEAD")
    if actual_revision != expected_revision:
        raise ValueError(
            f"G2PO source revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    selected = _select_source_worker_games(
        source_root=source,
        data_root=data,
        episode_count=episode_count,
        seed_start=seed_start,
    )
    source_files = {
        relative: _sha256_file(source / relative) for relative in G2PO_EVALUATION_SOURCE_FILES
    }
    return build_g2po_valid_seen_population(
        source_manifest=build_alfworld_task_manifest(data),
        selected_game_paths=selected,
        source_revision=actual_revision,
        source_files=source_files,
        seed_start=seed_start,
        dataset_root_label="ALFWORLD_DATA",
    )


def audit_g2po_valid_seen_schedule(
    *,
    source_root: str | Path,
    data_root: str | Path,
    schedule_path: str | Path,
    expected_revision: str,
) -> G2POEvaluationScheduleAudit:
    """Re-execute pinned worker resets and prove the frozen episode order."""

    source = Path(source_root).resolve()
    data = Path(data_root).resolve()
    schedule_file = Path(schedule_path).resolve()
    try:
        schedule = json.loads(schedule_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load G2PO evaluation schedule: {error}") from error
    if not isinstance(schedule, Mapping):
        raise ValueError("G2PO evaluation schedule must be a mapping")
    actual_revision = _git_output(source, "rev-parse", "HEAD")
    if actual_revision != expected_revision or schedule.get("source_revision") != expected_revision:
        raise ValueError("G2PO evaluation schedule/source revision mismatch")
    if (
        schedule.get("schema") != G2PO_EVALUATION_SCHEDULE_SCHEMA
        or schedule.get("split") != "valid_seen"
        or schedule.get("seed_start") != G2PO_VALID_SEEN_SEED_START
        or schedule.get("episode_count") != G2PO_VALID_SEEN_EPISODES
    ):
        raise ValueError("G2PO evaluation schedule header differs from official contract")
    selection_contract = schedule.get("selection_contract")
    if not isinstance(selection_contract, Mapping):
        raise ValueError("G2PO evaluation selection contract is invalid")
    recorded_hashes = selection_contract.get("source_files")
    if not isinstance(recorded_hashes, Mapping) or set(recorded_hashes) != set(
        G2PO_EVALUATION_SOURCE_FILES
    ):
        raise ValueError("G2PO evaluation source file set differs from pinned contract")
    actual_hashes = {
        relative: _sha256_file(source / relative) for relative in G2PO_EVALUATION_SOURCE_FILES
    }
    if dict(recorded_hashes) != actual_hashes:
        raise ValueError("G2PO evaluation source file hash mismatch")
    episodes = schedule.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != G2PO_VALID_SEEN_EPISODES:
        raise ValueError("G2PO evaluation schedule episodes are invalid")
    recorded_games = tuple(str(item.get("game_path", "")) for item in episodes)
    recorded_seeds = tuple(item.get("environment_seed") for item in episodes)
    expected_seeds = tuple(
        G2PO_VALID_SEEN_SEED_START + index for index in range(G2PO_VALID_SEEN_EPISODES)
    )
    if recorded_seeds != expected_seeds:
        raise ValueError("G2PO evaluation schedule seed sequence differs from source")
    selected = _select_source_worker_games(
        source_root=source,
        data_root=data,
        episode_count=G2PO_VALID_SEEN_EPISODES,
        seed_start=G2PO_VALID_SEEN_SEED_START,
    )
    selected_games: list[str] = []
    for raw_path in selected:
        path = Path(raw_path).resolve()
        try:
            selected_games.append(path.relative_to(data).as_posix())
        except ValueError as error:
            raise ValueError(
                "source-selected G2PO game is outside the configured data root"
            ) from error
    if tuple(selected_games) != recorded_games:
        raise ValueError(
            "frozen G2PO evaluation episode order/multiplicity differs from live source resets"
        )
    return G2POEvaluationScheduleAudit(
        source_revision=actual_revision,
        schedule_sha256=_sha256_file(schedule_file),
        episode_count=len(recorded_games),
        unique_game_count=len(set(recorded_games)),
        source_file_sha256=actual_hashes,
        exact_source_reset_match=True,
    )


def build_g2po_valid_seen_population(
    *,
    source_manifest: AlfworldTaskManifest,
    selected_game_paths: Sequence[str | Path],
    source_revision: str,
    source_files: Mapping[str, str],
    seed_start: int = G2PO_VALID_SEEN_SEED_START,
    dataset_root_label: str = "ALFWORLD_DATA",
) -> tuple[AlfworldTaskManifest, dict[str, Any]]:
    """Build a unique task manifest plus a repeated episode schedule."""

    if not selected_game_paths:
        raise ValueError("G2PO evaluation population is empty")
    by_absolute_game = {
        (Path(source_manifest.dataset_root) / task.game_path).resolve(): task
        for task in source_manifest.tasks
        if task.split == "valid_seen"
    }
    selected: list[AlfworldTask] = []
    for raw_path in selected_game_paths:
        task = by_absolute_game.get(Path(raw_path).resolve())
        if task is None:
            raise ValueError(f"source-selected game is outside valid_seen manifest: {raw_path}")
        selected.append(task)

    reference_steps = sorted(task.reference_steps for task in selected)
    short_max = reference_steps[math.ceil(len(reference_steps) / 3) - 1]
    medium_max = reference_steps[math.ceil(2 * len(reference_steps) / 3) - 1]
    if short_max >= medium_max:
        raise ValueError("source-selected valid_seen episodes do not form three horizon strata")

    def with_bucket(task: AlfworldTask) -> AlfworldTask:
        if task.reference_steps <= short_max:
            bucket = "short"
        elif task.reference_steps <= medium_max:
            bucket = "medium"
        else:
            bucket = "long"
        return replace(task, horizon_bucket=bucket)

    selected = [with_bucket(task) for task in selected]
    unique: dict[tuple[str, str], AlfworldTask] = {}
    for task in selected:
        unique.setdefault((task.task_id, task.game_path), task)
    manifest = AlfworldTaskManifest(
        schema=source_manifest.schema,
        benchmark=source_manifest.benchmark,
        dataset_root=dataset_root_label,
        horizon_policy={
            "stratification": "official_g2po_valid_seen_episode_terciles",
            "short_max_reference_steps": short_max,
            "medium_max_reference_steps": medium_max,
            "episode_count": len(selected),
            "unique_task_count": len(unique),
        },
        tasks=tuple(unique.values()),
    )
    episodes = [
        {
            "episode_index": index,
            "episode_id": f"g2po-valid-seen-{index:03d}",
            "environment_seed": seed_start + index,
            "task_id": task.task_id,
            "game_path": task.game_path,
        }
        for index, task in enumerate(selected)
    ]
    schedule = {
        "schema": G2PO_EVALUATION_SCHEDULE_SCHEMA,
        "source_revision": source_revision,
        "split": "valid_seen",
        "seed_start": seed_start,
        "episode_count": len(episodes),
        "selection_contract": {
            "eval_dataset": "eval_in_distribution",
            "worker_seed_formula": "1000 + episode_index",
            "worker_count": len(episodes),
            "source_files": dict(sorted(source_files.items())),
            "filesystem_walk_order_bound": True,
        },
        "episodes": episodes,
    }
    return manifest, schedule


def _select_source_worker_games(
    *,
    source_root: Path,
    data_root: Path,
    episode_count: int,
    seed_start: int,
) -> tuple[str, ...]:
    package_root = source_root / "agent_system/environments/env_package/alfworld"
    config_path = package_root / "configs/config_tw.yaml"
    if not package_root.is_dir() or not config_path.is_file():
        raise ValueError("pinned G2PO ALFWorld package/config is missing")
    for entry in (str(source_root), str(package_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.environ["ALFWORLD_DATA"] = str(data_root)
    try:
        import alfworld
        from alfworld.agents.environment import get_environment
    except ImportError as error:
        raise RuntimeError(
            "freezing G2PO evaluation requires pinned ALFWorld/TextWorld dependencies"
        ) from error
    module_path = Path(str(alfworld.__file__)).resolve()
    if package_root not in module_path.parents:
        raise RuntimeError("imported ALFWorld is not the pinned G2PO package")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = get_environment("AlfredTWEnv")(
        config,
        train_eval="eval_in_distribution",
    )
    selected: list[str] = []
    for index in range(episode_count):
        environment = base.init_env(batch_size=1)
        try:
            environment.seed(seed_start + index)
            _, infos = environment.reset()
            selected.append(str(_first_scalar(infos["extra.gamefile"])))
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
    return tuple(selected)


def _first_scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("official ALFWorld reset returned an empty gamefile")
        value = value[0]
    return value


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip())
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
