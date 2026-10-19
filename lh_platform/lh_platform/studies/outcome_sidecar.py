"""Native ALFWorld task-outcome sidecar and auditable upstream patch builder."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OUTCOME_SCHEMA = "lh_native_eval_outcome/v1"
_EVALUATION_STEP = -1


def set_evaluation_step(step: int) -> None:
    global _EVALUATION_STEP
    _EVALUATION_STEP = int(step)


def emit_native_outcome(*, gamefile: str, won: float | bool, episode_steps: int) -> None:
    """Append one benchmark-owned outcome; silently disabled without an output path."""

    output = os.environ.get("LH_NATIVE_OUTCOME_PATH")
    if not output:
        return
    game_path = Path(gamefile).resolve()
    trajectory_path = game_path.parent / "traj_data.json"
    trajectory: dict[str, Any] = {}
    try:
        raw = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            trajectory = raw
    except (OSError, ValueError):
        pass
    row = {
        "schema": OUTCOME_SCHEMA,
        "study_id": os.environ.get("LH_STUDY_ID", ""),
        "variant_id": os.environ.get("LH_VARIANT_ID", ""),
        "seed": int(os.environ.get("LH_SEED", "-1")),
        "checkpoint_step": _EVALUATION_STEP,
        "task_id": str(trajectory.get("task_id") or _relative_game_id(game_path)),
        "task_type": str(trajectory.get("task_type") or "unknown"),
        "gamefile": _relative_game_id(game_path),
        "won": bool(won),
        "episode_steps": int(episode_steps),
        "outcome_source": "info['won']",
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _relative_game_id(path: Path) -> str:
    parts = path.parts
    if "json_2.1.1" in parts:
        return "/".join(parts[parts.index("json_2.1.1") :])
    return path.as_posix()


@dataclass(frozen=True)
class OutcomeSidecarBundle:
    schema: str
    patch_path: str
    source_commit: str
    source_quotes: tuple[str, ...]

    def write_manifest(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target


def build_outcome_sidecar_patch(
    *, source_root: str | Path, source_commit: str, output_dir: str | Path
) -> OutcomeSidecarBundle:
    source = Path(source_root)
    manager = source / "recipe/GraphGPO/graphgpo_env_manager.py"
    trainer = source / "recipe/GraphGPO/graphgpo_ray_trainer.py"
    manager_text = manager.read_text(encoding="utf-8")
    trainer_text = trainer.read_text(encoding="utf-8")
    quotes = (
        "won_value = float(info['won'])",
        "success['success_rate'].append(won_value)",
        "def _validate(self):",
        "test_output_gen_batch = self.traj_collector.multi_turn_loop(",
    )
    if not all(quote in manager_text + trainer_text for quote in quotes):
        raise ValueError("official outcome emission seams do not match the pinned source")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    patch_path = target / "graphgpo_native_outcome.patch"
    patch_path.write_text(_PATCH, encoding="utf-8")
    bundle = OutcomeSidecarBundle(
        schema="lh_native_outcome_sidecar/v1",
        patch_path=str(patch_path),
        source_commit=source_commit,
        source_quotes=quotes,
    )
    bundle.write_manifest(target / "native_outcome_sidecar_manifest.json")
    return bundle


_PATCH = """diff --git a/recipe/GraphGPO/graphgpo_env_manager.py b/recipe/GraphGPO/graphgpo_env_manager.py
--- a/recipe/GraphGPO/graphgpo_env_manager.py
+++ b/recipe/GraphGPO/graphgpo_env_manager.py
@@ -124,5 +124,11 @@ class EnvironmentManagerBase:
                 info = total_infos[batch_idx][i]
                 won_value = float(info['won'])
                 success['success_rate'].append(won_value)
+                from lh_platform.studies.outcome_sidecar import emit_native_outcome
+                emit_native_outcome(
+                    gamefile=self.gamefile[batch_idx],
+                    won=won_value,
+                    episode_steps=i + 1,
+                )

                 data_source = info.get("data_source")
diff --git a/recipe/GraphGPO/graphgpo_ray_trainer.py b/recipe/GraphGPO/graphgpo_ray_trainer.py
--- a/recipe/GraphGPO/graphgpo_ray_trainer.py
+++ b/recipe/GraphGPO/graphgpo_ray_trainer.py
@@ -707,4 +707,6 @@ class RayPPOTrainer:
         self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

     def _validate(self):
+        from lh_platform.studies.outcome_sidecar import set_evaluation_step
+        set_evaluation_step(self.global_steps)
         reward_tensor_lst = []
"""
