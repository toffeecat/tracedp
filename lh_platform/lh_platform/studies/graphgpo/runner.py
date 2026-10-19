"""GraphGPO official subprocess preparation."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lh_platform.studies.protocol import (
    CompiledStudy,
    ExecutionPath,
    StudyProtocol,
    TrainingPath,
    TrialPlan,
)
from lh_platform.studies.runtime import (
    GRAPHGPO_MAX_EPISODE_STEPS,
    GRAPHGPO_MAX_RESPONSE_LENGTH,
    GRAPHGPO_REQUIRED_GPUS,
    GRAPHGPO_TRAIN_BATCH_SIZE,
    GRAPHGPO_VAL_BATCH_SIZE,
    RuntimePreflightReport,
    sha256_file,
)

GRAPHGPO_ALLOWED_ABLATIONS = ("episode_only", "graph_only", "similarity_merge")
PAPER_REPRODUCTION_CLAIM_SCOPE = "paper_reproduction_formal"
RESOURCE_CONSTRAINED_CLAIM_SCOPE = "resource_constrained_ablation"
SINGLE_5090_SMALL_RESOURCE_PROFILE = "single_5090_small"
SINGLE_5090_GROUP2_RESOURCE_PROFILE = "single_5090_group2"
SINGLE_5090_GROUP8_RESOURCE_PROFILE = "single_5090_group8"
SINGLE_5090_FULL_ROLLOUT_GROUP8_RESOURCE_PROFILE = "single_5090_full_rollout_group8"
SINGLE_5090_FULL_ROLLOUT_GROUP8_TRAINPATH_RESOURCE_PROFILE = (
    "single_5090_full_rollout_group8_trainpath"
)
SINGLE_5090_BASE_OVERRIDES: Mapping[str, Any] = {
    "actor_rollout_ref.model.path": "Qwen/Qwen2.5-0.5B-Instruct",
    "data.train_batch_size": 1,
    "data.val_batch_size": 1,
    "data.gen_batch_size": 1,
    "data.max_response_length": 256,
    "actor_rollout_ref.actor.ppo_mini_batch_size": 1,
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 3072,
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 3072,
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": 3072,
    "actor_rollout_ref.model.enable_activation_offload": True,
    "actor_rollout_ref.actor.fsdp_config.param_offload": True,
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.20,
    "actor_rollout_ref.rollout.enforce_eager": True,
    "actor_rollout_ref.rollout.free_cache_engine": True,
    "env.rollout.n": 1,
    "ray_init.num_cpus": 4,
    "+ray_init.include_dashboard": False,
    "trainer.logger": ["console"],
    "trainer.n_gpus_per_node": 1,
    "trainer.save_freq": -1,
    "trainer.max_actor_ckpt_to_keep": 1,
}
SINGLE_5090_SMALL_OVERRIDES: Mapping[str, Any] = {
    **SINGLE_5090_BASE_OVERRIDES,
    "env.rollout.n": 1,
}
SINGLE_5090_GROUP2_OVERRIDES: Mapping[str, Any] = {
    **SINGLE_5090_BASE_OVERRIDES,
    "env.rollout.n": 2,
}
SINGLE_5090_GROUP8_OVERRIDES: Mapping[str, Any] = {
    **SINGLE_5090_BASE_OVERRIDES,
    "env.rollout.n": 8,
}
SINGLE_5090_FULL_ROLLOUT_GROUP8_OVERRIDES: Mapping[str, Any] = {
    **SINGLE_5090_BASE_OVERRIDES,
    # The official trainer builds its dataloader from gen_batch_size when it
    # is present, then rollout_loop.py repeats that batch env.rollout.n times.
    # Bind both fields so this profile actually produces 16 * 8 trajectories.
    "data.train_batch_size": 16,
    "data.gen_batch_size": 16,
    "data.val_batch_size": 128,
    "env.rollout.n": 8,
    # Keep the source recipe's PPO minibatch.  GraphGPO's rollout loop flattens
    # every active environment step into the actor batch, so inheriting the
    # small-profile value (1) changes one source minibatch into many optimizer
    # steps.  One GPU may reduce *microbatch* capacity, but must not change
    # this optimization boundary.
    "actor_rollout_ref.actor.ppo_mini_batch_size": 256,
    # graphgpo_ray_trainer.py otherwise starts eight process DataLoader
    # workers for both train and validation.  On this 45-GB single-GPU
    # container their forked copies can trigger cgroup OOM after the update;
    # zero workers preserves row order and all training semantics.
    "+data.dataloader_num_workers": 0,
    # Keep the official 16 x 8 logical batch, while serializing environment
    # placement one prompt group at a time on the single-GPU host.  The source
    # collector aggregates all 128 trajectories before one PPO update.
    "+env.rollout.sequential_groups_per_chunk": 1,
    "+env.rollout.sequential_validation_groups_per_chunk": 1,
    "ray_init.num_cpus": 8,
}
SINGLE_5090_FULL_ROLLOUT_GROUP8_TRAINPATH_OVERRIDES: Mapping[str, Any] = {
    **SINGLE_5090_FULL_ROLLOUT_GROUP8_OVERRIDES,
    # Official trainer source performs validation before training when this
    # flag is true, and always validates the final step when test_freq > 0.
    # This profile is intentionally training-path-only: it proves one complete
    # 16 x 8 rollout/update before launching the separately auditable full eval.
    "trainer.val_before_train": False,
    "trainer.test_freq": -1,
}
RESOURCE_PROFILE_OVERRIDES: Mapping[str, Mapping[str, Any]] = {
    SINGLE_5090_SMALL_RESOURCE_PROFILE: SINGLE_5090_SMALL_OVERRIDES,
    SINGLE_5090_GROUP2_RESOURCE_PROFILE: SINGLE_5090_GROUP2_OVERRIDES,
    SINGLE_5090_GROUP8_RESOURCE_PROFILE: SINGLE_5090_GROUP8_OVERRIDES,
    SINGLE_5090_FULL_ROLLOUT_GROUP8_RESOURCE_PROFILE: SINGLE_5090_FULL_ROLLOUT_GROUP8_OVERRIDES,
    SINGLE_5090_FULL_ROLLOUT_GROUP8_TRAINPATH_RESOURCE_PROFILE: SINGLE_5090_FULL_ROLLOUT_GROUP8_TRAINPATH_OVERRIDES,
}
SINGLE_5090_RESOURCE_ENVIRONMENT: Mapping[str, str] = {
    "LH_ATTENTION_IMPLEMENTATION": "eager",
}
SINGLE_5090_SMALL_ENVIRONMENT = SINGLE_5090_RESOURCE_ENVIRONMENT
RESOURCE_PROFILE_ENVIRONMENTS: Mapping[str, Mapping[str, str]] = {
    SINGLE_5090_SMALL_RESOURCE_PROFILE: SINGLE_5090_RESOURCE_ENVIRONMENT,
    SINGLE_5090_GROUP2_RESOURCE_PROFILE: SINGLE_5090_RESOURCE_ENVIRONMENT,
    SINGLE_5090_GROUP8_RESOURCE_PROFILE: SINGLE_5090_RESOURCE_ENVIRONMENT,
    SINGLE_5090_FULL_ROLLOUT_GROUP8_RESOURCE_PROFILE: {
        **SINGLE_5090_RESOURCE_ENVIRONMENT,
        "LH_GRAPHGPO_ALFWORLD_EXECUTION_MODE": "sequential_scalar",
    },
    SINGLE_5090_FULL_ROLLOUT_GROUP8_TRAINPATH_RESOURCE_PROFILE: {
        **SINGLE_5090_RESOURCE_ENVIRONMENT,
        "LH_GRAPHGPO_ALFWORLD_EXECUTION_MODE": "sequential_scalar",
    },
}
ALLOWED_RESOURCE_PROFILES = tuple(RESOURCE_PROFILE_OVERRIDES.keys())

__all__ = [
    "GRAPHGPO_ALLOWED_ABLATIONS",
    "PAPER_REPRODUCTION_CLAIM_SCOPE",
    "RESOURCE_CONSTRAINED_CLAIM_SCOPE",
    "SINGLE_5090_SMALL_RESOURCE_PROFILE",
    "SINGLE_5090_GROUP2_RESOURCE_PROFILE",
    "SINGLE_5090_GROUP8_RESOURCE_PROFILE",
    "SINGLE_5090_FULL_ROLLOUT_GROUP8_RESOURCE_PROFILE",
    "SINGLE_5090_FULL_ROLLOUT_GROUP8_TRAINPATH_RESOURCE_PROFILE",
    "SINGLE_5090_BASE_OVERRIDES",
    "SINGLE_5090_SMALL_OVERRIDES",
    "SINGLE_5090_GROUP2_OVERRIDES",
    "SINGLE_5090_GROUP8_OVERRIDES",
    "SINGLE_5090_FULL_ROLLOUT_GROUP8_OVERRIDES",
    "SINGLE_5090_FULL_ROLLOUT_GROUP8_TRAINPATH_OVERRIDES",
    "RESOURCE_PROFILE_OVERRIDES",
    "SINGLE_5090_RESOURCE_ENVIRONMENT",
    "SINGLE_5090_SMALL_ENVIRONMENT",
    "RESOURCE_PROFILE_ENVIRONMENTS",
    "ALLOWED_RESOURCE_PROFILES",
    "resource_profile_required_gpus",
    "StudyLaunchBlocked",
    "SourceMappingEvidence",
    "PreparedTrialCommand",
    "PreparedStudy",
    "GraphGPOStudyRunner",
]


def resource_profile_required_gpus(profile: str) -> int:
    """Return the execution GPU count declared by a resource profile."""

    try:
        return int(RESOURCE_PROFILE_OVERRIDES[profile]["trainer.n_gpus_per_node"])
    except KeyError as error:
        raise ValueError(f"unknown resource profile: {profile}") from error


_VLLM_LORA_COMPAT = '''"""Runtime compatibility shims for dependency drift.

This module is loaded by Python's sitecustomize hook before GraphGPO imports.
It does not modify the pinned upstream source tree.  It only restores the
``vllm.lora.models`` import path expected by the vendored verl code when a newer
vLLM exposes ``LoRAModel`` from ``vllm.lora.worker_manager``.

It also provides a pure PyTorch fallback for ``flash_attn.bert_padding`` when
the official flash-attn package is unavailable.  The pinned upstream GraphGPO
path imports these padding helpers at module import time.  Real flash-attn
always wins when installed; the fallback is for calibration on environments
where no wheel exists and source compilation is not viable.

Finally, it restores the ``transformers.AutoModelForVision2Seq`` symbol used by
the vendored verl code when newer transformers releases rename/remove that
compatibility export.  Text-only Qwen runs still select ``AutoModelForCausalLM``.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types


class _VLLMLoraModelsLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        from vllm.lora.worker_manager import LoRAModel

        module.LoRAModel = LoRAModel
        module.__all__ = ["LoRAModel"]


class _VLLMLoraModelsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm.lora.models":
            return importlib.util.spec_from_loader(fullname, _VLLMLoraModelsLoader())
        return None


def _real_spec(fullname, path=None):
    return importlib.machinery.PathFinder.find_spec(fullname, path)


# The flash-attn compatibility helpers below reimplement flash-attention's bert_padding utilities.
# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file (BSD-3-Clause).
# The full license text is in THIRD_PARTY_NOTICES.md.
class _FlashAttnCompatLoader(importlib.abc.Loader):
    def __init__(self, *, is_package=False):
        self._is_package = is_package

    def create_module(self, spec):
        if self._is_package:
            module = types.ModuleType(spec.name)
            module.__path__ = []
            return module
        return None

    def exec_module(self, module):
        if self._is_package:
            module.__all__ = ["bert_padding"]
            return

        import torch
        import torch.nn.functional as F
        from einops import rearrange

        def index_first_axis(a, indices):
            return a[indices]

        def unpad_input(hidden_states, attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).reshape(-1)
            max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() else 0
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            unpadded = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)
            return unpadded, indices, cu_seqlens, max_seqlen_in_batch

        def pad_input(hidden_states, indices, batch, seqlen):
            output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
            output[indices] = hidden_states
            return rearrange(output, "(b s) ... -> b s ...", b=batch)

        module.index_first_axis = index_first_axis
        module.unpad_input = unpad_input
        module.pad_input = pad_input
        module.rearrange = rearrange
        module.__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]


class _FlashAttnCompatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flash_attn" and _real_spec(fullname, path) is None:
            return importlib.util.spec_from_loader(
                fullname, _FlashAttnCompatLoader(is_package=True), is_package=True
            )
        if fullname == "flash_attn.bert_padding" and _real_spec(fullname, path) is None:
            return importlib.util.spec_from_loader(fullname, _FlashAttnCompatLoader())
        return None


def _runtime_env_vars_for_nested_ray_workers():
    env_vars = {}
    for key in (
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "LH_ATTENTION_IMPLEMENTATION",
    ):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return env_vars


def _patch_verl_ray_base(module):
    cls = getattr(module, "RayClassWithInitArgs", None)
    if cls is None or getattr(cls, "_lh_runtime_env_patched", False):
        return
    original = cls.update_options

    def update_options(self, options):
        options = dict(options)
        runtime_env = dict(options.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})
        for key, value in _runtime_env_vars_for_nested_ray_workers().items():
            env_vars.setdefault(key, value)
        runtime_env["env_vars"] = env_vars
        options["runtime_env"] = runtime_env
        return original(self, options)

    cls.update_options = update_options
    cls._lh_runtime_env_patched = True


class _PatchVerlRayBaseLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        _patch_verl_ray_base(module)


class _PatchVerlRayBaseFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "verl.single_controller.ray.base":
            return None
        spec = _real_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchVerlRayBaseLoader(spec.loader)
        return spec


if not any(type(finder).__name__ == "_VLLMLoraModelsFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, _VLLMLoraModelsFinder())
if not any(type(finder).__name__ == "_FlashAttnCompatFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, _FlashAttnCompatFinder())
if not any(type(finder).__name__ == "_PatchVerlRayBaseFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, _PatchVerlRayBaseFinder())


def _patch_transformers_vision2seq_alias():
    try:
        import transformers
    except Exception:
        return
    if hasattr(transformers, "AutoModelForVision2Seq"):
        return
    replacement = getattr(transformers, "AutoModelForImageTextToText", None)
    if replacement is None:
        class _UnavailableAutoModelForVision2Seq:
            _model_mapping = {}

            @classmethod
            def from_config(cls, *args, **kwargs):
                raise ImportError("AutoModelForVision2Seq is unavailable in this transformers build")

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise ImportError("AutoModelForVision2Seq is unavailable in this transformers build")

        replacement = _UnavailableAutoModelForVision2Seq
    transformers.AutoModelForVision2Seq = replacement


_patch_transformers_vision2seq_alias()

# The full 16 x 8 single-GPU profile retains the source scalar workers and
# GraphGPO recipe collector, but bounds simultaneous environment placement to
# one same-seed group.  It executes before GraphGPO imports the recipe modules
# and never writes to the upstream mirror.
if os.environ.get("LH_GRAPHGPO_ALFWORLD_EXECUTION_MODE") == "grouped_scalar":
    from lh_platform.studies.graphgpo.grouped_alfworld import (
        install_grouped_scalar_alfworld_runtime,
    )

    install_grouped_scalar_alfworld_runtime()
elif os.environ.get("LH_GRAPHGPO_ALFWORLD_EXECUTION_MODE") == "sequential_scalar":
    from lh_platform.studies.graphgpo.grouped_alfworld import (
        install_sequential_scalar_alfworld_runtime,
    )

    install_sequential_scalar_alfworld_runtime()
'''
_TRANSFORMERS_PROXY_COMPAT = '''"""Proxy to the installed transformers package with GraphGPO API compatibility."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_COMPAT_ROOT = Path(__file__).resolve().parents[1]
_THIS_FILE = Path(__file__).resolve()
_REAL_SPEC = None

def _is_longhorizon_transformers_proxy(path: Path) -> bool:
    try:
        if path.resolve() == _THIS_FILE:
            return True
        return "Proxy to the installed transformers package with GraphGPO API compatibility" in path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return False

for entry in tuple(sys.path):
    if not entry:
        continue
    root = Path(entry).resolve()
    if root == _COMPAT_ROOT:
        continue
    candidate = root / "transformers" / "__init__.py"
    if candidate.is_file() and not _is_longhorizon_transformers_proxy(candidate):
        _REAL_SPEC = importlib.util.spec_from_file_location(
            __name__,
            candidate,
            submodule_search_locations=[str(candidate.parent)],
        )
        break

if _REAL_SPEC is None or _REAL_SPEC.loader is None:
    raise ImportError("could not locate installed transformers package behind LongHorizon proxy")

_REAL_MODULE = importlib.util.module_from_spec(_REAL_SPEC)
sys.modules[__name__] = _REAL_MODULE
_REAL_SPEC.loader.exec_module(_REAL_MODULE)

if not hasattr(_REAL_MODULE, "AutoModelForVision2Seq"):
    replacement = getattr(_REAL_MODULE, "AutoModelForImageTextToText", None)
    if replacement is None:
        class _UnavailableAutoModelForVision2Seq:
            _model_mapping = {}

            @classmethod
            def from_config(cls, *args, **kwargs):
                raise ImportError("AutoModelForVision2Seq is unavailable in this transformers build")

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise ImportError("AutoModelForVision2Seq is unavailable in this transformers build")

        replacement = _UnavailableAutoModelForVision2Seq
    _REAL_MODULE.AutoModelForVision2Seq = replacement

globals().update(_REAL_MODULE.__dict__)
'''


class StudyLaunchBlocked(RuntimeError):
    """Raised when a subprocess launch lacks complete runtime validity evidence."""


@dataclass(frozen=True)
class SourceMappingEvidence:
    path: str
    sha256: str
    source_quotes: tuple[str, ...]
    decision: str
    verified: bool


@dataclass(frozen=True)
class PreparedTrialCommand:
    trial_id: str
    role: str
    variant_id: str
    seed: int
    argv: tuple[str, ...]
    overrides: Mapping[str, Any]
    cwd: str
    environment: Mapping[str, str]
    output_dir: str
    source_commit: str
    source_recipe: str
    source_recipe_sha256: str
    planned_execution_path: str
    # ``resource_budget`` is the budget implied by the command overrides.  It
    # must never be the paper budget for a resource-constrained run, otherwise
    # a calibration can be reported as if it executed the formal recipe.
    resource_budget: Mapping[str, int]
    # The declared protocol budget remains immutable evidence for the formal
    # recipe and is deliberately kept separate from the effective command.
    official_budget: Mapping[str, int]
    claim_scope: str = PAPER_REPRODUCTION_CLAIM_SCOPE
    resource_profile: str = ""


@dataclass(frozen=True)
class PreparedStudy:
    schema: str
    study_id: str
    launch_allowed: bool
    blocked_by: tuple[str, ...]
    preprocess_argv: tuple[str, ...]
    preprocess_cwd: str
    preprocess_environment: Mapping[str, str]
    source_mapping: tuple[SourceMappingEvidence, ...]
    trials: tuple[PreparedTrialCommand, ...]

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for compat_dir in _prepared_compat_dirs(self):
            _write_runtime_compat(compat_dir)
        summary_path = root / "prepare_summary.json"
        commands_path = root / "planned_commands.jsonl"
        summary_path.write_text(
            json.dumps(
                {
                    "schema": self.schema,
                    "study_id": self.study_id,
                    "launch_allowed": self.launch_allowed,
                    "blocked_by": list(self.blocked_by),
                    "preprocess_argv": list(self.preprocess_argv),
                    "preprocess_cwd": self.preprocess_cwd,
                    "preprocess_environment": dict(self.preprocess_environment),
                    "source_mapping": [asdict(evidence) for evidence in self.source_mapping],
                    "trial_count": len(self.trials),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        commands_path.write_text(
            "".join(json.dumps(asdict(trial), sort_keys=True) + "\n" for trial in self.trials),
            encoding="utf-8",
        )
        return summary_path, commands_path


class GraphGPOStudyRunner:
    """Map one compiled study to the pinned official GraphGPO entrypoint."""

    def prepare(
        self,
        protocol: StudyProtocol,
        compiled: CompiledStudy,
        runtime_report: RuntimePreflightReport,
        *,
        source_root: str | Path,
        alfworld_data: str | Path,
        output_root: str | Path,
        python_executable: str,
        claim_scope: str = PAPER_REPRODUCTION_CLAIM_SCOPE,
        resource_profile: str = "",
        resource_training_steps: int | None = None,
        resource_model_path: str = "",
    ) -> PreparedStudy:
        self._require_native_protocol(protocol)
        if resource_training_steps is not None and not resource_profile:
            raise ValueError("resource_training_steps requires resource_profile")
        if resource_model_path and not resource_profile:
            raise ValueError("resource_model_path requires resource_profile")
        if resource_profile and claim_scope != RESOURCE_CONSTRAINED_CLAIM_SCOPE:
            raise ValueError("resource profiles must use claim_scope=resource_constrained_ablation")
        source = Path(source_root).resolve()
        output = Path(output_root).resolve()
        compat_dir = output / "python_compat"
        data_root = output / "upstream_data"
        source_recipe = source / "recipe/GraphGPO/run_qwen2.5_1.5b_alfworld_train.sh"
        source_commit = str(protocol.design.fixed_factors["source_commit"])
        source_mapping = _source_mapping_evidence(source, source_recipe)
        mapping_verified = all(evidence.verified for evidence in source_mapping)
        blocked_by = tuple(runtime_report.blocked_by)
        if not mapping_verified:
            blocked_by = (*blocked_by, "upstream_command_mapping_unverified")
        ablation_ids = tuple(str(item.get("id", "")) for item in protocol.design.ablations)
        requires_ablations = any(str(kind) == "ablation" for kind in protocol.kinds)
        ablations_verified = (
            ablation_ids == GRAPHGPO_ALLOWED_ABLATIONS if requires_ablations else not ablation_ids
        )
        if not ablations_verified:
            blocked_by = (*blocked_by, "ablation_set_not_preregistered")
        allowed_variant_ids = {
            protocol.design.control.id,
            protocol.design.treatment.id,
            *GRAPHGPO_ALLOWED_ABLATIONS,
        }
        environment = {
            "ALFWORLD_DATA": str(Path(alfworld_data)),
            "PYTHONPATH": _graphgpo_pythonpath(source, compat_dir),
        }
        preprocess = (
            python_executable,
            "-m",
            "examples.data_preprocess.prepare",
            "--mode",
            "text",
            "--local_dir",
            str(data_root),
            "--train_data_size",
            str(GRAPHGPO_TRAIN_BATCH_SIZE),
            "--val_data_size",
            str(GRAPHGPO_VAL_BATCH_SIZE),
        )
        trials = tuple(
            self._trial_command(
                protocol,
                trial,
                source=source,
                source_commit=source_commit,
                source_recipe=source_recipe,
                data_root=data_root,
                alfworld_data=Path(alfworld_data),
                output_root=output,
                compat_dir=compat_dir,
                python_executable=python_executable,
                claim_scope=claim_scope,
                resource_profile=resource_profile,
                resource_training_steps=resource_training_steps,
                resource_model_path=resource_model_path,
            )
            for trial in compiled.trials
            if trial.variant_id in allowed_variant_ids
        )
        return PreparedStudy(
            schema="lh_prepared_upstream_study/v1",
            study_id=protocol.id,
            launch_allowed=runtime_report.run_ready and mapping_verified and ablations_verified,
            blocked_by=blocked_by,
            preprocess_argv=preprocess,
            preprocess_cwd=str(source),
            preprocess_environment=environment,
            source_mapping=source_mapping,
            trials=trials,
        )

    @staticmethod
    def _require_native_protocol(protocol: StudyProtocol) -> None:
        """Prevent the native GraphGPO launcher from executing a mismatched plan."""

        if protocol.design.benchmark != "alfworld_native":
            raise ValueError("GraphGPOStudyRunner only supports benchmark='alfworld_native'")
        for role, variant in (
            ("control", protocol.design.control),
            ("treatment", protocol.design.treatment),
        ):
            if variant.requested_training_path != TrainingPath.UPSTREAM_TRAINING:
                raise ValueError(f"design.{role}.requested_training_path must be upstream_training")
            if variant.requested_execution_path != ExecutionPath.UPSTREAM_NATIVE_RUNNER:
                raise ValueError(
                    f"design.{role}.requested_execution_path must be upstream_native_runner"
                )

    def launch_preprocess(self, prepared: PreparedStudy) -> subprocess.CompletedProcess[str]:
        self._require_launch_allowed(prepared)
        return subprocess.run(
            prepared.preprocess_argv,
            cwd=prepared.preprocess_cwd,
            env={**os.environ, **prepared.preprocess_environment},
            check=True,
            text=True,
        )

    def launch(self, prepared: PreparedStudy, *, trial_id: str) -> subprocess.CompletedProcess[str]:
        self._require_launch_allowed(prepared)
        trial = next((item for item in prepared.trials if item.trial_id == trial_id), None)
        if trial is None:
            raise KeyError(f"unknown prepared trial: {trial_id}")
        Path(trial.output_dir).mkdir(parents=True, exist_ok=True)
        return subprocess.run(
            trial.argv,
            cwd=trial.cwd,
            env={**os.environ, **trial.environment},
            check=True,
            text=True,
        )

    @staticmethod
    def _require_launch_allowed(prepared: PreparedStudy) -> None:
        if not prepared.launch_allowed:
            raise StudyLaunchBlocked(
                "runtime preflight blocked launch: " + ", ".join(prepared.blocked_by)
            )

    @staticmethod
    def _trial_command(
        protocol: StudyProtocol,
        trial: TrialPlan,
        *,
        source: Path,
        source_commit: str,
        source_recipe: Path,
        data_root: Path,
        alfworld_data: Path,
        output_root: Path,
        compat_dir: Path,
        python_executable: str,
        claim_scope: str,
        resource_profile: str,
        resource_training_steps: int | None,
        resource_model_path: str,
    ) -> PreparedTrialCommand:
        official_budget = dict(trial.budget)
        budget = dict(official_budget)
        if resource_training_steps is not None:
            budget["training_steps"] = int(resource_training_steps)
        training_steps = int(budget["training_steps"])
        group_size = int(trial.fixed_factors["rollout_group_size"])
        trial_id = f"{protocol.id}:{trial.variant_id}:seed-{trial.seed}"
        trial_output = output_root / "trials" / trial.variant_id / f"seed-{trial.seed}"
        parameters = trial.parameters
        pythonpath = _graphgpo_pythonpath(source, compat_dir)
        overrides: dict[str, Any] = {
            "+ray_init.runtime_env.env_vars.PYTHONPATH": pythonpath,
            "algorithm.adv_estimator": parameters["advantage_estimator"],
            "algorithm.gamma": 0.10,
            "algorithm.graphgpo.step_advantage_w": parameters["step_advantage_w"],
            "algorithm.graphgpo.episode_advantage_w": parameters["episode_advantage_w"],
            "algorithm.graphgpo.mode": "mean_std_norm",
            "algorithm.graphgpo.enable_similarity": parameters["enable_similarity"],
            "algorithm.graphgpo.similarity_thresh": 0.95,
            "algorithm.graphgpo.normalize_distance": False,
            "algorithm.use_kl_in_reward": False,
            "data.train_files": str(data_root / "text/train.parquet"),
            "data.val_files": str(data_root / "text/test.parquet"),
            "data.train_batch_size": GRAPHGPO_TRAIN_BATCH_SIZE,
            "data.val_batch_size": GRAPHGPO_VAL_BATCH_SIZE,
            "data.max_prompt_length": 2048,
            "data.max_response_length": GRAPHGPO_MAX_RESPONSE_LENGTH,
            "data.filter_overlong_prompts": True,
            "data.truncation": "error",
            "data.return_raw_chat": True,
            "actor_rollout_ref.model.path": trial.fixed_factors["initial_checkpoint"],
            "actor_rollout_ref.actor.optim.lr": parameters["learning_rate"],
            "actor_rollout_ref.model.use_remove_padding": True,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 256,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": int(
                trial.fixed_factors.get("ppo_micro_batch_size_per_gpu", 32)
            ),
            "actor_rollout_ref.actor.use_kl_loss": True,
            "actor_rollout_ref.actor.kl_loss_coef": 0.01,
            "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
            "actor_rollout_ref.model.enable_gradient_checkpointing": True,
            "actor_rollout_ref.actor.fsdp_config.param_offload": False,
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": int(
                trial.fixed_factors.get("log_prob_micro_batch_size_per_gpu", 32)
            ),
            "actor_rollout_ref.rollout.tensor_model_parallel_size": int(
                trial.fixed_factors.get("tensor_model_parallel_size", 2)
            ),
            "actor_rollout_ref.rollout.name": "vllm",
            "actor_rollout_ref.rollout.gpu_memory_utilization": float(
                trial.fixed_factors.get("gpu_memory_utilization", 0.5)
            ),
            "actor_rollout_ref.rollout.enable_chunked_prefill": False,
            "actor_rollout_ref.rollout.enforce_eager": False,
            "actor_rollout_ref.rollout.free_cache_engine": False,
            "actor_rollout_ref.rollout.val_kwargs.temperature": 0.4,
            "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": int(
                trial.fixed_factors.get("log_prob_micro_batch_size_per_gpu", 32)
            ),
            "actor_rollout_ref.ref.fsdp_config.param_offload": True,
            "actor_rollout_ref.actor.use_invalid_action_penalty": True,
            "actor_rollout_ref.actor.invalid_action_penalty_coef": 0.1,
            "env.env_name": "alfworld/AlfredTWEnv",
            "env.seed": trial.seed,
            "env.max_steps": GRAPHGPO_MAX_EPISODE_STEPS,
            "env.rollout.n": group_size,
            "env.resources_per_worker.num_cpus": 0.2,
            "trainer.critic_warmup": 0,
            "trainer.logger": ["console", "wandb"],
            "trainer.project_name": "verl_agent_alfworld",
            "trainer.experiment_name": trial_id.replace(":", "__"),
            "trainer.n_gpus_per_node": int(
                trial.fixed_factors.get("num_gpus", GRAPHGPO_REQUIRED_GPUS)
            ),
            "trainer.nnodes": 1,
            "trainer.save_freq": int(trial.fixed_factors.get("save_freq", -1)),
            "trainer.max_actor_ckpt_to_keep": int(
                trial.fixed_factors.get("max_actor_ckpt_to_keep", 2)
            ),
            "trainer.resume_mode": "auto",
            "trainer.test_freq": 5,
            "trainer.total_epochs": 150,
            "trainer.total_training_steps": training_steps,
            "trainer.default_local_dir": str(trial_output / "checkpoints"),
            "trainer.rollout_data_dir": str(trial_output / "rollouts"),
            "trainer.val_before_train": True,
        }
        if resource_profile:
            profile_overrides = RESOURCE_PROFILE_OVERRIDES.get(resource_profile)
            if profile_overrides is None:
                raise ValueError(f"unknown resource profile: {resource_profile}")
            overrides.update(profile_overrides)
            overrides.update(
                {
                    f"+ray_init.runtime_env.env_vars.{key}": value
                    for key, value in RESOURCE_PROFILE_ENVIRONMENTS[resource_profile].items()
                }
            )
            overrides["trainer.total_training_steps"] = training_steps
            overrides["trainer.total_epochs"] = training_steps
            if resource_model_path:
                overrides["actor_rollout_ref.model.path"] = resource_model_path
        effective_budget = _effective_command_budget(overrides)
        environment = {
            "ALFWORLD_DATA": str(alfworld_data),
            "PYTHONPATH": pythonpath,
            "LH_STARTUP_COMPAT_PATH": str(compat_dir),
            "LH_UPSTREAM_SOURCE_ROOT": str(source),
            "LH_STUDY_ID": protocol.id,
            "LH_VARIANT_ID": trial.variant_id,
            "LH_SEED": str(trial.seed),
            "LH_NATIVE_OUTCOME_PATH": str(trial_output / "native_eval_outcomes.jsonl"),
        }
        if resource_profile:
            environment.update(RESOURCE_PROFILE_ENVIRONMENTS[resource_profile])
        argv = (
            python_executable,
            "-m",
            "recipe.GraphGPO.main_graphgpo",
            *tuple(f"{key}={_hydra_value(value)}" for key, value in overrides.items()),
        )
        return PreparedTrialCommand(
            trial_id=trial_id,
            role=trial.role,
            variant_id=trial.variant_id,
            seed=trial.seed,
            argv=argv,
            overrides=overrides,
            cwd=str(source),
            environment=environment,
            output_dir=str(trial_output),
            source_commit=source_commit,
            source_recipe=str(source_recipe),
            source_recipe_sha256=sha256_file(source_recipe),
            planned_execution_path="upstream_native_runner",
            resource_budget=effective_budget,
            official_budget=official_budget,
            claim_scope=claim_scope,
            resource_profile=resource_profile,
        )


def _effective_command_budget(overrides: Mapping[str, Any]) -> dict[str, int]:
    """Derive the work scheduled by one prepared command from its overrides."""
    training_steps = int(overrides["trainer.total_training_steps"])
    train_batch_size = int(overrides["data.train_batch_size"])
    rollout_group_size = int(overrides["env.rollout.n"])
    max_episode_steps = int(overrides["env.max_steps"])
    max_response_length = int(overrides["data.max_response_length"])
    scheduled_rollouts = training_steps * train_batch_size * rollout_group_size
    environment_steps = scheduled_rollouts * max_episode_steps
    return {
        "training_steps": training_steps,
        "train_batch_size": train_batch_size,
        "rollout_group_size": rollout_group_size,
        "max_episode_steps": max_episode_steps,
        "max_response_length": max_response_length,
        "scheduled_rollouts": scheduled_rollouts,
        "environment_steps": environment_steps,
        "generated_tokens": environment_steps * max_response_length,
    }


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return "[" + ",".join(repr(item) for item in value) + "]"
    return str(value)


def _write_runtime_compat(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sitecustomize.py").write_text(_VLLM_LORA_COMPAT, encoding="utf-8")
    transformers_dir = directory / "transformers"
    transformers_dir.mkdir(parents=True, exist_ok=True)
    (transformers_dir / "__init__.py").write_text(_TRANSFORMERS_PROXY_COMPAT, encoding="utf-8")
    return directory


def _prepared_compat_dirs(prepared: PreparedStudy) -> tuple[Path, ...]:
    paths: set[Path] = set()
    environments = [
        prepared.preprocess_environment,
        *(trial.environment for trial in prepared.trials),
    ]
    for environment in environments:
        for item in str(environment.get("PYTHONPATH", "")).split(os.pathsep):
            path = Path(item)
            if path.name == "python_compat":
                paths.add(path)
    return tuple(sorted(paths, key=str))


def _graphgpo_pythonpath(source: Path, compat_dir: Path) -> str:
    return os.pathsep.join((str(compat_dir), str(Path(__file__).resolve().parents[2]), str(source)))


def _source_mapping_evidence(
    source: Path, source_recipe: Path
) -> tuple[SourceMappingEvidence, ...]:
    trainer_path = source / "recipe/GraphGPO/graphgpo_ray_trainer.py"
    alfworld_env_path = source / "recipe/GraphGPO/alfworld_envs.py"
    rollout_path = source / "recipe/GraphGPO/rollout_loop.py"
    environment_manager_path = source / "recipe/GraphGPO/graphgpo_env_manager.py"
    script_quotes = (
        "train_data_size=16",
        "val_data_size=128",
        "group_size=8",
        "env.rollout.n=$group_size",
        "python3 -m recipe.GraphGPO.main_graphgpo",
        "data.max_response_length=512",
        "env.max_steps=50",
        "trainer.n_gpus_per_node=2",
        "trainer.total_epochs=150",
    )
    trainer_quotes = (
        "if self.config.trainer.total_training_steps is not None:",
        "total_training_steps = self.config.trainer.total_training_steps",
        'if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):',
        "if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):",
    )
    return (
        _quote_evidence(
            source_recipe,
            script_quotes,
            "Prepared commands preserve the official entrypoint, batch sizes, rollout group, horizon, response length, GPU count, and epoch default.",
        ),
        _quote_evidence(
            trainer_path,
            trainer_quotes,
            "Prepared commands bind the study budget through upstream trainer.total_training_steps without changing the official epoch loop.",
        ),
        _quote_evidence(
            alfworld_env_path,
            (
                "self.env = base_env.init_env(batch_size=1)",
                "self.num_processes = env_num * group_n",
                "worker = env_worker.remote(config, seed + (i // self.group_n), base_env)",
                "assert len(actions) == self.num_processes",
            ),
            "The single-GPU runtime preserves GraphGPO recipe scalar worker construction, same-seed rollout grouping, and logical action-vector cardinality while changing only environment placement.",
        ),
        _quote_evidence(
            environment_manager_path,
            (
                "from recipe.GraphGPO.alfworld_envs import build_alfworld_envs",
                "config.data.train_batch_size, group_n, is_train=True",
            ),
            "The adaptation patches the recipe module that GraphGPO actually imports, rather than a similarly named non-recipe environment package.",
        ),
        _quote_evidence(
            rollout_path,
            (
                "gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)",
                "self.vanilla_multi_turn_loop(",
                "self.gather_rollout_data(",
                "success_rate[key] = np.mean(value)",
            ),
            "Sequential collection preserves source repeat-interleave group order and defers the one gather until all logical groups are complete.",
        ),
    )


def _quote_evidence(path: Path, quotes: tuple[str, ...], decision: str) -> SourceMappingEvidence:
    try:
        text = path.read_text(encoding="utf-8")
        digest = sha256_file(path)
        verified = all(quote in text for quote in quotes)
    except OSError:
        digest = ""
        verified = False
    return SourceMappingEvidence(
        path=str(path),
        sha256=digest,
        source_quotes=quotes,
        decision=decision,
        verified=verified,
    )
