"""Bounded-actor ALFWorld runtime adaptation for single-GPU GraphGPO runs.

The pinned upstream implementation creates ``env_num * group_n`` Ray actors,
with each actor owning one ``init_env(batch_size=1)`` environment.  That is the
right *logical* topology for GRPO groups, but it is not schedulable on the
single-GPU host when ``16 * 8`` scalar workers each reserve CPU and Python
runtime memory.

This module keeps the scalar environment and seed semantics intact, but places
one same-seed rollout group in one Ray actor.  It is installed only by the
explicit ``LH_GRAPHGPO_ALFWORLD_EXECUTION_MODE=grouped_scalar`` compatibility
mode; the immutable upstream mirror is never modified.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GroupedActorTopology",
    "grouped_actor_topology",
    "ScalarEnvironmentGroup",
    "RecipeWorkerGroup",
    "collect_sequential_rollout_chunks",
    "SequentialAlfWorldRolloutSource",
    "install_grouped_scalar_alfworld_runtime",
    "install_sequential_scalar_alfworld_runtime",
]


@dataclass(frozen=True)
class GroupedActorTopology:
    """Resource topology, separate from the logical rollout budget."""

    logical_train_rollouts: int
    raw_upstream_train_actor_count: int
    train_actor_count: int
    validation_actor_count: int
    worker_cpus: float
    task_runner_cpus: float
    minimum_ray_cpus: float


def grouped_actor_topology(
    *,
    train_batch_size: int,
    rollout_group_size: int,
    validation_batch_size: int,
    worker_cpus: float,
    task_runner_cpus: float,
) -> GroupedActorTopology:
    """Calculate scalar-group actor demand without changing sample count."""

    if train_batch_size < 1 or rollout_group_size < 1 or validation_batch_size < 1:
        raise ValueError("train batch, rollout group, and validation batch must be positive")
    if worker_cpus <= 0 or task_runner_cpus <= 0:
        raise ValueError("Ray CPU reservations must be positive")
    logical_train_rollouts = train_batch_size * rollout_group_size
    train_actor_count = train_batch_size
    validation_actor_count = validation_batch_size
    minimum_ray_cpus = task_runner_cpus + worker_cpus * (train_actor_count + validation_actor_count)
    return GroupedActorTopology(
        logical_train_rollouts=logical_train_rollouts,
        raw_upstream_train_actor_count=logical_train_rollouts,
        train_actor_count=train_actor_count,
        validation_actor_count=validation_actor_count,
        worker_cpus=worker_cpus,
        task_runner_cpus=task_runner_cpus,
        minimum_ray_cpus=minimum_ray_cpus,
    )


def _take_scalar(value: Any) -> Any:
    """Mirror the upstream parent, which unwraps each scalar actor's [0]."""

    try:
        return value[0]
    except (IndexError, KeyError, TypeError):
        return value


class ScalarEnvironmentGroup:
    """A same-seed collection of independent upstream scalar environments."""

    def __init__(self, base_env: Any, *, seed: int, group_size: int) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        # Upstream anchor: AlfworldWorker calls base_env.init_env(batch_size=1)
        # then env.seed(seed).  Repeat that exact scalar construction per branch
        # rather than using ALFWorld's vector batch mode, which samples different
        # tasks for the eight positions.
        self.environments = []
        for _ in range(group_size):
            environment = base_env.init_env(batch_size=1)
            environment.seed(seed)
            self.environments.append(environment)

    def reset(self) -> tuple[list[Any], list[dict[str, Any]]]:
        observations: list[Any] = []
        infos: list[dict[str, Any]] = []
        for environment in self.environments:
            observation, info = environment.reset()
            copied_info = dict(info)
            copied_info["observation_text"] = observation
            observations.append(_take_scalar(observation))
            infos.append({key: _take_scalar(value) for key, value in copied_info.items()})
        return observations, infos

    def step(
        self, actions: Sequence[str]
    ) -> tuple[list[Any], list[Any], list[Any], list[dict[str, Any]]]:
        if len(actions) != len(self.environments):
            raise ValueError(
                f"action count {len(actions)} does not match grouped scalar environments "
                f"{len(self.environments)}"
            )
        observations: list[Any] = []
        scores: list[Any] = []
        dones: list[Any] = []
        infos: list[dict[str, Any]] = []
        for environment, action in zip(self.environments, actions, strict=True):
            observation, score, done, info = environment.step([action])
            copied_info = dict(info)
            copied_info["observation_text"] = observation
            observations.append(_take_scalar(observation))
            scores.append(_take_scalar(score))
            dones.append(_take_scalar(done))
            infos.append({key: _take_scalar(value) for key, value in copied_info.items()})
        return observations, scores, dones, infos


class RecipeWorkerGroup:
    """One Ray actor containing source-faithful GraphGPO worker branches.

    GraphGPO does not import ``agent_system/.../alfworld/envs.py`` for ALFWorld
    training.  Its real path is ``recipe/GraphGPO/alfworld_envs.py`` and its
    ``AlfworldWorker`` tracks ALFWorld state in addition to calling the base
    environment.  Every branch here is therefore an instance of that source
    worker, not a replacement base environment implementation.
    """

    def __init__(
        self,
        *,
        worker_factory: Callable[[Mapping[str, Any], int, Any], Any],
        config: Mapping[str, Any],
        seed: int,
        base_env: Any,
        group_size: int,
    ) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        # Official recipe anchor: AlfworldEnvs creates workers with
        # seed + (i // group_n); every branch in a logical rollout group must
        # receive the same seed, while retaining the source worker class.
        self.workers = [worker_factory(config, seed, base_env) for _ in range(group_size)]

    def reset(self) -> tuple[list[Any], list[dict[str, Any]]]:
        observations: list[Any] = []
        infos: list[dict[str, Any]] = []
        for worker in self.workers:
            observation, info = worker.reset()
            observations.append(_take_scalar(observation))
            infos.append({key: _take_scalar(value) for key, value in info.items()})
        return observations, infos

    def step(
        self, actions: Sequence[str]
    ) -> tuple[list[Any], list[Any], list[Any], list[dict[str, Any]]]:
        if len(actions) != len(self.workers):
            raise ValueError(
                f"action count {len(actions)} does not match source worker count "
                f"{len(self.workers)}"
            )
        observations: list[Any] = []
        scores: list[Any] = []
        dones: list[Any] = []
        infos: list[dict[str, Any]] = []
        for worker, action in zip(self.workers, actions, strict=True):
            observation, score, done, info = worker.step(action)
            observations.append(_take_scalar(observation))
            scores.append(_take_scalar(score))
            dones.append(_take_scalar(done))
            infos.append({key: _take_scalar(value) for key, value in info.items()})
        return observations, scores, dones, infos

    def getobs(self) -> list[Any]:
        return [worker.getobs() for worker in self.workers]


def _concatenate_rollout_values(chunks: list[Any]) -> Any:
    """Concatenate source numpy arrays without forcing test doubles to numpy."""

    if not chunks:
        return []
    if isinstance(chunks[0], list):
        return [value for chunk in chunks for value in chunk]
    import numpy as np

    return np.concatenate(chunks, axis=0)


def collect_sequential_rollout_chunks(
    *,
    gen_batch: Any,
    actor_rollout_wg: Any,
    rollout_group_size: int,
    groups_per_chunk: int,
    open_chunk: Callable[[int, int], Any],
    vanilla_multi_turn_loop: Callable[[Any, Any, Any], tuple[Any, Any, Any, Any, Any, Any]],
    gather_rollout_data: Callable[..., Any],
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> Any:
    """Serialise environment placement, not logical GraphGPO rollout data.

    The source collector repeats a prompt batch with
    ``repeat(..., interleave=True)`` and calls ``gather_rollout_data`` once.
    This adapter executes those same operations in bounded chunks, preserves
    group-contiguous order, and defers the one gather until all chunks finish.
    The caller enters the official GraphGPO advantage and PPO code only after
    this function returns.
    """

    if rollout_group_size < 1 or groups_per_chunk < 1:
        raise ValueError("rollout_group_size and groups_per_chunk must be positive")
    prompt_count = len(gen_batch)
    if prompt_count < 1:
        raise ValueError("sequential rollout requires at least one prompt")

    total_batch_list: list[Any] = []
    reward_chunks: list[Any] = []
    length_chunks: list[Any] = []
    success_chunks: dict[str, list[Any]] = {}
    traj_chunks: list[Any] = []
    tool_calling_chunks: list[Any] = []

    for group_offset in range(0, prompt_count, groups_per_chunk):
        group_count = min(groups_per_chunk, prompt_count - group_offset)
        if progress_callback is not None:
            progress_callback(group_offset, group_count, prompt_count)
        source_batch = gen_batch[group_offset : group_offset + group_count]
        repeated_batch = source_batch.repeat(repeat_times=rollout_group_size, interleave=True)
        chunk_envs = open_chunk(group_offset, group_count)
        try:
            (
                chunk_batch_list,
                chunk_rewards,
                chunk_lengths,
                chunk_success,
                chunk_traj_uid,
                chunk_tool_callings,
            ) = vanilla_multi_turn_loop(repeated_batch, actor_rollout_wg, chunk_envs)
        finally:
            close = getattr(chunk_envs, "close", None)
            if not callable(close):
                raise RuntimeError("sequential rollout chunk does not expose close()")
            close()

        expected_chunk_rollouts = group_count * rollout_group_size
        if len(chunk_batch_list) != expected_chunk_rollouts:
            raise RuntimeError(
                "source rollout count changed during sequential collection: "
                f"expected {expected_chunk_rollouts}, got {len(chunk_batch_list)}"
            )
        total_batch_list.extend(chunk_batch_list)
        reward_chunks.append(chunk_rewards)
        length_chunks.append(chunk_lengths)
        traj_chunks.append(chunk_traj_uid)
        tool_calling_chunks.append(chunk_tool_callings)
        for key, value in chunk_success.items():
            # The source manager emits data-source-specific success keys only
            # for trajectories that carry that source.  Concatenate each key's
            # observed values independently, matching source gather's
            # ``np.mean(value)`` rather than inventing zero-valued samples.
            success_chunks.setdefault(key, []).append(value)

    expected_rollouts = prompt_count * rollout_group_size
    if len(total_batch_list) != expected_rollouts:
        raise RuntimeError(
            f"sequential rollout produced {len(total_batch_list)} trajectories; "
            f"expected {expected_rollouts}"
        )
    return gather_rollout_data(
        total_batch_list=total_batch_list,
        episode_rewards=_concatenate_rollout_values(reward_chunks),
        episode_lengths=_concatenate_rollout_values(length_chunks),
        success={
            key: _concatenate_rollout_values(chunks) for key, chunks in success_chunks.items()
        },
        traj_uid=_concatenate_rollout_values(traj_chunks),
        tool_callings=_concatenate_rollout_values(tool_calling_chunks),
    )


class _ClosableEnvironmentManager:
    """Delegate the source manager while making its Ray-worker lifetime explicit."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def close(self) -> None:
        self._manager.envs.close()


class SequentialAlfWorldRolloutSource:
    """Lazy source environment factory for one bounded rollout microbatch."""

    def __init__(
        self,
        *,
        alf_config_path: str,
        seed: int,
        train_batch_size: int,
        rollout_group_size: int,
        resources_per_worker: Mapping[str, Any],
        env_kwargs: Mapping[str, Any],
        build_envs: Callable[..., Any],
        manager_type: Callable[[Any, Any, Any], Any],
        projection_f: Any,
        config: Any,
        is_train: bool,
    ) -> None:
        self.alf_config_path = alf_config_path
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.rollout_group_size = rollout_group_size
        self.resources_per_worker = resources_per_worker
        self.env_kwargs = env_kwargs
        self._build_envs = build_envs
        self._manager_type = manager_type
        self._projection_f = projection_f
        self._config = config
        self.is_train = is_train

    def open_chunk(self, group_offset: int, group_count: int) -> _ClosableEnvironmentManager:
        if group_offset < 0 or group_count < 1:
            raise ValueError("invalid sequential rollout chunk")
        if group_offset + group_count > self.train_batch_size:
            raise ValueError("sequential rollout chunk exceeds configured train batch")
        # Source seed expression is seed + (i // group_n).  At group granularity
        # the corresponding first index is group_offset * group_n, hence the
        # source-equivalent group seed is seed + group_offset.
        raw_envs = self._build_envs(
            self.alf_config_path,
            self.seed + group_offset,
            group_count,
            self.rollout_group_size,
            self.resources_per_worker,
            is_train=self.is_train,
            env_kwargs=dict(self.env_kwargs),
        )
        return _ClosableEnvironmentManager(
            self._manager_type(raw_envs, self._projection_f, self._config)
        )


def install_grouped_scalar_alfworld_runtime() -> bool:
    """Patch GraphGPO's recipe ALFWorld module, not its duplicate package path."""

    import ray
    import recipe.GraphGPO.alfworld_envs as upstream_envs

    if getattr(upstream_envs, "_lh_grouped_scalar_runtime", False):
        return False

    class GroupedRecipeAlfworldWorker:
        """One Ray actor holding ``group_n`` original recipe workers."""

        def __init__(self, config, seed, base_env, group_size):
            self.group = RecipeWorkerGroup(
                worker_factory=upstream_envs.AlfworldWorker,
                config=config,
                seed=seed,
                base_env=base_env,
                group_size=group_size,
            )

        def reset(self):
            return self.group.reset()

        def step(self, actions):
            return self.group.step(actions)

        def getobs(self):
            return self.group.getobs()

    class GroupedAlfworldEnvs(upstream_envs.gym.Env):
        def __init__(
            self,
            alf_config_path,
            seed,
            env_num,
            group_n,
            resources_per_worker,
            is_train=True,
            env_kwargs=None,
        ):
            super().__init__()
            if not ray.is_initialized():
                ray.init()
            env_kwargs = {} if env_kwargs is None else env_kwargs
            eval_dataset = env_kwargs.get("eval_dataset", "eval_in_distribution")
            config = upstream_envs.load_config_file(alf_config_path)
            env_type = config["env"]["type"]
            base_env = upstream_envs.get_environment(env_type)(
                config, train_eval="train" if is_train else eval_dataset
            )
            self.multi_modal = env_type == "AlfredThorEnv"
            self.num_processes = env_num * group_n
            self.group_n = group_n
            self.worker_count = env_num
            worker_type = ray.remote(**resources_per_worker)(GroupedRecipeAlfworldWorker)
            # The original recipe has one actor per branch.  Retain its source
            # worker and its seed rule, changing only where the eight branches
            # are placed.
            self.workers = [
                worker_type.remote(config, seed + group_index, base_env, group_n)
                for group_index in range(env_num)
            ]
            self.prev_admissible_commands = [None for _ in range(self.num_processes)]

        def step(self, actions):
            if len(actions) != self.num_processes:
                raise AssertionError("The num of actions must be equal to the num of processes")
            futures = [
                worker.step.remote(actions[index * self.group_n : (index + 1) * self.group_n])
                for index, worker in enumerate(self.workers)
            ]
            results = ray.get(futures)
            text_obs_list, rewards_list, dones_list, info_list = [], [], [], []
            logical_index = 0
            for observations, _scores, dones, infos in results:
                for observation, done, info in zip(observations, dones, infos, strict=True):
                    text_obs_list.append(observation)
                    dones_list.append(done)
                    info_list.append(info)
                    self.prev_admissible_commands[logical_index] = info["admissible_commands"]
                    rewards_list.append(upstream_envs.compute_reward(info, self.multi_modal))
                    logical_index += 1
            return text_obs_list, None, rewards_list, dones_list, info_list

        def reset(self):
            results = ray.get([worker.reset.remote() for worker in self.workers])
            text_obs_list, info_list = [], []
            logical_index = 0
            for observations, infos in results:
                for observation, info in zip(observations, infos, strict=True):
                    text_obs_list.append(observation)
                    info_list.append(info)
                    self.prev_admissible_commands[logical_index] = info["admissible_commands"]
                    logical_index += 1
            return text_obs_list, None, info_list

        def getobs(self):
            return [
                image
                for images in ray.get([worker.getobs.remote() for worker in self.workers])
                for image in images
            ]

        @property
        def get_admissible_commands(self):
            return self.prev_admissible_commands

        def close(self):
            for worker in self.workers:
                ray.kill(worker)

    def build_grouped_alfworld_envs(
        alf_config_path,
        seed,
        env_num,
        group_n,
        resources_per_worker,
        is_train=True,
        env_kwargs=None,
    ):
        return GroupedAlfworldEnvs(
            alf_config_path,
            seed,
            env_num,
            group_n,
            resources_per_worker,
            is_train,
            env_kwargs,
        )

    upstream_envs.AlfworldEnvs = GroupedAlfworldEnvs
    upstream_envs.build_alfworld_envs = build_grouped_alfworld_envs
    upstream_envs._lh_grouped_scalar_runtime = True
    return True


def _make_sequential_alfworld_envs(config: Any) -> tuple[Any, Any]:
    """Migrate only GraphGPO's ALFWorld branch of source ``make_envs``."""

    import os
    from functools import partial

    import agent_system.environments.env_package.alfworld as alfworld_package
    from agent_system.environments.env_package.alfworld import alfworld_projection
    from omegaconf import OmegaConf
    from recipe.GraphGPO.alfworld_envs import build_alfworld_envs
    from recipe.GraphGPO.graphgpo_env_manager import AlfWorldEnvironmentManager

    if config.env.env_name not in {"alfworld/AlfredThorEnv", "alfworld/AlfredTWEnv"}:
        raise ValueError(f"Unsupported environment: {config.env.env_name}")
    alfworld_dir = os.path.dirname(os.path.abspath(alfworld_package.__file__))
    alf_config_path = os.path.join(alfworld_dir, "configs/config_tw.yaml")
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    env_kwargs = {"eval_dataset": "eval_in_distribution"}
    projection_f = partial(alfworld_projection)

    train_envs = SequentialAlfWorldRolloutSource(
        alf_config_path=alf_config_path,
        seed=int(config.env.seed),
        train_batch_size=int(config.data.train_batch_size),
        rollout_group_size=int(config.env.rollout.n),
        resources_per_worker=resources_per_worker,
        env_kwargs=env_kwargs,
        build_envs=build_alfworld_envs,
        manager_type=AlfWorldEnvironmentManager,
        projection_f=projection_f,
        config=config,
        is_train=True,
    )
    # Source validation creates one scalar environment per row with seed + i.
    # Keep that logical topology, but defer its placement to the collector so
    # the 128-row official validation set does not create 128 persistent actors.
    val_envs = SequentialAlfWorldRolloutSource(
        alf_config_path=alf_config_path,
        seed=int(config.env.seed) + 1000,
        train_batch_size=int(config.data.val_batch_size),
        rollout_group_size=1,
        resources_per_worker=resources_per_worker,
        env_kwargs=env_kwargs,
        build_envs=build_alfworld_envs,
        manager_type=AlfWorldEnvironmentManager,
        projection_f=projection_f,
        config=config,
        is_train=False,
    )
    return train_envs, val_envs


def _sequential_groups_per_chunk(config: Any, *, is_train: bool) -> int:
    rollout = config.env.rollout
    key = "sequential_groups_per_chunk" if is_train else "sequential_validation_groups_per_chunk"
    value = rollout.get(key, 1)
    value = int(value)
    if value < 1:
        raise ValueError(f"env.rollout.{key} must be positive")
    return value


def install_sequential_scalar_alfworld_runtime() -> bool:
    """Install the bounded-memory adapter at GraphGPO's real recipe seam."""

    install_grouped_scalar_alfworld_runtime()
    import recipe.GraphGPO.graphgpo_env_manager as graphgpo_env_manager
    import recipe.GraphGPO.rollout_loop as rollout_loop

    if getattr(rollout_loop, "_lh_sequential_scalar_runtime", False):
        return False
    source_make_envs = graphgpo_env_manager.make_envs
    source_multi_turn_loop = rollout_loop.TrajectoryCollector.multi_turn_loop

    def make_envs(config):
        if "alfworld" not in str(config.env.env_name).lower():
            return source_make_envs(config)
        return _make_sequential_alfworld_envs(config)

    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        if not isinstance(envs, SequentialAlfWorldRolloutSource):
            return source_multi_turn_loop(
                self, gen_batch, actor_rollout_wg, envs, is_train=is_train
            )
        if is_train and self.config.algorithm.filter_groups.enable:
            raise RuntimeError(
                "sequential_scalar rollout is incompatible with dynamic group filtering"
            )
        if len(gen_batch) != envs.train_batch_size:
            raise RuntimeError(
                "GraphGPO prompt batch does not match configured sequential train batch"
            )

        phase = "train" if is_train else "validation"

        def progress(group_offset: int, group_count: int, total_groups: int) -> None:
            print(
                "lh_rollout_chunk "
                f"phase={phase} expected={total_groups} completed={group_offset} "
                f"current={group_count} rollout_group={envs.rollout_group_size}"
            )

        return collect_sequential_rollout_chunks(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            rollout_group_size=envs.rollout_group_size,
            groups_per_chunk=_sequential_groups_per_chunk(self.config, is_train=is_train),
            open_chunk=envs.open_chunk,
            vanilla_multi_turn_loop=self.vanilla_multi_turn_loop,
            gather_rollout_data=self.gather_rollout_data,
            progress_callback=progress,
        )

    graphgpo_env_manager.make_envs = make_envs
    rollout_loop.TrajectoryCollector.multi_turn_loop = multi_turn_loop
    rollout_loop._lh_sequential_scalar_runtime = True
    return True
