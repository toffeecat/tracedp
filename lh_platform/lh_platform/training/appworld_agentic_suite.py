"""Native AppWorld train/dev task suite with evaluator and restore evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import yaml

from lh_platform.training.agentic_rl_components import builtin_component
from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    AgenticRolloutBatch,
    EvaluationRequest,
    EvaluationResult,
    PolicyActionRequest,
    RolloutRequest,
    TrainablePolicy,
    canonical_sha256,
    token_ids_sha256,
)
from lh_platform.training.contracts import TrainingTrajectoryRecord
from lh_platform.training.model_io import (
    AssistantTarget,
    ConversationIR,
    ModelIOContractError,
    ModelIOProfile,
    ResponseContractDescriptor,
)
from lh_platform.training.subproblems.appworld import (
    AppWorldSubproblemCompiler,
    AppWorldTaskContract,
)
from lh_platform.training.subproblems.base import GoalVector, project_actor_awareness


class AppWorldSuiteError(RuntimeError):
    pass


APPWORLD_PROMPT_INTERVENTIONS = {
    "none": "",
    "structured_task_plan": (
        "Planning discipline: begin each code block with comments listing the residual "
        "task subgoals and choose exactly the next useful information-gathering or mutation "
        "step. Re-evaluate the residual subgoals from the visible history before acting."
    ),
    "api_argument_grounding": (
        "Argument-binding discipline: never guess an API argument, object ID, account, or "
        "field value. First query or list the relevant records and print them; in a later "
        "fresh-shell step, copy exact values from visible output into the mutating call."
    ),
    "documentation_first_recipe": (
        "Documentation discipline: when the exact API signature is absent from visible "
        "history, make this step documentation-only, print the official API document, and "
        "defer mutation to a later step. Do not mix a guessed mutation with a docs query."
    ),
    "task_plan_plus_argument_grounding": (
        "Planning and binding discipline: begin with comments listing residual subgoals; "
        "choose one next step; never guess identifiers or arguments; print exact records "
        "first and use only values visible in later history for mutations."
    ),
}


def render_appworld_actor_prompt(
    *,
    instruction: str,
    actor_context: Mapping[str, Any],
    history: list[Mapping[str, str]] | tuple[Mapping[str, str], ...],
    history_length: int,
    history_value_max_chars: int = 1536,
    prompt_intervention_id: str = "none",
) -> str:
    """Render the normal actor prompt shared by rollout and train-only SFT.

    The function intentionally has no GoalGraph/residual-state argument.  A
    competence warm-start therefore cannot accidentally inherit the oracle
    awareness extension used by a separate factorial experiment.
    """

    if history_length < 0:
        raise AppWorldSuiteError("AppWorld prompt history length is invalid")
    if history_value_max_chars < 256:
        raise AppWorldSuiteError("AppWorld history value budget is invalid")
    if prompt_intervention_id not in APPWORLD_PROMPT_INTERVENTIONS:
        raise AppWorldSuiteError("AppWorld prompt intervention is unsupported")
    visible_history = (
        [
            {
                "code": _compact_history_value(
                    str(item.get("code", "")),
                    field="code",
                    max_chars=history_value_max_chars,
                ),
                "output": _compact_history_value(
                    str(item.get("output", "")),
                    field="output",
                    max_chars=history_value_max_chars,
                ),
            }
            for item in history[-history_length:]
        ]
        if history_length
        else []
    )
    intervention = APPWORLD_PROMPT_INTERVENTIONS[prompt_intervention_id]
    intervention_block = f"\nPrompt intervention: {intervention}" if intervention else ""
    return (
        APPWORLD_PYTHON_FENCE_CONTRACT.render_instructions() + "\n\n"
        f"Task: {instruction}\n"
        f"Actor context: {json.dumps(dict(actor_context), sort_keys=True)}\n"
        f"Recent code/output history: {json.dumps(visible_history, sort_keys=True)}"
        f"{intervention_block}"
    )


def _compact_history_value(value: str, *, field: str, max_chars: int) -> str:
    """Bound history fields while retaining both schema headers and error tails."""

    if len(value) <= max_chars:
        return value
    marker = (
        f"<APPWORLD_HISTORY_COMPACTED field={field} original_chars={len(value)} "
        f"sha256={hashlib.sha256(value.encode('utf-8')).hexdigest()}>"
    )
    payload_budget = max_chars - len(marker) - 2
    if payload_budget < 2:
        raise AppWorldSuiteError("AppWorld history output budget cannot fit receipt")
    head_chars = (payload_budget * 3) // 4
    tail_chars = payload_budget - head_chars
    return f"{value[:head_chars]}\n{marker}\n{value[-tail_chars:]}"


@dataclass(frozen=True)
class ParsedAppWorldAction:
    code: str
    valid: bool
    error_code: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class AppWorldPythonFenceContract:
    """Own AppWorld's single self-contained Python-block response language."""

    descriptor: ResponseContractDescriptor = ResponseContractDescriptor(
        contract_id="appworld_single_python_fence/v1",
        reasoning_requirement="none",
        serializer_id="appworld_python_fence_serializer_v1",
        parser_id="appworld_python_fence_parser_v1",
        stop_texts=("\n```",),
    )

    @property
    def id(self) -> str:
        return self.descriptor.contract_id

    @property
    def stop_texts(self) -> tuple[str, ...]:
        return self.descriptor.stop_texts

    def render_instructions(self) -> str:
        return (
            "You are an AppWorld code agent. Return exactly one Python code block. "
            "Use the provided `apis` object. When an API name or parameter is "
            "uncertain, make the current step a public documentation query using "
            "`apis.api_docs.show_api_descriptions(app_name=...)` or "
            "`apis.api_docs.show_api_doc(app_name=..., api_name=...)`, and print "
            "its result before acting in a later step.\n\n"
            "Each code block runs in a fresh Python shell; do not rely on variables "
            "from an earlier step. End immediately after the closing code fence; "
            "do not add prose, JSON, or another code block."
        )

    def serialize_supervision(self, semantic_action: object) -> AssistantTarget:
        code = str(semantic_action).strip()
        if not code:
            raise ModelIOContractError("AppWorld supervision code is empty")
        try:
            compile(code, "<appworld-supervision>", "exec")
        except SyntaxError as error:
            raise ModelIOContractError(
                f"AppWorld supervision code has invalid Python syntax: {error.msg}"
            ) from error
        return AssistantTarget(f"```python\n{code}\n```")

    def parse_response(self, response_text: str, **_context: object) -> ParsedAppWorldAction:
        if str(response_text).count("```") != 2:
            return ParsedAppWorldAction(
                code="",
                valid=False,
                error_code="invalid_fence_structure",
                error="response must contain exactly one Python code block and no prose",
            )
        match = re.fullmatch(
            r"\s*```python[ \t]*\r?\n(?P<code>.*?)\r?\n?```\s*",
            str(response_text),
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is None:
            return ParsedAppWorldAction(
                code="",
                valid=False,
                error_code="invalid_fence_structure",
                error="response must contain exactly one Python code block and no prose",
            )
        code = match.group("code").strip()
        if not code:
            return ParsedAppWorldAction(
                code="",
                valid=False,
                error_code="empty_python_block",
                error="Python code block is empty",
            )
        try:
            compile(code, "<appworld-action>", "exec")
        except SyntaxError as error:
            return ParsedAppWorldAction(
                code=code,
                valid=False,
                error_code="python_syntax_error",
                error=f"Python syntax error: {error.msg}",
            )
        return ParsedAppWorldAction(code=code, valid=True)

    def validate_model_profile(self, profile: ModelIOProfile) -> None:
        if profile.selected_reasoning_mode != "off":
            raise ModelIOContractError("AppWorld Python-fence response requires reasoning mode off")


APPWORLD_PYTHON_FENCE_CONTRACT = AppWorldPythonFenceContract()


@dataclass(frozen=True)
class AppWorldSuitePairDocument:
    """Validated portable contract for one native AppWorld train/dev pair."""

    config_path: Path
    source_revision: str
    package_version: str
    dataset_version: str
    train_split: str
    eval_split: str
    sha256: str


def load_appworld_suite_pair_document(
    config_path: str | Path,
) -> AppWorldSuitePairDocument:
    """Validate the source, split, evaluator, and claim boundary before import."""

    path = Path(config_path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AppWorldSuiteError(f"cannot load AppWorld suite config {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise AppWorldSuiteError("AppWorld suite config must be a YAML mapping")
    _expect_keys(
        raw,
        {
            "schema",
            "id",
            "official_source",
            "train_suite",
            "eval_suite",
            "forbidden_splits",
            "runtime",
            "goal_compiler",
            "claim_boundary",
        },
        "AppWorld suite",
    )
    if raw.get("schema") != "lh_appworld_native_suite_pair/v1":
        raise AppWorldSuiteError("unsupported AppWorld suite schema")
    source = _required_mapping(raw, "official_source")
    _expect_keys(
        source,
        {"repository", "revision", "package_version", "dataset_version"},
        "AppWorld official_source",
    )
    if source.get("repository") != "https://github.com/StonyBrookNLP/appworld":
        raise AppWorldSuiteError("AppWorld suite must bind the official repository")
    revision = str(source.get("revision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise AppWorldSuiteError("AppWorld source revision must be a full commit SHA")
    package_version = str(source.get("package_version", ""))
    dataset_version = str(source.get("dataset_version", ""))
    if not package_version or not dataset_version:
        raise AppWorldSuiteError("AppWorld package and dataset versions must be pinned")

    train = _required_mapping(raw, "train_suite")
    evaluation = _required_mapping(raw, "eval_suite")
    for name, suite, expected_split in (
        ("train_suite", train, "train"),
        ("eval_suite", evaluation, "dev"),
    ):
        _expect_keys(suite, {"split", "ground_truth_mode"}, f"AppWorld {name}")
        if suite.get("split") != expected_split or suite.get("ground_truth_mode") != "minimal":
            raise AppWorldSuiteError(
                f"AppWorld {name} must use minimal {expected_split} ground truth"
            )
    if raw.get("forbidden_splits") != ["test_normal", "test_challenge"]:
        raise AppWorldSuiteError("AppWorld test splits must remain forbidden")
    if dict(_required_mapping(raw, "runtime")) != {
        "action_type": "python_code",
        "evaluator": "official_test_tracker",
        "checkpoint": "official_save_state_load_state_replay",
        "missing_package_or_data": "fail_closed",
    }:
        raise AppWorldSuiteError("AppWorld runtime differs from the native verified contract")
    if dict(_required_mapping(raw, "goal_compiler")) != {
        "implementation": "appworld_official_evaluator_goal_graph_v1",
        "solution_code_allowed": False,
        "api_call_trace_allowed": False,
    }:
        raise AppWorldSuiteError("AppWorld goal compiler enables privileged solution data")
    if dict(_required_mapping(raw, "claim_boundary")) != {
        "oracle_upper_bound": True,
        "intervention_verified_parent_child_edges": False,
        "official_leaderboard_comparison": False,
    }:
        raise AppWorldSuiteError("AppWorld claim boundary is unsafe")
    return AppWorldSuitePairDocument(
        config_path=path,
        source_revision=revision,
        package_version=package_version,
        dataset_version=dataset_version,
        train_split="train",
        eval_split="dev",
        sha256=canonical_sha256(dict(raw)),
    )


@dataclass(frozen=True)
class AppWorldSuiteConfig:
    role: str
    split: str
    artifacts_dir: str
    expected_package_version: str
    expected_dataset_version: str
    expected_source_revision: str
    tasks_per_update: int = 1
    eval_task_count: int = 1
    history_length: int = 5
    history_value_max_chars: int = 1536
    train_temperature: float = 1.0
    eval_temperature: float = 0.4
    actor_awareness_mode: str = "none"
    model_io_profile_id: str | None = None
    prompt_intervention_id: str = "none"
    action_candidates_per_step: int = 1
    candidate_selection_policy: str = "first"

    def __post_init__(self) -> None:
        if self.role not in {"train_task_suite", "eval_task_suite"}:
            raise AgenticRLContractError("AppWorld suite role must be train or eval")
        if self.split not in {"train", "dev"}:
            raise AgenticRLContractError("AppWorld DPPO study only permits train/dev")
        if self.role == "train_task_suite" and self.split != "train":
            raise AgenticRLContractError("AppWorld training suite must use train split")
        if self.role == "eval_task_suite" and self.split != "dev":
            raise AgenticRLContractError("AppWorld evaluation suite must use dev split")
        if not (
            self.expected_package_version
            and self.expected_dataset_version
            and self.expected_source_revision
        ):
            raise AgenticRLContractError(
                "AppWorld source, package, and dataset versions must be locked"
            )
        if (
            self.tasks_per_update < 1
            or self.eval_task_count < 1
            or self.history_length < 0
            or self.history_value_max_chars < 256
        ):
            raise AgenticRLContractError("AppWorld task and history budgets must be valid")
        if self.actor_awareness_mode not in {"none", "oracle_goal_graph"}:
            raise AgenticRLContractError("unsupported AppWorld actor awareness mode")
        if self.prompt_intervention_id not in APPWORLD_PROMPT_INTERVENTIONS:
            raise AgenticRLContractError("unsupported AppWorld prompt intervention")
        if self.action_candidates_per_step < 1:
            raise AgenticRLContractError("AppWorld action-candidate count must be positive")
        if self.candidate_selection_policy not in {
            "first",
            "valid_mutating_api_then_valid_api_then_valid_python",
        }:
            raise AgenticRLContractError("unsupported AppWorld candidate selection policy")


class AppWorldSession(Protocol):
    contract: AppWorldTaskContract
    actor_context: Mapping[str, Any]

    def evaluate(self) -> Mapping[str, Any]: ...
    def execute(self, code: str) -> str: ...
    def save_state(self, state_id: str) -> str: ...
    def load_state(self, state_id: str) -> None: ...
    def checkpoint_sha256(self, state_id: str) -> str: ...
    def checkpoint_locator(self, state_id: str) -> str: ...
    def task_completed(self) -> bool: ...
    def canonical_action_schemas(self, code: str) -> tuple[Mapping[str, Any], ...]: ...
    def close(self) -> None: ...


class AppWorldBackend(Protocol):
    package_version: str
    dataset_version: str
    source_revision: str

    def task_ids(self, split: str) -> tuple[str, ...]: ...
    def open_task(
        self,
        *,
        task_id: str,
        split: str,
        experiment_name: str,
        seed: int,
    ) -> AppWorldSession: ...


class AppWorldNativeTaskSuite:
    def __init__(self, config: AppWorldSuiteConfig, *, backend: AppWorldBackend) -> None:
        self.config = config
        self.backend = backend
        if backend.package_version != config.expected_package_version:
            raise AppWorldSuiteError(
                "AppWorld package version drift: "
                f"expected {config.expected_package_version}, got {backend.package_version}"
            )
        if backend.dataset_version != config.expected_dataset_version:
            raise AppWorldSuiteError(
                "AppWorld dataset version drift: "
                f"expected {config.expected_dataset_version}, got {backend.dataset_version}"
            )
        if backend.source_revision != config.expected_source_revision:
            raise AppWorldSuiteError(
                "AppWorld source revision drift: "
                f"expected {config.expected_source_revision}, got {backend.source_revision}"
            )
        self.task_ids = tuple(backend.task_ids(config.split))
        if not self.task_ids or len(self.task_ids) != len(set(self.task_ids)):
            raise AppWorldSuiteError("AppWorld split must contain unique task IDs")
        self.manifest_sha256 = canonical_sha256(
            {
                "schema": "lh_appworld_manifest/v1",
                "package_version": backend.package_version,
                "dataset_version": backend.dataset_version,
                "source_revision": backend.source_revision,
                "split": config.split,
                "task_ids": list(self.task_ids),
            }
        )
        self.dataset_content_sha256 = self.manifest_sha256
        self._artifacts_dir = Path(config.artifacts_dir).resolve()
        self._compiler = AppWorldSubproblemCompiler()
        self.component = builtin_component(
            role=config.role,
            name="appworld_native",
            config={
                "package_version": backend.package_version,
                "dataset_version": backend.dataset_version,
                "source_revision": backend.source_revision,
                "split": config.split,
                "manifest_sha256": self.manifest_sha256,
                "actor_awareness_mode": config.actor_awareness_mode,
                "model_io_profile_id": config.model_io_profile_id,
                "history_length": config.history_length,
                "history_value_max_chars": config.history_value_max_chars,
                "prompt_intervention_id": config.prompt_intervention_id,
                "action_candidates_per_step": config.action_candidates_per_step,
                "candidate_selection_policy": config.candidate_selection_policy,
                "response_contract_id": APPWORLD_PYTHON_FENCE_CONTRACT.id,
                "response_contract_sha256": (APPWORLD_PYTHON_FENCE_CONTRACT.descriptor.sha256),
                "checkpointing": "official_save_state_load_state_replay",
                "evaluator": "official_test_tracker",
            },
        )

    def prewarm(self, *, seeds: tuple[int, ...]) -> tuple[str, ...]:
        """Cold-load selected official tasks before a GPU policy is allocated."""

        if not seeds:
            raise AppWorldSuiteError("AppWorld prewarm seeds are empty")
        selected = tuple(
            dict.fromkeys(
                task_id
                for seed in seeds
                for task_id in self._rotated(
                    seed,
                    self.config.tasks_per_update
                    if self.config.role == "train_task_suite"
                    else self.config.eval_task_count,
                )
            )
        )
        for index, task_id in enumerate(selected):
            session = self.backend.open_task(
                task_id=task_id,
                split=self.config.split,
                experiment_name=f"lh-prewarm-{uuid.uuid4().hex}",
                seed=seeds[index % len(seeds)],
            )
            try:
                session.evaluate()
            finally:
                session.close()
        return selected

    def collect(self, policy: TrainablePolicy, request: RolloutRequest) -> AgenticRolloutBatch:
        if self.config.role != "train_task_suite":
            raise AppWorldSuiteError("evaluation suite cannot collect training rollouts")
        selected = self._rotated(request.seed, self.config.tasks_per_update)
        cursor = policy.sampling_cursor()
        records: list[TrainingTrajectoryRecord] = []
        for task_index, task_id in enumerate(selected):
            group_uid = f"appworld:{task_id}:u{request.update_index}:s{request.seed}"
            for branch_index in range(request.group_size):
                records.extend(
                    self._run_episode(
                        policy,
                        task_id=task_id,
                        traj_uid=f"{group_uid}:r{branch_index}",
                        group_uid=group_uid,
                        environment_seed=request.seed + task_index,
                        policy_seed=(request.seed + task_index * request.group_size + branch_index),
                        max_steps=request.max_episode_steps,
                        rollout_config_sha256=request.rollout_config_sha256,
                        temperature=self.config.train_temperature,
                    )
                )
        events_by_key: dict[tuple[str, int], list[Any]] = {}
        for sample in policy.sampling_events_since(cursor):
            events_by_key.setdefault((sample.traj_uid, sample.step_idx), []).append(sample)
        samples = tuple(
            events_by_key[(row.traj_uid, row.step_idx)][
                int(row.extra.get("selected_candidate_index", 0))
            ]
            for row in records
        )
        return AgenticRolloutBatch(
            records=tuple(records),
            policy_samples=samples,
            task_manifest_sha256=self.manifest_sha256,
            policy_checkpoint_sha256=policy.checkpoint_sha256,
        ).validate()

    def collect_frozen(
        self,
        policy: TrainablePolicy,
        request: RolloutRequest,
    ) -> AgenticRolloutBatch:
        """Collect dev rollouts for a frozen diagnostic, never for an update."""

        if self.config.role != "eval_task_suite":
            raise AppWorldSuiteError("only the evaluation suite can collect held-out data")
        selected = self._rotated(request.seed, self.config.eval_task_count)
        cursor = policy.sampling_cursor()
        records: list[TrainingTrajectoryRecord] = []
        for task_index, task_id in enumerate(selected):
            group_uid = f"appworld-heldout:{task_id}:b{request.update_index}:s{request.seed}"
            for branch_index in range(request.group_size):
                records.extend(
                    self._run_episode(
                        policy,
                        task_id=task_id,
                        traj_uid=f"{group_uid}:r{branch_index}",
                        group_uid=group_uid,
                        environment_seed=request.seed + task_index,
                        policy_seed=(request.seed + task_index * request.group_size + branch_index),
                        max_steps=request.max_episode_steps,
                        rollout_config_sha256=request.rollout_config_sha256,
                        temperature=self.config.eval_temperature,
                    )
                )
        events_by_key: dict[tuple[str, int], list[Any]] = {}
        for sample in policy.sampling_events_since(cursor):
            events_by_key.setdefault((sample.traj_uid, sample.step_idx), []).append(sample)
        samples = tuple(
            events_by_key[(row.traj_uid, row.step_idx)][
                int(row.extra.get("selected_candidate_index", 0))
            ]
            for row in records
        )
        return AgenticRolloutBatch(
            records=tuple(records),
            policy_samples=samples,
            task_manifest_sha256=self.manifest_sha256,
            policy_checkpoint_sha256=policy.checkpoint_sha256,
        ).validate()

    def evaluate(self, policy: TrainablePolicy, request: EvaluationRequest) -> EvaluationResult:
        if self.config.role != "eval_task_suite":
            raise AppWorldSuiteError("training suite cannot evaluate dev tasks")
        selected = self._rotated(request.seed, self.config.eval_task_count)
        successes = 0
        lengths: list[int] = []
        for index, task_id in enumerate(selected):
            rows = self._run_episode(
                policy,
                task_id=task_id,
                traj_uid=f"eval:{task_id}:{index}",
                group_uid=f"eval:{task_id}",
                environment_seed=request.seed + index,
                policy_seed=request.seed + index,
                max_steps=request.max_episode_steps,
                rollout_config_sha256=canonical_sha256(
                    {"evaluation": True, "max_steps": request.max_episode_steps}
                ),
                temperature=self.config.eval_temperature,
            )
            successes += int(rows[-1].success)
            lengths.append(len(rows))
        return EvaluationResult(
            task_suite=self.component,
            task_manifest_sha256=self.manifest_sha256,
            policy_checkpoint_sha256=policy.checkpoint_sha256,
            task_count=len(selected),
            success_count=successes,
            mean_action_steps=sum(lengths) / len(lengths),
            success_by_horizon={"appworld": {"successes": successes, "tasks": len(selected)}},
            sampling_temperature=self.config.eval_temperature,
            episodes_per_task=1,
            stochastic_sampling=self.config.eval_temperature > 0.0,
            sampling_uncertainty_interval_available=False,
        )

    def _run_episode(
        self,
        policy: TrainablePolicy,
        *,
        task_id: str,
        traj_uid: str,
        group_uid: str,
        environment_seed: int,
        policy_seed: int,
        max_steps: int,
        rollout_config_sha256: str,
        temperature: float,
    ) -> list[TrainingTrajectoryRecord]:
        session = self.backend.open_task(
            task_id=task_id,
            split=self.config.split,
            experiment_name=f"lh-{uuid.uuid4().hex}",
            seed=environment_seed,
        )
        history: list[Mapping[str, str]] = []
        records: list[TrainingTrajectoryRecord] = []
        try:
            graph = self._compiler.compile_task(session.contract)
            evaluator = dict(session.evaluate())
            vector = self._compiler.project_state(session.contract, evaluator)
            terminal_reason = "max_episode_steps"
            won = False
            for step_index in range(max_steps):
                remaining = max_steps - step_index
                prompt = self._prompt(session, history, graph, vector, remaining)
                request = PolicyActionRequest(
                    traj_uid=traj_uid,
                    step_idx=step_index,
                    conversation=ConversationIR.from_legacy_user_prompt(prompt),
                    model_io_profile_id=self.config.model_io_profile_id,
                    response_contract_id=APPWORLD_PYTHON_FENCE_CONTRACT.id,
                    response_contract_sha256=(APPWORLD_PYTHON_FENCE_CONTRACT.descriptor.sha256),
                    admissible_actions=(),
                    sampling_temperature=temperature,
                    response_stop_texts=APPWORLD_PYTHON_FENCE_CONTRACT.stop_texts,
                )
                candidates = policy.sample_actions(
                    (request,) * self.config.action_candidates_per_step,
                    seed=policy_seed + step_index * 100_000,
                )
                if len(candidates) != self.config.action_candidates_per_step or any(
                    (candidate.traj_uid, candidate.step_idx) != (traj_uid, step_index)
                    for candidate in candidates
                ):
                    raise AppWorldSuiteError("policy sample key differs from AppWorld request")
                candidate_audit = []
                for candidate_index, candidate in enumerate(candidates):
                    candidate_parsed = APPWORLD_PYTHON_FENCE_CONTRACT.parse_response(
                        candidate.action_text
                    )
                    candidate_code = candidate_parsed.code if candidate_parsed.valid else "pass"
                    candidate_schemas = tuple(
                        getattr(session, "canonical_action_schemas", lambda _code: ())(
                            candidate_code
                        )
                    )
                    candidate_audit.append(
                        {
                            "candidate_index": candidate_index,
                            "response_sha256": hashlib.sha256(
                                candidate.action_text.encode("utf-8")
                            ).hexdigest(),
                            "python_valid": candidate_parsed.valid,
                            "official_api_schema_count": len(candidate_schemas),
                            "mutating_api_schema_count": sum(
                                schema.get("side_effect_type") not in {None, "read"}
                                for schema in candidate_schemas
                            ),
                        }
                    )
                selected_candidate_index = 0
                if (
                    self.config.candidate_selection_policy
                    == "valid_mutating_api_then_valid_api_then_valid_python"
                ):
                    selected_candidate_index = max(
                        range(len(candidates)),
                        key=lambda index: (
                            bool(candidate_audit[index]["python_valid"]),
                            bool(candidate_audit[index]["mutating_api_schema_count"]),
                            bool(candidate_audit[index]["official_api_schema_count"]),
                            -index,
                        ),
                    )
                sample = candidates[selected_candidate_index]
                parsed = APPWORLD_PYTHON_FENCE_CONTRACT.parse_response(sample.action_text)
                code = parsed.code if parsed.valid else "pass"
                state_id = f"before-{step_index:04d}-{uuid.uuid4().hex}"
                session.save_state(state_id)
                snapshot_sha256 = session.checkpoint_sha256(state_id)
                checkpoint_locator = session.checkpoint_locator(state_id)
                before_evaluator = evaluator
                # AppWorld checkpoints contain database state, not arbitrary
                # Python-shell locals. Reset before both executions so the
                # rollout contract is deterministic and actions are explicitly
                # self-contained rather than depending on lost shell state.
                session.load_state(state_id)
                first_output = str(session.execute(code))
                first_evaluator = dict(session.evaluate())
                first_completed = bool(session.task_completed())
                session.load_state(state_id)
                replay_output = str(session.execute(code))
                replay_evaluator = dict(session.evaluate())
                replay_completed = bool(session.task_completed())
                comparisons = {
                    "output": first_output == replay_output,
                    "evaluator": first_evaluator == replay_evaluator,
                    "task_completed": first_completed == replay_completed,
                }
                if not all(comparisons.values()):
                    raise AppWorldSuiteError("AppWorld restore/replay verification failed")
                evaluator = replay_evaluator
                actor_output = replay_output if parsed.valid else f"INVALID_ACTION: {parsed.error}"
                next_vector = self._compiler.project_state(
                    session.contract,
                    evaluator,
                    previous=vector,
                )
                label = self._compiler.label_transition(graph, vector, code, next_vector)
                receipt_root = self._artifacts_dir / traj_uid.replace(":", "_")
                receipt_root.mkdir(parents=True, exist_ok=True)
                receipt_path = receipt_root / f"restore-{step_index:04d}.json"
                receipt = {
                    "schema": "lh_appworld_restore_receipt/v1",
                    "state_id": state_id,
                    "state_snapshot_sha256": snapshot_sha256,
                    "checkpoint_locator": checkpoint_locator,
                    "comparisons": comparisons,
                    "restore_verified": True,
                    "python_shell_contract": "fresh_per_step_self_contained_code",
                }
                receipt_path.write_text(
                    json.dumps(receipt, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                receipt_sha256 = _sha256_file(receipt_path)
                won = bool(evaluator.get("success", False))
                completed = replay_completed
                terminal = bool(won or completed or step_index + 1 >= max_steps)
                if won:
                    terminal_reason = "official_evaluator_success"
                elif completed:
                    terminal_reason = "agent_completed_without_evaluator_success"
                before_passes = len(before_evaluator.get("passes", []))
                after_passes = len(evaluator.get("passes", []))
                awareness = (
                    asdict(
                        project_actor_awareness(
                            graph,
                            vector,
                            remaining_budget=remaining,
                        )
                    )
                    if self.config.actor_awareness_mode == "oracle_goal_graph"
                    else None
                )
                records.append(
                    TrainingTrajectoryRecord(
                        dataset_id=f"online:{self.manifest_sha256}",
                        benchmark="appworld_native",
                        task_id=task_id,
                        split=self.config.split,
                        seed=environment_seed,
                        traj_uid=traj_uid,
                        group_uid=group_uid,
                        step_idx=step_index,
                        instruction=session.contract.instruction,
                        observation=json.dumps(
                            {"last_output": history[-1]["output"] if history else ""},
                            sort_keys=True,
                        ),
                        action_raw=sample.action_text,
                        action_type="python",
                        parsed_action={
                            "code": parsed.code,
                            "valid": parsed.valid,
                            "error": parsed.error,
                        },
                        next_observation=actor_output,
                        reward=float(after_passes - before_passes),
                        terminal=terminal,
                        success=won,
                        outcome_evidence={
                            "official_evaluator": evaluator,
                            "task_completed": completed,
                        },
                        action_language="python",
                        method_source="on_policy",
                        model_id=policy.component.name,
                        trajectory_length=None,
                        terminal_reason="pending",
                        decision_role="executor",
                        state_snapshot_sha256=snapshot_sha256,
                        goal_hash=graph.sha256,
                        remaining_budget=remaining,
                        action_affordance_hash=canonical_sha256(
                            {"action_type": "python_code", "official_shell": True}
                        ),
                        rollout_config_hash=rollout_config_sha256,
                        active=True,
                        action_token_count=len(sample.action_token_ids),
                        prompt_hash=token_ids_sha256(sample.prompt_token_ids),
                        prompt_ids=sample.prompt_token_ids,
                        response_ids=sample.action_token_ids,
                        response_mask=(1,) * len(sample.action_token_ids),
                        response_old_logprobs=sample.old_action_token_log_probs,
                        extra={
                            "remaining_budget_verified": True,
                            "action_affordance_verified": True,
                            "action_token_count_verified": True,
                            "task_manifest_lock_sha256": self.manifest_sha256,
                            "restore_verified": True,
                            "appworld_datetime_restore_verified": bool(
                                getattr(
                                    session,
                                    "datetime_restore_verified",
                                    False,
                                )
                            ),
                            "restore_probe_receipt_path": str(receipt_path),
                            "restore_probe_receipt_sha256": receipt_sha256,
                            "checkpoint_locator": checkpoint_locator,
                            "python_shell_contract": "fresh_per_step_self_contained_code",
                            "appworld_task_contract": asdict(session.contract),
                            "appworld_evaluator_before": before_evaluator,
                            "appworld_evaluator_after": evaluator,
                            "goal_transition_annotation": asdict(label),
                            "actor_awareness_mode": self.config.actor_awareness_mode,
                            "actor_awareness": awareness,
                            "actor_prompt": prompt,
                            "model_response": sample.action_text,
                            "action_candidates_per_step": self.config.action_candidates_per_step,
                            "candidate_selection_policy": self.config.candidate_selection_policy,
                            "selected_candidate_index": selected_candidate_index,
                            "candidate_audit": candidate_audit,
                            "response_contract_id": APPWORLD_PYTHON_FENCE_CONTRACT.id,
                            "response_contract_sha256": (
                                APPWORLD_PYTHON_FENCE_CONTRACT.descriptor.sha256
                            ),
                            "response_stop_texts": list(APPWORLD_PYTHON_FENCE_CONTRACT.stop_texts),
                            "response_terminated_at_contract_boundary": (
                                sample.matched_stop_text
                                in APPWORLD_PYTHON_FENCE_CONTRACT.stop_texts
                                or sample.action_text.rstrip().endswith("```")
                            ),
                            "model_io_profile_sha256": (sample.model_io_profile_sha256),
                            "normalized_messages_sha256": (sample.normalized_messages_sha256),
                            "chat_template_sha256": sample.chat_template_sha256,
                            "matched_stop_text": sample.matched_stop_text,
                            "finish_reason": sample.finish_reason,
                            "environment_action": parsed.code,
                            "executed_code": code,
                            "appworld_action_schemas": list(
                                getattr(
                                    session,
                                    "canonical_action_schemas",
                                    lambda _code: (),
                                )(code)
                            ),
                            "environment_output": actor_output,
                            "is_action_valid": parsed.valid,
                            "package_version": self.backend.package_version,
                            "dataset_version": self.backend.dataset_version,
                            "source_revision": self.backend.source_revision,
                        },
                    )
                )
                history.append({"code": parsed.code, "output": actor_output})
                vector = next_vector
                if terminal:
                    break
            length = len(records)
            return [
                replace(
                    row,
                    trajectory_length=length,
                    terminal=index == length - 1,
                    success=won,
                    terminal_reason=terminal_reason,
                )
                for index, row in enumerate(records)
            ]
        finally:
            session.close()

    def _prompt(
        self,
        session: AppWorldSession,
        history: list[Mapping[str, str]],
        graph: Any,
        vector: GoalVector,
        remaining_budget: int,
    ) -> str:
        prompt = render_appworld_actor_prompt(
            instruction=session.contract.instruction,
            actor_context=session.actor_context,
            history=history,
            history_length=self.config.history_length,
            history_value_max_chars=self.config.history_value_max_chars,
            prompt_intervention_id=self.config.prompt_intervention_id,
        )
        if self.config.actor_awareness_mode == "oracle_goal_graph":
            projection = project_actor_awareness(
                graph,
                vector,
                remaining_budget=remaining_budget,
            )
            prompt += f"\n\n<goal_awareness>{projection.to_json()}</goal_awareness>"
        return prompt

    def _rotated(self, seed: int, count: int) -> tuple[str, ...]:
        if count > len(self.task_ids):
            raise AppWorldSuiteError("requested more AppWorld tasks than the locked split")
        offset = seed % len(self.task_ids)
        return tuple(self.task_ids[(offset + index) % len(self.task_ids)] for index in range(count))


def build_appworld_suite_pair(
    document: AppWorldSuitePairDocument,
    *,
    backend: AppWorldBackend,
    artifacts_dir: str | Path,
    actor_awareness_mode: str = "none",
    tasks_per_update: int = 1,
    eval_task_count: int = 1,
    history_length: int = 5,
    history_value_max_chars: int = 1536,
    model_io_profile_id: str | None = None,
    prompt_intervention_id: str = "none",
) -> tuple[AppWorldNativeTaskSuite, AppWorldNativeTaskSuite]:
    """Bind one validated portable document to train/dev native suites."""

    root = Path(artifacts_dir).resolve()
    shared = {
        "expected_package_version": document.package_version,
        "expected_dataset_version": document.dataset_version,
        "expected_source_revision": document.source_revision,
        "actor_awareness_mode": actor_awareness_mode,
        "history_length": history_length,
        "history_value_max_chars": history_value_max_chars,
        "model_io_profile_id": model_io_profile_id,
        "prompt_intervention_id": prompt_intervention_id,
    }
    train = AppWorldNativeTaskSuite(
        AppWorldSuiteConfig(
            role="train_task_suite",
            split=document.train_split,
            artifacts_dir=str(root / "train"),
            tasks_per_update=tasks_per_update,
            eval_task_count=eval_task_count,
            **shared,
        ),
        backend=backend,
    )
    evaluation = AppWorldNativeTaskSuite(
        AppWorldSuiteConfig(
            role="eval_task_suite",
            split=document.eval_split,
            artifacts_dir=str(root / "dev"),
            tasks_per_update=tasks_per_update,
            eval_task_count=eval_task_count,
            **shared,
        ),
        backend=backend,
    )
    return train, evaluation


def preflight_official_appworld_suite_pair(
    config_path: str | Path,
    *,
    source_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
) -> Path:
    """Verify official code/data and persist a no-model, no-GPU suite receipt."""

    document = load_appworld_suite_pair_document(config_path)
    backend = OfficialAppWorldBackend(
        source_root=str(Path(source_root).resolve()),
        root=str(Path(data_root).resolve()),
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train, evaluation = build_appworld_suite_pair(
        document,
        backend=backend,
        artifacts_dir=output / "environment",
    )
    receipt = {
        "schema": "lh_appworld_native_suite_preflight/v1",
        "config_path": str(document.config_path),
        "config_sha256": document.sha256,
        "source_revision": backend.source_revision,
        "package_version": backend.package_version,
        "dataset_version": backend.dataset_version,
        "train": {
            "split": train.config.split,
            "task_count": len(train.task_ids),
            "manifest_sha256": train.manifest_sha256,
        },
        "eval": {
            "split": evaluation.config.split,
            "task_count": len(evaluation.task_ids),
            "manifest_sha256": evaluation.manifest_sha256,
        },
        "gpu_loaded": False,
        "model_loaded": False,
        "claim_scope": "native_suite_preflight_only_no_training_or_score",
    }
    path = output / "appworld_suite_preflight.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class OfficialAppWorldBackend:
    """Thin import adapter over the official package; never emulates missing data."""

    def __init__(self, *, source_root: str, root: str | None = None) -> None:
        try:
            import appworld
            from appworld.task import Task
        except ImportError as error:
            raise AppWorldSuiteError(
                "official AppWorld package is not installed; refusing mock execution"
            ) from error
        if root is not None:
            appworld.update_root(root)
        checkout = Path(source_root).resolve()
        package_file = Path(appworld.__file__).resolve()
        if checkout not in package_file.parents:
            raise AppWorldSuiteError(
                "installed AppWorld package is not imported from the pinned source checkout"
            )
        try:
            self.source_revision = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise AppWorldSuiteError(f"cannot verify AppWorld source checkout: {error}") from error
        if dirty:
            raise AppWorldSuiteError("AppWorld source checkout has uncommitted or untracked files")
        self._appworld = appworld
        self._task_cls = Task
        self.package_version = str(appworld.__version__)
        all_ids = tuple(appworld.load_task_ids("train")) + tuple(appworld.load_task_ids("dev"))
        if not all_ids:
            raise AppWorldSuiteError("official AppWorld train/dev data is not installed")
        versions = {str(Task.load(task_id).db_version) for task_id in all_ids}
        if len(versions) != 1:
            raise AppWorldSuiteError("AppWorld train/dev tasks have mixed dataset versions")
        self.dataset_version = next(iter(versions))

    def task_ids(self, split: str) -> tuple[str, ...]:
        return tuple(self._appworld.load_task_ids(split))

    def load_training_supervision(self, task_id: str) -> Mapping[str, Any]:
        """Load train-only official solution text outside the runtime/evaluator API.

        This method is deliberately absent from :class:`AppWorldBackend` and
        :class:`AppWorldSession`: rollout, GoalGraph compilation, and held-out
        evaluation cannot call it through their declared interfaces.
        """

        if task_id not in self.task_ids("train"):
            raise AppWorldSuiteError("AppWorld supervision is restricted to train tasks")
        task = self._task_cls.load(
            task_id,
            load_ground_truth=True,
            ground_truth_mode="full",
        )
        ground_truth = task.ground_truth
        body = str(getattr(ground_truth, "compiled_solution_code_body", "")).strip()
        if not body or "apis." not in body:
            raise AppWorldSuiteError(
                "AppWorld train supervision lacks executable compiled API code"
            )
        try:
            compile(body, f"appworld-train:{task_id}", "exec")
        except SyntaxError as error:
            raise AppWorldSuiteError(
                f"AppWorld compiled train solution is not executable Python: {error}"
            ) from error
        return {
            "task_id": str(task_id),
            "instruction": str(task.instruction),
            "actor_context": {
                "supervisor": dict(task.supervisor),
                "app_descriptions": dict(task.app_descriptions),
            },
            "compiled_solution_code_body": body,
            "required_apps": tuple(str(value) for value in ground_truth.required_apps),
        }

    def open_task(
        self,
        *,
        task_id: str,
        split: str,
        experiment_name: str,
        seed: int,
    ) -> AppWorldSession:
        if task_id not in self.task_ids(split):
            raise AppWorldSuiteError("AppWorld task does not belong to requested split")
        world = self._appworld.AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            random_seed=seed,
            load_ground_truth=True,
            ground_truth_mode="minimal",
        )
        return _OfficialAppWorldSession(world=world, split=split)


class _OfficialAppWorldSession:
    def __init__(self, *, world: Any, split: str) -> None:
        self.world = world
        ground_truth = world.task.ground_truth
        if ground_truth is None or not isinstance(ground_truth.test_data, list):
            world.close()
            raise AppWorldSuiteError(
                "AppWorld train/dev task lacks evaluator test_data in minimal ground truth"
            )
        requirements = tuple(str(item["requirement"]) for item in ground_truth.test_data)
        self.contract = AppWorldTaskContract(
            task_id=str(world.task_id),
            split=split,
            instruction=str(world.task.instruction),
            evaluator_requirements=requirements,
            required_apps=tuple(str(value) for value in ground_truth.required_apps),
        )
        self.actor_context = {
            "supervisor": dict(world.task.supervisor),
            "app_descriptions": dict(world.task.app_descriptions),
        }
        self._api_docs_cache: dict[str, tuple[Mapping[str, Any], ...]] = {}
        self.datetime_restore_verified = True

    def evaluate(self) -> Mapping[str, Any]:
        return self.world.evaluate().to_dict(stats_only=False)

    def execute(self, code: str) -> str:
        return str(self.world.execute(code))

    def save_state(self, state_id: str) -> str:
        result = self.world.save_state(state_id=state_id)
        if result != state_id:
            raise AppWorldSuiteError("official AppWorld did not retain requested checkpoint ID")
        return state_id

    def load_state(self, state_id: str) -> None:
        self.world.load_state(state_id=state_id)
        restore_datetime = getattr(self.world, "_set_datetime", None)
        if not callable(restore_datetime):
            raise AppWorldSuiteError("official AppWorld load_state cannot restore task datetime")
        # Upstream load_state() calls AppWorld.close_all(), which stops the
        # active freezegun context but does not recreate it.  Re-establish the
        # task clock after every restore so the next restore and final close
        # each stop exactly one live context.
        restore_datetime()

    def checkpoint_sha256(self, state_id: str) -> str:
        root = Path(self.world.output_checkpoints_directory) / state_id
        if not root.is_dir():
            raise AppWorldSuiteError("official AppWorld checkpoint directory is missing")
        digest = hashlib.sha256()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def checkpoint_locator(self, state_id: str) -> str:
        return str((Path(self.world.output_checkpoints_directory) / state_id).resolve())

    def task_completed(self) -> bool:
        return bool(self.world.task_completed())

    def canonical_action_schemas(self, code: str) -> tuple[Mapping[str, Any], ...]:
        """Bind executed API calls to official name/schema/side-effect docs."""

        from appworld.api_docs import prepare_api_docs

        output = []
        seen = set()
        for match in re.finditer(
            r"apis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)",
            code,
        ):
            app_name, api_name = match.groups()
            if app_name not in self._api_docs_cache:
                self._api_docs_cache[app_name] = tuple(
                    dict(value)
                    for value in prepare_api_docs(
                        app_name,
                        include_private_apis=False,
                        format="standard",
                    )
                )
            matches = tuple(
                value
                for value in self._api_docs_cache[app_name]
                if value.get("api_name") == api_name
            )
            if len(matches) != 1:
                continue
            document = matches[0]
            parameters = tuple(
                {
                    "name": str(parameter.get("name", "")),
                    "type": str(parameter.get("type", "")),
                    "required": bool(parameter.get("required", False)),
                }
                for parameter in document.get("parameters", ())
                if isinstance(parameter, Mapping)
            )
            method = str(document.get("method", "")).upper()
            schema = {
                "app_name": app_name,
                "api_name": api_name,
                "method": method,
                "parameters": list(parameters),
                "side_effect_type": {
                    "GET": "read",
                    "POST": "create_or_invoke",
                    "PATCH": "update",
                    "PUT": "replace",
                    "DELETE": "delete",
                }.get(method, "unknown"),
            }
            identity = canonical_sha256(schema)
            if identity not in seen:
                output.append({**schema, "schema_sha256": identity})
                seen.add(identity)
        return tuple(output)

    def close(self) -> None:
        self.world.close()


def _parse_python_action(text: str) -> ParsedAppWorldAction:
    """Compatibility entrypoint; new callers use the registered contract."""

    return APPWORLD_PYTHON_FENCE_CONTRACT.parse_response(text)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise AppWorldSuiteError(f"AppWorld suite field {key!r} must be a mapping")
    return value


def _expect_keys(
    raw: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(raw) != expected:
        raise AppWorldSuiteError(
            f"{label} fields differ from schema: "
            + json.dumps(
                {
                    "missing": sorted(expected - set(raw)),
                    "unknown": sorted(set(raw) - expected),
                },
                sort_keys=True,
            )
        )
