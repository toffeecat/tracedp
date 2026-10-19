"""Executable parity audit for the pinned G2PO ALFWorld actor contract."""

from __future__ import annotations

import hashlib
import runpy
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from lh_platform.training.g2po.actor_protocol import (
    G2PO_ALFWORLD_ACTOR_PROTOCOL_ID,
    MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID,
    QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_FAMILY_ID,
    G2POActorHistoryStep,
    G2POAlfWorldActorProtocol,
    alfworld_actor_protocol_semantic_family_id,
)

G2PO_ACTOR_PARITY_SCOPE = "prompt_projection_anchor_sampling_semantics_not_vllm_or_verl_bitwise"
MODEL_COMPATIBLE_ACTOR_PARITY_SCOPE = (
    "source_context_history_anchor_sampling_with_action_only_output_contract_"
    "not_official_g2po_actor"
)
QWEN_PROMPT_INJECTED_THINKING_ACTOR_PARITY_SCOPE = (
    "source_context_history_anchor_sampling_with_qwen_prompt_injected_"
    "thinking_action_contract_not_official_g2po_actor"
)

__all__ = [
    "G2PO_ACTOR_PARITY_SCOPE",
    "MODEL_COMPATIBLE_ACTOR_PARITY_SCOPE",
    "QWEN_PROMPT_INJECTED_THINKING_ACTOR_PARITY_SCOPE",
    "G2POActorParityReceipt",
    "audit_alfworld_actor_profile",
    "audit_g2po_actor_parity",
]


@dataclass(frozen=True)
class G2POActorParityReceipt:
    actor_protocol_id: str
    actor_protocol_semantic_family_id: str
    source_revision: str
    source_file_sha256: Mapping[str, str]
    prompt_cases: int
    projection_cases: int
    engine_exact: bool
    behavioral_contract_exact: bool
    source_top_k: int
    source_top_p: float
    hf_top_k: int | None
    hf_repetition_penalty: float | None
    full_support_sampling_verified: bool
    scope: str
    verified: bool


def audit_g2po_actor_parity(
    *,
    source_root: str | Path,
    actor_config_path: str | Path,
) -> G2POActorParityReceipt:
    """Backward-compatible name for the registered ALFWorld actor audit."""

    return audit_alfworld_actor_profile(
        source_root=source_root,
        actor_config_path=actor_config_path,
    )


def audit_alfworld_actor_profile(
    *,
    source_root: str | Path,
    actor_config_path: str | Path,
) -> G2POActorParityReceipt:
    """Audit source-exact and explicitly downgraded ALFWorld actor profiles."""

    root = Path(source_root).resolve()
    config_path = Path(actor_config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("G2PO actor profile must be a mapping")
    expected_revision = str(raw.get("source_revision", ""))
    actual_revision = _git_output(root, "rev-parse", "HEAD")
    if actual_revision != expected_revision:
        raise ValueError(
            "G2PO actor source revision mismatch: "
            f"expected {expected_revision}, got {actual_revision}"
        )
    configured_files = raw.get("source_files")
    if not isinstance(configured_files, Mapping):
        raise ValueError("G2PO actor profile source_files must be a mapping")
    actual_files: dict[str, str] = {}
    for relative, expected in configured_files.items():
        path = (root / str(relative)).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"G2PO actor source file is missing or escapes root: {relative}")
        actual = _sha256_file(path)
        if actual != str(expected):
            raise ValueError(
                f"G2PO actor source hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        actual_files[str(relative)] = actual

    prompt_source = runpy.run_path(str(root / "agent_system/environments/prompts/alfworld.py"))
    projection_source = runpy.run_path(
        str(root / "agent_system/environments/env_package/alfworld/projection.py")
    )
    protocol_config = raw.get("protocol")
    if not isinstance(protocol_config, Mapping):
        raise ValueError("ALFWorld actor protocol profile must be a mapping")
    protocol_id = str(protocol_config.get("id", ""))
    semantic_family_id = alfworld_actor_protocol_semantic_family_id(protocol_id)
    protocol = G2POAlfWorldActorProtocol(
        history_length=2,
        protocol_id=protocol_id,
    )
    actions = ("go to fridge 1", "help", "look")
    formatted_actions = "\n ".join(f"'{item}'" for item in actions if item != "help")
    initial = protocol.render_prompt(
        task_description="put the apple in the microwave",
        current_observation="You are in a kitchen.",
        admissible_actions=actions,
        history=(),
    )
    source_initial = prompt_source["ALFWORLD_TEMPLATE_NO_HIS"].format(
        current_observation="You are in a kitchen.",
        admissible_actions=formatted_actions,
    )
    history = (
        G2POActorHistoryStep("You are in a kitchen.", "go to fridge 1"),
        G2POActorHistoryStep("You arrive at fridge 1.", "open fridge 1"),
    )
    history_text = "\n".join(
        f"[Observation {index}: '{item.observation}', Action {index}: '{item.environment_action}']"
        for index, item in enumerate(history, start=1)
    )
    subsequent = protocol.render_prompt(
        task_description="put the apple in the microwave",
        current_observation="The fridge is open.",
        admissible_actions=("take apple 1 from fridge 1", "help"),
        history=history,
    )
    source_subsequent = prompt_source["ALFWORLD_TEMPLATE"].format(
        task_description="put the apple in the microwave",
        step_count=2,
        history_length=2,
        action_history=history_text,
        current_step=3,
        current_observation="The fridge is open.",
        admissible_actions="'take apple 1 from fridge 1'",
    )
    source_prompts = (source_initial, source_subsequent)
    local_prompts = (initial, subsequent)
    if protocol_id == G2PO_ALFWORLD_ACTOR_PROTOCOL_ID:
        if local_prompts != source_prompts:
            raise ValueError("local G2PO prompt renderer differs from pinned source")
    elif semantic_family_id == MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID:
        for local_prompt, source_prompt in zip(local_prompts, source_prompts, strict=True):
            local_prefix = local_prompt.split("\nNow it's your turn", maxsplit=1)[0]
            source_prefix = source_prompt.split("\nNow it's your turn", maxsplit=1)[0]
            if local_prefix != source_prefix:
                raise ValueError("model-compatible actor changed source context/history semantics")
            if "<think>" in local_prompt or "Do not add an explanation." not in local_prompt:
                raise ValueError("model-compatible actor output contract is invalid")
    elif semantic_family_id == QWEN_PROMPT_INJECTED_THINKING_ALFWORLD_ACTOR_FAMILY_ID:
        for local_prompt, source_prompt in zip(local_prompts, source_prompts, strict=True):
            local_prefix = local_prompt.split("\nNow it's your turn", maxsplit=1)[0]
            source_prefix = source_prompt.split("\nNow it's your turn", maxsplit=1)[0]
            if local_prefix != source_prefix:
                raise ValueError(
                    "prompt-injected thinking actor changed source context/history semantics"
                )
            if (
                "close it with </think>" not in local_prompt
                or "Do not generate another <think> tag." not in local_prompt
                or "enclosed within <think> </think>" in local_prompt
            ):
                raise ValueError("prompt-injected thinking actor output contract is invalid")
    else:
        raise ValueError(f"unsupported ALFWorld actor protocol: {protocol_id}")

    projection_cases = (
        "<think>Inspect.</think><action>GO TO FRIDGE 1</action>",
        "<action>look</action>",
        "<think>观察</think><action>look</action>",
        "response without action tags ending in go to desk 1",
        "<think>Try.</think><action>dance</action>",
    )
    local = tuple(
        protocol.project_response(
            response,
            admissible_actions=("look", "go to fridge 1"),
        )
        for response in projection_cases
    )
    if protocol_id == G2PO_ALFWORLD_ACTOR_PROTOCOL_ID:
        source_actions, source_valids = projection_source["alfworld_projection"](
            list(projection_cases),
            [["look", "go to fridge 1"] for _ in projection_cases],
        )
        if tuple(source_actions) != tuple(item.environment_action for item in local):
            raise ValueError("local G2PO action projection differs from pinned source")
        if tuple(bool(item) for item in source_valids) != tuple(
            item.is_action_valid for item in local
        ):
            raise ValueError("local G2PO action validity differs from pinned source")
    elif semantic_family_id == MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID:
        if not protocol.project_response(
            "<action>look</action>",
            admissible_actions=("look",),
        ).is_action_valid:
            raise ValueError("model-compatible actor still requires reasoning tags")
    elif not protocol.project_response(
        "Inspect.</think><action>look</action>",
        admissible_actions=("look",),
    ).is_action_valid:
        raise ValueError("prompt-injected thinking actor does not accept a closing-only response")

    trainer_config_path = root / "verl/trainer/config/ppo_trainer.yaml"
    trainer_config = yaml.safe_load(trainer_config_path.read_text(encoding="utf-8"))
    try:
        source_rollout = trainer_config["actor_rollout_ref"]["rollout"]
        source_top_k = int(source_rollout["top_k"])
        source_top_p = float(source_rollout["top_p"])
        source_temperature = float(source_rollout["temperature"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pinned G2PO sampling defaults are unreadable") from error
    generation = raw.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("G2PO actor generation profile must be a mapping")
    full_support = "top_k" in generation or "repetition_penalty" in generation
    hf_top_k = int(generation["top_k"]) if full_support else None
    hf_repetition_penalty = float(generation["repetition_penalty"]) if full_support else None
    if full_support:
        source_text = trainer_config_path.read_text(encoding="utf-8")
        mapping_is_documented = "top_k: -1 # 0 for hf rollout, -1 for vllm rollout" in source_text
        if (
            source_temperature != 1.0
            or source_top_k != -1
            or source_top_p != 1.0
            or float(generation.get("train_temperature", 0.0)) != 1.0
            or float(generation.get("top_p", 0.0)) != 1.0
            or hf_top_k != 0
            or hf_repetition_penalty != 1.0
            or not mapping_is_documented
        ):
            raise ValueError("HF full-support sampling does not map pinned G2PO defaults")

    parity = raw.get("platform_parity")
    if not isinstance(parity, Mapping):
        raise ValueError("G2PO actor platform_parity must be a mapping")
    scope = str(parity.get("scope", ""))
    engine_exact = bool(parity.get("engine_exact"))
    behavioral_exact = bool(parity.get("behavioral_contract_exact"))
    if protocol_id == G2PO_ALFWORLD_ACTOR_PROTOCOL_ID:
        expected_scope = G2PO_ACTOR_PARITY_SCOPE
    elif semantic_family_id == MODEL_COMPATIBLE_NON_THINKING_ALFWORLD_ACTOR_FAMILY_ID:
        expected_scope = MODEL_COMPATIBLE_ACTOR_PARITY_SCOPE
    else:
        expected_scope = QWEN_PROMPT_INJECTED_THINKING_ACTOR_PARITY_SCOPE
    expected_behavioral_exact = protocol_id == G2PO_ALFWORLD_ACTOR_PROTOCOL_ID
    if scope != expected_scope or engine_exact or behavioral_exact is not expected_behavioral_exact:
        raise ValueError("G2PO actor parity claim boundary is invalid")
    return G2POActorParityReceipt(
        actor_protocol_id=protocol_id,
        actor_protocol_semantic_family_id=semantic_family_id,
        source_revision=actual_revision,
        source_file_sha256=dict(sorted(actual_files.items())),
        prompt_cases=2,
        projection_cases=len(projection_cases),
        engine_exact=engine_exact,
        behavioral_contract_exact=behavioral_exact,
        source_top_k=source_top_k,
        source_top_p=source_top_p,
        hf_top_k=hf_top_k,
        hf_repetition_penalty=hf_repetition_penalty,
        full_support_sampling_verified=full_support,
        scope=scope,
        verified=True,
    )


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot inspect pinned G2PO source: {result.stderr.strip()}")
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
