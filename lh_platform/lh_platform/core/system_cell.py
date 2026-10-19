"""Typed, hash-bound policy, sampler, system-cell, and horizon contracts.

These contracts are pure CPU-side configuration evidence.  Loading them must
never import a model framework or imply that a rollout/training run is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

POLICY_IDENTITY_SCHEMA = "lh_policy_identity/v1"
SAMPLING_POLICY_SCHEMA = "lh_sampling_policy/v1"
HASH_BOUND_COMPONENT_SCHEMA = "lh_hash_bound_component/v1"
SYSTEM_CELL_MANIFEST_SCHEMA = "lh_system_cell_manifest/v1"
HORIZON_FACTOR_SCHEMA = "lh_horizon_factor/v1"
HORIZON_FACTOR_MANIFEST_SCHEMA = "lh_horizon_factor_manifest/v1"


class SystemCellContractError(ValueError):
    """Raised when a system-cell or horizon contract is ambiguous or mutable."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SystemCellContractError(
            f"system-cell evidence is not canonical JSON: {error!r}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class HashBoundComponentV1:
    component_id: str
    revision: str
    config: Mapping[str, Any]
    config_sha256: str
    schema: str = HASH_BOUND_COMPONENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HASH_BOUND_COMPONENT_SCHEMA:
            raise SystemCellContractError("unsupported hash-bound component schema")
        _require_identifier("component_id", self.component_id)
        _require_identifier("revision", self.revision)
        if not isinstance(self.config, Mapping) or not self.config:
            raise SystemCellContractError("hash-bound component config must be non-empty")
        _require_sha256("config_sha256", self.config_sha256)
        if canonical_sha256(dict(self.config)) != self.config_sha256:
            raise SystemCellContractError("hash-bound component config SHA-256 mismatch")

    @classmethod
    def build(
        cls,
        *,
        component_id: str,
        revision: str,
        config: Mapping[str, Any],
    ) -> HashBoundComponentV1:
        body = dict(config)
        return cls(
            component_id=component_id,
            revision=revision,
            config=body,
            config_sha256=canonical_sha256(body),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HashBoundComponentV1:
        _expect_exact_keys(
            payload,
            {"schema", "component_id", "revision", "config", "config_sha256"},
            "hash-bound component",
        )
        return cls(
            schema=str(payload["schema"]),
            component_id=str(payload["component_id"]),
            revision=str(payload["revision"]),
            config=_mapping(payload["config"], "hash-bound component.config"),
            config_sha256=str(payload["config_sha256"]),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "component_id": self.component_id,
            "revision": self.revision,
            "config": dict(self.config),
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True)
class PolicyIdentityV1:
    policy_id: str
    model_reference: str
    model_revision: str
    base_checkpoint_sha256: str
    adapter_artifact_sha256: str | None
    policy_checkpoint_sha256: str
    tokenizer_artifact_sha256: str
    model_io_profile_id: str
    model_io_profile_sha256: str
    backend_id: str
    backend_revision: str
    dtype: str
    policy_version: int
    frozen: bool
    schema: str = POLICY_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != POLICY_IDENTITY_SCHEMA:
            raise SystemCellContractError("unsupported policy-identity schema")
        for name, value in (
            ("policy_id", self.policy_id),
            ("model_reference", self.model_reference),
            ("model_revision", self.model_revision),
            ("model_io_profile_id", self.model_io_profile_id),
            ("backend_id", self.backend_id),
            ("backend_revision", self.backend_revision),
        ):
            _require_identifier(name, value)
        for name, value in (
            ("base_checkpoint_sha256", self.base_checkpoint_sha256),
            ("policy_checkpoint_sha256", self.policy_checkpoint_sha256),
            ("tokenizer_artifact_sha256", self.tokenizer_artifact_sha256),
            ("model_io_profile_sha256", self.model_io_profile_sha256),
        ):
            _require_sha256(name, value)
        if self.adapter_artifact_sha256 is not None:
            _require_sha256("adapter_artifact_sha256", self.adapter_artifact_sha256)
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise SystemCellContractError("policy dtype is unsupported")
        if (
            isinstance(self.policy_version, bool)
            or not isinstance(self.policy_version, int)
            or self.policy_version < 0
        ):
            raise SystemCellContractError("policy_version must be a non-negative integer")
        if not isinstance(self.frozen, bool):
            raise SystemCellContractError("policy frozen flag must be boolean")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PolicyIdentityV1:
        keys = {
            "schema",
            "policy_id",
            "model_reference",
            "model_revision",
            "base_checkpoint_sha256",
            "adapter_artifact_sha256",
            "policy_checkpoint_sha256",
            "tokenizer_artifact_sha256",
            "model_io_profile_id",
            "model_io_profile_sha256",
            "backend_id",
            "backend_revision",
            "dtype",
            "policy_version",
            "frozen",
        }
        _expect_exact_keys(payload, keys, "policy identity")
        adapter = payload["adapter_artifact_sha256"]
        return cls(
            schema=str(payload["schema"]),
            policy_id=str(payload["policy_id"]),
            model_reference=str(payload["model_reference"]),
            model_revision=str(payload["model_revision"]),
            base_checkpoint_sha256=str(payload["base_checkpoint_sha256"]),
            adapter_artifact_sha256=None if adapter is None else str(adapter),
            policy_checkpoint_sha256=str(payload["policy_checkpoint_sha256"]),
            tokenizer_artifact_sha256=str(payload["tokenizer_artifact_sha256"]),
            model_io_profile_id=str(payload["model_io_profile_id"]),
            model_io_profile_sha256=str(payload["model_io_profile_sha256"]),
            backend_id=str(payload["backend_id"]),
            backend_revision=str(payload["backend_revision"]),
            dtype=str(payload["dtype"]),
            policy_version=_integer(payload["policy_version"], "policy_version"),
            frozen=_boolean(payload["frozen"], "policy frozen"),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "model_reference": self.model_reference,
            "model_revision": self.model_revision,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "adapter_artifact_sha256": self.adapter_artifact_sha256,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "tokenizer_artifact_sha256": self.tokenizer_artifact_sha256,
            "model_io_profile_id": self.model_io_profile_id,
            "model_io_profile_sha256": self.model_io_profile_sha256,
            "backend_id": self.backend_id,
            "backend_revision": self.backend_revision,
            "dtype": self.dtype,
            "policy_version": self.policy_version,
            "frozen": self.frozen,
        }


@dataclass(frozen=True)
class SamplingPolicyV1:
    sampling_policy_id: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_response_tokens: int
    stop_texts: tuple[str, ...]
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    support_mode: str
    projection_id: str
    projection_sha256: str
    grammar_mask_sha256: str | None
    behavior_logprob_mode: str
    rng_scheme_id: str
    seed_namespace: str
    per_response_rng: bool
    schema: str = SAMPLING_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLING_POLICY_SCHEMA:
            raise SystemCellContractError("unsupported sampling-policy schema")
        for name, value in (
            ("sampling_policy_id", self.sampling_policy_id),
            ("projection_id", self.projection_id),
            ("rng_scheme_id", self.rng_scheme_id),
            ("seed_namespace", self.seed_namespace),
        ):
            _require_identifier(name, value)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise SystemCellContractError("sampling temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise SystemCellContractError("sampling top_p must be in (0, 1]")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise SystemCellContractError("sampling top_k must be a non-negative integer")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise SystemCellContractError("repetition_penalty must be finite and positive")
        if (
            isinstance(self.max_response_tokens, bool)
            or not isinstance(self.max_response_tokens, int)
            or self.max_response_tokens < 1
        ):
            raise SystemCellContractError("max_response_tokens must be positive")
        if any(not value or value != value.strip() for value in self.stop_texts):
            raise SystemCellContractError(
                "sampling stop_texts must contain only canonical non-empty values"
            )
        if len(self.stop_texts) != len(set(self.stop_texts)):
            raise SystemCellContractError("sampling stop_texts must be unique")
        if not self.eos_token_ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.eos_token_ids
        ):
            raise SystemCellContractError("sampling eos_token_ids are invalid")
        if len(self.eos_token_ids) != len(set(self.eos_token_ids)):
            raise SystemCellContractError("sampling eos_token_ids must be unique")
        if (
            isinstance(self.pad_token_id, bool)
            or not isinstance(self.pad_token_id, int)
            or self.pad_token_id < 0
        ):
            raise SystemCellContractError("sampling pad_token_id is invalid")
        if self.support_mode not in {
            "full_vocabulary_then_projection",
            "token_constrained_admissible",
            "grammar_masked",
        }:
            raise SystemCellContractError("sampling support_mode is unsupported")
        _require_sha256("projection_sha256", self.projection_sha256)
        if self.grammar_mask_sha256 is not None:
            _require_sha256("grammar_mask_sha256", self.grammar_mask_sha256)
        if self.support_mode == "grammar_masked" and self.grammar_mask_sha256 is None:
            raise SystemCellContractError("grammar-masked support requires a grammar hash")
        if self.support_mode != "grammar_masked" and self.grammar_mask_sha256 is not None:
            raise SystemCellContractError(
                "grammar hash is forbidden outside grammar-masked support"
            )
        if self.behavior_logprob_mode not in {
            "hf_generate_processed_scores",
            "processed_post_warp",
            "generation_time_constrained_support",
        }:
            raise SystemCellContractError("behavior_logprob_mode is unsupported")
        if not isinstance(self.per_response_rng, bool):
            raise SystemCellContractError("per_response_rng must be boolean")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SamplingPolicyV1:
        keys = {
            "schema",
            "sampling_policy_id",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "max_response_tokens",
            "stop_texts",
            "eos_token_ids",
            "pad_token_id",
            "support_mode",
            "projection_id",
            "projection_sha256",
            "grammar_mask_sha256",
            "behavior_logprob_mode",
            "rng_scheme_id",
            "seed_namespace",
            "per_response_rng",
        }
        _expect_exact_keys(payload, keys, "sampling policy")
        grammar = payload["grammar_mask_sha256"]
        return cls(
            schema=str(payload["schema"]),
            sampling_policy_id=str(payload["sampling_policy_id"]),
            temperature=_number(payload["temperature"], "temperature"),
            top_p=_number(payload["top_p"], "top_p"),
            top_k=_integer(payload["top_k"], "top_k"),
            repetition_penalty=_number(payload["repetition_penalty"], "repetition_penalty"),
            max_response_tokens=_integer(payload["max_response_tokens"], "max_response_tokens"),
            stop_texts=_string_tuple(payload["stop_texts"], "stop_texts"),
            eos_token_ids=_integer_tuple(payload["eos_token_ids"], "eos_token_ids"),
            pad_token_id=_integer(payload["pad_token_id"], "pad_token_id"),
            support_mode=str(payload["support_mode"]),
            projection_id=str(payload["projection_id"]),
            projection_sha256=str(payload["projection_sha256"]),
            grammar_mask_sha256=None if grammar is None else str(grammar),
            behavior_logprob_mode=str(payload["behavior_logprob_mode"]),
            rng_scheme_id=str(payload["rng_scheme_id"]),
            seed_namespace=str(payload["seed_namespace"]),
            per_response_rng=_boolean(payload["per_response_rng"], "per_response_rng"),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sampling_policy_id": self.sampling_policy_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_response_tokens": self.max_response_tokens,
            "stop_texts": list(self.stop_texts),
            "eos_token_ids": list(self.eos_token_ids),
            "pad_token_id": self.pad_token_id,
            "support_mode": self.support_mode,
            "projection_id": self.projection_id,
            "projection_sha256": self.projection_sha256,
            "grammar_mask_sha256": self.grammar_mask_sha256,
            "behavior_logprob_mode": self.behavior_logprob_mode,
            "rng_scheme_id": self.rng_scheme_id,
            "seed_namespace": self.seed_namespace,
            "per_response_rng": self.per_response_rng,
        }


@dataclass(frozen=True)
class SystemCellManifestV1:
    system_cell_id: str
    policy_identity: PolicyIdentityV1
    sampling_policy: SamplingPolicyV1
    actor_protocol: HashBoundComponentV1
    context: HashBoundComponentV1
    memory: HashBoundComponentV1
    runtime: HashBoundComponentV1
    environment: HashBoundComponentV1
    action_objective: HashBoundComponentV1
    frozen: bool
    adaptive_components_allowed: bool
    optimizer_present: bool
    schema: str = SYSTEM_CELL_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SYSTEM_CELL_MANIFEST_SCHEMA:
            raise SystemCellContractError("unsupported system-cell schema")
        _require_identifier("system_cell_id", self.system_cell_id)
        if not all(
            isinstance(value, bool)
            for value in (
                self.frozen,
                self.adaptive_components_allowed,
                self.optimizer_present,
            )
        ):
            raise SystemCellContractError("system-cell lifecycle flags must be boolean")
        _require_component_config(
            self.actor_protocol,
            "actor_protocol",
            {
                "history_length",
                "prompt_template_sha256",
                "response_contract_sha256",
                "action_projection_sha256",
            },
        )
        _require_component_config(
            self.context,
            "context",
            {"max_prompt_tokens", "history_policy", "context_overflow_policy"},
        )
        _require_component_config(self.memory, "memory", {"mode", "persistence"})
        _require_component_config(
            self.runtime,
            "runtime",
            {
                "generation_backend",
                "execution_mode",
                "batch_geometry_id",
                "kernel_profile",
                "cache_lifecycle",
            },
        )
        _require_component_config(
            self.environment,
            "environment",
            {"benchmark", "adapter_id", "adapter_sha256", "versions"},
        )
        _require_component_config(
            self.action_objective,
            "action_objective",
            {"unit", "reward_source", "token_ownership"},
        )
        for component, keys in (
            (
                self.actor_protocol,
                (
                    "prompt_template_sha256",
                    "response_contract_sha256",
                    "action_projection_sha256",
                ),
            ),
            (self.environment, ("adapter_sha256",)),
        ):
            for key in keys:
                _require_sha256(key, str(component.config[key]))
        max_prompt_tokens = self.context.config["max_prompt_tokens"]
        history_length = self.actor_protocol.config["history_length"]
        if isinstance(max_prompt_tokens, bool) or not isinstance(max_prompt_tokens, int):
            raise SystemCellContractError("context max_prompt_tokens must be an integer")
        if max_prompt_tokens < 1:
            raise SystemCellContractError("context max_prompt_tokens must be positive")
        if isinstance(history_length, bool) or not isinstance(history_length, int):
            raise SystemCellContractError("actor history_length must be an integer")
        if history_length < 0:
            raise SystemCellContractError("actor history_length must be non-negative")
        if not isinstance(self.environment.config["versions"], Mapping):
            raise SystemCellContractError("environment versions must be a mapping")
        for component in (
            self.actor_protocol,
            self.context,
            self.memory,
            self.runtime,
            self.environment,
            self.action_objective,
        ):
            for payload_key, hash_key in (
                ("semantic_config", "semantic_config_sha256"),
                ("implementation_lock", "implementation_lock_sha256"),
            ):
                present = payload_key in component.config or hash_key in component.config
                if not present:
                    continue
                payload = component.config.get(payload_key)
                if not isinstance(payload, Mapping) or not payload:
                    raise SystemCellContractError(
                        f"{component.component_id} {payload_key} must be a non-empty mapping"
                    )
                _require_sha256(hash_key, str(component.config.get(hash_key, "")))
                if canonical_sha256(dict(payload)) != component.config[hash_key]:
                    raise SystemCellContractError(
                        f"{component.component_id} {payload_key} SHA-256 mismatch"
                    )
                for key, value in payload.items():
                    if key in component.config and component.config[key] != value:
                        raise SystemCellContractError(
                            f"{component.component_id} {payload_key} differs from flat config"
                        )
        if "backend_component_sha256" in self.runtime.config:
            _require_sha256(
                "backend_component_sha256",
                str(self.runtime.config["backend_component_sha256"]),
            )
            _require_sha256(
                "runtime_policy_identity_sha256",
                str(self.runtime.config.get("runtime_policy_identity_sha256", "")),
            )
            if self.runtime.config.get("generation_backend") != self.policy_identity.backend_id:
                raise SystemCellContractError(
                    "runtime generation backend differs from policy backend"
                )
            if self.runtime.config.get("backend_revision") != self.policy_identity.backend_revision:
                raise SystemCellContractError(
                    "runtime backend revision differs from policy backend revision"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SystemCellManifestV1:
        keys = {
            "schema",
            "system_cell_id",
            "policy_identity",
            "sampling_policy",
            "actor_protocol",
            "context",
            "memory",
            "runtime",
            "environment",
            "action_objective",
            "frozen",
            "adaptive_components_allowed",
            "optimizer_present",
        }
        _expect_exact_keys(payload, keys, "system-cell manifest")
        return cls(
            schema=str(payload["schema"]),
            system_cell_id=str(payload["system_cell_id"]),
            policy_identity=PolicyIdentityV1.from_dict(
                _mapping(payload["policy_identity"], "policy_identity")
            ),
            sampling_policy=SamplingPolicyV1.from_dict(
                _mapping(payload["sampling_policy"], "sampling_policy")
            ),
            actor_protocol=HashBoundComponentV1.from_dict(
                _mapping(payload["actor_protocol"], "actor_protocol")
            ),
            context=HashBoundComponentV1.from_dict(_mapping(payload["context"], "context")),
            memory=HashBoundComponentV1.from_dict(_mapping(payload["memory"], "memory")),
            runtime=HashBoundComponentV1.from_dict(_mapping(payload["runtime"], "runtime")),
            environment=HashBoundComponentV1.from_dict(
                _mapping(payload["environment"], "environment")
            ),
            action_objective=HashBoundComponentV1.from_dict(
                _mapping(payload["action_objective"], "action_objective")
            ),
            frozen=_boolean(payload["frozen"], "system-cell frozen"),
            adaptive_components_allowed=_boolean(
                payload["adaptive_components_allowed"],
                "adaptive_components_allowed",
            ),
            optimizer_present=_boolean(payload["optimizer_present"], "optimizer_present"),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "system_cell_id": self.system_cell_id,
            "policy_identity": self.policy_identity.to_dict(),
            "sampling_policy": self.sampling_policy.to_dict(),
            "actor_protocol": self.actor_protocol.to_dict(),
            "context": self.context.to_dict(),
            "memory": self.memory.to_dict(),
            "runtime": self.runtime.to_dict(),
            "environment": self.environment.to_dict(),
            "action_objective": self.action_objective.to_dict(),
            "frozen": self.frozen,
            "adaptive_components_allowed": self.adaptive_components_allowed,
            "optimizer_present": self.optimizer_present,
        }


@dataclass(frozen=True)
class HorizonFactorV1:
    factor_id: str
    mode: str
    operationalization_id: str
    unit: str
    levels: tuple[str, ...]
    source_contract_sha256: str
    intrinsic_horizon: bool
    schema: str = HORIZON_FACTOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HORIZON_FACTOR_SCHEMA:
            raise SystemCellContractError("unsupported horizon-factor schema")
        for name, value in (
            ("factor_id", self.factor_id),
            ("operationalization_id", self.operationalization_id),
            ("unit", self.unit),
        ):
            _require_identifier(name, value)
        if self.mode not in {"observed_proxy", "manipulated", "held_constant"}:
            raise SystemCellContractError("horizon factor mode is unsupported")
        if not self.levels or any(not value or value != value.strip() for value in self.levels):
            raise SystemCellContractError("horizon factor levels must be canonical")
        if len(self.levels) != len(set(self.levels)):
            raise SystemCellContractError("horizon factor levels must be unique")
        _require_sha256("source_contract_sha256", self.source_contract_sha256)
        if not isinstance(self.intrinsic_horizon, bool):
            raise SystemCellContractError("intrinsic_horizon must be boolean")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HorizonFactorV1:
        keys = {
            "schema",
            "factor_id",
            "mode",
            "operationalization_id",
            "unit",
            "levels",
            "source_contract_sha256",
            "intrinsic_horizon",
        }
        _expect_exact_keys(payload, keys, "horizon factor")
        return cls(
            schema=str(payload["schema"]),
            factor_id=str(payload["factor_id"]),
            mode=str(payload["mode"]),
            operationalization_id=str(payload["operationalization_id"]),
            unit=str(payload["unit"]),
            levels=_string_tuple(payload["levels"], "horizon factor levels"),
            source_contract_sha256=str(payload["source_contract_sha256"]),
            intrinsic_horizon=_boolean(payload["intrinsic_horizon"], "intrinsic_horizon"),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "factor_id": self.factor_id,
            "mode": self.mode,
            "operationalization_id": self.operationalization_id,
            "unit": self.unit,
            "levels": list(self.levels),
            "source_contract_sha256": self.source_contract_sha256,
            "intrinsic_horizon": self.intrinsic_horizon,
        }


_HORIZON_FACTOR_FIELDS = (
    "goal_distance",
    "dependency_depth",
    "interaction_cap",
    "context_budget",
    "interruption",
    "actor_competence",
    "memory_runtime",
    "action_objective",
    "expert_trace_length_proxy",
)


@dataclass(frozen=True)
class HorizonFactorManifestV1:
    manifest_id: str
    goal_distance: HorizonFactorV1
    dependency_depth: HorizonFactorV1
    interaction_cap: HorizonFactorV1
    context_budget: HorizonFactorV1
    interruption: HorizonFactorV1
    actor_competence: HorizonFactorV1
    memory_runtime: HorizonFactorV1
    action_objective: HorizonFactorV1
    expert_trace_length_proxy: HorizonFactorV1
    intrinsic_horizon_claimed: bool
    schema: str = HORIZON_FACTOR_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HORIZON_FACTOR_MANIFEST_SCHEMA:
            raise SystemCellContractError("unsupported horizon-factor manifest schema")
        _require_identifier("manifest_id", self.manifest_id)
        for field_name in _HORIZON_FACTOR_FIELDS:
            factor = getattr(self, field_name)
            if factor.factor_id != field_name:
                raise SystemCellContractError(
                    f"horizon factor {field_name} has mismatched factor_id"
                )
        proxy = self.expert_trace_length_proxy
        if (
            proxy.mode != "observed_proxy"
            or proxy.intrinsic_horizon
            or proxy.operationalization_id != "expert_low_level_trace_length_tercile_v1"
            or proxy.levels != ("short", "medium", "long")
        ):
            raise SystemCellContractError(
                "expert trace-length labels must remain a non-intrinsic observed proxy"
            )
        if not isinstance(self.intrinsic_horizon_claimed, bool):
            raise SystemCellContractError("intrinsic_horizon_claimed must be boolean")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HorizonFactorManifestV1:
        keys = {"schema", "manifest_id", "intrinsic_horizon_claimed", *_HORIZON_FACTOR_FIELDS}
        _expect_exact_keys(payload, keys, "horizon-factor manifest")
        factors = {
            name: HorizonFactorV1.from_dict(_mapping(payload[name], name))
            for name in _HORIZON_FACTOR_FIELDS
        }
        return cls(
            schema=str(payload["schema"]),
            manifest_id=str(payload["manifest_id"]),
            intrinsic_horizon_claimed=_boolean(
                payload["intrinsic_horizon_claimed"],
                "intrinsic_horizon_claimed",
            ),
            **factors,
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            **{name: getattr(self, name).to_dict() for name in _HORIZON_FACTOR_FIELDS},
            "intrinsic_horizon_claimed": self.intrinsic_horizon_claimed,
        }


def load_system_cell_manifest(path: str | Path) -> SystemCellManifestV1:
    return SystemCellManifestV1.from_dict(_read_mapping(Path(path)))


def load_horizon_factor_manifest(path: str | Path) -> HorizonFactorManifestV1:
    return HorizonFactorManifestV1.from_dict(_read_mapping(Path(path)))


def _require_component_config(
    component: HashBoundComponentV1,
    label: str,
    required: set[str],
) -> None:
    missing = sorted(required - set(component.config))
    if missing:
        raise SystemCellContractError(f"{label} config is missing fields: {missing}")


def _expect_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise SystemCellContractError(
            f"{label} fields differ: missing={missing}, unknown={unknown}"
        )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemCellContractError(f"{label} must be a mapping")
    return dict(value)


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SystemCellContractError(
            f"cannot read system-cell artifact {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise SystemCellContractError(f"system-cell artifact is not a mapping: {path}")
    return dict(value)


def _require_identifier(name: str, value: str) -> None:
    if not value or value != value.strip() or any(character in value for character in "\r\n"):
        raise SystemCellContractError(f"{name} must be a canonical single-line identifier")


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SystemCellContractError(f"{name} must be a lowercase SHA-256")


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemCellContractError(f"{label} must be an integer")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemCellContractError(f"{label} must be numeric")
    return float(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SystemCellContractError(f"{label} must be boolean")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SystemCellContractError(f"{label} must be a sequence")
    return tuple(str(item) for item in value)


def _integer_tuple(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SystemCellContractError(f"{label} must be a sequence")
    return tuple(_integer(item, label) for item in value)


__all__ = [
    "HASH_BOUND_COMPONENT_SCHEMA",
    "HORIZON_FACTOR_MANIFEST_SCHEMA",
    "HORIZON_FACTOR_SCHEMA",
    "POLICY_IDENTITY_SCHEMA",
    "SAMPLING_POLICY_SCHEMA",
    "SYSTEM_CELL_MANIFEST_SCHEMA",
    "HashBoundComponentV1",
    "HorizonFactorManifestV1",
    "HorizonFactorV1",
    "PolicyIdentityV1",
    "SamplingPolicyV1",
    "SystemCellContractError",
    "SystemCellManifestV1",
    "canonical_sha256",
    "load_horizon_factor_manifest",
    "load_system_cell_manifest",
]
