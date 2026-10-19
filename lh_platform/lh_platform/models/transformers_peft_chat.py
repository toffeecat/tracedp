"""ModelBase adapter for the platform-native in-process Transformers policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from lh_platform.core.system_cell import SystemCellManifestV1
from lh_platform.models.base import Message, ModelBase, ModelOutput
from lh_platform.training.agentic_rl_runtime import (
    OnPolicyActionSample,
    PolicyActionRequest,
)
from lh_platform.training.model_io import (
    ChatMessage,
    ConversationIR,
    model_io_profile_by_id,
)
from lh_platform.training.transformers_peft_policy import (
    TransformersPeftPolicy,
    TransformersPeftPolicyConfig,
)


class _FrozenTransformersPolicy(Protocol):
    config: Any
    checkpoint_sha256: str
    model_io_profile_sha256: str | None
    chat_template_sha256: str | None
    component: Any
    optimizer: Any

    def sample_actions(
        self,
        requests: tuple[PolicyActionRequest, ...],
        *,
        seed: int,
    ) -> tuple[OnPolicyActionSample, ...]: ...

    def runtime_policy_identity_sha256(self) -> str: ...


class TransformersPeftChatModel(ModelBase):
    """Expose ``TransformersPeftPolicy.sample_actions`` through ``ModelBase``.

    The adapter owns one generation-session ledger but not the policy weights.
    Multiple fresh adapters may share one frozen in-process policy, matching the
    single-load execution path used by the AppWorld calibration runner.
    """

    name = "transformers_peft_chat"

    def __init__(
        self,
        *,
        policy: _FrozenTransformersPolicy,
        model_id: str,
        expected_checkpoint_sha256: str,
        seed: int,
        response_stop_texts: tuple[str, ...] = (),
    ) -> None:
        if not str(model_id).strip():
            raise ValueError("in-process model identity must be non-empty")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("in-process generation seed must be a non-negative integer")
        if not bool(getattr(policy.config, "inference_only", False)):
            raise ValueError("in-process chat adapter requires an inference-only policy")
        if getattr(policy, "optimizer", None) is not None:
            raise ValueError("in-process chat adapter forbids an optimizer")
        if str(policy.checkpoint_sha256) != str(expected_checkpoint_sha256):
            raise ValueError("loaded policy checkpoint does not match runtime binding")
        if any(not str(value) for value in response_stop_texts):
            raise ValueError("response stop texts must be non-empty")

        raw_profile = getattr(policy.config, "model_io_profile", None)
        if not isinstance(raw_profile, Mapping) or not str(raw_profile.get("profile_id", "")):
            raise ValueError("in-process chat adapter requires an explicit model I/O profile")

        self.policy = policy
        self.model_name = str(model_id)
        self.expected_checkpoint_sha256 = str(expected_checkpoint_sha256)
        self.seed = seed
        self.response_stop_texts = tuple(str(value) for value in response_stop_texts)
        self.model_io_profile_id = str(raw_profile["profile_id"])
        self._generation_count = 0
        self._generation_error_count = 0

    def generate(self, messages: list[Message], **kwargs: object) -> ModelOutput:
        if not messages:
            raise ValueError("in-process generation requires at least one message")
        configured_max_tokens = int(self.policy.config.max_new_tokens)
        requested_max_tokens = int(kwargs.get("max_tokens", configured_max_tokens))
        if requested_max_tokens != configured_max_tokens:
            raise ValueError("requested max_tokens differs from the frozen policy config")
        raw_temperature = (
            kwargs["temperature"] if "temperature" in kwargs else self.policy.config.temperature
        )
        temperature = float(raw_temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("in-process generation temperature must be finite and positive")
        requested_seed = kwargs.get("seed", self.seed)
        if isinstance(requested_seed, bool) or not isinstance(requested_seed, int):
            raise ValueError("in-process generation seed must be an integer")
        if requested_seed < 0:
            raise ValueError("in-process generation seed must be non-negative")

        conversation = ConversationIR(
            messages=tuple(
                ChatMessage(role=message.role, content=message.content) for message in messages
            )
        )
        request = PolicyActionRequest(
            traj_uid=f"modelbase:{self._generation_instance_id()}",
            step_idx=self._generation_count,
            conversation=conversation,
            model_io_profile_id=self.model_io_profile_id,
            sampling_temperature=temperature,
            sampling_seed=requested_seed,
            response_stop_texts=self.response_stop_texts,
        )
        try:
            samples = self.policy.sample_actions((request,), seed=requested_seed)
        except Exception:
            self._generation_error_count += 1
            raise
        if len(samples) != 1:
            self._generation_error_count += 1
            raise RuntimeError("in-process policy must return exactly one sample")
        sample = samples[0]
        if sample.policy_checkpoint_sha256 != self.expected_checkpoint_sha256:
            self._generation_error_count += 1
            raise RuntimeError("in-process sample checkpoint identity changed")

        self._generation_count += 1
        output = ModelOutput(
            text=sample.action_text,
            prompt_tokens=len(sample.prompt_token_ids),
            completion_tokens=len(sample.action_token_ids),
            raw=sample,
            info={
                "backend": "transformers_peft_in_process",
                "requested_model": self.model_name,
                "served_model": self.model_name,
                "policy_checkpoint_sha256": sample.policy_checkpoint_sha256,
                "runtime_policy_identity_sha256": (self.policy.runtime_policy_identity_sha256()),
                "rollout_seed": sample.rollout_seed,
                "model_io_profile_sha256": sample.model_io_profile_sha256,
                "normalized_messages_sha256": sample.normalized_messages_sha256,
                "chat_template_sha256": sample.chat_template_sha256,
                "dummy_fallback_used": False,
            },
        )
        return self._record_generation_event(
            messages,
            output,
            outcome="in_process_success",
            backend="transformers_peft_in_process",
            requested_model=self.model_name,
        )

    def runtime_evidence(self) -> dict[str, Any]:
        return {
            "backend": "transformers_peft_in_process",
            "model_instantiated": True,
            "requested_model": self.model_name,
            "generation_count": self._generation_count + self._generation_error_count,
            "generation_error_count": self._generation_error_count,
            "model_identity_ids": [self.model_name],
            "sampling_seed": self.seed,
            "policy_checkpoint_sha256": self.policy.checkpoint_sha256,
            "model_io_profile_sha256": self.policy.model_io_profile_sha256,
            "chat_template_sha256": self.policy.chat_template_sha256,
            "backend_component_sha256": self.policy.component.sha256,
            "runtime_policy_identity_sha256": self.policy.runtime_policy_identity_sha256(),
            "inference_only": bool(self.policy.config.inference_only),
            "optimizer_present": self.policy.optimizer is not None,
            "dummy_model_used": False,
            "dummy_fallback_used": False,
        }


class FrozenTransformersChatRuntime:
    """Load and bind one frozen policy, then mint independent chat sessions."""

    def __init__(
        self,
        *,
        policy: TransformersPeftPolicy,
        model_id: str,
        checkpoint_sha256: str,
        seed: int,
        response_stop_texts: tuple[str, ...],
    ) -> None:
        self._policy = policy
        self._model_id = model_id
        self._checkpoint_sha256 = checkpoint_sha256
        self._seed = seed
        self._response_stop_texts = response_stop_texts
        self._closed = False

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        system_cell: SystemCellManifestV1,
        model_id: str,
        device: str,
        seed: int,
    ) -> FrozenTransformersChatRuntime:
        identity = system_cell.policy_identity
        sampling = system_cell.sampling_policy
        runtime = system_cell.runtime.config
        if identity.backend_id != "transformers_frozen_causal_lm":
            raise ValueError(
                "in-process execution requires transformers_frozen_causal_lm System Cell"
            )
        canonical_model_id = f"{identity.model_reference}@{identity.model_revision}"
        if model_id != canonical_model_id:
            raise ValueError("in-process model identity differs from the System Cell")
        profile = model_io_profile_by_id(identity.model_io_profile_id)
        if profile.sha256 != identity.model_io_profile_sha256:
            raise ValueError("System Cell model I/O profile hash changed")
        if runtime.get("kernel_profile") != "sdpa_reference":
            raise ValueError("in-process self-trace currently requires sdpa_reference")
        quantization = str(runtime.get("quantization", ""))
        if quantization not in {"none", "bitsandbytes_nf4"}:
            raise ValueError("System Cell quantization is missing or unsupported")
        config = TransformersPeftPolicyConfig(
            model_path=str(Path(checkpoint_path).resolve()),
            base_checkpoint_sha256=checkpoint_sha256,
            model_revision="local",
            device=device,
            dtype=identity.dtype,
            max_prompt_tokens=int(system_cell.context.config["max_prompt_tokens"]),
            max_new_tokens=sampling.max_response_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            repetition_penalty=sampling.repetition_penalty,
            apply_chat_template=False,
            generation_micro_batch_size=1,
            quantization=quantization,
            attn_implementation="sdpa",
            initialization_seed=seed,
            model_io_profile=profile.to_dict(),
            model_io_profile_sha256=profile.sha256,
            inference_only=True,
        )
        policy = TransformersPeftPolicy(config)
        try:
            if policy.checkpoint_sha256 != checkpoint_sha256:
                raise ValueError("loaded in-process policy checkpoint identity changed")
            if policy.model_io_profile_sha256 != identity.model_io_profile_sha256:
                raise ValueError("loaded in-process model I/O profile identity changed")
            if policy.chat_template_sha256 != runtime.get("chat_template_sha256"):
                raise ValueError("loaded in-process chat template identity changed")
            if policy.component.sha256 != runtime.get("backend_component_sha256"):
                raise ValueError("loaded in-process backend component identity changed")
            if policy.runtime_policy_identity_sha256() != runtime.get(
                "runtime_policy_identity_sha256"
            ):
                raise ValueError("loaded in-process runtime policy identity changed")
            if policy.optimizer is not None or any(
                parameter.requires_grad for parameter in policy.model.parameters()
            ):
                raise ValueError("in-process self-trace policy is not frozen")
        except Exception:
            policy.close()
            raise
        return cls(
            policy=policy,
            model_id=model_id,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            response_stop_texts=sampling.stop_texts,
        )

    def new_session(self) -> TransformersPeftChatModel:
        if self._closed:
            raise RuntimeError("in-process chat runtime is closed")
        return TransformersPeftChatModel(
            policy=self._policy,
            model_id=self._model_id,
            expected_checkpoint_sha256=self._checkpoint_sha256,
            seed=self._seed,
            response_stop_texts=self._response_stop_texts,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._policy.close()


__all__ = ["FrozenTransformersChatRuntime", "TransformersPeftChatModel"]
