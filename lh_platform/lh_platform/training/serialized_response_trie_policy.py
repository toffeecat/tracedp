"""Experiment-local serialized-response trie sampling for Transformers policies."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from lh_platform.core.evidence_contract import canonical_sha256
from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    OnPolicyActionSample,
    PolicyActionRequest,
    validate_action_sample_for_request,
)
from lh_platform.training.model_io import ConversationIR
from lh_platform.training.transformers_peft_policy import TransformersPeftPolicy

SERIALIZED_RESPONSE_TRIE_COMPONENT_SCHEMA = "lh_serialized_response_trie_component/v1"


def serialized_response_trie_component() -> dict[str, str]:
    """Return the immutable identity of the experiment-only decoder treatment."""

    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    descriptor = {
        "schema": SERIALIZED_RESPONSE_TRIE_COMPONENT_SCHEMA,
        "implementation_source_sha256": source_sha256,
        "base_sampler": "TransformersPeftPolicy._sample_constrained_action",
        "candidate_language": "complete_response_contract_strings",
        "termination": "checkpoint_native_turn_end",
    }
    descriptor["component_sha256"] = "sha256:" + canonical_sha256(descriptor)
    return descriptor


class SerializedResponseTrieTransformersPolicy(TransformersPeftPolicy):
    """Add one explicit candidate-response sampler without changing the base runtime."""

    def _native_response_termination_ids(
        self,
        request: PolicyActionRequest,
    ) -> tuple[int, ...]:
        renderer = getattr(self, "_model_io_renderer", None)
        if renderer is not None:
            conversation = request.conversation
            if conversation is None:
                conversation = ConversationIR.from_legacy_user_prompt(str(request.prompt_text))
            return renderer.native_turn_end_ids(conversation)
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise AgenticRLContractError(
                "serialized-response trie sampling requires a native termination token"
            )
        return (int(eos),)

    def sample_serialized_response(
        self,
        request: PolicyActionRequest,
        *,
        constrained_response_candidates: Sequence[str],
        seed: int,
    ) -> OnPolicyActionSample:
        """Sample one complete response-contract string from a frozen token trie."""

        candidates = tuple(str(value) for value in constrained_response_candidates)
        if (
            request.admissible_actions
            or not candidates
            or any(not value.strip() for value in candidates)
            or len(candidates) != len(set(candidates))
        ):
            raise AgenticRLContractError(
                "serialized-response trie candidates must be unique non-empty strings"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AgenticRLContractError("serialized-response sampling seed is invalid")
        local_seed = request.sampling_seed if request.sampling_seed is not None else seed
        temperature = self._request_temperature(request)
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            torch.manual_seed(local_seed)
            if self._device.type == "cuda":
                torch.cuda.manual_seed_all(local_seed)
            rendered = self._render_action_request(request)
            if len(rendered.input_ids) > self.config.max_prompt_tokens:
                raise AgenticRLContractError("serialized-response prompt exceeds max_prompt_tokens")
            input_ids = torch.tensor(
                [rendered.input_ids],
                dtype=torch.long,
                device=self._device,
            )
            (
                response_text,
                response_ids,
                old_log_probs,
                allowed_token_ids,
            ) = self._sample_constrained_action(
                input_ids,
                candidates,
                termination_ids=self._native_response_termination_ids(request),
                temperature=temperature,
            )
        sample = OnPolicyActionSample(
            traj_uid=request.traj_uid,
            step_idx=request.step_idx,
            action_text=response_text,
            prompt_token_ids=rendered.input_ids,
            action_token_ids=response_ids,
            old_action_token_log_probs=old_log_probs,
            policy_checkpoint_sha256=self.checkpoint_sha256,
            rollout_seed=local_seed,
            response_mask=(1,) * len(response_ids),
            model_io_profile_sha256=rendered.model_io_profile_sha256,
            response_contract_sha256=request.response_contract_sha256,
            normalized_messages_sha256=rendered.normalized_messages_sha256,
            chat_template_sha256=rendered.chat_template_sha256,
            matched_stop_text=None,
            finish_reason="constrained_response",
            sampling_temperature=temperature,
            allowed_action_token_ids=allowed_token_ids,
        )
        validate_action_sample_for_request(
            request,
            sample,
            expected_model_io_profile_sha256=getattr(
                self,
                "model_io_profile_sha256",
                None,
            ),
        )
        self._samples.append(sample)
        return sample


__all__ = [
    "SERIALIZED_RESPONSE_TRIE_COMPONENT_SCHEMA",
    "SerializedResponseTrieTransformersPolicy",
    "serialized_response_trie_component",
]
