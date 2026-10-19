"""Child entrypoint for the isolated veRL backend."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from lh_platform.training.backends.verl_source import (
    GitSourceContract,
    SourceFilePin,
    verify_git_source,
)


def main() -> int:
    policy = None
    pending = None
    source_receipt = None
    for raw_line in sys.stdin.buffer:
        request = json.loads(raw_line)
        request_id = request.get("request_id")
        operation = request.get("operation")
        payload = request.get("payload")
        try:
            if operation == "hello":
                if policy is not None:
                    raise RuntimeError("veRL worker was initialized twice")
                source_receipt, policy = _initialize(payload)
                from lh_platform.training.backends.verl_transformers_policy import (
                    VERL_TRANSFORMERS_ADAPTER_REVISION,
                )

                result = {
                    "adapter_revision": (
                        VERL_TRANSFORMERS_ADAPTER_REVISION + "@" + source_receipt.commit_sha[:12]
                    ),
                    "checkpoint_sha256": policy.checkpoint_sha256,
                    "component": asdict(policy.component),
                    "adapter_initial_delta_zero": bool(policy.adapter_initial_delta_zero),
                    "model_io_profile_sha256": policy.model_io_profile_sha256,
                }
            elif operation == "sample_actions":
                _require_initialized(policy)
                if pending is not None:
                    raise RuntimeError("veRL worker cannot sample with a pending batch")
                from lh_platform.training.agentic_rl_runtime import PolicyActionRequest

                requests = tuple(
                    PolicyActionRequest(**_normalize_action_request(row))
                    for row in payload["requests"]
                )
                result = {
                    "samples": [
                        asdict(sample)
                        for sample in policy.sample_actions(
                            requests,
                            seed=payload["seed"],
                        )
                    ]
                }
            elif operation == "submit":
                _require_initialized(policy)
                if pending is not None:
                    raise RuntimeError("veRL worker already has a pending batch")
                from lh_platform.training.experience_bridge import (
                    ExperienceRecord,
                    experience_batch_sha256,
                    validate_experience_records,
                )

                pending = validate_experience_records(
                    tuple(ExperienceRecord(**_normalize_record(row)) for row in payload["records"])
                )
                digest = experience_batch_sha256(pending)
                if digest != payload.get("batch_sha256"):
                    raise RuntimeError("veRL worker received a different batch hash")
                result = {"batch_sha256": digest}
            elif operation == "update":
                _require_initialized(policy)
                if pending is None:
                    raise RuntimeError("veRL worker has no pending batch")
                result = asdict(policy.apply_experience_records(pending))
                pending = None
            elif operation == "checkpoint":
                _require_initialized(policy)
                if pending is not None:
                    raise RuntimeError("veRL worker cannot checkpoint a pending batch")
                result = asdict(policy.materialize_checkpoint(Path(payload["output_dir"])))
            elif operation == "close":
                if policy is not None:
                    close = getattr(policy, "close", None)
                    if callable(close):
                        close()
                result = {"closed": True}
            else:
                raise RuntimeError("unknown veRL worker operation")
            _write_response(request_id, ok=True, result=result)
            if operation == "close":
                return 0
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            _write_response(
                request_id,
                ok=False,
                error_type=type(error).__name__,
                error_message=str(error),
            )
    return 0


def _initialize(payload):
    contract_payload = payload["source_contract"]
    contract = GitSourceContract(
        schema=contract_payload["schema"],
        backend_id=contract_payload["backend_id"],
        canonical_origin=contract_payload["canonical_origin"],
        commit_sha=contract_payload["commit_sha"],
        tree_sha=contract_payload["tree_sha"],
        files=tuple(SourceFilePin(**row) for row in contract_payload["files"]),
    )
    receipt = verify_git_source(payload["source_root"], contract)
    if receipt.contract_sha256 != payload["source_receipt_contract_sha256"]:
        raise RuntimeError("veRL parent/child source contract drifted")
    source_root = Path(receipt.source_root)
    if any(name == "verl" or name.startswith("verl.") for name in sys.modules):
        raise RuntimeError("veRL was imported before source verification")
    sys.path.insert(0, str(source_root))
    from verl.trainer.ppo import core_algos

    module_path = Path(core_algos.__file__).resolve()
    if not module_path.is_relative_to(source_root):
        raise RuntimeError("veRL API resolved outside the verified source root")
    policy_payload = payload["policy"]
    if policy_payload.get("kind") != "transformers_peft":
        raise RuntimeError("unsupported veRL worker policy kind")
    from lh_platform.training.backends.verl_transformers_policy import (
        VerLTransformersPolicy,
    )
    from lh_platform.training.transformers_peft_policy import (
        TransformersPeftPolicyConfig,
    )

    config_payload = dict(policy_payload["config"])
    for field in ("lora_target_modules", "lora_exclude_modules"):
        if field in config_payload:
            config_payload[field] = tuple(config_payload[field])
    return receipt, VerLTransformersPolicy(TransformersPeftPolicyConfig(**config_payload))


def _normalize_record(payload):
    normalized = dict(payload)
    for field in (
        "prompt_ids",
        "response_ids",
        "response_mask",
        "response_old_logprobs",
        "allowed_response_token_ids",
    ):
        if field in normalized:
            normalized[field] = tuple(normalized[field])
    normalized["allowed_response_token_ids"] = tuple(
        tuple(row) for row in normalized.get("allowed_response_token_ids", ())
    )
    return normalized


def _normalize_action_request(payload):
    normalized = dict(payload)
    conversation = normalized.get("conversation")
    if conversation is not None:
        from lh_platform.training.model_io import ChatMessage, ConversationIR

        normalized["conversation"] = ConversationIR(
            messages=tuple(ChatMessage(**message) for message in conversation["messages"]),
            tools=tuple(conversation.get("tools", ())),
            documents=tuple(conversation.get("documents", ())),
        )
    normalized["admissible_actions"] = tuple(normalized.get("admissible_actions", ()))
    normalized["response_stop_texts"] = tuple(normalized.get("response_stop_texts", ()))
    return normalized


def _require_initialized(policy) -> None:
    if policy is None:
        raise RuntimeError("veRL worker is not initialized")


def _write_response(request_id, *, ok: bool, result=None, **error) -> None:
    payload = {"request_id": request_id, "ok": ok}
    if ok:
        payload["result"] = result
    else:
        payload.update(error)
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
