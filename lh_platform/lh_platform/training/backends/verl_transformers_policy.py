"""Transformers/PEFT policy whose PPO scalar is computed by pinned veRL."""

from __future__ import annotations

import math

from lh_platform.training.agentic_rl_runtime import (
    AgenticRLContractError,
    PolicyUpdateResult,
)
from lh_platform.training.backends.verl_objective import (
    build_verl_objective_rows,
)
from lh_platform.training.experience_bridge import (
    ExperienceRecord,
    validate_experience_records,
)
from lh_platform.training.transformers_peft_policy import TransformersPeftPolicy

VERL_TRANSFORMERS_ADAPTER_REVISION = "verl_transformers_peft/v1"


class VerLTransformersPolicy(TransformersPeftPolicy):
    """Reuse model/token code while delegating clipped-PG math to veRL."""

    def apply_experience_records(
        self,
        records: tuple[ExperienceRecord, ...],
    ) -> PolicyUpdateResult:
        records = validate_experience_records(records)
        try:
            from omegaconf import OmegaConf
            from verl.trainer.ppo.core_algos import (
                agg_loss,
                compute_policy_loss_vanilla,
                kl_penalty,
            )
        except ImportError as error:
            raise AgenticRLContractError(
                "isolated veRL policy requires the pinned veRL runtime dependencies"
            ) from error

        torch = self._torch
        before_sha256 = self.checkpoint_sha256
        if any(row.policy_checkpoint_sha256 != before_sha256 for row in records):
            raise AgenticRLContractError(
                "experience batch was not sampled by the current veRL policy checkpoint"
            )
        before_parameters = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        objective_rows = build_verl_objective_rows(records)
        total_tokens = objective_rows.active_token_count
        config = OmegaConf.create(
            {
                "clip_ratio": self.config.clip_ratio,
                "clip_ratio_low": self.config.clip_ratio,
                "clip_ratio_high": self.config.clip_ratio,
                # LongHorizon's local clipped-PG does not use PPO dual clipping.
                "clip_ratio_c": 1.0e9,
                "global_batch_info": {
                    "dp_size": 1,
                    "batch_num_tokens": total_tokens,
                    "global_batch_size": len(records),
                    "loss_scale_factor": None,
                },
            }
        )

        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        policy_loss_value = 0.0
        reference_kl_value = 0.0
        clip_fraction_weighted = 0.0
        for start in range(0, len(records), self.config.update_micro_batch_size):
            micro_records = records[start : start + self.config.update_micro_batch_size]
            current_batch = self._teacher_forced_batch_log_probs(micro_records)
            current_log_probs = tuple(log_probs for log_probs, _entropy in current_batch)
            reference_batch = (
                self._reference_batch_log_probs(micro_records)
                if self.config.reference_kl_coef > 0
                else (None,) * len(micro_records)
            )
            old_rows = objective_rows.old_logprobs[start : start + len(micro_records)]
            mask_rows = objective_rows.response_masks[start : start + len(micro_records)]
            advantage_rows = objective_rows.token_advantages[start : start + len(micro_records)]
            old, current, advantages, mask = _pad_policy_rows(
                torch,
                old_rows=old_rows,
                current_rows=current_log_probs,
                advantage_rows=advantage_rows,
                mask_rows=mask_rows,
                device=self._device,
            )
            policy_loss, policy_metrics = compute_policy_loss_vanilla(
                old_log_prob=old,
                log_prob=current,
                advantages=advantages,
                response_mask=mask,
                loss_agg_mode="token-mean",
                config=config,
            )
            micro_loss = policy_loss
            policy_loss_value += float(policy_loss.detach().float().cpu())
            clip_fraction_weighted += float(policy_metrics["actor/pg_clipfrac"]) * int(
                mask.sum().item()
            )
            if self.config.reference_kl_coef > 0:
                reference = _pad_reference_rows(
                    torch,
                    reference_rows=reference_batch,
                    width=mask.shape[1],
                    device=self._device,
                )
                kl_matrix = kl_penalty(
                    logprob=current,
                    ref_logprob=reference,
                    kl_penalty=self.config.reference_kl_type,
                )
                kl_loss = agg_loss(
                    loss_mat=kl_matrix,
                    loss_mask=mask,
                    loss_agg_mode="token-mean",
                    **config.global_batch_info,
                )
                micro_loss = micro_loss + self.config.reference_kl_coef * kl_loss
                reference_kl_value += float(kl_loss.detach().float().cpu())
            micro_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self._optimizer_step += 1
        delta_sq = 0.0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            delta = parameter.detach().float().cpu() - before_parameters[name]
            delta_sq += float(torch.sum(delta * delta))
        delta_l2 = math.sqrt(delta_sq)
        self.checkpoint_sha256 = self._checkpoint_sha256()
        total_loss = policy_loss_value + self.config.reference_kl_coef * reference_kl_value
        return PolicyUpdateResult(
            optimizer_step=self._optimizer_step,
            checkpoint_before_sha256=before_sha256,
            checkpoint_after_sha256=self.checkpoint_sha256,
            trainable_parameter_delta_l2=delta_l2,
            metrics={
                "loss": total_loss,
                "objective": -policy_loss_value,
                "policy_objective": -policy_loss_value,
                "reference_kl_loss": reference_kl_value,
                "reference_kl_coef": self.config.reference_kl_coef,
                "grad_norm": float(grad_norm.detach().float().cpu()),
                "action_tokens": float(total_tokens),
                "pg_clipfrac": clip_fraction_weighted / total_tokens,
                "objective_mapping_revision": 1.0,
            },
        )


def _pad_policy_rows(
    torch,
    *,
    old_rows,
    current_rows,
    advantage_rows,
    mask_rows,
    device,
):
    width = max(len(row) for row in old_rows)
    batch = len(old_rows)
    old = torch.zeros((batch, width), dtype=torch.float32, device=device)
    current = torch.zeros((batch, width), dtype=torch.float32, device=device)
    advantages = torch.zeros((batch, width), dtype=torch.float32, device=device)
    mask = torch.zeros((batch, width), dtype=torch.bool, device=device)
    for index, (old_row, current_row, advantage_row, mask_row) in enumerate(
        zip(old_rows, current_rows, advantage_rows, mask_rows, strict=True)
    ):
        length = len(old_row)
        if int(current_row.numel()) != length:
            raise AgenticRLContractError("veRL current logprob width drifted")
        old[index, :length] = torch.tensor(old_row, dtype=torch.float32, device=device)
        current[index, :length] = current_row.float()
        advantages[index, :length] = torch.tensor(advantage_row, dtype=torch.float32, device=device)
        mask[index, :length] = torch.tensor(mask_row, dtype=torch.bool, device=device)
    return old, current, advantages, mask


def _pad_reference_rows(torch, *, reference_rows, width: int, device):
    reference = torch.zeros((len(reference_rows), width), dtype=torch.float32, device=device)
    for index, row in enumerate(reference_rows):
        if row is None:
            raise AgenticRLContractError("veRL reference row is missing")
        reference[index, : int(row.numel())] = row.float()
    return reference


__all__ = [
    "VERL_TRANSFORMERS_ADAPTER_REVISION",
    "VerLTransformersPolicy",
]
