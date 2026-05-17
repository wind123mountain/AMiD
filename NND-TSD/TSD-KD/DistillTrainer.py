# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# Modified by adding NNM (Nuclear Norm Matching) regularizer.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import textwrap
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available, is_sagemaker_mp_enabled

from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.gkd_config import GKDConfig
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    generate_model_card,
    get_comet_experiment_url,
    entropy_from_logits
)

# ═══════════════════════════════════════════════════════════════
#  NNM: import the regularizer module (must live next to this file)
# ═══════════════════════════════════════════════════════════════
from nnm_module import compute_nnm_loss

if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


def normalize_entropy(entropy):
    total_entropy = entropy.sum(dim=-1, keepdim=True)  # [B, 1]
    norm_entropy = entropy / total_entropy             # [B, T]
    return norm_entropy


def top_k_entropy_position(norm_entropy, top_p=0.1):
    """
    norm_entropy: [B, T] (normalized so that sum = 1)
    Returns: [B] index where cumulative entropy surpasses (1 - top_p)
    """
    cum_entropy = torch.cumsum(norm_entropy, dim=-1)  # [B, T]
    threshold = top_p

    mask = cum_entropy >= threshold                    # [B, T]
    first_idx = mask.float().argmax(dim=-1)            # [B]
    return first_idx


def indirect_kd_loss(
    student_logits, teacher_logits, mask=None, T=1.0, K=10
):
    B, L, V = student_logits.shape

    # 1. Temperature scaling
    student_logits = student_logits / T
    teacher_logits = teacher_logits / T

    # 2. Get student Top-K candidate indices
    student_topk_logits, student_topk_indices = torch.topk(student_logits, K, dim=2)

    # 3. Get teacher logits on student-selected candidates
    teacher_topk_logits = torch.gather(teacher_logits, dim=2, index=student_topk_indices)

    # 4. Get permutation (descending) of teacher logits on those candidates
    teacher_sorted_idx = torch.argsort(teacher_topk_logits, dim=2, descending=True)  # [B, L, K]

    # 5. Apply that permutation to student logits (i.e., sort student_topk_logits by teacher ranking)
    batch_indices = torch.arange(B).view(B, 1, 1).expand(B, L, K)
    seq_indices = torch.arange(L).view(1, L, 1).expand(B, L, K)
    sorted_student_logits = student_topk_logits[
        batch_indices, seq_indices, teacher_sorted_idx
    ]  # [B, L, K], sorted by teacher preference

    # 6. Mean normalization (optional)
    sorted_student_logits = sorted_student_logits - sorted_student_logits.mean(dim=2, keepdim=True)

    # 7. Compute loss: -sum log-softmax
    log_cumsum_exp = torch.logcumsumexp(sorted_student_logits, dim=2)
    log_probs = sorted_student_logits - log_cumsum_exp
    loss = -log_probs.sum(dim=2)  # [B, L]

    # 8. Apply mask if provided
    if mask is not None:
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-8)
    else:
        loss = loss.mean()

    return loss


class DistillTrainer(SFTTrainer):
    _tag_names = ["trl", "tsd-kd", "nnm"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        teacher_model: Union[PreTrainedModel, nn.Module, str] = None,
        args: Optional[GKDConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
        token_entropy_percentile_threshold: Optional[float] = 1.0,
        indirect_kd_alpha: Optional[float] = 0.1,
        # ═══════════════════════════════════════════════════════════════
        #  NNM kwargs (all optional — NNM disabled if nnm_state is None)
        # ═══════════════════════════════════════════════════════════════
        nnm_state: Optional[dict] = None,
        nnm_ratio: float = 0.1,
        nnm_warmup_steps: int = 0,
        nnm_ramp_steps: int = 0,
        nnm_ns_iters: int = 5,
    ):
        # add remove_unused_columns=False to the dataclass args
        args.remove_unused_columns = False
        data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["torch_dtype"] = (
                teacher_model_init_kwargs["torch_dtype"]
                if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            )

        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        self.token_entropy_percentile_threshold = token_entropy_percentile_threshold
        self.indirect_kd_alpha = indirect_kd_alpha
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_k=0,
            use_cache=False if args.gradient_checkpointing else True,
            pad_token_id=self.processing_class.pad_token_id,
        )
        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # ═══════════════════════════════════════════════════════════════
        #  NNM: bind state + schedule. projectors were attached onto the
        #  model BEFORE Trainer.__init__ so the parent class wraps them
        #  with DeepSpeed/accelerator like the rest of the model.
        # ═══════════════════════════════════════════════════════════════
        self.nnm_state = nnm_state
        self.nnm_ratio = nnm_ratio
        self.nnm_warmup_steps = nnm_warmup_steps
        self.nnm_ramp_steps = nnm_ramp_steps
        self.nnm_ns_iters = nnm_ns_iters
        # populated lazily by training_step (counts optimizer steps)
        self._nnm_global_step = 0
        if self.nnm_state is not None:
            first_lid = self.nnm_state["s_mid"][0]
            K = self.nnm_state["t_centroids"][first_lid].shape[0]
            print(f"[NNM] enabled: ratio={self.nnm_ratio}, "
                  f"warmup={self.nnm_warmup_steps}, ramp={self.nnm_ramp_steps}, "
                  f"K={K}, layers={self.nnm_state['s_mid']}")
            
    def create_optimizer(self):
        proj_lr = getattr(self.args, 'proj_lr', None)
        if proj_lr is None or proj_lr < 0:
            proj_lr = self.args.learning_rate
        self.args.proj_lr = proj_lr

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {"params": list(self.projectors.parameters()), "lr": proj_lr}
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        manager.register_module_override(module, "weight", {"optim_bits": 32})

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


    # ═══════════════════════════════════════════════════════════════
    #  NNM helpers
    # ═══════════════════════════════════════════════════════════════
    def _nnm_effective_ratio(self) -> float:
        """Warmup + linear ramp schedule, keyed on _nnm_global_step."""
        if self.nnm_state is None or self.nnm_ratio <= 0.0:
            return 0.0
        step   = self._nnm_global_step
        warmup = self.nnm_warmup_steps
        ramp   = self.nnm_ramp_steps
        target = self.nnm_ratio
        if step < warmup:
            return 0.0
        if ramp <= 0:
            return target
        progress = (step - warmup) / ramp
        return target if progress >= 1.0 else target * progress

    def _get_student_unwrapped(self, model):
        m = model
        while hasattr(m, "module"):
            m = m.module
        return m

    @staticmethod
    def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.9, temperature=1.0, reduction="batchmean", token_entropy_percentile_threshold=0.1, indirect_kd_alpha=0.1
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.
        """
        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        beta = torch.tensor(beta, dtype=student_log_probs.dtype)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log(1 - beta), teacher_log_probs + torch.log(beta)]),
            dim=0,
        )

        if labels is not None:
            mask = labels != -100

        entropies = entropy_from_logits(student_logits)
        entropies_teacher = entropy_from_logits(teacher_logits)
        entropy_diff = entropies - entropies_teacher
        weight = torch.sigmoid(entropy_diff).detach() # stop gradient
        non_pad_entropies = entropies[mask]

        if non_pad_entropies.numel() == 0:
            entropy_threshold = float("inf")
            entropy_mask = torch.zeros_like(entropies, dtype=torch.bool)
            d_token_len = 0
        else:
            entropy_threshold = torch.quantile(non_pad_entropies.float(), token_entropy_percentile_threshold)
            entropy_mask = entropies >= entropy_threshold
            normalized_entropies = normalize_entropy(non_pad_entropies.float())
            d_token_len = top_k_entropy_position(normalized_entropies, token_entropy_percentile_threshold)
        regularization = entropies * entropy_mask

        topk_loss = indirect_kd_loss(student_logits[:, :d_token_len, :], teacher_logits[:, :d_token_len, :], mask=labels[:, :d_token_len] != -100, K=10)

        if student_logits.size(1) > 0:
            # Compute the Generalized Jensen-Shannon Divergence
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            jsd = beta * kl_teacher + (1 - beta) * kl_student
            jsd = jsd * weight.unsqueeze(-1)
        else:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            loss =  jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (jsd.size(0) * jsd.size(1))
            loss += (regularization.sum() / entropy_mask.sum())
            loss += indirect_kd_alpha * topk_loss
            return loss
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ═══ NNM: decide BEFORE forward whether we need hidden states
        eff_ratio = self._nnm_effective_ratio()
        use_nnm = eff_ratio > 0.0

        # compute student output (output_hidden_states only when NNM is active)
        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=use_nnm,
        )

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=use_nnm,
            )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]

        # compute loss (TSD-KD: JSD + entropy reg + indirect KD)
        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
            token_entropy_percentile_threshold=self.token_entropy_percentile_threshold,
            indirect_kd_alpha=self.indirect_kd_alpha,
        )

        # ═══════════════════════════════════════════════════════════════
        #  NNM regularizer
        #
        #  Mask = response tokens only. We start from inputs["labels"] != -100
        #  and ALSO zero out the prompt span. The extra prompt-span masking
        #  matters because in on-policy steps (random.random() <= self.lmbda),
        #  training_step replaces inputs["labels"] with generated tokens that
        #  are NOT prompt-masked, so labels alone is not enough.
        # ═══════════════════════════════════════════════════════════════
        nnm_value = torch.tensor(0.0, device=loss.device)
        if use_nnm:
            student_unwrapped = self._get_student_unwrapped(model)
            projectors = student_unwrapped.projectors

            # Build NNM-specific labels: copy of inputs["labels"] with the
            # prompt span explicitly zeroed (-100). Keeps the original
            # tensor intact for TSD-KD code paths.
            nnm_labels = inputs["labels"].clone()
            nnm_labels[:, :prompt_lengths] = -100

            nnm_value = compute_nnm_loss(
                projectors=projectors,
                s_hidden_states=outputs_student.hidden_states,
                t_hidden_states=outputs_teacher.hidden_states,
                labels=nnm_labels,
                student_layer_mapping=self.nnm_state["s_mid"],
                teacher_layer_mapping=self.nnm_state["t_mid"],
                t_centroids=self.nnm_state["t_centroids"],
                R=self.nnm_state["R"],
                layer_weights=self.nnm_state["layer_weights"],
                ns_iters=self.nnm_ns_iters,
            )
            loss = loss + eff_ratio * nnm_value

            # stash for the next log()
            self._metrics_buffer = getattr(self, "_metrics_buffer", {})
            self._metrics_buffer["nnm_loss"] = nnm_value.detach().float().item()
            self._metrics_buffer["nnm_eff_ratio"] = eff_ratio

        return (loss, outputs_student) if return_outputs else loss

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        # Generate output with respect to the prompt only
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs.get("prompt_attention_mask", None),
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the original inputs.
        """
        if self.seq_kd:
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        if random.random() <= self.lmbda:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )

            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        loss = super().training_step(model, inputs, num_items_in_batch)
        # ═══ NNM: tick global step (we use HF's state.global_step, which
        # only advances on optimizer step boundaries — same semantics we want)
        self._nnm_global_step = self.state.global_step
        # push buffered NNM metrics into HF's log stream
        buf = getattr(self, "_metrics_buffer", None)
        if buf:
            self.log(buf)
            self._metrics_buffer = {}
        return loss

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        tags.update(self._tag_names)