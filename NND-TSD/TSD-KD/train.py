"""
TSD-KD + NNM training driver.

Pipeline:
  1. Load student & teacher (model paths configurable via CLI).
  2. Resize student embeddings to match teacher vocab (TSD-KD requirement).
  3. Load HF dataset and add "messages" column for ChatML collator.
  4. (NEW) NNM pre-pass: pick layer pairs, build random projection R,
     attach Linear(d_s -> d_t) projectors to the student, and run a
     teacher-only forward pass over a few batches to build per-layer
     centroids `C_t`.
  5. Construct DistillTrainer with TSD-KD args + NNM state, then train.

Run example (4-GPU H200, NNM ON):
    accelerate launch --config_file accelerate_ddp_config.yaml train.py \
        --beta 0.9 --lmbda 1.0 --threshold 0.1 --indirect-kd-alpha 0.1 \
        --student-model Qwen/Qwen2.5-1.5B-Instruct \
        --teacher-model Qwen/Qwen2.5-14B-Instruct \
        --dataset Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen \
        --nnm --nnm-ratio 0.1 --nnm-K 128 --nnm-n-layers 4 \
        --nnm-warmup-steps 200 --nnm-ramp-steps 100

LoRA variant (add):
    ... --use-lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.05

Disable NNM (pure TSD-KD baseline):
    ... --no-nnm
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GKDConfig

from DistillTrainer import DistillTrainer
from nnm_module import (
    make_R,
    layer_weight,
    select_mid_layers,
    build_teacher_centroids,
)

import torch._dynamo
torch._dynamo.config.disable = True


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def _print0(*a, **kw):
    if is_main_process():
        print(*a, **kw)


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()

    # ── TSD-KD core ──
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--lmbda", type=float, default=1.0,
                   help="Probability of on-policy generation per step.")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Token entropy percentile threshold.")
    p.add_argument("--indirect-kd-alpha", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=1.0)

    # ── Models ──
    p.add_argument("--student-model", type=str, required=True,
                   help="HF model id / path for the student.")
    p.add_argument("--teacher-model", type=str, required=True,
                   help="HF model id / path for the teacher.")
    p.add_argument("--dataset", type=str,
                   default="Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen")
    p.add_argument("--attn", type=str, default="sdpa")

    # ── HF Trainer / GKDConfig ──
    p.add_argument("--output-dir", type=str, default="./out")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-acc", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--report-to", type=str, default="wandb")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # ── NNM ──
    p.add_argument("--nnm", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable NNM regularizer. Default: enabled. "
                        "Use --no-nnm to disable.")
    p.add_argument("--nnm-ratio", type=float, default=0.1)
    p.add_argument("--nnm-n-layers", type=int, default=4)
    p.add_argument("--nnm-K", type=int, default=128)
    p.add_argument("--nnm-eta", type=float, default=0.05)
    p.add_argument("--nnm-T-dead", type=int, default=50)
    p.add_argument("--nnm-centroid-batches", type=int, default=300)
    p.add_argument("--nnm-d-prime", type=int, default=256)
    p.add_argument("--nnm-ns-iters", type=int, default=5)
    p.add_argument("--nnm-warmup-steps", type=int, default=0)
    p.add_argument("--nnm-ramp-steps", type=int, default=0)

    # ── LoRA / PEFT ──
    p.add_argument("--use-lora", action="store_true",
                   help="Wrap student with LoRA (PEFT). NNM projectors stay "
                        "fully trainable via modules_to_save.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated module names to apply LoRA to. "
                        "For Llama/Qwen attention: q_proj,k_proj,v_proj,o_proj. "
                        "Add gate_proj,up_proj,down_proj to cover MLP too.")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  NNM setup
# ═══════════════════════════════════════════════════════════════


def _batches_from_chatml_dataset(dataset, tokenizer, batch_size, max_length,
                                 max_batches):
    """
    Yield {input_ids, attention_mask, labels} for the NNM centroid pre-pass.

    The TSD-KD dataset has a "messages" column (chat format). We render with
    the tokenizer's chat template, tokenize, and mask prompt tokens to -100
    so centroids are built from response tokens only — the same masking the
    NNM loss uses at training time.
    """
    n = min(len(dataset), batch_size * max_batches)
    for i in range(0, n, batch_size):
        batch = dataset[i : i + batch_size]
        msgs = batch["messages"]

        # Full chat with assistant response → input_ids
        full_texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in msgs]
        enc = tokenizer(full_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)

        # Prompt only (drop assistant turn) → length per row
        prompt_lens = []
        for m in msgs:
            prompt_only = [t for t in m if t["role"] != "assistant"]
            p_text = tokenizer.apply_chat_template(
                prompt_only, tokenize=False, add_generation_prompt=True
            )
            p_ids = tokenizer(p_text, return_tensors="pt",
                              truncation=True, max_length=max_length).input_ids[0]
            prompt_lens.append(p_ids.shape[0])

        labels = enc.input_ids.clone()
        # mask prompt span
        for row, pl in enumerate(prompt_lens):
            labels[row, :pl] = -100
        # mask padding
        labels[enc.attention_mask == 0] = -100

        yield {
            "input_ids":      enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels":         labels,
        }


def _is_rank0():
    """Rank 0 of the env. Works before torch.distributed is initialized
    by reading the RANK env var that `accelerate launch` / `torchrun` set."""
    return int(os.environ.get("RANK", "0")) == 0


def _world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def prepare_nnm(args, student, teacher, train_dataset, tokenizer, device):
    """
    Returns the nnm_state dict consumed by DistillTrainer.

    Multi-GPU strategy:
      - All ranks pick layer mappings deterministically (same seed) and
        attach identical projectors to their student (so Trainer can later
        wrap them under DDP/Accelerate uniformly).
      - Only rank 0 loads the teacher to GPU and runs the centroid pre-pass.
      - Rank 0 writes {t_centroids, R} to a temp file. Other ranks wait,
        then load that file. We use file-based sharing (not torch.distributed
        broadcast) because dist is not yet initialized at this point —
        accelerate/Trainer initializes it inside Trainer.__init__.

    Steps:
      1) probe d_s, d_t; pick layer indices via select_mid_layers
      2) attach projectors to the student
      3) (rank 0) build teacher centroids and R; save to file
      4) (other ranks) wait + load
      5) per-layer weights
    """
    import time
    import pickle

    s_cfg = student.config
    t_cfg = teacher.config

    d_s = s_cfg.hidden_size
    d_t = t_cfg.hidden_size
    n_s_layers = s_cfg.num_hidden_layers
    n_t_layers = t_cfg.num_hidden_layers

    s_mid = select_mid_layers(n_s_layers, args.nnm_n_layers)
    t_mid = select_mid_layers(n_t_layers, args.nnm_n_layers)
    _print0(f"[NNM] student layers ({n_s_layers}): selected {s_mid}")
    _print0(f"[NNM] teacher layers ({n_t_layers}): selected {t_mid}")

    # Attach projectors on every rank. Same seed (torch.manual_seed via
    # accelerate's set_seed if used) → same init across ranks. We add an
    # explicit local seed so even without accelerate's set_seed, projectors
    # match across ranks.
    proj_dtype = next(student.parameters()).dtype
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    projectors = nn.ModuleList([nn.Linear(d_s, d_t, bias=False) for _ in s_mid])
    with torch.no_grad():
        for p in projectors:
            p.weight.copy_(torch.randn(d_t, d_s, generator=g) * 0.02)
    projectors = projectors.to(dtype=proj_dtype)
    _print0(f"[NNM] attached {len(projectors)} projectors "
            f"({d_s} -> {d_t}) to student")

    # ── Shared file path for centroids + R ──
    cache_dir = os.path.join(args.output_dir, "_nnm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "centroids_R.pt")
    ready_file = os.path.join(cache_dir, "READY")

    if _is_rank0():
        # Clean any stale "READY" marker from a previous run
        if os.path.exists(ready_file):
            os.remove(ready_file)

        # Move teacher to GPU on rank 0 only
        teacher = teacher.to(device)

        # Build centroids
        teacher_device = next(teacher.parameters()).device
        batches = _batches_from_chatml_dataset(
            train_dataset, tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_batches=args.nnm_centroid_batches,
        )
        t_centroids = build_teacher_centroids(
            teacher=teacher,
            dataloader=batches,
            student_layer_mapping=s_mid,
            teacher_layer_mapping=t_mid,
            K=args.nnm_K,
            eta=args.nnm_eta,
            T_dead=args.nnm_T_dead,
            max_batches=args.nnm_centroid_batches,
            device=teacher_device,
        )
        R = make_R(d_t, args.nnm_d_prime, device=device, seed=args.seed)

        # Save to disk (move to CPU for portability)
        payload = {
            "t_centroids": {k: v.cpu() for k, v in t_centroids.items()},
            "R":           R.cpu(),
            "d_s":         d_s,
            "d_t":         d_t,
            "s_mid":       s_mid,
            "t_mid":       t_mid,
        }
        torch.save(payload, cache_file)
        # Atomic marker that file is fully written
        with open(ready_file, "w") as f:
            f.write("ok")
        _print0(f"[NNM] rank 0 saved centroids+R to {cache_file}")

        # Free teacher GPU memory — Trainer will re-prepare it
        teacher = teacher.cpu()
        torch.cuda.empty_cache()
    else:
        # Wait for rank 0 to finish (poll READY file)
        print(f"[NNM] rank {os.environ.get('RANK','?')} waiting for centroids...")
        timeout = 60 * 60  # 1h
        t0 = time.time()
        while not os.path.exists(ready_file):
            if time.time() - t0 > timeout:
                raise RuntimeError(f"[NNM] timeout waiting for {ready_file}")
            time.sleep(2)
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        t_centroids = payload["t_centroids"]
        R = payload["R"]

    # Move to local device on every rank
    t_centroids = {k: v.to(device) for k, v in t_centroids.items()}
    R = R.to(device)

    layer_weights = {s_lid: layer_weight(s_lid, n_s_layers) for s_lid in s_mid}

    _print0(f"[NNM] centroids ready: K={args.nnm_K}, d_t={d_t}, "
            f"d_prime={args.nnm_d_prime}")
    return {
        "s_mid":         s_mid,
        "t_mid":         t_mid,
        "t_centroids":   t_centroids,
        "R":             R,
        "layer_weights": layer_weights,
    }, projectors


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Student ──
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        attn_implementation=args.attn,
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True,
    )

    # ── Teacher ──
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        attn_implementation=args.attn,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # TSD-KD: resize student embeddings so vocab matches teacher
    student.resize_token_embeddings(teacher.lm_head.weight.shape[0])
    _print0(f"student lm_head: {student.lm_head.weight.shape}")
    _print0(f"teacher lm_head: {teacher.lm_head.weight.shape}")
    assert student.lm_head.weight.shape[0] == teacher.lm_head.weight.shape[0]

    # ── Dataset ──
    ds = ds = load_dataset(
        "VoCuc/UltraInteract-Infer",
        data_files=args.dataset,
        split="train"
    ).train_test_split(test_size=0.01)

    def add_messages(example):
        return {
            "messages": [
                {"role": "user",      "content": example["prompt"]},
                {"role": "assistant", "content": example["generated_text"]},
            ]
        }

    train_dataset = ds["train"].map(add_messages).remove_columns(["prompt"])
    eval_dataset  = ds["test"].map(add_messages).remove_columns(["prompt"])

    # ═══════════════════════════════════════════════════════════════
    #  NNM pre-pass (BEFORE Trainer is created)
    #
    #  Only rank 0 loads teacher to GPU and builds centroids; the result is
    #  written to {output_dir}/_nnm_cache/ and the other ranks read it.
    # ═══════════════════════════════════════════════════════════════
    nnm_state = None
    if args.nnm:
        nnm_state, projectors = prepare_nnm(args, student, teacher,
                                train_dataset, tokenizer, device)

    # ── Training args ──
    training_args = GKDConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        per_device_eval_batch_size=args.eval_batch_size,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        save_total_limit=3,
        report_to=args.report_to,
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        seed=args.seed,
    )

    # ═══════════════════════════════════════════════════════════════
    #  LoRA / PEFT config (optional)
    #
    #  When --use-lora is set we wrap the student with LoRA adapters.
    #  Crucially we list "projectors" in `modules_to_save` so PEFT keeps
    #  the NNM projectors fully trainable instead of freezing them with
    #  the rest of the base model.
    # ═══════════════════════════════════════════════════════════════
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")
                          if m.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            modules_to_save=(["projectors"] if args.nnm else None),
        )
        _print0(f"[LoRA] enabled: r={args.lora_r}, alpha={args.lora_alpha}, "
                f"dropout={args.lora_dropout}, targets={target_modules}")

    trainer = DistillTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        token_entropy_percentile_threshold=args.threshold,
        indirect_kd_alpha=args.indirect_kd_alpha,
        peft_config=peft_config,
        # NNM
        nnm_state=nnm_state,
        nnm_ratio=args.nnm_ratio,
        nnm_warmup_steps=args.nnm_warmup_steps,
        nnm_ramp_steps=args.nnm_ramp_steps,
        nnm_ns_iters=args.nnm_ns_iters,
        projectors=projectors,
    )

    # ═══════════════════════════════════════════════════════════════
    #  Safety net: force NNM projector grads ON.
    #
    #  PEFT freezes the entire base model and keeps only LoRA adapters
    #  + `modules_to_save` trainable. modules_to_save="projectors" above
    #  usually handles this, but PEFT's name resolution depends on the
    #  attribute name in the wrapped model. After Trainer is built, we
    #  walk the model and explicitly set requires_grad=True on anything
    #  whose path contains 'projectors'. Idempotent — safe whether or
    #  not LoRA is enabled.
    # ═══════════════════════════════════════════════════════════════
    if args.nnm:
        n_enabled = 0
        for name, p in trainer.model.named_parameters():
            if "projectors" in name:
                p.requires_grad = True
                n_enabled += p.numel()
        _print0(f"[NNM] re-enabled grad on projector params: {n_enabled} elements")

    if args.use_lora:
        # Print trainable/total breakdown so you can sanity-check
        if hasattr(trainer.model, "print_trainable_parameters"):
            if is_main_process():
                trainer.model.print_trainable_parameters()

    trainer.train()


if __name__ == "__main__":
    main()