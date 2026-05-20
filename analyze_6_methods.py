"""
analyze_layers.py — Layer-wise representation quality analysis.

Computes 5 metrics across all layers for 6 models:
    1. Nuclear Norm (raw)
    2. Nuclear Norm (normalized by sqrt(N*D))
    3. Effective Rank        — exp(H(σ_normalized))   (Roy & Vetterli 2007)
    4. Matrix Entropy α=1    — von Neumann entropy on Gram matrix
    5. Curvature             — avg angle between consecutive token diffs
                               (Hosseini & Fedorenko 2023)

Models compared (6 total):
    • Teacher          (Qwen2.5-14B-Instruct or similar)
    • Student-base     (no distillation)
    • DistiLLM         (SFKL baseline)
    • AMID
    • CSD
    • NNM (ours)

Reference:
    "Layer by Layer: Uncovering Hidden Representations in Language Models"
    (Skean et al., ICML 2025, arxiv:2502.02013)

Usage:
    python analyze_layers.py
    python analyze_layers.py --n-samples 100 --max-len 256
    python analyze_layers.py \\
        --ckpt-distillm  results/distillm/X \\
        --ckpt-amid      results/amid/X \\
        --ckpt-csd       results/csd/X \\
        --ckpt-nnm       results/nnm/X
"""

import os
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import transformers.integrations.peft as peft_integration


peft_integration.is_peft_available = lambda: False


# ════════════════════════════════════════════════════════════════
#  Style map — single source of truth for color/marker/order
# ════════════════════════════════════════════════════════════════

MODEL_STYLE = {
    "Teacher":      {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "lw": 2.2, "alpha": 0.95},
    "Student-base": {"color": "#8c564b", "marker": "x", "linestyle": "--", "lw": 1.5, "alpha": 0.75},
    "DistiLLM":     {"color": "#ff7f0e", "marker": "s", "linestyle": "--", "lw": 1.6, "alpha": 0.85},
    "AMID":         {"color": "#9467bd", "marker": "D", "linestyle": "--", "lw": 1.6, "alpha": 0.85},
    "CSD":          {"color": "#e377c2", "marker": "v", "linestyle": "--", "lw": 1.6, "alpha": 0.85},
    "NNM (ours)":   {"color": "#2ca02c", "marker": "^", "linestyle": "-",  "lw": 2.0, "alpha": 0.95},
}
MODEL_ORDER = list(MODEL_STYLE.keys())   # plot legend order


# ════════════════════════════════════════════════════════════════
#  Metric computations
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def nuclear_norm(Z: torch.Tensor) -> tuple[float, float]:
    """||Z||_*  and  ||Z||_* / sqrt(N*D)."""
    Z = Z.float()
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return 0.0, 0.0
    try:
        if Z.shape[0] > 512:
            idx = torch.randperm(Z.shape[0])[:512]
            Z = Z[idx]
        S = torch.linalg.svdvals(Z)
        nuc = S.sum().item()
        nuc_n = nuc / math.sqrt(Z.shape[0] * Z.shape[1])
        return nuc, nuc_n
    except Exception:
        return 0.0, 0.0


@torch.no_grad()
def effective_rank(Z: torch.Tensor) -> float:
    """exp(H(p)), p_i = σ_i / Σσ_j."""
    Z = Z.float()
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return 0.0
    try:
        if Z.shape[0] > 512:
            idx = torch.randperm(Z.shape[0])[:512]
            Z = Z[idx]
        S = torch.linalg.svdvals(Z)
        S = S[S > 1e-10]
        if S.numel() == 0:
            return 0.0
        p = S / S.sum()
        H = -(p * (p + 1e-12).log()).sum().item()
        return math.exp(H)
    except Exception:
        return 0.0


@torch.no_grad()
def matrix_entropy_alpha1(Z: torch.Tensor) -> float:
    """Von Neumann entropy on Gram K = Z Z^T."""
    Z = Z.float()
    if Z.shape[0] < 2:
        return 0.0
    if Z.shape[0] > 512:
        idx = torch.randperm(Z.shape[0])[:512]
        Z = Z[idx]
    try:
        S = torch.linalg.svdvals(Z)
        eig = S ** 2
        eig = eig[eig > 1e-10]
        if eig.numel() == 0:
            return 0.0
        p = eig / eig.sum()
        return (-p * (p + 1e-12).log()).sum().item()
    except Exception:
        return 0.0


@torch.no_grad()
def curvature(Z: torch.Tensor) -> float:
    """Mean arccos cosine between consecutive token diff vectors."""
    Z = Z.float()
    if Z.shape[0] < 3:
        return 0.0
    v = Z[1:] - Z[:-1]
    v_norm = F.normalize(v, dim=-1)
    cos = (v_norm[1:] * v_norm[:-1]).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.arccos(cos).mean().item()


# ════════════════════════════════════════════════════════════════
#  Per-layer metric extraction
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_layer_metrics(model, tokenizer, prompts, device, max_len=256):
    """Return dict[metric] -> list of (n_layers + 1) averaged values."""
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_total  = n_layers + 1

    metrics = {
        "nuclear_norm":      [[] for _ in range(n_total)],
        "nuclear_norm_norm": [[] for _ in range(n_total)],
        "effective_rank":    [[] for _ in range(n_total)],
        "matrix_entropy":    [[] for _ in range(n_total)],
        "curvature":         [[] for _ in range(n_total)],
    }

    for prompt in tqdm(prompts, desc="  layers", leave=False):
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_len,
        ).to(device)
        if enc["input_ids"].shape[1] < 5:
            continue

        out = model(**enc, output_hidden_states=True, return_dict=True)

        for lid in range(n_total):
            Z = out.hidden_states[lid].squeeze(0).float().cpu()
            nuc, nuc_n = nuclear_norm(Z)
            metrics["nuclear_norm"][lid].append(nuc)
            metrics["nuclear_norm_norm"][lid].append(nuc_n)
            metrics["effective_rank"][lid].append(effective_rank(Z))
            metrics["matrix_entropy"][lid].append(matrix_entropy_alpha1(Z))
            metrics["curvature"][lid].append(curvature(Z))

        del out
        torch.cuda.empty_cache()

    return {
        k: [float(np.mean(v)) if len(v) > 0 else 0.0 for v in vals]
        for k, vals in metrics.items()
    }


# ════════════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════════════

def load_model_safely(name_or_path: str, device: str, dtype=torch.float16):
    print(f"  Loading {name_or_path} ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map={"": device},
            use_safetensors=True,
        )
    except Exception as e:
        print(f"  device_map failed ({e}), falling back to .to(device)")
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  → {n_params:.2f}B params, {model.config.num_hidden_layers} layers")
    return model


def get_tokenizer_for(ckpt_path: str, fallback_id: str):
    """Try to load tokenizer from ckpt folder; fall back to model id."""
    try:
        return AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=True, padding_side="right",
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            fallback_id, trust_remote_code=True, padding_side="right",
        )


# ════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════

def plot_metric(metric_name: str, results: dict, save_dir: str,
                ylabel: str = None, log_scale: bool = False):
    """One metric, all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label in MODEL_ORDER:
        if label not in results:
            continue
        layer_vals = results[label]
        n = len(layer_vals)
        x_pct = np.linspace(0, 100, n)
        st = MODEL_STYLE[label]
        ax.plot(x_pct, layer_vals,
                label=label,
                color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                markersize=5, linewidth=st["lw"], alpha=st["alpha"])

    ax.set_xlabel("Layer Depth (%)", fontsize=12)
    ax.set_ylabel(ylabel or metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"{metric_name.replace('_', ' ').title()} across Layers",
                 fontsize=13, fontweight="bold")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{metric_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → saved {out_path}")
    plt.close()


def plot_combined_1x5(all_results: dict, save_dir: str):
    """1x5 panel — all 5 metrics side by side, 6 curves each."""
    metrics_spec = [
        ("nuclear_norm",      "Nuclear Norm ||Z||_*",          False),
        ("nuclear_norm_norm", "Nuclear Norm (normalized)",     False),
        ("effective_rank",    "Effective Rank",                False),
        ("matrix_entropy",    "Matrix Entropy (α=1)",          False),
        ("curvature",         "Curvature (rad)",               False),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))

    for ax, (mkey, mlabel, log) in zip(axes, metrics_spec):
        for label in MODEL_ORDER:
            if label not in all_results:
                continue
            vals = all_results[label][mkey]
            n = len(vals)
            x_pct = np.linspace(0, 100, n)
            st = MODEL_STYLE[label]
            ax.plot(x_pct, vals,
                    label=label,
                    color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                    markersize=4, linewidth=st["lw"], alpha=st["alpha"])
        ax.set_xlabel("Layer Depth (%)", fontsize=11)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # One shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels), fontsize=11,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.suptitle("Layer-wise Representation Quality (Teacher vs Baselines vs NNM)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    out = os.path.join(save_dir, "combined_1x5.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → saved {out}")
    plt.close()


# ════════════════════════════════════════════════════════════════
#  Data preparation
# ════════════════════════════════════════════════════════════════

def get_eval_prompts(tokenizer, n_samples: int = 50, dataset_name: str = "gsm8k"):
    if dataset_name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
    elif dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        texts = ds["question"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    np.random.seed(42)
    indices = np.random.choice(len(texts), min(n_samples, len(texts)), replace=False)
    prompts = [texts[i] for i in indices]
    print(f"  Selected {len(prompts)} prompts from {dataset_name}")
    return prompts


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Layer-wise metric analysis (6 models)")
    # base IDs
    p.add_argument("--teacher-id",  type=str, default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--student-id",  type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    # 4 distilled checkpoints — placeholders, fill in later
    p.add_argument("--ckpt-distillm", type=str,
                   default="results/PLACEHOLDER_distillm/checkpoint",
                   help="Path to DistiLLM checkpoint.")
    p.add_argument("--ckpt-amid",     type=str,
                   default="results/PLACEHOLDER_amid/checkpoint",
                   help="Path to AMID checkpoint.")
    p.add_argument("--ckpt-csd",      type=str,
                   default="results/PLACEHOLDER_csd/checkpoint",
                   help="Path to CSD checkpoint.")
    p.add_argument("--ckpt-nnm",      type=str,
                   default="results/PLACEHOLDER_nnm/checkpoint",
                   help="Path to NNM (ours) checkpoint.")

    # data
    p.add_argument("--n-samples",   type=int, default=100)
    p.add_argument("--max-len",     type=int, default=512)
    p.add_argument("--dataset",     type=str, default="gsm8k",
                   choices=["wikitext", "gsm8k"])

    # output / device
    p.add_argument("--save-dir",    type=str, default="./layer_analysis")
    p.add_argument("--device",      type=str, default="cuda:0")

    # skip flags — useful while ckpts aren't ready yet
    p.add_argument("--skip-teacher",      action="store_true")
    p.add_argument("--skip-student-base", action="store_true")
    p.add_argument("--skip-distillm",     action="store_true")
    p.add_argument("--skip-amid",         action="store_true")
    p.add_argument("--skip-csd",          action="store_true")
    p.add_argument("--skip-nnm",          action="store_true")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def _ckpt_available(path: str) -> bool:
    """Check if a ckpt path is real (vs placeholder)."""
    return os.path.isdir(path) and "PLACEHOLDER" not in path


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  Layer-wise representation analysis (6 models)")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset}, n_samples={args.n_samples}, max_len={args.max_len}")
    print(f"{'='*70}\n")

    # ── Tokenizer (shared default — Qwen family) ─────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_id, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Preparing eval prompts...")
    prompts = get_eval_prompts(tokenizer, args.n_samples, args.dataset)

    # ── Model list (label, path, tokenizer source, skip-flag) ────
    model_configs = [
        ("Teacher",      args.teacher_id,    args.teacher_id,  args.skip_teacher),
        ("Student-base", args.student_id,    args.student_id,  args.skip_student_base),
        ("DistiLLM",     args.ckpt_distillm, args.student_id,  args.skip_distillm),
        ("AMID",         args.ckpt_amid,     args.student_id,  args.skip_amid),
        ("CSD",          args.ckpt_csd,      args.student_id,  args.skip_csd),
        ("NNM (ours)",   args.ckpt_nnm,      args.student_id,  args.skip_nnm),
    ]

    all_results = {}

    for i, (label, path, tok_fallback, skip) in enumerate(model_configs, start=1):
        print(f"\n[{i}/{len(model_configs)}] {label}")

        if skip:
            print(f"  ⏭  skipped (--skip flag)")
            continue

        # For distilled checkpoints, check the path looks real
        is_base = label in ("Teacher", "Student-base")
        if not is_base and not _ckpt_available(path):
            print(f"  ⏭  skipped (checkpoint not found / placeholder: {path})")
            continue

        try:
            model = load_model_safely(path, device)
            tok   = tokenizer if is_base else get_tokenizer_for(path, tok_fallback)
            all_results[label] = compute_layer_metrics(
                model, tok, prompts, device, args.max_len,
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ✗ failed to process {label}: {e}")
            continue

    # ── Save raw numbers ────────────────────────────────────────
    json_path = os.path.join(args.save_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw metrics → {json_path}")

    if not all_results:
        print("No results to plot — exiting.")
        return

    # ── Individual plots (5 metrics) ────────────────────────────
    print("\nGenerating plots...")
    plot_metric("nuclear_norm",      {k: v["nuclear_norm"]      for k, v in all_results.items()},
                args.save_dir, ylabel="Nuclear Norm ||Z||_*")
    plot_metric("nuclear_norm_norm", {k: v["nuclear_norm_norm"] for k, v in all_results.items()},
                args.save_dir, ylabel="Nuclear Norm / sqrt(N*D)")
    plot_metric("effective_rank",    {k: v["effective_rank"]    for k, v in all_results.items()},
                args.save_dir, ylabel="Effective Rank")
    plot_metric("matrix_entropy",    {k: v["matrix_entropy"]    for k, v in all_results.items()},
                args.save_dir, ylabel="Matrix Entropy (α=1)")
    plot_metric("curvature",         {k: v["curvature"]         for k, v in all_results.items()},
                args.save_dir, ylabel="Curvature (rad)")

    # ── Combined 1x5 figure ─────────────────────────────────────
    plot_combined_1x5(all_results, args.save_dir)

    print(f"\nAll outputs in: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()