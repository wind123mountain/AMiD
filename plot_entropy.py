"""
plot_entropy_dist.py — Layer-wise entropy distribution plots from saved hidden states.

Reads the .pt files saved by analyze_layers.py (--save-hidden-states) and plots,
for each layer, a histogram of per-sample matrix entropy values across all models —
one subplot per layer, styled like Figure 4 in the reference image.

Each .pt file has shape [n_layers+1, seq_len, hidden_dim] (no padding tokens).

Directory layout expected (mirrors what analyze_layers.py writes):
    <root>/
        hidden_states/
            teacher/
                sample_0.pt
                sample_1.pt
                ...
            student_base/
                sample_0.pt
                ...
            amid/
            csd/
            nnm_ours/

Usage:
    python plot_entropy_dist.py --hs-root ./layer_analysis/6_method_tsd/hidden_states
    python plot_entropy_dist.py \
        --hs-root   ./layer_analysis/6_method/hidden_states \
        --save-dir  ./layer_analysis/plots \
        --bins      40 \
        --alpha     0.55 \
        --max-samples 200
"""

import os
import argparse
import math
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════
#  Style — must match analyze_layers.py MODEL_STYLE keys
# ════════════════════════════════════════════════════════════════

MODEL_STYLE = {
    "Teacher":      {"color": "#1f77b4", "label": "Teacher"},
    "Student-base": {"color": "#8c564b", "label": "Student-base"},
    "DistiLLM":     {"color": "#ff7f0e", "label": "DistiLLM"},
    "AMID":         {"color": "#9467bd", "label": "AMID"},
    "CSD":          {"color": "#e377c2", "label": "CSD"},
    "NNM (ours)":   {"color": "#2ca02c", "label": "NNM (ours)"},
}

# Map from folder name (lowercased, spaces→underscore, parens removed)
# to display label.  Extend if you add more models.
FOLDER_TO_LABEL = {
    "teacher":      "Teacher",
    "student_base": "Student-base",
    "distillm":     "DistiLLM",
    "amid":         "AMID",
    "csd":          "CSD",
    "nnm_ours":     "NNM (ours)",
}

# Plot order (determines legend order and draw order)
MODEL_ORDER = list(MODEL_STYLE.keys())


# ════════════════════════════════════════════════════════════════
#  Entropy metric (same formula as analyze_layers.py)
# ════════════════════════════════════════════════════════════════

def _subsample(Z: torch.Tensor, max_n: int = 512) -> torch.Tensor:
    if Z.shape[0] <= max_n:
        return Z
    g = torch.Generator()
    g.manual_seed(Z.shape[0] * Z.shape[1])
    idx = torch.randperm(Z.shape[0], generator=g)[:max_n]
    return Z[idx]


@torch.no_grad()
def matrix_entropy_alpha1(Z_raw: torch.Tensor) -> float:
    """
    Von Neumann entropy of the Gram matrix K = Z_c Z_c^T,
    where Z_c is Z centered by its row mean.
    Returns scalar entropy value for one [T, D] tensor.
    """
    Z_sub = _subsample(Z_raw.float())
    Z = Z_sub - Z_sub.mean(dim=0, keepdim=True)   # center

    if Z.shape[0] < 2:
        return float("nan")
    try:
        S = torch.linalg.svdvals(Z)
        eig = S ** 2
        eig = eig[eig > 1e-10]
        if eig.numel() == 0:
            return float("nan")
        p = eig / eig.sum()
        return (-p * (p + 1e-12).log()).sum().item()
    except Exception:
        return float("nan")


# ════════════════════════════════════════════════════════════════
#  Load hidden states and compute per-sample entropy per layer
# ════════════════════════════════════════════════════════════════

def load_entropy_per_layer(model_dir: str, max_samples: int = None) -> np.ndarray | None:
    """
    Load all sample_*.pt files from `model_dir` and compute matrix entropy
    for each (sample, layer).

    Returns:
        entropies : np.ndarray of shape [n_samples, n_layers+1]
                    NaN entries are kept (filtered at plot time).
        None      : if the directory is empty or missing.
    """
    pt_files = sorted(glob(os.path.join(model_dir, "sample_*.pt")))
    if not pt_files:
        return None

    if max_samples is not None:
        pt_files = pt_files[:max_samples]

    all_entropies = []
    for fpath in tqdm(pt_files, desc=f"  {os.path.basename(model_dir)}", leave=False):
        try:
            # shape: [n_layers+1, seq_len, hidden_dim]
            hs = torch.load(fpath, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  Warning: could not load {fpath}: {e}")
            continue

        n_layers_plus1 = hs.shape[0]
        sample_ents = []
        for lid in range(n_layers_plus1):
            e = matrix_entropy_alpha1(hs[lid])   # [seq_len, hidden_dim]
            sample_ents.append(e)
        all_entropies.append(sample_ents)

    if not all_entropies:
        return None

    return np.array(all_entropies, dtype=np.float32)  # [n_samples, n_layers+1]


# ════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════

def plot_entropy_distributions(
    entropy_data: dict,   # label -> np.ndarray [n_samples, n_layers+1]
    save_dir: str,
    bins: int = 40,
    alpha: float = 0.55,
    filename: str = "entropy_distributions.png",
):
    """
    Create a grid of subplots (one per layer) where each subplot shows
    overlapping histograms of per-sample entropy, one histogram per model.
    Matches the style of Figure 4 in the reference image.
    """
    # Determine number of layers from the first available model
    n_layers_plus1 = next(iter(entropy_data.values())).shape[1]
    n_layers = n_layers_plus1 - 1   # layer 0 = embedding, layers 1..N = transformer

    # Grid layout: 3 columns (same as reference image)
    ncols = 3
    nrows = math.ceil(n_layers_plus1 / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 3.2),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    # Build shared x-axis range across all layers and models for consistency
    all_vals = np.concatenate([
        arr.flatten() for arr in entropy_data.values()
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    global_xmin = float(np.percentile(all_vals, 0.5))
    global_xmax = float(np.percentile(all_vals, 99.5))

    for lid in range(n_layers_plus1):
        ax = axes_flat[lid]
        layer_label = f"layer{lid}"

        # Collect entropy values for this layer, per model, in plot order
        layer_xmin, layer_xmax = global_xmax, global_xmin
        for label in MODEL_ORDER:
            if label not in entropy_data:
                continue
            vals = entropy_data[label][:, lid]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            layer_xmin = min(layer_xmin, float(vals.min()))
            layer_xmax = max(layer_xmax, float(vals.max()))

        # Small buffer
        span = layer_xmax - layer_xmin
        pad = span * 0.05 if span > 0 else 0.01
        xmin = layer_xmin - pad
        xmax = layer_xmax + pad

        # Draw histograms (back → front: MODEL_ORDER reversed so Teacher is on top)
        for label in reversed(MODEL_ORDER):
            if label not in entropy_data:
                continue
            vals = entropy_data[label][:, lid]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            color = MODEL_STYLE[label]["color"]
            ax.hist(
                vals,
                bins=bins,
                range=(xmin, xmax),
                color=color,
                alpha=alpha,
                edgecolor="none",
                label=label,
            )

        ax.set_title(layer_label, fontsize=10, pad=3)
        ax.set_xlabel("Entropy", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(xmin, xmax)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplot slots
    for idx in range(n_layers_plus1, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend at the top (matches the reference image style)
    legend_handles = []
    for label in MODEL_ORDER:
        if label not in entropy_data:
            continue
        patch = mpatches.Patch(
            facecolor=MODEL_STYLE[label]["color"],
            alpha=0.7,
            label=label,
        )
        legend_handles.append(patch)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.suptitle(
        "Layer-wise Entropy Distributions",
        fontsize=13, fontweight="bold", y=1.03,
    )

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  → saved {out_path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot layer-wise entropy distributions from saved hidden states."
    )
    p.add_argument(
        "--hs-root", type=str,
        default="./layer_analysis/hidden_states",
        help="Root directory containing one sub-folder per model "
             "(e.g. teacher/, student_base/, amid/, …).",
    )
    p.add_argument(
        "--save-dir", type=str,
        default="./layer_analysis/plots",
        help="Where to write the output PNG.",
    )
    p.add_argument(
        "--bins", type=int, default=40,
        help="Number of histogram bins per subplot.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.55,
        help="Histogram transparency (0–1).  Lower = more see-through.",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of .pt files loaded per model (useful for quick checks).",
    )
    p.add_argument(
        "--filename", type=str, default="entropy_distributions.png",
        help="Output filename.",
    )
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Entropy distribution plots")
    print(f"  hs-root    : {args.hs_root}")
    print(f"  save-dir   : {args.save_dir}")
    print(f"  bins       : {args.bins}  alpha: {args.alpha}")
    if args.max_samples:
        print(f"  max-samples: {args.max_samples}")
    print(f"{'='*60}\n")

    if not os.path.isdir(args.hs_root):
        raise FileNotFoundError(
            f"Hidden-state root not found: {args.hs_root}\n"
            "Run analyze_layers.py with --save-hidden-states first."
        )

    # Discover model folders
    subdirs = sorted([
        d for d in os.listdir(args.hs_root)
        if os.path.isdir(os.path.join(args.hs_root, d))
    ])
    print(f"Found {len(subdirs)} model folder(s): {subdirs}\n")

    entropy_data = {}

    for folder in subdirs:
        label = FOLDER_TO_LABEL.get(folder.lower())
        if label is None:
            # Try a fuzzy match: strip common suffixes and retry
            label = FOLDER_TO_LABEL.get(
                folder.lower().replace("-", "_").replace(" ", "_")
            )
        if label is None:
            print(f"  ⚠  Unknown folder '{folder}' — skipping "
                  f"(add it to FOLDER_TO_LABEL to include it).")
            continue

        model_dir = os.path.join(args.hs_root, folder)
        print(f"[{label}]  loading from {model_dir}")
        arr = load_entropy_per_layer(model_dir, max_samples=args.max_samples)
        if arr is None:
            print(f"  ⚠  No .pt files found in {model_dir} — skipping.")
            continue

        print(f"  → loaded {arr.shape[0]} samples × {arr.shape[1]} layers")
        entropy_data[label] = arr

    if not entropy_data:
        print("No data loaded — nothing to plot.")
        return

    print(f"\nPlotting {len(entropy_data)} model(s) …")
    plot_entropy_distributions(
        entropy_data,
        save_dir=args.save_dir,
        bins=args.bins,
        alpha=args.alpha,
        filename=args.filename,
    )
    print(f"\nDone.  Output in: {args.save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()