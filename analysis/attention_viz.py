"""
analysis/attention_viz.py — Visualize Transformer attention weights from F1 SAINT.

Run from project root:
    python analysis/attention_viz.py

Uses PyTorch forward hooks to capture attention weight tensors from the
TransformerEncoderLayer. Since the SAINT uses seq_len=1 (single token),
the meaningful attention is within the feedforward layers, but the
multi-head attention scores are still extractable for analysis.

For interpretability, we also visualize the feature embedding magnitudes
as a proxy for learned feature saliency.

Outputs (results/figures/)
--------------------------
attention_feature_weights.png    — L2 norm of embedding weights per feature
attention_head_avg.png           — average attention across heads and layers
feature_embedding_importance.png — input gradient-based feature saliency
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    DATA_CSV, F1_MODEL_PATH, FEATURE_INFO_PATH,
    F1_HIDDEN_SIZE, F1_NUM_HEADS, F1_NUM_LAYERS, F1_DROPOUT, F1_OUTPUT_SIZE,
    OUTCOME_COL, PATIENT_ID_COL, CLINICAL_THRESHOLD,
)
from src.data.preprocessor import load_feature_info, standardize_like_f1

FIG_DIR = "results/figures"


def load_model_and_data():
    df   = pd.read_csv(DATA_CSV)
    fi   = load_feature_info(FEATURE_INFO_PATH)
    X    = standardize_like_f1(df, fi).values.astype(np.float32)
    y    = df[OUTCOME_COL].values.astype(np.float32)
    pids = df[PATIENT_ID_COL].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(gss.split(X, y, groups=pids))
    X_test = X[te_idx]
    y_test = y[te_idx]

    from src.models.saint import SAINT
    from src.utils.device import get_device
    device = get_device()
    model  = SAINT(
        input_size=X.shape[1], hidden_size=F1_HIDDEN_SIZE,
        output_size=F1_OUTPUT_SIZE,
        discrete_feature_indices=fi["discrete_feature_indices"],
        continuous_feature_indices=fi["continuous_feature_indices"],
        num_heads=F1_NUM_HEADS, num_layers=F1_NUM_LAYERS, dropout=F1_DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(F1_MODEL_PATH, map_location=device,
                                     weights_only=True))
    model.eval()
    return model, X_test, y_test, fi, device


def plot_embedding_weight_norms(model, feature_names: list, fi: dict, save_path: str):
    """Visualize L2 norms of embedding weights as feature importance proxy."""
    disc_names = fi["discrete_feature_names"]
    cont_names = fi["continuous_feature_names"]
    all_names  = fi["original_feature_names"]

    importances = np.zeros(len(all_names))

    if hasattr(model, "discrete_embedding") and len(disc_names) > 0:
        W = model.discrete_embedding[0].weight.detach().cpu().numpy()  # (hidden, n_disc)
        norms = np.linalg.norm(W, axis=0)
        for i, col in enumerate(disc_names):
            idx = all_names.index(col)
            importances[idx] = norms[i]

    if hasattr(model, "continuous_embedding") and len(cont_names) > 0:
        W = model.continuous_embedding[0].weight.detach().cpu().numpy()  # (hidden, n_cont)
        norms = np.linalg.norm(W, axis=0)
        for i, col in enumerate(cont_names):
            idx = all_names.index(col)
            importances[idx] = norms[i]

    # Sort descending
    order = np.argsort(importances)[::-1][:30]
    names = [all_names[i] for i in order]
    vals  = importances[order]

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.3)))
    colors = ["#d62728" if all_names[order[i]] in cont_names else "#1f77b4"
              for i in range(len(names))]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("L2 Norm of Embedding Weights")
    ax.set_title("Feature Saliency via Embedding Weight Norms\n"
                 "(red=continuous, blue=discrete; top 30)")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d62728", label="Continuous"),
                        Patch(color="#1f77b4", label="Discrete")],
              loc="lower right")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {save_path}")


def plot_gradient_saliency(model, X_test: np.ndarray, y_test: np.ndarray,
                           feature_names: list, device: torch.device,
                           save_path: str, n_samples: int = 200):
    """Input-gradient saliency: mean |∂output/∂input| across test samples."""
    model.eval()
    n = min(n_samples, len(X_test))
    Xt = torch.tensor(X_test[:n], dtype=torch.float32,
                      requires_grad=True, device=device)

    out = model(Xt)
    out.sum().backward()
    saliency = Xt.grad.abs().detach().cpu().numpy().mean(axis=0)  # (n_features,)

    order = np.argsort(saliency)[::-1][:30]
    names = [feature_names[i] for i in order]
    vals  = saliency[order]

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.3)))
    ax.barh(names[::-1], vals[::-1], color="#2ca02c")
    ax.set_xlabel("Mean |Gradient| (saliency)")
    ax.set_title("Input Gradient Saliency — F1 SAINT (top 30 features)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {save_path}")


def plot_attention_head_patterns(model, X_test: np.ndarray, device: torch.device,
                                 save_path: str, n_samples: int = 100):
    """
    Extract self-attention weights across heads and layers by hooking into
    TransformerEncoderLayer. Since seq_len=1, this gives a (1×1) attention
    matrix per head, but we can examine the query/key norms as a proxy.
    Instead, plot the per-head attention weight norms from in_proj_weight.
    """
    # For seq_len=1 SAINT, the attention score is trivially 1.0.
    # More informative: plot the per-layer, per-head Q·K weight magnitudes
    # as a proxy for what each head is focusing on.

    attn_data = []  # list of (layer_idx, head_idx, norm)
    d_model = F1_HIDDEN_SIZE
    n_heads = F1_NUM_HEADS
    head_dim = d_model // n_heads

    for layer_idx, layer in enumerate(model.transformer_encoder.layers):
        W = layer.self_attn.in_proj_weight.detach().cpu().numpy()
        # W shape: (3*d_model, d_model) — concat of Q, K, V weights
        W_q = W[:d_model, :]           # (d_model, d_model)
        W_k = W[d_model:2*d_model, :]
        for h in range(n_heads):
            Wq_h = W_q[h*head_dim:(h+1)*head_dim, :]
            Wk_h = W_k[h*head_dim:(h+1)*head_dim, :]
            norm = float(np.linalg.norm(Wq_h) * np.linalg.norm(Wk_h))
            attn_data.append((layer_idx, h, norm))

    # Plot as heatmap: layers × heads
    mat = np.zeros((F1_NUM_LAYERS, F1_NUM_HEADS))
    for li, hi, val in attn_data:
        mat[li, hi] = val

    fig, ax = plt.subplots(figsize=(max(6, F1_NUM_HEADS), max(4, F1_NUM_LAYERS * 0.6)))
    sns.heatmap(mat, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                xticklabels=[f"H{i}" for i in range(F1_NUM_HEADS)],
                yticklabels=[f"L{i}" for i in range(F1_NUM_LAYERS)])
    ax.set_title("Q·K Weight Norms per Head/Layer\n(proxy for attention specialization)")
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Encoder Layer")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {save_path}")


def main():
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    print("Loading model and data ...")
    model, X_test, y_test, fi, device = load_model_and_data()
    feature_names = fi["original_feature_names"]

    # 1. Embedding weight norms
    plot_embedding_weight_norms(
        model, feature_names, fi,
        f"{FIG_DIR}/attention_feature_weights.png",
    )

    # 2. Input gradient saliency
    plot_gradient_saliency(
        model, X_test, y_test, feature_names, device,
        f"{FIG_DIR}/feature_embedding_importance.png",
    )

    # 3. Attention head patterns
    plot_attention_head_patterns(
        model, X_test, device,
        f"{FIG_DIR}/attention_head_avg.png",
    )

    print("\nAttention visualization complete.")


if __name__ == "__main__":
    main()
