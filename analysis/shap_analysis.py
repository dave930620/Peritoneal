"""
analysis/shap_analysis.py — SHAP interpretability for the F1 SAINT model.

Run from project root:
    python analysis/shap_analysis.py

Prerequisites
-------------
pip install shap
train_f1.py must have been run first.

Outputs (results/figures/)
--------------------------
shap_summary_beeswarm.png    — global feature importance (beeswarm)
shap_summary_bar.png         — global mean |SHAP| bar chart
shap_waterfall_fail.png      — local explanation for a FAIL patient
shap_waterfall_pass.png      — local explanation for a PASS patient
shap_waterfall_border.png    — local explanation for a borderline patient
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

from src.config import (
    CLINICAL_THRESHOLD, DATA_CSV, F1_MODEL_PATH, FEATURE_INFO_PATH,
    F1_HIDDEN_SIZE, F1_NUM_HEADS, F1_NUM_LAYERS, F1_DROPOUT, F1_OUTPUT_SIZE,
    OUTCOME_COL, PATIENT_ID_COL, CONT_RX, CAT_RX,
)
from src.data.preprocessor import load_feature_info, standardize_like_f1

FIG_DIR = "results/figures"

# All prescription feature names (continuous + categorical)
RX_COLS = CONT_RX + [CAT_RX]


def load_data_and_model():
    df   = pd.read_csv(DATA_CSV)
    fi   = load_feature_info(FEATURE_INFO_PATH)
    X_df = standardize_like_f1(df, fi)
    X    = X_df.values.astype(np.float32)
    y    = df[OUTCOME_COL].values.astype(np.float32)
    pids = df[PATIENT_ID_COL].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=pids))
    X_train = X[tr_idx]
    X_test  = X[te_idx]
    y_test  = y[te_idx]

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
    return model, X_train, X_test, y_test, fi, device


def main():
    try:
        import shap
    except ImportError:
        print("shap not installed. Run: pip install shap")
        return

    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading model and data ...")
    model, X_train, X_test, y_test, fi, device = load_data_and_model()
    feature_names = fi["original_feature_names"]

    # Use a subset of train for SHAP background (speed)
    bg_size  = min(100, len(X_train))
    bg_idx   = np.random.choice(len(X_train), bg_size, replace=False)
    X_bg     = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(device)
    X_te_t   = torch.tensor(X_test, dtype=torch.float32).to(device)

    # ── DeepExplainer ────────────────────────────────────────────────────────
    print("Computing SHAP values (DeepExplainer) ...")
    try:
        explainer   = shap.DeepExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_te_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)
    except Exception as e:
        print(f"DeepExplainer failed ({e}), falling back to KernelExplainer ...")
        # KernelExplainer works on numpy
        def predict_fn(x):
            with torch.no_grad():
                t = torch.tensor(x, dtype=torch.float32).to(device)
                return model(t).cpu().numpy().flatten()
        explainer   = shap.KernelExplainer(predict_fn, X_train[bg_idx])
        n_explain   = min(50, len(X_test))
        shap_values = explainer.shap_values(X_test[:n_explain], nsamples=100)
        X_te_np     = X_test[:n_explain]
        y_test      = y_test[:n_explain]

    X_te_np = X_test if "X_te_np" not in dir() else X_te_np

    print(f"SHAP values shape: {shap_values.shape}")

    # ── Global summary — beeswarm ─────────────────────────────────────────────
    print("Plotting global beeswarm ...")
    plt.figure(figsize=(10, max(6, len(feature_names) * 0.25)))
    shap.summary_plot(shap_values, X_te_np, feature_names=feature_names,
                      show=False, max_display=30)
    plt.title("SHAP Summary — F1 SAINT (Global Feature Importance)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {FIG_DIR}/shap_summary_beeswarm.png")

    # ── Global summary — bar ──────────────────────────────────────────────────
    plt.figure(figsize=(8, max(5, len(feature_names) * 0.25)))
    shap.summary_plot(shap_values, X_te_np, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP| — F1 SAINT Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {FIG_DIR}/shap_summary_bar.png")

    # ── Local waterfall for 3 representative patients ─────────────────────────
    thr    = CLINICAL_THRESHOLD
    fail_  = np.where(y_test < thr)[0]
    pass_  = np.where(y_test >= thr)[0]
    border_vals = np.abs(y_test - thr)
    border_     = [np.argmin(border_vals)]

    explainer_exp = shap.Explainer(
        lambda x: model(torch.tensor(x, dtype=torch.float32).to(device))
                       .detach().cpu().numpy().flatten(),
        X_train[bg_idx],
        feature_names=feature_names,
    )

    for tag, indices in [("fail", fail_), ("pass", pass_), ("border", border_)]:
        if len(indices) == 0:
            continue
        i = indices[0]
        try:
            sv = explainer_exp(X_te_np[i:i+1])
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(sv[0], show=False, max_display=15)
            plt.title(f"SHAP Waterfall — {tag.title()} patient "
                      f"(Kt/V={y_test[i]:.2f}, thr={thr})")
            plt.tight_layout()
            plt.savefig(f"{FIG_DIR}/shap_waterfall_{tag}.png", dpi=300,
                        bbox_inches="tight")
            plt.close()
            print(f"  → {FIG_DIR}/shap_waterfall_{tag}.png")
        except Exception as e:
            print(f"  Waterfall for {tag} failed: {e}")

    # ── Prescription-focused SHAP analysis ───────────────────────────────────
    plot_prescription_shap(shap_values, X_te_np, feature_names, y_test)

    print("\nSHAP analysis complete.")


def plot_prescription_shap(
    shap_values: np.ndarray,
    X_te_np: np.ndarray,
    feature_names: list,
    y_test: np.ndarray,
) -> None:
    """Three-panel prescription SHAP analysis.

    1. Beeswarm  — prescription features only, shows direction and magnitude
    2. Bar chart — mean |SHAP| for each prescription feature, with global rank
    3. Dependence plots — one per prescription feature: SHAP value vs feature value

    The goal is to answer: do prescription variables matter for Kt/V prediction,
    and in which direction does each one push the prediction?
    """
    import shap

    # Find which column indices correspond to prescription features
    rx_indices = [
        (i, name) for i, name in enumerate(feature_names) if name in RX_COLS
    ]
    if not rx_indices:
        print("  [prescription SHAP] No prescription features found in feature_names — skipping.")
        return

    rx_idx   = [i for i, _ in rx_indices]
    rx_names = [n for _, n in rx_indices]

    shap_rx = shap_values[:, rx_idx]   # (N, n_rx)
    X_rx    = X_te_np[:, rx_idx]       # (N, n_rx)

    # ── (A) Prescription beeswarm ────────────────────────────────────────────
    print("Plotting prescription beeswarm ...")
    plt.figure(figsize=(9, max(4, len(rx_names) * 0.55)))
    shap.summary_plot(shap_rx, X_rx, feature_names=rx_names,
                      show=False, plot_size=None)
    plt.title("SHAP Summary — Prescription Features Only\n"
              "(each dot = one patient; color = feature value)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_rx_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {FIG_DIR}/shap_rx_beeswarm.png")

    # ── (B) Prescription bar with global rank annotation ────────────────────
    print("Plotting prescription importance bar ...")
    mean_abs_shap_all = np.abs(shap_values).mean(axis=0)   # all features
    mean_abs_shap_rx  = np.abs(shap_rx).mean(axis=0)        # prescription only
    sorted_rx         = sorted(
        zip(rx_names, mean_abs_shap_rx, rx_idx),
        key=lambda x: x[1], reverse=True,
    )
    names_s  = [x[0] for x in sorted_rx]
    shap_s   = [x[1] for x in sorted_rx]
    # Global rank among ALL features (1 = most important overall)
    global_ranks = np.argsort(mean_abs_shap_all)[::-1]
    rank_map = {feat_idx: rank + 1 for rank, feat_idx in enumerate(global_ranks)}
    ranks_s  = [rank_map[x[2]] for x in sorted_rx]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names_s) * 0.5)))
    bars = ax.barh(names_s[::-1], shap_s[::-1], color="#1f77b4", alpha=0.85)
    for bar, rank in zip(bars, ranks_s[::-1]):
        ax.text(
            bar.get_width() + max(shap_s) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"rank #{rank}",
            va="center", ha="left", fontsize=8, color="#555555",
        )
    ax.set_xlabel("Mean |SHAP value| (impact on Kt/V prediction)")
    ax.set_title("Prescription Feature Importance\n(number = global rank among all features)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_rx_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {FIG_DIR}/shap_rx_bar.png")

    # ── (C) Dependence plots — SHAP value vs feature value per prescription col
    print("Plotting prescription dependence plots ...")
    n_rx  = len(rx_names)
    ncols = min(3, n_rx)
    nrows = (n_rx + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    for ax_idx, (col_name, col_shap, col_feat) in enumerate(
        zip(rx_names, shap_rx.T, X_rx.T)
    ):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        sc = ax.scatter(col_feat, col_shap,
                        c=col_shap, cmap="RdBu_r",
                        alpha=0.6, s=18, linewidths=0)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"{col_name}\n(z-score)")
        ax.set_ylabel("SHAP value")
        ax.set_title(col_name, fontsize=9)
        plt.colorbar(sc, ax=ax, label="SHAP")
    # Hide empty subplots
    for ax_idx in range(n_rx, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)
    fig.suptitle(
        "SHAP Dependence — Prescription Features\n"
        "(x = feature value after z-score; y = contribution to Kt/V prediction;\n"
        " red = pushes prediction up, blue = pushes prediction down)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_rx_dependence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {FIG_DIR}/shap_rx_dependence.png")

    # ── (D) Prescription vs non-prescription importance summary ─────────────
    print("Plotting prescription vs non-prescription importance summary ...")
    rx_set      = set(RX_COLS)
    is_rx       = np.array([n in rx_set for n in feature_names])
    total_shap  = mean_abs_shap_all.sum()
    rx_shap_pct = mean_abs_shap_all[is_rx].sum() / total_shap * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: pie chart of prescription vs other
    ax = axes[0]
    wedge_vals  = [mean_abs_shap_all[is_rx].sum(),
                   mean_abs_shap_all[~is_rx].sum()]
    wedge_lbls  = [f"Prescription\n({rx_shap_pct:.1f}%)",
                   f"Non-prescription\n({100 - rx_shap_pct:.1f}%)"]
    ax.pie(wedge_vals, labels=wedge_lbls, colors=["#e74c3c", "#3498db"],
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title("Share of Total SHAP Importance\n(Prescription vs All Other Features)")

    # Right: top-20 global bar with prescription cols highlighted
    ax = axes[1]
    top20_idx   = np.argsort(mean_abs_shap_all)[-20:][::-1]
    top20_names = [feature_names[i] for i in top20_idx]
    top20_vals  = mean_abs_shap_all[top20_idx]
    top20_colors = ["#e74c3c" if n in rx_set else "#3498db" for n in top20_names]
    ax.barh(top20_names[::-1], top20_vals[::-1],
            color=top20_colors[::-1], alpha=0.85)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e74c3c", label="Prescription"),
                        Patch(color="#3498db", label="Other")],
              loc="lower right")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Features — Prescription Highlighted")
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Prescription Feature Contribution to Kt/V Prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_rx_vs_other.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {FIG_DIR}/shap_rx_vs_other.png")


if __name__ == "__main__":
    main()
