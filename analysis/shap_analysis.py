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
    OUTCOME_COL, PATIENT_ID_COL,
)
from src.data.preprocessor import load_feature_info, standardize_like_f1

FIG_DIR = "results/figures"


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    print("\nSHAP analysis complete.")


if __name__ == "__main__":
    main()
