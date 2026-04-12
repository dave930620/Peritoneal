"""
experiments/f1/run_baselines.py — Train and evaluate all F1 baselines (B1–B8).

Run from project root:
    python experiments/f1/run_baselines.py

Prerequisites
-------------
- train_f1.py must have been run first (produces feature_info.pkl + saint_pd_model.pth)

Output
------
results/f1/baselines.json   {model_name: {mae, rmse, r2, pearson, auroc, ...}}
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    CLINICAL_THRESHOLD, DATA_CSV, FEATURE_INFO_PATH, F1_MODEL_PATH,
    OUTCOME_COL, PATIENT_ID_COL,
)
from src.data.preprocessor import load_feature_info, standardize_like_f1
from src.models.f1_baselines import BASELINE_ORDER, get_f1_baseline
from src.utils.metrics import full_f1_metrics
from src.utils.results_io import result_exists, save_result

RESULTS_PATH = "results/f1/baselines.json"


def load_data_split():
    """Load data and return the SAME train/test split as train_f1.py."""
    print(f"Loading {DATA_CSV} ...")
    df    = pd.read_csv(DATA_CSV)
    fi    = load_feature_info(FEATURE_INFO_PATH)

    X_df        = standardize_like_f1(df, fi)
    X           = X_df.values.astype(np.float32)
    y           = df[OUTCOME_COL].values.astype(np.float32)
    patient_ids = df[PATIENT_ID_COL].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
            fi)


def eval_saint(X_test, y_test, feature_info):
    """Evaluate the saved SAINT model from train_f1.py on the test set."""
    from src.models.saint import SAINT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fi = feature_info

    model = SAINT(
        input_size=X_test.shape[1],
        hidden_size=64,
        output_size=1,
        discrete_feature_indices=fi["discrete_feature_indices"],
        continuous_feature_indices=fi["continuous_feature_indices"],
    ).to(device)
    sd = torch.load(F1_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        Xt    = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(Xt).cpu().numpy().flatten()
    return full_f1_metrics(y_test, preds, CLINICAL_THRESHOLD)


def main():
    X_train, X_test, y_train, y_test, fi = load_data_split()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # ── SAINT (ours) — load saved model ──────────────────────────────────────
    key = "SAINT (ours)"
    if result_exists(RESULTS_PATH, key):
        print(f"[skip] {key} already in results.")
    else:
        print(f"\n=== {key} ===")
        metrics = eval_saint(X_test, y_test, fi)
        save_result(RESULTS_PATH, key, metrics)
        print(metrics)

    # ── sklearn / XGBoost / CatBoost / TabNet / FT-Transformer ───────────────
    for name in BASELINE_ORDER:
        display = name.replace("_", " ").title()
        if result_exists(RESULTS_PATH, display):
            print(f"[skip] {display} already in results.")
            continue
        print(f"\n=== {display} ===")
        try:
            model = get_f1_baseline(name)
            model.fit(X_train, y_train)
            preds   = model.predict(X_test)
            metrics = full_f1_metrics(y_test, preds, CLINICAL_THRESHOLD)
            save_result(RESULTS_PATH, display, metrics)
            print(metrics)
        except Exception as e:
            print(f"  [ERROR] {display} failed: {e}")
            save_result(RESULTS_PATH, display, {"error": str(e)})

    print(f"\nDone. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
