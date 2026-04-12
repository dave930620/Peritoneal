"""
experiments/f1/run_kfold.py — 5-fold patient-wise GroupKFold for all F1 models.

Run from project root:
    python experiments/f1/run_kfold.py

Output
------
results/f1/kfold.json   {model_name: {metric_mean, metric_std, folds: [...]}}
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from src.config import (
    CLINICAL_THRESHOLD, DATA_CSV, FEATURE_INFO_PATH,
    OUTCOME_COL, PATIENT_ID_COL,
    F1_HIDDEN_SIZE, F1_NUM_HEADS, F1_NUM_LAYERS, F1_DROPOUT, F1_OUTPUT_SIZE,
    F1_EPOCHS, F1_BATCH_SIZE, F1_LR, F1_WEIGHT_DECAY,
)
from src.data.preprocessor import load_feature_info, standardize_like_f1
from src.models.f1_baselines import BASELINE_ORDER, get_f1_baseline
from src.utils.metrics import full_f1_metrics
from src.utils.results_io import save_result, load_results

RESULTS_PATH = "results/f1/kfold.json"
N_FOLDS      = 5
METRIC_KEYS  = ["mae", "rmse", "r2", "pearson", "auroc", "auprc",
                "accuracy", "sensitivity", "specificity", "f1_cls"]


def _saint_fold(X_tr, y_tr, X_va, y_va, fi):
    from src.models.saint import SAINT
    from src.training.f1_core import train_f1_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SAINT(
        input_size=X_tr.shape[1],
        hidden_size=F1_HIDDEN_SIZE, output_size=F1_OUTPUT_SIZE,
        discrete_feature_indices=fi["discrete_feature_indices"],
        continuous_feature_indices=fi["continuous_feature_indices"],
        num_heads=F1_NUM_HEADS, num_layers=F1_NUM_LAYERS, dropout=F1_DROPOUT,
    )
    return train_f1_model(
        model, X_tr, y_tr, X_va, y_va, device,
        epochs=F1_EPOCHS, batch_size=F1_BATCH_SIZE,
        lr=F1_LR, weight_decay=F1_WEIGHT_DECAY,
        threshold=CLINICAL_THRESHOLD, verbose=False,
    )


def run_model_kfold(name, X, y, patient_ids, fi) -> dict:
    """Run N_FOLDS for one model; return aggregated result dict."""
    gkf    = GroupKFold(n_splits=N_FOLDS)
    folds  = []
    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(X, y, groups=patient_ids), 1):
        print(f"  Fold {fold}/{N_FOLDS} ...", end=" ", flush=True)
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if name == "SAINT (ours)":
            metrics = _saint_fold(X_tr, y_tr, X_va, y_va, fi)
        else:
            try:
                model   = get_f1_baseline(name)
                model.fit(X_tr, y_tr)
                preds   = model.predict(X_va)
                metrics = full_f1_metrics(y_va, preds, CLINICAL_THRESHOLD)
            except Exception as e:
                print(f"ERROR: {e}")
                folds.append({k: float("nan") for k in METRIC_KEYS})
                continue
        folds.append(metrics)
        print({k: f"{v:.3f}" for k, v in metrics.items()
               if k in ("mae", "auroc")})

    result = {"folds": folds}
    for key in METRIC_KEYS:
        vals = [f[key] for f in folds if not np.isnan(f.get(key, float("nan")))]
        result[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        result[f"{key}_std"]  = float(np.std(vals))  if vals else float("nan")
    return result


def main():
    print(f"Loading {DATA_CSV} ...")
    df          = pd.read_csv(DATA_CSV)
    fi          = load_feature_info(FEATURE_INFO_PATH)
    X           = standardize_like_f1(df, fi).values.astype(np.float32)
    y           = df[OUTCOME_COL].values.astype(np.float32)
    patient_ids = df[PATIENT_ID_COL].values

    existing = load_results(RESULTS_PATH)
    all_models = ["SAINT (ours)"] + [n.replace("_", " ").title() for n in BASELINE_ORDER]

    for display in all_models:
        if display in existing:
            print(f"[skip] {display} already in results.")
            continue
        print(f"\n=== {display} ===")
        name = (display if display == "SAINT (ours)"
                else display.lower().replace(" ", "_"))
        result = run_model_kfold(name, X, y, patient_ids, fi)
        save_result(RESULTS_PATH, display, result)
        print(f"  → mae={result['mae_mean']:.3f}±{result['mae_std']:.3f}  "
              f"auroc={result['auroc_mean']:.3f}±{result['auroc_std']:.3f}")

    print(f"\nDone. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
