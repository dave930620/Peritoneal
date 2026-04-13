"""
experiments/f1/run_ablations.py — Train and evaluate all F1 ablations (A-F1-1 to A-F1-10).

Run from project root:
    python experiments/f1/run_ablations.py

Output
------
results/f1/ablations.json   {ablation_id: {mae, rmse, ...}}
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    CLINICAL_THRESHOLD, DATA_CSV, FEATURE_INFO_PATH,
    F1_EPOCHS, F1_BATCH_SIZE, F1_LR, F1_WEIGHT_DECAY,
    F1_HIDDEN_SIZE, F1_NUM_HEADS, F1_NUM_LAYERS, F1_DROPOUT, F1_OUTPUT_SIZE,
    OUTCOME_COL, PATIENT_ID_COL,
)
from src.data.preprocessor import load_feature_info, standardize_like_f1
from src.models.saint_variant import SAINTVariant
from src.training.f1_core import train_f1_model
from src.utils.results_io import result_exists, save_result

RESULTS_PATH = "results/f1/ablations.json"

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
# Each entry: (ablation_id, description, model_kwargs, training_kwargs)
# model_kwargs   → passed to SAINTVariant
# training_kwargs → passed to train_f1_model (overrides)

ABLATIONS = [
    # ── Architecture ablations ───────────────────────────────────────────────
    ("full_model",
     "Full SAINT (reference)",
     dict(use_dual_embedding=True, use_layer_norm=True, activation="gelu",
          use_transformer=True, num_layers=6, num_heads=8),
     dict()),

    ("A-F1-1_no_dual_emb",
     "No dual embedding (single Linear)",
     dict(use_dual_embedding=False, use_layer_norm=True, activation="gelu",
          use_transformer=True, num_layers=6, num_heads=8),
     dict()),

    ("A-F1-2_no_layernorm",
     "No LayerNorm in embeddings",
     dict(use_dual_embedding=True, use_layer_norm=False, activation="gelu",
          use_transformer=True, num_layers=6, num_heads=8),
     dict()),

    ("A-F1-3_relu",
     "ReLU activation (vs GELU)",
     dict(use_dual_embedding=True, use_layer_norm=True, activation="relu",
          use_transformer=True, num_layers=6, num_heads=8),
     dict()),

    ("A-F1-4_mlp_backbone",
     "MLP backbone (no Transformer)",
     dict(use_dual_embedding=True, use_layer_norm=True, activation="gelu",
          use_transformer=False, num_layers=6, num_heads=8),
     dict()),

    # ── Depth / heads ────────────────────────────────────────────────────────
    ("A-F1-5_layers_2",
     "num_layers=2",
     dict(use_dual_embedding=True, use_layer_norm=True, activation="gelu",
          use_transformer=True, num_layers=2, num_heads=8),
     dict()),

    ("A-F1-6_layers_4",
     "num_layers=4",
     dict(use_dual_embedding=True, use_layer_norm=True, activation="gelu",
          use_transformer=True, num_layers=4, num_heads=8),
     dict()),

    ("A-F1-7_heads_1",
     "num_heads=1",
     dict(use_dual_embedding=True, use_layer_norm=True, activation="gelu",
          use_transformer=True, num_layers=6, num_heads=1),
     dict()),

    ("A-F1-8_heads_4",
     "num_heads=4",
     dict(use_dual_embedding=True, use_layer_norm=True, activation="gelu",
          use_transformer=True, num_layers=6, num_heads=4),
     dict()),

    # ── Training strategy ────────────────────────────────────────────────────
    ("A-F1-9_no_scheduler",
     "No LR scheduler",
     dict(use_dual_embedding=True, use_layer_norm=True, activation="gelu",
          use_transformer=True, num_layers=6, num_heads=8),
     dict(use_scheduler=False)),

    ("A-F1-10_no_weight_decay",
     "No weight decay",
     dict(use_dual_embedding=True, use_layer_norm=True, activation="gelu",
          use_transformer=True, num_layers=6, num_heads=8),
     dict(weight_decay=0.0)),
]


def main():
    print(f"Loading {DATA_CSV} ...")
    df          = pd.read_csv(DATA_CSV)
    fi          = load_feature_info(FEATURE_INFO_PATH)
    X           = standardize_like_f1(df, fi).values.astype(np.float32)
    y           = df[OUTCOME_COL].values.astype(np.float32)
    patient_ids = df[PATIENT_ID_COL].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=patient_ids))
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    from src.utils.device import get_device, print_device_info
    device = get_device()
    print_device_info(device)

    for abl_id, desc, model_kw, train_kw in ABLATIONS:
        if result_exists(RESULTS_PATH, abl_id):
            print(f"[skip] {abl_id} already in results.")
            continue

        print(f"\n=== {abl_id}: {desc} ===")
        model = SAINTVariant(
            input_size=X_train.shape[1],
            hidden_size=F1_HIDDEN_SIZE,
            output_size=F1_OUTPUT_SIZE,
            discrete_feature_indices=fi["discrete_feature_indices"],
            continuous_feature_indices=fi["continuous_feature_indices"],
            dropout=F1_DROPOUT,
            **model_kw,
        )

        # Build training kwargs (start from defaults, apply overrides)
        tkw = dict(
            epochs=F1_EPOCHS, batch_size=F1_BATCH_SIZE,
            lr=F1_LR, weight_decay=F1_WEIGHT_DECAY,
            use_scheduler=True, threshold=CLINICAL_THRESHOLD, verbose=True,
        )
        tkw.update(train_kw)

        try:
            metrics = train_f1_model(model, X_train, y_train, X_test, y_test,
                                     device, **tkw)
            metrics["description"] = desc
            save_result(RESULTS_PATH, abl_id, metrics)
            print(f"  → mae={metrics['mae']:.4f}  auroc={metrics['auroc']:.4f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            save_result(RESULTS_PATH, abl_id, {"description": desc, "error": str(e)})

    print(f"\nDone. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
