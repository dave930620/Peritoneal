"""
experiments/f2/run_baselines.py — Train and evaluate all F2 baselines (C1–C5).

Run from project root:
    python experiments/f2/run_baselines.py

Prerequisites
-------------
train_f1.py must have been run first.

Baselines
---------
C1  doctor_rx         : doctor's recorded prescription (no model)
C2  stage_A_only      : Stage A CatBoost predictions, no Stage B optimization
C3  unconstrained_mlp : Stage B with LAMBDA_CONSTRAINT=0, no projection
C4  random_search     : random search over Rx space, pick best by F1 score
C5  no_stage_A_anchor : Stage B with LAMBDA_PROX=0 (no CatBoost teacher)
full_f2               : Full pipeline (reference)

Output
------
results/f2/baselines.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import io
import contextlib
import numpy as np
import pandas as pd
import torch

from src.config import (
    CAT_RX, CLINICAL_THRESHOLD, CONT_RX, DATA_CSV, F1_MODEL_PATH,
    FEATURE_INFO_PATH, OUTCOME_COL, PATIENT_ID_COL, SEED, STEP_MAP,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
)
from src.data.preprocessor import (
    load_feature_info, patient_split, standardize_like_f1, validate_columns,
)
from src.utils.metrics import f2_metrics, pearson_correlation
from src.utils.results_io import result_exists, save_result
from src.training.f2_core import (
    run_f2_pipeline, train_stage_A, get_stage_A_preds,
    _f1_predict_raw, infer_prescriptions, make_cfg,
)

RESULTS_PATH = "results/f2/baselines.json"
np.random.seed(SEED)


def load_f1(feature_info, device):
    from src.models.saint import SAINT
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = SAINT(
            input_size=len(feature_info["original_feature_names"]),
            hidden_size=64, output_size=1,
            discrete_feature_indices=feature_info["discrete_feature_indices"],
            continuous_feature_indices=feature_info["continuous_feature_indices"],
        )
    model.to(device)
    sd = torch.load(F1_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _compute_f2_metrics_from_rx(rx_df, df_test, feature_info, f1_model, device):
    orig_features = feature_info["original_feature_names"]
    ktv_actual    = df_test[OUTCOME_COL].values.astype(float)
    ktv_doctor    = _f1_predict_raw(df_test[orig_features], feature_info, f1_model, device)

    patched = df_test[orig_features].copy()
    for c in [CAT_RX] + CONT_RX:
        patched[c] = rx_df[c].values
    ktv_model = _f1_predict_raw(patched, feature_info, f1_model, device)

    rx_doctor_dict = {col: df_test[col].values for col in CONT_RX}
    rx_model_dict  = {col: rx_df[col].values   for col in CONT_RX}

    metrics = f2_metrics(ktv_doctor, ktv_model, ktv_actual,
                         rx_doctor_dict, rx_model_dict)
    metrics["cat_accuracy"] = float(
        np.mean(df_test[CAT_RX].astype(int).values ==
                rx_df[CAT_RX].astype(int).values)
    )
    return metrics


# ---------------------------------------------------------------------------
# C1: Doctor's prescription (identity — no model)
# ---------------------------------------------------------------------------

def baseline_doctor(df_test, feature_info, f1_model, device) -> dict:
    """Use doctor's recorded Rx as the 'model' output."""
    rx_df = df_test[[CAT_RX] + CONT_RX].copy()
    return _compute_f2_metrics_from_rx(rx_df, df_test, feature_info, f1_model, device)


# ---------------------------------------------------------------------------
# C2: Stage A CatBoost only
# ---------------------------------------------------------------------------

def baseline_stage_A(df_train, df_val, df_test, feature_info, f1_model, device) -> dict:
    print("[C2] Training Stage A CatBoost ...")
    cb = train_stage_A(df_train, df_val, feature_info)
    te_preds = get_stage_A_preds(df_test, cb, feature_info)

    # De-standardize CatBoost continuous predictions to raw space
    scaler     = feature_info["scaler"]
    cont_names = feature_info["continuous_feature_names"]

    rows = []
    for i in range(len(df_test)):
        row = {CAT_RX: int(te_preds["cat"][i])}
        for j, col in enumerate(CONT_RX):
            raw  = float(te_preds["cont"][i, j])
            step = STEP_MAP[col]
            row[col] = float(np.round(raw / step) * step)
        rows.append(row)

    rx_df = pd.DataFrame(rows, index=df_test.index)
    return _compute_f2_metrics_from_rx(rx_df, df_test, feature_info, f1_model, device)


# ---------------------------------------------------------------------------
# C3: Unconstrained MLP (LAMBDA_CONSTRAINT=0, no projection)
# ---------------------------------------------------------------------------

def baseline_unconstrained(df_train, df_val, df_test, feature_info,
                            f1_model, device) -> dict:
    print("[C3] Unconstrained MLP (no trust region) ...")
    return run_f2_pipeline(
        df_train, df_val, df_test, feature_info, f1_model, device,
        overrides={"LAMBDA_CONSTRAINT": 0.0, "use_projection": False},
        ckpt_dir="results/f2/checkpoints",
        label="C3_unconstrained",
    )


# ---------------------------------------------------------------------------
# C4: Random search over Rx space (upper-bound proxy)
# ---------------------------------------------------------------------------

def baseline_random_search(df_test, feature_info, f1_model, device,
                            n_samples: int = 300) -> dict:
    """For each test patient, sample N_samples random Rx from training distribution,
    pick the one that maximizes F1-predicted Kt/V."""
    print(f"[C4] Random search (N={n_samples} per patient) ...")
    orig_features = feature_info["original_feature_names"]
    Xstd_base     = standardize_like_f1(df_test, feature_info)
    scaler        = feature_info["scaler"]
    cont_names    = feature_info["continuous_feature_names"]

    # Rx column ranges from the standardized space
    def _rx_range(col):
        idx = cont_names.index(col)
        mu, sd = scaler.mean_[idx], scaler.scale_[idx]
        lo = (scaler.mean_[idx] - 3 * sd)
        hi = (scaler.mean_[idx] + 3 * sd)
        return lo, hi

    rx_ranges = {col: _rx_range(col) for col in CONT_RX}
    cat_max   = int(df_test[CAT_RX].max()) + 1

    best_rows = []
    for i in range(len(df_test)):
        x_base = Xstd_base.iloc[i].values.astype(np.float32)
        best_ktv = -np.inf
        best_row = {CAT_RX: int(df_test[CAT_RX].iloc[i])}
        for col in CONT_RX:
            best_row[col] = float(df_test[col].iloc[i])

        cands = []
        for _ in range(n_samples):
            row = {}
            x   = x_base.copy()
            row[CAT_RX] = int(np.random.randint(0, cat_max))
            x[orig_features.index(CAT_RX)] = float(row[CAT_RX])
            for col in CONT_RX:
                lo, hi = rx_ranges[col]
                z_raw = float(np.random.uniform(lo, hi))
                x[orig_features.index(col)] = z_raw
                idx  = cont_names.index(col)
                raw  = z_raw * scaler.scale_[idx] + scaler.mean_[idx]
                step = STEP_MAP[col]
                row[col] = float(np.round(raw / step) * step)
            cands.append((x.copy(), row))

        # Batch evaluate all candidates
        Xbatch = torch.tensor(
            np.stack([c[0] for c in cands]), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            ktv_batch = f1_model(Xbatch).squeeze(-1).cpu().numpy()

        best_idx = int(np.argmax(ktv_batch))
        best_rows.append(cands[best_idx][1])

        if (i + 1) % 50 == 0:
            print(f"  patient {i+1}/{len(df_test)}")

    rx_df = pd.DataFrame(best_rows, index=df_test.index)
    return _compute_f2_metrics_from_rx(rx_df, df_test, feature_info, f1_model, device)


# ---------------------------------------------------------------------------
# C5: No Stage A anchor (LAMBDA_PROX=0)
# ---------------------------------------------------------------------------

def baseline_no_anchor(df_train, df_val, df_test, feature_info,
                       f1_model, device) -> dict:
    print("[C5] No Stage A anchor (LAMBDA_PROX=0) ...")
    return run_f2_pipeline(
        df_train, df_val, df_test, feature_info, f1_model, device,
        overrides={"LAMBDA_PROX_CONT": 0.0, "LAMBDA_PROX_CAT": 0.0,
                   "skip_stage_A": True},
        ckpt_dir="results/f2/checkpoints",
        label="C5_no_anchor",
    )


# ---------------------------------------------------------------------------
# Full F2 (reference)
# ---------------------------------------------------------------------------

def baseline_full_f2(df_train, df_val, df_test, feature_info,
                     f1_model, device) -> dict:
    print("[full_f2] Full pipeline (reference) ...")
    return run_f2_pipeline(
        df_train, df_val, df_test, feature_info, f1_model, device,
        overrides=None,
        ckpt_dir="results/f2/checkpoints",
        label="full_f2",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {DATA_CSV} ...")
    df_raw        = pd.read_csv(DATA_CSV)
    feature_info  = load_feature_info(FEATURE_INFO_PATH)
    orig_features = feature_info["original_feature_names"]
    validate_columns(df_raw, orig_features + [OUTCOME_COL, PATIENT_ID_COL], "f2_baselines")

    df_use = df_raw[list(dict.fromkeys(
        [PATIENT_ID_COL, OUTCOME_COL] + orig_features + CONT_RX + [CAT_RX]
    ))].copy()
    df_use = df_use.loc[:, ~df_use.columns.duplicated()]

    df_train, df_val, df_test = patient_split(df_use, PATIENT_ID_COL,
                                               TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    print(f"train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    f1_model = load_f1(feature_info, device)

    runs = [
        ("doctor_rx",       lambda: baseline_doctor(df_test, feature_info, f1_model, device)),
        ("stage_A_only",    lambda: baseline_stage_A(df_train, df_val, df_test, feature_info, f1_model, device)),
        ("unconstrained",   lambda: baseline_unconstrained(df_train, df_val, df_test, feature_info, f1_model, device)),
        ("random_search",   lambda: baseline_random_search(df_test, feature_info, f1_model, device)),
        ("no_stage_A_anchor", lambda: baseline_no_anchor(df_train, df_val, df_test, feature_info, f1_model, device)),
        ("full_f2",         lambda: baseline_full_f2(df_train, df_val, df_test, feature_info, f1_model, device)),
    ]

    for key, fn in runs:
        if result_exists(RESULTS_PATH, key):
            print(f"[skip] {key} already in results.")
            continue
        print(f"\n{'='*50}\n=== {key} ===")
        try:
            metrics = fn()
            save_result(RESULTS_PATH, key, metrics)
            print(f"  p_pass={metrics.get('p_pass', float('nan')):.4f}  "
                  f"delta_ktv_fail={metrics.get('delta_ktv_fail', float('nan')):.4f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            save_result(RESULTS_PATH, key, {"error": str(e)})

    print(f"\nDone. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
