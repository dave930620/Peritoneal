"""
experiments/f2/run_ablations.py — Train and evaluate all F2 ablations (A-F2-1 to A-F2-8).

Run from project root:
    python experiments/f2/run_ablations.py

Output
------
results/f2/ablations.json
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
    FEATURE_INFO_PATH, OUTCOME_COL, PATIENT_ID_COL, SEED,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
    EPS_CONT_Z_FAIL, EPS_CAT_SOFT_FAIL,
)
from src.data.preprocessor import load_feature_info, patient_split, validate_columns
from src.utils.results_io import result_exists, save_result
from src.training.f2_core import run_f2_pipeline

RESULTS_PATH = "results/f2/ablations.json"
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
# (ablation_id, description, config_overrides)

ABLATIONS = [
    ("full_model",
     "Full F2 (reference)",
     {}),

    ("A-F2-1_no_stage_A",
     "No Stage A — no CatBoost teacher",
     {"skip_stage_A": True,
      "LAMBDA_PROX_CONT": 0.0, "LAMBDA_PROX_CAT": 0.0}),

    ("A-F2-2_no_trust_region",
     "No trust-region constraint",
     {"LAMBDA_CONSTRAINT": 0.0, "use_projection": False}),

    ("A-F2-3_no_thr_loss",
     "No threshold push loss",
     {"LAMBDA_THR_FAIL": 0.0, "LAMBDA_THR_PASS": 0.0}),

    ("A-F2-4_no_gate",
     "No PASS/FAIL gating (uniform weights)",
     {"uniform_weights": True}),

    ("A-F2-5_no_curriculum",
     "No curriculum (fixed lc=1.0)",
     {"fixed_lc": 1.0}),

    ("A-F2-6_no_prox",
     "No proximal loss (keep Stage A for reference)",
     {"LAMBDA_PROX_CONT": 0.0, "LAMBDA_PROX_CAT": 0.0}),

    ("A-F2-7_linear_head",
     "Linear Rx head (no MLP backbone)",
     {"use_linear_head": True}),

    ("A-F2-8_equal_trust",
     "Equal trust-region for PASS and FAIL",
     {"EPS_CONT_Z_PASS": EPS_CONT_Z_FAIL,
      "EPS_CAT_SOFT_PASS": EPS_CAT_SOFT_FAIL}),
]


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


def main():
    from src.utils.device import get_device, print_device_info
    device = get_device()
    print_device_info(device)

    print(f"Loading {DATA_CSV} ...")
    df_raw       = pd.read_csv(DATA_CSV)
    feature_info = load_feature_info(FEATURE_INFO_PATH)
    orig_features = feature_info["original_feature_names"]
    validate_columns(df_raw, orig_features + [OUTCOME_COL, PATIENT_ID_COL], "f2_ablations")

    df_use = df_raw[list(dict.fromkeys(
        [PATIENT_ID_COL, OUTCOME_COL] + orig_features + CONT_RX + [CAT_RX]
    ))].copy()
    df_use = df_use.loc[:, ~df_use.columns.duplicated()]

    df_train, df_val, df_test = patient_split(df_use, PATIENT_ID_COL,
                                               TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    print(f"train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    f1_model = load_f1(feature_info, device)

    for abl_id, desc, overrides in ABLATIONS:
        if result_exists(RESULTS_PATH, abl_id):
            print(f"[skip] {abl_id} already in results.")
            continue

        print(f"\n{'='*50}\n=== {abl_id}: {desc} ===")
        try:
            metrics = run_f2_pipeline(
                df_train, df_val, df_test,
                feature_info, f1_model, device,
                overrides=overrides,
                ckpt_dir=f"results/f2/checkpoints/{abl_id}",
                label=abl_id,
            )
            metrics["description"] = desc
            save_result(RESULTS_PATH, abl_id, metrics)
            print(f"  p_pass={metrics.get('p_pass', float('nan')):.4f}  "
                  f"delta_ktv_fail={metrics.get('delta_ktv_fail', float('nan')):.4f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            save_result(RESULTS_PATH, abl_id, {"description": desc, "error": str(e)})

    print(f"\nDone. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
