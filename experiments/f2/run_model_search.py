"""
experiments/f2/run_model_search.py — Two-phase model search for F2.

Phase 1 — Stage A search (fixed Stage B = MLP):
    Try: catboost, xgboost, rf, linear, mlp_nn, transformer_nn
    → Pick the best Stage A by delta_ktv_fail (Pearson >= 0.70)

Phase 2 — Stage B search (fixed Stage A = best from Phase 1):
    Try: mlp, deep_mlp, residual, transformer
    → Pick the best Stage B

Winner = final proposed model for the paper.

Run from project root:
    python experiments/f2/run_model_search.py

Output
------
results/f2/model_search.json        — all combination metrics
results/f2/model_search_report.txt  — ranked table + winner
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import io, contextlib, json
import numpy as np
import pandas as pd
import torch

from src.config import (
    CAT_RX, CONT_RX, DATA_CSV, F1_MODEL_PATH, FEATURE_INFO_PATH,
    OUTCOME_COL, PATIENT_ID_COL, SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
)
from src.data.preprocessor import load_feature_info, patient_split, validate_columns
from src.utils.results_io import result_exists, save_result

np.random.seed(SEED)

RESULTS_PATH = "results/f2/model_search.json"
REPORT_PATH  = "results/f2/model_search_report.txt"
CKPT_DIR     = "results/f2/checkpoints"
MIN_PEARSON  = 0.70   # clinical realism floor

# Phase 1: all Stage A options, fixed MLP Stage B
STAGE_A_TYPES = ["catboost", "xgboost", "rf", "linear", "mlp_nn", "transformer_nn"]

# Phase 2: all Stage B options, fixed best Stage A
STAGE_B_TYPES = ["mlp", "deep_mlp", "residual", "transformer"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def avg_pearson(metrics: dict) -> float:
    keys = [k for k in metrics if k.startswith("pearson_")]
    return float(np.mean([metrics[k] for k in keys])) if keys else 0.0


def best_from_results(key_prefix: str) -> str:
    """Return the label with highest delta_ktv_fail and Pearson >= MIN_PEARSON."""
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    candidates = {k: v for k, v in data.items()
                  if k.startswith(key_prefix) and "error" not in v}
    if not candidates:
        return None

    realistic = {k: v for k, v in candidates.items()
                 if avg_pearson(v) >= MIN_PEARSON}
    pool = realistic if realistic else candidates
    return max(pool, key=lambda k: pool[k].get("delta_ktv_fail", -999))


def run_combo(label, sa_type, sb_type, df_train, df_val, df_test,
              feature_info, f1_model, device):
    from src.training.f2_core import run_f2_pipeline

    if result_exists(RESULTS_PATH, label):
        print(f"  [skip] {label} already done.")
        return

    print(f"\n{'='*60}")
    print(f"  Stage A: {sa_type}   Stage B: {sb_type}   label: {label}")
    try:
        metrics = run_f2_pipeline(
            df_train, df_val, df_test,
            feature_info, f1_model, device,
            overrides=None,
            ckpt_dir=CKPT_DIR,
            label=label,
            stage_a_type=sa_type,
            stage_b_type=sb_type,
        )
        save_result(RESULTS_PATH, label, {
            **metrics,
            "_stage_a": sa_type,
            "_stage_b": sb_type,
        })
        print(f"  → p_pass={metrics.get('p_pass', float('nan')):.4f}  "
              f"delta_ktv_fail={metrics.get('delta_ktv_fail', float('nan')):.4f}  "
              f"avg_pearson={avg_pearson(metrics):.3f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        save_result(RESULTS_PATH, label, {"error": str(e),
                                          "_stage_a": sa_type, "_stage_b": sb_type})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from src.utils.device import get_device, print_device_info
    device = get_device()
    print_device_info(device)

    print(f"Loading {DATA_CSV} ...")
    df_raw       = pd.read_csv(DATA_CSV)
    feature_info = load_feature_info(FEATURE_INFO_PATH)
    orig_features = feature_info["original_feature_names"]
    validate_columns(df_raw, orig_features + [OUTCOME_COL, PATIENT_ID_COL], "model_search")

    df_use = df_raw[list(dict.fromkeys(
        [PATIENT_ID_COL, OUTCOME_COL] + orig_features + CONT_RX + [CAT_RX]
    ))].copy()
    df_use = df_use.loc[:, ~df_use.columns.duplicated()]

    df_train, df_val, df_test = patient_split(
        df_use, PATIENT_ID_COL, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    print(f"train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    f1_model = load_f1(feature_info, device)

    # ── Phase 1: Stage A search (Stage B = MLP) ───────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: Stage A search  (Stage B fixed = mlp)")
    print("="*60)

    for sa_type in STAGE_A_TYPES:
        label = f"p1_{sa_type}_mlp"
        run_combo(label, sa_type, "mlp",
                  df_train, df_val, df_test, feature_info, f1_model, device)

    best_sa_label = best_from_results("p1_")
    if best_sa_label is None:
        print("Phase 1 produced no valid results. Stopping.")
        return

    with open(RESULTS_PATH) as f:
        data = json.load(f)
    best_sa_type = data[best_sa_label].get("_stage_a", "catboost")

    print(f"\n★ Best Stage A: {best_sa_type}  (from {best_sa_label})")
    print(f"  delta_ktv_fail={data[best_sa_label].get('delta_ktv_fail', '?'):.4f}  "
          f"avg_pearson={avg_pearson(data[best_sa_label]):.3f}")

    # ── Phase 2: Stage B search (best Stage A fixed) ──────────────────────
    print("\n" + "="*60)
    print(f"PHASE 2: Stage B search  (Stage A fixed = {best_sa_type})")
    print("="*60)

    for sb_type in STAGE_B_TYPES:
        if sb_type == "mlp":
            # Already ran as p1_{best_sa_type}_mlp — copy result to avoid retraining
            src_label = f"p1_{best_sa_type}_mlp"
            dst_label = f"p2_{best_sa_type}_mlp"
            if not result_exists(RESULTS_PATH, dst_label) and \
               result_exists(RESULTS_PATH, src_label):
                save_result(RESULTS_PATH, dst_label, data[src_label])
            continue
        label = f"p2_{best_sa_type}_{sb_type}"
        run_combo(label, best_sa_type, sb_type,
                  df_train, df_val, df_test, feature_info, f1_model, device)

    # ── Final report ──────────────────────────────────────────────────────
    _write_report(best_sa_type)


def _write_report(best_sa_type: str):
    if not os.path.exists(RESULTS_PATH):
        return
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    rows = []
    for label, m in data.items():
        if "error" in m:
            rows.append({"label": label, "stage_a": m.get("_stage_a", "?"),
                         "stage_b": m.get("_stage_b", "?"),
                         "delta_ktv_fail": float("nan"), "p_pass": float("nan"),
                         "avg_pearson": float("nan"), "realistic": False})
            continue
        ap = avg_pearson(m)
        rows.append({
            "label":          label,
            "stage_a":        m.get("_stage_a", "?"),
            "stage_b":        m.get("_stage_b", "?"),
            "delta_ktv_fail": m.get("delta_ktv_fail", float("nan")),
            "p_pass":         m.get("p_pass", float("nan")),
            "avg_pearson":    ap,
            "realistic":      ap >= MIN_PEARSON,
        })

    df = pd.DataFrame(rows).sort_values("delta_ktv_fail", ascending=False)
    print("\n" + "="*60)
    print("FULL RESULTS (sorted by delta_ktv_fail):")
    print(df[["label","stage_a","stage_b","delta_ktv_fail",
              "p_pass","avg_pearson","realistic"]].to_string(index=False))

    # Best overall
    real = df[df["realistic"]]
    winner = real.iloc[0] if not real.empty else df.iloc[0]
    winner_msg = (
        f"\n★ RECOMMENDED MODEL\n"
        f"  Stage A : {winner['stage_a']}\n"
        f"  Stage B : {winner['stage_b']}\n"
        f"  delta_ktv_fail : {winner['delta_ktv_fail']:.4f}\n"
        f"  p_pass         : {winner['p_pass']:.4f}\n"
        f"  avg_pearson    : {winner['avg_pearson']:.4f}\n"
        f"  realistic      : {winner['realistic']}\n"
    )
    print(winner_msg)

    report = df.to_string(index=False) + "\n\n" + winner_msg
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
