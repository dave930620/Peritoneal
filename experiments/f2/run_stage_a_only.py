"""
experiments/f2/run_stage_a_only.py — Evaluate Stage A (doctor-mimic) with lag features.

Trains CatBoost Stage A twice:
  - Baseline : patient features only  (no lag, same as current pipeline)
  - Lag      : patient features + previous prescription + previous Kt/V

Compares per-column Pearson, average Pearson, and categorical accuracy.

Run from project root:
    python experiments/f2/run_stage_a_only.py

Output
------
results/stage_a_lag/metrics.json        — per-column Pearson for both variants
results/stage_a_lag/report.txt          — printed comparison table
results/stage_a_lag/pearson_bar.png     — bar chart comparing per-column Pearson
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.config import (
    CAT_RX, CONT_RX, DATA_CSV, FEATURE_INFO_PATH,
    OUTCOME_COL, PATIENT_ID_COL, SEED,
    CATBOOST_ITERATIONS, CATBOOST_LR, CATBOOST_DEPTH, CATBOOST_EARLY_STOP,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
)
from src.data.preprocessor import (
    load_feature_info, patient_split, validate_columns,
    build_lag_features, drop_first_visits, lag_feature_names,
)

np.random.seed(SEED)

RESULTS_DIR  = "results/stage_a_lag"
METRICS_PATH = f"{RESULTS_DIR}/metrics.json"
REPORT_PATH  = f"{RESULTS_DIR}/report.txt"
PLOT_PATH    = f"{RESULTS_DIR}/pearson_bar.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patient_feature_cols(feature_info: dict) -> list:
    """All original features except current Rx columns."""
    rx = set([CAT_RX] + CONT_RX)
    return [c for c in feature_info["original_feature_names"] if c not in rx]


def train_catboost(X_tr: pd.DataFrame, df_tr: pd.DataFrame,
                   X_va: pd.DataFrame, df_va: pd.DataFrame) -> dict:
    """Train one CatBoostRegressor per CONT_RX + one CatBoostClassifier for CAT_RX."""
    from catboost import CatBoostRegressor, CatBoostClassifier
    models = {}
    for col in CONT_RX:
        m = CatBoostRegressor(
            iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
            depth=CATBOOST_DEPTH, loss_function="RMSE", random_seed=SEED, verbose=0,
        )
        m.fit(X_tr, df_tr[col].values,
              eval_set=(X_va, df_va[col].values),
              early_stopping_rounds=CATBOOST_EARLY_STOP)
        models[col] = m

    m_cat = CatBoostClassifier(
        iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
        depth=CATBOOST_DEPTH, loss_function="MultiClass", random_seed=SEED, verbose=0,
    )
    m_cat.fit(X_tr, df_tr[CAT_RX].astype(int).values,
              eval_set=(X_va, df_va[CAT_RX].astype(int).values),
              early_stopping_rounds=CATBOOST_EARLY_STOP)
    models[CAT_RX] = m_cat
    return models


def evaluate(models: dict, X_te: pd.DataFrame, df_te: pd.DataFrame) -> tuple:
    """Compute per-column Pearson and categorical accuracy.

    Returns
    -------
    metrics : dict   — Pearson/accuracy summary
    preds   : dict   — raw predictions per column (for scatter plots)
    """
    metrics = {}
    preds   = {}
    pearsons = []
    for col in CONT_RX:
        pred  = models[col].predict(X_te).flatten()
        truth = df_te[col].values
        r, _  = pearsonr(pred, truth)
        metrics[f"pearson_{col}"] = float(r)
        preds[col] = {"pred": pred, "truth": truth}
        pearsons.append(r)
    metrics["avg_pearson_cont"] = float(np.mean(pearsons))

    cat_pred  = models[CAT_RX].predict(X_te).flatten().astype(int)
    cat_truth = df_te[CAT_RX].astype(int).values
    metrics["cat_accuracy"] = float((cat_pred == cat_truth).mean())
    preds[CAT_RX] = {"pred": cat_pred, "truth": cat_truth}
    return metrics, preds


def print_table(baseline: dict, lag: dict) -> str:
    cols = CONT_RX + [CAT_RX]
    lines = []
    lines.append(f"{'Column':<30}  {'Baseline':>10}  {'With Lag':>10}  {'Delta':>10}")
    lines.append("-" * 64)
    for col in CONT_RX:
        key = f"pearson_{col}"
        b = baseline.get(key, float("nan"))
        l = lag.get(key, float("nan"))
        lines.append(f"  {col:<28}  {b:>10.4f}  {l:>10.4f}  {l-b:>+10.4f}")
    lines.append("-" * 64)
    b_avg = baseline.get("avg_pearson_cont", float("nan"))
    l_avg = lag.get("avg_pearson_cont", float("nan"))
    lines.append(f"  {'Average Pearson (cont)':<28}  {b_avg:>10.4f}  {l_avg:>10.4f}  {l_avg-b_avg:>+10.4f}")
    b_acc = baseline.get("cat_accuracy", float("nan"))
    l_acc = lag.get("cat_accuracy", float("nan"))
    lines.append(f"  {'Cat accuracy':<28}  {b_acc:>10.4f}  {l_acc:>10.4f}  {l_acc-b_acc:>+10.4f}")
    return "\n".join(lines)


def plot_scatter_grid(preds_base: dict, preds_lag: dict):
    """2-column scatter grid: left = baseline, right = with lag, one row per Rx variable."""
    cols  = CONT_RX
    n     = len(cols)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n))

    for i, col in enumerate(cols):
        for j, (label, preds) in enumerate([("Baseline (no lag)", preds_base),
                                             ("With lag features",  preds_lag)]):
            ax    = axes[i, j]
            pred  = preds[col]["pred"]
            truth = preds[col]["truth"]
            r, _  = pearsonr(pred, truth)

            # Scatter
            ax.scatter(truth, pred, alpha=0.35, s=12,
                       color="#5b8db8" if j == 0 else "#e07b39")

            # Identity line
            lo = min(truth.min(), pred.min())
            hi = max(truth.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6)

            ax.set_xlabel("Ground truth", fontsize=8)
            ax.set_ylabel("Predicted",    fontsize=8)
            ax.set_title(f"{col}  |  {label}\nr = {r:.4f}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)

    plt.suptitle("Stage A: Predicted vs Ground Truth per Rx Variable", fontsize=12, y=1.01)
    plt.tight_layout()
    scatter_path = os.path.join(RESULTS_DIR, "scatter_grid.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter grid saved → {scatter_path}")


def plot_pearson(baseline: dict, lag: dict):
    cols  = CONT_RX
    keys  = [f"pearson_{c}" for c in cols]
    b_vals = [baseline.get(k, 0) for k in keys]
    l_vals = [lag.get(k, 0)      for k in keys]

    x     = np.arange(len(cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, b_vals, width, label="Baseline (no lag)", color="#5b8db8")
    ax.bar(x + width/2, l_vals, width, label="With lag features", color="#e07b39")
    ax.axhline(0.70, color="red", linestyle="--", linewidth=1, label="0.70 target")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r")
    ax.set_title("Stage A Pearson per Rx variable: Baseline vs Lag Features")
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Plot saved → {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load raw data ─────────────────────────────────────────────────────────
    print(f"Loading {DATA_CSV} ...")
    df_raw       = pd.read_csv(DATA_CSV)
    feature_info = load_feature_info(FEATURE_INFO_PATH)
    orig_features = feature_info["original_feature_names"]
    pat_cols      = patient_feature_cols(feature_info)

    validate_columns(df_raw, orig_features + [OUTCOME_COL, PATIENT_ID_COL, "記錄時間"], "stage_a_only")

    # ── Build lag features and drop first visits ───────────────────────────────
    print("Building lag features and dropping first visits ...")
    df_lag = build_lag_features(df_raw)
    n_before = len(df_lag)
    df_lag   = drop_first_visits(df_lag)
    n_after  = len(df_lag)
    print(f"  Dropped {n_before - n_after} first-visit rows  ({n_after} remaining)")

    lag_cols = lag_feature_names()
    print(f"  Lag feature columns: {lag_cols}")

    # ── Patient-level train / val / test split ─────────────────────────────────
    df_train, df_val, df_test = patient_split(
        df_lag, PATIENT_ID_COL, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    print(f"Train={len(df_train)}  Val={len(df_val)}  Test={len(df_test)}")

    # ── Variant A: Baseline (patient features only) ────────────────────────────
    print("\n" + "="*60)
    print("VARIANT A: Baseline — patient features only (no lag)")
    print("="*60)
    X_tr_base = df_train[pat_cols]
    X_va_base = df_val[pat_cols]
    X_te_base = df_test[pat_cols]
    models_base = train_catboost(X_tr_base, df_train, X_va_base, df_val)
    metrics_base, preds_base = evaluate(models_base, X_te_base, df_test)
    print(f"  avg_pearson={metrics_base['avg_pearson_cont']:.4f}  "
          f"cat_acc={metrics_base['cat_accuracy']:.4f}")

    # ── Variant B: With lag features ──────────────────────────────────────────
    print("\n" + "="*60)
    print("VARIANT B: With lag features")
    print("="*60)
    input_cols   = pat_cols + lag_cols
    X_tr_lag     = df_train[input_cols]
    X_va_lag     = df_val[input_cols]
    X_te_lag     = df_test[input_cols]
    models_lag   = train_catboost(X_tr_lag, df_train, X_va_lag, df_val)
    metrics_lag, preds_lag = evaluate(models_lag, X_te_lag, df_test)
    print(f"  avg_pearson={metrics_lag['avg_pearson_cont']:.4f}  "
          f"cat_acc={metrics_lag['cat_accuracy']:.4f}")

    # ── Save and report ───────────────────────────────────────────────────────
    with open(METRICS_PATH, "w") as f:
        json.dump({"baseline": metrics_base, "lag": metrics_lag}, f, indent=2)
    print(f"\nMetrics saved → {METRICS_PATH}")

    table = print_table(metrics_base, metrics_lag)
    print("\n" + table)

    with open(REPORT_PATH, "w") as f:
        f.write("Stage A Only: Baseline vs Lag Features\n")
        f.write("="*64 + "\n\n")
        f.write(table + "\n")
    print(f"Report saved → {REPORT_PATH}")

    plot_pearson(metrics_base, metrics_lag)
    plot_scatter_grid(preds_base, preds_lag)


if __name__ == "__main__":
    main()
