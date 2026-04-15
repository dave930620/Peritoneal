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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    CAT_RX, CONT_RX, DATA_CSV, F1_MODEL_PATH, FEATURE_INFO_PATH,
    OUTCOME_COL, PATIENT_ID_COL, SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
)
from src.data.preprocessor import (
    load_feature_info, patient_split, validate_columns,
    build_lag_features, drop_first_visits, lag_feature_names,
)
from src.utils.results_io import result_exists, save_result

np.random.seed(SEED)

RESULTS_PATH = "results/f2_lag/model_search.json"
REPORT_PATH  = "results/f2_lag/model_search_report.txt"
CKPT_DIR     = "results/f2_lag/checkpoints"
MIN_PEARSON  = 0.70   # clinical realism floor

# Phase 1: all Stage A options, fixed Residual Stage B
STAGE_A_TYPES = ["catboost", "rf", "linear"]

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
    """Overall average Pearson (stored directly in metrics dict)."""
    if "avg_pearson" in metrics:
        return float(metrics["avg_pearson"])
    # Fallback: compute from per-column keys (no _doc_ suffix)
    keys = [k for k in metrics if k.startswith("pearson_") and "_doc_" not in k]
    return float(np.nanmean([metrics[k] for k in keys])) if keys else 0.0


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
              feature_info, f1_model, device, lag_cols):
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
            lag_cols=lag_cols,
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
    validate_columns(df_raw, orig_features + [OUTCOME_COL, PATIENT_ID_COL, "記錄時間"], "model_search")

    df_use = df_raw[list(dict.fromkeys(
        [PATIENT_ID_COL, "記錄時間", OUTCOME_COL] + orig_features + CONT_RX + [CAT_RX]
    ))].copy()
    df_use = df_use.loc[:, ~df_use.columns.duplicated()]

    # ── Build lag features and drop first visits ──────────────────────────────
    print("Building lag features ...")
    df_use = build_lag_features(df_use)
    n_before = len(df_use)
    df_use   = drop_first_visits(df_use)
    print(f"  Dropped {n_before - len(df_use)} first-visit rows  ({len(df_use)} remaining)")
    lag_cols = lag_feature_names()

    df_train, df_val, df_test = patient_split(
        df_use, PATIENT_ID_COL, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    print(f"train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    f1_model = load_f1(feature_info, device)

    # ── Phase 1: Stage A search (Stage B = residual) ─────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: Stage A search  (Stage B fixed = residual)")
    print("="*60)

    for sa_type in STAGE_A_TYPES:
        label = f"p1_{sa_type}_residual"
        run_combo(label, sa_type, "residual",
                  df_train, df_val, df_test, feature_info, f1_model, device, lag_cols)

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
        if sb_type == "residual":
            # Already ran as p1_{best_sa_type}_residual — copy to avoid retraining
            src_label = f"p1_{best_sa_type}_residual"
            dst_label = f"p2_{best_sa_type}_residual"
            if not result_exists(RESULTS_PATH, dst_label) and \
               result_exists(RESULTS_PATH, src_label):
                save_result(RESULTS_PATH, dst_label, data[src_label])
            continue
        label = f"p2_{best_sa_type}_{sb_type}"
        run_combo(label, best_sa_type, sb_type,
                  df_train, df_val, df_test, feature_info, f1_model, device, lag_cols)

    # ── Final report ──────────────────────────────────────────────────────
    _write_report(best_sa_type)


def _build_rows(data: dict) -> pd.DataFrame:
    rows = []
    for label, m in data.items():
        err = "error" in m
        rows.append({
            "label":               label,
            "stage_a":             m.get("_stage_a", "?"),
            "stage_b":             m.get("_stage_b", "?"),
            "delta_ktv_doc_fail":  float("nan") if err else m.get("delta_ktv_doc_fail", float("nan")),
            "delta_ktv_doc_pass":  float("nan") if err else m.get("delta_ktv_doc_pass", float("nan")),
            "delta_ktv_all":       float("nan") if err else m.get("delta_ktv_all",       float("nan")),
            "p_rescue":            float("nan") if err else m.get("p_rescue",            float("nan")),
            "p_harm":              float("nan") if err else m.get("p_harm",              float("nan")),
            "avg_pearson":         float("nan") if err else avg_pearson(m),
            "avg_pearson_doc_fail":float("nan") if err else m.get("avg_pearson_doc_fail",float("nan")),
            "avg_pearson_doc_pass":float("nan") if err else m.get("avg_pearson_doc_pass",float("nan")),
            "cat_accuracy":        float("nan") if err else m.get("cat_accuracy",        float("nan")),
            "realistic":           False if err else avg_pearson(m) >= MIN_PEARSON,
            "error":               err,
        })
    return pd.DataFrame(rows).sort_values("delta_ktv_doc_fail", ascending=False)


def _write_report(best_sa_type: str):
    if not os.path.exists(RESULTS_PATH):
        return
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    df = _build_rows(data)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    # ── Console + text report ─────────────────────────────────────────────────
    show_cols = ["label","stage_a","stage_b",
                 "delta_ktv_doc_fail","delta_ktv_doc_pass","delta_ktv_all",
                 "p_rescue","p_harm",
                 "avg_pearson","avg_pearson_doc_fail","avg_pearson_doc_pass",
                 "cat_accuracy","realistic"]
    print("\n" + "="*60)
    print("FULL RESULTS (sorted by delta_ktv_doc_fail):")
    print(df[show_cols].to_string(index=False))

    real   = df[df["realistic"] & ~df["error"]]
    valid  = df[~df["error"]]
    winner = real.iloc[0] if not real.empty else (valid.iloc[0] if not valid.empty else df.iloc[0])
    winner_raw = data.get(winner["label"], {})

    winner_msg = (
        f"\n★ RECOMMENDED MODEL\n"
        f"  Stage A               : {winner['stage_a']}\n"
        f"  Stage B               : {winner['stage_b']}\n"
        f"  delta_ktv_doc_fail    : {winner['delta_ktv_doc_fail']:.4f}  (improvement on doctor-fail patients)\n"
        f"  delta_ktv_doc_pass    : {winner['delta_ktv_doc_pass']:.4f}  (change on doctor-pass patients)\n"
        f"  p_rescue              : {winner['p_rescue']:.4f}  (fraction of fail patients rescued)\n"
        f"  p_harm                : {winner['p_harm']:.4f}  (fraction of pass patients harmed)\n"
        f"  avg_pearson           : {winner['avg_pearson']:.4f}  (overall Rx similarity to doctor)\n"
        f"  avg_pearson_doc_fail  : {winner['avg_pearson_doc_fail']:.4f}  (Rx similarity, fail group — lower expected)\n"
        f"  avg_pearson_doc_pass  : {winner['avg_pearson_doc_pass']:.4f}  (Rx similarity, pass group — higher expected)\n"
        f"  cat_accuracy          : {winner['cat_accuracy']:.4f}\n"
        f"  realistic (pearson>=0.70) : {winner['realistic']}\n"
    )
    print(winner_msg)

    report_text = df[show_cols].to_string(index=False) + "\n\n" + winner_msg
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"Report saved → {REPORT_PATH}")

    # ── Visualizations ────────────────────────────────────────────────────────
    _plot_model_comparison(df)
    _plot_winner_detail(winner_raw, winner["label"])
    _plot_all_combos(data)


def _plot_model_comparison(df: pd.DataFrame):
    """3-panel bar chart comparing all model combinations."""
    valid = df[~df["error"]].reset_index(drop=True)
    if valid.empty:
        return

    labels = [f"{r.stage_a}\n{r.stage_b}" for _, r in valid.iterrows()]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Kt/V improvement
    ax = axes[0]
    ax.bar(x - 0.2, valid["delta_ktv_doc_fail"], 0.4, label="Doc-fail group", color="#e07b39")
    ax.bar(x + 0.2, valid["delta_ktv_doc_pass"], 0.4, label="Doc-pass group", color="#5b8db8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("avg(model Kt/V − doctor Kt/V)")
    ax.set_title("Kt/V Improvement by Group")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Panel 2: Rescue & Harm rates
    ax = axes[1]
    ax.bar(x - 0.2, valid["p_rescue"], 0.4, label="p_rescue (fail→pass) ↑", color="#2ca02c")
    ax.bar(x + 0.2, valid["p_harm"],   0.4, label="p_harm (pass→fail) ↓",   color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_title("Rescue Rate vs Harm Rate")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Panel 3: Pearson by group
    ax = axes[2]
    ax.bar(x - 0.27, valid["avg_pearson"],          0.27, label="Overall",   color="#9467bd")
    ax.bar(x,        valid["avg_pearson_doc_fail"],  0.27, label="Doc-fail",  color="#e07b39")
    ax.bar(x + 0.27, valid["avg_pearson_doc_pass"],  0.27, label="Doc-pass",  color="#5b8db8")
    ax.axhline(MIN_PEARSON, color="red", linestyle="--", linewidth=1, label=f"≥{MIN_PEARSON}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Pearson r  (model Rx vs doctor Rx)")
    ax.set_title("Rx Similarity by Patient Group")
    ax.set_ylim(-0.1, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("F2 Model Search: All Combinations", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(os.path.dirname(REPORT_PATH), "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved → {path}")


def _plot_all_combos(data: dict):
    """One figure per Stage A+B combination: Stage A only vs full pipeline."""
    plot_dir = os.path.join(os.path.dirname(REPORT_PATH), "combo_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Metrics to compare, with display labels and whether higher is better
    METRIC_SPECS = [
        ("delta_ktv_doc_fail",    "ΔKt/V\n(fail pts)",     True),
        ("delta_ktv_doc_pass",    "ΔKt/V\n(pass pts)",     None),   # want ≈0
        ("delta_ktv_all",         "ΔKt/V\n(all pts)",      True),
        ("p_rescue",              "p_rescue\n(fail→pass)",  True),
        ("p_harm",                "p_harm\n(pass→fail)",    False),
        ("avg_pearson",           "Pearson\n(overall)",     True),
        ("avg_pearson_doc_fail",  "Pearson\n(fail pts)",    None),
        ("avg_pearson_doc_pass",  "Pearson\n(pass pts)",    True),
        ("cat_accuracy",          "Cat\naccuracy",          True),
    ]

    for label, m in data.items():
        if "error" in m:
            continue
        sa_type = m.get("_stage_a", "?")
        sb_type = m.get("_stage_b", "?")

        stage_a_vals = [m.get(f"stage_a_{key}", float("nan")) for key, _, _ in METRIC_SPECS]
        full_vals    = [m.get(key,               float("nan")) for key, _, _ in METRIC_SPECS]

        x      = np.arange(len(METRIC_SPECS))
        width  = 0.35
        fig, ax = plt.subplots(figsize=(14, 5))

        bars_a = ax.bar(x - width/2, stage_a_vals, width,
                        label=f"Stage A only ({sa_type})", color="#5b8db8", alpha=0.85)
        bars_b = ax.bar(x + width/2, full_vals,    width,
                        label=f"Stage A+B ({sa_type} + {sb_type})", color="#e07b39", alpha=0.85)

        # Value labels on bars
        for bar in list(bars_a) + list(bars_b):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([spec[1] for spec in METRIC_SPECS], fontsize=9)
        ax.set_ylabel("Metric value")
        ax.set_title(f"Stage A only vs Stage A+B  —  {label}\n"
                     f"Stage A: {sa_type}   Stage B: {sb_type}",
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        safe_label = label.replace("/", "_")
        path = os.path.join(plot_dir, f"{safe_label}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Per-combo plots saved → {plot_dir}/")


def _plot_winner_detail(m: dict, label: str):
    """Per-column Pearson split for the winning model."""
    from src.config import CONT_RX
    cols      = CONT_RX
    safe_cols = [c.replace(" ", "_").replace("/", "_") for c in cols]

    overall   = [m.get(f"pearson_{s}",          float("nan")) for s in safe_cols]
    doc_fail  = [m.get(f"pearson_{s}_doc_fail",  float("nan")) for s in safe_cols]
    doc_pass  = [m.get(f"pearson_{s}_doc_pass",  float("nan")) for s in safe_cols]

    x     = np.arange(len(cols))
    width = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, overall,  width, label="Overall",  color="#9467bd")
    ax.bar(x,         doc_fail, width, label="Doc-fail (model explores more → lower OK)", color="#e07b39")
    ax.bar(x + width, doc_pass, width, label="Doc-pass (model stays close → higher OK)",  color="#5b8db8")
    ax.axhline(MIN_PEARSON, color="red", linestyle="--", linewidth=1, label=f"≥{MIN_PEARSON} target")
    ax.set_xticks(x); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Winner ({label}): Per-Column Rx Pearson — Overall / Doc-Fail / Doc-Pass")
    ax.set_ylim(-0.1, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(os.path.dirname(REPORT_PATH), "winner_pearson_split.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Winner detail plot saved → {path}")


if __name__ == "__main__":
    main()
