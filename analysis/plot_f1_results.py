"""
analysis/plot_f1_results.py — Visualize F1 baseline comparison results.

Run from project root:
    python analysis/plot_f1_results.py

Reads
-----
results/f1/baselines.json
results/f1/kfold.json

Outputs (results/figures/)
--------------------------
f1_comparison_heatmap.png   — color-coded metric table
f1_bar_mae.png              — horizontal bar: MAE
f1_bar_auroc.png            — horizontal bar: AUROC
f1_bar_sensitivity.png      — horizontal bar: Sensitivity
f1_kfold_mae.png            — kfold mean±std bar: MAE
f1_kfold_auroc.png          — kfold mean±std bar: AUROC
f1_radar.png                — radar chart all metrics
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.results_io import load_results
from src.utils.plotting import (
    plot_metrics_heatmap, plot_bar_comparison,
    plot_kfold_bars, plot_radar_chart,
)

BASELINES_PATH = "results/f1/baselines.json"
KFOLD_PATH     = "results/f1/kfold.json"
FIG_DIR        = "results/figures"

METRICS = ["mae", "rmse", "r2", "pearson", "auroc", "auprc",
           "accuracy", "sensitivity", "specificity", "f1_cls"]

HIGHER_BETTER = {
    "mae": False, "rmse": False,
    "r2": True, "pearson": True, "auroc": True, "auprc": True,
    "accuracy": True, "sensitivity": True, "specificity": True, "f1_cls": True,
}

MODEL_ORDER = [
    "SAINT (ours)",
    "Ft Transformer", "Tabnet", "Vanilla Mlp",
    "Catboost", "Xgboost", "Random Forest", "Ridge",
]

OUR_MODEL = "SAINT (ours)"


def main():
    results = load_results(BASELINES_PATH)
    kfold   = load_results(KFOLD_PATH)

    if not results:
        print(f"No results found at {BASELINES_PATH}. Run experiments first.")
        return

    # Filter to models that have numeric results
    valid = {k: v for k, v in results.items() if "error" not in v}
    order = [m for m in MODEL_ORDER if m in valid] + \
            [m for m in valid if m not in MODEL_ORDER]

    # 1. Heatmap
    plot_metrics_heatmap(
        valid, METRICS, model_order=order,
        title="F1 Model Comparison (Table 1)",
        our_model=OUR_MODEL,
        higher_better=HIGHER_BETTER,
        save_path=f"{FIG_DIR}/f1_comparison_heatmap.png",
    )

    # 2. Bar charts for key metrics
    for metric, ylabel, title in [
        ("mae",         "MAE (↓ better)",         "F1 — Mean Absolute Error"),
        ("auroc",       "AUROC (↑ better)",        "F1 — AUROC (threshold=1.7)"),
        ("sensitivity", "Sensitivity (↑ better)", "F1 — Sensitivity (recall >1.7)"),
        ("rmse",        "RMSE (↓ better)",         "F1 — RMSE"),
    ]:
        avail = {k: v for k, v in valid.items() if metric in v}
        if not avail:
            continue
        ord_m = [m for m in order if m in avail]
        plot_bar_comparison(
            avail, metric, model_order=ord_m,
            title=title, our_model=OUR_MODEL,
            higher_better=HIGHER_BETTER.get(metric, True),
            ylabel=ylabel,
            save_path=f"{FIG_DIR}/f1_bar_{metric}.png",
        )

    # 3. kfold bars
    if kfold:
        valid_kf = {k: v for k, v in kfold.items() if "error" not in v}
        kf_order = [m for m in order if m in valid_kf]
        for metric, ylabel in [("mae", "MAE"), ("auroc", "AUROC")]:
            plot_kfold_bars(
                valid_kf, metric, model_order=kf_order,
                our_model=OUR_MODEL,
                title=f"F1 {metric.upper()} — 5-Fold CV (mean ± std)",
                ylabel=ylabel,
                save_path=f"{FIG_DIR}/f1_kfold_{metric}.png",
            )

    # 4. Radar chart (subset of metrics for readability)
    radar_metrics = ["mae", "auroc", "sensitivity", "specificity", "r2", "f1_cls"]
    radar_higher  = {m: HIGHER_BETTER[m] for m in radar_metrics if m in HIGHER_BETTER}
    radar_valid   = {k: v for k, v in valid.items()
                     if all(m in v for m in radar_metrics)}
    if radar_valid:
        plot_radar_chart(
            radar_valid, radar_metrics,
            model_order=[m for m in order if m in radar_valid],
            title="F1 Multi-Metric Radar",
            higher_better=radar_higher,
            save_path=f"{FIG_DIR}/f1_radar.png",
        )

    print(f"\nAll F1 figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
