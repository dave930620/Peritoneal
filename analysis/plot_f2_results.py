"""
analysis/plot_f2_results.py — Visualize F2 baseline comparison results.

Run from project root:
    python analysis/plot_f2_results.py

Reads
-----
results/f2/baselines.json

Outputs (results/figures/)
--------------------------
f2_grouped_bars.png          — ΔKt/V FAIL group / PASS group per baseline
f2_p_pass_bar.png            — P_pass (proportion FAIL → PASS) per baseline
f2_cat_accuracy_bar.png      — categorical Rx accuracy per baseline
f2_pearson_heatmap.png       — Pearson r between model and doctor Rx per variable
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import CONT_RX
from src.utils.results_io import load_results
from src.utils.plotting import plot_bar_comparison, plot_f2_grouped_bars, _ensure_dir

BASELINES_PATH = "results/f2/baselines.json"
FIG_DIR        = "results/figures"

MODEL_ORDER = [
    "full_f2", "no_stage_A_anchor", "unconstrained",
    "stage_A_only", "random_search", "doctor_rx",
]
DISPLAY_NAMES = {
    "full_f2":          "Full F2 (ours)",
    "no_stage_A_anchor":"No Stage A anchor",
    "unconstrained":    "Unconstrained MLP",
    "stage_A_only":     "Stage A only",
    "random_search":    "Random search",
    "doctor_rx":        "Doctor (baseline)",
}


def plot_pearson_heatmap(results: dict, order: list, save_path: str):
    """Heatmap of Pearson r between model and doctor Rx per continuous variable."""
    # Collect pearson keys
    pearson_keys = [f"pearson_{c.replace(' ', '_').replace('/', '_')}" for c in CONT_RX]
    labels_x     = [c.replace("_", " ") for c in
                    [k.replace("pearson_", "") for k in pearson_keys]]
    labels_y     = [DISPLAY_NAMES.get(m, m) for m in order]

    data = np.full((len(order), len(pearson_keys)), np.nan)
    for i, m in enumerate(order):
        for j, pk in enumerate(pearson_keys):
            data[i, j] = results[m].get(pk, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(CONT_RX) * 1.2), max(4, len(order) * 0.55)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)

    for i in range(len(order)):
        for j in range(len(pearson_keys)):
            v = data[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    ax.set_xticks(range(len(pearson_keys)))
    ax.set_xticklabels([c.replace("_", " ") for c in
                        [k.replace("pearson_", "") for k in pearson_keys]],
                       rotation=35, ha="right")
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y)
    ax.set_title("Pearson r — Model vs. Doctor Rx per variable (↑ better)")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Pearson r")
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[plot] Saved → {save_path}")


def main():
    results = load_results(BASELINES_PATH)
    if not results:
        print(f"No results at {BASELINES_PATH}. Run experiments first.")
        return

    valid = {k: v for k, v in results.items() if "error" not in v}
    order = [m for m in MODEL_ORDER if m in valid] + \
            [m for m in valid  if m not in MODEL_ORDER]
    disp  = {DISPLAY_NAMES.get(m, m): v for m, v in valid.items()}
    disp_order = [DISPLAY_NAMES.get(m, m) for m in order]

    # 1. ΔKt/V grouped bars
    plot_f2_grouped_bars(
        disp,
        metric_fail="delta_ktv_fail",
        metric_pass="delta_ktv_pass",
        model_order=disp_order,
        our_model="Full F2 (ours)",
        title="F2 Prescription Optimization: ΔKt/V by Clinical Group",
        save_path=f"{FIG_DIR}/f2_grouped_bars.png",
    )

    # 2. P_pass bar chart
    plot_bar_comparison(
        disp, "p_pass",
        model_order=disp_order,
        title="P(Kt/V ≥ 1.7 | Doctor FAIL) — higher is better",
        our_model="Full F2 (ours)",
        higher_better=True,
        ylabel="P_pass",
        save_path=f"{FIG_DIR}/f2_p_pass_bar.png",
    )

    # 3. Categorical accuracy
    if any("cat_accuracy" in v for v in valid.values()):
        plot_bar_comparison(
            disp, "cat_accuracy",
            model_order=disp_order,
            title="Categorical Rx Accuracy vs. Doctor",
            our_model="Full F2 (ours)",
            higher_better=True,
            ylabel="Accuracy",
            save_path=f"{FIG_DIR}/f2_cat_accuracy_bar.png",
        )

    # 4. Pearson heatmap over Rx variables
    plot_pearson_heatmap(valid, order, f"{FIG_DIR}/f2_pearson_heatmap.png")

    print(f"\nAll F2 figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
