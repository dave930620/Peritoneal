"""
analysis/plot_ablations.py — Visualize F1 and F2 ablation study results.

Run from project root:
    python analysis/plot_ablations.py

Reads
-----
results/f1/ablations.json
results/f2/ablations.json

Outputs (results/figures/)
--------------------------
f1_ablation_delta.png   — Δ metric vs full model (F1)
f1_ablation_radar.png   — radar chart (F1 ablations)
f2_ablation_delta.png   — Δ metric vs full model (F2)
f2_ablation_radar.png   — radar chart (F2 ablations)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.results_io import load_results
from src.utils.plotting import plot_ablation_delta_table, plot_radar_chart

F1_PATH = "results/f1/ablations.json"
F2_PATH = "results/f2/ablations.json"
FIG_DIR = "results/figures"

F1_METRICS    = ["mae", "rmse", "auroc", "sensitivity", "specificity", "f1_cls"]
F1_FULL_KEY   = "full_model"
F1_HIGHER     = {"mae": False, "rmse": False, "auroc": True,
                 "sensitivity": True, "specificity": True, "f1_cls": True}

F2_METRICS    = ["delta_ktv_fail", "p_pass", "p_fail", "delta_ktv_pass", "cat_accuracy"]
F2_FULL_KEY   = "full_model"
F2_HIGHER     = {"delta_ktv_fail": True, "p_pass": True,
                 "p_fail": False,        "delta_ktv_pass": False, "cat_accuracy": True}

# Readable short labels for ablation IDs
F1_LABELS = {
    "A-F1-1_no_dual_emb":    "No dual emb",
    "A-F1-2_no_layernorm":   "No LayerNorm",
    "A-F1-3_relu":           "ReLU (vs GELU)",
    "A-F1-4_mlp_backbone":   "MLP backbone",
    "A-F1-5_layers_2":       "2 layers",
    "A-F1-6_layers_4":       "4 layers",
    "A-F1-7_heads_1":        "1 head",
    "A-F1-8_heads_4":        "4 heads",
    "A-F1-9_no_scheduler":   "No LR sched.",
    "A-F1-10_no_weight_decay": "No weight decay",
    "full_model":            "Full SAINT",
}

F2_LABELS = {
    "A-F2-1_no_stage_A":     "No Stage A",
    "A-F2-2_no_trust_region":"No trust region",
    "A-F2-3_no_thr_loss":    "No thr. loss",
    "A-F2-4_no_gate":        "No PASS/FAIL gate",
    "A-F2-5_no_curriculum":  "No curriculum",
    "A-F2-6_no_prox":        "No proximal loss",
    "A-F2-7_linear_head":    "Linear head",
    "A-F2-8_equal_trust":    "Equal trust region",
    "full_model":            "Full F2",
}


def _relabel(results: dict, label_map: dict) -> dict:
    return {label_map.get(k, k): v for k, v in results.items() if "error" not in v}


def main():
    # ── F1 ablations ─────────────────────────────────────────────────────────
    f1_raw = load_results(F1_PATH)
    if f1_raw:
        f1 = _relabel(f1_raw, F1_LABELS)
        full_label = F1_LABELS.get(F1_FULL_KEY, F1_FULL_KEY)

        plot_ablation_delta_table(
            f1, full_model_key=full_label,
            metrics=F1_METRICS,
            title="F1 Ablation Study (Δ vs. Full SAINT)",
            higher_better=F1_HIGHER,
            save_path=f"{FIG_DIR}/f1_ablation_delta.png",
        )

        radar_valid = {k: v for k, v in f1.items()
                       if all(m in v for m in F1_METRICS[:4])}
        if radar_valid:
            plot_radar_chart(
                radar_valid, F1_METRICS[:4],
                title="F1 Ablation Radar",
                higher_better={m: F1_HIGHER[m] for m in F1_METRICS[:4]},
                save_path=f"{FIG_DIR}/f1_ablation_radar.png",
            )
    else:
        print(f"No F1 ablation results at {F1_PATH}. Run experiments first.")

    # ── F2 ablations ─────────────────────────────────────────────────────────
    f2_raw = load_results(F2_PATH)
    if f2_raw:
        f2 = _relabel(f2_raw, F2_LABELS)
        full_label = F2_LABELS.get(F2_FULL_KEY, F2_FULL_KEY)

        # Only p_pass and delta_ktv_fail are always present; others may be nan
        avail_metrics = [m for m in F2_METRICS
                         if any(m in v for v in f2.values())]

        plot_ablation_delta_table(
            f2, full_model_key=full_label,
            metrics=avail_metrics,
            title="F2 Ablation Study (Δ vs. Full F2)",
            higher_better={m: F2_HIGHER.get(m, True) for m in avail_metrics},
            save_path=f"{FIG_DIR}/f2_ablation_delta.png",
        )

        radar_f2 = [m for m in ["delta_ktv_fail", "p_pass", "cat_accuracy"]
                    if all(m in v for v in f2.values())]
        if len(radar_f2) >= 3:
            plot_radar_chart(
                f2, radar_f2,
                title="F2 Ablation Radar",
                higher_better={m: F2_HIGHER.get(m, True) for m in radar_f2},
                save_path=f"{FIG_DIR}/f2_ablation_radar.png",
            )
    else:
        print(f"No F2 ablation results at {F2_PATH}. Run experiments first.")

    print(f"\nAll ablation figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
