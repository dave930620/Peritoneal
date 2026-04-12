"""
plotting.py — Shared publication-quality plot helpers for J-BHI figures.

All functions save to a given path at 300 DPI and return the figure object.
Use matplotlib with a clean style suitable for single-column / double-column figures.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PALETTE = sns.color_palette("tab10")
HIGHLIGHT_COLOR = "#d62728"   # red for "our model"
BASE_COLOR      = "#aec7e8"   # light blue for baselines


# ── Helper ───────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ── 1. Metrics heatmap table (models × metrics) ──────────────────────────────

def plot_metrics_heatmap(
    results:    Dict[str, dict],
    metrics:    List[str],
    model_order: Optional[List[str]] = None,
    title:      str = "Model Comparison",
    our_model:  str = "SAINT (ours)",
    save_path:  str = "results/figures/heatmap.png",
    higher_better: Optional[Dict[str, bool]] = None,
) -> plt.Figure:
    """Render a color-coded comparison table (rows = models, cols = metrics).

    Cell color reflects rank (green = best, red = worst per column).
    The row matching `our_model` is outlined in a thick border.

    Parameters
    ----------
    results      : {model_name: {metric: value}}
    metrics      : ordered list of metric keys to display
    model_order  : row ordering; defaults to results.keys()
    higher_better: {metric: True/False}; defaults to True for all except mae/rmse
    """
    if higher_better is None:
        higher_better = {m: (m not in ("mae", "rmse")) for m in metrics}

    models = model_order if model_order else list(results.keys())
    data   = np.array([[results[m].get(k, float("nan")) for k in metrics]
                       for m in models], dtype=float)

    # Per-column rank-normalize (0 = worst, 1 = best)
    normed = np.full_like(data, 0.5)
    for j, metric in enumerate(metrics):
        col  = data[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        lo, hi = col[valid].min(), col[valid].max()
        if hi == lo:
            normed[valid, j] = 1.0
            continue
        norm = (col[valid] - lo) / (hi - lo)
        normed[valid, j] = norm if higher_better.get(metric, True) else 1 - norm

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.1), max(4, len(models) * 0.55)))
    im = ax.imshow(normed, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Cell text
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            val = data[i, j]
            txt = f"{val:.3f}" if not np.isnan(val) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < normed[i, j] < 0.85 else "white")

    # Axes labels
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=40, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    # Highlight "our model" row
    if our_model in models:
        row = models.index(our_model)
        rect = mpatches.FancyBboxPatch(
            (-0.5, row - 0.5), len(metrics), 1,
            linewidth=2.5, edgecolor="#1f77b4", facecolor="none",
            boxstyle="square,pad=0",
        )
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Normalized rank (1=best)")
    ax.set_title(title)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved heatmap → {save_path}")
    return fig


# ── 2. Bar chart for a single metric ─────────────────────────────────────────

def plot_bar_comparison(
    results:    Dict[str, dict],
    metric:     str,
    model_order: Optional[List[str]] = None,
    title:      str = "",
    our_model:  str = "SAINT (ours)",
    yerr_key:   Optional[str] = None,
    higher_better: bool = True,
    ylabel:     str = "",
    save_path:  str = "results/figures/bar.png",
) -> plt.Figure:
    """Horizontal bar chart for one metric across all models.

    Parameters
    ----------
    yerr_key : if provided, use results[model][yerr_key] as ±error bar
    """
    models = model_order if model_order else list(results.keys())
    vals   = [results[m].get(metric, float("nan")) for m in models]
    errs   = [results[m].get(yerr_key, 0.0) for m in models] if yerr_key else None

    colors = [HIGHLIGHT_COLOR if m == our_model else BASE_COLOR for m in models]

    fig, ax = plt.subplots(figsize=(7, max(3, len(models) * 0.45)))
    bars = ax.barh(models, vals, xerr=errs, color=colors,
                   capsize=3, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(v + (max([x for x in vals if not np.isnan(x)]) * 0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)

    ax.set_xlabel(ylabel or metric)
    ax.set_title(title or metric)
    ax.invert_yaxis()
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved bar chart → {save_path}")
    return fig


# ── 3. Grouped bar chart for F2 (FAIL / PASS group side-by-side) ─────────────

def plot_f2_grouped_bars(
    results:    Dict[str, dict],
    metric_fail: str = "delta_ktv_fail",
    metric_pass: str = "delta_ktv_pass",
    model_order: Optional[List[str]] = None,
    our_model:  str = "full_f2",
    title:      str = "F2 Optimization Results",
    save_path:  str = "results/figures/f2_bars.png",
) -> plt.Figure:
    """Side-by-side bars for FAIL-group and PASS-group metrics."""
    models = model_order if model_order else list(results.keys())
    fail_vals = [results[m].get(metric_fail, float("nan")) for m in models]
    pass_vals = [results[m].get(metric_pass, float("nan")) for m in models]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    b1 = ax.bar(x - width / 2, fail_vals, width, label="FAIL group (Kt/V<1.7)",
                color="#d62728", alpha=0.8)
    b2 = ax.bar(x + width / 2, pass_vals, width, label="PASS group (Kt/V≥1.7)",
                color="#1f77b4", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("ΔKt/V (model − doctor, F1-predicted)")
    ax.set_title(title)
    ax.legend()

    # Outline our model
    if our_model in models:
        idx = models.index(our_model)
        for bar in [b1[idx], b2[idx]]:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved grouped bars → {save_path}")
    return fig


# ── 4. Ablation delta table ───────────────────────────────────────────────────

def plot_ablation_delta_table(
    ablation_results: Dict[str, dict],
    full_model_key:   str,
    metrics:          List[str],
    title:            str = "Ablation Study",
    higher_better:    Optional[Dict[str, bool]] = None,
    save_path:        str = "results/figures/ablation_delta.png",
) -> plt.Figure:
    """Heatmap showing Δ metric vs. full model.  Red = worse, Green = better."""
    if higher_better is None:
        higher_better = {m: (m not in ("mae", "rmse")) for m in metrics}

    full   = ablation_results.get(full_model_key, {})
    keys   = [k for k in ablation_results if k != full_model_key]
    labels = keys

    # Δ = ablation_value - full_value; positive = better for higher-is-better
    deltas = np.zeros((len(keys), len(metrics)), dtype=float)
    for i, key in enumerate(keys):
        for j, metric in enumerate(metrics):
            abl_val  = ablation_results[key].get(metric, float("nan"))
            full_val = full.get(metric, float("nan"))
            if np.isnan(abl_val) or np.isnan(full_val):
                deltas[i, j] = float("nan")
            else:
                d = abl_val - full_val
                deltas[i, j] = d if higher_better.get(metric, True) else -d

    abs_max = np.nanmax(np.abs(deltas)) if not np.all(np.isnan(deltas)) else 1.0

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.1), max(3, len(keys) * 0.55)))
    im = ax.imshow(deltas, aspect="auto", cmap="RdYlGn",
                   vmin=-abs_max, vmax=abs_max)

    for i in range(len(keys)):
        for j in range(len(metrics)):
            v = deltas[i, j]
            txt = f"{v:+.3f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(f"{title}\n(Δ vs. full model; green=better)")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Δ (normalized direction)")
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved ablation delta table → {save_path}")
    return fig


# ── 5. Radar chart ────────────────────────────────────────────────────────────

def plot_radar_chart(
    results:    Dict[str, dict],
    metrics:    List[str],
    model_order: Optional[List[str]] = None,
    title:      str = "Radar Comparison",
    higher_better: Optional[Dict[str, bool]] = None,
    save_path:  str = "results/figures/radar.png",
) -> plt.Figure:
    """Radar (spider) chart — each axis is a metric, normalized 0-1."""
    if higher_better is None:
        higher_better = {m: (m not in ("mae", "rmse")) for m in metrics}

    models = model_order if model_order else list(results.keys())
    N      = len(metrics)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    # Normalize each metric column to [0, 1]
    raw = np.array([[results[m].get(k, float("nan")) for k in metrics]
                    for m in models], dtype=float)
    normed = np.full_like(raw, 0.5)
    for j, metric in enumerate(metrics):
        col   = raw[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        lo, hi = col[valid].min(), col[valid].max()
        if hi == lo:
            normed[valid, j] = 1.0
            continue
        n = (col[valid] - lo) / (hi - lo)
        normed[valid, j] = n if higher_better.get(metric, True) else 1 - n

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = list(normed[i]) + [normed[i][0]]
        ax.plot(angles, vals, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved radar chart → {save_path}")
    return fig


# ── 6. kfold bar chart (mean ± std) ──────────────────────────────────────────

def plot_kfold_bars(
    kfold_results: Dict[str, dict],
    metric:        str,
    model_order:   Optional[List[str]] = None,
    our_model:     str = "SAINT (ours)",
    title:         str = "",
    ylabel:        str = "",
    save_path:     str = "results/figures/kfold_bar.png",
) -> plt.Figure:
    """Bar chart with mean ± std from cross-validation results."""
    models = model_order if model_order else list(kfold_results.keys())
    means  = [kfold_results[m].get(f"{metric}_mean", float("nan")) for m in models]
    stds   = [kfold_results[m].get(f"{metric}_std",  0.0)          for m in models]
    colors = [HIGHLIGHT_COLOR if m == our_model else BASE_COLOR for m in models]

    fig, ax = plt.subplots(figsize=(7, max(3, len(models) * 0.45)))
    ax.barh(models, means, xerr=stds, color=colors,
            capsize=4, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(ylabel or metric)
    ax.set_title(title or f"{metric} (5-fold CV mean ± std)")
    ax.invert_yaxis()
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved kfold bar chart → {save_path}")
    return fig
