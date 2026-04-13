"""
analysis/case_studies.py — Qualitative patient case studies for F1 + F2.

Selects 3 representative test patients (one FAIL, one PASS, one borderline),
shows the doctor's prescription, the model's suggested prescription,
and the predicted Kt/V change.

Run from project root:
    python analysis/case_studies.py

Prerequisites
-------------
train_f1.py and train_f2.py must have been run first.
(Uses saint_pd_model.pth and report_model2/f2_stageB_best.pth)

Outputs (results/figures/)
--------------------------
case_study_fail.png
case_study_pass.png
case_study_borderline.png
case_study_table.png     — summary table of all 3 cases
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    CAT_RX, CLINICAL_THRESHOLD, CONT_RX, DATA_CSV,
    F1_MODEL_PATH, F1_HIDDEN_SIZE, F1_NUM_HEADS, F1_NUM_LAYERS, F1_DROPOUT, F1_OUTPUT_SIZE,
    FEATURE_INFO_PATH, OUTCOME_COL, PATIENT_ID_COL,
    REPORT_DIR, STEP_MAP,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
)
from src.data.preprocessor import (
    load_feature_info, patient_split, standardize_like_f1,
)
from src.training.f2_core import _f1_predict_raw, infer_prescriptions, make_cfg

FIG_DIR = "results/figures"
F2_CKPT = os.path.join(REPORT_DIR, "f2_stageB_best.pth")


def load_f1(fi, device):
    from src.models.saint import SAINT
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = SAINT(
            input_size=len(fi["original_feature_names"]),
            hidden_size=F1_HIDDEN_SIZE, output_size=F1_OUTPUT_SIZE,
            discrete_feature_indices=fi["discrete_feature_indices"],
            continuous_feature_indices=fi["continuous_feature_indices"],
            num_heads=F1_NUM_HEADS, num_layers=F1_NUM_LAYERS, dropout=F1_DROPOUT,
        )
    model.to(device)
    model.load_state_dict(torch.load(F1_MODEL_PATH, map_location=device,
                                     weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_f2(fi, device):
    from src.models.f2_head import F2RxHead
    K      = None  # will be set after loading data
    in_dim = len(fi["original_feature_names"])
    return in_dim


def _patient_case_figure(patient_row: pd.Series, rx_doctor: dict, rx_model: dict,
                          ktv_actual: float, ktv_doctor_pred: float,
                          ktv_model_pred: float, tag: str, save_path: str):
    """Create a two-panel figure for one patient case study."""
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], figure=fig)

    # ── Left panel: Rx comparison table ──────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0])
    ax_tbl.axis("off")

    rows = []
    for col in CONT_RX:
        d = float(rx_doctor.get(col, float("nan")))
        m = float(rx_model.get(col,  float("nan")))
        delta = m - d
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        rows.append([col, f"{d:.2f}", f"{m:.2f}", f"{delta:+.2f} {arrow}"])

    d_cat = int(rx_doctor.get(CAT_RX, -1))
    m_cat = int(rx_model.get(CAT_RX,  -1))
    rows.append([CAT_RX + " (cat)", str(d_cat), str(m_cat),
                 "same" if d_cat == m_cat else f"{d_cat}→{m_cat}"])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Rx Variable", "Doctor", "Model", "Change"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    # Header style
    for j in range(4):
        tbl[(0, j)].set_facecolor("#2c7bb6")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight changed rows
    for i, row in enumerate(rows, 1):
        if row[3] not in ("same", "= 0.00"):
            for j in range(4):
                tbl[(i, j)].set_facecolor("#ffffcc")

    ax_tbl.set_title(f"Prescription Comparison — {tag.title()} Patient",
                     fontsize=11, fontweight="bold", pad=12)

    # ── Right panel: Kt/V comparison bar ─────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])

    labels = ["Actual\nKt/V", "F1 Pred.\n(Doctor Rx)", "F1 Pred.\n(Model Rx)"]
    vals   = [ktv_actual, ktv_doctor_pred, ktv_model_pred]
    colors = []
    for v in vals:
        colors.append("#d62728" if v < CLINICAL_THRESHOLD else "#2ca02c")

    bars = ax_bar.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.axhline(CLINICAL_THRESHOLD, color="black", linestyle="--",
                   linewidth=1.5, label=f"Adequacy threshold ({CLINICAL_THRESHOLD})")
    ax_bar.set_ylabel("PD Kt/V")
    ax_bar.set_title("PD Kt/V Comparison", fontweight="bold")
    ax_bar.legend(fontsize=8)

    # Value labels on bars
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    delta_str = f"{ktv_model_pred - ktv_doctor_pred:+.3f}"
    ax_bar.set_title(f"PD Kt/V Comparison\n"
                     f"(ΔKt/V model vs doctor: {delta_str})",
                     fontweight="bold")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[case study] Saved → {save_path}")


def plot_summary_table(cases: list, save_path: str):
    """One-figure summary table for all 3 cases."""
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    cols    = ["Patient", "Actual Kt/V", "F1(Doctor Rx)", "F1(Model Rx)",
               "ΔKt/V", "Doctor adequate?", "Model adequate?"]
    rows_d  = []
    for case in cases:
        tag, ktv_act, ktv_doc, ktv_mod = (
            case["tag"], case["ktv_actual"],
            case["ktv_doctor_pred"], case["ktv_model_pred"],
        )
        rows_d.append([
            tag.title(),
            f"{ktv_act:.3f}",
            f"{ktv_doc:.3f}",
            f"{ktv_mod:.3f}",
            f"{ktv_mod - ktv_doc:+.3f}",
            "✓" if ktv_doc >= CLINICAL_THRESHOLD else "✗",
            "✓" if ktv_mod >= CLINICAL_THRESHOLD else "✗",
        ])

    tbl = ax.table(cellText=rows_d, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)

    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#2c7bb6")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Case Study Summary", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[case study] Summary → {save_path}")


def main():
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    from src.utils.device import get_device, print_device_info
    device = get_device()
    print_device_info(device)

    print("Loading data ...")
    df_raw       = pd.read_csv(DATA_CSV)
    feature_info = load_feature_info(FEATURE_INFO_PATH)
    orig_features = feature_info["original_feature_names"]

    df_use = df_raw[list(dict.fromkeys(
        [PATIENT_ID_COL, OUTCOME_COL] + orig_features + CONT_RX + [CAT_RX]
    ))].copy()
    df_use = df_use.loc[:, ~df_use.columns.duplicated()]
    _, _, df_test = patient_split(df_use, PATIENT_ID_COL,
                                  TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    df_test = df_test.reset_index(drop=True)

    f1_model = load_f1(feature_info, device)

    # ── Load F2 model ─────────────────────────────────────────────────────────
    if not os.path.exists(F2_CKPT):
        print(f"F2 checkpoint not found at {F2_CKPT}. Run train_f2.py first.")
        return

    K = int(df_use[CAT_RX].max()) + 1
    from src.models.f2_head import F2RxHead
    model_b = F2RxHead(in_dim=len(orig_features), n_cont=len(CONT_RX), n_cat=K)
    model_b.load_state_dict(torch.load(F2_CKPT, map_location=device, weights_only=True))
    model_b.to(device).eval()

    # ── F1 predictions under doctor's Rx ─────────────────────────────────────
    ktv_doctor_pred = _f1_predict_raw(df_test[orig_features], feature_info,
                                      f1_model, device)
    ktv_actual      = df_test[OUTCOME_COL].values.astype(float)

    # ── F2 inference ─────────────────────────────────────────────────────────
    cfg    = make_cfg()
    rx_df  = infer_prescriptions(model_b, df_test, feature_info,
                                 f1_model, device, cfg)

    # F1 predictions under model Rx
    patched = df_test[orig_features].copy()
    for c in [CAT_RX] + CONT_RX:
        patched[c] = rx_df[c].values
    ktv_model_pred = _f1_predict_raw(patched, feature_info, f1_model, device)

    # ── Select representative patients ────────────────────────────────────────
    fail_idx   = np.where(ktv_actual < CLINICAL_THRESHOLD)[0]
    pass_idx   = np.where(ktv_actual >= CLINICAL_THRESHOLD)[0]
    border_abs = np.abs(ktv_actual - CLINICAL_THRESHOLD)
    border_idx = [np.argmin(border_abs)]

    # For FAIL, prefer the one where model improves the most
    if len(fail_idx) > 0:
        improvements = ktv_model_pred[fail_idx] - ktv_doctor_pred[fail_idx]
        best_fail = fail_idx[np.argmax(improvements)]
    else:
        best_fail = 0

    # For PASS, pick one where model is close (to show preservation)
    if len(pass_idx) > 0:
        deltas = np.abs(ktv_model_pred[pass_idx] - ktv_doctor_pred[pass_idx])
        best_pass = pass_idx[np.argmin(deltas)]
    else:
        best_pass = -1

    case_indices = [
        ("fail",       best_fail),
        ("pass",       best_pass if best_pass >= 0 else border_idx[0]),
        ("borderline", border_idx[0]),
    ]

    cases_summary = []

    for tag, idx in case_indices:
        if idx < 0 or idx >= len(df_test):
            continue
        rx_doc   = {col: float(df_test[col].iloc[idx]) for col in CONT_RX}
        rx_doc[CAT_RX] = int(df_test[CAT_RX].iloc[idx])
        rx_mod   = {col: float(rx_df[col].iloc[idx]) for col in CONT_RX}
        rx_mod[CAT_RX] = int(rx_df[CAT_RX].iloc[idx])

        _patient_case_figure(
            patient_row=df_test.iloc[idx],
            rx_doctor=rx_doc, rx_model=rx_mod,
            ktv_actual=float(ktv_actual[idx]),
            ktv_doctor_pred=float(ktv_doctor_pred[idx]),
            ktv_model_pred=float(ktv_model_pred[idx]),
            tag=tag,
            save_path=f"{FIG_DIR}/case_study_{tag}.png",
        )
        cases_summary.append({
            "tag":            tag,
            "ktv_actual":     float(ktv_actual[idx]),
            "ktv_doctor_pred": float(ktv_doctor_pred[idx]),
            "ktv_model_pred": float(ktv_model_pred[idx]),
        })

    if cases_summary:
        plot_summary_table(cases_summary, f"{FIG_DIR}/case_study_table.png")

    print("\nCase studies complete.")


if __name__ == "__main__":
    main()
