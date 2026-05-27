"""
train_f3.py — F3: first-visit prescription model (no previous prescription).

Key differences from F2
-----------------------
1. No lag features — every row treated as if it were a first visit.
2. Trust-region centred on Stage A, not doctor's current Rx.

Stage A modes  (--mode)
-----------------------
flat (default, recommended)
    One CatBoost per Rx variable.  Same as F2 Stage A but without lag
    features.  This is the strongest practical option because CAT_RX
    (PD system type) cannot be reliably predicted from clinical features
    alone (majority-class baseline ≈ model accuracy in stratified/chain).

stratified
    Step 1: classify PD system.  Step 2: separate CatBoost per class.
    Useful as an upper-bound probe — run with --oracle to see what
    Pearson you'd get if the classifier were perfect.

chain
    Predicted PD system appended as extra feature.  Propagates classifier
    noise into all regressors; generally worse than flat.

Usage
-----
    python train_f3.py                        # flat mode (default)
    python train_f3.py --mode stratified      # per-class Stage A
    python train_f3.py --mode stratified --oracle   # + oracle evaluation

Requirements
------------
    pip install catboost
    feature_info.pkl and saint_pd_model.pth must exist (run train_f1.py first).

Outputs  →  report_model3/
    f3_stageB_best.pth
    val_model_rx.csv / test_model_rx.csv
    val_scatter_*.png / test_scatter_*.png
"""

import argparse
import contextlib
import io
import os
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import (
    CAT_RX, CLINICAL_THRESHOLD, CONT_RX, DATA_CSV, DELTA_WIN,
    DISCRETE_COLUMNS, F1_MODEL_PATH, F2_BATCH_SIZE, F2_DROPOUT, F2_HIDDEN,
    FEATURE_INFO_PATH, OUTCOME_COL, PATIENT_ID_COL, SEED, STEP_MAP,
)
from src.data.preprocessor import (
    RxDataset, load_feature_info, patient_split, remove_outlier_patients,
    sanity_check_data, standardize_like_f1, validate_columns,
)
from src.models.f2_head import F2RxHead
from src.models.stage_a_hierarchical import HierarchicalStageA
from src.training.f2_core import (
    get_stage_A_preds, infer_prescriptions, make_cfg,
    refit_night_models_on_apd, train_stage_A, train_stage_B,
)
from src.utils.device import (
    get_dataloader_kwargs, get_device, is_amp_supported, print_device_info,
)
from src.utils.metrics import pearson_correlation
from torch.utils.data import DataLoader

REPORT_DIR = os.environ.get("REPORT_DIR_F3", "report_model3")


# =============================================================================
# Helpers
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_f1_model(model_path: str, feature_info: dict, device: torch.device) -> nn.Module:
    from src.models.saint import SAINT
    sd = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    d_model = None
    for k, v in sd.items():
        if "discrete_embedding.0.weight" in k or "continuous_embedding.0.weight" in k:
            d_model = v.shape[0]; break
    if d_model is None:
        for k, v in sd.items():
            if "self_attn.in_proj_weight" in k:
                d_model = v.shape[1]; break
    if d_model is None:
        raise RuntimeError("Cannot infer F1 hidden_size from checkpoint.")
    with contextlib.redirect_stdout(io.StringIO()):
        model = SAINT(
            input_size=len(feature_info["original_feature_names"]),
            hidden_size=d_model, output_size=1,
            discrete_feature_indices=feature_info["discrete_feature_indices"],
            continuous_feature_indices=feature_info["continuous_feature_indices"],
        )
    model.to(device).eval()
    model.load_state_dict(sd, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def f1_predict_raw(df_raw: pd.DataFrame, feature_info: dict,
                   f1_model: nn.Module, device: torch.device) -> np.ndarray:
    Xdf = standardize_like_f1(df_raw, feature_info)
    X   = torch.tensor(Xdf.values, dtype=torch.float32, device=device)
    return f1_model(X).squeeze(-1).cpu().numpy()


def get_patient_feature_cols(feature_info: dict) -> list:
    rx_cols = set([CAT_RX] + CONT_RX)
    return [c for c in feature_info["original_feature_names"] if c not in rx_cols]


STAGE_A_TYPES = [
    "rf", "catboost", "xgboost", "lgbm", "linear", "mlp_nn", "transformer_nn",
    "knn", "lasso", "multitask_lasso", "pca_linear", "gp", "cluster",
    "elasticnet", "svr",
]

# Nighttime prescription variables — only meaningful for APD patients.
# CAPD patients have no overnight machine, so these are always 0 for them.
NIGHT_RX = ["night time PD", "Fluid change times", "glucose_total_n", "calcium_total_n"]


def print_stage_a_similarity(df_val, teacher_va: dict, label: str = "Stage A") -> dict:
    """Print Pearson / RMSE between Stage A predictions and doctor ground truth.

    For nighttime prescription variables (NIGHT_RX), Pearson is computed only on
    APD patients (those with night time PD > 0), because CAPD patients always have
    zero nighttime prescriptions — including them inflates the denominator and
    makes the metric misleading.

    Returns a dict: {col: pearson, ..., CAT_RX: accuracy}
    """
    print(f"\n[A] {label} val-set similarity:")
    # APD mask: patients who truly use a nighttime machine
    apd_mask = df_val["night time PD"].values > 0
    n_apd    = int(apd_mask.sum())
    result   = {}
    for j, col in enumerate(CONT_RX):
        gt   = df_val[col].values.astype(float)
        pred = teacher_va["cont"][:, j].astype(float)
        if col in NIGHT_RX and n_apd >= 5:
            gt_r, pred_r = gt[apd_mask], pred[apd_mask]
            suffix = f"  [APD-only n={n_apd}]"
        else:
            gt_r, pred_r = gt, pred
            suffix = ""
        r    = pearson_correlation(gt_r, pred_r)
        rmse = float(np.sqrt(np.mean((gt_r - pred_r) ** 2)))
        print(f"  {col}: Pearson={r:.4f}  RMSE={rmse:.4f}{suffix}")
        result[col] = r
    acc = float((df_val[CAT_RX].astype(int).values == teacher_va["cat"]).mean())
    print(f"  {CAT_RX}: Accuracy={acc:.4f}")
    result[CAT_RX] = acc
    # Convenience summaries stored in result dict
    night_set  = set(NIGHT_RX)
    day_rs     = [result[c] for c in CONT_RX if c not in night_set and not np.isnan(result.get(c, float("nan")))]
    night_rs   = [result[c] for c in CONT_RX if c in     night_set and not np.isnan(result.get(c, float("nan")))]
    result["_day_mean"]   = float(np.mean(day_rs))   if day_rs   else float("nan")
    result["_night_mean"] = float(np.mean(night_rs)) if night_rs else float("nan")
    return result


def oracle_similarity(df_val, stage_a: HierarchicalStageA,
                      patient_cols: list) -> None:
    """
    Evaluate Stage A CONT_RX predictions assuming the true PD system class is
    always known at inference.  This is the upper bound of the stratified/chain
    approaches — it shows what Pearson you'd get with a perfect classifier.
    """
    cont_preds = np.zeros((len(df_val), len(CONT_RX)), dtype=np.float32)
    for cls in stage_a.pd_classes:
        mask   = df_val[CAT_RX].astype(int) == cls
        if not mask.any():
            continue
        models = stage_a.cont_models.get(cls) or stage_a._fallback
        for j, col in enumerate(CONT_RX):
            raw = models[col].predict(df_val.loc[mask, patient_cols]).astype(np.float32)
            raw = np.maximum(0.0, raw)
            cont_preds[mask, j] = np.round(raw / STEP_MAP[col]) * STEP_MAP[col]

    print("\n[A] Stage A oracle similarity (true PD-system class assumed known):")
    for j, col in enumerate(CONT_RX):
        gt   = df_val[col].values.astype(float)
        pred = cont_preds[:, j].astype(float)
        r    = pearson_correlation(gt, pred)
        rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
        print(f"  {col}: Pearson={r:.4f}  RMSE={rmse:.4f}")
    print("  (If oracle >> predicted, improve the classifier; "
          "if similar, the regression signal is the bottleneck.)")


def evaluate_and_save(
    df_eval: pd.DataFrame, rx_df: pd.DataFrame, prefix: str,
    feature_info: dict, f1_model: nn.Module, device: torch.device,
) -> None:
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\n=== Evaluation: {prefix} ===")
    doc_out   = df_eval[OUTCOME_COL].values.astype(float)
    mask_pass = doc_out >= CLINICAL_THRESHOLD
    mask_fail = ~mask_pass

    for col in CONT_RX:
        gt, pr = df_eval[col].values.astype(float), rx_df[col].values.astype(float)
        r_all  = pearson_correlation(gt, pr)
        r_pass = pearson_correlation(gt[mask_pass], pr[mask_pass]) if mask_pass.any() else float("nan")
        r_fail = pearson_correlation(gt[mask_fail], pr[mask_fail]) if mask_fail.any() else float("nan")
        rmse   = float(np.sqrt(np.mean((gt - pr) ** 2)))
        print(f"  {col}: Pearson(all/PASS/FAIL)={r_all:.4f}/{r_pass:.4f}/{r_fail:.4f}  RMSE={rmse:.4f}")

        plt.figure(figsize=(5, 5))
        plt.scatter(gt, pr, s=8, alpha=0.4)
        plt.xlabel("Doctor"); plt.ylabel("Model")
        plt.title(f"{prefix}: {col}\nPearson={r_all:.3f}  RMSE={rmse:.3f}")
        plt.grid(True, alpha=0.2); plt.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", col).strip("_")
        plt.savefig(os.path.join(REPORT_DIR, f"{prefix}_scatter_{safe}.png"), dpi=160)
        plt.close()

    gt_cat = df_eval[CAT_RX].values.astype(int)
    pr_cat = rx_df[CAT_RX].values.astype(int)
    acc    = float((gt_cat == pr_cat).mean())
    print(f"  {CAT_RX}: Accuracy={acc:.4f}")

    orig    = feature_info["original_feature_names"]
    patched = df_eval[orig].copy()
    for c in [CAT_RX] + CONT_RX:
        patched[c] = rx_df[c].values
    ktv_model = f1_predict_raw(patched, feature_info, f1_model, device)

    n_fail = int(mask_fail.sum())
    if n_fail > 0:
        p_rescue = float((ktv_model[mask_fail] >= CLINICAL_THRESHOLD).mean())
        p_win    = float((ktv_model[mask_fail] > doc_out[mask_fail] + DELTA_WIN).mean())
        print(f"  [Doctor-FAIL  N={n_fail}]  p_rescue={p_rescue:.4f}  p_win={p_win:.4f}")
    n_pass = int(mask_pass.sum())
    if n_pass > 0:
        p_fail = float((ktv_model[mask_pass] < CLINICAL_THRESHOLD).mean())
        print(f"  [Doctor-PASS  N={n_pass}]  p_fail={p_fail:.4f}")


# =============================================================================
# Patient-level aggregation
# =============================================================================

def _aggregate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse visit-level rows to one row per patient.

    Numeric columns → mean across visits.
    Discrete/categorical columns → mode (most common value) across visits.
    Returns one row per PatientID, indexed 0..N-1.
    """
    discrete_in_df = [c for c in DISCRETE_COLUMNS if c in df.columns]
    numeric_cols   = [c for c in df.columns
                      if c not in discrete_in_df and c != PATIENT_ID_COL]

    agg: dict = {c: "mean" for c in numeric_cols}
    for c in discrete_in_df:
        agg[c] = lambda x: x.mode().iloc[0]

    return df.groupby(PATIENT_ID_COL, as_index=False).agg(agg).reset_index(drop=True)


def _augment_patients(df_raw: pd.DataFrame, n_augment: int = 4,
                      sample_frac: float = 0.7, seed: int = SEED) -> pd.DataFrame:
    """Create synthetic training patients by subsampling each patient's visits.

    For each real patient, randomly sample `sample_frac` of their visits
    `n_augment` times, aggregating each subsample → one virtual patient row.
    Patients with fewer than 3 visits are skipped (not enough variation).

    Only call on the training split — never on val/test.
    """
    rng  = np.random.RandomState(seed)
    discrete_in_df = [c for c in DISCRETE_COLUMNS if c in df_raw.columns]
    rows = []

    for pid, grp in df_raw.groupby(PATIENT_ID_COL):
        n = len(grp)
        if n < 3:
            continue
        k = max(1, int(n * sample_frac))
        for aug_i in range(n_augment):
            sampled = grp.sample(n=k, replace=False,
                                 random_state=int(rng.randint(0, 2**31)))
            row = {PATIENT_ID_COL: f"aug_{pid}_{aug_i}"}
            for col in df_raw.columns:
                if col == PATIENT_ID_COL:
                    continue
                elif col in discrete_in_df:
                    vals = sampled[col].dropna()
                    row[col] = vals.mode().iloc[0] if len(vals) > 0 else np.nan
                else:
                    row[col] = sampled[col].mean()
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=df_raw.columns)
    return pd.DataFrame(rows)


# =============================================================================
# Distribution plot
# =============================================================================

def plot_distributions() -> None:
    """Plot prescription variable distributions and save to report dir.

    Daytime vars  — all patients (outliers excluded).
    Nighttime vars — APD patients only (night time PD > 0, outliers excluded).
    Outlier removal: patients whose mean value for that variable is >3σ from
    the population mean are excluded from that subplot.
    """
    print(f"[plot_dist] Loading {DATA_CSV} ...")
    df_raw = pd.read_csv(DATA_CSV)
    df_raw = remove_outlier_patients(df_raw)

    # Patient-level aggregation so each patient contributes one point
    cols_needed = [PATIENT_ID_COL] + [c for c in CONT_RX if c in df_raw.columns]
    df_agg = _aggregate_patients(df_raw[cols_needed + [
        c for c in DISCRETE_COLUMNS if c in df_raw.columns]])

    night_set       = set(NIGHT_RX)
    day_cols_plot   = [c for c in CONT_RX if c not in night_set]
    night_cols_plot = [c for c in CONT_RX if c in     night_set]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "Prescription variable distributions  "
        "(patient-level aggregated, outliers excluded)\n"
        "Nighttime vars: APD patients only (night time PD > 0)",
        fontsize=11,
    )

    for row_i, group in enumerate([day_cols_plot, night_cols_plot]):
        for col_i, col in enumerate(group):
            ax = axes[row_i][col_i]

            vals = df_agg[col].dropna()

            if col in night_set:
                vals = vals[vals > 0]          # APD-only for nighttime cols
                note = "  [APD-only]"
            else:
                note = ""

            # Remove per-column outliers (>3σ) for cleaner visualization
            if len(vals) > 3:
                mu, sd = float(vals.mean()), float(vals.std())
                vals = vals[(vals >= mu - 3 * sd) & (vals <= mu + 3 * sd)]

            n   = len(vals)
            mu  = float(vals.mean())  if n > 0 else float("nan")
            med = float(vals.median()) if n > 0 else float("nan")
            sd  = float(vals.std())   if n > 1 else float("nan")

            if n > 0:
                ax.hist(vals, bins=min(25, max(5, n // 3)),
                        color="steelblue", alpha=0.75, edgecolor="white")
                ax.axvline(mu,  color="red",    lw=1.5, ls="--", label=f"mean={mu:.2f}")
                ax.axvline(med, color="orange", lw=1.5, ls=":",  label=f"med={med:.2f}")

            ax.set_title(f"{col[:24]}{note}\nn={n}  σ={sd:.2f}", fontsize=8)
            ax.set_xlabel("value", fontsize=8)
            ax.set_ylabel("count", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    out = os.path.join(REPORT_DIR, "distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_dist] Saved → {out}")
    print(f"  Daytime  columns: {day_cols_plot}")
    print(f"  Nighttime columns (APD-only): {night_cols_plot}")


# =============================================================================
# Two-stage prediction helper
# =============================================================================

def _predict_two_stage(df, day_models: dict, day_cols: list,
                        night_models: dict, night_cols: list,
                        feature_info: dict) -> dict:
    """Combine day model (all patients) and night model (APD-only trained).

    day_models  → CAT_RX + 4 daytime vars, trained on ALL patients
    night_models → 4 nighttime vars, trained on APD patients only
    Returns {"cont": (N, 8), "cat": (N,)}
    """
    day_preds   = get_stage_A_preds(df, day_models,   feature_info,
                                    patient_cols=day_cols)
    night_preds = get_stage_A_preds(df, night_models, feature_info,
                                    patient_cols=night_cols)
    cont = day_preds["cont"].copy()
    night_set = set(NIGHT_RX)
    for j, col in enumerate(CONT_RX):
        if col in night_set:
            cont[:, j] = night_preds["cont"][:, j]
    return {"cont": cont, "cat": day_preds["cat"]}


# =============================================================================
# Top-k sweep
# =============================================================================

def sweep_top_k(args: argparse.Namespace) -> None:
    """Train with each k in K_VALUES and plot val Pearson vs k per prescription var.

    Uses --stageA model type (default: lasso).
    Daytime vars evaluated on all patients; nighttime vars on APD-only.
    """
    import copy

    K_VALUES = [5, 10, 15, 20, 25, 30, 40, 50, 75]

    # Load data without feature selection so we have the full feature set
    base_args        = copy.copy(args)
    base_args.top_k  = None
    ctx              = _prepare_data(base_args)
    tr               = ctx["tr"]
    va               = ctx["va"]
    feature_info     = ctx["feature_info"]
    all_patient_cols = ctx["patient_cols"]

    model_type = args.stageA or "lasso"
    n_all      = len(all_patient_cols)
    k_vals     = sorted({k for k in K_VALUES if k < n_all} | {n_all})
    k_labels   = [str(k) if k < n_all else f"all\n({n_all})" for k in k_vals]

    apd_mask_va = va["night time PD"].values > 0
    apd_mask_tr = tr["night time PD"].values > 0

    print(f"\n[sweep_top_k] model={model_type}  "
          f"k_values={[str(k) if k<n_all else 'all' for k in k_vals]}")

    val_r: dict = {col: [] for col in CONT_RX}
    tr_r:  dict = {col: [] for col in CONT_RX}

    for k in k_vals:
        label = k if k < n_all else "all"
        print(f"  k={label} ...", end="  ", flush=True)
        pcols = _select_features(tr, all_patient_cols, k) if k < n_all else all_patient_cols

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sa = train_stage_A(tr, va, feature_info, model_type=model_type,
                                   patient_cols=pcols, seed=SEED)
            pred_va = get_stage_A_preds(va, sa, feature_info, patient_cols=pcols)
            pred_tr = get_stage_A_preds(tr, sa, feature_info, patient_cols=pcols)

            for j, col in enumerate(CONT_RX):
                gv = va[col].values.astype(float)
                pv = pred_va["cont"][:, j].astype(float)
                gt = tr[col].values.astype(float)
                pt = pred_tr["cont"][:, j].astype(float)
                if col in NIGHT_RX:
                    gv, pv = gv[apd_mask_va], pv[apd_mask_va]
                    gt, pt = gt[apd_mask_tr], pt[apd_mask_tr]
                val_r[col].append(pearson_correlation(gv, pv))
                tr_r[col].append(pearson_correlation(gt, pt))
            print("done")
        except Exception as exc:
            print(f"ERROR: {exc}")
            for col in CONT_RX:
                val_r[col].append(float("nan"))
                tr_r[col].append(float("nan"))

    # ── Plot ──────────────────────────────────────────────────────────────────
    day_cols_plot   = [c for c in CONT_RX if c not in set(NIGHT_RX)]
    night_cols_plot = [c for c in CONT_RX if c in     set(NIGHT_RX)]
    x = list(range(len(k_vals)))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    fig.suptitle(f"Val Pearson vs top_k features  (model = {model_type})", fontsize=13)

    for row_i, group in enumerate([day_cols_plot, night_cols_plot]):
        for col_i, col in enumerate(group):
            ax    = axes[row_i][col_i]
            note  = "  [APD-only]" if col in set(NIGHT_RX) else ""
            v_arr = val_r[col]
            t_arr = tr_r[col]

            ax.plot(x, v_arr, "o-",  color="steelblue", lw=2,   label="val")
            ax.plot(x, t_arr, "s--", color="coral",     lw=1.5, alpha=0.7, label="train")
            ax.axhline(0, color="gray", lw=0.8, ls=":")

            # Mark best val k
            valid = [(i, v) for i, v in enumerate(v_arr) if not np.isnan(v)]
            if valid:
                best_i, best_v = max(valid, key=lambda t: t[1])
                ax.axvline(best_i, color="steelblue", lw=0.8, ls="--", alpha=0.5)
                ax.annotate(f"k={k_vals[best_i] if k_vals[best_i]<n_all else 'all'}\n{best_v:.3f}",
                            xy=(best_i, best_v), xytext=(5, -15),
                            textcoords="offset points", fontsize=7, color="steelblue")

            ax.set_xticks(x)
            ax.set_xticklabels(k_labels, fontsize=7)
            ax.set_xlabel("top_k", fontsize=8)
            ax.set_ylabel("Pearson", fontsize=8)
            ax.set_title(f"{col[:22]}{note}", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    out = os.path.join(REPORT_DIR, f"top_k_sweep_{model_type}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[sweep_top_k] Saved → {out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    short = [c[:8] for c in CONT_RX]
    print("\n" + f"{'k':>6s}  " + "  ".join(f"{s:>9s}" for s in short))
    for i, k in enumerate(k_vals):
        lbl = str(k) if k < n_all else "all"
        row = f"{lbl:>6s}  " + "  ".join(f"{val_r[c][i]:>9.4f}" for c in CONT_RX)
        print(row)


# =============================================================================
# Main
# =============================================================================

def _select_features(df_tr: pd.DataFrame, patient_cols: list, top_k: int) -> list:
    """Select top_k features via MultiTaskLasso importance on training data only.

    Importance = sum of |coef| across all CONT_RX targets.
    Uses the same model that already performs best, so the selection is principled.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import MultiTaskLassoCV

    print(f"\n[FeatureSelect] Selecting top {top_k} / {len(patient_cols)} features "
          f"via MultiTaskLasso ...")
    X = df_tr[patient_cols].values.astype(np.float64)
    Y = df_tr[CONT_RX].values.astype(np.float64)

    sc  = StandardScaler().fit(X)
    mtl = MultiTaskLassoCV(cv=5, max_iter=10000)
    mtl.fit(sc.transform(X), Y)

    # coef_ shape: (n_targets, n_features); sum |coef| across targets → importance per feature
    importance = np.abs(mtl.coef_).sum(axis=0)
    top_k      = min(top_k, len(patient_cols))
    top_idx    = np.argsort(importance)[::-1][:top_k]
    top_idx    = sorted(top_idx.tolist())          # keep original column order
    selected   = [patient_cols[i] for i in top_idx]

    print(f"  alpha={mtl.alpha_:.4f}  selected {len(selected)} features:")
    for rank, i in enumerate(np.argsort(importance)[::-1][:top_k], 1):
        print(f"    {rank:2d}. {patient_cols[i]:<45s}  importance={importance[i]:.4f}")
    return selected


def _prepare_data(args: argparse.Namespace):
    """Load data, F1 oracle, and split — shared across all Stage A runs."""
    set_seed(SEED)
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    device    = get_device()
    print_device_info(device)
    use_amp   = is_amp_supported(device)
    dl_kwargs = get_dataloader_kwargs(device)

    print(f"[F3] Loading data: {DATA_CSV}")
    df_raw = pd.read_csv(DATA_CSV)
    print(f"[F3] Shape: {df_raw.shape}  mode={args.mode}")

    feature_info  = load_feature_info(FEATURE_INFO_PATH)
    orig_features = feature_info["original_feature_names"]

    validate_columns(df_raw, orig_features + [OUTCOME_COL, PATIENT_ID_COL], "train_f3")
    sanity_check_data(df_raw)

    print(f"[F3] Loading F1 model: {F1_MODEL_PATH}")
    f1_model = load_f1_model(F1_MODEL_PATH, feature_info, device)

    def f1_fn(df_part: pd.DataFrame) -> np.ndarray:
        return f1_predict_raw(df_part, feature_info, f1_model, device)

    cols_needed = [PATIENT_ID_COL, OUTCOME_COL] + orig_features
    df_use = df_raw[cols_needed].copy()
    df_use = df_use.loc[:, ~df_use.columns.duplicated()]

    print("\n[F3] Removing outlier patients (>3σ in any prescription variable) ...")
    df_use = remove_outlier_patients(df_use)

    # F3 uses 75/15/10 instead of the default 75/10/15:
    # val needs more patients (32 → ~49) for stable Pearson estimates.
    tr, va, te = patient_split(df_use, PATIENT_ID_COL,
                               train_frac=0.75, val_frac=0.15, test_frac=0.10)
    patient_cols = get_patient_feature_cols(feature_info)

    # Patient-level aggregation: collapse all visits per patient into one row
    # (mean of numeric columns, mode of discrete columns).
    # Prevents models from memorising "patient X → prescription Y" across
    # their repeated visits, forcing generalisation to unseen patients.
    if not args.no_patient_level:
        max_v = getattr(args, "max_visits", None)
        if max_v and max_v > 0:
            rng = np.random.RandomState(SEED)
            def _subsample(df, k):
                parts = []
                for _, grp in df.groupby(PATIENT_ID_COL):
                    parts.append(grp.sample(n=min(k, len(grp)),
                                            random_state=int(rng.randint(0, 2**31))))
                return pd.concat(parts, ignore_index=True)
            tr_sub = _subsample(tr, max_v)
            n_before = tr[PATIENT_ID_COL].nunique()
            print(f"[F3] max_visits={max_v}: {len(tr_sub)} train rows "
                  f"({n_before} patients, avg {len(tr_sub)/n_before:.1f} visits/patient)")
            tr = tr_sub
        n_aug = getattr(args, "augment", 0)
        if n_aug > 0:
            tr_aug = _augment_patients(tr, n_augment=n_aug, seed=SEED)
            tr     = pd.concat([tr, tr_aug], ignore_index=True)
            print(f"[F3] Augmented training: +{len(tr_aug)} virtual visit-rows")
        tr = _aggregate_patients(tr)
        va = _aggregate_patients(va)
        te = _aggregate_patients(te)
        print(f"[F3] Patient-level: train={len(tr)} val={len(va)} test={len(te)} rows")

    if getattr(args, "top_k", None):
        patient_cols = _select_features(tr, patient_cols, args.top_k)

    # Two-stage: separate feature selection on APD patients for night vars
    night_cols_override = None
    if getattr(args, "two_stage", False):
        apd_mask = tr["night time PD"].values > 0
        tr_apd   = tr[apd_mask]
        n_apd_tr = int(apd_mask.sum())
        print(f"\n[F3] Two-stage: {n_apd_tr} APD training patients for night vars")
        if n_apd_tr >= 10:
            k = getattr(args, "top_k", None) or len(patient_cols)
            # Re-run feature selection on APD subset so chosen features are
            # relevant to nighttime prescription signal, not the full population
            all_pcols = get_patient_feature_cols(feature_info)
            night_cols_override = _select_features(tr_apd, all_pcols, k)
            print(f"  Night feature set ({len(night_cols_override)} features, APD-specific)")
        else:
            print("  Too few APD patients — night vars will use same feature set")

    return dict(
        tr=tr, va=va, te=te, df_use=df_use,
        feature_info=feature_info, orig_features=orig_features,
        patient_cols=patient_cols, f1_model=f1_model, f1_fn=f1_fn,
        device=device, use_amp=use_amp, dl_kwargs=dl_kwargs,
        night_cols_override=night_cols_override,
    )


def compare_all_stage_a(args: argparse.Namespace) -> None:
    """Run Stage A for every model type and print a comparison table.

    Stage B is NOT run — this is a fast Stage A benchmark only.
    Use `python train_f3.py --stageA <model>` to run the full pipeline.
    """
    ctx = _prepare_data(args)
    tr, va, feature_info    = ctx["tr"], ctx["va"], ctx["feature_info"]
    patient_cols            = ctx["patient_cols"]
    night_cols_override     = ctx["night_cols_override"]
    use_two_stage           = getattr(args, "two_stage", False) and night_cols_override is not None

    # For two-stage: identify APD subsets once
    if use_two_stage:
        apd_mask_tr = tr["night time PD"].values > 0
        apd_mask_va = va["night time PD"].values > 0
        tr_apd = tr[apd_mask_tr]
        va_apd = va[apd_mask_va] if apd_mask_va.sum() >= 3 else va

    n_runs = args.n_runs
    # all_results[model_type] = list of per-run result dicts
    all_results: dict = {m: [] for m in STAGE_A_TYPES}

    for model_type in STAGE_A_TYPES:
        print(f"\n{'='*60}")
        print(f"[F3] Stage A — {model_type.upper()}  ({n_runs} run(s))"
              + ("  [two-stage]" if use_two_stage else ""))
        for run_i in range(n_runs):
            seed = SEED + run_i
            print(f"  --- run {run_i+1}/{n_runs}  seed={seed} ---")
            try:
                sa_models = train_stage_A(tr, va, feature_info,
                                          model_type=model_type, seed=seed,
                                          patient_cols=patient_cols)
                if use_two_stage:
                    # Train a separate night model on APD patients only
                    night_models = train_stage_A(tr_apd, va_apd, feature_info,
                                                 model_type=model_type, seed=seed,
                                                 patient_cols=night_cols_override)
                    teacher_va = _predict_two_stage(
                        va, sa_models, patient_cols,
                        night_models, night_cols_override, feature_info)
                    teacher_tr = _predict_two_stage(
                        tr, sa_models, patient_cols,
                        night_models, night_cols_override, feature_info)
                else:
                    if getattr(args, "night_filter", False):
                        refit_night_models_on_apd(sa_models, tr, patient_cols)
                    teacher_va = get_stage_A_preds(va, sa_models, feature_info,
                                                   patient_cols=patient_cols)
                    teacher_tr = get_stage_A_preds(tr, sa_models, feature_info,
                                                   patient_cols=patient_cols)
                result     = print_stage_a_similarity(
                    va, teacher_va, label=f"Stage A ({model_type}, seed={seed})")
                tr_result  = print_stage_a_similarity(
                    tr, teacher_tr, label=f"  TRAIN ({model_type}, seed={seed})")
                result["_train"] = {c: tr_result[c] for c in CONT_RX}
                all_results[model_type].append(result)
            except Exception as exc:
                print(f"  [ERROR] run {run_i+1}: {exc}")

    # ------------------------------------------------------------------
    # Comparison table — mean ± std across runs
    # Daytime vars (all patients) and nighttime vars (APD-only) shown separately.
    # ------------------------------------------------------------------
    night_set  = set(NIGHT_RX)
    day_cols   = [c for c in CONT_RX if c not in night_set]
    night_cols = [c for c in CONT_RX if c in     night_set]
    n_apd_val  = int((va["night time PD"].values > 0).sum())

    short   = [col[:8] for col in CONT_RX]
    col_w   = 9
    header  = (f"{'Model':18s}"
               + "".join(f"  {s:>{col_w}s}" for s in short)
               + f"  {'PD_acc':>7s}"
               + f"  {'Day_r':>7s}"
               + f"  {'Ngt_r(APD)':>12s}"
               + f"  {'tr_Day':>7s}")
    div = "=" * len(header)
    print(f"\n{div}")
    title = "STAGE A COMPARISON — val-set Pearson"
    if n_runs > 1:
        title += f"  mean±std over {n_runs} seeds"
    title += f"  |  nighttime cols evaluated on {n_apd_val} APD val patients"
    print(title)
    print(header)
    print("-" * len(header))

    for model_type in STAGE_A_TYPES:
        runs = all_results[model_type]
        if not runs:
            print(f"{model_type:18s}  ERROR (all runs failed)")
            continue
        row = f"{model_type:18s}"
        col_means: dict = {}
        for col in CONT_RX:
            vals = [r[col] for r in runs if col in r and not np.isnan(r[col])]
            mu   = float(np.mean(vals)) if vals else float("nan")
            sd   = float(np.std(vals))  if vals else float("nan")
            col_means[col] = mu
            cell = f"{mu:.4f}±{sd:.4f}" if n_runs > 1 else f"{mu:.4f}"
            row += f"  {cell:>{col_w}s}"
        # CAT_RX accuracy
        cat_vals = [r[CAT_RX] for r in runs if CAT_RX in r]
        cmu = float(np.mean(cat_vals)) if cat_vals else float("nan")
        row += f"  {cmu:>7.4f}"
        # Day / Night mean
        day_vals   = [col_means[c] for c in day_cols   if not np.isnan(col_means.get(c, float("nan")))]
        night_vals = [col_means[c] for c in night_cols if not np.isnan(col_means.get(c, float("nan")))]
        day_r   = float(np.mean(day_vals))   if day_vals   else float("nan")
        night_r = float(np.mean(night_vals)) if night_vals else float("nan")
        row += f"  {day_r:>7.4f}"
        row += f"  {night_r:>12.4f}"
        # Train day mean
        tr_day_vals = []
        for c in day_cols:
            tv = [r["_train"][c] for r in runs if "_train" in r and c in r["_train"]
                  and not np.isnan(r["_train"][c])]
            if tv:
                tr_day_vals.append(float(np.mean(tv)))
        tr_day_r = float(np.mean(tr_day_vals)) if tr_day_vals else float("nan")
        row += f"  {tr_day_r:>7.4f}"
        print(row)

    print(div)
    print(f"\nDay_r  = mean Pearson of {day_cols} (all val patients)")
    print(f"Ngt_r  = mean Pearson of {night_cols} (APD-only, n={n_apd_val})")
    print("\nRun `python train_f3.py --stageA <best_model>` for the full Stage A + B pipeline.")


def main(args: argparse.Namespace) -> None:
    ctx = _prepare_data(args)
    tr           = ctx["tr"]
    va           = ctx["va"]
    te           = ctx["te"]
    df_use       = ctx["df_use"]
    feature_info = ctx["feature_info"]
    orig_features= ctx["orig_features"]
    patient_cols = ctx["patient_cols"]
    f1_model     = ctx["f1_model"]
    f1_fn        = ctx["f1_fn"]
    device       = ctx["device"]
    use_amp      = ctx["use_amp"]
    dl_kwargs    = ctx["dl_kwargs"]

    # ------------------------------------------------------------------
    # 4. Stage A
    # ------------------------------------------------------------------
    stage_a_hier = None   # kept for oracle evaluation if needed

    if args.mode == "flat":
        print(f"\n[F3] Stage A — flat {args.stageA.upper()} (no lag features)")
        sa_models  = train_stage_A(tr, va, feature_info, model_type=args.stageA,
                                   patient_cols=patient_cols)
        if getattr(args, "night_filter", False):
            refit_night_models_on_apd(sa_models, tr, patient_cols)
        teacher_tr = get_stage_A_preds(tr, sa_models, feature_info,
                                       patient_cols=patient_cols)
        teacher_va = get_stage_A_preds(va, sa_models, feature_info,
                                       patient_cols=patient_cols)
        teacher_te = get_stage_A_preds(te, sa_models, feature_info,
                                       patient_cols=patient_cols)

    else:
        # Hierarchical Stage A (stratified or chain).
        # Note: val_acc for CAT_RX is often near / below the majority baseline,
        # which limits the benefit of the hierarchy.  Use --oracle to see the
        # upper bound assuming a perfect classifier.
        print(f"\n[F3] Stage A — hierarchical ({args.mode})")
        stage_a_hier = HierarchicalStageA(mode=args.mode, save_dir="catboost_models_f3")
        stage_a_hier.fit(tr, va, patient_cols)
        teacher_tr = stage_a_hier.predict(tr, patient_cols)
        teacher_va = stage_a_hier.predict(va, patient_cols)
        teacher_te = stage_a_hier.predict(te, patient_cols)

    # Stage A val-set similarity
    sa_label = f"Stage A (flat/{args.stageA})" if args.mode == "flat" else f"Stage A ({args.mode})"
    print_stage_a_similarity(va, teacher_va, label=sa_label)

    # Oracle evaluation: shows upper bound when true class is always used
    if args.oracle and stage_a_hier is not None:
        oracle_similarity(va, stage_a_hier, patient_cols)

    # ------------------------------------------------------------------
    # 5. Datasets + DataLoaders (no lag_arr)
    # ------------------------------------------------------------------
    ds_tr = RxDataset(tr, feature_info, f1_fn, teacher_preds=teacher_tr)
    ds_va = RxDataset(va, feature_info, f1_fn, teacher_preds=teacher_va)

    dl_tr = DataLoader(ds_tr, batch_size=F2_BATCH_SIZE, shuffle=True,  **dl_kwargs)
    dl_va = DataLoader(ds_va, batch_size=F2_BATCH_SIZE, shuffle=False, **dl_kwargs)

    # ------------------------------------------------------------------
    # 6. Stage B (reuses f2_core — trust region on Stage A teacher)
    # ------------------------------------------------------------------
    K       = int(df_use[CAT_RX].max()) + 1
    model_b = F2RxHead(
        in_dim=len(orig_features), n_cont=len(CONT_RX), n_cat=K,
        hidden=F2_HIDDEN, dropout=F2_DROPOUT,
    ).to(device)

    # F3-specific Stage B config:
    # Stage A is weaker than F2 (no lag features, Pearson ~0.4–0.6 vs ~0.7+),
    # so we widen the trust region and reduce the proximal pull to give Stage B
    # more freedom to find prescriptions that improve Kt/V.
    cfg = make_cfg({
        "EPS_CONT_Z_PASS":  1.5,    # was 0.75 — PASS anchor less certain
        "EPS_CONT_Z_FAIL":  4.5,    # was 3.00 — more exploration for FAIL cases
        "EPS_CAT_SOFT_PASS": 0.25,  # was 0.15
        "EPS_CAT_SOFT_FAIL": 0.50,  # was 0.35
        "LAMBDA_PROX_CONT": 0.01,   # was 0.05 — weaker pull toward noisy anchor
        "LAMBDA_PROX_CAT":  0.005,  # was 0.01
    })
    ckpt_path = os.path.join(REPORT_DIR, "f3_stageB_best.pth")
    print("\n[F3] Stage B training ...")
    train_stage_B(model_b, dl_tr, dl_va, f1_model, feature_info,
                  device, cfg, use_amp=use_amp, ckpt_path=ckpt_path)

    if os.path.exists(ckpt_path):
        model_b.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # ------------------------------------------------------------------
    # 7. Inference — Stage A predictions as trust-region anchor
    # ------------------------------------------------------------------
    rx_val  = infer_prescriptions(model_b, va, feature_info, f1_model, device,
                                  cfg, stage_a_preds=teacher_va)
    rx_test = infer_prescriptions(model_b, te, feature_info, f1_model, device,
                                  cfg, stage_a_preds=teacher_te)

    # ------------------------------------------------------------------
    # 8. Evaluate + save
    # Stage A alone baseline is printed first so you can quantify
    # exactly how much Stage B adds on top of the anchor.
    # ------------------------------------------------------------------
    rx_stage_a_val = pd.DataFrame({
        **{col: teacher_va["cont"][:, j] for j, col in enumerate(CONT_RX)},
        CAT_RX: teacher_va["cat"],
    }, index=va.index)
    print("\n--- Stage A alone (val) ---")
    evaluate_and_save(va, rx_stage_a_val, "stageA_val", feature_info, f1_model, device)

    print("\n--- Full pipeline Stage A + B (val) ---")
    evaluate_and_save(va, rx_val,  "val",  feature_info, f1_model, device)
    evaluate_and_save(te, rx_test, "test", feature_info, f1_model, device)

    rx_val.to_csv(os.path.join(REPORT_DIR,  "val_model_rx.csv"),  index=False)
    rx_test.to_csv(os.path.join(REPORT_DIR, "test_model_rx.csv"), index=False)
    print(f"\n[F3] Done. Outputs saved to: {REPORT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F3 first-visit prescription model")
    parser.add_argument(
        "--mode", choices=["flat", "chain", "stratified"], default="flat",
        help=(
            "'flat' (default): one model per Rx variable — recommended. "
            "'stratified': separate models per PD-system class. "
            "'chain': predicted PD-system appended as extra feature."
        ),
    )
    parser.add_argument(
        "--stageA",
        choices=["catboost", "xgboost", "lgbm", "rf", "linear", "mlp_nn", "transformer_nn",
                 "knn", "lasso", "multitask_lasso", "pca_linear", "gp", "cluster"],
        default=None,
        help=(
            "Stage A model type for flat mode. "
            "Omit to benchmark ALL models (Stage A only, no Stage B). "
            "Specify one to run the full Stage A + B pipeline. "
            "Options: rf, catboost, xgboost, lgbm, linear, mlp_nn, transformer_nn."
        ),
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, metavar="N",
        help=(
            "Number of times to train each Stage A model with different random seeds "
            "(default: 1). Only applies to the comparison mode (no --stageA). "
            "Use 5+ to measure variance across seeds."
        ),
    )
    parser.add_argument(
        "--no_patient_level", action="store_true",
        help="Disable patient-level aggregation (default: on). "
             "When off, models train on visit-level rows and overfit badly.",
    )
    parser.add_argument(
        "--top_k", type=int, default=None, metavar="K",
        help="Select top K features via MultiTaskLasso importance before training "
             "(e.g. --top_k 25). Default: use all features.",
    )
    parser.add_argument(
        "--max_visits", type=int, default=None, metavar="K",
        help="Use at most K randomly-sampled visits per patient before aggregation "
             "(e.g. --max_visits 3). Keeps all 334 patients but makes each patient's "
             "feature vector noisier — closer to a real first-visit scenario. "
             "Default: use all visits (current behaviour).",
    )
    parser.add_argument(
        "--augment", type=int, default=0, metavar="N",
        help="Number of synthetic patients to create per real patient via visit "
             "subsampling (e.g. --augment 4). Each subsample draws 70%% of a "
             "patient's visits and aggregates them into one virtual patient row. "
             "Applied to training only. Default: 0 (off).",
    )
    parser.add_argument(
        "--night_filter", action="store_true",
        help="Re-fit nighttime prescription models using only APD patients after "
             "initial training. Nighttime Pearson is always evaluated on APD-only "
             "patients regardless of this flag.",
    )
    parser.add_argument(
        "--two_stage", action="store_true",
        help="Train a dedicated nighttime model on APD patients only with APD-specific "
             "feature selection. Combines a day model (all patients) with a night model "
             "(APD patients only). Requires --top_k to be set.",
    )
    parser.add_argument(
        "--plot_dist", action="store_true",
        help="Plot distributions of all prescription variables and exit. "
             "Nighttime vars show APD patients only (night time PD > 0). "
             "Outlier patients (>3σ) are excluded. "
             "Saves to report_model3/distributions.png",
    )
    parser.add_argument(
        "--sweep_k", action="store_true",
        help="Sweep top_k values [5,10,15,20,25,30,40,50,75,all] and plot val Pearson "
             "vs k for each prescription variable. Uses --stageA model (default: lasso). "
             "Saves figure to report_model3/top_k_sweep_<model>.png",
    )
    parser.add_argument(
        "--oracle", action="store_true",
        help="(stratified/chain only) Evaluate with true class labels — "
             "shows upper bound of hierarchical Stage A.",
    )
    args = parser.parse_args()
    if args.plot_dist:
        plot_distributions()
    elif args.sweep_k:
        sweep_top_k(args)
    elif args.stageA is None:
        compare_all_stage_a(args)
    else:
        main(args)
