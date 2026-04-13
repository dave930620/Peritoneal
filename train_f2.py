"""
train_f2.py — F2 prescription optimization training entry point.

Two-stage training
------------------
Stage A (CatBoost): Learn to mimic doctor's prescription decisions.
    - One CatBoostRegressor per continuous Rx variable.
    - One CatBoostClassifier for the categorical Rx variable.
    - Input: patient features (all ORIG_FEATURES except the Rx variables themselves).
    - Target: doctor's recorded prescription.
    - Saved to CATBOOST_DIR/*.cbm

Stage B (MLP + F1 oracle): Optimize prescriptions to improve predicted PD Kt/V.
    - F2RxHead (MLP) generates prescriptions in z-space.
    - F1 SAINT is frozen and used as a differentiable surrogate evaluator.
    - Loss = effect_loss + threshold_loss + trust_region_loss + proximal_to_catboost_teacher.
    - CatBoost Stage A predictions serve as the proximal anchor (teacher).

Usage
-----
    python train_f2.py

Requirements
------------
- feature_info.pkl and saint_pd_model.pth must exist (run train_f1.py first).
- catboost must be installed: pip install catboost

Outputs (saved to REPORT_DIR)
-------------------------------
- catboost_models/          : Stage A CatBoost model files
- report_model2/f2_stageB_best.pth   : best Stage B MLP checkpoint
- report_model2/val_model_rx.csv     : generated prescriptions for validation set
- report_model2/test_model_rx.csv    : generated prescriptions for test set
- report_model2/*.png                : scatter plots, confusion matrix, loss curves
"""

import io
import contextlib
import os
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    CAT_RX, CATBOOST_DIR, CATBOOST_DEPTH, CATBOOST_EARLY_STOP,
    CATBOOST_ITERATIONS, CATBOOST_LR, CLINICAL_THRESHOLD, CONT_RX,
    CURR_MID_FRAC, CURR_STRONG_FRAC, DATA_CSV, DELTA_WIN, EPOCHS_B,
    EPS_CAT_SOFT_FAIL, EPS_CAT_SOFT_PASS, EPS_CONT_Z_FAIL, EPS_CONT_Z_PASS,
    F1_MODEL_PATH, F2_BATCH_SIZE, F2_DROPOUT, F2_HIDDEN, FEATURE_INFO_PATH,
    GRAD_CLIP, LAMBDA_CONSTRAINT, LAMBDA_PROX_CAT, LAMBDA_PROX_CONT,
    LAMBDA_THR_FAIL, LAMBDA_THR_PASS, LR_B, OUTCOME_COL, PATIENT_ID_COL,
    REPORT_DIR, SEED, STEP_MAP, TEST_FRAC, THR_GATE, TRAIN_FRAC, VAL_FRAC,
    W_EFFECT_FAIL, W_EFFECT_PASS, W_PROX_FAIL,
)
from src.data.preprocessor import (
    RxDataset, load_feature_info, patient_split,
    sanity_check_data, standardize_like_f1, validate_columns,
)
from src.models.f2_head import F2RxHead
from src.utils.device import get_dataloader_kwargs, get_device, is_amp_supported, print_device_info
from src.utils.metrics import pearson_correlation


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

DEVICE    = get_device()
print_device_info(DEVICE)
USE_AMP   = is_amp_supported(DEVICE)
DL_KWARGS = get_dataloader_kwargs(DEVICE)


# =============================================================================
# Load F1 model (frozen oracle used throughout F2 training)
# =============================================================================

def load_f1_model(model_path: str, feature_info: dict, device: torch.device) -> nn.Module:
    """Load F1 SAINT checkpoint; infer hidden size from weights; return frozen eval model."""
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
        raise RuntimeError("Cannot infer F1 hidden_size from checkpoint weights.")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = SAINT(
            input_size=len(feature_info["original_feature_names"]),
            hidden_size=d_model,
            output_size=1,
            discrete_feature_indices=feature_info["discrete_feature_indices"],
            continuous_feature_indices=feature_info["continuous_feature_indices"],
        )
    model.to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# =============================================================================
# F1 prediction helpers
# =============================================================================

@torch.no_grad()
def f1_predict_raw(
    df_raw: pd.DataFrame,
    feature_info: dict,
    f1_model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Predict Kt/V for raw (unscaled) records using F1."""
    Xdf = standardize_like_f1(df_raw, feature_info)
    X   = torch.tensor(Xdf.values, dtype=torch.float32, device=device)
    return f1_model(X).squeeze(-1).cpu().numpy()


def predict_f1_from_z_batch(
    x_std: torch.Tensor,
    z_cont_pred: torch.Tensor,
    p_cat: torch.Tensor,
    feature_info: dict,
    f1_model: nn.Module,
) -> torch.Tensor:
    """Differentiable F1 call: write F2's Rx into the standardized feature vector."""
    scaler     = feature_info["scaler"]
    cont_names = feature_info["continuous_feature_names"]

    # Clamp in z-space so de-standardized Rx >= 0 (consistent with inference)
    z_min_vals = [-(scaler.mean_[cont_names.index(col)] /
                    max(float(scaler.scale_[cont_names.index(col)]), 1e-8))
                  for col in CONT_RX]
    z_min_t = torch.tensor(z_min_vals, device=z_cont_pred.device,
                           dtype=z_cont_pred.dtype)
    z_cont_pred = torch.clamp(z_cont_pred, min=z_min_t)

    orig_features = feature_info["original_feature_names"]
    Xstd = x_std.clone().float()
    for j, col in enumerate(CONT_RX):
        Xstd[:, orig_features.index(col)] = z_cont_pred[:, j]
    cat_idx = torch.argmax(p_cat, dim=1).float()
    Xstd[:, orig_features.index(CAT_RX)] = cat_idx
    return f1_model(Xstd).squeeze(-1)


# =============================================================================
# Stage A: CatBoost doctor mimic
# =============================================================================

def get_patient_feature_cols(feature_info: dict) -> List[str]:
    """Patient features: ORIG_FEATURES minus all Rx variables.

    CatBoost takes only patient characteristics as input to predict prescriptions,
    mirroring the zeroing of Rx columns in RxDataset.
    """
    rx_cols = set([CAT_RX] + CONT_RX)
    return [c for c in feature_info["original_feature_names"] if c not in rx_cols]


def train_stage_A_catboost(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_info: dict,
    save_dir: str = CATBOOST_DIR,
) -> dict:
    """Train one CatBoost model per Rx variable to mimic doctor prescriptions."""
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        raise ImportError("Run: pip install catboost")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    patient_cols = get_patient_feature_cols(feature_info)
    validate_columns(df_train, patient_cols + CONT_RX + [CAT_RX], "catboost_train")

    X_train = df_train[patient_cols]
    X_val   = df_val[patient_cols]
    models  = {}

    print("\n=== Stage A: CatBoost Doctor Mimic ===")

    for col in CONT_RX:
        print(f"[A] CatBoostRegressor → '{col}' ...", end=" ", flush=True)
        m = CatBoostRegressor(
            iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
            depth=CATBOOST_DEPTH, loss_function="RMSE", random_seed=SEED, verbose=0,
        )
        m.fit(X_train, df_train[col],
              eval_set=(X_val, df_val[col]),
              early_stopping_rounds=CATBOOST_EARLY_STOP)
        models[col] = m
        path = os.path.join(save_dir, f"catboost_{col.replace('/', '_')}.cbm")
        m.save_model(path)
        print(f"best_iter={m.best_iteration_}  saved→{path}")

    print(f"[A] CatBoostClassifier → '{CAT_RX}' ...", end=" ", flush=True)
    m_cat = CatBoostClassifier(
        iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
        depth=CATBOOST_DEPTH, loss_function="MultiClass", random_seed=SEED, verbose=0,
    )
    m_cat.fit(X_train, df_train[CAT_RX].astype(int),
              eval_set=(X_val, df_val[CAT_RX].astype(int)),
              early_stopping_rounds=CATBOOST_EARLY_STOP)
    models[CAT_RX] = m_cat
    path = os.path.join(save_dir, f"catboost_{CAT_RX.replace(' ', '_')}.cbm")
    m_cat.save_model(path)
    print(f"best_iter={m_cat.best_iteration_}  saved→{path}")

    return models


def get_catboost_predictions(
    df: pd.DataFrame,
    catboost_models: dict,
    feature_info: dict,
) -> dict:
    """Generate prescription predictions from all CatBoost Stage A models."""
    patient_cols = get_patient_feature_cols(feature_info)
    X = df[patient_cols]

    cont_preds = np.zeros((len(df), len(CONT_RX)), dtype=np.float32)
    for j, col in enumerate(CONT_RX):
        cont_preds[:, j] = catboost_models[col].predict(X).astype(np.float32)

    cat_preds = catboost_models[CAT_RX].predict(X).astype(np.int64).flatten()
    return {"cont": cont_preds, "cat": cat_preds}


# =============================================================================
# Stage B loss helpers
# =============================================================================

def hinge_penalty(dist: torch.Tensor, eps) -> torch.Tensor:
    if not torch.is_tensor(eps):
        eps = torch.tensor(float(eps), device=dist.device, dtype=dist.dtype)
    return torch.relu(dist - eps).mean()


def doc_distance(
    z_pred: torch.Tensor, z_doc: torch.Tensor,
    p_now: torch.Tensor, y_doc_cat: torch.Tensor,
) -> torch.Tensor:
    d_cont = torch.linalg.vector_norm(z_pred - z_doc, dim=1)
    p_doc  = torch.gather(p_now, 1, y_doc_cat.view(-1, 1)).squeeze(1)
    return d_cont + (1.0 - p_doc)


def move_batch(batch: tuple, device: torch.device) -> tuple:
    xb, yb_cont, yb_cat, yb_out, yb_doc_hat, yb_cont_teacher, yb_cat_teacher = batch
    f = lambda t: t.to(dtype=torch.float32, device=device, non_blocking=True)
    l = lambda t: t.to(dtype=torch.long,    device=device, non_blocking=True)
    return f(xb), f(yb_cont), l(yb_cat), f(yb_out), f(yb_doc_hat), f(yb_cont_teacher), l(yb_cat_teacher)


# =============================================================================
# Stage B: MLP effect optimization
# =============================================================================

def _stage_b_loss(
    z_pred, logit, xb, yb_cont, yb_cat, yb_doc_hat,
    yb_cont_teacher, yb_cat_teacher, f1_model, feature_info, step, total_steps,
) -> torch.Tensor:
    p_now = torch.softmax(logit, dim=1)
    ktv   = predict_f1_from_z_batch(xb, z_pred, p_now, feature_info, f1_model)
    gate  = (yb_doc_hat >= THR_GATE).float()

    # Push Kt/V up (weighted by PASS/FAIL)
    w_eff       = gate * W_EFFECT_PASS + (1.0 - gate) * W_EFFECT_FAIL
    loss_effect = -(w_eff * ktv).mean()

    # Encourage crossing the clinical threshold in FAIL cases
    thr_push  = nn.functional.softplus(THR_GATE - ktv)
    w_thr     = gate * LAMBDA_THR_PASS + (1.0 - gate) * LAMBDA_THR_FAIL
    loss_thr  = (w_thr * thr_push).mean()

    # Trust region: keep generated Rx close to doctor's Rx
    d   = doc_distance(z_pred, yb_cont, p_now, yb_cat)
    eps = gate * (EPS_CONT_Z_PASS + EPS_CAT_SOFT_PASS) + (1.0 - gate) * (EPS_CONT_Z_FAIL + EPS_CAT_SOFT_FAIL)
    loss_constraint = LAMBDA_CONSTRAINT * hinge_penalty(d, eps)

    # Proximal: stay close to CatBoost Stage A teacher
    p_teacher  = torch.softmax(nn.functional.one_hot(yb_cat_teacher, p_now.shape[1]).float(), dim=1)
    cont_prox  = ((z_pred - yb_cont_teacher) ** 2).mean(dim=1)
    kl_prox    = (p_now * (torch.log(p_now + 1e-8) - torch.log(p_teacher + 1e-8))).sum(dim=1)
    w_prox     = gate * 1.0 + (1.0 - gate) * W_PROX_FAIL
    loss_prox  = (w_prox * (LAMBDA_PROX_CONT * cont_prox + LAMBDA_PROX_CAT * kl_prox)).mean()

    frac = step / max(1, total_steps)
    lc   = 1.2 if frac < CURR_STRONG_FRAC else (1.0 if frac < CURR_MID_FRAC else 0.9)

    return loss_effect + loss_thr + lc * loss_constraint + lc * loss_prox


def train_stage_B(
    model: F2RxHead,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    f1_model: nn.Module,
    feature_info: dict,
    device: torch.device,
) -> None:
    model.to(device)
    opt       = optim.AdamW(model.parameters(), lr=LR_B, weight_decay=1e-4)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    total_steps = EPOCHS_B * max(1, len(dl_tr))
    step = 0
    train_losses, val_losses = [], []
    best_val = float("inf")

    print("\n=== Stage B: MLP Effect Optimization ===")

    for epoch in range(1, EPOCHS_B + 1):
        model.train(); tr_sum, n_tr = 0.0, 0
        for batch in dl_tr:
            xb, yb_cont, yb_cat, yb_out, yb_doc_hat, yb_cont_teacher, yb_cat_teacher = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                z_pred, logit = model(xb)
                loss = _stage_b_loss(z_pred, logit, xb, yb_cont, yb_cat, yb_doc_hat,
                                     yb_cont_teacher, yb_cat_teacher, f1_model, feature_info, step, total_steps)
            amp_scaler.scale(loss).backward()
            if GRAD_CLIP:
                amp_scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            amp_scaler.step(opt); amp_scaler.update()
            tr_sum += loss.item() * xb.size(0); n_tr += xb.size(0); step += 1

        model.eval(); va_sum, n_va = 0.0, 0
        with torch.no_grad():
            for batch in dl_va:
                xb, yb_cont, yb_cat, yb_out, yb_doc_hat, yb_cont_teacher, yb_cat_teacher = move_batch(batch, device)
                z_pred, logit = model(xb)
                loss = _stage_b_loss(z_pred, logit, xb, yb_cont, yb_cat, yb_doc_hat,
                                     yb_cont_teacher, yb_cat_teacher, f1_model, feature_info, step, total_steps)
                va_sum += loss.item() * xb.size(0); n_va += xb.size(0)

        tr_l = tr_sum / max(1, n_tr); va_l = va_sum / max(1, n_va)
        train_losses.append(tr_l); val_losses.append(va_l)
        print(f"[B] epoch {epoch:03d}  train={tr_l:.6f}  val={va_l:.6f}")

        if va_l < best_val:
            best_val = va_l
            torch.save(model.state_dict(), os.path.join(REPORT_DIR, "f2_stageB_best.pth"))

    _save_loss_curve(train_losses, val_losses, "stageB")


def _save_loss_curve(train: list, val: list, prefix: str) -> None:
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train)+1), train, label="Train")
    plt.plot(range(1, len(val)+1),   val,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{prefix} loss")
    plt.legend(); plt.tight_layout()
    out = os.path.join(REPORT_DIR, f"{prefix}_loss.png")
    plt.savefig(out, dpi=160); plt.close()
    print(f"[{prefix}] Loss curve → {out}")


# =============================================================================
# Inference with hard constraints (projection + step-size rounding)
# =============================================================================

def infer_prescriptions(
    model: F2RxHead,
    df_part: pd.DataFrame,
    feature_info: dict,
    f1_model: nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    """Generate constrained prescriptions: project into trust region + round to step sizes."""
    orig_features = feature_info["original_feature_names"]
    cont_names    = feature_info["continuous_feature_names"]
    scaler        = feature_info["scaler"]

    model.eval()
    doc_hat   = f1_predict_raw(df_part, feature_info, f1_model, device)
    gate_pass = doc_hat >= THR_GATE

    # Standardize features, zero out Rx columns (F2 generates them)
    Xstd = standardize_like_f1(df_part, feature_info)
    for c in [CAT_RX] + CONT_RX:
        if c in Xstd.columns:
            Xstd[c] = 0.0
    Xmat = Xstd.values.astype(np.float32)

    rows, N = [], Xmat.shape[0]

    for i in range(0, N, F2_BATCH_SIZE):
        xb    = torch.as_tensor(Xmat[i:i+F2_BATCH_SIZE], dtype=torch.float32, device=device)
        chunk = df_part.iloc[i:i+F2_BATCH_SIZE]

        with torch.no_grad():
            z_pred, logit = model(xb)
            p_now   = torch.softmax(logit, dim=1)
            cat_idx = torch.argmax(p_now, dim=1).cpu().numpy().astype(int)

        # Doctor's continuous Rx in z-space (trust region center)
        z_doc = np.zeros((len(chunk), len(CONT_RX)), dtype=np.float32)
        for j, col in enumerate(CONT_RX):
            idx = cont_names.index(col)
            mu  = scaler.mean_[idx]
            sd  = scaler.scale_[idx] if scaler.scale_[idx] > 0 else 1.0
            z_doc[:, j] = (chunk[col].values.astype(np.float32) - mu) / sd

        # Project into trust region
        z_np   = z_pred.cpu().numpy()
        z_proj = np.zeros_like(z_np)
        for r in range(z_np.shape[0]):
            eps = EPS_CONT_Z_PASS if gate_pass[i + r] else EPS_CONT_Z_FAIL
            d   = z_np[r] - z_doc[r]
            n   = np.linalg.norm(d)
            z_proj[r] = z_np[r] if (n == 0 or n <= eps) else z_doc[r] + d * (eps / n)

        # Clamp in z-space: ensures de-standardized raw >= 0 for all Rx variables
        z_min = np.array([
            -(scaler.mean_[cont_names.index(col)] /
              max(float(scaler.scale_[cont_names.index(col)]), 1e-8))
            for col in CONT_RX
        ], dtype=np.float32)
        z_proj = np.maximum(z_proj, z_min)

        # De-standardize + round to clinical step sizes
        for r in range(z_proj.shape[0]):
            row = {CAT_RX: int(cat_idx[r])}
            for j, col in enumerate(CONT_RX):
                idx  = cont_names.index(col)
                mu   = scaler.mean_[idx]
                sd   = scaler.scale_[idx] if scaler.scale_[idx] > 0 else 1.0
                raw  = z_proj[r, j] * sd + mu
                step = STEP_MAP[col]
                row[col] = float(np.round(raw / step) * step)
            rows.append(row)

    return pd.DataFrame(rows, index=df_part.index)


# =============================================================================
# Evaluation
# =============================================================================

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s)).strip("_")


def evaluate_and_save_plots(
    df_eval: pd.DataFrame,
    rx_df: pd.DataFrame,
    prefix: str,
    feature_info: dict,
    f1_model: nn.Module,
    device: torch.device,
) -> None:
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\n=== Evaluation: {prefix} ===")

    doc_out   = df_eval[OUTCOME_COL].values.astype(float)
    mask_pass = doc_out >= THR_GATE
    mask_fail = ~mask_pass

    # Continuous Rx similarity
    for col in CONT_RX:
        gt = df_eval[col].values.astype(float)
        pr = rx_df[col].values.astype(float)

        r_all  = pearson_correlation(gt, pr)
        r_pass = pearson_correlation(gt[mask_pass], pr[mask_pass]) if mask_pass.sum() > 0 else float("nan")
        r_fail = pearson_correlation(gt[mask_fail], pr[mask_fail]) if mask_fail.sum() > 0 else float("nan")

        print(f"  {col}:")
        print(f"    Pearson (all / PASS / FAIL) = {r_all:.4f} / {r_pass:.4f} / {r_fail:.4f}")

        plt.figure(figsize=(5, 5))
        plt.scatter(gt, pr, s=8, alpha=0.4)
        plt.xlabel("Doctor"); plt.ylabel("Model")
        plt.title(f"{prefix}: {col}\nPearson={r_all:.3f}")
        plt.grid(True, alpha=0.2); plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, f"{prefix}_scatter_{_safe_name(col)}.png"), dpi=160)
        plt.close()

    # Categorical accuracy
    gt_cat = df_eval[CAT_RX].values.astype(int)
    pr_cat = rx_df[CAT_RX].values.astype(int)
    acc_all  = float((gt_cat == pr_cat).mean())
    acc_pass = float((gt_cat[mask_pass] == pr_cat[mask_pass]).mean()) if mask_pass.sum() > 0 else float("nan")
    acc_fail = float((gt_cat[mask_fail] == pr_cat[mask_fail]).mean()) if mask_fail.sum() > 0 else float("nan")
    print(f"  {CAT_RX}: Accuracy (all / PASS / FAIL) = {acc_all:.4f} / {acc_pass:.4f} / {acc_fail:.4f}")

    # Threshold analysis using F1 with model's generated prescription
    patched = df_eval[feature_info["original_feature_names"]].copy()
    for c in [CAT_RX] + CONT_RX:
        patched[c] = rx_df[c].values
    ktv_model = f1_predict_raw(patched, feature_info, f1_model, device)

    thr = CLINICAL_THRESHOLD
    print(f"\n  Threshold analysis (thr={thr}):")

    mask_d = doc_out < thr
    Nd = int(mask_d.sum())
    if Nd > 0:
        p_pass = float((ktv_model[mask_d] >= thr).mean())
        p_win  = float((ktv_model[mask_d] > doc_out[mask_d] + DELTA_WIN).mean())
        print(f"  [Doctor FAIL group] N={Nd}  P_pass={p_pass:.4f}  P_win={p_win:.4f}")

    mask_e = doc_out >= thr
    Ne = int(mask_e.sum())
    if Ne > 0:
        p_fail  = float((ktv_model[mask_e] < thr).mean())
        p_win_e = float((ktv_model[mask_e] > doc_out[mask_e] + DELTA_WIN).mean())
        print(f"  [Doctor PASS group] N={Ne}  P_fail={p_fail:.4f}  P_win={p_win_e:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load data + feature_info
    print(f"[F2] Loading data: {DATA_CSV}")
    df_raw = pd.read_csv(DATA_CSV)
    print(f"[F2] Shape: {df_raw.shape}")

    feature_info  = load_feature_info(FEATURE_INFO_PATH)
    orig_features = feature_info["original_feature_names"]

    validate_columns(df_raw, orig_features + [OUTCOME_COL, PATIENT_ID_COL], "train_f2")

    # Safety check: Volume (L) must NOT be in orig_features
    if "Volume (L)" in orig_features:
        raise RuntimeError(
            "Volume (L) is in feature_info — retrain F1 first with the updated "
            "EXCLUDE_COLUMNS (which excludes Volume (L))."
        )

    sanity_check_data(df_raw)

    # 2. Load frozen F1 oracle
    print(f"[F2] Loading F1 model: {F1_MODEL_PATH}")
    f1_model = load_f1_model(F1_MODEL_PATH, feature_info, DEVICE)

    def f1_predict_fn(df_part: pd.DataFrame) -> np.ndarray:
        return f1_predict_raw(df_part, feature_info, f1_model, DEVICE)

    # 3. Patient-wise split
    df_use = df_raw[[PATIENT_ID_COL, OUTCOME_COL] + orig_features + CONT_RX + [CAT_RX]].copy()
    # Drop duplicate columns (CONT_RX and CAT_RX are already in orig_features)
    df_use = df_use.loc[:, ~df_use.columns.duplicated()]
    tr, va, te = patient_split(df_use, PATIENT_ID_COL, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)

    # 4. Stage A: CatBoost
    catboost_models = train_stage_A_catboost(tr, va, feature_info)

    teacher_tr = get_catboost_predictions(tr, catboost_models, feature_info)
    teacher_va = get_catboost_predictions(va, catboost_models, feature_info)
    teacher_te = get_catboost_predictions(te, catboost_models, feature_info)

    # Stage A similarity on val set
    print("\n[A] Stage A val-set similarity:")
    for j, col in enumerate(CONT_RX):
        r = pearson_correlation(va[col].values, teacher_va["cont"][:, j])
        print(f"  {col}: Pearson = {r:.4f}")
    acc_a = float((va[CAT_RX].astype(int).values == teacher_va["cat"]).mean())
    print(f"  {CAT_RX}: Accuracy = {acc_a:.4f}")

    # 5. Build datasets + DataLoaders
    ds_tr = RxDataset(tr, feature_info, f1_predict_fn, teacher_preds=teacher_tr)
    ds_va = RxDataset(va, feature_info, f1_predict_fn, teacher_preds=teacher_va)
    ds_te = RxDataset(te, feature_info, f1_predict_fn, teacher_preds=teacher_te)

    dl_tr = DataLoader(ds_tr, batch_size=F2_BATCH_SIZE, shuffle=True,  **DL_KWARGS)
    dl_va = DataLoader(ds_va, batch_size=F2_BATCH_SIZE, shuffle=False, **DL_KWARGS)

    # 6. Stage B: MLP
    K = int(df_use[CAT_RX].max()) + 1
    model_b = F2RxHead(
        in_dim=len(orig_features), n_cont=len(CONT_RX), n_cat=K,
        hidden=F2_HIDDEN, dropout=F2_DROPOUT,
    ).to(DEVICE)

    train_stage_B(model_b, dl_tr, dl_va, f1_model, feature_info, DEVICE)

    best_ckpt = os.path.join(REPORT_DIR, "f2_stageB_best.pth")
    model_b.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=True))

    # 7. Inference — pass the full split dataframes (no column-selection needed)
    rx_val  = infer_prescriptions(model_b, va,  feature_info, f1_model, DEVICE)
    rx_test = infer_prescriptions(model_b, te,  feature_info, f1_model, DEVICE)

    # 8. Evaluate
    evaluate_and_save_plots(va, rx_val,  "val",  feature_info, f1_model, DEVICE)
    evaluate_and_save_plots(te, rx_test, "test", feature_info, f1_model, DEVICE)

    # 9. Save prescriptions
    rx_val.to_csv(os.path.join(REPORT_DIR,  "val_model_rx.csv"),  index=False)
    rx_test.to_csv(os.path.join(REPORT_DIR, "test_model_rx.csv"), index=False)
    print(f"\n[F2] Done. Prescriptions and plots saved to: {REPORT_DIR}/")


if __name__ == "__main__":
    main()
