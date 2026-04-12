"""
f2_core.py — Configurable F2 pipeline used by baseline and ablation experiments.

The default config mirrors train_f2.py constants exactly.
Each ablation passes an `overrides` dict with only the keys that change.

Override keys
-------------
skip_stage_A      bool   Skip CatBoost Stage A; use doctor Rx as teacher anchor
use_projection    bool   Apply trust-region projection at inference
use_linear_head   bool   Replace F2RxHead MLP with a linear head
uniform_weights   bool   Disable PASS/FAIL gating (same weights for all)
fixed_lc          float  If set, override curriculum factor to this constant
LAMBDA_CONSTRAINT float
LAMBDA_PROX_CONT  float
LAMBDA_PROX_CAT   float
LAMBDA_THR_FAIL   float
LAMBDA_THR_PASS   float
W_EFFECT_PASS     float
W_EFFECT_FAIL     float
W_PROX_FAIL       float
EPS_CONT_Z_PASS   float
EPS_CONT_Z_FAIL   float
EPS_CAT_SOFT_PASS float
EPS_CAT_SOFT_FAIL float
THR_GATE          float
EPOCHS_B          int
LR_B              float
F2_HIDDEN         int
F2_DROPOUT        float
F2_BATCH_SIZE     int
"""

import io
import contextlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    CAT_RX, CATBOOST_DEPTH, CATBOOST_EARLY_STOP, CATBOOST_ITERATIONS,
    CATBOOST_LR, CLINICAL_THRESHOLD, CONT_RX,
    CURR_MID_FRAC, CURR_STRONG_FRAC, DELTA_WIN,
    EPS_CAT_SOFT_FAIL, EPS_CAT_SOFT_PASS, EPS_CONT_Z_FAIL, EPS_CONT_Z_PASS,
    EPOCHS_B, F2_BATCH_SIZE, F2_DROPOUT, F2_HIDDEN,
    GRAD_CLIP, LAMBDA_CONSTRAINT, LAMBDA_PROX_CAT, LAMBDA_PROX_CONT,
    LAMBDA_THR_FAIL, LAMBDA_THR_PASS, LR_B, OUTCOME_COL, PATIENT_ID_COL,
    SEED, STEP_MAP, THR_GATE, W_EFFECT_FAIL, W_EFFECT_PASS, W_PROX_FAIL,
)
from src.data.preprocessor import (
    RxDataset, standardize_like_f1, validate_columns,
)
from src.models.f2_head import F2RxHead
from src.utils.metrics import f2_metrics, pearson_correlation


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

def _default_cfg() -> dict:
    return {
        "skip_stage_A":      False,
        "use_projection":    True,
        "use_linear_head":   False,
        "uniform_weights":   False,
        "fixed_lc":          None,
        "LAMBDA_CONSTRAINT": LAMBDA_CONSTRAINT,
        "LAMBDA_PROX_CONT":  LAMBDA_PROX_CONT,
        "LAMBDA_PROX_CAT":   LAMBDA_PROX_CAT,
        "LAMBDA_THR_FAIL":   LAMBDA_THR_FAIL,
        "LAMBDA_THR_PASS":   LAMBDA_THR_PASS,
        "W_EFFECT_PASS":     W_EFFECT_PASS,
        "W_EFFECT_FAIL":     W_EFFECT_FAIL,
        "W_PROX_FAIL":       W_PROX_FAIL,
        "EPS_CONT_Z_PASS":   EPS_CONT_Z_PASS,
        "EPS_CONT_Z_FAIL":   EPS_CONT_Z_FAIL,
        "EPS_CAT_SOFT_PASS": EPS_CAT_SOFT_PASS,
        "EPS_CAT_SOFT_FAIL": EPS_CAT_SOFT_FAIL,
        "THR_GATE":          THR_GATE,
        "EPOCHS_B":          EPOCHS_B,
        "LR_B":              LR_B,
        "F2_HIDDEN":         F2_HIDDEN,
        "F2_DROPOUT":        F2_DROPOUT,
        "F2_BATCH_SIZE":     F2_BATCH_SIZE,
    }


def make_cfg(overrides: Optional[dict] = None) -> dict:
    cfg = _default_cfg()
    if overrides:
        cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Stage A helpers (same logic as train_f2.py)
# ---------------------------------------------------------------------------

def _patient_feature_cols(feature_info: dict) -> list:
    rx_cols = set([CAT_RX] + CONT_RX)
    return [c for c in feature_info["original_feature_names"] if c not in rx_cols]


def train_stage_A(df_tr, df_va, feature_info: dict, save_dir: str = "") -> dict:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        raise ImportError("pip install catboost")

    patient_cols = _patient_feature_cols(feature_info)
    X_tr, X_va  = df_tr[patient_cols], df_va[patient_cols]
    models: dict = {}

    for col in CONT_RX:
        m = CatBoostRegressor(
            iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
            depth=CATBOOST_DEPTH, loss_function="RMSE", random_seed=SEED, verbose=0,
        )
        m.fit(X_tr, df_tr[col],
              eval_set=(X_va, df_va[col]),
              early_stopping_rounds=CATBOOST_EARLY_STOP)
        models[col] = m
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            m.save_model(os.path.join(save_dir, f"cb_{col.replace('/', '_')}.cbm"))

    m_cat = CatBoostClassifier(
        iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
        depth=CATBOOST_DEPTH, loss_function="MultiClass", random_seed=SEED, verbose=0,
    )
    m_cat.fit(X_tr, df_tr[CAT_RX].astype(int),
              eval_set=(X_va, df_va[CAT_RX].astype(int)),
              early_stopping_rounds=CATBOOST_EARLY_STOP)
    models[CAT_RX] = m_cat
    return models


def get_stage_A_preds(df, catboost_models: dict, feature_info: dict) -> dict:
    patient_cols = _patient_feature_cols(feature_info)
    X = df[patient_cols]
    cont = np.zeros((len(df), len(CONT_RX)), dtype=np.float32)
    for j, col in enumerate(CONT_RX):
        cont[:, j] = catboost_models[col].predict(X).astype(np.float32)
    cat = catboost_models[CAT_RX].predict(X).astype(np.int64).flatten()
    return {"cont": cont, "cat": cat}


# ---------------------------------------------------------------------------
# Doctor prescription preds (teacher = doctor when Stage A skipped)
# ---------------------------------------------------------------------------

def _doctor_teacher_preds(df, feature_info: dict) -> dict:
    cont_names = feature_info["continuous_feature_names"]
    scaler     = feature_info["scaler"]
    cont = np.zeros((len(df), len(CONT_RX)), dtype=np.float32)
    for j, col in enumerate(CONT_RX):
        cont[:, j] = df[col].values.astype(np.float32)
    cat = df[CAT_RX].astype(np.int64).values
    return {"cont": cont, "cat": cat}


# ---------------------------------------------------------------------------
# Stage B loss (parameterised by cfg)
# ---------------------------------------------------------------------------

def _hinge(dist: torch.Tensor, eps) -> torch.Tensor:
    if not torch.is_tensor(eps):
        eps = torch.tensor(float(eps), device=dist.device, dtype=dist.dtype)
    return torch.relu(dist - eps).mean()


def _doc_dist(z_pred, z_doc, p_now, y_doc_cat) -> torch.Tensor:
    d_cont = torch.linalg.vector_norm(z_pred - z_doc, dim=1)
    p_doc  = torch.gather(p_now, 1, y_doc_cat.view(-1, 1)).squeeze(1)
    return d_cont + (1.0 - p_doc)


def _stage_b_loss(z_pred, logit, xb, yb_cont, yb_cat, yb_doc_hat,
                  yb_cont_teacher, yb_cat_teacher,
                  f1_model, feature_info, step, total_steps, cfg) -> torch.Tensor:
    orig_features = feature_info["original_feature_names"]
    p_now = torch.softmax(logit, dim=1)

    # F1 prediction with F2's Rx patched in
    Xstd  = xb.clone().float()
    for j, col in enumerate(CONT_RX):
        Xstd[:, orig_features.index(col)] = z_pred[:, j]
    cat_idx = torch.argmax(p_now, dim=1).float()
    Xstd[:, orig_features.index(CAT_RX)] = cat_idx
    ktv = f1_model(Xstd).squeeze(-1)

    thr  = cfg["THR_GATE"]
    gate = (yb_doc_hat >= thr).float() if not cfg["uniform_weights"] else torch.ones_like(yb_doc_hat)

    w_eff       = gate * cfg["W_EFFECT_PASS"] + (1 - gate) * cfg["W_EFFECT_FAIL"]
    loss_effect = -(w_eff * ktv).mean()

    thr_push  = nn.functional.softplus(thr - ktv)
    w_thr     = gate * cfg["LAMBDA_THR_PASS"] + (1 - gate) * cfg["LAMBDA_THR_FAIL"]
    loss_thr  = (w_thr * thr_push).mean()

    d = _doc_dist(z_pred, yb_cont, p_now, yb_cat)
    eps_c = cfg["EPS_CONT_Z_PASS"] + cfg["EPS_CAT_SOFT_PASS"]
    eps_f = cfg["EPS_CONT_Z_FAIL"] + cfg["EPS_CAT_SOFT_FAIL"]
    eps   = gate * eps_c + (1 - gate) * eps_f
    loss_constraint = cfg["LAMBDA_CONSTRAINT"] * _hinge(d, eps)

    n_cat       = logit.shape[1]
    p_teacher   = torch.softmax(
        nn.functional.one_hot(yb_cat_teacher, n_cat).float(), dim=1)
    cont_prox   = ((z_pred - yb_cont_teacher) ** 2).mean(dim=1)
    kl_prox     = (p_now * (torch.log(p_now + 1e-8) - torch.log(p_teacher + 1e-8))).sum(dim=1)
    w_prox      = gate * 1.0 + (1 - gate) * cfg["W_PROX_FAIL"]
    loss_prox   = (w_prox * (cfg["LAMBDA_PROX_CONT"] * cont_prox +
                             cfg["LAMBDA_PROX_CAT"]  * kl_prox)).mean()

    if cfg["fixed_lc"] is not None:
        lc = cfg["fixed_lc"]
    else:
        frac = step / max(1, total_steps)
        lc   = 1.2 if frac < CURR_STRONG_FRAC else (1.0 if frac < CURR_MID_FRAC else 0.9)

    return loss_effect + loss_thr + lc * loss_constraint + lc * loss_prox


# ---------------------------------------------------------------------------
# Stage B training loop
# ---------------------------------------------------------------------------

def train_stage_B(
    model: nn.Module, dl_tr, dl_va,
    f1_model, feature_info, device,
    cfg: dict, use_amp: bool = False,
    ckpt_path: str = "",
) -> None:
    model.to(device)
    opt        = optim.AdamW(model.parameters(), lr=cfg["LR_B"], weight_decay=1e-4)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total      = cfg["EPOCHS_B"] * max(1, len(dl_tr))
    step       = 0
    best_val   = float("inf")

    def _move(batch):
        xb, yb_cont, yb_cat, yb_out, yb_doc_hat, yb_cont_t, yb_cat_t = batch
        f = lambda t: t.to(dtype=torch.float32, device=device, non_blocking=True)
        l = lambda t: t.to(dtype=torch.long,    device=device, non_blocking=True)
        return f(xb), f(yb_cont), l(yb_cat), f(yb_out), f(yb_doc_hat), f(yb_cont_t), l(yb_cat_t)

    for epoch in range(1, cfg["EPOCHS_B"] + 1):
        model.train(); tr_s, n_tr = 0.0, 0
        for batch in dl_tr:
            xb, yb_cont, yb_cat, yb_out, yb_doc_hat, yb_cont_t, yb_cat_t = _move(batch)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                z_pred, logit = model(xb)
                loss = _stage_b_loss(z_pred, logit, xb, yb_cont, yb_cat,
                                     yb_doc_hat, yb_cont_t, yb_cat_t,
                                     f1_model, feature_info, step, total, cfg)
            amp_scaler.scale(loss).backward()
            if GRAD_CLIP:
                amp_scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            amp_scaler.step(opt); amp_scaler.update()
            tr_s += loss.item() * xb.size(0); n_tr += xb.size(0); step += 1

        model.eval(); va_s, n_va = 0.0, 0
        with torch.no_grad():
            for batch in dl_va:
                xb, yb_cont, yb_cat, yb_out, yb_doc_hat, yb_cont_t, yb_cat_t = _move(batch)
                z_pred, logit = model(xb)
                loss = _stage_b_loss(z_pred, logit, xb, yb_cont, yb_cat,
                                     yb_doc_hat, yb_cont_t, yb_cat_t,
                                     f1_model, feature_info, step, total, cfg)
                va_s += loss.item() * xb.size(0); n_va += xb.size(0)

        va_l = va_s / max(1, n_va)
        print(f"  [B] epoch {epoch:03d}  val={va_l:.6f}")
        if va_l < best_val:
            best_val = va_l
            if ckpt_path:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)


# ---------------------------------------------------------------------------
# Inference with optional trust-region projection
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_prescriptions(
    model: nn.Module, df_part, feature_info: dict,
    f1_model: nn.Module, device: torch.device,
    cfg: dict,
) -> "pd.DataFrame":
    import pandas as pd
    orig_features = feature_info["original_feature_names"]
    cont_names    = feature_info["continuous_feature_names"]
    scaler        = feature_info["scaler"]

    model.eval()
    doc_hat   = _f1_predict_raw(df_part, feature_info, f1_model, device)
    gate_pass = doc_hat >= cfg["THR_GATE"]

    Xstd = standardize_like_f1(df_part, feature_info)
    for c in [CAT_RX] + CONT_RX:
        if c in Xstd.columns:
            Xstd[c] = 0.0
    Xmat = Xstd.values.astype(np.float32)

    rows, N = [], Xmat.shape[0]
    bs = cfg["F2_BATCH_SIZE"]

    for i in range(0, N, bs):
        xb    = torch.as_tensor(Xmat[i:i+bs], dtype=torch.float32, device=device)
        chunk = df_part.iloc[i:i+bs]
        z_pred, logit = model(xb)
        p_now   = torch.softmax(logit, dim=1)
        cat_idx = torch.argmax(p_now, dim=1).cpu().numpy().astype(int)

        z_doc = np.zeros((len(chunk), len(CONT_RX)), dtype=np.float32)
        for j, col in enumerate(CONT_RX):
            idx = cont_names.index(col)
            mu  = scaler.mean_[idx]
            sd  = scaler.scale_[idx] if scaler.scale_[idx] > 0 else 1.0
            z_doc[:, j] = (chunk[col].values.astype(np.float32) - mu) / sd

        z_np   = z_pred.cpu().numpy()
        z_proj = z_np.copy()

        if cfg["use_projection"]:
            for r in range(z_np.shape[0]):
                eps = (cfg["EPS_CONT_Z_PASS"] if gate_pass[i + r]
                       else cfg["EPS_CONT_Z_FAIL"])
                d   = z_np[r] - z_doc[r]
                n   = np.linalg.norm(d)
                if n > 0 and n > eps:
                    z_proj[r] = z_doc[r] + d * (eps / n)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _f1_predict_raw(df_part, feature_info: dict, f1_model, device) -> np.ndarray:
    from src.data.preprocessor import standardize_like_f1
    Xdf = standardize_like_f1(df_part, feature_info)
    X   = torch.tensor(Xdf.values, dtype=torch.float32, device=device)
    return f1_model(X).squeeze(-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Full pipeline (used by run_baselines.py / run_ablations.py)
# ---------------------------------------------------------------------------

def run_f2_pipeline(
    df_train, df_val, df_test,
    feature_info: dict,
    f1_model: nn.Module,
    device: torch.device,
    overrides: Optional[dict] = None,
    ckpt_dir:  str = "results/f2/checkpoints",
    label:     str = "f2",
) -> dict:
    """
    Run full F2 pipeline and return metrics dict.

    Returns
    -------
    dict with all f2_metrics keys for the test split.
    """
    import pandas as pd
    use_amp = device.type == "cuda"
    cfg     = make_cfg(overrides)

    # ── Stage A ──────────────────────────────────────────────────────────────
    if cfg["skip_stage_A"]:
        teacher_tr = _doctor_teacher_preds(df_train, feature_info)
        teacher_va = _doctor_teacher_preds(df_val,   feature_info)
        teacher_te = _doctor_teacher_preds(df_test,  feature_info)
        print(f"[{label}] Stage A skipped — using doctor Rx as teacher anchor.")
    else:
        print(f"[{label}] Running Stage A (CatBoost)...")
        cb_models  = train_stage_A(df_train, df_val, feature_info)
        teacher_tr = get_stage_A_preds(df_train, cb_models, feature_info)
        teacher_va = get_stage_A_preds(df_val,   cb_models, feature_info)
        teacher_te = get_stage_A_preds(df_test,  cb_models, feature_info)

    # ── Datasets ─────────────────────────────────────────────────────────────
    def f1_fn(df_part):
        return _f1_predict_raw(df_part, feature_info, f1_model, device)

    orig_features = feature_info["original_feature_names"]
    ds_tr = RxDataset(df_train, feature_info, f1_fn, teacher_preds=teacher_tr)
    ds_va = RxDataset(df_val,   feature_info, f1_fn, teacher_preds=teacher_va)
    ds_te = RxDataset(df_test,  feature_info, f1_fn, teacher_preds=teacher_te)

    dl_kwargs = {"num_workers": 0, "pin_memory": False}
    dl_tr = DataLoader(ds_tr, batch_size=cfg["F2_BATCH_SIZE"], shuffle=True,  **dl_kwargs)
    dl_va = DataLoader(ds_va, batch_size=cfg["F2_BATCH_SIZE"], shuffle=False, **dl_kwargs)

    # ── Stage B model ────────────────────────────────────────────────────────
    K      = int(df_train[CAT_RX].max()) + 1
    in_dim = len(orig_features)

    if cfg["use_linear_head"]:
        # A-F2-7: replace MLP with linear
        class LinearRxHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.head_cont = nn.Linear(in_dim, len(CONT_RX))
                self.head_cat  = nn.Linear(in_dim, K)
            def forward(self, x):
                return self.head_cont(x), self.head_cat(x)
        model_b = LinearRxHead()
    else:
        model_b = F2RxHead(in_dim=in_dim, n_cont=len(CONT_RX), n_cat=K,
                           hidden=cfg["F2_HIDDEN"], dropout=cfg["F2_DROPOUT"])

    ckpt_path = os.path.join(ckpt_dir, f"{label}_best.pth")
    print(f"[{label}] Running Stage B (MLP optimization)...")
    train_stage_B(model_b, dl_tr, dl_va, f1_model, feature_info,
                  device, cfg, use_amp, ckpt_path)

    if os.path.exists(ckpt_path):
        model_b.load_state_dict(torch.load(ckpt_path, map_location=device,
                                           weights_only=True))

    # ── Inference on test set ─────────────────────────────────────────────────
    rx_test = infer_prescriptions(model_b, df_test, feature_info,
                                  f1_model, device, cfg)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    ktv_actual = df_test[OUTCOME_COL].values.astype(float)
    patched    = df_test[orig_features].copy()
    for c in [CAT_RX] + CONT_RX:
        patched[c] = rx_test[c].values
    ktv_model  = _f1_predict_raw(patched, feature_info, f1_model, device)
    ktv_doctor = _f1_predict_raw(df_test[orig_features], feature_info, f1_model, device)

    rx_doctor_dict = {col: df_test[col].values for col in CONT_RX}
    rx_model_dict  = {col: rx_test[col].values for col in CONT_RX}

    metrics = f2_metrics(ktv_doctor, ktv_model, ktv_actual,
                         rx_doctor_dict, rx_model_dict)

    # Categorical accuracy
    gt_cat = df_test[CAT_RX].astype(int).values
    pr_cat = rx_test[CAT_RX].astype(int).values
    metrics["cat_accuracy"] = float(np.mean(gt_cat == pr_cat))

    print(f"[{label}] p_pass={metrics.get('p_pass', float('nan')):.4f}  "
          f"delta_ktv_fail={metrics.get('delta_ktv_fail', float('nan')):.4f}")
    return metrics
