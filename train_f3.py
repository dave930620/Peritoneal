"""
train_f3.py — F3: first-visit prescription model (no previous prescription).

Key differences from F2
-----------------------
1. Hierarchical Stage A
       CatBoost classifier → predict long term PD system (CAT_RX)
       CatBoost regressors → predict CONT_RX conditioned on PD system
   Default: "chain" mode (predicted PD system appended as extra feature).
   Alternative: "stratified" mode (separate regressors per class).

2. No lag features
   Input is purely the patient's clinical features at the current visit.
   Every row is treated as if it were a first visit.

3. Trust-region centred on Stage A, not doctor's current Rx
   At real inference the doctor's Rx is unknown; Stage A provides the
   clinically-initialised anchor for Stage B.  (f2_core.py already
   implements this correctly — we reuse it unchanged.)

Usage
-----
    python train_f3.py                       # chain mode (default)
    python train_f3.py --mode stratified     # per-class Stage A

Requirements
------------
    pip install catboost
    feature_info.pkl and saint_pd_model.pth must exist (run train_f1.py first).

Outputs  →  report_model3/
    f3_stageB_best.pth          Stage B checkpoint
    val_model_rx.csv            Generated prescriptions (validation set)
    test_model_rx.csv           Generated prescriptions (test set)
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
    F1_MODEL_PATH, F2_BATCH_SIZE, F2_DROPOUT, F2_HIDDEN, FEATURE_INFO_PATH,
    OUTCOME_COL, PATIENT_ID_COL, SEED, TEST_FRAC, TRAIN_FRAC, VAL_FRAC,
)
from src.data.preprocessor import (
    RxDataset, load_feature_info, patient_split,
    sanity_check_data, standardize_like_f1, validate_columns,
)
from src.models.f2_head import F2RxHead
from src.models.stage_a_hierarchical import HierarchicalStageA
from src.training.f2_core import infer_prescriptions, make_cfg, train_stage_B
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


def evaluate_and_save(
    df_eval: pd.DataFrame, rx_df: pd.DataFrame, prefix: str,
    feature_info: dict, f1_model: nn.Module, device: torch.device,
) -> None:
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\n=== Evaluation: {prefix} ===")
    doc_out   = df_eval[OUTCOME_COL].values.astype(float)
    mask_pass = doc_out >= CLINICAL_THRESHOLD
    mask_fail = ~mask_pass

    # Continuous Rx similarity
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

    # Categorical accuracy
    gt_cat = df_eval[CAT_RX].values.astype(int)
    pr_cat = rx_df[CAT_RX].values.astype(int)
    acc_all  = float((gt_cat == pr_cat).mean())
    acc_pass = float((gt_cat[mask_pass] == pr_cat[mask_pass]).mean()) if mask_pass.any() else float("nan")
    acc_fail = float((gt_cat[mask_fail] == pr_cat[mask_fail]).mean()) if mask_fail.any() else float("nan")
    print(f"  {CAT_RX}: Acc(all/PASS/FAIL)={acc_all:.4f}/{acc_pass:.4f}/{acc_fail:.4f}")

    # Kt/V threshold analysis (model prescription evaluated through F1)
    orig = feature_info["original_feature_names"]
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
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    set_seed(SEED)
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    device    = get_device()
    print_device_info(device)
    use_amp   = is_amp_supported(device)
    dl_kwargs = get_dataloader_kwargs(device)

    # ------------------------------------------------------------------
    # 1. Data + feature_info
    # ------------------------------------------------------------------
    print(f"[F3] Loading data: {DATA_CSV}")
    df_raw = pd.read_csv(DATA_CSV)
    print(f"[F3] Shape: {df_raw.shape}")

    feature_info  = load_feature_info(FEATURE_INFO_PATH)
    orig_features = feature_info["original_feature_names"]

    validate_columns(df_raw, orig_features + [OUTCOME_COL, PATIENT_ID_COL], "train_f3")
    sanity_check_data(df_raw)

    # ------------------------------------------------------------------
    # 2. Frozen F1 oracle
    # ------------------------------------------------------------------
    print(f"[F3] Loading F1 model: {F1_MODEL_PATH}")
    f1_model = load_f1_model(F1_MODEL_PATH, feature_info, device)

    def f1_fn(df_part: pd.DataFrame) -> np.ndarray:
        return f1_predict_raw(df_part, feature_info, f1_model, device)

    # ------------------------------------------------------------------
    # 3. Patient-wise split  (same deterministic bucketing as F2)
    # ------------------------------------------------------------------
    cols_needed = [PATIENT_ID_COL, OUTCOME_COL] + orig_features
    df_use = df_raw[cols_needed].copy()
    df_use = df_use.loc[:, ~df_use.columns.duplicated()]
    tr, va, te = patient_split(df_use, PATIENT_ID_COL, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)

    # ------------------------------------------------------------------
    # 4. Hierarchical Stage A  (no lag features)
    # ------------------------------------------------------------------
    patient_cols = get_patient_feature_cols(feature_info)
    print(f"\n[F3] Hierarchical Stage A  mode={args.mode}")
    stage_a = HierarchicalStageA(mode=args.mode, save_dir="catboost_models_f3")
    stage_a.fit(tr, va, patient_cols)

    teacher_tr = stage_a.predict(tr, patient_cols)
    teacher_va = stage_a.predict(va, patient_cols)
    teacher_te = stage_a.predict(te, patient_cols)

    # Stage A val-set similarity report
    print("\n[A] Stage A val-set similarity:")
    for j, col in enumerate(CONT_RX):
        gt   = va[col].values.astype(float)
        pred = teacher_va["cont"][:, j].astype(float)
        r    = pearson_correlation(gt, pred)
        rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
        print(f"  {col}: Pearson={r:.4f}  RMSE={rmse:.4f}")
    acc_a = float((va[CAT_RX].astype(int).values == teacher_va["cat"]).mean())
    print(f"  {CAT_RX}: Accuracy={acc_a:.4f}")

    # ------------------------------------------------------------------
    # 5. Datasets + DataLoaders  (lag_arr=None throughout)
    # ------------------------------------------------------------------
    ds_tr = RxDataset(tr, feature_info, f1_fn, teacher_preds=teacher_tr)
    ds_va = RxDataset(va, feature_info, f1_fn, teacher_preds=teacher_va)

    dl_tr = DataLoader(ds_tr, batch_size=F2_BATCH_SIZE, shuffle=True,  **dl_kwargs)
    dl_va = DataLoader(ds_va, batch_size=F2_BATCH_SIZE, shuffle=False, **dl_kwargs)

    # ------------------------------------------------------------------
    # 6. Stage B — reuse f2_core (trust region already on Stage A teacher)
    # ------------------------------------------------------------------
    K       = int(df_use[CAT_RX].max()) + 1
    model_b = F2RxHead(
        in_dim=len(orig_features), n_cont=len(CONT_RX), n_cat=K,
        hidden=F2_HIDDEN, dropout=F2_DROPOUT,
    ).to(device)

    cfg      = make_cfg()
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
    # ------------------------------------------------------------------
    evaluate_and_save(va, rx_val,  "val",  feature_info, f1_model, device)
    evaluate_and_save(te, rx_test, "test", feature_info, f1_model, device)

    rx_val.to_csv(os.path.join(REPORT_DIR,  "val_model_rx.csv"),  index=False)
    rx_test.to_csv(os.path.join(REPORT_DIR, "test_model_rx.csv"), index=False)
    print(f"\n[F3] Done. All outputs saved to: {REPORT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F3 first-visit prescription model")
    parser.add_argument(
        "--mode", choices=["chain", "stratified"], default="chain",
        help="Hierarchical Stage A mode. "
             "'chain': append predicted PD system as feature (default, recommended). "
             "'stratified': train separate models per PD system class.",
    )
    main(parser.parse_args())
