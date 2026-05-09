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
    F1_MODEL_PATH, F2_BATCH_SIZE, F2_DROPOUT, F2_HIDDEN, FEATURE_INFO_PATH,
    OUTCOME_COL, PATIENT_ID_COL, SEED, STEP_MAP, TEST_FRAC, TRAIN_FRAC, VAL_FRAC,
)
from src.data.preprocessor import (
    RxDataset, load_feature_info, patient_split, remove_outlier_patients,
    sanity_check_data, standardize_like_f1, validate_columns,
)
from src.models.f2_head import F2RxHead
from src.models.stage_a_hierarchical import HierarchicalStageA
from src.training.f2_core import (
    get_stage_A_preds, infer_prescriptions, make_cfg,
    train_stage_A, train_stage_B,
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


def print_stage_a_similarity(df_val, teacher_va: dict, label: str = "Stage A") -> None:
    """Print Pearson / RMSE between Stage A predictions and doctor ground truth."""
    print(f"\n[A] {label} val-set similarity:")
    for j, col in enumerate(CONT_RX):
        gt   = df_val[col].values.astype(float)
        pred = teacher_va["cont"][:, j].astype(float)
        r    = pearson_correlation(gt, pred)
        rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
        print(f"  {col}: Pearson={r:.4f}  RMSE={rmse:.4f}")
    acc = float((df_val[CAT_RX].astype(int).values == teacher_va["cat"]).mean())
    print(f"  {CAT_RX}: Accuracy={acc:.4f}")


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
    print(f"[F3] Shape: {df_raw.shape}  mode={args.mode}")

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
    # 3. Patient-wise split (outliers removed before split)
    # ------------------------------------------------------------------
    cols_needed = [PATIENT_ID_COL, OUTCOME_COL] + orig_features
    df_use = df_raw[cols_needed].copy()
    df_use = df_use.loc[:, ~df_use.columns.duplicated()]

    print("\n[F3] Removing outlier patients (>3σ in any prescription variable) ...")
    df_use = remove_outlier_patients(df_use)

    tr, va, te = patient_split(df_use, PATIENT_ID_COL, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)

    patient_cols = get_patient_feature_cols(feature_info)

    # ------------------------------------------------------------------
    # 4. Stage A
    # ------------------------------------------------------------------
    stage_a_hier = None   # kept for oracle evaluation if needed

    if args.mode == "flat":
        print(f"\n[F3] Stage A — flat {args.stageA.upper()} (no lag features)")
        sa_models  = train_stage_A(tr, va, feature_info, model_type=args.stageA)
        teacher_tr = get_stage_A_preds(tr, sa_models, feature_info)
        teacher_va = get_stage_A_preds(va, sa_models, feature_info)
        teacher_te = get_stage_A_preds(te, sa_models, feature_info)

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
        choices=["catboost", "xgboost", "lgbm", "rf", "linear", "mlp_nn", "transformer_nn"],
        default="rf",
        help=(
            "Stage A model type for flat mode (default: rf). "
            "'rf': Random Forest — no early stopping, stable with weak signal. "
            "'lgbm': LightGBM — fixed budget + L2 reg, often beats RF. "
            "'xgboost': XGBoost — binary classification fix included. "
            "'catboost': CatBoost (original F2 default). "
            "'linear': Ridge/Logistic regression (interpretable baseline)."
        ),
    )
    parser.add_argument(
        "--oracle", action="store_true",
        help="(stratified/chain only) Evaluate with true class labels — "
             "shows upper bound of hierarchical Stage A.",
    )
    main(parser.parse_args())
