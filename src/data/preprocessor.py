"""
Data preprocessing for F1 and F2 models.

Key responsibilities
--------------------
- Separate discrete (categorical) and continuous features.
- Impute missing values (median strategy for continuous columns).
- Scale continuous features with z-score standardization.
- Save/load a feature_info dict so that inference uses identical transforms.
- Provide the RxDataset class used by F2 training.

Design notes
------------
- Volume (L) is excluded via EXCLUDE_COLUMNS in config.py.
- The RxDataset accepts pre-computed CatBoost teacher predictions so that
  Stage B has a clinically-initialized proximal anchor.
"""

import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.config import (
    CAT_RX, CONT_RX, DISCRETE_COLUMNS, EXCLUDE_COLUMNS, OUTCOME_COL,
    PATIENT_ID_COL,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    """Raise ValueError if any required column is missing from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing columns: {missing}")


def sanity_check_data(df: pd.DataFrame) -> None:
    """Print basic distribution checks. Does NOT modify df."""
    print(f"\n[Sanity] Shape: {df.shape}")

    # Prescription columns should have non-trivial variance
    rx_cols = [c for c in CONT_RX if c in df.columns]
    for col in rx_cols:
        mn, mx, sd = df[col].min(), df[col].max(), df[col].std()
        n_zero = (df[col] == 0).sum()
        pct_zero = n_zero / len(df) * 100
        print(f"[Sanity] {col}: min={mn:.2f} max={mx:.2f} std={sd:.3f} zeros={pct_zero:.1f}%")
        if sd < 1e-6:
            print(f"  [WARNING] {col} has near-zero variance — check your data!")

    # Target distribution
    if OUTCOME_COL in df.columns:
        ktv = df[OUTCOME_COL]
        pct_pass = (ktv > 1.7).mean() * 100
        print(f"[Sanity] {OUTCOME_COL}: mean={ktv.mean():.3f} std={ktv.std():.3f} "
              f"pass_rate(>1.7)={pct_pass:.1f}%")


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_data(
    data: pd.DataFrame,
    feature_info_path: str = "feature_info.pkl",
) -> Tuple:
    """Preprocess raw data for F1 training.

    Steps
    -----
    1. Drop excluded columns and the target.
    2. Separate discrete and continuous feature columns.
    3. Impute continuous features with median.
    4. Z-score scale continuous features.
    5. Save preprocessing artifacts to feature_info_path.

    Returns
    -------
    features : pd.DataFrame          preprocessed feature matrix
    target   : pd.Series             target values (PD Kt/V)
    patient_ids : pd.Series | None   PatientID series (for group-wise split)
    discrete_feature_indices : list  column indices of discrete features
    continuous_feature_indices : list column indices of continuous features
    feature_names : list[str]        final ordered feature names
    feature_info : dict              serializable config for inference
    """
    imputer = SimpleImputer(strategy="median")

    # Drop exclusions + target (keep PatientID temporarily for group split)
    cols_to_drop = [c for c in EXCLUDE_COLUMNS if c in data.columns] + [OUTCOME_COL]
    features = data.drop(columns=cols_to_drop, errors="ignore")

    # Extract PatientID for group-wise split, then drop it
    patient_ids = None
    if PATIENT_ID_COL in features.columns:
        patient_ids = features[PATIENT_ID_COL].copy()
        features = features.drop(columns=[PATIENT_ID_COL])

    target = data[OUTCOME_COL]

    # Partition into discrete vs continuous
    discrete_cols   = [c for c in DISCRETE_COLUMNS  if c in features.columns]
    continuous_cols = [c for c in features.columns  if c not in discrete_cols]

    print(f"[Preprocess] Discrete features  : {len(discrete_cols)}")
    print(f"[Preprocess] Continuous features: {len(continuous_cols)}")

    # Impute + scale continuous features
    cont_data         = features[continuous_cols]
    cont_imputed      = imputer.fit_transform(cont_data)
    scaler            = StandardScaler()
    cont_scaled       = scaler.fit_transform(cont_imputed)
    features[continuous_cols] = cont_scaled

    # Build index maps
    all_cols = list(features.columns)
    discrete_feature_indices   = [all_cols.index(c) for c in discrete_cols]
    continuous_feature_indices = [all_cols.index(c) for c in continuous_cols]

    feature_info = {
        "original_feature_names":    all_cols,
        "discrete_feature_names":    discrete_cols,
        "continuous_feature_names":  continuous_cols,
        "discrete_feature_indices":  discrete_feature_indices,
        "continuous_feature_indices": continuous_feature_indices,
        "feature_name_to_index":     {n: i for i, n in enumerate(all_cols)},
        "index_to_feature_name":     {i: n for i, n in enumerate(all_cols)},
        "exclude_columns":           EXCLUDE_COLUMNS,
        "output_column":             OUTCOME_COL,
        "imputer":                   imputer,
        "scaler":                    scaler,
    }

    with open(feature_info_path, "wb") as f:
        pickle.dump(feature_info, f)
    print(f"[Preprocess] Saved feature_info to '{feature_info_path}'")

    return (features, target, patient_ids,
            discrete_feature_indices, continuous_feature_indices,
            all_cols, feature_info)


def load_feature_info(path: str = "feature_info.pkl") -> dict:
    """Load a saved feature_info dict."""
    with open(path, "rb") as f:
        return pickle.load(f)


def standardize_like_f1(df: pd.DataFrame, feature_info: dict) -> pd.DataFrame:
    """Apply F1's fitted imputer and scaler to df.

    Returns a copy restricted to original_feature_names, with continuous
    features z-scored using F1's fitted transforms.
    """
    orig_features  = feature_info["original_feature_names"]
    cont_names     = feature_info["continuous_feature_names"]
    imputer        = feature_info["imputer"]
    scaler         = feature_info["scaler"]

    validate_columns(df, orig_features, "standardize_like_f1")

    X = df[orig_features].copy()
    cont_df      = X[cont_names].astype("float64")
    cont_imputed = imputer.transform(cont_df)
    cont_scaled  = scaler.transform(cont_imputed)
    X[cont_names] = cont_scaled
    return X


def patient_split(
    df: pd.DataFrame,
    pid_col: str = PATIENT_ID_COL,
    train_frac: float = 0.75,
    val_frac: float = 0.10,
    test_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic patient-wise split using hash bucketing.

    Ensures all records for a patient go to the same split,
    preventing data leakage between train/val/test.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"

    pids = df[pid_col].astype(str).unique()

    def _bucket(pid: str) -> str:
        v = (hash(pid) % 10**9) / 10**9
        if v < train_frac:
            return "train"
        if v < train_frac + val_frac:
            return "val"
        return "test"

    part = df[pid_col].astype(str).map(_bucket)
    tr = df[part == "train"].copy()
    va = df[part == "val"].copy()
    te = df[part == "test"].copy()
    print(f"[Split] train={len(tr)} val={len(va)} test={len(te)} records")
    return tr, va, te


# ---------------------------------------------------------------------------
# F2 Dataset
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lag feature construction (for F2 with temporal context)
# ---------------------------------------------------------------------------

def build_lag_features(
    df: pd.DataFrame,
    patient_id_col: str = PATIENT_ID_COL,
    time_col: str = "記錄時間",
) -> pd.DataFrame:
    """Add one-step lag features sorted by time per patient.

    For each record, the previous visit's prescription values and Kt/V are
    appended as new columns (prev_{col}).  First-visit rows have NaN lags
    and are flagged with is_first_visit=1.

    Call drop_first_visits() after this to remove untrainable first-visit rows.

    Added columns
    -------------
    prev_{OUTCOME_COL}       : previous Kt/V
    prev_{col} for col in CONT_RX + [CAT_RX]
    is_first_visit           : 1 if no prior record exists for this patient
    """
    df = df.sort_values([patient_id_col, time_col]).copy()
    lag_sources = [OUTCOME_COL] + CONT_RX + [CAT_RX]
    for col in lag_sources:
        df[f"prev_{col}"] = df.groupby(patient_id_col)[col].shift(1)
    df["is_first_visit"] = df[f"prev_{OUTCOME_COL}"].isna().astype(int)
    return df


def drop_first_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with no prior visit (lag features would be NaN).

    Must be called after build_lag_features().
    """
    return df[df["is_first_visit"] == 0].drop(columns=["is_first_visit"]).reset_index(drop=True)


def lag_feature_names() -> list:
    """Return the ordered list of lag feature column names."""
    return [f"prev_{col}" for col in [OUTCOME_COL] + CONT_RX + [CAT_RX]]


class RxDataset(Dataset):
    """Dataset for F2 prescription learning.

    Each item provides:
    - X             : standardized features with Rx columns zeroed
                      (F2 learns to regenerate them from patient characteristics)
    - y_cont        : doctor's continuous prescription in z-space (trust-region center)
    - y_cat         : doctor's categorical prescription class
    - y_outcome     : ground-truth PD Kt/V (for analysis only, not training loss)
    - y_doc_hat     : F1's predicted Kt/V under doctor's prescription (PASS/FAIL gate)
    - y_cont_teacher: CatBoost teacher prescription in z-space (proximal anchor for Stage B)
    - y_cat_teacher : CatBoost teacher categorical prescription

    Parameters
    ----------
    df : pd.DataFrame
        Records for this split. Must include ORIG_FEATURES + outcome + PatientID.
    feature_info : dict
        Loaded from feature_info.pkl (output of preprocess_data).
    f1_predict_fn : callable
        Function that takes a raw DataFrame and returns np.ndarray of predicted Kt/V.
        Used to compute y_doc_hat (PASS/FAIL gate).
    teacher_preds : dict | None
        Optional dict with keys 'cont' (np.ndarray, shape [N, len(CONT_RX)], raw values)
        and 'cat' (np.ndarray, shape [N,], int).
        When None, the doctor's own prescription is used as the teacher anchor.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_info: dict,
        f1_predict_fn,
        teacher_preds: Optional[dict] = None,
    ):
        self.df = df.reset_index(drop=True)
        orig_features = feature_info["original_feature_names"]
        cont_names    = feature_info["continuous_feature_names"]
        scaler        = feature_info["scaler"]

        # --- Build standardized feature matrix with Rx columns zeroed ---
        Xstd = standardize_like_f1(self.df, feature_info)
        for c in [CAT_RX] + CONT_RX:
            if c in Xstd.columns:
                Xstd[c] = 0.0 if c != CAT_RX else 0
        self.X = Xstd.values.astype(np.float32)

        # --- Doctor's continuous prescription in z-space ---
        self.y_cont = np.zeros((len(df), len(CONT_RX)), dtype=np.float32)
        for j, col in enumerate(CONT_RX):
            idx = cont_names.index(col)
            mu  = scaler.mean_[idx]
            sd  = scaler.scale_[idx] if scaler.scale_[idx] > 0 else 1.0
            self.y_cont[:, j] = (df[col].values.astype(np.float32) - mu) / sd

        # --- Doctor's categorical prescription ---
        self.y_cat = df[CAT_RX].astype(int).values

        # --- Ground-truth outcome (for reporting only) ---
        self.y_outcome = df[OUTCOME_COL].astype(float).values.astype(np.float32)

        # --- PASS/FAIL gate: F1 prediction under Stage A prescription ---
        # If teacher_preds are available (Stage A ran), evaluate F1 on Stage A's Rx
        # so that the gate is consistent with inference (where doctor's Rx is unknown).
        if teacher_preds is not None:
            df_for_gate = df[orig_features].copy()
            for j, col in enumerate(CONT_RX):
                df_for_gate[col] = teacher_preds["cont"][:, j]
            df_for_gate[CAT_RX] = teacher_preds["cat"]
            self.y_doc_hat = f1_predict_fn(df_for_gate).astype(np.float32)
        else:
            # Fallback (no Stage A): use doctor's Rx — only for C5/no-anchor baseline
            self.y_doc_hat = f1_predict_fn(df[orig_features]).astype(np.float32)

        # --- CatBoost teacher anchor ---
        if teacher_preds is not None:
            # Convert raw CatBoost predictions to z-space
            self.y_cont_teacher = np.zeros_like(self.y_cont)
            for j, col in enumerate(CONT_RX):
                idx = cont_names.index(col)
                mu  = scaler.mean_[idx]
                sd  = scaler.scale_[idx] if scaler.scale_[idx] > 0 else 1.0
                self.y_cont_teacher[:, j] = (
                    teacher_preds["cont"][:, j].astype(np.float32) - mu
                ) / sd
            self.y_cat_teacher = teacher_preds["cat"].astype(np.int64)
        else:
            # Fallback: use doctor's prescription as teacher
            self.y_cont_teacher = self.y_cont.copy()
            self.y_cat_teacher  = self.y_cat.copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_cont[idx],
            self.y_cat[idx],
            self.y_outcome[idx],
            self.y_doc_hat[idx],
            self.y_cont_teacher[idx],
            self.y_cat_teacher[idx],
        )
