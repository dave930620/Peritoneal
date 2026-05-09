"""
HierarchicalStageA — two-step CatBoost doctor-mimic for F3.

Step 1: CatBoost classifier  → predict long term PD system (CAT_RX)
Step 2: CatBoost regressors  → predict CONT_RX, conditioned on Step 1

Modes
-----
stratified (default)
    Separate CONT_RX regressor set per PD-system class, trained on TRUE class
    labels.  Zero-inflated variables (nighttime Rx) become trivial within each
    class (CAPD patients always have 0 nighttime PD, etc.), so within-class
    signal is much stronger.  Falls back to global models for rare classes.

chain
    Predicted PD system is appended as an extra numeric feature to every
    CONT_RX regressor.  Simpler but propagates CAT_RX prediction noise into
    all downstream regressors — only useful when class counts are very small.

Why CONT_RX regressors have NO early stopping
---------------------------------------------
When clinical features have weak signal for a target, CatBoost early stopping
on a small val set degenerates to best_iter=0 (constant predictor) and
Pearson=NaN.  Instead we use a fixed iteration budget with L2 regularisation
(l2_leaf_reg) to control overfitting, which is more robust in low-signal
settings.
"""

import numpy as np
from pathlib import Path

from src.config import (
    CAT_RX, CATBOOST_DEPTH, CATBOOST_EARLY_STOP,
    CATBOOST_ITERATIONS, CATBOOST_LR, CONT_RX, SEED, STEP_MAP,
)

MIN_CLASS_SAMPLES = 20

# Dedicated hyperparams for the Step-1 PD-system classifier.
# More capacity + balanced weights because the signal is weak and classes may
# be unbalanced; higher patience to avoid premature stopping.
_CLF_ITERATIONS  = 2000
_CLF_DEPTH       = 8
_CLF_EARLY_STOP  = 150

# Fixed-budget hyperparams for Step-2 CONT_RX regressors (no early stopping).
# L2 regularisation replaces early stopping to prevent overfitting.
_REG_ITERATIONS  = 800
_REG_DEPTH       = 6
_REG_L2          = 5.0


class HierarchicalStageA:
    """Hierarchical doctor-mimic: classify PD system, then regress dose."""

    def __init__(self, mode: str = "stratified", save_dir: str = ""):
        if mode not in ("chain", "stratified"):
            raise ValueError(f"mode must be 'chain' or 'stratified', got '{mode}'")
        self.mode      = mode
        self.save_dir  = save_dir
        self.cat_model = None
        self.cont_models: dict = {}   # chain: {col: model}  stratified: {cls: {col: model}}
        self._fallback: dict   = {}   # stratified only: global fallback per col
        self.pd_classes: list  = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df_train, df_val, patient_cols: list) -> None:
        if self.save_dir:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self._fit_step1(df_train, df_val, patient_cols)
        if self.mode == "chain":
            self._fit_chain(df_train, df_val, patient_cols)
        else:
            self._fit_stratified(df_train, df_val, patient_cols)

    def predict(self, df, patient_cols: list) -> dict:
        """Return {'cont': np.ndarray (N, len(CONT_RX)), 'cat': np.ndarray (N,)}."""
        X         = df[patient_cols]
        cat_preds = self.cat_model.predict(X).astype(int).flatten()
        cont_preds = np.zeros((len(df), len(CONT_RX)), dtype=np.float32)

        if self.mode == "chain":
            X_ext = X.copy()
            X_ext["_pd_system"] = cat_preds.astype(float)
            for j, col in enumerate(CONT_RX):
                raw = self.cont_models[col].predict(X_ext).astype(np.float32)
                raw = np.maximum(0.0, raw)
                cont_preds[:, j] = np.round(raw / STEP_MAP[col]) * STEP_MAP[col]
        else:
            for cls in self.pd_classes:
                mask = cat_preds == cls
                if not mask.any():
                    continue
                models = self.cont_models.get(cls) or self._fallback
                for j, col in enumerate(CONT_RX):
                    raw = models[col].predict(df.loc[mask, patient_cols]).astype(np.float32)
                    raw = np.maximum(0.0, raw)
                    cont_preds[mask, j] = np.round(raw / STEP_MAP[col]) * STEP_MAP[col]

        return {"cont": cont_preds, "cat": cat_preds}

    # ------------------------------------------------------------------
    # Step 1 — PD-system classifier
    # ------------------------------------------------------------------

    def _fit_step1(self, df_train, df_val, patient_cols: list) -> None:
        from catboost import CatBoostClassifier

        X_tr = df_train[patient_cols]
        X_va = df_val[patient_cols]
        self.pd_classes = sorted(df_train[CAT_RX].astype(int).unique().tolist())
        loss_fn = "Logloss" if len(self.pd_classes) == 2 else "MultiClass"

        print(f"[HierarchicalA] Step 1 — PD-system classifier ({loss_fn}) ...")
        self.cat_model = CatBoostClassifier(
            iterations=_CLF_ITERATIONS,
            learning_rate=CATBOOST_LR,
            depth=_CLF_DEPTH,
            loss_function=loss_fn,
            auto_class_weights="Balanced",   # handles class imbalance
            random_seed=SEED,
            verbose=0,
        )
        self.cat_model.fit(
            X_tr, df_train[CAT_RX].astype(int),
            eval_set=(X_va, df_val[CAT_RX].astype(int)),
            early_stopping_rounds=_CLF_EARLY_STOP,
        )
        val_acc = (
            self.cat_model.predict(X_va).flatten().astype(int)
            == df_val[CAT_RX].astype(int).values
        ).mean()
        # Majority-class baseline to contextualise the accuracy
        majority_acc = df_val[CAT_RX].value_counts(normalize=True).max()
        print(f"  best_iter={self.cat_model.best_iteration_}  "
              f"val_acc={val_acc:.4f}  (majority_baseline={majority_acc:.4f})  "
              f"classes={self.pd_classes}")

    # ------------------------------------------------------------------
    # Step 2a — chain mode
    # ------------------------------------------------------------------

    def _fit_chain(self, df_train, df_val, patient_cols: list) -> None:
        from catboost import CatBoostRegressor

        # Use predicted (not true) PD system: avoids teacher-forcing mismatch.
        pred_tr = self.cat_model.predict(df_train[patient_cols]).astype(float).flatten()

        X_tr = df_train[patient_cols].copy()
        X_tr["_pd_system"] = pred_tr

        print("[HierarchicalA] Step 2 (chain) — CONT_RX regressors (fixed budget, no early stop) ...")
        for col in CONT_RX:
            m = CatBoostRegressor(
                iterations=_REG_ITERATIONS,
                learning_rate=CATBOOST_LR,
                depth=_REG_DEPTH,
                l2_leaf_reg=_REG_L2,
                loss_function="RMSE",
                use_best_model=False,   # train for full budget; no val set
                random_seed=SEED,
                verbose=0,
            )
            m.fit(X_tr, df_train[col])   # no eval_set → no early stopping
            self.cont_models[col] = m
            print(f"  {col}: done ({_REG_ITERATIONS} iters)")

    # ------------------------------------------------------------------
    # Step 2b — stratified mode
    # ------------------------------------------------------------------

    def _fit_stratified(self, df_train, df_val, patient_cols: list) -> None:
        from catboost import CatBoostRegressor

        def _make_reg():
            return CatBoostRegressor(
                iterations=_REG_ITERATIONS,
                learning_rate=CATBOOST_LR,
                depth=_REG_DEPTH,
                l2_leaf_reg=_REG_L2,
                loss_function="RMSE",
                use_best_model=False,
                random_seed=SEED,
                verbose=0,
            )

        # Global fallback (all classes pooled, no early stopping)
        print("[HierarchicalA] Step 2 (stratified) — global fallback models ...")
        for col in CONT_RX:
            m = _make_reg()
            m.fit(df_train[patient_cols], df_train[col])
            self._fallback[col] = m
        print(f"  done ({_REG_ITERATIONS} iters each)")

        # Per-class models — trained on TRUE class labels (no CAT_RX prediction noise)
        for cls in self.pd_classes:
            mask_tr = df_train[CAT_RX].astype(int) == cls
            n_tr = int(mask_tr.sum())
            print(f"[HierarchicalA] Class {cls}: train_rows={n_tr}")

            if n_tr < MIN_CLASS_SAMPLES:
                print(f"  → < {MIN_CLASS_SAMPLES} samples; will use global fallback.")
                self.cont_models[cls] = None
                continue

            self.cont_models[cls] = {}
            X_tr_c = df_train.loc[mask_tr, patient_cols]
            for col in CONT_RX:
                m = _make_reg()
                m.fit(X_tr_c, df_train.loc[mask_tr, col])
                self.cont_models[cls][col] = m
            print(f"  → Trained {len(CONT_RX)} regressors for class {cls}.")
