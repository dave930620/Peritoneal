"""
HierarchicalStageA — two-step CatBoost doctor-mimic for F3.

Step 1: CatBoost classifier  → predict long term PD system (CAT_RX)
Step 2: CatBoost regressors  → predict CONT_RX, conditioned on Step 1

Modes
-----
chain (default)
    Predicted PD system is appended as an extra numeric feature to every
    CONT_RX regressor.  Single set of models; robust when per-class counts
    are small, and avoids the cold-start problem at inference.

stratified
    Separate CONT_RX regressor set per PD-system class.  Captures richer
    within-class patterns, but needs >= MIN_CLASS_SAMPLES rows per class.
    Falls back to global models for rare classes.
"""

import numpy as np
from pathlib import Path

from src.config import (
    CAT_RX, CATBOOST_DEPTH, CATBOOST_EARLY_STOP,
    CATBOOST_ITERATIONS, CATBOOST_LR, CONT_RX, SEED, STEP_MAP,
)

MIN_CLASS_SAMPLES = 20


class HierarchicalStageA:
    """Hierarchical doctor-mimic: classify PD system, then regress dose."""

    def __init__(self, mode: str = "chain", save_dir: str = ""):
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
                # Use per-class models if available, otherwise global fallback
                models = self.cont_models.get(cls) or self._fallback
                for j, col in enumerate(CONT_RX):
                    raw = models[col].predict(df.loc[mask, patient_cols]).astype(np.float32)
                    raw = np.maximum(0.0, raw)
                    cont_preds[mask, j] = np.round(raw / STEP_MAP[col]) * STEP_MAP[col]

        return {"cont": cont_preds, "cat": cat_preds}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_step1(self, df_train, df_val, patient_cols: list) -> None:
        from catboost import CatBoostClassifier

        X_tr = df_train[patient_cols]
        X_va = df_val[patient_cols]

        print("[HierarchicalA] Step 1 — PD-system classifier (CAT_RX) ...")
        self.cat_model = CatBoostClassifier(
            iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
            depth=CATBOOST_DEPTH, loss_function="MultiClass",
            random_seed=SEED, verbose=0,
        )
        self.cat_model.fit(
            X_tr, df_train[CAT_RX].astype(int),
            eval_set=(X_va, df_val[CAT_RX].astype(int)),
            early_stopping_rounds=CATBOOST_EARLY_STOP,
        )
        self.pd_classes = sorted(df_train[CAT_RX].astype(int).unique().tolist())
        val_acc = (
            self.cat_model.predict(X_va).flatten().astype(int)
            == df_val[CAT_RX].astype(int).values
        ).mean()
        print(f"  best_iter={self.cat_model.best_iteration_}  "
              f"val_acc={val_acc:.4f}  classes={self.pd_classes}")

    def _fit_chain(self, df_train, df_val, patient_cols: list) -> None:
        from catboost import CatBoostRegressor

        # Use predicted (not true) PD system so train and inference share the same
        # feature distribution — avoids teacher-forcing mismatch at test time.
        pred_tr = self.cat_model.predict(df_train[patient_cols]).astype(float).flatten()
        pred_va = self.cat_model.predict(df_val[patient_cols]).astype(float).flatten()

        X_tr = df_train[patient_cols].copy(); X_tr["_pd_system"] = pred_tr
        X_va = df_val[patient_cols].copy();   X_va["_pd_system"] = pred_va

        print("[HierarchicalA] Step 2 (chain) — CONT_RX regressors with PD-system feature ...")
        for col in CONT_RX:
            m = CatBoostRegressor(
                iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
                depth=CATBOOST_DEPTH, loss_function="RMSE",
                random_seed=SEED, verbose=0,
            )
            m.fit(X_tr, df_train[col],
                  eval_set=(X_va, df_val[col]),
                  early_stopping_rounds=CATBOOST_EARLY_STOP)
            self.cont_models[col] = m
            print(f"  {col}: best_iter={m.best_iteration_}")

    def _fit_stratified(self, df_train, df_val, patient_cols: list) -> None:
        from catboost import CatBoostRegressor

        # Global fallback used for rare classes
        print("[HierarchicalA] Step 2 (stratified) — global fallback models ...")
        X_tr = df_train[patient_cols]
        X_va = df_val[patient_cols]
        for col in CONT_RX:
            m = CatBoostRegressor(
                iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
                depth=CATBOOST_DEPTH, loss_function="RMSE",
                random_seed=SEED, verbose=0,
            )
            m.fit(X_tr, df_train[col],
                  eval_set=(X_va, df_val[col]),
                  early_stopping_rounds=CATBOOST_EARLY_STOP)
            self._fallback[col] = m

        # Per-class models
        for cls in self.pd_classes:
            mask_tr = df_train[CAT_RX].astype(int) == cls
            mask_va = df_val[CAT_RX].astype(int) == cls
            n_tr, n_va = int(mask_tr.sum()), int(mask_va.sum())
            print(f"[HierarchicalA] Class {cls}: train={n_tr}  val={n_va}")

            if n_tr < MIN_CLASS_SAMPLES:
                print(f"  → < {MIN_CLASS_SAMPLES} samples; will use global fallback at inference.")
                self.cont_models[cls] = None
                continue

            self.cont_models[cls] = {}
            X_tr_c = df_train.loc[mask_tr, patient_cols]
            X_va_c = df_val.loc[mask_va, patient_cols] if n_va >= 5 else None

            for col in CONT_RX:
                m = CatBoostRegressor(
                    iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR,
                    depth=CATBOOST_DEPTH, loss_function="RMSE",
                    random_seed=SEED, verbose=0,
                )
                if X_va_c is not None:
                    m.fit(X_tr_c, df_train.loc[mask_tr, col],
                          eval_set=(X_va_c, df_val.loc[mask_va, col]),
                          early_stopping_rounds=CATBOOST_EARLY_STOP)
                else:
                    m.fit(X_tr_c, df_train.loc[mask_tr, col])
                self.cont_models[cls][col] = m
            print(f"  → Trained {len(CONT_RX)} regressors for class {cls}.")
