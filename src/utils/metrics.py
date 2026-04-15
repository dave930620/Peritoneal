"""
Evaluation metrics for F1 (outcome prediction) and F2 (prescription evaluation).
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, f1_score as sk_f1_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics.

    Returns a dict with keys: mse, mae, r2.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2":  float(r2_score(y_true, y_pred)),
    }


def threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      threshold: float = 1.7) -> dict:
    """Compute classification metrics using a clinical threshold.

    Treats 'above threshold' as positive.

    Returns a dict with keys: accuracy, tp, tn, fp, fn.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    gt_pos  = y_true > threshold
    pr_pos  = y_pred > threshold

    tp = int(np.sum( gt_pos &  pr_pos))
    tn = int(np.sum(~gt_pos & ~pr_pos))
    fp = int(np.sum(~gt_pos &  pr_pos))
    fn = int(np.sum( gt_pos & ~pr_pos))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else float("nan")

    return {
        "accuracy": accuracy,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_pos": int(gt_pos.sum()),
        "n_neg": int((~gt_pos).sum()),
        "threshold": threshold,
    }


def print_f1_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     threshold: float = 1.7) -> None:
    """Print a full F1 evaluation summary to stdout."""
    reg = regression_metrics(y_true, y_pred)
    cls = threshold_metrics(y_true, y_pred, threshold)

    print("\n=== F1 Model Evaluation ===")
    print(f"  MSE : {reg['mse']:.4f}")
    print(f"  MAE : {reg['mae']:.4f}")
    print(f"  R²  : {reg['r2']:.4f}")
    print(f"\n  Threshold-based classification (threshold = {threshold})")
    print(f"  Accuracy : {cls['accuracy']*100:.2f}%  ({cls['tp']+cls['tn']}/{len(y_true.flatten())})")
    print(f"  TP={cls['tp']}  TN={cls['tn']}  FP={cls['fp']}  FN={cls['fn']}")

    if cls["n_pos"] > 0:
        sens = cls["tp"] / cls["n_pos"]
        print(f"  Sensitivity (recall for >1.7) : {sens:.4f}")
    if cls["n_neg"] > 0:
        spec = cls["tn"] / cls["n_neg"]
        print(f"  Specificity (recall for ≤1.7) : {spec:.4f}")


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays. Returns nan if degenerate."""
    a, b = a.flatten(), b.flatten()
    if len(a) < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Comprehensive F1 metric bundle (used by experiments)
# ---------------------------------------------------------------------------

def full_f1_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    threshold: float = 1.7) -> dict:
    """Return a flat dict with all F1 metrics needed for Table 1.

    Regression : mae, rmse, r2, pearson
    Classification (at threshold) : auroc, auprc, accuracy,
                                    sensitivity, specificity, f1_cls
    """
    y_true = y_true.flatten().astype(float)
    y_pred = y_pred.flatten().astype(float)

    # --- Regression ---
    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = float(r2_score(y_true, y_pred))
    pearson = pearson_correlation(y_true, y_pred)

    # --- Binary labels at threshold ---
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    n_pos = int(y_true_bin.sum())
    n_neg = int((1 - y_true_bin).sum())

    accuracy    = (tp + tn) / len(y_true) if len(y_true) > 0 else float("nan")
    sensitivity = tp / n_pos if n_pos > 0 else float("nan")
    specificity = tn / n_neg if n_neg > 0 else float("nan")
    f1_cls      = float(sk_f1_score(y_true_bin, y_pred_bin, zero_division=0))

    # AUROC / AUPRC need continuous scores
    try:
        auroc = float(roc_auc_score(y_true_bin, y_pred))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true_bin, y_pred))
    except Exception:
        auprc = float("nan")

    return {
        "mae":         mae,
        "rmse":        rmse,
        "r2":          r2,
        "pearson":     pearson,
        "auroc":       auroc,
        "auprc":       auprc,
        "accuracy":    accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_cls":      f1_cls,
    }


# ---------------------------------------------------------------------------
# F2 metric bundle
# ---------------------------------------------------------------------------

def f2_metrics(
    ktv_doctor: np.ndarray,
    ktv_model:  np.ndarray,
    ktv_actual: np.ndarray,
    rx_doctor:  dict,
    rx_model:   dict,
    threshold:  float = 1.7,
) -> dict:
    """Return flat dict of all F2 metrics.

    Parameters
    ----------
    ktv_doctor : F1's predicted Kt/V under doctor's recorded prescription.
    ktv_model  : F1's predicted Kt/V under model's generated prescription.
    ktv_actual : Ground-truth Kt/V from the dataset (for reference counts).
    rx_doctor  : dict {col_name: np.ndarray} for doctor's Rx variables.
    rx_model   : dict {col_name: np.ndarray} for model's Rx variables.
    threshold  : Clinical adequacy threshold (default 1.7).

    Metric groups
    -------------
    Split mask: ktv_actual (ground truth) < threshold → "actual fail" group.
    This is more reliable than using F1's prediction of the doctor's Rx.

    Kt/V improvement:
        delta_ktv_doc_fail  avg(model_predicted - actual) for actual-fail patients  ↑ high
        delta_ktv_doc_pass  avg(model_predicted - actual) for actual-pass patients  ≈ 0
        delta_ktv_all       avg(model_predicted - actual) across all patients

    Clinical threshold crossing:
        p_rescue  P(model >= thr | actual Kt/V < thr)   ↑ want high
        p_harm    P(model <  thr | actual Kt/V >= thr)  ↓ want low (safety)

    Prescription similarity:
        pearson_{col}            overall Pearson per Rx column
        pearson_{col}_doc_fail   Pearson for actual-fail patients (expect lower)
        pearson_{col}_doc_pass   Pearson for actual-pass patients (expect higher)
        avg_pearson              mean over columns, overall
        avg_pearson_doc_fail     mean over columns, fail group
        avg_pearson_doc_pass     mean over columns, pass group
        mae_rx_{col}             MAE between model and doctor Rx per column

    Counts:
        n_doc_fail / n_doc_pass  patient counts per group
    """
    ktv_doctor = ktv_doctor.flatten().astype(float)
    ktv_model  = ktv_model.flatten().astype(float)
    ktv_actual = ktv_actual.flatten().astype(float)

    # ── Split by ground-truth Kt/V ────────────────────────────────────────────
    doc_fail   = ktv_actual < threshold   # patients who actually failed
    doc_pass   = ~doc_fail
    n_doc_fail = int(doc_fail.sum())
    n_doc_pass = int(doc_pass.sum())

    metrics = {
        "threshold":  threshold,
        "n_doc_fail": n_doc_fail,
        "n_doc_pass": n_doc_pass,
    }

    # ── Kt/V improvement (model predicted vs actual ground truth) ─────────────
    metrics["delta_ktv_all"] = float(np.mean(ktv_model - ktv_actual))

    if n_doc_fail > 0:
        metrics["delta_ktv_doc_fail"] = float(np.mean(ktv_model[doc_fail] - ktv_actual[doc_fail]))
        metrics["p_rescue"]           = float(np.mean(ktv_model[doc_fail] >= threshold))
    else:
        metrics["delta_ktv_doc_fail"] = float("nan")
        metrics["p_rescue"]           = float("nan")

    if n_doc_pass > 0:
        metrics["delta_ktv_doc_pass"] = float(np.mean(ktv_model[doc_pass] - ktv_actual[doc_pass]))
        metrics["p_harm"]             = float(np.mean(ktv_model[doc_pass] < threshold))
    else:
        metrics["delta_ktv_doc_pass"] = float("nan")
        metrics["p_harm"]             = float("nan")

    # Backwards-compat aliases
    metrics["delta_ktv_fail"] = metrics["delta_ktv_doc_fail"]
    metrics["p_pass"]         = metrics["p_rescue"]
    metrics["delta_ktv_pass"] = metrics["delta_ktv_doc_pass"]
    metrics["p_fail"]         = metrics["p_harm"]

    # ── Prescription similarity ───────────────────────────────────────────────
    pearsons_all, pearsons_fail, pearsons_pass = [], [], []

    for col in rx_doctor:
        if col not in rx_model:
            continue
        doc_rx   = np.array(rx_doctor[col], dtype=float)
        mod_rx   = np.array(rx_model[col],  dtype=float)
        safe_col = col.replace(" ", "_").replace("/", "_")

        r_all = pearson_correlation(doc_rx, mod_rx)
        metrics[f"pearson_{safe_col}"] = r_all
        metrics[f"mae_rx_{safe_col}"]  = float(np.mean(np.abs(doc_rx - mod_rx)))
        pearsons_all.append(r_all)

        # Actual-fail group: model explores more → lower Pearson expected
        if n_doc_fail > 1:
            r_fail = pearson_correlation(doc_rx[doc_fail], mod_rx[doc_fail])
            metrics[f"pearson_{safe_col}_doc_fail"] = r_fail
            pearsons_fail.append(r_fail)

        # Actual-pass group: model stays close → higher Pearson expected
        if n_doc_pass > 1:
            r_pass = pearson_correlation(doc_rx[doc_pass], mod_rx[doc_pass])
            metrics[f"pearson_{safe_col}_doc_pass"] = r_pass
            pearsons_pass.append(r_pass)

    metrics["avg_pearson"]          = float(np.nanmean(pearsons_all))  if pearsons_all  else float("nan")
    metrics["avg_pearson_doc_fail"] = float(np.nanmean(pearsons_fail)) if pearsons_fail else float("nan")
    metrics["avg_pearson_doc_pass"] = float(np.nanmean(pearsons_pass)) if pearsons_pass else float("nan")

    return metrics
