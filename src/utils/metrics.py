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
    """Return flat dict of all F2 metrics needed for Table 2.

    Parameters
    ----------
    ktv_doctor : F1's predicted Kt/V under doctor's recorded prescription.
    ktv_model  : F1's predicted Kt/V under model's generated prescription.
    ktv_actual : Ground-truth Kt/V from the dataset.
    rx_doctor  : dict {col_name: np.ndarray} for doctor's Rx variables.
    rx_model   : dict {col_name: np.ndarray} for model's Rx variables.
    threshold  : Clinical adequacy threshold (default 1.7).
    """
    ktv_doctor = ktv_doctor.flatten()
    ktv_model  = ktv_model.flatten()
    ktv_actual = ktv_actual.flatten()

    fail_mask = ktv_actual < threshold   # patients doctor FAILS clinically
    pass_mask = ktv_actual >= threshold

    n_fail = int(fail_mask.sum())
    n_pass = int(pass_mask.sum())

    metrics = {"n_fail": n_fail, "n_pass": n_pass, "threshold": threshold}

    # FAIL group
    if n_fail > 0:
        delta_fail = float(np.mean(ktv_model[fail_mask] - ktv_doctor[fail_mask]))
        p_pass     = float(np.mean(ktv_model[fail_mask] >= threshold))
        metrics["delta_ktv_fail"] = delta_fail
        metrics["p_pass"]         = p_pass
    else:
        metrics["delta_ktv_fail"] = float("nan")
        metrics["p_pass"]         = float("nan")

    # PASS group
    if n_pass > 0:
        delta_pass = float(np.mean(ktv_model[pass_mask] - ktv_doctor[pass_mask]))
        p_fail     = float(np.mean(ktv_model[pass_mask] < threshold))
        metrics["delta_ktv_pass"] = delta_pass
        metrics["p_fail"]         = p_fail
    else:
        metrics["delta_ktv_pass"] = float("nan")
        metrics["p_fail"]         = float("nan")

    # Overall delta
    metrics["delta_ktv_all"] = float(np.mean(ktv_model - ktv_doctor))

    # Per-variable Rx similarity
    for col in rx_doctor:
        if col in rx_model:
            r = pearson_correlation(
                np.array(rx_doctor[col], dtype=float),
                np.array(rx_model[col],  dtype=float),
            )
            mae_rx = float(np.mean(np.abs(
                np.array(rx_doctor[col], dtype=float) -
                np.array(rx_model[col],  dtype=float)
            )))
            safe_col = col.replace(" ", "_").replace("/", "_")
            metrics[f"pearson_{safe_col}"] = r
            metrics[f"mae_rx_{safe_col}"]  = mae_rx

    return metrics
