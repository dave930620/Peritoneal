"""
train_f1.py — F1 SAINT model training entry point.

Usage
-----
    python train_f1.py

What this does
--------------
1. Loads final_nighttime_data.csv.
2. Validates that required columns are present.
3. Preprocesses features (excludes Volume (L) and other listed columns).
4. Splits data patient-wise into train / test.
5. Trains the SAINT model to predict PD Kt/V.
6. Evaluates on test set and prints metrics.
7. Saves: saint_pd_model.pth, feature_info.pkl, feature_importance.csv.

Outputs (saved to working directory)
-------------------------------------
- saint_pd_model.pth      : model weights
- feature_info.pkl        : preprocessing config (required by train_f2.py)
- feature_importance.csv  : permutation importance scores
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CLINICAL_THRESHOLD,
    DATA_CSV,
    EXCLUDE_COLUMNS,
    F1_BATCH_SIZE,
    F1_DROPOUT,
    F1_EPOCHS,
    F1_HIDDEN_SIZE,
    F1_LR,
    F1_NUM_HEADS,
    F1_NUM_LAYERS,
    F1_OUTPUT_SIZE,
    F1_WEIGHT_DECAY,
    FEATURE_INFO_PATH,
    F1_MODEL_PATH,
    OUTCOME_COL,
    PATIENT_ID_COL,
)
from src.data.preprocessor import (
    preprocess_data,
    sanity_check_data,
    validate_columns,
)
from src.models.saint import SAINT
from src.utils.device import get_dataloader_kwargs, get_device, is_amp_supported, print_device_info
from src.utils.metrics import print_f1_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_train_test(features, target, patient_ids):
    """Patient-wise 80/20 train/test split using GroupShuffleSplit."""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(features, target, groups=patient_ids))
    return train_idx, test_idx


def visualize_results(ground_truth: np.ndarray, predictions: np.ndarray,
                      threshold: float = 1.7, save_path: str = "f1_result.png") -> None:
    """Save 4-panel diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    gt, pr = ground_truth.flatten(), predictions.flatten()

    # Scatter: prediction vs ground truth
    ax = axes[0, 0]
    ax.scatter(gt, pr, alpha=0.4, s=8)
    lims = [min(gt.min(), pr.min()), max(gt.max(), pr.max())]
    ax.plot(lims, lims, "r--", lw=1.5)
    ax.set_xlabel("Ground Truth"); ax.set_ylabel("Prediction")
    ax.set_title("Prediction vs Ground Truth")

    # Residuals
    residuals = pr - gt
    ax = axes[0, 1]
    ax.scatter(gt, residuals, alpha=0.4, s=8)
    ax.axhline(0, color="r", linestyle="--", lw=1.5)
    ax.set_xlabel("Ground Truth"); ax.set_ylabel("Residual")
    ax.set_title("Residual Distribution")

    # Residual histogram
    ax = axes[1, 0]
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel("Residual"); ax.set_title("Residual Histogram")

    # Threshold classification
    colors = []
    for g, p in zip(gt, pr):
        if g > threshold and p > threshold:
            colors.append("green")
        elif g <= threshold and p <= threshold:
            colors.append("blue")
        elif g > threshold and p <= threshold:
            colors.append("red")
        else:
            colors.append("orange")

    ax = axes[1, 1]
    ax.scatter(gt, pr, alpha=0.4, s=8, c=colors)
    ax.axhline(threshold, color="r", linestyle="--", lw=1)
    ax.axvline(threshold, color="r", linestyle="--", lw=1)
    ax.set_xlabel("Ground Truth"); ax.set_ylabel("Prediction")
    ax.set_title(f"Threshold Classification (threshold={threshold})")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green",  markersize=8, label="TP (>1.7)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",   markersize=8, label="TN (≤1.7)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",    markersize=8, label="FN"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=8, label="FP"),
    ]
    ax.legend(handles=legend, loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"[F1] Result plot saved to: {save_path}")


def visualize_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                                threshold: float = 1.7,
                                save_path: str = "f1_confusion_matrix.png") -> None:
    gt_pos = y_true.flatten() > threshold
    pr_pos = y_pred.flatten() > threshold
    tn = int(np.sum(~gt_pos & ~pr_pos)); fp = int(np.sum(~gt_pos & pr_pos))
    fn = int(np.sum( gt_pos & ~pr_pos)); tp = int(np.sum( gt_pos &  pr_pos))
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cbar=False,
        xticklabels=[f"Pred ≤{threshold}", f"Pred >{threshold}"],
        yticklabels=[f"GT ≤{threshold}", f"GT >{threshold}"],
    )
    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.xlabel("Prediction"); plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"[F1] Confusion matrix saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load data
    print(f"[F1] Loading data from: {DATA_CSV}")
    data = pd.read_csv(DATA_CSV)
    print(f"[F1] Loaded: {data.shape}")

    # 2. Validate required columns
    validate_columns(data, [OUTCOME_COL, PATIENT_ID_COL], label="train_f1")
    sanity_check_data(data)

    # Confirm Volume (L) will be excluded
    if "Volume (L)" in data.columns:
        print("[F1] 'Volume (L)' found in data — will be EXCLUDED (unreliable column).")

    # 3. Preprocess
    (features, target, patient_ids,
     disc_idx, cont_idx, feature_names, feature_info) = preprocess_data(
        data, feature_info_path=FEATURE_INFO_PATH
    )
    print(f"[F1] Total features: {len(feature_names)}")

    # 4. Train/test split (patient-wise)
    device = get_device()
    print_device_info(device)
    dl_kwargs = get_dataloader_kwargs(device)
    use_amp   = is_amp_supported(device)

    if patient_ids is not None:
        train_idx, test_idx = split_train_test(features, target, patient_ids)
        X_train = features.iloc[train_idx].values
        X_test  = features.iloc[test_idx].values
        y_train = target.iloc[train_idx].values
        y_test  = target.iloc[test_idx].values
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features.values, target.values, test_size=0.2, random_state=42
        )

    # 5. Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1).to(device)
    print(f"[F1] Train: {X_train.shape}  Test: {X_test.shape}")

    # 6. Build model
    model = SAINT(
        input_size=X_train.shape[1],
        hidden_size=F1_HIDDEN_SIZE,
        output_size=F1_OUTPUT_SIZE,
        discrete_feature_indices=disc_idx,
        continuous_feature_indices=cont_idx,
        num_heads=F1_NUM_HEADS,
        num_layers=F1_NUM_LAYERS,
        dropout=F1_DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=F1_LR,
                            weight_decay=F1_WEIGHT_DECAY, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    # AMP scaler (no-op on CPU/MPS)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 7. Quick forward pass check
    print("[F1] Testing forward pass...")
    with torch.no_grad():
        _ = model(X_train[:F1_BATCH_SIZE])
    print("[F1] Forward pass OK.")

    # 8. Training loop
    train_dataset = TensorDataset(X_train, y_train)
    train_loader  = DataLoader(train_dataset, batch_size=F1_BATCH_SIZE, shuffle=True)

    train_losses, test_losses = [], []

    for epoch in range(1, F1_EPOCHS + 1):
        model.train()
        epoch_loss, n = 0.0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out  = model(bx)
                loss = criterion(out, by)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            epoch_loss += loss.item() * bx.size(0)
            n += bx.size(0)

        train_loss = epoch_loss / n
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_out  = model(X_test)
            test_loss = criterion(test_out, y_test).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)

        if epoch % 10 == 0:
            print(f"[F1] Epoch {epoch:3d}/{F1_EPOCHS}  "
                  f"train={train_loss:.4f}  test={test_loss:.4f}")

    # 9. Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
    gt = y_test.cpu().numpy()

    print_f1_metrics(gt, preds, CLINICAL_THRESHOLD)

    # 10. Visualize
    visualize_results(gt, preds, save_path="f1_result.png")
    visualize_confusion_matrix(gt, preds, save_path="f1_confusion_matrix.png")

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("F1 Training Loss Curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("f1_loss_curve.png", dpi=160)
    plt.close()
    print("[F1] Loss curve saved to: f1_loss_curve.png")

    # 11. Feature importance
    try:
        print("[F1] Computing permutation feature importance (may take a moment)...")
        importances = model.get_feature_importance(X_train, feature_names)
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False)
        imp_df.to_csv("feature_importance.csv", index=False)
        print("[F1] Feature importance saved to: feature_importance.csv")
        print(imp_df.head(15).to_string(index=False))
    except Exception as e:
        print(f"[F1] Feature importance failed: {e}")

    # 12. Save model
    torch.save(model.state_dict(), F1_MODEL_PATH)
    print(f"[F1] Model saved to: {F1_MODEL_PATH}")


if __name__ == "__main__":
    main()
