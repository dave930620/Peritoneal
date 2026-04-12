"""
f1_core.py — Reusable F1 training loop used by ablation experiments.

Accepts any nn.Module that has the same forward signature as SAINT.
Returns a metrics dict after training.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.metrics import full_f1_metrics


def train_f1_model(
    model:          nn.Module,
    X_train:        np.ndarray,
    y_train:        np.ndarray,
    X_test:         np.ndarray,
    y_test:         np.ndarray,
    device:         torch.device,
    epochs:         int   = 200,
    batch_size:     int   = 32,
    lr:             float = 1e-3,
    weight_decay:   float = 0.01,
    use_scheduler:  bool  = True,
    threshold:      float = 1.7,
    verbose:        bool  = True,
) -> dict:
    """Train model on (X_train, y_train) and return full_f1_metrics on test set.

    Parameters
    ----------
    use_scheduler : if False → ablation A-F1-9
    weight_decay  : set to 0.0 → ablation A-F1-10
    """
    model = model.to(device)
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    Xte = torch.tensor(X_test,  dtype=torch.float32).to(device)
    yte = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1).to(device)

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay, amsgrad=True)
    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, mode="min", factor=0.5, patience=10)
                 if use_scheduler else None)

    ds  = TensorDataset(Xtr, ytr)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in dl:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(bx), by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

        if scheduler is not None or verbose:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xte), yte).item()
            if scheduler is not None:
                scheduler.step(val_loss)
            if verbose and epoch % 20 == 0:
                print(f"  epoch {epoch:3d}/{epochs}  val_mse={val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(Xte).cpu().numpy()
    return full_f1_metrics(y_test, preds, threshold)
