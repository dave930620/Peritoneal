"""
f1_baselines.py — All F1 baseline models (B1–B8) behind a common interface.

Each baseline is wrapped so it exposes:
    .fit(X_train, y_train)
    .predict(X_test)  →  np.ndarray shape (N,)

Baseline map
------------
B1  Ridge Regression      (sklearn)
B2  Random Forest         (sklearn)
B3  XGBoost               (xgboost)
B4  CatBoost              (catboost)
B5  Vanilla MLP           (sklearn MLPRegressor)
B6  TabNet                (pytorch_tabnet)
B7  FT-Transformer        (minimal PyTorch implementation, no extra dep)
B8  SAINT (ours)          → loaded externally in run_baselines.py; not here
"""

import warnings
from typing import Optional

import numpy as np


# ── Common interface ──────────────────────────────────────────────────────────

class _BaselineWrapper:
    name: str = "baseline"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineWrapper":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ── B1: Ridge Regression ─────────────────────────────────────────────────────

class RidgeBaseline(_BaselineWrapper):
    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        self._model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self._model.fit(X, y); return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X).flatten()


# ── B2: Random Forest ─────────────────────────────────────────────────────────

class RandomForestBaseline(_BaselineWrapper):
    name = "random_forest"

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        from sklearn.ensemble import RandomForestRegressor
        self._model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state,
            n_jobs=-1, min_samples_leaf=2,
        )

    def fit(self, X, y):
        self._model.fit(X, y); return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X).flatten()


# ── B3: XGBoost ───────────────────────────────────────────────────────────────

class XGBoostBaseline(_BaselineWrapper):
    name = "xgboost"

    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05,
                 max_depth: int = 6, random_state: int = 42):
        try:
            import xgboost as xgb
            import torch
        except ImportError:
            raise ImportError("pip install xgboost")
        use_cuda = torch.cuda.is_available()
        self._model = xgb.XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state,
            n_jobs=-1, verbosity=0,
            device="cuda" if use_cuda else "cpu",
        )
        if use_cuda:
            print("[XGBoost] Using GPU (CUDA)")

    def fit(self, X, y):
        self._model.fit(X, y); return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X).flatten()


# ── B4: CatBoost ──────────────────────────────────────────────────────────────

class CatBoostBaseline(_BaselineWrapper):
    name = "catboost"

    def __init__(self, iterations: int = 500, learning_rate: float = 0.05,
                 depth: int = 6, random_seed: int = 42):
        try:
            from catboost import CatBoostRegressor
            import torch
        except ImportError:
            raise ImportError("pip install catboost")
        use_cuda = torch.cuda.is_available()
        self._model = CatBoostRegressor(
            iterations=iterations, learning_rate=learning_rate,
            depth=depth, random_seed=random_seed, verbose=0,
            task_type="GPU" if use_cuda else "CPU",
        )
        if use_cuda:
            print("[CatBoost] Using GPU (CUDA)")

    def fit(self, X, y):
        self._model.fit(X, y); return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X).flatten()


# ── B5: Vanilla MLP (sklearn) ─────────────────────────────────────────────────

class VanillaMLPBaseline(_BaselineWrapper):
    name = "vanilla_mlp"

    def __init__(self, hidden_layer_sizes=(128, 128, 64), max_iter: int = 500,
                 random_state: int = 42):
        from sklearn.neural_network import MLPRegressor
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu", solver="adam",
            max_iter=max_iter, random_state=random_state,
            early_stopping=True, validation_fraction=0.1,
        )

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X).flatten()


# ── B6: TabNet ────────────────────────────────────────────────────────────────

class TabNetBaseline(_BaselineWrapper):
    name = "tabnet"

    def __init__(self, n_d: int = 32, n_a: int = 32, n_steps: int = 3,
                 max_epochs: int = 200, patience: int = 20, seed: int = 42):
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError:
            raise ImportError("pip install pytorch-tabnet")
        import torch
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print("[TabNet] Using GPU (CUDA)")
        self._cls   = TabNetRegressor
        self._kw    = dict(n_d=n_d, n_a=n_a, n_steps=n_steps,
                          seed=seed, verbose=0, device_name=device_name)
        self._fit_kw = dict(max_epochs=max_epochs, patience=patience,
                            batch_size=256, virtual_batch_size=128)
        self._model  = None

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetRegressor
        self._model = TabNetRegressor(**self._kw)
        self._model.fit(
            X.astype(np.float32), y.reshape(-1, 1).astype(np.float32),
            **self._fit_kw,
        )
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X.astype(np.float32)).flatten()


# ── B7: FT-Transformer (minimal, no extra dep) ────────────────────────────────

class _FTTransformerNet:
    """Thin PyTorch wrapper (trained internally)."""

    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 3, dropout: float = 0.1, device: str = "cpu"):
        import torch
        import torch.nn as nn

        self.device = torch.device(device)

        # Feature tokenizer: each feature → d_model via a shared linear bank
        # Implementation: one Linear(1, d_model) per feature
        self.tokenizer = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_features)]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.net = nn.ModuleDict({
            "tokenizer":   self.tokenizer,
            "transformer": self.transformer,
            "head":        self.head,
        })
        # Combine into one nn.Module for optimizer
        self._all = nn.ModuleList([*self.tokenizer, self.transformer, self.head])
        self._cls = self.cls_token

    def parameters(self):
        import itertools
        return itertools.chain(self._all.parameters(), [self.cls_token])

    def to(self, device):
        self._all.to(device)
        self.cls_token.data = self.cls_token.data.to(device)
        self.device = device
        return self

    def train(self):
        self._all.train(); return self

    def eval(self):
        self._all.eval(); return self

    def __call__(self, x):
        import torch
        B = x.shape[0]
        # Tokenize each feature separately: (B, n_features, d_model)
        tokens = torch.stack([
            lin(x[:, i:i+1]) for i, lin in enumerate(self.tokenizer)
        ], dim=1)
        # Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_features+1, d_model)
        out = self.transformer(tokens)             # (B, n_features+1, d_model)
        return self.head(out[:, 0, :])             # use [CLS] output → (B, 1)


class FTTransformerBaseline(_BaselineWrapper):
    name = "ft_transformer"

    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 dropout: float = 0.1, lr: float = 1e-3, epochs: int = 200,
                 batch_size: int = 256, patience: int = 20, seed: int = 42):
        self._kw = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        dropout=dropout)
        self._train_kw = dict(lr=lr, epochs=epochs,
                              batch_size=batch_size, patience=patience)
        self._seed   = seed
        self._net    = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from src.utils.device import get_device

        torch.manual_seed(self._seed)
        device = get_device()

        n_features = X.shape[1]
        self._net  = _FTTransformerNet(n_features, **self._kw)
        self._net.to(device)

        lr        = self._train_kw["lr"]
        epochs    = self._train_kw["epochs"]
        batch_sz  = self._train_kw["batch_size"]
        patience  = self._train_kw["patience"]

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # 90/10 val split
        n_val = max(1, int(len(Xt) * 0.1))
        idx   = torch.randperm(len(Xt))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        tr_ds = TensorDataset(Xt[tr_idx], yt[tr_idx])
        va_ds = TensorDataset(Xt[val_idx], yt[val_idx])
        tr_dl = DataLoader(tr_ds, batch_size=batch_sz, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=batch_sz)

        opt      = optim.AdamW(self._net.parameters(), lr=lr, weight_decay=1e-4)
        crit     = nn.MSELoss()
        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            self._net.train()
            for bx, by in tr_dl:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = crit(self._net(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net._all.parameters(), 1.0)
                opt.step()

            self._net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in va_dl:
                    bx, by = bx.to(device), by.to(device)
                    val_loss += crit(self._net(bx), by).item() * bx.size(0)
            val_loss /= n_val

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                self._best_state = {k: v.clone()
                                    for k, v in self._net._all.state_dict().items()}
                self._best_cls   = self._net.cls_token.data.clone()
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Restore best
        self._net._all.load_state_dict(self._best_state)
        self._net.cls_token.data = self._best_cls
        self._device = device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self._net.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self._device)
            return self._net(Xt).squeeze(1).cpu().numpy()


# ── Factory function ──────────────────────────────────────────────────────────

_REGISTRY = {
    "ridge":          RidgeBaseline,
    "random_forest":  RandomForestBaseline,
    "xgboost":        XGBoostBaseline,
    "catboost":       CatBoostBaseline,
    "vanilla_mlp":    VanillaMLPBaseline,
    "tabnet":         TabNetBaseline,
    "ft_transformer": FTTransformerBaseline,
}

BASELINE_ORDER = [
    "ridge", "random_forest", "xgboost", "catboost",
    "vanilla_mlp", "tabnet", "ft_transformer",
]


def get_f1_baseline(name: str, **kwargs) -> _BaselineWrapper:
    """Return an unfitted baseline by name.

    Valid names: ridge, random_forest, xgboost, catboost,
                 vanilla_mlp, tabnet, ft_transformer
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Valid: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
