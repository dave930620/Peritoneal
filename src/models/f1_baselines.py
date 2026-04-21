"""
f1_baselines.py — All F1 baseline models (B1–B9) behind a common interface.

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
B8  TabM                  (Gorishniy et al., 2024 — batch-ensemble MLP)
B9  TabPFN                (Hollmann et al., 2022/2025 — requires tabpfn>=2.0)
B10 SAINT (ours)          → loaded externally in run_baselines.py; not here
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


# ── B8: TabM (Gorishniy et al., 2024) ────────────────────────────────────────
#
# "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"
# Core idea: K parallel MLPs sharing linear weights but with per-member
# learnable multiplicative scaling vectors at every layer input (batch ensemble).
# Prediction = mean of K members' outputs.

class _TabMNet:
    """PyTorch TabM network (constructed lazily inside fit)."""

    def __init__(self, n_features: int, d_model: int, n_layers: int,
                 dropout: float, n_members: int, device):
        import torch
        import torch.nn as nn

        self.device    = torch.device(device)
        self.n_members = n_members

        # Layer dimensions: input → [d_model] * n_layers
        dims = [n_features] + [d_model] * n_layers

        # Shared linear weights for each layer
        self.linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)
        ])
        # Batch-norm over the (B * K) dimension for stable training
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(dims[i + 1]) for i in range(n_layers)
        ])
        self.dropout_p = dropout

        # Per-member multiplicative scaling at each layer's input: (K, d_in)
        # Initialized to 1 so the model starts as a standard MLP ensemble
        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(n_members, dims[i]))
            for i in range(n_layers)
        ])

        # Separate output head per member to allow diverse ensemble predictions
        self.heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_members)
        ])

        self._all = nn.ModuleList([self.linears, self.norms, self.scales, self.heads])
        self._all.to(self.device)

    def parameters(self):
        return self._all.parameters()

    def train(self):
        self._all.train(); return self

    def eval(self):
        self._all.eval(); return self

    def __call__(self, x):
        import torch
        import torch.nn.functional as F

        B = x.shape[0]
        # Expand to K members: (B, K, d_in)
        h = x.unsqueeze(1).expand(-1, self.n_members, -1)

        for i, (lin, norm, scale) in enumerate(
            zip(self.linears, self.norms, self.scales)
        ):
            # Per-member scaling of layer input
            h = h * scale.unsqueeze(0)          # (B, K, d_in)
            h = h.reshape(B * self.n_members, -1)
            h = lin(h)                           # (B*K, d_out)
            h = norm(h)
            h = F.gelu(h)
            if i < len(self.linears) - 1:
                h = F.dropout(h, p=self.dropout_p, training=self._all.training)
            h = h.reshape(B, self.n_members, -1) # (B, K, d_out)

        # Per-member predictions then average
        preds = torch.stack(
            [self.heads[k](h[:, k, :]).squeeze(1) for k in range(self.n_members)],
            dim=1,
        )  # (B, K)
        return preds.mean(dim=1)  # (B,)

    def to(self, device):
        self._all.to(device)
        self.device = torch.device(device)
        return self

    def state_dict(self):
        return self._all.state_dict()

    def load_state_dict(self, sd):
        self._all.load_state_dict(sd)


class TabMBaseline(_BaselineWrapper):
    """TabM: batch-ensemble MLP with per-member multiplicative layer scalings.

    Reference: Gorishniy et al., "TabM: Advancing Tabular Deep Learning with
    Parameter-Efficient Ensembling," 2024.
    """
    name = "tabm"

    def __init__(self, d_model: int = 256, n_layers: int = 3, dropout: float = 0.1,
                 n_members: int = 16, lr: float = 1e-3, epochs: int = 200,
                 batch_size: int = 256, patience: int = 20, seed: int = 42):
        self._kw = dict(d_model=d_model, n_layers=n_layers,
                        dropout=dropout, n_members=n_members)
        self._train_kw = dict(lr=lr, epochs=epochs,
                              batch_size=batch_size, patience=patience)
        self._seed = seed
        self._net  = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from src.utils.device import get_device

        torch.manual_seed(self._seed)
        device = str(get_device())

        n_features = X.shape[1]
        self._net  = _TabMNet(n_features, device=device, **self._kw)

        lr       = self._train_kw["lr"]
        epochs   = self._train_kw["epochs"]
        batch_sz = self._train_kw["batch_size"]
        patience = self._train_kw["patience"]

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)

        n_val = max(1, int(len(Xt) * 0.1))
        idx   = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(self._seed))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        tr_dl = DataLoader(TensorDataset(Xt[tr_idx], yt[tr_idx]),
                           batch_size=batch_sz, shuffle=True)
        va_dl = DataLoader(TensorDataset(Xt[val_idx], yt[val_idx]),
                           batch_size=batch_sz)

        opt      = optim.AdamW(self._net.parameters(), lr=lr, weight_decay=1e-4)
        crit     = nn.MSELoss()
        best_val = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self._net.train()
            for bx, by in tr_dl:
                bx, by = bx.to(self._net.device), by.to(self._net.device)
                opt.zero_grad()
                crit(self._net(bx), by).backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()

            self._net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in va_dl:
                    bx, by = bx.to(self._net.device), by.to(self._net.device)
                    val_loss += crit(self._net(bx), by).item() * bx.size(0)
            val_loss /= n_val

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self._net.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self._net.device)
            return self._net(Xt).cpu().numpy()


# ── B9: TabPFN (Hollmann et al., 2022 / v2 2025) ─────────────────────────────
#
# Helper: read API key from the config.json that setup_tabpfn_key.py wrote.

def _read_tabpfn_key_from_disk() -> str:
    """Return the TabPFN API key from disk, or '' if not found."""
    import os, json
    candidates = []
    try:
        import platformdirs
        candidates.append(
            os.path.join(platformdirs.user_data_dir("tabpfn", "priorlabs"), "config.json")
        )
    except Exception:
        pass
    candidates += [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "priorlabs", "tabpfn", "config.json"),
        os.path.join(os.environ.get("APPDATA", ""),      "priorlabs", "tabpfn", "config.json"),
        os.path.join(os.path.expanduser("~"), ".tabpfn", "config.json"),
    ]
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            key = data.get("api_key", "").strip()
            if key:
                return key
        except Exception:
            continue
    return ""
#
# TabPFN is a Prior-Data Fitted Network trained once on synthetic data and then
# applied to new tasks via in-context learning — no per-dataset training needed.
# Requires: pip install tabpfn>=2.0   (v2 introduces TabPFNRegressor)

class TabPFNBaseline(_BaselineWrapper):
    """TabPFN regressor via in-context learning — no gradient training.

    Reference: Hollmann et al., "TabPFN: A Transformer That Solves Small
    Tabular Classification Problems in a Second," ICLR 2023.
    TabPFNRegressor available in tabpfn>=2.0.

    Authentication
    --------------
    Set the environment variable TABPFN_API_KEY to your API key to skip
    the browser login flow entirely (required on Windows):
        set TABPFN_API_KEY=your_key_here   (Windows CMD)
        $env:TABPFN_API_KEY="your_key"     (Windows PowerShell)

    Notes
    -----
    TabPFN processes the entire training set as context at inference time.
    When the training set exceeds `max_train` samples it is subsampled to
    avoid excessive memory use.
    """
    name = "tabpfn"

    def __init__(self, n_estimators: int = 8, seed: int = 42,
                 max_train: int = 10_000):
        import os
        try:
            from tabpfn import TabPFNRegressor  # requires tabpfn>=2.0
        except ImportError:
            raise ImportError(
                "TabPFN regressor requires tabpfn>=2.0.\n"
                "Install with: pip install tabpfn>=2.0"
            )

        # ── Fix Windows asyncio event loop FIRST ────────────────────────────
        # Python 3.8+ uses ProactorEventLoop on Windows by default, which
        # breaks TabPFN's localhost OAuth callback server (WinError 10038).
        # Switching to SelectorEventLoop fixes the socket behaviour.
        import sys, asyncio
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # ── Load API key: env var → config.json on disk → fail loudly ───────
        api_key = os.environ.get("TABPFN_API_KEY", "").strip()
        if not api_key:
            api_key = _read_tabpfn_key_from_disk()
        if api_key:
            os.environ["TABPFN_API_KEY"] = api_key
            print("[TabPFN] API key loaded — skipping browser auth.")
        else:
            raise RuntimeError(
                "TabPFN API key not found.\n"
                "Run this ONCE to save your key:\n"
                "    python setup_tabpfn_key.py YOUR_API_KEY\n"
                "Get your key at: https://ux.priorlabs.ai/account/licenses"
            )

        self._model     = TabPFNRegressor(n_estimators=n_estimators,
                                          random_state=seed)
        self._max_train = max_train
        self._seed      = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X) > self._max_train:
            rng = np.random.default_rng(self._seed)
            idx = rng.choice(len(X), self._max_train, replace=False)
            X, y = X[idx], y[idx]
            print(f"[TabPFN] Subsampled training set to {self._max_train} rows.")
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).flatten()


# ── Factory function ──────────────────────────────────────────────────────────

_REGISTRY = {
    "ridge":          RidgeBaseline,
    "random_forest":  RandomForestBaseline,
    "xgboost":        XGBoostBaseline,
    "catboost":       CatBoostBaseline,
    "vanilla_mlp":    VanillaMLPBaseline,
    "tabnet":         TabNetBaseline,
    "ft_transformer": FTTransformerBaseline,
    "tabm":           TabMBaseline,
    "tabpfn":         TabPFNBaseline,
}

BASELINE_ORDER = [
    "ridge", "random_forest", "xgboost", "catboost",
    "vanilla_mlp", "tabnet", "ft_transformer", "tabm",
    # "tabpfn",  # excluded: TabPFN v2 auth uses a localhost callback server
    #             # that crashes on Windows (WinError 10038). Run on Linux/macOS
    #             # or use setup_tabpfn_key.py on a supported platform.
]


def get_f1_baseline(name: str, **kwargs) -> _BaselineWrapper:
    """Return an unfitted baseline by name.

    Valid names: ridge, random_forest, xgboost, catboost,
                 vanilla_mlp, tabnet, ft_transformer, tabm, tabpfn
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Valid: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
