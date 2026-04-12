"""
saint_variant.py — Ablation-capable SAINT for F1 experiments.

Controlled by flags so that every ablation (A-F1-1 through A-F1-10)
can be instantiated from a single class rather than N separate files.

Ablation map
------------
A-F1-1  use_dual_embedding=False   → single Linear for all features
A-F1-2  use_layer_norm=False       → remove LayerNorm from embeddings
A-F1-3  activation="relu"         → ReLU instead of GELU everywhere
A-F1-4  use_transformer=False     → replace Transformer with 3-layer MLP
A-F1-5  num_layers=2              → shallow Transformer
A-F1-6  num_layers=4              → medium Transformer
A-F1-7  num_heads=1               → single-head attention
A-F1-8  num_heads=4               → 4-head attention
A-F1-9  (no LR scheduler)         → training-loop flag in run_ablations.py
A-F1-10 (no weight decay)         → training-loop flag in run_ablations.py
"""

import numpy as np
import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    return nn.GELU() if name == "gelu" else nn.ReLU(inplace=True)


def _make_embedding(in_dim: int, hidden: int, use_ln: bool, act: str,
                    dropout: float) -> nn.Sequential:
    layers: list = [nn.Linear(in_dim, hidden)]
    if use_ln:
        layers.append(nn.LayerNorm(hidden))
    layers.append(_act(act))
    layers.append(nn.Dropout(dropout))
    seq = nn.Sequential(*layers)
    nn.init.xavier_uniform_(seq[0].weight)
    nn.init.zeros_(seq[0].bias)
    return seq


class SAINTVariant(nn.Module):
    """Configurable SAINT for ablation studies."""

    def __init__(
        self,
        input_size:                  int,
        hidden_size:                 int,
        output_size:                 int,
        discrete_feature_indices:    list,
        continuous_feature_indices:  list,
        # --- ablation flags ---
        use_dual_embedding:  bool  = True,
        use_layer_norm:      bool  = True,
        activation:          str   = "gelu",
        use_transformer:     bool  = True,
        num_layers:          int   = 6,
        num_heads:           int   = 8,
        dropout:             float = 0.1,
    ):
        super().__init__()
        self.discrete_feature_indices   = discrete_feature_indices
        self.continuous_feature_indices = continuous_feature_indices
        self.use_dual_embedding         = use_dual_embedding
        self.use_transformer            = use_transformer
        self.hidden_size                = hidden_size

        n_disc = len(discrete_feature_indices)
        n_cont = len(continuous_feature_indices)

        act_fn = _act(activation)

        # ── Embedding ────────────────────────────────────────────────────────
        if use_dual_embedding:
            if n_disc > 0:
                self.disc_emb = _make_embedding(n_disc, hidden_size,
                                                use_layer_norm, activation, dropout)
            if n_cont > 0:
                self.cont_emb = _make_embedding(n_cont, hidden_size,
                                                use_layer_norm, activation, dropout)
        else:
            # A-F1-1: single linear projection over all features
            self.single_emb = _make_embedding(input_size, hidden_size,
                                              use_layer_norm, activation, dropout)

        # ── Backbone ─────────────────────────────────────────────────────────
        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout, batch_first=True, activation=activation,
            )
            self.backbone = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self._is_transformer = True
        else:
            # A-F1-4: 3-layer MLP backbone
            self.backbone = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), _act(activation), nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), _act(activation), nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), _act(activation), nn.Dropout(dropout),
            )
            self._is_transformer = False

        # ── Output head ──────────────────────────────────────────────────────
        head_layers: list = [nn.Linear(hidden_size, hidden_size // 2)]
        if use_layer_norm:
            head_layers.append(nn.LayerNorm(hidden_size // 2))
        head_layers += [_act(activation), nn.Dropout(dropout),
                        nn.Linear(hidden_size // 2, output_size)]
        self.fc_out = nn.Sequential(*head_layers)
        for m in self.fc_out:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SAINTVariant] dual={use_dual_embedding} transformer={use_transformer} "
              f"layers={num_layers} heads={num_heads} act={activation} "
              f"ln={use_layer_norm} | params={n_params:,}")

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        max_idx = x.shape[1] - 1
        if self.use_dual_embedding:
            disc_idx = [i for i in self.discrete_feature_indices   if i <= max_idx]
            cont_idx = [i for i in self.continuous_feature_indices if i <= max_idx]
            token = None
            if disc_idx and hasattr(self, "disc_emb"):
                token = self.disc_emb(x[:, disc_idx])
            if cont_idx and hasattr(self, "cont_emb"):
                emb = self.cont_emb(x[:, cont_idx])
                token = emb if token is None else token + emb
            if token is None:
                raise RuntimeError("No valid features for embedding.")
        else:
            token = self.single_emb(x)
        return token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        token = self._embed(x)
        if self._is_transformer:
            token = self.backbone(token.unsqueeze(1)).squeeze(1)
        else:
            token = self.backbone(token)
        return self.fc_out(token)

    @torch.no_grad()
    def get_feature_importance(self, X: torch.Tensor, feature_names: list,
                               n_repeats: int = 1) -> np.ndarray:
        self.eval()
        base = self(X).cpu().numpy()
        imp  = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            Xp = X.clone()
            Xp[:, i] = (torch.randn_like(Xp[:, i]) * Xp[:, i].std()
                        + Xp[:, i].mean())
            imp[i] = np.mean(np.abs(base - self(Xp).cpu().numpy()))
        return imp
