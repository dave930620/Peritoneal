"""
src/models/stage_b_models.py — Stage B prescription optimizer architectures.

All models share the same interface:
    forward(x) -> (z_cont, cat_logits)
        x          : (B, in_dim)   standardized features with Rx columns zeroed
        z_cont     : (B, n_cont)   continuous Rx in z-space
        cat_logits : (B, n_cat)    categorical Rx logits

Models
------
MlpRxHead         : current 3-layer MLP (default, same as F2RxHead)
DeepMlpRxHead     : 5-layer MLP with more capacity
ResidualRxHead    : MLP with residual (skip) connections
TransformerRxHead : CLS-token feature transformer
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Existing baseline (matches F2RxHead from f2_head.py)
# ---------------------------------------------------------------------------

class MlpRxHead(nn.Module):
    """3-layer MLP Stage B (same architecture as original F2RxHead)."""

    def __init__(self, in_dim: int, n_cont: int, n_cat: int,
                 hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
        )
        self.cont_head = nn.Linear(hidden // 2, n_cont)
        self.cat_head  = nn.Linear(hidden // 2, n_cat)

    def forward(self, x):
        h = self.net(x)
        return self.cont_head(h), self.cat_head(h)


# ---------------------------------------------------------------------------
# Deeper MLP
# ---------------------------------------------------------------------------

class DeepMlpRxHead(nn.Module):
    """5-layer MLP with larger hidden width for more capacity."""

    def __init__(self, in_dim: int, n_cont: int, n_cat: int,
                 hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),  nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),  nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),  nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4), nn.GELU(),
        )
        self.cont_head = nn.Linear(hidden // 4, n_cont)
        self.cat_head  = nn.Linear(hidden // 4, n_cat)

    def forward(self, x):
        h = self.net(x)
        return self.cont_head(h), self.cat_head(h)


# ---------------------------------------------------------------------------
# Residual MLP
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))


class ResidualRxHead(nn.Module):
    """MLP with 3 residual blocks for better gradient flow."""

    def __init__(self, in_dim: int, n_cont: int, n_cat: int,
                 hidden: int = 192, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.blocks = nn.Sequential(
            _ResBlock(hidden, dropout),
            _ResBlock(hidden, dropout),
            _ResBlock(hidden, dropout),
        )
        self.cont_head = nn.Linear(hidden, n_cont)
        self.cat_head  = nn.Linear(hidden, n_cat)

    def forward(self, x):
        h = self.blocks(self.input_proj(x))
        return self.cont_head(h), self.cat_head(h)


# ---------------------------------------------------------------------------
# Transformer Stage B
# ---------------------------------------------------------------------------

class TransformerRxHead(nn.Module):
    """Feature-tokenization transformer with CLS token for Stage B.

    Each standardized feature (Rx columns zeroed) becomes a d_model token.
    CLS token output drives the prescription output heads.
    """

    def __init__(self, in_dim: int, n_cont: int, n_cat: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Linear(1, d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.cont_head   = nn.Linear(d_model, n_cont)
        self.cat_head    = nn.Linear(d_model, n_cat)

    def forward(self, x):
        B = x.size(0)
        tokens = self.token_embed(x.unsqueeze(-1))       # (B, F, d_model)
        cls    = self.cls_token.expand(B, -1, -1)         # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, F+1, d_model)
        out    = self.transformer(tokens)
        cls_out = self.norm(out[:, 0])
        return self.cont_head(cls_out), self.cat_head(cls_out)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STAGE_B_REGISTRY = {
    "mlp":         MlpRxHead,
    "deep_mlp":    DeepMlpRxHead,
    "residual":    ResidualRxHead,
    "transformer": TransformerRxHead,
}


def build_stage_b(stage_b_type: str, in_dim: int, n_cont: int, n_cat: int,
                  cfg: dict) -> nn.Module:
    """Instantiate a Stage B model by name."""
    if stage_b_type not in STAGE_B_REGISTRY:
        raise ValueError(
            f"Unknown stage_b_type '{stage_b_type}'. "
            f"Choose from: {list(STAGE_B_REGISTRY)}")
    cls = STAGE_B_REGISTRY[stage_b_type]
    hidden  = cfg.get("F2_HIDDEN", 128)
    dropout = cfg.get("F2_DROPOUT", 0.1)
    if stage_b_type == "transformer":
        return cls(in_dim, n_cont, n_cat)
    return cls(in_dim, n_cont, n_cat, hidden=hidden, dropout=dropout)
