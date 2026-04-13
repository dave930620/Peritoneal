"""
src/models/stage_a_nn.py — Deep-learning Stage A architectures.

Both models take standardized patient features (Rx columns excluded) as input
and output (z_cont, cat_logits) in the same z-space used by Stage B.

Models
------
MultiTaskStageA      : shared MLP encoder → regression + classification heads
TransformerStageA    : CLS-token feature transformer → same heads
"""

import torch
import torch.nn as nn


class MultiTaskStageA(nn.Module):
    """Shared MLP encoder with separate heads for each Rx output.

    Parameters
    ----------
    n_patient_features : int    number of patient-only features (no Rx)
    n_cont             : int    number of continuous Rx variables
    n_cat_classes      : int    number of categorical Rx classes
    hidden             : int    hidden layer width
    dropout            : float
    """

    def __init__(self, n_patient_features: int, n_cont: int, n_cat_classes: int,
                 hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_patient_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.cont_head = nn.Linear(hidden // 2, n_cont)
        self.cat_head  = nn.Linear(hidden // 2, n_cat_classes)

    def forward(self, x: torch.Tensor):
        """
        x : (B, n_patient_features)
        Returns
        -------
        z_cont     : (B, n_cont)       continuous Rx in z-space
        cat_logits : (B, n_cat_classes)
        """
        h = self.encoder(x)
        return self.cont_head(h), self.cat_head(h)


class TransformerStageA(nn.Module):
    """Feature-tokenization transformer with CLS token for Stage A.

    Each patient feature becomes a d_model-dimensional token.
    A learned CLS token aggregates information via self-attention.
    CLS output goes through regression and classification heads.

    Parameters
    ----------
    n_patient_features : int
    n_cont             : int
    n_cat_classes      : int
    d_model            : int    token dimension
    n_heads            : int    attention heads
    n_layers           : int    transformer layers
    dropout            : float
    """

    def __init__(self, n_patient_features: int, n_cont: int, n_cat_classes: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Linear(1, d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model       = d_model,
            nhead         = n_heads,
            dim_feedforward = d_model * 4,
            dropout       = dropout,
            activation    = "gelu",
            batch_first   = True,
            norm_first    = True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.cont_head   = nn.Linear(d_model, n_cont)
        self.cat_head    = nn.Linear(d_model, n_cat_classes)

    def forward(self, x: torch.Tensor):
        """
        x : (B, n_patient_features)
        Returns
        -------
        z_cont     : (B, n_cont)
        cat_logits : (B, n_cat_classes)
        """
        B = x.size(0)
        # Each feature → a token of shape d_model
        tokens = self.token_embed(x.unsqueeze(-1))               # (B, F, d_model)
        cls    = self.cls_token.expand(B, -1, -1)                 # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)                  # (B, F+1, d_model)
        out    = self.transformer(tokens)                          # (B, F+1, d_model)
        cls_out = self.norm(out[:, 0])                            # (B, d_model)
        return self.cont_head(cls_out), self.cat_head(cls_out)
