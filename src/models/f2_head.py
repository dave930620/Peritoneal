"""
F2 MLP prescription head used in Stage B (effect optimization).

Takes the standardized feature vector (same space as F1, but with Rx columns
zeroed) and outputs:
- z_cont   : continuous prescription values in z-space (len(CONT_RX))
- cat_logits : class logits for the categorical prescription variable

The MLP is a simple 3-layer ReLU network. It is initialized randomly and
trained in Stage B using the F1 oracle gradient signal, anchored to CatBoost
Stage A teacher predictions via a proximal loss term.
"""

import torch
import torch.nn as nn


class F2RxHead(nn.Module):
    """3-layer MLP prescription generator for Stage B optimization."""

    def __init__(
        self,
        in_dim:  int,
        n_cont:  int,
        n_cat:   int,
        hidden:  int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head_cont = nn.Linear(hidden, n_cont)   # outputs z-space continuous Rx
        self.head_cat  = nn.Linear(hidden, n_cat)    # outputs class logits

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.head_cont(h), self.head_cat(h)
