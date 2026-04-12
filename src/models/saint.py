"""
SAINT model for tabular data (Self-Attention and Intersample Attention Transformer).

Architecture
------------
- Discrete features are projected via a linear embedding layer.
- Continuous features are projected via a separate linear embedding layer.
- Both embeddings are summed into a single token and passed through
  a Transformer encoder.
- The Transformer output is projected to a scalar (regression).

This is the F1 model used to predict PD Kt/V.
"""

import numpy as np
import torch
import torch.nn as nn


class SAINT(nn.Module):
    """SAINT model for mixed discrete/continuous tabular features."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        discrete_feature_indices: list,
        continuous_feature_indices: list,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.discrete_feature_indices   = discrete_feature_indices
        self.continuous_feature_indices = continuous_feature_indices
        self.num_discrete   = len(discrete_feature_indices)
        self.num_continuous = len(continuous_feature_indices)

        print(
            f"[SAINT] discrete={self.num_discrete}  "
            f"continuous={self.num_continuous}  "
            f"hidden={hidden_size}  layers={num_layers}  heads={num_heads}"
        )

        # Embedding layers
        if self.num_discrete > 0:
            self.discrete_embedding = nn.Sequential(
                nn.Linear(self.num_discrete, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            nn.init.xavier_uniform_(self.discrete_embedding[0].weight)
            nn.init.zeros_(self.discrete_embedding[0].bias)

        if self.num_continuous > 0:
            self.continuous_embedding = nn.Sequential(
                nn.Linear(self.num_continuous, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            nn.init.xavier_uniform_(self.continuous_embedding[0].weight)
            nn.init.zeros_(self.continuous_embedding[0].bias)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )
        for m in self.fc_out:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        max_idx = x.shape[1] - 1
        disc_idx = [i for i in self.discrete_feature_indices   if i <= max_idx]
        cont_idx = [i for i in self.continuous_feature_indices if i <= max_idx]

        discrete_x   = x[:, disc_idx] if disc_idx else torch.zeros((x.shape[0], 0), device=x.device)
        continuous_x = x[:, cont_idx] if cont_idx else torch.zeros((x.shape[0], 0), device=x.device)

        token = None
        if discrete_x.shape[1] > 0:
            token = self.discrete_embedding(discrete_x)
        if continuous_x.shape[1] > 0:
            cont_emb = self.continuous_embedding(continuous_x)
            token = cont_emb if token is None else token + cont_emb

        if token is None:
            raise RuntimeError("No valid features available for SAINT forward pass.")

        # Transformer expects shape (batch, seq_len, d_model); seq_len=1 here
        token = token.unsqueeze(1)
        out   = self.transformer_encoder(token)
        out   = out.squeeze(1)
        return self.fc_out(out)

    @torch.no_grad()
    def get_feature_importance(
        self, X: torch.Tensor, feature_names: list, n_repeats: int = 1
    ) -> np.ndarray:
        """Permutation importance: mean absolute change in output when feature is shuffled."""
        self.eval()
        base_pred = self(X).cpu().numpy()
        importances = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            X_perturbed = X.clone()
            X_perturbed[:, i] = (
                torch.randn_like(X_perturbed[:, i]) * X_perturbed[:, i].std()
                + X_perturbed[:, i].mean()
            )
            perturbed_pred = self(X_perturbed).cpu().numpy()
            importances[i] = np.mean(np.abs(base_pred - perturbed_pred))
        return importances
