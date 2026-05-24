"""FT-Transformer — Feature Tokenizer + Transformer for tabular data.

Based on: Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data"
https://arxiv.org/abs/2106.11959

Key idea: each numerical feature is independently projected into a d-dimensional
embedding (tokenisation), a learnable [CLS] token is prepended, and standard
Transformer encoder layers learn pairwise feature interactions through
self-attention. The [CLS] output drives classification.
"""

from typing import Dict, Optional

import math
import torch
import torch.nn as nn

from src.train.config import (
    FT_D_TOKEN, FT_N_HEADS, FT_N_LAYERS, FT_D_FFN,
    FT_DROPOUT, FT_ATTN_DROPOUT,
)


class NumericalTokenizer(nn.Module):
    """Project each scalar feature into a d-dimensional token embedding.

    For n features the output is (B, n, d_token).  Each feature gets its own
    learned linear projection (weight + bias).
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        # One weight & bias per feature — vectorised as (n_features, d_token)
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)  ->  (B, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformerClassifier(nn.Module):
    """FT-Transformer for tabular classification.

    1. NumericalTokenizer maps each feature to a d-dimensional embedding.
    2. A learnable [CLS] token is prepended to the token sequence.
    3. Standard Transformer encoder layers with multi-head self-attention.
    4. The [CLS] token output feeds a classification head.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 1,
        d_token: int = FT_D_TOKEN,
        n_heads: int = FT_N_HEADS,
        n_layers: int = FT_N_LAYERS,
        d_ffn: int = FT_D_FFN,
        dropout: float = FT_DROPOUT,
        attn_dropout: float = FT_ATTN_DROPOUT,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._num_classes = num_classes
        self._d_token = d_token
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._d_ffn = d_ffn
        self._dropout = dropout
        self._attn_dropout = attn_dropout

        # ── Feature tokenizer ──────────────────────────────────────────────
        self.tokenizer = NumericalTokenizer(input_size, d_token)

        # ── Learnable [CLS] token ──────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # ── Transformer encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # ── Classification head ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)
        tokens = self.tokenizer(x)                              # (B, n, d)
        cls = self.cls_token.expand(x.size(0), -1, -1)          # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)                # (B, n+1, d)

        tokens = self.transformer(tokens)                       # (B, n+1, d)
        cls_out = tokens[:, 0]                                  # (B, d)

        out = self.head(cls_out)                                # (B, num_classes)
        if self._num_classes == 1:
            return out.squeeze(-1)                              # (B,)
        return out

    def get_config(self) -> Dict:
        return {
            "input_size": self._input_size,
            "num_classes": self._num_classes,
            "d_token": self._d_token,
            "n_heads": self._n_heads,
            "n_layers": self._n_layers,
            "d_ffn": self._d_ffn,
            "dropout": self._dropout,
            "attn_dropout": self._attn_dropout,
        }
