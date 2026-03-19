"""Symmetric autoencoder for anomaly-based intrusion detection."""

from typing import Dict, List

import torch
import torch.nn as nn

from src.train.config import AE_ENCODER_DIMS, AE_DROPOUT


class Autoencoder(nn.Module):
    """Symmetric autoencoder — trained on normal traffic only.

    High reconstruction error signals an anomaly (attack).
    """

    def __init__(
        self,
        input_size: int,
        encoder_dims: List[int] = None,
        dropout: float = AE_DROPOUT,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._encoder_dims = encoder_dims or list(AE_ENCODER_DIMS)
        self._dropout = dropout

        # ── Encoder ─────────────────────────────────────────────────────────
        enc = []
        in_dim = input_size
        for out_dim in self._encoder_dims:
            enc.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*enc)

        # ── Decoder (mirror) ───────────────────────────────────────────────
        dec_dims = list(reversed(self._encoder_dims[:-1])) + [input_size]
        dec = []
        in_dim = self._encoder_dims[-1]
        for i, out_dim in enumerate(dec_dims):
            dec.append(nn.Linear(in_dim, out_dim))
            if i < len(dec_dims) - 1:
                dec.extend([
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            in_dim = out_dim
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_config(self) -> Dict:
        return {
            "input_size": self._input_size,
            "encoder_dims": self._encoder_dims,
            "dropout": self._dropout,
        }
