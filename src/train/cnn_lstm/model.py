"""CNN + LSTM hybrid model for binary and multi-class intrusion detection."""

from typing import Dict, List

import torch
import torch.nn as nn

from src.train.config import (
    CNN_LSTM_FILTERS, CNN_LSTM_KERNEL_SIZE,
    CNN_LSTM_HIDDEN_SIZE, CNN_LSTM_NUM_LAYERS, CNN_LSTM_DROPOUT,
)


class CNNLSTMClassifier(nn.Module):
    """Hybrid CNN-LSTM network.

    1D-CNN extracts local patterns from the feature vector, producing
    a sequence of feature maps.  The LSTM then processes that sequence
    to capture inter-region dependencies.  A fully-connected head
    produces the final classification.

    Data flow
    ---------
    (B, F)  ->  unsqueeze  ->  (B, 1, F)
            ->  Conv1d block  ->  (B, C, F)     C = last filter count
            ->  permute       ->  (B, F, C)     treat spatial positions as timesteps
            ->  LSTM          ->  (B, hidden)   last hidden state
            ->  FC head       ->  (B, num_classes) or (B,) for binary
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 1,
        cnn_filters: List[int] = None,
        kernel_size: int = CNN_LSTM_KERNEL_SIZE,
        lstm_hidden: int = CNN_LSTM_HIDDEN_SIZE,
        lstm_layers: int = CNN_LSTM_NUM_LAYERS,
        dropout: float = CNN_LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._num_classes = num_classes
        self._cnn_filters = cnn_filters or list(CNN_LSTM_FILTERS)
        self._kernel_size = kernel_size
        self._lstm_hidden = lstm_hidden
        self._lstm_layers = lstm_layers
        self._dropout = dropout

        # ── CNN feature extractor ───────────────────────────────────────────
        conv_layers = []
        in_ch = 1
        for out_ch in self._cnn_filters:
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.conv_block = nn.Sequential(*conv_layers)

        # ── LSTM sequential processor ──────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self._cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Classifier head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                         # (B, 1, F)
        x = self.conv_block(x)                     # (B, C, F)
        x = x.permute(0, 2, 1)                     # (B, F, C) — timesteps=F
        lstm_out, _ = self.lstm(x)                  # (B, F, H)
        last = lstm_out[:, -1, :]                   # (B, H)
        out = self.classifier(last)                 # (B, num_classes)
        if self._num_classes == 1:
            return out.squeeze(-1)                  # (B,)
        return out

    def get_config(self) -> Dict:
        return {
            "input_size": self._input_size,
            "num_classes": self._num_classes,
            "cnn_filters": self._cnn_filters,
            "kernel_size": self._kernel_size,
            "lstm_hidden": self._lstm_hidden,
            "lstm_layers": self._lstm_layers,
            "dropout": self._dropout,
        }
