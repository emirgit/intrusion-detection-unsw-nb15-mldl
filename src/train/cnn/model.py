"""1D-CNN model for binary and multi-class intrusion detection."""

from typing import Dict, List

import torch
import torch.nn as nn

from src.train.config import CNN_FILTERS, CNN_KERNEL_SIZE, CNN_DROPOUT


class CNNClassifier(nn.Module):
    """1D Convolutional network.

    The feature vector is treated as a 1-channel 1D signal.
    Stacked Conv1d layers extract local patterns, followed by
    global average pooling and a fully-connected classifier.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 1,
        filters: List[int] = None,
        kernel_size: int = CNN_KERNEL_SIZE,
        dropout: float = CNN_DROPOUT,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._num_classes = num_classes
        self._filters = filters or list(CNN_FILTERS)
        self._kernel_size = kernel_size
        self._dropout = dropout

        layers = []
        in_ch = 1
        for out_ch in self._filters:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.conv_block = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self._filters[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                         # (B, 1, F)
        x = self.conv_block(x)                     # (B, C, F)
        x = self.global_pool(x).squeeze(-1)        # (B, C)
        out = self.classifier(x)                   # (B, num_classes)
        if self._num_classes == 1:
            return out.squeeze(-1)                 # (B,)
        return out

    def get_config(self) -> Dict:
        return {
            "input_size": self._input_size,
            "num_classes": self._num_classes,
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "dropout": self._dropout,
        }
