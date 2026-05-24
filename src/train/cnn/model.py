"""1D-CNN model with Residual Blocks for improved accuracy on tabular data."""

from typing import Dict, List

import torch
import torch.nn as nn

from src.train.config import CNN_FILTERS, CNN_KERNEL_SIZE, CNN_DROPOUT


class ResidualBlock(nn.Module):
    """A 1D Convolutional Residual Block.
    
    Ensures that the input signal is preserved while learning new features.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection to handle dimension mismatch
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class CNNClassifier(nn.Module):
    """1D Convolutional network with Residual Blocks.

    The feature vector is treated as a 1-channel 1D signal.
    Residual blocks extract local patterns while preserving signal integrity,
    followed by global average pooling and a fully-connected classifier.
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
            layers.append(ResidualBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch

        self.conv_block = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1) # Capture strongest signals
        
        # Classifier head (input is concat of Avg and Max pooling)
        self.classifier = nn.Sequential(
            nn.Linear(self._filters[-1] * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                         # (B, 1, F)
        x = self.conv_block(x)                     # (B, C, F)
        
        avg_p = self.global_pool(x).squeeze(-1)    # (B, C)
        max_p = self.global_max_pool(x).squeeze(-1) # (B, C)
        x = torch.cat([avg_p, max_p], dim=1)       # (B, 2*C)
        
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
