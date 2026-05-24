"""TabNet — Attentive Interpretable Tabular Learning.

Based on: Arik & Pfister (2021) "TabNet: Attentive Interpretable Tabular Learning"
https://arxiv.org/abs/1908.07442

Key idea: at each decision step, a learned sparse attention mask selects which
features to process. This gives the model built-in feature selection and
interpretability.  Multiple decision steps are aggregated for the final output.
"""

from typing import Dict

import torch
import torch.nn as nn

from src.train.config import (
    TABNET_N_D, TABNET_N_A, TABNET_N_STEPS, TABNET_GAMMA,
    TABNET_N_SHARED, TABNET_N_INDEPENDENT, TABNET_MOMENTUM, TABNET_DROPOUT,
)


class GLUBlock(nn.Module):
    """Gated Linear Unit: FC -> BN -> GLU activation."""

    def __init__(self, in_dim: int, out_dim: int, bn_momentum: float,
                 virtual_batch_size: int = None) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2, bias=False)
        self.bn = nn.BatchNorm1d(out_dim * 2, momentum=bn_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class FeatureTransformer(nn.Module):
    """Shared + step-specific GLU blocks that process masked features.

    The first `n_shared` blocks have weights shared across all decision steps.
    The remaining `n_independent` blocks are unique to each step.
    """

    def __init__(
        self, in_dim: int, out_dim: int, shared_layers: nn.ModuleList,
        n_independent: int, bn_momentum: float,
    ) -> None:
        super().__init__()
        self.shared = shared_layers
        self.independent = nn.ModuleList()
        first_dim = in_dim
        for i in range(n_independent):
            d_in = first_dim if (i == 0 and len(shared_layers) == 0) else out_dim
            self.independent.append(GLUBlock(d_in, out_dim, bn_momentum))
            first_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.shared:
            x = layer(x) + x if x.size(-1) == layer.fc.out_features // 2 else layer(x)
        for layer in self.independent:
            x = layer(x) + x if x.size(-1) == layer.fc.out_features // 2 else layer(x)
        return x


class AttentiveTransformer(nn.Module):
    """Produces a sparse feature mask using prior scales and sparsemax."""

    def __init__(self, in_dim: int, out_dim: int, bn_momentum: float) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)

    def forward(self, x: torch.Tensor, prior_scales: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = x * prior_scales
        x = _sparsemax(x, dim=-1)
        return x


def _sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation (Martins & Astudillo, 2016).

    Projects onto the probability simplex — produces truly sparse outputs
    (exact zeros), unlike softmax.
    """
    sorted_x, _ = torch.sort(x, descending=True, dim=dim)
    cumsum = sorted_x.cumsum(dim=dim)
    k = torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype)
    shape = [1] * x.dim()
    shape[dim] = -1
    k = k.view(shape)
    support = (sorted_x - (cumsum - 1) / k) > 0
    k_z = support.sum(dim=dim, keepdim=True).float()
    tau = (cumsum.gather(dim, (k_z - 1).long().clamp(min=0)) - 1) / k_z
    return torch.clamp(x - tau, min=0)


class TabNetClassifier(nn.Module):
    """TabNet for tabular classification.

    1. Initial BatchNorm on raw features.
    2. For each decision step:
       a. AttentiveTransformer produces a sparse feature mask.
       b. Masked features are processed by a FeatureTransformer (shared + independent GLU blocks).
       c. ReLU output is aggregated across steps.
    3. Final FC layer for classification.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 1,
        n_d: int = TABNET_N_D,
        n_a: int = TABNET_N_A,
        n_steps: int = TABNET_N_STEPS,
        gamma: float = TABNET_GAMMA,
        n_shared: int = TABNET_N_SHARED,
        n_independent: int = TABNET_N_INDEPENDENT,
        momentum: float = TABNET_MOMENTUM,
        dropout: float = TABNET_DROPOUT,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._num_classes = num_classes
        self._n_d = n_d
        self._n_a = n_a
        self._n_steps = n_steps
        self._gamma = gamma
        self._n_shared = n_shared
        self._n_independent = n_independent
        self._momentum = momentum
        self._dropout = dropout

        # ── Initial batch normalisation ────────────────────────────────────
        self.initial_bn = nn.BatchNorm1d(input_size, momentum=momentum)

        # ── Shared GLU layers (shared across all decision steps) ───────────
        shared_layers = nn.ModuleList()
        for i in range(n_shared):
            d_in = input_size if i == 0 else n_d + n_a
            shared_layers.append(GLUBlock(d_in, n_d + n_a, momentum))
        self._shared_layers = shared_layers

        # ── Per-step modules ───────────────────────────────────────────────
        self.steps = nn.ModuleList()
        self.attention_steps = nn.ModuleList()

        for _ in range(n_steps):
            # Feature transformer: shared + step-specific
            ft = FeatureTransformer(
                in_dim=input_size, out_dim=n_d + n_a,
                shared_layers=shared_layers, n_independent=n_independent,
                bn_momentum=momentum,
            )
            self.steps.append(ft)

            # Attentive transformer for mask generation
            at = AttentiveTransformer(n_a, input_size, momentum)
            self.attention_steps.append(at)

        # ── Initial feature transformer for h_0 (attention seed) ──────────
        self.initial_ft = FeatureTransformer(
            in_dim=input_size, out_dim=n_d + n_a,
            shared_layers=shared_layers, n_independent=n_independent,
            bn_momentum=momentum,
        )

        # ── Classification head ────────────────────────────────────────────
        self.final_fc = nn.Linear(n_d, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.initial_bn(x)                             # (B, n_features)

        # Prior scales — tracks which features have been used
        prior_scales = torch.ones(B, self._input_size, device=x.device)

        # Initial split for attention seed
        h = self.initial_ft(x)                              # (B, n_d + n_a)
        d_out = h[:, :self._n_d]                            # decision part
        a_out = h[:, self._n_d:]                            # attention part

        aggregated = torch.zeros(B, self._n_d, device=x.device)

        for step_idx in range(self._n_steps):
            # Attention mask from prior attention output
            mask = self.attention_steps[step_idx](a_out, prior_scales)  # (B, n_features)

            # Update prior scales — penalise reused features
            prior_scales = prior_scales * (self._gamma - mask)

            # Apply mask and process
            masked_x = mask * x                             # (B, n_features)
            h = self.steps[step_idx](masked_x)              # (B, n_d + n_a)
            d_out = h[:, :self._n_d]                        # decision
            a_out = h[:, self._n_d:]                        # attention for next step

            aggregated = aggregated + torch.relu(d_out)

        # Classification
        out = self.dropout(aggregated)
        out = self.final_fc(out)                            # (B, num_classes)
        if self._num_classes == 1:
            return out.squeeze(-1)                          # (B,)
        return out

    def get_config(self) -> Dict:
        return {
            "input_size": self._input_size,
            "num_classes": self._num_classes,
            "n_d": self._n_d,
            "n_a": self._n_a,
            "n_steps": self._n_steps,
            "gamma": self._gamma,
            "n_shared": self._n_shared,
            "n_independent": self._n_independent,
            "momentum": self._momentum,
            "dropout": self._dropout,
        }
