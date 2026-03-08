"""Train Autoencoder for binary anomaly detection on UNSW-NB15."""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

from src.train.data_loader import UNSWDataLoader
from src.train.base_trainer import BaseTrainer
from src.train.autoencoder.model import Autoencoder
from src.train.config import (
    DL_MODELS_DIR,
    AE_LR, AE_EPOCHS, AE_PATIENCE, AE_THRESHOLD_PERCENTILE,
    AE_NOISE_FACTOR,
)


class AutoencoderBinaryTrainer(BaseTrainer):
    """Trains on normal traffic only; detects attacks via reconstruction error."""

    def __init__(
        self,
        n_features: int,
        num_classes: int = 1,
        class_names=None,
        device: Optional[torch.device] = None,
    ) -> None:
        model = Autoencoder(input_size=n_features)
        super().__init__(
            model=model,
            model_name="autoencoder_binary",
            learning_rate=AE_LR,
            epochs=AE_EPOCHS,
            patience=AE_PATIENCE,
            device=device,
        )
        self._criterion = nn.MSELoss()
        self._threshold: Optional[float] = None

    # ── epoch logic ─────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        self._model.train()
        total_loss = 0.0
        for X_batch, _ in loader:
            X_batch = X_batch.to(self._device)
            # Denoising: add Gaussian noise, reconstruct clean input
            noise = torch.randn_like(X_batch) * AE_NOISE_FACTOR
            X_noisy = X_batch + noise
            self._optimizer.zero_grad()
            loss = self._criterion(self._model(X_noisy), X_batch)
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def _validate_epoch(self, loader: DataLoader) -> float:
        """Validation loss on normal samples only (consistent with training)."""
        self._model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                mask = y_batch == 0
                if not mask.any():
                    continue
                X_normal = X_batch[mask]
                loss = self._criterion(self._model(X_normal), X_normal)
                total_loss += loss.item() * len(X_normal)
                count += len(X_normal)
        return total_loss / count if count > 0 else float("inf")

    # ── threshold ───────────────────────────────────────────────────────────

    def compute_threshold(self, val_loader: DataLoader) -> float:
        self._model.eval()
        all_errors, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self._device)
                recon = self._model(X_batch)
                mse = torch.mean((recon - X_batch) ** 2, dim=1)
                all_errors.append(mse.cpu().numpy())
                all_labels.append(y_batch.numpy())

        errors = np.concatenate(all_errors)
        labels = np.concatenate(all_labels)

        # Find threshold that maximises F1 on validation set
        normal_errors = errors[labels == 0]
        percentiles = np.arange(80, 100, 0.5)
        best_f1, best_thresh = 0.0, float(np.percentile(normal_errors, AE_THRESHOLD_PERCENTILE))
        for p in percentiles:
            t = float(np.percentile(normal_errors, p))
            preds = (errors > t).astype(int)
            f1 = float(f1_score(labels, preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        self._threshold = best_thresh
        print(f"Anomaly threshold: {self._threshold:.6f} (best val F1={best_f1:.4f})")
        return self._threshold

    # ── evaluation ──────────────────────────────────────────────────────────

    def evaluate(self, loader: DataLoader) -> Dict:
        if self._threshold is None:
            raise RuntimeError("Call compute_threshold() before evaluate()")

        self._model.eval()
        all_errors, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                recon = self._model(X_batch)
                mse = torch.mean((recon - X_batch) ** 2, dim=1)
                all_errors.append(mse.cpu().numpy())
                all_labels.append(y_batch.numpy())

        errors = np.concatenate(all_errors)
        y_true = np.concatenate(all_labels)
        y_pred = (errors > self._threshold).astype(int)

        lo, hi = errors.min(), errors.max()
        y_prob = (errors - lo) / (hi - lo) if hi > lo else np.zeros_like(errors)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
            "threshold": self._threshold,
            "mean_normal_error": float(errors[y_true == 0].mean()),
            "mean_attack_error": float(errors[y_true == 1].mean()),
        }

        print(f"\n{self._model_name} Test Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  Threshold: {metrics['threshold']:.6f}")
        return metrics

    # ── save (override to include threshold) ────────────────────────────────

    def save_model(self) -> Path:
        DL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = DL_MODELS_DIR / f"{self._model_name}.pt"
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "model_class": self._model.__class__.__name__,
                "model_config": self._model.get_config(),
                "threshold": self._threshold,
            },
            path,
        )
        print(f"Model saved to {path}")
        return path


def main():
    loader = UNSWDataLoader(mode="binary")
    X_train, y_train, X_test, y_test = loader.load_and_preprocess()
    loader.save_preprocessor()

    train_dl, val_dl, test_dl = loader.create_autoencoder_dataloaders(
        X_train, y_train, X_test, y_test
    )

    trainer = AutoencoderBinaryTrainer(n_features=loader.n_features)
    trainer.train(train_dl, val_dl)
    trainer.compute_threshold(val_dl)
    metrics = trainer.evaluate(test_dl)
    trainer.save_model()
    trainer.save_metrics(metrics)


if __name__ == "__main__":
    main()
