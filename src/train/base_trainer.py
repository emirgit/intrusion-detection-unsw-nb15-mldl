"""BaseTrainer and ClassifierTrainer — shared training infrastructure."""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

from src.train.config import DL_MODELS_DIR, DL_METRICS_DIR, RANDOM_SEED


# ═════════════════════════════════════════════════════════════════════════════
#  BaseTrainer — abstract foundation
# ═════════════════════════════════════════════════════════════════════════════

class BaseTrainer(ABC):
    """Abstract base with training loop, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        learning_rate: float,
        epochs: int,
        patience: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self._model_name = model_name
        self._epochs = epochs
        self._patience = patience

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = model.to(self._device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=learning_rate
        )
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode="min", factor=0.5, patience=3
        )
        self._history: Dict = {"train_loss": [], "val_loss": []}
        self._best_val_loss = float("inf")
        self._patience_counter = 0

        torch.manual_seed(RANDOM_SEED)
        if self._device.type == "cuda":
            torch.cuda.manual_seed(RANDOM_SEED)

        total_params = sum(p.numel() for p in self._model.parameters())
        print(f"Device: {self._device}")
        print(f"Model: {self._model_name} ({total_params:,} parameters)")

    # ── abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch. Return average loss."""

    @abstractmethod
    def _validate_epoch(self, loader: DataLoader) -> float:
        """Run one validation epoch. Return average loss."""

    @abstractmethod
    def evaluate(self, loader: DataLoader) -> Dict:
        """Evaluate on test set. Return metrics dict."""

    # ── training loop ───────────────────────────────────────────────────────

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        print(f"\n{'=' * 60}")
        print(f"Training {self._model_name}")
        print(f"{'=' * 60}")

        start = time.time()

        for epoch in range(1, self._epochs + 1):
            t_loss = self._train_epoch(train_loader)
            v_loss = self._validate_epoch(val_loader)

            self._history["train_loss"].append(t_loss)
            self._history["val_loss"].append(v_loss)
            self._scheduler.step(v_loss)

            improved = v_loss < self._best_val_loss
            if improved:
                self._best_val_loss = v_loss
                self._patience_counter = 0
                self._save_checkpoint()
            else:
                self._patience_counter += 1

            lr = self._optimizer.param_groups[0]["lr"]
            tag = " * saved" if improved else ""
            print(
                f"Epoch {epoch:3d}/{self._epochs} | "
                f"train_loss: {t_loss:.6f} | "
                f"val_loss: {v_loss:.6f} | "
                f"lr: {lr:.1e}{tag}"
            )

            if self._patience_counter >= self._patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Training completed in {time.time() - start:.1f}s")
        self._load_checkpoint()
        return self._history

    # ── persistence ─────────────────────────────────────────────────────────

    def save_model(self) -> Path:
        DL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = DL_MODELS_DIR / f"{self._model_name}.pt"
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "model_class": self._model.__class__.__name__,
                "model_config": self._model.get_config(),
            },
            path,
        )
        print(f"Model saved to {path}")
        return path

    def save_metrics(self, metrics: Dict) -> Path:
        DL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
        path = DL_METRICS_DIR / f"{self._model_name}_metrics.json"

        def _serialise(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _serialise(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialise(v) for v in obj]
            return obj

        with open(path, "w") as f:
            json.dump(_serialise(metrics), f, indent=2)
        print(f"Metrics saved to {path}")
        return path

    def _save_checkpoint(self) -> None:
        DL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._model.state_dict(),
            DL_MODELS_DIR / f"{self._model_name}_best.pt",
        )

    def _load_checkpoint(self) -> None:
        path = DL_MODELS_DIR / f"{self._model_name}_best.pt"
        if path.exists():
            self._model.load_state_dict(
                torch.load(path, map_location=self._device, weights_only=True)
            )
            print("Loaded best checkpoint")


# ═════════════════════════════════════════════════════════════════════════════
#  ClassifierTrainer — concrete logic for binary & multi-class classification
# ═════════════════════════════════════════════════════════════════════════════

class ClassifierTrainer(BaseTrainer):
    """Handles both binary (BCEWithLogitsLoss) and multi-class (CrossEntropyLoss)."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        mode: str,
        num_classes: int,
        learning_rate: float,
        epochs: int,
        patience: int,
        class_names: Optional[List[str]] = None,
        enable_grad_clip: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, model_name, learning_rate, epochs, patience, device)
        self._mode = mode
        self._num_classes = num_classes
        self._class_names = class_names
        self._enable_grad_clip = enable_grad_clip

        if mode == "binary":
            self._criterion = nn.BCEWithLogitsLoss()
        else:
            self._criterion = nn.CrossEntropyLoss()

    # ── epoch logic ─────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        self._model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self._device)
            y_batch = (
                y_batch.float().to(self._device)
                if self._mode == "binary"
                else y_batch.long().to(self._device)
            )

            self._optimizer.zero_grad()
            logits = self._model(X_batch)
            loss = self._criterion(logits, y_batch)
            loss.backward()

            if self._enable_grad_clip:
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)

            self._optimizer.step()
            total_loss += loss.item() * len(X_batch)

        return total_loss / len(loader.dataset)

    def _validate_epoch(self, loader: DataLoader) -> float:
        self._model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = (
                    y_batch.float().to(self._device)
                    if self._mode == "binary"
                    else y_batch.long().to(self._device)
                )
                logits = self._model(X_batch)
                loss = self._criterion(logits, y_batch)
                total_loss += loss.item() * len(X_batch)

        return total_loss / len(loader.dataset)

    # ── evaluation ──────────────────────────────────────────────────────────

    def evaluate(self, loader: DataLoader) -> Dict:
        self._model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                logits = self._model(X_batch)

                if self._mode == "binary":
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                else:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds = logits.argmax(dim=1).cpu().numpy()

                all_preds.append(preds)
                all_labels.append(y_batch.numpy())
                all_probs.append(probs)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        if self._mode == "binary":
            y_prob = np.concatenate(all_probs)
            metrics = self._binary_metrics(y_true, y_pred, y_prob)
        else:
            y_prob = np.vstack(all_probs)
            metrics = self._multiclass_metrics(y_true, y_pred, y_prob)

        self._print_results(metrics)
        return metrics

    # ── metric helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _binary_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
        }

    def _multiclass_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict:
        labels = list(range(self._num_classes))
        names = self._class_names or [str(i) for i in labels]

        metrics: Dict = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=labels, target_names=names,
                output_dict=True, zero_division=0,
            ),
            "class_names": names,
        }

        try:
            metrics["auc_roc_ovr"] = float(
                roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
            )
        except ValueError:
            pass

        return metrics

    def _print_results(self, metrics: Dict) -> None:
        print(f"\n{self._model_name} Test Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")

        if self._mode == "binary":
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1_score']:.4f}")
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        else:
            print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
            print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
            if "auc_roc_ovr" in metrics:
                print(f"  AUC-ROC (OVR): {metrics['auc_roc_ovr']:.4f}")
            report = metrics.get("classification_report", {})
            print("  Per-class F1:")
            for name in self._class_names or []:
                entry = report.get(name, {})
                if isinstance(entry, dict) and "f1-score" in entry:
                    f1 = entry["f1-score"]
                    sup = int(entry.get("support", 0))
                    print(f"    {name:20s}  F1={f1:.4f}  support={sup}")
