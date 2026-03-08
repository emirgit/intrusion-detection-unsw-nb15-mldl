"""ModelRepository — encapsulated registry of ML/DL models with a public prediction API."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, METRICS_FILE, PLACEHOLDER_MODELS, ATTACK_CATEGORIES


class ModelRepository:
    """Centralised, fully encapsulated model store.

    Public API
    ----------
    get_available_models() -> List[str]
    set_active_model(model_id)
    get_active_model() -> str
    predict(model_id, features) -> Dict
    predict_all(features) -> Dict[str, Dict]
    get_offline_metrics() -> Dict
    """

    def __init__(self) -> None:
        self._models: Dict[str, object] = {}
        self._active_model: Optional[str] = None
        self._offline_metrics: Dict = {}
        self._load_models()
        self._load_offline_metrics()

    # ── public API ────────────────────────────────────────────────────────

    def get_available_models(self) -> List[str]:
        return list(self._models.keys())

    def set_active_model(self, model_id: str) -> None:
        if model_id not in self._models:
            raise ValueError(f"Unknown model: {model_id}")
        self._active_model = model_id

    def get_active_model(self) -> str:
        return self._active_model

    def predict(self, model_id: str, features: pd.DataFrame) -> Dict:
        """Run inference with the specified model.

        Returns dict with keys: prediction (0/1), probability (float),
        confidence (str), predicted_label (str).
        """
        if model_id not in self._models:
            raise ValueError(f"Unknown model: {model_id}")

        model = self._models[model_id]

        if model is None:
            return self._placeholder_predict(model_id, features)

        return self._real_predict(model, model_id, features)

    def predict_all(self, features: pd.DataFrame) -> Dict[str, Dict]:
        """Run every loaded model on the same feature row."""
        return {mid: self.predict(mid, features) for mid in self._models}

    def get_offline_metrics(self) -> Dict:
        return dict(self._offline_metrics)

    # ── loading ───────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Try to load real serialized models; fall back to placeholders."""
        for model_id in PLACEHOLDER_MODELS:
            joblib_path = MODELS_DIR / f"{model_id}.joblib"
            if joblib_path.exists():
                try:
                    from joblib import load as jl_load
                    self._models[model_id] = jl_load(joblib_path)
                except Exception:
                    self._models[model_id] = None
            else:
                self._models[model_id] = None  # placeholder stub

        if self._models:
            self._active_model = list(self._models.keys())[0]

    def _load_offline_metrics(self) -> None:
        if METRICS_FILE.exists():
            with open(METRICS_FILE) as f:
                self._offline_metrics = json.load(f)

    # ── inference helpers ─────────────────────────────────────────────────

    @staticmethod
    def _placeholder_predict(model_id: str, features: pd.DataFrame) -> Dict:
        """Return a realistic-looking random prediction (UI-phase stub)."""
        rng = random.Random()

        if "isolation_forest" in model_id:
            prob = rng.betavariate(2, 5)
        elif "gradient_boosting" in model_id:
            prob = rng.betavariate(1.5, 3)
        else:
            prob = rng.random()

        prediction = 1 if prob >= 0.5 else 0
        return {
            "prediction": prediction,
            "probability": round(prob, 4),
            "confidence": _confidence_label(prob),
            "predicted_label": "attack" if prediction == 1 else "normal",
        }

    @staticmethod
    def _real_predict(model, model_id: str, features: pd.DataFrame) -> Dict:
        """Run a real sklearn / DL model."""
        X = features.values if isinstance(features, pd.DataFrame) else features

        if hasattr(model, "predict_proba") and "isolation" not in model_id:
            prob = float(model.predict_proba(X)[0][1])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(X)[0])
            prob = 1.0 / (1.0 + np.exp(-score))
        else:
            pred = int(model.predict(X)[0])
            prob = 1.0 if pred == 1 else 0.0

        prediction = 1 if prob >= 0.5 else 0
        return {
            "prediction": prediction,
            "probability": round(prob, 4),
            "confidence": _confidence_label(prob),
            "predicted_label": "attack" if prediction == 1 else "normal",
        }


def _confidence_label(prob: float) -> str:
    dist = abs(prob - 0.5) * 2  # 0 = uncertain, 1 = certain
    if dist >= 0.8:
        return "high"
    if dist >= 0.4:
        return "medium"
    return "low"
