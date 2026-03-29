"""ModelRepository — encapsulated registry of ML/DL models with a public prediction API."""

import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from src.config import MODELS_DIR, METRICS_FILE, PLACEHOLDER_MODELS, ATTACK_CATEGORIES, BASE_DIR

# Optional PyTorch imports
try:
    import torch
    from src.train.autoencoder.model import Autoencoder
    from src.train.cnn.model import CNNClassifier
    from src.train.cnn_lstm.model import CNNLSTMClassifier
    TORCH_AVAILABLE = True
    MODEL_CLASSES = {
        "Autoencoder": Autoencoder,
        "CNNClassifier": CNNClassifier,
        "CNNLSTMClassifier": CNNLSTMClassifier
    }
except ImportError:
    TORCH_AVAILABLE = False


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
        self._models: Dict[str, Any] = {}
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
        """Run inference with the specified model."""
        if model_id not in self._models:
            raise ValueError(f"Unknown model: {model_id}")

        model_info = self._models[model_id]

        if model_info is None:
            return self._placeholder_predict(model_id, features)

        return self._real_predict(model_info, model_id, features)

    def predict_all(self, features: pd.DataFrame) -> Dict[str, Dict]:
        """Run every loaded model on the same feature row."""
        return {mid: self.predict(mid, features) for mid in self._models}

    def get_offline_metrics(self) -> Dict:
        return dict(self._offline_metrics)

    # ── loading ───────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Try to load real serialized models; fall back to placeholders."""
        dl_dir = MODELS_DIR / "dl"
        
        # Load multi-class class names from preprocessor if available
        multi_class_names = []
        prep_path = dl_dir / "preprocessor.joblib"
        if prep_path.exists():
            prep_data = joblib.load(prep_path)
            if "class_names" in prep_data:
                multi_class_names = prep_data["class_names"]

        # 1. Load PyTorch models
        if TORCH_AVAILABLE and dl_dir.exists():
            for pt_file in dl_dir.glob("*.pt"):
                if pt_file.name.endswith("_best.pt"):
                    continue
                model_id = pt_file.stem
                try:
                    checkpoint = torch.load(pt_file, map_location="cpu", weights_only=True)
                    model_class_name = checkpoint["model_class"]
                    model_config = checkpoint["model_config"]
                    
                    if model_class_name in MODEL_CLASSES:
                        model = MODEL_CLASSES[model_class_name](**model_config)
                        model.load_state_dict(checkpoint["model_state_dict"])
                        model.eval()
                        
                        is_multi_class = ("multi_class" in model_id)
                        
                        self._models[model_id] = {
                            "type": "pytorch",
                            "model": model,
                            "class": model_class_name,
                            "threshold": checkpoint.get("threshold", 0.1),
                            "class_names": multi_class_names if is_multi_class else []
                        }
                except Exception as e:
                    warnings.warn(f"Failed to load PyTorch model {pt_file}: {e}")

        # 2. Load sklearn models (.joblib from root, .pkl/.joblib from ml/)
        for joblib_path in MODELS_DIR.glob("*.joblib"):
            model_id = joblib_path.stem
            if "preprocessor" in model_id:
                continue
            try:
                from joblib import load as jl_load
                self._models[model_id] = {"type": "sklearn", "model": jl_load(joblib_path)}
            except Exception:
                pass

        # 2b. Load sklearn models from models/ml/ directory (.pkl and .joblib)
        ml_dir = MODELS_DIR / "ml"
        if ml_dir.exists():
            import pickle
            for pkl_path in list(ml_dir.glob("*.pkl")) + list(ml_dir.glob("*.joblib")):
                model_id = pkl_path.stem
                if "preprocessor" in model_id:
                    continue
                if model_id in self._models:
                    continue  # skip if already loaded
                try:
                    if pkl_path.suffix == ".joblib":
                        from joblib import load as jl_load
                        self._models[model_id] = {"type": "sklearn", "model": jl_load(pkl_path)}
                    else:
                        with open(pkl_path, "rb") as f:
                            self._models[model_id] = {"type": "sklearn", "model": pickle.load(f)}
                except Exception:
                    pass
                
        # 3. Fallback to placeholders if missing
        if not self._models:
            for model_id in PLACEHOLDER_MODELS:
                self._models[model_id] = None

        if self._models:
            # Prefer DL models if available
            dl_keys = [k for k in self._models.keys() if "cnn" in k or "autoencoder" in k]
            self._active_model = dl_keys[0] if dl_keys else list(self._models.keys())[0]

    def _load_offline_metrics(self) -> None:
        metrics_dir = MODELS_DIR / "dl" / "metrics"
        if metrics_dir.exists():
            for metric_file in metrics_dir.glob("*_metrics.json"):
                model_id = metric_file.name.replace("_metrics.json", "")
                with open(metric_file) as f:
                    self._offline_metrics[model_id] = json.load(f)
                    
        if METRICS_FILE.exists():
            with open(METRICS_FILE) as f:
                self._offline_metrics["placeholders"] = json.load(f)

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
            "predicted_class": "attack" if prediction == 1 else "normal"
        }

    @staticmethod
    def _real_predict(model_info: Any, model_id: str, features: pd.DataFrame) -> Dict:
        """Run a real sklearn / DL model."""
        X = features.values if isinstance(features, pd.DataFrame) else features
        
        if isinstance(model_info, dict) and model_info.get("type") == "pytorch":
            model = model_info["model"]
            model_class = model_info["class"]
            class_names = model_info.get("class_names", [])
            
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                if model_class == "Autoencoder":
                    recon = model(X_tensor)
                    mse = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()
                    error = float(mse[0])
                    threshold = model_info.get("threshold", 0.1)
                    
                    prediction = 1 if error > threshold else 0
                    if prediction == 1:
                        prob = min(0.5 + 0.5 * ((error - threshold) / (threshold + 1e-9)), 1.0)
                    else:
                        prob = max(0.5 * (error / (threshold + 1e-9)), 0.0)
                    predicted_class = "attack" if prediction == 1 else "normal"
                        
                else:
                    logits = model(X_tensor)
                    if len(logits.shape) == 1 or logits.shape[1] == 1:
                        # Binary
                        probs = torch.sigmoid(logits).numpy()
                        prob = float(probs[0])
                        prediction = 1 if prob >= 0.5 else 0
                        predicted_class = "attack" if prediction == 1 else "normal"
                    else:
                        # Multi-class
                        probs = torch.softmax(logits, dim=1).numpy()[0]
                        pred_class_idx = int(np.argmax(probs))
                        
                        # Determine class name first
                        if class_names and pred_class_idx < len(class_names):
                            predicted_class = class_names[pred_class_idx]
                        else:
                            # Fallback if class names missing
                            predicted_class = "normal" if pred_class_idx == 0 else "attack"
                        
                        # 'Normal' (case-insensitive) means no attack
                        prediction = 0 if predicted_class.lower() == "normal" else 1
                        prob = float(probs[pred_class_idx])

                        if prediction == 0:
                            prob = 1.0 - prob # Convert to attack probability
                            
            return {
                "prediction": prediction,
                "probability": round(prob, 4),
                "confidence": _confidence_label(prob),
                "predicted_label": "attack" if prediction == 1 else "normal",
                "predicted_class": predicted_class
            }

        else:
            model = model_info["model"] if isinstance(model_info, dict) else model_info
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
                "predicted_class": "attack" if prediction == 1 else "normal"
            }


def _confidence_label(prob: float) -> str:
    dist = abs(prob - 0.5) * 2
    if dist >= 0.8:
        return "high"
    if dist >= 0.4:
        return "medium"
    return "low"
