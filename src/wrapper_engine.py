"""WrapperEngine — inference backend that connects PacketProcessor + ModelRepository."""

from typing import Dict, Optional

import pandas as pd

from src.model_repository import ModelRepository
from src.packet_processor import PacketProcessor
from src.alert_system import AlertSystem
from src.config import ATTACK_CAT_COL


class WrapperEngine:
    """Orchestrates preprocessing, prediction, and alert generation.

    The Streamlit layer feeds raw packet dicts in; this class returns
    enriched result dicts ready for rendering.
    """

    def __init__(self) -> None:
        self.model_repo = ModelRepository()
        self.processor = PacketProcessor()
        self.alert_system = AlertSystem()
        self._stats = {
            "packets_processed": 0,
            "attacks_detected": 0,
            "normal_count": 0,
        }

    # ── public API ────────────────────────────────────────────────────────

    def process_packet(self, raw_packet: Dict) -> Dict:
        """Run active-model inference on a single raw packet dict."""
        active_model = self.model_repo.get_active_model()
        if active_model is None:
            return self._empty_result(raw_packet)

        raw_df = pd.DataFrame([raw_packet])
        meta = self.processor.extract_meta(raw_df)
        features = self.processor.transform(raw_df, active_model)
        result = self.model_repo.predict(active_model, features)

        self._stats["packets_processed"] += 1

        attack_cat = meta.get("attack_cat") or ""
        if attack_cat in ("", "Normal", "nan", "None"):
            attack_cat = None

        alert = None
        if result["prediction"] == 1:
            self._stats["attacks_detected"] += 1
            alert = self.alert_system.create_alert(
                attack_category=attack_cat or "Generic",
                confidence=result["probability"],
                source_ip=raw_packet.get("srcip", "N/A"),
                dest_ip=raw_packet.get("dstip", "N/A"),
                service=str(raw_packet.get("service", "-")),
            )
        else:
            self._stats["normal_count"] += 1

        return {
            "timestamp": raw_packet.get("timestamp"),
            "srcip": raw_packet.get("srcip", "N/A"),
            "dstip": raw_packet.get("dstip", "N/A"),
            "service": str(raw_packet.get("service", "-")),
            "dur": raw_packet.get("dur"),
            "prediction": result["predicted_label"],
            "probability": result["probability"],
            "confidence": result["confidence"],
            "true_label": meta.get("label"),
            "attack_type": attack_cat,
            "alert": alert,
            "model_id": active_model,
            "raw_packet": raw_packet,
        }

    def predict_all_models(self, features: pd.DataFrame) -> Dict:
        """Run every model (for benchmarking)."""
        return self.model_repo.predict_all(features)

    def get_stats(self) -> Dict:
        return {
            **self._stats,
            "attack_ratio": (
                self._stats["attacks_detected"] / self._stats["packets_processed"] * 100
                if self._stats["packets_processed"] > 0
                else 0.0
            ),
            "active_model": self.model_repo.get_active_model(),
            "alert_summary": self.alert_system.get_summary(),
        }

    def reset_stats(self) -> None:
        self._stats = {"packets_processed": 0, "attacks_detected": 0, "normal_count": 0}
        self.alert_system.clear()

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(raw_packet: Dict) -> Dict:
        return {
            "timestamp": raw_packet.get("timestamp"),
            "srcip": raw_packet.get("srcip", "N/A"),
            "dstip": raw_packet.get("dstip", "N/A"),
            "service": str(raw_packet.get("service", "-")),
            "dur": raw_packet.get("dur"),
            "prediction": "unknown",
            "probability": 0.0,
            "confidence": "low",
            "true_label": None,
            "attack_type": None,
            "predicted_class": "unknown",
            "alert": None,
            "model_id": None,
            "raw_packet": raw_packet,
        }
