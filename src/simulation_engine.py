"""SimulationEngine — streams UNSW-NB15 records one-by-one to mimic live traffic."""

import random
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from src.config import (
    TESTING_SET_PATH,
    SIMULATED_SOURCE_IPS,
    SIMULATED_DEST_IPS,
)


class SimulationEngine:
    """Reads a dataset partition and emits records sequentially.

    The engine is controlled via ``start`` / ``stop`` and advances one record
    per call to ``next_packet``.  It does **not** spawn its own thread —
    the Streamlit rerun loop drives it.
    """

    def __init__(self, data_path=None, data_frame: Optional[pd.DataFrame] = None) -> None:
        self._data_path = data_path or TESTING_SET_PATH
        self._custom_df = data_frame
        self._data: Optional[pd.DataFrame] = None
        self._index: int = 0
        self._running: bool = False
        self._src_idx: int = 0
        self._dst_idx: int = 0

    # ── lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> bool:
        try:
            if self._custom_df is not None:
                self._data = self._custom_df.reset_index(drop=True)
            else:
                self._data = pd.read_parquet(self._data_path)
            self._index = 0
            return True
        except Exception as exc:
            print(f"SimulationEngine load error: {exc}")
            return False

    def start(self, shuffle: bool = True) -> None:
        if self._data is None:
            self.load()
        if shuffle and self._data is not None:
            self._data = self._data.sample(frac=1, random_state=None).reset_index(drop=True)
        self._index = 0
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self._index = 0
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def total_records(self) -> int:
        return len(self._data) if self._data is not None else 0

    @property
    def current_index(self) -> int:
        return self._index

    # ── streaming ─────────────────────────────────────────────────────────

    def next_packet(self) -> Optional[Dict]:
        """Return the next record as a dict, or None if stopped / exhausted."""
        if not self._running or self._data is None:
            return None

        if self._index >= len(self._data):
            self._index = 0  # loop

        row = self._data.iloc[self._index]
        packet = row.to_dict()
        packet["timestamp"] = datetime.now()
        packet["replay_index"] = self._index
        self._enrich_network_fields(packet)

        self._index += 1
        return packet

    # ── helpers ───────────────────────────────────────────────────────────

    def _enrich_network_fields(self, packet: Dict) -> None:
        """Ensure srcip / dstip are present (dataset has no IP columns)."""
        if not self._valid_ip(packet.get("srcip")):
            packet["srcip"] = SIMULATED_SOURCE_IPS[self._src_idx % len(SIMULATED_SOURCE_IPS)]
            self._src_idx += 1
        if not self._valid_ip(packet.get("dstip")):
            packet["dstip"] = SIMULATED_DEST_IPS[self._dst_idx % len(SIMULATED_DEST_IPS)]
            self._dst_idx += 1
        if not packet.get("service") or str(packet["service"]) in ("", "-", "nan"):
            packet["service"] = "-"

    @staticmethod
    def _valid_ip(val) -> bool:
        if val is None:
            return False
        s = str(val).strip()
        return bool(s) and s.upper() not in ("N/A", "NA", "NAN")
