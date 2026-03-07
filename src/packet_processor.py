"""PacketProcessor — transforms raw dataset records into ML-ready feature vectors."""

import pandas as pd
import numpy as np
from typing import Dict

from src.config import LABEL_COL, ATTACK_CAT_COL, CATEGORICAL_COLS


class PacketProcessor:
    """Preprocesses raw UNSW-NB15 records for model inference.

    During the placeholder phase this simply encodes categoricals as integer
    codes and drops non-feature columns. When real models with a fitted
    preprocessor (e.g. ColumnTransformer / StandardScaler) are provided, the
    ``transform`` method should delegate to that object instead.
    """

    def __init__(self, preprocessor=None) -> None:
        self._preprocessor = preprocessor  # e.g. a fitted sklearn ColumnTransformer

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return a numeric DataFrame ready for model input."""
        df = raw.copy()

        # Strip label / meta columns
        meta = {}
        if LABEL_COL in df.columns:
            meta[LABEL_COL] = df[LABEL_COL].values
            df = df.drop(columns=[LABEL_COL])
        if ATTACK_CAT_COL in df.columns:
            meta[ATTACK_CAT_COL] = df[ATTACK_CAT_COL].values
            df = df.drop(columns=[ATTACK_CAT_COL])

        # Drop non-numeric helper columns added during simulation
        for col in ("timestamp", "replay_index", "srcip", "dstip"):
            if col in df.columns:
                df = df.drop(columns=[col])

        if self._preprocessor is not None:
            return pd.DataFrame(
                self._preprocessor.transform(df),
                index=df.index,
            )

        # Placeholder encoding: categoricals -> integer codes
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                if hasattr(df[col], "cat"):
                    df[col] = df[col].cat.codes.astype(int)
                else:
                    df[col] = pd.Categorical(df[col]).codes.astype(int)

        # Ensure all columns are numeric
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        return df

    def extract_meta(self, raw: pd.DataFrame) -> Dict:
        """Pull label and attack_cat from a raw row without modifying it."""
        row = raw.iloc[0] if len(raw) == 1 else raw
        return {
            "label": int(row.get(LABEL_COL, -1)) if LABEL_COL in raw.columns else None,
            "attack_cat": str(row.get(ATTACK_CAT_COL, "")) if ATTACK_CAT_COL in raw.columns else None,
        }
