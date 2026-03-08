"""PacketProcessor — transforms raw dataset records into ML-ready feature vectors."""

import pandas as pd
import numpy as np
from typing import Dict
from joblib import load

from src.config import LABEL_COL, ATTACK_CAT_COL, CATEGORICAL_COLS, BASE_DIR


class PacketProcessor:
    """Preprocesses raw UNSW-NB15 records for model inference."""

    def __init__(self, preprocessor_path=None) -> None:
        self._preprocessor_data = None
        self._ae_preprocessor_data = None
        
        dl_dir = BASE_DIR / "models" / "dl"
        
        # Load main preprocessor
        prep_path = dl_dir / "preprocessor.joblib"
        if prep_path.exists():
            self._preprocessor_data = load(prep_path)
            
        # Load autoencoder preprocessor (if different)
        ae_prep_path = dl_dir / "ae_preprocessor.joblib"
        if ae_prep_path.exists():
            self._ae_preprocessor_data = load(ae_prep_path)

    def transform(self, raw: pd.DataFrame, model_id: str = None) -> pd.DataFrame:
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
        for col in ("timestamp", "replay_index", "srcip", "dstip", "id"):
            if col in df.columns:
                df = df.drop(columns=[col])

        prep_data = self._preprocessor_data
        if model_id and "autoencoder" in model_id.lower() and self._ae_preprocessor_data:
            prep_data = self._ae_preprocessor_data

        if prep_data is not None:
            scaler = prep_data["scaler"]
            label_encoders = prep_data["label_encoders"]
            feature_columns = prep_data["feature_columns"]

            # encode categorical features
            for col in CATEGORICAL_COLS:
                if col in df.columns and col in label_encoders:
                    le = label_encoders[col]
                    known_cats = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x, _k=known_cats, _le=le: (
                            _le.transform([x])[0] if x in _k else -1
                        )
                    )

            df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_columns]
            
            # transform and return
            X = scaler.transform(df.values).astype(np.float32)
            return pd.DataFrame(X, columns=feature_columns, index=df.index)

        # Placeholder encoding fallback
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                if hasattr(df[col], "cat"):
                    df[col] = df[col].cat.codes.astype(int)
                else:
                    df[col] = pd.Categorical(df[col]).codes.astype(int)

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        return df

    def extract_meta(self, raw: pd.DataFrame) -> Dict:
        """Pull label and attack_cat from a raw row without modifying it."""
        row = raw.iloc[0] if len(raw) == 1 else raw
        return {
            "label": int(row.get(LABEL_COL, -1)) if LABEL_COL in raw.columns else None,
            "attack_cat": str(row.get(ATTACK_CAT_COL, "")) if ATTACK_CAT_COL in raw.columns else None,
        }
