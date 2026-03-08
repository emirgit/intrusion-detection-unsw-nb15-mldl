"""UNSWDataLoader — loads and preprocesses UNSW-NB15 parquet data for PyTorch.

Supports two classification modes:
  - binary:      target = label column (0 / 1)
  - multi_class: target = attack_cat column encoded to integers (10 classes)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump, load

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train.config import (
    TRAIN_PARQUET, TEST_PARQUET,
    LABEL_COL, ATTACK_CAT_COL, ID_COL,
    CATEGORICAL_COLS, PREPROCESSOR_PATH,
    BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED, NUM_WORKERS,
)


class UNSWDataLoader:
    """Loads, preprocesses, and serves UNSW-NB15 data as PyTorch DataLoaders."""

    def __init__(self, mode: str = "binary") -> None:
        if mode not in ("binary", "multi_class"):
            raise ValueError(f"mode must be 'binary' or 'multi_class', got '{mode}'")
        self._mode = mode
        self._scaler = StandardScaler()
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._attack_cat_encoder = LabelEncoder()
        self._feature_columns: list = []
        self._is_fitted = False

    # ── public properties ───────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        return len(self._feature_columns)

    @property
    def num_classes(self) -> int:
        if self._mode == "binary":
            return 1
        return len(self._attack_cat_encoder.classes_)

    @property
    def class_names(self) -> List[str]:
        if self._mode == "binary":
            return ["normal", "attack"]
        return list(self._attack_cat_encoder.classes_)

    # ── data loading ────────────────────────────────────────────────────────

    def load_and_preprocess(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load parquet files and return preprocessed arrays."""
        print(f"Loading training data from {TRAIN_PARQUET}")
        train_df = pd.read_parquet(TRAIN_PARQUET)
        print(f"  {len(train_df)} rows, {len(train_df.columns)} columns")

        print(f"Loading test data from {TEST_PARQUET}")
        test_df = pd.read_parquet(TEST_PARQUET)
        print(f"  {len(test_df)} rows, {len(test_df.columns)} columns")

        X_train, y_train = self._prepare(train_df, fit=True)
        X_test, y_test = self._prepare(test_df, fit=False)

        print(f"Mode: {self._mode} | Features: {self.n_features}")
        if self._mode == "binary":
            print(
                f"Train: {len(X_train)} (attack={int(y_train.sum())}, "
                f"normal={int(len(y_train) - y_train.sum())})"
            )
            print(
                f"Test:  {len(X_test)} (attack={int(y_test.sum())}, "
                f"normal={int(len(y_test) - y_test.sum())})"
            )
        else:
            print(f"Train: {len(X_train)} | Test: {len(X_test)}")
            print(f"Classes ({self.num_classes}): {self.class_names}")

        return X_train, y_train, X_test, y_test

    # ── DataLoader creation ─────────────────────────────────────────────────

    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = BATCH_SIZE,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train / validation / test DataLoaders."""
        train_idx, val_idx = self._split_indices(len(X_train))

        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        return (
            self._make_loader(X_tr, y_tr, batch_size, shuffle=True),
            self._make_loader(X_val, y_val, batch_size, shuffle=False),
            self._make_loader(X_test, y_test, batch_size, shuffle=False),
        )

    def create_autoencoder_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = BATCH_SIZE,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Autoencoder loaders — training set contains only normal samples."""
        train_idx, val_idx = self._split_indices(len(X_train))

        # Training: normal samples only
        normal_mask = y_train[train_idx] == 0
        X_tr = X_train[train_idx][normal_mask]
        y_tr = y_train[train_idx][normal_mask]

        X_val, y_val = X_train[val_idx], y_train[val_idx]

        print(f"Autoencoder training on {len(X_tr)} normal samples")

        return (
            self._make_loader(X_tr, y_tr, batch_size, shuffle=True),
            self._make_loader(X_val, y_val, batch_size, shuffle=False),
            self._make_loader(X_test, y_test, batch_size, shuffle=False),
        )

    # ── persistence ─────────────────────────────────────────────────────────

    def save_preprocessor(self) -> None:
        PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scaler": self._scaler,
            "label_encoders": self._label_encoders,
            "feature_columns": self._feature_columns,
        }
        if self._mode == "multi_class" and hasattr(
            self._attack_cat_encoder, "classes_"
        ):
            data["attack_cat_encoder"] = self._attack_cat_encoder
            data["class_names"] = list(self._attack_cat_encoder.classes_)
        dump(data, PREPROCESSOR_PATH)
        print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    def load_preprocessor(self) -> None:
        data = load(PREPROCESSOR_PATH)
        self._scaler = data["scaler"]
        self._label_encoders = data["label_encoders"]
        self._feature_columns = data["feature_columns"]
        if "attack_cat_encoder" in data:
            self._attack_cat_encoder = data["attack_cat_encoder"]
        self._is_fitted = True
        print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")

    # ── internal ────────────────────────────────────────────────────────────

    def _prepare(
        self, df: pd.DataFrame, fit: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()

        # ── clean attack_cat ────────────────────────────────────────────────
        if ATTACK_CAT_COL in df.columns:
            df[ATTACK_CAT_COL] = df[ATTACK_CAT_COL].astype(str).str.strip()
            empty_mask = df[ATTACK_CAT_COL].isin(["", "nan", "None", "NaN"])
            df.loc[empty_mask & (df[LABEL_COL] == 0), ATTACK_CAT_COL] = "Normal"
            df.loc[empty_mask & (df[LABEL_COL] == 1), ATTACK_CAT_COL] = "Unknown"

        # ── extract target ──────────────────────────────────────────────────
        if self._mode == "binary":
            y = df[LABEL_COL].values.astype(np.float32)
        else:
            cats = df[ATTACK_CAT_COL].values
            if fit:
                y = self._attack_cat_encoder.fit_transform(cats).astype(np.int64)
            else:
                known = set(self._attack_cat_encoder.classes_)
                y = np.array(
                    [
                        self._attack_cat_encoder.transform([c])[0]
                        if c in known
                        else 0
                        for c in cats
                    ],
                    dtype=np.int64,
                )

        # ── drop non-feature columns ────────────────────────────────────────
        cols_to_drop = [
            c
            for c in [ID_COL, LABEL_COL, ATTACK_CAT_COL]
            if c in df.columns
        ]
        df = df.drop(columns=cols_to_drop)

        # ── encode categorical features ─────────────────────────────────────
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders[col]
                known_cats = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x, _k=known_cats, _le=le: (
                        _le.transform([x])[0] if x in _k else -1
                    )
                )

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if fit:
            self._feature_columns = list(df.columns)
            X = self._scaler.fit_transform(df.values).astype(np.float32)
            self._is_fitted = True
        else:
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self._feature_columns]
            X = self._scaler.transform(df.values).astype(np.float32)

        return X, y

    @staticmethod
    def _split_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.RandomState(RANDOM_SEED).permutation(n)
        val_size = int(n * VALIDATION_SPLIT)
        return indices[val_size:], indices[:val_size]

    @staticmethod
    def _make_loader(
        X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
    ) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
