"""UNSWDataLoader — loads and preprocesses UNSW-NB15 parquet data for PyTorch.

Supports two classification modes:
  - binary:      target = label column (0 / 1)
  - multi_class: target = attack_cat column encoded to integers (10 classes)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
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

    def __init__(self, mode: str = "binary", use_minmax: bool = False,
                 preprocessor_path: str = None) -> None:
        if mode not in ("binary", "multi_class"):
            raise ValueError(f"mode must be 'binary' or 'multi_class', got '{mode}'")
        self._mode = mode
        self._scaler = MinMaxScaler() if use_minmax else StandardScaler()
        self._attack_cat_encoder = LabelEncoder()
        self._onehot_columns: list = []
        self._selected_features: list = []
        self._feature_columns: list = []
        self._is_fitted = False
        self._last_train_labels = None
        self._preprocessor_path = preprocessor_path or PREPROCESSOR_PATH

    # ── public properties ───────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        return len(self._feature_columns)

    @property
    def last_train_labels(self) -> np.ndarray:
        """Training labels from the most recent create_dataloaders call
        (post-SMOTE if oversampling was applied)."""
        return self._last_train_labels

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

    def compute_class_weights(self, y_train: np.ndarray) -> torch.Tensor:
        """Compute class weights for balanced training.

        Binary: standard inverse-frequency pos_weight.
        Multi-class: sqrt-dampened balanced weights to avoid overcorrecting
        for extremely rare classes.
        """
        if self._mode == "binary":
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
            return pos_weight
        else:
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            weights = np.sqrt(weights)
            weights = weights / weights.min()
            return torch.tensor(weights, dtype=torch.float32)

    # ── DataLoader creation ─────────────────────────────────────────────────

    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = BATCH_SIZE,
        smote_strategy=None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train / validation / test DataLoaders.

        If `smote_strategy` is provided and mode is multi_class, SMOTE is
        applied to the training split only (val/test stay untouched).
        `smote_strategy` accepts the same forms as imblearn's
        `sampling_strategy` argument. Class names (strings) are also accepted
        and converted to encoded integer labels.
        """
        train_idx, val_idx = self._split_indices(len(X_train))

        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        if smote_strategy is not None and self._mode == "multi_class":
            X_tr, y_tr = self.apply_smote(X_tr, y_tr, sampling_strategy=smote_strategy)

        # Expose the actual training labels the model will see (post-SMOTE if
        # applied) so class weights can be computed from the real distribution.
        self._last_train_labels = y_tr

        return (
            self._make_loader(X_tr, y_tr, batch_size, shuffle=True),
            self._make_loader(X_val, y_val, batch_size, shuffle=False),
            self._make_loader(X_test, y_test, batch_size, shuffle=False),
        )

    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy="auto",
        k_neighbors: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling. Multi-class only.

        Accepts dict keys as class names (strings) or encoded integer labels.
        Prints class distribution before and after for transparency.
        """
        from imblearn.over_sampling import SMOTE

        if self._mode != "multi_class":
            raise ValueError("SMOTE is only supported for multi_class mode.")

        if isinstance(sampling_strategy, dict):
            classes = list(self._attack_cat_encoder.classes_)
            converted = {}
            for key, target in sampling_strategy.items():
                if isinstance(key, str):
                    if key not in classes:
                        raise ValueError(f"Unknown class name '{key}' in smote_strategy")
                    idx = classes.index(key)
                else:
                    idx = int(key)
                current = int((y == idx).sum())
                if current < target:
                    converted[idx] = int(target)
            sampling_strategy = converted

        print("Class distribution before SMOTE:")
        self._print_class_counts(y)

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_SEED,
            k_neighbors=k_neighbors,
        )
        X_res, y_res = smote.fit_resample(X, y)

        print("Class distribution after SMOTE:")
        self._print_class_counts(y_res)
        print(f"Total samples: {len(y)} -> {len(y_res)}")

        return X_res.astype(np.float32), y_res.astype(np.int64)

    def _print_class_counts(self, y: np.ndarray) -> None:
        for idx, name in enumerate(self._attack_cat_encoder.classes_):
            count = int((y == idx).sum())
            print(f"  {name:20s} {count:>7d}")

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
        path = self._preprocessor_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scaler": self._scaler,
            "onehot_columns": self._onehot_columns,
            "selected_features": self._selected_features,
            "feature_columns": self._feature_columns,
        }
        if self._mode == "multi_class" and hasattr(
            self._attack_cat_encoder, "classes_"
        ):
            data["attack_cat_encoder"] = self._attack_cat_encoder
            data["class_names"] = list(self._attack_cat_encoder.classes_)
        dump(data, path)
        print(f"Preprocessor saved to {path}")

    def load_preprocessor(self) -> None:
        path = self._preprocessor_path
        data = load(path)
        self._scaler = data["scaler"]
        self._onehot_columns = data.get("onehot_columns", [])
        self._selected_features = data.get("selected_features", [])
        self._feature_columns = data["feature_columns"]
        if "attack_cat_encoder" in data:
            self._attack_cat_encoder = data["attack_cat_encoder"]
        self._is_fitted = True
        print(f"Preprocessor loaded from {path}")

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

        # ── one-hot encode categorical features ─────────────────────────────
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str)
        df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns])

        if fit:
            self._onehot_columns = list(df.columns)
        else:
            # Add missing columns as 0, drop extra columns
            for col in self._onehot_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self._onehot_columns]

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # ── feature selection (RF importance + correlation hybrid) ──────
        if fit:
            self._selected_features = self._select_features(df, y)
            print(f"Feature selection: {len(df.columns)} -> {len(self._selected_features)} features")
        df = df[self._selected_features]

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
    def _select_features(df: pd.DataFrame, y: np.ndarray, top_k: int = 50,
                         corr_threshold: float = 0.1) -> list:
        """Select features using RF importance + correlation hybrid (same as ML pipeline)."""
        # Step 1: Correlation with target
        df_with_target = df.copy()
        df_with_target["_target"] = y
        corr = abs(df_with_target.corr()["_target"]).drop("_target")
        corr_features = set(corr[corr > corr_threshold].index)

        # Step 2: RF feature importance (top_k)
        rf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        )
        y_cls = y.astype(int) if y.dtype == np.float32 else y
        rf.fit(df.values, y_cls)
        importances = pd.Series(rf.feature_importances_, index=df.columns)
        rf_features = set(importances.nlargest(top_k).index)

        # Step 3: Union
        selected = sorted(list(corr_features | rf_features))
        return selected

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
