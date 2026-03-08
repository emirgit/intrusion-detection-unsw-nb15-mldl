"""Unified entry point for training all DL models on UNSW-NB15.

Usage
-----
    # Train everything (all architectures, all modes)
    python -m src.train.run_training

    # Single architecture, all applicable modes
    python -m src.train.run_training --arch cnn
    python -m src.train.run_training --arch cnn_lstm
    python -m src.train.run_training --arch autoencoder

    # Specific architecture + mode
    python -m src.train.run_training --arch cnn --mode binary
    python -m src.train.run_training --arch cnn_lstm --mode multi_class

    # Each sub-module can also be invoked directly
    python -m src.train.cnn.binary.train
    python -m src.train.cnn.multi_class.train
    python -m src.train.cnn_lstm.binary.train
    python -m src.train.cnn_lstm.multi_class.train
    python -m src.train.autoencoder.binary.train
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.data_loader import UNSWDataLoader
from src.train.config import DL_MODELS_DIR, AE_PREPROCESSOR_PATH

from src.train.cnn.binary.train import CNNBinaryTrainer
from src.train.cnn.multi_class.train import CNNMultiClassTrainer
from src.train.cnn_lstm.binary.train import CNNLSTMBinaryTrainer
from src.train.cnn_lstm.multi_class.train import CNNLSTMMultiClassTrainer
from src.train.autoencoder.binary.train import AutoencoderBinaryTrainer

# ── task registry ───────────────────────────────────────────────────────────
# (architecture, mode) -> (TrainerClass, is_autoencoder)
TASKS = {
    ("cnn", "binary"):          (CNNBinaryTrainer, False),
    ("cnn", "multi_class"):     (CNNMultiClassTrainer, False),
    ("cnn_lstm", "binary"):     (CNNLSTMBinaryTrainer, False),
    ("cnn_lstm", "multi_class"):(CNNLSTMMultiClassTrainer, False),
    ("autoencoder", "binary"):  (AutoencoderBinaryTrainer, True),
}


def _build_task_list(arch: str, mode: str):
    """Return list of (arch, mode, TrainerClass, is_ae) matching the request."""
    tasks = []
    for (a, m), (cls, is_ae) in TASKS.items():
        if arch not in ("all", a):
            continue
        if mode not in ("all", m):
            continue
        tasks.append((a, m, cls, is_ae))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DL models for network IDS")
    parser.add_argument(
        "--arch",
        choices=["cnn", "cnn_lstm", "autoencoder", "all"],
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "multi_class", "all"],
        default="all",
    )
    args = parser.parse_args()

    tasks = _build_task_list(args.arch, args.mode)

    if not tasks:
        print("No valid tasks for the given arch/mode combination.")
        return

    # Group by mode so we load data once per mode
    binary_tasks = [(a, cls, ae) for a, m, cls, ae in tasks if m == "binary"]
    multi_tasks = [(a, cls, ae) for a, m, cls, ae in tasks if m == "multi_class"]

    if binary_tasks:
        print("\n" + "=" * 60)
        print("BINARY CLASSIFICATION")
        print("=" * 60)

        # Separate classifier and autoencoder tasks (different scalers)
        clf_tasks = [(a, cls) for a, cls, ae in binary_tasks if not ae]
        ae_tasks = [(a, cls) for a, cls, ae in binary_tasks if ae]

        if clf_tasks:
            dl = UNSWDataLoader(mode="binary")
            X_tr, y_tr, X_te, y_te = dl.load_and_preprocess()
            dl.save_preprocessor()
            class_weights = dl.compute_class_weights(y_tr)

            for arch_name, TrainerClass in clf_tasks:
                t_dl, v_dl, te_dl = dl.create_dataloaders(
                    X_tr, y_tr, X_te, y_te
                )
                trainer = TrainerClass(
                    n_features=dl.n_features,
                    num_classes=dl.num_classes,
                    class_names=dl.class_names,
                    class_weights=class_weights,
                )
                trainer.train(t_dl, v_dl)
                metrics = trainer.evaluate(te_dl)
                trainer.save_model()
                trainer.save_metrics(metrics)

        if ae_tasks:
            # Autoencoder uses MinMaxScaler [0,1] for stable reconstruction
            ae_dl = UNSWDataLoader(mode="binary", use_minmax=True,
                                   preprocessor_path=AE_PREPROCESSOR_PATH)
            X_tr, y_tr, X_te, y_te = ae_dl.load_and_preprocess()
            ae_dl.save_preprocessor()

            for arch_name, TrainerClass in ae_tasks:
                t_dl, v_dl, te_dl = ae_dl.create_autoencoder_dataloaders(
                    X_tr, y_tr, X_te, y_te
                )
                trainer = TrainerClass(
                    n_features=ae_dl.n_features,
                    num_classes=ae_dl.num_classes,
                    class_names=ae_dl.class_names,
                )
                trainer.train(t_dl, v_dl)
                trainer.compute_threshold(v_dl)
                metrics = trainer.evaluate(te_dl)
                trainer.save_model()
                trainer.save_metrics(metrics)

    if multi_tasks:
        print("\n" + "=" * 60)
        print("MULTI-CLASS CLASSIFICATION")
        print("=" * 60)
        dl = UNSWDataLoader(mode="multi_class")
        X_tr, y_tr, X_te, y_te = dl.load_and_preprocess()
        dl.save_preprocessor()
        class_weights = dl.compute_class_weights(y_tr)

        for arch_name, TrainerClass, _ in multi_tasks:
            t_dl, v_dl, te_dl = dl.create_dataloaders(X_tr, y_tr, X_te, y_te)

            trainer = TrainerClass(
                n_features=dl.n_features,
                num_classes=dl.num_classes,
                class_names=dl.class_names,
                class_weights=class_weights,
            )
            trainer.train(t_dl, v_dl)
            metrics = trainer.evaluate(te_dl)
            trainer.save_model()
            trainer.save_metrics(metrics)

    # Clean up temporary best checkpoint files
    for best_pt in DL_MODELS_DIR.glob("*_best.pt"):
        best_pt.unlink()
        print(f"Removed checkpoint: {best_pt}")

    print(f"\nAll done. Models saved to {ROOT / 'models' / 'dl'}")


if __name__ == "__main__":
    main()
