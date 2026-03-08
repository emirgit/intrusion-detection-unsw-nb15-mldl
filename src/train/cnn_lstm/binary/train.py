"""Train CNN-LSTM for binary classification on UNSW-NB15."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.data_loader import UNSWDataLoader
from src.train.base_trainer import ClassifierTrainer
from src.train.cnn_lstm.model import CNNLSTMClassifier
from src.train.config import CNN_LSTM_LR, CNN_LSTM_EPOCHS, CNN_LSTM_PATIENCE


class CNNLSTMBinaryTrainer(ClassifierTrainer):
    def __init__(self, n_features, num_classes=1, class_names=None, device=None, class_weights=None):
        model = CNNLSTMClassifier(input_size=n_features, num_classes=1)
        super().__init__(
            model=model,
            model_name="cnn_lstm_binary",
            mode="binary",
            num_classes=1,
            learning_rate=CNN_LSTM_LR,
            epochs=CNN_LSTM_EPOCHS,
            patience=CNN_LSTM_PATIENCE,
            class_names=["normal", "attack"],
            enable_grad_clip=True,
            device=device,
            class_weights=class_weights,
        )


def main():
    loader = UNSWDataLoader(mode="binary")
    X_train, y_train, X_test, y_test = loader.load_and_preprocess()
    loader.save_preprocessor()

    train_dl, val_dl, test_dl = loader.create_dataloaders(
        X_train, y_train, X_test, y_test
    )

    trainer = CNNLSTMBinaryTrainer(n_features=loader.n_features)
    trainer.train(train_dl, val_dl)
    metrics = trainer.evaluate(test_dl)
    trainer.save_model()
    trainer.save_metrics(metrics)


if __name__ == "__main__":
    main()
