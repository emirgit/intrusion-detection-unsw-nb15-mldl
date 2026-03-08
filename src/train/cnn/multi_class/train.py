"""Train CNN for multi-class classification on UNSW-NB15."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.data_loader import UNSWDataLoader
from src.train.base_trainer import ClassifierTrainer
from src.train.cnn.model import CNNClassifier
from src.train.config import CNN_LR, CNN_EPOCHS, CNN_PATIENCE


class CNNMultiClassTrainer(ClassifierTrainer):
    def __init__(self, n_features, num_classes=10, class_names=None, device=None, class_weights=None):
        model = CNNClassifier(input_size=n_features, num_classes=num_classes)
        super().__init__(
            model=model,
            model_name="cnn_multi_class",
            mode="multi_class",
            num_classes=num_classes,
            learning_rate=CNN_LR,
            epochs=CNN_EPOCHS,
            patience=CNN_PATIENCE,
            class_names=class_names,
            device=device,
            class_weights=class_weights,
        )


def main():
    loader = UNSWDataLoader(mode="multi_class")
    X_train, y_train, X_test, y_test = loader.load_and_preprocess()
    loader.save_preprocessor()

    train_dl, val_dl, test_dl = loader.create_dataloaders(
        X_train, y_train, X_test, y_test
    )

    class_weights = loader.compute_class_weights(y_train)
    trainer = CNNMultiClassTrainer(
        n_features=loader.n_features,
        num_classes=loader.num_classes,
        class_names=loader.class_names,
        class_weights=class_weights,
    )
    trainer.train(train_dl, val_dl)
    metrics = trainer.evaluate(test_dl)
    trainer.save_model()
    trainer.save_metrics(metrics)


if __name__ == "__main__":
    main()
