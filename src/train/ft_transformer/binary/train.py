"""Train FT-Transformer for binary classification on UNSW-NB15."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.data_loader import UNSWDataLoader
from src.train.base_trainer import ClassifierTrainer
from src.train.ft_transformer.model import FTTransformerClassifier
from src.train.config import FT_LR, FT_EPOCHS, FT_PATIENCE


class FTTransformerBinaryTrainer(ClassifierTrainer):
    def __init__(self, n_features, num_classes=1, class_names=None, device=None, class_weights=None):
        model = FTTransformerClassifier(input_size=n_features, num_classes=1)
        super().__init__(
            model=model,
            model_name="ft_transformer_binary",
            mode="binary",
            num_classes=1,
            learning_rate=FT_LR,
            epochs=FT_EPOCHS,
            patience=FT_PATIENCE,
            class_names=["normal", "attack"],
            device=device,
            class_weights=class_weights,
            weight_decay=1e-5,
        )


def main():
    loader = UNSWDataLoader(mode="binary")
    X_train, y_train, X_test, y_test = loader.load_and_preprocess()
    loader.save_preprocessor()

    train_dl, val_dl, test_dl = loader.create_dataloaders(
        X_train, y_train, X_test, y_test
    )

    class_weights = loader.compute_class_weights(y_train)
    trainer = FTTransformerBinaryTrainer(
        n_features=loader.n_features,
        class_weights=class_weights,
    )
    trainer.train(train_dl, val_dl)
    metrics = trainer.evaluate(test_dl)
    trainer.save_model()
    trainer.save_metrics(metrics)


if __name__ == "__main__":
    main()
