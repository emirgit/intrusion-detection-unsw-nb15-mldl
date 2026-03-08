"""Training configuration — hyperparameters, paths, and column definitions."""

from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PARQUET = (
    BASE_DIR / "dataset" / "unsw-nb15-dataset"
    / "training-and-testing-parquet" / "UNSW_NB15_training-set.parquet"
)
TEST_PARQUET = (
    BASE_DIR / "dataset" / "unsw-nb15-dataset"
    / "training-and-testing-parquet" / "UNSW_NB15_testing-set.parquet"
)

DL_MODELS_DIR = BASE_DIR / "models" / "dl"
DL_METRICS_DIR = DL_MODELS_DIR / "metrics"
PREPROCESSOR_PATH = DL_MODELS_DIR / "preprocessor.joblib"

# ── Column definitions ──────────────────────────────────────────────────────
LABEL_COL = "label"
ATTACK_CAT_COL = "attack_cat"
ID_COL = "id"
CATEGORICAL_COLS = ["proto", "service", "state"]

# ── Common training settings ────────────────────────────────────────────────
RANDOM_SEED = 42
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
NUM_WORKERS = 2

# ── CNN hyperparameters ─────────────────────────────────────────────────────
CNN_FILTERS = [64, 128]
CNN_KERNEL_SIZE = 3
CNN_DROPOUT = 0.3
CNN_LR = 1e-3
CNN_EPOCHS = 50
CNN_PATIENCE = 7

# ── CNN-LSTM hyperparameters ────────────────────────────────────────────────
CNN_LSTM_FILTERS = [64, 128]
CNN_LSTM_KERNEL_SIZE = 3
CNN_LSTM_HIDDEN_SIZE = 128
CNN_LSTM_NUM_LAYERS = 1
CNN_LSTM_DROPOUT = 0.3
CNN_LSTM_LR = 1e-3
CNN_LSTM_EPOCHS = 200
CNN_LSTM_PATIENCE = 7

# ── Autoencoder hyperparameters ─────────────────────────────────────────────
AE_ENCODER_DIMS = [128, 64, 32]
AE_DROPOUT = 0.2
AE_LR = 1e-3
AE_EPOCHS = 50
AE_PATIENCE = 7
AE_THRESHOLD_PERCENTILE = 95
