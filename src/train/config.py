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
AE_PREPROCESSOR_PATH = DL_MODELS_DIR / "ae_preprocessor.joblib"

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
CNN_FILTERS = [128, 256, 128]
CNN_KERNEL_SIZE = 5
CNN_DROPOUT = 0.4
CNN_LR = 5e-4
CNN_EPOCHS = 150
CNN_PATIENCE = 12

# ── CNN-LSTM hyperparameters ────────────────────────────────────────────────
CNN_LSTM_FILTERS = [64, 128]
CNN_LSTM_KERNEL_SIZE = 3
CNN_LSTM_HIDDEN_SIZE = 128
CNN_LSTM_NUM_LAYERS = 1
CNN_LSTM_DROPOUT = 0.3
CNN_LSTM_LR = 1e-3
CNN_LSTM_EPOCHS = 150
CNN_LSTM_PATIENCE = 10

# ── Autoencoder hyperparameters ─────────────────────────────────────────────
AE_ENCODER_DIMS = [128, 64, 32]
AE_DROPOUT = 0.2
AE_LR = 1e-3
AE_EPOCHS = 80
AE_PATIENCE = 7
AE_THRESHOLD_PERCENTILE = 95
AE_NOISE_FACTOR = 0.3

# ── FT-Transformer hyperparameters ────────────────────────────────────────
FT_D_TOKEN = 64          # embedding dimension per feature token
FT_N_HEADS = 8           # attention heads
FT_N_LAYERS = 3          # transformer encoder layers
FT_D_FFN = 256           # feed-forward hidden dimension
FT_DROPOUT = 0.2
FT_ATTN_DROPOUT = 0.2
FT_LR = 1e-4
FT_EPOCHS = 150
FT_PATIENCE = 15

# ── TabNet hyperparameters ─────────────────────────────────────────────────
TABNET_N_D = 32           # width of decision embedding (smaller to prevent overfitting)
TABNET_N_A = 32           # width of attention embedding (matched to n_d)
TABNET_N_STEPS = 3        # number of sequential attention steps (fewer = less memorisation)
TABNET_GAMMA = 1.5        # coefficient for feature reusage in attention
TABNET_N_SHARED = 2       # number of shared FC layers in feature transformer
TABNET_N_INDEPENDENT = 2  # number of step-specific FC layers
TABNET_MOMENTUM = 0.02    # batch normalization momentum
TABNET_DROPOUT = 0.3
TABNET_LR = 5e-4
TABNET_EPOCHS = 150
TABNET_PATIENCE = 15
