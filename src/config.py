"""Central configuration — all paths, constants, thresholds, and simulation parameters."""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATASET_DIR = BASE_DIR / "dataset" / "unsw-nb15-dataset"

TESTING_SET_PATH = DATA_RAW / "testing-set.parquet"
X_TEST_PATH = DATA_PROCESSED / "X_test.csv"
Y_TEST_PATH = DATA_PROCESSED / "y_test.csv"

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = MODELS_DIR / "metrics"
METRICS_FILE = METRICS_DIR / "placeholder_metrics.json"

# ── Simulation parameters ─────────────────────────────────────────────────────
DEFAULT_SPEED = 1.0          # seconds per packet
MIN_SPEED = 0.1
MAX_SPEED = 3.0
SPEED_STEP = 0.1

RECENT_PACKETS_COUNT = 20    # live traffic feed rows
MAX_ALERTS = 100             # alert history cap

# ── IP pools for simulated network fields ─────────────────────────────────────
SIMULATED_SOURCE_IPS = [
    "192.168.1.10", "192.168.1.24", "192.168.1.37", "192.168.1.58",
    "192.168.2.17", "192.168.2.33", "10.1.10.42", "10.1.10.84",
]

SIMULATED_DEST_IPS = [
    "10.0.0.10", "10.0.0.15", "10.0.0.21", "10.0.1.5",
]

# ── Alert severity thresholds (based on confidence) ───────────────────────────
SEVERITY_CRITICAL_THRESHOLD = 0.90
SEVERITY_HIGH_THRESHOLD = 0.75
SEVERITY_MEDIUM_THRESHOLD = 0.50

# ── Prediction threshold ─────────────────────────────────────────────────────
ATTACK_PROBABILITY_THRESHOLD = 0.5

# ── Feature columns (all except label / attack_cat) ──────────────────────────
LABEL_COL = "label"
ATTACK_CAT_COL = "attack_cat"
CATEGORICAL_COLS = ["proto", "service", "state"]

# ── Placeholder model IDs ────────────────────────────────────────────────────
PLACEHOLDER_MODELS = [
    "random_forest",
    "gradient_boosting",
    "extra_trees",
    "sgd_classifier",
    "isolation_forest",
]

# ── UNSW-NB15 attack categories ──────────────────────────────────────────────
ATTACK_CATEGORIES = [
    "Backdoor", "Shellcode", "Exploits", "DoS", "Worms",
    "Reconnaissance", "Fuzzers", "Analysis", "Generic",
]

# ── Dashboard UI ──────────────────────────────────────────────────────────────
PAGE_TITLE = "Network IDS Dashboard"
PAGE_ICON = "IDS"
