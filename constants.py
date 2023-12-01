import enum

N_WARM_UP_CANDLES = 1440
MINUTE_IN_MS = 6e4
N_STEPS_CHECK_MISSING = 15
MAX_STEPS_WITH_NO_DATAPOINT = 5
EXPECTED_DELTA_MS = 60000
BINANCE_FEE = 0.075 / 100
CANDLES_FILENAME = "candles.pkl"
FEATURES_FILENAME = "features.pkl"
ORACLE_DECISIONS_FILENAME = "oracle_decisions.pkl"
ALIGNED_SLICES_FILENAME = "aligned_slices.pkl"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
TRAINING_DIR = "training"


class Exchange(enum.Enum):
    BINANCE = "binance"
