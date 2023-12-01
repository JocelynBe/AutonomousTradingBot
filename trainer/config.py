from dataclasses import dataclass, field

from constants import BINANCE_FEE
from models.config import ModelConfig
from utils.io_utils import SerializableDataclass


@dataclass
class OptimizerConfig(SerializableDataclass):
    optimizer: str = "Adam"
    learning_rate: float = 1e-4  # ??? Should I decrease or increase ???
    batch_size: int = 16
    n_epochs: int = 100
    num_workers: int = 4


@dataclass
class TrainingConfig(SerializableDataclass):
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # Fees HPs
    fee_schedule: str = "super_smooth"
    fee: float = BINANCE_FEE

    # Other HPs
    train_test_ratio: float = 0.95
    val_frequency: int = 5
