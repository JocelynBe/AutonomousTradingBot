import logging
from abc import abstractmethod
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.abstract import AbstractLoss, AbstractModelWrapper
from models.metrics.abstract import AbstractMetricsComputer
from models.metrics.contracts import MetricsDict, TrainingMetrics
from trainer.config import TrainingConfig
from trainer.contracts import Dataset
from trainer.cuda import get_device

logger = logging.getLogger(__name__)


class AbstractTrainer:
    optimizer: Optional[Optimizer]
    criterion: AbstractLoss
    metrics_computer: AbstractMetricsComputer
    training_metrics: TrainingMetrics

    def __init__(self, training_config: TrainingConfig, output_dir: str, seed: int):
        self.training_config = training_config
        self.output_dir = output_dir
        self.seed = seed
        self.device = get_device()

    @abstractmethod
    def fit_batch(
        self, model_wrapper: AbstractModelWrapper, batch: Tuple[torch.Tensor, ...]
    ) -> float:
        pass

    @abstractmethod
    def fit_epoch(
        self,
        epoch: int,
        model_wrapper: AbstractModelWrapper,
        train_dataloader: DataLoader,
    ) -> MetricsDict:
        pass

    @abstractmethod
    def eval_model(
        self,
        model_wrapper: AbstractModelWrapper,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        step: int,
    ) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        model_wrapper: AbstractModelWrapper,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> TrainingMetrics:
        pass

    @abstractmethod
    def save_model(self, model_wrapper: AbstractModelWrapper, output_path: str) -> None:
        pass

    @abstractmethod
    def fit_eval_model(
        self,
        model_wrapper: AbstractModelWrapper,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ) -> TrainingMetrics:
        pass
