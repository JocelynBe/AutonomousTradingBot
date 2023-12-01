from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exchange_api.contracts import Candle
from features.abstract import AbstractFeaturizer
from features.config import FeaturizerConfig
from models.config import ModelConfig, OrderedCurrencies
from models.contracts import DecisionsTensor, ModelInputs, ModelTargets, RawPrediction
from trainer.config import TrainingConfig
from trainer.contracts import ModeKeys


class AbstractLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, training_config: TrainingConfig):
        super().__init__()
        self.training_config = training_config
        self.rnn_warmup_steps = (
            training_config.model_config.encoder_config.rnn_warmup_steps
        )
        self.seq_length = training_config.model_config.encoder_config.seq_length
        self.fee = training_config.fee
        self.ordered_currencies = (
            training_config.model_config.decisioner_config.ordered_currencies
        )
        self.n_currencies = len(self.ordered_currencies)

    @abstractmethod
    def update_fee(self, new_fee: float) -> None:
        pass

    @abstractmethod
    def _forward(
        self, decisions: DecisionsTensor, targets: ModelTargets
    ) -> torch.Tensor:
        pass

    def forward(
        self, decisions: DecisionsTensor, targets: ModelTargets
    ) -> torch.Tensor:
        loss = self._forward(decisions, targets)
        if loss.isnan().sum() > 0:
            raise ValueError(f"Found NaN values: {int(loss.isnan().sum())}")

        return loss


class AbstractModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def forward(self, model_inputs: ModelInputs) -> DecisionsTensor:
        raise NotImplemented


class AbstractModelWrapper:
    MODEL_CLS = AbstractModel
    FEATURIZER_CLS = AbstractFeaturizer

    def __init__(
        self,
        model_config: ModelConfig,
        featurizer_config: FeaturizerConfig,
        ordered_currencies: OrderedCurrencies,
    ):
        self.model_config = model_config
        self.featurizer_config = featurizer_config
        self.ordered_currencies = ordered_currencies
        self.device = torch.device("cpu")

    @property
    @abstractmethod
    def model(self) -> AbstractModel:
        pass

    @property
    @abstractmethod
    def featurizer(self) -> AbstractFeaturizer:
        pass

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        self.device = device

    @abstractmethod
    def warmup(self, candles: List[Candle]) -> None:
        pass

    @abstractmethod
    def predict_dataloader(
        self, dataloader: DataLoader, mode: ModeKeys
    ) -> List[Tuple[DecisionsTensor, ModelTargets]]:
        pass

    @abstractmethod
    def update_and_predict(self, latest_candle: Candle) -> RawPrediction:
        pass

    @classmethod
    @abstractmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> "AbstractModelWrapper":
        pass

    @abstractmethod
    def save_to_json(self, output_path: str) -> None:
        pass
