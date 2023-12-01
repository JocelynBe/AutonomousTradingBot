from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from exchange_api.contracts import Candle
from features.config import FeaturizerConfig
from features.contracts import FeaturizerOutput
from models.contracts import Buffer


@dataclass
class AbstractTarget:
    pass


class AbstractFeaturizer:
    def __init__(self, config: FeaturizerConfig):
        self.config = config

    @abstractmethod
    def predict_last_candle(
        self, buffer: Buffer, last_candle: Candle
    ) -> FeaturizerOutput:
        pass

    @abstractmethod
    def predict(self, buffer: Buffer, candles: List[Candle]) -> FeaturizerOutput:
        # Returns one big Batch with all the data
        pass
