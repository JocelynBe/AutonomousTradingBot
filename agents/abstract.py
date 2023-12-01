from abc import abstractmethod
from typing import List

from exchange_api.contracts import Candle, Order, Portfolio
from models.abstract import AbstractModelWrapper


class AbstractAgent:
    model_wrapper: AbstractModelWrapper
    fee: float

    @abstractmethod
    def warmup(self, warmup_candles: List[Candle]) -> None:
        pass

    @abstractmethod
    def update_and_decide(
        self,
        latest_candle: Candle,
        portfolio: Portfolio,
    ) -> List[Order]:
        pass
