from abc import abstractmethod
from typing import List

from exchange_api.contracts import Candle, Order, Portfolio, TradeResponse


class AbstractExchangeAPI:
    @abstractmethod
    def get_latest_candle(self) -> Candle:
        pass

    @property
    @abstractmethod
    def fee(self) -> float:
        pass

    @property
    @abstractmethod
    def current_orders(self) -> List[Order]:
        pass

    @property
    @abstractmethod
    def portfolio(self) -> Portfolio:
        pass

    @abstractmethod
    def push_orders(self, orders: List[Order]) -> TradeResponse:
        pass

    @abstractmethod
    def stop_condition(self) -> None:
        pass

    @abstractmethod
    def get_warmup_candles(self, n_warmup_candles: int) -> List[Candle]:
        pass
