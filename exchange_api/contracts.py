from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Optional, Union

import numpy as np
import torch

from contracts import ValueAtTimeX
from utils.io_utils import SerializableDataclass

if TYPE_CHECKING:
    from features.contracts import TorchCandles
    from models.contracts import RawDecision


class Currency(Enum):
    USD = "usd"
    BTC = "btc"


class Status(Enum):
    OK = "ok"
    INSUFFICIENT_FUNDS = "insufficient_funds"


@dataclass
class ExchangePair(SerializableDataclass):
    src_currency: Currency  # Currency used to buy with
    dst_currency: Currency  # Currency being bought

    def __hash__(self) -> int:
        return hash((self.src_currency, self.dst_currency))


@dataclass
class Candle(ExchangePair, ValueAtTimeX):
    open: float
    close: float
    high: float
    low: float
    volume: float

    def to_list(self) -> List[float]:
        return [
            self.timestamp,
            self.open,
            self.close,
            self.high,
            self.low,
            self.volume,
        ]

    def __hash__(self):
        raise NotImplementedError()

    def to_torch(self) -> "TorchCandles":
        return torch.tensor(self.to_list()[1:], dtype=torch.float64)


@dataclass
class BuyTrade(ExchangePair, ValueAtTimeX):
    buy_amount: float  # Of dst_currency
    bid: float

    @property
    def cost(self):
        return self.buy_amount * self.bid


@dataclass
class SellTrade(ExchangePair, ValueAtTimeX):
    sell_amount: float  # Of src_currency
    ask: float

    def __post_init__(self):
        assert self.src_currency != Currency.USD

    @property
    def gain(self):
        return self.sell_amount * self.ask


@dataclass
class Order(ExchangePair, ValueAtTimeX):
    raw_decision: "RawDecision"
    buy: Optional[BuyTrade] = None
    sell: Optional[SellTrade] = None

    def __post_init__(self):
        assert (self.buy is not None) != (self.sell is not None)  # XOR


@dataclass
class TradeResponse:
    status: Status
    trade: Optional[Union[SellTrade, BuyTrade]] = None


@dataclass
class Portfolio:
    holdings: DefaultDict[Currency, float]

    def __getitem__(self, currency: Currency) -> float:
        return self.holdings[currency]

    def __setitem__(self, currency: Currency, amount: float):
        self.holdings[currency] = amount

    def __str__(self) -> str:
        s = "][".join(
            [
                f"{currency.value} = {np.round(amount, 5)}"
                for currency, amount in self.holdings.items()
            ]
        )
        return f"[{s}]"

    def usd_value(self, exchange_rate: Dict[ExchangePair, float], fee: float) -> float:
        values = []
        for currency, amount in self.holdings.items():
            if currency == Currency.USD:
                values.append(amount)
            else:
                conversion_rate = exchange_rate[
                    ExchangePair(src_currency=currency, dst_currency=Currency.USD)
                ]
                value = amount * (1 - fee) * conversion_rate
                values.append(value)
        return sum(values)
