from typing import List, Optional, Type

from agents.abstract import AbstractAgent
from constants import N_WARM_UP_CANDLES
from exchange_api.contracts import (
    BuyTrade,
    Candle,
    Currency,
    Order,
    Portfolio,
    SellTrade,
)
from models.contracts import RawDecision
from models.wrapper import BaseModelWrapper


class BaseAgent(AbstractAgent):
    MODEL_WRAPPER_CLS: Type[BaseModelWrapper] = BaseModelWrapper

    model_wrapper: BaseModelWrapper
    fee: float

    def __init__(self, model_wrapper: BaseModelWrapper, fee: float):
        self.model_wrapper = model_wrapper
        self.model_wrapper.eval()
        self.fee = fee
        self.latest_candle: Optional[Candle] = None

    def warmup(self, warmup_candles: List[Candle]) -> None:
        encoder_config = self.model_wrapper.model_config.encoder_config
        assert (
            len(warmup_candles)
            == N_WARM_UP_CANDLES + encoder_config.total_warmup_length
        )

        self.model_wrapper.warmup(warmup_candles)
        self.latest_candle = warmup_candles[-1]

    @staticmethod
    def get_buy_order(
        buy_amount: float, latest_candle: Candle, raw_decision: RawDecision
    ) -> Order:
        assert latest_candle.timestamp == raw_decision.timestamp
        order = Order(
            src_currency=Currency.USD,
            dst_currency=Currency.BTC,
            buy=BuyTrade(
                src_currency=Currency.USD,
                dst_currency=Currency.BTC,
                buy_amount=buy_amount,
                bid=latest_candle.close,
                timestamp=raw_decision.timestamp,
            ),
            raw_decision=raw_decision,
            timestamp=raw_decision.timestamp,
        )
        return order

    @staticmethod
    def get_sell_order(
        sell_amount: float, latest_candle: Candle, raw_decision: RawDecision
    ) -> Order:
        assert latest_candle.timestamp == raw_decision.timestamp
        order = Order(
            src_currency=Currency.BTC,
            dst_currency=Currency.USD,
            sell=SellTrade(
                src_currency=Currency.BTC,
                dst_currency=Currency.USD,
                sell_amount=sell_amount,
                ask=latest_candle.close,
                timestamp=raw_decision.timestamp,
            ),
            raw_decision=raw_decision,
            timestamp=raw_decision.timestamp,
        )
        return order

    def make_decision(
        self,
        raw_decision: RawDecision,
        portfolio: Portfolio,
        latest_candle: Candle,
        min_transaction_amount: float = 1.0,
    ) -> List[Order]:
        if raw_decision is None:
            return []

        btc_price = latest_candle.close

        amount_of_usd_to_sell = portfolio[Currency.USD] * raw_decision.src_to_dst(
            Currency.USD, Currency.BTC
        )
        amount_of_btc_to_sell = portfolio[Currency.BTC] * raw_decision.src_to_dst(
            Currency.BTC, Currency.USD
        )

        delta_usd = btc_price * amount_of_btc_to_sell - amount_of_usd_to_sell
        if abs(delta_usd) < min_transaction_amount:
            return []  # Do nothing
        if delta_usd > 0:  # Sell bitcoin
            amount_of_btc_to_sell = delta_usd / btc_price
            return [
                self.get_sell_order(
                    sell_amount=amount_of_btc_to_sell,
                    latest_candle=latest_candle,
                    raw_decision=raw_decision,
                )
            ]
        else:  # Buy bitcoin
            amount_of_usd_to_sell = int(-delta_usd)
            btc_buy_amount = amount_of_usd_to_sell / btc_price
            return [
                self.get_buy_order(
                    buy_amount=btc_buy_amount,
                    latest_candle=latest_candle,
                    raw_decision=raw_decision,
                )
            ]

    def simple_make_decision(  # TODO: Keep only one make_decision
        self,
        raw_decision: Optional[RawDecision],
        portfolio: Portfolio,
        latest_candle: Candle,
    ) -> List[Order]:
        btc_price = latest_candle.close
        amount_of_usd_to_sell = portfolio[Currency.USD] * raw_decision.src_to_dst(
            Currency.USD, Currency.BTC
        )
        amount_of_btc_to_sell = portfolio[Currency.BTC] * raw_decision.src_to_dst(
            Currency.BTC, Currency.USD
        )
        sell_btc_order = self.get_sell_order(
            sell_amount=amount_of_btc_to_sell,
            latest_candle=latest_candle,
            raw_decision=raw_decision,
        )

        btc_buy_amount = amount_of_usd_to_sell / btc_price
        buy_btc_order = self.get_buy_order(
            buy_amount=btc_buy_amount,
            latest_candle=latest_candle,
            raw_decision=raw_decision,
        )
        return [buy_btc_order, sell_btc_order]

    def update_and_decide(
        self,
        latest_candle: Candle,
        portfolio: Portfolio,
    ) -> List[Order]:
        assert self.latest_candle is not None
        if latest_candle == self.latest_candle:
            return []

        raw_decision = self.model_wrapper.update_and_predict(latest_candle)
        self.latest_candle = latest_candle

        return self.make_decision(raw_decision, portfolio, latest_candle)
