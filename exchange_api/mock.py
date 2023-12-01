import random
from typing import Dict, List

from tqdm import tqdm

from exchange_api.abstract import AbstractExchangeAPI
from exchange_api.contracts import (
    BuyTrade,
    Candle,
    Currency,
    ExchangePair,
    Order,
    Portfolio,
    SellTrade,
    Status,
    TradeResponse,
)


class BaseExchangeAPI(AbstractExchangeAPI):
    def __init__(
        self, data_points: List[Candle], init_portfolio: Portfolio, fee: float
    ):
        self.candles = sorted(data_points, key=lambda pt: pt.timestamp)
        self.tqdm = tqdm(total=len(data_points))
        self.index = 0
        self._fee = fee
        self._current_orders: List[Order] = []
        self._portfolio = init_portfolio
        self.sanity_check()

    def get_latest_candle(self) -> Candle:
        self.update_index(self.index + 1)
        return self.current_candle

    @property
    def fee(self) -> float:
        return self._fee

    @property
    def current_candle(self) -> Candle:
        return self.candles[self.index]

    @property
    def current_exchange_rate(self) -> Dict[ExchangePair, float]:
        price_btc_to_usd = self.current_candle.close
        return {
            ExchangePair(Currency.BTC, Currency.USD): price_btc_to_usd,
            ExchangePair(Currency.USD, Currency.BTC): 1 / price_btc_to_usd,
        }

    @property
    def current_orders(self) -> List[Order]:
        self._maybe_fullfill_orders()
        return self._current_orders

    @property
    def portfolio(self) -> Portfolio:
        self._maybe_fullfill_orders()
        return self._portfolio

    @property
    def portfolio_value(self) -> float:
        return self.portfolio.usd_value(self.current_exchange_rate, self.fee)

    def stop_condition(self):
        stop = self.index >= len(self.candles) - 1
        if stop:
            self.tqdm.close()
        return stop

    def get_warmup_candles(self, n_warmup_candles: int) -> List[Candle]:
        assert self.index == 0
        warmup_candles = [self.candles[i] for i in range(n_warmup_candles)]
        self.update_index(index=n_warmup_candles - 1)
        return warmup_candles

    def update_index(self, index: int) -> None:
        self.tqdm.update(index - self.index)
        self.index = index

    def sanity_check(self):
        assert 0 <= self.fee <= 0.02, f"Fee {self.fee} should be no more than 2%"

    def fullfill_buy_trade(self, buy_trade: BuyTrade) -> Status:
        assert isinstance(buy_trade, BuyTrade), buy_trade

        cost = buy_trade.cost
        if cost > self._portfolio[buy_trade.src_currency]:
            return Status.INSUFFICIENT_FUNDS

        self._portfolio[buy_trade.src_currency] += -cost
        self._portfolio[buy_trade.dst_currency] += buy_trade.buy_amount * (1 - self.fee)

        return Status.OK

    def fullfill_sell_trade(self, sell_trade: SellTrade) -> Status:
        assert isinstance(sell_trade, SellTrade), sell_trade

        if sell_trade.sell_amount > self._portfolio[sell_trade.src_currency]:
            return Status.INSUFFICIENT_FUNDS

        self._portfolio[sell_trade.src_currency] += -sell_trade.sell_amount
        self._portfolio[sell_trade.dst_currency] += (1 - self.fee) * sell_trade.gain

        return Status.OK

    def _maybe_fullfill_orders(self, fulfillment_probability: float = 1.0) -> None:
        fulfilled = []
        for order_idx, order in enumerate(self._current_orders):
            if random.random() > fulfillment_probability:
                continue

            if order.buy is not None:
                self.fullfill_buy_trade(order.buy)
            elif order.sell is not None:
                self.fullfill_sell_trade(order.sell)
            else:
                raise NotImplementedError(f"order = {order}")
            fulfilled.append(order_idx)

        self._current_orders = [
            order
            for order_idx, order in enumerate(self._current_orders)
            if order_idx not in fulfilled
        ]

    def buy(self, order: Order) -> TradeResponse:
        assert order.buy is not None
        buy_trade = order.buy

        self._maybe_fullfill_orders()
        cost = buy_trade.cost
        if cost > self.portfolio[buy_trade.src_currency]:
            return TradeResponse(status=Status.INSUFFICIENT_FUNDS, trade=buy_trade)

        self._current_orders.append(order)
        self._maybe_fullfill_orders()
        return TradeResponse(status=Status.OK, trade=buy_trade)

    def sell(self, order: Order) -> TradeResponse:
        assert order.sell is not None
        sell_trade = order.sell

        self._maybe_fullfill_orders()
        cost = sell_trade.sell_amount
        if cost > self.portfolio[sell_trade.src_currency]:
            return TradeResponse(status=Status.INSUFFICIENT_FUNDS, trade=sell_trade)

        self._current_orders.append(order)
        self._maybe_fullfill_orders()
        return TradeResponse(status=Status.OK, trade=sell_trade)

    def push_order(self, order: Order) -> TradeResponse:
        if order.buy:
            trade_response = self.buy(order)
        elif order.sell:
            trade_response = self.sell(order)
        else:
            raise NotImplementedError(f"order = {order}")

        assert trade_response.status == Status.OK, (order, trade_response)
        return trade_response

    def push_orders(self, orders: List[Order]) -> TradeResponse:
        return_trade_response = TradeResponse(Status.OK)
        for order in orders:
            self.push_order(order)

        return return_trade_response
