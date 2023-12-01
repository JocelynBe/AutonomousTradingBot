from typing import List

from exchange_api.abstract import AbstractExchangeAPI
from exchange_api.contracts import BuyTrade, Candle, Order, Portfolio, SellTrade


class SimpleMockExchangeAPI(AbstractExchangeAPI):
    def __init__(self, init_portfolio: Portfolio, fee: float):
        self._fee = fee
        self._current_orders: List[Order] = []
        self._portfolio = init_portfolio

    @property
    def fee(self) -> float:
        return self._fee

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    def add_order(self, order: Order) -> None:
        self._current_orders.append(order)

    def fullfill_buy_trade(self, buy_trade: BuyTrade, latest_candle: Candle) -> None:
        src_amount = buy_trade.buy_amount * buy_trade.bid

        assert buy_trade.src_currency == latest_candle.src_currency
        assert buy_trade.dst_currency == latest_candle.dst_currency

        conversion_rate = 1 / latest_candle.close
        dst_amount = src_amount * (1 - self.fee) * conversion_rate

        assert self.portfolio[buy_trade.src_currency] >= src_amount
        self.portfolio[buy_trade.src_currency] += -src_amount
        self.portfolio[buy_trade.dst_currency] += dst_amount

    def fullfill_sell_trade(self, sell_trade: SellTrade, latest_candle: Candle) -> None:
        sell_amount = sell_trade.sell_amount

        assert sell_trade.src_currency == latest_candle.dst_currency
        assert sell_trade.dst_currency == latest_candle.src_currency

        conversion_rate = latest_candle.close
        dst_amount = sell_amount * (1 - self.fee) * conversion_rate

        assert self.portfolio[sell_trade.src_currency] >= sell_amount
        self.portfolio[sell_trade.src_currency] += -sell_amount
        self.portfolio[sell_trade.dst_currency] += dst_amount

    def fullfill_orders(self, latest_candle: Candle) -> None:
        for order in self._current_orders:
            if order.buy is not None:
                self.fullfill_buy_trade(order.buy, latest_candle)
            elif order.sell is not None:
                self.fullfill_sell_trade(order.sell, latest_candle)
            else:
                raise RuntimeError(f"order = {order}")

        self._current_orders = []
