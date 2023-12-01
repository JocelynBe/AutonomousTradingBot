import logging
import os
import time
from typing import List

from agents.base import BaseAgent
from constants import N_WARM_UP_CANDLES
from exchange_api.abstract import AbstractExchangeAPI
from exchange_api.contracts import Candle, Currency, Order, Portfolio, Status
from trainer.config import TrainingConfig


class Pipeline:
    def __init__(
        self,
        agent: BaseAgent,
        exchange_api: AbstractExchangeAPI,
        training_config: TrainingConfig,
        output_dir: str,
    ):
        self.agent = agent
        self.exchange_api = exchange_api
        self.training_config = training_config
        self.output_dir = output_dir
        self.portfolio_filename = os.path.join(self.output_dir, "portfolio.csv")
        self.actions_filename = os.path.join(self.output_dir, "actions.csv")
        self.logger = self.init_logger()

    def init_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.output_dir, "backtest_pipeline.log"),
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )

        if not os.path.exists(self.portfolio_filename):
            with open(self.portfolio_filename, "w") as f:
                f.write("timestamp, usd_amount, btc_amount, close \n")

        if not os.path.exists(self.actions_filename):
            columns = [
                "buy_amount",
                "bid",
                "sell_amount",
                "ask",
                "timestamp",
                "close",
                "usd_to_usd",
                "btc_to_btc",
                "usd_to_btc",
                "btc_to_usd",
                "order.raw_decision.timestamp",
            ]
            with open(self.actions_filename, "w") as f:
                f.write(",".join(columns) + "\n")

        return logging.getLogger("BacktestPipeline")

    def log_action(self, orders: List[Order], latest_candle: Candle) -> None:
        rows = []
        for order in orders:
            rows.append(
                [
                    order.buy.buy_amount if order.buy else None,
                    order.buy.bid if order.buy else None,
                    order.sell.sell_amount if order.sell else None,
                    order.sell.ask if order.sell else None,
                    latest_candle.timestamp,
                    latest_candle.close,
                    order.raw_decision.src_to_dst(Currency.USD, Currency.USD),
                    order.raw_decision.src_to_dst(Currency.BTC, Currency.BTC),
                    order.raw_decision.src_to_dst(Currency.USD, Currency.BTC),
                    order.raw_decision.src_to_dst(Currency.BTC, Currency.USD),
                    order.raw_decision.timestamp,
                ]
            )
        with open(self.actions_filename, "a") as f:
            for row in rows:
                f.write(",".join(map(str, row)) + "\n")

    def log_portfolio(self, portfolio: Portfolio, latest_candle: Candle) -> None:
        row = [
            latest_candle.timestamp,
            portfolio[Currency.USD],
            portfolio[Currency.BTC],
            latest_candle.close,
        ]
        with open(self.portfolio_filename, "a") as f:
            f.write(",".join(map(str, row)) + "\n")

    def wait(self) -> None:
        time.sleep(5)

    def run(self):
        n_warmup_candles = (
            N_WARM_UP_CANDLES
            + self.training_config.model_config.encoder_config.total_warmup_length
        )
        warmup_candles = self.exchange_api.get_warmup_candles(n_warmup_candles)
        self.agent.warmup(warmup_candles)
        steps = 0
        while not self.exchange_api.stop_condition():
            latest_candle = self.exchange_api.get_latest_candle()
            orders = self.agent.update_and_decide(
                latest_candle, self.exchange_api.portfolio
            )
            if steps % 1000 == 0:
                self.logger.info(f"Portfolio = {self.exchange_api.portfolio}")
                self.logger.info(
                    f"Portfolio value = {self.exchange_api.portfolio_value}"
                )

            steps += 1

            if not orders:
                self.wait()
                continue

            response = self.exchange_api.push_orders(orders)
            self.log_action(orders, latest_candle)
            self.log_portfolio(self.exchange_api.portfolio, latest_candle)
            assert response.status == Status.OK, (orders, response)
