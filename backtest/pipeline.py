from exchange_api.mock import BaseExchangeAPI
from pipeline import Pipeline


class BacktestPipeline(Pipeline):
    exchange_api: BaseExchangeAPI

    def wait(self) -> None:
        pass
