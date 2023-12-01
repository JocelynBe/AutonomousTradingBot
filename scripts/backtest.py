import logging
import os
import shutil
from typing import List

import click
from mock import mock

from agents.base import BaseAgent
from agents.mock import CachedAgent, compute_cached_decisions
from backtest.pipeline import BacktestPipeline
from backtest.utils import init_portfolio
from constants import BINANCE_FEE, TRAINING_DIR
from exchange_api.contracts import Candle
from exchange_api.mock import BaseExchangeAPI
from models.metrics.contracts import TRAINING_METRICS_FILENAME
from models.wrapper import BaseModelWrapper
from scripts.train import TRAINING_CONFIG_FILENAME
from trainer.base import MODELS_DIR
from trainer.config import TrainingConfig
from trainer.pre_processing import TEST_CANDLES_FILENAME
from utils.io_utils import load

logger = logging.getLogger(__name__)


def create_backtest_directory(training_dir: str) -> str:
    output_dir = os.path.join(training_dir, "backtest")
    if os.path.exists(output_dir):
        delete = input(f"Backtest directory already exists. Delete? [y/n]")
        if delete == "y":
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError("Backtest as already been run")

    os.makedirs(output_dir)
    return output_dir


def load_best_model(training_dir: str) -> BaseModelWrapper:
    training_metrics = load(os.path.join(training_dir, TRAINING_METRICS_FILENAME))
    best_step = training_metrics["best_step"]
    logger.info(f"Loading model at best step {best_step}")
    path_to_model = os.path.join(
        training_dir, TRAINING_DIR, MODELS_DIR, f"model_wrapper_{best_step}.json"
    )
    model_wrapper = BaseModelWrapper.read_from_json(path_to_model)
    return model_wrapper


def load_test_candles(training_dir: str) -> List[Candle]:
    logger.info("Loading test candles")
    filepath = os.path.join(training_dir, TEST_CANDLES_FILENAME)
    test_candles = load(filepath)
    return test_candles


def load_training_config(training_dir: str) -> TrainingConfig:
    training_config_path = os.path.join(training_dir, TRAINING_CONFIG_FILENAME)
    return TrainingConfig.from_path(training_config_path)


def run_backtest(
    training_dir: str, fee: float = BINANCE_FEE, cached: bool = True
) -> None:
    logger.info(f"Kicking off backtest for training_dir = {training_dir}")
    backtest_dir = create_backtest_directory(training_dir)
    model_wrapper = load_best_model(training_dir)
    test_candles = load_test_candles(training_dir)
    training_config = load_training_config(training_dir)

    portfolio = init_portfolio()
    if cached:
        cached_decisions = compute_cached_decisions(model_wrapper, training_dir)
        agent = CachedAgent(
            model_wrapper=mock.MagicMock(),
            fee=fee,
            cached_decisions=cached_decisions,
        )
        candles_subset = [
            candle
            for candle in test_candles
            if candle.timestamp <= max(cached_decisions.keys())
        ]
        api = BaseExchangeAPI(
            data_points=candles_subset, init_portfolio=portfolio, fee=fee
        )
    else:
        agent = BaseAgent(model_wrapper, fee)
        api = BaseExchangeAPI(
            data_points=test_candles, init_portfolio=portfolio, fee=fee
        )

    pipeline = BacktestPipeline(agent, api, training_config, backtest_dir)
    pipeline.run()


@click.command()
@click.option("--training_dir", help="Path to training dir to backtest")
@click.option("--cached", is_flag=True, help="Whether to use cached decisions")
def main(training_dir: str, cached: bool) -> None:
    run_backtest(training_dir, cached=cached)


if __name__ == "__main__":
    main()
