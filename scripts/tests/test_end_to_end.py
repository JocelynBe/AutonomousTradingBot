import os
import shutil
import tempfile
import unittest
from typing import Optional

import pandas as pd

from constants import ALIGNED_SLICES_FILENAME, CANDLES_FILENAME, TRAINING_DIR
from data.synthetic.naive_data import generate_periodic_signal
from exchange_api.contracts import Currency, ExchangePair
from features.contracts import CANDLE_COLUMNS
from models.wrapper import BaseModelWrapper
from scripts.backtest import run_backtest
from scripts.run_datagen import run_datagen
from scripts.train import train_model
from trainer.base import BaseTrainer
from trainer.config import TrainingConfig


class SmokeTest(unittest.TestCase):
    def setUp(self, output_dir: Optional[str] = None) -> None:
        output_dir = "/tmp/test/"
        self.tmp_dir = output_dir or tempfile.mkdtemp()
        self.path_to_csv = os.path.join(self.tmp_dir, "candles_df.csv")
        self.generate_dataset()

    def tearDown(self) -> None:
        pass
        # shutil.rmtree(self.tmp_dir)

    def generate_dataset(self) -> None:
        exchange_pair = ExchangePair(Currency.USD, Currency.BTC)
        periodical_signal = generate_periodic_signal(
            n_steps=30000, period=29, exchange_pair=exchange_pair
        )
        df = pd.DataFrame(
            [candle.to_list() for candle in periodical_signal],
            columns=["timestamp"] + CANDLE_COLUMNS,
        )
        df.to_csv(self.path_to_csv, index=False)

    def test_smoke_test(self):
        run_datagen(
            path_to_csv=self.path_to_csv, start_time=-1, output_dir=self.tmp_dir
        )

        training_config = TrainingConfig()
        training_config.optimizer_config.learning_rate = 5e-3

        training_dir = os.path.join(self.tmp_dir, TRAINING_DIR)
        train_model(
            path_to_candles=os.path.join(self.tmp_dir, CANDLES_FILENAME),
            path_to_aligned_slices=os.path.join(self.tmp_dir, ALIGNED_SLICES_FILENAME),
            output_dir=training_dir,
            trainer_cls=BaseTrainer,
            model_wrapper_cls=BaseModelWrapper,
            training_config=training_config,
        )

        run_backtest(training_dir, cached=True)


if __name__ == "__main__":
    unittest.main()
