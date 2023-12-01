import os
import unittest
from typing import List

import pandas as pd
from box import Box
from deepdiff import DeepDiff

from constants import (
    CANDLES_FILENAME,
    FEATURES_FILENAME,
    N_WARM_UP_CANDLES,
    ORACLE_DECISIONS_FILENAME,
)
from exchange_api.contracts import Candle, Currency
from features.config import FeaturizerConfig, OracleConfig
from features.contracts import CANDLE_COLUMNS, FeaturizerOutput
from features.main import run_datagen
from models.contracts import CANDLE_DIM, DecisionsTensor
from utils.io_utils import file_compression, load
from utils.test_utils import generate_price_dirac_with_no_signal


def constant_function(*_, **__):
    return 0


class TestRunDatagen(unittest.TestCase):
    def setUp(self) -> None:
        self.seq_length: int = 10000
        self.temp_dir = Box(
            {"name": "/tmp/test_run_datagen/"}
        )  # tempfile.TemporaryDirectory()
        os.makedirs(self.temp_dir.name, exist_ok=True)  # TODO: Remove
        self.path_to_csv = os.path.join(self.temp_dir.name, "candles_df.csv.gz")
        self.candles_df = generate_price_dirac_with_no_signal(self.seq_length)
        self.candles_df.to_csv(
            self.path_to_csv,
            index=False,
            compression=file_compression(self.path_to_csv),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def verify_features(self, features: List[FeaturizerOutput]) -> None:
        self.assertEqual(len(features), 1)
        featurizer_output = features[0]

        # Timestamps
        batch_size, seq_length, hidden_dim = featurizer_output.timestamps.shape
        self.assertEqual(batch_size, 1)
        self.assertEqual(seq_length, self.seq_length - N_WARM_UP_CANDLES)
        self.assertEqual(hidden_dim, 1)

        # Candles
        batch_size, seq_length, candle_dim = featurizer_output.tensors.candles.shape
        self.assertEqual(batch_size, 1)
        self.assertEqual(seq_length, self.seq_length - N_WARM_UP_CANDLES)
        self.assertEqual(candle_dim, CANDLE_DIM)

        candles_df = pd.DataFrame(featurizer_output.tensors.candles[0].tolist())
        candles_df.columns = CANDLE_COLUMNS
        candles_df["timestamp"] = featurizer_output.timestamps[0].reshape(-1).tolist()

        og_candles_df = self.candles_df[N_WARM_UP_CANDLES:][
            ["timestamp"] + CANDLE_COLUMNS
        ].reset_index(drop=True)
        candles_df = candles_df[["timestamp"] + CANDLE_COLUMNS]

        self.assertTrue(((og_candles_df - candles_df).abs() < 1e-13).all().all())

        # TODO: Test that featurizer_output.features[:, :5] a bien les memes diracs que candles_df

    def verify_candles(self, candles: List[Candle]) -> None:
        self.assertEqual(len(candles), self.seq_length)
        og_candles_dict = self.candles_df.set_index("timestamp").to_dict(orient="index")
        for candle in candles:
            og_candle = og_candles_dict[candle.timestamp]  # type: ignore
            og_candle["timestamp"] = candle.timestamp
            og_candle["src_currency"] = Currency.USD
            og_candle["dst_currency"] = Currency.BTC
            self.assertEqual(
                DeepDiff(og_candle, vars(candle), significant_digits=12), {}
            )

    def verify_oracle_decisions(self, verify_oracle_decisions: DecisionsTensor) -> None:
        assert False, "TODO"

    def test_run_datagen(self) -> None:
        run_datagen(
            path_to_csv=self.path_to_csv,
            start_time=-1,
            output_dir=os.path.join(self.temp_dir.name, "datagen/"),
            num_cpus=2,
            interval_in_minutes=1,
            featurizer_config=FeaturizerConfig(oracle_config=OracleConfig()),
        )

        self.assertEqual(
            set(os.listdir(os.path.join(self.temp_dir.name, "datagen/"))),
            {
                "featurizer_config.json",
                "features.pkl",
                "candles.pkl",
                "oracle_decisions.pkl",
                "aligned_slices.pkl",
            },
        )
        features_path = os.path.join(self.temp_dir.name, "datagen", FEATURES_FILENAME)
        oracle_decisions_path = os.path.join(
            self.temp_dir.name, "datagen", ORACLE_DECISIONS_FILENAME
        )
        candles_path = os.path.join(self.temp_dir.name, "datagen", CANDLES_FILENAME)
        features = load(features_path)
        candles = load(candles_path)
        oracle_decisions = load(oracle_decisions_path)

        self.verify_features(features)
        self.verify_candles(candles)
        self.verify_oracle_decisions(oracle_decisions)
        assert False


if __name__ == "__main__":
    # unittest.main()
    import cProfile
    import pstats

    profiler = cProfile.Profile()

    test_run_datagen = TestRunDatagen()
    test_run_datagen.setUp()
    profiler.enable()
    test_run_datagen.test_run_datagen()
    profiler.disable()
    test_run_datagen.tearDown()

    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats()

    stats.dump_stats("/tmp/profile.dat")
