import unittest
from dataclasses import dataclass
from typing import List

import torch
from deepdiff import DeepDiff
from parameterized import parameterized

from exchange_api.contracts import Currency, ExchangePair
from features.config import FeaturizerConfig
from features.contracts import FeaturizerOutput
from models.config import OrderedCurrencies
from models.contracts import (
    CANDLE_DIM,
    Buffer,
    DecisionsTensor,
    ModelInputs,
    TransferDecision,
)
from trainer.base import set_seed
from trainer.config import TrainingConfig
from utils.test_utils import get_conversion_rate_from_price


class TestContracts(unittest.TestCase):
    def setUp(self) -> None:
        set_seed(0)
        self.batch_size = 8
        self.seq_length = 1201
        self.training_seq_length = 120
        self.input_embedding_dim = 13
        self.encoder_output_dim = 7
        self.n_currencies = 2

    def test_batch(self):
        features = torch.rand(
            (self.batch_size, self.training_seq_length, self.input_embedding_dim)
        )
        candles = torch.rand((self.batch_size, self.training_seq_length, CANDLE_DIM))
        timestamps = torch.rand((self.batch_size, self.training_seq_length, 1))
        batch = ModelInputs(features, candles, timestamps)
        self.assertEqual(batch.batch_size, self.batch_size)

        merged_batch = ModelInputs.merge([batch, batch])
        self.assertEqual(merged_batch.batch_size, 2 * self.batch_size)

    def test_featurizer_output(self):
        features = torch.rand((self.seq_length, self.input_embedding_dim))
        candles = torch.rand((self.seq_length, CANDLE_DIM))
        timestamps = torch.rand((self.seq_length, 1))
        column_names = [f"column_{i}" for i in range(self.input_embedding_dim)]
        featurizer_config = FeaturizerConfig()
        exchange_pair = ExchangePair(
            src_currency=Currency.USD, dst_currency=Currency.BTC
        )

        featurizer_output = FeaturizerOutput(
            column_names,
            features,
            candles,
            timestamps,
            featurizer_config,
            exchange_pair,
        )

        self.assertEqual(featurizer_output.seq_len, candles.shape[0])

        # Test merge_list
        featurizer_output_concat = FeaturizerOutput.from_single_steps(
            [featurizer_output, featurizer_output]
        )
        self.assertEqual(
            featurizer_output_concat.column_names, featurizer_output.column_names
        )
        self.assertEqual(
            featurizer_output_concat.featurizer_config,
            featurizer_output.featurizer_config,
        )
        for attr in ["features", "candles", "timestamps"]:
            self.assertTrue(
                torch.equal(
                    getattr(featurizer_output_concat, attr)[
                        : featurizer_output.seq_len
                    ],
                    getattr(featurizer_output, attr),
                )
            )
            self.assertTrue(
                torch.equal(
                    getattr(featurizer_output_concat, attr)[
                        featurizer_output.seq_len :
                    ],
                    getattr(featurizer_output, attr),
                )
            )

        # Test to_batch
        training_config = TrainingConfig()
        batch = featurizer_output.to_batch(training_config)
        rnn_warmup_steps = training_config.model_config.encoder_config.rnn_warmup_steps
        training_seq_length = training_config.model_config.encoder_config.seq_length
        expected_seq_length = rnn_warmup_steps + training_seq_length
        expected_n_batches = (self.seq_length - rnn_warmup_steps) // training_seq_length

        self.assertEqual(
            batch.features.shape,
            (expected_n_batches, expected_seq_length, self.input_embedding_dim),
        )
        self.assertEqual(
            batch.candles.shape,
            (expected_n_batches, expected_seq_length, CANDLE_DIM),
        )
        self.assertEqual(
            batch.timestamps.shape, (expected_n_batches, expected_seq_length, 1)
        )

        # Test __getitem__
        specific_time_step_features = featurizer_output[18]
        self.assertEqual(
            specific_time_step_features.features.shape,
            (1, self.input_embedding_dim),
        )
        self.assertEqual(
            specific_time_step_features.candles.shape,
            (1, CANDLE_DIM),
        )
        self.assertEqual(specific_time_step_features.timestamps.shape, (1, 1))

    def test_buffer(self):
        @dataclass
        class BufferValue:
            value: int
            time: float

        max_buffer_size = 10
        buffer = Buffer(
            buffer=[BufferValue(1, 0), BufferValue(2, 1), BufferValue(4, 2)],
            max_buffer_size=max_buffer_size,
        )
        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.last_value.value, 4)

        buffer.update(BufferValue(8, 3))
        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.last_value.value, 8)

        for i in range(10):
            buffer.update(BufferValue(i, 5 + i))
            self.assertEqual(len(buffer), min(max_buffer_size, 5 + i))

        expected_buffer = Buffer(
            buffer=[BufferValue(i, 5 + i) for i in range(10)], max_buffer_size=10
        )
        self.assertEqual(buffer, expected_buffer)

        with self.assertRaises(AssertionError):
            buffer.update(BufferValue(10, 0))

    def test_decisions_tensor(self) -> None:
        decisions_tensor = DecisionsTensor(
            decisions=torch.Tensor(
                [
                    [
                        [[0.25, 0.75], [0.8, 0.2]],
                        [[0.572, 0.428], [0.6, 0.4]],
                        [[0.21, 0.79], [0.23, 0.77]],
                    ]
                ]
                * 2
            ),
            timestamps=torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [10000.0, 10001.0, 10002.0],
                ]
            ),
            ordered_currencies=OrderedCurrencies([Currency.USD, Currency.BTC]),
        )
        res = decisions_tensor.percentage_currency_i_to_j_at_step_k(
            sample_idx=0, src_currency=Currency.BTC, dst_currency=Currency.USD, step_k=2
        )
        diff = DeepDiff(
            res,
            TransferDecision(
                timestamp=2.0,
                src_currency=Currency.BTC,
                dst_currency=Currency.USD,
                transfer_percentage=0.23,
            ),
            significant_digits=7,
        )
        self.assertEqual(diff, {})

        res = decisions_tensor.percentage_currency_i_to_j_at_step_k(
            sample_idx=1, src_currency=Currency.USD, dst_currency=Currency.BTC, step_k=1
        )
        diff = DeepDiff(
            res,
            TransferDecision(
                timestamp=10001.0,
                src_currency=Currency.USD,
                dst_currency=Currency.BTC,
                transfer_percentage=0.428,
            ),
            significant_digits=7,
        )
        self.assertEqual(diff, {})

    @parameterized.expand(
        [
            (1, 10, 2, 0.01),
            (8, 10, 2, 0.01),
            (4, 10, 0, 0.01),
            (32, 10, 10, 0.00001),
            (4, 100, 20, 0.00001),
            (4, 400, 20, 0.00001),
        ]
    )
    def test_conversion_rate(
        self, batch_size: int, seq_len: int, pad_len: int, step_change: float
    ):
        def get_conversion_rate_for_price(price: float) -> List[List[float]]:
            return [[1.0, 1.0 / price if price else float("inf")], [price, 1.0]]

        price_ts_one_batch = [
            1 + step_change * i if i < seq_len else 0 for i in range(seq_len + pad_len)
        ]
        expected_conversion_rate = [
            [get_conversion_rate_for_price(price) for price in price_ts_one_batch]
            for _ in range(batch_size)
        ]
        conversion_rate = get_conversion_rate_from_price(price_ts_one_batch, batch_size)
        diff = DeepDiff(
            conversion_rate.tolist(), expected_conversion_rate, significant_digits=5
        )
        self.assertFalse(diff)


if __name__ == "__main__":
    unittest.main()
