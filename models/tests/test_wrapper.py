import unittest

import numpy as np

from constants import N_WARM_UP_CANDLES
from exchange_api.contracts import Currency, ExchangePair
from features.config import FeaturizerConfig
from models.config import ModelConfig
from models.wrapper import BaseModelWrapper
from utils.test_utils import get_ordered_candles


class TestWrapper(unittest.TestCase):
    def get_model_wrapper(self) -> BaseModelWrapper:
        model_config = ModelConfig()
        featurizer_config = FeaturizerConfig()
        exchange_pair = ExchangePair(
            src_currency=Currency.USD, dst_currency=Currency.BTC
        )
        model_wrapper = BaseModelWrapper(model_config, featurizer_config, exchange_pair)
        return model_wrapper

    def test_warmup(self) -> None:
        model_wrapper = self.get_model_wrapper()
        n_candles = (
            N_WARM_UP_CANDLES
            + model_wrapper.model_config.encoder_config.total_warmup_length
        )
        offset = 10
        warmup_candles = get_ordered_candles(n_candles, offset)
        model_wrapper.warmup(warmup_candles)

        self.assertEqual(
            model_wrapper.featurizer_buffer.buffer, warmup_candles[-N_WARM_UP_CANDLES:]
        )

        self.assertEqual(
            len(model_wrapper.encoder_buffer.buffer),
            model_wrapper.model_config.encoder_config.total_warmup_length,
        )

        first_features_in_buffer = model_wrapper.encoder_buffer.buffer[0]
        last_features_in_buffer = model_wrapper.encoder_buffer.buffer[-1]
        expected_first_time = N_WARM_UP_CANDLES + offset
        expected_last_time = (
            N_WARM_UP_CANDLES
            + offset
            + model_wrapper.model_config.encoder_config.total_warmup_length
            - 1
        )
        self.assertEqual(first_features_in_buffer.time, expected_first_time)
        self.assertEqual(last_features_in_buffer.time, expected_last_time)

        normalized_candle_price = first_features_in_buffer.features[0].item()
        average_price_buffer = np.mean([i for i in range(offset, expected_first_time)])
        self.assertEqual(
            normalized_candle_price, expected_first_time / average_price_buffer
        )

        normalized_candle_price = last_features_in_buffer.features[0].item()
        average_price_buffer = np.mean(
            [
                i
                for i in range(
                    expected_last_time - N_WARM_UP_CANDLES, expected_last_time
                )
            ]
        )
        self.assertEqual(
            normalized_candle_price, expected_last_time / average_price_buffer
        )


if __name__ == "__main__":
    unittest.main()
