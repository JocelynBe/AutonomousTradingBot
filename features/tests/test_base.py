import unittest
from unittest.mock import patch

from parameterized import parameterized

from constants import N_WARM_UP_CANDLES
from exchange_api.contracts import Candle, Currency
from features.base import BaseFeaturizer
from features.config import FeaturizerConfig
from models.wrapper import CandleBuffer
from utils.test_utils import get_ordered_candles


def get_expected_candle(timestamp: int, value: float) -> Candle:
    return Candle(
        timestamp=timestamp,
        src_currency=Currency.USD,
        dst_currency=Currency.BTC,
        open=value,
        close=value,
        high=value,
        low=value,
        volume=value,
    )


class TestBaseFeaturizer(unittest.TestCase):
    @parameterized.expand([(10,), (0,), (100,), (42,)])
    def test_get_reference_candle(self, offset: int):
        warmup_candles = get_ordered_candles(N_WARM_UP_CANDLES, offset=offset)
        featurizer_buffer = CandleBuffer(warmup_candles, N_WARM_UP_CANDLES)
        reference_candle = BaseFeaturizer.get_reference_candle(featurizer_buffer)
        expected_average = (
            sum(i for i in range(offset, N_WARM_UP_CANDLES + offset))
            / N_WARM_UP_CANDLES
        )
        expected_candle = get_expected_candle(timestamp=offset, value=expected_average)
        self.assertEqual(reference_candle, expected_candle)

    def test_preprocess_candles(self) -> None:
        offset = 10
        value = 2.0
        warmup_candles = get_ordered_candles(N_WARM_UP_CANDLES, offset=offset)
        reference_candle = get_expected_candle(timestamp=offset, value=value)
        with patch.object(
            BaseFeaturizer, "get_reference_candle", return_value=reference_candle
        ):
            base_featurizer = BaseFeaturizer(config=FeaturizerConfig())
            base_featurizer.preprocess_candles(warmup_candles, None)  # type: ignore


if __name__ == "__main__":
    unittest.main()
