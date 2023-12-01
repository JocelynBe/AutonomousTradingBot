import inspect
from typing import List

import jesse.indicators as indicators
import numpy as np
import pandas as pd
import torch

from constants import N_WARM_UP_CANDLES
from exchange_api.contracts import Candle, ExchangePair
from features.abstract import AbstractFeaturizer
from features.contracts import FeaturizerOutput, SingleStepFeaturizerOutput
from models.contracts import Buffer


class BaseFeaturizer(AbstractFeaturizer):
    LIFETIME: float = 24 * 60 * 60 * 1000  # 1 day in ms
    FREQ: float = 60 * 1000  # Every minute

    def embedding_dim(self) -> int:
        return len(self.config.PATTERN_TYPES) + len(self.config.INDICATOR_NAMES)

    @staticmethod
    def get_reference_candle(candles_buffer: Buffer) -> Candle:
        warm_up_candles = candles_buffer.buffer
        assert len(warm_up_candles) == N_WARM_UP_CANDLES
        warmup_df = pd.DataFrame([candle.to_list() for candle in warm_up_candles])
        warmup_df.columns = ["timestamp", "open", "close", "high", "low", "volume"]

        warmup_means: pd.Series = warmup_df.describe().iloc[1]
        first_candle = warm_up_candles[0]
        reference_candle = Candle(
            src_currency=first_candle.src_currency,
            dst_currency=first_candle.dst_currency,
            timestamp=first_candle.timestamp,
            open=warmup_means.open,
            close=warmup_means.close,
            high=warmup_means.high,
            low=warmup_means.low,
            volume=warmup_means.volume,
        )
        return reference_candle

    def preprocess_candles(
        self, candles: List[Candle], buffer_candles: Buffer
    ) -> pd.DataFrame:
        reference_candle = self.get_reference_candle(buffer_candles)

        assert all(
            candle.timestamp - reference_candle.timestamp
            <= self.LIFETIME + N_WARM_UP_CANDLES * self.FREQ
            for candle in candles
        )
        assert len(candles) <= self.LIFETIME / self.FREQ, len(candles)

        df = pd.DataFrame([candle.to_list() for candle in candles])
        df.columns = ["time", "open", "close", "high", "low", "volume"]

        df["open"] = df["open"] / reference_candle.open
        df["close"] = df["close"] / reference_candle.close
        df["high"] = df["high"] / reference_candle.high
        df["low"] = df["low"] / reference_candle.low
        df["volume"] = df["volume"] / reference_candle.volume
        return df

    def featurize(self, normalized_df: pd.DataFrame) -> pd.DataFrame:
        candles_arr = normalized_df.values

        features = {}
        for indicator_name in self.config.INDICATOR_NAMES:
            indicator = indicators.__dict__[indicator_name]
            signature = inspect.signature(indicator)

            assert "sequential" in signature.parameters

            if indicator_name == "pattern_recognition":
                for pattern_type in self.config.PATTERN_TYPES:
                    assert pattern_type not in features
                    features[pattern_type] = indicator(
                        candles_arr, pattern_type=pattern_type, sequential=True
                    )
                continue

            feature = indicator(candles_arr, sequential=True)

            if not isinstance(feature, np.ndarray):
                for feature_name, feature_value in feature._asdict().items():
                    assert f"{indicator_name}_{feature_name}" not in features
                    features[f"{indicator_name}_{feature_name}"] = feature_value
                continue

            assert indicator_name not in features
            features[indicator_name] = feature

        for feature_name, feature in features.items():
            if "bool" in str(feature.dtype):
                features[feature_name] = 1 * feature

        features_df = pd.DataFrame(features)

        # Remove the initial candles since they don't have a full warmup buffer
        features_df = features_df[N_WARM_UP_CANDLES:]
        candles_df = normalized_df[N_WARM_UP_CANDLES:]

        columns = ["open", "close", "high", "low", "volume"]
        features_df = features_df.reset_index(drop=True)
        candles_df = candles_df[columns].reset_index(drop=True)

        features_df = candles_df.join(features_df, how="outer").reset_index(drop=True)
        return features_df

    @staticmethod
    def sanitize(features_df: pd.DataFrame) -> pd.DataFrame:
        features_df = features_df.fillna(0)
        features_df = features_df.replace([-np.inf, np.inf], 0)
        return features_df

    def df_to_featurizer_output(
        self, features_df: pd.DataFrame, last_candle: Candle
    ) -> SingleStepFeaturizerOutput:
        exchange_pair = ExchangePair(last_candle.src_currency, last_candle.dst_currency)
        column_names = list(features_df.columns)
        features = torch.tensor(features_df.values, dtype=torch.float64).unsqueeze(
            0
        )  # batch dim
        torch_candles = torch.vstack([last_candle.to_torch()]).unsqueeze(0)  # batch dim
        timestamps = torch.vstack(
            [torch.tensor(last_candle.timestamp, dtype=torch.float64)]
        ).unsqueeze(
            0
        )  # batch dim
        return FeaturizerOutput(
            candles=torch_candles,
            features=features,
            timestamps=timestamps,
            column_names=column_names,
            featurizer_config=self.config,
            exchange_pair=exchange_pair,
        )[0]

    def predict_last_candle(
        self, buffer: Buffer, last_candle: Candle
    ) -> SingleStepFeaturizerOutput:
        assert len(buffer) == N_WARM_UP_CANDLES
        assert last_candle.timestamp >= buffer.last_value.timestamp

        # Returns one big Batch with all the data
        normalized_buffer_df = self.preprocess_candles(buffer.buffer, buffer)
        normalized_candles_df = self.preprocess_candles([last_candle], buffer)

        # For the .featurize call we need warm up candles. Those candles have to be normalized as well.
        # In order to avoid doubling the amount of warmup candles we normalize
        # the warmup candles by the reference candle.
        # Furthermore, we use
        normalized_df = pd.concat([normalized_buffer_df, normalized_candles_df])

        features_df = self.featurize(normalized_df)
        sanitized_df = self.sanitize(features_df)
        featurizer_output = self.df_to_featurizer_output(sanitized_df, last_candle)
        return featurizer_output

    def predict(self, buffer: Buffer, candles: List[Candle]) -> FeaturizerOutput:
        predictions = []
        for candle in candles:
            predictions.append(self.predict_last_candle(buffer, candle))
            buffer.update(candle)

        return FeaturizerOutput.from_single_steps(predictions)
