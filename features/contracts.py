from dataclasses import dataclass
from typing import List

import torch

from contracts import TimedTensor, ValueAtTimeX
from exchange_api.contracts import Candle, ExchangePair
from features.config import FeaturizerConfig

CANDLE_COLUMNS = ["open", "close", "high", "low", "volume"]
TorchCandles = torch.Tensor  # shape = [bs, seq_len, [open, close, high, low, volume]]
TorchTimestamps = torch.Tensor
TorchFeatures = torch.Tensor  # shape = [bs, seq_len, embedding_dim]


@dataclass
class SingleStepFeaturizerOutput(ValueAtTimeX):
    column_names: List[str]
    features: torch.Tensor  # shape = [hidden_dim]
    candles: torch.Tensor  # shape = [5]
    featurizer_config: FeaturizerConfig
    exchange_pair: ExchangePair

    def __post_init__(self):
        # seq_len x hidden dim
        assert tuple(self.features.shape) == (
            len(self.column_names),
        ), self.features.shape
        assert tuple(self.candles.shape) == (5,), self.candles.shape


class FeaturesTensor(TimedTensor):
    def __init__(
        self,
        features: TorchFeatures,
        timestamps: TorchTimestamps,
    ):
        super().__init__(
            timestamps,
            tensors={"features": features},
        )
        self.timestamps = timestamps

    @property
    def features(self) -> TorchFeatures:
        return self.tensors.features


class FeaturizerOutput(TimedTensor):
    def __init__(
        self,
        column_names: List[str],
        features: TorchFeatures,
        candles: TorchCandles,
        timestamps: TorchTimestamps,
        featurizer_config: FeaturizerConfig,
        exchange_pair: ExchangePair,
    ):
        self.column_names = column_names
        super().__init__(
            timestamps,
            tensors={"features": features, "candles": candles},
            extras={
                "featurizer_config": featurizer_config,
                "column_names": column_names,
                "exchange_pair": exchange_pair,
            },
        )
        self.timestamps = timestamps
        self.featurizer_config = featurizer_config
        self.exchange_pair = exchange_pair

    @property
    def features(self) -> TorchFeatures:
        return self.tensors.features

    @property
    def candles(self) -> TorchCandles:
        return self.tensors.candles

    def __post_init__(self) -> None:
        self.sanity_check()

    def sanity_check(self) -> None:
        super().sanity_check()
        # batch_size x seq_len x hidden dim
        _, _, features_dim = self.tensors.features.shape
        assert len(self.column_names) == features_dim

    @classmethod
    def from_single_steps(
        cls, to_merge: List[SingleStepFeaturizerOutput]
    ) -> "FeaturizerOutput":
        column_names = to_merge[0].column_names
        featurizer_config = to_merge[0].featurizer_config
        exchange_pair = to_merge[0].exchange_pair
        assert all(x.column_names == column_names for x in to_merge)
        assert all(x.featurizer_config == featurizer_config for x in to_merge)
        assert all(x.exchange_pair == exchange_pair for x in to_merge)

        features = torch.cat([x.features.unsqueeze(0) for x in to_merge], dim=0)
        candles = torch.cat([x.candles.unsqueeze(0) for x in to_merge], dim=0)
        timestamps = torch.cat(
            [
                torch.tensor(x.timestamp, dtype=torch.float64).unsqueeze(0)
                for x in to_merge
            ],
            dim=0,
        ).reshape(-1, 1)
        return cls(
            candles=candles.unsqueeze(0),  # Batch dim
            features=features.unsqueeze(0),  # Batch dim
            timestamps=timestamps.unsqueeze(0),  # Batch dim
            column_names=column_names,
            exchange_pair=exchange_pair,
            featurizer_config=featurizer_config,
        )

    def __getitem__(self, item: int) -> SingleStepFeaturizerOutput:
        assert self.shape.features[0] == 1  # Batch size one
        assert self.shape.candles[0] == 1  # Batch size one
        assert self.timestamps.shape[0] == 1  # Batch size one

        return SingleStepFeaturizerOutput(
            column_names=self.column_names,
            features=self.tensors.features[0, item, :],
            candles=self.tensors.candles[0, item, :],
            timestamp=float(self.timestamps[0, item, :].item()),
            featurizer_config=self.featurizer_config,
            exchange_pair=self.exchange_pair,
        )


@dataclass
class CandleChunk:
    slice_id: int
    chunk_id: int
    candles: List[Candle]


@dataclass
class FeaturesChunk:
    slice_id: int
    chunk_id: int
    features: FeaturizerOutput
