import enum
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Generic, List, Tuple, TypeVar

import numpy as np
import torch

from contracts import TimedTensor, ValueAtTimeX
from exchange_api.contracts import Candle, Currency, ExchangePair
from features.contracts import FeaturesTensor, TorchCandles
from models.config import OrderedCurrencies

RawPrediction = torch.Tensor  # shape = [bs, seq_len, n_currencies, n_currencies]

Step = int

CANDLE_DIM = 5


class CandlesIndices(enum.Enum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4


@dataclass
class TransferDecision(ExchangePair, ValueAtTimeX):
    transfer_percentage: float


@dataclass
class RawDecision(ValueAtTimeX):
    src_to_dst_transfer_percentage: Dict[Currency, Dict[Currency, float]]

    def src_to_dst(self, src_currency: Currency, dst_currency: Currency) -> float:
        return self.src_to_dst_transfer_percentage[src_currency][dst_currency]


class DecisionsTensor(TimedTensor):
    """
    A decisions tensor is such that:
        DecisionsTensor[i, j] = proportion of currency i being transferred to currency j
    """

    def __init__(
        self,
        decisions: torch.Tensor,  # [bs, seq_len, n_currencies, n_currencies]
        timestamps: torch.Tensor,  # [bs, seq_len, 1]
        ordered_currencies: OrderedCurrencies,
        sanity_check: bool = True,
    ) -> None:
        super().__init__(
            timestamps,
            tensors={"decisions": decisions},
            extras={"ordered_currencies": ordered_currencies},
            sanity_check=sanity_check,
        )
        if sanity_check:
            self.sanity_check()

    @property
    def decisions(self) -> torch.Tensor:
        return self.tensors.decisions

    @property
    def ordered_currencies(self) -> OrderedCurrencies:
        return self.extras.ordered_currencies

    @staticmethod
    def from_list(decisions_tensor_list: List["DecisionsTensor"]) -> "DecisionsTensor":
        ordered_currencies = decisions_tensor_list[0].ordered_currencies
        assert all(
            d.ordered_currencies == ordered_currencies for d in decisions_tensor_list
        )
        return DecisionsTensor(
            decisions=torch.vstack([d.decisions for d in decisions_tensor_list]),
            timestamps=torch.vstack([d.timestamps for d in decisions_tensor_list]),
            ordered_currencies=ordered_currencies,
        )

    def sanity_check(self):
        super().sanity_check()
        assert all(
            np.isclose(x, 1) for x in self.decisions.sum(dim=3).reshape(-1).tolist()
        ), "Rows do not sum to 1"

    def percentage_currency_i_to_j_at_step_k(
        self,
        sample_idx: int,
        src_currency: Currency,
        dst_currency: Currency,
        step_k: Step,
    ) -> TransferDecision:
        currency_i = self.ordered_currencies.currency_to_idx[src_currency]
        currency_j = self.ordered_currencies.currency_to_idx[dst_currency]
        percentage = self.decisions[sample_idx, step_k, currency_i, currency_j].item()
        timestamp = self.timestamps[sample_idx, step_k].item()
        return TransferDecision(
            src_currency=src_currency,
            dst_currency=dst_currency,
            transfer_percentage=percentage,
            timestamp=timestamp,
        )

    def get_raw_decision_at_step(
        self, step_k: Step, sample_idx: int = 0
    ) -> RawDecision:
        timestamp = float(self.timestamps[sample_idx][step_k][0].item())
        src_to_dst_transfer_percentage = defaultdict(dict)
        for src_currency in self.ordered_currencies:
            for dst_currency in self.ordered_currencies:
                transfer_decision = self.percentage_currency_i_to_j_at_step_k(
                    sample_idx=sample_idx,
                    src_currency=src_currency,
                    dst_currency=dst_currency,
                    step_k=step_k,
                )
                src_to_dst_transfer_percentage[src_currency][
                    dst_currency
                ] = transfer_decision.transfer_percentage

        return RawDecision(
            src_to_dst_transfer_percentage=dict(src_to_dst_transfer_percentage),
            timestamp=timestamp,
        )


class ConversionRateAtStep(TimedTensor):
    def __init__(
        self,
        conversion_rate_at_step: torch.Tensor,
        timestamps: torch.Tensor,
        ordered_currencies: OrderedCurrencies,
    ) -> None:
        super().__init__(
            timestamps,
            tensors={"conversion_rate_at_step": conversion_rate_at_step},
            extras={"ordered_currencies": ordered_currencies},
        )

    @property
    def conversion_rate_at_step(self) -> torch.Tensor:
        return self.tensors.conversion_rate_at_step

    @property
    def ordered_currencies(self) -> OrderedCurrencies:
        return self.extras.ordered_currencies

    def conversion_rate_from_currency_a_to_currency_b(
        self, src_currency: Currency, dst_currency: Currency, sample_idx: int
    ) -> float:
        # Based on documentation/theory.md the conversion rate is such that
        # q_i * C_{i, j} = q_j
        src_currency_idx = self.ordered_currencies.currency_to_idx[src_currency]
        dst_currency_idx = self.ordered_currencies.currency_to_idx[dst_currency]
        conversion_rate_at_step = self.conversion_rate_at_step[
            sample_idx, 0, src_currency_idx, dst_currency_idx
        ].item()
        return float(conversion_rate_at_step)

    def conversion_rate_matrix(self) -> torch.Tensor:
        index = torch.LongTensor(
            [
                self.ordered_currencies.currency_to_idx[currency]
                for currency in self.ordered_currencies
            ]
        )
        ordered_rows = torch.index_select(
            self.conversion_rate_at_step, dim=2, index=index
        )
        ordered_columns = torch.index_select(ordered_rows, dim=3, index=index)
        return ordered_columns


class ConversionRate(TimedTensor):
    def __init__(
        self,
        conversion_rate: torch.Tensor,  # shape = [bs, seq_len, n_currencies, n_currencies]
        timestamps: torch.Tensor,  # shape = [bs, seq_len, 1]
        ordered_currencies: OrderedCurrencies,
    ) -> None:
        super().__init__(
            timestamps,
            {"conversion_rate": conversion_rate},
            extras={"ordered_currencies": ordered_currencies},
        )

    @property
    def conversion_rate(self) -> torch.Tensor:
        return self.tensors.conversion_rate

    def get_sample(self, sample_idx: int) -> "ConversionRate":
        return ConversionRate(
            self.tensors.conversion_rate[sample_idx].unsqueeze(0),
            self.timestamps[sample_idx].unsqueeze(0),
            self.extras.ordered_currencies,
        )

    def compute_portfolio_value_at_step(
        self,
        portfolio: torch.Tensor,  # shape = [bs, n_currencies]
        step: int,
        dst_currency: Currency = Currency.USD,
    ) -> torch.Tensor:  # shape = [bs, 1]
        batch_size, _, n_currencies, _ = self.shape.conversion_rate

        # First dim is batch_size, second dim is seq_len
        # Based on documentation/theory.md the conversion rate is such that
        # q_i * C_{i, j} = q_j
        dst_currency_idx = self.extras.ordered_currencies.currency_to_idx[dst_currency]
        conversion_rate_at_step = self.tensors.conversion_rate[
            torch.arange(batch_size), torch.tensor(step), :, dst_currency_idx
        ]
        portfolio_value = (
            portfolio.reshape(batch_size, n_currencies) * conversion_rate_at_step
        ).sum(dim=1)
        return portfolio_value

    def get_conversion_rate_at_step(self, step: int) -> ConversionRateAtStep:
        return ConversionRateAtStep(
            conversion_rate_at_step=self.tensors.conversion_rate[:, step, :, :]
            .to(self.device)
            .unsqueeze(1),  # seq dim
            timestamps=self.timestamps[:, step].unsqueeze(1),  # seq dim
            ordered_currencies=self.extras.ordered_currencies,
        )


class SubCandleTensor(TimedTensor):
    def __init__(
        self,
        sub_candle: torch.Tensor,
        timestamps: torch.Tensor,
        candle_type: CandlesIndices,
    ) -> None:
        super().__init__(
            timestamps,
            tensors={"sub_candle": sub_candle},
            extras={"candle_type": candle_type},
        )

    @property
    def sub_candle(self) -> torch.Tensor:
        return self.tensors.sub_candle

    @property
    def candle_type(self) -> CandlesIndices:
        return self.extras.candle_type

    def get_conversion_rate(self) -> ConversionRate:
        assert self.candle_type != CandlesIndices.VOLUME
        conversion_rate = (
            # TODO: check again that it is in the right order
            # It should be, but just to be safe at some point
            torch.tensor(
                [
                    [
                        [1] * self.seq_len,
                        self.tensors.sub_candle[i],
                        1 / self.tensors.sub_candle[i],
                        [1] * self.seq_len,
                    ]
                    for i in range(self.batch_size)
                ],
                dtype=torch.float64,
            )
            .transpose(1, 2)
            .reshape(self.batch_size, self.seq_len, 2, 2)
            .transpose(2, 3)
        ).to(self.device)
        return ConversionRate(
            conversion_rate=conversion_rate,
            timestamps=self.timestamps,
            ordered_currencies=OrderedCurrencies([Currency.USD, Currency.BTC]),
        )
        # TODO: Define the ordered_currencies from upstream


class CandlesTensor(TimedTensor):
    def __init__(
        self,
        candles: TorchCandles,
        timestamps: torch.Tensor,
    ) -> None:
        super().__init__(timestamps=timestamps, tensors={"candles": candles})
        self.sanity_check()

    @property
    def candles(self) -> TorchCandles:
        return self.tensors.candles

    def sanity_check(self) -> None:
        super().sanity_check()
        _, _, dim = self.candles.shape
        assert dim == CANDLE_DIM

    def get_variable(self, candle_type: CandlesIndices) -> SubCandleTensor:
        return SubCandleTensor(
            sub_candle=self.candles[:, :, candle_type.value].unsqueeze(2),
            timestamps=self.timestamps,
            candle_type=candle_type,
        )

    @staticmethod
    def from_candles(candles: List[Candle]) -> "CandlesTensor":
        torch_candles = torch.vstack(
            [candle.to_torch() for candle in candles]
        ).unsqueeze(
            0
        )  # batch dim
        timestamps = torch.tensor(
            [candle.timestamp for candle in candles], dtype=torch.float64
        )
        timestamps = timestamps.unsqueeze(0).unsqueeze(-1)  # bs, seq_len, hidden
        return CandlesTensor(torch_candles, timestamps)


@dataclass
class ModelTargets:
    candles_tensor: CandlesTensor
    oracle_decisions: DecisionsTensor

    def __post_init__(self) -> None:
        assert self.candles_tensor.is_time_aligned(self.oracle_decisions)

    @property
    def timestamps(self) -> torch.Tensor:
        return self.candles_tensor.timestamps

    def to(self, device: torch.device) -> "ModelTargets":
        self.oracle_decisions.to(device)
        self.candles_tensor.to(device)
        return self

    def split_train_test(
        self, train_test_time_junction: float
    ) -> Tuple["ModelTargets", "ModelTargets"]:
        train_mask = (
            self.oracle_decisions.timestamps[:, -1, 0] < train_test_time_junction
        )
        test_mask = (
            self.oracle_decisions.timestamps[:, -1, 0] >= train_test_time_junction
        )

        assert (train_mask * test_mask).sum() == 0

        train_timestamps = self.candles_tensor.timestamps[train_mask]
        test_timestamps = self.candles_tensor.timestamps[test_mask]

        train_candles = self.candles_tensor.candles[train_mask]
        test_candles = self.candles_tensor.candles[test_mask]

        train_oracle_decisions = self.oracle_decisions.decisions[train_mask]
        test_oracle_decisions = self.oracle_decisions.decisions[test_mask]

        train_targets = ModelTargets(
            candles_tensor=CandlesTensor(train_candles, train_timestamps),
            oracle_decisions=DecisionsTensor(
                train_oracle_decisions,
                train_timestamps,
                self.oracle_decisions.ordered_currencies,
            ),
        )
        test_targets = ModelTargets(
            candles_tensor=CandlesTensor(test_candles, test_timestamps),
            oracle_decisions=DecisionsTensor(
                test_oracle_decisions,
                test_timestamps,
                self.oracle_decisions.ordered_currencies,
            ),
        )
        return train_targets, test_targets

    def get_sub_targets(self, batch_indices: List[int]) -> "ModelTargets":
        return ModelTargets(
            candles_tensor=CandlesTensor(
                self.candles_tensor.candles[batch_indices],
                self.candles_tensor.timestamps[batch_indices],
            ),
            oracle_decisions=DecisionsTensor(
                self.oracle_decisions.decisions[batch_indices],
                self.oracle_decisions.timestamps[batch_indices],
                self.oracle_decisions.ordered_currencies,
            ),
        )


@dataclass
class ModelInputs:
    features_tensor: FeaturesTensor

    @property
    def timestamps(self) -> torch.Tensor:
        return self.features_tensor.timestamps

    def to(self, device: torch.device) -> "ModelInputs":
        self.features_tensor = self.features_tensor.to(device)
        return self

    def duplicate(self, n: int) -> "ModelInputs":
        return ModelInputs(
            features_tensor=FeaturesTensor(
                self.features_tensor.features.repeat(n, 1, 1),
                self.features_tensor.timestamps.repeat(n, 1, 1),
            )
        )

    def split_train_test(
        self, train_test_time_junction: float
    ) -> Tuple["ModelInputs", "ModelInputs"]:
        train_mask = (
            self.features_tensor.timestamps[:, -1, 0] < train_test_time_junction
        )
        test_mask = (
            self.features_tensor.timestamps[:, -1, 0] >= train_test_time_junction
        )

        assert (train_mask * test_mask).sum() == 0

        train_timestamps = self.features_tensor.timestamps[train_mask]
        test_timestamps = self.features_tensor.timestamps[test_mask]

        train_features = self.features_tensor.features[train_mask]
        test_features = self.features_tensor.features[test_mask]

        train_inputs = ModelInputs(
            features_tensor=FeaturesTensor(train_features, train_timestamps)
        )
        test_inputs = ModelInputs(
            features_tensor=FeaturesTensor(test_features, test_timestamps)
        )
        return train_inputs, test_inputs

    def get_sub_inputs(self, batch_indices: List[int]) -> "ModelInputs":
        return ModelInputs(
            features_tensor=FeaturesTensor(
                self.features_tensor.features[batch_indices],
                self.features_tensor.timestamps[batch_indices],
            )
        )


T = TypeVar("T")


@dataclass
class Buffer(Generic[T]):
    buffer: List[T]
    max_buffer_size: int

    def __post_init__(self) -> None:
        self.sanity_check()

    def sanity_check(self) -> None:
        assert len(self) <= self.max_buffer_size
        assert sorted(self.buffer, key=lambda c: c.timestamp) == self.buffer

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def last_value(self) -> T:
        return self.buffer[-1]

    def update(self, new_value: T) -> None:
        self.sanity_check()

        if len(self.buffer) == self.max_buffer_size:
            self.buffer = self.buffer[1:]
        self.buffer.append(new_value)

        self.sanity_check()
