from enum import Enum
from typing import List, Tuple

import torch
from torch.utils.data import TensorDataset

from contracts import Time
from features.config import FeaturizerConfig
from features.contracts import CANDLE_COLUMNS, FeaturesTensor
from models.config import OrderedCurrencies
from models.contracts import (
    CANDLE_DIM,
    CandlesTensor,
    DecisionsTensor,
    ModelInputs,
    ModelTargets,
)

EMPTY_TENSOR = torch.tensor(data=[], dtype=torch.float64)


class ModeKeys(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class BatchDim:
    FEATURES = 0
    TIMESTAMPS = 1
    CANDLES = 2
    ORACLE_DECISIONS = 3


def model_inputs_and_targets_to_batch_args(
    inputs: ModelInputs, targets: ModelTargets
) -> Tuple[torch.Tensor, ...]:
    args = [EMPTY_TENSOR] * 4

    args[BatchDim.FEATURES] = inputs.features_tensor.features
    args[BatchDim.TIMESTAMPS] = inputs.features_tensor.timestamps
    args[BatchDim.CANDLES] = targets.candles_tensor.candles
    args[BatchDim.ORACLE_DECISIONS] = targets.oracle_decisions.decisions

    assert targets.candles_tensor.is_time_aligned(inputs.features_tensor)

    for arg in args:
        assert not torch.equal(arg, EMPTY_TENSOR), (arg, EMPTY_TENSOR)

    return tuple(args)


def batch_to_model_inputs_and_targets(
    batch: Tuple[torch.Tensor, ...], ordered_currencies: OrderedCurrencies
) -> Tuple[ModelInputs, ModelTargets]:
    timestamps = batch[BatchDim.TIMESTAMPS]
    features = batch[BatchDim.FEATURES]
    candles = batch[BatchDim.CANDLES]
    oracle_decisions = batch[BatchDim.ORACLE_DECISIONS]

    inputs = ModelInputs(features_tensor=FeaturesTensor(features, timestamps))
    candles_tensor = CandlesTensor(candles, timestamps)
    oracle_decisions = DecisionsTensor(
        oracle_decisions, timestamps, ordered_currencies, sanity_check=False
    )
    targets = ModelTargets(
        candles_tensor=candles_tensor,
        oracle_decisions=oracle_decisions,
    )

    return inputs, targets


class Dataset(TensorDataset):
    def __init__(
        self,
        inputs: ModelInputs,
        targets: ModelTargets,
        features_name: List[str],
        featurizer_config: FeaturizerConfig,
        ordered_currencies: OrderedCurrencies,
    ) -> None:
        assert features_name[:CANDLE_DIM] == CANDLE_COLUMNS
        args = model_inputs_and_targets_to_batch_args(inputs, targets)
        super().__init__(*args)
        assert (
            len(self)
            == inputs.features_tensor.batch_size
            == targets.oracle_decisions.batch_size
            == targets.candles_tensor.batch_size
        ), "Size mismatch between tensors"

        assert torch.equal(inputs.timestamps, targets.timestamps)
        self.timestamps = inputs.timestamps
        self.inputs = inputs
        self.targets = targets
        self.features_name = features_name
        self.featurizer_config = featurizer_config
        self.ordered_currencies = ordered_currencies
        assert isinstance(self.ordered_currencies, OrderedCurrencies)

    def __len__(self):
        return self.tensors[0].size(0)

    def find_time_for_ratio(self, train_test_target_ratio: float) -> float:
        last_sequence_timestamp = self.timestamps[:, -1, 0]
        best_timestamp, _, _ = min(
            [
                (
                    timestamp,
                    i,
                    abs(i / len(last_sequence_timestamp) - train_test_target_ratio),
                )
                for i, timestamp in enumerate(last_sequence_timestamp)
            ],
            key=lambda x: x[2],
        )
        return best_timestamp.item()

    def get_sub_dataset(self, batch_indices: List[int]) -> "Dataset":
        inputs = self.inputs.get_sub_inputs(batch_indices)
        targets = self.targets.get_sub_targets(batch_indices)
        return self.__class__(
            inputs,
            targets,
            self.features_name,
            self.featurizer_config,
            self.ordered_currencies,
        )

    def split_train_test(
        self, train_test_target_ratio: float
    ) -> Tuple[Time, "Dataset", "Dataset"]:
        train_test_time_junction = self.find_time_for_ratio(train_test_target_ratio)
        train_inputs, test_inputs = self.inputs.split_train_test(
            train_test_time_junction
        )
        train_targets, test_targets = self.targets.split_train_test(
            train_test_time_junction
        )

        train_dataset = self.__class__(
            train_inputs,
            train_targets,
            self.features_name,
            self.featurizer_config,
            self.ordered_currencies,
        )
        test_dataset = self.__class__(
            test_inputs,
            test_targets,
            self.features_name,
            self.featurizer_config,
            self.ordered_currencies,
        )

        return train_test_time_junction, train_dataset, test_dataset
