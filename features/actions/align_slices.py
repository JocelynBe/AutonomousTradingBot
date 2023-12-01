import copy
from typing import List, Tuple

import torch
import tqdm

from features.contracts import FeaturizerOutput
from models.contracts import DecisionsTensor


def assert_contiguous_mask(mask: torch.BoolTensor) -> None:
    non_zero_indices = mask.nonzero().reshape(-1).tolist()
    previous = None
    for idx in non_zero_indices:
        if previous is None:
            previous = idx
            continue
        assert idx - previous == 1
        previous = idx


def trim_oracle_decisions(
    features_slice: FeaturizerOutput, oracle_decisions: DecisionsTensor
) -> DecisionsTensor:
    min_ts = features_slice.timestamps.min().item()
    max_ts = features_slice.timestamps.max().item()
    mask_decisions = (
        (oracle_decisions.timestamps >= min_ts)
        & (oracle_decisions.timestamps <= max_ts)
    ).reshape(-1)
    assert_contiguous_mask(mask_decisions)

    oracle_decisions.apply_mask(mask_decisions)
    return oracle_decisions


def trim_features_slice(
    features_slice: FeaturizerOutput, oracle_decisions: DecisionsTensor
) -> FeaturizerOutput:
    min_ts = max(
        features_slice.timestamps.min().item(), oracle_decisions.timestamps.min().item()
    )

    max_ts = min(
        features_slice.timestamps.max().item(), oracle_decisions.timestamps.max().item()
    )

    mask_features = (
        (features_slice.timestamps >= min_ts) & (features_slice.timestamps <= max_ts)
    ).reshape(-1)
    assert_contiguous_mask(mask_features)

    features_slice.apply_mask(mask_features)
    return features_slice


def get_aligned_slices(
    features: List[FeaturizerOutput], oracle_decisions: DecisionsTensor
) -> List[Tuple[FeaturizerOutput, DecisionsTensor]]:
    slices = []
    for i, features_slice in tqdm.tqdm(enumerate(features)):
        oracle_decisions_slice = trim_oracle_decisions(
            features_slice, copy.deepcopy(oracle_decisions)
        )
        features_slice = trim_features_slice(features_slice, oracle_decisions_slice)
        assert oracle_decisions_slice.is_time_aligned(features_slice), i
        slices.append((features_slice, oracle_decisions_slice))

    return slices
