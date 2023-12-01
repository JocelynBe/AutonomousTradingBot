import itertools
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import ray
import torch
from ray.actor import ActorHandle
from scipy.interpolate import UnivariateSpline

from exchange_api.contracts import Currency
from models.contracts import (
    CandlesIndices,
    CandlesTensor,
    ConversionRate,
    DecisionsTensor,
    SubCandleTensor,
)
from utils import ProgressBar, assert_equal, update_progress_bar
from utils.io_utils import write_pickle
from utils.ray_utils import RayProgressBar

END_NODE = "end_node"

StepIdx = int
CurrencyAtStep = Tuple[Currency, StepIdx]
Path = List[CurrencyAtStep]
BestPaths = Dict[CurrencyAtStep, Path]


def _add_step_to_graph(
    graph: nx.DiGraph,
    conversion_rate: ConversionRate,
    fee_matrix: np.ndarray,
    sample_idx: int,
    step_k: int,
    seq_len: int,
) -> nx.DiGraph:
    conversion_rate_at_step = conversion_rate.get_conversion_rate_at_step(step_k)
    conversion_rate_matrix = conversion_rate_at_step.conversion_rate_matrix()[
        sample_idx,
        0,
        :,
        :,  # 0 because there should be only one element in the sequence
    ].numpy()

    is_end_state = step_k == seq_len - 1
    if is_end_state:
        weights = conversion_rate_matrix
    else:
        weights = fee_matrix * conversion_rate_matrix

    neg_log_weights = -np.log(weights)
    assert not np.isnan(neg_log_weights).any(), (sample_idx, step_k, weights)

    ordered_currencies = conversion_rate.extras.ordered_currencies.ordered_currencies
    currency_to_idx = conversion_rate.extras.ordered_currencies.currency_to_idx
    for src_currency, dst_currency in itertools.product(ordered_currencies, repeat=2):
        i, j = currency_to_idx[src_currency], currency_to_idx[dst_currency]
        src_node = (src_currency, step_k)
        dst_node = (END_NODE, seq_len) if is_end_state else (dst_currency, step_k + 1)
        graph.add_edge(src_node, dst_node, weight=neg_log_weights[i, j])

    return graph


def build_graph(
    fee_matrix: np.ndarray, conversion_rate: ConversionRate, sample_idx: int = 0
) -> nx.DiGraph:
    """
    :param conversion_rate:
    :param fee_matrix:
    :param sample_idx: corresponding to the batch dimension
    """
    seq_len = conversion_rate.seq_len
    graph = nx.DiGraph()
    all_nodes = [
        [
            (currency, step_k),
            (currency, step_k + 1),
        ]
        for step_k in range(seq_len - 1)
        for currency in conversion_rate.extras.ordered_currencies
    ]
    all_nodes_flat = sum(all_nodes, [])
    graph.add_nodes_from([(END_NODE, seq_len)] + all_nodes_flat)
    for step_k in range(seq_len):
        graph = _add_step_to_graph(
            graph, conversion_rate, fee_matrix, sample_idx, step_k, seq_len
        )
    return graph


def _set_transitions_tensor(
    transitions_tensor: torch.DoubleTensor,
    currency_to_idx: Dict[Currency, int],
    best_paths: BestPaths,
    step_idx: int,
) -> torch.DoubleTensor:
    for currency, currency_idx in currency_to_idx.items():
        # Paths are reverted, end node to first node of the path
        # The second value (starting from the end) is the best node to transition from the first node of the path
        try:
            next_currency, check_idx = best_paths[(currency, step_idx)][-2]
        except KeyError as e:
            print("error", e)
            print("best_paths", best_paths)
            print("currency, step_idx", currency, step_idx)
            raise KeyError
        if next_currency == "end_node":
            transitions_tensor[step_idx, currency_idx, currency_idx] = 1
            continue
        assert step_idx + 1 == check_idx

        transitions_tensor[step_idx, currency_idx, currency_to_idx[next_currency]] = 1
    return transitions_tensor


def best_paths_to_decisions_tensor(
    best_paths: BestPaths,
    conversion_rate: ConversionRate,
) -> DecisionsTensor:
    n_currencies = len(conversion_rate.extras.ordered_currencies)
    currency_to_idx = conversion_rate.extras.ordered_currencies.currency_to_idx
    transitions_tensor = torch.zeros(
        conversion_rate.seq_len, n_currencies, n_currencies, dtype=torch.double
    )
    for step_idx in range(conversion_rate.seq_len):
        transitions_tensor = _set_transitions_tensor(
            transitions_tensor, currency_to_idx, best_paths, step_idx
        )

    return DecisionsTensor(
        decisions=transitions_tensor.unsqueeze(0),
        timestamps=conversion_rate.timestamps,
        ordered_currencies=conversion_rate.extras.ordered_currencies,
    )


def optimal_decision_tensor_for_sample(
    conversion_rate: ConversionRate, sample_idx: int, fee: float
) -> DecisionsTensor:
    seq_len = conversion_rate.seq_len
    fee_matrix = np.array([[1, 1 - fee], [1 - fee, 1]])

    graph = build_graph(fee_matrix, conversion_rate, sample_idx)
    best_paths = nx.single_source_bellman_ford_path(
        graph.reverse(), (END_NODE, seq_len)
    )
    decisions_tensor = best_paths_to_decisions_tensor(
        best_paths, conversion_rate.get_sample(sample_idx)
    )
    return decisions_tensor


def smooth_time_series(raw_ts: np.ndarray, smoothing_factor: float) -> np.ndarray:
    raw_ts = raw_ts / raw_ts[0]

    seq_len = len(raw_ts)
    indices = np.arange(seq_len)
    spline = UnivariateSpline(indices, raw_ts, s=smoothing_factor)
    smoothed_ts = spline(indices)
    return smoothed_ts


def smooth_close_price(
    close_price: SubCandleTensor, smoothing_factor: float
) -> SubCandleTensor:
    np_close_price = close_price.sub_candle.numpy()
    results = []
    for sample_idx in range(close_price.batch_size):
        smoothed_sample = smooth_time_series(
            np_close_price[sample_idx], smoothing_factor
        )
        results.append(smoothed_sample)

    smoothed_price = torch.tensor(np.vstack(results)).reshape(
        close_price.batch_size, close_price.seq_len, 1
    )

    assert_equal(smoothed_price.shape, close_price.sub_candle.shape)

    close_price.tensors.sub_candle = smoothed_price
    return close_price


def compute_oracle_decisions(
    candles: CandlesTensor,
    fee: float,
    smoothing_factor: float,
    progress_bar_actor: Optional[ActorHandle] = None,
) -> DecisionsTensor:
    close_price = candles.get_variable(candle_type=CandlesIndices.CLOSE)
    smoothed_close_price = smooth_close_price(
        close_price, smoothing_factor=smoothing_factor
    )
    conversion_rate = smoothed_close_price.get_conversion_rate()
    optimal_decisions_tensor = []
    progress_bar = progress_bar_actor or ProgressBar(range(candles.batch_size))
    for sample_idx in range(candles.batch_size):
        try:
            decisions_tensor = optimal_decision_tensor_for_sample(
                conversion_rate, sample_idx, fee
            )
        except KeyError:
            import random

            write_pickle(candles, f"/tmp/debug_{random.randint(0, int(1e8))}.pkl")
            assert False

        optimal_decisions_tensor.append(decisions_tensor)
        update_progress_bar(progress_bar)
    return DecisionsTensor.from_list(optimal_decisions_tensor)


@ray.remote
def remote_compute_oracle_decisions(
    candles: CandlesTensor,
    fee: float,
    smoothing_factor: float,
    progress_bar_actor: Optional[ActorHandle] = None,
) -> DecisionsTensor:
    return compute_oracle_decisions(candles, fee, smoothing_factor, progress_bar_actor)


def distributed_compute_oracle_decisions(
    candles_tensor: CandlesTensor, fee: float, smoothing_factor: float, num_cpus: int
) -> DecisionsTensor:
    num_chunks = min(candles_tensor.batch_size, num_cpus)
    candles_chunks = [
        CandlesTensor(candles_chunk, timestamps_chunk)
        for candles_chunk, timestamps_chunk in zip(
            candles_tensor.candles.chunk(num_chunks),
            candles_tensor.timestamps.chunk(num_chunks),
        )
    ]

    ray_progress_bar = None
    if num_cpus > 1:
        ray.init(num_cpus=num_cpus)
        ray_progress_bar = RayProgressBar(range(candles_tensor.batch_size))

    tasks = [
        remote_compute_oracle_decisions.remote(
            candles=chunk,
            fee=fee,
            smoothing_factor=smoothing_factor,
            progress_bar_actor=ray_progress_bar.progress_actor,
        )
        for chunk in candles_chunks
    ]
    ray_progress_bar.print_until_done()
    optimal_decisions_tensor = list(ray.get(tasks))
    if num_cpus > 1:
        ray.shutdown()

    write_pickle(optimal_decisions_tensor, "/tmp/optimal_decisions_tensor.pkl")

    decisions_tensor = DecisionsTensor.from_list(optimal_decisions_tensor)
    assert candles_tensor.is_time_aligned(decisions_tensor)

    return decisions_tensor
