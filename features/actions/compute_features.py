from collections import defaultdict
from typing import DefaultDict, List, Optional

import ray
from ray.actor import ActorHandle

from agents.mock import MockAgent
from backtest.utils import init_portfolio
from constants import BINANCE_FEE, N_WARM_UP_CANDLES
from exchange_api.contracts import Currency
from features.config import FeaturizerConfig
from features.contracts import CandleChunk, FeaturesChunk, FeaturizerOutput
from models.config import ModelConfig, OrderedCurrencies
from models.wrapper import MockModelWrapper
from utils import update_progress_bar
from utils.ray_utils import ProgressBar, RayProgressBar


def init_agent(
    ordered_currencies: OrderedCurrencies, featurizer_config: FeaturizerConfig
) -> MockAgent:
    model_config = ModelConfig()

    mock_model_wrapper = MockModelWrapper(
        model_config,
        featurizer_config,
        ordered_currencies,
    )

    mock_agent = MockAgent(mock_model_wrapper, BINANCE_FEE)
    return mock_agent


def compute_features(
    chunk: CandleChunk,
    featurizer_config: FeaturizerConfig,
    progress_bar: Optional[ActorHandle] = None,
) -> Optional[FeaturesChunk]:
    if len(chunk.candles) < N_WARM_UP_CANDLES:
        return None
    ordered_currencies = OrderedCurrencies([Currency.USD, Currency.BTC])
    mock_agent = init_agent(ordered_currencies, featurizer_config=featurizer_config)
    warmup_candles = chunk.candles[:N_WARM_UP_CANDLES]
    mock_agent.warmup(warmup_candles)
    portfolio = init_portfolio(0)
    progress_bar = progress_bar or ProgressBar(
        range(len(chunk.candles) - N_WARM_UP_CANDLES)
    )
    for latest_candle in chunk.candles[N_WARM_UP_CANDLES:]:
        mock_agent.update_and_decide(latest_candle, portfolio)
        update_progress_bar(progress_bar)
    features_chunks = mock_agent.model_wrapper.features
    if not features_chunks:
        return None
    return FeaturesChunk(
        chunk.slice_id,
        chunk.chunk_id,
        FeaturizerOutput.from_single_steps(features_chunks),
    )


@ray.remote
def remote_compute_features(
    chunk: CandleChunk,
    featurizer_config: FeaturizerConfig,
    progress_bar_actor: ActorHandle,
) -> Optional[FeaturesChunk]:
    return compute_features(chunk, featurizer_config, progress_bar_actor)


def distributed_compute_features(
    chunks: List[CandleChunk],
    num_cpus: int,
    seq_length: int,
    featurizer_config: FeaturizerConfig,
) -> List[FeaturizerOutput]:
    ray.init(num_cpus=num_cpus)
    ray_progress_bar = RayProgressBar(range(seq_length))
    tasks = [
        remote_compute_features.remote(
            chunk, featurizer_config, ray_progress_bar.progress_actor
        )
        for chunk in chunks
    ]
    ray_progress_bar.print_until_done()
    results = ray.get(tasks)
    ray.shutdown()

    slice_id_to_features_chunks: DefaultDict[int, List[FeaturesChunk]] = defaultdict(
        list
    )
    for features_chunk in results:
        slice_id_to_features_chunks[features_chunk.slice_id].append(features_chunk)

    for slice_id, features_chunks in slice_id_to_features_chunks.items():
        slice_id_to_features_chunks[slice_id] = sorted(
            features_chunks, key=lambda chunk: chunk.chunk_id
        )

    features_all_steps = [
        FeaturizerOutput.merge(
            [
                batch.features.reshape(-1, 1)
                for batch in slice_id_to_features_chunks[slice_id]
                if batch is not None
            ],
        ).reshape(1, -1)
        for slice_id in sorted(slice_id_to_features_chunks.keys())
    ]
    return features_all_steps
