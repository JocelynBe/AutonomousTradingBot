import json
import logging
import os
from typing import List, Optional

from constants import (
    ALIGNED_SLICES_FILENAME,
    BINANCE_FEE,
    CANDLES_FILENAME,
    FEATURES_FILENAME,
    N_WARM_UP_CANDLES,
    ORACLE_DECISIONS_FILENAME,
)
from data.contracts import NormalizedDataFrame
from exchange_api.contracts import Candle, Currency, ExchangePair
from features.actions.align_slices import get_aligned_slices
from features.actions.chunkify import get_chunks
from features.actions.compute_features import distributed_compute_features
from features.actions.compute_oracle import distributed_compute_oracle_decisions
from features.actions.handle_missing import handle_missing_timesteps
from features.config import FeaturizerConfig
from features.contracts import FeaturizerOutput
from models.contracts import CandlesTensor, DecisionsTensor
from utils import assert_equal
from utils.io_utils import load, write_json, write_pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("actions.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def read_normalized_candles_df(
    path_to_csv: str, start_time: float
) -> NormalizedDataFrame:
    normalized_df = NormalizedDataFrame.read_csv(path_to_csv)
    normalized_df.normalized_df = normalized_df.normalized_df[
        normalized_df.normalized_df.timestamp > start_time
    ]
    normalized_df.normalized_df = normalized_df.normalized_df.reset_index(drop=True)
    return normalized_df


def read_candles(path_to_csv: str, start_time: float) -> List[Candle]:
    logger.info(f"Reading {path_to_csv} from start time {start_time}ms")
    after_start_time = read_normalized_candles_df(path_to_csv, start_time)

    logger.info("Converting to candles")
    exchange_pair = ExchangePair(Currency.USD, Currency.BTC)
    candles = after_start_time.normalized_df.apply(
        lambda row: Candle(
            src_currency=exchange_pair.src_currency,
            dst_currency=exchange_pair.dst_currency,
            **row.to_dict(),
        ),
        axis=1,
    )
    return list(candles.values)


def make_candles_tensor_with_overlap(
    candles: List[Candle],
    seq_length: int = 800,
    suffix_padding: int = 200,
) -> CandlesTensor:
    # Suffix padding is necessary because the optimal decision at time t requires knowing where we are going
    # Ideally, the suffix_padding should be infinite but 200 seems like a good enough approximation

    candles_tensor = CandlesTensor.from_candles(candles)
    candles_tensor = candles_tensor.reshape_with_padding(
        seq_length=seq_length,
        prefix_padding=0,
        suffix_padding=suffix_padding,
    )
    return candles_tensor


def generate_features(
    slices: List[List[Candle]],
    featurizer_config: FeaturizerConfig,
    num_cpus: int,
    output_dir: str,
) -> None:
    output_features_path = os.path.join(output_dir, FEATURES_FILENAME)
    assert not os.path.exists(output_features_path), f"{output_features_path} exists"

    logger.info(f"num_cpus = {num_cpus}")
    chunks = get_chunks(slices, num_cpus)

    str_chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "slice_id": chunk.slice_id,
            "len(candles)": len(chunk.candles),
        }
        for chunk in chunks
    ]
    logger.info(f"chunks = {str_chunks}")
    sliced_features = distributed_compute_features(
        chunks,
        num_cpus,
        seq_length=sum(map(len, slices)) - len(slices) * N_WARM_UP_CANDLES,
        featurizer_config=featurizer_config,
    )

    assert_equal(len(slices), len(sliced_features))
    logger.info(f"Saving candles and features at {output_dir}")
    write_pickle(sliced_features, output_features_path)


def generate_optimal_decisions(
    slices: List[List[Candle]],
    featurizer_config: FeaturizerConfig,
    num_cpus: int,
    output_dir: str,
) -> None:
    output_oracle_decisions_path = os.path.join(output_dir, ORACLE_DECISIONS_FILENAME)
    assert not os.path.exists(
        output_oracle_decisions_path
    ), f"{output_oracle_decisions_path} exists"

    candles_tensors = CandlesTensor.merge(
        [
            make_candles_tensor_with_overlap(
                candles,
                suffix_padding=featurizer_config.oracle_config.optimal_decisions_end_seq_warmup,
            )
            for candles in slices
        ]
    )

    oracle_decisions = distributed_compute_oracle_decisions(
        candles_tensors,
        fee=featurizer_config.oracle_config.fee,
        smoothing_factor=featurizer_config.oracle_config.smoothing_factor,
        num_cpus=num_cpus,
    )

    # Removing suffix because optimal decisions are of bad quality towards the end
    # oracle_decisions.truncate_prefix(featurizer_config.oracle_config.window_size)
    oracle_decisions.truncate_suffix(
        featurizer_config.oracle_config.optimal_decisions_end_seq_warmup
    )
    oracle_decisions = oracle_decisions.reshape(1, -1)

    # Save Oracle Decisions
    write_pickle(oracle_decisions, output_oracle_decisions_path)


def generate_aligned_slices(output_dir: str) -> None:
    logger.info("Loading features and oracle decisions")

    features_path = os.path.join(output_dir, FEATURES_FILENAME)
    oracle_decisions_path = os.path.join(output_dir, ORACLE_DECISIONS_FILENAME)
    features: List[FeaturizerOutput] = load(features_path)
    oracle_decisions: DecisionsTensor = load(oracle_decisions_path)

    logger.info("Aligning slices")
    slices = get_aligned_slices(features, oracle_decisions)

    # Save slices
    output_path = os.path.join(output_dir, ALIGNED_SLICES_FILENAME)
    write_pickle(slices, output_path)


def run_datagen(
    path_to_csv: str,
    start_time: float,
    output_dir: str,
    interval_in_minutes: int,
    num_cpus: int = 4,
    featurizer_config: Optional[FeaturizerConfig] = None,
) -> None:
    # e.g. path_to_csv = '../data/crypto/archive/btcusd.csv'
    # start_time = 1611982800000

    if not featurizer_config:
        # TODO: read featurizer_config.interval_in_minutes from the input directory
        featurizer_config = FeaturizerConfig()

    import time

    start = time.time()
    parameters = {
        "path_to_csv": path_to_csv,
        "start_time": start_time,
        "output_dir": output_dir,
        "featurizer_config": featurizer_config.to_json_dict(),
    }
    logger.info(
        f"Starting datagen with parameters: {json.dumps(parameters, indent=4, sort_keys=True)}"
    )

    # assert not os.path.exists(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    write_json(
        featurizer_config.to_json_dict(),
        os.path.join(output_dir, "featurizer_config.json"),
    )

    candles = read_candles(path_to_csv, start_time)
    output_candles_path = os.path.join(output_dir, CANDLES_FILENAME)
    # assert not os.path.exists(output_candles_path), f"{output_candles_path} exists"
    write_pickle(candles, output_candles_path)

    slices = handle_missing_timesteps(candles, interval_in_minutes)
    if not os.path.exists(os.path.join(output_dir, FEATURES_FILENAME)):
        generate_features(slices, featurizer_config, num_cpus, output_dir)
    generate_optimal_decisions(slices, featurizer_config, num_cpus, output_dir)
    generate_aligned_slices(output_dir)

    logger.info(f"Datagen finished, took {int(time.time() - start)} seconds")
