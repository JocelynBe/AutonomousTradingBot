import logging
import os.path
from typing import List, Tuple

from contracts import Time
from features.contracts import FeaturesTensor, FeaturizerOutput
from models.config import EncoderConfig
from models.contracts import CandlesTensor, DecisionsTensor, ModelInputs, ModelTargets
from trainer.config import TrainingConfig
from trainer.contracts import Dataset
from utils import ProgressBar, assert_equal
from utils.io_utils import load, write_pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("actions.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

TRAIN_CANDLES_FILENAME = "train_candles.pkl"
TEST_CANDLES_FILENAME = "test_candles.pkl"
TRAINING_CONFIG_FILENAME = "training_config.json"


def save_candles(
    path_to_candles: str, train_test_time_junction: Time, output_dir: str
) -> None:
    candles = load(path_to_candles)
    train_candles = [
        candle for candle in candles if candle.timestamp <= train_test_time_junction
    ]
    test_candles = [
        candle for candle in candles if candle.timestamp > train_test_time_junction
    ]
    write_pickle(train_candles, os.path.join(output_dir, TRAIN_CANDLES_FILENAME))
    write_pickle(test_candles, os.path.join(output_dir, TEST_CANDLES_FILENAME))


def save_datasets(
    train_dataset: Dataset, test_dataset: Dataset, output_dir: str
) -> None:
    write_pickle(train_dataset, os.path.join(output_dir, f"train_dataset.pkl"))
    write_pickle(test_dataset, os.path.join(output_dir, f"test_dataset.pkl"))


def reshape_slices(
    features_slice: FeaturizerOutput,
    oracle_decisions_slice: DecisionsTensor,
    encoder_config: EncoderConfig,
) -> Tuple[FeaturizerOutput, DecisionsTensor]:
    prefix_padding = encoder_config.rnn_warmup_steps
    suffix_padding = 0
    seq_length = encoder_config.seq_length

    expected_remainder = prefix_padding + suffix_padding
    total_candles_to_keep = (
        (features_slice.seq_len - expected_remainder) // seq_length
    ) * seq_length + expected_remainder

    features_slice.truncate_prefix(features_slice.seq_len - total_candles_to_keep)
    features_slice = features_slice.reshape_with_padding(
        seq_length=seq_length,  # > 0
        prefix_padding=prefix_padding,  # >= 0
        suffix_padding=suffix_padding,
    )

    oracle_decisions_slice.truncate_prefix(
        oracle_decisions_slice.seq_len - total_candles_to_keep
    )
    oracle_decisions_slice = oracle_decisions_slice.reshape_with_padding(
        seq_length=seq_length,  # > 0
        prefix_padding=prefix_padding,  # >= 0
        suffix_padding=suffix_padding,
    )

    assert_equal(features_slice.batch_size, oracle_decisions_slice.batch_size)
    assert_equal(features_slice.seq_len, oracle_decisions_slice.seq_len)
    assert features_slice.is_time_aligned(oracle_decisions_slice)

    return features_slice, oracle_decisions_slice


def stack_slices(
    slices: List[Tuple[FeaturizerOutput, DecisionsTensor]],
    encoder_config: EncoderConfig,
) -> Tuple[FeaturizerOutput, DecisionsTensor]:
    logger.info("Reshaping")
    features, oracle_decisions = [], []
    print(
        "[slice_[0].shape for slice_ in slices]",
        [slice_[0].shape for slice_ in slices],
    )
    # slices = [slices[-1]]  # temporary test
    for features_slice, oracle_decisions_slice in ProgressBar(slices):
        features_slice, oracle_decisions_slice = reshape_slices(
            features_slice,
            oracle_decisions_slice,
            encoder_config,
        )
        features.append(features_slice)
        oracle_decisions.append(oracle_decisions_slice)

    logger.info("Merging")
    featurizer_output = FeaturizerOutput.merge(features)
    decisions_tensor = DecisionsTensor.merge(oracle_decisions)
    return featurizer_output, decisions_tensor


def preprocess_data(
    path_to_candles: str,
    path_to_aligned_slices: str,
    training_config: TrainingConfig,
    output_dir: str,
) -> tuple[Dataset, Dataset]:
    os.makedirs(output_dir, exist_ok=True)
    slices = load(path_to_aligned_slices)
    featurizer_output, decisions_tensor = stack_slices(
        slices, training_config.model_config.encoder_config
    )

    features_tensor = FeaturesTensor(
        features=featurizer_output.features, timestamps=featurizer_output.timestamps
    )
    candles_tensor = CandlesTensor(
        candles=featurizer_output.candles, timestamps=featurizer_output.timestamps
    )

    inputs = ModelInputs(features_tensor)
    targets = ModelTargets(candles_tensor, decisions_tensor)
    dataset = Dataset(
        inputs,
        targets,
        features_name=featurizer_output.column_names,
        featurizer_config=featurizer_output.featurizer_config,
        ordered_currencies=decisions_tensor.ordered_currencies,
    )
    train_test_time_junction, train_dataset, test_dataset = dataset.split_train_test(
        train_test_target_ratio=training_config.train_test_ratio
    )
    logger.info(
        f"Generated train/test dataset with lengths {len(train_dataset)} and {len(test_dataset)}"
    )
    logger.info(
        f"Features have shape: \n"
        f"\t train_dataset.inputs.features_tensor.shape = {train_dataset.inputs.features_tensor.shape} \n"
        f"\t train_dataset.targets.candles_tensor.shape = {train_dataset.targets.candles_tensor.shape} \n"
        f"\t train_dataset.targets.candles_tensor.shape = {train_dataset.targets.oracle_decisions.shape} \n"
        f"\t test_dataset.inputs.features_tensor.shape = {test_dataset.inputs.features_tensor.shape} \n"
        f"\t test_dataset.targets.candles_tensor.shape = {test_dataset.targets.candles_tensor.shape} \n"
        f"\t test_dataset.targets.candles_tensor.shape = {test_dataset.targets.oracle_decisions.shape} \n"
    )

    save_candles(path_to_candles, train_test_time_junction, output_dir)
    save_datasets(train_dataset, test_dataset, output_dir)

    return train_dataset, test_dataset
