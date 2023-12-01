import logging
import os.path
from typing import Optional, Type

import click

from constants import ALIGNED_SLICES_FILENAME, CANDLES_FILENAME
from models.abstract import AbstractModelWrapper
from models.wrapper import BaseModelWrapper
from trainer.abstract import AbstractTrainer
from trainer.base import BaseTrainer
from trainer.config import TrainingConfig
from trainer.pre_processing import TRAINING_CONFIG_FILENAME, preprocess_data
from utils.io_utils import load, write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("actions.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def train_model(
    path_to_candles: str,
    path_to_aligned_slices: str,
    output_dir: str,
    trainer_cls: Type[AbstractTrainer] = BaseTrainer,
    model_wrapper_cls: Type[AbstractModelWrapper] = BaseModelWrapper,
    training_config: Optional[TrainingConfig] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training"))

    logger.info(f"Starting model training")
    training_config = training_config or TrainingConfig()
    write_json(
        training_config.to_json_dict(),
        os.path.join(output_dir, TRAINING_CONFIG_FILENAME),
    )
    logger.info(f"training_config = {training_config}")

    logger.info(f"Loading aligned slices at path {path_to_aligned_slices}")

    logger.info(f"Preprocessing data")
    train_dataset_path = os.path.join(output_dir, f"train_dataset.pkl")
    test_dataset_path = os.path.join(output_dir, f"test_dataset.pkl")
    if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
        train_dataset = load(train_dataset_path)
        test_dataset = load(test_dataset_path)
    else:
        train_dataset, test_dataset = preprocess_data(
            path_to_candles, path_to_aligned_slices, training_config, output_dir
        )

    logger.info(f"Creating trainer with cls {trainer_cls.__name__}")
    trainer = trainer_cls(training_config, output_dir, seed=0)
    model_wrapper = model_wrapper_cls(
        training_config.model_config,
        train_dataset.featurizer_config,
        train_dataset.ordered_currencies,
    )

    logger.info(
        f"Kicking off training for {training_config.optimizer_config.n_epochs} epochs"
    )
    metrics = trainer.fit_eval_model(model_wrapper, train_dataset, test_dataset)
    metrics.save_to_json(output_dir)


@click.command()
@click.option("--features_dir", help="Path to features directory")
@click.option("--output_dir", help="Path to output dir")
def main(
    features_dir: str,
    output_dir: str,
) -> None:
    assert not os.path.exists(os.path.join(output_dir, "training"))
    path_to_candles = os.path.join(features_dir, CANDLES_FILENAME)
    path_to_aligned_slices = os.path.join(features_dir, ALIGNED_SLICES_FILENAME)
    return train_model(path_to_candles, path_to_aligned_slices, output_dir)


if __name__ == "__main__":
    main()
