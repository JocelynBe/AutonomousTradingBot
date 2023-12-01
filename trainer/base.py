import logging
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from constants import TRAINING_DIR
from models.abstract import AbstractLoss, AbstractModelWrapper
from models.loss import CombinedLoss, PortfolioOracleLoss, RealizedProfitsLoss
from models.metrics.base import BaseMetricsComputer
from models.metrics.contracts import MetricsDict, TrainingMetrics
from trainer.abstract import AbstractTrainer
from trainer.config import TrainingConfig
from trainer.contracts import Dataset, ModeKeys, batch_to_model_inputs_and_targets
from trainer.criterias import daily_return_mu, return_avg_gain_amounts
from utils import ProgressBar

logger = logging.getLogger(__name__)

MODELS_DIR = "models"


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int) -> None:  # To ensure reproducibility
    logger.info(f"Setting seed {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class BaseTrainer(AbstractTrainer):
    def __init__(
        self,
        training_config: TrainingConfig,
        output_dir: str,
        seed: int,
        criterion: Optional[AbstractLoss] = None,
    ):
        super().__init__(training_config, output_dir, seed)
        set_seed(self.seed)
        self.optimizer: Optional[Optimizer] = None
        self.criterion = criterion or CombinedLoss(training_config)
        self.criterion.update_fee(0)
        self.metrics_computer = BaseMetricsComputer(output_dir, training_config)
        self.training_metrics = TrainingMetrics(criteria=return_avg_gain_amounts)
        self.ordered_currencies = (
            self.training_config.model_config.decisioner_config.ordered_currencies
        )

    def save_model(self, model_wrapper: AbstractModelWrapper, output_path: str) -> None:
        model_wrapper.save_to_json(output_path)

    def fit_batch(
        self, model_wrapper: AbstractModelWrapper, batch: Tuple[torch.Tensor, ...]
    ) -> float:
        self.optimizer.zero_grad()

        inputs, targets = batch_to_model_inputs_and_targets(
            batch, ordered_currencies=self.ordered_currencies
        )
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        raw_decisions = model_wrapper.model.forward(inputs)
        loss = self.criterion.forward(raw_decisions, targets)

        loss.backward()
        self.optimizer.step()
        return loss.detach().item()

    @staticmethod
    def save_epoch(
        model_wrapper: AbstractModelWrapper, epoch: int, output_dir: str
    ) -> None:
        model_dir = os.path.join(output_dir, TRAINING_DIR, MODELS_DIR)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_filepath = os.path.join(model_dir, f"model_wrapper_{epoch}.json")
        model_wrapper.save_to_json(model_filepath)

    def fit_epoch(
        self,
        epoch: int,
        model_wrapper: AbstractModelWrapper,
        train_dataloader: DataLoader,
    ) -> MetricsDict:
        model_wrapper.train()

        train_losses = []
        n_batches = len(train_dataloader)
        batch_idx = 0
        for batch in ProgressBar(train_dataloader):
            self.update_criterion(epoch, batch_idx / n_batches)
            loss = self.fit_batch(model_wrapper, batch)
            train_losses.append(loss)
            batch_idx += 1

        return {"train_loss": float(np.mean(train_losses))}

    def eval_model(
        self,
        model_wrapper: AbstractModelWrapper,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        step: int,
    ) -> None:
        if (step + 1) % self.training_config.val_frequency != 0:
            return

        logger.info(f"eval_model step = {step}")
        train_metrics = self.metrics_computer.eval_model(
            model_wrapper,
            train_dataloader,
            device=self.device,
            mode=ModeKeys.TRAIN,
            step=step,
        )
        self.training_metrics.update(
            metrics_dict=train_metrics, step=step, mode=ModeKeys.TRAIN
        )
        val_metrics = self.metrics_computer.eval_model(
            model_wrapper,
            val_dataloader,
            mode=ModeKeys.VALIDATION,
            device=self.device,
            step=step,
        )
        self.training_metrics.update(
            metrics_dict=val_metrics, step=step, mode=ModeKeys.VALIDATION
        )

    def update_criterion(self, epoch: int, batch_ratio: float) -> None:
        fee_schedule = self.training_config.fee_schedule
        if fee_schedule == "zero":
            new_fee = 0.0
        elif fee_schedule == "super_smooth":
            if epoch <= 40:
                new_fee = 0.0
            else:
                assert 0 <= batch_ratio <= 1, batch_ratio
                n_epochs = self.training_config.optimizer_config.n_epochs - 40
                progress_ratio = (epoch - 40 + batch_ratio) / n_epochs
                target_fee = 2 * self.training_config.fee
                new_fee = progress_ratio * target_fee
        elif fee_schedule == "warmup":
            if epoch <= 20:
                new_fee = 0.0
            else:
                n_epochs = 80  # self.training_config.optimizer_config.n_epochs
                factor = 2
                fee = self.training_config.fee
                epoch = epoch - 20
                if epoch > 100:
                    new_fee = factor * fee
                else:
                    new_fee = (
                        fee
                        * factor
                        * (
                            1 + np.tanh((epoch - n_epochs / 2) / (n_epochs / 5))
                        )  # (1 + np.tanh((epoch - n_epochs / 1.8) / (n_epochs / 4)))
                        / 2
                    )
        elif fee_schedule == "linear":
            mid_step = self.training_config.optimizer_config.n_epochs // 2
            new_fee = epoch / mid_step * self.training_config.fee
        else:
            raise NotImplementedError(self.training_config.fee_schedule)

        self.criterion.update_fee(new_fee)

    def fit(
        self,
        model_wrapper: AbstractModelWrapper,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        for epoch in range(self.training_config.optimizer_config.n_epochs):
            train_loss = self.fit_epoch(epoch, model_wrapper, train_dataloader)
            train_loss["criterion_fee"] = self.criterion.fee
            self.metrics_computer.write_metrics(
                train_loss, mode=ModeKeys.TRAIN, step=epoch
            )
            self.training_metrics.update(
                metrics_dict=train_loss, step=epoch, mode=ModeKeys.TRAIN
            )
            self.eval_model(model_wrapper, train_dataloader, val_dataloader, epoch)
            self.training_metrics.print_epoch(epoch)
            self.save_epoch(model_wrapper, epoch, self.output_dir)

    def fit_eval_model(
        self,
        model_wrapper: AbstractModelWrapper,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> TrainingMetrics:
        torch.set_num_threads(6)

        # Moving model to the device
        model_wrapper.to(self.device)

        self.optimizer = optim.Adam(
            model_wrapper.model.parameters(),
            lr=self.training_config.optimizer_config.learning_rate,
        )

        generator = torch.Generator()
        generator.manual_seed(0)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_config.optimizer_config.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=generator,
            num_workers=self.training_config.optimizer_config.num_workers,
            pin_memory=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=2 * self.training_config.optimizer_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.optimizer_config.num_workers,
            pin_memory=True,
        )

        self.fit(model_wrapper, train_dataloader, val_dataloader)

        return self.training_metrics
