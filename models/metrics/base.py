import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from constants import TRAINING_DIR
from models.abstract import AbstractModelWrapper
from models.contracts import DecisionsTensor, ModelTargets
from models.loss import DiagonalLoss, PortfolioOracleLoss, RealizedProfitsLoss
from models.metrics.abstract import AbstractMetricOperator, AbstractMetricsComputer
from models.metrics.contracts import MetricsDict, MetricValue
from trainer.config import TrainingConfig
from trainer.contracts import ModeKeys


class Return(AbstractMetricOperator):
    NAME = "return"

    def __init__(self, training_config: TrainingConfig):
        self.criterion = RealizedProfitsLoss(training_config)
        self.rnn_warmup_steps = (
            training_config.model_config.encoder_config.rnn_warmup_steps
        )

    @staticmethod
    def boostrap_interval(
        gain_amounts: List[float],
    ) -> Tuple[
        MetricValue, MetricValue, MetricValue, MetricValue, MetricValue, MetricValue
    ]:
        df = pd.DataFrame(gain_amounts, columns=["gain_amount"])
        assert all(df.gain_amount > -1), f"df = {str(df)} \n Lost everything :("

        averages = []
        for _ in range(100):
            sample = df.sample(len(df), replace=True).values.reshape(-1)

            # The gain is expressed as (final_amount - init_amount) / init_amount
            # Therefore final_amount = (1 + sample) * init_amount
            sample = 1 + sample
            pi = np.exp(np.log(sample).sum())
            mu = pi ** (1 / len(df))
            averages.append(mu)

        stats_df = pd.DataFrame(averages)
        stats_df.columns = ["mu_2hours"]
        stats_df["mu_day"] = stats_df["mu_2hours"].apply(lambda x: x**12)
        stats_df["mu_month"] = stats_df["mu_2hours"].apply(lambda x: (x**12) ** 30)
        described = stats_df.describe(percentiles=[0.05, 0.5, 0.95])
        day_mu = round(described.mu_day.iloc[1], 3)
        day_p05, day_p95 = round(described.mu_day.iloc[4], 3), round(
            described.mu_day.iloc[6], 3
        )
        two_hours_mu = round(described["mu_2hours"].iloc[1], 3)
        two_hours_p05, two_hours_p95 = round(described["mu_2hours"].iloc[4], 3), round(
            described["mu_2hours"].iloc[6], 3
        )

        return day_mu, day_p05, day_p95, two_hours_mu, two_hours_p05, two_hours_p95

    def _measure(
        self, predictions: List[Tuple[DecisionsTensor, ModelTargets]]
    ) -> MetricsDict:
        gain_amounts = []
        for decisions, targets in predictions:
            decisions.truncate_prefix(self.rnn_warmup_steps)
            targets.candles_tensor.truncate_prefix(self.rnn_warmup_steps)
            gain_amounts.extend(
                self.criterion.compute_gain(decisions, targets.candles_tensor).tolist()
            )

        (
            day_mu,
            day_p05,
            day_p95,
            two_hours_mu,
            two_hours_p05,
            two_hours_p95,
        ) = self.boostrap_interval(gain_amounts)
        return {
            f"{self.NAME}_day_mu": day_mu,
            f"{self.NAME}_day_p05": day_p05,
            f"{self.NAME}_day_p95": day_p95,
            f"{self.NAME}_two_hours_mu": two_hours_mu,
            f"{self.NAME}_two_hours_p05": two_hours_p05,
            f"{self.NAME}_two_hours_p95": two_hours_p95,
            f"{self.NAME}_avg_gain_amounts": float(np.mean(gain_amounts)),
        }


class PercentageFromTo(AbstractMetricOperator):
    NAME = "percentage_from_to"

    def __init__(self, training_config: TrainingConfig):
        self.criterion = DiagonalLoss(training_config)

    def _measure(
        self, predictions: List[Tuple[DecisionsTensor, ModelTargets]]
    ) -> MetricsDict:
        percentage_usd_to_btc = []
        percentage_btc_to_usd = []
        for decisions, _ in predictions:
            usd_to_btc_p, btc_to_usd_p = self.criterion.percentage_on_antidiagonal(
                decisions
            )
            percentage_usd_to_btc.append(float(usd_to_btc_p.item()))
            percentage_btc_to_usd.append(float(btc_to_usd_p.item()))

        return {
            f"percentage_usd_to_btc": float(np.mean(percentage_usd_to_btc)),
            f"percentage_btc_to_usd": float(np.mean(percentage_btc_to_usd)),
        }


class PortfolioOracleLossMetric(AbstractMetricOperator):
    NAME = "portfolio_oracle_loss"

    def __init__(self, training_config: TrainingConfig):
        self.criterion = PortfolioOracleLoss(training_config)

    def _measure(
        self, predictions: List[Tuple[DecisionsTensor, ModelTargets]]
    ) -> MetricsDict:
        test_losses = []
        for decisions, targets in predictions:
            test_losses.append(
                self.criterion.forward(decisions, targets).detach().item()
            )

        return {f"{self.NAME}_mu": float(np.mean(test_losses))}


class BaseMetricsComputer(AbstractMetricsComputer):
    METRIC_OPERATORS_CLS = [Return, PercentageFromTo, PortfolioOracleLossMetric]

    def __init__(self, output_dir: str, training_config: TrainingConfig):
        self.output_dir = output_dir
        tensorboard_dir = os.path.join(output_dir, TRAINING_DIR, "tensorboard")
        os.makedirs(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)
        self.training_config = training_config
        self.metric_operators = [
            metric_operator(training_config)
            for metric_operator in self.METRIC_OPERATORS_CLS
        ]

    def write_metrics(self, metrics: MetricsDict, mode: ModeKeys, step: int) -> None:
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalars(
                metric_name, {mode.value: metric_value}, global_step=step
            )

    def eval_model(
        self,
        model_wrapper: AbstractModelWrapper,
        dataloader: DataLoader,
        device: torch.device,
        mode: ModeKeys,
        step: int,
    ) -> MetricsDict:
        predictions = model_wrapper.predict_dataloader(dataloader, mode)

        metrics = {}
        for metric_operator in self.metric_operators:
            _metrics = metric_operator.measure(predictions, device=device)
            assert set(_metrics.keys()).intersection(metrics) == set()
            metrics.update(_metrics)

        self.write_metrics(metrics, mode, step)
        return metrics
