import copy
from abc import abstractmethod
from typing import List, Tuple, Type

import torch
from torch.utils.data import DataLoader

from models.abstract import AbstractModelWrapper
from models.contracts import DecisionsTensor, ModelTargets
from models.metrics.contracts import MetricsDict
from trainer.contracts import ModeKeys


class AbstractMetricOperator:
    NAME: str

    @abstractmethod
    def _measure(
        self, predictions: List[Tuple[DecisionsTensor, ModelTargets]]
    ) -> MetricsDict:
        pass

    def measure(
        self,
        predictions: List[Tuple[DecisionsTensor, ModelTargets]],
        device: torch.device,
    ) -> MetricsDict:
        predictions = [
            (decisions.to(device), targets.to(device))
            for decisions, targets in predictions
        ]

        return self._measure(copy.deepcopy(predictions))


class AbstractMetricsComputer:
    METRIC_OPERATORS_CLS: List[Type[AbstractMetricOperator]]

    @abstractmethod
    def eval_model(
        self,
        model_wrapper: AbstractModelWrapper,
        dataloader: DataLoader,
        device: torch.device,
        mode: ModeKeys,
        step: int,
    ) -> MetricsDict:
        pass

    def write_metrics(self, metrics: MetricsDict, mode: ModeKeys, step: int) -> None:
        pass
