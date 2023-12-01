import json
import logging
import os
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict

import pandas as pd
from tabulate import tabulate

from trainer.contracts import ModeKeys

logger = logging.getLogger(__name__)

MetricName = str
MetricValue = float
MetricsDict = Dict[MetricName, MetricValue]
Step = int
Criteria = Callable[[MetricsDict], float]

TRAINING_METRICS_FILENAME = "train_metrics.json"


class TrainingMetrics:
    def __init__(self, criteria: Criteria):
        self._metrics: DefaultDict[Step, Dict[ModeKeys, MetricsDict]] = defaultdict(
            dict
        )
        self.criteria = criteria

    def update(self, metrics_dict: MetricsDict, step: Step, mode: ModeKeys):
        if mode in self._metrics[step]:
            current_metrics = self._metrics[step][mode]
            existing_metrics = set(metrics_dict.keys()).intersection(
                set(current_metrics.keys())
            )
            assert existing_metrics == set(), existing_metrics
            self._metrics[step][mode].update(metrics_dict)
        else:
            self._metrics[step][mode] = metrics_dict

    def print_epoch(self, step: Step) -> None:
        metrics_df = pd.DataFrame(self._metrics[step])
        logger.info("\n" + tabulate(metrics_df, headers="keys", tablefmt="psql"))

    def to_json(self) -> Dict[str, Any]:
        json_dict = {
            "metrics": {
                step: {str(mode): dic for mode, dic in step_dic.items()}
                for step, step_dic in self._metrics.items()
            },
            "best_step": self.find_best_step(),
        }
        return json_dict

    def save_to_json(self, output_dir: str) -> None:
        with open(os.path.join(output_dir, TRAINING_METRICS_FILENAME), "w") as f:
            json.dump(self.to_json(), f, indent=4, sort_keys=True)

    def find_best_step(self) -> Step:
        best_step = max(
            [
                (step, self.criteria(self._metrics[step][ModeKeys.VALIDATION]))
                for step in self._metrics
                if ModeKeys.VALIDATION in self._metrics[step]
            ],
            key=lambda x: x[1],
        )[0]
        return best_step
