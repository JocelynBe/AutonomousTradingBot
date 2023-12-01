import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch.utils.data import DataLoader

from constants import EXPECTED_DELTA_MS, N_WARM_UP_CANDLES
from exchange_api.contracts import Candle
from features.base import BaseFeaturizer
from features.config import FeaturizerConfig
from features.contracts import (
    FeaturesTensor,
    FeaturizerOutput,
    SingleStepFeaturizerOutput,
)
from models.abstract import AbstractModel, AbstractModelWrapper
from models.base import BaseModel
from models.config import ModelConfig, OrderedCurrencies
from models.contracts import (
    Buffer,
    DecisionsTensor,
    ModelInputs,
    ModelTargets,
    RawDecision,
)
from models.mock import MockModel
from trainer.contracts import ModeKeys, batch_to_model_inputs_and_targets
from utils import assert_equal
from utils.git_utils import get_current_commit
from utils.io_utils import load_json

CandleBuffer = Buffer[Candle]
FeaturizerOutputBuffer = Buffer[SingleStepFeaturizerOutput]


class BaseModelWrapper(AbstractModelWrapper):
    MODEL_CLS: Type[BaseModel] = BaseModel
    FEATURIZER_CLS: Type[BaseFeaturizer] = BaseFeaturizer

    def __init__(
        self,
        model_config: ModelConfig,
        featurizer_config: FeaturizerConfig,
        ordered_currencies: OrderedCurrencies,
    ):
        super().__init__(model_config, featurizer_config, ordered_currencies)
        self._model = self.MODEL_CLS(model_config)
        self._featurizer = (
            None
            if featurizer_config is None
            else self.FEATURIZER_CLS(featurizer_config)
        )
        self.featurizer_buffer: Optional[CandleBuffer] = None
        self.encoder_buffer: Optional[FeaturizerOutputBuffer] = None

    def update_seq_length(self, new_seq_length: int) -> None:
        assert self.encoder_buffer is None
        self.model_config.encoder_config.seq_length = new_seq_length

    @property
    def model(self) -> BaseModel:
        self._model = self._model.double()
        return self._model

    @property
    def featurizer(self) -> BaseFeaturizer:
        return self._featurizer

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def warmup_featurizer_buffer(self, warmup_candles: List[Candle]) -> None:
        assert self.featurizer_buffer is None
        assert warmup_candles == sorted(
            warmup_candles, key=lambda candle: candle.timestamp
        )
        self.featurizer_buffer = Buffer(
            buffer=warmup_candles[-N_WARM_UP_CANDLES:],
            max_buffer_size=N_WARM_UP_CANDLES,
        )

    def warmup_encoder_buffer(self, warmup_candles: List[Candle]) -> None:
        assert self.encoder_buffer is None
        assert_equal(
            len(warmup_candles),
            N_WARM_UP_CANDLES + self.model_config.encoder_config.total_warmup_length,
        )

        featurizer_buffer = Buffer(
            buffer=warmup_candles[:N_WARM_UP_CANDLES], max_buffer_size=N_WARM_UP_CANDLES
        )
        featurizer_output = self.featurizer.predict(
            featurizer_buffer, warmup_candles[N_WARM_UP_CANDLES:]
        )

        self.encoder_buffer = Buffer(
            buffer=[featurizer_output[i] for i in range(featurizer_output.seq_len)],
            max_buffer_size=self.model_config.encoder_config.total_warmup_length,
        )
        assert_equal(
            len(self.encoder_buffer),
            self.model_config.encoder_config.rnn_warmup_steps
            + self.model_config.encoder_config.stabilization_steps,
        )

    def _warmup_buffers(self, warmup_candles: List[Candle]) -> None:
        self.warmup_featurizer_buffer(warmup_candles)
        self.warmup_encoder_buffer(warmup_candles)

    @property
    def is_warmed_up(self) -> bool:
        return self.featurizer_buffer is not None and self.encoder_buffer is not None

    def warmup(self, warmup_candles: List[Candle]) -> None:
        assert not self.is_warmed_up
        self._warmup_buffers(warmup_candles)
        assert self.is_warmed_up

    def predict_batch(self, inputs: ModelInputs) -> DecisionsTensor:
        inputs = inputs.to(self.device)
        decisions_tensor = self.model.forward(inputs)
        return decisions_tensor

    def predict_dataloader(
        self, dataloader: DataLoader, mode: ModeKeys
    ) -> List[Tuple[DecisionsTensor, ModelTargets]]:
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch_to_model_inputs_and_targets(
                    batch, ordered_currencies=self.ordered_currencies
                )
                decisions_tensor = self.predict_batch(inputs)
                predictions.append((decisions_tensor, targets))
        return predictions

    def featurize_last_candle(self, last_candle: Candle) -> SingleStepFeaturizerOutput:
        assert self.is_warmed_up
        assert self.featurizer_buffer is not None
        assert last_candle.timestamp > self.featurizer_buffer.last_value.timestamp
        if (
            last_candle.timestamp - self.featurizer_buffer.last_value.timestamp
            > EXPECTED_DELTA_MS
        ):
            delta = (
                last_candle.timestamp - self.featurizer_buffer.last_value.timestamp
            ) / EXPECTED_DELTA_MS
            logging.info(
                f"Missed step between {self.featurizer_buffer.last_value.timestamp} and {last_candle.timestamp}, delta = {delta}"
            )

        featurizer_output = self.featurizer.predict_last_candle(
            self.featurizer_buffer, last_candle
        )
        return featurizer_output

    def update_and_featurize(self, last_candle: Candle) -> FeaturizerOutput:
        featurizer_output = self.featurize_last_candle(last_candle)
        assert last_candle.timestamp > self.featurizer_buffer.last_value.timestamp
        self.featurizer_buffer.update(last_candle)
        self.encoder_buffer.update(featurizer_output)

        return FeaturizerOutput.from_single_steps(self.encoder_buffer.buffer)

    def update_and_predict(self, latest_candle: Candle) -> RawDecision:
        featurizer_output = self.update_and_featurize(latest_candle)
        assert (
            featurizer_output.seq_len
            > self.model_config.encoder_config.rnn_warmup_steps
        )
        model_inputs = ModelInputs(
            features_tensor=FeaturesTensor(
                features=featurizer_output.features,
                timestamps=featurizer_output.timestamps,
            )
        )

        with torch.no_grad():
            decisions = self.model(model_inputs)

        assert (
            decisions.batch_size == 1
        ), f"batch size is expected to be 1, not {decisions.batch_size}"
        last_decision = decisions.get_raw_decision_at_step(step_k=-1)
        return last_decision

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> "BaseModelWrapper":
        featurizer_config = FeaturizerConfig.from_json_dict(
            json_dict["featurizer_config"]
        )
        model_config: ModelConfig = ModelConfig.from_json_dict(
            json_dict["model_config"]
        )
        ordered_currencies: OrderedCurrencies = OrderedCurrencies.from_json_dict(
            json_dict["ordered_currencies"]
        )
        model_wrapper = cls(model_config, featurizer_config, ordered_currencies)
        model_wrapper.model.hydrate_from_json(json_dict["model"])

        return model_wrapper

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_json_dict(),
            "model_config": self.model_config.to_json_dict(),
            "featurizer_config": self.featurizer_config.to_json_dict(),
            "ordered_currencies": self.ordered_currencies.to_json_dict(),
            "commit_hash": get_current_commit(),
        }

    def save_to_json(self, output_path: str) -> None:
        json_dict = self.to_json_dict()
        with open(output_path, "w") as f:
            json.dump(json_dict, f)

    @classmethod
    def read_from_json(cls, output_path: str) -> "BaseModelWrapper":
        json_dict = load_json(output_path)
        return cls.from_json_dict(json_dict)


class MockModelWrapper(BaseModelWrapper):
    MODEL_CLS: Type[AbstractModel] = MockModel

    def __init__(
        self,
        model_config: ModelConfig,
        featurizer_config: FeaturizerConfig,
        ordered_currencies: OrderedCurrencies,
    ):
        super().__init__(model_config, featurizer_config, ordered_currencies)
        self.features: List[SingleStepFeaturizerOutput] = []

    def featurize_last_candle(self, last_candle: Candle) -> SingleStepFeaturizerOutput:
        featurizer_output = super().featurize_last_candle(last_candle)
        self.features.append(featurizer_output)
        return featurizer_output

    def _warmup_buffers(self, warmup_candles: List[Candle]) -> None:
        self.warmup_featurizer_buffer(warmup_candles)

    @property
    def is_warmed_up(self) -> bool:
        return self.featurizer_buffer is not None

    def update_and_featurize(self, last_candle: Candle) -> None:
        self.featurize_last_candle(last_candle)
        assert last_candle.timestamp > self.featurizer_buffer.last_value.timestamp
        self.featurizer_buffer.update(last_candle)

    def update_and_predict(self, latest_candle: Candle) -> None:
        self.update_and_featurize(latest_candle)
