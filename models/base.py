from typing import Any, Dict

from models.abstract import AbstractModel
from models.config import ModelConfig
from models.contracts import DecisionsTensor, ModelInputs
from models.layers import Decisioner, Encoder


class BaseModel(AbstractModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.encoder = Encoder(config.encoder_config)
        self.decisioner = Decisioner(config.decisioner_config)

    def forward(self, model_inputs: ModelInputs) -> DecisionsTensor:
        encoded_seq = self.encoder(model_inputs.features_tensor.features)
        decisions_tensor = self.decisioner(
            encoded_seq=encoded_seq,
            timestamps=model_inputs.features_tensor.timestamps,
        )
        return decisions_tensor

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_json_dict(),
            "encoder": self.encoder.to_json_dict(),
            "decisioner": self.decisioner.to_json_dict(),
        }

    def hydrate_from_json(self, json_dict: Dict[str, Any]) -> None:
        self.encoder.hydrate_from_json(json_dict["encoder"])
        self.decisioner.hydrate_from_json(json_dict["decisioner"])

    @staticmethod
    def from_json_dict(json_dict: Dict[str, Any]) -> "BaseModel":
        config = ModelConfig.from_json_dict(json_dict["config"])
        base_model = BaseModel(config)
        base_model.hydrate_from_json(json_dict)
        return base_model
