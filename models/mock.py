from typing import Any, Dict

import torch

from models.base import BaseModel
from models.config import ModelConfig
from models.contracts import DecisionsTensor, FeaturesTensor


class MockModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def forward(self, features_tensor: FeaturesTensor) -> DecisionsTensor:
        assert False, "Fix this"
        return torch.tensor([-1])

    def to_json_dict(self) -> Dict[str, Any]:
        return {}

    def hydrate_from_json(self, json_dict: Dict[str, Any]) -> None:
        pass
