import base64
from io import BytesIO
from typing import Any, Dict

import torch
import torch.nn as nn


def dumps_torch_model(model: nn.Module) -> str:
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    model_str = base64.b64encode(model_bytes).decode("ascii")
    return model_str


def loads_torch_models(model: nn.Module, model_str: str) -> None:
    model_bytes = base64.b64decode(model_str)
    buffer = BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)


class SerializableModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "model_str": dumps_torch_model(self),
        }

    def hydrate_from_json(self, json_dict: Dict[str, Any]) -> None:
        loads_torch_models(self, json_dict["model_str"])
