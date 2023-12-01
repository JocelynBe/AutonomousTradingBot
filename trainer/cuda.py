from dataclasses import dataclass
from typing import List

import torch


@dataclass
class GPU:
    device_id: int

    def __post_init__(self):
        # Check device exists
        torch.cuda.get_device_properties(self.device_id)

    @property
    def name(self) -> str:
        return torch.cuda.get_device_properties(self.device_id)

    @property
    def available_memory(self) -> int:
        available_memory, _ = torch.cuda.mem_get_info(self.device_id)
        return available_memory

    @property
    def total_memory(self) -> int:
        _, total_memory = torch.cuda.mem_get_info(self.device_id)
        return total_memory

    @property
    def used_memory(self) -> int:
        available_memory, total_memory = torch.cuda.mem_get_info(self.device_id)
        return total_memory - available_memory

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.device_id}")


def get_gpus() -> List[GPU]:
    n_gpus = torch.cuda.device_count()
    return [GPU(device_id=device_id) for device_id in range(n_gpus)]


def get_device() -> torch.device:
    gpus = get_gpus()
    gpus = sorted(gpus, key=lambda gpu: gpu.available_memory, reverse=True)

    if gpus:
        return gpus[0].device

    return torch.device("cpu")
