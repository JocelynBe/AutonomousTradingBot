import functools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch
from box import Box

from utils import ProgressBar, assert_equal
from utils.io_utils import SerializableDataclass
from utils.tensor_utils import reshape_with_overlap

logger = logging.getLogger(__name__)

Time = float
NoneType = type(None)

T = TypeVar("T", bound="TimedTensor")


@dataclass
class ValueAtTimeX(SerializableDataclass):
    timestamp: Time


# TODO: make .tensors and .extras private with corresponding properties on child classes
class TimedTensor:
    def __init__(
        self,
        timestamps: torch.Tensor,
        tensors: Dict[str, torch.Tensor],
        extras: Optional[Dict[str, Any]] = None,
        sanity_check: bool = True,
    ):
        assert len(timestamps.shape) == 3 and timestamps.shape[2] == 1, timestamps.shape
        assert (
            timestamps.dtype == torch.float64
        ), "Timestamps should be float64 otherwise there are numerical issues"
        assert tensors
        for tensor_name, values in tensors.items():
            assert len(values.shape) >= 3, (
                tensor_name,
                values.shape,
            )  # batch_size, seq_len, hidden_dim
            assert values.shape[:1] == timestamps.shape[:1], (
                values.shape,
                timestamps.shape,
            )

        self.timestamps = timestamps
        self.tensors = Box(**tensors)
        self.extras = Box(**(extras or {}))

        tensor = list(self.tensors.values())[0]
        assert all(other.device == tensor.device for other in self.tensors.values())
        self._device = tensor.device

        if sanity_check:
            self.sanity_check()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def batch_size(self) -> int:
        return self.timestamps.shape[0]

    @property
    def seq_len(self) -> int:
        return self.timestamps.shape[1]

    @classmethod
    def merge(cls, timed_tensors: List[T]) -> T:
        assert timed_tensors

        def get_shape_without_batch_dim(
            timed_tensor: T,
        ) -> Dict[str, Tuple[int, ...]]:
            return {
                tensor_name: tuple(_tensor.shape[1:])
                for tensor_name, _tensor in timed_tensor.tensors.items()
            }

        first_tensor = timed_tensors[0]
        extras = first_tensor.extras
        tensors_shapes = get_shape_without_batch_dim(first_tensor)
        tensors_to_concat = {tensor_name: [] for tensor_name in first_tensor.tensors}
        timestamps_to_concat = []
        for tt in timed_tensors:
            timestamps_to_concat.append(tt.timestamps)
            assert_equal(tt.extras, extras)
            assert_equal(get_shape_without_batch_dim(tt), tensors_shapes)
            assert_equal(tt.timestamps.shape[1:], first_tensor.timestamps.shape[1:])
            for tensor_name, tensor in tt.tensors.items():
                tensors_to_concat[tensor_name].append(tensor)

        tensors = {
            tensor_name: torch.vstack(tensors)
            for tensor_name, tensors in tensors_to_concat.items()
        }
        timestamps = torch.vstack(timestamps_to_concat)
        return cls(timestamps=timestamps, **tensors, **extras)

    def reshape(self, *shape: int) -> T:
        assert len(shape) >= 2
        (
            batch_size,
            seq_len,
        ) = shape

        return self.__class__(
            timestamps=self.timestamps.reshape(batch_size, seq_len, 1),
            **{
                tensor_name: tensor.reshape(*shape, *tensor.shape[2:])
                for tensor_name, tensor in self.tensors.items()
            },
            **dict(self.extras),
        )

    def sanity_check(self) -> None:
        for tensor_name, shape in self.shape.items():
            assert tuple(shape[:2]) == (self.batch_size, self.seq_len), (
                tensor_name,
                shape,
                self.batch_size,
                self.seq_len,
            )
            assert len(shape) > 2

        assert torch.equal(
            self.timestamps, torch.sort(self.timestamps).values
        ), "Incorrect time order"

    def to(self, device: torch.device) -> T:
        for tensor_name, tensor in self.tensors.items():
            self.tensors[tensor_name] = tensor.to(device)
        self._device = device
        return self

    @property
    def shape(self) -> Box[str, torch.Size]:
        return Box(
            {tensor_name: tensor.shape for tensor_name, tensor in self.tensors.items()}
        )

    def is_time_aligned(self, other: T, arg_name: str = "timestamps") -> bool:
        return torch.equal(self.timestamps, getattr(other, arg_name))

    def truncate_prefix(self, prefix_length: int) -> None:
        for tensor_name, tensor in self.tensors.items():
            self.tensors[tensor_name] = self.tensors[tensor_name][:, prefix_length:]
        self.timestamps = self.timestamps[:, prefix_length:]

    def truncate_suffix(self, suffix_length: int) -> None:
        new_end = self.seq_len - suffix_length
        for tensor_name, tensor in self.tensors.items():
            self.tensors[tensor_name] = self.tensors[tensor_name][:, :new_end]
        self.timestamps = self.timestamps[:, :new_end]

    def reshape_with_padding(
        self,
        seq_length: int,  # > 0
        prefix_padding: int,  # >= 0
        suffix_padding: int,  # >= 0
    ) -> T:
        expected_remainder = prefix_padding + suffix_padding
        total_candles_to_keep = (
            (self.seq_len - expected_remainder) // seq_length
        ) * seq_length + expected_remainder

        self.truncate_prefix(self.seq_len - total_candles_to_keep)
        _reshape_with_padding = functools.partial(
            reshape_with_overlap,
            seq_length=seq_length,
            prefix_padding=prefix_padding,
            suffix_padding=suffix_padding,
        )
        return self.__class__(
            timestamps=_reshape_with_padding(self.timestamps),
            **{
                tensor_name: _reshape_with_padding(tensor)
                for tensor_name, tensor in self.tensors.items()
            },
            **dict(self.extras),
        )

    def apply_mask(self, mask: torch.BoolTensor) -> None:
        seq_len = mask.shape[0]
        assert seq_len <= self.seq_len

        self.timestamps = self.timestamps[:, mask]
        for tensor_name, tensor in self.tensors.items():
            self.tensors[tensor_name] = tensor[:, mask]
