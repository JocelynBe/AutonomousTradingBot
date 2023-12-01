import torch
from torch import nn


def reshape_with_overlap(
    tensor: torch.Tensor,
    seq_length: int,  # > 0
    prefix_padding: int,  # >= 0
    suffix_padding: int,  # >= 0
) -> torch.Tensor:
    (batch_size, n_time_steps), hidden_dims = tensor.shape[:2], tensor.shape[2:]
    assert batch_size == 1, batch_size
    new_batch_size, remainder = divmod(n_time_steps, seq_length)
    assert remainder == (prefix_padding + suffix_padding), (
        remainder,
        prefix_padding + suffix_padding,
    )
    flat_tensor = tensor.squeeze(1)

    tensors_seq = []
    start, end = 0, prefix_padding + seq_length + suffix_padding
    new_batch_size = 0
    while end <= n_time_steps:
        tensors_seq.append(flat_tensor[0, start:end, :].unsqueeze(0))
        start += seq_length
        end += seq_length
        new_batch_size += 1

    assert end - seq_length == n_time_steps, (end, seq_length, n_time_steps)

    res = torch.vstack(tensors_seq)
    assert res.shape == (
        (new_batch_size, seq_length + prefix_padding + suffix_padding) + hidden_dims
    )
    return res


def rolling_average(input_tensor: torch.Tensor, window_size: int) -> torch.Tensor:
    batch_size, seq_len, hidden_dim = input_tensor.shape
    assert window_size < seq_len
    res = input_tensor.detach().clone()
    rolling_mean_1d = nn.AvgPool1d(kernel_size=window_size, stride=1)
    rolling_tensor = rolling_mean_1d(input_tensor.transpose(1, 2)).transpose(1, 2)
    res[:, window_size - 1 :, :] = rolling_tensor
    return res
