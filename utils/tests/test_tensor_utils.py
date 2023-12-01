import unittest

import torch

from utils.tensor_utils import reshape_with_overlap


class TestTensorUtils(unittest.TestCase):
    def test_reshape_with_overlap(self):
        batch_size = 1
        seq_length = 4203
        hidden_dim = 5

        prefix_padding = 3
        suffix_padding = 200
        sub_seq_length = 800

        candles_ts = (
            torch.arange(batch_size * seq_length * hidden_dim)
            .reshape(batch_size, hidden_dim, seq_length)
            .transpose(1, 2)
        )

        reshaped = reshape_with_overlap(
            candles_ts,
            seq_length=sub_seq_length,  # > 0
            prefix_padding=prefix_padding,  # >= 0
            suffix_padding=suffix_padding,  # >= 0
        )

        expected_batch_size = seq_length // sub_seq_length
        assert reshaped.shape == (
            expected_batch_size,
            prefix_padding + suffix_padding + sub_seq_length,
            hidden_dim,
        )

        new_start = prefix_padding
        new_end = sub_seq_length
        reshaped = reshaped[:, new_start:]
        reshaped = reshaped[:, :new_end]
        reshaped = reshaped.reshape(1, -1, hidden_dim)
        expected_new_length = seq_length - prefix_padding - suffix_padding
        assert reshaped.shape == (1, expected_new_length, hidden_dim)

        assert reshaped[:, :, 0][0].tolist() == list(
            range(prefix_padding, expected_new_length + prefix_padding)
        )


if __name__ == "__main__":
    unittest.main()
