import unittest

import torch

from features.contracts import TorchFeatures
from models.base import BaseModel
from models.config import DecisionerConfig, EncoderConfig, ModelConfig
from models.layers import Decisioner, Encoder
from trainer.base import set_seed


class TestBaseModel(unittest.TestCase):
    def setUp(self) -> None:
        set_seed(0)
        self.batch_size = 8
        self.seq_length = 120
        self.input_embedding_dim = 5
        self.encoder_output_dim = 7
        self.n_currencies = 2

        self.encoder_config = EncoderConfig(
            input_embedding_dim=self.input_embedding_dim,
            output_hidden_dim=self.encoder_output_dim,
            seq_length=self.seq_length,
        )

        self.decisioner_config = DecisionerConfig(
            input_embedding_dim=self.encoder_output_dim
        )
        self.model_config = ModelConfig(
            encoder_config=self.encoder_config, decisioner_config=self.decisioner_config
        )

    def test_encoder(self):
        encoder = Encoder(self.encoder_config)
        features: TorchFeatures = torch.rand(
            (self.batch_size, self.seq_length, self.input_embedding_dim)
        )
        encoded_seq = encoder(features)
        self.assertEqual(encoded_seq.isnan().sum().item(), 0)
        self.assertEqual(
            encoded_seq.shape,
            (self.batch_size, self.seq_length, self.encoder_output_dim),
        )

    def test_decisioner(self):
        decisioner = Decisioner(self.decisioner_config)
        encoded_seq = torch.rand(
            (self.batch_size, self.seq_length, self.encoder_output_dim)
        )
        decisions = decisioner(encoded_seq)
        self.assertEqual(
            decisions.shape,
            (
                self.batch_size,
                self.seq_length,
                self.n_currencies,
                self.n_currencies,
            ),
        )

    def test_base_model(self):
        base_model = BaseModel(self.model_config)
        features: TorchFeatures = torch.rand(
            (self.batch_size, self.seq_length, self.input_embedding_dim)
        )
        decisions = base_model(features)
        self.assertEqual(
            decisions.shape,
            (
                self.batch_size,
                self.seq_length,
                self.n_currencies,
                self.n_currencies,
            ),
        )

        json_dict = base_model.to_json_dict()
        reconstructed_base_model = BaseModel.from_json_dict(json_dict)
        decisions_bis = reconstructed_base_model(features)
        self.assertTrue(torch.equal(decisions, decisions_bis))


if __name__ == "__main__":
    unittest.main()
