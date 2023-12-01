import torch
from torch import nn as nn
from torch.nn.modules.transformer import _generate_square_subsequent_mask

from features.contracts import TorchFeatures
from models.config import DecisionerConfig, EncoderConfig
from models.contracts import DecisionsTensor
from models.serialization_utils import SerializableModule


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_layers: int, batch_first: bool
    ):
        super(TimeSeriesTransformer, self).__init__()

        # Defining the positional encoding
        self.pos_encoder = PositionalEncoding(input_dim)

        # Defining the transformer decoder layer
        decoder_layers = nn.TransformerDecoderLayer(d_model=input_dim, nhead=input_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, num_layers=num_layers
        )

        # Linear layer to map the output of transformer to the desired output size
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        _, seq_len, _ = x.shape
        # x.shape = [batch_size, seq_len, hidden_dim]
        x = x.permute(
            1, 0, 2
        )  # Convert to [seq_len, batch_size, hidden_dim] for transformer

        # Adding positional encoding
        x = self.pos_encoder(x)

        # Since transformer decoder requires memory, we use the input itself as memory.
        causal_mask = _generate_square_subsequent_mask(seq_len, device=x.device)
        output = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_is_causal=True,
            memory_is_causal=True,
        )

        # Convert back to [batch_size, seq_len, hidden_dim]
        output = output.permute(1, 0, 2)

        # Passing the output through the linear layer
        output = self.fc(output)

        return output, None


class TransformerDecoder(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_layers: int, batch_first: bool
    ):
        super().__init__()
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            input_dim,
            nhead=8,
            batch_first=batch_first,
        )
        self.transformer_decoder = transformer_decoder_layer
        if num_layers > 1:
            self.transformer_decoder = nn.TransformerDecoder(
                transformer_decoder_layer, num_layers=num_layers
            )

        self.output_layer = nn.Linear(input_dim, output_dim)
        self.output_activation = nn.ReLU()

    def forward(self, activated_features: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = activated_features.shape
        causal_mask = _generate_square_subsequent_mask(
            seq_len, device=activated_features.device
        )
        decoded = self.transformer_decoder(
            tgt=activated_features,
            memory=activated_features,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_is_causal=True,
            memory_is_causal=True,
        )
        return self.output_activation(self.output_layer(decoded))


class Encoder(SerializableModule):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.layer_norm_0 = nn.LayerNorm(self.config.input_embedding_dim)
        h_dims_in = (self.config.input_embedding_dim,) + self.config.intermediary_layers
        h_dims_out = self.config.intermediary_layers + (
            self.config.internal_hidden_dim,
        )
        self.compressor = nn.Sequential(
            *[
                layer
                for h_dim_in, h_dim_out in zip(h_dims_in, h_dims_out)
                for layer in (
                    nn.LayerNorm(h_dim_in),
                    nn.Linear(h_dim_in, h_dim_out),
                    nn.ReLU(),
                    nn.Dropout(config.compressor_dropout),
                )
            ]
        )
        # nn.Linear(self.config.input_embedding_dim, 2 * self.config.internal_hidden_dim)
        # self.layer_norm_1 = nn.LayerNorm(2 * self.config.internal_hidden_dim)
        self.activation = nn.ReLU()
        self.rnn_layer = self.get_rnn(config)
        self.layer_norm_2 = nn.LayerNorm(self.config.internal_hidden_dim)
        self.out_linear = nn.Linear(
            self.config.internal_hidden_dim, self.config.output_hidden_dim
        )

        # for name, param in self.rnn_layer.named_parameters():
        #    if "bias" in name:
        #        nn.init.constant_(param, 0.0)
        #    elif "weight" in name:
        #        nn.init.xavier_normal_(param)

    @staticmethod
    def get_rnn(config: EncoderConfig) -> nn.Module:
        if config.rnn_type == "gru":
            rnn_cls = nn.GRU
        elif config.rnn_type == "lstm":
            rnn_cls = nn.LSTM
        elif config.rnn_type == "transformer_decoder":
            rnn_cls = TimeSeriesTransformer
        else:
            raise NotImplementedError(f"config.rnn_type = {config.rnn_type}")

        return rnn_cls(
            config.internal_hidden_dim,
            config.internal_hidden_dim,
            batch_first=True,
            num_layers=config.num_layers,
        )

    def forward(self, features: TorchFeatures):
        # normed_features = self.layer_norm_0(features)
        compressed_features = self.compressor(
            features
        )  # self.in_linear(normed_features))
        # activated_features = self.activation(compressed_features)
        encoded_seq, _ = self.rnn_layer(compressed_features)
        normed_encoded_seq = self.layer_norm_2(encoded_seq)
        encoded_seq = self.out_linear(self.activation(normed_encoded_seq))
        return self.activation(encoded_seq)


class Decisioner(SerializableModule):
    def __init__(self, config: DecisionerConfig):
        super().__init__(config)
        self.n_currencies = config.n_currencies
        self.config = config
        self.layer_norm = nn.LayerNorm(self.config.input_embedding_dim)
        self.sequence = nn.Sequential(
            nn.Linear(self.config.input_embedding_dim, self.config.input_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.input_embedding_dim, self.config.input_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.input_embedding_dim, self.n_currencies**2),
        )
        self.activation = nn.Softmax(dim=-1)

    def forward(
        self, encoded_seq: torch.Tensor, timestamps: torch.Tensor
    ) -> DecisionsTensor:
        batch_size, seq_len, hidden_dim = encoded_seq.shape
        normed_encoded_seq = self.layer_norm(encoded_seq)
        decision = self.sequence(normed_encoded_seq).reshape(
            batch_size, seq_len, self.n_currencies, self.n_currencies
        )
        return DecisionsTensor(
            decisions=self.activation(decision),
            timestamps=timestamps,
            ordered_currencies=self.config.ordered_currencies,
            sanity_check=False,
        )
