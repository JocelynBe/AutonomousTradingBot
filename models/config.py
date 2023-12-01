from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Iterator, List, Tuple

from exchange_api.contracts import Currency
from utils.io_utils import SerializableDataclass


@dataclass
class EncoderConfig(SerializableDataclass):
    input_embedding_dim: int = 295
    intermediary_layers: Tuple[int, ...] = (32,)  # Compressor
    compressor_dropout: float = 0.05
    internal_hidden_dim: int = 64
    num_layers: int = 2
    output_hidden_dim: int = 16
    rnn_type: str = "lstm"
    seq_length: int = 120  # 120 minutes
    rnn_warmup_steps: int = (
        30  # 30 minutes, https://github.com/pyannote/pyannote-audio/issues/158
    )
    stabilization_steps: int = 80

    @property
    def total_warmup_length(self) -> int:
        return self.rnn_warmup_steps + self.stabilization_steps


@dataclass(frozen=True)
class OrderedCurrencies:
    ordered_currencies: List[Currency]

    @cached_property
    def idx_to_currency(self) -> Dict[int, Currency]:
        return dict(enumerate(self.ordered_currencies))  # type: ignore

    @cached_property
    def currency_to_idx(self) -> Dict[Currency, int]:
        return {v: k for k, v in self.idx_to_currency.items()}

    def __hash__(self):
        return hash(tuple(self.ordered_currencies))

    def __iter__(self) -> Iterator[Currency]:
        yield from self.ordered_currencies

    def __len__(self) -> int:
        return len(self.ordered_currencies)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "ordered_currencies": [
                str(currency) for currency in self.ordered_currencies
            ],
        }

    @staticmethod
    def from_json_dict(json_dict: Dict[str, Any]) -> "OrderedCurrencies":
        json_dict["ordered_currencies"] = [
            value.split(".")[-1].lower() for value in json_dict["ordered_currencies"]
        ]
        ordered_currencies = [
            Currency(currency_str) for currency_str in json_dict["ordered_currencies"]
        ]
        return OrderedCurrencies(ordered_currencies)


@dataclass
class DecisionerConfig(SerializableDataclass):
    input_embedding_dim: int = 16
    ordered_currencies: OrderedCurrencies = OrderedCurrencies(
        [Currency.USD, Currency.BTC]
    )

    @property
    def n_currencies(self) -> int:
        return len(self.ordered_currencies)


@dataclass
class ModelConfig(SerializableDataclass):
    encoder_config: EncoderConfig = field(default_factory=lambda: EncoderConfig())
    decisioner_config: DecisionerConfig = field(
        default_factory=lambda: DecisionerConfig()
    )

    def __post_init__(self):
        assert (
            self.encoder_config.output_hidden_dim
            == self.decisioner_config.input_embedding_dim
        )
