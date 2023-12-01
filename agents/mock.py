import os
from typing import Dict, List, Type

import mock
from torch.utils.data import DataLoader

from agents.base import BaseAgent
from constants import N_WARM_UP_CANDLES
from contracts import Time
from exchange_api.contracts import Candle, Currency, Order, Portfolio
from models.contracts import CandlesIndices, RawDecision
from models.wrapper import BaseModelWrapper, MockModelWrapper
from trainer.config import TrainingConfig
from trainer.contracts import Dataset, ModeKeys
from utils.io_utils import load

CachedDecisions = Dict[Time, RawDecision]


def compute_cached_decisions(
    model_wrapper: BaseModelWrapper, training_dir: str
) -> CachedDecisions:
    test_dataset: Dataset = load(os.path.join(training_dir, f"test_dataset.pkl"))
    training_config = TrainingConfig()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2 * training_config.optimizer_config.batch_size,
        shuffle=False,
        num_workers=training_config.optimizer_config.num_workers,
        pin_memory=True,
    )
    predictions = model_wrapper.predict_dataloader(test_dataloader, mode=ModeKeys.TEST)
    cached_decisions = {}

    for decisions, _ in predictions:
        decisions.truncate_prefix(
            prefix_length=training_config.model_config.encoder_config.rnn_warmup_steps
        )

        for sample_idx in range(decisions.batch_size):
            for step_idx in range(decisions.seq_len):
                raw_decision = decisions.get_raw_decision_at_step(
                    sample_idx=sample_idx, step_k=step_idx
                )
                assert raw_decision.timestamp not in cached_decisions
                cached_decisions[raw_decision.timestamp] = raw_decision

    return cached_decisions


class CachedAgent(BaseAgent):
    MODEL_WRAPPER_CLS: Type[BaseModelWrapper] = MockModelWrapper
    model_wrapper: MockModelWrapper

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        fee: float,
        cached_decisions: CachedDecisions,
    ):
        super().__init__(model_wrapper, fee)
        self.cached_decisions = cached_decisions

    def warmup(self, warmup_candles: List[Candle]) -> None:
        pass

    def update_and_decide(
        self,
        latest_candle: Candle,
        portfolio: Portfolio,
    ) -> List[Order]:
        raw_decision = self.cached_decisions[latest_candle.timestamp]
        return self.simple_make_decision(
            raw_decision, portfolio, latest_candle
        )  # TODO: Use self.make_decision instead


class MockAgent(BaseAgent):
    MODEL_WRAPPER_CLS: Type[MockModelWrapper] = MockModelWrapper
    model_wrapper: MockModelWrapper

    def warmup(self, warmup_candles: List[Candle]) -> None:
        assert len(warmup_candles) == N_WARM_UP_CANDLES

        self.model_wrapper.warmup(warmup_candles)
        self.latest_candle = warmup_candles[-1]


class SimpleMockAgent(BaseAgent):
    MODEL_WRAPPER_CLS: Type[BaseModelWrapper] = MockModelWrapper
    model_wrapper: MockModelWrapper

    def __init__(self, fee: float):
        super().__init__(model_wrapper=mock.MagicMock(), fee=fee)

    def make_orders(
        self,
        raw_decision: RawDecision,
        portfolio: Portfolio,
        latest_candle: Candle,
    ) -> List[Order]:
        btc_price = latest_candle.close
        amount_of_usd_to_sell: float = portfolio[
            Currency.USD
        ] * raw_decision.src_to_dst(Currency.USD, Currency.BTC)
        amount_of_btc_to_sell: float = portfolio[
            Currency.BTC
        ] * raw_decision.src_to_dst(Currency.BTC, Currency.USD)
        sell_btc_order = self.get_sell_order(
            sell_amount=amount_of_btc_to_sell,
            latest_candle=latest_candle,
            raw_decision=raw_decision,
        )

        btc_buy_amount = amount_of_usd_to_sell / btc_price
        buy_btc_order = self.get_buy_order(
            buy_amount=btc_buy_amount,
            latest_candle=latest_candle,
            raw_decision=raw_decision,
        )
        return [buy_btc_order, sell_btc_order]
