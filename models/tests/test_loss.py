import random
import unittest
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from deepdiff import DeepDiff
from parameterized import parameterized

from constants import BINANCE_FEE
from exchange_api.contracts import Currency
from models.config import EncoderConfig, OrderedCurrencies
from models.contracts import ConversionRate, DecisionsTensor, Step
from models.loss import RealizedProfitsLoss
from trainer.base import set_seed
from trainer.config import TrainingConfig
from utils.test_utils import (
    DecisionAtStep,
    get_conversion_rate_from_price,
    super_dumb_execute_trades,
)


def generate_random_decisions(
    seq_len: int, proba_decision_at_each_step: float = 0.2
) -> Dict[Step, DecisionAtStep]:
    decisions = {}
    for step in range(seq_len):
        if random.random() < proba_decision_at_each_step:
            decisions[step] = DecisionAtStep(
                usd_to_btc=random.random(), btc_to_usd=random.random()
            )
    return decisions


def generate_random_price(seq_len: int) -> List[float]:
    return list(1 + np.random.rand(seq_len))


class TestBaseLoss(unittest.TestCase):
    def setUp(self) -> None:
        set_seed(1)
        self.encoder_config = EncoderConfig()

    def _assertAlmostEqual(
        self, obj_a: Any, obj_b: Any, significant_digits: int = 7
    ) -> None:
        diff = DeepDiff(obj_a, obj_b, significant_digits=significant_digits)
        self.assertFalse(diff, msg=f"obj_a = {obj_a} \n obj_b = {obj_b} ")

    def get_base_loss(
        self,
        fee: float = BINANCE_FEE,
        seq_length: Optional[int] = None,
        rnn_warmup_steps: Optional[int] = None,
    ) -> RealizedProfitsLoss:
        if rnn_warmup_steps is None:
            rnn_warmup_steps = self.encoder_config.rnn_warmup_steps
        if seq_length is None:
            seq_length = self.encoder_config.seq_length

        training_config = TrainingConfig()
        training_config.fee = fee
        training_config.model_config.encoder_config.rnn_warmup_steps = rnn_warmup_steps
        training_config.model_config.encoder_config.seq_length = seq_length
        base_loss = RealizedProfitsLoss(training_config=training_config)
        return base_loss

    @parameterized.expand(
        [
            (0.1, [[1.0, 0.9], [0.9, 1.0]]),
            (0.01, [[1.0, 0.99], [0.99, 1.0]]),
            (0.02, [[1.0, 0.98], [0.98, 1.0]]),
            (0.001, [[1.0, 0.999], [0.999, 1.0]]),
        ]
    )
    def test_fees(self, fee: float, expected_transformation: List[List[int]]):
        base_loss = self.get_base_loss(fee=fee)
        transformation = base_loss.fees
        self._assertAlmostEqual(
            transformation.tolist(), expected_transformation, significant_digits=7
        )

    @staticmethod
    def _get_leaky_decision_at_step(
        decision_at_step: Optional[DecisionAtStep], fee: float
    ) -> List[List[float]]:
        if decision_at_step is None:
            return [[1.0, 0], [0, 1.0]]

        decisions = torch.tensor(
            [
                [1 - decision_at_step.usd_to_btc, decision_at_step.usd_to_btc],
                [decision_at_step.btc_to_usd, 1 - decision_at_step.btc_to_usd],
            ]
        )
        fees = torch.tensor([[1.0, 1.0 - fee], [1.0 - fee, 1.0]])
        _leaky_decisions = fees * decisions
        return _leaky_decisions.tolist()

    @classmethod
    def _get_leaky_decisions(
        cls, decisions: Dict[Step, DecisionAtStep], fee: float, seq_len: int
    ) -> torch.Tensor:
        return torch.tensor(
            [
                cls._get_leaky_decision_at_step(decisions.get(step, None), fee)
                for step in range(seq_len)
            ]
        )

    @parameterized.expand(
        [
            (1.0, 10, 0),
            (1.0, 10, 0.1),
            (0.5, 10, 0),
            (0.5, 10, 0.1),
            (0.3, 10, 0.2),
            (0.7, 10, 0.3),
        ]
    )
    def test_get_leaky_decisions(self, amount: float, seq_len: int, fee: float) -> None:
        decisions = {
            3: DecisionAtStep(usd_to_btc=amount, btc_to_usd=0.0),
            7: DecisionAtStep(usd_to_btc=0.0, btc_to_usd=amount),
        }

        leaky_decisions = self._get_leaky_decisions(decisions, fee=fee, seq_len=seq_len)
        self._assertAlmostEqual(
            leaky_decisions[3].tolist(),
            [[1 - amount, amount * (1 - fee)], [0.0, 1.0]],
            significant_digits=7,
        )
        self._assertAlmostEqual(
            leaky_decisions[7].tolist(),
            [[1.0, 0.0], [amount * (1 - fee), 1 - amount]],
            significant_digits=7,
        )
        for i in range(seq_len):
            if i in [3, 7]:
                continue
            self.assertEqual(leaky_decisions[i].tolist(), [[1.0, 0.0], [0.0, 1.0]])

    @parameterized.expand(
        [
            (
                {
                    3: DecisionAtStep(usd_to_btc=1.0, btc_to_usd=0.0),
                    7: DecisionAtStep(usd_to_btc=0.0, btc_to_usd=1.0),
                },
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                100.0,
                0.0,
                0.0,
                400.0,
                0.0,
            ),
            (
                {
                    3: DecisionAtStep(usd_to_btc=1.0, btc_to_usd=0.0),
                    7: DecisionAtStep(usd_to_btc=0.0, btc_to_usd=1.0),
                },
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                100.0,
                0.0,
                0.5,
                100.0,
                0.0,
            ),
            (
                {
                    3: DecisionAtStep(usd_to_btc=1.0, btc_to_usd=0.0),
                    7: DecisionAtStep(usd_to_btc=0.0, btc_to_usd=1.0),
                },
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                100.0,
                0.0,
                0.1,
                324.0,
                0.0,
            ),
            (
                {
                    3: DecisionAtStep(usd_to_btc=0.5, btc_to_usd=0.0),
                    7: DecisionAtStep(usd_to_btc=0.0, btc_to_usd=0.5),
                },
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                100.0,
                0.0,
                0.1,
                131.0,
                45.0,
            ),
            (
                {
                    3: DecisionAtStep(usd_to_btc=0.5, btc_to_usd=0.0),
                    9: DecisionAtStep(usd_to_btc=0.0, btc_to_usd=0.5),
                },
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
                100.0,
                0.0,
                0.1,
                131.0,
                45.0,
            ),
        ]
    )
    def test_super_dumb_execute_trades(
        self,
        decisions: Dict[Step, DecisionAtStep],
        price_ts: List[float],
        amount_of_usd: float,
        amount_of_btc: float,
        fee: float,
        expected_final_amount_of_usd: float,
        expected_final_amount_of_btc: float,
    ):
        final_usd, final_btc = super_dumb_execute_trades(
            price_ts, decisions, amount_of_usd, amount_of_btc, fee, verbose=True
        )
        self.assertEqual(expected_final_amount_of_usd, final_usd)
        self.assertEqual(expected_final_amount_of_btc, final_btc)

    def test_dumb_execute_trades(self):
        batch_size, seq_len = 3, 100
        n_batches_to_generate = 10
        ordered_currencies = OrderedCurrencies([Currency.USD, Currency.BTC])
        payloads = {
            batch_idx: [None] * batch_size for batch_idx in range(n_batches_to_generate)
        }
        results = {
            batch_idx: [None] * batch_size for batch_idx in range(n_batches_to_generate)
        }
        for batch_idx in range(n_batches_to_generate):
            fee = random.random()
            for seq_idx in range(batch_size):
                random_decisions = generate_random_decisions(seq_len)
                random_price = generate_random_price(seq_len)
                amount_of_usd = 100 * random.random()
                amount_of_btc = 100 * random.random()
                final_usd, final_btc = super_dumb_execute_trades(
                    random_price,
                    random_decisions,
                    amount_of_usd,
                    amount_of_btc,
                    fee,
                    verbose=False,
                )
                payloads[batch_idx][seq_idx] = (
                    fee,
                    random_decisions,
                    random_price,
                    amount_of_usd,
                    amount_of_btc,
                )
                results[batch_idx][seq_idx] = (final_usd, final_btc)

        for batch_idx in range(n_batches_to_generate):
            (
                decisions,
                leaky_decisions,
                portfolio_init,
                conversion_rate,
                final_amounts,
                timestamps,
            ) = ([], [], [], [], [], [])
            for seq_idx in range(batch_size):
                (
                    fee,
                    random_decisions,
                    random_price,
                    amount_of_usd,
                    amount_of_btc,
                ) = payloads[batch_idx][seq_idx]
                decisions.append(random_decisions)
                leaky_decisions.append(
                    self._get_leaky_decisions(
                        random_decisions, fee=0, seq_len=seq_len
                    ).unsqueeze(0)
                )
                portfolio_init.append((amount_of_usd, amount_of_btc))
                conversion_rate.append(
                    get_conversion_rate_from_price(random_price, batch_size=1)
                )
                final_amounts.append(list(results[batch_idx][seq_idx]))
                timestamps.append(list(range(seq_len)))

            base_loss = self.get_base_loss(seq_length=seq_len, rnn_warmup_steps=0)
            res = base_loss.execute_trades(
                decisions=DecisionsTensor(
                    decisions=torch.vstack(decisions).to(torch.float64),
                    timestamps=torch.tensor(timestamps),
                    ordered_currencies=ordered_currencies,
                ),
                conversion_rate=ConversionRate(
                    conversion_rate=torch.vstack(conversion_rate).to(torch.float64),
                    timestamps=torch.tensor(timestamps),
                    ordered_currencies=ordered_currencies,
                ),
                portfolio_init=torch.tensor(portfolio_init).to(torch.float64),
            )

            self._assertAlmostEqual(
                final_amounts,
                [list(map(float, x)) for x in res.tolist()],
                significant_digits=4,
            )


if __name__ == "__main__":
    unittest.main()
