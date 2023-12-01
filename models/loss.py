import torch

from exchange_api.contracts import Currency
from models.abstract import AbstractLoss
from models.contracts import (
    CandlesIndices,
    CandlesTensor,
    ConversionRate,
    ConversionRateAtStep,
    DecisionsTensor,
    ModelTargets,
)
from trainer.config import TrainingConfig
from utils import assert_equal

"""
For details about this file see theory.md
"""


class RealizedProfitsLoss(AbstractLoss):
    INIT_AMOUNT = 100

    def update_fee(self, new_fee: float) -> None:
        self.fee = new_fee

    @property
    def fees(self) -> torch.Tensor:
        fees = torch.full(
            (self.n_currencies, self.n_currencies),
            (1.0 - self.fee),
            dtype=torch.float64,
        )
        fees.fill_diagonal_(1.0)
        return fees  # [n_currencies, n_currencies]

    def execute_trades(
        self,
        decisions: DecisionsTensor,  # [bs, seq_len - warmup, n_currencies, n_currencies]
        conversion_rate: ConversionRate,  # [bs, seq_len - warmup, n_currencies, n_currencies]
        portfolio_init: torch.Tensor,  # [bs, n_currencies]
    ) -> torch.Tensor:  # [bs, n_currencies]
        assert (
            decisions.seq_len
            == conversion_rate.seq_len
            == self.training_config.model_config.encoder_config.seq_length
        ), (
            decisions.shape,
            conversion_rate.shape,
            self.training_config.model_config.encoder_config.seq_length,
        )
        batch_size, seq_len, n_currencies, n_currencies_bis = decisions.shape.decisions
        assert n_currencies == n_currencies_bis, (n_currencies, n_currencies_bis)
        assert portfolio_init.shape == (batch_size, n_currencies)

        assert conversion_rate.is_time_aligned(decisions)
        # TODO: implement the multiplication in the TimedTensor and do the assert there
        decisions_device = decisions.device
        leaky_transfers = (
            decisions.tensors.decisions
            * conversion_rate.tensors.conversion_rate
            * self.fees.to(decisions_device)
        )
        portfolio_init = portfolio_init.double().unsqueeze(1)
        aggregated = torch.vstack(
            [
                torch.linalg.multi_dot(
                    [leaky_transfers[b, i, :, :] for i in range(seq_len)]
                ).unsqueeze(0)
                for b in range(batch_size)
            ]
        )
        portfolio_final = torch.bmm(
            portfolio_init.reshape(batch_size, 1, n_currencies), aggregated
        )
        return portfolio_final.squeeze(2)

    def portfolio_init(
        self, batch_size: int, init_conversion_rate: ConversionRateAtStep
    ) -> torch.Tensor:  # [bs, n_currencies]
        return (
            torch.tensor(
                [
                    [
                        self.INIT_AMOUNT
                        / self.n_currencies
                        * init_conversion_rate.conversion_rate_from_currency_a_to_currency_b(
                            src_currency=Currency.USD,
                            dst_currency=self.ordered_currencies.idx_to_currency[
                                currency_idx
                            ],
                            sample_idx=sample_idx,
                        )
                        for currency_idx in range(self.n_currencies)
                    ]
                    for sample_idx in range(batch_size)
                ],
                dtype=torch.float64,
            )
            .to(torch.float64)
            .to(init_conversion_rate.device)
        )

    def compute_gain(
        self, decisions: DecisionsTensor, candles: CandlesTensor
    ) -> torch.Tensor:
        batch_size, seq_len, _, _ = decisions.shape.decisions
        assert seq_len == self.seq_length

        close_price = candles.get_variable(candle_type=CandlesIndices.CLOSE)
        conversion_rate = close_price.get_conversion_rate()
        assert conversion_rate.is_time_aligned(decisions)

        init_portfolio = self.portfolio_init(
            batch_size, conversion_rate.get_conversion_rate_at_step(step=0)
        )
        init_value = conversion_rate.compute_portfolio_value_at_step(
            init_portfolio, step=0
        )

        final_portfolio = self.execute_trades(
            decisions, conversion_rate, init_portfolio
        )
        final_amount = conversion_rate.compute_portfolio_value_at_step(
            final_portfolio, step=self.seq_length - 1
        )
        return (final_amount - init_value) / init_value

    def _forward(
        self, decisions: DecisionsTensor, targets: ModelTargets
    ) -> torch.Tensor:
        candles = targets.candles_tensor

        assert decisions.seq_len == self.rnn_warmup_steps + self.seq_length, (
            decisions.seq_len,
            self.rnn_warmup_steps,
            self.seq_length,
        )
        assert candles.seq_len == self.rnn_warmup_steps + self.seq_length

        decisions.truncate_prefix(self.rnn_warmup_steps)
        candles.truncate_prefix(self.rnn_warmup_steps)

        gains_loss = -1 * self.compute_gain(decisions, candles).mean()
        return gains_loss


class DiagonalLoss(AbstractLoss):
    def update_fee(self, new_fee: float) -> None:
        self.fee = new_fee

    @staticmethod
    def percentage_on_antidiagonal(
        decisions: DecisionsTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        usd_to_btc = torch.tensor([[0, 1], [0, 0]], device=decisions.device)
        btc_to_usd = torch.tensor([[0, 0], [1, 0]], device=decisions.device)
        normalized = decisions.decisions.sum(dim=1) / decisions.decisions.sum()
        usd_to_btc_percentage = (normalized * usd_to_btc).sum()
        btc_to_usd_percentage = (normalized * btc_to_usd).sum()
        return usd_to_btc_percentage, btc_to_usd_percentage

    def _forward(
        self, decisions: DecisionsTensor, targets: ModelTargets
    ) -> torch.Tensor:
        usd_to_btc_percentage, btc_to_usd_percentage = self.percentage_on_antidiagonal(
            decisions
        )
        return (
            torch.exp(-1000 * usd_to_btc_percentage)
            + torch.exp(-1000 * btc_to_usd_percentage)
        ) / 2


class PortfolioOracleLoss(AbstractLoss):
    def __init__(self, training_config: TrainingConfig):
        super().__init__(training_config)
        self.realized_profits_loss = RealizedProfitsLoss(training_config)

    def update_fee(self, new_fee: float) -> None:
        self.fee = new_fee
        self.realized_profits_loss.update_fee(new_fee)

    def decisions_to_portfolio(
        self, decisions: DecisionsTensor, candles_tensor: CandlesTensor
    ) -> torch.Tensor:
        batch_size, _, _, _ = decisions.shape.decisions
        close_price = candles_tensor.get_variable(candle_type=CandlesIndices.CLOSE)
        conversion_rate = close_price.get_conversion_rate()

        init_portfolio = self.realized_profits_loss.portfolio_init(
            batch_size, conversion_rate.get_conversion_rate_at_step(step=0)
        )

        leaky_transfers = (
            decisions.tensors.decisions
            * conversion_rate.tensors.conversion_rate
            * self.realized_profits_loss.fees.to(decisions.device)
        )
        init_portfolio = init_portfolio.double().unsqueeze(1)

        portfolios = []
        portfolio = init_portfolio.squeeze(1).unsqueeze(2)
        for step_idx in range(
            self.rnn_warmup_steps, self.rnn_warmup_steps + self.seq_length
        ):
            step_decisions = leaky_transfers[:, step_idx].unsqueeze(1)
            portfolio = torch.matmul(
                step_decisions.squeeze(1).transpose(1, 2), portfolio.squeeze(1)
            )
            portfolios.append(portfolio.reshape(1, batch_size, self.n_currencies))

        portfolios = torch.vstack(portfolios).transpose(0, 1)
        assert tuple(portfolios.shape) == (
            batch_size,
            self.seq_length,
            self.n_currencies,
        )
        return portfolios / portfolios.sum(dim=2).unsqueeze(2)

    def _forward(
        self, decisions: DecisionsTensor, targets: ModelTargets, truncate: bool = True
    ) -> torch.Tensor:
        oracle_portfolio = self.decisions_to_portfolio(
            decisions=targets.oracle_decisions, candles_tensor=targets.candles_tensor
        )
        portfolio = self.decisions_to_portfolio(
            decisions=decisions, candles_tensor=targets.candles_tensor
        )
        loss = torch.nn.MSELoss()(
            portfolio, oracle_portfolio.to(portfolio.device)
        )  # Maybe need to improve normalization to better compare
        return loss


class CombinedLoss(AbstractLoss):
    def __init__(self, training_config: TrainingConfig):
        super().__init__(training_config)
        self.realized_profits = RealizedProfitsLoss(training_config)
        self.oracle_loss = PortfolioOracleLoss(training_config)
        self.diagonal_loss = DiagonalLoss(training_config)

    def update_fee(self, new_fee: float) -> None:
        self.fee = new_fee
        self.realized_profits.update_fee(new_fee)
        self.oracle_loss.update_fee(new_fee)
        self.diagonal_loss.update_fee(new_fee)

    def _forward(
        self, decisions: DecisionsTensor, targets: ModelTargets
    ) -> torch.Tensor:
        loss = (
            0.001 * self.oracle_loss._forward(decisions, targets)
            + self.realized_profits._forward(decisions, targets)
            + 0.01 * self.diagonal_loss._forward(decisions, targets)
        )
        if loss.isnan().sum() > 0:
            raise ValueError(f"Found NaN values: {int(loss.isnan().sum())}")

        return loss
