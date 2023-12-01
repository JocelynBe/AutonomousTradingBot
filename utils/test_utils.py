import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch

from constants import EXPECTED_DELTA_MS
from exchange_api.contracts import Candle, Currency
from features.contracts import CANDLE_COLUMNS
from models.contracts import CandlesIndices, ConversionRate, Step, SubCandleTensor

BASE_PRICE = 100.0
RESET_STEPS = 20
OFFSET = 0.01


@dataclass
class DecisionAtStep:
    usd_to_btc: float
    btc_to_usd: float


def generate_price_dirac_with_no_signal(seq_length: int) -> pd.DataFrame:
    timeseries = defaultdict(list)
    mode = "reset"
    counter = 0
    random.seed(42)
    last_event = 0
    for step in range(seq_length):
        timeseries["timestamp"].append(step * EXPECTED_DELTA_MS)
        price = BASE_PRICE
        if mode == "reset":
            if counter > RESET_STEPS:
                counter = 0
                mode = "random"
            else:
                counter += 1
            price += ((step % 3) - 1) * OFFSET
        elif mode == "random":
            x = random.random()
            if x > 0.9:
                counter = 0
                mode = "event"
            else:
                price += (random.random() - 0.5) * OFFSET
        elif mode == "event":
            price = 2 * BASE_PRICE
            counter = 0
            mode = "after_event"

        for column_id, column in enumerate(CANDLE_COLUMNS):
            timeseries[column].append(price + OFFSET * column_id)

        # When we have an event (high price), we need a corresponding low price
        # That's why we multiply a previous value by 0.9
        # which should be enough for the model to pick up the signal
        if mode == "after_event":
            mode = "reset"
            step_with_low_price = random.randint(last_event + 1, step - 2)
            last_event = step
            for column in CANDLE_COLUMNS:
                timeseries[column][step_with_low_price] *= 0.5

            for i in range(step_with_low_price, step - 1):
                for column in CANDLE_COLUMNS:
                    u = (i - step_with_low_price) / (step - step_with_low_price)
                    timeseries[column][i] = (1 - u) * timeseries[column][
                        step_with_low_price
                    ] + u * BASE_PRICE

    candles_df = pd.DataFrame(timeseries)
    return candles_df


def get_conversion_rate_from_price(
    price_ts_one_batch: List[float], batch_size: int
) -> ConversionRate:
    seq_len = len(price_ts_one_batch)
    price_ts = torch.tensor(
        [price_ts_one_batch for _ in range(batch_size)], dtype=torch.float64
    ).reshape(batch_size, seq_len, 1)
    timestamps = torch.tensor(
        [list(range(seq_len))] * batch_size, dtype=torch.float64
    ).reshape(batch_size, seq_len, 1)
    subcandle_tensor = SubCandleTensor(
        price_ts, timestamps, candle_type=CandlesIndices.CLOSE
    )

    conversion_rate = subcandle_tensor.get_conversion_rate()
    return conversion_rate


def super_dumb_execute_trades(
    price_ts: List[float],
    decisions: Dict[Step, DecisionAtStep],
    amount_of_usd: float,
    amount_of_btc: float,
    fee: float,
    verbose: bool = False,
) -> Tuple[float, float]:
    assert 0 <= fee <= 1
    for step, price in enumerate(price_ts):
        assert amount_of_usd >= 0 and amount_of_btc >= 0

        decision = decisions.get(step)
        if decision is None:
            continue

        if verbose:
            print("-" * 50)
            print(
                f"Applying decision {decision} at step {step} with price {price} and fee {fee}"
            )
            print(
                f"Current amounts (amount_of_usd, amount_of_btc) = {(amount_of_usd, amount_of_btc)}"
            )

        delta_usd, delta_btc = 0, 0
        if decision.usd_to_btc:
            delta_usd += -amount_of_usd * decision.usd_to_btc
            delta_btc += amount_of_usd * decision.usd_to_btc / price * (1 - fee)
            if verbose:
                print(
                    f"Converting usd to btc, (delta_usd, delta_btc) = {(delta_usd, delta_btc)}"
                )
        if decision.btc_to_usd:
            delta_btc += -amount_of_btc * decision.btc_to_usd
            delta_usd += amount_of_btc * decision.btc_to_usd / (1 / price) * (1 - fee)
            if verbose:
                print(
                    f"Converting btc to usd, (delta_usd, delta_btc) = {(delta_usd, delta_btc)}"
                )

        amount_of_usd += delta_usd
        amount_of_btc += delta_btc
        if verbose:
            print(
                f"New amounts (amount_of_usd, amount_of_btc) = {(amount_of_usd, amount_of_btc)}"
            )

    assert amount_of_usd >= 0 and amount_of_btc >= 0
    return float(amount_of_usd), float(amount_of_btc)


def get_ordered_candles(n_candles: int, offset: int = 10) -> List[Candle]:
    return [
        Candle(
            src_currency=Currency.USD,
            dst_currency=Currency.BTC,
            timestamp=i + offset,
            open=i + offset,
            close=i + offset,
            high=i + offset,
            low=i + offset,
            volume=i + offset,
        )
        for i in range(n_candles)
    ]
