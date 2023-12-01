from typing import Dict, List

import pandas as pd

from constants import (
    MAX_STEPS_WITH_NO_DATAPOINT,
    MINUTE_IN_MS,
    N_STEPS_CHECK_MISSING,
    N_WARM_UP_CANDLES,
)
from exchange_api.contracts import Candle


def build_should_stop_df(
    candles: List[Candle], interval_in_minutes: int
) -> pd.DataFrame:
    assert sorted(candles, key=lambda c: c.timestamp) == candles
    start_time, end_time = candles[0].timestamp, candles[-1].timestamp
    n_steps = int((end_time - start_time) / MINUTE_IN_MS) + 1
    all_times = [start_time + i * 6e4 * interval_in_minutes for i in range(0, n_steps)]
    missing_values = set(all_times) - {candle.timestamp for candle in candles}

    assert start_time not in missing_values
    assert end_time not in missing_values

    is_time_missing = []
    for time in all_times:
        is_missing = time in missing_values
        is_time_missing.append((time, is_missing))

    df = pd.DataFrame(is_time_missing, columns=["time", "is_missing"])
    df["should_stop"] = (
        df.is_missing.rolling(N_STEPS_CHECK_MISSING).sum() > MAX_STEPS_WITH_NO_DATAPOINT
    )
    return df


def make_uninterrupted_slices(
    df: pd.DataFrame, start_time: float, interval_in_minutes: int
) -> List[pd.DataFrame]:
    max_allowed_delta_in_minutes = (
        N_STEPS_CHECK_MISSING * MINUTE_IN_MS * interval_in_minutes
    )
    slices = []
    last_non_missing_time = start_time

    rows = []
    for row in df.itertuples():
        delta = getattr(row, "time") - last_non_missing_time
        if not getattr(row, "is_missing"):
            last_non_missing_time = getattr(row, "time")
        if delta > max_allowed_delta_in_minutes:
            slices.append(rows)
            rows = []
        rows.append(row)
    if len(rows) > 2 * N_WARM_UP_CANDLES:
        slices.append(rows)
    assert len(df) == sum(map(len, slices)), (len(df), list(map(len, slices)))
    slices = list(map(pd.DataFrame, slices))

    try:
        for slice_df in slices:
            assert (
                slice_df.is_missing.rolling(N_STEPS_CHECK_MISSING).sum().max()
                <= MAX_STEPS_WITH_NO_DATAPOINT
            )
    except:
        import ipdb

        ipdb.set_trace()  # XXX BREAKPOINT

    return slices


def get_timestep_to_slice(slices: List[pd.DataFrame]) -> Dict[float, int]:
    timestep_to_slice: Dict[float, int] = dict()
    for slice_idx, slice_df in enumerate(slices):
        slice_df = slice_df.loc[~slice_df.is_missing].reset_index(drop=True)
        slice_df["slice_idx"] = slice_idx
        timestep_to_slice.update(
            slice_df[["time", "slice_idx"]].set_index("time").to_dict()["slice_idx"]
        )
    return timestep_to_slice


def map_candles_to_slice(
    candles: List[Candle], timestep_to_slice: Dict[float, int], n_slices: int
) -> List[List[Candle]]:
    candles_slices = {slice_idx: [] for slice_idx in range(n_slices)}
    for candle in candles:
        if candle.timestamp not in timestep_to_slice:
            continue
        slice_idx = timestep_to_slice[candle.timestamp]
        candles_slices[slice_idx].append(candle)

    return [candles_slices[slice_idx] for slice_idx in sorted(candles_slices.keys())]


def handle_missing_timesteps(
    candles: List[Candle], interval_in_minutes: int
) -> List[List[Candle]]:
    should_stop_df = build_should_stop_df(candles, interval_in_minutes)
    n_missing = should_stop_df.is_missing.sum()
    print(
        f"Number of missing time steps = {n_missing}, % = {n_missing / len(should_stop_df)}"
    )

    cleaned = should_stop_df.loc[~should_stop_df.should_stop]
    start_time = cleaned["time"].min()
    slices = make_uninterrupted_slices(cleaned, start_time, interval_in_minutes)
    print(f"Number of slices = {len(slices)} with lengths {list(map(len, slices))}")

    timestep_to_slice = get_timestep_to_slice(slices)
    assert len(timestep_to_slice) <= (~should_stop_df.is_missing).sum()

    candles_slices = map_candles_to_slice(candles, timestep_to_slice, len(slices))
    return candles_slices
