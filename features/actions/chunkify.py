from typing import List

from constants import N_WARM_UP_CANDLES
from exchange_api.contracts import Candle
from features.contracts import CandleChunk


def chunkify_slice(candles: List[Candle], chunk_size: int) -> List[List[Candle]]:
    n_candles = len(candles)
    if chunk_size > n_candles:
        return [candles]

    n_chunks = n_candles // chunk_size
    slices_start_end = [
        (
            i * chunk_size,
            (i + 1) * chunk_size + N_WARM_UP_CANDLES if i < n_chunks - 1 else n_candles,
        )
        for i in range(n_chunks)
    ]
    chunks = [candles[start:end] for start, end in slices_start_end]
    return chunks


def get_chunks(slices: List[List[Candle]], num_cpus: int) -> List[CandleChunk]:
    total_len_slices = sum(map(len, slices))
    chunk_size = total_len_slices // (2 * num_cpus)
    assert chunk_size > 2 * N_WARM_UP_CANDLES, (chunk_size, N_WARM_UP_CANDLES)

    chunks = []
    for slice_idx, _slice in enumerate(slices):
        tmp = chunkify_slice(_slice, chunk_size)
        for chunk_idx, chunk in enumerate(tmp):
            if len(chunk) < 2 * N_WARM_UP_CANDLES:
                continue
            chunks.append(CandleChunk(slice_idx, chunk_idx, chunk))

    return chunks
