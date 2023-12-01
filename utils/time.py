import datetime

import numpy as np

from constants import DATETIME_FORMAT


def timestamp_to_str(timestamp: int) -> str:
    if int(np.log(timestamp) / np.log(10)) == 12:
        timestamp = timestamp / 1000

    assert int(np.log(timestamp) / np.log(10)) == 9
    datetime_value = datetime.datetime.fromtimestamp(timestamp)
    return datetime_to_str(datetime_value)


def datetime_to_str(datetime_value: datetime.datetime) -> str:
    return datetime_value.strftime(DATETIME_FORMAT)


def str_to_datetime(datetime_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(datetime_str, DATETIME_FORMAT)
