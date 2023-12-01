from typing import Any, Union

from ray.actor import ActorHandle
from tqdm import tqdm


class ProgressBar(tqdm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")


def update_progress_bar(progress_bar: Union[ProgressBar, ActorHandle]) -> None:
    if isinstance(progress_bar, ActorHandle):
        progress_bar.update.remote(1)
    else:
        progress_bar.update(1)


def assert_equal(first: Any, second: Any) -> None:
    b = first == second
    if first is None or second is None:
        b = first is None and second is None
    assert b, (first, second)
