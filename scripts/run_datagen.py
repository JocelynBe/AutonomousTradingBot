import logging
import multiprocessing as mp

import click

from features.main import run_datagen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("actions.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--path_to_csv", help="Path to candles CSV")
@click.option(
    "--start_time", default=1672549200000, help="Minimum timestamp.", type=int
)  # 2023-01-01
@click.option("--output_dir", help="Path to output dir")
@click.option("--interval_in_minutes", help="Interval in minutes", type=int)
def main(
    path_to_csv: str, start_time: float, output_dir: str, interval_in_minutes: int
) -> None:
    num_cpus = mp.cpu_count() - 1
    run_datagen(path_to_csv, start_time, output_dir, interval_in_minutes, num_cpus)


if __name__ == "__main__":
    main()
