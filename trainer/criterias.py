from models.metrics.contracts import MetricsDict


def daily_return_mu(metrics_dict: MetricsDict) -> float:
    return metrics_dict["return_day_mu"]


def return_avg_gain_amounts(metrics_dict: MetricsDict) -> float:
    return metrics_dict["return_avg_gain_amounts"]
