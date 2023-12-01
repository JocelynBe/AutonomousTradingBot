from collections import defaultdict

from exchange_api.contracts import Currency, Portfolio


def init_portfolio(amount: float = 1000.0) -> Portfolio:
    holdings = defaultdict(float)
    holdings[Currency.USD] = amount
    return Portfolio(holdings)
