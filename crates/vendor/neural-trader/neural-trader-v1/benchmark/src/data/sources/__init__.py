"""Data source adapters for market data feeds."""

from .yahoo_finance_adapter import YahooFinanceAdapter
from .alpha_vantage_adapter import AlphaVantageAdapter
from .fred_adapter import FREDAdapter
from .crypto_compare_adapter import CryptoCompareAdapter

__all__ = [
    "YahooFinanceAdapter",
    "AlphaVantageAdapter",
    "FREDAdapter",
    "CryptoCompareAdapter",
]