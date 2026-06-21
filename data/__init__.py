"""
Data package for market data utilities.
"""

from .data_fetcher import (
    MarketDataFetcher,
    load_or_fetch_data,
    generate_synthetic_data,
)

__all__ = [
    'MarketDataFetcher',
    'load_or_fetch_data',
    'generate_synthetic_data',
]
