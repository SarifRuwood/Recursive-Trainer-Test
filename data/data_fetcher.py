"""
Data fetching and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional
import ccxt
from pathlib import Path
import os


class MarketDataFetcher:
    """
    Fetch historical OHLCV data from exchanges via CCXT.
    """
    
    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize data fetcher.
        
        Args:
            exchange_name: CCXT exchange name
        """
        self.exchange_name = exchange_name
        self.exchange = getattr(ccxt, exchange_name)()
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (1m, 5m, 1h, 1d, etc.)
            since: Fetch data since this timestamp (ms)
            limit: Max candles to fetch per call
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def fetch_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = '1h',
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a date range.
        
        Args:
            symbol: Trading pair
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Candle timeframe
        
        Returns:
            DataFrame with all OHLCV data
        """
        start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        all_data = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            df = self.fetch_ohlcv(symbol, timeframe, current_timestamp, 1000)
            if df.empty:
                break
            all_data.append(df)
            current_timestamp = int(df.iloc[-1]['timestamp'].timestamp() * 1000)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return result[result['timestamp'] <= end_date]


def load_or_fetch_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '1h',
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load data from cache or fetch from exchange.
    
    Args:
        symbol: Trading pair
        start_date: Start date
        end_date: End date
        timeframe: Candle timeframe
        cache_path: Path to cache CSV
        use_cache: Use cached data if available
    
    Returns:
        DataFrame with OHLCV data
    """
    # Generate cache filename if not provided
    if cache_path is None:
        cache_dir = 'data/historical'
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace('/', '_')
        cache_path = os.path.join(cache_dir, f'{safe_symbol}_{timeframe}.csv')
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    # Fetch from exchange
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_range(symbol, start_date, end_date, timeframe)
    
    # Save to cache
    if not df.empty and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Data cached to {cache_path}")
    
    return df


def generate_synthetic_data(
    n_candles: int = 1000,
    start_price: float = 50000.0,
    drift: float = 0.0001,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.
    
    Args:
        n_candles: Number of candles to generate
        start_price: Starting price
        drift: Daily drift (mean return)
        volatility: Daily volatility (std dev)
        seed: Random seed
    
    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(seed)
    
    timestamps = pd.date_range(start='2023-01-01', periods=n_candles, freq='1h')
    prices = [start_price]
    
    for _ in range(n_candles - 1):
        log_return = np.random.normal(drift, volatility)
        next_price = prices[-1] * np.exp(log_return)
        prices.append(next_price)
    
    prices = np.array(prices)
    
    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_candles))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_candles))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_candles),
    }
    
    df = pd.DataFrame(data)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    return df
