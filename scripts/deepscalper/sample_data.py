"""Sample data generation for offline training and testing.

This module provides functionality to generate realistic market data
for training the DeepScalper agent when network access is limited
or for consistent testing environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple

try:
    from .data import add_indicators, MarketData
except Exception:
    from data import add_indicators, MarketData


def generate_synthetic_ohlcv(
    n_bars: int = 10000,
    initial_price: float = 150.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    mean_volume: float = 1000000,
    volume_std: float = 300000,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate synthetic OHLCV data using geometric Brownian motion with realistic intraday patterns.
    
    Args:
        n_bars: Number of 1-minute bars to generate
        initial_price: Starting price
        volatility: Daily volatility (will be scaled to 1-minute)
        trend: Daily trend (will be scaled to 1-minute)
        mean_volume: Average volume per bar
        volume_std: Standard deviation of volume
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with OHLC and volume data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Scale parameters for 1-minute intervals (390 minutes per trading day)
    minutes_per_day = 390
    dt = 1 / minutes_per_day
    vol_per_minute = volatility / np.sqrt(minutes_per_day)
    trend_per_minute = trend * dt
    
    # Generate price path using geometric Brownian motion
    returns = np.random.normal(trend_per_minute, vol_per_minute, n_bars)
    
    # Add intraday volatility patterns (higher volatility at market open/close)
    intraday_pattern = np.sin(np.arange(n_bars) * 2 * np.pi / minutes_per_day) * 0.3 + 1.0
    returns *= intraday_pattern
    
    # Calculate prices
    log_prices = np.log(initial_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)
    
    # Generate OHLC from close prices with realistic spreads
    ohlc_data = []
    for i, close in enumerate(close_prices):
        if i == 0:
            open_price = initial_price
        else:
            open_price = close_prices[i-1]
        
        # Random high/low around open-close range
        bar_range = abs(close - open_price) * 0.5 + close * vol_per_minute * np.random.uniform(0.5, 2.0)
        high = max(open_price, close) + bar_range * np.random.uniform(0, 1)
        low = min(open_price, close) - bar_range * np.random.uniform(0, 1)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        ohlc_data.append([open_price, high, low, close])
    
    # Generate volume with some correlation to price volatility
    volume_multiplier = 1.0 + np.abs(returns) * 5  # Higher volume on big moves
    volumes = np.maximum(
        np.random.normal(mean_volume, volume_std, n_bars) * volume_multiplier,
        1000  # Minimum volume
    ).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
    df['volume'] = volumes
    
    # Add realistic timestamps (trading days only, 9:30-16:00 ET)
    base_date = datetime(2023, 1, 3, 9, 30)  # Start on a Tuesday
    timestamps = []
    current_time = base_date
    
    for i in range(n_bars):
        timestamps.append(current_time)
        current_time += timedelta(minutes=1)
        
        # Skip weekends and after-hours
        if current_time.weekday() >= 5:  # Weekend
            current_time += timedelta(days=2)
            current_time = current_time.replace(hour=9, minute=30)
        elif current_time.hour >= 16:  # After market close
            current_time += timedelta(days=1)
            current_time = current_time.replace(hour=9, minute=30)
            # Skip weekends
            if current_time.weekday() >= 5:
                current_time += timedelta(days=2)
                current_time = current_time.replace(hour=9, minute=30)
    
    df.index = pd.DatetimeIndex(timestamps)
    return df


def create_sample_dataset(
    symbol: str = "AAPL",
    days: int = 30,
    save_path: Optional[str] = None
) -> MarketData:
    """Create a sample dataset with technical indicators for training/testing.
    
    Args:
        symbol: Symbol name (for reference)
        days: Number of trading days to generate
        save_path: Optional path to save the data as CSV
        
    Returns:
        MarketData object ready for use with DeepScalperEnv
    """
    # Generate approximately the right number of bars (390 minutes per trading day)
    n_bars = days * 390
    
    # Use different parameters based on symbol for variety
    params = {
        "AAPL": {"initial_price": 150.0, "volatility": 0.025, "trend": 0.0002},
        "MSFT": {"initial_price": 300.0, "volatility": 0.022, "trend": 0.0001},
        "GOOGL": {"initial_price": 2500.0, "volatility": 0.028, "trend": 0.0003},
        "TSLA": {"initial_price": 800.0, "volatility": 0.045, "trend": -0.0001},
        "SPY": {"initial_price": 400.0, "volatility": 0.015, "trend": 0.0001},
    }
    
    symbol_params = params.get(symbol, params["AAPL"])
    
    # Generate base OHLCV data
    df = generate_synthetic_ohlcv(
        n_bars=n_bars,
        initial_price=symbol_params["initial_price"],
        volatility=symbol_params["volatility"],
        trend=symbol_params["trend"],
        seed=hash(symbol) % 2**32  # Consistent seed based on symbol
    )
    
    # Add technical indicators
    df = add_indicators(df)
    
    # Save if requested
    if save_path:
        df.to_csv(save_path)
        print(f"Sample data saved to {save_path}")
    
    return MarketData(df=df)


def load_sample_data(symbol: str = "AAPL", days: int = 30) -> MarketData:
    """Load or generate sample data for the given symbol.
    
    This function provides a consistent interface that can be used as a drop-in
    replacement for yfinance data loading in offline environments.
    """
    print(f"[SampleData] Generating {days} days of synthetic data for {symbol}")
    return create_sample_dataset(symbol=symbol, days=days)


if __name__ == "__main__":
    # Generate sample data for common symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]
    
    for symbol in symbols:
        data = create_sample_dataset(symbol, days=30)
        print(f"Generated {len(data.df)} bars for {symbol}")
        print(f"Price range: ${data.df['close'].min():.2f} - ${data.df['close'].max():.2f}")
        print(f"Volume range: {data.df['volume'].min():,} - {data.df['volume'].max():,}")
        print()