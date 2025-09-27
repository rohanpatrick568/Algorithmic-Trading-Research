from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    df: pd.DataFrame  # columns: open, high, low, close, volume, indicators...


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    mfr = pos_sum / (neg_sum + 1e-12)
    return 100 - (100 / (1 + mfr))


def bollinger(df: pd.DataFrame, period: int = 20, ndev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    up = mid + ndev * std
    dn = mid - ndev * std
    return mid, up, dn


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi14"] = rsi(out["close"], 14)
    out["mfi14"] = mfi(out, 14)
    mid, up, dn = bollinger(out, 20, 2.0)
    out["bb_mid20"] = mid
    out["bb_up20"] = up
    out["bb_dn20"] = dn
    out["atr14"] = atr(out, 14)
    return out.dropna().reset_index(drop=True)


def generate_synthetic_data(symbol: str, start: str, end: str, initial_price: float = 150.0) -> MarketData:
    """Generate synthetic minute-level OHLCV data for testing purposes."""
    from datetime import datetime, timedelta
    
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    
    # Create minute-level timestamps during market hours (9:30 AM - 4:00 PM EST)
    timestamps = []
    current_dt = start_dt
    while current_dt < end_dt:
        # Add market hours (9:30 AM to 4:00 PM = 390 minutes)
        market_open = current_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        for minute in range(390):  # 6.5 hours * 60 minutes
            timestamps.append(market_open + timedelta(minutes=minute))
        current_dt += timedelta(days=1)
    
    # Generate synthetic price data with random walks
    np.random.seed(42)  # For reproducibility
    n_points = len(timestamps)
    
    # Generate price movements using random walk with drift
    returns = np.random.normal(0.0001, 0.002, n_points)  # Small drift, realistic volatility
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV bars
    data = []
    for i, timestamp in enumerate(timestamps):
        price = prices[i]
        volatility = abs(np.random.normal(0, 0.01))  # Random volatility for H-L spread
        
        # Simulate intraday price action
        high = price * (1 + volatility/2)
        low = price * (1 - volatility/2)
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = max(1000, int(np.random.exponential(50000)))  # Realistic volume distribution
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    df = add_indicators(df)
    return MarketData(df=df)


def load_minute_data(symbol: str, start: str, end: str, use_synthetic: bool = False) -> MarketData:
    """Load minute data from yfinance or generate synthetic data for testing."""
    if use_synthetic:
        print(f"[DataLoader] Generating synthetic data for {symbol} from {start} to {end}")
        return generate_synthetic_data(symbol, start, end)
    
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise RuntimeError("yfinance is required to load data") from e

    try:
        df = yf.download(symbol, start=start, end=end, interval="1m", progress=False)
        if df.empty:
            print(f"[DataLoader] No data from yfinance, falling back to synthetic data")
            return generate_synthetic_data(symbol, start, end)
    except Exception as e:
        print(f"[DataLoader] yfinance failed ({e}), using synthetic data")
        return generate_synthetic_data(symbol, start, end)

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })[["open", "high", "low", "close", "volume"]]
    df = add_indicators(df)
    return MarketData(df=df)
