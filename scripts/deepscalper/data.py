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


def load_minute_data(symbol: str, start: str, end: str) -> MarketData:
    """Load minute data, with fallback to synthetic data when network access fails."""
    try:
        import yfinance as yf  # type: ignore
        
        print(f"[DataLoader] Attempting to download {symbol} data from {start} to {end}")
        df = yf.download(symbol, start=start, end=end, interval="1m", progress=False)
        
        if df.empty:
            print(f"[DataLoader] No data returned from yfinance for {symbol}")
            raise RuntimeError("No data returned from yfinance")

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]
        df = add_indicators(df)
        print(f"[DataLoader] Successfully loaded {len(df)} bars from yfinance")
        return MarketData(df=df)
        
    except Exception as e:
        print(f"[DataLoader] Failed to load from yfinance: {e}")
        print(f"[DataLoader] Falling back to synthetic data for {symbol}")
        
        # Fallback to synthetic data
        try:
            from datetime import datetime
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d")
            days = (end_date - start_date).days
            
            # Import sample data generator
            try:
                from .sample_data import load_sample_data
            except Exception:
                from sample_data import load_sample_data
            
            return load_sample_data(symbol=symbol, days=max(days, 5))
            
        except Exception as fallback_error:
            raise RuntimeError(f"Both yfinance and synthetic data failed: {e}, {fallback_error}") from e
