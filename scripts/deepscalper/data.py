from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    df: pd.DataFrame  # columns: open, high, low, close, volume, indicators...
    symbol: str = "UNKNOWN"
    
    def __post_init__(self):
        """Validate data quality"""
        self.validate_data()
    
    def validate_data(self):
        """Perform basic data quality checks"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (self.df[col] <= 0).any():
                print(f"Warning: Found zero/negative prices in {col} for {self.symbol}")
        
        # Check for zero volume (warn but don't fail)
        if (self.df['volume'] == 0).any():
            zero_vol_count = (self.df['volume'] == 0).sum()
            print(f"Warning: Found {zero_vol_count} zero volume bars for {self.symbol}")
        
        # Check for gaps in data (missing timestamps) - simplified validation
        if len(self.df) > 1:
            try:
                time_diffs = self.df.index.to_series().diff().dropna()
                expected_diff = pd.Timedelta(minutes=1)
                # Simple count-based gap detection
                large_gaps = sum(1 for diff in time_diffs if diff > expected_diff * 2)
                if large_gaps > 0:
                    print(f"Warning: Found {large_gaps} time gaps in data for {self.symbol}")
            except Exception:
                # Skip gap detection if there are index issues
                pass


def get_cache_path(symbol: str, start: str, end: str) -> Path:
    """Get cache file path for market data"""
    cache_dir = Path.cwd() / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = f"{symbol}_{start}_{end}.parquet"
    return cache_dir / cache_file


def save_to_cache(df: pd.DataFrame, symbol: str, start: str, end: str):
    """Save dataframe to Parquet cache"""
    try:
        cache_path = get_cache_path(symbol, start, end)
        df.to_parquet(cache_path)
        print(f"[DataCache] Saved {symbol} data to {cache_path}")
    except Exception as e:
        print(f"[DataCache] Failed to save cache: {e}")


def load_from_cache(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Load dataframe from Parquet cache if available"""
    try:
        cache_path = get_cache_path(symbol, start, end)
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            print(f"[DataCache] Loaded {symbol} data from cache ({len(df)} rows)")
            return df
    except Exception as e:
        print(f"[DataCache] Failed to load cache: {e}")
    return None


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


def generate_synthetic_data(symbol: str, start: str, end: str, initial_price: float = 150.0, regime: str = "normal") -> MarketData:
    """Generate synthetic minute-level OHLCV data for testing purposes.
    
    Args:
        symbol: Stock symbol
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)  
        initial_price: Starting price for simulation
        regime: Market regime - "normal", "trending", "mean_reverting", "high_vol"
    """
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
    
    # Generate synthetic price data based on regime
    np.random.seed(42)  # For reproducibility
    n_points = len(timestamps)
    
    if regime == "trending":
        # Strong upward trend with lower volatility
        returns = np.random.normal(0.0005, 0.001, n_points)  # Higher drift, lower vol
    elif regime == "mean_reverting":
        # Mean-reverting with periodic cycles
        base_price = initial_price
        returns = []
        price = initial_price
        for i in range(n_points):
            # Pull back to mean with some noise
            mean_revert = -0.001 * (price - base_price) / base_price
            noise = np.random.normal(0, 0.002)
            ret = mean_revert + noise
            returns.append(ret)
            price *= (1 + ret)
        returns = np.array(returns)
    elif regime == "high_vol":
        # High volatility clustering
        returns = np.random.normal(0.0001, 0.004, n_points)  # Much higher vol
        # Add volatility clustering
        vol_regime = np.random.binomial(1, 0.1, n_points)  # 10% chance of high vol periods
        returns = returns * (1 + vol_regime * 2)  # 3x vol during high vol periods
    else:  # normal
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
    return MarketData(df=df, symbol=symbol)


def load_minute_data(symbol: str, start: str, end: str, use_synthetic: bool = False, use_cache: bool = True) -> MarketData:
    """Load minute data from yfinance, cache, or generate synthetic data.
    
    Args:
        symbol: Stock symbol to load
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        use_synthetic: Force synthetic data generation
        use_cache: Whether to use/save Parquet cache
    """
    if use_synthetic:
        print(f"[DataLoader] Generating synthetic data for {symbol} from {start} to {end}")
        return generate_synthetic_data(symbol, start, end)
    
    # Try cache first
    if use_cache:
        cached_df = load_from_cache(symbol, start, end)
        if cached_df is not None:
            return MarketData(df=cached_df, symbol=symbol)
    
    # Try to load from yfinance
    try:
        import yfinance as yf  # type: ignore
        print(f"[DataLoader] Downloading {symbol} data from yfinance...")
        
        df = yf.download(symbol, start=start, end=end, interval="1m", progress=False)
        if df.empty:
            print(f"[DataLoader] No data from yfinance for {symbol}, falling back to synthetic data")
            return generate_synthetic_data(symbol, start, end)
            
        # Clean up column names
        df = df.rename(columns={
            "Open": "open",
            "High": "high", 
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]
        
        # Add technical indicators
        df = add_indicators(df)
        
        # Save to cache
        if use_cache:
            save_to_cache(df, symbol, start, end)
            
        return MarketData(df=df, symbol=symbol)
        
    except Exception as e:
        print(f"[DataLoader] yfinance failed for {symbol} ({e}), using synthetic data")
        return generate_synthetic_data(symbol, start, end)


def load_multi_symbol_data(symbols: List[str], start: str, end: str, use_synthetic: bool = False, use_cache: bool = True) -> List[MarketData]:
    """Load data for multiple symbols"""
    results = []
    for symbol in symbols:
        try:
            md = load_minute_data(symbol, start, end, use_synthetic=use_synthetic, use_cache=use_cache)
            results.append(md)
        except Exception as e:
            print(f"[DataLoader] Failed to load {symbol}: {e}")
            # Generate synthetic fallback
            md = generate_synthetic_data(symbol, start, end)
            results.append(md)
    
    print(f"[DataLoader] Loaded {len(results)} symbols: {[md.symbol for md in results]}")
    return results
