import numpy as np
import pandas as pd
import backtrader as bt
import yfinance as yf


def _ensure_bt_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and standardize to OHLCV(+OpenInterest)."""
    out = df.copy()

    # Flatten MultiIndex columns by taking the first level if needed
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    else:
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    # Drop Adj Close if present
    if "Adj Close" in out.columns:
        out = out.drop(columns=["Adj Close"])

    # Case-insensitive rename to expected names
    rename = {}
    for c in out.columns:
        lc = str(c).lower().replace(" ", "")
        if lc == "open":
            rename[c] = "Open"
        elif lc == "high":
            rename[c] = "High"
        elif lc == "low":
            rename[c] = "Low"
        elif lc == "close":
            rename[c] = "Close"
        elif lc == "volume":
            rename[c] = "Volume"
        elif lc in ("openinterest", "oi"):
            rename[c] = "OpenInterest"
    out = out.rename(columns=rename)

    # Ensure OpenInterest column exists
    if "OpenInterest" not in out.columns:
        out["OpenInterest"] = 0

    # Return only required columns in the right order
    return out[["Open", "High", "Low", "Close", "Volume", "OpenInterest"]]


class SmaCross(bt.Strategy):
    params = dict(fast=10, slow=30)

    def __init__(self):
        self.sma_fast = bt.ind.SMA(self.data.close, period=self.p.fast)
        self.sma_slow = bt.ind.SMA(self.data.close, period=self.p.slow)
        self.crossover = bt.ind.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()


def fetch_yfinance_ohlcv(symbol: str = "AAPL", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV from yfinance and return a Backtrader-ready DataFrame.

    Columns: Open, High, Low, Close, Volume, OpenInterest
    Index: DatetimeIndex (tz-naive)
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return df
    # Ensure required columns and add OpenInterest
    if "OpenInterest" not in df.columns:
        df["OpenInterest"] = 0
    # Drop Adj Close if present (Backtrader doesn't need it)
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])  # keep standard OHLCV
    # yfinance may return tz-aware index; make it tz-naive for Backtrader
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def make_synthetic_ohlcv(rows=250, start_price=100.0, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)

    rets = rng.normal(loc=0.0005, scale=0.01, size=rows)
    close = start_price * (1 + pd.Series(rets, index=dates)).add(1).cumprod()

    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.001, rows))
    high = pd.concat([open_, close], axis=1).max(axis=1) * (1 + np.abs(rng.normal(0, 0.002, rows)))
    low = pd.concat([open_, close], axis=1).min(axis=1) * (1 - np.abs(rng.normal(0, 0.002, rows)))
    volume = rng.integers(1000, 5000, rows)
    ohlcv = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume, "OpenInterest": 0},
        index=dates,
    )
    return ohlcv


def main():
    symbol = "AAPL"
    start = "2020-01-01"
    end = None  # use today's date by default

    df = fetch_yfinance_ohlcv(symbol=symbol, start=start, end=end)
    if df.empty:
        print(f"No data returned for {symbol}. Check ticker or network.")
        return

    df = _ensure_bt_ohlcv(df)  # <-- normalize columns to avoid tuple/MI issues

    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)

    start_val = cerebro.broker.getvalue()
    cerebro.run()
    end_val = cerebro.broker.getvalue()

    print(f"Backtrader OK ({symbol}). Start: {start_val:.2f}  End: {end_val:.2f}")


if __name__ == "__main__":
    main()