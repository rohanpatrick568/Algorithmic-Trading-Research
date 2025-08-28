import numpy as np
import pandas as pd
import backtrader as bt


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


def make_synthetic_ohlcv(rows=250, start_price=100.0, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)

    # Random walk for close
    rets = rng.normal(loc=0.0005, scale=0.01, size=rows)
    close = start_price * (1 + pd.Series(rets, index=dates)).add(1).cumprod()

    # Construct simple OHLC around close with tiny noise
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
    df = make_synthetic_ohlcv()

    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)  # see: SmaCross in this file
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)

    start_val = cerebro.broker.getvalue()
    result = cerebro.run()
    end_val = cerebro.broker.getvalue()

    print(f"Backtrader OK. Start: {start_val:.2f}  End: {end_val:.2f}")


if __name__ == "__main__":
    main()