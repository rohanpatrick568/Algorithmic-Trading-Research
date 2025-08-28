import argparse
from dataclasses import dataclass
import datetime as dt
import math

import backtrader as bt
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


# ---------- Data utils (Backtrader expects OHLCV[+OpenInterest]) ----------
def _ensure_bt_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Flatten MultiIndex columns if needed
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    else:
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    # Drop Adj Close
    if "Adj Close" in out.columns:
        out = out.drop(columns=["Adj Close"])

    # Ensure OpenInterest
    if "OpenInterest" not in out.columns:
        out["OpenInterest"] = 0

    # tz-naive index
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    return out[["Open", "High", "Low", "Close", "Volume", "OpenInterest"]]


def fetch_yfinance_ohlcv(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return _ensure_bt_ohlcv(df)


# ---------- Risk-aware indicators (inspired by DeepScalper's volatility auxiliary task) ----------
class RealizedVolatility(bt.Indicator):
    """Rolling realized volatility of close-to-close returns.
    If annualize=True, multiplies by sqrt(252), else daily vol.
    """
    lines = ('vol',)
    params = dict(period=30, annualize=False)

    def __init__(self):
        ret = (self.data.close / self.data.close(-1)) - 1.0
        std = bt.ind.StdDev(ret, period=self.p.period)
        self.l.vol = std * (252.0 ** 0.5 if self.p.annualize else 1.0)


class DownsideDeviation(bt.Indicator):
    """Sortino-style downside deviation around target return (default 0)."""
    lines = ('dd',)
    params = dict(period=30, target=0.0)

    def __init__(self):
        ret = (self.data.close / self.data.close(-1)) - 1.0
        downside = bt.If(ret < self.p.target, (ret - self.p.target) * (ret - self.p.target), 0.0)
        mean_down = bt.ind.SMA(downside, period=self.p.period)
        self.l.dd = mean_down ** 0.5


# ---------- Custom Indicators ----------
class MoneyFlowIndex(bt.Indicator):
    """Money Flow Index (MFI) without TA-Lib.

    Typical Price (TP) = (H + L + C) / 3
    Money Flow (MF) = TP * Volume
    Positive MF = MF when TP > TP[-1], else 0
    Negative MF = MF when TP < TP[-1], else 0
    MFR = Sum(PosMF, n) / Sum(NegMF, n)
    MFI = 100 - 100 / (1 + MFR)
    """

    lines = ('mfi',)
    params = dict(period=14)

    def __init__(self):
        tp = (self.data.high + self.data.low + self.data.close) / 3.0
        mf = tp * self.data.volume
        upmf = bt.If(tp > tp(-1), mf, 0.0)
        dnmf = bt.If(tp < tp(-1), mf, 0.0)
        pos = bt.ind.Sum(upmf, self.p.period)
        neg = bt.ind.Sum(dnmf, self.p.period)

        # Avoid division by zero with a tiny epsilon
        ratio = pos / (neg + 1e-9)
        mfi = 100.0 - (100.0 / (1.0 + ratio))
        self.lines.mfi = mfi


# ---------- Flawless Victory Strategy (port of TradingView Pine) ----------
class FlawlessVictory(bt.Strategy):
    """
    Port of 'Flawless Victory Strategy' (Pine v4) with DeepScalper-inspired risk controls:
      - Volatility-aware position sizing
      - Drawdown brake
      - Downside-deviation (Sortino) gating

    See: TXTs/DeepScalper.txt for concepts.
    """

    params = dict(
        # Mode: 'v1' | 'v2' | 'v3'
        version="v1",

        # Sizing: base fraction of available cash to use on entry (0..1)
        cash_fraction=1.0,

        # v2 SL/TP (%)
        v2_stoploss_pct=6.604,
        v2_takeprofit_pct=2.328,

        # v3 SL/TP (%)
        v3_stoploss_pct=8.882,
        v3_takeprofit_pct=2.317,

        # RSI/MFI config (Pine defaults)
        rsi_period=14,
        mfi_period=14,

        # v1 guards
        v1_rsi_lower=42,
        v1_rsi_upper=70,

        # v2 guards
        v2_rsi_lower=42,
        v2_rsi_upper=76,

        # v3 guards
        v3_mfi_lower=60,
        v3_rsi_upper=65,
        v3_mfi_upper=64,

        # Bollinger configs
        v1_bb_period=20,
        v1_bb_dev=1.0,
        v2_bb_period=17,
        v2_bb_dev=1.0,

        # ---------- Risk-aware controls (DeepScalper context) ----------
        risk_enabled=True,
        # Target daily vol to budget position size against (DeepScalper uses vol auxiliary task)
        risk_target_daily_vol=0.015,     # 1.5% daily
        risk_vol_period=30,              # window for realized vol
        risk_min_scale=0.25,             # cap position scaling [min, max]
        risk_max_scale=1.25,
        # Drawdown brake: pause new entries when equity drawdown exceeds threshold
        risk_max_drawdown=0.20,          # 20%
        risk_dd_cooldown_bars=5,         # bars to pause after breach
        # Downside deviation cap (Sortino-style)
        risk_sortino_period=30,
        risk_max_downside_dev=0.02,      # 2% daily downside deviation
        risk_verbose=False,
    )

    def __init__(self):
        # Indicators
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        self.mfi = MoneyFlowIndex(self.data, period=self.p.mfi_period)

        self.bb1 = bt.ind.BollingerBands(self.data.close, period=self.p.v1_bb_period, devfactor=self.p.v1_bb_dev)
        self.bb2 = bt.ind.BollingerBands(self.data.close, period=self.p.v2_bb_period, devfactor=self.p.v2_bb_dev)

        # Risk metrics
        self.vol = RealizedVolatility(self.data, period=self.p.risk_vol_period, annualize=False)
        self.ddown = DownsideDeviation(self.data, period=self.p.risk_sortino_period, target=0.0)
        self.equity_peak = None
        self.dd_cooldown = 0

        # Track orders (no pyramiding)
        self.parent_order = None
        self.stop_order = None
        self.limit_order = None

    # Utilities
    def log(self, txt):
        dtstr = self.datas[0].datetime.datetime(0).strftime("%Y-%m-%d")
        print(f"{dtstr} {txt}")

    def has_open_children(self):
        kids = [self.stop_order, self.limit_order]
        return any(o and o.status in [bt.Order.Submitted, bt.Order.Accepted] for o in kids)

    def clear_orders(self):
        self.parent_order = None
        self.stop_order = None
        self.limit_order = None

    def cancel_children(self):
        for o in [self.stop_order, self.limit_order]:
            if o and o.status in [bt.Order.Submitted, bt.Order.Accepted]:
                try:
                    self.cancel(o)
                except Exception:
                    pass
        self.stop_order = None
        self.limit_order = None

    def notify_order(self, order):
        if order.status in [bt.Order.Completed, bt.Order.Canceled, bt.Order.Rejected, bt.Order.Margin]:
            if order is self.parent_order and order.status != bt.Order.Completed:
                self.cancel_children()
                self.parent_order = None
            if order is self.stop_order or order is self.limit_order:
                self.clear_orders()

    # ----- Risk core: compute scale and admissibility -----
    def _risk_scale_and_gate(self):
        if not self.p.risk_enabled:
            return 1.0, True, {}

        # Equity peak / drawdown monitor
        eq = self.broker.getvalue()
        if self.equity_peak is None:
            self.equity_peak = eq
        self.equity_peak = max(self.equity_peak, eq)
        dd = (eq / self.equity_peak) - 1.0  # negative when underwater

        # Cooldown if max drawdown breached
        gated = False
        reasons = {}
        if dd < -self.p.risk_max_drawdown:
            self.dd_cooldown = self.p.risk_dd_cooldown_bars
            gated = True
            reasons["drawdown"] = round(dd, 4)

        if self.dd_cooldown > 0:
            self.dd_cooldown -= 1
            gated = True
            reasons["cooldown"] = self.dd_cooldown

        # Volatility-aware scaling (DeepScalper: risk-aware auxiliary task via volatility)
        v = float(self.vol.vol[0])
        scale = 1.0
        if math.isnan(v) or v <= 0:
            scale = 1.0
        else:
            scale = self.p.risk_target_daily_vol / max(v, 1e-9)
            scale = max(self.p.risk_min_scale, min(self.p.risk_max_scale, scale))

        # Downside deviation gate (Sortino-like)
        ddv = float(self.ddown.dd[0])
        if not math.isnan(ddv) and ddv > self.p.risk_max_downside_dev:
            gated = True
            reasons["downside_dev"] = round(ddv, 4)

        return scale, (not gated), dict(vol=round(v if not math.isnan(v) else float('nan'), 6), dd=round(dd, 4), **reasons)

    def next(self):
        c = self.data.close[0]

        in_pos = self.position.size != 0
        have_open = any(
            o and o.status in [bt.Order.Submitted, bt.Order.Accepted]
            for o in [self.parent_order, self.stop_order, self.limit_order]
        )

        # Compute triggers/guards for each version
        v1_buy = (c < self.bb1.bot[0]) and (self.rsi[0] > self.p.v1_rsi_lower)
        v1_sell = (c > self.bb1.top[0]) and (self.rsi[0] > self.p.v1_rsi_upper)

        v2_buy = (c < self.bb2.bot[0]) and (self.rsi[0] > self.p.v2_rsi_lower)
        v2_sell = (c > self.bb2.top[0]) and (self.rsi[0] > self.p.v2_rsi_upper)

        v3_buy = (c < self.bb1.bot[0]) and (self.mfi[0] < self.p.v3_mfi_lower)
        v3_sell = (c > self.bb1.top[0]) and (self.rsi[0] > self.p.v3_rsi_upper) and (self.mfi[0] > self.p.v3_mfi_upper)

        # Risk scaling & gates
        scale, risk_ok, rmeta = self._risk_scale_and_gate()
        if self.p.risk_verbose:
            self.log(f"RISK vol={rmeta.get('vol')} dd={rmeta.get('dd')} scale={scale:.2f} gates={{{k: rmeta[k] for k in rmeta if k not in ('vol','dd')}}}")

        # No pyramiding: only if flat and no pending orders
        if not in_pos and not have_open and risk_ok:
            stake = max(1, int((self.broker.getcash() * self.p.cash_fraction * scale) / c))

            if self.p.version == "v1" and v1_buy:
                self.parent_order = self.buy(size=stake)
                self.log(f"BUY v1 size={stake} @ {c:.2f}")

            elif self.p.version == "v2" and v2_buy:
                sl = c * (1.0 - self.p.v2_stoploss_pct / 100.0)
                tp = c * (1.0 + self.p.v2_takeprofit_pct / 100.0)
                orders = self.buy_bracket(size=stake, limitprice=tp, stopprice=sl)
                self.parent_order, self.stop_order, self.limit_order = orders
                self.log(f"BUY v2 size={stake} @ {c:.2f} SL={sl:.2f} TP={tp:.2f}")

            elif self.p.version == "v3" and v3_buy:
                sl = c * (1.0 - self.p.v3_stoploss_pct / 100.0)
                tp = c * (1.0 + self.p.v3_takeprofit_pct / 100.0)
                orders = self.buy_bracket(size=stake, limitprice=tp, stopprice=sl)
                self.parent_order, self.stop_order, self.limit_order = orders
                self.log(f"BUY v3 size={stake} @ {c:.2f} SL={sl:.2f} TP={tp:.2f}")
        elif not risk_ok and not in_pos and not have_open and self.p.risk_verbose:
            self.log(f"ENTRY GATED by risk: {rmeta}")

        # Manage exits on sell signals (TradingView uses strategy.close in all versions)
        if in_pos:
            do_close = (
                (self.p.version == "v1" and v1_sell) or
                (self.p.version == "v2" and v2_sell) or
                (self.p.version == "v3" and v3_sell)
            )
            if do_close:
                self.cancel_children()
                self.close()
                self.log(f"CLOSE signal @ {c:.2f}")


# ---------- CLI / Runner ----------
@dataclass
class RunConfig:
    symbol: str = "AAPL"
    start: str = "2020-01-01"
    end: str | None = None
    cash: float = 100000.0
    commission: float = 0.001
    version: str = "v1"  # v1|v2|v3
    cash_fraction: float = 1.0


def run(cfg: RunConfig):
    df = fetch_yfinance_ohlcv(cfg.symbol, start=cfg.start, end=cfg.end)
    if df.empty:
        print(f"No data for {cfg.symbol}.")
        return

    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        FlawlessVictory,
        version=cfg.version,
        cash_fraction=cfg.cash_fraction,
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(cfg.cash)
    cerebro.broker.setcommission(commission=cfg.commission)

    start_val = cerebro.broker.getvalue()
    cerebro.run()
    end_val = cerebro.broker.getvalue()

    print(f"FlawlessVictory {cfg.version} ({cfg.symbol}) Start: {start_val:.2f}  End: {end_val:.2f}")


def main():
    p = argparse.ArgumentParser(description="Flawless Victory Strategy (Backtrader port).")
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cash", type=float, default=100000.0)
    p.add_argument("--commission", type=float, default=0.001)
    p.add_argument("--version", choices=["v1", "v2", "v3"], default="v1")
    p.add_argument("--cash-fraction", type=float, default=1.0)
    args = p.parse_args()

    cfg = RunConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        cash=args.cash,
        commission=args.commission,
        version=args.version,
        cash_fraction=args.cash_fraction,
    )
    run(cfg)


if __name__ == "__main__":
    main()