import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import os

import numpy as np
import pandas as pd

# Optional deps
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    import backtrader as bt  # type: ignore
except Exception:
    bt = None

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception:
    gym = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


# ============================== Data Utils ==============================
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


def fetch_yfinance_ohlcv(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return _ensure_bt_ohlcv(df)


def fetch_prices(symbol: str, start: Optional[str], end: Optional[str]) -> pd.Series:
    if yf is not None:
        try:
            df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty and 'Close' in df.columns:
                s = df['Close'].dropna()
                if getattr(s.index, 'tz', None) is not None:
                    s.index = s.index.tz_localize(None)
                return s
        except Exception:
            pass
    # Synthetic fallback
    n = 1000
    rng = np.random.default_rng(42)
    rets = rng.normal(loc=0.0005, scale=0.02, size=n)
    prices = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.Series(prices, index=idx, name='Close')


def series_to_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0.0).clip(-0.2, 0.2)


# ============================== Backtrader: Risk-aware FV ==============================
if bt is not None:

    class RealizedVolatility(bt.Indicator):
        lines = ('vol',)
        params = dict(period=30, annualize=False)

        def __init__(self):
            ret = (self.data.close / self.data.close(-1)) - 1.0
            std = bt.ind.StdDev(ret, period=self.p.period)
            self.l.vol = std * (252.0 ** 0.5 if self.p.annualize else 1.0)

    class DownsideDeviation(bt.Indicator):
        lines = ('dd',)
        params = dict(period=30, target=0.0)

        def __init__(self):
            ret = (self.data.close / self.data.close(-1)) - 1.0
            downside = bt.If(ret < self.p.target, (ret - self.p.target) * (ret - self.p.target), 0.0)
            mean_down = bt.ind.SMA(downside, period=self.p.period)
            self.l.dd = mean_down ** 0.5

    class MoneyFlowIndex(bt.Indicator):
        lines = ('mfi',)
        params = dict(period=14)

        def __init__(self):
            tp = (self.data.high + self.data.low + self.data.close) / 3.0
            mf = tp * self.data.volume
            upmf = bt.If(tp > tp(-1), mf, 0.0)
            dnmf = bt.If(tp < tp(-1), mf, 0.0)
            pos = bt.ind.Sum(upmf, self.p.period)
            neg = bt.ind.Sum(dnmf, self.p.period)
            ratio = pos / (neg + 1e-9)
            mfi = 100.0 - (100.0 / (1.0 + ratio))
            self.lines.mfi = mfi

    class FlawlessVictory(bt.Strategy):
        params = dict(
            version="v1",
            cash_fraction=1.0,
            v2_stoploss_pct=6.604,
            v2_takeprofit_pct=2.328,
            v3_stoploss_pct=8.882,
            v3_takeprofit_pct=2.317,
            rsi_period=14,
            mfi_period=14,
            v1_rsi_lower=42,
            v1_rsi_upper=70,
            v2_rsi_lower=42,
            v2_rsi_upper=76,
            v3_mfi_lower=60,
            v3_rsi_upper=65,
            v3_mfi_upper=64,
            v1_bb_period=20,
            v1_bb_dev=1.0,
            v2_bb_period=17,
            v2_bb_dev=1.0,
            # Risk-aware controls
            risk_enabled=True,
            risk_target_daily_vol=0.015,
            risk_vol_period=30,
            risk_min_scale=0.25,
            risk_max_scale=1.25,
            risk_max_drawdown=0.20,
            risk_dd_cooldown_bars=5,
            risk_sortino_period=30,
            risk_max_downside_dev=0.02,
            risk_verbose=False,
        )

        def __init__(self):
            self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
            self.mfi = MoneyFlowIndex(self.data, period=self.p.mfi_period)
            self.bb1 = bt.ind.BollingerBands(self.data.close, period=self.p.v1_bb_period, devfactor=self.p.v1_bb_dev)
            self.bb2 = bt.ind.BollingerBands(self.data.close, period=self.p.v2_bb_period, devfactor=self.p.v2_bb_dev)

            self.vol = RealizedVolatility(self.data, period=self.p.risk_vol_period, annualize=False)
            self.ddown = DownsideDeviation(self.data, period=self.p.risk_sortino_period, target=0.0)
            self.equity_peak = None
            self.dd_cooldown = 0

            self.parent_order = None
            self.stop_order = None
            self.limit_order = None

        def log(self, txt):
            dtstr = self.datas[0].datetime.datetime(0).strftime("%Y-%m-%d")
            print(f"{dtstr} {txt}")

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

        def _risk_scale_and_gate(self):
            if not self.p.risk_enabled:
                return 1.0, True, {}
            eq = self.broker.getvalue()
            if self.equity_peak is None:
                self.equity_peak = eq
            self.equity_peak = max(self.equity_peak, eq)
            dd = (eq / self.equity_peak) - 1.0
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
            v = float(self.vol.vol[0])
            if math.isnan(v) or v <= 0:
                scale = 1.0
            else:
                scale = self.p.risk_target_daily_vol / max(v, 1e-9)
                scale = max(self.p.risk_min_scale, min(self.p.risk_max_scale, scale))
            ddv = float(self.ddown.dd[0])
            if not math.isnan(ddv) and ddv > self.p.risk_max_downside_dev:
                gated = True
                reasons["downside_dev"] = round(ddv, 4)
            rmeta = {
                "vol": round(v if not math.isnan(v) else 0.0, 6),
                "dd": round(dd, 4),
                **reasons,
            }
            return scale, (not gated), rmeta

        def next(self):
            c = self.data.close[0]
            in_pos = self.position.size != 0
            have_open = any(
                o and o.status in [bt.Order.Submitted, bt.Order.Accepted]
                for o in [self.parent_order, self.stop_order, self.limit_order]
            )

            v1_buy = (c < self.bb1.bot[0]) and (self.rsi[0] > self.p.v1_rsi_lower)
            v1_sell = (c > self.bb1.top[0]) and (self.rsi[0] > self.p.v1_rsi_upper)
            v2_buy = (c < self.bb2.bot[0]) and (self.rsi[0] > self.p.v2_rsi_lower)
            v2_sell = (c > self.bb2.top[0]) and (self.rsi[0] > self.p.v2_rsi_upper)
            v3_buy = (c < self.bb1.bot[0]) and (self.mfi[0] < self.p.v3_mfi_lower)
            v3_sell = (c > self.bb1.top[0]) and (self.rsi[0] > self.p.v3_rsi_upper) and (self.mfi[0] > self.p.v3_mfi_upper)

            scale, risk_ok, rmeta = self._risk_scale_and_gate()

            if self.p.risk_verbose:
                gates = {k: v for k, v in rmeta.items() if k not in ("vol", "dd")}
                self.log(f"RISK vol={rmeta.get('vol')} dd={rmeta.get('dd')} scale={scale:.2f} gates={gates}")

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

            elif not in_pos and not have_open and (not risk_ok) and self.p.risk_verbose:
                self.log(f"ENTRY GATED by risk: {rmeta}")

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


@dataclass
class BacktestConfig:
    symbol: str = "AAPL"
    start: str = "2020-01-01"
    end: Optional[str] = None
    cash: float = 100000.0
    commission: float = 0.001
    version: str = "v1"
    cash_fraction: float = 1.0


def run_backtest(cfg: BacktestConfig):
    if bt is None:
        print("Backtrader not available. Install 'backtrader'.")
        return
    df = fetch_yfinance_ohlcv(cfg.symbol, start=cfg.start, end=cfg.end)
    if df.empty:
        print(f"No data for {cfg.symbol}.")
        return
    data = bt.feeds.PandasData(dataname=df)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(FlawlessVictory, version=cfg.version, cash_fraction=cfg.cash_fraction, risk_verbose=True)
    cerebro.adddata(data)
    cerebro.broker.setcash(cfg.cash)
    cerebro.broker.setcommission(commission=cfg.commission)
    start_val = cerebro.broker.getvalue()
    cerebro.run()
    end_val = cerebro.broker.getvalue()
    print(f"FlawlessVictory {cfg.version} ({cfg.symbol}) Start: {start_val:.2f}  End: {end_val:.2f}")


# ============================== Gymnasium + BDQ (DeepScalper-style) ==============================
@dataclass
class EnvConfig:
    window: int = 32
    fee_bps: float = 5.0
    max_steps: Optional[int] = 500
    # Hindsight bonus
    hindsight_h: int = 10
    hindsight_w: float = 0.2


if gym is not None:

    class TradingEnvGym(gym.Env):
        metadata = {"render.modes": []}

        def __init__(self, returns: pd.Series, prices: pd.Series, cfg: EnvConfig):
            super().__init__()
            rarr = returns.astype(np.float32).to_numpy()
            if isinstance(rarr, np.ndarray) and rarr.ndim > 1:
                rarr = np.squeeze(rarr)
            parr = prices.astype(np.float32).to_numpy()
            if isinstance(parr, np.ndarray) and parr.ndim > 1:
                parr = np.squeeze(parr)
            self.r: np.ndarray = rarr
            self.px: np.ndarray = parr
            self.cfg = cfg
            self.t = 0
            self.pos = 0.0
            self._step_count = 0

            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.window,), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([3, 5])

        def _obs(self) -> np.ndarray:
            w = self.cfg.window
            x = self.r[self.t - w:self.t]
            if x.shape[0] < w:
                pad = np.zeros(w, dtype=np.float32)
                pad[-x.shape[0]:] = x
                x = pad
            return np.asarray(x, dtype=np.float32).reshape(-1)

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            self.t = max(self.cfg.window, 1)
            self.pos = 0.0
            self._step_count = 0
            return self._obs(), {}

        def step(self, action):
            if isinstance(action, (list, tuple, np.ndarray)):
                a = np.asarray(action, dtype=np.int64).reshape(-1)
            else:
                a = np.array([int(action), 0], dtype=np.int64)
            dir_map = [-1.0, 0.0, +1.0]
            size_map = [0.0, 0.25, 0.5, 0.75, 1.0]
            direction = dir_map[int(a[0])]
            size = size_map[int(a[1])]
            target_pos = direction * size

            # Transaction cost
            dpos = abs(target_pos - self.pos)
            fee = dpos * (self.cfg.fee_bps / 1e4)

            terminated = False
            truncated = False
            if self.t >= len(self.r) - 1:
                terminated = True
                return self._obs(), 0.0, terminated, truncated, {}

            # Immediate reward
            ret_next = float(self.r[self.t])
            reward = target_pos * ret_next - fee

            # Hindsight bonus using price horizon h (DeepScalper)
            h = self.cfg.hindsight_h
            if self.t + h < len(self.px):
                bonus = (float(self.px[self.t + h]) - float(self.px[self.t])) / max(float(self.px[self.t]), 1e-9)
                reward += self.cfg.hindsight_w * target_pos * bonus

            # Advance
            self.pos = target_pos
            self.t += 1
            self._step_count += 1

            if self.cfg.max_steps is not None and self._step_count >= self.cfg.max_steps:
                truncated = True

            return self._obs(), float(reward), terminated, truncated, {}


if torch is not None:

    class BDQ(nn.Module):
        def __init__(self, state_dim: int, branch_sizes: List[int], hidden: int = 128, aux_vol: bool = True):
            super().__init__()
            self.aux_vol = aux_vol
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.value = nn.Linear(hidden, 1)
            self.advantages = nn.ModuleList([nn.Linear(hidden, n) for n in branch_sizes])
            self.vol_head = nn.Linear(hidden, 1) if aux_vol else None

        def forward(self, x: Any) -> Tuple[List[Any], Optional[Any]]:
            h = self.feature(x)
            v = self.value(h)
            qs = []  # type: List[Any]
            for head in self.advantages:
                a = head(h)
                a = a - a.mean(dim=1, keepdim=True)
                q = v + a
                qs.append(q)
            vpred = self.vol_head(h) if self.vol_head is not None else None
            return qs, vpred

        def q_for_actions(self, x: Any, actions: Any) -> Any:
            qs, _ = self.forward(x)
            total = 0.0
            for b, q in enumerate(qs):
                idx = actions[:, b].long().unsqueeze(1)
                qb = q.gather(1, idx).squeeze(1)
                total = qb if b == 0 else (total + qb)
            return total

        def greedy_actions(self, x: Any) -> Any:
            qs, _ = self.forward(x)
            acts = [q.argmax(dim=1) for q in qs]
            return torch.stack(acts, dim=1)


class Replay:
    def __init__(self, capacity: int, state_dim: int, n_branches: int):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, n_branches), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def push(self, s, a, r, ns, d):
        i = self.idx
        self.state[i] = np.asarray(s, dtype=np.float32).reshape(-1)
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = np.asarray(ns, dtype=np.float32).reshape(-1)
        self.done[i] = d
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch: int):
        n = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, n, size=batch)
        return (
            torch.from_numpy(self.state[idxs]),
            torch.from_numpy(self.action[idxs]),
            torch.from_numpy(self.reward[idxs]),
            torch.from_numpy(self.next_state[idxs]),
            torch.from_numpy(self.done[idxs]),
        )

    def __len__(self):
        return self.capacity if self.full else self.idx


@dataclass
class TrainConfig:
    symbol: str = 'AAPL'
    start: Optional[str] = '2015-01-01'
    end: Optional[str] = '2022-12-31'
    window: int = 32
    episodes: int = 1
    max_steps: Optional[int] = 500
    gamma: float = 0.99
    lr: float = 1e-3
    batch: int = 64
    replay: int = 100_000
    warmup: int = 1_000
    target_sync: int = 500
    steps_per_update: int = 1
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 20_000
    seed: int = 42
    # Aux volatility head
    aux_vol_lambda: float = 0.1
    # Checkpoints
    save_path: Optional[str] = os.path.join('models', 'bdq.pt')
    load_path: Optional[str] = None


def linear_epsilon(step: int, cfg: TrainConfig) -> float:
    t = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + t * (cfg.epsilon_final - cfg.epsilon_start)


def train_bdq(cfg: TrainConfig):
    if any(x is None for x in [gym, torch, nn, optim]):
        print("Gymnasium/PyTorch not available. Install 'gymnasium', 'torch'.")
        return
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    prices = fetch_prices(cfg.symbol, cfg.start, cfg.end)
    returns = series_to_returns(prices)
    env = TradingEnvGym(returns, prices, EnvConfig(window=cfg.window, max_steps=cfg.max_steps))

    state_dim = cfg.window
    branch_sizes = [3, 5]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    online = BDQ(state_dim, branch_sizes, aux_vol=True).to(device)
    target = BDQ(state_dim, branch_sizes, aux_vol=True).to(device)
    # Load checkpoint if provided
    if cfg.load_path and os.path.isfile(cfg.load_path):
        sd = torch.load(cfg.load_path, map_location=device)
        online.load_state_dict(sd)
    target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=cfg.lr)
    rb = Replay(cfg.replay, state_dim, len(branch_sizes))

    global_step = 0
    for ep in range(cfg.episodes):
        s, _ = env.reset(seed=cfg.seed + ep)
        ep_reward = 0.0
        done = False
        steps = 0
        while not done:
            eps = linear_epsilon(global_step, cfg)
            if np.random.rand() < eps:
                a = np.array([np.random.randint(n) for n in branch_sizes], dtype=np.int64)
            else:
                with torch.no_grad():
                    sa = torch.from_numpy(s).unsqueeze(0).to(device)
                    a_t = online.greedy_actions(sa).cpu().numpy()[0]
                    a = a_t.astype(np.int64)

            ns, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            rb.push(s, a, r, ns, float(done))
            s = ns
            ep_reward += r
            steps += 1
            global_step += 1

            if len(rb) >= cfg.warmup and (global_step % cfg.steps_per_update == 0):
                bs, ba, br, bns, bd = rb.sample(cfg.batch)
                bs = bs.float().to(device)
                ba = ba.long().to(device)
                br = br.float().to(device)
                bns = bns.float().to(device)
                bd = bd.float().to(device)

                with torch.no_grad():
                    next_qs, _ = target.forward(bns)
                    q_next_sum = 0.0
                    for q in next_qs:
                        q_next_sum = q.max(dim=1).values if isinstance(q_next_sum, float) else (q_next_sum + q.max(dim=1).values)
                    tgt = br + (1.0 - bd) * cfg.gamma * q_next_sum

                pred = online.q_for_actions(bs, ba)
                td_loss = nn.MSELoss()(pred, tgt)

                # Aux volatility prediction from next_state window
                _, vpred = online.forward(bs)
                vtarget = bns.std(dim=1, keepdim=True)  # daily realized vol over window
                aux_loss = (nn.MSELoss()(vpred, vtarget) if vpred is not None else torch.tensor(0.0, device=device))

                loss = td_loss + cfg.aux_vol_lambda * aux_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=1.0)
                opt.step()

                if global_step % 100 == 0:
                    print(f"step {global_step} loss {loss.item():.6f} td {td_loss.item():.6f} aux {aux_loss.item():.6f} eps {eps:.3f} ep_reward {ep_reward:.4f}")

            if global_step % cfg.target_sync == 0:
                target.load_state_dict(online.state_dict())

    print(f"episode {ep+1} steps {steps} reward {ep_reward:.4f}")

    # Save checkpoint
    if cfg.save_path:
        os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
        torch.save(online.state_dict(), cfg.save_path)
        print(f"Saved model to {cfg.save_path}")


# ============================== CLI ==============================
def main():
    p = argparse.ArgumentParser(description="Combined Backtrader risk-aware FV and Gymnasium BDQ (DeepScalper-style)")
    p.add_argument('--mode', choices=['backtest', 'train'], default=None)
    # Backtest args
    p.add_argument('--symbol', default='AAPL')
    p.add_argument('--start', default='2020-01-01')
    p.add_argument('--end', default=None)
    p.add_argument('--cash', type=float, default=100000.0)
    p.add_argument('--commission', type=float, default=0.001)
    p.add_argument('--version', choices=['v1', 'v2', 'v3'], default='v1')
    p.add_argument('--cash-fraction', type=float, default=1.0)
    # Train args
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--save-path', default=os.path.join('models', 'bdq.pt'))
    p.add_argument('--load-path', default=None)
    args = p.parse_args()

    # Simple helper to prompt with defaults
    def _ask(prompt: str, default: Optional[str] = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        val = input(f"{prompt}{suffix}: ").strip()
        return default if (val == "" and default is not None) else val

    # If mode not provided, prompt for it interactively
    mode = args.mode
    if mode is None:
        mode = _ask("Select mode (train/backtest)", "train").lower()
        if mode not in ("train", "backtest"):
            print("Invalid mode. Use 'train' or 'backtest'.")
            return

    if mode == 'backtest':
        # Fill from args or prompt
        symbol = args.symbol or _ask("Symbol", "AAPL")
        start = args.start or _ask("Start (YYYY-MM-DD)", "2020-01-01")
        end = args.end if args.end is not None else _ask("End (YYYY-MM-DD or blank)", "") or None
        version = args.version or _ask("Version (v1/v2/v3)", "v1")
        try:
            cash = float(args.cash) if args.cash is not None else float(_ask("Cash", "100000"))
        except Exception:
            cash = 100000.0
        try:
            commission = float(args.commission) if args.commission is not None else float(_ask("Commission", "0.001"))
        except Exception:
            commission = 0.001
        try:
            cash_fraction = float(args['cash_fraction']) if isinstance(args, dict) and 'cash_fraction' in args else float(_ask("Cash fraction (0..1)", "1.0"))
        except Exception:
            cash_fraction = 1.0
        cfg = BacktestConfig(
            symbol=symbol,
            start=start,
            end=end,
            cash=cash,
            commission=commission,
            version=version,
            cash_fraction=cash_fraction,
        )
        run_backtest(cfg)
    else:
        # Training: prompt for key params if not provided
        symbol = args.symbol or _ask("Symbol", "AAPL")
        window = args.window or int(_ask("Window (state size)", "32"))
        episodes = args.episodes or int(_ask("Episodes", "1"))
        max_steps = args.max_steps if args.max_steps is not None else int(_ask("Max steps per episode (0=all)", "500"))
        save_path = args.save_path or _ask("Save path", os.path.join('models', 'bdq.pt'))
        load_path = args.load_path or _ask("Load path (leave blank if none)", "") or None
        tcfg = TrainConfig(
            symbol=symbol,
            start='2015-01-01',
            end='2022-12-31',
            window=window,
            episodes=episodes,
            max_steps=(None if max_steps == 0 else max_steps),
            save_path=save_path,
            load_path=load_path,
        )
        train_bdq(tcfg)


if __name__ == '__main__':
    main()
