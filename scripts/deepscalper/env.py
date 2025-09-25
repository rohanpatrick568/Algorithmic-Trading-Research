from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .config import EnvConfig, TrainConfig, FEATURES
from .data import MarketData


def realized_vol(close: np.ndarray, horizon: int) -> float:
    if len(close) < 2:
        return 0.0
    r = np.diff(np.log(close + 1e-12))
    if len(r) == 0:
        return 0.0
    return float(np.std(r) * np.sqrt(252*390/horizon))


def _to_float(x) -> float:
    """Robustly extract a Python float from possible pandas/NumPy scalars or 1-length Series/arrays."""
    # Fast path for numeric types
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    # pandas Series/Index with 1 element often support .item()
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    # Fallback to iloc[0] if available (pandas Series/DataFrame cell)
    if hasattr(x, "iloc"):
        try:
            return float(x.iloc[0])
        except Exception:
            pass
    # Last resort: convert via NumPy array and take first element
    arr = np.asarray(x).reshape(-1)
    return float(arr[0]) if arr.size > 0 else float(x)


class DeepScalperEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, md: MarketData, env_cfg: EnvConfig, train_cfg: TrainConfig):
        super().__init__()
        self.md = md
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg

        # Build observation space: last lookback of selected features flattened
        feat_cols = FEATURES["ohlcv"] + FEATURES["indicators"]
        self.feat_cols = feat_cols
        self.lookback = env_cfg.lookback
        self.obs_dim = len(feat_cols) * self.lookback + 3  # + private state: pos, cash_pct, time_left

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([env_cfg.price_bins, env_cfg.qty_bins])

        # episode partition: per day
        self.day_indices = self._split_by_day()
        self._rng = np.random.RandomState( train_cfg.seed )
        self._reset_episode_state()

    def _split_by_day(self):
        # Assume minute index with calendar days; group by date
        dates = self.md.df.index.tz_convert(None) if hasattr(self.md.df.index, 'tz') else self.md.df.index
        date_keys = pd.to_datetime(dates).date
        idx = {}
        prev = None
        start = 0
        out = []
        for i, d in enumerate(date_keys):
            if prev is None:
                prev = d
                start = i
            elif d != prev:
                out.append((start, i))
                prev = d
                start = i
        out.append((start, len(self.md.df)))
        # filter days shorter than lookback + horizon
        horizon = self.train_cfg.hindsight_horizon
        out = [p for p in out if p[1]-p[0] >= self.lookback + max(60, horizon)+5]
        return out

    def _reset_episode_state(self):
        self.cash = self.env_cfg.cash_start
        self.pos = 0.0
        self.t = None
        self._episode_slice = None
        self._done = False
        self._last_prices = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_episode_state()
        # choose random day
        day_idx = self._rng.randint(0, len(self.day_indices))
        a, b = self.day_indices[day_idx]
        # limit to episode_minutes windows if available
        length = min(self.env_cfg.episode_minutes, b-a)
        self._episode_slice = (a, a+length)
        self.t = a + self.lookback
        obs = self._build_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        assert self._episode_slice is not None
        price_bin, qty_bin = int(action[0]), int(action[1])
        row = self.md.df.iloc[self.t]
        price = _to_float(row["close"])  # reference price

        # Map bins to relative price offset and quantity fraction
        atr_val = self.md.df.iloc[self.t]["atr14"]
        atr = _to_float(atr_val) or 0.01
        tick = max(0.01, 0.05 * atr)
        rel = (price_bin / (self.env_cfg.price_bins - 1)) * 2 - 1  # [-1,1]
        target_price = price + rel * 5 * tick

        qty_frac = (qty_bin / (self.env_cfg.qty_bins - 1))  # [0,1]
        side = 1 if qty_frac >= 0.5 else -1
        qty_frac = abs(qty_frac - 0.5) * 2

        max_shares = (self.cash + self.pos * price) * self.env_cfg.max_position_pct / max(price, 1e-6)
        shares = float(np.floor(qty_frac * max_shares))

        # Simulate limit-like execution: if favorable price traded inside [low,high]
        bar = self.md.df.iloc[self.t]
        low_v = _to_float(bar["low"]) 
        high_v = _to_float(bar["high"]) 

        executed = False
        exec_price = price
        if side > 0:
            if low_v <= target_price <= high_v:
                exec_price = target_price
                executed = True
        else:
            if low_v <= target_price <= high_v:
                exec_price = target_price
                executed = True

        if not executed and shares > 0:
            slip = self.env_cfg.slippage_bps * 1e-4 * price
            exec_price = price + (slip if side > 0 else -slip)
            executed = True

        fee = 0.0
        if executed and shares > 0:
            if side > 0:
                cost = exec_price * shares
                fee = self.env_cfg.fee_rate * cost
                self.cash -= cost + fee
                self.pos += shares
            else:
                revenue = exec_price * shares
                fee = self.env_cfg.fee_rate * revenue
                self.cash += revenue - fee
                self.pos -= shares

        # mark-to-market PnL from t->t+1
        next_price = _to_float(self.md.df.iloc[self.t + 1]["close"]) if self.t + 1 < self._episode_slice[1] else price
        pnl_inst = (next_price - price) * self.pos
        r = pnl_inst - fee

        # Hindsight bonus
        h = self.train_cfg.hindsight_horizon
        t2 = min(self.t + h, self._episode_slice[1] - 1)
        hindsight = (_to_float(self.md.df.iloc[t2]["close"]) - price) * self.pos
        r_h = r + self.train_cfg.hindsight_weight * hindsight

        self.t += 1
        self._last_prices.append(price)
        done = self.t >= self._episode_slice[1] - 1
        if done and self.pos != 0:
            final_price = _to_float(self.md.df.iloc[self.t]["close"])
            self.cash += self.pos * final_price
            self.pos = 0.0

        obs = self._build_obs()
        info = {
            "price": price,
            "exec_price": exec_price if executed else None,
            "shares": shares if executed else 0,
            "pos": self.pos,
            "cash": self.cash,
        }

        aux_target = realized_vol(np.array(self._last_prices[-h - 1 :]), h) if len(self._last_prices) >= h + 2 else 0.0
        return obs, r_h, done, False, {**info, "aux_target": aux_target, "r_raw": r}

    def _build_obs(self) -> np.ndarray:
        a, b = self._episode_slice
        idx = slice(self.t - self.lookback, self.t)
        window = self.md.df.iloc[idx][self.feat_cols].to_numpy(dtype=np.float32)
        # private state
        if self.t < b:
            price = _to_float(self.md.df.iloc[self.t]["close"])
        else:
            price = _to_float(self.md.df.iloc[b - 1]["close"])
        equity = self.cash + self.pos * price
        cash_pct = np.clip(self.cash / max(equity, 1e-6), 0.0, 1.0)
        time_left = (b - self.t) / max(b - a, 1)
        priv = np.array([self.pos, cash_pct, time_left], dtype=np.float32)
        return np.concatenate([window.flatten(), priv], dtype=np.float32)

    def render(self):
        return None


# (pandas imported at top)
