from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

try:  # normal package-relative usage
    from .config import EnvConfig, TrainConfig, FEATURES  # type: ignore
    from .data import MarketData  # type: ignore
except Exception:  # fallback when executed without package context
    from config import EnvConfig, TrainConfig, FEATURES  # type: ignore
    from data import MarketData  # type: ignore


def realized_vol(close: np.ndarray, horizon: int) -> float:
    if len(close) < 2:
        return 0.0
    r = np.diff(np.log(close + 1e-12))
    if len(r) == 0:
        return 0.0
    return float(np.std(r) * np.sqrt(252*390/horizon))


def downside_deviation(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Calculate downside deviation below threshold"""
    if len(returns) < 2:
        return 0.0
    downside_returns = returns[returns < threshold]
    if len(downside_returns) == 0:
        return 0.0
    return float(np.std(downside_returns))


def max_drawdown_depth(prices: np.ndarray) -> float:
    """Calculate maximum drawdown as percentage of peak"""
    if len(prices) < 2:
        return 0.0
    
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    max_dd = np.min(drawdown)
    return float(abs(max_dd))


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


def calculate_slippage(env_cfg: EnvConfig, price: float, volume: int, position_size: float, time_of_day: int) -> float:
    """Calculate adaptive slippage based on market conditions and position size"""
    base_slippage_bps = env_cfg.slippage_bps
    
    if env_cfg.slippage_model == "fixed":
        return price * base_slippage_bps / 10000
    
    # Proportional slippage based on position size and volume
    volume_impact = min(0.1, abs(position_size) / max(volume, 1000))  # Cap at 10% impact
    size_multiplier = 1.0 + volume_impact * 2  # Up to 3x slippage for large orders
    
    if env_cfg.slippage_model == "adaptive":
        # Add time-of-day effects (higher at open/close)
        minutes_from_open = time_of_day % 390  # Minutes since 9:30 AM
        if minutes_from_open < 30 or minutes_from_open > 360:  # First/last 30 mins
            size_multiplier *= 1.5
        
    # Add random shock component
    if env_cfg.slippage_random_shock > 0:
        random_shock = np.random.normal(0, env_cfg.slippage_random_shock / 10000)
        size_multiplier += random_shock
    
    return price * base_slippage_bps * size_multiplier / 10000


def calculate_fees(env_cfg: EnvConfig, notional: float, time_of_day: int) -> float:
    """Calculate trading fees with optional session-based variation"""
    if not env_cfg.fractional_fees:
        return notional * env_cfg.fee_rate
    
    # Use different fees based on time of day
    minutes_from_open = time_of_day % 390
    if minutes_from_open < 60:  # First hour - session open
        fee_rate = env_cfg.session_open_fee_rate
    elif minutes_from_open > 330:  # Last hour - session close
        fee_rate = env_cfg.session_close_fee_rate
    else:
        fee_rate = env_cfg.fee_rate
        
    return notional * fee_rate


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
        
        # Calculate daily volatility for hard day identification
        if self.env_cfg.episode_sampling_mode in ["mixed", "hard_days"]:
            self._day_volatilities = []
            for start_idx, end_idx in out:
                day_closes = self.md.df.iloc[start_idx:end_idx]['close'].values
                if len(day_closes) > 1:
                    day_vol = realized_vol(day_closes, len(day_closes))
                else:
                    day_vol = 0.0
                self._day_volatilities.append(day_vol)
            
            # Calculate volatility threshold for hard days
            if len(self._day_volatilities) > 0:
                self._vol_threshold = np.percentile(self._day_volatilities, self.env_cfg.hard_days_vol_threshold * 100)
            else:
                self._vol_threshold = 0.0
        else:
            self._day_volatilities = []
            self._vol_threshold = 0.0
            
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
        
        # Choose day based on sampling mode
        if self.env_cfg.episode_sampling_mode == "hard_days" and len(self._day_volatilities) > 0:
            # Only sample from high volatility days
            hard_day_indices = [i for i, vol in enumerate(self._day_volatilities) if vol >= self._vol_threshold]
            if hard_day_indices:
                day_idx = self._rng.choice(hard_day_indices)
            else:
                day_idx = self._rng.randint(0, len(self.day_indices))
        elif self.env_cfg.episode_sampling_mode == "mixed" and len(self._day_volatilities) > 0:
            # 70% normal days, 30% hard days
            if self._rng.random() < 0.7:
                # Normal day sampling
                day_idx = self._rng.randint(0, len(self.day_indices))
            else:
                # Hard day sampling
                hard_day_indices = [i for i, vol in enumerate(self._day_volatilities) if vol >= self._vol_threshold]
                if hard_day_indices:
                    day_idx = self._rng.choice(hard_day_indices)
                else:
                    day_idx = self._rng.randint(0, len(self.day_indices))
        else:
            # Random sampling (default)
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
        
        # Check position limits with configurable enforcement
        current_position_value = abs(self.pos * price)
        max_position_value = self.cash * self.env_cfg.max_position_pct
        position_exceeded = current_position_value > max_position_value
        
        if position_exceeded and self.env_cfg.hard_position_cap:
            # Hard cap - reject the trade
            shares = 0
            executed = False
            exec_price = price
        else:
            # Simulate limit-like execution: if favorable price traded inside [low,high]
            bar = self.md.df.iloc[self.t]
            low_v = _to_float(bar["low"]) 
            high_v = _to_float(bar["high"])
            volume_v = _to_float(bar["volume"])

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
                # Use enhanced slippage calculation
                slip = calculate_slippage(self.env_cfg, price, volume_v, shares, self.t - self._episode_slice[0])
                exec_price = price + (slip if side > 0 else -slip)
                executed = True

        fee = 0.0
        position_penalty = 0.0
        if executed and shares > 0:
            notional = exec_price * shares
            
            # Use enhanced fee calculation
            fee = calculate_fees(self.env_cfg, notional, self.t - self._episode_slice[0])
            
            if side > 0:
                self.cash -= notional + fee
                self.pos += shares
            else:
                self.cash += notional - fee
                self.pos -= shares
                
        # Apply position penalty if soft cap exceeded
        if position_exceeded and not self.env_cfg.hard_position_cap:
            excess_ratio = (current_position_value - max_position_value) / max_position_value
            position_penalty = excess_ratio * self.env_cfg.position_penalty_multiplier * abs(r)

        # mark-to-market PnL from t->t+1
        next_price = _to_float(self.md.df.iloc[self.t + 1]["close"]) if self.t + 1 < self._episode_slice[1] else price
        pnl_inst = (next_price - price) * self.pos
        r = pnl_inst - fee - position_penalty

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

        # Calculate all auxiliary targets
        h = self.train_cfg.hindsight_horizon
        if len(self._last_prices) >= h + 2:
            recent_prices = np.array(self._last_prices[-h - 1:])
            recent_returns = np.diff(np.log(recent_prices + 1e-12))
            
            aux_vol_target = realized_vol(recent_prices, h)
            aux_downside_target = downside_deviation(recent_returns)
            aux_drawdown_target = max_drawdown_depth(recent_prices)
        else:
            aux_vol_target = 0.0
            aux_downside_target = 0.0
            aux_drawdown_target = 0.0
            
        aux_targets = {
            "aux_target": aux_vol_target,  # Legacy compatibility
            "aux_vol_target": aux_vol_target,
            "aux_downside_target": aux_downside_target,
            "aux_drawdown_target": aux_drawdown_target,
            "r_raw": r
        }
        
        return obs, r_h, done, False, {**info, **aux_targets}

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
