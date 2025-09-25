from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from .config import EnvConfig, FEATURES
from .model import BranchingDuelingQNet


@dataclass
class InferenceConfig:
    model_path: str
    price_bins: int
    qty_bins: int
    lookback: int


try:
    from lumibot.strategies.strategy import Strategy  # type: ignore
except Exception:
    Strategy = object  # fallback for type checking


class DeepScalperLumibotStrategy(Strategy):  # type: ignore
    """
    Minimal Lumibot-compatible strategy wrapper calling the trained BDQ model.
    Assumes the hosting Lumibot Strategy will:
      - Feed 1-minute bars
      - Provide get_historical_prices to build features
      - Execute market or limit orders based on predicted bins
    """

    def initialize(self):  # type: ignore
        cfg: InferenceConfig = self.parameters["inference_cfg"]
        self.lookback = cfg.lookback
        self.price_bins = cfg.price_bins
        self.qty_bins = cfg.qty_bins
        # Symbols
        symbols = self.parameters.get("symbols", ["AAPL"])
        self.symbol = symbols[0]
        # Backtesting timing if provided
        bt = self.parameters.get("backtest_window")
        if bt and isinstance(bt, tuple) and len(bt) == 2:
            self._backtesting_start, self._backtesting_end = bt
        # Quote asset default (USD)
        try:
            from lumibot.entities import Asset
            self.quote_asset = Asset("USD", asset_type="forex")
        except Exception:
            pass
        # Drive minute loop
        self.sleeptime = "1M"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(cfg.model_path)
        self._feat_cols = FEATURES["ohlcv"] + FEATURES["indicators"]
        self._last_decision_minute = None

    def _load_model(self, path: str):
        import os
        # Build model first
        obs_dim = len(FEATURES["ohlcv"] + FEATURES["indicators"]) * self.lookback + 3
        self.model = BranchingDuelingQNet(obs_dim, self.price_bins, self.qty_bins).to(self.device)
        # Try loading a checkpoint if provided
        loaded = False
        if path and isinstance(path, str) and os.path.isfile(path):
            try:
                state = torch.load(path, map_location="cpu")
                loaded = True
            except Exception:
                try:
                    state = torch.load(path, map_location="cpu", weights_only=False)
                    loaded = True
                except Exception as e:
                    print(f"[DeepScalper] Warning: failed to load checkpoint at {path}: {e}. Using random weights.")
            if loaded:
                try:
                    self.model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
                except Exception as e:
                    print(f"[DeepScalper] Warning: incompatible checkpoint format: {e}. Using random weights.")
        else:
            print(f"[DeepScalper] No checkpoint found at {path}. Using random weights.")
        self.model.eval()

    def _build_obs(self):
        bars = self.get_historical_prices(self.symbol, self.lookback + 1, "minute")
        if bars is None or getattr(bars, "df", None) is None:
            return None
        df = bars.df.copy()
        # Pull a full multi-col frame
        # Ensure expected columns
        df = df.rename(columns={"adj_close": "close"})
        for c in ["open","high","low","close","volume"]:
            if c not in df.columns and c != "volume":
                # If only close exists, duplicate as fallback
                df[c] = df.get("close", np.nan)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        # Add indicators if available on host; else compute basic ones
        try:
            from ..deepscalper.data import add_indicators
            df = df[["open", "high", "low", "close", "volume"]]
            df = add_indicators(df)
        except Exception:
            pass
        if len(df) < self.lookback + 1:
            return None
        window = df.iloc[-self.lookback:][self._feat_cols].to_numpy(dtype=np.float32)
        # private state
        price = float(df.iloc[-1]["close"])
        pos = float(self.get_position(self.symbol).quantity) if self.get_position(self.symbol) else 0.0
        cash = float(self.get_cash())
        equity = cash + pos * price
        cash_pct = np.clip(cash / max(equity, 1e-6), 0, 1)
        # time_left ~ not available; set 0.5 heuristic or infer from timestamp if needed
        priv = np.array([pos, cash_pct, 0.5], dtype=np.float32)
        obs = np.concatenate([window.flatten(), priv], dtype=np.float32)
        return obs

    def on_trading_iteration(self):  # type: ignore
        # Act at most once per new minute
        dt = self.get_datetime()
        minute_key = dt.replace(second=0, microsecond=0)
        if self._last_decision_minute == minute_key:
            return
        self._last_decision_minute = minute_key

        obs = self._build_obs()
        if obs is None:
            return
        o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qp, qq, _, _ = self.model(o)
            ap = int(qp.argmax(dim=1).item())
            aq = int(qq.argmax(dim=1).item())

        # Map bins to desired order
        price = float(self.get_last_price(self.symbol))
        atr = 0.01
        try:
            # approximate ATR via past minute highs/lows if available
            hp = self.get_historical_prices(self.symbol, 15, "minute")
            atr = float((hp["high"] - hp["low"]).ewm(span=14).mean().iloc[-1])
        except Exception:
            pass
        tick = max(0.01, 0.05 * atr)
        rel = (ap / (self.price_bins - 1)) * 2 - 1
        target_price = price + rel * 5 * tick

        qty_frac = (aq / (self.qty_bins - 1))
        side = 1 if qty_frac >= 0.5 else -1
        qty_frac = abs(qty_frac - 0.5) * 2

        equity = float(self.get_portfolio_value())
        max_value = 0.9 * equity
        shares = int((qty_frac * max_value) / max(price, 1e-6))

        if shares <= 0:
            return

        if side > 0:
            self.buy(self.symbol, shares=shares, limit_price=target_price)
            self.log_message(f"DeepScalper BUY {shares} at ~{target_price:.2f} (bin {ap},{aq})")
        else:
            # For safety in equities, use sell of held shares only
            held = int(self.get_position(self.symbol).quantity) if self.get_position(self.symbol) else 0
            if held > 0:
                qty = min(held, shares)
                self.sell(self.symbol, shares=qty, limit_price=target_price)
                self.log_message(f"DeepScalper SELL {qty} at ~{target_price:.2f} (bin {ap},{aq})")
