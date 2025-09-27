from dataclasses import dataclass


@dataclass
class EnvConfig:
    symbol: str = "AAPL"
    timeframe: str = "1min"  # Intraday 1-minute bars
    lookback: int = 120  # history window for state
    episode_minutes: int = 6 * 60 + 30  # ~390 min per US trading day
    fee_rate: float = 0.0005
    slippage_bps: float = 1.0
    cash_start: float = 100_000.0
    max_position_pct: float = 0.9
    price_bins: int = 21  # BDQ discrete relative price levels
    qty_bins: int = 11    # BDQ discrete qty levels
    holdout_days: int = 0  # set >0 to reserve test days at tail
    
    # Phase 1 enhancements
    slippage_model: str = "proportional"  # "proportional", "fixed", "adaptive"
    slippage_random_shock: float = 0.5  # Additional random component in bps
    position_penalty_multiplier: float = 2.0  # Penalty multiplier when max position exceeded
    fractional_fees: bool = False  # Different fees for session open/close
    session_open_fee_rate: float = 0.0003  # Lower fees at session open
    session_close_fee_rate: float = 0.0007  # Higher fees at session close
    hard_position_cap: bool = True  # Hard cap vs soft penalty
    episode_sampling_mode: str = "random"  # "random", "mixed", "hard_days"
    hard_days_vol_threshold: float = 0.75  # Vol percentile for hard days


@dataclass
class TrainConfig:
    seed: int = 42
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 500_000
    warmup_steps: int = 10_000
    train_steps: int = 200_000
    target_update_tau: float = 0.005
    update_every: int = 1
    grad_clip: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 150_000

    # Hindsight bonus (Section 4.2)
    hindsight_weight: float = 0.5
    hindsight_horizon: int = 30  # 30 minutes lookahead within a day

    # Risk-aware auxiliary task (Section 4.3): predict realized volatility
    aux_weight: float = 0.1


FEATURES = {
    "ohlcv": ["open", "high", "low", "close", "volume"],
    # A compact technical set; can be extended to the 11 from the paper
    "indicators": ["rsi14", "mfi14", "bb_mid20", "bb_up20", "bb_dn20", "atr14"],
}
