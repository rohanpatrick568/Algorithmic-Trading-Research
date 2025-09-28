"""Run the FlawlessVictoryRiskAware strategy in live paper trading mode (Alpaca).

Usage:
  export ALPACA_API_KEY=...  # or set in .env (Lumibot auto-loads)
  export ALPACA_API_SECRET=...
  python -m scripts.live.live_paper_trading --symbol AAPL --version v1

This script intentionally avoids hardcoding credentials. DO NOT embed real keys
in source control. For convenience, both ALPACA_* and generic API_KEY / API_SECRET
are supported (ALPACA_* preferred).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Ensure repository root is on path (two levels up from scripts/live)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from lumibot.brokers import Alpaca  # type: ignore
from lumibot.traders import Trader  # type: ignore

from scripts.lumibot.fv_ra import FlawlessVictoryRiskAware, RiskConfig  # type: ignore
from scripts.live.config import get_alpaca_config  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Live paper trading runner")
    p.add_argument("--symbol", default="AAPL", help="Ticker symbol to trade")
    p.add_argument("--version", default="v1", choices=["v1", "v2", "v3"], help="Strategy version")
    p.add_argument("--cash-fraction", type=float, default=0.8, help="Fraction of available cash to deploy")
    p.add_argument("--target-daily-vol", type=float, default=0.015, help="Target daily vol for scaling")
    p.add_argument("--max-drawdown", type=float, default=0.15, help="Max drawdown before cooldown")
    p.add_argument("--dd-cooldown", type=int, default=3, help="Bars (days) to pause after drawdown breach")
    p.add_argument("--verbose-risk", action="store_true", help="Verbose risk logging")
    return p.parse_args()


def build_parameters(args) -> dict[str, Any]:
    risk_cfg = RiskConfig(
        enabled=True,
        target_daily_vol=args.target_daily_vol,
        max_drawdown=args.max_drawdown,
        dd_cooldown_bars=args.dd_cooldown,
        verbose=args.verbose_risk,
    )
    return {
        "version": args.version,
        "symbol": args.symbol,
        "cash_fraction": args.cash_fraction,
        "risk_config": risk_cfg,
    }


def main():
    args = parse_args()

    # Assemble broker config
    alpaca_cfg = get_alpaca_config(require=True)

    print("Starting Live Paper Trading: FlawlessVictoryRiskAware")
    print("=====================================================")
    print(f"Symbol: {args.symbol} | Version: {args.version}")
    print(f"Broker: Alpaca (paper={'YES' if alpaca_cfg['PAPER'] else 'NO'})")

    trader = Trader()
    broker = Alpaca(alpaca_cfg)

    params = build_parameters(args)
    strategy = FlawlessVictoryRiskAware(broker=broker, parameters=params)

    trader.add_strategy(strategy)
    trader.run_all()


if __name__ == "__main__":
    main()
