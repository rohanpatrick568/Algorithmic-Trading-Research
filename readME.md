# Algorithmic Trading Research

Backtesting strategies (Backtrader), risk-aware controls (DeepScalper-inspired), RL baselines, and LLM-driven sentiment research in a single workspace.

## Features
- Backtrader strategies
  - Flawless Victory (v1/v2/v3) with Bollinger/RSI/MFI and SL/TP: [`flawless_victory.FlawlessVictory`](scripts/backtrader/flawless_victory.py)
  - Risk-aware variant with volatility-based sizing, drawdown brake, downside deviation gate: [`fv_risk_aware.FlawlessVictory`](scripts/backtrader/fv_risk_aware.py)
  - Smoke test using synthetic OHLCV: [scripts/backtrader/test.py](scripts/backtrader/test.py)
- Reinforcement Learning (DeepScalper-style)
  - Gymnasium environment and BDQ-style scaffolding: [`gym_implementation.TradingEnvGym`](scripts/rl/gym_implementation.py)
  - Pretrained checkpoint: [models/bdq.pt](models/bdq.pt)
- LLM Sentiment (FinLlama-inspired)
  - Research notes and portfolio construction details: [TXTs/LLMSentiment.txt](TXTs/LLMSentiment.txt)

Key references in TXTs:
- DeepScalper (risk-aware auxiliary task, volatility, metrics): [TXTs/DeepScalper.txt](TXTs/DeepScalper.txt)
- DRL Day Trading (DDPG, allocation, drawdown): [TXTs/Beating_the_Stock_Market_with_a_Deep_Reinforcement_Learning_Day_Trading_System.txt](TXTs/Beating_the_Stock_Market_with_a_Deep_Reinforcement_Learning_Day_Trading_System.txt)
- Original Pine strategy: [TXTs/FlawlessVictoryStrategy.pine](TXTs/FlawlessVictoryStrategy.pine)

## Setup
- Create and activate virtual environment, then install deps:
  - Windows PowerShell
    - .\setup_env.sh
  - Or manual:
    - py -3 -m venv env
    - .\env\Scripts\Activate.ps1
    - pip install --upgrade pip
    - pip install -r requirements.txt

Files:
- Env bootstrap: [setup_env.sh](setup_env.sh)
- Requirements: [requirements.txt](requirements.txt)

## Quick Start

Backtrader smoke test:
```sh
python scripts/backtrader/test.py
```

Flawless Victory (Backtrader):
```sh
# v1: Bollinger(20, 1.0) + RSI guards
python scripts/backtrader/flawless_victory.py --symbol AAPL --version v1

# v2: Bollinger(17, 1.0) + RSI + SL/TP (6.604% / 2.328%)
python scripts/backtrader/flawless_victory.py --symbol AAPL --version v2

# v3: Bollinger v1 + MFI/RSI guards + SL/TP (8.882% / 2.317%)
python scripts/backtrader/flawless_victory.py --symbol AAPL --version v3
```

Risk-aware variant (DeepScalper-inspired):
```sh
python scripts/backtrader/fv_risk_aware.py --symbol AAPL --version v2
```

RL baseline (Gymnasium BDQ-style):
```sh
python scripts/rl/gym_implementation.py --symbol AAPL --start 2015-01-01 --end 2022-12-31 --episodes 1
```

## Strategy Details

- Flawless Victory port
  - v1: Bollinger(20, 1.0) buy when Close < lower band and RSI > 42; sell when Close > upper and RSI > 70.
  - v2: Same with Bollinger(17, 1.0), RSI upper=76 and bracket SL/TP (params in code).
  - v3: Uses MFI guards (buy if MFI < 60; sell if RSI > 65 and MFI > 64).
  - See: [`flawless_victory.FlawlessVictory`](scripts/backtrader/flawless_victory.py).

- Risk-aware controls (DeepScalper context)
  - Realized daily volatility scaling to target risk budget.
  - Drawdown brake with cooldown.
  - Downside deviation (Sortino-style) gating.
  - See: [`fv_risk_aware.FlawlessVictory`](scripts/backtrader/fv_risk_aware.py) and indicators in the same file.

## Reinforcement Learning

- Environment and training loop scaffolding modeled after DeepScalper’s intraday framing:
  - Env: [`gym_implementation.TradingEnvGym`](scripts/rl/gym_implementation.py)
  - CLI params: --window, --episodes, --max-steps, --gamma, --lr, --batch, --replay, --warmup, --target-sync, --epsilon-decay-steps.
  - Main checkpoint: [models/bdq.pt](models/bdq.pt)
- See DeepScalper methodology and metrics in [TXTs/DeepScalper.txt](TXTs/DeepScalper.txt).

## Data

- Market data via yfinance in Backtrader scripts.
- Utility to standardize OHLCV for Backtrader is included in [scripts/backtrader/test.py](scripts/backtrader/test.py) as `_ensure_bt_ohlcv`.
  - Fixes MultiIndex columns, enforces Open/High/Low/Close/Volume/OpenInterest.

## Repository Layout
- [scripts/backtrader](scripts/backtrader): Flawless Victory, risk-aware strategy, tests
  - [scripts/backtrader/flawless_victory.py](scripts/backtrader/flawless_victory.py)
  - [scripts/backtrader/fv_risk_aware.py](scripts/backtrader/fv_risk_aware.py)
  - [scripts/backtrader/test.py](scripts/backtrader/test.py)
- [scripts/rl](scripts/rl): Gymnasium RL baseline and tools
  - [scripts/rl/gym_implementation.py](scripts/rl/gym_implementation.py)
- [scripts/llm](scripts/llm): LLM sentiment utilities (see research in TXTs)
- [TXTs](TXTs): Papers and Pine sources
  - [TXTs/DeepScalper.txt](TXTs/DeepScalper.txt)
  - [TXTs/LLMSentiment.txt](TXTs/LLMSentiment.txt)
  - [TXTs/FlawlessVictoryStrategy.pine](TXTs/FlawlessVictoryStrategy.pine)
- [models](models): Saved RL models

## Troubleshooting
- PandasData AttributeError: tuple has no attribute lower
  - Ensure columns are flattened and named Open/High/Low/Close/Volume/OpenInterest (see `_ensure_bt_ohlcv` in [scripts/backtrader/test.py](scripts/backtrader/test.py)).

## Disclaimer
This repository is for research only. Nothing herein is financial advice. Use at your