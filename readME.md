# Algorithmic Trading Research

Backtesting strategies (Backtrader), risk-aware controls (DeepScalper-inspired), RL baselines, and LLM-driven sentiment research in a single workspace.

## Features
- Backtrader strategies
  - Flawless Victory (v1/v2/v3) with Bollinger/RSI/MFI and SL/TP: [`flawless_victory.FlawlessVictory`](scripts/backtrader/flawless_victory.py)
  - Risk-aware variant with volatility-based sizing, drawdown brake, downside deviation gate: [`fv_risk_aware.FlawlessVictory`](scripts/backtrader/fv_risk_aware.py)
  - Smoke test using synthetic OHLCV: [scripts/backtrader/test.py](scripts/backtrader/test.py)
- Reinforcement Learning (DeepScalper-style)
  - Gymnasium environment and BDQ-style scaffolding: [`gym_implementation.TradingEnvGym`](scripts/rl/gym_implementation.py)
  - Pretrained checkpoints: [models/bdq_test.pt](models/bdq_test.pt), [models/test_bdq.pt](models/test_bdq.pt)
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
- Live trading config helpers: [scripts/live/config.py](scripts/live/config.py)

### Live Paper Trading (Alpaca - FlawlessVictoryRiskAware)

Run the risk-aware Flawless Victory strategy against Alpaca paper trading:
```sh
export ALPACA_API_KEY=your_key
export ALPACA_API_SECRET=your_secret
# (Optional) export ALPACA_BASE_URL=https://paper-api.alpaca.markets
python -m scripts.live.live_paper_trading --symbol AAPL --version v1 --verbose-risk
```
Flags:
* `--version v1|v2|v3` select strategy logic
* `--cash-fraction 0.8` capital allocation per trade
* `--target-daily-vol 0.015` realized vol target for scaling
* `--max-drawdown 0.15` drawdown threshold gating new entries
* `--dd-cooldown 3` cooldown bars after breach

Security: Never commit raw API keys. Use environment variables or a local `.env` (Lumibot auto-loads). The repository deliberately does NOT store credentials.

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

### DeepScalper-Style Intraday Agent (Branching Dueling Q)

Implemented under `scripts/deepscalper/` with:
* Environment: [`env.DeepScalperEnv`](scripts/deepscalper/env.py) (minute bars, per‑day episodes, volatility aux target, hindsight reward bonus)
* Model: [`model.BranchingDuelingQNet`](scripts/deepscalper/model.py) (shared trunk + price and quantity branches + aux volatility head)
* Training loop: [`train.train_agent`](scripts/deepscalper/train.py)
* CLI Entrypoint: [`main.py`](scripts/deepscalper/main.py)
* Lumibot integration / backtest: [`run_lumibot_example.py`](scripts/deepscalper/run_lumibot_example.py)

#### 1. Train a Model
Downloads recent 1m data for the symbol (default AAPL) via `yfinance` and starts/continues training (resumes if `last.pt` present):
```sh
python -m scripts.deepscalper.main --symbol AAPL --steps 50000
```
Optional flags:
* `--start 2025-01-01 --end 2025-01-15` limit data range
* `--no-resume` start fresh even if checkpoints exist
* `--ckpt path/to/ckpts` custom checkpoint directory
* `--save-every 5000` adjust checkpoint save frequency

Checkpoint directory contents after/while training:
```
checkpoints/
  bdq_5000.pt          # Rolling historical snapshots (step number)
  bdq_10000.pt
  last.pt              # Always the most recent checkpoint
  manifest.json        # Inference-critical metadata
  baseline_metrics.json (optional; produced by baseline_report)
```

`manifest.json` includes: `price_bins`, `qty_bins`, `lookback`, feature ordering, the environment & train configs, and a SHA256 hash for integrity.

#### 2. Generate a Baseline Report
Runs a quick greedy-policy evaluation over a few random episodes and stores summary stats:
```sh
python -m scripts.deepscalper.baseline_report --ckpt_dir scripts/deepscalper/checkpoints --episodes 5
```
Output: `baseline_metrics.json` (avg_return, stdev, min/max, wall time, raw episode returns).

#### 3. Run a Lumibot Backtest with the Learned Policy
Uses Yahoo minute data (auto picks latest checkpoint; respects `manifest.json` if present):
```sh
python -m scripts.deepscalper.run_lumibot_example
```
Environment variable overrides:
```sh
export DS_CKPT_DIR=/absolute/path/to/another/checkpoint_dir
python -m scripts.deepscalper.run_lumibot_example
```

#### 4. Inference Mechanics
* Discrete action = (price_bin, qty_bin)
* `price_bin` maps to a relative price offset in [-1, +1] scaled by a multiple of recent ATR-derived tick size.
* `qty_bin` splits into side (>= midpoint ⇒ long, else short) and size fraction (distance from midpoint).
* Environment simulates limit-like fill if price trades inside the bar's high/low; otherwise applies a small slippage penalty.
* Auxiliary head predicts short-horizon realized volatility; its target is computed from recent log-return variance.

#### 5. Resume Training
Simply rerun the training command; it detects `last.pt` and continues from stored step:
```sh
python -m scripts.deepscalper.main --symbol AAPL --steps 200000
```
(If you previously ran 50k steps and now ask for 200k, it will aim for cumulative 200k.)

#### 6. Consistency / Manifest Checks
* During each checkpoint save the manifest is (re)written.
* Downstream consumers (Lumibot script) load bins/lookback from the manifest to avoid mismatches.

#### 7. Upcoming Enhancements (Roadmap Snippet)
Planned increments: prioritized replay tuning, distributional heads, regime-aware evaluation, multi-symbol data cache, and drift detection.

See full DeepScalper research summary in [TXTs/DeepScalper.txt](TXTs/DeepScalper.txt).

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
- [PDFs](PDFs): Paper PDFs

## Troubleshooting
- PandasData AttributeError: tuple has no attribute lower
  - Ensure columns are flattened and named Open/High/Low/Close/Volume/OpenInterest (see `_ensure_bt_ohlcv` in [scripts/backtrader/test.py](scripts/backtrader/test.py)).

## Disclaimer
This repository is for research only. Nothing herein is financial advice. Use at your