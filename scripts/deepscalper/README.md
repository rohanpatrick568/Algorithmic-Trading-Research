# DeepScalper (RL) – Intraday 1m Trading

Implements core ideas from DeepScalper: Branching Dueling Q-Network (BDQ) with action branching (price, quantity), a hindsight bonus, a risk-aware auxiliary head (realized volatility), and a Lumibot strategy wrapper for inference.

## What’s inside
- Gymnasium env: `DeepScalperEnv` (1‑minute bars, OHLCV+indicators, limit-like fills)
- PyTorch model: BDQ (shared trunk, value head, per-branch advantages) + auxiliary volatility head
- Prioritized replay buffer
- Trainer with soft target updates, epsilon-greedy, rolling checkpoints, and resume support
- Lumibot wrapper for backtests/live using a trained checkpoint

Project layout (relative to repo root):
- `scripts/deepscalper/` – code, CLI, checkpoints
	- `main.py` – CLI for training
	- `train.py` – training loop (resume + save)
	- `env.py` – Gym environment (1m episodes)
	- `model.py` – BDQ + aux head
	- `data.py` – data loader (yfinance 1m) + indicators
	- `lumibot_strategy.py` – Lumibot strategy wrapper
	- `run_lumibot_example.py` – small backtest example
	- `checkpoints/` – saved models (created on first run)

## Setup (Windows / PowerShell)
Create/activate your venv, then install deps:

```powershell
# From repo root
& "env/Scripts/Activate.ps1"  # if not already active

pip install --upgrade pip
pip install torch gymnasium yfinance pandas numpy lumibot
```

If torch with CUDA is desired, install a CUDA-enabled build from pytorch.org.

## Train (with resume by default)
The trainer saves rolling checkpoints to `scripts/deepscalper/checkpoints` and resumes from the latest `last.pt` automatically.

```powershell
# Short smoke test (module execution)
python -m scripts.deepscalper.main --symbol AAPL --steps 1000 --save-every 250

# Longer run (module execution)
python -m scripts.deepscalper.main --symbol AAPL --steps 50000 --save-every 2000

# Alternative: direct script execution
python scripts/deepscalper/main.py --symbol AAPL --steps 1000 --save-every 250
```

Flags:
- `--steps`       Number of additional steps to train. If a checkpoint exists, total steps = loaded_step + steps.
- `--ckpt`        Checkpoint directory (default: `scripts/deepscalper/checkpoints`).
- `--save-every`  Save frequency in steps (also saves `last.pt`).
- `--no-resume`   Start fresh (ignore existing checkpoints).
- `--start/--end` Date range for yfinance 1m (defaults to last 7 days). Note yfinance 1m allows ≤ 8 days per request.

Checkpoint files:
- `checkpoints/last.pt` – latest pointer (model + optimizer + meta).
- `checkpoints/bdq_<step>.pt` – rolling snapshots.

## Use the model in Lumibot
We provide `run_lumibot_example.py` that loads the most recent `bdq_*.pt` (or `ckpt_test.pth` fallback) and runs a short backtest via Yahoo.

Yahoo’s 1m API allows ≤ 8 days per request. The example uses a 5‑day window by default. You can tweak `symbol`, `start`, and `end` in the file.

Run the example (use your venv python):

```powershell
& "C:/Users/patri/Desktop/Algorithmic Trading Research/env/Scripts/python.exe" "scripts/deepscalper/run_lumibot_example.py"
```

If you see “possibly delisted; no price data found” or “Only 8 days worth of 1m granularity…”, shorten the window (e.g., 3–5 days).

Integrate in your own Lumibot strategy:
- Instantiate `DeepScalperLumibotStrategy` and pass `inference_cfg=InferenceConfig(model_path=..., price_bins=21, qty_bins=11, lookback=120)`.
- The wrapper computes features from broker bars and places limit orders per model output each minute.

## Troubleshooting
- Missing `gymnasium`: install it in your venv – `pip install gymnasium`.
- Yahoo 1m data error (≤ 8 days): reduce the date range (example already uses 5 days).
- Relative import error running scripts directly: Both `main.py` and `run_lumibot_example.py` include fallback imports to support direct script execution.
- No module named `lumibot`: install it in the active venv – `pip install lumibot` (then re-run with the same venv python).
- Checkpoint not loading: ensure files exist under `scripts/deepscalper/checkpoints`. Trainer logs `[DeepScalper] Resumed from step ...` when resume succeeds.

## Notes
- Episodes are per day at 1‑minute granularity; positions auto-close at day end.
- Hindsight reward bonus applies during training only (not inference).
- Auxiliary head predicts short‑horizon realized volatility; its loss is weighted during training.

## Try it quickly
```powershell
# 1) Train a little (creates checkpoints and verifies the loop)
python -m scripts.deepscalper.main --symbol AAPL --steps 1500 --save-every 500

# 2) Run the Lumibot backtest example (5-day window)
& "C:/Users/patri/Desktop/Algorithmic Trading Research/env/Scripts/python.exe" "scripts/deepscalper/run_lumibot_example.py"
```