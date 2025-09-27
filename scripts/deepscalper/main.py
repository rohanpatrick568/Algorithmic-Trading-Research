from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

"""CLI entrypoint for training the DeepScalper agent.

Supports running either as a module:
	python -m scripts.deepscalper.main --symbol AAPL

Or directly:
	python scripts/deepscalper/main.py --symbol AAPL

The try/except import block below allows both invocation styles by
falling back to local (non-relative) imports when __package__ is empty.
"""

try:  # Normal package-relative imports
	from .config import EnvConfig, TrainConfig  # type: ignore
	from .data import load_minute_data  # type: ignore
	from .env import DeepScalperEnv  # type: ignore
	from .train import train_agent  # type: ignore
except Exception:  # Fallback when executed directly (no parent package)
	from config import EnvConfig, TrainConfig  # type: ignore
	from data import load_minute_data  # type: ignore
	from env import DeepScalperEnv  # type: ignore
	from train import train_agent  # type: ignore


def parse_args():
	p = argparse.ArgumentParser(description="DeepScalper RL Training")
	p.add_argument("--symbol", default="AAPL")
	p.add_argument("--start", default=(datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d"))
	p.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
	p.add_argument("--steps", type=int, default=50_000)
	# Default checkpoints directory (relative to this file) instead of a user-specific Windows path.
	default_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints")
	p.add_argument("--ckpt", default=default_ckpt)
	p.add_argument("--no-resume", action="store_true", help="Start fresh and ignore existing checkpoints")
	p.add_argument("--save-every", type=int, default=2000, help="Checkpoint save frequency in steps")
	return p.parse_args()


def main():
	args = parse_args()
	env_cfg = EnvConfig(symbol=args.symbol)
	train_cfg = TrainConfig(train_steps=args.steps)

	md = load_minute_data(args.symbol, args.start, args.end)
	env = DeepScalperEnv(md, env_cfg, train_cfg)

	os.makedirs(args.ckpt, exist_ok=True)
	train_agent(env, train_cfg, ckpt_dir=args.ckpt, resume=(not args.no_resume), save_every=args.save_every)


if __name__ == "__main__":
	main()

