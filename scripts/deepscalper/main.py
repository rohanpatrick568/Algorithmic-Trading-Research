from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

# Support running as a module or as a script
try:
    from .config import EnvConfig, TrainConfig
    from .data import load_minute_data
    from .env import DeepScalperEnv
    from .train import train_agent
except ImportError:
    import importlib, pathlib, sys
    root = str(pathlib.Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.append(root)
    
    config_mod = importlib.import_module("scripts.deepscalper.config")
    EnvConfig = getattr(config_mod, "EnvConfig")
    TrainConfig = getattr(config_mod, "TrainConfig")
    
    data_mod = importlib.import_module("scripts.deepscalper.data")
    load_minute_data = getattr(data_mod, "load_minute_data")
    
    env_mod = importlib.import_module("scripts.deepscalper.env")
    DeepScalperEnv = getattr(env_mod, "DeepScalperEnv")
    
    train_mod = importlib.import_module("scripts.deepscalper.train")
    train_agent = getattr(train_mod, "train_agent")


def parse_args():
	p = argparse.ArgumentParser(description="DeepScalper RL Training")
	p.add_argument("--symbol", default="AAPL")
	p.add_argument("--start", default=(datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d"))
	p.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
	p.add_argument("--steps", type=int, default=50_000)
	p.add_argument("--ckpt", default="c:/Users/patri/Desktop/Algorithmic Trading Research/scripts/deepscalper/checkpoints")
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

