from __future__ import annotations
"""Generate a quick baseline metrics JSON for current model/environment.

Usage: python -m scripts.deepscalper.baseline_report --ckpt_dir path/to/checkpoints --steps 5000

Produces baseline_metrics.json inside the checkpoint dir.
"""
import argparse, json, time, statistics, os, pathlib
import numpy as np
import torch

from .config import EnvConfig, TrainConfig
from .data import MarketData
from .env import DeepScalperEnv
from .model import BranchingDuelingQNet
from .train import evaluate, _load_latest


def load_env(env_cfg: EnvConfig, tcfg: TrainConfig):
    md = MarketData(env_cfg.symbol, env_cfg.timeframe)
    env = DeepScalperEnv(md, env_cfg, tcfg)
    return env


def quick_eval(env: DeepScalperEnv, ckpt_dir: str, episodes: int = 3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BranchingDuelingQNet(env.obs_dim, env.action_space.nvec[0], env.action_space.nvec[1]).to(device)
    _ = _load_latest(ckpt_dir, model, torch.optim.Adam(model.parameters(), lr=1e-3), map_location=device)
    rets = []
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                qp, qq, _, _ = model(o)
                ap = qp.argmax(dim=1).item()
                aq = qq.argmax(dim=1).item()
                obs, r, done, trunc, info = env.step(np.array([ap, aq]))
                ep_ret += info.get('r_raw', r)
            rets.append(ep_ret)
    return rets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', type=str, default=str(pathlib.Path(__file__).parent / 'checkpoints'))
    ap.add_argument('--episodes', type=int, default=3)
    args = ap.parse_args()

    env_cfg = EnvConfig()
    tcfg = TrainConfig(train_steps=1)  # we only need config values
    env = load_env(env_cfg, tcfg)

    t0 = time.time()
    rets = quick_eval(env, args.ckpt_dir, episodes=args.episodes)
    dt = time.time() - t0

    metrics = {
        'episodes': len(rets),
        'avg_return': float(sum(rets)/max(len(rets),1)),
        'stdev_return': float(statistics.pstdev(rets)) if len(rets) > 1 else 0.0,
        'min_return': float(min(rets)) if rets else 0.0,
        'max_return': float(max(rets)) if rets else 0.0,
        'returns': rets,
        'wall_time_sec': dt,
    }

    out_path = os.path.join(args.ckpt_dir, 'baseline_metrics.json')
    try:
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Baseline metrics written to {out_path}")
    except Exception as e:
        print('Failed to write metrics:', e)


if __name__ == '__main__':
    main()
