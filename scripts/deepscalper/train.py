from __future__ import annotations

import os, json, hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:  # package imports
    from .config import EnvConfig, TrainConfig  # type: ignore
    from .env import DeepScalperEnv  # type: ignore
    from .model import BranchingDuelingQNet, act_epsilon_greedy  # type: ignore
    from .replay import PrioritizedReplay, Transition  # type: ignore
except Exception:  # fallback
    from config import EnvConfig, TrainConfig  # type: ignore
    from env import DeepScalperEnv  # type: ignore
    from model import BranchingDuelingQNet, act_epsilon_greedy  # type: ignore
    from replay import PrioritizedReplay, Transition  # type: ignore


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


def linear_schedule(start: float, end: float, step: int, total: int) -> float:
    t = min(step / total, 1.0)
    return start + t * (end - start)


def _ckpt_paths(ckpt_dir: str) -> tuple[Path, list[Path]]:
    d = Path(ckpt_dir) if ckpt_dir else Path(__file__).parent / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    series = sorted(d.glob("bdq_*.pt"), key=lambda p: p.stat().st_mtime)
    last = d / "last.pt"
    return last, series


def _load_latest(ckpt_dir: str, model: nn.Module, opt: optim.Optimizer, map_location="cpu") -> int:
    last, series = _ckpt_paths(ckpt_dir)
    path = last if last.exists() else (series[-1] if series else None)
    if not path or not path.exists():
        return 0
    try:
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])  # type: ignore[arg-type]
            if "opt" in state:
                opt.load_state_dict(state["opt"])  # type: ignore[arg-type]
            return int(state.get("step", 0))
        # raw state dict fallback
        model.load_state_dict(state)  # type: ignore[arg-type]
        return 0
    except Exception:
        return 0


def _write_manifest(ckpt_dir: str, env: DeepScalperEnv, tcfg: TrainConfig):
    """Persist a lightweight manifest describing inference-critical parameters.

    Includes:
      - price_bins / qty_bins
      - lookback
      - feature ordering
      - env / train config hashes (for traceability)
    """
    if not ckpt_dir:
        return
    path = Path(ckpt_dir) / "manifest.json"
    feat_order = list(getattr(env, "feat_cols", []))

    def _convert(obj):  # recursively ensure JSON-serializable primitives
        import numpy as _np
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        # dataclass or other object with __dict__
        if hasattr(obj, "__dict__"):
            return _convert(vars(obj))
        return str(obj)

    payload = {
        "price_bins": int(env.action_space.nvec[0]),
        "qty_bins": int(env.action_space.nvec[1]),
        "lookback": int(env.lookback),
        "features": feat_order,
        "env_cfg": _convert(getattr(env, "env_cfg", None)),
        "train_cfg": _convert(asdict(tcfg)),
    }
    # Add quick integrity hashes
    h_src = json.dumps(payload, sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(h_src).hexdigest()
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def _save_ckpt(ckpt_dir: str, step: int, model: nn.Module, opt: optim.Optimizer, tcfg: TrainConfig, env: DeepScalperEnv | None = None):
    if not ckpt_dir:
        return
    d_last, _ = _ckpt_paths(ckpt_dir)
    payload = {"model": model.state_dict(), "opt": opt.state_dict(), "cfg": asdict(tcfg), "step": step}
    # rolling named and a stable last.pt pointer
    step_path = d_last.parent / f"bdq_{step}.pt"
    try:
        torch.save(payload, step_path)
    except Exception:
        pass
    try:
        torch.save(payload, d_last)
    except Exception:
        pass
    # Write manifest once (or overwrite) if env provided
    if env is not None:
        _write_manifest(str(d_last.parent), env, tcfg)


def train_agent(env: DeepScalperEnv, tcfg: TrainConfig, ckpt_dir: str = "", *, resume: bool = True, save_every: int = 2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.obs_dim
    price_bins = env.action_space.nvec[0]
    qty_bins = env.action_space.nvec[1]

    online = BranchingDuelingQNet(obs_dim, price_bins, qty_bins).to(device)
    target = BranchingDuelingQNet(obs_dim, price_bins, qty_bins).to(device)
    target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=tcfg.lr)

    buffer = PrioritizedReplay(tcfg.buffer_size)

    obs, _ = env.reset()
    # Resume if requested
    step = 0
    if resume and ckpt_dir:
        step_loaded = _load_latest(ckpt_dir, online, opt, map_location=device)
        if step_loaded > 0:
            target.load_state_dict(online.state_dict())
            step = int(step_loaded)
            print(f"[DeepScalper] Resumed from step {step} in {ckpt_dir}")
    target_total = step + tcfg.train_steps
    episode_return = 0.0
    best_eval = -1e9

    while step < target_total:
        eps = linear_schedule(tcfg.eps_start, tcfg.eps_end, step, tcfg.eps_decay_steps)
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            qp, qq, _, _ = online(o)
            ap, aq = act_epsilon_greedy(qp, qq, eps)
            action = np.array([int(ap.item()), int(aq.item())], dtype=np.int64)

        obs2, r, done, trunc, info = env.step(action)
        tr = Transition(
            s=obs.astype(np.float32),
            a_price=int(action[0]),
            a_qty=int(action[1]),
            r=float(r),
            s2=obs2.astype(np.float32),
            done=bool(done),
            aux_target=float(info.get("aux_target", 0.0)),
        )
        buffer.push(tr)
        obs = obs2
        episode_return += float(info.get("r_raw", r))
        step += 1

        if done or trunc:
            print(f"Episode finished: step={step} return={episode_return:,.2f} cash={info.get('cash'):,.2f}")
            obs, _ = env.reset()
            episode_return = 0.0

        # learn
        if step > tcfg.warmup_steps and len(buffer) >= tcfg.batch_size and step % tcfg.update_every == 0:
            batch, idxs = buffer.sample(tcfg.batch_size)
            s = torch.tensor(batch.s, dtype=torch.float32, device=device)
            s2 = torch.tensor(batch.s2, dtype=torch.float32, device=device)
            a_price = torch.tensor(batch.a_price, dtype=torch.int64, device=device)
            a_qty = torch.tensor(batch.a_qty, dtype=torch.int64, device=device)
            r = torch.tensor(batch.r, dtype=torch.float32, device=device)
            done = torch.tensor(batch.done, dtype=torch.float32, device=device)
            w = torch.tensor(batch.weight, dtype=torch.float32, device=device)
            aux_t = torch.tensor(batch.aux_target, dtype=torch.float32, device=device)

            qp, qq, _, aux = online(s)
            with torch.no_grad():
                qp2, qq2, _, _ = target(s2)
                max_qp = qp2.max(dim=1).values
                max_qq = qq2.max(dim=1).values
                # branch TD targets added equally
                y_p = r + tcfg.gamma * (1 - done) * max_qp
                y_q = r + tcfg.gamma * (1 - done) * max_qq

            q_ap = qp.gather(1, a_price.view(-1, 1)).squeeze(1)
            q_aq = qq.gather(1, a_qty.view(-1, 1)).squeeze(1)

            td_p = y_p - q_ap
            td_q = y_q - q_aq
            q_loss = ((td_p.pow(2) + td_q.pow(2)) * 0.5 * w).mean()

            aux_loss = F.mse_loss(aux, aux_t)
            loss = q_loss + tcfg.aux_weight * aux_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=tcfg.grad_clip)
            opt.step()

            # priorities
            td_err = (td_p.abs() + td_q.abs()).detach().cpu().numpy()
            buffer.update_priorities(idxs, td_err)

            soft_update(target, online, tcfg.target_update_tau)

        if step % 10_000 == 0 and step > 0:
            eval_ret = evaluate(env, online, device, episodes=3)
            print(f"Eval@{step}: avg_return={eval_ret:,.2f}")
            _save_ckpt(ckpt_dir, step, online, opt, tcfg, env)

        if save_every and step % save_every == 0 and step > 0:
            _save_ckpt(ckpt_dir, step, online, opt, tcfg, env)

    # Final save
    _save_ckpt(ckpt_dir, step, online, opt, tcfg, env)


@torch.no_grad()
def evaluate(env: DeepScalperEnv, model: BranchingDuelingQNet, device, episodes=1):
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            qp, qq, _, _ = model(o)
            ap = qp.argmax(dim=1).item()
            aq = qq.argmax(dim=1).item()
            obs, r, done, trunc, info = env.step(np.array([ap, aq], dtype=np.int64))
            ret += info.get("r_raw", r)
        total += ret
    return total / episodes
