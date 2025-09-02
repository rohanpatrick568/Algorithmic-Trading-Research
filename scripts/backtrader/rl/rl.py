import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Optional market data
try:
	import yfinance as yf  # type: ignore
except Exception:
	yf = None


# ---------------------------- Data ----------------------------
def fetch_prices(symbol: str, start: Optional[str], end: Optional[str]) -> pd.Series:
	"""Return daily close prices for [start, end]. Fallback to synthetic if yfinance unavailable."""
	if yf is not None:
		try:
			df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
			if isinstance(df, pd.DataFrame) and not df.empty and 'Close' in df.columns:
				s = df['Close'].dropna()
				if getattr(s.index, 'tz', None) is not None:
					s.index = s.index.tz_localize(None)
				return s
		except Exception:
			pass
	# Synthetic fallback: geometric random walk
	n = 1000
	rng = np.random.default_rng(42)
	rets = rng.normal(loc=0.0005, scale=0.02, size=n)
	prices = 100 * np.exp(np.cumsum(rets))
	idx = pd.date_range('2020-01-01', periods=n, freq='B')
	return pd.Series(prices, index=idx, name='Close')


def series_to_returns(prices: pd.Series) -> pd.Series:
	r = prices.pct_change().fillna(0.0).clip(-0.2, 0.2)
	return r


# ---------------------------- Simple Trading Env ----------------------------
@dataclass
class EnvConfig:
	window: int = 32
	fee_bps: float = 5.0  # transaction fee in bps per notional change
	max_steps: Optional[int] = None  # None => until series end


class TradingEnv:
	"""Minimal environment over a return series.

	State: last `window` returns (float32 vector)
	Action (branched):
	  - branch 0 (direction): 3 choices => {-1, 0, +1}
	  - branch 1 (size): 5 choices => {0.0, 0.25, 0.5, 0.75, 1.0}

	Reward: position_value * next_return - transaction_cost
	Position_value = direction * size_fraction
	"""

	def __init__(self, returns: pd.Series, cfg: EnvConfig):
		arr = returns.astype(np.float32).to_numpy()
		if isinstance(arr, np.ndarray) and arr.ndim > 1:
			arr = np.squeeze(arr)
		self.r = arr
		self.cfg = cfg
		self.t = 0
		self.pos = 0.0  # current position exposure in [-1, 1]
		self.done = False

	@property
	def n_branches(self) -> int:
		return 2

	@property
	def action_sizes(self) -> List[int]:
		return [3, 5]

	def reset(self) -> np.ndarray:
		self.t = max(self.cfg.window, 1)
		self.pos = 0.0
		self.done = False
		return self._obs()

	def _obs(self) -> np.ndarray:
		w = self.cfg.window
		x = self.r[self.t - w:self.t]
		if x.shape[0] < w:
			pad = np.zeros(w, dtype=np.float32)
			pad[-x.shape[0]:] = x
			x = pad
		return np.asarray(x, dtype=np.float32).reshape(-1)

	def step(self, action_branches: List[int]) -> Tuple[np.ndarray, float, bool, dict]:
		if self.done:
			return self._obs(), 0.0, True, {}

		dir_map = [-1.0, 0.0, +1.0]
		size_map = [0.0, 0.25, 0.5, 0.75, 1.0]
		direction = dir_map[action_branches[0]]
		size = size_map[action_branches[1]]
		target_pos = direction * size

		# Transaction cost on position change
		dpos = abs(target_pos - self.pos)
		fee = dpos * (self.cfg.fee_bps / 1e4)

		# Next step reward uses next return
		if self.t >= len(self.r) - 1:
			self.done = True
			return self._obs(), 0.0, True, {}

		rn = self.r[self.t]
		ret_next = float(rn if np.isscalar(rn) else np.asarray(rn).reshape(-1)[0])
		reward = target_pos * ret_next - fee

		# Move to next time
		self.pos = target_pos
		self.t += 1

		# Episode end condition
		if self.cfg.max_steps is not None:
			self.done = (self.t >= self.cfg.window + self.cfg.max_steps)
		else:
			self.done = (self.t >= len(self.r) - 1)

		return self._obs(), float(reward), self.done, {}


# ---------------------------- Branching Dueling Q-Network ----------------------------
class BDQ(nn.Module):
	def __init__(self, state_dim: int, branch_sizes: List[int], hidden: int = 128):
		super().__init__()
		self.branches = branch_sizes
		self.feature = nn.Sequential(
			nn.Linear(state_dim, hidden), nn.ReLU(),
			nn.Linear(hidden, hidden), nn.ReLU(),
		)
		self.value = nn.Linear(hidden, 1)
		self.advantages = nn.ModuleList([nn.Linear(hidden, n) for n in branch_sizes])

	def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
		h = self.feature(x)
		v = self.value(h)
		qs: List[torch.Tensor] = []
		for head in self.advantages:
			a = head(h)
			a = a - a.mean(dim=1, keepdim=True)
			q = v + a
			qs.append(q)
		return qs

	def q_for_actions(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
		"""actions shape: [B, n_branches] with discrete indices."""
		qs = self.forward(x)
		# Gather selected Q per branch and sum
		total = 0.0
		for b, q in enumerate(qs):
			idx = actions[:, b].long().unsqueeze(1)
			qb = q.gather(1, idx).squeeze(1)
			total = qb if b == 0 else (total + qb)
		return total

	def greedy_actions(self, x: torch.Tensor) -> torch.Tensor:
		qs = self.forward(x)
		acts = [q.argmax(dim=1) for q in qs]
		return torch.stack(acts, dim=1)


# ---------------------------- Replay Buffer ----------------------------
class Replay:
	def __init__(self, capacity: int, state_dim: int, n_branches: int):
		self.capacity = capacity
		self.state = np.zeros((capacity, state_dim), dtype=np.float32)
		self.action = np.zeros((capacity, n_branches), dtype=np.int64)
		self.reward = np.zeros((capacity,), dtype=np.float32)
		self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
		self.done = np.zeros((capacity,), dtype=np.float32)
		self.idx = 0
		self.full = False

	def push(self, s, a, r, ns, d):
		i = self.idx
		self.state[i] = np.asarray(s, dtype=np.float32).reshape(-1)
		self.action[i] = a
		self.reward[i] = r
		self.next_state[i] = np.asarray(ns, dtype=np.float32).reshape(-1)
		self.done[i] = d
		self.idx = (self.idx + 1) % self.capacity
		if self.idx == 0:
			self.full = True

	def sample(self, batch: int):
		n = self.capacity if self.full else self.idx
		idxs = np.random.randint(0, n, size=batch)
		return (
			torch.from_numpy(self.state[idxs]),
			torch.from_numpy(self.action[idxs]),
			torch.from_numpy(self.reward[idxs]),
			torch.from_numpy(self.next_state[idxs]),
			torch.from_numpy(self.done[idxs]),
		)

	def __len__(self):
		return self.capacity if self.full else self.idx


# ---------------------------- Training ----------------------------
@dataclass
class TrainConfig:
	symbol: str = 'AAPL'
	start: Optional[str] = '2015-01-01'
	end: Optional[str] = '2022-12-31'
	window: int = 32
	episodes: int = 1
	max_steps: Optional[int] = None
	gamma: float = 0.99
	lr: float = 1e-3
	batch: int = 64
	replay: int = 100_000
	warmup: int = 1_000
	target_sync: int = 500
	steps_per_update: int = 1
	epsilon_start: float = 1.0
	epsilon_final: float = 0.05
	epsilon_decay_steps: int = 20_000
	seed: int = 42


def linear_epsilon(step: int, cfg: TrainConfig) -> float:
	t = min(1.0, step / max(1, cfg.epsilon_decay_steps))
	return cfg.epsilon_start + t * (cfg.epsilon_final - cfg.epsilon_start)


def train(cfg: TrainConfig):
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	prices = fetch_prices(cfg.symbol, cfg.start, cfg.end)
	returns = series_to_returns(prices)

	env = TradingEnv(returns, EnvConfig(window=cfg.window, max_steps=cfg.max_steps))
	state_dim = cfg.window
	branch_sizes = env.action_sizes

	device = torch.device('cpu')
	online = BDQ(state_dim, branch_sizes).to(device)
	target = BDQ(state_dim, branch_sizes).to(device)
	target.load_state_dict(online.state_dict())
	opt = optim.Adam(online.parameters(), lr=cfg.lr)

	rb = Replay(cfg.replay, state_dim, len(branch_sizes))

	global_step = 0
	for ep in range(cfg.episodes):
		s = env.reset()
		ep_reward = 0.0
		steps = 0

		while True:
			eps = linear_epsilon(global_step, cfg)
			if np.random.rand() < eps:
				a = np.array([np.random.randint(n) for n in branch_sizes], dtype=np.int64)
			else:
				with torch.no_grad():
					sa = torch.from_numpy(s).unsqueeze(0).to(device)
					a_t = online.greedy_actions(sa).cpu().numpy()[0]
					a = a_t.astype(np.int64)

			ns, r, d, _ = env.step(a.tolist())
			rb.push(s, a, r, ns, float(d))
			s = ns
			ep_reward += r
			steps += 1
			global_step += 1

			# Learn
			if len(rb) >= cfg.warmup and (global_step % cfg.steps_per_update == 0):
				bs, ba, br, bns, bd = rb.sample(cfg.batch)
				bs = bs.to(device).float()
				ba = ba.to(device).long()
				br = br.to(device).float()
				bns = bns.to(device).float()
				bd = bd.to(device).float()

				with torch.no_grad():
					# Branch-wise greedy for next state, sum maxes (branching approximation)
					next_qs = target.forward(bns)
					q_next_sum = 0.0
					for q in next_qs:
						q_next_sum = q.max(dim=1).values if isinstance(q_next_sum, float) else (q_next_sum + q.max(dim=1).values)
					tgt = br + (1.0 - bd) * cfg.gamma * q_next_sum

				pred = online.q_for_actions(bs, ba)
				loss = nn.MSELoss()(pred, tgt)
				opt.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(online.parameters(), max_norm=1.0)
				opt.step()

				if global_step % 1000 == 0:
					print(f"step {global_step} loss {loss.item():.6f} eps {eps:.3f} reward_ep {ep_reward:.4f}")

			if global_step % cfg.target_sync == 0:
				target.load_state_dict(online.state_dict())

			if d:
				print(f"episode {ep+1} steps {steps} reward {ep_reward:.4f}")
				break


def main():
	p = argparse.ArgumentParser(description='Branching Dueling Q-Network (DeepScalper-style)')
	p.add_argument('--symbol', default='AAPL')
	p.add_argument('--start', default='2015-01-01')
	p.add_argument('--end', default='2022-12-31')
	p.add_argument('--window', type=int, default=32)
	p.add_argument('--episodes', type=int, default=1)
	p.add_argument('--max-steps', type=int, default=0)
	p.add_argument('--gamma', type=float, default=0.99)
	p.add_argument('--lr', type=float, default=1e-3)
	p.add_argument('--batch', type=int, default=64)
	p.add_argument('--replay', type=int, default=100_000)
	p.add_argument('--warmup', type=int, default=1_000)
	p.add_argument('--target-sync', type=int, default=500)
	p.add_argument('--epsilon-decay-steps', type=int, default=20_000)
	args = p.parse_args()

	cfg = TrainConfig(
		symbol=args.symbol,
		start=args.start,
		end=args.end,
		window=args.window,
		episodes=args.episodes,
		max_steps=(None if args.max_steps == 0 else args.max_steps),
		gamma=args.gamma,
		lr=args.lr,
		batch=args.batch,
		replay=args.replay,
		warmup=args.warmup,
		target_sync=args.target_sync,
		epsilon_decay_steps=args.epsilon_decay_steps,
	)
	train(cfg)


if __name__ == '__main__':
	main()

