from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque

import numpy as np
import torch


@dataclass
class Transition:
    s: np.ndarray
    a_price: int
    a_qty: int
    r: float
    s2: np.ndarray
    done: bool
    aux_target: float
    weight: float = 1.0
    idx: int = -1


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.priorities: Deque[float] = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, tr: Transition, priority: float | None = None):
        self.buffer.append(tr)
        if priority is None:
            priority = max(self.priorities, default=1.0)
        self.priorities.append(priority)

    def sample(self, batch_size: int):
        probs = np.array(self.priorities, dtype=np.float64)
        probs = probs ** self.alpha
        probs = probs / probs.sum()
        idxs = np.random.choice(len(self.buffer), batch_size, p=probs)

        samples = [self.buffer[i] for i in idxs]
        weights = (len(self.buffer) * probs[idxs]) ** (-self.beta)
        weights = weights / weights.max()
        for i, w in zip(idxs, weights):
            # annotate
            pass
        batch = Transition(
            s=np.stack([t.s for t in samples]),
            a_price=np.array([t.a_price for t in samples]),
            a_qty=np.array([t.a_qty for t in samples]),
            r=np.array([t.r for t in samples], dtype=np.float32),
            s2=np.stack([t.s2 for t in samples]),
            done=np.array([t.done for t in samples], dtype=np.float32),
            aux_target=np.array([t.aux_target for t in samples], dtype=np.float32),
            weight=weights.astype(np.float32),
            idx=-1,
        )
        return batch, idxs

    def update_priorities(self, idxs, td_errors):
        for i, e in zip(idxs, td_errors):
            self.priorities[i] = float(abs(e) + 1e-5)
