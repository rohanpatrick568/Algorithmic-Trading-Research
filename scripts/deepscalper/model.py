from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BranchingDuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, price_bins: int, qty_bins: int):
        super().__init__()
        self.trunk = MLP(obs_dim, 256)
        self.value = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        # advantages for branches
        self.adv_price = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, price_bins))
        self.adv_qty = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, qty_bins))
        # risk-aware auxiliary head: predict next-horizon realized volatility
        self.aux_vol = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        v = self.value(z)  # [B,1]
        ap = self.adv_price(z)  # [B,P]
        aq = self.adv_qty(z)    # [B,Q]
        # center advantages
        ap = ap - ap.mean(dim=1, keepdim=True)
        aq = aq - aq.mean(dim=1, keepdim=True)
        qp = v + ap  # [B,P]
        qq = v + aq  # [B,Q]
        aux = self.aux_vol(z)  # [B,1]
        return qp, qq, v, aux.squeeze(-1)


@torch.no_grad()
def act_epsilon_greedy(qp: torch.Tensor, qq: torch.Tensor, eps: float):
    B = qp.size(0)
    ap = qp.argmax(dim=1)
    aq = qq.argmax(dim=1)
    if eps > 0:
        mask = torch.rand(B, device=qp.device) < eps
        rand_p = torch.randint(0, qp.size(1), (B,), device=qp.device)
        rand_q = torch.randint(0, qq.size(1), (B,), device=qp.device)
        ap = torch.where(mask, rand_p, ap)
        aq = torch.where(mask, rand_q, aq)
    return ap, aq
