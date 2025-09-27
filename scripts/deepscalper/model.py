from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

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


class DistributionalHead(nn.Module):
    """C51 distributional head for modeling value distribution"""
    def __init__(self, input_dim: int, action_dim: int, num_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Create value support
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Distributional head
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * num_atoms)
        )
    
    def forward(self, x):
        """Return log probabilities over value distribution for each action"""
        batch_size = x.size(0)
        logits = self.head(x)  # [B, action_dim * num_atoms]
        logits = logits.view(batch_size, self.action_dim, self.num_atoms)  # [B, A, N]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    
    def get_q_values(self, log_probs):
        """Convert log probabilities to Q-values by taking expectation"""
        probs = log_probs.exp()
        q_values = (probs * self.support).sum(dim=-1)  # [B, A]
        return q_values


class BranchingDuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, price_bins: int, qty_bins: int, use_distributional: bool = False, num_atoms: int = 51):
        super().__init__()
        self.use_distributional = use_distributional
        self.trunk = MLP(obs_dim, 256)
        
        if use_distributional:
            # Distributional value and advantage heads
            self.value_dist = DistributionalHead(256, 1, num_atoms)
            self.adv_price_dist = DistributionalHead(256, price_bins, num_atoms)
            self.adv_qty_dist = DistributionalHead(256, qty_bins, num_atoms)
        else:
            # Standard scalar heads
            self.value = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
            self.adv_price = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, price_bins))
            self.adv_qty = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, qty_bins))
        
        # Enhanced auxiliary heads for multi-task learning
        self.aux_vol = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))  # Realized volatility
        self.aux_downside = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))  # Downside deviation
        self.aux_drawdown = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))  # Max drawdown depth

    def forward(self, obs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        
        if self.use_distributional:
            # Distributional forward pass
            v_log_probs = self.value_dist(z)  # [B, 1, N]
            ap_log_probs = self.adv_price_dist(z)  # [B, P, N]
            aq_log_probs = self.adv_qty_dist(z)  # [B, Q, N]
            
            # Convert to Q-values for advantage centering
            v = self.value_dist.get_q_values(v_log_probs)  # [B, 1]
            ap = self.adv_price_dist.get_q_values(ap_log_probs)  # [B, P]
            aq = self.adv_qty_dist.get_q_values(aq_log_probs)  # [B, Q]
            
            # Center advantages
            ap = ap - ap.mean(dim=1, keepdim=True)
            aq = aq - aq.mean(dim=1, keepdim=True)
            
            qp = v + ap  # [B, P]
            qq = v + aq  # [B, Q]
        else:
            # Standard forward pass
            v = self.value(z)  # [B, 1]
            ap = self.adv_price(z)  # [B, P]
            aq = self.adv_qty(z)  # [B, Q]
            
            # Center advantages
            ap = ap - ap.mean(dim=1, keepdim=True)
            aq = aq - aq.mean(dim=1, keepdim=True)
            
            qp = v + ap  # [B, P]
            qq = v + aq  # [B, Q]
        
        # Auxiliary predictions
        aux_vol = self.aux_vol(z).squeeze(-1)  # [B]
        aux_downside = self.aux_downside(z).squeeze(-1)  # [B]  
        aux_drawdown = self.aux_drawdown(z).squeeze(-1)  # [B]
        
        return qp, qq, v, aux_vol, aux_downside, aux_drawdown


class EMAWrapper(nn.Module):
    """Exponential Moving Average wrapper for model parameters (Polyak averaging)"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters (after evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def cosine_epsilon_schedule(step: int, total_steps: int, eps_start: float = 1.0, eps_end: float = 0.05, warmup_steps: int = 1000) -> float:
    """Cosine annealing epsilon schedule with warmup and optimistic initialization"""
    if step < warmup_steps:
        # Optimistic initialization - start with lower epsilon for confident exploration
        return eps_start * 0.1 + (eps_start * 0.9) * (step / warmup_steps)
    
    # Cosine annealing after warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    
    cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
    return eps_end + (eps_start - eps_end) * cosine_factor


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
