"""Performance monitoring and evaluation metrics for the DeepScalper trading bot.

This module provides comprehensive monitoring capabilities including:
- Training progress tracking
- Model performance evaluation
- Risk metrics calculation
- Performance visualization
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch

try:
    from .env import DeepScalperEnv
    from .model import BranchingDuelingQNet
except Exception:
    from env import DeepScalperEnv
    from model import BranchingDuelingQNet


@dataclass
class TrainingMetrics:
    """Training metrics for a single episode/evaluation."""
    step: int
    episode: int
    episode_return: float
    final_cash: float
    final_equity: float
    total_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    max_position: float
    avg_hold_time: float
    timestamp: str


@dataclass
class ModelPerformance:
    """Overall model performance summary."""
    total_steps: int
    total_episodes: int
    avg_return: float
    best_return: float
    worst_return: float
    win_rate: float
    avg_sharpe: float
    avg_max_drawdown: float
    training_time_hours: float
    last_updated: str


class PerformanceMonitor:
    """Monitor and track trading bot performance during training and evaluation."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics_file = self.log_dir / "training_metrics.jsonl"
        self.summary_file = self.log_dir / "performance_summary.json"
        
        self.start_time = time.time()
        self.metrics_history: List[TrainingMetrics] = []
        
    def log_episode(self, step: int, episode: int, env_info: Dict[str, Any], 
                   additional_metrics: Optional[Dict[str, float]] = None):
        """Log metrics for a completed episode."""
        
        # Calculate basic metrics from environment info
        episode_return = float(env_info.get('total_return', 0.0))
        final_cash = float(env_info.get('cash', 100000.0))
        final_equity = float(env_info.get('equity', final_cash))
        
        # Calculate trading statistics
        trades = env_info.get('trades', [])
        total_trades = len(trades) if trades else 0
        
        # Win rate calculation
        if total_trades > 0 and trades:
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0.0
        
        # Calculate returns for risk metrics
        returns = env_info.get('returns', [])
        if returns and len(returns) > 1:
            returns_array = np.array(returns)
            volatility = np.std(returns_array) * np.sqrt(252 * 390)  # Annualized
            
            # Sharpe ratio (assuming risk-free rate of 0)
            if volatility > 0:
                sharpe_ratio = np.mean(returns_array) / volatility * np.sqrt(252 * 390)
            else:
                sharpe_ratio = 0.0
                
            # Max drawdown
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
        
        # Position and hold time metrics
        positions = env_info.get('positions', [])
        max_position = max([abs(p) for p in positions]) if positions else 0.0
        avg_hold_time = np.mean([t.get('hold_time', 0) for t in trades]) if trades else 0.0
        
        # Create metrics object
        metrics = TrainingMetrics(
            step=step,
            episode=episode,
            episode_return=episode_return,
            final_cash=final_cash,
            final_equity=final_equity,
            total_trades=total_trades,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            max_position=max_position,
            avg_hold_time=avg_hold_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Add additional metrics if provided
        if additional_metrics:
            for key, value in additional_metrics.items():
                setattr(metrics, key, value)
        
        self.metrics_history.append(metrics)
        
        # Write to log file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
        
        # Update summary
        self._update_summary()
        
        return metrics
    
    def _update_summary(self):
        """Update the performance summary."""
        if not self.metrics_history:
            return
        
        # Calculate summary statistics
        returns = [m.episode_return for m in self.metrics_history]
        win_rates = [m.win_rate for m in self.metrics_history]
        sharpe_ratios = [m.sharpe_ratio for m in self.metrics_history if not np.isnan(m.sharpe_ratio)]
        max_drawdowns = [m.max_drawdown for m in self.metrics_history]
        
        summary = ModelPerformance(
            total_steps=self.metrics_history[-1].step,
            total_episodes=len(self.metrics_history),
            avg_return=np.mean(returns),
            best_return=np.max(returns),
            worst_return=np.min(returns),
            win_rate=np.mean(win_rates),
            avg_sharpe=np.mean(sharpe_ratios) if sharpe_ratios else 0.0,
            avg_max_drawdown=np.mean(max_drawdowns),
            training_time_hours=(time.time() - self.start_time) / 3600,
            last_updated=datetime.now().isoformat()
        )
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
    
    def get_recent_performance(self, n_episodes: int = 10) -> Dict[str, float]:
        """Get performance metrics for the last n episodes."""
        if len(self.metrics_history) < n_episodes:
            recent = self.metrics_history
        else:
            recent = self.metrics_history[-n_episodes:]
        
        if not recent:
            return {}
        
        returns = [m.episode_return for m in recent]
        win_rates = [m.win_rate for m in recent]
        sharpe_ratios = [m.sharpe_ratio for m in recent if not np.isnan(m.sharpe_ratio)]
        
        return {
            'avg_return': np.mean(returns),
            'win_rate': np.mean(win_rates),
            'avg_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0.0,
            'consistency': 1.0 - (np.std(returns) / max(abs(np.mean(returns)), 1.0)),
            'trend': (returns[-1] - returns[0]) / max(abs(returns[0]), 1.0) if len(returns) > 1 else 0.0
        }
    
    def should_save_model(self, current_metrics: TrainingMetrics, 
                         patience: int = 5, min_episodes: int = 10) -> bool:
        """Determine if the current model should be saved based on performance."""
        if len(self.metrics_history) < min_episodes:
            return False
        
        # Look at recent performance
        recent = self.get_recent_performance(patience)
        historical = self.get_recent_performance(min(len(self.metrics_history), 50))
        
        # Save if recent performance is significantly better
        improvement_threshold = 0.1  # 10% improvement
        
        criteria = [
            recent['avg_return'] > historical['avg_return'] * (1 + improvement_threshold),
            recent['win_rate'] > historical['win_rate'] * (1 + improvement_threshold * 0.5),
            recent['avg_sharpe'] > historical['avg_sharpe'] * (1 + improvement_threshold * 0.5),
            recent['consistency'] > 0.5  # Reasonable consistency
        ]
        
        return sum(criteria) >= 2  # At least 2 criteria met
    
    def generate_report(self) -> str:
        """Generate a text report of the current performance."""
        if not self.metrics_history:
            return "No training data available."
        
        recent = self.get_recent_performance(10)
        all_time = self.get_recent_performance(len(self.metrics_history))
        
        report = f"""
=== DeepScalper Training Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Training Progress:
- Total Steps: {self.metrics_history[-1].step:,}
- Total Episodes: {len(self.metrics_history):,}
- Training Time: {(time.time() - self.start_time) / 3600:.2f} hours

Recent Performance (Last 10 Episodes):
- Average Return: {recent.get('avg_return', 0):.2f}
- Win Rate: {recent.get('win_rate', 0) * 100:.1f}%
- Average Sharpe: {recent.get('avg_sharpe', 0):.3f}
- Consistency: {recent.get('consistency', 0) * 100:.1f}%

All-Time Performance:
- Average Return: {all_time.get('avg_return', 0):.2f}
- Best Episode: {np.max([m.episode_return for m in self.metrics_history]):.2f}
- Worst Episode: {np.min([m.episode_return for m in self.metrics_history]):.2f}
- Overall Win Rate: {all_time.get('win_rate', 0) * 100:.1f}%

Latest Episode:
- Return: {self.metrics_history[-1].episode_return:.2f}
- Final Cash: ${self.metrics_history[-1].final_cash:,.2f}
- Trades: {self.metrics_history[-1].total_trades}
- Max Drawdown: {self.metrics_history[-1].max_drawdown * 100:.2f}%
"""
        return report
    
    def load_history(self) -> bool:
        """Load training history from log files."""
        if not self.metrics_file.exists():
            return False
        
        try:
            self.metrics_history = []
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    metrics = TrainingMetrics(**data)
                    self.metrics_history.append(metrics)
            return True
        except Exception as e:
            print(f"Error loading training history: {e}")
            return False


@torch.no_grad()
def evaluate_model(env: DeepScalperEnv, model: BranchingDuelingQNet, 
                  device: torch.device, episodes: int = 5) -> Dict[str, float]:
    """Comprehensive model evaluation with detailed metrics."""
    results = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        trades = []
        positions = []
        returns = []
        last_equity = env.cash
        
        while not done:
            # Get model prediction
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            qp, qq, _, _ = model(o)
            ap = qp.argmax(dim=1).item()
            aq = qq.argmax(dim=1).item()
            
            # Track position before action
            positions.append(env.pos)
            
            # Take action
            obs, r, done, trunc, info = env.step(np.array([ap, aq], dtype=np.int64))
            episode_return += info.get("r_raw", r)
            
            # Track returns
            current_equity = info.get('equity', env.cash + env.pos * info.get('price', 0))
            if abs(last_equity) > 1e-6:
                period_return = (current_equity - last_equity) / last_equity
                returns.append(period_return)
            last_equity = current_equity
            
            # Track trades (simplified)
            if 'trade' in info:
                trades.append(info['trade'])
        
        # Calculate episode metrics
        final_cash = env.cash
        total_trades = len(trades)
        
        # Risk metrics
        if returns:
            returns_array = np.array(returns)
            volatility = np.std(returns_array) * np.sqrt(252 * 390)
            sharpe = np.mean(returns_array) / volatility * np.sqrt(252 * 390) if volatility > 0 else 0
            
            # Max drawdown
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
        else:
            volatility = sharpe = max_drawdown = 0.0
        
        results.append({
            'episode_return': episode_return,
            'final_cash': final_cash,
            'total_trades': total_trades,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_position': max([abs(p) for p in positions]) if positions else 0.0
        })
    
    # Aggregate results
    metrics = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        metrics[f'avg_{key}'] = np.mean(values)
        metrics[f'std_{key}'] = np.std(values)
        metrics[f'min_{key}'] = np.min(values)
        metrics[f'max_{key}'] = np.max(values)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor("test_logs")
    
    # Simulate some episodes
    for episode in range(5):
        step = episode * 100 + 100
        env_info = {
            'total_return': np.random.normal(500, 2000),
            'cash': 100000 + np.random.normal(0, 5000),
            'equity': 100000 + np.random.normal(0, 5000),
            'trades': [{'pnl': np.random.normal(0, 100)} for _ in range(np.random.randint(5, 20))],
            'returns': np.random.normal(0, 0.001, 100),
            'positions': np.random.normal(0, 0.5, 100)
        }
        
        metrics = monitor.log_episode(step, episode, env_info)
        print(f"Episode {episode}: Return={metrics.episode_return:.2f}, Sharpe={metrics.sharpe_ratio:.3f}")
    
    print(monitor.generate_report())