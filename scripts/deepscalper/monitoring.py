"""
Enhanced monitoring and metrics for continuous ML trading bot development.

This module provides comprehensive monitoring capabilities including:
- Training metrics tracking and visualization
- Performance evaluation with multiple metrics
- Data drift detection
- Model health monitoring
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class TrainingMetrics:
    """Container for training metrics at each step."""
    step: int
    timestamp: float
    loss: float
    q_loss: float
    aux_loss: float
    episode_return: float
    epsilon: float
    buffer_size: int
    learning_rate: float
    grad_norm: float
    eval_return: Optional[float] = None
    cash: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Container for comprehensive performance evaluation metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    trades_count: int
    avg_holding_period: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsTracker:
    """Tracks and stores training and performance metrics."""
    
    def __init__(self, save_dir: str, max_history: int = 10000):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        
        # In-memory storage with bounded size
        self.training_history: deque = deque(maxlen=max_history)
        self.performance_history: List[PerformanceMetrics] = []
        
        # Running statistics
        self.running_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')})
        
        # Load existing history if available
        self._load_history()
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """Log training metrics for this step."""
        self.training_history.append(metrics)
        
        # Update running statistics
        for key, value in metrics.to_dict().items():
            if isinstance(value, (int, float)) and value is not None:
                stats = self.running_stats[key]
                stats['sum'] += value
                stats['count'] += 1
                stats['min'] = min(stats['min'], value)
                stats['max'] = max(stats['max'], value)
        
        # Periodically save to disk
        if metrics.step % 100 == 0:
            self._save_history()
    
    def log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log performance evaluation metrics."""
        self.performance_history.append(metrics)
        self._save_history()
    
    def get_recent_metrics(self, n: int = 100) -> List[TrainingMetrics]:
        """Get the most recent n training metrics."""
        return list(self.training_history)[-n:]
    
    def get_running_average(self, metric_name: str, window: int = 1000) -> float:
        """Get running average of a metric over the last window steps."""
        recent = self.get_recent_metrics(window)
        if not recent:
            return 0.0
        
        values = [getattr(m, metric_name, 0) for m in recent if hasattr(m, metric_name)]
        return np.mean(values) if values else 0.0
    
    def get_trend(self, metric_name: str, window: int = 1000) -> str:
        """Detect trend (improving/degrading/stable) for a metric."""
        recent = self.get_recent_metrics(window)
        if len(recent) < 10:
            return "insufficient_data"
        
        values = [getattr(m, metric_name, 0) for m in recent if hasattr(m, metric_name)]
        if len(values) < 10:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "improving" if metric_name in ['episode_return', 'eval_return'] else "degrading"
        else:
            return "degrading" if metric_name in ['episode_return', 'eval_return'] else "improving"
    
    def detect_anomalies(self, metric_name: str, threshold: float = 3.0) -> List[Tuple[int, float]]:
        """Detect anomalous values using z-score."""
        recent = self.get_recent_metrics(1000)
        values = [getattr(m, metric_name, 0) for m in recent if hasattr(m, metric_name)]
        
        if len(values) < 30:
            return []
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        anomalies = []
        for i, (metrics, value) in enumerate(zip(recent, values)):
            if abs(value - mean_val) > threshold * std_val:
                anomalies.append((metrics.step, value))
        
        return anomalies
    
    def _save_history(self):
        """Save metrics history to disk."""
        # Save training metrics
        training_file = self.save_dir / "training_metrics.jsonl"
        with open(training_file, 'w') as f:
            for metrics in self.training_history:
                f.write(json.dumps(metrics.to_dict()) + '\n')
        
        # Save performance metrics
        if self.performance_history:
            perf_file = self.save_dir / "performance_metrics.json"
            with open(perf_file, 'w') as f:
                json.dump([m.to_dict() for m in self.performance_history], f, indent=2)
    
    def _load_history(self):
        """Load existing metrics history from disk."""
        # Load training metrics
        training_file = self.save_dir / "training_metrics.jsonl"
        if training_file.exists():
            with open(training_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    metrics = TrainingMetrics(**data)
                    self.training_history.append(metrics)
        
        # Load performance metrics
        perf_file = self.save_dir / "performance_metrics.json"
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                data = json.load(f)
                self.performance_history = [PerformanceMetrics(**item) for item in data]


class DataDriftDetector:
    """Detects data drift in market data features."""
    
    def __init__(self, reference_window: int = 1000, detection_window: int = 100):
        self.reference_window = reference_window
        self.detection_window = detection_window
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        self.feature_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=reference_window))
    
    def update_reference(self, features: Dict[str, np.ndarray]):
        """Update reference statistics with new feature data."""
        for feature_name, values in features.items():
            self.feature_history[feature_name].extend(values.flatten())
            
            # Update reference statistics
            if len(self.feature_history[feature_name]) >= self.reference_window:
                data = np.array(self.feature_history[feature_name])
                self.reference_stats[feature_name] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'q25': np.percentile(data, 25),
                    'q75': np.percentile(data, 75)
                }
    
    def detect_drift(self, current_features: Dict[str, np.ndarray], threshold: float = 2.0) -> Dict[str, bool]:
        """Detect drift by comparing current features to reference distribution."""
        drift_detected = {}
        
        for feature_name, values in current_features.items():
            if feature_name not in self.reference_stats:
                drift_detected[feature_name] = False
                continue
            
            ref_stats = self.reference_stats[feature_name]
            current_mean = np.mean(values)
            current_std = np.std(values)
            
            # Z-score based drift detection
            mean_drift = abs(current_mean - ref_stats['mean']) / (ref_stats['std'] + 1e-8)
            std_drift = abs(current_std - ref_stats['std']) / (ref_stats['std'] + 1e-8)
            
            drift_detected[feature_name] = mean_drift > threshold or std_drift > threshold
        
        return drift_detected


class ModelHealthMonitor:
    """Monitors model health and performance degradation."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_norms: deque = deque(maxlen=1000)
        self.weight_norms: deque = deque(maxlen=1000)
        self.activation_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Register hooks for activation monitoring
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to monitor activations."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_stats[name].append({
                        'mean': float(output.mean()),
                        'std': float(output.std()),
                        'max': float(output.max()),
                        'min': float(output.min())
                    })
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.LSTM, nn.GRU)):
                module.register_forward_hook(hook_fn(name))
    
    def update_gradient_norms(self):
        """Update gradient norm statistics."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
    
    def update_weight_norms(self):
        """Update weight norm statistics."""
        total_norm = 0.0
        for param in self.model.parameters():
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.weight_norms.append(total_norm)
    
    def check_health(self) -> Dict[str, Any]:
        """Check overall model health and return diagnostic information."""
        health_report = {
            'gradient_explosion': False,
            'gradient_vanishing': False,
            'weight_explosion': False,
            'dead_neurons': 0,
            'unstable_activations': []
        }
        
        # Check gradient issues
        if len(self.gradient_norms) > 10:
            recent_grads = list(self.gradient_norms)[-10:]
            avg_grad = np.mean(recent_grads)
            health_report['gradient_explosion'] = avg_grad > 10.0
            health_report['gradient_vanishing'] = avg_grad < 1e-6
        
        # Check weight explosion
        if len(self.weight_norms) > 10:
            recent_weights = list(self.weight_norms)[-10:]
            avg_weight = np.mean(recent_weights)
            health_report['weight_explosion'] = avg_weight > 100.0
        
        # Check activation health
        for layer_name, stats_history in self.activation_stats.items():
            if len(stats_history) > 5:
                recent_stats = list(stats_history)[-5:]
                avg_std = np.mean([s['std'] for s in recent_stats])
                
                if avg_std < 1e-6:  # Dead neurons
                    health_report['dead_neurons'] += 1
                
                if avg_std > 10.0:  # Unstable activations
                    health_report['unstable_activations'].append(layer_name)
        
        return health_report


def calculate_performance_metrics(returns: np.ndarray, trades: List[Dict]) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics from returns and trade data."""
    if len(returns) == 0:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    # Basic return metrics
    total_return = np.prod(1 + returns) - 1
    volatility = np.std(returns) * np.sqrt(252)  # Annualized
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Calmar ratio
    calmar_ratio = total_return / (abs(max_drawdown) + 1e-8)
    
    # Trade-based metrics
    if trades:
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 1e-8
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        avg_trade_return = np.mean([t.get('return', 0) for t in trades])
        avg_holding_period = np.mean([t.get('holding_period', 0) for t in trades])
    else:
        win_rate = 0
        profit_factor = 0
        avg_trade_return = 0
        avg_holding_period = 0
    
    return PerformanceMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_return=avg_trade_return,
        volatility=volatility,
        calmar_ratio=calmar_ratio,
        trades_count=len(trades),
        avg_holding_period=avg_holding_period
    )