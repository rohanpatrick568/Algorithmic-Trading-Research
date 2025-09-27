"""Hyperparameter optimization for the DeepScalper trading bot.

This module provides tools for systematic hyperparameter tuning using
various optimization strategies including grid search, random search,
and Bayesian optimization.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np

try:
    from .config import EnvConfig, TrainConfig
    from .env import DeepScalperEnv
    from .train import train_agent
    from .data import load_minute_data
    from .monitoring import PerformanceMonitor, evaluate_model
    from .model import BranchingDuelingQNet
except Exception:
    from config import EnvConfig, TrainConfig
    from env import DeepScalperEnv
    from train import train_agent
    from data import load_minute_data
    from monitoring import PerformanceMonitor, evaluate_model
    from model import BranchingDuelingQNet


@dataclass
class HyperparameterSpace:
    """Define the search space for hyperparameters."""
    
    # Learning parameters
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    
    # Network architecture
    hidden_sizes: List[int] = None
    
    # Training parameters
    batch_size_options: List[int] = None
    buffer_size_min: int = 50_000
    buffer_size_max: int = 1_000_000
    
    # Exploration
    eps_start_min: float = 0.8
    eps_start_max: float = 1.0
    eps_end_min: float = 0.01
    eps_end_max: float = 0.1
    
    # Risk parameters
    hindsight_weight_min: float = 0.0
    hindsight_weight_max: float = 1.0
    aux_weight_min: float = 0.0
    aux_weight_max: float = 0.5
    
    # Environment parameters
    lookback_options: List[int] = None
    fee_rate_min: float = 0.0001
    fee_rate_max: float = 0.002
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 256, 512]
        if self.batch_size_options is None:
            self.batch_size_options = [64, 128, 256, 512]
        if self.lookback_options is None:
            self.lookback_options = [60, 120, 180, 240]


@dataclass
class HyperparameterConfig:
    """A specific hyperparameter configuration."""
    
    # Training config parameters
    lr: float
    batch_size: int
    buffer_size: int
    eps_start: float
    eps_end: float
    hindsight_weight: float
    aux_weight: float
    
    # Environment config parameters
    lookback: int
    fee_rate: float
    
    # Other parameters
    trial_id: str
    
    def to_train_config(self, base_config: TrainConfig) -> TrainConfig:
        """Convert to TrainConfig object."""
        config_dict = asdict(base_config)
        config_dict.update({
            'lr': self.lr,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size,
            'eps_start': self.eps_start,
            'eps_end': self.eps_end,
            'hindsight_weight': self.hindsight_weight,
            'aux_weight': self.aux_weight,
        })
        return TrainConfig(**config_dict)
    
    def to_env_config(self, base_config: EnvConfig) -> EnvConfig:
        """Convert to EnvConfig object."""
        config_dict = asdict(base_config)
        config_dict.update({
            'lookback': self.lookback,
            'fee_rate': self.fee_rate,
        })
        return EnvConfig(**config_dict)


class HyperparameterOptimizer:
    """Hyperparameter optimization coordinator."""
    
    def __init__(self, 
                 base_env_config: EnvConfig,
                 base_train_config: TrainConfig,
                 search_space: HyperparameterSpace,
                 results_dir: str = "hyperopt_results"):
        
        self.base_env_config = base_env_config
        self.base_train_config = base_train_config
        self.search_space = search_space
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.results_file = self.results_dir / "optimization_results.jsonl"
        self.best_config_file = self.results_dir / "best_config.json"
        
        self.trial_results: List[Dict[str, Any]] = []
        self.best_score = float('-inf')
        self.best_config: Optional[HyperparameterConfig] = None
        
        # Load existing results if available
        self.load_results()
    
    def sample_config(self, strategy: str = "random") -> HyperparameterConfig:
        """Sample a hyperparameter configuration."""
        
        if strategy == "random":
            return self._random_sample()
        elif strategy == "grid":
            return self._grid_sample()
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _random_sample(self) -> HyperparameterConfig:
        """Random sampling from the hyperparameter space."""
        
        trial_id = f"trial_{len(self.trial_results):04d}_{random.randint(1000, 9999)}"
        
        return HyperparameterConfig(
            lr=np.random.uniform(self.search_space.lr_min, self.search_space.lr_max),
            batch_size=random.choice(self.search_space.batch_size_options),
            buffer_size=random.randint(self.search_space.buffer_size_min, self.search_space.buffer_size_max),
            eps_start=np.random.uniform(self.search_space.eps_start_min, self.search_space.eps_start_max),
            eps_end=np.random.uniform(self.search_space.eps_end_min, self.search_space.eps_end_max),
            hindsight_weight=np.random.uniform(self.search_space.hindsight_weight_min, self.search_space.hindsight_weight_max),
            aux_weight=np.random.uniform(self.search_space.aux_weight_min, self.search_space.aux_weight_max),
            lookback=random.choice(self.search_space.lookback_options),
            fee_rate=np.random.uniform(self.search_space.fee_rate_min, self.search_space.fee_rate_max),
            trial_id=trial_id
        )
    
    def _grid_sample(self) -> HyperparameterConfig:
        """Grid search sampling (simplified version)."""
        # This is a simplified grid search - in practice, you'd want a more sophisticated approach
        return self._random_sample()  # Fallback to random for now
    
    def evaluate_config(self, 
                       config: HyperparameterConfig, 
                       train_steps: int = 5000,
                       eval_episodes: int = 5) -> Dict[str, float]:
        """Evaluate a hyperparameter configuration."""
        
        print(f"[HyperOpt] Evaluating {config.trial_id}")
        print(f"  lr={config.lr:.5f}, batch_size={config.batch_size}, lookback={config.lookback}")
        
        try:
            # Create configs
            env_config = config.to_env_config(self.base_env_config)
            train_config = config.to_train_config(self.base_train_config)
            train_config.train_steps = train_steps
            
            # Create temporary directory for this trial
            trial_dir = self.results_dir / config.trial_id
            trial_dir.mkdir(exist_ok=True)
            
            # Load data and create environment
            md = load_minute_data(env_config.symbol, "2023-01-01", "2023-01-31")
            env = DeepScalperEnv(md, env_config, train_config)
            
            # Train the model
            train_agent(env, train_config, str(trial_dir), resume=False, save_every=train_steps//2)
            
            # Load the trained model for evaluation
            import torch
            from .train import _load_latest
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = BranchingDuelingQNet(env.obs_dim, env_config.price_bins, env_config.qty_bins).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            
            steps_loaded = _load_latest(str(trial_dir), model, optimizer, map_location=device)
            
            if steps_loaded > 0:
                # Evaluate the model
                metrics = evaluate_model(env, model, device, episodes=eval_episodes)
                
                # Load monitoring data
                monitor = PerformanceMonitor(str(trial_dir / "logs"))
                monitor.load_history()
                recent_perf = monitor.get_recent_performance(eval_episodes)
                
                # Combine metrics
                combined_metrics = {
                    **metrics,
                    **recent_perf,
                    'training_steps': steps_loaded
                }
                
                print(f"[HyperOpt] {config.trial_id} completed: avg_return={metrics.get('avg_episode_return', 0):.2f}")
                return combined_metrics
            else:
                print(f"[HyperOpt] {config.trial_id} failed: no model saved")
                return {'avg_episode_return': float('-inf'), 'error': 'training_failed'}
                
        except Exception as e:
            print(f"[HyperOpt] {config.trial_id} failed with error: {e}")
            return {'avg_episode_return': float('-inf'), 'error': str(e)}
    
    def optimize(self, 
                n_trials: int = 20,
                train_steps_per_trial: int = 5000,
                eval_episodes: int = 5,
                strategy: str = "random") -> HyperparameterConfig:
        """Run hyperparameter optimization."""
        
        print(f"[HyperOpt] Starting optimization with {n_trials} trials")
        
        for trial_idx in range(n_trials):
            print(f"\n[HyperOpt] Trial {trial_idx + 1}/{n_trials}")
            
            # Sample configuration
            config = self.sample_config(strategy)
            
            # Evaluate configuration
            metrics = self.evaluate_config(config, train_steps_per_trial, eval_episodes)
            
            # Calculate composite score (you can customize this)
            score = self._calculate_score(metrics)
            
            # Store results
            result = {
                'trial_id': config.trial_id,
                'config': asdict(config),
                'metrics': metrics,
                'score': score,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            self.trial_results.append(result)
            
            # Update best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                print(f"[HyperOpt] New best configuration found! Score: {score:.4f}")
                
                # Save best config
                with open(self.best_config_file, 'w') as f:
                    json.dump(asdict(config), f, indent=2)
            
            # Save results
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
        
        print(f"\n[HyperOpt] Optimization completed. Best score: {self.best_score:.4f}")
        return self.best_config
    
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate a composite score from evaluation metrics."""
        
        # Handle failed trials
        if 'error' in metrics:
            return float('-inf')
        
        # Composite score combining multiple metrics
        return_score = metrics.get('avg_episode_return', 0) / 10000  # Normalize
        sharpe_score = metrics.get('avg_sharpe_ratio', 0) * 0.5
        consistency_score = metrics.get('consistency', 0) * 0.3
        
        # Penalize high drawdown
        drawdown_penalty = -metrics.get('avg_max_drawdown', 0) * 2
        
        total_score = return_score + sharpe_score + consistency_score + drawdown_penalty
        return total_score
    
    def load_results(self):
        """Load existing optimization results."""
        
        if not self.results_file.exists():
            return
        
        try:
            with open(self.results_file, 'r') as f:
                for line in f:
                    result = json.loads(line.strip())
                    self.trial_results.append(result)
                    
                    # Update best config
                    if result['score'] > self.best_score:
                        self.best_score = result['score']
                        self.best_config = HyperparameterConfig(**result['config'])
            
            print(f"[HyperOpt] Loaded {len(self.trial_results)} previous trials")
            
        except Exception as e:
            print(f"[HyperOpt] Error loading results: {e}")
    
    def get_best_configs(self, n: int = 5) -> List[Tuple[HyperparameterConfig, float]]:
        """Get the top N configurations by score."""
        
        sorted_results = sorted(self.trial_results, key=lambda x: x['score'], reverse=True)
        top_results = sorted_results[:n]
        
        return [(HyperparameterConfig(**r['config']), r['score']) for r in top_results]
    
    def generate_report(self) -> str:
        """Generate a summary report of the optimization results."""
        
        if not self.trial_results:
            return "No optimization results available."
        
        scores = [r['score'] for r in self.trial_results if r['score'] != float('-inf')]
        valid_trials = len(scores)
        failed_trials = len(self.trial_results) - valid_trials
        
        report = f"""
=== Hyperparameter Optimization Report ===

Total Trials: {len(self.trial_results)}
Valid Trials: {valid_trials}
Failed Trials: {failed_trials}

Best Score: {self.best_score:.4f}
Average Score: {np.mean(scores):.4f}
Std Score: {np.std(scores):.4f}

Best Configuration:
"""
        
        if self.best_config:
            report += f"  Learning Rate: {self.best_config.lr:.5f}\n"
            report += f"  Batch Size: {self.best_config.batch_size}\n"
            report += f"  Buffer Size: {self.best_config.buffer_size:,}\n"
            report += f"  Lookback: {self.best_config.lookback}\n"
            report += f"  Fee Rate: {self.best_config.fee_rate:.4f}\n"
            report += f"  Epsilon Start: {self.best_config.eps_start:.3f}\n"
            report += f"  Epsilon End: {self.best_config.eps_end:.3f}\n"
            report += f"  Hindsight Weight: {self.best_config.hindsight_weight:.3f}\n"
            report += f"  Aux Weight: {self.best_config.aux_weight:.3f}\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    base_env = EnvConfig(symbol="AAPL")
    base_train = TrainConfig(train_steps=2000)
    search_space = HyperparameterSpace()
    
    optimizer = HyperparameterOptimizer(base_env, base_train, search_space)
    
    # Run a small optimization
    best_config = optimizer.optimize(n_trials=3, train_steps_per_trial=1000)
    
    print(optimizer.generate_report())