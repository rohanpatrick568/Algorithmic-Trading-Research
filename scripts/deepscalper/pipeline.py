"""Automated training pipeline for continuous development of the DeepScalper trading bot.

This module provides a comprehensive pipeline for:
- Scheduled training runs
- Automated model evaluation and comparison
- Model versioning and deployment
- Performance tracking and alerts
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd

try:
    from .config import EnvConfig, TrainConfig
    from .env import DeepScalperEnv
    from .train import train_agent
    from .data import load_minute_data
    from .monitoring import PerformanceMonitor, evaluate_model
    from .model import BranchingDuelingQNet
    from .hyperopt import HyperparameterOptimizer, HyperparameterConfig, HyperparameterSpace
except Exception:
    from config import EnvConfig, TrainConfig
    from env import DeepScalperEnv
    from train import train_agent
    from data import load_minute_data
    from monitoring import PerformanceMonitor, evaluate_model
    from model import BranchingDuelingQNet
    from hyperopt import HyperparameterOptimizer, HyperparameterConfig, HyperparameterSpace


@dataclass
class ModelVersion:
    """Represents a versioned model with metadata."""
    version_id: str
    model_path: str
    config_path: str
    performance_metrics: Dict[str, float]
    training_steps: int
    created_at: str
    is_production: bool = False
    notes: str = ""


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    
    # Training schedule
    training_interval_hours: int = 6  # Train every 6 hours
    steps_per_training: int = 10_000
    
    # Model evaluation
    evaluation_episodes: int = 10
    performance_threshold: float = 0.1  # Minimum improvement to deploy
    
    # Model management
    max_model_versions: int = 10
    production_model_cooldown_hours: int = 24
    
    # Hyperparameter optimization
    hyperopt_interval_hours: int = 72  # Run hyperopt every 3 days
    hyperopt_trials: int = 20
    
    # Data management
    data_refresh_hours: int = 24
    training_data_days: int = 30
    
    # Alerts and monitoring
    alert_performance_drop: float = -0.2  # Alert if performance drops 20%
    alert_max_drawdown: float = 0.15  # Alert if max drawdown > 15%


class ModelRegistry:
    """Manages model versions and deployment."""
    
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        self.registry_file = self.registry_dir / "registry.json"
        self.models: Dict[str, ModelVersion] = {}
        
        self.load_registry()
    
    def register_model(self, model_path: str, config_path: str, 
                      performance_metrics: Dict[str, float], 
                      training_steps: int, notes: str = "") -> str:
        """Register a new model version."""
        
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}"
        
        # Copy model files to registry
        model_registry_path = self.registry_dir / version_id
        model_registry_path.mkdir(exist_ok=True)
        
        shutil.copy2(model_path, model_registry_path / "model.pt")
        shutil.copy2(config_path, model_registry_path / "config.json")
        
        # Create model version
        model_version = ModelVersion(
            version_id=version_id,
            model_path=str(model_registry_path / "model.pt"),
            config_path=str(model_registry_path / "config.json"),
            performance_metrics=performance_metrics,
            training_steps=training_steps,
            created_at=datetime.now().isoformat(),
            notes=notes
        )
        
        self.models[version_id] = model_version
        self.save_registry()
        
        print(f"[Registry] Registered model {version_id}")
        return version_id
    
    def get_production_model(self) -> Optional[ModelVersion]:
        """Get the current production model."""
        for model in self.models.values():
            if model.is_production:
                return model
        return None
    
    def promote_to_production(self, version_id: str, force: bool = False) -> bool:
        """Promote a model version to production."""
        
        if version_id not in self.models:
            print(f"[Registry] Model {version_id} not found")
            return False
        
        current_prod = self.get_production_model()
        new_model = self.models[version_id]
        
        # Check performance improvement
        if current_prod and not force:
            current_score = current_prod.performance_metrics.get('composite_score', 0)
            new_score = new_model.performance_metrics.get('composite_score', 0)
            
            if new_score <= current_score:
                print(f"[Registry] Model {version_id} performance not better than current production")
                return False
        
        # Demote current production model
        if current_prod:
            current_prod.is_production = False
        
        # Promote new model
        new_model.is_production = True
        self.save_registry()
        
        print(f"[Registry] Promoted model {version_id} to production")
        return True
    
    def cleanup_old_models(self, max_versions: int):
        """Remove old model versions, keeping the most recent and production models."""
        
        # Sort by creation date
        sorted_models = sorted(self.models.values(), 
                             key=lambda m: m.created_at, reverse=True)
        
        # Keep production model and most recent models
        to_keep = set()
        
        # Always keep production model
        for model in sorted_models:
            if model.is_production:
                to_keep.add(model.version_id)
        
        # Keep most recent models
        for model in sorted_models[:max_versions]:
            to_keep.add(model.version_id)
        
        # Remove old models
        to_remove = set(self.models.keys()) - to_keep
        
        for version_id in to_remove:
            model = self.models[version_id]
            # Remove files
            model_dir = Path(model.model_path).parent
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove from registry
            del self.models[version_id]
            print(f"[Registry] Removed old model {version_id}")
        
        if to_remove:
            self.save_registry()
    
    def get_model_history(self) -> pd.DataFrame:
        """Get model performance history as DataFrame."""
        
        if not self.models:
            return pd.DataFrame()
        
        data = []
        for model in self.models.values():
            row = {
                'version_id': model.version_id,
                'created_at': model.created_at,
                'training_steps': model.training_steps,
                'is_production': model.is_production,
                **model.performance_metrics
            }
            data.append(row)
        
        return pd.DataFrame(data).sort_values('created_at') if data else pd.DataFrame()
    
    def save_registry(self):
        """Save the model registry to disk."""
        registry_data = {version_id: asdict(model) for version_id, model in self.models.items()}
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def load_registry(self):
        """Load the model registry from disk."""
        if not self.registry_file.exists():
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                registry_data = json.load(f)
            
            self.models = {
                version_id: ModelVersion(**model_data) 
                for version_id, model_data in registry_data.items()
            }
            
            print(f"[Registry] Loaded {len(self.models)} model versions")
            
        except Exception as e:
            print(f"[Registry] Error loading registry: {e}")


class TrainingPipeline:
    """Automated training pipeline for continuous model development."""
    
    def __init__(self, 
                 base_env_config: EnvConfig,
                 base_train_config: TrainConfig,
                 pipeline_config: PipelineConfig,
                 workspace_dir: str = "pipeline_workspace"):
        
        self.base_env_config = base_env_config
        self.base_train_config = base_train_config
        self.pipeline_config = pipeline_config
        
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.registry = ModelRegistry(str(self.workspace_dir / "model_registry"))
        self.monitor = PerformanceMonitor(str(self.workspace_dir / "monitoring"))
        
        # Pipeline state
        self.state_file = self.workspace_dir / "pipeline_state.json"
        self.last_training = None
        self.last_hyperopt = None
        self.last_data_refresh = None
        
        self.load_state()
    
    def should_train(self) -> bool:
        """Check if it's time for a training run."""
        if self.last_training is None:
            return True
        
        time_since_training = datetime.now() - self.last_training
        return time_since_training.total_seconds() / 3600 >= self.pipeline_config.training_interval_hours
    
    def should_run_hyperopt(self) -> bool:
        """Check if it's time for hyperparameter optimization."""
        if self.last_hyperopt is None:
            return True
        
        time_since_hyperopt = datetime.now() - self.last_hyperopt
        return time_since_hyperopt.total_seconds() / 3600 >= self.pipeline_config.hyperopt_interval_hours
    
    def run_training(self, use_best_hyperparams: bool = True) -> str:
        """Run a training session and register the resulting model."""
        
        print(f"[Pipeline] Starting training run at {datetime.now()}")
        
        # Use best hyperparameters if available
        if use_best_hyperparams:
            config = self._get_best_config()
        else:
            config = (self.base_env_config, self.base_train_config)
        
        env_config, train_config = config
        train_config.train_steps = self.pipeline_config.steps_per_training
        
        # Create training directory
        training_dir = self.workspace_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_dir.mkdir(exist_ok=True)
        
        try:
            # Load data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=self.pipeline_config.training_data_days)).strftime("%Y-%m-%d")
            
            md = load_minute_data(env_config.symbol, start_date, end_date)
            env = DeepScalperEnv(md, env_config, train_config)
            
            # Train the model
            train_agent(env, train_config, str(training_dir), resume=False, 
                       save_every=self.pipeline_config.steps_per_training // 4)
            
            # Evaluate the model
            import torch
            from .train import _load_latest
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = BranchingDuelingQNet(env.obs_dim, env_config.price_bins, env_config.qty_bins).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            
            steps_loaded = _load_latest(str(training_dir), model, optimizer, map_location=device)
            
            if steps_loaded > 0:
                # Evaluate performance
                metrics = evaluate_model(env, model, device, episodes=self.pipeline_config.evaluation_episodes)
                
                # Calculate composite performance score
                composite_score = self._calculate_composite_score(metrics)
                metrics['composite_score'] = composite_score
                
                # Register the model
                model_path = str(training_dir / "last.pt")
                config_path = str(training_dir / "config.json")
                
                # Save config
                with open(config_path, 'w') as f:
                    json.dump({
                        'env_config': asdict(env_config),
                        'train_config': asdict(train_config)
                    }, f, indent=2)
                
                version_id = self.registry.register_model(
                    model_path, config_path, metrics, steps_loaded,
                    notes=f"Automated training run - {self.pipeline_config.steps_per_training} steps"
                )
                
                # Check if model should be promoted to production
                self._evaluate_for_production(version_id)
                
                # Update state
                self.last_training = datetime.now()
                self.save_state()
                
                print(f"[Pipeline] Training completed successfully - Model {version_id}")
                return version_id
            
            else:
                print("[Pipeline] Training failed - no model saved")
                return ""
                
        except Exception as e:
            print(f"[Pipeline] Training failed with error: {e}")
            return ""
    
    def run_hyperparameter_optimization(self) -> bool:
        """Run hyperparameter optimization to find better configurations."""
        
        print(f"[Pipeline] Starting hyperparameter optimization at {datetime.now()}")
        
        try:
            # Set up hyperparameter search
            search_space = HyperparameterSpace()
            optimizer = HyperparameterOptimizer(
                self.base_env_config, 
                self.base_train_config, 
                search_space,
                str(self.workspace_dir / "hyperopt")
            )
            
            # Run optimization
            best_config = optimizer.optimize(
                n_trials=self.pipeline_config.hyperopt_trials,
                train_steps_per_trial=self.pipeline_config.steps_per_training // 2
            )
            
            if best_config:
                # Save best hyperparameters
                hyperopt_results_file = self.workspace_dir / "best_hyperparams.json"
                with open(hyperopt_results_file, 'w') as f:
                    json.dump(asdict(best_config), f, indent=2)
                
                self.last_hyperopt = datetime.now()
                self.save_state()
                
                print("[Pipeline] Hyperparameter optimization completed successfully")
                return True
            else:
                print("[Pipeline] Hyperparameter optimization failed")
                return False
                
        except Exception as e:
            print(f"[Pipeline] Hyperparameter optimization failed: {e}")
            return False
    
    def _get_best_config(self) -> Tuple[EnvConfig, TrainConfig]:
        """Get the best known configuration from hyperparameter optimization."""
        
        hyperopt_results_file = self.workspace_dir / "best_hyperparams.json"
        
        if not hyperopt_results_file.exists():
            return self.base_env_config, self.base_train_config
        
        try:
            with open(hyperopt_results_file, 'r') as f:
                hyperparams = json.load(f)
            
            config = HyperparameterConfig(**hyperparams)
            env_config = config.to_env_config(self.base_env_config)
            train_config = config.to_train_config(self.base_train_config)
            
            return env_config, train_config
            
        except Exception as e:
            print(f"[Pipeline] Error loading best config: {e}")
            return self.base_env_config, self.base_train_config
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate a composite performance score."""
        
        # Weighted combination of metrics
        return_score = metrics.get('avg_episode_return', 0) / 10000
        sharpe_score = metrics.get('avg_sharpe_ratio', 0) * 0.5
        drawdown_penalty = -metrics.get('avg_max_drawdown', 0) * 2
        volatility_penalty = -metrics.get('avg_volatility', 0) * 0.1
        
        return return_score + sharpe_score + drawdown_penalty + volatility_penalty
    
    def _evaluate_for_production(self, version_id: str) -> bool:
        """Evaluate if a model should be promoted to production."""
        
        current_prod = self.registry.get_production_model()
        new_model = self.registry.models[version_id]
        
        # If no production model, promote this one
        if not current_prod:
            return self.registry.promote_to_production(version_id)
        
        # Check performance improvement threshold
        current_score = current_prod.performance_metrics.get('composite_score', 0)
        new_score = new_model.performance_metrics.get('composite_score', 0)
        
        improvement = (new_score - current_score) / max(abs(current_score), 0.01)
        
        if improvement >= self.pipeline_config.performance_threshold:
            return self.registry.promote_to_production(version_id)
        
        return False
    
    def run_pipeline_cycle(self) -> Dict[str, Any]:
        """Run a complete pipeline cycle."""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'training_run': False,
            'hyperopt_run': False,
            'model_cleanup': False,
            'alerts': []
        }
        
        # Check for hyperparameter optimization
        if self.should_run_hyperopt():
            results['hyperopt_run'] = self.run_hyperparameter_optimization()
        
        # Check for training
        if self.should_train():
            model_id = self.run_training()
            results['training_run'] = bool(model_id)
            results['new_model_id'] = model_id
        
        # Model cleanup
        self.registry.cleanup_old_models(self.pipeline_config.max_model_versions)
        results['model_cleanup'] = True
        
        # Check for alerts
        alerts = self._check_alerts()
        results['alerts'] = alerts
        
        return results
    
    def _check_alerts(self) -> List[str]:
        """Check for performance alerts."""
        alerts = []
        
        # Check recent performance
        if self.monitor.metrics_history:
            recent_perf = self.monitor.get_recent_performance(5)
            
            # Performance drop alert
            if recent_perf.get('trend', 0) < self.pipeline_config.alert_performance_drop:
                alerts.append(f"Performance trend declining: {recent_perf['trend']:.2%}")
            
            # Max drawdown alert
            recent_metrics = self.monitor.metrics_history[-5:] if len(self.monitor.metrics_history) >= 5 else self.monitor.metrics_history
            max_drawdown = max([m.max_drawdown for m in recent_metrics])
            
            if max_drawdown > self.pipeline_config.alert_max_drawdown:
                alerts.append(f"High max drawdown detected: {max_drawdown:.2%}")
        
        return alerts
    
    def save_state(self):
        """Save pipeline state."""
        state = {
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'last_hyperopt': self.last_hyperopt.isoformat() if self.last_hyperopt else None,
            'last_data_refresh': self.last_data_refresh.isoformat() if self.last_data_refresh else None,
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load pipeline state."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.last_training = datetime.fromisoformat(state['last_training']) if state.get('last_training') else None
            self.last_hyperopt = datetime.fromisoformat(state['last_hyperopt']) if state.get('last_hyperopt') else None
            self.last_data_refresh = datetime.fromisoformat(state['last_data_refresh']) if state.get('last_data_refresh') else None
            
        except Exception as e:
            print(f"[Pipeline] Error loading state: {e}")
    
    def generate_status_report(self) -> str:
        """Generate a comprehensive status report."""
        
        prod_model = self.registry.get_production_model()
        model_history = self.registry.get_model_history()
        
        report = f"""
=== DeepScalper Pipeline Status Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Pipeline Configuration:
- Training Interval: {self.pipeline_config.training_interval_hours} hours
- Steps per Training: {self.pipeline_config.steps_per_training:,}
- Max Model Versions: {self.pipeline_config.max_model_versions}

Production Model:
"""
        
        if prod_model:
            report += f"- Version: {prod_model.version_id}\n"
            report += f"- Created: {prod_model.created_at}\n"
            report += f"- Training Steps: {prod_model.training_steps:,}\n"
            report += f"- Performance Score: {prod_model.performance_metrics.get('composite_score', 0):.4f}\n"
        else:
            report += "- No production model deployed\n"
        
        report += f"""
Model Registry:
- Total Versions: {len(self.registry.models)}
- Last Training: {self.last_training.strftime('%Y-%m-%d %H:%M:%S') if self.last_training else 'Never'}
- Last Hyperopt: {self.last_hyperopt.strftime('%Y-%m-%d %H:%M:%S') if self.last_hyperopt else 'Never'}

Next Scheduled Actions:
- Training: {'Now' if self.should_train() else 'Not scheduled'}
- Hyperopt: {'Now' if self.should_run_hyperopt() else 'Not scheduled'}
"""
        
        # Add performance trends if available
        if not model_history.empty and len(model_history) > 1:
            recent_scores = model_history['composite_score'].tail(5)
            trend = "Improving" if recent_scores.iloc[-1] > recent_scores.iloc[0] else "Declining"
            report += f"- Performance Trend: {trend}\n"
        
        return report


def run_continuous_pipeline(pipeline: TrainingPipeline, 
                          check_interval_minutes: int = 60,
                          max_iterations: Optional[int] = None):
    """Run the pipeline continuously with periodic checks."""
    
    print(f"[Pipeline] Starting continuous pipeline (check every {check_interval_minutes} minutes)")
    
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        try:
            # Run pipeline cycle
            results = pipeline.run_pipeline_cycle()
            
            # Log results
            if results['training_run'] or results['hyperopt_run']:
                print(f"[Pipeline] Cycle {iteration}: Training={results['training_run']}, HyperOpt={results['hyperopt_run']}")
            
            # Print alerts
            for alert in results['alerts']:
                print(f"[Pipeline] ALERT: {alert}")
            
            # Wait for next check
            time.sleep(check_interval_minutes * 60)
            iteration += 1
            
        except KeyboardInterrupt:
            print("[Pipeline] Stopping continuous pipeline")
            break
        except Exception as e:
            print(f"[Pipeline] Error in continuous pipeline: {e}")
            time.sleep(check_interval_minutes * 60)  # Continue after error


if __name__ == "__main__":
    # Example usage
    base_env = EnvConfig(symbol="AAPL")
    base_train = TrainConfig()
    pipeline_config = PipelineConfig(
        training_interval_hours=1,  # More frequent for testing
        steps_per_training=2000
    )
    
    pipeline = TrainingPipeline(base_env, base_train, pipeline_config, "test_pipeline")
    
    # Run a single cycle
    results = pipeline.run_pipeline_cycle()
    print("Pipeline results:", results)
    
    print(pipeline.generate_status_report())