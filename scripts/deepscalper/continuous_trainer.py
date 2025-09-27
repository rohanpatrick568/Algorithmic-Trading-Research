"""
Continuous training orchestrator for the ML trading bot.

This module orchestrates continuous training, monitoring, and adaptation
of the ML trading bot with automated retraining triggers.
"""

from __future__ import annotations

import json
import os
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

try:
    from .config import EnvConfig, TrainConfig  # type: ignore
    from .env import DeepScalperEnv  # type: ignore
    from .train import train_agent, evaluate  # type: ignore
    from .data_pipeline import ContinuousDataPipeline, DataPipelineConfig  # type: ignore
    from .monitoring import MetricsTracker, PerformanceMetrics, calculate_performance_metrics  # type: ignore
    from .model import BranchingDuelingQNet  # type: ignore
except Exception:
    from config import EnvConfig, TrainConfig  # type: ignore
    from env import DeepScalperEnv  # type: ignore
    from train import train_agent, evaluate  # type: ignore
    from data_pipeline import ContinuousDataPipeline, DataPipelineConfig  # type: ignore
    from monitoring import MetricsTracker, PerformanceMetrics, calculate_performance_metrics  # type: ignore
    from model import BranchingDuelingQNet  # type: ignore


class RetrainingTrigger(Enum):
    """Reasons for triggering model retraining."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    MODEL_HEALTH = "model_health"


@dataclass
class ContinuousTrainingConfig:
    """Configuration for continuous training."""
    # Base configurations
    env_config: EnvConfig
    train_config: TrainConfig
    data_pipeline_config: DataPipelineConfig
    
    # Continuous training parameters
    retraining_interval_hours: int = 24  # Minimum time between retrainings
    performance_window_days: int = 7     # Window for performance evaluation
    performance_threshold: float = -0.1  # Trigger retraining if return drops below this
    
    # Adaptation parameters
    enable_incremental_learning: bool = True
    incremental_steps: int = 10000       # Steps for incremental training
    full_retrain_frequency: int = 7      # Full retrain every N incremental updates
    
    # Model management
    max_model_versions: int = 10         # Keep this many model versions
    validation_episodes: int = 100       # Episodes for model validation
    
    # Monitoring
    enable_monitoring: bool = True
    alert_channels: List[str] = None     # Email/Slack channels for alerts
    
    def __post_init__(self):
        if self.alert_channels is None:
            self.alert_channels = []


class ModelVersionManager:
    """Manages different versions of trained models."""
    
    def __init__(self, base_dir: str, max_versions: int = 10):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_versions = max_versions
        self.versions_file = self.base_dir / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version metadata."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {'versions': [], 'active_version': None}
    
    def _save_versions(self):
        """Save version metadata."""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2, default=str)
    
    def create_version(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                      metadata: Dict[str, Any]) -> str:
        """Create a new model version."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}"
        version_dir = self.base_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Save model state
        model_path = version_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata
        }, model_path)
        
        # Update version registry
        version_info = {
            'version_id': version_id,
            'created_at': datetime.now(),
            'model_path': str(model_path),
            'metadata': metadata,
            'performance': {}
        }
        
        self.versions['versions'].append(version_info)
        self.versions['active_version'] = version_id
        
        # Cleanup old versions
        self._cleanup_old_versions()
        self._save_versions()
        
        return version_id
    
    def load_version(self, version_id: str, model: nn.Module, 
                    optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Load a specific model version."""
        version_info = self.get_version_info(version_id)
        if not version_info:
            return False
        
        try:
            checkpoint = torch.load(version_info['model_path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            return True
        except Exception:
            return False
    
    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version."""
        for version in self.versions['versions']:
            if version['version_id'] == version_id:
                return version
        return None
    
    def get_active_version(self) -> Optional[str]:
        """Get the active version ID."""
        return self.versions.get('active_version')
    
    def set_active_version(self, version_id: str) -> bool:
        """Set the active version."""
        if self.get_version_info(version_id):
            self.versions['active_version'] = version_id
            self._save_versions()
            return True
        return False
    
    def update_performance(self, version_id: str, performance: Dict[str, Any]):
        """Update performance metrics for a version."""
        version_info = self.get_version_info(version_id)
        if version_info:
            version_info['performance'] = performance
            self._save_versions()
    
    def get_best_version(self, metric: str = 'sharpe_ratio') -> Optional[str]:
        """Get the version with the best performance on a specific metric."""
        best_version = None
        best_score = float('-inf')
        
        for version in self.versions['versions']:
            performance = version.get('performance', {})
            if metric in performance:
                score = performance[metric]
                if score > best_score:
                    best_score = score
                    best_version = version['version_id']
        
        return best_version
    
    def _cleanup_old_versions(self):
        """Remove old versions exceeding the limit."""
        if len(self.versions['versions']) <= self.max_versions:
            return
        
        # Sort by creation date and keep the most recent
        sorted_versions = sorted(
            self.versions['versions'],
            key=lambda v: v['created_at'] if isinstance(v['created_at'], str) 
            else v['created_at'].isoformat(),
            reverse=True
        )
        
        # Remove old versions
        for old_version in sorted_versions[self.max_versions:]:
            version_dir = Path(old_version['model_path']).parent
            if version_dir.exists():
                import shutil
                shutil.rmtree(version_dir)
        
        # Keep only the recent versions in metadata
        self.versions['versions'] = sorted_versions[:self.max_versions]


class ContinuousTrainer:
    """Main orchestrator for continuous training."""
    
    def __init__(self, config: ContinuousTrainingConfig, work_dir: str = "continuous_training"):
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_pipeline = ContinuousDataPipeline(config.data_pipeline_config)
        self.metrics_tracker = MetricsTracker(str(self.work_dir / "metrics"))
        self.model_manager = ModelVersionManager(str(self.work_dir / "models"), config.max_model_versions)
        
        # Training state
        self.current_model = None
        self.current_optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stop_event = threading.Event()
        self.training_thread = None
        self.incremental_update_count = 0
        self.last_retrain_time = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Performance tracking
        self.performance_history = []
        self.retraining_history = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for continuous trainer."""
        logger = logging.getLogger('ContinuousTrainer')
        logger.setLevel(logging.INFO)
        
        log_file = self.work_dir / "continuous_training.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def start(self):
        """Start continuous training."""
        self.logger.info("Starting continuous training system")
        
        # Start data pipeline
        self.data_pipeline.start()
        
        # Initialize or load existing model
        self._initialize_model()
        
        # Start main training loop
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        self.logger.info("Continuous training system started")
    
    def stop(self):
        """Stop continuous training."""
        self.logger.info("Stopping continuous training system")
        
        self.stop_event.set()
        self.data_pipeline.stop()
        
        if self.training_thread:
            self.training_thread.join(timeout=30)
        
        self.logger.info("Continuous training system stopped")
    
    def _initialize_model(self):
        """Initialize or load existing model."""
        # Get the latest data to determine observation dimensions
        sample_data = self.data_pipeline.get_latest_data(
            self.config.data_pipeline_config.symbols[0], 
            lookback_minutes=self.config.env_config.lookback
        )
        
        if sample_data is None:
            self.logger.error("No data available for model initialization")
            return
        
        # Create environment to get observation space
        env = DeepScalperEnv(sample_data, self.config.env_config, self.config.train_config)
        obs_dim = env.obs_dim
        
        # Create model
        self.current_model = BranchingDuelingQNet(
            obs_dim, 
            self.config.env_config.price_bins, 
            self.config.env_config.qty_bins
        ).to(self.device)
        
        self.current_optimizer = torch.optim.Adam(
            self.current_model.parameters(), 
            lr=self.config.train_config.lr
        )
        
        # Try to load existing model
        active_version = self.model_manager.get_active_version()
        if active_version:
            if self.model_manager.load_version(active_version, self.current_model, self.current_optimizer):
                self.logger.info(f"Loaded existing model version: {active_version}")
            else:
                self.logger.warning(f"Failed to load model version: {active_version}")
        else:
            self.logger.info("Initialized new model")
    
    def _training_loop(self):
        """Main continuous training loop."""
        while not self.stop_event.is_set():
            try:
                # Check if retraining is needed
                trigger = self._check_retraining_triggers()
                
                if trigger:
                    self.logger.info(f"Retraining triggered by: {trigger.value}")
                    self._perform_retraining(trigger)
                
                # Sleep before next check
                self.stop_event.wait(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                self.stop_event.wait(600)  # Wait 10 minutes before retry
    
    def _check_retraining_triggers(self) -> Optional[RetrainingTrigger]:
        """Check if retraining should be triggered."""
        # Check scheduled retraining
        if self._should_retrain_scheduled():
            return RetrainingTrigger.SCHEDULED
        
        # Check performance degradation
        if self._should_retrain_performance():
            return RetrainingTrigger.PERFORMANCE_DEGRADATION
        
        # Check data drift (if data pipeline supports it)
        if self._should_retrain_drift():
            return RetrainingTrigger.DATA_DRIFT
        
        # Check model health
        if self._should_retrain_health():
            return RetrainingTrigger.MODEL_HEALTH
        
        return None
    
    def _should_retrain_scheduled(self) -> bool:
        """Check if scheduled retraining is due."""
        if self.last_retrain_time is None:
            return True
        
        hours_since_retrain = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
        return hours_since_retrain >= self.config.retraining_interval_hours
    
    def _should_retrain_performance(self) -> bool:
        """Check if performance has degraded significantly."""
        # Get recent performance metrics
        recent_metrics = self.metrics_tracker.get_recent_metrics(1000)
        if len(recent_metrics) < 100:
            return False
        
        # Calculate recent average return
        recent_returns = [m.episode_return for m in recent_metrics[-100:]]
        avg_return = np.mean(recent_returns)
        
        return avg_return < self.config.performance_threshold
    
    def _should_retrain_drift(self) -> bool:
        """Check if data drift has been detected."""
        # This would integrate with data drift detection from the pipeline
        # For now, return False as a placeholder
        return False
    
    def _should_retrain_health(self) -> bool:
        """Check if model health has degraded."""
        # This would check model health metrics
        # For now, return False as a placeholder
        return False
    
    def _perform_retraining(self, trigger: RetrainingTrigger):
        """Perform model retraining."""
        self.logger.info(f"Starting retraining due to: {trigger.value}")
        
        # Get fresh data
        training_data = self._prepare_training_data()
        if not training_data:
            self.logger.error("No training data available")
            return
        
        try:
            # Decide between incremental and full retraining
            if (self.config.enable_incremental_learning and 
                trigger != RetrainingTrigger.SCHEDULED and
                self.incremental_update_count < self.config.full_retrain_frequency):
                
                self._perform_incremental_training(training_data)
                self.incremental_update_count += 1
            else:
                self._perform_full_retraining(training_data)
                self.incremental_update_count = 0
            
            # Validate new model
            if self._validate_model():
                # Save new model version
                metadata = {
                    'trigger': trigger.value,
                    'timestamp': datetime.now(),
                    'incremental': self.incremental_update_count > 0,
                    'data_samples': len(training_data.df)
                }
                
                version_id = self.model_manager.create_version(
                    self.current_model, 
                    self.current_optimizer, 
                    metadata
                )
                
                self.logger.info(f"Created new model version: {version_id}")
            else:
                self.logger.warning("New model failed validation, keeping previous version")
            
            self.last_retrain_time = datetime.now()
            
            # Record retraining event
            self.retraining_history.append({
                'timestamp': datetime.now(),
                'trigger': trigger.value,
                'incremental': self.incremental_update_count > 0
            })
            
        except Exception as e:
            self.logger.error(f"Retraining failed: {e}")
    
    def _prepare_training_data(self):
        """Prepare data for training."""
        # Get data for the primary symbol
        primary_symbol = self.config.data_pipeline_config.symbols[0]
        lookback_minutes = self.config.performance_window_days * 24 * 60
        
        return self.data_pipeline.get_latest_data(primary_symbol, lookback_minutes)
    
    def _perform_incremental_training(self, training_data):
        """Perform incremental training with limited steps."""
        self.logger.info(f"Performing incremental training with {self.config.incremental_steps} steps")
        
        # Create environment
        env = DeepScalperEnv(training_data, self.config.env_config, self.config.train_config)
        
        # Create modified config for incremental training
        incremental_config = TrainConfig(
            **{**asdict(self.config.train_config), 'train_steps': self.config.incremental_steps}
        )
        
        # Train with existing model as starting point
        train_agent(
            env, 
            incremental_config, 
            ckpt_dir=str(self.work_dir / "temp_training"),
            resume=False,  # Don't resume, use current model state
            save_every=self.config.incremental_steps  # Save at the end
        )
    
    def _perform_full_retraining(self, training_data):
        """Perform full retraining from scratch or previous checkpoint."""
        self.logger.info("Performing full retraining")
        
        # Create environment
        env = DeepScalperEnv(training_data, self.config.env_config, self.config.train_config)
        
        # Train model
        train_agent(
            env, 
            self.config.train_config, 
            ckpt_dir=str(self.work_dir / "temp_training"),
            resume=True,  # Resume from best checkpoint if available
            save_every=self.config.train_config.train_steps // 10
        )
    
    def _validate_model(self) -> bool:
        """Validate the newly trained model."""
        try:
            # Get validation data
            validation_data = self._prepare_training_data()
            if not validation_data:
                return False
            
            # Create validation environment
            env = DeepScalperEnv(validation_data, self.config.env_config, self.config.train_config)
            
            # Evaluate model
            performance = evaluate(env, self.current_model, self.device, episodes=self.config.validation_episodes)
            
            # Simple validation: performance should be reasonable
            return performance > -100.0  # Adjust threshold as needed
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of continuous training."""
        return {
            'active': not self.stop_event.is_set(),
            'current_version': self.model_manager.get_active_version(),
            'last_retrain': self.last_retrain_time,
            'incremental_updates': self.incremental_update_count,
            'data_pipeline_stats': self.data_pipeline.get_stats(),
            'model_versions': len(self.model_manager.versions['versions']),
            'retraining_events': len(self.retraining_history)
        }
    
    def trigger_manual_retraining(self):
        """Manually trigger retraining."""
        self.logger.info("Manual retraining triggered")
        self._perform_retraining(RetrainingTrigger.MANUAL)
    
    def switch_to_best_model(self, metric: str = 'sharpe_ratio') -> bool:
        """Switch to the best performing model version."""
        best_version = self.model_manager.get_best_version(metric)
        if best_version and best_version != self.model_manager.get_active_version():
            if self.model_manager.load_version(best_version, self.current_model, self.current_optimizer):
                self.model_manager.set_active_version(best_version)
                self.logger.info(f"Switched to best model version: {best_version}")
                return True
        return False


def create_default_config() -> ContinuousTrainingConfig:
    """Create a default configuration for continuous training."""
    return ContinuousTrainingConfig(
        env_config=EnvConfig(),
        train_config=TrainConfig(),
        data_pipeline_config=DataPipelineConfig()
    )