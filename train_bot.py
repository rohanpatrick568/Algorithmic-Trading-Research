#!/usr/bin/env python3
"""
ML Trading Bot Training and Continuous Development Script

This script provides a unified interface for training and continuously developing
the ML trading bot with both DeepScalper and Backtrader RL implementations.

Features:
- Automated training with configurable parameters
- Model comparison and evaluation
- Performance monitoring and logging
- Checkpoint management and resumption
- Hyperparameter optimization
- Backtesting and validation
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


class TrainingLogger:
    """Enhanced logging for training sessions"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_training_start(self, config: Dict):
        """Log training session start"""
        self.logger.info("="*60)
        self.logger.info("STARTING ML TRADING BOT TRAINING SESSION")
        self.logger.info("="*60)
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
    def log_episode_results(self, episode: int, reward: float, steps: int, epsilon: float = None):
        """Log episode results"""
        msg = f"Episode {episode:4d} | Reward: {reward:8.4f} | Steps: {steps:4d}"
        if epsilon is not None:
            msg += f" | Epsilon: {epsilon:.4f}"
        self.logger.info(msg)
        
    def log_checkpoint_saved(self, checkpoint_path: str, step: int):
        """Log checkpoint save"""
        self.logger.info(f"Checkpoint saved: {checkpoint_path} (step {step})")
        
    def log_performance_metrics(self, metrics: Dict):
        """Log performance metrics"""
        self.logger.info("Performance Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")


class ModelManager:
    """Manages model checkpoints and comparisons"""
    
    def __init__(self, models_dir: str = "models", checkpoints_dir: str = "scripts/deepscalper/checkpoints"):
        self.models_dir = Path(models_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
    def list_available_models(self) -> List[str]:
        """List all available model files"""
        model_files = []
        for ext in ['*.pt', '*.pth']:
            model_files.extend(self.models_dir.glob(ext))
            model_files.extend(self.checkpoints_dir.glob(ext))
        return [str(f) for f in model_files]
        
    def backup_current_models(self):
        """Backup current models before training"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.models_dir / f"backup_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        for model_file in self.models_dir.glob('*.pt'):
            backup_file = backup_dir / model_file.name
            backup_file.write_bytes(model_file.read_bytes())
            
        return str(backup_dir)
        
    def get_latest_checkpoint(self, checkpoint_dir: str = None) -> Optional[str]:
        """Get the latest checkpoint file"""
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoints_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)
            
        last_pt = checkpoint_dir / "last.pt"
        if last_pt.exists():
            return str(last_pt)
            
        # Find latest numbered checkpoint
        checkpoints = list(checkpoint_dir.glob("bdq_*.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
            return str(latest)
            
        return None


class DeepScalperTrainer:
    """DeepScalper training wrapper"""
    
    def __init__(self, logger: TrainingLogger):
        self.logger = logger
        
    def train(self, config: Dict) -> Dict:
        """Train DeepScalper model"""
        import subprocess
        
        cmd = [
            sys.executable, "-m", "scripts.deepscalper.main",
            "--symbol", config.get("symbol", "AAPL"),
            "--steps", str(config.get("steps", 10000)),
            "--save-every", str(config.get("save_every", 2000))
        ]
        
        if config.get("start_date"):
            cmd.extend(["--start", config["start_date"]])
        if config.get("end_date"):
            cmd.extend(["--end", config["end_date"]])
        if config.get("no_resume", False):
            cmd.append("--no-resume")
            
        self.logger.logger.info(f"Running DeepScalper training: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            training_time = time.time() - start_time
            
            return {
                "success": True,
                "training_time": training_time,
                "output": result.stdout,
                "error": result.stderr
            }
        except subprocess.CalledProcessError as e:
            training_time = time.time() - start_time
            return {
                "success": False,
                "training_time": training_time,
                "output": e.stdout,
                "error": e.stderr
            }


class BacktraderRLTrainer:
    """Backtrader RL training wrapper"""
    
    def __init__(self, logger: TrainingLogger):
        self.logger = logger
        
    def train(self, config: Dict) -> Dict:
        """Train Backtrader RL model"""
        import subprocess
        
        cmd = [
            sys.executable, "scripts/backtrader/rl/rl.py",
            "--symbol", config.get("symbol", "AAPL"),
            "--episodes", str(config.get("episodes", 100)),
            "--window", str(config.get("window", 30)),
            "--lr", str(config.get("lr", 1e-4)),
            "--batch", str(config.get("batch_size", 32)),
            "--replay", str(config.get("replay_size", 10000))
        ]
        
        if config.get("start_date"):
            cmd.extend(["--start", config["start_date"]])
        if config.get("end_date"):
            cmd.extend(["--end", config["end_date"]])
            
        self.logger.logger.info(f"Running Backtrader RL training: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            training_time = time.time() - start_time
            
            return {
                "success": True,
                "training_time": training_time,
                "output": result.stdout,
                "error": result.stderr
            }
        except subprocess.CalledProcessError as e:
            training_time = time.time() - start_time
            return {
                "success": False,
                "training_time": training_time,
                "output": e.stdout,
                "error": e.stderr
            }


class TradingBotTrainer:
    """Main trading bot trainer class"""
    
    def __init__(self):
        self.logger = TrainingLogger()
        self.model_manager = ModelManager()
        self.deepscalper_trainer = DeepScalperTrainer(self.logger)
        self.backtrader_trainer = BacktraderRLTrainer(self.logger)
        
    def train_continuous(self, config: Dict):
        """Continuous training loop"""
        self.logger.log_training_start(config)
        
        # Backup existing models
        backup_dir = self.model_manager.backup_current_models()
        self.logger.logger.info(f"Models backed up to: {backup_dir}")
        
        model_type = config.get("model_type", "deepscalper")
        iterations = config.get("iterations", 1)
        
        results = []
        
        for iteration in range(iterations):
            self.logger.logger.info(f"Starting training iteration {iteration + 1}/{iterations}")
            
            if model_type == "deepscalper":
                result = self.deepscalper_trainer.train(config)
            elif model_type == "backtrader":
                result = self.backtrader_trainer.train(config)
            else:
                # Train both models
                deepscalper_result = self.deepscalper_trainer.train(config)
                backtrader_result = self.backtrader_trainer.train(config)
                result = {
                    "deepscalper": deepscalper_result,
                    "backtrader": backtrader_result
                }
            
            results.append(result)
            
            # Log results
            if isinstance(result, dict) and "success" in result:
                status = "SUCCESS" if result["success"] else "FAILED"
                self.logger.logger.info(f"Iteration {iteration + 1} {status} - Time: {result.get('training_time', 0):.2f}s")
            else:
                self.logger.logger.info(f"Iteration {iteration + 1} completed")
                
            # Optional: evaluate model performance between iterations
            if config.get("evaluate_between_iterations", False):
                self.evaluate_models()
                
        return results
        
    def evaluate_models(self):
        """Evaluate trained models"""
        self.logger.logger.info("Evaluating models...")
        
        # List available models
        models = self.model_manager.list_available_models()
        self.logger.logger.info(f"Available models: {models}")
        
        # TODO: Implement backtesting evaluation
        # This would run backtests on the trained models and compare performance
        
    def optimize_hyperparameters(self, config: Dict):
        """Optimize hyperparameters using grid search or random search"""
        self.logger.logger.info("Starting hyperparameter optimization...")
        
        # Define parameter ranges
        param_ranges = {
            "lr": [1e-5, 1e-4, 1e-3],
            "batch_size": [32, 64, 128],
            "steps": [5000, 10000, 20000],
            "save_every": [1000, 2000, 5000]
        }
        
        best_params = None
        best_performance = float('-inf')
        
        # Simple grid search (in practice, you might want to use more sophisticated methods)
        for lr in param_ranges["lr"]:
            for batch_size in param_ranges["batch_size"]:
                test_config = config.copy()
                test_config.update({
                    "lr": lr,
                    "batch_size": batch_size,
                    "steps": min(param_ranges["steps"]),  # Use smaller steps for HP optimization
                    "no_resume": True  # Start fresh for each HP test
                })
                
                self.logger.logger.info(f"Testing hyperparameters: lr={lr}, batch_size={batch_size}")
                
                result = self.train_continuous(test_config)
                
                # Evaluate performance (simplified - would need actual performance metrics)
                performance = self._extract_performance_metric(result)
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = {"lr": lr, "batch_size": batch_size}
                    
        self.logger.logger.info(f"Best hyperparameters: {best_params} (performance: {best_performance})")
        return best_params
        
    def _extract_performance_metric(self, result) -> float:
        """Extract a performance metric from training results (placeholder)"""
        # This is a placeholder - in practice, you'd extract meaningful metrics
        # from the training output or run validation
        if isinstance(result, list) and result:
            if result[0].get("success", False):
                return result[0].get("training_time", 0) * -1  # Negative time as simple metric
        return float('-inf')


def main():
    parser = argparse.ArgumentParser(description="ML Trading Bot Training and Development")
    parser.add_argument("--model-type", choices=["deepscalper", "backtrader", "both"], 
                       default="deepscalper", help="Model type to train")
    parser.add_argument("--symbol", default="AAPL", help="Trading symbol")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps (DeepScalper)")
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes (Backtrader)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of training iterations")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh training")
    parser.add_argument("--optimize-hyperparams", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing models")
    parser.add_argument("--continuous", action="store_true", help="Run continuous training loop")
    
    args = parser.parse_args()
    
    # Set default date range if not provided
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    config = {
        "model_type": args.model_type,
        "symbol": args.symbol,
        "steps": args.steps,
        "episodes": args.episodes,
        "iterations": args.iterations,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "no_resume": args.no_resume,
        "save_every": 2000,
        "lr": 1e-4,
        "batch_size": 64,
        "window": 30,
        "replay_size": 10000,
        "evaluate_between_iterations": True
    }
    
    trainer = TradingBotTrainer()
    
    if args.evaluate_only:
        trainer.evaluate_models()
    elif args.optimize_hyperparams:
        trainer.optimize_hyperparameters(config)
    elif args.continuous:
        # Continuous training loop
        while True:
            try:
                trainer.train_continuous(config)
                time.sleep(3600)  # Wait 1 hour between training sessions
            except KeyboardInterrupt:
                print("\nContinuous training interrupted by user")
                break
    else:
        trainer.train_continuous(config)


if __name__ == "__main__":
    main()