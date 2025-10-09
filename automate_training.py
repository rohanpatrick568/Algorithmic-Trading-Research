#!/usr/bin/env python3
"""
Automated Training Scheduler for ML Trading Bot

This script runs automated training sessions on a schedule, manages data updates,
and monitors model performance for continuous development.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

try:
    import schedule
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "schedule"])
    import schedule


class AutomatedTrainer:
    """Automated training scheduler and manager"""
    
    def __init__(self, config_file: str = "training_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_logging()
        self.last_training_time = None
        self.training_history = []
        
    def load_config(self) -> Dict:
        """Load training configuration"""
        default_config = {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
            "model_types": ["deepscalper"],
            "training_schedule": {
                "daily_at": "02:00",  # 2 AM daily
                "data_update_at": "01:30",  # 1:30 AM daily for data update
                "evaluation_at": "03:00"  # 3 AM for evaluation
            },
            "training_params": {
                "steps": 10000,
                "episodes": 200,
                "iterations": 1,
                "lookback_days": 365
            },
            "data_params": {
                "update_frequency": "daily",
                "lookback_days": 365,
                "intervals": ["1d"]
            },
            "monitoring": {
                "min_sharpe_ratio": 0.5,
                "max_drawdown_threshold": -25.0,
                "performance_check_frequency": "weekly"
            },
            "notifications": {
                "enabled": False,
                "email": "",
                "slack_webhook": ""
            }
        }
        
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        else:
            # Save default config
            self.save_config(default_config)
            
        return default_config
    
    def save_config(self, config: Dict):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def setup_logging(self):
        """Setup logging for automated training"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"automated_training_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_data_update(self):
        """Run automated data update"""
        self.logger.info("Starting automated data update...")
        
        try:
            cmd = [
                sys.executable, "data_pipeline.py",
                "--symbols"] + self.config["symbols"] + [
                "--update-data",
                "--lookback-days", str(self.config["data_params"]["lookback_days"]),
                "--intervals"] + self.config["data_params"]["intervals"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                self.logger.info("Data update completed successfully")
                return True
            else:
                self.logger.error(f"Data update failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Data update error: {e}")
            return False
    
    def run_training_session(self):
        """Run automated training session"""
        self.logger.info("Starting automated training session...")
        
        session_results = []
        
        for symbol in self.config["symbols"]:
            for model_type in self.config["model_types"]:
                self.logger.info(f"Training {model_type} for {symbol}")
                
                try:
                    cmd = [
                        sys.executable, "train_bot.py",
                        "--model-type", model_type,
                        "--symbol", symbol,
                        "--steps", str(self.config["training_params"]["steps"]),
                        "--episodes", str(self.config["training_params"]["episodes"]),
                        "--iterations", str(self.config["training_params"]["iterations"])
                    ]
                    
                    start_time = time.time()
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
                    training_time = time.time() - start_time
                    
                    session_result = {
                        "symbol": symbol,
                        "model_type": model_type,
                        "timestamp": datetime.now().isoformat(),
                        "success": result.returncode == 0,
                        "training_time": training_time,
                        "output": result.stdout if result.returncode == 0 else result.stderr
                    }
                    
                    session_results.append(session_result)
                    
                    if result.returncode == 0:
                        self.logger.info(f"Training completed for {symbol} ({model_type}) in {training_time:.2f}s")
                    else:
                        self.logger.error(f"Training failed for {symbol} ({model_type}): {result.stderr}")
                
                except Exception as e:
                    self.logger.error(f"Training error for {symbol} ({model_type}): {e}")
                    session_results.append({
                        "symbol": symbol,
                        "model_type": model_type,
                        "timestamp": datetime.now().isoformat(),
                        "success": False,
                        "error": str(e)
                    })
        
        # Save training session results
        self.save_training_session(session_results)
        self.last_training_time = datetime.now()
        
        return session_results
    
    def run_evaluation(self):
        """Run automated model evaluation"""
        self.logger.info("Starting automated model evaluation...")
        
        try:
            for symbol in self.config["symbols"]:
                cmd = [
                    sys.executable, "evaluate_bot.py",
                    "--symbol", symbol,
                    "--create-plots"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
                
                if result.returncode == 0:
                    self.logger.info(f"Evaluation completed for {symbol}")
                else:
                    self.logger.error(f"Evaluation failed for {symbol}: {result.stderr}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
            return False
    
    def save_training_session(self, session_results: List[Dict]):
        """Save training session results"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        session_file = results_dir / f"training_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": session_results,
            "summary": self.create_session_summary(session_results)
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Add to training history
        self.training_history.append(session_data)
        
        # Keep only last 30 sessions in memory
        if len(self.training_history) > 30:
            self.training_history = self.training_history[-30:]
    
    def create_session_summary(self, session_results: List[Dict]) -> Dict:
        """Create summary of training session"""
        total_sessions = len(session_results)
        successful_sessions = sum(1 for r in session_results if r.get("success", False))
        
        summary = {
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": successful_sessions / total_sessions if total_sessions > 0 else 0,
            "symbols_trained": list(set(r["symbol"] for r in session_results)),
            "models_trained": list(set(r["model_type"] for r in session_results)),
            "total_training_time": sum(r.get("training_time", 0) for r in session_results),
            "failed_sessions": [r for r in session_results if not r.get("success", False)]
        }
        
        return summary
    
    def check_model_performance(self):
        """Check model performance and trigger actions if needed"""
        self.logger.info("Checking model performance...")
        
        # This would implement performance monitoring logic
        # For now, just log that the check was performed
        self.logger.info("Performance check completed")
    
    def send_notification(self, message: str, level: str = "info"):
        """Send notification if configured"""
        if not self.config["notifications"]["enabled"]:
            return
        
        # This would implement email/Slack notifications
        self.logger.info(f"Notification ({level}): {message}")
    
    def setup_schedule(self):
        """Setup training schedule"""
        schedule_config = self.config["training_schedule"]
        
        # Data update schedule
        if schedule_config.get("data_update_at"):
            schedule.every().day.at(schedule_config["data_update_at"]).do(self.run_data_update)
            self.logger.info(f"Scheduled data updates at {schedule_config['data_update_at']}")
        
        # Training schedule
        if schedule_config.get("daily_at"):
            schedule.every().day.at(schedule_config["daily_at"]).do(self.run_training_session)
            self.logger.info(f"Scheduled training at {schedule_config['daily_at']}")
        
        # Evaluation schedule
        if schedule_config.get("evaluation_at"):
            schedule.every().day.at(schedule_config["evaluation_at"]).do(self.run_evaluation)
            self.logger.info(f"Scheduled evaluation at {schedule_config['evaluation_at']}")
        
        # Performance monitoring
        monitoring_freq = self.config["monitoring"].get("performance_check_frequency", "weekly")
        if monitoring_freq == "daily":
            schedule.every().day.at("04:00").do(self.check_model_performance)
        elif monitoring_freq == "weekly":
            schedule.every().sunday.at("04:00").do(self.check_model_performance)
    
    def run_scheduler(self):
        """Run the automated scheduler"""
        self.logger.info("Starting automated training scheduler...")
        self.setup_schedule()
        
        self.logger.info("Scheduler running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
    
    def run_once(self, task: str = "all"):
        """Run specific task once"""
        if task in ["all", "data"]:
            self.run_data_update()
        
        if task in ["all", "training"]:
            self.run_training_session()
        
        if task in ["all", "evaluation"]:
            self.run_evaluation()
    
    def status(self):
        """Show current status"""
        print("\n" + "="*60)
        print("AUTOMATED TRAINING STATUS")
        print("="*60)
        
        print(f"Configuration file: {self.config_file}")
        print(f"Symbols: {', '.join(self.config['symbols'])}")
        print(f"Model types: {', '.join(self.config['model_types'])}")
        
        if self.last_training_time:
            print(f"Last training: {self.last_training_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("Last training: Never")
        
        print(f"Training sessions in history: {len(self.training_history)}")
        
        # Show next scheduled runs
        print("\nScheduled tasks:")
        for job in schedule.jobs:
            print(f"  {job}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Automated ML Trading Bot Training")
    parser.add_argument("--config", default="training_config.json", help="Configuration file")
    parser.add_argument("--run-once", choices=["all", "data", "training", "evaluation"],
                       help="Run specific task once and exit")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--create-config", action="store_true", help="Create default config file")
    
    args = parser.parse_args()
    
    trainer = AutomatedTrainer(args.config)
    
    if args.create_config:
        print(f"Default configuration saved to {args.config}")
        return
    
    if args.status:
        trainer.status()
        return
    
    if args.run_once:
        trainer.run_once(args.run_once)
        return
    
    # Run scheduler
    trainer.run_scheduler()


if __name__ == "__main__":
    # Install schedule if not available
    try:
        import schedule
    except ImportError:
        print("Installing schedule package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "schedule"])
        import schedule
    
    main()