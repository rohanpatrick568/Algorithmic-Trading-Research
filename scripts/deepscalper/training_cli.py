#!/usr/bin/env python3
"""
Command-line interface for managing continuous ML trading bot training.

This CLI provides comprehensive controls for training, monitoring, and managing
the ML trading bot with continuous learning capabilities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

try:
    from .continuous_trainer import ContinuousTrainer, ContinuousTrainingConfig, create_default_config  # type: ignore
    from .config import EnvConfig, TrainConfig  # type: ignore
    from .data_pipeline import DataPipelineConfig  # type: ignore
    from .monitoring import MetricsTracker, calculate_performance_metrics  # type: ignore
except Exception:
    from continuous_trainer import ContinuousTrainer, ContinuousTrainingConfig, create_default_config  # type: ignore
    from config import EnvConfig, TrainConfig  # type: ignore
    from data_pipeline import DataPipelineConfig  # type: ignore
    from monitoring import MetricsTracker, calculate_performance_metrics  # type: ignore


class TrainingCLI:
    """Command-line interface for continuous training management."""
    
    def __init__(self, work_dir: str = "continuous_training"):
        self.work_dir = Path(work_dir)
        self.trainer = None
        self.config_file = self.work_dir / "config.json"
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Continuous ML Trading Bot Training CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Start continuous training
  python training_cli.py start --symbols AAPL MSFT GOOGL
  
  # Check status
  python training_cli.py status
  
  # View training metrics
  python training_cli.py metrics --last 1000
  
  # Trigger manual retraining
  python training_cli.py retrain
  
  # Export model for deployment
  python training_cli.py export --version latest --output model.pt
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Start command
        start_parser = subparsers.add_parser('start', help='Start continuous training')
        start_parser.add_argument('--symbols', nargs='+', default=['AAPL'], 
                                 help='Trading symbols to monitor')
        start_parser.add_argument('--config', type=str, help='Configuration file path')
        start_parser.add_argument('--update-interval', type=int, default=60,
                                 help='Data update interval in minutes')
        start_parser.add_argument('--retrain-interval', type=int, default=24,
                                 help='Minimum retraining interval in hours')
        start_parser.add_argument('--performance-threshold', type=float, default=-0.1,
                                 help='Performance threshold for retraining trigger')
        
        # Stop command
        subparsers.add_parser('stop', help='Stop continuous training')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show training status')
        status_parser.add_argument('--detailed', action='store_true', 
                                  help='Show detailed status information')
        
        # Metrics command
        metrics_parser = subparsers.add_parser('metrics', help='Show training metrics')
        metrics_parser.add_argument('--last', type=int, default=100,
                                   help='Number of recent metrics to show')
        metrics_parser.add_argument('--export', type=str, help='Export metrics to CSV file')
        metrics_parser.add_argument('--plot', action='store_true', help='Generate plots')
        
        # Models command
        models_parser = subparsers.add_parser('models', help='Manage model versions')
        models_parser.add_argument('--list', action='store_true', help='List all model versions')
        models_parser.add_argument('--activate', type=str, help='Activate a specific version')
        models_parser.add_argument('--compare', nargs=2, help='Compare two model versions')
        models_parser.add_argument('--cleanup', action='store_true', help='Remove old model versions')
        
        # Retrain command
        retrain_parser = subparsers.add_parser('retrain', help='Trigger manual retraining')
        retrain_parser.add_argument('--full', action='store_true', 
                                   help='Force full retraining instead of incremental')
        retrain_parser.add_argument('--symbol', type=str, help='Specific symbol to retrain on')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export model for deployment')
        export_parser.add_argument('--version', type=str, default='latest',
                                  help='Model version to export (default: latest)')
        export_parser.add_argument('--output', type=str, required=True,
                                  help='Output file path')
        export_parser.add_argument('--format', choices=['pytorch', 'onnx'], default='pytorch',
                                  help='Export format')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_parser.add_argument('--show', action='store_true', help='Show current configuration')
        config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'),
                                  help='Set configuration parameter')
        config_parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
        
        # Dashboard command
        dashboard_parser = subparsers.add_parser('dashboard', help='Launch monitoring dashboard')
        dashboard_parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
        dashboard_parser.add_argument('--host', type=str, default='localhost', help='Dashboard host')
        
        return parser
    
    def load_config(self, config_path: Optional[str] = None) -> ContinuousTrainingConfig:
        """Load configuration from file or create default."""
        config_file = Path(config_path) if config_path else self.config_file
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Reconstruct config objects
                env_config = EnvConfig(**config_data.get('env_config', {}))
                train_config = TrainConfig(**config_data.get('train_config', {}))
                data_config = DataPipelineConfig(**config_data.get('data_pipeline_config', {}))
                
                # Create main config
                main_config_data = config_data.get('main_config', {})
                config = ContinuousTrainingConfig(
                    env_config=env_config,
                    train_config=train_config,
                    data_pipeline_config=data_config,
                    **main_config_data
                )
                
                return config
                
            except Exception as e:
                print(f"Warning: Failed to load config from {config_file}: {e}")
                print("Using default configuration")
        
        return create_default_config()
    
    def save_config(self, config: ContinuousTrainingConfig):
        """Save configuration to file."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'env_config': config.env_config.__dict__,
            'train_config': config.train_config.__dict__,
            'data_pipeline_config': config.data_pipeline_config.__dict__,
            'main_config': {
                k: v for k, v in config.__dict__.items() 
                if k not in ['env_config', 'train_config', 'data_pipeline_config']
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def cmd_start(self, args):
        """Start continuous training."""
        print("🚀 Starting continuous ML trading bot training...")
        
        # Load or create configuration
        config = self.load_config(args.config)
        
        # Update config with command line arguments
        config.data_pipeline_config.symbols = args.symbols
        config.data_pipeline_config.update_interval_minutes = args.update_interval
        config.retraining_interval_hours = args.retrain_interval
        config.performance_threshold = args.performance_threshold
        
        # Save updated configuration
        self.save_config(config)
        
        # Create and start trainer
        self.trainer = ContinuousTrainer(config, str(self.work_dir))
        self.trainer.start()
        
        print(f"✅ Continuous training started with symbols: {args.symbols}")
        print(f"📊 Monitoring dashboard available at http://localhost:8080")
        print(f"📁 Working directory: {self.work_dir}")
        print(f"⏰ Data updates every {args.update_interval} minutes")
        print(f"🔄 Retraining interval: {args.retrain_interval} hours")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(60)
                status = self.trainer.get_status()
                if not status['active']:
                    break
        except KeyboardInterrupt:
            print("\n⏹️  Stopping continuous training...")
            self.trainer.stop()
            print("✅ Training stopped successfully")
    
    def cmd_stop(self, args):
        """Stop continuous training."""
        print("⏹️  Stopping continuous training...")
        
        # Try to find running trainer instance
        if self.trainer:
            self.trainer.stop()
        else:
            # Could implement process-based stopping here
            print("No active trainer instance found")
        
        print("✅ Training stopped")
    
    def cmd_status(self, args):
        """Show training status."""
        if not self.work_dir.exists():
            print("❌ No training session found")
            return
        
        # Load trainer to get status
        config = self.load_config()
        trainer = ContinuousTrainer(config, str(self.work_dir))
        status = trainer.get_status()
        
        print("📊 Continuous Training Status")
        print("=" * 50)
        print(f"Active: {'✅ Yes' if status['active'] else '❌ No'}")
        print(f"Current Model Version: {status['current_version'] or 'None'}")
        print(f"Last Retraining: {status['last_retrain'] or 'Never'}")
        print(f"Incremental Updates: {status['incremental_updates']}")
        print(f"Model Versions: {status['model_versions']}")
        print(f"Retraining Events: {status['retraining_events']}")
        
        if args.detailed:
            print("\n📈 Data Pipeline Stats:")
            data_stats = status['data_pipeline_stats']
            print(f"  Updates Processed: {data_stats['updates_processed']}")
            print(f"  Quality Failures: {data_stats['quality_failures']}")
            print(f"  Cache Hits/Misses: {data_stats['cache_hits']}/{data_stats['cache_misses']}")
            print(f"  Last Update: {data_stats['last_update']}")
            
            if data_stats['errors']:
                print(f"  Recent Errors: {len(data_stats['errors'])}")
    
    def cmd_metrics(self, args):
        """Show training metrics."""
        metrics_dir = self.work_dir / "metrics"
        if not metrics_dir.exists():
            print("❌ No metrics found")
            return
        
        tracker = MetricsTracker(str(metrics_dir))
        recent_metrics = tracker.get_recent_metrics(args.last)
        
        if not recent_metrics:
            print("❌ No training metrics available")
            return
        
        print(f"📊 Training Metrics (Last {len(recent_metrics)} steps)")
        print("=" * 60)
        
        # Summary statistics
        losses = [m.loss for m in recent_metrics if m.loss is not None]
        returns = [m.episode_return for m in recent_metrics if m.episode_return is not None]
        
        if losses:
            print(f"Loss: {np.mean(losses):.6f} ± {np.std(losses):.6f}")
        if returns:
            print(f"Episode Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        
        # Recent trend analysis
        if len(recent_metrics) >= 10:
            return_trend = tracker.get_trend('episode_return')
            loss_trend = tracker.get_trend('loss')
            print(f"Return Trend: {return_trend}")
            print(f"Loss Trend: {loss_trend}")
        
        # Export to CSV if requested
        if args.export:
            df_data = []
            for m in recent_metrics:
                df_data.append(m.to_dict())
            
            df = pd.DataFrame(df_data)
            df.to_csv(args.export, index=False)
            print(f"📄 Metrics exported to {args.export}")
        
        # Generate plots if requested
        if args.plot:
            self._generate_metric_plots(recent_metrics)
    
    def cmd_models(self, args):
        """Manage model versions."""
        config = self.load_config()
        trainer = ContinuousTrainer(config, str(self.work_dir))
        
        if args.list:
            versions = trainer.model_manager.versions['versions']
            active_version = trainer.model_manager.get_active_version()
            
            print("📦 Model Versions")
            print("=" * 50)
            
            for version in sorted(versions, key=lambda v: v['created_at'], reverse=True):
                status = "🟢 ACTIVE" if version['version_id'] == active_version else "⚪"
                print(f"{status} {version['version_id']}")
                print(f"   Created: {version['created_at']}")
                print(f"   Trigger: {version['metadata'].get('trigger', 'Unknown')}")
                
                if version.get('performance'):
                    perf = version['performance']
                    print(f"   Performance: {perf}")
                print()
        
        elif args.activate:
            if trainer.model_manager.set_active_version(args.activate):
                print(f"✅ Activated model version: {args.activate}")
            else:
                print(f"❌ Failed to activate version: {args.activate}")
        
        elif args.compare:
            # Model comparison would be implemented here
            print(f"🔍 Comparing models {args.compare[0]} vs {args.compare[1]}")
            print("(Model comparison feature not yet implemented)")
        
        elif args.cleanup:
            # Cleanup old versions beyond the limit
            print("🧹 Cleaning up old model versions...")
            trainer.model_manager._cleanup_old_versions()
            print("✅ Cleanup completed")
    
    def cmd_retrain(self, args):
        """Trigger manual retraining."""
        config = self.load_config()
        trainer = ContinuousTrainer(config, str(self.work_dir))
        
        print("🔄 Triggering manual retraining...")
        
        if args.full:
            # Force full retraining
            trainer.incremental_update_count = trainer.config.full_retrain_frequency
        
        trainer.trigger_manual_retraining()
        print("✅ Retraining triggered successfully")
    
    def cmd_export(self, args):
        """Export model for deployment."""
        config = self.load_config()
        trainer = ContinuousTrainer(config, str(self.work_dir))
        
        version = args.version if args.version != 'latest' else trainer.model_manager.get_active_version()
        
        if not version:
            print("❌ No model version available for export")
            return
        
        print(f"📦 Exporting model version {version}...")
        
        # Load model
        if trainer.current_model is None:
            trainer._initialize_model()
        
        if not trainer.model_manager.load_version(version, trainer.current_model):
            print(f"❌ Failed to load model version {version}")
            return
        
        output_path = Path(args.output)
        
        try:
            if args.format == 'pytorch':
                # Save PyTorch model
                torch.save({
                    'model_state_dict': trainer.current_model.state_dict(),
                    'model_config': {
                        'obs_dim': trainer.current_model.trunk.net[0].in_features,
                        'price_bins': config.env_config.price_bins,
                        'qty_bins': config.env_config.qty_bins
                    },
                    'version': version,
                    'export_timestamp': datetime.now()
                }, output_path)
                
            elif args.format == 'onnx':
                # ONNX export (requires torch and onnx)
                import torch
                dummy_input = torch.randn(1, trainer.current_model.trunk.net[0].in_features)
                torch.onnx.export(
                    trainer.current_model, 
                    dummy_input, 
                    output_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['observation'],
                    output_names=['q_price', 'q_qty', 'value', 'aux']
                )
            
            print(f"✅ Model exported to {output_path}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def cmd_config(self, args):
        """Configuration management."""
        config = self.load_config()
        
        if args.show:
            print("⚙️  Current Configuration")
            print("=" * 50)
            print(json.dumps({
                'env_config': config.env_config.__dict__,
                'train_config': config.train_config.__dict__,
                'data_pipeline_config': config.data_pipeline_config.__dict__
            }, indent=2, default=str))
        
        elif args.set:
            key, value = args.set
            # Simple key-value setting (could be enhanced)
            print(f"Setting {key} = {value}")
            print("(Configuration editing not yet fully implemented)")
        
        elif args.reset:
            config = create_default_config()
            self.save_config(config)
            print("✅ Configuration reset to defaults")
    
    def cmd_dashboard(self, args):
        """Launch monitoring dashboard."""
        print(f"🌐 Launching dashboard at http://{args.host}:{args.port}")
        print("(Dashboard feature not yet implemented)")
        print("For now, use 'metrics' and 'status' commands for monitoring")
    
    def _generate_metric_plots(self, metrics):
        """Generate metric plots."""
        print("📊 Generating metric plots...")
        print("(Plotting feature requires matplotlib - not yet implemented)")
    
    def run(self, args: List[str] = None):
        """Run the CLI with given arguments."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return
        
        # Route to appropriate command handler
        command_method = getattr(self, f'cmd_{parsed_args.command}', None)
        if command_method:
            try:
                command_method(parsed_args)
            except Exception as e:
                print(f"❌ Command failed: {e}")
                return 1
        else:
            print(f"❌ Unknown command: {parsed_args.command}")
            return 1
        
        return 0


def main():
    """Main entry point."""
    cli = TrainingCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())