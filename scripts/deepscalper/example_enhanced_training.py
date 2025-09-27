#!/usr/bin/env python3
"""
Example of enhanced ML trading bot training with continuous learning capabilities.

This example demonstrates how to use the enhanced training system with
comprehensive monitoring, automated data pipeline, and intelligent retraining.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

try:
    from config import EnvConfig, TrainConfig  # type: ignore
    from data_pipeline import DataPipelineConfig, ContinuousDataPipeline  # type: ignore
    from continuous_trainer import ContinuousTrainingConfig, ContinuousTrainer, create_default_config  # type: ignore
    from monitoring import MetricsTracker  # type: ignore
except Exception:
    print("Import error - make sure you're running from the scripts/deepscalper directory")
    import sys
    sys.exit(1)


def example_basic_enhanced_training():
    """Example 1: Basic enhanced training with monitoring."""
    print("🚀 Example 1: Basic Enhanced Training")
    print("=" * 50)
    
    # Create configuration with enhanced settings
    config = create_default_config()
    
    # Customize for example (shorter intervals for demo)
    config.data_pipeline_config.symbols = ["AAPL"]  # Single symbol for demo
    config.data_pipeline_config.update_interval_minutes = 5  # More frequent for demo
    config.retraining_interval_hours = 1  # Shorter for demo
    config.train_config.train_steps = 1000  # Fewer steps for demo
    config.enable_incremental_learning = True
    config.incremental_steps = 200
    
    print(f"📊 Configuration:")
    print(f"  Symbols: {config.data_pipeline_config.symbols}")
    print(f"  Update Interval: {config.data_pipeline_config.update_interval_minutes} minutes")
    print(f"  Training Steps: {config.train_config.train_steps}")
    print(f"  Incremental Learning: {config.enable_incremental_learning}")
    
    # Create working directory
    work_dir = "example_enhanced_training"
    os.makedirs(work_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = ContinuousTrainer(config, work_dir)
    
    try:
        print("\n🔄 Starting continuous training...")
        trainer.start()
        
        # Let it run for a few minutes to demonstrate
        print("⏱️  Running for 2 minutes to demonstrate functionality...")
        time.sleep(120)
        
        # Check status
        status = trainer.get_status()
        print(f"\n📊 Training Status:")
        print(f"  Active: {status['active']}")
        print(f"  Current Version: {status['current_version']}")
        print(f"  Data Pipeline Updates: {status['data_pipeline_stats']['updates_processed']}")
        
        print("\n✅ Example completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    finally:
        print("🛑 Stopping training...")
        trainer.stop()


def example_monitoring_only():
    """Example 2: Using enhanced monitoring with existing training."""
    print("\n🚀 Example 2: Enhanced Monitoring Integration")
    print("=" * 50)
    
    # Create metrics tracker
    metrics_dir = "example_monitoring"
    os.makedirs(metrics_dir, exist_ok=True)
    
    tracker = MetricsTracker(metrics_dir)
    
    # Simulate training with enhanced metrics
    print("📊 Simulating training with enhanced metrics...")
    
    import numpy as np
    from monitoring import TrainingMetrics
    
    for step in range(100):
        # Simulate training metrics
        loss = 1.0 * np.exp(-step / 50) + np.random.normal(0, 0.1)
        episode_return = step * 0.1 + np.random.normal(0, 2)
        
        metrics = TrainingMetrics(
            step=step,
            timestamp=time.time(),
            loss=max(0, loss),
            q_loss=loss * 0.7,
            aux_loss=loss * 0.3,
            episode_return=episode_return,
            epsilon=max(0.05, 1.0 - step / 100),
            buffer_size=min(1000, step * 10),
            learning_rate=0.001,
            grad_norm=np.random.uniform(0.1, 2.0)
        )
        
        tracker.log_training_metrics(metrics)
        
        if step % 20 == 0:
            print(f"  Step {step}: Loss={loss:.4f}, Return={episode_return:.2f}")
    
    # Analyze trends
    print(f"\n📈 Trend Analysis:")
    print(f"  Loss Trend: {tracker.get_trend('loss')}")
    print(f"  Return Trend: {tracker.get_trend('episode_return')}")
    
    # Get recent performance
    avg_loss = tracker.get_running_average('loss', 20)
    avg_return = tracker.get_running_average('episode_return', 20)
    print(f"  Recent Avg Loss: {avg_loss:.4f}")
    print(f"  Recent Avg Return: {avg_return:.2f}")
    
    # Check for anomalies
    anomalies = tracker.detect_anomalies('loss')
    print(f"  Loss Anomalies: {len(anomalies)}")
    
    print("✅ Monitoring example completed!")


def example_data_pipeline():
    """Example 3: Standalone data pipeline usage."""
    print("\n🚀 Example 3: Data Pipeline Usage")
    print("=" * 50)
    
    # Create data pipeline configuration
    config = DataPipelineConfig(
        symbols=["AAPL"],
        update_interval_minutes=1,  # Very frequent for demo
        lookback_days=7,
        storage_dir="example_data_cache",
        max_cache_size_mb=100
    )
    
    print(f"📊 Data Pipeline Configuration:")
    print(f"  Symbols: {config.symbols}")
    print(f"  Update Interval: {config.update_interval_minutes} minutes")
    print(f"  Lookback Days: {config.lookback_days}")
    
    # Create pipeline
    pipeline = ContinuousDataPipeline(config)
    
    try:
        print("\n🔄 Starting data pipeline...")
        pipeline.start()
        
        # Let it collect some data
        print("⏱️  Collecting data for 30 seconds...")
        time.sleep(30)
        
        # Check statistics
        stats = pipeline.get_stats()
        print(f"\n📊 Pipeline Statistics:")
        print(f"  Updates Processed: {stats['updates_processed']}")
        print(f"  Cache Hits: {stats['cache_hits']}")
        print(f"  Cache Misses: {stats['cache_misses']}")
        print(f"  Quality Failures: {stats['quality_failures']}")
        
        # Try to get some data
        data = pipeline.get_latest_data("AAPL", lookback_minutes=60)
        if data:
            print(f"  Latest Data Shape: {data.df.shape}")
            print(f"  Data Columns: {list(data.df.columns)}")
        else:
            print("  No data available yet")
        
        print("✅ Data pipeline example completed!")
        
    except Exception as e:
        print(f"❌ Error in data pipeline: {e}")
    finally:
        print("🛑 Stopping data pipeline...")
        pipeline.stop()


def main():
    """Run all examples."""
    print("🎯 Enhanced ML Trading Bot Training Examples")
    print("=" * 60)
    
    try:
        # Run examples sequentially
        example_monitoring_only()
        example_data_pipeline()
        
        # Ask user if they want to run the full training example
        response = input("\n❓ Run full continuous training example? (takes ~2 minutes) [y/N]: ")
        if response.lower().startswith('y'):
            example_basic_enhanced_training()
        else:
            print("⏭️  Skipping full training example")
        
        print("\n🎉 All examples completed successfully!")
        print("\n💡 Next Steps:")
        print("  1. Try the CLI: python training_cli.py --help")
        print("  2. Start continuous training: python training_cli.py start")
        print("  3. Monitor progress: python training_cli.py status")
        print("  4. View metrics: python training_cli.py metrics")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())