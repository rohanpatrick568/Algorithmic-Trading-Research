#!/usr/bin/env python3
"""DeepScalper Bot Manager - CLI tool for managing the ML trading bot.

This script provides a unified interface for:
- Training the bot
- Running evaluations
- Managing model versions
- Monitoring performance
- Running hyperparameter optimization
- Managing the automated pipeline
"""

import argparse
import sys
from pathlib import Path

# Add the deepscalper module to path
sys.path.append(str(Path(__file__).parent))

try:
    from config import EnvConfig, TrainConfig
    from train import train_agent
    from data import load_minute_data
    from env import DeepScalperEnv
    from monitoring import PerformanceMonitor, evaluate_model
    from model import BranchingDuelingQNet
    from hyperopt import HyperparameterOptimizer, HyperparameterSpace
    from pipeline import TrainingPipeline, PipelineConfig, ModelRegistry, run_continuous_pipeline
    from sample_data import create_sample_dataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

import torch
import json
from datetime import datetime, timedelta


def cmd_train(args):
    """Train the bot with specified parameters."""
    print(f"[BotManager] Training {args.symbol} for {args.steps} steps")
    
    # Create configs
    env_config = EnvConfig(
        symbol=args.symbol,
        lookback=args.lookback,
        fee_rate=args.fee_rate
    )
    
    train_config = TrainConfig(
        train_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size
    )
    
    # Load data
    if args.start and args.end:
        md = load_minute_data(args.symbol, args.start, args.end)
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        md = load_minute_data(args.symbol, start_date, end_date)
    
    # Create environment and train
    env = DeepScalperEnv(md, env_config, train_config)
    train_agent(env, train_config, args.checkpoint_dir, resume=(not args.no_resume), 
               save_every=args.save_every)
    
    print("[BotManager] Training completed")


def cmd_evaluate(args):
    """Evaluate a trained model."""
    print(f"[BotManager] Evaluating model from {args.checkpoint_dir}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment for evaluation
    env_config = EnvConfig(symbol=args.symbol)
    train_config = TrainConfig()
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    md = load_minute_data(args.symbol, start_date, end_date)
    env = DeepScalperEnv(md, env_config, train_config)
    
    # Load trained model
    model = BranchingDuelingQNet(env.obs_dim, env_config.price_bins, env_config.qty_bins).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    from train import _load_latest
    steps_loaded = _load_latest(args.checkpoint_dir, model, optimizer, map_location=device)
    
    if steps_loaded == 0:
        print("[BotManager] No trained model found")
        return
    
    # Evaluate
    metrics = evaluate_model(env, model, device, episodes=args.episodes)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Model trained for {steps_loaded:,} steps")
    print(f"Average Episode Return: ${metrics.get('avg_episode_return', 0):,.2f}")
    print(f"Average Sharpe Ratio: {metrics.get('avg_sharpe_ratio', 0):.3f}")
    print(f"Average Max Drawdown: {metrics.get('avg_max_drawdown', 0)*100:.2f}%")
    print(f"Average Volatility: {metrics.get('avg_volatility', 0)*100:.2f}%")
    print(f"Max Position: {metrics.get('avg_max_position', 0):.2f}")


def cmd_hyperopt(args):
    """Run hyperparameter optimization."""
    print(f"[BotManager] Running hyperparameter optimization with {args.trials} trials")
    
    # Base configs
    base_env = EnvConfig(symbol=args.symbol)
    base_train = TrainConfig(train_steps=args.steps_per_trial)
    search_space = HyperparameterSpace()
    
    # Run optimization
    optimizer = HyperparameterOptimizer(base_env, base_train, search_space, args.results_dir)
    best_config = optimizer.optimize(
        n_trials=args.trials, 
        train_steps_per_trial=args.steps_per_trial,
        eval_episodes=args.eval_episodes
    )
    
    if best_config:
        print(f"\n=== Best Configuration Found ===")
        print(f"Learning Rate: {best_config.lr:.5f}")
        print(f"Batch Size: {best_config.batch_size}")
        print(f"Buffer Size: {best_config.buffer_size:,}")
        print(f"Lookback: {best_config.lookback}")
        print(f"Fee Rate: {best_config.fee_rate:.4f}")
        print(f"Trial ID: {best_config.trial_id}")
        
        print(optimizer.generate_report())
    else:
        print("[BotManager] Hyperparameter optimization failed")


def cmd_pipeline(args):
    """Manage the automated training pipeline."""
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        training_interval_hours=args.training_interval,
        steps_per_training=args.steps_per_training,
        hyperopt_interval_hours=args.hyperopt_interval,
        hyperopt_trials=args.hyperopt_trials
    )
    
    base_env = EnvConfig(symbol=args.symbol)
    base_train = TrainConfig()
    
    pipeline = TrainingPipeline(base_env, base_train, pipeline_config, args.workspace_dir)
    
    if args.action == "start":
        print(f"[BotManager] Starting continuous pipeline (check every {args.check_interval} minutes)")
        run_continuous_pipeline(pipeline, args.check_interval, args.max_iterations)
        
    elif args.action == "run-once":
        print("[BotManager] Running single pipeline cycle")
        results = pipeline.run_pipeline_cycle()
        print("Results:", json.dumps(results, indent=2))
        
    elif args.action == "status":
        print(pipeline.generate_status_report())
        
    elif args.action == "models":
        history = pipeline.registry.get_model_history()
        if not history.empty:
            print("\n=== Model History ===")
            for _, row in history.iterrows():
                status = "PROD" if row['is_production'] else ""
                print(f"{row['version_id']} | {row['created_at'][:19]} | Steps: {row['training_steps']:6,} | Score: {row.get('composite_score', 0):6.3f} {status}")
        else:
            print("No models in registry")


def cmd_monitor(args):
    """Monitor bot performance."""
    monitor = PerformanceMonitor(args.log_dir)
    monitor.load_history()
    
    if args.action == "report":
        print(monitor.generate_report())
        
    elif args.action == "recent":
        recent = monitor.get_recent_performance(args.episodes)
        print(f"\n=== Recent Performance (Last {args.episodes} Episodes) ===")
        print(f"Average Return: ${recent.get('avg_return', 0):,.2f}")
        print(f"Win Rate: {recent.get('win_rate', 0)*100:.1f}%")
        print(f"Average Sharpe: {recent.get('avg_sharpe', 0):.3f}")
        print(f"Consistency: {recent.get('consistency', 0)*100:.1f}%")
        print(f"Trend: {recent.get('trend', 0)*100:+.1f}%")


def cmd_data(args):
    """Manage training data."""
    if args.action == "generate":
        print(f"[BotManager] Generating sample data for {args.symbol}")
        data = create_sample_dataset(args.symbol, args.days, args.output)
        print(f"Generated {len(data.df)} bars")
        print(f"Features: {list(data.df.columns)}")
        print(f"Price range: ${data.df['close'].min():.2f} - ${data.df['close'].max():.2f}")
        
    elif args.action == "download":
        print(f"[BotManager] Downloading data for {args.symbol}")
        try:
            md = load_minute_data(args.symbol, args.start, args.end)
            print(f"Downloaded {len(md.df)} bars")
            if args.output:
                md.df.to_csv(args.output)
                print(f"Saved to {args.output}")
        except Exception as e:
            print(f"Download failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="DeepScalper Bot Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the bot')
    train_parser.add_argument('--symbol', default='AAPL', help='Trading symbol')
    train_parser.add_argument('--steps', type=int, default=10000, help='Training steps')
    train_parser.add_argument('--days', type=int, default=30, help='Days of training data')
    train_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    train_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    train_parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--save-every', type=int, default=2000, help='Save frequency')
    train_parser.add_argument('--no-resume', action='store_true', help='Start fresh')
    train_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    train_parser.add_argument('--buffer-size', type=int, default=500000, help='Replay buffer size')
    train_parser.add_argument('--lookback', type=int, default=120, help='Lookback window')
    train_parser.add_argument('--fee-rate', type=float, default=0.0005, help='Trading fee rate')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--symbol', default='AAPL', help='Trading symbol')
    eval_parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Evaluation episodes')
    eval_parser.add_argument('--days', type=int, default=10, help='Days of evaluation data')
    
    # Hyperopt command
    hyperopt_parser = subparsers.add_parser('hyperopt', help='Run hyperparameter optimization')
    hyperopt_parser.add_argument('--symbol', default='AAPL', help='Trading symbol')
    hyperopt_parser.add_argument('--trials', type=int, default=20, help='Number of trials')
    hyperopt_parser.add_argument('--steps-per-trial', type=int, default=5000, help='Training steps per trial')
    hyperopt_parser.add_argument('--eval-episodes', type=int, default=5, help='Evaluation episodes per trial')
    hyperopt_parser.add_argument('--results-dir', default='./hyperopt_results', help='Results directory')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Manage automated pipeline')
    pipeline_parser.add_argument('action', choices=['start', 'run-once', 'status', 'models'], 
                                help='Pipeline action')
    pipeline_parser.add_argument('--symbol', default='AAPL', help='Trading symbol')
    pipeline_parser.add_argument('--workspace-dir', default='./pipeline_workspace', help='Workspace directory')
    pipeline_parser.add_argument('--training-interval', type=int, default=6, help='Training interval (hours)')
    pipeline_parser.add_argument('--steps-per-training', type=int, default=10000, help='Steps per training')
    pipeline_parser.add_argument('--hyperopt-interval', type=int, default=72, help='Hyperopt interval (hours)')
    pipeline_parser.add_argument('--hyperopt-trials', type=int, default=20, help='Hyperopt trials')
    pipeline_parser.add_argument('--check-interval', type=int, default=60, help='Check interval (minutes)')
    pipeline_parser.add_argument('--max-iterations', type=int, help='Max iterations (for testing)')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor performance')
    monitor_parser.add_argument('action', choices=['report', 'recent'], help='Monitor action')
    monitor_parser.add_argument('--log-dir', default='./logs', help='Log directory')
    monitor_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for recent')
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Manage training data')
    data_parser.add_argument('action', choices=['generate', 'download'], help='Data action')
    data_parser.add_argument('--symbol', default='AAPL', help='Trading symbol')
    data_parser.add_argument('--days', type=int, default=30, help='Days of data')
    data_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    data_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    data_parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'hyperopt':
        cmd_hyperopt(args)
    elif args.command == 'pipeline':
        cmd_pipeline(args)
    elif args.command == 'monitor':
        cmd_monitor(args)
    elif args.command == 'data':
        cmd_data(args)


if __name__ == "__main__":
    main()