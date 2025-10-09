# ML Trading Bot Training and Development Guide

This guide provides comprehensive instructions for training and continuously developing the ML trading bot using the automated systems in this repository.

## Overview

The repository now includes three main automation scripts for ML trading bot development:

1. **`train_bot.py`** - Main training and continuous development script
2. **`evaluate_bot.py`** - Model evaluation and performance monitoring
3. **`data_pipeline.py`** - Data collection, preprocessing, and preparation

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install scikit-learn  # Additional dependency for data preprocessing
```

### 2. Update Market Data
```bash
python data_pipeline.py --symbols AAPL GOOGL MSFT --update-data --lookback-days 365
```

### 3. Train DeepScalper Model
```bash
python train_bot.py --model-type deepscalper --symbol AAPL --steps 10000 --iterations 1
```

### 4. Evaluate Models
```bash
python evaluate_bot.py --symbol AAPL --create-plots
```

## Detailed Usage

### Training Bot (`train_bot.py`)

The training bot provides a unified interface for training both DeepScalper and Backtrader RL models.

#### Basic Training
```bash
# Train DeepScalper model
python train_bot.py --model-type deepscalper --symbol AAPL --steps 20000

# Train Backtrader RL model
python train_bot.py --model-type backtrader --symbol AAPL --episodes 500

# Train both models
python train_bot.py --model-type both --symbol AAPL
```

#### Advanced Training Options
```bash
# Multiple training iterations
python train_bot.py --model-type deepscalper --iterations 5 --steps 5000

# Custom date range
python train_bot.py --symbol AAPL --start-date 2023-01-01 --end-date 2024-01-01

# Fresh training (ignore existing checkpoints)
python train_bot.py --no-resume --steps 10000

# Hyperparameter optimization
python train_bot.py --optimize-hyperparams --symbol AAPL

# Continuous training loop (runs indefinitely with 1-hour intervals)
python train_bot.py --continuous --model-type deepscalper
```

#### Key Features

- **Automated checkpoint management** - Backs up existing models before training
- **Progress logging** - Detailed logs saved to `logs/` directory
- **Model comparison** - Evaluates models between training iterations
- **Hyperparameter optimization** - Grid search for optimal parameters
- **Continuous training** - Long-running training sessions with periodic updates

### Data Pipeline (`data_pipeline.py`)

Manages data collection, preprocessing, and preparation for ML training.

#### Data Collection
```bash
# Update data for multiple symbols
python data_pipeline.py --symbols AAPL GOOGL MSFT TSLA --update-data

# Specify lookback period and intervals
python data_pipeline.py --symbols AAPL --update-data --lookback-days 730 --intervals 1d 1h

# Show data summary
python data_pipeline.py --summary
```

#### Training Data Preparation
```bash
# Prepare datasets for ML training
python data_pipeline.py --symbols AAPL GOOGL --prepare-training --lookback-window 50

# Custom data directory
python data_pipeline.py --data-dir custom_data --update-data --symbols AAPL
```

#### Features Created

The data pipeline automatically creates:
- **Technical indicators**: SMA, EMA, MACD, RSI, Bollinger Bands, ATR
- **Volume indicators**: OBV, volume ratios
- **Lagged features**: Historical price and volume data
- **Rolling statistics**: Mean, standard deviation over various windows
- **Target variables**: Next-period returns and direction classification
- **Normalized features**: MinMax or StandardScaler normalization

### Model Evaluation (`evaluate_bot.py`)

Comprehensive model evaluation and performance monitoring.

#### Basic Evaluation
```bash
# Evaluate all available models
python evaluate_bot.py --symbol AAPL --create-plots

# Evaluate specific model
python evaluate_bot.py --model-path models/bdq_test.pt --symbol AAPL

# Custom evaluation period
python evaluate_bot.py --start-date 2024-01-01 --end-date 2024-06-30
```

#### Performance Metrics Calculated

- **Returns**: Total return, annualized return
- **Risk**: Volatility, maximum drawdown, VaR
- **Risk-adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trading**: Win rate, profit factor, average trade
- **Advanced**: Number of trades, best/worst trades

#### Visualizations Generated

- Equity curves comparison
- Return distributions
- Risk-return scatter plots
- Key performance ratios
- Model comparison charts

## Model Types

### DeepScalper
- **Architecture**: Branching Dueling Q-Network (BDQ)
- **Strategy**: Intraday 1-minute trading with discrete price/quantity actions
- **Features**: Hindsight bonus, auxiliary volatility prediction
- **Training**: Prioritized experience replay, soft target updates

### Backtrader RL
- **Architecture**: Custom Branching DQN
- **Strategy**: Daily trading with position sizing
- **Features**: Transaction costs, reward engineering
- **Training**: Experience replay, epsilon-greedy exploration

## Directory Structure

```
├── train_bot.py              # Main training script
├── evaluate_bot.py           # Model evaluation
├── data_pipeline.py          # Data management
├── logs/                     # Training and evaluation logs
├── data/                     # Market data and processed datasets
├── results/                  # Evaluation results and plots
├── models/                   # Saved model files
│   ├── bdq.pt               # Pre-trained DeepScalper model
│   ├── bdq_test.pt          # Test model
│   └── backup_*/            # Automatic backups
└── scripts/
    ├── deepscalper/         # DeepScalper implementation
    │   ├── checkpoints/     # Training checkpoints
    │   └── main.py          # DeepScalper training entry
    └── backtrader/
        └── rl/              # Backtrader RL implementation
```

## Configuration

### Training Configuration

Default training parameters can be modified in the scripts or through command-line arguments:

```python
config = {
    "lr": 1e-4,              # Learning rate
    "batch_size": 64,        # Batch size
    "window": 30,            # Lookback window
    "replay_size": 10000,    # Replay buffer size
    "save_every": 2000,      # Checkpoint frequency
    "gamma": 0.99,           # Discount factor
    "epsilon_decay": 0.995   # Exploration decay
}
```

### Data Configuration

```python
data_config = {
    "lookback_days": 365,     # Historical data period
    "lookback_window": 30,    # Feature window
    "train_split": 0.8,       # Train/validation split
    "intervals": ["1d"],      # Data intervals
    "normalize": "minmax"     # Normalization method
}
```

## Best Practices

### Training Workflow

1. **Data Preparation**
   ```bash
   python data_pipeline.py --symbols AAPL --update-data --prepare-training
   ```

2. **Initial Training**
   ```bash
   python train_bot.py --model-type deepscalper --steps 10000 --no-resume
   ```

3. **Model Evaluation**
   ```bash
   python evaluate_bot.py --create-plots
   ```

4. **Hyperparameter Tuning**
   ```bash
   python train_bot.py --optimize-hyperparams
   ```

5. **Continuous Development**
   ```bash
   python train_bot.py --continuous --model-type deepscalper
   ```

### Performance Monitoring

- Monitor training logs in `logs/` directory
- Review evaluation reports in `results/` directory
- Check model backups in `models/backup_*/` directories
- Analyze performance plots for model comparison

### Data Management

- Update data regularly for fresh training data
- Use appropriate lookback periods (365+ days recommended)
- Prepare separate datasets for different symbols
- Monitor data quality and handle missing values

## Troubleshooting

### Common Issues

1. **Network connectivity issues**: Data fetching may fail due to network restrictions
2. **Memory issues**: Large datasets may require more RAM
3. **CUDA issues**: GPU training requires proper CUDA setup
4. **Import errors**: Ensure all dependencies are installed

### Solutions

1. **Use synthetic data** for testing when network is unavailable
2. **Reduce batch sizes** or use data streaming for large datasets
3. **Use CPU training** if GPU setup is problematic
4. **Check Python path** and virtual environment setup

### Logging

All scripts provide detailed logging:
- Training logs: `logs/training_YYYYMMDD_HHMMSS.log`
- Data pipeline logs: `logs/data_pipeline_YYYYMMDD.log`
- Evaluation results: `results/model_comparison_YYYYMMDD_HHMMSS.json`

## Advanced Usage

### Custom Model Development

1. **Add new model types** by extending the trainer classes
2. **Implement custom metrics** in the evaluation framework
3. **Create specialized data preprocessing** for specific strategies
4. **Add new visualization types** for performance analysis

### Integration with Existing Systems

The training system is designed to work with:
- **Existing DeepScalper implementation** in `scripts/deepscalper/`
- **Backtrader RL system** in `scripts/backtrader/rl/`
- **Lumibot strategies** for live trading
- **External data sources** through the data pipeline

### Production Deployment

For production use:
1. Set up automated data updates (cron jobs)
2. Implement model validation pipelines
3. Add performance monitoring and alerts
4. Use version control for model management
5. Implement A/B testing for model comparison

## Next Steps

1. **Expand model types**: Add more RL algorithms and traditional ML models
2. **Improve data sources**: Add more market data providers
3. **Enhance evaluation**: Add more sophisticated performance metrics
4. **Implement ensemble methods**: Combine multiple models for better performance
5. **Add risk management**: Implement position sizing and risk controls
6. **Create web interface**: Build a dashboard for monitoring and control

This guide provides the foundation for systematic ML trading bot development. The automated systems ensure consistent, repeatable training processes while maintaining comprehensive evaluation and monitoring capabilities.