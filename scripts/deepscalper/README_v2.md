# DeepScalper (RL) – Advanced ML Trading Bot

A comprehensive machine learning trading bot with automated training, hyperparameter optimization, performance monitoring, and continuous development pipeline.

## 🚀 New Features (v2.0)

### Offline Training Capability
- **Synthetic Data Generation**: Train without network dependencies using realistic market data simulation
- **Fallback System**: Automatically switches to synthetic data when market data unavailable
- **Multiple Market Patterns**: Different volatility and trend patterns for various symbols

### Performance Analytics
- **Real-time Monitoring**: Track training progress with detailed metrics
- **Risk Assessment**: Sharpe ratio, max drawdown, volatility analysis
- **Performance Trends**: Identify improving/declining model performance
- **Automated Alerts**: Get notified of performance issues

### Hyperparameter Optimization
- **Automated Tuning**: Find optimal learning rates, batch sizes, network parameters
- **Multiple Strategies**: Random search, grid search, Bayesian optimization
- **Configuration Management**: Save and reuse best hyperparameter sets
- **Trial Tracking**: Complete history of optimization attempts

### Automated Pipeline
- **Continuous Training**: Scheduled model retraining with fresh data
- **Model Versioning**: Automatic model registry with performance tracking
- **Production Deployment**: Smart model promotion based on performance
- **A/B Testing**: Compare model versions safely

### CLI Management Tool
- **Unified Interface**: Single command-line tool for all operations
- **Easy Monitoring**: Check performance, view reports, manage models
- **Pipeline Control**: Start/stop automated training, view status
- **Data Management**: Generate synthetic data, download market data

## 📊 What's Inside

Core Components:
- **BDQ Neural Network**: Branching Dueling Q-Network with auxiliary risk prediction
- **Gymnasium Environment**: 1-minute bar trading simulation with realistic fills
- **Prioritized Replay**: Experience replay with importance sampling
- **Risk-Aware Training**: Hindsight bonus and volatility prediction auxiliary task

New Modules:
- `sample_data.py` – Synthetic market data generation
- `monitoring.py` – Performance tracking and evaluation
- `hyperopt.py` – Hyperparameter optimization system
- `pipeline.py` – Automated training and model management
- `bot_manager.py` – CLI management interface

## 🛠 Setup

### Requirements
```bash
pip install torch gymnasium yfinance pandas numpy lumibot matplotlib seaborn
```

### Quick Start
```bash
# Train the bot (uses synthetic data if network unavailable)
python scripts/deepscalper/bot_manager.py train --symbol AAPL --steps 10000

# Evaluate performance
python scripts/deepscalper/bot_manager.py evaluate --symbol AAPL

# Run hyperparameter optimization
python scripts/deepscalper/bot_manager.py hyperopt --trials 10

# Start automated pipeline
python scripts/deepscalper/bot_manager.py pipeline start
```

## 📈 Training & Development

### Basic Training
```bash
# Train with default settings
python -m scripts.deepscalper.main --symbol AAPL --steps 50000

# Train with custom parameters
python scripts/deepscalper/bot_manager.py train \
    --symbol TSLA \
    --steps 20000 \
    --lr 0.001 \
    --batch-size 512 \
    --lookback 180
```

### Performance Monitoring
```bash
# View training report
python scripts/deepscalper/bot_manager.py monitor report

# Check recent performance
python scripts/deepscalper/bot_manager.py monitor recent --episodes 20
```

### Hyperparameter Optimization
```bash
# Run optimization with 50 trials
python scripts/deepscalper/bot_manager.py hyperopt \
    --trials 50 \
    --steps-per-trial 5000 \
    --symbol AAPL
```

### Automated Pipeline
```bash
# Check pipeline status
python scripts/deepscalper/bot_manager.py pipeline status

# Run single cycle (training + evaluation)
python scripts/deepscalper/bot_manager.py pipeline run-once

# View model history
python scripts/deepscalper/bot_manager.py pipeline models

# Start continuous operation
python scripts/deepscalper/bot_manager.py pipeline start \
    --training-interval 4 \
    --check-interval 30
```

## 📋 CLI Reference

### Training Commands
```bash
# Basic training
bot_manager.py train [options]
  --symbol SYMBOL         Trading symbol (default: AAPL)
  --steps STEPS          Training steps (default: 10000)
  --lr RATE             Learning rate (default: 3e-4)
  --batch-size SIZE     Batch size (default: 256)
  --lookback WINDOW     History window (default: 120)

# Evaluation
bot_manager.py evaluate [options]
  --episodes N          Evaluation episodes (default: 10)
  --checkpoint-dir DIR  Model directory

# Hyperparameter optimization  
bot_manager.py hyperopt [options]
  --trials N            Number of trials (default: 20)
  --steps-per-trial N   Steps per trial (default: 5000)
```

### Pipeline Commands
```bash
# Pipeline management
bot_manager.py pipeline {start|run-once|status|models}
  --training-interval H     Hours between training (default: 6)
  --steps-per-training N    Steps per session (default: 10000)
  --hyperopt-interval H     Hours between hyperopt (default: 72)

# Monitoring
bot_manager.py monitor {report|recent}
  --episodes N             Episodes to analyze (default: 10)
  --log-dir DIR           Log directory

# Data management
bot_manager.py data {generate|download}
  --symbol SYMBOL         Symbol to process
  --days N               Days of data (default: 30)
  --output FILE          Save to file
```

## 🔧 Advanced Configuration

### Environment Parameters
```python
env_config = EnvConfig(
    symbol="AAPL",
    lookback=120,           # History window
    episode_minutes=390,    # Trading day length
    fee_rate=0.0005,       # Transaction costs
    max_position_pct=0.9,  # Max position size
    price_bins=21,         # Price action bins
    qty_bins=11            # Quantity bins
)
```

### Training Parameters
```python
train_config = TrainConfig(
    lr=3e-4,                    # Learning rate
    batch_size=256,             # Batch size
    buffer_size=500_000,        # Replay buffer
    hindsight_weight=0.5,       # Hindsight bonus
    aux_weight=0.1,             # Risk prediction weight
    eps_decay_steps=150_000     # Exploration decay
)
```

### Pipeline Configuration
```python
pipeline_config = PipelineConfig(
    training_interval_hours=6,      # Training frequency
    steps_per_training=10_000,      # Steps per session
    performance_threshold=0.1,      # Min improvement for deployment
    max_model_versions=10,          # Model history limit
    hyperopt_interval_hours=72      # Hyperopt frequency
)
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Episode returns, cumulative performance
- **Risk Metrics**: Sharpe ratio, max drawdown, volatility
- **Trading Stats**: Win rate, trade frequency, position sizing
- **Model Metrics**: Training loss, exploration rate, learning progress

## 🚨 Monitoring & Alerts

Automated monitoring includes:
- Performance degradation detection
- High drawdown alerts
- Model drift identification
- Training failure notifications

## 🔄 Continuous Improvement Loop

1. **Data Collection**: Gather market data (real or synthetic)
2. **Training**: Train models with current best hyperparameters  
3. **Evaluation**: Test on holdout data with comprehensive metrics
4. **Model Selection**: Compare performance, promote best models
5. **Hyperparameter Optimization**: Periodic search for better configurations
6. **Deployment**: Automatic production model updates
7. **Monitoring**: Track live performance and detect issues

## 📈 Results & Analysis

View training progress and model performance:
```bash
# Generate comprehensive report
python scripts/deepscalper/bot_manager.py monitor report

# Recent performance trends
python scripts/deepscalper/bot_manager.py monitor recent --episodes 50

# Model comparison
python scripts/deepscalper/bot_manager.py pipeline models
```

## 🔧 Troubleshooting

### Common Issues
- **Network errors**: System automatically falls back to synthetic data
- **Memory issues**: Reduce batch size or buffer size in configuration
- **Poor performance**: Run hyperparameter optimization to find better settings
- **Training failures**: Check logs in monitoring directory

### Performance Optimization
- Use GPU if available for faster training
- Increase batch size for stable learning (if memory allows)
- Run longer training sessions for better convergence
- Use hyperparameter optimization to find optimal settings

## 🚀 Next Steps

The bot includes a complete development pipeline for continuous improvement:

1. **Monitor Performance**: Track metrics and identify improvement opportunities
2. **Optimize Hyperparameters**: Regularly search for better configurations  
3. **Experiment with Features**: Add new technical indicators or data sources
4. **Scale Training**: Use larger datasets and longer training sessions
5. **Deploy Safely**: Use A/B testing to validate improvements

## 📄 License & Disclaimer

This is research software for educational and experimental purposes only. Not financial advice. Use at your own risk.