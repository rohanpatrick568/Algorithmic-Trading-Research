# Flawless Victory Risk Aware - Lumibot Implementation

## Overview

This is a Lumibot implementation of the Flawless Victory trading strategy with DeepScalper-inspired risk management features. The strategy combines technical analysis with sophisticated risk controls to create a robust trading system.

## Key Features

### 🎯 Strategy Versions
- **v1**: Simple Bollinger Bands + RSI signals
- **v2**: Bollinger Bands + RSI + Stop Loss/Take Profit
- **v3**: Bollinger Bands + MFI + RSI + Stop Loss/Take Profit

### 🛡️ Risk Management (Inspired by DeepScalper)
- **Volatility-aware position sizing**: Scales positions based on realized volatility
- **Drawdown brake**: Pauses trading when drawdown exceeds threshold
- **Downside deviation gating**: Monitors downside risk (Sortino-style)
- **Real-time risk logging**: Verbose risk monitoring and alerts

### 📊 Technical Indicators
- RSI (Relative Strength Index)
- MFI (Money Flow Index)
- Bollinger Bands (multiple periods)
- Realized Volatility
- Downside Deviation

## Usage

### Basic Usage

```python
from scripts.lumibot.fv_ra import run_backtest
import datetime as dt

# Run v1 strategy on AAPL
results = run_backtest(
    version="v1",
    symbol="AAPL",
    start_date=dt.datetime(2023, 1, 1),
    end_date=dt.datetime(2023, 12, 31),
    risk_enabled=True
)
```

### Advanced Configuration

```python
from scripts.lumibot.fv_ra import FlawlessVictoryRiskAware, RiskConfig

# Customize risk parameters
custom_risk = RiskConfig(
    enabled=True,
    target_daily_vol=0.02,      # 2% daily vol target
    max_drawdown=0.10,          # 10% max drawdown
    dd_cooldown_bars=5,         # 5-day cooldown
    max_downside_dev=0.015,     # 1.5% max downside dev
    verbose=True
)

# Update strategy parameters
FlawlessVictoryRiskAware.parameters.update({
    "version": "v2",
    "cash_fraction": 0.8,
    "risk_config": custom_risk
})
```

### Running Tests

```bash
# Activate environment
.\env\Scripts\Activate.ps1

# Run basic functionality test
python scripts\lumibot\basic_test.py

# Run comprehensive tests
python scripts\lumibot\test_fv_ra.py

# Run simple example
python scripts\lumibot\simple_test.py
```

## Configuration Parameters

### Strategy Parameters
- `version`: "v1", "v2", or "v3"
- `cash_fraction`: Fraction of cash to use per trade (0.0-1.0)
- `rsi_period`: RSI calculation period (default: 14)
- `mfi_period`: MFI calculation period (default: 14)
- Bollinger Band periods and deviations for each version
- Entry/exit thresholds for RSI and MFI

### Risk Parameters (RiskConfig)
- `enabled`: Enable/disable risk management
- `target_daily_vol`: Target daily volatility for scaling (default: 0.015)
- `vol_period`: Window for volatility calculation (default: 30)
- `min_scale`/`max_scale`: Position scaling bounds (default: 0.25-1.25)
- `max_drawdown`: Maximum drawdown threshold (default: 0.20)
- `dd_cooldown_bars`: Cooldown period after drawdown breach (default: 5)
- `sortino_period`: Period for downside deviation (default: 30)
- `max_downside_dev`: Maximum downside deviation (default: 0.02)
- `verbose`: Enable detailed risk logging

## Strategy Logic

### Entry Signals
- **v1**: Price below lower Bollinger Band + RSI > threshold
- **v2**: Price below lower Bollinger Band + RSI > threshold
- **v3**: Price below lower Bollinger Band + MFI < threshold

### Exit Signals
- **v1**: Price above upper Bollinger Band + RSI > threshold
- **v2**: Price above upper Bollinger Band + RSI > threshold (or SL/TP)
- **v3**: Price above upper Bollinger Band + RSI > threshold + MFI > threshold (or SL/TP)

### Risk Controls
1. **Position Sizing**: `base_size * volatility_scale * cash_fraction`
2. **Entry Gating**: Block entries if:
   - Drawdown exceeds threshold
   - In cooldown period
   - Downside deviation too high
3. **Monitoring**: Log risk metrics every iteration when verbose=True

## Test Results

✅ **Basic Functionality**: All versions pass basic backtesting
✅ **Risk Calculations**: Volatility scaling and downside deviation working
✅ **Multi-Version Support**: v1, v2, v3 all functional
✅ **Parameter Flexibility**: Easy configuration and customization

## Files Structure

```
scripts/lumibot/
├── fv_ra.py              # Main strategy implementation
├── basic_test.py         # Basic functionality tests
├── test_fv_ra.py         # Comprehensive tests
├── simple_test.py        # Simple example runner
└── README_lumibot.md     # This documentation
```

## Next Steps

1. **Live Trading**: Configure broker credentials for live trading
2. **Parameter Optimization**: Use Lumibot's optimization features
3. **Multiple Assets**: Extend to trade multiple symbols
4. **Advanced Risk**: Add portfolio-level risk management
5. **Machine Learning**: Integrate with RL components from other scripts

## DeepScalper Integration

This implementation incorporates key DeepScalper concepts:
- **Risk-aware auxiliary tasks**: Volatility prediction and risk scaling
- **Multi-modal risk assessment**: Combines volatility, drawdown, and downside deviation
- **Adaptive position sizing**: Dynamic scaling based on market conditions
- **Real-time monitoring**: Continuous risk assessment and logging

The strategy can be further extended with the RL components from `scripts/final/combined.py` for a complete DeepScalper-inspired system.
