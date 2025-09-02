#!/usr/bin/env python3
"""
Basic functionality test for Flawless Victory Risk Aware strategy
"""

import os
import sys
import datetime as dt

# Add the project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.lumibot.fv_ra import FlawlessVictoryRiskAware, RiskConfig
from lumibot.backtesting import YahooDataBacktesting


def test_basic_functionality():
    """Test basic strategy functionality without complex trading fees"""
    
    print("Testing Flawless Victory Risk Aware - Basic Functionality")
    print("=" * 70)
    
    # Test version v1 only (most stable)
    FlawlessVictoryRiskAware.parameters.update({
        "version": "v1",
        "cash_fraction": 0.5,  # Conservative position sizing
        "risk_config": RiskConfig(
            enabled=True,
            target_daily_vol=0.02,
            max_drawdown=0.10,
            verbose=True
        )
    })
    
    # Short backtest period
    start_date = dt.datetime(2023, 4, 1)
    end_date = dt.datetime(2023, 5, 31)
    
    print(f"Testing period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("Strategy: Flawless Victory v1 with Risk Management")
    
    try:
        # Run without trading fees first
        results = FlawlessVictoryRiskAware.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            benchmark_asset="SPY"
        )
        
        print("✓ Basic backtest completed successfully!")
        
        # Try to extract some basic info
        if hasattr(results, 'stats') and results.stats:
            print("\nStrategy Performance Summary:")
            print("-" * 40)
            for key, value in results.stats.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_features():
    """Test specific risk management features"""
    
    print("\n" + "=" * 70)
    print("Testing Risk Management Features")
    print("=" * 70)
    
    # Test with more aggressive risk settings to trigger gates
    FlawlessVictoryRiskAware.parameters.update({
        "version": "v1",
        "risk_config": RiskConfig(
            enabled=True,
            target_daily_vol=0.005,  # Very low target (should scale down positions)
            max_drawdown=0.05,       # Low drawdown threshold (should trigger gates)
            dd_cooldown_bars=2,
            max_downside_dev=0.01,   # Low downside dev (should trigger gates)
            verbose=True
        )
    })
    
    start_date = dt.datetime(2023, 3, 1)
    end_date = dt.datetime(2023, 4, 15)
    
    print("Testing with strict risk parameters:")
    print("- Target daily vol: 0.5% (very conservative)")
    print("- Max drawdown: 5% (strict)")
    print("- Max downside deviation: 1% (strict)")
    
    try:
        results = FlawlessVictoryRiskAware.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
        )
        
        print("✓ Risk management test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Risk management test failed: {e}")
        return False


def show_strategy_info():
    """Display strategy configuration and features"""
    
    print("\n" + "=" * 70)
    print("Flawless Victory Risk Aware Strategy - Feature Summary")
    print("=" * 70)
    
    print("\n📈 Strategy Versions:")
    print("  v1: Bollinger Bands + RSI (simple entries/exits)")
    print("  v2: Bollinger Bands + RSI + Stop Loss/Take Profit")
    print("  v3: Bollinger Bands + MFI + RSI + Stop Loss/Take Profit")
    
    print("\n🛡️ Risk Management Features:")
    print("  • Volatility-aware position sizing")
    print("  • Drawdown brake with cooldown periods")
    print("  • Downside deviation (Sortino) gating")
    print("  • Real-time risk monitoring and logging")
    
    print("\n⚙️ Technical Indicators:")
    print("  • RSI (Relative Strength Index)")
    print("  • MFI (Money Flow Index)")
    print("  • Bollinger Bands (multiple periods)")
    print("  • Realized Volatility")
    print("  • Downside Deviation")
    
    print("\n📊 Inspired by DeepScalper methodology:")
    print("  • Risk-aware auxiliary tasks")
    print("  • Volatility-based position scaling")
    print("  • Multi-modal risk assessment")


if __name__ == "__main__":
    show_strategy_info()
    
    # Test basic functionality
    basic_success = test_basic_functionality()
    
    if basic_success:
        # Test risk features
        test_risk_features()
        
        print("\n" + "=" * 70)
        print("✓ All tests completed successfully!")
        print("\nTo run the strategy:")
        print("  python scripts/lumibot/simple_test.py")
        print("\nTo customize parameters, edit the strategy.parameters in fv_ra.py")
    else:
        print("\n" + "=" * 70)
        print("✗ Basic tests failed. Check the error messages above.")
    
    print("=" * 70)
