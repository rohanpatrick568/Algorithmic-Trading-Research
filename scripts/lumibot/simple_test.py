#!/usr/bin/env python3
"""
Simple test runner for Flawless Victory Risk Aware Lumibot strategy
"""

import os
import sys
import datetime as dt

# Add the project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.lumibot.fv_ra import FlawlessVictoryRiskAware, RiskConfig
from lumibot.backtesting import YahooDataBacktesting


def run_simple_backtest():
    """Run a simple backtest of the strategy"""
    
    print("Running Flawless Victory Risk Aware Strategy Backtest")
    print("=" * 60)
    
    # Configure strategy parameters
    strategy_params = {
        "version": "v1",  # Start with v1
        "cash_fraction": 0.8,
        "risk_config": RiskConfig(
            enabled=True,
            target_daily_vol=0.015,  # 1.5% daily vol target
            max_drawdown=0.15,       # 15% max drawdown
            dd_cooldown_bars=3,      # 3-day cooldown
            verbose=True
        )
    }
    
    # Update strategy parameters
    FlawlessVictoryRiskAware.parameters.update(strategy_params)
    
    # Set backtest period (shorter for faster testing)
    backtesting_start = dt.datetime(2023, 1, 1)
    backtesting_end = dt.datetime(2023, 6, 30)
    
    print(f"Backtest period: {backtesting_start.strftime('%Y-%m-%d')} to {backtesting_end.strftime('%Y-%m-%d')}")
    print(f"Strategy version: {strategy_params['version']}")
    print(f"Risk management: {'Enabled' if strategy_params['risk_config'].enabled else 'Disabled'}")
    
    try:
        # Run backtest
        results = FlawlessVictoryRiskAware.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            buy_trading_fees=[0.01],   # Must be a list
            sell_trading_fees=[0.01],  # Must be a list
            benchmark_asset="SPY"
        )
        
        print("\n✓ Backtest completed successfully!")
        
        # Display results if available
        if hasattr(results, 'stats_formatted') and results.stats_formatted:
            print("\nBacktest Results:")
            print("-" * 40)
            print(results.stats_formatted)
        else:
            print("\nBacktest completed but detailed stats not available")
            
        return results
        
    except Exception as e:
        print(f"\n✗ Backtest failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None


def test_different_versions():
    """Test all three strategy versions"""
    
    versions = ["v1", "v2", "v3"]
    
    for version in versions:
        print(f"\n{'='*60}")
        print(f"Testing Strategy Version {version}")
        print(f"{'='*60}")
        
        # Update version
        FlawlessVictoryRiskAware.parameters["version"] = version
        
        # Run short backtest
        backtesting_start = dt.datetime(2023, 3, 1)
        backtesting_end = dt.datetime(2023, 4, 30)
        
        try:
            results = FlawlessVictoryRiskAware.backtest(
                YahooDataBacktesting,
                backtesting_start,
                backtesting_end,
                buy_trading_fees=[0.005],   # Must be a list
                sell_trading_fees=[0.005],  # Must be a list
            )
            print(f"✓ Version {version} backtest completed")
            
        except Exception as e:
            print(f"✗ Version {version} failed: {e}")


if __name__ == "__main__":
    print("Flawless Victory Risk Aware - Lumibot Strategy Test")
    print("=" * 80)
    
    # Run simple backtest first
    results = run_simple_backtest()
    
    if results is not None:
        # If successful, test other versions
        print("\n" + "=" * 80)
        print("Testing All Strategy Versions")
        test_different_versions()
    
    print("\n" + "=" * 80)
    print("Testing completed!")
