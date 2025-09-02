#!/usr/bin/env python3
"""
Debug test for Flawless Victory strategy to identify why no trades are happening
"""

import os
import sys
import datetime as dt

# Add the project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.lumibot.fv_ra import FlawlessVictoryRiskAware, RiskConfig
from lumibot.backtesting import YahooDataBacktesting


def debug_strategy_signals():
    """Debug strategy to see why no trades are happening"""
    
    print("DEBUG: Analyzing Strategy Signals")
    print("=" * 50)
    
    # Use more relaxed parameters to increase trade frequency
    debug_params = {
        "version": "v1",
        "cash_fraction": 0.5,
        
        # More relaxed RSI thresholds
        "v1_rsi_lower": 30,    # Lower from 42
        "v1_rsi_upper": 65,    # Lower from 70
        
        # More relaxed Bollinger parameters
        "v1_bb_period": 15,    # Shorter period
        "v1_bb_dev": 0.8,      # Tighter bands
        
        "risk_config": RiskConfig(
            enabled=True,
            target_daily_vol=0.025,   # Higher vol target (less scaling down)
            max_drawdown=0.25,        # More lenient drawdown
            max_downside_dev=0.05,    # More lenient downside dev
            verbose=True              # Enable all debug logs
        )
    }
    
    # Update strategy parameters
    FlawlessVictoryRiskAware.parameters.update(debug_params)
    
    print("Updated Strategy Parameters:")
    print(f"  RSI Lower: {debug_params['v1_rsi_lower']}")
    print(f"  RSI Upper: {debug_params['v1_rsi_upper']}")
    print(f"  BB Period: {debug_params['v1_bb_period']}")
    print(f"  BB Dev: {debug_params['v1_bb_dev']}")
    print(f"  Target Vol: {debug_params['risk_config'].target_daily_vol}")
    
    # Test on a volatile period to increase chance of signals
    start_date = dt.datetime(2023, 3, 1)   # March 2023 - banking crisis volatility
    end_date = dt.datetime(2023, 4, 15)
    
    print(f"\nTesting period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("(Banking crisis period - high volatility should trigger more signals)")
    
    try:
        results = FlawlessVictoryRiskAware.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
        )
        
        print("\n✓ Debug backtest completed!")
        print("Check the logs above for detailed signal analysis")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_even_more_relaxed():
    """Test with extremely relaxed parameters"""
    
    print("\n" + "=" * 50)
    print("EXTREME DEBUG: Very Relaxed Parameters")
    print("=" * 50)
    
    # Extremely relaxed parameters
    extreme_params = {
        "version": "v1",
        "cash_fraction": 0.8,
        
        # Very relaxed RSI (almost always true)
        "v1_rsi_lower": 10,    # Very low
        "v1_rsi_upper": 90,    # Very high
        
        # Very relaxed Bollinger parameters
        "v1_bb_period": 10,    # Very short period
        "v1_bb_dev": 0.5,      # Very tight bands
        
        "risk_config": RiskConfig(
            enabled=False,        # Disable risk management entirely
            verbose=True
        )
    }
    
    FlawlessVictoryRiskAware.parameters.update(extreme_params)
    
    print("EXTREME Parameters (should definitely trigger trades):")
    print(f"  RSI Lower: {extreme_params['v1_rsi_lower']} (very low)")
    print(f"  RSI Upper: {extreme_params['v1_rsi_upper']} (very high)")
    print(f"  BB Period: {extreme_params['v1_bb_period']} (very short)")
    print(f"  BB Dev: {extreme_params['v1_bb_dev']} (very tight)")
    print(f"  Risk Management: DISABLED")
    
    # Very short test
    start_date = dt.datetime(2023, 3, 10)
    end_date = dt.datetime(2023, 3, 31)
    
    try:
        results = FlawlessVictoryRiskAware.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
        )
        
        print("\n✓ Extreme debug test completed!")
        return results
        
    except Exception as e:
        print(f"\n✗ Extreme debug test failed: {e}")
        return None


if __name__ == "__main__":
    print("Flawless Victory Strategy - Trade Debug Analysis")
    print("=" * 60)
    
    # First test with moderately relaxed parameters
    result1 = debug_strategy_signals()
    
    # Then test with extremely relaxed parameters
    result2 = test_even_more_relaxed()
    
    print("\n" + "=" * 60)
    if result1 or result2:
        print("✓ At least one test completed - check logs for trade signals")
    else:
        print("✗ Both tests failed - there may be a fundamental issue")
    
    print("\nIf you see 'ENTRY SIGNAL DETECTED' or 'EXIT SIGNAL DETECTED' in the logs,")
    print("the strategy is working. If not, there may be a data or logic issue.")
