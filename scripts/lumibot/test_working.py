#!/usr/bin/env python3
"""
Test the working strategy on a period with known signals
"""

import os
import sys
import datetime as dt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.lumibot.fv_ra import FlawlessVictoryRiskAware, RiskConfig
from lumibot.backtesting import YahooDataBacktesting


def test_strategy_with_signals():
    """Test strategy on Q4 2022 period which showed buy signals"""
    
    print("TESTING STRATEGY ON PERIOD WITH KNOWN SIGNALS")
    print("=" * 60)
    
    # Use the corrected parameters
    FlawlessVictoryRiskAware.parameters.update({
        "version": "v1",
        "cash_fraction": 0.8,
        "v1_bb_period": 20,
        "v1_bb_dev": 1.5,      # Tighter for more signals
        "v1_rsi_lower": 25,    # Lower threshold
        "v1_rsi_upper": 80,    # Higher threshold
        "risk_config": RiskConfig(
            enabled=True,
            target_daily_vol=0.025,
            max_drawdown=0.20,
            verbose=True
        )
    })
    
    # Test on Q4 2022 (had 8 buy signals)
    start_date = dt.datetime(2022, 10, 1)
    end_date = dt.datetime(2022, 12, 31)
    
    print(f"Testing: Q4 2022 (Oct-Dec 2022)")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("Expected: 8 buy signals based on analysis")
    print("\nRunning backtest...")
    
    try:
        results = FlawlessVictoryRiskAware.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
        )
        
        print("\n✓ Backtest completed successfully!")
        print("Check logs above for 'ENTRY SIGNAL DETECTED' messages")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_signal_periods():
    """Test on multiple periods that showed signals"""
    
    periods = [
        (dt.datetime(2022, 10, 1), dt.datetime(2022, 12, 31), "Q4 2022 - 8 buy signals"),
        (dt.datetime(2024, 1, 1), dt.datetime(2024, 3, 31), "Q1 2024 - 4 buy signals"),
        (dt.datetime(2022, 6, 1), dt.datetime(2022, 8, 31), "Q3 2022 - 2 buy, 5 sell signals"),
    ]
    
    for start, end, description in periods:
        print(f"\n{'='*60}")
        print(f"TESTING: {description}")
        print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        print("="*60)
        
        # Reset parameters for each test
        FlawlessVictoryRiskAware.parameters.update({
            "version": "v1",
            "cash_fraction": 0.5,  # Conservative
            "v1_bb_period": 20,
            "v1_bb_dev": 1.5,
            "v1_rsi_lower": 25,
            "v1_rsi_upper": 80,
            "risk_config": RiskConfig(
                enabled=True,
                verbose=True
            )
        })
        
        try:
            results = FlawlessVictoryRiskAware.backtest(
                YahooDataBacktesting,
                start,
                end,
            )
            
            print("✓ Period completed successfully!")
            
        except Exception as e:
            print(f"✗ Period failed: {e}")


if __name__ == "__main__":
    print("Testing Flawless Victory Strategy on Periods with Known Signals")
    print("=" * 80)
    
    # Test the main period first
    result = test_strategy_with_signals()
    
    if result:
        # If successful, test other periods
        test_multiple_signal_periods()
    
    print("\n" + "=" * 80)
    print("If you see 'ENTRY SIGNAL DETECTED' in the logs above,")
    print("the strategy is working correctly and making trades!")
