#!/usr/bin/env python3
"""
Test script for Flawless Victory Risk Aware Lumibot strategy
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.lumibot.fv_ra import FlawlessVictoryRiskAware, RiskConfig, run_backtest
import datetime as dt

def test_strategy():
    """Test the Flawless Victory Risk Aware strategy"""
    
    print("Testing Flawless Victory Risk Aware Strategy with Lumibot")
    print("=" * 60)
    
    # Test different versions
    versions = ["v1", "v2", "v3"]
    
    for version in versions:
        print(f"\n--- Testing Version {version} ---")
        
        # Update strategy parameters for this version
        FlawlessVictoryRiskAware.parameters.update({
            "version": version,
            "cash_fraction": 0.8,  # Use 80% of cash
            "risk_config": RiskConfig(
                enabled=True,
                target_daily_vol=0.02,  # 2% daily vol target
                max_drawdown=0.15,      # 15% max drawdown
                dd_cooldown_bars=3,     # 3-day cooldown
                verbose=True
            )
        })
        
        try:
            # Run a short backtest
            strategy = FlawlessVictoryRiskAware
            
            # Shorter period for faster testing
            backtesting_start = dt.datetime(2023, 1, 1)
            backtesting_end = dt.datetime(2023, 6, 30)
            
            from lumibot.backtesting import YahooDataBacktesting
            
            results = strategy.backtest(
                YahooDataBacktesting,
                backtesting_start,
                backtesting_end,
                buy_trading_fees=[0.005],   # Must be a list
                sell_trading_fees=[0.005],  # Must be a list
                benchmark_asset="SPY"
            )
            
            print(f"✓ {version} backtest completed successfully")
            
            # Print some basic stats if available
            if hasattr(results, 'stats_formatted'):
                print(f"Stats preview for {version}:")
                print(results.stats_formatted)
            
        except Exception as e:
            print(f"✗ Error testing {version}: {e}")
            print(f"Error type: {type(e).__name__}")

def test_risk_calculations():
    """Test the risk calculation methods"""
    print("\n" + "=" * 60)
    print("Testing Risk Calculation Methods")
    print("=" * 60)
    
    # Create a simple test class to test risk methods without full strategy initialization
    class RiskTester:
        def __init__(self):
            self.parameters = {"risk_config": RiskConfig(verbose=True)}
            self.price_history = []
            self.equity_peak = None
            self.dd_cooldown = 0
        
        # Copy the risk methods from FlawlessVictoryRiskAware
        def _calculate_vol_scaling(self):
            import numpy as np
            risk_cfg = self.parameters["risk_config"]
            
            if len(self.price_history) < risk_cfg.vol_period:
                return 1.0
            
            prices = np.array(self.price_history[-risk_cfg.vol_period:])
            returns = np.diff(prices) / prices[:-1]
            realized_vol = np.std(returns)
            
            if realized_vol <= 0 or np.isnan(realized_vol):
                return 1.0
            
            scale = risk_cfg.target_daily_vol / realized_vol
            return np.clip(scale, risk_cfg.min_scale, risk_cfg.max_scale)
        
        def _calculate_downside_deviation(self):
            import numpy as np
            risk_cfg = self.parameters["risk_config"]
            
            if len(self.price_history) < risk_cfg.sortino_period:
                return 0.0
            
            prices = np.array(self.price_history[-risk_cfg.sortino_period:])
            returns = np.diff(prices) / prices[:-1]
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return 0.0
            
            return np.std(downside_returns)
    
    strategy = RiskTester()
    
    # Simulate some price history
    import numpy as np
    np.random.seed(42)
    
    # Generate synthetic price series with some volatility
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 50)  # 2% daily vol
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    strategy.price_history = prices
    strategy.equity_peak = 100000
    
    # Test volatility scaling
    vol_scale = strategy._calculate_vol_scaling()
    print(f"Volatility scaling factor: {vol_scale:.3f}")
    
    # Test downside deviation
    downside_dev = strategy._calculate_downside_deviation()
    print(f"Downside deviation: {downside_dev:.4f}")
    
    print(f"✓ Risk calculations completed successfully")

if __name__ == "__main__":
    print("Flawless Victory Risk Aware - Lumibot Implementation Test")
    print("=" * 80)
    
    # Test risk calculations first
    test_risk_calculations()
    
    # Test strategy backtesting
    test_strategy()
    
    print("\n" + "=" * 80)
    print("Testing completed!")
