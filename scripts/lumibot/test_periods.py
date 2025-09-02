#!/usr/bin/env python3
"""
Test strategy with different time periods and parameters to find signals
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os

# Add the project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_multiple_periods():
    """Test multiple time periods to find one with signals"""
    
    periods = [
        ("2023-01-01", "2023-02-28", "Q1 2023"),
        ("2022-10-01", "2022-12-31", "Q4 2022 (Bear market)"),
        ("2022-06-01", "2022-08-31", "Q3 2022 (Volatile)"),
        ("2020-03-01", "2020-05-31", "COVID crash"),
        ("2024-01-01", "2024-03-31", "Recent 2024"),
    ]
    
    print("TESTING MULTIPLE PERIODS FOR SIGNALS")
    print("=" * 60)
    
    # Define more aggressive parameters that should catch signals
    class TestStrategy:
        def __init__(self):
            self.parameters = {
                "version": "v1",
                "rsi_period": 14,
                "v1_bb_period": 20,
                "v1_bb_dev": 1.5,      # Tighter than 2.0
                "v1_rsi_lower": 25,    # Lower threshold
                "v1_rsi_upper": 80,    # Higher threshold
            }
        
        def _calculate_rsi(self, prices, period=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        def _calculate_bollinger_bands(self, prices, period, std_dev):
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, lower_band
    
    strategy = TestStrategy()
    symbol = "AAPL"
    
    for start, end, description in periods:
        print(f"\n--- {description} ({start} to {end}) ---")
        
        try:
            # Get data
            data = yf.download(symbol, start=start, end=end, progress=False)
            if data.empty:
                print("No data available")
                continue
            
            # Prepare data
            df = data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            
            column_mapping = {}
            for col in df.columns:
                if 'close' in col.lower() and 'adj' not in col.lower():
                    column_mapping[col] = 'Close'
                elif 'volume' in col.lower():
                    column_mapping[col] = 'Volume'
            
            df = df.rename(columns=column_mapping)
            
            if 'Close' not in df.columns:
                print("No Close data")
                continue
            
            # Calculate indicators
            rsi = strategy._calculate_rsi(df['Close'], strategy.parameters["rsi_period"])
            bb_upper, bb_lower = strategy._calculate_bollinger_bands(
                df['Close'], 
                strategy.parameters["v1_bb_period"], 
                strategy.parameters["v1_bb_dev"]
            )
            
            # Check for valid data
            valid_rows = ~(rsi.isna() | bb_upper.isna() | bb_lower.isna())
            valid_data = df[valid_rows].copy()
            
            if len(valid_data) == 0:
                print("No valid data points")
                continue
            
            valid_rsi = rsi[valid_rows]
            valid_bb_upper = bb_upper[valid_rows]
            valid_bb_lower = bb_lower[valid_rows]
            
            # Count signals
            bb_buy_days = (valid_data['Close'] < valid_bb_lower).sum()
            rsi_buy_days = (valid_rsi > strategy.parameters["v1_rsi_lower"]).sum()
            both_buy_days = ((valid_data['Close'] < valid_bb_lower) & (valid_rsi > strategy.parameters["v1_rsi_lower"])).sum()
            
            bb_sell_days = (valid_data['Close'] > valid_bb_upper).sum()
            rsi_sell_days = (valid_rsi > strategy.parameters["v1_rsi_upper"]).sum()
            both_sell_days = ((valid_data['Close'] > valid_bb_upper) & (valid_rsi > strategy.parameters["v1_rsi_upper"])).sum()
            
            print(f"Data points: {len(valid_data)}")
            print(f"Price range: ${valid_data['Close'].min():.2f} - ${valid_data['Close'].max():.2f}")
            print(f"RSI range: {valid_rsi.min():.1f} - {valid_rsi.max():.1f}")
            print(f"BUY signals: {both_buy_days} (BB: {bb_buy_days}, RSI: {rsi_buy_days})")
            print(f"SELL signals: {both_sell_days} (BB: {bb_sell_days}, RSI: {rsi_sell_days})")
            
            if both_buy_days > 0 or both_sell_days > 0:
                print(f"✓ FOUND {both_buy_days} buy and {both_sell_days} sell signals!")
                
                # Show some actual signal dates
                buy_signals = (valid_data['Close'] < valid_bb_lower) & (valid_rsi > strategy.parameters["v1_rsi_lower"])
                sell_signals = (valid_data['Close'] > valid_bb_upper) & (valid_rsi > strategy.parameters["v1_rsi_upper"])
                
                if buy_signals.any():
                    buy_dates = valid_data[buy_signals].index[:3]  # First 3
                    print(f"   Buy signal dates: {[d.strftime('%Y-%m-%d') for d in buy_dates]}")
                
                if sell_signals.any():
                    sell_dates = valid_data[sell_signals].index[:3]  # First 3
                    print(f"   Sell signal dates: {[d.strftime('%Y-%m-%d') for d in sell_dates]}")
            else:
                print("   No signals found")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print("If any period shows signals, use those parameters and time ranges.")
    print("If no periods show signals, the strategy logic may need adjustment.")


def test_alternative_signal_logic():
    """Test with mean reversion logic instead of breakout"""
    
    print(f"\n{'='*60}")
    print("TESTING ALTERNATIVE SIGNAL LOGIC")
    print("(Mean reversion instead of breakout)")
    print("="*60)
    
    # Test mean reversion: buy when price is ABOVE upper band (overbought, expect reversion)
    # sell when price is BELOW lower band (oversold, expect bounce)
    
    symbol = "AAPL"
    data = yf.download(symbol, start="2023-03-01", end="2023-04-15", progress=False)
    
    if data.empty:
        print("No data")
        return
    
    # Prepare data
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df = df.rename(columns={'Close': 'Close'})
    
    # Calculate indicators
    rsi = df['Close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / df['Close'].diff().apply(lambda x: max(-x, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + rsi))
    
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    bb_upper = sma + (1.5 * std)
    bb_lower = sma - (1.5 * std)
    
    # Alternative logic: Mean reversion
    print("\nMean Reversion Logic:")
    print("BUY when: price > BB_upper (overbought, expect drop)")
    print("SELL when: price < BB_lower (oversold, expect bounce)")
    
    valid_idx = ~(bb_upper.isna() | bb_lower.isna())
    valid_data = df[valid_idx]
    valid_bb_upper = bb_upper[valid_idx]
    valid_bb_lower = bb_lower[valid_idx]
    
    # Count mean reversion signals
    mr_buy_signals = (valid_data['Close'] > valid_bb_upper).sum()
    mr_sell_signals = (valid_data['Close'] < valid_bb_lower).sum()
    
    print(f"\nMean Reversion Signals:")
    print(f"BUY signals (price > BB_upper): {mr_buy_signals}")
    print(f"SELL signals (price < BB_lower): {mr_sell_signals}")
    
    if mr_buy_signals > 0 or mr_sell_signals > 0:
        print("✓ Mean reversion logic shows signals!")
        print("Consider implementing this as an alternative strategy version")
    else:
        print("Even mean reversion shows no signals in this period")


if __name__ == "__main__":
    test_multiple_periods()
    test_alternative_signal_logic()
