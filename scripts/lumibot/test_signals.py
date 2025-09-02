#!/usr/bin/env python3
"""
Direct signal testing without running full backtest
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os

# Add the project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_signal_logic_directly():
    """Test the signal logic directly on real data"""
    
    print("DIRECT SIGNAL TESTING")
    print("=" * 50)
    
    # Get real data
    symbol = "AAPL"
    data = yf.download(symbol, start="2023-03-01", end="2023-04-15", progress=False)
    
    if data.empty:
        print("Failed to get data")
        return
    
    print(f"Got {len(data)} days of data for {symbol}")
    
    # Create strategy instance (just for method access)
    class TestStrategy:
        def __init__(self):
            self.parameters = {
                "version": "v1",
                "rsi_period": 14,
                "mfi_period": 14,
                "v1_bb_period": 20,    # Standard 20-day
                "v1_bb_dev": 2.0,      # Standard 2.0 std dev
                "v1_rsi_lower": 35,    # More reasonable
                "v1_rsi_upper": 75,    # More reasonable
                "v2_bb_period": 20,
                "v2_bb_dev": 2.0,
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
        
        def _calculate_mfi(self, df, period=14):
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            positive_mf = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_mf = money_flow.where(typical_price < typical_price.shift(1), 0)
            positive_mf_sum = positive_mf.rolling(window=period).sum()
            negative_mf_sum = negative_mf.rolling(window=period).sum()
            mfr = positive_mf_sum / (negative_mf_sum + 1e-9)
            mfi = 100 - (100 / (1 + mfr))
            return mfi
        
        def _calculate_bollinger_bands(self, prices, period, std_dev):
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, lower_band
    
    strategy = TestStrategy()
    
    # Prepare data (standardize column names)
    df = data.copy()
    print(f"Original columns: {list(df.columns)}")
    
    # yfinance returns different column structures, handle both
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Now standardize the names
    column_mapping = {}
    for col in df.columns:
        if 'open' in col.lower():
            column_mapping[col] = 'Open'
        elif 'high' in col.lower():
            column_mapping[col] = 'High'
        elif 'low' in col.lower():
            column_mapping[col] = 'Low'
        elif 'close' in col.lower() and 'adj' not in col.lower():
            column_mapping[col] = 'Close'
        elif 'volume' in col.lower():
            column_mapping[col] = 'Volume'
    
    df = df.rename(columns=column_mapping)
    print(f"Standardized columns: {list(df.columns)}")
    
    # Calculate indicators
    print("\nCalculating indicators...")
    rsi = strategy._calculate_rsi(df['Close'], strategy.parameters["rsi_period"])
    mfi = strategy._calculate_mfi(df, strategy.parameters["mfi_period"])
    bb_upper, bb_lower = strategy._calculate_bollinger_bands(
        df['Close'], 
        strategy.parameters["v1_bb_period"], 
        strategy.parameters["v1_bb_dev"]
    )
    
    # Check for valid data
    valid_rows = ~(rsi.isna() | bb_upper.isna() | bb_lower.isna())
    valid_data = df[valid_rows].copy()
    valid_rsi = rsi[valid_rows]
    valid_bb_upper = bb_upper[valid_rows]
    valid_bb_lower = bb_lower[valid_rows]
    
    print(f"Valid data points: {len(valid_data)}")
    
    if len(valid_data) == 0:
        print("No valid data points!")
        return
    
    # Check signals day by day
    print("\nChecking signals for last 10 days:")
    print("Date       | Price   | RSI   | BB_Low  | BB_High | Buy? | Sell?")
    print("-" * 70)
    
    signals_found = 0
    
    for i in range(max(0, len(valid_data) - 10), len(valid_data)):
        date = valid_data.index[i]
        price = valid_data['Close'].iloc[i]
        rsi_val = valid_rsi.iloc[i]
        bb_low = valid_bb_lower.iloc[i]
        bb_high = valid_bb_upper.iloc[i]
        
        # Check buy signal (v1): price < bb_lower AND rsi > threshold
        buy_condition1 = price < bb_low
        buy_condition2 = rsi_val > strategy.parameters["v1_rsi_lower"]
        buy_signal = buy_condition1 and buy_condition2
        
        # Check sell signal (v1): price > bb_upper AND rsi > threshold  
        sell_condition1 = price > bb_high
        sell_condition2 = rsi_val > strategy.parameters["v1_rsi_upper"]
        sell_signal = sell_condition1 and sell_condition2
        
        if buy_signal or sell_signal:
            signals_found += 1
        
        print(f"{date.strftime('%Y-%m-%d')} | ${price:6.2f} | {rsi_val:5.1f} | ${bb_low:6.2f} | ${bb_high:7.2f} | {'YES' if buy_signal else 'no':<4} | {'YES' if sell_signal else 'no':<4}")
    
    print(f"\nSignals found in last 10 days: {signals_found}")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"RSI range: {valid_rsi.min():.1f} - {valid_rsi.max():.1f}")
    print(f"RSI threshold: > {strategy.parameters['v1_rsi_lower']} for buy, > {strategy.parameters['v1_rsi_upper']} for sell")
    
    # Count how many days meet each condition
    bb_buy_days = (valid_data['Close'] < valid_bb_lower).sum()
    rsi_buy_days = (valid_rsi > strategy.parameters["v1_rsi_lower"]).sum()
    both_buy_days = ((valid_data['Close'] < valid_bb_lower) & (valid_rsi > strategy.parameters["v1_rsi_lower"])).sum()
    
    print(f"\nBuy condition analysis:")
    print(f"Days where price < BB lower: {bb_buy_days}")
    print(f"Days where RSI > {strategy.parameters['v1_rsi_lower']}: {rsi_buy_days}")
    print(f"Days where BOTH conditions met: {both_buy_days}")
    
    if both_buy_days == 0:
        print("\n⚠️  NO BUY SIGNALS FOUND!")
        print("This explains why the strategy isn't trading.")
        print("Try adjusting parameters:")
        print(f"- Lower RSI threshold (currently {strategy.parameters['v1_rsi_lower']})")
        print(f"- Wider Bollinger Bands (currently {strategy.parameters['v1_bb_dev']} std dev)")
    else:
        print(f"\n✓ Found {both_buy_days} potential buy signals")


if __name__ == "__main__":
    test_signal_logic_directly()
