"""
Flawless Victory Risk Aware Strategy - Lumibot Implementation

This module implements the Flawless Victory trading strategy using Lumibot framework,
enhanced with DeepScalper-inspired risk management features.

Strategy Overview:
- Multi-version strategy (v1, v2, v3) with different entry/exit criteria
- Bollinger Bands, RSI, and MFI-based signals
- Risk-aware position sizing based on realized volatility
- Drawdown protection with cooldown periods
- Downside deviation (Sortino-style) gating

Features:
1. **Strategy Versions:**
   - v1: Simple Bollinger + RSI signals
   - v2: Bollinger + RSI + Stop Loss/Take Profit
   - v3: Bollinger + MFI + RSI + Stop Loss/Take Profit

2. **Risk Management:**
   - Volatility-scaled position sizing
   - Maximum drawdown brake
   - Downside deviation monitoring
   - Real-time risk logging

3. **Technical Indicators:**
   - RSI (Relative Strength Index)
   - MFI (Money Flow Index) 
   - Bollinger Bands (configurable periods)
   - Realized Volatility calculation
   - Downside Deviation

Usage Example:
    ```python
    from scripts.lumibot.fv_ra import run_backtest
    
    # Run v1 strategy with risk management
    results = run_backtest(
        version="v1",
        symbol="AAPL", 
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        risk_enabled=True
    )
    ```

Risk Configuration:
    The RiskConfig dataclass allows customization of:
    - target_daily_vol: Target daily volatility for position scaling
    - max_drawdown: Maximum allowable drawdown before gating
    - dd_cooldown_bars: Bars to wait after drawdown breach
    - max_downside_dev: Maximum downside deviation threshold

Author: Algorithmic Trading Research
Inspired by: DeepScalper risk management methodology
Framework: Lumibot
"""

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset
import yfinance as yf


@dataclass
class RiskConfig:
    """Risk management configuration inspired by DeepScalper"""
    enabled: bool = True
    target_daily_vol: float = 0.015  # 1.5% daily vol target
    vol_period: int = 30  # window for realized vol calculation
    min_scale: float = 0.25  # minimum position scaling
    max_scale: float = 1.25  # maximum position scaling
    max_drawdown: float = 0.20  # 20% max drawdown threshold
    dd_cooldown_bars: int = 5  # bars to pause after drawdown breach
    sortino_period: int = 30  # period for downside deviation
    max_downside_dev: float = 0.02  # 2% max daily downside deviation
    verbose: bool = True


class FlawlessVictoryRiskAware(Strategy):
    """
    Lumibot implementation of Flawless Victory Strategy with DeepScalper-inspired risk controls.
    
    Features:
    - Multi-version strategy (v1, v2, v3) with Bollinger Bands + RSI/MFI
    - Risk-aware position sizing based on realized volatility
    - Drawdown brake with cooldown period
    - Downside deviation (Sortino) gating
    - Stop-loss and take-profit for v2/v3
    """
    
    parameters = {
        # Strategy version
        "version": "v1",  # v1|v2|v3
        
        # Position sizing
        "cash_fraction": 0.9,  # fraction of cash to use per trade
        
        # v2 SL/TP (%)
        "v2_stoploss_pct": 6.604,
        "v2_takeprofit_pct": 2.328,
        
        # v3 SL/TP (%)
        "v3_stoploss_pct": 8.882,
        "v3_takeprofit_pct": 2.317,
        
        # RSI/MFI periods
        "rsi_period": 14,
        "mfi_period": 14,
        
        # v1 guards (more reasonable thresholds)
        "v1_rsi_lower": 35,   # Raised from 42 to be less restrictive
        "v1_rsi_upper": 75,   # Raised from 70 to be less restrictive
        
        # v2 guards  
        "v2_rsi_lower": 35,   # More reasonable
        "v2_rsi_upper": 75,   # More reasonable
        
        # v3 guards
        "v3_mfi_lower": 70,   # Raised from 60 to be less restrictive
        "v3_rsi_upper": 70,   # Raised from 65
        "v3_mfi_upper": 75,   # Raised from 64
        
        # Bollinger Band configs (wider bands to catch more signals)
        "v1_bb_period": 20,   # Standard 20-day period
        "v1_bb_dev": 2.0,     # Standard 2.0 std dev (much wider than 0.8)
        "v2_bb_period": 20,   # Increased from 17
        "v2_bb_dev": 2.0,     # Increased from 1.0
        
        # Risk management
        "risk_config": RiskConfig(),
    }
    
    def initialize(self):
        """Initialize strategy state and symbols"""
        self.sleeptime = 1  # Check every minute in live trading
        
        # Set symbol
        self.symbol = "AAPL"
        self.asset = Asset(symbol=self.symbol, asset_type="stock")
        
        # Risk tracking
        self.equity_peak = None
        self.dd_cooldown = 0
        self.price_history = []
        self.max_history_len = 100  # Keep last 100 bars for calculations
        
        # Order tracking (prevent pyramiding)
        self.current_position_size = 0
        self.pending_orders = []
        
        self.log_message("FlawlessVictoryRiskAware initialized")
    
    def on_trading_iteration(self):
        """Main trading logic executed each iteration"""
        # Get current price data
        bars = self.get_historical_prices(self.asset, 100, "day")
        if bars is None or bars.df.empty:
            self.log_message("No price data available")
            return
            
        df = bars.df
        current_price = self.get_last_price(self.asset)
        
        if current_price is None:
            return
            
        # Update price history for risk calculations
        self._update_price_history(df)
        
        # Calculate indicators
        indicators = self._calculate_indicators(df)
        if not indicators:
            self.log_message("Failed to calculate indicators - insufficient data")
            return
            
        # Debug: Log key values
        if self.parameters["risk_config"].verbose:
            self.log_message(f"DEBUG: Price=${current_price:.2f}, RSI={indicators.get('rsi', 'N/A'):.1f}, "
                           f"MFI={indicators.get('mfi', 'N/A'):.1f}, "
                           f"BB1_lower={indicators.get('bb1_lower', 'N/A'):.2f}, "
                           f"BB1_upper={indicators.get('bb1_upper', 'N/A'):.2f}")
        
        # Risk assessment
        risk_scale, risk_ok, risk_meta = self._assess_risk()
        
        if self.parameters["risk_config"].verbose:
            self.log_message(f"Risk: scale={risk_scale:.2f}, ok={risk_ok}, meta={risk_meta}")
        
        # Get current position
        position = self.get_position(self.asset)
        current_qty = position.quantity if position else 0
        
        # Check for entry signals (only if flat and risk allows)
        if current_qty == 0 and risk_ok:
            entry_signal = self._check_entry_signals(current_price, indicators)
            if entry_signal:
                print(f"🎯 ENTRY SIGNAL DETECTED: {entry_signal} at ${current_price:.2f}")
                self.log_message(f"🎯 ENTRY SIGNAL DETECTED: {entry_signal} at ${current_price:.2f}")
                self._execute_entry(current_price, risk_scale, entry_signal)
            elif self.parameters["risk_config"].verbose:
                # Debug why no entry signal every 10th iteration to reduce spam
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 1
                
                if self._debug_counter % 10 == 0:  # Only log every 10th iteration
                    version = self.parameters["version"]
                    if version == "v1":
                        bb_condition = current_price < indicators.get('bb1_lower', float('inf'))
                        rsi_condition = indicators.get('rsi', 0) > self.parameters["v1_rsi_lower"]
                        print(f"V1 Entry Check: price<bb_lower={bb_condition} (${current_price:.2f}<${indicators.get('bb1_lower', 0):.2f}), rsi>threshold={rsi_condition} ({indicators.get('rsi', 0):.1f}>{self.parameters['v1_rsi_lower']})")
                    elif version == "v2":
                        bb_condition = current_price < indicators.get('bb2_lower', float('inf'))
                        rsi_condition = indicators.get('rsi', 0) > self.parameters["v2_rsi_lower"]
                        print(f"V2 Entry Check: price<bb_lower={bb_condition}, rsi>threshold={rsi_condition}")
                    elif version == "v3":
                        bb_condition = current_price < indicators.get('bb1_lower', float('inf'))
                        mfi_condition = indicators.get('mfi', 100) < self.parameters["v3_mfi_lower"]
                        print(f"V3 Entry Check: price<bb_lower={bb_condition}, mfi<threshold={mfi_condition}")
        elif current_qty == 0 and not risk_ok:
            if hasattr(self, '_risk_debug_counter'):
                self._risk_debug_counter += 1
            else:
                self._risk_debug_counter = 1
            
            if self._risk_debug_counter % 20 == 0:  # Only log every 20th iteration
                print(f"⚠️ Entry blocked by risk controls: {risk_meta}")
                self.log_message(f"⚠️ Entry blocked by risk controls: {risk_meta}")
        
        # Check for exit signals (if in position)
        elif current_qty != 0:
            exit_signal = self._check_exit_signals(current_price, indicators)
            if exit_signal:
                print(f"🔄 EXIT SIGNAL DETECTED: {exit_signal} at ${current_price:.2f}")
                self.log_message(f"🔄 EXIT SIGNAL DETECTED: {exit_signal} at ${current_price:.2f}")
                self._execute_exit()
            elif self.parameters["risk_config"].verbose:
                if hasattr(self, '_exit_debug_counter'):
                    self._exit_debug_counter += 1
                else:
                    self._exit_debug_counter = 1
                
                if self._exit_debug_counter % 20 == 0:  # Only log every 20th iteration
                    print(f"📊 In position ({current_qty} shares), no exit signal yet")
                    self.log_message(f"📊 In position ({current_qty} shares), no exit signal yet")
    
    def _update_price_history(self, df: pd.DataFrame):
        """Update internal price history for risk calculations"""
        if len(df) > 0:
            # Keep only the most recent prices
            self.price_history = df['close'].tolist()[-self.max_history_len:]
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Optional[dict]:
        """Calculate all required technical indicators"""
        if len(df) < max(self.parameters["v1_bb_period"], self.parameters["rsi_period"]):
            return None
            
        try:
            # RSI
            rsi = self._calculate_rsi(df['close'], self.parameters["rsi_period"])
            
            # MFI (Money Flow Index)
            mfi = self._calculate_mfi(df, self.parameters["mfi_period"])
            
            # Bollinger Bands for v1
            bb1_upper, bb1_lower = self._calculate_bollinger_bands(
                df['close'], self.parameters["v1_bb_period"], self.parameters["v1_bb_dev"]
            )
            
            # Bollinger Bands for v2
            bb2_upper, bb2_lower = self._calculate_bollinger_bands(
                df['close'], self.parameters["v2_bb_period"], self.parameters["v2_bb_dev"]
            )
            
            # Get the most recent values and handle NaN
            result = {
                'rsi': rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0,
                'mfi': mfi.iloc[-1] if len(mfi) > 0 and not pd.isna(mfi.iloc[-1]) else 50.0,
                'bb1_upper': bb1_upper.iloc[-1] if len(bb1_upper) > 0 and not pd.isna(bb1_upper.iloc[-1]) else float('inf'),
                'bb1_lower': bb1_lower.iloc[-1] if len(bb1_lower) > 0 and not pd.isna(bb1_lower.iloc[-1]) else 0.0,
                'bb2_upper': bb2_upper.iloc[-1] if len(bb2_upper) > 0 and not pd.isna(bb2_upper.iloc[-1]) else float('inf'),
                'bb2_lower': bb2_lower.iloc[-1] if len(bb2_lower) > 0 and not pd.isna(bb2_lower.iloc[-1]) else 0.0,
            }
            
            return result
        except Exception as e:
            self.log_message(f"Error calculating indicators: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and negative money flow
        positive_mf = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_mf = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Money flow ratio
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        mfr = positive_mf_sum / (negative_mf_sum + 1e-9)  # Avoid division by zero
        mfi = 100 - (100 / (1 + mfr))
        return mfi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band
    
    def _assess_risk(self) -> tuple[float, bool, dict]:
        """Assess risk and return scaling factor, approval, and metadata"""
        risk_cfg = self.parameters["risk_config"]
        
        if not risk_cfg.enabled:
            return 1.0, True, {}
        
        # Get current portfolio value
        portfolio_value = self.get_portfolio_value()
        
        # Track equity peak and drawdown
        if self.equity_peak is None:
            self.equity_peak = portfolio_value
        self.equity_peak = max(self.equity_peak, portfolio_value)
        
        current_dd = (portfolio_value / self.equity_peak) - 1.0
        
        # Check drawdown brake
        risk_ok = True
        risk_meta = {'drawdown': round(current_dd, 4)}
        
        if current_dd < -risk_cfg.max_drawdown:
            self.dd_cooldown = risk_cfg.dd_cooldown_bars
            risk_ok = False
            risk_meta['drawdown_breach'] = True
        
        if self.dd_cooldown > 0:
            self.dd_cooldown -= 1
            risk_ok = False
            risk_meta['cooldown_remaining'] = self.dd_cooldown
        
        # Calculate volatility scaling
        vol_scale = self._calculate_vol_scaling()
        risk_meta['vol_scale'] = round(vol_scale, 3)
        
        # Calculate downside deviation
        downside_dev = self._calculate_downside_deviation()
        if downside_dev > risk_cfg.max_downside_dev:
            risk_ok = False
            risk_meta['downside_dev_breach'] = round(downside_dev, 4)
        
        return vol_scale, risk_ok, risk_meta
    
    def _calculate_vol_scaling(self) -> float:
        """Calculate position scaling based on realized volatility"""
        risk_cfg = self.parameters["risk_config"]
        
        if len(self.price_history) < risk_cfg.vol_period:
            return 1.0
        
        # Calculate daily returns
        prices = np.array(self.price_history[-risk_cfg.vol_period:])
        returns = np.diff(prices) / prices[:-1]
        
        # Realized volatility (daily)
        realized_vol = np.std(returns)
        
        if realized_vol <= 0 or np.isnan(realized_vol):
            return 1.0
        
        # Scale position based on target vs realized vol
        scale = risk_cfg.target_daily_vol / realized_vol
        return np.clip(scale, risk_cfg.min_scale, risk_cfg.max_scale)
    
    def _calculate_downside_deviation(self) -> float:
        """Calculate downside deviation (Sortino-style)"""
        risk_cfg = self.parameters["risk_config"]
        
        if len(self.price_history) < risk_cfg.sortino_period:
            return 0.0
        
        # Calculate daily returns
        prices = np.array(self.price_history[-risk_cfg.sortino_period:])
        returns = np.diff(prices) / prices[:-1]
        
        # Downside deviation (negative returns only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        return np.std(downside_returns)
    
    def _check_entry_signals(self, current_price: float, indicators: dict) -> Optional[str]:
        """Check for entry signals based on strategy version"""
        version = self.parameters["version"]
        
        # Version 1: Bollinger + RSI
        if version == "v1":
            if (current_price < indicators['bb1_lower'] and 
                indicators['rsi'] > self.parameters["v1_rsi_lower"]):
                return "v1_buy"
        
        # Version 2: Bollinger + RSI  
        elif version == "v2":
            if (current_price < indicators['bb2_lower'] and
                indicators['rsi'] > self.parameters["v2_rsi_lower"]):
                return "v2_buy"
        
        # Version 3: Bollinger + MFI
        elif version == "v3":
            if (current_price < indicators['bb1_lower'] and
                indicators['mfi'] < self.parameters["v3_mfi_lower"]):
                return "v3_buy"
        
        return None
    
    def _check_exit_signals(self, current_price: float, indicators: dict) -> Optional[str]:
        """Check for exit signals based on strategy version"""
        version = self.parameters["version"]
        
        # Version 1: Bollinger + RSI
        if version == "v1":
            if (current_price > indicators['bb1_upper'] and
                indicators['rsi'] > self.parameters["v1_rsi_upper"]):
                return "v1_sell"
        
        # Version 2: Bollinger + RSI
        elif version == "v2":
            if (current_price > indicators['bb2_upper'] and
                indicators['rsi'] > self.parameters["v2_rsi_upper"]):
                return "v2_sell"
        
        # Version 3: Bollinger + RSI + MFI
        elif version == "v3":
            if (current_price > indicators['bb1_upper'] and
                indicators['rsi'] > self.parameters["v3_rsi_upper"] and
                indicators['mfi'] > self.parameters["v3_mfi_upper"]):
                return "v3_sell"
        
        return None
    
    def _execute_entry(self, current_price: float, risk_scale: float, signal: str):
        """Execute entry order with risk-scaled position size"""
        # Calculate position size
        cash = self.get_cash()
        base_size = int((cash * self.parameters["cash_fraction"]) / current_price)
        scaled_size = max(1, int(base_size * risk_scale))
        
        version = self.parameters["version"]
        
        # Log the trade attempt (this should definitely show)
        print(f"🚀 TRADE SIGNAL: {signal} - Buying {scaled_size} shares at ${current_price:.2f}")
        self.log_message(f"🚀 TRADE SIGNAL: {signal} - Buying {scaled_size} shares at ${current_price:.2f}")
        
        if version == "v1":
            # Simple market order for v1
            order = self.create_order(self.asset, scaled_size, "buy")
            self.submit_order(order)
            print(f"✅ BUY ORDER SUBMITTED: v1 - {scaled_size} shares @ ${current_price:.2f}")
            self.log_message(f"✅ BUY ORDER SUBMITTED: v1 - {scaled_size} shares @ ${current_price:.2f}")
        
        elif version in ["v2", "v3"]:
            # Bracket order with stop-loss and take-profit
            sl_pct = self.parameters[f"{version}_stoploss_pct"]
            tp_pct = self.parameters[f"{version}_takeprofit_pct"]
            
            stop_price = current_price * (1 - sl_pct / 100)
            limit_price = current_price * (1 + tp_pct / 100)
            
            # Create main buy order
            buy_order = self.create_order(self.asset, scaled_size, "buy")
            self.submit_order(buy_order)
            
            # Note: Lumibot doesn't have built-in bracket orders like Backtrader
            # We'll handle stop-loss and take-profit in the exit logic
            print(f"✅ BUY ORDER SUBMITTED: {version} - {scaled_size} shares @ ${current_price:.2f} "
                  f"(SL: ${stop_price:.2f}, TP: ${limit_price:.2f})")
            self.log_message(f"✅ BUY ORDER SUBMITTED: {version} - {scaled_size} shares @ ${current_price:.2f} "
                           f"(SL: ${stop_price:.2f}, TP: ${limit_price:.2f})")
    
    def _execute_exit(self):
        """Execute exit order to close position"""
        position = self.get_position(self.asset)
        if position and position.quantity > 0:
            print(f"🔄 EXIT SIGNAL: Selling {position.quantity} shares")
            self.log_message(f"🔄 EXIT SIGNAL: Selling {position.quantity} shares")
            
            order = self.create_order(self.asset, position.quantity, "sell")
            self.submit_order(order)
            
            print(f"✅ SELL ORDER SUBMITTED: {position.quantity} shares")
            self.log_message(f"✅ SELL ORDER SUBMITTED: {position.quantity} shares")
    
    def on_aborted_order(self, position, order, error):
        """Handle aborted orders"""
        self.log_message(f"Order aborted: {error}")
    
    def on_partially_filled_order(self, position, order, price, quantity, multiplier):
        """Handle partially filled orders"""
        self.log_message(f"Partial fill: {quantity} @ ${price}")
    
    def on_filled_order(self, position, order, price, quantity, multiplier):
        """Handle filled orders"""
        self.log_message(f"Order filled: {order.side} {quantity} @ ${price}")


# Backtesting configuration
def run_backtest(version="v1", symbol="AAPL", start_date=None, end_date=None, risk_enabled=True):
    """
    Run backtest with specified parameters
    
    Args:
        version: Strategy version ("v1", "v2", or "v3")
        symbol: Stock symbol to trade
        start_date: Backtest start date (datetime)
        end_date: Backtest end date (datetime)
        risk_enabled: Enable/disable risk management
    """
    
    if start_date is None:
        start_date = dt.datetime(2020, 1, 1)
    if end_date is None:
        end_date = dt.datetime(2023, 12, 31)
    
    # Create strategy instance
    strategy = FlawlessVictoryRiskAware
    
    # Set strategy parameters
    strategy.parameters = {
        **strategy.parameters,
        "version": version,
        "cash_fraction": 0.8,
        "risk_config": RiskConfig(
            enabled=risk_enabled,
            target_daily_vol=0.015,
            max_drawdown=0.15,
            verbose=True
        )
    }
    
    # Update symbol
    strategy.symbol = symbol
    
    print(f"Running backtest for {symbol} ({version})")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Risk Management: {'Enabled' if risk_enabled else 'Disabled'}")
    
    # Run backtest
    try:
        results = strategy.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            benchmark_asset="SPY"
        )
        
        print("✓ Backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        return None


def run_example():
    """Run example backtests for all versions"""
    
    print("Flawless Victory Risk Aware - Example Runs")
    print("=" * 60)
    
    # Test period
    start = dt.datetime(2023, 1, 1)
    end = dt.datetime(2023, 6, 30)
    
    versions = ["v1", "v2", "v3"]
    
    for version in versions:
        print(f"\n--- Testing {version} ---")
        
        try:
            # Run with risk management
            results = run_backtest(
                version=version,
                symbol="AAPL", 
                start_date=start,
                end_date=end,
                risk_enabled=True
            )
            
            if results:
                print(f"✓ {version} completed successfully")
            else:
                print(f"✗ {version} failed")
                
        except Exception as e:
            print(f"✗ {version} error: {e}")


if __name__ == "__main__":
    # Run examples
    run_example()
