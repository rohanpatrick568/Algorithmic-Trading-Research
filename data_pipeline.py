#!/usr/bin/env python3
"""
ML Trading Bot Data Pipeline

This script manages data collection, preprocessing, and preparation for training
the ML trading bot. It supports multiple data sources and automated updates.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


class DataCollector:
    """Collect market data from various sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def fetch_yahoo_data(self, symbol: str, start_date: str, end_date: str, 
                        interval: str = "1d") -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            data.index.name = 'date'
            
            self.logger.info(f"Fetched {len(data)} rows for {symbol} ({interval})")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: List[str], start_date: str, 
                             end_date: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        data_dict = {}
        
        for symbol in symbols:
            self.logger.info(f"Fetching data for {symbol}")
            data = self.fetch_yahoo_data(symbol, start_date, end_date, interval)
            if not data.empty:
                data_dict[symbol] = data
                
        return data_dict
    
    def save_data(self, data: pd.DataFrame, symbol: str, interval: str = "1d"):
        """Save data to CSV file"""
        filename = f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        
        data.to_csv(filepath)
        self.logger.info(f"Data saved to {filepath}")
        return str(filepath)
    
    def load_saved_data(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        """Load previously saved data"""
        pattern = f"{symbol}_{interval}_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            self.logger.warning(f"No saved data found for {symbol} ({interval})")
            return pd.DataFrame()
        
        # Use the most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        data = pd.read_csv(latest_file, index_col='date', parse_dates=True)
        
        self.logger.info(f"Loaded data from {latest_file}")
        return data


class DataPreprocessor:
    """Preprocess and clean market data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean market data"""
        cleaned_data = data.copy()
        
        # Remove rows with missing OHLC data
        essential_cols = ['open', 'high', 'low', 'close']
        cleaned_data = cleaned_data.dropna(subset=essential_cols)
        
        # Forward fill volume if missing
        if 'volume' in cleaned_data.columns:
            cleaned_data['volume'] = cleaned_data['volume'].fillna(method='ffill')
        
        # Remove obvious outliers (prices that change by more than 50% in one day)
        price_change = cleaned_data['close'].pct_change()
        outlier_mask = (abs(price_change) > 0.5)
        
        if outlier_mask.sum() > 0:
            self.logger.warning(f"Removing {outlier_mask.sum()} outlier rows")
            cleaned_data = cleaned_data[~outlier_mask]
        
        # Ensure high >= low, high >= open, high >= close, low <= open, low <= close
        cleaned_data['high'] = np.maximum.reduce([
            cleaned_data['high'], 
            cleaned_data['open'], 
            cleaned_data['close']
        ])
        
        cleaned_data['low'] = np.minimum.reduce([
            cleaned_data['low'], 
            cleaned_data['open'], 
            cleaned_data['close']
        ])
        
        return cleaned_data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        df = data.copy()
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # On-Balance Volume (OBV)
            price_change = df['close'].diff()
            volume_direction = np.where(price_change > 0, df['volume'], 
                                      np.where(price_change < 0, -df['volume'], 0))
            df['obv'] = volume_direction.cumsum()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        
        # Volatility (rolling standard deviation of returns)
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['price_change'].rolling(window=period).std()
        
        return df
    
    def create_features_for_ml(self, data: pd.DataFrame, lookback_window: int = 20) -> pd.DataFrame:
        """Create features specifically for ML training"""
        df = data.copy()
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Create lagged features
        for lag in range(1, lookback_window + 1):
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
        # Create rolling statistics
        for window in [5, 10, 20]:
            df[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean() if 'volume' in df.columns else 0
        
        # Create target variable (next period return)
        df['target_return'] = df['close'].shift(-1) / df['close'] - 1
        
        # Create categorical target (up/down/sideways)
        df['target_direction'] = pd.cut(
            df['target_return'], 
            bins=[-np.inf, -0.01, 0.01, np.inf], 
            labels=['down', 'sideways', 'up']
        )
        
        # Drop rows with NaN values (due to rolling windows and lags)
        df = df.dropna()
        
        return df
    
    def normalize_features(self, data: pd.DataFrame, method: str = "minmax") -> Tuple[pd.DataFrame, Dict]:
        """Normalize features for ML training"""
        df = data.copy()
        
        # Identify numeric columns (excluding target variables)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_cols = ['target_return', 'target_direction']
        feature_cols = [col for col in numeric_cols if col not in target_cols]
        
        normalization_params = {}
        
        if method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            normalization_params = {
                "method": "minmax",
                "min_": scaler.min_,
                "scale_": scaler.scale_
            }
        elif method == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            normalization_params = {
                "method": "standard",
                "mean_": scaler.mean_,
                "scale_": scaler.scale_
            }
        
        return df, normalization_params


class DataPipeline:
    """Main data pipeline class"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.collector = DataCollector(data_dir)
        self.preprocessor = DataPreprocessor()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"data_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def update_data(self, symbols: List[str], lookback_days: int = 365, 
                   intervals: List[str] = ["1d"]) -> Dict:
        """Update data for specified symbols"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        self.logger.info(f"Updating data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        results = {}
        
        for interval in intervals:
            self.logger.info(f"Processing {interval} data")
            
            # Fetch data
            data_dict = self.collector.fetch_multiple_symbols(symbols, start_date, end_date, interval)
            
            for symbol, data in data_dict.items():
                if data.empty:
                    continue
                    
                # Clean and preprocess
                cleaned_data = self.preprocessor.clean_data(data)
                
                # Add technical indicators
                processed_data = self.preprocessor.add_technical_indicators(cleaned_data)
                
                # Save processed data
                filepath = self.collector.save_data(processed_data, symbol, interval)
                
                results[f"{symbol}_{interval}"] = {
                    "rows": len(processed_data),
                    "start_date": processed_data.index.min().strftime("%Y-%m-%d"),
                    "end_date": processed_data.index.max().strftime("%Y-%m-%d"),
                    "filepath": filepath
                }
        
        return results
    
    def prepare_training_data(self, symbols: List[str], lookback_window: int = 30,
                            train_split: float = 0.8) -> Dict:
        """Prepare data specifically for ML training"""
        self.logger.info("Preparing training data")
        
        training_datasets = {}
        
        for symbol in symbols:
            # Load the most recent data
            data = self.collector.load_saved_data(symbol, "1d")
            
            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                continue
            
            # Create ML features
            ml_data = self.preprocessor.create_features_for_ml(data, lookback_window)
            
            if ml_data.empty:
                self.logger.warning(f"No ML features created for {symbol}")
                continue
            
            # Normalize features
            normalized_data, norm_params = self.preprocessor.normalize_features(ml_data)
            
            # Split into train/validation
            split_idx = int(len(normalized_data) * train_split)
            train_data = normalized_data.iloc[:split_idx]
            val_data = normalized_data.iloc[split_idx:]
            
            # Save training datasets
            train_file = self.data_dir / f"{symbol}_train_data.csv"
            val_file = self.data_dir / f"{symbol}_val_data.csv"
            
            train_data.to_csv(train_file)
            val_data.to_csv(val_file)
            
            training_datasets[symbol] = {
                "train_file": str(train_file),
                "val_file": str(val_file),
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "features": len([col for col in ml_data.columns if col not in ['target_return', 'target_direction']]),
                "normalization_params": norm_params
            }
            
            self.logger.info(f"Training data prepared for {symbol}: {len(train_data)} train, {len(val_data)} val samples")
        
        return training_datasets
    
    def create_data_summary(self) -> Dict:
        """Create summary of available data"""
        summary = {
            "data_files": [],
            "symbols": set(),
            "date_ranges": {},
            "total_files": 0
        }
        
        for csv_file in self.data_dir.glob("*.csv"):
            try:
                # Extract symbol from filename
                parts = csv_file.stem.split("_")
                if len(parts) >= 2:
                    symbol = parts[0]
                    summary["symbols"].add(symbol)
                    
                    # Load data to get date range
                    data = pd.read_csv(csv_file, index_col=0, parse_dates=True, nrows=1)
                    if not data.empty:
                        data_full = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        summary["date_ranges"][symbol] = {
                            "start": data_full.index.min().strftime("%Y-%m-%d"),
                            "end": data_full.index.max().strftime("%Y-%m-%d"),
                            "rows": len(data_full)
                        }
                    
                summary["data_files"].append({
                    "file": csv_file.name,
                    "size_mb": csv_file.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(csv_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing {csv_file}: {e}")
        
        summary["symbols"] = list(summary["symbols"])
        summary["total_files"] = len(summary["data_files"])
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="ML Trading Bot Data Pipeline")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL", "MSFT", "TSLA"], 
                       help="Trading symbols to process")
    parser.add_argument("--update-data", action="store_true", help="Update market data")
    parser.add_argument("--prepare-training", action="store_true", help="Prepare training datasets")
    parser.add_argument("--summary", action="store_true", help="Show data summary")
    parser.add_argument("--lookback-days", type=int, default=365, help="Days of historical data")
    parser.add_argument("--lookback-window", type=int, default=30, help="Lookback window for features")
    parser.add_argument("--intervals", nargs="+", default=["1d"], help="Data intervals")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(args.data_dir)
    
    if args.update_data:
        results = pipeline.update_data(
            args.symbols, 
            args.lookback_days, 
            args.intervals
        )
        print(f"\nData update completed for {len(results)} datasets:")
        for key, info in results.items():
            print(f"  {key}: {info['rows']} rows ({info['start_date']} to {info['end_date']})")
    
    if args.prepare_training:
        training_datasets = pipeline.prepare_training_data(
            args.symbols,
            args.lookback_window
        )
        print(f"\nTraining data prepared for {len(training_datasets)} symbols:")
        for symbol, info in training_datasets.items():
            print(f"  {symbol}: {info['train_samples']} train, {info['val_samples']} val samples")
    
    if args.summary:
        summary = pipeline.create_data_summary()
        print(f"\nData Summary:")
        print(f"Total files: {summary['total_files']}")
        print(f"Symbols: {', '.join(summary['symbols'])}")
        print("\nDate ranges:")
        for symbol, info in summary['date_ranges'].items():
            print(f"  {symbol}: {info['start']} to {info['end']} ({info['rows']} rows)")


if __name__ == "__main__":
    main()