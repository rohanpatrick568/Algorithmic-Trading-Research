"""
Automated data pipeline for continuous ML trading bot training.

This module provides automated data collection, processing, and feeding
for continuous training of the ML trading bot.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import logging

import numpy as np
import pandas as pd

try:
    from .data import load_minute_data, MarketData, add_indicators  # type: ignore
    from .config import EnvConfig, FEATURES  # type: ignore
except Exception:
    from data import load_minute_data, MarketData, add_indicators  # type: ignore
    from config import EnvConfig, FEATURES  # type: ignore


@dataclass
class DataPipelineConfig:
    """Configuration for the data pipeline."""
    symbols: List[str] = None
    update_interval_minutes: int = 60
    lookback_days: int = 30
    storage_dir: str = "data_cache"
    max_cache_size_mb: int = 1000
    enable_real_time: bool = True
    backup_interval_hours: int = 24
    data_quality_threshold: float = 0.95  # Minimum data completeness
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]


class DataQualityChecker:
    """Checks data quality and completeness."""
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        
    def check_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality and return metrics."""
        total_rows = len(df)
        if total_rows == 0:
            return {
                'completeness': 0.0,
                'quality_score': 0.0,
                'issues': ['no_data'],
                'passed': False
            }
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        completeness = 1.0 - (missing_counts.sum() / (total_rows * len(df.columns)))
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        duplicate_ratio = duplicates / total_rows
        
        # Check for gaps in time series (for minute data)
        if isinstance(df.index, pd.DatetimeIndex):
            expected_freq = pd.infer_freq(df.index[:100])  # Infer from first 100 points
            if expected_freq:
                expected_range = pd.date_range(df.index[0], df.index[-1], freq=expected_freq)
                gaps = len(expected_range) - len(df)
                gap_ratio = gaps / len(expected_range) if len(expected_range) > 0 else 0
            else:
                gap_ratio = 0
        else:
            gap_ratio = 0
        
        # Check for outliers (prices > 10x or < 0.1x median)
        if 'close' in df.columns:
            median_price = df['close'].median()
            outliers = ((df['close'] > 10 * median_price) | (df['close'] < 0.1 * median_price)).sum()
            outlier_ratio = outliers / total_rows
        else:
            outlier_ratio = 0
        
        # Calculate overall quality score
        quality_score = completeness * (1 - duplicate_ratio) * (1 - gap_ratio) * (1 - outlier_ratio)
        
        issues = []
        if completeness < self.threshold:
            issues.append(f'low_completeness_{completeness:.3f}')
        if duplicate_ratio > 0.01:
            issues.append(f'duplicates_{duplicate_ratio:.3f}')
        if gap_ratio > 0.05:
            issues.append(f'gaps_{gap_ratio:.3f}')
        if outlier_ratio > 0.01:
            issues.append(f'outliers_{outlier_ratio:.3f}')
        
        return {
            'completeness': completeness,
            'duplicate_ratio': duplicate_ratio,
            'gap_ratio': gap_ratio,
            'outlier_ratio': outlier_ratio,
            'quality_score': quality_score,
            'issues': issues,
            'passed': quality_score >= self.threshold
        }


class DataCache:
    """Manages cached market data with automatic cleanup."""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'files': {}, 'total_size': 0}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _cleanup_if_needed(self):
        """Remove old cache files if size limit exceeded."""
        if self.metadata['total_size'] <= self.max_size_bytes:
            return
        
        # Sort files by last access time and remove oldest
        files_by_access = sorted(
            self.metadata['files'].items(),
            key=lambda x: x[1]['last_access']
        )
        
        for filename, file_info in files_by_access:
            file_path = self.cache_dir / filename
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                self.metadata['total_size'] -= file_info['size']
                del self.metadata['files'][filename]
                
                if self.metadata['total_size'] <= self.max_size_bytes * 0.8:  # Keep 20% buffer
                    break
        
        self._save_metadata()
    
    def get_cache_key(self, symbol: str, start: str, end: str) -> str:
        """Generate cache key for data request."""
        return f"{symbol}_{start}_{end}.parquet"
    
    def get(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Get cached data if available and fresh."""
        cache_key = self.get_cache_key(symbol, start, end)
        cache_file = self.cache_dir / cache_key
        
        if not cache_file.exists():
            return None
        
        # Check if cache is still valid (not too old)
        file_info = self.metadata['files'].get(cache_key)
        if not file_info or time.time() - file_info['created'] > 3600:  # 1 hour expiry
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            # Update access time
            file_info['last_access'] = time.time()
            self._save_metadata()
            return df
        except Exception:
            return None
    
    def put(self, symbol: str, start: str, end: str, df: pd.DataFrame):
        """Cache data."""
        cache_key = self.get_cache_key(symbol, start, end)
        cache_file = self.cache_dir / cache_key
        
        try:
            df.to_parquet(cache_file)
            file_size = cache_file.stat().st_size
            
            # Update metadata
            self.metadata['files'][cache_key] = {
                'size': file_size,
                'created': time.time(),
                'last_access': time.time(),
                'symbol': symbol,
                'start': start,
                'end': end
            }
            self.metadata['total_size'] += file_size
            
            self._cleanup_if_needed()
            self._save_metadata()
            
        except Exception as e:
            logging.warning(f"Failed to cache data for {symbol}: {e}")


class ContinuousDataPipeline:
    """Manages continuous data collection and processing."""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.cache = DataCache(config.storage_dir, config.max_cache_size_mb)
        self.quality_checker = DataQualityChecker(config.data_quality_threshold)
        
        # Threading components
        self.data_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.workers = []
        
        # Statistics
        self.stats = {
            'updates_processed': 0,
            'quality_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_update': None,
            'errors': []
        }
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the data pipeline."""
        logger = logging.getLogger('DataPipeline')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path(self.config.storage_dir) / "pipeline.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def start(self):
        """Start the continuous data pipeline."""
        self.logger.info("Starting continuous data pipeline")
        
        # Start data collection worker
        collection_worker = threading.Thread(target=self._data_collection_worker, daemon=True)
        collection_worker.start()
        self.workers.append(collection_worker)
        
        # Start data processing worker
        processing_worker = threading.Thread(target=self._data_processing_worker, daemon=True)
        processing_worker.start()
        self.workers.append(processing_worker)
        
        # Start backup worker
        backup_worker = threading.Thread(target=self._backup_worker, daemon=True)
        backup_worker.start()
        self.workers.append(backup_worker)
        
        self.logger.info(f"Started {len(self.workers)} workers")
    
    def stop(self):
        """Stop the continuous data pipeline."""
        self.logger.info("Stopping continuous data pipeline")
        self.stop_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
        
        self.logger.info("Data pipeline stopped")
    
    def _data_collection_worker(self):
        """Worker thread for collecting data from sources."""
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                end_date = current_time.strftime("%Y-%m-%d")
                start_date = (current_time - timedelta(days=self.config.lookback_days)).strftime("%Y-%m-%d")
                
                for symbol in self.config.symbols:
                    # Check cache first
                    cached_data = self.cache.get(symbol, start_date, end_date)
                    if cached_data is not None:
                        self.stats['cache_hits'] += 1
                        continue
                    
                    self.stats['cache_misses'] += 1
                    
                    # Fetch new data
                    try:
                        market_data = load_minute_data(symbol, start_date, end_date)
                        self.data_queue.put({
                            'symbol': symbol,
                            'data': market_data,
                            'start': start_date,
                            'end': end_date,
                            'timestamp': time.time()
                        })
                        self.logger.info(f"Collected data for {symbol}: {len(market_data.df)} rows")
                        
                    except Exception as e:
                        error_msg = f"Failed to collect data for {symbol}: {e}"
                        self.logger.error(error_msg)
                        self.stats['errors'].append({
                            'timestamp': time.time(),
                            'error': error_msg,
                            'symbol': symbol
                        })
                
                # Sleep until next update
                self.stop_event.wait(self.config.update_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in data collection worker: {e}")
                self.stop_event.wait(60)  # Wait 1 minute before retrying
    
    def _data_processing_worker(self):
        """Worker thread for processing collected data."""
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                data_item = self.data_queue.get(timeout=1)
                
                # Process data
                self._process_data_item(data_item)
                self.stats['updates_processed'] += 1
                self.stats['last_update'] = time.time()
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in data processing worker: {e}")
    
    def _process_data_item(self, data_item: Dict[str, Any]):
        """Process a single data item."""
        symbol = data_item['symbol']
        market_data = data_item['data']
        start_date = data_item['start']
        end_date = data_item['end']
        
        # Quality check
        quality_result = self.quality_checker.check_quality(market_data.df)
        
        if not quality_result['passed']:
            self.stats['quality_failures'] += 1
            self.logger.warning(f"Quality check failed for {symbol}: {quality_result['issues']}")
            
            # Try to fix common issues
            market_data = self._fix_data_issues(market_data, quality_result)
        
        # Cache processed data
        self.cache.put(symbol, start_date, end_date, market_data.df)
        
        self.logger.info(f"Processed data for {symbol}: quality_score={quality_result['quality_score']:.3f}")
    
    def _fix_data_issues(self, market_data: MarketData, quality_result: Dict[str, Any]) -> MarketData:
        """Attempt to fix common data issues."""
        df = market_data.df.copy()
        
        # Remove duplicates
        if 'duplicates' in str(quality_result['issues']):
            df = df[~df.index.duplicated(keep='first')]
        
        # Fill missing values with forward fill then backward fill
        if 'low_completeness' in str(quality_result['issues']):
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove extreme outliers (keep within 5 standard deviations)
        if 'outliers' in str(quality_result['issues']) and 'close' in df.columns:
            mean_price = df['close'].mean()
            std_price = df['close'].std()
            df = df[abs(df['close'] - mean_price) <= 5 * std_price]
        
        return MarketData(df)
    
    def _backup_worker(self):
        """Worker thread for periodic data backup."""
        while not self.stop_event.is_set():
            try:
                backup_interval = self.config.backup_interval_hours * 3600
                self.stop_event.wait(backup_interval)
                
                if not self.stop_event.is_set():
                    self._create_backup()
                    
            except Exception as e:
                self.logger.error(f"Error in backup worker: {e}")
    
    def _create_backup(self):
        """Create backup of cache and metadata."""
        backup_dir = Path(self.config.storage_dir) / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"
        
        try:
            # Create backup directory
            backup_path.mkdir(exist_ok=True)
            
            # Copy cache files and metadata
            import shutil
            for file_path in Path(self.config.storage_dir).glob("*.parquet"):
                shutil.copy2(file_path, backup_path)
            
            if self.cache.metadata_file.exists():
                shutil.copy2(self.cache.metadata_file, backup_path)
            
            # Create stats summary
            stats_file = backup_path / "stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            
            self.logger.info(f"Created backup at {backup_path}")
            
            # Cleanup old backups (keep last 10)
            backups = sorted(backup_dir.glob("backup_*"), key=lambda x: x.name)
            for old_backup in backups[:-10]:
                shutil.rmtree(old_backup)
                
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
    
    def get_latest_data(self, symbol: str, lookback_minutes: int = 1440) -> Optional[MarketData]:
        """Get the latest processed data for a symbol."""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")
        
        cached_data = self.cache.get(symbol, start_date, end_date)
        if cached_data is not None:
            # Filter to requested time range
            if isinstance(cached_data.index, pd.DatetimeIndex):
                mask = (cached_data.index >= start_time) & (cached_data.index <= end_time)
                filtered_data = cached_data[mask]
                return MarketData(filtered_data)
            return MarketData(cached_data)
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.stats.copy()
    
    def get_multiple_symbols_data(self, symbols: List[str], lookback_minutes: int = 1440) -> Dict[str, MarketData]:
        """Get latest data for multiple symbols efficiently."""
        result = {}
        
        with ThreadPoolExecutor(max_workers=min(len(symbols), 10)) as executor:
            # Submit all data retrieval tasks
            future_to_symbol = {
                executor.submit(self.get_latest_data, symbol, lookback_minutes): symbol
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None:
                        result[symbol] = data
                except Exception as e:
                    self.logger.error(f"Failed to get data for {symbol}: {e}")
        
        return result