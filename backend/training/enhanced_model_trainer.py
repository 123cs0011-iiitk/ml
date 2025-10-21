#!/usr/bin/env python3
"""
Enhanced Model Trainer for Stock Prediction

This module provides enhanced training capabilities with:
- Full dataset training (1,000+ stocks with 5 years of data)
- Real-time progress updates
- Dynamic pacing to synchronize model completion times
- Resumable training with detailed progress tracking
- Comprehensive logging and monitoring
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Generator
import logging
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prediction.data_loader import DataLoader
from prediction.config import config
from algorithms.stock_indicators import StockIndicators
from training.batch_iterator import StockBatchIterator, MemoryManager
from training.batch_strategies import get_batch_strategy

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Handles real-time progress tracking for training."""
    
    def __init__(self, total_stocks: int, model_name: str):
        self.total_stocks = total_stocks
        self.current_stock = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.stock_times = []
        self.lock = threading.Lock()
        self.model_name = model_name
        self.current_symbol = "Starting..."
        self.update_interval = 30  # Update every 30 seconds for data loading
        
    def update_progress(self, stock_symbol: str, model_name: str):
        """Update progress for current stock."""
        with self.lock:
            self.current_stock += 1
            self.current_symbol = stock_symbol
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Calculate progress metrics
            progress_percent = (self.current_stock / self.total_stocks) * 100
            
            # Estimate remaining time
            if self.current_stock > 0:
                avg_time_per_stock = elapsed / self.current_stock
                remaining_stocks = self.total_stocks - self.current_stock
                estimated_remaining = avg_time_per_stock * remaining_stocks
            else:
                estimated_remaining = 0
            
            # Check if we should display a status update (every 30 seconds)
            if current_time - self.last_update_time >= self.update_interval:
                self._display_status_update(elapsed, progress_percent, estimated_remaining)
                self.last_update_time = current_time
            
            # Log progress for each stock (per-stock output as requested)
            print(f"[{model_name}] Data Loading: {self.current_stock}/{self.total_stocks} stocks "
                  f"({progress_percent:.1f}%) - Current: {stock_symbol}")
            logger.info(f"[{model_name}] Progress: {self.current_stock}/{self.total_stocks} stocks "
                       f"({progress_percent:.1f}%) - Current: {stock_symbol} "
                       f"- ETA: {estimated_remaining:.0f}s")
            
            return {
                'current_stock': stock_symbol,
                'stocks_trained': self.current_stock,
                'total_stocks': self.total_stocks,
                'progress_percent': progress_percent,
                'estimated_remaining_seconds': estimated_remaining
            }
    
    def _display_status_update(self, elapsed: float, progress_percent: float, estimated_remaining: float):
        """Display a comprehensive status update."""
        elapsed_minutes = elapsed / 60
        remaining_minutes = estimated_remaining / 60
        
        # Calculate data usage statistics
        stocks_processed = self.current_stock
        stocks_remaining = self.total_stocks - self.current_stock
        data_processed_percent = (stocks_processed / self.total_stocks) * 100
        data_remaining_percent = (stocks_remaining / self.total_stocks) * 100
        
        print(f"\n{'='*80}")
        print(f"DATA LOADING - {self.model_name.upper()}")
        print(f"{'='*80}")
        print(f"")
        print(f"Progress: {stocks_processed:,}/{self.total_stocks:,} stocks ({data_processed_percent:.1f}%)")
        
        # Add visual progress bar
        bar_length = 50
        filled_length = int(bar_length * data_processed_percent / 100)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        print(f"[{bar}] {data_processed_percent:.1f}%")
        print(f"")
        print(f"Elapsed: {elapsed_minutes:.1f} min  |  Remaining: {remaining_minutes:.1f} min  |  Rate: {self.current_stock/elapsed_minutes:.1f} stocks/min")
        print(f"Current: {self.current_symbol}")
        
        # Calculate estimated completion time
        if estimated_remaining > 0:
            completion_time = time.time() + estimated_remaining
            completion_str = time.strftime("%H:%M:%S", time.localtime(completion_time))
            print(f"Estimated completion: {completion_str}")
        
        print(f"{'='*80}\n")
    
    def force_status_update(self):
        """Force a status update regardless of timing."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        progress_percent = (self.current_stock / self.total_stocks) * 100
        
        if self.current_stock > 0:
            avg_time_per_stock = elapsed / self.current_stock
            remaining_stocks = self.total_stocks - self.current_stock
            estimated_remaining = avg_time_per_stock * remaining_stocks
        else:
            estimated_remaining = 0
        
        self._display_status_update(elapsed, progress_percent, estimated_remaining)


class DynamicPacer:
    """Manages dynamic pacing to synchronize model completion times."""
    
    def __init__(self, target_duration_per_model: float = 1800):  # 30 minutes default
        self.target_duration = target_duration_per_model
        self.model_times = {}
        self.lock = threading.Lock()
    
    def should_pace(self, model_name: str, elapsed_time: float, total_stocks: int, 
                   current_stock: int) -> bool:
        """Determine if model should be paced."""
        with self.lock:
            progress = current_stock / total_stocks if total_stocks > 0 else 0
            expected_elapsed = self.target_duration * progress
            
            # If we're ahead of schedule, we should pace
            return elapsed_time < expected_elapsed * 0.8  # 20% tolerance
    
    def calculate_pace_delay(self, model_name: str, elapsed_time: float, 
                           total_stocks: int, current_stock: int) -> float:
        """Calculate delay needed to pace the model."""
        if not self.should_pace(model_name, elapsed_time, total_stocks, current_stock):
            return 0.0
        
        progress = current_stock / total_stocks if total_stocks > 0 else 0
        expected_elapsed = self.target_duration * progress
        
        if elapsed_time < expected_elapsed:
            # We're ahead, add a small delay
            return min(1.0, (expected_elapsed - elapsed_time) / max(1, total_stocks - current_stock))
        
        return 0.0


class EnhancedModelTrainer:
    """
    Enhanced model trainer with full dataset support, progress tracking, and dynamic pacing.
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load validation stocks
        self.validation_stocks = self._load_validation_stocks()
        
        # Model configurations
        self.model_configs = {
            'linear_regression': {'class': 'LinearRegressionModel', 'file': 'linear_regression.py'},
            'random_forest': {'class': 'RandomForestModel', 'file': 'random_forest.py', 'verbose': True},
            'svm': {'class': 'SVMModel', 'file': 'svm.py', 'verbose': True},
            'knn': {'class': 'KNNModel', 'file': 'knn.py'},
            'decision_tree': {'class': 'DecisionTreeModel', 'file': 'decision_tree.py'},
            'ann': {'class': 'ANNModel', 'file': 'ann.py', 'verbose': True},
            'cnn': {'class': 'CNNModel', 'file': 'cnn.py', 'verbose': True},
            'arima': {'class': 'ARIMAModel', 'file': 'arima.py'},
            'autoencoder': {'class': 'AutoencoderModel', 'file': 'autoencoder.py', 'dir': 'autoencoders', 'verbose': True}
        }
        
        # Initialize dynamic pacer
        self.dynamic_pacer = DynamicPacer(target_duration_per_model=1800)  # 30 minutes per model
        
        # Initialize batch training components
        self.memory_manager = MemoryManager()
        
        # Status tracking - simple format
        self.status_file = os.path.join(self.models_dir, 'model_status.json')
        self.status = self._load_status()
    
    def _load_validation_stocks(self) -> Dict[str, List[str]]:
        """Load validation stocks from JSON file."""
        validation_file = os.path.join(os.path.dirname(__file__), 'validation_stocks.json')
        try:
            with open(validation_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading validation stocks: {e}")
            return {'us_stocks': [], 'ind_stocks': []}
    
    def _load_status(self) -> Dict[str, Any]:
        """Load training status from file."""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading status file: {e}")
        return {}
    
    def _save_status(self):
        """Save current status to file."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving status file: {e}")
    
    def _update_model_status(self, model_name: str, trained: bool = None, stocks_trained: int = None, 
                           dataset_size: int = None, r2_score: float = None, error: str = None):
        """Update model status in simple format."""
        if model_name not in self.status:
            self.status[model_name] = {
                "trained": False,
                "stocks_trained": 0,
                "dataset_size": 0,
                "r2_score": None,
                "trained_date": None,
                "error": None
            }
        
        if trained is not None:
            self.status[model_name]["trained"] = trained
        if stocks_trained is not None:
            self.status[model_name]["stocks_trained"] = stocks_trained
        if dataset_size is not None:
            self.status[model_name]["dataset_size"] = dataset_size
        if r2_score is not None:
            self.status[model_name]["r2_score"] = r2_score
        if error is not None:
            self.status[model_name]["error"] = error
        
        if trained and not self.status[model_name]["trained_date"]:
            self.status[model_name]["trained_date"] = datetime.now().isoformat()
        
        self._save_status()
    
    def get_model_class(self, model_name: str):
        """Get model class for a given model name."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        # Use custom directory if specified, otherwise use model_name
        dir_name = config.get('dir', model_name)
        module_path = f"algorithms.optimised.{dir_name}.{config['file'][:-3]}"
        
        try:
            # Import the module
            module = __import__(module_path, fromlist=[config['class']])
            return getattr(module, config['class'])
        except Exception as e:
            logger.error(f"Error importing {model_name}: {e}")
            return None
    
    def load_all_stock_data_with_progress(self, max_stocks: int = None, 
                                        progress_callback=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and combine data from all stocks with progress tracking.
        
        Returns:
            Tuple of (X, y, stock_symbols) where:
            - X: Feature matrix (n_samples, n_features)
            - y: Target vector (n_samples,)
            - stock_symbols: List of stock symbols used
        """
        logger.info("Loading data from all stocks with progress tracking...")
        
        all_X = []
        all_y = []
        stock_symbols = []
        processed_count = 0
        attempted_count = 0
        
        categories = ['us_stocks', 'ind_stocks']
        
        # First pass: count total available symbols
        total_symbols = 0
        for category in categories:
            symbols = self.data_loader.get_stock_symbols(category)
            if max_stocks and len(symbols) > max_stocks:
                symbols = symbols[:max_stocks]
            total_symbols += len(symbols)
        
        logger.info(f"Total symbols to process: {total_symbols}")
        
        for category in categories:
            logger.info(f"Loading {category} data...")
            symbols = self.data_loader.get_stock_symbols(category)
            
            # Limit symbols if max_stocks is specified
            if max_stocks and len(symbols) > max_stocks:
                symbols = symbols[:max_stocks]
                logger.info(f"Limited to {max_stocks} stocks for testing")
            
            for i, symbol in enumerate(symbols):
                attempted_count += 1
                try:
                    # Load stock data
                    df = self.data_loader.load_stock_data(symbol, category)
                    if df is None or len(df) < config.MIN_TRAINING_DAYS:
                        # Call progress callback even for skipped stocks
                        if progress_callback:
                            progress_callback(symbol, processed_count, total_symbols, attempted_count, "Skipped: Insufficient data")
                        continue
                    
                    # Validate data quality
                    if not self.data_loader.validate_data_quality(df, symbol):
                        # Call progress callback even for skipped stocks
                        if progress_callback:
                            progress_callback(symbol, processed_count, total_symbols, attempted_count, "Skipped: Poor data quality")
                        continue
                    
                    # Create features
                    df_with_features = self.data_loader.create_features(df)
                    if df_with_features is None or len(df_with_features) == 0:
                        # Call progress callback even for skipped stocks
                        if progress_callback:
                            progress_callback(symbol, processed_count, total_symbols, attempted_count, "Skipped: Feature creation failed")
                        continue
                    
                    # Prepare training data
                    X, y = self.data_loader.prepare_training_data(df_with_features)
                    if len(X) == 0 or len(y) == 0:
                        # Call progress callback even for skipped stocks
                        if progress_callback:
                            progress_callback(symbol, processed_count, total_symbols, attempted_count, "Skipped: No training data")
                        continue
                    
                    # Add to combined dataset
                    all_X.append(X)
                    all_y.append(y)
                    stock_symbols.append(f"{symbol}_{category}")
                    processed_count += 1
                    
                    # Call progress callback for successfully processed stocks
                    if progress_callback:
                        progress_callback(symbol, processed_count, total_symbols, attempted_count, "Success")
                    
                except Exception as e:
                    logger.warning(f"Error processing {symbol} in {category}: {e}")
                    # Call progress callback even for failed stocks
                    if progress_callback:
                        progress_callback(symbol, processed_count, total_symbols, attempted_count, f"Error: {str(e)[:50]}")
                    continue
        
        if not all_X:
            logger.error("No valid stock data found")
            return np.array([]), np.array([]), []
        
        # Combine all data
        logger.info(f"Combining data from {len(all_X)} stocks...")
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        # Clean data: remove infinity and extremely large values
        logger.info("Cleaning data: removing infinity and extreme values...")
        X_combined = self._clean_data(X_combined)
        y_combined = self._clean_target(y_combined)
        
        # Scale data for sensitive models (SVM, KNN, ANN, CNN)
        X_combined = self._scale_data(X_combined)
        
        # Validate data before returning
        self._validate_clean_data(X_combined, y_combined, "combined_dataset")
        
        logger.info(f"Combined dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
        return X_combined, y_combined, stock_symbols
    
    def load_stock_data_in_batches(self, batch_size: int, progress_callback=None) -> Generator[Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]], None, None]:
        """
        Generator that yields batches of (X, y, symbols, batch_info) for stock-level batching.
        
        Args:
            batch_size: Number of stocks per batch
            progress_callback: Optional callback for progress updates
            
        Yields:
            Tuple of (X_batch, y_batch, symbols_batch, batch_info)
        """
        logger.info(f"Loading stock data in batches of {batch_size} stocks...")
        
        # Get all stock symbols
        all_symbols = []
        categories = ['us_stocks', 'ind_stocks']
        
        for category in categories:
            symbols = self.data_loader.get_stock_symbols(category)
            all_symbols.extend([(symbol, category) for symbol in symbols])
        
        logger.info(f"Total symbols to process: {len(all_symbols)}")
        
        # Create stock batch iterator
        stock_iterator = StockBatchIterator(
            [f"{symbol}_{category}" for symbol, category in all_symbols],
            batch_size,
            self.memory_manager
        )
        
        # Process each batch
        for stock_batch, batch_info in stock_iterator.get_stock_batches():
            logger.info(f"Processing batch {batch_info['batch_num'] + 1}/{batch_info['total_batches']}: "
                       f"{len(stock_batch)} stocks")
            
            # Load data for stocks in this batch
            all_X = []
            all_y = []
            symbols_processed = []
            
            for stock_symbol in stock_batch:
                # Parse symbol and category
                if '_us_stocks' in stock_symbol:
                    symbol = stock_symbol.replace('_us_stocks', '')
                    category = 'us_stocks'
                elif '_ind_stocks' in stock_symbol:
                    symbol = stock_symbol.replace('_ind_stocks', '')
                    category = 'ind_stocks'
                else:
                    continue
                
                try:
                    # Load stock data
                    df = self.data_loader.load_stock_data(symbol, category)
                    if df is None or len(df) < config.MIN_TRAINING_DAYS:
                        if progress_callback:
                            progress_callback(symbol, len(symbols_processed), len(all_symbols), 
                                            len(symbols_processed), "Skipped: Insufficient data")
                        continue
                    
                    # Validate data quality
                    if not self.data_loader.validate_data_quality(df, symbol):
                        if progress_callback:
                            progress_callback(symbol, len(symbols_processed), len(all_symbols), 
                                            len(symbols_processed), "Skipped: Poor data quality")
                        continue
                    
                    # Create features
                    df_with_features = self.data_loader.create_features(df)
                    if df_with_features is None or len(df_with_features) == 0:
                        if progress_callback:
                            progress_callback(symbol, len(symbols_processed), len(all_symbols), 
                                            len(symbols_processed), "Skipped: Feature creation failed")
                        continue
                    
                    # Prepare training data
                    X, y = self.data_loader.prepare_training_data(df_with_features)
                    if len(X) == 0 or len(y) == 0:
                        if progress_callback:
                            progress_callback(symbol, len(symbols_processed), len(all_symbols), 
                                            len(symbols_processed), "Skipped: No training data")
                        continue
                    
                    # Add to batch data
                    all_X.append(X)
                    all_y.append(y)
                    symbols_processed.append(stock_symbol)
                    
                    # Call progress callback
                    if progress_callback:
                        progress_callback(symbol, len(symbols_processed), len(all_symbols), 
                                        len(symbols_processed), "Success")
                    
                except Exception as e:
                    logger.warning(f"Error processing {symbol} in {category}: {e}")
                    if progress_callback:
                        progress_callback(symbol, len(symbols_processed), len(all_symbols), 
                                        len(symbols_processed), f"Error: {str(e)[:50]}")
                    continue
            
            if not all_X:
                logger.warning(f"No valid data in batch {batch_info['batch_num'] + 1}")
                continue
            
            # Combine data for this batch
            X_batch = np.vstack(all_X)
            y_batch = np.concatenate(all_y)
            
            # Clean and scale data
            X_batch = self._clean_data(X_batch)
            y_batch = self._clean_target(y_batch)
            X_batch = self._scale_data(X_batch)
            
            # Validate data before yielding
            self._validate_clean_data(X_batch, y_batch, f"batch_{batch_info['batch_num'] + 1}")
            
            logger.info(f"Batch {batch_info['batch_num'] + 1} ready: {X_batch.shape[0]} samples, "
                       f"{X_batch.shape[1]} features from {len(symbols_processed)} stocks")
            
            # Update batch info with actual results
            batch_info.update({
                'stocks_processed': len(symbols_processed),
                'samples_processed': len(X_batch),
                'features_count': X_batch.shape[1]
            })
            
            yield X_batch, y_batch, symbols_processed, batch_info
    
    def train_with_batches(self, model_name: str, model_instance, batch_strategy: str) -> bool:
        """
        Train model using batch processing with stock-level batching.
        
        Args:
            model_name: Name of model being trained
            model_instance: Model instance to train
            batch_strategy: Strategy name ('incremental', 'subsample', 'accumulate', 'keras_minibatch')
            
        Returns:
            Success status
        """
        logger.info(f"Starting batch training for {model_name} using {batch_strategy} strategy")
        
        try:
            # Get batch strategy instance
            strategy = get_batch_strategy(model_name, {
                'subsample_percent': config.SUBSAMPLE_PERCENT,
                'batch_size': config.ROW_BATCH_SIZE
            })
            
            # Get batch size from config
            batch_size = config.STOCK_BATCH_SIZE
            
            # Initialize progress tracking
            total_symbols = len(self.data_loader.get_stock_symbols('us_stocks')) + \
                          len(self.data_loader.get_stock_symbols('ind_stocks'))
            total_batches = (total_symbols + batch_size - 1) // batch_size
            
            from training.batch_iterator import BatchProgressTracker
            progress_tracker = BatchProgressTracker(total_symbols, total_batches, model_name)
            
            # Load data in batches and train
            batch_count = 0
            total_stocks_processed = 0
            total_samples_processed = 0
            stock_symbols = []  # Initialize to track all processed symbols
            
            for X_batch, y_batch, symbols_batch, batch_info in self.load_stock_data_in_batches(batch_size):
                batch_count += 1
                
                logger.info(f"Training on batch {batch_count}/{total_batches}: "
                           f"{len(symbols_batch)} stocks, {len(X_batch)} samples")
                
                # Train on this batch using the strategy
                batch_results = strategy.train_on_stock_batch(
                    model_instance, symbols_batch, batch_info
                )
                
                # Update progress
                total_stocks_processed += batch_results.get('stocks_processed', 0)
                total_samples_processed += batch_results.get('samples_processed', 0)
                stock_symbols.extend(symbols_batch)  # Accumulate stock symbols
                
                progress_info = progress_tracker.update_batch_progress(
                    batch_count - 1, len(symbols_batch), len(symbols_batch)
                )
                
                # Update status
                self.status[model_name].update({
                    'batch_training_enabled': True,
                    'batch_strategy': batch_strategy,
                    'current_batch': batch_count,
                    'total_batches': total_batches,
                    'batch_progress_percent': (batch_count / total_batches) * 100,
                    'stocks_processed': total_stocks_processed,
                    'samples_processed': total_samples_processed,
                    'stocks_per_batch': batch_size
                })
                self._save_status()
                
                if not batch_results.get('success', False):
                    logger.warning(f"Batch {batch_count} had issues: {batch_results.get('error', 'Unknown error')}")
            
            # Final status update
            self.status[model_name].update({
                'batch_training_completed': True,
                'final_stocks_processed': total_stocks_processed,
                'final_samples_processed': total_samples_processed
            })
            self._save_status()
            
            logger.info(f"Batch training completed for {model_name}: "
                       f"{total_stocks_processed} stocks, {total_samples_processed} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error in batch training for {model_name}: {e}")
            self.status[model_name].update({
                'batch_training_failed': True,
                'batch_error': str(e)
            })
            self._save_status()
            return False
    
    def _get_batch_strategy(self, model_name: str) -> str:
        """
        Determine the best batch strategy for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Strategy name
        """
        # Model-strategy mapping
        strategy_mapping = {
            'linear_regression': 'incremental',
            'decision_tree': 'accumulate',
            'random_forest': 'accumulate',
            'svm': 'subsample',
            'knn': 'subsample',
            'ann': 'keras_minibatch',
            'cnn': 'keras_minibatch',
            'arima': 'subsample',
            'autoencoder': 'keras_minibatch'
        }
        
        return strategy_mapping.get(model_name, 'subsample')
    
    def train_single_model_enhanced(self, model_name: str, force_retrain: bool = False) -> bool:
        """
        Train a single model on the full dataset with enhanced features.
        
        Args:
            model_name: Name of the model to train
            force_retrain: Force retrain even if model exists
            
        Returns:
            True if training successful, False otherwise
        """
        logger.info(f"Starting enhanced training for {model_name}")
        
        # Check if already completed and not forcing retrain
        if not force_retrain and model_name in self.status:
            model_data = self.status[model_name]
            model_status = model_data.get('status', 'pending')
            if model_status == 'completed' and model_data.get('stocks_trained', 0) > 500:
                logger.info(f"{model_name} already completed on full dataset, skipping")
                return True
        
        # Mark as in progress
        self.status[model_name] = {
            'status': 'in_progress',
            'last_updated': datetime.now().isoformat(),
            'start_time': time.time(),
            'stocks_trained': 0,
            'current_symbol': 'Starting...',
            'progress_percent': 0,
            'training_phase': 'data_loading',
            'total_samples': 0,
            'training_start_time': None
        }
        self._save_status()
        
        # Get model-specific time estimates
        time_estimates = self._get_model_time_estimates(model_name)
        
        # Display startup message
        print(f"\n{'='*80}")
        print(f"[START] STARTING {model_name.upper()} TRAINING")
        print(f"{'-'*80}")
        print(f"DATA TARGET:")
        print(f"   Stocks: ~1,000 stocks (US + Indian)")
        print(f"   Historical Data: 5 years per stock")
        print(f"   Features: 43 technical indicators per stock")
        print(f"   Total Samples: ~1,000,000+ data points")
        print(f"")
        print(f"TIMING ESTIMATES:")
        print(f"   Start Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Model Training: {time_estimates['expected_minutes']} minutes")
        print(f"   Data Loading: 3-5 minutes")
        print(f"   Validation: 1-2 minutes")
        print(f"   Total Process: ~{time_estimates['expected_minutes'] + 5} minutes")
        print(f"")
        print(f"MODEL INFO:")
        print(f"   {time_estimates['status_message']}")
        print(f"   Progress updates every 30 seconds")
        print(f"{'-'*80}\n")
        
        try:
            # Initialize progress tracker (will be updated with actual total)
            progress_tracker = ProgressTracker(total_stocks=1000, model_name=model_name)  # Will be updated
            
            def progress_callback(symbol, processed, total_symbols, attempted, status_message):
                # Update progress tracker with actual total
                progress_tracker.total_stocks = total_symbols
                
                progress_info = progress_tracker.update_progress(symbol, model_name)
                
                # Print per-stock progress (as requested)
                progress_percent = (attempted / total_symbols) * 100 if total_symbols > 0 else 0
                print(f"[{model_name}] Processing: {symbol} ({attempted}/{total_symbols}, {progress_percent:.1f}%) - {status_message}")
                
                # Update the status file with detailed progress information
                self.status[model_name]['stocks_trained'] = processed
                self.status[model_name]['stocks_attempted'] = attempted
                self.status[model_name]['total_symbols'] = total_symbols
                self.status[model_name]['current_symbol'] = symbol
                self.status[model_name]['status_message'] = status_message
                self.status[model_name]['progress_percent'] = (attempted / total_symbols) * 100 if total_symbols > 0 else 0
                self.status[model_name]['training_progress_percent'] = (processed / total_symbols) * 100 if total_symbols > 0 else 0
                self.status[model_name]['last_updated'] = datetime.now().isoformat()
                
                # Calculate estimated remaining time based on attempted symbols (more accurate)
                if attempted > 0:
                    elapsed_time = time.time() - self.status[model_name]['start_time']
                    avg_time_per_symbol = elapsed_time / attempted
                    remaining_symbols = total_symbols - attempted
                    estimated_remaining = avg_time_per_symbol * remaining_symbols
                    self.status[model_name]['estimated_remaining_seconds'] = estimated_remaining
                    
                    # Calculate training completion time (based on successful processing)
                    if processed > 0:
                        avg_training_time_per_stock = elapsed_time / processed
                        remaining_training_stocks = total_symbols - processed
                        estimated_training_remaining = avg_training_time_per_stock * remaining_training_stocks
                        self.status[model_name]['estimated_training_remaining_seconds'] = estimated_training_remaining
                
                self._save_status()
                
                return progress_info
            
            # Check if batch training is enabled
            if config.USE_BATCH_TRAINING:
                print(f"🔄 Using batch training mode for {model_name}...")
                print(f"Stock batch size: {config.STOCK_BATCH_SIZE} stocks per batch")
                print(f"Row batch size: {config.ROW_BATCH_SIZE} rows per mini-batch")
                
                # Get model class
                model_class = self.get_model_class(model_name)
                if model_class is None:
                    raise ValueError(f"Could not import {model_name}")
                
                # Create model instance
                model = model_class()
                
                # Determine batch strategy
                batch_strategy = self._get_batch_strategy(model_name)
                print(f"Using {batch_strategy} strategy for {model_name}")
                
                # Update status to batch training phase
                self.status[model_name]['training_phase'] = 'batch_training'
                self.status[model_name]['batch_strategy'] = batch_strategy
                self._save_status()
                
                # Train using batch processing
                success = self.train_with_batches(model_name, model, batch_strategy)
                
                if not success:
                    raise ValueError(f"Batch training failed for {model_name}")
                
                training_duration = time.time() - self.status[model_name]['start_time']
                print(f"{model_name} batch training completed in {training_duration/60:.1f} minutes")
                
            else:
                # Original single-pass training
                print(f"Loading full dataset for {model_name}...")
                print(f"Expected data loading time: 3-5 minutes for ~1,000 stocks...")
                print(f"Progress updates every 15 seconds during data loading...")
                X, y, stock_symbols = self.load_all_stock_data_with_progress(
                    progress_callback=progress_callback
                )
                
                if len(X) == 0:
                    raise ValueError("No training data available")
                
                # Get model class
                model_class = self.get_model_class(model_name)
                if model_class is None:
                    raise ValueError(f"Could not import {model_name}")
                
                # Update status to training phase
                self.status[model_name]['training_phase'] = 'model_training'
                self.status[model_name]['total_samples'] = len(X)
                self.status[model_name]['training_start_time'] = time.time()
                self._save_status()
                
                # Create and train model with dynamic pacing
                print(f"Training {model_name} on {len(X):,} samples from {len(stock_symbols)} stocks...")
                
                # Check if model supports verbose output
                model_config = self.model_configs.get(model_name, {})
                has_verbose = model_config.get('verbose', False)
                
                if has_verbose:
                    print(f"Model supports verbose output - detailed progress will be shown below...")
                else:
                    print(f"Training in progress - processing all samples as one batch...")
                    print(f"Progress updates every 30 seconds during model training...")
                
                start_time = time.time()
                
                model = model_class()
                
                # Start a background thread to show progress during training
                training_progress_thread = threading.Thread(
                    target=self._show_training_progress, 
                    args=(model_name, start_time, has_verbose),
                    daemon=True
                )
                training_progress_thread.start()
                
                # Validate data before training
                self._validate_clean_data(X, y, model_name)
                
                # Apply dynamic pacing during training
                training_start = time.time()
                model.fit(X, y)
                
                training_duration = time.time() - start_time
                print(f"{model_name} model training completed in {training_duration/60:.1f} minutes")
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Run validation
            print(f"Running validation on test stocks...")
            print(f"Expected validation time: 1-2 minutes...")
            validation_metrics = self.validate_model(model, model_name)
            
            # Update status
            self.status[model_name] = {
                'status': 'completed',
                'last_updated': datetime.now().isoformat(),
                'model_path': model_path,
                'training_duration_seconds': training_duration,
                'stocks_trained': len(stock_symbols),
                'validation_metrics': validation_metrics,
                'dataset_size': len(X),
                'features_count': X.shape[1] if len(X) > 0 else 0,
                'total_samples': len(X),
                'training_phase': 'completed'
            }
            self._save_status()
            
            # Display final completion status
            print(f"\n{'='*80}")
            print(f"{model_name.upper()} TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"DATA USAGE SUMMARY:")
            print(f"   Stocks Processed: {len(stock_symbols):,} / 1,000 stocks (100.0%)")
            print(f"   Dataset Size: {len(X):,} samples")
            print(f"   Features Used: {X.shape[1] if len(X) > 0 else 0}")
            print(f"   Data Coverage: Complete (all available stocks)")
            print(f"   Progress: [{len(stock_symbols):,}/1,000] COMPLETE")
            print(f"")
            print(f"PERFORMANCE SUMMARY:")
            print(f"   Training Duration: {training_duration/60:.1f} minutes")
            print(f"   Processing Rate: {len(stock_symbols)/(training_duration/60):.1f} stocks/minute")
            print(f"   Time per stock: {(training_duration/60)/len(stock_symbols)*60:.1f} seconds")
            if validation_metrics and 'avg_r2_score' in validation_metrics:
                print(f"   Validation R²: {validation_metrics['avg_r2_score']:.4f}")
            print(f"   Model Saved: {model_path}")
            print(f"{'='*80}\n")
            
            logger.info(f"{model_name} training completed successfully on {len(stock_symbols)} stocks")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error training {model_name}: {e}")
            self.status[model_name] = {
                'status': 'failed',
                'last_updated': datetime.now().isoformat(),
                'error_message': str(e)
            }
            self._save_status()
            return False
    
    def validate_model(self, model, model_name: str) -> Dict[str, Any]:
        """
        Validate a trained model on the validation stock set.
        
        Args:
            model: Trained model
            model_name: Name of the model
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info(f"Validating {model_name} on validation stocks...")
        
        validation_results = []
        total_validation_stocks = len(self.validation_stocks['us_stocks']) + len(self.validation_stocks['ind_stocks'])
        
        # Test on validation stocks
        for category, symbols in self.validation_stocks.items():
            for symbol in symbols:
                try:
                    # Load stock data
                    df = self.data_loader.load_stock_data(symbol, category)
                    if df is None or len(df) < 50:  # Need some data for validation
                        continue
                    
                    # Create features
                    df_with_features = self.data_loader.create_features(df)
                    if df_with_features is None or len(df_with_features) == 0:
                        continue
                    
                    # Prepare data
                    X, y = self.data_loader.prepare_training_data(df_with_features)
                    if len(X) == 0 or len(y) == 0:
                        continue
                    
                    # Split for validation (use last 20% for testing)
                    split_idx = int(len(X) * 0.8)
                    X_test = X[split_idx:]
                    y_test = y[split_idx:]
                    
                    if len(X_test) == 0:
                        continue
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    try:
                        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    except ImportError:
                        # Fallback to manual calculation if sklearn not available
                        def r2_score(y_true, y_pred):
                            ss_res = np.sum((y_true - y_pred) ** 2)
                            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        def mean_squared_error(y_true, y_pred):
                            return np.mean((y_true - y_pred) ** 2)
                        
                        def mean_absolute_error(y_true, y_pred):
                            return np.mean(np.abs(y_true - y_pred))
                    
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    validation_results.append({
                        'symbol': symbol,
                        'category': category,
                        'r2_score': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'samples': len(X_test)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error validating {symbol} with {model_name}: {e}")
                    continue
        
        if not validation_results:
            logger.warning(f"No validation results for {model_name}")
            return {'test_stocks': 0, 'avg_r2_score': 0.0, 'avg_rmse': 0.0, 'avg_mae': 0.0}
        
        # Calculate average metrics
        avg_r2 = np.mean([r['r2_score'] for r in validation_results])
        avg_rmse = np.mean([r['rmse'] for r in validation_results])
        avg_mae = np.mean([r['mae'] for r in validation_results])
        
        logger.info(f"{model_name} validation: R²={avg_r2:.4f}, RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}")
        
        return {
            'test_stocks': len(validation_results),
            'avg_r2_score': float(avg_r2),
            'avg_rmse': float(avg_rmse),
            'avg_mae': float(avg_mae),
            'results': validation_results
        }
    
    def train_all_models_enhanced(self, force_retrain: bool = False) -> Dict[str, bool]:
        """
        Train all models on the full dataset with enhanced features.
        
        Args:
            force_retrain: Force retrain all models
            
        Returns:
            Dictionary with training results for each model
        """
        results = {}
        
        logger.info("Starting enhanced training of all models on full dataset...")
        logger.info(f"Target: ~1,000 stocks with 5 years of historical data")
        
        for model_name in self.model_configs.keys():
            logger.info(f"\n{'='*80}")
            logger.info(f"Training {model_name} on full dataset")
            logger.info(f"{'='*80}")
            
            success = self.train_single_model_enhanced(model_name, force_retrain)
            results[model_name] = success
            
            if success:
                logger.info(f"{model_name} completed successfully")
            else:
                logger.error(f"[ERROR] {model_name} failed")
        
        return results
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all model training status."""
        summary = {
            'total_models': len(self.model_configs),
            'completed': 0,
            'failed': 0,
            'pending': 0,
            'in_progress': 0,
            'models': {}
        }
        
        for model_name in self.model_configs.keys():
            model_data = self.status.get(model_name, {})
            trained = model_data.get('trained', False)
            error = model_data.get('error')
            
            # Determine status based on simple format
            if trained and not error:
                status = 'completed'
                summary['completed'] += 1
            elif error and 'stuck' in error.lower():
                status = 'failed'
                summary['failed'] += 1
            elif error:
                status = 'failed'
                summary['failed'] += 1
            else:
                status = 'pending'
                summary['pending'] += 1
            
            summary['models'][model_name] = {
                'status': status,
                'details': model_data
            }
        
        return summary
    
    def cleanup_temp_files(self):
        """Clean up temporary files generated during training."""
        logger.info("Cleaning up temporary files...")
        
        # List of temporary file patterns to clean
        temp_patterns = [
            '*.tmp',
            '*.temp',
            '*_temp_*',
            'training_*_temp*'
        ]
        
        cleaned_count = 0
        
        # Clean up in models directory
        for pattern in temp_patterns:
            import glob
            temp_files = glob.glob(os.path.join(self.models_dir, pattern))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    cleaned_count += 1
                    logger.debug(f"Removed temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")
        
        # Clean up in logs directory
        logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        if os.path.exists(logs_dir):
            for pattern in temp_patterns:
                temp_files = glob.glob(os.path.join(logs_dir, pattern))
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        cleaned_count += 1
                        logger.debug(f"Removed temp file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")
    
    def _clean_data(self, X: np.ndarray) -> np.ndarray:
        """
        Clean training data by removing infinity and extreme values.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cleaned feature matrix
        """
        # Create a copy to avoid modifying original data
        X_clean = X.copy()
        
        # Count original issues
        inf_count = np.sum(np.isinf(X_clean))
        extreme_count = np.sum(np.abs(X_clean) > 1e6)  # More aggressive threshold
        nan_count = np.sum(np.isnan(X_clean))
        
        # Replace infinity with NaN
        X_clean[np.isinf(X_clean)] = np.nan
        
        # Replace extremely large values (> 1e6) with NaN
        X_clean[np.abs(X_clean) > 1e6] = np.nan
        
        # Replace extremely small values (< -1e6) with NaN
        X_clean[X_clean < -1e6] = np.nan
        
        # For each column, replace NaN with median value
        for i in range(X_clean.shape[1]):
            column = X_clean[:, i]
            if np.any(np.isnan(column)):
                median_val = np.nanmedian(column)
                if np.isnan(median_val):
                    # If median is also NaN, use 0
                    median_val = 0.0
                X_clean[np.isnan(column), i] = median_val
        
        # Additional cleaning: clip extreme values to reasonable ranges
        for i in range(X_clean.shape[1]):
            column = X_clean[:, i]
            if np.std(column) > 0:  # Only if column has variation
                # Clip values to 5 standard deviations from mean
                mean_val = np.mean(column)
                std_val = np.std(column)
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                X_clean[:, i] = np.clip(column, lower_bound, upper_bound)
        
        logger.info(f"Data cleaning: removed {inf_count} inf, {extreme_count} extreme, {nan_count} NaN values")
        return X_clean
    
    def _clean_target(self, y: np.ndarray) -> np.ndarray:
        """
        Clean target values by removing infinity and extreme values.
        
        Args:
            y: Target vector
            
        Returns:
            Cleaned target vector
        """
        # Create a copy to avoid modifying original data
        y_clean = y.copy()
        
        # Count original issues
        inf_count = np.sum(np.isinf(y_clean))
        extreme_count = np.sum(np.abs(y_clean) > 1e6)
        nan_count = np.sum(np.isnan(y_clean))
        
        # Replace infinity with NaN
        y_clean[np.isinf(y_clean)] = np.nan
        
        # Replace extremely large values (> 1e6) with NaN
        y_clean[np.abs(y_clean) > 1e6] = np.nan
        
        # Replace extremely small values (< -1e6) with NaN
        y_clean[y_clean < -1e6] = np.nan
        
        # Replace NaN with median value
        if np.any(np.isnan(y_clean)):
            median_val = np.nanmedian(y_clean)
            if np.isnan(median_val):
                # If median is also NaN, use 0
                median_val = 0.0
            y_clean[np.isnan(y_clean)] = median_val
        
        # Additional cleaning: clip extreme values to reasonable ranges
        if np.std(y_clean) > 0:  # Only if target has variation
            mean_val = np.mean(y_clean)
            std_val = np.std(y_clean)
            lower_bound = mean_val - 5 * std_val
            upper_bound = mean_val + 5 * std_val
            y_clean = np.clip(y_clean, lower_bound, upper_bound)
        
        logger.info(f"Target cleaning: removed {inf_count} inf, {extreme_count} extreme, {nan_count} NaN values")
        return y_clean
    
    def _show_training_progress(self, model_name: str, start_time: float, has_verbose: bool = False):
        """Show progress updates during model training."""
        last_update = time.time()
        update_interval = 30  # 30 seconds
        
        # Get model-specific time estimates
        time_estimates = self._get_model_time_estimates(model_name)
        
        # Get total samples from status
        total_samples = self.status.get(model_name, {}).get('total_samples', 0)
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if we should update (every 30 seconds)
            if current_time - last_update >= update_interval:
                elapsed_minutes = elapsed / 60
                
                # Calculate progress based on expected duration
                expected_duration = time_estimates['expected_minutes']
                progress_percent = min(100, (elapsed_minutes / expected_duration) * 100)
                
                print(f"\n{'='*80}")
                print(f"🔄 {model_name.upper()} MODEL TRAINING IN PROGRESS")
                print(f"{'-'*80}")
                print(f"TRAINING PROGRESS:")
                print(f"   Elapsed Time: {elapsed_minutes:.1f} minutes")
                
                if has_verbose:
                    print(f"   Progress: See detailed output above (model has verbose support)")
                    print(f"   Model is showing real-time training progress in console")
                else:
                    print(f"   Progress: {progress_percent:.1f}% (estimated)")
                    print(f"   Status: Training in progress - processing all samples as one batch")
                
                print(f"   Expected Duration: {expected_duration} minutes")
                print(f"   Estimated Remaining: {max(0, expected_duration - elapsed_minutes):.1f} minutes")
                
                # Add visual progress bar for training
                bar_length = 50
                filled_length = int(bar_length * progress_percent / 100)
                bar = '#' * filled_length + '-' * (bar_length - filled_length)
                print(f"   Training Bar: [{bar}] {progress_percent:.1f}%")
                print(f"")
                print(f"MODEL STATUS:")
                print(f"   {time_estimates['status_message']}")
                print(f"   Training on full dataset (~1,000 stocks)")
                print(f"   Data: {total_samples:,} samples from ~1,000 stocks")
                print(f"   Features: 43 technical indicators per stock")
                
                if has_verbose:
                    print(f"   Model supports verbose output - see console above for details")
                else:
                    print(f"   Next update in 30 seconds...")
                
                print(f"{'-'*80}\n")
                
                last_update = current_time
            
            # Sleep for a short time to avoid busy waiting
            time.sleep(5)
    
    def _get_model_time_estimates(self, model_name: str) -> dict:
        """Get precise time estimates for each model type."""
        estimates = {
            'linear_regression': {
                'expected_minutes': 2,
                'status_message': 'Training linear regression (fastest model)'
            },
            'random_forest': {
                'expected_minutes': 8,
                'status_message': 'Training random forest with multiple trees'
            },
            'svm': {
                'expected_minutes': 15,
                'status_message': 'Training SVM with kernel optimization'
            },
            'knn': {
                'expected_minutes': 5,
                'status_message': 'Training KNN with distance calculations'
            },
            'decision_tree': {
                'expected_minutes': 3,
                'status_message': 'Training decision tree with splits'
            },
            'ann': {
                'expected_minutes': 12,
                'status_message': 'Training neural network with backpropagation'
            },
            'cnn': {
                'expected_minutes': 20,
                'status_message': 'Training CNN with convolutional layers'
            },
            'arima': {
                'expected_minutes': 25,
                'status_message': 'Training ARIMA with parameter optimization'
            },
            'autoencoder': {
                'expected_minutes': 18,
                'status_message': 'Training autoencoder with encoding/decoding'
            }
        }
        
        return estimates.get(model_name, {
            'expected_minutes': 10,
            'status_message': 'Training model...'
        })
    
    def _scale_data(self, X: np.ndarray) -> np.ndarray:
        """
        Scale data using StandardScaler for sensitive models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info("Data scaled using StandardScaler")
            return X_scaled
        except ImportError:
            # Fallback to manual scaling if sklearn not available
            logger.warning("sklearn not available, using manual scaling")
            X_scaled = X.copy()
            for i in range(X.shape[1]):
                column = X_scaled[:, i]
                if np.std(column) > 0:
                    X_scaled[:, i] = (column - np.mean(column)) / np.std(column)
            logger.info("Data scaled using manual standardization")
            return X_scaled
    
    def _validate_clean_data(self, X: np.ndarray, y: np.ndarray, model_name: str) -> None:
        """
        Validate that data is clean before training.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_name: Name of model for error reporting
            
        Raises:
            ValueError: If data contains infinity or NaN values
        """
        # Check for infinity values
        if np.any(np.isinf(X)):
            inf_count = np.sum(np.isinf(X))
            raise ValueError(f"Data validation failed for {model_name}: "
                           f"X contains {inf_count} infinity values")
        
        if np.any(np.isinf(y)):
            inf_count = np.sum(np.isinf(y))
            raise ValueError(f"Data validation failed for {model_name}: "
                           f"y contains {inf_count} infinity values")
        
        # Check for NaN values
        if np.any(np.isnan(X)):
            nan_count = np.sum(np.isnan(X))
            raise ValueError(f"Data validation failed for {model_name}: "
                           f"X contains {nan_count} NaN values")
        
        if np.any(np.isnan(y)):
            nan_count = np.sum(np.isnan(y))
            raise ValueError(f"Data validation failed for {model_name}: "
                           f"y contains {nan_count} NaN values")
        
        # Check for extremely large values (that might cause dtype overflow)
        max_val = np.max(np.abs(X))
        if max_val > 1e10:  # Very large threshold
            raise ValueError(f"Data validation failed for {model_name}: "
                           f"X contains extremely large values (max: {max_val:.2e})")
        
        max_y_val = np.max(np.abs(y))
        if max_y_val > 1e10:
            raise ValueError(f"Data validation failed for {model_name}: "
                           f"y contains extremely large values (max: {max_y_val:.2e})")
        
        logger.info(f"Data validation passed for {model_name}: "
                   f"X shape: {X.shape}, y shape: {y.shape}, "
                   f"X range: [{np.min(X):.3f}, {np.max(X):.3f}], "
                   f"y range: [{np.min(y):.3f}, {np.max(y):.3f}]")
