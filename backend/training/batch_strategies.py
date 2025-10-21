#!/usr/bin/env python3
"""
Batch Training Strategies for Stock Prediction

This module implements different batch training strategies optimized for
stock-level batching, where we process stocks in batches rather than
trying to load all data at once.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Generator
import logging
from abc import ABC, abstractmethod
import time

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.batch_iterator import StockBatchIterator, MemoryManager
from prediction.data_loader import DataLoader
from prediction.config import config

logger = logging.getLogger(__name__)


class BatchStrategy(ABC):
    """Abstract base class for batch training strategies."""
    
    def __init__(self, model_name: str, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize batch strategy.
        
        Args:
            model_name: Name of the model being trained
            memory_manager: Optional memory manager for dynamic sizing
        """
        self.model_name = model_name
        self.memory_manager = memory_manager or MemoryManager()
        self.data_loader = DataLoader()
        
    @abstractmethod
    def train_on_stock_batch(self, model, stock_batch: List[str], 
                           batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train model on a batch of stocks.
        
        Args:
            model: Model instance to train
            stock_batch: List of stock symbols for current batch
            batch_info: Information about current batch
            
        Returns:
            Dictionary with training results for this batch
        """
        pass
    
    @abstractmethod
    def supports_incremental_learning(self) -> bool:
        """Check if this strategy supports incremental learning."""
        pass
    
    def get_strategy_name(self) -> str:
        """Get human-readable name of this strategy."""
        return self.__class__.__name__.replace('Strategy', '')


class IncrementalStrategy(BatchStrategy):
    """
    Incremental learning strategy - train model incrementally on each stock batch.
    Best for models that support partial_fit (SGD-based models).
    """
    
    def __init__(self, model_name: str, memory_manager: Optional[MemoryManager] = None):
        super().__init__(model_name, memory_manager)
        self.is_first_batch = True
        
    def supports_incremental_learning(self) -> bool:
        return True
    
    def train_on_stock_batch(self, model, stock_batch: List[str], 
                           batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train model incrementally on stock batch.
        
        For incremental models, we:
        1. Load data for stocks in batch
        2. Use partial_fit to update model
        3. Track metrics per batch
        """
        start_time = time.time()
        batch_results = {
            'stocks_processed': 0,
            'samples_processed': 0,
            'batch_time': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Load and combine data for stocks in this batch
            X_batch, y_batch, symbols_processed = self._load_stock_batch_data(stock_batch)
            
            if len(X_batch) == 0:
                logger.warning(f"No data loaded for batch {batch_info['batch_num']}")
                return batch_results
            
            # Check if model supports incremental learning
            if hasattr(model, 'partial_fit'):
                # Use partial_fit for incremental learning
                if self.is_first_batch:
                    # First batch - initialize model
                    model.partial_fit(X_batch, y_batch)
                    self.is_first_batch = False
                else:
                    # Subsequent batches - update model
                    model.partial_fit(X_batch, y_batch)
            else:
                # Fallback to regular fit for first batch, then accumulate
                if self.is_first_batch:
                    model.fit(X_batch, y_batch)
                    self.is_first_batch = False
                else:
                    # For non-incremental models, we'll accumulate data
                    # This is handled by AccumulateStrategy
                    logger.warning(f"Model {self.model_name} doesn't support partial_fit, "
                                 f"consider using AccumulateStrategy")
                    return batch_results
            
            batch_results.update({
                'stocks_processed': len(symbols_processed),
                'samples_processed': len(X_batch),
                'batch_time': time.time() - start_time,
                'success': True
            })
            
            logger.info(f"[SUCCESS] Incremental training completed for batch {batch_info['batch_num']}: "
                       f"{len(symbols_processed)} stocks, {len(X_batch)} samples")
            
        except Exception as e:
            logger.error(f"[ERROR] Error in incremental training for batch {batch_info['batch_num']}: {e}")
            batch_results['error'] = str(e)
        
        return batch_results
    
    def _load_stock_batch_data(self, stock_batch: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and combine data for a batch of stocks."""
        all_X = []
        all_y = []
        symbols_processed = []
        
        for stock_symbol in stock_batch:
            # Parse symbol and category from stock_symbol
            if '_us_stocks' in stock_symbol:
                symbol = stock_symbol.replace('_us_stocks', '')
                category = 'us_stocks'
            elif '_ind_stocks' in stock_symbol:
                symbol = stock_symbol.replace('_ind_stocks', '')
                category = 'ind_stocks'
            else:
                logger.warning(f"Invalid stock symbol format: {stock_symbol}")
                continue
            
            try:
                # Load stock data from correct category only
                df = self.data_loader.load_stock_data(symbol, category)
                if df is None or len(df) < config.MIN_TRAINING_DAYS:
                    continue
                
                # Validate data quality
                if not self.data_loader.validate_data_quality(df, symbol):
                    continue
                
                # Create features
                df_with_features = self.data_loader.create_features(df)
                if df_with_features is None or len(df_with_features) == 0:
                    continue
                
                # Prepare training data
                X, y = self.data_loader.prepare_training_data(df_with_features)
                if len(X) == 0 or len(y) == 0:
                    continue
                
                # Add to batch data
                all_X.append(X)
                all_y.append(y)
                symbols_processed.append(stock_symbol)
                
            except Exception as e:
                logger.warning(f"Error processing {symbol} in {category}: {e}")
                continue
        
        if not all_X:
            return np.array([]), np.array([]), []
        
        # Combine data for this batch
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        # Clean and scale data
        X_combined = self._clean_data(X_combined)
        y_combined = self._clean_target(y_combined)
        X_combined = self._scale_data(X_combined)
        
        return X_combined, y_combined, symbols_processed
    
    def _clean_data(self, X: np.ndarray) -> np.ndarray:
        """Clean training data by removing infinity and extreme values."""
        X_clean = X.copy()
        
        # Replace infinity with NaN
        X_clean[np.isinf(X_clean)] = np.nan
        
        # Replace extremely large values with NaN
        X_clean[np.abs(X_clean) > 1e6] = np.nan
        
        # For each column, replace NaN with median value
        for i in range(X_clean.shape[1]):
            column = X_clean[:, i]
            if np.any(np.isnan(column)):
                median_val = np.nanmedian(column)
                if np.isnan(median_val):
                    median_val = 0.0
                X_clean[np.isnan(column), i] = median_val
        
        return X_clean
    
    def _clean_target(self, y: np.ndarray) -> np.ndarray:
        """Clean target values."""
        y_clean = y.copy()
        
        # Replace infinity with NaN
        y_clean[np.isinf(y_clean)] = np.nan
        
        # Replace extremely large values with NaN
        y_clean[np.abs(y_clean) > 1e6] = np.nan
        
        # Replace NaN with median value
        if np.any(np.isnan(y_clean)):
            median_val = np.nanmedian(y_clean)
            if np.isnan(median_val):
                median_val = 0.0
            y_clean[np.isnan(y_clean)] = median_val
        
        return y_clean
    
    def _scale_data(self, X: np.ndarray) -> np.ndarray:
        """Scale data using StandardScaler."""
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        except ImportError:
            # Fallback to manual scaling
            X_scaled = X.copy()
            for i in range(X.shape[1]):
                column = X_scaled[:, i]
                if np.std(column) > 0:
                    X_scaled[:, i] = (column - np.mean(column)) / np.std(column)
            return X_scaled


class SubsampleStrategy(BatchStrategy):
    """
    Subsample strategy - train on representative sample of data.
    Best for models that don't support incremental learning (SVM, KNN).
    """
    
    def __init__(self, model_name: str, subsample_percent: int = 50, 
                 memory_manager: Optional[MemoryManager] = None):
        super().__init__(model_name, memory_manager)
        self.subsample_percent = subsample_percent
        self.accumulated_data = {'X': [], 'y': [], 'symbols': []}
        self.is_trained = False
        
    def supports_incremental_learning(self) -> bool:
        return False
    
    def train_on_stock_batch(self, model, stock_batch: List[str], 
                           batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accumulate data from stock batch, then train on subsample.
        """
        start_time = time.time()
        batch_results = {
            'stocks_processed': 0,
            'samples_processed': 0,
            'batch_time': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Load data for this batch
            X_batch, y_batch, symbols_processed = self._load_stock_batch_data(stock_batch)
            
            if len(X_batch) == 0:
                logger.warning(f"No data loaded for batch {batch_info['batch_num']}")
                return batch_results
            
            # Accumulate data
            self.accumulated_data['X'].append(X_batch)
            self.accumulated_data['y'].append(y_batch)
            self.accumulated_data['symbols'].extend(symbols_processed)
            
            # Check if we should train now (every few batches or at the end)
            should_train = (
                batch_info['batch_num'] % 3 == 0 or  # Train every 3rd batch
                batch_info['batch_num'] == batch_info['total_batches'] - 1  # Last batch
            )
            
            if should_train and not self.is_trained:
                # Combine all accumulated data
                X_all = np.vstack(self.accumulated_data['X'])
                y_all = np.concatenate(self.accumulated_data['y'])
                
                # Subsample data
                n_samples = len(X_all)
                n_subsample = int(n_samples * self.subsample_percent / 100)
                
                if n_subsample < n_samples:
                    # Random subsample
                    indices = np.random.choice(n_samples, n_subsample, replace=False)
                    X_subsample = X_all[indices]
                    y_subsample = y_all[indices]
                else:
                    X_subsample, y_subsample = X_all, y_all
                
                # Clean and scale data
                X_subsample = self._clean_data(X_subsample)
                y_subsample = self._clean_target(y_subsample)
                X_subsample = self._scale_data(X_subsample)
                
                # Train model on subsample
                logger.info(f"Training {self.model_name} on {len(X_subsample)} samples "
                           f"({self.subsample_percent}% of {n_samples} total)")
                
                model.fit(X_subsample, y_subsample)
                self.is_trained = True
                
                logger.info(f"[SUCCESS] Subsample training completed: {len(X_subsample)} samples")
            
            batch_results.update({
                'stocks_processed': len(symbols_processed),
                'samples_processed': len(X_batch),
                'batch_time': time.time() - start_time,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Error in subsample training for batch {batch_info['batch_num']}: {e}")
            batch_results['error'] = str(e)
        
        return batch_results
    
    def _load_stock_batch_data(self, stock_batch: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and combine data for a batch of stocks (same as IncrementalStrategy)."""
        all_X = []
        all_y = []
        symbols_processed = []
        
        for stock_symbol in stock_batch:
            # Parse symbol and category from stock_symbol
            if '_us_stocks' in stock_symbol:
                symbol = stock_symbol.replace('_us_stocks', '')
                category = 'us_stocks'
            elif '_ind_stocks' in stock_symbol:
                symbol = stock_symbol.replace('_ind_stocks', '')
                category = 'ind_stocks'
            else:
                logger.warning(f"Invalid stock symbol format: {stock_symbol}")
                continue
            
            try:
                # Load stock data from correct category only
                df = self.data_loader.load_stock_data(symbol, category)
                if df is None or len(df) < config.MIN_TRAINING_DAYS:
                    continue
                
                if not self.data_loader.validate_data_quality(df, symbol):
                    continue
                
                df_with_features = self.data_loader.create_features(df)
                if df_with_features is None or len(df_with_features) == 0:
                    continue
                
                X, y = self.data_loader.prepare_training_data(df_with_features)
                if len(X) == 0 or len(y) == 0:
                    continue
                
                all_X.append(X)
                all_y.append(y)
                symbols_processed.append(stock_symbol)
                
            except Exception as e:
                logger.warning(f"Error processing {symbol} in {category}: {e}")
                continue
        
        if not all_X:
            return np.array([]), np.array([]), []
        
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        return X_combined, y_combined, symbols_processed
    
    def _clean_data(self, X: np.ndarray) -> np.ndarray:
        """Clean training data (same as IncrementalStrategy)."""
        X_clean = X.copy()
        X_clean[np.isinf(X_clean)] = np.nan
        X_clean[np.abs(X_clean) > 1e6] = np.nan
        
        for i in range(X_clean.shape[1]):
            column = X_clean[:, i]
            if np.any(np.isnan(column)):
                median_val = np.nanmedian(column)
                if np.isnan(median_val):
                    median_val = 0.0
                X_clean[np.isnan(column), i] = median_val
        
        return X_clean
    
    def _clean_target(self, y: np.ndarray) -> np.ndarray:
        """Clean target values (same as IncrementalStrategy)."""
        y_clean = y.copy()
        y_clean[np.isinf(y_clean)] = np.nan
        y_clean[np.abs(y_clean) > 1e6] = np.nan
        
        if np.any(np.isnan(y_clean)):
            median_val = np.nanmedian(y_clean)
            if np.isnan(median_val):
                median_val = 0.0
            y_clean[np.isnan(y_clean)] = median_val
        
        return y_clean
    
    def _scale_data(self, X: np.ndarray) -> np.ndarray:
        """Scale data (same as IncrementalStrategy)."""
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        except ImportError:
            X_scaled = X.copy()
            for i in range(X.shape[1]):
                column = X_scaled[:, i]
                if np.std(column) > 0:
                    X_scaled[:, i] = (column - np.mean(column)) / np.std(column)
            return X_scaled


class AccumulateStrategy(BatchStrategy):
    """
    Accumulate strategy - collect all data, then train once.
    Best for models that support warm_start (Random Forest, Decision Tree).
    """
    
    def __init__(self, model_name: str, memory_manager: Optional[MemoryManager] = None):
        super().__init__(model_name, memory_manager)
        self.accumulated_data = {'X': [], 'y': [], 'symbols': []}
        self.is_trained = False
        
    def supports_incremental_learning(self) -> bool:
        return False
    
    def train_on_stock_batch(self, model, stock_batch: List[str], 
                           batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accumulate data from stock batch.
        Train only on the last batch to avoid memory issues.
        """
        start_time = time.time()
        batch_results = {
            'stocks_processed': 0,
            'samples_processed': 0,
            'batch_time': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Load data for this batch
            X_batch, y_batch, symbols_processed = self._load_stock_batch_data(stock_batch)
            
            if len(X_batch) == 0:
                logger.warning(f"No data loaded for batch {batch_info['batch_num']}")
                return batch_results
            
            # Accumulate data
            self.accumulated_data['X'].append(X_batch)
            self.accumulated_data['y'].append(y_batch)
            self.accumulated_data['symbols'].extend(symbols_processed)
            
            # Train on accumulated data (only on last batch to save memory)
            if batch_info['batch_num'] == batch_info['total_batches'] - 1:
                # Last batch - train on all accumulated data
                X_all = np.vstack(self.accumulated_data['X'])
                y_all = np.concatenate(self.accumulated_data['y'])
                
                # Clean and scale data
                X_all = self._clean_data(X_all)
                y_all = self._clean_target(y_all)
                X_all = self._scale_data(X_all)
                
                logger.info(f"Training {self.model_name} on accumulated data: "
                           f"{len(X_all)} samples from {len(self.accumulated_data['symbols'])} stocks")
                
                model.fit(X_all, y_all)
                self.is_trained = True
                
                logger.info(f"[SUCCESS] Accumulate training completed: {len(X_all)} samples")
            
            batch_results.update({
                'stocks_processed': len(symbols_processed),
                'samples_processed': len(X_batch),
                'batch_time': time.time() - start_time,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Error in accumulate training for batch {batch_info['batch_num']}: {e}")
            batch_results['error'] = str(e)
        
        return batch_results
    
    def _load_stock_batch_data(self, stock_batch: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and combine data for a batch of stocks (same as other strategies)."""
        all_X = []
        all_y = []
        symbols_processed = []
        
        for stock_symbol in stock_batch:
            # Parse symbol and category from stock_symbol
            if '_us_stocks' in stock_symbol:
                symbol = stock_symbol.replace('_us_stocks', '')
                category = 'us_stocks'
            elif '_ind_stocks' in stock_symbol:
                symbol = stock_symbol.replace('_ind_stocks', '')
                category = 'ind_stocks'
            else:
                logger.warning(f"Invalid stock symbol format: {stock_symbol}")
                continue
            
            try:
                # Load stock data from correct category only
                df = self.data_loader.load_stock_data(symbol, category)
                if df is None or len(df) < config.MIN_TRAINING_DAYS:
                    continue
                
                if not self.data_loader.validate_data_quality(df, symbol):
                    continue
                
                df_with_features = self.data_loader.create_features(df)
                if df_with_features is None or len(df_with_features) == 0:
                    continue
                
                X, y = self.data_loader.prepare_training_data(df_with_features)
                if len(X) == 0 or len(y) == 0:
                    continue
                
                all_X.append(X)
                all_y.append(y)
                symbols_processed.append(stock_symbol)
                
            except Exception as e:
                logger.warning(f"Error processing {symbol} in {category}: {e}")
                continue
        
        if not all_X:
            return np.array([]), np.array([]), []
        
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        return X_combined, y_combined, symbols_processed
    
    def _clean_data(self, X: np.ndarray) -> np.ndarray:
        """Clean training data (same as other strategies)."""
        X_clean = X.copy()
        X_clean[np.isinf(X_clean)] = np.nan
        X_clean[np.abs(X_clean) > 1e6] = np.nan
        
        for i in range(X_clean.shape[1]):
            column = X_clean[:, i]
            if np.any(np.isnan(column)):
                median_val = np.nanmedian(column)
                if np.isnan(median_val):
                    median_val = 0.0
                X_clean[np.isnan(column), i] = median_val
        
        return X_clean
    
    def _clean_target(self, y: np.ndarray) -> np.ndarray:
        """Clean target values (same as other strategies)."""
        y_clean = y.copy()
        y_clean[np.isinf(y_clean)] = np.nan
        y_clean[np.abs(y_clean) > 1e6] = np.nan
        
        if np.any(np.isnan(y_clean)):
            median_val = np.nanmedian(y_clean)
            if np.isnan(median_val):
                median_val = 0.0
            y_clean[np.isnan(y_clean)] = median_val
        
        return y_clean
    
    def _scale_data(self, X: np.ndarray) -> np.ndarray:
        """Scale data (same as other strategies)."""
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        except ImportError:
            X_scaled = X.copy()
            for i in range(X.shape[1]):
                column = X_scaled[:, i]
                if np.std(column) > 0:
                    X_scaled[:, i] = (column - np.mean(column)) / np.std(column)
            return X_scaled


class KerasMiniBatchStrategy(BatchStrategy):
    """
    Keras mini-batch strategy - use Keras native batching.
    Best for neural networks (ANN, CNN, Autoencoder).
    """
    
    def __init__(self, model_name: str, batch_size: int = 32, 
                 memory_manager: Optional[MemoryManager] = None):
        super().__init__(model_name, memory_manager)
        self.batch_size = batch_size
        self.accumulated_data = {'X': [], 'y': [], 'symbols': []}
        self.is_trained = False
        
    def supports_incremental_learning(self) -> bool:
        return True  # Keras models support mini-batching
    
    def train_on_stock_batch(self, model, stock_batch: List[str], 
                           batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accumulate data and train using Keras mini-batching.
        """
        start_time = time.time()
        batch_results = {
            'stocks_processed': 0,
            'samples_processed': 0,
            'batch_time': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Load data for this batch
            X_batch, y_batch, symbols_processed = self._load_stock_batch_data(stock_batch)
            
            if len(X_batch) == 0:
                logger.warning(f"No data loaded for batch {batch_info['batch_num']}")
                return batch_results
            
            # Accumulate data
            self.accumulated_data['X'].append(X_batch)
            self.accumulated_data['y'].append(y_batch)
            self.accumulated_data['symbols'].extend(symbols_processed)
            
            # Train on accumulated data using Keras mini-batching
            if batch_info['batch_num'] == batch_info['total_batches'] - 1:
                # Last batch - train on all accumulated data
                X_all = np.vstack(self.accumulated_data['X'])
                y_all = np.concatenate(self.accumulated_data['y'])
                
                # Clean and scale data
                X_all = self._clean_data(X_all)
                y_all = self._clean_target(y_all)
                X_all = self._scale_data(X_all)
                
                # Validate data before training
                self._validate_clean_data(X_all, y_all, self.model_name)
                
                logger.info(f"Training {self.model_name} with Keras mini-batching: "
                           f"{len(X_all)} samples, batch_size={self.batch_size}")
                
                # Use model's fit method (batch_size is handled internally)
                model.fit(X_all, y_all)
                self.is_trained = True
                
                logger.info(f"Keras mini-batch training completed: {len(X_all)} samples")
            
            batch_results.update({
                'stocks_processed': len(symbols_processed),
                'samples_processed': len(X_batch),
                'batch_time': time.time() - start_time,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Error in Keras mini-batch training for batch {batch_info['batch_num']}: {e}")
            batch_results['error'] = str(e)
        
        return batch_results
    
    def _load_stock_batch_data(self, stock_batch: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and combine data for a batch of stocks (same as other strategies)."""
        all_X = []
        all_y = []
        symbols_processed = []
        
        for stock_symbol in stock_batch:
            # Parse symbol and category from stock_symbol
            if '_us_stocks' in stock_symbol:
                symbol = stock_symbol.replace('_us_stocks', '')
                category = 'us_stocks'
            elif '_ind_stocks' in stock_symbol:
                symbol = stock_symbol.replace('_ind_stocks', '')
                category = 'ind_stocks'
            else:
                logger.warning(f"Invalid stock symbol format: {stock_symbol}")
                continue
            
            try:
                # Load stock data from correct category only
                df = self.data_loader.load_stock_data(symbol, category)
                if df is None or len(df) < config.MIN_TRAINING_DAYS:
                    continue
                
                if not self.data_loader.validate_data_quality(df, symbol):
                    continue
                
                df_with_features = self.data_loader.create_features(df)
                if df_with_features is None or len(df_with_features) == 0:
                    continue
                
                X, y = self.data_loader.prepare_training_data(df_with_features)
                if len(X) == 0 or len(y) == 0:
                    continue
                
                all_X.append(X)
                all_y.append(y)
                symbols_processed.append(stock_symbol)
                
            except Exception as e:
                logger.warning(f"Error processing {symbol} in {category}: {e}")
                continue
        
        if not all_X:
            return np.array([]), np.array([]), []
        
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        return X_combined, y_combined, symbols_processed
    
    def _clean_data(self, X: np.ndarray) -> np.ndarray:
        """Clean training data (same as other strategies)."""
        X_clean = X.copy()
        X_clean[np.isinf(X_clean)] = np.nan
        X_clean[np.abs(X_clean) > 1e6] = np.nan
        
        for i in range(X_clean.shape[1]):
            column = X_clean[:, i]
            if np.any(np.isnan(column)):
                median_val = np.nanmedian(column)
                if np.isnan(median_val):
                    median_val = 0.0
                X_clean[np.isnan(column), i] = median_val
        
        return X_clean
    
    def _clean_target(self, y: np.ndarray) -> np.ndarray:
        """Clean target values (same as other strategies)."""
        y_clean = y.copy()
        y_clean[np.isinf(y_clean)] = np.nan
        y_clean[np.abs(y_clean) > 1e6] = np.nan
        
        if np.any(np.isnan(y_clean)):
            median_val = np.nanmedian(y_clean)
            if np.isnan(median_val):
                median_val = 0.0
            y_clean[np.isnan(y_clean)] = median_val
        
        return y_clean
    
    def _scale_data(self, X: np.ndarray) -> np.ndarray:
        """Scale data (same as other strategies)."""
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        except ImportError:
            X_scaled = X.copy()
            for i in range(X.shape[1]):
                column = X_scaled[:, i]
                if np.std(column) > 0:
                    X_scaled[:, i] = (column - np.mean(column)) / np.std(column)
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


def get_batch_strategy(model_name: str, config_params: Dict[str, Any] = None) -> BatchStrategy:
    """
    Get appropriate batch strategy for a model.
    
    Args:
        model_name: Name of the model
        config_params: Configuration parameters
        
    Returns:
        Appropriate batch strategy instance
    """
    if config_params is None:
        config_params = {}
    
    # Model-strategy mapping
    strategy_mapping = {
        'linear_regression': IncrementalStrategy,
        'decision_tree': AccumulateStrategy,
        'random_forest': AccumulateStrategy,
        'svm': SubsampleStrategy,
        'knn': SubsampleStrategy,
        'ann': KerasMiniBatchStrategy,
        'cnn': KerasMiniBatchStrategy,
        'arima': SubsampleStrategy,
        'autoencoder': KerasMiniBatchStrategy
    }
    
    strategy_class = strategy_mapping.get(model_name, SubsampleStrategy)
    
    # Create strategy instance with appropriate parameters
    if strategy_class == SubsampleStrategy:
        subsample_percent = config_params.get('subsample_percent', 50)
        return strategy_class(model_name, subsample_percent)
    elif strategy_class == KerasMiniBatchStrategy:
        batch_size = config_params.get('batch_size', 32)
        return strategy_class(model_name, batch_size)
    else:
        return strategy_class(model_name)


# Example usage and testing
if __name__ == "__main__":
    # Test strategy selection
    strategies = ['linear_regression', 'svm', 'ann', 'random_forest']
    
    print("Testing Batch Strategy Selection:")
    for model_name in strategies:
        strategy = get_batch_strategy(model_name)
        print(f"{model_name}: {strategy.get_strategy_name()} "
              f"(incremental: {strategy.supports_incremental_learning()})")
    
    print("\nBatch strategy tests completed successfully!")
