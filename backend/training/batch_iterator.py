#!/usr/bin/env python3
"""
Batch Iterator for Stock Prediction Training

This module provides efficient batch iteration for large datasets,
supporting both stock-level and row-level batching with memory management.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Optional, Dict, Any
import logging
import time
import psutil
from abc import ABC, abstractmethod

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


class MemoryManager:
    """Monitor and manage memory during batch processing."""
    
    def __init__(self, max_memory_usage: float = 0.8):
        """
        Initialize memory manager.
        
        Args:
            max_memory_usage: Maximum memory usage as fraction (0.0-1.0)
        """
        self.max_memory_usage = max_memory_usage
        self.process = psutil.Process()
    
    def get_available_memory(self) -> int:
        """Get available system memory in MB."""
        try:
            memory = psutil.virtual_memory()
            return int(memory.available / (1024 * 1024))  # Convert to MB
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return 1024  # Default to 1GB
    
    def get_current_memory_usage(self) -> int:
        """Get current process memory usage in MB."""
        try:
            return int(self.process.memory_info().rss / (1024 * 1024))
        except Exception as e:
            logger.warning(f"Could not get process memory: {e}")
            return 0
    
    def get_memory_usage_percent(self) -> float:
        """Get current memory usage as percentage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception as e:
            logger.warning(f"Could not get memory percentage: {e}")
            return 0.0
    
    def estimate_batch_memory(self, batch_size: int, features: int, 
                            data_type: str = 'float64') -> int:
        """
        Estimate memory needed for batch in MB.
        
        Args:
            batch_size: Number of samples in batch
            features: Number of features per sample
            data_type: Data type ('float64', 'float32', etc.)
            
        Returns:
            Estimated memory usage in MB
        """
        # Calculate bytes per element based on data type
        type_sizes = {
            'float64': 8,
            'float32': 4,
            'int64': 8,
            'int32': 4
        }
        bytes_per_element = type_sizes.get(data_type, 8)
        
        # Memory for X and y arrays
        x_memory = batch_size * features * bytes_per_element
        y_memory = batch_size * bytes_per_element
        
        # Add overhead (scaling, preprocessing, etc.)
        total_memory = (x_memory + y_memory) * 2  # 2x overhead
        
        return int(total_memory / (1024 * 1024))  # Convert to MB
    
    def adjust_batch_size(self, current_size: int, features: int, 
                         target_memory_mb: int = None) -> int:
        """
        Dynamically adjust batch size based on memory constraints.
        
        Args:
            current_size: Current batch size
            features: Number of features
            target_memory_mb: Target memory usage in MB
            
        Returns:
            Adjusted batch size
        """
        if target_memory_mb is None:
            available_memory = self.get_available_memory()
            target_memory_mb = int(available_memory * self.max_memory_usage)
        
        # Estimate current memory usage
        current_memory = self.estimate_batch_memory(current_size, features)
        
        if current_memory > target_memory_mb:
            # Reduce batch size proportionally
            reduction_factor = target_memory_mb / current_memory
            new_size = max(1, int(current_size * reduction_factor))
            logger.info(f"Reducing batch size from {current_size} to {new_size} "
                       f"due to memory constraints")
            return new_size
        
        return current_size
    
    def adjust_batch_size_dynamically(self, current_size: int, memory_threshold: float = 0.75) -> int:
        """
        Dynamically adjust batch size based on current memory usage.
        
        Args:
            current_size: Current batch size
            memory_threshold: Threshold (0-1) for triggering reduction
            
        Returns:
            Adjusted batch size
        """
        current_usage = self.get_memory_usage_percent()
        
        if current_usage > memory_threshold:
            # Reduce batch size proportionally
            reduction_factor = memory_threshold / current_usage
            new_size = max(1, int(current_size * reduction_factor * 0.8))  # 20% buffer
            logger.warning(f"Memory usage {current_usage:.1%} exceeds threshold {memory_threshold:.1%}")
            logger.warning(f"Reducing batch size from {current_size} to {new_size}")
            return new_size
        elif current_usage < memory_threshold * 0.5 and current_size < 10000:
            # Can increase batch size if memory is underutilized
            new_size = min(10000, int(current_size * 1.5))
            logger.info(f"Memory usage {current_usage:.1%} is low, increasing batch size to {new_size}")
            return new_size
        
        return current_size
    
    def should_reduce_batch_size(self) -> bool:
        """Check if batch size should be reduced due to memory pressure."""
        return self.get_memory_usage_percent() > self.max_memory_usage


class StockBatchIterator:
    """Iterator for processing stocks in batches - primary batching approach."""
    
    def __init__(self, symbols: List[str], batch_size: int, 
                 memory_manager: Optional[MemoryManager] = None):
        """
        Initialize stock batch iterator.
        
        Args:
            symbols: List of stock symbols
            batch_size: Number of stocks per batch (e.g., 100 stocks)
            memory_manager: Optional memory manager for dynamic sizing
        """
        self.symbols = symbols
        self.batch_size = batch_size
        self.memory_manager = memory_manager or MemoryManager()
        self.total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        # Track processed stocks for progress
        self.processed_stocks = 0
        self.current_batch_num = 0
        
    def get_stock_batches(self) -> Generator[Tuple[List[str], Dict[str, Any]], None, None]:
        """
        Yield batches of stock symbols with metadata.
        
        Yields:
            Tuple of (stock_symbols, batch_info) for current batch
        """
        for i in range(0, len(self.symbols), self.batch_size):
            batch = self.symbols[i:i + self.batch_size]
            self.current_batch_num = i // self.batch_size
            
            batch_info = {
                'batch_num': self.current_batch_num,
                'total_batches': self.total_batches,
                'batch_size': len(batch),
                'start_idx': i,
                'end_idx': min(i + self.batch_size, len(self.symbols)),
                'progress_percent': (self.current_batch_num / self.total_batches) * 100,
                'stocks_processed': self.processed_stocks,
                'total_stocks': len(self.symbols)
            }
            
            yield batch, batch_info
    
    def get_batch_info(self, batch_num: int) -> Dict[str, Any]:
        """
        Get information about a specific batch.
        
        Args:
            batch_num: Batch number (0-indexed)
            
        Returns:
            Dictionary with batch information
        """
        start_idx = batch_num * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.symbols))
        
        return {
            'batch_num': batch_num,
            'total_batches': self.total_batches,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'batch_size': len(self.symbols[start_idx:end_idx]),
            'progress_percent': (batch_num / self.total_batches) * 100,
            'stocks_processed': batch_num * self.batch_size,
            'total_stocks': len(self.symbols)
        }
    
    def update_processed_stocks(self, stocks_count: int):
        """Update count of processed stocks."""
        self.processed_stocks += stocks_count


class RowBatchIterator:
    """Iterator for processing data rows in batches."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int,
                 overlap_percent: float = 0.0, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize row batch iterator.
        
        Args:
            X: Feature matrix
            y: Target vector
            batch_size: Number of rows per batch
            overlap_percent: Overlap between batches (0-100)
            memory_manager: Optional memory manager for dynamic sizing
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.overlap_percent = overlap_percent
        self.memory_manager = memory_manager or MemoryManager()
        
        # Calculate step size based on overlap
        self.step_size = int(batch_size * (1 - overlap_percent / 100))
        self.total_batches = max(1, (len(X) - batch_size) // self.step_size + 1)
        
        # Adjust batch size based on memory constraints
        if len(X) > 0:
            features = X.shape[1]
            self.batch_size = self.memory_manager.adjust_batch_size(
                batch_size, features
            )
    
    def get_row_batches(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield batches of (X, y) data.
        
        Yields:
            Tuple of (X_batch, y_batch) for current batch
        """
        for i in range(0, len(self.X), self.step_size):
            end_idx = min(i + self.batch_size, len(self.X))
            
            if end_idx - i < self.batch_size // 2:  # Skip small batches
                break
                
            X_batch = self.X[i:end_idx]
            y_batch = self.y[i:end_idx]
            
            yield X_batch, y_batch
    
    def get_batch_info(self, batch_num: int) -> Dict[str, Any]:
        """
        Get information about a specific batch.
        
        Args:
            batch_num: Batch number (0-indexed)
            
        Returns:
            Dictionary with batch information
        """
        start_idx = batch_num * self.step_size
        end_idx = min(start_idx + self.batch_size, len(self.X))
        
        return {
            'batch_num': batch_num,
            'total_batches': self.total_batches,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'batch_size': end_idx - start_idx,
            'progress_percent': (batch_num / self.total_batches) * 100,
            'overlap_samples': int(self.batch_size * self.overlap_percent / 100)
        }


class BatchProgressTracker:
    """Track progress for batch training operations."""
    
    def __init__(self, total_stocks: int, total_batches: int, model_name: str):
        """
        Initialize batch progress tracker.
        
        Args:
            total_stocks: Total number of stocks to process
            total_batches: Total number of batches
            model_name: Name of model being trained
        """
        self.total_stocks = total_stocks
        self.total_batches = total_batches
        self.model_name = model_name
        self.current_batch = 0
        self.current_stock = 0
        self.start_time = time.time()
        self.batch_times = []
        self.last_update_time = time.time()
        self.update_interval = 20  # Update every 20 seconds
        self.current_stage = 'loading'  # Track current stage
        self.us_stock_count = 0  # Track US stocks in current batch
        self.ind_stock_count = 0  # Track Indian stocks in current batch
        self.sample_stocks = []  # Track sample stock symbols
    
    def set_stage(self, stage: str):
        """Set current processing stage."""
        self.current_stage = stage
    
    def set_batch_stocks(self, stock_symbols: List[str], sample_count: int = 8):
        """
        Set current batch stock information.
        
        Args:
            stock_symbols: List of stock symbols in format 'SYMBOL_category'
            sample_count: Number of sample stocks to keep for display
        """
        self.us_stock_count = 0
        self.ind_stock_count = 0
        clean_symbols = []
        
        for symbol in stock_symbols:
            # Parse symbol and category
            if '_us_stocks' in symbol:
                self.us_stock_count += 1
                clean_symbols.append(symbol.replace('_us_stocks', ''))
            elif '_ind_stocks' in symbol:
                self.ind_stock_count += 1
                clean_symbols.append(symbol.replace('_ind_stocks', ''))
            else:
                clean_symbols.append(symbol)
        
        # Keep first N symbols as samples
        self.sample_stocks = clean_symbols[:sample_count]
    
    def update_batch_progress(self, batch_num: int, batch_size: int, 
                             stocks_in_batch: int) -> Dict[str, Any]:
        """
        Update progress for current batch.
        
        Args:
            batch_num: Current batch number
            batch_size: Size of current batch
            stocks_in_batch: Number of stocks in current batch
            
        Returns:
            Dictionary with progress information
        """
        self.current_batch = batch_num
        self.current_stock += stocks_in_batch
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate progress metrics (batch_num is 0-indexed, so add 1 for percentage)
        batch_progress = ((batch_num + 1) / self.total_batches) * 100
        stock_progress = (self.current_stock / self.total_stocks) * 100
        
        # Calculate batch timing
        if batch_num > 0:
            avg_time_per_batch = elapsed / batch_num
            remaining_batches = self.total_batches - batch_num
            estimated_remaining = avg_time_per_batch * remaining_batches
        else:
            estimated_remaining = 0
        
        # Store batch timing
        if batch_num > 0:
            batch_time = current_time - self.last_update_time
            self.batch_times.append(batch_time)
        
        # Check if we should display a status update
        if current_time - self.last_update_time >= self.update_interval:
            self._display_batch_status_update(
                batch_num, batch_progress, stock_progress, 
                elapsed, estimated_remaining
            )
            self.last_update_time = current_time
        
        return {
            'batch_num': batch_num,
            'total_batches': self.total_batches,
            'batch_progress_percent': batch_progress,
            'stock_progress_percent': stock_progress,
            'stocks_processed': self.current_stock,
            'total_stocks': self.total_stocks,
            'elapsed_time': elapsed,
            'estimated_remaining': estimated_remaining,
            'current_stage': self.current_stage,
            'us_stock_count': self.us_stock_count,
            'ind_stock_count': self.ind_stock_count,
            'sample_stocks': self.sample_stocks
        }
    
    def _display_batch_status_update(self, batch_num: int, batch_progress: float,
                                   stock_progress: float, elapsed: float, 
                                   estimated_remaining: float):
        """Display a comprehensive batch status update."""
        elapsed_minutes = elapsed / 60
        remaining_minutes = estimated_remaining / 60
        
        print(f"\n{'='*80}")
        print(f"TRAINING - {self.model_name.upper()} - Batch {batch_num + 1}/{self.total_batches}")
        print(f"{'='*80}")
        print(f"")
        print(f"Batch Progress:  {batch_num + 1}/{self.total_batches} ({batch_progress:.1f}%)")
        
        # Add visual progress bar for batch
        bar_length = 50
        batch_filled = int(bar_length * batch_progress / 100)
        batch_bar = '#' * batch_filled + '-' * (bar_length - batch_filled)
        print(f"[{batch_bar}] {batch_progress:.1f}%")
        print(f"")
        print(f"Stocks Trained:  {self.current_stock:,}/{self.total_stocks:,} ({stock_progress:.1f}%)")
        
        # Add visual progress bar for stocks
        stock_filled = int(bar_length * stock_progress / 100)
        stock_bar = '#' * stock_filled + '-' * (bar_length - stock_filled)
        print(f"[{stock_bar}] {stock_progress:.1f}%")
        print(f"")
        print(f"Elapsed: {elapsed_minutes:.1f} min  |  Remaining: {remaining_minutes:.1f} min")
        
        if batch_num > 0:
            avg_batch_time = elapsed / batch_num
            print(f"Average: {avg_batch_time:.1f} sec/batch  |  Rate: {self.current_stock/elapsed_minutes:.1f} stocks/min")
        
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
        batch_progress = (self.current_batch / self.total_batches) * 100
        stock_progress = (self.current_stock / self.total_stocks) * 100
        
        if self.current_batch > 0:
            avg_time_per_batch = elapsed / self.current_batch
            remaining_batches = self.total_batches - self.current_batch
            estimated_remaining = avg_time_per_batch * remaining_batches
        else:
            estimated_remaining = 0
        
        self._display_batch_status_update(
            self.current_batch, batch_progress, stock_progress,
            elapsed, estimated_remaining
        )


def calculate_batch_metrics(batch_num: int, total_batches: int, 
                          batch_size: int, total_samples: int) -> Dict[str, Any]:
    """
    Calculate metrics for a specific batch.
    
    Args:
        batch_num: Current batch number (0-indexed)
        total_batches: Total number of batches
        batch_size: Size of current batch
        total_samples: Total number of samples
        
    Returns:
        Dictionary with batch metrics
    """
    # batch_num is 0-indexed, so add 1 for percentage calculation
    progress_percent = ((batch_num + 1) / total_batches) * 100
    samples_processed = (batch_num + 1) * batch_size
    remaining_samples = total_samples - samples_processed
    
    return {
        'batch_num': batch_num,
        'total_batches': total_batches,
        'progress_percent': progress_percent,
        'samples_processed': samples_processed,
        'remaining_samples': remaining_samples,
        'batch_size': batch_size
    }


# Example usage and testing
if __name__ == "__main__":
    # Test stock batch iterator
    symbols = [f"STOCK_{i:03d}" for i in range(250)]
    stock_iterator = StockBatchIterator(symbols, batch_size=50)
    
    print("Testing Stock Batch Iterator:")
    for i, batch in enumerate(stock_iterator.get_stock_batches()):
        print(f"Batch {i+1}: {len(batch)} stocks - {batch[:3]}...")
        if i >= 2:  # Test first 3 batches
            break
    
    # Test row batch iterator
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    row_iterator = RowBatchIterator(X, y, batch_size=100, overlap_percent=10)
    
    print("\nTesting Row Batch Iterator:")
    for i, (X_batch, y_batch) in enumerate(row_iterator.get_row_batches()):
        print(f"Batch {i+1}: {X_batch.shape[0]} samples")
        if i >= 2:  # Test first 3 batches
            break
    
    print("\nBatch iterator tests completed successfully!")
