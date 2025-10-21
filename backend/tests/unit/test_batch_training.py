#!/usr/bin/env python3
"""
Unit Tests for Batch Training Functionality

This module tests the batch training implementation including
stock batching, memory management, and strategy selection.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from training.batch_iterator import StockBatchIterator, RowBatchIterator, MemoryManager, BatchProgressTracker
from training.batch_strategies import get_batch_strategy, IncrementalStrategy, SubsampleStrategy, AccumulateStrategy
from prediction.config import config


class TestMemoryManager(unittest.TestCase):
    """Test memory management functionality."""
    
    def setUp(self):
        self.memory_manager = MemoryManager()
    
    def test_get_available_memory(self):
        """Test getting available memory."""
        memory = self.memory_manager.get_available_memory()
        self.assertIsInstance(memory, int)
        self.assertGreater(memory, 0)
    
    def test_estimate_batch_memory(self):
        """Test memory estimation for batches."""
        batch_size = 1000
        features = 50
        memory_mb = self.memory_manager.estimate_batch_memory(batch_size, features)
        
        self.assertIsInstance(memory_mb, int)
        self.assertGreater(memory_mb, 0)
    
    def test_adjust_batch_size(self):
        """Test dynamic batch size adjustment."""
        current_size = 10000
        features = 50
        target_memory = 100  # 100MB
        
        adjusted_size = self.memory_manager.adjust_batch_size(current_size, features, target_memory)
        
        self.assertIsInstance(adjusted_size, int)
        self.assertLessEqual(adjusted_size, current_size)


class TestStockBatchIterator(unittest.TestCase):
    """Test stock batch iteration."""
    
    def setUp(self):
        self.symbols = [f"STOCK_{i:03d}" for i in range(250)]
        self.batch_size = 50
        self.iterator = StockBatchIterator(self.symbols, self.batch_size)
    
    def test_initialization(self):
        """Test iterator initialization."""
        self.assertEqual(len(self.iterator.symbols), 250)
        self.assertEqual(self.iterator.batch_size, 50)
        self.assertEqual(self.iterator.total_batches, 5)  # 250 / 50 = 5
    
    def test_get_stock_batches(self):
        """Test getting stock batches."""
        batches = list(self.iterator.get_stock_batches())
        
        self.assertEqual(len(batches), 5)  # 5 batches
        
        # Check first batch
        first_batch, first_info = batches[0]
        self.assertEqual(len(first_batch), 50)
        self.assertEqual(first_info['batch_num'], 0)
        self.assertEqual(first_info['total_batches'], 5)
        
        # Check last batch (might be smaller)
        last_batch, last_info = batches[-1]
        self.assertEqual(last_info['batch_num'], 4)
        self.assertLessEqual(len(last_batch), 50)
    
    def test_get_batch_info(self):
        """Test getting batch information."""
        info = self.iterator.get_batch_info(2)
        
        self.assertEqual(info['batch_num'], 2)
        self.assertEqual(info['total_batches'], 5)
        self.assertEqual(info['start_idx'], 100)  # 2 * 50
        self.assertEqual(info['end_idx'], 150)    # 2 * 50 + 50


class TestRowBatchIterator(unittest.TestCase):
    """Test row batch iteration."""
    
    def setUp(self):
        # Create test data
        self.X = np.random.randn(1000, 10)
        self.y = np.random.randn(1000)
        self.batch_size = 200
        self.iterator = RowBatchIterator(self.X, self.y, self.batch_size, overlap_percent=10)
    
    def test_initialization(self):
        """Test iterator initialization."""
        self.assertEqual(self.iterator.batch_size, 200)
        self.assertEqual(self.iterator.overlap_percent, 10)
        self.assertGreater(self.iterator.total_batches, 0)
    
    def test_get_row_batches(self):
        """Test getting row batches."""
        batches = list(self.iterator.get_row_batches())
        
        self.assertGreater(len(batches), 0)
        
        # Check first batch
        X_batch, y_batch = batches[0]
        self.assertEqual(len(X_batch), 200)
        self.assertEqual(len(y_batch), 200)
        self.assertEqual(X_batch.shape[1], 10)


class TestBatchProgressTracker(unittest.TestCase):
    """Test batch progress tracking."""
    
    def setUp(self):
        self.tracker = BatchProgressTracker(1000, 10, "test_model")
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.total_stocks, 1000)
        self.assertEqual(self.tracker.total_batches, 10)
        self.assertEqual(self.tracker.model_name, "test_model")
    
    def test_update_batch_progress(self):
        """Test updating batch progress."""
        progress_info = self.tracker.update_batch_progress(2, 100, 100)
        
        self.assertEqual(progress_info['batch_num'], 2)
        self.assertEqual(progress_info['total_batches'], 10)
        self.assertEqual(progress_info['batch_progress_percent'], 20.0)  # 2/10 * 100
        self.assertEqual(progress_info['stocks_processed'], 300)  # 3 * 100


class TestBatchStrategies(unittest.TestCase):
    """Test batch strategy selection and functionality."""
    
    def test_strategy_selection(self):
        """Test strategy selection for different models."""
        # Test incremental strategy
        strategy = get_batch_strategy('linear_regression')
        self.assertIsInstance(strategy, IncrementalStrategy)
        
        # Test subsample strategy
        strategy = get_batch_strategy('svm')
        self.assertIsInstance(strategy, SubsampleStrategy)
        
        # Test accumulate strategy
        strategy = get_batch_strategy('random_forest')
        self.assertIsInstance(strategy, AccumulateStrategy)
    
    def test_incremental_strategy(self):
        """Test incremental strategy."""
        strategy = IncrementalStrategy('test_model')
        
        self.assertTrue(strategy.supports_incremental_learning())
        self.assertEqual(strategy.get_strategy_name(), 'Incremental')
    
    def test_subsample_strategy(self):
        """Test subsample strategy."""
        strategy = SubsampleStrategy('test_model', subsample_percent=30)
        
        self.assertFalse(strategy.supports_incremental_learning())
        self.assertEqual(strategy.get_strategy_name(), 'Subsample')
        self.assertEqual(strategy.subsample_percent, 30)
    
    def test_accumulate_strategy(self):
        """Test accumulate strategy."""
        strategy = AccumulateStrategy('test_model')
        
        self.assertFalse(strategy.supports_incremental_learning())
        self.assertEqual(strategy.get_strategy_name(), 'Accumulate')


class TestBatchTrainingIntegration(unittest.TestCase):
    """Test integration of batch training components."""
    
    def setUp(self):
        # Mock the data loader
        self.mock_data_loader = Mock()
        self.mock_data_loader.get_stock_symbols.return_value = [f"STOCK_{i}" for i in range(100)]
    
    @patch('training.batch_strategies.DataLoader')
    def test_strategy_with_mock_data(self, mock_data_loader_class):
        """Test strategy with mocked data."""
        # Setup mock
        mock_data_loader_class.return_value = self.mock_data_loader
        
        # Create test data
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        # Test subsample strategy
        strategy = SubsampleStrategy('test_model', subsample_percent=50)
        
        # Mock the data loading methods
        strategy._load_stock_batch_data = Mock(return_value=(X, y, ['STOCK_1', 'STOCK_2']))
        
        # Mock model
        mock_model = Mock()
        mock_model.fit = Mock()
        
        # Test training on batch
        batch_info = {'batch_num': 0, 'total_batches': 1}
        result = strategy.train_on_stock_batch(mock_model, ['STOCK_1', 'STOCK_2'], batch_info)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['stocks_processed'], 2)
        self.assertEqual(result['samples_processed'], 100)


class TestConfigIntegration(unittest.TestCase):
    """Test configuration integration."""
    
    def test_batch_config_parameters(self):
        """Test that batch configuration parameters exist."""
        self.assertTrue(hasattr(config, 'USE_BATCH_TRAINING'))
        self.assertTrue(hasattr(config, 'STOCK_BATCH_SIZE'))
        self.assertTrue(hasattr(config, 'ROW_BATCH_SIZE'))
        self.assertTrue(hasattr(config, 'SUBSAMPLE_PERCENT'))
        self.assertTrue(hasattr(config, 'ENABLE_INCREMENTAL_TRAINING'))
    
    def test_config_default_values(self):
        """Test default configuration values."""
        self.assertTrue(config.USE_BATCH_TRAINING)
        self.assertEqual(config.STOCK_BATCH_SIZE, 100)
        self.assertEqual(config.ROW_BATCH_SIZE, 50000)
        self.assertEqual(config.SUBSAMPLE_PERCENT, 50)
        self.assertTrue(config.ENABLE_INCREMENTAL_TRAINING)


def run_batch_training_tests():
    """Run all batch training tests."""
    print("Running Batch Training Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMemoryManager,
        TestStockBatchIterator,
        TestRowBatchIterator,
        TestBatchProgressTracker,
        TestBatchStrategies,
        TestBatchTrainingIntegration,
        TestConfigIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nBatch Training Tests: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_batch_training_tests()
    sys.exit(0 if success else 1)
