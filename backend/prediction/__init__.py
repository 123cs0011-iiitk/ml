"""
Stock Prediction Module

This module provides comprehensive stock price prediction capabilities using
multiple machine learning algorithms from the algorithms/real directory.

Components:
- config.py: Configuration for prediction parameters
- data_loader.py: Data loading and preprocessing
- predictor.py: Main prediction orchestrator
- prediction_saver.py: Save predictions to CSV files
- run_predictions.py: Standalone execution script

Usage:
    python backend/prediction/run_predictions.py
"""

__version__ = "1.0.0"
__author__ = "Stock Prediction System"

# Import main components for easy access
from .config import PredictionConfig
from .data_loader import DataLoader
from .predictor import StockPredictor
from .prediction_saver import PredictionSaver

__all__ = [
    'PredictionConfig',
    'DataLoader', 
    'StockPredictor',
    'PredictionSaver'
]
