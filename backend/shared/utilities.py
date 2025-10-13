"""
Shared Module

This module contains shared utilities and common functionality used across
different modules in the backend. It includes:
- Configuration management
- Logging utilities
- Data validation helpers
- Common data structures
- Utility functions
- Constants and enums

This module ensures consistency and reduces code duplication across the project.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Configuration Management
class Config:
    """
    Centralized configuration management
    """
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')
        self.permanent_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'permanent')
        self.cache_duration = int(os.getenv('CACHE_DURATION', 60))
        self.min_request_delay = float(os.getenv('MIN_REQUEST_DELAY', 2.0))
        self.port = int(os.getenv('PORT', 5000))
        
        # API Keys
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.alphavantage_api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    
    def get_data_path(self, *paths):
        """Get path relative to data directory"""
        return os.path.join(self.data_dir, *paths)
    
    def get_permanent_path(self, *paths):
        """Get path relative to permanent directory"""
        return os.path.join(self.permanent_dir, *paths)

# Logging Utilities
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with consistent formatting
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Data Validation
class DataValidator:
    """
    Common data validation utilities
    """
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        return len(symbol.strip()) > 0 and len(symbol.strip()) <= 10
    
    @staticmethod
    def validate_price(price: Any) -> bool:
        """Validate price value"""
        try:
            price_float = float(price)
            return price_float > 0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """Validate date format (YYYY-MM-DD)"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

# Common Data Structures
class StockData:
    """
    Standardized stock data structure
    """
    def __init__(self, symbol: str, price: float, timestamp: str, 
                 source: str, company_name: str = None):
        self.symbol = symbol.upper()
        self.price = float(price)
        self.timestamp = timestamp
        self.source = source
        self.company_name = company_name or symbol
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'timestamp': self.timestamp,
            'source': self.source,
            'company_name': self.company_name
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

class PredictionResult:
    """
    Standardized prediction result structure
    """
    def __init__(self, symbol: str, predicted_price: float, 
                 confidence: float, algorithm: str, timestamp: str):
        self.symbol = symbol.upper()
        self.predicted_price = float(predicted_price)
        self.confidence = float(confidence)
        self.algorithm = algorithm
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'predicted_price': self.predicted_price,
            'confidence': self.confidence,
            'algorithm': self.algorithm,
            'timestamp': self.timestamp
        }

# Utility Functions
def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def categorize_stock(symbol: str) -> str:
    """
    Categorize stock based on symbol suffix
    """
    symbol_upper = symbol.upper()
    
    # Indian stock suffixes
    indian_suffixes = ['.NS', '.BO']
    for suffix in indian_suffixes:
        if symbol_upper.endswith(suffix):
            return 'ind_stocks'
    
    # US stock suffixes
    us_suffixes = ['.US']
    for suffix in us_suffixes:
        if symbol_upper.endswith(suffix):
            return 'us_stocks'
    
    # Common US symbols
    common_us_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ']
    if symbol_upper.split('.')[0] in common_us_symbols:
        return 'us_stocks'
    
    return 'others_stocks'

def format_price(price: float, currency: str = 'USD') -> str:
    """Format price with currency symbol"""
    if currency == 'USD':
        return f"${price:.2f}"
    elif currency == 'INR':
        return f"₹{price:.2f}"
    else:
        return f"{price:.2f} {currency}"

# Constants
class Constants:
    """Application constants"""
    
    # Stock Categories
    US_STOCKS = 'us_stocks'
    INDIAN_STOCKS = 'ind_stocks'
    OTHER_STOCKS = 'others_stocks'
    
    # API Sources
    YFINANCE = 'yfinance'
    FINNHUB = 'finnhub'
    ALPHAVANTAGE = 'alphavantage'
    PERMANENT_DIR = 'permanent_directory'
    
    # File Extensions
    CSV_EXTENSION = '.csv'
    JSON_EXTENSION = '.json'
    
    # Default Values
    DEFAULT_CACHE_DURATION = 60
    DEFAULT_REQUEST_DELAY = 2.0
    DEFAULT_PORT = 5000
    MAX_SYMBOL_LENGTH = 10
    MAX_SEARCH_RESULTS = 20

# Error Classes
class StockDataError(Exception):
    """Base exception for stock data related errors"""
    pass

class InvalidSymbolError(StockDataError):
    """Exception for invalid stock symbols"""
    pass

class DataFetchError(StockDataError):
    """Exception for data fetching errors"""
    pass

class PredictionError(StockDataError):
    """Exception for prediction related errors"""
    pass

__all__ = [
    'Config',
    'setup_logger',
    'DataValidator',
    'StockData',
    'PredictionResult',
    'get_current_timestamp',
    'categorize_stock',
    'format_price',
    'Constants',
    'StockDataError',
    'InvalidSymbolError',
    'DataFetchError',
    'PredictionError'
]
