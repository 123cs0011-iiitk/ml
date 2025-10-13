"""
Live Data Module

This module handles all live market data fetching and updates.
Contains functionality for:
- Real-time stock price fetching
- Market data updates
- Data caching and storage
- Multiple API fallbacks (yfinance, Finnhub, Alpha Vantage)
"""

from live_fetcher import LiveFetcher

__all__ = ['LiveFetcher']
