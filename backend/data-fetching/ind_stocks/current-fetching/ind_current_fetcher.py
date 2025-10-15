"""
Indian Stocks Current Price Fetcher

Fetches current live stock prices for Indian stocks using multiple data sources.
Implements fallback chain: Upstox API → NSEPython → yfinance → NSELib

Rate-limited to respect API limits and ensure reliable data fetching.
All prices are returned in INR currency.
"""

import os
import sys
import requests
import pandas as pd
import time
import yfinance as yf
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from shared.utilities import (
    Config, Constants, standardize_csv_columns, 
    ensure_alphabetical_order, get_currency_for_category,
    get_live_exchange_rate, convert_usd_to_inr, get_current_timestamp
)

class IndianCurrentFetcher:
    """
    Fetches current live Indian stock prices using multiple data sources.
    Implements fallback chain for maximum reliability.
    All prices are returned in INR currency.
    """
    
    def __init__(self):
        self.config = Config()
        self.data_dir = self.config.data_dir
        self.latest_dir = os.path.join(self.data_dir, 'latest', 'ind_stocks')
        self.latest_prices_file = os.path.join(self.latest_dir, 'latest_prices.csv')
        
        # Ensure directories exist
        os.makedirs(self.latest_dir, exist_ok=True)
        
        # API configurations
        self.upstox_api_key = self.config.upstox_api_key
        self.upstox_access_token = self.config.upstox_access_token
        self.currency = get_currency_for_category('ind_stocks')  # INR
        
        # Rate limiting
        self.rate_limit_delay = 1.0  # 1 second between calls
        self.max_retries = 3
        
        # Cache configuration
        self.cache_duration = 60  # 60 seconds cache
        self.cache = {}  # {symbol: {data: dict, timestamp: datetime}}
        self.last_request_time = 0
        
        # Initialize NSE libraries
        self._init_nse_libraries()
    
    def _init_nse_libraries(self):
        """Initialize NSE libraries with error handling"""
        self.nsepython_available = False
        self.nselib_available = False
        
        try:
            import nsepython
            self.nsepython_available = True
            print("NSEPython library loaded successfully")
        except ImportError:
            print("NSEPython not available - install with: pip install nsepython")
        
        try:
            import nselib
            self.nselib_available = True
            print("NSELib library loaded successfully")
        except ImportError:
            print("NSELib not available - install with: pip install nselib")
    
    def prepare_yfinance_symbol(self, symbol: str) -> str:
        """
        Prepare symbol for yfinance by adding .NS suffix if needed.
        
        Args:
            symbol: Original symbol
        
        Returns:
            Symbol with .NS suffix for yfinance
        """
        # Remove any existing suffix first
        base_symbol = symbol.split('.')[0]
        
        # Add .NS suffix for NSE stocks
        yfinance_symbol = f"{base_symbol}.NS"
        return yfinance_symbol
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            print(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data for symbol is still valid"""
        if symbol not in self.cache:
            return False
        
        cache_entry = self.cache[symbol]
        cache_time = cache_entry['timestamp']
        
        # Check if cache is still within duration
        return datetime.now() - cache_time < timedelta(seconds=self.cache_duration)
    
    def fetch_price_from_upstox(self, symbol: str) -> Tuple[float, str]:
        """
        Fetch current stock price from Upstox API.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Tuple of (price, company_name)
        """
        if not self.upstox_api_key:
            raise ValueError("Upstox API key not configured")
        
        try:
            # Upstox API endpoint for market data
            url = "https://api.upstox.com/v2/market-quote/ltp"
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.upstox_access_token}',
                'Api-Version': '2.0'
            }
            
            # Prepare symbol for Upstox (NSE format)
            upstox_symbol = f"NSE_EQ|{symbol}"
            params = {'symbol': upstox_symbol}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and upstox_symbol in data['data']:
                price_data = data['data'][upstox_symbol]
                price = float(price_data['last_price'])
                company_name = price_data.get('company_name', symbol)
                
                print(f"Successfully fetched {symbol} from Upstox: ₹{price} ({company_name})")
                return price, company_name
            else:
                raise ValueError(f"No price data available for {symbol}")
                
        except Exception as e:
            print(f"Upstox error for {symbol}: {str(e)}")
            raise
    
    def fetch_price_from_nsepython(self, symbol: str) -> Tuple[float, str]:
        """
        Fetch current stock price from NSEPython library.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Tuple of (price, company_name)
        """
        if not self.nsepython_available:
            raise ValueError("NSEPython library not available")
        
        try:
            import nsepython
            
            # Get quote data
            quote_data = nsepython.nse_quote(symbol)
            
            if 'lastPrice' in quote_data and quote_data['lastPrice']:
                price = float(quote_data['lastPrice'])
                company_name = quote_data.get('companyName', symbol)
                
                print(f"Successfully fetched {symbol} from NSEPython: ₹{price} ({company_name})")
                return price, company_name
            else:
                raise ValueError(f"No price data available for {symbol}")
                
        except Exception as e:
            print(f"NSEPython error for {symbol}: {str(e)}")
            raise
    
    def fetch_price_from_yfinance(self, symbol: str) -> Tuple[float, str]:
        """
        Fetch current stock price from yfinance with multiple fallback methods.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Tuple of (price, company_name)
        """
        try:
            # Prepare symbol for yfinance
            yfinance_symbol = self.prepare_yfinance_symbol(symbol)
            print(f"Using yfinance symbol: {yfinance_symbol}")
            
            ticker = yf.Ticker(yfinance_symbol)
            
            # Method 1: Try with 30-day period for more data
            try:
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                print(f"Trying 30-day period for {symbol}...")
                hist = ticker.history(start=start_date, end=end_date, auto_adjust=False)
                
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    company_name = self._get_company_name(ticker, symbol)
                    print(f"Successfully fetched {symbol} from yfinance (30d): ₹{price} ({company_name})")
                    return price, company_name
                else:
                    print(f"No data in 30-day period for {symbol}")
                    raise ValueError("No data available")
                    
            except Exception as e:
                print(f"30-day method failed for {symbol}: {e}")
                
                # Method 2: Try with 7-day period
                try:
                    print(f"Trying 7-day period for {symbol}...")
                    hist = ticker.history(period="7d", auto_adjust=False)
                    
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])
                        company_name = self._get_company_name(ticker, symbol)
                        print(f"Successfully fetched {symbol} from yfinance (7d): ₹{price} ({company_name})")
                        return price, company_name
                    else:
                        raise ValueError("No data available")
                        
                except Exception as e2:
                    print(f"7-day method failed for {symbol}: {e2}")
                    
                    # Method 3: Try with 5-day period
                    try:
                        print(f"Trying 5-day period for {symbol}...")
                        hist = ticker.history(period="5d", auto_adjust=False)
                        
                        if not hist.empty:
                            price = float(hist['Close'].iloc[-1])
                            company_name = self._get_company_name(ticker, symbol)
                            print(f"Successfully fetched {symbol} from yfinance (5d): ₹{price} ({company_name})")
                            return price, company_name
                        else:
                            raise ValueError("No data available")
                            
                    except Exception as e3:
                        print(f"5-day method failed for {symbol}: {e3}")
                        
                        # Method 4: Try with 1-day period
                        try:
                            print(f"Trying 1-day period for {symbol}...")
                            hist = ticker.history(period="1d", auto_adjust=False)
                            
                            if not hist.empty:
                                price = float(hist['Close'].iloc[-1])
                                company_name = self._get_company_name(ticker, symbol)
                                print(f"Successfully fetched {symbol} from yfinance (1d): ₹{price} ({company_name})")
                                return price, company_name
                            else:
                                raise ValueError("No data available")
                                
                        except Exception as e4:
                            print(f"All yfinance methods failed for {symbol}: {e4}")
                            raise
                
        except Exception as e:
            print(f"yfinance error for {symbol}: {str(e)}")
            raise  # Ensure exception is raised for proper fallback chain
    
    def _get_company_name(self, ticker, symbol: str) -> str:
        """Helper method to get company name from ticker info."""
        try:
            info = ticker.info
            return info.get('longName', info.get('shortName', symbol))
        except:
            return symbol
    
    def _get_stock_metadata(self, symbol: str) -> Dict[str, str]:
        """
        Get stock metadata (sector, market_cap, headquarters, exchange) from index CSV files.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with metadata fields, or 'N/A' for missing fields
        """
        try:
            # Check permanent index file for Indian stocks
            index_path = os.path.join(
                '..', 'permanent', 'ind_stocks', 'index_ind_stocks.csv'
            )
            index_path = os.path.normpath(index_path)
            
            if not os.path.exists(index_path):
                print(f"Index file not found: {index_path}")
                return {
                    'sector': 'N/A',
                    'market_cap': 'N/A',
                    'headquarters': 'N/A',
                    'exchange': 'N/A'
                }
            
            # Read metadata from index file
            df = pd.read_csv(index_path)
            symbol_row = df[df['symbol'] == symbol]
            if not symbol_row.empty:
                row = symbol_row.iloc[0]
                return {
                    'sector': self._clean_metadata_value(str(row.get('sector', 'N/A'))),
                    'market_cap': self._clean_metadata_value(str(row.get('market_cap', 'N/A'))),
                    'headquarters': self._clean_metadata_value(str(row.get('headquarters', 'N/A'))),
                    'exchange': self._clean_metadata_value(str(row.get('exchange', 'N/A')))
                }
            
            # Symbol not found in index
            print(f"Symbol {symbol} not found in index file: {index_path}")
            return {
                'sector': 'N/A',
                'market_cap': 'N/A',
                'headquarters': 'N/A',
                'exchange': 'N/A'
            }
            
        except Exception as e:
            print(f"Error fetching metadata for {symbol}: {str(e)}")
            return {
                'sector': 'N/A',
                'market_cap': 'N/A',
                'headquarters': 'N/A',
                'exchange': 'N/A'
            }
    
    def _clean_metadata_value(self, value: str) -> str:
        """
        Clean metadata value by handling empty strings, NaN, and whitespace.
        
        Args:
            value: Raw value from CSV
            
        Returns:
            Cleaned value or 'N/A' if empty/invalid
        """
        if not value or value.strip() == '' or value.lower() in ['nan', 'none', 'null']:
            return 'N/A'
        return value.strip()
    
    def fetch_from_permanent_directory(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch stock data from permanent directory as last resort.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with stock data or None if not found
        """
        try:
            # Look for individual file in permanent directory (relative from current working directory)
            permanent_file = os.path.join(
                '..', 'permanent', 'ind_stocks', 'individual_files', f'{symbol}.csv'
            )
            permanent_file = os.path.normpath(permanent_file)
            
            if os.path.exists(permanent_file):
                df = pd.read_csv(permanent_file)
                
                if not df.empty:
                    # Get the latest row (most recent data)
                    latest_row = df.iloc[-1]
                    
                    # Try different column name formats (Close or close)
                    close_price = latest_row.get('Close', latest_row.get('close', latest_row.get('adjusted_close')))
                    
                    # Get additional metadata from index files
                    metadata = self._get_stock_metadata(symbol)
                    
                    return {
                        'symbol': symbol,
                        'price': float(close_price),
                        'company_name': symbol,
                        'currency': self.currency,
                        'source': 'permanent',
                        'timestamp': get_current_timestamp(),
                        'sector': metadata['sector'],
                        'market_cap': metadata['market_cap'],
                        'headquarters': metadata['headquarters'],
                        'exchange': metadata['exchange'],
                        'open': float(latest_row['open']) if 'open' in latest_row and pd.notna(latest_row['open']) else None,
                        'high': float(latest_row['high']) if 'high' in latest_row and pd.notna(latest_row['high']) else None,
                        'low': float(latest_row['low']) if 'low' in latest_row and pd.notna(latest_row['low']) else None,
                        'volume': int(latest_row['volume']) if 'volume' in latest_row and pd.notna(latest_row['volume']) else None,
                        'close': float(latest_row['close']) if 'close' in latest_row and pd.notna(latest_row['close']) else None
                    }
        except Exception as e:
            print(f"Error reading permanent data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def fetch_price_from_stock_market_india(self, symbol: str) -> Tuple[float, str]:
        """
        Fetch current stock price from stock-market-india Python package.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Tuple of (price, company_name)
        """
        try:
            from stock_market_india import StockMarketIndia
            print(f"Fetching {symbol} from stock-market-india package...")
            
            # Initialize the stock market India client
            smi = StockMarketIndia()
            
            # Get quote data
            quote_data = smi.get_quote(symbol)
            
            if quote_data and 'lastPrice' in quote_data:
                price = float(quote_data['lastPrice'])
                company_name = quote_data.get('companyName', symbol)
                
                print(f"Successfully fetched {symbol} from stock-market-india: ₹{price} ({company_name})")
                return price, company_name
            else:
                raise ValueError(f"No price data available for {symbol}")
                
        except ImportError:
            print(f"stock-market-india package not available - install with: pip install stock-market-india")
            raise
        except Exception as e:
            print(f"stock-market-india error for {symbol}: {str(e)}")
            raise
    
    def fetch_price_from_nselib(self, symbol: str) -> Tuple[float, str]:
        """
        Fetch current stock price from NSELib library.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Tuple of (price, company_name)
        """
        if not self.nselib_available:
            raise ValueError("NSELib library not available")
        
        try:
            import nselib
            
            # Get quote data
            quote_data = nselib.get_quote(symbol)
            
            if 'lastPrice' in quote_data and quote_data['lastPrice']:
                price = float(quote_data['lastPrice'])
                company_name = quote_data.get('companyName', symbol)
                
                print(f"Successfully fetched {symbol} from NSELib: ₹{price} ({company_name})")
                return price, company_name
            else:
                raise ValueError(f"No price data available for {symbol}")
                
        except Exception as e:
            print(f"NSELib error for {symbol}: {str(e)}")
            raise
    
    def fetch_current_price(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current stock price with fallback strategy and caching.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dict with price data
        """
        if not symbol or not symbol.strip():
            raise ValueError("Stock symbol is required")
        
        symbol = symbol.strip().upper()
        
        # Check cache first
        if self._is_cache_valid(symbol):
            cached_data = self.cache[symbol]['data']
            print(f"Returning cached data for {symbol} (age: {(datetime.now() - self.cache[symbol]['timestamp']).seconds}s)")
            return cached_data
        
        timestamp = datetime.now().isoformat()
        
        # Try APIs in order of preference
        # Try each API in order - prioritize yfinance for reliability
        apis = [
            ('yfinance', self.fetch_price_from_yfinance),
            ('upstox', self.fetch_price_from_upstox),
            ('nsepython', self.fetch_price_from_nsepython),
            ('stock-market-india', self.fetch_price_from_stock_market_india),
            ('nselib', self.fetch_price_from_nselib)
        ]
        
        last_error = None
        
        for api_name, api_func in apis:
            try:
                print(f"Trying {api_name} for symbol {symbol}")
                self._enforce_rate_limit()
                
                price, company_name = api_func(symbol)
                
                # Get additional metadata from index files
                metadata = self._get_stock_metadata(symbol)
                
                # Get additional data from latest CSV files
                additional_data = self._get_latest_day_data(symbol)
                
                result = {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': timestamp,
                    'source': api_name,
                    'company_name': company_name,
                    'currency': self.currency,
                    'sector': metadata['sector'],
                    'market_cap': metadata['market_cap'],
                    'headquarters': metadata['headquarters'],
                    'exchange': metadata['exchange'],
                    'open': additional_data.get('open'),
                    'high': additional_data.get('high'),
                    'low': additional_data.get('low'),
                    'volume': additional_data.get('volume'),
                    'close': additional_data.get('close')
                }
                
                # Cache the result
                self.cache[symbol] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
                
                # Save to CSV
                self.save_to_csv(result)
                
                print(f"Successfully fetched {symbol} price ₹{price} from {api_name}")
                return result
                
            except Exception as e:
                last_error = e
                print(f"{api_name} failed for {symbol}: {str(e)}")
                continue
        
        # All APIs failed, try to get data from permanent directory as last resort
        print(f"All APIs failed for {symbol}, trying permanent directory fallback...")
        try:
            permanent_data = self.fetch_from_permanent_directory(symbol)
            if permanent_data:
                print(f"✅ Found {symbol} in permanent directory")
                return permanent_data
        except Exception as e:
            print(f"Permanent directory fallback failed: {e}")
        
        # If everything fails
        error_msg = f"Unable to fetch price for {symbol}. All sources failed. Last error: {str(last_error)}"
        print(error_msg)
        raise Exception(error_msg)
    
    def save_to_csv(self, data: Dict[str, Any]):
        """Save stock data to latest_prices.csv file"""
        try:
            # Create DataFrame with the new data
            new_row = pd.DataFrame([{
                'symbol': data['symbol'],
                'price': data['price'],
                'timestamp': data['timestamp'],
                'source': data['source'],
                'company_name': data['company_name'],
                'currency': data.get('currency', self.currency)
            }])
            
            # Read existing data if file exists
            if os.path.exists(self.latest_prices_file):
                existing_df = pd.read_csv(self.latest_prices_file)
                # Remove any existing entry for this symbol
                existing_df = existing_df[existing_df['symbol'] != data['symbol']]
                # Append new data
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            else:
                updated_df = new_row
            
            # Save updated data
            updated_df.to_csv(self.latest_prices_file, index=False)
            
            # Update dynamic index
            self.update_dynamic_index()
            
            print(f"Saved {data['symbol']} data to {self.latest_prices_file}")
            
        except Exception as e:
            print(f"Error saving to CSV for {data['symbol']}: {str(e)}")
            # Don't raise exception to avoid breaking the main flow
    
    def update_dynamic_index(self):
        """Update the dynamic index CSV file with company info"""
        try:
            index_path = os.path.join(self.data_dir, 'index_ind_stocks_dynamic.csv')
            
            if os.path.exists(self.latest_prices_file):
                df = pd.read_csv(self.latest_prices_file)
                symbols = df['symbol'].unique().tolist()
                
                # Try to get company info from permanent index
                permanent_index_path = os.path.join(self.data_dir, '..', '..', '..', 'permanent', 'ind_stocks', 'index_ind_stocks.csv')
                company_info = {}
                
                if os.path.exists(permanent_index_path):
                    try:
                        permanent_df = pd.read_csv(permanent_index_path)
                        company_info = permanent_df.set_index('symbol').to_dict('index')
                    except Exception as e:
                        print(f"Could not read permanent index: {e}")
                
                # Create index with company info
                index_data = []
                for symbol in symbols:
                    if symbol in company_info:
                        # Use info from permanent index
                        info = company_info[symbol]
                        index_data.append({
                            'symbol': symbol,
                            'company_name': info.get('company_name', symbol),
                            'sector': info.get('sector', 'Unknown'),
                            'market_cap': info.get('market_cap', ''),
                            'headquarters': info.get('headquarters', 'Unknown'),
                            'exchange': info.get('exchange', 'NSE'),
                            'currency': info.get('currency', self.currency)
                        })
                    else:
                        # Use basic info
                        index_data.append({
                            'symbol': symbol,
                            'company_name': symbol,
                            'sector': 'Unknown',
                            'market_cap': '',
                            'headquarters': 'Unknown',
                            'exchange': 'NSE',
                            'currency': self.currency
                        })
                
                # Save enhanced index
                index_df = pd.DataFrame(index_data)
                index_df.to_csv(index_path, index=False)
                
                print(f"Updated dynamic index with {len(symbols)} symbols")
            else:
                # Create empty index file
                pd.DataFrame({'symbol': []}).to_csv(index_path, index=False)
                
        except Exception as e:
            print(f"Error updating dynamic index: {str(e)}")
    
    def get_latest_prices(self) -> pd.DataFrame:
        """Get all latest prices from CSV file"""
        try:
            if os.path.exists(self.latest_prices_file):
                return pd.read_csv(self.latest_prices_file)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading latest prices: {str(e)}")
            return pd.DataFrame()
    
    def _get_latest_day_data(self, symbol: str) -> Dict:
        """
        Get the latest day's open, high, low, volume data from CSV files.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with additional data fields
        """
        try:
            # Try latest directory first (2025 data)
            latest_path = os.path.join(
                self.config.get_data_path('latest', 'ind_stocks', 'individual_files', f'{symbol}.csv')
            )
            
            # Fallback to permanent directory if latest not available
            permanent_path = os.path.join(
                self.config.get_permanent_path('ind_stocks', 'individual_files', f'{symbol}.csv')
            )
            
            csv_path = None
            if os.path.exists(latest_path):
                csv_path = latest_path
            elif os.path.exists(permanent_path):
                csv_path = permanent_path
            
            if not csv_path:
                print(f"No CSV file found for {symbol}")
                return {}
            
            # Read the latest row from CSV
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                if df.empty:
                    return {}
                
                # Get the most recent row
                latest_row = df.iloc[-1]
                
                return {
                    'open': float(latest_row['open']) if 'open' in latest_row and pd.notna(latest_row['open']) else None,
                    'high': float(latest_row['high']) if 'high' in latest_row and pd.notna(latest_row['high']) else None,
                    'low': float(latest_row['low']) if 'low' in latest_row and pd.notna(latest_row['low']) else None,
                    'volume': int(latest_row['volume']) if 'volume' in latest_row and pd.notna(latest_row['volume']) else None,
                    'close': float(latest_row['close']) if 'close' in latest_row and pd.notna(latest_row['close']) else None
                }
            except ImportError:
                # Fallback to csv module
                import csv
                rows = []
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                if not rows:
                    return {}
                
                # Get the last row
                latest_row = rows[-1]
                
                return {
                    'open': float(latest_row.get('open', 0)) if latest_row.get('open') else None,
                    'high': float(latest_row.get('high', 0)) if latest_row.get('high') else None,
                    'low': float(latest_row.get('low', 0)) if latest_row.get('low') else None,
                    'volume': int(latest_row.get('volume', 0)) if latest_row.get('volume') else None,
                    'close': float(latest_row.get('close', 0)) if latest_row.get('close') else None
                }
            
        except Exception as e:
            print(f"Error getting latest day data for {symbol}: {e}")
            return {}

    def fetch_multiple_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch current prices for multiple symbols.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dict with results and statistics
        """
        print(f"Fetching current prices for {len(symbols)} Indian stock symbols...")
        print("Using fallback chain: Upstox → NSEPython → yfinance → NSELib")
        
        results = {
            'successful': [],
            'failed': [],
            'cached': [],
            'errors': []
        }
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
            
            try:
                # Check cache first
                if self._is_cache_valid(symbol):
                    cached_data = self.cache[symbol]['data']
                    results['cached'].append(cached_data)
                    print(f"{symbol}: Using cached data")
                    continue
                
                # Fetch fresh data
                data = self.fetch_current_price(symbol)
                results['successful'].append(data)
                
            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                results['failed'].append(symbol)
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
            
            # Rate limiting between symbols
            if i < len(symbols):
                time.sleep(self.rate_limit_delay)
        
        # Print summary
        print(f"\n" + "=" * 50)
        print("FETCH SUMMARY")
        print("=" * 50)
        print(f"Total symbols: {len(symbols)}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Cached: {len(results['cached'])}")
        print(f"Failed: {len(results['failed'])}")
        
        if results['errors']:
            print(f"\nErrors:")
            for error in results['errors']:
                print(f"  ❌ {error}")
        
        return results

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch current Indian stock prices using multiple sources")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch all symbols from index")
    
    args = parser.parse_args()
    
    fetcher = IndianCurrentFetcher()
    
    if args.all:
        # Load symbols from index
        past_index = os.path.join(fetcher.data_dir, 'past', 'ind_stocks', 'index_ind_stocks.csv')
        if os.path.exists(past_index):
            df = pd.read_csv(past_index)
            symbols = df['symbol'].tolist()
            results = fetcher.fetch_multiple_prices(symbols)
        else:
            print("Index file not found. Use --symbols to specify symbols.")
    elif args.symbols:
        results = fetcher.fetch_multiple_prices(args.symbols)
    else:
        print("Please specify --symbols or --all")
        return
    
    # Show latest prices
    latest_prices = fetcher.get_latest_prices()
    if not latest_prices.empty:
        print(f"\nLatest prices ({len(latest_prices)} symbols):")
        for _, row in latest_prices.head(10).iterrows():
            print(f"  {row['symbol']}: ₹{row['price']:.2f} ({row['source']})")
        if len(latest_prices) > 10:
            print(f"  ... and {len(latest_prices) - 10} more")

if __name__ == "__main__":
    main()