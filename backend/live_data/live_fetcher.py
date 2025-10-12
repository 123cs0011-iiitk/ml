import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Try to import pandas, fallback to CSV module if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    import csv
    PANDAS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveFetcher:
    """
    Live stock price fetcher with multiple API fallbacks:
    1. yfinance (primary, free)
    2. Finnhub (fallback, requires API key)
    3. Alpha Vantage (fallback, requires API key)
    """
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.latest_dir = os.path.join(self.data_dir, 'latest')
        
        # Ensure directories exist
        self._ensure_directories()
        
        # API configurations
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.alphavantage_api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        
        # Cache configuration
        self.cache_duration = int(os.getenv('CACHE_DURATION', 60))
        self.min_request_delay = float(os.getenv('MIN_REQUEST_DELAY', 2.0))
        self.cache = {}  # {symbol: {data: dict, timestamp: datetime}}
        self.last_request_time = 0  # Track last API request time
        
        # Stock categorization rules
        self.us_stock_suffixes = ['.US']
        self.indian_stock_suffixes = ['.NS', '.BO']
        
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.latest_dir,
            os.path.join(self.latest_dir, 'us_stocks'),
            os.path.join(self.latest_dir, 'ind_stocks'),
            os.path.join(self.latest_dir, 'others_stocks')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _categorize_stock(self, symbol: str) -> str:
        """Categorize stock based on symbol suffix"""
        symbol_upper = symbol.upper()
        
        for suffix in self.indian_stock_suffixes:
            if symbol_upper.endswith(suffix):
                return 'ind_stocks'
        
        for suffix in self.us_stock_suffixes:
            if symbol_upper.endswith(suffix):
                return 'us_stocks'
        
        # Default to US stocks for common symbols, others for everything else
        common_us_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ']
        if symbol_upper.split('.')[0] in common_us_symbols:
            return 'us_stocks'
        
        return 'others_stocks'
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def _fetch_from_yfinance(self, symbol: str) -> Tuple[float, str]:
        """Fetch stock price from yfinance with retry logic"""
        try:
            # Enforce rate limiting
            self._enforce_rate_limit()
            
            # Add delay to avoid rate limiting
            time.sleep(1.5)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if 'currentPrice' in info:
                price = info['currentPrice']
                company_name = info.get('longName', symbol)
                return float(price), company_name
            elif 'regularMarketPrice' in info:
                price = info['regularMarketPrice']
                company_name = info.get('longName', symbol)
                return float(price), company_name
            else:
                # Try getting recent price data
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    company_name = info.get('longName', symbol)
                    return float(price), company_name
                else:
                    raise ValueError(f"No price data available for {symbol}")
                    
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {str(e)}")
            raise
    
    def _fetch_from_finnhub(self, symbol: str) -> Tuple[float, str]:
        """Fetch stock price from Finnhub API"""
        if not self.finnhub_api_key:
            raise ValueError("Finnhub API key not configured")
        
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.finnhub_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'c' in data and data['c'] is not None:
                price = data['c']
                # Get company name
                profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={self.finnhub_api_key}"
                profile_response = requests.get(profile_url, timeout=10)
                
                company_name = symbol
                if profile_response.status_code == 200:
                    profile_data = profile_response.json()
                    company_name = profile_data.get('name', symbol)
                
                return float(price), company_name
            else:
                raise ValueError(f"No price data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {str(e)}")
            raise
    
    def _fetch_from_alphavantage(self, symbol: str) -> Tuple[float, str]:
        """Fetch stock price from Alpha Vantage API"""
        if not self.alphavantage_api_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.alphavantage_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                price_str = quote.get('05. price')
                if price_str and price_str != 'None':
                    price = float(price_str)
                    # Alpha Vantage doesn't provide company name in quote endpoint
                    company_name = symbol
                    return price, company_name
                else:
                    raise ValueError(f"No price data available for {symbol}")
            else:
                raise ValueError(f"No price data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {str(e)}")
            raise
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data for symbol is still valid"""
        if symbol not in self.cache:
            return False
        
        cache_entry = self.cache[symbol]
        cache_time = cache_entry['timestamp']
        
        # Check if cache is still within duration
        return datetime.now() - cache_time < timedelta(seconds=self.cache_duration)
    
    def _enforce_rate_limit(self):
        """Enforce minimum delay between API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_delay:
            sleep_time = self.min_request_delay - time_since_last_request
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_live_price(self, symbol: str) -> Dict:
        """Fetch live stock price using fallback strategy with caching"""
        if not symbol or not symbol.strip():
            raise ValueError("Stock symbol is required")
        
        symbol = symbol.strip().upper()
        
        # Check cache first
        if self._is_cache_valid(symbol):
            cached_data = self.cache[symbol]['data']
            logger.info(f"Returning cached data for {symbol} (age: {(datetime.now() - self.cache[symbol]['timestamp']).seconds}s)")
            return cached_data
        
        timestamp = datetime.now().isoformat()
        
        # Try APIs in order of preference
        apis = [
            ('yfinance', self._fetch_from_yfinance),
            ('finnhub', self._fetch_from_finnhub),
            ('alphavantage', self._fetch_from_alphavantage)
        ]
        
        last_error = None
        
        for api_name, api_func in apis:
            try:
                logger.info(f"Trying {api_name} for symbol {symbol}")
                price, company_name = api_func(symbol)
                
                result = {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': timestamp,
                    'source': api_name,
                    'company_name': company_name
                }
                
                # Cache the result
                self.cache[symbol] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
                
                # Save to CSV
                self.save_to_csv(result)
                
                logger.info(f"Successfully fetched {symbol} price ${price} from {api_name}")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"{api_name} failed for {symbol}: {str(e)}")
                continue
        
        # All APIs failed
        raise Exception(f"All APIs failed to fetch price for {symbol}. Last error: {str(last_error)}")
    
    def save_to_csv(self, data: Dict):
        """Save stock data to appropriate CSV file"""
        symbol = data['symbol']
        category = self._categorize_stock(symbol)
        
        csv_path = os.path.join(self.latest_dir, category, 'latest_prices.csv')
        
        try:
            if PANDAS_AVAILABLE:
                self._save_with_pandas(data, csv_path, symbol, category)
            else:
                self._save_with_csv_module(data, csv_path, symbol, category)
            
            logger.info(f"Saved {symbol} data to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving to CSV for {symbol}: {str(e)}")
            # Don't raise exception to avoid breaking the main flow
    
    def _save_with_pandas(self, data: Dict, csv_path: str, symbol: str, category: str):
        """Save using pandas"""
        # Create DataFrame with the new data
        new_row = pd.DataFrame([{
            'symbol': data['symbol'],
            'price': data['price'],
            'timestamp': data['timestamp'],
            'source': data['source'],
            'company_name': data['company_name']
        }])
        
        # Read existing data if file exists
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            # Remove any existing entry for this symbol
            existing_df = existing_df[existing_df['symbol'] != symbol]
            # Append new data
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            updated_df = new_row
        
        # Save updated data
        updated_df.to_csv(csv_path, index=False)
        
        # Update dynamic index
        self.update_dynamic_index(category)
    
    def _save_with_csv_module(self, data: Dict, csv_path: str, symbol: str, category: str):
        """Save using built-in csv module (fallback)"""
        # Read existing data
        existing_data = []
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_data = [row for row in reader if row['symbol'] != symbol]
        
        # Add new data
        existing_data.append({
            'symbol': data['symbol'],
            'price': str(data['price']),
            'timestamp': data['timestamp'],
            'source': data['source'],
            'company_name': data['company_name']
        })
        
        # Write updated data
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['symbol', 'price', 'timestamp', 'source', 'company_name']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
        
        # Update dynamic index
        self.update_dynamic_index(category)
    
    def update_dynamic_index(self, category: str):
        """Update the dynamic index CSV file for the category"""
        try:
            csv_path = os.path.join(self.latest_dir, category, 'latest_prices.csv')
            index_path = os.path.join(self.data_dir, f'index_{category}_dynamic.csv')
            
            if os.path.exists(csv_path):
                if PANDAS_AVAILABLE:
                    df = pd.read_csv(csv_path)
                    symbols = df['symbol'].unique().tolist()
                    
                    # Save symbols to index file
                    index_df = pd.DataFrame({'symbol': symbols})
                    index_df.to_csv(index_path, index=False)
                else:
                    # Use csv module
                    symbols = set()
                    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            symbols.add(row['symbol'])
                    
                    # Save symbols to index file
                    with open(index_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['symbol'])
                        for symbol in sorted(symbols):
                            writer.writerow([symbol])
                
                logger.info(f"Updated index for {category} with {len(symbols)} symbols")
            else:
                # Create empty index file
                if PANDAS_AVAILABLE:
                    pd.DataFrame({'symbol': []}).to_csv(index_path, index=False)
                else:
                    with open(index_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['symbol'])
                
        except Exception as e:
            logger.error(f"Error updating index for {category}: {str(e)}")
    
    def get_latest_prices(self, category: Optional[str] = None):
        """Get latest prices for a category or all categories"""
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available, returning empty list")
            return []
        
        if category:
            csv_path = os.path.join(self.latest_dir, category, 'latest_prices.csv')
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
            else:
                return pd.DataFrame()
        else:
            # Get all categories
            all_data = []
            for cat in ['us_stocks', 'ind_stocks', 'others_stocks']:
                csv_path = os.path.join(self.latest_dir, cat, 'latest_prices.csv')
                if os.path.exists(csv_path):
                    cat_df = pd.read_csv(csv_path)
                    cat_df['category'] = cat
                    all_data.append(cat_df)
            
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()
