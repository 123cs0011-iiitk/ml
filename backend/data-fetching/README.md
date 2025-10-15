# Stock Data Fetching Module

This module handles all stock data fetching and updates with a structured approach for different data types and time periods.

## Structure

### Current Fetching (Real-time Live Prices)
- **Purpose**: Fetch current live stock prices
- **US Stocks**: Finnhub API (rate-limited: 60 calls/min)
- **Indian Stocks**: Multi-source fallback chain:
  1. Upstox API (primary - requires API key)
  2. NSEPython library (fallback #1)
  3. yfinance with .NS suffix (fallback #2)
  4. NSELib library (fallback #3)
- **Usage**: For real-time price display in frontend
- **Cache**: 60 seconds
- **Output**: `data/latest/{category}/latest_prices.csv`
- **Currency**: USD for US stocks, INR for Indian stocks

### Historical Fetching (5-Year Data)
- **Purpose**: Fetch historical stock data for analysis
- **Period**: 2020-01-01 to 2024-12-31 (5 years)
- **API**: yfinance (no rate limits)
- **Output**: `data/past/{category}/individual_files/{SYMBOL}.csv`

### Latest Fetching (Recent Data)
- **Purpose**: Fetch data after historical period
- **Period**: 2025-01-01 to current date
- **US Stocks**: yfinance (primary), Alpha Vantage (fallback)
- **Indian Stocks**: yfinance with .NS suffix (primary), NSELib (fallback)
- **Output**: `data/latest/{category}/individual_files/{SYMBOL}.csv`
- **Currency**: USD for US stocks, INR for Indian stocks

## Data Format

All CSV files use lowercase column names:
- `date, open, high, low, close, volume, adjusted_close, currency`

Index files maintain detailed information in alphabetical order:
- `symbol, company_name, sector, market_cap, headquarters, exchange`

## Directory Structure

```
data-fetching/
├── __init__.py
├── data_manager.py              # Module manager
├── current_fetcher.py           # Main live price fetcher
├── us_stocks/
│   ├── current-fetching/
│   │   └── finnhub_fetcher.py  # Finnhub API for live prices
│   ├── historical-fetching/
│   │   └── yfinance_historical.py  # 2020-2024 data
│   └── latest-fetching/
│       └── yfinance_latest.py  # 2025-current data
├── ind_stocks/
│   ├── current-fetching/
│   │   └── ind_current_fetcher.py  # Multi-source: Upstox → NSEPython → yfinance → NSELib
│   ├── historical-fetching/
│   │   └── yfinance_historical.py  # 2020-2024 data (.NS suffix)
│   └── latest-fetching/
│       └── yfinance_latest.py  # 2025-current data (.NS suffix)
└── test/
    ├── __init__.py
    └── test_data_fetcher.py    # Test suite
```

## API Setup

### Upstox API Setup (for Indian Stocks)
1. Visit [Upstox Developer Portal](https://upstox.com/developer/)
2. Create an account and get API credentials
3. Add to your `.env` file:
   ```
   UPSTOX_API_KEY=your_api_key_here
   UPSTOX_ACCESS_TOKEN=your_access_token_here
   ```

### Currency Exchange Rates
- Live USD-INR rates are fetched automatically
- Fallback sources: forex-python → exchangerate-api.com → hardcoded (83.5)
- Rates are cached for 1 hour to avoid excessive API calls

## Usage Examples

### Current Price Fetching (Live Prices)

```python
from data_fetching.current_fetcher import CurrentFetcher

# Initialize fetcher
fetcher = CurrentFetcher()

# Fetch single US stock price
result = fetcher.fetch_live_price("AAPL")
print(f"AAPL: ${result['price']} from {result['source']}")

# Fetch single Indian stock price
result = fetcher.fetch_live_price("RELIANCE")
print(f"RELIANCE: ₹{result['price']} from {result['source']}")

# Fetch multiple prices (mixed US and Indian)
symbols = ["AAPL", "GOOGL", "RELIANCE", "TCS"]
results = fetcher.fetch_multiple_prices(symbols)
```

### Historical Data Fetching

```python
from data_fetching.us_stocks.historical_fetching.yfinance_historical import USHistoricalFetcher

# Initialize fetcher
fetcher = USHistoricalFetcher()

# Fetch historical data for all symbols
stats = fetcher.fetch_historical_data()

# Fetch specific symbols
stats = fetcher.fetch_historical_data(symbols_to_download=["AAPL", "GOOGL"])
```

### Latest Data Fetching

```python
from data_fetching.us_stocks.latest_fetching.yfinance_latest import USLatestFetcher

# Initialize fetcher
fetcher = USLatestFetcher()

# Fetch latest data
stats = fetcher.fetch_latest_data()
```

### Indian Stocks (with .NS suffix)

```python
from data_fetching.ind_stocks.historical_fetching.yfinance_historical import IndianHistoricalFetcher

# Initialize fetcher
fetcher = IndianHistoricalFetcher()

# Fetch Indian stock data (automatically adds .NS suffix)
stats = fetcher.fetch_historical_data(symbols_to_download=["RELIANCE", "TCS"])
```

## Command Line Usage

### Historical Data
```bash
# US stocks historical data
python -m data_fetching.us_stocks.historical_fetching.yfinance_historical --force

# Indian stocks historical data
python -m data_fetching.ind_stocks.historical_fetching.yfinance_historical --symbols RELIANCE TCS
```

### Latest Data
```bash
# US stocks latest data
python -m data_fetching.us_stocks.latest_fetching.yfinance_latest --force

# Indian stocks latest data
python -m data_fetching.ind_stocks.latest_fetching.yfinance_latest --symbols RELIANCE TCS
```

### Current Prices
```bash
# US stocks current prices
python -m data_fetching.us_stocks.current_fetching.finnhub_fetcher --all

# Indian stocks current prices
python -m data_fetching.ind_stocks.current_fetching.finnhub_fetcher --symbols RELIANCE TCS
```

## API Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Keys (Optional - for fallback when yfinance fails)
FINNHUB_API_KEY=your_finnhub_api_key_here
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key_here

# Server Configuration
PORT=5000
CACHE_DURATION=60
MIN_REQUEST_DELAY=2.0
```

### API Rate Limits

- **yfinance**: No rate limits (primary choice)
- **Finnhub**: 60 calls/minute (free tier)
- **Alpha Vantage**: 5 calls/minute (free tier), 500 calls/day

## Data Storage

### File Locations

```
data/
├── latest/
│   ├── us_stocks/
│   │   ├── latest_prices.csv          # Live price cache
│   │   └── individual_files/
│   │       ├── AAPL.csv               # Latest data (2025-current)
│   │       └── GOOGL.csv
│   └── ind_stocks/
│       ├── latest_prices.csv
│       └── individual_files/
│           ├── RELIANCE.csv
│           └── TCS.csv
├── past/
│   ├── us_stocks/
│   │   ├── index_us_stocks.csv        # Historical index
│   │   └── individual_files/
│   │       ├── AAPL.csv               # Historical data (2020-2024)
│   │       └── GOOGL.csv
│   └── ind_stocks/
│       ├── index_ind_stocks.csv
│       └── individual_files/
│           ├── RELIANCE.csv
│           └── TCS.csv
└── index_{category}_dynamic.csv       # Dynamic indexes
```

### Index Files

Index files contain detailed company information in alphabetical order:

```csv
symbol,company_name,sector,market_cap,headquarters,exchange
AAPL,Apple Inc.,Technology,,Cupertino California,NASDAQ
GOOGL,Alphabet Inc.,Technology,,Mountain View California,NASDAQ
```

## Error Handling

### Fallback Strategy

1. **yfinance** (primary) - No rate limits
2. **Finnhub** (fallback) - Rate limited
3. **Alpha Vantage** (fallback) - Rate limited
4. **Permanent directory** (last resort) - Local data

### Error Types

- **Invalid Symbol**: Returns 404 with error message
- **API Failures**: Tries fallback APIs, returns 500 if all fail
- **Network Issues**: Proper timeout handling and retry logic
- **Rate Limiting**: Automatic delays between requests

## Testing

### Run Tests

```bash
# Run all data fetching tests
python -m pytest data-fetching/test/ -v

# Run specific test file
python -m pytest data-fetching/test/test_data_fetcher.py -v
```

### Test Coverage

- ✅ Live price fetching (US and Indian stocks)
- ✅ Error handling and fallback APIs
- ✅ Caching functionality
- ✅ Rate limiting
- ✅ Data format validation
- ✅ Stock categorization
- ✅ CSV saving and loading
- ✅ Alphabetical ordering

## Key Features

### 1. Alphabetical Ordering
All data additions maintain alphabetical order in index files and individual data.

### 2. Standardized Format
All CSV files use lowercase column names for consistency.

### 3. Currency Support
- US stocks: USD
- Indian stocks: INR
- Others: USD (default)

### 4. Automatic Suffix Handling
Indian stocks automatically get `.NS` suffix for yfinance queries.

### 5. Caching
60-second cache for live prices to reduce API calls.

### 6. Rate Limiting
Respects API rate limits with automatic delays.

## Migration Notes

- **Backward Compatibility**: `LiveFetcher` alias maintained for existing code
- **Import Changes**: Updated from `live-data` to `data-fetching`
- **New Structure**: Organized by data type and time period
- **Enhanced Features**: Added currency support and alphabetical ordering

## Contributing

When adding new features:

1. **Follow the structure**: Place code in appropriate directories
2. **Use shared utilities**: Leverage common functions from `shared/`
3. **Maintain alphabetical order**: All data additions should be sorted
4. **Add tests**: Ensure comprehensive test coverage
5. **Update documentation**: Keep this README current
