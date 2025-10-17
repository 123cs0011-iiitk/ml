# Data Fetching Module

Handles real-time stock prices, historical data, and latest data for US and Indian markets with robust fallback systems and intelligent caching.

## Quick Start

```python
from data_fetching.current_fetcher import CurrentFetcher

fetcher = CurrentFetcher()
result = fetcher.fetch_live_price("AAPL")
print(f"AAPL: ${result['price']} from {result['source']}")
```

## Structure

```
data_fetching/
├── current_fetcher.py          # Main live price fetcher
├── data_manager.py             # Module manager
├── us_stocks/                  # US stock data fetching
│   ├── current_fetching/       # Live prices (Finnhub)
│   ├── historical_fetching/    # 2020-2024 data (yfinance)
│   └── latest_fetching/        # 2025-current data (yfinance)
├── ind_stocks/                 # Indian stock data fetching
│   ├── current_fetching/       # Live prices (Upstox → NSEPython → yfinance)
│   ├── historical_fetching/    # 2020-2024 data (yfinance)
│   └── latest_fetching/        # 2025-current data (yfinance)
└── test/                       # Test suite
```

## Data Sources

- **US Stocks**: yfinance (primary) → Finnhub (fallback)
- **Indian Stocks**: Upstox (primary) → NSEPython → yfinance → NSELib → Permanent data
- **Currency**: forex-python → exchangerate-api.com → Yahoo Finance → Hardcoded rate

## Usage

```python
# Current prices
result = fetcher.fetch_live_price("AAPL")
results = fetcher.fetch_multiple_prices(["AAPL", "GOOGL", "TCS"])

# Historical data
from data_fetching.us_stocks.historical_fetching.yfinance_historical import USHistoricalFetcher
us_fetcher = USHistoricalFetcher()
stats = us_fetcher.fetch_historical_data(symbols_to_download=["AAPL", "GOOGL"])
```

## Testing

```bash
# Run data fetching tests
python -m pytest data_fetching/test/ -v
```