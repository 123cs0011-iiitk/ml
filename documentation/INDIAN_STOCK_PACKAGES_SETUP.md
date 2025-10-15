# Indian Stock Data Packages Setup Guide

This guide explains how to set up and use various Python packages for fetching Indian stock market data.

## Available Packages

### 1. **yfinance** (Recommended for historical data)
- **Installation**: `pip install yfinance`
- **Pros**: Simple, good for historical data, widely used
- **Cons**: Delayed data, connectivity issues with Indian stocks
- **Usage**: `yf.Ticker("TCS.NS").history(period="1d")`

### 2. **stock-market-india** (Recommended for real-time quotes)
- **Installation**: `pip install stock-market-india`
- **Pros**: Real-time data, direct NSE access, reliable
- **Cons**: May not have historical data
- **Usage**: `StockMarketIndia().get_quote('TCS')`

### 3. **india-stocks-api** (For broker integration)
- **Installation**: `pip install india-stocks-api`
- **Pros**: Real-time data, broker integration
- **Cons**: Requires broker account (AngelOne/Upstox/Zerodha)
- **Usage**: `AngelOne(api_key="KEY").get_ltp(symbol="RELIANCE")`

### 4. **Alpha Vantage** (Alternative API)
- **Installation**: `pip install alpha_vantage`
- **Pros**: Free API, multiple data types
- **Cons**: Limited free tier, delayed data
- **Usage**: `TimeSeries(key='KEY').get_intraday(symbol='TCS.BSE')`

## Current Implementation

Our system now uses a **comprehensive fallback chain** for Indian stocks:

```
Upstox API → NSEPython → yfinance → stock-market-india → NSELib → Permanent Directory
```

## Setup Instructions

### 1. Install Required Packages

```bash
# Core packages
pip install yfinance
pip install stock-market-india

# Optional packages (for enhanced functionality)
pip install nsepython
pip install nselib
pip install alpha_vantage
pip install india-stocks-api
```

### 2. Test the Installation

```bash
cd backend
python test_indian_packages.py
```

### 3. Configure API Keys (Optional)

For enhanced functionality, set up API keys:

```bash
# For Upstox API
export UPSTOX_API_KEY="your_upstox_api_key"

# For Alpha Vantage
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

## Usage Examples

### Basic Usage

```python
from data_fetching.ind_stocks.current_fetching.ind_current_fetcher import IndianCurrentFetcher

# Initialize fetcher
fetcher = IndianCurrentFetcher()

# Fetch current price
result = fetcher.fetch_current_price("TCS")
print(f"TCS Price: ₹{result['price']}")
```

### Direct Package Usage

```python
# Using yfinance
import yfinance as yf
tcs = yf.Ticker("TCS.NS")
data = tcs.history(period="1d")
print(data)

# Using stock-market-india
from stock_market_india import StockMarketIndia
smi = StockMarketIndia()
quote = smi.get_quote('TCS')
print(quote)
```

## Fallback Chain Details

### 1. **Upstox API** (Primary)
- Real-time data from Upstox
- Requires API key
- Most reliable for live data

### 2. **NSEPython** (Secondary)
- Direct NSE data access
- No API key required
- Good for basic quotes

### 3. **yfinance** (Tertiary)
- Yahoo Finance data
- Good for historical data
- May have connectivity issues

### 4. **stock-market-india** (Quaternary)
- Python package for NSE data
- Real-time quotes
- Reliable fallback option

### 5. **NSELib** (Quinary)
- Alternative NSE library
- Backup option
- May require additional setup

### 6. **Permanent Directory** (Final)
- Cached data from previous fetches
- Ensures data availability
- Last resort fallback

## Troubleshooting

### Common Issues

1. **yfinance connectivity issues**
   - Solution: Use stock-market-india as fallback
   - Check internet connection
   - Try different time periods

2. **Package not found errors**
   - Solution: Install missing packages
   - Check virtual environment
   - Verify package names

3. **API key errors**
   - Solution: Set up API keys or use free alternatives
   - Check environment variables
   - Verify key format

4. **No data returned**
   - Solution: Check symbol format
   - Verify market hours
   - Try different data sources

### Debug Mode

Enable debug logging to see which data source is being used:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Use caching**: Data is cached for 5 minutes to reduce API calls
2. **Rate limiting**: Built-in delays prevent API rate limit issues
3. **Fallback order**: Most reliable sources are tried first
4. **Error handling**: Graceful degradation ensures data availability

## Testing

Run the test suite to verify all packages work:

```bash
python test_indian_packages.py
```

This will test:
- Package availability
- Data fetching from each source
- Fallback chain functionality
- Error handling

## Support

For issues with specific packages:
- **yfinance**: [GitHub Issues](https://github.com/ranaroussi/yfinance/issues)
- **stock-market-india**: [PyPI Page](https://pypi.org/project/stock-market-india/)
- **india-stocks-api**: [PyPI Page](https://pypi.org/project/india-stocks-api/)

## References

- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [stock-market-india PyPI](https://pypi.org/project/stock-market-india/)
- [india-stocks-api PyPI](https://pypi.org/project/india-stocks-api/)
- [Alpha Vantage API](https://www.alphavantage.co/)
