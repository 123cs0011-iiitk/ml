# Upstox API Integration

## ✅ Implementation Complete

The Upstox API integration has been successfully implemented for Indian stock data fetching with OAuth2 authentication and automatic token refresh.

## 🔧 Key Features

### OAuth2 Authentication
- **Automatic Token Refresh**: Proactive and reactive refresh strategies
- **Secure Token Storage**: `.env` file and JSON cache
- **Thread-safe Updates**: File locks for concurrent access
- **Error Handling**: Graceful fallback when tokens expire

### Data Sources
- **Primary**: Upstox API (real-time NSE data)
- **Fallback Chain**: NSEPython → yfinance → stock-market-india → NSELib → Permanent data
- **Rate Limits**: 50 requests/second, 500 requests/minute

### Supported Stocks
- **20 Major Indian Stocks**: Hardcoded mappings (RELIANCE, TCS, HDFCBANK, etc.)
- **Dynamic Fallback**: `NSE_EQ|SYMBOL` format for other stocks
- **500+ Total Stocks**: Available in permanent directory

## 🚀 Quick Setup

### 1. OAuth Setup (Recommended)

**Windows:**
```cmd
cd backend
venv\Scripts\activate
py scripts/setup_upstox_oauth.py
```

**Linux/macOS:**
```bash
cd backend
source venv/bin/activate
python scripts/setup_upstox_oauth.py
```

- Enter Client ID and Client Secret
- Complete authorization in browser
- Tokens automatically saved

### 2. Manual Setup (Legacy)
```env
UPSTOX_API_KEY=your_api_key
UPSTOX_ACCESS_TOKEN=your_access_token
UPSTOX_REDIRECT_URI=http://localhost:8080
```

## 🔧 Architecture

### Token Management
- **Primary Storage**: `.env` file (human-readable)
- **Backup Storage**: `backend/_cache/upstox_tokens.json`
- **Auto-sync**: Tokens kept in sync between both locations

### Fallback Strategy
1. **Upstox API** (primary) - Real-time NSE data
2. **NSEPython** (fallback 1) - If Upstox fails
3. **yfinance** (fallback 2) - If NSEPython fails
4. **stock-market-india** (fallback 3) - If yfinance fails
5. **NSELib** (fallback 4) - If stock-market-india fails
6. **Permanent directory** (last resort) - Historical data

## 📊 Usage

### Current Price Fetching
```python
from data_fetching.ind_stocks.current_fetching.ind_current_fetcher import IndianCurrentFetcher

fetcher = IndianCurrentFetcher()
result = fetcher.fetch_current_price("TCS")
print(f"TCS: ₹{result['price']} from {result['source']}")
```

### Batch Fetching
```python
# Fetch multiple stocks at once
symbols = ["RELIANCE", "TCS", "HDFCBANK"]
results = fetcher.fetch_batch_prices_upstox(symbols)
```

## 🔍 Troubleshooting

### Common Issues

**Token refresh failed:**
- Run OAuth setup script again
- Check refresh token validity

**Missing OAuth credentials:**
- Run `python scripts/setup_upstox_oauth.py`
- Or manually add credentials to `.env`

**Access forbidden:**
- Check Upstox Developer Console
- Ensure API access is enabled

**No ISIN mapping found:**
- Normal for new/unlisted stocks
- System will fallback to other sources

### Debug Information
```bash
# Check token status
python backend/test_upstox_api.py

# View token information
python -c "from shared.upstox_token_manager import UpstoxTokenManager; print(UpstoxTokenManager().get_token_info())"
```

## 📈 Performance

### Rate Limit Optimization
- **Upstox used ONLY for**: Current/live price (LTP) + today's OHLCV
- **NOT used for**: Historical data fetching (avoids rate limits)
- **Safe usage**: Checking current prices for monitoring/trading

### Data Storage Strategy
- **permanent/** - Historical data (2020-2024) - READ-ONLY
- **data/past/** - Same as permanent - READ-ONLY
- **data/latest/** - Daily data from 2025 onwards + current prices

## 🎯 Benefits

- **More reliable** Indian stock data through Upstox API
- **Real-time NSE data** instead of delayed yfinance data
- **Better performance** with batch fetching capability
- **Maintained compatibility** with existing codebase
- **Robust fallback chain** ensures data availability
- **Automatic daily data storage** for 2025+ data

## 🔄 Token Refresh

### Automatic Refresh
- **Proactive**: Checks token expiry before each API call
- **Reactive**: Detects 401 errors and refreshes automatically
- **Maximum Retries**: 2 retry attempts per API call

### Manual Refresh

**Windows:**
```cmd
cd backend
venv\Scripts\activate
py scripts/setup_upstox_oauth.py
py -c "from shared.upstox_token_manager import UpstoxTokenManager; UpstoxTokenManager().refresh_access_token()"
```

**Linux/macOS:**
```bash
cd backend
source venv/bin/activate
python scripts/setup_upstox_oauth.py
python -c "from shared.upstox_token_manager import UpstoxTokenManager; UpstoxTokenManager().refresh_access_token()"
```

## 📚 Documentation

- [Backend README](backend/README.md) - Complete API documentation
- [Data Fetching README](backend/data-fetching/README.md) - Data operations guide
- [Main README](README.md) - Project overview

---

**Ready to use?** Follow the Quick Setup guide above to get started with Upstox integration!