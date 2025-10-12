# Live Stock Price Fetching Feature

This feature implements real-time stock price fetching with a Python backend and React frontend integration.

## 🚀 Simple Instructions (TL;DR)

### Backend:
```bash
cd backend
py -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
py run_server.py
```

### Frontend (in another terminal):
```bash
cd frontend
npm install
npm run dev
```

### Test:
Open `http://localhost:5173` and search for stocks like `AAPL` or `GOOGL`!

---

## 🚀 Detailed Setup

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   py -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server:**
   ```bash
   py run_server.py
   ```
   
   The server will start on `http://localhost:5000` (or the next available port).

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies (if not already done):**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser to:**
   ```
   http://localhost:5173
   ```

## 🧪 Testing

### Backend API Testing

1. **Run the Python test suite:**
   ```bash
   cd backend
   # Make sure virtual environment is activated
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # macOS/Linux
   py test_api.py
   ```

2. **Or use the web-based test interface:**
   ```bash
   # Start backend server first
   py run_server.py
   
   # Then open in browser:
   open frontend/test-integration.html
   ```

### Manual Testing

1. **Test health endpoint:**
   ```bash
   curl http://localhost:5000/health
   ```

2. **Test live price:**
   ```bash
   curl "http://localhost:5000/live_price?symbol=AAPL"
   ```

3. **Test invalid symbol:**
   ```bash
   curl "http://localhost:5000/live_price?symbol=INVALID"
   ```

## 📁 Project Structure

```
ml/
├── backend/
│   ├── live_data/
│   │   └── live_fetcher.py      # Core live price fetching logic
│   ├── app.py                   # Flask API server
│   ├── run_server.py           # Server startup script
│   ├── test_api.py             # Backend test suite
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example           # Environment variables template
│   └── README.md              # Backend documentation
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── stockService.ts  # Updated to call backend API
│   │   ├── components/
│   │   │   └── StockInfo.tsx    # Updated to show live price data
│   │   └── App.tsx             # Updated to use live data
│   └── test-integration.html   # Frontend integration test
├── data/
│   ├── latest/                 # Live price CSV storage
│   │   ├── us_stocks/
│   │   ├── ind_stocks/
│   │   └── others_stocks/
│   └── index_*_stocks_dynamic.csv  # Dynamic symbol indexes
└── README_LIVE_PRICE_FEATURE.md    # This file
```

## 🔧 API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "Live Stock Price API",
  "timestamp": "2024-01-15T10:30:00"
}
```

### GET /live_price?symbol=SYMBOL
Fetch live stock price for a symbol.

**Example:** `GET /live_price?symbol=AAPL`

**Success Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "price": 175.43,
    "timestamp": "2024-01-15T10:30:00",
    "source": "yfinance",
    "company_name": "Apple Inc."
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Invalid symbol",
  "message": "No price data available for INVALID"
}
```

### GET /latest_prices?category=CATEGORY
Get latest prices for a category (us_stocks, ind_stocks, others_stocks).

### GET /symbols?category=CATEGORY
Get all available symbols by category.

## 🔄 Data Flow

1. **User searches for stock symbol** in frontend
2. **Frontend calls** `/live_price?symbol=SYMBOL` endpoint
3. **Backend LiveFetcher** tries APIs in order:
   - yfinance (primary)
   - Finnhub (fallback 1)
   - Alpha Vantage (fallback 2)
4. **Price data saved** to appropriate CSV file
5. **Dynamic index updated** with new symbol
6. **Response returned** to frontend
7. **Frontend displays** live price with source indicator

## 📊 Stock Categorization

- **US Stocks**: Common symbols (AAPL, GOOGL, etc.) and symbols ending with .US
- **Indian Stocks**: Symbols ending with .NS or .BO
- **Others**: Everything else

## 💾 CSV Storage

### Latest Prices
- `data/latest/us_stocks/latest_prices.csv`
- `data/latest/ind_stocks/latest_prices.csv`
- `data/latest/others_stocks/latest_prices.csv`

**CSV Format:**
```csv
symbol,price,timestamp,source,company_name
AAPL,175.43,2024-01-15T10:30:00,yfinance,Apple Inc.
```

### Dynamic Indexes
- `data/index_us_stocks_dynamic.csv`
- `data/index_ind_stocks_dynamic.csv`
- `data/index_others_stocks_dynamic.csv`

**Index Format:**
```csv
symbol
AAPL
GOOGL
MSFT
```

## 🛠️ Configuration

### Environment Variables

Create `backend/.env` file:
```env
# Optional API keys for fallback services
FINNHUB_API_KEY=your_finnhub_api_key_here
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key_here

# Server port
PORT=5000
```

### API Keys (Optional)

1. **Finnhub**: https://finnhub.io/register (60 requests/minute free)
2. **Alpha Vantage**: https://www.alphavantage.co/support/#api-key (25 requests/day free)

## 🔍 Error Handling

### Backend Errors
- **Invalid Symbol**: Returns 404 with clear message
- **API Failures**: Tries fallback APIs, returns 500 if all fail
- **Network Issues**: Proper timeout handling (10 seconds)

### Frontend Errors
- **Network Errors**: "Unable to connect to server"
- **Invalid Symbol**: "Symbol not found"
- **Timeout**: "Request timed out, please try again"

## 🚦 Rate Limiting

- **yfinance**: No rate limits (primary source)
- **Finnhub**: 60 requests/minute (free tier)
- **Alpha Vantage**: 25 requests/day (free tier)

## 🔧 Troubleshooting

### Backend Won't Start
1. Check if port 5000 is available: `netstat -an | grep 5000`
2. Try a different port: `PORT=5001 py run_server.py`
3. Check Python version: `py --version` (3.7+ required)
4. **Ensure virtual environment is activated**: `which python` should show venv path
5. **Reinstall dependencies**: `pip install -r requirements.txt --force-reinstall`

### Virtual Environment Issues
1. **Virtual environment not activated**: Look for `(venv)` in your terminal prompt
2. **Wrong Python version**: Create venv with specific version: `py -3.9 -m venv venv`
3. **Dependencies not found**: Reinstall in activated venv: `pip install -r requirements.txt`
4. **Permission errors**: Use `py -m venv venv` instead of `virtualenv venv`

### Frontend Can't Connect
1. Ensure backend is running on port 5000
2. Check browser console for CORS errors
3. Verify backend URL in `frontend/src/services/stockService.ts`

### No Price Data
1. Check internet connection
2. Verify symbol format (uppercase, no spaces)
3. Try different symbols (AAPL, GOOGL, MSFT)

### CSV Files Not Created
1. Check write permissions in `data/` directory
2. Ensure `data/latest/` subdirectories exist
3. Check backend logs for errors

## 🎯 Testing Checklist

- [ ] Backend health check returns 200
- [ ] Valid symbol (AAPL) returns live price
- [ ] Invalid symbol returns 404 error
- [ ] CSV files are created/updated
- [ ] Dynamic indexes are updated
- [ ] Frontend displays live price
- [ ] Source indicator shows correct API
- [ ] Error messages are user-friendly
- [ ] Port cleanup works on shutdown

## 🔮 Future Enhancements

- Historical price data endpoint
- Real-time WebSocket updates
- Price change indicators
- Company information fetching
- ML prediction integration
- Caching improvements
- Rate limiting per user
- Authentication system

## 📝 Notes

- This feature focuses only on live price fetching
- ML predictions and company info are placeholders
- CSV storage is optional but recommended for caching
- All APIs have fallback mechanisms
- Frontend maintains backward compatibility
- No database required - uses CSV files
