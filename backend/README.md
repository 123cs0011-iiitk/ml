# Stock Prediction API - Backend

A Flask-based API for stock prediction with organized modular architecture.

## Directory Structure

The backend has been reorganized for better code organization:

```
backend/
├── scripts/                    # Utility scripts
│   └── run_server.py          # Server startup script
├── tests/                     # All test files
│   ├── __init__.py           # Test package
│   ├── test_api.py           # API endpoint tests
│   ├── test_simple.py        # Simple functionality tests
│   └── test_improvements.py  # Rate limiting tests
├── docs/                      # Documentation
│   └── RATE_LIMITING_IMPROVEMENTS.md  # Technical docs
├── data-fetching/             # Stock data fetching modules
│   ├── __init__.py           # Package initialization
│   ├── data_manager.py       # Module manager
│   ├── current_fetcher.py    # Current live price fetcher
│   ├── us_stocks/            # US stocks data fetching
│   │   ├── current-fetching/ # Live prices (Finnhub)
│   │   ├── historical-fetching/ # 2020-2024 data (yfinance)
│   │   └── latest-fetching/  # 2025-current data (yfinance + Alpha Vantage)
│   ├── ind_stocks/           # Indian stocks data fetching
│   │   ├── current-fetching/ # Live prices (Finnhub)
│   │   ├── historical-fetching/ # 2020-2024 data (yfinance)
│   │   └── latest-fetching/  # 2025-current data (yfinance + Alpha Vantage)
│   └── test/                 # Data fetching tests
│       ├── __init__.py
│       └── test_data_fetcher.py
├── company-info/              # Company information management
│   └── company_info_manager.py
├── algorithms/                # Stock prediction algorithms
│   └── prediction_algorithms.py
├── prediction/               # Prediction orchestration
│   └── prediction_engine.py
├── shared/                   # Shared utilities and common code
│   └── utilities.py
├── main.py                   # Main API coordinator
├── requirements.txt
└── README.md
```

## Module Descriptions

### 🚀 Data Fetching (`data-fetching/`)
Handles all stock data fetching and updates:
- **current_fetcher.py**: Multi-API fallback system for live prices (yfinance → Finnhub → Alpha Vantage)
- **Historical Data**: 2020-2024 data using yfinance (no rate limits)
- **Latest Data**: 2025-current data using yfinance + Alpha Vantage fallback
- **Current Data**: Live prices using Finnhub (rate-limited: 60 calls/min)
- Automatic stock categorization (US, Indian, Others)
- CSV storage with dynamic index updates in alphabetical order
- Standardized lowercase column format across all data

### 🏢 Company Info (`company-info/`) - *Future Implementation*
Will handle company-related data:
- Company fundamentals (P/E ratio, market cap, etc.)
- Company metadata (sector, industry, description)
- Financial statements
- Company news and events
- Corporate actions

### 🤖 Algorithms (`algorithms/`) - *Future Implementation*
Will contain 10+ prediction algorithms:
- **Random Forest**: Ensemble learning for stock prediction
- **LSTM**: Long Short-Term Memory neural networks
- **ARIMA**: AutoRegressive Integrated Moving Average
- **Linear Regression**: Statistical prediction model
- **Support Vector Machine (SVM)**: Classification-based prediction
- **Gradient Boosting**: Advanced ensemble method
- **Neural Networks**: Deep learning approaches
- **Technical Analysis**: Indicator-based predictions
- **Sentiment Analysis**: News and social media sentiment
- **Ensemble Methods**: Combining multiple algorithms

### 📊 Prediction (`prediction/`) - *Future Implementation*
Will orchestrate algorithm usage:
- Algorithm selection and configuration
- Data preprocessing for different algorithms
- Prediction generation and validation
- Result aggregation and formatting
- Performance evaluation and comparison

### 🔧 Shared (`shared/`)
Common utilities used across modules:
- **Configuration Management**: Centralized config handling
- **Logging Utilities**: Consistent logging across modules
- **Data Validation**: Common validation functions
- **Data Structures**: Standardized stock data formats
- **Utility Functions**: Helper functions and constants
- **Error Handling**: Custom exception classes

## Current Features

✅ **Live Stock Prices**: Multi-API fallback system
✅ **Stock Search**: Symbol and company name search
✅ **Stock Categorization**: Automatic US/Indian/Others classification
✅ **CSV Storage**: Dynamic data storage with index management
✅ **CORS Support**: Frontend integration ready
✅ **Error Handling**: Comprehensive error management
✅ **Rate Limiting**: API request throttling
✅ **Caching**: Performance optimization

## Future Features (Placeholders Implemented)

🔄 **Stock Predictions**: Multiple algorithm support
🔄 **Company Information**: Comprehensive company data
🔄 **Algorithm Management**: Dynamic algorithm selection
🔄 **Performance Metrics**: Algorithm comparison and evaluation
🔄 **Historical Data**: Extended historical data fetching
🔄 **News Integration**: Market news and sentiment analysis

## Setup

1. **Create Virtual Environment**:
   ```bash
   cd backend
   py -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables** (Optional):
   Create a `.env` file in the backend directory:
   ```
   # API Keys (Optional - for fallback when Yahoo Finance fails)
   FINNHUB_API_KEY=your_finnhub_api_key_here
   ALPHAVANTAGE_API_KEY=your_alphavantage_api_key_here
   
   # Server Configuration
   PORT=5000
   CACHE_DURATION=60
   MIN_REQUEST_DELAY=2.0
   ```

## Quick Start

### **One Command to Start Backend:**
```bash
cd backend
py run.py
```

### **One Command to Start Frontend:**
```bash
cd frontend
npm run dev
```

That's it! Your stock price API will be running at `http://localhost:5000` and frontend at `http://localhost:5173`

## API Endpoints

### Current Endpoints

#### GET /health
Health check endpoint.

#### GET /live_price?symbol=SYMBOL
Fetch live stock price for a symbol using yfinance with fallback APIs.

**Example**: `GET /live_price?symbol=AAPL`

**Response**:
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "price": 252.20,
    "timestamp": "2025-10-13T12:41:08.296683",
    "source": "yfinance",
    "company_name": "Apple Inc."
  }
}
```

**Error Handling**:
- Invalid symbol: Returns 404 with error message
- API failures: Tries fallback APIs (Finnhub, Alpha Vantage, permanent directory)
- Network issues: Proper timeout handling and retry logic

#### GET /latest_prices?category=CATEGORY
Get latest prices for a category (us_stocks, ind_stocks, others_stocks).

#### GET /symbols?category=CATEGORY
Get all available symbols by category.

#### GET /search?q=QUERY
Search for stocks by symbol or company name.

### Future Endpoints (Placeholders)

#### POST /predict
Generate stock price prediction using selected algorithm.

**Expected JSON payload**:
```json
{
    "symbol": "AAPL",
    "algorithm": "LSTM",
    "days_ahead": 7
}
```

#### GET /company_info?symbol=SYMBOL&info_type=TYPE
Get comprehensive company information.

#### GET /algorithms
Get list of available prediction algorithms.

## Stock Categorization

- **US Stocks**: Common symbols (AAPL, GOOGL, etc.) and symbols ending with .US
- **Indian Stocks**: Symbols ending with .NS or .BO
- **Others**: Everything else

## Data Storage

Data is automatically saved to:
- `data/latest/us_stocks/latest_prices.csv`
- `data/latest/ind_stocks/latest_prices.csv`
- `data/latest/others_stocks/latest_prices.csv`

Dynamic indexes are maintained in:
- `data/index_us_stocks_dynamic.csv`
- `data/index_ind_stocks_dynamic.csv`
- `data/index_others_stocks_dynamic.csv`

## Development Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Modular architecture setup
- [x] Live data fetching
- [x] Shared utilities
- [x] Basic API endpoints

### Phase 2: Prediction Algorithms 🔄
- [ ] Implement Random Forest predictor
- [ ] Implement LSTM neural network
- [ ] Implement ARIMA model
- [ ] Implement Linear Regression
- [ ] Implement SVM predictor
- [ ] Implement Gradient Boosting
- [ ] Implement Neural Networks
- [ ] Implement Technical Analysis
- [ ] Implement Sentiment Analysis
- [ ] Implement Ensemble methods

### Phase 3: Company Information 🔄
- [ ] Company fundamentals fetcher
- [ ] Company metadata management
- [ ] Financial statements integration
- [ ] News and events fetcher

### Phase 4: Advanced Features 🔄
- [ ] Algorithm performance evaluation
- [ ] Dynamic algorithm selection
- [ ] Historical data integration
- [ ] Advanced caching strategies
- [ ] Real-time prediction updates

## Error Handling

- **Invalid Symbol**: Returns 404 with error message
- **API Failures**: Tries fallback APIs, returns 500 if all fail
- **Network Issues**: Proper timeout handling and error messages
- **Data Validation**: Input validation with clear error messages

## Port Management

- Default port: 5000
- Auto-selects free port if default is busy
- Configurable via PORT environment variable

## Contributing

When adding new features:

1. **Follow the modular structure**: Place code in appropriate directories
2. **Use shared utilities**: Leverage common functions from `shared/`
3. **Add proper error handling**: Use custom exception classes
4. **Update documentation**: Keep README and docstrings current
5. **Test thoroughly**: Ensure backward compatibility

## Testing

### Running Tests
```bash
# Run all tests
py -m pytest tests/ -v

# Run specific test file
py -m pytest tests/test_api.py -v

# Run data fetching tests
py -m pytest data-fetching/test/ -v
```

### Test Coverage
- ✅ Live price fetching (US and Indian stocks)
- ✅ Error handling and fallback APIs
- ✅ Caching functionality
- ✅ Rate limiting
- ✅ Data format validation
- ✅ Stock categorization

## Migration Notes

- **Reorganized Structure**: Files moved to appropriate directories (scripts/, tests/, docs/)
- **Import Fixes**: Fixed hyphenated directory imports using sys.path
- **Package Structure**: Added __init__.py files for proper Python packages
- **Configuration**: Centralized in `shared/Config` class
- **Legacy Cleanup**: Removed duplicate app.py file