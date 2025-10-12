# Live Stock Price API

A Flask-based API for fetching live stock prices with multiple data source fallbacks.

## Features

- **Multi-API Fallback**: yfinance → Finnhub → Alpha Vantage
- **Automatic Stock Categorization**: US stocks, Indian stocks, Others
- **Dynamic CSV Storage**: Stores latest prices with automatic index updates
- **CORS Enabled**: Ready for frontend integration
- **Error Handling**: Comprehensive error handling and logging

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
   Create a `.env` file in the backend directory with:
   ```
   # Finnhub API Key (Optional - for fallback when Yahoo Finance fails)
   # Get your free API key at: https://finnhub.io/register
   FINNHUB_API_KEY=your_finnhub_api_key_here

   # Alpha Vantage API Key (Optional - for fallback when Yahoo Finance fails)
   # Get your free API key at: https://www.alphavantage.co/support/#api-key
   ALPHAVANTAGE_API_KEY=your_alphavantage_api_key_here

   # Server Configuration (Optional)
   PORT=5000
   ```

4. **Run the Server**:
   ```bash
   py run_server.py
   ```

## API Endpoints

### GET /live_price?symbol=SYMBOL
Fetch live stock price for a symbol.

**Example**: `GET /live_price?symbol=AAPL`

**Response**:
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

### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "service": "Live Stock Price API",
  "timestamp": "2024-01-15T10:30:00"
}
```

### GET /latest_prices?category=CATEGORY
Get latest prices for a category (us_stocks, ind_stocks, others_stocks).

### GET /symbols?category=CATEGORY
Get all available symbols by category.

### GET /search?q=QUERY
Search for stocks by symbol or company name.

**Example**: `GET /search?q=Apple`

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc."
    }
  ]
}
```

## Stock Categorization

- **US Stocks**: Common symbols (AAPL, GOOGL, etc.) and symbols ending with .US
- **Indian Stocks**: Symbols ending with .NS or .BO
- **Others**: Everything else

## CSV Storage

Data is automatically saved to:
- `data/latest/us_stocks/latest_prices.csv`
- `data/latest/ind_stocks/latest_prices.csv`
- `data/latest/others_stocks/latest_prices.csv`

Dynamic indexes are maintained in:
- `data/index_us_stocks_dynamic.csv`
- `data/index_ind_stocks_dynamic.csv`
- `data/index_others_stocks_dynamic.csv`

## Error Handling

- **Invalid Symbol**: Returns 404 with error message
- **API Failures**: Tries fallback APIs, returns 500 if all fail
- **Network Issues**: Proper timeout handling and error messages

## Port Management

- Default port: 5000
- Auto-selects free port if default is busy
- Configurable via PORT environment variable
