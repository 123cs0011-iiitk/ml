# Stock Price Prediction

A comprehensive full-stack web application for real-time stock price analysis and prediction using machine learning algorithms. The system provides live stock data fetching, historical analysis, and AI-powered price predictions for both US and Indian markets.

## ✨ Features

### 🔴 **Currently Working**
- **Real-time Stock Prices**: Live price fetching for US and Indian stocks
- **Upstox API v2 Integration**: 90%+ success rate for Indian stocks with OAuth2
- **Multi-API Fallback System**: Robust data fetching with multiple fallback sources
- **Interactive Dashboard**: Modern React frontend with TypeScript
- **Historical Data**: 5-year historical data (2020-2025) with chart visualization
- **Chart Visualization**: Interactive charts for both US and Indian stocks (fixed currency issues)
- **Currency Conversion**: Real-time USD/INR conversion with live exchange rates
- **Stock Search**: Intelligent search across 1000+ stocks (500 US + 500 Indian)
- **Data Storage**: CSV-based storage with dynamic indexing
- **Caching System**: Smart caching for optimal performance
- **Automatic Token Refresh**: OAuth2 token management for Upstox
- **ISIN Management**: 500+ correct ISIN mappings for Indian stocks
- **Currency Data Integrity**: Fixed NaN currency issues in CSV files and JSON responses

### 🟡 **In Development**
- **Machine Learning Predictions**: KNN algorithm for price prediction
- **Advanced Analytics**: Technical indicators and trend analysis
- **Portfolio Management**: Watchlist and portfolio tracking

## 🛠️ Technology Stack

### Backend
- **Flask** - Python web framework
- **yfinance** - Primary stock data source
- **Upstox API** - Indian stock data (with OAuth2)
- **Finnhub API** - US stock fallback
- **Pandas** - Data manipulation
- **CSV Storage** - Lightweight file-based storage

### Frontend
- **React 18** - Frontend library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Radix UI** - Component library
- **Recharts** - Data visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### 1. Clone Repository
```bash
git clone <repository-url>
cd ml
```

### 2. Backend Setup

**Windows:**
```cmd
cd backend
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
py main.py
```

**Linux/macOS:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Access Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000

## 📊 Data Sources

### US Stocks
- **Primary**: yfinance (no rate limits)
- **Fallback**: Finnhub API (60 calls/min)
- **Storage**: 500+ stocks with complete metadata

### Indian Stocks
- **Primary**: Upstox API (real-time NSE data)
- **Fallback**: NSEPython → yfinance → NSELib → Permanent data
- **Storage**: 500+ stocks with complete metadata

### Currency Conversion
- **Primary**: forex-python
- **Fallback**: exchangerate-api.com → Yahoo Finance → Hardcoded rate
- **Caching**: 1-hour cache for exchange rates

## 🏗️ Architecture

### Data Storage
```
data/
├── index_us_stocks_dynamic.csv      # Master US stock index (503 stocks)
├── index_ind_stocks_dynamic.csv     # Master Indian stock index (500 stocks)
├── latest/                          # Recent data (2025+)
│   ├── us_stocks/
│   └── ind_stocks/
└── past/                            # Historical data (2020-2024)
    ├── us_stocks/
    └── ind_stocks/

permanent/ (READ-ONLY)               # Backup data source
├── us_stocks/
└── ind_stocks/
```

### API Endpoints
- `GET /health` - Health check
- `GET /live_price?symbol=SYMBOL` - Live stock price
- `GET /search?q=QUERY` - Stock search
- `GET /historical?symbol=SYMBOL&period=PERIOD` - Historical data
- `GET /stock_info?symbol=SYMBOL` - Stock metadata

## 🔧 Configuration

### Environment Variables
Create `backend/.env`:
```env
# Optional API keys
FINNHUB_API_KEY=your_finnhub_key
UPSTOX_API_KEY=your_upstox_key
UPSTOX_ACCESS_TOKEN=your_upstox_token

# Server settings
PORT=5000
CACHE_DURATION=60
```

### Upstox API Setup (Indian Stocks)
1. Visit [Upstox Developer Portal](https://upstox.com/developer/)
2. Create app and get credentials
3. Run: `python backend/scripts/setup_upstox_oauth.py`
4. Follow OAuth flow to get tokens

## 📈 Usage

### Stock Search
- Search by symbol (e.g., "AAPL", "TCS")
- Search by company name (e.g., "Apple", "Reliance")
- Results show symbol, company name, and current price

### Price Analysis
- View live prices with real-time updates
- Toggle between USD and INR currencies
- See price changes and percentage changes
- View historical charts (1 year, 5 years)

### Data Management
- Automatic data fetching and storage
- Smart caching for performance
- Fallback systems for reliability
- CSV-based storage for portability

## 🧪 Testing

### Backend Tests

**Windows:**
```cmd
cd backend
venv\Scripts\activate
py -m pytest tests/ -v
```

**Linux/macOS:**
```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Manual Testing

**Windows (PowerShell):**
```powershell
# Test API endpoints
Invoke-RestMethod http://localhost:5000/health
Invoke-RestMethod "http://localhost:5000/live_price?symbol=AAPL"
Invoke-RestMethod "http://localhost:5000/search?q=apple"
```

**Linux/macOS:**
```bash
# Test API endpoints
curl http://localhost:5000/health
curl "http://localhost:5000/live_price?symbol=AAPL"
curl "http://localhost:5000/search?q=apple"
```

## 🔍 Troubleshooting

### Common Issues

**Backend won't start:**
- Check if port 5000 is available
- Ensure virtual environment is activated:
  - **Windows**: `venv\Scripts\activate`
  - **Linux/macOS**: `source venv/bin/activate`
- Verify Python version (3.8+)

**Frontend can't connect:**
- Ensure backend is running on port 5000
- Check CORS configuration
- Verify API endpoints

**No stock data:**
- Check internet connection
- Verify API keys (if using paid services)
- Try different stock symbols

**Indian stocks not working:**
- Set up Upstox API credentials
- Check OAuth token validity
- System will fallback to permanent data

## 📚 Documentation

- [Documentation Index](documentation/README.md) - Complete documentation overview
- [Backend API Documentation](backend/README.md)
- [Data Fetching Guide](backend/data-fetching/README.md)
- [Project Status](documentation/PROJECT_STATUS_FINAL.md) - Current system status
- [Changelog](documentation/CHANGELOG.md) - Version history and updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend
- Write tests for new features
- Update documentation
- Maintain backward compatibility

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as sole investment advice. Always consult with financial advisors and conduct thorough research before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stock data provided by yfinance, Upstox, and Finnhub
- Frontend built with React and Radix UI
- Data visualization powered by Recharts
- Currency conversion by forex-python

---

**Ready to start?** Follow the Quick Start guide above to get the application running in minutes!
