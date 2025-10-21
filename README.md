# Stock Price Prediction System

A full-stack web application for real-time stock price analysis and prediction using 9 machine learning algorithms, supporting both US and Indian markets with live data fetching and interactive visualization.

## ✨ Key Features

- **9 ML Algorithms**: Linear Regression, Random Forest, KNN, SVM, ANN, CNN, ARIMA, Decision Tree, Autoencoders
- **Real-time Data**: Live price fetching for US stocks via Finnhub API and Indian stocks via Upstox API
- **Interactive Dashboard**: Modern React frontend with TypeScript and Tailwind CSS
- **Historical Analysis**: 5-year historical data with interactive charts and currency support (USD/INR)
- **Stock Search**: 1000+ stocks (500 US + 500 Indian) using OHLC data for analysis

## 🛠️ Technology Stack

**Backend**: Flask, Python, Finnhub API, Upstox API, yfinance, Pandas, Scikit-learn, TensorFlow, Keras  
**Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Radix UI, Recharts  
**ML**: 9 algorithms (Linear Regression, Random Forest, KNN, SVM, ANN, CNN, ARIMA, Decision Tree, Autoencoders)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+, Node.js 16+, Git

### Setup
```bash
# Clone and setup backend
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py

# Setup frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Access
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000

## 📊 Data Sources

**US Stocks**: Finnhub API (primary), Permanent directory (fallback)  
**Indian Stocks**: Upstox API (primary), Permanent directory (fallback)  
**Currency**: forex-python with real-time USD/INR conversion

### Important: ISIN Requirements

**Indian Stocks (REQUIRED):**
- **ISINs are mandatory** for Upstox API integration
- All 500 Indian stocks have ISINs (100% coverage)
- ISINs are stored in both permanent and dynamic index files
- Without ISINs, Upstox will return "wrong ISIN number" errors

**US Stocks (NOT Required):**
- Finnhub API uses ticker symbols, not ISINs
- ISINs are optional for US stocks
- System works perfectly without ISINs for US stocks

## 📚 Documentation

- **[Complete Documentation](documentation/README.md)** - All technical documentation
- **[API Usage Guide](documentation/API_USAGE.md)** - Detailed API examples
- **[Upstox Integration](documentation/UPSTOX_INTEGRATION_FINAL.md)** - Indian market integration

## 📊 Current System Status

**✅ Working**: Real-time data fetching, historical charts, stock search, currency conversion, interactive dashboard

**⚠️ ML Models** (Check with `python status.py`):
- **Working**: Random Forest (R²=0.994), Decision Tree (R²=0.85)
- **Poor**: SVM, KNN, ANN (negative R² scores)
- **Failed**: Linear Regression, CNN, ARIMA, Autoencoder

**ML Predictions**: Currently unreliable due to model performance issues

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as sole investment advice.

---

**Ready to start?** Follow the Quick Start guide above!
