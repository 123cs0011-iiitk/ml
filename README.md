# Stock Price Prediction System

A comprehensive full-stack web application for real-time stock price analysis and prediction using 9 machine learning algorithms, supporting both US and Indian markets with live data fetching, historical analysis, and interactive visualization.

## ✨ Key Features

- **9 ML Algorithms**: Linear Regression, Random Forest, Decision Tree, KNN, SVM, ANN, CNN, ARIMA, Autoencoders
- **Real-time Data**: US stocks via Finnhub API, Indian stocks via Upstox API with permanent storage fallback
- **Modern Dashboard**: React 18 + TypeScript + Tailwind CSS with interactive Recharts for 5-year historical analysis
- **1000+ Stocks**: 500 US + 500 Indian stocks with OHLC data and 50+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- **Currency Support**: Real-time USD/INR conversion via forex-python

## 🛠️ Technology Stack

**Backend**: Flask 2.3.3, Python 3.8+, TensorFlow 2.20, scikit-learn 1.5.2, statsmodels 0.14.4, pandas, numpy  
**Frontend**: React 18.3.1, TypeScript, Vite 6.4.0, Tailwind CSS, Radix UI, Recharts 2.15.2  
**APIs**: Finnhub (US stocks), Upstox (Indian stocks), yfinance (historical data)

## 📋 Data Structure

**Indian Stocks** (500): Include ISIN codes (12-char format: INExxxxxxxx) required for Upstox API
- ✅ **100% Verified**: All 500 ISINs tested and validated with live Upstox data
- Format: 12 characters starting with "INE" (e.g., INE009A01021 for Infosys)
- Location: `permanent/ind_stocks/index_ind_stocks.csv`

**US Stocks** (500): Use ticker symbols only - **NO ISINs**
- Identified by exchange (NYSE/NASDAQ) and ticker symbol only
- ISINs not required for Finnhub API
- Location: `permanent/us_stocks/index_us_stocks.csv`

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

## 📊 System Status

**Check**: `python status.py` | **Data**: Finnhub (US) + Upstox (India) → Permanent fallback | **Currency**: Real-time USD/INR

**✅ Working**: Data fetching, Historical (5yr), Search (1000+ stocks), Currency, Dashboard, 50+ indicators

**ML Models** (Oct 22/2025): 🔄 All 9 models ready for training | 0/9 trained | Models: Linear Reg, Random Forest, Decision Tree, KNN, SVM, ANN, CNN, ARIMA, Autoencoder

**Docs**: [Backend API](backend/README.md) | [Documentation Hub](documentation/README.md) | [Upstox Setup](documentation/UPSTOX_INTEGRATION.md)

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as sole investment advice.

---

**Ready to start?** Follow the Quick Start guide above!
