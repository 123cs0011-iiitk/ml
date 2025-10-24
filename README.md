# Stock Price Prediction System

A comprehensive full-stack web application for real-time stock price analysis and prediction using 7 machine learning algorithms, supporting both US and Indian markets with live data fetching, historical analysis, and interactive visualization.

## ✨ Key Features

- **7 ML Algorithms**: Linear Regression, Decision Tree, Random Forest, SVM (basic models) + KNN, ARIMA, Autoencoder (advanced models)
- **Real-time Data**: US stocks via Finnhub API, Indian stocks via Upstox API with permanent storage fallback
- **Modern Dashboard**: React 18 + TypeScript + Tailwind CSS with interactive Recharts for 5-year historical analysis
- **1000+ Stocks**: 501 US + 500 Indian stocks with OHLC data and 38 technical indicators
- **Currency Support**: Real-time USD/INR conversion via forex-python
- **Smart Training**: Percentage-based predictions with proper price conversion and confidence scoring
- **Standalone Trainers**: Independent training scripts for each model with progress tracking

## 🛠️ Technology Stack

**Backend**: Flask 2.3.3, Python 3.8+, TensorFlow 2.20, scikit-learn 1.5.2, statsmodels 0.14.4, pandas, numpy  
**Frontend**: React 18.3.1, TypeScript, Vite 6.4.0, Tailwind CSS, Radix UI, Recharts 2.15.2  
**APIs**: Finnhub (US stocks), Upstox (Indian stocks), yfinance (historical data)

## 📋 Data Structure

**Indian Stocks** (500): Include ISIN codes (12-char format: INExxxxxxxx) required for Upstox API
- ✅ **100% Verified**: All 500 ISINs tested and validated with live Upstox data
- Format: 12 characters starting with "INE" (e.g., INE009A01021 for Infosys)
- Location: `permanent/ind_stocks/`

**US Stocks** (501): Use ticker symbols only - **NO ISINs**
- Identified by exchange (NYSE/NASDAQ) and ticker symbol only
- ISINs not required for Finnhub API
- Location: `permanent/us_stocks/`

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

## 🔌 Offline Mode (No API Keys Required)

The system works **completely offline** without any API keys using the permanent directory fallback.

### How It Works
The system uses a three-tier data loading strategy:
1. **Primary**: `data/past/` (updated historical data, if available)
2. **Fallback**: `permanent/` (pre-loaded 1001 stocks with 5-year history 2020-2024)
3. **Graceful degradation**: Clear error messages if no data available

### What Works Offline
- ✅ **Stock Info Cards**: 500 Indian + 501 US stocks from permanent directory
- ✅ **Historical Charts**: Complete 5-year OHLCV data (2020-2024)
- ✅ **ML Predictions**: All trained models work with permanent data
- ✅ **Search**: Full-text search across 1001 stocks
- ✅ **Technical Indicators**: 38 indicators calculated from historical data
- ❌ **Live Prices**: Requires API keys (Finnhub for US, Upstox for India)

### Offline Setup (Fresh Clone)
```bash
# 1. Clone repository
git clone <repo-url>
cd ml

# 2. Backend setup
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create empty .env file (REQUIRED - even for offline mode)
# Windows
type nul > .env

# Mac/Linux
touch .env

# Leave the .env file empty - no API keys needed for offline mode

# 5. Start backend (uses permanent directory automatically)
python main.py

# 6. Frontend setup (new terminal)
cd frontend
npm install
npm run dev

# 7. Access application
# Frontend: http://localhost:5173
# Backend: http://localhost:5000
```

**Important**: The `.env` file must exist (even if empty) for the system to start properly.

### Transitioning to Live Mode
When ready for live data, add API keys to `backend/.env`:
```bash
# US Stocks (Finnhub)
FINNHUB_API_KEY=your_finnhub_key_here

# Indian Stocks (Upstox - requires OAuth)
UPSTOX_API_KEY=your_upstox_key_here
UPSTOX_CLIENT_ID=your_client_id_here
UPSTOX_CLIENT_SECRET=your_client_secret_here
```

**Detailed Setup Guides**:
- [Upstox Integration](documentation/UPSTOX_INTEGRATION.md) - Indian market API setup
- [Backend API](backend/README.md) - Full API documentation
- [Model Training](documentation/MODEL_TRAINING.md) - Training ML models

### Data Update Strategy
- **Without APIs**: System uses static permanent directory (2020-2024 data)
- **With APIs**: New data goes to `data/latest/`, permanent remains as fallback
- **Best Practice**: Keep permanent directory intact as safety net

## 📊 System Status

**Check**: `python status.py` | **Data**: Finnhub (US) + Upstox (India) → Permanent fallback | **Currency**: Real-time USD/INR

**✅ Working**: Data fetching, Historical (5yr), Search (1000+ stocks), Currency, Dashboard, 38 technical indicators

**ML Models** (Oct 24/2025): 
- **Training Progress**: 4/7 models trained
  - Linear Regression (R²=-0.002) ✅
  - Decision Tree (R²=0.001) ✅
  - Random Forest (R²=0.024) ✅ Best performer
  - SVM (R²=-0.0055) ✅ Working via UI
- **Next**: KNN (K-Nearest Neighbors)
- **Remaining**: ARIMA, Autoencoder
- **Architecture**: Standalone trainers in `basic_models/` and `advanced_models/`
- **Prediction Method**: Percentage change predictions with price conversion
- **Features**: 38 technical indicators from OHLC data (volume excluded from calculations)

**Recent Updates** (Oct 24/2025):
- ✅ Offline mode with permanent directory fallback (Mac/fresh clone support)
- ✅ Info & prediction card synchronization with visual indicators
- ✅ Data source tracking (live_api vs stored_data)
- ✅ SVM model integration fix (explicit model selection)

**Docs**: [Backend API](backend/README.md) | [Documentation Hub](documentation/README.md) | [Upstox Setup](documentation/UPSTOX_INTEGRATION.md)

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as sole investment advice.

---

**Ready to start?** Follow the Quick Start guide above!
