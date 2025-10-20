# Stock Price Prediction System

A comprehensive full-stack web application for real-time stock price analysis and prediction using 16 machine learning algorithms. Supports both US and Indian markets with live data fetching and AI-powered predictions.

## ✨ Key Features

- **16 ML Algorithms**: Linear Regression, Random Forest, KNN, SVM, ANN, CNN, ARIMA, K-Means, DBSCAN, PCA, SVD, t-SNE, and more
- **Real-time Data**: Live price fetching for US and Indian stocks
- **Interactive Dashboard**: Modern React frontend with TypeScript
- **Historical Analysis**: 5-year historical data with chart visualization
- **Currency Support**: Real-time USD/INR conversion
- **Stock Search**: 1000+ stocks (500 US + 500 Indian)
- **OHLC Data**: Uses Open, High, Low, Close data (volume not used)

## 🛠️ Technology Stack

**Backend**: Flask, Python, yfinance, Upstox API, Finnhub API, Pandas, Scikit-learn, TensorFlow, Keras  
**Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Radix UI, Recharts  
**ML**: 16 algorithms including Linear Regression, Random Forest, KNN, SVM, ANN, CNN, ARIMA, Clustering, PCA, SVD, t-SNE

## 🚀 Quick Start

### Prerequisites
- Python 3.8+, Node.js 16+, Git

### Setup
```bash
# Clone and setup backend
cd backend
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows
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

**US Stocks**: yfinance (primary), Finnhub API (fallback)  
**Indian Stocks**: Upstox API (primary), NSEPython → yfinance (fallback)  
**Currency**: forex-python with real-time USD/INR conversion

## 📚 Documentation

- **[Complete Documentation](documentation/README.md)** - All technical documentation
- **[API Usage Guide](documentation/API_USAGE.md)** - Detailed API examples
- **[Implementation Notes](documentation/IMPLEMENTATION_NOTES.md)** - Technical details
- **[Project Status](documentation/PROJECT_STATUS_FINAL.md)** - Current system status

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as sole investment advice.

---

**Ready to start?** Follow the Quick Start guide above!
