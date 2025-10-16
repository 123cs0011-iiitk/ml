# Stock Price Insight Arena
# 📈 Stock Price Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.0+-blue.svg)](https://typescriptlang.org)
[![CSV](https://img.shields.io/badge/Data_Storage-CSV-orange.svg)](https://en.wikipedia.org/wiki/Comma-separated_values)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive full-stack web application that leverages machine learning algorithms to predict future stock prices based on historical data. The system provides real-time stock price fetching, data analysis, and interactive visualizations for informed investment decisions using CSV files for efficient data storage.

## ✨ Features

- **Real-time Stock Data**: Fetch live stock prices and historical data from multiple sources
- **CSV Data Storage**: Lightweight file-based storage for stock price data and historical records
- **Machine Learning Predictions**: Multiple ML algorithms including LSTM, Random Forest, and Linear Regression
- **Interactive Dashboard**: React-based frontend with TypeScript for type safety and better UX
- **Data Visualization**: Charts and graphs showing price trends and prediction accuracy
- **RESTful API**: Well-documented API endpoints for data access and predictions
- **File-based Storage**: CSV files for efficient data management and easy data portability
- **Data Processing**: Advanced preprocessing and feature engineering capabilities
- **Model Evaluation**: Comprehensive model performance metrics and backtesting

## 🛠️ Technology Stack

### Backend (Modular Architecture)
- **Flask** - Python web framework for API development
- **Modular Design** - Organized into specialized modules (live-data, algorithms, prediction, etc.)
- **CSV Files** - Lightweight data storage for stock prices and historical records
- **Pandas** - Data manipulation and CSV file operations
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms (Random Forest, SVM, etc.)
- **TensorFlow/Keras** - Deep learning models (LSTM, Neural Networks)
- **Matplotlib/Seaborn** - Data visualization
- **yfinance** - Real-time stock data fetching
- **Multiple API Fallbacks** - Finnhub, Alpha Vantage for reliability

### Frontend
- **React** - Frontend library for building user interfaces
- **TypeScript** - Type-safe JavaScript for better development experience
- **Axios** - HTTP client for API communication
- **Chart.js/D3.js** - Interactive charts and visualizations
- **Material-UI** - Component library for consistent design

### Data Storage Architecture

#### Master Index Files
- **`data/index_us_stocks_dynamic.csv`** - Master index for all US stocks (500+ stocks)
- **`data/index_ind_stocks_dynamic.csv`** - Master index for all Indian stocks (500+ stocks)
- These are the ONLY index files and single source of truth for stock metadata
- Automatically updated when new stocks are discovered
- Maintained in alphabetical order by symbol

#### Data Directory (Read/Write)
- **`data/latest/`** - Recent stock price data (2025+)
- **`data/past/`** - Historical stock price data (2020-2024)
- Backend actively reads and writes to this directory

#### Permanent Directory (Read-Only)
- **`permanent/`** - Manually curated historical data (2020-2024)
- Used ONLY as fallback when APIs fail
- Backend NEVER modifies this directory
- Contains 500 US stocks and 500 Indian stocks with complete historical data

#### Index File Structure
```csv
symbol,company_name,sector,market_cap,headquarters,exchange,currency
AAPL,Apple,Technology,,"Cupertino, California",NASDAQ,USD
RELIANCE,Reliance Industries,Energy,,"Mumbai, India",NSE,INR
```

### Data Storage
- **CSV Files** - Human-readable data storage format
- **JSON** - Configuration and metadata storage
- **File System** - Organized directory structure for data management

## 🏗️ Backend Architecture

The backend has been restructured with a modular architecture for better organization and maintainability:

```
backend/
├── main.py                 # Central coordinator and API entry point
├── app.py                  # Legacy Flask app (backward compatibility)
├── live-data/             # Live market data fetching
│   ├── live_data_manager.py
│   └── live_fetcher.py    # Multi-API fallback system
├── company-info/          # Company information management
│   └── company_info_manager.py  # [FUTURE] Fundamentals, metadata
├── algorithms/            # Stock prediction algorithms
│   └── prediction_algorithms.py  # [FUTURE] 10+ ML algorithms
├── prediction/           # Prediction orchestration
│   └── prediction_engine.py  # [FUTURE] Algorithm coordination
├── shared/               # Shared utilities and common code
│   └── utilities.py      # Configuration, logging, validation
└── requirements.txt
```

### Module Responsibilities

- **live-data/**: Real-time stock price fetching with multiple API fallbacks
- **company-info/**: Company fundamentals, metadata, and financial data
- **algorithms/**: 10+ prediction algorithms (Random Forest, LSTM, ARIMA, etc.)
- **prediction/**: Algorithm selection, data preprocessing, and result formatting
- **shared/**: Common utilities, configuration, and data structures

## 📋 Prerequisites

Before running this project, make sure you have the following installed:

- Python 3.8 or higher
- Node.js 16.0 or higher
- Git

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/123cs0011-iiitk/ml.git
cd ml
```

### 2. Backend Setup

#### Create Virtual Environment
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Environment Configuration
Create a `.env` file in the backend directory:
```text
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
STOCK_API_KEY=your-stock-api-key
DATA_DIRECTORY=../data/stock_data
```

#### Initialize Data Directory Structure
```bash
python -c "from app.utils.data_manager import initialize_data_structure; initialize_data_structure()"
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
# or, to build for production:
# npm run build
```


#### Environment Configuration
Create a `.env` file in the frontend directory:
```text
REACT_APP_API_BASE_URL=http://localhost:5000/api
REACT_APP_ENVIRONMENT=development
```

## 🏃‍♂️ Running the Application

### Start Backend Server
```bash
cd backend
flask run
# Server will start on http://localhost:5000
```

### Start Frontend Development Server
```bash
cd frontend
npm start
# Application will open on http://localhost:3000
```

## 📁 Project Structure

```
stock-price-prediction/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── stock_data.py
│   │   │   └── prediction.py
│   │   ├── routes/
│   │   │   ├── api.py
│   │   │   └── stock_routes.py
│   │   ├── services/
│   │   │   ├── data_fetcher.py
│   │   │   ├── ml_models.py
│   │   │   └── predictor.py
│   │   └── utils/
│   │       ├── csv_manager.py
│   │       ├── data_processor.py
│   │       └── validators.py
│   ├── requirements.txt
│   ├── app.py
│   └── config.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── StockChart.tsx
│   │   │   └── PredictionPanel.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── types/
│   │   │   └── stock.ts
│   │   ├── utils/
│   │   │   └── formatters.ts
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── package.json
│   └── tsconfig.json
│
├── data/
│   ├── stock_data/
│   │   ├── daily/
│   │   │   ├── AAPL.csv
│   │   │   ├── GOOGL.csv
│   │   │   └── MSFT.csv
│   │   ├── predictions/
│   │   │   └── prediction_results.csv
│   │   └── metadata/
│   │       └── stock_info.json
│   └── sample_data/
│
├── ml_models/
│   ├── trained_models/
│   ├── lstm_model.py
│   ├── random_forest_model.py
│   └── model_evaluator.py
│
├── docs/
│   └── api_documentation.md
│
└── README.md
```

## 📊 Machine Learning Models

### Supported Algorithms

1. **LSTM (Long Short-Term Memory)**
   - Best for sequential time series prediction
   - Captures long-term dependencies in stock price movements

2. **Random Forest**
   - Ensemble method reducing overfitting
   - Good for handling non-linear relationships

3. **Linear Regression**
   - Baseline model for comparison
   - Fast training and prediction

### Model Training
```bash
cd backend
python -c "from app.services.ml_models import train_models; train_models()"
```

## 📄 Data Storage Format

### Stock Price CSV Structure
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,150.00,155.50,149.75,154.25,1000000
2024-01-02,154.25,158.00,153.50,157.75,1200000
2024-01-03,157.75,159.25,156.00,158.50,950000
```

### Prediction Results CSV Structure
```csv
Symbol,Date,Actual_Price,Predicted_Price,Model,Accuracy
AAPL,2024-01-04,160.00,159.25,LSTM,0.995
AAPL,2024-01-04,160.00,158.75,Random_Forest,0.992
AAPL,2024-01-04,160.00,161.50,Linear_Regression,0.990
```

## 🔌 API Endpoints

### Stock Data Endpoints
```
GET /api/stocks                 # Get all tracked stocks
GET /api/stocks/{symbol}        # Get specific stock data from CSV
POST /api/stocks                # Add new stock to tracking
DELETE /api/stocks/{symbol}     # Remove stock CSV file
```

### Data Management Endpoints
```
GET /api/data/historical/{symbol}  # Read historical data from CSV
POST /api/data/update              # Update CSV files with new data
GET /api/data/stats                # Get CSV file statistics
POST /api/data/export              # Export data in various formats
```

### Prediction Endpoints
```
GET /api/predict/{symbol}       # Get price prediction for stock
POST /api/predict/batch         # Batch prediction for multiple stocks
GET /api/models/performance     # Get model performance metrics
```

## 💻 Usage

### 1. Adding Stocks for Tracking
Navigate to the dashboard and use the stock search feature to add stocks. The system will create CSV files for each stock automatically.

### 2. Data Management
- **Automatic CSV Creation**: New stock CSV files are created when adding stocks to tracking
- **Data Updates**: Historical data is appended to existing CSV files
- **File Organization**: Each stock has its own CSV file in the `data/stock_data/daily/` directory

### 3. Viewing Predictions
Select a stock from your watchlist to view:
- Historical price charts loaded from CSV files
- ML model predictions
- Accuracy metrics
- Confidence intervals

## 🔧 Configuration

### CSV File Management
```python
# Example CSV operations
import pandas as pd

# Load stock data
df = pd.read_csv('data/stock_data/daily/AAPL.csv')

# Append new data
new_data = pd.DataFrame({
    'Date': ['2024-01-04'],
    'Open': [158.50],
    'High': [162.00],
    'Low': [157.25],
    'Close': [161.75],
    'Volume': [1100000]
})
df = pd.concat([df, new_data], ignore_index=True)
df.to_csv('data/stock_data/daily/AAPL.csv', index=False)
```

### Data Directory Structure
The system automatically creates and maintains the following structure:
```
data/
├── stock_data/
│   ├── daily/          # Daily stock price CSV files
│   ├── predictions/    # Model prediction results
│   └── metadata/       # Stock information and configurations
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Data Integrity Tests
```bash
cd backend
python -c "from app.utils.csv_manager import validate_csv_files; validate_csv_files()"
```

## 🚧 Troubleshooting

### Common Issues

#### CSV File Corruption
- Check file permissions in data directory
- Validate CSV format using pandas
- Restore from backup if available

#### Missing Data Files
- Run data initialization: `python -c "from app.utils.data_manager import initialize_data_structure; initialize_data_structure()"`
- Check if stock symbol CSV files exist

#### API Key Issues
- Ensure stock API key is valid and has sufficient quota
- Check API rate limits

#### Model Training Fails
- Verify sufficient historical data in CSV files
- Check CSV file format and data quality

#### Frontend Build Errors
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`

## 🔮 Future Enhancements

- Real-time WebSocket connections for live price updates
- Data compression for large CSV files
- Advanced technical indicators integration
- Sentiment analysis using news data
- Portfolio optimization features
- Mobile app development
- Advanced charting capabilities
- User authentication and personalized dashboards
- Database migration option for scaling
- Cloud storage integration for CSV files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint and Prettier for TypeScript/React code
- Write tests for new features
- Update documentation for API changes
- Ensure CSV file integrity in data operations

## ⚠️ Known Limitations

- CSV files may become large with extensive historical data
- No built-in data backup mechanism (manual backup recommended)
- Concurrent access to CSV files may cause data conflicts
- Stock market predictions are inherently uncertain and should not be used as sole investment advice
- API rate limits may affect real-time data fetching
- Model performance varies significantly across different market conditions

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation in the `docs/` directory

## 🙏 Acknowledgments

- Stock data provided by various financial APIs
- Machine learning libraries: scikit-learn, TensorFlow
- Frontend libraries: React, TypeScript community
- Data management: Pandas community for CSV operations

---

**Disclaimer**: This application is for educational and research purposes only. Stock market investments carry risk, and past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.

