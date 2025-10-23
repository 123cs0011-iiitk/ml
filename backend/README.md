# Backend API Documentation

ML backend with 9 algorithms, automated training, RESTful API, and real-time data for 1000+ stocks.

## 🎯 Core Features

- **9 ML Algorithms**: Linear Regression, Random Forest, Decision Tree, KNN, SVM, ANN, CNN, ARIMA, Autoencoders with automated training
- **Technical Indicators**: 50+ features from OHLC data (SMA, EMA, MACD, RSI, Bollinger Bands, ATR)
- **Multi-horizon Forecasting**: 1d/1w/1m/1y/5y predictions via Flask 2.3.3 API with CORS
- **Real-time Data**: Finnhub (US) + Upstox (India) APIs with permanent storage fallback
- **Status Tracking**: `python status.py` (table/JSON/simple formats) with USD/INR currency conversion

## Quick Start

```bash
# Check model status
python status.py

# Train all models (first time setup)
python backend/training/train_full_dataset.py

# Start the Flask server
python main.py

# Test the API
curl "http://localhost:5000/health"
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1d"
```

## Training System

Unified training system combining data from all 1000+ stocks with single-pass training.

```bash
python status.py [--json|--simple]                          # Check model status
python backend/training/train_full_dataset.py               # Train all models
python backend/training/train_full_dataset.py --model <name> # Train specific model
python backend/training/generate_predictions.py --max-stocks 10 # Generate predictions
```

**Process**: Data loading (1000+ stocks) → Feature engineering (50+ indicators) → Single-pass training → Validation → Save to `backend/models/{model_name}/` subdirectories with status tracking in `model_status.json`

### Model Status (Oct 22, 2025)

Run `python status.py` for real-time status.

| Model | Status | Stocks | R² | Notes |
|-------|--------|--------|----|----|
| Linear Regression | 🔄 | 0 | N/A | Ready for training |
| Random Forest | 🔄 | 0 | N/A | Ready for training |
| Decision Tree | 🔄 | 0 | N/A | Ready for training |
| KNN | 🔄 | 0 | N/A | Ready for training |
| SVM | 🔄 | 0 | N/A | Ready for training |
| ANN | 🔄 | 0 | N/A | Ready for training |
| CNN | 🔄 | 0 | N/A | Ready for training |
| ARIMA | 🔄 | 0 | N/A | Ready for training |
| Autoencoder | 🔄 | 0 | N/A | Ready for training |

**Summary**: 0/9 trained, 9/9 ready for training. All models are ready for fresh training with 37 features (no volume). Models will be saved in `backend/models/{model_name}/` subdirectories.

### Training Time & Storage Requirements

| Model | Training Time | Model Size | Memory Peak | Notes |
|-------|--------------|------------|-------------|-------|
| Linear Regression | 5-10 min | ~3 MB | 2-3 GB | Fastest, uses SGD |
| Decision Tree | 5-10 min | ~150 MB | 3-4 GB | Fast, single tree |
| Random Forest | 10-15 min | ~9.5 GB | 6-8 GB | Largest file, ensemble |
| KNN | 15-25 min | ~10 MB | 4-5 GB | Stores training data |
| SVM | 20-30 min | ~10 MB | 4-6 GB | Uses subsampling |
| Autoencoder | 40-60 min | ~1 MB (x3 files) | 4-6 GB | 3 files (.h5 + metadata) |
| ANN | 30-45 min | ~130 KB + ~128 KB | 4-6 GB | 2 files (.pkl + .h5) |
| CNN | 45-75 min | ~5 MB | 5-7 GB | Memory optimized |
| ARIMA | 90-180 min | ~170 MB | 3-5 GB | Slowest, per-stock models |

**Total Disk Space Required**: ~10-11 GB for all 9 models

**Notes**:
- Training times based on ~1,000 stocks with 5 years of data
- Times may vary ±20% based on hardware (CPU, RAM speed)
- Model sizes are for trained models on full dataset
- Deep learning models (ANN, CNN, Autoencoder) create multiple files

## 🧮 Feature Engineering (37 Features)

### Why 37 Features? Quality > Quantity

**Feature Reduction: 43 → 37**
- Removed 7 volume-based features (`volume_ma`, `volume_ratio`, `volume_lag_1-10`)
- Reason: Volume data is unavailable/unreliable for all 1,001 stocks

**ML Principle: The "Curse of Dimensionality"**
```
Few samples + Many features = Poor generalization
43 incomplete features → WORSE than 37 complete features
```

**Problems with Incomplete Features:**
- ❌ Overfitting: Memorizes training, fails on new data
- ❌ Noise: 16% incomplete features dilute 84% good features
- ❌ Instability: Different prediction quality per stock
- ❌ Speed: 14% slower training

**Benefits of 37 Complete Features:**
- ✅ 100% data completeness (vs 84%)
- ✅ Consistent quality across all 1,001 stocks
- ✅ 32,432 samples/feature (well above 10 minimum)
- ✅ Faster training, stable predictions

| Metric | 43 Features (Old) | 37 Features (New) |
|--------|-------------------|-------------------|
| Completeness | 84% | 100% |
| Consistency | Variable | Stable |
| Speed | Slower | 14% faster |

**Key Insight**: "More data beats better algorithms, but better data beats more data."

### Feature Categories (37 Total)
- 2 Basic price features (price_change, abs)
- 10 Moving averages (MA 5/10/20/50/200 + ratios)
- 1 Volatility
- 1 RSI (momentum)
- 2 Intraday ratios (HL, OC)
- 1 Price position
- 5 Lagged prices (1/2/3/5/10 days)
- 9 Rolling statistics (std/min/max for 5/10/20)
- 3 Time features (day/month/quarter)
- 3 Raw OHLC (open/high/low)

All calculated from 5-year historical OHLC data only.

## 🔌 API Endpoints

### Prediction Endpoints

| Endpoint | Method | Description | Query Parameters |
|----------|--------|-------------|------------------|
| `/api/predict` | GET | Get ML price prediction | `symbol`, `horizon` (1d/1w/1m/1y/5y), `model` (optional) |
| `/api/train` | POST | Train models for a symbol | Body: `{"symbol": "AAPL", "models": ["random_forest"]}` |
| `/api/models/<symbol>` | GET | List trained models | Path: `symbol` |

### Data Endpoints

| Endpoint | Method | Description | Query Parameters |
|----------|--------|-------------|------------------|
| `/live_price` | GET | Get current stock price | `symbol`, `category` (optional) |
| `/latest_prices` | GET | Get latest prices for all stocks | `category` (us_stocks/ind_stocks, optional) |
| `/historical` | GET | Get historical OHLC data | `symbol`, `period` (5d/1mo/3mo/6mo/1y/2y/5y/max) |
| `/search` | GET | Fuzzy search for stocks | `q` (query string), `limit` (optional, default 10) |
| `/symbols` | GET | Get all available symbols | `category` (optional) |

**Note:** All data endpoints return volume column for compatibility, but ML models never use volume data. Volume values are typically NaN or unreliable and are excluded from all calculations and predictions.

### Utility Endpoints

| Endpoint | Method | Description | Query Parameters |
|----------|--------|-------------|------------------|
| `/health` | GET | API health check | None |
| `/stock_info` | GET | Get stock metadata | `symbol` |
| `/convert_currency` | GET | Convert USD to INR or vice versa | `amount`, `from_currency`, `to_currency` |

## 📖 Usage Examples

### Prediction Endpoints

```bash
# Basic prediction (1 day horizon)
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1d"

# Specific model
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1w&model=random_forest"

# Response format
{
  "success": true,
  "prediction": 175.50,
  "current_price": 173.25,
  "model": "random_forest",
  "confidence": 0.95
}
```

### Training Models

```bash
# Train all models for a symbol (requires historical data)
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# Train specific models
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "models": ["random_forest", "decision_tree"]}'
```

### Data Fetching

```bash
# Get live price (real-time)
curl "http://localhost:5000/live_price?symbol=AAPL"

# Get historical data (OHLC + Volume)
curl "http://localhost:5000/historical?symbol=AAPL&period=1y"

# Search stocks (fuzzy search)
curl "http://localhost:5000/search?q=apple&limit=5"

# Get all US stock symbols
curl "http://localhost:5000/symbols?category=us_stocks"

# Get latest prices for all stocks
curl "http://localhost:5000/latest_prices?category=ind_stocks"
```

### Currency Conversion

```bash
# Convert USD to INR
curl "http://localhost:5000/convert_currency?amount=100&from_currency=USD&to_currency=INR"

# Response format
{
  "success": true,
  "amount": 100,
  "from_currency": "USD",
  "to_currency": "INR",
  "converted_amount": 8325.50,
  "rate": 83.255
}
```

### Standard Response Format

All endpoints return JSON in this format:

```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

Or on error:

```json
{
  "success": false,
  "error": "Error message here",
  "data": null
}
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the backend directory (or set environment variables):

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000
HOST=0.0.0.0

# Data Configuration
DATA_DIR=../data
PERMANENT_DIR=../permanent
MODEL_SAVE_DIR=backend/models
LOG_DIR=backend/logs

# API Configuration
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# API Keys (Required for real-time data)
FINNHUB_API_KEY=your_finnhub_api_key_here
UPSTOX_CLIENT_ID=your_upstox_client_id_here
UPSTOX_CLIENT_SECRET=your_upstox_secret_here
UPSTOX_REDIRECT_URI=http://localhost:5173

# Optional: Cache settings
CACHE_DIR=backend/_cache
CACHE_EXPIRY_HOURS=24
```

### API Keys Setup

**Finnhub** (https://finnhub.io): Free account → Get API key (60 calls/min)  
**Upstox** (https://upstox.com/developer): Developer account → Create app → Get Client ID & Secret → Run OAuth (expires daily, auto-refresh enabled)

```bash
python backend/scripts/generate_new_token.py       # Generate Upstox token
python backend/scripts/test_upstox_realtime.py     # Test Upstox data
```

### Model Configuration

Models configured in `backend/algorithms/optimised/<model_name>/model.py`. Example: Random Forest (n_estimators=200, max_depth=30), Decision Tree (max_depth=20), ANN (128→64→32→1 with Dropout 0.3).

## 📋 Data Requirements

### ISIN Codes

**Indian Stocks**: Require ISINs (12-character code, format: INExxxxxxxx, e.g., INE009A01021) for Upstox API
- ✅ **100% coverage**: All 500 stocks in `permanent/ind_stocks/index_ind_stocks.csv`
- ✅ **Live verified**: Random sample of 25 stocks tested with live Upstox API (100% success)
- ✅ **Format validated**: All ISINs are 12 characters starting with "INE"
- Used for: Real-time price fetching via Upstox API
- Verification: Run `python backend/scripts/verify_indian_isins.py --count 25`

**US Stocks**: Do NOT have ISINs
- ❌ No ISIN column in `permanent/us_stocks/index_us_stocks.csv`
- Use ticker symbols only (e.g., AAPL, MSFT, GOOGL)
- Identified by exchange (NYSE/NASDAQ)
- ISINs not required for Finnhub API

### Data Sources

**Real-time**: Finnhub (US) + Upstox (India) → Permanent storage fallback  
**Historical**: yfinance (2020-2025, 5yr) for both markets  
**Permanent Storage**: `permanent/` directory with 1000+ stocks (500 US + 500 Indian)

## 🧠 Model Architecture

9 algorithms: Linear Regression, Random Forest, Decision Tree, KNN, SVM, ANN, CNN, ARIMA, Autoencoder (all ready for training, 0/9 trained).

**50+ Technical Indicators** (OHLC only): Moving Averages (SMA 5,10,20,50,200, EMA 12,26), Momentum (RSI 14, MACD 12,26,9, Price momentum 1,5,10d), Volatility (Bollinger Bands 20,2σ, ATR 14, Rolling std 5,10,20d), Price Patterns (High/Low, Open/Close ratios), Lagged (1,2,3,5,10d prices), Rolling Stats (Min/Max/Std 5,10,20d), Time features (day/month/quarter).

**Training Pipeline**: Load 1000+ stocks → Generate 50+ indicators (`StockIndicators`) → Preprocess → Train (single-pass, 913 stocks, 1M+ rows) → Validate → Save to `backend/models/`

## 🧪 Testing

```bash
pytest                                  # All tests
pytest tests/unit/                      # Unit tests (algorithms, data, currency, indicators)
pytest tests/integration/               # Integration tests (API, training, data flow)
pytest --cov=algorithms --cov=main      # With coverage
python tests/manual/test_finnhub.py     # Manual interactive tests
```

## 📝 Logging

Logs in `backend/logs/`: `app.log`, `training.log`, `prediction.log`, `error.log`, `api.log`. Enable debug: Set `FLASK_DEBUG=True` and `LOG_LEVEL=DEBUG` in `.env`.

## ⚡ Performance

API: <100ms | Prediction: <1s (RF/DT) | Training: 5-180min (LR:5-10min, RF:10-15min, DT:5-10min, KNN:15-25min, SVM:20-30min, ANN:30-45min, CNN:45-75min, Autoencoder:40-60min, ARIMA:90-180min) | Memory: 2-8GB peak | Data load: 10-30s (1000+ stocks) | Model size: 1-50MB

**Tips**: Enable caching, permanent storage fallback, limit stocks for testing (`--max-stocks` flag)

## 🔧 Troubleshooting

**Import Errors**: `pip install -r requirements.txt` | Verify: `python -c "import flask, pandas, numpy, sklearn, tensorflow"`

**API Keys**: Check `.env` file | Test Finnhub: `curl "https://finnhub.io/api/v1/quote?symbol=AAPL&token=YOUR_KEY"` | Test Upstox: `python backend/scripts/test_upstox_realtime.py`

**Memory**: Close other applications, ensure 16GB RAM for optimal training

**Models**: Check `backend/models/` | Retrain: `python backend/training/train_full_dataset.py --model <name>`

**Debug**: Set `FLASK_DEBUG=True` and `LOG_LEVEL=DEBUG` in `.env` or environment

## 📖 Additional Resources

See [Main README](../README.md) for overview | [Documentation Hub](../documentation/README.md) for all guides | [Upstox Integration](../documentation/UPSTOX_INTEGRATION.md) for Indian market setup

## ⚠️ Important Notes

- **Model Status**: All 9 models ready for training - 0/9 currently trained
- **API Limits**: Finnhub 60 calls/min (free tier), Upstox tokens expire daily
- **Educational Use**: For learning and research only, verify data before financial decisions

---
**Version**: 0.1.0 | **Status**: Development (0/9 models trained, ready for training) | **Updated**: Oct 22, 2025