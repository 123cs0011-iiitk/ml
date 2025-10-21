# Backend API

Machine learning backend for stock price prediction with 9 algorithms, automated training system, and comprehensive RESTful API.

## Features

- **9 ML Algorithms**: Linear Regression, Random Forest, Decision Tree, KNN, SVM, ANN, CNN, ARIMA, Autoencoders
- **Training System**: Automated model training with status tracking and validation
- **OHLC Data**: Uses Open, High, Low, Close data with technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ATR)
- **Multi-horizon Forecasting**: 1 day, 1 week, 1 month, 1 year, 5 years
- **RESTful API**: Flask-based API with comprehensive endpoints
- **Status Tracking**: Simple status checking with `python status.py`

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

The backend now includes a unified training system that trains models on combined data from all stocks:

### Training Commands

```bash
# Check model status
python status.py                    # Table format
python status.py --json             # JSON format
python status.py --simple           # Simple format

# Train all models
python backend/training/train_full_dataset.py

# Train specific model
python backend/training/train_full_dataset.py --model linear_regression

# Generate predictions using trained models
python backend/training/generate_predictions.py --max-stocks 10
```

### Model Training Process

1. **Data Loading**: Combines historical and latest data from all 1,000+ stocks
2. **Feature Engineering**: Creates technical indicators using `StockIndicators`
3. **Unified Training**: Trains each model on the combined dataset
4. **Validation**: Tests models on predefined validation stocks
5. **Status Tracking**: Maintains training progress in `backend/models/model_status.json`
6. **Model Storage**: Saves trained models to `backend/models/`

### Current Model Status

Check with `python status.py`:

- **✅ Working**: Random Forest (R²=0.994), Decision Tree (R²=0.85)
- **⚠️ Poor Performance**: SVM (R²=-26.2), KNN (R²=-27.9), ANN (R²=-9.7M)
- **❌ Failed/Stuck**: Linear Regression, CNN, ARIMA, Autoencoder

⚠️ **Note**: Only 2 out of 9 models are currently producing reliable predictions. ML predictions are unreliable.

### Prediction Generation

The system generates predictions for all stocks using pre-trained models:

```bash
# Generate predictions for all stocks
python backend/training/generate_predictions.py

# Generate predictions for specific category
python backend/training/generate_predictions.py --category us_stocks

# Test with limited stocks
python backend/training/generate_predictions.py --max-stocks 10 --test
```

## API Endpoints

### Prediction Endpoints

- `GET /api/predict` - Get stock price prediction
- `POST /api/train` - Train models for a symbol
- `GET /api/models/<symbol>` - List trained models for a symbol

### Data Endpoints

- `GET /live_price` - Get current stock price
- `GET /latest_prices` - Get latest prices for all stocks
- `GET /historical` - Get historical data for charting
- `GET /search` - Search for stocks
- `GET /symbols` - Get available stock symbols

### Utility Endpoints

- `GET /health` - Health check
- `GET /stock_info` - Get stock metadata

## Usage Examples

### Get Prediction
```bash
# Basic prediction
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1d"

# Specific model
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1w&model=lstm"

# All models (ensemble)
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1m&model=all"
```

### Train Models
```bash
# Train all models
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# Train specific models
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "models": ["lstm", "random_forest"]}'
```

### Data Endpoints
```bash
# Get live price
curl "http://localhost:5000/live_price?symbol=AAPL"

# Get historical data
curl "http://localhost:5000/historical?symbol=AAPL&period=year"

# Search stocks
curl "http://localhost:5000/search?q=apple"
```

### Response Format
```json
{
  "success": true,
  "data": { ... },
  "error": "Error message if any"
}
```

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000

# Data Configuration
DATA_DIR=data

# Model Configuration
MODEL_SAVE_DIR=backend/models
LOG_DIR=backend/logs

# API Configuration
CORS_ORIGINS=http://localhost:3000

# API Keys (Required)
FINNHUB_API_KEY=your_finnhub_key_here
UPSTOX_CLIENT_ID=your_upstox_client_id_here
UPSTOX_CLIENT_SECRET=your_upstox_secret_here
UPSTOX_REDIRECT_URI=http://localhost:3000
```

### Model Configuration

Models can be configured through their constructors:

```python
# LSTM Configuration
lstm_model = LSTMWrapper(
    lookback=60,
    lstm_units=(128, 64, 32),
    dropout_rate=0.2,
    learning_rate=0.001,
    patience=20
)

# Random Forest Configuration
rf_model = RandomForestWrapper(
    n_estimators=[200, 500],
    max_depth=[10, 20, 30],
    random_state=42
)
```

## Data Requirements

The system expects stock data in CSV format with the following important considerations:

### ISIN Requirements

**Indian Stocks (MANDATORY):**
- **ISINs are required** for Upstox API to work correctly
- All 500 Indian stocks have ISINs populated (100% coverage)
- ISINs are stored in:
  - `permanent/ind_stocks/index_ind_stocks.csv` (permanent index)
  - `data/index_ind_stocks_dynamic.csv` (dynamic index)
- ISIN format: 12-character alphanumeric code (e.g., `INE009A01021` for Infosys)
- Without correct ISINs, Upstox will return "wrong ISIN number" errors

**US Stocks (NOT Required):**
- Finnhub API uses ticker symbols, not ISINs
- ISINs are optional for US stocks
- System works perfectly without ISINs for US stocks

### Data Sources

**Real-time Fetching:**
- **US Stocks**: Finnhub API → Permanent directory (fallback)
- **Indian Stocks**: Upstox API → Permanent directory (fallback)
- **Note**: yfinance has been removed from real-time fetching (only used for historical data)

**Historical Data:**
- Both US and Indian stocks use yfinance for historical data fetching
- Period: 2020-01-01 to 2024-12-31 (5 years)

## Model Architecture

### Available Algorithms (9 total)
1. **Linear Regression** - Standard linear regression with feature scaling
2. **Random Forest** - Ensemble of decision trees with hyperparameter tuning
3. **Decision Tree** - Single decision tree with interpretable rules
4. **K-Nearest Neighbors** - Instance-based learning with distance-based prediction
5. **Support Vector Regression** - SVM for regression with multiple kernels
6. **Artificial Neural Network** - Multi-layer perceptron with dropout
7. **1D Convolutional Neural Network** - Time series CNN with sequence modeling
8. **ARIMA** - AutoRegressive Integrated Moving Average for time series
9. **Autoencoders** - Feature extraction + prediction

### Technical Indicators (Volume Excluded)
- **Moving Averages**: SMA (5,10,20,50,200), EMA (12,26)
- **Momentum**: RSI (14), MACD (12,26,9), Price momentum (1,5,10 day)
- **Volatility**: Bollinger Bands, ATR, Rolling standard deviation
- **Price Patterns**: High/Low ratios, Open/Close ratios, Price position
- **Lagged Features**: 1,2,3,5,10 day price lags
- **Rolling Statistics**: Min/Max/Std over 5,10,20 day windows

### Ensemble System
- Weighted averaging based on model performance
- Confidence calculation from prediction variance
- Multi-horizon forecasting support
- Individual model predictions available

## Feature Engineering

The system automatically generates technical indicators from OHLC data only (volume excluded):

- **Moving Averages**: SMA(5,10,20,50,200), EMA(12,26)
- **Momentum Indicators**: RSI(14), MACD(12,26,9), Price momentum
- **Volatility**: Bollinger Bands(20,2σ), ATR(14), Rolling volatility
- **Price Patterns**: High/Low ratios, Open/Close ratios, Price position
- **Lagged Features**: 1,2,3,5,10 day price lags
- **Rolling Statistics**: Min/Max/Std over 5,10,20 day windows
- **Time-based Features**: Day of week, month, quarter

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_api.py
pytest tests/test_models.py

# Run with coverage
pytest --cov=algorithms --cov=main
```

## Logging

Logs are written to `backend/logs/` directory:

- Application logs
- Model training logs
- Prediction logs
- Error logs

## Performance Considerations

- **Memory Usage**: LSTM models require significant memory
- **Training Time**: Initial training can take several minutes
- **Prediction Speed**: Ensemble predictions are fast (< 1 second)
- **Data Storage**: Models are saved to disk for persistence

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or model complexity
3. **Data Issues**: Check CSV format and data quality
4. **Model Loading**: Ensure model files are not corrupted

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the [documentation](../documentation/README.md)
- Review the test cases for examples
- See [API Usage Guide](../documentation/API_USAGE.md) for detailed examples