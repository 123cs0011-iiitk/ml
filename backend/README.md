# Stock Prediction ML Backend

A comprehensive machine learning backend for stock price prediction using multiple algorithms and ensemble methods.

## Features

- **Multiple ML Algorithms**: LSTM, Random Forest, ARIMA, SVR, Linear Regression, KNN
- **Ensemble Prediction**: Weighted ensemble based on validation performance
- **Feature Engineering**: Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, OBV)
- **Multi-horizon Forecasting**: 1 day, 1 week, 1 month, 1 year, 5 years
- **RESTful API**: Flask-based API with comprehensive endpoints
- **Model Persistence**: Save and load trained models
- **Comprehensive Testing**: Unit and integration tests

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ml/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Quick Start

1. **Start the Flask server**:
   ```bash
   python main.py
   ```

2. **Test the API**:
   ```bash
   curl "http://localhost:5000/health"
   ```

3. **Get a prediction**:
   ```bash
   curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1d"
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

# Train with data limit
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "max_data_points": 1000}'
```

### List Models

```bash
curl "http://localhost:5000/api/models/AAPL"
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

The system expects stock data in CSV format with the following columns:

- `date`: Date in YYYY-MM-DD format
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `adjusted_close`: Adjusted closing price
- `currency`: Currency code (USD, INR, etc.)

## Model Architecture

### LSTM Model
- 3-layer LSTM architecture (128, 64, 32 units)
- Dropout regularization (0.2)
- Early stopping with patience=20
- MinMaxScaler for normalization

### Random Forest
- GridSearchCV for hyperparameter tuning
- Parameters: n_estimators, max_depth
- Time series cross-validation
- Feature importance analysis

### ARIMA
- Automatic parameter selection using pmdarima
- Seasonal and non-seasonal variants
- Prediction intervals (95% confidence)
- Fallback to statsmodels if pmdarima unavailable

### Ensemble System
- Weighted averaging based on inverse RMSE
- Confidence calculation from prediction variance
- Multi-horizon forecasting support
- Individual model predictions available

## Feature Engineering

The system automatically generates technical indicators:

- **Moving Averages**: SMA(20,50,200), EMA(12,26)
- **MACD**: 12,26,9 with signal and histogram
- **RSI**: 14-period Relative Strength Index
- **Bollinger Bands**: 20-period, 2 standard deviations
- **OBV**: On-Balance Volume
- **Volume Indicators**: Volume moving averages
- **Lag Features**: 1,2,3,5,10,30 day lags

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
- Check the documentation
- Review the test cases for examples