# API Usage Guide

This guide provides detailed examples of how to use the Stock Prediction ML API.

## Base URL

```
http://localhost:5000
```

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

## Response Format

All API responses follow this format:

```json
{
  "success": true,
  "data": { ... },
  "error": "Error message if any"
}
```

## Endpoints

### 1. Health Check

Check if the API is running.

**Request:**
```bash
curl "http://localhost:5000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Stock Prediction API",
  "version": "1.0.0",
  "timestamp": "2025-01-17T10:00:00Z"
}
```

### 2. Get Stock Price Prediction

Generate ML-based stock price predictions.

**Request:**
```bash
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1d&model=all"
```

**Parameters:**
- `symbol` (required): Stock symbol (e.g., AAPL, GOOGL, MSFT)
- `horizon` (optional): Prediction horizon (1d, 1w, 1m, 1y, 5y) - default: 1d
- `model` (optional): Model to use (all, lstm, rf, arima, svr, linear, knn) - default: all

**Response:**
```json
{
  "symbol": "AAPL",
  "horizon": "1d",
  "predicted_price": 178.45,
  "confidence": 72.1,
  "price_range": [172.10, 184.60],
  "time_frame_days": 1,
  "model_info": {
    "algorithm": "Ensemble (weighted)",
    "members": ["lstm", "random_forest", "arima", "svr", "linear_ridge", "knn"],
    "weights": {
      "lstm": 0.35,
      "random_forest": 0.25,
      "arima": 0.20,
      "svr": 0.10,
      "linear_ridge": 0.05,
      "knn": 0.05
    },
    "ensemble_size": 6
  },
  "data_points_used": 1250,
  "last_updated": "2025-01-17T14:21:22Z",
  "currency": "USD",
  "individual_predictions": {
    "lstm": {
      "prediction": 179.2,
      "model_info": { ... }
    },
    "random_forest": {
      "prediction": 177.8,
      "model_info": { ... }
    }
  }
}
```

**Examples:**

```bash
# 1-day prediction using all models
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1d"

# 1-week prediction using only LSTM
curl "http://localhost:5000/api/predict?symbol=GOOGL&horizon=1w&model=lstm"

# 1-month prediction using Random Forest
curl "http://localhost:5000/api/predict?symbol=MSFT&horizon=1m&model=rf"

# 1-year prediction using ARIMA
curl "http://localhost:5000/api/predict?symbol=TSLA&horizon=1y&model=arima"
```

### 3. Train Models

Train ML models for a specific stock symbol.

**Request:**
```bash
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "models": ["lstm", "random_forest", "arima"],
    "force": true,
    "max_data_points": 1000
  }'
```

**Request Body:**
- `symbol` (required): Stock symbol to train models for
- `models` (optional): List of models to train (default: all)
- `force` (optional): Retrain even if models exist (default: false)
- `max_data_points` (optional): Limit training data size (default: all available)

**Response:**
```json
{
  "symbol": "AAPL",
  "models_trained": ["lstm", "random_forest", "arima", "svr", "linear_ridge", "knn"],
  "training_metrics": {
    "lstm": {
      "prediction": 178.5,
      "model_info": {
        "model_name": "LSTM",
        "is_trained": true,
        "training_metrics": {
          "rmse": 2.45,
          "mae": 1.89
        }
      }
    },
    "random_forest": {
      "prediction": 179.1,
      "model_info": {
        "model_name": "RandomForest",
        "is_trained": true,
        "training_metrics": {
          "rmse": 2.12,
          "mae": 1.67
        }
      }
    }
  },
  "data_points_used": 1000,
  "last_updated": "2025-01-17T14:21:22Z"
}
```

**Examples:**

```bash
# Train all models for AAPL
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# Train specific models
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "GOOGL", "models": ["lstm", "random_forest"]}'

# Force retrain with data limit
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "MSFT", "force": true, "max_data_points": 500}'
```

### 4. List Trained Models

Get information about trained models for a symbol.

**Request:**
```bash
curl "http://localhost:5000/api/models/AAPL"
```

**Response:**
```json
{
  "symbol": "AAPL",
  "models": [
    {
      "model_name": "lstm_model",
      "saved_at": "2025-01-17T14:21:22Z",
      "training_metrics": {
        "rmse": 2.45,
        "mae": 1.89
      },
      "model_params": {
        "lookback": 60,
        "lstm_units": [128, 64, 32],
        "dropout_rate": 0.2
      }
    },
    {
      "model_name": "random_forest_model",
      "saved_at": "2025-01-17T14:21:22Z",
      "training_metrics": {
        "rmse": 2.12,
        "mae": 1.67
      },
      "model_params": {
        "n_estimators": 500,
        "max_depth": 20
      }
    }
  ],
  "count": 2
}
```

### 5. Get Live Stock Price

Get current stock price information.

**Request:**
```bash
curl "http://localhost:5000/live_price?symbol=AAPL"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "price": 178.45,
    "timestamp": "2025-01-17T14:21:22Z",
    "source": "yfinance",
    "company_name": "Apple Inc.",
    "currency": "USD",
    "exchange_rate": 1.0,
    "price_inr": 14850.25
  }
}
```

### 6. Get Historical Data

Get historical stock data for charting.

**Request:**
```bash
curl "http://localhost:5000/historical?symbol=AAPL&period=year"
```

**Parameters:**
- `symbol` (required): Stock symbol
- `period` (required): Time period (week, month, year, 5year)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "date": "2024-01-01",
      "open": 185.0,
      "high": 187.5,
      "low": 184.2,
      "close": 186.8,
      "volume": 45000000,
      "currency": "USD"
    }
  ]
}
```

### 7. Search Stocks

Search for stocks by symbol or company name.

**Request:**
```bash
curl "http://localhost:5000/search?q=apple"
```

**Response:**
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

## Error Handling

All endpoints return appropriate HTTP status codes and error messages:

### 400 Bad Request
```json
{
  "error": "Symbol parameter is required"
}
```

### 404 Not Found
```json
{
  "error": "Symbol not found",
  "message": "Symbol AAPL not found in us_stocks index"
}
```

### 500 Internal Server Error
```json
{
  "error": "Prediction failed",
  "message": "Model training failed: Insufficient data"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting to prevent abuse.

## Data Requirements

The API expects stock data in CSV format with these columns:
- `date`: Date in YYYY-MM-DD format
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume
- `adjusted_close`: Adjusted closing price
- `currency`: Currency code

## Model Performance

Typical performance metrics:
- **Training Time**: 2-5 minutes per model
- **Prediction Time**: < 1 second
- **Memory Usage**: 1-2 GB for LSTM models
- **Accuracy**: Varies by model and market conditions

## Best Practices

1. **Use ensemble predictions** for better accuracy
2. **Train models regularly** with fresh data
3. **Monitor prediction confidence** scores
4. **Handle errors gracefully** in your application
5. **Cache predictions** when appropriate
6. **Use appropriate horizons** for your use case

## Integration Examples

### Python
```python
import requests

# Get prediction
response = requests.get(
    "http://localhost:5000/api/predict",
    params={"symbol": "AAPL", "horizon": "1d"}
)
prediction = response.json()

# Train models
response = requests.post(
    "http://localhost:5000/api/train",
    json={"symbol": "AAPL", "models": ["lstm", "rf"]}
)
training_result = response.json()
```

### JavaScript
```javascript
// Get prediction
const response = await fetch(
  'http://localhost:5000/api/predict?symbol=AAPL&horizon=1d'
);
const prediction = await response.json();

// Train models
const trainingResponse = await fetch('http://localhost:5000/api/train', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ symbol: 'AAPL', models: ['lstm', 'rf'] })
});
const trainingResult = await trainingResponse.json();
```

### cURL Examples

```bash
# Quick prediction
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1d"

# Train and predict
curl -X POST "http://localhost:5000/api/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
curl "http://localhost:5000/api/predict?symbol=AAPL&horizon=1d"

# Check health
curl "http://localhost:5000/health"
```
