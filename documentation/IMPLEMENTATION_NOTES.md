# Implementation Notes

This document provides detailed technical information about the ML prediction system implementation.

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
backend/
├── algorithms/
│   ├── model_interface.py      # Abstract base class
│   ├── utils.py               # Data pipeline & ensemble
│   ├── real/                  # Production ML models
│   │   ├── lstm_wrapper.py
│   │   ├── random_forest.py
│   │   ├── arima_wrapper.py
│   │   ├── svr.py
│   │   ├── linear_models.py
│   │   └── knn.py
│   └── scratch/               # From-scratch implementations
│       ├── knn_scratch.py
│       ├── linear_regression_scratch.py
│       ├── logistic_regression_scratch.py
│       └── naive_bayes_scratch.py
├── main.py                    # Flask API endpoints
└── tests/                     # Test suite
```

## Model Interface Design

All models implement the `ModelInterface` abstract base class:

```python
class ModelInterface(ABC):
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface'
    def predict(self, X: np.ndarray) -> np.ndarray
    def save(self, path: str) -> None
    def load(self, path: str) -> 'ModelInterface'
```

This ensures:
- Consistent API across all models
- Easy model swapping and comparison
- Standardized persistence
- Type safety and documentation

## Data Pipeline

### Feature Engineering

The system automatically generates 20+ technical indicators:

**Moving Averages:**
- SMA(20, 50, 200): Simple moving averages
- EMA(12, 26): Exponential moving averages

**Momentum Indicators:**
- MACD(12, 26, 9): Moving Average Convergence Divergence
- RSI(14): Relative Strength Index

**Volatility Indicators:**
- Bollinger Bands(20, 2): Price volatility bands
- Price volatility: 20-day rolling standard deviation

**Volume Indicators:**
- OBV: On-Balance Volume
- Volume moving averages
- Volume volatility

**Lag Features:**
- Price lags: 1, 2, 3, 5, 10, 30 days
- Volume lags: 1, 2, 3, 5, 10, 30 days

### Data Preprocessing

1. **Data Loading**: Combines historical and latest data
2. **Feature Engineering**: Generates technical indicators
3. **Time Series Split**: 80% train, 20% test (chronological)
4. **Scaling**: MinMaxScaler for LSTM, StandardScaler for others
5. **Sequence Creation**: For time series models (LSTM)

## Model Implementations

### LSTM Wrapper

**Architecture:**
```python
Sequential([
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
```

**Key Features:**
- 3-layer LSTM with dropout regularization
- Early stopping (patience=20)
- MinMaxScaler normalization
- Multi-step prediction support
- Model persistence (.h5 format)

**Hyperparameters:**
- Lookback window: 60 days (configurable)
- LSTM units: (128, 64, 32)
- Dropout rate: 0.2
- Learning rate: 0.001
- Optimizer: Adam

### Random Forest Wrapper

**Configuration:**
```python
RandomForestRegressor(
    n_estimators=[200, 500],      # Grid search
    max_depth=[10, 20, 30],       # Grid search
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

**Key Features:**
- GridSearchCV with time series cross-validation
- Feature importance analysis
- Uncertainty estimation via individual trees
- Model persistence (joblib format)

**Hyperparameter Tuning:**
- Cross-validation: TimeSeriesSplit(3)
- Scoring: neg_mean_squared_error
- Parallel processing: n_jobs=-1

### ARIMA Wrapper

**Implementation:**
- Primary: `pmdarima.auto_arima` for automatic parameter selection
- Fallback: `statsmodels.tsa.arima.model.ARIMA` if pmdarima unavailable
- Stationarity testing: Augmented Dickey-Fuller test
- Automatic differencing for non-stationary data

**Key Features:**
- Automatic (p,d,q) parameter selection
- Prediction intervals (95% confidence)
- Seasonal ARIMA support
- Model persistence (joblib format)

**Parameters:**
- max_p, max_d, max_q: (5, 2, 5)
- Seasonal: False (configurable)
- Stepwise selection: True

### SVR Wrapper

**Configuration:**
```python
SVR(
    kernel='rbf',
    C=[0.1, 1, 10, 100],           # Grid search
    gamma=['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Grid search
    epsilon=[0.01, 0.1, 0.2, 0.5]  # Grid search
)
```

**Key Features:**
- RBF kernel with hyperparameter tuning
- GridSearchCV optimization
- StandardScaler normalization
- Model persistence (joblib format)

### Linear Models Wrapper

**Supported Models:**
- Ridge Regression
- Lasso Regression
- ElasticNet
- Linear Regression

**Configuration:**
```python
# Ridge
Ridge(alpha=[0.01, 0.1, 1.0, 10.0, 100.0])

# Lasso
Lasso(alpha=[0.01, 0.1, 1.0, 10.0, 100.0])

# ElasticNet
ElasticNet(alpha=[0.01, 0.1, 1.0, 10.0, 100.0], 
           l1_ratio=[0.1, 0.5, 0.7, 0.9])
```

### KNN Wrapper

**Configuration:**
```python
KNeighborsRegressor(
    n_neighbors=[3, 5, 7, 9, 11, 15, 20],  # Grid search
    weights=['uniform', 'distance'],        # Grid search
    p=[1, 2],                              # Grid search
    algorithm='auto',
    leaf_size=30
)
```

**Key Features:**
- Distance-based prediction
- Weighted voting (uniform/distance)
- Manhattan/Euclidean distance metrics
- Uncertainty estimation via neighbor distances

## Ensemble System

### Weight Calculation

Models are weighted based on inverse RMSE:

```python
def calculate_weights_from_rmse(self, validation_metrics):
    rmse_values = {name: metrics['rmse'] for name, metrics in validation_metrics.items()}
    inverse_rmse = {name: 1.0 / rmse for name, rmse in rmse_values.items()}
    total_inverse_rmse = sum(inverse_rmse.values())
    weights = {name: inv_rmse / total_inverse_rmse for name, inv_rmse in inverse_rmse.items()}
    return weights
```

### Confidence Calculation

Confidence is calculated from prediction variance:

```python
def _calculate_confidence(self, predictions, individual_results):
    mean_pred = sum(predictions.values()) / len(predictions)
    weighted_variance = sum(
        self.weights.get(name, 0) * (pred - mean_pred) ** 2
        for name, pred in predictions.items()
    )
    relative_std = (weighted_variance ** 0.5) / abs(mean_pred)
    confidence = max(0, min(100, 100 * (1 - relative_std)))
    return confidence
```

### Multi-horizon Forecasting

**Single-step models** (Random Forest, SVR, Linear, KNN):
- Use iterative forecasting
- Predict one step ahead, then use prediction as input for next step

**Multi-step models** (LSTM, ARIMA):
- LSTM: `predict_sequence()` method for multi-step prediction
- ARIMA: Native multi-step forecasting with `forecast(steps=N)`

## API Design

### Endpoint Structure

```
/api/predict     - GET  - Generate predictions
/api/train       - POST - Train models
/api/models/<id> - GET  - List trained models
```

### Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error message",
  "message": "Detailed error description"
}
```

**HTTP Status Codes:**
- 200: Success
- 400: Bad Request (missing/invalid parameters)
- 404: Not Found (symbol not found)
- 500: Internal Server Error (model training/prediction failed)

### Response Format

**Prediction Response:**
```json
{
  "symbol": "AAPL",
  "horizon": "1d",
  "predicted_price": 178.45,
  "confidence": 72.1,
  "price_range": [172.10, 184.60],
  "time_frame_days": 1,
  "model_info": { ... },
  "data_points_used": 1250,
  "last_updated": "2025-01-17T14:21:22Z",
  "currency": "USD",
  "individual_predictions": { ... }
}
```

## Model Persistence

### Save Format

**LSTM Models:**
- Model: `.h5` (Keras format)
- Scalers: `joblib` format
- Metadata: `JSON` format

**Other Models:**
- Model: `joblib` format
- Scaler: `joblib` format
- Metadata: `JSON` format

### Directory Structure

```
backend/models/
└── <symbol>/
    ├── lstm_model/
    │   ├── lstm_model.h5
    │   ├── target_scaler.joblib
    │   ├── feature_scaler.joblib
    │   └── metadata.json
    ├── random_forest_model/
    │   ├── random_forest_model.joblib
    │   ├── scaler.joblib
    │   └── metadata.json
    └── ...
```

## Testing Strategy

### Test Coverage

**Unit Tests:**
- Model interface compliance
- Feature engineering functions
- Ensemble prediction logic
- Individual model wrappers

**Integration Tests:**
- API endpoint functionality
- End-to-end prediction pipeline
- Model persistence (save/load)
- Error handling

**Test Data:**
- Demo CSV with 70 days of synthetic data
- Mock responses for external API calls
- Edge cases (insufficient data, invalid parameters)

### Test Structure

```
tests/
├── test_api.py          # API endpoint tests
├── test_models.py       # Model functionality tests
└── conftest.py         # Test fixtures and configuration
```

## Performance Considerations

### Memory Usage

**LSTM Models:**
- Training: ~2-4 GB RAM
- Inference: ~500 MB RAM
- Model size: ~50-100 MB

**Other Models:**
- Training: ~500 MB - 1 GB RAM
- Inference: ~100-200 MB RAM
- Model size: ~10-50 MB

### Training Time

**Per Model (1000 data points):**
- LSTM: 2-5 minutes
- Random Forest: 30-60 seconds
- ARIMA: 10-30 seconds
- SVR: 1-3 minutes
- Linear: 5-15 seconds
- KNN: 10-30 seconds

**Total Ensemble Training:** 5-10 minutes

### Prediction Time

**Single Model:** < 100ms
**Ensemble (6 models):** < 500ms
**API Response Time:** < 1 second

## Scalability

### Horizontal Scaling

- Stateless API design
- Model persistence enables load balancing
- Database integration possible for model metadata

### Vertical Scaling

- GPU support for LSTM training (optional)
- Parallel model training
- Caching for frequently requested predictions

## Security Considerations

### Input Validation

- Symbol format validation
- Horizon parameter validation
- Model parameter validation
- Data size limits

### Error Handling

- No sensitive information in error messages
- Graceful degradation on model failures
- Rate limiting (recommended for production)

## Monitoring and Logging

### Logging Levels

- DEBUG: Detailed model training information
- INFO: API requests and responses
- WARNING: Non-critical errors (model failures)
- ERROR: Critical errors (API failures)

### Metrics

- Prediction accuracy (RMSE, MAE)
- API response times
- Model training times
- Error rates by endpoint

## Future Enhancements

### Model Improvements

1. **Additional Models:**
   - XGBoost
   - LightGBM
   - Prophet
   - Transformer models

2. **Feature Engineering:**
   - Sentiment analysis features
   - Economic indicators
   - Market volatility indices

3. **Ensemble Methods:**
   - Stacking
   - Blending
   - Dynamic weighting

### API Enhancements

1. **Authentication:**
   - API key authentication
   - Rate limiting
   - Usage tracking

2. **Caching:**
   - Redis integration
   - Prediction caching
   - Model result caching

3. **Real-time Updates:**
   - WebSocket support
   - Live prediction updates
   - Model retraining triggers

### Infrastructure

1. **Containerization:**
   - Docker containers
   - Kubernetes deployment
   - Auto-scaling

2. **Data Pipeline:**
   - Real-time data ingestion
   - Automated model retraining
   - A/B testing framework

3. **Monitoring:**
   - Prometheus metrics
   - Grafana dashboards
   - Alerting system
